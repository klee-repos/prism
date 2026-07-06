"""Prism configuration: a friendly, provider-agnostic YAML that generates a LiteLLM
proxy config.

A stranger swaps providers by editing ``~/.prism/config.yaml`` — never any code. The
one field that decides the backend is ``providers.<name>.type``, a litellm provider
*slug* (``openrouter``, ``zai``, ``openai``, ``gemini``, ``anthropic``, …), validated
against ``litellm.provider_list``. Model ids are provider-native (OpenRouter carries a
vendor segment, e.g. ``z-ai/glm-5.2``).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
REQUIRED_ROLES = ("coder", "background", "multimodal")
MAPPING_SLOTS = ("opus", "sonnet", "haiku")

HOOK_REF = "prism_hook.instance"
WEBSEARCH_CALLBACK = "websearch_interception"
WEBSEARCH_SEARCH_TOOL_NAME = "prism-search"

# litellm LlmProviders slugs Prism can route to. The websearch_interception callback
# gate skips any provider not in enabled_providers (None/[] both collapse to
# bedrock-only inside its __init__, which would skip our backends), so list them
# explicitly. Keep ZAI for direct z.ai routing; add others here as Prism supports them.
WEBSEARCH_ENABLED_PROVIDERS = (
    "openrouter", "openai", "gemini", "anthropic", "zai", "azure", "bedrock", "vertex_ai",
)

# litellm SearchProviders slugs that take an API key. Used to validate `search.provider`.
# (DuckDuckGo is keyless in litellm but returns ~0 results for real queries, so we
# still accept it but warn; see cmd_doctor.)
KEYED_SEARCH_PROVIDERS = {
    "tavily", "perplexity", "serper", "exa_ai", "brave",
    "firecrawl", "you_com", "searchapi", "linkup", "google_pse",
    "dataforseo", "apiserpent", "tinyfish", "parallel_ai",
}
KEYLESS_SEARCH_PROVIDERS = {"duckduckgo", "searxng", "fastcrw"}


def prism_home() -> Path:
    return Path(os.environ.get("PRISM_HOME", str(Path.home() / ".prism")))


def config_path() -> Path:
    return prism_home() / "config.yaml"


def hook_shim_path() -> Path:
    return prism_home() / "prism_hook.py"


def gen_config_path() -> Path:
    return prism_home() / ".litellm.gen.yaml"


def log_path() -> Path:
    return prism_home() / "proxy.log"


class ConfigError(Exception):
    """Raised on an invalid Prism config (surfaced to the user, no traceback)."""


def _default_config_text() -> str:
    return (Path(__file__).parent / "resources" / "default_config.yaml").read_text()


def default_config() -> dict:
    return yaml.safe_load(_default_config_text())


def load_config(path: Path | None = None) -> dict:
    p = path or config_path()
    if not p.exists():
        raise ConfigError(
            f"no config at {p} — run `prism setup` first (creates it with sane defaults)."
        )
    try:
        cfg = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{p} is not valid YAML: {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError(f"{p} must be a YAML mapping.")
    return cfg


def validate(cfg: dict, known_providers: set[str] | None = None) -> None:
    """Referential + slug validation. ``known_providers`` defaults to litellm's list."""
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version must be {SCHEMA_VERSION} (got {cfg.get('schema_version')!r}); "
            "your config may be from a different Prism version."
        )
    providers = cfg.get("providers") or {}
    routes = cfg.get("routes") or {}
    mapping = cfg.get("mapping") or {}
    if not isinstance(providers, dict) or not providers:
        raise ConfigError("`providers` must be a non-empty mapping.")
    if not isinstance(routes, dict):
        raise ConfigError("`routes` must be a mapping.")

    if known_providers is None:
        known_providers = _litellm_provider_list()

    for name, prov in providers.items():
        if not isinstance(prov, dict) or "type" not in prov:
            raise ConfigError(f"provider '{name}' needs a `type` (a litellm provider slug).")
        slug = prov["type"]
        if known_providers and slug not in known_providers:
            sample = ", ".join(sorted(list(known_providers))[:12])
            raise ConfigError(
                f"provider '{name}' has unknown type '{slug}'. "
                f"Use a litellm provider slug (e.g. {sample}, …)."
            )
        if "api_key_env" not in prov:
            raise ConfigError(f"provider '{name}' needs `api_key_env` (the env var holding the key).")

    for role in REQUIRED_ROLES:
        if role not in routes:
            raise ConfigError(f"missing required route '{role}' (need: {', '.join(REQUIRED_ROLES)}).")
        r = routes[role]
        if not isinstance(r, dict) or "provider" not in r or "model" not in r:
            raise ConfigError(f"route '{role}' needs `provider` and `model`.")
        if r["provider"] not in providers:
            raise ConfigError(
                f"route '{role}' references provider '{r['provider']}' which is not defined."
            )
        if "extra_body" in r and not isinstance(r["extra_body"], dict):
            raise ConfigError(f"route '{role}' has a non-mapping `extra_body` (must be a YAML mapping).")

    for slot, target in mapping.items():
        if slot not in MAPPING_SLOTS:
            raise ConfigError(f"mapping has unknown slot '{slot}' (allowed: {', '.join(MAPPING_SLOTS)}).")
        if target not in routes:
            raise ConfigError(f"mapping.{slot} → '{target}' is not a defined route.")

    _validate_search(cfg)


def _validate_search(cfg: dict) -> None:
    """Validate the optional `search` section (enables Claude Code's WebSearch).

    If absent, Prism strips hosted web_search tools (the cheap backends can't run
    them). If present, Prism converts them to litellm's callable ``litellm_web_search``
    function tool and litellm's websearch_interception callback executes the search
    via ``litellm.asearch``.

    Shape::

        search:
          provider: tavily            # a litellm SearchProviders slug
          api_key_env: TAVILY_API_KEY # env var holding the key (omit for keyless)
          # api_base: https://...     # optional override
    """
    search = cfg.get("search")
    if search is None:
        return
    if not isinstance(search, dict):
        raise ConfigError("`search` must be a mapping (or omitted to disable web search).")
    provider = search.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ConfigError("`search.provider` is required (a litellm search provider slug).")
    known = KEYED_SEARCH_PROVIDERS | KEYLESS_SEARCH_PROVIDERS
    if provider not in known:
        sample = ", ".join(sorted(known)[:12])
        raise ConfigError(
            f"search.provider '{provider}' is not a known litellm search provider "
            f"(e.g. {sample}, …). See https://docs.litellm.ai/docs/search."
        )
    if provider in KEYED_SEARCH_PROVIDERS and "api_key_env" not in search:
        raise ConfigError(
            f"search.provider '{provider}' needs `api_key_env` (the env var holding its key). "
            "Keyless options: duckduckgo, searxng, fastcrw."
        )


def _litellm_provider_list() -> set[str]:
    try:
        import litellm

        return set(litellm.provider_list)  # type: ignore[attr-defined]
    except Exception:
        return set()  # can't import litellm here → skip slug check (validated at proxy start)


def _route_model_string(cfg: dict, role: str) -> str:
    route = cfg["routes"][role]
    prov = cfg["providers"][route["provider"]]
    return f"{prov['type']}/{route['model']}"


def _route_params(cfg: dict, role: str) -> dict:
    route = cfg["routes"][role]
    prov = cfg["providers"][route["provider"]]
    params: dict[str, Any] = {
        "model": _route_model_string(cfg, role),
        "api_key": f"os.environ/{prov['api_key_env']}",
    }
    if prov.get("api_base"):  # only override when a provider explicitly needs it
        params["api_base"] = prov["api_base"]
    if route.get("extra_body"):  # provider-native passthrough (e.g. OpenRouter provider routing)
        params["extra_body"] = route["extra_body"]
    return params


def _search_params(cfg: dict) -> dict | None:
    """litellm_params for the search tool, or None if web search is unconfigured."""
    search = cfg.get("search") or {}
    provider = search.get("provider")
    if not provider:
        return None
    params: dict[str, Any] = {"search_provider": provider}
    if search.get("api_key_env"):
        params["api_key"] = f"os.environ/{search['api_key_env']}"
    if search.get("api_base"):
        params["api_base"] = search["api_base"]
    return params


def to_litellm_config(cfg: dict) -> dict:
    """Generate the LiteLLM proxy config dict from the friendly Prism config."""
    model_list = []
    for role in REQUIRED_ROLES:
        model_list.append({"model_name": role, "litellm_params": _route_params(cfg, role)})
    # Catch-all so any model name Claude Code sends (e.g. `--model opus`) resolves to coder.
    coder_params = dict(_route_params(cfg, "coder"))
    model_list.append({"model_name": "*", "litellm_params": coder_params})
    # No `general_settings.master_key`: the proxy is loopback-only (127.0.0.1) and does
    # not gate requests on a key. Claude Code, when signed in to a Claude subscription,
    # forwards its own OAuth token regardless of env vars — a mandatory proxy key would
    # reject that token and break the whole point. Loopback binding is the trust boundary.
    litellm_settings: dict[str, Any] = {"callbacks": [HOOK_REF]}
    out: dict[str, Any] = {"model_list": model_list, "litellm_settings": litellm_settings}

    search_params = _search_params(cfg)
    if search_params is not None:
        # Register litellm's websearch_interception callback (instantiated by name) and
        # a search_tools entry the callback resolves via litellm.asearch(). The hook
        # (prism.routing) converts Claude Code's hosted web_search tool into the callable
        # litellm_web_search function tool; this callback runs the search when the model
        # calls it, then re-calls the model with the results — keeping you on the cheap
        # coder model instead of routing web-search turns to the multimodal route.
        litellm_settings["callbacks"] = [HOOK_REF, WEBSEARCH_CALLBACK]
        litellm_settings["websearch_interception_params"] = {
            # The gate skips any provider NOT in this list. None/[] both default to
            # bedrock-only (litellm's __init__ rewrites None → [bedrock]), which would
            # skip every Prism backend. List the litellm provider slugs Prism can route
            # to so GLM-via-OpenRouter, Gemini, etc. are all intercepted.
            "enabled_providers": list(WEBSEARCH_ENABLED_PROVIDERS),
            "search_tool_name": WEBSEARCH_SEARCH_TOOL_NAME,
        }
        out["search_tools"] = [{
            "search_tool_name": WEBSEARCH_SEARCH_TOOL_NAME,
            "litellm_params": search_params,
        }]
    return out


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def mapping_env(cfg: dict) -> dict[str, str]:
    """The ANTHROPIC_DEFAULT_*_MODEL values (route names) from the mapping."""
    mapping = cfg.get("mapping") or {}
    out = {}
    if "opus" in mapping:
        out["ANTHROPIC_DEFAULT_OPUS_MODEL"] = mapping["opus"]
    if "sonnet" in mapping:
        out["ANTHROPIC_DEFAULT_SONNET_MODEL"] = mapping["sonnet"]
    if "haiku" in mapping:
        out["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = mapping["haiku"]
    return out


def active_key_envs(cfg: dict) -> list[str]:
    """Env vars whose keys the active (mapped) routes need — for the pre-flight check."""
    mapping = cfg.get("mapping") or {}
    routes = cfg.get("routes") or {}
    providers = cfg.get("providers") or {}
    envs: list[str] = []
    active_roles = set(mapping.values()) | {"coder", "multimodal"}
    for role in active_roles:
        route = routes.get(role) or {}
        prov = providers.get(route.get("provider")) or {}
        env = prov.get("api_key_env")
        if env and env not in envs:
            envs.append(env)
    # Web search key (if configured) — the proxy won't be able to run searches without it.
    search = cfg.get("search") or {}
    search_env = search.get("api_key_env")
    if search_env and search_env not in envs:
        envs.append(search_env)
    return envs


def search_enabled(cfg: dict) -> bool:
    """True if a web-search provider is configured (so the hook converts, not strips)."""
    return bool((cfg.get("search") or {}).get("provider"))
