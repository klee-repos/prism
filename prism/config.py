"""Prism configuration: a friendly, provider-agnostic YAML that generates a LiteLLM
proxy config.

A stranger swaps providers by editing ``~/.prism/config.yaml`` — never any code. The
one field that decides the backend is ``providers.<name>.type``, a litellm provider
*slug* (``openrouter``, ``zai``, ``openai``, ``gemini``, ``anthropic``, …), validated
against ``litellm.provider_list``. Model ids are provider-native (OpenRouter carries a
vendor segment, e.g. ``z-ai/glm-4.6``).
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

MASTER_KEY_ENV = "PRISM_MASTER_KEY"
HOOK_REF = "prism_hook.instance"


def prism_home() -> Path:
    return Path(os.environ.get("PRISM_HOME", str(Path.home() / ".prism")))


def config_path() -> Path:
    return prism_home() / "config.yaml"


def master_key_path() -> Path:
    return prism_home() / "master_key"


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

    for slot, target in mapping.items():
        if slot not in MAPPING_SLOTS:
            raise ConfigError(f"mapping has unknown slot '{slot}' (allowed: {', '.join(MAPPING_SLOTS)}).")
        if target not in routes:
            raise ConfigError(f"mapping.{slot} → '{target}' is not a defined route.")


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
    return params


def to_litellm_config(cfg: dict) -> dict:
    """Generate the LiteLLM proxy config dict from the friendly Prism config."""
    model_list = []
    for role in REQUIRED_ROLES:
        model_list.append({"model_name": role, "litellm_params": _route_params(cfg, role)})
    # Catch-all so any model name Claude Code sends (e.g. `--model opus`) resolves to coder.
    coder_params = dict(_route_params(cfg, "coder"))
    model_list.append({"model_name": "*", "litellm_params": coder_params})
    return {
        "model_list": model_list,
        "litellm_settings": {"callbacks": HOOK_REF},
        "general_settings": {"master_key": f"os.environ/{MASTER_KEY_ENV}"},
    }


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
    return envs
