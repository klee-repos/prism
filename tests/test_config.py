"""Unit tests for config validation + LiteLLM config generation."""
import pytest

from prism import config as c

KNOWN = {"openrouter", "zai", "openai", "gemini", "anthropic"}


def base():
    return c.default_config()


def test_default_config_is_valid():
    c.validate(base(), known_providers=KNOWN)


def test_to_litellm_config_shape():
    out = c.to_litellm_config(base())
    names = [m["model_name"] for m in out["model_list"]]
    assert names.count("coder") == 1
    assert "multimodal" in names and "background" in names and "*" in names
    coder = next(m for m in out["model_list"] if m["model_name"] == "coder")
    assert coder["litellm_params"]["model"] == "openrouter/z-ai/glm-5.2"
    assert coder["litellm_params"]["api_key"] == "os.environ/OPENROUTER_API_KEY"
    assert out["litellm_settings"]["callbacks"] == ["prism_hook.instance"]
    # No search section in the default config → no search_tools / interception callback.
    assert "search_tools" not in out
    assert "websearch_interception_params" not in out["litellm_settings"]
    # Loopback-only proxy: no master-key gate is emitted.
    assert "general_settings" not in out


def test_wildcard_route_targets_coder_model():
    out = c.to_litellm_config(base())
    star = next(m for m in out["model_list"] if m["model_name"] == "*")
    assert star["litellm_params"]["model"] == "openrouter/z-ai/glm-5.2"


def test_zai_direct_swap_uses_native_slug_no_api_base():
    cfg = base()
    cfg["providers"]["zai"] = {"type": "zai", "api_key_env": "ZAI_API_KEY"}
    c.resolved_routes(cfg)["coder"] = {"provider": "zai", "model": "glm-5.2"}
    coder = next(m for m in c.to_litellm_config(cfg)["model_list"] if m["model_name"] == "coder")
    assert coder["litellm_params"]["model"] == "zai/glm-5.2"
    assert "api_base" not in coder["litellm_params"]


def test_provider_api_base_override_is_emitted():
    cfg = base()
    cfg["providers"]["openrouter"]["api_base"] = "https://example/v1"
    coder = next(m for m in c.to_litellm_config(cfg)["model_list"] if m["model_name"] == "coder")
    assert coder["litellm_params"]["api_base"] == "https://example/v1"


def test_coder_route_emits_openrouter_provider_routing():
    coder = next(m for m in c.to_litellm_config(base())["model_list"] if m["model_name"] == "coder")
    provider = coder["litellm_params"]["extra_body"]["provider"]
    assert provider["require_parameters"] is True
    # FP8-or-better only: no int4/int8/fp4/fp6 and no undeclared `unknown`.
    assert provider["quantizations"] == ["fp8", "fp16", "bf16", "fp32"]
    assert "int8" not in provider["quantizations"] and "unknown" not in provider["quantizations"]


def test_background_route_has_fp8_floor_without_require_parameters():
    bg = next(m for m in c.to_litellm_config(base())["model_list"] if m["model_name"] == "background")
    provider = bg["litellm_params"]["extra_body"]["provider"]
    assert provider["quantizations"] == ["fp8", "fp16", "bf16", "fp32"]
    # Background stays maximally available: the FP8 quality floor, but no tool-calling pin.
    assert "require_parameters" not in provider


def test_wildcard_route_inherits_coder_extra_body():
    star = next(m for m in c.to_litellm_config(base())["model_list"] if m["model_name"] == "*")
    assert star["litellm_params"]["extra_body"]["provider"]["require_parameters"] is True


def test_route_without_extra_body_omits_the_key():
    cfg = base()
    del c.resolved_routes(cfg)["coder"]["extra_body"]
    coder = next(m for m in c.to_litellm_config(cfg)["model_list"] if m["model_name"] == "coder")
    assert "extra_body" not in coder["litellm_params"]


def test_validate_rejects_non_mapping_extra_body():
    cfg = base()
    c.resolved_routes(cfg)["coder"]["extra_body"] = ["not", "a", "mapping"]
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_unknown_provider_type():
    cfg = base()
    cfg["providers"]["openrouter"]["type"] = "notaslug"
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_dangling_provider_ref():
    cfg = base()
    c.resolved_routes(cfg)["coder"]["provider"] = "ghost"
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_bad_mapping_target():
    cfg = base()
    cfg["mapping"]["opus"] = "nope"
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_missing_role():
    cfg = base()
    del c.resolved_routes(cfg)["multimodal"]
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_wrong_schema_version():
    cfg = base()
    cfg["schema_version"] = 999
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_provider_without_key_env():
    cfg = base()
    del cfg["providers"]["openrouter"]["api_key_env"]
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_mapping_env():
    env = c.mapping_env(base())
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "coder"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "coder"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "background"


def test_active_key_envs():
    assert c.active_key_envs(base()) == ["OPENROUTER_API_KEY"]


def test_active_key_envs_includes_search_key():
    cfg = base()
    cfg["search"] = {"provider": "firecrawl", "api_key_env": "FIRECRAWL_API_KEY"}
    assert "FIRECRAWL_API_KEY" in c.active_key_envs(cfg)


def test_search_section_emits_callback_and_search_tools():
    cfg = base()
    cfg["search"] = {"provider": "firecrawl", "api_key_env": "FIRECRAWL_API_KEY"}
    out = c.to_litellm_config(cfg)
    cb = out["litellm_settings"]["callbacks"]
    assert "websearch_interception" in cb and "prism_hook.instance" in cb
    params = out["litellm_settings"]["websearch_interception_params"]
    # Must explicitly include Prism's providers — None/[] both collapse to bedrock-only
    # inside litellm's __init__, which would skip GLM-via-OpenRouter.
    assert "openrouter" in params["enabled_providers"]
    assert "gemini" in params["enabled_providers"]
    assert params["search_tool_name"] == "prism-search"
    assert out["search_tools"][0]["litellm_params"]["search_provider"] == "firecrawl"
    assert out["search_tools"][0]["litellm_params"]["api_key"] == "os.environ/FIRECRAWL_API_KEY"


def test_search_section_supports_firecrawl():
    # Firecrawl is the documented default. Verify it validates and emits the right slug.
    cfg = base()
    cfg["search"] = {"provider": "firecrawl", "api_key_env": "FIRECRAWL_API_KEY"}
    c.validate(cfg, known_providers=KNOWN)  # must not raise
    out = c.to_litellm_config(cfg)
    assert out["search_tools"][0]["litellm_params"]["search_provider"] == "firecrawl"



def test_search_section_keyless_provider_needs_no_key_env():
    cfg = base()
    cfg["search"] = {"provider": "duckduckgo"}
    c.validate(cfg, known_providers=KNOWN)  # no api_key_env required
    out = c.to_litellm_config(cfg)
    assert out["search_tools"][0]["litellm_params"]["search_provider"] == "duckduckgo"
    assert "api_key" not in out["search_tools"][0]["litellm_params"]


def test_search_section_rejects_unknown_provider():
    cfg = base()
    cfg["search"] = {"provider": "googly", "api_key_env": "G"}
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_search_section_keyed_provider_needs_api_key_env():
    cfg = base()
    cfg["search"] = {"provider": "tavily"}  # missing api_key_env
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_no_search_section_emits_no_search_tools():
    out = c.to_litellm_config(base())
    assert "search_tools" not in out
    assert out["litellm_settings"]["callbacks"] == ["prism_hook.instance"]
    assert "websearch_interception_params" not in out["litellm_settings"]


def test_config_hash_is_stable_and_sensitive():
    cfg = base()
    h1 = c.config_hash(cfg)
    assert h1 == c.config_hash(c.default_config())
    c.resolved_routes(cfg)["coder"]["model"] = "z-ai/glm-5"
    assert c.config_hash(cfg) != h1
    # An active_profile flip also changes the (display-only) hash.
    cfg2 = base()
    cfg2["active_profile"] = "k3"
    assert c.config_hash(cfg2) != h1


# ── profiles (named route bundles + one-word switch) ────────────────────────────────

def legacy_config():
    """A pre-profiles config: a single top-level `routes` block, no profiles. Must keep
    working unchanged (the user's existing ~/.prism/config.yaml has this shape)."""
    return {
        "schema_version": 1,
        "providers": {"openrouter": {"type": "openrouter", "api_key_env": "OPENROUTER_API_KEY"}},
        "routes": {
            "coder": {"provider": "openrouter", "model": "z-ai/glm-5.2"},
            "background": {"provider": "openrouter", "model": "z-ai/glm-4.7-flash"},
            "multimodal": {"provider": "openrouter", "model": "google/gemini-2.5-flash"},
        },
        "mapping": {"opus": "coder", "sonnet": "coder", "haiku": "background"},
    }


def test_default_config_is_profiles_shaped_active_glm():
    cfg = base()
    assert cfg["active_profile"] == "glm"
    assert set(cfg["profiles"]) == {"glm", "k3"}
    assert "routes" not in cfg  # profiles config carries no top-level routes


def test_resolved_routes_returns_active_profile_live_reference():
    cfg = base()
    assert c.resolved_routes(cfg) is cfg["profiles"]["glm"]  # live reference, not a copy
    c.resolved_routes(cfg)["coder"]["model"] = "z-ai/changed"
    coder = next(m for m in c.to_litellm_config(cfg)["model_list"] if m["model_name"] == "coder")
    assert coder["litellm_params"]["model"] == "openrouter/z-ai/changed"


def test_resolved_routes_legacy_fallback():
    cfg = legacy_config()
    assert c.resolved_routes(cfg) is cfg["routes"]


def test_resolved_routes_raises_configerror_not_keyerror_on_unknown_active():
    cfg = base()
    cfg["active_profile"] = "nope"
    # A raw KeyError here would crash the un-validated `prism status`; must be ConfigError.
    with pytest.raises(c.ConfigError):
        c.resolved_routes(cfg)


def test_k3_profile_kimi_for_text_and_vision_with_no_quant_pin():
    cfg = base()
    cfg["active_profile"] = "k3"
    c.validate(cfg, known_providers=KNOWN)
    ml = {m["model_name"]: m["litellm_params"] for m in c.to_litellm_config(cfg)["model_list"]}
    # K3 handles BOTH text and images — coder AND multimodal (and the * catch-all) point at it.
    assert ml["coder"]["model"] == "openrouter/moonshotai/kimi-k3"
    assert ml["multimodal"]["model"] == "openrouter/moonshotai/kimi-k3"
    assert ml["*"]["model"] == "openrouter/moonshotai/kimi-k3"
    # int4 safety (Gate B): K3's only OpenRouter endpoint is int4, so a fp8-or-better quant pin
    # would leave zero eligible providers. EVERY kimi-k3 route must emit NO extra_body.
    for name, p in ml.items():
        if p["model"] == "openrouter/moonshotai/kimi-k3":
            assert "extra_body" not in p, f"{name}: kimi-k3 route must not pin quantization (int4)"
    # background stays cheap on GLM-flash with its fp8 floor (K3 is $3/$15 — wasteful there).
    assert ml["background"]["model"] == "openrouter/z-ai/glm-4.7-flash"
    assert ml["background"]["extra_body"]["provider"]["quantizations"] == ["fp8", "fp16", "bf16", "fp32"]


def test_validate_rejects_both_routes_and_profiles():
    cfg = base()
    cfg["routes"] = {"coder": {"provider": "openrouter", "model": "z-ai/glm-5.2"}}
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_profiles_without_active_profile():
    cfg = base()
    del cfg["active_profile"]
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_active_profile_without_profiles():
    cfg = {
        "schema_version": 1,
        "providers": {"openrouter": {"type": "openrouter", "api_key_env": "OPENROUTER_API_KEY"}},
        "active_profile": "glm",
        "mapping": {"opus": "coder"},
    }
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_unknown_active_profile():
    cfg = base()
    cfg["active_profile"] = "ghost"
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_non_dict_profile_bundle_with_configerror():
    # Gate B #1: a hand-edited `k3:` line with nothing under it parses to None — must be a
    # friendly ConfigError, NOT a raw TypeError that escapes the CLI's error handler.
    for bad in (None, ["a"], "oops"):
        cfg = base()
        cfg["profiles"]["k3"] = bad
        with pytest.raises(c.ConfigError):
            c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_non_string_provider_or_model():
    # Gate B #2: a list `provider` would otherwise raise TypeError (unhashable) at the
    # membership check; a list/empty `model` would bake a broken "openrouter/['a','b']" string.
    for mutate in (
        lambda r: r["coder"].__setitem__("provider", ["openrouter"]),
        lambda r: r["coder"].__setitem__("model", ["a", "b"]),
        lambda r: r["coder"].__setitem__("model", ""),
    ):
        cfg = base()
        mutate(c.resolved_routes(cfg))
        with pytest.raises(c.ConfigError):
            c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_defect_in_inactive_profile():
    # Gate B #3: validate must check EVERY profile, not just the active one — otherwise an
    # "only-active" implementation would be fake-green while shipping a broken profile.
    cfg = base()
    cfg["active_profile"] = "glm"
    cfg["profiles"]["k3"]["coder"]["provider"] = "ghost"  # break the INACTIVE profile
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_legacy_config_validates_and_generates():
    cfg = legacy_config()
    c.validate(cfg, known_providers=KNOWN)  # must not raise
    out = c.to_litellm_config(cfg)
    coder = next(m for m in out["model_list"] if m["model_name"] == "coder")
    assert coder["litellm_params"]["model"] == "openrouter/z-ai/glm-5.2"
    assert c.active_key_envs(cfg) == ["OPENROUTER_API_KEY"]


@pytest.mark.parametrize("break_it", [
    lambda cfg: cfg["routes"].pop("multimodal"),
    lambda cfg: cfg["routes"]["coder"].__setitem__("provider", "ghost"),
    lambda cfg: cfg["routes"]["coder"].__setitem__("extra_body", ["nope"]),
])
def test_legacy_reject_path_preserved(break_it):
    # The refactor extracted a shared bundle validator; the legacy branch must still reject the
    # same defects (guards against the legacy path being silently gutted).
    cfg = legacy_config()
    break_it(cfg)
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)
