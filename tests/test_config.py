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
    assert out["litellm_settings"]["callbacks"] == "prism_hook.instance"
    # Loopback-only proxy: no master-key gate is emitted.
    assert "general_settings" not in out


def test_wildcard_route_targets_coder_model():
    out = c.to_litellm_config(base())
    star = next(m for m in out["model_list"] if m["model_name"] == "*")
    assert star["litellm_params"]["model"] == "openrouter/z-ai/glm-5.2"


def test_zai_direct_swap_uses_native_slug_no_api_base():
    cfg = base()
    cfg["providers"]["zai"] = {"type": "zai", "api_key_env": "ZAI_API_KEY"}
    cfg["routes"]["coder"] = {"provider": "zai", "model": "glm-5.2"}
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
    del cfg["routes"]["coder"]["extra_body"]
    coder = next(m for m in c.to_litellm_config(cfg)["model_list"] if m["model_name"] == "coder")
    assert "extra_body" not in coder["litellm_params"]


def test_validate_rejects_non_mapping_extra_body():
    cfg = base()
    cfg["routes"]["coder"]["extra_body"] = ["not", "a", "mapping"]
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_unknown_provider_type():
    cfg = base()
    cfg["providers"]["openrouter"]["type"] = "notaslug"
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_dangling_provider_ref():
    cfg = base()
    cfg["routes"]["coder"]["provider"] = "ghost"
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_bad_mapping_target():
    cfg = base()
    cfg["mapping"]["opus"] = "nope"
    with pytest.raises(c.ConfigError):
        c.validate(cfg, known_providers=KNOWN)


def test_validate_rejects_missing_role():
    cfg = base()
    del cfg["routes"]["multimodal"]
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


def test_config_hash_is_stable_and_sensitive():
    cfg = base()
    h1 = c.config_hash(cfg)
    assert h1 == c.config_hash(c.default_config())
    cfg["routes"]["coder"]["model"] = "z-ai/glm-5"
    assert c.config_hash(cfg) != h1
