"""Unit tests for the modality router (the routing brain)."""
from prism.routing import ModalityRouter, needs_multimodal


def test_text_string_is_not_multimodal():
    assert needs_multimodal([{"role": "user", "content": "hello"}]) is False


def test_text_parts_are_not_multimodal():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert needs_multimodal(msgs) is False


def test_openai_image_url_shape():
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}]
    assert needs_multimodal(msgs) is True


def test_anthropic_image_source_shape():
    msgs = [{"role": "user", "content": [{"type": "image", "source": {"data": "x"}}]}]
    assert needs_multimodal(msgs) is True


def test_document_block():
    msgs = [{"role": "user", "content": [{"type": "document", "source": {"data": "x"}}]}]
    assert needs_multimodal(msgs) is True


def test_tool_result_nested_image_is_detected():
    # R1: the canonical Claude Code shape when a tool returns an image.
    msgs = [{
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [{"type": "image", "source": {"data": "x"}}],
        }],
    }]
    assert needs_multimodal(msgs) is True


def test_tool_result_nested_document_is_detected():
    msgs = [{"role": "user", "content": [
        {"type": "tool_result", "content": [{"type": "document", "source": {"data": "x"}}]},
    ]}]
    assert needs_multimodal(msgs) is True


def test_tool_result_nested_text_is_not_multimodal():
    msgs = [{"role": "user", "content": [
        {"type": "tool_result", "content": [{"type": "text", "text": "ok"}]},
    ]}]
    assert needs_multimodal(msgs) is False


def test_system_field_with_image_is_detected():
    # R2: the top-level system field is scanned too.
    assert needs_multimodal([], system=[{"type": "image", "source": {"data": "x"}}]) is True


def test_system_text_is_not_multimodal():
    assert needs_multimodal([], system="you are helpful") is False


def test_router_reroutes_image_to_multimodal():
    data = {"model": "coder", "messages": [
        {"role": "user", "content": [{"type": "image", "source": {}}, {"type": "text", "text": "?"}]},
    ]}
    assert ModalityRouter().route(data)["model"] == "multimodal"


def test_router_leaves_text_on_incoming_model():
    data = {"model": "coder", "messages": [{"role": "user", "content": "hi"}]}
    assert ModalityRouter().route(data)["model"] == "coder"


def test_router_leaves_background_on_incoming_model():
    data = {"model": "background", "messages": [{"role": "user", "content": "hi"}]}
    assert ModalityRouter().route(data)["model"] == "background"


def test_router_honors_env_route(monkeypatch):
    monkeypatch.setenv("PRISM_MULTIMODAL_ROUTE", "vision")
    data = {"model": "coder", "messages": [{"role": "user", "content": [{"type": "image"}]}]}
    assert ModalityRouter().route(data)["model"] == "vision"


def test_router_no_messages_key_is_safe():
    data = {"model": "coder"}
    assert ModalityRouter().route(data)["model"] == "coder"


# ── hosted web_search tool handling (the WebSearch InputValidationError-loop fix) ──
# Without a search provider configured, the hosted tool is *stripped* (Prism's
# backends can't execute it). With PRISM_SEARCH=1 (a provider is wired), it is
# *converted* to litellm's callable `litellm_web_search` function tool, which
# litellm's websearch_interception callback then serves.

def test_hosted_web_search_tool_is_stripped_by_default(monkeypatch):
    # No search provider → strip the hosted tool, keep custom tools.
    monkeypatch.delenv("PRISM_SEARCH", raising=False)
    data = {"model": "coder", "messages": [{"role": "user", "content": "hi"}], "tools": [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
        {"name": "Bash", "description": "x", "input_schema": {"type": "object"}},
    ]}
    tools = ModalityRouter().route(data)["tools"]
    assert [t.get("name") for t in tools] == ["Bash"]


def test_hosted_web_search_tool_is_converted_when_search_enabled(monkeypatch):
    # With a search provider wired, the hosted tool becomes the callable standard tool.
    monkeypatch.setenv("PRISM_SEARCH", "1")
    data = {"model": "coder", "messages": [{"role": "user", "content": "hi"}], "tools": [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
        {"name": "Bash", "description": "x", "input_schema": {"type": "object"}},
    ]}
    tools = ModalityRouter().route(data)["tools"]
    names = [t.get("name") for t in tools]
    assert "litellm_web_search" in names and "Bash" in names
    ws = next(t for t in tools if t.get("name") == "litellm_web_search")
    assert ws["input_schema"]["required"] == ["query"]
    # No hosted-type remnants leak through.
    assert "web_search_20250305" not in str(tools)


def test_duplicate_hosted_web_search_becomes_single_function_tool(monkeypatch):
    monkeypatch.setenv("PRISM_SEARCH", "1")
    data = {"tools": [
        {"type": "web_search_20250305", "name": "web_search"},
        {"type": "web_search_20250305", "name": "web_search"},
    ]}
    tools = ModalityRouter().route(data)["tools"]
    assert [t.get("name") for t in tools] == ["litellm_web_search"]


# ── mixed-tool agentic follow-up model qualification (litellm bug workaround) ─────

def test_qualify_followup_prepends_provider_to_vendor_prefixed_model():
    # The whole reason the mixed-tool follow-up broke: litellm's `"/" in model` check
    # sees "z-ai/glm-5.2" (vendor/model), assumes it's already provider-qualified, and
    # skips prepending "openrouter/" — so the follow-up can't resolve a provider. The
    # patch must qualify it against the provider litellm already populated.
    from prism.routing import _qualify_followup_model
    assert _qualify_followup_model("z-ai/glm-5.2", "openrouter") == "openrouter/z-ai/glm-5.2"


def test_qualify_followup_prepends_provider_to_openai_vendor_segment():
    # The latent bug the old slug-list check masked: "openai/gpt-4o-mini" is an OpenRouter
    # *vendor* segment, not the openai provider. The slug list treated it as already
    # qualified → the follow-up hit OpenAI directly. Qualifying against custom_llm_provider
    # correctly routes it back through OpenRouter.
    from prism.routing import _qualify_followup_model
    assert (_qualify_followup_model("openai/gpt-4o-mini", "openrouter")
            == "openrouter/openai/gpt-4o-mini")
    # Same for anthropic/cohere vendor segments that also collide with provider names.
    assert (_qualify_followup_model("anthropic/claude-3.5-sonnet", "openrouter")
            == "openrouter/anthropic/claude-3.5-sonnet")


def test_qualify_followup_leaves_already_qualified_model_untouched():
    # A model already carrying its call's provider must not be double-prefixed.
    from prism.routing import _qualify_followup_model
    assert _qualify_followup_model("openrouter/z-ai/glm-5.2", "openrouter") is None
    # When the provider genuinely IS openai, an "openai/..." model is already qualified.
    assert _qualify_followup_model("openai/gpt-4o-mini", "openai") is None
    # Missing provider or model → nothing to do (defensive, no crash).
    assert _qualify_followup_model("z-ai/glm-5.2", "") is None
    assert _qualify_followup_model("", "openrouter") is None


def test_hosted_tool_future_version_prefix_matches(monkeypatch):
    # The `startswith` check covers future Anthropic web_search versions.
    monkeypatch.delenv("PRISM_SEARCH", raising=False)
    data = {"tools": [{"type": "web_search_20261120", "name": "web_search"}]}
    assert "tools" not in ModalityRouter().route(data)


def test_custom_tool_with_type_custom_is_kept():
    # A plain custom tool may carry type "custom" — must NOT be stripped/converted.
    data = {"tools": [{"type": "custom", "name": "Bash", "input_schema": {"type": "object"}}]}
    assert ModalityRouter().route(data)["tools"] == data["tools"]


def test_all_hosted_tools_drops_tools_and_tool_choice(monkeypatch):
    # No search provider + every tool hosted → drop `tools` entirely (an empty array
    # is rejected by some backends) and take the now-orphaned `tool_choice` with it.
    monkeypatch.delenv("PRISM_SEARCH", raising=False)
    data = {"model": "coder", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "auto"}}
    out = ModalityRouter().route(data)
    assert "tools" not in out
    assert "tool_choice" not in out


def test_no_tools_key_is_safe():
    data = {"model": "coder", "messages": [{"role": "user", "content": "hi"}]}
    assert ModalityRouter().route(data) == {**data, "model": "coder"}
