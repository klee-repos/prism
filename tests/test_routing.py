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
