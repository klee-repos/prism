"""Crown e2e proof: a REAL litellm proxy with the Prism hook routes text -> coder and
image -> multimodal, proven by what a mock upstream actually received. The LLM decides
nothing; the mock records the underlying model per request.

Red/green: disable the hook (routing.py) and the image request lands on the coder model,
failing `test_image_request_routes_to_multimodal`.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import yaml

pytest.importorskip("litellm")

from prism import cli
from prism import config as cfgmod
from prism import proxy as proxymod

# A tiny valid base64 PNG payload; the exact bytes are irrelevant to routing.
IMG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _messages_have_image(messages):
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    return True
    return False


class MockUpstream:
    """Records the model + image-presence of every completion request it receives."""

    def __init__(self):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # some SDKs probe /models on init
                self._json({"data": [], "object": "list"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(body)
                except Exception:
                    payload = {}
                # S6: never persist the inbound Authorization header.
                outer.requests.append({
                    "model": payload.get("model"),
                    "had_image": _messages_have_image(payload.get("messages", [])),
                    "tools": payload.get("tools"),
                })
                self._json({
                    "id": "chatcmpl-mock", "object": "chat.completion", "created": 0,
                    "model": payload.get("model", "mock"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })

            def _json(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()


@pytest.fixture()
def running_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_HOME", str(tmp_path / ".prism"))
    monkeypatch.setenv("PRISM_TEST_KEY", "sk-dummy-not-real")
    with MockUpstream() as mock:
        cfg = {
            "schema_version": 1,
            "providers": {"mock": {
                "type": "openai", "api_key_env": "PRISM_TEST_KEY",
                "api_base": f"http://127.0.0.1:{mock.port}",
            }},
            "routes": {
                "coder": {"provider": "mock", "model": "prism-coder-model"},
                "background": {"provider": "mock", "model": "prism-bg-model"},
                "multimodal": {"provider": "mock", "model": "prism-mm-model"},
            },
            "mapping": {"opus": "coder", "sonnet": "coder", "haiku": "background"},
        }
        home = cfgmod.prism_home()
        home.mkdir(parents=True, exist_ok=True)
        cfgmod.config_path().write_text(yaml.safe_dump(cfg))
        cli._write_hook_shim()
        proxy = proxymod.start(cfg)
        try:
            yield mock, proxy
        finally:
            proxy.stop()


def _send(proxy, messages):
    # No Authorization header: the proxy is loopback-only and does not gate on a key.
    r = httpx.post(
        f"{proxy.base_url}/v1/messages",
        headers={"content-type": "application/json"},
        json={"model": "coder", "max_tokens": 16, "messages": messages},
        timeout=30.0,
    )
    assert r.status_code == 200, r.text
    return r


def test_text_request_routes_to_coder(running_proxy):
    mock, proxy = running_proxy
    _send(proxy, [{"role": "user", "content": "hello there"}])
    assert mock.requests, "mock upstream received nothing"
    assert mock.requests[-1]["model"] == "prism-coder-model"


def test_image_request_routes_to_multimodal(running_proxy):
    mock, proxy = running_proxy
    _send(proxy, [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMG_B64}},
            {"type": "text", "text": "what is this"},
        ],
    }])
    assert mock.requests[-1]["model"] == "prism-mm-model", (
        "image request did not reroute to the multimodal model — the hook did not fire or route"
    )


def test_hook_fired_marker_in_proxy_log(running_proxy):
    mock, proxy = running_proxy
    _send(proxy, [{"role": "user", "content": "ping"}])
    log = cfgmod.log_path().read_text(errors="replace")
    assert "prism.hook.fired" in log, "hook never logged its fire marker (was it importable?)"


def test_run_passthrough_boots_a_routed_proxy_and_wires_env(tmp_path, monkeypatch):
    """Prove the whole `prism <args>` launch chain: provision -> key preflight -> boot
    proxy -> build env -> (claude) reaches a working, routing proxy. `claude` is replaced
    by a probe that uses the exact env Prism wired and hits the proxy with an image."""
    monkeypatch.setenv("PRISM_HOME", str(tmp_path / ".prism"))
    monkeypatch.setenv("PRISM_TEST_KEY", "sk-dummy-not-real")
    with MockUpstream() as mock:
        cfg = {
            "schema_version": 1,
            "providers": {"mock": {
                "type": "openai", "api_key_env": "PRISM_TEST_KEY",
                "api_base": f"http://127.0.0.1:{mock.port}",
            }},
            "routes": {
                "coder": {"provider": "mock", "model": "prism-coder-model"},
                "background": {"provider": "mock", "model": "prism-bg-model"},
                "multimodal": {"provider": "mock", "model": "prism-mm-model"},
            },
            "mapping": {"opus": "coder", "sonnet": "coder", "haiku": "background"},
        }
        home = cfgmod.prism_home()
        home.mkdir(parents=True, exist_ok=True)
        cfgmod.config_path().write_text(yaml.safe_dump(cfg))

        seen = {}

        def fake_exec(forward_args, env=None):
            seen["base_url"] = env["ANTHROPIC_BASE_URL"]
            seen["opus_model"] = env["ANTHROPIC_DEFAULT_OPUS_MODEL"]
            r = httpx.post(
                env["ANTHROPIC_BASE_URL"] + "/v1/messages",
                headers={"Authorization": f"Bearer {env['ANTHROPIC_AUTH_TOKEN']}"},
                json={"model": "coder", "max_tokens": 16, "messages": [{
                    "role": "user",
                    "content": [{"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": IMG_B64}}],
                }]},
                timeout=30.0,
            )
            seen["status"] = r.status_code
            return 0

        monkeypatch.setattr(cli, "_exec_claude", fake_exec)
        rc = cli.run_passthrough(["-p", "hi"])

        assert rc == 0
        assert seen["status"] == 200
        assert seen["base_url"].startswith("http://127.0.0.1:")
        assert seen["opus_model"] == "coder"  # mapping wired into claude's env
        # An image sent through the Prism-launched proxy routed to the multimodal model.
        assert mock.requests[-1]["model"] == "prism-mm-model"


def test_tool_result_nested_image_routes_to_multimodal(running_proxy):
    # R1 end-to-end: an image returned inside a tool_result must still route multimodal.
    mock, proxy = running_proxy
    _send(proxy, [
        {"role": "user", "content": "look at this"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "t1",
            "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMG_B64}}],
        }]},
    ])
    assert mock.requests[-1]["model"] == "prism-mm-model"


@pytest.fixture()
def running_proxy_k3_profile(tmp_path, monkeypatch):
    """A REAL litellm proxy booted from a PROFILES config with `active_profile: k3`. Proves the
    whole new chain — active-profile resolution -> to_litellm_config -> litellm boot -> the
    routing hook — end to end, verified by what the mock upstream actually received."""
    monkeypatch.setenv("PRISM_HOME", str(tmp_path / ".prism"))
    monkeypatch.setenv("PRISM_TEST_KEY", "sk-dummy-not-real")
    with MockUpstream() as mock:
        cfg = {
            "schema_version": 1,
            "active_profile": "k3",
            "providers": {"mock": {
                # openrouter type -> chat-completions path + Anthropic->OpenAI image conversion,
                # the exact path moonshotai/kimi-k3 uses in production.
                "type": "openrouter", "api_key_env": "PRISM_TEST_KEY",
                "api_base": f"http://127.0.0.1:{mock.port}",
            }},
            "profiles": {
                "glm": {
                    "coder": {"provider": "mock", "model": "z-ai/glm-5.2"},
                    "background": {"provider": "mock", "model": "z-ai/glm-4.7-flash"},
                    "multimodal": {"provider": "mock", "model": "google/gemini-2.5-flash"},
                },
                "k3": {
                    "coder": {"provider": "mock", "model": "moonshotai/kimi-k3"},
                    "background": {"provider": "mock", "model": "z-ai/glm-4.7-flash"},
                    "multimodal": {"provider": "mock", "model": "moonshotai/kimi-k3"},
                },
            },
            "mapping": {"opus": "coder", "sonnet": "coder", "haiku": "background"},
        }
        home = cfgmod.prism_home()
        home.mkdir(parents=True, exist_ok=True)
        cfgmod.config_path().write_text(yaml.safe_dump(cfg))
        cli._write_hook_shim()
        proxy = proxymod.start(cfg)
        try:
            yield mock, proxy
        finally:
            proxy.stop()


def test_k3_profile_text_and_image_both_route_to_kimi(running_proxy_k3_profile):
    # Deterministic real-artifact verdict: with the k3 profile active, a TEXT request AND an
    # IMAGE request must BOTH reach the upstream as moonshotai/kimi-k3 — K3 serves vision too,
    # so no separate model is needed. The mock records the underlying model; the LLM decides
    # nothing. Red/green: point k3.multimodal at a different model and the image assert fails.
    mock, proxy = running_proxy_k3_profile
    _send(proxy, [{"role": "user", "content": "hello there"}])
    assert mock.requests[-1]["model"] == "moonshotai/kimi-k3", "text did not route to k3 coder"
    _send(proxy, [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMG_B64}},
            {"type": "text", "text": "what is this"},
        ],
    }])
    assert mock.requests[-1]["model"] == "moonshotai/kimi-k3", "image did not route to k3 multimodal"
    assert mock.requests[-1]["had_image"] is True, "the image did not actually reach the upstream"


def test_hosted_web_search_tool_is_stripped_before_upstream(running_proxy):
    # No search provider configured → strip the hosted tool (backends can't run it),
    # keep custom tools. This is the default and the guard against the
    # InputValidationError loop.
    mock, proxy = running_proxy
    r = httpx.post(
        f"{proxy.base_url}/v1/messages",
        headers={"content-type": "application/json"},
        json={"model": "coder", "max_tokens": 16,
              "messages": [{"role": "user", "content": "search the web"}],
              "tools": [
                  {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
                  {"name": "Bash", "description": "x", "input_schema": {"type": "object"}},
              ]},
        timeout=30.0,
    )
    assert r.status_code == 200, r.text
    last = mock.requests[-1]
    tools = last["tools"]
    # The hosted web_search tool never reached the upstream…
    assert tools is not None and not any(
        isinstance(t, dict) and isinstance(t.get("type"), str) and t["type"].startswith("web_search")
        for t in tools
    ), "hosted web_search tool was not stripped before the upstream model"
    # …but the plain custom Bash tool did survive.
    assert any(isinstance(t, dict) and t.get("name") == "Bash" for t in tools)


@pytest.fixture()
def search_enabled_proxy(tmp_path, monkeypatch):
    """A proxy with a search provider wired (PRISM_SEARCH=1) so the hook *converts*
    the hosted web_search tool to litellm's callable ``litellm_web_search`` instead
    of stripping it. The search provider points at a dummy host; we only assert the
    tool-shape the LLM upstream receives, not the search-loop's execution."""
    monkeypatch.setenv("PRISM_HOME", str(tmp_path / ".prism"))
    monkeypatch.setenv("PRISM_TEST_KEY", "sk-dummy-not-real")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-dummy-not-real")
    monkeypatch.setenv("PRISM_SEARCH", "1")
    with MockUpstream() as mock:
        cfg = {
            "schema_version": 1,
            "providers": {"mock": {
                # openrouter routes /v1/messages through the chat-completions path
                # (where the conversion is observable at the upstream), unlike `openai`
                # which uses the Responses API path.
                "type": "openrouter", "api_key_env": "PRISM_TEST_KEY",
                "api_base": f"http://127.0.0.1:{mock.port}",
            }},
            "routes": {
                "coder": {"provider": "mock", "model": "prism-coder-model"},
                "background": {"provider": "mock", "model": "prism-bg-model"},
                "multimodal": {"provider": "mock", "model": "prism-mm-model"},
            },
            "mapping": {"opus": "coder", "sonnet": "coder", "haiku": "background"},
            "search": {"provider": "firecrawl", "api_key_env": "FIRECRAWL_API_KEY"},
        }
        home = cfgmod.prism_home()
        home.mkdir(parents=True, exist_ok=True)
        cfgmod.config_path().write_text(yaml.safe_dump(cfg))
        cli._write_hook_shim()
        proxy = proxymod.start(cfg)
        try:
            yield mock, proxy
        finally:
            proxy.stop()


def _tool_name(t: dict) -> str:
    """Effective tool name, handling the OpenAI ``{type:function, function:{name}}`` shape."""
    if t.get("type") == "function" and isinstance(t.get("function"), dict):
        return t["function"].get("name", "")
    return t.get("name", "")


def test_hosted_web_search_is_converted_when_search_enabled(search_enabled_proxy):
    # With a search provider wired, the hook converts the hosted web_search tool to
    # litellm's standard callable ``litellm_web_search`` function tool — so the model
    # can call it and litellm's websearch_interception callback serves the search.
    mock, proxy = search_enabled_proxy
    r = httpx.post(
        f"{proxy.base_url}/v1/messages",
        headers={"content-type": "application/json"},
        json={"model": "coder", "max_tokens": 16,
              "messages": [{"role": "user", "content": "search the web"}],
              "tools": [
                  {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
                  {"name": "Bash", "description": "x", "input_schema": {"type": "object"}},
              ]},
        timeout=30.0,
    )
    assert r.status_code == 200, r.text
    last = mock.requests[-1]
    tools = last["tools"]
    names = [_tool_name(t) for t in tools if isinstance(t, dict)]
    # The model now sees a callable litellm_web_search function tool (not the hosted
    # server tool that would loop it).
    assert "litellm_web_search" in names, (
        "hosted web_search was not converted to the callable litellm_web_search tool; "
        f"model saw: {names}"
    )
    # No hosted-type remnants leak through.
    assert not any(
        isinstance(t, dict) and isinstance(t.get("type"), str) and t["type"].startswith("web_search")
        for t in tools
    )
    # Custom Bash tool still survives alongside.
    assert "Bash" in names


def test_mixed_tool_websearch_loop_completes(tmp_path, monkeypatch):
    # Regression for the litellm mixed-tool agentic-loop bug: when web_search is sent
    # alongside Bash, litellm's _execute_chat_completion_agentic_plan left the follow-up
    # model as the bare "z-ai/..." deployment id → "LLM Provider NOT provided" → the
    # loop aborted before the follow-up call. Prism's follow-up patch qualifies the
    # model, so the loop must now reach the second LLM call.
    monkeypatch.setenv("PRISM_HOME", str(tmp_path / ".prism"))
    monkeypatch.setenv("PRISM_TEST_KEY", "sk-dummy-not-real")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-dummy-not-real")
    monkeypatch.setenv("PRISM_SEARCH", "1")

    # A mock LLM upstream that calls litellm_web_search on turn 1, then synthesizes.
    import json as _json
    llm_calls = {"n": 0}

    class LLMHandler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            self._json({"data": [], "object": "list"})

        def do_POST(self):
            llm_calls["n"] += 1
            length = int(self.headers.get("Content-Length", 0))
            payload = _json.loads(self.rfile.read(length)) if length else {}
            outer = self  # for capturing in the closure below
            if llm_calls["n"] == 1:
                resp = {"id": "x", "object": "chat.completion", "created": 0, "model": "glm",
                        "choices": [{"index": 0, "finish_reason": "tool_calls",
                                     "message": {"role": "assistant", "content": None,
                                                 "tool_calls": [{"id": "c1", "type": "function",
                                                                 "function": {"name": "litellm_web_search",
                                                                              "arguments": _json.dumps({"query": "maestro yaml"})}}]}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            else:
                resp = {"id": "y", "object": "chat.completion", "created": 0, "model": "glm",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "synthesized answer"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            self._json(resp)

        def _json(self, obj):
            data = _json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    llm_srv = ThreadingHTTPServer(("127.0.0.1", 0), LLMHandler)
    llm_port = llm_srv.server_address[1]
    threading.Thread(target=llm_srv.serve_forever, daemon=True).start()

    # Stub litellm.asearch so the search runs without a real Firecrawl key/network.
    import litellm

    async def fake_asearch(**kwargs):
        from litellm.types.utils import SearchResponse, SearchAPIResult
        return SearchResponse(results=[SearchAPIResult(
            title="Maestro YAML Guide", url="https://maestro.mobile/yaml",
            content="flow: and commands: are top-level keys", score=0.9,
        )])
    monkeypatch.setattr(litellm, "asearch", fake_asearch)

    cfg = {
        "schema_version": 1,
        "providers": {"mock": {
            "type": "openrouter", "api_key_env": "PRISM_TEST_KEY",
            "api_base": f"http://127.0.0.1:{llm_port}",
        }},
        "routes": {
            "coder": {"provider": "mock", "model": "z-ai/glm-5.2"},
            "background": {"provider": "mock", "model": "z-ai/glm-4.7-flash"},
            "multimodal": {"provider": "mock", "model": "google/gemini-2.5-flash"},
        },
        "mapping": {"opus": "coder", "sonnet": "coder", "haiku": "background"},
        "search": {"provider": "firecrawl", "api_key_env": "FIRECRAWL_API_KEY"},
    }
    home = cfgmod.prism_home()
    home.mkdir(parents=True, exist_ok=True)
    cfgmod.config_path().write_text(yaml.safe_dump(cfg))
    cli._write_hook_shim()
    proxy = proxymod.start(cfg)
    try:
        r = httpx.post(
            f"{proxy.base_url}/v1/messages",
            headers={"content-type": "application/json"},
            json={"model": "coder", "max_tokens": 200,
                  "messages": [{"role": "user", "content": "search for maestro yaml syntax"}],
                  "tools": [
                      {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
                      {"name": "Bash", "description": "x",
                       "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
                  ]},
            timeout=60.0,
        )
    finally:
        proxy.stop()
        llm_srv.shutdown()

    assert r.status_code == 200, r.text
    # Two LLM upstream calls = the model called web_search on turn 1, litellm ran the
    # search, and the follow-up synthesis call (turn 2) actually executed. Before the
    # patch, the second call never happened (the loop aborted with "LLM Provider NOT
    # provided").
    assert llm_calls["n"] >= 2, (
        f"mixed-tool agentic loop did not reach the follow-up call "
        f"(only {llm_calls['n']} LLM call(s)) — the provider-qualification patch "
        "may have regressed."
    )


def test_disconnect_metadata_patch_handles_none():
    # litellm 1.91.0's _record_streaming_client_disconnect_if_needed hands metadata
    # obtained via setdefault("metadata", {}) to _apply_client_disconnect_metadata,
    # but a slot can pre-exist as None → setdefault returns None → the bare helper
    # raises TypeError on the item assignment. Prism wraps the helper to no-op on a
    # non-dict. Verify the guard without depending on a live disconnect: just confirm
    # the wrapped helper tolerates None where the original would crash.
    from litellm.proxy import common_request_processing as _crp
    from prism.routing import _install_disconnect_metadata_patch

    _install_disconnect_metadata_patch()  # idempotent
    fn = _crp._apply_client_disconnect_metadata
    assert getattr(fn, "_prism_guarded", True)  # patched (or litellm already fixed)

    # A dict is still mutated (real path):
    d: dict = {}
    fn(d)
    assert d.get("client_disconnected") is True

    # The bug: None / a non-dict must NOT raise.
    fn(None)
    fn("not-a-dict")


def test_mixed_tool_websearch_loop_qualifies_openai_vendored_model(tmp_path, monkeypatch):
    # Regression for the latent follow-up-qualification bug: a model whose OpenRouter
    # *vendor* segment collides with a real litellm provider name ("openai/gpt-4o-mini",
    # "anthropic/…", "cohere/…"). litellm leaves the mixed-tool follow-up model as the bare
    # "openai/gpt-4o-mini"; the old guard saw `.startswith("openai/")` and skipped
    # prepending the provider, so the synthesis turn resolved provider "openai" and hit
    # OpenAI directly instead of the configured OpenRouter deployment. The tell is the
    # model the upstream receives on turn 2: OpenRouter forwards the vendor-prefixed slug
    # verbatim ("openai/gpt-4o-mini"), whereas the OpenAI provider transform strips it to
    # the bare "gpt-4o-mini". Mirrors test_mixed_tool_websearch_loop_completes (which
    # must NOT regress) but with the vendor-colliding model.
    monkeypatch.setenv("PRISM_HOME", str(tmp_path / ".prism"))
    monkeypatch.setenv("PRISM_TEST_KEY", "sk-dummy-not-real")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-dummy-not-real")
    monkeypatch.setenv("PRISM_SEARCH", "1")

    import json as _json
    llm_calls = {"n": 0}
    seen_models: list = []  # the `model` field of each request the upstream receives, in order

    class LLMHandler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            self._json({"data": [], "object": "list"})

        def do_POST(self):
            llm_calls["n"] += 1
            length = int(self.headers.get("Content-Length", 0))
            payload = _json.loads(self.rfile.read(length)) if length else {}
            seen_models.append(payload.get("model"))
            if llm_calls["n"] == 1:
                resp = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                        "choices": [{"index": 0, "finish_reason": "tool_calls",
                                     "message": {"role": "assistant", "content": None,
                                                 "tool_calls": [{"id": "c1", "type": "function",
                                                                 "function": {"name": "litellm_web_search",
                                                                              "arguments": _json.dumps({"query": "maestro yaml"})}}]}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            else:
                resp = {"id": "y", "object": "chat.completion", "created": 0, "model": "m",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "synthesized answer"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            self._json(resp)

        def _json(self, obj):
            data = _json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    llm_srv = ThreadingHTTPServer(("127.0.0.1", 0), LLMHandler)
    llm_port = llm_srv.server_address[1]
    threading.Thread(target=llm_srv.serve_forever, daemon=True).start()

    # Stub litellm.asearch so the search runs without a real Firecrawl key/network.
    import litellm

    async def fake_asearch(**kwargs):
        from litellm.types.utils import SearchResponse, SearchAPIResult
        return SearchResponse(results=[SearchAPIResult(
            title="Maestro YAML Guide", url="https://maestro.mobile/yaml",
            content="flow: and commands: are top-level keys", score=0.9,
        )])
    monkeypatch.setattr(litellm, "asearch", fake_asearch)

    cfg = {
        "schema_version": 1,
        "providers": {"mock": {
            "type": "openrouter", "api_key_env": "PRISM_TEST_KEY",
            "api_base": f"http://127.0.0.1:{llm_port}",
        }},
        "routes": {
            # Vendor segment "openai" collides with the litellm provider name — the bug.
            "coder": {"provider": "mock", "model": "openai/gpt-4o-mini"},
            "background": {"provider": "mock", "model": "openai/gpt-4o-mini"},
            "multimodal": {"provider": "mock", "model": "google/gemini-2.5-flash"},
        },
        "mapping": {"opus": "coder", "sonnet": "coder", "haiku": "background"},
        "search": {"provider": "firecrawl", "api_key_env": "FIRECRAWL_API_KEY"},
    }
    home = cfgmod.prism_home()
    home.mkdir(parents=True, exist_ok=True)
    cfgmod.config_path().write_text(yaml.safe_dump(cfg))
    cli._write_hook_shim()
    proxy = proxymod.start(cfg)
    try:
        r = httpx.post(
            f"{proxy.base_url}/v1/messages",
            headers={"content-type": "application/json"},
            json={"model": "coder", "max_tokens": 200,
                  "messages": [{"role": "user", "content": "search for maestro yaml syntax"}],
                  "tools": [
                      {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
                      {"name": "Bash", "description": "x",
                       "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
                  ]},
            timeout=60.0,
        )
    finally:
        proxy.stop()
        llm_srv.shutdown()

    assert r.status_code == 200, r.text
    # The loop must reach the follow-up synthesis call…
    assert llm_calls["n"] >= 2, (
        f"mixed-tool agentic loop did not reach the follow-up call "
        f"(only {llm_calls['n']} LLM call(s), models={seen_models})"
    )
    # …and, crucially, that follow-up must go through OpenRouter (provider-qualified to
    # "openrouter/openai/gpt-4o-mini"), which forwards the vendor-prefixed slug verbatim.
    # If the guard regresses, the follow-up resolves provider "openai" and the OpenAI
    # transform strips the prefix to "gpt-4o-mini" — hitting OpenAI directly.
    assert seen_models[-1] == "openai/gpt-4o-mini", (
        f"mixed-tool follow-up was not routed through OpenRouter — upstream saw "
        f"model={seen_models[-1]!r} (expected 'openai/gpt-4o-mini'; a bare 'gpt-4o-mini' "
        f"means the OpenAI provider path was taken). Full sequence: {seen_models}"
    )

