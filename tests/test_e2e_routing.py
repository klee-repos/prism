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
        master_key = cli._ensure_master_key()
        proxy = proxymod.start(cfg, master_key)
        try:
            yield mock, proxy, master_key
        finally:
            proxy.stop()


def _send(proxy, master_key, messages):
    r = httpx.post(
        f"{proxy.base_url}/v1/messages",
        headers={"Authorization": f"Bearer {master_key}", "content-type": "application/json"},
        json={"model": "coder", "max_tokens": 16, "messages": messages},
        timeout=30.0,
    )
    assert r.status_code == 200, r.text
    return r


def test_text_request_routes_to_coder(running_proxy):
    mock, proxy, key = running_proxy
    _send(proxy, key, [{"role": "user", "content": "hello there"}])
    assert mock.requests, "mock upstream received nothing"
    assert mock.requests[-1]["model"] == "prism-coder-model"


def test_image_request_routes_to_multimodal(running_proxy):
    mock, proxy, key = running_proxy
    _send(proxy, key, [{
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
    mock, proxy, key = running_proxy
    _send(proxy, key, [{"role": "user", "content": "ping"}])
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
    mock, proxy, key = running_proxy
    _send(proxy, key, [
        {"role": "user", "content": "look at this"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "t1",
            "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMG_B64}}],
        }]},
    ])
    assert mock.requests[-1]["model"] == "prism-mm-model"
