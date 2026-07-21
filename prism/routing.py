"""The Prism modality router — a LiteLLM pre-call hook.

Any request carrying an image or file is rerouted to the ``multimodal`` route;
plain-text requests keep whatever model Claude Code chose (``coder`` / ``background``).

Verified against litellm 1.91.0: on the standard ``/v1/messages`` path the hook fires,
sees **raw Anthropic-native content** (``{"type": "image", "source": {...}}``), and a
``data["model"]`` rewrite survives to the router. We still detect the OpenAI
(``image_url``) shape defensively, and we recurse into nested ``content`` lists so an
image returned inside a ``tool_result`` block (the shape Claude Code produces when a
tool returns an image) is not missed.
"""
from __future__ import annotations

import os
import sys
from typing import Any

try:  # Real base class inside the proxy; a shim keeps unit tests litellm-free.
    from litellm.integrations.custom_logger import CustomLogger
except Exception:  # pragma: no cover
    class CustomLogger:  # type: ignore[no-redef]
        pass

# Content-part `type` values that indicate non-text input, across Anthropic + OpenAI.
MULTIMODAL_TYPES = frozenset(
    {"image_url", "image", "input_image", "file", "document", "input_file"}
)

DEFAULT_MULTIMODAL_ROUTE = "multimodal"

# Anthropic *hosted* (server-side) tool `type` prefixes. These tools are executed
# on Anthropic's servers, not by the client — Claude Code's built-in WebSearch is
# sent as ``{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}``.
# Prism's backends (GLM, Gemini-via-OpenRouter, …) cannot execute them: litellm
# rewrites a hosted ``web_search_*`` tool to OpenAI's ``{"type": "web_search_preview"}``
# and forwards it to a model that doesn't support it, the model then emits a malformed
# tool_call, Claude Code rejects it with InputValidationError, and the request loops
# forever. Instead of passing the hosted tool through, we *convert* it to litellm's
# standard callable ``litellm_web_search`` function tool — which the model CAN call,
# and which litellm's WebSearchInterceptionLogger (registered by Prism when a search
# provider is configured) intercepts and executes. Matched by prefix so future-dated
# variants (``web_search_2026…``) are covered without a code change.
HOSTED_WEB_SEARCH_TYPE_PREFIXES = ("web_search",)

# The standard tool litellm's interception loop recognizes and executes. Mirrors
# litellm.integrations.websearch_interception.tools.get_litellm_web_search_tool() so
# the hook is unit-testable without importing litellm.
LITELLM_WEB_SEARCH_TOOL = {
    "name": "litellm_web_search",
    "description": (
        "Search the web for information. Use this when you need current "
        "information or answers to questions that require up-to-date data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query to execute"},
        },
        "required": ["query"],
    },
}

def _content_has_multimodal(content: Any) -> bool:
    """Depth-first scan of a message ``content`` value for any multimodal part.

    Descends into part dicts that themselves nest a ``content`` list (e.g. Anthropic
    ``tool_result`` blocks carrying an image/document), at any depth.
    """
    if content is None or isinstance(content, str):
        return False
    if isinstance(content, dict):
        if content.get("type") in MULTIMODAL_TYPES:
            return True
        return _content_has_multimodal(content.get("content"))
    if isinstance(content, (list, tuple)):
        return any(_content_has_multimodal(part) for part in content)
    return False


def needs_multimodal(messages: Any, system: Any = None) -> bool:
    """True if any message — or the top-level ``system`` field — carries an image/file."""
    if system is not None and _content_has_multimodal(system):
        return True
    if not isinstance(messages, (list, tuple)):
        return False
    for msg in messages:
        if isinstance(msg, dict) and _content_has_multimodal(msg.get("content")):
            return True
    return False


class ModalityRouter(CustomLogger):
    """Reroutes image/file requests to the multimodal route via ``data['model']``
    and converts Anthropic-hosted web_search tools into litellm's standard callable
    ``litellm_web_search`` function tool, so cheap backends can still search."""

    def __init__(self) -> None:
        self._fired = False

    def _multimodal_route(self) -> str:
        return os.environ.get("PRISM_MULTIMODAL_ROUTE", DEFAULT_MULTIMODAL_ROUTE)

    @staticmethod
    def _is_hosted_web_search(tool: Any) -> bool:
        """True if ``tool`` is an Anthropic-hosted web_search server tool.

        Claude Code sends ``{"type": "web_search_20250305", "name": "web_search", ...}``;
        the version suffix is date-stamped, so match by prefix.
        """
        if not isinstance(tool, dict):
            return False
        t = tool.get("type")
        if isinstance(t, str) and t.startswith(HOSTED_WEB_SEARCH_TYPE_PREFIXES):
            return True
        # The Claude Code CLI shape uses ``name="web_search"`` with a versioned type;
        # a bare ``name`` (no versioned type) is not a hosted tool.
        return False

    def _web_search_enabled(self) -> bool:
        """True when Prism's generated config registered a search provider.

        Set by ``cli.build_claude_env`` from ``config.search_enabled``; the hook runs
        in the litellm subprocess so it reads the flag from the environment, not the
        Prism config object. When False, hosted web_search tools are stripped (the
        cheap backends can't execute them) rather than converted to a function tool
        that would never be served.
        """
        return os.environ.get("PRISM_SEARCH") == "1"

    def _convert_hosted_web_search(self, data: dict) -> None:
        """Handle hosted web_search tools: convert (if search is configured) or strip.

        The hosted tool is server-side (Anthropic runs it); Prism's backends can't.
        litellm's adapter would otherwise push it to the provider as OpenAI's
        ``web_search_preview``, which GLM/Gemini-via-OpenRouter ignore → the model
        emits a malformed tool_call → InputValidationError loop.

        When a search provider is configured, we convert it to litellm's standard
        ``litellm_web_search`` function tool — the model can call it, and litellm's
        ``websearch_interception`` callback runs the search via ``litellm.asearch``
        then re-calls the model with the results. When no search provider is
        configured, we strip it (the model simply won't call WebSearch) rather than
        leave a function tool whose results nothing can fulfill.
        """
        tools = data.get("tools")
        if not isinstance(tools, list) or not tools:
            return
        if not any(self._is_hosted_web_search(t) for t in tools):
            return

        if self._web_search_enabled():
            new_tools: list = []
            replaced = False
            for t in tools:
                if self._is_hosted_web_search(t):
                    if not replaced:
                        new_tools.append(dict(LITELLM_WEB_SEARCH_TOOL))
                        replaced = True
                    # Drop duplicate hosted web_search entries (one function tool suffices).
                else:
                    new_tools.append(t)
            data["tools"] = new_tools
        else:
            kept = [t for t in tools if not self._is_hosted_web_search(t)]
            if kept:
                data["tools"] = kept
            else:
                del data["tools"]
                data.pop("tool_choice", None)  # orphaned without any tools

    def route(self, data: dict) -> dict:
        """Pure routing + web-search-tool decision — unit-testable without litellm."""
        if needs_multimodal(data.get("messages"), data.get("system")):
            data["model"] = self._multimodal_route()
        self._convert_hosted_web_search(data)
        return data

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if not self._fired:
            self._fired = True
            # Proof marker (lands in the proxy log). Never log request bodies/keys.
            print("prism.hook.fired", file=sys.stderr, flush=True)
            # When web search is enabled, patch litellm's follow-up model resolution so
            # the mixed-tool agentic loop (web_search + Bash in one request) can re-call
            # the model. litellm's _build_anthropic_request_patch reads the bare
            # deployment model from agentic_loop_params["model"] (e.g. "z-ai/glm-5.2")
            # but ignores the adjacent "custom_llm_provider" — so the follow-up call
            # fails with "LLM Provider NOT provided". We qualify the model using the
            # provider litellm already populated. Idempotent; guarded so a litellm
            # layout change degrades to the original behavior rather than crashing.
            _install_websearch_followup_patch()
            _install_disconnect_metadata_patch()
        return self.route(data)


def _qualify_followup_model(model: str, custom_llm_provider: str) -> str | None:
    """Provider-qualify an agentic follow-up model, or ``None`` if already qualified.

    litellm resolves the follow-up ``patch.model`` to the bare deployment id and, because
    it contains a ``/`` *vendor* separator, skips prepending the provider. That leaves the
    follow-up model either provider-less (``z-ai/glm-5.2`` → ``LLM Provider NOT provided``)
    or resolvable to the WRONG provider (``openai/gpt-4o-mini`` → litellm dispatches to
    OpenAI directly instead of the configured OpenRouter deployment). The only reliable
    signal for "which provider serves *this* call" is the ``custom_llm_provider`` litellm
    already populated — not the vendor segment of the model id, which for OpenRouter
    (``openai/…``, ``anthropic/…``, ``cohere/…``) collides with real provider names.

    So prepend ``{custom_llm_provider}/`` unless the model is already qualified with it.
    Returns the qualified model, or ``None`` to leave it unchanged.
    """
    if not model or not custom_llm_provider:
        return None
    if model.startswith(f"{custom_llm_provider}/"):
        return None
    return f"{custom_llm_provider}/{model}"


_websearch_patch_installed = False


def _install_websearch_followup_patch() -> None:
    """Qualify the agentic follow-up model with its provider (litellm bug workaround).

    litellm 1.91.0's ``_execute_chat_completion_agentic_plan`` (the mixed-tool websearch
    agentic path) builds the follow-up model as ``patch.model or model`` then prepends
    ``custom_llm_provider/`` *only when the model has no* ``/``. OpenRouter model ids
    contain ``/`` as the *vendor* separator (``z-ai/glm-5.2``), so the check is skipped
    and the follow-up calls ``litellm.acompletion(model="z-ai/glm-5.2")`` — which has no
    resolvable provider, raising ``LLM Provider NOT provided``. We wrap the method to
    re-qualify the model against the ``custom_llm_provider`` litellm already populated
    (see ``_qualify_followup_model``). Idempotent + defensive: if litellm's layout moves,
    the original (buggy) behavior stands rather than crashing the follow-up.
    """
    global _websearch_patch_installed
    if _websearch_patch_installed:
        return
    try:
        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    except Exception:  # pragma: no cover
        return  # litellm not importable (unit tests) → nothing to patch

    target = getattr(BaseLLMHTTPHandler, "_execute_chat_completion_agentic_plan", None)
    if target is None or getattr(target, "_prism_qualified", False):
        _websearch_patch_installed = True
        return

    async def patched(self, plan, model, messages, optional_params, kwargs,
                      custom_llm_provider, depth, max_loops, fingerprints, fingerprint):
        # Re-qualify the model litellm's plan resolved (the router left it as the bare
        # deployment id, e.g. "z-ai/glm-5.2" or "openai/gpt-4o-mini"). Prepend the
        # provider litellm already populated unless the model already carries it. Mirrors
        # litellm's own intent at line 5221 — the bug is its `"/" in full_model_name`
        # check, which false-positives on any vendor-prefixed id.
        patch = plan.request_patch
        if patch is not None and patch.model:
            qualified = _qualify_followup_model(patch.model, custom_llm_provider)
            if qualified is not None:
                plan.request_patch = patch.model_copy(update={"model": qualified})
        return await target(
            self, plan, model, messages, optional_params, kwargs,
            custom_llm_provider, depth, max_loops, fingerprints, fingerprint,
        )

    patched._prism_qualified = True  # type: ignore[attr-defined]
    BaseLLMHTTPHandler._execute_chat_completion_agentic_plan = patched  # type: ignore[assignment]
    _websearch_patch_installed = True


_disconnect_patch_installed = False


def _install_disconnect_metadata_patch() -> None:
    """Guard litellm's streaming client-disconnect bookkeeping against a None metadata.

    litellm 1.91.0's ``_record_streaming_client_disconnect_if_needed`` calls
    ``_apply_client_disconnect_metadata(metadata)`` on several metadata dicts obtained
    via ``setdefault("metadata", {})``. But a metadata slot can already be present with
    value ``None`` (some call paths pre-set it to None), and ``setdefault`` then returns
    that ``None`` without replacing it — so ``target_metadata["client_disconnected"]=True``
    raises ``TypeError: 'NoneType' object does not support item assignment``. The request
    has already streamed its 200; this only crashes the post-response cleanup, but it
    spams the proxy log and can mask real errors. We wrap the helper to no-op (and warn)
    when handed a non-dict, which is the only behavior change. Idempotent + defensive.
    """
    global _disconnect_patch_installed
    if _disconnect_patch_installed:
        return
    try:
        from litellm.proxy import common_request_processing as _crp
    except Exception:  # pragma: no cover
        return  # litellm not importable (unit tests) → nothing to patch
    target = getattr(_crp, "_apply_client_disconnect_metadata", None)
    if target is None or getattr(target, "_prism_guarded", False):
        _disconnect_patch_installed = True
        return

    def guarded(target_metadata):  # type: ignore[no-untyped-def]
        if not isinstance(target_metadata, dict):
            # Don't crash the cleanup path; litellm's own contract assumes a dict here.
            return
        return target(target_metadata)

    guarded._prism_guarded = True  # type: ignore[attr-defined]
    _crp._apply_client_disconnect_metadata = guarded  # type: ignore[assignment]
    _disconnect_patch_installed = True


# The instance litellm loads via `litellm_settings.callbacks: prism_hook.instance`.
instance = ModalityRouter()
