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
    """Reroutes image/file requests to the multimodal route via ``data['model']``."""

    def __init__(self) -> None:
        self._fired = False

    def _multimodal_route(self) -> str:
        return os.environ.get("PRISM_MULTIMODAL_ROUTE", DEFAULT_MULTIMODAL_ROUTE)

    def route(self, data: dict) -> dict:
        """Pure routing decision — unit-testable without litellm."""
        if needs_multimodal(data.get("messages"), data.get("system")):
            data["model"] = self._multimodal_route()
        return data

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if not self._fired:
            self._fired = True
            # Proof marker (lands in the proxy log). Never log request bodies/keys.
            print("prism.hook.fired", file=sys.stderr, flush=True)
        return self.route(data)


# The instance litellm loads via `litellm_settings.callbacks: prism_hook.instance`.
instance = ModalityRouter()
