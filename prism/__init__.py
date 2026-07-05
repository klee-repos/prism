"""Prism — route Claude Code to any model.

Text/coding stays on a cheap coder model; images and files are rerouted to a
multimodal model. A local LiteLLM proxy does the translation and per-request
routing; Prism supervises it and launches `claude` with every flag passed through.
"""

__version__ = "0.1.0"
