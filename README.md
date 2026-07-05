# Prism

**Route Claude Code to any model.** Prism sits in front of the `claude` CLI and splits each
request by "wavelength": text and coding go to a cheap coder model, images and files go to a
multimodal model — each to the right place, any provider. You keep the entire Claude Code
experience (dynamic workflows, MCP, `--dangerously-skip-permissions`, everything); only the model
underneath changes.

```
        prism  <your usual claude args>
          │
          ▼
   ┌─────────────────────────────┐
   │  local LiteLLM proxy         │   text/coding ─▶  coder      (e.g. GLM-4.6, cheap)
   │  (per-session, 127.0.0.1)    │   image / file ─▶ multimodal (e.g. Gemini, sees it)
   └─────────────────────────────┘
```

Why it exists: GLM's coder models can't see images, and Claude Code can only talk the Anthropic
protocol. Prism runs a local [LiteLLM](https://github.com/BerriAI/litellm) proxy that translates
the protocol *and* reroutes any request carrying an image/file to a model that can actually read it —
while your text stays on the cheap coder. One tool, full capability parity.

## Install

```sh
pipx install prism-cc      # installs the `prism` command
prism setup                # writes ~/.prism/config.yaml (OpenRouter defaults)
export OPENROUTER_API_KEY=sk-or-...
prism                      # === claude, now routed
```

Requires Python 3.10–3.13 and the [`claude`](https://claude.com/claude-code) CLI on your PATH.

## Use it exactly like `claude`

`prism` forwards **every** argument to `claude` unchanged:

```sh
prism -p "explain this repo"
prism --model opus --dangerously-skip-permissions
prism --resume
```

Only three words are reserved for Prism itself: `setup`, `status`, `doctor`. If you ever need to
send one of those to claude as a prompt, use the escape hatch: `prism -- setup`.

Set `PRISM_BYPASS=1` to run `claude` completely untouched (no proxy, no routing).

## Configure any provider (no code change)

Everything lives in `~/.prism/config.yaml`. Swap providers by editing it. The one field that picks
the backend is `providers.<name>.type` — a LiteLLM provider slug.

```yaml
schema_version: 1
providers:
  openrouter:
    type: openrouter
    api_key_env: OPENROUTER_API_KEY
routes:
  coder:      { provider: openrouter, model: z-ai/glm-4.6 }        # OpenRouter ids are vendor-prefixed
  background: { provider: openrouter, model: z-ai/glm-4.6 }
  multimodal: { provider: openrouter, model: google/gemini-2.5-flash }
mapping:
  opus: coder
  sonnet: coder
  haiku: background
```

**GLM direct from z.ai** (native provider — auto base + key):

```yaml
providers:
  zai: { type: zai, api_key_env: ZAI_API_KEY }
routes:
  coder: { provider: zai, model: glm-4.6 }
```

Other `type` values: `openai`, `gemini`, `anthropic`, `azure`, `bedrock`, … (any LiteLLM provider
slug). Direct providers use bare model ids; OpenRouter ids carry a vendor segment.

## How routing works

A LiteLLM pre-call hook inspects each request: if it carries an image or file (at any depth —
including an image returned inside a tool result), it's routed to `multimodal`; otherwise it stays
on whatever tier Claude Code chose (`coder` / `background`). You only pay multimodal rates on the
requests that actually contain a picture or file.

## Known limits (honest)

- **Video** isn't reachable — the Anthropic Messages API (what Claude Code speaks) has no video
  content block, so there's no way to send video regardless of model.
- **PDFs / documents** are best-effort: LiteLLM's token counter chokes on some Anthropic `document`
  blocks. Route documents to a provider that ingests them natively (e.g. Gemini) and test your case.
- **Anthropic Files-API `file_id`** references can't work on non-Anthropic backends.

## Security

- The proxy binds **127.0.0.1 only** and requires a generated master key (mandatory — no unauthenticated
  mode). Keys are referenced by env var, never written into config. `~/.prism/` is `0700`, its files `0600`.
- The proxy runs **per session** and is torn down when `claude` exits — no lingering daemon.
- Note: any process `claude` spawns (MCP servers, hooks, the Bash tool) inherits the proxy auth token,
  same as with a real Anthropic key. Keep it loopback-only.

## Develop

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

MIT licensed.
