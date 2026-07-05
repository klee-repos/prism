# Prism

**Route Claude Code to any model — pay coder prices for text, see images for free.**

Prism is a thin launcher that sits in front of the `claude` CLI and splits each request by
"wavelength": text and code go to a cheap coder model, images and files get rerouted to a
multimodal model — each to the right provider, automatically. You keep the **entire** Claude
Code experience (dynamic workflows, MCP, sub-agents, `--dangerously-skip-permissions`,
everything) — only the model underneath changes.

```
        prism  <your usual claude args>
          │
          ▼
   ┌─────────────────────────────┐
   │  local LiteLLM proxy         │   text / code  ─▶ coder      (e.g. GLM-5.2, cheap)
   │  (per-session, 127.0.0.1)    │   image / file ─▶ multimodal (e.g. Gemini, sees it)
   └─────────────────────────────┘
```

### Why

GLM's coder models are fast and cheap — but they can't see images, and Claude Code only speaks
the Anthropic protocol. Prism runs a local [LiteLLM](https://github.com/BerriAI/litellm) proxy
that translates the protocol *and* reroutes any request carrying an image or file to a model that
can actually read it — while your text stays on the cheap coder. One tool, full capability parity,
no code changes.

### Highlights

- **Passthrough-first.** `prism` forwards every flag to `claude` verbatim. Nothing is intercepted
  or rewritten except the model destination.
- **Provider-agnostic.** Swap backends by editing one YAML file — OpenRouter, z.ai, OpenAI,
  Gemini, Anthropic, Azure, Bedrock, any LiteLLM provider slug.
- **Loopback-only & per-session.** The proxy binds `127.0.0.1`, is torn down when `claude` exits,
  and never holds a port open between runs. No daemon, no keys on disk.
- **Honest about limits.** Video isn't reachable; PDFs are best-effort. Documented below, not
  hidden.

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
send one of those to `claude` as a prompt, use the escape hatch: `prism -- setup`.

| Command | What it does |
| --- | --- |
| `prism setup` | Provision `~/.prism/` and write a default config. Safe to re-run. |
| `prism status` | Show the active routes and a config hash (drift check). |
| `prism doctor` | Verify litellm version, config validity, provider keys, and `claude` on PATH. |
| `PRISM_BYPASS=1 prism …` | Run `claude` completely untouched — no proxy, no routing. |

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
  coder:      { provider: openrouter, model: z-ai/glm-5.2 }        # OpenRouter ids are vendor-prefixed
  background: { provider: openrouter, model: z-ai/glm-4.7-flash }
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
  coder: { provider: zai, model: glm-5.2 }
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

- The proxy binds **127.0.0.1 only** — never a routable interface. That loopback binding is the
  trust boundary: only processes already on your machine can reach it. Provider keys are referenced
  by env var, never written into config. `~/.prism/` is `0700`, its files `0600`.
- The proxy runs **per session** and is torn down when `claude` exits — no lingering daemon, no
  port held open between runs.
- **The proxy is intentionally unauthenticated** (no master key). Claude Code, when you're signed
  in to a Claude subscription, forwards *its own* OAuth token regardless of `ANTHROPIC_AUTH_TOKEN`
  — so a mandatory proxy key would reject that token and break the launch. Loopback-only +
  per-session lifetime is the boundary instead; anything that can reach the port can already run
  code as you. Don't rebind it off `127.0.0.1`.

## Develop

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## License

MIT © Kevin Lee
