# Prism

**Prism lets you run Claude Code with cheaper, third-party AI models instead of
Anthropic's — without losing any of Claude Code's features.**

Normally, Claude Code only talks to Anthropic's servers. Prism steps in between: it sends your
plain-text coding questions to a cheap model (like GLM), and automatically sends anything with a
picture or file attached to a model that can actually see images (like Gemini). You get the full
Claude Code experience — tools, agents, MCP, every flag — just running on models that cost less.

```
        prism  <your usual claude args>
          │
          ▼
   ┌─────────────────────────────┐
   │  small local proxy           │   text / code  ─▶ cheap model    (e.g. GLM-5.2)
   │  (runs only on your machine) │   image / file ─▶ vision model   (e.g. Gemini)
   └─────────────────────────────┘
```

### Why this exists

GLM's coding models are fast and cheap, but they can't see images. Claude Code can't talk to them
directly either. Prism fixes both: it translates between Claude Code and the cheaper models, and
whenever a request has an image or file, it sends that part to a model with vision. Your text stays
on the cheap model. One tool, no code changes to your projects.

### Highlights

- **Works just like `claude`.** Every flag you pass to `prism` is passed straight through to
  Claude Code. Nothing is intercepted or rewritten except which model answers.
- **Any provider.** Swap models by editing one text file — OpenRouter, z.ai, OpenAI, Gemini,
  Anthropic, Azure, Bedrock, and more.
- **Runs only on your machine.** The proxy talks to `127.0.0.1` (your loopback address), starts
  when you start `prism`, and stops when you stop. No background daemon, no keys saved to disk.
- **Honest about limits.** Video isn't supported; PDFs are best-effort. Documented below, not
  hidden.

## Install

```sh
pipx install prism-cc      # installs the `prism` command
prism setup                # writes ~/.prism/config.yaml with sensible defaults
export OPENROUTER_API_KEY=sk-or-...
prism                      # this runs claude, now routed
```

You need Python 3.10–3.13 and the [`claude`](https://claude.com/claude-code) CLI installed.

## Use it exactly like `claude`

`prism` takes the same arguments as `claude`:

```sh
prism -p "explain this repo"
prism --model opus --dangerously-skip-permissions
prism --resume
```

Only three words belong to Prism itself: `setup`, `status`, `doctor`. If you ever want to send one
of those words to `claude` as a prompt instead, write `prism -- setup`.

| Command | What it does |
| --- | --- |
| `prism setup` | Create `~/.prism/` and write a default config. Safe to run again. |
| `prism status` | Show which models your routes point to, plus a config hash. |
| `prism doctor` | Check that litellm, your config, your keys, and `claude` are all good. |

## Pick your models (edit one file)

Everything lives in `~/.prism/config.yaml`. To change a model, just edit that file. The `type`
field under each provider is what decides which service to call.

```yaml
schema_version: 1
providers:
  openrouter:
    type: openrouter
    api_key_env: OPENROUTER_API_KEY
routes:
  coder:      { provider: openrouter, model: z-ai/glm-5.2 }        # text and code
  background: { provider: openrouter, model: z-ai/glm-4.7-flash }   # cheap background tasks
  multimodal: { provider: openrouter, model: google/gemini-2.5-flash }  # images and files
mapping:
  opus: coder
  sonnet: coder
  haiku: background
```

**Use GLM directly from z.ai** (no middleman):

```yaml
providers:
  zai: { type: zai, api_key_env: ZAI_API_KEY }
routes:
  coder: { provider: zai, model: glm-5.2 }
```

Other `type` values: `openai`, `gemini`, `anthropic`, `azure`, `bedrock`, and more. Direct
providers use plain model ids; OpenRouter ids carry a vendor prefix (like `z-ai/` or `google/`).

## How routing works

Before each request is sent, Prism looks at it: if it contains an image or file (even one nested
inside a tool's output), that request goes to the `multimodal` model; otherwise it stays on
whichever model Claude Code picked. You only pay the vision-model price on requests that actually
have a picture or file.

## Develop

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```
