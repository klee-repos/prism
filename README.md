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
- **Honest about limits.** Video isn't supported; PDFs are best-effort. Web search
  needs a one-time provider key (below). Documented, not hidden.

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

Only four words belong to Prism itself: `setup`, `status`, `doctor`, `profile`. If you ever want to
send one of those words to `claude` as a prompt instead, write `prism -- setup`.

| Command | What it does |
| --- | --- |
| `prism setup` | Create `~/.prism/` and write a default config. Safe to run again. |
| `prism status` | Show the active profile and which models your routes point to, plus a config hash. |
| `prism doctor` | Check that litellm, your config, your keys, and `claude` are all good. |
| `prism profile` | List your profiles (marks the active one). `prism profile <name>` switches. |

## Pick your models (edit one file)

Everything lives in `~/.prism/config.yaml`. A **profile** is a named set of routes;
`active_profile` picks the one in use. The default config ships two, so you can switch with a
single word — either edit `active_profile:` by hand, or run `prism profile <name>`:

```yaml
schema_version: 1
active_profile: glm          # ← the one word that switches everything (glm | k3)

providers:
  openrouter: { type: openrouter, api_key_env: OPENROUTER_API_KEY }

profiles:
  glm:   # GLM for text/code + Google Gemini for images (two specialized models)
    coder:      { provider: openrouter, model: z-ai/glm-5.2 }
    background: { provider: openrouter, model: z-ai/glm-4.7-flash }
    multimodal: { provider: openrouter, model: google/gemini-2.5-flash }
  k3:    # Kimi K3 — one model for BOTH text and images
    coder:      { provider: openrouter, model: moonshotai/kimi-k3 }
    background: { provider: openrouter, model: z-ai/glm-4.7-flash }   # stays cheap
    multimodal: { provider: openrouter, model: moonshotai/kimi-k3 }

mapping: { opus: coder, sonnet: coder, haiku: background }
```

### Try Kimi K3, switch back to GLM + Gemini

Kimi K3 handles both text and images, so its profile points **both** the `coder` and
`multimodal` routes at it — no separate vision model needed:

```sh
prism profile k3     # use Kimi K3 for text and images
prism profile glm    # back to GLM (text) + Gemini (vision)
prism profile        # list profiles; the active one is marked with *
```

Switching rewrites only the `active_profile:` line (your comments and formatting are kept).
Two notes on the K3 profile: it keeps the cheap `background` route on GLM-flash (K3 is pricier),
and its routes carry no `quantizations` pin because K3's OpenRouter endpoint is served at int4 —
pinning FP8-or-better (like the GLM route does) would leave it with no provider to run on.

**Use GLM directly from z.ai** (no middleman) — add a provider and point a profile's route at it:

```yaml
providers:
  zai: { type: zai, api_key_env: ZAI_API_KEY }
profiles:
  glm:
    coder: { provider: zai, model: glm-5.2 }
```

Other `type` values: `openai`, `gemini`, `anthropic`, `azure`, `bedrock`, and more. Direct
providers use plain model ids; OpenRouter ids carry a vendor prefix (like `z-ai/` or `google/`).
Older configs that use a single top-level `routes:` block (no profiles) still work unchanged.

## How routing works

Before each request is sent, Prism looks at it: if it contains an image or file (even one nested
inside a tool's output), that request goes to the `multimodal` model; otherwise it stays on
whichever model Claude Code picked. You only pay the vision-model price on requests that actually
have a picture or file.

## Web search

Claude Code's built-in WebSearch is a hosted Anthropic tool — Anthropic's servers run it,
not your machine. GLM, Gemini, and the other cheap backends can't execute it, so by default
Prism strips it (the model simply won't call WebSearch; it can still fetch URLs you name
with WebFetch).

To keep WebSearch working on the cheap coder model, add a `search` section to
`~/.prism/config.yaml`:

```yaml
search:
  provider: firecrawl              # litellm search slug (firecrawl, tavily, perplexity,
  api_key_env: FIRECRAWL_API_KEY  #   serper, exa_ai, brave, … keyless: duckduckgo)
```

Then export the key:

```sh
export FIRECRAWL_API_KEY=fc-...
```

With a provider wired, Prism converts WebSearch into a plain function tool the model can
call, and litellm runs the search via your provider and feeds the results back to the
model — so you stay on the cheap model instead of paying for a search-capable one. Get a
key from Firecrawl's dashboard (free tier available; see firecrawl.dev). Keyless
`duckduckgo` works but returns very few results — fine for trivial lookups, not for real
research.

## Develop

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```
