# autoresearch

Autonomous research agent inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Runs an iterative loop: **plan → search → read → synthesize → evaluate gaps → repeat**.

## Quick Start

```bash
# Install
uv sync

# Set your LLM API key (at least one required)
export ANTHROPIC_API_KEY=sk-...
# or: export OPENAI_API_KEY=sk-...
# or: export DEEPSEEK_API_KEY=sk-...

# Edit the research program
vim research_program.md

# Run
uv run research.py
```

## How It Works

```
┌─────────────────────────────────────┐
│         research_program.md         │  ← You write this
│  (topic, questions, scope)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│          research.py loop           │
│                                     │
│  1. Read program + current findings │
│  2. LLM → identify gaps            │
│  3. LLM → plan search strategy     │
│  4. Execute searches (3 tiers)      │
│  5. LLM → synthesize findings      │
│  6. LLM → evaluate coverage        │
│  7. Log progress → repeat          │
│                                     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│           findings.md               │  ← Agent writes this
│  (structured research output)       │
└─────────────────────────────────────┘
```

## Search Tiers

| Tier | Sources | API Key? |
|------|---------|----------|
| **1: Deep Research** | Metaso (秘塔搜索), Gemini, Perplexity | Yes |
| **2: Academic** | Semantic Scholar, arXiv | No |
| **3: Web** | DuckDuckGo | No |

The LLM decides which tiers to use each iteration based on topic and remaining gaps.

## Configuration

### LLM Model (env vars)

```bash
# Default: Claude Sonnet
export AUTORESEARCH_MODEL="claude-sonnet-4-20250514"

# Or use any litellm-supported model:
export AUTORESEARCH_MODEL="openai/gpt-4o"
export AUTORESEARCH_MODEL="deepseek/deepseek-chat"
export AUTORESEARCH_MODEL="ollama/llama3"
```

### Deep Research Services (optional)

```bash
export METASO_API_KEY=...        # 秘塔搜索
export PERPLEXITY_API_KEY=...    # Perplexity sonar
export GOOGLE_API_KEY=...        # Gemini
```

### CLI Options

```bash
uv run research.py --max-iterations 10 --target-coverage 0.9 --time-budget 30 --mode supervised
```

### Interaction Modes

Control how much human involvement each run has via `--mode`:

| Mode | Behavior | Use case |
|------|----------|----------|
| `auto` (default) | No pauses — fully autonomous | Overnight runs, trusted topics |
| `supervised` | Pauses at 2 checkpoints. Enter to continue, `q` to stop | First run on a new topic |
| `interactive` | Same pauses, but accepts typed guidance injected into the next step | Exploratory, unknown territory |

**Checkpoints:**
1. **After gap analysis** — Review planned searches before they execute. In interactive mode, type a query to add it.
2. **After evaluation** — See coverage score and remaining gaps. In interactive mode, type guidance for the next iteration.

## Files

| File | Purpose |
|------|---------|
| `research_program.md` | Your research brief (edit this) |
| `search.py` | Utilities: LLM, search, evaluation (don't edit) |
| `research.py` | The autonomous loop (don't edit) |
| `findings.md` | Research output (auto-generated) |
| `progress.tsv` | Iteration log (auto-generated) |
