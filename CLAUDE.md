# CLAUDE.md — autoresearch

## Project Overview
Autonomous research agent. Iterative loop: identify gaps → search (3 tiers) → synthesize → evaluate coverage → repeat. Inspired by Karpathy's autoresearch.

## File Roles
- `research.py` — Main loop, CLI, interaction modes. **Editable.**
- `search.py` — Fixed infrastructure: LLM calls, search APIs, evaluation. **Do not edit per-project.**
- `research_program.md` — Research brief written by the user. Hot-reloaded each iteration.
- `findings.md` — Auto-generated structured research output.
- `progress.tsv` — Iteration log (coverage, sources, duration).

## Running
```bash
uv sync                          # Install deps
uv run research.py               # Default: auto mode
uv run research.py --mode supervised --max-iterations 10
uv run research.py --mode interactive --target-coverage 0.9 --time-budget 30
```

## Environment Variables
```bash
# Required (at least one LLM key)
ANTHROPIC_API_KEY=sk-...
# or OPENAI_API_KEY, DEEPSEEK_API_KEY

# Optional: change model
AUTORESEARCH_MODEL="claude-sonnet-4-20250514"  # default

# Optional: Tier 1 deep research sources
METASO_API_KEY=...
PERPLEXITY_API_KEY=...
GOOGLE_API_KEY=...
```

## Code Conventions
- Python 3.10+, `from __future__ import annotations` in every file
- `httpx` for all HTTP calls (no `requests`)
- Config via env vars, not config files
- No external LLM library — manual provider routing in `search.py`
- `dataclass` for structured data, `Enum` for fixed choices
- Print-based logging (no logging module)
- Error handling: catch specific exceptions, print context, continue loop

## Interaction Modes (`--mode`)
- `auto` (default) — No pauses. Fully autonomous.
- `supervised` — Pauses at 2 checkpoints. Enter to continue, `q` to stop.
- `interactive` — Same pauses + accepts typed guidance injected into next step.

## Testing
No test suite yet. Verify manually:
1. `--mode auto` should behave identically to pre-mode code
2. `--mode supervised` should pause twice per iteration
3. `--mode interactive` should accept and use typed guidance
