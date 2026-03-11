# Plan: Progressive Disclosure Knowledge Store

## Problem
Each iteration of the research loop feeds the entire `findings.md` + all search results into every LLM call. As research accumulates, context grows unboundedly — causing slower responses, higher costs, and eventual context window overflow.

## Solution
A hierarchical knowledge store with **3-layer progressive disclosure** — like a file system where the directory listing is always cheap, but file contents are only loaded when needed. This mirrors how Claude Code's skill system works: skills are *listed by name* (index layer) but only *loaded when invoked* (detail layer).

## Design Principles

### Progressive Disclosure (Core Idea)
Each LLM call gets a **token budget** (default: 8k context tokens). The system:
1. **Always loads the index** (~500 tokens) — the "directory listing"
2. **Selects relevant summaries** — based on keyword overlap × (1 - coverage_score), no LLM needed
3. **Loads summaries in relevance order** until budget is ~60% full
4. **Leaves 40%** for the LLM to reason and respond

### Incremental Delta Merge (Not Full Rewrite)
Instead of rewriting all findings each iteration, the system:
- Identifies which topic a new search result belongs to
- Loads only that topic's existing summary
- Merges the new results into the existing summary (delta)
- Tracks `last_synthesized_source_id` so it knows exactly what's new

### Staleness Tracking
Each topic tracks `unsynthesized_sources` — sources that were found but not yet merged into the summary. This prevents the anti-pattern of searching for things the system already has data on but hasn't processed yet.

### Cross-Topic Relations
Topics aren't isolated. A lightweight edge list (`relations[]`) lets the system pull in related summaries at lower priority, giving the LLM cross-cutting awareness without loading everything.

## How Each LLM Call Changes

| Step | Now (full context) | With progressive disclosure |
|---|---|---|
| `identify_gaps()` | Full `findings.md` | **index.json** + only summaries for low-coverage topics |
| `extract_findings()` | All search results + all findings | New search results + **only the relevant summary** being updated |
| `evaluate_coverage()` | Full `findings.md` | **index.json** (scores already there!) + spot-check 1-2 summaries |

## Architecture

```
research_store/
├── index.json              ← Always loaded (~500 tokens)
│   ├── topics[]            ← {id, title, keywords, coverage, source_count, unsynthesized_sources, stale}
│   ├── questions[]         ← {id, text, status, related_topics}
│   ├── relations[]         ← {from, to, type}
│   └── meta               ← {total_sources, avg_coverage, iterations}
│
├── summaries/              ← Loaded on-demand per topic (~1-2k each)
│   ├── architecture_patterns.md
│   ├── framework_comparisons.md
│   └── safety_challenges.md
│   (each file has frontmatter: last_synthesized_source_id, source_count)
│
├── details/
│   ├── sources/            ← Individual source records (immutable once written)
│   │   ├── src_001.json    ← {id, title, url, snippet, full_text, topic_ids, iteration}
│   │   └── src_002.json
│   └── raw_results/        ← Raw search API responses per iteration
│       └── iter_003_search_001.json
│
└── findings.md             ← Final output, regenerated from summaries on demand
```

## Design Decisions

1. **Storage format**: JSON for index + structured data (machine-readable), Markdown for summaries (human-readable + LLM-friendly)
2. **Scope**: New module `knowledge_store.py` that `research.py` imports — clean separation of concerns
3. **Migration**: Auto-detect existing `findings.md` and bootstrap the store from it via LLM extraction
4. **Relevance routing**: Keyword overlap + coverage weighting — no embeddings, no extra LLM calls
5. **Token estimation**: `len(text) // 4` as rough approximation (good enough for budgeting)

---

## Stage 1: Knowledge Store Core
**Goal**: `knowledge_store.py` module with index CRUD, summary read/write, source management
**Success Criteria**:
- `pytest tests/test_knowledge_store.py` passes
- Can create store, add topics/sources, read/write summaries
- Index serializes/deserializes correctly
- Store bootstraps from existing `findings.md` via LLM extraction
**Files**:
- `knowledge_store.py` (create)
- `tests/test_knowledge_store.py` (create)
**Depends on**: —
**Context**:
- Data types in `search.py:SearchResult, EvaluationResult`
- Current file I/O pattern: `read_file()` / `write_file()` in `research.py`

---

## Stage 2: Integration with Research Loop
**Goal**: Rewire `research.py` to use the knowledge store instead of raw `findings.md`
**Success Criteria**:
- `pytest tests/test_research.py` passes (existing tests updated)
- `identify_gaps()` receives index + relevant summaries instead of full findings
- `extract_findings()` does incremental delta merge per topic instead of full rewrite
- `evaluate_coverage()` uses index scores + spot-checks instead of full findings
- `findings.md` regenerated from summaries at end of each iteration
**Files**:
- `research.py` (modify)
- `tests/test_research.py` (modify)
**Depends on**: Stage 1
**Context**:
- `identify_gaps()` at research.py — currently takes full findings text
- `extract_findings()` at research.py — currently takes all search results + findings
- `evaluate_coverage()` at research.py — currently takes full findings + program

---

## Stage 3: Smart Routing & Staleness
**Goal**: Lightweight relevance routing (no LLM), token budgeting, staleness tracking
**Success Criteria**:
- `pytest tests/test_knowledge_store.py` passes (new tests for routing)
- `select_relevant_summaries()` picks topics by keyword overlap × (1 - coverage)
- Token budget enforced: summaries loaded in relevance order until budget hit
- Stale topics (unsynthesized_sources > 0) prioritized for synthesis over new searches
- Cross-topic relations used to pull related summaries at lower priority
**Files**:
- `knowledge_store.py` (modify)
- `tests/test_knowledge_store.py` (modify)
- `research.py` (modify) — use routing in gap analysis
**Depends on**: Stage 2
**Context**:
- Keyword extraction: simple tokenize + stopword removal, no embeddings
- Token budget default: 8000 tokens, configurable
- Token estimation: `len(text) // 4` as rough approximation

---

## Status

| Stage | Status |
|-------|--------|
| 1. Knowledge Store Core | **Complete** (35 tests) |
| 2. Integration with Research Loop | **Complete** (12 new tests) |
| 3. Smart Routing & Staleness | **Complete** (9 new tests) |

**Total: 147 tests passing (was 91 before this feature)**
