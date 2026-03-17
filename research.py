"""
research.py — Autonomous research loop.

Usage: uv run research.py [--max-iterations N] [--target-coverage 0.8] [--time-budget 60] [--mode auto|supervised|interactive]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from knowledge_store import KnowledgeStore, Topic
from search import (
    EvaluationResult,
    IterationLog,
    SearchResult,
    execute_searches,
    init_progress_log,
    llm_call,
    llm_json,
    log_iteration,
)

# ── Interaction Modes ─────────────────────────────────────────────────────


class InteractionMode(Enum):
    AUTO = "auto"
    SUPERVISED = "supervised"
    INTERACTIVE = "interactive"


@dataclass
class CheckpointResult:
    approved: bool  # False = stop the loop
    guidance: str = ""  # Typed guidance (interactive mode only)


def prompt_checkpoint(
    mode: InteractionMode,
    title: str,
    summary_lines: list[str],
) -> CheckpointResult:
    """Pause for human input at a checkpoint. Returns immediately in AUTO mode."""
    if mode == InteractionMode.AUTO:
        return CheckpointResult(approved=True)

    print(f"\n{'▶' * 3} CHECKPOINT: {title}")
    for line in summary_lines:
        print(f"  {line}")

    if mode == InteractionMode.SUPERVISED:
        print("\n  [Enter] continue | [q] stop")
        try:
            response = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return CheckpointResult(approved=False)
        return CheckpointResult(approved=response != "q")

    # INTERACTIVE mode
    print("\n  [Enter] continue | [q] stop | type guidance to inject")
    try:
        response = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return CheckpointResult(approved=False)
    if response.lower() == "q":
        return CheckpointResult(approved=False)
    return CheckpointResult(approved=True, guidance=response)


# ── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_MAX_ITERATIONS = 20
DEFAULT_TARGET_COVERAGE = 0.8
DEFAULT_TIME_BUDGET_MIN = 60
RESEARCH_PROGRAM_PATH = "research_program.md"
FINDINGS_PATH = "findings.md"
PROGRESS_PATH = "progress.tsv"

# ── Helper Functions ───────────────────────────────────────────────────────


def read_file(path: str) -> str:
    """Read a text file, return empty string if missing."""
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def write_file(path: str, content: str) -> None:
    """Write content to a text file."""
    Path(path).write_text(content, encoding="utf-8")


def identify_gaps(research_program: str, store: KnowledgeStore, human_guidance: str = "") -> dict:
    """Ask LLM to identify what questions remain unanswered.

    Uses the knowledge store index context and low-coverage summaries
    instead of the full findings text.
    """
    guidance_block = ""
    if human_guidance:
        guidance_block = f"""
## Human Guidance
The researcher has provided the following direction for this iteration:
{human_guidance}
Prioritize this guidance when identifying gaps and planning searches.
"""

    # Use budget-aware context building with relevance routing
    context = store.get_context_with_budget(
        gap_queries=[human_guidance] if human_guidance else [],
        token_budget=8000,
    )

    # Check if store has topics for first-iteration detection
    index = store.load_index()

    # Build the context block
    if not index.topics:
        findings_block = "(No findings yet — this is the first iteration)"
    else:
        findings_block = context

    # Build existing topic list for topic_id assignment
    existing_topics_list = ""
    if index.topics:
        topic_items = [f'  - "{t.id}": {t.title}' for t in index.topics]
        existing_topics_list = "\nExisting topic IDs:\n" + "\n".join(topic_items) + "\n"

    prompt = f"""You are analyzing a research project to identify knowledge gaps.

## Research Program
{research_program}

## Current Knowledge State
{findings_block}
{guidance_block}
## Task
Identify what questions from the research program are NOT yet answered.
Plan a search strategy to fill these gaps.
{existing_topics_list}
Respond with JSON only:
{{
  "gaps": ["list of specific unanswered questions or topics"],
  "search_plan": [
    {{
      "query": "specific search query",
      "sources": ["list of sources to use: semantic_scholar, arxiv, duckduckgo, metaso, perplexity, gemini"],
      "rationale": "why this query and these sources",
      "topic_id": "existing_topic_id_or_new"
    }}
  ],
  "new_topics": [
    {{"id": "slug", "title": "Title", "keywords": ["kw1", "kw2"]}}
  ]
}}

Guidelines for source selection:
- For academic/scientific topics → prefer "semantic_scholar" and "arxiv"
- For recent developments or practical topics → prefer "duckduckgo"
- For comprehensive synthesis → prefer "metaso", "perplexity", or "gemini" (if API keys available)
- Always include at least one freely available source (semantic_scholar, arxiv, or duckduckgo)
- Plan 2-5 searches per iteration, focused on the biggest gaps
- For topic_id: use an existing topic ID if the query relates to one, or "new" if it's a brand new topic
- If topic_id is "new", add a corresponding entry to "new_topics" """

    return llm_json(prompt)


def extract_topic_findings(
    search_results: list[SearchResult],
    research_program: str,
    store: KnowledgeStore,
    topic_id: str,
) -> str:
    """Ask LLM to merge new search results into an existing topic summary (delta merge)."""
    # Format search results for the LLM
    formatted = []
    for i, r in enumerate(search_results, 1):
        entry = f"### Source {i}: {r.title}\n"
        entry += f"- **URL**: {r.url}\n"
        if r.authors:
            entry += f"- **Authors**: {r.authors}\n"
        if r.year:
            entry += f"- **Year**: {r.year}\n"
        entry += f"- **Source type**: {r.source}\n"
        content = r.full_text if r.full_text else r.snippet
        if content:
            entry += f"- **Content**: {content[:2000]}\n"
        formatted.append(entry)

    sources_text = "\n".join(formatted)

    # Get existing summary for this topic
    existing_summary = store.read_summary(topic_id)
    topic = store.get_topic(topic_id)
    topic_title = topic.title if topic else topic_id

    prompt = f"""You are merging new research results into an existing topic summary.

## Research Program
{research_program}

## Topic: {topic_title} ({topic_id})

## Existing Summary
{existing_summary if existing_summary.strip() else "(Empty — this is the first synthesis for this topic)"}

## New Search Results
{sources_text}

## Task
Merge the new search results into the existing summary for this topic.

Requirements:
1. KEEP all existing findings that are still relevant
2. ADD new information from the search results
3. RESOLVE any contradictions between old and new findings
4. Use inline citations like [Author, Year] or [Source Name]
5. Be specific and evidence-based, not vague
6. Output ONLY the updated summary content as markdown (no frontmatter, no topic title header)
7. This is a DELTA MERGE — integrate new info, don't rewrite from scratch"""

    return llm_call(prompt, max_tokens=8192)


def evaluate_coverage(research_program: str, store: KnowledgeStore) -> EvaluationResult:
    """Use LLM-as-judge to evaluate coverage using the knowledge store index.

    Uses the index context primarily, plus spot-checks 1-2 low-coverage summaries.
    """
    index_context = store.get_context_for_gaps()

    # Spot-check: load summaries for the 1-2 lowest-coverage topics
    index = store.load_index()
    sorted_topics = sorted(index.topics, key=lambda t: t.coverage)
    spot_check = ""
    for t in sorted_topics[:2]:
        summary = store.read_summary(t.id)
        if summary.strip():
            spot_check += f"\n### {t.title} ({t.id}) — coverage: {t.coverage:.2f}\n{summary}\n"

    findings_block = index_context
    if spot_check.strip():
        findings_block += f"\n## Spot-Check Summaries (lowest coverage){spot_check}"

    prompt = f"""Evaluate how well the current research findings answer the research program.

## Research Program
{research_program}

## Current Knowledge State
{findings_block}

## Instructions
Score coverage from 0.0 (nothing answered) to 1.0 (fully answered).
Be strict — only score high if findings are thorough with supporting evidence.

Respond with JSON only:
{{
  "coverage_score": <float 0.0-1.0>,
  "questions_answered": ["list of questions that are well-answered"],
  "questions_remaining": ["list of questions still unanswered or poorly answered"],
  "gaps": ["specific knowledge gaps identified"],
  "suggested_queries": ["2-4 search queries to fill the gaps"],
  "topic_scores": {{
    "topic_id": <float 0.0-1.0>
  }}
}}"""

    try:
        data = llm_json(prompt)
        # Update per-topic coverage scores in the store
        topic_scores = data.get("topic_scores", {})
        for tid, score in topic_scores.items():
            store.update_topic_coverage(tid, float(score))

        return EvaluationResult(
            coverage_score=float(data.get("coverage_score", 0.0)),
            questions_answered=data.get("questions_answered", []),
            questions_remaining=data.get("questions_remaining", []),
            gaps=data.get("gaps", []),
            suggested_queries=data.get("suggested_queries", []),
        )
    except Exception as e:
        print(f"  [evaluate] Error: {e}")
        return EvaluationResult(coverage_score=0.0, gaps=[str(e)])


# ── Loop Quality: Stopping, Synthesis, Search History, Quality Gates ───────


@dataclass
class StoppingSignal:
    should_stop: bool
    reason: str
    confidence: float  # 0.0-1.0

def should_stop_research(
    recent_scores: list[float],
    iteration: int,
    target_coverage: float,
    max_iterations: int,
) -> StoppingSignal:
    """Multi-signal stopping heuristic beyond simple threshold."""
    if iteration >= max_iterations:
        return StoppingSignal(True, "max_iterations_reached", 1.0)

    if len(recent_scores) < 3:
        return StoppingSignal(False, "too_few_iterations", 0.0)

    # Signal 1: Target coverage with stability (last 3 scores all above target)
    last_3 = recent_scores[-3:]
    if all(s >= target_coverage for s in last_3):
        return StoppingSignal(True, "coverage_stable_at_target", 0.95)

    # Signal 2: Persistent oscillation (3+ direction changes in last 5 scores)
    if len(recent_scores) >= 5:
        last_5 = recent_scores[-5:]
        direction_changes = sum(
            1 for i in range(1, len(last_5) - 1)
            if (last_5[i] - last_5[i-1]) * (last_5[i+1] - last_5[i]) < 0
        )
        if direction_changes >= 3:
            avg = sum(last_5) / len(last_5)
            return StoppingSignal(True, f"oscillating_around_{avg:.2f}", 0.7)

    # Signal 3: Diminishing returns (improvement < 0.02 over last 4 iterations)
    if len(recent_scores) >= 4:
        recent_improvement = recent_scores[-1] - recent_scores[-4]
        if abs(recent_improvement) < 0.02 and iteration > 5:
            return StoppingSignal(True, "diminishing_returns", 0.6)

    # Signal 4: Single score at target
    if recent_scores[-1] >= target_coverage:
        return StoppingSignal(True, "target_reached", 0.9)

    return StoppingSignal(False, "continue", 0.0)


def cross_topic_synthesis(store, research_program: str) -> str:
    """Analyze connections and contradictions across all topics.

    Returns a cross-topic analysis string, or empty if < 2 topics.
    """
    index = store.load_index()
    if len(index.topics) < 2:
        return ""

    topic_summaries = []
    for t in index.topics:
        summary = store.read_summary(t.id)
        if summary.strip():
            topic_summaries.append(f"### {t.title} (coverage: {t.coverage:.2f})\n{summary[:1500]}")

    if len(topic_summaries) < 2:
        return ""

    prompt = f"""You are analyzing research findings across multiple topics to identify connections.

## Research Program
{research_program}

## Topic Summaries
{chr(10).join(topic_summaries)}

## Task
Analyze the topics above and identify:
1. SHARED CONCEPTS: Core principles or findings that appear across multiple topics
2. CROSS-TOPIC CONTRADICTIONS: Do any topics contradict each other?
3. EMERGENT INSIGHTS: What conclusions emerge from viewing topics together that aren't obvious from any single topic?
4. DEPENDENCY MAP: Which topics support, inform, or build on others?

Be specific and cite which topics you're referencing. Output as concise markdown.
Keep your response under 500 words — focus on the most important connections only."""

    try:
        result = llm_call(prompt, max_tokens=2048)
        return result
    except Exception as e:
        print(f"  [cross-topic] Error: {e}")
        return ""


_search_history: list[dict] = []

def record_search_result(query: str, sources: list[str], result_count: int, iteration: int) -> None:
    """Record search execution for future reference."""
    _search_history.append({
        "query": query,
        "sources": sources,
        "result_count": result_count,
        "iteration": iteration,
    })

def get_failed_search_summary() -> str:
    """Return a summary of searches that yielded 0 results, for prompt injection."""
    failed = [h for h in _search_history if h["result_count"] == 0]
    if not failed:
        return ""
    lines = [f"- \"{h['query']}\" via {', '.join(h['sources'])}" for h in failed[-5:]]
    return "These previous searches returned NO results — avoid similar queries:\n" + "\n".join(lines)

def is_similar_to_failed(query: str, threshold: float = 0.7) -> bool:
    """Check if a query is too similar to a previously failed query."""
    failed_queries = [h["query"] for h in _search_history if h["result_count"] == 0]
    query_words = set(query.lower().split())
    for fq in failed_queries:
        fq_words = set(fq.lower().split())
        if not query_words or not fq_words:
            continue
        overlap = len(query_words & fq_words) / max(len(query_words), len(fq_words))
        if overlap >= threshold:
            return True
    return False


def check_synthesis_quality(old_summary: str, new_summary: str) -> tuple[bool, str]:
    """Quick heuristic check: did synthesis improve or degrade the summary?

    Returns: (is_ok, reason)
    """
    if not old_summary.strip():
        return True, "first_synthesis"

    # Check 1: New summary shouldn't be dramatically shorter
    old_len = len(old_summary)
    new_len = len(new_summary)
    if new_len < old_len * 0.5 and old_len > 200:
        return False, f"summary_shrunk ({new_len} vs {old_len} chars)"

    # Check 2: Citation count shouldn't decrease
    old_cites = len(re.findall(r'\[([^\]]+)\]', old_summary))
    new_cites = len(re.findall(r'\[([^\]]+)\]', new_summary))
    if new_cites < old_cites * 0.5 and old_cites >= 2:
        return False, f"citations_lost ({new_cites} vs {old_cites})"

    return True, "ok"


def verify_citations_against_sources(summary: str, source_titles: list[str]) -> dict:
    """Lightweight check: do citations in the summary plausibly match provided sources?

    Returns: {"total": int, "plausible": int, "suspicious": list[str]}
    """
    citations = re.findall(r'\[([^\]]+)\]', summary)
    # Filter out non-citation brackets
    citations = [c for c in citations if any(ch.isalpha() for ch in c)
                 and not c.startswith("http") and c not in ("CONFLICT",)]

    source_words = set()
    for title in source_titles:
        source_words.update(w.lower() for w in title.split() if len(w) > 3)

    plausible = 0
    suspicious = []
    for cite in citations:
        cite_words = set(w.lower() for w in cite.split() if len(w) > 2)
        if cite_words & source_words:
            plausible += 1
        else:
            suspicious.append(cite)

    return {"total": len(citations), "plausible": plausible, "suspicious": suspicious}


# ── Main Loop ──────────────────────────────────────────────────────────────


def run_research_loop(
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
    time_budget_min: float = DEFAULT_TIME_BUDGET_MIN,
    mode: InteractionMode = InteractionMode.AUTO,
) -> None:
    """Run the autonomous research loop."""

    # Validate research program exists
    research_program = read_file(RESEARCH_PROGRAM_PATH)
    if not research_program.strip():
        print(f"Error: {RESEARCH_PROGRAM_PATH} is empty or missing.")
        print("Create a research_program.md with your research questions and scope.")
        sys.exit(1)

    print("=" * 60)
    print("AUTORESEARCH — Autonomous Research Agent")
    print("=" * 60)
    print(f"Model: {os.environ.get('AUTORESEARCH_MODEL', 'claude-sonnet-4-20250514')}")
    print(f"Max iterations: {max_iterations}")
    print(f"Target coverage: {target_coverage}")
    print(f"Time budget: {time_budget_min} min")
    print(f"Mode: {mode.value}")
    print("=" * 60)

    # Initialize progress log
    init_progress_log(PROGRESS_PATH)

    # Initialize knowledge store
    store = KnowledgeStore()

    # If findings.md exists but store doesn't, bootstrap
    findings = read_file(FINDINGS_PATH)
    if findings.strip() and not store.exists():
        print("  Bootstrapping knowledge store from existing findings...")
        store.bootstrap_from_findings(findings, research_program, llm_json)

    start_time = time.time()
    iteration = 0
    human_guidance = ""  # Carries guidance from one checkpoint to the next iteration
    recent_scores: list[float] = []  # C3: Track scores for stopping analysis

    while iteration < max_iterations:
        iteration += 1
        iter_start = time.time()
        elapsed_min = (iter_start - start_time) / 60

        # Check time budget
        if elapsed_min > time_budget_min:
            print(f"\n⏰ Time budget exceeded ({elapsed_min:.1f} min). Stopping.")
            break

        print(f"\n{'─' * 60}")
        print(f"Iteration {iteration}/{max_iterations} | Elapsed: {elapsed_min:.1f} min")
        print(f"{'─' * 60}")

        # Step 1: Read research program
        research_program = read_file(RESEARCH_PROGRAM_PATH)
        if not research_program.strip():
            print(f"  Warning: {RESEARCH_PROGRAM_PATH} is empty or missing mid-run. Skipping iteration.")
            log_iteration(IterationLog(iteration, 0.0, 0, time.time() - iter_start, "error"), PROGRESS_PATH)
            continue

        # Step 2: Identify gaps (now uses store)
        print("  Analyzing gaps and planning searches...")
        try:
            # C5: Inject failed search context
            failed_context = get_failed_search_summary()
            combined_guidance = human_guidance
            if failed_context:
                combined_guidance = f"{human_guidance}\n\n{failed_context}" if human_guidance else failed_context
            gap_analysis = identify_gaps(research_program, store, combined_guidance)
            human_guidance = ""  # Consumed
        except Exception as e:
            print(f"  Error in gap analysis: {e}")
            log_iteration(IterationLog(iteration, 0.0, 0, time.time() - iter_start, "error"), PROGRESS_PATH)
            continue

        gaps = gap_analysis.get("gaps", [])
        search_plan = gap_analysis.get("search_plan", [])
        print(f"  Gaps identified: {len(gaps)}")
        for g in gaps[:3]:
            print(f"    - {g[:80]}")
        print(f"  Searches planned: {len(search_plan)}")

        # Step 2.5: Create any new topics from gap analysis
        for new_topic in gap_analysis.get("new_topics", []):
            store.add_topic(Topic(
                id=new_topic["id"],
                title=new_topic.get("title", new_topic["id"]),
                keywords=new_topic.get("keywords", []),
            ))

        # ── Checkpoint 1: Review search plan before executing ──
        cp1_lines = [f"Gaps: {len(gaps)}"]
        for s in search_plan:
            query = s.get("query", "(no query)")
            sources = s.get("sources", [])
            cp1_lines.append(f"  Search: \"{query}\" via {', '.join(sources)}")
        cp1 = prompt_checkpoint(mode, "Search Plan Review", cp1_lines)
        if not cp1.approved:
            print("  Stopped by user at search plan checkpoint.")
            break
        if cp1.guidance:
            # Add user's guidance as an extra search query
            search_plan.append({
                "query": cp1.guidance,
                "sources": ["duckduckgo"],
                "rationale": "User-provided query",
            })
            print(f"  Added user query: \"{cp1.guidance}\"")

        # Step 3: Execute searches
        # C5: Skip queries similar to previously failed ones
        filtered_plan = []
        for s in search_plan:
            query = s.get("query", "")
            if not query:
                continue
            if is_similar_to_failed(query):
                print(f"  Skipping similar-to-failed query: {query[:60]}...")
                continue
            filtered_plan.append(s)

        queries = [
            {"query": s.get("query", ""), "sources": s.get("sources", ["duckduckgo"])}
            for s in filtered_plan
        ]
        results = execute_searches(queries)
        print(f"  Results collected: {len(results)}")

        # C5: Record search history per plan entry
        for s in filtered_plan:
            query = s.get("query", "")
            sources = s.get("sources", [])
            # Estimate per-query result count (approximate: total / num queries)
            per_query_count = len(results) // max(len(filtered_plan), 1)
            record_search_result(query, sources, per_query_count, iteration)

        if not results:
            print("  No results found. Trying broader search...")
            # Fallback: use DuckDuckGo with gap descriptions
            fallback_queries = [{"query": g[:120], "sources": ["duckduckgo"]} for g in gaps[:3]]
            results = execute_searches(fallback_queries)
            print(f"  Fallback results: {len(results)}")

        if not results:
            print("  Still no results. Skipping synthesis.")
            log_iteration(IterationLog(iteration, 0.0, 0, time.time() - iter_start, "no_results"), PROGRESS_PATH)
            continue

        # Step 3.5 + 4: Store results and synthesize per-topic (delta merge)
        print("  Synthesizing findings...")
        try:
            # Build topic mapping from search plan
            query_topic_map = {}
            for plan_entry in search_plan:
                q = plan_entry.get("query", "")
                tid = plan_entry.get("topic_id", "")
                if q and tid:
                    query_topic_map[q] = tid

            # Save raw results
            store.save_raw_results(iteration, [
                {"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source}
                for r in results
            ])

            # Assign results to topics and store as sources
            for result in results:
                # Default: use first topic_id from search_plan or empty
                assigned_topic = ""
                for plan_entry in search_plan:
                    tid = plan_entry.get("topic_id", "")
                    if tid and tid != "new":
                        assigned_topic = tid
                        break

                source_id = store.next_source_id()
                topic_ids = [assigned_topic] if assigned_topic else []
                store.add_source(source_id, {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "full_text": result.full_text or "",
                    "source": result.source,
                    "authors": result.authors,
                    "year": result.year,
                    "topic_ids": topic_ids,
                    "iteration": iteration,
                })
            # Prioritize stale topics (synthesize existing data before searching more)
            stale_first = store.prioritize_stale_topics()
            topics_from_plan = set()
            for plan_entry in search_plan:
                tid = plan_entry.get("topic_id", "")
                if tid and tid != "new":
                    topics_from_plan.add(tid)

            # Combine: stale topics first, then plan topics
            topics_to_update = list(dict.fromkeys(stale_first + list(topics_from_plan)))

            for tid in topics_to_update:
                unsynthesized = store.get_unsynthesized_sources(tid)
                if not unsynthesized:
                    continue
                # Convert unsynthesized sources to SearchResult format
                topic_results = []
                for src in unsynthesized:
                    topic_results.append(SearchResult(
                        title=src.get("title", ""),
                        url=src.get("url", ""),
                        snippet=src.get("snippet", ""),
                        source=src.get("source", ""),
                        authors=src.get("authors", ""),
                        year=src.get("year", ""),
                        full_text=src.get("full_text", ""),
                    ))

                updated_summary = extract_topic_findings(topic_results, research_program, store, tid)

                # C6: Quality gate — check if synthesis degraded the summary
                existing = store.read_summary(tid)
                is_ok, reason = check_synthesis_quality(existing, updated_summary)
                if not is_ok:
                    print(f"  ⚠️ Quality regression in {tid}: {reason}. Keeping existing summary.")
                    continue

                # B5: Lightweight citation verification
                source_titles = [src.get("title", "") for src in unsynthesized]
                cite_check = verify_citations_against_sources(updated_summary, source_titles)
                if cite_check["suspicious"]:
                    print(f"  [citations] {len(cite_check['suspicious'])} unverified citations in {tid}")

                max_source_id = store.next_source_id() - 1
                store.write_summary(tid, updated_summary, last_source_id=max_source_id)

                # Update stale flag
                idx = store.load_index()
                for t in idx.topics:
                    if t.id == tid:
                        t.unsynthesized_sources = 0
                        t.stale = False
                        break
                store.save_index(idx)

            # Regenerate findings.md from summaries
            updated_findings = store.regenerate_findings()
            write_file(FINDINGS_PATH, updated_findings)

            # B4: Cross-topic synthesis (every 3 iterations or on last)
            if iteration % 3 == 0 or iteration == max_iterations:
                print("  Running cross-topic synthesis...")
                cross_analysis = cross_topic_synthesis(store, research_program)
                if cross_analysis:
                    print(f"  Cross-topic insights generated ({len(cross_analysis)} chars)")
        except Exception as e:
            print(f"  Error in synthesis: {e}")
            log_iteration(IterationLog(iteration, 0.0, len(results), time.time() - iter_start, "error"), PROGRESS_PATH)
            continue

        # Step 5: Evaluate coverage (now uses store)
        print("  Evaluating coverage...")
        try:
            evaluation = evaluate_coverage(research_program, store)
        except Exception as e:
            print(f"  Error in evaluation: {e}")
            evaluation = EvaluationResult(coverage_score=0.0)

        # Update index meta
        idx = store.load_index()
        idx.meta.iterations = iteration
        idx.meta.avg_coverage = evaluation.coverage_score
        store.save_index(idx)

        duration = time.time() - iter_start
        status = "target_reached" if evaluation.coverage_score >= target_coverage else "continue"

        # Step 6: Log iteration
        log_iteration(
            IterationLog(iteration, evaluation.coverage_score, len(results), duration, status),
            PROGRESS_PATH,
        )

        print(f"  Coverage: {evaluation.coverage_score:.2f} | Sources: {len(results)} | Time: {duration:.1f}s")
        if evaluation.gaps:
            print(f"  Remaining gaps: {len(evaluation.gaps)}")
            for g in evaluation.gaps[:2]:
                print(f"    - {g[:80]}")

        # ── Checkpoint 2: Review evaluation before next iteration ──
        cp2_lines = [f"Coverage: {evaluation.coverage_score:.2f} (target: {target_coverage})"]
        if evaluation.gaps:
            for g in evaluation.gaps[:3]:
                cp2_lines.append(f"  Gap: {g[:80]}")
        cp2 = prompt_checkpoint(mode, "Iteration Complete", cp2_lines)
        if not cp2.approved:
            print("  Stopped by user at evaluation checkpoint.")
            break
        if cp2.guidance:
            human_guidance = cp2.guidance  # Carry to next iteration's identify_gaps()
            print(f"  Guidance noted for next iteration: \"{human_guidance}\"")

        # Track scores for stopping analysis
        recent_scores.append(evaluation.coverage_score)

        # Step 7: Multi-signal stopping (C3)
        stop_signal = should_stop_research(recent_scores, iteration, target_coverage, max_iterations)
        if stop_signal.should_stop:
            reasons = {
                "target_reached": f"Target coverage reached ({evaluation.coverage_score:.2f} >= {target_coverage})",
                "coverage_stable_at_target": f"Coverage stable at target for 3 iterations",
                "max_iterations_reached": f"Max iterations ({max_iterations}) reached",
                "diminishing_returns": "Diminishing returns — coverage plateaued",
            }
            reason_msg = reasons.get(stop_signal.reason, stop_signal.reason)
            if stop_signal.confidence >= 0.8:
                print(f"\n✅ Stopping: {reason_msg}")
            else:
                print(f"\n⚠️ Stopping ({stop_signal.confidence:.0%} confidence): {reason_msg}")
            break
    else:
        print(f"\n⚠️  Max iterations ({max_iterations}) reached.")

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Research complete! {iteration} iterations in {total_time:.0f}s")
    print(f"Findings saved to: {FINDINGS_PATH}")
    print(f"Progress log: {PROGRESS_PATH}")
    print(f"{'=' * 60}")


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Autonomous research agent")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--target-coverage", type=float, default=DEFAULT_TARGET_COVERAGE)
    parser.add_argument("--time-budget", type=float, default=DEFAULT_TIME_BUDGET_MIN, help="Time budget in minutes")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in InteractionMode],
        default=InteractionMode.AUTO.value,
        help="Interaction mode: auto (no pauses), supervised (pause at checkpoints), interactive (pause + accept guidance)",
    )
    args = parser.parse_args()

    run_research_loop(
        max_iterations=args.max_iterations,
        target_coverage=args.target_coverage,
        time_budget_min=args.time_budget,
        mode=InteractionMode(args.mode),
    )


if __name__ == "__main__":
    main()
