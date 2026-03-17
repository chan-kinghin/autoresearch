"""
research.py — Autonomous research loop.

Usage: uv run research.py [--max-iterations N] [--target-coverage 0.8] [--time-budget 60] [--mode auto|supervised|interactive]
"""

from __future__ import annotations

import argparse
import os
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
    extract_webpage,
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


def identify_gaps(
    research_program: str,
    store: KnowledgeStore,
    human_guidance: str = "",
    evaluator_suggestions: list[str] | None = None,
) -> dict:
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
    if evaluator_suggestions:
        suggestions_text = "\n".join(f"- {s}" for s in evaluator_suggestions)
        guidance_block += f"""
## Evaluator Suggestions
The evaluator suggested focusing on:
{suggestions_text}
Consider these when planning searches.
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
        # Tiered content budget: Tier 1 sources with full_text get 6000 chars,
        # Tier 2/3 snippet-only sources get 2000 chars
        if r.full_text:
            content = r.full_text[:6000]
        elif r.snippet:
            content = r.snippet[:2000]
        else:
            content = ""
        if content:
            entry += f"- **Content**: {content}\n"
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
    evaluator_suggestions: list[str] = []  # Carries evaluator suggestions to next identify_gaps()

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
            gap_analysis = identify_gaps(research_program, store, human_guidance, evaluator_suggestions)
            human_guidance = ""  # Consumed
            evaluator_suggestions = []  # Consumed
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
        queries = [
            {"query": s.get("query", ""), "sources": s.get("sources", ["duckduckgo"])}
            for s in search_plan
            if s.get("query")
        ]
        results = execute_searches(queries)
        print(f"  Results collected: {len(results)}")

        if not results:
            print("  No results found. Trying broader search...")
            # Fallback: use DuckDuckGo with gap descriptions
            fallback_queries = [{"query": g[:120], "sources": ["duckduckgo"]} for g in gaps[:3]]
            results = execute_searches(fallback_queries)
            print(f"  Fallback results: {len(results)}")

        # Enrich top web results (no full_text) with extract_webpage()
        enriched = 0
        for r in results:
            if enriched >= 5:
                break
            if r.url and not r.full_text and r.source in ("duckduckgo",):
                try:
                    print(f"  Extracting page: {r.url[:60]}...")
                    text = extract_webpage(r.url)
                    if text:
                        r.full_text = text
                        enriched += 1
                except Exception as e:
                    print(f"  [extract] Failed for {r.url[:60]}: {e}")

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

        # Store evaluator suggestions for next iteration's identify_gaps()
        evaluator_suggestions = evaluation.suggested_queries or []

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

        # Step 7: Check stopping condition
        if evaluation.coverage_score >= target_coverage:
            print(f"\n✅ Target coverage reached ({evaluation.coverage_score:.2f} >= {target_coverage})")
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
