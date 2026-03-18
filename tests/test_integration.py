"""Integration tests — full lifecycle simulation with mocked LLM/search.

Exercises the complete pipeline:
  bootstrap → identify gaps → search → store sources → delta merge →
  evaluate → smart routing → second iteration → findings regeneration

No API keys needed — all external calls are mocked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_store import KnowledgeStore, Topic, Question, StoreIndex, IndexMeta
from research import (
    CheckpointResult,
    InteractionMode,
    StoppingSignal,
    check_synthesis_quality,
    cross_topic_synthesis,
    evaluate_coverage,
    extract_topic_findings,
    get_failed_search_summary,
    identify_gaps,
    is_similar_to_failed,
    record_search_result,
    run_research_loop,
    should_stop_research,
    verify_citations_against_sources,
)
from search import (
    EvaluationResult,
    QueryRecord,
    SearchResult,
    SourceStats,
    filter_results_by_relevance,
    get_effective_queries,
    get_failed_queries,
    get_query_history,
    get_source_stats,
    is_source_exhausted,
    record_query,
    reset_source_stats,
    score_query_specificity,
    validate_query_sources,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh knowledge store in a temp directory."""
    return KnowledgeStore(str(tmp_path / "research_store"))


@pytest.fixture
def research_program():
    return """# LLM Agents Survey
Research Questions:
1. What are the main architectural patterns for LLM agents?
2. How do frameworks like LangGraph and CrewAI compare?
3. What are the safety and evaluation challenges?
"""


@pytest.fixture
def sample_search_results():
    return [
        SearchResult(
            title="ReAct: Synergizing Reasoning and Acting",
            url="https://arxiv.org/abs/2210.03629",
            snippet="We propose ReAct, a framework that combines reasoning traces and actions.",
            source="arxiv",
            authors="Yao et al.",
            year="2022",
            full_text="ReAct combines chain-of-thought reasoning with action generation. "
            "The agent interleaves thinking steps with tool calls, enabling more grounded "
            "decision-making. Experiments show ReAct outperforms pure reasoning on QA tasks.",
        ),
        SearchResult(
            title="LangGraph: Multi-Agent Workflows",
            url="https://blog.langchain.dev/langgraph",
            snippet="LangGraph enables cyclic computation graphs for LLM agents.",
            source="duckduckgo",
            full_text="LangGraph is a library for building stateful, multi-actor applications. "
            "It extends LangChain with support for cycles and persistent state. "
            "Key features: checkpointing, human-in-the-loop, and streaming.",
        ),
        SearchResult(
            title="Agent Safety Challenges",
            url="https://example.com/safety",
            snippet="Autonomous agents face unique safety challenges including goal misalignment.",
            source="semantic_scholar",
            authors="Smith, Jones",
            year="2024",
            full_text="Safety challenges for LLM agents include: 1) Goal misalignment — agents may "
            "pursue instrumental goals. 2) Tool misuse — agents with code execution can cause harm. "
            "3) Evaluation difficulty — hard to test agents in open-ended environments.",
        ),
    ]


# ── Test: Full Lifecycle (Single Iteration) ─────────────────────────────


class TestFullLifecycle:
    """Simulates one complete research iteration end-to-end."""

    def test_single_iteration_lifecycle(self, store, research_program, sample_search_results):
        """Walk through: create topics → add sources → delta merge → evaluate → regenerate."""

        # ── Step 1: Create topics ──
        store.add_topic(Topic(
            id="arch_patterns", title="Architecture Patterns",
            keywords=["react", "architecture", "patterns", "agent"],
        ))
        store.add_topic(Topic(
            id="frameworks", title="Framework Comparisons",
            keywords=["langgraph", "crewai", "autogen", "framework"],
        ))
        store.add_topic(Topic(
            id="safety", title="Safety Challenges",
            keywords=["safety", "evaluation", "alignment", "risks"],
        ))

        index = store.load_index()
        assert len(index.topics) == 3
        assert all(t.coverage == 0.0 for t in index.topics)

        # ── Step 2: Add sources with topic assignments ──
        store.add_source(1, {
            "title": "ReAct: Synergizing Reasoning and Acting",
            "url": "https://arxiv.org/abs/2210.03629",
            "snippet": "We propose ReAct...",
            "full_text": sample_search_results[0].full_text,
            "source": "arxiv",
            "topic_ids": ["arch_patterns"],
            "iteration": 1,
        })
        store.add_source(2, {
            "title": "LangGraph: Multi-Agent Workflows",
            "url": "https://blog.langchain.dev/langgraph",
            "snippet": "LangGraph enables...",
            "full_text": sample_search_results[1].full_text,
            "source": "duckduckgo",
            "topic_ids": ["frameworks"],
            "iteration": 1,
        })
        store.add_source(3, {
            "title": "Agent Safety Challenges",
            "url": "https://example.com/safety",
            "snippet": "Autonomous agents face...",
            "full_text": sample_search_results[2].full_text,
            "source": "semantic_scholar",
            "topic_ids": ["safety"],
            "iteration": 1,
        })

        # Verify staleness
        arch = store.get_topic("arch_patterns")
        assert arch.stale is True
        assert arch.unsynthesized_sources == 1
        assert arch.source_count == 1

        # ── Step 3: Delta merge — synthesize per topic ──
        for topic_id in ["arch_patterns", "frameworks", "safety"]:
            unsynthesized = store.get_unsynthesized_sources(topic_id)
            assert len(unsynthesized) == 1  # Each topic got 1 source

            # Simulate LLM synthesis (just use the source full_text as summary)
            summary = f"## {topic_id}\n\n" + unsynthesized[0]["full_text"]
            store.write_summary(topic_id, summary, last_source_id=3)

            # Clear stale flag
            idx = store.load_index()
            for t in idx.topics:
                if t.id == topic_id:
                    t.unsynthesized_sources = 0
                    t.stale = False
            store.save_index(idx)

        # Verify synthesis cleared staleness
        arch = store.get_topic("arch_patterns")
        assert arch.stale is False
        assert arch.unsynthesized_sources == 0

        # Verify no unsynthesized sources remain
        for tid in ["arch_patterns", "frameworks", "safety"]:
            assert store.get_unsynthesized_sources(tid) == []

        # ── Step 4: Update coverage ──
        store.update_topic_coverage("arch_patterns", 0.4)
        store.update_topic_coverage("frameworks", 0.3)
        store.update_topic_coverage("safety", 0.5)

        # ── Step 5: Smart routing ──
        relevant = store.select_relevant_summaries(
            ["react architecture patterns for LLM agents"]
        )
        # arch_patterns should rank highest (keyword match + low coverage)
        assert len(relevant) >= 1
        assert relevant[0][0] == "arch_patterns"
        assert "ReAct" in relevant[0][1]

        # ── Step 6: Budget-aware context ──
        context = store.get_context_with_budget(
            ["architecture patterns"], token_budget=8000
        )
        assert "Knowledge Store Index" in context
        assert "Architecture Patterns" in context

        # ── Step 7: Regenerate findings ──
        findings = store.regenerate_findings()
        assert "Research Findings" in findings
        assert "ReAct" in findings
        assert "LangGraph" in findings
        assert "Safety" in findings

        # Verify findings.md was written
        findings_path = store.root / "findings.md"
        assert findings_path.exists()
        assert findings_path.read_text(encoding="utf-8") == findings


# ── Test: Multi-Iteration Simulation ────────────────────────────────────


class TestMultiIteration:
    """Simulates two iterations to verify delta merge and staleness tracking."""

    def test_second_iteration_delta_merge(self, store, research_program):
        """After iteration 1 synthesis, add new sources and verify delta merge works."""

        # ── Iteration 1: Initial setup ──
        store.add_topic(Topic(
            id="arch", title="Architecture",
            keywords=["react", "architecture"],
        ))
        store.add_source(1, {
            "title": "ReAct Paper",
            "url": "https://arxiv.org/react",
            "full_text": "ReAct combines reasoning and acting.",
            "topic_ids": ["arch"],
            "iteration": 1,
        })
        store.write_summary("arch", "ReAct combines reasoning and acting.", last_source_id=1)

        # Clear stale
        idx = store.load_index()
        for t in idx.topics:
            if t.id == "arch":
                t.unsynthesized_sources = 0
                t.stale = False
        store.save_index(idx)

        # ── Iteration 2: New source arrives ──
        store.add_source(2, {
            "title": "Plan-and-Execute Pattern",
            "url": "https://arxiv.org/plan",
            "full_text": "Plan-and-execute decomposes tasks into subtasks.",
            "topic_ids": ["arch"],
            "iteration": 2,
        })

        # Verify staleness tracking
        arch = store.get_topic("arch")
        assert arch.stale is True
        assert arch.unsynthesized_sources == 1

        # Only source 2 should be unsynthesized (source 1 was already merged)
        unsynthesized = store.get_unsynthesized_sources("arch")
        assert len(unsynthesized) == 1
        assert unsynthesized[0]["title"] == "Plan-and-Execute Pattern"

        # Simulate delta merge
        existing = store.read_summary("arch")
        assert "ReAct" in existing
        new_summary = existing.strip() + "\n\nPlan-and-execute decomposes tasks into subtasks."
        store.write_summary("arch", new_summary, last_source_id=2)

        # Verify merged content
        final = store.read_summary("arch")
        assert "ReAct" in final
        assert "Plan-and-execute" in final

    def test_stale_prioritization_across_topics(self, store):
        """Stale topics with more unsynthesized sources are prioritized."""
        store.add_topic(Topic(id="a", title="Topic A"))
        store.add_topic(Topic(id="b", title="Topic B"))
        store.add_topic(Topic(id="c", title="Topic C"))

        # Make "b" have 3 unsynthesized, "a" have 1, "c" have 0
        for i in range(1, 4):
            store.add_source(i, {"title": f"Source {i}", "topic_ids": ["b"], "iteration": 1})
        store.add_source(4, {"title": "Source 4", "topic_ids": ["a"], "iteration": 1})

        priority = store.prioritize_stale_topics()
        assert priority == ["b", "a"]  # b has 3, a has 1, c has 0


# ── Test: Bootstrap from Existing Findings ──────────────────────────────


class TestBootstrapIntegration:
    """Tests bootstrapping a store from an existing findings.md."""

    def test_bootstrap_then_add_sources(self, store, research_program):
        """Bootstrap from findings, then add new sources and verify the store works."""
        mock_llm = MagicMock(return_value={
            "topics": [
                {
                    "id": "arch_patterns",
                    "title": "Architecture Patterns",
                    "keywords": ["react", "architecture"],
                    "summary": "ReAct and plan-and-execute are the main patterns.",
                },
                {
                    "id": "frameworks",
                    "title": "Framework Comparisons",
                    "keywords": ["langgraph", "crewai"],
                    "summary": "LangGraph focuses on stateful graphs.",
                },
            ],
            "questions": [
                {"id": "q1", "text": "How do agents handle failures?", "status": "unanswered", "related_topics": ["arch_patterns"]},
            ],
        })

        existing_findings = "# Research Findings\nSome initial research about LLM agent architectures."
        store.bootstrap_from_findings(existing_findings, research_program, mock_llm)

        # Verify bootstrap created topics and summaries
        index = store.load_index()
        assert len(index.topics) == 2
        assert len(index.questions) == 1

        summary = store.read_summary("arch_patterns")
        assert "ReAct" in summary

        # Now add a new source — should work seamlessly
        store.add_source(1, {
            "title": "New paper on Tree-of-Thought",
            "url": "https://arxiv.org/tot",
            "full_text": "Tree-of-thought explores multiple reasoning paths.",
            "topic_ids": ["arch_patterns"],
            "iteration": 1,
        })

        arch = store.get_topic("arch_patterns")
        assert arch.stale is True

        unsynthesized = store.get_unsynthesized_sources("arch_patterns")
        assert len(unsynthesized) == 1


# ── Test: Smart Routing Edge Cases ──────────────────────────────────────


class TestSmartRoutingIntegration:
    """Tests relevance routing in realistic scenarios."""

    def test_routing_prefers_low_coverage_relevant_topics(self, store):
        """With two relevant topics, the lower-coverage one should rank higher."""
        store.add_topic(Topic(
            id="safety", title="Safety",
            keywords=["safety", "alignment", "risks"],
            coverage=0.8,
        ))
        store.add_topic(Topic(
            id="eval", title="Evaluation",
            keywords=["evaluation", "benchmarks", "safety"],
            coverage=0.2,
        ))
        store.write_summary("safety", "Safety is well-covered already.", last_source_id=5)
        store.write_summary("eval", "Evaluation needs more work.", last_source_id=2)

        results = store.select_relevant_summaries(["safety evaluation challenges"])
        assert len(results) == 2
        # "eval" has lower coverage AND keyword match → should rank first
        assert results[0][0] == "eval"

    def test_routing_with_no_keyword_match(self, store):
        """Topics with no keyword overlap should not be selected."""
        store.add_topic(Topic(
            id="unrelated", title="Unrelated Topic",
            keywords=["quantum", "physics"],
            coverage=0.1,
        ))
        store.write_summary("unrelated", "Quantum physics content.", last_source_id=1)

        results = store.select_relevant_summaries(["LLM agent architecture"])
        assert len(results) == 0

    def test_context_budget_limits_output(self, store):
        """Token budget should cap how many summaries are loaded."""
        for i in range(5):
            tid = f"topic_{i}"
            store.add_topic(Topic(
                id=tid, title=f"Topic {i}",
                keywords=["shared_keyword"],
                coverage=0.1,
            ))
            # ~500 tokens each
            store.write_summary(tid, "x" * 2000, last_source_id=1)

        # Very tight budget — should only fit index + maybe 1 summary
        context = store.get_context_with_budget(
            ["shared_keyword"], token_budget=1000
        )
        # Should still have the index
        assert "Knowledge Store Index" in context
        # Token count of context should be reasonable
        tokens = KnowledgeStore.estimate_tokens(context)
        assert tokens < 1500  # some slack for index


# ── Test: Research Loop with Mocked Externals ───────────────────────────


class TestResearchLoopIntegration:
    """Full run_research_loop simulation with mocked LLM and search."""

    @patch("research.evaluate_coverage")
    @patch("research.extract_topic_findings", return_value="# Updated findings for this topic")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    @patch("research.write_file")
    @patch("research.KnowledgeStore")
    def test_two_iteration_loop(
        self, MockStore, mock_write, mock_read, mock_init_log,
        mock_gaps, mock_search, mock_extract, mock_eval,
    ):
        """Run 2 iterations: first creates topics, second adds more sources."""
        # Setup mock store
        mock_store = MagicMock()
        MockStore.return_value = mock_store
        mock_store.exists.return_value = False
        mock_store.load_index.return_value = StoreIndex(
            topics=[Topic(id="arch", title="Architecture", stale=True, unsynthesized_sources=1)],
            questions=[],
            relations=[],
            meta=IndexMeta(total_sources=1),
        )
        mock_store.get_context_with_budget.return_value = "## Index\n- arch: 0.00"
        mock_store.get_context_for_gaps.return_value = "## Index\n- arch: 0.00"
        mock_store.read_summary.return_value = "Existing arch summary"
        mock_store.get_topic.return_value = Topic(id="arch", title="Architecture")
        mock_store.next_source_id.return_value = 1
        mock_store.prioritize_stale_topics.return_value = ["arch"]
        mock_store.get_unsynthesized_sources.return_value = [
            {"title": "Paper A", "url": "https://example.com", "snippet": "...",
             "full_text": "Full text", "source": "arxiv", "authors": "", "year": "2024",
             "topic_ids": ["arch"]},
        ]
        mock_store.regenerate_findings.return_value = "# Regenerated findings"

        mock_read.return_value = "research program content"

        # Iteration 1: low coverage → continue
        # Iteration 2: high coverage → stop
        mock_gaps.return_value = {
            "gaps": ["What are architecture patterns?"],
            "search_plan": [{"query": "LLM agent architecture", "sources": ["arxiv"], "topic_id": "arch"}],
            "new_topics": [],
        }
        mock_search.return_value = [
            SearchResult(title="Paper A", url="https://example.com/a", snippet="snip", source="arxiv"),
        ]
        mock_eval.side_effect = [
            EvaluationResult(coverage_score=0.4),
            EvaluationResult(coverage_score=0.9),
        ]

        run_research_loop(max_iterations=2, target_coverage=0.8)

        # Verify key interactions
        assert mock_gaps.call_count == 2
        assert mock_search.call_count == 2
        assert mock_store.add_source.called
        assert mock_store.regenerate_findings.called
        assert mock_store.save_index.called


# ── Test: Token Budget Verification ─────────────────────────────────────


class TestTokenBudget:
    """Verify that progressive disclosure actually reduces context size."""

    def test_context_size_stays_constant(self, store):
        """Even with many topics and sources, context stays within budget."""
        # Create 10 topics with substantial summaries
        for i in range(10):
            tid = f"topic_{i}"
            store.add_topic(Topic(
                id=tid, title=f"Topic {i}",
                keywords=[f"kw_{i}", "shared"],
                coverage=0.1 * i,
            ))
            # Each summary is ~500 tokens
            store.write_summary(tid, f"Detailed findings for topic {i}. " * 50, last_source_id=i)

        # Add 50 sources
        for i in range(1, 51):
            store.add_source(i, {
                "title": f"Source {i}",
                "topic_ids": [f"topic_{i % 10}"],
                "iteration": (i // 10) + 1,
            })

        # Context should still fit within budget
        context = store.get_context_with_budget(
            ["topic_0 findings"], token_budget=4000
        )
        tokens = KnowledgeStore.estimate_tokens(context)
        assert tokens < 4000, f"Context exceeded budget: {tokens} tokens"

        # Index alone should be compact
        index_context = store.get_context_for_gaps()
        index_tokens = KnowledgeStore.estimate_tokens(index_context)
        assert index_tokens < 1000, f"Index too large: {index_tokens} tokens"

    def test_full_findings_vs_progressive_disclosure(self, store):
        """Demonstrate that progressive disclosure uses less context than full findings."""
        # Create a realistic store
        for i in range(5):
            tid = f"topic_{i}"
            store.add_topic(Topic(
                id=tid, title=f"Topic {i}",
                keywords=[f"keyword_{i}"],
                coverage=0.2 * i,
            ))
            store.write_summary(tid, f"Detailed research findings for topic {i}. " * 100, last_source_id=i * 5)

        # Full findings (old approach) — everything concatenated
        full_findings = store.regenerate_findings()
        full_tokens = KnowledgeStore.estimate_tokens(full_findings)

        # Progressive disclosure (new approach) — only relevant context
        progressive_context = store.get_context_with_budget(
            ["keyword_0"], token_budget=4000
        )
        progressive_tokens = KnowledgeStore.estimate_tokens(progressive_context)

        # Progressive disclosure should use significantly less context
        assert progressive_tokens < full_tokens, (
            f"Progressive ({progressive_tokens}) should be less than full ({full_tokens})"
        )
        # Should be at least 50% smaller
        ratio = progressive_tokens / full_tokens
        assert ratio < 0.7, f"Expected >30% reduction, got {(1-ratio)*100:.0f}%"


# ── Test: New search.py Functions Are Importable & Callable ──────────


class TestSearchModuleNewFunctions:
    """Verify new functions in search.py are importable and work correctly."""

    def test_dataclass_instantiation(self):
        """QueryRecord and SourceStats dataclasses can be instantiated."""
        qr = QueryRecord(query="test query", sources=["arxiv"], results_count=5)
        assert qr.query == "test query"
        assert qr.sources == ["arxiv"]
        assert qr.results_count == 5
        assert qr.useful_results == 0  # default
        assert qr.iteration == 0  # default

        ss = SourceStats()
        assert ss.calls == 0
        assert ss.unique_urls == 0
        assert ss.total_results == 0

    def test_score_query_specificity_basic(self):
        """score_query_specificity returns float in [0, 1]."""
        # Very short query => low specificity
        score_short = score_query_specificity("LLM")
        assert 0.0 <= score_short <= 1.0
        assert score_short < 0.3

        # Longer, specific query
        score_long = score_query_specificity(
            "transformer attention mechanism multi-head self-attention architecture"
        )
        assert 0.0 <= score_long <= 1.0
        assert score_long > score_short

        # Vague query penalized
        score_vague = score_query_specificity("general overview information about things")
        score_specific = score_query_specificity("BERT pre-training masked language model")
        assert score_specific > score_vague

    def test_score_query_specificity_with_keywords(self):
        """Domain keywords boost specificity score."""
        keywords = {"transformer", "attention", "bert"}
        score_no_kw = score_query_specificity("transformer attention mechanism")
        score_with_kw = score_query_specificity("transformer attention mechanism", keywords)
        assert score_with_kw >= score_no_kw

    def test_is_source_exhausted(self):
        """is_source_exhausted detects diminishing returns."""
        reset_source_stats()
        # Unknown source => not exhausted
        assert is_source_exhausted("nonexistent") is False

    def test_filter_results_by_relevance(self):
        """filter_results_by_relevance keeps relevant results, removes irrelevant ones."""
        results = [
            SearchResult(title="LLM Agent Architecture", url="http://a.com", snippet="Agent design patterns", source="arxiv"),
            SearchResult(title="Cooking Recipes", url="http://b.com", snippet="How to bake a cake", source="duckduckgo"),
            SearchResult(title="Transformer Models", url="http://c.com", snippet="Attention mechanisms in agents", source="arxiv"),
            SearchResult(title="Weather Forecast", url="http://d.com", snippet="Rain tomorrow", source="duckduckgo"),
        ]
        keywords = {"agent", "architecture", "transformer", "llm", "attention"}
        filtered = filter_results_by_relevance(results, keywords, min_relevance=0.1)
        # min_keep = max(2, int(4 * 0.7)) = 2; irrelevant results below threshold get dropped
        assert len(filtered) >= 2
        # The top results should be relevant ones (sorted by keyword overlap)
        relevant_titles = {r.title for r in filtered}
        assert "LLM Agent Architecture" in relevant_titles
        assert "Transformer Models" in relevant_titles

    def test_filter_results_empty_keywords(self):
        """filter_results_by_relevance returns all results when no keywords."""
        results = [
            SearchResult(title="A", url="http://a.com", snippet="a", source="arxiv"),
            SearchResult(title="B", url="http://b.com", snippet="b", source="arxiv"),
        ]
        filtered = filter_results_by_relevance(results, set())
        assert len(filtered) == 2

    def test_validate_query_sources_statistical(self):
        """Statistical queries should get academic sources added."""
        sources = ["duckduckgo"]
        adjusted = validate_query_sources("correlation between variables regression", sources)
        assert any(s in adjusted for s in ("semantic_scholar", "arxiv"))

    def test_validate_query_sources_recent(self):
        """Queries about recent topics should get web search added."""
        sources = ["semantic_scholar"]
        adjusted = validate_query_sources("latest 2025 breakthrough in AI", sources)
        assert "duckduckgo" in adjusted

    def test_validate_query_sources_academic(self):
        """Academic queries should get academic sources."""
        sources = ["duckduckgo"]
        adjusted = validate_query_sources("research paper on benchmark algorithms", sources)
        assert any(s in adjusted for s in ("semantic_scholar", "arxiv"))

    def test_query_history_tracking(self):
        """record_query, get_query_history, get_effective_queries, get_failed_queries work."""
        # These functions modify module-level state, so we just verify they're callable
        # and return the right types
        history = get_query_history()
        assert isinstance(history, list)

        effective = get_effective_queries()
        assert isinstance(effective, list)

        failed = get_failed_queries()
        assert isinstance(failed, list)


# ── Test: New research.py Functions Are Importable & Callable ────────


class TestResearchModuleNewFunctions:
    """Verify new functions in research.py are importable and work correctly."""

    def test_stopping_signal_dataclass(self):
        """StoppingSignal can be instantiated."""
        sig = StoppingSignal(should_stop=True, reason="test", confidence=0.9)
        assert sig.should_stop is True
        assert sig.reason == "test"
        assert sig.confidence == 0.9

    def test_checkpoint_result_dataclass(self):
        """CheckpointResult can be instantiated."""
        cr = CheckpointResult(approved=True, guidance="focus on safety")
        assert cr.approved is True
        assert cr.guidance == "focus on safety"

    def test_should_stop_research_max_iterations(self):
        """should_stop_research returns stop at max iterations."""
        signal = should_stop_research([0.3, 0.4, 0.5], iteration=10, target_coverage=0.8, max_iterations=10)
        assert signal.should_stop is True
        assert signal.reason == "max_iterations_reached"

    def test_should_stop_research_too_few_iterations(self):
        """should_stop_research continues with too few data points."""
        signal = should_stop_research([0.3, 0.4], iteration=2, target_coverage=0.8, max_iterations=20)
        assert signal.should_stop is False
        assert signal.reason == "too_few_iterations"

    def test_should_stop_research_stable_at_target(self):
        """should_stop_research detects stable coverage at target."""
        signal = should_stop_research([0.5, 0.6, 0.85, 0.82, 0.81], iteration=5, target_coverage=0.8, max_iterations=20)
        assert signal.should_stop is True
        assert signal.reason == "coverage_stable_at_target"

    def test_should_stop_research_target_reached(self):
        """should_stop_research detects single score at target."""
        signal = should_stop_research([0.3, 0.5, 0.85], iteration=3, target_coverage=0.8, max_iterations=20)
        assert signal.should_stop is True
        assert signal.reason == "target_reached"

    def test_should_stop_research_continue(self):
        """should_stop_research continues when coverage is below target."""
        signal = should_stop_research([0.3, 0.4, 0.5], iteration=3, target_coverage=0.8, max_iterations=20)
        assert signal.should_stop is False
        assert signal.reason == "continue"

    def test_check_synthesis_quality_first_synthesis(self):
        """First synthesis (empty old) is always OK."""
        ok, reason = check_synthesis_quality("", "New summary content here")
        assert ok is True
        assert reason == "first_synthesis"

    def test_check_synthesis_quality_shrunk(self):
        """Summary that shrinks dramatically is flagged."""
        old = "A" * 500
        new = "B" * 100
        ok, reason = check_synthesis_quality(old, new)
        assert ok is False
        assert "shrunk" in reason

    def test_check_synthesis_quality_ok(self):
        """Summary of similar length is OK."""
        old = "Original summary with [cite1] and [cite2] references."
        new = "Updated summary with [cite1] and [cite2] and [cite3] new references."
        ok, reason = check_synthesis_quality(old, new)
        assert ok is True

    def test_verify_citations_against_sources(self):
        """verify_citations_against_sources checks citation plausibility."""
        summary = "The [Reasoning Acting] showed improvements. [Unknown Paper] also contributed."
        sources = ["ReAct Synergizing Reasoning and Acting", "LangGraph Workflows"]
        result = verify_citations_against_sources(summary, sources)
        assert result["total"] == 2
        assert result["plausible"] >= 1  # "Reasoning Acting" matches source words
        assert isinstance(result["suspicious"], list)
        assert len(result["suspicious"]) >= 1  # "Unknown Paper" should be suspicious

    def test_record_and_query_search_history(self):
        """record_search_result and get_failed_search_summary work together."""
        # These modify module-level state; verify callable and correct types
        record_search_result("test query", ["arxiv"], result_count=5, iteration=1)
        summary = get_failed_search_summary()
        assert isinstance(summary, str)

    def test_is_similar_to_failed(self):
        """is_similar_to_failed detects overlapping queries."""
        # Record a failed query first
        record_search_result("specific failed query terms", ["duckduckgo"], result_count=0, iteration=1)
        # A very similar query (high word overlap) should be flagged
        assert is_similar_to_failed("specific failed query terms here") is True
        # A completely different query should not
        assert is_similar_to_failed("transformer architecture deep learning") is False


# ── Test: KnowledgeStore New Methods ─────────────────────────────────


class TestKnowledgeStoreNewMethods:
    """Verify new KnowledgeStore methods exist and work."""

    def test_build_citation_index(self, store):
        """build_citation_index returns a dict mapping citations to sources."""
        store.add_topic(Topic(id="t1", title="Topic 1"))
        store.write_summary("t1", "According to [Smith, 2024], results are promising.", last_source_id=1)
        store.add_source(1, {
            "title": "AI Research Results",
            "url": "http://example.com",
            "authors": "John Smith",
            "year": "2024",
            "source": "arxiv",
            "topic_ids": ["t1"],
        })
        index = store.build_citation_index()
        assert isinstance(index, dict)
        # "Smith, 2024" should match the source with author "Smith" and year "2024"
        assert "Smith, 2024" in index
        assert index["Smith, 2024"]["year"] == "2024"

    def test_generate_question_matrix_empty(self, store):
        """generate_question_matrix returns empty string with no questions."""
        result = store.generate_question_matrix()
        assert result == ""

    def test_generate_question_matrix_with_questions(self, store):
        """generate_question_matrix generates a table when questions exist."""
        store.add_topic(Topic(id="t1", title="Topic 1"))
        idx = store.load_index()
        idx.questions = [
            Question(id="q1", text="What is X?", status="answered", related_topics=["t1"]),
            Question(id="q2", text="How does Y work?", status="unanswered", related_topics=["t1"]),
        ]
        store.save_index(idx)

        matrix = store.generate_question_matrix()
        assert "Research Questions" in matrix
        assert "What is X?" in matrix
        assert "How does Y work?" in matrix
        assert "2/2" in matrix or "1/2" in matrix  # summary counts

    def test_analyze_source_recency_empty(self, store):
        """analyze_source_recency returns empty string with no dated sources."""
        result = store.analyze_source_recency()
        assert result == ""

    def test_analyze_source_recency_with_sources(self, store):
        """analyze_source_recency reports year distribution."""
        store.add_topic(Topic(id="t1", title="Topic 1"))
        for i, year in enumerate([2020, 2022, 2024], start=1):
            store.add_source(i, {
                "title": f"Paper {i}",
                "year": str(year),
                "topic_ids": ["t1"],
            })
        result = store.analyze_source_recency()
        assert "Source Recency" in result
        assert "2024" in result
        assert "2020" in result

    def test_get_contradictions_from_summaries(self, store):
        """get_contradictions_from_summaries detects contradiction markers."""
        store.add_topic(Topic(id="t1", title="Topic 1"))
        store.write_summary("t1", "Finding A is strong. However, source B contradicts this view.", last_source_id=1)
        contradictions = store.get_contradictions_from_summaries()
        assert len(contradictions) >= 1
        assert "Topic 1" in contradictions[0]

    def test_validate_citations(self, store):
        """validate_citations returns structured validation results."""
        store.add_topic(Topic(id="t1", title="Topic 1"))
        store.write_summary("t1", "As shown by [Smith, 2024], results are good.", last_source_id=1)
        result = store.validate_citations()
        assert "total_citations" in result
        assert "matched" in result
        assert "unmatched" in result
        assert "match_rate" in result
        assert isinstance(result["match_rate"], float)

    def test_export_executive_summary(self, store):
        """export_executive_summary produces structured markdown."""
        store.add_topic(Topic(id="t1", title="Architecture", coverage=0.6, source_count=3))
        store.write_summary("t1", "Architecture patterns include ReAct and plan-and-execute approaches.", last_source_id=3)
        summary = store.export_executive_summary()
        assert "Executive Summary" in summary
        assert "Architecture" in summary

    def test_export_bibtex(self, store):
        """export_bibtex produces BibTeX entries."""
        store.add_topic(Topic(id="t1", title="Topic 1"))
        store.add_source(1, {
            "title": "ReAct Paper",
            "authors": "Yao et al.",
            "year": "2022",
            "url": "https://arxiv.org/react",
            "source": "arxiv",
            "topic_ids": ["t1"],
        })
        bibtex = store.export_bibtex()
        assert "@article" in bibtex
        assert "ReAct Paper" in bibtex
        assert "2022" in bibtex


# ── Test: Cross-Module Integration ───────────────────────────────────


class TestCrossModuleIntegration:
    """Verify research.py correctly uses search.py functions."""

    def test_search_functions_used_in_research_module(self):
        """Verify the search.py functions that research.py imports are callable."""
        # These are imported in research.py and used in the loop
        from search import execute_searches, init_progress_log, log_iteration, llm_call, llm_json

        # Verify they are callable
        assert callable(execute_searches)
        assert callable(init_progress_log)
        assert callable(log_iteration)
        assert callable(llm_call)
        assert callable(llm_json)

    def test_search_utility_functions_used_cross_module(self):
        """search.py utility functions used by research.py are consistent."""
        # score_query_specificity is used in research.py's identify_gaps flow
        assert callable(score_query_specificity)
        result = score_query_specificity("transformer attention mechanism")
        assert isinstance(result, float)

        # is_source_exhausted is called inside execute_searches
        assert callable(is_source_exhausted)

        # filter_results_by_relevance is used after search
        assert callable(filter_results_by_relevance)

        # validate_query_sources is used in search planning
        assert callable(validate_query_sources)
        adjusted = validate_query_sources("research paper review", ["duckduckgo"])
        assert isinstance(adjusted, list)
        assert len(adjusted) >= 1

    @patch("research.llm_call")
    def test_cross_topic_synthesis_requires_two_topics(self, mock_llm, store):
        """cross_topic_synthesis returns empty with < 2 topics."""
        store.add_topic(Topic(id="t1", title="Single Topic"))
        store.write_summary("t1", "Only one topic.", last_source_id=1)
        result = cross_topic_synthesis(store, "research program")
        assert result == ""
        mock_llm.assert_not_called()

    @patch("research.llm_call", return_value="Cross-topic connections found between A and B.")
    def test_cross_topic_synthesis_with_two_topics(self, mock_llm, store):
        """cross_topic_synthesis calls LLM with >= 2 topics."""
        store.add_topic(Topic(id="t1", title="Topic A", coverage=0.5))
        store.add_topic(Topic(id="t2", title="Topic B", coverage=0.3))
        store.write_summary("t1", "Topic A findings about architecture.", last_source_id=1)
        store.write_summary("t2", "Topic B findings about safety.", last_source_id=2)
        result = cross_topic_synthesis(store, "research program")
        assert "Cross-topic" in result
        mock_llm.assert_called_once()


# ── Test: Enhanced regenerate_findings Output ────────────────────────


class TestEnhancedFindings:
    """Verify regenerate_findings includes new sections."""

    def test_findings_contains_confidence_levels(self, store):
        """Findings should include confidence levels (HIGH/MEDIUM/LOW)."""
        store.add_topic(Topic(
            id="t1", title="Well Covered Topic",
            coverage=0.8, source_count=6,
        ))
        store.write_summary("t1", "Comprehensive findings with many sources.", last_source_id=6)
        findings = store.regenerate_findings()
        assert "Confidence: HIGH" in findings

    def test_findings_contains_low_confidence(self, store):
        """Topics with few sources get LOW confidence."""
        store.add_topic(Topic(
            id="t1", title="Sparse Topic",
            coverage=0.2, source_count=1,
        ))
        store.write_summary("t1", "Limited findings.", last_source_id=1)
        findings = store.regenerate_findings()
        assert "Confidence: LOW" in findings

    def test_findings_contains_question_matrix(self, store):
        """Findings should include the question resolution matrix when questions exist."""
        store.add_topic(Topic(id="t1", title="Topic 1"))
        idx = store.load_index()
        idx.questions = [
            Question(id="q1", text="What is the main pattern?", status="answered", related_topics=["t1"]),
            Question(id="q2", text="How to evaluate?", status="unanswered", related_topics=["t1"]),
        ]
        store.save_index(idx)
        store.write_summary("t1", "Some findings here.", last_source_id=1)

        findings = store.regenerate_findings()
        assert "Research Questions: Coverage Status" in findings
        assert "What is the main pattern?" in findings

    def test_findings_contains_contradiction_log(self, store):
        """Findings should include contradiction log when contradictions exist."""
        store.add_topic(Topic(id="t1", title="Debated Topic"))
        store.write_summary(
            "t1",
            "Source A claims X. However, source B contradicts this with evidence for Y.",
            last_source_id=2,
        )
        findings = store.regenerate_findings()
        assert "Known Conflicts" in findings

    def test_findings_contains_citation_validation(self, store):
        """Findings should include citation validation section."""
        store.add_topic(Topic(id="t1", title="Cited Topic", source_count=2))
        store.write_summary("t1", "According to [Smith, 2024], the approach works.", last_source_id=1)
        store.add_source(1, {
            "title": "AI Approach Paper",
            "authors": "John Smith",
            "year": "2024",
            "url": "http://example.com",
            "source": "arxiv",
            "topic_ids": ["t1"],
        })
        findings = store.regenerate_findings()
        assert "Citation validation" in findings or "citation" in findings.lower()

    def test_findings_full_structure(self, store):
        """Full findings output has all expected top-level sections."""
        store.add_topic(Topic(
            id="t1", title="Architecture",
            coverage=0.7, source_count=5,
        ))
        store.write_summary("t1", "Architecture findings with detail.", last_source_id=5)

        idx = store.load_index()
        idx.meta.total_sources = 5
        idx.meta.avg_coverage = 0.7
        idx.meta.iterations = 3
        store.save_index(idx)

        findings = store.regenerate_findings()

        # Check structural elements
        assert "# Research Findings" in findings
        assert "## Architecture" in findings
        assert "Coverage:" in findings
        assert "Confidence:" in findings
        assert "Sources" in findings
        assert "Generated:" in findings
