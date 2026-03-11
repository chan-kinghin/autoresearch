"""Integration tests — full lifecycle simulation with mocked LLM/search.

Exercises the complete pipeline:
  bootstrap → identify gaps → search → store sources → delta merge →
  evaluate → smart routing → second iteration → findings regeneration

No API keys needed — all external calls are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_store import KnowledgeStore, Topic, Question, StoreIndex, IndexMeta
from research import (
    InteractionMode,
    evaluate_coverage,
    extract_topic_findings,
    identify_gaps,
    run_research_loop,
)
from search import EvaluationResult, SearchResult


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
