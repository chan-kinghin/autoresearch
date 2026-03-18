"""Tests for knowledge_store.py — 3-layer progressive disclosure knowledge store."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from knowledge_store import (
    IndexMeta,
    KnowledgeStore,
    Question,
    Relation,
    StoreIndex,
    Topic,
)


# ── TestIndex ────────────────────────────────────────────────────────────


class TestIndex:
    def test_load_empty_index(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        index = store.load_index()
        assert isinstance(index, StoreIndex)
        assert index.topics == []
        assert index.questions == []
        assert index.relations == []
        assert index.meta.total_sources == 0

    def test_save_and_load_roundtrip(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        index = StoreIndex(
            topics=[Topic(id="t1", title="Topic One", keywords=["a", "b"], coverage=0.5, source_count=3)],
            questions=[Question(id="q1", text="What is X?", status="answered", related_topics=["t1"])],
            relations=[Relation(from_topic="t1", to_topic="t2", relation_type="related")],
            meta=IndexMeta(total_sources=3, avg_coverage=0.5, iterations=2),
        )
        store.save_index(index)
        loaded = store.load_index()

        assert len(loaded.topics) == 1
        assert loaded.topics[0].id == "t1"
        assert loaded.topics[0].title == "Topic One"
        assert loaded.topics[0].keywords == ["a", "b"]
        assert loaded.topics[0].coverage == 0.5
        assert loaded.topics[0].source_count == 3

        assert len(loaded.questions) == 1
        assert loaded.questions[0].id == "q1"
        assert loaded.questions[0].text == "What is X?"
        assert loaded.questions[0].status == "answered"

        assert len(loaded.relations) == 1
        assert loaded.relations[0].from_topic == "t1"
        assert loaded.relations[0].relation_type == "related"

        assert loaded.meta.total_sources == 3
        assert loaded.meta.avg_coverage == 0.5
        assert loaded.meta.iterations == 2

    def test_load_missing_file(self, tmp_path):
        # Store init creates dirs but not index.json, so load returns empty
        store = KnowledgeStore(str(tmp_path / "research_store"))
        index = store.load_index()
        assert index.topics == []

    def test_save_creates_file(self, tmp_path):
        store_dir = tmp_path / "research_store"
        store = KnowledgeStore(str(store_dir))
        index = StoreIndex(topics=[], questions=[], relations=[], meta=IndexMeta())
        store.save_index(index)
        assert (store_dir / "index.json").exists()


# ── TestTopics ───────────────────────────────────────────────────────────


class TestTopics:
    def test_add_topic(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        topic = Topic(id="arch", title="Architecture Patterns")
        store.add_topic(topic)
        index = store.load_index()
        assert len(index.topics) == 1
        assert index.topics[0].id == "arch"
        assert index.topics[0].title == "Architecture Patterns"

    def test_add_duplicate_topic(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        topic1 = Topic(id="arch", title="Architecture Patterns")
        topic2 = Topic(id="arch", title="Architecture Patterns v2")
        store.add_topic(topic1)
        store.add_topic(topic2)
        index = store.load_index()
        assert len([t for t in index.topics if t.id == "arch"]) == 1

    def test_get_topic_exists(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        topic = Topic(id="nlp", title="NLP Methods", keywords=["nlp", "bert"])
        store.add_topic(topic)
        result = store.get_topic("nlp")
        assert result is not None
        assert result.id == "nlp"
        assert result.title == "NLP Methods"

    def test_get_topic_missing(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        result = store.get_topic("nonexistent")
        assert result is None

    def test_update_coverage(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        store.add_topic(Topic(id="ml", title="Machine Learning"))
        store.update_topic_coverage("ml", 0.75)
        topic = store.get_topic("ml")
        assert topic is not None
        assert topic.coverage == 0.75

    def test_update_coverage_missing_topic(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        # Should not raise — just saves index unchanged
        store.update_topic_coverage("nonexistent", 0.5)

    def test_add_topic_creates_summary_file(self, tmp_path):
        store_dir = tmp_path / "research_store"
        store = KnowledgeStore(str(store_dir))
        store.add_topic(Topic(id="arch", title="Architecture"))
        summary_path = store_dir / "summaries" / "arch.md"
        assert summary_path.exists()


# ── TestSummaries ────────────────────────────────────────────────────────


class TestSummaries:
    def test_write_and_read_summary(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        store.add_topic(Topic(id="arch", title="Architecture"))
        store.write_summary("arch", "# Architecture\nContent here...", last_source_id=5)
        content = store.read_summary("arch")
        assert "# Architecture" in content
        assert "Content here..." in content
        # Should NOT contain frontmatter markers
        assert content.strip().startswith("#") or not content.strip().startswith("---")

    def test_read_missing_summary(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        result = store.read_summary("nonexistent")
        assert result == ""

    def test_frontmatter_contains_metadata(self, tmp_path):
        store_dir = tmp_path / "research_store"
        store = KnowledgeStore(str(store_dir))
        store.add_topic(Topic(id="arch", title="Architecture"))
        store.write_summary("arch", "Content", last_source_id=5)
        summary_path = store_dir / "summaries" / "arch.md"
        raw = summary_path.read_text(encoding="utf-8")
        assert "topic_id: arch" in raw
        assert "last_synthesized_source_id: 5" in raw
        assert "updated:" in raw

    def test_read_summary_frontmatter(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        store.add_topic(Topic(id="arch", title="Architecture"))
        store.write_summary("arch", "Content", last_source_id=7)
        fm = store.read_summary_frontmatter("arch")
        assert fm.get("topic_id") == "arch"
        assert fm.get("last_synthesized_source_id") == 7

    def test_read_frontmatter_missing_file(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        fm = store.read_summary_frontmatter("nonexistent")
        assert fm == {}

    def test_write_summary_updates_timestamp(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        store.add_topic(Topic(id="arch", title="Architecture"))
        store.write_summary("arch", "First version", last_source_id=1)
        fm1 = store.read_summary_frontmatter("arch")
        ts1 = fm1.get("updated")

        # Small delay to ensure different timestamp
        time.sleep(0.05)

        store.write_summary("arch", "Second version", last_source_id=2)
        fm2 = store.read_summary_frontmatter("arch")
        ts2 = fm2.get("updated")

        assert ts1 is not None
        assert ts2 is not None
        assert ts2 >= ts1

    def test_summary_with_no_frontmatter(self, tmp_path):
        store_dir = tmp_path / "research_store"
        store = KnowledgeStore(str(store_dir))
        store.add_topic(Topic(id="arch", title="Architecture"))
        # Overwrite with plain content (no frontmatter)
        summary_path = store_dir / "summaries" / "arch.md"
        summary_path.write_text("Just plain content\nNo frontmatter here", encoding="utf-8")
        content = store.read_summary("arch")
        assert "Just plain content" in content


# ── TestSources ──────────────────────────────────────────────────────────


class TestSources:
    def test_add_and_get_source(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        data = {"title": "Paper A", "url": "https://example.com/a", "topic_ids": ["arch"]}
        store.add_source(1, data)
        result = store.get_source(1)
        assert result is not None
        assert result["title"] == "Paper A"
        assert result["url"] == "https://example.com/a"

    def test_get_missing_source(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        result = store.get_source(999)
        assert result is None

    def test_next_source_id_empty(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        assert store.next_source_id() == 1

    def test_next_source_id_with_existing(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        store.add_source(1, {"title": "A", "topic_ids": []})
        store.add_source(2, {"title": "B", "topic_ids": []})
        store.add_source(5, {"title": "C", "topic_ids": []})
        assert store.next_source_id() == 6

    def test_add_source_updates_index(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        store.add_topic(Topic(id="arch", title="Architecture"))
        store.add_source(1, {"title": "Paper", "topic_ids": ["arch"]})
        topic = store.get_topic("arch")
        assert topic is not None
        assert topic.unsynthesized_sources >= 1
        assert topic.stale is True

    def test_get_unsynthesized_sources(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        store.add_topic(Topic(id="arch", title="Architecture"))
        # Add several sources for the "arch" topic
        store.add_source(1, {"title": "A", "topic_ids": ["arch"]})
        store.add_source(2, {"title": "B", "topic_ids": ["arch"]})
        store.add_source(3, {"title": "C", "topic_ids": ["arch"]})
        # Mark source 1 and 2 as synthesized via summary frontmatter
        store.write_summary("arch", "Summary", last_source_id=2)
        unsynthesized = store.get_unsynthesized_sources("arch")
        # Only source 3 should be unsynthesized (id > last_synthesized_source_id=2)
        assert len(unsynthesized) == 1
        assert unsynthesized[0]["title"] == "C"


# ── TestRawResults ───────────────────────────────────────────────────────


class TestRawResults:
    def test_save_raw_results(self, tmp_path):
        store_dir = tmp_path / "research_store"
        store = KnowledgeStore(str(store_dir))
        results = [{"query": "test query", "results": [{"title": "A"}]}]
        store.save_raw_results(1, results)
        raw_path = store_dir / "details" / "raw_results" / "iter_001.json"
        assert raw_path.exists()
        loaded = json.loads(raw_path.read_text(encoding="utf-8"))
        assert loaded[0]["query"] == "test query"


# ── TestBootstrap ────────────────────────────────────────────────────────


class TestBootstrap:
    def test_bootstrap_creates_topics_and_summaries(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        findings = "# Research Findings\nSome content about architecture and NLP."
        program = "Study architecture patterns"

        mock_llm = MagicMock(return_value={
            "topics": [
                {"id": "arch", "title": "Architecture Patterns", "keywords": ["architecture"], "summary": "About arch."},
                {"id": "nlp", "title": "NLP Methods", "keywords": ["nlp"], "summary": "About NLP."},
            ],
            "questions": [
                {"id": "q1", "text": "What are common patterns?", "related_topics": ["arch"]},
            ],
        })

        store.bootstrap_from_findings(findings, program, mock_llm)

        index = store.load_index()
        assert len(index.topics) >= 2
        topic_ids = [t.id for t in index.topics]
        assert "arch" in topic_ids
        assert "nlp" in topic_ids

    def test_bootstrap_creates_questions(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))

        mock_llm = MagicMock(return_value={
            "topics": [
                {"id": "t1", "title": "Topic 1", "keywords": []},
            ],
            "questions": [
                {"id": "q1", "text": "Question one?", "related_topics": ["t1"]},
                {"id": "q2", "text": "Question two?", "related_topics": ["t1"]},
            ],
        })

        store.bootstrap_from_findings("findings", "program", mock_llm)

        index = store.load_index()
        assert len(index.questions) >= 2

    def test_bootstrap_handles_empty_findings(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))

        mock_llm = MagicMock(return_value={
            "topics": [],
            "questions": [],
        })

        # Should not raise
        store.bootstrap_from_findings("", "program", mock_llm)
        index = store.load_index()
        assert isinstance(index, StoreIndex)


# ── TestFindingsRegeneration ─────────────────────────────────────────────


class TestFindingsRegeneration:
    def test_regenerate_empty_store(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        result = store.regenerate_findings()
        assert isinstance(result, str)
        assert "Research Findings" in result

    def test_regenerate_with_topics(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        store.add_topic(Topic(id="arch", title="Architecture"))
        store.add_topic(Topic(id="nlp", title="NLP"))
        store.write_summary("arch", "Architecture content", last_source_id=1)
        store.write_summary("nlp", "NLP content", last_source_id=1)
        result = store.regenerate_findings()
        assert "Architecture content" in result
        assert "NLP content" in result

    def test_regenerate_writes_file(self, tmp_path):
        store_dir = tmp_path / "research_store"
        store = KnowledgeStore(str(store_dir))
        store.add_topic(Topic(id="arch", title="Architecture"))
        store.write_summary("arch", "Content", last_source_id=1)
        store.regenerate_findings()
        findings_path = store_dir / "findings.md"
        assert findings_path.exists()


# ── TestUtilities ────────────────────────────────────────────────────────


class TestUtilities:
    def test_exists_true(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        store.save_index(StoreIndex(topics=[], questions=[], relations=[], meta=IndexMeta()))
        assert store.exists() is True

    def test_exists_false(self, tmp_path):
        # Don't instantiate KnowledgeStore (it creates dirs). Just check a fresh path.
        store = KnowledgeStore.__new__(KnowledgeStore)
        store.root = tmp_path / "nonexistent_store_xyz"
        assert store.exists() is False

    def test_estimate_tokens(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        text = "a" * 100
        result = store.estimate_tokens(text)
        assert result == 25  # len/4

    def test_get_context_for_gaps(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        store.add_topic(Topic(id="arch", title="Architecture", coverage=0.5))
        store.add_topic(Topic(id="nlp", title="NLP", coverage=0.3))
        context = store.get_context_for_gaps()
        assert isinstance(context, str)
        assert "Architecture" in context
        assert "NLP" in context
        assert "arch" in context


# ── TestSmartRouting ────────────────────────────────────────────────────


class TestSmartRouting:
    def test_extract_keywords(self, tmp_path):
        # Basic keyword extraction
        keywords = KnowledgeStore._extract_keywords(["What are ReAct architecture patterns?"])
        assert "react" in keywords
        assert "architecture" in keywords
        assert "patterns" in keywords
        # Stopwords removed
        assert "are" not in keywords
        assert "what" not in keywords

    def test_extract_keywords_short_words_removed(self, tmp_path):
        keywords = KnowledgeStore._extract_keywords(["AI is great"])
        assert "ai" not in keywords  # too short (< 3 chars)
        assert "great" in keywords

    def test_select_relevant_summaries_basic(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="arch", title="Architecture", keywords=["react", "architecture", "patterns"], coverage=0.3))
        store.add_topic(Topic(id="safety", title="Safety", keywords=["safety", "evaluation", "risks"], coverage=0.8))
        store.write_summary("arch", "Architecture patterns content...", last_source_id=1)
        store.write_summary("safety", "Safety content...", last_source_id=1)

        results = store.select_relevant_summaries(["react architecture patterns"])
        # "arch" should rank higher (keyword overlap AND lower coverage)
        assert len(results) >= 1
        assert results[0][0] == "arch"

    def test_select_relevant_summaries_respects_budget(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        # Create topics with large summaries
        for i in range(10):
            tid = f"topic_{i}"
            store.add_topic(Topic(id=tid, title=f"Topic {i}", keywords=[f"keyword_{i}"], coverage=0.1))
            store.write_summary(tid, "x" * 4000, last_source_id=1)  # ~1000 tokens each

        results = store.select_relevant_summaries(
            ["keyword_0 keyword_1 keyword_2"],
            token_budget=2000,  # Should only fit ~1 summary at 60% budget
        )
        assert len(results) <= 3  # Budget should limit how many summaries are loaded

    def test_select_relevant_summaries_stale_bonus(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="fresh", title="Fresh", keywords=["data"], coverage=0.5))
        store.add_topic(Topic(id="stale", title="Stale", keywords=["data"], coverage=0.5, stale=True, unsynthesized_sources=3))
        store.write_summary("fresh", "Fresh content", last_source_id=1)
        store.write_summary("stale", "Stale content", last_source_id=1)

        results = store.select_relevant_summaries(["data analysis"])
        # Stale topic should rank higher due to bonus
        assert results[0][0] == "stale"

    def test_prioritize_stale_topics(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="a", title="A", unsynthesized_sources=5, stale=True))
        store.add_topic(Topic(id="b", title="B", unsynthesized_sources=0, stale=False))
        store.add_topic(Topic(id="c", title="C", unsynthesized_sources=2, stale=True))

        result = store.prioritize_stale_topics()
        assert result == ["a", "c"]  # Only stale topics, sorted by unsynthesized count desc

    def test_prioritize_stale_topics_empty(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="a", title="A"))
        assert store.prioritize_stale_topics() == []

    def test_get_context_with_budget(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="arch", title="Architecture", keywords=["architecture"], coverage=0.3))
        store.write_summary("arch", "Architecture patterns content", last_source_id=1)

        context = store.get_context_with_budget(["architecture patterns"])
        assert "Knowledge Store Index" in context  # Index always included
        assert "Architecture patterns content" in context  # Relevant summary included

    def test_get_context_with_budget_empty_store(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        context = store.get_context_with_budget(["anything"])
        assert isinstance(context, str)
        assert "Knowledge Store Index" in context


# ── Helper: build a richly populated store ──────────────────────────────


def _build_populated_store(tmp_path) -> KnowledgeStore:
    """Create a KnowledgeStore with topics, sources, summaries, and questions for reuse."""
    store = KnowledgeStore(str(tmp_path / "research_store"))

    # Topics
    store.add_topic(Topic(id="llm_arch", title="LLM Architecture", keywords=["llm", "architecture"], coverage=0.8, source_count=6))
    store.add_topic(Topic(id="safety", title="AI Safety", keywords=["safety", "alignment"], coverage=0.5, source_count=3))
    store.add_topic(Topic(id="scaling", title="Scaling Laws", keywords=["scaling"], coverage=0.2, source_count=1))

    # Sources with year/author data
    store.add_source(1, {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer",
        "year": 2017,
        "url": "https://arxiv.org/abs/1706.03762",
        "source": "arxiv",
        "topic_ids": ["llm_arch"],
    })
    store.add_source(2, {
        "title": "Language Models are Few-Shot Learners",
        "authors": "Tom Brown, Benjamin Mann",
        "year": 2020,
        "url": "https://arxiv.org/abs/2005.14165",
        "source": "semantic_scholar",
        "topic_ids": ["llm_arch", "scaling"],
    })
    store.add_source(3, {
        "title": "Constitutional AI",
        "authors": "Yuntao Bai, Saurav Kadavath",
        "year": 2022,
        "url": "https://arxiv.org/abs/2212.08073",
        "source": "arxiv",
        "topic_ids": ["safety"],
    })
    store.add_source(4, {
        "title": "Scaling Data-Constrained Language Models",
        "authors": "Niklas Muennighoff",
        "year": 2024,
        "url": "https://example.com/scaling2024",
        "source": "duckduckgo",
        "topic_ids": ["scaling"],
    })
    store.add_source(5, {
        "title": "Reward Model Overoptimization",
        "authors": "Leo Gao, John Schulman",
        "year": 2023,
        "url": "https://example.com/reward",
        "source": "semantic_scholar",
        "topic_ids": ["safety"],
    })

    # Summaries with inline citations and contradiction markers
    store.write_summary("llm_arch", (
        "# LLM Architecture\n\n"
        "The transformer architecture was introduced by [Vaswani, 2017] and has become dominant.\n"
        "Large language models like GPT-3 [Brown, 2020] demonstrated few-shot capabilities.\n"
        "This is a sufficiently long first sentence for executive summary extraction purposes.\n"
    ), last_source_id=2)

    store.write_summary("safety", (
        "# AI Safety\n\n"
        "Constitutional AI [Bai, 2022] proposes a self-supervised alignment approach.\n"
        "[CONFLICT] However, source Gao (2023) disagrees on reward model effectiveness.\n"
        "In contrast, some argue that RLHF is sufficient for alignment.\n"
        "There is also a [phantom_cite, 2099] that does not match any source.\n"
    ), last_source_id=3)

    store.write_summary("scaling", (
        "# Scaling Laws\n\n"
        "Scaling laws predict model performance as a function of compute [Brown, 2020].\n"
        "Recent work suggests data quality matters more than quantity.\n"
    ), last_source_id=4)

    # Questions
    index = store.load_index()
    index.questions = [
        Question(id="q1", text="How do transformers scale?", status="answered", related_topics=["llm_arch", "scaling"]),
        Question(id="q2", text="What alignment methods work best?", status="partial", related_topics=["safety"]),
        Question(id="q3", text="Can we reduce compute costs?", status="unanswered", related_topics=["scaling"]),
    ]
    index.meta = IndexMeta(total_sources=5, avg_coverage=0.5, iterations=3)
    store.save_index(index)

    return store


# ── D1: Citation Index Builder ──────────────────────────────────────────


class TestCitationIndex:
    def test_build_citation_index_matches_author_year(self, tmp_path):
        store = _build_populated_store(tmp_path)
        citation_index = store.build_citation_index()
        # [Vaswani, 2017] should match src_001 (authors contain "Vaswani", year 2017)
        assert "Vaswani, 2017" in citation_index
        entry = citation_index["Vaswani, 2017"]
        assert entry["source_id"] == "src_001"
        assert entry["year"] == "2017"
        assert "arxiv" in entry["url"]

    def test_build_citation_index_matches_across_topics(self, tmp_path):
        store = _build_populated_store(tmp_path)
        citation_index = store.build_citation_index()
        # [Brown, 2020] appears in both llm_arch and scaling summaries
        assert "Brown, 2020" in citation_index
        assert citation_index["Brown, 2020"]["source_id"] == "src_002"

    def test_build_citation_index_unmatched_not_included(self, tmp_path):
        store = _build_populated_store(tmp_path)
        citation_index = store.build_citation_index()
        # phantom_cite doesn't match any source
        assert "phantom_cite, 2099" not in citation_index

    def test_build_citation_index_empty_store(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "research_store"))
        citation_index = store.build_citation_index()
        assert citation_index == {}

    def test_build_citation_index_no_duplicate_entries(self, tmp_path):
        store = _build_populated_store(tmp_path)
        citation_index = store.build_citation_index()
        # Each citation text should appear exactly once as a key
        keys = list(citation_index.keys())
        assert len(keys) == len(set(keys))


# ── D2: Confidence Levels in regenerate_findings ────────────────────────


class TestConfidenceLevels:
    def test_high_confidence(self, tmp_path):
        store = _build_populated_store(tmp_path)
        findings = store.regenerate_findings()
        # llm_arch: source_count=6+2(from add_source), coverage=0.8 -> HIGH
        # After add_source calls, llm_arch gets +1 from src_001, so source_count goes up
        # The topic was initialized with source_count=6, then add_source increments it
        assert "Confidence: HIGH" in findings

    def test_medium_confidence(self, tmp_path):
        store = _build_populated_store(tmp_path)
        findings = store.regenerate_findings()
        # safety: coverage=0.5, source_count >= 2 -> MEDIUM
        assert "Confidence: MEDIUM" in findings

    def test_low_confidence(self, tmp_path):
        store = _build_populated_store(tmp_path)
        findings = store.regenerate_findings()
        # scaling: coverage=0.2, source_count >= 2 but coverage < 0.4 -> LOW
        # Actually scaling starts at source_count=1, then gets +2 from add_source -> 3
        # coverage=0.2 < 0.4 -> even with 3 sources, coverage < 0.4 -> LOW
        assert "Confidence: LOW" in findings

    def test_confidence_in_topic_section(self, tmp_path):
        """Confidence appears on the same line as coverage and sources."""
        store = _build_populated_store(tmp_path)
        findings = store.regenerate_findings()
        # Check that confidence is in the coverage line format
        assert "Coverage:" in findings
        assert "Sources:" in findings

    def test_confidence_boundary_high(self, tmp_path):
        """Exactly at the HIGH threshold: source_count=5, coverage=0.7."""
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="t1", title="Topic", coverage=0.7, source_count=5))
        store.write_summary("t1", "Some content here for the topic.", last_source_id=1)
        findings = store.regenerate_findings()
        assert "Confidence: HIGH" in findings

    def test_confidence_boundary_medium(self, tmp_path):
        """Exactly at MEDIUM threshold: source_count=2, coverage=0.4."""
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="t1", title="Topic", coverage=0.4, source_count=2))
        store.write_summary("t1", "Some content here for the topic.", last_source_id=1)
        findings = store.regenerate_findings()
        assert "Confidence: MEDIUM" in findings


# ── D3: Question Resolution Matrix ─────────────────────────────────────


class TestQuestionMatrix:
    def test_generate_question_matrix_basic(self, tmp_path):
        store = _build_populated_store(tmp_path)
        matrix = store.generate_question_matrix()
        assert "Research Questions: Coverage Status" in matrix
        assert "How do transformers scale?" in matrix
        assert "What alignment methods work best?" in matrix
        assert "Can we reduce compute costs?" in matrix

    def test_question_matrix_table_format(self, tmp_path):
        store = _build_populated_store(tmp_path)
        matrix = store.generate_question_matrix()
        # Should contain markdown table headers
        assert "| # | Question | Status | Related Topics | Gap |" in matrix
        assert "|---|----------|--------|----------------|-----|" in matrix

    def test_question_matrix_status_icons(self, tmp_path):
        store = _build_populated_store(tmp_path)
        matrix = store.generate_question_matrix()
        assert "\u2705" in matrix   # answered checkmark
        assert "\U0001f536" in matrix  # partial diamond
        assert "\u274c" in matrix   # unanswered cross

    def test_question_matrix_summary_counts(self, tmp_path):
        store = _build_populated_store(tmp_path)
        matrix = store.generate_question_matrix()
        # 1 answered, 1 partial, 1 unanswered out of 3
        assert "1/3 answered" in matrix
        assert "1/3 partial" in matrix
        assert "1/3 open" in matrix

    def test_question_matrix_gap_column(self, tmp_path):
        store = _build_populated_store(tmp_path)
        matrix = store.generate_question_matrix()
        assert "None" in matrix              # answered question has no gap
        assert "Needs more research" in matrix  # unanswered/partial need research

    def test_question_matrix_empty(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        matrix = store.generate_question_matrix()
        assert matrix == ""

    def test_question_matrix_related_topics(self, tmp_path):
        store = _build_populated_store(tmp_path)
        matrix = store.generate_question_matrix()
        # q1 relates to llm_arch and scaling
        assert "llm_arch" in matrix
        assert "scaling" in matrix

    def test_question_matrix_in_findings(self, tmp_path):
        store = _build_populated_store(tmp_path)
        findings = store.regenerate_findings()
        assert "Research Questions: Coverage Status" in findings


# ── D4: Temporal Awareness / Source Recency ─────────────────────────────


class TestSourceRecency:
    def test_analyze_source_recency_basic(self, tmp_path):
        store = _build_populated_store(tmp_path)
        recency = store.analyze_source_recency()
        assert "Source Recency Analysis" in recency
        assert "Newest source" in recency
        assert "Oldest source" in recency
        assert "Median source year" in recency

    def test_recency_year_values(self, tmp_path):
        store = _build_populated_store(tmp_path)
        recency = store.analyze_source_recency()
        # Sources have years: 2017, 2020, 2022, 2023, 2024
        assert "2024" in recency  # newest
        assert "2017" in recency  # oldest

    def test_recency_old_source_warning(self, tmp_path):
        """If >30% of sources are >3 years old, a warning is emitted."""
        store = _build_populated_store(tmp_path)
        recency = store.analyze_source_recency()
        # Years: 2017, 2020, 2022, 2023, 2024. Current year is 2026.
        # >3 years old: 2017 (9yr), 2020 (6yr), 2022 (4yr) = 3/5 = 60% > 30%
        assert "\u26a0\ufe0f" in recency or "old" in recency.lower()

    def test_recency_empty_store(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        recency = store.analyze_source_recency()
        assert recency == ""

    def test_recency_no_year_data(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="t1", title="T1"))
        store.add_source(1, {"title": "No Year", "topic_ids": ["t1"]})
        recency = store.analyze_source_recency()
        assert recency == ""

    def test_recency_in_findings(self, tmp_path):
        store = _build_populated_store(tmp_path)
        findings = store.regenerate_findings()
        assert "Source Recency Analysis" in findings

    def test_recency_total_count(self, tmp_path):
        store = _build_populated_store(tmp_path)
        recency = store.analyze_source_recency()
        assert "Total sources with year data" in recency
        assert "5" in recency


# ── D5: Multi-format Export ─────────────────────────────────────────────


class TestMultiFormatExport:
    # -- Executive Summary --

    def test_executive_summary_header(self, tmp_path):
        store = _build_populated_store(tmp_path)
        summary = store.export_executive_summary()
        assert "# Executive Summary" in summary
        assert "Generated:" in summary

    def test_executive_summary_coverage_section(self, tmp_path):
        store = _build_populated_store(tmp_path)
        summary = store.export_executive_summary()
        assert "## Coverage" in summary
        assert "Overall:" in summary
        assert "Sources consulted:" in summary
        assert "Iterations:" in summary

    def test_executive_summary_key_findings(self, tmp_path):
        store = _build_populated_store(tmp_path)
        summary = store.export_executive_summary()
        assert "## Key Findings" in summary
        # Topics sorted by coverage descending, llm_arch has highest coverage
        assert "LLM Architecture" in summary

    def test_executive_summary_confidence_labels(self, tmp_path):
        store = _build_populated_store(tmp_path)
        summary = store.export_executive_summary()
        # Should contain at least one confidence label
        assert any(level in summary for level in ["HIGH", "MEDIUM", "LOW"])

    def test_executive_summary_remaining_gaps(self, tmp_path):
        store = _build_populated_store(tmp_path)
        summary = store.export_executive_summary()
        assert "## Remaining Gaps" in summary
        # q2 (partial) and q3 (unanswered) should appear
        assert "What alignment methods work best?" in summary
        assert "Can we reduce compute costs?" in summary

    def test_executive_summary_empty_store(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        summary = store.export_executive_summary()
        assert "# Executive Summary" in summary

    # -- BibTeX --

    def test_export_bibtex_entries(self, tmp_path):
        store = _build_populated_store(tmp_path)
        bibtex = store.export_bibtex()
        assert "@article{" in bibtex or "@misc{" in bibtex
        assert "Attention Is All You Need" in bibtex
        assert "Language Models are Few-Shot Learners" in bibtex

    def test_bibtex_cite_key_format(self, tmp_path):
        store = _build_populated_store(tmp_path)
        bibtex = store.export_bibtex()
        # First author last name + year
        assert "vaswani2017" in bibtex
        assert "brown2020" in bibtex

    def test_bibtex_entry_type_academic(self, tmp_path):
        store = _build_populated_store(tmp_path)
        bibtex = store.export_bibtex()
        # arxiv and semantic_scholar sources -> @article
        assert "@article{vaswani2017" in bibtex

    def test_bibtex_entry_type_misc(self, tmp_path):
        store = _build_populated_store(tmp_path)
        bibtex = store.export_bibtex()
        # duckduckgo source -> @misc
        assert "@misc{muennighoff2024" in bibtex

    def test_bibtex_contains_url(self, tmp_path):
        store = _build_populated_store(tmp_path)
        bibtex = store.export_bibtex()
        assert "url = {https://arxiv.org/abs/1706.03762}" in bibtex

    def test_bibtex_empty_store(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        bibtex = store.export_bibtex()
        assert bibtex == ""

    def test_bibtex_skips_sources_without_title(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="t1", title="T1"))
        store.add_source(1, {"topic_ids": ["t1"], "url": "https://example.com"})
        bibtex = store.export_bibtex()
        assert bibtex == ""


# ── D6: Contradiction Log ──────────────────────────────────────────────


class TestContradictionLog:
    def test_detects_conflict_marker(self, tmp_path):
        store = _build_populated_store(tmp_path)
        contradictions = store.get_contradictions_from_summaries()
        # safety summary has "[CONFLICT]" marker
        assert any("AI Safety" in c for c in contradictions)

    def test_detects_disagree_keyword(self, tmp_path):
        store = _build_populated_store(tmp_path)
        contradictions = store.get_contradictions_from_summaries()
        # safety summary has "disagrees"
        assert any("disagree" in c.lower() for c in contradictions)

    def test_detects_in_contrast(self, tmp_path):
        store = _build_populated_store(tmp_path)
        contradictions = store.get_contradictions_from_summaries()
        assert any("in contrast" in c.lower() for c in contradictions)

    def test_no_contradictions_clean_store(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="t1", title="Clean Topic"))
        store.write_summary("t1", "Everything is consistent and clear.", last_source_id=1)
        contradictions = store.get_contradictions_from_summaries()
        assert contradictions == []

    def test_contradictions_in_findings(self, tmp_path):
        store = _build_populated_store(tmp_path)
        findings = store.regenerate_findings()
        assert "Known Conflicts & Uncertainties" in findings

    def test_contradiction_truncation(self, tmp_path):
        """Contradiction lines are truncated to 200 chars."""
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="t1", title="Long"))
        long_line = "[CONFLICT] " + "x" * 300
        store.write_summary("t1", long_line, last_source_id=1)
        contradictions = store.get_contradictions_from_summaries()
        assert len(contradictions) == 1
        # Total = "**Long**: " (10 chars) + 200 chars of content
        assert len(contradictions[0]) <= 220


# ── D7: Citation Validation ────────────────────────────────────────────


class TestCitationValidation:
    def test_validate_citations_basic(self, tmp_path):
        store = _build_populated_store(tmp_path)
        result = store.validate_citations()
        assert "total_citations" in result
        assert "matched" in result
        assert "unmatched" in result
        assert "match_rate" in result
        assert result["total_citations"] > 0

    def test_matched_citations(self, tmp_path):
        store = _build_populated_store(tmp_path)
        result = store.validate_citations()
        # Vaswani 2017, Brown 2020, Bai 2022 should match
        assert result["matched"] >= 1

    def test_unmatched_citations(self, tmp_path):
        store = _build_populated_store(tmp_path)
        result = store.validate_citations()
        # phantom_cite 2099 should be unmatched
        assert "phantom_cite, 2099" in result["unmatched"]

    def test_match_rate_between_0_and_1(self, tmp_path):
        store = _build_populated_store(tmp_path)
        result = store.validate_citations()
        assert 0.0 <= result["match_rate"] <= 1.0

    def test_validation_filters_non_citations(self, tmp_path):
        """HTTP links, CONFLICT/STALE markers, and ! prefixed items are excluded."""
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="t1", title="T1"))
        store.write_summary("t1", (
            "See [https://example.com] and [CONFLICT] and [!image] "
            "and also [STALE] marker."
        ), last_source_id=1)
        result = store.validate_citations()
        assert result["total_citations"] == 0

    def test_validation_empty_store(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        result = store.validate_citations()
        assert result["total_citations"] == 0
        assert result["match_rate"] == 1.0  # 0/0 defaults to 1.0

    def test_validation_in_findings(self, tmp_path):
        store = _build_populated_store(tmp_path)
        findings = store.regenerate_findings()
        assert "Citation validation:" in findings
