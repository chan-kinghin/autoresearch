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
