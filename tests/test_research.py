"""Tests for research.py — autonomous research loop."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from research import (
    CheckpointResult,
    InteractionMode,
    extract_topic_findings,
    identify_gaps,
    main,
    prompt_checkpoint,
    read_file,
    run_research_loop,
    write_file,
)
from search import EvaluationResult, IterationLog, SearchResult


# ── read_file ─────────────────────────────────────────────────────────────


class TestReadFile:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert read_file(str(f)) == "hello world"

    def test_returns_empty_for_missing_file(self, tmp_path):
        assert read_file(str(tmp_path / "nonexistent.txt")) == ""

    def test_reads_utf8_content(self, tmp_path):
        f = tmp_path / "unicode.txt"
        f.write_text("你好世界", encoding="utf-8")
        assert read_file(str(f)) == "你好世界"


# ── write_file ────────────────────────────────────────────────────────────


class TestWriteFile:
    def test_creates_and_writes(self, tmp_path):
        path = str(tmp_path / "output.txt")
        write_file(path, "written content")
        assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "written content"

    def test_overwrites_existing(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old", encoding="utf-8")
        write_file(str(f), "new")
        assert f.read_text(encoding="utf-8") == "new"


# ── identify_gaps ─────────────────────────────────────────────────────────


class TestIdentifyGaps:
    def _make_mock_store(self, topics=None, summaries=None):
        """Create a mock KnowledgeStore with configurable behavior."""
        store = MagicMock()
        from knowledge_store import IndexMeta, StoreIndex, Topic

        if topics is None:
            topics = []
        index = StoreIndex(topics=topics, meta=IndexMeta())
        store.load_index.return_value = index
        store.get_context_for_gaps.return_value = "## Knowledge Store Index\nSources: 0 | Avg coverage: 0.00 | Iterations: 0"
        store.get_context_with_budget.return_value = "## Knowledge Store Index\nSources: 0 | Avg coverage: 0.00 | Iterations: 0"
        store.read_summary.return_value = ""

        if summaries:
            def read_summary_side_effect(topic_id):
                return summaries.get(topic_id, "")
            store.read_summary.side_effect = read_summary_side_effect

        return store

    @patch("research.llm_json")
    def test_returns_llm_json_result(self, mock_llm_json):
        expected = {
            "gaps": ["What is X?"],
            "search_plan": [{"query": "X", "sources": ["arxiv"], "rationale": "find X", "topic_id": "new"}],
            "new_topics": [],
        }
        mock_llm_json.return_value = expected

        # Use a store with at least one topic so it doesn't show "first iteration"
        from knowledge_store import Topic
        store = self._make_mock_store(topics=[Topic(id="t1", title="T1", coverage=0.7)])
        result = identify_gaps("program text", store)

        assert result == expected
        mock_llm_json.assert_called_once()
        # Prompt should contain research program and index context
        prompt_arg = mock_llm_json.call_args[0][0]
        assert "program text" in prompt_arg
        assert "Knowledge Store Index" in prompt_arg

    @patch("research.llm_json")
    def test_includes_human_guidance(self, mock_llm_json):
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}

        store = self._make_mock_store()
        identify_gaps("program", store, human_guidance="focus on NLP")

        prompt_arg = mock_llm_json.call_args[0][0]
        assert "focus on NLP" in prompt_arg
        assert "Human Guidance" in prompt_arg

    @patch("research.llm_json")
    def test_no_guidance_block_when_empty(self, mock_llm_json):
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}

        store = self._make_mock_store()
        identify_gaps("program", store, human_guidance="")

        prompt_arg = mock_llm_json.call_args[0][0]
        assert "Human Guidance" not in prompt_arg

    @patch("research.llm_json")
    def test_empty_store_shows_first_iteration(self, mock_llm_json):
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}

        store = self._make_mock_store()
        # No topics means first iteration message
        identify_gaps("program", store)

        prompt_arg = mock_llm_json.call_args[0][0]
        assert "first iteration" in prompt_arg

    @patch("research.llm_json")
    def test_uses_budget_aware_context(self, mock_llm_json):
        """Verify identify_gaps calls store.get_context_with_budget()."""
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}

        from knowledge_store import Topic
        topics = [
            Topic(id="topic_a", title="Topic A", coverage=0.3),
            Topic(id="topic_b", title="Topic B", coverage=0.8),
        ]
        store = self._make_mock_store(topics=topics)
        store.get_context_with_budget.return_value = (
            "## Knowledge Store Index\nBudget-aware context\n"
            "## Relevant Topic Summaries\n### Topic A (topic_a)\nSummary for A"
        )

        identify_gaps("program", store)

        # get_context_with_budget should have been called
        store.get_context_with_budget.assert_called_once()
        prompt_arg = mock_llm_json.call_args[0][0]
        assert "Budget-aware context" in prompt_arg

    @patch("research.llm_json")
    def test_includes_existing_topic_ids(self, mock_llm_json):
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}

        from knowledge_store import Topic
        topics = [Topic(id="ml_models", title="ML Models", coverage=0.5)]
        store = self._make_mock_store(topics=topics)

        identify_gaps("program", store)

        prompt_arg = mock_llm_json.call_args[0][0]
        assert "ml_models" in prompt_arg
        assert "Existing topic IDs" in prompt_arg


# ── extract_topic_findings ────────────────────────────────────────────────


class TestExtractTopicFindings:
    @patch("research.llm_call")
    def test_formats_search_results_and_calls_llm(self, mock_llm_call):
        mock_llm_call.return_value = "# Updated Findings"

        results = [
            SearchResult(
                title="Paper A",
                url="https://example.com/a",
                snippet="snippet A",
                source="arxiv",
                authors="Alice",
                year="2024",
            ),
            SearchResult(
                title="Paper B",
                url="https://example.com/b",
                snippet="snippet B",
                source="duckduckgo",
            ),
        ]

        store = MagicMock()
        store.read_summary.return_value = "existing summary text"
        from knowledge_store import Topic
        store.get_topic.return_value = Topic(id="ml_models", title="ML Models")

        output = extract_topic_findings(results, "research program", store, "ml_models")

        assert output == "# Updated Findings"
        mock_llm_call.assert_called_once()
        prompt_arg = mock_llm_call.call_args[0][0]
        assert "Paper A" in prompt_arg
        assert "Alice" in prompt_arg
        assert "2024" in prompt_arg
        assert "Paper B" in prompt_arg
        assert "research program" in prompt_arg
        assert "existing summary text" in prompt_arg
        assert "ml_models" in prompt_arg

    @patch("research.llm_call")
    def test_uses_full_text_over_snippet(self, mock_llm_call):
        mock_llm_call.return_value = "findings"

        results = [
            SearchResult(
                title="Paper",
                url="https://example.com",
                snippet="short",
                source="metaso",
                full_text="full detailed text content",
            ),
        ]

        store = MagicMock()
        store.read_summary.return_value = ""
        store.get_topic.return_value = None

        extract_topic_findings(results, "prog", store, "topic1")

        prompt_arg = mock_llm_call.call_args[0][0]
        assert "full detailed text content" in prompt_arg

    @patch("research.llm_call")
    def test_truncates_content_to_2000_chars(self, mock_llm_call):
        mock_llm_call.return_value = "findings"

        long_text = "x" * 5000
        results = [
            SearchResult(
                title="Paper",
                url="https://example.com",
                snippet="",
                source="arxiv",
                full_text=long_text,
            ),
        ]

        store = MagicMock()
        store.read_summary.return_value = ""
        store.get_topic.return_value = None

        extract_topic_findings(results, "prog", store, "topic1")

        prompt_arg = mock_llm_call.call_args[0][0]
        # full_text is truncated to 2000 chars via content[:2000]
        assert "x" * 2000 in prompt_arg
        assert "x" * 2001 not in prompt_arg

    @patch("research.llm_call")
    def test_empty_summary_shows_first_synthesis(self, mock_llm_call):
        mock_llm_call.return_value = "findings"

        store = MagicMock()
        store.read_summary.return_value = "  "
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog",
            store,
            "topic1",
        )

        prompt_arg = mock_llm_call.call_args[0][0]
        assert "first synthesis" in prompt_arg

    @patch("research.llm_call")
    def test_max_tokens_passed(self, mock_llm_call):
        mock_llm_call.return_value = "findings"

        store = MagicMock()
        store.read_summary.return_value = "existing"
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog",
            store,
            "topic1",
        )

        assert mock_llm_call.call_args[1]["max_tokens"] == 8192

    @patch("research.llm_call")
    def test_delta_merge_instruction_in_prompt(self, mock_llm_call):
        mock_llm_call.return_value = "findings"

        store = MagicMock()
        store.read_summary.return_value = "existing"
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog",
            store,
            "topic1",
        )

        prompt_arg = mock_llm_call.call_args[0][0]
        assert "DELTA MERGE" in prompt_arg


# ── prompt_checkpoint ─────────────────────────────────────────────────────


class TestPromptCheckpoint:
    def test_auto_mode_returns_approved_immediately(self):
        result = prompt_checkpoint(InteractionMode.AUTO, "Test", ["line1"])
        assert result.approved is True
        assert result.guidance == ""

    @patch("builtins.input", return_value="")
    def test_supervised_continue_on_enter(self, mock_input):
        result = prompt_checkpoint(InteractionMode.SUPERVISED, "Test", ["line1"])
        assert result.approved is True

    @patch("builtins.input", return_value="q")
    def test_supervised_stop_on_q(self, mock_input):
        result = prompt_checkpoint(InteractionMode.SUPERVISED, "Test", ["line1"])
        assert result.approved is False

    @patch("builtins.input", side_effect=EOFError)
    def test_supervised_stop_on_eof(self, mock_input):
        result = prompt_checkpoint(InteractionMode.SUPERVISED, "Test", ["line1"])
        assert result.approved is False

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_supervised_stop_on_keyboard_interrupt(self, mock_input):
        result = prompt_checkpoint(InteractionMode.SUPERVISED, "Test", ["line1"])
        assert result.approved is False

    @patch("builtins.input", return_value="")
    def test_interactive_continue_on_enter(self, mock_input):
        result = prompt_checkpoint(InteractionMode.INTERACTIVE, "Test", ["line1"])
        assert result.approved is True
        assert result.guidance == ""

    @patch("builtins.input", return_value="q")
    def test_interactive_stop_on_q(self, mock_input):
        result = prompt_checkpoint(InteractionMode.INTERACTIVE, "Test", ["line1"])
        assert result.approved is False

    @patch("builtins.input", return_value="focus on transformers")
    def test_interactive_returns_guidance(self, mock_input):
        result = prompt_checkpoint(InteractionMode.INTERACTIVE, "Test", ["line1"])
        assert result.approved is True
        assert result.guidance == "focus on transformers"

    @patch("builtins.input", side_effect=EOFError)
    def test_interactive_stop_on_eof(self, mock_input):
        result = prompt_checkpoint(InteractionMode.INTERACTIVE, "Test", ["line1"])
        assert result.approved is False


# ── evaluate_coverage ─────────────────────────────────────────────────────


class TestEvaluateCoverage:
    @patch("research.llm_json")
    def test_uses_store_context(self, mock_llm_json):
        from research import evaluate_coverage

        mock_llm_json.return_value = {
            "coverage_score": 0.7,
            "questions_answered": ["Q1"],
            "questions_remaining": ["Q2"],
            "gaps": ["gap1"],
            "suggested_queries": ["query1"],
            "topic_scores": {"topic_a": 0.8},
        }

        store = MagicMock()
        store.get_context_for_gaps.return_value = "## Knowledge Store Index\nTopics info"
        from knowledge_store import IndexMeta, StoreIndex, Topic
        store.load_index.return_value = StoreIndex(
            topics=[Topic(id="topic_a", title="Topic A", coverage=0.3)],
            meta=IndexMeta(),
        )
        store.read_summary.return_value = "Some summary"

        result = evaluate_coverage("program", store)

        assert result.coverage_score == 0.7
        assert result.questions_answered == ["Q1"]
        assert result.gaps == ["gap1"]
        store.get_context_for_gaps.assert_called_once()
        # Should update per-topic coverage
        store.update_topic_coverage.assert_called_once_with("topic_a", 0.8)

    @patch("research.llm_json")
    def test_spot_checks_lowest_coverage(self, mock_llm_json):
        from research import evaluate_coverage

        mock_llm_json.return_value = {
            "coverage_score": 0.5,
            "topic_scores": {},
        }

        store = MagicMock()
        store.get_context_for_gaps.return_value = "index context"
        from knowledge_store import IndexMeta, StoreIndex, Topic
        topics = [
            Topic(id="high", title="High", coverage=0.9),
            Topic(id="low", title="Low", coverage=0.1),
            Topic(id="mid", title="Mid", coverage=0.5),
        ]
        store.load_index.return_value = StoreIndex(topics=topics, meta=IndexMeta())

        def read_summary_side_effect(topic_id):
            return {"high": "High summary", "low": "Low summary", "mid": "Mid summary"}.get(topic_id, "")
        store.read_summary.side_effect = read_summary_side_effect

        evaluate_coverage("program", store)

        prompt_arg = mock_llm_json.call_args[0][0]
        # Should include lowest-coverage topics (low and mid)
        assert "Low summary" in prompt_arg
        assert "Mid summary" in prompt_arg


# ── run_research_loop ─────────────────────────────────────────────────────


class TestRunResearchLoop:
    """Tests for the main research loop. All external calls are mocked."""

    def _make_search_result(self, title="Paper", url="https://example.com"):
        return SearchResult(
            title=title, url=url, snippet="snippet", source="duckduckgo"
        )

    def _make_mock_store(self):
        """Create a mock KnowledgeStore for loop tests."""
        store = MagicMock()
        from knowledge_store import IndexMeta, StoreIndex, Topic
        store.exists.return_value = False
        store.load_index.return_value = StoreIndex(meta=IndexMeta())
        store.get_context_for_gaps.return_value = "index context"
        store.get_context_with_budget.return_value = "index context"
        store.read_summary.return_value = ""
        store.next_source_id.return_value = 1
        store.get_unsynthesized_sources.return_value = []
        store.prioritize_stale_topics.return_value = []
        store.regenerate_findings.return_value = "# Regenerated findings"
        return store

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Topic Findings")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_stops_when_target_coverage_reached(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [self._make_search_result()]
        mock_eval.return_value = EvaluationResult(coverage_score=0.9)

        run_research_loop(max_iterations=5, target_coverage=0.8)

        # Should stop after 1 iteration since coverage 0.9 >= 0.8
        assert mock_gaps.call_count == 1
        assert mock_search.call_count == 1

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Topic Findings")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_respects_max_iterations(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [self._make_search_result()]
        mock_eval.return_value = EvaluationResult(coverage_score=0.3)

        run_research_loop(max_iterations=3, target_coverage=0.95)

        assert mock_gaps.call_count == 3

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Topic Findings")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    @patch("research.time")
    def test_stops_on_time_budget(
        self, mock_time, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        call_count = {"n": 0}

        def fake_time():
            call_count["n"] += 1
            if call_count["n"] > 3:
                return 4000
            return 0

        mock_time.time.side_effect = fake_time

        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [self._make_search_result()]
        mock_eval.return_value = EvaluationResult(coverage_score=0.3)

        run_research_loop(max_iterations=10, target_coverage=0.95, time_budget_min=60)

        # Only 1 iteration should complete before time budget stops loop
        assert mock_gaps.call_count == 1

    @patch("research.KnowledgeStore")
    @patch("research.log_iteration")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_fallback_search_on_no_results(
        self, mock_read, mock_init_log, mock_gaps, mock_search, mock_log, mock_store_cls,
    ):
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap about X", "gap about Y"],
            "search_plan": [{"query": "q1", "sources": ["arxiv"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        # First search returns nothing, fallback also returns nothing
        mock_search.return_value = []

        run_research_loop(max_iterations=1, target_coverage=0.95)

        # execute_searches called twice: original + fallback
        assert mock_search.call_count == 2
        # Fallback queries should use gaps as queries with duckduckgo
        fallback_call = mock_search.call_args_list[1]
        fallback_queries = fallback_call[0][0]
        assert fallback_queries[0]["sources"] == ["duckduckgo"]
        assert fallback_queries[0]["query"] == "gap about X"

    @patch("research.KnowledgeStore")
    @patch("research.log_iteration")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_continues_on_gap_analysis_error(
        self, mock_read, mock_init_log, mock_gaps, mock_log, mock_store_cls,
    ):
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        mock_store_cls.return_value = mock_store

        mock_gaps.side_effect = [
            RuntimeError("LLM error"),
            {
                "gaps": ["gap1"],
                "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
                "new_topics": [],
            },
        ]

        with patch("research.execute_searches", return_value=[]) as mock_search:
            run_research_loop(max_iterations=2, target_coverage=0.95)

        # First iteration errors out, second proceeds
        assert mock_gaps.call_count == 2
        # log_iteration called for the error iteration
        assert mock_log.call_count >= 1

    @patch("research.KnowledgeStore")
    @patch("research.log_iteration")
    @patch("research.evaluate_coverage")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_continues_on_synthesis_error(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_eval, mock_log, mock_store_cls,
    ):
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        # Make store operations raise during synthesis
        mock_store.save_raw_results.side_effect = RuntimeError("synthesis failed")
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [self._make_search_result()]

        run_research_loop(max_iterations=1, target_coverage=0.95)

        # Should log an error status, not crash
        mock_log.assert_called_once()
        log_entry = mock_log.call_args[0][0]
        assert log_entry.status == "error"

    @patch("research.read_file", return_value="")
    def test_exits_if_research_program_missing(self, mock_read):
        with pytest.raises(SystemExit) as exc_info:
            run_research_loop()
        assert exc_info.value.code == 1

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Topic Findings")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_evaluation_error_defaults_to_zero_coverage(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [self._make_search_result()]
        mock_eval.side_effect = RuntimeError("eval error")

        # Should not crash — evaluation error is caught and defaults to 0.0
        run_research_loop(max_iterations=1, target_coverage=0.95)

        mock_eval.assert_called_once()

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Topic Findings")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_bootstraps_store_from_existing_findings(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        """When findings.md exists but store doesn't, bootstrap the store."""
        def read_side_effect(path):
            if "research_program" in path:
                return "research program content"
            if "findings" in path:
                return "# Existing findings\nSome content here"
            return ""

        mock_read.side_effect = read_side_effect
        mock_store = self._make_mock_store()
        mock_store.exists.return_value = False
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [self._make_search_result()]
        mock_eval.return_value = EvaluationResult(coverage_score=0.9)

        run_research_loop(max_iterations=1, target_coverage=0.8)

        # Bootstrap should have been called
        mock_store.bootstrap_from_findings.assert_called_once()

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Topic Findings")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_creates_new_topics_from_gap_analysis(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "new_topic"}],
            "new_topics": [{"id": "new_topic", "title": "New Topic", "keywords": ["kw1"]}],
        }
        mock_search.return_value = [self._make_search_result()]
        mock_eval.return_value = EvaluationResult(coverage_score=0.9)

        run_research_loop(max_iterations=1, target_coverage=0.8)

        # add_topic should have been called with the new topic
        mock_store.add_topic.assert_called_once()
        topic_arg = mock_store.add_topic.call_args[0][0]
        assert topic_arg.id == "new_topic"
        assert topic_arg.title == "New Topic"
        assert topic_arg.keywords == ["kw1"]

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Updated Summary")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_per_topic_delta_merge(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        """Verify that synthesis calls extract_topic_findings per stale topic."""
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        from knowledge_store import IndexMeta, StoreIndex, Topic
        stale_topic = Topic(id="topic1", title="Topic 1", stale=True, unsynthesized_sources=2)
        mock_store.load_index.return_value = StoreIndex(
            topics=[stale_topic], meta=IndexMeta()
        )
        mock_store.get_unsynthesized_sources.return_value = [
            {"title": "Src 1", "url": "http://a.com", "snippet": "s", "source": "arxiv"},
        ]
        # topic1 is stale, so prioritize_stale_topics should return it
        mock_store.prioritize_stale_topics.return_value = ["topic1"]
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [self._make_search_result()]
        mock_eval.return_value = EvaluationResult(coverage_score=0.9)

        run_research_loop(max_iterations=1, target_coverage=0.8)

        # extract_topic_findings should have been called for topic1
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[0][3] == "topic1"  # topic_id argument

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Updated")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_stale_topic_handling(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        """Stale topics should be included in topics_to_update even if not in search_plan."""
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        from knowledge_store import IndexMeta, StoreIndex, Topic
        # topic2 is stale but NOT in the search_plan
        topics = [
            Topic(id="topic1", title="Topic 1", stale=False),
            Topic(id="topic2", title="Topic 2", stale=True, unsynthesized_sources=1),
        ]
        mock_store.load_index.return_value = StoreIndex(topics=topics, meta=IndexMeta())
        mock_store.get_unsynthesized_sources.return_value = [
            {"title": "Src", "url": "http://a.com", "snippet": "s", "source": "arxiv"},
        ]
        # topic2 is stale, so prioritize_stale_topics should return it
        mock_store.prioritize_stale_topics.return_value = ["topic2"]
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [self._make_search_result()]
        mock_eval.return_value = EvaluationResult(coverage_score=0.9)

        run_research_loop(max_iterations=1, target_coverage=0.8)

        # extract_topic_findings should be called for both topic2 (stale first) and topic1
        assert mock_extract.call_count == 2
        called_topics = {c[0][3] for c in mock_extract.call_args_list}
        assert "topic1" in called_topics
        assert "topic2" in called_topics

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Findings")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_regenerates_findings_from_store(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        """Findings.md should be regenerated from the store after synthesis."""
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        mock_store.regenerate_findings.return_value = "# Regenerated from store"
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [self._make_search_result()]
        mock_eval.return_value = EvaluationResult(coverage_score=0.9)

        run_research_loop(max_iterations=1, target_coverage=0.8)

        # regenerate_findings should have been called
        mock_store.regenerate_findings.assert_called_once()
        # write_file should write the regenerated content
        write_calls = [c for c in mock_write.call_args_list if "findings" in str(c)]
        assert len(write_calls) > 0

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Findings")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_updates_index_meta_after_evaluation(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        """Index meta should be updated with iteration count and coverage."""
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        from knowledge_store import IndexMeta, StoreIndex
        idx = StoreIndex(meta=IndexMeta())
        mock_store.load_index.return_value = idx
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [self._make_search_result()]
        mock_eval.return_value = EvaluationResult(coverage_score=0.75)

        run_research_loop(max_iterations=1, target_coverage=0.95)

        # save_index should have been called to update meta
        mock_store.save_index.assert_called()

    @patch("research.KnowledgeStore")
    @patch("research.evaluate_coverage")
    @patch("research.write_file")
    @patch("research.extract_topic_findings", return_value="# Findings")
    @patch("research.execute_searches")
    @patch("research.identify_gaps")
    @patch("research.init_progress_log")
    @patch("research.read_file")
    def test_stores_search_results_as_sources(
        self, mock_read, mock_init_log, mock_gaps, mock_search,
        mock_extract, mock_write, mock_eval, mock_store_cls,
    ):
        """Search results should be stored via store.add_source."""
        mock_read.return_value = "research program content"
        mock_store = self._make_mock_store()
        mock_store_cls.return_value = mock_store

        mock_gaps.return_value = {
            "gaps": ["gap1"],
            "search_plan": [{"query": "q1", "sources": ["duckduckgo"], "topic_id": "topic1"}],
            "new_topics": [],
        }
        mock_search.return_value = [
            self._make_search_result("Paper1", "http://a.com"),
            self._make_search_result("Paper2", "http://b.com"),
        ]
        mock_eval.return_value = EvaluationResult(coverage_score=0.9)

        run_research_loop(max_iterations=1, target_coverage=0.8)

        # add_source should have been called for each result
        assert mock_store.add_source.call_count == 2


# ── main / CLI ────────────────────────────────────────────────────────────


class TestMainCLI:
    @patch("research.run_research_loop")
    def test_default_arguments(self, mock_loop):
        with patch("sys.argv", ["research.py"]):
            main()

        mock_loop.assert_called_once_with(
            max_iterations=20,
            target_coverage=0.8,
            time_budget_min=60,
            mode=InteractionMode.AUTO,
        )

    @patch("research.run_research_loop")
    def test_custom_arguments(self, mock_loop):
        with patch("sys.argv", [
            "research.py",
            "--max-iterations", "5",
            "--target-coverage", "0.9",
            "--time-budget", "30",
            "--mode", "interactive",
        ]):
            main()

        mock_loop.assert_called_once_with(
            max_iterations=5,
            target_coverage=0.9,
            time_budget_min=30,
            mode=InteractionMode.INTERACTIVE,
        )

    @patch("research.run_research_loop")
    def test_supervised_mode(self, mock_loop):
        with patch("sys.argv", ["research.py", "--mode", "supervised"]):
            main()

        mock_loop.assert_called_once()
        assert mock_loop.call_args[1]["mode"] == InteractionMode.SUPERVISED

    def test_invalid_mode_exits(self):
        with patch("sys.argv", ["research.py", "--mode", "bogus"]):
            with pytest.raises(SystemExit):
                main()
