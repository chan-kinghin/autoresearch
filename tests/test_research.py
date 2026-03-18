"""Tests for research.py — autonomous research loop."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from research import (
    CheckpointResult,
    InteractionMode,
    StoppingSignal,
    check_synthesis_quality,
    cross_topic_synthesis,
    extract_topic_findings,
    get_failed_search_summary,
    identify_gaps,
    is_similar_to_failed,
    main,
    prompt_checkpoint,
    read_file,
    record_search_result,
    run_research_loop,
    should_stop_research,
    verify_citations_against_sources,
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
    def test_truncates_full_text_to_6000_chars(self, mock_llm_call):
        mock_llm_call.return_value = "findings"

        long_text = "x" * 8000
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
        # full_text is truncated to 6000 chars (tiered content budget)
        assert "x" * 6000 in prompt_arg
        assert "x" * 6001 not in prompt_arg

    @patch("research.llm_call")
    def test_truncates_snippet_to_2000_chars(self, mock_llm_call):
        mock_llm_call.return_value = "findings"

        long_snippet = "s" * 5000
        results = [
            SearchResult(
                title="Paper",
                url="https://example.com",
                snippet=long_snippet,
                source="arxiv",
            ),
        ]

        store = MagicMock()
        store.read_summary.return_value = ""
        store.get_topic.return_value = None

        extract_topic_findings(results, "prog", store, "topic1")

        prompt_arg = mock_llm_call.call_args[0][0]
        # snippet (no full_text) is truncated to 2000 chars
        assert "s" * 2000 in prompt_arg
        assert "s" * 2001 not in prompt_arg

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

        # Multi-signal stopping needs 3+ scores for stable detection.
        # Scores are appended twice per iteration, so after iteration 2
        # recent_scores has 4 entries, all >= 0.8 -> stops.
        assert mock_gaps.call_count <= 2

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


# ── A6: Per-topic depth strategy ─────────────────────────────────────────


class TestPerTopicDepthStrategy:
    """A6: identify_gaps classifies topics as NEEDS_BREADTH/NEEDS_DEPTH/WELL_COVERED."""

    def _make_mock_store(self, topics):
        store = MagicMock()
        from knowledge_store import IndexMeta, StoreIndex
        index = StoreIndex(topics=topics, meta=IndexMeta())
        store.load_index.return_value = index
        store.get_context_with_budget.return_value = "## Knowledge Store Index\nContext"
        store.read_summary.return_value = ""
        return store

    @patch("research.llm_json")
    def test_low_coverage_labeled_needs_breadth(self, mock_llm_json):
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}
        from knowledge_store import Topic
        store = self._make_mock_store([Topic(id="t1", title="Topic 1", coverage=0.1)])

        identify_gaps("program", store)

        prompt = mock_llm_json.call_args[0][0]
        assert "NEEDS_BREADTH" in prompt
        assert "t1" in prompt

    @patch("research.llm_json")
    def test_mid_coverage_labeled_needs_depth(self, mock_llm_json):
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}
        from knowledge_store import Topic
        store = self._make_mock_store([Topic(id="t1", title="Topic 1", coverage=0.5)])

        identify_gaps("program", store)

        prompt = mock_llm_json.call_args[0][0]
        assert "NEEDS_DEPTH" in prompt

    @patch("research.llm_json")
    def test_high_coverage_labeled_well_covered(self, mock_llm_json):
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}
        from knowledge_store import Topic
        store = self._make_mock_store([Topic(id="t1", title="Topic 1", coverage=0.85)])

        identify_gaps("program", store)

        prompt = mock_llm_json.call_args[0][0]
        assert "WELL_COVERED" in prompt

    @patch("research.llm_json")
    def test_depth_block_absent_for_empty_store(self, mock_llm_json):
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}
        store = self._make_mock_store([])

        identify_gaps("program", store)

        prompt = mock_llm_json.call_args[0][0]
        assert "Topic Depth Analysis" not in prompt

    @patch("research.llm_json")
    def test_depth_boundary_030(self, mock_llm_json):
        """Coverage exactly 0.3 should be NEEDS_DEPTH, not NEEDS_BREADTH for that topic."""
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}
        from knowledge_store import Topic
        store = self._make_mock_store([Topic(id="t1", title="T", coverage=0.3)])

        identify_gaps("program", store)

        prompt = mock_llm_json.call_args[0][0]
        # The topic line should show NEEDS_DEPTH
        assert '"t1"' in prompt
        assert "coverage=0.30" in prompt
        # Topic line: NEEDS_DEPTH (not NEEDS_BREADTH) since 0.3 >= 0.3
        import re
        topic_line = [l for l in prompt.split("\n") if '"t1"' in l and "coverage=" in l][0]
        assert "NEEDS_DEPTH" in topic_line

    @patch("research.llm_json")
    def test_depth_boundary_070(self, mock_llm_json):
        """Coverage exactly 0.7 should be WELL_COVERED for that topic."""
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}
        from knowledge_store import Topic
        store = self._make_mock_store([Topic(id="t1", title="T", coverage=0.7)])

        identify_gaps("program", store)

        prompt = mock_llm_json.call_args[0][0]
        import re
        topic_line = [l for l in prompt.split("\n") if '"t1"' in l and "coverage=" in l][0]
        assert "WELL_COVERED" in topic_line

    @patch("research.llm_json")
    def test_search_strategy_guidance_present(self, mock_llm_json):
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}
        from knowledge_store import Topic
        store = self._make_mock_store([Topic(id="t1", title="T", coverage=0.1)])

        identify_gaps("program", store)

        prompt = mock_llm_json.call_args[0][0]
        assert "use web searches" in prompt
        assert "use academic sources" in prompt


# ── B1: Comparative synthesis prompts ────────────────────────────────────


class TestComparativeSynthesis:
    """B1: extract_topic_findings includes convergence/divergence/interactions/hierarchies."""

    @patch("research.llm_call")
    def test_convergence_instruction_in_prompt(self, mock_llm_call):
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = "existing"
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "CONVERGENCE" in prompt

    @patch("research.llm_call")
    def test_divergence_instruction_in_prompt(self, mock_llm_call):
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = "existing"
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "DIVERGENCE" in prompt

    @patch("research.llm_call")
    def test_interactions_instruction_in_prompt(self, mock_llm_call):
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = "existing"
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "INTERACTIONS" in prompt

    @patch("research.llm_call")
    def test_hierarchies_instruction_in_prompt(self, mock_llm_call):
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = "existing"
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "HIERARCHIES" in prompt


# ── B2: Contradiction detection & reconciliation ────────────────────────


class TestContradictionDetection:
    """B2: [CONFLICT] markers in synthesis prompts."""

    @patch("research.llm_call")
    def test_conflict_marker_in_prompt(self, mock_llm_call):
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = "existing"
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "[CONFLICT]" in prompt

    @patch("research.llm_call")
    def test_contradiction_protocol_in_prompt(self, mock_llm_call):
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = "existing"
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "Contradiction Protocol" in prompt
        assert "peer-reviewed" in prompt

    @patch("research.llm_call")
    def test_competing_claims_section_in_prompt(self, mock_llm_call):
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = "existing"
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "Competing Claims" in prompt


# ── B3: Evidence tier classification ─────────────────────────────────────


class TestEvidenceTierClassification:
    """B3: T1-T4 labeling in synthesis prompt."""

    @patch("research.llm_call")
    def test_evidence_tiers_in_prompt(self, mock_llm_call):
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = "existing"
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="arxiv")],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "T1" in prompt
        assert "T2" in prompt
        assert "T3" in prompt
        assert "T4" in prompt
        assert "Evidence Tier Classification" in prompt

    @patch("research.llm_call")
    def test_authority_indicator_for_academic(self, mock_llm_call):
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = ""
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="semantic_scholar")],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "Academic" in prompt

    @patch("research.llm_call")
    def test_authority_indicator_for_web(self, mock_llm_call):
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = ""
        store.get_topic.return_value = None

        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="s", source="duckduckgo")],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "Web" in prompt

    @patch("research.llm_call")
    def test_tiered_content_budget_full_text_6000(self, mock_llm_call):
        """Tier 1 sources with full_text get 6000 char budget."""
        mock_llm_call.return_value = "findings"
        store = MagicMock()
        store.read_summary.return_value = ""
        store.get_topic.return_value = None

        long_text = "y" * 8000
        extract_topic_findings(
            [SearchResult(title="T", url="u", snippet="short", source="metaso", full_text=long_text)],
            "prog", store, "topic1",
        )

        prompt = mock_llm_call.call_args[0][0]
        assert "y" * 6000 in prompt
        assert "y" * 6001 not in prompt


# ── B5: Citation verification ────────────────────────────────────────────


class TestCitationVerification:
    """B5: verify_citations_against_sources()."""

    def test_matching_citations_are_plausible(self):
        # Citation words (split on space, len>2) must overlap with source words (split on space, len>3)
        summary = "According to [Deep Neural Survey], neural networks are effective."
        sources = ["Smith et al. - Deep Neural Network Survey"]
        result = verify_citations_against_sources(summary, sources)

        assert result["total"] == 1
        assert result["plausible"] == 1
        assert result["suspicious"] == []

    def test_unmatched_citation_is_suspicious(self):
        summary = "According to [Phantom Author], this is true."
        sources = ["Smith et al. - Deep Neural Network Survey"]
        result = verify_citations_against_sources(summary, sources)

        assert result["total"] == 1
        assert result["plausible"] == 0
        assert "Phantom Author" in result["suspicious"]

    def test_conflict_markers_excluded(self):
        summary = "Sources conflict [CONFLICT] but [Smith, 2024] says X."
        sources = ["Smith et al."]
        result = verify_citations_against_sources(summary, sources)

        # [CONFLICT] should NOT be counted as a citation
        assert result["total"] == 1

    def test_http_links_excluded(self):
        summary = "See [http://example.com] and [Smith, 2024]."
        sources = ["Smith et al."]
        result = verify_citations_against_sources(summary, sources)

        assert result["total"] == 1

    def test_empty_summary(self):
        result = verify_citations_against_sources("", ["Source A"])
        assert result["total"] == 0
        assert result["plausible"] == 0
        assert result["suspicious"] == []

    def test_multiple_citations_mixed(self):
        # cite_words are lowered split on space, len>2; source_words are lowered split on space, len>3
        summary = "[Smith Network] agrees with [Jones Learning] but [Unknown Stuff] differs."
        sources = ["Smith Neural Network Paper", "Jones Machine Learning"]
        result = verify_citations_against_sources(summary, sources)

        assert result["total"] == 3
        assert result["plausible"] == 2
        assert len(result["suspicious"]) == 1


# ── C1: Evaluation calibration ───────────────────────────────────────────


class TestEvaluationCalibration:
    """C1: calibration_confidence in evaluate_coverage."""

    @patch("research.llm_json")
    def test_calibration_confidence_in_prompt(self, mock_llm_json):
        from research import evaluate_coverage
        mock_llm_json.return_value = {
            "coverage_score": 0.5,
            "calibration_confidence": 0.8,
            "topic_scores": {},
        }
        store = MagicMock()
        store.get_context_for_gaps.return_value = "context"
        from knowledge_store import IndexMeta, StoreIndex
        store.load_index.return_value = StoreIndex(meta=IndexMeta())
        store.read_summary.return_value = ""

        evaluate_coverage("program", store)

        prompt = mock_llm_json.call_args[0][0]
        assert "calibration_confidence" in prompt
        assert "Calibration Instruction" in prompt

    @patch("research.llm_json")
    def test_calibration_clamped_to_valid_range(self, mock_llm_json):
        from research import evaluate_coverage
        mock_llm_json.return_value = {
            "coverage_score": 0.5,
            "calibration_confidence": 1.5,  # Out of range
            "topic_scores": {},
        }
        store = MagicMock()
        store.get_context_for_gaps.return_value = "context"
        from knowledge_store import IndexMeta, StoreIndex
        store.load_index.return_value = StoreIndex(meta=IndexMeta())
        store.read_summary.return_value = ""

        # Should not crash; clamping happens internally
        result = evaluate_coverage("program", store)
        assert result.coverage_score == 0.5

    @patch("research.llm_json")
    def test_invalid_calibration_defaults_to_05(self, mock_llm_json):
        from research import evaluate_coverage
        mock_llm_json.return_value = {
            "coverage_score": 0.5,
            "calibration_confidence": "not_a_number",
            "topic_scores": {},
        }
        store = MagicMock()
        store.get_context_for_gaps.return_value = "context"
        from knowledge_store import IndexMeta, StoreIndex
        store.load_index.return_value = StoreIndex(meta=IndexMeta())
        store.read_summary.return_value = ""

        result = evaluate_coverage("program", store)
        # Should not crash
        assert result.coverage_score == 0.5


# ── C2: Per-question evaluation ──────────────────────────────────────────


class TestPerQuestionEvaluation:
    """C2: question_scores parsing in evaluate_coverage."""

    @patch("research.llm_json")
    def test_question_scores_in_prompt(self, mock_llm_json):
        from research import evaluate_coverage
        mock_llm_json.return_value = {
            "coverage_score": 0.6,
            "question_scores": [],
            "topic_scores": {},
        }
        store = MagicMock()
        store.get_context_for_gaps.return_value = "context"
        from knowledge_store import IndexMeta, StoreIndex
        store.load_index.return_value = StoreIndex(meta=IndexMeta())
        store.read_summary.return_value = ""

        evaluate_coverage("program", store)

        prompt = mock_llm_json.call_args[0][0]
        assert "question_scores" in prompt
        assert "score" in prompt
        assert "status" in prompt

    @patch("research.llm_json")
    def test_question_status_updated_in_store(self, mock_llm_json):
        from research import evaluate_coverage
        from knowledge_store import IndexMeta, Question, StoreIndex

        mock_llm_json.return_value = {
            "coverage_score": 0.6,
            "question_scores": [
                {"question": "What is machine learning?", "score": 0.8, "status": "answered",
                 "covered_by_topics": ["ml_basics"]},
            ],
            "topic_scores": {},
        }
        store = MagicMock()
        store.get_context_for_gaps.return_value = "context"
        q = Question(id="q1", text="What is machine learning?", status="unanswered")
        idx = StoreIndex(questions=[q], meta=IndexMeta())
        store.load_index.return_value = idx
        store.read_summary.return_value = ""

        evaluate_coverage("program", store)

        # save_index should be called to persist question status
        store.save_index.assert_called()
        # The question's status should have been updated
        assert q.status == "answered"
        assert q.related_topics == ["ml_basics"]

    @patch("research.llm_json")
    def test_question_matching_uses_prefix(self, mock_llm_json):
        """Questions are matched by first 50 chars of lowercase text."""
        from research import evaluate_coverage
        from knowledge_store import IndexMeta, Question, StoreIndex

        long_question = "What are the main approaches to natural language processing in modern systems?"
        mock_llm_json.return_value = {
            "coverage_score": 0.6,
            "question_scores": [
                {"question": long_question, "score": 0.5, "status": "partial"},
            ],
            "topic_scores": {},
        }
        store = MagicMock()
        store.get_context_for_gaps.return_value = "context"
        q = Question(id="q1", text=long_question + " Extra details here.", status="unanswered")
        idx = StoreIndex(questions=[q], meta=IndexMeta())
        store.load_index.return_value = idx
        store.read_summary.return_value = ""

        evaluate_coverage("program", store)

        assert q.status == "partial"


# ── C3: Multi-signal stopping ────────────────────────────────────────────


class TestMultiSignalStopping:
    """C3: StoppingSignal and should_stop_research()."""

    def test_max_iterations_reached(self):
        signal = should_stop_research([0.5, 0.6, 0.7], iteration=10, target_coverage=0.8, max_iterations=10)
        assert signal.should_stop is True
        assert signal.reason == "max_iterations_reached"
        assert signal.confidence == 1.0

    def test_too_few_iterations(self):
        signal = should_stop_research([0.5, 0.6], iteration=2, target_coverage=0.8, max_iterations=10)
        assert signal.should_stop is False
        assert signal.reason == "too_few_iterations"

    def test_coverage_stable_at_target(self):
        signal = should_stop_research([0.5, 0.6, 0.85, 0.83, 0.82], iteration=5, target_coverage=0.8, max_iterations=10)
        assert signal.should_stop is True
        assert signal.reason == "coverage_stable_at_target"
        assert signal.confidence == 0.95

    def test_coverage_not_stable_below_target(self):
        signal = should_stop_research([0.5, 0.6, 0.75], iteration=3, target_coverage=0.8, max_iterations=10)
        assert signal.should_stop is False

    def test_oscillation_detected(self):
        # Scores: 0.5 -> 0.6 -> 0.5 -> 0.6 -> 0.5 (3 direction changes)
        signal = should_stop_research([0.5, 0.6, 0.5, 0.6, 0.5], iteration=5, target_coverage=0.8, max_iterations=10)
        assert signal.should_stop is True
        assert "oscillating" in signal.reason
        assert signal.confidence == 0.7

    def test_diminishing_returns(self):
        # Less than 0.02 improvement over last 4, iteration > 5
        signal = should_stop_research([0.5, 0.6, 0.61, 0.61, 0.615, 0.618], iteration=6, target_coverage=0.8, max_iterations=20)
        assert signal.should_stop is True
        assert signal.reason == "diminishing_returns"
        assert signal.confidence == 0.6

    def test_single_score_at_target(self):
        signal = should_stop_research([0.3, 0.5, 0.82], iteration=3, target_coverage=0.8, max_iterations=10)
        assert signal.should_stop is True
        assert signal.reason == "target_reached"
        assert signal.confidence == 0.9

    def test_continue_when_improving(self):
        signal = should_stop_research([0.3, 0.4, 0.5], iteration=3, target_coverage=0.8, max_iterations=10)
        assert signal.should_stop is False
        assert signal.reason == "continue"

    def test_stopping_signal_dataclass(self):
        s = StoppingSignal(should_stop=True, reason="test", confidence=0.5)
        assert s.should_stop is True
        assert s.reason == "test"
        assert s.confidence == 0.5


# ── C4: Systematic gap analysis ──────────────────────────────────────────


class TestSystematicGapAnalysis:
    """C4: question-by-question coverage in identify_gaps and evaluate_coverage."""

    @patch("research.llm_json")
    def test_systematic_question_check_in_gaps_prompt(self, mock_llm_json):
        mock_llm_json.return_value = {"gaps": [], "search_plan": [], "new_topics": []}
        from knowledge_store import IndexMeta, StoreIndex, Topic
        store = MagicMock()
        store.load_index.return_value = StoreIndex(
            topics=[Topic(id="t1", title="T", coverage=0.5)],
            meta=IndexMeta(),
        )
        store.get_context_with_budget.return_value = "context"
        store.read_summary.return_value = ""

        identify_gaps("program", store)

        prompt = mock_llm_json.call_args[0][0]
        assert "Systematic Question Check" in prompt
        assert "LOWEST coverage" in prompt

    @patch("research.llm_json")
    def test_scoring_rubric_in_eval_prompt(self, mock_llm_json):
        from research import evaluate_coverage
        mock_llm_json.return_value = {"coverage_score": 0.5, "topic_scores": {}}
        store = MagicMock()
        store.get_context_for_gaps.return_value = "context"
        from knowledge_store import IndexMeta, StoreIndex
        store.load_index.return_value = StoreIndex(meta=IndexMeta())
        store.read_summary.return_value = ""

        evaluate_coverage("program", store)

        prompt = mock_llm_json.call_args[0][0]
        assert "Scoring Rubric" in prompt
        assert "Citation Validation Rule" in prompt

    @patch("research.llm_json")
    def test_coverage_score_clamped(self, mock_llm_json):
        from research import evaluate_coverage
        mock_llm_json.return_value = {"coverage_score": 1.5, "topic_scores": {}}
        store = MagicMock()
        store.get_context_for_gaps.return_value = "context"
        from knowledge_store import IndexMeta, StoreIndex
        store.load_index.return_value = StoreIndex(meta=IndexMeta())
        store.read_summary.return_value = ""

        result = evaluate_coverage("program", store)
        assert result.coverage_score == 1.0

    @patch("research.llm_json")
    def test_negative_coverage_score_clamped(self, mock_llm_json):
        from research import evaluate_coverage
        mock_llm_json.return_value = {"coverage_score": -0.5, "topic_scores": {}}
        store = MagicMock()
        store.get_context_for_gaps.return_value = "context"
        from knowledge_store import IndexMeta, StoreIndex
        store.load_index.return_value = StoreIndex(meta=IndexMeta())
        store.read_summary.return_value = ""

        result = evaluate_coverage("program", store)
        assert result.coverage_score == 0.0


# ── C5: Search history / failed query avoidance ─────────────────────────


class TestSearchHistory:
    """C5: record_search_result, get_failed_search_summary, is_similar_to_failed."""

    def setup_method(self):
        """Clear the module-level _search_history before each test."""
        import research
        research._search_history.clear()

    def test_record_search_result(self):
        record_search_result("test query", ["arxiv"], 5, 1)
        import research
        assert len(research._search_history) == 1
        assert research._search_history[0]["query"] == "test query"
        assert research._search_history[0]["result_count"] == 5

    def test_get_failed_search_summary_empty(self):
        assert get_failed_search_summary() == ""

    def test_get_failed_search_summary_with_failures(self):
        record_search_result("failed query", ["arxiv"], 0, 1)
        record_search_result("success query", ["duckduckgo"], 3, 1)

        summary = get_failed_search_summary()
        assert "failed query" in summary
        assert "success query" not in summary
        assert "NO results" in summary

    def test_get_failed_search_summary_max_5(self):
        for i in range(8):
            record_search_result(f"failed {i}", ["arxiv"], 0, i)

        summary = get_failed_search_summary()
        # Should only include the last 5 failures
        assert "failed 3" in summary
        assert "failed 7" in summary

    def test_is_similar_to_failed_true(self):
        record_search_result("machine learning algorithms overview", ["arxiv"], 0, 1)

        assert is_similar_to_failed("machine learning algorithms survey") is True

    def test_is_similar_to_failed_false(self):
        record_search_result("machine learning algorithms overview", ["arxiv"], 0, 1)

        assert is_similar_to_failed("quantum computing applications") is False

    def test_is_similar_to_failed_no_history(self):
        assert is_similar_to_failed("any query") is False

    def test_is_similar_to_failed_only_checks_failures(self):
        record_search_result("machine learning algorithms overview", ["arxiv"], 5, 1)  # success

        assert is_similar_to_failed("machine learning algorithms overview") is False

    def test_is_similar_custom_threshold(self):
        record_search_result("machine learning deep", ["arxiv"], 0, 1)

        # With very high threshold, only exact overlap matches
        # "machine learning deep" vs "machine learning deep extra" => overlap=3/4=0.75
        assert is_similar_to_failed("machine learning deep extra words here", threshold=0.9) is False
        # With low threshold, partial overlap matches
        assert is_similar_to_failed("machine learning wide", threshold=0.3) is True

    def test_is_similar_empty_query(self):
        record_search_result("", ["arxiv"], 0, 1)
        assert is_similar_to_failed("test") is False


# ── C6: Synthesis quality gates ──────────────────────────────────────────


class TestSynthesisQualityGates:
    """C6: check_synthesis_quality()."""

    def test_first_synthesis_always_ok(self):
        is_ok, reason = check_synthesis_quality("", "new content")
        assert is_ok is True
        assert reason == "first_synthesis"

    def test_first_synthesis_whitespace_only(self):
        is_ok, reason = check_synthesis_quality("   ", "new content")
        assert is_ok is True
        assert reason == "first_synthesis"

    def test_normal_growth_is_ok(self):
        old = "Some existing findings with [Citation, 2024]."
        new = "Some existing findings with [Citation, 2024]. Plus new data from [New, 2025]."
        is_ok, reason = check_synthesis_quality(old, new)
        assert is_ok is True
        assert reason == "ok"

    def test_dramatic_shrinkage_rejected(self):
        old = "x" * 500
        new = "x" * 100
        is_ok, reason = check_synthesis_quality(old, new)
        assert is_ok is False
        assert "summary_shrunk" in reason

    def test_small_shrinkage_ok_when_short(self):
        old = "x" * 100
        new = "x" * 40
        is_ok, reason = check_synthesis_quality(old, new)
        assert is_ok is True  # old_len <= 200, so shrinkage check skipped

    def test_citation_loss_rejected(self):
        old = "Data from [Smith, 2024] and [Jones, 2023] and [Lee, 2022] and [Wang, 2021]."
        new = "Data exists."
        is_ok, reason = check_synthesis_quality(old, new)
        assert is_ok is False
        assert "citations_lost" in reason

    def test_citation_maintained_ok(self):
        old = "From [Smith, 2024] and [Jones, 2023]."
        new = "From [Smith, 2024] and [Jones, 2023] and [New, 2025]."
        is_ok, reason = check_synthesis_quality(old, new)
        assert is_ok is True

    def test_single_citation_not_flagged(self):
        """When old has only 1 citation, losing it shouldn't trigger (old_cites < 2)."""
        old = "From [Smith, 2024] only."
        new = "Updated content without citations."
        is_ok, reason = check_synthesis_quality(old, new)
        assert is_ok is True


# ── B4: Cross-topic synthesis ────────────────────────────────────────────


class TestCrossTopicSynthesis:
    """B4: cross_topic_synthesis()."""

    @patch("research.llm_call")
    def test_returns_empty_for_single_topic(self, mock_llm_call):
        store = MagicMock()
        from knowledge_store import IndexMeta, StoreIndex, Topic
        store.load_index.return_value = StoreIndex(
            topics=[Topic(id="t1", title="T1")],
            meta=IndexMeta(),
        )

        result = cross_topic_synthesis(store, "program")

        assert result == ""
        mock_llm_call.assert_not_called()

    @patch("research.llm_call")
    def test_returns_empty_for_no_topics(self, mock_llm_call):
        store = MagicMock()
        from knowledge_store import IndexMeta, StoreIndex
        store.load_index.return_value = StoreIndex(meta=IndexMeta())

        result = cross_topic_synthesis(store, "program")

        assert result == ""
        mock_llm_call.assert_not_called()

    @patch("research.llm_call")
    def test_calls_llm_with_multiple_topics(self, mock_llm_call):
        mock_llm_call.return_value = "Cross-topic analysis"
        store = MagicMock()
        from knowledge_store import IndexMeta, StoreIndex, Topic
        store.load_index.return_value = StoreIndex(
            topics=[
                Topic(id="t1", title="Topic 1", coverage=0.7),
                Topic(id="t2", title="Topic 2", coverage=0.5),
            ],
            meta=IndexMeta(),
        )
        store.read_summary.side_effect = lambda tid: {"t1": "Summary 1", "t2": "Summary 2"}.get(tid, "")

        result = cross_topic_synthesis(store, "program")

        assert result == "Cross-topic analysis"
        mock_llm_call.assert_called_once()
        prompt = mock_llm_call.call_args[0][0]
        assert "SHARED CONCEPTS" in prompt
        assert "CROSS-TOPIC CONTRADICTIONS" in prompt
        assert "EMERGENT INSIGHTS" in prompt
        assert "DEPENDENCY MAP" in prompt
        assert "Summary 1" in prompt
        assert "Summary 2" in prompt

    @patch("research.llm_call")
    def test_skips_empty_summaries(self, mock_llm_call):
        store = MagicMock()
        from knowledge_store import IndexMeta, StoreIndex, Topic
        store.load_index.return_value = StoreIndex(
            topics=[
                Topic(id="t1", title="T1", coverage=0.7),
                Topic(id="t2", title="T2", coverage=0.5),
            ],
            meta=IndexMeta(),
        )
        # One topic has content, one is empty
        store.read_summary.side_effect = lambda tid: {"t1": "Summary 1", "t2": ""}.get(tid, "")

        result = cross_topic_synthesis(store, "program")

        # Only 1 non-empty summary = less than 2, so return empty
        assert result == ""
        mock_llm_call.assert_not_called()

    @patch("research.llm_call")
    def test_handles_llm_error(self, mock_llm_call):
        mock_llm_call.side_effect = RuntimeError("LLM error")
        store = MagicMock()
        from knowledge_store import IndexMeta, StoreIndex, Topic
        store.load_index.return_value = StoreIndex(
            topics=[
                Topic(id="t1", title="T1"),
                Topic(id="t2", title="T2"),
            ],
            meta=IndexMeta(),
        )
        store.read_summary.side_effect = lambda tid: "Summary text"

        result = cross_topic_synthesis(store, "program")

        assert result == ""

    @patch("research.llm_call")
    def test_truncates_summaries_to_1500(self, mock_llm_call):
        mock_llm_call.return_value = "analysis"
        store = MagicMock()
        from knowledge_store import IndexMeta, StoreIndex, Topic
        store.load_index.return_value = StoreIndex(
            topics=[
                Topic(id="t1", title="T1", coverage=0.5),
                Topic(id="t2", title="T2", coverage=0.5),
            ],
            meta=IndexMeta(),
        )
        long_summary = "z" * 3000
        store.read_summary.side_effect = lambda tid: long_summary

        cross_topic_synthesis(store, "program")

        prompt = mock_llm_call.call_args[0][0]
        assert "z" * 1500 in prompt
        assert "z" * 1501 not in prompt
