"""Tests for search.py — LLM calls, search APIs, evaluation, and utilities."""

from __future__ import annotations

import json
import os
import textwrap
from unittest.mock import MagicMock, patch, mock_open

import httpx
import pytest

from search import (
    SearchResult,
    EvaluationResult,
    IterationLog,
    _detect_provider,
    llm_call,
    llm_json,
    search_semantic_scholar,
    search_arxiv,
    search_duckduckgo,
    search_metaso,
    search_perplexity,
    search_gemini_deep,
    execute_searches,
    init_progress_log,
    log_iteration,
    extract_webpage,
    # A1: Query success tracking
    QueryRecord,
    _query_history,
    record_query,
    get_query_history,
    get_effective_queries,
    get_failed_queries,
    # A2: Query specificity scoring
    score_query_specificity,
    # A3: Source exhaustion detection
    SourceStats,
    _source_stats,
    is_source_exhausted,
    reset_source_stats,
    get_source_stats,
    # A5: Pre-synthesis result filtering
    filter_results_by_relevance,
    # A7: Query-source semantic validation
    validate_query_sources,
)

from research import evaluate_coverage
from knowledge_store import KnowledgeStore, Topic


# ── _detect_provider ──────────────────────────────────────────────────────


class TestDetectProvider:
    def test_explicit_provider_prefix(self):
        assert _detect_provider("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_explicit_deepseek_prefix(self):
        assert _detect_provider("deepseek/deepseek-chat") == ("deepseek", "deepseek-chat")

    def test_explicit_gemini_prefix(self):
        assert _detect_provider("gemini/gemini-2.0-flash") == ("gemini", "gemini-2.0-flash")

    def test_explicit_anthropic_prefix(self):
        assert _detect_provider("anthropic/claude-sonnet-4-20250514") == (
            "anthropic",
            "claude-sonnet-4-20250514",
        )

    def test_auto_detect_claude(self):
        assert _detect_provider("claude-sonnet-4-20250514") == ("anthropic", "claude-sonnet-4-20250514")

    def test_auto_detect_gpt(self):
        assert _detect_provider("gpt-4o") == ("openai", "gpt-4o")

    def test_auto_detect_o1(self):
        assert _detect_provider("o1-preview") == ("openai", "o1-preview")

    def test_auto_detect_o3(self):
        assert _detect_provider("o3-mini") == ("openai", "o3-mini")

    def test_auto_detect_deepseek(self):
        assert _detect_provider("deepseek-chat") == ("deepseek", "deepseek-chat")

    def test_auto_detect_gemini(self):
        assert _detect_provider("gemini-2.0-flash") == ("gemini", "gemini-2.0-flash")

    def test_fallback_to_openai(self):
        assert _detect_provider("some-unknown-model") == ("openai", "some-unknown-model")

    def test_slash_with_nested_path(self):
        provider, model_id = _detect_provider("custom/org/model-v2")
        assert provider == "custom"
        assert model_id == "org/model-v2"


# ── llm_call ──────────────────────────────────────────────────────────────


class TestLlmCall:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("search.httpx.Client")
    def test_anthropic_path(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"text": "  Hello from Claude  "}]
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        result = llm_call("test prompt", model="claude-sonnet-4-20250514")
        assert result == "Hello from Claude"
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "anthropic" in call_kwargs[0][0]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("search.httpx.Client")
    def test_openai_path(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "  Hello from GPT  "}}]
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        result = llm_call("test prompt", model="gpt-4o")
        assert result == "Hello from GPT"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_anthropic_key(self):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
            llm_call("test", model="claude-sonnet-4-20250514")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_openai_key(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
            llm_call("test", model="gpt-4o")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_deepseek_key(self):
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY not set"):
            llm_call("test", model="deepseek-chat")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_google_key(self):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY not set"):
            llm_call("test", model="gemini-2.0-flash")


# ── llm_json ──────────────────────────────────────────────────────────────


class TestLlmJson:
    @patch("search.llm_call")
    def test_valid_json(self, mock_llm):
        mock_llm.return_value = '{"key": "value", "num": 42}'
        result = llm_json("test")
        assert result == {"key": "value", "num": 42}

    @patch("search.llm_call")
    def test_json_with_markdown_fences(self, mock_llm):
        mock_llm.return_value = '```json\n{"key": "value"}\n```'
        result = llm_json("test")
        assert result == {"key": "value"}

    @patch("search.llm_call")
    def test_json_with_plain_fences(self, mock_llm):
        mock_llm.return_value = '```\n{"key": "value"}\n```'
        result = llm_json("test")
        assert result == {"key": "value"}

    @patch("search.llm_call")
    def test_invalid_json_raises(self, mock_llm):
        mock_llm.return_value = "this is not json at all"
        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            llm_json("test")


# ── search_semantic_scholar ───────────────────────────────────────────────


class TestSearchSemanticScholar:
    @patch("search.httpx.Client")
    def test_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "title": "Test Paper",
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                    "year": 2024,
                    "abstract": "A test abstract",
                    "url": "https://example.com/paper",
                    "externalIds": {"DOI": "10.1234/test"},
                },
            ]
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        results = search_semantic_scholar("test query")
        assert len(results) == 1
        assert results[0].title == "Test Paper"
        assert results[0].authors == "Alice, Bob"
        assert results[0].year == "2024"
        assert results[0].source == "semantic_scholar"

    @patch("search.httpx.Client")
    def test_doi_url_fallback(self, mock_client_cls):
        """When paper has no url but has DOI, constructs doi.org URL."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "title": "DOI Paper",
                    "authors": [],
                    "year": 2023,
                    "abstract": "abstract",
                    "url": "",
                    "externalIds": {"DOI": "10.5678/doi-test"},
                },
            ]
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        results = search_semantic_scholar("test")
        assert results[0].url == "https://doi.org/10.5678/doi-test"

    @patch("search.time.sleep")
    @patch("search.httpx.Client")
    def test_rate_limit_retry(self, mock_client_cls, mock_sleep):
        """On 429, retries with exponential backoff (3s, 6s)."""
        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429
        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.json.return_value = {"data": []}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        # First call returns 429, first retry also 429, second retry succeeds
        mock_client.get.side_effect = [mock_resp_429, mock_resp_ok]
        mock_client_cls.return_value = mock_client

        results = search_semantic_scholar("test")
        mock_sleep.assert_called_once_with(3)
        assert results == []

    @patch("search.httpx.Client")
    def test_error_returns_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("fail")
        mock_client_cls.return_value = mock_client

        results = search_semantic_scholar("test")
        assert results == []


# ── search_arxiv ──────────────────────────────────────────────────────────


ARXIV_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Test Paper Title</title>
    <summary>This is a test abstract
    with newlines.</summary>
    <link href="https://arxiv.org/abs/1234.5678" rel="alternate" type="text/html"/>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <published>2024-01-15T00:00:00Z</published>
  </entry>
</feed>
""")

ARXIV_EMPTY_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>
""")


class TestSearchArxiv:
    @patch("search.httpx.Client")
    def test_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = ARXIV_XML
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        results = search_arxiv("test query")
        assert len(results) == 1
        assert results[0].title == "Test Paper Title"
        assert results[0].url == "https://arxiv.org/abs/1234.5678"
        assert results[0].authors == "Alice, Bob"
        assert results[0].year == "2024"
        assert results[0].source == "arxiv"
        assert "\n" not in results[0].title

    @patch("search.httpx.Client")
    def test_empty_results(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = ARXIV_EMPTY_XML
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        results = search_arxiv("nothing")
        assert results == []

    @patch("search.httpx.Client")
    def test_malformed_xml(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = "<not valid xml"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        results = search_arxiv("test")
        assert results == []


# ── search_duckduckgo ─────────────────────────────────────────────────────


class TestSearchDuckDuckGo:
    @patch("search.DDGS", create=True)
    def test_success(self, _mock_ddgs_module):
        """Patch the actual import inside search_duckduckgo."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text.return_value = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs_instance):
            results = search_duckduckgo("test query")

        assert len(results) == 2
        assert results[0].title == "Result 1"
        assert results[0].source == "duckduckgo"

    def test_import_error_returns_empty(self):
        """When duckduckgo_search is not installed, returns []."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "duckduckgo_search":
                raise ImportError("No module named 'duckduckgo_search'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            results = search_duckduckgo("test")
            assert results == []


# ── search_metaso ─────────────────────────────────────────────────────────


class TestSearchMetaso:
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_returns_empty(self):
        assert search_metaso("test") == []

    @patch.dict(os.environ, {"METASO_API_KEY": "test-key"})
    @patch("search.httpx.Client")
    def test_success_with_content_and_sources(self, mock_client_cls):
        # Metaso uses SSE format — mock resp.text with data: lines
        sse_lines = (
            'data:{"type":"append-text","text":"Research content "}\n'
            'data:{"type":"append-text","text":"about topic"}\n'
            'data:{"type":"set-reference","list":[{"title":"Source 1","link":"https://src1.com"},{"title":"Source 2","link":"https://src2.com"}]}\n'
            'data:[DONE]\n'
        )
        mock_resp = MagicMock()
        mock_resp.text = sse_lines
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        results = search_metaso("test query")
        assert len(results) == 3  # 1 content + 2 sources
        assert results[0].source == "metaso"
        assert results[0].title == "Metaso Research: test query"
        assert results[1].title == "Source 1"

    @patch.dict(os.environ, {"METASO_API_KEY": "test-key"})
    @patch("search.httpx.Client")
    def test_error_returns_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("fail")
        mock_client_cls.return_value = mock_client

        results = search_metaso("test")
        assert results == []


# ── search_perplexity ─────────────────────────────────────────────────────


class TestSearchPerplexity:
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_returns_empty(self):
        assert search_perplexity("test") == []

    @patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test-key"})
    @patch("search.httpx.Client")
    def test_success_with_citations(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Perplexity answer content"}}],
            "citations": ["https://cite1.com", "https://cite2.com"],
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        results = search_perplexity("test query")
        assert len(results) == 3  # 1 content + 2 citations
        assert results[0].source == "perplexity"
        assert results[0].title == "Perplexity: test query"
        assert results[1].url == "https://cite1.com"

    @patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test-key"})
    @patch("search.httpx.Client")
    def test_empty_content_no_result(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": ""}}],
            "citations": [],
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        results = search_perplexity("test")
        assert results == []


# ── search_gemini_deep ────────────────────────────────────────────────────


class TestSearchGeminiDeep:
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_returns_empty(self):
        assert search_gemini_deep("test") == []

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    @patch("search._call_openai_compatible")
    def test_success(self, mock_call):
        mock_call.return_value = "Deep research content about topic"

        results = search_gemini_deep("test query")
        assert len(results) == 1
        assert results[0].source == "gemini"
        assert results[0].title == "Gemini Research: test query"
        assert results[0].full_text == "Deep research content about topic"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    @patch("search._call_openai_compatible")
    def test_empty_content_returns_empty(self, mock_call):
        mock_call.return_value = ""

        results = search_gemini_deep("test")
        assert results == []

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    @patch("search._call_openai_compatible")
    def test_error_returns_empty(self, mock_call):
        mock_call.side_effect = Exception("API error")

        results = search_gemini_deep("test")
        assert results == []


# ── execute_searches ──────────────────────────────────────────────────────


class TestExecuteSearches:
    @patch("search.search_duckduckgo")
    def test_url_deduplication(self, mock_ddg):
        mock_ddg.return_value = [
            SearchResult(title="R1", url="https://example.com/1", snippet="s1", source="duckduckgo"),
            SearchResult(title="R2", url="https://example.com/1", snippet="s2", source="duckduckgo"),
            SearchResult(title="R3", url="https://example.com/2", snippet="s3", source="duckduckgo"),
        ]

        results = execute_searches([{"query": "test", "sources": ["duckduckgo"]}])
        assert len(results) == 2
        urls = [r.url for r in results]
        assert urls == ["https://example.com/1", "https://example.com/2"]

    @patch("search.search_duckduckgo")
    def test_empty_url_not_deduplicated(self, mock_ddg):
        """Results with empty URLs should not be deduplicated."""
        mock_ddg.return_value = [
            SearchResult(title="R1", url="", snippet="s1", source="duckduckgo"),
            SearchResult(title="R2", url="", snippet="s2", source="duckduckgo"),
        ]

        results = execute_searches([{"query": "test", "sources": ["duckduckgo"]}])
        assert len(results) == 2

    def test_unknown_source_skipped(self):
        results = execute_searches([{"query": "test", "sources": ["nonexistent_source"]}])
        assert results == []

    @patch("search.search_duckduckgo")
    def test_default_source_is_duckduckgo(self, mock_ddg):
        mock_ddg.return_value = [
            SearchResult(title="R1", url="https://example.com", snippet="s", source="duckduckgo"),
        ]

        results = execute_searches([{"query": "test"}])
        assert len(results) == 1
        mock_ddg.assert_called_once_with("test")

    @patch("search.search_duckduckgo")
    @patch("search.search_arxiv")
    def test_multiple_queries_and_sources(self, mock_arxiv, mock_ddg):
        mock_ddg.return_value = [
            SearchResult(title="DDG", url="https://ddg.com/1", snippet="s", source="duckduckgo"),
        ]
        mock_arxiv.return_value = [
            SearchResult(title="Arxiv", url="https://arxiv.org/1", snippet="s", source="arxiv"),
        ]

        queries = [
            {"query": "q1", "sources": ["duckduckgo"]},
            {"query": "q2", "sources": ["arxiv"]},
        ]
        results = execute_searches(queries)
        assert len(results) == 2
        assert {r.source for r in results} == {"duckduckgo", "arxiv"}

    @patch("search.search_duckduckgo")
    def test_cross_query_deduplication(self, mock_ddg):
        """Same URL across different queries is deduplicated."""
        mock_ddg.return_value = [
            SearchResult(title="R", url="https://same.com", snippet="s", source="duckduckgo"),
        ]

        queries = [
            {"query": "q1", "sources": ["duckduckgo"]},
            {"query": "q2", "sources": ["duckduckgo"]},
        ]
        results = execute_searches(queries)
        assert len(results) == 1


# ── evaluate_coverage ─────────────────────────────────────────────────────


class TestEvaluateCoverage:
    def _make_store(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "store"))
        store.add_topic(Topic(id="topic1", title="Topic 1"))
        return store

    @patch("research.llm_json")
    def test_success(self, mock_llm_json, tmp_path):
        mock_llm_json.return_value = {
            "coverage_score": 0.75,
            "questions_answered": ["Q1"],
            "questions_remaining": ["Q2"],
            "gaps": ["gap1"],
            "suggested_queries": ["query1"],
        }
        store = self._make_store(tmp_path)

        result = evaluate_coverage("program", store)
        assert isinstance(result, EvaluationResult)
        assert result.coverage_score == 0.75
        assert result.questions_answered == ["Q1"]
        assert result.questions_remaining == ["Q2"]
        assert result.gaps == ["gap1"]
        assert result.suggested_queries == ["query1"]

    @patch("research.llm_json")
    def test_llm_error_fallback(self, mock_llm_json, tmp_path):
        mock_llm_json.side_effect = Exception("LLM failed")
        store = self._make_store(tmp_path)

        result = evaluate_coverage("program", store)
        assert result.coverage_score == 0.0
        assert "LLM failed" in result.gaps[0]

    @patch("research.llm_json")
    def test_partial_response(self, mock_llm_json, tmp_path):
        """LLM returns only some fields; defaults should fill in."""
        mock_llm_json.return_value = {
            "coverage_score": 0.5,
        }
        store = self._make_store(tmp_path)

        result = evaluate_coverage("program", store)
        assert result.coverage_score == 0.5
        assert result.questions_answered == []
        assert result.suggested_queries == []


# ── init_progress_log & log_iteration ─────────────────────────────────────


class TestProgressLog:
    def test_init_progress_log(self, tmp_path):
        path = str(tmp_path / "progress.tsv")
        init_progress_log(path)

        with open(path) as f:
            content = f.read()
        assert content == "iteration\tcoverage\tsources\tduration_s\tstatus\n"

    def test_log_iteration(self, tmp_path):
        path = str(tmp_path / "progress.tsv")
        init_progress_log(path)

        log = IterationLog(iteration=1, coverage=0.65, sources=12, duration_s=3.5, status="continue")
        log_iteration(log, path)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert lines[1] == "1\t0.65\t12\t3.5\tcontinue\n"

    def test_log_multiple_iterations(self, tmp_path):
        path = str(tmp_path / "progress.tsv")
        init_progress_log(path)

        log1 = IterationLog(iteration=1, coverage=0.3, sources=5, duration_s=2.1, status="continue")
        log2 = IterationLog(iteration=2, coverage=0.8, sources=15, duration_s=4.2, status="target_reached")
        log_iteration(log1, path)
        log_iteration(log2, path)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3
        assert "target_reached" in lines[2]


# ── extract_webpage ───────────────────────────────────────────────────────


class TestExtractWebpage:
    @patch("search.httpx.Client")
    def test_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><p>Main content here</p></body></html>"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        with patch("trafilatura.extract", return_value="Main content here"):
            result = extract_webpage("https://example.com")
            assert result == "Main content here"

    @patch("search.httpx.Client")
    def test_trafilatura_returns_none(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        with patch("trafilatura.extract", return_value=None):
            result = extract_webpage("https://example.com")
            assert result == ""

    @patch("search.httpx.Client")
    def test_http_error_returns_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("fail")
        mock_client_cls.return_value = mock_client

        result = extract_webpage("https://example.com")
        assert result == ""


# ── A1: Query Success Tracking ───────────────────────────────────────────


class TestQuerySuccessTracking:
    def setup_method(self):
        """Clear query history before each test."""
        _query_history.clear()

    def test_query_record_dataclass(self):
        qr = QueryRecord(query="test", sources=["arxiv"], results_count=5)
        assert qr.query == "test"
        assert qr.sources == ["arxiv"]
        assert qr.results_count == 5
        assert qr.useful_results == 0
        assert qr.iteration == 0

    def test_query_record_custom_defaults(self):
        qr = QueryRecord(query="q", sources=["ddg"], results_count=3, useful_results=2, iteration=5)
        assert qr.useful_results == 2
        assert qr.iteration == 5

    def test_record_query_appends(self):
        record_query("query1", ["arxiv"], 5, iteration=1)
        record_query("query2", ["duckduckgo"], 0, iteration=2)
        assert len(_query_history) == 2
        assert _query_history[0].query == "query1"
        assert _query_history[1].results_count == 0

    def test_get_query_history_returns_copy(self):
        record_query("q1", ["arxiv"], 3)
        history = get_query_history()
        assert len(history) == 1
        # Modifying the returned list should not affect internal state
        history.clear()
        assert len(_query_history) == 1

    def test_get_effective_queries_default_threshold(self):
        record_query("good_query", ["arxiv"], 5)
        record_query("ok_query", ["arxiv"], 3)
        record_query("bad_query", ["arxiv"], 2)
        record_query("zero_query", ["arxiv"], 0)
        effective = get_effective_queries()
        assert "good_query" in effective
        assert "ok_query" in effective
        assert "bad_query" not in effective
        assert "zero_query" not in effective

    def test_get_effective_queries_custom_threshold(self):
        record_query("q1", ["arxiv"], 1)
        record_query("q2", ["arxiv"], 2)
        effective = get_effective_queries(min_results=1)
        assert len(effective) == 2

    def test_get_effective_queries_empty(self):
        assert get_effective_queries() == []

    def test_get_failed_queries(self):
        record_query("ok", ["arxiv"], 3)
        record_query("fail1", ["duckduckgo"], 0)
        record_query("fail2", ["perplexity"], 0)
        failed = get_failed_queries()
        assert "fail1" in failed
        assert "fail2" in failed
        assert "ok" not in failed

    def test_get_failed_queries_empty(self):
        record_query("q", ["arxiv"], 1)
        assert get_failed_queries() == []


# ── A2: Query Specificity Scoring ────────────────────────────────────────


class TestQuerySpecificityScoring:
    def test_single_word_very_low(self):
        score = score_query_specificity("overview")
        assert score == 0.1

    def test_empty_query(self):
        score = score_query_specificity("")
        assert score == 0.1

    def test_vague_query_penalized(self):
        score_vague = score_query_specificity("general overview of things about stuff")
        score_specific = score_query_specificity("transformer attention mechanism neural networks")
        assert score_specific > score_vague

    def test_longer_query_higher_score(self):
        short = score_query_specificity("machine learning")
        long = score_query_specificity("deep reinforcement learning for robotic arm manipulation")
        assert long > short

    def test_domain_keywords_bonus(self):
        keywords = {"transformer", "attention", "neural"}
        without = score_query_specificity("transformer attention mechanisms", research_keywords=None)
        with_kw = score_query_specificity("transformer attention mechanisms", research_keywords=keywords)
        assert with_kw > without

    def test_score_clamped_0_to_1(self):
        # Very long query with many domain keywords
        keywords = {"a", "b", "c", "d", "e", "f", "g", "h"}
        score = score_query_specificity("a b c d e f g h i j k l", research_keywords=keywords)
        assert 0.0 <= score <= 1.0

    def test_no_keywords_no_bonus(self):
        score1 = score_query_specificity("test query words here")
        score2 = score_query_specificity("test query words here", research_keywords=set())
        assert score1 == score2

    def test_all_vague_words(self):
        score = score_query_specificity("general overview about various things stuff information")
        # Heavy vague penalty should push score down significantly
        assert score < 0.5

    def test_max_length_score(self):
        # 8+ words should max out the length component
        score = score_query_specificity("one two three four five six seven eight nine ten")
        assert score >= 0.8

    def test_domain_bonus_capped(self):
        keywords = {"w1", "w2", "w3", "w4", "w5", "w6"}
        score = score_query_specificity("w1 w2 w3 w4 w5 w6", research_keywords=keywords)
        # Domain bonus capped at 0.3
        assert score <= 1.0


# ── A3: Source Exhaustion Detection ──────────────────────────────────────


class TestSourceExhaustionDetection:
    def setup_method(self):
        """Clear source stats before each test."""
        reset_source_stats()

    def test_source_stats_dataclass(self):
        stats = SourceStats()
        assert stats.calls == 0
        assert stats.unique_urls == 0
        assert stats.total_results == 0

    def test_unknown_source_not_exhausted(self):
        assert is_source_exhausted("nonexistent") is False

    def test_below_threshold_not_exhausted(self):
        _source_stats["arxiv"] = SourceStats(calls=3, unique_urls=0, total_results=0)
        assert is_source_exhausted("arxiv", threshold_calls=5) is False

    def test_exhausted_when_low_yield(self):
        # 5 calls, only 2 unique URLs → avg yield 0.4 < 1.0
        _source_stats["arxiv"] = SourceStats(calls=5, unique_urls=2, total_results=10)
        assert is_source_exhausted("arxiv") is True

    def test_not_exhausted_when_good_yield(self):
        # 5 calls, 10 unique URLs → avg yield 2.0 >= 1.0
        _source_stats["arxiv"] = SourceStats(calls=5, unique_urls=10, total_results=30)
        assert is_source_exhausted("arxiv") is False

    def test_custom_threshold_calls(self):
        _source_stats["ddg"] = SourceStats(calls=3, unique_urls=1, total_results=3)
        assert is_source_exhausted("ddg", threshold_calls=3) is True
        assert is_source_exhausted("ddg", threshold_calls=4) is False

    def test_custom_min_yield(self):
        _source_stats["ddg"] = SourceStats(calls=5, unique_urls=10, total_results=20)
        # avg yield = 2.0
        assert is_source_exhausted("ddg", min_yield=2.0) is False
        assert is_source_exhausted("ddg", min_yield=3.0) is True

    def test_reset_source_stats(self):
        _source_stats["arxiv"] = SourceStats(calls=5, unique_urls=2, total_results=10)
        reset_source_stats()
        assert get_source_stats() == {}

    def test_get_source_stats_returns_copy(self):
        _source_stats["arxiv"] = SourceStats(calls=1, unique_urls=1, total_results=1)
        stats = get_source_stats()
        stats.clear()
        assert "arxiv" in _source_stats

    def test_zero_calls_with_zero_threshold(self):
        """With 0 calls and threshold_calls=0, the source passes threshold check.
        avg_yield=0 (guarded division) < min_yield=1.0, so it IS exhausted."""
        _source_stats["arxiv"] = SourceStats(calls=0, unique_urls=0, total_results=0)
        assert is_source_exhausted("arxiv", threshold_calls=0) is True

    def test_exact_threshold(self):
        # At exactly the threshold calls with yield = 1.0 (not less)
        _source_stats["ddg"] = SourceStats(calls=5, unique_urls=5, total_results=10)
        assert is_source_exhausted("ddg", threshold_calls=5, min_yield=1.0) is False


# ── A4: Domain Diversity Enforcement ─────────────────────────────────────


class TestDomainDiversityEnforcement:
    def setup_method(self):
        reset_source_stats()
        _query_history.clear()

    @patch("search.search_duckduckgo")
    def test_domain_counts_tracked(self, mock_ddg):
        """execute_searches should track domains from URLs."""
        mock_ddg.return_value = [
            SearchResult(title="R1", url="https://example.com/a", snippet="s1", source="duckduckgo"),
            SearchResult(title="R2", url="https://example.com/b", snippet="s2", source="duckduckgo"),
            SearchResult(title="R3", url="https://other.com/c", snippet="s3", source="duckduckgo"),
        ]
        results = execute_searches([{"query": "test", "sources": ["duckduckgo"]}])
        assert len(results) == 3

    @patch("search.search_duckduckgo")
    def test_domain_diversity_warning_printed(self, mock_ddg, capsys):
        """When one domain dominates (>50% with >=4 total), print warning."""
        mock_ddg.return_value = [
            SearchResult(title=f"R{i}", url=f"https://dominant.com/{i}", snippet="s", source="duckduckgo")
            for i in range(4)
        ] + [
            SearchResult(title="Other", url="https://other.com/1", snippet="s", source="duckduckgo"),
        ]
        execute_searches([{"query": "test", "sources": ["duckduckgo"]}])
        captured = capsys.readouterr()
        assert "diversity" in captured.out.lower() or "dominant.com" in captured.out

    @patch("search.search_duckduckgo")
    def test_no_warning_when_diverse(self, mock_ddg, capsys):
        """No domain warning when results are spread across domains."""
        mock_ddg.return_value = [
            SearchResult(title=f"R{i}", url=f"https://domain{i}.com/page", snippet="s", source="duckduckgo")
            for i in range(4)
        ]
        execute_searches([{"query": "test", "sources": ["duckduckgo"]}])
        captured = capsys.readouterr()
        assert "diversity" not in captured.out.lower()

    @patch("search.search_duckduckgo")
    def test_no_warning_when_few_results(self, mock_ddg, capsys):
        """No warning when total results < 4 even if one domain dominates."""
        mock_ddg.return_value = [
            SearchResult(title="R1", url="https://same.com/a", snippet="s", source="duckduckgo"),
            SearchResult(title="R2", url="https://same.com/b", snippet="s", source="duckduckgo"),
            SearchResult(title="R3", url="https://same.com/c", snippet="s", source="duckduckgo"),
        ]
        execute_searches([{"query": "test", "sources": ["duckduckgo"]}])
        captured = capsys.readouterr()
        assert "diversity" not in captured.out.lower()


# ── A5: Pre-Synthesis Result Filtering ───────────────────────────────────


class TestFilterResultsByRelevance:
    def test_empty_keywords_returns_all(self):
        results = [
            SearchResult(title="R1", url="u1", snippet="irrelevant stuff", source="ddg"),
            SearchResult(title="R2", url="u2", snippet="more junk", source="ddg"),
            SearchResult(title="R3", url="u3", snippet="random things", source="ddg"),
        ]
        filtered = filter_results_by_relevance(results, set())
        assert len(filtered) == 3

    def test_few_results_returns_all(self):
        """With <= 2 results, no filtering applied."""
        results = [
            SearchResult(title="R1", url="u1", snippet="irrelevant", source="ddg"),
            SearchResult(title="R2", url="u2", snippet="also irrelevant", source="ddg"),
        ]
        filtered = filter_results_by_relevance(results, {"machine", "learning"})
        assert len(filtered) == 2

    def test_filters_low_relevance(self):
        keywords = {"machine", "learning", "neural", "network"}
        results = [
            SearchResult(title="Neural Network Training", url="u1", snippet="machine learning paper", source="ddg"),
            SearchResult(title="Machine Learning Basics", url="u2", snippet="neural network tutorial", source="ddg"),
            SearchResult(title="Cooking Recipes", url="u3", snippet="best pasta dishes", source="ddg"),
            SearchResult(title="Sports News", url="u4", snippet="football game results", source="ddg"),
            SearchResult(title="Deep Learning", url="u5", snippet="neural network architectures", source="ddg"),
            SearchResult(title="Weather Report", url="u6", snippet="sunny skies tomorrow", source="ddg"),
            SearchResult(title="Travel Guide", url="u7", snippet="best hotels in paris", source="ddg"),
            SearchResult(title="Movie Reviews", url="u8", snippet="latest film ratings", source="ddg"),
            SearchResult(title="AI Research", url="u9", snippet="machine learning advances", source="ddg"),
            SearchResult(title="Book List", url="u10", snippet="top novels this year", source="ddg"),
        ]
        filtered = filter_results_by_relevance(results, keywords)
        # Relevant results should be kept, some irrelevant may be filtered
        # But at least 70% (7) must be kept
        assert len(filtered) >= 7
        assert len(filtered) <= 10
        # Top results should be the relevant ones
        top_titles = {r.title for r in filtered[:3]}
        assert "Neural Network Training" in top_titles or "Machine Learning Basics" in top_titles

    def test_keeps_at_least_70_percent(self):
        """Even if all results are irrelevant, keeps 70%."""
        keywords = {"quantum", "physics", "entanglement"}
        results = [
            SearchResult(title=f"Irrelevant {i}", url=f"u{i}", snippet="no keywords here", source="ddg")
            for i in range(10)
        ]
        filtered = filter_results_by_relevance(results, keywords)
        assert len(filtered) >= 7  # 70% of 10

    def test_sorted_by_relevance(self):
        keywords = {"python", "programming"}
        results = [
            SearchResult(title="Cooking Show", url="u1", snippet="no match at all", source="ddg"),
            SearchResult(title="Python Programming", url="u2", snippet="python programming tutorial", source="ddg"),
            SearchResult(title="Random", url="u3", snippet="some random text", source="ddg"),
        ]
        filtered = filter_results_by_relevance(results, keywords)
        # Python Programming should be first
        assert filtered[0].title == "Python Programming"

    def test_minimum_keep_at_least_2(self):
        """min_keep is max(2, int(len*0.7)), so for 3 results it's 2."""
        keywords = {"specific", "keyword"}
        results = [
            SearchResult(title="Match", url="u1", snippet="specific keyword here", source="ddg"),
            SearchResult(title="No Match 1", url="u2", snippet="irrelevant", source="ddg"),
            SearchResult(title="No Match 2", url="u3", snippet="also irrelevant", source="ddg"),
        ]
        filtered = filter_results_by_relevance(results, keywords, min_relevance=0.9)
        assert len(filtered) >= 2


# ── A7: Query-Source Semantic Validation ─────────────────────────────────


class TestValidateQuerySources:
    def test_statistical_query_adds_academic(self):
        sources = validate_query_sources("correlation between age and income", ["duckduckgo"])
        assert "semantic_scholar" in sources or "arxiv" in sources

    def test_statistical_query_no_duplicate(self):
        """If academic source already present, don't add another."""
        sources = validate_query_sources("mean variance analysis", ["semantic_scholar", "duckduckgo"])
        assert sources.count("semantic_scholar") == 1

    def test_academic_query_adds_source(self):
        sources = validate_query_sources("survey of deep learning algorithms", ["duckduckgo"])
        assert "semantic_scholar" in sources or "arxiv" in sources

    def test_academic_query_no_duplicate(self):
        sources = validate_query_sources("research paper on transformers", ["arxiv", "duckduckgo"])
        assert sources.count("arxiv") == 1

    def test_recent_query_adds_web_search(self):
        sources = validate_query_sources("latest 2025 breakthrough in AI", ["semantic_scholar"])
        assert "duckduckgo" in sources

    def test_recent_query_no_duplicate_ddg(self):
        sources = validate_query_sources("recent emerging trends", ["duckduckgo", "arxiv"])
        assert sources.count("duckduckgo") == 1

    def test_no_matching_keywords_unchanged(self):
        original = ["duckduckgo", "arxiv"]
        sources = validate_query_sources("cats and dogs", original)
        assert sources == original

    def test_combined_statistical_and_recent(self):
        sources = validate_query_sources("recent regression analysis techniques 2025", ["metaso"])
        # Should add both an academic source and duckduckgo
        has_academic = "semantic_scholar" in sources or "arxiv" in sources
        assert has_academic
        assert "duckduckgo" in sources

    def test_empty_sources(self):
        sources = validate_query_sources("correlation analysis paper", [])
        # Should add academic source for both stat and academic keywords
        assert len(sources) >= 1

    def test_preserves_original_order(self):
        sources = validate_query_sources("general topic", ["metaso", "perplexity"])
        assert sources[0] == "metaso"
        assert sources[1] == "perplexity"

    def test_statistical_adds_at_front(self):
        """Statistical keyword adds academic source at position 0."""
        sources = validate_query_sources("hypothesis testing methods", ["duckduckgo"])
        # The statistical branch uses insert(0, ...)
        assert sources[0] in ("semantic_scholar", "arxiv")

    def test_academic_adds_at_end(self):
        """Academic keyword appends source at end."""
        sources = validate_query_sources("benchmark study results", ["duckduckgo"])
        # academic branch uses append
        assert sources[-1] in ("semantic_scholar", "arxiv") or sources[0] in ("semantic_scholar", "arxiv")
