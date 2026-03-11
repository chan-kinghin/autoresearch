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
    evaluate_coverage,
    init_progress_log,
    log_iteration,
    extract_webpage,
)


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
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "content": "Research content about topic" * 20,
                "sources": [
                    {"title": "Source 1", "url": "https://src1.com", "snippet": "s1"},
                    {"title": "Source 2", "url": "https://src2.com", "snippet": "s2"},
                ],
            }
        }
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
    @patch("search.llm_json")
    def test_success(self, mock_llm_json):
        mock_llm_json.return_value = {
            "coverage_score": 0.75,
            "questions_answered": ["Q1"],
            "questions_remaining": ["Q2"],
            "gaps": ["gap1"],
            "suggested_queries": ["query1"],
        }

        result = evaluate_coverage("program", "findings")
        assert isinstance(result, EvaluationResult)
        assert result.coverage_score == 0.75
        assert result.questions_answered == ["Q1"]
        assert result.questions_remaining == ["Q2"]
        assert result.gaps == ["gap1"]
        assert result.suggested_queries == ["query1"]

    @patch("search.llm_json")
    def test_llm_error_fallback(self, mock_llm_json):
        mock_llm_json.side_effect = Exception("LLM failed")

        result = evaluate_coverage("program", "findings")
        assert result.coverage_score == 0.0
        assert "LLM failed" in result.gaps[0]

    @patch("search.llm_json")
    def test_partial_response(self, mock_llm_json):
        """LLM returns only some fields; defaults should fill in."""
        mock_llm_json.return_value = {
            "coverage_score": 0.5,
        }

        result = evaluate_coverage("program", "findings")
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
