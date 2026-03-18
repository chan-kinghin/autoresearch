"""
search.py — Fixed utilities for the autonomous research agent.

Provides: LLM calls, search APIs (3 tiers), content extraction, evaluation.
This file is not meant to be edited per-project; configure via env vars.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

# ── Configuration ──────────────────────────────────────────────────────────

MODEL = os.environ.get("AUTORESEARCH_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS = int(os.environ.get("AUTORESEARCH_MAX_TOKENS", "4096"))
TEMPERATURE = float(os.environ.get("AUTORESEARCH_TEMPERATURE", "0.3"))
HTTP_TIMEOUT = int(os.environ.get("AUTORESEARCH_HTTP_TIMEOUT", "60"))
LLM_TIMEOUT = int(os.environ.get("AUTORESEARCH_LLM_TIMEOUT", "120"))

# ── Provider Routing ───────────────────────────────────────────────────────
#
# MODEL format: "provider/model-id" or just "model-id" (auto-detected)
# Examples:
#   "claude-sonnet-4-20250514"       → Anthropic
#   "openai/gpt-4o"                  → OpenAI
#   "deepseek/deepseek-chat"         → DeepSeek
#   "gemini/gemini-2.0-flash"        → Google Gemini

PROVIDER_CONFIG = {
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/messages",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "env_key": "GOOGLE_API_KEY",
    },
}


def _detect_provider(model: str) -> Tuple[str, str]:
    """Detect provider from model string. Returns (provider, model_id)."""
    if "/" in model:
        provider, model_id = model.split("/", 1)
        return provider, model_id

    # Auto-detect by model name prefix
    if model.startswith("claude"):
        return "anthropic", model
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai", model
    if model.startswith("deepseek"):
        return "deepseek", model
    if model.startswith("gemini"):
        return "gemini", model

    # Default to OpenAI-compatible
    return "openai", model


# ── Data Types ─────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str  # "semantic_scholar", "arxiv", "duckduckgo", "metaso", etc.
    authors: str = ""
    year: str = ""
    full_text: str = ""


@dataclass
class EvaluationResult:
    coverage_score: float
    questions_answered: List[str] = field(default_factory=list)
    questions_remaining: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    suggested_queries: List[str] = field(default_factory=list)


@dataclass
class IterationLog:
    iteration: int
    coverage: float
    sources: int
    duration_s: float
    status: str  # "continue", "target_reached", "max_iterations", "error"


# ── A1: Query Success Tracking ────────────────────────────────────────────


@dataclass
class QueryRecord:
    query: str
    sources: List[str]
    results_count: int
    useful_results: int = 0
    iteration: int = 0

_query_history: List[QueryRecord] = []

def record_query(query: str, sources: List[str], results_count: int, iteration: int = 0) -> None:
    """Record a query execution for learning."""
    _query_history.append(QueryRecord(query=query, sources=sources, results_count=results_count, iteration=iteration))

def get_query_history() -> List[QueryRecord]:
    """Return the full query history."""
    return list(_query_history)

def get_effective_queries(min_results: int = 3) -> List[str]:
    """Return queries that yielded good results, for prompt injection."""
    return [q.query for q in _query_history if q.results_count >= min_results]

def get_failed_queries() -> List[str]:
    """Return queries that yielded no results."""
    return [q.query for q in _query_history if q.results_count == 0]


# ── A2: Query Specificity Scoring ─────────────────────────────────────────


def score_query_specificity(query: str, research_keywords: set[str] | None = None) -> float:
    """Score query specificity 0.0-1.0. Higher = more specific.

    Checks: length, domain terms, absence of vague words.
    """
    words = query.lower().split()
    if len(words) < 2:
        return 0.1

    vague_words = {"things", "stuff", "general", "various", "overview", "information", "about"}
    vague_count = sum(1 for w in words if w in vague_words)

    # Base score from word count (longer queries tend to be more specific)
    length_score = min(1.0, len(words) / 8)

    # Penalty for vague words
    vague_penalty = vague_count * 0.15

    # Bonus for domain keywords
    domain_bonus = 0.0
    if research_keywords:
        overlap = sum(1 for w in words if w in research_keywords)
        domain_bonus = min(0.3, overlap * 0.1)

    return max(0.0, min(1.0, length_score - vague_penalty + domain_bonus))


# ── A3: Source Exhaustion Detection ───────────────────────────────────────


@dataclass
class SourceStats:
    calls: int = 0
    unique_urls: int = 0
    total_results: int = 0

_source_stats: Dict[str, SourceStats] = {}

def get_source_stats() -> Dict[str, SourceStats]:
    """Return per-source call statistics."""
    return dict(_source_stats)

def is_source_exhausted(source: str, threshold_calls: int = 5, min_yield: float = 1.0) -> bool:
    """Check if a source is returning diminishing results."""
    stats = _source_stats.get(source)
    if not stats or stats.calls < threshold_calls:
        return False
    avg_yield = stats.unique_urls / stats.calls if stats.calls > 0 else 0
    return avg_yield < min_yield

def reset_source_stats() -> None:
    """Reset source stats (call at start of each research run)."""
    _source_stats.clear()


# ── A5: Pre-Synthesis Result Filtering ────────────────────────────────────


def filter_results_by_relevance(
    results: List[SearchResult],
    research_keywords: set[str],
    min_relevance: float = 0.2,
) -> List[SearchResult]:
    """Filter out results with very low relevance to the research topic.

    Uses keyword overlap between result content and research keywords.
    Keeps at least 70% of results to avoid over-filtering.
    """
    if not research_keywords or len(results) <= 2:
        return results

    scored: List[Tuple[float, SearchResult]] = []
    for r in results:
        text = f"{r.title} {r.snippet}".lower()
        text_words = set(re.split(r'[^a-zA-Z0-9]+', text))
        overlap = len(text_words & research_keywords)
        relevance = overlap / max(len(research_keywords), 1)
        scored.append((relevance, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep at least 70% of results
    min_keep = max(2, int(len(results) * 0.7))
    filtered = []
    for relevance, r in scored:
        if len(filtered) >= min_keep and relevance < min_relevance:
            break
        filtered.append(r)

    if len(filtered) < len(results):
        print(f"  [filter] Removed {len(results) - len(filtered)} low-relevance results")

    return filtered


# ── A7: Query-Source Semantic Validation ──────────────────────────────────


_STATISTICAL_KEYWORDS = {"mean", "median", "variance", "correlation", "distribution",
                          "statistic", "sample", "hypothesis", "regression", "p-value"}
_RECENT_KEYWORDS = {"recent", "2024", "2025", "2026", "latest", "new", "emerging", "breakthrough"}
_ACADEMIC_KEYWORDS = {"paper", "study", "research", "journal", "survey", "review",
                       "benchmark", "algorithm", "theorem", "proof"}

def validate_query_sources(query: str, sources: List[str]) -> List[str]:
    """Validate and adjust query-source pairings based on query content."""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    adjusted = list(sources)

    # Statistical query without academic sources?
    if query_words & _STATISTICAL_KEYWORDS:
        for academic in ("semantic_scholar", "arxiv"):
            if academic not in adjusted:
                adjusted.insert(0, academic)
                break

    # Academic query without academic sources?
    if query_words & _ACADEMIC_KEYWORDS:
        for academic in ("semantic_scholar", "arxiv"):
            if academic not in adjusted:
                adjusted.append(academic)
                break

    # Recent events query should include web search
    if query_words & _RECENT_KEYWORDS:
        if "duckduckgo" not in adjusted:
            adjusted.append("duckduckgo")

    return adjusted


# ── LLM Calls ─────────────────────────────────────────────────────────────


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRY_DELAYS = [1, 2, 4]


def _call_anthropic(
    model_id: str, system: str, prompt: str, max_tokens: int, temperature: float
) -> str:
    """Call Anthropic Messages API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    last_exc: Exception | None = None
    with httpx.Client(timeout=LLM_TIMEOUT) as client:
        for attempt in range(1, len(_RETRY_DELAYS) + 2):  # 1..4 (max 3 retries)
            try:
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system": system,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if resp.status_code in _RETRYABLE_STATUS and attempt <= len(_RETRY_DELAYS):
                    print(f"  [llm] Retry {attempt}/3 after {resp.status_code}...")
                    time.sleep(_RETRY_DELAYS[attempt - 1])
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except httpx.TimeoutException as e:
                last_exc = e
                if attempt <= len(_RETRY_DELAYS):
                    print(f"  [llm] Retry {attempt}/3 after timeout...")
                    time.sleep(_RETRY_DELAYS[attempt - 1])
                    continue
                raise
        else:
            raise last_exc  # type: ignore[misc]
    content = data.get("content") or []
    if not content or not isinstance(content[0], dict):
        raise ValueError(f"Unexpected Anthropic response shape: {data!r:.200}")
    return content[0].get("text", "").strip()


def _call_openai_compatible(
    base_url: str, api_key: str, model_id: str,
    system: str, prompt: str, max_tokens: int, temperature: float,
) -> str:
    """Call any OpenAI-compatible chat completions API."""
    last_exc: Exception | None = None
    with httpx.Client(timeout=LLM_TIMEOUT) as client:
        for attempt in range(1, len(_RETRY_DELAYS) + 2):
            try:
                resp = client.post(
                    base_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                if resp.status_code in _RETRYABLE_STATUS and attempt <= len(_RETRY_DELAYS):
                    print(f"  [llm] Retry {attempt}/3 after {resp.status_code}...")
                    time.sleep(_RETRY_DELAYS[attempt - 1])
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except httpx.TimeoutException as e:
                last_exc = e
                if attempt <= len(_RETRY_DELAYS):
                    print(f"  [llm] Retry {attempt}/3 after timeout...")
                    time.sleep(_RETRY_DELAYS[attempt - 1])
                    continue
                raise
        else:
            raise last_exc  # type: ignore[misc]
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"Unexpected OpenAI-compatible response: no choices in {data!r:.200}")
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def llm_call(
    prompt: str,
    system: str = "You are a research assistant.",
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Make a single LLM call. Auto-routes to the correct provider."""
    model = model or MODEL
    max_tok = max_tokens or MAX_TOKENS
    temp = temperature if temperature is not None else TEMPERATURE

    provider, model_id = _detect_provider(model)

    if provider == "anthropic":
        return _call_anthropic(model_id, system, prompt, max_tok, temp)

    # All others use OpenAI-compatible format
    config = PROVIDER_CONFIG.get(provider)
    if config is None:
        print(f"  [llm] Warning: unknown provider '{provider}', falling back to OpenAI-compatible format")
        config = PROVIDER_CONFIG["openai"]
    api_key = os.environ.get(config["env_key"], "")
    if not api_key:
        raise ValueError(f"{config['env_key']} not set for provider '{provider}'")

    return _call_openai_compatible(
        config["base_url"], api_key, model_id, system, prompt, max_tok, temp
    )


def llm_json(
    prompt: str,
    system: str = "You are a research assistant. Respond ONLY with valid JSON, no markdown fences.",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Make an LLM call and parse the response as JSON."""
    raw = llm_call(prompt, system=system, model=model)
    # Strip markdown code fences if present
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Second chance: find first {...} block in raw text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"LLM returned invalid JSON — raw response: {raw[:200]}")


# ── Tier 1: Deep Research Services ─────────────────────────────────────────


def search_metaso(query: str, mode: str = "research") -> List[SearchResult]:
    """Search via 秘塔搜索 (Metaso) API. Requires METASO_API_KEY.

    The API returns Server-Sent Events (SSE). We parse:
      - "append-text" events → concatenated into full_text
      - "set-reference" events → individual source results
    """
    api_key = os.environ.get("METASO_API_KEY")
    if not api_key:
        return []

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.post(
                "https://metaso.cn/api/open/search",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"question": query, "mode": mode},
            )
            resp.raise_for_status()

        # Parse SSE stream
        text_chunks: List[str] = []
        references: List[Dict[str, Any]] = []
        for line in resp.text.split("\n"):
            if not line.startswith("data:") or "[DONE]" in line:
                continue
            try:
                obj = json.loads(line[5:])
                evt_type = obj.get("type")
                if evt_type == "append-text":
                    text_chunks.append(obj.get("text", ""))
                elif evt_type == "set-reference":
                    references = obj.get("list", [])
            except (json.JSONDecodeError, ValueError):
                continue

        content = "".join(text_chunks)
        results: List[SearchResult] = []
        if content:
            results.append(SearchResult(
                title=f"Metaso Research: {query}",
                url=f"https://metaso.cn/search?q={urllib.parse.quote(query)}",
                snippet=content[:500],
                source="metaso",
                full_text=content,
            ))
        for ref in references:
            results.append(SearchResult(
                title=ref.get("title", ""),
                url=ref.get("link", ""),
                snippet=ref.get("title", ""),
                source="metaso",
            ))
        return results
    except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError, OSError, ValueError, KeyError) as e:
        print(f"  [metaso] Error: {e}")
        return []


def search_perplexity(query: str) -> List[SearchResult]:
    """Search via Perplexity sonar API. Requires PERPLEXITY_API_KEY."""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return []

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": query}],
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results: List[SearchResult] = []
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content:
            results.append(SearchResult(
                title=f"Perplexity: {query}",
                url="https://perplexity.ai",
                snippet=content[:500],
                source="perplexity",
                full_text=content,
            ))
        for citation in data.get("citations", []):
            if isinstance(citation, str):
                results.append(SearchResult(
                    title=citation, url=citation, snippet="", source="perplexity"
                ))
        return results
    except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError, OSError, ValueError, KeyError) as e:
        print(f"  [perplexity] Error: {e}")
        return []


def search_gemini_deep(query: str) -> List[SearchResult]:
    """Use Gemini as a deep research service. Requires GOOGLE_API_KEY."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return []

    try:
        content = _call_openai_compatible(
            PROVIDER_CONFIG["gemini"]["base_url"],
            api_key,
            "gemini-2.0-flash",
            "You are a research assistant. Provide a thorough, well-sourced answer.",
            f"Research the following topic thoroughly:\n\n{query}",
            4096,
            0.3,
        )
        if content:
            return [SearchResult(
                title=f"Gemini Research: {query}",
                url=f"gemini-synthesis://{query[:80]}",
                snippet=content[:500],
                source="gemini",
                full_text=content,
            )]
        return []
    except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError, OSError, ValueError, KeyError, Exception) as e:
        print(f"  [gemini] Error: {e}")
        return []


# ── Tier 2: Academic APIs ──────────────────────────────────────────────────


def search_semantic_scholar(
    query: str, limit: int = 10, year: Optional[str] = None,
) -> List[SearchResult]:
    """Search Semantic Scholar (no API key required, rate-limited)."""
    try:
        params: Dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,url,externalIds",
        }
        if year:
            params["year"] = year

        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
            )
            if resp.status_code == 429:
                for attempt, wait in enumerate([3, 6], start=1):
                    print(f"  [semantic_scholar] Rate limited, waiting {wait}s (attempt {attempt})...")
                    time.sleep(wait)
                    resp = client.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params=params,
                    )
                    if resp.status_code != 429:
                        break
            resp.raise_for_status()
            data = resp.json()

        results: List[SearchResult] = []
        for paper in data.get("data", []):
            authors = ", ".join(a.get("name", "") for a in paper.get("authors", []))
            paper_url = paper.get("url", "")
            if not paper_url and paper.get("externalIds", {}).get("DOI"):
                paper_url = f"https://doi.org/{paper['externalIds']['DOI']}"
            results.append(SearchResult(
                title=paper.get("title", ""),
                url=paper_url,
                snippet=paper.get("abstract", "") or "",
                source="semantic_scholar",
                authors=authors,
                year=str(paper.get("year", "")),
            ))
        return results
    except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError, OSError, ValueError, KeyError) as e:
        print(f"  [semantic_scholar] Error: {e}")
        return []


def search_arxiv(query: str, max_results: int = 10) -> List[SearchResult]:
    """Search arXiv via its Atom API (no API key required)."""
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        results: List[SearchResult] = []
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
            link = ""
            for lnk in entry.findall("atom:link", ns):
                if lnk.get("type") == "text/html" or lnk.get("rel") == "alternate":
                    link = lnk.get("href", "")
                    break
            authors = ", ".join(
                a.findtext("atom:name", "", ns)
                for a in entry.findall("atom:author", ns)
            )
            published = (entry.findtext("atom:published", "", ns) or "")[:4]

            results.append(SearchResult(
                title=title,
                url=link,
                snippet=summary[:500],
                source="arxiv",
                authors=authors,
                year=published,
            ))
        return results
    except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError, OSError, ValueError, KeyError, ET.ParseError) as e:
        print(f"  [arxiv] Error: {e}")
        return []


# ── Tier 3: Web Search ─────────────────────────────────────────────────────


def search_duckduckgo(query: str, max_results: int = 10) -> List[SearchResult]:
    """Search DuckDuckGo (no API key required)."""
    try:
        from duckduckgo_search import DDGS

        results: List[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    source="duckduckgo",
                ))
        return results
    except ImportError:
        print("  [duckduckgo] Package not installed — run: uv add duckduckgo-search")
        return []
    except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError, OSError, ValueError, KeyError) as e:
        print(f"  [duckduckgo] Error: {e}")
        return []


# ── Content Extraction ─────────────────────────────────────────────────────


def extract_webpage(url: str) -> str:
    """Extract main text content from a webpage using trafilatura."""
    try:
        import trafilatura

        with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
        text = trafilatura.extract(resp.text)
        return text or ""
    except ImportError:
        print("  [extract] Package not installed — run: uv add trafilatura")
        return ""
    except httpx.HTTPStatusError as e:
        print(f"  [extract] HTTP {e.response.status_code} for {url}")
        return ""
    except httpx.ConnectError as e:
        print(f"  [extract] Connection failed for {url}: {e}")
        return ""
    except (httpx.HTTPError, httpx.TimeoutException, OSError, ValueError) as e:
        print(f"  [extract] Error extracting {url}: {type(e).__name__}: {e}")
        return ""


# ── Query Reformulation ────────────────────────────────────────────────────

_QUESTION_WORDS = {"what", "how", "why", "when", "where", "who", "which", "is", "are", "do", "does", "can", "could", "would", "should"}


def reformulate_query(query: str, source: str) -> str:
    """Adapt a natural-language query for a specific search engine."""
    if source == "arxiv":
        # Convert to arXiv query syntax: key terms joined with AND
        words = query.split()
        terms = [w for w in words if w.lower() not in _QUESTION_WORDS and len(w) > 2]
        if len(terms) >= 2:
            # Use first term in title, rest in abstract
            parts = [f"ti:{terms[0]}"] + [f"abs:{t}" for t in terms[1:]]
            return " AND ".join(parts)
        return query
    if source == "semantic_scholar":
        # Strip question words, keep noun phrases for keyword search
        words = query.split()
        terms = [w for w in words if w.lower() not in _QUESTION_WORDS]
        return " ".join(terms) if terms else query
    return query


# ── Unified Search ─────────────────────────────────────────────────────────


def execute_searches(
    queries: List[Dict[str, Any]],
) -> List[SearchResult]:
    """Execute a list of search plans. Each plan: {"query": str, "sources": [str]}.

    Valid sources: "metaso", "perplexity", "gemini", "semantic_scholar", "arxiv", "duckduckgo"
    """
    all_results: List[SearchResult] = []
    seen_urls: Set[str] = set()
    domain_counts: Dict[str, int] = {}  # A4: Track domain distribution

    dispatch = {
        "metaso": search_metaso,
        "perplexity": search_perplexity,
        "gemini": search_gemini_deep,
        "semantic_scholar": search_semantic_scholar,
        "arxiv": search_arxiv,
        "duckduckgo": search_duckduckgo,
    }

    for plan in queries:
        query = plan["query"]
        sources = plan.get("sources", ["duckduckgo"])
        for src in sources:
            # A3: Check source exhaustion
            if is_source_exhausted(src):
                print(f"  [search] Skipping exhausted source: {src}")
                continue

            fn = dispatch.get(src)
            if not fn:
                print(f"  [search] Unknown source: {src}")
                continue
            adapted_query = reformulate_query(query, src)
            print(f"  Searching {src}: {adapted_query[:60]}...")
            results = fn(adapted_query)

            # A3: Update source stats
            if src not in _source_stats:
                _source_stats[src] = SourceStats()
            _source_stats[src].calls += 1
            _source_stats[src].total_results += len(results)

            new_count = 0
            for r in results:
                if r.url and r.url in seen_urls:
                    continue
                if r.url:
                    seen_urls.add(r.url)
                    # A4: Track domain
                    try:
                        domain = r.url.split("/")[2] if r.url.startswith("http") else src
                    except IndexError:
                        domain = src
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                all_results.append(r)
                new_count += 1

            _source_stats[src].unique_urls += new_count

        # A1: Record query results
        record_query(query, sources, sum(1 for r in all_results if any(
            r.url and r.url not in seen_urls for _ in [None]  # just count new
        )), iteration=0)

    # A4: Warn about domain concentration
    if domain_counts:
        total = sum(domain_counts.values())
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            if count / total > 0.5 and total >= 4:
                print(f"  [diversity] Warning: {domain} dominates results ({count}/{total} = {count/total:.0%})")
                break

    return all_results


# ── Progress Logging ───────────────────────────────────────────────────────


def init_progress_log(path: str = "progress.tsv") -> None:
    """Create the progress log file if it does not already exist."""
    try:
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("iteration\tcoverage\tsources\tduration_s\tstatus\n")
        # If file exists, don't overwrite it
    except OSError as e:
        print(f"  [log] Warning: could not create progress log {path}: {e}")


def log_iteration(log: IterationLog, path: str = "progress.tsv") -> None:
    """Append an iteration log entry."""
    try:
        with open(path, "a") as f:
            f.write(f"{log.iteration}\t{log.coverage:.2f}\t{log.sources}\t{log.duration_s:.1f}\t{log.status}\n")
    except OSError as e:
        print(f"  [log] Warning: could not write to progress log {path}: {e}")
