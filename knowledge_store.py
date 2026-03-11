"""
knowledge_store.py — Progressive disclosure knowledge store.

3-layer architecture:
  Layer 1 (Index):    Always loaded. ~500 tokens. Topics, questions, relations, meta.
  Layer 2 (Summaries): On-demand. ~1-2k tokens each. Per-topic markdown with frontmatter.
  Layer 3 (Details):  Drill-down only. Source records, raw search results.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ── Constants ────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "it", "its", "not", "no", "what", "how", "which", "who",
})


# ── Data Types ────────────────────────────────────────────────────────────


@dataclass
class Topic:
    id: str                      # slug, e.g. "architecture_patterns"
    title: str                   # "Architecture Patterns"
    keywords: list[str] = field(default_factory=list)
    coverage: float = 0.0
    source_count: int = 0
    unsynthesized_sources: int = 0
    stale: bool = False


@dataclass
class Question:
    id: str
    text: str
    status: str = "unanswered"   # "unanswered", "partial", "answered"
    related_topics: list[str] = field(default_factory=list)


@dataclass
class Relation:
    from_topic: str
    to_topic: str
    relation_type: str           # "implements", "measures", "extends"


@dataclass
class IndexMeta:
    total_sources: int = 0
    avg_coverage: float = 0.0
    iterations: int = 0


@dataclass
class StoreIndex:
    topics: list[Topic] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    meta: IndexMeta = field(default_factory=IndexMeta)


# ── Knowledge Store ───────────────────────────────────────────────────────


class KnowledgeStore:
    """Progressive disclosure knowledge store backed by the filesystem.

    Directory layout:
        research_store/
        ├── index.json
        ├── summaries/{topic_id}.md
        ├── details/sources/src_{NNN}.json
        ├── details/raw_results/iter_{NNN}.json
        └── findings.md
    """

    def __init__(self, store_dir: str = "research_store") -> None:
        """Initialize store at given path. Create directory structure if needed."""
        self.root = Path(store_dir)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create all required subdirectories."""
        for sub in ["summaries", "details/sources", "details/raw_results"]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ── Index operations ──────────────────────────────────────────────────

    def load_index(self) -> StoreIndex:
        """Load index.json and deserialize into StoreIndex.

        Returns an empty StoreIndex if the file doesn't exist.
        """
        index_path = self.root / "index.json"
        if not index_path.exists():
            return StoreIndex()

        raw = json.loads(index_path.read_text(encoding="utf-8"))

        topics = [Topic(**t) for t in raw.get("topics", [])]
        questions = [Question(**q) for q in raw.get("questions", [])]
        relations = [Relation(**r) for r in raw.get("relations", [])]
        meta = IndexMeta(**raw.get("meta", {}))

        return StoreIndex(topics=topics, questions=questions, relations=relations, meta=meta)

    def save_index(self, index: StoreIndex) -> None:
        """Serialize StoreIndex to index.json."""
        index_path = self.root / "index.json"
        index_path.write_text(
            json.dumps(asdict(index), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Topic operations ──────────────────────────────────────────────────

    def add_topic(self, topic: Topic) -> None:
        """Add a topic to the index (skip if ID already exists) and create an empty summary."""
        index = self.load_index()

        if any(t.id == topic.id for t in index.topics):
            return

        index.topics.append(topic)
        self.save_index(index)

        # Create empty summary file
        summary_path = self._summary_path(topic.id)
        if not summary_path.exists():
            summary_path.write_text("", encoding="utf-8")

    def update_topic_coverage(self, topic_id: str, coverage: float) -> None:
        """Update a topic's coverage score in the index."""
        index = self.load_index()
        for topic in index.topics:
            if topic.id == topic_id:
                topic.coverage = coverage
                break
        self.save_index(index)

    def get_topic(self, topic_id: str) -> Optional[Topic]:
        """Find and return a topic by ID, or None."""
        index = self.load_index()
        for topic in index.topics:
            if topic.id == topic_id:
                return topic
        return None

    # ── Summary operations ────────────────────────────────────────────────

    def _summary_path(self, topic_id: str) -> Path:
        """Return the filesystem path for a topic summary."""
        return self.root / "summaries" / f"{topic_id}.md"

    def read_summary(self, topic_id: str) -> str:
        """Read summary file, strip frontmatter, return just the content.

        Returns empty string if file doesn't exist.
        """
        path = self._summary_path(topic_id)
        if not path.exists():
            return ""

        text = path.read_text(encoding="utf-8")
        _, content = self._parse_frontmatter(text)
        return content

    def write_summary(self, topic_id: str, content: str, last_source_id: int = 0) -> None:
        """Write summary with YAML-like frontmatter."""
        index = self.load_index()
        source_count = 0
        for topic in index.topics:
            if topic.id == topic_id:
                source_count = topic.source_count
                break

        now = datetime.now(timezone.utc).isoformat()
        frontmatter = (
            f"---\n"
            f"topic_id: {topic_id}\n"
            f"last_synthesized_source_id: {last_source_id}\n"
            f"source_count: {source_count}\n"
            f"updated: \"{now}\"\n"
            f"---\n"
        )

        path = self._summary_path(topic_id)
        path.write_text(f"{frontmatter}\n{content}\n", encoding="utf-8")

    def read_summary_frontmatter(self, topic_id: str) -> dict:
        """Parse only the frontmatter section from a topic summary.

        Returns empty dict if no frontmatter or file missing.
        """
        path = self._summary_path(topic_id)
        if not path.exists():
            return {}

        text = path.read_text(encoding="utf-8")
        fm, _ = self._parse_frontmatter(text)
        return fm

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        """Split text into (frontmatter_dict, content).

        Frontmatter is between the first two ``---`` lines.
        Parses ``key: value`` pairs from the frontmatter section.
        """
        if not text or not text.startswith("---"):
            return {}, text

        # Find the closing ---
        end_match = re.search(r"\n---\s*\n", text[3:])
        if end_match is None:
            return {}, text

        fm_text = text[4:3 + end_match.start()]  # between first --- and closing ---
        content = text[3 + end_match.end():]

        result: dict[str, Any] = {}
        for line in fm_text.strip().splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"')
            # Try numeric conversion
            try:
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                result[key] = value

        return result, content

    # ── Source operations ──────────────────────────────────────────────────

    def _source_path(self, source_id: int) -> Path:
        """Return the filesystem path for a source record."""
        return self.root / "details" / "sources" / f"src_{source_id:03d}.json"

    def add_source(self, source_id: int, data: dict) -> None:
        """Write source as JSON and update relevant topic entries in the index."""
        path = self._source_path(source_id)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Update index: increment unsynthesized_sources and set stale for matching topics
        topic_ids = data.get("topic_ids", [])
        if topic_ids:
            index = self.load_index()
            for topic in index.topics:
                if topic.id in topic_ids:
                    topic.unsynthesized_sources += 1
                    topic.source_count += 1
                    topic.stale = True
            index.meta.total_sources += 1
            self.save_index(index)

    def get_source(self, source_id: int) -> Optional[dict]:
        """Read and return source JSON, or None if it doesn't exist."""
        path = self._source_path(source_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def get_unsynthesized_sources(self, topic_id: str) -> list[dict]:
        """Find all sources for a topic that haven't been synthesized yet.

        A source is unsynthesized if its topic_ids include this topic_id AND
        its source_id > the topic summary's last_synthesized_source_id.
        """
        fm = self.read_summary_frontmatter(topic_id)
        last_id = int(fm.get("last_synthesized_source_id", 0))

        sources_dir = self.root / "details" / "sources"
        results: list[dict] = []

        for path in sorted(sources_dir.glob("src_*.json")):
            # Extract NNN from src_NNN.json
            match = re.search(r"src_(\d+)\.json$", path.name)
            if match is None:
                continue
            sid = int(match.group(1))
            if sid <= last_id:
                continue

            data = json.loads(path.read_text(encoding="utf-8"))
            if topic_id in data.get("topic_ids", []):
                results.append(data)

        return results

    def next_source_id(self) -> int:
        """Return the next available source ID (max existing + 1, or 1)."""
        sources_dir = self.root / "details" / "sources"
        max_id = 0
        for path in sources_dir.glob("src_*.json"):
            match = re.search(r"src_(\d+)\.json$", path.name)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return max_id + 1

    # ── Raw results ───────────────────────────────────────────────────────

    def save_raw_results(self, iteration: int, results: list) -> None:
        """Save results list as JSON to details/raw_results/iter_{NNN}.json."""
        path = self.root / "details" / "raw_results" / f"iter_{iteration:03d}.json"
        path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # ── Findings generation ───────────────────────────────────────────────

    def regenerate_findings(self) -> str:
        """Build findings.md by concatenating all topic summaries.

        Writes the result to store_dir/findings.md and returns the content.
        """
        index = self.load_index()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        sections: list[str] = []
        for topic in index.topics:
            summary = self.read_summary(topic.id)
            if summary.strip():
                sections.append(summary.strip())

        body = "\n\n---\n\n".join(sections) if sections else "(No findings yet.)"

        content = (
            f"# Research Findings\n"
            f"> Auto-generated from knowledge store | {now}\n\n"
            f"{body}\n"
        )

        findings_path = self.root / "findings.md"
        findings_path.write_text(content, encoding="utf-8")
        return content

    # ── Bootstrap ─────────────────────────────────────────────────────────

    def bootstrap_from_findings(
        self,
        findings_text: str,
        research_program: str,
        llm_fn: Callable[[str], dict],
    ) -> None:
        """Use LLM to extract structure from existing findings.md.

        1. Calls llm_fn with a prompt to extract topics and questions.
        2. Creates topics in the index and writes each topic's summary.
        3. Saves the index.
        """
        prompt = f"""You are extracting structured data from research findings.

## Research Program
{research_program}

## Current Findings
{findings_text}

## Task
Extract the topics and open questions from the findings above.
Return ONLY valid JSON with this exact structure:

{{
  "topics": [
    {{
      "id": "short_snake_case_slug",
      "title": "Human Readable Title",
      "keywords": ["keyword1", "keyword2"],
      "summary": "2-4 sentence summary of findings for this topic"
    }}
  ],
  "questions": [
    {{
      "id": "q1",
      "text": "What is the specific question?",
      "status": "unanswered",
      "related_topics": ["topic_slug"]
    }}
  ]
}}

Rules:
- Topic IDs must be short snake_case slugs (e.g. "architecture_patterns")
- Extract 3-8 topics covering the major themes
- Status should be "unanswered", "partial", or "answered"
- Each question should reference at least one related topic"""

        result = llm_fn(prompt)

        index = self.load_index()

        for t_data in result.get("topics", []):
            topic = Topic(
                id=t_data["id"],
                title=t_data.get("title", t_data["id"]),
                keywords=t_data.get("keywords", []),
            )
            if not any(t.id == topic.id for t in index.topics):
                index.topics.append(topic)

            summary_text = t_data.get("summary", "")
            if summary_text:
                self.write_summary(topic.id, summary_text)

        for q_data in result.get("questions", []):
            question = Question(
                id=q_data["id"],
                text=q_data.get("text", ""),
                status=q_data.get("status", "unanswered"),
                related_topics=q_data.get("related_topics", []),
            )
            if not any(q.id == question.id for q in index.questions):
                index.questions.append(question)

        self.save_index(index)

    # ── Smart Routing ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(texts: list[str]) -> set[str]:
        """Extract keywords from a list of texts.

        Simple approach: lowercase, split on non-alphanumeric, remove stopwords,
        remove short words (< 3 chars).
        """
        keywords: set[str] = set()
        for text in texts:
            tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
            for token in tokens:
                if len(token) >= 3 and token not in _STOPWORDS:
                    keywords.add(token)
        return keywords

    def select_relevant_summaries(
        self,
        gap_queries: list[str],
        token_budget: int = 8000,
        budget_ratio: float = 0.6,
    ) -> list[tuple[str, str]]:
        """Select which topic summaries to load based on relevance scoring.

        Uses keyword overlap x (1 - coverage_score) to rank topics.
        Loads summaries in relevance order until the budget is ~60% full.
        No LLM call needed — pure keyword matching.

        Args:
            gap_queries: List of search queries or gap descriptions to match against
            token_budget: Total token budget for context (default 8000)
            budget_ratio: Fraction of budget to fill with summaries (default 0.6)

        Returns:
            List of (topic_id, summary_content) tuples, most relevant first
        """
        index = self.load_index()
        if not index.topics:
            return []

        query_keywords = self._extract_keywords(gap_queries)
        if not query_keywords:
            return []

        # Score each topic
        scored: list[tuple[float, str]] = []
        for topic in index.topics:
            topic_keywords = set(kw.lower() for kw in topic.keywords)
            keyword_overlap = len(query_keywords & topic_keywords)
            score = keyword_overlap * (1.0 - topic.coverage)
            if topic.stale:
                score += 0.5
            if score > 0:
                scored.append((score, topic.id))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Load summaries until budget is reached
        max_tokens = int(token_budget * budget_ratio)
        used_tokens = 0
        results: list[tuple[str, str]] = []

        for _score, topic_id in scored:
            summary = self.read_summary(topic_id)
            if not summary.strip():
                continue
            summary_tokens = self.estimate_tokens(summary)
            if used_tokens + summary_tokens > max_tokens and results:
                break
            results.append((topic_id, summary))
            used_tokens += summary_tokens

        return results

    def prioritize_stale_topics(self) -> list[str]:
        """Return topic IDs that should be synthesized before new searches.

        A topic with unsynthesized_sources > 0 should be synthesized first
        to avoid redundant searching. Returns topic IDs sorted by
        unsynthesized_sources count descending.
        """
        index = self.load_index()
        stale = [
            (t.unsynthesized_sources, t.id)
            for t in index.topics
            if t.unsynthesized_sources > 0
        ]
        stale.sort(key=lambda x: x[0], reverse=True)
        return [tid for _, tid in stale]

    def get_context_with_budget(
        self,
        gap_queries: list[str],
        token_budget: int = 8000,
    ) -> str:
        """Build context string that fits within a token budget.

        Always includes:
        1. Index context (~500 tokens)
        2. Relevant summaries based on gap_queries (up to 60% of remaining budget)

        Returns a formatted string ready to inject into an LLM prompt.
        """
        # Always include index context
        index_context = self.get_context_for_gaps()
        index_tokens = self.estimate_tokens(index_context)

        remaining_budget = token_budget - index_tokens
        if remaining_budget < 0:
            remaining_budget = 0

        # Select relevant summaries with the remaining budget
        relevant = self.select_relevant_summaries(
            gap_queries,
            token_budget=remaining_budget,
            budget_ratio=0.6,
        )

        parts: list[str] = [index_context]

        if relevant:
            summary_lines: list[str] = []
            for topic_id, content in relevant:
                topic = self.get_topic(topic_id)
                title = topic.title if topic else topic_id
                summary_lines.append(f"### {title} ({topic_id})\n{content}")
            parts.append("\n## Relevant Topic Summaries\n" + "\n\n".join(summary_lines))

        return "\n".join(parts)

    # ── Utility ───────────────────────────────────────────────────────────

    def exists(self) -> bool:
        """Return True if the store directory and index.json both exist."""
        return self.root.exists() and (self.root / "index.json").exists()

    def get_context_for_gaps(self) -> str:
        """Format the index as a compact, readable string for gap analysis.

        This is the 'always loaded' layer — kept small for LLM context windows.
        """
        index = self.load_index()

        lines: list[str] = []
        lines.append("## Knowledge Store Index")
        lines.append(f"Sources: {index.meta.total_sources} | "
                      f"Avg coverage: {index.meta.avg_coverage:.2f} | "
                      f"Iterations: {index.meta.iterations}")
        lines.append("")

        if index.topics:
            lines.append("### Topics")
            for t in index.topics:
                stale_marker = " [STALE]" if t.stale else ""
                unsynth = f" ({t.unsynthesized_sources} unsynthesized)" if t.unsynthesized_sources else ""
                lines.append(f"- **{t.title}** ({t.id}): coverage={t.coverage:.2f}, "
                             f"sources={t.source_count}{unsynth}{stale_marker}")
            lines.append("")

        if index.questions:
            lines.append("### Questions")
            for q in index.questions:
                topics = ", ".join(q.related_topics) if q.related_topics else "none"
                lines.append(f"- [{q.status}] {q.text} (topics: {topics})")
            lines.append("")

        if index.relations:
            lines.append("### Relations")
            for r in index.relations:
                lines.append(f"- {r.from_topic} --{r.relation_type}--> {r.to_topic}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count: len(text) // 4."""
        return len(text) // 4
