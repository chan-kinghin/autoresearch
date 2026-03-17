"""
knowledge_store.py — Progressive disclosure knowledge store.

3-layer architecture:
  Layer 1 (Index):    Always loaded. ~500 tokens. Topics, questions, relations, meta.
  Layer 2 (Summaries): On-demand. ~1-2k tokens each. Per-topic markdown with frontmatter.
  Layer 3 (Details):  Drill-down only. Source records, raw search results.
"""

from __future__ import annotations

import json
import os
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
        Includes confidence levels (D2), question matrix (D3), source recency (D4),
        contradiction log (D6), and citation validation (D7).
        """
        index = self.load_index()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        parts: list[str] = []

        # Header
        parts.append(f"# Research Findings\n> Auto-generated from knowledge store | {now}\n")

        # D2: Topic sections with confidence levels
        topic_sections: list[str] = []
        for topic in index.topics:
            summary = self.read_summary(topic.id)
            if summary.strip():
                coverage_pct = int(topic.coverage * 100)
                # D2: Confidence level based on sources and coverage
                if topic.source_count >= 5 and topic.coverage >= 0.7:
                    confidence = "HIGH"
                elif topic.source_count >= 2 and topic.coverage >= 0.4:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
                section = (
                    f"## {topic.title}\n\n"
                    f"Coverage: {coverage_pct}% | Confidence: {confidence} | Sources: {topic.source_count}\n\n"
                    f"{summary.strip()}"
                )
                topic_sections.append(section)

        if topic_sections:
            parts.append("\n\n---\n\n".join(topic_sections))
        else:
            parts.append("(No findings yet.)")

        # D3: Question resolution matrix
        question_matrix = self.generate_question_matrix()
        if question_matrix:
            parts.append(f"\n---\n\n{question_matrix}")

        # Open questions (existing)
        open_qs = [q for q in index.questions if q.status in ("unanswered", "partial")]
        if open_qs:
            parts.append("\n---\n\n## Open Questions\n")
            for q in open_qs:
                parts.append(f"- [{q.status}] {q.text}")
            parts.append("")

        # D6: Contradiction log
        contradictions = self.get_contradictions_from_summaries()
        if contradictions:
            parts.append("\n---\n\n## Known Conflicts & Uncertainties\n")
            for c in contradictions:
                parts.append(f"- {c}")
            parts.append("")

        # D4: Source recency analysis
        recency = self.analyze_source_recency()
        if recency:
            parts.append(f"\n{recency}")

        # Sources summary (existing)
        if index.meta.total_sources > 0:
            parts.append(f"\n---\n\n## Sources\n\nTotal sources: {index.meta.total_sources}\n")

        # D7: Citation validation
        validation = self.validate_citations()
        if validation["total_citations"] > 0:
            parts.append(f"Citation validation: {validation['matched']}/{validation['total_citations']} "
                         f"citations matched to sources ({validation['match_rate']:.0%})")
            if validation["unmatched"]:
                parts.append(f"Unmatched citations: {', '.join(validation['unmatched'][:5])}")
            parts.append("")

        # Metadata footer
        parts.append(f"\n---\n\n_Coverage: {index.meta.avg_coverage:.0%} | "
                      f"Iterations: {index.meta.iterations} | "
                      f"Generated: {now}_\n")

        content = "\n".join(parts)

        findings_path = self.root / "findings.md"
        findings_path.write_text(content, encoding="utf-8")
        return content

    # ── D1: Citation Index Builder ─────────────────────────────────────────

    def build_citation_index(self) -> Dict[str, Dict[str, Any]]:
        """Build a mapping from inline citations to source records.

        Scans all topic summaries for citation patterns like [Author, Year] or [Source Name]
        and maps them to source records in the store.

        Returns: {citation_text: {source_id, url, title, year, source_type}}
        """
        index = self.load_index()
        citation_map: Dict[str, Dict[str, Any]] = {}

        # Load all sources
        sources_dir = self.root / "details" / "sources"
        all_sources: list[dict] = []
        for path in sorted(sources_dir.glob("src_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data["_source_id"] = path.stem
                all_sources.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        # Scan summaries for citations
        for topic in index.topics:
            summary = self.read_summary(topic.id)
            # Match [Author, Year], [Author et al., Year], [Source Name]
            citations = re.findall(r'\[([^\]]+)\]', summary)
            for cite in citations:
                if cite in citation_map:
                    continue
                # Try to match against sources
                cite_lower = cite.lower()
                for src in all_sources:
                    title = src.get("title", "").lower()
                    authors = src.get("authors", "").lower()
                    year = str(src.get("year", ""))
                    if (year and year in cite and (
                        any(name.strip().split()[-1].lower() in cite_lower
                            for name in authors.split(",") if name.strip())
                        or title[:20].lower() in cite_lower
                    )):
                        citation_map[cite] = {
                            "source_id": src["_source_id"],
                            "url": src.get("url", ""),
                            "title": src.get("title", ""),
                            "year": year,
                            "source_type": src.get("source", "unknown"),
                        }
                        break

        return citation_map

    # ── D3: Question Resolution Matrix ─────────────────────────────────────

    def generate_question_matrix(self) -> str:
        """Generate a question resolution matrix showing coverage status per research question."""
        index = self.load_index()
        if not index.questions:
            return ""

        lines: list[str] = []
        lines.append("## Research Questions: Coverage Status\n")
        lines.append("| # | Question | Status | Related Topics | Gap |")
        lines.append("|---|----------|--------|----------------|-----|")

        for i, q in enumerate(index.questions, 1):
            status_icon = {"answered": "\u2705", "partial": "\U0001f536", "unanswered": "\u274c"}.get(q.status, "\u2753")
            topics_str = ", ".join(q.related_topics) if q.related_topics else "\u2014"
            gap = "None" if q.status == "answered" else "Needs more research"
            lines.append(f"| {i} | {q.text} | {status_icon} {q.status.title()} | {topics_str} | {gap} |")

        lines.append("")

        # Summary counts
        answered = sum(1 for q in index.questions if q.status == "answered")
        partial = sum(1 for q in index.questions if q.status == "partial")
        unanswered = sum(1 for q in index.questions if q.status == "unanswered")
        total = len(index.questions)
        lines.append(f"**Summary**: {answered}/{total} answered, {partial}/{total} partial, {unanswered}/{total} open\n")

        return "\n".join(lines)

    # ── D4: Temporal Awareness ─────────────────────────────────────────────

    def analyze_source_recency(self) -> str:
        """Analyze the age distribution of sources and return a summary."""
        sources_dir = self.root / "details" / "sources"
        years: list[int] = []
        current_year = datetime.now().year

        for path in sources_dir.glob("src_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                year_str = str(data.get("year", ""))
                if year_str and year_str.isdigit():
                    years.append(int(year_str))
            except (json.JSONDecodeError, OSError, ValueError):
                continue

        if not years:
            return ""

        years.sort()
        newest = max(years)
        oldest = min(years)
        median_idx = len(years) // 2
        median = years[median_idx]
        old_count = sum(1 for y in years if current_year - y > 3)
        old_pct = int(old_count / len(years) * 100)

        lines: list[str] = []
        lines.append("## Source Recency Analysis\n")
        lines.append(f"- **Newest source**: {newest}")
        lines.append(f"- **Oldest source**: {oldest}")
        lines.append(f"- **Median source year**: {median}")
        lines.append(f"- **Total sources with year data**: {len(years)}")
        if old_pct > 30:
            lines.append(f"- \u26a0\ufe0f {old_pct}% of sources are >3 years old. Consider searching for recent developments.")
        lines.append("")

        return "\n".join(lines)

    # ── D5: Multi-format Export ────────────────────────────────────────────

    def export_executive_summary(self) -> str:
        """Generate a 1-page executive summary of research findings."""
        index = self.load_index()
        model = os.environ.get("AUTORESEARCH_MODEL", "claude-sonnet-4-20250514")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        answered = sum(1 for q in index.questions if q.status == "answered")
        partial = sum(1 for q in index.questions if q.status == "partial")
        total_q = len(index.questions) or 1

        lines: list[str] = []
        lines.append("# Executive Summary\n")
        lines.append(f"_Generated: {now} | Model: {model}_\n")
        lines.append(f"## Coverage")
        lines.append(f"- Overall: {index.meta.avg_coverage:.0%}")
        lines.append(f"- Questions: {answered}/{total_q} answered, {partial}/{total_q} partial")
        lines.append(f"- Sources consulted: {index.meta.total_sources}")
        lines.append(f"- Iterations: {index.meta.iterations}\n")

        # Top findings per topic
        lines.append("## Key Findings\n")
        for topic in sorted(index.topics, key=lambda t: t.coverage, reverse=True)[:5]:
            summary = self.read_summary(topic.id)
            # Extract first meaningful sentence
            first_line = ""
            for line in summary.split("\n"):
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and len(stripped) > 20:
                    first_line = stripped[:200]
                    break
            if first_line:
                confidence = "HIGH" if topic.source_count >= 5 and topic.coverage >= 0.7 else \
                             "MEDIUM" if topic.source_count >= 2 else "LOW"
                lines.append(f"- **{topic.title}** [{confidence}]: {first_line}")

        # Open gaps
        open_q = [q for q in index.questions if q.status in ("unanswered", "partial")]
        if open_q:
            lines.append(f"\n## Remaining Gaps\n")
            for q in open_q[:5]:
                lines.append(f"- {q.text}")

        lines.append("")
        return "\n".join(lines)

    def export_bibtex(self) -> str:
        """Export source citations in BibTeX format."""
        sources_dir = self.root / "details" / "sources"
        entries: list[str] = []

        for path in sorted(sources_dir.glob("src_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            title = data.get("title", "")
            authors = data.get("authors", "")
            year = str(data.get("year", ""))
            url = data.get("url", "")
            source = data.get("source", "misc")

            if not title:
                continue

            # Generate a cite key
            first_author = authors.split(",")[0].strip().split()[-1] if authors else "unknown"
            cite_key = f"{first_author.lower()}{year}" if year else first_author.lower()
            cite_key = re.sub(r'[^a-z0-9]', '', cite_key)

            entry_type = "article" if source in ("semantic_scholar", "arxiv") else "misc"
            entry = f"@{entry_type}{{{cite_key},\n"
            entry += f"  title = {{{title}}},\n"
            if authors:
                entry += f"  author = {{{authors}}},\n"
            if year:
                entry += f"  year = {{{year}}},\n"
            if url:
                entry += f"  url = {{{url}}},\n"
            entry += f"  note = {{Retrieved via {source}}}\n"
            entry += "}"
            entries.append(entry)

        return "\n\n".join(entries)

    # ── D6: Contradiction Log ──────────────────────────────────────────────

    def get_contradictions_from_summaries(self) -> list[str]:
        """Scan summaries for contradiction markers added by synthesis."""
        index = self.load_index()
        contradictions: list[str] = []

        for topic in index.topics:
            summary = self.read_summary(topic.id)
            # Look for contradiction markers from synthesis
            for line in summary.split("\n"):
                stripped = line.strip()
                if any(marker in stripped.lower() for marker in [
                    "[conflict]", "contradict", "disagree", "competing claim",
                    "however, source", "in contrast,",
                ]):
                    contradictions.append(f"**{topic.title}**: {stripped[:200]}")

        return contradictions

    # ── D7: Citation Validation ────────────────────────────────────────────

    def validate_citations(self) -> Dict[str, Any]:
        """Validate that inline citations in summaries map to actual sources.

        Returns: {
            "total_citations": int,
            "matched": int,
            "unmatched": list[str],
            "match_rate": float,
        }
        """
        citation_index = self.build_citation_index()
        index = self.load_index()

        all_citations: set[str] = set()
        for topic in index.topics:
            summary = self.read_summary(topic.id)
            citations = re.findall(r'\[([^\]]+)\]', summary)
            # Filter out markdown links and status markers
            for cite in citations:
                if cite.startswith("http") or cite in ("CONFLICT", "STALE") or cite.startswith("!"):
                    continue
                if any(c.isalpha() for c in cite):
                    all_citations.add(cite)

        matched = [c for c in all_citations if c in citation_index]
        unmatched = [c for c in all_citations if c not in citation_index]

        total = len(all_citations)
        return {
            "total_citations": total,
            "matched": len(matched),
            "unmatched": unmatched,
            "match_rate": len(matched) / total if total > 0 else 1.0,
        }

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
