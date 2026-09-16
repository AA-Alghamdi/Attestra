"""Literature-grounded research: search, read, extract, implement.

This module enables the RESEARCHER aspect of Attestra:
  1. SEARCH: Query arXiv, Papers with Code, Semantic Scholar for relevant work
  2. READ: Extract key techniques, architectures, hyperparameters from papers
  3. IMPLEMENT: Translate paper techniques into executable proposals
  4. VERIFY: Validate implementations match paper claims

The LLM acts as a research assistant:
  - Given a problem, it searches for relevant literature
  - Extracts the key techniques from top papers
  - Generates implementation proposals grounded in those papers
  - Cites sources for reproducibility
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import quote_plus


@dataclass
class Paper:
    """A research paper reference."""
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    arxiv_id: str = ""
    year: int = 0
    venue: str = ""
    url: str = ""
    citations: int = 0
    # Extracted info
    key_techniques: List[str] = field(default_factory=list)
    datasets_used: List[str] = field(default_factory=list)
    reported_results: Dict[str, float] = field(default_factory=dict)


@dataclass
class TechniqueExtraction:
    """A technique extracted from literature."""
    name: str
    description: str
    paper: Paper
    implementation_hint: str = ""          # pseudo-code or key insight
    applicable_to: List[str] = field(default_factory=list)  # task types
    estimated_improvement: str = ""       # e.g. "+2-5% on tabular"
    requirements: List[str] = field(default_factory=list)    # e.g. ["torch", "large data"]


@dataclass
class ResearchContext:
    """Full literature context for a problem."""
    query: str
    papers: List[Paper] = field(default_factory=list)
    techniques: List[TechniqueExtraction] = field(default_factory=list)
    landscape_summary: str = ""           # LLM-generated summary of the field
    recommended_approach: str = ""
    sota_baseline: str = ""               # what's the current SOTA for this problem type


class LiteratureSearch:
    """Search and analyze ML literature.

    Usage:
        search = LiteratureSearch(llm_call=my_llm)
        context = search.research("tabular classification with class imbalance")
        # context.papers -> relevant papers
        # context.techniques -> extractable techniques
        # context.landscape_summary -> field overview
    """

    def __init__(self, llm_call: Optional[Callable] = None,
                 cache_dir: Optional[str] = None):
        self.llm_call = llm_call
        self._cache: Dict[str, List[Paper]] = {}

    def research(self, query: str, task_type: str = "",
                 max_papers: int = 10) -> ResearchContext:
        """Full research pipeline: search -> read -> extract -> summarize."""
        # Search for papers from multiple sources
        papers = self.search_arxiv(query, max_results=max_papers)
        pwc_papers = self.search_papers_with_code(query, max_results=5)

        # Merge, deduplicate by title
        seen_titles = {p.title.lower().strip() for p in papers}
        for p in pwc_papers:
            if p.title.lower().strip() not in seen_titles:
                papers.append(p)
                seen_titles.add(p.title.lower().strip())

        # Search HuggingFace Hub for relevant models
        hf_models = self.search_huggingface(query, task_type=task_type, max_results=5)

        # Extract techniques (with LLM if available)
        techniques = []
        if self.llm_call and papers:
            techniques = self._extract_techniques(papers, query, task_type)

        # Generate landscape summary (include HF model context)
        landscape = ""
        recommended = ""
        if self.llm_call:
            hf_context = ""
            if hf_models:
                hf_context = "\nRelevant HuggingFace models:\n" + "\n".join(
                    f"  - {m['id']} (downloads={m.get('downloads', '?')}, "
                    f"tags={', '.join(m.get('tags', [])[:5])})"
                    for m in hf_models[:5]
                )
            landscape, recommended = self._summarize_landscape(
                papers, query, task_type, extra_context=hf_context,
            )

        return ResearchContext(
            query=query,
            papers=papers,
            techniques=techniques,
            landscape_summary=landscape,
            recommended_approach=recommended,
        )

    def search_huggingface(self, query: str, task_type: str = "",
                           max_results: int = 5) -> List[Dict]:
        """Search HuggingFace Hub for relevant pretrained models."""
        try:
            # Map task types to HF pipeline tags
            task_map = {
                "binary": "text-classification",
                "multiclass": "text-classification",
                "regression": "tabular-regression",
                "image_classification": "image-classification",
                "text_classification": "text-classification",
                "tabular_classification": "tabular-classification",
                "tabular_regression": "tabular-regression",
            }
            hf_task = task_map.get(task_type, "")

            # Search HF API
            encoded = quote_plus(query)
            url = f"https://huggingface.co/api/models?search={encoded}&limit={max_results}"
            if hf_task:
                url += f"&pipeline_tag={hf_task}"
            url += "&sort=downloads&direction=-1"

            req = Request(url, headers={"User-Agent": "Attestra/0.3"})
            response = urlopen(req, timeout=10)
            data = json.loads(response.read().decode("utf-8"))

            models = []
            for item in data[:max_results]:
                models.append({
                    "id": item.get("modelId", item.get("id", "")),
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "tags": item.get("tags", []),
                    "pipeline_tag": item.get("pipeline_tag", ""),
                    "library": item.get("library_name", ""),
                })
            return models

        except (URLError, TimeoutError, Exception):
            return []

    def search_arxiv(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search arXiv for relevant papers."""
        if query in self._cache:
            return self._cache[query]

        try:
            # arXiv API
            encoded = quote_plus(query)
            url = (f"http://export.arxiv.org/api/query?"
                   f"search_query=all:{encoded}&start=0&max_results={max_results}"
                   f"&sortBy=relevance&sortOrder=descending")

            req = Request(url, headers={"User-Agent": "Attestra/0.3"})
            response = urlopen(req, timeout=10)
            xml_data = response.read().decode("utf-8")

            papers = self._parse_arxiv_xml(xml_data)
            self._cache[query] = papers
            return papers

        except (URLError, TimeoutError, Exception):
            return []

    def search_papers_with_code(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search Papers with Code for implementations."""
        try:
            encoded = quote_plus(query)
            url = f"https://paperswithcode.com/api/v1/papers/?q={encoded}&items_per_page={max_results}"
            req = Request(url, headers={"User-Agent": "Attestra/0.3"})
            response = urlopen(req, timeout=10)
            data = json.loads(response.read().decode("utf-8"))

            papers = []
            for item in data.get("results", [])[:max_results]:
                papers.append(Paper(
                    title=item.get("title", ""),
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    abstract=item.get("abstract", ""),
                    arxiv_id=item.get("arxiv_id", ""),
                    url=item.get("url_abs", ""),
                ))
            return papers

        except (URLError, TimeoutError, Exception):
            return []

    def _parse_arxiv_xml(self, xml: str) -> List[Paper]:
        """Parse arXiv Atom XML response."""
        papers = []
        # Simple regex parsing (avoid xml.etree for robustness)
        entries = re.findall(r'<entry>(.*?)</entry>', xml, re.DOTALL)
        for entry in entries:
            title = self._extract_tag(entry, "title").strip().replace("\n", " ")
            abstract = self._extract_tag(entry, "summary").strip().replace("\n", " ")
            arxiv_id_match = re.search(r'<id>http://arxiv.org/abs/(.*?)</id>', entry)
            arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else ""

            # Authors
            authors = re.findall(r'<name>(.*?)</name>', entry)

            # Year from published date
            published = self._extract_tag(entry, "published")
            year = int(published[:4]) if published and len(published) >= 4 else 0

            papers.append(Paper(
                title=title,
                authors=authors[:5],
                abstract=abstract[:500],
                arxiv_id=arxiv_id,
                year=year,
                url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            ))
        return papers

    def _extract_tag(self, text: str, tag: str) -> str:
        match = re.search(f'<{tag}[^>]*>(.*?)</{tag}>', text, re.DOTALL)
        return match.group(1) if match else ""

    def _extract_techniques(self, papers: List[Paper], query: str,
                            task_type: str) -> List[TechniqueExtraction]:
        """Use LLM to extract implementable techniques from papers."""
        if not self.llm_call:
            return []

        # Format papers for LLM
        papers_text = ""
        for i, p in enumerate(papers[:5], 1):
            papers_text += f"\n{i}. {p.title} ({p.year})\n   {p.abstract[:200]}...\n"

        system = """You are an ML research engineer. Given papers, extract the KEY TECHNIQUES
that could be implemented. For each technique, provide:
1. A short name
2. Description (2 sentences max)
3. Implementation hint (pseudo-code or key insight)
4. Estimated improvement

Respond in JSON:
{"techniques": [
  {"name": "...", "description": "...", "implementation_hint": "...",
   "applicable_to": ["classification"], "estimated_improvement": "+2-5%"}
]}"""

        user_msg = f"Query: {query}\nTask: {task_type}\n\nPapers:{papers_text}"

        try:
            raw, _ = self.llm_call(system, user_msg)
            parsed = self._parse_json(raw)
            techniques = []
            for t in parsed.get("techniques", []):
                techniques.append(TechniqueExtraction(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    paper=papers[0] if papers else Paper(title=""),
                    implementation_hint=t.get("implementation_hint", ""),
                    applicable_to=t.get("applicable_to", []),
                    estimated_improvement=t.get("estimated_improvement", ""),
                ))
            return techniques
        except Exception:
            return []

    def _summarize_landscape(self, papers: List[Paper], query: str,
                             task_type: str, extra_context: str = "") -> Tuple[str, str]:
        """LLM-generated field summary and recommendation."""
        if not self.llm_call:
            return "", ""

        papers_text = "\n".join(f"- {p.title} ({p.year})" for p in papers[:5])

        system = """Given relevant papers and models for an ML problem, provide:
1. A 3-sentence landscape summary (what's the state of the art, key trends)
2. A recommended approach for a new practitioner starting this problem

Respond in JSON: {"landscape": "...", "recommended_approach": "..."}"""

        user_msg = f"Problem: {query}\nTask: {task_type}\nPapers:\n{papers_text}"
        if extra_context:
            user_msg += f"\n{extra_context}"

        try:
            raw, _ = self.llm_call(system, user_msg)
            parsed = self._parse_json(raw)
            return parsed.get("landscape", ""), parsed.get("recommended_approach", "")
        except Exception:
            return "", ""

    def _parse_json(self, raw: str) -> Dict:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}


class ResearchProposer:
    """Generate research proposals grounded in literature.

    Uses literature search + LLM to produce implementation-ready proposals
    for the research engine to execute.
    """

    def __init__(self, llm_call: Callable, literature: Optional[LiteratureSearch] = None):
        self.llm_call = llm_call
        self.literature = literature or LiteratureSearch(llm_call=llm_call)

    def propose(self, goal: str, task_type: str, data_summary: str,
                current_best: float, tried: List[str],
                max_proposals: int = 3) -> List[Dict]:
        """Generate literature-grounded proposals.

        Returns:
            List of proposal dicts with keys: name, code, rationale, source_paper
        """
        # Search literature
        context = self.literature.research(f"{task_type} {goal}", task_type=task_type)

        # Generate proposals grounded in literature
        techniques_text = ""
        for t in context.techniques[:5]:
            techniques_text += f"\n- {t.name}: {t.description}\n  Hint: {t.implementation_hint}\n"

        system = f"""You are an ML engineer implementing techniques from recent papers.
Given the problem and relevant techniques, write {max_proposals} COMPLETE Python functions.

Each function must:
1. Be named `build_estimator(seed)` 
2. Return a fitted sklearn-compatible estimator (with .fit() and .predict() methods)
3. Be grounded in one of the techniques listed
4. Be different from what's already been tried

Output JSON:
{{"proposals": [
  {{"name": "...", "rationale": "Based on [paper technique]...", 
    "code": "def build_estimator(seed):\\n    ..."}}
]}}"""

        tried_str = ", ".join(tried[-10:]) if tried else "nothing"
        user_msg = f"""Problem: {goal}
Task: {task_type}
Data: {data_summary}
Current best score: {current_best}
Already tried: {tried_str}
Landscape: {context.landscape_summary}
Techniques from literature:
{techniques_text}"""

        try:
            raw, _ = self.llm_call(system, user_msg)
            parsed = self._parse_json(raw)
            proposals = []
            for p in parsed.get("proposals", [])[:max_proposals]:
                proposals.append({
                    "name": p.get("name", "literature_proposal"),
                    "code": p.get("code", ""),
                    "rationale": p.get("rationale", ""),
                    "source": "literature",
                    "landscape": context.landscape_summary,
                })
            return proposals
        except Exception:
            return []

    def _parse_json(self, raw: str) -> Dict:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}
