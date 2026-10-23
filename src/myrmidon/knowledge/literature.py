"""Literature search: arXiv + Crossref, returning citable records."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, Field

from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext

_ARXIV = "https://export.arxiv.org/api/query"
_CROSSREF = "https://api.crossref.org/works"
_ATOM = "{http://www.w3.org/2005/Atom}"


class Paper(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    abstract_snippet: str | None = None


class LiteratureObservation(Observation):
    papers: list[Paper] = Field(default_factory=list)


class LitSearchParams(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)
    source: str = Field(
        default="both", pattern="^(arxiv|crossref|both)$",
    )


def _search_arxiv(query: str, limit: int) -> list[Paper]:
    resp = httpx.get(
        _ARXIV,
        params={"search_query": f"all:{query}", "max_results": limit},
        timeout=30,
        follow_redirects=True,
    )
    resp.raise_for_status()
    papers = []
    root = ET.fromstring(resp.text)
    for entry in root.findall(f"{_ATOM}entry"):
        title = (entry.findtext(f"{_ATOM}title") or "").strip()
        if not title:
            continue
        arxiv_id = (entry.findtext(f"{_ATOM}id") or "").rsplit("/", 1)[-1]
        published = entry.findtext(f"{_ATOM}published") or ""
        summary = re.sub(r"\s+", " ", entry.findtext(f"{_ATOM}summary") or "").strip()
        papers.append(
            Paper(
                title=re.sub(r"\s+", " ", title),
                authors=[
                    a.findtext(f"{_ATOM}name") or ""
                    for a in entry.findall(f"{_ATOM}author")
                ][:8],
                year=int(published[:4]) if published[:4].isdigit() else None,
                arxiv_id=arxiv_id,
                venue="arXiv",
                abstract_snippet=summary[:300] or None,
            )
        )
    return papers


def _search_crossref(query: str, limit: int) -> list[Paper]:
    resp = httpx.get(
        _CROSSREF, params={"query": query, "rows": limit}, timeout=30
    )
    resp.raise_for_status()
    papers = []
    for item in resp.json().get("message", {}).get("items", []):
        title_list = item.get("title") or []
        if not title_list:
            continue
        year = None
        issued = item.get("issued", {}).get("date-parts", [[None]])
        if issued and issued[0] and issued[0][0]:
            year = issued[0][0]
        papers.append(
            Paper(
                title=title_list[0],
                authors=[
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in item.get("author", [])
                ][:8],
                year=year,
                doi=item.get("DOI"),
                venue=(item.get("container-title") or [None])[0],
            )
        )
    return papers


def search_literature(ctx: ToolContext, params: LitSearchParams) -> LiteratureObservation:
    papers: list[Paper] = []
    errors: list[str] = []
    if params.source in ("arxiv", "both"):
        try:
            papers += _search_arxiv(params.query, params.max_results)
        except (httpx.HTTPError, ET.ParseError) as exc:
            errors.append(f"arxiv: {exc}")
    if params.source in ("crossref", "both"):
        try:
            papers += _search_crossref(params.query, params.max_results)
        except (httpx.HTTPError, ValueError) as exc:
            errors.append(f"crossref: {exc}")
    if not papers and errors:
        return LiteratureObservation(
            ok=False, summary="literature search failed: " + "; ".join(errors)
        )
    return LiteratureObservation(
        ok=True,
        summary=f"{len(papers)} paper(s) found for '{params.query}'"
        + (f" (partial: {'; '.join(errors)})" if errors else ""),
        papers=papers[: 2 * params.max_results],
    )
