"""RCSB PDB text search (fetch_pdb lives in basic_tools; this finds entries)."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext

_RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"


class PdbHit(BaseModel):
    pdb_id: str
    score: float | None = None


class PdbSearchObservation(Observation):
    hits: list[PdbHit] = Field(default_factory=list)


class PdbSearchParams(BaseModel):
    query: str = Field(description="free-text query, e.g. 'trp-cage miniprotein'")
    max_results: int = Field(default=10, ge=1, le=50)


def search_pdb(ctx: ToolContext, params: PdbSearchParams) -> PdbSearchObservation:
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": params.query},
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": params.max_results}},
    }
    try:
        resp = httpx.post(_RCSB_SEARCH, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        return PdbSearchObservation(
            ok=False, summary=f"RCSB search failed: {exc}"
        )
    except ValueError:
        return PdbSearchObservation(ok=False, summary="RCSB returned unparsable JSON")
    hits = [
        PdbHit(pdb_id=item["identifier"], score=item.get("score"))
        for item in data.get("result_set", [])
        if item.get("identifier")
    ]
    return PdbSearchObservation(
        ok=True,
        summary=f"{len(hits)} PDB entr{'y' if len(hits) == 1 else 'ies'} matched "
        f"'{params.query}'"
        + (f"; top: {hits[0].pdb_id}" if hits else ""),
        hits=hits,
    )
