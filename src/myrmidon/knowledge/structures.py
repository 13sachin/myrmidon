"""RCSB PDB text search and the chemical component dictionary (fetch_pdb lives in
tools/structure.py; this finds entries, and says what a deposited ligand's code is)."""

from __future__ import annotations

import re

import httpx
from pydantic import BaseModel, Field

from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext

_RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
_RCSB_CHEMCOMP = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"


class PdbHit(BaseModel):
    pdb_id: str
    score: float | None = None


class PdbSearchParams(BaseModel):
    query: str = Field(description="free-text query, e.g. 'trp-cage miniprotein'")
    max_results: int = Field(default=10, ge=1, le=50)


def search_pdb(ctx: ToolContext, params: PdbSearchParams) -> Observation:
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
        return Observation(
            ok=False, summary=f"RCSB search failed: {exc}"
        )
    except ValueError:
        return Observation(ok=False, summary="RCSB returned unparsable JSON")
    hits = [
        PdbHit(pdb_id=item["identifier"], score=item.get("score"))
        for item in data.get("result_set", [])
        if item.get("identifier")
    ]
    return Observation(
        ok=True,
        summary=f"{len(hits)} PDB entr{'y' if len(hits) == 1 else 'ies'} matched "
        f"'{params.query}'"
        + (f"; top: {hits[0].pdb_id}" if hits else ""),
        hits=hits,
    )


def chem_component(code: str) -> Observation:
    """What a residue code of the PDB names: its chemical component's name, formula,
    weight, SMILES and the charge of the dictionary's reference form — not the charge it
    carries at a study's pH, which is the protonation the researcher states."""
    code = code.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{1,5}", code):
        return Observation(ok=False, summary=f"{code!r} is no PDB residue code: 1-5 letters "
                           "or digits, as the deposit names the residue")
    try:
        resp = httpx.get(_RCSB_CHEMCOMP.format(code=code), timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        missing = exc.response.status_code == 404
        return Observation(ok=False, summary=f"RCSB holds no component {code!r}" if missing
                           else f"RCSB chemical component lookup failed: {exc}")
    except httpx.HTTPError as exc:
        return Observation(ok=False, summary=f"RCSB chemical component lookup failed: {exc}")
    except ValueError:
        return Observation(ok=False, summary="RCSB returned unparsable JSON")
    comp = data.get("chem_comp") or {}
    if not comp:
        return Observation(ok=False, summary=f"RCSB holds no component {code!r}")
    descriptor = data.get("rcsb_chem_comp_descriptor") or {}
    charge = comp.get("pdbx_formal_charge")
    return Observation(
        ok=True, source="rcsb-chemcomp",
        summary=(f"PDB chemical component {code}: {comp.get('name')}, {comp.get('formula')}; "
                 f"formal charge {charge} in the dictionary's reference form, not its "
                 "protonation at the study's pH (cite as: wwPDB Chemical Component "
                 f"Dictionary, {code})"),
        name=comp.get("name"), formula=comp.get("formula"),
        molecular_weight=comp.get("formula_weight"), reference_charge=charge,
        smiles=descriptor.get("SMILES_stereo") or descriptor.get("SMILES"),
    )
