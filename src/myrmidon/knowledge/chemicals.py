"""PubChem lookup: ground molecular facts (formula, charge, SMILES) in a
citable database instead of the model's memory."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext

_PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_PROPS = "MolecularFormula,MolecularWeight,Charge,CanonicalSMILES,IUPACName"


class ChemicalObservation(Observation):
    cid: int | None = None
    name: str | None = None
    formula: str | None = None
    molecular_weight: float | None = None
    charge: int | None = None
    smiles: str | None = None
    iupac_name: str | None = None
    source: str = "pubchem"


class PubchemParams(BaseModel):
    query: str = Field(description="compound name, SMILES, or CID")
    query_type: str = Field(
        default="name", pattern="^(name|smiles|cid)$",
        description="how to interpret the query",
    )


def lookup_pubchem(ctx: ToolContext, params: PubchemParams) -> ChemicalObservation:
    url = f"{_PUG}/compound/{params.query_type}/{params.query}/property/{_PROPS}/JSON"
    try:
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
        props = resp.json()["PropertyTable"]["Properties"][0]
    except httpx.HTTPError as exc:
        return ChemicalObservation(
            ok=False, summary=f"PubChem lookup failed for '{params.query}': {exc}"
        )
    except (KeyError, IndexError, ValueError):
        return ChemicalObservation(
            ok=False, summary=f"PubChem returned no properties for '{params.query}'"
        )
    cid = props.get("CID")
    charge = props.get("Charge")
    formula = props.get("MolecularFormula")
    weight = float(props["MolecularWeight"]) if props.get("MolecularWeight") else None
    return ChemicalObservation(
        ok=True,
        summary=f"PubChem CID {cid}: {formula}, MW {weight}, net charge {charge} "
        f"(cite as: PubChem CID {cid})",
        cid=cid,
        name=params.query if params.query_type == "name" else None,
        formula=formula,
        molecular_weight=weight,
        charge=charge,
        smiles=props.get("CanonicalSMILES"),
        iupac_name=props.get("IUPACName"),
    )
