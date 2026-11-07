"""Partition a system into its QM region and the rest, in one call, by the engine that runs
the region (``EngineAdapter.partition_qm``); which atoms are quantum is the researcher's."""

from __future__ import annotations

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.mandate import read_mandate
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.facts import data_dir_of, system_parts
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.registry import adapter_with


class QMRegionParams(BaseModel):
    topology: str = Field(description="GROMACS topology (.top)")
    coords: str = Field(description="coordinate file (.gro)")
    selections: list[str] = Field(default_factory=list, description=(
        "MiMiCPy selection commands, in order: 'add SEL', 'add-bound SEL' (boundary atoms, given "
        "pp_info's boundary pseudopotential), 'delete SEL', 'clear'. SEL is 'KEYWORD is VALUE' "
        "(resname, resid, name, id, ...) joined by 'and'/'or': 'add resid is 42 and name is OG'"))
    subsystems: list[str] = Field(default_factory=list, description=(
        "whole subsystems taken in before the selections, by the mandate's name or by kind "
        "('ligand'): each adds its residues"))
    padding_nm: float = Field(gt=0, description=(
        "vacuum between the QM atoms and the QM cell wall, per side (nm); prepqm's default of 0 "
        "crashes CPMD; 0.3-0.5 nm is typical for an isolated-molecule Poisson solver"))
    template: str | None = Field(default=None, description="CPMD template input — prepqm -inp")
    pp_info: str | None = Field(default=None, description=(
        "pseudopotential table (element, file, ..., LMAX) — prepqm -pp, its files in CPMD's "
        "PP_LIBRARY_PATH; without it every species gets LMAX=S"))
    atomtypes_map: dict[str, str] = Field(default_factory=dict, description=(
        "atom type -> element, for a type guessed_elements shows misread — prepqm -nsa"))
    qm_group: str = Field(default="QMatoms", description="index group name for the QM atoms")
    stage: str = Field(default="qmmm", description="the coupled runs' stage (its &MIMIC PATHS)")
    _unresolved: list[str] = PrivateAttr(default_factory=list)  # names no part resolves

    @model_validator(mode="after")
    def _names_atoms(self) -> QMRegionParams:
        if not (self.selections or self.subsystems):
            raise ValueError("name the region's atoms: selections, subsystems or both")
        return self

    def sized(self, ctx: ToolContext) -> QMRegionParams:
        """Each subsystem as the commands it comes to, so the ledger records what made it."""
        if not self.subsystems:
            return self
        commands, unresolved = region_selections(ctx, self.topology, self.subsystems)
        out = self.model_copy(update={"selections": [*commands, *self.selections]})
        out._unresolved = unresolved
        return out


def region_selections(ctx: ToolContext, topology: str, names: list[str]
                      ) -> tuple[list[str], list[str]]:
    """(the selection commands the named subsystems come to, why a name does not resolve):
    the mandate's parts, else the topology's, a part named by molecule type taking the
    residue names the topology gives that type."""
    engine = adapter_with("select_subsystems")
    if engine is None:
        return [], [f"{n}: no engine here selects a QM region" for n in names]
    laid_out = system_parts({"topology": resolve_in_ws(ctx, topology)}, data_dir_of(ctx.engines))
    mandate = read_mandate(ctx.workspace)
    parts = ({k: p.model_dump() for k, p in mandate.system.subsystems.items()}
             if mandate is not None else laid_out)
    for part in parts.values():
        if not part.get("resnames") and part.get("molecules"):
            part["resnames"] = [r for p in laid_out.values()
                                if set(p.get("molecules") or ()) & set(part["molecules"])
                                for r in p.get("resnames") or ()]
    return engine.select_subsystems(parts, names)


@tool(
    "qm_region",
    description="Select the QM region; writes the system's CPMD input, the QM index group and "
    "the selection file, and says the charge, multiplicity and electrons the selection leaves "
    "and the elements guessed for atom types the topology gives no atomic number for.",
    tags={"qmmm"}, requires={"mimicpy"},
    contract=Contract(consumes={"topology": "topology", "coords": "structure",
                                "template": "cpmd_input"}, upstream="template",
                      produces=(Produces("cpmd_input"), Produces("index"),
                                Produces("qm_selection"))),
)
def qm_region(ctx: ToolContext, params: QMRegionParams) -> Observation:
    engine = adapter_with("partition_qm")
    if params._unresolved or engine is None:
        return Observation(ok=False, summary=f"cannot select {'; '.join(params._unresolved)}: "
                           "name those atoms as selections"
                           if params._unresolved else "no engine here partitions a QM region")
    return engine.partition_qm(ctx, params)
