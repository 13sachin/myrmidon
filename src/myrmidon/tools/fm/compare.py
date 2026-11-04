"""fm_force_compare: how far one topology's forces are from a QM/MM reference.

The topology's forces come from a classical rerun of reference frames (``gmx_rerun``);
the reference forces from the force-matching dataset. Per atom type and per atom, in
atomic units, for any topology, stock or fitted, so a comparison of candidates is the
same call on each, over the same frames.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef, fresh_path
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, precheck, tool
from myrmidon.core.validation import Finding
from myrmidon.engines.inputs import require_input, resolve_in_ws
from myrmidon.engines.mimic.fm import (
    FORCE_UNITS,
    force_residual_rows,
    parse_validation_force_residuals,
)


class FMForceCompareParams(BaseModel):
    topology: str = Field(description="the topology whose forces the rerun computed (.top)")
    structure: str = Field(description="the reference structure that defines atom order "
                           "(.gro)")
    forces: str = Field(description="the rerun trajectory carrying that topology's forces "
                        "(the trajectory gmx_rerun writes)")
    fmdata: str = Field(description="the force-matching dataset (.h5, from fm_collect)")
    selection: str = Field(description="the QM selection file, as fm_fit takes it")
    frames: list[int] = Field(
        default_factory=list,
        description="the dataset configuration each rerun frame is, in order (the indices "
        "traj_frames wrote the rerun's trajectory from); empty: frame k is configuration k")
    label: str = Field(default="", description="the topology's name in tables (default: "
                       "its file name)")
    stage: str = Field(default="compare")




def compare_gate(ctx: ToolContext, params: FMForceCompareParams):
    warnings: list[Finding] = []
    for name in ("topology", "structure", "forces", "fmdata", "selection"):
        _, missing = require_input(ctx, name, getattr(params, name))
        if missing:
            warnings.append(missing)
    return warnings, ["input_files"]


@tool(
    "fm_force_compare",
    description="Compare one topology's forces with a QM/MM force-matching reference on the QM "
                "atoms (gmx_rerun forces vs the dataset's): RMSE overall, per atom type and per "
                "atom, with outliers, in Hartree/Bohr. Call it per topology over the same frames "
                "to compare candidates.",
    tags={"mimic", "fm", "analysis"},
    requires={"mimicpy"},
    contract=Contract(consumes={"topology": "topology", "structure": ("structure", "tpr"),
                                "forces": "trajectory", "fmdata": "fm_dataset"},
                      produces=(Produces("data"),)),
)
def fm_force_compare(ctx: ToolContext, params: FMForceCompareParams
                     ) -> Observation:
    precheck(ctx, compare_gate(ctx, params)[0])
    from mimicpy.force_matching.qm_region import QMRegion
    from mimicpy.scripts.fmdata import FMDataset

    top = resolve_in_ws(ctx, params.topology)
    structure = resolve_in_ws(ctx, params.structure)
    forces = resolve_in_ws(ctx, params.forces)
    label = params.label or top.stem
    info = ctx.engines.get("gromacs")
    gmxdata = info.data_dir if info is not None and info.data_dir else None
    qm_region = QMRegion(str(top), str(structure), gmxdata=gmxdata)
    qm_region.setup_qm_region(Path(resolve_in_ws(ctx, params.selection)))
    fmdata = FMDataset(str(resolve_in_ws(ctx, params.fmdata)))
    frames = list(params.frames)
    if not frames:
        import MDAnalysis as mda

        frames = list(range(len(mda.Universe(str(structure), str(forces)).trajectory)))
    try:
        header, rows = force_residual_rows(forces, structure, fmdata, qm_region, frames)
    except ValueError as exc:
        return Observation(ok=False, summary=f"{label}: {exc}", topology=str(top),
                                         label=label)
    stage = ctx.workspace.stage(params.stage)
    csv_path = fresh_path(stage / f"force_residuals_{label}.csv")
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n")
    parsed = parse_validation_force_residuals(csv_path.read_text())
    parsed.pop("worst_atoms", None)
    summary_path = fresh_path(stage / f"{label}_force_compare.json")
    summary_path.write_text(json.dumps({"label": label, "frames": frames, **parsed},
                                       indent=1) + "\n")
    root = ctx.workspace.root
    return Observation(
        ok=True,
        summary=f"{label}: force RMSE {parsed['rmse']:.4f} {FORCE_UNITS} on {parsed['n_atoms']} "
        f"QM atoms over {parsed['n_frames']} frames"
        + (f" ({parsed['rel_rmse']:.1%} of the reference RMS {parsed['ref_rms']:.4f})"
           if parsed.get("rel_rmse") is not None else "")
        + f"; {len(parsed['outliers'])} outlier atom(s)",
        artifacts=[ArtifactRef.from_path(summary_path, "data", root=root),
                   ArtifactRef.from_path(csv_path, "force_residuals", root=root)],
        topology=ctx.workspace.relpath(top), label=label, units=FORCE_UNITS,
        **{k: parsed.get(k) for k in ("n_frames", "n_atoms", "n_samples", "rmse", "ref_rms",
                                      "rel_rmse", "median_atom_rmse", "by_type", "outliers")},
    )
