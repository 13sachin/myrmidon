"""fm_displace_plan: the displaced single-point windows a reference is extended with."""

from __future__ import annotations

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.inputs import resolve_in_ws


class DisplacedWindow(BaseModel):
    label: str
    target_a: float
    coords: str  # workspace-relative displaced .gro




class FMDisplacePlanParams(BaseModel):
    coords: str = Field(description="base structure (.gro) to displace")
    pair: list[int] = Field(
        min_length=2, max_length=2,
        description="[anchor, mobile] GRO atom serials (1-based): the mobile atom is "
        "moved along the anchor->mobile axis to each target distance",
    )
    pair_label: str = Field(description="a name for the pair, e.g. 'C1-O1' or 'ion-ligand O'")
    targets_a: list[float] = Field(min_length=1, description="target distances (Å)")
    rigid_group: list[int] = Field(
        default_factory=list,
        description="GRO serials translated RIGIDLY together with the mobile "
        "atom (e.g. a whole ligand). Internal geometry is preserved exactly, "
        "so displaced single points carry no intramolecular strain, which "
        "moving one atom of a molecule alone introduces. Must include "
        "the mobile atom; must not include the anchor.",
    )
    stage: str = Field(default="displace")


@tool(
    "fm_displace_plan",
    description="Build displaced structures for force-matching single points: the pair's mobile "
                "atom moved along the pair axis to each target distance. Feed each .gro to "
                "qm_region, gmx_grompp, mimic_prepare and mimic_run; the restraint-free way "
                "to sample off-equilibrium geometries.",
    tags={"mimic", "fm"},
    requires={"mimicpy"},
    contract=Contract(consumes={"coords": "structure"},
                      produces=(Produces("structure"),)),
)
def fm_displace_plan(ctx: ToolContext, params: FMDisplacePlanParams) -> Observation:
    src = resolve_in_ws(ctx, params.coords)
    lines = src.read_text().splitlines()
    n_atoms = int(lines[1].strip())
    ia, ib = params.pair
    if not (1 <= ia <= n_atoms and 1 <= ib <= n_atoms):
        return Observation(
            ok=False, summary=f"pair {params.pair} outside 1..{n_atoms}"
        )
    group = sorted(set(params.rigid_group))
    if group:
        if ib not in group:
            return Observation(
                ok=False, summary=f"rigid_group must include the mobile atom {ib}")
        if ia in group:
            return Observation(
                ok=False, summary=f"rigid_group must not include the anchor {ia}")
        bad = [s for s in group if not (1 <= s <= n_atoms)]
        if bad:
            return Observation(
                ok=False, summary=f"rigid_group serials outside 1..{n_atoms}: {bad[:5]}")

    def xyz(line: str) -> list[float]:
        return [float(line[20:28]), float(line[28:36]), float(line[36:44])]

    la, lb = lines[1 + ia], lines[1 + ib]
    a, b = xyz(la), xyz(lb)
    vec = [b[k] - a[k] for k in range(3)]
    d0_nm = sum(v * v for v in vec) ** 0.5
    if d0_nm < 1e-6:
        return Observation(ok=False, summary="pair atoms coincide")
    unit = [v / d0_nm for v in vec]

    stage = ctx.workspace.stage(params.stage)
    windows: list[DisplacedWindow] = []
    artifacts = []
    for target in params.targets_a:
        t_nm = target / 10.0
        out_lines = list(lines)
        if group:
            # rigid translation: every group atom shifts by the same vector
            # that brings the mobile atom to the target distance
            shift = [(a[k] + unit[k] * t_nm) - b[k] for k in range(3)]
            for s in group:
                ls = out_lines[1 + s]
                c = xyz(ls)
                nc = [c[k] + shift[k] for k in range(3)]
                out_lines[1 + s] = ls[:20] + "".join(f"{v:8.3f}" for v in nc) + ls[44:]
        else:
            new_b = [a[k] + unit[k] * t_nm for k in range(3)]
            moved = lb[:20] + "".join(f"{c:8.3f}" for c in new_b) + lb[44:]
            out_lines[1 + ib] = moved
        out = stage / f"disp_{target:g}A.gro"
        out.write_text("\n".join(out_lines) + "\n")
        windows.append(
            DisplacedWindow(
                label=f"{params.pair_label}@{target:g}A",
                target_a=target,
                coords=ctx.workspace.relpath(out),
            )
        )
        artifacts.append(ArtifactRef.from_path(out, "structure", root=ctx.workspace.root))
    return Observation(
        ok=True,
        summary=f"{len(windows)} displaced structure(s) for {params.pair_label} "
        f"(gro atoms {ia}->{ib}, original {d0_nm * 10:.3f} Å) at "
        f"{', '.join(f'{t:g}' for t in params.targets_a)} Å",
        artifacts=artifacts,
        windows=windows,
        pair=params.pair,
        original_distance_a=round(d0_nm * 10, 4),
    )
