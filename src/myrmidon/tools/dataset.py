"""The reference dataset a parameter fit trains against, collected from a stage's runs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.lineage import stage_runs
from myrmidon.core.mandate import read_mandate
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.registry import fit_adapter
from myrmidon.tools.inputsets import system_file


class DatasetCollectParams(BaseModel):
    stage: str = Field(default="", description="the reference stage: its collected runs' reference "
                       "data and trajectories, merged when there are several")
    fmdata: list[str] = Field(default_factory=list, description="reference force files in place "
                              "of a stage's, in order")
    trajectory: list[str] = Field(default_factory=list, description="the matching trajectories, "
                                  "in the same order")
    topology: str | None = Field(default=None, description="default: the system's")
    coords: str | None = Field(default=None, description="default: the system's structure")
    selection: str | None = Field(default=None, description="the QM selection file; default: "
                                  "the one qm_region wrote")
    check_frames: int = Field(default=20, ge=1, description="frames to spot-check for alignment")
    align_if_desynced: bool = Field(default=True, description="when frames and forces have "
                                    "drifted apart, register an index-aligned pair instead: the "
                                    "fit refuses a pair that is not")
    monitor_pairs: dict[str, list[int]] = Field(
        default_factory=dict, description="named atom pairs to track per frame, by global atom "
        "id as the dataset stores them; distances in Å. A reprocessed or aligned dataset stores "
        "its QM atoms in its own record order, so a first-shell pair reading 8-13 Å names the "
        "wrong atoms")
    cpmd_logs: list[str] = Field(default_factory=list, description="logs of the generating runs, "
                                 "for temperature and SCF health per segment; default: the "
                                 "stage's")
    duplicate_rmsd_a: float = Field(default=0.05, gt=0, description="QM RMSD (Å) below which "
                                    "consecutive frames count as near-duplicates")
    output_stage: str = Field(default="fm", description="where the dataset is written")


@tool(
    "dataset_collect",
    description="Collect the reference dataset a parameter_fit stage reads from a stage's runs: "
    "merged when several, aligned with their trajectories, frames counted, force outliers and "
    "near-duplicates summarised, both files hashed.",
    tags={"qmmm", "fm"},
    contract=Contract(consumes={"fmdata": ("fm_reference", "reference_dataset"),
                                "trajectory": "trajectory", "topology": "topology",
                                "coords": "structure", "selection": "qm_selection"},
                      produces=(Produces("reference_dataset"),)),
)
def dataset_collect(ctx: ToolContext, params: DatasetCollectParams) -> Observation:
    from myrmidon.engines.mimic.fm_dataset import DatasetRequest

    engine = fit_adapter()
    mandate = read_mandate(ctx.workspace)
    runs = stage_runs(ctx.workspace, params.stage) if params.stage else []
    if params.stage and not runs:
        return Observation(ok=False, summary=f"no collected run of stage {params.stage}: run and "
                           "collect it, or name fmdata and trajectory")

    def of_runs(role: str) -> list[str]:
        return [a.path for outputs, _ in runs for a in outputs if a.role == role]

    def system(kind: str, given: str | None) -> str | None:
        return given or (system_file(ctx, mandate, kind) if mandate is not None else None)

    fmdata = params.fmdata or of_runs("fm_reference")
    trajectory = params.trajectory or of_runs("trajectory")
    topology, coords = system("topology", params.topology), system("structure", params.coords)
    selection = system("qm_selection", params.selection)
    missing = [name for name, value in (("reference forces", fmdata), ("trajectory", trajectory),
                                        ("topology", topology), ("coords", coords),
                                        ("selection", selection)) if not value]
    if missing:
        return Observation(ok=False, summary=f"a dataset needs its {', '.join(missing)}: name "
                           "them, or the stage whose runs wrote them")
    return engine.collect_dataset(ctx, DatasetRequest(
        fmdata=fmdata, trajectory=trajectory, topology=topology, coords=coords,
        selection=selection, stage=params.output_stage, check_frames=params.check_frames,
        align_if_desynced=params.align_if_desynced, monitor_pairs=params.monitor_pairs,
        cpmd_logs=params.cpmd_logs or of_runs("qm_log"),
        duplicate_rmsd_a=params.duplicate_rmsd_a))
