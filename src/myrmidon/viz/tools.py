"""Visualization tools: drive a live ChimeraX session, render grounded figures.

The protocol is a feedback loop, not a one-shot script: ``viz_open`` loads a
structure and reports what is actually in it (atoms, chains, ligands — parsed
from ChimeraX, never assumed), ``viz_command`` iterates on the scene with a
quick draft snapshot after every batch so the agent can look at what it did,
and ``viz_command`` with an ``output`` renders the final high-resolution figure
with a style preset. Every snapshot is an artifact in the ledger; the camera matrix is
recorded so the exact view can be restored.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import VizObservation
from myrmidon.core.tool import ToolContext, tool
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.errors import MyrmidonError
from myrmidon.viz import presets
from myrmidon.viz.session import CommandResult, VizSession

_NOTE_LIMIT = 2000  # keep observations readable; full log stays in ChimeraX


def _session(ctx: ToolContext) -> tuple[VizSession, bool]:
    info = ctx.engines.get("chimerax")
    if info is None or not info.available:
        hint = getattr(info, "hint", "") or "chimerax not discovered on this machine"
        raise MyrmidonError(f"ChimeraX unavailable: {hint}")
    return VizSession.ensure(info.path, ctx.workspace.state_dir)


def _quote(path: str) -> str:
    return f'"{path}"' if " " in path else path


def _next_draft(stage) -> str:
    n = len(list(stage.glob("draft_*.png"))) + 1
    return f"draft_{n:03d}.png"


def _snapshot(ctx, session, stage, name: str, width: int, height: int | None,
              supersample: int = 1, transparent: bool = False) -> tuple[str, ArtifactRef]:
    out = stage / name
    cmd = f"save {_quote(str(out))} width {width}"
    if height:
        cmd += f" height {height}"
    cmd += f" supersample {supersample}"
    if transparent:
        cmd += " transparentBackground true"
    result = session.run(cmd)
    if not result.ok or not out.exists():
        raise MyrmidonError(f"snapshot failed: {result.error or 'no file written'}")
    role = "figure" if supersample > 1 else "snapshot"
    return ctx.workspace.relpath(out), ArtifactRef.from_path(
        out, role, root=ctx.workspace.root
    )


def _view_matrix(session) -> str | None:
    result = session.run("view matrix")
    return result.text().strip() or None


def _notes(results: list[CommandResult]) -> str:
    text = "".join(note for r in results for note in r.notes).strip()
    return text[:_NOTE_LIMIT]


class VizOpenParams(BaseModel):
    structure: str = Field(description="structure file (.pdb/.gro/.cif), workspace-relative")
    close_existing: bool = Field(
        default=True, description="close previously opened models first"
    )


@tool(
    "viz_open",
    description="Open a structure in the live ChimeraX session (launching it on first "
    "use — on macOS a window appears that the user can watch and rotate). Reports atom "
    "count, chains, and non-polymer residues parsed from the file, plus a draft "
    "snapshot; build all selections from these reported identifiers.",
    returns=VizObservation,
    tags={"viz"},
    requires={"chimerax"},
)
def viz_open(ctx: ToolContext, params: VizOpenParams) -> VizObservation:
    structure = resolve_in_ws(ctx, params.structure)
    stage = ctx.workspace.stage("viz")
    session, started = _session(ctx)
    executed: list[str] = []
    if params.close_existing:
        session.run("close session")
        executed.append("close session")

    opened = session.run(f"open {_quote(str(structure))}")
    executed.append(opened.command)
    if not opened.ok:
        return VizObservation(
            ok=False,
            summary=f"ChimeraX could not open {params.structure}: {opened.error}",
            session_started=started,
            executed=executed,
            log_tail=_notes([opened]) or None,
        )

    models = session.run("info models attribute num_atoms")
    n_models, n_atoms = 0, 0
    for entry in models.json_values or [[]]:
        for model in entry if isinstance(entry, list) else []:
            if isinstance(model, dict) and isinstance(model.get("value"), int):
                n_models += 1
                n_atoms += model["value"]

    chains: dict[str, int] = {}
    chain_info = session.run("info chains")
    for entry in chain_info.json_values or [[]]:
        for chain in entry if isinstance(entry, list) else []:
            if isinstance(chain, dict) and chain.get("value"):
                chains[str(chain["value"])] = len(chain.get("residues") or [])

    ligands: list[str] = []
    # ``info residues`` reports attribute ``name`` by default: keep it next to
    # the spec so the agent can tell NAP from ICT without guessing.
    ligand_info = session.run("info residues ligand")
    for entry in ligand_info.json_values or [[]]:
        for res in entry if isinstance(entry, list) else []:
            if isinstance(res, dict) and res.get("spec"):
                name = res.get("value") if res.get("attribute") == "name" else None
                ligands.append(f"{res['spec']} {name}" if name else str(res["spec"]))
    executed += [models.command, chain_info.command, ligand_info.command]

    image, artifact = _snapshot(ctx, session, stage, _next_draft(stage), 800, 600)
    chain_desc = ", ".join(f"{c} ({n} res)" for c, n in chains.items()) or "none parsed"
    return VizObservation(
        ok=True,
        summary=f"Opened {params.structure}: {n_atoms or '?'} atoms, "
        f"chains {chain_desc}, {len(ligands)} ligand residue(s). "
        f"Draft snapshot at {image}.",
        session_started=started,
        n_models=n_models,
        n_atoms=n_atoms or None,
        chains=chains,
        ligands=ligands[:50],
        image=image,
        executed=executed,
        artifacts=[artifact],
    )


class VizCommandParams(BaseModel):
    commands: list[str] = Field(
        default_factory=list,
        description="ChimeraX commands, run in order (e.g. 'color :AKG byhetero')",
    )
    preset: str | None = Field(
        default=None, description="apply a style preset first: paper | flat | qmmm"
    )
    qm_selection: str | None = Field(
        default=None, description="QM-region selection, required by the qmmm preset"
    )
    snapshot: bool = Field(default=True, description="render a png afterwards: a draft "
                           "(800 px) unless output is given")
    output: str | None = Field(
        default=None, description="render a publication figure to this filename (png/tiff) "
        "instead of a draft: width_px wide, supersampled, the camera matrix recorded so "
        "the exact view can be restored — pair with preset='paper'")
    width_px: int = Field(default=2400, description="figure width (2400 ≈ 8 in @ 300 dpi)")
    height_px: int | None = Field(default=None, description="figure height (default: aspect)")
    supersample: int = Field(default=3, description="figure antialiasing supersample factor")
    transparent_background: bool = Field(default=False)
    close_session: bool = Field(default=False, description="close the workspace's ChimeraX "
                                "session and its window instead of running anything")


@tool(
    "viz_command",
    description="Drive the live ChimeraX session: run commands in order and return a draft "
    "snapshot to look at and iterate on (stops at the first failing command and reports "
    "its error; a preset applies a style block first); with output=, render the current "
    "scene as a publication figure — 2400 px wide, supersampled, camera matrix recorded; "
    "with close_session=true, close the session and its window.",
    returns=VizObservation,
    tags={"viz"},
    requires={"chimerax"},
)
def viz_command(ctx: ToolContext, params: VizCommandParams) -> VizObservation:
    if params.close_session:
        session = VizSession.attach(ctx.workspace.state_dir)
        if session is None:
            return VizObservation(ok=True, summary="no live visualization session")
        session.close(ctx.workspace.state_dir)
        return VizObservation(ok=True, summary="ChimeraX session closed")
    stage = ctx.workspace.stage("viz")
    session, started = _session(ctx)
    commands = list(params.commands)
    if params.preset:
        commands = presets.resolve(params.preset, params.qm_selection) + commands
    if not commands and not params.snapshot and not params.output:
        return VizObservation(ok=False, summary="nothing to do: no commands, no snapshot")

    executed: list[str] = []
    results: list[CommandResult] = []
    for command in commands:
        result = session.run(command)
        executed.append(command)
        results.append(result)
        if not result.ok:
            return VizObservation(
                ok=False,
                summary=f"command failed: '{command}' — {result.error} "
                f"({len(executed) - 1} command(s) before it succeeded)",
                session_started=started,
                executed=executed,
                log_tail=_notes(results) or None,
            )

    if params.output:
        image, artifact = _snapshot(
            ctx, session, stage, params.output,
            params.width_px, params.height_px,
            supersample=params.supersample,
            transparent=params.transparent_background,
        )
        return VizObservation(
            ok=True,
            summary=(f"{len(executed)} command(s) ran clean; " if executed else "")
            + f"figure rendered to {image} ({params.width_px} px wide, "
            f"supersample {params.supersample})",
            session_started=started,
            image=image,
            view_matrix=_view_matrix(session),
            executed=executed,
            artifacts=[artifact],
        )
    image, artifacts = None, []
    if params.snapshot:
        image, artifact = _snapshot(ctx, session, stage, _next_draft(stage), 800, 600)
        artifacts = [artifact]
    return VizObservation(
        ok=True,
        summary=f"{len(executed)} command(s) ran clean"
        + (f"; draft snapshot at {image}" if image else ""),
        session_started=started,
        image=image,
        executed=executed,
        artifacts=artifacts,
    )
