"""A protocol's stages tried on the built system, at no cost.

In a throwaway study under ``.myrmidon/preview/``, removed afterwards so the study's ledger
records none of it, a mandate is written from the protocol's stages over the study's own
structure, topology and index, and each step that cites a stage is rendered from it and
its tool called on the rendered file — for a run input, the engine's own preprocessor. A
step that reads what only an earlier stage makes (a QM region's index, a restart) is tried
once that exists. A stage that runs is priced from the protocol's measurement, scaled by
the length its rendered input states (the kind's ``run_length``) against the measured
length (``measured_ps``) or the stage's own. What a stage could not render or build is said
with it; a value only the researcher can state is asked for.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from myrmidon.core.contracts import canonical, run_length
from myrmidon.core.tool import REGISTRY, ToolContext
from myrmidon.core.workspace import Workspace
from myrmidon.errors import UnknownToolError
from myrmidon.knowledge.inputsets import RENDERED_FILES

PREVIEW_DIR = "preview"


@dataclass
class Stage:
    name: str
    tool: str
    core_h: float | None = None  # the price, where the stage runs a job
    basis: str = ""
    rendered: bool | None = None  # None: not tried (no values to render, or not yet)
    built: bool | None = None  # its tool took the rendered file
    said: list[str] = field(default_factory=list)  # what refused it, or what it waits on

    def line(self) -> str:
        cost = f"{self.core_h:g} core-h" if self.core_h is not None else "books nothing"
        state = ("; renders and builds" if self.built else
                 "; renders" if self.rendered and self.built is None else
                 "; does NOT build" if self.rendered else
                 "; does NOT render" if self.rendered is False else
                 "; not tried" if self.said else "")
        return (f"{self.name} ({self.tool}): {cost}{state}"
                + (f" — {self.basis}" if self.basis else "")
                + "".join(f"\n    {s}" for s in self.said))


def stages(ctx: ToolContext, protocol, system: dict[str, str],
           overrides: dict[str, dict]) -> list[Stage]:
    """Every stage of ``protocol``, priced, and — given the study's ``structure`` and
    ``topology`` (and ``index``) in ``system`` — each stage that renders tried on them."""
    trial = None
    if ctx.workspace is not None and system.get("structure") and system.get("topology"):
        root = Path(ctx.workspace.state_dir) / PREVIEW_DIR / uuid.uuid4().hex[:8]
        trial = replace(ctx, workspace=Workspace.create(root, name="preview"),
                        unattended=True)
    try:
        return _stages(ctx, trial, protocol, system, overrides)
    finally:
        if trial is not None:
            shutil.rmtree(trial.workspace.root, ignore_errors=True)


def _stages(ctx, trial, protocol, system, overrides) -> list[Stage]:
    files = {k: str(ctx.workspace.resolve(v)) for k, v in system.items() if v} if trial else {}
    unwritten = _mandate(trial, protocol, files) if trial else {}
    out: list[Stage] = []
    length: dict[str, tuple] = {}  # a role a tried stage makes -> (ps, steps, set's steps)
    for step in protocol.steps:
        stage = Stage(step.name, step.tool, step.estimate_core_h or None,
                      step.estimate_basis or "")
        out.append(stage)
        if trial is not None and step.cites:
            later = [r for r in step.needs if not _stands_in(r, files)]
            made_earlier = {r for s in protocol.steps[:protocol.steps.index(step)]
                            for r in s.produces}
            cited = _cited(step)[0]
            if later:
                stage.said.append("tried once " + ", ".join(later) + " exist(s): "
                                  + ("an earlier stage makes it" if set(later) <= made_earlier
                                     else "pass the file the system has, if it has one"))
            elif cited in unwritten:
                stage.said.append(f"stage {cited} is not written over this system: "
                                  + unwritten[cited]
                                  + ("; its overrides were not used"
                                     if overrides.get(step.name) else ""))
            else:
                made = _try(trial, step, files, overrides.get(step.name, {}), stage)
                length.update({role: made for role in step.produces})
        if stage.core_h:
            _price(step, stage, next((length[r] for r in step.needs if r in length), None))
    unknown = sorted(set(overrides) - {s.name for s in protocol.steps})
    if unknown and out:
        out[0].said.append(f"no stage {', '.join(unknown)}: their overrides were not used")
    return out


def _price(step, stage: Stage, made) -> None:
    """The protocol's estimate, scaled by the length the stage's input states."""
    ps, steps, own = made or (None, None, None)
    if ps and step.measured_ps:
        ratio, what = ps / step.measured_ps, f"for {step.measured_ps:g} ps, × "
        what += f"{ps / step.measured_ps:.3g} for the {ps:g} ps set here"
    elif steps and own and not step.measured_ps:
        ratio = steps / own
        what = f"for the stage's own length, × {ratio:.3g} for the length set here"
    else:
        return
    if ratio > 0 and ratio != 1.0:
        stage.core_h = round(stage.core_h * ratio, 3)
        stage.basis = f"the protocol's {step.estimate_core_h:g} core-h {what}"


def _stands_in(role: str, files: dict[str, str]) -> bool:
    """Whether the built system can stand in for what a stage reads: its structure for
    any later structure, its topology, its index."""
    return ((role.startswith("structure") and "structure" in files)
            or (role in ("topology", "index") and role in files))


def _cited(step) -> tuple[str, str]:
    """(the stage a step cites, the file it is rendered as)."""
    name, _, renders = (step.cites or "").removeprefix("stage:").partition("/")
    return name, renders


def _mandate(trial, protocol, files: dict[str, str]) -> dict[str, str]:
    """Write the protocol's stages as the trial study's mandate over the study's files;
    stage name -> why it is not in it. Stages a QM/MM Hamiltonian or a QM region cannot be
    written for on this system are left out, so the classical ones are still tried."""
    stages = [s.stage.model_dump(mode="json", exclude_none=True)
              for s in protocol.steps if s.stage is not None]
    call = {"question": f"preview of {protocol.id}", "ceiling_core_h": 1e9,
            "protocol": f"{protocol.id}@v{protocol.version}",  # its cross-step rules too
            "system_files": [{"path": files[k], "kind": k}
                             for k in ("structure", "topology", "index") if k in files],
            **({"hamiltonian": protocol.hamiltonian.model_dump(mode="json", exclude_none=True)}
               if protocol.hamiltonian is not None else {})}
    written = REGISTRY.call("mandate_write", trial, {**call, "stages": stages})
    if written.ok:
        return {}
    why = written.error.message if written.error else written.summary
    classical = [s for s in stages if s.get("regime", "mm") == "mm"]
    if classical and len(classical) < len(stages):
        call.pop("hamiltonian", None)
        if REGISTRY.call("mandate_write", trial, {**call, "stages": classical}).ok:
            return {s["name"]: why for s in stages if s not in classical}
    return {s["name"]: why for s in stages}


def _try(trial, step, files, overrides, stage: Stage) -> tuple:
    """Render the stage the step cites and call the step's tool on it: (the rendered run's
    ps and steps, the stage's own steps before any override), each None when unknown."""
    name, renders = _cited(step)
    kind, suffix = RENDERED_FILES.get(renders or "", (None, ""))
    common = {k: files[k] for k in ("structure", "topology", "index") if k in files}
    rendered = REGISTRY.call("inputset_apply", trial, {
        "stage": name, "overrides": overrides, **common,
        **({"output": f"01_preview/{step.name}{suffix}"} if kind else {})})
    asked = [f for f in rendered.findings if f.id == "INPUTSET_REQUIRED_PARAM_MISSING"]
    if asked:  # the researcher's value, not a broken stage: tried once given
        stage.said += [f"needs, from the researcher: {f.message}" for f in asked]
        return None, None, None
    stage.rendered = rendered.ok
    if not rendered.ok:
        stage.said.append(rendered.error.message if rendered.error else rendered.summary)
        return None, None, None
    path = next((a.path for a in rendered.artifacts), None)
    # a stage that is the rendering itself has nothing further to build
    built = (None if step.tool == "inputset_apply"
             else _call_tool(trial, step, files, kind, path, stage))
    stage.built = built.ok if built is not None else None
    stage.said += [f.message for f in [*rendered.findings,
                                       *(built.findings if built else [])]
                   if f.severity != "info"][:4]
    if built is not None and not built.ok:
        stage.said.append(built.error.message if built.error else built.summary)
        # the engine's own words, which say why
        stage.said += [line for line in (built.log_tail or "").splitlines()
                       if line.strip()][-8:]
    values = getattr(rendered, "values", None) or {}
    ps, steps = run_length(kind, values) if kind else (None, None)
    own = steps
    if overrides and kind:  # the stage's own length, as the protocol states it
        plain = REGISTRY.call("inputset_apply", trial, {
            "stage": name, **common, "output": f"01_preview/{step.name}.own{suffix}"})
        own = run_length(kind, getattr(plain, "values", None) or {})[1]
        for artifact in plain.artifacts:  # read, then gone: the overridden render stands
            (trial.workspace.root / artifact.path).unlink(missing_ok=True)
    return ps, steps, own


def _call_tool(trial, step, files, kind, path, stage: Stage):
    """The stage's tool on the rendered file and the built system, by the parameters the
    step wires to roles and the one its contract takes the rendered kind in; None when the
    tool takes no such file or needs a parameter nothing here supplies."""
    if step.tool == "stage_prepare":
        return REGISTRY.call("stage_prepare", trial, {
            "stage": _cited(step)[0], **{k: files[k] for k in ("structure", "topology", "index")
                                         if k in files}})
    try:
        spec = REGISTRY.get(step.tool)
    except UnknownToolError:
        return None
    contract = spec.contract
    takes = next((param for param in (contract.consumes if contract else {})
                  if kind and canonical(kind) in contract.accepts(param)), None)
    if takes is None or path is None:
        return None
    params = {param: files[_system_key(role)] for param, role in step.values.items()
              if _stands_in(role, files)}
    params[takes] = path
    missing = [name for name, fld in spec.params_model.model_fields.items()
               if fld.is_required() and name not in params]
    if missing:
        stage.said.append(f"{step.tool} also needs {', '.join(missing)}: tried in the session")
        return None
    return REGISTRY.call(step.tool, trial, params)


def _system_key(role: str) -> str:
    return "structure" if role.startswith("structure") else role
