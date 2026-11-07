"""A protocol's stages tried on the built system, at no cost.

In a throwaway study under ``.myrmidon/preview/``, removed afterwards so the study's ledger
records none of it, a mandate is written from the protocol — its stages, deliverables and
Hamiltonian — over the study's own structure, topology and index: the compiler prices each
stage for this system on this machine and checks the protocol's rules across its stages on
what it compiles. Each stage is then prepared on the built system as ``stage_prepare``
compiles it (for a run input, the engine's own preprocessor); a stage that starts from what
an earlier one leaves is prepared on the system's structure in its place. With no system, the
stages are priced here for no system in particular (``Protocol.priced``). A length the study
sets differently scales a stage's price by the steps it adds or takes away. What a stage
could not render or build is said with it; a value only the researcher can state is asked
for.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from myrmidon.core.contracts import run_length
from myrmidon.core.tool import REGISTRY, ToolContext
from myrmidon.core.workspace import Workspace

PREVIEW_DIR = "preview"
#: the system's files a stage is prepared on
SYSTEM_KINDS = ("structure", "topology", "index")


@dataclass
class Stage:
    name: str
    what: str  # its regime and kind
    core_h: float | None = None  # the price, where the stage runs a job
    basis: str = ""
    rendered: bool | None = None  # None: not tried (no system, or not written over it)
    built: bool | None = None  # its run input compiled
    said: list[str] = field(default_factory=list)  # what refused it, or what it waits on

    def line(self) -> str:
        cost = f"{self.core_h:g} core-h" if self.core_h is not None else "unpriced"
        state = ("; renders and builds" if self.built else
                 "; renders" if self.rendered and self.built is None else
                 "; does NOT build" if self.rendered else
                 "; does NOT render" if self.rendered is False else
                 "; not tried" if self.said else "")
        return (f"{self.name} ({self.what}): {cost}{state}"
                + (f" — {self.basis}" if self.basis else "")
                + "".join(f"\n    {s}" for s in self.said))


def stages(ctx: ToolContext, protocol, system: dict[str, str],
           overrides: dict[str, dict]) -> list[Stage]:
    """Every stage of ``protocol``, priced, and — given the study's ``structure`` and
    ``topology`` (and ``index``) in ``system`` — each prepared on them."""
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
    from myrmidon.execution.pace import read_paces

    files = {k: str(ctx.workspace.resolve(v)) for k, v in system.items() if v} if trial else {}
    prices = {name: price.model_dump() for name, price in protocol.priced(
        ctx.config.compute, read_paces(ctx.config.studies_root)).items()}
    unwritten = _mandate(trial, protocol, files, prices) if trial else {}
    common = {k: files[k] for k in SYSTEM_KINDS if k in files}
    out: list[Stage] = []
    for physical in protocol.stages:
        price = prices.get(physical.name, {})
        stage = Stage(physical.name, f"{physical.regime} {physical.kind}", price.get("core_h"),
                      price.get("basis", ""))
        out.append(stage)
        if trial is None:
            continue
        if physical.name in unwritten:
            stage.said.append(f"not written over this system: {unwritten[physical.name]}"
                              + ("; its overrides were not used"
                                 if overrides.get(physical.name) else ""))
            continue
        made = _prepared(trial, physical.name, common, overrides.get(physical.name, {}), stage)
        if stage.core_h:
            _scaled(stage, made)
    unknown = sorted(set(overrides) - {s.name for s in protocol.stages})
    if unknown and out:
        out[0].said.append(f"no stage {', '.join(unknown)}: their overrides were not used")
    return out


def _scaled(stage: Stage, made) -> None:
    """The stage's price for the length its input is set to, against the stage's own."""
    _, steps, own = made or (None, None, None)
    if steps and own and steps != own:
        stage.core_h = round(stage.core_h * steps / own, 3)
        stage.basis += f"; × {steps / own:.3g} for the length set here"


def _mandate(trial, protocol, files: dict[str, str], prices: dict[str, dict]
             ) -> dict[str, str]:
    """Write the protocol as the trial study's mandate over the study's files, its price of
    each stage taking the place of ``prices``'; stage name -> why it is not in it. Stages a
    QM/MM Hamiltonian or a QM region cannot be written for on this system are left out, so
    the classical ones are still tried."""
    stages = [s.model_dump(mode="json", exclude_none=True) for s in protocol.stages]
    call = {"question": f"preview of {protocol.id}", "ceiling_core_h": 1e9,
            "protocol": protocol.ref,
            "system_files": [{"path": files[k], "kind": k} for k in SYSTEM_KINDS if k in files],
            **({"hamiltonian": protocol.hamiltonian.model_dump(mode="json", exclude_none=True)}
               if protocol.hamiltonian is not None else {})}
    written = REGISTRY.call("mandate_write", trial, {**call, "stages": stages})
    if written.ok:
        prices.update(getattr(written, "stage_prices", None) or {})
        return {}
    why = written.error.message if written.error else written.summary
    classical = [s for s in stages if s.get("regime", "mm") == "mm"]
    if classical and len(classical) < len(stages):
        call.pop("hamiltonian", None)
        written = REGISTRY.call("mandate_write", trial, {**call, "stages": classical})
        if written.ok:
            prices.update(getattr(written, "stage_prices", None) or {})
            return {s["name"]: why for s in stages if s not in classical}
    return {s["name"]: why for s in stages}


def _prepared(trial, name: str, common: dict, overrides: dict, stage: Stage) -> tuple:
    """The stage prepared as stage_prepare compiles every file of it: (the run's ps and steps,
    the stage's own steps before any override), each None when unknown."""
    def length(obs) -> tuple:
        for kind, values in (getattr(obs, "values", None) or {}).items():
            ps, steps = run_length(kind, values)
            if ps or steps:
                return ps, steps
        return None, None

    own = (length(REGISTRY.call("stage_prepare", trial, {"stage": name, **common}))[1]
           if overrides else None)
    prepared = REGISTRY.call("stage_prepare", trial, {
        "stage": name, **common, **({"overrides": overrides} if overrides else {})})
    asked = [f for f in prepared.findings if f.id == "INPUTSET_REQUIRED_PARAM_MISSING"]
    if asked:  # the researcher's value, not a broken stage: tried once given
        stage.said += [f"needs, from the researcher: {f.message}" for f in asked]
        return None, None, None
    stage.rendered = prepared.ok or bool(getattr(prepared, "files", None))
    stage.built = prepared.ok if stage.rendered else None
    stage.said += [f.message for f in prepared.findings if f.severity != "info"][:4]
    if not prepared.ok:
        stage.said.append(prepared.error.message if prepared.error else prepared.summary)
        stage.said += [line for line in (prepared.log_tail or "").splitlines()
                       if line.strip()][-8:]
        return None, None, None
    ps, steps = length(prepared)
    return ps, steps, own if overrides else steps
