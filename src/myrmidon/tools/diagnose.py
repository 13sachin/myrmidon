"""What a run's log says, whichever engine wrote it."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.contracts import Contract
from myrmidon.core.lineage import stage_left, stage_segments
from myrmidon.core.observation import Observation
from myrmidon.core.tool import FileReader, ToolContext, tool
from myrmidon.engines.facts import is_run_log
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.registry import adapter_of_file, adapters

#: what an engine calls the account it writes, in the order a run's are read
_LOG_ROLES = ("qm_log", "fm_log", "log")


class RunDiagnoseParams(BaseModel):
    stage: str = Field(default="", description="a stage: its latest run, as the ledger knows it, "
                       "else the latest log under its directory")
    log: str = Field(default="", description="a run's log to read, in place of a stage's: a "
                     "running job's, say")


def _located(ctx: ToolContext, params: RunDiagnoseParams) -> tuple[Path | None, str]:
    """(the log to read, how it was found); (None, why) when there is none."""
    if params.log:
        return resolve_in_ws(ctx, params.log), ""
    left = {a.role: a.path for a in stage_left(ctx.workspace, params.stage)}
    for role in _LOG_ROLES:
        if role in left:
            return ctx.workspace.resolve(left[role]), ""
    where = ctx.workspace.stage(params.stage or "qmmm")
    logs = sorted((p for p in where.rglob("*") if p.suffix in (".out", ".log")
                   and (is_run_log(p) or any(p.name in a.log_names for a in adapters()))),
                  key=lambda p: p.stat().st_mtime)
    if not logs:
        return None, f"no engine log under {ctx.workspace.relpath(where)}: pass log explicitly"
    # a coupled run's directory holds both engines' logs: its account is its main program's
    beside = [p for p in logs if p.parent == logs[-1].parent]
    chosen = next((p for engine in adapters() for p in beside
                   if p.name in engine.log_names or (engine.is_log and engine.is_log(p))),
                  logs[-1])
    return chosen, (f"{ctx.workspace.relpath(chosen)}, the latest of {len(logs)} log(s) in "
                    f"{ctx.workspace.relpath(where)}. ")


@tool(
    "run_diagnose",
    description="Read a run's log, whichever engine wrote it, into typed facts: whether it "
    "finished, steps, convergence, temperatures, drift, a fit's health. ok=False when it did not "
    "finish or converge, with the log's last lines to read the cause from.",
    tags={"analysis"},
    contract=Contract(consumes={"log": "log"}),
    reads=(FileReader("log", is_run_log, "a run's log"),),
)
def run_diagnose(ctx: ToolContext, params: RunDiagnoseParams) -> Observation:
    path, said = _located(ctx, params)
    if path is None:
        return Observation(ok=False, summary=said)
    if not path.is_file():
        return Observation(ok=False, summary=f"no log at {ctx.workspace.relpath(path)}")
    engine = adapter_of_file(path)
    if engine is None or engine.diagnose is None:
        return Observation(ok=False, summary=f"{ctx.workspace.relpath(path)} is no log an engine "
                           "here wrote: nothing could be read from it")
    obs = engine.diagnose(ctx, path)
    obs.summary = said + obs.summary
    if params.stage and not params.log:
        _over_segments(ctx, params.stage, obs)
    return obs


def _over_segments(ctx: ToolContext, stage: str, obs: Observation) -> None:
    """A continued stage's run as a whole: the MD steps its segments' logs state, summed over
    the logs each wrote in a directory of its own (a log a continuation appends to holds all
    of them already). A segment stopped after its last restart point is counted for every step
    it ran, which the next one runs again."""
    logs: dict[str, None] = {}
    for _, written, _ in stage_segments(ctx.workspace, stage):
        found = {a.role: a.path for a in written}
        log = next((found[r] for r in _LOG_ROLES if r in found), None)
        if log:
            logs[log] = None
    if len(logs) < 2:
        return
    steps = 0
    for log in logs:
        path = ctx.workspace.resolve(log)
        engine = adapter_of_file(path) if path.is_file() else None
        if engine is None or engine.run_facts is None:
            return
        steps += engine.run_facts(path.read_text(errors="replace")).get("md_steps_completed") or 0
    obs.segments = len(logs)
    obs.stage_md_steps = steps
    obs.summary += f"; stage {stage}: {steps} MD steps over its {len(logs)} segments"
