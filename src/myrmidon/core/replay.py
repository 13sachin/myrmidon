"""Replay a study from its ledger alone — the reproducibility claim, as code.

``replay_script()`` renders the shell commands; this module re-runs the *tools*
with their recorded parameters (and acknowledgments) in a fresh workspace, so
the replay goes through the same gates and backend the original did. Job ids
are remapped in order of appearance; paths under the source root are rewritten;
stages keep their names so artifacts land where they did. At the end every
output artifact the source recorded is compared by hash with its twin.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.provenance import ProvenanceEvent
from myrmidon.core.tool import REGISTRY, ToolContext, params_schema_hash
from myrmidon.core.workspace import META_FILENAME, STATE_DIRNAME, Workspace
from myrmidon.errors import UnknownToolError

#: tools whose replay would be meaningless or harmful besides the navigation ones (which
#: declare it: `ToolSpec.navigation`): session bookkeeping, a look at a file, interactive
#: viewers, and tool names a ledger may carry that no registered tool answers to
SKIP_TOOLS = frozenset({
    "study_create", "study_open", "file_check", "viz_open", "viz_command",
    "lesson_record", "lesson_search",
})


def replays_nothing(tool: str) -> bool:
    """A call a replay leaves out, and a study report does not count as work."""
    if tool in SKIP_TOOLS:
        return True
    try:
        return REGISTRY.get(tool).navigation
    except UnknownToolError:
        return False
_JOB_RE = re.compile(r"^job-[0-9a-f]{8}$")
_STAGE_PREFIX_RE = re.compile(r"^(\d{2,}_)+")


class ReplayStep(BaseModel):
    event_id: str
    tool: str
    action: str  # "replayed" | "skipped: <why>"
    ok: bool | None = None
    summary: str = ""
    job_id: str | None = None  # new job id when the call submitted one


class ArtifactCheck(BaseModel):
    path: str  # source relpath
    role: str
    status: str  # identical | differs | missing
    replay_path: str | None = None


class SchemaDrift(BaseModel):
    """A tool whose params model changed since the ledger was written."""

    tool: str
    recorded: str
    current: str
    n_events: int


class ReplayReport(BaseModel):
    source: str
    target: str
    steps: list[ReplayStep] = Field(default_factory=list)
    artifacts: list[ArtifactCheck] = Field(default_factory=list)
    job_map: dict[str, str] = Field(default_factory=dict)
    schema_drift: list[SchemaDrift] = Field(default_factory=list)

    @property
    def n_replayed(self) -> int:
        return sum(1 for s in self.steps if s.action == "replayed")

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if s.action == "replayed" and s.ok is False)


def plan(events: list[ProvenanceEvent]) -> list[tuple[ProvenanceEvent, str]]:
    """What a replay would do with each event: the successful tool calls, in order."""
    out = []
    for ev in events:
        if ev.kind != "call":
            out.append((ev, f"skipped: {ev.kind}"))
        elif replays_nothing(ev.tool):
            out.append((ev, "skipped: session/knowledge/viewer tool"))
        elif not ev.ok:
            out.append((ev, "skipped: failed in the source (the retry is replayed)"))
        else:
            out.append((ev, "replay"))
    return out


def schema_drift(events: list[ProvenanceEvent]) -> list[SchemaDrift]:
    """Tools whose params model is not the one the ledger was written against.

    An event recorded before a params model changed may fail to validate, or —
    worse — may validate into something else. The replay reports the drift by tool
    rather than discovering it as a failure halfway through.
    """
    seen: dict[tuple[str, str], int] = {}
    for ev in events:
        if not ev.params_schema:  # an event that recorded no schema has nothing to compare
            continue
        seen[(ev.tool, ev.params_schema)] = seen.get((ev.tool, ev.params_schema), 0) + 1
    drift = []
    for (tool_name, recorded), n in seen.items():
        try:
            current = params_schema_hash(REGISTRY.get(tool_name).params_model)
        except UnknownToolError:
            current = "tool not registered"
        if current != recorded:
            drift.append(
                SchemaDrift(tool=tool_name, recorded=recorded, current=current, n_events=n)
            )
    return sorted(drift, key=lambda d: d.tool)


def stage_key(relpath: str) -> str:
    """'03_02_nvt/nvt.edr' and '02_nvt/nvt.edr' name the same artifact: strip the
    stage numbering an agent may have folded into a stage name."""
    parts = Path(relpath).parts
    if len(parts) < 2:
        return relpath
    return str(Path(_STAGE_PREFIX_RE.sub("", parts[0]), *parts[1:]))


def stage_map(src_stages: list[str], dst_stages: list[str]) -> dict[str, str]:
    """Source stage directory -> the twin's directory with the same name once the
    numbering (possibly doubled, '03_02_nvt') is stripped."""
    by_key = {stage_key(f"{st}/x").split("/")[0]: st for st in dst_stages}
    return {
        st: by_key[k] for st in src_stages
        if (k := stage_key(f"{st}/x").split("/")[0]) in by_key and by_key[k] != st
    }


def _rewrite(value, src_root: str, dst_root: str, job_map: dict[str, str],
             stages: dict[str, str] | None = None):
    if isinstance(value, str):
        if value in job_map:
            return job_map[value]
        value = value.replace(src_root, dst_root)
        if stages:
            parts = Path(value).parts
            if parts and parts[0] in stages:
                value = str(Path(stages[parts[0]], *parts[1:]))
            elif value in stages:
                value = stages[value]
        return value
    if isinstance(value, list):
        return [_rewrite(v, src_root, dst_root, job_map, stages) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite(v, src_root, dst_root, job_map, stages) for k, v in value.items()}
    return value


def _job_ids_in(params: dict) -> list[str]:
    found: list[str] = []

    def walk(v):
        if isinstance(v, str) and _JOB_RE.match(v):
            found.append(v)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    walk(params)
    return found


def _job_ran_in(stage: Path) -> bool:
    return any(stage.glob(".slurm-*")) or any(stage.glob("*.stdout"))


def planted_files(src: Workspace, events: list[ProvenanceEvent]) -> list[str]:
    """Files no tool created: the planted inputs a replay must start from.

    Everything a tool wrote is recorded as an output artifact and will be re-created;
    anything inside a stage where a job ran is that job's.

    A call that both consumes and produces the same path *modified* it rather than
    creating it — ``gmx_solvate`` and ``gmx_genion`` rewrite the topology in place — so
    that path still has to be planted, or the call cannot run in the twin and the study
    does not reproduce.
    """
    produced = {
        art.path
        for ev in events
        for art in ev.outputs
        if art.path not in {consumed.path for consumed in ev.inputs}
    }
    out = []
    for st in src.stages():
        stage = src.root / st
        if _job_ran_in(stage):
            continue
        for p in sorted(stage.rglob("*")):
            if p.is_file() or p.is_symlink():
                rel = str(p.relative_to(src.root))
                if rel not in produced:
                    out.append(rel)
    return out


def _wait_for(ctx: ToolContext, job_id: str, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        obs = REGISTRY.call("job_status", ctx, {"job_id": job_id, "wait_seconds": 60})
        if getattr(obs, "state", "") in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
            return
    raise TimeoutError(f"{job_id} did not finish within {timeout_s} s")


def replay(
    src: Workspace, ctx: ToolContext, target: Path, name: str | None = None,
    job_timeout_s: int = 7200,
) -> ReplayReport:
    """Reproduce ``src`` into a new workspace at ``target`` through the tools."""
    events = list(src.ledger.events())
    meta = src.meta if hasattr(src, "meta") else {}
    target = target.resolve()
    if (target / META_FILENAME).exists():
        raise FileExistsError(f"target workspace already exists: {target}")
    dst = Workspace.create(
        target, name or f"replay-of-{src.root.name}",
        f"replay of {src.root} from its ledger" + (f" ({meta.get('description')})" if meta else ""),
    )
    ctx.workspace = dst
    report = ReplayReport(
        source=str(src.root), target=str(dst.root), schema_drift=schema_drift(events)
    )

    for rel in planted_files(src, events):
        target_file = dst.root / rel
        target_file.parent.mkdir(parents=True, exist_ok=True)
        source_file = src.root / rel
        if source_file.is_symlink():
            target_file.symlink_to(os.readlink(source_file))
        else:
            shutil.copy2(source_file, target_file)

    src_root, dst_root = str(src.root.resolve()), str(dst.root)
    job_map: dict[str, str] = {}
    pending_old: list[str] = []  # source job ids seen, in order of first appearance
    new_jobs: list[str] = []
    for ev, action in plan(events):
        if action != "replay":
            report.steps.append(ReplayStep(event_id=ev.event_id, tool=ev.tool, action=action))
            continue
        for old in _job_ids_in(ev.params):
            if old not in pending_old:
                pending_old.append(old)
        # pair source job ids with replay submissions in order of appearance
        for i, old in enumerate(pending_old):
            if old not in job_map and i < len(new_jobs):
                job_map[old] = new_jobs[i]
        params = _rewrite(ev.params, src_root, dst_root, job_map,
                          stage_map(src.stages(), dst.stages()))
        unmapped = [j for j in _job_ids_in(params) if j not in new_jobs]
        if unmapped:
            report.steps.append(ReplayStep(
                event_id=ev.event_id, tool=ev.tool, ok=False,
                action="replayed", summary=f"job id(s) {unmapped} have no replay twin",
            ))
            continue
        # a replayed call that names a job waits for that job first: a collecting
        # job_status (a `call` event, unlike a poll, which is an `observe`) included
        for job_id in _job_ids_in(params):
            _wait_for(ctx, job_id, job_timeout_s)
        obs = REGISTRY.call(ev.tool, ctx, params)
        job_id = getattr(obs, "job_id", None) if obs.ok else None
        if job_id and _JOB_RE.match(job_id) and ev.tool != "job_status":
            new_jobs.append(job_id)
        report.steps.append(ReplayStep(
            event_id=ev.event_id, tool=ev.tool, action="replayed", ok=obs.ok,
            summary=obs.summary.splitlines()[0][:160], job_id=job_id,
        ))
    report.job_map = job_map

    # every artifact the source recorded, compared with its twin by hash
    seen: set[str] = set()
    twins = {stage_key(str(p.relative_to(dst.root))): p for p in dst.root.rglob("*") if p.is_file()}
    for ev in events:
        for art in ev.outputs:
            if art.path in seen or art.path.startswith(STATE_DIRNAME):
                continue
            seen.add(art.path)
            twin = twins.get(stage_key(art.path))
            if twin is None:
                status, rel = "missing", None
            else:
                rel = str(twin.relative_to(dst.root))
                status = (
                    "identical" if ArtifactRef.from_path(twin, art.role, root=dst.root).sha256
                    == art.sha256 else "differs"
                )
            report.artifacts.append(
                ArtifactCheck(path=art.path, role=art.role, status=status, replay_path=rel)
            )
    return report


def render(report: ReplayReport) -> str:
    lines = [f"replay of {report.source}", f"      into {report.target}", ""]
    for st in report.steps:
        if st.action != "replayed":
            lines.append(f"  -    {st.tool:<22} {st.action}")
        else:
            mark = "ok" if st.ok else "FAILED"
            job = f" -> {st.job_id}" if st.job_id else ""
            lines.append(f"  {mark:<4} {st.tool:<22} {st.summary}{job}")
    lines.append("")
    for drift in report.schema_drift:
        lines.append(
            f"  WARNING  {drift.tool}: params schema changed since this ledger "
            f"({drift.recorded} -> {drift.current}); {drift.n_events} event(s) were "
            "recorded against the old one"
        )
    if report.schema_drift:
        lines.append("")
    if report.job_map:
        lines.append("jobs: " + ", ".join(f"{a} -> {b}" for a, b in report.job_map.items()))
    counts = {}
    for a in report.artifacts:
        counts[a.status] = counts.get(a.status, 0) + 1
    lines.append(
        f"artifacts: {len(report.artifacts)} recorded — "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    )
    for a in report.artifacts:
        moved = a.replay_path and a.replay_path != a.path
        lines.append(f"  {a.status:<9} {a.role:<11} {a.path}"
                     + (f"  (replay: {a.replay_path})" if moved else ""))
    lines.append("")
    lines.append(
        f"{report.n_replayed} tool call(s) replayed, {report.n_failed} failed. "
        "Trajectories and energies from parallel MD are not bit-identical run to run; "
        "prep artifacts (topologies, run inputs) should be."
    )
    return "\n".join(lines)
