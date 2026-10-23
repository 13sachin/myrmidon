"""The acetone reference campaign: three questions run through the loop.

One study per question, built from the plan files beside this script, approved once, and
advanced by `myrmidon tick` until the plan is done or a branch waits on a person. On the
mock engines it runs in a minute and leaves the ledger, the spool, the journal, the inbox
and the report to read; on real engines it is the tutorial system's own campaign
(docs/TUTORIAL.md).

    python tests/reference_campaign/reference_campaign.py --mock ROOT [q1|q2|q3 ...]

`tests/unit/test_reference_campaign.py` runs the same three plans through the same
functions, so the example and the test cannot drift apart.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
#: the tutorial system's entry files, as the test fixtures ship them
FIXTURES = HERE.parent / "fixtures" / "acetone"
QUESTIONS = ("q1_repeat_until", "q2_chain", "q3_sweep")


def plan_params(which: str) -> dict:
    """The plan_write parameters of one question, as the file states them."""
    name = next(q for q in QUESTIONS if q.startswith(which))
    return yaml.safe_load((HERE / f"{name}.yaml").read_text())


def stage_inputs(ws, which: str) -> None:
    """What the researcher brings, under 01_input/ of the study."""
    stage = ws.stage("input")
    shutil.copy(FIXTURES / "acetone_stageA_entry.gro", stage / "conf.gro")
    shutil.copy(FIXTURES / "acetone_stageA_entry.top", stage / "topol.top")
    shutil.copy(FIXTURES / "acetone.itp", stage / "acetone.itp")
    if which.startswith("q3"):
        for i in (1, 2, 3):  # the frames sampled from Q2's production trajectory
            shutil.copy(FIXTURES / "acetone_stageA_entry.gro", stage / f"frame_{i}.gro")
        # the MiMiC-ready tpr Q2's mimic_prep step made; on the mocks a stand-in
        (stage / "mimic.tpr").write_text("MOCKTPR\n")


def write_and_approve(ctx, which: str, approver: str = "researcher"):
    """plan_write from the file, then plan_approve acknowledging the budget — the one
    approval the loop runs under. Returns the plan observation."""
    from myrmidon.core.tool import REGISTRY

    written = REGISTRY.call("plan_write", ctx, plan_params(which))
    if not written.ok:
        raise RuntimeError(f"plan_write: {written.summary}")
    if written.unresolved:
        raise RuntimeError("the plan is not executable: "
                           + "; ".join(f"{u.step}.{u.key}" for u in written.unresolved))
    approved = REGISTRY.call("plan_approve", ctx, {
        "digest": written.digest, "acknowledge_warnings": ["COMPUTE_BUDGET"]})
    if not approved.ok:
        raise RuntimeError(f"plan_approve: {approved.summary}")
    return written


def run_until_settled(ctx, max_ticks: int = 80, pause_s: float = 0.4, wake=None,
                      echo=None):
    """Tick until the plan is done, or every branch waits on a person, or nothing moves.
    Returns the last report."""
    from myrmidon.core.loop import render, tick

    report = None
    for _ in range(max_ticks):
        report = tick(ctx, wake=wake)
        if echo:
            echo(render(report))
        if report.blocked or report.done:
            break
        if report.idle and not report.running:
            break
        time.sleep(pause_s)
    return report


def write_report(ctx, narrative: dict | None = None):
    from myrmidon.core.tool import REGISTRY

    return REGISTRY.call("study_report", ctx, {"narrative": narrative or {}})


def _mock_context(root: Path, name: str):
    """A study on the mock engines and the mock scheduler, the way `myrmidon run --mock`
    builds one."""
    from myrmidon.config import MyrmidonConfig
    from myrmidon.core.tool import REGISTRY
    from myrmidon.engines import mock
    from myrmidon.server.mcp_server import build_context

    mock.enter(root)
    ctx = mock.mock_context(build_context(MyrmidonConfig(studies_root=root)))
    obs = REGISTRY.call("study_create", ctx, {"name": name,
                                              "description": f"reference campaign {name}"})
    if not obs.ok:
        raise RuntimeError(obs.summary)
    return ctx


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="where the studies go")
    parser.add_argument("questions", nargs="*", default=list(QUESTIONS))
    parser.add_argument("--mock", action="store_true", help="mock engines and scheduler")
    args = parser.parse_args(argv)
    if not args.mock:
        print("real engines: build the study with `myrmidon study create`, write the plan "
              "from the file with `myrmidon call plan_write`, approve it, and run "
              "`myrmidon tick --follow 600`; this script drives the mocks only")
        return 2
    for which in args.questions:
        name = next(q for q in QUESTIONS if q.startswith(which))
        print(f"== {name}")
        ctx = _mock_context(args.root.resolve(), name)
        stage_inputs(ctx.workspace, name)
        write_and_approve(ctx, name)
        report = run_until_settled(ctx, echo=print)
        done = write_report(ctx)
        print(done.summary)
        print("done" if report and report.done else "waiting: `myrmidon inbox --study "
              f"{ctx.workspace.root}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
