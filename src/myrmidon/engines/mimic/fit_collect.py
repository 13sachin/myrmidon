"""A parameter fit as a stage's job, and what it left: fm.log, optimization_results.txt and the
optimized .itp read into typed facts, and the candidate topology assembled from them."""

from __future__ import annotations

from pathlib import Path

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.observation import Observation
from myrmidon.core.tool import ToolContext
from myrmidon.core.validation import Finding, Precondition
from myrmidon.engines.gromacs.common import gmx_bin, stage_topology, tail
from myrmidon.engines.gromacs.itp import atom_charges
from myrmidon.engines.inputs import resolve_in_ws
from myrmidon.engines.mimic.fm import (
    FORCE_OUTLIER_FACTOR,
    FORCE_UNITS,
    PARAMETER_FINDINGS,
    VALIDATION_RESIDUALS_FILENAME,
    fitted_pairs,
    gmx_data_flags,
    implausible_parameters,
    includes_of,
    itp_atomtypes_warnings,
    parse_fm_log,
    parse_fminput,
    parse_optimization_results,
    parse_validation_force_residuals,
    rerun_mdp_warnings,
)
from myrmidon.engines.mimic.qm_region import mimicpy_bin
from myrmidon.engines.registry import Layout, StageJob
from myrmidon.execution.backend import JobSpec, JobState
from myrmidon.execution.collect import collector

#: the kinds a fit reads, none of which it can run without
FIT_INPUTS = ("topology", "structure", "reference_dataset", "trajectory", "index", "qm_selection",
              "fm_input", "mdp")


#: what mimicpy names a fitted file after the one it read: a fit's, a charges-only fit's
FITTED_PREFIXES = ("opt_", "resp_")


def stage_inputs(ctx: ToolContext, mandate, stage, prepared: dict, continuing: bool) -> dict:
    """What a fit reads besides the files its preparation rendered (its input and its classical
    rerun's): the system's files, the selection, and the latest reference dataset with the
    trajectory collected beside it."""
    from myrmidon.core.lineage import latest_dataset, system_file

    begins = mandate.method.starts_from(stage)

    out: dict[str, str] = {}
    # the selection of the run the fit reads, whose dataset holds those atoms
    reads = begins.name if begins is not None else stage.name
    for kind in ("topology", "structure", "index", "qm_selection"):
        named = system_file(ctx.workspace, mandate, kind, reads)
        if named and kind not in prepared:
            out[kind] = named
    dataset = latest_dataset(ctx.workspace, begins.name if begins is not None else None)
    if dataset:
        out["reference_dataset"] = dataset
        for ev in reversed(list(ctx.workspace.ledger.events())):
            if ev.ok and any(a.path == dataset for a in ev.outputs):
                found = next((a.path for a in ev.outputs if a.role == "trajectory"), None)
                if found:
                    out["trajectory"] = found
                break
    return out


def fm_dataset_frames(ctx: ToolContext, dataset: Path) -> int | None:
    """How many configurations a reference dataset holds; None when it cannot be counted."""
    from myrmidon.engines.mimic.fm import dataset_facts

    fact = dataset_facts(ctx, dataset).get("stride")
    return int(fact.value.strip("[]").split(",")[1]) if fact else None


def job_findings(ctx: ToolContext, stage, inputs: dict[str, Path], walltime_min: int,
                 extend_ps: float | None = None, continuing: bool = False) -> list[Finding]:
    """What stops a fit: an input it cannot run without, one its input file does not state, a
    topology mimicpy cannot rewrite."""
    missing = [k for k in FIT_INPUTS if k not in inputs]
    if missing:
        return [Precondition(
            id="STAGE_NOT_PREPARED",
            message=f"stage {stage.name} has no {', '.join(missing)}",
            suggestion="stage_prepare the fit, which renders its input and its classical "
            "rerun's; dataset_collect the reference run it fits against; qm_region writes the "
            "selection and the index")]
    findings = list(itp_atomtypes_warnings(inputs["topology"]))
    if "mdp" in inputs:
        from myrmidon.engines.gromacs.mdp import parse_mdp_file

        findings += rerun_mdp_warnings(parse_mdp_file(inputs["mdp"]))
    keys = parse_fminput(inputs["fm_input"])
    held = fm_dataset_frames(ctx, inputs["reference_dataset"])
    stride = str(keys.get("stride", "")).strip("[] ").split(",")
    if held and len(stride) == 3 and stride[1].strip().isdigit() and int(stride[1]) > held:
        findings.append(Precondition(
            id="FM_STRIDE_BEYOND_DATASET",
            message=f"the fit's stride ends at {stride[1].strip()}, and its dataset holds {held} "
            "configurations: re-collect, or narrow the stride"))
    for required in ("stride", "qm_total_charge"):
        if required not in keys:
            findings.append(Precondition(
                id="FM_INPUT_INCOMPLETE",
                message=f"fminput has no '{required}': stage_prepare writes it (explicit, never "
                "assumed)"))
    return findings


def layout_facts(_inputs, _stage=None) -> dict:
    return {}


def resolve_layout(cores: int, _facts) -> Layout:
    """One process forking a worker for each core: a fit's optimiser does not run under MPI."""
    return Layout(1, cores, why=f"one process of {cores} workers")


def size_walltime(ctx: ToolContext, stage, inputs: dict[str, Path], layout, cores: int,
                  extend_ps: float | None = None, continuing: bool = False,
                  frames: list | None = None
                  ) -> tuple[int | None, str]:
    """A fit's walltime, the template's."""
    from myrmidon.knowledge.stage import TemplateStore, templates_for

    fixed = next((t.pricing["walltime_min"] for t in templates_for(
        stage.model_dump(mode="json"), TemplateStore.default())
        if "walltime_min" in t.pricing), None)
    return fixed, f"{fixed} min, the template's for a parameter fit"


def build_job(ctx: ToolContext, stage, inputs: dict[str, Path], layout, request) -> StageJob:
    """``mimicpy fm`` in the stage's directory, on the system's selection and index."""
    run_dir = ctx.workspace.stage(stage.name)
    top = stage_topology(run_dir, inputs["topology"])
    argv = [mimicpy_bin(ctx), "fm", "-top", str(top), "-coords", str(inputs["structure"]),
            "-fmdata", str(inputs["reference_dataset"]), "-trr", str(inputs["trajectory"]),
            "-ndx", str(inputs["index"]), "-sele", str(inputs["qm_selection"]),
            "-fi", str(inputs["fm_input"]), "-gmx", gmx_bin(ctx),
            "-n_processes", str(layout.cpus_per_task), *gmx_data_flags(ctx)]
    argv += ["-mdp", str(inputs["mdp"])]
    flag = {"charges": "-dresp", "bonded": "-opt_ff"}.get(request.fit_mode)
    if flag:
        argv.append(flag)
    return StageJob(
        spec=JobSpec(argv=argv, cwd=run_dir, stdout="fm.stdout", stderr="fm.stderr",
                     resources=request.resources()),
        kind="parameter_fit", label=f"stage:{stage.name}",
        meta={"topology": ctx.workspace.relpath(inputs["topology"])},
        summary=f"fit of {'the ' + request.fit_mode if flag else 'charges and bonded terms'} "
        f"by {layout.cpus_per_task} workers in {ctx.workspace.relpath(run_dir)}")


def collect_fit(ctx: ToolContext, stage_dir: Path, original_itp: str | None = None,
                topology: str | None = None, *, completed: bool = True) -> Observation:
    """A force-matching fit that has ended, read into typed facts: snapshot counts,
    train/validation split, validation SDF/cost, optimizer status, wall time, the fitted
    charge deltas against the original .itp, and — given the topology the fit started
    from — the candidate topology with the fitted include(s) swapped in. This is what
    ``job_status`` returns for a fit's job; ``original_itp`` is inferred from the
    topology's includes when not given."""
    log_path = stage_dir / "fm.log"
    if not log_path.exists():
        return Observation(
            ok=False,
            summary=f"no fm.log at {ctx.workspace.relpath(log_path)} — has the fit run?",
        )
    facts = parse_fm_log(log_path.read_text(errors="replace"))
    results_file = log_path.parent / "optimization_results.txt"
    if results_file.exists():
        for key, value in parse_optimization_results(results_file.read_text()).items():
            facts.setdefault(key, value)

    artifacts = [ArtifactRef.from_path(log_path, "fm_log", root=ctx.workspace.root)]
    max_dq = mean_dq = n_q = None
    opt_itps: list[Path] = []
    for name in facts.get("itp_files", []):
        candidate = log_path.parent / name
        if candidate.exists() and name.startswith(FITTED_PREFIXES):
            opt_itps.append(candidate)
            artifacts.append(
                ArtifactRef.from_path(candidate, "fitted_topology", root=ctx.workspace.root)
            )
    top = resolve_in_ws(ctx, topology) if topology else None
    if original_itp is None and top is not None and opt_itps:
        for inc in fitted_pairs(top, opt_itps):
            candidate_orig = top.parent / inc
            if candidate_orig.exists():
                original_itp = ctx.workspace.relpath(candidate_orig)
                break
    opt_itp = None
    if original_itp and opt_itps:
        # the fitted twin of the original, else the only one
        original = resolve_in_ws(ctx, original_itp)
        twins = [log_path.parent / f"{prefix}{original.name}" for prefix in FITTED_PREFIXES]
        opt_itp = next((t for t in twins if t in opt_itps),
                       opt_itps[0] if len(opt_itps) == 1 else None)
    if original_itp and opt_itp is not None:
        before = dict(atom_charges(resolve_in_ws(ctx, original_itp).read_text()))
        after = dict(atom_charges(opt_itp.read_text()))
        common = sorted(set(before) & set(after))
        if common:
            deltas = [abs(after[a] - before[a]) for a in common]
            max_dq = round(max(deltas), 5)
            mean_dq = round(sum(deltas) / len(deltas), 5)
            n_q = len(common)

    forces: dict = {}
    residuals_csv = log_path.parent / VALIDATION_RESIDUALS_FILENAME
    if residuals_csv.exists():
        forces = parse_validation_force_residuals(residuals_csv.read_text(errors="replace"))
        if forces:
            artifacts.append(ArtifactRef.from_path(residuals_csv, "force_residuals",
                                                   root=ctx.workspace.root))

    ok = (completed and facts.get("completed", False)
          and facts.get("optimizer_success") is not False)
    warnings: list[Finding] = []
    if forces.get("outliers"):
        worst = forces["outliers"][:5]
        warnings.append(Finding(
            id="FM_FORCE_OUTLIER_ATOMS",
            message=f"{len(forces['outliers'])} QM atom(s) have a held-out force RMSE above "
            f"{FORCE_OUTLIER_FACTOR:g}× the median atom's ({forces['median_atom_rmse']} "
            f"{FORCE_UNITS}): "
            + ", ".join(f"{a['resname']}{a['resid']}:{a['name']} ({a['type']}) {a['rmse']}"
                        for a in worst)
            + (" ..." if len(forces["outliers"]) > 5 else "")
            + ". Look at these atoms' bonded terms and the QM/MM boundary before trusting "
            "the candidate.",
        ))
    # Every candidate, read against the topology it was fitted from where that can be
    # found, so a parameter that was replaced rather than refined is visible.
    pairs = fitted_pairs(top, opt_itps) if top is not None else {}
    by_finding: dict[str, list[str]] = {}
    for itp in opt_itps:
        source = next((top.parent / inc for inc, fitted in pairs.items()
                       if Path(fitted) == itp and (top.parent / inc).is_file()), None)
        rows_by_finding = implausible_parameters(
            itp.read_text(errors="replace"),
            source.read_text(errors="replace") if source is not None else None,
        )
        for finding, rows in rows_by_finding.items():
            by_finding.setdefault(finding, []).extend(f"{itp.name}: {r}" for r in rows)
    unphysical = by_finding.get("FM_UNPHYSICAL_PARAMS", [])
    for finding, rows in sorted(by_finding.items()):
        shown = "; ".join(rows[:3]) + (" ..." if len(rows) > 3 else "")
        warnings.append(Finding(
            id=finding,
            message=f"{len(rows)} fitted {PARAMETER_FINDINGS[finding]} ({shown})",
            suggestion="do not use this candidate as it stands: add reference snapshots, "
            "or restrict what the fit may change (charges only). Read the parameter "
            "tables in the study report — validation scores can look acceptable while the "
            "parameters are not.",
        ))
    bits = []
    if unphysical:
        bits.append(f"{len(unphysical)} UNPHYSICAL bonded term(s)")
    if facts.get("n_snapshots") is not None:
        bits.append(f"{facts['n_snapshots']} snapshots")
    if facts.get("n_train") is not None:
        bits.append(f"{facts['n_train']}/{facts['n_val']} train/val")
    if facts.get("validation_sdf") is not None:
        bits.append(f"validation SDF {facts['validation_sdf']}")
    if opt_itps:
        bits.append(f"fitted itp(s): {', '.join(p.name for p in opt_itps)}")
    if max_dq is not None:
        bits.append(f"max |dq| {max_dq}")
    if forces:
        worst_type = forces["by_type"][0]
        bits.append(
            f"held-out bonded-force RMSE {forces['rmse']} {FORCE_UNITS} over "
            f"{forces['n_frames']} frames "
            f"(rel {forces['rel_rmse']}; worst type {worst_type['type']} {worst_type['rmse']})"
        )
    summary = (
        ("FM fit completed: " if ok else "FM fit NOT clean: ")
        + (", ".join(bits) if bits else "no facts parsed from fm.log")
        + (
            f". Optimizer: {facts.get('optimizer_message')}"
            if facts.get("optimizer_message")
            else ""
        )
    )
    obs = Observation(
        ok=ok,
        summary=summary,
        artifacts=artifacts,
        findings=warnings,
        n_unphysical_terms=len(unphysical),
        log_tail=None if ok else tail(log_path.read_text(errors="replace"), 25),
        n_snapshots=facts.get("n_snapshots"),
        n_train=facts.get("n_train"),
        n_val=facts.get("n_val"),
        validation_sdf=facts.get("validation_sdf"),
        validation_cost=facts.get("validation_cost"),
        optimizer_success=facts.get("optimizer_success"),
        optimizer_message=facts.get("optimizer_message"),
        qm_total_charge=facts.get("qm_total_charge"),
        wall_time_s=facts.get("wall_time_s"),
        max_abs_charge_delta=max_dq,
        mean_abs_charge_delta=mean_dq,
        n_charges=n_q,
        force_rmse=forces.get("rmse"),
        force_ref_rms=forces.get("ref_rms"),
        force_rel_rmse=forces.get("rel_rmse"),
        force_rmse_by_type=forces.get("by_type", []),
        force_outliers=forces.get("outliers", []),
        n_force_frames=forces.get("n_frames"),
    )
    if original_itp:
        orig_path = resolve_in_ws(ctx, original_itp)
        if orig_path.exists():
            obs.inputs.append(ArtifactRef.from_path(orig_path, "original_topology",
                                                    root=ctx.workspace.root))
    # the candidate topology: the fit's own topology with every fitted include swapped in
    if ok and top is not None and opt_itps:
        pairs = fitted_pairs(top, opt_itps)
        if pairs:
            cand = assemble_candidate(ctx, stage_dir, top, pairs)
            if cand.ok:
                known = {a.path for a in obs.artifacts}
                obs.artifacts += [a for a in cand.artifacts if a.path not in known]
                obs.summary += f". Candidate topology: {cand.summary}"
            else:
                obs.summary += f". No candidate topology: {cand.summary}"
        else:
            obs.summary += (". No candidate topology: none of the fitted itp(s) is named "
                            "after an include of the topology")
    return obs


def assemble_candidate(ctx: ToolContext, stage: Path, top: Path, replace: dict[str, Path],
                       output: str = "candidate.top") -> Observation:
    """The candidate topology: ``top`` copied into ``stage`` with its same-directory
    includes, each include named in ``replace`` swapped for the fitted file. Done by the
    fit's collection (``job_status`` on a fit's job); the result is a topology a
    comparison builds its rerun and minimisations from."""
    import shutil

    staged = stage_topology(stage, top)
    text = staged.read_text(errors="replace")
    includes = includes_of(staged)
    missing = [k for k in replace if k not in includes]
    if missing:
        return Observation(
            ok=False,
            summary=f"{missing} are not #include lines of {ctx.workspace.relpath(top)}; it "
            f"includes {includes}",
        )
    swapped = []
    for inc, fitted in replace.items():
        src = Path(fitted)
        dst = stage / src.name
        if dst.resolve() != src.resolve():
            shutil.copy(src, dst)
        text = text.replace(f'#include "{inc}"', f'#include "{src.name}"')
        swapped.append(f"{inc} -> {src.name}")
    out = stage / output
    out.write_text(text)
    if staged != out and staged.name != out.name:
        staged.unlink()
    root = ctx.workspace.root
    return Observation(
        ok=True,
        summary=f"{ctx.workspace.relpath(out)} ({'; '.join(swapped)})",
        artifacts=[ArtifactRef.from_path(out, "topology", root=root)]
        + [ArtifactRef.from_path(stage / Path(f).name, "fitted_topology", root=root)
           for f in replace.values()],
    )


@collector("parameter_fit")
def _collect_fit_job(ctx: ToolContext, record, state, exit_code) -> Observation:
    meta = record.meta
    return collect_fit(
        ctx, Path(record.spec.cwd), original_itp=meta.get("original_itp"),
        topology=meta.get("topology"), completed=state == JobState.COMPLETED,
    )


def diagnose_log(ctx: ToolContext, path: Path) -> Observation:
    """A fit's log read as its collection reads it."""
    return collect_fit(ctx, path.parent)


def force_comparison(ctx: ToolContext, top: Path, structure: Path, forces: Path, dataset: Path,
                     selection: Path, frames: list[int], label: str = "") -> Observation:
    """How far one topology's forces, from a classical rerun of reference frames, are from the
    QM/MM reference on the QM atoms: per atom type and per atom, in atomic units, for any
    topology, stock or fitted, so a comparison of candidates is the same call on each."""
    import json

    from mimicpy.force_matching.qm_region import QMRegion
    from mimicpy.scripts.fmdata import FMDataset

    from myrmidon.core.artifacts import fresh_path
    from myrmidon.engines.mimic.fm import force_residual_rows

    label = label or top.stem
    info = ctx.engines.get("gromacs")
    gmxdata = info.data_dir if info is not None and info.data_dir else None
    qm_region = QMRegion(str(top), str(structure), gmxdata=gmxdata)
    qm_region.setup_qm_region(Path(selection))
    fmdata = FMDataset(str(dataset))
    if not frames:
        import MDAnalysis as mda

        frames = list(range(len(mda.Universe(str(structure), str(forces)).trajectory)))
    try:
        header, rows = force_residual_rows(forces, structure, fmdata, qm_region, frames)
    except ValueError as exc:
        return Observation(ok=False, summary=f"{label}: {exc}", topology=str(top), label=label)
    stage = ctx.workspace.stage("analysis")
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
