---
name: fm-reference
description: >-
  Turn frames of a QM/MM trajectory into a reference force dataset for force matching: pick
  the frames, render the CPMD and GROMACS halves once, run the single points, check that the
  wavefunction converged before the rest are submitted, and merge and register the dataset
  with its hashes. Use when the user asks for QM/MM reference data or reference forces, for
  single points over a trajectory, or for the expensive half of a force-matching study. Do
  not use to fit or validate parameters (fm-fit-and-validate), to build the classical box
  (setup-and-equilibrate), or to review a QM region before a dynamics run (qmmm-preflight).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "Make me QM/MM reference forces for these frames."
  myrmidon-tools: "study_open plan_write plan_check protocol_lookup inputset_lookup inputset_apply knowledge_search traj_frames gmx_grompp mimic_select_qm mimic_prepare mimic_run mimic_diagnose fm_merge fm_collect fm_frame_report fm_displace_plan job_status file_write"
  myrmidon-protocols: "fm_reference_singlepoint"
  myrmidon-inputsets: "fm_singlepoint_cpmd fm_singlepoint_mm mimic_qm_region"
---

## Scope

Frames in, a registered reference dataset out. This is the expensive half of force matching
and the half that is reusable: the dataset is cited by hash, and a later comparison of force
fields cites it rather than remaking it.

Not here: the fit and its validation (`fm-fit-and-validate`), the classical box
(`setup-and-equilibrate`), a dynamics preflight (`qmmm-preflight`). Sampling geometries away
from equilibrium is `fm_displace_plan`, a different request.

## The researcher decides

- **Which frames, and how many.** It is a sampling decision and a budget decision at once;
  it is recorded as the researcher's. `fm_singlepoint_cpmd` lists the QM total
  charge as `required_from_user`.
- **The QM selection, the total charge and the multiplicity** — as in `qmmm-preflight`, and
  confirmed the same way.
- **The budget.** One single point is cheap and the set is not; `COMPUTE_BUDGET` fires on
  the set.
- **What to do with a frame that fails.** Excluding it changes the dataset.

## Start from

`study_open`, then this skill, then `protocol_lookup` for `fm_reference_singlepoint`: its
three steps are `cpmd_input`, `mm_mdp` and `reference_run`, and its invariant
`FM_PACE_MATCHES_NSTXOUT` is the one that matters — CPMD writes one reference record per
pace step and the fit pairs it with the GROMACS frame of the same index, so a mismatch pairs
QM forces with the wrong geometry and nothing downstream notices.

Values: `fm_singlepoint_cpmd` and `fm_singlepoint_mm`, rendered **once** and reused for
every frame. A dynamics input set is not a single-point one. `mimic_qm_region` for the
region, whose padding is derived for it and judged by `QM_CELL_TIGHT`.

## Drive

`plan_write` before submitting anything: the frames, the per-frame cost, the total.

1. `traj_frames` over the trajectory the user named, each frame written as its own
   structure. `traj_frames.frames` is the manifest the rest of the stage is indexed by.
2. `inputset_apply` for `fm_singlepoint_cpmd` (with the QM total charge in `overrides`) and
   `fm_singlepoint_mm`. Once, not per frame.
3. Per frame: `mimic_select_qm` → `gmx_grompp` → `mimic_prepare` → `mimic_run`.
   Leave `maxsteps` unset. Under a wavefunction optimisation CPMD's limit bounds SCF
   iterations rather than steps, and `CPMD_SCF_LIMIT_TOO_LOW` is the finding that says a
   small one was given.
4. **The first frame decides whether the rest are submitted.** `mimic_diagnose` on it:
   - `mimic_diagnose.wavefunction_converged` — true, or the reference is not a reference.
   - `mimic_diagnose.gemax_last` against `.gemax_criterion`, and `.scf_iterations` against
     `.scf_max_iterations`.
   - `mimic_diagnose.energy_total` — the number the frame contributes.
5. Only then the remaining frames. `mimic_diagnose` each one as it lands.
6. `fm_merge` the per-frame records into one dataset, then `fm_collect` to register it:
   frame count, trajectory-to-dataset alignment, and hashes. Read
   `fm_collect.index_aligned` and `fm_collect.dropped` — a dataset whose QM atom ids do not
   match the trajectory's produces monitor distances that are wrong without being absurd
   (`fm_frame_report.monitor_pairs` says how the ids are ordered).

## Stop

- `wavefunction_converged` is false on the first frame: **stop**. Do not submit the rest.
  The run exits cleanly and writes a reference file that looks like a good one, which is why
  the check is before the fan-out and not after it.
- `mimic_diagnose` reports it could read nothing: the log is not what the stage assumed.
- A frame that produced no log at all: read `mimic_diagnose.cpmd_error`, which carries what
  CPMD wrote to its own error file, and stop on what it names. A job that died before it
  computed anything is a machine problem — memory, ranks, a scheduler limit — which
  `job_status` diagnoses from the scheduler and the log, and which the skill cannot name in
  advance.
- `FM_TRR_H5_DESYNC` or `FM_FRAMES_DROPPED` from `fm_collect`: the dataset is not the set
  that was planned. Report the difference before anything is fitted to it.

## Report

`plan_check`, then the dataset by role with its hash, the frame count planned against
collected, every excluded frame **with its reason**, the convergence numbers of the frame
that was checked first, and core-hours booked against consumed. Leave a failed frame's stage
in place: a failed attempt is part of the record.
