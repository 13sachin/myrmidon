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
  myrmidon-tools: "study_open protocol_lookup protocol_preview mandate_write mandate_show inputset_lookup inputset_apply knowledge_search traj_frames gmx_grompp qm_region mimic_prepare mimic_run run_diagnose dataset_collect traj_concat dataset_collect dataset_collect fm_displace_plan job_status file_write"
  myrmidon-protocols: "fm_reference_singlepoint"
  myrmidon-stages: "cpmd.single_point gromacs.mimic"
---

## Scope

Frames in, a registered reference dataset out: the expensive, reusable half of force
matching, cited later by its hash. The sequence is `fm_reference_singlepoint`. Not here: the
fit (`fm-fit-and-validate`), the classical box (`setup-and-equilibrate`), a dynamics
preflight (`qmmm-preflight`); displaced geometries are `fm_displace_plan`.

## The researcher decides

Into `decisions_yours`, quoted: how many frames, as a count or a stride; the QM selection;
the QM method when they name one; what happens to a frame that fails; the ceiling. Into
`decisions_proposed`, with the source: the QM total charge the probe's `qm_region`
declares (an integer sum of the selection's topology charges), and the multiplicity
`mimic_prepare` derives.
Yours, recorded: frames spread over the trajectory, the stage rendered once per engine, the
first frame as the probe, the rest's resources from what the probe measured.

## Drive

1. `traj_frames` over the named trajectory, one structure per frame; `frames` is the
   manifest the rest is indexed by.
2. `qm_region` on the first frame, the probe: the index holding the QM group, and the
   QM total charge it declares.
3. `mandate_write` names the system, the probe's index with it, and the protocol; the
   reference as a `single_point` stage with `regime: qmmm`, its `qm_region` (subsystems, the
   probe's charge, multiplicity) and the frame count as its `steps`, the charge proposed
   with the probe as its source; the decisions; the ceiling. It prices the stage from the coupled steps measured
   here. The approval is the researcher's: `/approve` in Hermes, the dialog in Pi, or
   `myrmidon approve` at a terminal. Say the cost and the digest and end the turn.
4. `inputset_apply` with `stage=<the reference>` once with an `.inp` output and once with an
   `.mdp` output and the probe's index; the protocol keeps the CPMD and GROMACS outputs
   paired.
5. The probe alone: `gmx_grompp` → `mimic_prepare` with `method` the rendered `.inp` (its
   wavefunction optimisation and force output) →
   `mimic_run`; when it lands,
   `run_diagnose`: `wavefunction_converged`, `gemax_last` against `gemax_criterion`,
   `scf_iterations` against `scf_max_iterations`, `energy_total`.
6. The remaining frames the same way, `qm_region` first, `run_diagnose` on each as
   it lands.
7. `dataset_collect` the records and `traj_concat` the trajectories in the same order, then
   `dataset_collect`; read `index_aligned` and `dropped`. `dataset_collect` with `monitor_pairs`
   gives `monitor_stats` per pair.

## Stop

- `needs_input` from `mimic_prepare` or `mandate_write`: ask.
- After each submission: the job id and the core-hours, then stop.
- The probe's wavefunction did not converge: submit nothing more.
- No log, or nothing `run_diagnose` can read: report `cpmd_error` and the scheduler's
  word from `job_status`; do not guess.
- A submission the gate did not make: say what it named.
- Dropped or misaligned frames: report them before anything is fitted.

## Report

`mandate_show`, then the deliverables the mandate names: the dataset with its hash, frames
asked for against collected, each excluded frame with its reason, the probe's convergence
numbers, core-hours booked against consumed.
