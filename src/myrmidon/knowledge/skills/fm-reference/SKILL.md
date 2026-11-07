---
name: fm-reference
description: >-
  Turn frames of a QM/MM trajectory into a reference force dataset for force matching: name
  the frames, prepare the stage once, run the first frame as a probe, check that its
  wavefunction converged, run the rest as one job, and merge and register the dataset with its
  hashes. Use when the user asks for QM/MM reference data or reference forces, for
  single points over a trajectory, or for the expensive half of a force-matching study. Do
  not use to fit or validate parameters (fm-fit-and-validate), to build the classical box
  (setup-and-equilibrate), or to review a QM region before a dynamics run (qmmm-preflight).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "Make me QM/MM reference forces for these frames."
  myrmidon-tools: "study_open protocol_lookup protocol_preview mandate_write mandate_show inputset_lookup inputset_apply knowledge_search snapshot_select stage_prepare stage_run run_diagnose dataset_collect job_status file_write"
  myrmidon-protocols: "fm_reference_singlepoint"
  myrmidon-stages: "cpmd.single_point gromacs.mimic"
---

## Scope

Frames in, a registered reference dataset out: the expensive, reusable half of force
matching, cited later by its hash. The sequence is `fm_reference_singlepoint`. Not here: the
fit (`fm-fit-and-validate`), the classical box (`setup-and-equilibrate`), a dynamics
preflight (`qmmm-preflight`); displaced geometries are `snapshot_select`.

## The researcher decides

Into `decisions_yours`, quoted: how many frames, as a count or a stride; the QM selection;
the QM method when they name one; what happens to a frame that fails; the ceiling. Into
`decisions_proposed`, with the source: the QM total charge `mandate_write` read from the
topology (an integer sum of the subsystems' charges), and the multiplicity the partition's
electrons decide (`derived_multiplicity`).
Yours, recorded: frames spread over the trajectory, the stage prepared once, the first frame
as the probe, the rest's resources from what the probe measured.

## Drive

1. `mandate_write` with `protocol=fm_reference_singlepoint` names the system (structure,
   topology) and the trajectory the frames are in; the reference stage's `scan` holds the
   frame indices the researcher chose, its `qm_region` the subsystems that are quantum (the
   charge is read from the topology and proposed, unless stated); the decisions; the ceiling.
   It prices the stage over every frame from the coupled steps measured here. The approval is
   the researcher's: `/approve` in Hermes, the dialog in Pi, or `myrmidon approve` at a
   terminal. Say the cost and the digest and end the turn.
2. `stage_prepare` with `stage=<the reference>` once: it partitions the region on the
   system's structure and writes the method's CPMD input and the compiled MM half; review its
   `partition`. Each frame is partitioned by prepqm on its own coordinates inside the job.
3. The probe: `stage_run` with `frames=[<the first frame>]`; when it lands, `run_diagnose`:
   `wavefunction_converged`, `gemax_last` against `gemax_criterion`, `scf_iterations` against
   `scf_max_iterations`, `energy_total`.
4. The rest: `stage_run` with no `frames`, one job over every frame no run has computed, its
   walltime from the probe's measured pace. Its collection says which frames finished.
5. `dataset_collect` with `stage=<the reference>` merges every frame's records and
   trajectories, in order; read `index_aligned` and `dropped`. `monitor_pairs` gives
   `monitor_stats` per pair.

## Stop

- `needs_input` from `mandate_write`, or a multiplicity `qm_region` leaves open: ask.
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
