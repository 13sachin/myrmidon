---
name: diagnose-a-failed-run
description: >-
  Say why a simulation died, stopped early or produced numbers that cannot be right, and what
  has to change before it is resubmitted: a GROMACS or a MiMiC/CPMD run, tracked by the
  harness or only named by its directory. Use when the user says a run crashed, stopped,
  hung, blew up, was killed, or asks "what went wrong?" or "why did my job fail?". Do not
  use to check an input before anything ran (audit-run-parameters, qmmm-preflight), or to
  extend a run that finished cleanly (setup-and-equilibrate).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "My run stopped — what went wrong, and what do I change before I resubmit?"
  myrmidon-tools: "study_open study_create file_read job_status mimic_diagnose file_check knowledge_search checkpoint_continue"
  myrmidon-findings: "CPMD_MULTIPLICITY_WITHOUT_LSD CPMD_SEGFAULT CPMD_SCF_NOT_CONVERGED CPMD_SHAKE_WRONG_GEOMETRY CPMD_KILLED_OUT_OF_MEMORY CPMD_RESTART_FILE_MISSING SCHEDULER_TIME_LIMIT WALLTIME_INSUFFICIENT CPT_NAME_MISMATCH"
---

## Scope

A run that ended badly in, a named cause and the change it calls for out. The cause comes
from a parser, never from reading a log by eye: `job_status` diagnoses a tracked job from the
scheduler and the engine log, and `mimic_diagnose` parses a CPMD log and the error file CPMD
leaves beside it. Both name a stable id when they recognise the signature.

Not here: checking an input before it runs (`audit-run-parameters`, `qmmm-preflight`), or
continuing a run that ended cleanly (`setup-and-equilibrate`).

## The researcher decides

- **Whether to resubmit**, and with what budget.
- **Any change to the chemistry or the method** the cause points at: the charge, the
  multiplicity, the QM region, the timestep, the force field. You name the change and the
  evidence; they make it.
- **Whether a run that was cut short counts.** Its partial output is part of the record.

## Start from

`study_open` with no path: the study that ran the job, if there is one. A run the harness
never tracked, only named by its directory, is read from that directory inside a study
(`study_create` when none is open: the project's files stay readable from it). `file_read` on
the directory lists what the run left, which says which engine wrote it.

## Drive

1. **A tracked job:** `job_status` with its id. Read `job_status.state`,
   `job_status.diagnosis` and `job_status.diagnosis_id`, and the log tail it returns.
2. **A CPMD or MiMiC run:** `mimic_diagnose` on its `cpmd.out` (the error file beside it is
   read too). Read `mimic_diagnose.diagnosis_id`, `.cpmd_error`, `.finished_normally`,
   `.md_steps_completed`, and for a wavefunction optimisation `.wavefunction_converged`
   with `.scf_iterations` against `.scf_max_iterations`. For an NVE stage whose energy
   "cannot be right": `.eclassic_drift_ha_per_ps` against `.eclassic_fluctuation_ha`, read
   only when `.ion_temperature_control` is None; how much drift is too much is theirs.
3. **Name the cause by its id**, then find the input it lives in and `file_check` that file:
   its findings say which key is wrong and where the value came from. A cause in the method
   (`CPMD_MULTIPLICITY_WITHOUT_LSD`, `CPMD_SHAKE_WRONG_GEOMETRY`) is a change to an input; a
   cause in the machine (`CPMD_KILLED_OUT_OF_MEMORY`, `SCHEDULER_TIME_LIMIT`,
   `WALLTIME_INSUFFICIENT`) is a change to the resources, not the physics.
4. `knowledge_search` for the keyword or option the cause names, so the change is cited from
   the engine's documentation for the installed version.
5. A run cut by the time limit continues from its checkpoint (`checkpoint_continue`), not
   from the start; `CPT_NAME_MISMATCH` is the refusal that keeps a continuation honest. Only
   when the researcher asks.

Then stop and report. Resubmitting is a separate request.

## Stop

- No parser recognises the failure: say so, quote the lines the tool returned, and do not
  guess a cause. An unknown signature is itself the finding.
- The directory holds no log at all: the run never started computing. That is the scheduler
  or the launch, and `job_status` is the tool that sees it; without a job id, say what is
  missing.
- `CPMD_RESTART_FILE_MISSING`: the stage restarted from nothing. The earlier stage's restart
  is the input to find, not a keyword to delete.

## Report

The cause by id, the fields that show it, the input and key it lives in, and the change you
propose with its documentation citation. Then what the researcher has to decide, and what was
not checked. Say plainly when the cause is not known.
