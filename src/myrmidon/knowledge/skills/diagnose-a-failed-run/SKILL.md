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
  myrmidon-tools: "study_open study_create file_read job_status mimic_diagnose file_check knowledge_search checkpoint_continue mimic_run journal_write mandate_write mandate_show"
---

## Scope

A run that ended badly in, a named cause and the change it calls for out. The tools return
the evidence and its findings; the reading is yours. Not here: an input before it runs
(`audit-run-parameters`, `qmmm-preflight`), continuing a clean run
(`setup-and-equilibrate`).

## The researcher decides

Whether to resubmit and with what budget; any change to the chemistry or the method the
cause points at; whether a run cut short counts. Under an approved mandate these are its
`decisions_yours`. Yours: the cause, with the line that shows it; a resource change sized
from the log and the scheduler; a method change proposed with its citation; the diagnosis,
or a cause ruled out, in `journal_write`.

## Drive

1. `study_open` with no path finds the study that ran the job. An untracked run is read
   from its directory inside a study (`study_create` when none is open); `file_read` lists
   what it left.
2. A tracked job: `job_status` with its id; read `state`, `exit_code`, `diagnosis`,
   `log_tail`.
3. A CPMD or MiMiC run: `mimic_diagnose` on its `cpmd.out`; read `log_tail`, `cpmd_error`,
   `finished_normally`, `md_steps_completed` against `md_steps_requested`, and for a
   wavefunction optimisation `wavefunction_converged` and `scf_iterations` against
   `scf_max_iterations`. For an NVE stage without ion temperature control,
   `eclassic_drift_ha_per_ps` against `eclassic_fluctuation_ha`; how much is too much is the
   researcher's.
4. Name the cause, quoting the line, and `file_check` the input it lives in. A cause in the
   method changes an input; a cause in the machine changes the resources.
5. `knowledge_search` for the keyword the cause names, cited for the installed version.
6. A run cut by the time limit continues, when asked: a GROMACS run with
   `checkpoint_continue`, a coupled run through `mimic_run` with `restart_from`.

## Stop

- The lines do not settle the cause: quote them and name what would (a longer tail with
  `file_read`, the scheduler's accounting, an identical rerun). Do not guess.
- No log at all: the run never computed; `job_status` sees the scheduler's side.
- A fix that changes a researcher's decision, the system, or exceeds the ceiling:
  `mandate_write` with `revise=true` and the reason, approved with `/approve` in Hermes,
  the dialog in Pi, or `myrmidon approve` at a terminal; say the digest and end the turn.
- A resubmission the gate did not make: say what it named.

## Report

The cause with the line that shows it, the input and key it lives in, the change with its
citation, and under a mandate what `mandate_show` says remains of the ceiling. Then what the
researcher decides, what was not checked, and plainly when the cause is not known.
