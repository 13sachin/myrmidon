---
name: qmmm-preflight
description: >-
  Check a prepared QM/MM system before any MiMiC core-hour is spent: the QM region and its
  boundary, the total charge and the multiplicity, the vacuum around the QM cell, the
  pseudopotentials, and the review surface qm_region returns. Use when the user asks
  whether a QM/MM input is right, wants a QM region picked or reviewed, asks what MiMiC
  needs before it will run, or is about to launch a coupled run. Do not use to build and
  equilibrate the classical box first (setup-and-equilibrate), to make reference forces for
  a fit (fm-reference), to fit a force field (fm-fit-and-validate), or to explain a run
  that failed (diagnose-a-failed-run).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "Is my QM/MM input right before I spend the allocation?"
  myrmidon-tools: "study_open protocol_lookup protocol_preview mandate_write mandate_show inputset_lookup inputset_apply knowledge_search tool_lookup qm_region stage_prepare stage_run run_diagnose observable_measure job_status"
  myrmidon-protocols: "mimic_qmmm_solvated_solute"
  myrmidon-stages: "cpmd.minimize cpmd.dynamics cpmd.single_point gromacs.mimic"
---

## Scope

Everything between an equilibrated classical box and the first coupled run: the QM region
and the coupled stages of `mimic_qmmm_solvated_solute`. Not here: building the
box (`setup-and-equilibrate`), reference forces (`fm-reference`), the fit
(`fm-fit-and-validate`), a failed run (`diagnose-a-failed-run`). When `qm_region` is
not among your tools, `tool_lookup` names the engine it needs: say so and stop.

## The researcher decides

Into `decisions_yours`, quoted:

- which atoms are quantum: `qm_region`'s `selections`, naming this system's own
  residues and atoms;
- whether a selection that cuts a covalent bond is acceptable, and its boundary treatment
  (`add-bound` marks a boundary atom; the harness places no link atom);
- the QM method when the question names one, and the ceiling.

Into `decisions_proposed`, with the source: the charge and multiplicity `qm_region`
derived (`declared_charge`, `derived_multiplicity`, why in its sidecar), the QM method the protocol's `hamiltonian` names, the
padding around the QM atoms (the stage's `qm_region.padding_nm`). Yours, recorded: the order
the protocol states.

## Drive

The preflight runs in the session and spends nothing:

1. `qm_region` with the topology, the coordinates, the researcher's `selections`, the padding
   as `padding_nm` and `pp_info`. It writes the system's CPMD input, the index with the QM
   group and the selection file, and returns the review surface: `qm_atoms` (count and order
   against what the researcher described), `declared_charge`, `derived_multiplicity`,
   `electrons`, `guessed_elements` (an element misread goes back through `atomtypes_map`),
   `min_vacuum_bohr` and `qm_cell_bohr`.
2. `mandate_write` names the structure and topology as the system, and what the selection wrote
   joins them; the Hamiltonian (`hamiltonian.qm` with the method and any cutoff to converge,
   `hamiltonian.coupling`); the coupled stages in physical terms, each with `regime: qmmm`, its
   `starts_from` when it branches, and its `qm_region` (subsystems, the charge and multiplicity
   step 1 derived, proposed with its source), a cutoff scan as a stage with a `scan`; the
   decisions, locking the lengths the researcher stated; the deliverables; the ceiling. It
   compiles and prices each stage, a first coupled stage from the machine's reference step.
3. `stage_prepare` with `stage=<the first coupled stage>`: the complete run input and its
   compiled MM half. Read its findings and `file_read` the input in full.
   The approval is the researcher's: `/approve` in Hermes, the dialog in Pi, or
   `myrmidon approve` at a terminal.

Per stage, under the approved mandate: `stage_prepare`, then `stage_run`; each finds what the
stage it starts from left in the ledger. `observable_measure` with `deliverable=<name>` measures
each deliverable; `snapshot_select` picks a frame for a later stage to start from.

## Stop

- `qm_region` leaves the charge or the multiplicity open: ask; the answer goes into
  `decisions_yours`, and the stage's `qm_region` states it.
- A selection prepqm refuses: report the commands it names; never shrink the region.
- A boundary the selection cuts: say what it cuts and hand the decision back.
- `mandate_write` returns `needs_input` (ask); step 6's `ready`: say the cost and the
  digest and end the turn.
- After a submission, or one the gate did not make: say what it said and stop.

## Report

`mandate_show` and the deliverables it names; the selection as given and the atom count
parsed, the charge and multiplicity with who decided each, the vacuum, the
pseudopotentials, the CPMD sections `stage_prepare` assembled, what was not checked, and
what the first coupled run would book.
