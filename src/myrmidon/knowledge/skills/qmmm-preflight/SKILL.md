---
name: qmmm-preflight
description: >-
  Check a prepared QM/MM system before any MiMiC core-hour is spent: the QM region and its
  boundary, the total charge and the multiplicity, the vacuum around the QM cell, the
  pseudopotentials, and the review surface mimic_prepare returns. Use when the user asks
  whether a QM/MM input is right, wants a QM region picked or reviewed, asks what MiMiC
  needs before it will run, or is about to launch a coupled run. Do not use to build and
  equilibrate the classical box first (setup-and-equilibrate), to make reference forces for
  a fit (fm-reference), to fit a force field (fm-fit-and-validate), or to explain a run
  that failed (diagnose-a-failed-run).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "Is my QM/MM input right before I spend the allocation?"
  myrmidon-tools: "study_open protocol_lookup protocol_preview mandate_write mandate_show inputset_lookup inputset_apply knowledge_search tool_lookup qm_region mimic_prepare gmx_grompp mimic_run mimic_diagnose job_status"
  myrmidon-protocols: "mimic_qmmm_solvated_solute"
  myrmidon-stages: "cpmd.minimize cpmd.dynamics cpmd.single_point gromacs.mimic"
---

## Scope

Everything between an equilibrated classical box and the first coupled run: the
`qm_region` and `mimic_prep` steps of `mimic_qmmm_solvated_solute`. Not here: building the
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

Into `decisions_proposed`, with the source: the charge and multiplicity `mimic_prepare`
derived (its `facts` say from what), the QM method the protocol's `hamiltonian` names, the
padding around the QM atoms (the stage's `qm_region.padding_nm`). Yours, recorded: the order
the protocol states, the stage that renders the CPMD template.

## Drive

The preflight runs in the session and spends nothing:

1. `mandate_write` names the structure and topology as the system; the Hamiltonian
   (`hamiltonian.qm` with the method and any cutoff to converge, `hamiltonian.coupling`);
   the coupled stages in physical terms, each with `regime: qmmm` and its `qm_region`
   (subsystems, the QM subsystem's topology charge, multiplicity, padding), a cutoff scan as
   a stage with a `scan`; the decisions, locking the lengths the researcher stated; the
   deliverables; the ceiling. It compiles and prices each stage. Its `ready` is not yet the
   one to approve: the draft renders, and nothing books until step 6's is approved.
2. `inputset_apply` with `stage=<the first coupled stage>` and an `.inp` output: the CPMD
   template.
3. `qm_region` with the topology, the coordinates, the researcher's `selections`, the
   stage's padding as `padding_nm` and that `template`. It returns the index with the QM
   group, the CPMD input, and the review surface: `qm_atoms` (count and order against what
   the researcher described), `declared_charge`, `min_vacuum_bohr` and `qm_cell_bohr`.
4. `inputset_apply` with the same stage, an `.mdp` output and the index from step 3, then
   `gmx_grompp` with that index.
5. `mimic_prepare` with the CPMD input from step 3. Leave the charge and multiplicity unset
   when the system decides them: it derives both and says how in `facts`. Read `charge`,
   `multiplicity`, `functional`, `sections` and `paths` in full.
6. `mandate_write` again, the QM index among the system files and each `qm_region` with the
   charge and multiplicity `mimic_prepare` derived, proposed with its `facts` as the source:
   the draft is written in place, and the approval makes the region's chemistry the
   researcher's.
   The approval is the researcher's: `/approve` in Hermes, the dialog in Pi, or
   `myrmidon approve` at a terminal.

Per stage: `inputset_apply` with `stage=<name>` for the `.inp` (the method `mimic_prepare`
takes) and the `.mdp`; each later stage names the earlier one's restart with
`restart_from`.

## Stop

- `mimic_prepare` returns `needs_input`: ask its question; the answer goes into
  `decisions_yours`.
- A selection prepqm refuses: report the commands it names; never shrink the region.
- A boundary the selection cuts: say what it cuts and hand the decision back.
- `mandate_write` returns `needs_input` (ask); step 6's `ready`: say the cost and the
  digest and end the turn.
- After a submission, or one the gate did not make: say what it said and stop.

## Report

`mandate_show` and the deliverables it names; the selection as given and the atom count
parsed, the charge and multiplicity with who decided each, the vacuum, the
pseudopotentials, the CPMD sections `mimic_prepare` assembled, what was not checked, and
what the first coupled run would book.
