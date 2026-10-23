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
  myrmidon-tools: "study_open plan_write plan_check protocol_lookup inputset_lookup inputset_apply knowledge_search mimic_select_qm mimic_prepare gmx_grompp mimic_run mimic_diagnose job_status"
  myrmidon-protocols: "mimic_qmmm_solvated_solute"
  myrmidon-inputsets: "mimic_qm_region mimic_cpmd_template mimic_mm_mdp"
  myrmidon-findings: "CPMD_MULTIPLICITY_WITHOUT_LSD QM_CELL_NO_VACUUM CPMD_SHAKE_WRONG_GEOMETRY CPMD_RESTRAINTS_UNDER_MIMIC QM_CHARGE_MULTIPLICITY_UNCONFIRMED QM_SELECTION_COMMAND_UNKNOWN QM_SELECTION_REJECTED QM_CHARGE_ROUNDED"
---

## Scope

Everything between an equilibrated classical box and the first coupled run, and nothing
after it. This is the question worth asking once: a QM/MM run that is wrong in the charge,
the multiplicity, the cell or the atom order still starts, still bills, and answers with
numbers that look like numbers.

Not here: building the box (`setup-and-equilibrate`), reference forces (`fm-reference`), the
fit (`fm-fit-and-validate`), a run that already failed (`diagnose-a-failed-run`). Launching
is `mimic_run`, a separate request; a later stage names the earlier one's restart with
`restart_from`, which records the hand-off. Nothing measures energy drift: whether an NVE
check conserved energy is the researcher's call on what `mimic_diagnose` parsed.

If `mimic_select_qm` is not among your tools this machine lacks the engine it needs
(mimicpy, or a MiMiC-enabled CPMD): name it and let the user install it or move to a
machine that has it, rather than working around it.

## The researcher decides

- **Which atoms are quantum.** `mimic_qm_region` lists `selections` as
  `required_from_user` because it is chemistry, and no default is safe. Selection commands
  name this system's own residues and atoms; nothing from another system carries over.
- **The total charge and the multiplicity.** `mimic_select_qm` reports a charge it derived
  and marks it unconfirmed on purpose; `mimic_prepare` requires both explicitly. Confirm
  against the selection's own report and against what the user said, and never infer one
  from the other.
- **Whether the boundary is acceptable** when the selection cuts a bond.
- **The budget**, when `COMPUTE_BUDGET` fires.

## Start from

`study_open`, then this skill. `protocol_lookup` for a sequence whose `system_class` fits:
in `mimic_qmmm_solvated_solute` the `qm_region` and `mimic_prep` steps are this check, and
the `annealing`, `nve_test`, `heating` and `production` steps are what it protects.

Values: `mimic_qm_region` names the selection, which it refuses to supply, and the padding,
which it says how to derive for this region; it renders no file, so `inputset_apply` with no
`output` resolves it. `mimic_cpmd_template` for the CPMD side, `mimic_mm_mdp` for the GROMACS
side. Read each set's `system_class` against the system in hand: a set for a closed-shell
region of light elements is not one for an open-shell or metal-containing region, and
`CPMD_MULTIPLICITY_WITHOUT_LSD` is what a wrong multiplicity costs.

## Drive

`plan_write` first when the answer will be followed by a run. A preflight that stops at the
answer spends nothing and needs no plan.

1. `inputset_apply` with no `output` for the QM region's values; supply `selection` in
   `overrides`, which records it as the user's.
2. `mimic_select_qm`. A command prepqm cannot apply is a refusal, not a smaller region
   (`QM_SELECTION_COMMAND_UNKNOWN`, `QM_SELECTION_REJECTED` names each one). The
   observation is the first review surface:
   - `mimic_select_qm.qm_atoms` — the atom count and order against what the user described.
   - `mimic_select_qm.declared_charge` — prepqm's rounding of the selected atoms' MM
     charges, unconfirmed until the user says so. `rounded_from` (`QM_CHARGE_ROUNDED`) is
     the sum it rounded: a non-integer sum is a region that cuts a bond or part of a charge
     distribution, which is a Stop below, not a charge to adopt.
   - `mimic_select_qm.min_vacuum_bohr` and `qm_cell_bohr` — vacuum between the region and
     the cell. `QM_CELL_NO_VACUUM` and `QM_CELL_TIGHT` are the difference between a run and
     a segmentation fault in the short-range potential.
3. `gmx_grompp` with the index it produced. The QM atoms are frozen on the GROMACS side,
   which is what the suppressed warnings are about: acknowledge `GROMPP_MAXWARN`
   deliberately, after reading what it names.
4. `mimic_prepare` with the confirmed charge and multiplicity. Its observation is the
   review surface to read out in full: `mimic_prepare.charge`, `.multiplicity`,
   `.functional`, `.sections` and `.paths`. `MIMIC_PATHS_MISSING` and `MIMIC_BOX_MISSING`
   are refusals about the coupling itself; `CPMD_MULTIPLICITY_WITHOUT_LSD` is the one that
   stops CPMD in its input parser rather than in the science.

Then stop and report. Launching is a separate request.

## Stop

- An unconfirmed charge or multiplicity: ask. This is the one question worth a turn, and
  `mimic_prepare` refuses without it (`QM_CHARGE_MULTIPLICITY_UNCONFIRMED`) unless an
  approved plan states both.
- A selection that cuts a covalent bond, reaches a metal centre, or leaves the region
  disconnected. `add-bound` marks an atom as a boundary atom, which gets `pp_info`'s boundary
  pseudopotential; whether that atom and that pseudopotential are the right treatment of the
  cut is the researcher's, and the harness places no hydrogen link atom. Say what the
  selection cuts and hand the decision back.
- `QM_CELL_NO_VACUUM`: refuse to continue rather than acknowledge it.
- A restart or a geometry from a differently solvated box: CPMD aborts inside its
  constraint solver, and the message is about geometry rather than about the mismatch
  (`CPMD_SHAKE_WRONG_GEOMETRY`, as the CPMD log diagnosis names it).

## Report

`plan_check` when a plan was written. Then: the selection as the user gave it and the atom
count the tool parsed, the charge and multiplicity with who confirmed each, the vacuum, the
pseudopotential mapping, and the exact CPMD sections `mimic_prepare` assembled. Say what was
**not** checked — a boundary the harness cannot judge, an embedding scheme it does not
model — and what the first coupled run would book.
