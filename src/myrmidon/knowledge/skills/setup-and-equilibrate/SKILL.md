---
name: setup-and-equilibrate
description: >-
  Take a structure to a boxed, solvated, neutralised, minimised and equilibrated system with
  GROMACS, and report the numbers the engine measured. Use when the user asks to set up,
  prepare, solvate, neutralise, minimise, equilibrate, run or extend classical MD on a
  protein, nucleic acid, lipid or small molecule. Not for a QM region or MiMiC
  (qmmm-preflight), reference forces (fm-reference), a fit (fm-fit-and-validate), auditing
  an input without running it (audit-run-parameters), a run that failed
  (diagnose-a-failed-run), or the cost of a study before it runs (plan-and-cost).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "Set up this system in water and equilibrate it — and what will it cost?"
  myrmidon-tools: "study_open study_create protocol_lookup plan_write plan_approve plan_check inputset_lookup inputset_apply knowledge_search tool_lookup gmx_pdb2gmx gmx_editconf gmx_solvate gmx_genion gmx_make_ndx gmx_genrestr gmx_grompp gmx_mdrun checkpoint_continue gmx_energy series_convergence traj_analyze job_status"
  myrmidon-protocols: "mimic_qmmm_solvated_solute"
  myrmidon-inputsets: "genion_run_input box_qmmm_solute em_solute_in_water npt_equilibration_solute_in_water em_solvated_biomolecule nvt_equilibration_protein npt_equilibration_protein"
  myrmidon-findings: "STRUCTURE_HAS_NO_ATOMS GENION_TOPOLOGY_ALREADY_EDITED NDX_COMMAND_REJECTED GENRESTR_NOT_MOLECULE_NUMBERING GROMPP_NET_CHARGE ENERGY_TERM_NOT_IN_EDR COMPUTE_BUDGET_EXCEEDED CPT_NAME_MISMATCH"
---

## Scope

A structure in, an equilibrated system and its measured properties out. The steps are the
same for a protein, a nucleic acid, a bilayer or a small molecule; what changes between them
is values, and values come from an input set.

Not here: the QM region and anything MiMiC (`qmmm-preflight`), reference forces
(`fm-reference`), a fit (`fm-fit-and-validate`). Production beyond equilibration has no
stopping criterion in the harness — say so rather than choosing one.

## The researcher decides

- **The force field and the water model.** `gmx_pdb2gmx` takes both and neither has a safe
  default. A topology the user brings instead answers this already.
- **What to equilibrate to, and for how long** — the ensemble, and the length in the
  language of time rather than of steps. The input sets leave the length to them
  (`required_from_user`); it is recorded as theirs.
- **Restraints**, and on what — a protein or a ligand held while the solvent relaxes.
- **The budget**, when `COMPUTE_BUDGET` fires: show what it says, ask, acknowledge by id.
- **Salt**, when the system wants more than neutrality.

Nothing else. The box shape, the solvent count, ion placement, stage and file names are
yours to take from the input set and to report. The clearance is the request's when it gives
one; otherwise derive it as the box input set says, and say what you chose.

## Start from

`study_open` first, then this skill. `protocol_lookup` for a published sequence whose
`system_class` fits: `mimic_qmmm_solvated_solute` states the classical half for a solute
headed for QM/MM, with the values each step cites.

Values: pick by each set's `system_class`, never by name. `box_qmmm_solute` for a solute
that will carry a QM region (it renders no file — `inputset_apply` with no `output` resolves
it, and the clearance is derived for the solute); `em_solute_in_water` and
`npt_equilibration_solute_in_water` for a small solute's two classical stages;
`em_solvated_biomolecule`, `nvt_equilibration_protein` and `npt_equilibration_protein` for a
protein or a complex. **No protocol cites the protein ones**, so a study using them is
assembling its own sequence and `plan_write` records that as an adaptation.
`inputset_lookup` before writing any run-parameter file by hand.

## Drive

**`plan_write` before anything that spends**, with the steps, what the user gave, and the
estimate. Above the threshold the acknowledgement is the study's approval; `plan_approve`
records a later one.

1. `gmx_pdb2gmx` only when no topology was given — read `gmx_pdb2gmx.net_charge`, which
   decides whether step 4 runs at all. A topology the user brings is used as it is, and its
   net charge is what grompp reports (`GROMPP_NET_CHARGE`). A structure the tool reports as
   having no atoms is a selection or format problem (`STRUCTURE_HAS_NO_ATOMS`), not a reason
   to try another force field.
2. `gmx_editconf` with the resolved `box_type` and `distance_nm`. `BOX_CLEARANCE_TIGHT`
   means the solute can see its own periodic image; acknowledge it only with a reason.
3. `gmx_solvate` — it edits a **copy** of the topology and returns it. Every later step
   takes the returned one; using the planted one is how a study silently diverges.
   `SOLVATE_ALREADY_SOLVATED` is a correct refusal, not an obstacle.
4. `gmx_genion` when the net charge is not zero, given the solvated `structure` and the
   topology `gmx_solvate` returned: it builds the run input it reads itself, from
   `genion_run_input`. `output` is a bare file name. A failure after genion rewrote the
   topology says so (`GENION_TOPOLOGY_ALREADY_EDITED`): retry from the original, not the
   edited copy.
5. `inputset_apply` for each stage's run parameters, then `gmx_grompp`, then `gmx_mdrun`.
   `GROMPP_NET_CHARGE` before neutralisation is the expected one; `GROMPP_MAXWARN` is a
   decision, read what it names. `MDP_RANDOM_SEED` left unfixed makes the run unreplayable.
   Position restraints under pressure coupling have an option of their own:
   `knowledge_search` it. Groups past the defaults come from `gmx_make_ndx`; a group's
   restraints from `gmx_genrestr`, which names the moleculetype they belong inside.
6. `gmx_energy` and `traj_analyze` for the numbers. `gmx_energy.terms` carries each term's
   mean and its drift; a term string that matches nothing fails with the labels the file
   offers (`ENERGY_TERM_NOT_IN_EDR`, `available_terms`) rather than guessing.

A submission ends the turn: say the job id and what was booked, then stop. `job_status` with
`wait_seconds` only when the run should land inside the guidance's window. Extending a
finished run is `checkpoint_continue`, never a fresh `gmx_mdrun`: the names it continues are
checked (`CPT_NAME_MISMATCH`).

## Stop

- `gmx_solvate` or `gmx_genion` refused: the system is not what the sequence assumed. Say
  what it said.
- `gmx_grompp` names a physics warning: it is a decision for the user, not a flag to raise.
- A booking above the machine's block threshold: the study does not start. The request is
  for more than this allocation allows (`COMPUTE_BUDGET_EXCEEDED`).
- **Whether it is equilibrated is the researcher's criterion.** `series_convergence` on the
  `.xvg` measures where the series settles, its drift and its halves; it gives a verdict
  only on a criterion they stated. Do not supply one, and do not chain another segment on
  your own judgement — an extension the researcher asked for is theirs, and is done.

## Report

`plan_check` first: steps done and pending, and planned against booked against consumed.
Then the artifacts by role, the two decisions and whose they were, and every number with the
tool that measured it. A final-frame value is one sample — report the mean and the drift.
What the harness could not decide (equilibration, production length) is stated as open.

## What goes wrong

- `MDP_NO_THERMOSTAT`, `MDP_NSTEPS_UNSET` — an input set that was overridden into an
  incomplete state; read the sidecar rather than adding the key.
- `INPUTSET_VALUE_OVERRIDDEN` — you changed a value the input set states. Say which and why.
