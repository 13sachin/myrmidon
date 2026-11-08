---
name: setup-and-equilibrate
description: >-
  Take a structure to a boxed, solvated, neutralised, minimised and equilibrated system with
  GROMACS, and report the numbers the engine measured. Use when the user asks to set up,
  prepare, solvate, neutralise, minimise, equilibrate, run or extend classical MD on a
  protein, a protein–ligand complex, nucleic acid, lipid or small molecule. Not for a QM region or MiMiC
  (qmmm-preflight), reference forces (fm-reference), a fit (fm-fit-and-validate), auditing
  an input without running it (audit-run-parameters), a run that failed
  (diagnose-a-failed-run).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "Set up this system in water and equilibrate it — and what will it cost?"
  myrmidon-tools: "study_open study_create protocol_lookup protocol_preview lookup mandate_write mandate_show inputset_lookup inputset_apply knowledge_search tool_lookup journal_write fetch_pdb structure_select ligand_parametrize complex_assemble gmx_pdb2gmx gmx_editconf gmx_solvate gmx_genion gmx_make_ndx gmx_genrestr observable_measure job_status study_report stage_prepare stage_run"
  myrmidon-protocols: "mimic_qmmm_solvated_solute protein_ligand_complex_in_water"
  myrmidon-stages: "gromacs.minimize gromacs.dynamics"
---

## Scope

A structure in, an equilibrated system and its measured properties out, for a protein, a
complex, a nucleic acid, a bilayer or a small molecule: the moves are the same, the values
come from the mandate's stages, compiled with their templates and the force field's
method. `protocol_lookup` names a published sequence whose `system_class` fits
(`protein_ligand_complex_in_water` for a deposited complex,
`mimic_qmmm_solvated_solute` for the classical half of a solute headed for QM/MM). Not here:
the QM region (`qmmm-preflight`), reference forces (`fm-reference`), a fit
(`fm-fit-and-validate`).

## The researcher decides

Into `decisions_yours`, with their words in `said` and in `locks` the stage fields the
words fix:

- the length of a stage, in the language of time (`production.duration_ps`);
- the temperature and pressure (`*.temperature_k`, `npt.pressure_bar`);
- what is held during equilibration, by subsystem (`nvt.restraints`, `npt.restraints`);
- the force field and the water model when no topology answers them, a box the question
  states, salt beyond neutrality, and the ceiling — words with no lock: the system files,
  the ceiling and the approval hold them;
- for a complex: the chains, the ligand copy, its protonation and net charge, its atom
  types and charge method;
- the criterion for "equilibrated", when they ask for a verdict;
- how each number they asked for is measured, when the question leaves it open: its atoms
  (heavy atoms or all, alike on every side), what an RMSD is superposed on and from which
  frame. Ask, or propose it in the deliverable.

Into `decisions_proposed`, with the source: what a published protocol or the literature
supports and the conversation left open. Ask only when no source supports a value. Yours,
free and recorded: the stages the locks leave open, the box, the solvent and ions, file
names, sampling intervals the deliverables need, ranks.

## Drive

Build in the session, one call at a time, using the file each call returns:

- A deposited complex: `structure_select` (chains kept, one ligand copy, the rest counted
  and discarded, heavy atoms the deposit leaves out built and listed: name them in the
  report) → `lookup source=ligand` with its residue code for what it is (name,
  formula, the dictionary's reference charge; its protonation and net charge here stay the
  researcher's) → `gmx_pdb2gmx` on the protein → `ligand_parametrize` →
  `gmx_make_ndx` and `gmx_genrestr` on the ligand's own coordinates and topology →
  `complex_assemble`, whose `net_charge` the neutralisation uses. The ligand's restraints
  must exist before a stage can hold it.
- Otherwise `gmx_pdb2gmx` only when no topology was given; a topology the user brings is
  used as it is.
- `gmx_editconf` with the box type and clearance the question or the protocol states; for a
  solute that will carry a QM region, a cell whose clearance exceeds the run's cutoffs.
- `gmx_solvate` returns a copy of the topology; every later call takes that copy.
- `gmx_genion` when the net charge is not zero or the researcher asked for salt, with the
  solvated structure and the returned topology; `output` is a bare file name.
- `gmx_make_ndx` for the groups a stage needs: group names in double quotes, numbers bare.

Read the findings each build call returns before the next one. `protocol_lookup` names a
published sequence whose stages are a starting point. Then `mandate_write`: the system files
with their kinds (structure, topology, index); the subsystems — left out, they are read from
the topology for you to confirm; the stages in physical terms, in order (a minimisation, the
restrained equilibration, the production, each with its ensemble, temperature, pressure,
length and restraints); the decisions with their locks; the deliverables as observables
(`rmsd` of the ligand superposed on the protein backbone, `min_distance` from the ligand to
the protein, `over` the production stage, with the `statistics` asked for); the ceiling. It
compiles and prices every stage. The approval is the researcher's: `/approve` in Hermes, the
dialog in Pi, or `myrmidon approve` at a terminal. Say the stages and their core-hours and end the
turn.

Under the approved mandate, per stage: `stage_prepare` with `stage=<name>` writes its `.mdp` and
compiles its run input, from the structure the stage it starts from left; `stage_run` books it,
its ranks and walltime sized by the harness. Extending a finished run is `stage_run` with
`continue_run: true`, never a fresh run. Then `observable_measure` with `deliverable=<name>` for
each deliverable, with `stage=<name>` and `property: thermodynamics` for the thermodynamics, and
with `property: settling` on the series the question is about, in the session and without being
asked.

## Stop

- `mandate_write` returns `needs_input`: ask its questions.
- `mandate_write` returns `ready`: say the stages and their core-hours; the turn ends at approval.
- A submission: say the job id and the core-hours booked, then stop.
- A submission the gate did not make: say what it named; a different system or a bigger
  ceiling is `mandate_write` with `revise=true` and the reason.
- A refusal from `gmx_solvate`, `gmx_genion` or `stage_prepare`: the system is not what the
  sequence assumed; say what the engine said.
- Another segment past the ceiling, or "equilibrated" with no criterion stated: theirs.

## Report

`study_report` after `mandate_show`, its `results` narrative saying in physical terms what
the deliverables' numbers mean, with a verdict only against the researcher's criterion;
every decision and whose it was, and what stays open; `stages` and `methods` readings when
you have them. The harness writes each deliverable as its measurement read it, the stage
checks and the record; tell the researcher those numbers. A reading to add or change is
`study_report` again with that section's `narrative`, never an edit of the file.
