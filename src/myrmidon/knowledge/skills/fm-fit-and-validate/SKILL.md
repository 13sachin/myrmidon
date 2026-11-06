---
name: fm-fit-and-validate
description: >-
  Fit force-field parameters to a QM/MM reference dataset and judge whether the result is
  usable: the fitting input and its weights, the fit itself, validation of every candidate
  and of the starting topology on the same held-out frames, and the parameter table. Use
  when the user asks to fit or refit a force field, refit charges or bonded parameters, to
  compare two or more force fields or topologies for one molecule, or to say which one
  agrees best with a QM/MM reference. Do not use to make the reference data itself
  (fm-reference), to build the classical box (setup-and-equilibrate), or to review a QM
  region (qmmm-preflight).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "Build a better force field for this molecule — and of these candidates, which one agrees best?"
  myrmidon-tools: "study_open mandate_write mandate_show inputset_lookup inputset_apply knowledge_search tool_lookup fm_plan fm_fit snapshot_select gmx_grompp gmx_rerun observable_measure gmx_minimize_frames observable_measure study_report job_status file_write file_check"
  myrmidon-protocols: "fm_reference_singlepoint"
  myrmidon-stages: "gromacs.single_point gromacs.minimize"
---

## Scope

A registered reference dataset in, fitted parameters and a verdict out, including that no
candidate is usable. Fits of your own and topologies the user brought compare the same way:
one metric, the same held-out frames, one table, the starting topology as an arm. No
protocol publishes this half; the mandate states its stages as the study's own. Not here:
the reference (`fm-reference`), the classical box (`setup-and-equilibrate`). Without
`fm_plan` among your tools, `tool_lookup` names the missing engine: say so and stop.

## The researcher decides

Into `decisions_yours`, quoted: which parameters the fit may move and the strategy; the
candidates and what "better" means, as one constraint listing every candidate value (a
sweep is one mandate); whether a fitted term is defensible chemistry; the ceiling. Into
`decisions_proposed`, with the source: the QM total charge the reference's `mimic_prepare`
recorded in its sidecar, and the restraint weights `fm_plan` states it assumes. Yours,
recorded: the held-out frames, the starting topology as an arm, a topology edit the fit
needs (with `file_write`).

## Drive

1. `fm_plan` in the session, with `dataset_frames` (the `frames` `dataset_collect` registered),
   the QM total charge and the stride; say which of its assumed weights you accept.
2. `fm_fit` with `preview=true` prices it. `mandate_write`: the files the fit and the arms
   start from as the system; the fit as a `parameter_fit` stage, the validation as a
   `single_point` stage whose `scan` runs over the candidate topologies, and, when the
   minimised geometry is asked for, a `minimize` stage with `regime: mm`; the deliverables
   (`force_rmse` over the validation); the ceiling.
   The approval is the researcher's: `/approve` in Hermes, the dialog in Pi, or
   `myrmidon approve` at a terminal. Say the cost and the digest and end the turn.
3. `fm_fit` with the selection as a file path and a classical rerun mdp:
   `inputset_apply` with `stage=<the validation stage>`. An edit it asks of the topology goes
   through `file_write`, then `file_check`.
4. Per arm, the starting topology included: `snapshot_select` with `output: trajectory` writes
   the held-out frames once; `gmx_grompp` with that rerun mdp and the arm's topology →
   `gmx_rerun` over those frames → `observable_measure` with the same `frames`. For the
   minimised geometry, `gmx_minimize_frames` with that `minimize` stage's mdp, then
   `observable_measure` with `metric: pair_distance` against the reference trajectory.
5. `study_report` for the parameter table.

## Stop

- `mandate_write` returns `needs_input` (ask) or `ready` (the cost and the digest).
- After a submission: the job id and the core-hours, then stop.
- A submission the gate did not make: say what it named; a candidate outside the approved
  list is `mandate_write` with `revise=true` and the reason.
- `fm_fit` refuses its input: fix the input; never lower the requirement.
- Scores good and parameters not: say so.

## Report

`mandate_show`, then the deliverables the mandate names: one table of every arm, the
starting topology included, on the same metric and frames; the parameter findings with the
rows they name; the candidate you would use and why, in scores and in parameters, or that
none is usable. A candidate built from columns of different fits is not one.
