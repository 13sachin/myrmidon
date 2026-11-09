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
  myrmidon-tools: "study_open mandate_write mandate_show inputset_lookup inputset_apply knowledge_search tool_lookup stage_prepare stage_run snapshot_select observable_measure study_report job_status file_write file_check"
  myrmidon-protocols: "fm_reference_singlepoint"
  myrmidon-stages: "gromacs.single_point gromacs.minimize"
---

## Scope

A registered reference dataset in, fitted parameters and a verdict out, including that no
candidate is usable. Fits of your own and topologies the user brought compare the same way:
one metric, the same held-out frames, one table, the starting topology as an arm. No
protocol publishes this half; the mandate states its stages as the study's own. Not here:
the reference (`fm-reference`), the classical box (`setup-and-equilibrate`). Without
a fitting engine on this machine, `tool_lookup` names the missing one: say so and stop.

## The researcher decides

Into `decisions_yours`, quoted: which parameters the fit may move, the fit stage's
`fit_mode` (`charges`, `bonded` or `both`) locked as `<fit>.fit_mode`, and the strategy; the
candidates and what "better" means, as one constraint listing every candidate value (a
sweep is one mandate); whether a fitted term is defensible chemistry; the ceiling. Into
`decisions_proposed`, with the source: the QM total charge the reference's stage states, and
the restraint weights the fit's input takes from its template, each with its rationale. Yours,
recorded: the held-out frames, the starting topology as an arm, a topology edit the fit
needs (with `file_write`).

## Drive

1. `mandate_write`: the files the fit and the arms start from as the system; the fit as a
   `parameter_fit` stage with its `qm_region` (the QM total charge) and `fit_mode`, each comparison of an arm
   as a `single_point` stage with `regime: mm`, and, when the minimised geometry is asked for,
   a `minimize` stage with `regime: mm` and a `scan` of `frames`; the deliverables (`force_rmse`
   over a comparison); the ceiling. It prices each stage.
   The approval is the researcher's: `/approve` in Hermes, the dialog in Pi, or
   `myrmidon approve` at a terminal. Say the stages and their core-hours and end the turn.
2. `stage_prepare` with `stage=<the fit>` writes its input and its classical rerun's: every
   frame of the collected dataset unless `overrides` narrows the `stride`. Say which of its weights you accept. An edit it asks
   of the topology goes through `file_write`, then `file_check`.
3. `stage_run` with `stage=<the fit>`, `preview=true` first if the price is in doubt; it finds
   the dataset, the selection and the index itself. `job_status` collects the fitted topologies
   and the candidate.
4. Per arm, the starting topology included: `snapshot_select` with `output: trajectory` writes
   the held-out frames once; `stage_prepare` with `stage=<the comparison>` and that arm's
   `topology`, then `stage_run` with `inputs: {trajectory: <the frames>}`, then
   `observable_measure` with `deliverable=<its force_rmse>` and the same `frames`. For the
   minimised geometry, `stage_run` on the `minimize` stage, then `observable_measure` with
   `property: pair_distance` against the reference trajectory.
5. `study_report` for the parameter table.

## Stop

- `mandate_write` returns `needs_input` (ask) or `ready` (say the stages and the core-hours).
- After a submission: the job id and the core-hours, then stop.
- A submission the gate did not make: say what it named; a candidate outside the approved
  list is `mandate_write` with `revise=true` and the reason.
- A fit whose run refuses its input: fix the input; never lower the requirement.
- Scores good and parameters not: say so.

## Report

`study_report` after `mandate_show`: the harness writes each deliverable as its measurement
read it (every arm, the starting topology included, on the same metric and frames), the fit's
tables and the record. Its `results` narrative says in physical terms the parameter findings
with the rows they name and the candidate you would use and why, in scores and in
parameters, or that none is usable; `stages` and `methods` readings when you have them. A
candidate built from columns of different fits is not one.
