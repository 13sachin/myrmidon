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
  myrmidon-tools: "study_open plan_write plan_check inputset_lookup inputset_apply knowledge_search tool_lookup fm_plan fm_fit fm_validate fm_report job_status file_write file_check"
  myrmidon-protocols: "fm_reference_singlepoint"
  myrmidon-inputsets: "fm_classical_rerun"
  myrmidon-findings: "FM_ITP_DEFINES_ATOMTYPES FM_RERUN_MDP_IS_MIMIC FM_NET_CHARGE_DRIFTED"
---

## Scope

A reference dataset in, fitted parameters and a verdict out — including the verdict that no
candidate is usable. The comparison shape is the same whether the arms are two fits of your
own or three topologies the user brought: one metric, the same held-out frames, one table,
and the starting topology as an arm so the comparison has a floor.

Not here: making the reference (`fm-reference`), the classical box
(`setup-and-equilibrate`). If `fm_plan` is not among your tools this machine lacks the
engine it needs (mimicpy): name it and let the user install it or move to a machine that
has it, rather than computing forces another way.

## The researcher decides

- **Which parameters the fit may move**, and the fitting strategy — charges first and then
  bonded terms, or everything at once. Both are legitimate; they answer different questions.
- **The QM total charge**, which `fm_plan` requires and will not derive.
- **Which candidates are in the comparison**, and what "better" means to them.
- **Whether a fitted parameter is chemistry.** The harness raises implausibility as
  findings; whether a term is defensible is the researcher's.
- **The budget.** A fit and a validation need no quantum compute, so this is usually small.

## Start from

`study_open`, then this skill. **No protocol publishes this half** — there is a published
sequence for the reference (`fm_reference_singlepoint`, driven by `fm-reference`) and none
for the fit, so `plan_write` will record the study as its own sequence rather than as
following one. That is the honest state, not a gap to paper over.

Values: `fm_classical_rerun` for the rerun run-parameters. `FM_ITP_DEFINES_ATOMTYPES` and
`FM_RERUN_MDP_IS_MIMIC` are the two findings that cost a whole run: know them before anything
is submitted.

## Drive

`plan_write` first, with the arms declared: `arms` is what makes an overridden value the
study's experiment rather than an unexplained change.

1. **The topology check that costs the whole run if it is skipped.** The molecule include
   file must not define its own atom types: the fit rewrites that file and the replacement
   does not carry them, so the run dies in the classical rerun *after* the quantum compute
   has been spent (`FM_ITP_DEFINES_ATOMTYPES`, raised by `fm_fit`'s gate). Move the block into the parent topology above the
   include, with `file_write` so the edit is recorded as yours, and `file_check` after.
2. `fm_plan` writes the fitting input. The QM total charge and the stride are required; its
   restraint weights surface as assumption warnings carrying the values they will take —
   read them and say which you accepted. `FM_FEW_SNAPSHOTS` reports a dataset smaller than
   the fit wants, and its suggestion is not actionable inside this stage.
3. `fm_fit`. The selection is a **file path**, not selection text. The run-parameter file is
   a plain classical rerun: a coupled-mode one is refused, because the bonded target would
   lose the QM electrostatics (`FM_RERUN_MDP_IS_MIMIC`). It is a job —
   `job_status`, then read what it raised.
4. `fm_validate` every candidate **and the starting topology**, on the same frames: the
   held-out force error and the minimised monitor-pair geometry. An arm dropped from the
   table is not a comparison.
5. `fm_report` for the parameter table, which is the part that decides.

## Stop

- `FM_INPUT_INCOMPLETE` or `FM_MDP_NO_FORCES`: the fit would train on nothing. Fix the
  input, do not lower the requirement.
- `FM_ANGLE_COLLAPSED`, `FM_FORCE_CONSTANT_IMPLAUSIBLE`, `FM_CHARGE_IMPLAUSIBLE`,
  `FM_NET_CHARGE_DRIFTED`, `FM_BOND_LENGTH_IMPLAUSIBLE`, `FM_UNPHYSICAL_PARAMS`: warnings,
  not refusals — a collapsed term is occasionally a legitimate stage of a staged fit. They
  are acknowledged by a person who says "yes, I know", and that acknowledgement is recorded.
- A candidate assembled by taking some columns from one fit and some from another is not a
  candidate: parameters from different fits are not interchangeable.
- Scores good and parameters not: that is the outcome this skill exists for. Say it.

## Report

`plan_check`, then one table: every arm including the starting topology, the same metric,
the same frames. Then the parameter findings, each with the rows it named and who
acknowledged it. Say which candidate you would use **and why, in terms of both the scores
and the parameters** — and if none is usable, say that. A fit that reproduces held-out
forces with broken parameters is worse than no fit, because it will be believed.

## What goes wrong

- A fitted ion charge that is exactly minus the sum of the charge-group targets means the
  declared QM total charge was wrong.
- A pinned atom that drifts from its stock charge means the restraint did not bind
  — check it against the stock value in the fit's output.
- The implausibility findings are a floor, not a ceiling. Read the table as well.
