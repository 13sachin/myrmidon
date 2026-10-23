---
name: plan-and-cost
description: >-
  Say what a study would do, step by step, and what it would cost in core-hours, before
  anything runs: find the published pipeline it follows, write it as this study's plan, and
  report the steps and the plan's total. Use when the user asks what a study or a pipeline
  would do, how long or how expensive it would be, or wants the plan before deciding to run
  — "what would this do and cost me?", "don't start it yet". Do not use to run any of it
  (setup-and-equilibrate, qmmm-preflight, fm-reference, fm-fit-and-validate are for doing
  the work once the plan is accepted).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "What is this study going to do, and what will it cost me?"
  myrmidon-tools: "study_open protocol_lookup plan_write plan_check plan_approve inputset_lookup"
  myrmidon-protocols: "mimic_qmmm_solvated_solute fm_reference_singlepoint"
  myrmidon-findings: "PLAN_PROTOCOL_UNKNOWN PLAN_ADAPTS_PROTOCOL PLAN_FOLLOWS_PROTOCOL PLAN_ALREADY_WRITTEN PLAN_STEP_INPUT_UNSATISFIED"
---

## Scope

The question asked before spending: what would this study do, and what would it cost? The
answer is a plan the study holds — written by `plan_write`, costed by it, approvable by the
researcher — not a sequence written out in prose. A plan costs nothing to write and runs
nothing.

Not here: running any step. When the researcher accepts the plan, the work itself belongs to
the skill for that kind of step.

## The researcher decides

- **Whether to run it at all**, and at what cost. You say what it would book; they decide.
  Approving the plan is theirs (`plan_approve`), never yours.
- **Which pipeline**, when more than one published pipeline fits the request. Name the
  candidates and what differs between them.
- **The inputs a protocol asks for and the request does not give.** List them as missing;
  never invent a file or a value to fill a step.

## Start from

`study_open`, then this skill, then `protocol_lookup` with the words of the request — the
system and the kind of study. A published pipeline is the starting point: `mimic_qmmm_solvated_solute`
for a solute treated by MiMiC QM/MM in a water box, `fm_reference_singlepoint` for QM/MM
reference forces. A protocol's `system_class` says what its estimates were measured on.
Input files the user named are what the plan is `given`.

## Drive

1. `protocol_lookup`. A pipeline fits when its `system_class` covers the system described,
   not when its name matches the request. One that fits: that is the plan's source. Several:
   name them and ask which. None: say so, and offer to write the plan step by step from
   input sets whose `system_class` fits, only if the researcher wants it. A protocol's
   estimates hold for the system its `system_class` describes; for a larger one, say the
   total is that system's, not this one's.
2. `plan_write` with `from_protocol` set to that pipeline's id, `question` in the user's own
   words, and `given` naming the inputs they supplied. This writes the study's plan and
   costs it. Read the observation:
   - `plan_write.steps` — the sequence, in order, as the plan holds it.
   - `plan_write.total_core_h` — what the plan would book in total.
   - `plan_write.unestimated` — steps that submit a job but state no scheduler resources,
     so they carry no cost, the total is a floor, and the plan cannot be approved. A
     step's cost is computed from the resources in its `values`, never given.
   - `plan_write.digest` — what an approval would be recorded against.
3. A refusal names what the plan lacks: `PLAN_STEP_INPUT_UNSATISFIED` is an input no
   earlier step produces and the request did not give — report it as missing.
   `PLAN_PROTOCOL_UNKNOWN` means the id was not a published pipeline.
4. `plan_check` only if the study already held a plan (`PLAN_ALREADY_WRITTEN`): say what
   it holds and what has run against it, rather than writing over it.

Then stop and report. Nothing is submitted: a plan is the whole answer to this question.

## Stop

- After `plan_write` succeeds: report and stop. Do not run the first step to "show" it.
- A protocol that needs an input nobody has given: report the plan with that step marked,
  and ask for the input.
- The researcher asked to run it: that is a new request, and the plan's approval comes first;
  once approved, `run-the-plan` starts the loop that runs it.

## Report

The steps in the researcher's terms — building the box, minimisation, equilibration,
annealing, heating, production, as the plan names them — with the tool each would call. Then
`plan_write.total_core_h` as the plan's cost, saying that it is a floor when
`plan_write.unestimated` is not empty and which steps state no resources — walltime and
ranks are the researcher's to agree, so ask for them. `plan_write.unresolved` lists every
value the plan still needs before it can be approved — the researcher's decisions, the
resources (with what a run like it booked where the protocol was measured), and the
parameters nothing states; say each one as a question. Name the
pipeline it follows and the inputs it is missing, say that nothing ran, and that
`plan_approve` is the researcher's step if they want it run.
