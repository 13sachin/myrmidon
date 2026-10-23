---
name: run-the-plan
description: >-
  Run an approved plan without the researcher at the keyboard: start the study's loop,
  say where it stands when they come back, and record their answer to a question the loop
  parked. Use when the researcher approves a plan and wants it run, asks how a running
  study is doing, or answers a question the loop asked — "run it", "where is it", "go with
  the rerun". Not for writing or costing the plan (plan-and-cost), for a step done by hand
  in the session (setup-and-equilibrate, qmmm-preflight, fm-reference), or for a run that
  failed outside the loop (diagnose-a-failed-run).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "Run the plan while I am away, and tell me what it needs from me when I am back."
  myrmidon-tools: "study_open plan_check plan_approve loop_start loop_status loop_tick loop_answer loop_stop journal_write study_report"
  myrmidon-findings: "LOOP_PLAN_NOT_APPROVED LOOP_QUESTION_UNKNOWN LOOP_OPTION_UNKNOWN LOOP_ANSWER_IS_THE_RESEARCHERS"
---

## Scope

An approved plan runs itself: the loop collects what ended, checks each run against the
plan's sentinels and criterion, continues what is ready, reruns or extends inside the
envelope the researcher approved, and parks what it cannot settle as a question with the
evidence and the options. Your part is the pipe between the researcher and that loop:
start it, read it back, carry an answer in.

Not here: writing or costing the plan (`plan-and-cost`), running a step yourself, or
polling jobs by hand — a plan that is running is the loop's, and `job_status` in a loop of
your own would double its bookkeeping.

## The researcher decides

- **That it runs.** `plan_approve` is theirs; `loop_start` refuses without it
  (`LOOP_PLAN_NOT_APPROVED`).
- **Every parked question.** `loop_status` lists each with its options and costs; put it
  to the researcher in those words and call `loop_answer` only on their choice
  (`LOOP_ANSWER_IS_THE_RESEARCHERS` is their yes, never yours).
- **Whether to stop.** `loop_stop` on their word; submitted jobs keep running.

## Start from

`study_open`, then `plan_check`: the plan, whether its approval holds, what has run. If the
plan is not approved, `plan-and-cost` is the skill and approval comes first.

## Drive

1. On "run it", once `plan_check.approval_holds`: `loop_start`. Report the process id and
   that the study now advances on its own; nothing else to do in this session.
2. On "where is it": `loop_status`. Say which steps are done, which job is live, what the
   last events were, and — first of all — what waits on the researcher.
3. On a question: show `loop_status.questions` as the loop put them — the step, what
   stopped it, the evidence, each option with its cost. When the researcher chooses,
   `loop_answer` with the question's id and the option's id. Say that the next pass
   carries it out.
4. A decision the researcher states that the loop did not ask for — why they chose, what
   they ruled out — goes to `journal_write`, so the loop's next wake-up sees it.
5. No loop process and the researcher wants a pass now: `loop_tick` once, and report what
   it did.
6. When `loop_status.done`: `study_report`, with prose around the values it injects.

## Stop

- `loop_start` refused on `LOOP_PLAN_NOT_APPROVED`: the plan changed or was never
  approved — back to `plan-and-cost`.
- A question whose options all cost more than the researcher wants to spend: `stop` is
  always one of them, and a larger ceiling is a new plan to approve.
- A loop process that is not alive while jobs are live: `loop_start` again; the study lost
  nothing.

## Report

Where the study stands in the plan's own step names; what waits on the researcher, each
question with its options; the core-hours booked against the ceiling; and, when the plan
is done, the report.
