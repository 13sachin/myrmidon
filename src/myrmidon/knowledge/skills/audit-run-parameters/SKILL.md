---
name: audit-run-parameters
description: >-
  Check a run-parameter file without running it — a GROMACS .mdp or a CPMD input — and
  explain it: every finding with what it means, what the engine assumes for each key left unset, where
  each value came from, and the documented alternatives for a choice the user is weighing.
  Use when the user asks to audit, review or explain an .mdp or a CPMD input, asks what an
  option does or what GROMACS would assume, or asks which thermostat, barostat or algorithm
  to use. Do not use to build and run a system (setup-and-equilibrate), to review a QM
  region (qmmm-preflight), or to explain a run that already failed (diagnose-a-failed-run).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "Is this input right, and what would the engine assume where it says nothing?"
  myrmidon-tools: "study_open study_create file_check knowledge_search inputset_lookup file_read mandate_write mandate_show"
---

## Scope

One input file in, an explained audit out; nothing runs and nothing is spent. A fix is a
separate request. Not here: running anything (`setup-and-equilibrate`), the QM region
(`qmmm-preflight`), a failed run (`diagnose-a-failed-run`).

## The researcher decides

Which alternative to use when the question is a choice, any change to the file, and whether
a stage template's or input set's applicability fits their system. A value they decided
under an approved mandate stays as its `decisions_yours` states it. Yours: the template or
set to compare against and the passages to cite.

## Drive

1. `study_open` (or `study_create`; the file is readable from a new study), then
   `file_check` on the file: its findings, `provenance` (a stage's template, an input set, a
   hand edit, or unknown) and `keys`, where each value came from.
2. For each key left unset, `knowledge_search` for that option and quote what the engine
   assumes, with its citation.
3. For a choice, `knowledge_search` for the decision and each alternative it names.
4. When a stage template or input set fits, `inputset_lookup` with its id for the sourced
   values; a key that differs is reported, not changed.

## Stop

- At the report; nothing is submitted.
- An option the installed version's documentation does not describe, or a file
  `file_check` cannot classify: say so.
- A choice: lay out the alternatives and ask; apply none.
- A change to a value the researcher decided is a revision: `mandate_write` with
  `revise=true` and the reason, approved with `/approve` in Hermes, the dialog in Pi, or
  `myrmidon approve` at a terminal.

## Report

A table: each finding with its key, the value or "unset", what the engine assumes, and the
citation; for a choice, the alternatives and the question; which values came from an input
set and which from a hand edit.
