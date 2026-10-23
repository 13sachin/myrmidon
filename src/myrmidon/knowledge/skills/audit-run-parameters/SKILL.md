---
name: audit-run-parameters
description: >-
  Check a run-parameter file without running it — a GROMACS .mdp or a CPMD input — and
  explain it: every finding by id, what the engine assumes for each key left unset, where
  each value came from, and the documented alternatives for a choice the user is weighing.
  Use when the user asks to audit, review or explain an .mdp or a CPMD input, asks what an
  option does or what GROMACS would assume, or asks which thermostat, barostat or algorithm
  to use. Do not use to build and run a system (setup-and-equilibrate), to review a QM
  region (qmmm-preflight), or to explain a run that already failed (diagnose-a-failed-run).
license: Apache-2.0
compatibility: myrmidon MCP server
metadata:
  myrmidon-question: "Is this input right, and what would the engine assume where it says nothing?"
  myrmidon-tools: "study_open study_create file_check knowledge_search inputset_lookup file_read"
  myrmidon-findings: "MDP_DT_UNSET MDP_NO_THERMOSTAT MDP_NSTEPS_UNSET MDP_REF_T_UNSET MDP_RANDOM_SEED MDP_CUTOFF_MISMATCH MDP_GEN_VEL_ON_CONTINUATION INPUTSET_VALUE_OVERRIDDEN"
---

## Scope

One input file in, an explained audit out, and nothing run. The file's own gate says what is
wrong with it; the engine's documentation says what each key means and what the engine does
when the key is absent; an input set, when one fits the system, says what a sourced value
would be. The audit is the deliverable: a fix is a separate request.

Not here: running anything (`setup-and-equilibrate`), the QM region (`qmmm-preflight`), a run
that already failed (`diagnose-a-failed-run`).

## The researcher decides

- **Which alternative to use**, when the question is a choice. You lay out the documented
  options and what each is recommended for; they pick.
- **Any change to the file.** An audit does not edit it.
- **Whether an input set fits their system.** Its `system_class` says what it holds for;
  you say whether that matches what they described.

## Start from

`study_open` (or `study_create`: the file is readable from a new study), then `file_check` on
the file the user named. A protocol is not needed: nothing is planned or spent.

## Drive

1. `file_check` on the file. Read its findings by id, `file_check.provenance` (an input set,
   a hand edit, or unknown) and `file_check.keys` for where each value came from.
2. For each key the findings say is unset, `knowledge_search` for that option: the passage
   states what the engine assumes. Quote it with its citation; do not state a default the
   documentation did not give you.
3. For a choice the user is weighing, `knowledge_search` with `mode` options for the
   decision, and passages for each alternative it names, cited for the installed version.
4. When an input set's `system_class` matches the system, `inputset_lookup` gives the
   sourced value to compare against. A key that differs is reported, not changed.

Then stop and report.

## Stop

- An option the documentation for the installed version does not describe: say so. A
  default from memory is not a cited one.
- The file is not one `file_check` can classify: say what it is and stop, rather than
  auditing it by eye.
- The user asked for a choice: ask it, after the alternatives, and do not apply one.

## Report

A table of findings by id with the key, the value or "unset", what the engine assumes when it
is unset, and the citation. Then, for a choice, the alternatives with what each is for and
the question to the researcher. Say which values came from an input set and which from a
hand edit.

## What goes wrong

- `MDP_NO_THERMOSTAT`: the run is NVE whatever the file's comments say.
- `MDP_REF_T_UNSET`: a thermostat with no target temperature; the gate blocks it.
- `MDP_GEN_VEL_ON_CONTINUATION`: a continuation that draws new velocities is not one.
- `MDP_CUTOFF_MISMATCH`: the two cutoffs differ. Allowed with PME; whether it is intended is
  the force field's convention, not the file's.
