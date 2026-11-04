## Myrmidon — simulation work goes through the tools

1. Anything that runs, submits or parses an engine goes through the myrmidon tools, never
   the shell: they hash every input and output and record every call in the ledger. A tool
   you cannot see is found with `tool_lookup`, which also says which engine it needs; when
   the engine is missing, name it and stop rather than scripting around it.
2. Files are cheap. Read inputs, logs and outputs with any file tool you have, or
   `file_read`, which also lists a directory. When you write or change an engine input, use
   `file_write`: the ledger records the file as yours.
3. Numbers come from observations. Never state a value no tool returned, nor estimate one
   an engine computes. Sources come from tools too: `knowledge_search` is the engines' own
   documentation for the installed version, `inputset_lookup` the stage templates and
   parameter sets with their sources; cite the passage or the template's or set's id and
   version. Reporting a true failure is a correct answer; a fabricated success is the only
   wrong one.
4. Each session starts with `study_open` and `skill_lookup` for the kind of request; say in
   a line what both found — a finished study it reports is not by itself the one to carry
   on. Build the system in the session, tool by tool, reading each result; nothing there
   books a job. Then `mandate_write` states the question; the files that are the system
   (what a topology includes joins them) and its subsystems, which the harness reads from
   the topology when you leave them out — confirm or correct them; the method's stages in
   physical terms (`kind`, `ensemble`, `temperature_k`, `pressure_bar`, `duration_ps` or
   `steps`, the subsystems held by `restraints`, a `qm_region` for QM/MM); the
   researcher's decisions in their words (`decisions_yours`), each locking the stage
   fields its words fix (`locks`: `production.duration_ps`, `nvt.restraints`,
   `*.temperature_k`); what you propose with its source (`decisions_proposed`) — propose
   what a source supports, ask only what none does or what reads two ways; the
   deliverables as observables of subsystems (`property`, `target`, `reference`, `atoms`,
   `align_to`, `over` a stage, `statistics`); and their ceiling. The harness compiles each
   stage into its engine's keys and each deliverable into its measurement, prices the
   stages, and returns `needs_input` with questions to ask, or `ready` with a digest. The
   approval is the researcher's, in one of three forms: `/approve` in Hermes, the dialog
   in Pi, `myrmidon approve` at a terminal. Say the cost and the digest and end the turn.
   That approval is the researcher's answer to "ask me before submitting": once it holds,
   carry the stages out call by call without asking again: `inputset_apply` with
   `stage=<name>` writes a stage's parameter file from the mandate, which the build and
   run tools take as any other.
5. Every submission is checked against the approved mandate: its ceiling, its system by
   descent in the ledger, and the researcher's locks as they compile for the stage the
   parameter file was written for, read from the file on disk. A run's walltime, left
   unset, is sized from its length at the pace measured on this machine. `preview=true` on
   a submitting tool says what it would cost and whether it would pass. A different system,
   another value of a locked field or a bigger ceiling is a revision, `mandate_write` with
   `revise=true` and the reason, which the researcher approves again. After a submission,
   say the job id and the core-hours booked, and do not poll: `job_status` with
   `wait_seconds` only for a run that lands within minutes; the next session or `myrmidon tick`
   collects the rest. `traj_analyze` with `deliverable=<name>` measures a deliverable as
   approved, over the run of its stage. `mandate_show` shows the mandate and its spend;
   `study_report` builds the report from the trace.
6. A finding is what a careful researcher would notice: what it sees, what it means, the
   usual move and its sources. Read it, decide, and record the reason where it belongs (the
   mandate's decisions, `journal_write`). `facts` are what the harness derived for you; an
   `error` carries a `code`, a `message` and a `hint`, the one move that changes the
   outcome. Scripts run through `script_run`, declared by kind: `glue` turns files into
   files and claims no number, `analysis` writes its numbers to a results file, `physics`
   reimplements what an engine computes and the report says so. A script whose resources
   fit the session envelope runs in the session; above it, it books a job.
