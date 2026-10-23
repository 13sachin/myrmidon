## Myrmidon — simulation work goes through the tools

1. Anything that runs, submits, parses or plans goes through the myrmidon tools, never the
   shell: they validate inputs, gate expensive runs on a core-hour estimate, and record
   every call, parameter and artifact hash in the ledger. A tool you need but cannot see
   is not installed on this machine — `tool_lookup` says what exists and which engine it
   needs; name it and stop, never work around it with a script.
2. Files are cheap, tools are not. Read engine files, logs and outputs with whatever file
   tool you have — your own, or `file_read`, which also lists a directory; the harness
   hashes every file a tool consumes, so a read records nothing extra. When you change an
   engine input file, use `file_write` so the ledger records the edit as yours.
3. A warning to acknowledge is a decision: read it, decide, then pass its id in
   `acknowledge_warnings`, never pre-emptively; a refusal's message is its diagnosis.
   `next_actions` in a result are the steps known open — an approved retry, a missing input,
   a job to wait for, a file's parser, or `escalate` (stop, report); never physics or consent.
4. Numbers come from tool observations. Never restate a value the tools did not return, and
   never estimate one an engine would have computed. Unsure of an engine option or a method?
   `knowledge_search` first — the engines' own documentation, versioned; quote it and cite
   it, rather than recalling it.
   Never invent a parameter either: `inputset_lookup` is the source, cite the id and version
   you used, and what it lists as `required_from_user` is chemistry you obtain and confirm,
   never fill in. Reporting a true failure — missing engine, corrupt input, impossible
   request — is a correct answer; fabricating success is the only wrong one.
5. First thing in every session: `study_open` with no path, then `skill_lookup` for the kind
   of request; say in a line what both found. `job_status` collects each job that ended, and
   waits on one still running (rule 6). A study it reports as finished is not automatically
   the one to carry on with — say what it was, and carry on only when the request is plainly
   about it; `study_create` starts a new one. **A study that will spend core-hours starts with
   a plan**: `protocol_lookup`, then `plan_write` — it costs the study and runs nothing.
   Approving it is the researcher's (`plan_approve`); `plan_check` asks later if it held.
6. After a submission, say the job id, the core-hours and the walltime booked. If it
   should end within ten minutes, call `job_status` with `wait_seconds` (≤600) so the
   session reports the result the moment it lands instead of going silent; if it is longer,
   stop — it runs without you, and the next session collects it. Never poll turn by turn.
7. Two things are the researcher's, never yours: a physics choice the input set marks
   `required_from_user` or that the request left open, and a finding that needs a person
   (`COMPUTE_BUDGET` above the threshold) when no approval covers it. A finding the study's
   approval already accepts is acknowledged by it — the call is not refused for it. The box,
   the solvent, input set values and file names are yours to decide from the input set and
   to report.
8. Research scripts run through `script_run` (hashed, gated, ledgered) — but a script is for
   what no engine computes: reimplement one and you are the source of the physics, and its
   numbers look grounded. `tool_lookup` first.
9. Engine input files are the researcher's. `file_check` on one says what its gate finds
   and, if an input set rendered it, what changed since and whose each value is: one on the
   study's declared axis is theirs, any other the input set states is a finding.
