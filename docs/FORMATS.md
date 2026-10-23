# File formats

What myrmidon leaves on disk, for anyone who reads it without importing myrmidon: a study's
manifest and ledger, and a session's transcript. Every file is UTF-8; every `.jsonl` file
has one JSON object per line, appended and never rewritten.

**Versions.** A study states its format in its manifest (`format`, **2**); it covers the
manifest, the ledger and the study's layout. A transcript states its format in the `trace`
event that opens every session in it (`format`, **1**). Adding an optional field keeps a
version, so a reader ignores keys it does not know; removing or renaming an event or a field,
or changing what one means, moves it. There is one current format of each: myrmidon refuses
a study of another format rather than migrating it.

`tests/unit/test_formats.py` holds this page to the code: every event kind written is
described here and nothing else is, every ledger field is listed, every field a session
writes is named under its event, and the version numbers are the ones above.

## Study (workspace format 2)

A study is a directory:

| Path | What it is |
|---|---|
| `myrmidon.workspace.json` | the manifest (below) |
| `myrmidon.ledger.jsonl` | the provenance ledger (below) |
| `NN_name/` | stages, numbered in the order they were made (`01_input`, `02_em`, ...) |
| `NN_name/NN_deffnm/` | one coupled run each: `mimic_run` allocates a numbered directory under its stage per submission and never reuses one, since CPMD's output names are fixed |
| `plan.yaml` | the study's plan, written by `plan_write` |
| `report.md` | the plan's report, written by `study_report`: the harness's sections, the model's prose after each |
| `events.jsonl` | the loop's event spool (below), appended by `myrmidon tick` and `myrmidon answer` |
| `journal.jsonl` | the journal (below): decisions, ruled-out causes, suspicions, intentions |
| `inbox.jsonl` | parked questions and notices, and the answers to them (below) |
| `.myrmidon/` | the harness's own state, not study artifacts: the job store, run scratch, `transcripts/` of `myrmidon run` sessions started in the study, `loop.json` (the loop's working record, below) and `wakes/` (every wake's context and reply) |

The manifest is one JSON object. Fields: `format`, `name`, `description`, `created`, `origin`.

- `format`: 2.
- `created`: ISO 8601, UTC, to the second.
- `origin`: the project directory that made the study; absent when none did.

## Ledger

`myrmidon.ledger.jsonl`, one event per line, appended by the registry for every tool call made
with the study open — except an unchanged look at something already running (a job poll that
saw nothing new), which is not worth a line. A refused call is an event too.

| Field | What it holds |
|---|---|
| `event_id` | `ev-` and 12 hex digits; what an observation's `provenance_id` names |
| `timestamp` | ISO 8601, UTC |
| `tool` | the tool's name |
| `params` | the call's parameters as validated, defaults filled in |
| `command` | the argv of the binary that ran, or null; `replay` renders these as a script |
| `engine_versions` | engine name → version string, as discovered for the session |
| `myrmidon_version` | the harness version that wrote the event |
| `inputs` | artifacts the call read (below) |
| `outputs` | artifacts the call wrote (below) |
| `warnings_acknowledged` | finding ids the call accepted |
| `ok` | whether the tool reports success; a refused call is false |
| `duration_s` | seconds the tool ran; 0 for a refusal |
| `exit_code` | the binary's exit code, or null |
| `kind` | `call` (it did something), `observe` (a look at something that exists: counted apart, not replayed), `gate_refused` (the validation gate refused it; nothing ran) |
| `knowledge_refs` | ids of the findings the call surfaced: gate warnings, advisories, matched failure signatures |
| `approval` | the `event_id` of the approval an acknowledgement relied on; empty when the agent acknowledged on its own |
| `params_schema` | a short hash of the tool's parameter schema, so a replay can tell a tool that has since changed |
| `plan_id` | the `plan_id` of the study's `plan.yaml` when the call was made; empty when the study had no plan. Only a plan's own calls carry out its steps |
| `plan_step` | the step of that plan the call carried out, when an approved plan bound it; empty otherwise |

An artifact is an object with fields `path` (relative to the study when it is inside it,
else absolute), `role` (`structure`, `topology`, `mdp`, `tpr`, `trajectory`, `log`, ...),
`sha256` and `size_bytes`.

**An approval** is an event with `tool` `plan_approve` and `ok` true: its
`warnings_acknowledged` are the findings a person accepted, `params.approver` says who, and
`params.call`, when present, scopes it to one call (that tool with exactly those
parameters), used up by the call that relies on it. Without `params.call` and with no plan
attached it stands for the study.

The ledger does not hold what a tool returned; the transcript does (`tool_result.obs`).

## The loop's files

`myrmidon tick` advances a study one pass at a time and leaves everything it knows in the
study (`core/spool.py`). Three append-only JSONL files sit beside the ledger; the loop's
working record sits under `.myrmidon/`.

### Spool: `events.jsonl`

One event per line. Fields: `event_id` (`sp-` and 12 hex digits), `ts` (ISO 8601, UTC),
`kind`, `subject` (the plan step the event is about, with its arm in brackets for a sweep:
`sp[frame2]`; `plan` for the plan as a whole), `detail` (an object whose keys depend on the
kind), `plan_id`.

The kinds, all machine-detectable: `job` (a tracked job ended, failed, or a call was
refused: `job_id`, `state`, `ok`, `exit_code`, `diagnosis_id`, `summary`, `event` — the
collecting ledger event — or `refused` with `findings`), `sentinel` (a step's sentinel
did not hold: `check`, `value`, `detail`, `job_id`; or the until-check was evaluated:
`until`, `met`, `segments`, `findings`), `checkpoint` (reserved for a declared look at
parsed data), `human` (an answer: `inbox`, `option`, `action`, `args`, `by`), `budget`
(`booked_core_h`, `ceiling_core_h`, `share`), `deadline` (`job_id`, `state`,
`waited_min`, `deadline_min`, `since`).

### Journal: `journal.jsonl`

Fields: `entry_id` (`jn-`…), `ts`, `kind` (`decision`, `ruled_out`, `suspicion`,
`intention`), `by` (`machinery`, `model`, `human`), `text`, `cites` (ledger and spool
event ids), `status` (`open` or `closed`), `subject` (a step, or empty), `closes` (the
entry this line settles, or null). An entry is never rewritten: closing one is a new line
naming it. Open entries are what a wake is shown.

### Inbox: `inbox.jsonl`

A question or notice: `entry_id` (`in-`…), `ts`, `kind` (`question` or `notice`),
`subject`, `question`, `evidence` (lines), `options` (each `id`, `action`, `text`,
`cost_core_h`, `args`), `cites`. An answer is a line with `answers` (the question's id),
`option` (the option's id) and `by` (a user name, or `model:<name>` for a wake), and the
same `subject`. A question with no answer line is open, and its branch is parked.

### The loop's record: `.myrmidon/loop.json`

One JSON object, replaced atomically: `handled` (spool event ids), `attempts` and
`extensions` per step, `runs` (step → the ledger events of the calls the loop made for
it), `dropped`, `stopped`, `deadline_from`, `budget_notified`, `refused` (step → finding
ids), `until_closed`, and `in_flight` (`action`, `step`, `arm`, `tool`, `started`,
`event`) while a call is being made — the next tick reconciles it against the ledger and
the job store. `.myrmidon/wakes/<inbox id>-<6 hex>.json` holds each wake's `context`,
its `context_sha256` and the `reply`.

## Transcript (format 1)

A session's transcript is `<id>.jsonl`. `myrmidon run` writes it into the study it starts in
(`.myrmidon/transcripts/`), or under `<studies_root>/.sessions/runs/` with no study open; a
chat session writes it under `<studies_root>/.sessions/<id>/`, beside its `state.json`.

Every event has `ts` (ISO 8601, UTC, to the second) and `kind`; the fields below are the
rest. A session opens with `trace`, then `system`. Each question is `user`, then the work,
then `episode_end`. A resumed chat session appends to the same file, opening again with
`trace` and then `resumed`.

### The model's requests

Every request to the model is recorded verbatim. `model_calls()` in
`myrmidon.agent.transcript` rebuilds each one — the full message list, the tools, the
settings — by concatenating `append`s and resolving schema hashes, and refuses a trace that
does not add up.

#### `trace`

Fields: `format`, `resumed`, `model`, `endpoint`, `adapter`, `versions`.

What served this session: the transcript format, whether this is a resume, the model as the
endpoint names it, the endpoint's name (null: the provider's own server), the adapter making
the calls (`PydanticCompletion` for a real model), and `versions` (`myrmidon`,
`pydantic-ai-slim`).

#### `tool_schemas`

Fields: `sha256`, `schemas`.

A set of tool definitions offered to the model, written the first time the session offers
it; `model_call.tools` names it by `sha256` (of the set as sorted-key JSON). Each schema is
`{"type": "function", "function": {"name", "description", "parameters"}}`, the parameters a
JSON Schema.

#### `model_call`

Fields: `call`, `turn`, `model`, `model_s`, `append`, `tools`, `settings`, `response`, `error`.

One request. `call` counts every request in the session, `turn` the model replies of the
current question, the failed one included. `append` holds the messages added to the history
since the previous request, in OpenAI chat form (`role` `system`, `user`, `assistant` with
optional `tool_calls`, or `tool` with `tool_call_id`) — every text the model was given is in
some request's `append`. `tools` is a `tool_schemas` hash. `settings` holds the sampling
parameters and, on a session's first turn, `tool_choice: "required"`; never the endpoint's
address or key. `model_s` is the request's wall time.

Exactly one of `response` and `error`. `response` has `content` (text, or null),
`tool_calls` (`id`, `name`, `arguments` as a JSON string), `reasoning` (what a reasoning
model wrote first, or null; recorded, never sent back), `finish_reason` (`stop`,
`tool_calls`, `length`), `provider_finish_reason` (the server's own word) and `usage`
(`prompt_tokens`, `completion_tokens`, or null). `error` is the failure's text with any key
replaced by `[api_key]`. A request cut short by Ctrl-C leaves no `model_call`; what it
carried is in the next one.
A request that timed out or met a server error (5xx, 429, a lost connection) is made again
up to twice: each failed attempt is a `model_call` with `error` and the same `turn`, a retry's
`append` is empty, and only the last failure ends the question with `error`.

#### `resumed`

Fields: `n_messages`, `sha256`, `append`, `replace`.

A chat session restored from its `state.json`. Exactly one of `append` (the part of the
restored history the trace does not hold yet — the answer given before the session was left)
and `replace` (the whole history, when it is not the one the trace implies). `sha256` is of
the restored history (sorted-key JSON); `model_calls()` checks it.

### The work

#### `system`

Fields: `content`.

The system prompt the session started with.

#### `user`

Fields: `content`.

A question, as typed.

#### `tools`

Fields: `n_tools`, `schema_chars`.

How many tools the session shows the model and their size as JSON, once per session.

#### `turn`

Fields: `turn`, `prompt_tokens`, `completion_tokens`, `model_s`, `history_chars`, `tool_calls`, `finish_reason`, `opening`, `tools_s`.

One model reply: its tokens (null when the adapter reports none), the request's wall time,
the size of the history sent, how many tool calls it made, and the finish reason. `opening`
is true on the session's forced first turn; `tools_s` is the time its tool calls took.

#### `usage`

Fields: `prompt_tokens`, `completion_tokens`.

The question's token totals so far, after each reply.

#### `tool_call`

Fields: `tool`, `args`, `turn`, `error`.

A tool call the model made. `error` is `bad-args` (the arguments were not JSON; no `args`)
or `unknown-tool` (no such tool); either way the call did not reach the registry, and the
model was told why.

#### `tool_result`

Fields: `tool`, `ok`, `provenance_id`, `obs`, `turn`, `wall_s`, `elapsed_s`, `result_chars`, `truncated`, `failure`, `observation_type`, `tags`.

What the registry returned. `obs` is the whole observation, numbers and findings included —
the model was shown a view of it, capped at 8000 characters (`result_chars` before the cap,
`truncated` if it bit; the exact text is in the next `model_call.append`). `provenance_id`
is the ledger event, or null with no study open. `failure` is why the registry did not run
the tool (`invalid_params`, `no_workspace`, `gate_refused`, `raised`), null when it ran.
`observation_type` is the observation's class; `tags` are the tool's. `wall_s` is the call's
time, `elapsed_s` the time since the question.

#### `researcher_decision`

Fields: `tool`, `findings`, `approved`, `approval`, `turn`.

A person at the keyboard decided findings only a person may accept. `approval` is the ledger
event of a yes, absent for a no.

#### `grounding_bounce`

Fields: `ungrounded`.

The answer cited numbers no tool returned; the model was asked again.

#### `assistant`

Fields: `content`.

An answer attempt (a reply with no tool calls), before the grounding check.

#### `truncated`

Fields: `content`.

The reply hit the output-token limit; what came back is not an answer.

#### `error`

Fields: `error`.

The model request failed (the same text as `model_call.error`); the question ends.

#### `interrupted`

Fields: `n_turns`.

The researcher pressed Ctrl-C; the question is dropped, the session kept.

#### `model_change`

Fields: `model`, `endpoint`.

The researcher switched the model or endpoint; it applies from the next question.

#### `episode_end`

Fields: `stop_reason`, `n_turns`, `grounding_ok`.

How the question ended: `COMPLETED`, `MAX_TURNS`, `TOKEN_BUDGET`, `WALL_CLOCK`,
`MODEL_ERROR`, `TRUNCATED` or `INTERRUPTED`; `grounding_ok` is null without an answer.
