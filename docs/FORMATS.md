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
| `NN_name/NN_deffnm/` | one coupled run each: `stage_run` allocates a numbered directory under its stage per submission and never reuses one, since CPMD's output names are fixed |
| `mandate.yaml` | the study's mandate, written by `mandate_write` (below) |
| `<file>.myrmidon.json` | beside an engine input `inputset_apply` rendered: where each of its values came from (below) |
| `report.md` | the study's report, written by `study_report` from the mandate and the trace: the harness's sections, the model's prose after each, and any section of the model's own after them |
| `journal.jsonl` | the journal (below): decisions, ruled-out causes, suspicions, intentions |
| `inbox.jsonl` | parked questions and notices, and the answers to them (below) |
| `NN_name/.receipt-*/` | a job's node receipt (below) |
| `.myrmidon/` | the harness's own state, not study artifacts: the job store, run scratch, `transcripts/` of `myrmidon run` sessions started in the study, `wakes/` (every wake's transcript and its index), `recovery.json` (the continuation under the mandate, below) and `mandates/` (every mandate revision written) |

The manifest is one JSON object. Fields: `format`, `name`, `description`, `created`, `origin`.

- `format`: 2.
- `created`: ISO 8601, UTC, to the second.
- `origin`: the project directory that made the study; absent when none did.

## Mandate: `mandate.yaml`

Written by `mandate_write` only (`file_write` refuses it); its fields are `core/mandate.py`'s
`Mandate`, in a research proposal's order: `schema` (2; a v1 mandate loads and migrates its
stage strings, decisions and deliverables), `question` (verbatim), `system` (`description`,
`files`: each `path`, `kind` — an artifact kind of the contracts, with a topology's local
`#include` `.itp` files added automatically — and `sha256` when written; `subsystems`: name
→ `kind`, `molecules`, `resnames`, `chains`, `resid`, `charge`, read from the topology when
omitted; `hamiltonian`: `mm` method id read from the topology's force field, plus optional
`qm` and `coupling`), `method` (`protocol` as `<id>@v<N>`, `stages`: each a `StageSpec` with
`name`, `kind` — `minimize`, `dynamics`, `single_point`, `parameter_fit` — `regime`,
`starts_from` (the stage whose collected run it continues: by default the one before it, an
earlier one when stages branch, itself to extend a finished run), `ensemble`, `temperature_k`, `pressure_bar`, `duration_ps`, `steps`, `restraints`,
`sampling_ps`, and optional `qm_region`, `scan` and `converge`, compiled on the fly by
`knowledge/stage.py`), `decisions` (`yours`: each `said`, the researcher's words, and
`locks`, a mapping of `<stage>.<field>` or `*.<field>` paths — or a `target` and `keys` —
to a scalar, a list of allowed values or `{min, max}`; `proposed`: each `what`, `locks` or
`target`/`keys`, and `source`), `deliverables` (each `name`, `property` — compiled by
`knowledge/observables.py` into `tool` and `params` — `target`, `reference`, `atoms`,
`align_to`, `reference_frame`, `over` naming the stage whose run is measured, and
`statistics`), `cost` (`estimate_core_h`, `ceiling_core_h`), `recovery` (`reruns`,
`model_fixes`, `deadline_h`), `status` (`ready` or `needs_input`), `questions` (at most
three, while `needs_input`), `revision`, `reason` (why a revision was written) and
`digest`.

The digest is the sha256 of the file's other fields (as sorted JSON) followed by each system
file's path and current sha256, in the order listed: a mandate edited, or a system file
replaced, has another digest. An approval is a ledger event with `tool` `mandate_approve`,
written by the researcher's act (`myrmidon approve`, a client's dialog, Hermes's `/approve`),
with `params` `digest`, `approver`, `via`, `origin`, `revision` and `ceiling_core_h`, the
mandate file and the system files as `inputs`, and `mandate` set to the digest. A mandate
holds while its digest, computed now, is one an approval names. Every revision written is
kept as `.myrmidon/mandates/r<revision>-<first 12 digest characters>.yaml`.

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
| `command` | the argv of the binary that ran, or null |
| `engine_versions` | engine name → version string, as discovered for the session |
| `myrmidon_version` | the harness version that wrote the event |
| `inputs` | artifacts the call read (below) |
| `outputs` | artifacts the call wrote (below) |
| `warnings_acknowledged` | warn-severity finding ids the call ran under, accepted by the person making the call |
| `ok` | whether the tool reports success; a refused call is false |
| `duration_s` | seconds the tool ran; 0 for a refusal |
| `exit_code` | the binary's exit code, or null |
| `log_tail` | what the tool kept of the binary's last lines, or null; tools keep them on failure. A call that failed in the session has no job record, so this is where its words are |
| `kind` | `call` (it did something), `observe` (a look at something that exists: counted apart from the work), `gate_refused` (the validation gate refused it; nothing ran) |
| `knowledge_refs` | ids of the findings the call surfaced: gate warnings, advisories, an input set's pitfalls; for a file `inputset_apply` rendered, the set it came from as `inputset:<id>@v<N>#<sha256 of its YAML>`; and for a job's collection, the submitting call as `collects:<event_id>`, so a run's files lead back to what the submission read |
| `params_schema` | a short hash of the tool's parameter schema, so a reader can tell a tool that has since changed what it takes |
| `mandate` | the digest of the approved mandate (`mandate.yaml`) a call of a tool that books resources was made under, and of the mandate a `mandate_approve` event approves; empty otherwise |

An artifact is an object with fields `path` (relative to the study when it is inside it,
else absolute), `role` (`structure`, `topology`, `mdp`, `tpr`, `trajectory`, `log`, ...),
`sha256` and `size_bytes`.

**An approval** is an event with `tool` `mandate_approve` (above, under the mandate).

The ledger does not hold what a tool returned; the transcript does (`tool_result.obs`).

## The node receipt: `.receipt-<8 hex>/`

A job whose node reads files gets a receipt directory beside its run, named in the job's
spec (`receipt`; `receipt_inputs`, the files the tool names, else those the submitting call
consumed), and their hashes at submission in the job store (`meta.input_hashes`: absolute
path → sha256; `meta.after`: the jobs it waits on). On the node the job script writes
`input_hashes.before` (a `sha256sum` listing of those files, after any dependency it waits
on has ended) before the engine starts, and `execution.json` once it ends: `job` (the
scheduler's id, or the local handle), `started`, `finished` (ISO 8601, UTC) and
`exit_code`. Collection compares `before` with the submission's hashes; a file that differs
— other than into what a job it waited on wrote, by that job's collection in the ledger —
refuses the job's result (`RECEIPT_INPUT_CHANGED`), and the collecting observation carries
the `execution` record. A job refused so is never rerun identically by a tick.

## Rendered engine inputs

Beside every file `inputset_apply` writes (an `.mdp`, a CPMD input) lies a JSON object of the
same name plus `.myrmidon.json`, which `file_check` diffs the file against:

| Field | What it holds |
|---|---|
| `inputset`, `protocol_version` | the stage template rendered (or, outside a mandate's stages, the input set) and its version |
| `stage` | the mandate stage the file was rendered for; null for an input set rendered outside them |
| `method` | the methods it was compiled with, `<id>@v<N>`; null for a set rendered alone |
| `renders` | the file kind: `mdp`, `cpmd_inp`, `fminput` |
| `values` | key → the value written |
| `origin` | key → `template:<id>@v<N>`, `stage:<name>.<field>` (the stage's physical terms), `method:<id>@v<N>`, `inputset:<id>@v<N>` (an input set), `system:<rule>` (read from the system's files), or `user` |
| `owner` | key → whose decision it is, read from its origin: `purpose` (a template's or purpose set's value), `method`, `system`, `researcher` (a mandate stage's terms, an override) |
| `overridden` | key → `{owner, by, why}`: a set that decided another owner's key, and its reason |
| `read` | key → `{rule, files, sha256}`: the study's files a fact was read from, and their hash |
| `engine_defaults` | the key catalog (`gromacs-2021.6`) whose default every key not written takes |
| `overrides` | key → `{inputset, user}`: a stated value the caller changed |
| `invariants` | each rule checked at render: `id`, `status`, `message`, `detail` |
| `sources` | the sets' sources: `kind`, `text`, `locator`, `sha256`, `evidence` |
| `written_by` | `inputset_apply` |

## The continuation's files

`myrmidon tick` carries a study on under its approved mandate one pass at a time and leaves
everything it knows in the study (`orchestration/continuation.py`, `orchestration/spool.py`):
two append-only JSONL files beside the ledger, and its working record under `.myrmidon/`.

### Journal: `journal.jsonl`

Fields: `entry_id` (`jn-`…), `ts`, `kind` (`decision`, `ruled_out`, `suspicion`,
`intention`), `by` (`machinery`, `model`, `human`), `text`, `cites` (ledger event ids),
`status` (`open` or `closed`), `subject` (a stage, or empty), `closes` (the entry this line
settles, or null). An entry is never rewritten: closing one is a new line naming it.

### Inbox: `inbox.jsonl`

A question or notice: `entry_id` (`in-`…), `ts`, `kind` (`question` or `notice`),
`subject`, `question`, `evidence` (lines), `options` (each `id` and `text`), `cites`. An answer is a line with `answers` (the question's id),
`option` (the option's id) and `by` (a user name), and the same `subject`. A question with
no answer line is open, and the study waits on it.

### The continuation's record: `.myrmidon/recovery.json` and `.myrmidon/wakes/index.jsonl`

`recovery.json` is one JSON object, replaced atomically and written before each
action it records. Facts about the study persist across mandates: `handled` (the failed
`job_status` collections dealt with, whoever collected them), `pending` (those of them a
wake has yet to be told of), `collected` (the jobs a tick
collected: only their failures are rerun; one the agent collected is the agent's to fix),
`chain` (the ledger event of each rerun → the submission it repeats) and `shown` (the
collecting ledger events a wake has been told of). The rest belongs to the approved mandate
named by `digest` and starts afresh with a newly approved one: `reruns` (a chain's first submission → the identical reruns made of
it, at most the mandate's `recovery.reruns`), `fixes` (the wakes that carried a failure,
against `recovery.model_fixes`), `granted` (wakes the researcher's `continue` answers
added), `woke` (the agent has been woken under this mandate), `notified` (with no wake model,
the researcher was told the study waits on a session), `parked` (the inbox questions asked
under it), `read` (the answers to them carried out) and `stopped` (the researcher answered
`stop`). A failure from before the study's first mandate was approved is not acted on. A
question offers `continue`, `stop` and `revise` (the deadline's, `stop` and `revise`). Only a
plain failure of a job that waits on nothing is rerun; a cancelled job is left as it is.

Each wake appends two lines to `wakes/index.jsonl`, both with `ts`, `fix` (it carries a
failure) and `prompt_sha256`: `event: "started"` before the agent runs, and `event: "ended"`
with `stop` (how its turn ended), `answer` (its last reply) and `transcript` (the native
loop's transcript in `wakes/`) with its `sha256`.

## Transcript (format 1)

A run's transcript is `<id>.jsonl`. `myrmidon run` writes it into the study it starts in
(`.myrmidon/transcripts/`), or under `<studies_root>/.sessions/runs/` with no study open.

Every event has `ts` (ISO 8601, UTC, to the second) and `kind`; the fields below are the
rest. A run opens with `trace`, then `system`. Each question is `user`, then the work,
then `episode_end`. A loop started from a saved history appends to the same file, opening
again with `trace` and then `resumed`.

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

Fields: `call`, `turn`, `model`, `model_s`, `append`, `tools`, `settings`, `stubbed`, `response`, `error`.

One request. `call` counts every request in the session, `turn` the model replies of the
current question, the failed one included. `append` holds the messages added to the history
since the previous request, in OpenAI chat form (`role` `system`, `user`, `assistant` with
optional `tool_calls`, or `tool` with `tool_call_id`) — every text the model was given is in
some request's `append`. `tools` is a `tool_schemas` hash. `settings` holds the sampling
parameters and, on a session's first turn, `tool_choice: "required"`; never the endpoint's
address or key. `model_s` is the request's wall time. `stubbed`, when present, counts the
tool results sent as their outcome and summary line alone: those from before the current step
of ten model calls (the whole is in the earlier request's `append`).

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

A loop started from a saved history. Exactly one of `append` (the part of the restored
history the trace does not hold yet) and `replace` (the whole history, when it is not the
one the trace implies). `sha256` is of the restored history (sorted-key JSON);
`model_calls()` checks it.

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

Fields: `tool`, `ok`, `provenance_id`, `obs`, `turn`, `wall_s`, `elapsed_s`, `result_chars`, `truncated`, `failure`, `tags`.

What the registry returned. `obs` is the whole observation, numbers and findings included —
the model was shown a view of it, capped at 8000 characters (`result_chars` before the cap,
`truncated` if it bit; the exact text is in the next `model_call.append`). `provenance_id`
is the ledger event, or null with no study open. `failure` is why the registry did not run
the tool (`invalid_params`, `no_workspace`, `gate_refused`, `raised`), null when it ran.
`tags` are the tool's. `wall_s` is the call's
time, `elapsed_s` the time since the question.

#### `assistant`

Fields: `content`.

The answer: a reply with no tool calls.

#### `truncated`

Fields: `content`.

The reply hit the output-token limit; what came back is not an answer.

#### `error`

Fields: `error`.

The model request failed (the same text as `model_call.error`); the question ends.

#### `interrupted`

Fields: `n_turns`.

The researcher pressed Ctrl-C; the question is dropped, the session kept.

#### `episode_end`

Fields: `stop_reason`, `n_turns`.

How the question ended: `COMPLETED`, `MAX_TURNS`, `TOKEN_BUDGET`, `WALL_CLOCK`,
`MODEL_ERROR`, `TRUNCATED` or `INTERRUPTED`.
