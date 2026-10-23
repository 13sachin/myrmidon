# Getting started with Myrmidon

**Path A** works on any machine with GROMACS. **Path B** is
the QM/MM force-matching protocol on a cluster with the MiMiC stack (CPMD + MiMiC-enabled
GROMACS + mimicpy); do Path A first.

Myrmidon is an agent harness: you start your coding agent in a project directory with
`myrmidon client` (or Myrmidon's own agent on a model you choose, with `myrmidon`) and say
what you want. Behind that the model gets a set of simulation tools that validate
inputs, gate expensive runs on a core-hour estimate, and record every call and
artifact hash in a per-study ledger. The agent never types a number — values are parsed
from engine output — and it never spends core-hours you have not approved.

## 1. Install

```bash
conda create -n myrmidon python=3.11 -y      # 3.12 works too, unless you use MiMiCPy
conda activate myrmidon
pip install ".[analysis]"                     # from a clone of this repository
myrmidon --version
```

`[analysis]` brings MDAnalysis (trajectory analysis, the validation drivers). Add
`[agent-pydantic,agent-cli]` for Myrmidon's own agent (`myrmidon`, section 3); you do not
need it to work through an agent client such as Claude Code (`myrmidon client`).

The simulation programs are not installed by any of this: GROMACS, CPMD with MiMiC, and
mimicpy come from their own projects and must already be on the machine.

## 2. Configure the machine (once per machine)

```bash
myrmidon init                                 # workstation: local backend
myrmidon init --backend slurm --partition devel --account <project> \
              --studies-root /scratch/<you>/myrmidon-studies      # cluster
```

This writes `~/.config/myrmidon/config.toml` from a commented template: engine binaries found on
PATH (marked found / not found), where studies live, the backend, SLURM and compute-accounting
blocks. Edit it; every key is explained. It never overwrites an existing config — it prints the
diff instead (`--force` to replace).

Two keys decide what the agent may spend without asking you: `warn_core_h` (above this, a
submission needs your approval by name) and `block_core_h` (above this, it is refused).

If your engines come from `module load`, put the loads in a shell script and name it in the
config as `env_script = "~/.config/myrmidon/env_engines.sh"` (one `module load` per line, never
piped). Myrmidon sources it itself — in the server, in `doctor`, and at the top of every batch
script — so the agent never has to be started from a prepared shell.

```bash
myrmidon doctor            # engines and versions, backend, env_script state, wired clients
myrmidon serve --check     # proves the server finds every engine from a bare shell
```

`serve --check` exits non-zero while *any* configured engine is missing. For Path A only GROMACS
matters: `doctor` showing `✓ gromacs` is enough, and the tools that need a missing engine are
simply not exposed. The other engines are separate installs, not pip dependencies:

- **mimicpy** (force matching, QM-region tools): `pip install -e /path/to/mimicpy` into the
  same environment — the force-matching tools drive the MiMiC project's mimicpy through its
  `fm` command.
- **CPMD + MiMiC-enabled GROMACS**: your site's modules, named in `env_script`.
- **ChimeraX** (figures): optional; `chimerax_bin` in the config.

## 3. Work

### Through your agent client

```bash
cd my-md-project
myrmidon client
```

The first time in a directory it writes what the agent needs and says so:

```
  written    .mcp.json                                  the tools
  created    CLAUDE.md                                  how to use them
  created    .claude/settings.json                      status line, session-start look
  created    .claude/settings.local.json                approve this project's server
  created    .claude/skills/setup-and-equilibrate/SKILL.md   recipes it can follow
```

Then your agent starts in that directory and you ask for what you want. Nothing is hidden
in the command: every one of those files is yours to read and edit, and plain `claude` in
that directory behaves exactly the same — `myrmidon client` adds no flags of its own.

Three habits are worth knowing, because they are what the agent was told to do:

- **It opens the study first.** Every session begins by looking at what is still
  running and what finished while nobody was watching, so you can close the terminal after
  a submission and pick the results up tomorrow.
- **A submission ends its turn.** It tells you the job id and what it booked, and stops.
- **It asks you two things at most**: a physics decision the input set will not make for
  you, and the budget when a run costs more than your `warn_core_h`. Everything else it
  decides from the input set and reports.

Useful variations: `myrmidon client "…"` starts with the request already typed;
`myrmidon client -c` continues this directory's last conversation; `myrmidon client --model
…` picks the client's model. `myrmidon init-project .` does the setup without starting
anything. The server shows the model every tool the machine's engines support; `myrmidon
doctor` prints how many that is and what their schemas cost per turn.

**When the agent is not sure, it asks the manual first.** `knowledge_search` holds the
installed engines' own documentation, versioned and cited by page and section, so an mdp
option or a method question comes back as a quoted passage with its URL rather than a guess
or a third question to you. Ask what `refcoord-scaling` does and the passage comes back
cited.

### Myrmidon's own agent

Name your model endpoint once, beside the machine config, and keep its key in a secrets file
the endpoint names (never the key itself in the endpoint file):

```toml
# ~/.config/myrmidon/endpoints.toml
[endpoints.site]
api_base = "https://host/v1"
key_env = "SITE_API_KEY"
```

```bash
source ~/.config/myrmidon/secrets.env     # `export SITE_API_KEY=…`; chmod 600
cd my-md-project
myrmidon
```

The first time, pick a model from the list the endpoint serves; after that `myrmidon`
starts with the model you used last. Ask in your own words. You see each tool it calls and
what came back; when a run would cost more than your `warn_core_h`, it shows you the finding
and asks — your yes approves that one submission, and nothing else. In the session: `/model`
to change model (remembered), `/status`, `/jobs`, `/plan`, `/help`. Ctrl-C stops the agent
and keeps the session; Ctrl-D or `/exit` saves and leaves; jobs keep running either way.
`myrmidon -c` continues the last session, `myrmidon sessions` lists them, and `myrmidon run
"…"` asks one question without a terminal (for scripts).

### Other clients

`myrmidon client` starts Claude Code unless told otherwise. Qwen Code and Goose are wired
the same way and started the same way:

```bash
myrmidon init-project . --client qwen     # .mcp.json + QWEN.md; the project remembers qwen
myrmidon client --client qwen             # or just `myrmidon client` there from now on
myrmidon mcp install --client goose       # goose reads only its user config; once per machine
myrmidon client --client goose            # `goose session` here, AGENTS.md carries the guidance
```

Qwen Code holds a project's server until you approve it once (in the session when it asks,
or `qwen mcp approve myrmidon` in the project); Goose has no such step. The machine-wide
default is the `client` key in `~/.config/myrmidon/config.toml`; a project's
`.myrmidon.toml` overrides it, `--client` overrides both, and `myrmidon doctor` says which
one won. Cursor and Codex are wired by `init-project --client cursor|codex` and opened by
hand. Only Claude Code gets the status line and the session-start hook; Qwen Code gets the
skills as well (`.qwen/skills/`); every client gets the server and the guidance, which is
what the tools and the gates need.

Qwen Code talks to any OpenAI-compatible endpoint, so an on-site model works: set
`OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL` in the shell you start it from (or a
`modelProviders` entry in `~/.qwen/settings.json`), never in a file the scheduler exports to
every job. With Qwen Code, an open model and this harness, the whole stack is open.

## 4. Path A: a solvated protein to a short NVT run

In the project directory, start your agent (`myrmidon client`, or `myrmidon` for
Myrmidon's own) and ask:

> Take the Trp-cage structure (PDB 1L2Y, model 1), build an amber99sb-ildn/tip3p topology
> ignoring the file's hydrogens, put it in a cubic box with 1 nm padding, solvate,
> neutralize, minimize, and run 10 ps of NVT at 300 K. Report the mean temperature and the
> protein RMSD.

The acetone system has a sentence of its own and a skill that ships with Myrmidon;
[docs/TUTORIAL.md](TUTORIAL.md) walks it start to finish.

What you see in the session, in order: `study_create` opens a study and makes it active;
`fetch_pdb`, `gmx_pdb2gmx`, `gmx_editconf`, `gmx_solvate` and `gmx_genion` build the
system, each reporting what it parsed from the engine (atom count, net charge, box
vectors, solvent molecules); `gmx_grompp` writes the run inputs; `gmx_mdrun` submits the
minimisation and then the dynamics; `job_status` collects each run when it has ended;
`gmx_energy` and `traj_analyze` read the temperature and the RMSD from the energy file and
the trajectory. Three things to notice:

- **The gate refuses, and the agent has to say why it goes on.** Passing `max_warn` to
  grompp hides a physics warning, so `gmx_grompp` refuses until `GROMPP_MAXWARN` is
  acknowledged by id; an `integrator = md` input with no thermostat is refused the same way
  (`MDP_NO_THERMOSTAT`). Without the gate both pass silently. The acknowledgement is recorded in the ledger with the call.
- **Every number comes from a parser.** The temperature is read from the `.edr`, the RMSD
  from the trajectory, and the answer names the tool each came from. An answer that states
  a number no tool returned is bounced back to the model before you see it.
- **The cost is accounted.** A submission books core-hours before it reaches the queue, and
  `session_summary` reports what was consumed. Above the configured `warn_core_h` a
  submission needs an explicit acknowledgement (`COMPUTE_BUDGET`); above `block_core_h` it
  is refused.

Afterwards, from the shell:

```bash
myrmidon summary            # what went through the harness this session and what it cost
myrmidon report <study>     # the study from the ledger: steps, acknowledgments, hashes, jobs
myrmidon replay <study>     # the plan; --into DIR --execute re-runs it through the same tools
```

A study is a directory of numbered stages plus `myrmidon.ledger.jsonl`. Copy it anywhere;
`replay` reproduces the study from the ledger alone.

## 5. Path B: QM/MM force matching with MiMiC

Requires CPMD with MiMiC, a MiMiC-enabled GROMACS and `mimicpy` on the machine (`myrmidon
doctor` lists them). The tools: `mimic_select_qm` (QM region from a selection, charge and
multiplicity are inputs, never inferred), `mimic_prepare` (CPMD input), `mimic_run` (coupled
launch through the backend), `mimic_diagnose` (SCF and energy-conservation checks on real logs),
then the force-matching chain `fm_merge → fm_collect → fm_plan → fm_fit → job_status (the
parsed fit and the candidate topology) → fm_validate → job_status (the tables) → fm_report`.

**[docs/TUTORIAL.md](TUTORIAL.md) walks the whole chain** on the MiMiC project's own
acetone-in-water inputs: the classical box, the coupled QM/MM runs, the reference single
points and a force-matching fit, each through the tools, with the gates in the way and
nothing assumed. `myrmidon fm-report` writes the report of a fit — fit settings, per-atom
and per-atom-type charge changes, changed bonded terms, the validation table, core-hours —
from the ledger.

Cluster notes:

- Put `module load` lines in the `env_script`; batch scripts source it.
- `myrmidon init --backend slurm` reads each partition's node size and billing from the
  scheduler into `[compute.partitions]`, so a booking is priced as the partition it goes to
  bills it. Re-run it when the machine changes; `myrmidon doctor` says when they disagree.
- MiMiC needs equal CPMD ranks per node: `coupled_launch = "multi-prog"` with
  `srun_args = ["--distribution=cyclic:cyclic"]` places them so under SLURM.

## 6. When something goes wrong

- `myrmidon doctor` first. It says which engine is missing and why, whether the env script
  sourced, and which agent clients point at a command that does not exist.
- Every refusal names a finding id, and its message is the diagnosis. `mimic_diagnose` names
  known CPMD/MiMiC failure signatures the same way.
- Ask the agent for `session_summary`, or run `myrmidon summary --since all` yourself: if a
  step is not in it, it did not go through the harness.
- `myrmidon watch` follows a run from another terminal, one line at a time, including the
  engine's own step counter — the difference between "waiting" and "hung".
- A session that has lost the thread: `myrmidon study open` says where the study stands
  and which jobs ended without being collected. That is the same look the agent makes at
  the start of every session.
