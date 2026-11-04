# Getting started

Five steps from nothing to a finished study. Each step says what to run and what you should
see. A step that fails points to [Troubleshooting](#troubleshooting).

## 1. Install Myrmidon

```bash
conda create -n myrmidon python=3.11 -y && conda activate myrmidon
git clone https://github.com/13sachin/myrmidon myrmidon && cd myrmidon && pip install ".[analysis]"
myrmidon --version
```

Python 3.12 works too, unless you use MiMiCPy.

## 2. Install the engines and the agent

**On a workstation**, conda-forge has GROMACS and the ligand tools (AmberTools, acpype and
Open Babel) in one line, into the same environment:

```bash
conda install -c conda-forge gromacs ambertools acpype openbabel -y
```

**On a cluster**, use your site's modules. Put the `module load` lines in a script (one per
line, never piped); step 3 asks for its path, and every batch job sources it.

Myrmidon is tested with GROMACS 2021.6. Newer releases are expected to work but are not yet
verified; if a tool reports that it could not read a GROMACS program's output, please report
it as an issue with the GROMACS version (`myrmidon doctor` shows it).

**The agent client** is Pi:

```bash
npm install -g @earendil-works/pi-coding-agent
```

Claude Code, Qwen Code, Goose, Cursor and Codex work too; see [Other clients](#other-clients).

## 3. Configure the machine and your model

```bash
myrmidon setup --client pi --endpoint openrouter \
    --api-base https://openrouter.ai/api/v1 --key-env OPENROUTER_API_KEY
```

`setup` finds the scheduler, the engines and the agent, and asks only what it cannot find:
on a cluster the engine script, the partition and account, and where studies live (a
filesystem the compute nodes see). It writes three files under `~/.config/myrmidon/`:

| file | holds |
| --- | --- |
| `config.toml` | the machine: engines, scheduler, where studies live |
| `endpoints.toml` | your model servers (no secrets) |
| `secrets.env` | your keys, `chmod 600` |

Open `secrets.env` and fill in the key on the line `setup` wrote:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

**Any OpenAI-compatible server works**: an institutional vLLM, a local Ollama, Blablador.
Give it a name with `--endpoint`, or add it to `endpoints.toml` by hand:

```toml
[endpoints.cluster_vllm]
api_base = "https://llm.example.org/v1"
key_env = "CLUSTER_LLM_KEY"
context_window = 262144     # optional: what the server serves; Pi assumes 128k otherwise
```

Keys go only in `secrets.env`, never in `.bashrc` (batch jobs export it) or on a command line.

## 4. Check

```bash
myrmidon doctor       # should show ✓ gromacs, ✓ acpype, and pi as the agent
myrmidon endpoints    # each endpoint, and whether its key is set
```

## 5. Run a study

Make a project directory and start Pi in it on your model, as `ENDPOINT/MODEL`:

```bash
mkdir ~/trpcage && cd ~/trpcage
myrmidon --model openrouter/nvidia/nemotron-3.5-super-vl
```

The first time, Pi asks whether to trust the project's files: answer **yes**. Then ask for
the study.

### Example 1: a solvated peptide (minutes on a workstation)

> Set up Trp-cage (PDB 1L2Y) in water with Amber99SB-ILDN and TIP3P, neutralize,
> minimize, and run 10 ps of NVT at 300 K. Report the mean temperature and the protein
> backbone RMSD. Budget: 2 core-h.

### Example 2: a protein–ligand complex

Needs the ligand tools of step 2. The system is about 33,000 atoms: on one 48-core cluster
node it takes about twenty minutes and 15 core-h; on a workstation the 1 ns of production
takes hours.

> Set up the T4 lysozyme L99A–JZ4 complex (PDB 3HTB) in water with Amber99SB-ILDN for the
> protein and GAFF2 with AM1-BCC charges for the ligand. Equilibrate it with the protein
> and ligand restrained, then run 1 ns of unrestrained production at 300 K and 1 bar.
> Report the ligand heavy-atom RMSD in its pocket and the minimum protein–ligand distance
> over the production run. Budget: 64 core-h.

### What happens

1. **The agent builds the system** in the session: it fetches the structure, builds the
   topology, the box, the water and the ions. This costs no compute.
2. **It writes the mandate and stops.** A dialog shows what will run, what it costs and
   your budget as the ceiling. **Approve** it, or say what to change.
3. **It runs the jobs**, each checked against what you approved, and collects them as they
   end. You can close the terminal; `myrmidon` in the directory picks the study up again.
4. **It reports the numbers**, each parsed from the engine's output.

Afterwards:

```bash
myrmidon status     # the mandate and its spend, the jobs, what waits on you
myrmidon report     # report.md: every value with the event and file it came from
```

A study is a directory plus its ledger, `myrmidon.ledger.jsonl`. Copy it anywhere;
`myrmidon report` rebuilds the report from the ledger alone.

## Next

- **QM/MM and force matching**: [TUTORIAL.md](TUTORIAL.md) walks acetone in water from the
  classical box through MiMiC QM/MM to a force-matching fit. It needs CPMD with MiMiC, a
  MiMiC-enabled GROMACS and `mimicpy` (`myrmidon doctor` lists them).
- **What a study directory, its ledger and its mandate hold**: [FORMATS.md](FORMATS.md).

## Troubleshooting

| what you see | what to do |
| --- | --- |
| Pi shows no Myrmidon tools | Answer yes when Pi asks to trust the project, or start once with `pi --approve`. `/mcp` in the session lists the server. |
| `doctor` shows GROMACS or acpype missing | Activate the `myrmidon` environment, or on a cluster check the engine script `setup` recorded (`env_script` in `config.toml`). |
| `endpoints` says the key is not set | Fill in the `export` line in `~/.config/myrmidon/secrets.env`. |
| `--model` is not found in Pi | Use the endpoint's name from `endpoints.toml`, then the model id the server lists: `myrmidon endpoints --json --refresh` shows both. |
| A tool refuses a call | The refusal names the problem and what to change; the agent reads it and adjusts. |
| A job seems stuck | `myrmidon status` in the project, or `myrmidon watch` to follow it line by line. |

`myrmidon workshop` lists every other command.

## Other clients

`myrmidon` starts the first installed of Pi, Hermes, Claude Code, Qwen Code and Goose;
`myrmidon setup` asks when more than one is installed, and `--client NAME` picks another
once. In clients other than Pi, approve the mandate with `myrmidon approve` at a terminal.

```bash
myrmidon --client claude                  # Claude Code, this once
myrmidon init-project . --client qwen     # Qwen Code; the project remembers it
myrmidon mcp install --client goose       # Goose reads only its user config; once per machine
myrmidon mcp install --client hermes      # Hermes likewise
```

Cursor and Codex are wired by `myrmidon init-project . --client cursor|codex` and opened
by hand. **Hermes support is in testing**: its plugin approves in the chat (`/approve`,
then `/approve yes`), but launching it on an open model through `myrmidon --model` is not
yet verified, so use Pi for open models.

## Cluster notes

- `setup` reads each partition's node size and billing from the scheduler, so a job is
  priced as its partition bills it. Run `setup` again when the machine changes.
- One setting caps any single submission whatever was approved: `block_core_h` in
  `config.toml`.
- MiMiC needs equal CPMD ranks per node: `coupled_launch = "multi-prog"` with
  `srun_args = ["--distribution=cyclic:cyclic"]` places them so under SLURM.
