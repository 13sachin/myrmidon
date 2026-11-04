# Myrmidon

**An agent harness for molecular simulation.** Your coding agent drives GROMACS, CPMD/MiMiC
QM/MM and MiMiCPy force matching through Myrmidon's tools: every number is parsed from the
engine, nothing runs until you approve its cost, and every study leaves a ledger of what
ran.

## Install

```bash
# 1. Myrmidon
conda create -n myrmidon python=3.11 -y && conda activate myrmidon
git clone https://github.com/13sachin/myrmidon myrmidon && cd myrmidon && pip install ".[analysis]"

# 2. The engines (workstation; on a cluster your site's modules, which step 4 asks for)
conda install -c conda-forge gromacs ambertools acpype openbabel -y

# 3. The agent client
npm install -g @earendil-works/pi-coding-agent

# 4. The machine and your model endpoint (OpenRouter here; any OpenAI-compatible server works)
myrmidon setup --client pi --endpoint openrouter \
    --api-base https://openrouter.ai/api/v1 --key-env OPENROUTER_API_KEY

# 5. Your key: fill in the line setup wrote in ~/.config/myrmidon/secrets.env
#      export OPENROUTER_API_KEY="sk-or-..."

# 6. Check: GROMACS, acpype and your agent should show as found
myrmidon doctor
```

## Run your first study

```bash
mkdir ~/trpcage && cd ~/trpcage
myrmidon --model openrouter/nvidia/nemotron-3.5-super-vl
```

1. When Pi asks whether to trust this project's files, answer **yes**; until you do, Pi
   shows no Myrmidon tools.
2. Ask:
   > Set up Trp-cage (PDB 1L2Y) in water with Amber99SB-ILDN and TIP3P, neutralize,
   > minimize, and run 10 ps of NVT at 300 K. Report the mean temperature and the protein
   > backbone RMSD. Budget: 2 core-h.
3. The agent builds the system and opens a dialog with the study's mandate: what will run
   and what it costs. **Approve** it. The agent runs the jobs and reports the two numbers.

Afterwards, from any terminal in the project:

```bash
myrmidon status    # where the study stands
myrmidon report    # the report, every value with the file it came from
```

Next: a protein–ligand study, cluster setup and troubleshooting are in
**[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**.

## Documentation

| | |
| --- | --- |
| [Getting started](docs/GETTING_STARTED.md) | install, configure, two example studies, troubleshooting |
| [Tutorial](docs/TUTORIAL.md) | acetone in water: a classical box, QM/MM and a force-matching fit |
| [Formats](docs/FORMATS.md) | the study directory, the ledger and the mandate |
| [Changelog](CHANGELOG.md) | what each version contains |

## Commands

| command | what it does |
| --- | --- |
| `myrmidon ["QUESTION"]` | start your agent in this project, wired to Myrmidon |
| `myrmidon setup`, `doctor` | configure the machine; check what was found |
| `myrmidon approve`, `status`, `answer` | approve the study's mandate; where it stands; answer a parked question |
| `myrmidon report` | the study's report, from its ledger |
| `myrmidon workshop` | every other command |

## Development

```bash
pip install -e ".[dev,analysis,agent-pydantic]"
pytest tests/unit                                           # stand-in engines, runs anywhere
MYRMIDON_REAL_ENGINES=1 pytest tests/integration -m gromacs # real GROMACS
```

## Citing and license

[CITATION.cff](CITATION.cff) has the reference (on GitHub: *Cite this repository*).
Licensed under Apache-2.0.
