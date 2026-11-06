# Changelog


## 0.6.0 (2026-11-04)

First public release: an agent harness for molecular simulation. Your coding agent connects
to Myrmidon over the Model Context Protocol and drives GROMACS, CPMD coupled to GROMACS
through MiMiC, and MiMiCPy through typed, gated and recorded tools, on a workstation or a
SLURM cluster.

- **Tools for the whole path.** GROMACS preparation, runs and analysis (`fetch_pdb`,
  `gmx_pdb2gmx`, `gmx_editconf`, `gmx_solvate`, `gmx_genion`, `gmx_make_ndx`, `gmx_grompp`,
  `gmx_mdrun`, `observable_measure`, `observable_measure`, `observable_measure`); protein–ligand complexes
  (`structure_select`, `ligand_parametrize` with acpype GAFF/GAFF2 and AM1-BCC or Gasteiger
  charges and a PDB chemical-component lookup, `complex_assemble`); MiMiC QM/MM (QM region,
  CPMD input, coupled launch, diagnosis); MiMiCPy force matching. Building a system runs in
  the session and books no compute.
- **A mandate the researcher approves.** A study that spends core-hours is written as
  `mandate.yaml` (`mandate_write`) in a research proposal's shape: the question; the system
  (files, chemical subsystems read from the topology, the Hamiltonian); the method as
  physical stages (`minimize`, `dynamics`, `single_point`, `parameter_fit`); the decisions
  locked in the researcher's own words and the model's proposals with their sources; the
  deliverables as physical observables of named subsystems; the cost estimate and ceiling;
  and the recovery bounds. The researcher approves it once: in Pi's dialog, with `/approve`
  in Hermes, or with `myrmidon approve`. The agent has no way to approve its own.
- **One gate on every booking.** `gmx_mdrun`, `mimic_run`, `fm_fit`, and `script_run` above
  the session envelope are checked against the approved mandate: the price within its
  ceiling, every input traced back to the mandate's system files through the ledger, and
  the locked decisions as the run input on disk states them. Each job writes a receipt on
  the compute node with the hashes of the inputs it read, verified when it is collected.
- **Run inputs compiled, every value sourced.** `.mdp` and CPMD inputs are compiled from
  stage templates (`gromacs.minimize`, `gromacs.dynamics`, `gromacs.single_point`,
  `cpmd.minimize`, `cpmd.dynamics`, `cpmd.single_point`), methods (`amber`, `charmm36`,
  `blyp_mt`, `mimic`), facts read from the system's own files, and the mandate's stage
  parameters. A `.myrmidon.json` sidecar beside each file names every value's source, so a
  new molecule needs no new input file.
- **Walltimes from measured pace.** Each completed run records its throughput (ns/day from
  GROMACS, s/step from CPMD) for the machine; later stages and mandate estimates are sized
  from it, and a job left unsized fills the node it pays for.
- **Physics checks and the engines' manuals.** Modelling pitfalls are versioned rule records
  with their sources, raised as findings on the built system and on rendered run inputs.
  `knowledge_search` searches the GROMACS `.mdp` options and command pages, the CPMD
  keywords and the MiMiC QM/MM rules, and cites each passage.
- **Every study on record.** Every call lands in the study's append-only ledger with its
  command line, engine versions, inputs and outputs by checksum. `myrmidon report` writes
  `report.md` from the ledger, each value naming the event and file it was parsed from;
  `myrmidon status`, `summary` and `tick` follow a study from any terminal without a model.
- **Open models through Pi.** Every endpoint in `endpoints.toml` is registered as a Pi
  provider when the session starts, so `myrmidon --model ENDPOINT/MODEL` runs a study on
  any OpenAI-compatible server. Claude Code, Qwen Code, Goose, Cursor and Codex are wired
  too.
- **Hermes, in testing.** A session plugin (`/approve`, `/approve yes`, `/status`, `/plan`,
  `/inbox`, `/answer`) and a model-provider plugin for the endpoints of `endpoints.toml`.
  Launching Hermes on an open model through `myrmidon --model` is not yet verified; use Pi
  for open models.
