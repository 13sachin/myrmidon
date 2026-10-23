# Tutorial: running Myrmidon

This page walks the tool through one system, acetone in water, from a bare molecule to a
coupled QM/MM run and a force-matching fit, in a coding agent's session. It shows how work
is asked for, what the harness does with the request, where it refuses, and what it leaves
behind. It is about the tool, not the system: the inputs are the MiMiC project's own
tutorial files, and what your runs produce is yours to read from your study.

Everything here is one conversation with your agent. You type sentences; the agent calls
tools over MCP; the harness checks, records and runs. From the terminal you only read back.

## What you need

| to run | you need |
| --- | --- |
| the classical stage | GROMACS |
| the coupled QM/MM stages | GROMACS and a CPMD built with MiMiC, and mimicpy |
| the force-matching fit | GROMACS and mimicpy, no quantum program |

`myrmidon doctor` says which of these the machine has. Installation, the machine config and
wiring your agent are in [GETTING_STARTED.md](GETTING_STARTED.md); on a cluster, name the
script that loads the programs as `env_script` and the harness sources it itself.

The inputs come from the MiMiC project:

```bash
curl https://gitlab.com/mimic-project/mimic-docs/-/archive/main/mimic-docs-main.tar.gz?path=docs/tutorials/AcetoneTutorial \
  -o AcetoneTutorial.tar.gz
tar --strip-components=3 -zxvf AcetoneTutorial.tar.gz     # AcetoneTutorial/data/
```

`acetone.gro` there is the bare molecule, ten atoms, with its topology and pseudopotential
table beside it. Everything below starts from that directory.

## 1. A project, and the agent in it

```bash
mkdir acetone && cd acetone && mv ../AcetoneTutorial .
myrmidon client
```

The first time in a directory, `myrmidon client` makes it a **project**: the server entry
your agent connects to, the guidance block that tells it to work through the tools, the
skills it may follow. Then it starts your agent there, Claude Code unless told otherwise,
and plain `claude` in the directory does the same from then on. `myrmidon init-project .`
does the setup without starting anything.

A **study** is one question and the directory that answers it, under the `workspaces_root`
of the machine config: numbered stage folders, and `myrmidon.ledger.jsonl`, the append-only
record of every call. The agent opens one with `study_create` when you first ask for work,
and begins every later session by looking at what ended in it while nobody was watching.

## 2. The classical box, in a sentence

> Take the acetone tutorial inputs in AcetoneTutorial/data, build the acetone-in-water
> box, minimise it, and run the tutorial's equilibration step.

What happens, in order:

1. **It reads the recipe, not its memory.** The `setup-and-equilibrate` skill and
   `protocol_lookup` give it the published sequence, `mimic_qmmm_solvated_solute`, whose
   first steps are this stage. Ask "show me the protocol" and it prints the same thing for
   you.
2. **Run inputs are rendered from input sets, not typed.** `inputset_apply` writes the
   minimisation and equilibration `.mdp` files from `em_solute_in_water` and
   `npt_equilibration_solute_in_water`, and records, for every value in them, whether it
   came from the input set or from you. A value an input set refuses to supply is a fact
   about your system, and the agent asks you for it rather than guessing.
3. **Changing a sourced value is a decision.** If you ask for the velocity seed to be
   pinned so the study replays bit for bit, the tool refuses until the change is
   acknowledged by name:

   ```
   INPUTSET_VALUE_OVERRIDDEN: 1 validated value(s) of npt_equilibration_solute_in_water replaced:
                              gen_seed: -1 -> <your seed>
   ```

   The agent acknowledges it with the id, and the acknowledgement is recorded in the
   ledger with the call. Declaring a value as the thing your study varies (a plan with
   `arms`) turns the same edit into an advisory line, because then it is the experiment.
4. **The box.** `gmx_editconf` defines the cell from a solute-to-wall distance,
   `gmx_solvate` fills it and hands back the topology it edited in its own stage. Handing
   `gmx_solvate` an already solvated box raises `SOLVATE_ALREADY_SOLVATED` instead of
   wrapping a second shell of water around the first.
5. **Submissions end the turn.** `gmx_grompp` then `gmx_mdrun` for the minimisation,
   again for the equilibration. Each `gmx_mdrun` books its core-hours before the queue;
   above the machine config's `warn_core_h` it refuses until `COMPUTE_BUDGET` is
   acknowledged, above `block_core_h` it refuses outright. The agent tells you the job id
   and what it booked, and stops. Close the terminal if you like.
6. **Collection, next time you ask.** `job_status` collects the run: typed outputs by
   role, the energies read from the `.edr` by `gmx_energy`, and the diagnosis if it
   failed. The temperature the agent reports is parsed from the energy file, and an
   answer that states a number no tool returned is bounced back to the model before you
   see it.

## 3. The coupled QM/MM chain

Four coupled runs, each starting CPMD and GROMACS together in one allocation: an annealing,
an NVE test, a heating ramp and a production run, each continuing from the previous one's
restart.

> From this box, run the staged QM/MM: anneal, check energy conservation in NVE, heat to
> 300 K, then produce a trajectory. Acetone is neutral and closed-shell.

The last sentence is the kind of thing only you can say. The tools that need it will not
infer it.

1. **The QM region.** `mimic_select_qm` takes a selection (`add resname is ACT`), a
   padding for the QM cell and the CPMD template rendered from `mimic_cpmd_template`, runs
   mimicpy's `prepqm`, and writes the index group and the CPMD input with the system in it:
   `&ATOMS`, `&MIMIC` with its `OVERLAPS` and `PATHS`, the cell. A padding too small for
   the isolated-molecule Poisson solver is a finding, not a silent default.
2. **The MM half.** `gmx_grompp` with the `mimic_mm_mdp` input set and the QM index group
   builds the `.tpr` for the coupled run.
3. **The method, per stage.** Each stage's `&CPMD` and `&DFT` blocks come from an input set
   (`mimic_cpmd_annealing`, `mimic_cpmd_nve_test`, `mimic_cpmd_heating`,
   `mimic_cpmd_production`); `mimic_prepare` puts them on the system input and **requires
   the total charge and the multiplicity**. The annealing input set carries no `RESTART`
   line, because the first coupled run has nothing to restart from; the three that follow
   do, and `mimic_run` refuses a restarting input with no `restart_from`
   (`CPMD_RESTART_FILE_MISSING`) before anything is booked.
4. **The launch.** `mimic_run` submits both programs as one job, with the ranks and
   walltime as parameters it prices. Every submission executes in a numbered directory of
   its own under its stage, because CPMD writes its log, `RESTART.1`, `LATEST`, `GEOMETRY`
   and `ENERGIES` under fixed names; the continuation names the previous run's `restart`
   artifact as `restart_from`, and the harness copies it under the name CPMD reads.
   `&MIMIC PATHS` is re-pointed to the run's own directory, since that is how the two
   programs find each other.
5. **Reading the log.** `mimic_diagnose` parses a CPMD log into typed facts: steps
   completed against steps requested, the subsystem temperatures `T(QM)` and `T(MM)`, the
   SCF iterations, the energies, and for an NVE run the drift of the conserved quantity.
   A failed run is named by its signature, `CPMD_SHAKE_WRONG_GEOMETRY` for example, with
   the line it matched quoted; a run that ended normally with every step it asked for is
   never failed by a pattern. `job_status` on the ended job returns the same facts.
6. **The heating ramp starts where the annealing ended.** That temperature is a fact about
   your run, so the input set refuses to supply it: the agent reads it from the
   annealing's diagnosis and passes it as the ramp's start, and tells you so.

One trap the input set names for you: the tutorial's `GEO_295K` geometry file belongs to
the pre-solvated box the MiMiC tutorial ships, not to the box you built. Restarting from
it makes CPMD read coordinates for the wrong atoms and abort in SHAKE; the diagnosis names
it. Let the production thermostat bring the system to temperature instead.

## 4. Reference forces and a force-matching fit

Force matching fits a classical force field to the forces of the QM/MM trajectory. Two
parts: single points on frames of the production run, and the fit.

**Single points.**

> Make a force-matching reference from the production trajectory: twenty frames, one
> single point each.

The `fm-reference` skill drives this. `traj_frames` picks the frames, `inputset_apply`
renders the single-point CPMD template and the coupled-run parameters
(`fm_singlepoint_cpmd`, `fm_singlepoint_mm`), and the chain of section 3 runs each frame.
One rule spans two files: the GROMACS output pace must equal CPMD's `FORCE_MATCHING` pace,
and `FM_PACE_MATCHES_NSTXOUT` reports itself as deferred when only one of the two exists,
and resolves when the second lands. The skill runs the first frame alone and reads its
diagnosis before submitting the rest: a single point is a `WAVEFUNCTION OPTIMIZATION`,
where CPMD's step limit bounds SCF iterations, and one that hits the limit exits normally
and writes a reference file that looks like a good one. `wavefunction_converged` is the
field that tells them apart; leave the step limit unset and `CPMD_SCF_LIMIT_TOO_LOW`
catches a small one before submission.

**The dataset.** `fm_merge` joins the frames' reference files; `fm_collect` registers the
dataset, checks that the quantum forces and the classical frames are index-aligned, and
hashes both so a refit after an extended trajectory is a new, linked dataset.

**The fit.**

> Fit a force field to that dataset, hierarchically, and validate it against the stock
> topology on the same frames.

The `fm-fit-and-validate` skill is the recipe. `fm_plan` writes the fitting input and
names every weight nobody set rather than defaulting in silence:

```
FM_FEW_SNAPSHOTS: only 3 snapshot(s) selected: small fits overfit and validate poorly
FM_WEIGHT_DEFAULT_WV: wv not given, using 1.0
FM_WEIGHT_DEFAULT_WE: we not given, using 0.01
```

`fm_fit` runs mimicpy's fit as a job (charges first, then bonded parameters, with a floor
on force constants if you ask for one); `job_status` collects the parsed fit and the
candidate topology, and raises `FM_ANGLE_COLLAPSED`, `FM_BOND_LENGTH_IMPLAUSIBLE`,
`FM_CHARGE_IMPLAUSIBLE`, `FM_FORCE_CONSTANT_IMPLAUSIBLE` or `FM_NET_CHARGE_DRIFTED` on the
parameter table when it has reason to. `fm_validate` puts every candidate, the stock
topology included, against the same reference frames with the same metrics: a classical
rerun for the held-out force error, energy minimisations for the geometry. `fm_report`
writes the report from the ledger: fit settings, the charge and bonded-parameter tables
original against fitted, the validation table, core-hours.

Read the parameter table, not only the scores. A fit can reproduce held-out forces and
still carry an equilibrium angle no molecule has; the findings above are a floor, and the
table is where you see it. One more trap the tool names: keep `[ atomtypes ]` out of the
molecule `.itp`, because mimicpy rewrites that file for the fit (`FM_ITP_DEFINES_ATOMTYPES`).

## 5. The same work, unattended

Everything above can run without anyone at the keyboard. The shape is **plan, approve,
loop, return**: the agent writes a plan and runs nothing; you approve it once; the loop runs
it; you come back to where it stands.

> Plan the staged QM/MM from this box: anneal, NVE check, heat to 300 K, produce. Do not
> run anything yet.

`plan_write` records the steps, the values with their origins, the resources of every
submitting step, what each step must satisfy (**sentinels**: the NVE drift under a limit,
the heating's tail temperature in range), what the loop may do on its own (the
**envelope**: one rerun of identical inputs on a failure with no known signature, one more
segment of a stage whose sentinel fails) and a core-hour ceiling. `plan_check` says
whether it is approvable, which means executable: nothing unresolved, nothing unpriced.
What the plan cannot know, the agent asks you in this conversation, so that nothing is
asked after the approval. Then it stops: approving is yours.

> Approved. Run it.

`plan_approve` records your approval with the findings you accepted under it;
`loop_start` starts the loop, a process that outlives the session the way a submitted job
does. Each pass collects the jobs that ended, evaluates the sentinels on the parsed logs,
continues what is ready, does inside the envelope what you approved, and parks what it
cannot settle as a question with the evidence and the options and their costs. A pass is
crash-only: killed at any instant, the next one picks up what it left. The loop's own test,
`tests/reference_campaign/`, runs this chain, a repeated equilibration and a sweep of single
points end to end on the mock engines.

Come back in the agent with `/loop` or "where is it", or from any terminal, with no model
involved:

```bash
myrmidon status                                 # where the study stands
myrmidon inbox                                  # what waits on you
myrmidon answer QUESTION_ID extend              # your decision; the next pass carries it out
```

When the plan is done, `study_report` writes the report with every value re-parsed from
the runs' own output.

## 6. Reading a study back

Replace `STUDY` with a study directory; `myrmidon status` names the active one.

```bash
myrmidon watch                        # one line per call, refusal, approval and job state change, live
myrmidon summary STUDY --since all    # tools, gates, jobs, core-hours booked and consumed
myrmidon report STUDY                 # the study from its ledger: steps, acknowledgements, hashes, jobs
myrmidon replay STUDY                 # the plan it would re-run; --into DIR --execute re-runs it
```

`replay` reproduces a study from the ledger alone and compares checksums: preparation
files come back identical, simulation output does not, because parallel dynamics is not
reproducible bit for bit, and the comparison says which is which. The ledger is a text file
in the study directory; copy the directory anywhere and all of this still works.

## Without a cluster

Nothing here needs a scheduler. `myrmidon init` without `--backend slurm` runs jobs as
subprocesses with every gate, ledger entry and replay intact, and counts the machine's
cores so a submission that asks for more is flagged. The classical stage and the fit run on
a workstation; the coupled stages want a cluster and a MiMiC build.
