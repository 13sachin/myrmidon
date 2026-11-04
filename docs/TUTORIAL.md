# Tutorial: QM/MM and force matching

One system, acetone in water, from the MiMiC project's tutorial inputs: a classical box, a
coupled QM/MM trajectory, reference forces, and a force-matching fit. Everything happens
in the chat: you ask, the agent shows you a mandate with its cost, you approve, and it
reports back.

## What you need

| stage | engines (`myrmidon doctor` lists them) |
| --- | --- |
| 1. Classical box | GROMACS |
| 2. QM/MM dynamics | GROMACS, MiMiC-enabled CPMD, `mimicpy` |
| 3. Reference forces | GROMACS, MiMiC-enabled CPMD, `mimicpy` |
| 4. Force-matching fit | GROMACS, `mimicpy` |

Installation and model setup: [GETTING_STARTED.md](GETTING_STARTED.md).

## Get the inputs and start

```bash
mkdir acetone && cd acetone
curl https://gitlab.com/mimic-project/mimic-docs/-/archive/main/mimic-docs-main.tar.gz?path=docs/tutorials/AcetoneTutorial \
  -o AcetoneTutorial.tar.gz
tar --strip-components=3 -zxvf AcetoneTutorial.tar.gz     # AcetoneTutorial/data/
myrmidon --model openrouter/nvidia/nemotron-3.5-super-vl
```

`AcetoneTutorial/data/` holds a single acetone molecule (`acetone.gro`, `acetone.itp`,
`acetone.top`), the pseudopotential table `pp_info.dat`, and the MiMiC tutorial's own
solvated box and run inputs.

Each request below states a budget. It becomes the ceiling of the mandate you approve, and
the approval shows the agent's estimate beside it. Nothing runs past the ceiling: a job that
would go over it is refused and the question comes back to you.

## 1. The classical box

> Take the acetone tutorial inputs in AcetoneTutorial/data, build the acetone-in-water
> box, minimise it, and run 200 ps of NPT equilibration at 300 K and 1 bar.
> Budget: 16 core-h.

**You approve** a minimisation and a 200 ps NPT run.
**You get** the equilibrated box, with its temperature, pressure and density.

## 2. QM/MM dynamics

> From this equilibrated box, run the staged QM/MM: anneal, check energy conservation in
> NVE, heat to 300 K, then run 1000 steps of production at 300 K. The QM region is the
> acetone molecule; acetone is neutral and closed-shell. Budget: 240 core-h.

State the charge and spin of the QM region yourself, as the last sentence does: Myrmidon
never guesses chemistry.

**You approve** four coupled runs: annealing, an NVE check, heating, production.
**You get** the energy drift of the NVE check, the temperatures of each run, and a
production trajectory with its dipole.

If the agent proposes restarting from the tutorial's `GEO_295K` file, say no: it belongs
to the tutorial's own solvated box, not the one you built, and CPMD stops on it.

## 3. Reference forces

> Take 20 frames from the production trajectory and compute single-point QM/MM reference
> forces for force matching. Budget: 60 core-h.

**You approve** 20 single points. The agent runs the first frame alone and checks that it
converged before it runs the rest.
**You get** a force-matching dataset of the 20 frames.

## 4. The force-matching fit

> Fit the force field to that dataset, charges first and then bonded terms, and compare
> it with the original topology on the same held-out frames. Budget: 64 core-h.

**You approve** the fit and the comparison runs.
**You get** the fitted charges and bonded terms beside the original ones, the force error
and geometry of both topologies on the held-out frames, and a warning for anything
unphysical (a collapsed angle, a drifted net charge).

## Read it back

Ask the agent for the report, or from a terminal in the project:

```bash
myrmidon report     # report.md: every value with the file it came from
```
