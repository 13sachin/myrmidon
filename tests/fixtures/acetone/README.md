# Acetone fixtures — which stage each file enters

The acetone tutorial has four stages (`docs/TUTORIAL.md`); a file here is the
*entry* to one of them, and using the wrong one silently changes the system: the
equilibrated box given as stage A's input solvates an already-solvated system into
22 378 atoms and burns its minimisation walltime. The names say which stage a file enters.

| file | enters | what it is |
| --- | --- | --- |
| `acetone_stageA_entry.gro` | stage A | one acetone molecule, 10 atoms, no solvent and no box — what `gmx_editconf` and `gmx_solvate` are given. Cut from `acetone_equilibrated.gro`'s first residue. |
| `acetone_stageA_entry.top` | stage A | its topology: one ACT, no SOL, no water include |
| `acetone.itp` | both | the molecule's own include file (GAFF) |
| `acetone_equilibrated.gro` | stage B | stage A's **output**: acetone in 506 SPC waters, 1528 atoms, 2.50363 nm box |
| `acetone_equilibrated.top` | stage B | its topology, water included, `SOL 506` |
| `mimic.mdp`, `mimic_fm.mdp`, `template.inp`, `fm_template.inp`, `fm_singlepoint_template.inp`, `sele.dat`, `pp_info.dat` | stages B–D | the QM/MM and force-matching inputs |
| `ffnonbonded.itp` | stage D | the non-bonded parameters a fit rewrites |
| `amber03.ff/` | stage B | `forcefield.itp` (without its `ffbonded.itp` include), `ffnonbonded.itp` and `spc.itp` from the GROMACS 2021 distribution: what `acetone_equilibrated.top` includes, so mimicpy reads the topology without a GROMACS installation |

The published tutorial inputs, hashed against the MiMiC project's own archive, are in
`tests/fixtures/mimic_tutorial/`.
