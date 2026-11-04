# MiMiC acetone-in-water tutorial — the published inputs

Downloaded from the MiMiC project's own documentation repository, with the command the
tutorial itself gives:

    curl https://gitlab.com/mimic-project/mimic-docs/-/archive/main/mimic-docs-main.tar.gz?path=docs/tutorials/AcetoneTutorial -o AcetoneTutorial.tar.gz
    tar --strip-components=3 -zxvf AcetoneTutorial.tar.gz

Tutorial: https://mimic-project.org/en/latest/tutorials/acetone_tutorial.html

These files are the truth the shipped MiMiC input sets are distilled from, and
`test_mimic_inputsets.py` compares each rendered stage against them keyword by keyword, so
a transcription error fails a test rather than a simulation.

| file | what it is | sha256 |
| --- | --- | --- |
| template.inp | the CPMD template prepqm turns into the first QM/MM input | 51e3952370e541dac4f2da20dd526cceba4aa016117daa62878bb73dbe425440 |
| mini.mdp | classical energy minimisation | 937765373404c55f8071a84ec393c487b757a13a0c1ceecfd5325c88c69580a9 |
| heat.mdp | 200 ps classical NPT equilibration | 43fd9b80f72d27f6a365505f3c7732881db5080d2a48987af14609a01e5551f0 |
| mimic.mdp | the GROMACS side of the coupled run | 10cdcb1949bb07d10a1496b14a7932f1f1285fce0005eddfb53fa3edd15de8ad |
| sele.dat | the QM region selection | de91f54e1ad1a37a9ccb18a01ecde60da055427f5ce0571ed445cb06e63c7cf4 |
| pp_info.dat | pseudopotential table | eba0b4f70f33c3e548228c59abfdb6686ca113125f3c98fab6e998dc454811aa |

## What is here and what is not

The archive ships the **template**, not the four dynamics inputs. A reader produces
`annealing.inp` from the template with `mimicpy prepqm`, then edits it into the NVE test,
the heating run and the production run. The per-stage `&CPMD` blocks are printed on the
tutorial page, so the protocol's dynamics stages cite the page rather than a file, and the
test checks them as the template plus their documented differences.

## A discrepancy worth knowing about

For three keys, `template.inp` in the archive and the `annealing.inp` listing on the page
disagree:

| key | archive template | page listing |
| --- | --- | --- |
| ANNEALING IONS | 0.95 | 0.99 |
| TRAJECTORY SAMPLE | 10 | 1 |
| STORE / RESTFILE | 10 / 1 | 100 / 10 |

`prepqm` does not change any of them, so one of the two is ahead of the other. The protocol
takes the archive's values, because those are in a file that can be hashed, and records the
page's in the rationale. Worth reporting upstream.

Only the published tutorial is a source here: older copies of the tutorial in circulation
use **Car-Parrinello** dynamics with a fictitious electron mass and a 5 a.u. timestep,
whereas the published tutorial uses **Born-Oppenheimer** with 10 a.u., and an input set
distilled from such a copy would match nothing a reader of the website runs.
