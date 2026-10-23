Real MiMiC stage inputs from one staged run of the MiMiC acetone tutorial (CPMD 4.3, MiMiCPy
prepqm), with the run directory in `&MIMIC PATHS` replaced by `/path/to/run`.

- `system_from_prepqm.inp` — what prepqm wrote for the QM selection, with the annealing method.
- `<stage>_method.inp` — the stage's method, rendered from its input set.
- `<stage>_run_real.inp` — the run input that stage actually ran with (prepqm re-run with the
  stage's template, then the charge confirmed).

A stage's input made from the system input and the stage's method must be the real one.
