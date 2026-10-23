#!/usr/bin/env python3
"""Mock CPMD: prints a BOMD-style step table, and behaves like the real one about
restarts and about the temperature the input asks for.

Restart, as CPMD 4.3 does it: an input that asks to RESTART reads a file
named ``RESTART`` in the working directory and dies in ZHRWF with "FILE NOT FOUND"
when it is not there. A run that ends writes ``RESTART.1`` (a different name),
LATEST, ENERGIES and GEOMETRY, which is what the next segment needs.

Temperature: the ionic temperature follows the &CPMD block the way the real program's
would in shape — a TEMPERATURE RAMP climbs from its first value towards its second at its
rate (here: kelvin per MD step, the mock's own unit), BERENDSEN or NOSE holds its target,
ANNEALING cools by its factor each step, and no thermostat keeps the temperature it started
with. The mock restart carries the final temperature (``T_FINAL``), so a segment that
continues from it continues from where the last one ended, which is what makes a stage
extension mean something on the mocks. The header prints the ION DYNAMICS line the real
log prints, so the drift parser knows an NVE run from a thermostatted one.

Five step lines are printed at NFI = MAXSTEP·k/5, so a log says how many steps it ran
without being MAXSTEP lines long.

Failures: MOCK_CPMD_FAIL=scf produces a non-convergence failure; =lsd the multiplicity
error; =segv a segfault; =shake the SHAKE-at-start abort (reproduces); =node_once exits
143 with the log cut mid-step the first time it runs under a stage and not again (a
machine fault, which a rerun of identical inputs does not reproduce; every submission
executes in its own directory under the stage, so the marker sits one level up).
"""

import os
import re
import sys
import time

time.sleep(0.2)
fail = os.environ.get("MOCK_CPMD_FAIL")
if fail == "lsd":
    # real CPMD 4.3 behaviour: MULTIPLICITY in &SYSTEM without LSD in &CPMD
    print("    100.0% CP    GGA_XC_BLYP")
    print("  MULTIPLICITY            1  INCOMPATIBLE WITH LDA")
    with open("LocalError-0-0-0.log", "w") as fh:
        fh.write(" process id's: 0, 0, 0, 0, 0\n process stops in file: sysin_utils.mod.F90\n"
                 "               at line: 1513\n               in procedure: sysin\n"
                 " error message: Multiplicity requires local spin density approximation (LSDA)\n")
    sys.exit(15)
if fail == "segv":
    print(" ATRHO| CHARGE(R-SPACE):   24.000000 (G-SPACE):   24.000000")
    sys.stderr.write(
        "[node:1:0:1] Caught signal 11 (Segmentation fault: Sent by the kernel at address (nil))\n"
        "==== backtrace (tid:1) ====\n"
        " 0 0x000000000003fb30 __GI___sigaction()  :0\n"
        " 1 0x0000000000028b15 mimic_short_range_mp_compute_sr_potential_electrons_()  :0\n"
    )
    sys.exit(139)
if fail == "scf":
    print("       1  0.00050  300.0  -17.123456  -17.120000")
    print(" ! SELF-CONSISTENCY NOT REACHED IN 100 STEPS")
    sys.exit(1)
if fail == "shake":
    # real CPMD 4.3 + MiMiC: a restart whose GEOMETRY belongs to another solvation
    print(" PROGRAM CPMD STARTED")
    print("       NFI   TEMPP           EKS      ECLASSIC         DIS    TCPU")
    print(" NEW_SHAKE| PCG DID NOT CONVERGE AFTER SHAKE_CG_ITER =   100 ITERATIONS")
    sys.stderr.write("application called MPI_Abort(comm=0x84000005, 1) - process 0\n")
    sys.exit(1)
_ONCE = os.path.join("..", ".mock_failed_once")
if fail == "node_once" and not os.path.exists(_ONCE):
    with open(_ONCE, "w") as fh:
        fh.write("the node fault happened here once\n")
    print(" PROGRAM CPMD STARTED")
    print("       NFI   TEMPP           EKS      ECLASSIC         DIS    TCPU")
    print("         1     0.1     -17.12345     -17.12045   0.557E-04   71.89")
    print("        INFR         GEMAX            ETOT         DETOT     TCPU")
    print("           1     1.000E-04      -25.723456    -5.587E-06     1.22")
    sys.stderr.write("slurmstepd: error: *** STEP 1234.0 ON node CANCELLED ***\n")
    sys.exit(143)

fm_pace = None
inp_text = ""
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    inp_text = open(sys.argv[1]).read()


def _section(name):
    m = re.search(r"&" + name + r"\b(.*?)&END", inp_text, re.S)
    return m.group(1) if m else ""


def _value_after(block, keyword):
    """The line after a keyword line of a &CPMD-style block, split into numbers."""
    lines = [ln.strip() for ln in block.splitlines()]
    for i, ln in enumerate(lines):
        if ln.upper().startswith(keyword) and i + 1 < len(lines):
            try:
                return [float(x) for x in lines[i + 1].split()]
            except ValueError:
                return None
    return None


cpmd = _section("CPMD")
maxstep = _value_after(cpmd, "MAXSTEP")
n_steps = int(maxstep[0]) if maxstep else 5
ramp = _value_after(cpmd, "TEMPERATURE RAMP")
nose = _value_after(cpmd, "NOSE IONS")
berendsen = _value_after(cpmd, "BERENDSEN IONS")
anneal = _value_after(cpmd, "ANNEALING IONS")
restarting = "RESTART" in inp_text.upper() and bool(re.search(r"^\s*RESTART", inp_text,
                                                              re.M | re.I))

# a RESTART line means the wavefunction is read from the file LATEST names; without it
# CPMD stops in ZHRWF before the first step (real signature, see the module docstring)
t_start = None
if restarting:
    # CPMD reads RESTART and writes RESTART.1 — two names: segment N's RESTART.1 is
    # segment N+1's RESTART, byte for byte.
    if not os.path.exists("RESTART"):
        print(" PROGRAM CPMD STARTED")
        with open("LocalError-0-0-0.log", "w") as fh:
            fh.write(" process id's: 0, 0, 0, 0, 0\n"
                     " process stops in file: rv30_utils.mod.F90\n"
                     "               at line: 174\n"
                     "               in procedure: ZHRWF\n"
                     " error message: FILE NOT FOUND\n")
        sys.stderr.write("application called MPI_Abort(comm=0x84000005, 999) - process 0\n")
        sys.exit(999 % 256)
    for ln in open("RESTART"):
        if ln.startswith("T_FINAL"):
            t_start = float(ln.split()[1])

if ramp:
    t0, t1, rate = (ramp + [0.0, 0.0, 0.0])[:3]
    start = t_start if t_start is not None else t0

    def temperature(step):
        return min(t1, start + rate * step) if t1 >= start else max(t1, start - rate * step)
    control = "TEMPERATURE RAMP"
elif nose:
    target = nose[0]

    def temperature(step):
        return target + 0.3 * ((step * 7919) % 11 - 5) / 5.0
    control = "TEMPERATURE CONTROL (NOSE-HOOVER THERMOSTATS)"
elif berendsen:
    target = berendsen[0]
    start = t_start if t_start is not None else target

    def temperature(step):
        return target + (start - target) * (0.5 ** (step / max(1.0, n_steps / 5)))
    control = "BERENDSEN-STYLE THERMOSTAT"
elif anneal:
    factor = anneal[0] if anneal else 0.95
    start = t_start if t_start is not None else 300.0

    def temperature(step):
        return start * factor ** (2 * step)
    control = "SIMULATED ANNEALING OF IONS"
else:
    start = t_start if t_start is not None else 0.5

    def temperature(step):
        return start + 0.02 * ((step * 7919) % 11 - 5) / 5.0
    control = "ION DYNAMICS:          THE TEMPERATURE IS NOT CONTROLLED"

if "FORCE_MATCHING" in inp_text:
    with open("FMTRAJECTORY.json", "w") as fh:
        for step in (1, 2):
            fh.write('{\n  "time_step": %d,\n  "atoms": [\n' % step)
            fh.write('    {"region": 1, "id": 1, "coordinate": [0.0, 0.0, 0.0], '
                     '"force": [0.001, 0.0, 0.0], "electric_field": [0.0, 0.0, 0.0], '
                     '"electric_potential": 0.0, "charge": 4.0, "hirshfeld_charge": 0}\n')
            fh.write("  ]\n}\n")
print(" PROGRAM CPMD STARTED")
print(" TIME STEP FOR IONS:                                 10.0000")
print(f" {control}")
print(" NFI      GEMAX       CNORM           ETOT        DETOT      TCPU")
print("   1     5.153E-05   1.2E-03      -17.130293     0.000E+00    31.66")
print("       NFI   TEMPP           EKS      ECLASSIC         DIS    TCPU")
nfis = sorted({max(1, round(n_steps * k / 5)) for k in range(1, 6)})
t_final = None
for nfi in nfis:
    # a temperature is never below zero, and a "-0.0" would read as an energy column
    t_final = max(0.0, temperature(nfi))
    # a conserved energy that drifts by 1e-8 Ha per step: 4e-5 Ha/ps at a 10 a.u. step,
    # the order a sound NVE run shows
    e = -17.123456 + 0.00000001 * nfi
    print(f"      {nfi:4d}  {t_final:8.1f}     {e:.5f}     {e + 0.003:.5f}   0.557E-04   71.89")
    print("        SHAKE CONVERGED AFTER   3 ITERATIONS. TCPU =         0.00")
    print("        INFR         GEMAX            ETOT         DETOT     TCPU")
    for it in range(1, 13):
        print(f"           {it}     {1e-4 / it:.3E}      {e - 8.6:.6f}    -5.587E-06     1.22")
    print("        SUBSYSTEM TEMPERATURES:")
    print(f"        T(QM) =                                              {t_final:.2f}")
    print(f"        T(MM) =                                              {t_final * 0.15:.2f}")
# what a finished run leaves for the next segment
with open("RESTART.1", "w") as fh:
    fh.write(f"MOCK CPMD RESTART\nT_FINAL {t_final:.3f}\n")
with open("LATEST", "w") as fh:
    fh.write("{:<1024}\n{:>12}\n".format("./RESTART.1", 1))
with open("ENERGIES", "w") as fh:
    for nfi in nfis:
        fh.write(f"   {nfi}   0.00000000   {max(0.0, temperature(nfi)):.3f}  -17.12345678\n")
with open("GEOMETRY", "w") as fh:
    fh.write("  0.0  0.0  0.0  0.0  0.0  0.0\n")

print(" PROGRAM CPMD ENDED AT:   Thu Jan  1 12:00:00 2026")
print(" CPU TIME :    0 HOURS  0 MINUTES  1 SECONDS")
