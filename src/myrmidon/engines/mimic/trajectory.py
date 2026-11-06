"""What a coupled run leaves of its frames: CPMD's ENERGIES and TRAJECTORY read, a stored step
written as the GEOMETRY file a later stage starts from, and the steps it shares with the frames
of the GROMACS half (tests/fixtures/cpmd/acetone_coupled.*).

TRAJECTORY holds every atom, QM and MM, each line ``step x y z vx vy vz`` in atomic units, at
steps 1, 1 + N, 1 + 2N for a stride N; the GROMACS half numbers the same frames 0, N, 2N. GEOMETRY
is one such frame without the step.
"""

from __future__ import annotations

from pathlib import Path


def gromacs_step(cpmd_step: int) -> int:
    """The step the GROMACS half numbers a CPMD step's frame with."""
    return cpmd_step - 1


def parse_energies(text: str) -> dict[int, float]:
    """Step -> the ionic temperature in K, from ENERGIES (``step ekinc tempp ...``)."""
    out: dict[int, float] = {}
    for line in text.splitlines():
        cols = line.split()
        if len(cols) >= 3:
            try:
                out[int(cols[0])] = float(cols[2])
            except ValueError:
                continue
    return out


def trajectory_steps(path: Path) -> list[int]:
    """The steps TRAJECTORY stores, in order."""
    steps: list[int] = []
    with path.open() as fh:
        for line in fh:
            first = line.split(None, 1)[0] if line.strip() else ""
            if first.isdigit() and (not steps or steps[-1] != int(first)):
                steps.append(int(first))
    return steps


def frame_rows(path: Path, step: int) -> list[list[float]]:
    """Each atom's x, y, z, vx, vy, vz at a stored step; empty when it is not stored."""
    rows: list[list[float]] = []
    wanted = str(step)
    with path.open() as fh:
        for line in fh:
            cols = line.split()
            if len(cols) >= 7 and cols[0] == wanted:
                rows.append([float(v) for v in cols[1:7]])
    return rows


def write_geometry(rows: list[list[float]], path: Path) -> None:
    """A frame as CPMD's GEOMETRY file: positions then velocities, one atom a line."""
    path.write_text("".join(
        "".join(f"{v:20.12f}" for v in row[:3]) + "  " + "".join(f"{v:20.12f}" for v in row[3:])
        + "\n" for row in rows))
