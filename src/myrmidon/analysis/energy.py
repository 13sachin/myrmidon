"""gmx energy's .xvg read into terms, and the statistics computed from them."""

from __future__ import annotations

import re

import numpy as np

from myrmidon.core.observation import EnergyTermStats

UNIT_BY_TERM = {
    "temperature": "K",
    "pressure": "bar",
    "density": "kg/m^3",
    "volume": "nm^3",
}


def parse_xvg(text: str) -> tuple[np.ndarray, list[str]]:
    """Return (data, legends) from xvg text; data col 0 is time."""
    legends: list[str] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            m = re.match(r'@\s+s\d+\s+legend\s+"(.+)"', line)
            if m:
                legends.append(m.group(1))
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows), legends


def term_stats(time_ps: np.ndarray, values: np.ndarray, unit: str) -> EnergyTermStats:
    drift = None
    if len(values) > 2 and time_ps[-1] > time_ps[0]:
        slope_per_ps = float(np.polyfit(time_ps, values, 1)[0])
        drift = slope_per_ps * 1000.0  # per ns
    return EnergyTermStats(
        unit=unit,
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        drift_per_ns=drift,
        n_points=len(values),
    )


#: one label of gmx energy's selection menu: "  12  Total-Energy   "
_MENU_ENTRY = re.compile(r"(\d+)\s+([^\s\d][^\s]*)")
_MENU_LINE = re.compile(r"^\s*(?:\d+\s+[^\s\d][^\s]*\s*)+$")


def energy_menu(text: str) -> list[str]:
    """The term labels gmx energy printed, in menu order."""
    labels: list[str] = []
    for line in text.splitlines():
        if _MENU_LINE.match(line):
            labels += [label for _n, label in _MENU_ENTRY.findall(line)]
    return labels
