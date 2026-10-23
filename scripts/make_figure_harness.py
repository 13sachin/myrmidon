#!/usr/bin/env python3
"""Draw the harness figure used in the README.

    python scripts/make_figure_harness.py          # writes docs/figures/harness.{png,svg}

The figure is drawn, not photographed: every count it shows (typed tools, named findings,
failure signatures, input sets, protocols, skills) is read from the code, so it cannot
drift from what ships without the numbers changing. Matplotlib is the only dependency
beyond myrmidon itself; it comes with the ``analysis`` extra.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# embed glyphs as paths so the SVG looks the same without DejaVu Sans installed
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
KNOWLEDGE = ROOT / "src" / "myrmidon" / "knowledge"

INK = "#1b2333"
MUTED = "#5f6778"
BLUE, BLUE_FILL = "#2f5597", "#e9eff9"
SLATE, SLATE_FILL, SLATE_SOFT = "#2c3a5c", "#f5f6fa", "#ccd3e2"
GREEN, GREEN_FILL, GREEN_SOFT = "#2e7d4f", "#eaf5ee", "#b6d8c3"
AMBER, AMBER_FILL = "#b07818", "#fdf4e3"


def counts() -> dict[str, int]:
    """Every number on the figure, read from the code and the knowledge files."""
    sys.path.insert(0, str(ROOT / "src"))
    from myrmidon import basic_tools
    from myrmidon.core.tool import REGISTRY
    from myrmidon.engines.mimic.diagnostics import _CPMD_FAILURE_PATTERNS

    basic_tools.register_all()
    ids = {failure_id for failure_id, _pattern, _text in _CPMD_FAILURE_PATTERNS}
    for path in (ROOT / "src" / "myrmidon").rglob("*.py"):
        ids |= set(re.findall(r'\b(?:id=f?|note\()"([A-Z][A-Z0-9_]+)"', path.read_text()))
    return {
        "tools": len(REGISTRY._tools),
        "findings": len(ids),
        "signatures": len(_CPMD_FAILURE_PATTERNS),
        "inputsets": len(list((KNOWLEDGE / "inputsets").glob("*.yaml"))),
        "protocols": len(list((KNOWLEDGE / "protocols").glob("*.yaml"))),
        "skills": sum(1 for p in (KNOWLEDGE / "skills").iterdir() if p.is_dir()),
    }


def box(ax, x0, y0, x1, y1, *, fc, ec, lw=1.4, r=1.2, z=1):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
        )
    )


def text(ax, x, y, s, *, size=9, color=INK, weight="normal", ha="center", va="center",
         style="normal", z=3):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            style=style, zorder=z, linespacing=1.45)


def badge(ax, x, y, n, color):
    ax.add_patch(Circle((x, y), 1.05, facecolor=color, edgecolor="none", zorder=4))
    ax.text(x, y - 0.03, str(n), fontsize=7.5, color="white", fontweight="bold",
            ha="center", va="center", zorder=5)


def arrow(ax, x, y0, y1, color):
    ax.add_patch(
        FancyArrowPatch(
            (x, y0), (x, y1), arrowstyle="-|>", mutation_scale=11,
            color=color, linewidth=1.2, zorder=3,
        )
    )


def draw(n: dict[str, int]):
    fig, ax = plt.subplots(figsize=(10.4, 8.0))
    ax.set_xlim(1.5, 103)
    ax.set_ylim(2.0, 74.5)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor("white")

    text(ax, 44, 72.6, "Myrmidon: the agent harness", size=16, weight="bold")
    text(ax, 44, 69.3,
         "every action an AI agent takes passes through it, and nothing happens off the record",
         size=9.5, color=MUTED)

    # row labels
    for y, s, c in ((62.0, "THINKS", BLUE), (37.0, "CHECKS\n& RECORDS", SLATE),
                    (10.5, "RUNS", GREEN)):
        text(ax, 5.5, y, s, size=8.5, color=c, weight="bold")

    # 1: the model
    box(ax, 14, 57.5, 74, 66.0, fc=BLUE_FILL, ec=BLUE)
    badge(ax, 26.5, 63.4, 1, BLUE)
    text(ax, 28.6, 63.4, "AI model", size=12, weight="bold", ha="left")
    text(ax, 38.8, 63.35, "your coding agent over MCP, or Myrmidon's own",
         size=9.5, color=MUTED, ha="left")
    text(ax, 44, 60.2,
         "plans each step and reads the typed results back;\n"
         "it is never the source of a number",
         size=9.5)

    arrow(ax, 31, 57.0, 53.2, BLUE)
    text(ax, 29.6, 55.1, "asks for a step", size=8.5, color=MUTED, ha="right")
    arrow(ax, 57, 53.2, 57.0, BLUE)
    text(ax, 58.4, 55.1, "typed results, named findings", size=8.5, color=MUTED, ha="left")

    # 2 to 7: the harness
    box(ax, 14, 22.0, 74, 52.5, fc=SLATE_FILL, ec=SLATE, lw=2.0)
    text(ax, 44, 49.8, "THE HARNESS", size=11.5, weight="bold", color=SLATE)
    text(ax, 44, 47.2, "the one way through: there is no shell on the other side",
         size=9, color=MUTED)

    rows = ((38.9, 45.0), (31.4, 37.5), (23.9, 30.0))
    cols = ((16.5, 43.7), (44.3, 71.5))
    cells = [
        (2, "Tool registry",
         f"{n['tools']} typed tools, one job each:\nall the agent can do"),
        (3, "Validation gates",
         f"{n['findings']} named findings before expensive\nwork; an override is a decision"),
        (4, "Provenance ledger",
         "every call, command, version and\nfile hash; any study replays"),
        (5, "Numeric grounding",
         "every number parsed from program\noutput; ungrounded answers bounce"),
        (6, "Diagnosis by signature",
         f"{n['signatures']} CPMD and MiMiC failure signatures\nwith stable ids, quoting the line"),
        (7, "The loop",
         "plan, approve once, tick: collect,\ncheck, continue, park what needs you"),
    ]
    for i, (num, title, body) in enumerate(cells):
        (x0, x1), (y0, y1) = cols[i % 2], rows[i // 2]
        box(ax, x0, y0, x1, y1, fc="white", ec=SLATE_SOFT, lw=1.1, r=0.8, z=2)
        badge(ax, x0 + 2.2, y1 - 1.7, num, SLATE)
        text(ax, x0 + 4.1, y1 - 1.7, title, size=10, weight="bold", ha="left")
        text(ax, x0 + 4.1, y1 - 4.2, body, size=8.3, color=MUTED, ha="left")

    arrow(ax, 31, 21.5, 18.1, GREEN)
    text(ax, 29.6, 19.8, "validated commands", size=8.5, color=MUTED, ha="right")
    arrow(ax, 57, 18.1, 21.5, GREEN)
    text(ax, 58.4, 19.8, "numbers read from output", size=8.5, color=MUTED, ha="left")

    # 8: engines and machines
    box(ax, 14, 3.5, 74, 17.3, fc=GREEN_FILL, ec=GREEN)
    badge(ax, 24.6, 15.0, 8, GREEN)
    text(ax, 26.5, 15.0, "Simulation programs and machines", size=11, weight="bold",
         ha="left")
    chips = [("GROMACS", 17.0, 29.8), ("CPMD + MiMiC", 30.8, 47.2),
             ("MiMiCPy", 48.2, 58.6), ("MDAnalysis", 59.6, 71.0)]
    for label, x0, x1 in chips:
        box(ax, x0, 9.9, x1, 13.1, fc="white", ec=GREEN_SOFT, lw=1.0, r=0.6, z=2)
        text(ax, (x0 + x1) / 2, 11.5, label, size=8.5)
    box(ax, 27.0, 6.3, 61.0, 9.5, fc="white", ec=GREEN_SOFT, lw=1.0, r=0.6, z=2)
    text(ax, 44, 7.9, "ChimeraX  ·  PDB  ·  PubChem  ·  literature  ·  mock engines",
         size=8.5)
    text(ax, 44, 4.8,
         "batch jobs on a SLURM cluster or subprocesses on a workstation, coupled QM/MM included",
         size=8.3, color=GREEN, weight="bold")

    # 9: knowledge
    box(ax, 77.5, 22.0, 101.5, 52.5, fc=AMBER_FILL, ec=AMBER)
    text(ax, 89.5, 55.0, "KNOWS", size=8.5, color=AMBER, weight="bold")
    badge(ax, 89.5, 49.8, 9, AMBER)
    text(ax, 89.5, 46.6, "Knowledge", size=11, weight="bold")
    text(ax, 89.5, 43.4,
         f"{n['inputsets']} input sets with a source\nfor every value they supply,\n"
         "and a list of the values they\nrefuse to guess for you",
         size=8.5, color=MUTED, va="top")
    text(ax, 89.5, 33.6,
         f"{n['protocols']} protocols compose them;\n{n['skills']} skills say how to\n"
         "approach a class of task",
         size=8.5, color=MUTED, va="top")
    text(ax, 89.5, 26.2, "knowledge lives where it acts:\na gate, a parsed field, a signature",
         size=8.5, color=AMBER, weight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "harness.png", dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / "harness.svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    n = counts()
    draw(n)
    print("docs/figures/harness.{png,svg}: " + ", ".join(f"{k} {v}" for k, v in n.items()))
