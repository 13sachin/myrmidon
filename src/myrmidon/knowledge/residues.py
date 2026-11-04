"""What a residue name is, where residuetypes.dat is silent: lipids, cofactors, ions."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

FILE = Path(__file__).parent / "residues.yaml"


@cache
def residue_classes() -> dict[str, str]:
    """residue name (upper case) -> the subsystem kind it belongs to."""
    data = yaml.safe_load(FILE.read_text())
    return {name.upper(): kind for kind in ("membrane", "cofactor", "ions")
            for name in data.get(kind, [])}
