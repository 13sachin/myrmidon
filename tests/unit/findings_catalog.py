"""Every finding id the shipped code can raise, read from the source.

Input sets, protocols and skills name findings. A name nothing can raise is a pointer to
nowhere, so the tests check each one against this catalogue: ids given to a
ValidationWarning (directly or through fm_fit's `note` helper), and the ids of
mimic_diagnose's failure signatures.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "myrmidon"


def known_finding_ids() -> set[str]:
    from myrmidon.engines.mimic.diagnostics import _CPMD_FAILURE_PATTERNS

    ids = {failure_id for failure_id, _pattern, _text in _CPMD_FAILURE_PATTERNS}
    for path in SRC.rglob("*.py"):
        ids |= set(re.findall(r'\b(?:id=f?|note\()"([A-Z][A-Z0-9_]+)"', path.read_text()))
    return ids
