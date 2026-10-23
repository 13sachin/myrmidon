"""Numeric grounding: every number in the model's answer must trace to data.

The episode builds an index of numbers, each labelled with where it came from:

- ``observation`` — a value in a registry tool's observation: the only source that is a
  measurement the harness made or parsed;
- ``shell`` — output of shell or file tools run by a loop on a registry other than the
  harness's (what it read is what it had);
- ``script`` — the output of ``script_run``: the registry ran it, but the model may have
  written it, so what it prints is not an engine's measurement (a model's own
  reimplementation of molecular mechanics prints numbers that are not grounded);
- ``parameter`` — a numeric (not free-text) argument of a call that succeeded: the model's
  own stated choice, which the registry validated;
- ``prompt`` — a number the user stated in the task.

The final answer is scanned; a number that matches nothing is UNGROUNDED — the loop bounces
the answer back, and `myrmidon run` exits 1 if it is still there after the retries. The
report says which source grounded each number, so a reader can require that the values a
question asks for came from observations, not from the prompt or the model's own arguments.

Never indexed: free-text arguments (where a computed number is smuggled into a call to
make it "grounded"), and identifiers inside observations — event and job ids, hashes, file
sizes, paths, argv, timestamps — whose digits are not quantities.

Small integers (|x| <= 12) are exempt: counts like "two files" or list numbering would drown
the signal. One that no source states is reported as exempt, not as grounded.
"""

from __future__ import annotations

import contextlib
import math
import re

from pydantic import BaseModel, Field

# Units a number is written against with no space: "300K", "5ns", "0.0027Ha/Bohr". The
# unit must end at a word boundary, so identifiers stay identifiers: "amber99sb", "1l2y",
# "spc216", "tip3p" and "t906" carry no quantitative claim (the two 99s in "amber99sb-ildn"
# are not numbers in the answer), while "300K" is a number with its unit and is scanned.
_UNITS = (
    r"K|mK|C|ns|ps|fs|us|µs|ms|s|min|h|d|nm|pm|fm|Å|A|Angstrom|bohr|Bohr|au|a\.u\.|Ha|"
    r"Hartree|eV|meV|kJ|kcal|mol|J|bar|atm|Pa|GPa|MPa|Ry|Rydberg|GB|MB|TB|kB|KB|GiB|MiB|"
    r"M|mM|uM|µM|mL|L|g|kg|Da|kDa|W|%"
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    rf"(?:(?=(?:{_UNITS})(?![A-Za-z0-9_]))|(?![A-Za-z0-9_]))"
)
_SMALL_INT_EXEMPT = 12
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
# "1.224×10⁵", "1.224 x 10^5", "1.224·10^-3" → "1.224e5" before scanning
_SCI_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[×x·*]\s*10\s*(?:\^|\*\*)?\s*([⁻−-]?[\d⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
)
#: observation keys whose values are names, not quantities
_IDENTIFIER_KEYS = frozenset({
    "provenance_id", "event_id", "episode_id", "job_id", "sha256", "size_bytes", "path",
    "paths", "command", "timestamp", "workspace", "cwd", "stage", "git_commit",
})
#: the order sources are credited in when a number matches more than one
SOURCES = ("observation", "shell", "script", "parameter", "prompt")
#: tools whose observation carries what a model-authored program printed
SCRIPT_TOOLS = frozenset({"script_run"})
MEASURED = frozenset({"observation", "shell"})


#: how many times an ungrounded answer is sent back, and what it is told: one text for every
#: driver that can send it back (the native loop, or a client session resumed with it)
GROUNDING_RETRIES = 2
BOUNCE = (
    "GROUNDING CHECK FAILED. These numbers do not match any tool observation from this "
    "session: {bad}. Rewrite your answer using only values actually present in observations "
    "(cite the tool). Derived arithmetic (differences, conversions) counts as ungrounded — "
    "report the raw values instead. Do NOT mention the flagged numbers at all, not even to "
    "explain their removal."
)


def citable(obs: dict) -> dict:
    """What an observation may be cited for. Its next actions are not evidence: a retry
    echoes the model's own refused arguments, which would otherwise become citable. Every
    driver indexes through this, or their grounding is not the same check."""
    return {k: v for k, v in obs.items() if k != "next_actions"}


#: minus signs as typeset: the minus sign, and hyphens and dashes written directly before a
#: digit — "‑4197.5 kJ/mol" with a non-breaking hyphen (U+2011) is the observed value
#: -4197.5, not +4197.5
_MINUS_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2212\uFE63\uFF0D](?=\d|\.\d)")


def normalize_numbers(text: str) -> str:
    """Typographic forms models like → forms the scanner understands.

    Superscripts are translated only inside a ×10 exponent: "4.392³ nm³" is a
    box edge cubed, not the number 4.3923."""
    text = _MINUS_RE.sub("-", text.replace("\u2212", "-"))
    return _SCI_RE.sub(
        lambda m: f"{m.group(1)}e{m.group(2).translate(_SUPERSCRIPTS)}", text
    )


def numbers_in(text: str) -> list[str]:
    """The numeric claims in ``text``, as written."""
    return _NUMBER_RE.findall(normalize_numbers(text))


def _decimals(text_form: str) -> int:
    """Decimal places a written number claims, exponent included: '1.224e5'
    is precise to the hundreds (-2), '3e-4' to 4 places."""
    mantissa, _, exponent = text_form.lower().partition("e")
    places = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return places - (int(exponent) if exponent else 0)


def _significant(text_form: str) -> int:
    """Significant figures a written number claims: '0.0027' two, '300' three, '1e5' one."""
    mantissa = text_form.lower().partition("e")[0].lstrip("-+")
    digits = mantissa.replace(".", "").lstrip("0")
    return len(digits)


class GroundingReport(BaseModel):
    checked: int = 0
    ungrounded: list[str] = Field(default_factory=list)
    # every number the answer states, by the source that grounded it
    by_source: dict[str, list[str]] = Field(default_factory=dict)
    exempt: list[str] = Field(default_factory=list)  # small integers: counts, numbering

    @property
    def ok(self) -> bool:
        return not self.ungrounded

    @property
    def measured(self) -> int:
        """Numbers grounded by a measurement (an observation, or a baseline's shell)."""
        return sum(len(self.by_source.get(s, [])) for s in MEASURED)


class NumericIndex:
    def __init__(self) -> None:
        self._values: dict[str, list[float]] = {s: [] for s in SOURCES}

    def add_value(self, value: float, source: str = "observation") -> None:
        if isinstance(value, bool):
            return
        if math.isfinite(value):
            self._values[source].append(float(value))

    def add_tree(self, obj, source: str = "observation") -> None:
        """Recursively index every number in a dict/list tree (an observation dump) or a
        string, skipping the keys that hold identifiers."""
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            self.add_value(obj, source)
        elif isinstance(obj, dict):
            for key, v in obj.items():
                if key not in _IDENTIFIER_KEYS:
                    self.add_tree(v, source)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self.add_tree(v, source)
        elif isinstance(obj, str):
            # numbers quoted inside observation text (e.g. summaries) count too
            for m in numbers_in(obj):
                with contextlib.suppress(ValueError):
                    self.add_value(float(m), source)

    def add_parameters(self, args) -> None:
        """The numeric arguments of a call that succeeded. Strings are not indexed: a
        free-text argument is where a computed number would be written to launder it."""
        if isinstance(args, bool):
            return
        if isinstance(args, (int, float)):
            self.add_value(args, "parameter")
        elif isinstance(args, dict):
            for v in args.values():
                self.add_parameters(v)
        elif isinstance(args, (list, tuple)):
            for v in args:
                self.add_parameters(v)

    @staticmethod
    def _matches(v: float, x: float, text_form: str) -> bool:
        if v == x or abs(v - x) <= 1e-9 + 1e-6 * abs(v):  # float formatting noise
            return True
        # the answer may round a value to the precision it writes...
        if round(v, _decimals(text_form)) != x:
            return False
        # ...but a claim with one significant figure ('1e5', '0.003') rounds over up to half
        # an order of magnitude: '1e5' matched anything from 50 000 to 150 000. It stays a
        # rounding only within 5 % of the value ('3e-4' for 2.9e-4 is; '1e5' for 6e4 is not)
        if _significant(text_form) <= 1:
            return abs(v - x) <= 0.05 * abs(x)
        return True

    def source_of(self, x: float, text_form: str) -> str | None:
        for source in SOURCES:
            if any(self._matches(v, x, text_form) for v in self._values[source]):
                return source
        return None

    def grounded(self, x: float, text_form: str) -> bool:
        if abs(x) <= _SMALL_INT_EXEMPT and float(x).is_integer():
            return True
        return self.source_of(x, text_form) is not None

    def check(self, text: str) -> GroundingReport:
        report = GroundingReport()
        for token in numbers_in(text):
            try:
                x = float(token)
            except ValueError:
                continue
            report.checked += 1
            source = self.source_of(x, token)
            if source is not None:
                # a small integer an observation states is a measurement like any other
                report.by_source.setdefault(source, []).append(token)
            elif abs(x) <= _SMALL_INT_EXEMPT and float(x).is_integer():
                report.exempt.append(token)
            else:
                report.ungrounded.append(token)
        return report
