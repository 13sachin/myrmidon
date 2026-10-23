"""The documentation corpus: engine documentation the agent can ask when it is not sure.
Built by `scripts/build_docs_corpus.py`, read here.

Every passage is a verbatim excerpt with the page it came from, the section, the engine
version it documents and the page's sha256, so a citation the agent gives is checkable and
a value it quotes is one a tool returned. The corpus is filtered by the versions `doctor`
found: an option whose default changed between versions must never come back from the
wrong one. A page whose licence does not let its text ship is present as pointers only —
section titles and the URL — and a pointer is still an answer.

Search is keyword overlap over tokens, with two additions:
an exact hit on the section's own name (the mdp option asked about) outranks everything,
and the version filter.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

_STOPWORDS = frozenset(
    ["the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
     "are", "was", "were", "be", "been", "at", "by", "from", "this", "that"]
)


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9_.-]+", text.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


BUILTIN_DOCS = Path(__file__).parent / "docs"


class Passage(BaseModel):
    source: str  # gromacs | cpmd | mimic | mimicpy | tutorial
    version: str
    page: str
    section: str
    text: str  # verbatim excerpt; empty when the licence allows pointers only
    locator: str  # URL with fragment, the citation
    sha256: str = ""
    licence: str = ""

    def citation(self) -> str:
        return f"{self.source} {self.version}, {self.section} — {self.locator}"


def _major(version: str) -> str:
    """'2021.6' -> '2021'; 'unknown' -> ''. Documentation is per release series."""
    m = re.match(r"(\d{4}|\d+)", version or "")
    return m.group(1) if m else ""


class DocStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else BUILTIN_DOCS
        self._passages: list[Passage] | None = None

    def all(self) -> list[Passage]:
        if self._passages is None:
            found: list[Passage] = []
            if self.path.is_dir():
                for file in sorted(self.path.glob("*.jsonl")):
                    for line in file.read_text().splitlines():
                        if line.strip():
                            found.append(Passage.model_validate(json.loads(line)))
            self._passages = found
        return self._passages

    def versions(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for p in self.all():
            out.setdefault(p.source, set()).add(p.version)
        return out

    def series_for(self, installed: dict[str, str] | None) -> dict[str, tuple[str, str]]:
        """For every documented source: the installed version and the documented release
        series the search will use — the installed one when the corpus has it, else the
        nearest. An engine newer than the corpus gets the newest documentation there is,
        and the caller says so; refusing would leave the agent with nothing."""
        out: dict[str, tuple[str, str]] = {}
        for source, versions in self.versions().items():
            have = (installed or {}).get(source)
            if have is None:
                continue
            majors = {_major(v) for v in versions if _major(v)}
            want = _major(have)
            if not want or want in majors or not majors:
                out[source] = (have, want)
            else:
                nearest = min(majors, key=lambda m: abs(int(m) - int(want)))
                out[source] = (have, nearest)
        return out

    def search(self, query: str, *, limit: int = 3, source: str | None = None,
               installed: dict[str, str] | None = None) -> list[Passage]:
        """The best passages for a question. ``installed`` is engine -> version as
        `doctor` found them; a source whose installed version is known is filtered to
        its release series (or the nearest documented one, see ``series_for``), and a
        source that is not installed is dropped unless the caller names it."""
        probe = _tokens(query)
        if not probe:
            return []
        names = {t.replace("_", "-") for t in probe}
        series = self.series_for(installed) if installed is not None else {}
        scored: list[tuple[float, Passage]] = []
        for p in self.all():
            if source and p.source != source:
                continue
            if installed is not None and not source:
                if p.source not in series:
                    continue
                _, use = series[p.source]
                if use and _major(p.version) and _major(p.version) != use:
                    continue
            section_tokens = _tokens(p.section)
            body = _tokens(p.text)
            score = float(len(probe & body))
            section_name = p.section.replace("mdp option ", "").replace("_", "-").lower()
            if section_name in names:
                score += 100.0  # the option asked about, by name
            elif probe & section_tokens:
                score += 10.0 * len(probe & section_tokens)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [p for _, p in scored[:limit]]


def excerpt(text: str, query: str, width: int = 700) -> str:
    """The part of a long passage nearest the query's words, so three passages stay
    under the token cap a tool answer allows."""
    if len(text) <= width:
        return text
    probe = _tokens(query)
    words = text.split()
    best, best_at = -1, 0
    for i in range(0, len(words), 20):
        window = " ".join(words[i:i + 120]).lower()
        hits = sum(1 for t in probe if t in window)
        if hits > best:
            best, best_at = hits, i
    piece = " ".join(words[best_at:best_at + 120])
    return ("… " if best_at else "") + piece + (" …" if best_at + 120 < len(words) else "")
