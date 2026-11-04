"""The documentation corpus: engine documentation, CLI tool reference, CPMD keywords and
selection grammars the agent can query when it is not sure.
Built by `scripts/build_docs_corpus.py`, read here.

Every passage is a verbatim or catalog-derived excerpt with the page it came from, the
section, the engine version it documents and the page's sha256, so a citation the agent
gives is checkable and a value it quotes is one a tool returned. The corpus is filtered by
the versions `doctor` found: an option whose default changed between versions must never
come back from the wrong one.

Search combines BM25 term weighting with field boosts and engine-catalog alias/rename
expansion: an exact hit on the option, keyword, or tool name asked about outranks prose,
and catalog aliases (`vdwtype` -> `vdw-type`, `nstxtcout` -> `nstxout-compressed`,
`ANNEAL` -> `ANNEALING`) expand automatically.
"""

from __future__ import annotations

import json
import math
import re
from functools import cache
from pathlib import Path

from pydantic import BaseModel

_STOPWORDS = frozenset(
    ["the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
     "are", "was", "were", "be", "been", "at", "by", "from", "this", "that"]
)


def _tokens(text: str) -> set[str]:
    raw = {
        t
        for t in re.findall(r"[a-z0-9_.-]+", text.lower())
        if len(t) > 1 and t not in _STOPWORDS
    }
    expanded = set(raw)
    for t in raw:
        if "-" in t or "_" in t:
            expanded.add(t.replace("_", "-"))
            expanded.add(t.replace("-", "_"))
            expanded.add(t.replace("-", "").replace("_", ""))
            for part in re.split(r"[-_]+", t):
                if len(part) > 2 and part not in _STOPWORDS:
                    expanded.add(part)
    return expanded


@cache
def _catalog_aliases() -> dict[str, set[str]]:
    """Map stripped/aliased option names to their canonical dashed/underscored tokens."""
    from myrmidon.knowledge.catalog import load, shipped_versions

    aliases: dict[str, set[str]] = {}
    for engine in ("gromacs", "cpmd"):
        for ver in shipped_versions(engine):
            cat = load(engine, ver)
            for _key, entry in cat.keys.items():
                canon = entry.name.lower().replace("_", "-")
                bare = canon.replace("-", "").replace(" ", "")
                forms = {canon, canon.replace("-", "_"), entry.name.lower()}
                for al in entry.aliases:
                    forms.add(al.lower())
                aliases.setdefault(bare, set()).update(forms)
                for f in list(forms):
                    aliases.setdefault(f, set()).update(forms)
            for old, new in cat.renamed.items():
                old_c = old.lower().replace("_", "-")
                new_c = new.lower().replace("_", "-")
                aliases.setdefault(old_c, set()).update({new_c, new_c.replace("-", "_")})
                aliases.setdefault(old.lower(), set()).update({new_c, new_c.replace("-", "_")})
    return aliases


def _expand_query(tokens: set[str]) -> set[str]:
    amap = _catalog_aliases()
    out = set(tokens)
    for t in list(tokens):
        if len(t) <= 3:
            continue
        if t in amap:
            out.update(_tokens(" ".join(amap[t])))
        bare = t.replace("-", "").replace("_", "")
        if len(bare) > 3 and bare in amap:
            out.update(_tokens(" ".join(amap[bare])))
    return out


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
        self._df: dict[str, int] | None = None
        self._avg_len: float = 100.0

    def all(self) -> list[Passage]:
        if self._passages is None:
            found: list[Passage] = []
            if self.path.is_dir():
                for file in sorted(self.path.glob("*.jsonl")):
                    for line in file.read_text().splitlines():
                        if line.strip():
                            found.append(Passage.model_validate(json.loads(line)))
            self._passages = found
            df: dict[str, int] = {}
            total_len = 0
            for p in found:
                toks = _tokens(p.section) | _tokens(p.text)
                total_len += max(len(toks), 1)
                for t in toks:
                    df[t] = df.get(t, 0) + 1
            self._df = df
            self._avg_len = (total_len / len(found)) if found else 100.0
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
                # Reference sources that do not need a separate binary discovery (e.g.
                # cpmd keywords or mimic/selection grammar when gromacs is present) remain
                # searchable so the agent can always read documented syntax.
                if installed and source in ("cpmd", "mimic", "mimicpy", "tutorial"):
                    newest = sorted(versions)[-1]
                    out[source] = (newest, _major(newest))
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
        """The best passages for a question using BM25 + section/option exact-match boosts."""
        raw_probe = _tokens(query)
        if not raw_probe:
            return []
        probe = _expand_query(raw_probe)
        raw_names = {t.replace("_", "-") for t in raw_probe}
        raw_bare = {t.replace("-", "").replace("_", "") for t in raw_probe if len(t) > 3}
        exp_names = {t.replace("_", "-") for t in probe}
        exp_bare = {t.replace("-", "").replace("_", "") for t in probe if len(t) > 3}

        passages = self.all()
        n_docs = max(len(passages), 1)
        df = self._df or {}
        avgdl = self._avg_len

        series = self.series_for(installed) if installed is not None else {}
        scored: list[tuple[float, Passage]] = []
        for p in passages:
            if source and p.source != source:
                continue
            if installed is not None and not source:
                if p.source not in series:
                    continue
                _, use = series[p.source]
                if use and _major(p.version) and _major(p.version) != use:
                    continue
            section_tokens = _tokens(p.section)
            body_tokens = _tokens(p.text)
            matched = probe & (section_tokens | body_tokens)
            if not matched:
                continue

            dl = max(len(body_tokens), 1)
            score = float(len(raw_probe & body_tokens))
            for t in matched:
                doc_freq = df.get(t, 1)
                idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                tf = (1 if t in body_tokens else 0) + (2 if t in section_tokens else 0)
                bm25 = idf * (tf * 2.2) / (tf + 1.2 * (0.15 + 0.85 * min(dl / avgdl, 1.5)))
                weight = 0.5 if t in raw_probe else 0.25
                score += weight * bm25

            section_clean = (
                p.section.removeprefix("mdp option ")
                .removeprefix("gmx ")
                .removeprefix("CPMD &")
                .replace("_", "-")
                .lower()
            )
            section_leaf = section_clean.split("/")[-1]
            section_bare = section_leaf.replace("-", "").replace(" ", "")
            if (section_clean in raw_names or section_leaf in raw_names
                    or (section_bare and section_bare in raw_bare)):
                score += 100.0  # the option, tool, or keyword asked about directly
            elif (section_clean in exp_names or section_leaf in exp_names
                    or (section_bare and section_bare in exp_bare)):
                score += 60.0  # matched via catalog alias/rename
            # a section title is strong evidence only when it accounts for most of what
            # was asked, so the boost scales with the share of the question it covers: a
            # title holding one word of a long question says far less about the passage
            # than the body of the option that answers it
            elif raw_probe & section_tokens:
                hits = len(raw_probe & section_tokens)
                score += 15.0 * hits * hits / len(raw_probe)
            elif probe & section_tokens:
                hits = len(probe & section_tokens)
                score += 6.0 * hits * hits / len(probe)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [p for _, p in scored[:limit]]


def excerpt(text: str, query: str, width: int = 700) -> str:
    """The part of a long passage nearest the query's words, so three passages stay
    under the token cap a tool answer allows."""
    if len(text) <= width:
        return text
    probe = _expand_query(_tokens(query))
    words = text.split()
    best, best_at = -1, 0
    for i in range(0, len(words), 20):
        window = " ".join(words[i:i + 120]).lower()
        hits = sum(1 for t in probe if t in window)
        if hits > best:
            best, best_at = hits, i
    piece = " ".join(words[best_at:best_at + 120])
    return ("… " if best_at else "") + piece + (" …" if best_at + 120 < len(words) else "")
