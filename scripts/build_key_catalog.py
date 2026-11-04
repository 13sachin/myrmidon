#!/usr/bin/env python3
"""Build the shipped key catalogs: every key an engine reads, from the engine itself.

GROMACS: the keys and defaults are what `gmx grompp` writes to mdout.mdp for an empty mdp
(tests/fixtures/gromacs/mdout_<version>_defaults.mdp); the obsolete and renamed keys are
the ones grompp's readir.cpp replaces; units and the keys grompp writes only when their
feature is on come from the shipped mdp-options corpus.

CPMD: the keywords, their sections and options are the manual's `\\keyword` entries; the
other spellings CPMD accepts, and the keywords a build accepts that its manual does not
describe, are read from the input parsers that dispatch on them.

    scripts/build_key_catalog.py gromacs --readir SRC/gmxpreprocess/readir.cpp
    scripts/build_key_catalog.py cpmd --cpmd CPMD_SOURCE_TREE --commit HASH
    scripts/build_key_catalog.py … --check     # rebuild and compare, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "src" / "myrmidon" / "knowledge" / "catalog"
DOCS = ROOT / "src" / "myrmidon" / "knowledge" / "docs"
FIXTURES = ROOT / "tests" / "fixtures" / "gromacs"

#: The CPMD sections a MiMiC run writes, and the parser that reads each one.
CPMD_PARSERS = {
    "CPMD": "control_utils.mod.F90",
    "SYSTEM": "sysin_utils.mod.F90",
    "DFT": "dftin_utils.mod.F90",
    "MIMIC": "mimic_wrapper.mod.F90",
    "ATOMS": "ratom_utils.mod.F90",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(key: str) -> str:
    return key.strip().lower().replace("-", "_")


# --- GROMACS --------------------------------------------------------------------------


def mdout_keys(text: str) -> list[dict]:
    """Keys in mdout order, each with its default and the comment heading it sits under."""
    keys, group, after_blank = [], "", True
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            after_blank = True
            continue
        if stripped.startswith(";"):
            if after_blank:
                group = stripped.lstrip("; ").strip().lower()
            after_blank = False
            continue
        after_blank = False
        name, _, value = line.partition("=")
        keys.append({"name": name.strip(), "default": value.strip(), "group": group})
    return keys


def readir_replacements(text: str) -> tuple[list[str], dict[str, str]]:
    """grompp's replace_inp_entry calls: obsolete keys (replaced by nothing) and renames."""
    obsolete, renamed = [], {}
    for old, new in re.findall(r'replace_inp_entry\(inp,\s*"([^"]+)",\s*(nullptr|"[^"]+")\)',
                               text):
        if new == "nullptr":
            obsolete.append(old)
        else:
            renamed[old] = new.strip('"')
    return obsolete, renamed


def corpus_options(path: Path) -> dict[str, dict]:
    """The mdp-options corpus by option name: locator, unit, and default when stated."""
    out = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        name = record["section"].removeprefix("mdp option ")
        if name == "general":
            continue
        head = record["text"][len(name):].lstrip()
        default = re.match(r"\(([^)]*)\)", head)
        unit = re.search(r"^(?:\([^)]*\)\s*)?\[([^\]]+)\]", head)
        out[name] = {"doc": record["locator"], "sha256": record["sha256"],
                     "unit": unit.group(1) if unit else "",
                     "default": default.group(1) if default else None}
    return out


def gromacs_catalog(version: str, readir: Path) -> dict:
    mdout = FIXTURES / f"mdout_{version}_defaults.mdp"
    corpus_path = DOCS / f"gromacs-{version}-mdp-options.jsonl"
    corpus = corpus_options(corpus_path)
    obsolete, renamed = readir_replacements(readir.read_text())
    # the manual's anchors sometimes drop the separator grompp's spelling has (vdwtype)
    by_letters = {_canonical(name).replace("_", ""): name for name in corpus}
    keys: dict[str, dict] = {}
    for key in mdout_keys(mdout.read_text()):
        doc = corpus.get(by_letters.get(_canonical(key["name"]).replace("_", ""), ""), {})
        keys[_canonical(key["name"])] = {
            "name": key["name"], "default": key["default"], "group": key["group"],
            "unit": doc.get("unit", ""), "doc": doc.get("doc", ""),
        }
    gone = {_canonical(k) for k in obsolete} | {_canonical(k) for k in renamed}
    written = {k.replace("_", "") for k in keys}
    for name in sorted(corpus):
        canonical = _canonical(name)
        if canonical in gone or canonical.replace("_", "") in written:
            continue
        doc = corpus[name]
        keys[canonical] = {"name": name, "default": doc["default"], "group": "",
                           "unit": doc["unit"], "doc": doc["doc"], "written": False}
    first_doc = next(iter(corpus.values()))
    return {
        "engine": "gromacs",
        "version": version,
        "sources": [
            {"kind": "engine_output",
             "text": f"mdout.mdp written by gmx grompp {version} for an empty mdp: every key "
                     "grompp reads and its default",
             "locator": f"tests/fixtures/gromacs/{mdout.name}", "sha256": _sha(mdout)},
            {"kind": "engine_source",
             "text": f"GROMACS {version} src/gromacs/gmxpreprocess/readir.cpp: the obsolete "
                     "and renamed mdp entries grompp replaces",
             "locator": "https://gitlab.com/gromacs/gromacs/-/blob/v"
                        f"{version}/src/gromacs/gmxpreprocess/readir.cpp",
             "sha256": _sha(readir)},
            {"kind": "engine_manual",
             "text": f"GROMACS {version} mdp options: units, and the keys grompp writes only "
                     "when their feature is on",
             "locator": first_doc["doc"].split("#")[0], "sha256": first_doc["sha256"]},
        ],
        "obsolete": sorted(_canonical(k) for k in obsolete),
        "renamed": {_canonical(k): v for k, v in sorted(renamed.items())},
        "keys": keys,
    }


# --- CPMD -----------------------------------------------------------------------------


def _groups(text: str, start: int, n: int) -> tuple[list[str], int]:
    """`n` brace groups from `start`; `\\{` and `\\}` are literal braces."""
    out, i = [], start
    for _ in range(n):
        while text[i] != "{":
            i += 1
        depth, j = 0, i
        while True:
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(text[i + 1:j])
        i = j + 1
    return out, i


def _latex(s: str) -> str:
    s = s.replace("\\_", "_").replace("\\&", "&").replace("\\{", "{").replace("\\}", "}")
    s = re.sub(r"\\(?:bf|it|em|tt)\b\s*", "", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = re.sub(r"\\(?:dots|ldots)\b", "...", s)
    return re.sub(r"\s+", " ", s).strip()


def manual_keywords(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"\\keyword\{", text):
        (name, required, optional, args, section), end = _groups(text, m.start() + 8, 5)
        desc = text[end:end + 4000]
        nxt = desc.find("\\keyword{")
        desc = desc[:nxt] if nxt >= 0 else desc
        default = re.search(r"\\defaultvalue\{([^}]*)\}", desc)
        line = text.count("\n", 0, m.start()) + 1
        for sec in re.findall(r"&([A-Z]+)", _latex(section)):
            out.append({"name": _latex(name), "section": sec,
                        "options": " ".join(x for x in (_latex(required), _latex(optional),
                                                         _latex(args)) if x),
                        "default": _latex(default.group(1)) if default else None,
                        "line": line})
    return out


_CALL = re.compile(r"keyword_contains\(\s*line\s*,([^()]*)\)", re.I)
_BRANCH = re.compile(r"^( *)(?:else *if|elseif|if) *\(", re.I)


def parser_keywords(text: str) -> list[tuple[int, list[tuple[str, ...]]]]:
    """The keywords a section parser dispatches on: each branch at the level where it
    checks for &END, with its line and the words (in the parser's order) any one set of
    which selects it."""
    lines = text.splitlines()
    level = next(len(m.group(1)) for line in lines
                 if (m := _BRANCH.match(line)) and "'&END'" in line.upper())
    out = []
    for i, line in enumerate(lines):
        m = _BRANCH.match(line)
        if not m or len(m.group(1)) != level:
            continue
        statement, j = line, i
        while statement.rstrip().endswith("&"):
            j += 1
            statement = statement.rstrip()[:-1] + " " + lines[j].strip().lstrip("&")
        alts = []
        for call in _CALL.findall(statement):
            parts = [p.strip() for p in call.split(",")]
            word = parts[0].strip("'\"").upper()
            kw = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
            kw = {k.strip().lower(): v.strip().strip("'\"").upper() for k, v in kw.items()}
            alts.append(tuple(w for w in (word, kw.get("and")) if w))
            if "alias" in kw:
                alts.append((kw["alias"],))
        if alts and alts[0] != ("&END",):
            out.append((i + 1, alts))
    return out


def cpmd_catalog(cpmd: Path, commit: str) -> dict:
    manual = cpmd / "doc" / "manual" / "manual.tex"
    entries = [e for e in manual_keywords(manual.read_text()) if e["section"] in CPMD_PARSERS]
    branches = {sec: parser_keywords((cpmd / "src" / f).read_text(errors="replace"))
                for sec, f in CPMD_PARSERS.items()}
    named = {(e["section"], frozenset(e["name"].split())) for e in entries}
    keys: dict[str, dict] = {}
    claimed: set[tuple[str, int]] = set()
    for e in entries:
        words = frozenset(e["name"].split())
        aliases = set()
        for line, alts in branches[e["section"]]:
            sets = [frozenset(a) for a in alts]
            if any(s == words for s in sets):
                aliases |= {" ".join(a) for a in alts if frozenset(a) != words}
            # the parser tests fewer words than the manual names (POISSON for POISSON
            # SOLVER), or one of the keyword's options with it (NOSE IONS for NOSE)
            if any(s <= words and (e["section"], s) not in named or words <= s
                   for s in sets):
                aliases |= {" ".join(a) for a, s in zip(alts, sets, strict=True)
                            if s < words and (e["section"], s) not in named}
                claimed.add((e["section"], line))
        keys.setdefault(f"{e['section']}/{e['name']}", {
            "name": e["name"], "section": e["section"], "options": e["options"],
            "default": e["default"], "aliases": sorted(aliases),
            "doc": f"doc/manual/manual.tex:{e['line']}",
        })
    # what the parser accepts and the manual does not describe (a fork's own keywords)
    for sec, found in branches.items():
        for line, alts in found:
            name = " ".join(alts[0])
            if (sec, line) in claimed or f"{sec}/{name}" in keys:
                continue
            keys[f"{sec}/{name}"] = {
                "name": name, "section": sec, "options": "", "default": None,
                "aliases": sorted({" ".join(a) for a in alts[1:]}),
                "doc": f"src/{CPMD_PARSERS[sec]}:{line}",
            }
    version = re.search(r"CPMD (\d+\.\d+)", (cpmd / "configure.sh").read_text())
    tree = f"https://gitlab.com/mimic-project/cpmd/-/tree/{commit}"
    return {
        "engine": "cpmd",
        "version": version.group(1) if version else "unknown",
        "sources": [
            {"kind": "engine_manual",
             "text": "the CPMD manual's \\keyword entries for the sections a MiMiC run writes",
             "locator": f"{tree}/doc/manual/manual.tex", "sha256": _sha(manual)},
            {"kind": "engine_source",
             "text": "CPMD's input parsers, for the other spellings each keyword accepts: "
                     + ", ".join(sorted(CPMD_PARSERS.values())),
             "locator": f"{tree}/src",
             "sha256": hashlib.sha256(b"".join(
                 (cpmd / "src" / f).read_bytes() for f in sorted(CPMD_PARSERS.values())
             )).hexdigest()},
        ],
        "obsolete": [],
        "renamed": {},
        "keys": dict(sorted(keys.items())),
    }


def write(catalog: dict, check: bool) -> str:
    target = DST / f"{catalog['engine']}-{catalog['version']}.yaml"
    text = ("# Generated by scripts/build_key_catalog.py: do not edit.\n"
            + yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True, width=100))
    state = ("new" if not target.exists()
             else "unchanged" if target.read_text() == text else "CHANGED")
    if not check:
        DST.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return f"{state:10s} {target.name}: {len(catalog['keys'])} keys"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["gromacs", "cpmd"])
    ap.add_argument("--version", default="2021.6", help="GROMACS version of the mdout fixture")
    ap.add_argument("--readir", type=Path, help="GROMACS readir.cpp")
    ap.add_argument("--cpmd", type=Path, help="the CPMD source tree")
    ap.add_argument("--commit", default="master", help="the CPMD tree's commit, for locators")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.engine == "gromacs":
        if not args.readir:
            ap.error("gromacs needs --readir")
        catalog = gromacs_catalog(args.version, args.readir)
    else:
        if not args.cpmd:
            ap.error("cpmd needs --cpmd")
        catalog = cpmd_catalog(args.cpmd, args.commit)
    print(write(catalog, args.check))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
