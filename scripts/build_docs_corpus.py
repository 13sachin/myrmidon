#!/usr/bin/env python3
"""Build the shipped documentation corpus: named pages of engine documentation, fetched,
chunked by heading, hashed, into src/myrmidon/knowledge/docs/<source>-<version>-<page>.jsonl.

The corpus is what `knowledge_search` reads. It is curated and versioned,
never scraped live: every record carries the page's URL and sha256, the section it came
from, the engine version it documents and the licence it ships under. Re-running the script
rebuilds the same records from the same pages and says when a page changed.

Licences decide what ships as text. GROMACS documentation is LGPL-2.1 and ships as text
with the notice in `src/myrmidon/knowledge/docs/NOTICE`. A page whose licence does not
allow that is listed with `text: false` and ships as pointers only: section titles and the
URL, which is still an answer, since it tells the reader where to look.

    scripts/build_docs_corpus.py            # rebuild every page in PAGES
    scripts/build_docs_corpus.py --check    # fetch and compare hashes, write nothing
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "src" / "myrmidon" / "knowledge" / "docs"

#: What is fetched. `kind` names the chunker; `text` whether the licence lets the text ship.
PAGES = [
    {"source": "gromacs", "version": "2021.6", "page": "mdp-options", "kind": "mdp-options",
     "url": "https://manual.gromacs.org/2021.6/user-guide/mdp-options.html",
     "licence": "GROMACS documentation, LGPL-2.1", "text": True},
    {"source": "gromacs", "version": "2021.6", "page": "mimic-qmmm", "kind": "sections",
     "url": "https://manual.gromacs.org/2021.6/reference-manual/special/mimic-qmmm.html",
     "licence": "GROMACS documentation, LGPL-2.1", "text": True},
]

NOTICE = """The files in this directory are excerpts of engine documentation, chunked by
section for `knowledge_search`, each record naming the page it came from and its sha256.

GROMACS documentation (manual.gromacs.org) is distributed under the GNU Lesser General
Public License, version 2.1 (https://www.gnu.org/licenses/lgpl-2.1.html). The excerpts
here are reproduced under that licence; the GROMACS authors hold the copyright.
"""


def _clean(fragment: str) -> str:
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    fragment = html.unescape(fragment).replace("¶", "")
    fragment = re.sub(r"\s*<\S*$", "", fragment)
    return re.sub(r"\s+", " ", fragment).strip()


def chunk_mdp_options(text: str) -> list[tuple[str, str]]:
    """One chunk per mdp option: from its anchor to the next option's (values are nested
    anchors `mdp-value-<option>-<value>` and stay inside their option's chunk)."""
    anchors = [(m.start(), m.group(1))
               for m in re.finditer(r'id="mdp-((?!value-)[A-Za-z0-9_-]+)"', text)]
    out = []
    for i, (pos, name) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(text)
        block = _clean(text[pos:end])
        block = re.sub(r'^id="mdp-[^"]+">\s*', "", block)
        block = re.split(r"\s(?=\S+ Previous Next)", block)[0]
        if len(block) >= 20:
            out.append((f"mdp option {name}", block))
    return out


def chunk_sections(text: str) -> list[tuple[str, str]]:
    """One chunk per section heading of a Sphinx page (h1–h3)."""
    parts = re.split(r"(<h[123][^>]*>.*?</h[123]>)", text, flags=re.S)
    out = []
    for i in range(1, len(parts) - 1, 2):
        title = _clean(parts[i])
        body = _clean(parts[i + 1])
        body = re.split(r"\s(?=\S+ Previous Next)", body)[0]
        if title and len(body) >= 40:
            out.append((title, body))
    return out


CHUNKERS = {"mdp-options": chunk_mdp_options, "sections": chunk_sections}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 — named pages only
        return resp.read()


def build(page: dict, check: bool = False) -> str:
    raw = fetch(page["url"])
    sha = hashlib.sha256(raw).hexdigest()
    target = DST / f"{page['source']}-{page['version']}-{page['page']}.jsonl"
    if target.exists():
        first = json.loads(target.read_text().splitlines()[0])
        state = "unchanged" if first.get("sha256") == sha else "CHANGED"
    else:
        state = "new"
    if check:
        return f"{state:10s} {target.name}"
    chunks = CHUNKERS[page["kind"]](raw.decode("utf-8", "replace"))
    today = dt.date.today().isoformat()
    with target.open("w") as fh:
        for section, body in chunks:
            record = {
                "source": page["source"], "version": page["version"], "page": page["page"],
                "section": section, "text": body[:3000] if page["text"] else "",
                "locator": page["url"] + _fragment(section, page["kind"]),
                "sha256": sha, "fetched": today, "licence": page["licence"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    (DST / "NOTICE").write_text(NOTICE)
    return f"{state:10s} {target.name}: {len(chunks)} chunk(s)"


def _fragment(section: str, kind: str) -> str:
    if kind == "mdp-options" and section.startswith("mdp option "):
        return "#mdp-" + section[len("mdp option "):]
    return ""


def main(argv: list[str]) -> int:
    check = "--check" in argv
    for page in PAGES:
        print(build(page, check=check))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
