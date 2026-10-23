#!/usr/bin/env python3
"""Propose input set drafts from input files that already exist.

    scripts/distil_protocol.py DIR [DIR ...] --out drafts/ [--kind mdp]

Reads .mdp, CPMD .inp and mimicpy fminput files under DIR, groups the ones that set
the same keys, and writes one candidate YAML per group. Every value in a draft was
read from a file and the draft names the files and their hashes; nothing is invented.

Nothing is installed. The drafts are for a person to read, answer the TODOs in, and
move into an input sets/ directory themselves — the harness proposes, a human merges.

Read-only on the inputs, and symlinks are not followed: a research tree's shared
force field must not be walked into.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrmidon.knowledge.distil import (  # noqa: E402
    MAX_RECIPE_KEYS,
    collect,
    distil,
    write_drafts,
)

KINDS = ("mdp", "cpmd_inp", "fminput")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="directories or files to read")
    ap.add_argument("--out", type=Path, required=True, help="directory to write drafts into")
    ap.add_argument("--kind", choices=KINDS, action="append", help="limit to these kinds")
    ap.add_argument(
        "--max-keys",
        type=int,
        default=MAX_RECIPE_KEYS,
        help="skip files with more keys than this — an engine echo, not a recipe",
    )
    ap.add_argument(
        "--min-files",
        type=int,
        default=1,
        help="skip groups smaller than this (a recipe used once is often a one-off)",
    )
    args = ap.parse_args()

    found = collect(args.roots, set(args.kind) if args.kind else None)
    if not found:
        print("no .mdp, CPMD .inp or fminput files found", file=sys.stderr)
        return 1

    written = []
    for kind, paths in sorted(found.items()):
        all_drafts, skipped = distil(paths, kind, max_keys=args.max_keys)
        drafts = [d for d in all_drafts if d.n_files >= args.min_files]
        print(
            f"{kind}: {len(paths)} file(s) -> {len(drafts)} draft(s)"
            + (f", {len(skipped)} skipped" if skipped else ""),
            file=sys.stderr,
        )
        for path, reason in skipped[:10]:
            print(f"  skipped {path}: {reason}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more skipped", file=sys.stderr)
        for draft in drafts:
            print(
                f"  {draft.suggested_id}: {draft.n_files} files "
                f"({draft.n_distinct} distinct), {len(draft.constant)} constant key(s), "
                f"{len(draft.varying)} varying",
                file=sys.stderr,
            )
        written += write_drafts(drafts, args.out / kind)

    print(f"\n{len(written)} draft(s) written to {args.out}", file=sys.stderr)
    print("Answer the TODOs before moving any of them into an input sets/ directory.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
