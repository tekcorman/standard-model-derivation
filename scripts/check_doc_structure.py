#!/usr/bin/env python3
"""Validate the docs/ tree structure and cross-references.

Two checks:

1. **Top-level docs/ stays curated.** Only canonical entry-points are allowed at
   `docs/*.md`. Everything else must live under a thematic subdir.

2. **No flat `docs/<basename>.md` references survive when the file actually
   lives in a subdir.** After the 2026-05-01 reorg, references to a file's old
   flat path resolve to nothing — they should point at `docs/<subdir>/<file>`.

Exit codes:
- 0 = clean
- 1 = at least one violation
- 2 = usage error

Run with `--strict` in CI / pre-commit to enforce.

Run with `--report` to get a verbose breakdown.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical top-level docs allowed at `docs/*.md`. Everything else must live
# under a subdir (framework/, theorems/, parameters/, audits/, etc.).
ALLOWED_TOP_LEVEL = {
    "README.md",
    "orientation.md",
    "quickstart.md",
    "master_plan.md",
    "honest_assessment.md",
}

DOCS = REPO_ROOT / "docs"


def list_top_level_violations() -> list[Path]:
    if not DOCS.exists():
        return []
    return sorted(
        p
        for p in DOCS.iterdir()
        if p.is_file() and p.suffix == ".md" and p.name not in ALLOWED_TOP_LEVEL
    )


def list_tracked_text_files() -> list[Path]:
    out = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    text_exts = {".md", ".py", ".txt", ".yaml", ".yml", ".json"}
    return [
        REPO_ROOT / rel
        for rel in out
        if Path(rel).suffix in text_exts and "_scratch" not in rel and "retracted" not in rel
    ]


def index_docs_subdir_files() -> dict[str, str]:
    """Map basename → 'subdir/basename.md' for files that live under docs/<subdir>/."""
    index: dict[str, str] = {}
    if not DOCS.exists():
        return index
    for f in DOCS.rglob("*.md"):
        rel = f.relative_to(DOCS).as_posix()
        if "/" not in rel:
            continue  # top-level, not a subdir
        if rel.startswith("an internal working note"):
            continue  # scratch is allowed to be referenced directly
        basename = f.name
        # If multiple subdirs contain the same basename, prefer the first found
        # (deterministic via rglob ordering).
        index.setdefault(basename, rel)
    return index


# Match `docs/<basename>.md` not followed by another `/` — i.e., a flat ref.
flat_ref_pattern = re.compile(r"docs/([A-Za-z0-9_\-]+\.md)")


def list_stale_flat_refs(
    subdir_index: dict[str, str],
) -> dict[str, list[tuple[Path, int]]]:
    """Return {basename: [(citing_file, line_no), ...]} for flat refs that should
    be rewritten to subdir form."""
    violations: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for p in list_tracked_text_files():
        try:
            text = p.read_text()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        if "docs/" not in text:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for m in flat_ref_pattern.finditer(line):
                basename = m.group(1)
                if basename not in subdir_index:
                    continue
                # Skip if the file is at the top of docs/ (still allowed there).
                if (DOCS / basename).exists():
                    continue
                violations[basename].append((p, line_no))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit non-zero on violations")
    parser.add_argument("--report", action="store_true", help="verbose per-violation report")
    args = parser.parse_args()

    top_level_violations = list_top_level_violations()
    subdir_index = index_docs_subdir_files()
    flat_ref_violations = list_stale_flat_refs(subdir_index)

    n_top = len(top_level_violations)
    n_flat = sum(len(v) for v in flat_ref_violations.values())

    if n_top:
        print(f"FAIL: {n_top} non-canonical file(s) at docs/ top level:")
        for p in top_level_violations:
            print(f"  - {p.relative_to(REPO_ROOT)}")
        print("  (Allowed top-level docs:", ", ".join(sorted(ALLOWED_TOP_LEVEL)) + ")")
        print()

    if n_flat:
        print(f"FAIL: {n_flat} stale flat docs/ reference(s) across {len(flat_ref_violations)} basename(s):")
        for basename, sites in sorted(flat_ref_violations.items()):
            print(f"  - docs/{basename} → docs/{subdir_index[basename]} ({len(sites)} citing site(s))")
            if args.report:
                for citing, line_no in sites[:5]:
                    print(f"      {citing.relative_to(REPO_ROOT)}:{line_no}")
                if len(sites) > 5:
                    print(f"      ... and {len(sites) - 5} more")

    if not n_top and not n_flat:
        print("OK: docs/ structure clean.")
        return 0

    if args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
