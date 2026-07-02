#!/usr/bin/env python3
"""
scripts/validate_citations.py

Operationalises the audit discipline introduced in `master_plan.md` §1
(2026-04-26): every new prediction or theorem doc must consult:

  - The structural uniqueness ledger (`docs/audits/registers/uniqueness_ledger.md`, Rows 1-22)
  - The parameter uniqueness ledger (`docs/parameters/parameter_uniqueness_ledger.md`, P-rows)
  - The structural residue register (`docs/audits/registers/structural_residue_register.md`, R-N)
  - The operator catalog (`docs/operator_sweep/operator_sweep_from_A1.md`, Op N.M)

This script scans target files and reports each file's citation profile.
Files lacking citations to upstream rows / operations are flagged as
WARNINGS (informative, not blocking). The script is intended as a
pre-commit advisory: it surfaces files that may be making claims without
linking them to the audit framework.

USAGE:
  python3 scripts/validate_citations.py                   # scan all default targets
  python3 scripts/validate_citations.py --files <paths>   # scan specific files
  python3 scripts/validate_citations.py --strict          # exit 1 on any warnings

Optional integration as a git pre-commit hook:
  ln -s ../../scripts/validate_citations.py .git/hooks/pre-commit
  (or symlink with --files staged-files filter, see Makefile target if added)

EXIT CODES:
  0 = all targets cite at least one upstream row / operation
  1 = warnings found (with --strict)
  2 = script error (missing dependencies, etc.)
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# Citation patterns
# ============================================================================
#
# Look for any of: "Row 12", "Row P3", "R-7", "Op 4.5", "operator 5.34",
# "operator_sweep_from_A1.md", "uniqueness_ledger", "structural_residue_register",
# theorem doc filenames, etc.

CITATION_PATTERNS = {
    "structural_row": re.compile(r"\bRow\s+(\d+)(?:[a-z])?\b", re.IGNORECASE),
    "parameter_row": re.compile(r"\bP\s*-?\s*(\d+)\b|\bRow\s+P(\d+)\b"),
    "residue": re.compile(r"\bR\s*-\s*(\d+)\b"),
    "operator": re.compile(r"\b(?:Op(?:eration)?|operation)\s+(\d+)\.(\d+)\b", re.IGNORECASE),
    "ledger_doc": re.compile(r"uniqueness_ledger\.md|parameter_uniqueness_ledger\.md|structural_residue_register\.md|operator_sweep_from_A1\.md"),
    "framework_axioms": re.compile(r"framework_axioms\.md"),
    "theorem_doc": re.compile(r"theorem_[A-Za-z0-9_]+\.md"),
}

# Files to scan by default
DEFAULT_TARGETS = [
    "predictions/*_derivation.md",
    "predictions/*.py",
    "docs/theorem_*.md",
    "docs/forward_construction_*.md",
]

# Files that DON'T need citation profiles (utility files, indexes, etc.)
EXEMPT_PATTERNS = [
    re.compile(r"_validate_"),
    re.compile(r"__init__"),
    re.compile(r"_index"),
    re.compile(r"common\.py"),
    re.compile(r"run_predictions\.py"),
]


def is_exempt(path):
    name = path.name
    return any(p.search(name) for p in EXEMPT_PATTERNS)


def scan_file(path):
    """Return a citation profile for a single file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, IOError):
        return None
    profile = {}
    for label, pat in CITATION_PATTERNS.items():
        matches = pat.findall(text)
        profile[label] = sorted(set(str(m) for m in matches if m))
    return profile


def has_any_citation(profile):
    """Return True if profile has at least one upstream citation."""
    if profile is None:
        return False
    return any(profile.get(k) for k in [
        "structural_row", "parameter_row", "residue", "operator",
        "ledger_doc", "theorem_doc"
    ])


def expand_targets(target_globs):
    paths = []
    for glob_pat in target_globs:
        paths.extend(REPO_ROOT.glob(glob_pat))
    return [p for p in paths if not is_exempt(p) and p.is_file()]


def main():
    parser = argparse.ArgumentParser(
        description="Validate citations in framework prediction/theorem files."
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Specific files to scan (default: prediction + theorem files)"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 1 if any file lacks citations (default: just report)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print files that fail the citation check"
    )
    args = parser.parse_args()

    if args.files:
        targets = [Path(f).resolve() for f in args.files]
    else:
        targets = expand_targets(DEFAULT_TARGETS)

    n_total = 0
    n_with_citations = 0
    n_without = 0
    failures = []

    for path in sorted(targets):
        if not path.exists() or is_exempt(path):
            continue
        n_total += 1
        profile = scan_file(path)
        if profile is None:
            continue

        has_cites = has_any_citation(profile)
        if has_cites:
            n_with_citations += 1
            if not args.quiet:
                rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
                cites = []
                if profile["structural_row"]:
                    cites.append(f"struct:{','.join(profile['structural_row'][:5])}")
                if profile["parameter_row"]:
                    cites.append(f"param:{','.join(profile['parameter_row'][:5])}")
                if profile["residue"]:
                    cites.append(f"R-:{','.join(profile['residue'][:5])}")
                if profile["operator"]:
                    cites.append(f"Op:{len(profile['operator'])} ops")
                if profile["ledger_doc"]:
                    cites.append("ledger-doc")
                if profile["theorem_doc"]:
                    cites.append(f"thm:{len(profile['theorem_doc'])}")
                print(f"OK  {rel}  [{' '.join(cites)}]")
        else:
            n_without += 1
            failures.append(path)
            rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            print(f"WARN  {rel}  no upstream citations found")

    print()
    print(f"Scanned {n_total} target files.")
    print(f"  With citations: {n_with_citations}")
    print(f"  Without citations: {n_without}")

    if args.strict and failures:
        print(f"\nSTRICT mode: {len(failures)} file(s) without citations -- exiting 1.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
