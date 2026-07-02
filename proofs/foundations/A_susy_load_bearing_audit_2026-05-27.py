"""
proofs/foundations/A_susy_load_bearing_audit_2026-05-27.py

SUSY-load-bearing audit — comprehensive grep + classification across
the framework's prediction/theorem/framework-doc tier.

Design doc: an internal working note

For each SUSY-related occurrence, classify:
  LOAD-BEARING (LB) — removing literal SUSY breaks the conclusion
  NAMED-CONVENTION (NC) — SUSY label but conclusion stands as named convention
  HISTORICAL/RETRACTED (H) — in deprecated context
  META — discusses the SUSY adoption itself
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

REPO = Path(__file__).resolve().parents[2]


# ============================================================================
# Configuration
# ============================================================================

# Files to audit (per design doc §2.1)
AUDIT_DIRS_FILES = [
    ("predictions/", "*.py"),
    ("predictions/", "*.md"),
    ("docs/theorems/", "*.md"),
    ("docs/framework/", "*.md"),
    ("docs/parameters/", "*.md"),
    ("docs/audits/registers/", "*.md"),
]
AUDIT_LOOSE_FILES = [
    "docs/honest_assessment.md",
    "docs/master_plan.md",
    "README.md",
]

# Exclusions
EXCLUDE_DIRS = ["_archive", "retracted", "__pycache__", "sessions", "scoping", "open_problems"]

# Search patterns (case-insensitive)
PATTERNS = [
    r"\bMSSM\b",
    r"\bsparticle",
    r"\bsuperpartner",
    r"\bgaugino",
    r"\bsfermion",
    r"\bHiggsino",
    r"\bgravitino",
    r"\bsquark",
    r"\bslepton",
    r"\bneutralino",
    r"\bchargino",
    r"\bsupersymm",
    r"\bSUSY\b",
    r"\bM_SUSY\b",
    r"\btan.?beta\b",
    r"\btan\sβ\b",
    r"\bLayer 5\b",
    r"\bADOPTED-MSSM\b",
]
PATTERN_COMBINED = re.compile("|".join(PATTERNS), re.IGNORECASE)


# ============================================================================
# Classification heuristics
# ============================================================================

def classify_line(line: str, filepath: str) -> str:
    """Heuristic classification of a single line based on contextual keywords.

    LB: load-bearing
    NC: named-convention
    H:  historical/retracted
    M:  meta (about the adoption itself)
    U:  unclassified - needs walk
    """
    low = line.lower()

    # Historical / retracted markers (highest precedence)
    if any(x in low for x in [
        "retracted", "deprecated", "superseded", "stale", "previous version",
        "pre-2026-05-14", "the prior", "former", "snapshot", "obsoleted",
    ]):
        return "H"

    # Meta — discusses the adoption itself
    if any(x in low for x in [
        "adopted-mssm-sb", "literal-particle residue", "literal sparticles",
        "literal mssm particle", "literal susy particle",
        "still adopted", "named convention", "graduates the adoption",
        "matter-content interpretation", "adoption itself",
        "branch a", "branch c", "path e", "candidate d", "a4 ", "a3 ", "a1 ",
        "δb_2", "δb_i", "delta_b",
    ]):
        return "M"

    # Named-convention markers (β values, RG running scheme, comparison)
    if any(x in low for x in [
        "mssm convention", "mssm-named", "mssm benchmark",
        "mssm-norm", "mssm-style running", "mssm-rg", "mssm rg",
        "mssm β", "mssm beta", "β-coefficient", "beta-coefficient",
        "β-coefficient values", "beta coefficient",
        "named convention", "mssm-sb", "mssm β-function",
        "single-regime mssm", "ms-bar", "msbar",
        "mssm running", "mssm rge", "mssm rge running",
        "(33/5, 1, -3)", "(33/5, 1, −3)", "33/5", "b_1 = 33/5",
    ]):
        return "NC"

    # Load-bearing candidates — explicit dependence on literal SUSY
    if any(x in low for x in [
        "sparticle", "superpartner", "gaugino", "sfermion", "higgsino",
        "gravitino", "squark", "slepton", "neutralino", "chargino",
        "susy breaking", "m_susy", "susy-breaking scale", "susy spectrum",
        "literal mssm particle", "physical susy particles",
        "below 10 tev", "tev-scale susy", "tev susy",
        "would falsify", "would eliminate the mssm-rg",
    ]):
        return "LB"

    return "U"  # Unclassified - needs manual walk


# ============================================================================
# Audit walk
# ============================================================================

def walk_files() -> list[Path]:
    """Collect all in-scope files."""
    files = []
    for relpath in AUDIT_LOOSE_FILES:
        p = REPO / relpath
        if p.exists():
            files.append(p)
    for dirpath, glob in AUDIT_DIRS_FILES:
        d = REPO / dirpath
        if not d.exists():
            continue
        for p in d.glob(glob):
            # Apply exclusions
            if any(x in str(p) for x in EXCLUDE_DIRS):
                continue
            files.append(p)
    return sorted(set(files))


def audit_file(filepath: Path) -> list[dict]:
    """Find all SUSY-pattern occurrences in a file with classifications."""
    hits = []
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return hits
    for ln, line in enumerate(content.splitlines(), start=1):
        if PATTERN_COMBINED.search(line):
            klass = classify_line(line, str(filepath))
            hits.append({
                "file": str(filepath.relative_to(REPO)),
                "line": ln,
                "content": line.strip()[:200],
                "class": klass,
            })
    return hits


def main():
    print("=" * 100)
    print("SUSY-load-bearing audit — A_susy_load_bearing_audit_2026-05-27.py")
    print("=" * 100)
    print()

    files = walk_files()
    print(f"In-scope file count: {len(files)}")
    print()

    all_hits = []
    files_with_hits = 0
    for f in files:
        hits = audit_file(f)
        if hits:
            files_with_hits += 1
            all_hits.extend(hits)

    print(f"Files with SUSY-pattern hits: {files_with_hits}/{len(files)}")
    print(f"Total occurrence count: {len(all_hits)}")
    print()

    # Classification breakdown
    klass_counter = Counter(h["class"] for h in all_hits)
    print("Classification breakdown:")
    for k in ["LB", "NC", "M", "H", "U"]:
        name = {
            "LB": "LOAD-BEARING",
            "NC": "NAMED-CONVENTION",
            "M": "META (adoption-itself)",
            "H": "HISTORICAL/RETRACTED",
            "U": "UNCLASSIFIED (needs walk)",
        }[k]
        print(f"  {k:>3}  {name:<32}  {klass_counter.get(k, 0)}")
    print()

    # Per-file hit counts (top 10)
    file_counter = Counter(h["file"] for h in all_hits)
    print("Top 10 files by occurrence count:")
    for filepath, cnt in file_counter.most_common(10):
        # Sub-breakdown per file
        per_class = Counter(h["class"] for h in all_hits if h["file"] == filepath)
        breakdown = "/".join(f"{k}={per_class.get(k, 0)}" for k in ["LB", "NC", "M", "H", "U"])
        print(f"  {cnt:>4}  {filepath:<60}  [{breakdown}]")
    print()

    # LB candidates — highest-priority manual walk
    lb_hits = [h for h in all_hits if h["class"] == "LB"]
    if lb_hits:
        print(f"LOAD-BEARING candidates ({len(lb_hits)} hits) — require manual derivation walk:")
        print()
        # Group by file
        lb_by_file = defaultdict(list)
        for h in lb_hits:
            lb_by_file[h["file"]].append(h)
        for filepath in sorted(lb_by_file.keys()):
            hits = lb_by_file[filepath]
            print(f"  {filepath} ({len(hits)} hits):")
            for h in hits[:6]:  # cap at 6 per file for readability
                print(f"    L{h['line']:>5}: {h['content']}")
            if len(hits) > 6:
                print(f"    ... ({len(hits) - 6} more)")
            print()

    # Unclassified — sample for manual walk
    u_hits = [h for h in all_hits if h["class"] == "U"]
    if u_hits:
        print(f"UNCLASSIFIED occurrences ({len(u_hits)} hits) — sample (10):")
        for h in u_hits[:10]:
            print(f"  {h['file']}:L{h['line']}: {h['content']}")
        if len(u_hits) > 10:
            print(f"  ... ({len(u_hits) - 10} more)")
        print()

    # Files with NO MSSM hits in scope — sanity check
    files_no_hits = [f.relative_to(REPO) for f in files if not audit_file(f)]
    if len(files_no_hits) > 0:
        print(f"Files in scope with NO SUSY-pattern hits: {len(files_no_hits)}")
        print(f"  (predictions/theorems without SUSY-mentions — these are unambiguously")
        print(f"   not load-bearing on SUSY by construction.)")
        print()

    print("=" * 100)
    print("Step 1 (grep + heuristic classification) complete.")
    print("Step 2 (manual derivation walk for LB candidates) — see audit doc.")
    print("=" * 100)


if __name__ == "__main__":
    main()
