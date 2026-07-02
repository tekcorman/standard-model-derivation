#!/usr/bin/env python3
"""Orphan-probe scanner: classify every proofs/ probe by citation tier.

Tiers (refines proofs/README.md's 3-tier scheme):
  backbone      -- referenced by verify.py / run_predictions.py (string ref)
  current       -- basename cited by >=1 tracked, non-archived .md doc
  local-only    -- cited only by gitignored lab-notebook docs (internally fine;
                   public-hygiene gap: the public tree carries no citation)
  archive-only  -- cited only by archived docs (predictions/retracted/, proofs/_archive/)
  code-util     -- uncited by docs but imported/referenced by other live code
  orphan        -- cited by nothing anywhere

Run from repo root:

    python3 scripts/orphan_probe_scan.py [--tier orphan]
"""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

ARCHIVED_DOC_PREFIXES = ("predictions/retracted/", "proofs/_archive/")
INFRA = {"common.py", "__init__.py"}


def tracked_files():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["current", "archive-only", "code-util", "orphan"],
                    help="only print probes in this tier")
    args = ap.parse_args()

    tracked = set(tracked_files())
    probes = [f for f in sorted(tracked)
              if f.startswith("proofs/") and f.endswith(".py")
              and not f.startswith("proofs/_archive/")
              and os.path.basename(f) not in INFRA]

    # ALL local .md files (tracked + gitignored lab notebook), so we can
    # distinguish "publicly cited" from "cited only in the local notebook".
    all_md = []
    for root, dirs, names in os.walk("."):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules"}]
        for n in names:
            if n.endswith(".md"):
                all_md.append(os.path.relpath(os.path.join(root, n)))
    code = [f for f in tracked if f.endswith(".py")]
    entry_points = [f for f in ("verify.py", "run_predictions.py") if os.path.exists(f)]

    doc_text = {f: open(f, encoding="utf-8", errors="replace").read() for f in all_md}
    code_text = {f: open(f, encoding="utf-8", errors="replace").read() for f in code}

    tiers = defaultdict(list)
    detail = {}
    for probe in probes:
        base = os.path.basename(probe)
        stem = base[:-3]
        tracked_cites, local_cites, archived_cites = [], [], []
        for doc, text in doc_text.items():
            if base in text or stem in text:
                if doc.startswith(ARCHIVED_DOC_PREFIXES):
                    archived_cites.append(doc)
                elif doc in tracked:
                    tracked_cites.append(doc)
                else:
                    local_cites.append(doc)
        is_backbone = any(stem in code_text[e] for e in entry_points)
        importers = []
        ref = re.compile(r"\b" + re.escape(stem) + r"\b")
        for src, text in code_text.items():
            if src == probe or src in entry_points:
                continue
            if ref.search(text):
                importers.append(src)
        if is_backbone:
            tier = "backbone"
        elif tracked_cites:
            tier = "current"
        elif local_cites:
            tier = "local-only"
        elif archived_cites:
            tier = "archive-only"
        elif importers:
            tier = "code-util"
        else:
            tier = "orphan"
        tiers[tier].append(probe)
        detail[probe] = (tracked_cites, local_cites, archived_cites, importers)

    order = ("backbone", "current", "local-only", "archive-only", "code-util", "orphan")
    for tier in order:
        if args.tier and tier != args.tier:
            continue
        print(f"\n== {tier}: {len(tiers[tier])} ==")
        if tier not in ("current", "local-only"):  # long, uninteresting lists
            for p in tiers[tier]:
                extra = ""
                if tier == "code-util":
                    extra = "  <- " + ", ".join(detail[p][3][:3])
                print(f"  {p}{extra}")
    print("\nTotal probes scanned: %d  (%s)" % (
        len(probes), ", ".join(f"{t} {len(tiers[t])}" for t in order)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
