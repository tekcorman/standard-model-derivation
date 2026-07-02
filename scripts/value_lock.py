#!/usr/bin/env python3
"""
value_lock.py — the value-lock regression harness (audit instrument #7).

Purpose: make silent value drift impossible. Every live predicted value in the
`predictions/` DAG is pinned in `predictions/_value_locks.json`; this script
recomputes all of them (via run_predictions' own introspection machinery) and
fails loudly if any value moved without a deliberate re-freeze.

Motivation: docstring/value drift has repeatedly cost reconciliation effort
(e.g. the m_H 125.578-in-prose vs 125.195-live case, fixed 2026-07-01). Prose
can lag; the lock cannot.

Usage:
    python3 scripts/value_lock.py            # CHECK mode: recompute + compare; exit 1 on drift
    python3 scripts/value_lock.py --freeze   # FREEZE mode: deliberately (re)write the lock file

Semantics:
  - A value change is ALLOWED only via an explicit --freeze in the same change
    that alters the derivation — making every value move visible in review.
  - Relative tolerance 1e-9 (values are deterministic computations; the
    tolerance only absorbs cross-platform float noise). Locked zeros must stay
    |current| < 1e-12.
  - NEW slugs (computed now, absent from the lock) are reported and FAIL the
    check until frozen — a new prediction is also a reviewable event.
  - MISSING slugs (locked, no longer computed) FAIL the check.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_predictions as rp  # noqa: E402  (reuses SECTORS + introspection)

LOCK_PATH = os.path.join(ROOT, "predictions", "_value_locks.json")
REL_TOL = 1e-9
ZERO_TOL = 1e-12


def collect_values():
    """Recompute every live predicted value via run_predictions' machinery."""
    values = {}
    problems = []
    for _sector, params in rp.SECTORS:
        for entry in params:
            _symbol, slug, _obs, _sigma, _units, _notes = entry
            if slug is None:
                continue
            mod = rp._load_module(slug)
            if mod is None:
                problems.append(f"import error: {slug}")
                continue
            p, _o, _s, _d = rp._find_result_vars(mod, slug)
            if p is None or isinstance(p, complex):
                continue  # no scalar predicted value to lock
            try:
                values[slug] = float(p)
            except (TypeError, ValueError):
                continue
    return values, problems


def freeze(values):
    meta = {"frozen": None, "commit": None, "n_values": len(values)}
    try:
        meta["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        meta["frozen"] = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        pass
    with open(LOCK_PATH, "w") as f:
        json.dump({"_meta": meta, "values": values}, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"FROZEN: {len(values)} values -> {LOCK_PATH}")


def check(values, problems):
    if not os.path.exists(LOCK_PATH):
        print(f"NO LOCK FILE at {LOCK_PATH} — run with --freeze first.")
        return 1
    with open(LOCK_PATH) as f:
        lock = json.load(f)
    locked = lock["values"]

    drifted, new, missing = [], [], []
    for slug, cur in sorted(values.items()):
        if slug not in locked:
            new.append(slug)
            continue
        ref = locked[slug]
        if ref == 0.0:
            ok = abs(cur) < ZERO_TOL
        else:
            ok = abs(cur - ref) <= REL_TOL * abs(ref)
        if not ok:
            drifted.append((slug, ref, cur))
    for slug in sorted(locked):
        if slug not in values:
            missing.append(slug)

    n_checked = len(values) - len(new)
    print(f"value-lock: {n_checked} checked against lock "
          f"(frozen {lock['_meta'].get('frozen')}, commit "
          f"{str(lock['_meta'].get('commit'))[:9]})")

    fail = False
    if drifted:
        fail = True
        print(f"\nDRIFT ({len(drifted)}) — a predicted value moved without a re-freeze:")
        for slug, ref, cur in drifted:
            print(f"  {slug}: locked {ref!r} -> current {cur!r} "
                  f"(rel {abs(cur-ref)/abs(ref) if ref else float('inf'):.2e})")
    if new:
        fail = True
        print(f"\nNEW ({len(new)}) — computed but not in the lock (freeze to accept):")
        for slug in new:
            print(f"  {slug} = {values[slug]!r}")
    if missing:
        fail = True
        print(f"\nMISSING ({len(missing)}) — locked but no longer computed:")
        for slug in missing:
            print(f"  {slug} (locked {locked[slug]!r})")
    if problems:
        # import errors are reported but only fail if they hide a locked slug
        print(f"\nnote — modules with no lockable value this run: {len(problems)}")

    if fail:
        print("\nvalue-lock: FAIL — if the change is intentional, re-freeze "
              "deliberately: python3 scripts/value_lock.py --freeze")
        return 1
    print("value-lock: PASS — no silent value drift.")
    return 0


if __name__ == "__main__":
    vals, probs = collect_values()
    if "--freeze" in sys.argv:
        freeze(vals)
        sys.exit(0)
    sys.exit(check(vals, probs))
