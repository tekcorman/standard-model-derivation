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


def _manual_locks():
    """Manually-registered locks (S1b orphan cleanup, 2026-07-09, user-approved re-freeze).

    These entries are NOT in run_predictions' SECTORS, so the sector sweep cannot
    recompute them; each is recomputed here from its own source module so the
    harness stays the single verifier of every lock entry.
      - T_nu_dec: predictions/T_nu_dec.py (scalar, MeV)
      - h_walker_eigenvalue_re/_im: the walker root h = (sqrt3 + i*sqrt5)/2 from
        predictions/h_walker_eigenvalue.py (complex, frozen as re/im pair — the
        sector sweep skips complex values by design)."""
    values = {}
    problems = []
    mod = rp._load_module("T_nu_dec")
    if mod is None:
        problems.append("import error: T_nu_dec (manual lock)")
    else:
        # Explicit attribute: the module's canonical result is T_nu_dec_pred_MeV
        # (the generic _find_result_vars fallback would wrongly grab the shorter
        # G_F_pred, the module's internal anchor — not the decoupling temperature).
        p = getattr(mod, "T_nu_dec_pred_MeV", None)
        if p is not None and not isinstance(p, complex):
            values["T_nu_dec"] = float(p)
        else:
            problems.append("manual lock T_nu_dec: T_nu_dec_pred_MeV not found")
    mod = rp._load_module("h_walker_eigenvalue")
    if mod is None:
        problems.append("import error: h_walker_eigenvalue (manual lock)")
    else:
        p, _o, _s, _d = rp._find_result_vars(mod, "h_walker_eigenvalue")
        if isinstance(p, complex):
            values["h_walker_eigenvalue_re"] = float(p.real)
            values["h_walker_eigenvalue_im"] = float(p.imag)
        else:
            problems.append("manual lock h_walker_eigenvalue: expected complex value")
    hv, hp = _r1_harvest_locks()
    values.update(hv)
    problems.extend(hp)
    return values, problems


def _r1_harvest_locks():
    """R1 HARVEST (2026-07-10) — additive lock registration for
    internal research notes's H-1/H-2/H-3/H-5 engine composites.

    These are NOT predictions/*.py files (per the pre-reg: "NO NEW PHYSICS ... exact
    compositions of already-certified engine reads") so run_predictions' SECTORS sweep
    cannot see them; each is recomputed straight from the single new appended engine
    section derivation_topdown/bridge/the_run.py's read_r1_harvest() (H-1 coasting-chain
    curve at z in {0.5,1.0,2.0}; H-2 Sigma_m_nu/Omega_k/Omega_b_h2/Omega_c_h2; H-3 m_bb
    under both phase-convention placements; H-5's fermion_content/h_walker_abs2/
    cone_velocity_v0/T_of_N_now scalar wiring) — see R1_HARVEST_2026-07-10.py for the full
    per-contract report; this function is the SAME computation, frozen for drift-detection."""
    values, problems = {}, []
    bridge_dir = os.path.join(ROOT, "derivation_topdown", "bridge")
    if bridge_dir not in sys.path:
        sys.path.insert(0, bridge_dir)
    try:
        import the_run as _R1  # noqa: E402  (the engine; read-only call, no edit)
        h = _R1.read_r1_harvest()
    except Exception as exc:  # pragma: no cover — surfaced as a problem, not a crash
        return {}, [f"R1 harvest locks: the_run.read_r1_harvest() failed: {exc!r}"]

    KEEP_Z = ("z0p5", "z1p0", "z2p0")   # the 3 lock-worthy declared z points (curve subset)
    for base in ("H", "D_C", "D_A", "D_L", "D_V"):
        for ztag in KEEP_Z:
            k = f"{base}_{ztag}"
            if k in h:
                values[f"harvest_{k}"] = float(h[k])
            else:
                problems.append(f"R1 harvest lock missing curve key: {k}")

    SCALAR_KEYS = ("q_0", "w_eff", "Omega_k", "Sigma_m_nu_eV", "Omega_b_h2", "Omega_c_h2",
                   "m_bb_meV_conv1", "m_bb_meV_conv2", "fermion_content", "h_walker_abs2",
                   "cone_velocity_v0", "T_of_N_now_eV")
    for k in SCALAR_KEYS:
        if k in h:
            values[f"harvest_{k}"] = float(h[k])
        else:
            problems.append(f"R1 harvest lock missing scalar key: {k}")
    return values, problems


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
    mvals, mprobs = _manual_locks()
    values.update(mvals)
    problems.extend(mprobs)
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
