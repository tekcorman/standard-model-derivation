#!/usr/bin/env python3
"""
scripts/export_bridge_vectors.py

CB-1 -- THE BRIDGE KIT: exporter + --check regression for the the upstream engine bridge (side quest).
Pre-registered FROZEN in internal research notes (BK-0..BK-7,
verbatim below). Schema frozen in internal research notes
sec CB-1 -- normative; any deviation from it is a printed FINDING, never a silent change.

WHAT THIS FILE IS: a pure EXPORT/CHECK instrument, zero new physics. Every value is either
  (a) a READ off the two importable engine libraries
        derivation_topdown/dirac_srs_mdl/srs.py   (walled-off clean room; never edited)
        derivation_topdown/bridge/the_run.py       (the engine, Layer 1; never edited)
  (b) a RE-EXPRESSION (REUSE MAP comment at each site) of a recipe living in a FLAT-SCRIPT
      adapter that executes-and-sys.exit()s on import -- those adapters are therefore NEVER
      imported here:
        derivation_topdown/adapters/thermal_time.py   (u_c, beta_eff)
        derivation_topdown/adapters/zeta_gauge.py      (W_INT / trW_INT / loop-identity)
  (c) an IMPORT of proofs/foundations/dl_comparison.py's dl_*() scorer functions -- that file
      HAS an `if __name__ == "__main__":` guard (verified: only main() runs top-level side
      effects), so importing it is the frozen BK-5 "import route", not a re-expression.
  (d) a READ-ONLY comparison against predictions/_value_locks.json (BK-2) -- never written,
      never touched by --freeze (this script never calls value_lock.py).

Two modes:
  bare invocation   -> CHECK mode: recompute everything fresh, diff against the committed
                       bridge_kit/bridge_vectors.json at rtol 1e-12, print per-block/per-field
                       PASS/FAIL, sys.exit(0) all-green / sys.exit(1) otherwise. If the JSON is
                       absent, print instructions and sys.exit(1) (no recompute attempted).
  --export           -> write bridge_kit/bridge_vectors.json (and ONLY that file), after
                       running the exact same BK-1..BK-5 contract checks inline.

POISONS (binding, see the pre-reg and the roadmap sec 0): no --freeze; no edits to
the_run.py / the_net.py / any proofs/ file / verify.py / predictions/; no hand-typed floats
anywhere (structural integers, frozen seeds, and documented tolerances only -- every float is
computed on-screen); no importing the flat-script adapters; no new getters added to the
engine "for convenience" (gap_additive below is computed INLINE, not via a new srs.py/
the_run.py accessor).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

_T0 = time.time()

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
sys.path.insert(0, os.path.join(REPO, "proofs", "foundations"))

import srs                # noqa: E402 -- engine primitives, walled-off clean room. Never edited.
import the_run as R       # noqa: E402 -- the engine (Layer 1). Never edited.

# BK-5 import route: dl_comparison.py's top-level code is only defs + WYCKOFF_DATA + a
# `if __name__ == "__main__":` guard around main() (verified by reading the file; grep below
# is the same check re-run at every invocation so a future edit that removes the guard is
# caught, not silently trusted).
_dlsrc = open(os.path.join(REPO, "proofs", "foundations", "dl_comparison.py")).read()
if '__name__ == "__main__"' not in _dlsrc and "__name__ == '__main__'" not in _dlsrc:
    print("FATAL: proofs/foundations/dl_comparison.py lost its __main__ guard -- the BK-5 "
          "import route is no longer safe. Re-express instead of importing (see the pre-reg).")
    sys.exit(1)
import dl_comparison as DL  # noqa: E402 -- guarded; safe to import (BK-5 import route).

OUT_PATH = os.path.join(REPO, "bridge_kit", "bridge_vectors.json")
LOCKS_PATH = os.path.join(REPO, "predictions", "_value_locks.json")

FAILURES = []


def banner(t):
    print("=" * 88)
    print(f" {t}")
    print("=" * 88)


def check(name, cond, detail=""):
    cond = bool(cond)
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def kp_label(kp):
    return "Gamma" if kp == (0.0, 0.0, 0.0) else "k(%.10f,%.10f,%.10f)" % kp


def frozen_kpoints():
    """Gamma + 3 pseudo-random k-points, seed 0 (documented in bridge_kit/README.md). This is
    an INDEPENDENT np.random.default_rng(0) stream from the one used later for the BK-4
    self-test u-samples (two distinct seed(0) draws, not one shared sequential stream --
    documented explicitly in the README so a reader is never surprised by the ordering)."""
    rng = np.random.default_rng(0)
    pts = [(0.0, 0.0, 0.0)]
    for _ in range(3):
        pts.append(tuple(float(x) for x in rng.random(3)))
    return pts


# ══════════════════════════════════════════════════════════════════════════
# BUILD -- one function per JSON top-level block
# ══════════════════════════════════════════════════════════════════════════
def build_substrate():
    # srs.py:11-15 (NV, DEG, EDGES); b1 = NE-NV+1 (Euler formula for the Z^3 deck rank, cross-
    # checked against the_run.read_geometry()'s own b1 at the_run.py:94).
    K = srs.DEG
    NV = srs.NV
    NE = len(srs.EDGES)
    b1 = NE - NV + 1
    GIRTH = R.GIRTH  # the_run.py:78 -- read_girth() off the Hashimoto renewal sequence, not typed
    edges = [[i, j, list(v)] for (i, j, v) in srs.EDGES]  # srs.EDGES verbatim (srs.py:14-15)
    return {"K": K, "GIRTH": GIRTH, "NV": NV, "NE": NE, "b1": b1, "edges": edges}


def build_run():
    alpha_1 = float(R.U_RUN)              # the_run.py:324
    rho_survival = float(R.RHO)           # the_run.py:323 -- Fraction(K-1,K) = 2/3, computed off K
    rho_step, arrow, _G = R.read_run()    # the_run.py:481-486
    return {"alpha_1": alpha_1, "rho_survival": rho_survival,
            "rho_step": float(rho_step), "arrow": bool(arrow)}


def build_thermo(alpha_1):
    # REUSE MAP (re-expressed, never imported -- thermal_time.py is a flat script that
    # sys.exit()s on import):
    #   u_c = 1.0/q  -- thermal_time.py:151          (q = k-1, k = srs.DEG)
    #   beta_eff = 2*math.log(u_c/alpha1) -- thermal_time.py:209
    K = srs.DEG
    u_c = 1.0 / (K - 1)
    beta_eff = 2.0 * math.log(u_c / alpha_1)
    ln2 = math.log(2)
    return {"u_c": u_c, "beta_eff": beta_eff, "ln2": ln2}


def build_clock():
    eps, clock = R.read_clock()  # the_run.py:83-90
    return {"eps": float(eps), "clock": float(clock)}


def build_spectral(kpts):
    perron, irrep3 = R.adjacency_energies()  # the_run.py:71-76
    K = srs.DEG
    # gap_additive: the frozen pre-reg schema names the literal quantity "2 - sqrt(3)"
    # (proofs/foundations/srs_ramanujan_theorem.py:676, "the spectral gap lambda_1 =
    # 2 - sqrt(3)"). Expressed here as (K-1) - sqrt(K) so no bare float literal appears (the
    # no-hand-typed-floats poison) -- numerically IDENTICAL to 2-sqrt(3) for this crystal's
    # K=srs.DEG=3; booked as a FINDING (design choice), not a silent renumbering. No getter is
    # added to srs.py/the_run.py for this -- computed inline here only, per the pre-reg's
    # explicit instruction.
    gap_additive = (K - 1) - math.sqrt(K)
    adjacency_eigs_at_k = {}
    for kp in kpts:
        eigs = np.linalg.eigvalsh(srs.adjacency(kp))  # srs.py:17-22; Hermitian, 4x4 (NV=4)
        adjacency_eigs_at_k[kp_label(kp)] = [float(e) for e in sorted(eigs)]
    return {"perron": float(perron), "irrep3": float(irrep3), "gap_additive": float(gap_additive),
            "adjacency_eigs_at_k": adjacency_eigs_at_k}


def _det_coeffs(B):
    """Coefficients of det(I - u*B) = prod_i (1 - lambda_i * u), ASCENDING powers of u
    (index n = coefficient of u^n), built from the eigenvalues of B by polynomial
    convolution (this exporter's own math -- not an engine recipe)."""
    eigs = np.linalg.eigvals(B)
    coeffs = np.array([1.0 + 0.0j])
    for lam in eigs:
        coeffs = np.convolve(coeffs, np.array([1.0 + 0.0j, -lam]))
    return coeffs


def build_zeta(kpts):
    # ── det_coeffs_at_k + BK-4 self-test ─────────────────────────────────────
    det_coeffs_at_k = {}
    for kp in kpts:
        B = srs.hashimoto(kp)  # srs.py:42-49
        c = _det_coeffs(B)
        det_coeffs_at_k[kp_label(kp)] = [[float(x.real), float(x.imag)] for x in c]

    # BK-4 self-test: the recorded coefficients reproduce slogdet(I-uB(k)) at 5 random u
    # (seed 0), to 1e-9. Independent np.random.default_rng(0) stream (documented in README).
    rng_u = np.random.default_rng(0)
    # REUSE MAP: the 1.2*(rand-0.5) complex u-sampling recipe is zeta_gauge.py:149 verbatim
    # (US = 1.2 * (RNG1.random(30) - 0.5) + 1.2j * ...); added at integration per adversarial
    # check issue #1 (provenance previously README-only).
    us = 1.2 * (rng_u.random(5) - 0.5) + 1.2j * (rng_u.random(5) - 0.5)
    worst = 0.0
    for kp in kpts:
        B = srs.hashimoto(kp)
        c = _det_coeffs(B)
        I_nd = np.eye(B.shape[0], dtype=complex)
        for u in us:
            poly_val = sum(c[n] * u ** n for n in range(len(c)))
            sign, logabsdet = np.linalg.slogdet(I_nd - u * B)
            direct_val = sign * np.exp(logabsdet)
            worst = max(worst, abs(poly_val - direct_val))
    check("BK-4 self-test: det_coeffs(k) polynomial reproduces slogdet(I-uB(k)) at 5 random u "
          "(seed 0), Gamma + 3 k-points", worst < 1e-9, detail=f"worst |poly-direct| = {worst:.3e}")

    # ── W_INT (REUSE MAP: zeta_gauge.py:522-541, verbatim block-assignment logic; the Cl(6)
    #    generators come from simulator.srs_engine.utils.AlgebraicUtility.cl6_generators() per
    #    zeta_gauge.py:522-524). W_INT is 8*ND x 8*ND with ND = 2*len(srs.EDGES) = 12 -> 96x96.
    from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402 -- not the flat adapter

    g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
    NE = len(srs.EDGES)

    def gam(x):
        return sum(x[a] * g6[a] for a in range(NE))

    ND = 2 * NE  # = 12, the dart count (READ from the code, not assumed)
    EDGE_OF_DART = [d // 2 for d in range(ND)]
    B_GAMMA = srs.hashimoto((0.0, 0.0, 0.0)).real
    W_INT = np.zeros((8 * ND, 8 * ND), dtype=complex)
    for dp in range(ND):
        for d in range(ND):
            if abs(B_GAMMA[dp, d]) > 0.5:
                W_INT[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = gam(np.eye(NE)[:, EDGE_OF_DART[dp]])

    # trW_INT for L=1..20 (recorded) via 20 successive matmuls; the same running power is
    # carried on to L=40 (not recorded, only used for the residual below) per the pre-reg's
    # BK-4 tolerance (zeta_gauge.py:558-589's own L<=40 truncation).
    u_a1 = float(R.U_RUN)
    dim = W_INT.shape[0]
    Wp = np.eye(dim, dtype=complex)
    trW_INT = []
    loop_total = 0.0 + 0.0j
    for L in range(1, 41):
        Wp = Wp @ W_INT
        tr = np.trace(Wp)
        if L <= 20:
            trW_INT.append([float(tr.real), float(tr.imag)])
        loop_total += (u_a1 ** L / L) * tr

    sign_W, logabsdet_W = np.linalg.slogdet(np.eye(dim) - u_a1 * W_INT)
    lhs = -(logabsdet_W + 1j * np.angle(sign_W))
    residual = abs(lhs - loop_total)
    check("BK-4 loop_identity_residual_at_alpha1 < 1e-9 (expected ~1e-17)", residual < 1e-9,
          detail=f"residual = {residual:.3e}")

    return {"det_coeffs_at_k": det_coeffs_at_k, "trW_INT": trW_INT,
            "loop_identity_residual_at_alpha1": float(residual)}


def build_mdl():
    # BK-5 import route (dl_comparison.py has the __main__ guard, verified above).
    entries = [
        ("srs (Laves)", DL.dl_srs, "crystal_3d"),
        ("ths (ThSi2)", DL.dl_ths, "crystal_3d"),
        ("eta net", DL.dl_eta, "crystal_3d"),
        ("utj net", DL.dl_utj, "crystal_3d"),
        ("honeycomb (2D)", DL.dl_honeycomb_2d, "crystal_2d"),
        ("Petersen", DL.dl_petersen, "finite"),
        ("K_{3,3}", DL.dl_k33, "finite"),
    ]
    dl_table = {}
    cats = {}
    srs_breakdown = None
    for name, fn, cat in entries:
        val, bd = fn()
        dl_table[name] = float(val)
        cats[name] = cat
        if name == "srs (Laves)":
            srs_breakdown = {k: float(v) for k, v in bd.items()}

    for N in (100, 1000):  # structural ints, same frozen sample points as dl_comparison.main()
        val, _bd = DL.dl_random(N)
        name = f"random (N={N})"
        dl_table[name] = float(val)
        cats[name] = "finite"

    crystal3d = {n: v for n, v in dl_table.items() if cats[n] == "crystal_3d"}
    srs_val = dl_table["srs (Laves)"]
    others = sorted(v for n, v in crystal3d.items() if n != "srs (Laves)")
    check("BK-5 srs (Laves) ranks strictly minimal DL among the crystal_3d candidates",
          all(srs_val < v for v in others),
          detail=f"srs={srs_val:.6f} bits; next-lowest crystal_3d competitor={others[0]:.6f} bits")

    return {"dl_table": dl_table, "srs_breakdown": srs_breakdown}


def compute_all():
    substrate = build_substrate()
    run = build_run()
    thermo = build_thermo(run["alpha_1"])
    clock = build_clock()
    kpts = frozen_kpoints()
    spectral = build_spectral(kpts)
    zeta = build_zeta(kpts)
    mdl = build_mdl()
    meta = {
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "export_utc": datetime.now(timezone.utc).isoformat(),
        "exporter": "scripts/export_bridge_vectors.py",
        "schema": 1,
    }
    return {"meta": meta, "substrate": substrate, "run": run, "thermo": thermo, "clock": clock,
            "spectral": spectral, "zeta": zeta, "mdl": mdl}


# ══════════════════════════════════════════════════════════════════════════
# CONTRACT CHECKS -- BK-1, BK-2, BK-3 (BK-4/BK-5 print inline inside build_zeta/build_mdl)
# ══════════════════════════════════════════════════════════════════════════
def check_bk1(substrate):
    banner("BK-1  substrate block === srs module (exact structural identity)")
    check("BK-1a K == srs.DEG == 3", substrate["K"] == srs.DEG == 3)
    check("BK-1b GIRTH == 10", substrate["GIRTH"] == 10)
    check("BK-1c NV == 4", substrate["NV"] == 4)
    check("BK-1d NE == 6", substrate["NE"] == 6)
    check("BK-1e b1 == NE - NV + 1 == 3",
          substrate["b1"] == substrate["NE"] - substrate["NV"] + 1 == 3)
    edges_expected = [[i, j, list(v)] for (i, j, v) in srs.EDGES]
    check("BK-1f edges == srs.EDGES verbatim", substrate["edges"] == edges_expected)


def check_bk2(run_block):
    banner("BK-2  run.alpha_1 == the_run.U_RUN == predictions/_value_locks.json['alpha_1'] "
           "(rtol 1e-12, READ-ONLY); run.arrow == True")
    check("BK-2a run.alpha_1 == the_run.U_RUN", run_block["alpha_1"] == float(R.U_RUN))
    with open(LOCKS_PATH) as f:          # READ-ONLY; never written by this script
        locks = json.load(f)
    locked_alpha1 = locks["values"]["alpha_1"]
    rel = abs(run_block["alpha_1"] - locked_alpha1) / abs(locked_alpha1)
    check("BK-2b run.alpha_1 == predictions/_value_locks.json['alpha_1'] to rtol 1e-12",
          rel < 1e-12, detail=f"alpha_1={run_block['alpha_1']!r} lock={locked_alpha1!r} "
                               f"rel={rel:.3e}")
    check("BK-2c run.arrow == True", run_block["arrow"] is True)


def check_bk3(thermo, run_block, substrate):
    banner("BK-3  thermo block internal consistency (exact, recomputed from the JSON's own "
           "fields)")
    K = substrate["K"]
    check("BK-3a u_c == 1/(K-1)", thermo["u_c"] == 1.0 / (K - 1))
    beta_recomputed = 2.0 * math.log(thermo["u_c"] / run_block["alpha_1"])
    check("BK-3b beta_eff == 2*ln(u_c/alpha_1) recomputed from the JSON's own fields",
          math.isclose(thermo["beta_eff"], beta_recomputed, rel_tol=1e-12))
    check("BK-3c ln2 == math.log(2)", thermo["ln2"] == math.log(2))


# ══════════════════════════════════════════════════════════════════════════
# BK-6 CHECK MODE -- deep diff of the payload blocks at rtol 1e-12
# ══════════════════════════════════════════════════════════════════════════
def deep_diff(path, a, b, rtol=1e-12, mismatches=None):
    if mismatches is None:
        mismatches = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                mismatches.append((f"{path}.{k}", "missing-key", a.get(k), b.get(k)))
            else:
                deep_diff(f"{path}.{k}", a[k], b[k], rtol, mismatches)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            mismatches.append((path, "length", len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deep_diff(f"{path}[{i}]", x, y, rtol, mismatches)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            mismatches.append((path, "bool", a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-15):
            mismatches.append((path, "float", a, b))
    else:
        if a != b:
            mismatches.append((path, "other", a, b))
    return mismatches


def diff_against_committed(committed, fresh):
    banner("BK-6  CHECK MODE -- diff committed bridge_vectors.json vs a fresh recompute "
           "(rtol 1e-12)")
    # meta stamps: INFORMATIONAL ONLY, never gated. export_utc always differs (wall clock).
    # repo_commit can legitimately differ if a commit landed between export and check (this
    # repo runs an auto-sync cron per the standing env note) -- that is NOT a payload mismatch.
    print(f"  meta.export_utc   committed={committed['meta'].get('export_utc')}   "
          f"fresh={fresh['meta'].get('export_utc')}  (informational; always differs)")
    same_commit = committed["meta"].get("repo_commit") == fresh["meta"].get("repo_commit")
    print(f"  meta.repo_commit  committed={committed['meta'].get('repo_commit')}   "
          f"fresh={fresh['meta'].get('repo_commit')}  "
          f"({'SAME' if same_commit else 'DIFFERENT -- informational only'})")

    for block in ("substrate", "run", "thermo", "clock", "spectral", "zeta", "mdl"):
        mism = deep_diff(block, committed.get(block), fresh.get(block))
        for path, kind, a, b in mism:
            print(f"    [FAIL] {path}: committed={a!r} fresh={b!r}  ({kind} mismatch)")
        check(f"BK-6 '{block}' block matches committed JSON at rtol 1e-12", len(mism) == 0,
              detail=f"{len(mism)} field mismatch(es)")


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", action="store_true",
                     help="write bridge_kit/bridge_vectors.json (default: --check mode)")
    args = ap.parse_args()

    banner("CB-1 BRIDGE KIT -- scripts/export_bridge_vectors.py "
           f"[{'EXPORT' if args.export else 'CHECK'} mode]")

    committed = None
    if not args.export:
        if not os.path.exists(OUT_PATH):
            print(f"  bridge_kit/bridge_vectors.json not found at {OUT_PATH}.")
            print("  Run first:   python3 scripts/export_bridge_vectors.py --export")
            print("  Then re-run this script bare (no flags) to CHECK the export against a "
                  "fresh recompute.")
            sys.exit(1)
        with open(OUT_PATH) as f:
            committed = json.load(f)

    data = compute_all()   # BK-4/BK-5 gates print inline as they are built, every invocation
    check_bk1(data["substrate"])
    check_bk2(data["run"])
    check_bk3(data["thermo"], data["run"], data["substrate"])

    if args.export:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"\n  wrote {OUT_PATH}")
    else:
        diff_against_committed(committed, data)

    dt = time.time() - _T0
    check("BK-0 exporter completed in < 120s", dt < 120, detail=f"elapsed={dt:.2f}s")

    print(f"\n  elapsed: {dt:.2f}s")
    ok = len(FAILURES) == 0
    if ok:
        print("\n  *** ALL GREEN ***")
    else:
        print(f"\n  *** FAILED ({len(FAILURES)}) ***")
        for f_ in FAILURES:
            print(f"    - {f_}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
