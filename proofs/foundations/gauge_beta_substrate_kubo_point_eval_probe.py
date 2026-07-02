#!/usr/bin/env python3
"""
gauge_beta_substrate_kubo_point_eval_probe.py — point evaluation, no fit bias.

Follow-up to the convergence diagnostic, which revealed that BOTH Π_TT and Π_v
show ~3% deviation from their structural candidates under linear `a + d/ω²` fit
extraction. The framework's audited 0.07% precision for Π_TT (audit v2) was
achieved via POINT EVALUATION at specific ω with the structural Drude weight
D = -1/36 subtracted independently — a different extraction.

This probe does the framework's exact method, then extends it to Π_v.

  STEP 1 (Π_TT VALIDATION). At N=16, evaluate a_2_phys_TT(ω) at ω ∈ {0.50, 0.70}.
  Subtract -1/(36 ω²) and compare to 4/π² = 0.4053. The framework quote:
    ω = 0.50: a_2_phys + 1/(36·0.25) = 0.4034
    ω = 0.70: a_2_phys + 1/(36·0.49) = 0.4065
    average = 0.4050 vs 4/π² = 0.4053 (0.07% off)
  If my probe reproduces these numbers → probe is consistent with the audited
  Π_TT machinery; the linear-fit slow convergence was an extraction artifact.

  STEP 2 (PAIRWISE EXTRACTION, NO FIT). For each pair (ω₁, ω₂), the Drude
  form a + d/ω² is uniquely solved:
    d_pair = (a_2_phys(ω₁) - a_2_phys(ω₂)) / (1/ω₁² - 1/ω₂²)
    a_pair = a_2_phys(ω₁) - d_pair/ω₁²
  If the Drude form is exact at finite N, all pairs give identical (a, d).
  Variance across pairs measures deviation directly — no fit bias.

  STEP 3 (Π_v PAIRWISE EXTRACTION). Same method applied to Π_v.
  Cross-pair (a, d) consistency tells us whether Π_v has a clean Drude form
  at this N. If yes, the extracted (a, d) are robust structural values.
"""
from __future__ import annotations

import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lorentz_sig_g_sub_dynamic_omega_T import Pi_BZ as Pi_TT_BZ, TT_xyxy
from gauge_beta_from_substrate_kubo_probe import Pi_JJ_BZ


P_Z_VALUES = (0.0, 0.05, 0.10, 0.15, 0.20)


def a2_phys_TT(omega, T, N):
    """Direct point evaluation of a_2_phys_TT(ω) = -a_2_TT/2 at single (ω, N)."""
    Pi_list = []
    for p_z in P_Z_VALUES:
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_TT_BZ(p_cart, omega, T, N=N)
        Pi_list.append(TT_xyxy(K))
    p_arr = np.array(P_Z_VALUES)
    coeffs = np.polyfit(p_arr ** 2, Pi_list, 2)
    a_4, a_2, a_0 = coeffs
    return -a_2 / 2


def a2_phys_v(omega, T, N):
    """Direct point evaluation of a_2_phys_v(ω) = -π_2_xx at single (ω, N)."""
    Pi_list = []
    for p_z in P_Z_VALUES:
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_JJ_BZ(p_cart, omega, T, N=N)
        Pi_list.append(K[0, 0])
    p_arr = np.array(P_Z_VALUES)
    coeffs = np.polyfit(p_arr ** 2, Pi_list, 2)
    a_4, a_2, a_0 = coeffs
    return -a_2


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def pairwise_extract(omegas, a_phys_values):
    """For each pair (ωᵢ, ωⱼ), solve a + d/ω² = a_phys for (a, d). Return list of dicts."""
    results = []
    for i, j in combinations(range(len(omegas)), 2):
        w_i, w_j = omegas[i], omegas[j]
        a_i, a_j = a_phys_values[i], a_phys_values[j]
        d = (a_i - a_j) / (1 / w_i ** 2 - 1 / w_j ** 2)
        a = a_i - d / w_i ** 2
        results.append({"omega_pair": (w_i, w_j), "a": a, "d": d})
    return results


def main():
    header("Point-evaluation diagnostic — Π_TT validation + Π_v pairwise extraction")
    N = 16
    omegas = [0.50, 0.70]  # framework's exact validation points
    omegas_v = [0.30, 0.40, 0.50, 0.55, 0.70]  # broader for Π_v pairwise
    D_TT = -1 / 36.0
    a_TT_target = 4 / np.pi ** 2

    # -----------------------------------------------------------------
    # STEP 1: Π_TT validation
    # -----------------------------------------------------------------
    header(f"STEP 1: Π_TT validation at N={N}, ω ∈ {omegas}")
    print()
    print(f"  Framework audit v2 quote (Π_TT, theorem-grade):")
    print(f"    a_2_phys(0.50) + 1/(36 × 0.25) = 0.4034  (predicted)")
    print(f"    a_2_phys(0.70) + 1/(36 × 0.49) = 0.4065  (predicted)")
    print(f"    Average                          = 0.4050 vs 4/π² = {a_TT_target:.4f}")
    print()
    print(f"  This probe's point evaluation at same (N, ω, p_z, T=ω):")
    print()
    print(f"  {'ω':>6s}  {'a_2_phys_TT':>14s}  {'+1/(36 ω²)':>13s}  {'≈ 4/π²?':>12s}")
    a_TT_extracted = []
    for omega in omegas:
        t0 = time.time()
        a = a2_phys_TT(omega, omega, N)
        dt = time.time() - t0
        subtracted = a + 1 / (36 * omega ** 2)
        a_TT_extracted.append(subtracted)
        dev = abs(subtracted - a_TT_target) / a_TT_target * 100
        print(f"  {omega:>6.3f}  {a:>+14.6f}  {subtracted:>+13.6f}  "
              f"{dev:>+9.4f}%  (t = {dt:.1f}s)")
    a_TT_avg = np.mean(a_TT_extracted)
    a_TT_dev = abs(a_TT_avg - a_TT_target) / a_TT_target * 100
    print()
    print(f"  Average:                {a_TT_avg:+.6f}  (vs 4/π² = {a_TT_target:.6f})")
    print(f"  Deviation from 4/π²:    {a_TT_dev:+.4f}%")
    print(f"  Framework's quoted avg: 0.4050 (0.07% off)")

    # Probe-validation sentinel
    probe_consistent = a_TT_dev < 0.5  # within 0.5% of 4/π²
    if probe_consistent:
        print()
        print(f"  [PASS] Probe consistent with framework audited Π_TT machinery.")
        print(f"  The linear-fit ~3% deviation was an EXTRACTION-METHOD ARTIFACT,")
        print(f"  not a structural property of Π_v. Phase A+B retraction was")
        print(f"  premature — proceed with point-eval extraction for Π_v.")
    else:
        print()
        print(f"  [FAIL] Probe deviation {a_TT_dev:.2f}% > 0.5%. Probe has a precision")
        print(f"  issue at the framework's required level. Need to investigate before")
        print(f"  trusting Π_v point evaluation.")

    # -----------------------------------------------------------------
    # STEP 2: Π_TT pairwise extraction (cross-check on the structural Drude D = -1/36)
    # -----------------------------------------------------------------
    header(f"STEP 2: Π_TT pairwise (a, d) extraction at N={N}")
    print()
    print(f"  Compute a_2_phys_TT at more ω, then for each pair solve")
    print(f"  a + d/ω² = a_2_phys; check (a, d) constancy across pairs.")
    print()
    # We already have ω = 0.50 and 0.70; add 0.30, 0.55 to enable pairwise
    omegas_TT = [0.30, 0.50, 0.55, 0.70]
    a_TT_values = []
    for omega in omegas_TT:
        if omega in omegas:
            idx = omegas.index(omega)
            a = a_TT_extracted[idx] - 1 / (36 * omega ** 2)  # recover raw a_2_phys
        else:
            t0 = time.time()
            a = a2_phys_TT(omega, omega, N)
            print(f"    additional point: ω = {omega}, a_2_phys = {a:+.6f}  (t = {time.time()-t0:.1f}s)")
        a_TT_values.append(a)
    print()
    print(f"  Π_TT a_2_phys point values at N={N}:")
    for omega, a in zip(omegas_TT, a_TT_values):
        print(f"    ω = {omega}: a_2_phys = {a:+.6f}")
    print()
    pairs_TT = pairwise_extract(omegas_TT, a_TT_values)
    print(f"  Pairwise (a, d) extraction (Drude form: a_2_phys = a + d/ω²):")
    print(f"  {'pair':>14s}  {'a_pair':>13s}  {'d_pair':>13s}  {'|d - D_TT|/|D_TT|':>18s}")
    a_pairs_TT = []
    d_pairs_TT = []
    for r in pairs_TT:
        a_pairs_TT.append(r["a"])
        d_pairs_TT.append(r["d"])
        dev_d = abs(r["d"] - D_TT) / abs(D_TT) * 100
        print(f"    ({r['omega_pair'][0]}, {r['omega_pair'][1]})  "
              f"{r['a']:>+13.6f}  {r['d']:>+13.6f}  {dev_d:>+15.4f}%")
    a_TT_pair_mean = np.mean(a_pairs_TT)
    d_TT_pair_mean = np.mean(d_pairs_TT)
    a_TT_pair_std = np.std(a_pairs_TT)
    d_TT_pair_std = np.std(d_pairs_TT)
    print()
    print(f"  Mean a:   {a_TT_pair_mean:+.6f} ± {a_TT_pair_std:.6f}    "
          f"(target 4/π² = {a_TT_target:.6f}, dev "
          f"{(a_TT_pair_mean - a_TT_target)/a_TT_target*100:+.4f}%)")
    print(f"  Mean d:   {d_TT_pair_mean:+.6f} ± {d_TT_pair_std:.6f}    "
          f"(target -1/36 = {D_TT:+.6f}, dev "
          f"{(d_TT_pair_mean - D_TT)/D_TT*100:+.4f}%)")

    # -----------------------------------------------------------------
    # STEP 3: Π_v pairwise extraction
    # -----------------------------------------------------------------
    header(f"STEP 3: Π_v pairwise (a, d) extraction at N={N}")
    print()
    print(f"  No fit, no assumed structural form. Compute a_2_phys_v at multiple ω,")
    print(f"  pairwise-solve for (a, d). Cross-pair variance = Drude-form deviation.")
    print()
    print(f"  Computing a_2_phys_v at N={N} for ω ∈ {omegas_v}:")
    a_v_values = []
    for omega in omegas_v:
        t0 = time.time()
        a = a2_phys_v(omega, omega, N)
        a_v_values.append(a)
        print(f"    ω = {omega}: a_2_phys_v = {a:+.7f}  (t = {time.time()-t0:.1f}s)")
    print()
    pairs_v = pairwise_extract(omegas_v, a_v_values)
    print(f"  Pairwise (a, d) extraction for Π_v:")
    print(f"  {'pair':>16s}  {'a_pair':>13s}  {'d_pair':>13s}  "
          f"{'a × π²':>10s}  {'-1/d':>10s}")
    a_pairs_v = []
    d_pairs_v = []
    for r in pairs_v:
        a_pairs_v.append(r["a"])
        d_pairs_v.append(r["d"])
        # Structural candidate readouts
        a_times_pi2 = r["a"] * np.pi ** 2
        inv_neg_d = -1.0 / r["d"] if r["d"] != 0 else float("inf")
        print(f"    ({r['omega_pair'][0]:.2f}, {r['omega_pair'][1]:.2f})  "
              f"{r['a']:>+13.6f}  {r['d']:>+13.6f}  "
              f"{a_times_pi2:>10.4f}  {inv_neg_d:>10.4f}")
    a_v_mean = np.mean(a_pairs_v)
    d_v_mean = np.mean(d_pairs_v)
    a_v_std = np.std(a_pairs_v)
    d_v_std = np.std(d_pairs_v)
    print()
    print(f"  Mean a:   {a_v_mean:+.7f} ± {a_v_std:.7f}    (a × π² = {a_v_mean * np.pi**2:.5f})")
    print(f"  Mean d:   {d_v_mean:+.7f} ± {d_v_std:.7f}    (-1/d = {-1/d_v_mean:.3f})")
    print()
    print(f"  Candidate matches:")
    print(f"    a vs 1/π² = {1/np.pi**2:.6f}: deviation {(a_v_mean - 1/np.pi**2)/(1/np.pi**2)*100:+.4f}%")
    print(f"    a vs 1/g  = {0.1:.6f}: deviation {(a_v_mean - 0.1)/0.1*100:+.4f}%")
    print(f"    d vs -1/168 = {-1/168:.6f}: deviation {(d_v_mean - (-1/168))/(-1/168)*100:+.4f}%")
    print(f"    d vs -1/144 = {-1/144:.6f}: deviation {(d_v_mean - (-1/144))/(-1/144)*100:+.4f}%")
    print(f"    d vs -1/180 = {-1/180:.6f}: deviation {(d_v_mean - (-1/180))/(-1/180)*100:+.4f}%")
    print(f"    d vs -1/(12 × 3 × 5) = {-1/(12*3*5):.6f}: deviation "
          f"{(d_v_mean - (-1/(12*3*5)))/(-1/(12*3*5))*100:+.4f}%")
    print(f"    d vs -1/(⟨Tr H²⟩ × 14) = -1/168: same")

    # -----------------------------------------------------------------
    # OVERALL VERDICT
    # -----------------------------------------------------------------
    header("OVERALL VERDICT")
    print()
    print(f"  Probe consistency check (Π_TT):")
    print(f"    Audit-method extraction at N={N}: {a_TT_avg:+.6f}")
    print(f"    Framework's expected value:        0.4050")
    print(f"    Deviation: {a_TT_dev:+.4f}%")
    print(f"    {'CONSISTENT' if probe_consistent else 'INCONSISTENT'}")
    print()
    if probe_consistent:
        print(f"  Π_v structural extraction (no fit bias):")
        print(f"    a = {a_v_mean:+.6f} ± {a_v_std:.6f}")
        print(f"    d = {d_v_mean:+.6f} ± {d_v_std:.6f}")
        print()
        print(f"  Drude-form consistency for Π_v:")
        print(f"    Cross-pair std on a: {a_v_std:.6f} (= {a_v_std/a_v_mean*100:.3f}% of mean)")
        print(f"    Cross-pair std on d: {d_v_std:.6f} (= {d_v_std/abs(d_v_mean)*100:.3f}% of mean)")
        if a_v_std / a_v_mean < 0.01 and d_v_std / abs(d_v_mean) < 0.05:
            print(f"    [OK] Drude form a + d/ω² holds for Π_v with consistent (a, d).")
        else:
            print(f"    [WARN] Cross-pair variation > 1% on a or 5% on d — Drude form")
            print(f"           is only approximate at finite N; need higher-order correction.")


if __name__ == "__main__":
    main()
