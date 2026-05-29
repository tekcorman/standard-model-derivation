#!/usr/bin/env python3
"""
gauge_beta_substrate_kubo_convergence_diagnostic_probe.py — Π_v vs Π_TT

Side-by-side convergence diagnostic for the Phase A+B audit's "genuinely open"
question: why does Π_v converge so slowly vs Π_TT?

Background. Phase A+B's structural identifications (a = 1/π², d = -1/168,
factor π closure) were retracted because higher-N test showed Π_v's
deviation from 1/π² GROWING with N (1.88% → 2.05% from N=14 to N=18) —
opposite of the Phase A claim that 1.88% was grid noise. Meanwhile,
Π_TT's audit v2 reports closure to a = 4/π² at <0.07% precision at the
same N values. Why the qualitative difference?

Hypothesis tested. Π_v's slow convergence is a fit-window artifact from
using ω ∈ [0.15, 0.70] (Phase A) vs Π_TT's [0.15, 0.50] (existing
verifier). If we match Π_v's window to Π_TT's, do BOTH converge cleanly?

Method. Compute Π_TT and Π_v at N = 12, 14, 16, 18 using IDENTICAL
fit window [0.15, 0.50], identical p_z = (0, 0.05, 0.10, 0.15, 0.20),
identical regulator T = ω, identical MP-shifted grid. The ONLY
difference is the vertex (strain A^{ac}(k) = k_a · v^c(k) vs velocity
v^μ(k) = ∂H/∂k_μ).

Three possible outcomes:

  (A) Π_v converges cleanly to a candidate (1/π² or 1/g) under the
      narrowed window — Phase A's slow convergence WAS a window artifact;
      structural form is pinnable, Phase B was right modulo precision.

  (B) Π_v converges to a clean value DIFFERENT from Phase B's candidates —
      structural form exists but is something else.

  (C) Π_v's slow convergence persists under matched conditions — Π_v
      genuinely differs structurally from Π_TT (different mechanism,
      different K[1/π²] hierarchy, or no clean structural form).

Each outcome is informative for what to do next.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lorentz_sig_g_sub_dynamic_omega_T import Pi_BZ as Pi_TT_BZ, TT_xyxy
from gauge_beta_from_substrate_kubo_probe import Pi_JJ_BZ


# Π_TT verifier's ω range exactly (per `lorentz_sig_g_sub_drude_pole_verify.py`)
OMEGAS_NARROW = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.18, 0.15]
# Phase A's (wider) window for comparison
OMEGAS_WIDE = [0.70, 0.55, 0.45, 0.35, 0.30, 0.25, 0.20, 0.18, 0.15]
P_Z_VALUES = (0.0, 0.05, 0.10, 0.15, 0.20)


def extract_a2_TT(omega, T, N):
    """Π_TT: leading p_z² coefficient of K[0,1,0,1] = TT_xyxy.
    Matches `lorentz_sig_g_sub_drude_pole_verify.py` extract_a2 exactly."""
    Pi_list = []
    for p_z in P_Z_VALUES:
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_TT_BZ(p_cart, omega, T, N=N)
        Pi_list.append(TT_xyxy(K))
    p_arr = np.array(P_Z_VALUES)
    coeffs = np.polyfit(p_arr ** 2, Pi_list, 2)
    a_4, a_2, a_0 = coeffs
    return a_2


def extract_pi2_v(omega, T, N):
    """Π_v: leading p_z² coefficient of K[0,0] = Π^{xx}(p_z ẑ)."""
    Pi_list = []
    for p_z in P_Z_VALUES:
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_JJ_BZ(p_cart, omega, T, N=N)
        Pi_list.append(K[0, 0])
    p_arr = np.array(P_Z_VALUES)
    coeffs = np.polyfit(p_arr ** 2, Pi_list, 2)
    a_4, a_2, a_0 = coeffs
    return a_2


def fit_drude(omegas, raw_a_2_arr):
    """Fit Drude form a_raw + d_raw/ω² over the omega range. Returns (a_raw, d_raw)."""
    inv_om2 = 1.0 / np.array(omegas) ** 2
    d_raw, a_raw = np.polyfit(inv_om2, raw_a_2_arr, 1)
    return a_raw, d_raw


def convergence_run(extract_fn, name, omegas, N_values, sign_flip_factor):
    """Run extract_fn(omega, T=omega, N) over (omegas × N_values). Return dict per N
    with (a_phys, d_phys, deviations from candidates).
    sign_flip_factor = -1/2 for Π_TT, -1 for Π_v.
    """
    print(f"\n  --- {name}: window ω ∈ [{omegas[-1]}, {omegas[0]}], "
          f"{len(omegas)} points, T = ω ---")
    out = {}
    for N in N_values:
        t0 = time.time()
        raw_vals = []
        for omega in omegas:
            a_2 = extract_fn(omega, omega, N)
            raw_vals.append(a_2)
        raw_arr = np.array(raw_vals)
        # Apply sign flip / factor convention
        phys_arr = sign_flip_factor * raw_arr
        a_raw, d_raw = fit_drude(omegas, phys_arr)
        elapsed = time.time() - t0
        out[N] = {"a_phys": a_raw, "d_phys": d_raw, "time": elapsed}
        print(f"    N = {N:2d}: a_phys = {a_raw:+.7f}, "
              f"d_phys = {d_raw:+.7f}   ({elapsed:.1f}s)")
    return out


def main():
    print("=" * 78)
    print("  Convergence diagnostic: Π_v vs Π_TT under MATCHED window/grid/regulator")
    print("=" * 78)
    print(f"  Common parameters: T = ω, p_z = {P_Z_VALUES}, fit a + d/ω² in p² extract.")
    print(f"  ONLY difference: vertex (strain A^{{ac}} = k_a v^c for TT vs velocity v^μ for v).")

    N_values = [12, 14, 16, 18]
    candidate_TT_a = 4 / np.pi ** 2
    candidate_TT_d = -1 / 36
    candidate_v_a_pi2 = 1 / np.pi ** 2
    candidate_v_a_g = 0.10
    candidate_v_d = -1 / 168

    print()
    print("=" * 78)
    print("  RUN 1: NARROW WINDOW ω ∈ [0.15, 0.50] (matches Π_TT verifier exactly)")
    print("=" * 78)

    # Π_TT extraction — convention a_phys = -a_2 / 2
    TT_results = convergence_run(
        extract_a2_TT, "Π_TT (strain vertex)", OMEGAS_NARROW, N_values,
        sign_flip_factor=-0.5,
    )

    # Π_v extraction — convention a_phys = -π_2_xx (sign flip only)
    v_results = convergence_run(
        extract_pi2_v, "Π_v (velocity vertex)", OMEGAS_NARROW, N_values,
        sign_flip_factor=-1.0,
    )

    # === Convergence summary ===
    print()
    print("=" * 78)
    print("  CONVERGENCE SUMMARY — narrow window")
    print("=" * 78)
    print(f"\n  Π_TT vs candidate a = 4/π² = {candidate_TT_a:.7f}")
    print(f"  {'N':>3s}   {'a_phys':>13s}   {'|dev|':>10s}   {'d_phys':>13s}   {'|dev|':>10s}")
    for N in N_values:
        a = TT_results[N]["a_phys"]
        d = TT_results[N]["d_phys"]
        a_dev = abs(a - candidate_TT_a) / candidate_TT_a * 100
        d_dev = abs(d - candidate_TT_d) / abs(candidate_TT_d) * 100
        print(f"  {N:>3d}   {a:>+13.7f}   {a_dev:>+9.4f}%   {d:>+13.7f}   {d_dev:>+9.4f}%")

    print(f"\n  Π_v vs candidates a = 1/π² = {candidate_v_a_pi2:.7f}  AND  1/g = 0.10")
    print(f"  {'N':>3s}   {'a_phys':>13s}   {'|dev_π²|':>11s}   {'|dev_g|':>10s}   "
          f"{'d_phys':>13s}   {'|dev_168|':>11s}")
    for N in N_values:
        a = v_results[N]["a_phys"]
        d = v_results[N]["d_phys"]
        a_dev_pi2 = abs(a - candidate_v_a_pi2) / candidate_v_a_pi2 * 100
        a_dev_g = abs(a - candidate_v_a_g) / candidate_v_a_g * 100
        d_dev = abs(d - candidate_v_d) / abs(candidate_v_d) * 100
        print(f"  {N:>3d}   {a:>+13.7f}   {a_dev_pi2:>+10.4f}%   {a_dev_g:>+9.4f}%   "
              f"{d:>+13.7f}   {d_dev:>+10.4f}%")

    print()
    print("=" * 78)
    print("  RUN 2: WIDE WINDOW ω ∈ [0.15, 0.70] (Phase A's choice)")
    print("=" * 78)

    TT_results_wide = convergence_run(
        extract_a2_TT, "Π_TT wide", OMEGAS_WIDE, N_values, sign_flip_factor=-0.5,
    )
    v_results_wide = convergence_run(
        extract_pi2_v, "Π_v wide", OMEGAS_WIDE, N_values, sign_flip_factor=-1.0,
    )

    print()
    print("=" * 78)
    print("  CONVERGENCE SUMMARY — wide window")
    print("=" * 78)
    print(f"\n  Π_TT vs 4/π²:")
    for N in N_values:
        a = TT_results_wide[N]["a_phys"]
        a_dev = abs(a - candidate_TT_a) / candidate_TT_a * 100
        print(f"    N={N:>3d}: a = {a:+.7f}, |dev| = {a_dev:.4f}%")
    print(f"\n  Π_v vs 1/π² / 1/g:")
    for N in N_values:
        a = v_results_wide[N]["a_phys"]
        a_dev_pi2 = abs(a - candidate_v_a_pi2) / candidate_v_a_pi2 * 100
        a_dev_g = abs(a - candidate_v_a_g) / candidate_v_a_g * 100
        print(f"    N={N:>3d}: a = {a:+.7f}, |dev_π²| = {a_dev_pi2:.4f}%, "
              f"|dev_g| = {a_dev_g:.4f}%")

    # === Verdict ===
    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    # Compute trend
    TT_trend_narrow = abs(TT_results[18]["a_phys"] - candidate_TT_a) / candidate_TT_a * 100 - \
                      abs(TT_results[12]["a_phys"] - candidate_TT_a) / candidate_TT_a * 100
    TT_trend_wide = abs(TT_results_wide[18]["a_phys"] - candidate_TT_a) / candidate_TT_a * 100 - \
                    abs(TT_results_wide[12]["a_phys"] - candidate_TT_a) / candidate_TT_a * 100
    v_trend_narrow_pi2 = abs(v_results[18]["a_phys"] - candidate_v_a_pi2) / candidate_v_a_pi2 * 100 - \
                         abs(v_results[12]["a_phys"] - candidate_v_a_pi2) / candidate_v_a_pi2 * 100
    v_trend_narrow_g = abs(v_results[18]["a_phys"] - candidate_v_a_g) / candidate_v_a_g * 100 - \
                       abs(v_results[12]["a_phys"] - candidate_v_a_g) / candidate_v_a_g * 100
    v_trend_wide_pi2 = abs(v_results_wide[18]["a_phys"] - candidate_v_a_pi2) / candidate_v_a_pi2 * 100 - \
                       abs(v_results_wide[12]["a_phys"] - candidate_v_a_pi2) / candidate_v_a_pi2 * 100
    v_trend_wide_g = abs(v_results_wide[18]["a_phys"] - candidate_v_a_g) / candidate_v_a_g * 100 - \
                     abs(v_results_wide[12]["a_phys"] - candidate_v_a_g) / candidate_v_a_g * 100

    print(f"  N=12 → 18 TREND (negative = converging, positive = diverging):")
    print()
    print(f"  Π_TT (a vs 4/π²):")
    print(f"    narrow window: {TT_trend_narrow:+.3f}%")
    print(f"    wide window:   {TT_trend_wide:+.3f}%")
    print()
    print(f"  Π_v (a vs 1/π²):")
    print(f"    narrow window: {v_trend_narrow_pi2:+.3f}%")
    print(f"    wide window:   {v_trend_wide_pi2:+.3f}%")
    print()
    print(f"  Π_v (a vs 1/g = 0.10):")
    print(f"    narrow window: {v_trend_narrow_g:+.3f}%")
    print(f"    wide window:   {v_trend_wide_g:+.3f}%")
    print()
    print(f"  DECISION TREE:")
    print(f"    If Π_TT shows clean convergence (negative trend, |dev| → 0) AND")
    print(f"    Π_v shows the same (under narrow window): Phase A's slow convergence")
    print(f"    was a wide-window artifact, structural form pinnable.")
    print()
    print(f"    If Π_TT converges but Π_v doesn't (under matched conditions):")
    print(f"    Π_v genuinely differs structurally — different mechanism/hierarchy.")
    print()
    print(f"    If neither converges cleanly: my probe may have a precision issue")
    print(f"    common to both; need to re-examine the BZ integration setup.")


if __name__ == "__main__":
    main()
