#!/usr/bin/env python3
"""
G_sub Step 1.3: extract β-function b from a_2(ω) RUNNING in saturated regime.

Per Phase 5/5b/5c findings, the matter polarization Π_TT(p²) has more structure
than a single QED-like p² log(p²) term. Direct fit of `b × p² × log(p²+ω²)`
gives b that varies wildly with ω.

The cleaner interpretation: at fixed regulator ω, polynomial fit gives an
EFFECTIVE a_2(ω) = a_2_true + b × log(ω²) (up to higher-order corrections in
ω/p ratios). Then b is recovered as the slope of a_2(ω) vs log(ω²).

This script runs in the SATURATED regime where ω is comparable to or larger
than typical p values used (ω > 0.15 for our p_z ≤ 0.2). In this regime,
the matter polarization at small p is regularized cleanly by ω, and the
polynomial fit is well-defined.

Method
------
1. At N=14, fixed (ω, T) with T=ω: compute Π_TT(p²) at small p.
2. Polynomial fit Π = a_0 + a_2 p² + a_4 (p²)²; extract a_2(ω).
3. Repeat for ω ∈ {0.5, 0.4, 0.3, 0.2, 0.15}.
4. Linear fit a_2(ω) vs log(ω²): slope = b (gravitational β-function).
5. Identify b in K[1/π].
"""
from __future__ import annotations

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_dynamic_omega_T import Pi_BZ, TT_xyxy


def extract_a2_pure(omega, T, N, p_z_values=(0.0, 0.05, 0.1, 0.15, 0.2)):
    Pi_xyxy_list = []
    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_BZ(p_cart, omega, T, N=N)
        Pi_xyxy_list.append(TT_xyxy(K))
    p_arr = np.array(p_z_values)
    Pi_arr = np.array(Pi_xyxy_list)
    coeffs = np.polyfit(p_arr ** 2, Pi_arr, 2)
    a_4, a_2, a_0 = coeffs
    return a_2, a_0, Pi_arr


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("G_sub Step 1.3: extract β from a_2(ω) running in saturated regime")
    print()

    # Saturated regime: ω comparable to or larger than typical p (= 0.05 - 0.2)
    omega_values = [0.5, 0.4, 0.3, 0.25, 0.20, 0.15]
    N = 14

    print(f"  N={N}, T = ω. Saturated regime: ω ∈ {omega_values}")
    print(f"  p_z values: 0.0, 0.05, 0.1, 0.15, 0.2 (all ≤ smallest ω)")
    print()
    print(f"  {'ω=T':>6s}  {'time':>5s}  {'a_0':>12s}  {'a_2_my':>12s}  "
          f"{'a_2_phys':>12s}  {'log(ω²)':>9s}")

    results = []
    for omega in omega_values:
        t0 = time.time()
        a_2_my, a_0, _ = extract_a2_pure(omega, omega, N)
        elapsed = time.time() - t0
        a_2_phys = -a_2_my / 2  # path-#3 sign convention; factor 2 for direction-doubling
        results.append((omega, a_2_my, a_2_phys, np.log(omega ** 2)))
        print(f"  {omega:>6.3f}  {elapsed:>4.1f}s  {a_0:>+.6e}  "
              f"{a_2_my:>+.6e}  {a_2_phys:>+.6e}  {np.log(omega**2):>+8.4f}")

    print()
    # Linear fit a_2_phys vs log(ω²)
    omegas = np.array([r[0] for r in results])
    a_2_phys_arr = np.array([r[2] for r in results])
    log_omega2_arr = np.array([r[3] for r in results])

    slope, intercept = np.polyfit(log_omega2_arr, a_2_phys_arr, 1)
    print(f"  Linear fit a_2_phys(ω) = a_2_true + b × log(ω²):")
    print(f"    slope (= b) = {slope:.6f}")
    print(f"    intercept (= a_2_true at ω = 1) = {intercept:.6f}")
    print()
    # Goodness-of-fit: residuals
    a_2_pred = intercept + slope * log_omega2_arr
    residuals = a_2_phys_arr - a_2_pred
    rss = np.sum(residuals ** 2)
    print(f"    residuals: {residuals}")
    print(f"    RSS = {rss:.6e}")
    print()

    # K[1/π] candidate identification for b
    candidates = {
        "+2/π":   2 / np.pi,
        "+1/π":   1 / np.pi,
        "+1/3":   1 / 3,
        "+1/4":   1 / 4,
        "+3/(8π)": 3 / (8 * np.pi),
        "+1/(2π)": 1 / (2 * np.pi),
        "+π/8":   np.pi / 8,
        "+π/12":  np.pi / 12,
        "+√3/(2π)": np.sqrt(3) / (2 * np.pi),
        "+1/(π√3)": 1 / (np.pi * np.sqrt(3)),
    }
    # Also add negative versions
    neg_candidates = {f"-{name[1:]}": -val for name, val in candidates.items()}
    candidates.update(neg_candidates)

    print(f"  K[1/π] candidate identification:")
    print(f"  {'candidate':>12s}  {'value':>12s}  {'b - candidate':>15s}  {'rel error':>12s}")
    for name, val in sorted(candidates.items(), key=lambda x: abs(x[1] - slope)):
        rel_err = (slope - val) / val * 100 if abs(val) > 1e-10 else float('inf')
        print(f"  {name:>12s}  {val:>+.6e}  {slope - val:>+12.6e}  {rel_err:>+8.3f}%")
        if abs(slope - val) > 0.5 * abs(val):
            break  # stop printing after the close ones

    print()
    print("  Methodological note: this works in the SATURATED regime where ω > p")
    print("  (so polynomial fit gives a clean a_2 + b log(ω²) offset). At small ω,")
    print("  the log term mixes with p²-running and the simple offset breaks down.")


if __name__ == "__main__":
    main()
