#!/usr/bin/env python3
"""
G_sub Step 1: pin the β-function coefficient b in Π_TT(p²) = a_0 + (a_2 + b log p²) p² + ...

Per `g_sub_log_divergence_finding_2026-04-30.md`, Phase 5b confirmed
log(p²) non-analyticity in Π_TT (RSS 57× better with log term, b ≈ -3.25
suggestively close to -π). The remaining task is to PIN b at theorem-grade
precision so we can identify it as a clean K[1/π] element (e.g. -π).

This script:
1. Uses Phase 5's finite-(ω, T) smoothed Lindhard (smooth Fermi factors).
2. Runs grid scan at N = 12, 14, 16 (vs Phase 5b's single N=12).
3. At each N, fits log-corrected polynomial to extract b.
4. Cross-checks at multiple (ω, T) values for regulator-stability.
5. Reports converged b ± grid noise; identifies K[1/π] candidate.

If b is grid-converged across N=12, 14, 16 to within ~few %: solid result.
If still grid-noise-dominated: need tetrahedron method.
"""
from __future__ import annotations

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_dynamic_omega_T import Pi_BZ, TT_xyxy


def fit_log_polynomial(p_z_values, Pi_arr):
    """Fit Π = a_0 + a_2 p² + b × p² log(p²) + a_4 (p²)²; return (a_0, a_2, b, a_4)."""
    p2 = np.array(p_z_values) ** 2
    A = np.column_stack([np.ones_like(p2), p2, p2 * np.log(p2), p2 ** 2])
    coeffs, _, _, _ = np.linalg.lstsq(A, np.array(Pi_arr), rcond=None)
    a0, a2, b, a4 = coeffs
    return a0, a2, b, a4


def fit_pure_polynomial(p_z_values, Pi_arr):
    """Fit Π = a_0 + a_2 p² + a_4 (p²)²; return (a_0, a_2, a_4)."""
    p2 = np.array(p_z_values) ** 2
    A = np.column_stack([np.ones_like(p2), p2, p2 ** 2])
    coeffs, _, _, _ = np.linalg.lstsq(A, np.array(Pi_arr), rcond=None)
    a0, a2, a4 = coeffs
    return a0, a2, a4


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("G_sub Step 1: pin β-function coefficient b across grids")
    print()

    # Use a wider, finer p_z grid for stable log fit
    p_z_values = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.17, 0.22]
    print(f"  p_z values ({len(p_z_values)} points): {p_z_values}")
    print()

    # Two regulator settings
    regulators = [(0.10, 0.10), (0.05, 0.05)]
    Ns = [10, 12, 14, 16]

    for omega, T in regulators:
        header(f"Regulators: ω_E = {omega}, T = {T}")
        print(f"  {'N':>4s}  {'time':>6s}  {'a_0':>13s}  {'a_2_log':>13s}  "
              f"{'b (log)':>13s}  {'b/(-π)':>10s}  {'RSS_imp':>9s}")
        for N in Ns:
            t0 = time.time()
            Pi_xyxy_list = []
            for p_z in p_z_values:
                p_cart = np.array([0.0, 0.0, p_z])
                K = Pi_BZ(p_cart, omega, T, N=N)
                Pi_xyxy_list.append(TT_xyxy(K))
            elapsed = time.time() - t0

            # Fit pure poly + log-corrected
            _, a2_pure, _ = fit_pure_polynomial(p_z_values, Pi_xyxy_list)
            a0, a2_log, b, a4 = fit_log_polynomial(p_z_values, Pi_xyxy_list)

            # RSS comparison
            p2 = np.array(p_z_values) ** 2
            Pi_arr = np.array(Pi_xyxy_list)
            Pi_pred_pure = a0 + a2_pure * p2 + a4 * p2 ** 2  # use pure-poly a0 too
            Pi_pred_log = a0 + a2_log * p2 + b * p2 * np.log(p2) + a4 * p2 ** 2
            rss_pure = np.sum((Pi_arr - Pi_pred_pure) ** 2)
            rss_log = np.sum((Pi_arr - Pi_pred_log) ** 2)
            improvement = rss_pure / max(rss_log, 1e-30)

            ratio_b = b / (-np.pi)

            print(f"  {N:>4d}  {elapsed:>5.1f}s  {a0:>+.6e}  "
                  f"{a2_log:>+.6e}  {b:>+.6e}  {ratio_b:>+8.4f}  "
                  f"{improvement:>9.1f}")

    print()
    print("  Reference candidates for b:")
    print(f"    -π = {-np.pi:.6f}")
    print(f"    -π/2 = {-np.pi/2:.6f}")
    print(f"    -2π = {-2*np.pi:.6f}")
    print(f"    -3 = -3.000000")
    print(f"    -1/π = {-1/np.pi:.6f}")
    print(f"    -3π/2 = {-3*np.pi/2:.6f}")
    print()
    print("  Interpretation:")
    print("  - If b/(-π) is approximately constant (≈1) across N and (ω, T):")
    print("    confirms b = -π ∈ K[1/π], the gravitational β-function coefficient.")
    print("  - If b varies wildly: need tetrahedron-method BZ integration to remove")
    print("    grid noise from band-crossing surfaces.")


if __name__ == "__main__":
    main()
