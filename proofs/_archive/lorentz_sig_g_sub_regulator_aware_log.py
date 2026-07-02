#!/usr/bin/env python3
"""
G_sub Step 1.2: regulator-aware log fit.

`lorentz_sig_g_sub_log_pin_b.py` showed that the simple fit
`Π = a_0 + a_2 p² + b p² log(p²) + a_4 (p²)²` gives a regulator-dependent
b coefficient (b ≈ -0.36 at (ω, T) = (0.1, 0.1) but b ≈ -3 at (0.05, 0.05)).

For a QED-style β-function, the proper analytic form with finite IR
regulator ω is:

  Π(p²; ω) = a_0 + p² × [a_2 + b × log((p² + ω²)/μ_0²)]

where μ_0 is some reference scale. With this form, b should be regulator-
INDEPENDENT (the regulator dependence is built into log argument).

This script:
1. Re-fits the same data using the regulator-aware log form.
2. Tests whether b is now regulator-stable across (ω, T) values.
3. If yes: confirms b ∈ K[1/π] and identifies it.
"""
from __future__ import annotations

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_dynamic_omega_T import Pi_BZ, TT_xyxy


def fit_regulator_aware(p_z_values, Pi_arr, omega):
    """Fit Π = a_0 + a_2 p² + b × p² × log((p² + ω²)/μ²_0) + a_4 (p²)²
    with μ_0 = 1 (lattice unit). Returns (a_0, a_2, b, a_4)."""
    p2 = np.array(p_z_values) ** 2
    A = np.column_stack([
        np.ones_like(p2),
        p2,
        p2 * np.log(p2 + omega ** 2),  # μ_0 = 1
        p2 ** 2,
    ])
    coeffs, _, _, _ = np.linalg.lstsq(A, np.array(Pi_arr), rcond=None)
    a0, a2, b, a4 = coeffs
    return a0, a2, b, a4


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("G_sub Step 1.2: regulator-aware log fit Π = a_0 + p²[a_2 + b log((p²+ω²))] + a_4(p²)²")
    print()

    # Wider p_z range to better resolve large-p log behavior
    p_z_values = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.17, 0.22]
    print(f"  p_z values ({len(p_z_values)} points): {p_z_values}")
    print()

    # Multiple (ω, T) values to test regulator stability
    regulators = [(0.20, 0.20), (0.15, 0.15), (0.10, 0.10), (0.075, 0.075), (0.05, 0.05)]
    N = 14  # large enough for grid convergence at moderate ω

    print(f"  N={N}; testing regulator-independence of b:")
    print(f"  {'ω=T':>6s}  {'time':>5s}  {'a_0':>13s}  {'a_2':>12s}  "
          f"{'b':>12s}  {'b/(-1/π)':>10s}  {'b/(-π/8)':>10s}  {'b/(-π)':>9s}")

    results = []
    for omega, T in regulators:
        t0 = time.time()
        Pi_xyxy_list = []
        for p_z in p_z_values:
            p_cart = np.array([0.0, 0.0, p_z])
            K = Pi_BZ(p_cart, omega, T, N=N)
            Pi_xyxy_list.append(TT_xyxy(K))
        elapsed = time.time() - t0

        a0, a2, b, a4 = fit_regulator_aware(p_z_values, Pi_xyxy_list, omega)
        ratio_inv_pi = b / (-1 / np.pi)
        ratio_pi_8 = b / (-np.pi / 8)
        ratio_pi = b / (-np.pi)
        results.append((omega, b, ratio_inv_pi, ratio_pi_8, ratio_pi))
        print(f"  {omega:>6.3f}  {elapsed:>4.1f}s  {a0:>+.6e}  {a2:>+.6e}  "
              f"{b:>+.6e}  {ratio_inv_pi:>+8.4f}  {ratio_pi_8:>+8.4f}  {ratio_pi:>+7.4f}")

    print()
    print("  Reference candidates for b (K[1/π] elements):")
    print(f"    -1/π  = {-1/np.pi:.6f}")
    print(f"    -π/8  = {-np.pi/8:.6f}")
    print(f"    -1/3  = -0.333333")
    print(f"    -π/12 = {-np.pi/12:.6f}")
    print(f"    -2/3π = {-2/(3*np.pi):.6f}")
    print()
    bs = np.array([r[1] for r in results])
    if len(bs) >= 3:
        b_mean = np.mean(bs)
        b_std = np.std(bs)
        print(f"  b across regulators: mean = {b_mean:.6f}, std = {b_std:.6f}")
        print(f"    relative std = {abs(b_std/b_mean)*100:.2f}%")
        if abs(b_std / b_mean) < 0.05:
            print(f"  ✓ b is regulator-stable. The β-function coefficient is b ≈ {b_mean:.5f}.")
            # Identify K[1/π] candidate
            candidates = {
                "-1/π": -1 / np.pi,
                "-π/8": -np.pi / 8,
                "-π/12": -np.pi / 12,
                "-1/3": -1/3,
                "-3/(8π)": -3/(8*np.pi),
                "-1/(2π)": -1/(2*np.pi),
                "-2/(3π)": -2/(3*np.pi),
            }
            best = min(candidates.items(),
                        key=lambda x: abs(x[1] - b_mean))
            best_name, best_val = best
            print(f"  Closest K[1/π] match: {best_name} = {best_val:.6f}, "
                  f"deviation: {(b_mean - best_val)/best_val * 100:.2f}%")
        else:
            print(f"  ✗ b is NOT regulator-stable; fit form still wrong or grid noise.")


if __name__ == "__main__":
    main()
