#!/usr/bin/env python3
"""
G_sub session 5: convergence study of μ_iso interband on grid refinement.

After settling the sign convention (subtractive K_dia - K_para is correct
for interband elastic modulus, per `lorentz_sig_g_sub_finite_diff_fixed_filling.py`),
we need to nail the structural form.

Candidates from session 5 entry:
  15/(8π²) ≈ 0.189977  (Class-A audit form, (2k*-1)·k* / (8π²))
  3/(16) ≈ 0.18750
  1/(8π²) × 15 ≈ 0.189977 (same as above)
  some O(1)/π² combination

This script:
  1. Computes K_dia - K_para μ_iso on grids 12³, 16³, 20³, 24³, 32³.
  2. Extrapolates to N → ∞.
  3. Compares to candidate clean rationals.
"""
from __future__ import annotations

import numpy as np
from fractions import Fraction
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_elastic_moduli import bz_average_full, voigt_components


def voigt_iso_mu(C_11, C_12, C_44):
    return (C_11 - C_12 + 3 * C_44) / 5


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("μ_iso convergence study: K_dia - K_para on increasing grids")
    print()
    print("  Grid  | μ_iso    | C_11      | C_12      | C_44      | cubic-aniso")
    print("  ------+----------+-----------+-----------+-----------+------------")
    results = []
    for N in [8, 12, 16, 20, 24]:
        K_para, K_dia, _ = bz_average_full(N_grid=N, mu=0.0, half_extent=2*np.pi)
        K_sub = K_dia - K_para
        v = voigt_components(K_sub)
        mu = voigt_iso_mu(v['C_11'], v['C_12'], v['C_44'])
        aniso = v['2C_44 - (C_11 - C_12)']
        results.append((N, mu, v['C_11'], v['C_12'], v['C_44'], aniso))
        print(f"  {N:>4d}³ | {mu:+.6f} | {v['C_11']:+.6f} | {v['C_12']:+.6f} | {v['C_44']:+.6f} | {aniso:+.6f}")

    # Candidates to compare
    candidates = {
        '15/(8π²)':       15/(8*np.pi**2),
        '3/16':           3/16,
        '1/(2π² + π/3)':  1/(2*np.pi**2 + np.pi/3),
        '5/(27)':         5/27,
        '5/(8π)':         5/(8*np.pi),
        '6/π²':           6/np.pi**2,
        'π²/52':          np.pi**2/52,
        '1/(2π² × 0.27)': 1/(2*np.pi**2 * 0.27),
        '4/(3π²)':        4/(3*np.pi**2),
        '15/(2π³)':       15/(2*np.pi**3),
        '5/(2π³)':        5/(2*np.pi**3),
        '(√3-1)/(4π³)':   (np.sqrt(3)-1)/(4*np.pi**3),
        '3/(2π²) × 1/8':  3/(2*np.pi**2) / 8,
        '1/(8π²) × 4(√3-1)': 4*(np.sqrt(3)-1)/(8*np.pi**2),
    }

    header("Candidate form matching")
    print()
    print(f"  {'candidate':<30s}  {'value':>10s}  {'ratio (best grid)':>18s}")
    print("  " + "-" * 64)
    best_grid_mu = results[-1][1]
    for name, val in sorted(candidates.items(), key=lambda x: abs(x[1] - best_grid_mu)):
        ratio = best_grid_mu / val
        flag = "  ←" if abs(ratio - 1) < 0.05 else ""
        print(f"  {name:<30s}  {val:>10.6f}  {ratio:>18.6f}{flag}")

    header("Richardson extrapolation N → ∞")
    print()
    # Use last 3 grids for extrapolation; assume convergence as 1/N^p for some p
    N1, mu1 = results[-3][0], results[-3][1]
    N2, mu2 = results[-2][0], results[-2][1]
    N3, mu3 = results[-1][0], results[-1][1]
    # Try p=1 (1/N), p=2 (1/N²), p=4 (1/N^4) extrapolations
    for p in [1, 2, 3, 4, 6]:
        x1, x2, x3 = (1/N1)**p, (1/N2)**p, (1/N3)**p
        # Linear extrap from (x2, mu2) and (x3, mu3): mu(0) = mu3 + x3 * (mu3 - mu2)/(x3 - x2)... slope*(0-x3)
        slope = (mu3 - mu2) / (x3 - x2)
        mu_inf = mu3 - slope * x3
        print(f"  Assume convergence as 1/N^{p}: μ_inf ≈ {mu_inf:+.6f}")

    header("Conclusion")
    print()
    print(f"  Best-grid {results[-1][0]}³: μ_iso = {best_grid_mu:.6f}")
    print(f"  Closest clean form (above): 15/(8π²) = 0.189977")
    print()
    print(f"  Convergence behavior:")
    for r in results:
        print(f"    {r[0]}³: μ_iso = {r[1]:.6f}")


if __name__ == "__main__":
    main()
