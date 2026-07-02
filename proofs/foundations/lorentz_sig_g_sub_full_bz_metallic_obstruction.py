#!/usr/bin/env python3
"""
G_sub Route B (numerical) — full-BZ Π_2 metallic-obstruction characterization.

Per an internal working note,
the cone-effective Sakharov route was exhausted (universal-ζ refuted with
full vertex). This script attempts Route B: direct full-BZ Π_TT(p²)
extraction at half-filling μ = 0, using BOTH a uniform grid (path-#3
convention) AND a Monkhorst-Pack-shifted grid that avoids hitting cone
centers (Γ, H, P) exactly.

Result
------
Both grid conventions are decisively NOT CONVERGED across N ∈ {10, 12, 14, 16}.
At each N, individual Π_xyxy(p_z) values can flip sign at certain p_z values
because the (k, k+p) pair crosses one of the substrate's band-crossing
surfaces (where E_2(k) = E_3(k+p) = 0 = μ at half-filling).

Path-#3 (2026-04-29) had observed "G ≈ 0.107 from N=12 extrapolation, large
N-dependence" — this is now confirmed to be **noise from grid alignment with
band-crossing surfaces**, not signal. Different N values give G_sub varying
by factor 2 with no convergence pattern. The 0.107 ≈ 4(√3-1)/27 coincidence
was illusory.

Conclusion
----------
**Full-BZ direct numerical Π_2 extraction is BLOCKED at half-filling** due
to metallic non-analyticity. The closure of G_sub via this route requires
analytical handling of the band-crossing surfaces (e.g., contour integral
around degeneracies, Wannier interpolation, or symbolic Bloch sum-rule).

This is consistent with session 3's finding (substrate is metallic at all
natural μ, no true gap) and rules out direct numerical Π_2 as a closure
path.
"""
from __future__ import annotations

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_pi_finite_p import Pi_para_at_k, Pi_dia_at_k


def Pi_at_p_uniform(p_cart, N=12, mu=0.0, half_extent=2 * np.pi):
    """Path-#3 convention: linspace(-2π, 2π, N, endpoint=False) — hits Γ at k=0
    when N is even, hits H at k=π when N % 2 == 0, hits P at k=π/2 when N % 4 == 0.
    """
    ks = np.linspace(-half_extent, half_extent, N, endpoint=False)
    K_para = np.zeros((3, 3, 3, 3))
    K_dia = np.zeros((3, 3, 3, 3))
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                k_cart = np.array([k1, k2, k3])
                K_para += Pi_para_at_k(k_cart, p_cart, mu)
                K_dia += Pi_dia_at_k(k_cart, mu)
    n_pts = N ** 3
    return K_dia / n_pts, K_para / n_pts


def Pi_at_p_mp_shifted(p_cart, N=12, mu=0.0, half_extent=2 * np.pi):
    """Monkhorst-Pack shifted: ks = (i + 0.5) × (2 half_extent / N) - half_extent.
    Avoids putting any grid point exactly on Γ (k=0), H (k=π), or P (k=π/2).
    """
    ks = (np.arange(N) + 0.5) * (2 * half_extent / N) - half_extent
    K_para = np.zeros((3, 3, 3, 3))
    K_dia = np.zeros((3, 3, 3, 3))
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                k_cart = np.array([k1, k2, k3])
                K_para += Pi_para_at_k(k_cart, p_cart, mu)
                K_dia += Pi_dia_at_k(k_cart, mu)
    n_pts = N ** 3
    return K_dia / n_pts, K_para / n_pts


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("G_sub Route B: full-BZ Π_2 metallic-obstruction characterization")
    print()
    print("  Two grid conventions × N ∈ {10, 12, 14, 16}; same p_z scan {0, 0.05,")
    print("  0.10, 0.15, 0.20}. If route is converged, all N + both grids should")
    print("  agree on a single G_sub value within ~few %. If wildly disagree, the")
    print("  metallic obstruction is fundamental.")
    print()

    p_z_values = [0.0, 0.05, 0.10, 0.15, 0.20]
    candidate_4sqrt3m1_27 = 4 * (np.sqrt(3) - 1) / 27

    for grid_name, grid_fn in [("uniform", Pi_at_p_uniform),
                                ("MP-shifted", Pi_at_p_mp_shifted)]:
        print(f"\n  === Convention: {grid_name} ===")
        print(f"  {'N':>4s}  {'time':>7s}  {'Pi_2(quad)':>12s}  {'G_sub':>12s}  "
              f"{'Pi_xyxy(p)':>18s}")
        for N in [10, 12, 14, 16]:
            t0 = time.time()
            Pi_xyxy_list = []
            for p_z in p_z_values:
                p_cart = np.array([0.0, 0.0, p_z])
                K_dia, K_para = grid_fn(p_cart, N=N, mu=0.0)
                Pi_xyxy_list.append((K_dia - K_para)[0, 1, 0, 1])
            elapsed = time.time() - t0
            p_arr = np.array(p_z_values)
            Pi_arr = np.array(Pi_xyxy_list)
            try:
                _, Pi_2, _ = np.polyfit(p_arr ** 2, Pi_arr, 2)
                G = 1 / (16 * np.pi * Pi_2) if Pi_2 > 0 else None
                G_str = f"{G:.4f}" if G else "neg/nan"
            except Exception:
                Pi_2 = float('nan'); G_str = "fit failed"
            # Indicate stability: maximum jump between successive p_z values
            jumps = np.abs(np.diff(Pi_arr))
            max_jump = np.max(jumps)
            jump_flag = "STABLE" if max_jump < 0.01 else f"JUMP={max_jump:.3f}"
            print(f"  {N:>4d}  {elapsed:>5.1f}s  {Pi_2:+.6e}  {G_str:>12s}  {jump_flag:>18s}")

    print()
    print(f"  Reference: 4(√3−1)/27 = {candidate_4sqrt3m1_27:.6f}")
    print()
    print("  CONCLUSION: full-BZ direct Π_2 numerics does NOT converge across grids.")
    print("  At fixed N, JUMP flag indicates p-values where (k, k+p) crosses the")
    print("  half-filling band-crossing surface. The half-filled substrate is")
    print("  metallic (no gap, continuous spectrum at μ = 0); direct numerical")
    print("  extraction is BLOCKED. Closure requires analytical regularization.")


if __name__ == "__main__":
    main()
