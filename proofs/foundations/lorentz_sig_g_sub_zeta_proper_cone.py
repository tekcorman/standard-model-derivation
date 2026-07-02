#!/usr/bin/env python3
"""
G_sub multi-valley closure attempt: ζ_P with full vertex at PROPER cone-effective Λ.

K-meta-theorem (`docs/theorems/theorem_lattice_coupling_general.md`) demands G_sub ∈ K =
ℚ(√2, √3, √5). Of session-5 candidates, only session-4's 4(√3-1)/27 ≈ 0.108
passes this filter. But its derivation used universal-ζ assumption that
path-#4 (`g_sub_session5_path4_finding.md`) numerically falsified using NAIVE
P-cone vertex.

This script:
1. Confirms the path-#4 falsification used naive vertex (V_1 only); the
   actual P-cone strain vertex has a non-zero V_0 piece (path-#5).
2. Path-#6 used full vertex but at Λ=π, which extends outside cone-effective
   validity → result is suspect.
3. **This script: compute ζ_P_full at small Λ_cone where cone-effective is
   valid.** Test whether ζ_P_full → ζ_Γ = 27/(512π³) at small Λ_cone (i.e.,
   universal-ζ holds with proper vertex).

If yes: session-4's 4(√3-1)/27 is theorem-grade with K-meta-theorem support.

Method: same as path-#6 (`lorentz_sig_g_sub_p_cone_full_vertex.py`) but at
small Λ_cone ∈ {0.2, 0.3, 0.5, 0.7}.
"""
from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_p_cone_full_vertex import (
    Pi_p_cone_at_external_p, TT_project_zhat,
)


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def extract_a2(P_cart, Lambda_cone, target_ev, n_radial=15, n_theta=10, n_phi=10):
    """Compute Π_TT(p_z) at finite p_z, extract a_2 (= leading p² coefficient)."""
    # Use small p_z values relative to Λ_cone for clean p² extraction
    p_max = min(0.15, Lambda_cone * 0.3)
    p_z_values = [0.0, 0.5*p_max, p_max]
    Pi_TT_list = []
    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        # Modify Pi_p_cone_at_external_p to take target_ev
        # For Γ: target = -1; for P: target = +√3
        Pi = Pi_p_cone_at_external_p(p_cart, P_cart,
                                      n_radial=n_radial,
                                      n_theta=n_theta,
                                      n_phi=n_phi,
                                      Lambda_cone=Lambda_cone,
                                      target_ev=target_ev)
        Pi_TT = TT_project_zhat(Pi)
        Pi_TT_list.append(Pi_TT)
    p_arr = np.array(p_z_values)
    Pi_arr = np.array(Pi_TT_list)
    # Quadratic fit in p²
    coeffs = np.polyfit(p_arr ** 2, Pi_arr, 2)
    a_4, a_2, a_0 = coeffs
    return a_0, a_2, a_4


def main():
    header("ζ_P_full vs ζ_Γ at proper cone-effective Λ_cone")
    print()
    print("  Test: does ζ_P_full → ζ_Γ as Λ_cone shrinks (cone-effective regime)?")
    print("  If yes: universal-ζ holds with proper vertex → session-4's 4(√3-1)/27")
    print("    is theorem-grade with K-meta-theorem confirmation.")
    print()

    P_cart = np.pi * np.array([1.0, 1.0, 1.0])  # P-point
    Gamma_cart = np.array([0.0, 0.0, 0.0])      # Γ-point
    target_P = np.sqrt(3)
    target_Gamma = -1.0
    v_F_P = np.sqrt(3) / 6
    v_F_Gamma = 1/2

    zeta_universal = 27 / (512 * np.pi ** 3)

    print(f"  Reference: ζ_universal = 27/(512π³) = {zeta_universal:.6e}")
    print()

    print(f"  {'Λ_cone':>8s}  {'a_2_Γ':>14s}  {'ζ_Γ':>14s}  {'a_2_P':>14s}  {'ζ_P':>14s}  {'ζ_P/ζ_Γ':>9s}")
    print(f"  {'-'*78}")

    Lambda_values = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, np.pi]
    for Lambda_cone in Lambda_values:
        # Γ-cone (3-dim subspace, but our function expects 2-dim subspace)
        # For Γ, use target = -1 (eigenvalue of A_adjacency at Γ; spin-1 cone with 3 bands)
        # Note: our function picks 2 bands closest to target, but Γ has 3 bands at -1
        # So function will pick 2 of those 3 randomly. Result may differ from spin-1 calc.
        try:
            a0_G, a2_G, _ = extract_a2(Gamma_cart, Lambda_cone, target_Gamma,
                                        n_radial=15, n_theta=10, n_phi=10)
            zeta_G = a2_G * v_F_Gamma / Lambda_cone ** 2

            a0_P, a2_P, _ = extract_a2(P_cart, Lambda_cone, target_P,
                                        n_radial=15, n_theta=10, n_phi=10)
            zeta_P = a2_P * v_F_P / Lambda_cone ** 2

            ratio = zeta_P / zeta_G if zeta_G != 0 else float('inf')
            print(f"  {Lambda_cone:>8.3f}  {a2_G:>+14.4e}  {zeta_G:>+14.4e}  {a2_P:>+14.4e}  {zeta_P:>+14.4e}  {ratio:>9.4f}")
        except Exception as e:
            print(f"  {Lambda_cone:>8.3f}  ERROR: {e}")

    header("Interpretation")
    print()
    print("  If ζ_P_full → ζ_Γ as Λ_cone shrinks: universal-ζ holds with full vertex.")
    print("  Then session-4's structural form 4(√3-1)/27 is correct.")
    print()
    print("  If ζ_P_full ≠ ζ_Γ: universal-ζ falsified even with full vertex.")
    print("  Multi-valley sum needs revision; K-meta-theorem still demands K-element.")


if __name__ == "__main__":
    main()
