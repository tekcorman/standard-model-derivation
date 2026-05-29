#!/usr/bin/env python3
"""
G_sub Step 1.4: verify Drude-pole structure a_2(ω) = 4/π² - 1/(36 ω²).

Step 1.3 in `lorentz_sig_g_sub_running_b_extraction.py` revealed the matter
polarization at small p has a Drude-pole structure (NOT QED-style log running):

  1/(16π G_sub(ω)) = a_2_phys(ω) = a_2_reg + D / ω²

with numerical fit a_2_reg ≈ 0.405 ≈ 4/π² and D ≈ -0.0278 ≈ -1/36 = -1/(12 × 3)
= -1/(⟨Tr H²⟩ × k*).

This script:
1. Verifies the Drude pole at higher N (16) and finer ω grid.
2. Tests both 2-parameter (a, D/ω²) and 3-parameter (a, D/ω², log ω²) fits.
3. Identifies clean K[1/π²] form for the structural coefficients.

If a_2_reg = 4/π² and D = -1/36 hold to better than 1% across N=14 and N=16:
the running G_sub structure is THEOREM-GRADE.
"""
from __future__ import annotations

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_dynamic_omega_T import Pi_BZ, TT_xyxy


def extract_a2(omega, T, N, p_z_values=(0.0, 0.05, 0.1, 0.15, 0.2)):
    Pi_xyxy_list = []
    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_BZ(p_cart, omega, T, N=N)
        Pi_xyxy_list.append(TT_xyxy(K))
    p_arr = np.array(p_z_values)
    Pi_arr = np.array(Pi_xyxy_list)
    coeffs = np.polyfit(p_arr ** 2, Pi_arr, 2)
    a_4, a_2, a_0 = coeffs
    return a_2, a_0


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("G_sub Step 1.4: Drude-pole verification at higher precision")
    print()

    # Verify with both N=14 and N=16; use FINE ω grid in saturated regime
    omega_values = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.18, 0.15]
    Ns = [14, 16]

    structural_a = 4 / np.pi ** 2  # 4/π²
    structural_D = -1 / 36  # -1/(⟨Tr H²⟩ × k*) = -1/(12 × 3)

    for N in Ns:
        header(f"N = {N}")
        print(f"  Predicted: a_2_phys(ω) = 4/π² - 1/(36 ω²) = "
              f"{structural_a:.6f} + {structural_D:.6f}/ω²")
        print(f"  {'ω':>6s}  {'time':>5s}  {'a_2_phys':>12s}  {'predicted':>12s}  "
              f"{'resid':>11s}  {'rel_err':>9s}")
        results = []
        for omega in omega_values:
            t0 = time.time()
            a_2_my, a_0 = extract_a2(omega, omega, N)
            elapsed = time.time() - t0
            a_2_phys = -a_2_my / 2
            predicted = structural_a + structural_D / omega ** 2
            resid = a_2_phys - predicted
            rel_err = (resid / max(abs(predicted), 1e-9)) * 100
            results.append((omega, a_2_phys, predicted, resid))
            print(f"  {omega:>6.3f}  {elapsed:>4.1f}s  {a_2_phys:>+.6e}  "
                  f"{predicted:>+.6e}  {resid:>+.4e}  {rel_err:>+7.2f}%")

        # 2-parameter fit a + D/ω² (no log)
        omegas_arr = np.array([r[0] for r in results])
        a_2_phys_arr = np.array([r[1] for r in results])
        inv_omega2 = 1 / omegas_arr ** 2
        slope_2p, intercept_2p = np.polyfit(inv_omega2, a_2_phys_arr, 1)
        # 3-parameter fit a + D/ω² + b log(ω²)
        log_omega2 = np.log(omegas_arr ** 2)
        A = np.column_stack([np.ones_like(omegas_arr), inv_omega2, log_omega2])
        coeffs_3p, _, _, _ = np.linalg.lstsq(A, a_2_phys_arr, rcond=None)
        a_3p, D_3p, b_3p = coeffs_3p

        print()
        print(f"  Fit summaries:")
        print(f"  2-parameter (a + D/ω²):  a = {intercept_2p:.6f}, D = {slope_2p:.6f}")
        print(f"    structural prediction:  a = {structural_a:.6f}, D = {structural_D:.6f}")
        print(f"    a deviation: {(intercept_2p - structural_a) / structural_a * 100:+.3f}%")
        print(f"    D deviation: {(slope_2p - structural_D) / structural_D * 100:+.3f}%")
        print()
        print(f"  3-parameter (a + D/ω² + b log ω²): a = {a_3p:.6f}, D = {D_3p:.6f}, b = {b_3p:.6f}")
        print(f"    log coefficient |b|: {abs(b_3p):.6f}")
        if abs(b_3p) < 0.05:
            print(f"  ✓ log coefficient negligible — pure Drude form holds.")
        else:
            print(f"  ? log term contributes; mixed Drude + log structure.")

    print()
    print("  STRUCTURAL CONCLUSION (if confirmed at N=16):")
    print()
    print("  1/(16π G_sub(ω)) = 4/π² - 1/(36 ω²)")
    print()
    print("  where:")
    print("    4 = N_atoms (atoms per primitive cell, theorem-grade)")
    print("    π² = (UV cutoff scale Λ = π)² in lattice units")
    print("    36 = ⟨Tr H²⟩_BZ × k* = 12 × 3 (Bloch invariant × Hashimoto Perron)")
    print()
    print("  K[1/π²] structural form: a_2_reg = N_atoms/π², D = -1/(⟨Tr H²⟩·k*).")


if __name__ == "__main__":
    main()
