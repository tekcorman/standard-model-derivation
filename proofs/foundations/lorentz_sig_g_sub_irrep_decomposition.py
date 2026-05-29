#!/usr/bin/env python3
"""
G_sub session 5 path #2: O_h irrep decomposition of C^{abcd}.

Tests interpretation #3 of `g_sub_session5_path1_finding.md`: μ_iso ≠ 0
might be a "wrong projection" artifact for the substrate's specific cubic
anisotropy. The graviton TT mode might live in only one O_h irrep of the
5-dim traceless symmetric strain decomposition.

Group theory
------------
A symmetric 3×3 strain tensor ε^{ab} (6 independent components) decomposes
under cubic O_h as:
  6 = 1 (A_1g, trace) ⊕ 2 (E_g, axial) ⊕ 3 (T_2g, cross)

Explicit basis:
- A_1g (1-dim): ε_iso = (1/3) Tr ε × δ^{ab}
- E_g  (2-dim): {(1/√6)(ε_xx + ε_yy - 2 ε_zz), (1/√2)(ε_xx - ε_yy)}
             (or any orthonormal basis of traceless diagonal)
- T_2g (3-dim): {ε_xy + ε_yx, ε_xz + ε_zx, ε_yz + ε_zy} × normalization

Each irrep has a single elastic modulus (Schur's lemma):
- C_A1g = bulk modulus K = (C_11 + 2 C_12) / 3
- C_E_g  = (C_11 - C_12) / 2 (axial-shear modulus)
- C_T2g = 2 C_44 (cross-shear modulus)

For an isotropic medium: C_E_g = C_T_2g (Cauchy relation 2 C_44 = C_11 - C_12).
For cubic with violation: C_E_g ≠ C_T_2g.

For the substrate (24³ data): C_11 = 0.145, C_12 = 0.041, C_44 = 0.274.
- C_A1g = (0.145 + 2×0.041)/3 = 0.076 (bulk)
- C_E_g  = (0.145 - 0.041)/2 = 0.052 (axial)
- C_T2g = 2 × 0.274 = 0.548 (cross)

Voigt iso μ = (C_11 - C_12 + 3 C_44)/5 = (2 C_E_g + 3 C_T2g)/5 averages
the two shear moduli with weights matching their irrep dimensions.

The graviton TT mode is a spin-2 traceless symmetric mode. Under cubic O_h,
spin-2 (5-dim) → E_g (2) ⊕ T_2g (3). The substrate's emergent graviton
might project onto:
  (a) Voigt iso = ALL 5 traceless modes (current default).
  (b) E_g only (2 modes).
  (c) T_2g only (3 modes).
  (d) Some other linear combination determined by the substrate's specific
      structure (e.g., the Iorio vielbein's symmetry properties).

For each of (b), (c), (d), the relevant elastic modulus differs. If one of
these gives a value vanishing in the continuum limit (or a clean rational
form distinct from 15/(8π²)), interpretation #3 of session 5 path #1 is
viable.

Method
------
For each grid size N ∈ {12, 16, 20, 24, 28, 32}, compute K_dia − K_para and
extract:
  C_A1g(N), C_E_g(N), C_T2g(N), μ_iso_Voigt(N)

Plot convergence; check if any irrep's modulus → 0 or to a different clean
value than 15/(8π²).
"""
from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_elastic_moduli import bz_average_full, voigt_components


def irrep_moduli(C_11, C_12, C_44):
    """Decompose into A_1g (bulk), E_g (axial-shear), T_2g (cross-shear)."""
    return {
        'A_1g (bulk K)':    (C_11 + 2*C_12) / 3,
        'E_g  (axial)':     (C_11 - C_12) / 2,
        'T_2g (cross)':     2 * C_44,
        'Voigt μ_iso':      (C_11 - C_12 + 3 * C_44) / 5,
        'Cauchy violation': 2*C_44 - (C_11 - C_12),
    }


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("O_h irrep decomposition of substrate elastic tensor C^{abcd}")
    print()
    print("  Decomposes K_dia − K_para (interband static) on multiple grids.")
    print("  Checks if any specific irrep modulus → 0 (Ward-protected) or")
    print("  → a different clean form than 15/(8π²) ≈ 0.190 (Voigt).")
    print()

    print("  Grid  |  A_1g (K)    E_g (axial)   T_2g (cross)  μ_Voigt    Cauchy viol")
    print("  ------+---------------------------------------------------------------")

    results = []
    for N in [8, 12, 16, 20, 24]:
        K_para, K_dia, _ = bz_average_full(N_grid=N, mu=0.0, half_extent=2*np.pi)
        K_sub = K_dia - K_para
        v = voigt_components(K_sub)
        irr = irrep_moduli(v['C_11'], v['C_12'], v['C_44'])
        results.append((N, irr))
        print(f"  {N:>4d}³ | "
              f"{irr['A_1g (bulk K)']:+.6f}   "
              f"{irr['E_g  (axial)']:+.6f}   "
              f"{irr['T_2g (cross)']:+.6f}   "
              f"{irr['Voigt μ_iso']:+.6f}   "
              f"{irr['Cauchy violation']:+.6f}")

    header("Convergence analysis (mean ± std across N=12..24)")
    print()
    means = {}
    for key in results[0][1].keys():
        vals = [r[1][key] for r in results[1:]]  # skip 8³
        mean = np.mean(vals)
        std = np.std(vals)
        means[key] = (mean, std)
        print(f"  {key:<30s}: {mean:+.6f} ± {std:.6f}  (rel std = {std/abs(mean)*100:.1f}%)")

    header("Candidate clean forms")
    print()

    candidates = {
        '0 (Ward-protected zero)':  0.0,
        '15/(8π²)':                 15/(8*np.pi**2),
        '5/27':                     5/27,
        '3/16':                     3/16,
        '1/(8π²)':                  1/(8*np.pi**2),
        '2/π²':                     2/np.pi**2,
        '1/π²':                     1/np.pi**2,
        '3/π³':                     3/np.pi**3,
        '1/(4π)':                   1/(4*np.pi),
        '1/(2π)':                   1/(2*np.pi),
        '15/(4π²)':                 15/(4*np.pi**2),
        '3/(8π²)':                  3/(8*np.pi**2),
        '5/(8π²)':                  5/(8*np.pi**2),
        '1/(2π²)':                  1/(2*np.pi**2),
        '1/(8π)':                   1/(8*np.pi),
        '1/12':                     1/12,
        '1/9':                      1/9,
        '1/16':                     1/16,
    }

    for irrep_name, (mean, std) in means.items():
        print(f"\n  Closest candidates to {irrep_name} = {mean:+.4f}:")
        sorted_candidates = sorted(candidates.items(),
                                     key=lambda x: abs(x[1] - mean))
        for cand_name, cand_val in sorted_candidates[:4]:
            ratio = mean / cand_val if cand_val != 0 else float('inf')
            err = abs(mean - cand_val)
            flag = "  ←" if err < 0.05 * abs(mean) else ""
            print(f"    {cand_name:<30s} = {cand_val:+.6f}  (|diff|={err:.4f}, ratio={ratio:+.4f}){flag}")

    header("Interpretation")
    print()
    K_mean, K_std = means['A_1g (bulk K)']
    Eg_mean, Eg_std = means['E_g  (axial)']
    T2g_mean, T2g_std = means['T_2g (cross)']
    Voigt_mean, Voigt_std = means['Voigt μ_iso']
    Cauchy_mean, Cauchy_std = means['Cauchy violation']

    print(f"  Cauchy violation: {Cauchy_mean:+.4f} ± {Cauchy_std:.4f}")
    print(f"  → substrate has STRONG cubic anisotropy (E_g ≠ T_2g).")
    print()

    print(f"  C_E_g (axial)  = {Eg_mean:+.4f} ± {Eg_std:.4f}")
    print(f"  C_T2g (cross)  = {T2g_mean:+.4f} ± {T2g_std:.4f}")
    print(f"  Ratio T_2g/E_g = {T2g_mean/Eg_mean:.2f}")
    print()

    if abs(Eg_mean) < 3*Eg_std:
        print(f"  ⚠ C_E_g ≈ 0 within numerical noise — POSSIBLE Ward-protected vanishing")
        print(f"    in the E_g (axial) channel. If true, graviton TT in E_g would be")
        print(f"    massless. Check: does C_E_g → 0 as N → ∞ more rigorously?")
    else:
        print(f"  C_E_g is finite and nonzero — no Ward protection in this channel.")

    if abs(T2g_mean) < 3*T2g_std:
        print(f"  ⚠ C_T2g ≈ 0 — Ward protection in cross channel.")
    else:
        print(f"  C_T2g is finite and nonzero — no Ward protection in this channel.")


if __name__ == "__main__":
    main()
