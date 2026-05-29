#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_sigma_norms.py — exact symbolic σ-vector norms.

Goal. Numerical Q' perturbation gives:
  |σ_Hermitian|²    ≈ 0.977   (close to 1 = ?)
  |σ_anti-Herm|²    ≈ 2.870   (close to k* = 3 = ?)
  Berry monopole charge ≈ 1.2570  (close to 5/4 = Im(h)²)

These ~3% gaps are likely finite-difference (eps = 1e-7) numerical error.
This script computes the exact symbolic values via traces on the 12×12
Bloch operator, bypassing the 2×2 eigenvector reduction.

Method. Let Q be the 12×2 orthonormal basis of the +h band at k_P, so
  P_+h = Q · Q†   (12×12 projector, rank 2).

The 2×2 reduction is M^a = Q† · ∂_a B · Q. Decompose into Hermitian +
anti-Hermitian Pauli parts:
  M^a_H  := (M^a + (M^a)†)/2 = Q† X^a_H  Q     where X^a_H  = (∂_a B + ∂_a B†)/2
  M^a_AH := (M^a − (M^a)†)/(2i) = Q† X^a_AH Q  where X^a_AH = (∂_a B − ∂_a B†)/(2i)

These are 2×2 traceless Hermitian (one per axis). Decomposing M = h_x σ_x +
h_y σ_y + h_z σ_z:
  |σ|² = h_x² + h_y² + h_z² = (1/2) Tr_2(M²).

Using P_+h = Q Q†:
  Tr_2((M^a_H)²) = Tr_12(Q Q† X^a_H Q Q† X^a_H) = Tr_12(P_+h · X^a_H · P_+h · X^a_H)

This is a trace on 12×12 matrices that are symbolically known.

Closed-form projector. Distinct eigenvalues of B(k_P) are
  {h, −h, h̄, −h̄, +1, −1}
so by Lagrange interpolation,
  P_+h = (B+hI)(B−h̄I)(B+h̄I)(B−I)(B+I) / [2h · (h−h̄)(h+h̄) · (h−1)(h+1)]
       = (B+hI)(B²−h̄²I)(B²−I) / [2h · (h²−h̄²) · (h²−1)]
       = (B+hI)(B²−h̄²I)(B²−I) / (−12 i √5)

(denominator simplification: 2h = √3 + i√5; h²−h̄² = i√15; h²−1 =
(−3+i√15)/2; product = −12 i √5.)

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_q_prime_sigma_norms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proofs" / "cosmology"))

from arg_h_path_b_q_prime_symbolic_setup import (
    extract_bond_table,
    build_B_symbolic,
)


PRINT_WIDTH = 78


def main():
    print("=" * PRINT_WIDTH)
    print("Q' σ-vector norms (exact symbolic)")
    print("=" * PRINT_WIDTH)

    bonds = extract_bond_table()
    k1, k2, k3 = sp.symbols("k1 k2 k3", real=True)
    B_sym = build_B_symbolic(bonds, (k1, k2, k3))

    kP_subs = {k1: sp.Rational(1, 4),
               k2: sp.Rational(1, 4),
               k3: sp.Rational(1, 4)}

    print("\nStep 1 — B(k_P) symbolic")
    B_kP = sp.simplify(B_sym.subs(kP_subs))

    h = (sp.sqrt(3) + sp.I * sp.sqrt(5)) / 2
    hbar = (sp.sqrt(3) - sp.I * sp.sqrt(5)) / 2

    # Verify denominator simplification.
    den_sym = 2*h * (h**2 - hbar**2) * (h**2 - 1)
    den_simp = sp.simplify(den_sym)
    print(f"  Denominator 2h·(h²−h̄²)·(h²−1) = {den_simp}  (expect −12 i √5)")

    print("\nStep 2 — Build spectral projector P_spec via Lagrange interpolation")
    print("  P_spec = (B+hI)(B²−h̄²I)(B²−I) / (−12 i √5)")

    I12 = sp.eye(12)
    Bsq = B_kP * B_kP
    factor1 = B_kP + h * I12
    factor2 = Bsq - hbar**2 * I12
    factor3 = Bsq - I12
    Pnum = factor1 * factor2 * factor3
    P_spec = Pnum / den_simp
    P_spec = sp.simplify(P_spec)

    # Spectral projector P² = P but is NOT Hermitian (B is non-normal).
    print("  Checking P_spec² = P_spec ...")
    Psq = sp.simplify(P_spec * P_spec)
    diff = sp.simplify(Psq - P_spec)
    diff_norm = max([abs(diff[i, j]) for i in range(12) for j in range(12)])
    print(f"    max |P²−P| entry = {diff_norm}")

    print("  Checking trace(P_spec) = 2 (dim of +h band) ...")
    tr_P = sp.simplify(sum(P_spec[i, i] for i in range(12)))
    print(f"    tr(P_spec) = {tr_P}  (expect 2)")

    # -------------------------------------------------------------------------
    # Build the ORTHOGONAL projector P_+h onto col(P_spec).
    # Since rank(P_spec) = 2, take any 2 linearly-independent columns of P_spec
    # to form A (12×2), then P_+h = A · (A†A)^(-1) · A†.
    # -------------------------------------------------------------------------
    print("\nStep 2b — Build ORTHOGONAL projector onto +h eigenspace")
    cols = []
    chosen_idx = []
    for j in range(12):
        col = P_spec[:, j]
        if sp.simplify(col.norm()) == 0:
            continue
        if not cols:
            cols.append(col); chosen_idx.append(j); continue
        # Test linear independence with first column.
        # Build [c0 | col] (12×2) and check it has rank 2.
        test = sp.Matrix.hstack(cols[0], col)
        if test.rank() == 2:
            cols.append(col); chosen_idx.append(j); break
    print(f"  Picked columns {chosen_idx} of P_spec as A (rank 2 ✓)")
    A = sp.Matrix.hstack(*cols)        # 12×2
    A_dag = A.H
    AdA = sp.simplify(A_dag * A)       # 2×2 Gram matrix
    AdA_inv = sp.simplify(AdA.inv())
    P_plus_h = sp.simplify(A * AdA_inv * A_dag)

    # Verify P_+h is Hermitian and idempotent.
    print("  Checking P_+h = P_+h† ...")
    diff_herm = sp.simplify(P_plus_h - P_plus_h.H)
    diff_herm_max = max([abs(diff_herm[i, j]) for i in range(12) for j in range(12)])
    print(f"    max |P − P†| entry = {diff_herm_max}")

    print("  Checking P_+h² = P_+h ...")
    Psq = sp.simplify(P_plus_h * P_plus_h)
    diff = sp.simplify(Psq - P_plus_h)
    diff_norm = max([abs(diff[i, j]) for i in range(12) for j in range(12)])
    print(f"    max |P²−P| entry = {diff_norm}")

    print("  Checking trace(P_+h) = 2 ...")
    tr_P = sp.simplify(sum(P_plus_h[i, i] for i in range(12)))
    print(f"    tr(P_+h) = {tr_P}  (expect 2)")

    print("\nStep 3 — ∂_a B at k_P (per axis)")
    dB_da = []
    for a, ka in enumerate((k1, k2, k3)):
        dB = sp.diff(B_sym, ka)
        dBkP = sp.simplify(dB.subs(kP_subs))
        dB_da.append(dBkP)

    print("\nStep 4 — Compute σ-norms via trace formulas")
    print("  For each axis a:")
    print("    |σ_H^a |² = (1/2) Tr(P_+h · X^a_H  · P_+h · X^a_H )")
    print("    |σ_AH^a|² = (1/2) Tr(P_+h · X^a_AH · P_+h · X^a_AH)")

    sigma_H_norms = []
    sigma_AH_norms = []

    for a in range(3):
        dB_a = dB_da[a]
        dB_a_dag = dB_a.H  # Hermitian conjugate (transpose conjugate)
        X_H = (dB_a + dB_a_dag) / 2
        X_AH = (dB_a - dB_a_dag) / (2 * sp.I)
        # Hermitian part squared norm via trace
        M_H = P_plus_h * X_H * P_plus_h * X_H
        tr_H = sp.simplify(sum(M_H[i, i] for i in range(12)))
        sigma_sq_H = sp.simplify(tr_H / 2)
        # Anti-Hermitian part
        M_AH = P_plus_h * X_AH * P_plus_h * X_AH
        tr_AH = sp.simplify(sum(M_AH[i, i] for i in range(12)))
        sigma_sq_AH = sp.simplify(tr_AH / 2)

        sigma_H_norms.append(sigma_sq_H)
        sigma_AH_norms.append(sigma_sq_AH)

        print(f"\n  axis {a + 1}:")
        print(f"    |σ_H|²  = {sigma_sq_H}")
        try:
            print(f"    |σ_H|² (numerical)  = {complex(sigma_sq_H):.6f}")
        except Exception:
            print(f"    |σ_H|² (cannot float)")
        print(f"    |σ_AH|² = {sigma_sq_AH}")
        try:
            print(f"    |σ_AH|² (numerical) = {complex(sigma_sq_AH):.6f}")
        except Exception:
            print(f"    |σ_AH|² (cannot float)")

    print("\nStep 5 — Sum-over-axes (= |H_eff|² Frobenius² of the rank-2 projection)")
    total_H = sp.simplify(sum(sigma_H_norms))
    total_AH = sp.simplify(sum(sigma_AH_norms))
    def fmt(x):
        try:
            return f"{complex(x):.6f}"
        except Exception:
            return "(non-numeric)"
    print(f"  Σ_a |σ_H^a |² = {total_H}  ({fmt(total_H)})")
    print(f"  Σ_a |σ_AH^a|² = {total_AH}  ({fmt(total_AH)})")
    grand = sp.simplify(total_H + total_AH)
    print(f"  total |σ|²    = {grand}  ({fmt(grand)})")

    print("\nStep 6 — Frobenius norm² of M^a (full 2×2 H_eff^a)")
    print("  |M^a|² = Tr(P_+h · ∂_a B · P_+h · ∂_a B†)")
    for a in range(3):
        dB_a = dB_da[a]
        M_full = P_plus_h * dB_a * P_plus_h * dB_a.H
        tr_M = sp.simplify(sum(M_full[i, i] for i in range(12)))
        print(f"  axis {a+1}: |M^a|² = {tr_M}  ({fmt(tr_M)})")

    print(f"\n" + "=" * PRINT_WIDTH)
    print("Identification check — clean-rational candidates:")
    print(f"  k* = 3, Im(h)² = 5/4, Re(h)² = 3/4, |h|² = 2")
    print(f"  Higgs c = 5/12, ε_CP = 1/5, Class A constants...")
    print(f"  numerical |σ_H|² = 0.977, |σ_AH|² = 2.870 from FD probe")
    print("=" * PRINT_WIDTH)


if __name__ == "__main__":
    main()
