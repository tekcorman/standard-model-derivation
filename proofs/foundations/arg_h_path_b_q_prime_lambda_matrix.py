#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_lambda_matrix.py — extract the σ-vector 3×3 matrix
Λ symbolically, then compute its determinant.

Background. From q_prime_sigma_norms.py we know:
  |σ_H|²/axis  = 7π²(3−√5)/54 = 7π²/(27φ²)
  |σ_AH|²/axis = π²(3+√5)/18  = π²φ²/9

These are sums of squares of the per-axis σ-vector components. To get the
Berry monopole charge for the rank-2 band crossing, need the full 3×3
matrix:
  Λ_H[i, a]  = ⟨σ_H^a⟩_i = i-th Pauli component of Hermitian part of M^a
  Λ_AH[i, a] = ⟨σ_AH^a⟩_i

The map δk → σ̂_H(δk) sends S² to S² with degree sign(det Λ_H). For a
single Weyl monopole, |det| ∝ R³ where R is the σ-sphere radius; the
sign tells the chirality.

Method. Need to find an orthonormal 12×2 basis Q for the +h band. Build
Q from sympy by Gram-Schmidt-orthonormalizing two columns of the
spectral projector (basis-dependent up to U(2) but Λ_ij rotates with
basis — det(Λ) is basis-INVARIANT modulo det(U)·det(U)† = 1).

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_q_prime_lambda_matrix.py
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


def gram_schmidt(cols):
    """Gram-Schmidt orthonormalize a list of sympy column vectors."""
    out = []
    for v in cols:
        u = v
        for w in out:
            u = u - (w.H * v)[0, 0] * w
        u = sp.simplify(u)
        nrm = sp.sqrt(sp.simplify((u.H * u)[0, 0]))
        if nrm == 0:
            continue
        u = u / nrm
        out.append(sp.simplify(u))
    return out


def pauli_components(M):
    """Decompose 2×2 Hermitian M = h_0 I + h_x σ_x + h_y σ_y + h_z σ_z.

    Returns (h_0, h_x, h_y, h_z) as sympy expressions (real for true Hermitian M).
    """
    h_0 = (M[0, 0] + M[1, 1]) / 2
    h_z = (M[0, 0] - M[1, 1]) / 2
    h_x = (M[0, 1] + M[1, 0]) / 2
    h_y = (M[1, 0] - M[0, 1]) / (2 * sp.I)
    return h_0, h_x, h_y, h_z


def main():
    print("=" * PRINT_WIDTH)
    print("Q' Λ-matrix: σ-vectors per axis, symbolic")
    print("=" * PRINT_WIDTH)

    bonds = extract_bond_table()
    k1, k2, k3 = sp.symbols("k1 k2 k3", real=True)
    B_sym = build_B_symbolic(bonds, (k1, k2, k3))
    kP_subs = {k1: sp.Rational(1, 4),
               k2: sp.Rational(1, 4),
               k3: sp.Rational(1, 4)}
    B_kP = sp.simplify(B_sym.subs(kP_subs))

    h = (sp.sqrt(3) + sp.I * sp.sqrt(5)) / 2
    hbar = (sp.sqrt(3) - sp.I * sp.sqrt(5)) / 2

    I12 = sp.eye(12)
    Bsq = B_kP * B_kP
    den = -12 * sp.I * sp.sqrt(5)
    P_spec = (B_kP + h * I12) * (Bsq - hbar**2 * I12) * (Bsq - I12) / den
    P_spec = sp.simplify(P_spec)

    print("\nStep 1 — Find orthonormal 12×2 basis Q via Gram-Schmidt on P_spec columns.")
    # Pick first two LI columns of P_spec.
    cols = []
    for j in range(12):
        c = P_spec[:, j]
        if sp.simplify((c.H * c)[0, 0]) == 0:
            continue
        if not cols:
            cols.append(c)
            continue
        test = sp.Matrix.hstack(cols[0], c)
        if test.rank() == 2:
            cols.append(c)
            break
    print(f"  Picked 2 columns of P_spec for Gram-Schmidt input")
    Qcols = gram_schmidt(cols)
    print(f"  Got {len(Qcols)} orthonormal vectors")
    Q = sp.Matrix.hstack(*Qcols)   # 12×2
    print(f"  Verifying Q† Q = I_2 ...")
    QQ = sp.simplify(Q.H * Q)
    print(f"    Q† Q =\n{QQ}")
    print(f"  Verifying B Q = h Q (eigenvector property) ...")
    eigval_check = sp.simplify(B_kP * Q - h * Q)
    eigval_max = max([abs(eigval_check[i, j]) for i in range(12) for j in range(2)])
    print(f"    max |B Q − h Q| = {eigval_max}")

    # -------------------------------------------------------------------------
    # Step 2: Compute M^a = Q† · ∂_a B · Q (2×2 matrices, one per axis)
    # -------------------------------------------------------------------------
    print("\nStep 2 — M^a = Q† · ∂_a B · Q per axis")
    M_per_axis = []
    for a, ka in enumerate((k1, k2, k3)):
        dB = sp.diff(B_sym, ka)
        dBkP = sp.simplify(dB.subs(kP_subs))
        M = sp.simplify(Q.H * dBkP * Q)
        M_per_axis.append(M)
        print(f"\n  axis {a + 1} (M^{a+1}):")
        for i in range(2):
            row = "    "
            for j in range(2):
                row += f"  {sp.simplify(M[i, j])}"
            print(row)

    # -------------------------------------------------------------------------
    # Step 3: Decompose M^a into Hermitian + anti-Hermitian Pauli components.
    # -------------------------------------------------------------------------
    print("\nStep 3 — Pauli decomposition of M^a (Hermitian + anti-Hermitian parts)")
    sigma_H_per_axis = []
    sigma_AH_per_axis = []
    for a in range(3):
        M = M_per_axis[a]
        M_H = sp.simplify((M + M.H) / 2)
        M_AH = sp.simplify((M - M.H) / (2 * sp.I))
        h_0_H, h_x_H, h_y_H, h_z_H = pauli_components(M_H)
        h_0_AH, h_x_AH, h_y_AH, h_z_AH = pauli_components(M_AH)
        sigma_H_per_axis.append((h_x_H, h_y_H, h_z_H))
        sigma_AH_per_axis.append((h_x_AH, h_y_AH, h_z_AH))
        print(f"\n  axis {a + 1} Hermitian σ:")
        print(f"    h_0 = {sp.simplify(h_0_H)}")
        print(f"    h_x = {sp.simplify(h_x_H)}")
        print(f"    h_y = {sp.simplify(h_y_H)}")
        print(f"    h_z = {sp.simplify(h_z_H)}")
        print(f"\n  axis {a + 1} anti-Hermitian σ:")
        print(f"    h_0 = {sp.simplify(h_0_AH)}")
        print(f"    h_x = {sp.simplify(h_x_AH)}")
        print(f"    h_y = {sp.simplify(h_y_AH)}")
        print(f"    h_z = {sp.simplify(h_z_AH)}")

    # -------------------------------------------------------------------------
    # Step 4: Build 3×3 matrices Λ_H[i, a] and Λ_AH[i, a], compute determinants.
    # -------------------------------------------------------------------------
    print("\nStep 4 — 3×3 matrices Λ_H and Λ_AH; determinants for Berry charge")
    Lambda_H = sp.Matrix(3, 3, lambda i, a: sigma_H_per_axis[a][i])
    Lambda_AH = sp.Matrix(3, 3, lambda i, a: sigma_AH_per_axis[a][i])
    Lambda_H = sp.simplify(Lambda_H)
    Lambda_AH = sp.simplify(Lambda_AH)
    print(f"\n  Λ_H =\n{Lambda_H}")
    det_H = sp.simplify(Lambda_H.det())
    print(f"\n  det(Λ_H) = {det_H}")
    print(f"           ≈ {complex(det_H):.6f}")
    print(f"\n  Λ_AH =\n{Lambda_AH}")
    det_AH = sp.simplify(Lambda_AH.det())
    print(f"\n  det(Λ_AH) = {det_AH}")
    print(f"           ≈ {complex(det_AH):.6f}")

    # -------------------------------------------------------------------------
    # Step 5: Berry monopole charge for the σ_H map (linearized).
    # For a linear map δk → σ̂_H, the Berry monopole charge of the rank-2
    # band crossing equals sign(det Λ_H) for a single Weyl. But actual
    # numerical gave 1.2570 ≈ 5/4 — perhaps a non-minimal monopole, or
    # the rank-2 contribution comes from both H + AH parts.
    # -------------------------------------------------------------------------
    print("\nStep 5 — Berry-charge identification candidates")
    print(f"  numerical Q' winding ≈ 1.2570 (close to 5/4 = Im(h)²)")
    print(f"  det(Λ_H) / clean rationals: comparing to 5/4, π³, etc.")
    print(f"    det(Λ_H) / (5/4) = {sp.simplify(det_H / sp.Rational(5, 4))}")
    print(f"    det(Λ_H) / π³ = {sp.simplify(det_H / sp.pi**3)}")
    print(f"    |det(Λ_H)|² = {sp.simplify(det_H * sp.conjugate(det_H))}")

    # -------------------------------------------------------------------------
    # Step 6: Cross-products for Berry curvature 2-form structure.
    # -------------------------------------------------------------------------
    print("\nStep 6 — σ × σ structure (Berry curvature components)")
    sH = [sp.Matrix(list(s)) for s in sigma_H_per_axis]
    sAH = [sp.Matrix(list(s)) for s in sigma_AH_per_axis]
    print(f"  σ_H^1 × σ_H^2 = {sp.simplify(sH[0].cross(sH[1])).T}")
    print(f"  σ_H^2 × σ_H^3 = {sp.simplify(sH[1].cross(sH[2])).T}")
    print(f"  σ_H^3 × σ_H^1 = {sp.simplify(sH[2].cross(sH[0])).T}")

    # Triple product (volume form)
    triple = sp.simplify(sH[0].dot(sH[1].cross(sH[2])))
    print(f"\n  σ_H^1 · (σ_H^2 × σ_H^3) = {triple}  (= det(Λ_H))")
    print(f"  numerical = {complex(triple):.6f}")

    print(f"\n" + "=" * PRINT_WIDTH)
    print("OK: lambda_matrix completed")


if __name__ == "__main__":
    main()
