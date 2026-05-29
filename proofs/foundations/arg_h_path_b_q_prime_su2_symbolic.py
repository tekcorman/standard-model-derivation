#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_su2_symbolic.py — symbolic SU(2) Wilson holonomy
around a small C_3-invariant triangle, in the eps → 0 limit.

Background. Numerics (q_prime_su2_convergence.py) cleanly converge to
SU(2) Wilson holonomy θ_∞ ≈ 269.030° (cos(θ/2) ≈ −0.70110). Doesn't
match obvious targets (270° = 3π/2, 2π−2·arg(h) = 255.52°, etc.).

The SU(2) holonomy around a small loop encircling a band-crossing point
depends only on the σ-vector structure of H_eff(δk) in the eps → 0 limit
(scale-invariant). Using the symbolic Λ_H, Λ_AH matrices computed in
q_prime_lambda_matrix.py, this script computes the SU(2) Wilson product
analytically for a C_3-symmetric triangle around the C_3 axis (1,1,1)/√3.

Method. At each triangle vertex, the (non-Hermitian) eigenvector of
H_eff(δk) at the +h-band side is computed symbolically. The Wilson
holonomy is the product of inner products around the 3-vertex loop,
extracted to SU(2) form.

If the SU(2) angle has a clean form, this script will identify it.
"""

from __future__ import annotations

import math
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
from arg_h_path_b_q_prime_lambda_matrix import (
    gram_schmidt,
    pauli_components,
)


PRINT_WIDTH = 78


def main():
    print("=" * PRINT_WIDTH)
    print("Q' SU(2) Wilson holonomy: symbolic eps→0 limit")
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

    # Build orthonormal Q (12×2) at k_P.
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
    Qcols = gram_schmidt(cols)
    Q = sp.Matrix.hstack(*Qcols)

    # Compute M^a = Q† · ∂_a B · Q
    M_per_axis = []
    for a, ka in enumerate((k1, k2, k3)):
        dB = sp.diff(B_sym, ka)
        dBkP = sp.simplify(dB.subs(kP_subs))
        M = sp.simplify(Q.H * dBkP * Q)
        M_per_axis.append(M)

    # -------------------------------------------------------------------------
    # The C_3 axis is (1,1,1)/√3. Define a perpendicular vector and rotate by
    # 120°, 240° to get the triangle. The triangle is in the plane perpendicular
    # to (1,1,1)/√3.
    # -------------------------------------------------------------------------
    print("\nStep 1 — C_3-symmetric triangle at small distance ε from k_P")
    # Pick perp vector. Same as numerical script:
    #   perp = (1,1,1)×(1,0,0) / |·| = (0, 1, -1)/√2
    perp_vec = sp.Matrix([0, 1, -1]) / sp.sqrt(2)
    print(f"  Initial perp direction n̂_0 = (0, 1, -1)/√2")

    # The triangle vertices' positions on the perp plane:
    # The C_3 cyclic shift in the script is v_1 = (v_0[2], v_0[0], v_0[1])
    # = right cyclic shift. Apply twice for v_2.
    def cyclic_shift(v):
        return sp.Matrix([v[2], v[0], v[1]])

    v0 = perp_vec
    v1 = cyclic_shift(v0)
    v2 = cyclic_shift(v1)
    print(f"  v_0 = {v0.T}")
    print(f"  v_1 = (cyclic shift) = {v1.T}")
    print(f"  v_2 = (cyclic shift²) = {v2.T}")

    # Verify v_0 + v_1 + v_2 = 0 (C_3 symmetry: sum of 3 rotated vectors)
    sum_v = sp.simplify(v0 + v1 + v2)
    print(f"  v_0 + v_1 + v_2 = {sum_v.T}  (should be 0)")

    # -------------------------------------------------------------------------
    # H_eff at each direction n̂. (eps → 0 limit, so direction-only.)
    # -------------------------------------------------------------------------
    print("\nStep 2 — H_eff(n̂) at each triangle vertex (2×2 matrix)")

    def H_eff_at(n_hat):
        """Build 2×2 H_eff at unit direction n_hat = (nx, ny, nz)."""
        H = sp.zeros(2, 2)
        for a in range(3):
            H = H + n_hat[a] * M_per_axis[a]
        return H

    H0 = sp.simplify(H_eff_at(v0))
    H1 = sp.simplify(H_eff_at(v1))
    H2 = sp.simplify(H_eff_at(v2))

    # -------------------------------------------------------------------------
    # Eigenvectors at each vertex.
    # H_eff(n̂) is 2×2 with traceless structure.
    # Eigenvalues: ±√(det) wait, for a 2×2 traceless M, eigenvalues are ±√(-det(M)).
    #   M = [[a, b], [c, -a]] → λ² - (a² + bc) = 0 → λ = ±√(a² + bc).
    # Eigenvectors at λ = +√(a²+bc): (b, λ - a) (up to normalization).
    # -------------------------------------------------------------------------
    print("\nStep 3 — Eigenvectors of H_eff(n̂) at each vertex")

    def plus_eigenvector(M):
        """Right eigenvector of 2×2 traceless M at the eigenvalue with positive
        Re part."""
        a = M[0, 0]
        b = M[0, 1]
        c = M[1, 0]
        # λ² = a² + bc
        lam2 = sp.simplify(a**2 + b*c)
        lam = sp.sqrt(lam2)
        # Pick sign so Re(λ) ≥ 0 (or some consistent rule).
        # Eigenvector (b, λ - a) for eigenvalue λ.
        v = sp.Matrix([b, lam - a])
        # Normalize: ⟨v|v⟩ = b·b̄ + (λ-a)·(λ̄-ā). Actually for non-Hermitian
        # M, the right eigenvector + bi-orthogonal left make a complete pair.
        # For Wilson loop, we typically use right eigenvectors normalized in
        # standard L² inner product.
        nrm = sp.sqrt(sp.simplify(v.H * v)[0, 0])
        v_normalized = sp.simplify(v / nrm) if nrm != 0 else v
        return v_normalized, lam

    print("  Computing eigenvectors at v_0, v_1, v_2 ...")
    psi_0, lam_0 = plus_eigenvector(H0)
    psi_1, lam_1 = plus_eigenvector(H1)
    psi_2, lam_2 = plus_eigenvector(H2)
    print(f"  λ at v_0: {sp.simplify(lam_0)}")
    print(f"  λ at v_1: {sp.simplify(lam_1)}")
    print(f"  λ at v_2: {sp.simplify(lam_2)}")
    print(f"  (should all have same magnitude, by C_3 symmetry)")

    print(f"\n  ψ_0 = {sp.simplify(psi_0).T}")
    print(f"  ψ_1 = {sp.simplify(psi_1).T}")
    print(f"  ψ_2 = {sp.simplify(psi_2).T}")

    # -------------------------------------------------------------------------
    # Wilson loop product = ⟨ψ_0|ψ_1⟩⟨ψ_1|ψ_2⟩⟨ψ_2|ψ_0⟩ (rank-2 reduction).
    # But this is the U(1) holonomy on a single band. For full SU(2) on the
    # rank-2 band, we need a 12×2 basis at each vertex and the SU(2) Wilson
    # holonomy is product of 2×2 overlap matrices.
    #
    # This script first checks the SINGLE-BAND eigenvector route (which gives
    # one specific eigenvalue branch, not the full SU(2) story).
    # -------------------------------------------------------------------------
    print("\nStep 4 — U(1) (single-band) Wilson product")
    overlap_01 = sp.simplify((psi_0.H * psi_1)[0, 0])
    overlap_12 = sp.simplify((psi_1.H * psi_2)[0, 0])
    overlap_20 = sp.simplify((psi_2.H * psi_0)[0, 0])
    W_U1 = sp.simplify(overlap_01 * overlap_12 * overlap_20)
    print(f"  ⟨ψ_0|ψ_1⟩ = {overlap_01}")
    print(f"  ⟨ψ_1|ψ_2⟩ = {overlap_12}")
    print(f"  ⟨ψ_2|ψ_0⟩ = {overlap_20}")
    print(f"  W_U1 = ⟨ψ_0|ψ_1⟩⟨ψ_1|ψ_2⟩⟨ψ_2|ψ_0⟩ = {W_U1}")
    arg_W = sp.simplify(sp.arg(W_U1))
    print(f"  arg(W_U1) = {arg_W}")
    try:
        arg_deg = float(arg_W) * 180 / math.pi
        print(f"  arg(W_U1) in degrees = {arg_deg:.6f}°")
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Step 5: Full SU(2) Wilson holonomy.
    # At each vertex, find the FULL 2-dim +h band (rank 2).
    # H_eff(n̂) has eigenvalues ±λ(n̂); both give a 1-dim eigenspace each.
    # The full +h band's +λ subspace is 1-dim (within the 2-dim rank-2 band).
    # That makes the U(1) Wilson loop above the right object.
    #
    # Wait — the rank-2 band contracts to 1-dim per +/- eigenvalue under the
    # H_eff splitting. So the SU(2) Wilson is over the FULL rank-2 (which
    # encompasses both ±λ eigenstates). Compute that:
    # W_SU2 = product of 2×2 matrices [⟨bands_a | bands_b⟩] over the loop.
    # -------------------------------------------------------------------------
    print("\nStep 5 — Full SU(2) Wilson holonomy (rank-2 band, both ±λ eigenstates)")

    def both_eigenvectors(M):
        """Right eigenvectors of 2×2 traceless M at ±λ. Returns 2×2 matrix
        with columns = eigenvectors."""
        a = M[0, 0]
        b = M[0, 1]
        c = M[1, 0]
        lam2 = sp.simplify(a**2 + b*c)
        lam = sp.sqrt(lam2)
        # Eigenvectors (b, λ - a) and (b, -λ - a)
        v_plus = sp.Matrix([b, lam - a])
        v_minus = sp.Matrix([b, -lam - a])
        # Normalize each
        nrm_plus = sp.sqrt(sp.simplify((v_plus.H * v_plus)[0, 0]))
        nrm_minus = sp.sqrt(sp.simplify((v_minus.H * v_minus)[0, 0]))
        if nrm_plus == 0 or nrm_minus == 0:
            return None
        v_plus = v_plus / nrm_plus
        v_minus = v_minus / nrm_minus
        return sp.Matrix.hstack(v_plus, v_minus)

    Q0 = both_eigenvectors(H0)
    Q1 = both_eigenvectors(H1)
    Q2 = both_eigenvectors(H2)
    if Q0 is None or Q1 is None or Q2 is None:
        print("  WARN: degenerate eigenvector at one of vertices")
    else:
        # SU(2) Wilson holonomy = Q_0† · Q_1 · Q_1† · Q_2 · Q_2† · Q_0
        # Equivalently: O_01 = Q_0† · Q_1, etc., and W = O_20 · O_12 · O_01.
        O01 = sp.simplify(Q0.H * Q1)
        O12 = sp.simplify(Q1.H * Q2)
        O20 = sp.simplify(Q2.H * Q0)
        W_SU2 = sp.simplify(O20 * O12 * O01)
        print(f"  O_01 (2×2) =\n{O01}")
        print(f"  O_12 (2×2) =\n{O12}")
        print(f"  O_20 (2×2) =\n{O20}")
        print(f"\n  Wilson holonomy W (2×2) =\n{W_SU2}")

        # det(W) and tr(W) — extract SU(2) angle
        det_W = sp.simplify(sp.det(W_SU2))
        tr_W = sp.simplify(W_SU2.trace())
        print(f"\n  det(W) = {det_W}")
        try:
            print(f"    |det(W)| ≈ {abs(complex(det_W)):.6f}  (should be 1 for unitary)")
        except Exception:
            pass
        print(f"  tr(W) = {tr_W}")
        try:
            print(f"    tr(W) ≈ {complex(tr_W):.6f}")
        except Exception:
            pass
        # SU(2) angle from tr(W/√det(W))
        sqrt_det_W = sp.sqrt(det_W)
        W_SU2_normalized = W_SU2 / sqrt_det_W
        tr_SU2 = sp.simplify(W_SU2_normalized.trace())
        print(f"\n  tr(W/√det(W)) = 2 cos(θ/2) = {tr_SU2}")
        try:
            cos_half = complex(tr_SU2) / 2
            print(f"    ≈ {cos_half:.6f}")
            theta_half = math.acos(cos_half.real)
            theta_deg = 2 * math.degrees(theta_half)
            print(f"    ⇒ θ ≈ {theta_deg:.6f}°")
            print(f"    ⇒ θ via arccos branch (forced [0,2π]):")
            print(f"      candidate 269.030° (numerical) — match?")
        except Exception as e:
            print(f"    cannot evaluate: {e}")

    print(f"\n" + "=" * PRINT_WIDTH)
    print("OK")


if __name__ == "__main__":
    main()
