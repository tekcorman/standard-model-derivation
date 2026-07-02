#!/usr/bin/env python3
"""
Iorio-elastic Session 2 — REDONE with the correct find_bonds() bond convention.

Context. The original Session 2 (lorentz_sig_iorio_session2_vielbein.py) used
the cell_edges from theorem_B2_signature.py, which give Cartesian displacements
|r_b| = sqrt(50)/4 (far periodic images, NOT nearest-neighbour bonds). For
spectral quantities like the dispersion coefficients, the two conventions are
gauge-equivalent (H_findbonds = H_theorem_B2^T), but the strain perturbation
A^{ac}(k) ~ k_a r_b^c depends on the actual r_b and so requires the NN bond
list from find_bonds().

This script:
1. Builds the Bloch H(k) using find_bonds() (NN_DIST = sqrt(2)/4).
2. Recomputes V_1, projects onto the lambda=-1 subspace at Gamma.
3. Re-extracts the spin-1 generators on the T-irrep and verifies SO(3).
4. Recomputes the strain perturbation A^{ac}(k) and projects.
5. Re-extracts the vielbein prefactor beta.

If the qualitative result (beta = 1) is robust, the new finding is at theorem
grade. If the prefactor differs, we update the framework's understanding.
"""

import os
import sys

import sympy as sp

# We need find_bonds() with exact rational arithmetic. Let's reproduce
# its logic symbolically rather than depending on the numerical version.

ATOMS_S = [
    sp.Matrix([sp.Rational(1, 8), sp.Rational(1, 8), sp.Rational(1, 8)]),
    sp.Matrix([sp.Rational(3, 8), sp.Rational(7, 8), sp.Rational(5, 8)]),
    sp.Matrix([sp.Rational(7, 8), sp.Rational(5, 8), sp.Rational(3, 8)]),
    sp.Matrix([sp.Rational(5, 8), sp.Rational(3, 8), sp.Rational(7, 8)]),
]
A_PRIM_S = [
    sp.Matrix([sp.Rational(-1, 2), sp.Rational(1, 2), sp.Rational(1, 2)]),
    sp.Matrix([sp.Rational(1, 2), sp.Rational(-1, 2), sp.Rational(1, 2)]),
    sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(-1, 2)]),
]

NN_DIST_SQ = sp.Rational(1, 8)   # (sqrt(2)/4)^2 = 2/16 = 1/8


def find_bonds_symbolic():
    """Symbolic version of proofs/common.find_bonds() with exact rationals.

    Returns list of (src, tgt, (n1, n2, n3), r_b) tuples where r_b is the
    exact rational Cartesian displacement.
    """
    bonds = []
    for i in range(4):
        ri = ATOMS_S[i]
        for j in range(4):
            for n1 in range(-2, 3):
                for n2 in range(-2, 3):
                    for n3 in range(-2, 3):
                        rj = (ATOMS_S[j]
                              + n1 * A_PRIM_S[0]
                              + n2 * A_PRIM_S[1]
                              + n3 * A_PRIM_S[2])
                        dr = rj - ri
                        dist_sq = (dr[0]**2 + dr[1]**2 + dr[2]**2)
                        if dist_sq == sp.Integer(0):
                            continue
                        if dist_sq == NN_DIST_SQ:
                            bonds.append((i, j, (n1, n2, n3), dr))
    return bonds


# =============================================================================
# Build Bloch H(k) and verify spectrum
# =============================================================================

k1, k2, k3 = sp.symbols('k1 k2 k3', real=True)
TWO_PI_I = 2 * sp.pi * sp.I


def bloch_H_findbonds(k_vec, bonds):
    H = sp.zeros(4, 4)
    for src, tgt, cell, _r in bonds:
        phase = sp.exp(TWO_PI_I * (cell[0]*k_vec[0] + cell[1]*k_vec[1] + cell[2]*k_vec[2]))
        H[tgt, src] = H[tgt, src] + phase
    return H


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("Iorio-elastic Session 2 (FIXED bond convention)")

    bonds = find_bonds_symbolic()
    print(f"\nFound {len(bonds)} NN bonds (expected 12):")
    for src, tgt, cell, rb in bonds[:4]:
        print(f"  bond {src} -> {tgt}, cell {cell}, r_b = {rb.T.tolist()[0]}, "
              f"|r_b|² = {rb.dot(rb)}")
    print(f"  ... ({len(bonds) - 4} more) all with |r_b|² = 1/8 = NN_DIST².")

    # Verify H(0) is K_4
    H0 = bloch_H_findbonds((0, 0, 0), bonds)
    print(f"\nH(0) = K_4 adjacency? {sp.simplify(H0 - sp.Matrix([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])).is_zero_matrix}")

    # Verify spec H(P)
    P_pt = (sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 4))
    H_P = sp.simplify(bloch_H_findbonds(P_pt, bonds))
    eigs_P = H_P.eigenvals()
    print(f"\nspec H(P) (with find_bonds convention): {dict(eigs_P)}")
    expected_P = {sp.sqrt(3): 2, -sp.sqrt(3): 2}
    print(f"Expected: {{sqrt(3): 2, -sqrt(3): 2}}.  Match: {eigs_P == expected_P}")

    # =========================================================================
    # Build V_1 and project to Γ-cone
    # =========================================================================
    print()
    header("Step 2: V_1 with find_bonds, project to lambda=-1 subspace")

    V1 = sp.zeros(4, 4)
    for src, tgt, cell, _r in bonds:
        coef = TWO_PI_I * (cell[0]*k1 + cell[1]*k2 + cell[2]*k3)
        V1[tgt, src] = V1[tgt, src] + coef

    g1 = sp.Matrix([1, -1, 0, 0]) / sp.sqrt(2)
    g2 = sp.Matrix([1, 1, -2, 0]) / sp.sqrt(6)
    g3 = sp.Matrix([1, 1, 1, -3]) / sp.sqrt(12)
    G = sp.Matrix.hstack(g1, g2, g3)

    M = sp.simplify(G.H * V1 * G)
    print("\n  M = G^† V_1 G (3×3 projection) =")
    sp.pprint(M)

    # Substitute Cartesian k
    kx, ky, kz = sp.symbols('kx ky kz', real=True)
    subs = {
        k1: (-kx + ky + kz) / (4 * sp.pi),
        k2: (kx - ky + kz) / (4 * sp.pi),
        k3: (kx + ky - kz) / (4 * sp.pi),
    }
    M_cart = sp.simplify(M.subs(subs))
    print("\n  M(k_cart) =")
    sp.pprint(M_cart)

    # Extract spin-1 generators: M_cart = (1/2) k_cart · S
    cSx = sp.simplify(sp.diff(M_cart, kx))
    cSy = sp.simplify(sp.diff(M_cart, ky))
    cSz = sp.simplify(sp.diff(M_cart, kz))

    # Pull out (1/2 v_F factor) · i, leaving real anti-symmetric S_a.
    # S_a = -2i · cSa  (assuming v_F = 1/2)
    Sx = sp.simplify(-2 * sp.I * cSx)
    Sy = sp.simplify(-2 * sp.I * cSy)
    Sz = sp.simplify(-2 * sp.I * cSz)

    print("\n  Spin-1 generators (real anti-symmetric form):")
    print("\n  S_x =")
    sp.pprint(Sx)
    print("\n  S_y =")
    sp.pprint(Sy)
    print("\n  S_z =")
    sp.pprint(Sz)

    # SO(3) algebra check
    Cxy = sp.simplify(Sx*Sy - Sy*Sx)
    Cyz = sp.simplify(Sy*Sz - Sz*Sy)
    Czx = sp.simplify(Sz*Sx - Sx*Sz)
    so3_xy = sp.simplify(Cxy - Sz).is_zero_matrix
    so3_yz = sp.simplify(Cyz - Sx).is_zero_matrix
    so3_zx = sp.simplify(Czx - Sy).is_zero_matrix
    print(f"\n  SO(3) algebra:  [S_x, S_y] = S_z: {so3_xy}")
    print(f"                  [S_y, S_z] = S_x: {so3_yz}")
    print(f"                  [S_z, S_x] = S_y: {so3_zx}")

    # If commutators come out with -1 sign, generators are reflected.
    # Check each up to sign:
    if not (so3_xy and so3_yz and so3_zx):
        print("  Trying [S_a, S_b] = -ε_abc S_c (mirrored convention):")
        so3_xy_minus = sp.simplify(Cxy + Sz).is_zero_matrix
        so3_yz_minus = sp.simplify(Cyz + Sx).is_zero_matrix
        so3_zx_minus = sp.simplify(Czx + Sy).is_zero_matrix
        print(f"                  [S_x, S_y] = -S_z: {so3_xy_minus}")
        print(f"                  [S_y, S_z] = -S_x: {so3_yz_minus}")
        print(f"                  [S_z, S_x] = -S_y: {so3_zx_minus}")

    # Casimir check
    S_sq = sp.simplify(Sx*Sx + Sy*Sy + Sz*Sz)
    print(f"\n  Casimir S² = -2·I? {sp.simplify(S_sq - (-2)*sp.eye(3)).is_zero_matrix}")

    # =========================================================================
    # Compute the v_F prefactor
    # =========================================================================
    print()
    header("Step 3: Verify v_F = 1/2 with find_bonds convention")

    # The dispersion magnitude: a²(k) = tr(M²)/2 = ?·|k_cart|².
    M_sq = sp.simplify(M_cart * M_cart)
    tr_M_sq = sp.simplify((M_cart * M_cart).trace())
    a_sq = sp.simplify(tr_M_sq / 2)
    k_cart_sq = sp.expand(kx**2 + ky**2 + kz**2)
    v_F_sq = sp.simplify(sp.expand(a_sq / k_cart_sq))
    v_F_sq_simplified = sp.together(v_F_sq)
    print(f"\n  a² / |k_cart|² = {v_F_sq_simplified}")
    if v_F_sq_simplified.is_rational:
        v_F = sp.sqrt(v_F_sq_simplified)
        print(f"  v_F = sqrt({v_F_sq_simplified}) = {v_F}")
    else:
        print(f"  Not a constant rational; printing in expanded form:")
        sp.pprint(v_F_sq)

    # =========================================================================
    # Compute strain perturbation A^{ac}(k) and project
    # =========================================================================
    print()
    header("Step 4: Strain perturbation A^{ac}(k_cart) with find_bonds, project")

    # T^c[β,α] = i · sum_{bonds (α→β)} r_b^c (k-independent factor)
    T_cart = {}
    for c in range(3):
        Tc = sp.zeros(4, 4)
        for src, tgt, _cell, rb in bonds:
            Tc[tgt, src] = Tc[tgt, src] + sp.I * rb[c]
        T_cart[c] = sp.simplify(Tc)

    # Project T^c to Γ-cone
    M_strain = {}
    for c in range(3):
        M_strain[c] = sp.simplify(G.H * T_cart[c] * G)

    print("\n  G^† T^x G =")
    sp.pprint(M_strain[0])

    # Compare with (1/2) S_a (since unperturbed projects to (1/2) k_cart · S)
    print("\n  Compare to (1/2) S_a  (unperturbed dispersion factor):")
    print("\n  (1/2) S_x =")
    sp.pprint(sp.Rational(1, 2) * Sx)

    # Match: G^† T^c G = (1/2) S_c · (some prefactor), or more general.
    # Specifically, G^† T^c G should be related to S_c by a specific scalar.

    # Check: G^† T^c G = c · S_c for c = constant.
    # Take ratio of matrix elements (e.g., [1, 0]):
    if M_strain[0][1, 0] != 0 and Sx[1, 0] != 0:
        ratio = sp.simplify(M_strain[0][1, 0] / Sx[1, 0])
        print(f"\n  [G^† T^x G][1,0] / [S_x][1,0] = {ratio}")
        if ratio.is_constant():
            print(f"  ⇒ G^† T^x G = ({ratio}) · S_x")
            beta = sp.simplify(2 * ratio)  # because canonical form is (1/2) β · S_a
            print(f"\n  Vielbein prefactor: β = 2 · ratio = {beta}")
        else:
            print(f"  Ratio not a constant; β extraction requires more analysis.")


if __name__ == "__main__":
    main()
