#!/usr/bin/env python3
"""
G_sub session 5 path #5: rigorous P-cone strain vertex derivation.

Per `g_sub_session5_path4_finding.md`, the previous P-cone ζ used naive Iorio
strain vertex V^{ab}(q) = v_F q^b σ^a. This may differ from the actual
projection of the substrate's full Bloch strain perturbation A^{ac}(k)
onto the 2-dim eigensubspace at the P-point.

This script derives the rigorous V^{ac,d}_P symbolically and outputs the
corrected ζ_P.

Method
------
1. Compute strain perturbation matrix A^{ac}(k) = i Σ_bonds exp(i k r_b) k_a r_b^c
   in the substrate's full 4×4 Bloch space.
2. At the P-point P_frac = (1/4, 1/4, 1/4), find the 2-dim eigensubspace at +√3
   (and -√3 separately for full multi-valley sum).
3. Compute the LINEAR-IN-δk expansion of A^{ac}(P+δk) projected onto P-subspace:
     V^{ac,d}_P := ⟨U_P| ∂A^{ac}/∂δk_d |_{δk=0} |U_P⟩
   (a 3×3 tensor in the substrate strain (a,c)+momentum (d) indices, acting
   on the 2-dim P-subspace.)
4. Decompose V^{ac,d}_P in terms of Pauli matrices and check tensor structure.
5. Recompute ζ_P with the corrected vertex.

Status
------
First-principles symbolic derivation. Tests whether session 4's universal-ζ
(or path-#4's naive Iorio extension) is correct for the P-cone.
"""
from __future__ import annotations

import sympy as sp
import numpy as np


# Atom positions and bond list (matching srs_dirac_cone_velocities.py)
ATOMS = [
    sp.Matrix([sp.Rational(1, 8), sp.Rational(1, 8), sp.Rational(1, 8)]),
    sp.Matrix([sp.Rational(3, 8), sp.Rational(7, 8), sp.Rational(5, 8)]),
    sp.Matrix([sp.Rational(7, 8), sp.Rational(5, 8), sp.Rational(3, 8)]),
    sp.Matrix([sp.Rational(5, 8), sp.Rational(3, 8), sp.Rational(7, 8)]),
]

A_PRIM = [
    sp.Matrix([sp.Rational(-1, 2), sp.Rational(1, 2), sp.Rational(1, 2)]),
    sp.Matrix([sp.Rational(1, 2), sp.Rational(-1, 2), sp.Rational(1, 2)]),
    sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(-1, 2)]),
]

CELL_EDGES = (
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
)

BONDS = []
for src, tgt, cell in CELL_EDGES:
    BONDS.append((src, tgt, cell))
    BONDS.append((tgt, src, tuple(-c for c in cell)))


def bond_displacement(src, tgt, cell):
    rb = ATOMS[tgt] - ATOMS[src]
    for i in range(3):
        rb = rb + cell[i] * A_PRIM[i]
    return rb


# Cartesian k coordinates: kx, ky, kz
kx, ky, kz = sp.symbols('kx ky kz', real=True)


def H_at_k_cart(kx_v, ky_v, kz_v):
    """4×4 symbolic Bloch Hamiltonian at Cartesian k = (kx, ky, kz)."""
    H = sp.zeros(4, 4)
    for src, tgt, cell in BONDS:
        rb = bond_displacement(src, tgt, cell)
        phase = sp.exp(sp.I * (kx_v * rb[0] + ky_v * rb[1] + kz_v * rb[2]))
        H[tgt, src] += phase
    return H


def A_strain_at_k_cart(kx_v, ky_v, kz_v, a, c):
    """4×4 strain perturbation matrix A^{ac}(k) at given Cartesian k.

    A^{ac}_{βα}(k) = i Σ_bonds(α→β,n) exp(i k · r_b) k_a r_b^c
    """
    A_mat = sp.zeros(4, 4)
    k_cart = [kx_v, ky_v, kz_v]
    for src, tgt, cell in BONDS:
        rb = bond_displacement(src, tgt, cell)
        phase = sp.exp(sp.I * (kx_v * rb[0] + ky_v * rb[1] + kz_v * rb[2]))
        A_mat[tgt, src] += sp.I * phase * k_cart[a] * rb[c]
    return A_mat


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("G_sub session 5 path #5: rigorous P-cone strain vertex")
    print()

    # P-point at fractional (1/4, 1/4, 1/4). Cartesian via b_recip:
    # b_1 = 2π(0,1,1), b_2 = 2π(1,0,1), b_3 = 2π(1,1,0)
    # P_cart = (1/4) × (b_1 + b_2 + b_3) = (2π/4)(2, 2, 2) = π(1, 1, 1)
    P_cart_x = sp.pi
    P_cart_y = sp.pi
    P_cart_z = sp.pi
    print(f"  P_cart = (π, π, π)")

    # Verify spectrum at P
    H_P = H_at_k_cart(P_cart_x, P_cart_y, P_cart_z)
    H_P = sp.simplify(H_P)
    print()
    print(f"  Spectrum at P (sympy):")
    eigs = H_P.eigenvals()
    for ev, mult in eigs.items():
        print(f"    λ = {sp.simplify(ev)}, multiplicity {mult}")

    # Find 2-dim eigensubspace at +√3
    target_ev = sp.sqrt(3)
    eigvecs = None
    for ev, mult, vecs in H_P.eigenvects():
        if sp.simplify(ev - target_ev) == 0:
            eigvecs = vecs
            break
    assert eigvecs is not None and len(eigvecs) == 2
    print(f"\n  Found 2-dim eigensubspace at λ = +√3")

    # Gram-Schmidt orthonormalisation to U_P (4×2)
    u1 = eigvecs[0]
    u1 = u1 / sp.sqrt(sp.simplify((u1.H * u1)[0]))
    u2 = eigvecs[1]
    u2 = u2 - (u1.H * u2)[0] * u1
    u2 = u2 / sp.sqrt(sp.simplify((u2.H * u2)[0]))
    U_P = sp.Matrix.hstack(sp.simplify(u1), sp.simplify(u2))
    print(f"  U_P (4×2) constructed.")

    # Strain perturbation at P (all 9 (a,c) pairs)
    print()
    print(f"  Computing 9 strain matrices A^{{ac}}(P), and their projections to 2-dim subspace.")
    A_proj_at_P = {}  # (a,c) → 2×2 matrix
    for a in range(3):
        for c in range(3):
            A_full = A_strain_at_k_cart(P_cart_x, P_cart_y, P_cart_z, a, c)
            A_full = sp.simplify(A_full)
            A_proj = sp.simplify(U_P.H * A_full * U_P)
            A_proj_at_P[(a, c)] = A_proj

    print()
    print("  A^{ac}(P) projected onto 2-dim P-subspace at +√3 (zero-th order in δk):")
    for (a, c), M in A_proj_at_P.items():
        coord = ['x', 'y', 'z']
        print(f"    A^{{{coord[a]}{coord[c]}}}(P)|_{{2×2}} =")
        sp.pprint(M)
        print()

    # Now compute LINEAR-IN-δk expansion: ∂A^{ac}/∂δk_d at δk=0, projected.
    # A^{ac}(P + δk)_full = i Σ_bonds exp(i (P + δk) · r_b) (P + δk)_a r_b^c.
    # Differentiating w.r.t. δk_d at δk=0:
    # ∂A^{ac}/∂δk_d|_{δk=0} = i Σ_bonds exp(i P · r_b) [δ_{ad} r_b^c + i r_b^d P_a r_b^c]
    print()
    print("  Computing ∂A^{ac}/∂δk_d|_{δk=0} (linear-in-δk vertex), projected:")
    dA_proj = {}  # (a,c,d) → 2×2 matrix
    for a in range(3):
        for c in range(3):
            for d in range(3):
                # Differentiate symbolic A^{ac} w.r.t. δk_d
                # δk_d adds to k_d: in our function, k = (kx, ky, kz), so δk_d adds to coord d.
                # Use sympy diff:
                A_sym = A_strain_at_k_cart(kx, ky, kz, a, c)
                k_sym = [kx, ky, kz]
                dA = sp.diff(A_sym, k_sym[d])
                # Evaluate at P
                dA_at_P = dA.subs([(kx, sp.pi), (ky, sp.pi), (kz, sp.pi)])
                dA_at_P = sp.simplify(dA_at_P)
                # Project
                dA_proj_at_P = sp.simplify(U_P.H * dA_at_P * U_P)
                dA_proj[(a, c, d)] = dA_proj_at_P

    # The full vertex: V^{ac}(δk) ≈ A^{ac}(P)|_proj + δk_d × dA_proj[(a,c,d)] + O(δk²)
    # For the matter loop (which uses LINEAR vertex in δk), the relevant part is dA_proj.

    print()
    print("  Compare to naive Iorio form V^{ac}(δk) = v_F δk_c σ^a (cone-effective):")
    print(f"    v_F^P = √3/6")
    v_F_P = sp.sqrt(3) / 6

    # Pauli matrices on 2-dim subspace
    sigma_x = sp.Matrix([[0, 1], [1, 0]])
    sigma_y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sigma_z = sp.Matrix([[1, 0], [0, -1]])
    sigma = [sigma_x, sigma_y, sigma_z]

    # Naive: V_naive[(a,c,d)] should be v_F × δ_{cd} × σ^a (one form to compare)
    # Or: v_F × δ_{ad} × σ^c (another form)
    # Generic: linear in δk_c × σ^a
    print()
    print("  Compute decomposition of dA_proj[(a,c,d)] in σ_a basis:")
    print()
    print(f"    {'(a,c,d)':>10s}  {'I':>14s} {'σ_x':>14s} {'σ_y':>14s} {'σ_z':>14s}")
    coord = ['x', 'y', 'z']

    decomp_table = {}
    for a in range(3):
        for c in range(3):
            for d in range(3):
                M = dA_proj[(a, c, d)]
                # Decompose M = c_I I + c_x σ_x + c_y σ_y + c_z σ_z
                # c_I = (1/2) Tr M, c_α = (1/2) Tr(σ_α M)
                c_I = sp.simplify(M.trace() / 2)
                c_x = sp.simplify((sigma_x * M).trace() / 2)
                c_y = sp.simplify((sigma_y * M).trace() / 2)
                c_z = sp.simplify((sigma_z * M).trace() / 2)
                decomp_table[(a, c, d)] = (c_I, c_x, c_y, c_z)
                # Print key (a,c,d) entries
                if (a, c) in [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)] or d == 2:
                    print(f"    ({coord[a]},{coord[c]},{coord[d]}):  "
                          f"{str(c_I):>14s} {str(c_x):>14s} {str(c_y):>14s} {str(c_z):>14s}")

    print()
    print("  Identify dominant tensor structure...")
    print()

    # Compute total numerical "Frobenius norm" of dA_proj per cone (numerical evaluation of v_F)
    print(f"  For comparison, naive Iorio gives:")
    print(f"    V_naive^{{ac}}(δk) = v_F δk_c σ^a")
    print(f"    Decomp: c_α = v_F δk_c δ_{{α,a}} (only σ^a component, with weight v_F δk_c)")
    print(f"    So in our index notation:")
    print(f"    ∂V_naive^{{ac}}/∂δk_d|0 = v_F δ_{{cd}} σ^a")
    print(f"    Decomp coeff: c_a = v_F δ_{{cd}}, all other zero.")
    print()
    print(f"  Compare actual symbolic dA_proj[(a,c,d)] decomp to v_F δ_{{cd}} σ^a structure:")
    naive_match_count = 0
    naive_mismatch_count = 0
    for (a, c, d), (cI, cx, cy, cz) in decomp_table.items():
        # Naive: c_a = v_F δ_{cd}, c_others = 0
        coeffs_actual = {0: cx, 1: cy, 2: cz}
        coeffs_naive = {alpha: (v_F_P if (alpha == a and c == d) else 0) for alpha in range(3)}
        # Note: also c_I should be zero in naive (only σ contribution)
        naive_match = all(sp.simplify(coeffs_actual[alpha] - coeffs_naive[alpha]) == 0 for alpha in range(3)) and (cI == 0)
        if naive_match:
            naive_match_count += 1
        else:
            naive_mismatch_count += 1
    print(f"  Naive Iorio match count: {naive_match_count} / 27 (a,c,d) entries")
    print(f"  Mismatch count:           {naive_mismatch_count} / 27")

    # Output a clean summary of the structural form
    print()
    print(f"  Total Frobenius norm of dA_proj tensor (numerical):")
    total_norm_sq = 0
    for (a, c, d), (cI, cx, cy, cz) in decomp_table.items():
        for cα in [cI, cx, cy, cz]:
            total_norm_sq += float(sp.Abs(cα)**2)
    print(f"    ‖dA_proj‖² = {total_norm_sq:.6f}")
    print(f"    Naive Iorio prediction (v_F^P)² × 9 (3 a × 3 d, c=d) = {float(v_F_P**2 * 9):.6f}")
    print(f"    Ratio: {total_norm_sq / float(v_F_P**2 * 9):.4f}")


if __name__ == "__main__":
    main()
