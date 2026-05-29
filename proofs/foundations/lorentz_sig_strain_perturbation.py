#!/usr/bin/env python3
"""
Strain perturbation of the srs Bloch Hamiltonian — Iorio-elastic Session 1.

Deliverable for an internal working note Session 1:
explicit sympy computation of the 9 strain-perturbation matrices
A^{ab}(k) ∈ Mat_4(C[k_x, k_y, k_z]) such that

    H^{strain}(k, x) = sum_{a,b ∈ {x,y,z}} A^{ab}(k) (∂_a u_b)(x)

where (∂_a u_b) is the deformation gradient (covering both strain u_{ab} and
rotation ω_{ab}).

Derivation. Under slow deformation u(x), each NN bond's displacement
vector r_b acquires a perturbation δr_b = (∂_a u_b) (r_b)^a (linear in
∂u, slow-variation regime). The Bloch matrix entry

    H_{βα}(k) = sum_{bonds (α→β,n)} exp(i k_cart · r_b)

picks up at first order in ∂u

    δH_{βα}(k) = i sum_{bonds (α→β,n)} exp(i k · r_b) · (k_a (∂_a u_c) r_b^c)
               = sum_{a,c} (∂_a u_c) · [i sum_{bonds (α→β,n)} exp(i k · r_b) k_a r_b^c]
               = sum_{a,c} (∂_a u_c) · A^{ac}_{βα}(k)

so

    A^{ac}_{βα}(k) = i sum_{bonds (α→β,n)} exp(i k_cart · r_b) k_a r_b^c.

This script computes these 9 matrices explicitly using exact-rational atom
positions and primitive vectors, verifies their Hermiticity properties, and
prints them for use in the Session-2 projection-to-Γ-cone derivation.

Notes on r_b. The Cartesian bond displacement is

    r_b = R_β + n_1 a_1 + n_2 a_2 + n_3 a_3 - R_α

with R_α the Cartesian Wyckoff-8a position of atom α and a_i the BCC
primitive vectors. Both are exact rationals at x = 1/8.
"""

import sympy as sp


# =============================================================================
# Setup: exact-rational Wyckoff 8a + BCC primitives + bond list
# =============================================================================

# Wyckoff 8a positions (4 atoms in primitive cell; remaining 4 are body-centred)
ATOMS = [
    sp.Matrix([sp.Rational(1, 8), sp.Rational(1, 8), sp.Rational(1, 8)]),
    sp.Matrix([sp.Rational(3, 8), sp.Rational(7, 8), sp.Rational(5, 8)]),
    sp.Matrix([sp.Rational(7, 8), sp.Rational(5, 8), sp.Rational(3, 8)]),
    sp.Matrix([sp.Rational(5, 8), sp.Rational(3, 8), sp.Rational(7, 8)]),
]

# BCC primitive vectors (a = 1)
A_PRIM = [
    sp.Matrix([sp.Rational(-1, 2), sp.Rational(1, 2), sp.Rational(1, 2)]),
    sp.Matrix([sp.Rational(1, 2), sp.Rational(-1, 2), sp.Rational(1, 2)]),
    sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(-1, 2)]),
]

# 6 undirected edges of the srs primitive cell K_4 quotient (matches
# proofs/foundations/theorem_B2_signature.py).
CELL_EDGES = [
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
]

# 12 directed bonds (forward + reverse) for the Bloch construction.
BONDS = []
for src, tgt, cell in CELL_EDGES:
    BONDS.append((src, tgt, cell))
    BONDS.append((tgt, src, tuple(-c for c in cell)))


def bond_displacement(src, tgt, cell):
    """Return the exact rational Cartesian bond displacement r_b = R_β + cell - R_α."""
    rb = ATOMS[tgt] - ATOMS[src]
    for i in range(3):
        rb = rb + cell[i] * A_PRIM[i]
    return rb


# =============================================================================
# Symbolic Cartesian k
# =============================================================================

kx, ky, kz = sp.symbols('kx ky kz', real=True)
k_cart = [kx, ky, kz]


# =============================================================================
# Compute A^{ac}_{βα}(k) for the 9 (a, c) pairs
# =============================================================================

def build_strain_matrices():
    """Compute the 9 sympy 4×4 matrices A^{ac}(k) for a, c in {0,1,2}."""
    A_matrices = {}
    for a in range(3):
        for c in range(3):
            A_ac = sp.zeros(4, 4)
            for src, tgt, cell in BONDS:
                rb = bond_displacement(src, tgt, cell)
                phase = sp.exp(sp.I * sum(k_cart[i] * rb[i] for i in range(3)))
                # A^{ac}_{βα}(k) += i · phase · k_a · r_b^c
                contribution = sp.I * phase * k_cart[a] * rb[c]
                A_ac[tgt, src] = A_ac[tgt, src] + contribution
            A_matrices[(a, c)] = sp.simplify(A_ac)
    return A_matrices


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("Iorio-elastic Session 1: strain-perturbation matrices A^{ac}(k) on srs")

    print("\nBond list (exact Cartesian displacements):")
    for src, tgt, cell in BONDS:
        rb = bond_displacement(src, tgt, cell)
        print(f"  bond {src} -> {tgt}, cell {cell}: r_b = {rb.T.tolist()[0]}")

    A = build_strain_matrices()

    # Verify Hermiticity: H is Hermitian, so δH = sum A^{ac} (∂_a u_c) is also
    # Hermitian. Since (∂_a u_c) is REAL, each individual A^{ac} must satisfy
    # A^{ac}(k)^† = A^{ac}(-k) — equivalently, A^{ac}(k) is Hermitian after
    # the Bloch convention is Hermitised.
    #
    # Specifically: H_{βα}(k) = (H_{αβ}(k))^* implies that
    #   A^{ac}_{βα}(k) = (A^{ac}_{αβ}(k))^*
    # for each (a, c). Verify this.

    print("\n--- Hermiticity check: A^{ac}(k)^† = A^{ac}(k) ? ---")
    for (a, c), mat in A.items():
        diff = sp.simplify(mat - mat.H)
        is_herm = diff.is_zero_matrix
        print(f"  A^{{{a},{c}}}(k):  Hermitian = {is_herm}")
        if not is_herm:
            # Print the diff
            sp.pprint(diff)

    # Print A^{ac} for the diagonal (a == c) cases — these are the "stretch"
    # contributions to the symmetric strain u_{ac}.
    print("\n--- Diagonal A^{aa}(k) (stretch contributions) ---")
    for a in range(3):
        coord = ['x', 'y', 'z'][a]
        print(f"\n  A^{{{coord},{coord}}}(k) =")
        sp.pprint(A[(a, a)])

    # Symmetric combinations A^{(ac)} = (A^{ac} + A^{ca})/2 enter via the
    # symmetric strain u_{ac}; antisymmetric A^{[ac]} = (A^{ac} - A^{ca})/2
    # enter via the rotation ω_{ac}.
    print("\n--- Symmetrisation check: A^{(xy)} = (A^{xy} + A^{yx})/2 ---")
    A_xy_sym = sp.simplify((A[(0, 1)] + A[(1, 0)]) / 2)
    print(f"  Symmetric part dim: {A_xy_sym.shape}; sample entry [1, 0]:")
    sp.pprint(A_xy_sym[1, 0])

    # Complete the table of all 9 in terse output format
    header("Full 9-matrix A^{ac}(k) table summary")
    print()
    print("  Each A^{ac}(k) is a 4×4 Hermitian (after k → k_cart) matrix with linear-in-k entries.")
    print("  The strain-perturbation Bloch matrix is")
    print("      δH(k, x) = Σ_{a,c} A^{ac}(k) · (∂_a u_c)(x).")
    print()
    print("  Trace check: each A^{ac}(k) is traceless (since each diagonal entry vanishes")
    print("  -- bonds connect distinct atoms, no self-loops).")
    for (a, c), mat in A.items():
        coord_a = ['x', 'y', 'z'][a]
        coord_c = ['x', 'y', 'z'][c]
        tr = sp.simplify(mat.trace())
        print(f"  A^{{{coord_a},{coord_c}}}(k):  trace = {tr}")

    # Save a concise reference of the matrices to stdout for next-session use
    header("READY FOR SESSION 2")
    print()
    print("  Next: project each A^{ac}(k) onto the 3-dim λ=-1 subspace at Γ")
    print("  (basis {g_1, g_2, g_3} of v_0^⊥, see")
    print("   proofs/foundations/lorentz_sig_dirac_cone_symbolic.py).")
    print("  Match the result against (1/2) e^a_b k^b S_a to extract the vielbein")
    print("  prefactor β in e^a_b = δ^a_b + β ∂^a u_b.")
    print()
    print("  This script computes the 9 A^{ac}(k) matrices ON-DEMAND;")
    print("  Session 2 imports build_strain_matrices() and consumes them.")


if __name__ == "__main__":
    main()
