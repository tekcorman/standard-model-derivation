#!/usr/bin/env python3
"""
Verify the particle-hole conjugation between Γ and H Bloch matrices on srs.

Claim: H(H) = U · (-H(Γ)) · U^† for some diagonal unitary U.

This is the structural origin of the symmetric spectra
  spec H(Γ) = {3, -1, -1, -1}        (top-1 + bottom-3-fold)
  spec H(H) = {-3, +1, +1, +1}       (bottom-1 + top-3-fold)
and pins down whether Γ and H are independent valleys or one pair under
particle-hole conjugation.

If U exists, then the Γ Dirac cone (lower 3 bands at λ=-1) and the H Dirac
cone (upper 3 bands at λ=+1) are conjugate manifestations of ONE structural
feature. The framework's MDL ranking should treat them as a 2-valley pair,
NOT as two independent cones with a Γ-wins tiebreaker.
"""

import sympy as sp

TWO_PI_I = 2 * sp.pi * sp.I

CELL_EDGES = [
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
]
BONDS = []
for src, tgt, cell in CELL_EDGES:
    BONDS.append((src, tgt, cell))
    BONDS.append((tgt, src, tuple(-c for c in cell)))


def bloch_H(k_vec):
    H = sp.zeros(4, 4)
    for src, tgt, cell in BONDS:
        phase = sp.exp(TWO_PI_I * (cell[0]*k_vec[0] + cell[1]*k_vec[1] + cell[2]*k_vec[2]))
        H[tgt, src] = H[tgt, src] + phase
    return H


def main():
    print("=" * 78)
    print("  Particle-hole conjugation between Γ and H")
    print("=" * 78)

    H_G = bloch_H((0, 0, 0))
    H_H = sp.simplify(bloch_H((sp.Rational(-1, 2), sp.Rational(1, 2), sp.Rational(1, 2))))

    print("\nH(Γ) =")
    sp.pprint(H_G)
    print("\nH(H) =")
    sp.pprint(H_H)

    # Direct test: H(H) = -H(Γ) (entrywise)?
    diff = sp.simplify(H_H + H_G)
    print(f"\nH(H) + H(Γ) = (should be 0 if direct entrywise PH):")
    sp.pprint(diff)
    if diff.is_zero_matrix:
        print("\n  ✓ H(H) = -H(Γ) ENTRYWISE.")
        print("    Particle-hole symmetry U = identity (trivial).")
        print("    Γ and H are conjugate via the IDENTITY map composed with energy reflection.")
        print()
        print("  This means: spec H(H) = -spec H(Γ) automatically,")
        print("  and eigenvectors of H(H) at eigenvalue -E are the SAME as eigenvectors")
        print("  of H(Γ) at eigenvalue +E.")
        print()
        print("  In particular, the Γ Dirac cone (lower 3 bands at λ=-1, eigenvectors")
        print("  spanning v_0^⊥) and the H Dirac cone (upper 3 bands at λ=+1) have")
        print("  IDENTICAL eigenvectors -- they are ONE STRUCTURAL FEATURE seen from")
        print("  opposite energy ends.")
        return True

    # If not entrywise, search for a diagonal unitary U
    print("\n  Not entrywise. Searching for non-trivial unitary conjugation...")
    # ... (would need a search if the gauge differs)
    return False


if __name__ == "__main__":
    ok = main()
    print()
    if ok:
        print("=" * 78)
        print("  CONCLUSION")
        print("=" * 78)
        print()
        print("  Γ and H are particle-hole-conjugate manifestations of ONE Dirac")
        print("  sector on srs. The framework's MDL ranking should treat them as a")
        print("  2-valley pair (analogous to graphene's K, K' valleys), not as two")
        print("  independent cones requiring a Γ-vs-H tiebreaker.")
        print()
        print("  Multi-valley picture (revised):")
        print("    Sector 1: triple-Dirac, v_F = 1/2, 2 valleys (Γ + H paired by PH).")
        print("    Sector 2: double-Dirac, v_F = √3/6, 2 valleys (P_lower + P_upper paired by PH).")
        print()
        print("  Total: 4 cones, organised into 2 sectors via particle-hole pairing.")
