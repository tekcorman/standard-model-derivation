#!/usr/bin/env python3
"""
R2: TBM b_0 = 1/2 verification on the 4x4 Bloch adjacency A(P) of srs.

Sprint alpha, Task alpha.1 (R2 execution) per an internal working note

GOAL: derive b_0 = 1/2 structurally from TBM basis overlap, closing the
dark-map Class 2 coefficient derivation.

Specifically, the claim in predictions/dark_extraction_map.py line 113 is:
    eps_Re^2 = Re^2(h) * b_0 = Re^2(h) * (1/2) = 3/8
    eps_Im^2 = Im^2(h) = 5/4
    Class 2 coefficient = eps_Im^2 / (2 * eps_Re^2) = 5/3

This script verifies b_0 = 1/2 as the TBM overlap magnitude squared
|<nu_2|mu>|^2 or |<nu_3|tau>|^2 (whichever appears in the mass-squared
decomposition), computed from the explicit TBM eigenvectors of the
4x4 Bloch adjacency A(P) at the P-point.

STATUS: EXECUTION SCRIPT. Verifies or refutes the structural claim
from an internal working note §7e.

If verified: ADOPTED-DARK-MAP's b_0 component graduates to derived.
If refuted: the structural identification in §7e was wrong and we
need a different derivation route.
"""

import sympy as sp
from sympy import (
    Matrix, I, sqrt, Rational, simplify, conjugate, exp, pi,
    symbols, zeros, eye, trigsimp, radsimp, nsimplify, fraction, N
)


def header(title):
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)
    print()


def build_A_P():
    """
    Build the 4x4 Bloch adjacency A(P) at the P-point of srs.
    From predictions/B_P_doubly_degenerate_h_derivation.md Step 4:

        A(P) = [[ 0, -i, -i, -i],
                [ i,  0, -i,  i],
                [ i,  i,  0, -i],
                [ i, -i,  i,  0]]

    This is 4x4 Hermitian with char poly (lambda^2 - 3)^2,
    giving eigenvalues +sqrt(3) and -sqrt(3), each with multiplicity 2.
    """
    A = Matrix([
        [0, -I, -I, -I],
        [I,  0, -I,  I],
        [I,  I,  0, -I],
        [I, -I,  I,  0]
    ])
    assert A == A.H, "A(P) must be Hermitian"
    return A


def build_C3_permutation():
    """
    The body-diagonal C_3 acts on the 4-vertex primitive cell as
    sigma = (v_0)(v_1 v_3 v_2), i.e. fixes v_0 and cycles v_1 -> v_3 -> v_2 -> v_1.
    From predictions/B_P_doubly_degenerate_h_derivation.md Step 3.
    """
    P_sigma = Matrix([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]
    ])
    assert P_sigma**3 == eye(4), "C_3 must satisfy P^3 = I"
    return P_sigma


def verify_commutation(A, C3):
    """A(P) must commute with C_3 for simultaneous eigenbasis."""
    comm = A * C3 - C3 * A
    return comm == zeros(4, 4)


def main():
    header("R2: TBM b_0 verification on 4x4 Bloch A(P)")

    omega = exp(2 * pi * I / 3)
    omega = sp.simplify(omega)
    omega_bar = sp.simplify(conjugate(omega))

    A = build_A_P()
    C3 = build_C3_permutation()
    print("A(P) = "); sp.pprint(A)
    print()
    print("C_3 permutation matrix: "); sp.pprint(C3)
    print()

    print(f"[A, C_3] = 0 (commutation check): {verify_commutation(A, C3)}")
    print()

    # ===============================================================
    # Step 1: A(P) eigenstructure
    # ===============================================================
    header("Step 1: A(P) eigenvalues and eigenspaces")

    char_poly = A.charpoly(sp.Symbol('lam'))
    print(f"Characteristic polynomial: {char_poly.as_expr()}")
    print(f"Factored: {sp.factor(char_poly.as_expr())}")
    print()

    eig_data = A.eigenvects()
    # eigenvects returns [(eigval, multiplicity, [eigvectors]), ...]

    for eval_, mult, vecs in eig_data:
        eval_simp = sp.simplify(eval_)
        print(f"Eigenvalue: {eval_simp}, algebraic multiplicity: {mult}")
        for i, v in enumerate(vecs):
            v_simp = sp.simplify(v)
            print(f"  Eigenvector {i}: {v_simp.T}")
        print()

    # ===============================================================
    # Step 2: C_3 adapted basis
    # ===============================================================
    header("Step 2: C_3 adapted basis for the (v_1, v_2, v_3) block")

    # Basis vectors
    # e0 = |v_0>  (trivial, fixed by sigma)
    e0 = Matrix([1, 0, 0, 0])
    # e1 = (|v_1> + |v_2> + |v_3>) / sqrt(3)  (trivial, symmetric)
    e1 = Matrix([0, 1, 1, 1]) / sqrt(3)
    # e2 = (|v_1> + omega|v_2> + omega^2|v_3>) / sqrt(3)  (omega eigenstate)
    e2 = Matrix([0, 1, omega, omega**2]) / sqrt(3)
    # e3 = (|v_1> + omega^2|v_2> + omega|v_3>) / sqrt(3)  (omega^2 eigenstate)
    e3 = Matrix([0, 1, omega**2, omega]) / sqrt(3)

    # Verify C_3 eigenvalues
    for i, (name, vec) in enumerate([('e_0 (trivial, v_0)', e0),
                                      ('e_1 (trivial, sym of v_1,v_2,v_3)', e1),
                                      ('e_2 (omega)', e2),
                                      ('e_3 (omega^2)', e3)]):
        Cv = C3 * vec
        # Find proportionality factor
        nonzero_idx = None
        for j in range(4):
            if vec[j] != 0:
                nonzero_idx = j
                break
        if nonzero_idx is not None and vec[nonzero_idx] != 0:
            ratio = sp.simplify(Cv[nonzero_idx] / vec[nonzero_idx])
            print(f"  {name}: C_3 eigenvalue = {ratio}")

    print()

    # ===============================================================
    # Step 3: A(P) in C_3 adapted basis
    # ===============================================================
    header("Step 3: A(P) matrix elements in C_3 adapted basis")

    basis = [e0, e1, e2, e3]
    A_C3 = zeros(4, 4)
    for i in range(4):
        for j in range(4):
            elem = (basis[i].H * A * basis[j])[0, 0]
            A_C3[i, j] = sp.simplify(elem)

    print("A(P) in (e_0, e_1, e_2, e_3) basis:")
    sp.pprint(A_C3)
    print()

    # The (e_0, e_1) block should be a 2x2 trivial-sector submatrix.
    # e_2 should give diagonal +sqrt(3), e_3 should give diagonal -sqrt(3).
    print(f"<e_2|A(P)|e_2> = {sp.simplify(A_C3[2,2])}  (should be +sqrt(3))")
    print(f"<e_3|A(P)|e_3> = {sp.simplify(A_C3[3,3])}  (should be -sqrt(3))")
    print()

    # Trivial 2x2 block
    triv_block = A_C3[:2, :2]
    print("Trivial subspace A(P) block (2x2):")
    sp.pprint(triv_block)
    triv_evals = triv_block.eigenvals()
    print(f"Eigenvalues of trivial block: {triv_evals}")
    print()

    # ===============================================================
    # Step 4: +sqrt(3) and -sqrt(3) eigenspaces and their C_3 content
    # ===============================================================
    header("Step 4: eigenspaces decomposed by C_3 irrep")

    # +sqrt(3) eigenspace: the diag +sqrt(3) combination in trivial block + e_2
    # -sqrt(3) eigenspace: the other diag combination + e_3

    # Diagonalize the 2x2 trivial block
    triv_evects = triv_block.eigenvects()
    for eval_, mult, vecs in triv_evects:
        for v in vecs:
            v_simp = sp.simplify(v)
            print(f"  Trivial block eigenvector for lambda={eval_}: {v_simp.T}")

    # In the trivial 2x2 block [[0, -i sqrt(3)], [i sqrt(3), 0]], the
    # eigenvalues are +sqrt(3) and -sqrt(3) with eigenvectors:
    # +sqrt(3): (1, i) / sqrt(2)  -> |T+> = (e_0 + i e_1) / sqrt(2)
    # -sqrt(3): (1, -i) / sqrt(2) -> |T-> = (e_0 - i e_1) / sqrt(2)

    T_plus = (e0 + I * e1) / sqrt(2)
    T_minus = (e0 - I * e1) / sqrt(2)

    # Verify
    print()
    print(f"A(P)|T+> = sqrt(3) |T+>? {sp.simplify(A * T_plus - sqrt(3) * T_plus) == zeros(4, 1)}")
    print(f"A(P)|T-> = -sqrt(3) |T->? {sp.simplify(A * T_minus + sqrt(3) * T_minus) == zeros(4, 1)}")
    print()

    # So the +sqrt(3) eigenspace is spanned by |T+> (trivial) and e_2 (omega).
    # The -sqrt(3) eigenspace is spanned by |T-> (trivial) and e_3 (omega^2).

    print("SUMMARY:")
    print("  +sqrt(3) eigenspace (dim 2): span{|T+>, |omega>} where |omega> = e_2")
    print("  -sqrt(3) eigenspace (dim 2): span{|T->, |omega^2>} where |omega^2> = e_3")
    print("  Under C_3: |T+>, |T-> are trivial; |omega> has C_3 eigenvalue omega;")
    print("             |omega^2> has C_3 eigenvalue omega^2.")

    # ===============================================================
    # Step 5: Attempt TBM eigenvector identification
    # ===============================================================
    header("Step 5: TBM eigenvector identification")

    print("In TBM, the 3 generations map to flavor states via U_TBM:")
    print("  |nu_e> = (2/sqrt(6))|nu_1> + (1/sqrt(3))|nu_2> + 0 |nu_3>")
    print("  |nu_mu> = -(1/sqrt(6))|nu_1> + (1/sqrt(3))|nu_2> + (1/sqrt(2))|nu_3>")
    print("  |nu_tau> = -(1/sqrt(6))|nu_1> + (1/sqrt(3))|nu_2> - (1/sqrt(2))|nu_3>")
    print()
    print("Equal TBM weight |U_{mu3}|^2 = |U_{tau3}|^2 = 1/2.")
    print("This is the CANDIDATE b_0 = 1/2 factor.")
    print()

    # In our 4-dim space, we have 4 degenerate states (all at m^2 = 3).
    # TBM assigns 3 of them to (nu_1, nu_2, nu_3) and drops one.
    #
    # The natural candidate for nu_3 (heaviest, with U_{mu3} = 1/sqrt(2)):
    #   antisymmetric combination of |omega> and |omega^2>: (e_2 - e_3) / sqrt(2)
    # This has NO v_0 or e_1 (trivial) component -- consistent with U_{e3} = 0 in TBM.

    nu3_candidate = (e2 - e3) / sqrt(2)
    nu3_simp = sp.simplify(nu3_candidate)
    print(f"|nu_3> candidate (antisym of |omega>, |omega^2>): ")
    sp.pprint(nu3_simp.T)
    print()

    # Check that this has no |v_0> component (<v_0|nu_3> = 0):
    v0_overlap = sp.simplify(nu3_candidate[0])
    print(f"<v_0|nu_3> = {v0_overlap}  (should be 0 for TBM U_{{e3}} = 0)")

    # Also it should have no e_1 overlap (trivial projection):
    e1_overlap = sp.simplify((e1.H * nu3_candidate)[0, 0])
    print(f"<e_1|nu_3> = {e1_overlap}  (should be 0 for TBM, no trivial content)")
    print()

    # Now |nu_2>: the symmetric combination? It should have equal weight in
    # e, mu, tau (|U_{alpha 2}|^2 = 1/3 for all alpha).
    # Candidate: (e_2 + e_3) / sqrt(2) is symmetric in omega <-> omega^2,
    # but has content ~ -(|v_1> + |v_2> + |v_3>)/sqrt(3) = -e_1 (trivial).

    sym_cand = (e2 + e3) / sqrt(2)
    sym_simp = sp.simplify(sym_cand)
    print(f"|(omega) + (omega^2)>/sqrt(2) candidate: ")
    sp.pprint(sym_simp.T)
    # Simplify: (e_2 + e_3)/sqrt(2) should equal some multiple of e_1
    # since (omega + omega^2) = -1, giving coefficient -1 for |v_2> and |v_3>
    # and coefficient 2 for |v_1>... hmm let me compute:
    # e_2 = (|v_1> + omega|v_2> + omega^2|v_3>)/sqrt(3)
    # e_3 = (|v_1> + omega^2|v_2> + omega|v_3>)/sqrt(3)
    # (e_2 + e_3)/sqrt(2) = (2|v_1> - |v_2> - |v_3>)/(sqrt(3)sqrt(2)) = (2|v_1> - |v_2> - |v_3>)/sqrt(6)
    #
    # This is NOT equal to e_1 = (|v_1> + |v_2> + |v_3>)/sqrt(3).
    # It's orthogonal to e_1:
    e1_overlap_sym = sp.simplify((e1.H * sym_cand)[0, 0])
    print(f"<e_1|(omega)+(omega^2)>/sqrt(2) = {e1_overlap_sym}  (should be 0, orthogonal)")

    # So (e_2 + e_3)/sqrt(2) is a trivial-irrep vector orthogonal to e_1.
    # In the 2-dim trivial subspace spanned by e_0 and e_1, we have another
    # orthogonal direction... wait, no. (e_2 + e_3)/sqrt(2) transforms as
    # omega + omega^2 = -1 under C_3, so it's C_3-trivial? Let me check.

    C3_on_sym = C3 * sym_cand
    ratio_sym = sp.simplify(C3_on_sym[1] / sym_cand[1]) if sym_cand[1] != 0 else "undefined"
    print(f"C_3 * (e_2 + e_3)/sqrt(2) gives C_3 eigenvalue... let me verify transforms")

    # Actually (e_2 + e_3)/sqrt(2) is NOT a C_3 eigenvector since e_2 and e_3
    # have DIFFERENT C_3 eigenvalues. The sum transforms non-trivially.
    # So our candidate nu_2 is NOT a C_3 eigenvector in this basis.

    print()
    print("OBSERVATION: (e_2 + e_3)/sqrt(2) is NOT a C_3 eigenvector since")
    print("             e_2 has C_3 eigenvalue omega and e_3 has omega^2.")
    print("             So this candidate for nu_2 is problematic structurally.")
    print()

    # ===============================================================
    # Step 6: Partial result + honest assessment
    # ===============================================================
    header("Step 6: honest assessment")

    print("PARTIAL RESULT:")
    print("  - A(P) 4x4 structure verified: eigenvalues +-sqrt(3), mult 2 each.")
    print("  - +sqrt(3) = trivial + omega; -sqrt(3) = trivial + omega^2 verified.")
    print("  - nu_3 candidate (e_2 - e_3)/sqrt(2) has NO v_0 or e_1 component,")
    print("    consistent with TBM U_{e3} = 0.")
    print("  - But nu_3 candidate is a SUPERPOSITION of +sqrt(3) and -sqrt(3)")
    print("    eigenvectors -- NOT a single-energy eigenstate.")
    print()
    print("This means: at tree level (A(P) alone), nu_3 is NOT a mass eigenstate.")
    print("The TBM structure is induced by the dark perturbation Sigma(h),")
    print("which breaks the 4-fold mass-squared degeneracy (all at m^2 = 3)")
    print("and selects specific superpositions as mass eigenstates.")
    print()
    print("Consequence for b_0 = 1/2 derivation:")
    print("  The overlap |<nu_3|mu>|^2 = 1/2 cannot be computed from A(P)")
    print("  eigenvectors alone -- it requires the PERTURBED mass eigenvectors")
    print("  after applying Sigma(h).")
    print()
    print("Remaining work (for a future session):")
    print("  1. Define the dark perturbation Sigma(h) on the 4-dim subspace explicitly.")
    print("  2. Diagonalize A(P) + alpha_1 * Sigma(h) to get perturbed eigenvectors.")
    print("  3. Project to (nu_e, nu_mu, nu_tau) flavor basis via TBM ansatz.")
    print("  4. Compute <nu_i|nu_alpha> overlaps and verify |<nu_3|mu>|^2 = 1/2.")
    print()
    print("Current R2 status: PARTIAL PROGRESS, not closure.")
    print("The structural framework is verified; the perturbation-level calculation")
    print("is the outstanding work.")


if __name__ == "__main__":
    main()
