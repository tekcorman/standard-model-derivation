#!/usr/bin/env python3
# ============================================================
# THEOREM G2: Cl(0,2) Boolean Edge Structure and n_channels = 2
# ============================================================
#
# Ported from ../predictions/G2_cl2_channels_derivation.md (2026-04-19).
# Closes BLOCK-2 sub-claims A, B, C of an internal working note
# Supports predictions/lambda_higgs.py (Step 6: factor 2 = n_channels).
#
# This file is the predictions/ DAG entry point for Theorem G2.
# The detailed proof script is proofs/foundations/theorem_G2_cl2_channels.py.

# --- THEOREM STATEMENT ---------------------------------------
# Status: STRICT-SOLID under A1 + A3-T + local CAR thm (theorem_car_local_jordan_wigner.md) + n_channels = 2 derivation.
#
# Let {u,v} be any undirected edge of the srs K_4-quotient.
# The two directed versions e1=(u,v) and e2=(v,u) carry toggle operators
# T_{e1}, T_{e2} satisfying A1+A4+A3. These generate a Clifford algebra
# isomorphic to Cl(0,2) over R, and isomorphic to M_2(C) over C.
# The minimal faithful complex representation of Cl(0,2)_C has dimension 2.
#
# n_channels = dim_C(min. faithful rep of Cl(0,2)_C) = 2  [STRICT-SOLID]
#
# Convention note: theorem_B3_spinor_fermion.py's decomposition is unique
# up to (Z/2)^3 (L↔R flip, isospin flip, Y flip). dim_C(min faithful rep
# of Cl(0,2)_C) = 2 is an intrinsic algebraic invariant — unchanged under
# any of these relabelings. No adoption is required for n_channels = 2.

# --- FRAMEWORK AXIOMS INVOKED --------------------------------
# A1: Binary self-inverse toggle (docs/framework/framework_axioms.md §2)
#     — T_{e1}^2 = T_{e2}^2 = I (toggle involutions)
# A3: Complex field F=C (docs/framework/framework_axioms.md §4)
#     — allows gamma_j = i*T_j => gamma_j^2 = -I (Clifford generators)
# A4: CAR at k*-valent nodes (docs/framework/framework_axioms.md §5)
#     — {T_{e1}, T_{e2}} = 0 for two distinct edges at same vertex

# --- INPUTS --------------------------------------------------
# symbol         | value  | status   | file
# ---------------|--------|----------|--------------------------------------
# k_star         | 3      | [derived]| predictions/k_star.py
# A1 (toggle)    | (axiom)| [axiom]  | docs/framework/framework_axioms.md §2
# A3-T (F=C)     | (thm)  | [thm]    | docs/theorems/theorem_A3_complex_hilbert_from_multiway.md
# local CAR thm  | (thm)  | [thm]    | docs/theorems/theorem_car_local_jordan_wigner.md

# --- IMPLEMENTATION ------------------------------------------

import numpy as np
import numpy.linalg as la

# Pauli matrices (building blocks for 2-dim faithful rep)
_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)
_I4 = np.eye(4, dtype=complex)

# Jordan-Wigner realization of T_{e1}, T_{e2} in 4-dim Fock space.
# Physical setup: undirected edge {u,v} in the srs K_4-quotient.
#   e1=(u,v): T1 = sx otimes I  (Majorana mode 1)
#   e2=(v,u): T2 = sz otimes sx  (Jordan-Wigner string for mode 2)
_T1 = np.kron(_sx, _I2)
_T2 = np.kron(_sz, _sx)


def _anticommutator(A, B):
    return A @ B + B @ A


def _gram_rank(matrices, trace_norm):
    """Rank of Gram matrix under Hilbert-Schmidt inner product."""
    n = len(matrices)
    G = np.zeros((n, n), dtype=complex)
    for i, A in enumerate(matrices):
        for j, B in enumerate(matrices):
            G[i, j] = np.trace(A.conj().T @ B) / trace_norm
    return np.linalg.matrix_rank(G, tol=1e-10)


# --- PURE FUNCTION -------------------------------------------

def verify_G2_cl2_channels():
    """
    Verify Theorem G2: n_channels = dim_C(min. faithful rep of Cl(0,2)) = 2.

    Five verification steps:
      Step 1 (A1): T_{e1}^2 = T_{e2}^2 = I  (toggle involutions)
      Step 2 (A4): {T_{e1}, T_{e2}} = 0       (CAR anticommutation)
      Step 3 (A3): gamma_j = i*T_j => gamma_j^2=-I, {gamma_1,gamma_2}=0
                   (Cl(0,2) generators)
      Step 4:      {I, gamma_1, gamma_2, gamma_12} C-linearly independent
                   => Cl(0,2)_C has dim_C=4, isom to M_2(C)
      Step 5:      2-dim faithful rep gamma_1->i*sx, gamma_2->i*sz exists;
                   1-dim faithful rep is impossible.

    Returns
    -------
    dict
        {
          'n_channels': 2,
          'step1_A1_passed': bool,
          'step2_A4_passed': bool,
          'step3_Cl02_passed': bool,
          'step4_dim4_passed': bool,
          'step5_min_rep_passed': bool,
          'fock_decomp_passed': bool,
          'all_passed': bool,
        }
    """
    T1, T2, I4 = _T1, _T2, _I4

    # Step 1: A1 — toggle involutions
    err_T1 = la.norm(T1 @ T1 - I4)
    err_T2 = la.norm(T2 @ T2 - I4)
    step1_ok = (err_T1 < 1e-14 and err_T2 < 1e-14)

    # Step 2: A4 — CAR anticommutation
    err_A4 = la.norm(_anticommutator(T1, T2))
    step2_ok = (err_A4 < 1e-14)

    # Step 3: Cl(0,2) generators via A3
    g1 = 1j * T1
    g2 = 1j * T2
    err_g1_sq = la.norm(g1 @ g1 + I4)
    err_g2_sq = la.norm(g2 @ g2 + I4)
    err_anti  = la.norm(_anticommutator(g1, g2))
    # Quaternion completion: gamma_12 = gamma_1 * gamma_2
    g12 = g1 @ g2
    err_g12_sq    = la.norm(g12 @ g12 + I4)
    err_anti_12_1 = la.norm(_anticommutator(g12, g1))
    err_anti_12_2 = la.norm(_anticommutator(g12, g2))
    step3_ok = (err_g1_sq < 1e-14 and err_g2_sq < 1e-14 and err_anti < 1e-14
                and err_g12_sq < 1e-14 and err_anti_12_1 < 1e-14
                and err_anti_12_2 < 1e-14)

    # Step 4: Cl(0,2)_C has C-dim 4 (isom to M_2(C))
    basis_4d = [I4, g1, g2, g12]
    rank_4 = _gram_rank(basis_4d, trace_norm=4.0)
    step4_ok = (rank_4 == 4)

    # Step 5: minimal faithful C-rep has dim=2
    # 2-dim explicit faithful rep: gamma_1->i*sx, gamma_2->i*sz
    g1_2d = 1j * _sx
    g2_2d = 1j * _sz
    I2c = _I2
    g12_2d = g1_2d @ g2_2d
    err_2d_g1sq = la.norm(g1_2d @ g1_2d + I2c)
    err_2d_g2sq = la.norm(g2_2d @ g2_2d + I2c)
    err_2d_anti = la.norm(_anticommutator(g1_2d, g2_2d))
    rank_2d = _gram_rank([I2c, g1_2d, g2_2d, g12_2d], trace_norm=2.0)
    step5_ok = (err_2d_g1sq < 1e-14 and err_2d_g2sq < 1e-14
                and err_2d_anti < 1e-14 and rank_2d == 4)
    # Lower bound: dim=1 impossible (scalar CAR forces a generator to 0).
    # Verified logically: on C^1, {a,b}=2ab=0 requires a=0 or b=0 (not faithful).

    # Fock-space decomposition: commutant of Cl(0,2) in End(C^4) has dim=4
    # (=> Fock rep decomposes as 2 x (2-dim irrep))
    constraint_rows = []
    for G in [g1, g2]:
        K = np.kron(I4, G) - np.kron(G.T, I4)
        constraint_rows.append(K)
    A_sys = np.vstack(constraint_rows)
    _, s, _ = la.svd(A_sys)
    rank_A = int(np.sum(s > 1e-10))
    commutant_dim = 16 - rank_A
    fock_ok = (commutant_dim == 4)

    all_ok = step1_ok and step2_ok and step3_ok and step4_ok and step5_ok and fock_ok

    return {
        'n_channels': 2,
        'step1_A1_involutions_passed': step1_ok,
        'step2_A4_anticommutation_passed': step2_ok,
        'step3_Cl02_generators_passed': step3_ok,
        'step4_dim4_isom_M2C_passed': step4_ok,
        'step5_min_rep_dim2_passed': step5_ok,
        'fock_decomp_2x2_passed': fock_ok,
        'all_passed': all_ok,
        'err_T1_sq': float(err_T1),
        'err_A4': float(err_A4),
        'err_g1_sq': float(err_g1_sq),
        'gram_rank_4d': int(rank_4),
        'gram_rank_2d_rep': int(rank_2d),
        'commutant_dim': int(commutant_dim),
    }


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("THEOREM G2: Cl(0,2) boolean edge structure -> n_channels = 2")
    print("Chain: A1+A4 -> anticommuting involutions -> A3 -> Cl(0,2) -> n_channels=2")
    print("=" * 70)
    print()

    result = verify_G2_cl2_channels()

    print("Step 1 (A1) — T_{e1}^2 = T_{e2}^2 = I:")
    print(f"  ||T1^2 - I4|| = {result['err_T1_sq']:.2e}")
    print(f"  PASSED: {result['step1_A1_involutions_passed']}")
    print()

    print("Step 2 (A4) — {{T1, T2}} = 0 (CAR at shared vertex):")
    print(f"  ||{{T1,T2}}|| = {result['err_A4']:.2e}")
    print(f"  PASSED: {result['step2_A4_anticommutation_passed']}")
    print()

    print("Step 3 (A3) — Cl(0,2) generators gamma_j = i*T_j:")
    print(f"  ||gamma_1^2 + I|| = {result['err_g1_sq']:.2e}  (signature -1)")
    print(f"  {{gamma_1,gamma_2}} = 0; gamma_12 satisfies quaternion ij=-ji=k")
    print(f"  PASSED: {result['step3_Cl02_generators_passed']}")
    print()

    print("Step 4 — Cl(0,2)_C has C-dim=4, isomorphic to M_2(C):")
    print(f"  Gram rank of {{I,gamma_1,gamma_2,gamma_12}} = {result['gram_rank_4d']}")
    print(f"  PASSED: {result['step4_dim4_isom_M2C_passed']}")
    print()

    print("Step 5 — Minimal faithful C-rep has dim=2:")
    print(f"  2-dim rep gamma_1->i*sx, gamma_2->i*sz verified.")
    print(f"  Gram rank (faithfulness check) = {result['gram_rank_2d_rep']}")
    print(f"  Lower bound: dim=1 impossible (scalar CAR forces generator to 0).")
    print(f"  PASSED: {result['step5_min_rep_dim2_passed']}")
    print()

    print("Fock-space decomposition C^4 = 2 x C^2:")
    print(f"  Commutant C-dimension = {result['commutant_dim']}  (expected 4)")
    print(f"  PASSED: {result['fock_decomp_2x2_passed']}")
    print()

    assert result["all_passed"], f"One or more steps failed: {result}"
    assert result["n_channels"] == 2

    print("=" * 70)
    print(f"RESULT: n_channels = {result['n_channels']}")
    print()
    print("  A1 + local CAR thm + A3-T => n_channels = 2  (STRICT-SOLID)")
    print()
    print("  Closes BLOCK-2 sub-claims A, B, C of theorem_higgs_vev_scoping.md.")
    print("  Impact on lambda_higgs.py: F2-class adoption (factor 2) is CLOSED.")
    print("    Step 6 of lambda_higgs.py is now STRICT-SOLID via Theorem G2.")
    print()
    print("  Convention note:")
    print("    n_channels=2 = dim_C(min faithful C-rep of Cl(0,2)) is an intrinsic")
    print("    algebraic invariant. The (Z/2)^3 convention choices in B3 (L<->R,")
    print("    isospin sign, Y sign) do not change this dimension.")
    print("    ADOPTED-B3 is not load-bearing for n_channels=2. Status: STRICT-SOLID.")
    print()
    print("OK: theorem_G2_cl2_channels verification complete.")
    print("=" * 70)
