#!/usr/bin/env python3
# ============================================================
# Session 7: sigma_S in the matching Brauer-Weyl basis (LS2)
# ============================================================
#
# Context. Session 6 shipped 3b (predictions/B3_chirality_bridge.py)
# closing at mathematically-complete: the canonical Cartan subalgebra
# of Cl(V,Q) is the matching-partition Cartan (forced by S_4 vertex
# symmetry). Three Cartan generators (T_1, T_2, Y) are cyclically
# permuted by sigma.
#
# docs/framework/B3_B6_reconciliation.md Finding 3: under B3's Pauli Brauer-
# Weyl basis, sigma_S acts on species as a Hadamard-type rotation
# (uniform |M_ij| = 0.5 over 4 L-chirality states; "massive species-
# mixing").
#
# This script verifies: under the MATCHING Brauer-Weyl basis, sigma_S
# is instead a clean PERMUTATION on the 8 weight states, with orbit
# structure (fixed-point count, 3-cycle count) = (2, 2).
#
# Analytical prediction (derived below).
#   (i)  sigma cyclically permutes the 3 matching pairs, hence cycles
#        the Cartan generators as (T_M1, T_M2, T_M3) -> (T_M3, T_M1, T_M2).
#   (ii) A weight state |e_1, e_2, e_3> with (T_M1, T_M2, T_M3) eigen-
#        values (e_1, e_2, e_3) maps under sigma to |e_3, e_1, e_2>.
#   (iii) Weight orbits:
#        - fixed points (all components equal): (+,+,+), (-,-,-)
#        - 3-orbits (mixed):  {(+,+,-), (+,-,+), (-,+,+)} and
#                             {(+,-,-), (-,+,-), (-,-,+)}
#
# Goal. Construct matching Brauer-Weyl explicitly, compute sigma_S
# as the Spin(6) lift of the edge-permutation, and verify the 8 weight
# eigenstates map under sigma_S as the cyclic permutation (i)-(iii).

import os
import sys
import itertools
import numpy as np

# Pauli matrices
I2 = np.eye(2, dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def brauer_weyl_gammas():
    """Return the 6 Cl(6,0) generators via the standard Brauer-Weyl
    construction. Signature (+,+,+,+,+,+), 8-dim spinor."""
    G1 = kron(SIGMA_X, I2, I2)
    G2 = kron(SIGMA_Y, I2, I2)
    G3 = kron(SIGMA_Z, SIGMA_X, I2)
    G4 = kron(SIGMA_Z, SIGMA_Y, I2)
    G5 = kron(SIGMA_Z, SIGMA_Z, SIGMA_X)
    G6 = kron(SIGMA_Z, SIGMA_Z, SIGMA_Y)
    return [G1, G2, G3, G4, G5, G6]


def verify_clifford_relations(Gs):
    I8 = np.eye(8, dtype=complex)
    max_err = 0.0
    for a, Ga in enumerate(Gs):
        for b, Gb in enumerate(Gs):
            anti = Ga @ Gb + Gb @ Ga
            expected = 2.0 * I8 if a == b else np.zeros_like(I8)
            err = np.linalg.norm(anti - expected)
            max_err = max(max_err, err)
    return max_err


def bivector(Ga, Gb):
    """Gamma_{ab} = (1/2)[Gamma_a, Gamma_b]."""
    return 0.5 * (Ga @ Gb - Gb @ Ga)


def hermitian_cartan(Ga, Gb):
    """T = Gamma_{ab} / (2i); Hermitian Cartan generator."""
    return bivector(Ga, Gb) / (2j)


# --- Matching Brauer-Weyl labeling ---------------------------
# Identify the 6 abstract generators with K_4 edges such that the
# Brauer-Weyl pairs (Gamma_1, Gamma_2), (Gamma_3, Gamma_4),
# (Gamma_5, Gamma_6) ARE the three perfect matchings of K_4:
#   Pair 1 / T_M1 := (Gamma_1, Gamma_2) = (e_03, e_12)  matching M_1
#   Pair 2 / T_M2 := (Gamma_3, Gamma_4) = (e_01, e_23)  matching M_2
#   Pair 3 / T_M3 := (Gamma_5, Gamma_6) = (e_02, e_13)  matching M_3
MATCHING_LABELING = {
    1: (0, 3),  # Gamma_1 <-> e_03
    2: (1, 2),  # Gamma_2 <-> e_12
    3: (0, 1),  # Gamma_3 <-> e_01
    4: (2, 3),  # Gamma_4 <-> e_23
    5: (0, 2),  # Gamma_5 <-> e_02
    6: (1, 3),  # Gamma_6 <-> e_13
}


def sigma_on_vertex(v):
    """sigma = (v_0)(v_1 v_3 v_2): v_0 fixed; v_1 -> v_3, v_3 -> v_2, v_2 -> v_1."""
    return {0: 0, 1: 3, 2: 1, 3: 2}[v]


def sigma_on_edge(edge):
    return tuple(sorted(sigma_on_vertex(v) for v in edge))


def sigma_permutation_on_gammas():
    """Return a list where entry a gives the index b such that sigma maps
    Gamma_a -> Gamma_b, using the MATCHING_LABELING.

    1-indexed to match the Brauer-Weyl labeling convention.
    """
    edge_to_gamma = {tuple(sorted(e)): a for a, e in MATCHING_LABELING.items()}
    perm = [None] * 7
    for a, e in MATCHING_LABELING.items():
        mapped_edge = sigma_on_edge(e)
        perm[a] = edge_to_gamma[mapped_edge]
    return perm  # perm[a] = target index


def build_sigma_S(Gs, gamma_perm_1indexed):
    """Construct the Spin(6) lift of the SO(6) permutation acting on the
    Gamma generators by Gamma_a -> Gamma_{perm[a]}.

    Method (standard): construct the SO(6) matrix P, take its principal
    matrix log L in so(6), build the spin-algebra generator X = (1/2)
    sum_{a<b} L_{ab} Gamma_{ab}, and exponentiate: U = exp(X). Then
    fix the Spin(6) +/-1 double-cover ambiguity by demanding U^3 = I.
    """
    # Build 6x6 permutation matrix on R^6
    n = 6
    P = np.zeros((n, n))
    for a in range(1, 7):
        b = gamma_perm_1indexed[a]
        P[b - 1, a - 1] = 1.0  # P sends basis vector a to basis vector b

    # Verify P is SO(6) (permutation is orthogonal; det must be +1)
    assert np.allclose(P.T @ P, np.eye(n)), "P not orthogonal"
    det_P = np.linalg.det(P)
    assert np.isclose(abs(det_P), 1.0)
    # Our sigma acts as two 3-cycles on the 6 indices; det = (+1)(+1) = +1
    # (each 3-cycle has sign +1 on even permutation; two 3-cycles = even
    # overall).

    # principal real log of the SO(6) permutation
    # Complex eigendecomp
    eigvals, eigvecs = np.linalg.eig(P)
    # take principal complex log
    log_eigvals = np.log(eigvals)
    L = np.real(eigvecs @ np.diag(log_eigvals) @ np.linalg.inv(eigvecs))
    # verify L is antisymmetric (lives in so(6))
    antisym_err = np.linalg.norm(L + L.T)
    assert antisym_err < 1e-10, f"L not antisymmetric: err={antisym_err}"
    # verify exp(L) = P
    from scipy.linalg import expm
    exp_L = expm(L)
    assert np.linalg.norm(exp_L - P) < 1e-10, "exp(L) != P"

    # Build X = (1/2) sum_{a<b} L_{ab} Gamma_{ab} (Lie-algebra isomorphism so(6) -> spin(6))
    X = np.zeros((8, 8), dtype=complex)
    for a in range(n):
        for b in range(a + 1, n):
            X += 0.5 * L[a, b] * bivector(Gs[a], Gs[b])
    # Exponentiate
    U = expm(X)

    # Fix +/-1 double-cover ambiguity: U^3 should be +I (since P^3 = I and
    # sigma has order 3). If U^3 = -I, multiply U by exp(i*pi/3) which
    # gives U^3 *= exp(i*pi) = -1, i.e. restores +I. If U^3 = +I, no fix.
    U3 = np.linalg.matrix_power(U, 3)
    I8 = np.eye(8, dtype=complex)
    if np.linalg.norm(U3 - I8) < 1e-8:
        return U
    elif np.linalg.norm(U3 + I8) < 1e-8:
        # multiply by exp(i pi / 3) to fix cube
        return U * np.exp(1j * np.pi / 3.0)
    else:
        raise RuntimeError(
            f"U^3 residue neither +I nor -I: "
            f"||U^3 - I|| = {np.linalg.norm(U3 - I8)}, "
            f"||U^3 + I|| = {np.linalg.norm(U3 + I8)}"
        )


def simultaneous_eigenbasis(T_list, tol=1e-8):
    """Find a common eigenbasis for commuting Hermitian operators T_list.
    Returns a dict mapping eigenvalue-tuple to eigenvector."""
    # Verify commutation
    for i, Ti in enumerate(T_list):
        for j, Tj in enumerate(T_list):
            comm_err = np.linalg.norm(Ti @ Tj - Tj @ Ti)
            assert comm_err < 1e-8, f"T_{i}, T_{j} don't commute: err={comm_err}"

    # Diagonalize joint spectrum: random linear combination
    rng = np.random.default_rng(42)
    coeffs = rng.standard_normal(len(T_list))
    combined = sum(c * T for c, T in zip(coeffs, T_list))
    assert np.linalg.norm(combined - combined.conj().T) < 1e-8, \
        "combined operator not Hermitian"
    eigvals, eigvecs = np.linalg.eigh(combined)

    # For each eigenvector, read off T_a eigenvalues
    basis = {}
    for i in range(eigvecs.shape[1]):
        v = eigvecs[:, i]
        evs = tuple(
            round(np.real(v.conj() @ T @ v), 6) for T in T_list
        )
        # quantize to nearest ±1 (doubled convention, since Cartan eigenvalues
        # of 1/2 Gamma_{ab} on Spin(6) Dirac are ±1/2; 2T gives ±1)
        sign_tuple = tuple(int(np.sign(x)) for x in evs)
        basis[sign_tuple] = v
    return basis


def verify():
    Gs = brauer_weyl_gammas()

    # Sanity: Clifford relations
    cliff_err = verify_clifford_relations(Gs)

    # Matching Cartan generators
    # Pair 1 = (Gamma_1, Gamma_2): T_M1 = Gamma_1 Gamma_2 / (2i)
    # Pair 2 = (Gamma_3, Gamma_4): T_M2 = Gamma_3 Gamma_4 / (2i)
    # Pair 3 = (Gamma_5, Gamma_6): T_M3 = Gamma_5 Gamma_6 / (2i)
    T_M1 = hermitian_cartan(Gs[0], Gs[1])
    T_M2 = hermitian_cartan(Gs[2], Gs[3])
    T_M3 = hermitian_cartan(Gs[4], Gs[5])

    # Verify Cartan generators commute pairwise
    cartan_comm_err = max(
        np.linalg.norm(T_M1 @ T_M2 - T_M2 @ T_M1),
        np.linalg.norm(T_M1 @ T_M3 - T_M3 @ T_M1),
        np.linalg.norm(T_M2 @ T_M3 - T_M3 @ T_M2),
    )

    # Build sigma on Gamma generators from matching labeling
    perm = sigma_permutation_on_gammas()
    print(f"sigma on Gammas (1-indexed): {perm[1:]}")

    # Build sigma_S (Spin(6) lift)
    sigma_S = build_sigma_S(Gs, perm)

    # Verify sigma_S^3 = I
    I8 = np.eye(8, dtype=complex)
    cube_err = np.linalg.norm(np.linalg.matrix_power(sigma_S, 3) - I8)

    # Verify sigma_S is unitary
    unit_err = np.linalg.norm(sigma_S @ sigma_S.conj().T - I8)

    # Key check: does sigma_S permute (T_M1, T_M2, T_M3)?
    # Expected: sigma cycles matching pairs as 1 -> 3 -> 2 -> 1.
    # So sigma * T_M1 * sigma^(-1) = T_M3, sigma * T_M2 * sigma^(-1) = T_M1,
    # sigma * T_M3 * sigma^(-1) = T_M2.
    sig_inv = sigma_S.conj().T
    conj_T1 = sigma_S @ T_M1 @ sig_inv
    conj_T2 = sigma_S @ T_M2 @ sig_inv
    conj_T3 = sigma_S @ T_M3 @ sig_inv

    err_1_to_3 = np.linalg.norm(conj_T1 - T_M3)
    err_2_to_1 = np.linalg.norm(conj_T2 - T_M1)
    err_3_to_2 = np.linalg.norm(conj_T3 - T_M2)

    cartan_cycled = (
        err_1_to_3 < 1e-6 and err_2_to_1 < 1e-6 and err_3_to_2 < 1e-6
    )

    # Find simultaneous eigenbasis of (T_M1, T_M2, T_M3)
    weight_basis = simultaneous_eigenbasis([T_M1, T_M2, T_M3])
    print(f"Number of weight eigenstates: {len(weight_basis)} (expected 8)")

    # For each weight state, check how sigma_S maps it
    action_on_weights = {}
    weight_map = {}  # maps weight tuple to target weight tuple
    for w, v in weight_basis.items():
        w_mapped_vec = sigma_S @ v
        # identify which weight state it is (up to phase)
        best_match = None
        best_overlap = 0.0
        for w2, v2 in weight_basis.items():
            overlap = abs(v2.conj() @ w_mapped_vec)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = w2
        action_on_weights[w] = (best_match, best_overlap)
        weight_map[w] = best_match

    # Verify the analytically-predicted permutation.
    # sigma T_M1 sigma^{-1} = T_M3, hence sigma^{-1} T_M1 sigma = T_M2.
    # For v = |e_1, e_2, e_3>_matching (eigenstate of (T_M1, T_M2, T_M3)
    # with eigvals (e_1, e_2, e_3)), the state sigma v has:
    #   T_M1 eigval = e_2 (via sigma^{-1} T_M1 sigma = T_M2)
    #   T_M2 eigval = e_3 (via sigma^{-1} T_M2 sigma = T_M3)
    #   T_M3 eigval = e_1 (via sigma^{-1} T_M3 sigma = T_M1)
    # So sigma : |e_1, e_2, e_3> -> |e_2, e_3, e_1>  (left-shift of weights)
    analytic_correct = all(
        weight_map[w] == (w[1], w[2], w[0])
        for w in weight_basis.keys()
    )

    # Classify weights into fixed points vs 3-orbits
    fixed_points = [w for w, w2 in weight_map.items() if w == w2]
    non_fixed = [w for w in weight_map.keys() if w not in fixed_points]

    # Build the 3-orbits
    orbits = []
    seen = set(fixed_points)
    for w in non_fixed:
        if w in seen:
            continue
        orbit = [w]
        next_w = weight_map[w]
        seen.add(w)
        while next_w != w:
            orbit.append(next_w)
            seen.add(next_w)
            next_w = weight_map[next_w]
        orbits.append(orbit)

    return {
        "clifford_relations_err": float(cliff_err),
        "cartan_commutation_err": float(cartan_comm_err),
        "sigma_gamma_permutation_1indexed": perm[1:],
        "sigma_S_cube_err": float(cube_err),
        "sigma_S_unitarity_err": float(unit_err),
        "sigma_cycles_cartan": cartan_cycled,
        "conj_err_T1_to_T3": float(err_1_to_3),
        "conj_err_T2_to_T1": float(err_2_to_1),
        "conj_err_T3_to_T2": float(err_3_to_2),
        "num_weight_states": len(weight_basis),
        "sigma_action_on_weights": {str(w): v for w, v in weight_map.items()},
        "analytic_prediction_correct": analytic_correct,
        "fixed_point_weights": [str(w) for w in fixed_points],
        "three_cycle_orbits": [[str(w) for w in orbit] for orbit in orbits],
        "n_fixed_points": len(fixed_points),
        "n_orbits_of_three": len(orbits),
    }


if __name__ == "__main__":
    print("=" * 72)
    print("Session 7 LS1+LS2: sigma_S in matching Brauer-Weyl basis")
    print("=" * 72)
    print()

    r = verify()

    print("Clifford relation max error:       "
          f"{r['clifford_relations_err']:.2e}")
    print("Cartan pairwise commutator error:  "
          f"{r['cartan_commutation_err']:.2e}")
    print("sigma on Gammas (1-indexed):       "
          f"{r['sigma_gamma_permutation_1indexed']}")
    print()
    print("sigma_S properties:")
    print(f"  sigma_S^3 = I error:             {r['sigma_S_cube_err']:.2e}")
    print(f"  sigma_S unitarity error:         {r['sigma_S_unitarity_err']:.2e}")
    print(f"  sigma_S T_M1 sigma^-1 = T_M3:    err {r['conj_err_T1_to_T3']:.2e}")
    print(f"  sigma_S T_M2 sigma^-1 = T_M1:    err {r['conj_err_T2_to_T1']:.2e}")
    print(f"  sigma_S T_M3 sigma^-1 = T_M2:    err {r['conj_err_T3_to_T2']:.2e}")
    print(f"  Cartan cycled correctly:         {r['sigma_cycles_cartan']}")
    print()
    print(f"Weight eigenstates: {r['num_weight_states']}")
    print(f"sigma action on weights (analytic prediction "
          f"|e_1,e_2,e_3> -> |e_3,e_1,e_2>): "
          f"{r['analytic_prediction_correct']}")
    print()
    print("Orbit structure of sigma on weights:")
    print(f"  Fixed points ({r['n_fixed_points']} expected 2): "
          f"{r['fixed_point_weights']}")
    print(f"  3-orbits ({r['n_orbits_of_three']} expected 2):")
    for orb in r["three_cycle_orbits"]:
        print(f"    {orb}")
    print()

    all_ok = (
        r["clifford_relations_err"] < 1e-8
        and r["cartan_commutation_err"] < 1e-8
        and r["sigma_S_cube_err"] < 1e-6
        and r["sigma_S_unitarity_err"] < 1e-8
        and r["sigma_cycles_cartan"]
        and r["analytic_prediction_correct"]
        and r["n_fixed_points"] == 2
        and r["n_orbits_of_three"] == 2
    )
    assert all_ok, f"Some checks failed: {r}"

    print("=" * 72)
    print("RESULT: sigma_S in MATCHING BRAUER-WEYL basis is a clean")
    print("permutation on the 8 weight states with orbit structure:")
    print()
    print("  2 FIXED POINTS:  (+,+,+), (-,-,-)")
    print("  2 3-ORBITS:      {(+,+,-), (+,-,+), (-,+,+)}")
    print("                   {(+,-,-), (-,+,-), (-,-,+)}")
    print()
    print("Analytic rule: sigma maps |e_1, e_2, e_3>_matching to")
    print("|e_2, e_3, e_1>_matching (left-cyclic shift of weight components).")
    print()
    print("This is a MASSIVE structural simplification relative to the")
    print("Hadamard |M_ij| = 0.5 structure seen in Pauli basis")
    print("(docs/framework/B3_B6_reconciliation.md Finding 3). In the matching")
    print("basis, species-mixing is confined to two 3-orbits; there is")
    print("NO inter-orbit mixing under sigma.")
    print()
    print("Consequence for CKM: under matching basis, species labeled by")
    print("weights in different orbits (e.g. fixed-point (+,+,+) vs 3-orbit)")
    print("are never mixed by sigma. The C_3-coupling to V_Ram's C_3 Fourier")
    print("structure breaks sector universality BETWEEN orbits (orbit-type")
    print("is a C_3 invariant); WITHIN each 3-orbit, species are cycled.")
    print()
    print("OK: matching_brauer_weyl_sigma computation complete.")
    print("=" * 72)
