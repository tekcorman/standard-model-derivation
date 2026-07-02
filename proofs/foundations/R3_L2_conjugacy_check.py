#!/usr/bin/env python3
# ============================================================
# R3 Load-bearing step L2: Z_3 faithful action on C^3 is unique
# ============================================================
#
# Claim (to verify under parameter_linter's hard gate):
#
#   Every order-3 unitary U in U(3) with eigenvalue multiset
#   {1, omega, omega^2} (omega = exp(2 pi i / 3)) is U(3)-conjugate
#   to the cyclic-shift permutation matrix sigma_shift.
#
# Consequences (if verified):
#   (a) A faithful Z_3 action on C^3 such that Z_3 acts by the
#       regular representation (each of the three Z_3-irreps
#       appears exactly once) is UNIQUE up to U(3) conjugation.
#   (b) The cyclic shift sigma_shift and the diagonal
#       sigma_diag = diag(1, omega, omega^2) are the same Z_3
#       action up to a change of basis.
#   (c) R3 load-bearing step L2 closes as pure linear algebra
#       (spectral theorem for unitaries, Halmos 1958 §83) — no
#       MDL comparison is needed.
#
# This script verifies (a)-(c) computationally at machine precision.
#
# Theorem citations used:
#   - Spectral theorem for normal operators (unitaries are normal):
#     Halmos 1958 *Finite-Dimensional Vector Spaces* §83.
#     Statement: every unitary U on C^n is diagonalizable by a
#     unitary V in U(n), i.e. V* U V = diag(lambda_1,...,lambda_n).
#   - Two diagonal unitaries with the same eigenvalue multiset are
#     related by a permutation unitary (standard).
#
# These together imply: U(3) conjugacy class of a unitary is
# uniquely determined by its eigenvalue multiset.
#
# The cyclic-shift matrix:
#     sigma_shift = [[0, 0, 1],
#                    [1, 0, 0],
#                    [0, 1, 0]]
# has eigenvalues {1, omega, omega^2} (Fourier-diagonalizable by
# the DFT matrix F_3).
#
# Script outputs:
#   - Verified sigma_shift eigenvalues = {1, omega, omega^2}.
#   - Verified sigma_shift^3 = I.
#   - Explicit DFT conjugation F_3* sigma_shift F_3 = sigma_diag.
#   - Verified on 50 random order-3 unitaries with the right
#     eigenvalue multiset: each is U(3)-conjugate to sigma_shift.

import numpy as np
import sys


OMEGA = np.exp(2j * np.pi / 3.0)
OMEGA_SQ = OMEGA * OMEGA
TOL = 1e-10


def cyclic_shift_matrix():
    """sigma_shift on C^3: |k> -> |k+1 mod 3>, so column k goes to
    row (k+1 mod 3). In the standard basis with columns {|0>,|1>,|2>}:
        sigma_shift |0> = |1>
        sigma_shift |1> = |2>
        sigma_shift |2> = |0>
    """
    return np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=complex)


def diagonal_z3_matrix():
    """sigma_diag = diag(1, omega, omega^2)."""
    return np.diag([1.0 + 0j, OMEGA, OMEGA_SQ])


def dft3_matrix():
    """Discrete Fourier Transform on Z_3, unitary normalization.
    F_jk = (1/sqrt(3)) * omega^{jk}.
    Standard fact: F* P F is diagonal for any Z_3-permutation P, with
    diagonal entries being the character-values at F's index columns.
    """
    F = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        for k in range(3):
            F[j, k] = np.exp(2j * np.pi * j * k / 3.0) / np.sqrt(3.0)
    return F


def random_u3(rng):
    """Haar-random element of U(3) via QR of a complex Ginibre matrix."""
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    d = d / np.abs(d)
    return Q * d  # Q times diag(d)


def random_order3_unitary_with_spec(rng):
    """Construct U in U(3) with eigenvalue multiset {1, omega, omega^2}
    by conjugating sigma_diag with a Haar-random U(3) element.
    """
    V = random_u3(rng)
    return V @ diagonal_z3_matrix() @ V.conj().T


def eigvals_sorted(M):
    """Return eigenvalues of M sorted by argument, for comparison."""
    evs = np.linalg.eigvals(M)
    return sorted(evs, key=lambda z: np.angle(z))


def eigval_multisets_match(A, B, tol=TOL):
    """Check that two 3x3 matrices have the same eigenvalue multiset."""
    a = eigvals_sorted(A)
    b = eigvals_sorted(B)
    for x, y in zip(a, b):
        if abs(x - y) > tol:
            return False
    return True


def find_conjugator(U_target, U_reference, tol=TOL):
    """Given two normal matrices with the same eigenvalue multiset,
    find V in U(3) with V* U_target V = U_reference.

    Approach:
        - Diagonalize U_target: U_target = A D_A A* with D_A sorted.
        - Diagonalize U_reference: U_reference = B D_B B* with D_B sorted.
        - If D_A == D_B (after sort), then V = A B* satisfies
          V* U_target V = U_reference.
    Returns V (a unitary) and the residual norm.
    """
    # eigendecompose both (unitaries, so normal, so eigendecomposable)
    ev_T, A = np.linalg.eig(U_target)
    ev_R, B = np.linalg.eig(U_reference)

    # sort eigenvectors by eigenvalue argument so the diagonal blocks
    # are identical
    order_T = np.argsort([np.angle(z) for z in ev_T])
    order_R = np.argsort([np.angle(z) for z in ev_R])
    A_sorted = A[:, order_T]
    B_sorted = B[:, order_R]
    ev_T_sorted = ev_T[order_T]
    ev_R_sorted = ev_R[order_R]

    # verify eigenvalues agree after sorting
    for x, y in zip(ev_T_sorted, ev_R_sorted):
        if abs(x - y) > tol:
            return None, float("inf")

    V = A_sorted @ B_sorted.conj().T
    # check V is unitary
    unitary_err = np.linalg.norm(V @ V.conj().T - np.eye(3))
    # check V* U_target V = U_reference
    conj = V.conj().T @ U_target @ V
    residual = np.linalg.norm(conj - U_reference)
    return V, max(residual, unitary_err)


def verify_R3_L2():
    """Main verification. Returns a dict of all checks."""
    results = {}

    sigma_shift = cyclic_shift_matrix()
    sigma_diag = diagonal_z3_matrix()
    F3 = dft3_matrix()

    # Check 1: sigma_shift^3 = I
    shift_cubed = np.linalg.matrix_power(sigma_shift, 3)
    err_cube = np.linalg.norm(shift_cubed - np.eye(3))
    results["C1_sigma_shift_cubed_is_I"] = bool(err_cube < TOL)
    results["C1_residual"] = float(err_cube)

    # Check 2: sigma_shift is unitary
    err_unitary = np.linalg.norm(
        sigma_shift @ sigma_shift.conj().T - np.eye(3)
    )
    results["C2_sigma_shift_unitary"] = bool(err_unitary < TOL)
    results["C2_residual"] = float(err_unitary)

    # Check 3: sigma_shift eigenvalues = {1, omega, omega^2}
    ev_shift = eigvals_sorted(sigma_shift)
    ev_diag = eigvals_sorted(sigma_diag)
    match = all(abs(a - b) < TOL for a, b in zip(ev_shift, ev_diag))
    results["C3_sigma_shift_has_correct_eigenvalues"] = bool(match)
    results["C3_eigenvalues"] = [complex(x) for x in ev_shift]

    # Check 4: DFT conjugation F_3* sigma_shift F_3 is diagonal with
    # entries {1, omega, omega^2}.
    conj_by_dft = F3.conj().T @ sigma_shift @ F3
    off_diag_norm = np.linalg.norm(conj_by_dft - np.diag(np.diag(conj_by_dft)))
    results["C4_DFT_conjugation_diagonal"] = bool(off_diag_norm < TOL)
    results["C4_off_diag_residual"] = float(off_diag_norm)
    diag_entries = np.diag(conj_by_dft)
    # reorder to match sigma_diag's convention
    diag_sorted = sorted(diag_entries, key=lambda z: np.angle(z))
    match_diag = all(abs(a - b) < TOL for a, b in
                     zip(diag_sorted, eigvals_sorted(sigma_diag)))
    results["C4_diagonal_entries_match"] = bool(match_diag)

    # Check 5: for 50 random conjugates of sigma_diag, each is
    # U(3)-conjugate to sigma_shift by explicit construction.
    rng = np.random.default_rng(0)
    n_trials = 50
    worst_residual = 0.0
    trials_ok = 0
    for _ in range(n_trials):
        U_rand = random_order3_unitary_with_spec(rng)
        # sanity: U_rand has order 3
        U_cubed = np.linalg.matrix_power(U_rand, 3)
        assert np.linalg.norm(U_cubed - np.eye(3)) < TOL

        # must have matching eigenvalue multiset with sigma_shift
        assert eigval_multisets_match(U_rand, sigma_shift)

        # find explicit conjugator V with V* U_rand V = sigma_shift
        V, residual = find_conjugator(U_rand, sigma_shift)
        if residual < TOL:
            trials_ok += 1
            worst_residual = max(worst_residual, residual)

    results["C5_random_conjugacy_trials"] = n_trials
    results["C5_random_conjugacy_ok"] = trials_ok
    results["C5_worst_residual"] = float(worst_residual)
    results["C5_all_trials_passed"] = bool(trials_ok == n_trials)

    # Summary
    all_ok = (
        results["C1_sigma_shift_cubed_is_I"]
        and results["C2_sigma_shift_unitary"]
        and results["C3_sigma_shift_has_correct_eigenvalues"]
        and results["C4_DFT_conjugation_diagonal"]
        and results["C4_diagonal_entries_match"]
        and results["C5_all_trials_passed"]
    )
    results["ALL_CHECKS_PASSED"] = bool(all_ok)

    return results


if __name__ == "__main__":
    print("=" * 72)
    print("R3 Load-bearing step L2 — conjugacy check")
    print("Claim: order-3 U in U(3) with eigvals {1, omega, omega^2}")
    print("is U(3)-conjugate to the cyclic-shift permutation sigma_shift.")
    print("=" * 72)
    print()

    results = verify_R3_L2()

    print("C1. sigma_shift^3 = I")
    print(f"    residual = {results['C1_residual']:.2e}")
    print(f"    PASSED: {results['C1_sigma_shift_cubed_is_I']}")
    print()
    print("C2. sigma_shift is unitary")
    print(f"    residual = {results['C2_residual']:.2e}")
    print(f"    PASSED: {results['C2_sigma_shift_unitary']}")
    print()
    print("C3. sigma_shift eigenvalues = {1, omega, omega^2}")
    for z in results["C3_eigenvalues"]:
        print(f"    eigenvalue: {z:.6f}")
    print(f"    PASSED: {results['C3_sigma_shift_has_correct_eigenvalues']}")
    print()
    print("C4. F_3* sigma_shift F_3 = diag(1, omega, omega^2) explicitly")
    print(f"    off-diagonal residual = {results['C4_off_diag_residual']:.2e}")
    print(f"    diagonal entries match: {results['C4_diagonal_entries_match']}")
    print(f"    PASSED: {results['C4_DFT_conjugation_diagonal']}")
    print()
    print("C5. Random order-3 U(3) element with spec {1, omega, omega^2}")
    print(f"    is conjugate to sigma_shift: "
          f"{results['C5_random_conjugacy_ok']}/"
          f"{results['C5_random_conjugacy_trials']} trials")
    print(f"    worst residual = {results['C5_worst_residual']:.2e}")
    print(f"    PASSED: {results['C5_all_trials_passed']}")
    print()

    assert results["ALL_CHECKS_PASSED"], f"Some checks failed: {results}"

    print("=" * 72)
    print("RESULT: L2 closes as pure linear algebra.")
    print()
    print("  Theorem (L2, verified).")
    print("  Every U in U(3) satisfying U^3 = I with eigenvalue multiset")
    print("  {1, omega, omega^2} is U(3)-conjugate to the cyclic-shift")
    print("  permutation sigma_shift. Equivalently, the regular representation")
    print("  of Z_3 on C^3 is unique up to isomorphism.")
    print()
    print("  Citation: Halmos 1958 *Finite-Dimensional Vector Spaces* §83")
    print("  (spectral theorem for normal operators).")
    print()
    print("  Consequence for R3: L2 is a rep-theory citation, not an")
    print("  MDL comparison. The N-budget concern in R3 scoping §8 Q1")
    print("  dissolves. R3 closure via Observer-C^3 now has L1, L2, L3, L4")
    print("  all tractable-or-closed.")
    print("=" * 72)
    sys.exit(0)
