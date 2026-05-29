#!/usr/bin/env python3
# ============================================================
# Session 7 re-verification: does σ_combined-invariant M on
# S ⊗ C^3_obs genuinely produce CKM ≠ I, or was Option B's
# claim an artifact of degenerate eigenvalues?
# ============================================================
#
# Analytical claim to test:
#
#   If M = Σ α_{a,b} σ_S^a ⊗ σ_obs^b on S ⊗ C^3_obs, and |X⟩ is a
#   σ_S eigenstate with eigenvalue λ_X on S, then the restricted
#   Yukawa matrix Y_X on C^3_obs is
#
#     (Y_X)_{ij} = Σ_a α_{a, (i-j) mod 3} · λ_X^a,
#
#   which is a CIRCULANT matrix on C^3_obs. All circulants on
#   C^3_obs diagonalize via the DFT_3 matrix — regardless of λ_X.
#   Hence U_X = DFT_3 for every species X, and the CKM matrix
#   V = U_u^† U_d = DFT_3^† DFT_3 = I.
#
#   Therefore σ_combined-invariant tensor-product M on S ⊗ C^3_obs
#   gives CKM = I IDENTICALLY, regardless of species σ_S eigenvalues.
#
# This overturns my earlier claim in LS3_tensor_factor_mass_operator.py
# that Option B gives CKM ≠ I. The "non-trivial V = 0.7071" there was
# an artifact of DEGENERATE eigenvalues (Y_u had doubly-degenerate
# eigenvalue 1.4, giving eigenvector ambiguity in the 2-dim subspace
# that np.linalg.eigh resolved arbitrarily).

import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import matching_brauer_weyl_sigma as mbs


OMEGA = np.exp(2j * np.pi / 3.0)


def observer_sigma():
    S = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        S[(k + 1) % 3, k] = 1.0
    return S


def dft3():
    """Unitary DFT_3 matrix: F_{jk} = (1/sqrt(3)) ω^{jk}."""
    F = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        for k in range(3):
            F[j, k] = (OMEGA ** (j * k)) / np.sqrt(3)
    return F


def circulant_Y(alpha, lam_X):
    """Build Y_X = Σ α_{a,b} λ_X^a σ_obs^b directly."""
    sigma_obs = observer_sigma()
    Y = np.zeros((3, 3), dtype=complex)
    for a in range(3):
        for b in range(3):
            Y += alpha[a, b] * (lam_X ** a) * np.linalg.matrix_power(sigma_obs, b)
    return Y


def verify_circulant_property(Y):
    """Circulant means Y_{ij} depends only on (i-j) mod n."""
    n = Y.shape[0]
    circ_err = 0.0
    c = [Y[i, 0] for i in range(n)]  # first column
    for i in range(n):
        for j in range(n):
            expected = c[(i - j) % n]
            circ_err = max(circ_err, abs(Y[i, j] - expected))
    return circ_err


def verify_diagonalization_by_dft(Y):
    """Verify F^† Y F is diagonal."""
    F = dft3()
    diag = F.conj().T @ Y @ F
    off_diag_err = np.linalg.norm(diag - np.diag(np.diag(diag)))
    return off_diag_err, np.diag(diag)


def run_nondegenerate_test():
    # Use truly generic α with no symmetries
    rng = np.random.default_rng(77)
    alpha = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    # Enforce Hermiticity at M level by constructing conjugate structure
    # but we want Y itself non-degenerate and non-Hermitian initially
    # For clarity, just keep alpha generic; focus on circulant property

    results = {}
    for name, lam in [("λ=1", 1.0 + 0j), ("λ=ω", OMEGA), ("λ=ω²", OMEGA ** 2)]:
        Y = circulant_Y(alpha, lam)
        eigs_direct = np.linalg.eigvals(Y)
        circ_err = verify_circulant_property(Y)
        off_diag_err, dft_diag = verify_diagonalization_by_dft(Y)

        # Hermitize to get physical mass-matrix eigenvalues
        Y_herm = 0.5 * (Y + Y.conj().T)
        herm_eigs = np.linalg.eigvalsh(Y_herm)

        results[name] = {
            "Y": Y,
            "is_circulant": circ_err < 1e-10,
            "circulant_err": circ_err,
            "dft_diagonalizes_Y": off_diag_err < 1e-10,
            "dft_off_diag_err": off_diag_err,
            "dft_diagonal_entries": dft_diag,
            "raw_eigenvalues": sorted(eigs_direct, key=lambda z: (np.real(z), np.imag(z))),
            "hermitized_eigenvalues": sorted(herm_eigs),
        }

    # Now critical check: in the standard (non-Fourier) basis, does
    # np.linalg.eigh give different eigenvector bases for different
    # species? (This is what was misinterpreted as "non-trivial CKM.")
    # In truth, circulants all share the DFT eigenbasis, so U_X = DFT
    # for all X up to column permutations (which represent eigenvalue
    # ordering, not physical mixing).

    Y1_herm = 0.5 * (results["λ=1"]["Y"] + results["λ=1"]["Y"].conj().T)
    Yomega_herm = 0.5 * (results["λ=ω"]["Y"] + results["λ=ω"]["Y"].conj().T)
    ev1, U1 = np.linalg.eigh(Y1_herm)
    ev_omega, Uomega = np.linalg.eigh(Yomega_herm)
    V_proxy = U1.conj().T @ Uomega
    abs_V = np.abs(V_proxy)

    # Compute whether V = (eigenvalue-ordering permutation) × (diagonal phases)
    # i.e., is V a "signed permutation matrix"?
    # For each row, find column of max; check that mags are pure permutation.
    n = abs_V.shape[0]
    row_max_col = [int(np.argmax(abs_V[i])) for i in range(n)]
    col_max_row = [int(np.argmax(abs_V[:, j])) for j in range(n)]
    permutation_like = (
        (set(row_max_col) == {0, 1, 2})
        and (set(col_max_row) == {0, 1, 2})
        and all(abs_V[i, row_max_col[i]] > 0.99 for i in range(n))
    )

    return {
        "per_species": results,
        "V_proxy_magnitudes": abs_V.tolist(),
        "V_is_permutation_like": permutation_like,
    }


def run_degenerate_test():
    """Replicate the Option B α choice that produced degenerate eigenvalues."""
    alpha = np.array([
        [1.0, 0.3, 0.3],
        [0.5, 0.2, 0.1],
        [0.5, 0.1, 0.2],
    ], dtype=complex)
    alpha[2, 1] = alpha[1, 2].conjugate()
    alpha[2, 2] = alpha[1, 1].conjugate()
    alpha[2, 0] = alpha[1, 0].conjugate()

    Y1 = circulant_Y(alpha, 1.0 + 0j)
    Y1_herm = 0.5 * (Y1 + Y1.conj().T)
    evs = np.linalg.eigvalsh(Y1_herm)
    degeneracies = [(ev, sum(1 for e in evs if abs(e - ev) < 1e-6)) for ev in sorted(set([round(e, 4) for e in evs]))]
    return {
        "eigenvalues": sorted(evs),
        "has_degeneracy": any(count > 1 for ev, count in degeneracies),
    }


if __name__ == "__main__":
    print("=" * 72)
    print("Re-verification of LS3 Option B: is CKM truly nontrivial?")
    print("=" * 72)
    print()

    print("Test 1: generic (non-degenerate) α")
    print("-" * 72)
    r = run_nondegenerate_test()
    for name, d in r["per_species"].items():
        print(f"\n  Species {name}:")
        print(f"    Y is circulant: {d['is_circulant']} (err {d['circulant_err']:.2e})")
        print(f"    DFT diagonalizes Y: {d['dft_diagonalizes_Y']} (err {d['dft_off_diag_err']:.2e})")
        print(f"    Hermitized Y eigenvalues: {[round(e, 3) for e in d['hermitized_eigenvalues']]}")
    print()
    print(f"  V_proxy magnitudes:")
    for row in r["V_proxy_magnitudes"]:
        print(f"    {[round(x, 4) for x in row]}")
    print(f"  V is permutation-like (only eigenvalue ordering): "
          f"{r['V_is_permutation_like']}")

    print()
    print("Test 2: Option B's original α (contains degeneracy)")
    print("-" * 72)
    r2 = run_degenerate_test()
    print(f"  Y_1 (λ=1) Hermitized eigenvalues: {[round(e, 3) for e in r2['eigenvalues']]}")
    print(f"  Has degeneracy: {r2['has_degeneracy']}")

    print()
    print("=" * 72)
    print("CONCLUSION:")
    print()
    print("All Y_X matrices are CIRCULANT in the standard C^3_obs basis.")
    print("All circulants diagonalize via the same DFT_3 matrix.")
    print("Therefore U_X = DFT_3 for every species X (up to eigenvalue-")
    print("ordering permutations, which have no physical meaning).")
    print()
    print("V_CKM = U_u^† U_d = DFT_3^† DFT_3 = I (identity) up to an")
    print("eigenvalue-ordering permutation matrix — NOT a physical mixing.")
    print()
    print("Option B's earlier 'non-trivial V = 0.7071' was an ARTIFACT of")
    print("degenerate eigenvalues in Y_u causing eigh() to pick an")
    print("arbitrary orthonormal basis in the 2-dim degenerate subspace.")
    print()
    print("CORRECTION: σ_combined-invariant tensor-product M on S ⊗ C^3_obs")
    print("gives CKM = I IDENTICALLY. The structural unblock claimed in")
    print("LS3_tensor_factor_mass_operator.py is INCORRECT.")
    print()
    print("TRUE blocker for CKM ≠ I: mass operator M must break σ_obs")
    print("invariance (i.e., not be a tensor-product form α_{a,b} σ_S^a ⊗")
    print("σ_obs^b, but rather include entangled cross-terms between S")
    print("and C^3_obs that do not factorize). The structural route to")
    print("such an M is the OPEN gap (G-LS3').")
    print()
    print("OK: LS3 Option B re-verification complete (negative result).")
    print("=" * 72)
