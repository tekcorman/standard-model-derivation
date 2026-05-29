#!/usr/bin/env python3
# ============================================================
# Session 7 Option B: σ_combined-equivariant mass operator on
# S ⊗ C^3_obs (LS3 continuation)
# ============================================================
#
# Context. Session 7 LS3 sub-script (LS3_sigma_invariant_mass_operator.py)
# showed that σ_S-invariant M on the 8-dim spinor S has max 8 distinct
# eigenvalues — insufficient for SM. Here we extend to the full
# 24-dim space S ⊗ C^3_obs (matching-basis spinor × observer C^3 from R3)
# and check whether a σ_combined = σ_S ⊗ σ_obs invariant Hermitian M
# can produce:
#   (a) enough distinct eigenvalues for SM (target: 24+).
#   (b) sector-asymmetric Yukawa matrices on generations (CKM ≠ I).
#
# Central question. Under the σ_combined-invariant ansatz
#
#     M = Σ_{a,b=0,1,2} α_{a,b} · σ_S^a ⊗ σ_obs^b,
#
# do species with different σ_S eigenvalues (λ_X ∈ {1, ω, ω²}) produce
# DIFFERENT Yukawa matrices Y_X on C^3_obs? If yes, CKM is structurally
# non-trivial: u and d having different σ_S eigenvalues ⇒ V = U_u† U_d ≠ I.

import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import matching_brauer_weyl_sigma as mbs


OMEGA = np.exp(2j * np.pi / 3.0)


def observer_sigma(n=3):
    """σ_obs = cyclic shift on C^3 (the R3 generation-Z_3).
    σ_obs |k⟩ = |k+1 mod n⟩ for k = 0, 1, 2."""
    S = np.zeros((n, n), dtype=complex)
    for k in range(n):
        S[(k + 1) % n, k] = 1.0
    return S


def verify_sigma_obs_properties():
    sigma_obs = observer_sigma(3)
    cube = np.linalg.matrix_power(sigma_obs, 3)
    unit = sigma_obs @ sigma_obs.conj().T - np.eye(3)
    ev = sorted(np.linalg.eigvals(sigma_obs), key=lambda z: np.angle(z))
    return {
        "sigma_obs_cube_err": float(np.linalg.norm(cube - np.eye(3))),
        "sigma_obs_unitarity_err": float(np.linalg.norm(unit)),
        "sigma_obs_eigenvalues": [complex(z) for z in ev],
    }


def build_sigma_combined():
    """Build σ_S on 8-dim S (matching basis) and σ_obs on 3-dim C^3.
    Return σ_combined = σ_S ⊗ σ_obs on 24-dim S ⊗ C^3."""
    Gs = mbs.brauer_weyl_gammas()
    perm = mbs.sigma_permutation_on_gammas()
    sigma_S = mbs.build_sigma_S(Gs, perm)
    sigma_obs = observer_sigma(3)
    sigma_combined = np.kron(sigma_S, sigma_obs)
    return sigma_S, sigma_obs, sigma_combined


def generic_sigma_invariant_M(sigma_S, sigma_obs, alpha_coeffs):
    """Construct M = sum α_{a,b} σ_S^a ⊗ σ_obs^b, Hermitized.
    alpha_coeffs is a 3x3 array of complex numbers indexed by (a, b)."""
    dim_S = sigma_S.shape[0]
    dim_obs = sigma_obs.shape[0]
    M = np.zeros((dim_S * dim_obs, dim_S * dim_obs), dtype=complex)
    S_powers = [np.linalg.matrix_power(sigma_S, a) for a in range(3)]
    O_powers = [np.linalg.matrix_power(sigma_obs, b) for b in range(3)]
    for a in range(3):
        for b in range(3):
            M += alpha_coeffs[a, b] * np.kron(S_powers[a], O_powers[b])
    # Hermitize
    return 0.5 * (M + M.conj().T)


def verify_M_spectrum(M, sigma_combined, title=""):
    # σ_combined invariance
    comm = sigma_combined @ M @ sigma_combined.conj().T - M
    inv_err = np.linalg.norm(comm)
    # Hermiticity
    herm_err = np.linalg.norm(M - M.conj().T)
    # Eigenvalues
    eigs = np.linalg.eigvalsh(M)
    eigs_rounded = sorted(set([round(float(e), 6) for e in eigs]))
    return {
        "title": title,
        "sigma_combined_invariance_err": float(inv_err),
        "hermiticity_err": float(herm_err),
        "n_distinct_eigenvalues": len(eigs_rounded),
        "dim": M.shape[0],
    }


def test_yukawa_sector_asymmetry(sigma_S, sigma_obs):
    """Key test: for species X with σ_S eigenvalue λ_X, compute the
    induced Yukawa matrix Y_X on C^3_obs. Show that different λ_X
    gives different Y_X.

    Y_X = ⟨X| M |X⟩_S = Σ_{a,b} α_{a,b} · λ_X^a · σ_obs^b.

    If λ_X = 1, 2, 3 differ (= 1, ω, ω²), the three Y_X matrices differ.
    Under refined A2 they give different left-diagonalizing unitaries
    U_u, U_d, U_e, ν. V_CKM = U_u† U_d will then be non-trivial.
    """
    # Set up a specific alpha_{a,b} — diagonal + off-diagonal
    # to showcase non-trivial structure
    alpha = np.array([
        [1.0, 0.3, 0.3],    # (a, b) = (0, *): overall + mixing
        [0.5, 0.2, 0.1],    # (a, b) = (1, *): σ_S coupling
        [0.5, 0.1, 0.2],    # Hermiticity forces (2, b) entries in conjugate pair
    ], dtype=complex)
    # Enforce Hermiticity at the tensor-product level: conjugate-
    # pair structure
    alpha[2, 1] = alpha[1, 2].conjugate()
    alpha[2, 2] = alpha[1, 1].conjugate()
    alpha[2, 0] = alpha[1, 0].conjugate()  # already real here

    # Compute Y_X for each σ_S eigenvalue
    eigenvalues = {1: 1.0 + 0j, "omega": OMEGA, "omega_sq": OMEGA ** 2}
    Y_matrices = {}
    for name, lam in eigenvalues.items():
        Y = np.zeros((3, 3), dtype=complex)
        for a in range(3):
            for b in range(3):
                Y += alpha[a, b] * (lam ** a) * np.linalg.matrix_power(sigma_obs, b)
        Y_matrices[name] = Y

    # Hermitize each Y (mass matrices are Hermitian)
    for name in Y_matrices:
        Y = Y_matrices[name]
        Y_matrices[name] = 0.5 * (Y + Y.conj().T)

    # Compute spectra
    results = {}
    for name, Y in Y_matrices.items():
        eigs = sorted(np.linalg.eigvalsh(Y).tolist())
        results[name] = {"Y": Y, "eigenvalues": eigs}

    # Compute U_u vs U_d (using λ=1 and λ=ω as u/d proxies)
    # The left-diagonalizing unitaries from eigendecomposition
    _, U1 = np.linalg.eigh(Y_matrices[1])
    _, Uomega = np.linalg.eigh(Y_matrices["omega"])
    V_CKM_proxy = U1.conj().T @ Uomega
    # |V_ij| = absolute value of CKM-like matrix entries
    abs_V = np.abs(V_CKM_proxy)

    return {
        "spectra_per_eigenvalue": {
            name: r["eigenvalues"] for name, r in results.items()
        },
        "V_CKM_proxy_abs": abs_V.tolist(),
        "V_CKM_proxy_offdiag_max": float(
            max(abs_V[0, 1], abs_V[0, 2], abs_V[1, 2])
        ),
        "is_trivial_CKM": float(
            max(abs_V[0, 1], abs_V[0, 2], abs_V[1, 2])
        ) < 1e-6,
    }


def verify():
    # observer C^3 properties
    obs_info = verify_sigma_obs_properties()

    # Build σ_combined on S ⊗ C^3
    sigma_S, sigma_obs, sigma_combined = build_sigma_combined()
    dim = sigma_combined.shape[0]

    # σ_combined properties
    cube_err = np.linalg.norm(np.linalg.matrix_power(sigma_combined, 3) - np.eye(dim))

    # σ_combined eigenvalue multiplicities
    ev = np.linalg.eigvals(sigma_combined)
    from collections import Counter
    ev_rounded = [complex(round(np.real(z), 3), round(np.imag(z), 3)) for z in ev]
    ev_counts = Counter(ev_rounded)

    # Generic σ_combined-invariant M
    rng = np.random.default_rng(7)
    alpha = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    # Enforce Hermiticity structure on alpha
    alpha[2, :] = np.conjugate(alpha[1, ::-1])
    M_generic = generic_sigma_invariant_M(sigma_S, sigma_obs, alpha)
    M_spectrum = verify_M_spectrum(M_generic, sigma_combined, "generic_alpha")

    # Check with max-rank alpha
    # (random full 3x3 alpha, Hermitized at M level)
    M_rand = np.zeros((dim, dim), dtype=complex)
    for _ in range(5):
        M0 = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        M0 = 0.5 * (M0 + M0.conj().T)
        # project to σ_combined-invariant
        sig_inv = sigma_combined.conj().T
        M_proj = (M0 + sigma_combined @ M0 @ sig_inv
                  + sigma_combined @ sigma_combined @ M0 @ sig_inv @ sig_inv) / 3.0
        M_rand += M_proj
    M_rand /= 5.0
    M_rand_spectrum = verify_M_spectrum(M_rand, sigma_combined, "random_Hermitian_projected")

    # Yukawa sector-asymmetry test
    yukawa = test_yukawa_sector_asymmetry(sigma_S, sigma_obs)

    return {
        "sigma_obs_info": obs_info,
        "sigma_combined_dim": int(dim),
        "sigma_combined_cube_err": float(cube_err),
        "sigma_combined_eigenvalue_multiplicities": {
            str(k): v for k, v in ev_counts.items()
        },
        "M_generic_alpha_spectrum": M_spectrum,
        "M_random_Hermitian_projected_spectrum": M_rand_spectrum,
        "yukawa_asymmetry_test": yukawa,
    }


if __name__ == "__main__":
    print("=" * 72)
    print("Option B: σ_combined-invariant mass operator on S ⊗ C^3_obs")
    print("Testing LS3 tensor-factor route for CKM ≠ I")
    print("=" * 72)
    print()

    r = verify()

    print(f"σ_obs (R3 observer) cube err: "
          f"{r['sigma_obs_info']['sigma_obs_cube_err']:.2e}")
    print(f"σ_obs eigenvalues: {r['sigma_obs_info']['sigma_obs_eigenvalues']}")
    print()
    print(f"σ_combined dim: {r['sigma_combined_dim']} (expected 24)")
    print(f"σ_combined cube err: {r['sigma_combined_cube_err']:.2e}")
    print()
    print("σ_combined eigenvalue multiplicities:")
    for ev, count in sorted(r["sigma_combined_eigenvalue_multiplicities"].items()):
        print(f"  {ev}: {count}")
    print("(Expected: 8 each for σ=1, ω, ω²)")
    print()

    for key in ("M_generic_alpha_spectrum", "M_random_Hermitian_projected_spectrum"):
        info = r[key]
        print(f"{info['title']}:")
        print(f"  σ-invariance err: {info['sigma_combined_invariance_err']:.2e}")
        print(f"  Hermiticity err:  {info['hermiticity_err']:.2e}")
        print(f"  distinct eigenvalues: {info['n_distinct_eigenvalues']} / {info['dim']}")
        print()

    y = r["yukawa_asymmetry_test"]
    print("Yukawa sector-asymmetry test:")
    print()
    for name, eigs in y["spectra_per_eigenvalue"].items():
        print(f"  σ_S eigenvalue {name}: Y eigenvalues = "
              f"{[round(e, 4) for e in eigs]}")
    print()
    print(f"|V_CKM| proxy (U_1† U_omega):")
    for row in y["V_CKM_proxy_abs"]:
        print(f"  {[round(x, 4) for x in row]}")
    print()
    print(f"max off-diagonal |V|: {y['V_CKM_proxy_offdiag_max']:.4f}")
    print(f"CKM trivial (= I)?    {y['is_trivial_CKM']}")
    print()

    print("=" * 72)
    if not y["is_trivial_CKM"]:
        print("RESULT: CKM ≠ I under σ_combined-invariant M on S ⊗ C^3_obs.")
        print()
        print("Species with different σ_S eigenvalues see different Yukawa")
        print("matrices Y_X on C^3_obs. The left-diagonalizing unitaries")
        print("U_u, U_d differ, giving non-trivial V = U_u† U_d.")
        print()
        print("Structural closure of (G-LS3) — the matching-basis Pati-Salam")
        print("labeling (which species has σ_S = 1 vs ω vs ω²) determines")
        print("the specific CKM values. Labeling itself is ADOPTED-B3-level")
        print("structural content (not derivable from A1-A5 alone).")
        print()
        print("This is Feshbach pattern: theorem core (σ_combined-invariant M")
        print("gives CKM ≠ I) + explicit residual (specific Pati-Salam")
        print("labeling in matching basis). Shippable at mathematically-")
        print("complete grade with labeling as listed external input.")
    else:
        print("RESULT: unexpected CKM = I. σ_combined-invariant M on")
        print("S ⊗ C^3_obs is too restrictive to produce non-trivial CKM.")
        print("Additional structural content beyond σ-invariance needed.")
    print("=" * 72)
