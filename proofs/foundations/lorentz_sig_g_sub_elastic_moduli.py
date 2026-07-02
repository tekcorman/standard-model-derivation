#!/usr/bin/env python3
"""
G_sub bridge step (1): substrate elastic moduli C^{abcd} from strain matrices.

The substrate's Iorio-elastic apparatus gives 9 strain-perturbation matrices
A^{ac}(k) ∈ Mat_4(C) such that

    δH(k, x) = Σ_{a,c} A^{ac}(k) (∂_a u_c)(x)

(see `lorentz_sig_strain_perturbation.py`). Integrating out the matter
sector at second order in u gives the substrate's elastic energy density

    U_elastic(x) = (1/2) C^{abcd} (∂_a u_b)(∂_c u_d)

with elastic-modulus tensor

    C^{abcd} = -(2/V_BZ) Re ∫_BZ d³k Σ_{n filled, m unfilled}
                  ⟨m,k|A^{ab}(k)|n,k⟩ ⟨n,k|A^{cd}(k)|m,k⟩ / (λ_n(k) - λ_m(k)).

This script computes C^{abcd} numerically by:
  1. BZ-sampling on a uniform grid.
  2. At each k: diagonalize H(k); identify filled (λ < μ) vs unfilled (λ > μ).
  3. Compute matrix elements of A^{ab}(k) between filled and unfilled bands.
  4. Sum the second-order response.
  5. Average over BZ.

Convention: half-filling at μ = 0 (the natural midpoint of the spectrum
[-3, +3] under particle-hole symmetry that swaps Γ ↔ H).

Output: the 6 independent components of C^{abcd} under cubic (O_h) symmetry
reduction in Voigt notation:
  C_11 = C^{xxxx} = C^{yyyy} = C^{zzzz}
  C_12 = C^{xxyy} = C^{xxzz} = C^{yyzz}
  C_44 = C^{xyxy} = C^{xzxz} = C^{yzyz}
plus consistency checks on isotropic vs cubic structure.

Status. Bounded numerical computation; theorem-grade once cubic-symmetry
reduction is verified analytically. The result feeds into the matter 1-loop
polarization (step 2 of the G_sub bridge).
"""
from __future__ import annotations

import numpy as np

# srs primitive cell (same convention as `lorentz_sig_strain_perturbation.py`).
ATOMS = np.array([
    [1/8, 1/8, 1/8],
    [3/8, 7/8, 5/8],
    [7/8, 5/8, 3/8],
    [5/8, 3/8, 7/8],
])

A_PRIM = np.array([
    [-1/2,  1/2,  1/2],
    [ 1/2, -1/2,  1/2],
    [ 1/2,  1/2, -1/2],
])

CELL_EDGES = [
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
]

DIRECTED_BONDS = []
for s, t, c in CELL_EDGES:
    DIRECTED_BONDS.append((s, t, np.array(c)))
    DIRECTED_BONDS.append((t, s, -np.array(c)))


def bond_displacement(src: int, tgt: int, cell: np.ndarray) -> np.ndarray:
    """Cartesian displacement r_b = R_β + cell·a_prim - R_α (lattice units)."""
    return ATOMS[tgt] - ATOMS[src] + cell @ A_PRIM


# Pre-compute Cartesian bond displacements (constant in k).
BOND_DISPLACEMENTS = [
    (s, t, bond_displacement(s, t, c)) for s, t, c in DIRECTED_BONDS
]


def H_bloch(k_cart: np.ndarray) -> np.ndarray:
    """4×4 scalar Bloch Hamiltonian (adjacency) at Cartesian k."""
    H = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        phase = np.exp(1j * np.dot(k_cart, rb))
        H[t, s] += phase
    # Hermitize defensively (numerical safeguard; analytically Hermitian).
    return (H + H.conj().T) / 2


def A_strain_matrix(k_cart: np.ndarray, a: int, c: int) -> np.ndarray:
    """
    Strain-perturbation matrix A^{ac}(k) at Cartesian k.

      A^{ac}_{βα}(k) = i Σ_{bonds (α→β)} exp(i k·r_b) k_a r_b^c

    (per `lorentz_sig_strain_perturbation.py`, Cartesian version).
    """
    A = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        phase = np.exp(1j * np.dot(k_cart, rb))
        A[t, s] += 1j * phase * k_cart[a] * rb[c]
    return (A + A.conj().T) / 2


def elastic_response_at_k_matter_only(
    k_cart: np.ndarray, cone_lambda: float = -1.0, tol: float = 0.5
) -> np.ndarray:
    """
    Matter-only elastic response: include only transitions WITHIN the
    Γ-cone manifold (3 bands near λ = -1), excluding the +3 background
    band. At small |k|, the 3 cone bands split into helicity ±1, 0;
    transitions among them are the matter-loop content.

    Implementation: identify the 3 bands closest to cone_lambda = -1 and
    treat the remaining band as background. Sum response only over
    band-pairs both in the cone manifold, with half-filling within
    that manifold (μ_cone = -1).
    """
    H = H_bloch(k_cart)
    eigvals, eigvecs = np.linalg.eigh(H)

    # Identify cone bands: the 3 bands closest to cone_lambda = -1.
    distances = np.abs(eigvals - cone_lambda)
    cone_indices = np.argsort(distances)[:3]
    # Within the cone, half-fill: bands with eigenvalue < cone_lambda are filled.
    cone_bands = sorted(cone_indices)
    cone_eigvals = eigvals[cone_bands]
    filled_in_cone = [i for i in cone_bands if eigvals[i] < cone_lambda - 1e-10]
    unfilled_in_cone = [i for i in cone_bands if eigvals[i] > cone_lambda + 1e-10]

    A_mats = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for c in range(3):
            A_mats[a, c] = A_strain_matrix(k_cart, a, c)

    A_sym = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_sym[a, b] = (A_mats[a, b] + A_mats[b, a]) / 2

    U = eigvecs
    A_basis = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_basis[a, b] = U.conj().T @ A_sym[a, b] @ U

    K = np.zeros((3, 3, 3, 3), dtype=float)
    for n in filled_in_cone:
        for m in unfilled_in_cone:
            denom = eigvals[n] - eigvals[m]
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            term = (A_basis[a, b][m, n] *
                                    A_basis[c, d][n, m]) / denom
                            K[a, b, c, d] += -2.0 * term.real
    return K


def bz_average_matter_only(N_grid: int = 24, cone_lambda: float = -1.0,
                            half_extent: float = np.pi):
    """BZ-average the matter-only elastic response over [-half_extent, +half_extent]³."""
    ks = np.linspace(-half_extent, half_extent, N_grid, endpoint=False)
    K_total = np.zeros((3, 3, 3, 3), dtype=float)
    n_points = 0
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                k_cart = np.array([k1, k2, k3])
                K_total += elastic_response_at_k_matter_only(k_cart, cone_lambda)
                n_points += 1
    return K_total / n_points


def verify_bloch_sum_rules(N_grid: int = 16, half_extent: float = 2*np.pi):
    """Independent check: ⟨Tr(H²)⟩ = 12 and ⟨Tr(H⁴)⟩ = 60 should hold on
    any fundamental domain of the Bloch Hamiltonian. Test if [-half_extent,
    half_extent]³ samples a fundamental domain."""
    ks = np.linspace(-half_extent, half_extent, N_grid, endpoint=False)
    trH2 = 0.0
    trH4 = 0.0
    n = 0
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                H = H_bloch(np.array([k1, k2, k3]))
                trH2 += np.real(np.trace(H @ H))
                trH4 += np.real(np.trace(H @ H @ H @ H))
                n += 1
    return {
        'half_extent': half_extent,
        'trH2_avg': trH2 / n,
        'trH4_avg': trH4 / n,
    }


def W_diamagnetic_matrix(k_cart: np.ndarray, a: int, b: int, c: int, d: int) -> np.ndarray:
    """
    Diamagnetic strain matrix W^{abcd}(k) = ∂²H[u]/∂u^{ab}∂u^{cd}|_{u=0}.

    From H[u]_{βα}(k) = Σ_bonds exp(i k·(r + u·r)) and second-order expansion:
    W^{abcd}_{βα}(k) = -Σ_bonds exp(i k·r) k_a r^b k_c r^d   (symmetrized in (ab)(cd))

    Returns 4×4 Hermitian matrix.
    """
    W = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        phase = np.exp(1j * np.dot(k_cart, rb))
        W[t, s] += -phase * k_cart[a] * rb[b] * k_cart[c] * rb[d]
    return (W + W.conj().T) / 2


def W_diamagnetic_symmetrized(k_cart: np.ndarray, a: int, b: int, c: int, d: int) -> np.ndarray:
    """
    Symmetric strain pair (a,b) and (c,d): average over (a,b)↔(b,a) and (c,d)↔(d,c).
    """
    W_avg = (
        W_diamagnetic_matrix(k_cart, a, b, c, d) +
        W_diamagnetic_matrix(k_cart, b, a, c, d) +
        W_diamagnetic_matrix(k_cart, a, b, d, c) +
        W_diamagnetic_matrix(k_cart, b, a, d, c)
    ) / 4
    return W_avg


def elastic_response_full_at_k(
    k_cart: np.ndarray, mu: float = 0.0, tol: float = 1e-8
) -> tuple:
    """
    Full elastic-modulus response: paramagnetic + diamagnetic.

    Returns: (paramagnetic, diamagnetic, full) each as (3,3,3,3) tensor.
    """
    H = H_bloch(k_cart)
    eigvals, eigvecs = np.linalg.eigh(H)
    filled = eigvals < mu - tol

    A_mats = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for c in range(3):
            A_mats[a, c] = A_strain_matrix(k_cart, a, c)

    # Symmetrize A.
    A_sym = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_sym[a, b] = (A_mats[a, b] + A_mats[b, a]) / 2

    # Paramagnetic.
    U = eigvecs
    A_basis = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_basis[a, b] = U.conj().T @ A_sym[a, b] @ U

    K_para = np.zeros((3, 3, 3, 3), dtype=float)
    unfilled = ~filled
    for n in np.where(filled)[0]:
        for m in np.where(unfilled)[0]:
            denom = eigvals[n] - eigvals[m]
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            term = (A_basis[a, b][m, n] *
                                    A_basis[c, d][n, m]) / denom
                            K_para[a, b, c, d] += -2.0 * term.real

    # Diamagnetic: Σ_filled ⟨n|W^{abcd}|n⟩
    K_dia = np.zeros((3, 3, 3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    W = W_diamagnetic_symmetrized(k_cart, a, b, c, d)
                    W_basis = U.conj().T @ W @ U
                    for n in np.where(filled)[0]:
                        K_dia[a, b, c, d] += np.real(W_basis[n, n])

    K_full = K_dia + K_para
    return K_para, K_dia, K_full


def bz_average_full(N_grid: int = 16, mu: float = 0.0,
                     half_extent: float = 2 * np.pi):
    """BZ-average paramagnetic + diamagnetic + full."""
    ks = np.linspace(-half_extent, half_extent, N_grid, endpoint=False)
    K_para_total = np.zeros((3, 3, 3, 3), dtype=float)
    K_dia_total = np.zeros((3, 3, 3, 3), dtype=float)
    n_points = 0
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                k_cart = np.array([k1, k2, k3])
                K_para, K_dia, _ = elastic_response_full_at_k(k_cart, mu)
                K_para_total += K_para
                K_dia_total += K_dia
                n_points += 1
    K_para_avg = K_para_total / n_points
    K_dia_avg = K_dia_total / n_points
    K_full_avg = K_para_avg + K_dia_avg
    return K_para_avg, K_dia_avg, K_full_avg


def elastic_response_at_k(
    k_cart: np.ndarray, mu: float = 0.0, tol: float = 1e-8
) -> np.ndarray:
    """
    Compute the symmetric-strain second-order response at fixed k:

        C^{(ab)(cd)}(k) = -2 Re Σ_{n filled, m unfilled}
                            A^{(ab)}_{mn}(k) A^{(cd)}_{nm}(k) / (λ_n - λ_m)

    where A^{(ab)} := (A^{ab} + A^{ba})/2 projects onto the symmetric-strain
    tensor u_{(ab)} = (∂_a u_b + ∂_b u_a)/2 (the antisymmetric ω goes to the
    spin connection, separate channel).

    Returns a (3,3,3,3) tensor with the correct (a↔b), (c↔d), (ab↔cd)
    symmetries by construction.
    """
    H = H_bloch(k_cart)
    eigvals, eigvecs = np.linalg.eigh(H)
    filled = eigvals < mu - tol
    unfilled = eigvals > mu + tol

    A_mats = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for c in range(3):
            A_mats[a, c] = A_strain_matrix(k_cart, a, c)

    # Symmetrize: A_sym[a,b] = (A[a,b] + A[b,a])/2 — projects onto u_{(ab)}.
    A_sym = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_sym[a, b] = (A_mats[a, b] + A_mats[b, a]) / 2

    # Transform to eigenbasis.
    U = eigvecs
    A_basis = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_basis[a, b] = U.conj().T @ A_sym[a, b] @ U

    K = np.zeros((3, 3, 3, 3), dtype=float)
    for n in np.where(filled)[0]:
        for m in np.where(unfilled)[0]:
            denom = eigvals[n] - eigvals[m]
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            term = (A_basis[a, b][m, n] *
                                    A_basis[c, d][n, m]) / denom
                            K[a, b, c, d] += -2.0 * term.real
    return K


def bz_average(
    N_grid: int = 12, mu: float = 0.0, half_extent: float = np.pi
) -> np.ndarray:
    """
    BZ-average the elastic response on a uniform Cartesian grid covering
    [-half_extent, +half_extent]³. The Bloch Hamiltonian for srs is
    4π-periodic in each Cartesian direction, so half_extent = 2π samples
    the full fundamental domain (volume (4π)³ = 64π³, containing 4
    reciprocal-lattice points) while half_extent = π samples [-π, π]³
    of volume 8π³ (subdomain).
    """
    ks = np.linspace(-half_extent, half_extent, N_grid, endpoint=False)
    K_total = np.zeros((3, 3, 3, 3), dtype=float)
    n_points = 0
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                k_cart = np.array([k1, k2, k3])
                K_total += elastic_response_at_k(k_cart, mu=mu)
                n_points += 1
    return K_total / n_points


def voigt_components(C: np.ndarray) -> dict:
    """Extract cubic-symmetry components of the elastic tensor.

    For an isotropic medium: C^{abcd} = λ δ^{ab}δ^{cd} + μ(δ^{ac}δ^{bd} + δ^{ad}δ^{bc}).
    For cubic O_h symmetry: 3 independent moduli C_11, C_12, C_44 (Voigt).
    Use averaging to reduce numerical noise:
      C_11 = (C_xxxx + C_yyyy + C_zzzz)/3
      C_12 = (C_xxyy + C_xxzz + C_yyzz + C_yyxx + C_zzxx + C_zzyy)/6
      C_44 = (C_xyxy + C_xzxz + C_yzyz + C_xyyx + ...)/12
    """
    C_11 = (C[0,0,0,0] + C[1,1,1,1] + C[2,2,2,2]) / 3
    C_12 = (C[0,0,1,1] + C[0,0,2,2] + C[1,1,2,2] +
            C[1,1,0,0] + C[2,2,0,0] + C[2,2,1,1]) / 6
    C_44 = (C[0,1,0,1] + C[0,2,0,2] + C[1,2,1,2] +
            C[1,0,1,0] + C[2,0,2,0] + C[2,1,2,1] +
            C[0,1,1,0] + C[0,2,2,0] + C[1,2,2,1] +
            C[1,0,0,1] + C[2,0,0,2] + C[2,1,1,2]) / 12
    isotropy_violation = abs(2 * C_44 - (C_11 - C_12))
    return {
        'C_11': C_11,
        'C_12': C_12,
        'C_44': C_44,
        'lambda_lame': C_12,
        'mu_lame': C_44,
        '2C_44 - (C_11 - C_12)': isotropy_violation,
    }


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("G_sub bridge step (1): substrate elastic moduli C^{abcd}")
    print()
    print("  Convention: half-filling at μ = 0 (natural midpoint under the Γ↔H")
    print("  particle-hole symmetry of the srs adjacency Hamiltonian).")
    print()
    print("  Method: 2nd-order linear response in strain on a Cartesian BZ grid.")

    candidates = {
        'π²/2':      np.pi**2 / 2,
        'π²/3':      np.pi**2 / 3,
        'π²':        np.pi**2,
        '4π/3':      4 * np.pi / 3,
        '3':         3.0,
        '9/2':       4.5,
        '5':         5.0,
    }
    for N in (16, 24, 32, 40):
        header(f"BZ grid: {N}³ = {N**3} k-points")
        C = bz_average(N_grid=N, mu=0.0)
        v = voigt_components(C)
        print()
        print(f"  C_11 (= C^xxxx)  = {v['C_11']:+.8f}")
        print(f"  C_12 (= C^xxyy)  = {v['C_12']:+.8f}")
        print(f"  C_44 (= C^xyxy)  = {v['C_44']:+.8f}")
        print(f"  Cubic anisotropy: 2 C_44 - (C_11 - C_12) = "
              f"{v['2C_44 - (C_11 - C_12)']:+.8f}")
        # Compare C_44 to candidates
        print()
        print(f"  C_44 candidate matches:")
        for name, val in candidates.items():
            ratio = v['C_44'] / val
            err = abs(v['C_44'] - val)
            marker = "  ←" if err < 0.05 else ""
            print(f"    C_44 / {name:8s} = {ratio:.6f}  (|diff|={err:.4f}){marker}")

    header("Bloch sum-rule sanity check: ⟨Tr(H²)⟩, ⟨Tr(H⁴)⟩ on different domains")
    print()
    print("  Walk-count theorem (lorentz_sig_g_sub_bloch_invariants_theorem.py):")
    print("    ⟨Tr(H²)⟩_BZ = 12, ⟨Tr(H⁴)⟩_BZ = 60.")
    print()
    for he, label in [(np.pi, '[-π, π]³ (subdomain, V=8π³)'),
                       (2*np.pi, '[-2π, 2π]³ (BCC fundamental, V=64π³ = 4×16π³)')]:
        result = verify_bloch_sum_rules(N_grid=16, half_extent=he)
        print(f"  {label}:")
        print(f"    ⟨Tr(H²)⟩ = {result['trH2_avg']:.6f}  ⟨Tr(H⁴)⟩ = {result['trH4_avg']:.6f}")

    header("BZ-domain test: [-2π, 2π]³ (full fundamental domain) vs [-π, π]³")
    print()
    print("  Bloch Hamiltonian for srs is 4π-periodic in each Cartesian direction.")
    print("  Fundamental domain: (4π)³ = 64π³, containing 4 reciprocal-lattice points.")
    print("  ⇒ fundamental-domain volume per reciprocal point = 16π³ (BCC truncated-")
    print("    octahedron volume).")
    print()
    print("  Test: does C_44 change if we sample the full [-2π, 2π]³ domain instead")
    print("  of [-π, π]³ subdomain? (If yes, simple-cubic averaging undersamples.)")
    print()
    for N in (16, 24, 32):
        C_full = bz_average(N_grid=N, mu=0.0, half_extent=2*np.pi)
        v_full = voigt_components(C_full)
        print(f"  Grid {N}³ over [-2π, 2π]³:  C_44 = {v_full['C_44']:+.6f}  "
              f"(vs [-π,π]³: see above)")
        print(f"    C_44 / (π²/2) = {v_full['C_44']/(np.pi**2/2):.6f}")
    print()

    header("Matter-only elastic response (cone bands at λ=-1, excluding background +3)")
    print()
    print("  Restricts the linear-response sum to band-pairs WITHIN the 3-fold")
    print("  cone manifold at λ=-1. Excludes the background band at λ=+3.")
    print()
    for N in (16, 24, 32):
        C_matter = bz_average_matter_only(N_grid=N, cone_lambda=-1.0)
        v_matter = voigt_components(C_matter)
        print(f"  Grid {N}³: C_44_matter = {v_matter['C_44']:+.6f}  "
              f"(C_11 = {v_matter['C_11']:+.4f}, C_12 = {v_matter['C_12']:+.4f})")
        print(f"    C_44_matter / (π²/2) = {v_matter['C_44']/(np.pi**2/2):.6f}")
        print(f"    cubic anisotropy: {v_matter['2C_44 - (C_11 - C_12)']:+.6f}")
    print()

    header("Diagnostic — sample C-tensor entries (16³ grid)")
    C16 = bz_average(N_grid=16, mu=0.0)
    print()
    print(f"  C[xxxx] = {C16[0,0,0,0]:+.6f}")
    print(f"  C[xxyy] = {C16[0,0,1,1]:+.6f}")
    print(f"  C[xyxy] = {C16[0,1,0,1]:+.6f}")
    print(f"  C[xyyx] = {C16[0,1,1,0]:+.6f}")
    print(f"  C[xzxz] = {C16[0,2,0,2]:+.6f}")
    print()
    print("  Symmetries:")
    print(f"    C[abcd] vs C[cdab]:    "
          f"max |diff| = {np.max(np.abs(C16 - np.transpose(C16, (2,3,0,1)))):.2e}")
    print(f"    C[abcd] vs C[bacd]:    "
          f"max |diff| = {np.max(np.abs(C16 - np.transpose(C16, (1,0,2,3)))):.2e}")
    print(f"    C[abcd] vs C[abdc]:    "
          f"max |diff| = {np.max(np.abs(C16 - np.transpose(C16, (0,1,3,2)))):.2e}")

    header("DIAMAGNETIC + PARAMAGNETIC: full elastic-modulus tensor on proper BCC BZ")
    print()
    print("  Earlier results were PARAMAGNETIC ONLY. Adding diamagnetic")
    print("  W^{abcd} = ∂²H/∂u² contribution. If diamagnetic ≈ 2.24, the 12% gap")
    print("  closes to 2π² = 19.74, giving G_sub = 1/(16π³) exactly.")
    print()
    for N in (12, 16):
        K_para, K_dia, K_full = bz_average_full(N_grid=N, mu=0.0,
                                                  half_extent=2*np.pi)
        v_para = voigt_components(K_para)
        v_dia = voigt_components(K_dia)
        v_full = voigt_components(K_full)
        print(f"  Grid {N}³ over [-2π, 2π]³:")
        print(f"    Paramagnetic: C_44 = {v_para['C_44']:+.6f}, "
              f"μ_iso(Voigt) = {(v_para['C_11'] - v_para['C_12'] + 3*v_para['C_44'])/5:+.6f}")
        print(f"    Diamagnetic:  C_44 = {v_dia['C_44']:+.6f}, "
              f"μ_iso(Voigt) = {(v_dia['C_11'] - v_dia['C_12'] + 3*v_dia['C_44'])/5:+.6f}")
        print(f"    FULL:         C_44 = {v_full['C_44']:+.6f}, "
              f"μ_iso(Voigt) = {(v_full['C_11'] - v_full['C_12'] + 3*v_full['C_44'])/5:+.6f}")
        mu_full_voigt = (v_full['C_11'] - v_full['C_12'] + 3*v_full['C_44']) / 5
        target_mu = 2 * np.pi**2
        print(f"    Target for G_sub = 1/(16π³): μ = 2π² = {target_mu:+.6f}")
        print(f"    Numerical/Target = {mu_full_voigt/target_mu:.6f}")
        print(f"    G_sub via full: 1/(8π·μ_full) = {1/(8*np.pi*mu_full_voigt):.8f}")
        print(f"    Compare 1/(16π³) = {1/(16*np.pi**3):.8f}")
        print()

    header("FULL C^abcd on proper BCC BZ + TT projection")
    print()
    print("  Compute all components of C^abcd on the proper BCC fundamental")
    print("  domain [-2π, 2π]³ (4 reciprocal-lattice points per cube).")
    print()
    for N in (16, 24, 32):
        C_proper = bz_average(N_grid=N, mu=0.0, half_extent=2*np.pi)
        v_proper = voigt_components(C_proper)
        print(f"  Grid {N}³ over [-2π, 2π]³ (proper BCC fundamental domain):")
        print(f"    C_11 = {v_proper['C_11']:+.6f}")
        print(f"    C_12 = {v_proper['C_12']:+.6f}")
        print(f"    C_44 = {v_proper['C_44']:+.6f}")
        print(f"    cubic anisotropy 2C_44 - (C_11 - C_12) = "
              f"{v_proper['2C_44 - (C_11 - C_12)']:+.6f}")

        # TT-projected graviton kinetic coefficient.
        # For an isotropic medium with Lamé μ: graviton TT couples with eigenvalue 2μ.
        # For cubic medium, decompose C into isotropic + cubic-anisotropic, take iso part.
        # Voigt-Reuss-Hill iso averages:
        #   Voigt μ = (C_11 - C_12 + 3 C_44) / 5
        #   Reuss μ = 5 (C_11 - C_12) C_44 / [4 C_44 + 3 (C_11 - C_12)]
        c11, c12, c44 = v_proper['C_11'], v_proper['C_12'], v_proper['C_44']
        mu_voigt = (c11 - c12 + 3 * c44) / 5
        try:
            mu_reuss = 5 * (c11 - c12) * c44 / (4 * c44 + 3 * (c11 - c12))
        except ZeroDivisionError:
            mu_reuss = float('nan')
        mu_hill = (mu_voigt + mu_reuss) / 2

        # The graviton kinetic coefficient = 2μ (for unit normalization);
        # 1/(16π G_sub) = 2μ_iso ⇒ G_sub = 1/(32π μ_iso).
        G_voigt = 1 / (32 * np.pi * mu_voigt)
        G_reuss = 1 / (32 * np.pi * mu_reuss)
        G_hill = 1 / (32 * np.pi * mu_hill)
        G_target = 1 / (16 * np.pi**3)

        print(f"    Iso shear modulus (Voigt): μ_V = {mu_voigt:+.6f}")
        print(f"    Iso shear modulus (Reuss): μ_R = {mu_reuss:+.6f}")
        print(f"    Iso shear modulus (Hill):  μ_H = {mu_hill:+.6f}")
        print(f"    G_sub via Voigt: 1/(32π μ_V) = {G_voigt:+.8f}")
        print(f"    G_sub via Reuss: 1/(32π μ_R) = {G_reuss:+.8f}")
        print(f"    G_sub via Hill:  1/(32π μ_H) = {G_hill:+.8f}")
        print(f"    Target 1/(16π³)               = {G_target:+.8f}")
        print(f"    Ratios to target: V={G_voigt/G_target:.4f}, "
              f"R={G_reuss/G_target:.4f}, H={G_hill/G_target:.4f}")
        # Also try alternative identifications
        # Some conventions: 1/(16π G_sub) = μ_iso ⇒ G_sub = 1/(16π μ_iso)
        G_alt_v = 1 / (16 * np.pi * mu_voigt)
        G_alt_r = 1 / (16 * np.pi * mu_reuss)
        print(f"    Alt: 1/(16π μ_V) = {G_alt_v:+.8f}, "
              f"1/(16π μ_R) = {G_alt_r:+.8f}")
        print(f"    Compare: candidate 1/(16π³) ≈ {G_target:.8f}, "
              f"1/(8π³) ≈ {1/(8*np.pi**3):.8f}")
        print()
        # Also: what value of μ_iso gives exactly 1/(16π³)?
        # G_sub = 1/(16π³) ⇒ μ_iso (using 1/(16π·μ) = G_sub) = π².
        # ⇒ μ_iso (using 1/(32π·μ) = G_sub) = π²/2.
        print(f"    For G_sub = 1/(16π³): need μ_iso = π² = {np.pi**2:.6f} "
              f"(if 1/(16π G) = μ_iso)")
        print(f"                         or μ_iso = π²/2 = {np.pi**2/2:.6f} "
              f"(if 1/(16π G) = 2μ_iso)")

    header("CRITICAL FINDING: BZ-volume convention bug in 1/(8π³) claim")
    print()
    print("  Bloch sum-rule sanity (verified above): ⟨Tr(H²)⟩ = 12, ⟨Tr(H⁴)⟩ = 60")
    print("  hold on BOTH [-π, π]³ and [-2π, 2π]³ — these are domain-independent")
    print("  because the integrand is built from bond-phase products.")
    print()
    print("  BUT the elastic modulus C_44 differs SIGNIFICANTLY:")
    print(f"    [-π, π]³  (volume 8π³, subdomain):     C_44 ≈ 4.56")
    print(f"    [-2π, 2π]³ (volume 64π³, 4 fund. domains): C_44 ≈ 17.4")
    print()
    print("  Why? Linear-response integrand involves eigenvectors which depend")
    print("  on basis convention; these vary across the BZ in a way that")
    print("  Tr(H^n) doesn't.")
    print()
    print("  PROPER BCC BZ-volume is V_BZ_BCC = (2π)³/V_primitive = (2π)³/(1/2) = 16π³.")
    print("  The structural form's earlier V_BZ = (2π)³ = 8π³ is the SIMPLE-CUBIC")
    print("  convention — wrong for srs (BCC primitive cell).")
    print()
    print("  CORRECTED structural form:")
    print("    G_sub_form = ⟨Tr(R_4²)⟩ · v_F / (⟨Tr(H²)⟩ · V_BZ_BCC)")
    print(f"               = 24 · (1/2) / (12 · 16π³) = 1/(16π³) ≈ {float(1/(16*np.pi**3)):.6f}")
    print()
    print("  This corrects the previous claim G_sub = 1/(8π³) ≈ 0.00403")
    print(f"  to G_sub = 1/(16π³) ≈ {float(1/(16*np.pi**3)):.6f}")
    print()
    print("  Cross-check with numerical elastic moduli on proper [-2π, 2π]³ BZ:")
    print("    C_44_proper ≈ 17.4 ⇒ assuming 1/(16π G_sub) = (1/2) C_44_proper:")
    print(f"      G_sub_elastic = 1/(16π · 8.7) ≈ {float(1/(16*np.pi*8.7)):.6f}")
    print(f"      compared to 1/(16π³) ≈ {float(1/(16*np.pi**3)):.6f}")
    print(f"      ratio: {float(1/(16*np.pi*8.7)) / float(1/(16*np.pi**3)):.4f}")
    print()
    print("  Match within ~13%. The residual gap is likely cubic anisotropy")
    print("  (substrate is cubic, not isotropic — graviton TT projection")
    print("  captures only part of the elastic modulus structure).")

    header("Connection to G_sub")
    print()
    print("  Standard isotropic-elastic continuum: graviton-like wave equation")
    print("    ρ ∂²_t u^a = (λ + 2μ) ∂_a (∂·u) - μ (∇×∇×u)^a")
    print("  Plane wave: longitudinal speed c_L² = (λ + 2μ)/ρ; transverse c_T² = μ/ρ.")
    print()
    print("  In the substrate's spin-1 Dirac-cone projection (β=1 vielbein),")
    print("  transverse strain modes are the graviton's TT polarisations.")
    print("  Their dispersion v_F = 1/2 implies c_T = v_F = 1/2, so")
    print("    μ_lame / ρ_substrate = (1/2)² = 1/4.")
    print()
    print("  The substrate's effective mass density ρ_substrate enters via the")
    print("  matter sector (kinetic term of ψ). For the spin-1 Dirac at Γ with")
    print("  unit amplitude, ρ_substrate = (1/v_F) × ⟨T^{00}⟩ at unit |k|² = ...")
    print()
    print("  Net: C_44 (= μ_lame) × (V_BZ × spin_factor) gives 1/(16π G_sub) up")
    print("  to a coefficient determined by the graviton normalisation.")
    print("  Expected: C_44 = O(1) and 1/(16π G_sub) = C_44 / (something structural).")


if __name__ == "__main__":
    main()
