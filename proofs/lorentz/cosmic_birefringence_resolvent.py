#!/usr/bin/env python3
"""
cosmic_birefringence_resolvent.py — D1 of the resolvent route for c=1 in
β = c · sin(arg h) · α_EM (a separate private derivation by the author port).

Goal. Test empirically whether the walker resolvent G(z) = (I − zB)^{-1}
projected onto the photon channel gives a parity-odd content of
sin(arg h) with coefficient exactly 1, as a separate private derivation by the author claimed.

If yes: c = 1 follows from spectral non-degeneracy + MDL Lemma 1.
If no: identify which a separate private derivation by the author premise fails. Likely candidate: at k_P,
photon ⊥ V_Ram (this repo's prior finding), so the leading +h
eigenmode (which lives in V_Ram) doesn't couple directly to the photon.

Concrete tests:

1. **Direct G(z=1−ε) projection.** Compute resolvent G(z=1−ε) at small ε,
   project onto photon Hodge bundle's L/R basis. Extract Im(diagonal)
   in L/R basis. Is it ∝ sin(arg h)?

2. **Long-walk B^N projection.** Compute B^N for large N (say N = 50),
   normalize by |h|^(2N), project onto photon. Does it converge to a
   parity-odd projection that's sin(arg h)?

3. **V_Ram-restricted G.** Project G onto V_Ram first, then to photon.
   Does the V_Ram-restricted resolvent give sin(arg h)?

4. **Generic-k test.** Same computation at k near (but not at) k_P,
   where photon ⊥ V_Ram fails. Maybe c=1 emerges in the integrated
   contribution.

Honest empirical investigation — not a "port the answer" exercise.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proofs" / "cosmology"))

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
    canonical_edges_primitive,
    incidence_matrix_primitive,
)
from srs_photon_c3_chainmap import K_P_RED, build_C3_edge, build_delta_1
from srs_photon_chirality_coefficient import (
    build_B_directed,
    build_C3_directed,
    build_pi_projector,
)
from srs_photon_hodge import build_d1, build_edge_lookup
from srs_cycle_enumerator import enumerate_simple_cycles


H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
SIN_ARG_H = math.sqrt(5.0 / 8.0)
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))
PRINT_WIDTH = 78


def find_photon_LR_basis(bonds, edges, k_red, all_verts, cycles):
    """Return 12×2 matrix Q whose columns are L/R photon polarization
    eigenstates at k_red, lifted via π to directed bonds.

    L = ω-irrep of C₃, R = ω²-irrep of C₃ (per srs_photon_chirality_coefficient).
    """
    n_verts = len(all_verts)
    n_edges = len(edges)
    edge_lookup = build_edge_lookup(edges)
    d = incidence_matrix_primitive(k_red, edges, n_verts)
    d1 = build_d1(cycles, edge_lookup, k_red, n_edges)
    Delta_1 = build_delta_1(d, d1)
    # Make Hermitian (numerical safety)
    Delta_1 = (Delta_1 + Delta_1.conj().T) / 2
    evs, evecs = la.eigh(Delta_1)
    # Photon eigenspace = top 2 eigenvalues (largest, ≈ 36 at P).
    idx_sort = np.argsort(evs)
    idx_phot = idx_sort[-2:]
    Q_undir = evecs[:, idx_phot]    # 6×2
    # Lift to 12-dim directed via π.
    pi = build_pi_projector(bonds, edges, k_red)   # 12×6
    Q_dir = pi @ Q_undir            # 12×2
    # Re-orthonormalize Q_dir (since π isn't an isometry on this 2-dim subspace)
    Q_dir, _ = la.qr(Q_dir)
    Q_dir = Q_dir[:, :2]
    # Decompose into L = ω, R = ω² C₃ irreps via C₃-eigenvalue
    C3_dir = build_C3_directed(bonds)
    Q_dir_in_C3 = Q_dir.conj().T @ C3_dir @ Q_dir   # 2×2 in photon basis
    evs_C3, evecs_C3 = la.eig(Q_dir_in_C3)
    omega = np.exp(2j * math.pi / 3)
    # Identify L (ω) and R (ω²)
    L_idx = int(np.argmin([abs(evs_C3[0] - omega), abs(evs_C3[1] - omega)]))
    R_idx = 1 - L_idx
    Q_LR_basis = Q_dir @ np.column_stack([evecs_C3[:, L_idx], evecs_C3[:, R_idx]])
    return Q_LR_basis


def compute_resolvent_projection(B, Q_LR, z):
    """Compute G(z) = (I - zB)^(-1), project onto Q_LR (12×2 photon L/R basis).

    Returns the 2×2 matrix M = Q_LR† G(z) Q_LR.
    """
    n = B.shape[0]
    G = la.inv(np.eye(n, dtype=complex) - z * B)
    return Q_LR.conj().T @ G @ Q_LR


def analyze_2x2(M, label=""):
    """Decompose 2×2 M = a₀I + ⃗a·σ; extract parity-odd content.

    Returns dict of structural numbers.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    a0 = (M[0, 0] + M[1, 1]) / 2
    az = (M[0, 0] - M[1, 1]) / 2
    ax = (M[0, 1] + M[1, 0]) / 2
    ay = 1j * (M[0, 1] - M[1, 0]) / 2
    # Diff = M_LL - M_RR (twice z-component)
    diff = M[0, 0] - M[1, 1]
    return {
        "M_LL": M[0, 0],
        "M_RR": M[1, 1],
        "diff/2": diff / 2,
        "Im(diff)/2": diff.imag / 2,
        "Re(diff)/2": diff.real / 2,
        "a_x": ax,
        "a_y": ay,
        "a_z": az,
        "a_0": a0,
    }


def main():
    print("=" * PRINT_WIDTH)
    print("Resolvent route to c=1: empirical test of a separate private derivation by the author argument")
    print("=" * PRINT_WIDTH)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    k_P = np.array(K_P_RED)
    B_kP = build_B_directed(bonds, k_P)
    print(f"  {len(cycles)} length-10 cycles enumerated")

    print(f"\n  sin(arg h) = √(5/8) = {SIN_ARG_H:.6f}")
    print(f"  arg(h) = {math.degrees(ARG_H):.4f}°")

    # Build photon L/R basis at k_P
    print(f"\n— Photon L/R basis at k_P —")
    Q_LR = find_photon_LR_basis(bonds, edges, k_P, verts, cycles)
    print(f"  Q_LR shape: {Q_LR.shape}")

    # Verify orthonormality
    norm_check = Q_LR.conj().T @ Q_LR
    print(f"  Q_LR† Q_LR =\n{norm_check}")

    # Verify photon ⊥ V_Ram (this repo's prior finding)
    evs_B, evecs_B = la.eig(B_kP)
    h_idx = [i for i, e in enumerate(evs_B) if abs(e - H_EXACT) < 1e-10]
    V_Ram_h = evecs_B[:, h_idx]
    overlap = la.norm(Q_LR.conj().T @ V_Ram_h)
    print(f"\n  ⟨photon | V_Ram(+h) ⟩  Frobenius overlap = {overlap:.2e}")
    print(f"  (expect ~10⁻¹⁵ at k_P since photon ⊥ V_Ram per prior finding)")

    # =========================================================================
    # TEST 1: Direct resolvent G(z=1-ε) projection
    # =========================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("TEST 1: Direct resolvent G(z=1−ε) → photon L/R")
    print("=" * PRINT_WIDTH)
    for eps in [1e-2, 1e-4, 1e-6]:
        z = 1.0 - eps
        M = compute_resolvent_projection(B_kP, Q_LR, z)
        result = analyze_2x2(M)
        print(f"\n  z = 1 - {eps}:")
        print(f"    M_LL = {result['M_LL']:.6f}")
        print(f"    M_RR = {result['M_RR']:.6f}")
        print(f"    diff/2 = {result['diff/2']:.6f}")
        print(f"    Im(diff)/2 = {result['Im(diff)/2']:.6e}")
        print(f"    expect ratio to sin(arg h)·α_EM if a separate private derivation by the author right; or 0 if not")
        # Ratio to sin(arg h):
        if abs(SIN_ARG_H) > 0:
            ratio = result['Im(diff)/2'] / SIN_ARG_H
            print(f"    Im(diff)/2 / sin(arg h) = {ratio:.6f}")

    # =========================================================================
    # TEST 2: Long-walk B^N projection
    # =========================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("TEST 2: Long-walk B^N projection (normalized by |h|^{2N})")
    print("=" * PRINT_WIDTH)
    H_norm_sq = abs(H_EXACT)**2  # = 2
    for N in [4, 8, 16, 32]:
        BN = la.matrix_power(B_kP, N)
        # Normalize by leading-mode amplitude; use |h|^N for absolute mag
        BN_norm = BN / (H_norm_sq ** (N / 2))
        # Project to photon L/R
        M = Q_LR.conj().T @ BN_norm @ Q_LR
        result = analyze_2x2(M)
        print(f"\n  N = {N}:")
        print(f"    |M_LL| = {abs(result['M_LL']):.6f}")
        print(f"    |M_RR| = {abs(result['M_RR']):.6f}")
        print(f"    arg(M_LL) = {np.angle(result['M_LL'], deg=True):+.4f}°")
        print(f"    arg(M_RR) = {np.angle(result['M_RR'], deg=True):+.4f}°")
        print(f"    Im(diff)/2 = {result['Im(diff)/2']:.6e}")

    # =========================================================================
    # TEST 3: Resolvent restricted to V_Ram subspace before projection
    # =========================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("TEST 3: V_Ram-restricted resolvent → photon")
    print("=" * PRINT_WIDTH)
    # Build V_Ram (8-dim subspace where |B|² = 2)
    V_Ram_idx = [i for i, e in enumerate(evs_B) if abs(abs(e)**2 - 2) < 1e-9]
    V_Ram = evecs_B[:, V_Ram_idx]
    Q_VR, _ = la.qr(V_Ram)
    Q_VR = Q_VR[:, :len(V_Ram_idx)]
    print(f"  V_Ram dim = {len(V_Ram_idx)}")
    print(f"  Photon overlap into V_Ram: {la.norm(Q_VR.conj().T @ Q_LR):.4e}")
    print(f"  (zero by prior finding photon ⊥ V_Ram at k_P)")

    # =========================================================================
    # TEST 4: Generic k near k_P (test if photon-V_Ram coupling emerges)
    # =========================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("TEST 4: Photon-V_Ram coupling at generic k near k_P")
    print("=" * PRINT_WIDTH)
    print("  k near k_P: photon Hodge eigenspace shifts; overlap with V_Ram?")
    for delta in [1e-3, 1e-2, 5e-2, 1e-1]:
        k_perturbed = k_P + delta * np.array([0, 1, -1]) / np.sqrt(2)
        try:
            Q_LR_pert = find_photon_LR_basis(bonds, edges, k_perturbed, verts, cycles)
            B_pert = build_B_directed(bonds, k_perturbed)
            evs_pert, evecs_pert = la.eig(B_pert)
            # Find +h-like eigenmode (closest to H_EXACT)
            h_idx_pert = sorted(range(len(evs_pert)),
                                key=lambda i: abs(evs_pert[i] - H_EXACT))[:2]
            V_Ram_pert = evecs_pert[:, h_idx_pert]
            overlap = la.norm(Q_LR_pert.conj().T @ V_Ram_pert)
            print(f"  δk = {delta:.0e}: photon-V_Ram_+h overlap = {overlap:.4e}")
        except Exception as e:
            print(f"  δk = {delta:.0e}: failed ({e})")

    print(f"\n" + "=" * PRINT_WIDTH)
    print("VERDICT")
    print("=" * PRINT_WIDTH)
    print("""
    The empirical question: does the photon-channel resolvent give
    sin(arg h) with coefficient 1?

    Key constraint: at k_P, photon ⊥ V_Ram (prior repo finding).
    The leading +h Ramanujan eigenmode (which carries Im(h)) doesn't
    directly couple to the photon at the high-symmetry point.

    If photon-V_Ram overlap emerges at generic k (Test 4), the integrated
    contribution might still give sin(arg h)·α_EM via leading-mode
    dominance. If not, a separate private derivation by the author resolvent argument doesn't port directly.
    """)


if __name__ == "__main__":
    main()
