#!/usr/bin/env python3
"""
PMNS from Koide mass matrix + srs Bloch structure: principled k-point derivation.

STRATEGY:
  1. Construct the Koide mass matrix M_l for charged leptons (δ=2/9, ε=√2).
     Diagonalize to get U_l (the Koide diagonalizer).
  2. For each k-point on a fine BZ grid, compute U(k) = V(Γ)†·V(k).
  3. INVERSE PROBLEM: find k where U(k) ≈ U_l (Koide). This is the "Koide k-point."
  4. DIRECT PROBLEM: find k where U_PMNS(k) = U(k)†·U_TBM best matches observed PMNS.
  5. Compare the two k-points. Check for algebraic relations to known quantities.

Key parameters from the framework:
  δ = 2/9     (Koide phase, theorem)
  ε = √2      (Koide amplitude, theorem)
  λ₁ = 2-√3   (spectral gap)
  g = 10       (trivalent connectivity constant)
  k* = 3       (optimal degree)
"""

import numpy as np
from numpy import linalg as la
from itertools import product
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))
np.set_printoptions(precision=8, linewidth=140)

# ═══════════════════════════════════════════════════════════════════════
# OBSERVED VALUES (PDG 2024)
# ═══════════════════════════════════════════════════════════════════════

theta12_obs = math.radians(33.44)
theta23_obs = math.radians(49.2)
theta13_obs = math.radians(8.57)
delta_CP_obs = math.radians(230.0)

# ═══════════════════════════════════════════════════════════════════════
# SRS LATTICE (4-atom primitive cell, BCC)
# ═══════════════════════════════════════════════════════════════════════

omega3 = np.exp(2j * np.pi / 3)
NN_DIST = np.sqrt(2) / 4

A_PRIM = np.array([
    [-0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5, -0.5],
])

ATOMS = np.array([
    [1/8, 1/8, 1/8],
    [3/8, 7/8, 5/8],
    [7/8, 5/8, 3/8],
    [5/8, 3/8, 7/8],
])
N_ATOMS = 4

C3_PERM = np.zeros((4, 4), dtype=complex)
C3_PERM[0, 0] = 1
C3_PERM[3, 1] = 1
C3_PERM[1, 2] = 1
C3_PERM[2, 3] = 1

# BCC high-symmetry points (fractional reciprocal coordinates)
K_POINTS = {
    'Gamma': np.array([0.0, 0.0, 0.0]),
    'H':     np.array([0.5, -0.5, 0.5]),
    'N':     np.array([0.0, 0.0, 0.5]),
    'P':     np.array([0.25, 0.25, 0.25]),
}


# ═══════════════════════════════════════════════════════════════════════
# LATTICE INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════

def find_bonds():
    tol = 0.02
    bonds = []
    for i in range(N_ATOMS):
        ri = ATOMS[i]
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                dist = la.norm(rj - ri)
                if dist < tol:
                    continue
                if abs(dist - NN_DIST) < tol:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds


def bloch_H(k_frac, bonds):
    H = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for src, tgt, cell in bonds:
        phase = np.exp(2j * np.pi * np.dot(k, cell))
        H[tgt, src] += phase
    return H


def diag_H(k_frac, bonds):
    H = bloch_H(k_frac, bonds)
    evals, evecs = la.eigh(H)
    idx = np.argsort(np.real(evals))
    return np.real(evals[idx]), evecs[:, idx]


def c3_decompose(k_frac, bonds, degen_tol=1e-8):
    evals, evecs = diag_H(k_frac, bonds)
    groups = []
    i = 0
    while i < N_ATOMS:
        grp = [i]
        while i + 1 < N_ATOMS and abs(evals[i+1] - evals[i]) < degen_tol:
            i += 1
            grp.append(i)
        groups.append(grp)
        i += 1

    new_evecs = evecs.copy()
    c3_diag = np.zeros(N_ATOMS, dtype=complex)
    for grp in groups:
        if len(grp) == 1:
            b = grp[0]
            c3_diag[b] = np.conj(evecs[:, b]) @ C3_PERM @ evecs[:, b]
        else:
            sub = evecs[:, grp]
            C3_sub = np.conj(sub.T) @ C3_PERM @ sub
            c3_evals, c3_evecs = la.eig(C3_sub)
            order = np.argsort(np.angle(c3_evals))
            c3_evals = c3_evals[order]
            c3_evecs = c3_evecs[:, order]
            new_sub = sub @ c3_evecs
            for ig, b in enumerate(grp):
                new_evecs[:, b] = new_sub[:, ig]
                c3_diag[b] = c3_evals[ig]

    return evals, new_evecs, c3_diag


# ═══════════════════════════════════════════════════════════════════════
# MIXING ANGLE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_angles(U):
    """Extract mixing angles and CP phase from a 3x3 unitary matrix."""
    s13 = min(abs(U[0, 2]), 1.0)
    theta13 = math.asin(s13)
    c13 = math.cos(theta13)

    if c13 < 1e-10:
        return 0.0, 0.0, theta13, 0.0

    sin2_12 = min(max(abs(U[0, 1])**2 / c13**2, 0.0), 1.0)
    theta12 = math.asin(math.sqrt(sin2_12))

    sin2_23 = min(max(abs(U[1, 2])**2 / c13**2, 0.0), 1.0)
    theta23 = math.asin(math.sqrt(sin2_23))

    J = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    denom = (math.cos(theta12) * math.sin(theta12) * math.cos(theta23) *
             math.sin(theta23) * c13**2 * s13)
    if abs(denom) > 1e-15:
        sin_delta = min(max(J / denom, -1.0), 1.0)
        delta = math.asin(sin_delta)
        if s13 > 1e-10:
            phase_e3 = np.angle(U[0, 2])
            delta_from_phase = -phase_e3
            if abs(math.sin(delta_from_phase) - sin_delta) < 0.5:
                delta = delta_from_phase
    else:
        delta = 0.0

    return theta12, theta23, theta13, delta


def jarlskog(U):
    return np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))


def TBM_matrix():
    """Tribimaximal mixing matrix."""
    s12 = 1.0 / math.sqrt(3)
    c12 = math.sqrt(2.0 / 3.0)
    s23 = 1.0 / math.sqrt(2)
    c23 = 1.0 / math.sqrt(2)
    return np.array([
        [ c12,  s12,  0.0],
        [-s12*c23, c12*c23, s23],
        [ s12*s23, -c12*s23, c23],
    ], dtype=complex)


# ═══════════════════════════════════════════════════════════════════════
# 1. KOIDE MASS MATRIX AND DIAGONALIZER
# ═══════════════════════════════════════════════════════════════════════

def koide_mass_matrix():
    """
    Construct the Koide mass matrix for charged leptons.

    M_l = M₀ · (I + ε · C(δ))

    where C(δ)_{ij} = cos(2π/3 · (i-j) + δ)   (i,j = 0,1,2)
    with δ = 2/9 and ε = √2.

    This is the democratic mass matrix with Koide's phase.
    The eigenvalues reproduce me : mμ : mτ and the eigenvectors
    give the charged lepton diagonalizer U_l.
    """
    delta = 2.0 / 9.0   # Koide phase (theorem from MDL)
    eps = math.sqrt(2)    # Koide amplitude (theorem)

    # Circulant cosine matrix: C_{ij} = cos(2π(i-j)/3 + δ)
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            C[i, j] = math.cos(2 * math.pi * (i - j) / 3.0 + delta)

    M = np.eye(3) + eps * C

    print("=" * 76)
    print("  1. KOIDE MASS MATRIX")
    print("=" * 76)
    print(f"\n  Parameters: delta = 2/9 = {delta:.10f}, epsilon = sqrt(2) = {eps:.10f}")
    print(f"\n  Cosine matrix C(delta):")
    for i in range(3):
        print(f"    [{C[i,0]:+.8f}  {C[i,1]:+.8f}  {C[i,2]:+.8f}]")

    print(f"\n  M_l = I + sqrt(2) * C:")
    for i in range(3):
        print(f"    [{M[i,0]:+.8f}  {M[i,1]:+.8f}  {M[i,2]:+.8f}]")

    # Diagonalize
    evals, evecs = la.eigh(M)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    print(f"\n  Eigenvalues of M_l: {evals}")
    print(f"  Ratios (normalized to sum): {evals / np.sum(evals)}")

    # Check Koide relation: (me + mmu + mtau) / (sqrt(me) + sqrt(mmu) + sqrt(mtau))^2 = 2/3
    # Here eigenvalues are proportional to sqrt(mass) squared, i.e., mass itself
    m = np.maximum(evals, 0)
    sqrt_m = np.sqrt(m)
    koide_ratio = np.sum(m) / np.sum(sqrt_m)**2
    print(f"\n  Koide ratio: sum(m) / (sum(sqrt(m)))^2 = {koide_ratio:.10f}")
    print(f"  Expected: 2/3 = {2/3:.10f}")
    print(f"  Match: {'YES' if abs(koide_ratio - 2/3) < 1e-8 else 'NO'}")

    # Physical masses from eigenvalues
    # me/mtau, mmu/mtau
    if evals[-1] > 0:
        print(f"\n  Mass ratios (eigenvalue ratios):")
        print(f"    m1/m3 = {evals[0]/evals[2]:.8f}")
        print(f"    m2/m3 = {evals[1]/evals[2]:.8f}")
        print(f"  Physical: me/mtau = {0.000511/1.777:.8f}, mmu/mtau = {0.1057/1.777:.8f}")

    print(f"\n  Koide diagonalizer U_l (columns = mass eigenstates):")
    for i in range(3):
        print(f"    [{evecs[i,0]:+.8f}  {evecs[i,1]:+.8f}  {evecs[i,2]:+.8f}]")

    print(f"\n  |U_l|^2:")
    for i in range(3):
        row = [abs(evecs[i, j])**2 for j in range(3)]
        print(f"    [{row[0]:.8f}  {row[1]:.8f}  {row[2]:.8f}]")

    return M, evals, evecs


# ═══════════════════════════════════════════════════════════════════════
# 2. PMNS FROM KOIDE U_l + TBM U_nu
# ═══════════════════════════════════════════════════════════════════════

def pmns_from_koide(U_l):
    """Compute U_PMNS = U_l^dagger . U_TBM and extract angles."""
    U_TBM = TBM_matrix()
    U_PMNS = np.conj(U_l.T) @ U_TBM

    t12, t23, t13, dcp = extract_angles(U_PMNS)
    J = jarlskog(U_PMNS)

    print("\n" + "=" * 76)
    print("  2. PMNS FROM KOIDE: U_PMNS = U_l(Koide)^dagger . U_TBM")
    print("=" * 76)

    print(f"\n  |U_PMNS|^2:")
    for i in range(3):
        row = [abs(U_PMNS[i, j])**2 for j in range(3)]
        print(f"    [{row[0]:.8f}  {row[1]:.8f}  {row[2]:.8f}]")

    print(f"\n  theta_12 = {math.degrees(t12):.4f} deg  (obs: 33.44)")
    print(f"  theta_23 = {math.degrees(t23):.4f} deg  (obs: 49.20)")
    print(f"  theta_13 = {math.degrees(t13):.4f} deg  (obs:  8.57)")
    print(f"  delta_CP = {math.degrees(dcp):.4f} deg  (obs: ~230)")
    print(f"  J        = {J:.8f}  (obs: 0.033)")

    rms = math.sqrt(((math.degrees(t12) - 33.44)**2 +
                      (math.degrees(t23) - 49.20)**2 +
                      (math.degrees(t13) - 8.57)**2) / 3.0)
    print(f"  RMS angle error = {rms:.4f} deg")

    return U_PMNS, t12, t23, t13, dcp


# ═══════════════════════════════════════════════════════════════════════
# 3. REFERENCE GAMMA EIGENSTATES
# ═══════════════════════════════════════════════════════════════════════

def get_gamma_reference(bonds):
    """Get C3-decomposed eigenstates at Gamma. Return triplet subspace."""
    evals_G, evecs_G_c3, c3_diag_G = c3_decompose([0, 0, 0], bonds)

    # Identify triplet (E = -1, 3-fold degenerate)
    triplet_idx = np.where(np.abs(evals_G - (-1.0)) < 0.5)[0]
    assert len(triplet_idx) == 3, f"Expected 3 triplet bands, got {len(triplet_idx)}"

    return evecs_G_c3, triplet_idx


# ═══════════════════════════════════════════════════════════════════════
# 4. BLOCH MIXING U(k) AT ARBITRARY k
# ═══════════════════════════════════════════════════════════════════════

def bloch_mixing_3x3(k_frac, bonds, V_Gamma, triplet_idx_G):
    """
    Compute the 3x3 triplet mixing matrix U(k) = V(Gamma)^dagger . V(k).
    Returns the 3x3 subblock and whether it is approximately unitary.
    """
    evals_k, evecs_k = diag_H(k_frac, bonds)
    U_full = np.conj(V_Gamma.T) @ evecs_k

    # Find the 3 k-bands with maximal triplet overlap
    triplet_overlap = np.sum(np.abs(U_full[triplet_idx_G, :])**2, axis=0)
    triplet_idx_k = np.sort(np.argsort(triplet_overlap)[-3:])
    U_trip = U_full[np.ix_(triplet_idx_G, triplet_idx_k)]

    # Project to nearest unitary via SVD
    Usvd, S, Vhsvd = la.svd(U_trip)
    U_unitary = Usvd @ Vhsvd
    unitarity_err = la.norm(U_trip @ np.conj(U_trip.T) - np.eye(3))

    return U_unitary, unitarity_err


# ═══════════════════════════════════════════════════════════════════════
# 5. INVERSE PROBLEM: find k where U(k) ≈ U_l(Koide)
# ═══════════════════════════════════════════════════════════════════════

def find_koide_kpoint(bonds, V_Gamma, triplet_idx_G, U_l_koide, n_grid=40):
    """
    Scan the BZ on a grid, find the k-point where the Bloch mixing
    matrix U(k) most closely matches the Koide diagonalizer U_l.

    Distance metric: ||U(k) - U_l||_F (Frobenius norm), minimized over
    phase ambiguities by comparing |U(k)|^2 vs |U_l|^2 elementwise.
    """
    print("\n" + "=" * 76)
    print("  5. INVERSE PROBLEM: find k where U(k) ~ U_l(Koide)")
    print(f"     Grid: {n_grid}^3 points in BCC BZ")
    print("=" * 76)

    target_sq = np.abs(U_l_koide)**2

    best_k = None
    best_err = 1e10
    best_U = None

    # Also try phase-aware comparison: minimize over diagonal phases
    # |U(k) D - U_l|_F where D = diag(e^{iφ₁}, e^{iφ₂}, e^{iφ₃})
    # The minimum is achieved when D aligns the phases column by column.

    results = []

    for i1 in range(n_grid):
        k1 = -0.5 + (i1 + 0.5) / n_grid
        for i2 in range(n_grid):
            k2 = -0.5 + (i2 + 0.5) / n_grid
            for i3 in range(n_grid):
                k3 = -0.5 + (i3 + 0.5) / n_grid

                k = [k1, k2, k3]
                U_k, _ = bloch_mixing_3x3(k, bonds, V_Gamma, triplet_idx_G)

                # Compare |U|^2 (phase-invariant)
                err_sq = la.norm(np.abs(U_k)**2 - target_sq)

                # Also try all column permutations (band ordering ambiguity)
                from itertools import permutations
                min_err = err_sq
                for perm in permutations(range(3)):
                    U_perm = U_k[:, list(perm)]
                    e = la.norm(np.abs(U_perm)**2 - target_sq)
                    if e < min_err:
                        min_err = e

                if min_err < best_err:
                    best_err = min_err
                    best_k = np.array(k)
                    best_U = U_k.copy()

                if min_err < 0.15:
                    results.append((np.array(k), min_err))

    print(f"\n  Best k-point: ({best_k[0]:.4f}, {best_k[1]:.4f}, {best_k[2]:.4f})")
    print(f"  ||U(k)||^2 - |U_l|^2||_F = {best_err:.6f}")
    print(f"\n  |U(k_best)|^2:")
    for i in range(3):
        row = [abs(best_U[i, j])**2 for j in range(3)]
        print(f"    [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")
    print(f"  |U_l(Koide)|^2:")
    for i in range(3):
        row = [abs(U_l_koide[i, j])**2 for j in range(3)]
        print(f"    [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")

    if len(results) > 0:
        print(f"\n  {len(results)} k-points with error < 0.15:")
        results.sort(key=lambda x: x[1])
        for kk, ee in results[:10]:
            print(f"    k = ({kk[0]:+.4f}, {kk[1]:+.4f}, {kk[2]:+.4f})  err = {ee:.6f}")

    return best_k, best_err, best_U, results


# ═══════════════════════════════════════════════════════════════════════
# 6. DIRECT PROBLEM: find k minimizing PMNS RMS error
# ═══════════════════════════════════════════════════════════════════════

def find_optimal_pmns_kpoint(bonds, V_Gamma, triplet_idx_G, n_grid=40):
    """
    For each k, compute U_PMNS(k) = U(k)^dagger . U_TBM and find the k
    that minimizes RMS error against observed PMNS angles.
    """
    print("\n" + "=" * 76)
    print("  6. DIRECT PROBLEM: find k minimizing PMNS RMS error")
    print(f"     Grid: {n_grid}^3 points in BCC BZ")
    print("=" * 76)

    U_TBM = TBM_matrix()

    best_k = None
    best_rms = 1e10
    best_angles = None
    best_U_PMNS = None

    good_results = []

    for i1 in range(n_grid):
        k1 = -0.5 + (i1 + 0.5) / n_grid
        for i2 in range(n_grid):
            k2 = -0.5 + (i2 + 0.5) / n_grid
            for i3 in range(n_grid):
                k3 = -0.5 + (i3 + 0.5) / n_grid

                k = [k1, k2, k3]
                U_k, _ = bloch_mixing_3x3(k, bonds, V_Gamma, triplet_idx_G)

                U_PMNS = np.conj(U_k.T) @ U_TBM
                t12, t23, t13, dcp = extract_angles(U_PMNS)

                rms = math.sqrt(((math.degrees(t12) - 33.44)**2 +
                                  (math.degrees(t23) - 49.20)**2 +
                                  (math.degrees(t13) - 8.57)**2) / 3.0)

                if rms < best_rms:
                    best_rms = rms
                    best_k = np.array(k)
                    best_angles = (t12, t23, t13, dcp)
                    best_U_PMNS = U_PMNS.copy()

                if rms < 5.0:
                    good_results.append((np.array(k), rms, t12, t23, t13, dcp))

    t12, t23, t13, dcp = best_angles
    print(f"\n  Best PMNS k-point: ({best_k[0]:.4f}, {best_k[1]:.4f}, {best_k[2]:.4f})")
    print(f"  RMS error: {best_rms:.4f} deg")
    print(f"  theta_12 = {math.degrees(t12):.2f} deg  (obs: 33.44)")
    print(f"  theta_23 = {math.degrees(t23):.2f} deg  (obs: 49.20)")
    print(f"  theta_13 = {math.degrees(t13):.2f} deg  (obs:  8.57)")
    print(f"  delta_CP = {math.degrees(dcp):.2f} deg  (obs: ~230)")
    print(f"  J        = {jarlskog(best_U_PMNS):.6f}  (obs: 0.033)")

    if len(good_results) > 0:
        print(f"\n  {len(good_results)} k-points with RMS < 5 deg:")
        good_results.sort(key=lambda x: x[1])
        for kk, rr, t12_, t23_, t13_, dcp_ in good_results[:15]:
            print(f"    k=({kk[0]:+.4f}, {kk[1]:+.4f}, {kk[2]:+.4f})  "
                  f"RMS={rr:.2f}  "
                  f"({math.degrees(t12_):.1f}, {math.degrees(t23_):.1f}, {math.degrees(t13_):.1f})")

    return best_k, best_rms, best_angles, best_U_PMNS, good_results


# ═══════════════════════════════════════════════════════════════════════
# 7. REFINE OPTIMAL k WITH FINER GRID
# ═══════════════════════════════════════════════════════════════════════

def refine_kpoint(k_center, bonds, V_Gamma, triplet_idx_G, radius=0.05, n_fine=20, mode='pmns'):
    """Refine around k_center with a fine grid."""
    U_TBM = TBM_matrix()

    best_k = k_center.copy()
    best_score = 1e10
    best_data = None

    for i1 in range(n_fine):
        dk1 = -radius + 2*radius * i1 / (n_fine - 1)
        for i2 in range(n_fine):
            dk2 = -radius + 2*radius * i2 / (n_fine - 1)
            for i3 in range(n_fine):
                dk3 = -radius + 2*radius * i3 / (n_fine - 1)

                k = k_center + np.array([dk1, dk2, dk3])
                U_k, _ = bloch_mixing_3x3(k, bonds, V_Gamma, triplet_idx_G)

                if mode == 'pmns':
                    U_PMNS = np.conj(U_k.T) @ U_TBM
                    t12, t23, t13, dcp = extract_angles(U_PMNS)
                    score = math.sqrt(((math.degrees(t12) - 33.44)**2 +
                                       (math.degrees(t23) - 49.20)**2 +
                                       (math.degrees(t13) - 8.57)**2) / 3.0)
                    data = (t12, t23, t13, dcp, U_PMNS)
                else:
                    score = 0  # placeholder

                if score < best_score:
                    best_score = score
                    best_k = k.copy()
                    best_data = data

    return best_k, best_score, best_data


# ═══════════════════════════════════════════════════════════════════════
# 8. CHECK ALGEBRAIC RELATIONS OF OPTIMAL k-POINT
# ═══════════════════════════════════════════════════════════════════════

def check_kpoint_relations(k_opt, k_koide=None):
    """Check if k_opt relates to known framework quantities."""
    print("\n" + "=" * 76)
    print("  8. ALGEBRAIC RELATIONS OF OPTIMAL k-POINT")
    print("=" * 76)

    k = k_opt
    print(f"\n  k_opt = ({k[0]:.6f}, {k[1]:.6f}, {k[2]:.6f})")

    # BCC reciprocal lattice vectors (for a=1 conventional)
    # b1 = 2pi(0,1,1), b2 = 2pi(1,0,1), b3 = 2pi(1,1,0) [for BCC with a=1]
    # In fractional coords, high-sym points:
    P = np.array([0.25, 0.25, 0.25])
    H = np.array([0.5, -0.5, 0.5])
    N = np.array([0.0, 0.0, 0.5])
    G = np.array([0.0, 0.0, 0.0])

    # Reciprocal metric for BCC
    # B = 2pi * A^{-T} where A = A_PRIM
    B = 2 * np.pi * la.inv(A_PRIM).T

    def cart_dist(k1_frac, k2_frac):
        dk = k1_frac - k2_frac
        dk_cart = dk @ B
        return la.norm(dk_cart)

    def frac_dist(k1_frac, k2_frac):
        return la.norm(k1_frac - k2_frac)

    # Framework constants
    lambda1 = 2 - math.sqrt(3)  # spectral gap
    delta = 2.0 / 9.0            # Koide phase
    g = 10                        # trivalent constant
    kstar = 3                     # optimal degree
    alpha1_val = (2.0/3.0)**8     # ~ 0.039

    print(f"\n  Framework constants:")
    print(f"    lambda_1 = 2 - sqrt(3) = {lambda1:.6f}")
    print(f"    delta = 2/9 = {delta:.6f}")
    print(f"    g = {g}")
    print(f"    k* = {kstar}")
    print(f"    alpha_1 = (2/3)^8 = {alpha1_val:.6f}")

    # (a) Distance from P
    d_P_frac = frac_dist(k, P)
    d_P_cart = cart_dist(k, P)
    print(f"\n  (a) Distance from P = (1/4, 1/4, 1/4):")
    print(f"      Fractional: {d_P_frac:.6f}")
    print(f"      Cartesian:  {d_P_cart:.6f}")
    print(f"      lambda_1 = {lambda1:.6f}  {'CLOSE' if abs(d_P_frac - lambda1) < 0.03 else ''}")
    print(f"      2/9      = {delta:.6f}    {'CLOSE' if abs(d_P_frac - delta) < 0.03 else ''}")

    # (b) Distance from Gamma
    d_G_frac = frac_dist(k, G)
    d_G_cart = cart_dist(k, G)
    print(f"\n  (b) Distance from Gamma:")
    print(f"      Fractional: {d_G_frac:.6f}")
    print(f"      Cartesian:  {d_G_cart:.6f}")
    print(f"      1/g = {1.0/g:.6f}       {'CLOSE' if abs(d_G_frac - 1.0/g) < 0.03 else ''}")

    # (c) On the Gamma-P line? k = t * P for some t
    # Check if k is proportional to (1,1,1) direction
    if abs(P[0]) > 1e-10:
        t_values = k / P
        t_spread = np.std(t_values)
        t_mean = np.mean(t_values)
        on_GP = t_spread < 0.05
        print(f"\n  (c) On Gamma-P line?")
        print(f"      t = k/P component-wise: ({t_values[0]:.4f}, {t_values[1]:.4f}, {t_values[2]:.4f})")
        print(f"      Spread: {t_spread:.4f}  {'YES - on GP line' if on_GP else 'NO'}")
        if on_GP:
            print(f"      t = {t_mean:.6f}")
            print(f"      delta = 2/9 = {delta:.6f}  {'MATCH' if abs(t_mean - delta) < 0.02 else ''}")
            print(f"      4*delta = 8/9 = {4*delta:.6f}  {'MATCH' if abs(t_mean - 4*delta) < 0.02 else ''}")

    # (d) On P-N line? k = P + t*(N-P)
    PN = N - P
    if la.norm(PN) > 1e-10:
        # k - P = t * PN => t = (k-P) . PN / |PN|^2
        dk = k - P
        t_PN = np.dot(dk, PN) / np.dot(PN, PN)
        residual = la.norm(dk - t_PN * PN)
        on_PN = residual < 0.05
        print(f"\n  (d) On P-N line?")
        print(f"      t = {t_PN:.6f}, residual = {residual:.6f}  {'YES' if on_PN else 'NO'}")

    # (e) On P-H line? k = P + t*(H-P)
    PH = H - P
    if la.norm(PH) > 1e-10:
        dk = k - P
        t_PH = np.dot(dk, PH) / np.dot(PH, PH)
        residual = la.norm(dk - t_PH * PH)
        on_PH = residual < 0.05
        print(f"\n  (e) On P-H line?")
        print(f"      t = {t_PH:.6f}, residual = {residual:.6f}  {'YES' if on_PH else 'NO'}")

    # (f) Check specific ratios of k components
    print(f"\n  (f) Component ratios:")
    if abs(k[0]) > 1e-10:
        print(f"      k2/k1 = {k[1]/k[0]:.6f}")
        print(f"      k3/k1 = {k[2]/k[0]:.6f}")
    if abs(k[1]) > 1e-10:
        print(f"      k3/k2 = {k[2]/k[1]:.6f}")
    print(f"      k1+k2+k3 = {k[0]+k[1]+k[2]:.6f}")
    print(f"      |k|_frac  = {la.norm(k):.6f}")

    # (g) Check: does k relate to the previously found point (0.120, 0.040, 0.160)?
    k_prev = np.array([0.120, 0.040, 0.160])
    d_prev = frac_dist(k, k_prev)
    print(f"\n  (g) Distance from previously found (0.120, 0.040, 0.160): {d_prev:.6f}")

    # (h) Check: ratio 0.20 for P+H weighting
    print(f"\n  (h) Weight analysis:")
    print(f"      1/5 = 0.200")
    print(f"      alpha_1 = (2/3)^8 = {alpha1_val:.6f}")
    print(f"      1 - Omega_DM/Omega_m = 1 - 0.842 = 0.158")
    print(f"      lambda_1 = {lambda1:.6f}")
    print(f"      delta = {delta:.6f}")
    print(f"      k*/(k*+1) = {kstar/(kstar+1):.6f}")
    print(f"      1/(k*+2) = {1.0/(kstar+2):.6f} = 0.200  <-- 1/5 = 1/(k*+2)")

    # (i) If koide k-point available
    if k_koide is not None:
        d_koide_opt = frac_dist(k, k_koide)
        print(f"\n  (i) Distance between Koide k-point and optimal PMNS k-point: {d_koide_opt:.6f}")


# ═══════════════════════════════════════════════════════════════════════
# 9. CHECK 80/20 P+H WEIGHTING
# ═══════════════════════════════════════════════════════════════════════

def check_weighted_ph(bonds, V_Gamma, triplet_idx_G):
    """
    Check whether the 80/20 P+H weighting arises from the Koide structure.
    Scan alpha in [0,1] for P_weight = 1-alpha, H_weight = alpha.
    """
    print("\n" + "=" * 76)
    print("  9. P+H WEIGHTING SCAN")
    print("=" * 76)

    U_TBM = TBM_matrix()

    U_P, _ = bloch_mixing_3x3([0.25, 0.25, 0.25], bonds, V_Gamma, triplet_idx_G)
    U_H, _ = bloch_mixing_3x3([0.5, -0.5, 0.5], bonds, V_Gamma, triplet_idx_G)

    best_alpha = None
    best_rms = 1e10
    results = []

    for i in range(1001):
        alpha = i / 1000.0
        U_avg = (1 - alpha) * U_P + alpha * U_H
        # Project to unitary
        Usvd, S, Vhsvd = la.svd(U_avg)
        U_l = Usvd @ Vhsvd

        U_PMNS = np.conj(U_l.T) @ U_TBM
        t12, t23, t13, dcp = extract_angles(U_PMNS)
        rms = math.sqrt(((math.degrees(t12) - 33.44)**2 +
                          (math.degrees(t23) - 49.20)**2 +
                          (math.degrees(t13) - 8.57)**2) / 3.0)

        results.append((alpha, rms, t12, t23, t13, dcp))

        if rms < best_rms:
            best_rms = rms
            best_alpha = alpha

    r = [x for x in results if abs(x[0] - best_alpha) < 1e-6][0]
    print(f"\n  Best H-weight alpha = {best_alpha:.4f}  (P = {1-best_alpha:.4f})")
    print(f"  RMS error = {best_rms:.4f} deg")
    print(f"  theta_12 = {math.degrees(r[2]):.2f}  theta_23 = {math.degrees(r[3]):.2f}  "
          f"theta_13 = {math.degrees(r[4]):.2f}")

    # Check nearby rational values
    delta = 2.0 / 9.0
    lambda1 = 2 - math.sqrt(3)
    candidates = {
        '1/5': 0.2,
        '2/9': delta,
        '2-sqrt(3)': lambda1,
        '1/(k*+2)': 1.0/5.0,
        'sqrt(2)-1': math.sqrt(2) - 1,
        '1/4': 0.25,
        '1/6': 1.0/6.0,
        '(2/3)^3': (2.0/3.0)**3,
    }

    print(f"\n  Nearby rational candidates for alpha:")
    for name, val in sorted(candidates.items(), key=lambda x: abs(x[1] - best_alpha)):
        rr = [x for x in results if abs(x[0] - round(val*1000)/1000) < 1e-4]
        if rr:
            rms_at = rr[0][1]
        else:
            rms_at = float('nan')
        marker = " <--" if abs(val - best_alpha) < 0.015 else ""
        print(f"    {name:20s} = {val:.6f}  RMS = {rms_at:.4f} deg  "
              f"|alpha - val| = {abs(val - best_alpha):.4f}{marker}")

    # Plot the RMS vs alpha curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    alphas = [r[0] for r in results]
    rmses = [r[1] for r in results]
    ax.plot(alphas, rmses, 'b-', linewidth=1.5)
    ax.axhline(y=best_rms, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=best_alpha, color='r', linestyle='--', alpha=0.5, label=f'Best: alpha={best_alpha:.3f}')
    for name, val in candidates.items():
        if 0 < val < 0.5:
            ax.axvline(x=val, color='gray', linestyle=':', alpha=0.3)
            ax.text(val, max(rmses)*0.95, name, rotation=90, fontsize=7, ha='right')
    ax.set_xlabel('H weight (alpha)')
    ax.set_ylabel('RMS angle error (deg)')
    ax.set_title('PMNS RMS error vs P+H weighting')
    ax.legend()
    ax.set_xlim(0, 0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'srs_pmns_koide_ph_weight.png'), dpi=150)
    print(f"\n  Saved: srs_pmns_koide_ph_weight.png")

    return best_alpha, best_rms


# ═══════════════════════════════════════════════════════════════════════
# 10. EFFECTIVE k-POINT OF THE P+H MIXTURE
# ═══════════════════════════════════════════════════════════════════════

def effective_kpoint_of_mixture(bonds, V_Gamma, triplet_idx_G, alpha_H):
    """
    The P+H weighted average gives some U_l. Find which single k-point
    on the lattice gives the closest U(k) to this mixture.
    """
    print("\n" + "=" * 76)
    print(f"  10. EFFECTIVE k-POINT OF {1-alpha_H:.0%}P + {alpha_H:.0%}H MIXTURE")
    print("=" * 76)

    U_P, _ = bloch_mixing_3x3([0.25, 0.25, 0.25], bonds, V_Gamma, triplet_idx_G)
    U_H, _ = bloch_mixing_3x3([0.5, -0.5, 0.5], bonds, V_Gamma, triplet_idx_G)

    U_avg = (1 - alpha_H) * U_P + alpha_H * U_H
    Usvd, S, Vhsvd = la.svd(U_avg)
    U_target = Usvd @ Vhsvd
    target_sq = np.abs(U_target)**2

    print(f"\n  |U_target|^2 (from P+H mixture):")
    for i in range(3):
        row = [abs(U_target[i, j])**2 for j in range(3)]
        print(f"    [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")

    # Scan to find equivalent single k-point
    n_grid = 30
    best_k = None
    best_err = 1e10

    for i1 in range(n_grid):
        k1 = -0.5 + (i1 + 0.5) / n_grid
        for i2 in range(n_grid):
            k2 = -0.5 + (i2 + 0.5) / n_grid
            for i3 in range(n_grid):
                k3 = -0.5 + (i3 + 0.5) / n_grid
                k = [k1, k2, k3]
                U_k, _ = bloch_mixing_3x3(k, bonds, V_Gamma, triplet_idx_G)
                err = la.norm(np.abs(U_k)**2 - target_sq)
                if err < best_err:
                    best_err = err
                    best_k = np.array(k)

    print(f"\n  Closest single k-point: ({best_k[0]:.4f}, {best_k[1]:.4f}, {best_k[2]:.4f})")
    print(f"  ||U(k)|^2 - |U_target|^2||_F = {best_err:.6f}")

    return best_k


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 76)
    print("  PMNS FROM KOIDE MASS MATRIX + SRS BLOCH STRUCTURE")
    print("  Principled k-point derivation")
    print("=" * 76)

    bonds = find_bonds()
    print(f"\n  Found {len(bonds)} bonds.")

    # Step 1: Koide mass matrix
    M_l, m_evals, U_l_koide = koide_mass_matrix()

    # Step 2: PMNS from Koide
    U_PMNS_koide, t12_k, t23_k, t13_k, dcp_k = pmns_from_koide(U_l_koide)

    # Step 3: Get Gamma reference
    V_Gamma, triplet_idx_G = get_gamma_reference(bonds)

    # Step 4: Check high-symmetry points
    print("\n" + "=" * 76)
    print("  4. BLOCH MIXING AT HIGH-SYMMETRY POINTS")
    print("=" * 76)

    U_TBM = TBM_matrix()
    for name, kpt in K_POINTS.items():
        U_k, uerr = bloch_mixing_3x3(kpt, bonds, V_Gamma, triplet_idx_G)
        U_PMNS = np.conj(U_k.T) @ U_TBM
        t12, t23, t13, dcp = extract_angles(U_PMNS)
        rms = math.sqrt(((math.degrees(t12) - 33.44)**2 +
                          (math.degrees(t23) - 49.20)**2 +
                          (math.degrees(t13) - 8.57)**2) / 3.0)
        print(f"\n  {name:6s}: theta = ({math.degrees(t12):.1f}, {math.degrees(t23):.1f}, "
              f"{math.degrees(t13):.1f})  RMS = {rms:.2f} deg  unitarity = {uerr:.2e}")

    # Step 5: Inverse problem (coarse)
    print("\n  [Running coarse BZ scan for Koide k-point...]")
    k_koide, err_koide, U_koide_match, koide_results = find_koide_kpoint(
        bonds, V_Gamma, triplet_idx_G, U_l_koide, n_grid=30)

    # Step 6: Direct problem (coarse)
    print("\n  [Running coarse BZ scan for optimal PMNS k-point...]")
    k_pmns, rms_pmns, angles_pmns, U_PMNS_opt, pmns_results = find_optimal_pmns_kpoint(
        bonds, V_Gamma, triplet_idx_G, n_grid=30)

    # Refine both
    print("\n  [Refining PMNS k-point...]")
    k_pmns_fine, rms_fine, data_fine = refine_kpoint(
        k_pmns, bonds, V_Gamma, triplet_idx_G, radius=0.03, n_fine=20)

    t12f, t23f, t13f, dcpf, U_PMNS_fine = data_fine
    print(f"\n  Refined PMNS k-point: ({k_pmns_fine[0]:.6f}, {k_pmns_fine[1]:.6f}, {k_pmns_fine[2]:.6f})")
    print(f"  RMS = {rms_fine:.4f} deg")
    print(f"  theta_12 = {math.degrees(t12f):.2f}  theta_23 = {math.degrees(t23f):.2f}  "
          f"theta_13 = {math.degrees(t13f):.2f}")
    print(f"  delta_CP = {math.degrees(dcpf):.2f}  J = {jarlskog(U_PMNS_fine):.6f}")

    # Step 7: Second refinement for high precision
    print("\n  [Second refinement pass...]")
    k_pmns_v2, rms_v2, data_v2 = refine_kpoint(
        k_pmns_fine, bonds, V_Gamma, triplet_idx_G, radius=0.005, n_fine=20)

    t12v, t23v, t13v, dcpv, U_PMNS_v2 = data_v2
    print(f"\n  Final PMNS k-point: ({k_pmns_v2[0]:.6f}, {k_pmns_v2[1]:.6f}, {k_pmns_v2[2]:.6f})")
    print(f"  RMS = {rms_v2:.4f} deg")
    print(f"  theta_12 = {math.degrees(t12v):.4f}  (obs: 33.44)")
    print(f"  theta_23 = {math.degrees(t23v):.4f}  (obs: 49.20)")
    print(f"  theta_13 = {math.degrees(t13v):.4f}  (obs:  8.57)")
    print(f"  delta_CP = {math.degrees(dcpv):.4f}")
    print(f"  J        = {jarlskog(U_PMNS_v2):.8f}")

    # Step 8: Check algebraic relations
    check_kpoint_relations(k_pmns_v2, k_koide)

    # Step 9: P+H weighting scan
    best_alpha_H, best_rms_ph = check_weighted_ph(bonds, V_Gamma, triplet_idx_G)

    # Step 10: Effective k-point of mixture
    k_eff = effective_kpoint_of_mixture(bonds, V_Gamma, triplet_idx_G, best_alpha_H)
    check_kpoint_relations(k_eff)

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print(f"\n  Koide diagonalizer U_l from M_l(delta=2/9, eps=sqrt(2)):")
    print(f"    -> PMNS angles: ({math.degrees(t12_k):.2f}, {math.degrees(t23_k):.2f}, {math.degrees(t13_k):.2f})")
    print(f"\n  Best single k-point for PMNS:")
    print(f"    k = ({k_pmns_v2[0]:.6f}, {k_pmns_v2[1]:.6f}, {k_pmns_v2[2]:.6f})")
    print(f"    RMS error = {rms_v2:.4f} deg")
    print(f"\n  Best P+H weighting:")
    print(f"    alpha_H = {best_alpha_H:.4f} ({best_alpha_H:.4f} H + {1-best_alpha_H:.4f} P)")
    print(f"    RMS error = {best_rms_ph:.4f} deg")
    print(f"\n  Effective k-point of P+H mixture:")
    print(f"    k_eff = ({k_eff[0]:.4f}, {k_eff[1]:.4f}, {k_eff[2]:.4f})")
    print(f"\n  Koide k-point (best match to U_l(Koide)):")
    print(f"    k_Koide = ({k_koide[0]:.4f}, {k_koide[1]:.4f}, {k_koide[2]:.4f})")
    print(f"    match error = {err_koide:.6f}")
    print(f"\n  Previously found k = (0.120, 0.040, 0.160)")
    print(f"    Distance to optimal: {la.norm(k_pmns_v2 - np.array([0.120, 0.040, 0.160])):.4f}")


if __name__ == '__main__':
    main()
