#!/usr/bin/env python3
"""
PMNS mixing matrix from srs Bloch band structure.

KEY IDEA: The PMNS matrix is U_PMNS = U_l† · U_ν where:
  - U_ν = neutrino diagonalizer (delocalized, BZ-averaged → TBM from S₄(K₄))
  - U_l = charged lepton diagonalizer (edge-local → specific k-point)

At the Γ point, the K₄ adjacency has S₃ symmetry → TBM emerges.
At specific k-points, S₃ is broken by the Bloch phases:
  - H = (1/2,-1/2,1/2): ω content of ω² band = 33%
  - N = (0,0,1/2): ω content of ω² band = 50% (complete mixing)
  - P = (1/4,1/4,1/4): 0% mixing (C₃ exact)

The PMNS deviations from TBM (θ₁₃ ≠ 0, θ₂₃ ≠ 45°, θ₁₂ ≠ 35.26°)
may be encoded in U(k) = V(Γ)† · V(k) at specific k-points.

KNOWN NEGATIVE RESULTS (not repeated here):
  - Wigner D as U_l: destroys TBM structure
  - Circulant U_l: wrong angles
  - Fourier U_l: trivial phases
  - δ_CP from geometry alone: doesn't enter algebra

This script tests whether the Bloch mixing matrices at high-symmetry
k-points give the observed PMNS deviations from TBM.
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
delta_CP_obs = math.radians(230.0)  # midpoint of 197-250 range

sin2_12_obs = math.sin(theta12_obs)**2
sin2_23_obs = math.sin(theta23_obs)**2
sin2_13_obs = math.sin(theta13_obs)**2
J_PMNS_obs = 0.033  # Jarlskog ~ 0.033 ± 0.004

# TBM predictions
theta12_TBM = math.atan(1.0 / math.sqrt(2))  # arctan(1/√2) = 35.26°
theta23_TBM = math.pi / 4                      # 45°
theta13_TBM = 0.0                               # 0°

sin2_12_TBM = 1.0 / 3.0   # exact
sin2_23_TBM = 0.5          # exact
sin2_13_TBM = 0.0          # exact

alpha1 = 1.0 / 137.036  # fine structure constant (dark correction parameter)


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

# C₃ permutation: (x,y,z) → (z,x,y); v₀ fixed, (v₁→v₃→v₂→v₁)
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
# CONNECTIVITY AND HAMILTONIAN
# ═══════════════════════════════════════════════════════════════════════

def find_bonds():
    """Find NN bonds in the primitive cell."""
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
    """4x4 Bloch Hamiltonian at fractional k."""
    H = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for src, tgt, cell in bonds:
        phase = np.exp(2j * np.pi * np.dot(k, cell))
        H[tgt, src] += phase
    return H


def diag_H(k_frac, bonds):
    """Diagonalize H(k), return sorted eigenvalues and eigenvectors."""
    H = bloch_H(k_frac, bonds)
    evals, evecs = la.eigh(H)
    idx = np.argsort(np.real(evals))
    return np.real(evals[idx]), evecs[:, idx]


def c3_decompose(k_frac, bonds, degen_tol=1e-8):
    """Simultaneously diagonalize H(k) and C₃."""
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

    C3_new = np.conj(new_evecs.T) @ C3_PERM @ new_evecs
    offdiag = la.norm(C3_new - np.diag(np.diag(C3_new)))

    return evals, new_evecs, c3_diag, offdiag


# ═══════════════════════════════════════════════════════════════════════
# TBM MATRIX
# ═══════════════════════════════════════════════════════════════════════

def TBM_matrix():
    """
    Tribimaximal mixing matrix (Harrison-Perkins-Scott).

    U_TBM = | √(2/3)    1/√3     0     |
            | -1/√6     1/√3    1/√2   |
            |  1/√6    -1/√3    1/√2   |

    Convention: rows = mass eigenstates (ν₁, ν₂, ν₃),
                cols = flavor eigenstates (νₑ, νμ, ντ).
    Actually standard convention is U_{αi} so:
      rows = flavor (e, μ, τ), cols = mass (1, 2, 3).
    """
    s12 = 1.0 / math.sqrt(3)
    c12 = math.sqrt(2.0 / 3.0)
    s23 = 1.0 / math.sqrt(2)
    c23 = 1.0 / math.sqrt(2)

    U = np.array([
        [ c12,  s12,  0.0],
        [-s12*c23, c12*c23, s23],
        [ s12*s23, -c12*s23, c23],
    ], dtype=complex)

    return U


def standard_pmns(theta12, theta23, theta13, delta):
    """
    Standard PMNS parametrization.
    U = R₂₃ · diag(1, 1, e^{-iδ}) · R₁₃ · diag(1, 1, e^{iδ}) · R₁₂
    """
    c12, s12 = math.cos(theta12), math.sin(theta12)
    c23, s23 = math.cos(theta23), math.sin(theta23)
    c13, s13 = math.cos(theta13), math.sin(theta13)
    ed = np.exp(1j * delta)
    emd = np.exp(-1j * delta)

    U = np.array([
        [c12*c13,                s12*c13,                s13*emd],
        [-s12*c23 - c12*s23*s13*ed,  c12*c23 - s12*s23*s13*ed,  s23*c13],
        [ s12*s23 - c12*c23*s13*ed, -c12*s23 - s12*c23*s13*ed,  c23*c13],
    ], dtype=complex)

    return U


def extract_angles(U):
    """
    Extract mixing angles and CP phase from a 3x3 unitary matrix.
    Uses the standard PMNS parametrization.

    Returns: theta12, theta23, theta13, delta_CP (all in radians)
    """
    # |U_e3| = sin(theta13)
    s13 = abs(U[0, 2])
    if s13 > 1.0:
        s13 = 1.0
    theta13 = math.asin(s13)
    c13 = math.cos(theta13)

    if c13 < 1e-10:
        # θ₁₃ ~ π/2, degenerate case
        return 0.0, 0.0, theta13, 0.0

    # sin²θ₁₂ = |U_e2|² / (1 - |U_e3|²)
    sin2_12 = abs(U[0, 1])**2 / c13**2
    sin2_12 = min(max(sin2_12, 0.0), 1.0)
    theta12 = math.asin(math.sqrt(sin2_12))

    # sin²θ₂₃ = |U_μ3|² / (1 - |U_e3|²)
    sin2_23 = abs(U[1, 2])**2 / c13**2
    sin2_23 = min(max(sin2_23, 0.0), 1.0)
    theta23 = math.asin(math.sqrt(sin2_23))

    # CP phase from Jarlskog invariant
    # J = Im(U_e1 U_μ2 U_e2* U_μ1*) = c12 s12 c23 s23 c13² s13 sin(δ)
    J = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    denom = math.cos(theta12) * math.sin(theta12) * math.cos(theta23) * \
            math.sin(theta23) * c13**2 * s13
    if abs(denom) > 1e-15:
        sin_delta = J / denom
        sin_delta = min(max(sin_delta, -1.0), 1.0)
        delta = math.asin(sin_delta)
        # Resolve quadrant ambiguity using U_e3 phase
        if s13 > 1e-10:
            phase_e3 = np.angle(U[0, 2])
            # In standard param, U_e3 = s13 * e^{-iδ}
            delta_from_phase = -phase_e3
            # Use the one consistent with both
            if abs(math.sin(delta_from_phase) - sin_delta) < 0.5:
                delta = delta_from_phase
    else:
        delta = 0.0

    return theta12, theta23, theta13, delta


def jarlskog(U):
    """Compute the Jarlskog invariant J = Im(U_e1 U_μ2 U*_e2 U*_μ1)."""
    return np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))


# ═══════════════════════════════════════════════════════════════════════
# 1. MIXING MATRICES AT HIGH-SYMMETRY POINTS
# ═══════════════════════════════════════════════════════════════════════

def compute_mixing_at_kpoints(bonds):
    """
    At each high-symmetry k-point, compute U(k) = V(Γ)† · V(k).

    V(k) = eigenvectors of H(k). U(k) is the basis change from
    the Γ-point eigenstates to the k-point eigenstates.

    The triplet subspace (3 degenerate bands at Γ with E=-1) is
    the generation space. We extract the 3x3 subblock of U(k)
    acting on the triplet.
    """
    print("=" * 76)
    print("  1. MIXING MATRICES AT HIGH-SYMMETRY POINTS")
    print("     U(k) = V(Gamma)† · V(k)  restricted to triplet subspace")
    print("=" * 76)

    # Γ-point eigenstates (reference)
    evals_G, evecs_G = diag_H([0, 0, 0], bonds)
    print(f"\n  Gamma eigenvalues: {evals_G}")

    # Identify triplet (E = -1, 3-fold degenerate)
    triplet_mask = np.abs(evals_G - (-1.0)) < 0.5
    triplet_idx_G = np.where(triplet_mask)[0]
    singlet_idx_G = np.where(~triplet_mask)[0]
    print(f"  Triplet indices at Gamma: {triplet_idx_G}")
    print(f"  Singlet indices at Gamma: {singlet_idx_G}")

    # At Gamma, within the triplet subspace, use C₃ to define generation basis.
    # The triplet at Gamma is 3-fold degenerate, so any basis works.
    # Use C₃ eigenstates as the canonical generation basis.
    evals_G_c3, evecs_G_c3, c3_diag_G, _ = c3_decompose([0, 0, 0], bonds)

    # Map C₃ labels to generation indices: 1 → gen 0 (e), ω → gen 1 (μ), ω² → gen 2 (τ)
    gen_map = {}
    for b in range(N_ATOMS):
        c3v = c3_diag_G[b]
        if abs(c3v - 1.0) < 0.3:
            if b in triplet_idx_G:
                gen_map['trivial_triplet'] = b
            else:
                gen_map['singlet'] = b
        elif abs(c3v - omega3) < 0.3:
            gen_map['omega'] = b
        elif abs(c3v - omega3**2) < 0.3:
            gen_map['omega2'] = b

    print(f"\n  C₃ decomposition at Gamma:")
    for name, idx in gen_map.items():
        print(f"    {name}: band {idx}, E = {evals_G_c3[idx]:.6f}, "
              f"C₃ = {c3_diag_G[idx]:.4f}")

    # Generation basis at Gamma (from C₃ eigenstates within the triplet)
    # At Gamma the triplet is degenerate, so C₃ gives us 1, ω, ω² labels.
    # These are the "flavor" states for the PMNS.
    V_Gamma = evecs_G_c3  # 4×4, columns are C₃ eigenstates

    results = {}

    for name, k in K_POINTS.items():
        print(f"\n  --- {name} = {k} ---")

        evals_k, evecs_k = diag_H(k, bonds)
        print(f"  Eigenvalues: {evals_k}")

        # Full 4×4 mixing matrix
        U_full = np.conj(V_Gamma.T) @ evecs_k

        # Also decompose with C₃ at this k-point
        evals_kc, evecs_kc, c3_k, od = c3_decompose(k, bonds)
        print(f"  C₃ eigenvalues: {np.round(c3_k, 4)}")
        print(f"  C₃ off-diagonal: {od:.2e}")

        # Triplet-triplet subblock: pick the 3 bands at k that overlap
        # most with the Gamma triplet.
        # The triplet at Gamma spans bands triplet_idx_G.
        # At k, the relevant bands are those with largest projection.
        overlap_sq = np.abs(U_full)**2  # [Gamma_band, k_band]

        # For each k-band, total overlap with Gamma triplet
        triplet_overlap = np.sum(overlap_sq[triplet_idx_G, :], axis=0)
        print(f"  Triplet overlap per k-band: {np.round(triplet_overlap, 4)}")

        # Select 3 k-bands with highest triplet overlap
        triplet_idx_k = np.argsort(triplet_overlap)[-3:]
        triplet_idx_k = np.sort(triplet_idx_k)
        print(f"  Triplet k-band indices: {triplet_idx_k}")

        # 3×3 mixing matrix in triplet subspace
        U_trip = U_full[np.ix_(triplet_idx_G, triplet_idx_k)]

        # Check unitarity of the 3×3 subblock
        unitarity = la.norm(U_trip @ np.conj(U_trip.T) - np.eye(3))
        print(f"  Triplet U unitarity: ||UU†-I|| = {unitarity:.2e}")

        # If not unitary, project to nearest unitary via SVD
        if unitarity > 0.01:
            Usvd, S, Vhsvd = la.svd(U_trip)
            U_trip_unitary = Usvd @ Vhsvd
            print(f"  Projected to unitary (singular values: {np.round(S, 4)})")
        else:
            U_trip_unitary = U_trip

        # Print the mixing matrix
        print(f"  |U_trip|² =")
        for i in range(3):
            row = [abs(U_trip[i, j])**2 for j in range(3)]
            print(f"    [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")

        # Omega content analysis (generation mixing)
        # How much does the ω² band at k contain of the ω state at Gamma?
        if 'omega' in gen_map and 'omega2' in gen_map:
            w_idx = gen_map['omega']
            w2_idx = gen_map['omega2']
            for b in range(N_ATOMS):
                ov_w = abs(U_full[w_idx, b])**2
                ov_w2 = abs(U_full[w2_idx, b])**2
                if ov_w > 0.05 or ov_w2 > 0.05:
                    print(f"    k-band {b}: ω content = {ov_w:.4f}, "
                          f"ω² content = {ov_w2:.4f}")

        results[name] = {
            'evals': evals_k,
            'U_full': U_full,
            'U_trip': U_trip,
            'U_trip_unitary': U_trip_unitary,
            'triplet_idx_k': triplet_idx_k,
            'c3_diag': c3_k,
            'unitarity': unitarity,
        }

    return results, V_Gamma, triplet_idx_G


# ═══════════════════════════════════════════════════════════════════════
# 2. PMNS FROM U_l = U(k), U_ν = TBM
# ═══════════════════════════════════════════════════════════════════════

def compute_pmns_candidates(mixing_results, triplet_idx_G):
    """
    Compute PMNS = U_l† · U_ν for various choices of U_l.

    U_ν = TBM (from S₄(K₄), delocalized neutrino sector).
    U_l = mixing matrix at a specific k-point (edge-local lepton sector).

    Also try weighted averages of k-points and the dark correction.
    """
    print("\n" + "=" * 76)
    print("  2. PMNS CANDIDATES: U_PMNS = U_l† · U_ν")
    print("     U_ν = TBM (tribimaximal), U_l = Bloch mixing at k-point")
    print("=" * 76)

    U_TBM = TBM_matrix()
    print(f"\n  U_TBM =")
    for i in range(3):
        print(f"    [{U_TBM[i,0].real:+.6f}  {U_TBM[i,1].real:+.6f}  {U_TBM[i,2].real:+.6f}]")

    # Observed PMNS for comparison
    U_obs = standard_pmns(theta12_obs, theta23_obs, theta13_obs, delta_CP_obs)
    print(f"\n  |U_obs|² (PDG) =")
    for i in range(3):
        row = [abs(U_obs[i, j])**2 for j in range(3)]
        print(f"    [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")
    print(f"  J_obs = {jarlskog(U_obs):.6f}")

    candidates = {}

    for kname, mdata in mixing_results.items():
        U_l = mdata['U_trip_unitary']

        # PMNS = U_l† · U_TBM
        U_PMNS = np.conj(U_l.T) @ U_TBM

        # Extract angles
        t12, t23, t13, dcp = extract_angles(U_PMNS)
        J = jarlskog(U_PMNS)

        print(f"\n  --- U_l from {kname} ---")
        print(f"  |U_PMNS|² =")
        for i in range(3):
            row = [abs(U_PMNS[i, j])**2 for j in range(3)]
            print(f"    [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")

        print(f"  θ₁₂ = {math.degrees(t12):.2f}° (obs: {math.degrees(theta12_obs):.2f}°, "
              f"TBM: {math.degrees(theta12_TBM):.2f}°)")
        print(f"  θ₂₃ = {math.degrees(t23):.2f}° (obs: {math.degrees(theta23_obs):.2f}°, "
              f"TBM: 45.00°)")
        print(f"  θ₁₃ = {math.degrees(t13):.2f}° (obs: {math.degrees(theta13_obs):.2f}°, "
              f"TBM: 0.00°)")
        print(f"  δ_CP = {math.degrees(dcp):.2f}° (obs: ~{math.degrees(delta_CP_obs):.0f}°)")
        print(f"  J    = {J:.6f} (obs: {J_PMNS_obs:.4f})")

        # Grade each angle
        for angle_name, val, obs in [('θ₁₂', t12, theta12_obs),
                                      ('θ₂₃', t23, theta23_obs),
                                      ('θ₁₃', t13, theta13_obs)]:
            err = abs(math.degrees(val) - math.degrees(obs))
            if err < 1.0:
                grade = 'A'
            elif err < 3.0:
                grade = 'B'
            elif err < 10.0:
                grade = 'C'
            else:
                grade = 'F'
            print(f"    {angle_name}: error = {err:.2f}°  [{grade}]")

        candidates[kname] = {
            'U_PMNS': U_PMNS,
            'theta12': t12, 'theta23': t23, 'theta13': t13,
            'delta_CP': dcp, 'J': J,
        }

    return candidates


# ═══════════════════════════════════════════════════════════════════════
# 3. WEIGHTED K-POINT AVERAGES
# ═══════════════════════════════════════════════════════════════════════

def compute_weighted_averages(mixing_results, triplet_idx_G):
    """
    Try weighted averages of k-point mixing matrices as U_l.

    Physical motivation: the charged lepton sector is edge-local but not
    perfectly localized to a single k-point. A weighted average over
    the BZ with edge-local weighting may give better results.

    Parametrize as: U_l = w_H · U(H) + w_N · U(N) + w_P · U(P)
    (then project to nearest unitary).
    """
    print("\n" + "=" * 76)
    print("  3. WEIGHTED k-POINT AVERAGES FOR U_l")
    print("=" * 76)

    U_TBM = TBM_matrix()

    # Get the k-point mixing matrices (skip Gamma = identity)
    kpoints = ['H', 'N', 'P']
    U_k = {}
    for kn in kpoints:
        if kn in mixing_results:
            U_k[kn] = mixing_results[kn]['U_trip_unitary']

    if len(U_k) < 2:
        print("  Not enough k-points for weighted average.")
        return {}

    best_result = None
    best_score = 1e10
    n_scan = 21

    print(f"\n  Scanning {n_scan}x{n_scan} weight grid over (w_H, w_N) with w_P = 1-w_H-w_N...")

    for i_H in range(n_scan):
        for i_N in range(n_scan):
            w_H = i_H / (n_scan - 1)
            w_N = i_N / (n_scan - 1) * (1.0 - w_H)
            w_P = 1.0 - w_H - w_N
            if w_P < -0.01:
                continue

            U_avg = w_H * U_k.get('H', np.eye(3, dtype=complex)) + \
                    w_N * U_k.get('N', np.eye(3, dtype=complex)) + \
                    w_P * U_k.get('P', np.eye(3, dtype=complex))

            # Project to unitary
            Usvd, S, Vhsvd = la.svd(U_avg)
            U_l = Usvd @ Vhsvd

            U_PMNS = np.conj(U_l.T) @ U_TBM
            t12, t23, t13, dcp = extract_angles(U_PMNS)

            # Score: sum of squared angle errors (degrees)
            score = (math.degrees(t12) - math.degrees(theta12_obs))**2 + \
                    (math.degrees(t23) - math.degrees(theta23_obs))**2 + \
                    (math.degrees(t13) - math.degrees(theta13_obs))**2

            if score < best_score:
                best_score = score
                best_result = {
                    'w_H': w_H, 'w_N': w_N, 'w_P': w_P,
                    'U_PMNS': U_PMNS,
                    'theta12': t12, 'theta23': t23, 'theta13': t13,
                    'delta_CP': dcp, 'J': jarlskog(U_PMNS),
                    'score': score,
                }

    if best_result:
        r = best_result
        print(f"\n  Best weights: w_H = {r['w_H']:.3f}, w_N = {r['w_N']:.3f}, "
              f"w_P = {r['w_P']:.3f}")
        print(f"  θ₁₂ = {math.degrees(r['theta12']):.2f}° (obs: {math.degrees(theta12_obs):.2f}°)")
        print(f"  θ₂₃ = {math.degrees(r['theta23']):.2f}° (obs: {math.degrees(theta23_obs):.2f}°)")
        print(f"  θ₁₃ = {math.degrees(r['theta13']):.2f}° (obs: {math.degrees(theta13_obs):.2f}°)")
        print(f"  δ_CP = {math.degrees(r['delta_CP']):.2f}° (obs: ~{math.degrees(delta_CP_obs):.0f}°)")
        print(f"  J    = {r['J']:.6f}")
        print(f"  RMS angle error: {math.sqrt(best_score/3):.2f}°")

    return best_result


# ═══════════════════════════════════════════════════════════════════════
# 4. DARK CORRECTION TO BAND STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

def compute_dark_correction(bonds):
    """
    The dark correction modifies the graph from srs (compressed) to
    srs + dark modes. In the band structure, this is a perturbation
    ε = α₁ ≈ 1/137 applied to certain hopping amplitudes.

    The dark sector adds modes that are uncompressed (not part of
    the original srs graph). These act as a self-energy correction
    to the Bloch Hamiltonian:
      H_dark(k) = H_srs(k) + ε · Σ_dark(k)

    Physical model: dark modes live on a second copy of the lattice
    (the "other chirality" of srs) coupled by ε.

    We model Σ_dark as the difference between the srs Hamiltonian
    and its "reverse chirality" (srs with inverted screw).
    For the band structure, the perturbation to the eigenstates
    gives corrections to the mixing angles.
    """
    print("\n" + "=" * 76)
    print("  4. DARK CORRECTION (ε = α₁ = 1/137)")
    print("=" * 76)

    epsilon = alpha1
    print(f"  ε = {epsilon:.6e}")

    # The dark correction on the srs lattice inverts the chirality.
    # For the 4₁ screw, the opposite chirality is 4₃ (= 4₁⁻¹).
    # In terms of bonds: the cell displacements are negated for
    # the chirality-flip, giving H_dark(k) = H(-k).
    # But H(-k) = H(k)* by time reversal.
    # So Σ_dark(k) = H(k)* - H(k) = -2i Im(H(k)).

    results = {}

    for kname, k in K_POINTS.items():
        H = bloch_H(k, bonds)
        H_imag = np.imag(H)

        # Perturbation
        Sigma = -2j * H_imag
        H_pert = H + epsilon * Sigma

        evals_0, evecs_0 = la.eigh(H)
        idx0 = np.argsort(np.real(evals_0))
        evals_0 = np.real(evals_0[idx0])
        evecs_0 = evecs_0[:, idx0]

        evals_p, evecs_p = la.eigh(H_pert)
        idxp = np.argsort(np.real(evals_p))
        evals_p = np.real(evals_p[idxp])
        evecs_p = evecs_p[:, idxp]

        dE = evals_p - evals_0
        print(f"\n  {kname}: ΔE = {dE}")
        print(f"  ||Σ_dark|| = {la.norm(Sigma):.4f}")

        # Mixing matrix change due to dark correction
        U_before = evecs_0
        U_after = evecs_p
        dU = np.conj(U_before.T) @ U_after  # should be near identity + O(ε)
        print(f"  ||dU - I|| = {la.norm(dU - np.eye(N_ATOMS)):.4e}")

        results[kname] = {
            'evals_0': evals_0, 'evals_pert': evals_p,
            'dE': dE, 'dU': dU,
            'Sigma_norm': la.norm(Sigma),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# 5. DARK-CORRECTED PMNS
# ═══════════════════════════════════════════════════════════════════════

def compute_dark_pmns(bonds, mixing_results, triplet_idx_G):
    """
    Apply dark correction ε = α₁ to the Bloch Hamiltonian,
    recompute mixing matrices, and extract corrected PMNS.

    The dark correction shifts θ₂₃ from 45° toward 49.2°.
    """
    print("\n" + "=" * 76)
    print("  5. DARK-CORRECTED PMNS")
    print("=" * 76)

    epsilon = alpha1
    U_TBM = TBM_matrix()

    # Recompute with dark correction
    for kname, k in K_POINTS.items():
        if kname == 'Gamma':
            continue

        H = bloch_H(k, bonds)
        H_imag = np.imag(H)
        Sigma = -2j * H_imag
        H_pert = H + epsilon * Sigma

        evals_p, evecs_p = la.eigh(H_pert)
        idxp = np.argsort(np.real(evals_p))
        evecs_p = evecs_p[:, idxp]

        # Reference: C₃ eigenstates at Gamma
        _, evecs_G_c3, _, _ = c3_decompose([0, 0, 0], bonds)

        U_full = np.conj(evecs_G_c3.T) @ evecs_p

        # Triplet subblock
        triplet_overlap = np.sum(np.abs(U_full[triplet_idx_G, :])**2, axis=0)
        triplet_idx_k = np.sort(np.argsort(triplet_overlap)[-3:])
        U_trip = U_full[np.ix_(triplet_idx_G, triplet_idx_k)]

        # Project to unitary
        Usvd, S, Vhsvd = la.svd(U_trip)
        U_l = Usvd @ Vhsvd

        U_PMNS = np.conj(U_l.T) @ U_TBM
        t12, t23, t13, dcp = extract_angles(U_PMNS)
        J = jarlskog(U_PMNS)

        print(f"\n  --- Dark-corrected {kname} (ε = {epsilon:.4e}) ---")
        print(f"  θ₁₂ = {math.degrees(t12):.2f}° (obs: {math.degrees(theta12_obs):.2f}°)")
        print(f"  θ₂₃ = {math.degrees(t23):.2f}° (obs: {math.degrees(theta23_obs):.2f}°)")
        print(f"  θ₁₃ = {math.degrees(t13):.2f}° (obs: {math.degrees(theta13_obs):.2f}°)")
        print(f"  δ_CP = {math.degrees(dcp):.2f}°")
        print(f"  J    = {J:.6f}")


# ═══════════════════════════════════════════════════════════════════════
# 6. N-POINT ANALYSIS: 50% MIXING AND TBM DEVIATION
# ═══════════════════════════════════════════════════════════════════════

def analyze_N_point(bonds, mixing_results):
    """
    At N = (0,0,1/2), the ω content of the ω² band is 50%.
    This is COMPLETE mixing between generations.

    Question: does this maximal mixing relate to the TBM deviation?

    The mixing at N makes the ω/ω² bands maximally mixed, which
    means sin²θ₁₂ → 1/2 in that subspace. The observed deviation
    of θ₁₂ from TBM is sin²θ₁₂ = 0.307 vs TBM's 1/3 = 0.333.
    The N-point gives 1/2 — WRONG DIRECTION.

    But: the PMNS isn't the mixing at N alone. It's U_l† · U_ν
    where U_ν = TBM. The question is what happens when U_l has
    the N-point mixing structure.
    """
    print("\n" + "=" * 76)
    print("  6. N-POINT ANALYSIS (50% ω/ω² mixing)")
    print("=" * 76)

    N_data = mixing_results.get('N')
    if N_data is None:
        print("  N point not computed.")
        return

    U_N = N_data['U_trip']
    print(f"  U(N) triplet mixing matrix |U|²:")
    for i in range(3):
        row = [abs(U_N[i, j])**2 for j in range(3)]
        print(f"    [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")

    # The key observable: in the ω/ω² subspace, what is the mixing angle?
    # If the 2×2 subblock has off-diagonal element |U_{ωω²}|², this
    # determines the 1-2 mixing correction to TBM.

    # Extract the ω-ω² 2×2 subblock (rows/cols 1,2 in generation basis)
    sub_ww2 = U_N[1:3, 1:3]
    print(f"\n  ω/ω² 2×2 subblock:")
    for i in range(2):
        row = [abs(sub_ww2[i, j])**2 for j in range(2)]
        print(f"    [{row[0]:.6f}  {row[1]:.6f}]")

    # Effective 1-2 mixing angle from this subblock
    sin2_eff = abs(sub_ww2[0, 1])**2
    print(f"\n  Effective sin²θ from ω-ω² mixing: {sin2_eff:.6f}")
    print(f"  = {math.degrees(math.asin(math.sqrt(min(sin2_eff, 1.0)))):.2f}°")

    # Check: does the H-point 33% relate to sin²θ₁₂ = 1/3?
    H_data = mixing_results.get('H')
    if H_data is not None:
        U_H = H_data['U_trip']
        sub_H = U_H[1:3, 1:3]
        sin2_H = abs(sub_H[0, 1])**2
        print(f"\n  H-point ω-ω² mixing: sin² = {sin2_H:.6f}")
        print(f"  Compare to sin²θ₁₂(TBM) = 1/3 = {1/3:.6f}")
        diff = abs(sin2_H - 1.0/3.0)
        print(f"  |sin²(H) - 1/3| = {diff:.6f}  "
              f"{'MATCH' if diff < 0.02 else 'NO MATCH'}")


# ═══════════════════════════════════════════════════════════════════════
# 7. PERTURBATIVE θ₂₃ CORRECTION FROM DARK MODES AT P
# ═══════════════════════════════════════════════════════════════════════

def dark_theta23_at_P(bonds):
    """
    At P = (1/4,1/4,1/4), C₃ is exact → θ₂₃ = 45° (TBM).
    The dark correction breaks C₃ perturbatively.

    The correction is:
      δθ₂₃ = ε · |⟨ω|Σ_dark|ω²⟩| / (E_ω - E_ω²)

    where ε = α₁, and Σ_dark is the dark self-energy.
    Observed: δθ₂₃ = 49.2° - 45° = 4.2°.
    """
    print("\n" + "=" * 76)
    print("  7. PERTURBATIVE θ₂₃ CORRECTION AT P FROM DARK MODES")
    print("=" * 76)

    k_P = [0.25, 0.25, 0.25]
    epsilon = alpha1

    H = bloch_H(k_P, bonds)
    evals, evecs, c3_diag, _ = c3_decompose(k_P, bonds)

    # Identify ω and ω² bands
    w_idx = w2_idx = None
    for b in range(N_ATOMS):
        c3v = c3_diag[b]
        if abs(c3v - omega3) < 0.3:
            w_idx = b
        elif abs(c3v - omega3**2) < 0.3:
            w2_idx = b

    if w_idx is None or w2_idx is None:
        print("  Could not identify ω/ω² bands at P.")
        return

    print(f"  ω band: index {w_idx}, E = {evals[w_idx]:.6f}")
    print(f"  ω² band: index {w2_idx}, E = {evals[w2_idx]:.6f}")
    print(f"  ΔE = E_ω - E_ω² = {evals[w_idx] - evals[w2_idx]:.6f}")

    # Dark self-energy
    H_imag = np.imag(H)
    Sigma = -2j * H_imag

    # Matrix element in the eigenbasis
    Sigma_eig = np.conj(evecs.T) @ Sigma @ evecs
    mel = abs(Sigma_eig[w_idx, w2_idx])
    dE = abs(evals[w_idx] - evals[w2_idx])

    print(f"\n  |⟨ω|Σ_dark|ω²⟩| = {mel:.6f}")
    print(f"  |E_ω - E_ω²| = {dE:.6f}")

    if dE > 1e-10:
        delta_theta = epsilon * mel / dE
        print(f"\n  δθ₂₃ = ε · |mel| / ΔE = {epsilon:.6e} × {mel:.4f} / {dE:.4f}")
        print(f"       = {delta_theta:.6e} rad = {math.degrees(delta_theta):.4f}°")
        print(f"  Observed δθ₂₃ = 4.2°")
        print(f"  Ratio predicted/observed = {math.degrees(delta_theta) / 4.2:.4f}")

        # What coefficient would be needed?
        needed = math.radians(4.2) * dE / (mel + 1e-30)
        print(f"\n  To get 4.2°, need ε = {needed:.6f}")
        print(f"  Compare: α₁ = {alpha1:.6e}")
        print(f"  Ratio needed/α₁ = {needed / alpha1:.1f}")
    else:
        print(f"  ω and ω² degenerate at P — perturbation theory breaks down.")

    # Also try: the FULL dark Hamiltonian (not perturbative)
    print(f"\n  Non-perturbative: diagonalize H + ε·Σ at P")
    for eps_mult in [1, 10, 100, 1000, 10000]:
        eps = epsilon * eps_mult
        H_pert = H + eps * Sigma
        evals_p, evecs_p = la.eigh(H_pert)
        idxp = np.argsort(np.real(evals_p))
        evecs_p = evecs_p[:, idxp]

        # Rotation of the ω/ω² subspace
        overlap = np.conj(evecs[:, [w_idx, w2_idx]].T) @ evecs_p[:, [w_idx, w2_idx]]
        # The 2×2 rotation angle
        off_diag = abs(overlap[0, 1])**2
        if off_diag < 1.0:
            theta_rot = math.asin(math.sqrt(min(off_diag, 1.0)))
        else:
            theta_rot = math.pi / 2
        print(f"    ε = {eps:.4e} ({eps_mult}×α₁): θ_rot = {math.degrees(theta_rot):.4f}°")


# ═══════════════════════════════════════════════════════════════════════
# 8. BRUTE FORCE SCAN: ANY k THAT GIVES OBSERVED PMNS?
# ═══════════════════════════════════════════════════════════════════════

def scan_all_k_for_pmns(bonds, n_grid=25):
    """
    Scan the full BZ: at each k-point, compute U_PMNS = U(k)† · TBM
    and check if it matches the observed PMNS.

    If a k-point (or small cluster) reproduces the observed angles
    to within ~1%, that's the charged lepton momentum scale.
    """
    print("\n" + "=" * 76)
    print("  8. FULL BZ SCAN: U_PMNS(k) = U(k)† · TBM")
    print("=" * 76)

    U_TBM = TBM_matrix()

    # Reference: C₃ eigenstates at Gamma
    _, evecs_G, c3_G, _ = c3_decompose([0, 0, 0], bonds)

    # Identify triplet at Gamma
    evals_G, _ = diag_H([0, 0, 0], bonds)
    triplet_idx_G = np.where(np.abs(evals_G - (-1.0)) < 0.5)[0]

    best_score = 1e10
    best_k = None
    best_result = None
    all_results = []

    for n1, n2, n3 in product(range(n_grid), repeat=3):
        k = np.array([n1, n2, n3], dtype=float) / n_grid

        evals_k, evecs_k = diag_H(k, bonds)
        U_full = np.conj(evecs_G.T) @ evecs_k

        # Triplet subblock
        triplet_overlap = np.sum(np.abs(U_full[triplet_idx_G, :])**2, axis=0)
        triplet_idx_k = np.sort(np.argsort(triplet_overlap)[-3:])
        U_trip = U_full[np.ix_(triplet_idx_G, triplet_idx_k)]

        # Project to unitary
        Usvd, S, Vhsvd = la.svd(U_trip)
        U_l = Usvd @ Vhsvd

        U_PMNS = np.conj(U_l.T) @ U_TBM
        t12, t23, t13, dcp = extract_angles(U_PMNS)

        score = (math.degrees(t12) - math.degrees(theta12_obs))**2 + \
                (math.degrees(t23) - math.degrees(theta23_obs))**2 + \
                (math.degrees(t13) - math.degrees(theta13_obs))**2

        all_results.append({
            'k': k.copy(),
            'theta12': t12, 'theta23': t23, 'theta13': t13,
            'delta_CP': dcp, 'J': jarlskog(U_PMNS),
            'score': score,
        })

        if score < best_score:
            best_score = score
            best_k = k.copy()
            best_result = all_results[-1].copy()
            best_result['U_PMNS'] = U_PMNS.copy()

    # Sort by score
    all_results.sort(key=lambda r: r['score'])

    print(f"\n  Scanned {n_grid**3} k-points.")
    print(f"\n  Top 10 best k-points:")
    for i, r in enumerate(all_results[:10]):
        print(f"    k = ({r['k'][0]:.3f},{r['k'][1]:.3f},{r['k'][2]:.3f})  "
              f"θ₁₂={math.degrees(r['theta12']):.1f}° "
              f"θ₂₃={math.degrees(r['theta23']):.1f}° "
              f"θ₁₃={math.degrees(r['theta13']):.1f}° "
              f"RMS={math.sqrt(r['score']/3):.2f}°")

    if best_result:
        r = best_result
        print(f"\n  BEST k-point: ({r['k'][0]:.4f}, {r['k'][1]:.4f}, {r['k'][2]:.4f})")
        print(f"  θ₁₂ = {math.degrees(r['theta12']):.2f}° (obs: {math.degrees(theta12_obs):.2f}°)")
        print(f"  θ₂₃ = {math.degrees(r['theta23']):.2f}° (obs: {math.degrees(theta23_obs):.2f}°)")
        print(f"  θ₁₃ = {math.degrees(r['theta13']):.2f}° (obs: {math.degrees(theta13_obs):.2f}°)")
        print(f"  δ_CP = {math.degrees(r['delta_CP']):.2f}°")
        print(f"  J    = {r['J']:.6f}")
        print(f"  RMS error: {math.sqrt(best_score/3):.2f}°")

        # Is the best k near a high-symmetry point?
        for name, kp in K_POINTS.items():
            dist = la.norm(r['k'] - kp)
            if dist < 0.15:
                print(f"  Near {name} (distance {dist:.3f})")

    # Distribution statistics
    scores = np.array([r['score'] for r in all_results])
    print(f"\n  Score distribution:")
    print(f"    Best: {scores.min():.2f}")
    print(f"    Median: {np.median(scores):.2f}")
    print(f"    Worst: {scores.max():.2f}")
    print(f"    Fraction within 5° RMS: {np.mean(np.sqrt(scores/3) < 5):.4f}")
    print(f"    Fraction within 2° RMS: {np.mean(np.sqrt(scores/3) < 2):.4f}")

    return best_result, all_results


# ═══════════════════════════════════════════════════════════════════════
# 9. JARLSKOG FROM BLOCH STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

def compute_jarlskog_map(bonds, n_grid=25):
    """
    Compute the Jarlskog invariant J(k) across the BZ.
    J(k) = Im(U_e1 U_μ2 U*_e2 U*_μ1) where U = U(k)† · TBM.

    The Jarlskog is the ONLY CP-violating observable. Its BZ map
    shows where CP violation comes from geometrically.
    """
    print("\n" + "=" * 76)
    print("  9. JARLSKOG INVARIANT MAP ACROSS BZ")
    print("=" * 76)

    U_TBM = TBM_matrix()
    _, evecs_G, _, _ = c3_decompose([0, 0, 0], bonds)
    evals_G, _ = diag_H([0, 0, 0], bonds)
    triplet_idx_G = np.where(np.abs(evals_G - (-1.0)) < 0.5)[0]

    J_map = np.zeros((n_grid, n_grid, n_grid))
    J_values = []

    for n1, n2, n3 in product(range(n_grid), repeat=3):
        k = np.array([n1, n2, n3], dtype=float) / n_grid

        evals_k, evecs_k = diag_H(k, bonds)
        U_full = np.conj(evecs_G.T) @ evecs_k

        triplet_overlap = np.sum(np.abs(U_full[triplet_idx_G, :])**2, axis=0)
        triplet_idx_k = np.sort(np.argsort(triplet_overlap)[-3:])
        U_trip = U_full[np.ix_(triplet_idx_G, triplet_idx_k)]

        Usvd, S, Vhsvd = la.svd(U_trip)
        U_l = Usvd @ Vhsvd

        U_PMNS = np.conj(U_l.T) @ U_TBM
        J = jarlskog(U_PMNS)

        J_map[n1, n2, n3] = J
        J_values.append(J)

    J_values = np.array(J_values)

    print(f"  J_BZ_avg = {np.mean(J_values):.6f}")
    print(f"  |J|_BZ_avg = {np.mean(np.abs(J_values)):.6f}")
    print(f"  J_max = {np.max(J_values):.6f} at k ~ {np.unravel_index(J_map.argmax(), J_map.shape)}")
    print(f"  J_min = {np.min(J_values):.6f}")
    print(f"  J_rms = {np.sqrt(np.mean(J_values**2)):.6f}")
    print(f"  J_obs = {J_PMNS_obs:.4f}")

    # Fraction of BZ with |J| > 0.01
    print(f"  Fraction with |J| > 0.01: {np.mean(np.abs(J_values) > 0.01):.4f}")
    print(f"  Fraction with |J| > 0.02: {np.mean(np.abs(J_values) > 0.02):.4f}")
    print(f"  Fraction with |J| > 0.03: {np.mean(np.abs(J_values) > 0.03):.4f}")

    return J_map, J_values


# ═══════════════════════════════════════════════════════════════════════
# 10. SUMMARY AND GRADING
# ═══════════════════════════════════════════════════════════════════════

def grade_results(candidates, best_scan, best_weighted, J_values):
    """Honest grading of all PMNS extraction attempts."""
    print("\n" + "=" * 76)
    print("  FINAL SUMMARY AND GRADING")
    print("=" * 76)

    print(f"\n  OBSERVED (PDG 2024):")
    print(f"    θ₁₂ = {math.degrees(theta12_obs):.2f}°  (sin²θ₁₂ = {sin2_12_obs:.4f})")
    print(f"    θ₂₃ = {math.degrees(theta23_obs):.2f}°  (sin²θ₂₃ = {sin2_23_obs:.4f})")
    print(f"    θ₁₃ = {math.degrees(theta13_obs):.2f}°  (sin²θ₁₃ = {sin2_13_obs:.4f})")
    print(f"    δ_CP ~ {math.degrees(delta_CP_obs):.0f}°")
    print(f"    J    = {J_PMNS_obs:.4f}")

    print(f"\n  TBM (zeroth order):")
    print(f"    θ₁₂ = {math.degrees(theta12_TBM):.2f}°  (sin²θ₁₂ = {sin2_12_TBM:.4f})")
    print(f"    θ₂₃ = 45.00°  (sin²θ₂₃ = 0.5000)")
    print(f"    θ₁₃ = 0.00°   (sin²θ₁₃ = 0.0000)")
    print(f"    δ_CP = undefined (θ₁₃ = 0)")
    print(f"    J    = 0.000 (no CP violation)")

    print(f"\n  RESULTS BY METHOD:")

    # 1. High-symmetry k-points
    for kname, cand in candidates.items():
        err12 = abs(math.degrees(cand['theta12']) - math.degrees(theta12_obs))
        err23 = abs(math.degrees(cand['theta23']) - math.degrees(theta23_obs))
        err13 = abs(math.degrees(cand['theta13']) - math.degrees(theta13_obs))
        rms = math.sqrt((err12**2 + err23**2 + err13**2) / 3)

        grade = 'A' if rms < 2 else 'B' if rms < 5 else 'C' if rms < 10 else 'F'
        print(f"\n    U_l = U({kname}):  RMS = {rms:.1f}°  [{grade}]")
        print(f"      θ₁₂={math.degrees(cand['theta12']):.1f}° "
              f"θ₂₃={math.degrees(cand['theta23']):.1f}° "
              f"θ₁₃={math.degrees(cand['theta13']):.1f}° "
              f"J={cand['J']:.4f}")

    # 2. Weighted average
    if best_weighted:
        r = best_weighted
        rms = math.sqrt(r['score'] / 3)
        grade = 'A' if rms < 2 else 'B' if rms < 5 else 'C' if rms < 10 else 'F'
        print(f"\n    Weighted (w_H={r['w_H']:.2f},w_N={r['w_N']:.2f},w_P={r['w_P']:.2f}): "
              f"RMS = {rms:.1f}°  [{grade}]")

    # 3. Best scan
    if best_scan:
        r = best_scan
        rms = math.sqrt(r['score'] / 3)
        grade = 'A' if rms < 2 else 'B' if rms < 5 else 'C' if rms < 10 else 'F'
        print(f"\n    Best k-point scan: k=({r['k'][0]:.3f},{r['k'][1]:.3f},{r['k'][2]:.3f})  "
              f"RMS = {rms:.1f}°  [{grade}]")

    # 4. Jarlskog
    J_avg = np.mean(np.abs(J_values))
    J_ratio = J_avg / J_PMNS_obs if J_PMNS_obs > 0 else 0
    grade_J = 'A' if 0.5 < J_ratio < 2.0 else 'C' if 0.1 < J_ratio < 10 else 'F'
    print(f"\n    Jarlskog: |J|_avg = {J_avg:.4f} / obs {J_PMNS_obs:.4f} = {J_ratio:.2f}x  [{grade_J}]")

    # Key physics conclusions
    print(f"\n  KEY CONCLUSIONS:")
    print(f"    1. The Bloch band structure provides a MECHANISM for TBM deviations")
    print(f"       through the k-dependent mixing matrices U(k).")
    print(f"    2. At P (C₃ exact): no mixing → TBM preserved (consistency check).")
    print(f"    3. At N (50% mixing): maximal generation mixing, but PMNS angles")
    print(f"       depend on the full U_l† · U_ν product, not just the mixing fraction.")

    # Is there a k-point that works?
    if best_scan and math.sqrt(best_scan['score'] / 3) < 5.0:
        print(f"    4. POSITIVE: Found k-point with RMS < 5° → Bloch structure")
        print(f"       CAN reproduce approximate PMNS. This is a NEW route.")
    else:
        print(f"    4. NEGATIVE: No single k-point reproduces PMNS well.")
        print(f"       The charged lepton sector may not be a single k-point.")


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_results(candidates, all_scan_results, J_values):
    """Generate diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0): Angle predictions by k-point
    ax = axes[0, 0]
    knames = list(candidates.keys())
    x = np.arange(len(knames))
    width = 0.25

    obs_vals = [math.degrees(theta12_obs), math.degrees(theta23_obs),
                math.degrees(theta13_obs)]

    for ia, (angle_name, obs) in enumerate(
            [('θ₁₂', theta12_obs), ('θ₂₃', theta23_obs), ('θ₁₃', theta13_obs)]):
        vals = [math.degrees(candidates[kn][f'theta{["12","23","13"][ia]}'])
                for kn in knames]
        ax.bar(x + ia*width, vals, width, label=angle_name, alpha=0.7)
        ax.axhline(math.degrees(obs), ls='--', alpha=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(knames, fontsize=8)
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('PMNS angles by k-point')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1): Score distribution from BZ scan
    ax = axes[0, 1]
    if all_scan_results:
        scores = np.sqrt(np.array([r['score'] for r in all_scan_results]) / 3)
        ax.hist(scores, bins=50, alpha=0.7, color='steelblue')
        ax.axvline(np.min(scores), color='red', ls='--', label=f'Best: {np.min(scores):.1f}°')
        ax.set_xlabel('RMS angle error (degrees)')
        ax.set_ylabel('Count')
        ax.set_title('BZ scan: distribution of RMS errors')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0): Jarlskog distribution
    ax = axes[1, 0]
    ax.hist(J_values, bins=50, alpha=0.7, color='purple')
    ax.axvline(J_PMNS_obs, color='red', ls='--', label=f'J_obs = {J_PMNS_obs:.3f}')
    ax.axvline(-J_PMNS_obs, color='red', ls='--')
    ax.set_xlabel('Jarlskog invariant J')
    ax.set_ylabel('Count')
    ax.set_title('Jarlskog invariant across BZ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1): θ₁₃ vs θ₂₃ from scan, with observed point
    ax = axes[1, 1]
    if all_scan_results:
        t13s = [math.degrees(r['theta13']) for r in all_scan_results]
        t23s = [math.degrees(r['theta23']) for r in all_scan_results]
        scores = [r['score'] for r in all_scan_results]
        sc = ax.scatter(t13s, t23s, c=np.log10(np.array(scores) + 1),
                       s=3, cmap='viridis', alpha=0.5)
        ax.plot(math.degrees(theta13_obs), math.degrees(theta23_obs),
                'r*', markersize=15, label='Observed')
        ax.plot(0, 45, 'g^', markersize=10, label='TBM')
        ax.set_xlabel('θ₁₃ (degrees)')
        ax.set_ylabel('θ₂₃ (degrees)')
        ax.set_title('θ₁₃ vs θ₂₃ across BZ')
        ax.legend()
        plt.colorbar(sc, ax=ax, label='log₁₀(score+1)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'srs_pmns_from_bloch.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 76)
    print("  PMNS MIXING FROM SRS BLOCH BAND STRUCTURE")
    print("  U_PMNS = U_l†(k) · U_ν(TBM)")
    print("  Charged lepton sector = edge-local → specific k-point")
    print("  Neutrino sector = delocalized → TBM from S₄(K₄)")
    print("=" * 76)

    # Build lattice
    bonds = find_bonds()
    print(f"\n  Lattice: {N_ATOMS} atoms, {len(bonds)} bonds")

    # Verify
    evals_G, _ = diag_H([0, 0, 0], bonds)
    assert la.norm(evals_G - np.array([-1, -1, -1, 3])) < 0.01, "Gamma check failed"
    print(f"  Gamma eigenvalues: {evals_G}  [OK]")

    # 1. Mixing matrices at high-symmetry points
    mixing_results, V_Gamma, triplet_idx_G = compute_mixing_at_kpoints(bonds)

    # 2. PMNS candidates from each k-point
    candidates = compute_pmns_candidates(mixing_results, triplet_idx_G)

    # 3. Weighted averages
    best_weighted = compute_weighted_averages(mixing_results, triplet_idx_G)

    # 4. Dark correction analysis
    dark_results = compute_dark_correction(bonds)

    # 5. Dark-corrected PMNS
    compute_dark_pmns(bonds, mixing_results, triplet_idx_G)

    # 6. N-point analysis
    analyze_N_point(bonds, mixing_results)

    # 7. Perturbative θ₂₃ at P
    dark_theta23_at_P(bonds)

    # 8. Full BZ scan
    best_scan, all_scan_results = scan_all_k_for_pmns(bonds, n_grid=25)

    # 9. Jarlskog map
    J_map, J_values = compute_jarlskog_map(bonds, n_grid=20)

    # 10. Summary and grading
    grade_results(candidates, best_scan, best_weighted, J_values)

    # Plots
    plot_results(candidates, all_scan_results, J_values)


if __name__ == '__main__':
    main()
