#!/usr/bin/env python3
"""
Derive alpha_21 = 162 deg from first-principles M_R via Ihara zeta poles of K4.

Target: alpha_21 = 10*arctan(2-sqrt(3)) + pi/15 = 9*pi/10 = 162 deg exactly.

Strategy:
  1. Construct M_R from K4 Ihara zeta residues projected onto C3 generation basis
  2. Use P-point Bloch resolvent M_D (diagonal, from srs_majorana_md_principled.py)
  3. Seesaw: m_nu = -M_D^T M_R^{-1} M_D -> Takagi -> PMNS -> alpha_21
  4. Interpolation scan between K4 enantiomer and hierarchical M_R
  5. Principled M_R from Ihara residues: does it give 162 without tuning?

Key lattice invariants:
  k* = 3 (coordination), g = 10 (girth), n_g = 15 (girth cycles/vertex)
  lambda_1 = 2 - sqrt(3) (spectral gap), |u_pole| = 1/sqrt(2)
  Ihara triplet poles: u = (-1 +/- i*sqrt(7))/4, phase = pi - arctan(sqrt(7))
"""

import numpy as np
from numpy import linalg as la
from numpy import sqrt, pi, exp, conj, arccos, arctan, arctan2, log, cos, sin
from itertools import product

np.set_printoptions(precision=8, linewidth=120)

DEG = 180.0 / pi
RAD = pi / 180.0
omega3 = np.exp(2j * pi / 3)
NN_DIST = sqrt(2) / 4

# Targets
TARGET_ALPHA_21 = 162.0
TARGET_ALPHA_31 = 289.5
TARGET_DELTA_CP = 250.5

# Exact formula
ALPHA_21_EXACT = (10 * arctan(2 - sqrt(3)) + pi / 15) * DEG

# Graph invariants
K_COORD = 3
GIRTH = 10
N_G = 15
LAMBDA_1 = 2 - sqrt(3)
SQRT7 = sqrt(7)

# BCC primitive vectors
A_PRIM = np.array([
    [-0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5, -0.5],
])

# 4 atoms (Wyckoff 8a, x=1/8)
ATOMS = np.array([
    [1/8, 1/8, 1/8],
    [3/8, 7/8, 5/8],
    [7/8, 5/8, 3/8],
    [5/8, 3/8, 7/8],
])
N_ATOMS = 4

# C3 permutation: v0->v0, v1->v3, v2->v1, v3->v2
C3_PERM = np.zeros((4, 4), dtype=complex)
C3_PERM[0, 0] = 1
C3_PERM[3, 1] = 1
C3_PERM[1, 2] = 1
C3_PERM[2, 3] = 1

# Generation states: {trivial_s, omega, omega^2} on {v1,v2,v3}
GEN_BASIS = [
    np.array([0, 1, 1, 1], dtype=complex) / sqrt(3),           # trivial_s
    np.array([0, 1, omega3, omega3**2], dtype=complex) / sqrt(3),  # omega
    np.array([0, 1, omega3**2, omega3], dtype=complex) / sqrt(3),  # omega^2
]
GEN_LABELS = ['trivial_s', 'omega', 'omega^2']


# ===========================================================================
# LATTICE INFRASTRUCTURE
# ===========================================================================

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
        phase = np.exp(2j * pi * np.dot(k, cell))
        H[tgt, src] += phase
    return H


# ===========================================================================
# TAKAGI DECOMPOSITION & PHASE EXTRACTION
# ===========================================================================

def takagi_decompose(M):
    """Takagi: M = U* D U^dagger for complex symmetric M."""
    H = M @ conj(M).T
    eigvals, V = la.eigh(H)
    masses = sqrt(np.maximum(eigvals, 0))
    order = np.argsort(masses)
    masses = masses[order]
    V = V[:, order]
    D_check = V.T @ M @ V
    for i in range(len(masses)):
        if masses[i] > 0:
            phase = D_check[i, i] / masses[i]
            V[:, i] *= sqrt(conj(phase) / abs(phase))
    return masses, V


def extract_majorana_phases(U):
    """Extract alpha_21, alpha_31, delta_CP from 3x3 PMNS matrix."""
    U_r = U.copy()
    for i in range(3):
        ph = np.angle(U_r[i, 0])
        U_r[i, :] *= exp(-1j * ph)
    ph0 = np.angle(U_r[0, 0])
    U_r[0, :] *= exp(-1j * ph0)

    s13 = min(abs(U_r[0, 2]), 1.0)
    alpha_21 = 2 * np.angle(U_r[0, 1])
    alpha_31 = 2 * np.angle(U_r[1, 2])
    if s13 > 1e-10:
        delta_CP = alpha_31 / 2 - np.angle(U_r[0, 2])
    else:
        delta_CP = 0
    J = np.imag(U[0, 0] * conj(U[0, 2]) * conj(U[2, 0]) * U[2, 2])

    return {
        'alpha_21': alpha_21 * DEG % 360,
        'alpha_31': alpha_31 * DEG % 360,
        'delta_CP': delta_CP * DEG % 360,
        'J': J,
        's13': s13,
    }


# ===========================================================================
# SEESAW
# ===========================================================================

def run_seesaw(M_D, M_R, label=""):
    """m_nu = -M_D^T M_R^{-1} M_D, then Takagi decompose."""
    M_R_inv = la.inv(M_R)
    M_nu = -M_D.T @ M_R_inv @ M_D

    asym = la.norm(M_nu - M_nu.T)
    if la.norm(M_nu) > 0:
        asym /= la.norm(M_nu)

    masses, U = takagi_decompose(M_nu)
    phases = extract_majorana_phases(U)

    return {
        'M_nu': M_nu,
        'masses': masses,
        'U': U,
        'asym': asym,
        'phases': phases,
        'label': label,
        'M_D': M_D,
    }


def print_result(r, verbose=True):
    ph = r['phases']
    err_a21 = min(abs(ph['alpha_21'] - TARGET_ALPHA_21),
                  360 - abs(ph['alpha_21'] - TARGET_ALPHA_21))

    if verbose:
        M_D = r['M_D']
        print(f"\n  --- {r['label']} ---")
        print(f"    M_D magnitudes:")
        for i in range(3):
            print("      " + "  ".join(f"{abs(M_D[i,j]):.6f}" for j in range(3)))
        print(f"    M_D phases (deg):")
        for i in range(3):
            print("      " + "  ".join(f"{np.angle(M_D[i,j])*DEG:+8.2f}" for j in range(3)))

        masses = r['masses']
        if masses[0] > 0:
            print(f"    Takagi masses (ratio): {masses / masses[0]}")

    marker = " <<<" if err_a21 < 2 else (" <-" if err_a21 < 10 else "")
    print(f"    alpha_21 = {ph['alpha_21']:7.2f} deg  (target {TARGET_ALPHA_21}, err {err_a21:.2f} deg){marker}")
    print(f"    alpha_31 = {ph['alpha_31']:7.2f} deg  (target {TARGET_ALPHA_31})")
    print(f"    delta_CP = {ph['delta_CP']:7.2f} deg  (target {TARGET_DELTA_CP})")
    print(f"    J = {ph['J']:.6e},  s13 = {ph['s13']:.6f}")
    return err_a21


# ===========================================================================
# M_D AT P POINT (from srs_majorana_md_principled.py)
# ===========================================================================

def compute_MD_at_P(bonds, E_F=0.0, eta=0.5):
    """M_D from Bloch resolvent at P = (1/4,1/4,1/4) with eta=0.5."""
    k_frac = [0.25, 0.25, 0.25]
    H = bloch_H(k_frac, bonds)
    G = la.inv((E_F + 1j * eta) * np.eye(N_ATOMS) - H)

    M_D = np.zeros((3, 3), dtype=complex)
    for m in range(3):
        for n in range(3):
            M_D[m, n] = conj(GEN_BASIS[m]) @ G @ GEN_BASIS[n]

    return M_D


# ===========================================================================
# PART 1: IHARA ZETA OF K4 — POLES AND RESIDUES
# ===========================================================================

def part1_ihara_poles():
    print("=" * 76)
    print("  PART 1: IHARA ZETA POLES OF K4 AND RESIDUE STRUCTURE")
    print("=" * 76)

    N, k = 4, 3
    E = N * k // 2  # 6 edges
    r = E - N + 1   # 3 = rank of fundamental group

    # K4 adjacency eigenvalues: {3, -1, -1, -1}
    # (one trivial eigenvalue k=3, three triplet eigenvalues -1)
    adj_evals = [3, -1, -1, -1]
    print(f"\n  K4: N={N}, |E|={E}, k={k}, rank r={r}")
    print(f"  Adjacency eigenvalues: {adj_evals}")

    # Ihara zeta: zeta_I(u)^{-1} = (1-u^2)^{r-1} * prod_i (1 - lambda_i*u + (k-1)*u^2)
    # For K4 (k=3): (k-1) = 2
    #
    # Trivial factor: 1 - 3u + 2u^2 = (1-u)(1-2u)
    #   poles: u = 1, u = 1/2
    #
    # Triplet factor (multiplicity 3): 1 + u + 2u^2
    #   poles: u = (-1 +/- i*sqrt(7))/4
    #   |u|^2 = (1+7)/16 = 1/2, so |u| = 1/sqrt(2)
    #   arg(u_+) = pi - arctan(sqrt(7))

    u_plus = (-1 + 1j * SQRT7) / 4
    u_minus = (-1 - 1j * SQRT7) / 4
    print(f"\n  Triplet poles: u = (-1 +/- i*sqrt(7))/4")
    print(f"    u_+ = {u_plus:.8f}")
    print(f"    u_- = {u_minus:.8f}")
    print(f"    |u| = {abs(u_plus):.8f} = 1/sqrt(2) = {1/sqrt(2):.8f}")
    print(f"    |u|^2 = {abs(u_plus)**2:.8f} = 1/(k-1) = 1/2")
    print(f"    arg(u_+) = {np.angle(u_plus)*DEG:.4f} deg = pi - arctan(sqrt(7)) = {(pi - arctan(SQRT7))*DEG:.4f} deg")

    # Hashimoto eigenvalues: h = 2u (for the triplet sector)
    h_plus = 2 * u_plus
    h_minus = 2 * u_minus
    print(f"\n  Hashimoto eigenvalues h = 2u:")
    print(f"    h_+ = {h_plus:.8f}")
    print(f"    |h| = {abs(h_plus):.8f} = sqrt(k-1) = sqrt(2) = {sqrt(2):.8f}")

    # Residue at triplet pole u_+:
    # zeta_I(u)^{-1} near u_+ behaves as (u - u_+) * stuff
    # The Ihara determinant: (1-u^2)^2 * (1-3u+2u^2) * (1+u+2u^2)^3
    # The triplet factor (1+u+2u^2) = 2(u-u_+)(u-u_-)
    # Residue of zeta_I at u_+ from the triplet:
    #   Res = 1 / [2(u_+ - u_-)] * 1/[(1-u_+^2)^2 * (1-3u_+ + 2u_+^2)]
    # But what matters for M_R is how the pole projects onto generations.

    u_diff = u_plus - u_minus  # = i*sqrt(7)/2
    print(f"\n  u_+ - u_- = {u_diff:.8f} = i*sqrt(7)/2 = {1j*SQRT7/2:.8f}")

    # The triplet eigenspace of K4 adjacency: any vector orthogonal to (1,1,1,1)/2
    # The C3 rotation (on {v1,v2,v3}) decomposes this into:
    #   trivial_s sector: (0, 1, 1, 1)/sqrt(3) — eigenvalue of C3 = 1
    #   omega sector: (0, 1, w, w^2)/sqrt(3) — eigenvalue of C3 = omega
    #   omega^2 sector: (0, 1, w^2, w)/sqrt(3) — eigenvalue of C3 = omega^2
    # These are EXACTLY the generation basis states!
    # So the triplet Ihara pole at u_+ has EQUAL weight in all 3 generations.

    # The RESIDUE projected onto generation (m,n):
    # R_mn = <gen_m | Res(zeta_I, u_+) | gen_n>
    # Since the triplet eigenspace is spanned by the 3 generation states,
    # the projection matrix is the identity in generation space.
    # The PHASE of the residue carries the Ihara phase.

    print(f"\n  Triplet eigenspace = span of 3 generation states")
    print(f"  => Residue projection onto generation basis = scalar * I_3")
    print(f"  => M_R from residues is proportional to identity (no generation mixing)")
    print(f"  This is TOO SYMMETRIC to give 162 deg from seesaw alone.")

    return u_plus, u_minus


# ===========================================================================
# PART 2: M_R VARIANTS
# ===========================================================================

def build_MR_variants(u_plus, u_minus):
    print("\n" + "=" * 76)
    print("  PART 2: M_R CONSTRUCTIONS FROM IHARA POLES")
    print("=" * 76)

    phi_ihara = arctan(SQRT7)  # ~69.3 deg
    phi_pole = pi - phi_ihara   # ~110.7 deg = arg(u_+)
    u_mod = abs(u_plus)         # 1/sqrt(2)
    eps_R = u_mod               # off-diagonal coupling = |u|

    phi_arccos13 = arccos(1.0 / 3.0)  # ~70.5 deg (K4 enantiomer dihedral)

    variants = {}

    # Method A: K4 enantiomer — off-diagonal phase = arccos(1/3) (tetrahedral dihedral)
    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R[i, j] = eps_R * exp(-1j * phi_arccos13)
    variants['A: K4 enantiomer'] = M_R
    print(f"\n  A: K4 enantiomer: off-diag phase = arccos(1/3) = {phi_arccos13*DEG:.2f} deg")

    # Method B: Hierarchical (diagonal weights 1, 2/3, 4/9)
    M_R = np.diag([1.0, 2.0/3.0, (2.0/3.0)**2]).astype(complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R[i, j] = eps_R * sqrt(M_R[i,i].real * M_R[j,j].real) * exp(-1j * phi_arccos13)
    variants['B: Hierarchical'] = M_R
    print(f"  B: Hierarchical: diag = [1, 2/3, 4/9], off-diag phase = arccos(1/3)")

    # Method C: Ihara pole phase — off-diagonal phase = arg(u_+) = pi - arctan(sqrt(7))
    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R[i, j] = eps_R * exp(-1j * phi_pole)
    variants['C: Ihara pole phase'] = M_R
    print(f"  C: Ihara pole phase: off-diag phase = pi - arctan(sqrt(7)) = {phi_pole*DEG:.2f} deg")

    # Method D: Spectral gap phase — off-diagonal phase = arctan(lambda_1) = arctan(2-sqrt(3)) = pi/12
    phi_gap = arctan(LAMBDA_1)  # = pi/12 exactly
    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R[i, j] = eps_R * exp(-1j * phi_gap)
    variants['D: Spectral gap phase'] = M_R
    print(f"  D: Spectral gap phase: off-diag phase = arctan(2-sqrt(3)) = {phi_gap*DEG:.4f} deg = pi/12 = {(pi/12)*DEG:.4f} deg")

    # Method E: Target encoding — phase = 162/2 = 81 deg (the phase that would
    # directly give alpha_21 = 162 if M_D is diagonal real)
    phi_target = 81 * RAD
    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R[i, j] = eps_R * exp(-1j * phi_target)
    variants['E: Target/2 phase'] = M_R
    print(f"  E: Target/2 phase: off-diag phase = 81 deg")

    # Method F: Girth-weighted Ihara phase
    # 10 * arctan(lambda_1) = 10 * pi/12 = 5*pi/6 = 150 deg
    # Plus pi/15 = 12 deg -> 162 deg
    # Encode: off-diagonal phase = pi/15 (the correction from girth contribution)
    phi_correction = pi / 15
    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R[i, j] = eps_R * exp(-1j * phi_correction)
    variants['F: pi/15 correction'] = M_R
    print(f"  F: pi/15 correction: off-diag phase = pi/15 = {phi_correction*DEG:.4f} deg")

    # Method G: Combined girth + gap
    # Phase = girth * arctan(lambda_1) / (k-1) = 10 * (pi/12) / 2 = 5*pi/12
    phi_combined = GIRTH * arctan(LAMBDA_1) / (K_COORD - 1)
    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R[i, j] = eps_R * exp(-1j * phi_combined)
    variants['G: g*arctan(lam1)/(k-1)'] = M_R
    print(f"  G: g*arctan(lam1)/(k-1) = {phi_combined*DEG:.4f} deg")

    # Method H: Residue matrix — use the actual K4 adjacency projected into C3 basis
    # K4 adjacency = J - I (J = all-ones matrix)
    # In the triplet sector (orthogonal to (1,1,1,1)/2), eigenvalue = -1
    # The triplet projector in generation basis is I_3 (since gen states span triplet)
    # So residue gives M_R = -I_3 * (residue scalar)
    # But with the PHASE of the Ihara pole:
    # M_R_mn = delta_mn + (residue phase) * off-diagonal from NB walks
    # NB walks of length g=10 returning to origin contribute |u|^g * exp(i*g*arg(u))
    u_g = u_plus ** GIRTH
    print(f"\n  H: NB girth return: u_+^10 = {u_g:.8f}")
    print(f"    |u_+^10| = {abs(u_g):.8f} = (1/sqrt(2))^10 = {(1/sqrt(2))**10:.8f} = 1/32")
    print(f"    arg(u_+^10) = {np.angle(u_g)*DEG:.4f} deg")
    # arg(u_+^10) = 10 * (pi - arctan(sqrt(7))) mod 2pi
    # = 10*pi - 10*arctan(sqrt(7)) mod 2pi
    # 10*pi mod 2pi = 0
    # So arg = -10*arctan(sqrt(7)) mod 2pi
    # arctan(sqrt(7)) = 69.295... deg
    # 10 * 69.295 = 692.95 deg
    # 692.95 mod 360 = 332.95 deg -> -332.95 = 27.05 deg... complex
    # Let's just compute it
    arg_u10 = 10 * np.angle(u_plus)
    print(f"    10*arg(u_+) = {arg_u10*DEG:.4f} deg = {(arg_u10*DEG) % 360:.4f} deg (mod 360)")

    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R[i, j] = abs(u_g) * exp(1j * arg_u10)
    variants['H: NB girth return u^g'] = M_R

    # Method I: Ihara residue with C3 phase structure
    # The C3 rotation gives generation m a phase omega^m
    # The NB walk of length g around a girth cycle picks up:
    #   amplitude |u|^g, Ihara phase g*arg(u), and C3 phase 2*pi*m/3 per generation
    # M_R_mn = delta_mn + sum_cycles |u|^g * exp(i*(g*arg(u) + 2pi(m-n)/3))
    M_R = np.eye(3, dtype=complex)
    for m in range(3):
        for n in range(3):
            if m != n:
                c3_phase = 2 * pi * (m - n) / 3
                M_R[m, n] = N_G * abs(u_g) * exp(1j * (arg_u10 + c3_phase))
    variants['I: NB girth + C3 phase'] = M_R
    print(f"  I: NB girth + C3 phase: amp = n_g * |u|^g = {N_G * abs(u_g):.6f}")

    # Method J: Full NB walk Green's function at Ihara pole
    # G_NB(u) = sum_{n=0}^{inf} T^n u^n where T = NB transfer matrix
    # At u = u_+ (a pole), the residue dominates
    # Residue at u_+ of (1 + u + 2u^2)^{-1} = 1/(2*(u_+ - u_-)) * 1/(2u_+ + 1)
    denom = 2 * u_plus + 1  # derivative of (1+u+2u^2) at u_+
    # Actually d/du (1+u+2u^2) = 1 + 4u, evaluated at u_+
    deriv = 1 + 4 * u_plus
    residue = 1.0 / deriv
    print(f"\n  J: Residue of (1+u+2u^2)^{{-1}} at u_+:")
    print(f"    d/du(1+u+2u^2)|_{{u_+}} = 1 + 4u_+ = {deriv:.8f}")
    print(f"    Residue = 1/deriv = {residue:.8f}")
    print(f"    |Residue| = {abs(residue):.8f}")
    print(f"    arg(Residue) = {np.angle(residue)*DEG:.4f} deg")

    M_R = np.eye(3, dtype=complex)
    for m in range(3):
        for n in range(3):
            if m != n:
                c3_phase = 2 * pi * (m - n) / 3
                M_R[m, n] = abs(residue) * exp(1j * (np.angle(residue) + c3_phase))
    variants['J: Ihara residue + C3'] = M_R

    return variants


# ===========================================================================
# PART 3: SEESAW SCAN
# ===========================================================================

def part3_seesaw_scan(bonds, variants):
    print("\n" + "=" * 76)
    print("  PART 3: SEESAW WITH P-POINT M_D AND VARIOUS M_R")
    print("=" * 76)

    # M_D at P with EF=0, eta=0.5 (the setting that previously worked best)
    M_D = compute_MD_at_P(bonds, E_F=0.0, eta=0.5)
    scale = la.norm(M_D)
    M_D_n = M_D / scale

    print(f"\n  M_D at P (EF=0, eta=0.5), normalized:")
    print(f"    Magnitudes:")
    for i in range(3):
        print("      " + "  ".join(f"{abs(M_D_n[i,j]):.6f}" for j in range(3)))
    print(f"    Phases (deg):")
    for i in range(3):
        print("      " + "  ".join(f"{np.angle(M_D_n[i,j])*DEG:+8.2f}" for j in range(3)))

    results = []
    for name, M_R in variants.items():
        r = run_seesaw(M_D_n, M_R, name)
        results.append(r)

    # Sort by alpha_21 error
    scored = []
    for r in results:
        a21 = r['phases']['alpha_21']
        err = min(abs(a21 - TARGET_ALPHA_21), 360 - abs(a21 - TARGET_ALPHA_21))
        scored.append((err, r))
    scored.sort(key=lambda x: x[0])

    print(f"\n  Results sorted by alpha_21 error:")
    for i, (err, r) in enumerate(scored):
        print_result(r, verbose=(i < 5))

    return M_D_n, scored


# ===========================================================================
# PART 4: INTERPOLATION SCAN
# ===========================================================================

def part4_interpolation(bonds, M_D_n):
    print("\n" + "=" * 76)
    print("  PART 4: INTERPOLATION BETWEEN K4-ENANTIOMER AND HIERARCHICAL M_R")
    print("=" * 76)

    phi_arccos13 = arccos(1.0 / 3.0)
    eps_R = 1.0 / sqrt(2)

    # K4 enantiomer M_R
    M_R_K4 = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R_K4[i, j] = eps_R * exp(-1j * phi_arccos13)

    # Hierarchical M_R
    M_R_hier = np.diag([1.0, 2.0/3.0, (2.0/3.0)**2]).astype(complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R_hier[i, j] = eps_R * sqrt(M_R_hier[i,i].real * M_R_hier[j,j].real) * exp(-1j * phi_arccos13)

    # Scan interpolation parameter t: M_R(t) = (1-t)*M_R_K4 + t*M_R_hier
    N_scan = 10000
    t_values = np.linspace(0, 1, N_scan)
    best_err = 360
    best_t = None
    best_result = None

    a21_values = []

    for t in t_values:
        M_R = (1 - t) * M_R_K4 + t * M_R_hier
        try:
            r = run_seesaw(M_D_n, M_R)
        except la.LinAlgError:
            a21_values.append(None)
            continue
        a21 = r['phases']['alpha_21']
        a21_values.append(a21)
        err = min(abs(a21 - TARGET_ALPHA_21), 360 - abs(a21 - TARGET_ALPHA_21))
        if err < best_err:
            best_err = err
            best_t = t
            best_result = r

    print(f"\n  Interpolation: M_R(t) = (1-t)*M_R_K4 + t*M_R_hier")
    print(f"  Scanned {N_scan} values of t in [0,1]")

    if best_result:
        print(f"\n  Best t = {best_t:.6f}")
        print(f"  alpha_21 error = {best_err:.4f} deg")
        best_result['label'] = f"Interpolation t={best_t:.6f}"
        print_result(best_result, verbose=True)

        # Check if best_t has a meaning
        print(f"\n  Checking if t* = {best_t:.6f} has algebraic meaning:")
        candidates = {
            '1/3': 1/3,
            '2/3': 2/3,
            '1/2': 1/2,
            'lambda_1 = 2-sqrt(3)': LAMBDA_1,
            '1-lambda_1 = sqrt(3)-1': sqrt(3)-1,
            'sqrt(2)/2': sqrt(2)/2,
            '1/sqrt(3)': 1/sqrt(3),
            'arctan(sqrt(7))/pi': arctan(SQRT7)/pi,
            '1/7': 1/7,
            '2/7': 2/7,
            '3/7': 3/7,
            'arctan(2-sqrt(3))/pi*10': 10*arctan(LAMBDA_1)/pi,
            'pi/15/pi = 1/15': 1/15,
            '1/10': 1/10,
            '2/9 (delta_Koide)': 2/9,
            '1/(2+sqrt(3))': 1/(2+sqrt(3)),
            '(sqrt(3)-1)/2': (sqrt(3)-1)/2,
            'sin(pi/10)': sin(pi/10),
            'cos(pi/5)': cos(pi/5),
            'sin(pi/12)': sin(pi/12),
        }
        for name, val in sorted(candidates.items(), key=lambda x: abs(x[1] - best_t)):
            err_t = abs(val - best_t)
            if err_t < 0.05:
                marker = " <<<" if err_t < 0.005 else " <-" if err_t < 0.02 else ""
                print(f"    {name:35s} = {val:.6f}  (diff = {err_t:.6f}){marker}")

    return best_t


# ===========================================================================
# PART 5: PRINCIPLED M_R FROM IHARA — DIRECT PHASE ENCODING
# ===========================================================================

def part5_principled_phase(bonds, M_D_n):
    print("\n" + "=" * 76)
    print("  PART 5: PRINCIPLED M_R — PHASE ENCODING FROM LATTICE INVARIANTS")
    print("=" * 76)

    # The formula: alpha_21 = 10*arctan(2-sqrt(3)) + pi/15 = 162
    # 10 = girth, 2-sqrt(3) = spectral gap, 15 = girth cycles per vertex
    # arctan(2-sqrt(3)) = pi/12

    # The seesaw phases come from the INTERPLAY of M_D phases and M_R phases.
    # For diagonal M_D (which we have at P), the phases in M_nu come from M_R^{-1}.
    # If M_R is democratic (uniform off-diagonal), the seesaw preserves its phase structure.

    # Key insight: The off-diagonal phase of M_R should encode the
    # non-backtracking (NB) walk structure. The NB walk Green's function
    # at the Ihara pole has phase per step = arg(u_+) = pi - arctan(sqrt(7)).
    # But the GIRTH walk (10 steps) accumulates:
    #   total phase = 10 * arg(u_+) = 10*pi - 10*arctan(sqrt(7))
    # mod 2*pi: 10*pi mod 2*pi = 0
    # So total phase = -10*arctan(sqrt(7)) mod 2*pi

    # Alternative: the spectral gap phase per girth cycle
    # arctan(lambda_1) = arctan(2-sqrt(3)) = pi/12
    # 10 girth cycles: 10*pi/12 = 5*pi/6 = 150 deg
    # Plus the 15 girth-cycles-per-vertex correction: pi/15 = 12 deg
    # Total: 162 deg

    # Strategy: encode the off-diagonal phase as the TARGET alpha_21 / 2 = 81 deg
    # since alpha_21 = 2*arg(U_{e1}) and the seesaw roughly doubles phases.
    # But this is circular. Instead:

    # The M_R off-diagonal phase that would produce alpha_21 = 162 deg
    # depends on the specific M_D. Let's SCAN the off-diagonal phase.

    eps_R = 1.0 / sqrt(2)

    print(f"\n  Scanning off-diagonal phase phi_R of democratic M_R...")
    print(f"  M_R = I + eps_R * exp(-i*phi_R) * (J - I), eps_R = 1/sqrt(2)")

    N_scan = 3600
    phi_values = np.linspace(0, 2*pi, N_scan, endpoint=False)
    best_err = 360
    best_phi = None
    best_result = None

    for phi_R in phi_values:
        M_R = np.eye(3, dtype=complex)
        for i in range(3):
            for j in range(3):
                if i != j:
                    M_R[i, j] = eps_R * exp(-1j * phi_R)
        try:
            r = run_seesaw(M_D_n, M_R)
        except la.LinAlgError:
            continue
        a21 = r['phases']['alpha_21']
        err = min(abs(a21 - TARGET_ALPHA_21), 360 - abs(a21 - TARGET_ALPHA_21))
        if err < best_err:
            best_err = err
            best_phi = phi_R
            best_result = r

    print(f"\n  Best off-diagonal phase: phi_R = {best_phi*DEG:.4f} deg")
    print(f"  alpha_21 error = {best_err:.4f} deg")
    if best_result:
        best_result['label'] = f"Phase scan: phi_R = {best_phi*DEG:.2f} deg"
        print_result(best_result, verbose=True)

    # Check if best_phi has algebraic meaning
    print(f"\n  Checking algebraic meaning of phi_R = {best_phi*DEG:.4f} deg:")
    candidates = {
        'arccos(1/3) = K4 dihedral': arccos(1.0/3.0),
        'pi - arctan(sqrt(7)) = arg(u_+)': pi - arctan(SQRT7),
        'arctan(sqrt(7))': arctan(SQRT7),
        'pi/12 = arctan(2-sqrt(3))': pi/12,
        '5*pi/6 = 10*pi/12': 5*pi/6,
        'pi/15': pi/15,
        '9*pi/10 = 162 deg': 9*pi/10,
        'pi/2': pi/2,
        'pi/3': pi/3,
        '2*pi/3': 2*pi/3,
        'pi/4': pi/4,
        '3*pi/4': 3*pi/4,
        'pi/5': pi/5,
        '2*pi/5': 2*pi/5,
        '3*pi/5': 3*pi/5,
        'pi/6': pi/6,
        '5*pi/12': 5*pi/12,
        '7*pi/12': 7*pi/12,
        'pi/10': pi/10,
        '3*pi/10': 3*pi/10,
        '7*pi/10': 7*pi/10,
        '9*pi/10': 9*pi/10,
        'arctan(1/sqrt(7))': arctan(1/SQRT7),
        'pi - arccos(1/3)': pi - arccos(1.0/3.0),
        'arccos(-1/3)': arccos(-1.0/3.0),
    }
    sorted_cands = sorted(candidates.items(), key=lambda x: abs(x[1] - best_phi))
    for name, val in sorted_cands[:10]:
        err_c = abs(val - best_phi)
        marker = " <<<" if err_c < 0.002 else (" <-" if err_c < 0.02 else "")
        print(f"    {name:40s} = {val*DEG:8.4f} deg  (diff = {err_c*DEG:.4f} deg){marker}")

    # Also scan with hierarchical diagonal
    print(f"\n  Scanning phi_R with hierarchical diagonal [1, 2/3, 4/9]...")
    best_err2 = 360
    best_phi2 = None
    best_result2 = None

    for phi_R in phi_values:
        M_R = np.diag([1.0, 2.0/3.0, (2.0/3.0)**2]).astype(complex)
        for i in range(3):
            for j in range(3):
                if i != j:
                    M_R[i, j] = eps_R * sqrt(M_R[i,i].real * M_R[j,j].real) * exp(-1j * phi_R)
        try:
            r = run_seesaw(M_D_n, M_R)
        except la.LinAlgError:
            continue
        a21 = r['phases']['alpha_21']
        err = min(abs(a21 - TARGET_ALPHA_21), 360 - abs(a21 - TARGET_ALPHA_21))
        if err < best_err2:
            best_err2 = err
            best_phi2 = phi_R
            best_result2 = r

    print(f"\n  Best hierarchical phi_R: {best_phi2*DEG:.4f} deg")
    print(f"  alpha_21 error = {best_err2:.4f} deg")
    if best_result2:
        best_result2['label'] = f"Hier phase scan: phi_R = {best_phi2*DEG:.2f} deg"
        print_result(best_result2, verbose=True)

    sorted_cands2 = sorted(candidates.items(), key=lambda x: abs(x[1] - best_phi2))
    for name, val in sorted_cands2[:10]:
        err_c = abs(val - best_phi2)
        marker = " <<<" if err_c < 0.002 else (" <-" if err_c < 0.02 else "")
        print(f"    {name:40s} = {val*DEG:8.4f} deg  (diff = {err_c*DEG:.4f} deg){marker}")

    return best_phi, best_phi2


# ===========================================================================
# PART 6: 2D SCAN — OFF-DIAGONAL PHASE + DIAGONAL HIERARCHY
# ===========================================================================

def part6_2d_scan(bonds, M_D_n):
    print("\n" + "=" * 76)
    print("  PART 6: 2D SCAN — OFF-DIAGONAL PHASE phi_R AND HIERARCHY PARAMETER h")
    print("=" * 76)

    # Parameterize: diag = [1, 1-h, (1-h)^2], off-diag phase = phi_R
    # h = 0: democratic, h = 1/3: hierarchical [1, 2/3, 4/9]
    eps_R = 1.0 / sqrt(2)

    N_phi = 360
    N_h = 100
    phi_values = np.linspace(0, 2*pi, N_phi, endpoint=False)
    h_values = np.linspace(0, 0.5, N_h)

    best_err = 360
    best_phi = None
    best_h = None
    best_result = None

    for phi_R in phi_values:
        for h in h_values:
            d = np.array([1.0, 1.0 - h, (1.0 - h)**2])
            M_R = np.diag(d).astype(complex)
            for i in range(3):
                for j in range(3):
                    if i != j:
                        M_R[i, j] = eps_R * sqrt(d[i] * d[j]) * exp(-1j * phi_R)
            try:
                r = run_seesaw(M_D_n, M_R)
            except la.LinAlgError:
                continue
            a21 = r['phases']['alpha_21']
            err = min(abs(a21 - TARGET_ALPHA_21), 360 - abs(a21 - TARGET_ALPHA_21))
            if err < best_err:
                best_err = err
                best_phi = phi_R
                best_h = h
                best_result = r

    print(f"\n  Best 2D: phi_R = {best_phi*DEG:.4f} deg, h = {best_h:.6f}")
    print(f"  alpha_21 error = {best_err:.4f} deg")
    if best_result:
        best_result['label'] = f"2D scan: phi={best_phi*DEG:.2f}, h={best_h:.4f}"
        print_result(best_result, verbose=True)

    # Check algebraic meaning of both
    print(f"\n  Checking h = {best_h:.6f}:")
    h_candidates = {
        '0 (democratic)': 0,
        '1/3': 1/3,
        '2/9': 2/9,
        '1/7': 1/7,
        '2-sqrt(3)': LAMBDA_1,
        'sqrt(3)-1-1 = sqrt(3)-2': sqrt(3)-2 if sqrt(3) > 2 else 2-sqrt(3),
        '1/sqrt(7)': 1/SQRT7,
        '1/sqrt(3)-1/3': 1/sqrt(3) - 1/3,
        '1/10': 0.1,
        '1/15': 1/15,
        '1/4': 0.25,
        '1/6': 1/6,
    }
    sorted_h = sorted(h_candidates.items(), key=lambda x: abs(x[1] - best_h))
    for name, val in sorted_h[:8]:
        err_h = abs(val - best_h)
        marker = " <<<" if err_h < 0.005 else (" <-" if err_h < 0.02 else "")
        print(f"    {name:30s} = {val:.6f}  (diff = {err_h:.6f}){marker}")

    return best_phi, best_h, best_err


# ===========================================================================
# PART 7: DOES arctan(sqrt(7)) ENCODE IN M_R?
# ===========================================================================

def part7_arctan_sqrt7(bonds, M_D_n):
    print("\n" + "=" * 76)
    print("  PART 7: M_R BUILT FROM arctan(sqrt(7)) — THE IHARA SPLITTING ANGLE")
    print("=" * 76)

    phi_ihara = arctan(SQRT7)  # ~69.3 deg
    eps_R = 1.0 / sqrt(2)

    # The screw phase per girth cycle in the NB walk is arctan(sqrt(7))
    # But the physical M_R should encode how the girth cycles contribute
    # to the right-handed neutrino mass matrix.
    #
    # Key: arctan(sqrt(7)) = pi/2 - arctan(1/sqrt(7))
    # And arctan(1/sqrt(7)) ~= 20.7 deg
    # Also: arctan(sqrt(7)) + arctan(2-sqrt(3)) = 69.3 + 15 = 84.3 deg
    # Not obvious...
    #
    # But: the TARGET formula says alpha_21 = 10*arctan(lambda_1) + pi/15
    # = 10*pi/12 + pi/15 = 5*pi/6 + pi/15 = (25*pi + 2*pi)/30 = 27*pi/30 = 9*pi/10
    #
    # Try: M_R with phase = arctan(sqrt(7))/g_cycles = arctan(sqrt(7))/15
    # or phase = arctan(sqrt(7)) * lambda_1
    # or the Ihara pole phase modulated by generation

    constructions = {}

    # 7a: Phase = arctan(sqrt(7)) on off-diagonal
    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R[i, j] = eps_R * exp(-1j * phi_ihara)
    constructions['7a: arctan(sqrt(7))'] = M_R

    # 7b: Phase = arctan(sqrt(7))/3 per generation
    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                gen_order = abs(i - j)
                M_R[i, j] = eps_R * exp(-1j * phi_ihara * gen_order / 3)
    constructions['7b: arctan(sqrt(7))*|m-n|/3'] = M_R

    # 7c: Phase = 9*pi/10 / 2 = 9*pi/20 (half the target, anticipating doubling)
    phi_half = 9 * pi / 20
    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R[i, j] = eps_R * exp(-1j * phi_half)
    constructions['7c: 9*pi/20 = 81 deg'] = M_R

    # 7d: Combination of spectral gap + girth cycle phases on off-diagonal
    # phi_mn = arctan(lambda_1) * girth/3 + pi/15 * |m-n|
    for i_const in range(3):
        for j_const in range(3):
            pass
    M_R = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                gen_diff = (i - j) % 3  # 1 or 2
                phi_mn = arctan(LAMBDA_1) * GIRTH / 3 + pi / 15 * gen_diff
                M_R[i, j] = eps_R * exp(-1j * phi_mn)
    constructions['7d: g/3*arctan(lam1) + (pi/15)*|m-n|'] = M_R

    # 7e: NB walk with C3 twist: each girth step advances C3 by 2*pi/10
    # The screw on srs has C3 advancement of 2*pi/(3*girth/3) = 2*pi/10 per bond
    # Over girth=10 bonds: total C3 rotation = 2*pi
    # So the generation-dependent phase per girth cycle = 2*pi*m/3
    # Total: girth * arg(u_pole) + 2*pi*m/3
    arg_u = np.angle((-1 + 1j*SQRT7)/4)
    M_R = np.eye(3, dtype=complex)
    for m in range(3):
        for n in range(3):
            if m != n:
                total_phase = GIRTH * arg_u + 2*pi*(m-n)/3
                M_R[m, n] = eps_R * exp(-1j * total_phase)
    constructions['7e: g*arg(u_+) + 2pi(m-n)/3'] = M_R

    # 7f: Use |u_+|^2 = 1/2 as the off-diagonal STRENGTH, and the full
    # Ihara residue phase structure
    # Residue phase at u_+: arg(1/(1+4u_+))
    u_plus = (-1 + 1j*SQRT7)/4
    deriv = 1 + 4*u_plus
    res_phase = -np.angle(deriv)  # residue = 1/deriv
    M_R = np.eye(3, dtype=complex)
    for m in range(3):
        for n in range(3):
            if m != n:
                c3_phase = 2*pi*(m-n)/3
                M_R[m, n] = abs(u_plus)**2 * exp(1j * (res_phase + c3_phase))
    constructions['7f: |u|^2 * exp(i*(res_phase + C3))'] = M_R

    # Run all
    print(f"\n  Running seesaw for {len(constructions)} M_R constructions...")
    results = []
    for name, M_R in constructions.items():
        try:
            r = run_seesaw(M_D_n, M_R, name)
            results.append(r)
        except la.LinAlgError:
            print(f"    {name}: M_R singular, skipped")

    scored = []
    for r in results:
        a21 = r['phases']['alpha_21']
        err = min(abs(a21 - TARGET_ALPHA_21), 360 - abs(a21 - TARGET_ALPHA_21))
        scored.append((err, r))
    scored.sort(key=lambda x: x[0])

    for i, (err, r) in enumerate(scored):
        print_result(r, verbose=(i < 3))


# ===========================================================================
# PART 8: EPSILON SCAN — OFF-DIAGONAL STRENGTH
# ===========================================================================

def part8_eps_scan(bonds, M_D_n, phi_R_democratic, phi_R_hier):
    print("\n" + "=" * 76)
    print("  PART 8: JOINT SCAN — OFF-DIAGONAL STRENGTH eps AND PHASE phi_R")
    print("=" * 76)

    # For each of the best phases, scan epsilon (off-diagonal strength)
    N_eps = 200
    N_phi = 720
    eps_values = np.linspace(0.01, 2.0, N_eps)
    phi_values = np.linspace(0, 2*pi, N_phi, endpoint=False)

    best_err = 360
    best_eps = None
    best_phi = None
    best_result = None

    for eps_R in eps_values:
        for phi_R in phi_values:
            M_R = np.eye(3, dtype=complex)
            for i in range(3):
                for j in range(3):
                    if i != j:
                        M_R[i, j] = eps_R * exp(-1j * phi_R)
            try:
                r = run_seesaw(M_D_n, M_R)
            except la.LinAlgError:
                continue
            a21 = r['phases']['alpha_21']
            err = min(abs(a21 - TARGET_ALPHA_21), 360 - abs(a21 - TARGET_ALPHA_21))
            if err < best_err:
                best_err = err
                best_eps = eps_R
                best_phi = phi_R
                best_result = r

    print(f"\n  Best eps-phi: eps = {best_eps:.6f}, phi = {best_phi*DEG:.4f} deg")
    print(f"  alpha_21 error = {best_err:.4f} deg")
    if best_result:
        best_result['label'] = f"eps-phi scan: eps={best_eps:.4f}, phi={best_phi*DEG:.2f}"
        print_result(best_result, verbose=True)

    # Check algebraic meaning
    print(f"\n  eps = {best_eps:.6f}:")
    eps_candidates = {
        '1/sqrt(2)': 1/sqrt(2),
        'sqrt(2)': sqrt(2),
        '1': 1.0,
        '1/2': 0.5,
        '2-sqrt(3)': LAMBDA_1,
        '2+sqrt(3)': 2+sqrt(3),
        '1/sqrt(3)': 1/sqrt(3),
        'sqrt(3)': sqrt(3),
        'sqrt(7)/4': SQRT7/4,
        '1/sqrt(7)': 1/SQRT7,
        '3/4': 0.75,
    }
    sorted_eps = sorted(eps_candidates.items(), key=lambda x: abs(x[1] - best_eps))
    for name, val in sorted_eps[:6]:
        err_e = abs(val - best_eps)
        marker = " <<<" if err_e < 0.01 else (" <-" if err_e < 0.05 else "")
        print(f"    {name:20s} = {val:.6f}  (diff = {err_e:.6f}){marker}")

    print(f"\n  phi = {best_phi*DEG:.4f} deg:")
    phi_candidates = {
        'arccos(1/3)': arccos(1.0/3.0),
        'pi - arctan(sqrt(7))': pi - arctan(SQRT7),
        'arctan(sqrt(7))': arctan(SQRT7),
        'pi/12': pi/12,
        '5*pi/6': 5*pi/6,
        'pi/15': pi/15,
        '9*pi/10': 9*pi/10,
        'pi/2': pi/2,
        'pi/3': pi/3,
        '2*pi/3': 2*pi/3,
        'arccos(-1/3)': arccos(-1.0/3.0),
        'pi/4': pi/4,
        '3*pi/4': 3*pi/4,
        '7*pi/12': 7*pi/12,
    }
    sorted_phi = sorted(phi_candidates.items(), key=lambda x: abs(x[1] - best_phi))
    for name, val in sorted_phi[:8]:
        err_p = abs(val - best_phi)
        marker = " <<<" if err_p < 0.005 else (" <-" if err_p < 0.02 else "")
        print(f"    {name:30s} = {val*DEG:8.4f} deg  (diff = {err_p*DEG:.4f} deg){marker}")

    return best_eps, best_phi


# ===========================================================================
# SUMMARY
# ===========================================================================

def summary():
    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print(f"""
  Target: alpha_21 = 162 deg = 10*arctan(2-sqrt(3)) + pi/15 = 9*pi/10

  Exact formula verification:
    10 * arctan(2 - sqrt(3)) = 10 * pi/12 = 5*pi/6 = {10*arctan(LAMBDA_1)*DEG:.6f} deg
    pi/15 = {(pi/15)*DEG:.6f} deg
    Sum = {ALPHA_21_EXACT:.6f} deg

  Key lattice invariants:
    k* = {K_COORD} (coordination number)
    g = {GIRTH} (girth)
    n_g = {N_G} (girth cycles per vertex)
    lambda_1 = 2 - sqrt(3) = {LAMBDA_1:.8f} (spectral gap)
    |u_pole| = 1/sqrt(2) = {1/sqrt(2):.8f}
    arg(u_pole) = pi - arctan(sqrt(7)) = {(pi - arctan(SQRT7))*DEG:.4f} deg
    arctan(sqrt(7)) = {arctan(SQRT7)*DEG:.4f} deg
    sqrt(7) = sqrt(4(k-1) - 1) = {SQRT7:.8f}

  Ihara pole relation: h = 2u (Hashimoto eigenvalue = 2 * Ihara pole)
  Ramanujan: |h| = sqrt(k-1) = sqrt(2)
  |u|^2 = 1/(k-1) = 1/2
""")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 76)
    print("  ALPHA_21 FROM IHARA ZETA M_R: FIRST-PRINCIPLES DERIVATION")
    print(f"  Target: alpha_21 = {TARGET_ALPHA_21} deg = 10*arctan(2-sqrt(3)) + pi/15")
    print(f"  Exact: {ALPHA_21_EXACT:.6f} deg")
    print("=" * 76)

    bonds = find_bonds()
    print(f"\n  {len(bonds)} directed bonds in primitive cell")

    # Part 1: Ihara poles
    u_plus, u_minus = part1_ihara_poles()

    # Part 2: Build M_R variants
    variants = build_MR_variants(u_plus, u_minus)

    # Part 3: Seesaw with all variants
    M_D_n, scored = part3_seesaw_scan(bonds, variants)

    # Part 4: Interpolation scan
    best_t = part4_interpolation(bonds, M_D_n)

    # Part 5: Phase scan for principled M_R
    phi_dem, phi_hier = part5_principled_phase(bonds, M_D_n)

    # Part 6: 2D scan (phase + hierarchy)
    best_phi_2d, best_h_2d, best_err_2d = part6_2d_scan(bonds, M_D_n)

    # Part 7: arctan(sqrt(7)) constructions
    part7_arctan_sqrt7(bonds, M_D_n)

    # Part 8: Full eps + phi scan
    best_eps, best_phi = part8_eps_scan(bonds, M_D_n, phi_dem, phi_hier)

    # Summary
    summary()


if __name__ == '__main__':
    main()
