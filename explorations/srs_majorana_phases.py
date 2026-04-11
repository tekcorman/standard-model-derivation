#!/usr/bin/env python3
"""
Majorana CP phases from the FULL srs lattice seesaw.

KEY INSIGHT: On K4, P_D and P_R commute because all generations have the
same energy. On the FULL lattice, the omega and omega^2 bands are at
DIFFERENT energies (+-sqrt(3) at P). This breaks the commutation.
[P_D, P_R] != 0 should now be possible.

BACKGROUND:
  - K4 quotient: all 64 sign patterns of arccos(1/3) give alpha_31 = 0
    (proven in seesaw_cp_proof.py, chirality_sign_pattern.py)
  - Root cause: P_D and P_R both functions of J (all-ones matrix) on K4
  - The C3 irreps at P = (1/4,1/4,1/4) give:
      omega band at E = +sqrt(3)
      omega^2 band at E = -sqrt(3)
    These are DIFFERENT energy bands -> mass matrices can distinguish them

METHOD:
  1. Build the Bloch Hamiltonian H(k) for srs (4 atoms, BCC primitive cell)
  2. Construct M_D via BZ-averaged resolvent in generation basis:
       (M_D)_mn = integral_BZ dk <gen_m|G(k,E_F)|gen_n>
  3. Construct M_R from the delocalized (|000>) sector via Ihara structure
     on K4, using R = 32.68 from arctan(sqrt(7))
  4. Seesaw: M_nu = -M_D^T M_R^{-1} M_D
  5. Takagi decompose, extract Majorana phases alpha_21, alpha_31
  6. Compare [P_D, P_R] on K4 vs full lattice

Targets: alpha_21 ~ 162 deg, alpha_31 ~ 289.5 deg, delta_CP ~ 197 or 250 deg
"""

import numpy as np
from numpy import linalg as la
from numpy import sqrt, cos, sin, pi, arccos, arctan2, exp, conj
from itertools import product

np.set_printoptions(precision=8, linewidth=120)

# ==========================================================================
# CONSTANTS
# ==========================================================================

DEG = 180.0 / pi
ARCCOS_1_3 = arccos(1.0 / 3.0)
omega3 = np.exp(2j * pi / 3)
NN_DIST = sqrt(2) / 4

# BCC primitive vectors
A_PRIM = np.array([
    [-0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5, -0.5],
])

# 4 atoms in Cartesian coords (Wyckoff 8a, x=1/8)
ATOMS = np.array([
    [1/8, 1/8, 1/8],   # v0 -- on C3 axis
    [3/8, 7/8, 5/8],   # v1
    [7/8, 5/8, 3/8],   # v2
    [5/8, 3/8, 7/8],   # v3
])
N_ATOMS = 4

# C3 permutation: v0->v0, v1->v3, v2->v1, v3->v2
C3_PERM = np.zeros((4, 4), dtype=complex)
C3_PERM[0, 0] = 1
C3_PERM[3, 1] = 1
C3_PERM[1, 2] = 1
C3_PERM[2, 3] = 1

# C3 generation eigenstates on {v1,v2,v3} subspace
# Trivial (on v0): (1, 0, 0, 0)
# Trivial (symmetric): (0, 1, 1, 1)/sqrt(3)
# omega generation: (0, 1, w, w^2)/sqrt(3)
# omega^2 generation: (0, 1, w^2, w)/sqrt(3)
GEN_STATES = {
    'trivial_0': np.array([1, 0, 0, 0], dtype=complex),
    'trivial_s': np.array([0, 1, 1, 1], dtype=complex) / sqrt(3),
    'gen1_w':    np.array([0, 1, omega3, omega3**2], dtype=complex) / sqrt(3),
    'gen2_w2':   np.array([0, 1, omega3**2, omega3], dtype=complex) / sqrt(3),
}

# Experimental targets
TARGET_ALPHA_21 = 162.0   # degrees
TARGET_ALPHA_31 = 289.5   # degrees
TARGET_DELTA_CP_A = 197.0 # degrees (one possibility)
TARGET_DELTA_CP_B = 250.5 # degrees (pi + arccos(1/3))

# Neutrino mass ratio from Ihara zeta
# arctan(sqrt(7)) appears in K4 Ihara triplet pole phases
# R = dm31^2 / dm21^2 ~ 32.58 (PDG)
ARCTAN_SQRT7 = np.arctan(sqrt(7))
R_IHARA = 32.68  # target neutrino mass ratio

# Experimental mass splittings
DM21_SQ = 7.53e-5   # eV^2
DM31_SQ = 2.453e-3  # eV^2


# ==========================================================================
# LATTICE INFRASTRUCTURE (from srs_generation_c3.py)
# ==========================================================================

def find_bonds():
    """Find NN bonds in the primitive cell."""
    tol = 0.02
    bonds = []
    for i in range(N_ATOMS):
        ri = ATOMS[i]
        nbrs = []
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                dist = la.norm(rj - ri)
                if dist < tol:
                    continue
                if abs(dist - NN_DIST) < tol:
                    nbrs.append((j, (n1, n2, n3), rj - ri))
        assert len(nbrs) == 3, f"Atom {i} has {len(nbrs)} NN (expected 3)"
        for j, cell, dr in nbrs:
            bonds.append((i, j, cell))
    return bonds


def bloch_H(k_frac, bonds):
    """4x4 Bloch Hamiltonian at fractional k = (k1,k2,k3)."""
    H = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for src, tgt, cell in bonds:
        phase = np.exp(2j * pi * np.dot(k, cell))
        H[tgt, src] += phase
    return H


def diag_H(k_frac, bonds):
    """Diagonalize H(k), return sorted eigenvalues and eigenvectors."""
    H = bloch_H(k_frac, bonds)
    evals, evecs = la.eigh(H)
    idx = np.argsort(np.real(evals))
    return np.real(evals[idx]), evecs[:, idx]


def c3_decompose(k_frac, bonds, degen_tol=1e-8):
    """Simultaneously diagonalize H(k) and C3."""
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
            c3_diag[b] = conj(evecs[:, b]) @ C3_PERM @ evecs[:, b]
        else:
            sub = evecs[:, grp]
            C3_sub = conj(sub.T) @ C3_PERM @ sub
            c3_evals, c3_evecs = la.eig(C3_sub)
            order = np.argsort(np.angle(c3_evals))
            c3_evals = c3_evals[order]
            c3_evecs = c3_evecs[:, order]
            new_sub = sub @ c3_evecs
            for ig, b in enumerate(grp):
                new_evecs[:, b] = new_sub[:, ig]
                c3_diag[b] = c3_evals[ig]

    return evals, new_evecs, c3_diag


def label_c3(c3_val):
    if abs(c3_val - 1.0) < 0.3:
        return '1'
    elif abs(c3_val - omega3) < 0.3:
        return 'w'
    elif abs(c3_val - omega3**2) < 0.3:
        return 'w2'
    return '?'


# ==========================================================================
# TAKAGI DECOMPOSITION
# ==========================================================================

def takagi_decompose(M):
    """
    Takagi decomposition: M = U* D U^dagger for complex symmetric M.
    Returns (masses, U) with masses real non-negative, sorted ascending.
    """
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
    """
    Extract Majorana phases alpha_21, alpha_31 and Dirac phase delta_CP
    from a 3x3 unitary PMNS matrix U.

    Convention: U = V * diag(1, e^{i*alpha_21/2}, e^{i*alpha_31/2})
    where V is the standard Dirac parametrization.
    """
    # Make U[0,0] real positive (rephasing freedom)
    U_r = U.copy()
    for i in range(3):
        ph = np.angle(U_r[i, 0])
        U_r[i, :] *= exp(-1j * ph)
    ph0 = np.angle(U_r[0, 0])
    U_r[0, :] *= exp(-1j * ph0)

    # |U_e3| = s13
    s13 = abs(U_r[0, 2])
    s13 = min(s13, 1.0)

    # alpha_21 from column 1: U[0,1] = s12*c13 * e^{i*alpha_21/2}
    # (V[0,1] = s12*c13 is real positive in standard parametrization)
    alpha_21 = 2 * np.angle(U_r[0, 1])

    # alpha_31 from column 2: V[1,2] = s23*c13 is real positive
    # U[1,2] = s23*c13 * e^{i*alpha_31/2}
    alpha_31 = 2 * np.angle(U_r[1, 2])

    # delta_CP from U[0,2]: U[0,2] = s13 * e^{-i*delta} * e^{i*alpha_31/2}
    if s13 > 1e-10:
        delta_CP = alpha_31 / 2 - np.angle(U_r[0, 2])
    else:
        delta_CP = 0

    # Jarlskog invariant
    J = np.imag(U[0, 0] * conj(U[0, 2]) * conj(U[2, 0]) * U[2, 2])

    return {
        'alpha_21': alpha_21 * DEG % 360,
        'alpha_31': alpha_31 * DEG % 360,
        'delta_CP': delta_CP * DEG % 360,
        'J': J,
        's13': s13,
    }


# ==========================================================================
# STEP 1: VERIFY GENERATION STRUCTURE AT P
# ==========================================================================

def print_header(title):
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)
    print()


def step1_verify_generations(bonds):
    """Verify the generation splitting at P point."""
    print_header("STEP 1: GENERATION STRUCTURE AT P = (1/4, 1/4, 1/4)")

    k_P = [0.25, 0.25, 0.25]
    evals, evecs, c3d = c3_decompose(k_P, bonds)

    print("  Band structure at P:")
    gen_bands = {}
    for b in range(N_ATOMS):
        lab = label_c3(c3d[b])
        print(f"    Band {b}: E = {evals[b]:+.8f}  C3 = {c3d[b]:.4f}  label = {lab}")
        gen_bands[lab] = gen_bands.get(lab, [])
        gen_bands[lab].append((b, evals[b], evecs[:, b]))

    # Energy splitting between omega and omega^2
    if 'w' in gen_bands and 'w2' in gen_bands:
        E_w = gen_bands['w'][0][1]
        E_w2 = gen_bands['w2'][0][1]
        print(f"\n  E(omega)  = {E_w:+.10f}  (expected +sqrt(3) = {sqrt(3):+.10f})")
        print(f"  E(omega2) = {E_w2:+.10f}  (expected -sqrt(3) = {-sqrt(3):+.10f})")
        print(f"  Splitting = {E_w - E_w2:.10f}  (expected 2*sqrt(3) = {2*sqrt(3):.10f})")
        print(f"\n  >>> omega and omega^2 bands at DIFFERENT energies <<<")
    else:
        print("\n  WARNING: Could not identify omega/omega^2 bands")

    return evals, evecs, c3d, gen_bands


# ==========================================================================
# STEP 2: DIRAC MASS MATRIX FROM BZ-AVERAGED RESOLVENT
# ==========================================================================

def step2_dirac_mass(bonds, gen_bands):
    """
    Construct M_D via BZ-averaged resolvent in generation basis.

    (M_D)_mn = integral_BZ dk <gen_m | G(k, E_F) | gen_n>

    where G(k, E) = (E*I - H(k))^{-1} is the Green's function,
    and gen_m, gen_n are the C3 generation eigenstates at P.

    Physical meaning: the Dirac mass matrix elements are the amplitude
    for a particle in generation m to propagate to generation n,
    averaged over the BZ. The ENERGY DIFFERENCE between omega and omega^2
    bands means this propagator is NOT symmetric under m <-> n exchange
    in the complex plane.
    """
    print_header("STEP 2: DIRAC MASS MATRIX FROM BZ-AVERAGED RESOLVENT")

    # Generation states (defined at P, but used as projectors everywhere)
    gen_w  = GEN_STATES['gen1_w']    # omega generation
    gen_w2 = GEN_STATES['gen2_w2']   # omega^2 generation
    gen_s  = GEN_STATES['trivial_s'] # trivial symmetric (electron-like)

    # Use these three as the generation basis for the seesaw
    # gen_s = generation 1 (electron), gen_w = generation 2 (muon),
    # gen_w2 = generation 3 (tau)
    gen_basis = [gen_s, gen_w, gen_w2]
    gen_labels = ['1(trivial)', 'w(omega)', 'w2(omega^2)']

    # BZ integration grid (fractional coordinates)
    N_BZ = 24  # 24^3 = 13824 k-points
    dk = 1.0 / N_BZ
    vol = dk**3

    # Fermi energy: midgap between trivial bands and generation bands
    # At Gamma: E = {-1, -1, -1, 3} -> gap center at 1
    # At P: E = {-sqrt(3), -sqrt(3), +sqrt(3), +sqrt(3)} -> gap center at 0
    # Use E_F = 0 (particle-hole symmetric point)
    E_F = 0.0
    eta = 0.05  # broadening for Green's function (retarded)

    print(f"  BZ grid: {N_BZ}^3 = {N_BZ**3} k-points")
    print(f"  Fermi energy: E_F = {E_F}")
    print(f"  Broadening: eta = {eta}")

    # Compute resolvent matrix elements
    M_D_raw = np.zeros((3, 3), dtype=complex)

    for n1 in range(N_BZ):
        for n2 in range(N_BZ):
            for n3 in range(N_BZ):
                k = [(n1 + 0.5) * dk, (n2 + 0.5) * dk, (n3 + 0.5) * dk]
                H = bloch_H(k, bonds)
                # Retarded Green's function
                G = la.inv((E_F + 1j * eta) * np.eye(N_ATOMS) - H)

                for m in range(3):
                    for n in range(3):
                        M_D_raw[m, n] += vol * (conj(gen_basis[m]) @ G @ gen_basis[n])

    print(f"\n  Raw resolvent matrix (M_D_raw):")
    print(f"    Magnitudes:")
    for i in range(3):
        row = "    " + "  ".join(f"{abs(M_D_raw[i,j]):.6e}" for j in range(3))
        print(row)
    print(f"    Phases (deg):")
    for i in range(3):
        row = "    " + "  ".join(f"{np.angle(M_D_raw[i,j])*DEG:+8.2f}" for j in range(3))
        print(row)

    # The Dirac mass matrix is proportional to this resolvent.
    # Normalize so that the diagonal has magnitude ~ 1 for the seesaw.
    # The OVERALL scale drops out of CP phases.
    scale = abs(M_D_raw[0, 0])
    if scale < 1e-30:
        scale = la.norm(M_D_raw) / 3
    M_D = M_D_raw / scale

    print(f"\n  Normalized M_D (scale = {scale:.4e}):")
    print(f"    Magnitudes:")
    for i in range(3):
        row = "    " + "  ".join(f"{abs(M_D[i,j]):.6f}" for j in range(3))
        print(row)
    print(f"    Phases (deg):")
    for i in range(3):
        row = "    " + "  ".join(f"{np.angle(M_D[i,j])*DEG:+8.2f}" for j in range(3))
        print(row)

    # Check: is M_D circulant? (C3 symmetry should make it so)
    print(f"\n  Circulant check:")
    print(f"    |M_D[0,1] - M_D[1,2]| = {abs(M_D[0,1] - M_D[1,2]):.2e}")
    print(f"    |M_D[0,2] - M_D[1,0]| = {abs(M_D[0,2] - M_D[1,0]):.2e}")
    print(f"    |M_D[0,0] - M_D[1,1]| = {abs(M_D[0,0] - M_D[1,1]):.2e}")

    return M_D, M_D_raw


# ==========================================================================
# STEP 3: MAJORANA MASS MATRIX FROM IHARA/K4 STRUCTURE
# ==========================================================================

def step3_majorana_mass():
    """
    Construct M_R from the delocalized (|000>) sector.

    Neutrinos are |000> Fock states -- delocalized, no edge occupation.
    M_R comes from the K4 quotient spectral structure (Ihara zeta).

    K4 Ihara zeta triplet poles: u = (-1 +/- i*sqrt(7))/4
      |u| = sqrt(2)/2, phase = pi - arctan(sqrt(7)) ~ 110.7 deg

    The Majorana mass hierarchy uses R = dm31^2 / dm21^2 ~ 32.68.

    We construct M_R as a 3x3 matrix on the K4 triplet sector.
    The K4 adjacency on the triplet: eigenvalue -1 (triple degenerate).

    Structure: M_R_ij = M_0 * [delta_ij + epsilon * (J-I)_ij * e^{-i*phi}]
    where phi = arccos(1/3) for K4 enantiomer, and epsilon encodes the
    off-diagonal coupling from the Ihara pole structure.
    """
    print_header("STEP 3: MAJORANA MASS FROM IHARA / K4 STRUCTURE")

    # The Ihara triplet pole phase
    phi_ihara = np.arctan(sqrt(7))  # ~ 69.3 deg
    print(f"  Ihara triplet pole phase: arctan(sqrt(7)) = {phi_ihara * DEG:.4f} deg")
    print(f"  K4 dihedral: arccos(1/3) = {ARCCOS_1_3 * DEG:.4f} deg")

    # Option A: M_R from K4 enantiomer (RH srs) with arccos(1/3) phase
    # This is the standard seesaw_cp_proof.py approach
    phi_R = ARCCOS_1_3  # K4 dihedral for enantiomer

    # Off-diagonal coupling from Ihara pole structure
    # The triplet pole |u| = sqrt(2)/2, giving off-diagonal strength
    eps_R = sqrt(2) / 2  # Ihara pole modulus

    # Build M_R with Ihara-informed structure
    # Method 1: pure K4 phase (enantiomeric)
    M_R_K4 = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R_K4[i, j] = eps_R * exp(-1j * phi_R)

    print(f"\n  M_R (K4 enantiomer, eps={eps_R:.4f}, phi={phi_R*DEG:.2f} deg):")
    print(f"    Magnitudes:")
    for i in range(3):
        row = "    " + "  ".join(f"{abs(M_R_K4[i,j]):.6f}" for j in range(3))
        print(row)
    print(f"    Phases (deg):")
    for i in range(3):
        row = "    " + "  ".join(f"{np.angle(M_R_K4[i,j])*DEG:+8.2f}" for j in range(3))
        print(row)

    # Method 2: Use the FULL Ihara pole phase (not just K4 dihedral)
    # The triplet poles are at u = (-1 +/- i*sqrt(7))/4
    # Phase of u: pi - arctan(sqrt(7)) (second quadrant)
    phi_pole = pi - phi_ihara
    M_R_ihara = np.eye(3, dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R_ihara[i, j] = eps_R * exp(-1j * phi_pole)

    print(f"\n  M_R (Ihara pole phase = {phi_pole*DEG:.2f} deg):")
    print(f"    Magnitudes:")
    for i in range(3):
        row = "    " + "  ".join(f"{abs(M_R_ihara[i,j]):.6f}" for j in range(3))
        print(row)
    print(f"    Phases (deg):")
    for i in range(3):
        row = "    " + "  ".join(f"{np.angle(M_R_ihara[i,j])*DEG:+8.2f}" for j in range(3))
        print(row)

    # Method 3: Hierarchy-aware M_R
    # Include the (2/3)^g mass hierarchy in M_R diagonal
    # to get the physical neutrino mass ratio
    M_R_hier = np.diag([1.0, (2.0/3.0), (2.0/3.0)**2]).astype(complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                M_R_hier[i, j] = eps_R * sqrt(M_R_hier[i,i].real * M_R_hier[j,j].real) * exp(-1j * phi_R)

    print(f"\n  M_R (hierarchical + K4 enantiomer):")
    print(f"    Magnitudes:")
    for i in range(3):
        row = "    " + "  ".join(f"{abs(M_R_hier[i,j]):.6f}" for j in range(3))
        print(row)
    print(f"    Phases (deg):")
    for i in range(3):
        row = "    " + "  ".join(f"{np.angle(M_R_hier[i,j])*DEG:+8.2f}" for j in range(3))
        print(row)

    return M_R_K4, M_R_ihara, M_R_hier


# ==========================================================================
# STEP 4: SEESAW + TAKAGI DECOMPOSITION
# ==========================================================================

def step4_seesaw(M_D, M_R, label=""):
    """
    Compute seesaw: M_nu = -M_D^T M_R^{-1} M_D
    Takagi decompose, extract phases.
    """
    M_R_inv = la.inv(M_R)
    M_nu = -M_D.T @ M_R_inv @ M_D

    # Verify complex symmetric (Majorana condition)
    asym = la.norm(M_nu - M_nu.T)
    if la.norm(M_nu) > 0:
        asym /= la.norm(M_nu)

    # Takagi decomposition
    masses, U = takagi_decompose(M_nu)

    # Extract phases
    phases = extract_majorana_phases(U)

    return {
        'M_nu': M_nu,
        'masses': masses,
        'U': U,
        'asym': asym,
        'phases': phases,
        'label': label,
    }


def print_seesaw_result(result):
    """Print seesaw results."""
    print(f"\n  --- {result['label']} ---")
    M_nu = result['M_nu']
    print(f"    M_nu magnitudes (normalized):")
    for i in range(3):
        row = "      " + "  ".join(f"{abs(M_nu[i,j]):.6f}" for j in range(3))
        print(row)
    print(f"    M_nu phases (deg):")
    for i in range(3):
        row = "      " + "  ".join(f"{np.angle(M_nu[i,j])*DEG:+8.2f}" for j in range(3))
        print(row)
    print(f"    Symmetry: ||M_nu - M_nu^T||/||M_nu|| = {result['asym']:.2e}")

    masses = result['masses']
    if masses[0] > 0:
        print(f"    Takagi masses (ratio): {masses / masses[0]}")
    else:
        print(f"    Takagi masses: {masses}")

    if masses[0] > 0 and masses[1] > masses[0]:
        dm21 = masses[1]**2 - masses[0]**2
        dm31 = masses[2]**2 - masses[0]**2
        if dm21 > 0:
            ratio = dm31 / dm21
            err = abs(ratio - R_IHARA) / R_IHARA * 100
            print(f"    dm31^2/dm21^2 = {ratio:.2f}  (target {R_IHARA:.2f}, err {err:.1f}%)")

    ph = result['phases']
    print(f"    PMNS phases:")
    print(f"      alpha_21 = {ph['alpha_21']:.2f} deg  (target {TARGET_ALPHA_21:.1f} deg)")
    print(f"      alpha_31 = {ph['alpha_31']:.2f} deg  (target {TARGET_ALPHA_31:.1f} deg)")
    print(f"      delta_CP = {ph['delta_CP']:.2f} deg  (target {TARGET_DELTA_CP_B:.1f} deg)")
    print(f"      J (Jarlskog) = {ph['J']:.6e}")
    print(f"      s13 = {ph['s13']:.6f}")

    # Errors
    err_a21 = min(abs(ph['alpha_21'] - TARGET_ALPHA_21),
                  360 - abs(ph['alpha_21'] - TARGET_ALPHA_21))
    err_a31 = min(abs(ph['alpha_31'] - TARGET_ALPHA_31),
                  360 - abs(ph['alpha_31'] - TARGET_ALPHA_31))
    err_dcp = min(abs(ph['delta_CP'] - TARGET_DELTA_CP_B),
                  360 - abs(ph['delta_CP'] - TARGET_DELTA_CP_B))
    print(f"    Errors: |da21|={err_a21:.1f} deg, |da31|={err_a31:.1f} deg, |ddCP|={err_dcp:.1f} deg")


# ==========================================================================
# STEP 5: COMMUTATOR [P_D, P_R] ANALYSIS
# ==========================================================================

def step5_commutator(M_D, M_R_K4, M_R_list):
    """
    Compare [P_D, P_R] on K4 (should be 0) vs full lattice (should be nonzero).

    P_D = projection onto Dirac mass matrix structure
    P_R = projection onto Majorana mass matrix structure

    On K4, both are functions of J (all-ones matrix), so they commute.
    On the full lattice, M_D has BZ-resolved structure that breaks this.
    """
    print_header("STEP 5: COMMUTATOR [P_D, P_R] — K4 vs FULL LATTICE")

    # K4 phase matrices (pure phase, magnitudes = 1)
    phi_K4 = ARCCOS_1_3
    P_D_K4 = np.ones((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                P_D_K4[i, j] = exp(1j * phi_K4)

    P_R_K4 = np.ones((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                P_R_K4[i, j] = exp(-1j * phi_K4)

    comm_K4 = P_D_K4 @ P_R_K4 - P_R_K4 @ P_D_K4
    print(f"  K4 commutator ||[P_D, P_R]|| = {la.norm(comm_K4):.2e}")
    print(f"  (Expected: 0 because both are functions of J = all-ones matrix)")

    # Verify: P_D_K4 = I + (e^{i*phi}-1)*(J-I)/1 where off-diag = e^{i*phi}
    # P_D_K4 = (1 - e^{i*phi})*I + e^{i*phi}*J
    # P_R_K4 = (1 - e^{-i*phi})*I + e^{-i*phi}*J
    # Both are linear in I and J, and I and J commute, so [P_D, P_R] = 0. QED.
    print(f"  Algebraic proof: P_D = a*I + b*J, P_R = c*I + d*J")
    a = 1 - exp(1j * phi_K4)
    b = exp(1j * phi_K4)
    print(f"    a = {a:.4f}, b = {b:.4f}")
    c = 1 - exp(-1j * phi_K4)
    d = exp(-1j * phi_K4)
    print(f"    c = {c:.4f}, d = {d:.4f}")
    print(f"    [aI+bJ, cI+dJ] = (ad-bc)[I,J] + ... = 0 since [I,J] = 0")

    # Full lattice: M_D from resolvent is NOT of the form a*I + b*J
    # Check by projecting M_D onto I and J
    J = np.ones((3, 3))
    I3 = np.eye(3)
    # M_D = alpha*I + beta*J + remainder
    alpha_fit = np.trace(M_D @ (I3 - J/3)) / 2  # proj onto (I - J/3)
    beta_fit = np.trace(M_D @ J) / 9             # proj onto J/3
    M_D_fit = alpha_fit * I3 + beta_fit * J
    remainder = M_D - M_D_fit
    frac_remainder = la.norm(remainder) / la.norm(M_D) if la.norm(M_D) > 0 else 0

    print(f"\n  Full-lattice M_D decomposition into I + J:")
    print(f"    alpha (I coeff) = {alpha_fit:.6f}")
    print(f"    beta (J coeff)  = {beta_fit:.6f}")
    print(f"    ||remainder|| / ||M_D|| = {frac_remainder:.6f}")
    if frac_remainder > 0.01:
        print(f"    >>> M_D is NOT a function of J alone <<<")
        print(f"    >>> The BZ resolvent breaks the K4 degeneracy <<<")
    else:
        print(f"    M_D is approximately a*I + b*J (K4-like)")
        print(f"    Commutator will be approximately zero")

    # Compute [M_D, M_R] for each M_R variant
    for M_R, lab in M_R_list:
        comm = M_D @ M_R - M_R @ M_D
        print(f"\n  Full-lattice ||[M_D, M_R]|| ({lab}): {la.norm(comm):.6e}")
        if la.norm(M_D) > 0 and la.norm(M_R) > 0:
            rel = la.norm(comm) / (la.norm(M_D) * la.norm(M_R))
            print(f"    Relative: {rel:.6e}")

    # Also check [M_D^dag M_D, M_R^dag M_R] which determines if CP phases survive
    for M_R, lab in M_R_list:
        HH_D = conj(M_D.T) @ M_D
        HH_R = conj(M_R.T) @ M_R
        comm2 = HH_D @ HH_R - HH_R @ HH_D
        print(f"\n  ||[M_D^dag M_D, M_R^dag M_R]|| ({lab}): {la.norm(comm2):.6e}")
        if la.norm(HH_D) > 0 and la.norm(HH_R) > 0:
            rel2 = la.norm(comm2) / (la.norm(HH_D) * la.norm(HH_R))
            print(f"    Relative: {rel2:.6e}")


# ==========================================================================
# STEP 6: ALTERNATIVE — DIRECT LATTICE SEESAW (no resolvent)
# ==========================================================================

def step6_direct_lattice_seesaw(bonds):
    """
    Alternative approach: build M_D directly from P-point eigenstates
    and their energy eigenvalues, without BZ integration.

    At P, the generation states have energies:
      omega band: E_w = +sqrt(3)
      omega^2 band: E_w2 = -sqrt(3)
      trivial: E_0 is known

    M_D in the seesaw comes from NB walk amplitudes.
    The energy difference introduces a PHASE DIFFERENCE
    between the omega and omega^2 contributions.
    """
    print_header("STEP 6: DIRECT P-POINT SEESAW (energy-split generations)")

    k_P = [0.25, 0.25, 0.25]
    evals, evecs, c3d = c3_decompose(k_P, bonds)

    # Find bands by C3 label
    band_info = {}
    for b in range(N_ATOMS):
        lab = label_c3(c3d[b])
        band_info[lab] = band_info.get(lab, [])
        band_info[lab].append((b, evals[b], evecs[:, b]))

    E_w = band_info['w'][0][1]    # +sqrt(3)
    E_w2 = band_info['w2'][0][1]  # -sqrt(3)

    print(f"  E(omega) = {E_w:.8f}")
    print(f"  E(omega^2) = {E_w2:.8f}")
    print(f"  Energy ratio: E_w / E_w2 = {E_w / E_w2:.8f}")

    # The key: Dirac mass elements pick up a phase from the energy difference
    # In the propagator: G(E) ~ 1/(E - H), the generation-dependent energy
    # means <gen_m|G|gen_n> picks up phases from the pole structure.
    #
    # For the seesaw, the relevant quantity is:
    #   M_D proportional to the Yukawa coupling in the generation basis
    #
    # With energy-split generations, the "Yukawa" at scale mu has:
    #   y_mn ~ <gen_m|H|gen_n> at the P point
    # But H is diagonal in the generation basis at P, so this gives
    # a diagonal M_D at P. That's too simple.
    #
    # The CP violation comes from the INTERPLAY between:
    #   1. The P-point structure (diagonal, energy-split)
    #   2. The K4 quotient structure (off-diagonal, phase from arccos(1/3))
    #
    # Build M_D that interpolates: use energy-dependent magnitudes
    # with K4 off-diagonal phases.

    phi_K4 = ARCCOS_1_3

    # Generation energies determine mass magnitudes
    # m_D ~ E (in natural units, the Yukawa ~ energy)
    # But we need to be careful about signs (E_w2 < 0)
    # Use |E| for magnitude, sign -> phase

    results = []

    # Approach A: magnitude from |E|, off-diagonal from K4 dihedral
    print(f"\n  Approach A: |E|-magnitude + K4 off-diagonal phase")
    E_gen = [1.0, abs(E_w), abs(E_w2)]  # trivial, omega, omega^2
    M_D_A = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            if i == j:
                M_D_A[i, j] = E_gen[i]
            else:
                M_D_A[i, j] = sqrt(E_gen[i] * E_gen[j]) * exp(1j * phi_K4)

    # Approach B: use actual (signed) energies on diagonal
    # omega band is POSITIVE energy, omega^2 is NEGATIVE
    # This sign difference IS the source of CP violation
    print(f"\n  Approach B: signed-E diagonal + K4 off-diagonal phase")
    E_signed = [1.0, E_w, E_w2]  # 1, +sqrt(3), -sqrt(3)
    M_D_B = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            if i == j:
                M_D_B[i, j] = E_signed[i]
            else:
                M_D_B[i, j] = sqrt(abs(E_signed[i] * E_signed[j])) * exp(1j * phi_K4)

    # Approach C: Full complex — diagonal = E (signed), off-diagonal
    # includes BOTH K4 dihedral AND energy-dependent running
    print(f"\n  Approach C: complex diagonal (E as phase) + K4 off-diag")
    M_D_C = np.zeros((3, 3), dtype=complex)
    # Encode energy difference as a phase:
    # phi_gen = arctan(E/scale) where scale is the bandwidth
    bandwidth = E_w - E_w2  # 2*sqrt(3)
    for i in range(3):
        for j in range(3):
            E_avg = (E_signed[i] + E_signed[j]) / 2
            phi_E = np.arctan2(E_signed[i] - E_signed[j], bandwidth)
            if i == j:
                M_D_C[i, j] = abs(E_signed[i]) * exp(1j * phi_E / 2)
            else:
                M_D_C[i, j] = sqrt(abs(E_signed[i] * E_signed[j])) * exp(1j * (phi_K4 + phi_E))

    approaches = [
        (M_D_A, "A: |E|-magnitude + K4 phase"),
        (M_D_B, "B: signed-E diagonal + K4 phase"),
        (M_D_C, "C: E-as-phase + K4 phase"),
    ]

    # Use M_R variants from step 3
    _, _, M_R_hier = step3_results  # will be set in main

    for M_D_var, M_D_label in approaches:
        for M_R_var, M_R_label in [(M_R_K4_global, "K4"), (M_R_ihara_global, "Ihara"),
                                    (M_R_hier_global, "Hierarchical")]:
            result = step4_seesaw(M_D_var, M_R_var,
                                  f"Direct: {M_D_label} x M_R({M_R_label})")
            results.append(result)

    return results


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == '__main__':
    print("=" * 76)
    print("  MAJORANA CP PHASES FROM FULL SRS LATTICE SEESAW")
    print("  Key question: does [P_D, P_R] != 0 on the full lattice?")
    print("=" * 76)

    # Build lattice
    bonds = find_bonds()
    print(f"\n  Found {len(bonds)} bonds in primitive cell")

    # Step 1: verify generation splitting
    evals, evecs, c3d, gen_bands = step1_verify_generations(bonds)

    # Step 2: Dirac mass from BZ resolvent
    M_D, M_D_raw = step2_dirac_mass(bonds, gen_bands)

    # Step 3: Majorana mass matrices
    M_R_K4, M_R_ihara, M_R_hier = step3_majorana_mass()
    # Store globally for step 6
    step3_results = (M_R_K4, M_R_ihara, M_R_hier)
    M_R_K4_global = M_R_K4
    M_R_ihara_global = M_R_ihara
    M_R_hier_global = M_R_hier

    # Step 4: Seesaw with BZ-resolvent M_D
    print_header("STEP 4: SEESAW RESULTS (BZ-resolvent M_D)")

    seesaw_results = []
    for M_R, lab in [(M_R_K4, "K4 enantiomer"), (M_R_ihara, "Ihara pole"),
                     (M_R_hier, "Hierarchical")]:
        result = step4_seesaw(M_D, M_R, f"BZ-resolvent x M_R({lab})")
        seesaw_results.append(result)
        print_seesaw_result(result)

    # Step 5: Commutator analysis
    M_R_list = [(M_R_K4, "K4"), (M_R_ihara, "Ihara"), (M_R_hier, "Hierarchical")]
    step5_commutator(M_D, M_R_K4, M_R_list)

    # Step 6: Direct P-point seesaw approaches
    print_header("STEP 6: DIRECT P-POINT SEESAW RESULTS")
    direct_results = step6_direct_lattice_seesaw(bonds)
    for r in direct_results:
        print_seesaw_result(r)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_header("SUMMARY: HONEST ASSESSMENT")

    print("  Question: Does the full srs lattice give nonzero Majorana CP phases")
    print("  where K4 quotient gives zero?")
    print()

    # Collect all alpha_31 results
    all_results = seesaw_results + direct_results
    print("  All alpha_31 results:")
    for r in all_results:
        a31 = r['phases']['alpha_31']
        a21 = r['phases']['alpha_21']
        dcp = r['phases']['delta_CP']
        print(f"    {r['label'][:55]:55s}  a21={a21:6.1f}  a31={a31:6.1f}  dCP={dcp:6.1f}")

    print(f"\n  Targets: alpha_21={TARGET_ALPHA_21} deg, alpha_31={TARGET_ALPHA_31} deg, "
          f"delta_CP={TARGET_DELTA_CP_B} deg")

    # Check if ANY result hits targets
    best_a21_err = 360
    best_a31_err = 360
    best_result = None
    for r in all_results:
        a21 = r['phases']['alpha_21']
        a31 = r['phases']['alpha_31']
        err_a21 = min(abs(a21 - TARGET_ALPHA_21), 360 - abs(a21 - TARGET_ALPHA_21))
        err_a31 = min(abs(a31 - TARGET_ALPHA_31), 360 - abs(a31 - TARGET_ALPHA_31))
        total_err = err_a21 + err_a31
        if total_err < best_a21_err + best_a31_err:
            best_a21_err = err_a21
            best_a31_err = err_a31
            best_result = r

    if best_result:
        print(f"\n  Best result: {best_result['label']}")
        print(f"    alpha_21 error: {best_a21_err:.1f} deg")
        print(f"    alpha_31 error: {best_a31_err:.1f} deg")

    print(f"\n  K4 commutator check: [P_D, P_R] on K4 = 0 (proven algebraically)")
    print(f"  Full lattice commutator: see Step 5 above")
    print()
    print("  DONE.")
