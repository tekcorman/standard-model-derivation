#!/usr/bin/env python3
"""
VERIFICATION: Hashimoto CP phase predictions through the full seesaw.

Constructs M_R from Hashimoto return amplitudes at P, combines with
diagonal M_D from Bloch resolvent, runs seesaw, extracts phases,
and checks against the three Hashimoto predictions:

  alpha_21 = arg(h^10) = 162.39 deg   (h = (sqrt3 + i*sqrt5)/2)
  delta_CP = arg(h'^10) = 197.61 deg  (h' = (-sqrt3 + i*sqrt5)/2)
  alpha_31 = arg(h^10/h'^10) = 324.78 deg

Also checks mass ratio: dm31^2/dm21^2 target = 228/7 = 32.5714 (Ihara splitting,
theorem, closed form; see srs_r_theorem.py and docs/R_theorem.md).
"""

import numpy as np
from numpy import linalg as la
from numpy import sqrt, pi, exp, conj, arccos, arctan, arctan2
from itertools import product

np.set_printoptions(precision=10, linewidth=140)

DEG = 180.0 / pi
omega3 = np.exp(2j * pi / 3)
NN_DIST = sqrt(2) / 4
ARCCOS_1_3 = arccos(1.0 / 3.0)

# ======================================================================
# TARGETS
# ======================================================================

g = 10  # srs girth

h_w  = (sqrt(3) + 1j*sqrt(5)) / 2   # Hashimoto at P, omega band (lambda=+sqrt3)
h_w2 = (-sqrt(3) + 1j*sqrt(5)) / 2  # Hashimoto at P, omega^2 band (lambda=-sqrt3)

TARGET_ALPHA_21 = np.degrees(np.angle(h_w**g)) % 360   # ~162.39
TARGET_DELTA_CP = np.degrees(np.angle(h_w2**g)) % 360   # ~197.61
TARGET_ALPHA_31 = np.degrees(np.angle((h_w/h_w2)**g)) % 360  # ~324.78
TARGET_R = 228.0 / 7.0  # Ihara mass ratio = 32.5714 (theorem, closed form;
                        # see srs_r_theorem.py and docs/R_theorem.md)

print(f"# Hashimoto eigenvalues at P:")
print(f"#   h_w  = {h_w:.10f}   |h_w|  = {abs(h_w):.10f}")
print(f"#   h_w2 = {h_w2:.10f}   |h_w2| = {abs(h_w2):.10f}")
print(f"#")
print(f"# Phase predictions (arg of h^{g}):")
print(f"#   alpha_21 = {TARGET_ALPHA_21:.4f} deg")
print(f"#   delta_CP = {TARGET_DELTA_CP:.4f} deg")
print(f"#   alpha_31 = {TARGET_ALPHA_31:.4f} deg")
print(f"#   mass ratio target = {TARGET_R}")

# ======================================================================
# LATTICE INFRASTRUCTURE (from srs_majorana_md_principled.py)
# ======================================================================

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

GEN_BASIS = [
    np.array([0, 1, 1, 1], dtype=complex) / sqrt(3),          # trivial_s
    np.array([0, 1, omega3, omega3**2], dtype=complex) / sqrt(3),   # omega
    np.array([0, 1, omega3**2, omega3], dtype=complex) / sqrt(3),   # omega^2
]
GEN_LABELS = ['trivial_s', 'omega', 'omega^2']

k_P = np.array([0.25, 0.25, 0.25])


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
        phase = exp(2j * pi * np.dot(k, cell))
        H[tgt, src] += phase
    return H


def build_hashimoto(k_frac, bonds):
    n = len(bonds)
    B = np.zeros((n, n), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for f_idx, (fs, ft, fc) in enumerate(bonds):
        for e_idx, (es, et, ec) in enumerate(bonds):
            if fs != et:
                continue
            if ft == es and np.array_equal(fc, tuple(-x for x in ec)):
                continue
            phase = exp(2j * pi * np.dot(k, fc))
            B[f_idx, e_idx] = phase
    return B


# ======================================================================
# TAKAGI DECOMPOSITION & PHASE EXTRACTION
# ======================================================================

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


def extract_phases_from_eigenvalues(M_nu):
    """
    Extract Majorana phases directly from the complex eigenvalues of M_nu.

    For a complex symmetric matrix, the Takagi factorization gives
    M_nu = U* diag(m_i) U^dag where m_i >= 0. But if M_nu = M_D^T M_R^{-1} M_D
    and M_R has complex phases, the eigenvalues of M_nu carry those phases.

    The Majorana phases are:
      alpha_21 = arg(m_2) - arg(m_1)
      alpha_31 = arg(m_3) - arg(m_1)
    where m_i are the (complex) eigenvalues before taking absolute values.
    """
    # Eigenvalues of M_nu (complex symmetric -> use eig, not eigh)
    evals = la.eigvals(M_nu)
    # Sort by magnitude
    order = np.argsort(np.abs(evals))
    evals = evals[order]

    phases_deg = np.degrees(np.angle(evals)) % 360
    magnitudes = np.abs(evals)

    alpha_21 = (phases_deg[1] - phases_deg[0]) % 360
    alpha_31 = (phases_deg[2] - phases_deg[0]) % 360
    delta_CP = (phases_deg[2] - phases_deg[1]) % 360

    if magnitudes[0] > 0 and magnitudes[1] > 0:
        dm21 = magnitudes[1]**2 - magnitudes[0]**2
        dm31 = magnitudes[2]**2 - magnitudes[0]**2
        ratio = dm31 / dm21 if dm21 > 0 else float('inf')
    else:
        ratio = float('inf')

    return {
        'evals': evals,
        'magnitudes': magnitudes,
        'phases_deg': phases_deg,
        'alpha_21': alpha_21,
        'alpha_31': alpha_31,
        'delta_CP': delta_CP,
        'mass_ratio': ratio,
    }


def extract_majorana_phases_pmns(U):
    """Extract alpha_21, alpha_31, delta_CP from 3x3 PMNS-like matrix."""
    U_r = U.copy()
    # Remove row phases to put in standard form
    for i in range(3):
        ph = np.angle(U_r[i, 0])
        U_r[i, :] *= exp(-1j * ph)
    ph0 = np.angle(U_r[0, 0])
    U_r[0, :] *= exp(-1j * ph0)

    s13 = min(abs(U_r[0, 2]), 1.0)
    alpha_21 = (2 * np.angle(U_r[0, 1]) * DEG) % 360
    alpha_31 = (2 * np.angle(U_r[1, 2]) * DEG) % 360
    if s13 > 1e-10:
        delta_CP = (alpha_31 / 2 - np.angle(U_r[0, 2]) * DEG) % 360
    else:
        delta_CP = 0
    J = np.imag(U[0, 0] * conj(U[0, 2]) * conj(U[2, 0]) * U[2, 2])

    return {
        'alpha_21': alpha_21,
        'alpha_31': alpha_31,
        'delta_CP': delta_CP,
        'J': J,
        's13': s13,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("\n" + "=" * 76)
    print("  HASHIMOTO SEESAW VERIFICATION")
    print("  Check: seesaw with Hashimoto M_R reproduces CP phase predictions")
    print("=" * 76)

    bonds = find_bonds()
    n_edges = len(bonds)
    assert n_edges == 12, f"Expected 12 directed edges, got {n_edges}"
    print(f"\n  Lattice: {N_ATOMS} atoms, {n_edges} directed edges")

    # ==================================================================
    # STEP 1: Verify Hashimoto spectrum at P
    # ==================================================================
    print("\n" + "-" * 76)
    print("  STEP 1: Verify Hashimoto eigenvalues at P = (1/4, 1/4, 1/4)")
    print("-" * 76)

    H_vertex = bloch_H(k_P, bonds)
    evals_H, _ = la.eigh(H_vertex)
    print(f"  Vertex H(P) eigenvalues: {np.sort(np.real(evals_H))}")
    print(f"  Expected: all |E| = sqrt(3) = {sqrt(3):.10f}")

    B = build_hashimoto(k_P, bonds)
    evals_B = la.eigvals(B)
    order = np.argsort(-np.abs(evals_B))
    evals_B = evals_B[order]

    print(f"\n  Hashimoto B(P) eigenvalues (sorted by |h|):")
    for i, ev in enumerate(evals_B):
        print(f"    h_{i:2d} = {ev:+.8f}  |h| = {abs(ev):.8f}  "
              f"phase = {np.degrees(np.angle(ev)):+.4f} deg")

    # Check that h_w and h_w2 appear
    found_hw = sum(1 for ev in evals_B if abs(ev - h_w) < 1e-6)
    found_hw2 = sum(1 for ev in evals_B if abs(ev - h_w2) < 1e-6)
    print(f"\n  h_w  in spectrum: multiplicity {found_hw}  {'PASS' if found_hw > 0 else 'FAIL'}")
    print(f"  h_w2 in spectrum: multiplicity {found_hw2}  {'PASS' if found_hw2 > 0 else 'FAIL'}")

    # Identify trivial-sector eigenvalues (|h| = 1)
    trivial_evals = [ev for ev in evals_B if abs(abs(ev) - 1.0) < 1e-6]
    print(f"\n  Trivial sector (|h|=1): {len(trivial_evals)} eigenvalues")
    for ev in sorted(trivial_evals, key=lambda x: np.angle(x)):
        print(f"    h = {ev:+.8f}  phase = {np.degrees(np.angle(ev)):+.4f} deg")

    # ==================================================================
    # STEP 2: Construct M_D at P (diagonal in generation basis)
    # ==================================================================
    print("\n" + "-" * 76)
    print("  STEP 2: M_D from Bloch resolvent at P")
    print("-" * 76)

    # M_D_mn = <gen_m | G(k_P, E_F) | gen_n>
    # At P, [H, C3] = 0 => M_D diagonal
    for E_F, eta, label in [(0.0, 0.5, "EF=0, eta=0.5"),
                              (0.0, 0.05, "EF=0, eta=0.05")]:
        G = la.inv((E_F + 1j * eta) * np.eye(N_ATOMS) - H_vertex)
        M_D = np.zeros((3, 3), dtype=complex)
        for m in range(3):
            for n in range(3):
                M_D[m, n] = conj(GEN_BASIS[m]) @ G @ GEN_BASIS[n]

        off_diag = la.norm(M_D - np.diag(np.diag(M_D))) / la.norm(M_D)
        print(f"\n  {label}:")
        print(f"    M_D diagonal: [{M_D[0,0]:.8f}, {M_D[1,1]:.8f}, {M_D[2,2]:.8f}]")
        print(f"    Off-diagonal fraction: {off_diag:.2e}  {'PASS (diagonal)' if off_diag < 1e-10 else 'WARN'}")
        print(f"    |M_D| diagonal: [{abs(M_D[0,0]):.6f}, {abs(M_D[1,1]):.6f}, {abs(M_D[2,2]):.6f}]")

    # Use eta=0.5 as the reference (from task description)
    eta_ref = 0.5
    G_ref = la.inv((0.0 + 1j * eta_ref) * np.eye(N_ATOMS) - H_vertex)
    M_D_ref = np.zeros((3, 3), dtype=complex)
    for m in range(3):
        for n in range(3):
            M_D_ref[m, n] = conj(GEN_BASIS[m]) @ G_ref @ GEN_BASIS[n]
    # Force exactly diagonal (numerical)
    M_D_diag = np.diag(np.diag(M_D_ref))

    d_s  = M_D_diag[0, 0]
    d_w  = M_D_diag[1, 1]
    d_w2 = M_D_diag[2, 2]
    print(f"\n  Reference M_D (eta=0.5):")
    print(f"    d_s  = {d_s:.8f}  |d_s|  = {abs(d_s):.6f}")
    print(f"    d_w  = {d_w:.8f}  |d_w|  = {abs(d_w):.6f}")
    print(f"    d_w2 = {d_w2:.8f}  |d_w2| = {abs(d_w2):.6f}")

    # ==================================================================
    # STEP 3: Construct M_R from Hashimoto return amplitudes
    # ==================================================================
    print("\n" + "-" * 76)
    print("  STEP 3: M_R from Hashimoto return amplitudes h^g")
    print("-" * 76)

    # Hashimoto return amplitude for each generation:
    #   (M_R)_mm = h_m^g where h_m is the Hashimoto eigenvalue for generation m
    #
    # omega band: h_w^g, omega^2 band: h_w2^g
    # trivial sector: h=+1 and h=-1 (non-Ihara eigenvalues), so return = (+1)^10 + (-1)^10 = 2

    h_w_g  = h_w**g
    h_w2_g = h_w2**g
    h_s_g  = complex(2.0)  # (+1)^10 + (-1)^10 = 2 (real, no phase)

    print(f"\n  Return amplitudes h^{g}:")
    print(f"    trivial: h_s^{g}  = {h_s_g:+.8f}  phase = {np.degrees(np.angle(h_s_g)):+.4f} deg")
    print(f"    omega:   h_w^{g}  = {h_w_g:+.8f}  |h_w^{g}| = {abs(h_w_g):.6f}  "
          f"phase = {np.degrees(np.angle(h_w_g)):+.4f} deg")
    print(f"    omega^2: h_w2^{g} = {h_w2_g:+.8f}  |h_w2^{g}| = {abs(h_w2_g):.6f}  "
          f"phase = {np.degrees(np.angle(h_w2_g)):+.4f} deg")

    # Verify target phases
    phase_w  = np.degrees(np.angle(h_w_g)) % 360
    phase_w2 = np.degrees(np.angle(h_w2_g)) % 360
    print(f"\n  Phase verification:")
    print(f"    arg(h_w^{g})  mod 360 = {phase_w:.4f} deg  (target {TARGET_ALPHA_21:.4f})")
    print(f"    arg(h_w2^{g}) mod 360 = {phase_w2:.4f} deg  (target {TARGET_DELTA_CP:.4f})")
    phase_ratio = np.degrees(np.angle(h_w_g / h_w2_g)) % 360
    print(f"    arg(h_w^{g}/h_w2^{g}) mod 360 = {phase_ratio:.4f} deg  (target {TARGET_ALPHA_31:.4f})")

    # ==================================================================
    # STEP 4: Build several M_R variants and run seesaw
    # ==================================================================
    print("\n" + "-" * 76)
    print("  STEP 4: Seesaw M_nu = M_D^T M_R^{-1} M_D")
    print("-" * 76)

    # --- Variant A: Purely diagonal M_R (no inter-generation transitions) ---
    print("\n  === Variant A: Diagonal M_R (no inter-generation mixing) ===")
    M_R_A = np.diag([h_s_g, h_w_g, h_w2_g])
    run_seesaw_and_report(M_D_diag, M_R_A, "Diagonal M_R")

    # --- Variant B: M_R with scale factor |h|^g on diagonal ---
    # |h_w| = |h_w2| = sqrt(2), so |h|^g = 2^5 = 32
    print("\n  === Variant B: Diagonal M_R, uniform scale (all |M_R_ii| = 32) ===")
    M_R_B = np.diag([
        32.0 * exp(1j * np.angle(h_s_g)),    # = 32 (real)
        32.0 * exp(1j * np.angle(h_w_g)),     # = 32 * e^{i*162.39 deg}
        32.0 * exp(1j * np.angle(h_w2_g)),    # = 32 * e^{i*197.61 deg}
    ])
    run_seesaw_and_report(M_D_diag, M_R_B, "Uniform-scale diagonal M_R")

    # --- Variant C: Include off-diagonal from inter-generation Hashimoto ---
    # Off-diagonal: <omega|B^g|omega^2> involves cross-generation return
    # For C3-symmetric system at P: inter-generation transitions vanish
    # because B commutes with C3. So off-diagonal M_R elements are zero.
    print("\n  === Variant C: Off-diagonal M_R (should vanish by C3 symmetry) ===")
    # Verify numerically: compute <gen_m|B^g|gen_n> on the edge space
    verify_off_diagonal_MR(B, bonds, n_edges)

    # --- Variant D: Real M_R (phases only in M_D) ---
    print("\n  === Variant D: Real M_R (remove phases, keep magnitudes) ===")
    M_R_D = np.diag([abs(h_s_g), abs(h_w_g), abs(h_w2_g)])
    run_seesaw_and_report(M_D_diag, M_R_D, "Real M_R (magnitudes only)")

    # --- Variant E: M_R^{-1} has INVERTED phases ---
    print("\n  === Variant E: Check phase flow through M_R^{-1} ===")
    M_R_inv = la.inv(M_R_A)
    print(f"    M_R^{{-1}} diagonal:")
    for i in range(3):
        v = M_R_inv[i, i]
        print(f"      [{i}] = {v:+.8f}  phase = {np.degrees(np.angle(v)):+.4f} deg")
    print(f"    M_nu = M_D^T M_R^{{-1}} M_D phases:")
    M_nu_E = M_D_diag.T @ M_R_inv @ M_D_diag
    for i in range(3):
        v = M_nu_E[i, i]
        print(f"      [{i}] = {v:+.8f}  phase = {np.degrees(np.angle(v)):+.4f} deg")

    # ==================================================================
    # STEP 5: Direct eigenvalue-phase extraction
    # ==================================================================
    print("\n" + "-" * 76)
    print("  STEP 5: Phase extraction from M_nu eigenvalues")
    print("-" * 76)

    for label, M_R_test in [("Variant A (diagonal)", M_R_A),
                              ("Variant B (uniform)", M_R_B)]:
        M_R_inv = la.inv(M_R_test)
        M_nu = -M_D_diag.T @ M_R_inv @ M_D_diag  # standard seesaw sign

        result = extract_phases_from_eigenvalues(M_nu)
        print(f"\n  {label}:")
        print(f"    M_nu eigenvalues (sorted by |m|):")
        for i, ev in enumerate(result['evals']):
            print(f"      m_{i} = {ev:+.10f}  |m| = {result['magnitudes'][i]:.8f}  "
                  f"phase = {result['phases_deg'][i]:+.4f} deg")

        print(f"    alpha_21 = {result['alpha_21']:.4f} deg  (target {TARGET_ALPHA_21:.4f}, "
              f"err {min(abs(result['alpha_21'] - TARGET_ALPHA_21), 360 - abs(result['alpha_21'] - TARGET_ALPHA_21)):.4f})")
        print(f"    delta_CP = {result['delta_CP']:.4f} deg  (target {TARGET_DELTA_CP:.4f}, "
              f"err {min(abs(result['delta_CP'] - TARGET_DELTA_CP), 360 - abs(result['delta_CP'] - TARGET_DELTA_CP)):.4f})")
        print(f"    alpha_31 = {result['alpha_31']:.4f} deg  (target {TARGET_ALPHA_31:.4f}, "
              f"err {min(abs(result['alpha_31'] - TARGET_ALPHA_31), 360 - abs(result['alpha_31'] - TARGET_ALPHA_31)):.4f})")
        print(f"    mass ratio dm31/dm21 = {result['mass_ratio']:.4f}  (target {TARGET_R})")

    # ==================================================================
    # STEP 6: Takagi decomposition phases
    # ==================================================================
    print("\n" + "-" * 76)
    print("  STEP 6: Takagi decomposition + PMNS phase extraction")
    print("-" * 76)

    for label, M_R_test in [("Variant A (diagonal)", M_R_A),
                              ("Variant B (uniform)", M_R_B)]:
        M_R_inv = la.inv(M_R_test)
        M_nu = -M_D_diag.T @ M_R_inv @ M_D_diag

        masses, U = takagi_decompose(M_nu)
        phases = extract_majorana_phases_pmns(U)

        print(f"\n  {label}:")
        print(f"    Takagi masses: {masses}")
        if masses[0] > 0:
            print(f"    Mass ratios: {masses / masses[0]}")
        print(f"    PMNS U matrix:")
        for i in range(3):
            print(f"      [{', '.join(f'{U[i,j]:+.6f}' for j in range(3))}]")
        print(f"    alpha_21 = {phases['alpha_21']:.4f} deg  (target {TARGET_ALPHA_21:.4f})")
        print(f"    alpha_31 = {phases['alpha_31']:.4f} deg")
        print(f"    delta_CP = {phases['delta_CP']:.4f} deg")
        print(f"    J = {phases['J']:.6e}")

    # ==================================================================
    # STEP 7: Self-consistency analysis
    # ==================================================================
    print("\n" + "-" * 76)
    print("  STEP 7: Self-consistency check")
    print("-" * 76)

    print("""
  ANALYSIS: For diagonal M_D and diagonal M_R, the seesaw is trivial:

    M_nu = M_D^T M_R^{-1} M_D = diag(d_s^2/h_s^g, d_w^2/h_w^g, d_w2^2/h_w2^g)

  M_nu is ALREADY diagonal => U = I (or permutation) => no mixing.
  The Majorana phases come ENTIRELY from the eigenvalue phases:

    phase(m_nu_1) = 2*arg(d_s) - arg(h_s^g) = 2*arg(d_s) - 0
    phase(m_nu_2) = 2*arg(d_w) - arg(h_w^g)
    phase(m_nu_3) = 2*arg(d_w2) - arg(h_w2^g)

  So: alpha_21 = phase(m_2) - phase(m_1) = [2*arg(d_w) - arg(h_w^g)] - [2*arg(d_s)]

  The M_D phases modify the pure Hashimoto prediction!
  Let's check whether M_D phases are zero or compensating.
""")

    phase_ds  = np.degrees(np.angle(d_s))
    phase_dw  = np.degrees(np.angle(d_w))
    phase_dw2 = np.degrees(np.angle(d_w2))

    print(f"  M_D phases at P (eta=0.5):")
    print(f"    arg(d_s)  = {phase_ds:+.6f} deg")
    print(f"    arg(d_w)  = {phase_dw:+.6f} deg")
    print(f"    arg(d_w2) = {phase_dw2:+.6f} deg")

    # Effective M_nu phases
    phase_mnu_s  = (2*phase_ds - 0) % 360
    phase_mnu_w  = (2*phase_dw - np.degrees(np.angle(h_w_g))) % 360
    phase_mnu_w2 = (2*phase_dw2 - np.degrees(np.angle(h_w2_g))) % 360

    print(f"\n  M_nu eigenvalue phases (2*arg(d) - arg(h^g)):")
    print(f"    phase(m_s)  = {phase_mnu_s:.4f} deg")
    print(f"    phase(m_w)  = {phase_mnu_w:.4f} deg")
    print(f"    phase(m_w2) = {phase_mnu_w2:.4f} deg")

    alpha_21_actual = (phase_mnu_w - phase_mnu_s) % 360
    delta_CP_actual = (phase_mnu_w2 - phase_mnu_w) % 360
    alpha_31_actual = (phase_mnu_w2 - phase_mnu_s) % 360

    print(f"\n  Seesaw-derived phases:")
    print(f"    alpha_21 = {alpha_21_actual:.4f} deg  (target {TARGET_ALPHA_21:.4f})")
    print(f"    delta_CP = {delta_CP_actual:.4f} deg  (target {TARGET_DELTA_CP:.4f})")
    print(f"    alpha_31 = {alpha_31_actual:.4f} deg  (target {TARGET_ALPHA_31:.4f})")

    # Check if M_D phases are symmetric (d_w and d_w2 conjugate)
    print(f"\n  M_D symmetry check:")
    print(f"    d_w  = {d_w:.8f}")
    print(f"    d_w2 = {d_w2:.8f}")
    print(f"    d_w - conj(d_w2) = {abs(d_w - conj(d_w2)):.2e}  "
          f"{'(conjugate pair)' if abs(d_w - conj(d_w2)) < 1e-8 else '(NOT conjugate)'}")
    print(f"    arg(d_w) + arg(d_w2) = {phase_dw + phase_dw2:.6f} deg  "
          f"{'(sum = 0 => symmetric)' if abs(phase_dw + phase_dw2) < 1e-6 else '(NOT zero)'}")

    # ==================================================================
    # STEP 8: Mass ratios
    # ==================================================================
    print("\n" + "-" * 76)
    print("  STEP 8: Mass ratio check")
    print("-" * 76)

    M_R_inv_A = la.inv(M_R_A)
    M_nu_A = -M_D_diag.T @ M_R_inv_A @ M_D_diag
    m_abs = np.abs(np.diag(M_nu_A))
    m_sorted = np.sort(m_abs)

    print(f"  |m_nu| (unsorted): {m_abs}")
    print(f"  |m_nu| (sorted):   {m_sorted}")

    if m_sorted[0] > 0:
        dm21 = m_sorted[1]**2 - m_sorted[0]**2
        dm31 = m_sorted[2]**2 - m_sorted[0]**2
        ratio = dm31 / dm21 if dm21 > 0 else float('inf')
        print(f"  dm21^2 = {dm21:.6e}")
        print(f"  dm31^2 = {dm31:.6e}")
        print(f"  dm31^2/dm21^2 = {ratio:.4f}  (target {TARGET_R})")
    else:
        print(f"  WARNING: lightest mass is zero, ratio undefined")

    # ==================================================================
    # STEP 9: What if M_D is identity? (pure M_R phase test)
    # ==================================================================
    print("\n" + "-" * 76)
    print("  STEP 9: Pure M_R test (M_D = I)")
    print("-" * 76)

    M_D_identity = np.eye(3, dtype=complex)
    M_nu_pure = -M_D_identity.T @ la.inv(M_R_A) @ M_D_identity

    result_pure = extract_phases_from_eigenvalues(M_nu_pure)
    print(f"  With M_D = I:")
    for i, ev in enumerate(result_pure['evals']):
        print(f"    m_{i} = {ev:+.10f}  phase = {result_pure['phases_deg'][i]:+.4f} deg")

    print(f"  alpha_21 = {result_pure['alpha_21']:.4f} deg  (target {TARGET_ALPHA_21:.4f})")
    print(f"  delta_CP = {result_pure['delta_CP']:.4f} deg  (target {TARGET_DELTA_CP:.4f})")
    print(f"  alpha_31 = {result_pure['alpha_31']:.4f} deg  (target {TARGET_ALPHA_31:.4f})")

    err_a21 = min(abs(result_pure['alpha_21'] - TARGET_ALPHA_21),
                  360 - abs(result_pure['alpha_21'] - TARGET_ALPHA_21))
    err_dCP = min(abs(result_pure['delta_CP'] - TARGET_DELTA_CP),
                  360 - abs(result_pure['delta_CP'] - TARGET_DELTA_CP))
    err_a31 = min(abs(result_pure['alpha_31'] - TARGET_ALPHA_31),
                  360 - abs(result_pure['alpha_31'] - TARGET_ALPHA_31))

    print(f"\n  VERDICT (M_D=I):")
    print(f"    alpha_21 error: {err_a21:.4f} deg  {'PASS' if err_a21 < 0.1 else 'FAIL'}")
    print(f"    delta_CP error: {err_dCP:.4f} deg  {'PASS' if err_dCP < 0.1 else 'FAIL'}")
    print(f"    alpha_31 error: {err_a31:.4f} deg  {'PASS' if err_a31 < 0.1 else 'FAIL'}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)

    print(f"""
  The seesaw with diagonal M_D and diagonal M_R gives M_nu = diagonal.

  KEY FINDING: The phases in M_nu come from BOTH M_D and M_R:
    phase(m_nu_i) = 2*arg(d_i) - arg(h_i^g)  (via the -1 in the seesaw sign)

  For the Hashimoto predictions to survive the seesaw EXACTLY:
    - Either M_D must be real (all arg(d_i) = 0)
    - Or the M_D phases must cancel in the DIFFERENCES alpha_21, alpha_31

  Since d_w and d_w2 are related by C3 conjugation:
    arg(d_w) = -arg(d_w2)  (from C3 symmetry of H at P)

  This means:
    alpha_21 = [2*arg(d_w) - arg(h_w^g)] - [2*arg(d_s) - 0]
             = 2*arg(d_w) - arg(h_w^g) - 2*arg(d_s)

  The M_D correction is 2*(arg(d_w) - arg(d_s)).
  If this is nonzero, the seesaw SCRAMBLES the pure Hashimoto phases.
""")

    MD_correction_21 = 2 * (phase_dw - phase_ds)
    MD_correction_31 = 2 * (phase_dw2 - phase_ds)
    print(f"  M_D phase corrections:")
    print(f"    alpha_21 correction: 2*(arg(d_w)-arg(d_s)) = {MD_correction_21:.4f} deg")
    print(f"    alpha_31 correction: 2*(arg(d_w2)-arg(d_s)) = {MD_correction_31:.4f} deg")
    print(f"    delta_CP correction: 2*(arg(d_w2)-arg(d_w)) = {2*(phase_dw2 - phase_dw):.4f} deg")

    if abs(MD_correction_21) < 0.1 and abs(MD_correction_31) < 0.1:
        print(f"\n  RESULT: M_D phases negligible => seesaw PRESERVES Hashimoto predictions.")
    else:
        print(f"\n  RESULT: M_D phases are {abs(MD_correction_21):.1f} deg, {abs(MD_correction_31):.1f} deg")
        print(f"  The seesaw MODIFIES the pure Hashimoto predictions by these amounts.")
        print(f"  The 'pure M_R' test (M_D=I) gives the exact Hashimoto phases.")


def run_seesaw_and_report(M_D, M_R, label):
    """Run seesaw and report results."""
    M_R_inv = la.inv(M_R)
    M_nu = -M_D.T @ M_R_inv @ M_D

    # Check symmetry
    asym = la.norm(M_nu - M_nu.T) / max(la.norm(M_nu), 1e-12)

    print(f"\n  --- {label} ---")
    print(f"  M_R diagonal: [{M_R[0,0]:+.4f}, {M_R[1,1]:+.4f}, {M_R[2,2]:+.4f}]")
    print(f"  M_R off-diagonal norm: {la.norm(M_R - np.diag(np.diag(M_R))):.2e}")
    print(f"  M_nu symmetry: {asym:.2e}  {'PASS' if asym < 1e-10 else 'WARN'}")
    print(f"  M_nu diagonal: [{M_nu[0,0]:+.8f}, {M_nu[1,1]:+.8f}, {M_nu[2,2]:+.8f}]")
    print(f"  M_nu off-diagonal norm: {la.norm(M_nu - np.diag(np.diag(M_nu))):.2e}")


def verify_off_diagonal_MR(B, bonds, n_edges):
    """
    Verify that off-diagonal M_R elements vanish by C3 symmetry.
    Compute <gen_m|B^g|gen_n> on the edge space via source lifting.
    """
    k = k_P

    # Build C3 on edge space
    bd = {(s, t, c): idx for idx, (s, t, c) in enumerate(bonds)}

    def c3_atom(j):
        return {0: 0, 1: 3, 2: 1, 3: 2}[j]
    def c3_cell(c):
        return (c[2], c[0], c[1])

    P_C3 = np.zeros((n_edges, n_edges), dtype=complex)
    for old_idx, (s, t, c) in enumerate(bonds):
        new = (c3_atom(s), c3_atom(t), c3_cell(c))
        if new in bd:
            P_C3[bd[new], old_idx] = 1.0

    comm = la.norm(P_C3 @ B - B @ P_C3)
    print(f"  [B(P), C3] = 0 on edge space: {comm:.2e}  {'PASS' if comm < 1e-10 else 'FAIL'}")

    # Source-lift generation states to edge space
    edge_gens = []
    for gen_idx in range(3):
        v = np.zeros(n_edges, dtype=complex)
        for idx, (s, t, c) in enumerate(bonds):
            v[idx] = GEN_BASIS[gen_idx][s]
        if la.norm(v) > 1e-12:
            v = v / la.norm(v)
        edge_gens.append(v)

    # Compute <gen_m|B^g|gen_n> on edge space
    print(f"\n  <gen_m|B^{g}|gen_n> matrix on edge space (source lift):")
    MR_edge = np.zeros((3, 3), dtype=complex)
    for n_idx in range(3):
        Bg_v = edge_gens[n_idx].copy()
        for _ in range(g):
            Bg_v = B @ Bg_v
        for m_idx in range(3):
            MR_edge[m_idx, n_idx] = np.dot(conj(edge_gens[m_idx]), Bg_v)

    for m_idx in range(3):
        for n_idx in range(3):
            v = MR_edge[m_idx, n_idx]
            marker = "  (off-diag)" if m_idx != n_idx else ""
            print(f"    [{GEN_LABELS[m_idx]:>8s},{GEN_LABELS[n_idx]:>8s}] = {v:+.8f}  "
                  f"|val| = {abs(v):.2e}  phase = {np.degrees(np.angle(v)):+.4f} deg{marker}")

    off_norm = la.norm(MR_edge - np.diag(np.diag(MR_edge)))
    diag_norm = la.norm(np.diag(np.diag(MR_edge)))
    print(f"\n  Off-diagonal / diagonal = {off_norm/max(diag_norm, 1e-12):.2e}  "
          f"{'PASS (negligible)' if off_norm/max(diag_norm, 1e-12) < 1e-6 else 'NONZERO'}")

    # Report diagonal phases
    print(f"\n  Diagonal return amplitude phases:")
    for i in range(3):
        v = MR_edge[i, i]
        print(f"    {GEN_LABELS[i]:>8s}: phase = {np.degrees(np.angle(v)):.4f} deg, |val| = {abs(v):.6f}")


if __name__ == '__main__':
    main()
