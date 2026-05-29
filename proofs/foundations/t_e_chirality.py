#!/usr/bin/env python3
"""
T_e chirality analysis — edge-reversal Z_2 involution on the srs 12-dim fiber.

CLAIM UNDER TEST
----------------
The A1 toggle T_e (edge-reversal map: e -> e_bar) is a Z_2 involution on the
12-dim directed-edge fiber of the srs lattice.  In the Standard Model, fermion
masses arise from chiral mixing — off-diagonal coupling between left-handed
(T_e = -1) and right-handed (T_e = +1) sectors.

The prior result (an internal working note) showed that the
spectral projector P (B(P)-eigenspace) gives Q_BP = 0 (eigenspace identity,
BLOCKED).  This script tests the CHIRAL projector P_R / P_L (T_e-eigenspace
projectors within V_Ram), checking whether P_R B(P) P_L != 0 since [T_e, B(P)]
may be nonzero (B(P) is directional — it depends on edge orientation).

COMPUTATION STEPS
-----------------
1. Build B(P): 12x12 Hashimoto Bloch at P = (1/4, 1/4, 1/4).
2. Build T_e: 12x12 permutation matrix for edge-reversal.
3. Check [T_e, B(P)] (commutator) and {T_e, B(P)} (anti-commutator).
4. Extract V_Ram (8 eigenvectors of B(P) with |eigenvalue|^2 ~ 2).
5. Decompose V_Ram under T_e: project T_e onto V_Ram; find eigenvalues.
6. Compute P_R B(P) P_L (chiral mixing matrix); find singular values.
7. Extract phases of singular values and compare to delta_obs and arg(h)/4.
8. Structural diagnosis.

UPSTREAM FILES
--------------
- proofs/common.py (find_bonds, C3_PERM, omega3)
- proofs/foundations/theorem_B5_3_core.py (bloch_hashimoto infrastructure)
- ../../predictions/B_P_doubly_degenerate_h_derivation.md (B(P) spectrum, V_Ram structure)
- docs/theorem_B5_3_core.md (C_3-isotypic decomposition)
- docs/framework/framework_axioms.md (A1 toggle, A2 MDL, A3 purification)

Run with:
    PYTHONPATH=. python3 proofs/foundations/t_e_chirality.py
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, C3_PERM, omega3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2   # h = (sqrt(3)+i*sqrt(5))/2
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))       # arg(h) in radians
ARG_H_DEG = math.degrees(ARG_H)                       # ~52.24 deg
DELTA_OBS = 12.735                                     # PDG delta_obs in degrees
K_P = (0.25, 0.25, 0.25)                              # P-point in reduced coords


# ---------------------------------------------------------------------------
# Infrastructure: directed edges and Bloch Hashimoto B(k)
# (pattern from koide_delta_phase.py and t_v_eigenstructure.py)
# ---------------------------------------------------------------------------

def build_directed_edges(bonds):
    """Return list of (src, tgt, cell) tuples for all 12 directed edges."""
    directed = [tuple(b) for b in bonds]
    assert len(directed) == 12, f"Expected 12 directed edges, got {len(directed)}"
    return directed


def bloch_hashimoto(k_frac, directed):
    """12x12 Bloch Hashimoto B(k) on directed edges.

    B(k)[e', e] = exp(2*pi*i * k . cell_{e'})   if e -> e' is a valid NB step,
                  0                              otherwise.

    Valid NB: target(e) = source(e') and e' != reverse(e).
    Reverse of (src, tgt, cell) is (tgt, src, -cell).
    """
    n = len(directed)
    B = np.zeros((n, n), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for i_p, (src_p, tgt_p, cell_p) in enumerate(directed):
        for i_e, (src_e, tgt_e, cell_e) in enumerate(directed):
            if tgt_e != src_p:
                continue
            is_reverse = (tgt_p == src_e and
                          tuple(np.array(cell_p) + np.array(cell_e)) == (0, 0, 0))
            if is_reverse:
                continue
            phase = np.exp(2j * np.pi * np.dot(k, cell_p))
            B[i_p, i_e] += phase
    return B


# ---------------------------------------------------------------------------
# Step 2: Build T_e (edge-reversal permutation matrix)
# ---------------------------------------------------------------------------

def build_t_e(directed):
    """
    Build the 12x12 edge-reversal involution T_e.

    For each directed edge i = (src, tgt, cell), find the index j of the
    reversed edge (tgt, src, -cell).  T_e is the permutation matrix with
    T_e[j, i] = 1, i.e., T_e maps column-vector e_i to e_j.

    This is the ABSTRACT permutation on labeled directed edges; at a given
    Bloch k-point, the relevant operator on the Bloch-transformed basis is
    T_e conjugated by the Bloch phase matrix D(k) = diag(exp(2*pi*i * k . cell_i)).
    We compute both the abstract T_e and the Bloch-conjugated version T_e^Bloch(k).

    Returns:
        T_e   : (12, 12) complex permutation matrix (abstract, k-independent)
        sigma : list of length 12, sigma[i] = j such that T_e[j, i] = 1
    """
    n = len(directed)
    edge_to_idx = {de: i for i, de in enumerate(directed)}

    sigma = []
    for i, (src, tgt, cell) in enumerate(directed):
        rev_cell = tuple(-c for c in cell)
        rev_edge = (tgt, src, rev_cell)
        j = edge_to_idx.get(rev_edge)
        if j is None:
            raise RuntimeError(
                f"Edge {i} = {(src, tgt, cell)} has no reverse {rev_edge} in directed list"
            )
        sigma.append(j)

    T_e = np.zeros((n, n), dtype=complex)
    for i, j in enumerate(sigma):
        T_e[j, i] = 1.0

    return T_e, sigma


def bloch_phase_matrix(k_frac, directed):
    """
    D(k) = diag(exp(2*pi*i * k . cell_i)) for edge i with lattice vector cell_i.

    This is the diagonal gauge factor that appears in the Fourier transform to
    the Bloch basis.  The Bloch-conjugated T_e is D(k) T_e D(k)^{-1}.
    """
    k = np.asarray(k_frac, dtype=float)
    n = len(directed)
    phases = np.array([
        np.exp(2j * np.pi * np.dot(k, cell)) for (src, tgt, cell) in directed
    ])
    return np.diag(phases)


# ---------------------------------------------------------------------------
# C_3 infrastructure (reused from koide_delta_phase.py pattern)
# ---------------------------------------------------------------------------

def c3_vertex_perm():
    """C_3 vertex permutation sigma = (v0)(v1 v3 v2) from common.C3_PERM."""
    perm = {}
    for i in range(4):
        for j in range(4):
            if abs(C3_PERM[i, j] - 1.0) < 1e-12:
                perm[j] = i
    assert perm == {0: 0, 1: 3, 2: 1, 3: 2}
    return perm


def c3_cell_perm(cell):
    return (cell[2], cell[0], cell[1])


def build_c3_on_directed_edges(directed):
    """12x12 permutation matrix U_{C_3} for the C_3 action on directed edges."""
    vp = c3_vertex_perm()
    n = len(directed)
    edge_to_idx = {de: i for i, de in enumerate(directed)}
    U = np.zeros((n, n), dtype=complex)
    for i, (src, tgt, cell) in enumerate(directed):
        new_edge = (vp[src], vp[tgt], c3_cell_perm(cell))
        j = edge_to_idx.get(new_edge)
        if j is None:
            raise RuntimeError(f"C_3 mapped {(src, tgt, cell)} -> {new_edge} not found")
        U[j, i] = 1.0
    return U


def c3_isotypic_dims_of_projector(projector, U_C3, dim, tol=0.1):
    """
    Given an orthogonal projector (dim x dim), compute the C_3-isotypic
    dimensions of its image.

    Strategy: find the eigenvectors of projector with eigenvalue ~1 (the image),
    orthonormalize, project U_C3 onto that subspace, count eigenvalues near
    1, omega, omega^2.
    """
    evals_proj, evecs_proj = la.eigh(projector)
    image_idx = [i for i, ev in enumerate(evals_proj) if abs(ev - 1.0) < tol]
    if not image_idx:
        return (0, 0, 0)
    image_basis = evecs_proj[:, image_idx]
    Q, _ = la.qr(image_basis)
    Q = Q[:, :len(image_idx)]
    U_sub = Q.conj().T @ U_C3 @ Q
    u_evals = la.eigvals(U_sub)
    m1, mw, mw2 = 0, 0, 0
    for ev in u_evals:
        if abs(ev - 1.0) < tol:
            m1 += 1
        elif abs(ev - omega3) < tol:
            mw += 1
        elif abs(ev - omega3 ** 2) < tol:
            mw2 += 1
        else:
            pass  # may be intermediate if subspace mixes sectors
    return (m1, mw, mw2)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("T_e chirality analysis — edge-reversal involution on srs fiber")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Step 1: Build B(P) and T_e
    # -----------------------------------------------------------------------
    print()
    print("Step 1 — Build B(P) and T_e")

    bonds = find_bonds()
    directed = build_directed_edges(bonds)

    B_P = bloch_hashimoto(K_P, directed)
    assert B_P.shape == (12, 12), f"B(P) shape {B_P.shape}"
    print(f"  B(P) shape: {B_P.shape}")

    T_e, sigma = build_t_e(directed)
    assert T_e.shape == (12, 12), f"T_e shape {T_e.shape}"
    print(f"  T_e shape: {T_e.shape}")

    # Verify T_e^2 = I (involution)
    T_e_sq = T_e @ T_e
    involution_err = la.norm(T_e_sq - np.eye(12))
    print(f"  ||T_e^2 - I|| = {involution_err:.3e}   (expected 0: involution check)")
    assert involution_err < 1e-10, f"T_e is not an involution: ||T_e^2 - I|| = {involution_err}"

    # Verify T_e is a real permutation matrix (all entries 0 or 1)
    assert np.allclose(T_e, T_e.real), "T_e has imaginary parts"
    assert np.allclose(np.sort(T_e.real.flatten()), np.sort(
        np.concatenate([np.ones(12), np.zeros(12 * 12 - 12)])
    )), "T_e is not a permutation matrix"
    print(f"  T_e verified as real permutation matrix with T_e^2 = I")

    # Print the reversal permutation
    print(f"  Reversal permutation sigma (i -> sigma[i]):")
    print(f"    {sigma}")
    # Verify sigma is a fixed-point-free involution (no edge is its own reverse)
    for i, j in enumerate(sigma):
        assert sigma[j] == i, f"sigma is not involutive at i={i}: sigma[sigma[{i}]] = {sigma[j]}"
        assert i != j, f"Edge {i} is its own reverse (self-loop)"
    print(f"  sigma is fixed-point-free: no edge is its own reverse. OK.")

    # -----------------------------------------------------------------------
    # Step 2: Build Bloch-conjugated T_e^Bloch = D(k) T_e D(k)^{-1}
    # -----------------------------------------------------------------------
    print()
    print("Step 2 — Bloch-conjugated chirality operator T_e^Bloch(P)")

    D_P = bloch_phase_matrix(K_P, directed)
    D_P_inv = np.diag(np.conj(np.diag(D_P)))   # D is diagonal unitary, D^{-1} = D^*
    T_e_Bloch = D_P @ T_e @ D_P_inv

    # Check whether T_e_Bloch is still an involution
    T_e_Bloch_sq_err = la.norm(T_e_Bloch @ T_e_Bloch - np.eye(12))
    print(f"  ||T_e^Bloch(P)^2 - I|| = {T_e_Bloch_sq_err:.3e}")
    # T_e_Bloch is generally NOT an involution unless all phases are ±1

    # -----------------------------------------------------------------------
    # Step 3: Check [T_e, B(P)] and {T_e, B(P)}
    # -----------------------------------------------------------------------
    print()
    print("Step 3 — Commutator [T_e, B(P)] and anti-commutator {T_e, B(P)}")

    comm = T_e @ B_P - B_P @ T_e
    anticomm = T_e @ B_P + B_P @ T_e

    max_comm = np.max(np.abs(comm))
    max_anticomm = np.max(np.abs(anticomm))
    norm_comm = la.norm(comm)
    norm_anticomm = la.norm(anticomm)

    print(f"  max |[T_e, B(P)]|    = {max_comm:.6f}   (norm = {norm_comm:.6f})")
    print(f"  max |{{T_e, B(P)}}|    = {max_anticomm:.6f}   (norm = {norm_anticomm:.6f})")

    # Also check T_e @ B - B.conj() @ T_e (phase-conjugate relation)
    B_conj = np.conj(B_P)
    mixed_comm = T_e @ B_P - B_conj @ T_e
    max_mixed_comm = np.max(np.abs(mixed_comm))
    print(f"  max |T_e B(P) - B*(P) T_e| = {max_mixed_comm:.6f}")

    # Report whether B(P) commutes or anti-commutes with T_e
    if max_comm < 1e-8:
        comm_status = "COMMUTES: [T_e, B(P)] = 0"
    elif max_anticomm < 1e-8:
        comm_status = "ANTI-COMMUTES: {T_e, B(P)} = 0"
    else:
        comm_status = f"NEITHER: [T_e, B(P)] nonzero (max {max_comm:.4f}), {{T_e, B(P)}} nonzero (max {max_anticomm:.4f})"
    print(f"  Status: {comm_status}")

    if max_mixed_comm < 1e-8:
        print(f"  Phase-conjugate relation T_e B(P) = B*(P) T_e: HOLDS")
        phase_conj_holds = True
    else:
        print(f"  Phase-conjugate relation T_e B(P) = B*(P) T_e: does NOT hold")
        phase_conj_holds = False

    # Also check Bloch-conjugated T_e
    comm_Bloch = T_e_Bloch @ B_P - B_P @ T_e_Bloch
    anticomm_Bloch = T_e_Bloch @ B_P + B_P @ T_e_Bloch
    max_comm_Bloch = np.max(np.abs(comm_Bloch))
    max_anticomm_Bloch = np.max(np.abs(anticomm_Bloch))
    print(f"  Bloch-conjugated: max |[T_e^Bloch, B(P)]| = {max_comm_Bloch:.6f}")
    print(f"  Bloch-conjugated: max |{{T_e^Bloch, B(P)}}| = {max_anticomm_Bloch:.6f}")

    # -----------------------------------------------------------------------
    # Step 4: Extract V_Ram
    # -----------------------------------------------------------------------
    print()
    print("Step 4 — Extract V_Ram (8 eigenvectors with |eigenvalue|^2 ~ 2)")

    evals_B, evecs_B = la.eig(B_P)

    ram_idx = [i for i, ev in enumerate(evals_B) if abs(abs(ev) ** 2 - 2.0) < 1e-5]
    tree_idx = [i for i, ev in enumerate(evals_B) if abs(abs(ev) ** 2 - 1.0) < 1e-5]

    print(f"  Ramanujan eigenvalues (|mu|^2 = 2): {len(ram_idx)} eigenvectors")
    print(f"  Tree      eigenvalues (|mu|^2 = 1): {len(tree_idx)} eigenvectors")

    assert len(ram_idx) == 8, f"Expected 8 Ramanujan eigenvectors, got {len(ram_idx)}"
    assert len(tree_idx) == 4, f"Expected 4 tree eigenvectors, got {len(tree_idx)}"

    h_targets = [H_EXACT, H_EXACT.conjugate(), -H_EXACT, -H_EXACT.conjugate()]
    for mu in evals_B[ram_idx]:
        assert any(abs(mu - t) < 1e-5 for t in h_targets), (
            f"Ramanujan eigenvalue {mu} not in {{h, h*, -h, -h*}}"
        )

    evals_ram = evals_B[ram_idx]
    print(f"  B(P) eigenvalues in V_Ram:")
    for ev in sorted(evals_ram, key=lambda z: (round(z.real, 4), round(z.imag, 4))):
        print(f"    {ev.real:+.6f}{ev.imag:+.6f}i  |mu|={abs(ev):.6f}  arg={math.degrees(np.angle(ev)):+.4f} deg")

    # Orthonormal basis for V_Ram
    evecs_ram_raw = evecs_B[:, ram_idx]   # 12 x 8
    V_Ram, _ = la.qr(evecs_ram_raw)
    V_Ram = V_Ram[:, :8]   # 12 x 8 orthonormal basis

    # Verify V_Ram is 8-dim
    rank_V = la.matrix_rank(V_Ram)
    assert rank_V == 8, f"V_Ram rank = {rank_V}"

    # -----------------------------------------------------------------------
    # Step 5: Decompose V_Ram under T_e
    # -----------------------------------------------------------------------
    print()
    print("Step 5 — Decompose V_Ram under T_e")

    # Project T_e onto V_Ram: T_e_Ram = V_Ram^dag T_e V_Ram (8x8 matrix)
    T_e_Ram = V_Ram.conj().T @ T_e @ V_Ram

    # Find eigenvalues of T_e_Ram
    evals_T_Ram, evecs_T_Ram = la.eig(T_e_Ram)

    print(f"  T_e restricted to V_Ram: eigenvalues")
    for ev in sorted(evals_T_Ram, key=lambda z: (round(z.real, 4), round(z.imag, 4))):
        print(f"    {ev.real:+.6f}{ev.imag:+.6f}i   |ev| = {abs(ev):.6f}")

    # Count eigenvalues near +1, -1, and complex
    tol_eig = 0.1
    n_plus1 = sum(1 for ev in evals_T_Ram if abs(ev - 1.0) < tol_eig)
    n_minus1 = sum(1 for ev in evals_T_Ram if abs(ev + 1.0) < tol_eig)
    n_complex = sum(1 for ev in evals_T_Ram
                    if abs(ev - 1.0) >= tol_eig and abs(ev + 1.0) >= tol_eig)

    print(f"  Multiplicity of eigenvalue +1: {n_plus1}")
    print(f"  Multiplicity of eigenvalue -1: {n_minus1}")
    print(f"  Multiplicity of complex (non-real) eigenvalues: {n_complex}")
    print(f"  Total: {n_plus1 + n_minus1 + n_complex} (expected 8)")

    assert n_plus1 + n_minus1 + n_complex == 8, "Eigenvalue count mismatch"

    # Report the T_e-eigenvalue structure
    if n_plus1 == 4 and n_minus1 == 4:
        print(f"  => V_Ram splits into 4-dim right-handed (T_e=+1) and 4-dim left-handed (T_e=-1)")
        chiral_split = True
    elif n_plus1 > 0 and n_minus1 > 0:
        print(f"  => V_Ram has partial chiral split: ({n_plus1}, {n_minus1}) under T_e = (+1, -1)")
        chiral_split = True
    else:
        print(f"  => V_Ram does NOT split into T_e = +-1 eigenspaces (no real +-1 eigenvalues)")
        chiral_split = False

    # -----------------------------------------------------------------------
    # Step 6: Build P_R, P_L and compute M = P_R B(P) P_L
    # -----------------------------------------------------------------------
    print()
    print("Step 6 — Chiral mixing matrix M = P_R B(P) P_L within V_Ram")

    if not chiral_split:
        print("  BLOCKED: T_e has no +-1 eigenvalues on V_Ram; cannot form chiral projectors.")
        print("  Gap: T_e restricted to V_Ram has no real eigenvalues => no chiral decomposition.")
        # We still compute the projection to document what T_e does
        print("  Proceeding with near-eigenvalue projectors for diagnostic purposes...")

    # Build projectors from T_e_Ram eigenvectors
    # For each eigenvector of T_e_Ram with eigenvalue near +1: right-handed (R)
    # For each eigenvector of T_e_Ram with eigenvalue near -1: left-handed (L)

    R_cols = [k for k, ev in enumerate(evals_T_Ram) if abs(ev - 1.0) < tol_eig]
    L_cols = [k for k, ev in enumerate(evals_T_Ram) if abs(ev + 1.0) < tol_eig]

    print(f"  Right-handed (T_e=+1) eigenvectors in V_Ram-basis: {len(R_cols)}")
    print(f"  Left-handed  (T_e=-1) eigenvectors in V_Ram-basis: {len(L_cols)}")

    if R_cols and L_cols:
        # Build subspace bases in full 12-dim space
        # V_Ram @ evecs_T_Ram[:, R_cols] gives right-handed subspace
        evecs_T_Ram_R = evecs_T_Ram[:, R_cols]   # 8 x n_R in V_Ram basis
        evecs_T_Ram_L = evecs_T_Ram[:, L_cols]   # 8 x n_L in V_Ram basis

        V_R_raw = V_Ram @ evecs_T_Ram_R   # 12 x n_R in full space
        V_L_raw = V_Ram @ evecs_T_Ram_L   # 12 x n_L in full space

        # Orthonormalize
        Q_R, _ = la.qr(V_R_raw)
        Q_R = Q_R[:, :len(R_cols)]
        Q_L, _ = la.qr(V_L_raw)
        Q_L = Q_L[:, :len(L_cols)]

        # Projectors
        P_R = Q_R @ Q_R.conj().T   # 12 x 12
        P_L = Q_L @ Q_L.conj().T   # 12 x 12

        # Verify projectors
        err_PR = la.norm(P_R @ P_R - P_R)
        err_PL = la.norm(P_L @ P_L - P_L)
        print(f"  ||P_R^2 - P_R|| = {err_PR:.3e},  ||P_L^2 - P_L|| = {err_PL:.3e}")
        assert err_PR < 1e-8, f"P_R is not a projector: {err_PR}"
        assert err_PL < 1e-8, f"P_L is not a projector: {err_PL}"

        # Chiral mixing matrix M = P_R @ B_P @ P_L (12 x 12 matrix)
        M = P_R @ B_P @ P_L

        # SVD of M
        U_svd, sv, Vh_svd = la.svd(M)
        print(f"  Singular values of M = P_R B(P) P_L:")
        for i, s in enumerate(sv):
            print(f"    sigma_{i} = {s:.8f}")

        # Expected Koide Yukawa magnitudes: (2, sqrt(2), sqrt(2))
        koide_expected = np.array([2.0, math.sqrt(2), math.sqrt(2)])
        print(f"  Expected Koide magnitudes: (2, sqrt(2), sqrt(2)) = ({koide_expected[0]:.6f}, {koide_expected[1]:.6f}, {koide_expected[2]:.6f})")

        # Compare leading singular values (only nonzero ones are meaningful)
        sv_nonzero = sv[sv > 1e-8]
        print(f"  Nonzero singular values: {sv_nonzero}")

        if len(sv_nonzero) >= 1:
            match_2 = abs(sv_nonzero[0] - 2.0) < 0.1
            print(f"  Leading singular value {sv_nonzero[0]:.6f} matches 2.0: {match_2}")
        if len(sv_nonzero) >= 3:
            match_sqrt2 = (abs(sv_nonzero[1] - math.sqrt(2)) < 0.1 and
                           abs(sv_nonzero[2] - math.sqrt(2)) < 0.1)
            print(f"  Next pair ({sv_nonzero[1]:.6f}, {sv_nonzero[2]:.6f}) matches (sqrt(2), sqrt(2)): {match_sqrt2}")

    else:
        print("  Cannot build chiral projectors: one or both of R, L sectors is empty.")
        print("  Proceeding with diagnostic computations...")
        M = None
        sv = np.array([])
        sv_nonzero = np.array([])

    # -----------------------------------------------------------------------
    # Step 7: Phase extraction from singular vectors
    # -----------------------------------------------------------------------
    print()
    print("Step 7 — Phase extraction from singular vectors of M")

    if M is not None and len(sv_nonzero) >= 1:
        # M = U_svd Sigma V^dag
        # The phases of the top singular vectors encode the Yukawa phases.
        # Specifically: for the dominant singular triplet, extract arg(U^dag B V)
        # diagonal.

        # Select top singular vectors (by singular value)
        top_n = min(3, len(sv_nonzero))
        U_top = U_svd[:, :top_n]    # left singular vectors
        V_top = Vh_svd[:top_n, :].conj().T   # right singular vectors

        # Overlap matrix between left and right singular subspaces via B(P)
        # This gives the "Yukawa matrix" in the chiral basis
        yukawa_matrix = U_top.conj().T @ B_P @ V_top   # top_n x top_n
        print(f"  Yukawa matrix (U^dag B(P) V) for top {top_n} singular vectors:")
        for row in yukawa_matrix:
            print(f"    " + "  ".join(f"{x.real:+.4f}{x.imag:+.4f}i" for x in row))

        # Extract diagonal elements (the relevant Yukawa amplitudes)
        yukawa_diag = np.diag(yukawa_matrix)
        print(f"  Diagonal Yukawa amplitudes:")
        for i, y in enumerate(yukawa_diag):
            arg_deg = math.degrees(np.angle(y))
            print(f"    y_{i} = {y.real:+.6f}{y.imag:+.6f}i   |y| = {abs(y):.6f}   arg = {arg_deg:+.6f} deg")

        # Phase comparison
        print(f"\n  Reference phases:")
        print(f"    delta_obs       = {DELTA_OBS:+.4f} deg  (PDG CKM convention)")
        print(f"    arg(h)/4        = {ARG_H_DEG / 4:+.4f} deg")
        print(f"    arg(h)          = {ARG_H_DEG:+.4f} deg")
        print(f"    arg(h)/2        = {ARG_H_DEG / 2:+.4f} deg")

        # Extract relative phase between first and second Yukawa elements
        if len(yukawa_diag) >= 2:
            rel_phase_12 = math.degrees(np.angle(yukawa_diag[1]) - np.angle(yukawa_diag[0]))
            print(f"  Relative phase arg(y_1) - arg(y_0) = {rel_phase_12:+.6f} deg")
            match_delta = abs(rel_phase_12 - DELTA_OBS) < 2.0
            match_argh4 = abs(rel_phase_12 - ARG_H_DEG / 4) < 2.0
            print(f"    Matches delta_obs ({DELTA_OBS:.2f} deg)? {match_delta}")
            print(f"    Matches arg(h)/4  ({ARG_H_DEG/4:.2f} deg)? {match_argh4}")

        # Also compute singular value phases (arg of singular values)
        # If M = U Sigma V^dag, the singular values are real non-negative.
        # The "phases" in the Yukawa sense come from the U and V vectors.
        # We compute the phase of each entry in the SVD product.
        print(f"\n  SVD-phase analysis (phases of U^dag B(P) V diagonal):")
        for i in range(top_n):
            u_i = U_top[:, i]
            v_i = V_top[:, i]
            yukawa_i = u_i.conj() @ B_P @ v_i
            print(f"    i={i}: u_i^dag B(P) v_i = {yukawa_i.real:+.6f}{yukawa_i.imag:+.6f}i   "
                  f"arg = {math.degrees(np.angle(yukawa_i)):+.4f} deg")

    else:
        print("  Cannot extract phases: M is None or no nonzero singular values.")

    # -----------------------------------------------------------------------
    # Step 8: C_3-isotypic content of P_R and P_L
    # -----------------------------------------------------------------------
    print()
    print("Step 8 — C_3-isotypic content of P_R and P_L sectors")

    U_C3 = build_c3_on_directed_edges(directed)

    if R_cols and L_cols:
        dims_R = c3_isotypic_dims_of_projector(P_R, U_C3, 12)
        dims_L = c3_isotypic_dims_of_projector(P_L, U_C3, 12)
        print(f"  C_3-isotypic dims (trivial, omega, omega^2) of P_R image: {dims_R}")
        print(f"  C_3-isotypic dims (trivial, omega, omega^2) of P_L image: {dims_L}")
    else:
        print("  Cannot compute: P_R or P_L not constructed.")

    # Also check how T_e relates to C_3 (do they commute?)
    comm_T_e_C3 = T_e @ U_C3 - U_C3 @ T_e
    norm_comm_T_e_C3 = la.norm(comm_T_e_C3)
    print(f"\n  ||[T_e, U_C3]|| = {norm_comm_T_e_C3:.6f}")
    if norm_comm_T_e_C3 < 1e-8:
        print(f"  T_e and U_C3 commute: T_e is C_3-equivariant.")
    else:
        print(f"  T_e and U_C3 do NOT commute: T_e mixes C_3 sectors.")

    # -----------------------------------------------------------------------
    # Step 9: Structural diagnosis
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("STRUCTURAL DIAGNOSIS")
    print("=" * 72)

    print()
    print(f"  [T_e, B(P)] nonzero:   max = {max_comm:.6f}   norm = {norm_comm:.6f}")
    print(f"  {{T_e, B(P)}} nonzero:   max = {max_anticomm:.6f}   norm = {norm_anticomm:.6f}")
    print()

    if max_comm > 1e-8:
        print("  KEY RESULT: [T_e, B(P)] != 0.")
        print("  T_e does not commute with B(P): B(P) does NOT preserve T_e-eigenspaces.")
        print("  This means B(P) implements CHIRAL MIXING between T_e = +1 and T_e = -1 sectors.")
    else:
        print("  BLOCKED: [T_e, B(P)] = 0. B(P) commutes with T_e, no chiral mixing via T_e.")

    if max_anticomm < 1e-8:
        print("  BONUS: {T_e, B(P)} = 0. T_e anti-commutes with B(P).")
        print("  B(P) is a chiral operator: it maps R -> L and L -> R purely.")
    else:
        print(f"  {{T_e, B(P)}} != 0 (max = {max_anticomm:.6f}): B(P) does not purely anti-commute with T_e.")

    if phase_conj_holds:
        print("  Phase-conjugate relation: T_e B(P) = B*(P) T_e holds.")
        print("  This is the 'time-reversal' property of the NB operator.")

    print()
    print(f"  T_e eigenvalues on V_Ram: (+1 mult {n_plus1}, -1 mult {n_minus1}, complex {n_complex})")

    if chiral_split and M is not None:
        sv_rounded = np.round(sv_nonzero, 6)
        print(f"  Singular values of M = P_R B(P) P_L: {sv_rounded}")
        koide_check = (len(sv_nonzero) >= 3 and
                       abs(sv_nonzero[0] - 2.0) < 0.1 and
                       abs(sv_nonzero[1] - math.sqrt(2)) < 0.1 and
                       abs(sv_nonzero[2] - math.sqrt(2)) < 0.1)
        if koide_check:
            print("  STRONG CONFIRMATION: singular values (2, sqrt(2), sqrt(2)) match Koide Yukawa magnitudes.")
        else:
            print(f"  Singular values do NOT match expected (2, sqrt(2), sqrt(2)) = ({2.0:.4f}, {math.sqrt(2):.4f}, {math.sqrt(2):.4f}).")
            if len(sv_nonzero) > 0:
                print(f"  BLOCKED: singular values = {sv_nonzero[:3]} != (2, sqrt(2), sqrt(2)).")
                print(f"  Exact gap: leading singular value = {sv_nonzero[0]:.6f}, expected 2.0.")

    print()
    print("RIGOR STATUS SUMMARY:")
    if max_comm > 1e-8 and chiral_split:
        if M is not None and len(sv_nonzero) >= 3 and abs(sv_nonzero[0] - 2.0) < 0.1:
            print("  strict-solid: [T_e, B(P)] != 0 (chiral mixing), V_Ram chiral-splits (4,4),")
            print("  and P_R B(P) P_L has leading singular value consistent with Koide reading rule.")
        else:
            print("  FESHBACH-PATTERN (partial): [T_e, B(P)] != 0 confirmed; chiral split exists;")
            print("  but singular values of P_R B(P) P_L do not match Koide (2, sqrt(2), sqrt(2)).")
            print("  Exact gap: need additional structure (e.g., sector restriction or phase projection)")
            print("  to recover the Koide magnitudes from the chiral mixing matrix.")
    elif max_comm < 1e-8:
        print("  BLOCKED: [T_e, B(P)] = 0. T_e commutes with B(P), no chiral mixing.")
        print("  Exact gap: T_e is a symmetry of B(P), not a chiral operator for it.")
    else:
        print("  BLOCKED: chiral split not established on V_Ram.")
        print("  Exact gap: T_e restricted to V_Ram has non-real eigenvalues or wrong multiplicities.")

    print()
    print("OK: t_e_chirality.py completed without assertion failures.")


if __name__ == "__main__":
    main()
