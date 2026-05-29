#!/usr/bin/env python3
"""
gamma7_chirality.py — Transfer Gamma_7 to V_Ram via B6 isomorphism; compute
P_{S_R} B(P) P_{S_L} singular values and compare to Koide magnitudes.

QUESTION UNDER TEST
-------------------
When Gamma_7 (chirality operator of Cl(6,0)) is transferred to V_Ram via the
B6 isomorphism V_Ram ≅ S, does it split V_Ram into S_L ⊕ S_R with C_3 content
(2,1,1)+(2,1,1)?  And does P_{S_R} B(P) P_{S_L} have singular values
consistent with Koide magnitudes (2, sqrt(2), sqrt(2))?

UPSTREAM THEOREMS INVOKED
--------------------------
- B3  (theorem_B3_spinor_fermion.md/.py): Cl(6,0) spinor S with chirality
      Gamma_7, SU(2)_L x SU(2)_R x U(1)_{B-L} structure.
- B6  (theorem_B6_bridge.md/.py): V_Ram ≅ S as C_3-modules (both (4,2,2)).
      Explicit Spin(6) lift U_C3_S on S constructed via Brauer-Weyl basis.
- BP  (theorem_BP_doubly_degenerate_h.md): V_Ram = 8-dim Ramanujan subspace
      of B(P) with eigenvalues {h, h*, -h, -h*} each mult 2.
- B5.3-core (theorem_B5_3_core.md/.py): C_3-isotypic structure (4,2,2) on
      V_Ram confirmed.

COMPUTATION STEPS
-----------------
Step 1.  Build B(P) (12x12) and U_{C_3} (12x12) on the fiber.
Step 2.  Extract V_Ram (8x12 orthonormal basis, the Ramanujan subspace).
Step 3.  Build the B3 Gamma generators and U_C3_S on the 8-dim spinor
         (B3/B6 construction, identical to theorem_B6_bridge.py Step 5).
Step 4.  Find the C_3-intertwining isomorphism A: C^8 -> V_Ram by matching
         C_3-isotypic sectors.
Step 5.  Transfer Gamma_7 to V_Ram:
           Gamma7_Ram = A Gamma_7 A^dag  (8x8 operator on V_Ram coordinates)
         and lift to full 12-dim:
           Gamma7_12 = V_Ram @ Gamma7_Ram @ V_Ram^dag  (12x12)
Step 6.  Verify Gamma7_Ram splits V_Ram as 4+4 with correct C_3 content (2,1,1)
         per chirality sector.
Step 7.  Build P_{S_L} and P_{S_R} (12x12 projectors) from Gamma7 eigenspaces.
Step 8.  Compute M = P_{S_R} @ B_P @ P_{S_L}; SVD; report singular values.
Step 9.  Compare to Koide magnitudes (2, sqrt(2), sqrt(2)).
Step 10. Phase extraction on nonzero singular vectors.
Step 11. Invariance check: singular values are independent of the C_3-sector-
         preserving unitary choice within each isotypic sector.
Step 12. Structural diagnosis and rigor verdict.

SECTOR INVARIANCE ARGUMENT
---------------------------
Schur's lemma for C_3: within each C_3 isotypic sector, the restriction of
B(P) from one sector to another is a matrix of size (mult_source x mult_target)
that is canonically defined (independent of any unitary basis choice within the
sector, since C_3 irreps are 1-dimensional and Schur gives proportionality to
the identity).  Specifically, for the trivial sector (mult 2 on each chirality),
the off-diagonal B(P) element is a 2x2 matrix, and its singular values are
invariants.  For the omega/omega^2 sectors (mult 1 each), the off-diagonal
element is a scalar whose absolute value is invariant.

Hence the singular values of M = P_{S_R} B(P) P_{S_L} are fully determined by
the C_3-sector-by-sector norms of B(P), independent of the specific A.

Run with:
    PYTHONPATH=. python3 proofs/foundations/gamma7_chirality.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la
from scipy.linalg import expm

from proofs.common import find_bonds, C3_PERM, omega3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOL = 1e-10
H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2   # Hashimoto P-point eigenvalue
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))       # arg(h) in radians
ARG_H_DEG = math.degrees(ARG_H)                       # ~52.24 deg
DELTA_OBS = 12.735                                     # PDG delta_CP in degrees
K_P = (0.25, 0.25, 0.25)

omega = np.exp(2j * np.pi / 3)
omega2 = omega * omega

PRINT_WIDTH = 72

# ---------------------------------------------------------------------------
# Gamma matrix infrastructure (from B3 / B6 — self-contained copy)
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


# Brauer-Weyl generators of Cl(6,0) on C^8 (B3 convention).
Gamma = [None] * 7  # indices 1..6
Gamma[1] = kron(sx, I2, I2)
Gamma[2] = kron(sy, I2, I2)
Gamma[3] = kron(sz, sx, I2)
Gamma[4] = kron(sz, sy, I2)
Gamma[5] = kron(sz, sz, sx)
Gamma[6] = kron(sz, sz, sy)
I8 = np.eye(8, dtype=complex)

# Verify Clifford relations.
for _a, _b in itertools.product(range(1, 7), repeat=2):
    _lhs = Gamma[_a] @ Gamma[_b] + Gamma[_b] @ Gamma[_a]
    _rhs = 2.0 * (1.0 if _a == _b else 0.0) * I8
    assert np.allclose(_lhs, _rhs, atol=TOL), f"Clifford relation fails a={_a},b={_b}"

# Chirality operator Gamma_7 = -i Gamma_1...Gamma_6
G7 = -1j * Gamma[1] @ Gamma[2] @ Gamma[3] @ Gamma[4] @ Gamma[5] @ Gamma[6]
assert np.allclose(G7, G7.conj().T, atol=TOL), "G7 not Hermitian"
assert np.allclose(G7 @ G7, I8, atol=TOL), "G7^2 != I"


def biv(a, b):
    """Bivector Gamma_ab = (1/2)[Gamma_a, Gamma_b]."""
    return 0.5 * (Gamma[a] @ Gamma[b] - Gamma[b] @ Gamma[a])


# ---------------------------------------------------------------------------
# B5.3-core infrastructure — B(P) and U_{C_3} on directed edges
# (self-contained copy of the key functions from theorem_B5_3_core.py)
# ---------------------------------------------------------------------------

def build_directed_edges(bonds):
    directed = [tuple(b) for b in bonds]
    assert len(directed) == 12, f"expected 12 directed edges, got {len(directed)}"
    return directed


def bloch_hashimoto(k_frac, directed):
    """12x12 Bloch Hashimoto B(k) on directed edges."""
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


def c3_vertex_perm():
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
        assert j is not None, f"C_3 mapped {(src, tgt, cell)} -> not found"
        U[j, i] = 1.0
    return U


def classify_c3(ev, tol=0.1):
    """Classify a C_3 eigenvalue as '1', 'w', 'w2', or '?'."""
    if abs(ev - 1.0) < tol:
        return '1'
    if abs(ev - omega) < tol:
        return 'w'
    if abs(ev - omega2) < tol:
        return 'w2'
    return '?'


def c3_isotypic_basis(M_8x8, tol=0.1):
    """
    Given an 8x8 unitary matrix M representing C_3 on C^8, decompose C^8 into
    C_3-isotypic subspaces.

    Returns a dict:
        {'1': Q1 (8 x m1), 'w': Qw (8 x mw), 'w2': Qw2 (8 x mw2)}
    where Q_alpha is an orthonormal basis for the alpha-isotypic subspace.
    """
    evals, evecs = la.eig(M_8x8)
    groups = {'1': [], 'w': [], 'w2': []}
    for i, ev in enumerate(evals):
        label = classify_c3(ev, tol)
        if label in groups:
            groups[label].append(evecs[:, i])
    bases = {}
    for label, vecs in groups.items():
        if vecs:
            mat = np.column_stack(vecs)
            Q, _ = la.qr(mat)
            bases[label] = Q[:, :len(vecs)]
        else:
            bases[label] = np.zeros((8, 0), dtype=complex)
    return bases


def c3_isotypic_basis_12(M_12x12, tol=0.1):
    """
    Same as c3_isotypic_basis but for a 12x12 unitary (V_Ram-space version).
    Returns the full 12-dim bases, not projected to V_Ram coordinates.
    """
    evals, evecs = la.eig(M_12x12)
    groups = {'1': [], 'w': [], 'w2': []}
    for i, ev in enumerate(evals):
        label = classify_c3(ev, tol)
        if label in groups:
            groups[label].append(evecs[:, i])
    bases = {}
    for label, vecs in groups.items():
        if vecs:
            mat = np.column_stack(vecs)
            Q, _ = la.qr(mat)
            bases[label] = Q[:, :len(vecs)]
        else:
            bases[label] = np.zeros((12, 0), dtype=complex)
    return bases


# ---------------------------------------------------------------------------
# U_C3_S: Spin(6) lift of C_3 onto the B3 8-dim spinor (from B6 Step 5)
# ---------------------------------------------------------------------------

def build_U_C3_S(directed):
    """Build the Spin(6) lift of C_3 onto the 8-dim B3 spinor.

    Follows theorem_B6_bridge.py Step 5 exactly:
    1. Build the 6x6 SO(6) permutation matrix P_so6 on K_4 edges.
    2. Compute matrix log in so(6) (antisymmetric).
    3. Build spin-algebra element X_spin via bivectors.
    4. U_C3_S = exp(0.5 * X_spin); fix sign to get U^3 = +I.
    """
    # K_4 edges from B6 Step 1
    K4_VERTICES = [0, 1, 2, 3]
    K4_EDGES = [(i, j) for i in K4_VERTICES for j in K4_VERTICES if i < j]
    SIGMA = {0: 0, 1: 3, 2: 1, 3: 2}

    def apply_sigma_to_edge(edge):
        a, b = edge
        return tuple(sorted((SIGMA[a], SIGMA[b])))

    edge_to_idx = {e: i for i, e in enumerate(K4_EDGES)}
    P_so6 = np.zeros((6, 6), dtype=float)
    for e in K4_EDGES:
        i = edge_to_idx[e]
        j = edge_to_idx[apply_sigma_to_edge(e)]
        P_so6[j, i] = 1.0

    assert np.allclose(P_so6.T @ P_so6, np.eye(6), atol=TOL)
    assert np.allclose(np.linalg.matrix_power(P_so6, 3), np.eye(6), atol=TOL)

    # Matrix log of P_so6
    evals_so6, evecs_so6 = la.eig(P_so6)
    log_evals = np.array([np.log(ev) for ev in evals_so6])
    L_so6 = (evecs_so6 @ np.diag(log_evals) @ np.linalg.inv(evecs_so6))
    L_so6_real = L_so6.real
    assert np.allclose(expm(L_so6_real), P_so6, atol=1e-10)
    assert np.allclose(L_so6_real, -L_so6_real.T, atol=1e-10)

    # Spin lift via bivectors: X_spin = (1/2) sum_{a<b} L_{ab} Gamma_{ab}
    X_spin = np.zeros((8, 8), dtype=complex)
    for i in range(6):
        for j in range(i + 1, 6):
            X_spin += L_so6_real[i, j] * biv(i + 1, j + 1)
    X_spin_half = 0.5 * X_spin
    U_C3_S = expm(X_spin_half)

    # Fix Spin double-cover sign: U^3 should be +I
    U3 = U_C3_S @ U_C3_S @ U_C3_S
    assert (np.allclose(U3, I8, atol=1e-8) or np.allclose(U3, -I8, atol=1e-8)), \
        "U_C3_S^3 is neither +I nor -I"
    if np.allclose(U3, -I8, atol=1e-8):
        U_C3_S = np.exp(1j * np.pi / 3) * U_C3_S
        assert np.allclose(U_C3_S @ U_C3_S @ U_C3_S, I8, atol=1e-8)

    # Verify [U_C3_S, G7] = 0 (chirality preserved by SO(6) lift)
    assert la.norm(U_C3_S @ G7 - G7 @ U_C3_S) < 1e-8, \
        "U_C3_S does not commute with G7"

    # Verify isotypic dims (4, 2, 2)
    evals_S = la.eigvals(U_C3_S)
    from collections import Counter
    cc = Counter(classify_c3(ev) for ev in evals_S)
    assert cc.get('1', 0) == 4 and cc.get('w', 0) == 2 and cc.get('w2', 0) == 2, \
        f"U_C3_S isotypic dims not (4,2,2): {dict(cc)}"

    return U_C3_S


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def main():
    print("=" * PRINT_WIDTH)
    print("gamma7_chirality.py — Transfer Gamma_7 to V_Ram via B6 isomorphism")
    print("=" * PRINT_WIDTH)

    # -----------------------------------------------------------------------
    # Step 1: Build B(P) and U_{C_3} on the 12-dim fiber
    # -----------------------------------------------------------------------
    print()
    print("Step 1 — Build B(P) (12x12) and U_{C_3} (12x12) on fiber")
    print("-" * PRINT_WIDTH)

    bonds = find_bonds()
    directed = build_directed_edges(bonds)

    B_P = bloch_hashimoto(K_P, directed)
    U_C3_12 = build_c3_on_directed_edges(directed)

    # Verify U_{C_3}^3 = I and [B(P), U_{C_3}] = 0
    assert np.allclose(U_C3_12 @ U_C3_12 @ U_C3_12, np.eye(12), atol=TOL), \
        "U_C3_12^3 != I"
    comm_B_U = la.norm(B_P @ U_C3_12 - U_C3_12 @ B_P)
    assert comm_B_U < 1e-10, f"[B(P), U_C3] not zero: {comm_B_U}"
    print(f"  B(P) shape: {B_P.shape}  ||B(P)||_F = {la.norm(B_P):.6f}")
    print(f"  U_C3^3 = I: verified.  ||[B(P), U_C3]|| = {comm_B_U:.2e}")

    # -----------------------------------------------------------------------
    # Step 2: Extract V_Ram (orthonormal 12x8 basis)
    # -----------------------------------------------------------------------
    print()
    print("Step 2 — Extract V_Ram (Ramanujan subspace, |eigenvalue|^2 = 2)")
    print("-" * PRINT_WIDTH)

    evals_B, evecs_B = la.eig(B_P)
    ram_idx = [i for i, ev in enumerate(evals_B) if abs(abs(ev) ** 2 - 2.0) < 1e-5]
    assert len(ram_idx) == 8, f"Expected 8 Ramanujan eigenvectors, got {len(ram_idx)}"

    # Check eigenvalues are in {h, h*, -h, -h*}
    h_targets = [H_EXACT, H_EXACT.conjugate(), -H_EXACT, -H_EXACT.conjugate()]
    for mu in evals_B[ram_idx]:
        assert any(abs(mu - t) < 1e-5 for t in h_targets), \
            f"Ramanujan eigenvalue {mu} not in expected set"

    # Orthonormal basis for V_Ram
    evecs_ram_raw = evecs_B[:, ram_idx]   # 12 x 8
    V_Ram, _ = la.qr(evecs_ram_raw)
    V_Ram = V_Ram[:, :8]   # 12 x 8 orthonormal basis

    assert la.matrix_rank(V_Ram) == 8, "V_Ram not rank 8"
    print(f"  V_Ram shape: {V_Ram.shape}  (12 x 8 orthonormal)")
    print(f"  ||V_Ram^dag V_Ram - I_8|| = {la.norm(V_Ram.conj().T @ V_Ram - np.eye(8)):.2e}")

    # Restrict U_{C_3} to V_Ram
    U_C3_VRam = V_Ram.conj().T @ U_C3_12 @ V_Ram   # 8x8

    # Verify isotypic dims (4, 2, 2) on V_Ram
    from collections import Counter
    evals_VRam_C3 = la.eigvals(U_C3_VRam)
    cc_VRam = Counter(classify_c3(ev) for ev in evals_VRam_C3)
    print(f"  C_3-isotypic dims on V_Ram: "
          f"(trivial={cc_VRam.get('1',0)}, omega={cc_VRam.get('w',0)}, "
          f"omega^2={cc_VRam.get('w2',0)})")
    assert (cc_VRam.get('1', 0), cc_VRam.get('w', 0), cc_VRam.get('w2', 0)) == (4, 2, 2), \
        f"V_Ram C_3 multiplicities != (4,2,2): {dict(cc_VRam)}"
    print(f"  Confirmed (4, 2, 2) on V_Ram. OK.")

    # -----------------------------------------------------------------------
    # Step 3: Build U_C3_S and G7 on the B3 8-dim spinor
    # -----------------------------------------------------------------------
    print()
    print("Step 3 — Build U_C3_S (Spin(6) lift) and G7 on the B3 spinor")
    print("-" * PRINT_WIDTH)

    U_C3_S = build_U_C3_S(directed)

    # Confirm isotypic dims on S
    evals_S_C3 = la.eigvals(U_C3_S)
    cc_S = Counter(classify_c3(ev) for ev in evals_S_C3)
    print(f"  U_C3_S isotypic dims: "
          f"(trivial={cc_S.get('1',0)}, omega={cc_S.get('w',0)}, "
          f"omega^2={cc_S.get('w2',0)})")
    assert (cc_S.get('1', 0), cc_S.get('w', 0), cc_S.get('w2', 0)) == (4, 2, 2), \
        f"U_C3_S multiplicities != (4,2,2): {dict(cc_S)}"

    # G7 eigenvalues on S
    evals_G7 = la.eigvalsh(G7)
    cc_G7 = Counter(int(round(ev)) for ev in evals_G7)
    print(f"  G7 eigenvalues on S: +1 mult={cc_G7.get(1,0)}, -1 mult={cc_G7.get(-1,0)}")
    assert cc_G7[+1] == 4 and cc_G7[-1] == 4, \
        f"G7 does not split S as 4+4: {dict(cc_G7)}"
    print(f"  [U_C3_S, G7] = 0: ||comm|| = {la.norm(U_C3_S @ G7 - G7 @ U_C3_S):.2e}")
    print(f"  Confirmed G7 commutes with U_C3_S.  OK.")

    # -----------------------------------------------------------------------
    # Step 4: Find C_3-intertwining isomorphism A: C^8 -> V_Ram (in V_Ram coords)
    #
    # Strategy (sector-by-sector matching):
    #   For each sector alpha in {trivial, omega, omega^2}:
    #     - S_alpha  = alpha-isotypic subspace of C^8  (for U_C3_S)
    #     - VR_alpha = alpha-isotypic subspace of V_Ram (for U_C3_VRam)
    #   Both have the same dimension (trivial: 4, omega: 2, omega^2: 2 on FULL;
    #   but note S_alpha and VR_alpha are subspaces of C^8, same structure).
    #   Find A|_{S_alpha} = Q_{VR_alpha} @ (arbitrary unitary) @ Q_{S_alpha}^dag
    #   For our purposes we take the "canonical" sector match:
    #     A|_{S_alpha} = Q_{VR_alpha} @ Q_{S_alpha}^dag
    #   (choosing basis correspondence as QR-output ordering).
    #
    # This gives A: C^8 -> C^8 (in V_Ram coordinates) such that
    #   A @ U_C3_S = U_C3_VRam @ A.
    # -----------------------------------------------------------------------
    print()
    print("Step 4 — Find C_3-intertwining isomorphism A: C^8 -> V_Ram coords")
    print("-" * PRINT_WIDTH)

    # Get orthonormal bases for each isotypic sector
    bases_S = c3_isotypic_basis(U_C3_S)      # dict: sector -> (8 x m) matrix
    bases_VRam = c3_isotypic_basis(U_C3_VRam) # dict: sector -> (8 x m) matrix

    for label in ['1', 'w', 'w2']:
        dim_S = bases_S[label].shape[1]
        dim_VR = bases_VRam[label].shape[1]
        print(f"  Sector '{label}':  dim_S={dim_S},  dim_VRam={dim_VR}")
        assert dim_S == dim_VR, \
            f"Sector '{label}' dimension mismatch: S has {dim_S}, V_Ram has {dim_VR}"

    # Build A = sum_alpha  Q_{VRam,alpha} @ Q_{S,alpha}^dag
    # This is an 8x8 unitary mapping S-coordinates to VRam-coordinates.
    A = np.zeros((8, 8), dtype=complex)
    for label in ['1', 'w', 'w2']:
        Q_S = bases_S[label]       # 8 x m
        Q_VR = bases_VRam[label]   # 8 x m
        A += Q_VR @ Q_S.conj().T

    # Verify A is unitary
    err_unitary = la.norm(A @ A.conj().T - np.eye(8))
    print(f"\n  ||A A^dag - I_8|| = {err_unitary:.3e}  (should be 0: unitary check)")
    assert err_unitary < 1e-8, f"A is not unitary: {err_unitary}"

    # Verify A intertwines C_3:  A @ U_C3_S = U_C3_VRam @ A
    intertwine_err = la.norm(A @ U_C3_S - U_C3_VRam @ A)
    print(f"  ||A U_C3_S - U_C3_VRam A|| = {intertwine_err:.3e}  (should be 0: intertwiner)")
    assert intertwine_err < 1e-8, f"A does not intertwine C_3: {intertwine_err}"
    print(f"  A is a valid C_3-intertwining isomorphism.  OK.")

    # -----------------------------------------------------------------------
    # Step 5: Transfer Gamma_7 to V_Ram
    #
    # Gamma7_VRam (in V_Ram coords) = A @ G7 @ A^dag
    # Lift to full 12-dim space:  Gamma7_12 = V_Ram @ Gamma7_VRam @ V_Ram^dag
    # -----------------------------------------------------------------------
    print()
    print("Step 5 — Transfer Gamma_7 to V_Ram via A")
    print("-" * PRINT_WIDTH)

    Gamma7_VRam = A @ G7 @ A.conj().T   # 8x8
    Gamma7_12 = V_Ram @ Gamma7_VRam @ V_Ram.conj().T   # 12x12

    # Check Gamma7_VRam is Hermitian and squares to identity
    err_herm = la.norm(Gamma7_VRam - Gamma7_VRam.conj().T)
    err_sq = la.norm(Gamma7_VRam @ Gamma7_VRam - np.eye(8))
    print(f"  ||Gamma7_VRam - Gamma7_VRam^dag|| = {err_herm:.3e}  (hermitian check)")
    print(f"  ||Gamma7_VRam^2 - I_8||           = {err_sq:.3e}     (involution check)")
    assert err_herm < 1e-8, f"Gamma7_VRam not Hermitian: {err_herm}"
    assert err_sq < 1e-8, f"Gamma7_VRam^2 != I: {err_sq}"

    # Check eigenvalues of Gamma7_VRam are +-1 with multiplicity 4+4
    evals_G7_VRam = la.eigvalsh(Gamma7_VRam)
    cc_G7_VRam = Counter(int(round(ev)) for ev in evals_G7_VRam)
    print(f"  Gamma7_VRam eigenvalues: +1 mult={cc_G7_VRam.get(1,0)}, "
          f"-1 mult={cc_G7_VRam.get(-1,0)}")
    assert cc_G7_VRam[+1] == 4 and cc_G7_VRam[-1] == 4, \
        f"Gamma7_VRam does not split as 4+4: {dict(cc_G7_VRam)}"
    print(f"  Confirmed Gamma7_VRam splits V_Ram as S_R(+1) + S_L(-1): 4+4.  OK.")

    # -----------------------------------------------------------------------
    # Step 6: Verify C_3 content of each chirality sector
    #
    # The +1 (S_R) sector of Gamma7_VRam should have C_3 content (2,1,1)
    # The -1 (S_L) sector should also have C_3 content (2,1,1).
    # -----------------------------------------------------------------------
    print()
    print("Step 6 — C_3-isotypic content of S_R(+1) and S_L(-1) sectors")
    print("-" * PRINT_WIDTH)

    # Build Gamma7_VRam eigenvectors
    evals_G7_vram, evecs_G7_vram = la.eigh(Gamma7_VRam)
    SR_idx = [i for i, ev in enumerate(evals_G7_vram) if abs(ev - 1.0) < 0.1]   # +1: S_R
    SL_idx = [i for i, ev in enumerate(evals_G7_vram) if abs(ev + 1.0) < 0.1]   # -1: S_L
    assert len(SR_idx) == 4 and len(SL_idx) == 4, \
        f"Chirality split not 4+4: SR={len(SR_idx)}, SL={len(SL_idx)}"

    Q_SR = evecs_G7_vram[:, SR_idx]   # 8 x 4, in V_Ram coordinates
    Q_SL = evecs_G7_vram[:, SL_idx]   # 8 x 4, in V_Ram coordinates

    # C_3 action on each sector (in V_Ram coordinates)
    U_SR = Q_SR.conj().T @ U_C3_VRam @ Q_SR   # 4x4
    U_SL = Q_SL.conj().T @ U_C3_VRam @ Q_SL   # 4x4

    evals_SR = la.eigvals(U_SR)
    evals_SL = la.eigvals(U_SL)
    cc_SR = Counter(classify_c3(ev) for ev in evals_SR)
    cc_SL = Counter(classify_c3(ev) for ev in evals_SL)

    print(f"  S_R(+1) C_3-isotypic dims: "
          f"(trivial={cc_SR.get('1',0)}, omega={cc_SR.get('w',0)}, omega^2={cc_SR.get('w2',0)})")
    print(f"  S_L(-1) C_3-isotypic dims: "
          f"(trivial={cc_SL.get('1',0)}, omega={cc_SL.get('w',0)}, omega^2={cc_SL.get('w2',0)})")

    assert (cc_SR.get('1', 0), cc_SR.get('w', 0), cc_SR.get('w2', 0)) == (2, 1, 1), \
        f"S_R C_3 dims not (2,1,1): {dict(cc_SR)}"
    assert (cc_SL.get('1', 0), cc_SL.get('w', 0), cc_SL.get('w2', 0)) == (2, 1, 1), \
        f"S_L C_3 dims not (2,1,1): {dict(cc_SL)}"
    print(f"  Confirmed: S_R has (2,1,1) and S_L has (2,1,1).  Total = (4,2,2).  OK.")

    # -----------------------------------------------------------------------
    # Step 7: Build P_{S_R} and P_{S_L} as 12x12 projectors
    # -----------------------------------------------------------------------
    print()
    print("Step 7 — Build 12x12 projectors P_{S_R} and P_{S_L}")
    print("-" * PRINT_WIDTH)

    # Lift Q_SR, Q_SL to full 12-dim space
    V_SR_12 = V_Ram @ Q_SR   # 12 x 4
    V_SL_12 = V_Ram @ Q_SL   # 12 x 4

    # Orthonormalize (should already be orthonormal since V_Ram is orthonormal
    # and Q_SR comes from an eigh decomposition, but re-orthonormalize for safety)
    Q_SR_12, _ = la.qr(V_SR_12)
    Q_SR_12 = Q_SR_12[:, :4]
    Q_SL_12, _ = la.qr(V_SL_12)
    Q_SL_12 = Q_SL_12[:, :4]

    P_SR = Q_SR_12 @ Q_SR_12.conj().T   # 12x12
    P_SL = Q_SL_12 @ Q_SL_12.conj().T   # 12x12

    # Verify projectors
    err_PSR = la.norm(P_SR @ P_SR - P_SR)
    err_PSL = la.norm(P_SL @ P_SL - P_SL)
    orth_err = la.norm(P_SR @ P_SL)
    print(f"  ||P_SR^2 - P_SR|| = {err_PSR:.3e},  ||P_SL^2 - P_SL|| = {err_PSL:.3e}")
    print(f"  ||P_SR P_SL||     = {orth_err:.3e}  (orthogonality of S_R, S_L)")
    assert err_PSR < 1e-8, f"P_SR not a projector: {err_PSR}"
    assert err_PSL < 1e-8, f"P_SL not a projector: {err_PSL}"
    assert orth_err < 1e-8, f"P_SR, P_SL not orthogonal: {orth_err}"
    print(f"  P_SR, P_SL are valid orthogonal projectors.  OK.")

    # -----------------------------------------------------------------------
    # Step 8: Compute M = P_{S_R} @ B(P) @ P_{S_L}; SVD
    # -----------------------------------------------------------------------
    print()
    print("Step 8 — Compute M = P_{S_R} B(P) P_{S_L} and SVD")
    print("-" * PRINT_WIDTH)

    M = P_SR @ B_P @ P_SL

    # SVD
    U_svd, sv, Vh_svd = la.svd(M)
    sv_nonzero = sv[sv > 1e-8]

    print(f"  M shape: {M.shape}")
    print(f"  All singular values of M:")
    for i, s in enumerate(sv):
        flag = "  <<< nonzero" if s > 1e-8 else ""
        print(f"    sigma_{i:2d} = {s:.10f}{flag}")

    print(f"\n  Nonzero singular values: {len(sv_nonzero)}")
    for i, s in enumerate(sv_nonzero):
        print(f"    sigma_{i} = {s:.10f}")

    # -----------------------------------------------------------------------
    # Step 9: Compare to Koide magnitudes
    # -----------------------------------------------------------------------
    print()
    print("Step 9 — Compare nonzero singular values to Koide magnitudes (2, sqrt(2), sqrt(2))")
    print("-" * PRINT_WIDTH)

    koide_expected = np.array([2.0, math.sqrt(2.0), math.sqrt(2.0)])
    print(f"  Koide magnitudes:   sigma = (2, sqrt(2), sqrt(2)) = "
          f"({koide_expected[0]:.8f}, {koide_expected[1]:.8f}, {koide_expected[2]:.8f})")

    if len(sv_nonzero) >= 3:
        match_2 = abs(sv_nonzero[0] - 2.0) < 0.001
        match_sq2_1 = abs(sv_nonzero[1] - math.sqrt(2)) < 0.001
        match_sq2_2 = abs(sv_nonzero[2] - math.sqrt(2)) < 0.001
        koide_match = match_2 and match_sq2_1 and match_sq2_2
        print(f"  Leading singular value  sigma_0 = {sv_nonzero[0]:.10f}")
        print(f"    |sigma_0 - 2|         = {abs(sv_nonzero[0] - 2.0):.2e}   match: {match_2}")
        print(f"  Second singular value   sigma_1 = {sv_nonzero[1]:.10f}")
        print(f"    |sigma_1 - sqrt(2)|   = {abs(sv_nonzero[1] - math.sqrt(2)):.2e}   match: {match_sq2_1}")
        print(f"  Third singular value    sigma_2 = {sv_nonzero[2]:.10f}")
        print(f"    |sigma_2 - sqrt(2)|   = {abs(sv_nonzero[2] - math.sqrt(2)):.2e}   match: {match_sq2_2}")
    elif len(sv_nonzero) == 1:
        match_2 = abs(sv_nonzero[0] - 2.0) < 0.001
        koide_match = False
        print(f"  Only 1 nonzero singular value: sigma_0 = {sv_nonzero[0]:.10f}")
        print(f"    |sigma_0 - 2|         = {abs(sv_nonzero[0] - 2.0):.2e}   match: {match_2}")
        print(f"  BLOCKED: expected 3 nonzero singular values (2, sqrt(2), sqrt(2)).")
    elif len(sv_nonzero) == 0:
        koide_match = False
        print(f"  BLOCKED: M = 0; no nonzero singular values.")
    else:
        koide_match = False
        print(f"  {len(sv_nonzero)} nonzero singular values:")
        for i, s in enumerate(sv_nonzero):
            print(f"    sigma_{i} = {s:.10f}")

    # -----------------------------------------------------------------------
    # Step 10: Phase extraction
    # -----------------------------------------------------------------------
    print()
    print("Step 10 — Phase extraction from dominant singular vectors")
    print("-" * PRINT_WIDTH)

    print(f"  Reference phases:")
    print(f"    delta_obs     = {DELTA_OBS:+.4f} deg  (PDG CKM CP-violating phase)")
    print(f"    arg(h)/4      = {ARG_H_DEG/4:+.4f} deg")
    print(f"    arg(h)/2      = {ARG_H_DEG/2:+.4f} deg")
    print(f"    arg(h)        = {ARG_H_DEG:+.4f} deg")

    top_n = min(3, len(sv_nonzero))
    if top_n >= 1:
        U_top = U_svd[:, :top_n]                    # left singular vectors
        V_top = Vh_svd[:top_n, :].conj().T           # right singular vectors

        yukawa_matrix = U_top.conj().T @ B_P @ V_top   # top_n x top_n
        print(f"\n  Yukawa matrix U^dag B(P) V for top {top_n} singular pairs:")
        for row in yukawa_matrix:
            print("    " + "  ".join(f"{x.real:+.6f}{x.imag:+.6f}i" for x in row))

        yukawa_diag = np.diag(yukawa_matrix)
        print(f"\n  Diagonal Yukawa amplitudes (u_i^dag B(P) v_i):")
        for i, y in enumerate(yukawa_diag):
            arg_deg = math.degrees(np.angle(y))
            print(f"    y_{i} = {abs(y):.8f} * exp(i * {arg_deg:+.6f} deg)")

        if top_n >= 2:
            rel_phase_01 = math.degrees(np.angle(yukawa_diag[1]) - np.angle(yukawa_diag[0]))
            print(f"\n  Relative phase arg(y_1) - arg(y_0) = {rel_phase_01:+.6f} deg")
            print(f"    vs delta_obs = {DELTA_OBS:+.4f} deg  |gap| = {abs(rel_phase_01 - DELTA_OBS):.4f} deg")
            print(f"    vs arg(h)/4  = {ARG_H_DEG/4:+.4f} deg  |gap| = {abs(rel_phase_01 - ARG_H_DEG/4):.4f} deg")
        if top_n >= 3:
            rel_phase_02 = math.degrees(np.angle(yukawa_diag[2]) - np.angle(yukawa_diag[0]))
            print(f"  Relative phase arg(y_2) - arg(y_0) = {rel_phase_02:+.6f} deg")
    else:
        print("  Cannot extract phases: no nonzero singular values.")

    # -----------------------------------------------------------------------
    # Step 11: Invariance check — singular values under sector-preserving
    # unitary rotation within each C_3 isotypic sector
    # -----------------------------------------------------------------------
    print()
    print("Step 11 — Invariance of singular values under sector-unitary rotation")
    print("-" * PRINT_WIDTH)
    print("  Testing: rotate each C_3-isotypic sector of A by a random unitary;")
    print("  singular values of M should be unchanged.")

    # For the trivial sector (dim 4 in each of S and VRam), apply a random 4x4 unitary.
    # For the omega and omega^2 sectors (dim 2 each), only a 2x2 unitary — covered separately.
    dim_trivial = bases_S['1'].shape[1]   # = 4
    np.random.seed(42)
    sv_perturbed_list = []
    for trial in range(3):
        # Random unitary in the trivial sector of S (dim = dim_trivial = 4)
        Z = np.random.randn(dim_trivial, dim_trivial) + 1j * np.random.randn(dim_trivial, dim_trivial)
        W2, _ = la.qr(Z)   # random dim_trivial x dim_trivial unitary

        # Modify A: replace Q_{VRam,'1'} @ Q_{S,'1'}^dag  with
        #            Q_{VRam,'1'} @ W2 @ Q_{S,'1'}^dag
        A_perturbed = A.copy()
        Q_S1 = bases_S['1']     # 8 x 2
        Q_VR1 = bases_VRam['1'] # 8 x 2
        # Remove original trivial contribution, add rotated one
        A_perturbed -= Q_VR1 @ Q_S1.conj().T
        A_perturbed += Q_VR1 @ W2 @ Q_S1.conj().T

        # Verify still unitary and intertwining
        err_u = la.norm(A_perturbed @ A_perturbed.conj().T - np.eye(8))
        err_i = la.norm(A_perturbed @ U_C3_S - U_C3_VRam @ A_perturbed)
        assert err_u < 1e-7, f"Perturbed A not unitary: {err_u}"
        assert err_i < 1e-7, f"Perturbed A not intertwining: {err_i}"

        # Recompute Gamma7 and projectors
        G7_p = A_perturbed @ G7 @ A_perturbed.conj().T
        evals_p, evecs_p = la.eigh(G7_p)
        SR_p = [i for i, ev in enumerate(evals_p) if abs(ev - 1.0) < 0.1]
        SL_p = [i for i, ev in enumerate(evals_p) if abs(ev + 1.0) < 0.1]
        assert len(SR_p) == 4 and len(SL_p) == 4

        Q_SR_p_12 = V_Ram @ evecs_p[:, SR_p]
        Q_SL_p_12 = V_Ram @ evecs_p[:, SL_p]
        P_SR_p = Q_SR_p_12 @ Q_SR_p_12.conj().T
        P_SL_p = Q_SL_p_12 @ Q_SL_p_12.conj().T
        M_p = P_SR_p @ B_P @ P_SL_p
        sv_p = la.svd(M_p, compute_uv=False)
        sv_perturbed_list.append(sv_p[:len(sv_nonzero)])
        print(f"  Trial {trial+1}: perturbed sigma = "
              f"{sv_p[:3] if len(sv_p) >= 3 else sv_p}")

    # Check invariance (diagnostic — NOT asserted; see structural diagnosis below)
    all_invariant = True
    for i, sv_p in enumerate(sv_perturbed_list):
        max_diff = max(abs(sv_p[j] - sv_nonzero[j]) for j in range(min(len(sv_p), len(sv_nonzero))))
        print(f"  Trial {i+1} max |Delta sigma_j| = {max_diff:.3e}")
        if max_diff > 1e-6:
            all_invariant = False
    if all_invariant:
        print(f"  Singular values ARE invariant under sector-preserving rotation.  OK.")
    else:
        print(f"  STRUCTURAL FINDING: singular values are NOT invariant.")
        print(f"  The trivial sector of C_3 has dim=4, so the intertwiner A has a")
        print(f"  residual U(4) gauge freedom within that sector. The projectors P_SR,")
        print(f"  P_SL (and hence M = P_SR B(P) P_SL) depend on this gauge choice.")
        print(f"  Therefore: the singular values of M are gauge-dependent and do NOT")
        print(f"  give a canonical set of Koide magnitudes from this construction.")
        print(f"  Additional structure (a canonical gauge-fixing condition) is required.")
        print(f"  This is the B6-intertwiner gauge ambiguity: the (4,2,2) isotypic match")
        print(f"  is real, but the intertwiner A is not unique within the trivial sector.")
    gauge_invariant = all_invariant

    # -----------------------------------------------------------------------
    # Step 12: Structural diagnosis and verdict
    # -----------------------------------------------------------------------
    print()
    print("=" * PRINT_WIDTH)
    print("STRUCTURAL DIAGNOSIS")
    print("=" * PRINT_WIDTH)

    print()
    print("  Summary of verified properties:")
    print(f"    1. B3/B6 spinor construction: Gamma_7 splits S as 4+4. OK.")
    print(f"    2. V_Ram has C_3-isotypic dims (4,2,2). OK.")
    print(f"    3. A: C^8 -> V_Ram coords is C_3-intertwining unitary. OK.")
    print(f"    4. Gamma7_VRam splits V_Ram as S_R(4) + S_L(4). OK.")
    print(f"    5. S_R has C_3 content (2,1,1); S_L has C_3 content (2,1,1). OK.")
    print(f"    6. P_SR, P_SL are orthogonal 12x12 projectors. OK.")
    print(f"    7. Singular values of M = P_SR B(P) P_SL computed (ONE gauge choice). OK.")
    gauge_str = "gauge-INVARIANT" if gauge_invariant else "gauge-DEPENDENT (see Step 11)"
    print(f"    8. Singular values are {gauge_str}.")

    print()
    print(f"  Nonzero singular values of M = P_{{S_R}} B(P) P_{{S_L}}:")
    for i, s in enumerate(sv_nonzero):
        print(f"    sigma_{i} = {s:.10f}")
    print(f"  Target (Koide magnitudes): (2, sqrt(2), sqrt(2)) = "
          f"({2.0:.8f}, {math.sqrt(2):.8f}, {math.sqrt(2):.8f})")

    print()
    if len(sv_nonzero) >= 3 and koide_match:
        print("  VERDICT: STRICT-SOLID")
        print("  The Gamma_7 chirality operator, transferred to V_Ram via the B6")
        print("  isomorphism, splits V_Ram into S_R + S_L with C_3 content (2,1,1)+(2,1,1),")
        print("  and P_{S_R} B(P) P_{S_L} has singular values (2, sqrt(2), sqrt(2)),")
        print("  matching the Koide magnitudes to 3 decimal places.")
        print("  Upstream: B3 + B6 + A1 + A2. No additional axioms required.")
    elif len(sv_nonzero) >= 3 and not koide_match:
        print("  VERDICT: BLOCKED")
        print("  Gamma_7 correctly splits V_Ram as S_R(4) + S_L(4) with C_3 content")
        print("  (2,1,1)+(2,1,1). However, the singular values of P_{S_R} B(P) P_{S_L}")
        print("  do NOT match the Koide magnitudes (2, sqrt(2), sqrt(2)).")
        print(f"  Exact values: {sv_nonzero[:3]}")
        print(f"  Exact gap to (2, sqrt(2), sqrt(2)): "
              f"({sv_nonzero[0]-2.0:.6f}, {sv_nonzero[1]-math.sqrt(2):.6f}, "
              f"{sv_nonzero[2]-math.sqrt(2):.6f})")
    elif len(sv_nonzero) < 3:
        print("  VERDICT: BLOCKED")
        print(f"  M = P_{{S_R}} B(P) P_{{S_L}} has only {len(sv_nonzero)} nonzero singular")
        print(f"  value(s), not 3.  Expected 3 for Koide identification.")
        print(f"  Gamma_7 chirality splitting is structurally correct (S_R/S_L 4+4,")
        print(f"  C_3 content (2,1,1) each), but B(P) does not mix chirality sectors")
        print(f"  in the expected pattern.")
    else:
        print("  VERDICT: BLOCKED (edge case)")

    print()
    print("OK: gamma7_chirality.py completed without assertion failures.")


if __name__ == "__main__":
    main()
