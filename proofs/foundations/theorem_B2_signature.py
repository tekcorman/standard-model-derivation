# ============================================================
# THEOREM: B2 — Signature of the quadratic form Q on the 6-edge space of K_4
# ============================================================
#
# Audit anchor: Row 22 of `docs/audits/registers/uniqueness_ledger.md` (Cl(2) pseudoscalar
# orientation UNIQUE under A2-T waterline). Conditional on Rows 4, 6
# (k* = 3 + srs identification).

# --- THEOREM STATEMENT ---------------------------------------
# Among three natural graph-theoretic quadratic forms on the 6-dimensional
# undirected-edge space of the K_4 quotient of the srs primitive cell, only
# the P-point Ramanujan projector form (candidate 3) is non-degenerate on
# all of R^6, and its signature is (6, 0) — Euclidean Cl(6, 0).
# Status: STRICT-SOLID (theorem-grade, CAS+numerical verification)

# --- FRAMEWORK AXIOMS INVOKED --------------------------------
# A1 (self-inverse toggle): enters via the srs NB-walker → Hashimoto → Bloch
#    chain that defines the P-point Ramanujan projector.
# A2 (MDL): selects the invariant Clifford formulation (upstream B1.b).
# Non-degeneracy criterion: Clifford algebras require a non-degenerate form
#    (Lawson & Michelsohn 1989 Ch. I §1); this selects candidate 3.

# --- INPUTS --------------------------------------------------
# K_4 graph: 4-vertex complete graph, oriented incidence matrix B_inc.
# Hashimoto NB-walk operator B on 12 directed edges of K_4.
# B(P): 12x12 Bloch Hashimoto at k = P = (1/4, 1/4, 1/4) for the srs
#       primitive cell (theorem_BP_doubly_degenerate_h).
# Three candidate forms:
#   1. Edge Laplacian   Q_1 = B_inc^T B_inc   on R^6
#   2. Hashimoto sym    Q_2 = S^T ((B+B^T)/2) S  on R^6
#   3. P-point proj     Q_3 = S^T P_R S          on C^6  (Hermitian)

# --- IMPLEMENTATION ------------------------------------------
# Import and re-run the candidate computations from the proof script.

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repo root (file now under proofs/foundations/)
sys.path.insert(0, str(REPO))

import numpy as np
import sympy as sp

TOL = 1e-10

# K_4 combinatorics — vertex / edge counts sourced from leaf primitives
sys.path.insert(0, str(REPO / "predictions"))
from V_count import V_count_pred as N_V   # = 4
from E_count import E_count_pred as N_E   # = 6 via handshake 2|E| = k·|V|

EDGES = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
]

B_inc = np.zeros((N_V, N_E), dtype=float)
for _k, (_u, _v) in enumerate(EDGES):
    B_inc[_u, _k] = -1.0
    B_inc[_v, _k] = +1.0

DIR_EDGES: list[tuple[int, int]] = []
for _u, _v in EDGES:
    DIR_EDGES.append((_u, _v))
    DIR_EDGES.append((_v, _u))
assert len(DIR_EDGES) == 12


def _signature(matrix: np.ndarray, tol: float = TOL) -> tuple[int, int, int]:
    w = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2)
    p = int(np.sum(w > tol))
    q = int(np.sum(w < -tol))
    z = int(np.sum(np.abs(w) <= tol))
    return p, q, z


def _hashimoto_k4() -> np.ndarray:
    B = np.zeros((12, 12), dtype=float)
    for i, (a, b) in enumerate(DIR_EDGES):
        for j, (c, d) in enumerate(DIR_EDGES):
            if b == c and (c, d) != (b, a):
                B[i, j] = 1.0
    assert np.allclose(B.sum(axis=1), 2.0 * np.ones(12))
    return B


def _symmetrizer_k4() -> np.ndarray:
    S = np.zeros((12, 6), dtype=float)
    for j, (u, v) in enumerate(EDGES):
        S[DIR_EDGES.index((u, v)), j] = 1.0 / np.sqrt(2.0)
        S[DIR_EDGES.index((v, u)), j] = 1.0 / np.sqrt(2.0)
    assert np.allclose(S.T @ S, np.eye(6), atol=TOL)
    return S


def _build_B_at_P():
    """Build the 12x12 Bloch Hashimoto B(P) for the srs primitive cell
    at k = P = (1/4, 1/4, 1/4) using sympy for exact phase construction."""
    cell_edges = [
        (0, 1, (1, 1, 1)), (0, 2, (1, 1, 1)), (0, 3, (1, 1, 1)),
        (1, 2, (-1, 0, 0)), (1, 3, (0, 1, 0)), (2, 3, (0, 0, -1)),
    ]
    P_vec = (sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 4))
    dir_list = []
    dir_phase = []
    for (u, v, c) in cell_edges:
        phase = sp.exp(sp.I * 2 * sp.pi * (
            c[0] * P_vec[0] + c[1] * P_vec[1] + c[2] * P_vec[2]
        ))
        dir_list.append((u, v))
        dir_phase.append(phase)
        dir_list.append((v, u))
        dir_phase.append(1 / phase)
    assert len(dir_list) == 12

    BP = sp.zeros(12, 12)
    for i, (e_tail, e_head) in enumerate(dir_list):
        for j, (f_tail, f_head) in enumerate(dir_list):
            if f_tail == e_head and (f_head, f_tail) != (e_tail, e_head):
                BP[j, i] = dir_phase[j]
    return np.array(BP.evalf(), dtype=complex), dir_list


def _candidate_3_p_projector():
    BP_np, dir_list = _build_B_at_P()
    # Verify spectrum
    h = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
    eigs = np.linalg.eigvals(BP_np)
    expected = [h, h, np.conj(h), np.conj(h), -h, -h,
                -np.conj(h), -np.conj(h), 1+0j, 1+0j, -1+0j, -1+0j]
    exp_s = sorted(expected, key=lambda z: (round(z.real, 4), round(z.imag, 4)))
    got_s = sorted(eigs, key=lambda z: (round(z.real, 4), round(z.imag, 4)))
    assert max(abs(a - b) for a, b in zip(exp_s, got_s)) < 1e-9

    # Ramanujan projector
    w, V = np.linalg.eig(BP_np)
    ram_mask = np.array([abs(abs(wi) - np.sqrt(2)) < 1e-7 for wi in w])
    assert ram_mask.sum() == 8
    Q_orth, _ = np.linalg.qr(V[:, ram_mask])
    P_R = Q_orth @ Q_orth.conj().T
    assert np.allclose(P_R, P_R.conj().T, atol=1e-9)
    assert np.allclose(P_R @ P_R, P_R, atol=1e-8)
    assert abs(P_R.trace() - 8) < 1e-8

    # Symmetrizer for the cell directed-edge ordering
    cell_undirected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    S_cell = np.zeros((12, 6), dtype=complex)
    for j, (a, b) in enumerate(cell_undirected):
        S_cell[dir_list.index((a, b)), j] = 1.0 / np.sqrt(2.0)
        S_cell[dir_list.index((b, a)), j] = 1.0 / np.sqrt(2.0)
    assert np.allclose(S_cell.conj().T @ S_cell, np.eye(6), atol=TOL)

    Q3 = S_cell.conj().T @ P_R @ S_cell
    Q3 = (Q3 + Q3.conj().T) / 2
    return Q3


# --- PURE FUNCTION -------------------------------------------
def verify_theorem_B2_signature():
    """Verify Theorem B2: among three candidate quadratic forms on the
    6-edge space of K_4, only the P-point Ramanujan projector (candidate 3)
    is non-degenerate, with signature (6, 0).

    Returns a dict with keys:
      'candidate_1_sig': (p, q, z)  -- Edge Laplacian
      'candidate_2_sig': (p, q, z)  -- Hashimoto symmetric
      'candidate_3_sig': (p, q, z)  -- P-point projector
      'result': True if the theorem statement holds
    """
    # Candidate 1: Edge Laplacian
    Q1 = B_inc.T @ B_inc
    sig1 = _signature(Q1)

    # Candidate 2: Hashimoto symmetric, restricted to 6-dim quotient
    B_hash = _hashimoto_k4()
    S_k4 = _symmetrizer_k4()
    Q2 = S_k4.T @ ((B_hash + B_hash.T) / 2.0) @ S_k4
    sig2 = _signature(Q2)

    # Candidate 3: P-point projector
    Q3 = _candidate_3_p_projector()
    sig3 = _signature(Q3)

    # Verify closed-form eigenvalues of Q3
    w3 = np.sort(np.linalg.eigvalsh(Q3))
    closed = np.sort(np.array([
        (3 - np.sqrt(3)) / 6, (3 - np.sqrt(3)) / 6,
        (3 + np.sqrt(3)) / 6, (3 + np.sqrt(3)) / 6,
        1.0, 1.0,
    ]))
    cf_err = float(np.max(np.abs(w3 - closed)))
    assert cf_err < 1e-8, f"closed-form mismatch: err={cf_err}"

    # Theorem assertions:
    # Candidates 1 and 2 are rank-deficient (not Cl(6) generators)
    # Candidate 3 has signature (6, 0, 0) -- Euclidean Cl(6, 0)
    result = (
        sig1 == (3, 0, 3)
        and sig2 == (1, 2, 3)
        and sig3 == (6, 0, 0)
    )
    return {
        'candidate_1_sig': sig1,
        'candidate_2_sig': sig2,
        'candidate_3_sig': sig3,
        'Q3_closed_form_err': cf_err,
        'result': result,
    }


# --- VALIDATION ----------------------------------------------
if __name__ == "__main__":
    out = verify_theorem_B2_signature()
    print(f"Candidate 1 (Edge Laplacian) signature: {out['candidate_1_sig']}")
    print(f"Candidate 2 (Hashimoto sym) signature:  {out['candidate_2_sig']}")
    print(f"Candidate 3 (P-point proj) signature:   {out['candidate_3_sig']}")
    print(f"Q3 closed-form eigenvalue error:         {out['Q3_closed_form_err']:.2e}")
    print(f"Result: {out['result']}")
    assert out['result'], "Theorem B2 verification failed"
    print(
        "Theorem B2 verified: only the P-point Ramanujan projector gives\n"
        "a non-degenerate form; its signature is (6, 0) => Cl(6, 0)."
    )
    print("OK")
