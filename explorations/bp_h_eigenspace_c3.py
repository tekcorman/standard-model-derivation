#!/usr/bin/env python3
"""
Compute the C3-irrep decomposition of the h-eigenspace of B(P) on srs.

Background
----------
From ../predictions/walker_dynamics_derivation.md: the walker's L-step amplitude at the
P-point is governed by B(P)^L on the 12-dim directed-edge space of the srs
primitive cell. From ../predictions/B_P_doubly_degenerate_h_derivation.md: B(P) has
h = (sqrt(3) + i*sqrt(5))/2 as an eigenvalue with multiplicity 2.

This script determines:
  - The C3 action on the 12-dim directed-edge space.
  - The C3-irrep content of the h-eigenspace of B(P) (the 2-dim C3-protected
    subspace).
  - Amplitudes of C3-trivial vs C3-charged components.

If the h-eigenspace decomposes as (trivial + omega) with some specific
amplitude ratio, then under the postulate (P-mass-spectrum) from
an internal working note, this computation resolves whether the
Koide epsilon = sqrt(2) is recovered or not.

Outcomes (per the scoping doc):
  A: h-eigenspace = (trivial + omega) with ratio |trivial|/|omega| specific;
     check if the implied Koide epsilon = sqrt(2).
  B: h-eigenspace is pure C3-irrep (all trivial or all omega).
  C: h-eigenspace is (omega + omega^2) with NO trivial component.
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import (
    find_bonds, K_STAR, N_ATOMS, C3_PERM, omega3,
)


K_P = (0.25, 0.25, 0.25)
H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2


# ======================================================================
# Directed-edge infrastructure
# ======================================================================

def build_directed_edges(bonds):
    return [tuple(b) for b in bonds]


def bloch_hashimoto(k_frac, directed):
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


# ======================================================================
# C3 action on directed edges
# ======================================================================

def c3_vertex_perm():
    """From proofs/common.py C3_PERM. The C3 rotation (x,y,z)->(z,x,y)
    permutes vertices: v0->v0, v1->v3, v2->v1, v3->v2."""
    perm = {}
    for i in range(4):
        for j in range(4):
            if abs(C3_PERM[i, j] - 1.0) < 1e-12:
                perm[j] = i  # C3 maps atom j to atom i
    return perm


def c3_cell_perm(cell):
    """C3: (n1,n2,n3) -> (n3, n1, n2) since C3 permutes a1->a2, a2->a3,
    a3->a1 (verified: C3 on real space maps a1=(-1,1,1)/2 to (1,-1,1)/2=a2)."""
    return (cell[2], cell[0], cell[1])


def build_c3_on_directed_edges(directed):
    """12x12 permutation matrix representing the C3 action on directed
    edges at the P-point (C3 fixes P so no phase)."""
    vp = c3_vertex_perm()
    n = len(directed)
    # Index lookup for post-permuted edges
    edge_to_idx = {de: i for i, de in enumerate(directed)}
    C3_mat = np.zeros((n, n), dtype=complex)
    for i, (src, tgt, cell) in enumerate(directed):
        new_edge = (vp[src], vp[tgt], c3_cell_perm(cell))
        j = edge_to_idx.get(new_edge)
        if j is None:
            # The permuted edge is the same undirected bond in a different
            # periodic image; find the equivalent representative.
            # srs primitive has each bond in a single representation;
            # if we don't find exact match, something's wrong.
            raise RuntimeError(
                f"C3 mapped edge {(src,tgt,cell)} to {new_edge}, not in directed set"
            )
        C3_mat[j, i] = 1.0
    return C3_mat


# ======================================================================
# Simultaneous diagonalization
# ======================================================================

def simultaneous_eigenspaces(B_P, C3_mat, target_eig=H_EXACT, tol=1e-7):
    """Find the h-eigenspace of B(P), then diagonalize C3 within it.
    Returns list of (h_eigenvec, c3_eigenval) pairs.

    B(P) is non-Hermitian so numpy's eig() returns non-orthonormal
    eigenvectors at a degenerate eigenvalue. We orthonormalize via QR
    before projecting C3 onto the subspace."""
    # Step 1: verify [B, C3] = 0
    commutator = B_P @ C3_mat - C3_mat @ B_P
    commutator_norm = la.norm(commutator)
    assert commutator_norm < 1e-10, f"[B(P), C3] != 0: ||.|| = {commutator_norm}"

    # Step 2: find h-eigenspace of B(P)
    evals, evecs = la.eig(B_P)
    idx = [i for i, ev in enumerate(evals) if abs(ev - target_eig) < tol]
    assert len(idx) == 2, f"expected h-multiplicity 2, got {len(idx)}"
    h_basis_raw = evecs[:, idx]  # 12x2 matrix (not orthonormal)

    # Step 2b: orthonormalize via QR
    Q, _ = la.qr(h_basis_raw)   # 12x2 orthonormal
    h_basis = Q

    # Step 2c: verify h_basis is still an eigenspace for B(P)
    residual_B = B_P @ h_basis - target_eig * h_basis
    assert la.norm(residual_B) < 1e-8, (
        f"orthonormalized h_basis is not B-eigenvector: "
        f"||(B - h*I) Q|| = {la.norm(residual_B)}"
    )

    # Step 3: project C3 onto this subspace (now correct since h_basis is ON)
    C3_sub = h_basis.conj().T @ C3_mat @ h_basis
    # C3_sub should have unit-modulus eigenvalues
    c3_evals, c3_evecs = la.eig(C3_sub)

    # Step 4: construct simultaneous eigenvectors
    simult_vecs = h_basis @ c3_evecs  # 12 x 2
    return simult_vecs, c3_evals


def label_c3(c3_val):
    """Classify a C3 eigenvalue as trivial (1), omega, or omega^2."""
    if abs(c3_val - 1.0) < 0.3:
        return 'trivial'
    elif abs(c3_val - omega3) < 0.3:
        return 'omega'
    elif abs(c3_val - omega3**2) < 0.3:
        return 'omega^2'
    else:
        return f'?({c3_val})'


# ======================================================================
# Vertex-level projection: what do h-eigenvectors look like on the
# vertex-weighted representation?
# ======================================================================

def vertex_amplitudes(vec, directed):
    """Collapse a 12-dim directed-edge vector to a 4-dim vertex vector
    by summing |amplitude|^2 over outgoing edges at each vertex."""
    weights = np.zeros(N_ATOMS)
    for i, (src, tgt, cell) in enumerate(directed):
        weights[src] += abs(vec[i])**2
    return weights


def c3_irrep_projections(vec_vertex):
    """Project a 4-vertex amplitude distribution onto C3 irreps using the
    C3_ESTATES basis from common.py.
    Returns (|trivial|^2, |omega|^2, |omega^2|^2), normalized."""
    # The 4-vertex space decomposes as 2*trivial + omega + omega^2.
    # Basis (from common.py):
    # - trivial_0 = e_0 (vertex v0 alone)
    # - trivial_s = (e_1 + e_2 + e_3)/sqrt(3) (sum over v1,v2,v3)
    # - omega  = (0, 1, omega, omega^2)/sqrt(3)
    # - omega^2 = (0, 1, omega^2, omega)/sqrt(3)

    trivial_0 = np.array([1, 0, 0, 0], dtype=complex)
    trivial_s = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)
    omega_v   = np.array([0, 1, omega3, omega3**2], dtype=complex) / np.sqrt(3)
    omega2_v  = np.array([0, 1, omega3**2, omega3], dtype=complex) / np.sqrt(3)

    # "vec_vertex" here is a 4-dim REAL non-negative weight vector, not a
    # complex amplitude vector. For amplitude projection we should use
    # the actual complex vector. But for weight analysis we square it.
    # Return the raw weights for diagnostic.
    return {
        'weight_v0': float(vec_vertex[0]),
        'weight_v1': float(vec_vertex[1]),
        'weight_v2': float(vec_vertex[2]),
        'weight_v3': float(vec_vertex[3]),
        'weight_v1v2v3': float(sum(vec_vertex[1:])),
        'trivial_fraction': float(vec_vertex[0] / sum(vec_vertex)) if sum(vec_vertex) > 0 else 0,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    assert len(directed) == 12

    B_P = bloch_hashimoto(K_P, directed)
    C3_mat = build_c3_on_directed_edges(directed)

    # Verify C3 has order 3 on directed edges
    C3_cubed = C3_mat @ C3_mat @ C3_mat
    assert la.norm(C3_cubed - np.eye(12)) < 1e-10, f"C3^3 != I on directed edges"

    # Find h-eigenspace and diagonalize C3 within it
    h_evecs, h_c3_evals = simultaneous_eigenspaces(B_P, C3_mat, H_EXACT)

    print("=" * 70)
    print("  h-EIGENSPACE OF B(P): C3 DECOMPOSITION")
    print("=" * 70)
    print(f"h = {H_EXACT}")
    print(f"Directed-edge space dim = 12, h-eigenspace dim = 2")
    print()

    for k in range(h_evecs.shape[1]):
        vec = h_evecs[:, k]
        c3_eig = h_c3_evals[k]
        label = label_c3(c3_eig)
        # Normalize vec
        norm = la.norm(vec)
        vec_n = vec / norm
        v_weights = vertex_amplitudes(vec_n, directed)
        proj = c3_irrep_projections(v_weights)
        print(f"--- h-eigenvector #{k} ---")
        print(f"  C3 eigenvalue         = {c3_eig}")
        print(f"  C3 irrep label        = {label}")
        print(f"  vertex weights        = {v_weights}")
        print(f"  weight(v0)/total      = {proj['trivial_fraction']:.6f}")
        print()

    # Same analysis for h* eigenspace
    print("=" * 70)
    print("  h*-EIGENSPACE OF B(P): C3 DECOMPOSITION")
    print("=" * 70)
    hstar = H_EXACT.conjugate()
    hs_evecs, hs_c3_evals = simultaneous_eigenspaces(B_P, C3_mat, hstar)
    for k in range(hs_evecs.shape[1]):
        vec = hs_evecs[:, k]
        c3_eig = hs_c3_evals[k]
        label = label_c3(c3_eig)
        norm = la.norm(vec)
        vec_n = vec / norm
        v_weights = vertex_amplitudes(vec_n, directed)
        proj = c3_irrep_projections(v_weights)
        print(f"--- h*-eigenvector #{k} ---")
        print(f"  C3 eigenvalue         = {c3_eig}")
        print(f"  C3 irrep label        = {label}")
        print(f"  vertex weights        = {v_weights}")
        print(f"  weight(v0)/total      = {proj['trivial_fraction']:.6f}")
        print()

    # -h eigenspace (also Ramanujan-saturated, from -sqrt(3) A-eigenspace)
    print("=" * 70)
    print("  -h-EIGENSPACE OF B(P): C3 DECOMPOSITION")
    print("=" * 70)
    mh_evecs, mh_c3_evals = simultaneous_eigenspaces(B_P, C3_mat, -H_EXACT)
    for k in range(mh_evecs.shape[1]):
        vec = mh_evecs[:, k]
        c3_eig = mh_c3_evals[k]
        label = label_c3(c3_eig)
        norm = la.norm(vec)
        vec_n = vec / norm
        v_weights = vertex_amplitudes(vec_n, directed)
        print(f"--- -h-eigenvector #{k} ---")
        print(f"  C3 eigenvalue         = {c3_eig}")
        print(f"  C3 irrep label        = {label}")
        print(f"  vertex weights        = {v_weights}")
        print()

    # -h* eigenspace
    print("=" * 70)
    print("  -h*-EIGENSPACE OF B(P): C3 DECOMPOSITION")
    print("=" * 70)
    mhs_evecs, mhs_c3_evals = simultaneous_eigenspaces(B_P, C3_mat, -H_EXACT.conjugate())
    for k in range(mhs_evecs.shape[1]):
        vec = mhs_evecs[:, k]
        c3_eig = mhs_c3_evals[k]
        label = label_c3(c3_eig)
        norm = la.norm(vec)
        vec_n = vec / norm
        v_weights = vertex_amplitudes(vec_n, directed)
        print(f"--- -h*-eigenvector #{k} ---")
        print(f"  C3 eigenvalue         = {c3_eig}")
        print(f"  C3 irrep label        = {label}")
        print(f"  vertex weights        = {v_weights}")
        print()

    # Combined h + h* eigenspace analysis: the "mass 3-vector" candidate
    # spans both.
    print("=" * 70)
    print("  COMBINED (h, h*) EIGENSPACE: 4-DIM, CANDIDATE FOR 3-GENERATION")
    print("=" * 70)
    # The combined space is 4-dim. Its C3 content:
    c3_labels_h = [label_c3(ev) for ev in h_c3_evals]
    c3_labels_hs = [label_c3(ev) for ev in hs_c3_evals]
    print(f"h-eigenspace C3 content:  {c3_labels_h}")
    print(f"h*-eigenspace C3 content: {c3_labels_hs}")
    all_labels = c3_labels_h + c3_labels_hs
    from collections import Counter
    counts = Counter(all_labels)
    print(f"Combined count: {dict(counts)}")

    # Check: if combined = {trivial: 2, omega: 1, omega^2: 1}, this
    # matches the natural 4-dim C3 rep of the vertex space.
    # If it's {trivial: 1, omega: 1, omega^2: 1, ???}, something's off.

    # Amplitude ratio computation for the scoping test:
    # On h-eigenspace alone, what's |charged|^2 / |trivial|^2?
    # Using the C3-diagonalized vectors, each vector has a single C3 label,
    # so within h-eigenspace we have ONE trivial + ONE charged (or
    # two-of-a-kind).
    if Counter(c3_labels_h) == Counter(['trivial', 'omega']):
        print()
        print("Outcome B (per scoping): h-eigenspace is (trivial + omega).")
        print("Under (P-mass-spectrum), we need a specific ratio rule to recover")
        print("epsilon = sqrt(2). Each C3 eigenvector has equal normalization")
        print("(both unit vectors), so the 'ratio' is 1 : 1. This gives")
        print("epsilon = 2 * 1 / 1 = 2, NOT sqrt(2). Q_Koide fails to close at")
        print("the simplest ratio assumption.")
    elif Counter(c3_labels_h) == Counter(['omega', 'omega^2']):
        print()
        print("Outcome A: h-eigenspace is pure C3-charged (omega + omega^2),")
        print("contains NO trivial component. This does not match the")
        print("Koide structure (which needs a trivial component).")
    elif Counter(c3_labels_h) == Counter(['trivial', 'trivial']):
        print()
        print("Outcome B': h-eigenspace is doubly-trivial. No Koide structure.")
    else:
        print()
        print(f"Outcome C: unexpected decomposition {dict(Counter(c3_labels_h))}.")


if __name__ == "__main__":
    main()
