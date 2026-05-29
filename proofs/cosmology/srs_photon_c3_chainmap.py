#!/usr/bin/env python3
"""
Lemma 3 (β closure) — construct C₃ as a chain-map across the Hodge complex
C^0 ← C^1 ← C^2 of the srs primitive cell at the P-point, with proper
Bloch-phase corrections.

Goals
-----
1. Build C₃_v on vertices (4×4), C₃_e on canonical undirected edges (6×6,
   with sign flips for orientation reversals + Bloch phases at k_P).
2. Verify chain-map condition d·C₃_v = C₃_e·d  exactly at k_P.
3. Verify [C₃_e, Δ_1] = 0 to machine precision (where Δ_1 = d·d† + d_1†·d_1
   is the photon Hodge Laplacian).
4. Restrict C₃_e to the doubly-degenerate photon eigenspace at ω² = 36
   and read off the C₃-irrep decomposition (expected: ω ⊕ ω², matching
   helicity ±1 for the photon propagating along [111] at k_P).

Output: a single, self-contained verification script. If all checks PASS,
this closes the structural piece needed to assign L = ω-irrep, R = ω²-irrep
on the photon Hodge bundle at P.

Notation reminders
------------------
* The 4 primitive vertices, 6 canonical undirected edges, and 6 inequivalent
  length-10 cycles are taken from `srs_photon_bloch_primitive.py` and
  `srs_cycle_enumerator.py`.
* The C₃ rotation about [111] is the matrix R_3 = [[0,0,1],[1,0,0],[0,1,0]],
  which permutes lattice vectors a₁→a₂→a₃→a₁ (i.e. σ(1)=2, σ(2)=3, σ(3)=1)
  and primitive vertices via atom_perm = {0:0, 1:3, 2:1, 3:2}.
* The script's Bloch convention (deduced from the d-matrix in
  `incidence_matrix_primitive`):  ψ(v at R) ∝ e^{-2πi k·R} ψ̃(v, k).
  This makes the orientation-flip Bloch phase under C₃ equal to
  -exp(+2πi k_P · cell_source) — note the sign in the exponent.
* k_P = (1/4, 1/4, 1/4) is C₃-invariant: C₃·k_P = k_P. So no inter-k mixing;
  C₃ acts as a closed operator on the Bloch fibre at k_P.
"""

import os
import sys
import math
import numpy as np
from numpy import linalg as la

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
    canonical_edges_primitive,
    incidence_matrix_primitive,
    HIGH_SYM_POINTS,
)
from srs_photon_hodge import build_d1, build_edge_lookup
from srs_cycle_enumerator import enumerate_simple_cycles


# =============================================================================
# Group-action primitives.
# =============================================================================

# C₃ rotation about [111]: (x,y,z) → (z,x,y).  Permutes primitive lattice
# vectors a₁→a₂→a₃→a₁, i.e. σ on the index set {1,2,3} is (1→2,2→3,3→1).
ATOM_PERM = {0: 0, 1: 3, 2: 1, 3: 2}        # vertex permutation under C₃.

def c3_cell(c):
    """Primitive-cell-coordinate action of C₃: new[b] = old[σ⁻¹(b)].
    σ⁻¹: 1→3, 2→1, 3→2.  In 0-indexed: new = (old[2], old[0], old[1]).
    Equivalently: cell coefficients cycle (c₁,c₂,c₃) → (c₃,c₁,c₂)."""
    return (c[2], c[0], c[1])

def c3_cell_inv(c):
    """Inverse: new[b] = old[σ(b)] → (c₁,c₂,c₃) → (c₂,c₃,c₁)."""
    return (c[1], c[2], c[0])

K_P_RED = np.array([0.25, 0.25, 0.25])      # C₃-invariant high-symmetry point.


# =============================================================================
# Build C₃ on vertices and edges with Bloch phases at k_P.
# =============================================================================

def build_C3_vertex(n_verts=4):
    """C₃_v on the 4 primitive vertices.  k_P-invariant ⇒ no Bloch phase.
    Convention: (C₃_v ψ)[v] = ψ[atom_perm⁻¹(v)] (pull-back), equivalently
    column j of C₃_v is the unit vector e_{atom_perm[j]}."""
    C = np.zeros((n_verts, n_verts), dtype=complex)
    for j, i_target in ATOM_PERM.items():
        C[i_target, j] = 1.0
    return C


def build_C3_edge(edges, k_red):
    """C₃_e on the 6 canonical undirected edges, at fixed Bloch momentum k_red.

    For each canonical edge l = (e_idx, v_s, v_t, cell), C₃ sends
    (v_s at R, v_t at R+cell) to (atom_perm[v_s] at C₃R, atom_perm[v_t]
    at C₃R + C₃·cell).

    Case A — no orientation flip (atom_perm[v_s] ≤ atom_perm[v_t]):
        the image is exactly canonical edge k = (atom_perm[v_s],
        atom_perm[v_t], C₃·cell), at home cell C₃R.  Bloch phase = +1
        (k_P-invariance of the home cell sum).
        ⇒  C₃_e[k, l] = +1.

    Case B — orientation flip (atom_perm[v_s] > atom_perm[v_t]):
        the image equals -1 × (canonical edge k = (atom_perm[v_t],
        atom_perm[v_s], -C₃·cell) at home cell C₃R + C₃·cell).  In Bloch
        space this picks up an extra phase exp(+2πi k_P · cell)
        (positive exponent, due to the conjugate Bloch convention used
        by `incidence_matrix_primitive`).  Net coefficient:
        ⇒  C₃_e[k, l] = -1 × exp(+2πi k_P · cell).
    """
    n_edges = len(edges)
    C = np.zeros((n_edges, n_edges), dtype=complex)

    canon_lookup = {(vs, vt, c): e for (e, vs, vt, c) in edges}

    for (l_idx, v_s, v_t, cell) in edges:
        new_vs, new_vt = ATOM_PERM[v_s], ATOM_PERM[v_t]
        new_cell = c3_cell(cell)
        if new_vs <= new_vt:
            key = (new_vs, new_vt, new_cell)
            assert key in canon_lookup, f"C₃ image of e{l_idx} = {key} not in canon"
            k_idx = canon_lookup[key]
            C[k_idx, l_idx] = 1.0
        else:
            neg_cell = tuple(-c for c in new_cell)
            key = (new_vt, new_vs, neg_cell)
            assert key in canon_lookup, f"C₃ image (rev) of e{l_idx} = {key} not in canon"
            k_idx = canon_lookup[key]
            C[k_idx, l_idx] = -np.exp(1j * 2 * math.pi * np.dot(k_red, cell))
    return C


# =============================================================================
# Build C₃_2 on cycles by lifting from the action on edges.
# =============================================================================

def build_C3_cycle(cycles, edge_lookup, k_red, n_edges, C3_e, d1):
    """C₃_2 on the 6 length-10 cycles.  Strategy: each row of d_1 is a
    cycle's edge-vector χ_C(k); C₃ acts on this edge-vector via C₃_e.
    The image C₃_e · χ_C must equal another cycle's edge-vector (up to
    sign), since C₃ permutes cycles.  We solve d_1 · C₃_e = C₃_2 · d_1
    by least-squares matching: for each row j of d_1, find the row k of
    d_1 that best matches (C₃_e · row_j)†."""
    n_cycles = len(cycles)
    # Build d_1 image rows: (d_1 · C₃_e†) with C₃_e acting on edges.
    # Actually we want d_1 · C₃_e (as 6×6).  Then for each row j of
    # (d_1 · C₃_e), find row k of d_1 such that row_j ≈ s · row_k for
    # some scalar s.  C₃_2[k, j] = s, all other entries 0.
    LHS = d1 @ C3_e        # (n_cycles × n_edges)
    C2 = np.zeros((n_cycles, n_cycles), dtype=complex)
    for j in range(n_cycles):
        target_row = LHS[j]                                # length n_edges
        best = (None, None, np.inf)
        for k in range(n_cycles):
            ref = d1[k]
            # Find scalar s such that target_row ≈ s · ref.
            denom = np.vdot(ref, ref).real
            if denom < 1e-12:
                continue
            s = np.vdot(ref, target_row) / denom
            err = la.norm(target_row - s * ref)
            if err < best[2]:
                best = (k, s, err)
        k_best, s_best, err_best = best
        C2[k_best, j] = s_best
        # diagnostic: store residual for verification print
    return C2


# =============================================================================
# Build the photon Hodge Laplacian at k_P.
# =============================================================================

def build_delta_1(d, d1):
    """Δ_1 = d · d† + d_1† · d_1, hermitized."""
    L = d @ d.conj().T + d1.conj().T @ d1
    return (L + L.conj().T) / 2


# =============================================================================
# Driver.
# =============================================================================

def main():
    print("=" * 72)
    print("Lemma 3 — C₃ chain-map on the srs photon Hodge complex at k_P")
    print("=" * 72)

    # Geometry.
    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    n_verts, n_edges = len(verts), len(edges)
    edge_lookup = build_edge_lookup(edges)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    n_cycles = len(cycles)

    print(f"\nPrimitive cell: {n_verts} vertices, {n_edges} edges, "
          f"{n_cycles} length-10 cycles")

    # Operators at k_P.
    k_red = K_P_RED
    d = incidence_matrix_primitive(k_red, edges, n_verts)
    d1 = build_d1(cycles, edge_lookup, k_red, n_edges)

    print(f"\n--- Step 1: Build C₃_v (4×4) and C₃_e (6×6) ---")
    C3_v = build_C3_vertex(n_verts)
    C3_e = build_C3_edge(edges, k_red)
    print(f"  C₃_v unitary?  max|C₃_v†·C₃_v − I| = "
          f"{np.max(np.abs(C3_v.conj().T @ C3_v - np.eye(n_verts))):.2e}")
    print(f"  C₃_e unitary?  max|C₃_e†·C₃_e − I| = "
          f"{np.max(np.abs(C3_e.conj().T @ C3_e - np.eye(n_edges))):.2e}")
    print(f"  C₃_v³ = I?     max|C₃_v³ − I|       = "
          f"{np.max(np.abs(C3_v @ C3_v @ C3_v - np.eye(n_verts))):.2e}")
    print(f"  C₃_e³ = I?     max|C₃_e³ − I|       = "
          f"{np.max(np.abs(C3_e @ C3_e @ C3_e - np.eye(n_edges))):.2e}")

    print(f"\n--- Step 2: Chain-map condition d · C₃_v = C₃_e · d at k_P ---")
    chain_err = d @ C3_v - C3_e @ d
    chain_err_max = np.max(np.abs(chain_err))
    print(f"  max|d · C₃_v − C₃_e · d| = {chain_err_max:.2e}")
    if chain_err_max < 1e-10:
        print(f"  PASS — chain-map at vertex/edge level.")
    else:
        print(f"  FAIL — chain map broken; printing the offending matrix:")
        print(np.round(chain_err, 4))
        return

    print(f"\n--- Step 3: Verify C₃_e preserves im(d_1†) ⇔ [C₃_e, d_1†·d_1] = 0 ---")
    print(f"  (rank(d_1) = {int(np.sum(la.svd(d1, compute_uv=False) > 1e-9))} "
          f"at k_P, so C₃_2 on cycles is determined only on im(d_1).  We verify")
    print(f"  the operator-level condition [C₃_e, d_1†·d_1] = 0 directly.)")
    L_curl = d1.conj().T @ d1
    curl_commutator = C3_e @ L_curl - L_curl @ C3_e
    curl_commutator_max = np.max(np.abs(curl_commutator))
    print(f"  max|[C₃_e, d_1† · d_1]| = {curl_commutator_max:.2e}")
    if curl_commutator_max < 1e-10:
        print(f"  PASS — C₃_e commutes with the curl-sector Laplacian.")
    else:
        print(f"  FAIL — C₃_e does not preserve im(d_1†).")
        return

    print(f"\n--- Step 4: Verify [C₃_e, Δ_1] = 0 ---")
    Delta_1 = build_delta_1(d, d1)
    commutator = C3_e @ Delta_1 - Delta_1 @ C3_e
    commutator_max = np.max(np.abs(commutator))
    print(f"  max|[C₃_e, Δ_1]| = {commutator_max:.2e}")
    if commutator_max < 1e-10:
        print(f"  PASS — C₃_e commutes with the photon Hodge Laplacian.")
    else:
        print(f"  FAIL — commutator non-vanishing.")
        return

    print(f"\n--- Step 5: Photon spectrum + ω²=36 eigenspace identification ---")
    eigs_full, vecs_full = la.eig(Delta_1)
    # sort by real eigenvalue
    order = np.argsort(eigs_full.real)
    eigs_full = eigs_full[order]
    vecs_full = vecs_full[:, order]
    for i, ev in enumerate(eigs_full):
        print(f"  ω²_{i} = {ev.real:+.6f}  (Im = {ev.imag:+.2e})")

    # Photon = transverse modes in ker d†.  Identify the doubly-degenerate
    # eigenvalue at ω² = 36.
    target = 36.0
    mask = np.abs(eigs_full.real - target) < 1e-6
    photon_basis = vecs_full[:, mask]
    n_photon = photon_basis.shape[1]
    print(f"\n  Photon eigenspace at ω² = {target}: dim = {n_photon} "
          f"(expected 2)")
    assert n_photon == 2, f"expected doubly-degenerate, got {n_photon}-fold"
    # Confirm photon basis is in ker d† (transverse).
    longitudinal_norm = la.norm(d.conj().T @ photon_basis)
    print(f"  ‖d† · photon_basis‖ = {longitudinal_norm:.2e} "
          f"(transverse if ≈ 0)")

    print(f"\n--- Step 6: Diagonalize C₃_e | (photon ω² = 36 eigenspace) ---")
    # Orthonormalize the photon basis.
    Q, _ = la.qr(photon_basis)
    # Restrict C₃_e to span(Q): C₃_in_photon = Q† · C₃_e · Q.
    C3_photon = Q.conj().T @ C3_e @ Q
    print(f"  C₃ in 2D photon eigenspace (matrix entries):")
    for row in C3_photon:
        print("   ", "  ".join(f"{x.real:+.4f}{x.imag:+.4f}j" for x in row))
    print(f"  trace(C₃|photon) = {np.trace(C3_photon):+.6f}")
    print(f"  det(C₃|photon)   = {la.det(C3_photon):+.6f}")
    print(f"  expected for ω⊕ω²: trace = ω + ω² = -1, det = ω·ω² = +1")

    eigvals_C3, eigvecs_C3 = la.eig(C3_photon)
    omega = np.exp(2j * math.pi / 3)
    omega2 = omega.conjugate()
    print(f"\n  Eigenvalues of C₃|photon:")
    for i, ev in enumerate(eigvals_C3):
        # Identify which root of unity this is.
        d_omega = abs(ev - omega)
        d_omega2 = abs(ev - omega2)
        d_one = abs(ev - 1.0)
        if d_omega < 1e-8:
            label = "ω = e^(2πi/3)"
        elif d_omega2 < 1e-8:
            label = "ω² = e^(-2πi/3)"
        elif d_one < 1e-8:
            label = "1 (trivial)"
        else:
            label = "?? not a cube root of unity"
        print(f"    eigval {i}: {ev.real:+.6f} {ev.imag:+.6f}j  → {label}")

    expected = {abs(round(ev.real, 6) - omega.real) + abs(round(ev.imag, 6)
                  - omega.imag) for ev in eigvals_C3}
    irreps = sorted([
        ("ω" if abs(ev - omega) < 1e-8 else
         "ω²" if abs(ev - omega2) < 1e-8 else
         "1" if abs(ev - 1.0) < 1e-8 else "?")
        for ev in eigvals_C3
    ])
    print(f"\n  Photon C₃-irrep decomposition: {' ⊕ '.join(irreps)}")
    if set(irreps) == {"ω", "ω²"}:
        print(f"  PASS — photon at P-point splits as ω ⊕ ω²,")
        print(f"        matching helicity ±1 of a spin-1 photon along [111].")
        print(f"        L (helicity +1) = ω-irrep, R (helicity −1) = ω²-irrep.")
    else:
        print(f"  UNEXPECTED — irrep decomposition is {irreps}")
        print(f"  (expected {{ω, ω²}})")

    print("\n" + "=" * 72)
    print("Lemma 3 (Step A) — chain-map construction complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
