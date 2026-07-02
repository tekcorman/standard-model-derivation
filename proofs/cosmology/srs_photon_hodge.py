#!/usr/bin/env python3
"""
Path B'' session 3 step 1: build d_1: C^1 -> C^2 from 10-cycles of srs
and compute the photon Hodge Laplacian d_1^† d_1 restricted to ker d†.

Uses the 6 inequivalent 10-cycles found by srs_cycle_enumerator.py as
the 2-cell complex.

Tests:
1. Hodge consistency: d_1(k) · d(k)^† = 0 and d_1(k) · χ_C(k) has
   zero vertex boundary (equivalently, d(k)^† · χ_C(k)^* = 0).
   [Actually we test that each χ_C lies in ker d†.]
2. Rank of d_1(k) at each k-point matches dim ker d†(k).
3. Compute d_1^† d_1 restricted to ker d† and report photon
   spectrum at Γ, H, P, N.
"""

import os
import sys
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
from srs_cycle_enumerator import enumerate_simple_cycles


def build_edge_lookup(canonical_edges):
    """
    Build a lookup from (v_src, v_tgt, cell) to (edge_idx, sign).
    The canonical edge stores only one orientation; traversals in the
    other direction get sign = -1.
    """
    lookup = {}
    for (e_idx, v_s, v_t, cell) in canonical_edges:
        lookup[(v_s, v_t, tuple(cell))] = (e_idx, +1)
        neg_cell = tuple(-c for c in cell)
        lookup[(v_t, v_s, neg_cell)] = (e_idx, -1)
    return lookup


def cycle_to_edge_vector(cycle_path, edge_lookup, k_red):
    """
    Express a cycle as a vector in C^1(k).

    cycle_path: list of (vertex_idx, cell_offset) with
        cycle_path[0] == cycle_path[-1]
    edge_lookup: dict from (v_s, v_t, cell) to (edge_idx, sign)
    k_red: reduced momentum vector

    The Bloch phase convention: each traversal v_i@c_i → v_{i+1}@c_{i+1}
    contributes (sign) · exp(2πi k · c_i) to the edge index (where
    c_i is the cell offset of the step's source vertex).

    Returns: complex vector of length 6 (the number of primitive edges).
    """
    n_edges = max(e[0] for e in edge_lookup.values()) + 1
    chi = np.zeros(n_edges, dtype=complex)
    for i in range(len(cycle_path) - 1):
        v_i, c_i = cycle_path[i]
        v_ip1, c_ip1 = cycle_path[i + 1]
        bond_cell = tuple(c_ip1[j] - c_i[j] for j in range(3))
        e_idx, sign = edge_lookup[(v_i, v_ip1, bond_cell)]
        # Phase for the traversal at cell offset c_i. The correct cell
        # depends on the orientation: if sign = +1 (canonical direction),
        # the edge's source lives at c_i; if sign = -1 (reverse), the
        # edge's *canonical* source lives at c_{i+1} = c_i + bond_cell.
        if sign == +1:
            c_ref = c_i
        else:
            c_ref = c_ip1
        phase = np.exp(-2j * np.pi * np.dot(k_red, np.array(c_ref, dtype=float)))
        chi[e_idx] += sign * phase
    return chi


def build_d1(cycles, edge_lookup, k_red, n_edges):
    """
    Build d_1(k): C^1 -> C^2 as an (n_cycles × n_edges) matrix.

    Row i = χ_{cycle_i}(k) — the edge vector of the i-th cycle.
    """
    n_cycles = len(cycles)
    d1 = np.zeros((n_cycles, n_edges), dtype=complex)
    for i, (length, path) in enumerate(cycles):
        d1[i] = cycle_to_edge_vector(path, edge_lookup, k_red)
    return d1


def main():
    print("=" * 70)
    print("  Path B'' session 3 step 1: build d_1 from srs 10-cycles")
    print("=" * 70)

    # Rebuild srs primitive cell
    verts, lat_vecs = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat_vecs)
    edges = canonical_edges_primitive(bonds)
    n_edges = len(edges)
    n_verts = len(verts)
    edge_lookup = build_edge_lookup(edges)

    print(f"\n  {n_verts} primitive vertices, {n_edges} primitive edges")

    # Enumerate 10-cycles
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    n_cycles = len(cycles)
    print(f"  {n_cycles} inequivalent simple cycles (length ≤ 10)")

    # Hodge consistency check: the chain complex condition is d_1 · d_0 = 0,
    # equivalently d_0† · d_1† = 0. With row-cycles in d_1, d_1† column j is
    # conj(d_1[j])^T. So we need d_0†(k) · conj(d_1[j])^T = 0 for every row j.
    print("\n--- Hodge consistency: d_1 · d_0 = 0 (chain complex) ---")
    for label in ["Γ", "H", "P", "N"]:
        k_red = HIGH_SYM_POINTS[label]
        d = incidence_matrix_primitive(k_red, edges, n_verts)
        d1 = build_d1(cycles, edge_lookup, k_red, n_edges)
        # Chain complex: d_1 · d_0 should be zero
        chain_err = d1 @ d
        max_chain = np.max(np.abs(chain_err))
        # Equivalent adjoint form: d_0† · d_1† = 0
        adjoint_err = d.conj().T @ d1.conj().T
        max_adj = np.max(np.abs(adjoint_err))
        print(f"  {label:4s}: max |d_1 · d_0|       = {max_chain:.3e}")
        print(f"        max |d_0† · d_1†|     = {max_adj:.3e}")
        # rank of d_1(k)
        s = la.svd(d1, compute_uv=False)
        rank_d1 = int(np.sum(s > 1e-9))
        ker_dT_dim = n_edges - int(np.sum(la.svd(d, compute_uv=False) > 1e-9))
        print(f"        rank(d_1({label})) = {rank_d1}   "
              f"vs   dim ker d†({label}) = {ker_dT_dim}")

    print("\n--- Photon spectrum via d_1^† d_1 restricted to ker d† ---")
    for label in ["Γ", "H", "P", "N"]:
        k_red = HIGH_SYM_POINTS[label]
        d = incidence_matrix_primitive(k_red, edges, n_verts)
        d1 = build_d1(cycles, edge_lookup, k_red, n_edges)

        # Full 1-form Hodge Laplacian: Δ_1 = d d† + d_1† d_1
        L_long = d @ d.conj().T
        L_curl = d1.conj().T @ d1
        Delta_1 = L_long + L_curl
        Delta_1 = (Delta_1 + Delta_1.conj().T) / 2

        eigvals = np.sort(la.eigvalsh(Delta_1).real)
        print(f"\n  k = {label}")
        print(f"    Δ_1 spectrum (full, {n_edges} modes):")
        for i, ev in enumerate(eigvals):
            print(f"      ω²_{i:2d} = {ev: .6f}")

        # Photon spectrum = eigenvalues of Δ_1 restricted to ker d†
        # Build projector onto ker d†:
        U, S, Vt = la.svd(d)
        # Null space of d^T = null space of V^T where singular values ≈ 0
        # Actually: ker(d^†) = col space of U for zero singular values.
        # For an (n_edges × n_verts) matrix d, U is (n_edges × n_edges).
        # Zero singular values in S indicate cokernel directions.
        n_sv = len(S)
        mask_zero = S < 1e-9
        # If d has more rows than cols, U has extra "cokernel" columns:
        cokern_basis = U[:, list(range(n_sv))][:, mask_zero]
        # Add the "pure cokernel" columns beyond the first n_verts:
        if n_edges > n_verts:
            cokern_basis = np.hstack([cokern_basis, U[:, n_verts:]])

        print(f"    cokern_basis shape = {cokern_basis.shape} (should equal "
              f"({n_edges}, dim ker d†))")

        # Restrict Δ_1 to cokern_basis
        Delta_1_transverse = cokern_basis.conj().T @ Delta_1 @ cokern_basis
        Delta_1_transverse = (Delta_1_transverse + Delta_1_transverse.conj().T) / 2
        photon_eigs = np.sort(la.eigvalsh(Delta_1_transverse).real)
        print(f"    photon ω² (transverse): {photon_eigs}")


if __name__ == "__main__":
    main()
