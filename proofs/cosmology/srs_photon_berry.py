#!/usr/bin/env python3
"""
Path B'' session 5a: non-Abelian Berry curvature on the srs photon
multiplet.

Uses the Fukui-Hatsugai-Suzuki link variable method. For each pair
of adjacent k-points k, k' in a grid, compute the overlap matrix
M_{μν}(k, k') = ⟨ψ_μ(k)|ψ_ν(k')⟩ restricted to the degenerate photon
multiplet at k (here 2-dim for srs generic k). The link variable is
U(k, k') = M / (det M) · SVD cleanup  — a U(2) matrix.

The plaquette:
    F(k, μν) = U_μ(k) · U_ν(k + δ_μ) · U_μ†(k + δ_ν) · U_ν†(k)
is a U(2) matrix whose determinant is the Abelian Berry flux on the
plaquette.

For validation: compute the first Chern number on a 2D (k_x, k_y) slice
at fixed k_z, for several k_z values, via
    c_1(k_z) = (1/2πi) Σ_plaq log det F
This should be an integer (topological) and may or may not vary with
k_z depending on whether Weyl points lie in between.

Main script flow:
1. Build photon eigenvector grabber
2. Build link variable function (with gauge parallel transport)
3. Compute 2D slice Chern number at fixed k_z
4. Scan k_z and report
5. If non-trivial, flag for session-5b full 3D axion θ calculation.
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
)
from srs_cycle_enumerator import enumerate_simple_cycles
from srs_photon_hodge import build_edge_lookup, build_d1


# =============================================================================
# 1. PHOTON EIGENVECTOR GRABBER
# =============================================================================

def photon_eigenvectors(k_red, edges, cycles, edge_lookup, n_verts=4,
                        n_photon=2, tol=1e-9):
    """
    At momentum k_red, return an orthonormal basis of the 2-dim photon
    subspace = ker d†(k) (for generic k).

    Returns: (6, n_photon) complex matrix whose columns span ker d†.
    """
    n_edges = 6
    d = incidence_matrix_primitive(k_red, edges, n_verts)

    # Compute an orthonormal basis of ker d†(k) = cokernel of d
    # d is (n_edges, n_verts). ker d† = left null space of d.
    # Use SVD: d = U S V†, then U[:, rank(d):] is an orthonormal basis of ker d†.
    U, S, Vt = la.svd(d)
    rank_d = int(np.sum(S > tol))
    # Left null space has dim n_edges - rank_d
    n_null = n_edges - rank_d
    if n_null < n_photon:
        raise ValueError(f"ker d†(k) has dim {n_null} < {n_photon} at k={k_red}")
    # First n_null columns of U[:, rank_d:] span the null space
    basis = U[:, rank_d:rank_d + n_photon]
    return basis  # shape (6, n_photon)


# =============================================================================
# 2. LINK VARIABLES (GAUGE PARALLEL TRANSPORT)
# =============================================================================

def link_variable(psi_k, psi_kp, tol=1e-12):
    """
    Compute the U(N) link variable from psi_k to psi_kp via maximum
    overlap (SVD parallel transport).

    psi_k, psi_kp: (n_edges, N) orthonormal bases.

    Returns: (N, N) U(N) matrix M such that psi_kp · M† is "aligned"
    with psi_k, and M = exp(-i A · δk) in the continuum limit.
    """
    M = psi_k.conj().T @ psi_kp   # N x N overlap matrix
    # Project onto U(N) via SVD polar decomposition
    U_svd, S_svd, Vh_svd = la.svd(M)
    U = U_svd @ Vh_svd
    return U


# =============================================================================
# 3. 2D SLICE CHERN NUMBER via FHS method
# =============================================================================

def slice_chern_number(k_z, N, edges, cycles, edge_lookup, verbose=False):
    """
    Compute the first Chern number of the photon bundle on a 2D
    (k_x, k_y) slice at fixed k_z, using an N×N grid.

    FHS formula:
      c_1 = (1/2πi) Σ_plaq  log det F_plaq
    where F_plaq = U_x(k) U_y(k+x̂) U_x†(k+ŷ) U_y†(k)
    is the plaquette Wilson loop and log is the principal branch.

    The determinant is used because for non-Abelian bundles, we sum
    the trace of F (= det of link variables for U(N)).
    """
    # Build grid
    ks_x = np.linspace(-0.5, 0.5, N, endpoint=False)
    ks_y = np.linspace(-0.5, 0.5, N, endpoint=False)
    # Offset slightly to avoid Γ if k_z = 0
    offset = 1.0 / (2 * N) if abs(k_z) < 1e-10 else 0.0
    ks_x = ks_x + offset
    ks_y = ks_y + offset

    # Compute photon eigenvectors on the grid (with periodic boundary)
    psi_grid = np.empty((N, N, 6, 2), dtype=complex)
    for i in range(N):
        for j in range(N):
            k = np.array([ks_x[i], ks_y[j], k_z])
            try:
                psi_grid[i, j] = photon_eigenvectors(k, edges, cycles, edge_lookup)
            except ValueError as e:
                if verbose:
                    print(f"  skipping k={k}: {e}")
                return None

    # Compute link variables
    # Ux[i,j] = link from (i,j) to (i+1,j)  (wrap around in x)
    # Uy[i,j] = link from (i,j) to (i,j+1)  (wrap around in y)
    Ux = np.empty((N, N, 2, 2), dtype=complex)
    Uy = np.empty((N, N, 2, 2), dtype=complex)
    for i in range(N):
        for j in range(N):
            ip = (i + 1) % N
            jp = (j + 1) % N
            Ux[i, j] = link_variable(psi_grid[i, j], psi_grid[ip, j])
            Uy[i, j] = link_variable(psi_grid[i, j], psi_grid[i, jp])

    # Plaquette flux
    chern_sum = 0.0
    for i in range(N):
        for j in range(N):
            ip = (i + 1) % N
            jp = (j + 1) % N
            F = Ux[i, j] @ Uy[ip, j] @ la.inv(Ux[i, jp]) @ la.inv(Uy[i, j])
            # trace of log of F / i gives the non-Abelian flux
            # For the 1st Chern number, we want (1/2πi) log det F
            det_F = la.det(F)
            flux = np.angle(det_F)  # principal branch of log det
            chern_sum += flux

    c1 = chern_sum / (2 * np.pi)
    return c1


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Path B'' session 5a: non-Abelian Berry on srs photon bundle")
    print("=" * 70)

    verts, lat_vecs = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat_vecs)
    edges = canonical_edges_primitive(bonds)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    edge_lookup = build_edge_lookup(edges)

    print(f"\n  {len(cycles)} cycles, {len(edges)} edges, {len(verts)} vertices")

    # Sanity check: eigenvector computation at a few k
    print("\n--- Eigenvector sanity check ---")
    k_test = np.array([0.12, 0.34, 0.21])
    psi = photon_eigenvectors(k_test, edges, cycles, edge_lookup)
    print(f"  at k = {k_test}")
    print(f"  psi shape: {psi.shape}")
    overlap = psi.conj().T @ psi
    print(f"  psi† psi (should be I_2):\n{np.round(overlap, 4)}")
    # Check that psi is in ker d†
    d = incidence_matrix_primitive(k_test, edges, 4)
    div = d.conj().T @ psi
    print(f"  max |d† · psi| = {np.max(np.abs(div)):.3e}")

    # Link variable sanity check
    print("\n--- Link variable sanity check ---")
    k1 = np.array([0.12, 0.34, 0.21])
    k2 = np.array([0.13, 0.34, 0.21])  # close to k1
    psi1 = photon_eigenvectors(k1, edges, cycles, edge_lookup)
    psi2 = photon_eigenvectors(k2, edges, cycles, edge_lookup)
    U = link_variable(psi1, psi2)
    print(f"  U = \n{np.round(U, 4)}")
    print(f"  det(U) = {la.det(U):.4f}  (should have |det|=1)")
    print(f"  |det(U)| = {abs(la.det(U)):.6f}")
    U_hermit = U @ U.conj().T
    print(f"  U U† (should be I_2):\n{np.round(U_hermit, 4)}")

    # 2D slice Chern number scan
    print("\n--- 2D (k_x, k_y) slice Chern numbers ---")
    N = 16
    print(f"  grid: {N}×{N}")
    print(f"  k_z scan: -0.45 to 0.45 in steps of 0.1")
    print()
    print(f"  {'k_z':>8s} | {'c_1 (integer if topological)':>30s}")
    for k_z in np.arange(-0.45, 0.46, 0.1):
        c1 = slice_chern_number(k_z, N, edges, cycles, edge_lookup)
        print(f"  {k_z:>8.3f} | {c1:>30.6f}")

    print("\n--- N=24 grid for finer resolution at a few k_z ---")
    N = 24
    for k_z in [0.0, 0.25, -0.25, 0.1, 0.33]:
        c1 = slice_chern_number(k_z, N, edges, cycles, edge_lookup)
        print(f"  N=24  k_z={k_z:+.3f}  c_1 = {c1:.6f}")


if __name__ == "__main__":
    main()
