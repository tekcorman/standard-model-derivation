#!/usr/bin/env python3
"""
Path B'' session 7e: Berry charge of the Γ defect on srs photon bundle.

Session 7d identified that d(k) has rank 4 everywhere except at Γ,
where it drops to 3. This is a single topological defect of the
photon Hodge bundle. The claim (session 7d, A-grade) is that this
defect has Berry charge -1/k, giving θ = 2π/k.

This script computes the charge directly by integrating the Berry
curvature trace (= U(1) projection) over a small sphere around Γ.

Strategy:
1. Parameterize a small sphere S² of radius r around Γ in the 3D BZ
   by spherical coordinates (θ, φ).
2. For each point on the sphere, compute the 2-dim ker d†(k) using
   the photon_eigenvectors function.
3. Compute non-Abelian link variables between adjacent points on the
   sphere.
4. Sum the plaquette fluxes to get the total U(1) Chern number /
   Berry charge of the bundle over the sphere.
5. Verify the charge equals -1/k (or equivalently 2π/k in angular units).

Numerical: if |Chern charge over S²| = 1/k (say 1/3 for srs), this
confirms the claim and promotes θ = 2π/k to theorem grade for srs.
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
from srs_photon_hodge import build_edge_lookup
from srs_photon_berry import photon_eigenvectors, link_variable


def sphere_grid(r, N_theta, N_phi):
    """
    Parameterize S² around origin by (theta, phi) grid.
    Returns: array of shape (N_theta, N_phi, 3) with Cartesian points.
    Avoids poles by offsetting slightly.
    """
    thetas = np.linspace(0, np.pi, N_theta)[1:-1]  # avoid poles
    phis = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
    pts = np.zeros((len(thetas), N_phi, 3))
    for i, t in enumerate(thetas):
        for j, p in enumerate(phis):
            pts[i, j] = r * np.array([
                np.sin(t)*np.cos(p),
                np.sin(t)*np.sin(p),
                np.cos(t),
            ])
    return pts, thetas, phis


def photon_grid_on_sphere(sphere_pts, edges, cycles, edge_lookup):
    """Compute photon eigenvectors at each sphere point."""
    N_theta, N_phi, _ = sphere_pts.shape
    psi = np.empty((N_theta, N_phi, 6, 2), dtype=complex)
    for i in range(N_theta):
        for j in range(N_phi):
            k = sphere_pts[i, j]
            psi[i, j] = photon_eigenvectors(k, edges, cycles, edge_lookup)
    return psi


def sphere_chern(psi):
    """
    Compute the integer (U(1)) Chern number of the 2-band bundle over
    the sphere S² using the Fukui-Hatsugai-Suzuki link-variable method,
    with the trace/determinant projection for U(1).

    Treats (i, j) as a grid with i ∈ [0, N_theta), j ∈ [0, N_phi).
    The sphere has two "poles" not included in the grid (avoided), so
    the total integral is approximate. For fine grids it converges.
    """
    N_theta, N_phi, _, _ = psi.shape
    # Compute link variables
    # U_theta[i,j] = link from (i,j) to (i+1,j)
    # U_phi[i,j] = link from (i,j) to (i, (j+1) mod N_phi)
    U_theta = np.empty((N_theta-1, N_phi, 2, 2), dtype=complex)
    U_phi = np.empty((N_theta, N_phi, 2, 2), dtype=complex)
    for i in range(N_theta):
        for j in range(N_phi):
            jp = (j+1) % N_phi
            U_phi[i, j] = link_variable(psi[i, j], psi[i, jp])
    for i in range(N_theta-1):
        for j in range(N_phi):
            U_theta[i, j] = link_variable(psi[i, j], psi[i+1, j])

    # Plaquette fluxes: for each (i, j), F_{i,j} uses U_theta(i,j)·U_phi(i+1,j)·
    # U_theta(i,j+1)^{-1}·U_phi(i,j)^{-1}
    chern_sum = 0.0
    for i in range(N_theta - 1):
        for j in range(N_phi):
            jp = (j+1) % N_phi
            F = U_theta[i, j] @ U_phi[i+1, j] @ la.inv(U_theta[i, jp]) @ la.inv(U_phi[i, j])
            det_F = la.det(F)
            flux = np.angle(det_F)
            chern_sum += flux
    return chern_sum / (2 * np.pi)


def main():
    print("=" * 70)
    print("  Path B'' session 7e: Γ defect Berry charge of photon bundle")
    print("=" * 70)

    verts, lat_vecs = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat_vecs)
    edges = canonical_edges_primitive(bonds)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    edge_lookup = build_edge_lookup(edges)

    print("\nTesting Γ defect charge by sphere integration:")
    print("Claim: defect carries U(1) Chern charge = -1/k = -1/3 for srs")
    print()
    print(f"  {'radius r':>10s} | {'N_theta':>8s} | {'N_phi':>6s} | {'U(1) Chern':>12s} | {'expected':>10s}")
    print("  " + "-" * 60)

    # Run at multiple radii and grid sizes
    for r in [0.01, 0.02, 0.05]:
        for (Nt, Np) in [(16, 24), (24, 32), (32, 48)]:
            try:
                sphere_pts, _, _ = sphere_grid(r, Nt, Np)
                psi = photon_grid_on_sphere(sphere_pts, edges, cycles, edge_lookup)
                c1 = sphere_chern(psi)
                print(f"  {r:>10.4f} | {Nt:>8d} | {Np:>6d} | {c1:>12.6f} | {-1/3:>10.6f}")
            except Exception as e:
                print(f"  {r:>10.4f} | {Nt:>8d} | {Np:>6d} | ERROR: {e}")

    print()
    print("  Notes:")
    print("  - For a clean Dirac-like defect, the Chern is an integer (e.g., ±1).")
    print("  - Fractional charge -1/3 would indicate a non-standard defect structure.")
    print("  - The 'polar cap' exclusion gives some numerical noise.")


if __name__ == "__main__":
    main()
