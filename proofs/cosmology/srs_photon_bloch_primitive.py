#!/usr/bin/env python3
"""
Path B'' session 2: srs photon Bloch H in the bcc PRIMITIVE cell.

Session 1 built d(k) in the 8-vertex conventional cubic cell. That cell
is an index-2 supercell of the bcc primitive cell, so conventional
k-points contain FOLDED primitive spectra (each conventional 8-band
spectrum = union of two primitive 4-band spectra).

This script rebuilds everything in the 4-vertex bcc primitive cell so
that:
  - Γ, H, P, N are *unfolded* high-symmetry points
  - The scalar vertex Laplacian is 4×4 instead of 8×8
  - The vertex-edge incidence matrix d(k) is 6×4 instead of 12×8
  - The 3-fold degeneracy at Γ (if real) is directly visible
  - The P-point spectrum is the UNFOLDED primitive P spectrum,
    which should connect to h via the Ihara–Bass relation

High-symmetry points in primitive bcc reduced coordinates (using the
dual basis b_i where b₁ = 2π/a·(0,1,1), b₂ = 2π/a·(1,0,1), b₃ =
2π/a·(1,1,0)):

    Γ = (0, 0, 0)
    H = (−½, ½, ½)   [equivalent to +2π/a·(1,0,0) in Cartesian]
    P = (¼, ¼, ¼)    [equivalent to +2π/a·(½,½,½) in Cartesian]
    N = (0, 0, ½)    [equivalent to +2π/a·(½,½,0) in Cartesian]
"""

import os
import sys
import numpy as np
from numpy import linalg as la
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# =============================================================================
# 1. PRIMITIVE UNIT CELL
# =============================================================================

def build_primitive_unit_cell():
    """
    srs primitive cell (bcc lattice, 4 vertices at Wyckoff 8a with x=1/8).

    Vertices given in Cartesian coordinates (in units of a, the
    conventional cubic lattice constant).

    Returns:
        verts: (4, 3) Cartesian positions
        lat_vecs: (3, 3) primitive lattice vectors [a1; a2; a3]
    """
    verts = np.array([
        [1/8, 1/8, 1/8],
        [3/8, 7/8, 5/8],
        [7/8, 5/8, 3/8],
        [5/8, 3/8, 7/8],
    ])
    lat_vecs = np.array([
        [-1/2, 1/2, 1/2],
        [1/2, -1/2, 1/2],
        [1/2, 1/2, -1/2],
    ])
    return verts, lat_vecs


def reciprocal_lattice(lat_vecs):
    """
    Compute the primitive reciprocal lattice vectors b_i satisfying
    b_i · a_j = 2π δ_ij.

    (We omit the 2π factor and store pure dual basis; Bloch phase uses
    exp(2πi k_red · n) where n is the integer cell displacement.)
    """
    a1, a2, a3 = lat_vecs
    V = np.dot(a1, np.cross(a2, a3))
    b1 = np.cross(a2, a3) / V
    b2 = np.cross(a3, a1) / V
    b3 = np.cross(a1, a2) / V
    return np.array([b1, b2, b3])


def find_primitive_connectivity(verts, lat_vecs):
    """
    For each vertex in the primitive cell, find its 3 nearest neighbors
    (including in adjacent primitive cells).

    Returns: list of (source_idx, target_idx, cell_vector_primitive_coords, dr)
    where cell_vector_primitive_coords is (n1, n2, n3) — integer multiples
    of the primitive lattice vectors.
    """
    n_verts = len(verts)

    # Empirically find NN distance
    all_dists = []
    for i in range(n_verts):
        for j in range(n_verts):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                if i == j and (n1, n2, n3) == (0, 0, 0):
                    continue
                disp = n1 * lat_vecs[0] + n2 * lat_vecs[1] + n3 * lat_vecs[2]
                rj = verts[j] + disp
                dist = la.norm(rj - verts[i])
                if dist < 0.5:
                    all_dists.append(dist)
    nn_dist = np.min(all_dists)
    print(f"  NN distance in primitive cell: {nn_dist:.6f} "
          f"(expected √2/4 = {np.sqrt(2)/4:.6f})")
    tol = 0.05 * nn_dist

    bonds = []
    for i in range(n_verts):
        neighbors = []
        for j in range(n_verts):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                if i == j and (n1, n2, n3) == (0, 0, 0):
                    continue
                disp = n1 * lat_vecs[0] + n2 * lat_vecs[1] + n3 * lat_vecs[2]
                rj = verts[j] + disp
                dr = rj - verts[i]
                dist = la.norm(dr)
                if abs(dist - nn_dist) < tol:
                    neighbors.append((j, (n1, n2, n3), dist, dr))
        neighbors.sort(key=lambda x: x[2])
        assert len(neighbors) >= 3, f"v{i} has only {len(neighbors)} NN"
        for j, cell, dist, dr in neighbors[:3]:
            bonds.append((i, j, cell, dr))
    return bonds


def bloch_hamiltonian_primitive(k_red, bonds, n_verts):
    """
    Scalar Bloch Hamiltonian in primitive reduced coordinates.

    k_red is in the dual basis (b1, b2, b3), and the phase factor is
    exp(2πi k_red · n) where n = (n1, n2, n3) is the primitive cell
    displacement.
    """
    H = np.zeros((n_verts, n_verts), dtype=complex)
    for (src, tgt, cell, dr) in bonds:
        phase = np.exp(1j * 2 * np.pi * np.dot(k_red, cell))
        H[tgt, src] += phase
    return H


def canonical_edges_primitive(bonds):
    """
    Deduplicate bond list into undirected edges in the primitive cell.
    """
    seen = set()
    edges = []
    for (i, j, cell, dr) in bonds:
        neg_cell = tuple(-c for c in cell)
        key1 = (i, j, cell)
        key2 = (j, i, neg_cell)
        if key1 in seen or key2 in seen:
            continue
        if i < j:
            canon = (i, j, cell)
        elif i > j:
            canon = (j, i, neg_cell)
        else:
            canon = (i, j, cell) if cell >= neg_cell else (j, i, neg_cell)
        seen.add(canon)
        edges.append((len(edges), canon[0], canon[1], canon[2]))
    return edges


def incidence_matrix_primitive(k_red, edges, n_verts):
    """
    d(k)[e, v] with convention
      d[e, v_s] = +1
      d[e, v_t] = -exp(-i·2π·k_red·R_e)
    so that d†d = deg·I − A.
    """
    n_edges = len(edges)
    d = np.zeros((n_edges, n_verts), dtype=complex)
    for (e_idx, v_s, v_t, cell) in edges:
        phase = np.exp(-1j * 2 * np.pi * np.dot(k_red, cell))
        d[e_idx, v_s] += 1.0
        d[e_idx, v_t] += -phase
    return d


# =============================================================================
# 2. HIGH-SYMMETRY POINTS IN PRIMITIVE REDUCED COORDINATES
# =============================================================================

HIGH_SYM_POINTS = {
    "Γ":    np.array([0.0,  0.0,  0.0]),
    "H":    np.array([-0.5, 0.5,  0.5]),    # 2π/a·(1,0,0)
    "P":    np.array([0.25, 0.25, 0.25]),   # 2π/a·(½,½,½)
    "N":    np.array([0.0,  0.0,  0.5]),    # 2π/a·(½,½,0)
    "N_x":  np.array([0.5,  0.0,  0.0]),
    "N_y":  np.array([0.0,  0.5,  0.0]),
}


def cartesian_from_reduced(k_red, recip_vecs):
    """k_cart = k_red · B  where rows of B are primitive reciprocal vectors."""
    return k_red @ recip_vecs


# =============================================================================
# 3. VERIFICATION AND SPECTRUM
# =============================================================================

def verify_laplacian(edges, bonds, n_verts, k_red, label):
    d = incidence_matrix_primitive(k_red, edges, n_verts)
    dagd = d.conj().T @ d
    A = bloch_hamiltonian_primitive(k_red, bonds, n_verts)
    L = 3 * np.eye(n_verts, dtype=complex) - A
    err = np.max(np.abs(dagd - L))
    print(f"  {label}: max |d†d - (3I - A)| = {err:.3e}")
    return err < 1e-10


def scalar_spectrum(k_red, bonds, n_verts):
    """Return sorted real eigenvalues of the scalar Bloch Hamiltonian A(k)."""
    A = bloch_hamiltonian_primitive(k_red, bonds, n_verts)
    A = (A + A.conj().T) / 2  # hermitize (should already be Hermitian)
    eigvals = la.eigvalsh(A)
    return np.sort(eigvals)


def dd_dagger_spectrum(k_red, edges, n_verts):
    """Return sorted real eigenvalues of d(k) d(k)† on C^1."""
    d = incidence_matrix_primitive(k_red, edges, n_verts)
    L1 = d @ d.conj().T
    L1 = (L1 + L1.conj().T) / 2
    eigvals = la.eigvalsh(L1)
    return np.sort(eigvals)


def ker_dT_dim(k_red, edges, n_verts, tol=1e-9):
    """Dimension of the transverse subspace ker d†."""
    d = incidence_matrix_primitive(k_red, edges, n_verts)
    s = la.svd(d, compute_uv=False)
    rank_d = int(np.sum(s > tol))
    return d.shape[0] - rank_d, rank_d


# =============================================================================
# NON-BACKTRACKING (HASHIMOTO) WALK OPERATOR B(k)
# =============================================================================

def nb_walk_operator(k_red, bonds):
    """
    Bloch form of the non-backtracking walk operator B(k) on directed
    edges of the srs primitive cell.

    Derivation of the Bloch phase: let |e, R⟩ denote "directed edge e
    of the primitive cell, with source at Bloch site R." In Bloch form,
        |e, k⟩ = Σ_R exp(2πi k·R) |e, R⟩.
    The position-space NB kernel acts as
        B |e, R⟩ = Σ_{f non-backtracking continues e} |f, R + e.cell⟩
    (since after taking edge e from cell R, we land in cell R + e.cell,
     and any continuation f starts there).
    Fourier-transforming and factoring out the overall phase:
        B(k)[f, e] = [e.target = f.source] · [f ≠ rev(e)] · exp(-2πi k·e.cell)
    """
    n_directed = len(bonds)
    B = np.zeros((n_directed, n_directed), dtype=complex)
    for e_idx, (e_src, e_tgt, e_cell, _) in enumerate(bonds):
        for f_idx, (f_src, f_tgt, f_cell, _) in enumerate(bonds):
            if f_src != e_tgt:
                continue  # f must start where e ends
            # Non-backtracking: f is not the reverse of e
            rev_cell = tuple(-c for c in e_cell)
            if (f_src == e_tgt and f_tgt == e_src and f_cell == rev_cell):
                continue
            phase = np.exp(-1j * 2 * np.pi * np.dot(k_red, np.array(e_cell)))
            B[f_idx, e_idx] += phase
    return B


def nb_walk_spectrum(k_red, bonds):
    """Return sorted complex eigenvalues of B(k)."""
    B = nb_walk_operator(k_red, bonds)
    eigs = la.eigvals(B)
    # Sort by |eig| descending, then by arg
    idx = np.argsort(-np.abs(eigs) + 1j * np.angle(eigs) * 1e-10)
    return eigs[idx]


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Path B'' session 2: srs photon Bloch H in the PRIMITIVE cell")
    print("=" * 70)

    print("\n--- 1. Build primitive unit cell ---")
    verts, lat_vecs = build_primitive_unit_cell()
    recip = reciprocal_lattice(lat_vecs)
    n_verts = len(verts)
    print(f"  n_verts (primitive) = {n_verts}")
    print(f"  lattice vectors:")
    for i, a in enumerate(lat_vecs):
        print(f"    a{i+1} = {a}")
    print(f"  reciprocal vectors (without 2π):")
    for i, b in enumerate(recip):
        print(f"    b{i+1} = {b}")

    print("\n--- 2. Connectivity ---")
    bonds = find_primitive_connectivity(verts, lat_vecs)
    print(f"  n_bonds (directed) = {len(bonds)}")
    edges = canonical_edges_primitive(bonds)
    print(f"  n_edges (undirected) = {len(edges)}")
    assert len(edges) == 6, f"Expected 6 primitive edges, got {len(edges)}"

    print("\n--- 3. Verify d†d = 3I - A at high-symmetry points ---")
    all_ok = True
    for label, k in HIGH_SYM_POINTS.items():
        ok = verify_laplacian(edges, bonds, n_verts, k, label)
        all_ok = all_ok and ok
    print(f"  d†d identity: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        return

    print("\n--- 4. Scalar spectrum A(k) at high-symmetry points ---")
    for label, k in HIGH_SYM_POINTS.items():
        spec = scalar_spectrum(k, bonds, n_verts)
        spec_fmt = "  ".join(f"{x: .4f}" for x in spec)
        print(f"  {label:4s} A-spec: [{spec_fmt}]")

    print("\n--- 5. d(k) d(k)† spectrum (longitudinal sector) ---")
    for label, k in HIGH_SYM_POINTS.items():
        spec = dd_dagger_spectrum(k, edges, n_verts)
        null_dim, rank = ker_dT_dim(k, edges, n_verts)
        spec_fmt = "  ".join(f"{x: .4f}" for x in spec)
        print(f"  {label:4s}  rank={rank}  ker_dim={null_dim}")
        print(f"        d d† spec: [{spec_fmt}]")

    print("\n--- 6. Framework connection to h ---")
    h = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
    print(f"  h = (√3+i√5)/2")
    print(f"  |h|² = {abs(h)**2:.6f}   (expected 2 = k−1, Ramanujan)")
    print(f"  2·Re(h) = √3 = {2*h.real:.6f}")
    print(f"  arg(h) = {np.angle(h, deg=True):.4f}°")
    print(f"  sin(arg h) = {np.sin(np.angle(h)):.6f}")

    print("\n  Scalar spectrum at P should contain ±√3 (via Ihara–Bass):")
    A_at_P = bloch_hamiltonian_primitive(HIGH_SYM_POINTS["P"], bonds, n_verts)
    eigs_P = np.sort(la.eigvalsh((A_at_P + A_at_P.conj().T)/2))
    print(f"    A(P) eigenvalues = {eigs_P}")
    near_sqrt3 = [e for e in eigs_P if abs(abs(e) - np.sqrt(3)) < 1e-3]
    if near_sqrt3:
        print(f"    MATCH: {len(near_sqrt3)} eigenvalue(s) at ±√3")
    else:
        print(f"    no ±√3 match — need to examine")

    print("\n  Scalar spectrum at Γ: expect 3 (singlet) + -1 (triplet)")
    eigs_G = scalar_spectrum(HIGH_SYM_POINTS["Γ"], bonds, n_verts)
    print(f"    A(Γ) eigenvalues = {eigs_G}")

    print("\n--- 7. Symmetry interpretation ---")
    print("  A 4-vertex A(Γ) on a 3-regular graph has eigenvalues {deg, ...}")
    print("  where 'deg' = 3 is a singlet (constant eigenvector) and the")
    print("  other 3 eigenvalues transform under the site-permutation rep.")
    print("  For srs primitive (4 sites permuted by the point group 432")
    print("  restricted to stabilizer of a site), the site rep decomposes")
    print("  into irreps that the photon Bloch H inherits.")

    print("\n--- 8. Non-backtracking walk operator B(k) ---")
    print("  Framework claim: h = (√3+i√5)/2 is the leading eigenvalue")
    print("  of B(k=P). Verifying directly.")
    print()
    for label in ["Γ", "H", "P", "N"]:
        k = HIGH_SYM_POINTS[label]
        B = nb_walk_operator(k, bonds)
        eigs = nb_walk_spectrum(k, bonds)
        print(f"  k = {label}:")
        print(f"    B({label}) shape: {B.shape}")
        print(f"    eigenvalues of B({label}) (sorted by |·|):")
        for i, ev in enumerate(eigs):
            print(f"      μ_{i:2d} = {ev.real:+.4f} {ev.imag:+.4f}i  "
                  f"|μ| = {abs(ev):.4f}  arg = {np.angle(ev, deg=True):+.2f}°")

    print("\n  Check: is h = (√3+i√5)/2 = ({:.4f} + {:.4f}i) among B(P) eigenvalues?"
          .format(np.sqrt(3)/2, np.sqrt(5)/2))
    eigs_P = nb_walk_spectrum(HIGH_SYM_POINTS["P"], bonds)
    h_target = (np.sqrt(3) + 1j*np.sqrt(5))/2
    diffs = [abs(ev - h_target) for ev in eigs_P]
    min_diff = min(diffs)
    if min_diff < 1e-6:
        idx = diffs.index(min_diff)
        print(f"    YES — eigenvalue {idx} at distance {min_diff:.2e}: {eigs_P[idx]}")
    else:
        print(f"    NO — closest eigenvalue at distance {min_diff:.4f}")

    print("\n" + "=" * 70)
    print("  Session 2 step 1 complete: primitive-cell rebuild + NB walk check.")
    print("=" * 70)


if __name__ == "__main__":
    main()
