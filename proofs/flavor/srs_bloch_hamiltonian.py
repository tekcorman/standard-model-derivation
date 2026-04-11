#!/usr/bin/env python3
"""
Z3-twisted Bloch Hamiltonian on the srs (Laves graph) crystal net.

The srs net is the unique chiral 3-connected crystal with space group I4_132,
girth 10, vertex-transitive. This script computes:

  1. Unit cell connectivity with cell displacement vectors
  2. Untwisted Bloch Hamiltonian H(k) and band structure
  3. Z3-twisted Bloch Hamiltonian H_tw(k) and band structure
  4. Twisted spectral gap and diffusion lengths
  5. Twist-angle scan L(phi) from phi=0 to pi
  6. Real-space twisted Green's function and decay length
  7. Comparison of all extracted lengths with g/e = 3.6788

Target: find a spectral quantity equal to g/e = 10/e = 3.67879...
"""

import numpy as np
from numpy import linalg as la
from itertools import product
from collections import defaultdict
import os

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 1. UNIT CELL AND CONNECTIVITY
# =============================================================================

def build_unit_cell():
    """
    SRS net conventional cubic cell: 8 vertices.
    Space group I4_132 (#214), Wyckoff 8a with x=1/8.

    The 8a position generates 4 vertices in the primitive cell, plus their
    body-centered translates (offset by (1/2, 1/2, 1/2)), giving 8 total
    in the conventional cubic cell.
    """
    base = np.array([
        [1/8, 1/8, 1/8],   # v0
        [3/8, 7/8, 5/8],   # v1
        [7/8, 5/8, 3/8],   # v2
        [5/8, 3/8, 7/8],   # v3
    ])

    # Body-centered translates
    bc = (base + 0.5) % 1.0

    verts = np.vstack([base, bc])
    return verts


def find_connectivity(verts, a=1.0):
    """
    For each vertex in the unit cell, find its 3 nearest neighbors,
    including those in neighboring cells (periodic boundary conditions).

    Returns: list of (source_idx, target_idx, cell_displacement) tuples,
    where cell_displacement is (n1, n2, n3) in Z^3.

    The NN distance in srs with a=1 is sqrt(2)/4 ~ 0.3536.
    """
    n_verts = len(verts)
    # NN distance: sqrt((1/4)^2 + (1/4)^2) = sqrt(2)/4 for adjacent vertices
    # But let's find it empirically by computing all short distances
    all_dists = []
    for i in range(n_verts):
        for j in range(n_verts):
            for n1, n2, n3 in product(range(-1, 2), repeat=3):
                if i == j and n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                disp = np.array([n1, n2, n3], dtype=float) * a
                rj = verts[j] * a + disp
                ri = verts[i] * a
                dr = rj - ri
                dist = la.norm(dr)
                if dist < 0.5 * a:
                    all_dists.append(dist)

    if all_dists:
        all_dists = np.array(all_dists)
        nn_dist = np.min(all_dists)
        print(f"  Detected NN distance: {nn_dist:.6f} (sqrt(2)/4 = {np.sqrt(2)/4:.6f})")
    else:
        nn_dist = np.sqrt(2) / 4 * a
        print(f"  Using expected NN distance: {nn_dist:.6f}")

    tol = 0.05 * nn_dist
    bonds = []

    for i in range(n_verts):
        neighbors = []
        for j in range(n_verts):
            for n1, n2, n3 in product(range(-1, 2), repeat=3):
                if i == j and n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                disp = np.array([n1, n2, n3], dtype=float) * a
                rj = verts[j] * a + disp
                ri = verts[i] * a
                dr = rj - ri
                dist = la.norm(dr)
                if abs(dist - nn_dist) < tol:
                    neighbors.append((j, (n1, n2, n3), dist, dr))

        # Sort by distance, take closest 3
        neighbors.sort(key=lambda x: x[2])
        if len(neighbors) < 3:
            print(f"WARNING: vertex {i} has only {len(neighbors)} neighbors!")
        elif len(neighbors) > 3:
            # Check if 4th neighbor is at same distance (would indicate wrong cell)
            if abs(neighbors[3][2] - neighbors[0][2]) < tol:
                print(f"WARNING: vertex {i} has {len(neighbors)} equidistant neighbors "
                      f"(expected 3), taking first 3")

        for j, cell, dist, dr in neighbors[:3]:
            bonds.append((i, j, cell, dr))

    return bonds


def verify_connectivity(bonds, n_verts):
    """Verify each vertex has degree 3."""
    degree = defaultdict(int)
    for i, j, cell, dr in bonds:
        degree[i] += 1

    print("\n=== Connectivity Verification ===")
    for i in range(n_verts):
        print(f"  v{i}: degree {degree[i]}")

    all_ok = all(degree[i] == 3 for i in range(n_verts))
    print(f"  All degree 3: {all_ok}")
    return all_ok


def print_bonds(bonds):
    """Print bond table."""
    print("\n=== Bond Table ===")
    print(f"  {'src':>3} -> {'tgt':>3}  cell_disp          distance")
    for i, j, cell, dr in bonds:
        dist = la.norm(dr)
        print(f"  v{i}  -> v{j}   ({cell[0]:+d},{cell[1]:+d},{cell[2]:+d})  "
              f"  dr=({dr[0]:+.4f},{dr[1]:+.4f},{dr[2]:+.4f})  |dr|={dist:.6f}")


# =============================================================================
# 2. UNTWISTED BLOCH HAMILTONIAN
# =============================================================================

def bloch_hamiltonian(k, bonds, n_verts):
    """
    Construct the untwisted Bloch Hamiltonian H(k).

    H_{ij}(k) = sum over bonds (i -> j, cell R): exp(i k . R)

    This is the adjacency matrix in reciprocal space.
    k is in Cartesian coordinates (units of 2pi/a).
    R is the real-space cell displacement vector.
    """
    H = np.zeros((n_verts, n_verts), dtype=complex)

    for src, tgt, cell, dr in bonds:
        # Cell displacement in real space (a=1, so cell displacement = cell vector)
        R = np.array(cell, dtype=float)
        phase = np.exp(1j * np.dot(k, R) * 2 * np.pi)
        H[tgt, src] += phase

    return H


def compute_band_structure(bonds, n_verts, n_pts=200):
    """
    Compute band structure along high-symmetry path in BZ.

    For a simple cubic lattice with a=1:
    Gamma = (0,0,0), X = (1/2,0,0), M = (1/2,1/2,0), R = (1/2,1/2,1/2)
    (in units of 2pi/a)

    We use k in reduced coordinates directly (the phase is exp(i k.R * 2pi)
    where R is in lattice units).
    """
    # High-symmetry points (fractional/reduced coordinates of the cubic BZ)
    Gamma = np.array([0, 0, 0], dtype=float)
    X = np.array([0.5, 0, 0], dtype=float)
    M = np.array([0.5, 0.5, 0], dtype=float)
    R = np.array([0.5, 0.5, 0.5], dtype=float)

    path_labels = ['Γ', 'X', 'M', 'R', 'Γ']
    path_points = [Gamma, X, M, R, Gamma]

    all_k = []
    all_E = []
    tick_positions = [0]

    cumulative = 0
    for seg in range(len(path_points) - 1):
        k_start = path_points[seg]
        k_end = path_points[seg + 1]

        for i in range(n_pts):
            t = i / n_pts
            k = k_start + t * (k_end - k_start)
            H = bloch_hamiltonian(k, bonds, n_verts)
            evals = la.eigvalsh(H)
            all_k.append(cumulative + t * la.norm(k_end - k_start))
            all_E.append(np.sort(np.real(evals)))

        cumulative += la.norm(k_end - k_start)
        tick_positions.append(cumulative)

    all_E = np.array(all_E)
    all_k = np.array(all_k)

    return all_k, all_E, tick_positions, path_labels


# =============================================================================
# 3. Z3-TWISTED BLOCH HAMILTONIAN
# =============================================================================

def assign_edge_labels(bonds, n_verts):
    """
    Assign Z3 edge labels (0, 1, 2) to the three bonds at each vertex.

    The srs graph is vertex-transitive under I4_132. The three edges at each
    vertex are related by the 3-fold screw axis. We label them by the
    crystallographic direction of the bond vector:

    Sort the three bond displacement vectors at each vertex and assign
    labels 0, 1, 2 in a consistent way inherited from the chiral structure.

    For consistency: at each vertex, sort bonds by the angle of their
    displacement vector projected onto a reference plane, measured from a
    reference direction. The chirality of the srs net ensures this gives
    a consistent global labeling.
    """
    # Group bonds by source vertex
    vertex_bonds = defaultdict(list)
    for idx, (src, tgt, cell, dr) in enumerate(bonds):
        vertex_bonds[src].append((idx, dr))

    labels = [0] * len(bonds)

    for v in range(n_verts):
        vbonds = vertex_bonds[v]
        assert len(vbonds) == 3, f"vertex {v} has {len(vbonds)} bonds"

        # Project displacement vectors and sort by angle around (1,1,1) axis
        # The 3-fold screw axis of I4_132 is along (1,1,1)
        axis = np.array([1, 1, 1]) / np.sqrt(3)

        # Reference direction perpendicular to axis
        ref = np.array([1, -1, 0]) / np.sqrt(2)
        ref2 = np.cross(axis, ref)

        angles = []
        for bond_idx, dr in vbonds:
            # Project dr perpendicular to axis
            dr_perp = dr - np.dot(dr, axis) * axis
            angle = np.arctan2(np.dot(dr_perp, ref2), np.dot(dr_perp, ref))
            angles.append((angle, bond_idx))

        # Sort by angle to get consistent labeling
        angles.sort()
        for label, (angle, bond_idx) in enumerate(angles):
            labels[bond_idx] = label

    return labels


def twisted_bloch_hamiltonian(k, bonds, edge_labels, n_verts, omega):
    """
    Construct the Z3-twisted Bloch Hamiltonian.

    H_tw_{ij}(k) = sum over bonds (i -> j, label l): omega^l * exp(i k . R)

    omega = exp(2pi i / 3) for Z3 twist, or exp(i phi) for general twist.
    """
    H = np.zeros((n_verts, n_verts), dtype=complex)

    for idx, (src, tgt, cell, dr) in enumerate(bonds):
        R = np.array(cell, dtype=float)
        phase = np.exp(1j * np.dot(k, R) * 2 * np.pi)
        label = edge_labels[idx]
        twist = omega ** label
        H[tgt, src] += twist * phase

    return H


def compute_twisted_band_structure(bonds, edge_labels, n_verts, omega, n_pts=200):
    """Compute twisted band structure along high-symmetry path."""
    Gamma = np.array([0, 0, 0], dtype=float)
    X = np.array([0.5, 0, 0], dtype=float)
    M = np.array([0.5, 0.5, 0], dtype=float)
    R = np.array([0.5, 0.5, 0.5], dtype=float)

    path_labels = ['Γ', 'X', 'M', 'R', 'Γ']
    path_points = [Gamma, X, M, R, Gamma]

    all_k = []
    all_E = []
    tick_positions = [0]

    cumulative = 0
    for seg in range(len(path_points) - 1):
        k_start = path_points[seg]
        k_end = path_points[seg + 1]

        for i in range(n_pts):
            t = i / n_pts
            k = k_start + t * (k_end - k_start)
            H = twisted_bloch_hamiltonian(k, bonds, edge_labels, n_verts, omega)
            evals = la.eigvalsh(H)
            all_k.append(cumulative + t * la.norm(k_end - k_start))
            all_E.append(np.sort(np.real(evals)))

        cumulative += la.norm(k_end - k_start)
        tick_positions.append(cumulative)

    all_E = np.array(all_E)
    all_k = np.array(all_k)

    return all_k, all_E, tick_positions, path_labels


# =============================================================================
# 4. SPECTRAL GAP AND DIFFUSION LENGTHS
# =============================================================================

def compute_spectral_properties(bonds, edge_labels, n_verts, omega, n_grid=50):
    """
    Scan the full BZ on a grid and extract spectral properties.

    Returns:
        band_min: minimum eigenvalue over BZ for each band
        band_max: maximum eigenvalue over BZ for each band
        gap: spectral gap (min of lowest band above ground state)
        bandwidth: total bandwidth
    """
    all_evals = []

    for n1, n2, n3 in product(range(n_grid), repeat=3):
        k = np.array([n1, n2, n3], dtype=float) / n_grid
        H = twisted_bloch_hamiltonian(k, bonds, edge_labels, n_verts, omega)
        evals = np.sort(np.real(la.eigvalsh(H)))
        all_evals.append(evals)

    all_evals = np.array(all_evals)

    band_min = all_evals.min(axis=0)
    band_max = all_evals.max(axis=0)

    # The "ground state" is the minimum eigenvalue overall
    global_min = all_evals[:, 0].min()

    # Spectral gap: separation between lowest band max and second band min
    # Or more precisely, the gap in the density of states
    gap_between_bands = []
    for b in range(n_verts - 1):
        gap = band_min[b + 1] - band_max[b]
        gap_between_bands.append(gap)

    return {
        'band_min': band_min,
        'band_max': band_max,
        'global_min': global_min,
        'global_max': all_evals[:, -1].max(),
        'gaps_between_bands': gap_between_bands,
        'all_evals': all_evals,
    }


def compute_diffusion_lengths(spec):
    """
    Extract various diffusion length candidates from spectral data.

    L1 = 1/Delta where Delta = spectral gap
    L2 = 1/sqrt(m*) from band curvature (effective mass)
    L3 = bandwidth-derived
    """
    results = {}

    # Spectral gap (between bands)
    for i, gap in enumerate(spec['gaps_between_bands']):
        if gap > 0.01:
            results[f'L_gap_{i}_{i+1}'] = 1.0 / gap

    # Spread of lowest band
    bw0 = spec['band_max'][0] - spec['band_min'][0]
    if bw0 > 1e-10:
        results['L_bw_band0'] = 1.0 / bw0

    # Total bandwidth
    total_bw = spec['global_max'] - spec['global_min']
    results['L_total_bw'] = 1.0 / total_bw

    # |global_min| as an energy scale
    if abs(spec['global_min']) > 1e-10:
        results['L_inv_Emin'] = 1.0 / abs(spec['global_min'])

    # Sum of reciprocal eigenvalues at Gamma (related to Green's function)
    return results


# =============================================================================
# 5. TWIST ANGLE SCAN
# =============================================================================

def scan_twist_angles(bonds, edge_labels, n_verts, n_phi=100, n_grid=30):
    """
    Scan twist phase phi from 0 to pi.
    For each phi, omega = exp(i * phi), compute spectral properties.
    """
    phis = np.linspace(0, np.pi, n_phi)
    results = []

    for phi in phis:
        omega = np.exp(1j * phi)
        spec = compute_spectral_properties(bonds, edge_labels, n_verts, omega, n_grid=n_grid)

        # Extract key quantities
        entry = {
            'phi': phi,
            'band_min': spec['band_min'].copy(),
            'band_max': spec['band_max'].copy(),
            'global_min': spec['global_min'],
            'global_max': spec['global_max'],
            'gaps': list(spec['gaps_between_bands']),
        }
        results.append(entry)

    return results


# =============================================================================
# 6. REAL-SPACE GREEN'S FUNCTION
# =============================================================================

def compute_greens_function_realspace(bonds, edge_labels, n_verts, omega,
                                      E_probe=None, n_grid=15, max_R=4):
    """
    Compute the twisted Green's function in real space:
    G_tw(R) = (1/N_k) sum_k exp(ik.R) * [E - H_tw(k)]^{-1}

    Precomputes G(k) for all k, then Fourier transforms for each R.
    """
    # Build k-grid
    k_list = []
    for n1, n2, n3 in product(range(n_grid), repeat=3):
        k_list.append(np.array([n1, n2, n3], dtype=float) / n_grid)
    k_arr = np.array(k_list)
    N_k = len(k_list)

    # Precompute all H(k) and find global min
    H_all = np.zeros((N_k, n_verts, n_verts), dtype=complex)
    emin = 1e10
    for ik, k in enumerate(k_list):
        H_all[ik] = twisted_bloch_hamiltonian(k, bonds, edge_labels, n_verts, omega)
        evals = la.eigvalsh(H_all[ik])
        emin = min(emin, np.min(np.real(evals)))

    if E_probe is None:
        E_probe = emin - 0.1

    # Precompute G(k) = (E - H(k))^{-1}
    G_all = np.zeros_like(H_all)
    eye = np.eye(n_verts)
    for ik in range(N_k):
        G_inv = E_probe * eye - H_all[ik]
        try:
            G_all[ik] = la.inv(G_inv)
        except la.LinAlgError:
            pass

    # Compute G(R) for various R vectors via Fourier transform
    R_vectors = []
    for n1, n2, n3 in product(range(-max_R, max_R + 1), repeat=3):
        R = np.array([n1, n2, n3], dtype=float)
        if la.norm(R) < 0.01:
            continue
        R_vectors.append(R)

    G_data = []
    for R in R_vectors:
        # phases[ik] = exp(2pi i k.R)
        phases = np.exp(2j * np.pi * k_arr.dot(R))
        G_R = np.einsum('k,kij->ij', phases, G_all) / N_k
        G_data.append((la.norm(R), la.norm(G_R)))

    G_data.sort(key=lambda x: x[0])

    dist_groups = defaultdict(list)
    for r, g in G_data:
        dist_groups[round(r, 4)].append(g)

    distances = sorted(dist_groups.keys())
    avg_G = [np.mean(dist_groups[d]) for d in distances]

    return distances, avg_G, E_probe


def fit_decay_length(distances, avg_G):
    """
    Fit G(r) ~ A * exp(-r / L) to extract decay length L.
    Uses log-linear fit on points where G > 0.
    """
    valid = [(d, g) for d, g in zip(distances, avg_G) if g > 1e-15 and d > 0.5]
    if len(valid) < 3:
        return None, None

    ds = np.array([v[0] for v in valid])
    gs = np.array([v[1] for v in valid])
    log_gs = np.log(gs)

    # Linear fit: log(G) = log(A) - r/L
    coeffs = np.polyfit(ds, log_gs, 1)
    slope = coeffs[0]

    if slope >= 0:
        return None, None  # Not decaying

    L = -1.0 / slope
    A = np.exp(coeffs[1])
    return L, A


# =============================================================================
# 7. UNTWISTED SPECTRAL ANALYSIS
# =============================================================================

def untwisted_gamma_eigenvalues(bonds, n_verts):
    """Eigenvalues of the adjacency matrix at k=Gamma (k=0)."""
    k0 = np.array([0, 0, 0], dtype=float)
    H = bloch_hamiltonian(k0, bonds, n_verts)
    evals = np.sort(np.real(la.eigvalsh(H)))
    return evals


# =============================================================================
# 8. LAPLACIAN ANALYSIS
# =============================================================================

def bloch_laplacian(k, bonds, n_verts):
    """L(k) = D - A(k) where D = 3*I for 3-regular graph."""
    A = bloch_hamiltonian(k, bonds, n_verts)
    return 3 * np.eye(n_verts) - A


def twisted_laplacian(k, bonds, edge_labels, n_verts, omega):
    """Twisted Laplacian: L_tw(k) = D - A_tw(k)."""
    A = twisted_bloch_hamiltonian(k, bonds, edge_labels, n_verts, omega)
    return 3 * np.eye(n_verts) - A


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Z3-TWISTED BLOCH HAMILTONIAN ON THE SRS (LAVES) NET")
    print("=" * 70)

    TARGET = 10.0 / np.e  # g/e = 3.67879...
    COMPARE_1 = 2 + np.sqrt(3)  # 3.73205...

    print(f"\nTarget: g/e = 10/e = {TARGET:.6f}")
    print(f"Compare: 2+sqrt(3) = {COMPARE_1:.6f}")

    # ----- Step 1: Build unit cell and connectivity -----
    print("\n" + "=" * 70)
    print("1. UNIT CELL AND CONNECTIVITY")
    print("=" * 70)

    verts = build_unit_cell()
    n_verts = len(verts)
    print(f"\nUnit cell: {n_verts} vertices")
    for i, v in enumerate(verts):
        print(f"  v{i} = ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})")

    bonds = find_connectivity(verts)
    verify_connectivity(bonds, n_verts)
    print_bonds(bonds)

    # ----- Step 2: Untwisted Bloch Hamiltonian -----
    print("\n" + "=" * 70)
    print("2. UNTWISTED BLOCH HAMILTONIAN")
    print("=" * 70)

    # Eigenvalues at Gamma
    evals_gamma = untwisted_gamma_eigenvalues(bonds, n_verts)
    print(f"\nAdjacency eigenvalues at Gamma (k=0): {evals_gamma}")
    print(f"  Expected for srs: should include 3 (flat band) and -1 (triply degenerate)")

    # Laplacian at Gamma
    k0 = np.array([0, 0, 0], dtype=float)
    L_gamma = bloch_laplacian(k0, bonds, n_verts)
    L_evals = np.sort(np.real(la.eigvalsh(L_gamma)))
    print(f"\nLaplacian eigenvalues at Gamma: {L_evals}")

    # Band structure
    print("\nComputing untwisted band structure...")
    all_k, all_E, ticks, labels = compute_band_structure(bonds, n_verts, n_pts=150)

    print(f"  Band ranges:")
    for b in range(n_verts):
        print(f"    Band {b}: [{all_E[:, b].min():.6f}, {all_E[:, b].max():.6f}]")

    # Untwisted spectral properties on full BZ
    omega_trivial = 1.0
    edge_labels_trivial = [0] * len(bonds)  # All labels 0, omega^0 = 1
    spec_untwisted = compute_spectral_properties(bonds, edge_labels_trivial, n_verts, omega_trivial, n_grid=30)

    print(f"\n  Full BZ scan (untwisted):")
    print(f"    Global eigenvalue range: [{spec_untwisted['global_min']:.6f}, {spec_untwisted['global_max']:.6f}]")
    print(f"    Band gaps: {[f'{g:.6f}' for g in spec_untwisted['gaps_between_bands']]}")

    # Untwisted Laplacian spectral gap
    print(f"\n  Untwisted Laplacian top eigenvalue: {3 - spec_untwisted['global_min']:.6f}")
    print(f"    (2+sqrt(3) = {COMPARE_1:.6f})")

    # Plot untwisted band structure
    fig, ax = plt.subplots(figsize=(8, 6))
    for b in range(n_verts):
        ax.plot(all_k, all_E[:, b], 'b-', linewidth=1.5)
    for t in ticks:
        ax.axvline(t, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel('E(k)')
    ax.set_title('SRS Net: Untwisted Bloch Band Structure')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'srs_untwisted_bands.png'), dpi=150)
    plt.close(fig)
    print("\n  Saved: srs_untwisted_bands.png")

    # ----- Step 3: Edge labeling and Z3 twist -----
    print("\n" + "=" * 70)
    print("3. Z3-TWISTED BLOCH HAMILTONIAN")
    print("=" * 70)

    edge_labels = assign_edge_labels(bonds, n_verts)

    print("\nEdge labels (Z3 arm index):")
    for idx, (src, tgt, cell, dr) in enumerate(bonds):
        print(f"  v{src} -> v{tgt} ({cell[0]:+d},{cell[1]:+d},{cell[2]:+d})  label={edge_labels[idx]}")

    # Verify: each vertex should have labels {0, 1, 2}
    for v in range(n_verts):
        v_labels = sorted([edge_labels[i] for i, (s, t, c, d) in enumerate(bonds) if s == v])
        print(f"  v{v} edge labels: {v_labels}  (should be [0, 1, 2])")

    # Z3 twist: omega = exp(2pi i/3)
    omega_z3 = np.exp(2j * np.pi / 3)
    print(f"\nomega = exp(2pi i/3) = {omega_z3:.6f}")

    # Twisted eigenvalues at Gamma
    H_tw_gamma = twisted_bloch_hamiltonian(k0, bonds, edge_labels, n_verts, omega_z3)
    tw_evals_gamma = np.sort(np.real(la.eigvalsh(H_tw_gamma)))
    print(f"\nTwisted adjacency eigenvalues at Gamma: {tw_evals_gamma}")

    # Twisted Laplacian at Gamma
    L_tw_gamma = twisted_laplacian(k0, bonds, edge_labels, n_verts, omega_z3)
    L_tw_evals = np.sort(np.real(la.eigvalsh(L_tw_gamma)))
    print(f"Twisted Laplacian eigenvalues at Gamma: {L_tw_evals}")

    # Twisted band structure
    print("\nComputing Z3-twisted band structure...")
    tw_k, tw_E, tw_ticks, tw_labels = compute_twisted_band_structure(
        bonds, edge_labels, n_verts, omega_z3, n_pts=150)

    print(f"  Twisted band ranges:")
    for b in range(n_verts):
        print(f"    Band {b}: [{tw_E[:, b].min():.6f}, {tw_E[:, b].max():.6f}]")

    # Full BZ spectral properties
    spec_twisted = compute_spectral_properties(bonds, edge_labels, n_verts, omega_z3, n_grid=30)

    print(f"\n  Full BZ scan (Z3-twisted):")
    print(f"    Global eigenvalue range: [{spec_twisted['global_min']:.6f}, {spec_twisted['global_max']:.6f}]")
    print(f"    Band gaps: {[f'{g:.6f}' for g in spec_twisted['gaps_between_bands']]}")

    # Plot twisted band structure
    fig, ax = plt.subplots(figsize=(8, 6))
    for b in range(n_verts):
        ax.plot(tw_k, tw_E[:, b], 'r-', linewidth=1.5)
    for t in tw_ticks:
        ax.axvline(t, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(tw_ticks)
    ax.set_xticklabels(tw_labels)
    ax.set_ylabel('E(k)')
    ax.set_title('SRS Net: Z₃-Twisted Bloch Band Structure')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'srs_z3_twisted_bands.png'), dpi=150)
    plt.close(fig)
    print("\n  Saved: srs_z3_twisted_bands.png")

    # Overlay plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for b in range(n_verts):
        ax.plot(all_k, all_E[:, b], 'b-', linewidth=1.5,
                label='Untwisted' if b == 0 else None)
        ax.plot(tw_k, tw_E[:, b], 'r--', linewidth=1.5,
                label='Z₃-twisted' if b == 0 else None)
    for t in ticks:
        ax.axvline(t, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel('E(k)')
    ax.set_title('SRS Net: Untwisted vs Z₃-Twisted Band Structure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'srs_bands_comparison.png'), dpi=150)
    plt.close(fig)
    print("  Saved: srs_bands_comparison.png")

    # ----- Step 4: Diffusion lengths -----
    print("\n" + "=" * 70)
    print("4. DIFFUSION LENGTHS FROM TWISTED HAMILTONIAN")
    print("=" * 70)

    lengths_tw = compute_diffusion_lengths(spec_twisted)
    lengths_untw = compute_diffusion_lengths(spec_untwisted)

    print("\n  Untwisted diffusion lengths:")
    for name, val in sorted(lengths_untw.items()):
        dev = (val - TARGET) / TARGET * 100
        print(f"    {name:25s} = {val:.6f}  (deviation from g/e: {dev:+.3f}%)")

    print("\n  Z3-Twisted diffusion lengths:")
    for name, val in sorted(lengths_tw.items()):
        dev = (val - TARGET) / TARGET * 100
        print(f"    {name:25s} = {val:.6f}  (deviation from g/e: {dev:+.3f}%)")

    # Additional length: from Laplacian spectral gap
    # The Laplacian smallest nonzero eigenvalue gives the Cheeger-like mixing
    print("\n  Additional spectral quantities:")

    # Untwisted adjacency bandwidth
    adj_bw = spec_untwisted['global_max'] - spec_untwisted['global_min']
    print(f"    Untwisted adjacency bandwidth: {adj_bw:.6f}")
    print(f"    1/bandwidth: {1/adj_bw:.6f}")

    # Twisted adjacency bandwidth
    tw_bw = spec_twisted['global_max'] - spec_twisted['global_min']
    print(f"    Twisted adjacency bandwidth: {tw_bw:.6f}")
    print(f"    1/bandwidth: {1/tw_bw:.6f}")

    # Ratio of twisted to untwisted bandwidth
    if adj_bw > 0:
        print(f"    Bandwidth ratio (tw/untw): {tw_bw/adj_bw:.6f}")

    # ----- Step 5: Twist angle scan -----
    print("\n" + "=" * 70)
    print("5. TWIST ANGLE SCAN")
    print("=" * 70)

    print("\nScanning twist angle phi from 0 to pi...")
    scan = scan_twist_angles(bonds, edge_labels, n_verts, n_phi=60, n_grid=15)

    phis = [s['phi'] for s in scan]
    global_mins = [s['global_min'] for s in scan]
    global_maxs = [s['global_max'] for s in scan]
    bandwidths = [s['global_max'] - s['global_min'] for s in scan]

    # Various length candidates as function of phi
    L_inv_bw = [1.0 / (s['global_max'] - s['global_min']) if (s['global_max'] - s['global_min']) > 0.01 else np.nan for s in scan]
    L_inv_Emin = [1.0 / abs(s['global_min']) if abs(s['global_min']) > 0.01 else np.nan for s in scan]

    # Find phi where various lengths match g/e
    print(f"\n  Searching for phi where L = g/e = {TARGET:.6f}...")

    for name, L_arr in [('1/bandwidth', L_inv_bw), ('1/|E_min|', L_inv_Emin)]:
        best_dev = 1e10
        best_phi = None
        best_L = None
        for i, (phi, L) in enumerate(zip(phis, L_arr)):
            if np.isnan(L):
                continue
            dev = abs(L - TARGET) / TARGET
            if dev < best_dev:
                best_dev = dev
                best_phi = phi
                best_L = L
        if best_phi is not None:
            print(f"    {name:20s}: best L = {best_L:.6f} at phi = {best_phi:.4f} "
                  f"({np.degrees(best_phi):.2f} deg), deviation = {best_dev*100:.4f}%")

    # Also search for phi where spectral gap matches
    gap_lengths = []
    for s in scan:
        gaps = s['gaps']
        max_gap = max(gaps) if gaps else 0
        gap_lengths.append(1.0 / max_gap if max_gap > 0.01 else np.nan)

    best_dev = 1e10
    best_phi = None
    best_L = None
    for i, (phi, L) in enumerate(zip(phis, gap_lengths)):
        if np.isnan(L):
            continue
        dev = abs(L - TARGET) / TARGET
        if dev < best_dev:
            best_dev = dev
            best_phi = phi
            best_L = L
    if best_phi is not None:
        print(f"    {'1/max_gap':20s}: best L = {best_L:.6f} at phi = {best_phi:.4f} "
              f"({np.degrees(best_phi):.2f} deg), deviation = {best_dev*100:.4f}%")

    # Scan for spectral quantities that might match g/e
    print(f"\n  Checking ALL spectral quantities at Z3 twist (phi=2pi/3)...")
    phi_z3 = 2 * np.pi / 3
    omega_z3_check = np.exp(1j * phi_z3)
    spec_z3 = compute_spectral_properties(bonds, edge_labels, n_verts, omega_z3_check, n_grid=30)

    quantities = {}
    quantities['E_min'] = spec_z3['global_min']
    quantities['E_max'] = spec_z3['global_max']
    quantities['bandwidth'] = spec_z3['global_max'] - spec_z3['global_min']
    quantities['|E_min|'] = abs(spec_z3['global_min'])
    quantities['|E_max|'] = abs(spec_z3['global_max'])
    for i, g in enumerate(spec_z3['gaps_between_bands']):
        quantities[f'gap_{i}_{i+1}'] = g
    for i in range(n_verts):
        quantities[f'band_{i}_width'] = spec_z3['band_max'][i] - spec_z3['band_min'][i]
        quantities[f'band_{i}_min'] = spec_z3['band_min'][i]
        quantities[f'band_{i}_max'] = spec_z3['band_max'][i]

    # Also try ratios and reciprocals
    derived = {}
    for name, val in quantities.items():
        if abs(val) > 1e-10:
            derived[f'1/{name}'] = 1.0 / val
            derived[f'1/|{name}|'] = 1.0 / abs(val)
    quantities.update(derived)

    print(f"\n  Quantities closest to g/e = {TARGET:.6f}:")
    ranked = sorted(quantities.items(), key=lambda x: abs(x[1] - TARGET))
    for name, val in ranked[:15]:
        dev = (val - TARGET) / TARGET * 100
        print(f"    {name:35s} = {val:+.6f}  (dev: {dev:+.4f}%)")

    print(f"\n  Quantities closest to 2+sqrt(3) = {COMPARE_1:.6f}:")
    ranked2 = sorted(quantities.items(), key=lambda x: abs(x[1] - COMPARE_1))
    for name, val in ranked2[:10]:
        dev = (val - COMPARE_1) / COMPARE_1 * 100
        print(f"    {name:35s} = {val:+.6f}  (dev: {dev:+.4f}%)")

    # Plot twist scan
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(np.degrees(phis), global_mins, 'b-', label='E_min')
    ax.plot(np.degrees(phis), global_maxs, 'r-', label='E_max')
    ax.axvline(120, color='green', linestyle='--', label='phi=2pi/3 (Z3)')
    ax.set_xlabel('Twist angle phi (degrees)')
    ax.set_ylabel('Energy')
    ax.set_title('Global Eigenvalue Extrema vs Twist Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(np.degrees(phis), bandwidths, 'k-')
    ax.axvline(120, color='green', linestyle='--', label='Z3')
    ax.axhline(1.0/TARGET, color='orange', linestyle=':', label=f'1/(g/e)={1/TARGET:.4f}')
    ax.set_xlabel('Twist angle phi (degrees)')
    ax.set_ylabel('Bandwidth')
    ax.set_title('Total Bandwidth vs Twist Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(np.degrees(phis), L_inv_bw, 'b-', label='1/bandwidth')
    ax.plot(np.degrees(phis), L_inv_Emin, 'r-', label='1/|E_min|')
    ax.axvline(120, color='green', linestyle='--', label='Z3')
    ax.axhline(TARGET, color='orange', linestyle=':', label=f'g/e={TARGET:.4f}')
    ax.axhline(COMPARE_1, color='purple', linestyle=':', label=f'2+sqrt(3)={COMPARE_1:.4f}')
    ax.set_xlabel('Twist angle phi (degrees)')
    ax.set_ylabel('Length')
    ax.set_title('Diffusion Lengths vs Twist Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(np.degrees(phis), gap_lengths, 'g-', label='1/max_gap')
    ax.axvline(120, color='green', linestyle='--', label='Z3')
    ax.axhline(TARGET, color='orange', linestyle=':', label=f'g/e={TARGET:.4f}')
    ax.set_xlabel('Twist angle phi (degrees)')
    ax.set_ylabel('Length')
    ax.set_title('Gap-Derived Length vs Twist Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'srs_twist_scan.png'), dpi=150)
    plt.close(fig)
    print("\n  Saved: srs_twist_scan.png")

    # ----- Step 6: Real-space Green's function -----
    print("\n" + "=" * 70)
    print("6. REAL-SPACE TWISTED GREEN'S FUNCTION")
    print("=" * 70)

    print("\nComputing real-space Green's function (Z3 twist)...")
    print("  (This may take a minute with n_grid=20, max_R=3)")

    distances, avg_G, E_probe = compute_greens_function_realspace(
        bonds, edge_labels, n_verts, omega_z3, n_grid=20, max_R=3)

    print(f"\n  Probe energy: E = {E_probe:.6f}")
    print(f"  G(r) vs distance:")
    for d, g in zip(distances[:20], avg_G[:20]):
        print(f"    |R| = {d:.4f}  |G| = {g:.6e}")

    L_decay, A_fit = fit_decay_length(distances, avg_G)
    if L_decay is not None:
        dev = (L_decay - TARGET) / TARGET * 100
        print(f"\n  Fitted decay length: L = {L_decay:.6f}  (deviation from g/e: {dev:+.3f}%)")
        print(f"  Fitted amplitude: A = {A_fit:.6e}")
    else:
        print("\n  Could not fit decay length (data may not show exponential decay)")

    # Try different probe energies
    print("\n  Scanning probe energies...")
    for E_offset in [0.0, -0.01, -0.05, -0.1, -0.5, -1.0]:
        E_try = spec_twisted['global_min'] + E_offset
        try:
            dists, gs, _ = compute_greens_function_realspace(
                bonds, edge_labels, n_verts, omega_z3, E_probe=E_try, n_grid=12, max_R=3)
            L, A = fit_decay_length(dists, gs)
            if L is not None and L > 0 and L < 100:
                dev = (L - TARGET) / TARGET * 100
                print(f"    E = {E_try:+.4f}: L = {L:.6f} (dev from g/e: {dev:+.3f}%)")
            else:
                print(f"    E = {E_try:+.4f}: no valid decay")
        except Exception as e:
            print(f"    E = {E_try:+.4f}: error: {e}")

    # Plot Green's function
    if len(distances) > 0 and len(avg_G) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        valid_mask = np.array(avg_G) > 1e-15
        if np.any(valid_mask):
            ds_valid = np.array(distances)[valid_mask]
            gs_valid = np.array(avg_G)[valid_mask]
            ax.semilogy(ds_valid, gs_valid, 'bo-', markersize=4)
            if L_decay is not None:
                r_fit = np.linspace(ds_valid.min(), ds_valid.max(), 100)
                g_fit = A_fit * np.exp(-r_fit / L_decay)
                ax.semilogy(r_fit, g_fit, 'r--',
                           label=f'Fit: L = {L_decay:.4f}')
                ax.legend()
            ax.set_xlabel('|R| (lattice units)')
            ax.set_ylabel('|G(R)| (Frobenius norm)')
            ax.set_title(f'Real-Space Green\'s Function (Z₃ twist, E={E_probe:.3f})')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, 'srs_greens_function.png'), dpi=150)
            plt.close(fig)
            print("\n  Saved: srs_greens_function.png")

    # ----- Step 7: Summary comparison -----
    print("\n" + "=" * 70)
    print("7. SUMMARY: ALL EXTRACTED LENGTHS vs TARGETS")
    print("=" * 70)

    print(f"\n  Target: g/e = 10/e = {TARGET:.6f}")
    print(f"  Reference: 2+sqrt(3) = {COMPARE_1:.6f}")
    print(f"  Difference: {COMPARE_1 - TARGET:.6f} ({(COMPARE_1-TARGET)/TARGET*100:.3f}%)")

    all_lengths = {}

    # From untwisted spectrum
    for name, val in lengths_untw.items():
        all_lengths[f'untw_{name}'] = val

    # From Z3-twisted spectrum
    for name, val in lengths_tw.items():
        all_lengths[f'z3tw_{name}'] = val

    # Laplacian spectral gap
    L_evals_full = []
    for n1, n2, n3 in product(range(15), repeat=3):
        k = np.array([n1, n2, n3], dtype=float) / 15
        L = bloch_laplacian(k, bonds, n_verts)
        evals = np.sort(np.real(la.eigvalsh(L)))
        L_evals_full.append(evals)
    L_evals_full = np.array(L_evals_full)
    lap_max = L_evals_full.max()
    lap_min_nonzero = L_evals_full[L_evals_full > 0.01].min() if np.any(L_evals_full > 0.01) else None

    all_lengths['untw_lap_max'] = lap_max
    all_lengths['untw_1/lap_max'] = 1.0 / lap_max
    if lap_min_nonzero:
        all_lengths['untw_lap_min_nonzero'] = lap_min_nonzero
        all_lengths['untw_1/lap_min_nonzero'] = 1.0 / lap_min_nonzero

    # Green's function decay
    if L_decay is not None:
        all_lengths['z3tw_greens_decay'] = L_decay

    # Direct spectral quantities that are close
    for name, val in ranked[:5]:
        all_lengths[f'z3tw_spectral_{name}'] = val

    print(f"\n  {'Quantity':50s}  {'Value':>10s}  {'dev(g/e)':>10s}  {'dev(2+rt3)':>10s}")
    print("  " + "-" * 85)

    for name, val in sorted(all_lengths.items(), key=lambda x: abs(x[1] - TARGET)):
        dev_ge = (val - TARGET) / TARGET * 100
        dev_rt3 = (val - COMPARE_1) / COMPARE_1 * 100
        marker = " <-- CLOSE" if abs(dev_ge) < 2 else ""
        print(f"  {name:50s}  {val:10.6f}  {dev_ge:+9.4f}%  {dev_rt3:+9.4f}%{marker}")

    # Check some simple algebraic combinations
    print(f"\n  Algebraic combinations:")
    combos = {
        'g/e = 10/e': TARGET,
        '2+sqrt(3)': COMPARE_1,
        'sqrt(10/e)': np.sqrt(TARGET),
        '(10/e)^2': TARGET**2,
        'log(10)': np.log(10),
        '3/sqrt(e)': 3/np.sqrt(np.e),
        'pi/sqrt(e)': np.pi/np.sqrt(np.e),
        'sqrt(3)*e/pi': np.sqrt(3)*np.e/np.pi,
        '10/(e*sqrt(3))': 10/(np.e*np.sqrt(3)),
    }
    for name, val in combos.items():
        print(f"    {name:25s} = {val:.6f}")


if __name__ == '__main__':
    main()
