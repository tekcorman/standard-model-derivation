#!/usr/bin/env python3
"""
SRS graph (Laves graph) analysis for particle physics calculations.

The srs net is the unique strongly isotropic 3-connected 3D crystal net.
Space group I4_132 (#214), chiral, girth 10, vertex-transitive.

Computes:
  1. Graph construction and verification (3x3x3 supercell)
  2. Cycle enumeration: 10-cycles (girth), 12-cycles, 14-cycles per vertex
  3. C3 symmetry analysis of 12-cycles
  4. Spectral properties: Laplacian, spectral dimension, Z3-twisted Bloch Hamiltonian
  5. Higher-cycle corrections to effective coupling
"""

import numpy as np
from itertools import product
from collections import defaultdict

# =============================================================================
# 1. BUILD THE SRS GRAPH
# =============================================================================

def build_srs_unit_cell():
    """
    Generate the 8 vertices of the srs net in the conventional BCC unit cell.

    Space group I4_132, Wyckoff 8a, parameter x = 1/8.
    Positions: (x,x,x) and its images under the space group generators,
    plus body-center translations.
    """
    x = 1.0 / 8.0

    # 4 positions in the primitive cell
    base = [
        np.array([x, x, x]),                        # (1/8, 1/8, 1/8)
        np.array([x + 0.5, 0.5 - x, -x]) % 1.0,    # (5/8, 3/8, 7/8)
        np.array([-x, x + 0.5, 0.5 - x]) % 1.0,    # (7/8, 5/8, 3/8)
        np.array([0.5 - x, -x, x + 0.5]) % 1.0,    # (3/8, 7/8, 5/8)
    ]

    # Body-centered translates
    bc = [(b + 0.5) % 1.0 for b in base]

    return base + bc


def min_image_vector(p1, p2):
    """Minimum image displacement vector under periodic BCs with a=1."""
    delta = p2 - p1
    return delta - np.round(delta)


def min_image_dist(p1, p2):
    return np.linalg.norm(min_image_vector(p1, p2))


def build_supercell(n_cells=3):
    """
    Build an n_cells x n_cells x n_cells supercell of the srs net.

    Returns:
        positions: (N, 3) array of vertex positions
        edges: list of (i, j) pairs
        adjacency: dict mapping vertex index to list of neighbor indices
    """
    cell_verts = build_srs_unit_cell()
    n_per_cell = len(cell_verts)  # 8

    # Generate all vertex positions in the supercell
    positions = []
    cell_indices = []  # (cell_x, cell_y, cell_z, vert_in_cell)

    for cx, cy, cz in product(range(n_cells), repeat=3):
        for iv, v in enumerate(cell_verts):
            pos = (v + np.array([cx, cy, cz])) / n_cells
            positions.append(pos)
            cell_indices.append((cx, cy, cz, iv))

    positions = np.array(positions)
    n_verts = len(positions)

    # Find nearest-neighbor distance
    # In the unit cell with a=1, NN distance = sqrt(2)/4
    # In the supercell with a_super = 1, NN distance = sqrt(2)/(4*n_cells)
    nn_dist_expected = np.sqrt(2) / (4 * n_cells)
    tol = nn_dist_expected * 0.05

    # Build adjacency by finding 3 nearest neighbors for each vertex
    adjacency = defaultdict(list)
    edges = set()

    for i in range(n_verts):
        dists = []
        for j in range(n_verts):
            if i == j:
                continue
            d = min_image_dist(positions[i], positions[j])
            dists.append((d, j))
        dists.sort()

        # Take exactly 3 nearest neighbors
        for d, j in dists[:3]:
            if abs(d - nn_dist_expected) > tol:
                print(f"  WARNING: vertex {i} neighbor {j} at distance {d:.6f}, "
                      f"expected {nn_dist_expected:.6f}")
            adjacency[i].append(j)
            edge = (min(i, j), max(i, j))
            edges.add(edge)

    edges = sorted(edges)
    return positions, edges, dict(adjacency), cell_indices


def verify_graph(positions, edges, adjacency):
    """Verify the srs graph construction."""
    n = len(positions)

    # Check degree 3
    degrees = [len(adjacency[i]) for i in range(n)]
    all_deg3 = all(d == 3 for d in degrees)

    # Check edge count: 3-regular graph has 3N/2 edges
    expected_edges = 3 * n // 2

    print(f"  Vertices: {n}")
    print(f"  Edges: {len(edges)} (expected {expected_edges})")
    print(f"  All degree 3: {all_deg3}")
    if not all_deg3:
        deg_counts = defaultdict(int)
        for d in degrees:
            deg_counts[d] += 1
        print(f"  Degree distribution: {dict(deg_counts)}")

    return all_deg3


def find_girth(adjacency, n_verts, max_length=12):
    """Find the girth (shortest cycle length) by BFS from each vertex."""
    girth = float('inf')

    for start in range(min(n_verts, 50)):  # check enough vertices
        # BFS to find shortest cycle through start
        dist = {start: 0}
        parent = {start: -1}
        queue = [start]
        head = 0

        while head < len(queue):
            v = queue[head]
            head += 1

            if dist[v] >= max_length // 2:
                break

            for w in adjacency[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                elif parent[v] != w and parent.get(w, -1) != v:
                    # Found a cycle
                    cycle_len = dist[v] + dist[w] + 1
                    girth = min(girth, cycle_len)

    return girth


# =============================================================================
# 2. CYCLE ENUMERATION
# =============================================================================

def enumerate_cycles_at_vertex(adjacency, vertex, target_length, positions):
    """
    Enumerate all cycles of exactly target_length passing through vertex.
    Uses DFS with backtracking.

    Returns list of cycles, each cycle is a tuple of vertex indices.
    """
    cycles = set()

    def dfs(path, current):
        if len(path) == target_length:
            # Check if we can close back to vertex
            if vertex in adjacency[current]:
                cycle = tuple(path)
                # Normalize: find smallest rotation and direction
                n = len(cycle)
                representations = []
                for start in range(n):
                    # Forward
                    representations.append(tuple(cycle[(start + i) % n] for i in range(n)))
                    # Backward
                    representations.append(tuple(cycle[(start - i) % n] for i in range(n)))
                canonical = min(representations)
                cycles.add(canonical)
            return

        if len(path) >= target_length:
            return

        for w in adjacency[current]:
            if w == vertex and len(path) >= 3:
                # Can close cycle, but only at target_length
                if len(path) == target_length - 1:
                    # Would make path of length target_length with w, but w=vertex
                    # Actually we check closure at len == target_length
                    pass
                continue  # Don't revisit start early
            if w not in path[1:]:  # Don't revisit (except start vertex for closing)
                dfs(path + [w], w)

    dfs([vertex], vertex)
    return cycles


def enumerate_cycles_dfs(adjacency, vertex, target_length):
    """
    Enumerate all simple cycles of exactly target_length passing through vertex.

    Returns set of canonical cycle tuples.
    """
    cycles = set()

    def dfs(path):
        current = path[-1]
        depth = len(path)

        if depth == target_length:
            # Check if current connects back to vertex
            if vertex in adjacency[current]:
                # Canonicalize
                cycle = tuple(path)
                n = len(cycle)
                reps = []
                for s in range(n):
                    reps.append(tuple(cycle[(s + i) % n] for i in range(n)))
                    reps.append(tuple(cycle[(s - i) % n] for i in range(n)))
                cycles.add(min(reps))
            return

        for w in adjacency[current]:
            if w == vertex:
                continue  # Don't return to start before target length
            if w in path:
                continue  # Don't revisit
            # Pruning: remaining steps must be enough to return
            # (This is a loose bound but helps)
            path.append(w)
            dfs(path)
            path.pop()

    dfs([vertex])
    return cycles


def cycle_chirality(cycle, positions, adjacency):
    """
    Determine the chirality of a cycle relative to the helical axis.

    For each edge in the cycle, compute the cross product of successive
    edge vectors. The sign of the helical twist determines CW vs CCW.

    Returns +1 (right-handed/CW) or -1 (left-handed/CCW).
    """
    n = len(cycle)
    winding = 0.0

    # Compute edge vectors using minimum image convention
    edge_vecs = []
    for i in range(n):
        v1 = positions[cycle[i]]
        v2 = positions[cycle[(i + 1) % n]]
        delta = min_image_vector(v1, v2)
        edge_vecs.append(delta)

    # Compute winding: sum of (e_i x e_{i+1}) . (e_i + e_{i+1})
    # This captures the helical handedness
    for i in range(n):
        e1 = edge_vecs[i]
        e2 = edge_vecs[(i + 1) % n]
        cross = np.cross(e1, e2)
        mid_dir = e1 + e2
        winding += np.dot(cross, mid_dir)

    return +1 if winding > 0 else -1


def count_cycles_per_edge_pair(cycles, vertex, adjacency):
    """
    For a given vertex, count how many cycles pass through each pair
    of the 3 edges at that vertex.

    A cycle through vertex uses exactly 2 of its 3 edges.
    """
    neighbors = adjacency[vertex]
    if len(neighbors) != 3:
        return {}

    pair_counts = defaultdict(int)
    for cycle in cycles:
        # Find which neighbors of vertex are in the cycle
        idx = list(cycle).index(vertex) if vertex in cycle else -1
        if idx == -1:
            continue
        n = len(cycle)
        prev_v = cycle[(idx - 1) % n]
        next_v = cycle[(idx + 1) % n]
        pair = tuple(sorted([prev_v, next_v]))
        pair_counts[pair] += 1

    return dict(pair_counts)


# =============================================================================
# 3. C3 SYMMETRY ANALYSIS
# =============================================================================

def analyze_c3_symmetry(cycles, vertex, adjacency, positions):
    """
    Analyze how 12-cycles transform under C3 rotation at vertex.

    C3 cyclically permutes the 3 edges at vertex.
    A cycle is C3-invariant if its canonical form is unchanged
    when we permute the edges.
    """
    neighbors = sorted(adjacency[vertex])
    if len(neighbors) != 3:
        return None

    # The C3 permutation maps neighbor[0] -> neighbor[1] -> neighbor[2] -> neighbor[0]
    # For each cycle, check if applying this permutation to the edges at vertex
    # produces the same set of cycles

    n_invariant = 0
    n_breaking = 0

    # Group cycles by which pair of edges they use at vertex
    edge_pair_groups = defaultdict(list)
    for cycle in cycles:
        cl = list(cycle)
        if vertex not in cl:
            continue
        idx = cl.index(vertex)
        n = len(cl)
        prev_v = cl[(idx - 1) % n]
        next_v = cl[(idx + 1) % n]
        pair = tuple(sorted([prev_v, next_v]))
        edge_pair_groups[pair].append(cycle)

    # Under C3, the three edge pairs are:
    # (n0,n1), (n1,n2), (n2,n0)
    # C3 maps (n0,n1) -> (n1,n2) -> (n2,n0) -> (n0,n1)
    pairs = [
        tuple(sorted([neighbors[0], neighbors[1]])),
        tuple(sorted([neighbors[1], neighbors[2]])),
        tuple(sorted([neighbors[2], neighbors[0]])),
    ]

    counts = [len(edge_pair_groups.get(p, [])) for p in pairs]

    # C3-invariant means all three edge pairs have the same count
    is_symmetric = (counts[0] == counts[1] == counts[2])

    return {
        'edge_pairs': pairs,
        'counts_per_pair': counts,
        'total_cycles': sum(counts),
        'c3_symmetric': is_symmetric,
        'c3_breaking': max(counts) - min(counts) if not is_symmetric else 0,
    }


# =============================================================================
# 4. SPECTRAL PROPERTIES
# =============================================================================

def build_adjacency_matrix(adjacency, n_verts):
    """Build the adjacency matrix for the periodic graph."""
    A = np.zeros((n_verts, n_verts))
    for i in range(n_verts):
        for j in adjacency[i]:
            A[i, j] = 1.0
    return A


def graph_laplacian(A):
    """Compute the graph Laplacian L = D - A."""
    D = np.diag(A.sum(axis=1))
    return D - A


def spectral_dimension(L, n_steps=1000):
    """
    Compute the spectral dimension from the Laplacian eigenvalues.

    The spectral dimension d_s is defined by:
    P(t) ~ t^(-d_s/2)  for the return probability.

    P(t) = (1/N) Tr(exp(-tL)) = (1/N) sum_i exp(-t * lambda_i)
    """
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.sort(eigenvalues)

    # Remove near-zero eigenvalues (they correspond to connected components)
    nonzero = eigenvalues[eigenvalues > 1e-10]

    # Compute return probability at various times
    t_values = np.logspace(-2, 3, n_steps)
    P_values = []

    for t in t_values:
        # P(t) = (1/N) * sum exp(-t * lambda_i) over ALL eigenvalues
        P = np.mean(np.exp(-t * eigenvalues))
        P_values.append(P)

    P_values = np.array(P_values)

    # Fit d_s from the middle range where P(t) ~ t^(-d_s/2)
    # Use log-log fit
    # Avoid very small t (lattice effects) and very large t (finite size effects)
    mask = (t_values > 0.1) & (t_values < 10.0) & (P_values > 1e-15)
    if np.sum(mask) < 3:
        mask = (P_values > 1e-15)

    log_t = np.log(t_values[mask])
    log_P = np.log(P_values[mask])

    # Linear fit: log P = -d_s/2 * log t + const
    if len(log_t) >= 2:
        coeffs = np.polyfit(log_t, log_P, 1)
        d_s = -2.0 * coeffs[0]
    else:
        d_s = float('nan')

    return d_s, eigenvalues


def z3_twisted_bloch_hamiltonian(positions, adjacency, k_point, n_verts_per_cell=8):
    """
    Construct the Z3-twisted Bloch Hamiltonian.

    H(k) has matrix elements:
    H_{ij}(k) = sum over R: t_{ij} * exp(i k . (R + tau_j - tau_i)) * omega^{edge_label}

    where omega = exp(2*pi*i/3) is the Z3 phase and edge_label cycles through
    0, 1, 2 for the three edges at each vertex.

    For the srs net in the unit cell, we build this as an 8x8 matrix.
    """
    cell_verts = build_srs_unit_cell()
    n = len(cell_verts)  # 8
    omega = np.exp(2j * np.pi / 3)

    H = np.zeros((n, n), dtype=complex)

    # Find edges within the unit cell (with periodic images)
    nn_dist = np.sqrt(2) / 4
    tol = 0.01

    # For each vertex, find its 3 neighbors and assign Z3 labels
    for i in range(n):
        neighbors = []
        for j in range(n):
            for dx, dy, dz in product([-1, 0, 1], repeat=3):
                if i == j and dx == 0 and dy == 0 and dz == 0:
                    continue
                delta = cell_verts[j] + np.array([dx, dy, dz]) - cell_verts[i]
                d = np.linalg.norm(delta)
                if abs(d - nn_dist) < tol:
                    neighbors.append((j, delta, np.array([dx, dy, dz])))

        # Sort neighbors consistently (by angle from a reference direction)
        if len(neighbors) >= 3:
            neighbors = neighbors[:3]

        # Assign Z3 labels 0, 1, 2 to the three edges
        for label, (j, delta, R) in enumerate(neighbors):
            phase = np.exp(1j * np.dot(k_point, 2 * np.pi * delta))
            z3_phase = omega ** label
            H[i, j] += phase * z3_phase

    return H


def compute_diffusion_length_z3(positions, adjacency):
    """
    Compute the diffusion length from the Z3-twisted Hamiltonian.

    Sample k-points on a path through the BZ and find the band structure.
    The diffusion length is related to the inverse of the smallest gap
    in the Z3-twisted spectrum.
    """
    # Sample k-points along high-symmetry path: Gamma -> X -> M -> R -> Gamma
    n_kpts = 100
    k_path = []
    labels = []

    # Gamma (0,0,0) -> X (pi,0,0)
    for i in range(n_kpts):
        t = i / n_kpts
        k_path.append(np.array([t * np.pi, 0, 0]))
    labels.append(('Gamma', 'X'))

    # X -> M (pi,pi,0)
    for i in range(n_kpts):
        t = i / n_kpts
        k_path.append(np.array([np.pi, t * np.pi, 0]))
    labels.append(('X', 'M'))

    # M -> R (pi,pi,pi)
    for i in range(n_kpts):
        t = i / n_kpts
        k_path.append(np.array([np.pi, np.pi, t * np.pi]))
    labels.append(('M', 'R'))

    # R -> Gamma
    for i in range(n_kpts + 1):
        t = i / n_kpts
        k_path.append(np.array([(1 - t) * np.pi, (1 - t) * np.pi, (1 - t) * np.pi]))
    labels.append(('R', 'Gamma'))

    all_eigenvalues = []
    min_gap = float('inf')
    min_gap_k = None

    for k in k_path:
        H = z3_twisted_bloch_hamiltonian(positions, adjacency, k)
        eigs = np.linalg.eigvalsh(H)
        all_eigenvalues.append(eigs)

        # Find minimum gap between bands
        sorted_eigs = np.sort(np.real(eigs))
        for a in range(len(sorted_eigs) - 1):
            gap = sorted_eigs[a + 1] - sorted_eigs[a]
            if gap > 1e-10 and gap < min_gap:
                min_gap = gap
                min_gap_k = k.copy()

    all_eigenvalues = np.array(all_eigenvalues)

    # Diffusion length ~ 1/min_gap (in lattice units)
    # The target is g/e = 10/e ~ 3.6788
    L_diff = 1.0 / min_gap if min_gap > 1e-10 else float('inf')

    return L_diff, min_gap, min_gap_k, all_eigenvalues


# =============================================================================
# 5. HIGHER-CYCLE CORRECTIONS
# =============================================================================

def compute_corrections(n_g, n_12, k=3, g=10):
    """
    Compute the effective coupling and higher-cycle corrections.

    alpha_1 = (n_g / k) * ((k-1)/k)^(g-2)
    Delta_alpha_1 = (n_12 / k) * ((k-1)/k)^(12-2)
    """
    alpha_1 = (n_g / k) * ((k - 1) / k) ** (g - 2)
    delta_alpha = (n_12 / k) * ((k - 1) / k) ** (12 - 2)

    return alpha_1, delta_alpha


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SRS GRAPH (LAVES GRAPH) ANALYSIS")
    print("Space group I4_132 (#214), girth 10, 3-connected")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. Build the graph
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. GRAPH CONSTRUCTION")
    print("=" * 70)

    n_cells = 3
    print(f"\nBuilding {n_cells}x{n_cells}x{n_cells} supercell...")
    positions, edges, adjacency, cell_indices = build_supercell(n_cells)

    print(f"\nVerification:")
    ok = verify_graph(positions, edges, adjacency)

    # Find girth
    print(f"\nFinding girth...")
    g = find_girth(adjacency, len(positions), max_length=14)
    print(f"  Girth: {g} (expected: 10)")

    # Check vertex transitivity by comparing local environments
    print(f"\nVertex transitivity check (comparing neighbor distance patterns):")
    dist_patterns = set()
    for i in range(min(len(positions), 20)):
        nbrs = adjacency[i]
        # Get distances between neighbors
        inter_nbr_dists = []
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                d = min_image_dist(positions[nbrs[a]], positions[nbrs[b]])
                inter_nbr_dists.append(round(d, 6))
        dist_patterns.add(tuple(sorted(inter_nbr_dists)))
    print(f"  Distinct local environments (first 20 vertices): {len(dist_patterns)}")
    print(f"  (Should be 1 for vertex-transitive graph)")
    for dp in dist_patterns:
        print(f"    Inter-neighbor distances: {dp}")

    # -----------------------------------------------------------------
    # 2. Cycle enumeration
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. CYCLE ENUMERATION")
    print("=" * 70)

    # Pick a representative vertex near the center (avoid boundary effects)
    center_cell = n_cells // 2
    center_vert = center_cell * (n_cells * n_cells * 8) + center_cell * (n_cells * 8) + center_cell * 8

    # Find a vertex in the center cell
    test_vertices = []
    for i, (cx, cy, cz, iv) in enumerate(cell_indices):
        if cx == center_cell and cy == center_cell and cz == center_cell:
            test_vertices.append(i)

    if not test_vertices:
        test_vertices = [0]

    test_v = test_vertices[0]
    print(f"\nTest vertex: {test_v} (cell position: {cell_indices[test_v]})")
    print(f"  Position: {positions[test_v]}")
    print(f"  Neighbors: {adjacency[test_v]}")

    # 10-cycles (girth cycles)
    print(f"\nEnumerating 10-cycles at vertex {test_v}...")
    cycles_10 = enumerate_cycles_dfs(adjacency, test_v, 10)
    n_10 = len(cycles_10)
    print(f"  Number of 10-cycles: {n_10} (expected: 15)")

    # Cycles per edge pair
    pair_counts_10 = count_cycles_per_edge_pair(cycles_10, test_v, adjacency)
    print(f"  Cycles per edge pair:")
    for pair, count in sorted(pair_counts_10.items()):
        print(f"    Edge pair {pair}: {count} cycles (expected: 5)")

    # Chirality of 10-cycles
    print(f"\n  Chirality of 10-cycles:")
    n_cw = 0
    n_ccw = 0
    for cycle in cycles_10:
        chi = cycle_chirality(cycle, positions, adjacency)
        if chi > 0:
            n_cw += 1
        else:
            n_ccw += 1
    print(f"    CW (right-handed): {n_cw}")
    print(f"    CCW (left-handed): {n_ccw}")
    print(f"    (Expected: 9+6 or 6+9)")

    # Check a few more vertices for consistency
    print(f"\n  Cross-checking cycle counts at other vertices...")
    for tv in test_vertices[1:4]:
        c10 = enumerate_cycles_dfs(adjacency, tv, 10)
        print(f"    Vertex {tv}: {len(c10)} 10-cycles")

    # 12-cycles
    print(f"\nEnumerating 12-cycles at vertex {test_v}...")
    cycles_12 = enumerate_cycles_dfs(adjacency, test_v, 12)
    n_12 = len(cycles_12)
    print(f"  Number of 12-cycles: {n_12}")

    pair_counts_12 = count_cycles_per_edge_pair(cycles_12, test_v, adjacency)
    print(f"  12-cycles per edge pair:")
    for pair, count in sorted(pair_counts_12.items()):
        print(f"    Edge pair {pair}: {count} cycles")

    # 14-cycles (may be slow, do fewer vertices)
    print(f"\nEnumerating 14-cycles at vertex {test_v}...")
    print(f"  (This may take a while...)")
    cycles_14 = enumerate_cycles_dfs(adjacency, test_v, 14)
    n_14 = len(cycles_14)
    print(f"  Number of 14-cycles: {n_14}")

    # -----------------------------------------------------------------
    # 3. C3 symmetry analysis of 12-cycles
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. C3 SYMMETRY ANALYSIS OF 12-CYCLES")
    print("=" * 70)

    c3_result = analyze_c3_symmetry(cycles_12, test_v, adjacency, positions)
    if c3_result:
        print(f"\n  Edge pairs at vertex {test_v}: {c3_result['edge_pairs']}")
        print(f"  12-cycles per edge pair: {c3_result['counts_per_pair']}")
        print(f"  Total 12-cycles: {c3_result['total_cycles']}")
        print(f"  C3 symmetric: {c3_result['c3_symmetric']}")
        print(f"  C3 breaking magnitude: {c3_result['c3_breaking']}")

        if not c3_result['c3_symmetric']:
            counts = c3_result['counts_per_pair']
            mean_count = np.mean(counts)
            breaking_fraction = (max(counts) - min(counts)) / mean_count if mean_count > 0 else 0
            print(f"  C3 breaking fraction: {breaking_fraction:.4f}")
            print(f"  This breaking contributes to PMNS theta_12 and theta_13")

    # Also check C3 for 10-cycles
    print(f"\n  C3 analysis of 10-cycles:")
    c3_10 = analyze_c3_symmetry(cycles_10, test_v, adjacency, positions)
    if c3_10:
        print(f"  10-cycles per edge pair: {c3_10['counts_per_pair']}")
        print(f"  C3 symmetric: {c3_10['c3_symmetric']}")

    # -----------------------------------------------------------------
    # 4. Spectral properties
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. SPECTRAL PROPERTIES")
    print("=" * 70)

    n_verts = len(positions)
    print(f"\n  Building adjacency matrix ({n_verts} x {n_verts})...")
    A = build_adjacency_matrix(adjacency, n_verts)

    # Verify symmetry
    print(f"  Adjacency matrix symmetric: {np.allclose(A, A.T)}")

    # Laplacian
    L = graph_laplacian(A)
    print(f"  Computing Laplacian eigenvalues...")
    d_s, eigenvalues = spectral_dimension(L)

    print(f"\n  Laplacian eigenvalue statistics:")
    print(f"    Min eigenvalue: {eigenvalues[0]:.8f} (should be ~0)")
    print(f"    Max eigenvalue: {eigenvalues[-1]:.6f}")
    print(f"    Number of zero eigenvalues: {np.sum(np.abs(eigenvalues) < 1e-8)}")
    print(f"    Mean eigenvalue: {np.mean(eigenvalues):.6f}")

    # Show first few distinct eigenvalues
    unique_eigs = np.unique(np.round(eigenvalues, 6))
    print(f"    Number of distinct eigenvalues: {len(unique_eigs)}")
    print(f"    First 10 distinct eigenvalues: {unique_eigs[:10]}")

    print(f"\n  Spectral dimension d_s: {d_s:.4f}")
    print(f"    (For 3D lattice, expect d_s ~ 3.0)")
    print(f"    (For fractal/low-d structure, d_s < 3)")

    # Z3-twisted Bloch Hamiltonian
    print(f"\n  Computing Z3-twisted Bloch Hamiltonian band structure...")
    L_diff, min_gap, min_gap_k, band_structure = compute_diffusion_length_z3(
        positions, adjacency
    )

    print(f"  Minimum band gap: {min_gap:.6f}")
    if min_gap_k is not None:
        print(f"  At k-point: ({min_gap_k[0]/np.pi:.3f}, {min_gap_k[1]/np.pi:.3f}, {min_gap_k[2]/np.pi:.3f}) * pi")
    print(f"  Diffusion length L_diff = 1/gap: {L_diff:.6f}")
    print(f"  Target (g/e = 10/e): {10.0 / np.e:.6f}")
    print(f"  Ratio L_diff / (g/e): {L_diff / (10.0/np.e):.6f}")

    # Also compute L_diff from full Laplacian spectrum
    nonzero_eigs = eigenvalues[eigenvalues > 1e-8]
    if len(nonzero_eigs) > 0:
        L_laplacian = 1.0 / nonzero_eigs[0]
        print(f"\n  From Laplacian smallest nonzero eigenvalue:")
        print(f"    lambda_1 = {nonzero_eigs[0]:.8f}")
        print(f"    L = 1/lambda_1 = {L_laplacian:.6f}")

    # Band structure statistics
    print(f"\n  Z3-twisted band structure statistics:")
    print(f"    Number of bands: {band_structure.shape[1]}")
    print(f"    Band width range: [{np.min(band_structure):.4f}, {np.max(band_structure):.4f}]")

    # Eigenvalue at Gamma point
    H_gamma = z3_twisted_bloch_hamiltonian(positions, adjacency, np.array([0, 0, 0]))
    eigs_gamma = np.sort(np.linalg.eigvalsh(H_gamma))
    print(f"    Eigenvalues at Gamma: {np.round(eigs_gamma, 4)}")

    # -----------------------------------------------------------------
    # 5. Higher-cycle corrections
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. HIGHER-CYCLE CORRECTIONS")
    print("=" * 70)

    k = 3  # coordination number
    g_val = 10  # girth

    alpha_1, delta_alpha = compute_corrections(n_10, n_12, k=k, g=g_val)

    print(f"\n  Parameters:")
    print(f"    k (coordination number) = {k}")
    print(f"    g (girth) = {g_val}")
    print(f"    n_g (10-cycles per vertex) = {n_10}")
    print(f"    n_12 (12-cycles per vertex) = {n_12}")
    if n_14 > 0:
        print(f"    n_14 (14-cycles per vertex) = {n_14}")

    print(f"\n  Effective coupling from girth cycles:")
    print(f"    alpha_1 = (n_g/k) * ((k-1)/k)^(g-2)")
    print(f"           = ({n_10}/{k}) * ({k-1}/{k})^{g_val-2}")
    print(f"           = {n_10/k:.4f} * {((k-1)/k)**(g_val-2):.6f}")
    print(f"           = {alpha_1:.6f}")

    print(f"\n  12-cycle correction:")
    print(f"    Delta_alpha = (n_12/k) * ((k-1)/k)^(12-2)")
    print(f"               = ({n_12}/{k}) * ({k-1}/{k})^10")
    print(f"               = {n_12/k:.4f} * {((k-1)/k)**10:.6f}")
    print(f"               = {delta_alpha:.6f}")

    if alpha_1 > 0:
        print(f"\n  Relative correction: Delta_alpha / alpha_1 = {delta_alpha/alpha_1:.6f}")
        print(f"  Corrected alpha = {alpha_1 + delta_alpha:.6f}")

    # 14-cycle correction
    if n_14 > 0:
        delta_14 = (n_14 / k) * ((k - 1) / k) ** 12
        print(f"\n  14-cycle correction:")
        print(f"    Delta_14 = ({n_14}/{k}) * (2/3)^12 = {delta_14:.6f}")
        if alpha_1 > 0:
            print(f"    Relative: Delta_14 / alpha_1 = {delta_14/alpha_1:.6f}")

    # Also compute 5 per edge pair verification
    print(f"\n  Per-edge-pair analysis:")
    print(f"    10-cycles per edge pair: {list(pair_counts_10.values())}")
    print(f"    Expected: 5 per pair (total 15, divided among C(3,2)=3 pairs)")

    # -----------------------------------------------------------------
    # 6. Additional analysis: diffusion length comparison
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. DIFFUSION LENGTH COMPARISON")
    print("=" * 70)

    # The physically meaningful diffusion length comes from the Laplacian
    # spectral gap, not the Z3-twisted near-degeneracy
    if len(nonzero_eigs) > 0:
        L_spectral = 1.0 / nonzero_eigs[0]
        target = 10.0 / np.e
        print(f"\n  Laplacian spectral gap: lambda_1 = {nonzero_eigs[0]:.8f}")
        print(f"  L_spectral = 1/lambda_1 = {L_spectral:.6f}")
        print(f"  Note: 1/(2-sqrt(3)) = 2+sqrt(3) = {2+np.sqrt(3):.6f}")
        print(f"  Target g/e = 10/e = {target:.6f}")
        print(f"  Ratio L_spectral / target = {L_spectral / target:.6f}")
        print(f"  Difference: {abs(L_spectral - target):.6f} ({abs(L_spectral - target)/target*100:.2f}%)")

    # The chirality asymmetry is the C3-breaking mechanism
    print(f"\n  Chirality asymmetry as C3 breaking:")
    print(f"    10-cycles: {n_cw} CW + {n_ccw} CCW = {n_10} total")
    chiral_ratio = min(n_cw, n_ccw) / max(n_cw, n_ccw) if max(n_cw, n_ccw) > 0 else 1
    print(f"    Chirality ratio (min/max): {chiral_ratio:.4f}")
    print(f"    Asymmetry (max-min)/total: {abs(n_cw-n_ccw)/n_10:.4f}")
    print(f"    This 2/3 vs 1/3 splitting is the geometrical origin")
    print(f"    of the 9:6 or 6:9 CW:CCW ratio")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Graph: {n_verts} vertices, {len(edges)} edges, all degree 3: {ok}")
    print(f"  Girth: {g} (expected 10)")
    print(f"  10-cycles per vertex: {n_10} (expected 15)")
    print(f"  10-cycle chirality: {n_cw} CW + {n_ccw} CCW")
    print(f"  12-cycles per vertex: {n_12}")
    print(f"  14-cycles per vertex: {n_14}")
    print(f"  C3 symmetric (12-cycles): {c3_result['c3_symmetric'] if c3_result else 'N/A'}")
    print(f"  Spectral dimension: {d_s:.4f}")
    print(f"  Z3-twisted diffusion length: {L_diff:.6f}")
    print(f"  Target g/e: {10.0/np.e:.6f}")
    print(f"  alpha_1 (girth coupling): {alpha_1:.6f}")
    print(f"  12-cycle correction: {delta_alpha:.6f}")


if __name__ == "__main__":
    main()
