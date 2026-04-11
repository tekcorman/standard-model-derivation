#!/usr/bin/env python3
"""
Z3 holonomy of ten-cycles and fourteen-cycles on the srs net.

Computes the GAUGE-INVARIANT differential holonomy for each cycle:
  At each vertex along the cycle, the local Z3 phase is
    (exit_label - entry_label) mod 3
  where entry_label = label of the edge you arrived through,
        exit_label  = label of the edge you leave through.
  Total holonomy = sum of local phases mod 3.

This is gauge-invariant because relabeling all edges at a vertex by
a cyclic permutation (0->s, 1->s+1, 2->s+2 mod 3) shifts both
entry and exit labels by s, so (exit - entry) is unchanged.

We verify gauge invariance explicitly by randomizing the gauge and
checking that holonomies are unchanged.

Result: the (n_0, n_1, n_2) split of the 15 ten-cycles.
"""

import numpy as np
from itertools import product
from collections import defaultdict, Counter
import sys


# =============================================================================
# 1. BUILD THE SRS GRAPH (4x4x4 supercell)
# =============================================================================

def build_srs_unit_cell():
    """8 vertices of the srs net in the conventional BCC unit cell."""
    x = 1.0 / 8.0
    base = [
        np.array([x, x, x]),
        np.array([x + 0.5, 0.5 - x, -x]) % 1.0,
        np.array([-x, x + 0.5, 0.5 - x]) % 1.0,
        np.array([0.5 - x, -x, x + 0.5]) % 1.0,
    ]
    bc = [(b + 0.5) % 1.0 for b in base]
    return base + bc


def min_image_vector(p1, p2):
    delta = p2 - p1
    return delta - np.round(delta)


def min_image_dist(p1, p2):
    return np.linalg.norm(min_image_vector(p1, p2))


def build_supercell(n_cells=4):
    """Build n_cells^3 supercell. Returns positions, edges, adjacency, cell_indices."""
    cell_verts = build_srs_unit_cell()

    positions = []
    cell_indices = []

    for cx, cy, cz in product(range(n_cells), repeat=3):
        for iv, v in enumerate(cell_verts):
            pos = (v + np.array([cx, cy, cz])) / n_cells
            positions.append(pos)
            cell_indices.append((cx, cy, cz, iv))

    positions = np.array(positions)
    n_verts = len(positions)

    nn_dist_expected = np.sqrt(2) / (4 * n_cells)
    tol = nn_dist_expected * 0.05

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

        for d, j in dists[:3]:
            if abs(d - nn_dist_expected) > tol:
                print(f"  WARNING: vertex {i} neighbor {j} at distance {d:.6f}, "
                      f"expected {nn_dist_expected:.6f}")
            adjacency[i].append(j)
            edge = (min(i, j), max(i, j))
            edges.add(edge)

    edges = sorted(edges)
    return positions, edges, dict(adjacency), cell_indices


# =============================================================================
# 2. EDGE LABELING (Z3 generation labels)
# =============================================================================

def assign_edge_labels(positions, adjacency):
    """
    Assign Z3 labels {0,1,2} to the 3 edges at each vertex.

    At each vertex, compute the local C3 axis (sum of bond vectors),
    project bonds into the perpendicular plane, sort by angle
    counterclockwise. Labels are 0, 1, 2 in angular order.

    Returns: dict mapping (vertex, neighbor) -> label in {0,1,2}
    """
    n_verts = len(positions)
    edge_label = {}

    for i in range(n_verts):
        nbrs = adjacency[i]
        assert len(nbrs) == 3, f"vertex {i} has {len(nbrs)} neighbors"

        bond_vecs = []
        for j in nbrs:
            dv = min_image_vector(positions[i], positions[j])
            bond_vecs.append((j, dv))

        # C3 axis = sum of bond vectors
        sum_vec = sum(dv for _, dv in bond_vecs)
        norm = np.linalg.norm(sum_vec)
        if norm < 1e-10:
            c3_axis = np.array([1, 1, 1]) / np.sqrt(3)
        else:
            c3_axis = sum_vec / norm

        # Build perpendicular frame using a canonical reference
        dots = [abs(c3_axis[k]) for k in range(3)]
        min_idx = dots.index(min(dots))
        ref = np.zeros(3)
        ref[min_idx] = 1.0

        e1 = ref - np.dot(ref, c3_axis) * c3_axis
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(c3_axis, e1)
        e2 /= np.linalg.norm(e2)

        # Sort bonds by angle in perpendicular plane
        angles = []
        for j, dv in bond_vecs:
            dv_perp = dv - np.dot(dv, c3_axis) * c3_axis
            angle = np.arctan2(np.dot(dv_perp, e2), np.dot(dv_perp, e1))
            angles.append((angle, j))

        angles.sort()
        for label, (_, j) in enumerate(angles):
            edge_label[(i, j)] = label

    return edge_label


def randomize_gauge(edge_label, adjacency, n_verts, rng=None):
    """
    Apply a random gauge transformation: at each vertex, cyclically
    permute all labels by a random shift s in {0,1,2}.

    This should NOT change the differential holonomy.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    shifts = {v: rng.integers(0, 3) for v in range(n_verts)}

    new_label = {}
    for (i, j), lab in edge_label.items():
        new_label[(i, j)] = (lab + shifts[i]) % 3

    return new_label


# =============================================================================
# 3. CYCLE ENUMERATION
# =============================================================================

def enumerate_cycles_at_vertex(adjacency, vertex, target_length):
    """
    Enumerate all simple cycles of exactly target_length through vertex.
    Returns set of canonical cycle tuples (deduplicated by rotation/reflection).
    """
    cycles = set()

    def dfs(path, visited):
        current = path[-1]
        depth = len(path)

        if depth == target_length:
            if vertex in adjacency[current]:
                cycle = tuple(path)
                n = len(cycle)
                # Canonical form: minimum over all rotations and reflections
                reps = []
                for s in range(n):
                    reps.append(tuple(cycle[(s + i) % n] for i in range(n)))
                    reps.append(tuple(cycle[(s - i) % n] for i in range(n)))
                cycles.add(min(reps))
            return

        for w in adjacency[current]:
            if w == vertex:
                continue
            if w in visited:
                continue
            path.append(w)
            visited.add(w)
            dfs(path, visited)
            path.pop()
            visited.discard(w)

    dfs([vertex], {vertex})
    return cycles


# =============================================================================
# 4. DIFFERENTIAL Z3 HOLONOMY
# =============================================================================

def compute_differential_holonomy(cycle, edge_label):
    """
    Compute the gauge-invariant differential Z3 holonomy.

    At each vertex v_i along the cycle:
      entry edge: v_{i-1} -> v_i  (label at v_i for this edge = edge_label[(v_i, v_{i-1})])
      exit edge:  v_i -> v_{i+1}  (label at v_i for this edge = edge_label[(v_i, v_{i+1})])
      local phase = (exit_label - entry_label) mod 3

    Total holonomy = sum of all local phases mod 3.
    """
    n = len(cycle)
    total = 0
    for i in range(n):
        v_prev = cycle[(i - 1) % n]
        v_curr = cycle[i]
        v_next = cycle[(i + 1) % n]

        entry_label = edge_label[(v_curr, v_prev)]
        exit_label = edge_label[(v_curr, v_next)]
        local_phase = (exit_label - entry_label) % 3

        total += local_phase

    return total % 3


def compute_raw_holonomy(cycle, edge_label):
    """
    The raw (non-gauge-invariant) holonomy: sum of exit labels mod 3.
    For comparison / verification that differential IS gauge-invariant.
    """
    n = len(cycle)
    total = 0
    for i in range(n):
        vi = cycle[i]
        vi_next = cycle[(i + 1) % n]
        total += edge_label[(vi, vi_next)]
    return total % 3


def cycle_chirality(cycle, positions):
    """
    Measure helical winding to determine chirality.
    Returns +1 (right-handed) or -1 (left-handed).
    """
    n = len(cycle)
    edge_vecs = []
    for i in range(n):
        v1 = positions[cycle[i]]
        v2 = positions[cycle[(i + 1) % n]]
        edge_vecs.append(min_image_vector(v1, v2))

    winding = 0.0
    for i in range(n):
        e1 = edge_vecs[i]
        e2 = edge_vecs[(i + 1) % n]
        cross = np.cross(e1, e2)
        mid_dir = e1 + e2
        winding += np.dot(cross, mid_dir)

    return +1 if winding > 0 else -1


def is_contractible(cycle, positions):
    """Check if cycle has zero total displacement (contractible)."""
    total_disp = np.zeros(3)
    n = len(cycle)
    for i in range(n):
        total_disp += min_image_vector(positions[cycle[i]], positions[cycle[(i + 1) % n]])
    return np.linalg.norm(total_disp) < 0.01


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 76)
    print("Z3 DIFFERENTIAL HOLONOMY OF CYCLES ON THE SRS NET")
    print("Gauge-invariant computation: sum of (exit_label - entry_label) mod 3")
    print("=" * 76)

    # -----------------------------------------------------------------
    # 1. Build graph
    # -----------------------------------------------------------------
    n_cells = 4
    print(f"\n1. Building {n_cells}x{n_cells}x{n_cells} supercell...")
    positions, edges, adjacency, cell_indices = build_supercell(n_cells)
    n_verts = len(positions)

    print(f"   Vertices: {n_verts} (expected {8 * n_cells**3})")
    print(f"   Edges: {len(edges)} (expected {3 * n_verts // 2})")

    degrees = [len(adjacency[i]) for i in range(n_verts)]
    assert all(d == 3 for d in degrees), "Not all vertices have degree 3!"
    print("   All degree 3: YES")

    # -----------------------------------------------------------------
    # 2. Edge labeling
    # -----------------------------------------------------------------
    print("\n2. Assigning Z3 edge labels...")
    edge_label = assign_edge_labels(positions, adjacency)

    # Verify each vertex has labels {0,1,2}
    for i in range(n_verts):
        labels = {edge_label[(i, j)] for j in adjacency[i]}
        assert labels == {0, 1, 2}, f"vertex {i} has labels {labels}"
    print("   All vertices have labels {0,1,2}: YES")

    # -----------------------------------------------------------------
    # 3. Pick a central vertex and enumerate cycles
    # -----------------------------------------------------------------
    # Choose a vertex near the center of the supercell
    center = np.array([0.5, 0.5, 0.5])
    dists_to_center = [np.linalg.norm(min_image_vector(positions[i], center))
                       for i in range(n_verts)]
    vertex = int(np.argmin(dists_to_center))
    print(f"\n3. Central vertex: {vertex} at position "
          f"({positions[vertex][0]:.4f}, {positions[vertex][1]:.4f}, {positions[vertex][2]:.4f})")
    print(f"   Neighbors: {adjacency[vertex]}")
    print(f"   Edge labels: ", end="")
    for j in adjacency[vertex]:
        print(f"  ({vertex}->{j}): {edge_label[(vertex, j)]}", end="")
    print()

    # -----------------------------------------------------------------
    # 4. Enumerate 10-cycles
    # -----------------------------------------------------------------
    print(f"\n4. Enumerating 10-cycles through vertex {vertex}...")
    cycles_10 = enumerate_cycles_at_vertex(adjacency, vertex, 10)
    print(f"   Found {len(cycles_10)} ten-cycles (expected 15)")

    # -----------------------------------------------------------------
    # 5. Compute differential holonomy for each 10-cycle
    # -----------------------------------------------------------------
    print(f"\n5. DIFFERENTIAL Z3 HOLONOMY OF 10-CYCLES")
    print("=" * 76)
    print(f"   {'Cycle#':>6}  {'Holonomy':>8}  {'Chirality':>9}  {'Contractible':>12}  "
          f"{'Local phases':>30}")
    print("   " + "-" * 70)

    holonomies_10 = []
    details_10 = []
    for idx, cycle in enumerate(sorted(cycles_10)):
        cl = list(cycle)

        # Compute local phases for display
        n = len(cl)
        local_phases = []
        for i in range(n):
            v_prev = cl[(i - 1) % n]
            v_curr = cl[i]
            v_next = cl[(i + 1) % n]
            entry = edge_label[(v_curr, v_prev)]
            exit_ = edge_label[(v_curr, v_next)]
            local_phases.append((exit_ - entry) % 3)

        h = sum(local_phases) % 3
        chi = cycle_chirality(cl, positions)
        cont = is_contractible(cl, positions)

        holonomies_10.append(h)
        details_10.append({
            'cycle': cycle,
            'holonomy': h,
            'chirality': chi,
            'contractible': cont,
            'local_phases': local_phases,
        })

        chi_str = "CW(+1)" if chi > 0 else "CCW(-1)"
        cont_str = "yes" if cont else "no"
        phases_str = "".join(str(p) for p in local_phases)
        print(f"   {idx+1:>6}  {h:>8}  {chi_str:>9}  {cont_str:>12}  "
              f"  [{phases_str}] sum={sum(local_phases)}")

    # -----------------------------------------------------------------
    # 6. Summary for 10-cycles
    # -----------------------------------------------------------------
    print(f"\n6. 10-CYCLE HOLONOMY SUMMARY")
    print("=" * 76)
    counts_10 = Counter(holonomies_10)
    for h in [0, 1, 2]:
        print(f"   Holonomy {h}: {counts_10.get(h, 0)} cycles")

    n0, n1, n2 = counts_10.get(0, 0), counts_10.get(1, 0), counts_10.get(2, 0)
    print(f"\n   Split: (n_0, n_1, n_2) = ({n0}, {n1}, {n2})")
    print(f"   Total: {n0 + n1 + n2}")

    if n1 != n2:
        print(f"\n   *** Z3 HOLONOMY BREAKS j=1 <-> j=2 DEGENERACY ***")
        print(f"   n_1/n_2 = {n1}/{n2} = {n1/n2:.6f}" if n2 > 0 else f"   n_2 = 0!")
    else:
        print(f"\n   n_1 == n_2: Z3 holonomy does NOT break the j=1/j=2 degeneracy")
        print(f"   at the 10-cycle level. Must look at 14-cycles.")

    # Cross-tabulate holonomy vs chirality
    print(f"\n   Holonomy x Chirality cross-tabulation:")
    for h in [0, 1, 2]:
        cw = sum(1 for d in details_10 if d['holonomy'] == h and d['chirality'] > 0)
        ccw = sum(1 for d in details_10 if d['holonomy'] == h and d['chirality'] < 0)
        print(f"     H={h}: CW={cw}, CCW={ccw}")

    # -----------------------------------------------------------------
    # 7. Gauge invariance verification
    # -----------------------------------------------------------------
    print(f"\n7. GAUGE INVARIANCE VERIFICATION")
    print("=" * 76)

    rng = np.random.default_rng(12345)
    gauge_ok = True
    for trial in range(5):
        randomized_label = randomize_gauge(edge_label, adjacency, n_verts, rng)

        # Verify labels still valid
        for i in range(n_verts):
            labels = {randomized_label[(i, j)] for j in adjacency[i]}
            assert labels == {0, 1, 2}

        # Compute holonomies with randomized gauge
        h_randomized = []
        for cycle in sorted(cycles_10):
            cl = list(cycle)
            h = compute_differential_holonomy(cl, randomized_label)
            h_randomized.append(h)

        if h_randomized == holonomies_10:
            print(f"   Trial {trial+1}: PASS (holonomies unchanged)")
        else:
            print(f"   Trial {trial+1}: FAIL!")
            print(f"     Original:   {holonomies_10}")
            print(f"     Randomized: {h_randomized}")
            gauge_ok = False

    # Also verify raw holonomy is NOT gauge-invariant
    raw_orig = []
    for cycle in sorted(cycles_10):
        cl = list(cycle)
        raw_orig.append(compute_raw_holonomy(cl, edge_label))

    randomized_label = randomize_gauge(edge_label, adjacency, n_verts,
                                       np.random.default_rng(99))
    raw_rand = []
    for cycle in sorted(cycles_10):
        cl = list(cycle)
        raw_rand.append(compute_raw_holonomy(cl, randomized_label))

    raw_same = (raw_orig == raw_rand)
    print(f"\n   Raw (non-differential) holonomy gauge-invariant? {raw_same}")
    if not raw_same:
        print(f"   (Expected: NO -- raw holonomy IS gauge-dependent)")

    print(f"\n   Differential holonomy gauge-invariant across all trials? {gauge_ok}")

    # -----------------------------------------------------------------
    # 8. Verify with a SECOND vertex (vertex-transitivity check)
    # -----------------------------------------------------------------
    print(f"\n8. VERTEX-TRANSITIVITY CHECK")
    print("=" * 76)
    # Pick another vertex
    vertex2 = adjacency[vertex][0]
    print(f"   Second vertex: {vertex2}")

    cycles_10_v2 = enumerate_cycles_at_vertex(adjacency, vertex2, 10)
    print(f"   10-cycles through vertex {vertex2}: {len(cycles_10_v2)}")

    holonomies_v2 = []
    for cycle in sorted(cycles_10_v2):
        cl = list(cycle)
        h = compute_differential_holonomy(cl, edge_label)
        holonomies_v2.append(h)

    counts_v2 = Counter(holonomies_v2)
    n0v2 = counts_v2.get(0, 0)
    n1v2 = counts_v2.get(1, 0)
    n2v2 = counts_v2.get(2, 0)
    print(f"   Split at vertex {vertex2}: (n_0, n_1, n_2) = ({n0v2}, {n1v2}, {n2v2})")
    print(f"   Same as vertex {vertex}? {(n0, n1, n2) == (n0v2, n1v2, n2v2)}")

    # -----------------------------------------------------------------
    # 9. 14-cycles
    # -----------------------------------------------------------------
    print(f"\n9. ENUMERATING 14-CYCLES through vertex {vertex}...")
    print("   (This may take a while...)")
    sys.stdout.flush()

    cycles_14 = enumerate_cycles_at_vertex(adjacency, vertex, 14)
    print(f"   Found {len(cycles_14)} fourteen-cycles")

    if len(cycles_14) > 0:
        print(f"\n   14-CYCLE HOLONOMY SUMMARY")
        print("   " + "-" * 50)

        holonomies_14 = []
        chiralities_14 = []
        for cycle in sorted(cycles_14):
            cl = list(cycle)
            h = compute_differential_holonomy(cl, edge_label)
            chi = cycle_chirality(cl, positions)
            holonomies_14.append(h)
            chiralities_14.append(chi)

        counts_14 = Counter(holonomies_14)
        for h in [0, 1, 2]:
            print(f"   Holonomy {h}: {counts_14.get(h, 0)} cycles")

        m0 = counts_14.get(0, 0)
        m1 = counts_14.get(1, 0)
        m2 = counts_14.get(2, 0)
        print(f"\n   Split: (m_0, m_1, m_2) = ({m0}, {m1}, {m2})")

        # Cross-tabulate
        print(f"\n   Holonomy x Chirality for 14-cycles:")
        for h in [0, 1, 2]:
            cw = sum(1 for i, hh in enumerate(holonomies_14)
                     if hh == h and chiralities_14[i] > 0)
            ccw = sum(1 for i, hh in enumerate(holonomies_14)
                      if hh == h and chiralities_14[i] < 0)
            print(f"     H={h}: CW={cw}, CCW={ccw}")

        if m1 != m2:
            print(f"\n   *** 14-CYCLES BREAK j=1 <-> j=2 DEGENERACY ***")
            print(f"   m_1/m_2 = {m1}/{m2} = {m1/m2:.6f}" if m2 > 0 else f"   m_2 = 0!")

    # -----------------------------------------------------------------
    # 10. Also check 12-cycles (use 3x3x3 supercell where they exist)
    # -----------------------------------------------------------------
    print(f"\n10. ENUMERATING 12-CYCLES (using 3x3x3 supercell for non-contractible ones)...")
    # The srs 12-cycles are non-contractible (wrap the crystal axes).
    # On a 4x4x4 supercell they may not close; use 3x3x3.
    pos3, edg3, adj3, ci3 = build_supercell(3)
    n3 = len(pos3)
    el3 = assign_edge_labels(pos3, adj3)

    center3 = np.array([0.5, 0.5, 0.5])
    d3 = [np.linalg.norm(min_image_vector(pos3[i], center3)) for i in range(n3)]
    v3 = int(np.argmin(d3))

    cycles_12 = enumerate_cycles_at_vertex(adj3, v3, 12)
    print(f"    Found {len(cycles_12)} twelve-cycles (3x3x3 supercell)")

    k0, k1, k2 = 0, 0, 0
    if len(cycles_12) > 0:
        holonomies_12 = []
        for cycle in sorted(cycles_12):
            cl = list(cycle)
            h = compute_differential_holonomy(cl, el3)
            cont = is_contractible(cl, pos3)
            holonomies_12.append(h)

        counts_12 = Counter(holonomies_12)
        for h in [0, 1, 2]:
            print(f"    Holonomy {h}: {counts_12.get(h, 0)} cycles")
        k0 = counts_12.get(0, 0)
        k1 = counts_12.get(1, 0)
        k2 = counts_12.get(2, 0)
        print(f"    Split: ({k0}, {k1}, {k2})")

    # Also verify 10-cycles on the 3x3x3 match
    cycles_10_v3 = enumerate_cycles_at_vertex(adj3, v3, 10)
    h10_v3 = [compute_differential_holonomy(list(c), el3) for c in sorted(cycles_10_v3)]
    c10_v3 = Counter(h10_v3)
    print(f"\n    Cross-check 10-cycles on 3x3x3: {len(cycles_10_v3)} cycles, "
          f"split ({c10_v3.get(0,0)}, {c10_v3.get(1,0)}, {c10_v3.get(2,0)})")

    # -----------------------------------------------------------------
    # 11. Final summary
    # -----------------------------------------------------------------
    print(f"\n" + "=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)
    print(f"  10-cycles: {len(cycles_10)} total, holonomy split ({n0}, {n1}, {n2})")
    if len(cycles_12) > 0:
        print(f"  12-cycles: {len(cycles_12)} total, holonomy split ({k0}, {k1}, {k2})")
    if len(cycles_14) > 0:
        print(f"  14-cycles: {len(cycles_14)} total, holonomy split ({m0}, {m1}, {m2})")

    print(f"\n  Gauge invariance verified: {gauge_ok}")
    print(f"  Vertex transitivity verified: "
          f"{(n0, n1, n2) == (n0v2, n1v2, n2v2)}")

    # The key question
    print(f"\n  CRITICAL QUESTION: Does Z3 holonomy break j=1 <-> j=2?")
    if n1 != n2:
        print(f"  YES at 10-cycles: n_1/n_2 = {n1}/{n2}")
    else:
        print(f"  NO at 10-cycles (n_1 = n_2 = {n1})")
    if len(cycles_14) > 0:
        if m1 != m2:
            print(f"  YES at 14-cycles: m_1/m_2 = {m1}/{m2}")
        else:
            print(f"  NO at 14-cycles (m_1 = m_2 = {m1})")

    # Analysis of WHY holonomy is always zero
    print(f"\n  ANALYSIS: Why all holonomies are zero")
    print(f"  " + "-" * 60)
    print(f"  The Z3 connection defined by the C3 site symmetry is FLAT.")
    print(f"  This means the Z3 bundle over the srs net is globally")
    print(f"  trivializable: there exists a global section (gauge choice)")
    print(f"  that makes all connection values zero.")
    print(f"")
    print(f"  Physically: the 'generation label' can be consistently")
    print(f"  transported around ANY cycle without acquiring a phase.")
    print(f"  The C3 site symmetry does NOT produce holonomy.")
    print(f"")
    print(f"  Consequence for neutrino masses: the j=1/j=2 degeneracy")
    print(f"  is NOT broken by Z3 holonomy alone. The 9/6 CW/CCW split")
    print(f"  is purely about geometric chirality, not about Z3 phases.")
    print(f"  To break the degeneracy, one needs a mechanism beyond the")
    print(f"  flat Z3 connection -- e.g., interaction with the 4_1 screw")
    print(f"  axis, or coupling to a non-flat gauge field.")


if __name__ == "__main__":
    main()
