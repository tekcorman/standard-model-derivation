#!/usr/bin/env python3
"""
a separate private derivation by the author port #8 — verification probe.

QUESTION: Do the 15 girth-10 NB cycles through an srs vertex split into
3 CW + 2 CCW per edge pair (totaling 9 + 6 = 15 per vertex), as a separate private derivation by the author claims (and as our framework's `gut_baryogenesis.py` /
`baryogenesis_calc.py` postulates with n_cw=6, n_ccw=9)?

The split is currently POSTULATED in baryogenesis code (used to set
ε_chiral = 1/5). It has NOT been verified from srs graph + geometry.
This probe verifies it.

METHOD:
  1. Build srs directed-edge graph over a 3-cell supercell (same as
     `lambda_2cycle_amplitude.py`).
  2. Enumerate all girth-10 NB cycles through vertex 0, as ORDERED
     vertex sequences (preserve traversal direction).
  3. For each cycle: project the vertex coordinates onto the plane
     perpendicular to the C_3 axis through vertex 0 (the [111] axis
     in fractional coords for I4_132 Wyckoff 8a).
  4. Compute the signed projected area via shoelace.
  5. Classify by sign: + → CCW (right-hand rule about [111]); − → CW.
  6. Report distribution. Verify a separate private derivation by the author (and our framework's
     baryogenesis postulate).

If 9 CCW + 6 CW (or symmetric counts under C_3): a separate private derivation by the author port #8 step 1
verified, framework's baryogenesis postulate now structurally supported.
If different: honest negative.
"""

import numpy as np
from itertools import product
from collections import Counter

# =============================================================================
# srs structure (matches lambda_2cycle_amplitude.py and other framework probes)
# =============================================================================
A_PRIM = np.array([[-0.5, 0.5, 0.5],
                   [ 0.5,-0.5, 0.5],
                   [ 0.5, 0.5,-0.5]])
ATOMS = np.array([[1/8, 1/8, 1/8],
                  [3/8, 7/8, 5/8],
                  [7/8, 5/8, 3/8],
                  [5/8, 3/8, 7/8]])
N_ATOMS = 4
k_star  = 3
girth   = 10
SUPERCELL = 3

def frac_to_cart(frac):
    """Fractional → Cartesian coords."""
    return A_PRIM.T @ np.asarray(frac, dtype=float)


def find_bonds():
    """Return list of (src_atom, tgt_atom, (n1,n2,n3)) NN bonds."""
    tol, NN = 0.02, np.sqrt(2) / 4
    bonds = []
    for i in range(N_ATOMS):
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                d = np.linalg.norm(rj - ATOMS[i])
                if d < tol:
                    continue
                if abs(d - NN) < tol:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds


bonds = find_bonds()
assert len(bonds) == N_ATOMS * k_star

def get_nbrs(atom, cell):
    out = []
    for src, tgt, dc in bonds:
        if src != atom:
            continue
        nc = (cell[0]+dc[0], cell[1]+dc[1], cell[2]+dc[2])
        if all(abs(c) <= SUPERCELL for c in nc):
            out.append((tgt, nc))
    return out


def vertex_cart(atom, cell):
    """Cartesian coordinates of vertex (atom, cell)."""
    frac = ATOMS[atom] + cell[0]*A_PRIM[0] + cell[1]*A_PRIM[1] + cell[2]*A_PRIM[2]
    return frac_to_cart(frac)

# =============================================================================
# Enumerate girth-10 NB cycles through vertex 0 as ORDERED vertex sequences
# =============================================================================

start = (0, (0, 0, 0))

cycles_ordered = []  # each entry: list of 10 (atom, cell) vertices in traversal order

def dfs(path, current, depth):
    atom, cell = current
    prev = path[-2] if depth >= 1 else None
    for tgt, nc in get_nbrs(atom, cell):
        if prev is not None and (tgt, nc) == prev:
            continue
        if depth == girth - 1:
            if (tgt, nc) == start:
                # Closed back to start; record the ordered traversal
                # NB: path has length depth+1; closing edge brings us back
                if start not in path[1:]:
                    cycles_ordered.append(path[:])  # 10 distinct vertices
        elif depth < girth - 1:
            if (tgt, nc) == start and depth < girth - 1:
                continue
            dfs(path + [(tgt, nc)], (tgt, nc), depth + 1)

dfs([start], start, 0)
print(f"Raw ordered cycles found through vertex 0: {len(cycles_ordered)}")

# =============================================================================
# Deduplicate to undirected cycles
# Each undirected 10-cycle through vertex 0 appears in the DFS in 2 forms
# (forward / reverse traversal starting from vertex 0). Dedup by edge SET
# while keeping ONE representative ordered traversal per undirected cycle.
# =============================================================================

def unoriented_edge(v1, v2):
    return tuple(sorted([v1, v2]))

def cycle_edge_set(path):
    edges = []
    for i in range(len(path)):
        v1 = path[i]
        v2 = path[(i+1) % len(path)]
        edges.append(unoriented_edge(v1, v2))
    return frozenset(edges)

seen = {}
for path in cycles_ordered:
    es = cycle_edge_set(path)
    if es not in seen:
        seen[es] = path

cycles_unique = list(seen.values())
n_g_count = len(cycles_unique)
print(f"Unique undirected girth-10 cycles through vertex 0: {n_g_count}")
print(f"Expected: 15 (Sunada 2012; framework's n_g)")

if n_g_count != 15:
    print(f"  WARN: got {n_g_count}, not 15. May be supercell artifact.")

# =============================================================================
# CHIRALITY CLASSIFICATION
#
# At vertex 0 of srs (I4_132 Wyckoff 8a at (1/8,1/8,1/8)), there is a 3-fold
# rotation axis along [1,1,1] (fractional coords). The C_3 acts on the three
# nearest neighbours by cyclic permutation.
#
# For each undirected 10-cycle through vertex 0:
#   - Pick the canonical orientation (first edge to lowest-keyed neighbour).
#   - Project each vertex onto the plane perpendicular to [1,1,1] (Cartesian).
#   - Compute the signed projected area via shoelace (signed angular winding
#     about the axis).
#   - Sign convention: positive = CCW relative to right-hand rule along [1,1,1].
#
# The undirected cycle has TWO traversals (forward / reverse) giving opposite
# signs. To get an unambiguous chirality, we use the canonical orientation
# (smallest-vertex-keyed first neighbour). All cycles get the SAME canonical
# rule, so the relative count of CW vs CCW is meaningful.
# =============================================================================

# C_3 axis at vertex 0 in Cartesian coords:
# In fractional coords, [1,1,1] axis: A_PRIM.T @ [1,1,1] direction
axis_cart = A_PRIM.T @ np.array([1.0, 1.0, 1.0])
axis_cart = axis_cart / np.linalg.norm(axis_cart)
print(f"\nC_3 axis at vertex 0 (Cartesian unit vector): {axis_cart}")

# Build orthonormal basis for the plane perpendicular to axis
# Pick any vector not parallel to axis
ref = np.array([1.0, 0.0, 0.0])
if abs(np.dot(ref, axis_cart)) > 0.99:
    ref = np.array([0.0, 1.0, 0.0])
e1 = ref - np.dot(ref, axis_cart) * axis_cart
e1 = e1 / np.linalg.norm(e1)
e2 = np.cross(axis_cart, e1)  # right-hand rule: e1 × e2 = axis  =>  axis × e1 = e2

def project_perp(v_cart, origin_cart):
    """Project v_cart onto the plane through origin_cart perpendicular to axis."""
    rel = v_cart - origin_cart
    rel_axial = np.dot(rel, axis_cart) * axis_cart
    return rel - rel_axial


def project_2d(v_cart, origin_cart):
    """Return (x,y) in the (e1, e2) basis on the perp plane."""
    rel = v_cart - origin_cart
    return np.array([np.dot(rel, e1), np.dot(rel, e2)])


def signed_area(points_2d):
    """Shoelace signed area for an ordered closed polygon."""
    n = len(points_2d)
    s = 0.0
    for i in range(n):
        x1, y1 = points_2d[i]
        x2, y2 = points_2d[(i+1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def canonical_orientation(path):
    """
    Return the path in a canonical orientation. Convention: among the two
    possible traversals (forward / reverse), pick the one whose SECOND vertex
    has the smaller (atom, cell) key.
    """
    fwd_second = path[1]
    rev_second = path[-1]
    if fwd_second <= rev_second:
        return path
    else:
        return [path[0]] + path[:0:-1]


# Origin = vertex 0 in Cartesian
origin_cart = vertex_cart(0, (0, 0, 0))

results = []
for path in cycles_unique:
    canon = canonical_orientation(path)
    pts_2d = [project_2d(vertex_cart(a, c), origin_cart) for (a, c) in canon]
    area = signed_area(pts_2d)
    chir = "CCW" if area > 1e-12 else ("CW" if area < -1e-12 else "FLAT")
    results.append((canon, area, chir))

# =============================================================================
# Report
# =============================================================================
print("\n" + "=" * 78)
print("Per-cycle chirality (canonical orientation; signed area on plane ⊥ [111])")
print("=" * 78)
for i, (path, area, chir) in enumerate(sorted(results, key=lambda r: r[1])):
    print(f"  cycle {i+1:2d}: signed area = {area:+.4f}  → {chir}")

counts = Counter(r[2] for r in results)
print("\n" + "=" * 78)
print("Chirality distribution at vertex 0:")
print("=" * 78)
print(f"  CCW: {counts.get('CCW', 0)}")
print(f"  CW : {counts.get('CW', 0)}")
print(f"  flat (signed area ~ 0): {counts.get('FLAT', 0)}")
print(f"  total: {sum(counts.values())}")

n_ccw = counts.get('CCW', 0)
n_cw = counts.get('CW', 0)

# a separate private derivation by the author: 3 CW + 2 CCW PER EDGE PAIR (= 9 + 6 = 15 per vertex)
# Equivalent to (n_majority - n_minority)/(n_majority + n_minority) = 1/5
expected_majority = 9
expected_minority = 6

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
print(f"\n  Framework postulate (gut_baryogenesis.py): n_ccw=9, n_cw=6 → ε_chiral = 1/5")
print(f"  a separate private derivation by the author: 3 CW + 2 CCW per edge pair = 9+6 per vertex; ε_baryon = 1/5")
print()

# Order-independent comparison
n_majority = max(n_ccw, n_cw)
n_minority = min(n_ccw, n_cw)
total = n_majority + n_minority

if total == 15 and n_majority == expected_majority and n_minority == expected_minority:
    eps = (n_majority - n_minority) / total
    print(f"  ✓ VERIFIED: split is {n_majority}+{n_minority} = 15, ε = {eps:.4f} = 1/{total/(n_majority-n_minority):.0f}")
    print(f"    a separate private derivation by the author + framework's baryogenesis postulate now STRUCTURAL.")
elif total == 15:
    eps = (n_majority - n_minority) / total
    print(f"  ✗ NEGATIVE: split is {n_majority}+{n_minority} = 15, ε = {eps:.4f}")
    print(f"    Different from a separate private derivation by the author/framework expected 9+6.")
    print(f"    Framework's baryogenesis postulate REQUIRES re-derivation.")
else:
    print(f"  ✗ ANOMALY: cycle count {total} ≠ 15.")
    print(f"    May be supercell artifact; expand and rerun.")

# =============================================================================
# Per-edge-pair refinement
# =============================================================================
# a separate private derivation by the author claim is "3 CW + 2 CCW per edge pair". Each edge pair at vertex 0
# corresponds to choosing 2 of the 3 NN edges that form the entry/exit of the
# cycle. There are C(3,2) = 3 unordered edge pairs at vertex 0; each carries
# n_g/3 = 5 cycles.

# Identify the first and last cycle edge incident to vertex 0
print("\n" + "=" * 78)
print("Per-edge-pair refinement (a separate private derivation by the author per-pair 3+2 claim)")
print("=" * 78)

# For each canonical cycle, identify the two NN edges (first and last)
nbrs = get_nbrs(0, (0,0,0))
nbr_keys = [(a, c) for (a, c) in nbrs]
nbr_keys.sort()

def cycle_edge_pair(path):
    """Return the unordered pair of NN-of-vertex-0 used by this cycle."""
    nbr_at_start = path[1]
    nbr_at_end   = path[-1]
    return tuple(sorted([nbr_at_start, nbr_at_end]))

pair_chirality = {}
for path, area, chir in results:
    pair = cycle_edge_pair(path)
    pair_chirality.setdefault(pair, []).append(chir)

print(f"\n  {len(pair_chirality)} distinct edge-pairs at vertex 0 (expected: 3)")
for pair, chirs in pair_chirality.items():
    cw = chirs.count('CW')
    ccw = chirs.count('CCW')
    print(f"    pair {pair}: {ccw} CCW + {cw} CW (total {len(chirs)})")
