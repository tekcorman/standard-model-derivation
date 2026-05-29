#!/usr/bin/env python3
"""
Diagnostic for srs_ten_cycle_chirality_split.py findings:
  - 3 CCW + 6 CW + 6 FLAT per vertex (not 9+6 as postulated)
  - Asymmetry 1/5 reproduced via 3 / 15 = 1/5

Questions:
  (a) Are the 6 FLAT cycles genuinely flat (signed area = 0 exactly), or is
      this floating-point noise that would be resolved at higher precision?
  (b) What is the geometric structure of a flat cycle? Does it have
      symmetry under reflection through a plane containing [111]?
  (c) Is there an alternative chirality measure (e.g., 4_1 screw-axis
      winding) under which all 15 cycles are classifiable?
"""

import numpy as np
from itertools import product

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
    return A_PRIM.T @ np.asarray(frac, dtype=float)

def find_bonds():
    tol, NN = 0.02, np.sqrt(2) / 4
    bonds = []
    for i in range(N_ATOMS):
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                d = np.linalg.norm(rj - ATOMS[i])
                if d < tol: continue
                if abs(d - NN) < tol:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds

bonds = find_bonds()

def get_nbrs(atom, cell):
    out = []
    for src, tgt, dc in bonds:
        if src != atom: continue
        nc = (cell[0]+dc[0], cell[1]+dc[1], cell[2]+dc[2])
        if all(abs(c) <= SUPERCELL for c in nc):
            out.append((tgt, nc))
    return out

def vertex_cart(atom, cell):
    frac = ATOMS[atom] + cell[0]*A_PRIM[0] + cell[1]*A_PRIM[1] + cell[2]*A_PRIM[2]
    return frac_to_cart(frac)

start = (0, (0, 0, 0))
cycles_ordered = []

def dfs(path, current, depth):
    atom, cell = current
    prev = path[-2] if depth >= 1 else None
    for tgt, nc in get_nbrs(atom, cell):
        if prev is not None and (tgt, nc) == prev: continue
        if depth == girth - 1:
            if (tgt, nc) == start and start not in path[1:]:
                cycles_ordered.append(path[:])
        elif depth < girth - 1:
            if (tgt, nc) == start: continue
            dfs(path + [(tgt, nc)], (tgt, nc), depth + 1)

dfs([start], start, 0)

def unoriented_edge(v1, v2):
    return tuple(sorted([v1, v2]))

def cycle_edge_set(path):
    return frozenset(unoriented_edge(path[i], path[(i+1)%len(path)]) for i in range(len(path)))

seen = {}
for path in cycles_ordered:
    es = cycle_edge_set(path)
    if es not in seen:
        seen[es] = path
cycles_unique = list(seen.values())

# Axis & basis
axis_cart = A_PRIM.T @ np.array([1.0, 1.0, 1.0])
axis_cart = axis_cart / np.linalg.norm(axis_cart)
ref = np.array([1.0, 0.0, 0.0])
e1 = ref - np.dot(ref, axis_cart) * axis_cart
e1 = e1 / np.linalg.norm(e1)
e2 = np.cross(axis_cart, e1)

origin = vertex_cart(0, (0,0,0))

def project_2d(v):
    rel = v - origin
    return np.array([np.dot(rel, e1), np.dot(rel, e2)])

def axial_proj(v):
    return np.dot(v - origin, axis_cart)

def signed_area(pts):
    n = len(pts)
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1)%n]
        s += x1*y2 - x2*y1
    return s / 2

def canonical(path):
    return path if path[1] <= path[-1] else [path[0]] + path[:0:-1]

# =============================================================================
# Re-classify with high-precision printing + axial range
# =============================================================================
print("=" * 90)
print("Per-cycle: signed area (high precision), axial range, 3D writhe sign proxy")
print("=" * 90)

results = []
for path in cycles_unique:
    canon = canonical(path)
    pts2d = [project_2d(vertex_cart(a, c)) for (a, c) in canon]
    area = signed_area(pts2d)
    # Axial range = max - min projection along [111]
    axials = [axial_proj(vertex_cart(a, c)) for (a, c) in canon]
    axial_range = max(axials) - min(axials)
    # 3D writhe sign proxy: sum of signed cross-products of consecutive edges,
    # projected onto axis. Approximates the discretized writhe.
    coords = [vertex_cart(a, c) for (a, c) in canon]
    coords = coords + [coords[0]]
    edges = [coords[i+1] - coords[i] for i in range(len(coords)-1)]
    writhe_proxy = 0.0
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            cross = np.cross(edges[i], edges[j])
            writhe_proxy += np.dot(cross, axis_cart)
    results.append((canon, area, axial_range, writhe_proxy))

for i, (path, area, axial_range, writhe) in enumerate(sorted(results, key=lambda r: r[1])):
    print(f"  cycle {i+1:2d}: signed_area = {area:+.10e}  axial_range = {axial_range:.4f}  writhe_axial = {writhe:+.4f}")

# =============================================================================
# Look at ONE flat cycle in detail
# =============================================================================
flats = [r for r in results if abs(r[1]) < 1e-10]
print()
print("=" * 90)
print(f"FLAT cycles ({len(flats)} total) — geometric structure of cycle 1:")
print("=" * 90)
if flats:
    path, area, axial_range, writhe = flats[0]
    print(f"  signed area = {area:+.4e}, axial range = {axial_range:.4f}")
    print(f"  Vertex coordinates (Cartesian) and projections:")
    for i, (a, c) in enumerate(path):
        v = vertex_cart(a, c)
        ax = axial_proj(v)
        p2 = project_2d(v)
        print(f"    v{i:2d}: cart = ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})   "
              f"axial = {ax:+.4f}   perp_2d = ({p2[0]:+.4f}, {p2[1]:+.4f})")

# =============================================================================
# Check for reflection symmetry of FLAT cycles
# A flat cycle would have signed area = 0 if its projection on the perpendicular
# plane is symmetric under some reflection (line of symmetry through origin).
# =============================================================================
print()
print("=" * 90)
print("Test: do FLAT cycles have a reflection symmetry in the perpendicular plane?")
print("=" * 90)
for k, (path, area, _, _) in enumerate(flats[:3]):
    pts2d = np.array([project_2d(vertex_cart(a, c)) for (a, c) in path])
    # For each candidate reflection axis (parameterized by angle θ), test if
    # the polygon is invariant (as a set of points) under reflection about
    # the line through origin at angle θ.
    found_axis = None
    for theta_deg in range(0, 180, 1):
        theta = np.radians(theta_deg)
        # Reflection matrix about line at angle theta
        c, s = np.cos(2*theta), np.sin(2*theta)
        R = np.array([[c, s], [s, -c]])
        reflected = pts2d @ R.T
        # Are reflected points a permutation of original?
        match = 0
        for rp in reflected:
            if any(np.linalg.norm(rp - p) < 1e-6 for p in pts2d):
                match += 1
        if match == len(pts2d):
            found_axis = theta_deg
            break
    print(f"  flat cycle {k+1}: reflection symmetry at angle = {found_axis}° "
          f"({'YES' if found_axis is not None else 'no'})")

# =============================================================================
# Try alternative chirality measure: 3D handedness via signed solid angle
# (Gauss-Bonnet / signed volume of the polygonal cone from origin)
# =============================================================================
print()
print("=" * 90)
print("Alternative chirality measure: signed solid angle from C_3 axis viewpoint")
print("=" * 90)
print("  (signed solid angle of polygon as seen from far along +[111])")

def signed_solid_angle(coords, viewpoint_dir):
    """
    For a closed polygon in 3D, signed solid angle subtended at infinity along
    viewpoint_dir. = (signed area projected onto plane perpendicular to viewpoint)
    / r² → just signed area at infinity.
    """
    # This is the same as signed_area but explicitly noting it's a solid angle proxy
    n = len(coords)
    e1_local = np.array([1.0, 0.0, 0.0]) - viewpoint_dir * np.dot([1,0,0], viewpoint_dir)
    e1_local = e1_local / np.linalg.norm(e1_local)
    e2_local = np.cross(viewpoint_dir, e1_local)
    pts2d = [(np.dot(coords[i], e1_local), np.dot(coords[i], e2_local)) for i in range(n)]
    s = 0.0
    for i in range(n):
        x1, y1 = pts2d[i]
        x2, y2 = pts2d[(i+1)%n]
        s += x1*y2 - x2*y1
    return s/2


# Compare to signed-area-on-perpendicular-plane (already computed)
# Try other axes: [100], [010], [001], [110]
test_axes = [
    ("[111]", np.array([1,1,1])/np.sqrt(3)),
    ("[100]", np.array([1,0,0], dtype=float)),
    ("[110]", np.array([1,1,0], dtype=float)/np.sqrt(2)),
    ("[1,1,-1]", np.array([1,1,-1], dtype=float)/np.sqrt(3)),
]

for axis_name, ax in test_axes:
    counts = {'CCW': 0, 'CW': 0, 'FLAT': 0}
    for path, area, _, _ in results:
        coords = [vertex_cart(a, c) - origin for (a, c) in path]
        sa = signed_solid_angle(coords, ax)
        if sa > 1e-10: counts['CCW'] += 1
        elif sa < -1e-10: counts['CW'] += 1
        else: counts['FLAT'] += 1
    print(f"  axis {axis_name}: CCW = {counts['CCW']}, CW = {counts['CW']}, FLAT = {counts['FLAT']}")
