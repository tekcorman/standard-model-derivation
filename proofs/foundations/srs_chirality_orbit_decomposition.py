#!/usr/bin/env python3
"""
Decompose 9 chiral + 6 P-symmetric girth-10 cycles through srs vertex 0
under the local C_3 rotation about [111].

QUESTION 1: Do the 9 chiral cycles split as 3 orbits of size 3 (compatible
with a "3 generations × 3 colors" reading)?

QUESTION 2: Do the 6 P-symmetric cycles split as 2 orbits of size 3
(compatible with "2 lepton-like states per color triplet")?

If yes to both, the 15-cycle decomposition naturally factors as:
  9 chiral = 3 orbits × 3 cycles  [could carry color×generation indexing]
  6 P-sym  = 2 orbits × 3 cycles  [could carry 2 lepton-like / Higgs-like states]

This would suggest a structural — not adopted — quark/lepton differentiation
mechanism, potentially feeding R-14 closure via a non-a separate private derivation by the author path.

Note (per a separate private derivation by the author port #6 NEGATIVE 2026-05-03): generation count = 3 in this
framework comes from B7.1 observer C³ (Gleason d ≥ 3), not substrate
triality. Any substrate 3-fold structure is therefore either:
  (a) a coincident identification with the observer-side 3-gen, or
  (b) a different 3-fold (e.g., the C_3 vertex symmetry itself).
The probe CANNOT independently derive generation count; it can only
identify whether substrate cycle counts have a natural 3×3 substructure
that ALIGNS with the observer-side decomposition.
"""

import numpy as np
from itertools import product
from collections import Counter

# =============================================================================
# srs structure (matches lambda_2cycle_amplitude.py)
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

# =============================================================================
# Enumerate 15 girth-10 cycles through vertex 0 (canonical undirected)
# =============================================================================
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
    return frozenset(unoriented_edge(path[i], path[(i+1)%len(path)])
                     for i in range(len(path)))

seen = {}
for path in cycles_ordered:
    es = cycle_edge_set(path)
    if es not in seen:
        seen[es] = path
cycles_unique = list(seen.values())
assert len(cycles_unique) == 15, f"expected 15 cycles, got {len(cycles_unique)}"

# =============================================================================
# Classify chiral vs P-symmetric using signed-projected-area on perp([111])
# =============================================================================
axis = A_PRIM.T @ np.array([1.0, 1.0, 1.0]); axis /= np.linalg.norm(axis)
ref = np.array([1.0, 0.0, 0.0])
e1 = ref - np.dot(ref, axis)*axis; e1 /= np.linalg.norm(e1)
e2 = np.cross(axis, e1)
origin = vertex_cart(0, (0,0,0))

def project_2d(v):
    rel = v - origin
    return np.array([np.dot(rel, e1), np.dot(rel, e2)])

def signed_area(pts):
    n = len(pts); s = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i+1)%n]
        s += x1*y2 - x2*y1
    return s / 2

chiral_cycles = []
psym_cycles = []
for path in cycles_unique:
    pts = [project_2d(vertex_cart(a, c)) for (a, c) in path]
    sa = signed_area(pts)
    if abs(sa) > 1e-10:
        chiral_cycles.append(path)
    else:
        psym_cycles.append(path)

assert len(chiral_cycles) == 9
assert len(psym_cycles) == 6
print(f"Cycles: 9 chiral + 6 P-symmetric (= 15) verified.\n")

# =============================================================================
# C_3 rotation about [111] through origin in Cartesian = cyclic axis permutation
# Acts on Cartesian (x, y, z) → (z, x, y).
# In fractional coords, A_PRIM is symmetric under cyclic permutation of basis
# vectors? Let's check: A_PRIM rows are [(-1,1,1),(1,-1,1),(1,1,-1)]/2.
# Cyclic permutation of components within each row → row 0 (-1,1,1)→(1,-1,1) = row 1!
# So C_3 on Cartesian (x,y,z) → (z,x,y) corresponds to permuting which atom of
# the 4 in the unit cell maps to which (since the atom positions are themselves
# C_3-related in srs Wyckoff 8a).
# =============================================================================

def C3_cart(v):
    """Apply C_3 rotation about [111] through origin to a Cartesian vector v."""
    # (x, y, z) -> (z, x, y)
    return np.array([v[2], v[0], v[1]])


def apply_C3_to_vertex(atom_cell):
    """
    Apply the C_3 rotation to a vertex (atom, cell) and return the resulting
    (atom', cell') tuple. The rotation acts on Cartesian coords; we identify
    the result by closest vertex match in the same supercell.
    """
    a, c = atom_cell
    v = vertex_cart(a, c)
    v_rot = C3_cart(v)
    # Find which (a', c') has Cartesian coords closest to v_rot
    best = None; best_d = float('inf')
    for ap in range(N_ATOMS):
        for n1, n2, n3 in product(range(-SUPERCELL-1, SUPERCELL+2), repeat=3):
            cp = (n1, n2, n3)
            vp = vertex_cart(ap, cp)
            d = np.linalg.norm(vp - v_rot)
            if d < best_d:
                best_d = d; best = (ap, cp)
    if best_d > 1e-6:
        return None
    return best


def apply_C3_to_cycle(path):
    """Apply C_3 to each vertex in the cycle. Return new path or None."""
    new_path = []
    for v in path:
        nv = apply_C3_to_vertex(v)
        if nv is None: return None
        new_path.append(nv)
    return new_path


def cycle_match(path, candidates):
    """Find the index of the candidate cycle whose edge-set matches path."""
    target = cycle_edge_set(path)
    for i, p in enumerate(candidates):
        if cycle_edge_set(p) == target:
            return i
    return None

# =============================================================================
# Build C_3 permutation on 9 chiral cycles
# =============================================================================
def find_orbits(cycles_subset):
    """Find C_3 orbits on the given cycle subset (returned as lists of indices)."""
    n = len(cycles_subset)
    perm = {}
    for i, path in enumerate(cycles_subset):
        rotated = apply_C3_to_cycle(path)
        if rotated is None:
            print(f"  warning: C_3 image of cycle {i} fell outside supercell")
            return None
        j = cycle_match(rotated, cycles_subset)
        if j is None:
            print(f"  warning: C_3 image of cycle {i} not found in subset")
            return None
        perm[i] = j

    # Find orbits
    visited = set()
    orbits = []
    for i in range(n):
        if i in visited: continue
        orbit = [i]; visited.add(i)
        j = perm[i]
        while j != i:
            orbit.append(j); visited.add(j)
            j = perm[j]
        orbits.append(orbit)
    return orbits, perm


print("=" * 80)
print("C_3 orbit structure on the 9 chiral cycles")
print("=" * 80)
result = find_orbits(chiral_cycles)
if result is None:
    print("  FAILED — C_3 image fell outside enumeration. Need larger supercell.")
else:
    orbits_chiral, perm_chiral = result
    sizes = sorted(Counter(len(o) for o in orbits_chiral).items())
    print(f"  {len(orbits_chiral)} orbits with size distribution: "
          + ", ".join(f"{c}×size-{s}" for s, c in sizes))
    for k, orb in enumerate(orbits_chiral):
        print(f"  orbit {k+1} (size {len(orb)}): cycle indices {orb}")

print()
print("=" * 80)
print("C_3 orbit structure on the 6 P-symmetric cycles")
print("=" * 80)
result = find_orbits(psym_cycles)
if result is None:
    print("  FAILED")
else:
    orbits_psym, perm_psym = result
    sizes = sorted(Counter(len(o) for o in orbits_psym).items())
    print(f"  {len(orbits_psym)} orbits with size distribution: "
          + ", ".join(f"{c}×size-{s}" for s, c in sizes))
    for k, orb in enumerate(orbits_psym):
        print(f"  orbit {k+1} (size {len(orb)}): cycle indices {orb}")

# =============================================================================
# Verdict
# =============================================================================
print()
print("=" * 80)
print("VERDICT")
print("=" * 80)
chiral_ok = result is not None and \
            len(orbits_chiral) > 0 and \
            all(len(o) == 3 for o in orbits_chiral) and \
            len(orbits_chiral) == 3
psym_ok = result is not None and \
          len(orbits_psym) > 0 and \
          all(len(o) == 3 for o in orbits_psym) and \
          len(orbits_psym) == 2

if chiral_ok and psym_ok:
    print("""
  ✓ POSITIVE: Both decompositions match the predicted structure.
    9 chiral = 3 orbits × 3 cycles  (consistent with 3-color × 3-flavor labeling)
    6 P-sym  = 2 orbits × 3 cycles  (consistent with 2 lepton-like states / triplet)

  The 15-cycle vertex structure factors naturally as:
    [3 chiral C_3-orbits] × [3 cycles per orbit] + [2 P-symmetric orbits] × [3]
    = 9 chiral + 6 P-sym

  Forward path: examine whether the 3 chiral orbits correspond to flavors
  (up/down/strange or u/c/t etc.) and whether the 2 P-symmetric orbits
  correspond to charged-lepton + neutrino (or similar).
""")
else:
    print(f"""
  ◐ MIXED: structure differs from predicted 3×3 / 2×3.
    Chiral expected 3×3, got: {[len(o) for o in (orbits_chiral if result is not None else [])]}
    P-sym  expected 2×3, got: {[len(o) for o in (orbits_psym if result is not None else [])]}

  This is informative regardless: the actual orbit structure is a structural
  fact about srs, and may suggest a different interpretation than the
  speculative 3×3 / 2×3 reading.
""")
