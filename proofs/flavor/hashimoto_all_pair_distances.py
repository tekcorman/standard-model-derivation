#!/usr/bin/env python3
"""
proofs/flavor/hashimoto_all_pair_distances.py

FRAMEWORK LEVEL: Level 3 (Hashimoto graph = causal observer graph).

PURPOSE
-------
Enumerate ALL pairwise cycle-distances between directed edge types
(classified by C3 orbit index and position) on the srs Hashimoto graph.

This is CAS infrastructure for future CKM derivations.  It does NOT
assign species (quarks) to orbit types — any such assignment is an
ADOPTION that must be justified separately.

MOTIVATION
----------
The V_cb derivation uses same-orbit (b1, b2) pairs at cycle-distance
g−2 = 8.  For other CKM elements (V_us, V_ub, V_cd, V_cs, V_td, V_ts),
the relevant Hashimoto node pairs may be of different orbit types or
at different cycle-distances.  This script enumerates ALL combinations
so that any future species identification has a complete CAS record.

GATE-FIRST ANALYSIS (parameter_linter.md hard gate)
────────────────────────────────────────────────────
Definition [Type 3, Hashimoto 1989; Terras 2011 §2.1]:
  H(G) has nodes = directed edges of G, arcs = NB continuations.

Definition [Type 3, Terras 2011 §1.4]:
  A girth cycle of H(G) is a closed NB walk of minimum length g.

Step 1 [Type 4, proofs/common.py find_bonds()]:
  srs primitive cell: 4 atoms, k*=3, 12 directed bonds.

Step 2 [Type 2, construction]:
  Build supercell; Hashimoto node = (src_atom, src_cell, tgt_atom, tgt_cell).

Step 3 [Type 2]:
  C3 orbit classification of all 12 bond types → 4 orbits of size 3.
  Each bond has type (orbit_index, orbit_position) in {0,1,2,3} x {0,1,2}.

Step 4 [Type 2]:
  DFS girth cycle enumeration (same as vcb_hashimoto_bfs.py).

Step 5 [Type 2]:
  For each girth cycle, record the cycle-distance from every directed edge
  of type (oi, pi) to every directed edge of type (oj, pj) that appears
  later in the same cycle.  Tabulate ALL (oi, pi, oj, pj) → {distances}.

GATE STATUS
───────────
Steps 1–5: gate-pass (Type 4 / 2 / 2 / 2 / 2).
Output: complete inventory of cycle-distances between all directed edge
type pairs in girth cycles.  No species identification is made here.

NOTE ON V_us BLOCK
──────────────────
The geometric series formula V = (2/3)^L/(1−(2/3)^L) with integer L
cannot produce V_us ≈ 0.22501:
  L=4: V = 0.246 (+31 sigma)
  L=5: V = 0.152 (-108 sigma)
No integer L falls in the required range.  V_us requires either:
  (a) A non-integer L from a spectral property (Level 2, retired formula)
  (b) A different formula structure entirely
The BFS enumeration here does NOT unblock V_us.

NOTE ON V_ub
────────────
L=14 gives V_ub = 16384/4766585 ≈ 0.003437 (−2.30σ from PDG 0.00369).
For this to constitute a derivation, one needs:
  (1) A gate-passing reason for L=14 from a specific orbit-pair type
  (2) A species identification (u→b quark assignment to Hashimoto nodes)
  (3) Explicit user permission before filing in predictions/ (session 7 rule)
This script provides (1) as CAS inventory only; (2) and (3) are open.
"""

import sys
import os
from fractions import Fraction
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))
from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial

# ──────────────────────────────────────────────────────────────────────────────
# INPUTS (Type 4)
# ──────────────────────────────────────────────────────────────────────────────

d_space = predict_d_spatial()
k = predict_k_star(d_space)
g = predict_g_girth(k, d_space)

assert k == 3
assert g == 10

bonds_prim = find_bonds()
assert len(bonds_prim) == 12

# ──────────────────────────────────────────────────────────────────────────────
# C3 orbit structure (Type 2 — same as vcb_hashimoto_bfs.py)
# ──────────────────────────────────────────────────────────────────────────────

C3_CART = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)

c3_atom = {i: int(np.argmax(C3_PERM[:, i])) for i in range(N_ATOMS)}


def bond_disp(src, tgt, cell):
    return (np.array(ATOMS[tgt])
            + cell[0]*np.array(A_PRIM[0])
            + cell[1]*np.array(A_PRIM[1])
            + cell[2]*np.array(A_PRIM[2])
            - np.array(ATOMS[src]))


prim_disps = [bond_disp(src, tgt, cell) for src, tgt, cell in bonds_prim]
prim_type_key = {(src, tgt, cell): i for i, (src, tgt, cell) in enumerate(bonds_prim)}


def c3_of_bond(i):
    src, _, _ = bonds_prim[i]
    new_src = c3_atom[src]
    rotated = C3_CART @ prim_disps[i]
    for j, (s, t, c) in enumerate(bonds_prim):
        if s == new_src and np.allclose(prim_disps[j], rotated, atol=1e-8):
            return j
    raise ValueError(f"C3 image of bond {i} not found")


c3_map = [c3_of_bond(i) for i in range(12)]

visited_orb = [False]*12
orbits = []
for start in range(12):
    if visited_orb[start]:
        continue
    b0, b1, b2 = start, c3_map[start], c3_map[c3_map[start]]
    assert c3_map[b2] == b0 and len({b0, b1, b2}) == 3
    orbits.append((b0, b1, b2))
    visited_orb[b0] = visited_orb[b1] = visited_orb[b2] = True

assert len(orbits) == 4

type_label = {}  # bond_index -> (orbit_index, position_in_orbit)
for oi, (b0, b1, b2) in enumerate(orbits):
    type_label[b0] = (oi, 0)   # canonical
    type_label[b1] = (oi, 1)   # "C3=omega^2"
    type_label[b2] = (oi, 2)   # "C3=omega"

# ──────────────────────────────────────────────────────────────────────────────
# Supercell and NB continuations (Level 3 — Hashimoto arcs)
# ──────────────────────────────────────────────────────────────────────────────

N_SUPER = 8   # 8^3 = 512 primitive cells


def in_bounds(cell):
    return all(0 <= cell[d] < N_SUPER for d in range(3))


def nb_successors(src_a, src_c, tgt_a, tgt_c):
    """Level 3: NB continuations of directed edge (src_a,src_c)->(tgt_a,tgt_c)."""
    result = []
    for (s, t, dc) in bonds_prim:
        if s != tgt_a:
            continue
        nc = tuple(tgt_c[d] + dc[d] for d in range(3))
        if not in_bounds(nc):
            continue
        if t == src_a and all(nc[d] == src_c[d] for d in range(3)):
            continue   # reverse edge — NB-invalid
        result.append((tgt_a, tgt_c, t, nc))
    return result


def edge_prim_type(src_a, src_c, tgt_a, tgt_c):
    """Primitive bond type index of a supercell directed edge."""
    dc = tuple(tgt_c[d] - src_c[d] for d in range(3))
    return prim_type_key.get((src_a, tgt_a, dc))


# ──────────────────────────────────────────────────────────────────────────────
# Girth cycle DFS (same as vcb_hashimoto_bfs.py)
# ──────────────────────────────────────────────────────────────────────────────

def find_girth_cycles(start_edge, girth=10, max_cycles=20):
    """DFS to find simple closed NB walks of length exactly girth."""
    found = []
    path_set = {start_edge}

    def dfs(current, path, depth):
        if len(found) >= max_cycles:
            return
        if depth == girth:
            for succ in nb_successors(*current):
                if succ == start_edge:
                    found.append(list(path))
                    return
            return
        for succ in nb_successors(*current):
            if succ == start_edge:
                if depth == girth - 1:
                    found.append(list(path))
                continue
            if succ in path_set:
                continue
            path_set.add(succ)
            path.append(succ)
            dfs(succ, path, depth + 1)
            path.pop()
            path_set.discard(succ)

    dfs(start_edge, [start_edge], 1)
    return found


# ──────────────────────────────────────────────────────────────────────────────
# All-pair distance enumeration
# ──────────────────────────────────────────────────────────────────────────────

def analyse_cycle_all_pairs(cycle, g):
    """
    Given a girth cycle (list of g Hashimoto nodes), classify each edge by
    (orbit_idx, position_in_orbit) and record ALL forward cycle-distances
    between ALL pairs of edge types.

    Returns: list of (type_i, type_j, fwd_distance) tuples.
    """
    labels = []
    for e in cycle:
        pt = edge_prim_type(*e)
        if pt is None:
            return []
        labels.append(type_label[pt])

    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            fwd = (j - i) % g
            pairs.append((labels[i], labels[j], fwd))
    return pairs


def run_all_pairs(n_starts=40):
    """
    Try up to n_starts different starting directed edges, find girth cycles,
    record ALL pairwise cycle-distances between ALL edge types.

    Returns: dict mapping (type_i, type_j) -> Counter of cycle-distances.
    """
    center = (N_SUPER // 2,) * 3
    pair_dist = defaultdict(Counter)  # (type_i, type_j) -> Counter(distance -> count)
    cycles_found = 0
    starts_tried = 0

    for bond_idx in range(12):
        if starts_tried >= n_starts:
            break
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell):
            continue
        start = (prim_bond[0], center, prim_bond[1], tgt_cell)
        starts_tried += 1

        cycles = find_girth_cycles(start, girth=g, max_cycles=10)
        for cycle in cycles:
            cycles_found += 1
            if len(cycle) != g:
                continue
            pairs = analyse_cycle_all_pairs(cycle, g)
            for ti, tj, fwd in pairs:
                pair_dist[(ti, tj)][fwd] += 1

    return pair_dist, cycles_found


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("Hashimoto girth-cycle all-pair distance inventory")
    print("Level 3 (causal observer graph) — srs net")
    print("=" * 70)
    print(f"\n  k* = {k},  g = {g},  N_SUPER = {N_SUPER}")
    print(f"\n  C3 orbits (4 orbits x 3 positions = 12 directed bond types):")
    for oi, (b0, b1, b2) in enumerate(orbits):
        s0, t0, c0 = bonds_prim[b0]
        s1, t1, _ = bonds_prim[b1]
        s2, t2, _ = bonds_prim[b2]
        print(f"    Orbit {oi}: pos0=(a{s0}→a{t0})  pos1=C3(b0)=(a{s1}→a{t1})  "
              f"pos2=C3²(b0)=(a{s2}→a{t2})")
    print()
    print("  Running all-pair distance enumeration...")

    pair_dist, n_cycles = run_all_pairs(n_starts=40)

    print(f"  Girth cycles found: {n_cycles}")
    print()

    # Compute geometric series value V = u^d/(1-u^d) for reference
    def V_geom(L):
        u = Fraction(2, 3)
        alpha = u**L
        return float(alpha / (1 - alpha))

    print("  Reference: V = (2/3)^L/(1-(2/3)^L) for integer L:")
    for L in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        print(f"    L={L:2d}: V = {V_geom(L):.6f}")

    print()
    print("  PDG reference values:")
    print("    V_cb = 0.04050 ± 0.00150  (L=8 → 0.04060, +0.07sigma)")
    print("    V_us = 0.22501 ± 0.00068  (NO integer L gives this)")
    print("    V_ub = 0.00369 ± 0.00011  (L=14 → 0.003437, -2.30sigma)")
    print()

    # Print pairwise distance tables grouped by orbit relationship
    print("  " + "=" * 66)
    print("  CYCLE-DISTANCE DISTRIBUTIONS FOR ALL ORDERED PAIRS OF EDGE TYPES")
    print("  " + "=" * 66)
    print("  Format: (orbit_i, pos_i) -> (orbit_j, pos_j)  |  distances")
    print()

    # Group: same-orbit pairs
    print("  --- SAME-ORBIT PAIRS ---")
    for oi in range(4):
        for pi in range(3):
            for pj in range(3):
                if pi == pj:
                    continue
                key = ((oi, pi), (oi, pj))
                if key not in pair_dist:
                    continue
                dist_ctr = pair_dist[key]
                dists = sorted(dist_ctr.keys())
                print(f"    Orbit {oi}: ({oi},{pi}) -> ({oi},{pj})  |  ", end="")
                for d in dists:
                    v_approx = V_geom(d)
                    print(f"d={d}:count={dist_ctr[d]}", end="  ")
                print()

    print()
    print("  --- CROSS-ORBIT PAIRS (same position) ---")
    for pos in range(3):
        for oi in range(4):
            for oj in range(4):
                if oi == oj:
                    continue
                key = ((oi, pos), (oj, pos))
                if key not in pair_dist:
                    continue
                dist_ctr = pair_dist[key]
                dists = sorted(dist_ctr.keys())
                print(f"    pos={pos}: ({oi},{pos}) -> ({oj},{pos})  |  ", end="")
                for d in dists:
                    print(f"d={d}:count={dist_ctr[d]}", end="  ")
                print()

    print()
    print("  --- CROSS-ORBIT PAIRS (different positions) ---")
    for pi in range(3):
        for pj in range(3):
            if pi == pj:
                continue
            for oi in range(4):
                for oj in range(4):
                    if oi == oj:
                        continue
                    key = ((oi, pi), (oj, pj))
                    if key not in pair_dist:
                        continue
                    dist_ctr = pair_dist[key]
                    dists = sorted(dist_ctr.keys())
                    print(f"    ({oi},{pi}) -> ({oj},{pj})  |  ", end="")
                    for d in dists:
                        print(f"d={d}:count={dist_ctr[d]}", end="  ")
                    print()

    print()
    print("  " + "=" * 66)
    print("  SUMMARY: minimum cycle-distances per pair type")
    print("  " + "=" * 66)
    print(f"  {'type_i':>12}  {'type_j':>12}  {'min_dist':>8}  {'V_geom':>10}  {'PDG match?':>12}")
    pdg_vals = {
        'V_cb': (0.04050, 0.00150),
        'V_us': (0.22501, 0.00068),
        'V_ub': (0.00369, 0.00011),
    }
    print()
    for (ti, tj), ctr in sorted(pair_dist.items()):
        dists = sorted(ctr.keys())
        min_d = dists[0]
        v = V_geom(min_d)
        # Check PDG matches
        match = []
        for name, (obs, sig) in pdg_vals.items():
            dev = abs(v - obs) / sig
            if dev < 3.0:
                match.append(f"{name} ({dev:+.1f}sigma)")
        match_str = ', '.join(match) if match else ''
        print(f"  {str(ti):>12}  {str(tj):>12}  {min_d:>8d}  {v:>10.6f}  {match_str}")

    print()
    print("  GATE STATUS: CAS inventory complete.")
    print("  No species identification is made. Any mapping of (orbit, pos) pairs")
    print("  to physical quarks (u, d, s, c, b, t) is an ADOPTION requiring")
    print("  explicit user permission before filing in predictions/.")
    print()
    print("  KEY FINDING ON V_us:")
    print("  The geometric series formula V=(2/3)^L/(1-(2/3)^L) with integer L")
    print("  cannot produce V_us ~ 0.225.  No cycle-distance from any pair type")
    print("  in girth cycles gives V in the range [0.218, 0.232] (the 10-sigma")
    print("  window around V_us = 0.22501 +/- 0.00068).")
    print("  CONCLUSION: V_us is BLOCKED under the current Level-3 Hashimoto")
    print("  girth-cycle framework with the geometric series formula.")
    print()
    print("  THREE-LEVEL CHECK:")
    print("    Level 1 (toggles): branch measure μ  — NOT computed here")
    print("    Level 2 (srs crystal): bond geometry from find_bonds() — input")
    print("    Level 3 (Hashimoto graph): girth cycles on directed edges — THIS FILE")
