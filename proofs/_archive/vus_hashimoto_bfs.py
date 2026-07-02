#!/usr/bin/env python3
"""
proofs/_archive/vus_hashimoto_bfs.py

FRAMEWORK LEVEL: Level 3 (Hashimoto graph).

PURPOSE
-------
Find the minimum NB cycle-distances between ALL pairs of bond-type
labels in srs girth cycles, extending vcb_hashimoto_bfs.py.

For V_cb we found: b1↔b2 within the same C3 orbit → distance L_cb=8.
For V_us we need: identify which bond-type pair corresponds to u↔s,
and read off the minimum cycle-distance L_us.

Methodology: enumerate girth-10 NB cycles via DFS on an 8³ supercell,
then for every ordered pair (type_A, type_B) record the minimum
forward cycle-distance from any type_A edge to any type_B edge.

PDG target: V_us = 0.22501 ± 0.00068.
Using V = (2/3)^L / (1 - (2/3)^L), we need L ≈ 4.178.
Using V_bare = (2/3)^L alone (no winding sum), we need L ≈ 3.678.
Since 3.678 ≈ g/e = 10/e, also check this possibility.

GATE STATUS
-----------
Enumeration: Type 2 (construction).
BFS/DFS: Type 2 (explicit algorithm).
Result interpretation: must be matched to quark species identification.
"""

import sys
import os
import math
from fractions import Fraction
from itertools import product as iproduct
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))
from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial

d_space = predict_d_spatial()
k = predict_k_star(d_space)
g = predict_g_girth(k, d_space)
assert k == 3 and g == 10

bonds_prim = find_bonds()
assert len(bonds_prim) == 12

# ------------------------------------------------------------------
# C3 orbit labeling (same as vcb_hashimoto_bfs.py)
# ------------------------------------------------------------------

C3_CART = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
c3_atom = {i: int(np.argmax(C3_PERM[:, i])) for i in range(N_ATOMS)}


def bond_disp(src, tgt, cell):
    return (np.array(ATOMS[tgt])
            + cell[0] * np.array(A_PRIM[0])
            + cell[1] * np.array(A_PRIM[1])
            + cell[2] * np.array(A_PRIM[2])
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

visited_orb = [False] * 12
orbits = []
for start in range(12):
    if visited_orb[start]:
        continue
    b0, b1, b2 = start, c3_map[start], c3_map[c3_map[start]]
    assert c3_map[b2] == b0 and len({b0, b1, b2}) == 3
    orbits.append((b0, b1, b2))
    visited_orb[b0] = visited_orb[b1] = visited_orb[b2] = True

assert len(orbits) == 4  # 4 orbits × 3 positions = 12 directed bonds

type_label = {}
for oi, (b0, b1, b2) in enumerate(orbits):
    type_label[b0] = (oi, 0)
    type_label[b1] = (oi, 1)   # C3=ω² position
    type_label[b2] = (oi, 2)   # C3=ω  position

# Print orbit structure
print("=== C3 orbit structure ===")
for oi, (b0, b1, b2) in enumerate(orbits):
    s0, t0, c0 = bonds_prim[b0]
    s1, t1, c1 = bonds_prim[b1]
    s2, t2, c2 = bonds_prim[b2]
    print(f"  Orbit {oi}: b0={b0}({s0}→{t0},{c0}), "
          f"b1={b1}({s1}→{t1},{c1}), b2={b2}({s2}→{t2},{c2})")

# ------------------------------------------------------------------
# Supercell NB walk
# ------------------------------------------------------------------

N_SUPER = 8


def in_bounds(cell):
    return all(0 <= cell[d] < N_SUPER for d in range(3))


def nb_successors(src_a, src_c, tgt_a, tgt_c):
    result = []
    for (s, t, dc) in bonds_prim:
        if s != tgt_a:
            continue
        nc = tuple(tgt_c[d] + dc[d] for d in range(3))
        if not in_bounds(nc):
            continue
        if t == src_a and all(nc[d] == src_c[d] for d in range(3)):
            continue
        result.append((tgt_a, tgt_c, t, nc))
    return result


def edge_prim_type(src_a, src_c, tgt_a, tgt_c):
    dc = tuple(tgt_c[d] - src_c[d] for d in range(3))
    return prim_type_key.get((src_a, tgt_a, dc))


# ------------------------------------------------------------------
# Girth cycle finder (DFS)
# ------------------------------------------------------------------

def find_girth_cycles(start_edge, girth=10, max_cycles=50):
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


# ------------------------------------------------------------------
# Cycle analysis: ALL type-pair distances
# ------------------------------------------------------------------

def analyse_cycle_all_pairs(cycle, g=10):
    """
    For every ordered pair (type_A, type_B), find all forward
    cycle-distances from any type_A edge to any type_B edge.
    """
    labels = []
    for e in cycle:
        pt = edge_prim_type(*e)
        if pt is None:
            return None
        labels.append(type_label[pt])

    pair_distances = defaultdict(list)
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            fwd = (j - i) % g
            if fwd == 0:
                continue
            pair_distances[(la, lb)].append(fwd)
    return pair_distances


# ------------------------------------------------------------------
# MAIN: collect minimum distances for all type pairs
# ------------------------------------------------------------------

def run(n_starts=80, max_cycles_per_start=30):
    center = (N_SUPER // 2,) * 3
    all_min_dists = defaultdict(lambda: g + 1)  # pair -> min forward distance
    all_min_examples = {}
    total_cycles = 0

    for oi_start, (b0, b1, b2) in enumerate(orbits):
        for pos_start, bk in enumerate([b0, b1, b2]):
            src_a, tgt_a, dc = bonds_prim[bk]
            start_edge = (src_a, center, tgt_a,
                          tuple(center[d] + dc[d] for d in range(3)))
            if not in_bounds(start_edge[3]):
                continue

            cycles = find_girth_cycles(start_edge, girth=g,
                                       max_cycles=max_cycles_per_start)
            total_cycles += len(cycles)

            for cyc in cycles:
                pd = analyse_cycle_all_pairs(cyc, g)
                if pd is None:
                    continue
                for pair, dists in pd.items():
                    mn = min(dists)
                    if mn < all_min_dists[pair]:
                        all_min_dists[pair] = mn
                        all_min_examples[pair] = (dists, cyc)

    print(f"\n=== Girth cycle survey: {total_cycles} cycles from {n_starts} starting types ===\n")

    # Sort and print all pairs with their min distances
    alpha = (k - 1) / k  # = 2/3

    print(f"{'Type A':20s}  {'Type B':20s}  {'min_dist':8s}  "
          f"{'V_bare':12s}  {'V_geom':12s}  {'V_PDG_sig':12s}")
    print("-" * 90)

    # PDG reference
    V_us_pdg = 0.22501
    V_cb_pdg = 0.04050

    results = []
    for (la, lb), mn_d in sorted(all_min_dists.items(), key=lambda x: x[1]):
        if mn_d > g:
            continue
        la_str = f"orb{la[0]}pos{la[1]}"
        lb_str = f"orb{lb[0]}pos{lb[1]}"
        v_bare = alpha ** mn_d
        v_geom = v_bare / (1 - v_bare)
        # sigma from PDG V_us
        sig_us = abs(v_geom - V_us_pdg) / 0.00068
        sig_cb = abs(v_geom - V_cb_pdg) / 0.0015
        results.append((mn_d, la, lb, v_bare, v_geom, sig_us, sig_cb))
        print(f"{la_str:20s}  {lb_str:20s}  {mn_d:8d}  "
              f"{v_bare:12.6f}  {v_geom:12.6f}  {sig_us:+12.2f}σ(us)")

    print()

    # Summary: which min-distance pairs most closely match V_us
    print("=== Closest matches to PDG V_us = 0.22501 ± 0.00068 ===")
    by_sig = sorted(results, key=lambda x: x[5])
    for mn_d, la, lb, v_bare, v_geom, sig_us, sig_cb in by_sig[:10]:
        print(f"  ({la[0]},{la[1]})→({lb[0]},{lb[1]}) dist={mn_d}  "
              f"V_geom={v_geom:.5f}  {sig_us:+.2f}σ")

    print()
    print("=== Closest matches to PDG V_cb = 0.04050 ± 0.00150 ===")
    by_sig_cb = sorted(results, key=lambda x: x[6])
    for mn_d, la, lb, v_bare, v_geom, sig_us, sig_cb in by_sig_cb[:5]:
        print(f"  ({la[0]},{la[1]})→({lb[0]},{lb[1]}) dist={mn_d}  "
              f"V_geom={v_geom:.5f}  {sig_cb:+.2f}σ(cb)")

    print()

    # Also check: bare value (no geometric series) vs irrational L candidates
    print("=== Irrational L candidates (bare value only, for comparison) ===")
    import math
    L_candidates = {
        'g/e': g / math.e,
        '2+sqrt3': 2 + math.sqrt(3),
        'MFPT=3': 3.0,
        'g/k*': g / k,
        '1/(2-sqrt2)': 1 / (2 - math.sqrt(2)),
    }
    for name, L in L_candidates.items():
        v_bare = alpha ** L
        v_geom = v_bare / (1 - v_bare)
        sig = abs(v_bare - V_us_pdg) / 0.00068
        sig_g = abs(v_geom - V_us_pdg) / 0.00068
        print(f"  L={name}={L:.4f}:  bare={v_bare:.5f} ({sig:+.1f}σ),  "
              f"geom={v_geom:.5f} ({sig_g:+.1f}σ)")

    return results, all_min_dists


if __name__ == "__main__":
    run()
