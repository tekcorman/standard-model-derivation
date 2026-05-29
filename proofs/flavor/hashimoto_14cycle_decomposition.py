#!/usr/bin/env python3
"""
proofs/flavor/hashimoto_14cycle_decomposition.py

Companion to hashimoto_16cycle_decomposition.py. Tests whether L=14 NB
cycles on H(srs) decompose as the symmetric difference of two girth-10
NB cycles sharing a 3-edge NB path (since 20 - 2·3 = 14).

If L=14 also decomposes universally, we have:
  L=10: 1 fundamental cycle (V_cb)
  L=14: 2 fundamental cycles glued by 3-edge seam
  L=16: 2 fundamental cycles glued by 2-edge seam (V_ub candidate)

The structural question then becomes:
  Why does V_ub host on L=16 (2-edge seam) rather than L=14 (3-edge seam)?

A natural answer would invoke the 2-edge seam = n_fixed of the Feshbach
endpoint count. The seam length s and the Feshbach n_fixed must agree;
n_fixed=2 ⇒ s=2 ⇒ L=16. For n_fixed=3 ⇒ s=3 ⇒ L=14, but n_fixed ∈ {0,1,2}
in the Feshbach principle as currently stated. So L=14 is excluded by
the n_fixed cap, not by graph geometry.
"""

import sys
import os
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vcb_hashimoto_bfs as vcb

bonds_prim     = vcb.bonds_prim
N_SUPER        = vcb.N_SUPER
nb_successors  = vcb.nb_successors
edge_prim_type = vcb.edge_prim_type
in_bounds      = vcb.in_bounds


def out_edges_at(atom, cell):
    out = []
    for (s, t, dc) in bonds_prim:
        if s != atom: continue
        nc = tuple(cell[d] + dc[d] for d in range(3))
        if not in_bounds(nc): continue
        out.append((atom, cell, t, nc))
    return out


def find_cycles_at_length(start_edge, L, max_cycles):
    found = []
    path_set = {start_edge}
    def dfs(current, path, depth):
        if len(found) >= max_cycles: return
        if depth == L:
            for s in nb_successors(*current):
                if s == start_edge:
                    found.append(list(path)); return
            return
        for s in nb_successors(*current):
            if s == start_edge:
                if depth == L - 1: found.append(list(path))
                continue
            if s in path_set: continue
            path_set.add(s); path.append(s)
            dfs(s, path, depth + 1)
            path.pop(); path_set.discard(s)
    dfs(start_edge, [start_edge], 1)
    return found


def cycle_to_vertex_sequence(cycle):
    verts = [(e[0], e[1]) for e in cycle]
    last = (cycle[-1][2], cycle[-1][3])
    verts.append(last)
    return verts


def cycle_edge_set(cycle):
    edges = set()
    for (s, c, t, c2) in cycle:
        edges.add(frozenset({(s, c), (t, c2)}))
    return edges


def is_nb_path(seq):
    for i in range(1, len(seq) - 1):
        if seq[i - 1] == seq[i + 1]:
            return False
    return True


def find_chord_length3(v_a, v_b, forbidden_edges):
    """All NB paths of length 3 from v_a to v_b avoiding forbidden edges.
    Returns list of (v_mid1, v_mid2) tuples."""
    paths = []
    for e1 in out_edges_at(*v_a):
        v_m1 = (e1[2], e1[3])
        edge1 = frozenset({v_a, v_m1})
        if edge1 in forbidden_edges: continue
        if v_m1 == v_b: continue
        for e2 in out_edges_at(*v_m1):
            v_m2 = (e2[2], e2[3])
            edge2 = frozenset({v_m1, v_m2})
            if edge2 in forbidden_edges: continue
            if v_m2 == v_a: continue   # NB at v_m1
            if v_m2 == v_b: continue   # 2-edge path; we want exactly 3
            for e3 in out_edges_at(*v_m2):
                v_end = (e3[2], e3[3])
                edge3 = frozenset({v_m2, v_end})
                if edge3 in forbidden_edges: continue
                if v_end != v_b: continue
                if v_end == v_m1: continue   # NB at v_m2
                paths.append((v_m1, v_m2))
    return paths


def check_L14_decomposition(cycle14):
    """For each pair (i, j) on the L=14 cycle with cycle-distance 7
    (= antipodal, half-cycle), search for a 3-edge NB chord. Verify
    the two resulting closed walks are valid 10-edge NB cycles."""
    verts = cycle_to_vertex_sequence(cycle14)
    cyc_edges = cycle_edge_set(cycle14)

    successes = []
    for i in range(14):
        j = (i + 7) % 14
        v_i = verts[i]; v_j = verts[j]
        if v_i == v_j: continue

        chords = find_chord_length3(v_i, v_j, cyc_edges)
        for (v_m1, v_m2) in chords:
            # Cycle A: 7 forward arc edges + 3 chord edges = 10 edges
            arc_A = [verts[(i + k) % 14] for k in range(8)]   # 8 verts, 7 edges
            cycle_A_seq = arc_A + [v_m2, v_m1, verts[i]]      # 8 + 3 = 11 verts, 10 edges
            if not is_nb_path(cycle_A_seq): continue

            # Cycle B: 7 backward arc edges (= forward starting from j) + 3 chord edges (reversed)
            arc_B = [verts[(j + k) % 14] for k in range(8)]   # 8 verts, 7 edges from v_j to v_i
            cycle_B_seq = arc_B + [v_m1, v_m2, verts[j]]      # 11 verts, 10 edges
            if not is_nb_path(cycle_B_seq): continue

            successes.append((i, j, v_m1, v_m2))
    return successes


if __name__ == '__main__':
    print("=" * 70)
    print("L=14 cycle decomposition test on H(srs)")
    print("=" * 70)
    print("  Testing: do L=14 NB cycles decompose as 2 girth-10 cycles")
    print("  sharing a 3-edge NB path (so 20 - 2·3 = 14)?")
    print()

    center = (N_SUPER // 2,) * 3
    all_14 = []
    t0 = time.time()
    for bond_idx in range(12):
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell): continue
        start = (prim_bond[0], center, prim_bond[1], tgt_cell)
        cycles = find_cycles_at_length(start, 14, max_cycles=200)
        for c in cycles:
            if len(c) == 14: all_14.append(c)
        if time.time() - t0 > 60:
            print(f"  (time budget hit at start {bond_idx})")
            break

    n14 = len(all_14)
    print(f"  Enumerated {n14} length-14 cycles in {time.time()-t0:.1f}s")
    print()

    n_decomp = 0
    n_undecomp = 0
    decomp_hist = Counter()
    for c in all_14:
        succ = check_L14_decomposition(c)
        decomp_hist[len(succ)] += 1
        if succ: n_decomp += 1
        else: n_undecomp += 1

    print(f"  RESULTS")
    print(f"  -------")
    print(f"  Total 14-cycles tested:     {n14}")
    print(f"  Decomposable (≥1 chord):    {n_decomp}  ({100*n_decomp/max(n14,1):.1f}%)")
    print(f"  Undecomposable (0 chords):  {n_undecomp}  ({100*n_undecomp/max(n14,1):.1f}%)")
    print()
    print(f"  Histogram of #valid (i, v_m1, v_m2) decompositions per 14-cycle:")
    for k in sorted(decomp_hist):
        print(f"    {k:3d} chord-decompositions: {decomp_hist[k]:5d} cycles")
    print()
    print(f"  STRUCTURAL READING")
    print(f"  ------------------")
    if n_undecomp == 0:
        print(f"  Every L=14 NB cycle on H(srs) decomposes as two girth-10 NB")
        print(f"  cycles glued by a 3-edge NB path (s=3 seam).")
        print()
        print(f"  Combined with hashimoto_16cycle_decomposition.py:")
        print(f"    L=10: irreducible girth (1 cycle)")
        print(f"    L=14: 2 girth cycles, s=3 seam ← would need n_fixed=3")
        print(f"    L=16: 2 girth cycles, s=2 seam ← V_ub host (n_fixed=2)")
        print()
        print(f"  Both L=14 and L=16 are 'two glued girth cycles' topologically,")
        print(f"  differing only by seam length s. The Feshbach principle as")
        print(f"  currently stated caps n_fixed in {{0,1,2}}, so:")
        print(f"    - n_fixed=2 ⇒ seam s=2 ⇒ L=16 (V_ub host)")
        print(f"    - n_fixed=3 would give L=14, but is structurally outside")
        print(f"      the Feshbach principle's domain.")
        print()
        print(f"  This explains why V_ub picks L=16, not L=14: the seam of")
        print(f"  the 'two glued cycles' topology must equal the n_fixed of")
        print(f"  the Feshbach pinning.")
    elif n_decomp == 0:
        print(f"  NO L=14 NB cycle on H(srs) admits a 3-edge-seam decomposition.")
        print(f"  L=14 is structurally NOT a 'two glued girth cycles' object.")
        print(f"  This explains the d=7 same-orbit hole and excludes L=14 as")
        print(f"  a V_ub candidate without invoking n_fixed cap.")
    else:
        print(f"  PARTIAL: {n_decomp}/{n14} = {100*n_decomp/max(n14,1):.1f}% of L=14")
        print(f"  cycles admit the 3-edge-seam decomposition. The picture")
        print(f"  is more nuanced than a clean dichotomy.")
