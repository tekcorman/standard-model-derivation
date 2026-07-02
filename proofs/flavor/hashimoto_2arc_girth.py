#!/usr/bin/env python3
"""
proofs/flavor/hashimoto_2arc_girth.py

PURPOSE
-------
For each ordered 2-edge NB path P = (e_1, e_2) in H(srs), determine the
minimum length L_min(P) of a closed NB walk that contains P as a
contiguous subpath.

If L_min(P) = g = 10 for every 2-arc P, then srs is "girth-2-arc-uniform":
every 2-edge boundary lies on some girth cycle, and the Feshbach pinning
n_fixed=2 always reads the girth-cycle internal length L_eff = g − 2 = 8.
In that case V_cb's reading is universal and V_ub cannot arise from a
boundary placement choice.

If L_min(P) > g for some 2-arcs, then those 2-arcs are V_ub-style
boundaries: the minimum closed walk through them is a non-girth cycle
of length L_min > g, and the Feshbach reading gives L_eff = L_min − 2
> g − 2.

This is the load-bearing CAS check for G-Vub-2.

GATE STATUS
-----------
CAS verification only.
"""

import sys
import os
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vcb_hashimoto_bfs as vcb

bonds_prim     = vcb.bonds_prim
N_SUPER        = vcb.N_SUPER
nb_successors  = vcb.nb_successors
in_bounds      = vcb.in_bounds
edge_prim_type = vcb.edge_prim_type
type_label     = vcb.type_label


def find_cycle_through_2arc(e1, e2, L, max_cycles):
    """Find closed NB walks of length exactly L that start with the 2-arc
    (e1, e2). Returns count of such walks (not the walks themselves)."""
    found = 0
    path_set = {e1, e2}

    def dfs(current, depth):
        nonlocal found
        if found >= max_cycles:
            return
        if depth == L:
            for s in nb_successors(*current):
                if s == e1:
                    found += 1
                    return
            return
        for s in nb_successors(*current):
            if s == e1:
                if depth == L - 1:
                    found += 1
                continue
            if s in path_set:
                continue
            path_set.add(s)
            dfs(s, depth + 1)
            path_set.discard(s)

    dfs(e2, 2)
    return found


if __name__ == '__main__':
    print("=" * 70)
    print("Minimum-closure-length spectrum of 2-edge NB arcs on H(srs)")
    print("=" * 70)
    print()
    print("  For each 2-edge NB path P = (e_1, e_2), find the smallest L")
    print("  such that a closed NB walk of length L contains P contiguously.")
    print()

    center = (N_SUPER // 2,) * 3
    # Enumerate 2-arcs from a representative bond from each C3 orbit.
    # By symmetry, all bonds within one orbit are equivalent.
    arc_results = []
    visited_arc_keys = set()

    for bond_idx in range(12):
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell):
            continue
        e1 = (prim_bond[0], center, prim_bond[1], tgt_cell)
        # NB successors = 2 candidate e_2
        for e2 in nb_successors(*e1):
            # Classify e_1 and e_2 by orbit/position
            t1 = edge_prim_type(*e1)
            t2 = edge_prim_type(*e2)
            label1 = type_label[t1] if t1 is not None else None
            label2 = type_label[t2] if t2 is not None else None
            arc_key = (label1, label2)
            # Test minimum closure length
            L_min = None
            for L in (10, 14, 16, 18):
                count = find_cycle_through_2arc(e1, e2, L, max_cycles=20)
                if count > 0:
                    L_min = L
                    break
            arc_results.append((bond_idx, label1, label2, L_min))

    # Aggregate by arc_key (label1, label2)
    by_key = defaultdict(list)
    for (bi, l1, l2, lm) in arc_results:
        by_key[(l1, l2)].append(lm)

    print(f"  {'arc type':>30s}  {'L_min':>6s}  {'count':>6s}")
    print("  " + "-" * 50)
    for key in sorted(by_key):
        lmins = by_key[key]
        unique_L = sorted(set(lmins))
        for L in unique_L:
            cnt = lmins.count(L)
            print(f"  {str(key):>30s}  {L if L is not None else '∞':>6}  {cnt:>6d}")

    print()
    L_min_hist = Counter(lm for (_, _, _, lm) in arc_results)
    print("  Aggregate L_min histogram (over all 2-arcs from primitive bonds):")
    for L in sorted(L_min_hist):
        print(f"    L_min = {L if L is not None else '∞'}: {L_min_hist[L]:5d} 2-arcs")

    print()
    if all(lm == 10 for (_, _, _, lm) in arc_results):
        print("  CONCLUSION: every 2-arc lies on a girth cycle (L_min = 10 universal).")
        print("  → srs is girth-2-arc-uniform.")
        print("  → Feshbach n_fixed=2 boundary always reads L_eff = g − 2 = 8.")
        print("  → V_ub cannot arise from boundary-placement choice on the same graph;")
        print("     the L=16 host needs a structural input beyond '2-arc not on girth.'")
    else:
        nontrivial = [k for k in L_min_hist if k != 10]
        print(f"  CONCLUSION: some 2-arcs have L_min > g = 10. Specifically:")
        for L in sorted(nontrivial):
            print(f"    {L_min_hist[L]} 2-arcs with L_min = {L if L is not None else '∞'}")
        print()
        print("  This is the structural input for V_ub via boundary placement:")
        print("  the b→u 2-arc boundary is on a 2-arc with L_min > g, hosted by")
        print("  a non-girth cycle.")
