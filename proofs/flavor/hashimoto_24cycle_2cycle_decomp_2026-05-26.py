#!/usr/bin/env python3
"""
proofs/flavor/hashimoto_24cycle_2cycle_decomp_2026-05-26.py

Follow-up to hashimoto_24cycle_decomposition.py (W12 result: 12.2%
decomposition rate for triple-girth, FALSIFIED).

Test 2-CYCLE decompositions of length-24 NB cycles on H(srs):

  L_A + L_B - 2s = 24, with s ≥ 1 (connected sharing) and L_A, L_B
  being lengths of established short NB cycles on H(srs).

Candidate configurations:
  (a) 24 = 10 + 16 - 2   (girth + 16-cycle, seam s=1)
  (b) 24 = 14 + 14 - 4   (two 14-cycles, seam s=2)
  (c) 24 = 14 + 16 - 6   (14-cycle + 16-cycle, seam s=3)
  (d) 24 = 16 + 16 - 8   (two 16-cycles, seam s=4)

(Note: "girth + 14-cycle" = 10 + 14 = 24 - 2s requires s=0 which is
topologically invalid — disjoint cycles don't form a single closed walk.)

For each 24-cycle, test which (if any) of (a)-(d) decompositions exist.
A 100% rate would be analogous to the m=2 closed-bubble result and
support a different α₁³ mechanism.
"""

import sys, os, time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vcb_hashimoto_bfs as vcb
bonds_prim    = vcb.bonds_prim
N_SUPER       = vcb.N_SUPER
nb_successors = vcb.nb_successors
in_bounds     = vcb.in_bounds


def find_cycles_at_length(start_edge, L, max_cycles, time_limit_s=None, t_start=None):
    found = []
    path_set = {start_edge}

    def dfs(current, path, depth):
        if len(found) >= max_cycles: return
        if time_limit_s is not None and (time.time() - t_start) > time_limit_s: return
        if depth == L:
            for succ in nb_successors(*current):
                if succ == start_edge:
                    found.append(list(path))
                    return
            return
        for succ in nb_successors(*current):
            if succ == start_edge:
                if depth == L - 1:
                    found.append(list(path))
                continue
            if succ in path_set: continue
            path_set.add(succ)
            path.append(succ)
            dfs(succ, path, depth + 1)
            path.pop()
            path_set.discard(succ)

    dfs(start_edge, [start_edge], 1)
    return found


def directed_edge_to_undirected(de):
    s, c, t, c2 = de
    return frozenset({(s, c), (t, c2)})


def cycle_undirected_edges(cycle):
    return frozenset(directed_edge_to_undirected(de) for de in cycle)


def enumerate_cycles_at_length(L, time_limit_s=120):
    """Enumerate distinct cycles of given length from center cells."""
    t_start = time.time()
    center = (N_SUPER // 2,) * 3
    cycles_by_undir = {}   # undirected edge set → first directed representative
    for src_cell in [center,
                     tuple(center[i] + (1 if i == 0 else 0) for i in range(3)),
                     tuple(center[i] + (1 if i == 1 else 0) for i in range(3)),
                     tuple(center[i] + (1 if i == 2 else 0) for i in range(3))]:
        for (s, t, dc) in bonds_prim:
            nc = tuple(src_cell[d] + dc[d] for d in range(3))
            if not in_bounds(nc): continue
            start = (s, src_cell, t, nc)
            cycles = find_cycles_at_length(start, L, max_cycles=500,
                                           time_limit_s=time_limit_s, t_start=t_start)
            for c in cycles:
                u = cycle_undirected_edges(c)
                if u not in cycles_by_undir:
                    cycles_by_undir[u] = c
            if time.time() - t_start > time_limit_s:
                return list(cycles_by_undir.keys()), list(cycles_by_undir.values())
    return list(cycles_by_undir.keys()), list(cycles_by_undir.values())


def test_2cycle_decomposition(cycle24_undir, cycles_A_undir, cycles_B_undir, L_A, L_B, target_s):
    """Test whether the given 24-cycle decomposes as A Δ B where A has
    L_A edges, B has L_B edges, sharing target_s edges."""
    decomps = []
    for uA in cycles_A_undir:
        # Check if uA shares edges with cycle24
        n_shared_with_24 = len(uA & cycle24_undir)
        # In sym diff: edges in uA but not in 24 must be in uB (and equal s)
        # edges in 24 but not in uA must be the rest of uB
        # Specifically: 24 Δ A = B (if A Δ B = 24)
        # So B = uA Δ cycle24_undir
        uB_candidate = uA ^ cycle24_undir
        if len(uB_candidate) != L_B: continue
        # Check sharing: |uA ∩ uB| = target_s
        shared = len(uA & uB_candidate)
        if shared != target_s: continue
        # Check uB_candidate is a valid B-length cycle in our set
        if uB_candidate in cycles_B_undir:
            decomps.append((uA, uB_candidate))
    return decomps


if __name__ == '__main__':
    print("=" * 78)
    print("24-cycle 2-cycle decomposition tests on H(srs)")
    print("=" * 78)
    print()
    print("Test candidate configurations:")
    print("  (a) 24 = 10 + 16 - 2  (girth + 16-cycle, seam s=1)")
    print("  (b) 24 = 14 + 14 - 4  (two 14-cycles, seam s=2)")
    print("  (c) 24 = 14 + 16 - 6  (14 + 16-cycle, seam s=3)")
    print("  (d) 24 = 16 + 16 - 8  (two 16-cycles, seam s=4)")
    print()

    # Step 1: enumerate cycles of lengths 10, 14, 16, 24
    t0 = time.time()
    print("STEP 1: enumerate cycles of lengths 10, 14, 16, 24 ...")
    cyc10_undir, _ = enumerate_cycles_at_length(10, time_limit_s=10)
    print(f"  L=10: {len(cyc10_undir)} distinct undirected girth cycles ({time.time()-t0:.1f}s)")
    cyc14_undir, _ = enumerate_cycles_at_length(14, time_limit_s=30)
    print(f"  L=14: {len(cyc14_undir)} distinct undirected 14-cycles ({time.time()-t0:.1f}s)")
    cyc16_undir, _ = enumerate_cycles_at_length(16, time_limit_s=60)
    print(f"  L=16: {len(cyc16_undir)} distinct undirected 16-cycles ({time.time()-t0:.1f}s)")
    cyc24_undir, _ = enumerate_cycles_at_length(24, time_limit_s=120)
    print(f"  L=24: {len(cyc24_undir)} distinct undirected 24-cycles ({time.time()-t0:.1f}s)")
    print()

    # Wrap into sets for fast membership
    set10 = set(cyc10_undir)
    set14 = set(cyc14_undir)
    set16 = set(cyc16_undir)

    n_24 = len(cyc24_undir)
    n_decomp_a = 0   # girth + 16-cycle, s=1
    n_decomp_b = 0   # two 14-cycles, s=2
    n_decomp_c = 0   # 14 + 16, s=3
    n_decomp_d = 0   # two 16-cycles, s=4
    n_any = 0

    print("STEP 2: test each 24-cycle for each candidate decomposition ...")
    t0 = time.time()
    test_budget = 240
    n_tested = 0
    for i, c24_u in enumerate(cyc24_undir):
        if time.time() - t0 > test_budget:
            print(f"  (test budget {test_budget}s hit at {n_tested} cycles)")
            break
        n_tested += 1

        # (a) girth + 16: target_s = 1
        a_decomps = test_2cycle_decomposition(c24_u, cyc10_undir, set16, 10, 16, 1)
        # (b) 14 + 14: target_s = 2
        b_decomps = test_2cycle_decomposition(c24_u, cyc14_undir, set14, 14, 14, 2)
        # (c) 14 + 16: target_s = 3
        c_decomps = test_2cycle_decomposition(c24_u, cyc14_undir, set16, 14, 16, 3)
        # (d) 16 + 16: target_s = 4
        d_decomps = test_2cycle_decomposition(c24_u, cyc16_undir, set16, 16, 16, 4)

        if a_decomps: n_decomp_a += 1
        if b_decomps: n_decomp_b += 1
        if c_decomps: n_decomp_c += 1
        if d_decomps: n_decomp_d += 1
        if a_decomps or b_decomps or c_decomps or d_decomps:
            n_any += 1

    print(f"  Tested {n_tested}/{n_24} cycles in {time.time()-t0:.1f}s")
    print()

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print()
    print(f"  Total 24-cycles enumerated:           {n_24}")
    print(f"  Tested:                                {n_tested}")
    print()
    print(f"  (a) girth + 16, s=1:  {n_decomp_a:5d}  ({100.0*n_decomp_a/max(n_tested,1):.1f}%)")
    print(f"  (b) 14 + 14, s=2:     {n_decomp_b:5d}  ({100.0*n_decomp_b/max(n_tested,1):.1f}%)")
    print(f"  (c) 14 + 16, s=3:     {n_decomp_c:5d}  ({100.0*n_decomp_c/max(n_tested,1):.1f}%)")
    print(f"  (d) 16 + 16, s=4:     {n_decomp_d:5d}  ({100.0*n_decomp_d/max(n_tested,1):.1f}%)")
    print(f"  ANY of (a)-(d):        {n_any:5d}  ({100.0*n_any/max(n_tested,1):.1f}%)")
    print()

    print("STRUCTURAL READING")
    print("-" * 40)
    if 100.0*n_any/max(n_tested,1) >= 95:
        print("  → 24-cycle space decomposes via 2-cycle compositions of")
        print("    {girth, 14-cycle, 16-cycle} at sub-percent precision.")
        print("    This SUPPORTS an α₁³ mechanism via composite-cycle structure.")
    elif 100.0*n_any/max(n_tested,1) >= 50:
        print("  → Partial decomposition (majority but not all).")
        print("    The 2-cycle decompositions capture significant 24-cycle structure,")
        print("    but a complete mechanism requires additional topologies.")
    else:
        print("  → Most 24-cycles don't decompose via these candidates.")
        print("    Suggests further decomposition types needed (e.g., 3-cycle")
        print("    asymmetric chains, longer-cycle compositions).")
