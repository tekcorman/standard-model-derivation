#!/usr/bin/env python3
"""
proofs/flavor/hashimoto_24cycle_decomposition.py

EXTENSION of hashimoto_16cycle_decomposition.py to length 24.

PURPOSE
-------
Test the structural conjecture (Conjecture B of W11) that length-24 NB
cycles on H(srs) decompose as THREE girth-10 NB cycles glued at NB seams,
analogous to the m=2 closed-bubble decomposition at length 16.

If TRUE, this provides the substrate-side derivation of c_H^(α₁³) = α₁³
for Family-D Route C extension: at m=3 closed bubble with L_closed = 24,
joint NB walker survival = q_NB^24 = α₁_bare³.

COMBINATORIAL CONSTRAINT
------------------------
Three girth-10 NB cycles A, B, C with sharing structure (|A∩B|, |B∩C|, |A∩C|)
and triple intersection |A∩B∩C| = t.  The symmetric difference has
    |A Δ B Δ C| = 30 - 2·(s_AB + s_BC + s_AC) + 4·t.
For |Δ| = 24:  s_AB + s_BC + s_AC = 3 + 2t.

The cleanest topology candidates (t=0):
  (i)  (3, 0, 0): one pair shares 3 edges (a 3-edge NB path), other pairs disjoint
  (ii) (2, 1, 0): asymmetric chain
  (iii)(1, 1, 1): triangle, each pair shares 1 edge

VERDICT EXPECTED
----------------
- Best-case: every length-24 NB cycle admits at least one of these
  decompositions → m=3 closed-bubble structure confirmed
- Worst-case: zero 24-cycles decompose this way → Conjecture B falsified
  via Route C; Route H 3-way joint walker would need separate verification
- Middle: partial fraction decompose → structural reading needs refinement

METHODOLOGY
-----------
1. Use the same supercell + NB infrastructure as 16-cycle file
2. Enumerate length-24 NB cycles via DFS with time budget
3. For each 24-cycle, ENUMERATE girth-10 NB cycles that share a length-≤3
   sub-path with the 24-cycle
4. For each candidate triple (G_A, G_B, G_C), compute the symmetric
   difference and check if it equals the 24-cycle's edge set
5. Report: fraction decomposable, histogram of topology types
"""

import sys
import os
import time
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vcb_hashimoto_bfs as vcb

bonds_prim     = vcb.bonds_prim
prim_disps     = vcb.prim_disps
prim_type_key  = vcb.prim_type_key
type_label     = vcb.type_label
g              = vcb.g
N_SUPER        = vcb.N_SUPER
nb_successors  = vcb.nb_successors
edge_prim_type = vcb.edge_prim_type
in_bounds      = vcb.in_bounds


# ──────────────────────────────────────────────────────────────────────────────
# DFS for cycles of arbitrary length
# ──────────────────────────────────────────────────────────────────────────────

def find_cycles_at_length(start_edge, L, max_cycles, time_limit_s=None, t_start=None):
    """DFS from start_edge to find simple closed NB walks of length exactly L.
    Returns list of cycles (each as list of Hashimoto nodes)."""
    found = []
    path_set = {start_edge}

    def dfs(current, path, depth):
        if len(found) >= max_cycles:
            return
        if time_limit_s is not None and (time.time() - t_start) > time_limit_s:
            return
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
            if succ in path_set:
                continue
            path_set.add(succ)
            path.append(succ)
            dfs(succ, path, depth + 1)
            path.pop()
            path_set.discard(succ)

    dfs(start_edge, [start_edge], 1)
    return found


def cycle_edge_set_directed(cycle):
    """Directed-edge set of cycle (frozenset of Hashimoto-node tuples)."""
    return frozenset(cycle)


def cycle_edge_set_undirected(cycle):
    """Undirected-edge set of cycle (frozenset of frozensets of (atom, cell))."""
    edges = set()
    for (s, c, t, c2) in cycle:
        e = frozenset({(s, c), (t, c2)})
        edges.add(e)
    return frozenset(edges)


# ──────────────────────────────────────────────────────────────────────────────
# Decomposition test: 24-cycle = A Δ B Δ C with three girth-10 cycles
# ──────────────────────────────────────────────────────────────────────────────

def find_all_girth_cycles_in_supercell(time_limit_s=60.0):
    """Enumerate all girth-10 NB cycles in the supercell.
    Returns set of frozensets of directed edges (Hashimoto cycles up to cyclic rotation)."""
    t_start = time.time()
    all_girth_directed = set()        # frozen directed-edge tuples (canonical rotation)
    all_girth_undirected = []          # list of undirected-edge frozensets

    for src_atom in range(4):
        for cell in [(N_SUPER//2, N_SUPER//2, N_SUPER//2),
                     (N_SUPER//2 + 1, N_SUPER//2, N_SUPER//2),
                     (N_SUPER//2, N_SUPER//2 + 1, N_SUPER//2)]:
            for (s, t, dc) in bonds_prim:
                if s != src_atom:
                    continue
                nc = tuple(cell[d] + dc[d] for d in range(3))
                if not in_bounds(nc):
                    continue
                start = (s, cell, t, nc)
                cycles = find_cycles_at_length(start, 10, max_cycles=200,
                                                time_limit_s=time_limit_s, t_start=t_start)
                for c in cycles:
                    # Canonicalize: rotate so minimum element is first
                    min_idx = min(range(10), key=lambda i: c[i])
                    canonical = tuple(c[(min_idx + i) % 10] for i in range(10))
                    all_girth_directed.add(canonical)
                if time.time() - t_start > time_limit_s:
                    return all_girth_directed
    return all_girth_directed


def directed_edge_to_undirected(de):
    """Convert directed edge (s,c,t,c2) → frozenset{(s,c),(t,c2)}"""
    s, c, t, c2 = de
    return frozenset({(s, c), (t, c2)})


def cycle_undirected_edges(cycle_directed):
    """Convert directed cycle to undirected edge set."""
    return frozenset(directed_edge_to_undirected(de) for de in cycle_directed)


def test_triple_decomposition(cycle24_directed, girth_cycles_directed):
    """For a given length-24 directed-edge cycle, try to find three girth-10
    cycles whose symmetric difference equals the 24-cycle.

    Returns list of triples (g_A, g_B, g_C) that decompose the 24-cycle.
    """
    cycle24_undir = cycle_undirected_edges(cycle24_directed)
    if len(cycle24_undir) != 24:
        # Cycle is degenerate (repeats undirected edges)
        return []

    # Convert all girth cycles to undirected edge sets
    girth_undir = [(g, cycle_undirected_edges(g)) for g in girth_cycles_directed]

    # Filter to girth cycles that share at least 1 edge with cycle24
    relevant_girth = [(g, u) for (g, u) in girth_undir if len(u & cycle24_undir) > 0]

    # Try triples
    decompositions = []
    n = len(relevant_girth)
    for i in range(n):
        gA, uA = relevant_girth[i]
        # Edges in A that ARE in cycle24:
        # we need: A Δ B Δ C = cycle24
        # i.e., (A ∪ B ∪ C) \ (A∩B ∪ A∩C ∪ B∩C ∪ A∩B∩C) = cycle24
        # Simpler: iterate (j, k) and check (uA ^ uB ^ uC) == cycle24_undir
        for j in range(i+1, n):
            gB, uB = relevant_girth[j]
            partial = uA ^ uB  # A Δ B
            for k in range(j+1, n):
                gC, uC = relevant_girth[k]
                triple_sym_diff = partial ^ uC  # A Δ B Δ C
                if triple_sym_diff == cycle24_undir:
                    decompositions.append((gA, gB, gC))
    return decompositions


def topology_type(triple, cycle24_undir):
    """Classify the topology of a decomposition triple (A, B, C).
    Returns (s_AB, s_BC, s_AC, s_ABC) as edge-share counts."""
    uA = cycle_undirected_edges(triple[0])
    uB = cycle_undirected_edges(triple[1])
    uC = cycle_undirected_edges(triple[2])
    s_AB = len(uA & uB)
    s_BC = len(uB & uC)
    s_AC = len(uA & uC)
    s_ABC = len(uA & uB & uC)
    return (s_AB, s_BC, s_AC, s_ABC)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 76)
    print("24-cycle decomposition test on H(srs)")
    print("=" * 76)
    print("  Testing Conjecture B (W11): every length-24 NB cycle on H(srs)")
    print("  decomposes as three girth-10 NB cycles via symmetric difference,")
    print("  giving substrate-side derivation of c_H^(α₁³) = α₁³ for Family-D")
    print("  Route C extension at m=3 (analog of 16-cycle = m=2).")
    print()
    print(f"  Supercell: N_SUPER = {N_SUPER}^3 = {N_SUPER**3} primitive cells")
    print(f"  Total directed edges in supercell: {N_SUPER**3 * 12}")
    print()

    # Step 1: Enumerate girth-10 cycles
    print("STEP 1: Enumerate girth-10 cycles on H(srs) ...")
    t0 = time.time()
    girth_cycles = find_all_girth_cycles_in_supercell(time_limit_s=60.0)
    print(f"  Found {len(girth_cycles)} distinct girth-10 directed cycles in {time.time()-t0:.1f}s")
    print()

    # Step 2: Enumerate 24-cycles
    print("STEP 2: Enumerate length-24 NB cycles on H(srs) ...")
    center = (N_SUPER // 2,) * 3
    all_24 = []
    seen_canonical = set()
    t0 = time.time()
    time_budget_24 = 240  # 4 minutes
    for bond_idx in range(12):
        if time.time() - t0 > time_budget_24:
            print(f"  (time budget {time_budget_24}s hit at bond_idx {bond_idx})")
            break
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell):
            continue
        start = (prim_bond[0], center, prim_bond[1], tgt_cell)
        cycles = find_cycles_at_length(start, 24, max_cycles=300,
                                        time_limit_s=time_budget_24, t_start=t0)
        for c in cycles:
            if len(c) == 24:
                # Canonicalize for dedup
                min_idx = min(range(24), key=lambda i: c[i])
                canonical = tuple(c[(min_idx + i) % 24] for i in range(24))
                if canonical not in seen_canonical:
                    seen_canonical.add(canonical)
                    all_24.append(c)
    n24 = len(all_24)
    print(f"  Found {n24} distinct length-24 cycles in {time.time()-t0:.1f}s")
    print()

    if n24 == 0:
        print("  ⚠ No 24-cycles found within time budget. Abort.")
        sys.exit(1)

    # Step 3: For each 24-cycle, test triple decomposition
    print(f"STEP 3: Test triple-girth-cycle decomposition for each 24-cycle ...")
    t0 = time.time()
    n_decomp = 0
    n_undecomp = 0
    n_degenerate = 0
    topology_hist = Counter()
    sample_decomp = None
    sample_undecomp = None
    test_budget = 240
    n_tested = 0

    for ci, c24 in enumerate(all_24):
        if time.time() - t0 > test_budget:
            print(f"  (test budget {test_budget}s hit after {n_tested} cycles)")
            break
        n_tested += 1
        c24_undir = cycle_undirected_edges(c24)
        if len(c24_undir) != 24:
            n_degenerate += 1
            continue

        decomps = test_triple_decomposition(c24, list(girth_cycles))
        if decomps:
            n_decomp += 1
            for triple in decomps:
                topo = topology_type(triple, c24_undir)
                topology_hist[topo] += 1
            if sample_decomp is None:
                sample_decomp = (c24, decomps[0])
        else:
            n_undecomp += 1
            if sample_undecomp is None:
                sample_undecomp = c24

    print(f"  Tested {n_tested}/{n24} cycles in {time.time()-t0:.1f}s")
    print()

    # Results
    print("=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(f"  Total 24-cycles enumerated:       {n24}")
    print(f"  Tested:                            {n_tested}")
    print(f"  Decomposable (≥1 triple):          {n_decomp}  ({100.0*n_decomp/max(n_tested,1):.1f}%)")
    print(f"  Undecomposable (0 triples):        {n_undecomp}  ({100.0*n_undecomp/max(n_tested,1):.1f}%)")
    print(f"  Degenerate (repeated edges):       {n_degenerate}")
    print()
    print(f"  Topology histogram (s_AB, s_BC, s_AC, s_ABC):")
    for topo, count in sorted(topology_hist.items(), key=lambda x: -x[1])[:10]:
        s_total = sum(topo[:3])
        s_triple = topo[3]
        constraint = 30 - 2*s_total + 4*s_triple
        print(f"    {topo}: {count} occurrences  | check: 30 - 2·{s_total} + 4·{s_triple} = {constraint}")
    print()

    print("STRUCTURAL READING")
    print("------------------")
    if n_undecomp == 0 and n_tested > 0:
        print("  Every tested 24-cycle decomposes as three girth-10 cycles.")
        print("  Conjecture B (Route C extension at m=3) → CONFIRMED for tested cycles.")
        print("  → c_H^(α₁³) = q_NB^24 = α₁_bare³ at theorem-grade-conditional")
        print("    (conditional on full enumeration sufficiency).")
    elif n_decomp > 0:
        frac = 100.0 * n_decomp / max(n_tested, 1)
        print(f"  {frac:.1f}% of tested 24-cycles admit triple-girth decomposition.")
        if frac >= 95.0:
            print(f"  → Conjecture B PROBABLY HOLDS — exceptions are likely combinatorial edge cases.")
        elif frac >= 50.0:
            print(f"  → Conjecture B PARTIAL — needs refinement of topology hypotheses.")
        else:
            print(f"  → Conjecture B QUESTIONABLE — only a minority decompose; mechanism")
            print(f"    differs from simple m=2 extension.")
    else:
        print(f"  Zero 24-cycles decompose as three glued girth cycles.")
        print(f"  → Conjecture B FALSIFIED via Route C extension.")
        print(f"    Route H 3-way joint walker would need separate verification.")
    print()
