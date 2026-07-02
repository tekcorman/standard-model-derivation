#!/usr/bin/env python3
# ============================================================
# Bound-state existence probe — does MDL ever reward a compound
# closed walk over its constituents described independently?
# ============================================================
#
# Scoping context: docs/scoping/bound_state_sector_scoping_2026-05-28.md
# (action F1 of docs/scoping/frontier_synthesis_2026-05-28.md).
#
# THE QUESTION (the cheap, decisive first experiment):
#   Under the MDL waterline, is a COMPOUND closed walk (two girth cycles
#   sharing >=1 edge on srs) ever description-length-cheaper than its two
#   constituent cycles described INDEPENDENTLY?
#
# If yes -> the framework has a substrate-native notion of a bound state:
#   binding energy = OEF energy of the compression saving, E_bind = -kappa*dS,
#   dS = L(independent) - L(compound) = mutual information between constituents.
#   (Right sign: dS>0 => E_bind<0 => bound.)
# If no (dS<=0 for all overlaps) -> the framework has NO bound states;
#   composites are always "just their parts" (a strong, falsifiable statement).
#
# WHY THIS EVADES THE B_VD=0 NO-GO (H_multiway_construction.py):
#   That no-go kills binding via the canonical *dynamical* coupling (the
#   visible<->dark Schur block is identically zero). This probe gets binding,
#   if any, from the *description length* of the joint configuration — a
#   purely combinatorial/kinematic quantity, NOT a Hamiltonian matrix element.
#   The description-length route is the one the no-go leaves open.
#
# MDL CONVENTION (framework-native, stated explicitly, parameter-free):
#   The framework's per-non-backtracking-step description cost is
#   b_edge = log2(k*-1) = log2(2) = 1 bit. This is the SAME convention behind
#   alpha_1 = ((k-1)/k)^(g-2): the exponent (g-2) is the count of FREE NB
#   choices in a girth cycle (start reference + closure are forced).
#
#   - Describe two girth cycles INDEPENDENTLY ("sum of parts"): the shared
#     edges are specified TWICE (once in each cycle's NB trace).
#   - Describe the UNION as ONE compound object: each distinct edge specified
#     once; but each branch vertex (degree >= 3 in the union) now costs an
#     NB choice that was "forced" (degree 2) inside a pure cycle.
#   => dS = L(indep) - L(compound) = (s - n_branch) * b_edge
#      where s = # shared edges, n_branch = # vertices of degree >= 3 in the
#      union edge set. Binding clears the waterline iff dS > 0, i.e. s > n_branch.
#
#   For a single contiguous shared run of length s_run the union is a theta
#   graph (n_branch = 2), so dS = (s_run - 2) bits: binding iff s_run >= 3.
#
#   We ALSO report the parameter-free mutual information dS_MI = -log2 P(overlap)
#   (unambiguous; measures how informative "two cycles overlap" is) as a
#   convention-independent cross-check on the sign of the effect.
#
# Reuses the real srs graph + girth-cycle enumeration from srs_graph_analysis.py
# (build_supercell, enumerate_cycles_dfs). No new dynamics, no new axiom.

import os
import sys
import math
from collections import defaultdict
from itertools import combinations

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import srs_graph_analysis as srs

K_STAR = 3
GIRTH = 10
B_EDGE = math.log2(K_STAR - 1)          # = 1 bit, the framework's per-NB-step cost
B_EDGE_BM = math.log2(K_STAR / (K_STAR - 1))   # branch-measure variant = log2(3/2)


def cycle_edges(cycle):
    """Edge set of a cycle given as a tuple of vertex indices (closed)."""
    n = len(cycle)
    return frozenset(
        frozenset((cycle[i], cycle[(i + 1) % n])) for i in range(n)
    )


def collect_distinct_cycles(adjacency, vertices, length):
    """All distinct girth cycles through any vertex in `vertices`, deduped."""
    seen = set()
    for v in vertices:
        for cyc in srs.enumerate_cycles_dfs(adjacency, v, length):
            seen.add(cyc)
    return [tuple(c) for c in seen]


def union_branch_count(edges_a, edges_b):
    """# vertices of degree >= 3 in the union edge set (branch vertices)."""
    union = edges_a | edges_b
    deg = defaultdict(int)
    for e in union:
        for v in e:
            deg[v] += 1
    return sum(1 for v, d in deg.items() if d >= 3)


def shared_run_structure(shared_edges):
    """Connected-component analysis of the shared-edge subgraph.
    Returns (n_runs, max_run_len) — runs = contiguous shared paths."""
    if not shared_edges:
        return 0, 0
    # adjacency among shared edges (share a vertex)
    edges = list(shared_edges)
    vert_to_edges = defaultdict(list)
    for idx, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(idx)
    # union-find over edge indices
    parent = list(range(len(edges)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for v, eidxs in vert_to_edges.items():
        for a, b in combinations(eidxs, 2):
            union(a, b)
    comp_sizes = defaultdict(int)
    for idx in range(len(edges)):
        comp_sizes[find(idx)] += 1
    sizes = list(comp_sizes.values())
    return len(sizes), max(sizes)


def main():
    print("=" * 72)
    print("BOUND-STATE EXISTENCE PROBE — MDL compression of compound walks")
    print("Does describing two overlapping girth cycles as ONE object cost")
    print("fewer bits than describing them independently? (decides if the")
    print("framework has bound states AT ALL)")
    print("=" * 72)

    # ---- 1. Build the real srs graph + enumerate girth cycles ----
    n_cells = 3
    print(f"\n[1] Building {n_cells}^3 srs supercell and verifying...")
    positions, edges, adjacency, cell_indices = srs.build_supercell(n_cells)
    n_verts = len(positions)
    ok_deg3 = srs.verify_graph(positions, edges, adjacency)
    g = srs.find_girth(adjacency, n_verts, max_length=14)
    print(f"    girth = {g} (expected {GIRTH}); all degree 3: {ok_deg3}")
    if g != GIRTH:
        print("    ABORT: girth != 10, supercell too small / wrapped. ")
        return

    # distinct girth cycles across the whole supercell (torus, all vtx equivalent)
    print(f"\n[2] Enumerating distinct girth-{GIRTH} cycles over all {n_verts} vertices...")
    cycles = collect_distinct_cycles(adjacency, range(n_verts), GIRTH)
    print(f"    distinct girth cycles in supercell: {len(cycles)}")
    # sanity: cycles per vertex should be 15
    v0 = 0
    per_v0 = len(srs.enumerate_cycles_dfs(adjacency, v0, GIRTH))
    print(f"    cycles through vertex 0: {per_v0} (expected 15)")

    # edge sets + edge->cycle index
    edgesets = [cycle_edges(c) for c in cycles]
    edge_to_cyc = defaultdict(set)
    for ci, es in enumerate(edgesets):
        for e in es:
            edge_to_cyc[e].add(ci)

    # ---- 3. Overlapping-pair statistics ----
    print(f"\n[3] Computing shared-edge structure over all OVERLAPPING cycle pairs...")
    overlapping = set()
    for e, cs in edge_to_cyc.items():
        for a, b in combinations(sorted(cs), 2):
            overlapping.add((a, b))

    total_pairs = len(cycles) * (len(cycles) - 1) // 2
    n_overlap = len(overlapping)
    P_overlap = n_overlap / total_pairs if total_pairs else 0.0
    print(f"    total cycle pairs: {total_pairs}")
    print(f"    overlapping pairs (share >=1 edge): {n_overlap}")
    print(f"    P(overlap) = {P_overlap:.5f}  ->  MI of overlap = -log2 P = "
          f"{(-math.log2(P_overlap) if P_overlap>0 else float('inf')):.3f} bits")

    s_hist = defaultdict(int)          # shared-edge-count histogram
    run_hist = defaultdict(int)        # max contiguous run histogram
    branch_hist = defaultdict(int)     # n_branch histogram
    dS_hist = defaultdict(int)         # dS (minimal-overhead) histogram
    dS_values = []
    max_dS = -10**9
    max_dS_example = None
    s_run_max_global = 0

    for (a, b) in overlapping:
        ea, eb = edgesets[a], edgesets[b]
        shared = ea & eb
        s = len(shared)
        n_branch = union_branch_count(ea, eb)
        n_runs, s_run_max = shared_run_structure(shared)
        dS = (s - n_branch) * B_EDGE          # minimal-overhead reading, in bits
        s_hist[s] += 1
        run_hist[s_run_max] += 1
        branch_hist[n_branch] += 1
        dS_hist[round(dS)] += 1
        dS_values.append(dS)
        s_run_max_global = max(s_run_max_global, s_run_max)
        if dS > max_dS:
            max_dS = dS
            max_dS_example = (a, b, s, n_branch, n_runs, s_run_max)

    def hist_str(h):
        return ", ".join(f"{k}:{h[k]}" for k in sorted(h))

    print(f"\n    shared-edge count s  -> #pairs : {hist_str(s_hist)}")
    print(f"    max contiguous run   -> #pairs : {hist_str(run_hist)}")
    print(f"    branch vertices n_br -> #pairs : {hist_str(branch_hist)}")
    print(f"    longest shared contiguous run found anywhere: {s_run_max_global} edges")

    # ---- 4. MDL verdict ----
    print(f"\n[4] MDL compression dS = (s - n_branch) * b_edge  [b_edge = {B_EDGE:.3f} bit]")
    n_bind = sum(1 for d in dS_values if d > 1e-9)
    n_break = sum(1 for d in dS_values if abs(d) <= 1e-9)
    n_anti = sum(1 for d in dS_values if d < -1e-9)
    print(f"    overlapping pairs with dS > 0 (BIND, clear waterline): {n_bind}")
    print(f"    overlapping pairs with dS = 0 (break-even)          : {n_break}")
    print(f"    overlapping pairs with dS < 0 (anti-bind)           : {n_anti}")
    print(f"    dS histogram (bits -> #pairs): {hist_str(dS_hist)}")
    print(f"    max dS = {max_dS:.3f} bits", end="")
    if max_dS_example:
        a, b, s, nb, nr, srm = max_dS_example
        print(f"  (example pair: s={s} shared, n_branch={nb}, "
              f"n_runs={nr}, max_run={srm})")
    else:
        print()

    # branch-measure cross-check (same threshold, scaled cost)
    max_dS_bm = (max([ (len(edgesets[a]&edgesets[b]) - union_branch_count(edgesets[a],edgesets[b]))
                        for (a,b) in overlapping ], default=0)) * B_EDGE_BM
    print(f"    [branch-measure variant b_edge=log2(3/2): max dS = {max_dS_bm:.3f} bits]")

    # ---- 5. Verdict block ----
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if n_bind > 0:
        frac = n_bind / n_overlap
        print(f"  GREEN LIGHT: bound states EXIST in the framework.")
        print(f"  {n_bind}/{n_overlap} overlapping girth-cycle pairs ({100*frac:.1f}%) have")
        print(f"  a compound MDL description strictly shorter than independent parts.")
        print(f"  Max compression dS = {max_dS:.3f} bits -> binding energy")
        print(f"  E_bind = -kappa * dS  (kappa = k_B T ln2; physical scale = Stage 2).")
        print(f"  Mechanism: composite = compound closed walk; binding = mutual")
        print(f"  information of shared srs structure. The B_VD=0 no-go is evaded")
        print(f"  (this is description length, not a dynamical coupling).")
        print(f"  NEXT: Stage 1 (formalize E_bind=kappa*dS as a 2-subsystem OEF")
        print(f"  extension), then Stage 2 calibrate on deuteron / hydrogen.")
    else:
        print(f"  RED LIGHT: NO bound states. dS <= 0 for ALL {n_overlap} overlapping")
        print(f"  pairs -> MDL never rewards a compound over its parts. The framework")
        print(f"  would predict no nuclear/atomic binding via this mechanism (strong,")
        print(f"  falsifiable). Longest shared run = {s_run_max_global} edges; with")
        print(f"  n_branch overhead this never clears the waterline. The composite")
        print(f"  layer needs a DIFFERENT mechanism (or a named adoption).")
    print("=" * 72)


if __name__ == "__main__":
    main()
