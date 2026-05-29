#!/usr/bin/env python3
"""
proofs/flavor/vub_endpoint_class_check.py

PURPOSE
-------
Test whether m=1 (girth) host cycles and m=2 (length-16) host cycles on
H(srs) pin DIFFERENT causal-state endpoint pairs.

Per V_cb derivation (vcb_hashimoto_bfs.py + vcb_nfixed_proof.py), the
m=1 host's same-orbit (b1, b2) endpoint pair at cycle-distance d=8 IS
the V_cb pinning. The endpoints are b1 = orbit-position-1 and b2 =
orbit-position-2 within the SAME C3 orbit.

If m=2 hosts pin a STRUCTURALLY DIFFERENT endpoint pair (e.g.,
cross-orbit, or different position-difference, or different cycle-
distance), then by A5(b) Case B (theorem_A5b_level_prescription.md
§3.2), the m=1 and m=2 hosts contribute to DIFFERENT couplings
(V_cb vs V_ub respectively).

If m=2 hosts pin the SAME endpoint pair as m=1, both contribute to V_cb
and the m=1-only assignment is wrong.

This is the load-bearing structural step for closing G-Vub-1.

GATE STATUS
-----------
CAS verification only.
"""

import sys, os
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vcb_hashimoto_bfs as vcb

bonds_prim     = vcb.bonds_prim
N_SUPER        = vcb.N_SUPER
nb_successors  = vcb.nb_successors
in_bounds      = vcb.in_bounds
edge_prim_type = vcb.edge_prim_type
type_label     = vcb.type_label
g              = vcb.g     # 10


def find_cycles_at_length(start_edge, L, max_cycles):
    found = []; path_set = {start_edge}
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


def label_cycle(cycle):
    out = []
    for e in cycle:
        pt = edge_prim_type(*e)
        if pt is None: return None
        out.append(type_label[pt])
    return out


def endpoint_pair_classes(cycle, L):
    """For each ordered pair (i, j) in the cycle with i≠j, classify by
    (orbit_match, position_match, cycle_distance).
    Return Counter keyed by (kind_str, distance).

    'kind' values:
      'same-orbit-same-pos': (oi==oj, pi==pj) — same edge type
      'same-orbit-diff-pos':  (oi==oj, pi!=pj) — V_cb-type pinning
      'diff-orbit-same-pos': (oi!=oj, pi==pj)
      'diff-orbit-diff-pos': (oi!=oj, pi!=pj)
    """
    labels = label_cycle(cycle)
    if labels is None: return None
    counts = Counter()
    for i in range(L):
        oi, pi = labels[i]
        for j in range(L):
            if i == j: continue
            oj, pj = labels[j]
            d = (j - i) % L
            if oi == oj:
                kind = 'same-orbit-same-pos' if pi == pj else 'same-orbit-diff-pos'
            else:
                kind = 'diff-orbit-same-pos' if pi == pj else 'diff-orbit-diff-pos'
            counts[(kind, d)] += 1
    return counts


def scan_m_host(L, n_starts=12, max_cycles=200, max_time_s=120):
    center = (N_SUPER // 2,) * 3
    aggregate = Counter()
    n_cycles = 0
    t0 = time.time()
    for bond_idx in range(min(n_starts, 12)):
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell): continue
        start = (prim_bond[0], center, prim_bond[1], tgt_cell)
        if time.time() - t0 > max_time_s: break
        cycles = find_cycles_at_length(start, L, max_cycles)
        for c in cycles:
            cnts = endpoint_pair_classes(c, L)
            if cnts is None: continue
            n_cycles += 1
            aggregate.update(cnts)
    return n_cycles, aggregate


if __name__ == '__main__':
    print("=" * 70)
    print("Endpoint-pair class spectrum on m=1 (L=10) vs m=2 (L=16) hosts")
    print("=" * 70)
    print()

    L_cb = 10   # m=1 host (girth)
    L_ub = 16   # m=2 host

    n10, agg10 = scan_m_host(L_cb, n_starts=12, max_cycles=200)
    n16, agg16 = scan_m_host(L_ub, n_starts=12, max_cycles=200)

    print(f"L={L_cb} (m=1 host): {n10} cycles enumerated")
    print(f"L={L_ub} (m=2 host): {n16} cycles enumerated")
    print()

    # Focus on V_cb-style pinning: same-orbit-diff-pos at d=8 (= L_cb - n_fixed)
    print(f"  V_cb pinning class is same-orbit-diff-pos at d = L_cb - n_fixed = {L_cb - 2} = 8")
    print(f"    On m=1 host: count = {agg10.get(('same-orbit-diff-pos', 8), 0)}")
    print(f"    Per cycle:   {agg10.get(('same-orbit-diff-pos', 8), 0) / max(n10, 1):.4f}")
    print()
    # On m=2 host, the analogous V_ub pinning at d = L_ub - n_fixed = 14
    print(f"  V_ub pinning class is same-orbit-diff-pos at d = L_ub - n_fixed = {L_ub - 2} = 14")
    print(f"    On m=2 host: count = {agg16.get(('same-orbit-diff-pos', 14), 0)}")
    print(f"    Per cycle:   {agg16.get(('same-orbit-diff-pos', 14), 0) / max(n16, 1):.4f}")
    print()

    # CRITICAL: does the m=2 host ALSO contain V_cb-style pinning at d=8?
    print(f"  CRITICAL: V_cb-style pinning (same-orbit-diff-pos at d=8) on m=2 host?")
    cnt_cb_on_m2 = agg16.get(('same-orbit-diff-pos', 8), 0)
    print(f"    On m=2 host (L=16): count at d=8 = {cnt_cb_on_m2}")
    print(f"    Per cycle:                       {cnt_cb_on_m2 / max(n16, 1):.4f}")
    print()
    # And the V_ub pinning at d=14 on m=1 host?
    print(f"  V_ub-style pinning (same-orbit-diff-pos at d=14) on m=1 host?")
    cnt_ub_on_m1 = agg10.get(('same-orbit-diff-pos', 14), 0)
    print(f"    On m=1 host (L=10): count at d=14 = {cnt_ub_on_m1}  (impossible since L=10 < 14)")
    print()

    # Full distance spectra for same-orbit-diff-pos
    print(f"  Full same-orbit-diff-pos distance histogram:")
    print(f"  {'d':>4s} {'L=10 count':>12s} {'L=16 count':>12s}")
    all_distances = sorted({d for (kind, d) in agg10 if kind == 'same-orbit-diff-pos'} |
                          {d for (kind, d) in agg16 if kind == 'same-orbit-diff-pos'})
    for d in all_distances:
        c10 = agg10.get(('same-orbit-diff-pos', d), 0)
        c16 = agg16.get(('same-orbit-diff-pos', d), 0)
        print(f"  {d:>4d} {c10:>12d} {c16:>12d}")

    print()
    print("  ANALYSIS")
    print("  --------")
    if cnt_cb_on_m2 > 0:
        per_cycle_m1_cb = agg10.get(('same-orbit-diff-pos', 8), 0) / max(n10, 1)
        per_cycle_m2_cb = cnt_cb_on_m2 / max(n16, 1)
        print(f"  ⚠ m=2 host CONTAINS V_cb-style pinning (same-orbit-diff-pos at d=8).")
        print(f"     Per cycle: m=1 host = {per_cycle_m1_cb:.4f},  m=2 host = {per_cycle_m2_cb:.4f}")
        print(f"     This means m=2 hosts contribute to BOTH V_cb-pinning AND V_ub-pinning.")
        print(f"     The 'A5(b) Case B groups walks by endpoint pair' argument needs")
        print(f"     to be sharpened: the GLOBAL TOPOLOGY of the host (m=2 vs m=1)")
        print(f"     must be a separate distinguishing feature, not just the local pinning.")
    else:
        print(f"  ✓ m=2 host does NOT contain V_cb-style pinning at d=8.")
        print(f"     m=1 and m=2 hosts pin DISJOINT causal-state endpoint pairs.")
        print(f"     A5(b) Case B then assigns m=1 hosts to V_cb and m=2 hosts to V_ub")
        print(f"     by endpoint structure alone — no further structural input needed.")
