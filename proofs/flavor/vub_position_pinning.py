#!/usr/bin/env python3
"""
proofs/flavor/vub_position_pinning.py

PURPOSE
-------
V_cb pins the (b, c) causal-state pair at SAME-ORBIT positions (1, 2)
of an orbit (= C3=ω, C3=ω² eigenstates). Its CAS verification finds
20 such pairs at cycle-distance d=8 in girth-10 cycles.

The natural completion: s_u = position 0 of the same orbit
(= C3=trivial eigenstate). This is consistent with R3's
generation-Z₃ on C³_obs, and with the V_cb proof's identification
of s_b, s_c with C3 eigenstates.

CAS check the cycle-distance distributions for analogous pinnings:
- (s_u, s_c) = (pos 0, pos 1): for V_us-analog at Level 3
  (but V_us is actually at Level 2, so this is a sanity check)
- (s_u, s_b) = (pos 0, pos 2): for V_ub
- (s_c, s_b) = (pos 1, pos 2): for V_cb (already known)

If u-b pinning at (pos 0, pos 2) lacks d = g - n_fixed = 8 occurrences
in girth cycles, then V_ub's substrate amplitude is forced off the
m=1 (girth) host onto m≥2 multi-cycle hosts. That would be a
gate-passing structural derivation of the m=1 vs m≥2 split — closing
ADOPTED-VUB-MULTICYCLE-IDENTIFICATION.
"""

import sys, os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vcb_hashimoto_bfs as vcb

bonds_prim     = vcb.bonds_prim
N_SUPER        = vcb.N_SUPER
nb_successors  = vcb.nb_successors
in_bounds      = vcb.in_bounds
edge_prim_type = vcb.edge_prim_type
type_label     = vcb.type_label
g              = vcb.g
find_girth_cycles = vcb.find_girth_cycles
analyse_cycle  = vcb.analyse_cycle


def position_pair_distances(labels, pos_a, pos_b, g_val):
    """For each edge in cycle with position pos_a, find same-orbit
    edges with position pos_b appearing at forward distance fwd."""
    pairs = []
    for i, (oi, pi) in enumerate(labels):
        if pi != pos_a:
            continue
        for j, (oj, pj) in enumerate(labels):
            if oj != oi or pj != pos_b:
                continue
            fwd = (j - i) % g_val
            pairs.append(fwd)
    return pairs


def run_pinning_scan():
    center = (N_SUPER // 2,) * 3
    histograms = {}
    n_cycles = 0
    for bond_idx in range(12):
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell):
            continue
        start = (prim_bond[0], center, prim_bond[1], tgt_cell)
        cycles = find_girth_cycles(start, girth=g, max_cycles=20)
        for cyc in cycles:
            if len(cyc) != g:
                continue
            labels = analyse_cycle(cyc)
            if labels is None:
                continue
            n_cycles += 1
            for (pa, pb) in [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]:
                key = (pa, pb)
                if key not in histograms:
                    histograms[key] = Counter()
                for fwd in position_pair_distances(labels, pa, pb, g):
                    histograms[key][fwd] += 1
    return n_cycles, histograms


if __name__ == '__main__':
    print("=" * 70)
    print("Same-orbit position-pinning cycle-distance histograms on g=10")
    print("=" * 70)
    print()
    print("  Convention (per V_cb proof + V_cb identification):")
    print("    Position 0 = canonical bond, C3=trivial eigenstate of orbit")
    print("    Position 1 = C3·canonical, C3=ω eigenstate")
    print("    Position 2 = C3²·canonical, C3=ω² eigenstate")
    print()
    print("  V_cb identification (per vcb_nfixed_proof.py Step D):")
    print("    s_b ↔ position 1 (C3=ω)  [or position 2 — convention dependent]")
    print("    s_c ↔ position 2 (C3=ω²)")
    print("  The d=8 (b1, b2) at (pos 1, pos 2) pinning is the V_cb host.")
    print()
    print("  Natural u-quark identification (R3 + completion):")
    print("    s_u ↔ position 0 (C3=trivial)")
    print()

    n_cycles, hists = run_pinning_scan()
    print(f"  Girth cycles enumerated: {n_cycles}")
    print()

    L_cb = g - 2  # 8
    print(f"  V_cb's L_eff = g - n_fixed = {L_cb} corresponds to d_pin = {L_cb}")
    print(f"  (forward distance equal to L_eff = the longer-arc pinning)")
    print()

    print(f"  {'(pos_a, pos_b)':>14s}  {'d=':>5s}", end='')
    for d in range(g):
        print(f"  d={d}: ", end='')
    print()

    for (pa, pb) in [(1, 2), (2, 1), (0, 1), (1, 0), (0, 2), (2, 0)]:
        h = hists.get((pa, pb), Counter())
        total = sum(h.values())
        print(f"  ({pa},{pb}) total={total:4d}: ", end='')
        for d in range(g):
            cnt = h.get(d, 0)
            print(f"  {cnt:>4d}", end='')
        # Highlight d=L_cb
        cnt_at_Lcb = h.get(L_cb, 0)
        per_cycle = cnt_at_Lcb / max(n_cycles, 1)
        print(f"   d={L_cb}/cycle: {per_cycle:.4f}")

    print()
    print("  STRUCTURAL ANALYSIS:")
    print(f"  V_cb pinning (1, 2) at d={L_cb}: count =", hists.get((1, 2), Counter()).get(L_cb, 0),
          f"per cycle =", f"{hists.get((1, 2), Counter()).get(L_cb, 0) / max(n_cycles, 1):.4f}")
    print(f"  V_ub pinning (0, 2) at d={L_cb}: count =", hists.get((0, 2), Counter()).get(L_cb, 0),
          f"per cycle =", f"{hists.get((0, 2), Counter()).get(L_cb, 0) / max(n_cycles, 1):.4f}")
    print(f"  V_us pinning (0, 1) at d={L_cb}: count =", hists.get((0, 1), Counter()).get(L_cb, 0),
          f"per cycle =", f"{hists.get((0, 1), Counter()).get(L_cb, 0) / max(n_cycles, 1):.4f}")
    print()

    if hists.get((0, 2), Counter()).get(L_cb, 0) == 0:
        print(f"  ✓ V_ub pinning (pos 0, pos 2) has ZERO occurrences at d={L_cb}")
        print(f"     on girth cycles → m=1 host EXCLUDED for V_ub.")
        print(f"     Substrate forces V_ub onto m≥2 multi-cycle hosts.")
        print(f"     This is the missing structural argument for V_ub theorem.")
    elif hists.get((0, 2), Counter()).get(L_cb, 0) > 0:
        print(f"  ⚠ V_ub pinning (pos 0, pos 2) HAS occurrences at d={L_cb}")
        print(f"     on girth cycles → m=1 host NOT structurally excluded.")
        print(f"     V_ub's restriction to m≥2 needs a different structural argument.")
