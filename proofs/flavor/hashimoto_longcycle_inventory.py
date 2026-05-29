#!/usr/bin/env python3
"""
proofs/flavor/hashimoto_longcycle_inventory.py

FRAMEWORK LEVEL: Level 3 (Hashimoto graph = causal observer graph).

PURPOSE
-------
V_cb closure used same-orbit (b1, b2) pairs at cycle-distance d=8 within
girth-10 cycles. The Feshbach Exponent Principle as currently written
caps n_fixed in {0,1,2}, so on H(srs) at girth g=10 the only L values
admitted are {8, 9, 10}. None of those reproduce V_ub ≈ 3.7e-3.

This script asks: do there exist NON-girth (longer) NB cycles on H(srs)
that admit some pair-distance giving V = (2/3)^L_eff / (1−(2/3)^L_eff)
in the V_ub window? Concretely: enumerate closed NB walks of length
12, 14, 16 (and possibly 18) and tabulate same-orbit (b1, b2)
cycle-distance distributions per cycle-length.

If a longer-cycle distance gives V_ub, that is a substrate-level
candidate channel for V_ub, with the structural question reduced to
"what selects this particular cycle length / pair type for the b→u
process?" That's a separate gate — but at least there's a candidate.

GATE STATUS
-----------
This script is CAS INVENTORY only. It does not assign physical species
to orbit/position labels and does not file a prediction. The output is
a target sheet for any future V_ub closure attempt.
"""

import sys
import os
import time
from fractions import Fraction
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

# Import lattice + bond infrastructure from V_cb's BFS file
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

# Reuse the BFS module's structures by import (it builds at module load).
import vcb_hashimoto_bfs as vcb

bonds_prim    = vcb.bonds_prim
prim_disps    = vcb.prim_disps
prim_type_key = vcb.prim_type_key
type_label    = vcb.type_label
g             = vcb.g                 # 10
N_SUPER       = vcb.N_SUPER           # 8
nb_successors = vcb.nb_successors
edge_prim_type= vcb.edge_prim_type
in_bounds     = vcb.in_bounds


# ──────────────────────────────────────────────────────────────────────────────
# Cycle finder for arbitrary length L (uses same DFS structure as girth case)
# ──────────────────────────────────────────────────────────────────────────────

def find_cycles_at_length(start_edge, L, max_cycles):
    """DFS for simple NB cycles of length exactly L through start_edge."""
    found = []
    path_set = {start_edge}

    def dfs(current, path, depth):
        if len(found) >= max_cycles:
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


# ──────────────────────────────────────────────────────────────────────────────
# Pair-distance enumeration on a cycle of length L
# ──────────────────────────────────────────────────────────────────────────────

def label_cycle(cycle):
    out = []
    for e in cycle:
        pt = edge_prim_type(*e)
        if pt is None:
            return None
        out.append(type_label[pt])
    return out


def same_orbit_pair_dists(labels, L):
    """For each ordered pair (i, j) with i != j on the cycle, if labels[i]
    and labels[j] are same orbit but different positions (b1, b2 of
    the V_cb pinning), record the forward distance (j - i) % L."""
    dists = Counter()
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            oi, pi = labels[i]
            oj, pj = labels[j]
            if oi == oj and pi != pj:
                dists[(j - i) % L] += 1
    return dists


def cross_orbit_pair_dists(labels, L):
    """Cross-orbit (different orbit) pair distances on the cycle."""
    dists = Counter()
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            oi, pi = labels[i]
            oj, pj = labels[j]
            if oi != oj:
                dists[(j - i) % L] += 1
    return dists


# ──────────────────────────────────────────────────────────────────────────────
# Main scan
# ──────────────────────────────────────────────────────────────────────────────

def V_geom(L):
    u = Fraction(2, 3)
    a = u ** L
    return float(a / (1 - a))


def scan_length(L, n_starts=12, max_cycles=80):
    """Enumerate NB cycles of length L starting from each primitive bond
    type, count cycles, tabulate same-orbit pair-distance histogram
    aggregated over all found cycles, average over cycles."""
    center = (N_SUPER // 2,) * 3
    same_dists  = Counter()
    cross_dists = Counter()
    n_cycles    = 0
    starts_used = 0
    t0 = time.time()
    for bond_idx in range(min(n_starts, 12)):
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell):
            continue
        start = (prim_bond[0], center, prim_bond[1], tgt_cell)
        starts_used += 1
        cycles = find_cycles_at_length(start, L, max_cycles)
        for cyc in cycles:
            if len(cyc) != L:
                continue
            labels = label_cycle(cyc)
            if labels is None:
                continue
            n_cycles += 1
            same_dists.update(same_orbit_pair_dists(labels, L))
            cross_dists.update(cross_orbit_pair_dists(labels, L))
        # If runtime budget exceeded, stop and report what we have
        if time.time() - t0 > 60.0:
            print(f"    (time budget exceeded at start {starts_used}/{n_starts})")
            break
    return n_cycles, same_dists, cross_dists


if __name__ == '__main__':
    print("=" * 70)
    print("Hashimoto longer-cycle inventory on H(srs)")
    print("=" * 70)
    print(f"  k* = 3,  girth g = {g},  N_SUPER = {N_SUPER}")
    print()
    print("  V_cb pin: same-orbit (b1, b2) at cycle-distance d=8 in girth-10")
    print("            cycles → V_cb = (2/3)^8/(1-(2/3)^8) = 256/6305.")
    print()
    print("  V_ub window (PDG 2024):")
    print("    excl: 3.69 ± 0.11 e-3      incl: ~4.13 ± 0.15 e-3")
    print()
    print("  Reference V_geom(L) = (2/3)^L / (1-(2/3)^L) at relevant L:")
    for L in (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18):
        print(f"    L={L:2d}: V = {V_geom(L):.6e}")
    print()

    target_lengths = [10, 12, 14, 16]
    for L in target_lengths:
        # girth-10 already enumerated by vcb_hashimoto_bfs.py — we re-run
        # for self-consistency check at L=10
        n_starts = 4 if L >= 16 else (8 if L >= 14 else 12)
        max_cycles_per_start = 30 if L >= 16 else (50 if L >= 14 else 80)
        print(f"  -- Cycle length L = {L} (n_starts={n_starts}, max_cycles_per_start={max_cycles_per_start}) --")
        n_cyc, same_d, cross_d = scan_length(
            L, n_starts=n_starts, max_cycles=max_cycles_per_start)
        print(f"    Cycles found: {n_cyc}")
        if not same_d:
            print(f"    No same-orbit pairs found.")
            continue

        print(f"    Same-orbit (b1, b2) pair-distance histogram (forward d on cycle):")
        for d in sorted(same_d):
            cnt = same_d[d]
            # Walk-rep length on this cycle: L_eff = L - d (the 'other' arc)
            L_eff_complement = L - d
            v = V_geom(L_eff_complement)
            tag = ""
            if 3.5e-3 < v < 4.3e-3:
                tag = "  ← V_ub window!"
            print(f"      d={d:2d}: count={cnt:5d}  L_eff=L-d={L_eff_complement:2d}  "
                  f"V_geom(L_eff)={v:.6e}{tag}")

        # Also print the smallest-distance (forward) — V_geom at min-d
        min_d = min(same_d)
        v_at_min_d = V_geom(min_d)
        print(f"    Min same-orbit forward d: {min_d}  V_geom(d) = {v_at_min_d:.6e}")
        print()

    print()
    print("  KEY READING")
    print("  -----------")
    print("  For V_cb the 'walk-rep' length is L_cb = g - n_fixed = 8 = g - 2.")
    print("  This is the COMPLEMENT of the same-orbit pair distance d=2 on a")
    print("  girth-10 cycle. So the relevant distance to look for above is")
    print("  L_complement = L_cycle - d = L_cb-style 'internal' walk length.")
    print()
    print("  We are looking for a longer-cycle case where L_eff = L - d gives")
    print("  V in the V_ub window (3.5e-3 to 4.3e-3). That translates to")
    print(f"  L_eff between ~13 and ~14 (V_geom(13)={V_geom(13):.4e}, "
          f"V_geom(14)={V_geom(14):.4e}).")
    print()
    print("  GATE STATUS: CAS inventory only — no physical species assignment,")
    print("  no derivation of which cycle length should host V_ub.")
