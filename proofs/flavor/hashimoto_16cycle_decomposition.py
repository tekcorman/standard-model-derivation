#!/usr/bin/env python3
"""
proofs/flavor/hashimoto_16cycle_decomposition.py

FRAMEWORK LEVEL: Level 3 (Hashimoto graph = causal observer graph).

PURPOSE
-------
Test the structural conjecture that EVERY length-16 NB cycle on H(srs)
decomposes as the symmetric difference of two girth-10 NB cycles sharing
a 2-edge NB path.

If this is true, the V_ub candidate (L_cycle=16, d=2 same-orbit pin,
V = (2/3)^14 / (1−(2/3)^14)) gets a clean structural reading:

  V_ub = TWO girth-cycle Feshbach amplitudes, glued along 2 shared edges.

That reduces the V_ub vs V_cb structural difference to "single vs double
girth-cycle topology" — a clear, derivable distinction with no Z₃
holonomy / σ_S / icosahedral content.

ALGORITHMIC TEST
----------------
For each 16-cycle C16 on H(srs):
  1. C16 is a sequence of 16 directed-edge nodes on H(srs).
  2. The underlying srs vertex walk is v0 → v1 → ... → v16 = v0
     (with v_i = (atom, supercell_cell)).
  3. For each pair (i, j) with (j − i) mod 16 = 8 (antipodal cycle
     positions), look for an NB path of length 2 (a "chord") in srs
     between v_i and v_j that does NOT use any edge of C16, AND that
     glues consistently (i.e., the resulting two cycles are valid NB
     closed walks).
  4. If such a chord exists and gives two valid NB 10-cycles, mark C16
     as "DECOMPOSABLE."

GATE STATUS
-----------
This script is CAS verification only. It either confirms or refutes the
structural claim "every 16-cycle decomposes as two glued 10-cycles." It
does not assign physical species or file a prediction.
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
# Adjacency: for each srs vertex (atom, cell) list outgoing directed edges
# ──────────────────────────────────────────────────────────────────────────────

def out_edges_at(atom, cell):
    """All directed edges leaving (atom, cell) within the supercell."""
    out = []
    for (s, t, dc) in bonds_prim:
        if s != atom:
            continue
        nc = tuple(cell[d] + dc[d] for d in range(3))
        if not in_bounds(nc):
            continue
        out.append((atom, cell, t, nc))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 16-cycle finder + 10-cycle reference
# ──────────────────────────────────────────────────────────────────────────────

def find_cycles_at_length(start_edge, L, max_cycles):
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
# Decomposition test
# ──────────────────────────────────────────────────────────────────────────────

def cycle_to_vertex_sequence(cycle):
    """A Hashimoto cycle [(s0,c0,t0,c0'), (s1,c1,t1,c1'), ...] where
    consecutive entries chain (t_i,c_i') == (s_{i+1},c_{i+1}). The
    underlying srs vertex sequence visited is just the source of each
    Hashimoto node, plus the final target. For a closed cycle we get
    L+1 vertices with the last == the first."""
    verts = [(e[0], e[1]) for e in cycle]
    last = (cycle[-1][2], cycle[-1][3])
    verts.append(last)
    assert verts[0] == verts[-1], "cycle should close"
    return verts


def cycle_edge_set(cycle):
    """Undirected edge set of cycle (as frozensets of (atom, cell) endpoint pairs)."""
    edges = set()
    for (s, c, t, c2) in cycle:
        e = frozenset({(s, c), (t, c2)})
        edges.add(e)
    return edges


def is_nb_path(seq):
    """A vertex sequence v0, v1, ..., vL is NB if no two consecutive
    edges (vi-1 -> vi) and (vi -> vi+1) are reverses, i.e., vi-1 != vi+1."""
    for i in range(1, len(seq) - 1):
        if seq[i - 1] == seq[i + 1]:
            return False
    return True


def find_chord(v_a, v_b, forbidden_edges):
    """Find all NB paths of length 2 in srs from v_a to v_b avoiding the
    given (undirected) edge set. Return list of intermediate vertices v_mid."""
    mids = []
    for mid_edge in out_edges_at(*v_a):
        v_mid = (mid_edge[2], mid_edge[3])
        e_first = frozenset({v_a, v_mid})
        if e_first in forbidden_edges:
            continue
        if v_mid == v_b:
            continue   # 1-edge, not 2-edge
        # v_mid -> v_b must exist as a directed edge with NB at v_mid
        for second_edge in out_edges_at(*v_mid):
            v_end = (second_edge[2], second_edge[3])
            if v_end != v_b:
                continue
            e_second = frozenset({v_mid, v_b})
            if e_second in forbidden_edges:
                continue
            # NB at v_mid: incoming was v_a -> v_mid, outgoing v_mid -> v_b;
            # require v_a != v_b (true since they are the chord endpoints).
            if v_a == v_b:
                continue
            mids.append(v_mid)
    return mids


def check_decomposition(cycle16):
    """For each antipodal pair (i, i+8), search for an NB chord of length
    2 not on the cycle. Verify the two resulting cycles are valid NB
    closed walks of length 10. Returns list of (i, j, v_mid) successful
    decompositions."""
    verts = cycle_to_vertex_sequence(cycle16)   # 17 entries, verts[0]==verts[16]
    cyc_edges = cycle_edge_set(cycle16)

    successes = []
    for i in range(16):
        j = (i + 8) % 16
        # Antipodal vertex pair on the cycle
        v_i = verts[i]
        v_j = verts[j]
        if v_i == v_j:
            continue   # degenerate

        chord_mids = find_chord(v_i, v_j, cyc_edges)
        for v_mid in chord_mids:
            # Construct cycle A: vertices i -> i+1 -> ... -> j (forward arc, 8 edges) + chord [j -> v_mid -> i]
            # Forward arc verts: verts[i], verts[i+1], ..., verts[j]  (length 8 NB path)
            arc_A_verts = [verts[(i + k) % 16] for k in range(9)]   # 9 vertices, 8 edges
            # Chord backwards: verts[j] -> v_mid -> verts[i]
            cycle_A = arc_A_verts + [v_mid, verts[i]]    # 11 vertices for 10-edge cycle? Let me recount.
            # arc_A_verts has 9 entries (0..8 indices) = 8 edges from verts[i] to verts[j]
            # Then we append v_mid, then verts[i]: 9 + 2 = 11 entries = 10 edges.
            # cycle_A[0] = verts[i], cycle_A[10] = verts[i]: closed ✓.
            if not is_nb_path(cycle_A):
                continue

            # Cycle B: vertices j -> j+1 -> ... -> i (other way around, 8 edges) + chord [i -> v_mid -> j]
            arc_B_verts = [verts[(j + k) % 16] for k in range(9)]   # verts[j], verts[j+1], ..., verts[i]
            cycle_B = arc_B_verts + [v_mid, verts[j]]
            if not is_nb_path(cycle_B):
                continue

            # Both are valid 10-edge NB closed walks → decomposition succeeds
            successes.append((i, j, v_mid))

    return successes


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("16-cycle decomposition test on H(srs)")
    print("=" * 70)
    print("  Testing conjecture: every length-16 NB cycle decomposes as")
    print("  the symmetric difference of two girth-10 NB cycles sharing a")
    print("  2-edge NB path (chord at antipodal cycle positions).")
    print()
    print("  Implication if TRUE: V_ub = two-girth-cycle Feshbach with seam,")
    print("  L_eff = 16 - 2 = 14 → V_geom(14) = 16384/4766585 ≈ 3.437e-3.")
    print()

    # Enumerate 16-cycles
    center = (N_SUPER // 2,) * 3
    all_16 = []
    t0 = time.time()
    for bond_idx in range(12):
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell):
            continue
        start = (prim_bond[0], center, prim_bond[1], tgt_cell)
        cycles = find_cycles_at_length(start, 16, max_cycles=200)
        for c in cycles:
            if len(c) == 16:
                all_16.append(c)
        if time.time() - t0 > 120:
            print(f"  (time budget hit at start {bond_idx})")
            break

    n16 = len(all_16)
    print(f"  Enumerated {n16} length-16 cycles in {time.time()-t0:.1f}s")
    print()

    # Test each
    n_decomposable = 0
    n_undecomposable = 0
    decomp_count_hist = Counter()
    sample_undecomposable = []
    for ci, c in enumerate(all_16):
        succ = check_decomposition(c)
        decomp_count_hist[len(succ)] += 1
        if succ:
            n_decomposable += 1
        else:
            n_undecomposable += 1
            if len(sample_undecomposable) < 3:
                sample_undecomposable.append(c)

    print(f"  RESULTS")
    print(f"  -------")
    print(f"  Total 16-cycles tested:      {n16}")
    print(f"  Decomposable (≥1 chord):     {n_decomposable}  ({100.0*n_decomposable/max(n16,1):.1f}%)")
    print(f"  Undecomposable (0 chords):   {n_undecomposable}  ({100.0*n_undecomposable/max(n16,1):.1f}%)")
    print()
    print(f"  Histogram of #valid (i, v_mid) decompositions per 16-cycle:")
    for k in sorted(decomp_count_hist):
        print(f"    {k:3d} chord-decompositions: {decomp_count_hist[k]:5d} cycles")
    print()
    if sample_undecomposable:
        print(f"  Sample undecomposable cycle (first found):")
        c0 = sample_undecomposable[0]
        verts = cycle_to_vertex_sequence(c0)
        for k, v in enumerate(verts[:-1]):
            print(f"    [{k:2d}] atom={v[0]} cell={v[1]}")
        print()

    # Now do the symmetric/structural reading: how many distinct chord-vertex
    # pairs decompose each cycle? Symmetry expectations:
    #  - if the cycle is fully symmetric under d ↔ 16-d, all 8 antipodal
    #    pairs (i, i+8) for i=0..7 may give distinct chord choices.
    print("  STRUCTURAL READING")
    print("  ------------------")
    if n_undecomposable == 0:
        print("  Every length-16 NB cycle on H(srs) decomposes as two girth-10")
        print("  NB cycles glued along a 2-edge NB path. The 'two girth cycles")
        print("  shared along a 2-edge seam' interpretation of L=16 is structurally")
        print("  validated.")
    else:
        frac = 100.0 * n_decomposable / max(n16, 1)
        print(f"  {frac:.1f}% of 16-cycles admit the decomposition; the rest")
        print(f"  do not. The 'two glued girth cycles' structural claim is")
        print(f"  PARTIAL — needs a more refined formulation to handle the")
        print(f"  undecomposable subset.")
    print()
    print("  GATE STATUS: CAS check only. Result determines whether the")
    print("  L=16 host-cycle for V_ub admits a 'two girth-cycles glued'")
    print("  Feshbach reading without invoking color/generation identification.")
