#!/usr/bin/env python3
"""
probe_C_winding_invariants_srs.py
=================================

Probe C of the four-thread investigation: classify non-backtracking closed
walks on the srs 3-torus by their winding class (n_1, n_2, n_3) ∈ Z^3 and
look for structural patterns beyond the existing length-based classification.

Motivation (per scoping doc + exploration agent finding): the framework's
existing walk classification is by LENGTH (girth multiples, multi-cycle hosts).
Walks of the same length can have different winding classes (homology classes
on the 3-torus), and different winding classes might carry distinct physical
roles — generations, color charges, or possibly statistics tags.  Listed as
genuinely-unexplored frontier in `closed_walk_length_mdl_ranking_2026-05-02.md`.

Approach:
  1. Enumerate non-backtracking closed walks on srs up to L_max (=14).
  2. For each walk, compute its winding W = sum of cell offsets traversed.
  3. Tabulate (length L, winding W) → count.
  4. Look for structural patterns:
     - Are winding classes evenly distributed or clumped?
     - Is there a Z_2 invariant (e.g., (n_1 + n_2 + n_3) mod 2 carries info)?
     - Do specific windings correlate with C_3 outer eigenvalue?

No graded content changes.  Pattern-finding probe.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import find_bonds  # noqa: E402
from proofs.foundations.theorem_B5_3_core import (  # noqa: E402
    build_directed_edges,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)


# ---------------------------------------------------------------------------
# Enumerate non-backtracking closed walks by winding class
# ---------------------------------------------------------------------------

def reverse_edge(edge):
    """Reverse of (src, tgt, cell)."""
    src, tgt, cell = edge
    return (tgt, src, tuple(-c for c in cell))


def enumerate_NB_closed_walks_by_winding(directed, L_max):
    """For each length L from 2 to L_max, enumerate non-backtracking closed
    walks starting at each directed edge.  A closed walk: return to the
    starting VERTEX (regardless of cell — winding is computed separately).

    Returns dict: (L, winding) → count
    where winding = tuple(int) sum of cell offsets along the walk.
    """
    # Build successor index: for each directed edge e, list valid NB next edges
    edge_idx = {e: i for i, e in enumerate(directed)}
    succ = [[] for _ in range(len(directed))]
    for i, e in enumerate(directed):
        e_rev = reverse_edge(e)
        for j, ep in enumerate(directed):
            if ep[0] == e[1] and ep != e_rev:
                succ[i].append(j)

    counts = defaultdict(int)
    # We use DFS up to L_max
    for start_edge_idx, start_edge in enumerate(directed):
        # Walk state: current edge index, accumulated cell, current length
        stack = [(start_edge_idx, np.array(start_edge[2], dtype=int), 1)]
        while stack:
            cur, win, L = stack.pop()
            cur_edge = directed[cur]
            # Check if this is a closed walk:
            # The walk has visited edges e_1 -> e_2 -> ... -> e_L.
            # It is closed if the target of e_L equals source of e_1
            # (which is start_edge's source = directed[start_edge_idx][0]).
            if cur_edge[1] == start_edge[0] and L >= 2:
                W = tuple(int(x) for x in win)
                counts[(L, W)] += 1
            if L < L_max:
                for nxt in succ[cur]:
                    nxt_edge = directed[nxt]
                    new_win = win + np.array(nxt_edge[2], dtype=int)
                    stack.append((nxt, new_win, L + 1))
    return counts


def report_by_length(counts):
    """Print summary by length."""
    print("=" * 100)
    print("PART A — closed-walk counts by length and winding")
    print("=" * 100)
    by_length = defaultdict(lambda: defaultdict(int))
    for (L, W), c in counts.items():
        by_length[L][W] += c
    for L in sorted(by_length):
        total = sum(by_length[L].values())
        n_windings = len(by_length[L])
        print(f"\n  L = {L:2d}: {total} closed walks total, {n_windings} distinct winding classes")
        # Show top winding classes by count
        top = sorted(by_length[L].items(), key=lambda kv: -kv[1])[:8]
        for W, c in top:
            print(f"      W = {W}:  count = {c}")


def report_parity_structure(counts):
    """Z_2 invariants: parity of sum of winding components, etc."""
    print("\n" + "=" * 100)
    print("PART B — parity / Z_2 invariants on winding")
    print("=" * 100)
    # For each length, split by (sum W) mod 2, (each W_i) mod 2, etc.
    for L in sorted(set(L for (L, _) in counts)):
        if L < 6: continue
        even_sum = sum(c for (Ll, W), c in counts.items()
                       if Ll == L and (W[0] + W[1] + W[2]) % 2 == 0)
        odd_sum = sum(c for (Ll, W), c in counts.items()
                       if Ll == L and (W[0] + W[1] + W[2]) % 2 != 0)
        # Individual parities
        even_each = sum(c for (Ll, W), c in counts.items()
                        if Ll == L and W[0] % 2 == 0 and W[1] % 2 == 0 and W[2] % 2 == 0)
        zero_winding = sum(c for (Ll, W), c in counts.items()
                            if Ll == L and W == (0, 0, 0))
        print(f"\n  L = {L:2d}:")
        print(f"    Total: {even_sum + odd_sum}")
        print(f"    (sum_W) even: {even_sum} ({100*even_sum/(even_sum+odd_sum):.1f}%)")
        print(f"    (sum_W) odd:  {odd_sum} ({100*odd_sum/(even_sum+odd_sum):.1f}%)")
        print(f"    each W_i even: {even_each}")
        print(f"    winding = (0,0,0): {zero_winding}")


def report_c3_structure(counts):
    """C_3 outer maps cell (n_1, n_2, n_3) → (n_3, n_1, n_2).
    Is the count distribution by winding invariant under C_3?"""
    print("\n" + "=" * 100)
    print("PART C — C_3 outer action on winding")
    print("=" * 100)
    # For each (L, W), check whether C_3-rotated winding has same count
    def c3(W):
        return (W[2], W[0], W[1])
    by_length = defaultdict(dict)
    for (L, W), c in counts.items():
        by_length[L][W] = c
    for L in sorted(by_length):
        if L < 4: continue
        bad = 0
        good = 0
        for W, c in by_length[L].items():
            W_c3 = c3(W)
            W_c3_2 = c3(W_c3)
            c_rot = by_length[L].get(W_c3, 0)
            c_rot_2 = by_length[L].get(W_c3_2, 0)
            # All three counts should be equal
            if c == c_rot == c_rot_2:
                good += 1
            else:
                bad += 1
        print(f"  L = {L:2d}: {good} C_3-symmetric windings, {bad} broken")


def report_girth_structure(counts):
    """Find girth-length walks (L = 10 for srs).  These should be the
    "pure girth cycles" used by the framework for Majorana phases."""
    print("\n" + "=" * 100)
    print("PART D — girth-walk-length analysis (L = 10)")
    print("=" * 100)
    girth = 10
    girth_walks = {W: c for (L, W), c in counts.items() if L == girth}
    if not girth_walks:
        print(f"\n  No walks of length {girth} enumerated (L_max < {girth})")
        return
    print(f"\n  Total girth-walks: {sum(girth_walks.values())}")
    print(f"  Distinct winding classes: {len(girth_walks)}")
    print(f"  Top winding classes:")
    for W, c in sorted(girth_walks.items(), key=lambda kv: -kv[1])[:10]:
        sumW = W[0] + W[1] + W[2]
        parity = "EVEN" if sumW % 2 == 0 else "ODD "
        print(f"    W = {W}  [Σ={sumW:+d}, {parity}]:  count = {c}")


# ---------------------------------------------------------------------------
def main():
    print(r"""
==========================================================================================
PROBE C — Winding-class invariants of non-backtracking closed walks on srs
==========================================================================================""")
    L_max = 12
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    print(f"\n  Enumerating NB closed walks up to L = {L_max} ...")
    counts = enumerate_NB_closed_walks_by_winding(directed, L_max)
    print(f"  Done. {sum(counts.values())} closed walks total across all lengths.")

    report_by_length(counts)
    report_parity_structure(counts)
    report_c3_structure(counts)
    report_girth_structure(counts)

    print("\n" + "=" * 100)
    print("Probe C sentinel: done.")
    print("=" * 100)


if __name__ == "__main__":
    main()
