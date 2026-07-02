#!/usr/bin/env python3
"""
y_τ sub-leading via super-girth cycle contributions — Phase 1B probe.

CONTEXT
-------
Direction 2 from an internal working note:
"sub-leading cycle amplitudes (length g+2, g+4) — already in framework's NB
walk enumeration but haven't been included in y_τ. Would they give a
structural correction, and is its magnitude ~0.13%?"

After Direction 1's saddle-vs-BZ probe (Phase 1A) returned a clean negative
result on tan²(arg h(k)) (the function doesn't BZ-integrate smoothly),
this probe tests Direction 2's simplest form.

HYPOTHESIS
----------
Tree-level y_τ = α_1_full / k*² where α_1_full = (n_g_edge/k*) × ((k*-1)/k*)^(g-2)
captures girth-only (L=g=10) NB-walk closed cycles.

For each super-girth length L = g+2, g+4, g+6, ..., there exist additional
closed simple NB walks. By Feshbach exponent principle (`predictions/feshbach_exponent_principle.py`),
each contributes a survival factor ((k*-1)/k*)^(L-2). With combinatorial
weight n_L_edge/k* (cycles of length L per ordered edge pair, divided by
coordination), super-girth amplitude is

    α_L = (n_L_edge / k*) × ((k*-1)/k*)^(L-2)

If y_τ_full = (1/k*²) × Σ_L α_L (uniform cycle-length weighting), then

    y_τ_correction = Σ_{L>g} α_L / α_g

QUESTIONS THIS PROBE ANSWERS
----------------------------
1. What are n_L_edge for L = 10, 12, 14, 16 on srs?
2. What is Σ_{L>g} α_L / α_g numerically?
3. Sign: does super-girth INCREASE or DECREASE y_τ vs tree-level?
4. Magnitude: is it in the 0.13% ballpark, or way off?

DOES NOT DO
-----------
- DOES NOT compare to PDG y_τ in the form "this is the closure" (per
  an internal note).
- DOES NOT modify `predictions/y_tau.py`.
- DOES NOT claim the simple uniform-Σ_L weighting is the framework's
  correct mechanism. That's a structural question post-probe.

METHOD
------
1. Build srs primitive cell (4 atoms, 12 directed bonds).
2. Construct supercell of N³ primitive cells, N large enough to contain
   cycles up to L_max without wrapping.
3. From a fixed starting directed edge in the central primitive cell,
   DFS to enumerate ALL simple closed NB walks of length L, for each
   L in {10, 12, 14, 16}.
4. Count cycles per starting edge → n_L_edge.
5. Compute α_L = (n_L / k*) × (2/3)^(L-2) for each L.
6. Compute correction ratio Σ_{L>g} α_L / α_g.
7. Report finding.

REFERENCES
----------
- `predictions/alpha_1_full.py` (n_g_edge = 5 for srs)
- `predictions/feshbach_exponent_principle.py`
- `proofs/flavor/vcb_hashimoto_bfs.py` (DFS pattern on supercell)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from proofs.common import find_bonds, N_ATOMS

K_STAR = 3
G_GIRTH = 10
N_G_EDGE_EXPECTED = 5  # per `predictions/alpha_1_full.py`


def build_supercell_nb(N_super=8):
    """Return supercell NB-successor function and bond list.

    Each directed edge in the supercell is identified by (src_atom, src_cell, tgt_atom, tgt_cell).
    NB successor: (src_a, src_c, tgt_a, tgt_c) → list of (tgt_a, tgt_c, next_t, next_c)
    where next_c respects bounds and (next_t, next_c) ≠ (src_a, src_c) (no backtrack).
    """
    bonds_prim = find_bonds()
    assert len(bonds_prim) == 12

    def in_bounds(cell):
        return all(0 <= cell[d] < N_super for d in range(3))

    def nb_successors(src_a, src_c, tgt_a, tgt_c):
        result = []
        for (s, t, dc) in bonds_prim:
            if s != tgt_a:
                continue
            nc = tuple(tgt_c[d] + dc[d] for d in range(3))
            if not in_bounds(nc):
                continue
            if t == src_a and all(nc[d] == src_c[d] for d in range(3)):
                continue
            result.append((tgt_a, tgt_c, t, nc))
        return result

    return nb_successors, bonds_prim


def enumerate_simple_closed_nb_walks(start_edge, target_length, nb_successors):
    """DFS enumerate simple closed NB walks of EXACT length L starting from
    start_edge.

    A "simple closed NB walk" of length L starting at directed edge e_0:
      - sequence of L directed edges e_0, e_1, ..., e_{L-1}
      - each e_{i+1} is an NB-successor of e_i
      - the walk closes: e_L = e_0 (i.e., NB-successors of e_{L-1} include e_0)
      - no repeated directed edges in the walk (simple)

    Returns: count of such walks.
    """
    src_a, src_c, tgt_a, tgt_c = start_edge
    count = [0]

    def dfs(current_edge, depth, visited):
        if depth == target_length:
            # Check closure: is the start_edge a valid NB successor of current_edge?
            successors = nb_successors(*current_edge)
            for succ in successors:
                if succ == start_edge:
                    count[0] += 1
                    break
            return
        successors = nb_successors(*current_edge)
        for succ in successors:
            if succ in visited:
                continue
            visited.add(succ)
            dfs(succ, depth + 1, visited)
            visited.remove(succ)

    dfs(start_edge, 1, {start_edge})
    return count[0]


def main():
    print("=" * 78)
    print("y_τ super-girth probe — Phase 1B (Direction 2 simplest form)")
    print("Question: do super-girth NB cycles contribute to y_τ?")
    print("=" * 78)
    print()

    nb_successors, bonds_prim = build_supercell_nb(N_super=8)

    # Pick a starting directed edge in the central cell
    # bond (0, 1, (1,1,1)) places us at center, easy to work with.
    central_cell = (4, 4, 4)
    src_a, _, dc = bonds_prim[0]
    tgt_a = bonds_prim[0][1]
    start_cell_src = central_cell
    start_cell_tgt = tuple(central_cell[i] + dc[i] for i in range(3))
    start_edge = (src_a, start_cell_src, tgt_a, start_cell_tgt)
    print(f"Starting directed edge: atom {src_a} cell {start_cell_src} → "
          f"atom {tgt_a} cell {start_cell_tgt}")
    print()

    target_lengths = [10, 12, 14]
    print(f"{'L':>4s}  {'n_L_edge':>10s}  {'(2/3)^(L-2)':>14s}  "
          f"{'α_L':>14s}  {'α_L/α_10':>12s}  {'wall_time':>10s}")
    print("-" * 78)

    import time
    alpha = {}
    for L in target_lengths:
        t0 = time.time()
        n_L = enumerate_simple_closed_nb_walks(start_edge, L, nb_successors)
        elapsed = time.time() - t0
        feshbach = (2 / 3) ** (L - 2)
        alpha_L = (n_L / K_STAR) * feshbach
        alpha[L] = (n_L, alpha_L)
        ratio = alpha_L / alpha[10][1] if 10 in alpha and alpha[10][1] > 0 else float('nan')
        print(f"{L:>4d}  {n_L:>10d}  {feshbach:>14.6e}  {alpha_L:>14.6e}  "
              f"{ratio:>+12.6e}  {elapsed:>9.2f}s")

    print()
    print("=" * 78)
    print("CONSISTENCY CHECK with framework's α_1_full = (5/3)(2/3)^8")
    print("=" * 78)
    n_10, alpha_10 = alpha[10]
    expected_n10 = N_G_EDGE_EXPECTED  # n_g_edge = 5
    expected_alpha = (5 / 3) * (2 / 3) ** 8
    print(f"  Probe n_10_edge = {n_10}; framework expected = {expected_n10}")
    print(f"  Probe α_10 = {alpha_10:.6e}; framework α_1_full = {expected_alpha:.6e}")
    if n_10 == expected_n10:
        print(f"  ✓ n_10_edge match — probe consistent with framework tree-level")
    else:
        print(f"  ⚠ n_10_edge mismatch: probe found {n_10} closed simple NB walks "
              f"of length 10 starting from one directed edge; framework value is {expected_n10}.")
        print(f"    This may reflect: (a) different enumeration convention (e.g. divided "
              f"by 2 for orientation, or per ordered-pair vs per starting-edge), or "
              f"(b) supercell finite-size artifact. Need to reconcile before trusting "
              f"super-girth counts.")

    if 12 in alpha:
        n_12, alpha_12 = alpha[12]
        ratio_12 = alpha_12 / alpha_10 if alpha_10 > 0 else float('nan')
        print()
        print(f"  Super-girth at L=12: n_12_edge = {n_12}, α_12 = {alpha_12:.6e}")
        print(f"  α_12 / α_10 = {ratio_12:+.4%}")
        if abs(ratio_12) > 1.0:
            print(f"  ⚠ ratio is order-unity or larger; super-girth would dominate, "
                  f"not be sub-leading. Mechanism is wrong, OR the simple uniform-Σ_L "
                  f"weighting is NOT the framework's structural rule.")

    if 14 in alpha:
        n_14, alpha_14 = alpha[14]
        ratio_14 = alpha_14 / alpha_10 if alpha_10 > 0 else float('nan')
        print(f"  Super-girth at L=14: n_14_edge = {n_14}, α_14 = {alpha_14:.6e}")
        print(f"  α_14 / α_10 = {ratio_14:+.4%}")

    print()
    print("=" * 78)
    print("INTERPRETATION (DO NOT compare to PDG y_τ here)")
    print("=" * 78)
    print()
    print("  This probe tests the SIMPLEST form of Direction 2: super-girth")
    print("  cycles contribute to y_τ uniformly via Σ_L α_L = Σ_L (n_L/k*)(2/3)^(L-2).")
    print()
    print("  Possible outcomes:")
    print("  (a) n_10_edge matches framework value 5 AND super-girth ratios are O(0.001):")
    print("      → magnitude consistent with 0.13%; mechanism candidate; needs structural")
    print("        identification of WHY uniform Σ_L is correct (NOT post-hoc).")
    print("  (b) n_10_edge matches AND super-girth ratios are O(1) or larger:")
    print("      → uniform Σ_L overshoots; wrong mechanism. The framework has SOME")
    print("        selection rule for which lengths enter (only girth, not super-girth).")
    print("        Pivot to Direction 4 (Feshbach analog of 5/12) or Direction 3.")
    print("  (c) n_10_edge mismatch:")
    print("      → enumeration convention or supercell issue; reconcile before any")
    print("        super-girth claim.")
    print()
    print("  Phase 1B is empirical fact-finding. No predictions changes.")


if __name__ == '__main__':
    main()
