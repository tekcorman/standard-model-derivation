#!/usr/bin/env python3
"""
proofs/flavor/vcb_hashimoto_bfs.py

FRAMEWORK LEVEL: Level 3 (Hashimoto graph = causal observer graph).
NOT Level 2 (srs crystal). NOT Level 1 (toggle dynamics).
See an internal note.

PURPOSE
-------
CAS verification of ADOPTED-species-generation:
  "s_b (C3=ω²) and s_c (C3=ω) are distinct causal states at
   Hashimoto distance exactly L_cb = g − 2 = 8 WITHIN a girth cycle."
  [vcb_nfixed_proof.py Step D, CAS closure route (ii)]

GATE-FIRST ANALYSIS (parameter_linter.md hard gate)
─────────────────────────────────────────────────────

Definition (Type 3 — Hashimoto 1989; Terras 2011 §2.1):
  The Hashimoto graph H(G) has nodes = directed edges of G and arcs =
  NB continuations: e1→e2 iff head(e1)=tail(e2), e2≠rev(e1).

Definition (Type 3 — Terras 2011 §1.4 "girth"):
  A girth cycle of H(G) is a closed NB walk of minimum length g
  (= girth of the underlying graph G, for k-regular G).

Step 1 [Type 4, proofs/common.py find_bonds()]:
  srs primitive cell: 4 atoms, k*=3 bonds per atom, 12 directed bonds.

Step 2 [Type 2 — construction]:
  Build supercell of N³ primitive cells (open boundary, no PBC).
  Hashimoto node = (src_atom, src_cell, tgt_atom, tgt_cell).

Step 3 [Type 2 — C3 symmetry action on directed edges]:
  C3: (x,y,z)→(z,x,y).  On BCC cell coordinates: (n1,n2,n3)→(n3,n1,n2).
  On atoms: v0→v0, v1→v3, v2→v1, v3→v2 [Type 4, C3_PERM in common.py].
  12 bond types form 4 orbits {b0, b1=C3(b0), b2=C3²(b0)} of size 3.
  b1 = "C3=ω² representative", b2 = "C3=ω representative" per the C3
  eigenvalue convention:
    C3=ω² eigenstate: b0 + ω·b1 + ω²·b2  (eigenvalue ω²)
    C3=ω  eigenstate: b0 + ω²·b1 + ω·b2  (eigenvalue ω)

Step 4 [Type 2 — girth cycle enumeration + DFS]:
  DFS from a starting directed edge to find a SIMPLE closed NB walk of
  length g=10 (a girth cycle).  Since each Hashimoto node has out-degree
  k−1=2, the DFS explores ≤ 2^10 = 1024 paths: tractable.

Step 5 [Type 2 — cycle analysis]:
  For each girth cycle: classify each edge by C3 orbit position (b0,b1,b2).
  Record cycle-distance from each b1-type edge to each b2-type edge in
  the SAME C3 orbit (i.e., the step count going forward around the cycle).
  Claim to verify: some girth cycle contains a (b1, b2) pair from the
  SAME orbit at cycle-distance exactly g−2 = 8.

WHY NOT GLOBAL MIN-DISTANCE BFS
────────────────────────────────
In the Hashimoto graph of an infinite k-regular lattice (k=3):
  N_L ~ (k−1)^L = 2^L  paths of length L from any starting edge.
  u = (k−1)/k = 2/3 > 1/(k−1) = 1/2, so Σ N_L u^L diverges.
The global minimum Hashimoto distance between C3 orbit members is 1
(adjacent bonds can be in different orbit positions) — not relevant to L_cb.

L_cb = g − n_fixed = 8 counts NB steps WITHIN the girth cycle
(the g−2 internal edges between the two pinned endpoint edges).
This is the correct combinatorial object to verify.

GATE STATUS
───────────
Steps 1–5: gate-pass (Type 4 / 2 / 2 / 2 / 2).
ADOPTED-species-generation: CLOSED if Step 5 finds a (b1,b2) pair
at cycle-distance g−n_fixed = 8.
"""

import sys
import os
from fractions import Fraction
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))
from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial

# ──────────────────────────────────────────────────────────────────────────────
# INPUTS (Type 4)
# ──────────────────────────────────────────────────────────────────────────────

d_space = predict_d_spatial()
k = predict_k_star(d_space)
g = predict_g_girth(k, d_space)

assert k == 3
assert g == 10

bonds_prim = find_bonds()
assert len(bonds_prim) == 12

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: C3 orbit structure
# ──────────────────────────────────────────────────────────────────────────────

C3_CART = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)

c3_atom = {i: int(np.argmax(C3_PERM[:, i])) for i in range(N_ATOMS)}


def bond_disp(src, tgt, cell):
    return (np.array(ATOMS[tgt])
            + cell[0]*np.array(A_PRIM[0])
            + cell[1]*np.array(A_PRIM[1])
            + cell[2]*np.array(A_PRIM[2])
            - np.array(ATOMS[src]))


prim_disps = [bond_disp(src, tgt, cell) for src, tgt, cell in bonds_prim]
prim_type_key = {(src, tgt, cell): i for i, (src, tgt, cell) in enumerate(bonds_prim)}


def c3_of_bond(i):
    src, _, _ = bonds_prim[i]
    new_src = c3_atom[src]
    rotated = C3_CART @ prim_disps[i]
    for j, (s, t, c) in enumerate(bonds_prim):
        if s == new_src and np.allclose(prim_disps[j], rotated, atol=1e-8):
            return j
    raise ValueError(f"C3 image of bond {i} not found")


c3_map = [c3_of_bond(i) for i in range(12)]

visited_orb = [False]*12
orbits = []
for start in range(12):
    if visited_orb[start]:
        continue
    b0, b1, b2 = start, c3_map[start], c3_map[c3_map[start]]
    assert c3_map[b2] == b0 and len({b0, b1, b2}) == 3
    orbits.append((b0, b1, b2))
    visited_orb[b0] = visited_orb[b1] = visited_orb[b2] = True

assert len(orbits) == 4

type_label = {}
for oi, (b0, b1, b2) in enumerate(orbits):
    type_label[b0] = (oi, 0)   # canonical
    type_label[b1] = (oi, 1)   # "C3=ω²"
    type_label[b2] = (oi, 2)   # "C3=ω"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Supercell and NB continuations (Level 3 — Hashimoto arcs)
# ──────────────────────────────────────────────────────────────────────────────

N_SUPER = 8   # 8³ = 512 primitive cells; girth cycle spans ≤ 4 cells in each dim


def in_bounds(cell):
    return all(0 <= cell[d] < N_SUPER for d in range(3))


def nb_successors(src_a, src_c, tgt_a, tgt_c):
    """Level 3: NB continuations of directed edge (src_a,src_c)→(tgt_a,tgt_c)."""
    result = []
    for (s, t, dc) in bonds_prim:
        if s != tgt_a:
            continue
        nc = tuple(tgt_c[d] + dc[d] for d in range(3))
        if not in_bounds(nc):
            continue
        if t == src_a and all(nc[d] == src_c[d] for d in range(3)):
            continue   # reverse edge — NB-invalid
        result.append((tgt_a, tgt_c, t, nc))
    return result


def edge_prim_type(src_a, src_c, tgt_a, tgt_c):
    """Primitive bond type index of a supercell directed edge."""
    dc = tuple(tgt_c[d] - src_c[d] for d in range(3))
    return prim_type_key.get((src_a, tgt_a, dc))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Find girth cycles by DFS
# ──────────────────────────────────────────────────────────────────────────────

def find_girth_cycles(start_edge, girth=10, max_cycles=20):
    """
    DFS to find simple closed NB walks of length exactly `girth` through
    `start_edge`.  Returns up to max_cycles found.

    A simple cycle visits no Hashimoto node more than once.
    Branching factor ≤ k−1 = 2, depth = girth = 10 → ≤ 2^10 = 1024 paths.
    """
    found = []
    path_set = {start_edge}

    def dfs(current, path, depth):
        if len(found) >= max_cycles:
            return
        if depth == girth:
            # Check if a continuation of `current` returns to start_edge
            for succ in nb_successors(*current):
                if succ == start_edge:
                    found.append(list(path))   # cycle: path[0] == start_edge
                    return
            return
        for succ in nb_successors(*current):
            if succ == start_edge:
                if depth == girth - 1:
                    found.append(list(path))
                continue   # don't close cycle early / don't revisit start
            if succ in path_set:
                continue   # keep cycle simple
            path_set.add(succ)
            path.append(succ)
            dfs(succ, path, depth + 1)
            path.pop()
            path_set.discard(succ)

    dfs(start_edge, [start_edge], 1)
    return found


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Classify cycle edges and measure b1↔b2 cycle-distances
# ──────────────────────────────────────────────────────────────────────────────

def analyse_cycle(cycle):
    """
    Given a girth cycle (list of g Hashimoto nodes [e0,...,e_{g-1}]),
    classify each edge by (orbit_idx, position_in_orbit).

    Returns list of (orbit_idx, pos) labels, one per cycle edge.
    None if any edge has no primitive type (shouldn't happen on valid supercell).
    """
    labels = []
    for e in cycle:
        pt = edge_prim_type(*e)
        if pt is None:
            return None
        labels.append(type_label[pt])
    return labels


def cycle_b1_b2_distances(labels, g):
    """
    For each b1 edge (orbit pos=1) in `labels`, find all b2 edges (pos=2)
    in the SAME orbit that appear later in the cycle.  Return the cycle-
    distance (forward steps from b1 to b2 position).

    Also returns the minimum such distance found.
    """
    pairs = []
    for i, (oi_i, pos_i) in enumerate(labels):
        if pos_i != 1:
            continue
        for j, (oi_j, pos_j) in enumerate(labels):
            if oi_j != oi_i or pos_j != 2:
                continue
            fwd = (j - i) % g    # forward cycle-distance from b1 to b2
            pairs.append((i, j, oi_i, fwd))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run(n_starts=40):
    """
    Try up to n_starts different starting directed edges, find girth cycles,
    analyse them for b1↔b2 pairs, report results.
    """
    center = (N_SUPER // 2,) * 3

    # Collect all girth cycles found across starting edges
    all_pair_distances = []
    cycles_found = 0
    starts_tried = 0

    for bond_idx in range(12):
        if starts_tried >= n_starts:
            break
        prim_bond = bonds_prim[bond_idx]
        # Starting directed edge: primitive bond bond_idx at center cell
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell):
            continue
        start = (prim_bond[0], center, prim_bond[1], tgt_cell)
        starts_tried += 1

        cycles = find_girth_cycles(start, girth=g, max_cycles=10)
        for cycle in cycles:
            cycles_found += 1
            if len(cycle) != g:
                continue
            labels = analyse_cycle(cycle)
            if labels is None:
                continue
            pairs = cycle_b1_b2_distances(labels, g)
            for p in pairs:
                all_pair_distances.append(p)

    return all_pair_distances, cycles_found


if __name__ == '__main__':
    print("=" * 70)
    print("V_cb — girth cycle C3 classification")
    print("=" * 70)
    print(f"\n  k* = {k},  g = {g},  N_SUPER = {N_SUPER}")
    print(f"\n  C3 orbits:")
    for oi, (b0, b1, b2) in enumerate(orbits):
        s0, t0, c0 = bonds_prim[b0]
        s1, t1, _ = bonds_prim[b1]
        s2, t2, _ = bonds_prim[b2]
        print(f"    Orbit {oi}: b0=(a{s0}→a{t0})  b1=C3(b0)=(a{s1}→a{t1})  "
              f"b2=C3²(b0)=(a{s2}→a{t2})")
    print()

    pairs, n_cycles = run()

    if n_cycles == 0:
        print("  ERROR: no girth cycles found — check supercell size.")
    else:
        print(f"  Girth cycles found: {n_cycles}")
        print(f"  (b1, b2) same-orbit pairs found: {len(pairs)}\n")

    if pairs:
        from collections import Counter
        dist_counter = Counter(fwd for (_, _, _, fwd) in pairs)
        print(f"  Cycle-distance distribution (b1 → b2, same orbit):")
        for d in sorted(dist_counter):
            mark = " ← g-2=8 ✓" if d == g - 2 else ""
            print(f"    d={d:3d}: count={dist_counter[d]:4d}{mark}")

        L_cb_expected = g - 2   # = 8
        if L_cb_expected in dist_counter:
            print(f"\n  GATE CHECK — ADOPTED-species-generation:")
            print(f"    Cycle-distance g−2={L_cb_expected} found in {dist_counter[L_cb_expected]} "
                  f"(b1,b2) pairs  ✓")
            print(f"    ADOPTED-species-generation: CLOSED (CAS verified)")
        else:
            print(f"\n  GATE CHECK — ADOPTED-species-generation:")
            print(f"    Cycle-distance g−2={L_cb_expected} NOT found in any (b1,b2) pair  ✗")
            print(f"    Distances found: {sorted(dist_counter.keys())}")
            print(f"    ADOPTED-species-generation: OPEN")
    else:
        print("  No (b1, b2) same-orbit pairs found in any girth cycle.")
        print("  All orbit positions in found cycles:")
        center = (N_SUPER // 2,) * 3
        for bond_idx in range(12):
            prim_bond = bonds_prim[bond_idx]
            dc = prim_bond[2]
            tgt_cell = tuple(center[d] + dc[d] for d in range(3))
            if not in_bounds(tgt_cell):
                continue
            start = (prim_bond[0], center, prim_bond[1], tgt_cell)
            cycles = find_girth_cycles(start, girth=g, max_cycles=3)
            for cycle in cycles[:1]:
                labels = analyse_cycle(cycle)
                if labels:
                    print(f"    Start bond {bond_idx}: "
                          f"orbit/pos = {labels}")
                    break

    print()
    print("  THREE-LEVEL CHECK:")
    print("    Level 1 (toggles): branch measure μ  — NOT computed here")
    print("    Level 2 (srs crystal): bond geometry from find_bonds() — input")
    print("    Level 3 (Hashimoto graph): girth cycles on directed edges — THIS FILE")
