#!/usr/bin/env python3
"""
proofs/flavor/vub_bridge_d14_pinning_probe.py

PURPOSE
-------
EXPLORATORY probe for `docs/theorems/theorem_C3obs_substrate_bridge_attempt.md` Step 2.

V_cb's structural argument is anchored by CAS-verified same-orbit (b1, b2) pinning at
cycle-distance d = g − n_fixed = 8 within girth cycles of H(srs) (20 pairs in 8³ supercell,
per `proofs/flavor/vcb_hashimoto_bfs.py`).

This probe tests the analogous claim for m=2 multi-cycle hosts (L_cycle=16):
  same-orbit pinning at cycle-distance d = L_eff(2) = L_cycle − n_fixed = 14.

If positive: structural extension of V_cb's argument is CAS-supported, contributing
evidence for the bridge composition theorem (Step BR4 of the scoping doc).

If negative or zero: the bridge composition argument's k=2 step needs reconsideration.

Run with:
    PYTHONPATH=. python3 proofs/flavor/vub_bridge_d14_pinning_probe.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vcb_hashimoto_bfs as vcb
import hashimoto_16cycle_decomposition as h16

bonds_prim     = vcb.bonds_prim
prim_disps     = vcb.prim_disps
prim_type_key  = vcb.prim_type_key
type_label     = vcb.type_label
g              = vcb.g
nb_successors  = vcb.nb_successors
edge_prim_type = vcb.edge_prim_type
in_bounds      = vcb.in_bounds


import numpy as np


def _bond_eq(b1, b2):
    """Compare two primitive bonds robustly (handles numpy arrays)."""
    s1, t1, d1 = b1
    s2, t2, d2 = b2
    if int(s1) != int(s2) or int(t1) != int(t2):
        return False
    return tuple(int(x) for x in d1) == tuple(int(x) for x in d2)


# C_3 vertex permutation as a dict atom_idx -> atom_idx
C3_VERTEX_PERM = {}
for i in range(4):
    for j in range(4):
        if abs(vcb.C3_PERM[i, j] - 1.0) < 1e-12:
            C3_VERTEX_PERM[j] = i


def cycle_orbit_signature(cycle):
    """For each Hashimoto edge in the cycle, return the C_3 orbit class label
    (an integer 0..3 corresponding to which size-3 C_3 orbit the primitive
    bond type belongs to)."""
    n_prim = len(bonds_prim)
    orbits = []
    visited = [False] * n_prim
    for i in range(n_prim):
        if visited[i]:
            continue
        orbit = []
        cur = i
        for _ in range(3):
            orbit.append(cur)
            visited[cur] = True
            (s, t, d) = bonds_prim[cur]
            sn = C3_VERTEX_PERM[int(s)]
            tn = C3_VERTEX_PERM[int(t)]
            d_int = tuple(int(x) for x in d)
            dn = (d_int[2], d_int[0], d_int[1])
            cur_new = None
            for j in range(n_prim):
                if _bond_eq(bonds_prim[j], (sn, tn, dn)):
                    cur_new = j
                    break
            if cur_new is None:
                raise RuntimeError(f"C_3 image of bond {cur} not found")
            cur = cur_new
        orbits.append(orbit)

    orbit_id_of = {}
    for orb_id, orb in enumerate(orbits):
        for pt in orb:
            orbit_id_of[pt] = orb_id

    sig = []
    for (s_a, s_c, t_a, t_c) in cycle:
        d = tuple(int(t_c[i]) - int(s_c[i]) for i in range(3))
        pt_idx = None
        for j, (s, t, dd) in enumerate(bonds_prim):
            if int(s) == int(s_a) and int(t) == int(t_a) and tuple(int(x) for x in dd) == d:
                pt_idx = j
                break
        if pt_idx is None:
            raise RuntimeError(f"Cycle edge ({s_a, s_c, t_a, t_c}) not in bonds_prim")
        sig.append((pt_idx, orbit_id_of[pt_idx]))
    return sig


def count_same_orbit_pinning(cycle, target_distance):
    """Count pairs of edges (i, j) in the cycle with cycle-distance |i-j| (mod L)
    equal to target_distance such that both edges are in the SAME C_3 orbit.

    Returns (count, examples).
    """
    sig = cycle_orbit_signature(cycle)
    L = len(cycle)
    count = 0
    examples = []
    for i in range(L):
        j = (i + target_distance) % L
        if i >= j:
            continue   # avoid double-counting
        pt_i, orb_i = sig[i]
        pt_j, orb_j = sig[j]
        if orb_i == orb_j and pt_i != pt_j:
            count += 1
            if len(examples) < 5:
                examples.append((i, j, pt_i, pt_j, orb_i))
    return count, examples


def main():
    print("=" * 76)
    print("V_ub bridge probe — same-orbit pinning at cycle-distance d=14 on m=2 hosts")
    print("=" * 76)
    print()
    print("Reference: docs/theorems/theorem_C3obs_substrate_bridge_attempt.md Step 2 (induction step)")
    print("Mode: EXPLORATORY. Reports CAS findings for the V_cb-analog at L=16 (m=2).")
    print()

    # First, verify the V_cb baseline at L=10, d=8
    print("--- BASELINE: same-orbit pinning at d=8 on L=10 girth cycles ---")
    print("(This is V_cb's structural anchor; 20 pairs in 8³ supercell per existing CAS.)")
    print()

    # Find a few girth cycles — iterate over primitive bonds at center of supercell
    N_SUPER = vcb.N_SUPER
    center = (N_SUPER // 2,) * 3
    print(f"Generating girth-10 cycles from center starts (N_SUPER={N_SUPER}, center={center})...")

    girth_cycles = []
    for bond_idx in range(12):
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell):
            continue
        start_edge = (prim_bond[0], center, prim_bond[1], tgt_cell)
        cycles = h16.find_cycles_at_length(start_edge, 10, max_cycles=10)
        girth_cycles.extend(cycles[:5])   # cap per starting edge
        if len(girth_cycles) >= 30:
            break
    print(f"  Found {len(girth_cycles)} girth-10 cycles from this start.")

    if girth_cycles:
        sample = girth_cycles[0]
        sig = cycle_orbit_signature(sample)
        orbits_visited = sorted(set(orb for (_, orb) in sig))
        print(f"  Sample cycle 0: orbit signature = {[orb for (_, orb) in sig]}")
        print(f"  Orbits visited: {orbits_visited}")
        d8_count, d8_examples = count_same_orbit_pinning(sample, 8)
        print(f"  Same-orbit (b1, b2) pairs at d=8 in this cycle: {d8_count}")
        if d8_examples:
            for (i, j, pt_i, pt_j, orb_id) in d8_examples[:3]:
                print(f"    pos {i} (prim_type={pt_i}, orbit {orb_id}) ↔ pos {j} (prim_type={pt_j}, orbit {orb_id})")

    # Now the m=2 host: L=16
    print()
    print("--- m=2 HOST: same-orbit pinning at d=14 on L=16 multi-cycle hosts ---")
    print()
    print("Generating L=16 multi-cycle hosts from center starts...")

    cycles_16 = []
    for bond_idx in range(12):
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell):
            continue
        start_edge = (prim_bond[0], center, prim_bond[1], tgt_cell)
        cs = h16.find_cycles_at_length(start_edge, 16, max_cycles=20)
        cycles_16.extend(cs[:10])
        if len(cycles_16) >= 50:
            break
    print(f"  Found {len(cycles_16)} L=16 cycles from this start.")

    if cycles_16:
        sample = cycles_16[0]
        sig = cycle_orbit_signature(sample)
        orbits_visited = sorted(set(orb for (_, orb) in sig))
        print(f"  Sample L=16 cycle 0: orbit signature = {[orb for (_, orb) in sig]}")
        print(f"  Orbits visited: {orbits_visited}")
        d14_count, d14_examples = count_same_orbit_pinning(sample, 14)
        print(f"  Same-orbit (b1, b2) pairs at d=14 in this cycle: {d14_count}")
        if d14_examples:
            for (i, j, pt_i, pt_j, orb_id) in d14_examples[:3]:
                print(f"    pos {i} (prim_type={pt_i}, orbit {orb_id}) ↔ pos {j} (prim_type={pt_j}, orbit {orb_id})")

        # Also report d=8 (within-girth-cycle analog) on the L=16 host
        d8_count_16, _ = count_same_orbit_pinning(sample, 8)
        print(f"  Comparison: same-orbit (b1,b2) pairs at d=8 on this L=16 cycle: {d8_count_16}")

    # Aggregate over multiple L=16 cycles
    if len(cycles_16) > 1:
        d14_total = 0
        d8_total = 0
        for c in cycles_16:
            cnt14, _ = count_same_orbit_pinning(c, 14)
            cnt8, _ = count_same_orbit_pinning(c, 8)
            d14_total += cnt14
            d8_total += cnt8
        print()
        print(f"  AGGREGATE over {len(cycles_16)} L=16 cycles from this start:")
        print(f"    Same-orbit pairs at d=14: {d14_total}")
        print(f"    Same-orbit pairs at d=8:  {d8_total}")
        print(f"    (V_cb baseline: 20 pairs at d=8 over the full 8³ supercell.)")

    print()
    print("=" * 76)
    print("DIAGNOSIS")
    print("=" * 76)
    if cycles_16 and d14_count > 0:
        print(f"  ✓ Same-orbit pinning at d=14 EXISTS on m=2 multi-cycle hosts.")
        print(f"  This supports the bridge composition argument (Step 2 of the bridge attempt):")
        print(f"  the m=2 host has the structural analog of V_cb's d=8 within-girth pinning,")
        print(f"  scaled to cycle-distance 14 = L_eff(2) = 6m + 2 with m=2.")
        print(f"  Bridge derivation V_ub = Σ_{{m≥2}} α_m/(1−α_m) ≈ 3.767e-3 receives CAS support.")
    elif cycles_16 and d14_count == 0:
        print(f"  ✗ NO same-orbit pinning at d=14 on the sampled m=2 host.")
        print(f"  This casts doubt on the direct V_cb-analog reading of the bridge.")
        print(f"  The bridge composition argument may need a different CAS anchor for k=2.")
    else:
        print(f"  ! No L=16 cycles found from this start in the small supercell.")
        print(f"  Run with larger supercell (vcb_hashimoto_bfs N_SUPER) to extend coverage.")


if __name__ == "__main__":
    main()
