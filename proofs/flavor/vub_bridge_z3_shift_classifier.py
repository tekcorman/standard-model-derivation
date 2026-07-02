#!/usr/bin/env python3
"""
proofs/flavor/vub_bridge_z3_shift_classifier.py

PURPOSE
-------
Step 2 of Path T in an internal working note.

Refines `vub_bridge_higher_m_pinning_probe.py` by classifying each same-
C₃-orbit pinned pair (b₁, b₂) at cycle-distance d on m-cycle hosts as
EITHER:

  ΔGen=1 case: b₂ = C₃ b₁ (single Z₃ cyclic shift within orbit)
  ΔGen=2 case: b₂ = C₃² b₁ (two Z₃ cyclic shifts within orbit)

The hypothesis under test is the "combinatorial Z₃ symmetry (not holonomy)"
reading of the bridge:

  - V_cb (ΔGen=1) sums Z₃-shift pinnings over all hosts
  - V_ub (ΔGen=2) sums Z₃²-shift pinnings over all hosts
  - "All-m sum" emerges naturally because every host class admits same-orbit
    pinnings, but only the Z₃ vs Z₃² subset contributes to each V_ij

This is independent of the flat-Z₃-connection theorem (which refutes the
holonomy version of the lemma's argument): the symmetry exists as a
combinatorial automorphism of srs even when its connection has trivial
holonomy. The C₃ orbit structure of bonds (4 orbits of size 3) IS the
mechanism.

WHAT THE DATA SHOULD SHOW (if the hypothesis holds)
---------------------------------------------------
For each m ∈ {1, 2, 3, 4} and each d ∈ {8, 14, 20, 26}, the same-orbit
pair count splits into Z₃-shift (ΔGen=1) and Z₃²-shift (ΔGen=2)
sub-counts. Both should exist (typically in roughly equal proportion since
both Z₃-shifts are equally likely a priori on a random pair from a size-3
orbit), but the WEIGHTED sums:

  V_cb_predicted = Σ_m (Z₃-shift count at d=L_eff(m) on m-host) · α_m / (1−α_m) / N_pairs_m
  V_ub_predicted = Σ_m (Z₃²-shift count at d=L_eff(m) on m-host) · α_m / (1−α_m) / N_pairs_m

…should reproduce the framework's working values:

  V_cb ≈ 0.0406 (PDG 0.0405 ± 0.0015)
  V_ub ≈ 3.767e-3 OR 3.439e-3 (PDG combined 3.82 ± 0.20 × 10⁻³)

depending on whether the diagonal (d=L_eff(m)) is the only contribution or
whether off-diagonal d-pinnings (which the previous probe found to be
plentiful) ALSO contribute.

GATE STATUS
-----------
EXPLORATORY CAS. Reports Z₃-shift vs Z₃²-shift breakdown of same-orbit
pinned pair counts. The structural inference is in the companion docs.

Run with:
    PYTHONPATH=. python3 proofs/flavor/vub_bridge_z3_shift_classifier.py
"""

import sys
import os
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vcb_hashimoto_bfs as vcb
import hashimoto_16cycle_decomposition as h16
import vub_bridge_d14_pinning_probe as d14probe

bonds_prim     = vcb.bonds_prim
nb_successors  = vcb.nb_successors
in_bounds      = vcb.in_bounds
N_SUPER        = vcb.N_SUPER
type_label     = vcb.type_label    # prim_bond_idx -> (orbit_id, position_within_orbit ∈ {0,1,2})
c3_map         = vcb.c3_map        # prim_bond_idx -> C₃-shifted prim_bond_idx

# Sanity: c3_map[i] takes pos within orbit (i_pos) -> (i_pos + 1) mod 3
# so position difference (j_pos - i_pos) mod 3 == k iff b_j = C₃^k b_i.

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (matched to higher_m_pinning_probe)
# ──────────────────────────────────────────────────────────────────────────────

M_VALUES = [1, 2, 3, 4]
L_CYCLE  = {m: 6 * m + 4 for m in M_VALUES}
L_EFF    = {m: 6 * m + 2 for m in M_VALUES}
D_VALUES = [8, 14, 20, 26]
MAX_CYCLES_PER_START = {1: 10, 2: 20, 3: 20, 4: 10}
NUM_STARTS = 12


# ──────────────────────────────────────────────────────────────────────────────
# Cycle generation (re-using existing infrastructure)
# ──────────────────────────────────────────────────────────────────────────────

def generate_cycles_for_m(m, max_total=200, time_budget_sec=60.0):
    L = L_CYCLE[m]
    center = (N_SUPER // 2,) * 3
    cycles = []
    t0 = time.time()
    for bond_idx in range(NUM_STARTS):
        if time.time() - t0 > time_budget_sec or len(cycles) >= max_total:
            break
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + int(dc[d]) for d in range(3))
        if not in_bounds(tgt_cell):
            continue
        start_edge = (prim_bond[0], center, prim_bond[1], tgt_cell)
        cs = h16.find_cycles_at_length(start_edge, L,
                                       max_cycles=MAX_CYCLES_PER_START[m])
        cycles.extend(cs)
    return cycles[:max_total], (time.time() - t0)


# ──────────────────────────────────────────────────────────────────────────────
# Z₃-shift classification of same-orbit pairs
# ──────────────────────────────────────────────────────────────────────────────

def count_z3_shifted_pairs(cycle, target_distance):
    """For each (i, j) pair in the cycle with j-i ≡ d mod L, classify by
    Z₃-shift relationship if both are in the same C₃ orbit:

        Z₃¹: b_j = C₃ b_i  (position diff +1 mod 3)
        Z₃²: b_j = C₃² b_i (position diff +2 mod 3)
        same: b_j = b_i (position diff 0; degenerate, not counted)

    Returns (count_z31, count_z32) and small example lists.
    """
    sig = d14probe.cycle_orbit_signature(cycle)   # list of (prim_idx, orbit_id)
    L = len(cycle)
    cnt_z31 = 0
    cnt_z32 = 0
    examples_z31 = []
    examples_z32 = []
    for i in range(L):
        j = (i + target_distance) % L
        if i >= j:
            continue
        pt_i, orb_i = sig[i]
        pt_j, orb_j = sig[j]
        if orb_i != orb_j or pt_i == pt_j:
            continue
        # Both in same orbit, different bonds: identify Z₃ relation.
        # type_label[pt] = (orbit_id, position 0/1/2)
        _, pos_i = type_label[pt_i]
        _, pos_j = type_label[pt_j]
        delta = (pos_j - pos_i) % 3
        if delta == 1:
            cnt_z31 += 1
            if len(examples_z31) < 2:
                examples_z31.append((i, j, pt_i, pt_j, orb_i))
        elif delta == 2:
            cnt_z32 += 1
            if len(examples_z32) < 2:
                examples_z32.append((i, j, pt_i, pt_j, orb_i))
        # delta == 0 means same bond, already filtered above
    return cnt_z31, cnt_z32, examples_z31, examples_z32


def aggregate_z3_pairs(cycles, d_values):
    """For all cycles + all d, return {d: (cnt_z31, cnt_z32)}."""
    counts = {d: [0, 0] for d in d_values}
    for c in cycles:
        for d in d_values:
            if d >= len(c):
                continue
            z31, z32, _, _ = count_z3_shifted_pairs(c, d)
            counts[d][0] += z31
            counts[d][1] += z32
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("V_ub bridge Z₃-shift classifier — same-orbit pairs split by ΔGen index")
    print("=" * 80)
    print()
    print(f"  N_SUPER = {N_SUPER}")
    print(f"  Hypothesis: pairs (b₁, b₂) with b₂=C₃b₁ → ΔGen=1 (V_cb-class);")
    print(f"              pairs with b₂=C₃²b₁ → ΔGen=2 (V_ub-class).")
    print()

    table_z31 = {}    # m -> {d: count of Z₃-shift pairs}
    table_z32 = {}    # m -> {d: count of Z₃²-shift pairs}
    cycle_counts = {}

    for m in M_VALUES:
        L = L_CYCLE[m]
        d_lim = [d for d in D_VALUES if d < L]
        cycles, dt = generate_cycles_for_m(m)
        cycle_counts[m] = len(cycles)
        print(f"  --- m={m}, L_cycle={L}, L_eff={L_EFF[m]} ({len(cycles)} cycles, {dt:.1f}s) ---")
        counts = aggregate_z3_pairs(cycles, d_lim)
        z31_dict = {d: counts[d][0] for d in d_lim}
        z32_dict = {d: counts[d][1] for d in d_lim}
        table_z31[m] = z31_dict
        table_z32[m] = z32_dict
        for d in D_VALUES:
            if d in d_lim:
                z31 = z31_dict[d]; z32 = z32_dict[d]; tot = z31 + z32
                ratio_str = "—" if tot == 0 else f"{z31}:{z32} ({z31/tot:.2f}/{z32/tot:.2f})"
                print(f"    d={d:>2d}: Z₃¹={z31:>4d}  Z₃²={z32:>4d}  total={tot:>4d}  ratio Z₃¹:Z₃² = {ratio_str}")
            else:
                print(f"    d={d:>2d}: — (d ≥ L_cycle)")
        print()

    # Side-by-side summary
    print("=" * 80)
    print("SUMMARY — Z₃¹ pairs (ΔGen=1, V_cb-class) at d (rows) × m (cols)")
    print("=" * 80)
    print()
    header = f"  {'d \\ m':>10s} " + " ".join(f"{m:>10d}" for m in M_VALUES)
    print(header)
    print(f"  {'L_cycle':>10s} " + " ".join(f"{L_CYCLE[m]:>10d}" for m in M_VALUES))
    print(f"  {'cycles':>10s} " + " ".join(f"{cycle_counts[m]:>10d}" for m in M_VALUES))
    print()
    for d in D_VALUES:
        row = f"  d={d:>3d}    "
        for m in M_VALUES:
            if d < L_CYCLE[m]:
                row += f" {table_z31[m][d]:>10d}"
            else:
                row += f" {'—':>10s}"
        print(row)
    print()

    print("=" * 80)
    print("SUMMARY — Z₃² pairs (ΔGen=2, V_ub-class) at d (rows) × m (cols)")
    print("=" * 80)
    print()
    print(header)
    print(f"  {'L_cycle':>10s} " + " ".join(f"{L_CYCLE[m]:>10d}" for m in M_VALUES))
    print(f"  {'cycles':>10s} " + " ".join(f"{cycle_counts[m]:>10d}" for m in M_VALUES))
    print()
    for d in D_VALUES:
        row = f"  d={d:>3d}    "
        for m in M_VALUES:
            if d < L_CYCLE[m]:
                row += f" {table_z32[m][d]:>10d}"
            else:
                row += f" {'—':>10s}"
        print(row)
    print()

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostic interpretation
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("STRUCTURAL DIAGNOSIS")
    print("=" * 80)
    print()

    print("  Test 1 — Z₃¹/Z₃² parity within same-orbit pairs:")
    for m in M_VALUES:
        for d in D_VALUES:
            if d not in table_z31.get(m, {}):
                continue
            z31 = table_z31[m][d]; z32 = table_z32[m][d]
            tot = z31 + z32
            if tot == 0:
                continue
            balance = abs(z31 - z32) / tot
            label = "BALANCED" if balance < 0.2 else "SKEWED"
            print(f"    (m={m}, d={d:>2d}): Z₃¹={z31}, Z₃²={z32}, |Δ|/tot={balance:.2f} → {label}")
    print()

    print("  Test 2 — Diagonal d=L_eff(m) restricted to ΔGen=k segregation:")
    print("    Hypothesis: (m, d=L_eff(m)) hosts have Z₃¹ pairs only if m ≡ 1 mod 3,")
    print("                Z₃² pairs only if m ≡ 2 mod 3, neither if m ≡ 0 mod 3.")
    for m in M_VALUES:
        d = L_EFF[m]
        if d not in table_z31.get(m, {}):
            continue
        z31 = table_z31[m][d]; z32 = table_z32[m][d]
        m_mod = m % 3
        expect = {1: "Z₃¹>0, Z₃²=0", 2: "Z₃¹=0, Z₃²>0", 0: "both=0"}[m_mod]
        actual = f"Z₃¹={z31}, Z₃²={z32}"
        match = ((m_mod == 1 and z32 == 0 and z31 > 0) or
                 (m_mod == 2 and z31 == 0 and z32 > 0) or
                 (m_mod == 0 and z31 == 0 and z32 == 0))
        verdict = "✓ matches hypothesis" if match else "✗ refutes hypothesis"
        print(f"    (m={m}, m mod 3={m_mod}, d={d}): expected {expect:>15s}, got {actual:>22s} — {verdict}")
    print()

    print("  Test 3 — Total Z₃¹ vs Z₃² counts across all (m, d):")
    total_z31 = sum(table_z31[m][d] for m in M_VALUES for d in table_z31[m])
    total_z32 = sum(table_z32[m][d] for m in M_VALUES for d in table_z32[m])
    print(f"    Total Z₃¹ pairs: {total_z31}")
    print(f"    Total Z₃² pairs: {total_z32}")
    if total_z31 + total_z32 > 0:
        ratio = total_z31 / (total_z31 + total_z32)
        print(f"    Z₃¹ fraction: {ratio:.3f}")
        print(f"    (Random chance would give ≈ 0.5 for any large sample.)")


if __name__ == "__main__":
    main()
