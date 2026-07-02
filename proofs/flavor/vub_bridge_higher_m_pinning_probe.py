#!/usr/bin/env python3
"""
proofs/flavor/vub_bridge_higher_m_pinning_probe.py

PURPOSE
-------
Step 1 of Path T in an internal working note.

Extend `vub_bridge_d14_pinning_probe.py` to higher-m multi-cycle hosts to
test whether the framework's empirical "V_cb = m=1 only; V_ub = Σ_{m≥2}"
split has structural support in pinning topology.

For each multi-cycle host class m ∈ {1, 2, 3, 4}, with L_cycle(m) = 6m + 4
(from `theorem_bridge_functoriality_lemma.md`'s seam construction; CAS-
verified for m=1 trivially and m=2 via `hashimoto_16cycle_decomposition.py`),
count same-C₃-orbit pinned pairs at every candidate cycle-distance
d ∈ {8, 14, 20, 26} (= L_eff(1), L_eff(2), L_eff(3), L_eff(4)).

CONTEXT
-------
The bridge functoriality lemma's §3 step F9 derives "ΔGen=k iff m ≡ k mod 3"
from k-fold composition of the Z₃ cyclic shift accumulating Z₃^m holonomy.
But `proofs/flavor/z3_holonomy_cycles.py` shows the Z₃ connection on srs
cycles is FLAT (all gauge-invariant holonomies vanish; bundle globally
trivializable). The mod-3 rule has no underlying mechanism — the structural
argument is REFUTED by the existing flat-connection theorem.

The framework's working numerical formula sums all m ≥ 2 for V_ub (matches
PDG combined V_ub at −0.26σ), but the structural reason is open. This probe
collects the CAS data needed to identify which m-host classes admit which
d-pinning topologies, so a replacement structural argument can be built.

QUESTIONS THIS PROBE ANSWERS
---------------------------
For each (m, d) pair in {1, 2, 3, 4} × {8, 14, 20, 26}:
  Q(m, d) — Do any L_cycle(m) hosts admit same-orbit pinned pairs at
  cycle-distance d?

Specifically:
  Q1. Does m=2 admit d=8 pinning?
      (Already known: yes, 60 pairs in 50 L=16 cycles per existing probe.)
      Implication if yes: m=2 hosts ALSO contribute to a V_cb-like amplitude
      at d=8. V_cb being "m=1 only" cannot be a strict pinning-topology
      consequence — there must be a separate selection rule.

  Q2. Does m=3 admit d=20 pinning?
      Implication if yes: m=3 hosts have a "V_ub-style" pinning at d=20 that
      corresponds to ΔGen=2 only if the lemma's mod-3 argument holds.
      Under flat Z₃ (z3_holonomy_cycles), the Z₃ phase is identity on m=3,
      so naively m=3 should mediate the diagonal — but if d=20 pinning has a
      different generation interpretation, the all-m sum becomes plausible.

  Q3. Does m=3 admit d=14 pinning?
      If yes, m=3 contributes to V_ub via the same d=14 mechanism as m=2.
      This would support the all-m≥2 reading via a "host-class-independent
      d=14 sum" structural argument.

  Q4. Does m=4 admit d=26 pinning?
      Tests whether the "L_eff(m) = 6m+2" pattern continues, or whether the
      d-pinning saturates.

GATE STATUS
-----------
EXPLORATORY CAS data collection. Reports raw counts; structural inferences
are made in the companion docs (v_ub_lemma_reconciliation.md follow-up).

Run with:
    PYTHONPATH=. python3 proofs/flavor/vub_bridge_higher_m_pinning_probe.py
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

# ──────────────────────────────────────────────────────────────────────────────
# Probe parameters
# ──────────────────────────────────────────────────────────────────────────────

# Host class m and the corresponding cycle length L_cycle(m) = 6m + 4
# m=1 is a single girth cycle (L=10); m≥2 are multi-cycle hosts.
M_VALUES = [1, 2, 3, 4]
L_CYCLE  = {m: 6 * m + 4 for m in M_VALUES}
L_EFF    = {m: 6 * m + 2 for m in M_VALUES}     # = L_cycle - n_fixed = L_cycle - 2

# All candidate pinning distances to test on each host
D_VALUES = [8, 14, 20, 26]

# DFS budget. The DFS in find_cycles_at_length is exponential in L; we cap
# at max_cycles per starting edge and at most NUM_STARTS distinct starts.
# Sample sizes need to be large enough to detect rare pinnings (m=2 had
# only 3 d=14 pairs in 50 cycles).
MAX_CYCLES_PER_START = {1: 10, 2: 20, 3: 20, 4: 10}     # sampling budget per m
NUM_STARTS = 12       # iterate over the 12 directed bonds at center


# ──────────────────────────────────────────────────────────────────────────────
# Cycle generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_cycles_for_m(m, max_total=200, time_budget_sec=60.0):
    """Find NB cycles of length L_cycle(m) starting from center cells.

    Returns a list of cycles. Bounded by max_total cycles or time_budget.
    """
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
        # Cap per-start budget so one slow start doesn't dominate
        cs = h16.find_cycles_at_length(start_edge, L,
                                       max_cycles=MAX_CYCLES_PER_START[m])
        cycles.extend(cs)
    return cycles[:max_total], (time.time() - t0)


# ──────────────────────────────────────────────────────────────────────────────
# Same-orbit pinning counts at every d
# ──────────────────────────────────────────────────────────────────────────────

def count_pinnings_at_all_d(cycles, d_values):
    """Aggregate same-orbit pinned-pair counts at every requested d over all
    given cycles."""
    counts = {d: 0 for d in d_values}
    examples_per_d = {d: [] for d in d_values}
    for c in cycles:
        for d in d_values:
            if d >= len(c):
                continue   # d larger than cycle length is meaningless
            cnt, exs = d14probe.count_same_orbit_pinning(c, d)
            counts[d] += cnt
            for ex in exs[:1]:
                if len(examples_per_d[d]) < 3:
                    examples_per_d[d].append((ex, len(c)))
    return counts, examples_per_d


# ──────────────────────────────────────────────────────────────────────────────
# Main probe
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("V_ub bridge higher-m probe — same-orbit pinning across m ∈ {1,2,3,4}")
    print("=" * 78)
    print()
    print(f"  N_SUPER = {N_SUPER} (8³ = 512 cells)")
    print(f"  Host class m → L_cycle(m) = 6m+4 → L_eff(m) = 6m+2")
    print(f"  Candidate pinning distances d ∈ {D_VALUES}")
    print()

    # Per-m: generate cycles and count pinnings at all d
    table = {}            # m -> {d: count}
    cycle_counts = {}     # m -> num cycles found
    timings = {}          # m -> seconds

    for m in M_VALUES:
        L = L_CYCLE[m]
        d_lim = [d for d in D_VALUES if d < L]
        print(f"  --- m={m}, L_cycle={L}, L_eff={L_EFF[m]} ---")

        cycles, dt = generate_cycles_for_m(m)
        cycle_counts[m] = len(cycles)
        timings[m] = dt
        print(f"    cycles found: {len(cycles)} (budget {dt:.1f}s)")

        counts, examples = count_pinnings_at_all_d(cycles, d_lim)
        table[m] = counts

        for d in D_VALUES:
            if d in d_lim:
                cnt = counts[d]
                print(f"    same-orbit pairs at d={d:>2d}: {cnt}")
                if cnt > 0 and examples[d]:
                    ex, cyc_len = examples[d][0]
                    i, j, pt_i, pt_j, orb_id = ex
                    print(f"      example: pos {i} (prim_type={pt_i}, orbit {orb_id})"
                          f" ↔ pos {j} (prim_type={pt_j}, orbit {orb_id}) on L={cyc_len}")
            else:
                print(f"    same-orbit pairs at d={d:>2d}: — (d ≥ L_cycle, N/A)")
        print()

    # Summary table
    print("=" * 78)
    print("SUMMARY — same-orbit pinning counts at d (rows) by host class m (cols)")
    print("=" * 78)
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
                row += f" {table[m][d]:>10d}"
            else:
                row += f" {'—':>10s}"
        print(row)
    print()

    # Diagnostic interpretation
    print("=" * 78)
    print("DIAGNOSIS")
    print("=" * 78)
    print()

    # Q1: Does m=2 admit d=8 pinning?
    if 2 in table and table[2].get(8, 0) > 0:
        print(f"  Q1 [d=8 pinning on m=2]: YES, {table[2][8]} pairs")
        print(f"      → m=2 hosts also support V_cb-style pinning at d=8.")
        print(f"      → 'V_cb = m=1 only' is NOT a strict pinning-topology consequence.")
    else:
        print(f"  Q1 [d=8 pinning on m=2]: NO")
        print(f"      → 'V_cb = m=1 only' has direct pinning-topology support.")

    # Q2: Does m=3 admit d=20 pinning?
    if 3 in table and 20 in table[3]:
        cnt = table[3][20]
        if cnt > 0:
            print(f"  Q2 [d=20 pinning on m=3]: YES, {cnt} pairs")
            print(f"      → m=3 hosts admit a 'V_ub-style' pinning at d=20.")
        else:
            print(f"  Q2 [d=20 pinning on m=3]: NO")
            print(f"      → m=3 hosts do NOT admit d=20 pinning in this sample.")
    else:
        print(f"  Q2 [d=20 pinning on m=3]: not collected (insufficient cycles)")

    # Q3: Does m=3 admit d=14 pinning?
    if 3 in table and 14 in table[3]:
        cnt = table[3][14]
        if cnt > 0:
            print(f"  Q3 [d=14 pinning on m=3]: YES, {cnt} pairs")
            print(f"      → m=3 contributes to V_ub at d=14, supporting all-m≥2 reading.")
        else:
            print(f"  Q3 [d=14 pinning on m=3]: NO")
            print(f"      → m=3 does NOT contribute to V_ub at d=14 from this sample.")

    # Q4: Does m=4 admit d=26 pinning?
    if 4 in table and 26 in table[4]:
        cnt = table[4][26]
        if cnt > 0:
            print(f"  Q4 [d=26 pinning on m=4]: YES, {cnt} pairs")
            print(f"      → 'L_eff(m) = 6m+2' pattern continues to m=4.")
        else:
            print(f"  Q4 [d=26 pinning on m=4]: NO")
            print(f"      → L_eff pattern may saturate; d=26 not observed.")
    else:
        print(f"  Q4 [d=26 pinning on m=4]: not collected (insufficient cycles)")

    print()
    print("  Interpretation rules (for the companion structural analysis):")
    print("    - If Q1 = YES: V_cb = m=1 only is a SELECTION rule, not pinning-topology.")
    print("    - If Q2 = YES + Q3 = YES: m=3 hosts have multiple d-pinnings;")
    print("      the all-m≥2 reading needs a 'd-pinning class' tie to ΔGen.")
    print("    - If Q3 = YES + Q2 = NO: m=3 contributes to V_ub via d=14, not d=20.")
    print("    - If all-NO at higher m: pinning saturates at m=2; lemma's")
    print("      L_eff(m) = 6m+2 extension is unsupported beyond m=2.")
    print()


if __name__ == "__main__":
    main()
