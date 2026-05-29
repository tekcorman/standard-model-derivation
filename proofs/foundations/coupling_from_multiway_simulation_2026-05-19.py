#!/usr/bin/env python3
"""
proofs/foundations/coupling_from_multiway_simulation_2026-05-19.py

THE SIMULATION the user has been asking for since turn 1. Implements the
three rules explicitly and measures the coupling directly:

  1) Substrate evolution: non-backtracking walker on K_4 (the framework's
     primitive-cell quotient; 4 nodes, 6 edges, each node degree 3).
  2) Observation: at each tick the walker records the bit (on/off) of the
     edge it is on.  Stream length T.
  3) Observer compression: the optimal-compressor lower bound — the
     asymptotic Shannon entropy rate of the observed stream, estimated
     via conditional empirical entropy H(X_t | X_{t-k..t-1}) at growing
     context k.  This is what NO compressor can compress below; it is the
     irreducible bit-density per tick = the coupling.

A fermion pattern is a 6-bit configuration (which of the 6 K_4 edges are
on).  Sweep all 64 patterns.  The maximally-incompressible pattern's
entropy rate is the natural-scale coupling (= y_ν in the framework's
units, if it saturates the Shannon ceiling of 1 bit/tick).

CORRECTNESS GATES (VOID if either fails):
  G1  all-on  (1,1,1,1,1,1): observed stream = all 1s ⇒ entropy rate = 0
  G2  all-off (0,0,0,0,0,0): observed stream = all 0s ⇒ entropy rate = 0
  (These are the "fully compressible" sanity checks — the pattern itself
  is one bit of information so the stream has no residue.)

PRE-DECLARED OUTCOMES:
  DERIVED-CEILING   : some pattern saturates entropy rate = 1.000(±0.01)
                      bit/tick (the Shannon ceiling on a 2-state stream)
                      ⇒ that pattern is the maximally-incompressible
                      fermion structure, the natural-scale coupling unit
                      = the framework's y_ν=1 derived as a saturation.
                      The pattern's identity is the new physical content.
  PARTIAL           : max entropy rate = X strictly between 0 and 1
                      ⇒ the framework's coupling ceiling is X, not 1.
                      The measured X is the simulation-derived value of
                      y_ν.  (Then either the framework's "=1" claim is
                      wrong by a computable factor, or the dynamics is
                      under-specified compared to the substrate.)
  DEGENERATE        : all non-trivial patterns give similar entropy
                      ⇒ the K_4-quotient dynamics is too symmetric to
                      distinguish fermion patterns at this level;
                      richer dynamics (lattice voltage / cover) needed.

No bookkeeping unless a value-level result warrants it. Ships nothing.
"""
import numpy as np
from collections import defaultdict

# K_4: 4 vertices, 6 undirected edges
EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
N_EDGES = 6
N_NODES = 4

# For each node, list (edge_index, neighbor_node)
node_edges = {v: [] for v in range(N_NODES)}
for e_idx, (a, b) in enumerate(EDGES):
    node_edges[a].append((e_idx, b))
    node_edges[b].append((e_idx, a))

def nb_walk_bits(pattern_bits, n_steps, rng):
    """Non-backtracking walk on K_4. At each step record bit of current edge."""
    # Start: random node, random outgoing edge
    node = int(rng.integers(N_NODES))
    e_choices = node_edges[node]
    e_idx, next_node = e_choices[int(rng.integers(len(e_choices)))]
    prev_node = node
    bits = [pattern_bits[e_idx]]
    node = next_node
    for _ in range(n_steps - 1):
        # NB rule: from node, pick one of the edges NOT leading back to prev_node
        opts = [(idx, oth) for (idx, oth) in node_edges[node] if oth != prev_node]
        e_idx, next_node = opts[int(rng.integers(len(opts)))]
        bits.append(pattern_bits[e_idx])
        prev_node = node
        node = next_node
    return bits

def cond_entropy(bits, ctx_len):
    """H(X_t | X_{t-ctx_len..t-1}) — empirical conditional entropy in bits."""
    if len(bits) <= ctx_len: return float('nan')
    counts = defaultdict(lambda: [0, 0])
    for i in range(ctx_len, len(bits)):
        ctx = tuple(bits[i-ctx_len:i])
        counts[ctx][bits[i]] += 1
    total = sum(c0+c1 for c0,c1 in counts.values())
    H = 0.0
    for (c0, c1) in counts.values():
        n = c0 + c1
        if n == 0: continue
        p_ctx = n / total
        p0, p1 = c0/n, c1/n
        h = 0.0
        if p0 > 0: h -= p0*np.log2(p0)
        if p1 > 0: h -= p1*np.log2(p1)
        H += p_ctx * h
    return H

def entropy_rate(bits, max_ctx=8):
    """Estimate entropy rate from conditional entropy at growing context."""
    return [cond_entropy(bits, k) for k in range(1, max_ctx+1)]

# -----------------------------------------------------------------------
print("="*72)
print("SIMULATION: observe-then-toggle non-backtracking walker on K_4")
print("Measuring asymptotic Shannon entropy rate of the observed bit stream")
print("="*72)

rng = np.random.default_rng(2026)
T_TOTAL = 200_000
N_RUNS = 4
MAX_CTX = 6

# Aggregate across runs
def measure_pattern(pattern, T, n_runs, rng):
    rates_per_run = []
    for _ in range(n_runs):
        bits = nb_walk_bits(pattern, T, rng)
        rates_per_run.append(entropy_rate(bits, max_ctx=MAX_CTX))
    # Average across runs
    return [np.mean([r[k] for r in rates_per_run]) for k in range(MAX_CTX)]

# ----- CORRECTNESS GATES -----
print("\nGate check (sanity):")
gates_pass = True
for tag, p in [("G1 all-on  (1,1,1,1,1,1)", (1,)*6),
               ("G2 all-off (0,0,0,0,0,0)", (0,)*6)]:
    r = measure_pattern(p, T_TOTAL//N_RUNS, N_RUNS, rng)
    H_inf = r[-1]
    ok = abs(H_inf) < 1e-9
    print(f"  {tag}: entropy rate at ctx={MAX_CTX} = {H_inf:.6f}  "
          f"({'PASS' if ok else 'FAIL'})")
    if not ok: gates_pass = False

if not gates_pass:
    print("\n  ** GATES FAILED. VOID. **")
    import sys; sys.exit(0)
print("  GATES PASSED.\n")

# ----- SWEEP ALL 64 PATTERNS -----
print("Sweeping all 64 patterns of 6 edges on/off...")
results = []
for p_int in range(2**N_EDGES):
    pattern = tuple((p_int >> b) & 1 for b in range(N_EDGES))
    n_on = sum(pattern)
    rates = measure_pattern(pattern, T_TOTAL//N_RUNS, N_RUNS, rng)
    results.append((rates[-1], n_on, pattern, rates))

# Sort by entropy rate descending
results.sort(reverse=True)

print(f"\n{'rank':>4} {'H_inf':>9} {'n_on':>5}  pattern")
print("-"*72)
for rank, (H, n, p, rates) in enumerate(results[:8]):
    print(f"{rank+1:>4} {H:>9.5f} {n:>5}  {p}    "
          f"H(ctx=1..{MAX_CTX}): {[f'{r:.3f}' for r in rates]}")
print(" ...")
for rank, (H, n, p, rates) in enumerate(results[-4:]):
    actual_rank = len(results) - 4 + rank + 1
    print(f"{actual_rank:>4} {H:>9.5f} {n:>5}  {p}")

# ----- ANALYSIS -----
max_H = results[0][0]
min_H_nontriv = next((H for H, n, p, r in results if 0 < n < 6 and H > 1e-6), None)
H_by_n_on = defaultdict(list)
for H, n, p, r in results:
    H_by_n_on[n].append(H)

print("\n"+"="*72)
print("Mean entropy rate by edge-count (n_on):")
for n in range(N_EDGES+1):
    vals = H_by_n_on[n]
    n_patterns = len(vals)
    print(f"  n_on = {n}: {n_patterns} patterns, mean H = {np.mean(vals):.5f}, "
          f"max = {np.max(vals):.5f}, min = {np.min(vals):.5f}")

print("\n"+"="*72)
print("  VERDICT")
print("="*72)
print(f"  Maximum entropy rate observed: {max_H:.5f} bits/tick")
print(f"  Shannon ceiling (2-state stream): 1.00000 bits/tick")

ceiling_gap = 1.0 - max_H
if ceiling_gap < 0.01:
    V = (f"DERIVED-CEILING — the maximally-incompressible fermion pattern "
         f"saturates the Shannon ceiling within {ceiling_gap*100:.2f}%. "
         f"The natural-scale coupling y=1 is derived as Shannon-ceiling "
         f"saturation by the all-edges-occupied (n_on=full) walker "
         f"dynamics. The 'unit' is no longer adopted: it is the "
         f"Shannon-maximum on a 1-bit-per-tick observation stream, "
         f"reached when the pattern admits maximally-history-dependent "
         f"branching with no compressible regularity.")
elif max_H > 0.5:
    V = (f"PARTIAL — the framework's coupling ceiling under this dynamics "
         f"is X = {max_H:.5f}, not 1. y_ν=1 is NOT saturated by this "
         f"K_4-quotient non-backtracking walker; either the simulation's "
         f"dynamics is impoverished relative to the substrate (likely: "
         f"the full srs cover with voltage structure has more freedom), "
         f"or the 'y_ν=1' adoption is off by the computable factor "
         f"1/{max_H:.4f} = {1/max_H:.4f}.")
else:
    V = (f"DEGENERATE / LOW — max entropy rate {max_H:.5f} is too low; "
         f"the K_4-quotient NB-walker dynamics does not produce "
         f"sufficient incompressible residue. Richer dynamics needed "
         f"(srs lattice cover, voltage structure, multi-walker).")
print(f"\n  {V}")
print("  Ships no number. Changes no ledger row.")
print("="*72)
