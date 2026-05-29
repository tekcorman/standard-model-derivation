#!/usr/bin/env python3
"""
proofs/_archive/a5b_waterline_savings.py

Standalone proof that the A2 waterline retains ALL girth-cycle winding
classes, giving a geometric series V = u^L / (1 − u^L), not a single term.

Gate type: Type 1+2 (A2 waterline + explicit arithmetic + CAS count).

THEOREM
───────
For any species-pair coupling on the srs Hashimoto graph with:
  u  = (k*−1)/k* = 2/3   (NB survival amplitude per step)
  L  = g − n_fixed = 8   (minimum hop length; from vcb_branch_measure.py)
  N  = number of distinct primitive L-step species-pair cycles (≤ 120)

the n-th winding class has raw description length nL bits and model
description length ≤ log₂N + log₂(n+1) bits. Savings > 0 for all n ≥ 1
iff L > log₂N + 1, which holds since 8 > log₂(120) + 1 ≈ 7.91.

All windings are above the A2 waterline →
  V = Σ_{n=1}^∞ u^{nL} = u^L / (1 − u^L)

For V_cb: V_cb = (2/3)^8 / (1−(2/3)^8) = 256/6305 ≈ 40.60 × 10^{-3}.

DERIVATION
──────────
Step 1 [Type 2]: Per-step raw cost.
  On a k*-regular graph, each NB step has k*−1 choices (non-backtracking).
  Per-step raw description length: log₂(k*−1) = log₂2 = 1 bit.
  An n-winding walk of L steps each costs:  L_raw(n) = n·L bits.

Step 2 [Type 2 + CAS]: Model description length.
  The model class is "repeat the primitive b→c L-cycle n times."
  Encoding cost:
    (a) Identify which primitive L-cycle: ≤ log₂(N_primitive) bits.
        From proofs/flavor/vcb_hashimoto_bfs.py [CAS, session 13]:
        N_girth_total = 120  (total girth-10 NB cycles in srs 8³ supercell)
        N_species_paired = 20  (same-orbit b1→b2 pairs at cycle-distance 8)
        Conservative upper bound: log₂(120) ≈ 6.91 bits.
    (b) Specify winding count n: log₂(n+1) bits (prefix-free code for n≥1).
  Total: L_model(n) ≤ log₂(120) + log₂(n+1) bits.

Step 3 [Type 1+2, A2 waterline]: Savings.
  savings(n) = L_raw(n) − L_model(n) ≥ 8n − log₂(120) − log₂(n+1)
  savings(1) ≥ 8 − log₂(120) − log₂(2) = 8 − 6.907 − 1 = 0.093 > 0  ✓
  savings(n) → +∞ as n → ∞.

Step 4 [Type 2, algebra]: Geometric series.
  All windings n ≥ 1 above waterline → all retained by A2.
    V = Σ_{n=1}^∞ u^{nL} = u^L / (1 − u^L)   (converges since u^L < 1)

NOTE ON THE PRIMITIVE CYCLE ITSELF (n=1):
  The primitive n=1 walk is also retained because savings(1) > 0.
  The "single-term" prescription V = u^L = (2/3)^8 ≈ 39.02×10^{-3}
  discards all n≥2 windings, which are all above the waterline.
  It is therefore incomplete under A2.
"""

import math
import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))
from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial

# ────────────────────────────────────────────────────────────────────────
# Inputs  [Type 4 — upstream closed files]
# ────────────────────────────────────────────────────────────────────────

d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)

assert k == 3 and g == 10

n_fixed = 2                   # endpoint count (vcb_nfixed_proof.py, Type 2)
L = g - n_fixed               # = 8
u = Fraction(k - 1, k)        # = 2/3

# CAS-verified counts (proofs/flavor/vcb_hashimoto_bfs.py, session 13)
N_girth_total   = 120   # total girth-10 NB cycles in srs (8³ supercell)
N_species_pairs = 20    # same-orbit (b1,b2) pairs at cycle-distance L

print("=" * 65)
print("  A5(b) waterline savings — geometric series proof")
print("=" * 65)
print(f"\n  k* = {k},  g = {g},  L = g − n_fixed = {L},  u = {u}")
print(f"  N_girth_total   = {N_girth_total}  (vcb_hashimoto_bfs.py)")
print(f"  N_species_pairs = {N_species_pairs}  (same-orbit b→c pairs)")

# ────────────────────────────────────────────────────────────────────────
# Step 1 — Per-step raw cost  [Type 2]
# ────────────────────────────────────────────────────────────────────────

bits_per_step = math.log2(k - 1)   # = log₂2 = 1 bit
assert abs(bits_per_step - 1.0) < 1e-15

print(f"\n  Step 1 — raw description length:")
print(f"    bits_per_NB_step = log₂(k*−1) = log₂({k-1}) = {bits_per_step:.1f}")
for n_wind in [1, 2, 3, 5, 10]:
    L_raw = n_wind * L * bits_per_step
    print(f"    L_raw({n_wind:2d} windings) = {n_wind}×{L} = {L_raw:.1f} bits")

# ────────────────────────────────────────────────────────────────────────
# Step 2 — Model description length  [Type 2 + CAS]
# ────────────────────────────────────────────────────────────────────────

# Upper bound: log₂(N_girth_total) + log₂(n+1)
# (Conservative: use all girth-10 cycles as the cycle pool)
log2_N = math.log2(N_girth_total)

print(f"\n  Step 2 — model description length upper bound:")
print(f"    log₂(N_girth_total) = log₂({N_girth_total}) = {log2_N:.4f} bits  (cycle ID)")
for n_wind in [1, 2, 3, 5, 10]:
    L_model_ub = log2_N + math.log2(n_wind + 1)
    print(f"    L_model_ub({n_wind:2d} windings) ≤ {log2_N:.4f} + log₂({n_wind+1}) = {L_model_ub:.4f} bits")

# ────────────────────────────────────────────────────────────────────────
# Step 3 — Savings for each winding  [Type 1+2, A2]
# ────────────────────────────────────────────────────────────────────────

print(f"\n  Step 3 — savings(n) = L_raw(n) − L_model_ub(n):")
print(f"    {'n':>4}  {'L_raw':>8}  {'L_model_ub':>12}  {'savings_lb':>12}  above_waterline")
print(f"    {'─'*4}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*15}")

all_positive = True
for n_wind in range(1, 21):
    L_raw = n_wind * L
    L_model_ub = log2_N + math.log2(n_wind + 1)
    savings_lb = L_raw - L_model_ub
    above = savings_lb > 0
    if not above:
        all_positive = False
    marker = "✓" if above else "✗ FAIL"
    print(f"    {n_wind:>4}  {L_raw:>8.1f}  {L_model_ub:>12.4f}  {savings_lb:>12.4f}  {marker}")

assert all_positive, "Some winding has savings ≤ 0 — waterline argument fails"

# Prove for all n analytically
# savings(n) = 8n - log₂(120) - log₂(n+1) ≥ 8n - log₂(120) - log₂(n+1)
# For n≥1: 8n ≥ 8; log₂(120) + log₂(n+1) = log₂(120(n+1)) ≤ log₂(120n·2) = log₂(240n)
# savings ≥ 8n - log₂(240n) = 8n - log₂(240) - log₂(n)
# d/dn [8n - log₂(n)/ln2] = 8 - 1/(n·ln2) > 0 for all n ≥ 1
# So savings is increasing → min at n=1 → savings(1) > 0 (verified above).
savings_n1 = L - log2_N - math.log2(2)
print(f"\n  ✓ All {20} checked windings above waterline.")
print(f"  ✓ savings(1) = {L} − {log2_N:.3f} − {math.log2(2):.3f} = {savings_n1:.3f} > 0")
print(f"  ✓ savings(n) is increasing (d/dn = 8 − 1/(n·ln2) > 0 for all n ≥ 1)")
print(f"  ✓ Therefore ALL windings n ≥ 1 are above the A2 waterline.")

# ────────────────────────────────────────────────────────────────────────
# Step 4 — Geometric series  [Type 2, algebra]
# ────────────────────────────────────────────────────────────────────────

alpha1 = u ** L                          # = (2/3)^8 = 256/6561
V_series = alpha1 / (1 - alpha1)         # = 256/6305

assert alpha1 == Fraction(256, 6561)
assert V_series == Fraction(256, 6305)

V_single_term = float(alpha1)            # single-term prescription (incomplete)
V_geometric   = float(V_series)          # correct geometric series

pdg_central = 40.5e-3
pdg_sigma   = 1.5e-3

dev_single  = (V_single_term  - pdg_central) / pdg_sigma
dev_geom    = (V_geometric    - pdg_central) / pdg_sigma

print(f"\n  Step 4 — geometric series:")
print(f"    α₁_bare = u^L = {alpha1} ≈ {float(alpha1)*1e3:.4f} × 10^-3")
print(f"    V = α₁/(1−α₁) = {V_series} ≈ {V_geometric*1e3:.4f} × 10^-3")
print(f"\n  Comparison with PDG 2024 exclusive ({pdg_central*1e3:.1f} ± {pdg_sigma*1e3:.1f} × 10^-3):")
print(f"    Single-term  (2/3)^8 :   {V_single_term*1e3:.4f} × 10^-3  ({dev_single:+.2f}σ)  [INCOMPLETE]")
print(f"    Geometric series      :   {V_geometric*1e3:.4f} × 10^-3  ({dev_geom:+.2f}σ)  [CORRECT]")

print(f"\n" + "=" * 65)
print(f"  RESULT: V_cb = {V_series} = {V_geometric*1e3:.4f} × 10^-3  ({dev_geom:+.2f}σ)")
print(f"")
print(f"  The single-term prescription discards all n≥2 windings.")
print(f"  Under A2, windings n≥2 also clear the waterline (savings > 0)")
print(f"  → single-term is incomplete; geometric series is correct.")
print(f"")
print(f"  Gate: Type 1 (A2) + Type 2 (algebra) + CAS (vcb_hashimoto_bfs.py).")
print("=" * 65)
