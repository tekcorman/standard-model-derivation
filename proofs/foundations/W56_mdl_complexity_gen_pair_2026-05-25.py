#!/usr/bin/env python3
"""
W56 — MDL-complexity probe on §8's 3 CKM reading-types: does [GEN-PAIR]
       close substrate-internally via Boltzmann-weighted projection complexity?

Per an internal working note,
the §8 labeling residue decomposed in 2026-05-21 as [ORDER] ⊕ [GEN-PAIR],
with [ORDER] = δ = Need-B δ-physical (deep frontier, bounded surface
5-way-eliminated 2026-05-23) and [GEN-PAIR] = ordinal naming (mild,
non-blocking). The §8 CKM block has exactly 3 distinct reading-types of
G_NB = (I − u·B_NB)⁻¹:

  T1 = counting projection k*²/(g·N)            → V_us = 9/40
  T2 = resummed unit a/(1−a)                    → V_cb = 256/6305
  T3 = higher-winding multi-cycle host-sum      → V_ub ≈ 3.767×10⁻³

HYPOTHESIS: the natural MDL description-length complexity ordering on these
3 reading-types is ℓ(T1) < ℓ(T2) < ℓ(T3). Under A2-T MDL with A5(b)
amplitude ∝ exp(−ℓ) Boltzmann weighting, this would reproduce the observed
CKM hierarchy V_us > V_cb > V_ub substrate-internally.

The probe tests this against multiple MDL-cost heuristics. The hypothesis
PASSES at the rank-ordering level if any heuristic gives the right direction;
it PASSES at the magnitude level only if the heuristic also reproduces the
log-spacings of (V_us, V_cb, V_ub) to within O(few %).

PRE-DECLARED GATES (per scoping doc §5):
  G1: each reading-type has a well-defined ℓ_i for ≥1 heuristic.
  G2: the ordering ℓ(T1) < ℓ(T2) < ℓ(T3) holds for ≥1 heuristic.
  G3: amplitude_i ∝ exp(−ℓ_i) produces ordering matching V_us > V_cb > V_ub
      *without* using PDG values as input.
  G4: log-spacings: log(V_us/V_cb) and log(V_cb/V_ub) match (ℓ_T2 − ℓ_T1)
      and (ℓ_T3 − ℓ_T2) within O(few %).

PRE-DECLARED ABORTS (per scoping doc §4):
  AB1: if no heuristic gives well-defined costs, abort.
  AB2: if all heuristics give wrong direction (G2 fails for all), abort.
  AB3: literal-claim — every step from MDL bits → amplitude must derive
       framework-internally. No fitting.
  AB4: if multiple heuristics give different observed-matching CKM
       hierarchies, abort with "labeling data-anchored across MDL orderings".
  AB5: if a clean closure requires NEW framework axioms, abort.
  AB6: scope-creep into Need-B δ-physical / [ORDER] = δ → abort + re-scope.
"""

from __future__ import annotations
import math
from fractions import Fraction

# ----------------------------------------------------------------------
# Observed CKM (PDG-derived; framework derives same values via §8) — used
# only for the post-hoc comparison gates G3 and G4, not as input.
# ----------------------------------------------------------------------
V_us_frame = Fraction(9, 40)                         # framework counting projection
V_cb_frame = Fraction(256, 6305)                     # framework resummed unit
V_ub_frame = 3.767e-3                                # framework higher-winding (irrational; float)

V_us = float(V_us_frame)                              # 0.225
V_cb = float(V_cb_frame)                              # 0.0406
V_ub = V_ub_frame                                     # 0.003767

log_V_us = math.log(V_us)
log_V_cb = math.log(V_cb)
log_V_ub = math.log(V_ub)

obs_logsp_12 = log_V_us - log_V_cb                    # = ℓ(T2) − ℓ(T1) if amp∝exp(−ℓ)
obs_logsp_23 = log_V_cb - log_V_ub                    # = ℓ(T3) − ℓ(T2) if amp∝exp(−ℓ)

print("=" * 78)
print("W56 — MDL-complexity probe on §8 CKM reading-types")
print("=" * 78)
print()
print("Observed CKM magnitudes (framework values via §8 — not PDG inputs):")
print(f"  V_us = 9/40         = {V_us:.5f}")
print(f"  V_cb = 256/6305     = {V_cb:.5f}")
print(f"  V_ub ≈ 3.767e-3     = {V_ub:.5f}")
print()
print("Observed log-spacings (target for G4):")
print(f"  log(V_us/V_cb)      = {obs_logsp_12:.4f} nats")
print(f"  log(V_cb/V_ub)      = {obs_logsp_23:.4f} nats")
print()


# ----------------------------------------------------------------------
# Heuristic A — primitive-operation count
# ----------------------------------------------------------------------
# Count substrate-primitive atomic operations to specify each projection
# within the framework's MDL machinery.

# T1: counting projection k*²/(g·N) = 9/40
#   Primitives: k* (3), g (10), N (4) — 3 substrate integers
#   Ops: square (k*→k*²), product (g·N), ratio (k*²/(g·N)) — 3 ops
ℓA_T1 = 3 + 3   # 6 atomic descriptors

# T2: resummed unit a/(1-a)
#   Primitives: a = α₁_bare = (2/3)^8 — 1 primitive
#   Ops: resolvent operation Σ a^k → 1/(1-a) — 1 op; unit projection — 1 op;
#        the resolvent operation itself encodes infinite series — log-cost?
#   Conservative: 1 primitive + 2 ops + log₂(∞) → use a step-count proxy of 1
ℓA_T2 = 1 + 2 + 1   # = 4 atomic descriptors

# T3: higher-winding Σ_{m≥2} (2/3)^{6m+2} / (1 − (2/3)^{6m+2})
#   Primitives: a, the winding-cutoff m_min=2 — 2 primitives
#   Ops: per-m geometric series — 1 op per m; sum over m≥2 — 1 op;
#        per-m inner resolvent (each ≈ T2 internal cost ≈ 4); cutoff = 1 op
#   Cost: 2 + 1_(outer sum) + (inner T2-cost) + 1_(cutoff) ≈ 2 + 1 + 4 + 1 = 8
ℓA_T3 = 8

print("--- Heuristic A (primitive-operation count) ---")
print(f"  ℓ(T1) = {ℓA_T1}  (3 substrate ints + 3 ops)")
print(f"  ℓ(T2) = {ℓA_T2}  (1 primitive + 2 ops + 1 ∞-series tag)")
print(f"  ℓ(T3) = {ℓA_T3}  (2 primitives + outer-sum + inner-resolvent + cutoff)")
g1A = True
g2A = (ℓA_T1 < ℓA_T2 < ℓA_T3) is False and (ℓA_T2 < ℓA_T1)  # check the actual direction
g2A_correct = (ℓA_T1 < ℓA_T2 < ℓA_T3)
print(f"  G1 (well-defined cost): {g1A}")
print(f"  G2 ordering ℓ(T1)<ℓ(T2)<ℓ(T3): {g2A_correct}  → "
      f"{'PASS' if g2A_correct else 'FAIL'}")
# Note: Heuristic A gives 6 > 4 < 8, so T1 > T2 < T3 — NOT monotonic
# T2 has fewer atomic descriptors than T1 because the resolvent is a single
# named operation, while the counting ratio uses 3 integers + 3 ops.

# Amplitude ordering check (Boltzmann)
ampsA = {"T1": math.exp(-ℓA_T1), "T2": math.exp(-ℓA_T2), "T3": math.exp(-ℓA_T3)}
# Normalize
total = sum(ampsA.values())
ampsA = {k: v/total for k, v in ampsA.items()}
print(f"  Normalized amplitudes ∝ exp(−ℓ): T1={ampsA['T1']:.4f}, "
      f"T2={ampsA['T2']:.4f}, T3={ampsA['T3']:.4f}")
ordA_correct = ampsA["T1"] > ampsA["T2"] > ampsA["T3"]
print(f"  G3 amplitude ordering T1>T2>T3 (matches V_us>V_cb>V_ub): "
      f"{'PASS' if ordA_correct else 'FAIL'}")
print()


# ----------------------------------------------------------------------
# Heuristic B — algebraic-complexity (Kolmogorov-style)
# ----------------------------------------------------------------------
# ℓ(T_i) = log(smallest rational denominator) for the framework expression.
# Each rational expression has a smallest-integer-form denominator that
# captures the algebraic complexity.

# T1: 9/40 → denom 40
ℓB_T1 = math.log(40)
# T2: 256/6305 → denom 6305
ℓB_T2 = math.log(6305)
# T3: higher-winding host-sum — leading term Σ_{m≥2} contributes:
# m=2: (2/3)^14 / (1 − (2/3)^14) ≈ (2/3)^14 (1 + (2/3)^14 + ...)
# Denom-equivalent of the leading m=2 term: (3^14 − 2^14)/2^14 ≈ huge
# Use the LEADING-TERM denominator as the proxy
m2_num = 2**14
m2_denom_minus_one = (3**14 - 2**14)
ℓB_T3 = math.log(m2_denom_minus_one / 1.0)

print("--- Heuristic B (Kolmogorov / smallest-rational-denominator) ---")
print(f"  ℓ(T1) = ln(40)             = {ℓB_T1:.4f}")
print(f"  ℓ(T2) = ln(6305)           = {ℓB_T2:.4f}")
print(f"  ℓ(T3) = ln(3^14 − 2^14)   = {ℓB_T3:.4f}  (leading m=2 term)")
g1B = True
g2B = ℓB_T1 < ℓB_T2 < ℓB_T3
print(f"  G1 (well-defined cost): {g1B}")
print(f"  G2 ordering ℓ(T1)<ℓ(T2)<ℓ(T3): {g2B}  → "
      f"{'PASS' if g2B else 'FAIL'}")

# Amplitude via Boltzmann
ampsB = {"T1": math.exp(-ℓB_T1), "T2": math.exp(-ℓB_T2), "T3": math.exp(-ℓB_T3)}
# Note: exp(-ℓ) = 1/denom for this heuristic
total = sum(ampsB.values())
ampsB_norm = {k: v/total for k, v in ampsB.items()}
print(f"  Raw amplitudes 1/denom: T1={ampsB['T1']:.4e}, T2={ampsB['T2']:.4e}, "
      f"T3={ampsB['T3']:.4e}")
ordB_correct = ampsB_norm["T1"] > ampsB_norm["T2"] > ampsB_norm["T3"]
print(f"  G3 amplitude ordering T1>T2>T3 (matches V_us>V_cb>V_ub): "
      f"{'PASS' if ordB_correct else 'FAIL'}")

# G4: log-spacings under Heuristic B
predB_logsp_12 = ℓB_T2 - ℓB_T1
predB_logsp_23 = ℓB_T3 - ℓB_T2
err12_B = abs(predB_logsp_12 - obs_logsp_12) / abs(obs_logsp_12) * 100
err23_B = abs(predB_logsp_23 - obs_logsp_23) / abs(obs_logsp_23) * 100
print(f"  G4 log-spacings:")
print(f"    predicted ℓ(T2)−ℓ(T1) = {predB_logsp_12:.4f}  vs obs "
      f"log(V_us/V_cb) = {obs_logsp_12:.4f}  →  {err12_B:.1f}% off")
print(f"    predicted ℓ(T3)−ℓ(T2) = {predB_logsp_23:.4f}  vs obs "
      f"log(V_cb/V_ub) = {obs_logsp_23:.4f}  →  {err23_B:.1f}% off")
g4B = (err12_B < 10) and (err23_B < 10)
print(f"  G4 (log-spacings within 10%): {'PASS' if g4B else 'FAIL'}")
print()


# ----------------------------------------------------------------------
# Heuristic C — A2-T MDL waterline-style (bits to specify projection-type)
# ----------------------------------------------------------------------
# In A2-T MDL, each register holds a finite description; the cost to specify
# a "projection-type" is log₂(number of distinct projection-types available)
# plus the internal-complexity cost.
#
# Number of distinct G_NB index projections from §8:
#   {counting, resummed-Perron, resummed-unit, multi-cycle host-sum, Feshbach}
#   = 5 projection-types in §8's catalogue
# log₂(5) ≈ 2.32 bits to specify a projection.
#
# Then each projection has an INTERNAL complexity cost:
#   T1 (counting): the integer ratio k*²/(g·N) — 0 internal bits beyond
#     specifying which 3 substrate integers. ≈ 0.
#   T2 (resummed): the geometric series — 1 internal bit (specify the series
#     is unit-projected, not Perron-projected). ≈ 1.
#   T3 (higher-winding): the multi-cycle host-sum — log₂(possible cutoffs)
#     ≈ log₂(g − 2) = log₂(8) ≈ 3 bits. Plus 1 for the inner-resolvent
#     specification. ≈ 4.

ℓC_T1 = math.log2(5)               # ≈ 2.32 bits
ℓC_T2 = math.log2(5) + 1           # ≈ 3.32 bits
ℓC_T3 = math.log2(5) + 4           # ≈ 6.32 bits

# Convert to nats for comparison with the observed log-spacings
ℓC_T1_nats = ℓC_T1 * math.log(2)
ℓC_T2_nats = ℓC_T2 * math.log(2)
ℓC_T3_nats = ℓC_T3 * math.log(2)

print("--- Heuristic C (A2-T waterline + internal complexity) ---")
print(f"  ℓ(T1) = log₂(5) + 0       = {ℓC_T1:.4f} bits  ({ℓC_T1_nats:.4f} nats)")
print(f"  ℓ(T2) = log₂(5) + 1       = {ℓC_T2:.4f} bits  ({ℓC_T2_nats:.4f} nats)")
print(f"  ℓ(T3) = log₂(5) + 4       = {ℓC_T3:.4f} bits  ({ℓC_T3_nats:.4f} nats)")
g1C = True
g2C = ℓC_T1 < ℓC_T2 < ℓC_T3
print(f"  G1 (well-defined): {g1C}")
print(f"  G2 ordering: {g2C}  → {'PASS' if g2C else 'FAIL'}")

# G4: log-spacings in nats
predC_logsp_12 = ℓC_T2_nats - ℓC_T1_nats
predC_logsp_23 = ℓC_T3_nats - ℓC_T2_nats
err12_C = abs(predC_logsp_12 - obs_logsp_12) / abs(obs_logsp_12) * 100
err23_C = abs(predC_logsp_23 - obs_logsp_23) / abs(obs_logsp_23) * 100
print(f"  G4 log-spacings:")
print(f"    pred ℓ(T2)−ℓ(T1) = {predC_logsp_12:.4f} nats  vs obs = "
      f"{obs_logsp_12:.4f}  →  {err12_C:.1f}% off")
print(f"    pred ℓ(T3)−ℓ(T2) = {predC_logsp_23:.4f} nats  vs obs = "
      f"{obs_logsp_23:.4f}  →  {err23_C:.1f}% off")
g4C = (err12_C < 10) and (err23_C < 10)
print(f"  G4 (log-spacings within 10%): {'PASS' if g4C else 'FAIL'}")
print()


# ----------------------------------------------------------------------
# AB4 check: do multiple heuristics give different observed-matching CKM
# hierarchies?
# ----------------------------------------------------------------------
print("=" * 78)
print("AB4 audit — do multiple heuristics give DIFFERENT outcomes?")
print("=" * 78)
print(f"  Heuristic A G2 (rank ordering): {'PASS' if (ℓA_T1 < ℓA_T2 < ℓA_T3) else 'FAIL'}")
print(f"  Heuristic B G2 (rank ordering): {'PASS' if g2B else 'FAIL'}")
print(f"  Heuristic C G2 (rank ordering): {'PASS' if g2C else 'FAIL'}")
print()
print(f"  Heuristic A G4 (magnitudes):    skipped (G2 failed for A)")
print(f"  Heuristic B G4 (magnitudes):    {'PASS' if g4B else 'FAIL'}  "
      f"(errs {err12_B:.1f}% + {err23_B:.1f}%)")
print(f"  Heuristic C G4 (magnitudes):    {'PASS' if g4C else 'FAIL'}  "
      f"(errs {err12_C:.1f}% + {err23_C:.1f}%)")
print()

# AB4 fires if multiple distinct heuristics give different conclusions
heuristics_g2 = [(ℓA_T1 < ℓA_T2 < ℓA_T3), g2B, g2C]
heuristics_g4 = [None, g4B, g4C]
n_g2_pass = sum(1 for h in heuristics_g2 if h)
n_g4_pass = sum(1 for h in heuristics_g4 if h is True)
print(f"  G2 PASSES under {n_g2_pass}/3 heuristics")
print(f"  G4 PASSES under {n_g4_pass}/3 heuristics")
ab4_fires = (n_g2_pass > 0 and n_g2_pass < 3) or (n_g4_pass > 0 and n_g4_pass < 3)
if ab4_fires:
    print(f"  AB4 FIRES — heuristics disagree on rank or magnitude. The MDL")
    print(f"  ordering is NOT uniquely substrate-internal across heuristics.")
else:
    print(f"  AB4 does NOT fire — heuristics agree.")
print()


# ----------------------------------------------------------------------
# VERDICT
# ----------------------------------------------------------------------
print("=" * 78)
print("W56 VERDICT")
print("=" * 78)
print()
print("Gate summary:")
print(f"  G1 (well-defined ℓ): PASS for all 3 heuristics")
print(f"  G2 (rank ordering ℓ(T1)<ℓ(T2)<ℓ(T3)):")
print(f"    Heuristic A: FAIL — T2 is SIMPLER than T1 (resolvent is 1 named op)")
print(f"    Heuristic B (Kolmogorov): {'PASS' if g2B else 'FAIL'}")
print(f"    Heuristic C (A2-T waterline + internal): {'PASS' if g2C else 'FAIL'}")
print(f"  G3 (Boltzmann ordering matches V_us>V_cb>V_ub):")
print(f"    Heuristic A: FAIL (since G2 failed for A — amplitudes give T2>T1)")
print(f"    Heuristic B: {'PASS' if ordB_correct else 'FAIL'}")
print(f"    Heuristic C: ranks would match (since G2 PASS); ")
print(f"  G4 (log-spacings within 10%):")
print(f"    Heuristic B: {'PASS' if g4B else 'FAIL'} "
      f"(errs {err12_B:.1f}% + {err23_B:.1f}%)")
print(f"    Heuristic C: {'PASS' if g4C else 'FAIL'} "
      f"(errs {err12_C:.1f}% + {err23_C:.1f}%)")
print()
print("Honest reading:")
print()
print("  RANK ORDERING (G2): the substrate-natural ordering of §8 reading-")
print("  types by complexity is DEPENDENT on the MDL-cost heuristic.")
print("  - Heuristic A (primitive-op count): T2 < T1 (resolvent is more")
print("    compact than counting-ratio). FAILS the substrate hypothesis.")
print("  - Heuristic B (Kolmogorov denominator): T1 < T2 < T3 ✓")
print("  - Heuristic C (waterline + internal cost): T1 < T2 < T3 ✓")
print("  Two of three heuristics give the right rank, one does not.")
print("  This is borderline: NOT a clean 'rank is substrate-internal'.")
print()
print("  MAGNITUDE (G4): the log-spacings of (V_us, V_cb, V_ub) are NOT")
print("  matched by any tested MDL-cost heuristic to within 10%. The")
print("  observed log-spacings (1.71, 2.37 nats) do not correspond cleanly")
print("  to any MDL-bit pattern in the substrate primitive set.")
print()
print("  AB4 FIRES (or at minimum borderline-fires): the heuristics give")
print("  different outcomes (Heuristic A FAILS, B and C PASS rank but FAIL")
print("  magnitude). The labeling residue [GEN-PAIR] does NOT close as a")
print("  clean substrate-internal MDL ordering.")
print()
print("VERDICT: HONEST NEGATIVE on the magnitude side (G4 fails universally);")
print("BORDERLINE on the rank side (G2 fails for one of 3 heuristics).")
print()
print("Implication for §8 closure:")
print("  [GEN-PAIR] remains 'mild, ordinal, non-blocking' — the prior C_36-")
print("  twist verdict (2026-05-21) stands. The MDL-complexity ordering")
print("  gives the DIRECTION of the CKM hierarchy under 2 of 3 heuristics")
print("  but cannot reproduce the magnitudes. The §8 reading magnitudes")
print("  come from the G_NB algebra itself, not from MDL bit-counting.")
print()
print("  This sharpens the [GEN-PAIR] residue: it is irreducibly ORDINAL —")
print("  the framework derives the magnitudes (V_us, V_cb, V_ub) and the")
print("  ordering directly via G_NB's projection algebra; the additional")
print("  step of assigning each AMPLITUDE to a NAMED V_ij requires the")
print("  observed mass-ordering as a data anchor.")
print()
print("  §8 resolvent-index formalization is CONFIRMED closed for what it")
print("  closes; the remaining [GEN-PAIR] residue is genuinely non-bounded")
print("  and likely irreducible without breaking δ-physical (deep frontier).")
print()
print("=" * 78)
print("W56 — HONEST NEGATIVE (G4 universal FAIL; G2 borderline)")
print("=" * 78)
