#!/usr/bin/env python3
"""
Down-sector closure attempt: derive vertex-level α_12 = α_1·(k*-2)/k* formally.

CONTEXT
=======
After R-14 partial closure for Rows P15 + P34 (δ_CP) and Row P37 (Koide
deviation ratio f(n) closure), the remaining open candidate was:

  y_b = (3k*-2)/k* · y_τ at -0.81% deviation from PDG.

User asked to ship this as down-sector closure. Earlier estimate was
"~1 session of structural work" to derive vertex-level α_12 = α_1·(k*-2)/k*
analogous to cycle-level α_12 = α_1·(g-2)/g (Row P37, theorem-grade).

THIS PROBE
==========
Audits whether the (3k*-2)/k* prefactor on y_b can be DERIVED structurally
from existing framework apparatus, or whether it requires new structural
content.

OBSTACLE IDENTIFIED
===================
The framework's y_τ derivation (theorem_ytau_corollary.md §3-§7) factorizes
the Yukawa vertex as:

  y_τ = (cycle amplitude α_1_full) × (fermion edge 1/k*)² × (Higgs edge 1) × (Cl(2) channel 1)
      = α_1_full / k*²

The factorization is SECTOR-BLIND: cycle amplitude, fermion edges, Higgs
edge, Cl(2) channel — none depend on which species (lepton vs quark) is
being computed.

If we APPLY this formula to y_b, we get y_b_framework = α_1_full/k*² = y_τ.
Observed: y_b ≈ 2.35·y_τ. Discrepancy.

For the (3k*-2)/k* prefactor to enter, sector-DEPENDENT structure must
appear in one of the four factors:
  - Cycle amplitude α_1_full: graph-theoretic, sector-blind by construction.
  - Fermion edge factor 1/k*: per-edge probability, uniform under MDL +
    site stabilizer transitivity. Sector-blind.
  - Higgs edge factor 1: forced by complement (only one unoccupied edge).
    Sector-blind.
  - Cl(2) channel factor 1: per-process waterline, sector-blind.

NONE of these naturally give a (3k*-2)/k* factor in the down-sector vs lepton-
sector ratio. The numerical match remains, but the structural derivation
isn't in framework apparatus.

WHAT FRAMEWORK CONVENTIONS REQUIRE
==================================
Per `parameter_linter.md` Type 6 algebraicity gate (added 2026-04-29,
strict-minimum-smuggle reformulation 2026-05-05):
  "Selection step is waterline-consistent. Every selection step in the
  derivation must be one of canonical_encoding or channel_select —
  bare 'MDL bit-cost minimum' is forbidden."

For y_b = (3k*-2)/k* · y_τ to close at Type 6:
  - Selection step needs to identify (3k*-2)/k* via either:
    (a) canonical_encoding(S) where all elements of S are encoding-equivalent
    (b) channel_select(S, c) where structural channel c picks (3k*-2)/k*

The framework's Row P37 derivation (cycle-level Koide ratio) achieves this
via many-body expansion + a separate private derivation by the author (pair-correlation length identity).
For the VERTEX-level analog, the framework lacks an analogous structural
result — there's no derivation of "α_12_vertex / α_1 = (k*-2)/k* on srs"
in the existing apparatus.

VERDICT
=======
The (3k*-2)/k* prefactor on y_b is a NUMERICAL MATCH at <1% but
STRUCTURAL DERIVATION is NOT achievable in one session within framework's
current apparatus. Closing this requires:

1. Deriving vertex-level α_12 structurally (analog of a separate private derivation by the author for
   pair correlation at vertex level instead of cycle level).
2. Showing the y_τ factorization extends with a sector-DEPENDENT prefactor
   that gives (3k*-2)/k* for n=1 (down) vs 1 for n=3 (charged lepton).

Both steps are research-level, requiring extension of theorem_ytau_corollary
to handle sector dependence. This was underestimated in the prior session
estimate ("~1 session").

PROBE OUTPUT
============
Honest negative — down-sector closure NOT achievable in this session.
Numerical match documented; structural derivation flagged as research-level.
"""

from __future__ import annotations

import math
from fractions import Fraction


# ============================================================================
# 1. Audit existing y_τ factorization for sector-dependence
# ============================================================================
print("=" * 78)
print("y_τ derivation factorization (per theorem_ytau_corollary §3-§7)")
print("=" * 78)
print()
print(f"  Factor 1: α_1_full = (5/3)·(2/3)^8 [cycle amplitude, theorem-grade]")
print(f"            Source: alpha_1_full.py, Type-4 upstream.")
print(f"            Sector-dependence: NONE (graph-theoretic α_1 + tan²(arg h_P)).")
print()
print(f"  Factor 2: (1/k*)² [fermion edge factors, premise-derived]")
print(f"            Source: y_τ §5 L3, MDL uniform on k* indistinguishable edges.")
print(f"            Sector-dependence: NONE (site stabilizer transitivity).")
print()
print(f"  Factor 3: 1 [Higgs edge factor, complement-forced]")
print(f"            Source: y_τ §6.")
print(f"            Sector-dependence: NONE (single unoccupied edge complement).")
print()
print(f"  Factor 4: 1 [Cl(2) channel factor, single-process waterline]")
print(f"            Source: y_τ §7 L11-L12, A2-T per-process reading.")
print(f"            Sector-dependence: NONE.")
print()
print(f"  Product: y_τ = α_1_full / k*² = sector-INDEPENDENT under existing derivation.")
print()


# ============================================================================
# 2. Apply same formula to y_b → discrepancy
# ============================================================================
print("=" * 78)
print("Applying y_τ formula to y_b: structural prediction")
print("=" * 78)
print()

K_STAR = 3
G_GIRTH = 10
ALPHA_1_FULL = (5/3) * (2/3)**8

y_tau_pred = ALPHA_1_FULL / K_STAR**2
y_tau_obs = 1.77686 / 246.22  # m_τ/v
y_b_obs = 4.18 / 246.22  # m_b/v

print(f"  y_τ_framework_pred = α_1_full/k*² = {y_tau_pred:.6e}")
print(f"  y_τ_obs = m_τ/v = {y_tau_obs:.6e}")
print(f"  Match: +{(y_tau_pred - y_tau_obs)/y_tau_obs * 100:+.2f}% ✓")
print()

print(f"  Applying same formula to y_b (sector-blind):")
print(f"  y_b_framework_pred = α_1_full/k*² = {y_tau_pred:.6e}")
print(f"  y_b_obs = m_b/v = {y_b_obs:.6e}")
print(f"  Match: {(y_tau_pred - y_b_obs)/y_b_obs * 100:+.2f}% ✗")
print()
print(f"  → Sector-blind framework formula GIVES SAME y_τ AND y_b.")
print(f"  → Observed y_b/y_τ ≈ {y_b_obs/y_tau_obs:.4f}.")
print(f"  → Sector-DEPENDENT structure is REQUIRED for the framework to")
print(f"    distinguish y_b from y_τ. This structure is NOT in the existing")
print(f"    y_τ derivation.")
print()


# ============================================================================
# 3. Test the (3k*-2)/k* candidate
# ============================================================================
print("=" * 78)
print("Candidate: y_b = (3k*-2)/k* · y_τ")
print("=" * 78)
print()

prefactor = Fraction(3*K_STAR - 2, K_STAR)  # = 7/3
y_b_candidate = float(prefactor) * y_tau_pred

print(f"  Candidate prefactor: (3k*-2)/k* = {prefactor} = {float(prefactor):.6f}")
print(f"  y_b_candidate = (7/3)·y_τ = {y_b_candidate:.6e}")
print(f"  y_b_obs = {y_b_obs:.6e}")
print(f"  Match: {(y_b_candidate - y_b_obs)/y_b_obs * 100:+.2f}% ✓")
print()


# ============================================================================
# 4. Type 6 (algebraicity gate) audit
# ============================================================================
print("=" * 78)
print("Type 6 (algebraicity gate) audit — does the candidate pass?")
print("=" * 78)
print()
print(f"  Per `parameter_linter.md` Type 6:")
print()
print(f"  (6a) L-expression: (3k*-2)/k* ∈ ℚ ⊂ K = ℚ(√2,√3,√5). ✓")
print(f"  (6b) K-membership: 7/3 ∈ K ✓")
print(f"  (6c) Selection step: REQUIRES canonical_encoding or channel_select")
print(f"        with structural argument.")
print()
print(f"        Current selection of (3k*-2)/k*:")
print(f"          Argument: 'analogous to (3g-2)/g at cycle level (Row P37).'")
print(f"          Channel: 'vertex-level many-body expansion'.")
print(f"          Structural derivation of α_12_vertex/α_1 = (k*-2)/k*: MISSING.")
print()
print(f"        This is closer to 'MDL bit-cost minimum' framing than to")
print(f"        canonical_encoding (would need encoding-equivalence proof) or")
print(f"        channel_select (would need structural channel constraint).")
print()
print(f"        Per the strict-minimum-smuggle reformulation 2026-05-05 EOD+1:")
print(f"          bare 'MDL bit-cost minimum' BLOCKS Type 6 closure.")
print()
print(f"  VERDICT: Type 6 closure BLOCKED for y_b candidate at present.")
print()


# ============================================================================
# 5. What WOULD close this
# ============================================================================
print("=" * 78)
print("What WOULD close the down-sector y_b derivation")
print("=" * 78)
print()
print(f"""  Step 1: Derive α_12_vertex / α_1 = (k*-2)/k* structurally.

    Cycle-level analog (a separate private derivation by the author = Row P37 derivation):
      α_12_cycle / α_1 = (g-2)/g comes from "pair-correlation length =
      g-2 internal edges of girth-g cycle" identity on srs lattice.

    Vertex-level analog needed:
      Show that the pair correlation between two occupied edges at one
      trivalent vertex passes through k*-2 = 1 unoccupied edge at k*=3,
      giving α_12_vertex = α_1 · (k*-2)/k*.

    This is bounded structural work (1-2 sessions of explicit substrate
    walk computation), but not done in this session.

  Step 2: Show sector-DEPENDENT prefactor in Yukawa vertex factorization.

    Current y_τ factorization (theorem_ytau_corollary §3): sector-blind.
    Down-sector extension needs to introduce a sector-DEPENDENT factor
    that gives (3k*-2)/k* for n=1 (down) vs 1 for n=3 (charged lepton).

    Plausible mechanism: many-body expansion at the trivalent vertex
    distinguishes Hamming-weight n=1 (single-occupation) from n=3
    (full-occupation). For n=1: contribution from (3k*-2)/k* factor.
    For n=3: full-occupation gives different structure (the existing
    y_τ derivation).

    This requires extending theorem_ytau_corollary with sector dependence.
    Not a one-session task.

  TOTAL ESTIMATE: 2-3 sessions of bounded structural work + audit.
""")

# ============================================================================
# 6. Honest verdict
# ============================================================================
print("=" * 78)
print("HONEST VERDICT")
print("=" * 78)
print()
print(f"""  Down-sector y_b closure NOT ACHIEVABLE in this session.

  - NUMERICAL MATCH documented: y_b = (3k*-2)/k* · y_τ at -0.81% deviation
    from PDG, within Clause 8 tree-level tolerance.

  - STRUCTURAL DERIVATION blocked: Type 6 (algebraicity gate) selection
    step requires canonical_encoding or channel_select with structural
    argument. Current "by analogy to Row P37" framing is closer to
    bare-MDL-minimum and BLOCKS Type 6 closure per linter conventions.

  - PRIOR ESTIMATE was optimistic: I said "~1 session" earlier in this
    session arc; the actual closure work is 2-3 sessions due to:
    (a) Vertex-level α_12 identity structural derivation
    (b) Extending theorem_ytau_corollary with sector dependence
    (c) Audit + linter verification

  R-14 SESSION ARC RESULTS (final, post P38+P39 attempt):

  CLOSED:
  - Row P37 f(n) Open Question 1 (commit 3f2efae)
  - Rows P15 + P34 (δ_CP) PARTIAL CLOSURE via V_{{-1}}-T_{{B-L}} identity
    (commit 13595d2)

  STILL OPEN:
  - Row P39 down (m_b): numerical match candidate, structural derivation
    pending (2-3 sessions).
  - Row P39 up (m_top, m_c, m_u): no clean candidate, research-level.
  - Row P38 (m_top): inherits up-sector status.

  R-14 has been substantially de-blocked but is not closed. The δ_CP
  partial closure is real progress; quark mass closures require
  multi-session research-level work.
""")

print("=" * 78)
print("END")
print("=" * 78)
