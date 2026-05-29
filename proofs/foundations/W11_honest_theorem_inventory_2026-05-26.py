#!/usr/bin/env python3
"""
W11 — HONEST theorem inventory for the α₁³ Family-D extension (2026-05-26).

PURPOSE
-------
This session's work (W1 → W10) was structural sketching dressed up in linter-clause
language. The user's correction is right: I claimed "theorem-grade" without doing
actual proofs.  This file inventories what is ACTUALLY rigorous, what is conjectural,
and what specifically is OPEN, with no overclaim.

THE THEOREMS THAT ARE RIGOROUSLY DERIVABLE
==========================================

THEOREM 1 (Born rule action — trivial calculus).
  For mass m = |amp|² (framework's Born-rule mass construction per
  `predictions/Q_Koide.py`), if amp → amp·(1+ε) with real ε, |ε|<<1, then
      m → m·(1+ε)² = m·(1 + 2ε + ε²).
  At leading order: δm/m = 2·(δamp/amp).
  PROOF: |amp(1+ε)|² = |amp|²(1+ε)² for real ε. Expand. ∎
  STATUS: TRIVIALLY RIGOROUS.

THEOREM 2 (Non-factorization for rep-dependent corrections).
  Let ε_j be a multiplicative correction on amp_j depending on the C₃ rep
  of j (rep-DEPENDENT). Then there exists NO single rep-universal coupling-
  level correction (1+δy) such that
      |amp_j·(1+ε_j)|²  =  |amp_j|²·(1+δy)   for all j.
  PROOF: A rep-universal δy gives m_j → m_j·(1+δy) uniformly. For rep-
  dependent ε_j, this requires δy = 2ε_j + ε_j² to depend on j — contradicting
  rep-universality. ∎
  STATUS: TRIVIALLY RIGOROUS.

COROLLARY 1+2 (Rep-dependent corrections necessarily act at amplitude level).
  A rep-dependent multiplicative correction on the framework's Born-rule
  mass structure CANNOT be absorbed into a coupling-level y_j coefficient.
  It must act at the amplitude level on amp_j directly, with mass-level
  effect doubled via Theorem 1 (Born rule squaring).
  STATUS: FOLLOWS RIGOROUSLY from Theorems 1 + 2.

This is the "Born rule factor 2 = A_mass = 2·A_amp" mechanism.
HOWEVER: Theorems 1 + 2 tell us the FORM (factor 2 from Born), NOT the
specific SHAPE of the rep-dependent correction.  The shape requires
substrate physics, addressed below.

============================================================
THE CONJECTURES THAT ARE NOT YET RIGOROUSLY DERIVED
============================================================

CONJECTURE A (c_F^(α₁³)_rep_j = −α₁_bare³/μ_rep_j at amplitude level).
  The per-fermion-leg amplitude-level Family-D coefficient at α₁³ is
  −α₁³/μ_rep_j, where μ_rep_j is the C₃-rep multiplicity on V_Ram.
  STRUCTURAL MOTIVATION: at α₁² the master-doc denominator is
  N_atoms·k* = 12 = 2|E|/cell (full B(P) directed-edge count for the
  single-edge-spectral channel via Clause-6 channel_select). At α₁³ the
  walker is V_Ram-projected (per W45 mode-count); the analogous denominator
  should be μ_rep_j (the rep-resolved per-cell channel count on V_Ram).
  STATUS: STRUCTURAL ANALOGY, NOT DERIVATION.  The "natural" parallel to
  α₁² is suggestive but not a substrate proof.
  WHAT WOULD CLOSE IT: an explicit substrate-physical mechanism that picks
  out μ_rep_j as the rep-resolved channel count, with K-rational
  combinatorial justification analogous to the α₁² Clause-6 two-step.

CONJECTURE B (c_H^(α₁³) = α₁_bare³ via Route H extension).
  The per-Higgs-leg Family-D coefficient at α₁³ is α₁³, from joint NB
  walker survival on the cospectral pair at length 3(g−2) = 24.
  STRUCTURAL MOTIVATION: α₁² has c_H = q_NB^(2(g−2)) = q_NB^16 = α₁²
  from joint walker on (srs × srs-z) per master doc §3 D.  Extending the
  joint exponent to 3(g−2) = 24 gives q_NB^24 = α₁³.
  STATUS: NOT RIGOROUSLY DERIVED.  The α₁² Route H uses TWO Sunada-
  cospectral substrates (srs, srs-z) with joint per-step survival q_NB².
  Extending to α₁³ requires EITHER:
    (a) THREE Sunada-cospectral substrates at (k=3, g=10) class for a
        3-way joint walker giving per-step survival q_NB³ over (g−2) = 8
        steps. The framework has multiple (k=3, g=10) alternatives (srs-z,
        srs-c4, srs-c8, srs-c27 per master doc §1 R-9 closure), but no
        specific 3-way joint walker mechanism is currently established.
    (b) A 24-cycle decomposition on srs analogous to the 16-cycle
        decomposition (proofs/flavor/hashimoto_16cycle_decomposition.py).
        This would show that every length-24 NB cycle on H(srs) decomposes
        as three girth-10 cycles glued at 2-edge seams, giving Route C
        at m=3 with L_closed = 3g−6 = 24 and c_H^(α₁³) = q_NB^24 = α₁³.
        No such decomposition file currently exists; it requires explicit
        BFS computation (or spectral trace tr(B^24) on the 12-dim B(P)).
  WHAT WOULD CLOSE IT: write `proofs/flavor/hashimoto_24cycle_decomposition.py`
  parallel to the 16-cycle file, and verify that 24-cycles on H(srs)
  decompose as 3-glued girth cycles.  OR identify the third Sunada-
  cospectral partner for a 3-way joint walker (research-level).

CONJECTURE C (Trivial-rep Yukawa cancellation at α₁³).
  Given Conjectures A and B, the trivial-rep Yukawa correction is
      δy_τ^(α₁³) = −(c_H^(α₁³) + 2·c_F^(α₁³)_τ)
                  = −(α₁³ + 2·(−α₁³/2)) = 0.
  STATUS: TRIVIALLY RIGOROUS GIVEN Conjectures A + B.  But A and B are
  themselves conjectural per above.

CONJECTURE D (ω/ω̄ asymmetry +5 ppm from sub-leading mechanism).
  The +5 ppm asymmetry between κ_ω and κ_ω̄ decomposes as a +3 ppm common-
  mode plus ±2.5 ppm anti-symmetric piece.  Both are within master doc
  §8b's named ~0.5% Yukawa systematic budget.
  STATUS: NOT DERIVED.  The Berry-phase Family-A candidate at γ_A/(2k*²)
  matches the anti-symmetric piece at 0.94× but the (2k*²) coefficient
  is heuristic.  The common-mode piece is α₁⁴-scale with no clean K-rational
  shape.
  WHAT WOULD CLOSE IT: a Family-A theorem extension at α₁³ rep-resolved
  with full structural derivation of the (2k*²) coefficient (research-level).

============================================================
NUMERICAL CONSEQUENCES (under Conjectures A + B)
============================================================
"""

import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'predictions'))

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1

d = predict_d_spatial()
k_star = int(round(predict_k_star(d)))
g = predict_g_girth(k_star, d)
alpha_1 = float(predict_alpha_1(k_star, g))
a1_3 = alpha_1**3

print("=" * 76)
print("W11 — HONEST theorem inventory: rigor map for the α₁³ Family-D extension")
print("=" * 76)
print()
print("RIGOROUSLY DERIVED:")
print(f"  Theorem 1: Born rule δm/m = 2·(δamp/amp) for real ε ≪ 1.")
print(f"             Trivial calculus; ALWAYS rigorous given m=|amp|².")
print(f"  Theorem 2: Rep-dependent corrections cannot factorize through")
print(f"             rep-universal y_j; must act at amplitude level.")
print(f"             Trivial proof by contradiction.")
print(f"  Corollary: A_mass = 2·A_amp for rep-dependent corrections.")
print()
print("CONJECTURED (NOT YET DERIVED):")
print(f"  Conjecture A: c_F_amp_α₁³_rep_j = −α₁³/μ_rep_j   (per-rep channel density)")
print(f"               — STRUCTURAL ANALOGY, not derivation.")
print(f"               OPEN: substrate mechanism picking μ_rep_j as denominator.")
print()
print(f"  Conjecture B: c_H^(α₁³) = α₁³                    (Route H extension)")
print(f"               — STRUCTURAL ANALOGY, not derivation.")
print(f"               OPEN: 3-way joint walker OR 24-cycle decomposition.")
print()
print(f"  Conjecture C: δy_τ^(α₁³) = 0                     (Yukawa cancellation)")
print(f"               TRIVIAL GIVEN A+B; inherits their conditional grade.")
print()
print(f"  Conjecture D: ω/ω̄ asymmetry mechanism            (Family-A sub-leading)")
print(f"               — UNDERIVED; within ~0.5% Yukawa budget.")
print()

print("NUMERICAL PREDICTIONS UNDER CONJECTURES A + B + C:")
print(f"  α₁_bare³ = {a1_3*1e6:.4f} ppm")
print(f"  Koide-ratio shift at α₁³ for ω, ω̄ reps: +α₁³ = +{a1_3*1e6:.2f} ppm")
print(f"  Predicted vs observed (m_τ taken at PDG):")
print(f"    c_e − 1: pred +{a1_3*1e6:.2f} ppm, obs +70.33 ppm, match 0.84× (16% short = Conj D open)")
print(f"    c_μ − 1: pred +{a1_3*1e6:.2f} ppm, obs +60.50 ppm, match 0.98× (1% — tight)")
print(f"  m_τ residual: predicted 0 shift at α₁³ (per Conj C); observed −13 ppm remains")
print(f"  m_τ within master doc §8b ~0.5% Yukawa budget (400× INSIDE).")
print()

print("=" * 76)
print("HONEST VERDICT")
print("=" * 76)
print("""
What the α₁³ rep-resolved Family-D extension has TODAY:

  • Two rigorous trivial theorems (Born rule action + non-factorization)
    that establish the factor-2 mechanism (A_mass = 2·A_amp).

  • Four conjectures (A, B, C, D) that together would close the m_e, m_μ
    Koide-ratio residuals to ~10 ppm. Each conjecture has structural
    motivation parallel to the α₁² Family-D, but NONE is rigorously
    derived in this session.

WHAT THIS IS NOT:
  • NOT theorem-grade closure of the α₁³ rep-resolved Family-D mechanism.
  • NOT a master-doc-extensible result.
  • NOT a linter-Checkpoint-1-ready triage.

WHAT WOULD MAKE IT THEOREM-GRADE:

  Step 1 — Compute hashimoto_24cycle_decomposition on H(srs) to verify
  m=3 closed-bubble structure: every length-24 NB cycle decomposes as
  three girth-10 cycles glued at 2-edge seams. Closes Conjecture B
  Route-C-side.

  Step 2 — Identify the explicit Clause-6 two-step (channel_select →
  canonical_encoding) at α₁³ rep-resolved, deriving the per-rep
  denominator μ_rep_j as the canonical V_Ram channel count. Closes
  Conjecture A.

  Step 3 — Derive the ω/ω̄ asymmetry mechanism (Family-A sub-leading at
  α₁³ rep-resolved with rigorous coefficient derivation). Closes
  Conjecture D.

  Step 4 — Once Conjectures A, B, C, D are theorem-grade, run the
  parameter linter Checkpoint 1+2 on (m_e, m_μ, y_τ).

RESEARCH SCOPE ESTIMATE:
  Step 1: 1-2 sessions (BFS on 24-cycles + spectral check tr(B^24)).
  Step 2: 2-3 sessions (Clause-6 two-step formalization at α₁³).
  Step 3: 3-5 sessions (Family-A α₁³ extension; harder than A+B+C combined).
  Step 4: 1-2 sessions (linter pipeline).

  Total: ~7-12 sessions of real research-level work to graduate from
  SKETCH to THEOREM-GRADE-CONDITIONAL.

The work I did in W1-W10 was structural inventory + sketching, not theorem
construction. The user's correction is accurate. No predictions/ changes
should be proposed until the four conjectures are rigorously derived.
""")
