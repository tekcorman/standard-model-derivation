#!/usr/bin/env python3
"""
W16 — α₁²/54 candidate for m_e/m_μ Koide-ratio common-mode (2026-05-26).

The framework's A_s primordial-amplitude derivation uses the prefactor

    1/54 = c_S · q² · (1/2)_orient = (1/12) · (4/9) · (1/2)

per `docs/theorems/theorem_unified_oblique.md` §9.3.

Each factor is structurally derived:
  c_S = 1/(2|E|) = 1/12       Perron-residue singlet projection of B_NB
  q²  = ((k*-1)/k*)² = 4/9    two-step NB walker survival (girth completion)
  (1/2)_orient                directed→undirected cycle-count factor

THIS PROBE: test whether α₁²/54 matches the m_e/m_μ Koide-ratio
common-mode (the symmetric piece, NOT the ω/ω̄ asymmetry).
"""
from fractions import Fraction
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from predictions.k_star import predict_k_star
from predictions.g_girth import predict_g_girth
from predictions.alpha_1 import predict_alpha_1

k_star = predict_k_star(d=3)
g = predict_g_girth(k_star, 3)
alpha_1 = float(predict_alpha_1(k_star, g))
N_atoms = 4

# A_s framework factors
c_S_frac = Fraction(1, 2*6)            # 1/(2|E|) = 1/12 for srs cell |E|=6
q2_frac = Fraction((k_star-1)**2, k_star**2)  # 4/9
half_orient = Fraction(1, 2)
inv_54 = c_S_frac * q2_frac * half_orient
print("=" * 76)
print("W16 — α₁²/54 candidate for m_e/m_μ Koide-ratio common-mode")
print("=" * 76)
print()
print("Framework's 1/54 factor (theorem_unified_oblique.md §9.3):")
print(f"  c_S    = 1/(2|E|) = 1/(2·6) = {c_S_frac}")
print(f"  q²     = ((k*-1)/k*)² = {q2_frac}")
print(f"  (1/2)_orient = {half_orient}")
print(f"  Product = c_S · q² · (1/2) = {inv_54} = {float(inv_54):.6f}")
print(f"  Check: 1/54 = {1/54:.6f} → MATCH: {abs(float(inv_54) - 1/54) < 1e-15}")
print()

# α₁²/54 candidate
alpha2_over_54 = alpha_1**2 / 54
print(f"Candidate prefactor α₁²/54 = α₁² · c_S · q² · (1/2)_orient:")
print(f"  α₁²      = (2/3)¹⁶ = {alpha_1**2:.6e}")
print(f"  α₁²/54    = {alpha2_over_54:.6e}")
print(f"           = {alpha2_over_54*1e6:.2f} ppm")
print()

# Observed Koide-ratio residuals (taking m_τ at PDG = 1.77686)
obs_e_minus_tau  = 35.166e-6
obs_mu_minus_tau = 30.249e-6
common_mode = (obs_e_minus_tau + obs_mu_minus_tau) / 2
anti_sym = (obs_e_minus_tau - obs_mu_minus_tau) / 2

print(f"Observed (from W1 back-solve with m_τ at PDG):")
print(f"  κ_ω − κ_τ           = +{obs_e_minus_tau*1e6:.3f} ppm")
print(f"  κ_ω̄ − κ_τ          = +{obs_mu_minus_tau*1e6:.3f} ppm")
print(f"  Common-mode (avg)    = +{common_mode*1e6:.3f} ppm")
print(f"  Anti-symmetric (½(ω−ω̄)) = +{anti_sym*1e6:.3f} ppm")
print()

# Match
match_ratio = alpha2_over_54 / common_mode
print(f"Candidate α₁²/54 vs observed common-mode:")
print(f"  Predicted: {alpha2_over_54*1e6:.3f} ppm")
print(f"  Observed:  {common_mode*1e6:.3f} ppm")
print(f"  Match ratio: {match_ratio:.4f}× ({(match_ratio-1)*100:+.1f}%)")
print()

# Residual after this candidate
residual_after = common_mode - alpha2_over_54
print(f"Residual after α₁²/54 candidate: +{residual_after*1e6:.3f} ppm")
print(f"  Scale check: α₁⁴ = {alpha_1**4*1e6:.3f} ppm  → residual is {residual_after/alpha_1**4:.2f}·α₁⁴")
print()

# What this would mean
print("=" * 76)
print("STRUCTURAL INTERPRETATION")
print("=" * 76)
print(f"""
α₁²/54 = α₁² · c_S · q² · (1/2)_orient matches the m_e/m_μ Koide-ratio
common-mode at {match_ratio:.3f}× ({(match_ratio-1)*100:+.1f}%). The structural
mapping would be:

  m_j Koide-ratio correction (common-mode piece) =
      - α₁² · c_S · q² · (1/2)_orient · (rep-universal factor)
      = -α₁²/54   ≈ -{alpha2_over_54*1e6:.1f} ppm

This is the SAME mechanism that gives A_s = (1/54)·a·(M_GUT/M_Pl)²
per theorem_unified_oblique.md §9.

WHAT THIS DOES NOT EXPLAIN (gaps):

(1) The {residual_after*1e6:.1f} ppm residual on the common-mode (~{residual_after/alpha_1**4:.1f}·α₁⁴)
    — possibly α₁⁴ sub-leading, but no clean K-rational shape yet.

(2) The ω/ω̄ asymmetry of ±{anti_sym*1e6:.2f} ppm — separate mechanism needed,
    α₁²/54 is rep-universal.

(3) Whether the same Yukawa-vertex mechanism (c_S projection on B_NB at Γ)
    applies to mass-RATIO observables m_j/m_τ. A_s is a SCALAR amplitude
    observable; Koide ratios are DIMENSIONLESS mass ratios. The structural
    parallelism needs explicit verification — currently CONJECTURAL.

(4) Why the mechanism affects Koide RATIO common-mode but NOT m_τ
    absolute scale (which sits at −0.17σ within ~0.5% Yukawa budget).

LINTER 9-CLAUSE STATUS for this candidate:

  Clause 1 (axiom):           N/A
  Clause 2 (algebra):         PASS (K-rational arithmetic)
  Clause 3 (known theorem):   CITES `theorem_unified_oblique.md` §9.3 — BUT the
                              theorem is for A_s SCALAR amplitude, not Koide
                              MASS ratios. Extension requires NEW derivation
                              showing the same mechanism applies to Yukawa
                              vertex mass-ratio observables.
  Clause 4 (predictions/):    α₁_bare via alpha_1.py [closed]; c_S structurally
                              via |E|=6 in srs; q² via k*=3
  Clause 5 (master-theorem):  unified oblique master theorem PARTIAL — A_s case
                              closed, Koide-ratio extension OPEN
  Clause 6 (K-meta-theorem):  PASS (α₁²/54 ∈ ℚ ⊂ K)
  Clause 7 (audit-v2):        NOT yet attempted
  Clause 8 (numerical):       PARTIAL — 87% match on common-mode; 13% residual
                              within master doc §8b ~0.5% Yukawa budget
  Clause 9 (π-audit):         PASS (no π in α₁_bare, c_S, q²)

VERDICT: CANDIDATE-GRADE at best. The numerical match (87%) is suggestive
but the structural justification (Yukawa-vertex application of c_S·q²·(1/2)_orient)
requires explicit derivation parallel to the A_s §9 closure.

WITHOUT that derivation, this is NUMEROLOGY per master doc §6 Step 6
("numerical match is INDICATIVE but not the test of correctness").

THE HONEST POSITION:

α₁²/54 = 28.2 ppm matching common-mode 32.5 ppm at 87% is INTRIGUING because:
  • 1/54 is already a derived framework number (for A_s)
  • The 13% miss is α₁⁴-scale, consistent with sub-leading corrections
  • The structural shape c_S·q²·(1/2)_orient applies to Born-rule scalar observables

But it's NOT theorem-grade until:
  • A specific derivation shows the SAME mechanism applies to Yukawa vertices
  • The 13% miss is structurally accounted (α₁⁴ + ω/ω̄ asymmetry)
  • Audit-v2 Phase-3 row for this Koide-ratio extension passes

Recommended next step (research-level): scope a derivation of the unified
oblique mechanism at the Yukawa vertex — does c_S·q²·(1/2)_orient extend
naturally from scalar A_s to mass-ratio Koide observables?

If yes (multi-session derivation), this gives a CANDIDATE structural
mechanism for the Koide-ratio common-mode at theorem-grade.

If no (which would be discovered by the derivation attempt), we have a
NUMEROLOGY caveat — close numerical match without underlying mechanism.
""")
