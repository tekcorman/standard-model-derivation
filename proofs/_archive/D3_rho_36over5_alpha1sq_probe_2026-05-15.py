#!/usr/bin/env python3
"""
proofs/_archive/D3_rho_36over5_alpha1sq_probe_2026-05-15.py

D3 — Structural derivation attempt for (36/5)·α₁² as the ρ-parameter
custodial-breaking magnitude.

Empirical:    ρ_obs − 1 = +1.048%
Suggestive:   (Koide_quark_ratio − 1) · N_atoms · α₁² = (9/5)·4·α₁² = (36/5)·α₁²
              = 7.2 · α₁²  ≈ +1.096%   (off by +4.7%)

The (9/5) comes from the Koide quark ratio (ε²_up − 2)/(ε²_down − 2) = 14/5
(theorem-grade, Row P37), minus 1.  The N_atoms = 4 and α₁² = (2/3)^16 are
theorem-grade upstream.

THIS PROBE asks: is there a STRUCTURAL DERIVATION reaching (9/5)·N·α₁² for
the ρ-parameter custodial-breaking, or is the +4.7% match coincidence
absorbing two pieces undifferentiated?

Following an internal note discipline:
identify the framework's theorem-grade structural identity, check
empirical violation, decompose residual into structural pieces.
Magnitude-matching of graph-native quantities = numerology.

DECOMPOSITION CHECK (per the discipline):
The empirical δρ is dominated by Δρ ~ 3·G_F·(m_t² − m_b²)/(8√2 π²).
In framework variables: 3·y_t² /(32 π²) at low energy.

  Step 1: Is the dominant SM contribution structurally captured by (9/5)·4·α₁²?
  Step 2: If not, what structural pieces does Family-D vertex correction
          on the top loop SHOULD capture?
  Step 3: What's the residual after the structural piece, and is the (9/5)
          factor present or absent?

PRE-DECLARED VERDICT CRITERIA:

  PASS structurally if the (9/5)·N·α₁² form is reached by:
    - Direct application of Koide quark ratio (theorem) AND
    - Standard Family-D per-leg counting on top loop AND
    - No additional fitting

  FAIL (numerology) if:
    - The (9/5) cannot be motivated structurally from a top-loop graph
    - Or the N_atoms factor doesn't appear in a derived form
    - Or further structural factors (color 3, top-loop count 2) make it
      incompatible with (9/5)·4·α₁²
"""
from fractions import Fraction
import math

# --- Framework theorem-grade upstream ---
k_star  = 3
g       = 10
N_ATOMS = 4
alpha_1_bare = Fraction(2, 3) ** 8          # = (k*-1)/k* over (g-2) NB steps
alpha_1_sq = alpha_1_bare ** 2              # = (2/3)^16

koide_quark_ratio = Fraction(14, 5)         # (ε²_up - 2)/(ε²_down - 2), Row P37 theorem-grade
asymmetry_up_down = koide_quark_ratio - 1   # = 9/5

# Per Family D theorem-grade (master doc §3 (D))
c_H = alpha_1_sq                                  # +α₁²  per Higgs leg
c_F = -alpha_1_sq / (N_ATOMS * k_star)            # -α₁²/12 per fermion leg
N_color = 3                                       # SU(3)_C colour multiplicity

# --- Empirical target ---
M_Z_pred = 91.5134
m_W_pred = 80.2373
M_Z_obs  = 91.1876
m_W_obs  = 80.3692
sin2_theta_W_pred = 0.23126
cos2_theta_W_pred = 1 - sin2_theta_W_pred
rho_pred = 1.0   # by construction at tree level
rho_obs = (m_W_obs**2) / (M_Z_obs**2 * cos2_theta_W_pred)
delta_rho_emp = rho_obs - 1

print("=" * 76)
print("D3 — (36/5)·α₁² structural-derivation probe for ρ-parameter")
print("=" * 76)
print()
print("Theorem-grade upstream:")
print(f"  k*                          = {k_star}")
print(f"  g (girth)                   = {g}")
print(f"  N_atoms                     = {N_ATOMS}")
print(f"  N_color (SU(3)_C)           = {N_color}")
print(f"  α₁²                          = {float(alpha_1_sq):.6e}  = {float(alpha_1_sq)*100:.5f}%")
print(f"  koide_quark_ratio (Row P37) = 14/5")
print(f"  asymmetry_up_down (14/5 - 1) = 9/5  =  {float(asymmetry_up_down):.4f}")
print(f"  c_H (Family D per-Higgs)    = +α₁²  =  {float(c_H)*100:+.5f}%")
print(f"  c_F (Family D per-fermion)  = -α₁²/12 = {float(c_F)*100:+.5f}%")
print()
print(f"Empirical target:")
print(f"  δρ = ρ_obs − 1              = {delta_rho_emp*100:+.5f}%")
print()
print(f"Suggestive numerology:")
suggestive = float(asymmetry_up_down) * N_ATOMS * float(alpha_1_sq)
print(f"  (9/5)·N_atoms·α₁²           = {suggestive*100:+.5f}%")
print(f"  Match relative error        = {(suggestive - delta_rho_emp)/delta_rho_emp*100:+.2f}%")
print()
print("=" * 76)
print("STRUCTURAL ATTEMPT — derive δρ from Family D + Koide quark ratio")
print("=" * 76)
print()

# Step 1: SM-side reference — Δρ = 3·G_F·(m_t² − m_b²)/(8√2 π²)
m_t_pdg = 172.69
m_b_pdg = 4.18
v_pdg = 246.22
G_F = 1.1663787e-5
delta_rho_SM_formula = 3 * G_F * (m_t_pdg**2 - m_b_pdg**2) / (8 * math.sqrt(2) * math.pi**2)
print(f"SM reference Δρ (using PDG m_t, m_b):")
print(f"  Δρ_SM = 3·G_F·(m_t² − m_b²)/(8√2 π²) = {delta_rho_SM_formula*100:+.5f}%")
print()

# Step 2: framework structural identity — Family D on top loop in Z self-energy
# A top quark loop in the Z propagator has TWO fermion legs (top in + top out),
# plus 2 external Z gauge legs.  Color: 3.
# Per-leg corrections: c_F per fermion leg, c_G per gauge leg
# Family D applied: δ ~ -(2·c_G + 2·c_F)
#
# But this is sign-uniform (negative) and cannot produce custodial-breaking.
# Per A1 / A2 analysis from this morning: custodial-breaking requires the
# UP-DOWN ASYMMETRY (top vs bottom) at the loop level.
#
# Hypothesis: the per-fermion-leg dark correction has SECTOR DEPENDENCE
# from the Koide quark ratio:
#   c_F^up   = c_F × ratio_up
#   c_F^down = c_F × ratio_down
# with (ratio_up - ratio_down) = asymmetry_up_down = 9/5

# Suppose: c_F^up - c_F^down = c_F × (asymmetry) = (-α₁²/12) × (9/5)
diff_c_F = c_F * asymmetry_up_down  # = -(9/60)·α₁² = -(3/20)·α₁²
print(f"Structural attempt 1 — Koide asymmetry on Family D:")
print(f"  Δc_F = c_F·(9/5)    = {float(diff_c_F)*100:+.5f}%")
print()

# Z self-energy from top-bottom loop asymmetry:
# δρ ~ N_color · 2 · |Δc_F|  (2 fermion legs in the loop, 3 colors)
delta_rho_attempt1 = N_color * 2 * abs(float(diff_c_F))
print(f"  δρ_attempt = N_c · 2 · |Δc_F| = {delta_rho_attempt1*100:+.5f}%")
print(f"  vs empirical δρ              = {delta_rho_emp*100:+.5f}%")
print(f"  Ratio attempt/empirical       = {delta_rho_attempt1/delta_rho_emp:.4f}")
print()
print("⇒ Attempt 1 gives N_c · 2 · (9/60)·α₁² = (9/10)·α₁² = 0.137% — too small by 7×.")
print()

# Step 3: alternative structure — Δρ comes from Family D on the (m_t² - m_b²)
# difference at the per-flavour level, scaled by N_atoms (atoms in primitive cell)
# rather than N_color.
delta_rho_attempt2 = float(asymmetry_up_down) * N_ATOMS * float(alpha_1_sq)
print(f"Structural attempt 2 — (9/5)·N_atoms·α₁² [the suggestive form]:")
print(f"  δρ_attempt = (9/5)·N_atoms·α₁² = {delta_rho_attempt2*100:+.5f}%")
print(f"  vs empirical δρ              = {delta_rho_emp*100:+.5f}%")
print(f"  Ratio attempt/empirical       = {delta_rho_attempt2/delta_rho_emp:.4f}")
print()
print(f"  But what derivation gives N_atoms·(9/5)·α₁²?  Required ingredients:")
print(f"    (a) (9/5) from Koide quark ratio asymmetry")
print(f"    (b) N_atoms = 4 (substrate atoms per primitive cell)")
print(f"    (c) α₁² (joint NB walker survival on srs × srs-z)")
print()
print(f"  Issue: N_atoms appears in c_F = -α₁²/(N_atoms·k*) as a DENOMINATOR.")
print(f"  Multiplying by N_atoms cancels its appearance in c_F.")
print(f"  Net: (9/5)·N_atoms·α₁² = (9/5)·(N·k*)·|c_F| = 12·(9/5)·|c_F| = (108/5)·|c_F|.")
print(f"  Effectively: (108/5)·(-α₁²/12) inverted in sign → +(108/(5·12))·α₁² = (9/5)·α₁²")
print()
print(f"  So we need: δρ = (108/5)·|c_F| factor.  No clean structural")
print(f"  derivation of 108/5 in terms of substrate primitives is apparent.")
print()

# Step 4: try Family D on combined sector + color counting
# δρ from top-loop pure-Family D, no Koide-asymmetry baked in
# 2 fermion legs (top loop) × 3 colors × m_t²/v² weight (= ½ at GUT) × c_F structure
# But this gives uniform sign, not custodial-breaking
delta_rho_attempt3 = N_color * 2 * abs(float(c_F)) * 0.5  # m_t²/v² ≈ 1/2 at GUT
print(f"Structural attempt 3 — Pure-Family D on top loop (sign-uniform):")
print(f"  δ = N_c · 2 · |c_F| · (m_t²/v²)_{{GUT}} = 3·2·(α₁²/12)·(1/2) = α₁²/4")
print(f"     = {float(alpha_1_sq)/4*100:+.5f}%   (vs empirical {delta_rho_emp*100:+.4f}%)")
print(f"  ⇒ Too small by ~28× and sign-uniform (no custodial breaking).")
print()

# --- Verdict ---
print("=" * 76)
print("VERDICT — D3 probe NEGATIVE-NUMEROLOGY")
print("=" * 76)
print()
print("(9/5)·N_atoms·α₁² = (36/5)·α₁² = 1.096% is empirically close to δρ_emp")
print("= 1.048%, but the +4.7% relative gap is itself the size of the α₁²")
print("correction we're trying to derive — i.e., the residual on the residual")
print("is the same order as the residual.")
print()
print("Structural attempts:")
print(f"  Koide asymmetry on Family D                 → 0.137% (7× too small)")
print(f"  Pure Family D top loop                       → 0.038% (28× too small)")
print(f"  (9/5)·N_atoms·α₁² as a free product          → 1.096% (matches, but ad hoc)")
print()
print("No derivation reaches (36/5)·α₁² from theorem-grade primitives.  The")
print("structural ingredients to put into the substrate top-loop Δρ are:")
print()
print(f"  - top mass m_t² (R-14-blocked dimensional input)")
print(f"  - bottom mass m_b² (R-14-blocked)")
print(f"  - color factor N_c = 3 (theorem-grade)")
print(f"  - Family D per-fermion c_F = -α₁²/12 (theorem-grade)")
print(f"  - vertex topology of Z 2-point with internal top-bottom loop")
print()
print("The (9/5)·N_atoms·α₁² form CANNOT be assembled from these without")
print("either fitting or borrowing the m_t²/v² ≈ 1/2 ratio (which uses PDG).")
print()
print("DISCIPLINE:, a")
print("magnitude match without derived structural identity is NUMEROLOGY.  Filed")
print("as a CANDIDATE TARGET for future R-14 closure (if substrate y_t derivation")
print("lands at (36/5)·α₁² scale, the form is recovered; if not, the form is")
print("permanently false-friend).")
print()
print("STATUS: D3 NEGATIVE.  (36/5)·α₁² ≈ 1.05% is a false friend at present;")
print("permitted to remain as a candidate target post-R-14, not before.")
print()
print("=" * 76)
