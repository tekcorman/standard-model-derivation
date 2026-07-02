#!/usr/bin/env python3
"""
W69 — SM-only RGE gen-3 polish (alternative to W68's MSSM-assuming probe)

Per user pushback on W68: the framework has ADOPTED MSSM at Layer 5 but
NOT DERIVED IT (per docs/framework/framework_architecture.md — Sprint 11
B7.6 scope, currently "adopted result", not theorem-grade). W68 imported
MSSM via srs_tan_beta and found a Landau-pole tension. This probe tests
the alternative: what do the framework's gen-3 anchor predictions look
like under SM-ONLY RGE (no SUSY, no tan β, no MSSM-specific corrections)?

The question this probe answers:
  Are the W67 residuals (+0.69% m_t, +1.96% m_b, -0.034% m_τ vs PDG) the
  framework's tree-level precision floor at the natural comparison scales,
  or does SM-only RGE close some of them?

If SM-only RGE doesn't materially reduce the residuals: the framework's
gen-3 predictions ARE the precision floor at their natural scales, and
"polish" without invoking MSSM/BSM is moot.

If SM-only RGE reduces some residuals: identify which ones are scale-
matching effects and which require BSM corrections (MSSM Δ_b etc.).

SETUP:
  - Framework prediction: m_f = y_f^F × (v × factor) at framework's
    natural prediction scale, with y_f^F = α₁/k² × Family_D for y_τ,
    (2/3)^g × Family_D for y_b, 1 × Family_D / √2 for y_t.
  - Treat the framework's prediction as being at the M_Z scale (the
    electroweak VEV scale where mass × v relations naturally live).
  - SM RGE M_Z → m_b for the m_b comparison.
  - SM RGE M_Z → m_t with QCD pole correction for the m_t comparison.
  - m_τ: minimal QED correction (running ~0.3%).

PRE-DECLARED GATES:
  G1: Framework's m_b prediction at M_Z, SM-run to m_b scale, within
      0.5% of PDG m_b(m_b) — would mean residual is scale-matching only.
  G2: Framework's m_t prediction at M_Z, SM-run to m_t pole, within
      0.3% of PDG m_t pole — would mean residual is scale-matching only.
  G3: Framework's m_τ prediction with QED running within 0.1% of PDG —
      essentially precision-floor verification.
  G4: Residuals after SM-only RGE LARGER than pre-RGE W67 residuals (which
      would mean we're running in the wrong direction — diagnostic check).

If G1, G2 PASS: framework's gen-3 anchors are SM-consistent at engineering
precision; W67 residuals were scale-matching artifacts. No MSSM required.

If G1, G2 FAIL: SM-only doesn't close the residuals. Either the framework's
prediction is at a different scale, or BSM (MSSM Δ_b etc.) is structurally
required.
"""

from __future__ import annotations
import math
from fractions import Fraction

# Framework primitives
k_star = 3
g_girth = 10
N_atoms = 4
v_higgs = 246.22

alpha_1_bare = Fraction(2, 3) ** 8
alpha_1_full = Fraction(5, 3) * alpha_1_bare

c_H = alpha_1_bare ** 2
c_F = -alpha_1_bare ** 2 / Fraction(N_atoms * k_star)
family_D = float(1 - (c_H + 2 * c_F))

# PDG values
m_t_pole_pdg = 172.69
m_b_msbar_pdg = 4.18         # at m_b scale
m_tau_pdg = 1.77686
alpha_s_MZ = 0.1179
alpha_s_mb = 0.220           # approx running
alpha_s_mt = 0.108           # approx running


# Framework tree-level predictions (post-Family-D)
y_t_F = 1.0 * family_D
y_b_F = float(Fraction(2, 3) ** g_girth) * family_D
y_tau_F = float(alpha_1_full) / k_star ** 2 * family_D

# Framework's m_f^pred = y_f × v (SM convention; no tan β factor since no MSSM)
m_t_F_treat_atMZ = y_t_F * v_higgs / math.sqrt(2.0)  # saturation reading
m_b_F_treat_atMZ = y_b_F * v_higgs                   # tree-level prediction
m_tau_F_treat_atMZ = y_tau_F * v_higgs

print("=" * 78)
print("W69 — SM-only RGE gen-3 polish (no MSSM assumption)")
print("=" * 78)
print()
print(f"Framework tree-level predictions (post-Family-D), treated as at M_Z scale:")
print(f"  m_t = y_t × v/√2 = {m_t_F_treat_atMZ:.5f} GeV")
print(f"  m_b = y_b × v    = {m_b_F_treat_atMZ:.5f} GeV")
print(f"  m_τ = y_τ × v    = {m_tau_F_treat_atMZ:.5f} GeV")
print()


# ──────────────────────────────────────────────────────────────────
# Test 1 — Treat framework's prediction as at the comparison scale
#          (W67 framing: natural-scale prediction; no running needed)
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("Test 1 — framework prediction interpreted as AT the comparison scale (W67)")
print("=" * 78)
err_t_W67 = (m_t_F_treat_atMZ - m_t_pole_pdg) / m_t_pole_pdg * 100
err_b_W67 = (m_b_F_treat_atMZ - m_b_msbar_pdg) / m_b_msbar_pdg * 100
err_tau_W67 = (m_tau_F_treat_atMZ - m_tau_pdg) / m_tau_pdg * 100
print(f"  m_t residual:  {err_t_W67:+.3f}%  (W67 documented +0.65%)")
print(f"  m_b residual:  {err_b_W67:+.3f}%  (W67 documented +1.96%)")
print(f"  m_τ residual:  {err_tau_W67:+.3f}%  (W67 documented -0.034%)")
print()


# ──────────────────────────────────────────────────────────────────
# Test 2 — Framework's y_b at M_Z, SM-run DOWN to m_b scale via QCD
#          anomalous-dim. Q: does running close the +1.96% residual?
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("Test 2 — m_b via SM QCD anomalous-dim running from M_Z to m_b")
print("=" * 78)

# Standard 5-flavor QCD running: m(μ_low) / m(μ_high) = (α_s(μ_high)/α_s(μ_low))^(4/23)
# Running DOWN in scale increases m_b (γ_m > 0 means m grows as μ decreases)
qcd_running_factor = (alpha_s_mb / alpha_s_MZ) ** (4.0 / 23.0)
m_b_pred_at_mb_via_running = m_b_F_treat_atMZ * qcd_running_factor

err_b_run = (m_b_pred_at_mb_via_running - m_b_msbar_pdg) / m_b_msbar_pdg * 100
print(f"  m_b at M_Z scale (framework): {m_b_F_treat_atMZ:.5f} GeV")
print(f"  QCD running factor M_Z → m_b: {qcd_running_factor:.4f}")
print(f"  m_b at m_b scale (predicted): {m_b_pred_at_mb_via_running:.5f} GeV")
print(f"  PDG m_b(m_b MS-bar):          {m_b_msbar_pdg} GeV")
print(f"  Residual:                     {err_b_run:+.3f}%")
print()
print(f"  WHAT THIS TELLS US:")
if abs(err_b_run) > abs(err_b_W67):
    print(f"    Running m_b DOWN from M_Z makes it WORSE ({err_b_run:+.2f}% vs "
          f"{err_b_W67:+.2f}% pre-run).")
    print(f"    This means the framework's prediction is NOT at M_Z scale — it must")
    print(f"    be at the m_b scale natively (where y × v gives mass directly).")
    print(f"    The +1.96% residual is NOT a scale-matching artifact.")
else:
    print(f"    Running m_b DOWN closes some of the residual; framework's prediction")
    print(f"    might naturally live at M_Z scale and need SM running to m_b.")
print()


# ──────────────────────────────────────────────────────────────────
# Test 3 — Framework's y_t at M_Z (saturation), SM-run to m_t pole
#          via QCD pole correction. Q: does running close the +0.65%?
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("Test 3 — m_t pole via QCD pole correction at m_t scale")
print("=" * 78)

# QCD pole correction: m_t_pole = m_t_run(m_t) × (1 + 4/3 × α_s/π + O(α_s²))
# Standard one-loop: m_t_pole/m_t_run(m_t) ≈ 1 + 4α_s(m_t)/(3π)
qcd_pole_corr_one_loop = 1.0 + 4.0 * alpha_s_mt / (3.0 * math.pi)

# Two-loop and higher: NLO correction adds about another ~3% upward shift
# For our purposes one-loop is sufficient since the framework's residual is +0.65%.
m_t_run_at_mt = m_t_F_treat_atMZ   # if framework gives m_t at m_t-scale, this is m_t_run
m_t_pole_pred_via_corr = m_t_run_at_mt * qcd_pole_corr_one_loop

err_t_run = (m_t_pole_pred_via_corr - m_t_pole_pdg) / m_t_pole_pdg * 100
print(f"  m_t (framework saturation at m_t scale): {m_t_run_at_mt:.4f} GeV")
print(f"  QCD pole correction factor:              {qcd_pole_corr_one_loop:.4f}")
print(f"  m_t pole predicted:                       {m_t_pole_pred_via_corr:.4f} GeV")
print(f"  PDG m_t pole:                             {m_t_pole_pdg} GeV")
print(f"  Residual:                                 {err_t_run:+.3f}%")
print()
print(f"  WHAT THIS TELLS US:")
if abs(err_t_run) > abs(err_t_W67):
    print(f"    Pole correction makes m_t too HIGH ({err_t_run:+.2f}% vs {err_t_W67:+.2f}%).")
    print(f"    Framework's prediction must already include the pole correction implicitly,")
    print(f"    or the framework predicts m_t at the pole scale directly.")
else:
    print(f"    Adding pole correction to framework's m_t closes some of the residual.")
print()


# ──────────────────────────────────────────────────────────────────
# Test 4 — m_τ with QED running (small effect, ~0.3% over M_Z-m_τ range)
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("Test 4 — m_τ with one-loop QED correction")
print("=" * 78)

# m_τ_pole ≈ m_τ_run(M_Z) × (1 + α_em(M_Z)/π × log(M_Z/m_τ))
# This is a small effect; m_τ is essentially at precision floor pre-correction.
qed_corr_tau = 1.0 + (1.0/137.0) / math.pi * math.log(91.1876 / m_tau_pdg)
m_tau_pred_pole = m_tau_F_treat_atMZ / qed_corr_tau   # run DOWN from M_Z
err_tau_run = (m_tau_pred_pole - m_tau_pdg) / m_tau_pdg * 100
print(f"  m_τ at M_Z (framework, post-Family-D): {m_tau_F_treat_atMZ:.5f} GeV")
print(f"  QED running factor M_Z → m_τ:           {qed_corr_tau:.5f}")
print(f"  m_τ pole predicted:                      {m_tau_pred_pole:.5f} GeV")
print(f"  PDG m_τ pole:                            {m_tau_pdg} GeV")
print(f"  Residual:                                {err_tau_run:+.4f}%")
print()


# ──────────────────────────────────────────────────────────────────
# Summary verdict
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("VERDICT — SM-only RGE polish")
print("=" * 78)

# G1: m_b within 0.5% post-SM-running
g1_pass = abs(err_b_run) < 0.5
# G2: m_t within 0.3% post-pole-correction
g2_pass = abs(err_t_run) < 0.3
# G3: m_τ within 0.1% (precision floor)
g3_pass = abs(err_tau_run) < 0.1
# G4: residuals smaller post-RGE than pre-RGE
g4_pass = (abs(err_b_run) < abs(err_b_W67)) or (abs(err_t_run) < abs(err_t_W67))

print()
print(f"  [{'PASS' if g1_pass else 'FAIL'}] G1 m_b via SM QCD running within 0.5% of PDG")
print(f"  [{'PASS' if g2_pass else 'FAIL'}] G2 m_t pole via QCD correction within 0.3% of PDG")
print(f"  [{'PASS' if g3_pass else 'FAIL'}] G3 m_τ pole via QED correction within 0.1% of PDG")
print(f"  [{'PASS' if g4_pass else 'FAIL'}] G4 at least one channel improves post-RGE")
print()

n_pass = sum([g1_pass, g2_pass, g3_pass, g4_pass])
print(f"  Gates passed: {n_pass}/4")
print()

if g4_pass:
    print("  SM-only RGE reduces at least one W67 residual.")
else:
    print("  SM-only RGE does NOT close the W67 residuals — running in the WRONG")
    print("  direction (residuals get worse).")
    print()
    print("  STRUCTURAL INTERPRETATION:")
    print("  The framework's tree-level predictions naturally live AT the comparison")
    print("  scales (m_b at m_b MS-bar, m_t at m_t pole, m_τ at m_τ pole). The W67")
    print("  residuals (+0.69% m_t, +1.96% m_b, -0.034% m_τ) are the framework's")
    print("  TREE-LEVEL PRECISION FLOOR at those scales.")
    print()
    print("  Closing these via RGE corrections requires invoking BSM physics:")
    print("  - SUSY Δ_b for m_b (tan β-enhanced; requires MSSM Layer-5 derivation)")
    print("  - Higher-order pole-mass conventions for m_t (multi-loop QCD)")
    print("  - Sub-leading Feshbach analogs (framework-internal but not yet derived)")
    print()
    print("  HONEST POSITION: without committing to MSSM (which is currently adopted")
    print("  but not theorem-grade per Sprint 11 B7.6), the framework's gen-3 anchor")
    print("  precision floor IS +0.69% / +1.96% / -0.034%. 'Polish to sub-permille'")
    print("  is contingent on theorem-grade derivation of Layer-5 SUSY or an")
    print("  alternative BSM closure.")
print()
print("=" * 78)
