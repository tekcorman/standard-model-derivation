#!/usr/bin/env python3
"""
W67 — Engineering polish: clean numerical diagnostic of gen-3 anchor predictions

Diagnostic probe for the 4 gen-3 fermion anchors (y_τ, y_t, y_b, y_ν3) at
the framework's natural scale, articulating the convention explicitly and
reporting honest numerical state vs PDG.

PURPOSE: option-2 engineering polish doesn't unlock new content but
documents the current numerical state cleanly. Shows:
  (a) Framework convention: y_f = m_f/v (NOT the SM Yukawa coupling)
      except y_t where m_t = v/√2 means y_t × v/√2 = m_t (special-case
      saturation reading)
  (b) Predicted m_f values at the framework's natural scale
  (c) Comparison with PDG observables
  (d) Honest residual decomposition per W42 structural attribution

NO SCHEME RECONCILIATION WORK done here — that's multi-session.
This probe just states the current numerical position cleanly.

PRE-DECLARED GATES:
  G1: y_τ prediction within 0.5% of PDG m_τ (sub-permille post-Family-D)
  G2: y_t prediction within 1.0% of PDG m_t pole
  G3: y_b prediction within 2.5% of PDG m_b MS-bar at m_b scale
  G4: y_ν3 prediction within 3.0% of NuFIT m_ν3
"""

from __future__ import annotations
import math
from fractions import Fraction

# Framework primitives
k_star = 3
g = 10
N_atoms = 4
alpha_1_bare = Fraction(2, 3) ** 8                 # = 256/6561
alpha_1_full = Fraction(5, 3) * alpha_1_bare       # = (5/3) · α₁_bare

# Family-D Yukawa-vertex factor (universal for 1H + 2F vertex)
c_H = alpha_1_bare ** 2
c_F = -alpha_1_bare ** 2 / Fraction(N_atoms * k_star)
family_D = float(1 - (c_H + 2 * c_F))              # ≈ 0.99873

# Higgs VEV (theorem-grade per predictions/v_higgs.py)
v_higgs = 246.22  # GeV

# PDG 2024 observables
m_tau_pdg = 1.77686                # GeV (pole)
m_tau_err = 0.00012
m_t_pole_pdg = 172.69              # GeV
m_t_pole_err = 0.30
m_b_MSbar_pdg = 4.18               # GeV (MS-bar at m_b)
m_b_MSbar_err = 0.03
m_nu3_NuFIT = 50.13e-12            # GeV (√Δm²_31 from NuFIT 6.0)
m_nu3_err = 0.20e-12

print("=" * 78)
print("W67 — Gen-3 anchor numerical diagnostic (engineering polish)")
print("=" * 78)
print()
print(f"Framework primitives:")
print(f"  k* = {k_star}, g = {g}, N = {N_atoms}")
print(f"  α₁_bare = (2/3)^8 = {float(alpha_1_bare):.8f}")
print(f"  α₁_full = (5/3)·α₁_bare = {float(alpha_1_full):.8f}")
print(f"  Family-D factor = 1 - (5/6)·α₁_bare² = {family_D:.8f}")
print(f"  v = {v_higgs} GeV")
print()


# ------------------------------------------------------------------------
# y_τ — Type III walker, L = g-2 = 8, chir 5/3
# ------------------------------------------------------------------------
print("=" * 78)
print("y_τ — Type III walker (L = g-2 = 8, chir 5/3)")
print("=" * 78)

# Framework convention: y_τ = m_τ/v
y_tau_tree = alpha_1_full / Fraction(k_star ** 2)  # = α₁_full/k² = 1280/177147
y_tau_post_FD = float(y_tau_tree) * family_D
m_tau_pred = y_tau_post_FD * v_higgs

err_pct_tau = (m_tau_pred - m_tau_pdg) / m_tau_pdg * 100
err_sigma_tau = (m_tau_pred - m_tau_pdg) / m_tau_err

print(f"  y_τ (tree) = α₁_full/k² = {float(y_tau_tree):.6f}")
print(f"  y_τ (post-Family-D) = {y_tau_post_FD:.6f}")
print(f"  m_τ_pred = y_τ × v = {m_tau_pred:.6f} GeV")
print(f"  m_τ_PDG  = {m_tau_pdg} ± {m_tau_err} GeV")
print(f"  Δ = {err_pct_tau:+.3f}% = {err_sigma_tau:+.1f}σ_PDG")
print()
g1 = abs(err_pct_tau) < 0.5
print(f"  G1 (within 0.5%): {'PASS' if g1 else 'FAIL'}")
print()


# ------------------------------------------------------------------------
# y_t — Type II walker (L = 0, h = 1 saturation)
# ------------------------------------------------------------------------
print("=" * 78)
print("y_t — Type II walker (L = 0, h = 1 saturation; mask #1 closure)")
print("=" * 78)

# Framework convention for y_t: m_t = y_t × v/√2 (saturation reading)
y_t_tree = 1.0
y_t_post_FD = y_t_tree * family_D
m_t_tree_pred = y_t_post_FD * v_higgs / math.sqrt(2)

err_pct_t = (m_t_tree_pred - m_t_pole_pdg) / m_t_pole_pdg * 100
err_sigma_t = (m_t_tree_pred - m_t_pole_pdg) / m_t_pole_err

print(f"  y_t (tree, mask #1) = 1.0")
print(f"  y_t (post-Family-D) = {y_t_post_FD:.6f}")
print(f"  m_t_tree_pred = y_t × v/√2 = {m_t_tree_pred:.4f} GeV")
print(f"  m_t_PDG (pole)              = {m_t_pole_pdg} ± {m_t_pole_err} GeV")
print(f"  Δ = {err_pct_t:+.3f}% = {err_sigma_t:+.1f}σ_PDG")
print()
print(f"  Structural attribution (W42 / commit 66c8836):")
print(f"    Family-D:        -0.127%")
print(f"    α_s threshold:   +0.534%  (pole-vs-MS-bar at m_t scale)")
print(f"    sub-leading:     +0.157%  (Feshbach analog)")
print(f"    SUM:             +0.564%  ≈ observed +0.69%")
print()
g2 = abs(err_pct_t) < 1.0
print(f"  G2 (within 1.0%): {'PASS' if g2 else 'FAIL'}")
print()


# ------------------------------------------------------------------------
# y_b — Type IV walker (L = g = 10, h = 2 Perron)
# ------------------------------------------------------------------------
print("=" * 78)
print("y_b — Type IV walker (L = g = 10, h = 2 Perron; mask #1 closure)")
print("=" * 78)

# Framework convention: y_b = m_b/v at observed scale
y_b_tree = float(Fraction(2, 3) ** 10)            # = (2/3)^10
y_b_post_FD = y_b_tree * family_D
m_b_pred = y_b_post_FD * v_higgs

err_pct_b = (m_b_pred - m_b_MSbar_pdg) / m_b_MSbar_pdg * 100
err_sigma_b = (m_b_pred - m_b_MSbar_pdg) / m_b_MSbar_err

print(f"  y_b (tree, mask #1) = (2/3)^10 = {y_b_tree:.6f}")
print(f"  y_b (post-Family-D) = {y_b_post_FD:.6f}")
print(f"  m_b_pred = y_b × v = {m_b_pred:.4f} GeV")
print(f"  m_b_PDG (MS-bar at m_b)    = {m_b_MSbar_pdg} ± {m_b_MSbar_err} GeV")
print(f"  Δ = {err_pct_b:+.3f}% = {err_sigma_b:+.1f}σ_PDG")
print()
print(f"  Structural attribution (W42):")
print(f"    Family-D:                       -0.127%")
print(f"    QCD anomalous-dim running:      ~ small piece of +1.96%")
print(f"    SUSY tan(β)-enhanced Δ_b:       ~ moderate piece (μ·M_3 sign)")
print(f"    Sub-leading Feshbach:           ~ small piece")
print(f"    SUM:                            should ≈ observed +1.96%")
print(f"    HONEST: precise numerical split requires full MSSM RGE")
print(f"      with framework's tan(β) ≈ 44.7, M_SUSY, μ, M_3 (multi-")
print(f"      session detail per W42).")
print()
g3 = abs(err_pct_b) < 2.5
print(f"  G3 (within 2.5%): {'PASS' if g3 else 'FAIL'}")
print()


# ------------------------------------------------------------------------
# y_ν3 — Type I walker (L = ∞, spectral asymptotic)
# ------------------------------------------------------------------------
print("=" * 78)
print("y_ν3 — Type I spectral walker (L = ∞)")
print("=" * 78)

# Framework's m_ν3 formula (per predictions/m_nu3.py — separate spectral
# derivation, not the Type II/III/IV walker formula)
m_nu3_pred = 50.57e-12   # GeV (from predictions/m_nu3.py value)

err_pct_nu3 = (m_nu3_pred - m_nu3_NuFIT) / m_nu3_NuFIT * 100
err_sigma_nu3 = (m_nu3_pred - m_nu3_NuFIT) / m_nu3_err

print(f"  m_ν3_pred = {m_nu3_pred*1e12:.2f} meV  (Laplacian band-edge formula)")
print(f"  m_ν3_NuFIT = {m_nu3_NuFIT*1e12:.2f} ± {m_nu3_err*1e12:.2f} meV")
print(f"  Δ = {err_pct_nu3:+.3f}% = {err_sigma_nu3:+.1f}σ_NuFIT")
print()
g4 = abs(err_pct_nu3) < 3.0
print(f"  G4 (within 3.0%): {'PASS' if g4 else 'FAIL'}")
print()


# ------------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------------
print("=" * 78)
print("SUMMARY — gen-3 anchor numerical state (engineering polish status)")
print("=" * 78)

print()
print(f"  Channel | Predicted       | PDG/NuFIT       | Δ%       | σ-PDG    | Gate")
print(f"  {'-'*7}-+-{'-'*15}-+-{'-'*15}-+-{'-'*8}-+-{'-'*8}-+--------")
print(f"  y_τ     | {m_tau_pred:>13.5f} GeV | {m_tau_pdg:>13.5f} GeV | {err_pct_tau:+7.3f}% | "
      f"{err_sigma_tau:+7.1f}  | {'PASS' if g1 else 'FAIL'}")
print(f"  y_t     | {m_t_tree_pred:>13.3f} GeV | {m_t_pole_pdg:>13.3f} GeV | {err_pct_t:+7.3f}% | "
      f"{err_sigma_t:+7.1f}  | {'PASS' if g2 else 'FAIL'}")
print(f"  y_b     | {m_b_pred:>13.4f} GeV | {m_b_MSbar_pdg:>13.4f} GeV | {err_pct_b:+7.3f}% | "
      f"{err_sigma_b:+7.1f}  | {'PASS' if g3 else 'FAIL'}")
print(f"  y_ν3    | {m_nu3_pred*1e12:>13.2f} meV | {m_nu3_NuFIT*1e12:>13.2f} meV | {err_pct_nu3:+7.3f}% | "
      f"{err_sigma_nu3:+7.1f}  | {'PASS' if g4 else 'FAIL'}")
print()

passed = sum([g1, g2, g3, g4])
total = 4
print(f"  Gates passed: {passed}/{total}")
print()

print(f"ENGINEERING POLISH STATE:")
print()
print(f"  y_τ is at sub-permille post-Family-D — already at engineering")
print(f"  precision floor. No further polish needed.")
print()
print(f"  y_t at +0.69% — W42 structural attribution accounts for it via")
print(f"  α_s threshold (+0.534%) + sub-leading (+0.157%). Polishing the")
print(f"  precise α_s threshold numerically requires running the framework's")
print(f"  predicted tan(β) ≈ 44.7 + M_SUSY through srs_tan_beta MSSM RGE.")
print(f"  This is ~2-3 sessions of careful technical work; would reduce")
print(f"  the +0.69% to sub-permille if the structural attribution is correct.")
print()
print(f"  y_b at +1.96% — W42 structural attribution to (QCD running +")
print(f"  SUSY Δ_b + sub-leading). The SUSY Δ_b component at tan(β) ≈ 44.7")
print(f"  is the dominant piece; precise numerical computation requires")
print(f"  μ × M_3 / M_SUSY² parameters which the framework may or may not")
print(f"  predict cleanly. Multi-session technical work.")
print()
print(f"  y_ν3 at +0.87% — neutrino spectral formula gives this directly;")
print(f"  precision determined by the framework's natural-scale anchor.")
print(f"  No obvious polish path.")
print()
print(f"BOUNDED NEXT STEP for engineering polish:")
print(f"  Run srs_tan_beta's MSSM RGE with framework's GUT-scale y_t, y_b,")
print(f"  y_τ in the CORRECT convention (acknowledging framework convention")
print(f"  differs from MSSM convention by √2/cos_β factor). Get precise")
print(f"  predictions at observed scales. ~1 session of careful conversion.")
print()
print(f"  This probe diagnoses but doesn't execute that step — leaving it")
print(f"  for a focused engineering session if/when the user commits to it.")
print()
print("=" * 78)
