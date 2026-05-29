#!/usr/bin/env python3
"""
W17 — Test candidate κ_j = α₁²·c_S/μ_rep_j at f-level for Koide-ratio residuals
       (2026-05-26).

W16 found α₁²/54 = 28.2 ppm matches the Koide-ratio common-mode at 87%.
That's the α₁² extension of A_s's 1/54 prefactor. But the A_s mechanism
acts at Γ-fiber (B_NB Perron-residue singlet), while Koide ratios live
at P-fiber via V_Ram amplitudes — different sectors.

ALTERNATIVE CANDIDATE: κ_j (on f_j) = α₁²·c_S/μ_rep_j
                                    = α₁²/(12·μ_rep_j)

where:
  α₁² = Family-D leading dark-correction scale
  c_S = 1/12 = 1/(2|E|) = handshake-lemma form of N_atoms·k*
        (Perron-residue singlet projection per unified-oblique §3.2)
  μ_rep_j = (4, 2, 2) = C₃-rep multiplicities on V_Ram (Q_Koide.py)

This shape uses ONLY framework theorem-grade ingredients:
  • c_S = 1/12 — unified oblique §3.2 (theorem-grade)
  • μ_rep_j — Q_Koide.py Ramanujan-subspace multiplicities (theorem-grade)
  • α₁² — Family-D scale (theorem-grade)

The structural interpretation (CANDIDATE):
  At α₁² order, the per-fermion-leg dark correction is c_S/μ_rep_j
  instead of the rep-universal c_S = 1/12. The rep-resolution arises
  because each fermion's amplitude couples to V_Ram via the rep-j
  subspace (dim μ_rep_j). The Perron-residue projection at Γ remains
  c_S = 1/12, but per-rep its CONTRIBUTION TO Koide-ratio observables
  scales by 1/μ_rep_j.

The master-doc Family-D c_F_universal = -α₁²/12 corresponds to the
universal Yukawa-vertex correction that CANCELS in mass ratios. The
sub-leading rep-resolved correction κ_j = α₁²·c_S/μ_rep_j is the
RESIDUAL piece that survives in ratios because of μ_rep_j-dependence.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1
from Q_Koide import chain_import_ramanujan_multiplicities
from fractions import Fraction

d = predict_d_spatial()
k_star = int(round(predict_k_star(d)))
g = predict_g_girth(k_star, d)
alpha_1 = float(predict_alpha_1(k_star, g))
mu_t, mu_o, mu_w = chain_import_ramanujan_multiplicities()
N_atoms = 4
two_E = 2 * 6   # 2|E| for srs primitive cell with |E|=6

# Framework primitives
c_S_frac = Fraction(1, two_E)         # 1/12 from unified oblique §3.2
alpha_1_sq = alpha_1 ** 2

print("=" * 76)
print("W17 — κ_j = α₁²·c_S/μ_rep_j candidate for Koide-ratio common-mode")
print("=" * 76)
print()
print(f"Framework primitives (theorem-grade):")
print(f"  α₁_bare² = {alpha_1_sq*1e6:.4f} ppm")
print(f"  c_S      = 1/(2|E|) = 1/{two_E} = {float(c_S_frac):.6f}")
print(f"           (Perron-residue singlet projection, unified-oblique §3.2)")
print(f"  μ_rep_j  = (μ_trivial, μ_ω, μ_ω̄) = ({mu_t}, {mu_o}, {mu_w})")
print(f"           (V_Ram C₃ multiplicities, Q_Koide.py theorem-grade)")
print()

# Compute κ for each rep
kappa_t  = alpha_1_sq * float(c_S_frac) / mu_t
kappa_o  = alpha_1_sq * float(c_S_frac) / mu_o
kappa_ob = alpha_1_sq * float(c_S_frac) / mu_w

print(f"Predicted κ_j (at f-level) = α₁²·c_S/μ_rep_j:")
print(f"  κ_trivial = α₁²·(1/12)/4 = α₁²/48 = {kappa_t*1e6:.3f} ppm  (τ)")
print(f"  κ_ω        = α₁²·(1/12)/2 = α₁²/24 = {kappa_o*1e6:.3f} ppm  (e)")
print(f"  κ_ω̄       = α₁²·(1/12)/2 = α₁²/24 = {kappa_ob*1e6:.3f} ppm  (μ)")
print()

# Differences (predicted)
dk_o  = kappa_o  - kappa_t
dk_ob = kappa_ob - kappa_t
print(f"Predicted Koide-ratio shifts (κ_j − κ_τ at f-level):")
print(f"  κ_ω − κ_τ  = α₁²·c_S·(1/μ_ω − 1/μ_t) = α₁²·c_S·(1/2 − 1/4) = α₁²/48 = {dk_o*1e6:.3f} ppm")
print(f"  κ_ω̄ − κ_τ = α₁²·c_S·(1/μ_ω̄ − 1/μ_t) = α₁²/48 = {dk_ob*1e6:.3f} ppm")
print()

# Observed
obs_e_minus_tau  = 35.166e-6
obs_mu_minus_tau = 30.249e-6
print(f"Observed (from W1 back-solve, m_τ at PDG):")
print(f"  κ_ω − κ_τ  obs = +{obs_e_minus_tau*1e6:.3f} ppm")
print(f"  κ_ω̄ − κ_τ obs = +{obs_mu_minus_tau*1e6:.3f} ppm")
print()

# Match
match_e  = dk_o / obs_e_minus_tau
match_mu = dk_ob / obs_mu_minus_tau
print(f"Match ratios:")
print(f"  ω (electron):  predicted/observed = {match_e:.4f} ({(match_e-1)*100:+.1f}%)")
print(f"  ω̄ (muon):     predicted/observed = {match_mu:.4f} ({(match_mu-1)*100:+.1f}%)")
print()

# Common-mode and asymmetry
pred_avg = (dk_o + dk_ob) / 2
pred_asym = (dk_o - dk_ob) / 2
obs_avg = (obs_e_minus_tau + obs_mu_minus_tau) / 2
obs_asym = (obs_e_minus_tau - obs_mu_minus_tau) / 2
print(f"Common-mode + asymmetry decomposition:")
print(f"  Common-mode (avg):     predicted {pred_avg*1e6:.3f} ppm, obs {obs_avg*1e6:.3f} ppm, ratio {pred_avg/obs_avg:.4f}")
print(f"  Asymmetric (ω−ω̄)/2:    predicted {pred_asym*1e6:.3f} ppm, obs {obs_asym*1e6:.3f} ppm")
print()
print(f"→ Common-mode match: {pred_avg/obs_avg*100:.1f}% precision")
print(f"→ The α₁²·c_S/μ_rep_j shape captures the common-mode at ~3% precision.")
print(f"→ The ω/ω̄ asymmetry +{obs_asym*1e6:.3f} ppm remains a SEPARATE open piece")
print(f"  (the proposed shape predicts κ_ω = κ_ω̄ identically since μ_ω = μ_ω̄ = 2).")
print()

# Born-rule interpretation
print("=" * 76)
print("Mass-level interpretation (Born rule m = |amp|², or equivalently m ∝ f²)")
print("=" * 76)
print()
print(f"If the κ_j shape is on the f_j (sqrt-amplitude) level:")
print(f"  m_j = m_τ·(f_j/f_max)² gets correction (1 + 2(κ_j - κ_τ)) by Born squaring.")
print()
print(f"Mass-level Koide-ratio predictions:")
print(f"  c_e − 1 (at m-level) = 2·(κ_ω − κ_τ) = 2·α₁²/48 = α₁²/24 = {2*dk_o*1e6:.3f} ppm")
print(f"  c_μ − 1 (at m-level) = 2·(κ_ω̄ − κ_τ) = α₁²/24 = {2*dk_ob*1e6:.3f} ppm")
print()
# Observed at m-level
obs_c_e_minus_1  = 70.33e-6
obs_c_mu_minus_1 = 60.50e-6
print(f"Observed c_e − 1 (m-level)  = +{obs_c_e_minus_1*1e6:.3f} ppm")
print(f"Observed c_μ − 1 (m-level)  = +{obs_c_mu_minus_1*1e6:.3f} ppm")
print(f"Average                       = +{(obs_c_e_minus_1+obs_c_mu_minus_1)/2*1e6:.3f} ppm")
print(f"Predicted (avg, both reps)    = +{2*dk_o*1e6:.3f} ppm")
print(f"Match: {2*dk_o/((obs_c_e_minus_1+obs_c_mu_minus_1)/2)*100:.1f}%")
print()

# Structural derivation status
print("=" * 76)
print("STRUCTURAL DERIVATION ATTEMPT")
print("=" * 76)
print("""
PROPOSED MECHANISM:
At α₁² order, the per-fermion-leg dark correction at the Yukawa vertex
is rep-RESOLVED via the V_Ram C₃ structure. The leading (rep-universal)
piece gives Family-D c_F = -α₁²/(N_atoms·k*) = -α₁²/12 per leg, which
cancels in mass ratios. The SUB-LEADING rep-RESOLVED piece is

    c_F^(rep)_j = -α₁² · c_S / μ_rep_j = -α₁²/(12 μ_rep_j)
                                           per fermion leg

For τ (trivial, μ=4):  c_F^(rep)_τ = -α₁²/48
For e (ω, μ=2):        c_F^(rep)_ω = -α₁²/24
For μ (ω̄, μ=2):       c_F^(rep)_ω̄ = -α₁²/24

The Yukawa vertex correction:
    δy_j^(rep) = -2·c_F^(rep)_j = +α₁²·c_S/(... )
    no — let me redo

Family-D leg-counting: δy_j/y_j = -(n_H·c_H + n_F·c_F).
For y_τ with n_H=1, n_F=2, c_H = α₁² rep-universal, c_F has BOTH rep-universal
and rep-resolved pieces:
    c_F_total = c_F_universal + c_F^(rep)
              = -α₁²/12 - α₁²/(12 μ_rep_j)
              = -α₁²/12 · (1 + 1/μ_rep_j)

For τ:   c_F_τ = -α₁²/12 · (1 + 1/4) = -α₁²/12 · 5/4 = -5α₁²/48
For ω:   c_F_ω = -α₁²/12 · (1 + 1/2) = -α₁²/12 · 3/2 = -α₁²/8
For ω̄:  c_F_ω̄ = -α₁²/12 · 3/2 = -α₁²/8

δy_j/y_j = -(c_H + 2·c_F_j) = -(α₁² + 2c_F_j)
For τ:   δy_τ = -(α₁² + 2·(-5α₁²/48)) = -(α₁² - 5α₁²/24) = -(24-5)/24·α₁² = -19α₁²/24
For ω:   δy_ω = -(α₁² + 2·(-α₁²/8)) = -(α₁² - α₁²/4) = -3α₁²/4
For ω̄:  δy_ω̄ = -3α₁²/4

c_e - 1 = δy_ω - δy_τ = -3α₁²/4 - (-19α₁²/24) = (-18 + 19)/24·α₁² = α₁²/24

Numerically: α₁²/24 = 63.4 ppm at m-level.

Observed c_e - 1 = 70.3 ppm. Match at 90.2%.
Observed c_μ - 1 = 60.5 ppm. Match at 105%.
""")

# Verify with rep-resolved c_F
c_F_universal = -alpha_1_sq / two_E   # = -α₁²/12
c_F_rep_tau   = c_F_universal / mu_t  # extra rep-resolved piece
c_F_rep_o     = c_F_universal / mu_o
c_F_rep_ob    = c_F_universal / mu_w

c_F_total_tau = c_F_universal + c_F_rep_tau
c_F_total_o   = c_F_universal + c_F_rep_o
c_F_total_ob  = c_F_universal + c_F_rep_ob

# Yukawa correction at α₁² (n_H = 1, n_F = 2):
c_H_alpha2 = alpha_1_sq
dy_tau = -(c_H_alpha2 + 2*c_F_total_tau)
dy_o   = -(c_H_alpha2 + 2*c_F_total_o)
dy_ob  = -(c_H_alpha2 + 2*c_F_total_ob)

print(f"Numerical check with c_F = c_F_universal + c_F^(rep) = -α₁²/12·(1 + 1/μ_rep_j):")
print(f"  c_F_τ = -α₁²/12·(1 + 1/4) = {c_F_total_tau*1e6:.3f} ppm")
print(f"  c_F_ω = -α₁²/12·(1 + 1/2) = {c_F_total_o*1e6:.3f} ppm")
print(f"  c_F_ω̄ = -α₁²/12·(1 + 1/2) = {c_F_total_ob*1e6:.3f} ppm")
print()
print(f"Yukawa-vertex correction δy_j/y_j = -(c_H + 2c_F_j):")
print(f"  δy_τ = {dy_tau*1e6:.3f} ppm")
print(f"  δy_ω = {dy_o*1e6:.3f} ppm")
print(f"  δy_ω̄ = {dy_ob*1e6:.3f} ppm")
print()
print(f"Koide-ratio shifts (at m-level, using m = v·y so δm = δy):")
print(f"  c_e − 1 = δy_ω − δy_τ = {(dy_o - dy_tau)*1e6:.3f} ppm")
print(f"  c_μ − 1 = δy_ω̄ − δy_τ = {(dy_ob - dy_tau)*1e6:.3f} ppm")
print()
print(f"Compared to observation:")
print(f"  c_e − 1 obs = 70.33 ppm.  Predicted = {(dy_o - dy_tau)*1e6:.2f} ppm.  Match: {(dy_o - dy_tau)/obs_c_e_minus_1*100:.1f}%")
print(f"  c_μ − 1 obs = 60.50 ppm.  Predicted = {(dy_ob - dy_tau)*1e6:.2f} ppm.  Match: {(dy_ob - dy_tau)/obs_c_mu_minus_1*100:.1f}%")
print()
print(f"→ Rep-symmetric prediction (κ_ω = κ_ω̄ = α₁²/24); ω/ω̄ asymmetry remains open.")
print()

print("=" * 76)
print("STATUS RELATIVE TO LINTER 9-CLAUSE GATE")
print("=" * 76)
print(f"""
Candidate shape: c_F^(rep)_j = -α₁²·c_S/μ_rep_j per fermion leg,
                with c_S = 1/(2|E|) = 1/12 from unified oblique §3.2.

Total Family-D c_F (with rep-resolved sub-leading):
    c_F_total_j = -α₁²/12 · (1 + 1/μ_rep_j)

Predicted Koide-ratio shifts at m-level (rep-symmetric piece):
    c_e − 1 = c_μ − 1 = α₁²/24 = 63.4 ppm

Observed: c_e − 1 = 70.3, c_μ − 1 = 60.5 ppm. Match at 90% (ω) and 105% (ω̄).
Common-mode average match: 97%.

9-CLAUSE GATE STATUS:
  Clause 1 (axiom):           N/A
  Clause 2 (algebra):         PASS (K-rational arithmetic)
  Clause 3 (known theorem):   c_S from unified oblique §3.2 [theorem-grade];
                              μ_rep_j from Q_Koide.py [theorem-grade].
                              BUT: combining them as c_F^(rep) = -α₁²·c_S/μ_rep_j
                              at Yukawa vertex requires NEW derivation
                              (the master doc Family-D Clause-6 two-step
                              gives universal c_F via canonical_encoding; the
                              rep-resolved analog needs an analogous
                              channel_select → canonical_encoding at the
                              rep-resolved channel).
  Clause 4 (predictions/):    α₁ [closed]; c_S [closed via unified oblique];
                              μ_rep_j [closed via Q_Koide]. ALL closed.
  Clause 5 (master-theorem):  Master doc Family-D §3 D contains rep-universal
                              c_F; rep-resolved sub-leading would extend §3 D
                              with the new theorem. NOT yet present.
  Clause 6 (K-meta-theorem):  PASS (all K-rational; α₁²/μ_rep_j ∈ ℚ ⊂ K).
                              Selection step: channel_select to the rep-resolved
                              single-edge channel; canonical_encoding within
                              that channel gives μ_rep_j denominator. Needs
                              explicit formalization.
  Clause 7 (audit-v2):        NOT attempted for this extension.
  Clause 8 (numerical):       PARTIAL — 97% common-mode match; ω/ω̄ asymmetry
                              ±5 ppm remains open (within master doc §8b
                              ~0.5% Yukawa systematic).
  Clause 9 (π-audit):         PASS (no π).

VERDICT: CANDIDATE-GRADE STRUCTURAL extension.
  Strengths: uses ONLY theorem-grade framework ingredients (c_S, μ_rep_j, α₁²);
             97% common-mode match.
  Gaps: explicit derivation of the rep-resolved c_F^(rep) shape via Clause-6
        two-step (channel_select at rep-resolved level → canonical_encoding
        with μ_rep_j denominator) is REQUIRED to upgrade to theorem-grade.

HONEST POSITION:
This is genuinely more promising than the α₁³ Family-D extension I tried in
W4-W10. It uses framework-native theorem-grade objects and gives a clean
~3% common-mode match without requiring 24-cycle decomposition or 3-way
joint walker structures.

But it's STILL not theorem-grade. The structural derivation of why c_F
at the Yukawa vertex decomposes as c_F_universal + c_F^(rep) at α₁² order
is the missing piece.

WHAT WOULD CLOSE IT:
A formal derivation of the Yukawa-vertex C3-rep-resolved Clause-6 two-step:
  - channel_select(rep_j, c=single-edge-spectral-rep-j): the rep-resolved
    single-edge channel at vertex of generation j
  - canonical_encoding S' with mu_rep_j-dependent Hilbert dimension count
    giving denominator mu_rep_j

If this derivation closes, c_F^(rep) = -α₁²·c_S/μ_rep_j is theorem-grade,
and the m_e/m_μ Koide-ratio common-mode at 97% is theorem-grade-numerical.

This is a 1-3 session research probe at most. Much more bounded than the
α₁³ extension chain (which required 24-cycle decomposition or 3-way joint
walker, neither of which closed).
""")
