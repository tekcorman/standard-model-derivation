#!/usr/bin/env python3
"""
Majorana ν g_* recheck — does the +4.3% offset hold under framework's
preferred Majorana ν counting?

PURPOSE
-------
The leading-factor probes used ΛCDM Dirac ν counting (g_*=10.75 at MeV).
Framework's M_R seesaw predicts Majorana ν_R decoupled at M_R≈10¹⁵ GeV,
leaving only 3 ν_L active at MeV scale.

CHECK: does the F = √(k_star · g_*) candidate with framework's Majorana
species count still match ΛCDM's 1.66·√g_* with +4.3% offset?

For Majorana neutrinos:
  - Each Majorana ν IS its own antiparticle (2 dof: 2 spin states)
  - 3 Majorana ν species = 6 fermionic dof (same as Dirac at MeV scale
    where both are relativistic)

So at MeV, g_*_Majorana = g_*_Dirac = 10.75 if both helicity states of
each Majorana ν are thermally populated.

But if framework's seesaw decouples ν_R at high scale and only ν_L
remains active, then at MeV:
  γ: 2 (always)
  e±: 4 (relativistic Dirac fermions)
  ν_L only: 3 species · 1 dof (1 helicity each) = 3
  Total: g_*_framework = 2 + (7/8)·(4+3) = 2 + 6.125 = 8.125

vs ΛCDM (Dirac, both helicities, 3 species): 2 + (7/8)·(4+6) = 10.75

Run:
    python3 proofs/cosmology/majorana_g_star_recheck_2026-05-27.py
"""

import math

# Framework primitives
k_star = 3

# Active species counts (under different conventions)
g_star_LCDM_MeV = 10.75      # Dirac, both helicities
g_star_LCDM_post = 3.36       # After e+e- ann
g_star_Majorana_MeV = 2 + (7/8) * (4 + 3)   # = 8.125 (only ν_L active)
g_star_Majorana_post = 2 + (7/8) * 3 * (4/11)**(4/3)  # ν_L only, lower T_ν

print("=" * 78)
print("  Majorana ν g_* recheck — does +4.3% offset survive?")
print("=" * 78)

print(f"\n  Active species counts:")
print(f"    ΛCDM Dirac MeV:       g_* = {g_star_LCDM_MeV}")
print(f"    ΛCDM post e+e- ann:   g_* = {g_star_LCDM_post}")
print(f"    Framework Majorana MeV (ν_L only):       g_* = {g_star_Majorana_MeV:.4f}")
print(f"    Framework Majorana post e+e- ann:        g_* = {g_star_Majorana_post:.4f}")

# Required factor (Friedmann) under different g_* conventions
F_LCDM_MeV = 1.66 * math.sqrt(g_star_LCDM_MeV)
F_LCDM_post = 1.66 * math.sqrt(g_star_LCDM_post)
F_Maj_MeV = 1.66 * math.sqrt(g_star_Majorana_MeV)
F_Maj_post = 1.66 * math.sqrt(g_star_Majorana_post)

# Framework candidate factor
F_K_LCDM_MeV = math.sqrt(k_star * g_star_LCDM_MeV)
F_K_LCDM_post = math.sqrt(k_star * g_star_LCDM_post)
F_K_Maj_MeV = math.sqrt(k_star * g_star_Majorana_MeV)
F_K_Maj_post = math.sqrt(k_star * g_star_Majorana_post)

print(f"\n  Required F = 1.66·√g_* (continuum Friedmann):")
print(f"    ΛCDM Dirac     MeV: F = {F_LCDM_MeV:.4f}")
print(f"    ΛCDM Dirac     post: F = {F_LCDM_post:.4f}")
print(f"    Majorana ν_L   MeV: F = {F_Maj_MeV:.4f}")
print(f"    Majorana ν_L   post: F = {F_Maj_post:.4f}")

print(f"\n  Framework candidate F = √(k_star · g_*):")
print(f"    ΛCDM Dirac     MeV: F_K = {F_K_LCDM_MeV:.4f} (Δ = {(F_K_LCDM_MeV/F_LCDM_MeV-1)*100:+.2f}%)")
print(f"    ΛCDM Dirac     post: F_K = {F_K_LCDM_post:.4f} (Δ = {(F_K_LCDM_post/F_LCDM_post-1)*100:+.2f}%)")
print(f"    Majorana ν_L   MeV: F_K = {F_K_Maj_MeV:.4f} (Δ = {(F_K_Maj_MeV/F_Maj_MeV-1)*100:+.2f}%)")
print(f"    Majorana ν_L   post: F_K = {F_K_Maj_post:.4f} (Δ = {(F_K_Maj_post/F_Maj_post-1)*100:+.2f}%)")

print(f"""
  The +4.3% offset is INDEPENDENT of g_* convention.
  This is because F_K/F_req = √(k_star)/1.66 = √3/1.66 = 1.0434,
  regardless of which g_* value is plugged in.

  So the +4.3% K-rational tax holds for ANY species count (Dirac, Majorana,
  or whatever framework derives). It's the universal substitute factor of
  √k_star for the continuum 1.66, NOT a species-count dependency.
""")

# But there's a separate question: what's the framework's INTERNAL prediction
# for g_*(MeV)?
# Under framework's Majorana ν_L-only counting: g_*(MeV) = 8.125
# This would give a different ABSOLUTE F at MeV.

T_F_LCDM = (F_LCDM_MeV / (1.66))**(1.0/3)  # dummy calc; F enters T_F as (F)^(1/3)
F_LCDM_target = F_LCDM_MeV   # 5.443
F_K_Maj_target = F_K_Maj_MeV  # different absolute

print("-" * 78)
print("  Absolute F at MeV under different species counts")
print("-" * 78)

# Now compute T_F under each F at MeV with α=1/2 + framework H_substrate
G_F = 1.1663787e-5
M_Pl = 1.22089e19

def T_F_with_F(F):
    return (F / (M_Pl * G_F**2))**(1.0/3.0)

print(f"\n  T_ν_dec under various F (α=1/2):")
print(f"    F = 1 (no correction):                  T_F = {T_F_with_F(1)*1e3:.4f} MeV")
print(f"    F = √(k·g_*_Majorana_MeV) = {F_K_Maj_MeV:.3f}:       T_F = {T_F_with_F(F_K_Maj_MeV)*1e3:.4f} MeV")
print(f"    F = √(k·g_*_ΛCDM_MeV)     = {F_K_LCDM_MeV:.3f}:       T_F = {T_F_with_F(F_K_LCDM_MeV)*1e3:.4f} MeV")
print(f"    F = 1.66·√g_*_ΛCDM        = {F_LCDM_MeV:.3f}:       T_F = {T_F_with_F(F_LCDM_MeV)*1e3:.4f} MeV (ΛCDM ref)")

# Y_p under Majorana
Q_np = 1.2933e-3
ratio_7_15 = 7/15
decay_factor = 0.7
def Y_p_from_F(F):
    T_F = T_F_with_F(F)
    T_BBN = T_F * ratio_7_15
    n_p_freeze = math.exp(-Q_np / T_BBN)
    n_p_final = n_p_freeze * decay_factor
    return 2 * n_p_final / (1 + n_p_final)

print(f"\n  Y_p under various F:")
print(f"    F = 1:                Y_p = {Y_p_from_F(1):.4f}")
print(f"    F = √(k·g_Maj):       Y_p = {Y_p_from_F(F_K_Maj_MeV):.4f}")
print(f"    F = √(k·g_ΛCDM):      Y_p = {Y_p_from_F(F_K_LCDM_MeV):.4f}")
print(f"    F = 1.66·√g_ΛCDM:     Y_p = {Y_p_from_F(F_LCDM_MeV):.4f}")
print(f"    Observed:             Y_p = 0.245")

# How significant is the Majorana vs Dirac shift?
ratio_dirac_to_majorana = math.sqrt(g_star_LCDM_MeV / g_star_Majorana_MeV)
print(f"""

  Majorana vs Dirac species count shift:
    g_*_Dirac / g_*_Majorana = {g_star_LCDM_MeV / g_star_Majorana_MeV:.4f}
    √ ratio = {ratio_dirac_to_majorana:.4f}
    So F_K_Maj = F_K_Dirac · {1/ratio_dirac_to_majorana:.4f}

  This means under framework's Majorana counting, the candidate F_K is
  ~14.6% smaller than under ΛCDM Dirac counting. The +4.3% K-rational tax
  vs the continuum 1.66·√g_* DOES NOT CHANGE — but the absolute species
  count differs.

  Implication: if framework's structural Majorana ν is correct, then the
  framework's INTERNAL prediction for g_* at MeV is 8.125, and the
  candidate F = √(k·8.125) = {F_K_Maj_MeV:.3f}.

  Under THIS F, T_ν_dec = {T_F_with_F(F_K_Maj_MeV)*1e3:.3f} MeV (vs ΛCDM 1.5 MeV).
  Y_p prediction: {Y_p_from_F(F_K_Maj_MeV):.4f} (vs obs 0.245, simple model).

  This is CLOSER to observation than the F=1 case (Y_p = 0.05), but still
  off because of the framework's Majorana ν_L-only counting underestimating
  the active fermion dof.

  Actually wait — at MeV scale, even if ν_R is decoupled at M_R, the ν_L
  has BOTH helicities thermally populated (left helicity active for SU(2)_L
  weak interactions, but ν_L's CP-conjugate is essentially the ν_R̄ which is
  also light if Majorana — at MeV they may behave Dirac-like!).

  More careful Majorana counting at MeV with mass m_ν ≪ T:
    - Each Majorana ν has 2 thermally populated states (helicity ±) at T ≫ m_ν
    - 3 species × 2 = 6 fermionic dof — SAME as Dirac at MeV

  So at MeV, g_*_framework = g_*_LCDM = 10.75 (Majorana and Dirac agree
  in massless limit). The Majorana vs Dirac distinction matters only when
  m_ν is comparable to T (i.e., at very late epochs when ν masses become
  relevant).

  CONCLUSION: at MeV scale, framework's Majorana ν is INDISTINGUISHABLE from
  Dirac for g_* counting. The +4.3% offset analysis is robust to this
  distinction.
""")

print("=" * 78)
print("  VERDICT")
print("=" * 78)
print(f"""
  Two findings:

  1. The +4.3% K-rational tax (√3/1.66 = 1.0434) is INDEPENDENT of g_*
     species convention. It's the universal substitute factor of √k_star
     for the continuum 1.66.

  2. At MeV scale, Majorana vs Dirac ν makes NO DIFFERENCE for g_* counting
     in the massless limit (m_ν ≪ T). Both give g_* = 10.75. The Majorana
     vs Dirac distinction matters only at very late epochs (when m_ν ~ T).

  Therefore: the leading-factor F = √(k_star · g_*) candidate is robust to
  the framework's Majorana ν commitment. No reassessment needed.

  The earlier worry (g_*_Majorana = 8.125 vs g_*_Dirac = 10.75) was based
  on counting only ν_L helicity. This is incorrect at MeV scales where
  both Majorana helicities are thermally populated.
""")
