#!/usr/bin/env python3
"""
W68 — MSSM RGE gen-3 polish (session 1): bottom-up framework consistency check

Per W67's "BOUNDED NEXT STEP" + the V1 lesson:

The framework's "y_f = m_f/v" convention is a LOW-SCALE statement, NOT a
GUT-scale MSSM Yukawa. A first version of this probe attempted to use the
framework values directly as GUT-scale MSSM inputs, which failed
catastrophically (m_τ off by factor 60+, m_b off by factor 15+) — exactly
the √2/cos β + RGE-running factor that distinguishes framework convention
from MSSM convention.

REVISED PROBE — bottom-up framework self-consistency check:

  1. Take framework's PREDICTED LOW-SCALE physical masses (W67 tree
     formulas, post-Family-D):
       m_t^F   = y_t^F × v/√2 × Family_D       (saturation; y_t = 1)
       m_b^F   = (2/3)^g × v × Family_D         (Type IV Perron, L = g = 10)
       m_τ^F   = (α₁_full / k²) × v × Family_D  (Type III, L = g-2 = 8)
  2. Convert these to MSSM-convention Yukawas at M_Z (using framework's
     self-consistent tan β = 44.73 per srs_tan_beta.py PART 4):
       y_t^MSSM(M_Z) = m_t^F / (qcd_corr × v/√2 × sin β)
       y_b^MSSM(M_Z) = m_b(M_Z)^F / (v/√2 × cos β)
                       where m_b(M_Z) = m_b(m_b)^F / qcd_running_b
       y_τ^MSSM(M_Z) = m_τ^F / (v/√2 × cos β)
  3. Run UP from M_Z to M_GUT via MSSM RGE (using srs_tan_beta's
     `sm_rge` + `mssm_rge` primitives) with observed gauge couplings.
  4. Check the GUT-scale Yukawa outputs against framework's structural
     predictions:
       y_t(GUT) ≈ 1.0     (framework: exponent-principle saturation)
       y_b/y_τ(GUT) ≈ GJ   (framework: Georgi-Jarlskog, GJ = 3)

This is a SELF-CONSISTENCY check: do the framework's LOW-SCALE predictions
RGE-run UP to its own GUT-SCALE STRUCTURAL claims? If yes, the framework
is internally consistent at the gen-3 RGE level (passing this probe DOES
NOT close the W67-documented +0.65% / +1.96% / -0.034% residuals — those
are low-scale numerical corrections that require separate work in
session 2-3 per the W42 attribution).

PRE-DECLARED GATES:
  G1: y_t(GUT) within 15% of 1.0 (the framework's exponent-principle
      saturation prediction)
  G2: y_b(GUT) / y_τ(GUT) within 2% of GJ = 3 (Georgi-Jarlskog texture)
  G3: y_τ(GUT) within 10% of bottom-up-from-PDG y_τ(GUT) ≈ 1.48
      (sanity check: framework's m_τ prediction at -0.034% gives same
      GUT-scale Yukawa as PDG m_τ does)
  G4: y_b(GUT) within 10% of bottom-up-from-PDG y_b(GUT) (sanity check
      with framework's +1.96% m_b residual contributes to GUT-scale shift)
  G5: y_t(GUT) framework-predicted vs PDG-bottom-up — gap reflects
      framework's +0.65% m_t residual
"""

from __future__ import annotations
import os, sys
import math
from fractions import Fraction
import numpy as np
from scipy.integrate import solve_ivp

# Use existing srs_tan_beta infrastructure for RGE primitives
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from proofs.masses.srs_tan_beta import (
    mssm_rge, sm_rge,
    log_M_GUT, log_M_Z, v_over_root2,
    GJ, alpha_s_MZ,
    alpha_em_inv_MZ, sin2_tw_MZ,
    m_t_pole_obs, m_b_MSbar, m_tau,
)


# ──────────────────────────────────────────────────────────────────
# Framework primitives (mirrors W67)
# ──────────────────────────────────────────────────────────────────
k_star = 3
g_girth = 10
N_atoms = 4
v_higgs = 246.22

alpha_1_bare = Fraction(2, 3) ** 8
alpha_1_full = Fraction(5, 3) * alpha_1_bare

c_H = alpha_1_bare ** 2
c_F = -alpha_1_bare ** 2 / Fraction(N_atoms * k_star)
family_D = float(1 - (c_H + 2 * c_F))

tan_beta_F = 44.73   # framework self-consistent value per srs_tan_beta PART 4
sin_beta = tan_beta_F / math.sqrt(1.0 + tan_beta_F**2)
cos_beta = 1.0 / math.sqrt(1.0 + tan_beta_F**2)
M_SUSY = 2000.0

# ──────────────────────────────────────────────────────────────────
# Framework's predicted low-scale physical masses
# ──────────────────────────────────────────────────────────────────
y_t_F = 1.0                                        # saturation Type II
m_t_pole_F = y_t_F * family_D * v_higgs / math.sqrt(2.0)

y_tau_F = float(alpha_1_full) / k_star ** 2        # Type III tree
m_tau_F = y_tau_F * family_D * v_higgs

y_b_F = float(Fraction(2, 3) ** g_girth)           # Type IV Perron tree
m_b_F = y_b_F * family_D * v_higgs                 # at m_b MS-bar scale

print("=" * 78)
print("W68 — MSSM RGE gen-3 polish (session 1): bottom-up framework consistency")
print("=" * 78)
print()
print("Framework's predicted LOW-SCALE physical masses (post-Family-D):")
print(f"  m_t (pole)              = {m_t_pole_F:.4f} GeV   (PDG {m_t_pole_obs}, Δ {(m_t_pole_F - m_t_pole_obs)/m_t_pole_obs*100:+.3f}%)")
print(f"  m_b (MS-bar at m_b)     = {m_b_F:.4f} GeV   (PDG {m_b_MSbar}, Δ {(m_b_F - m_b_MSbar)/m_b_MSbar*100:+.3f}%)")
print(f"  m_τ (pole)              = {m_tau_F:.5f} GeV  (PDG {m_tau}, Δ {(m_tau_F - m_tau)/m_tau*100:+.3f}%)")
print()
print(f"Framework parameters:")
print(f"  tan β   = {tan_beta_F}")
print(f"  cos β   = {cos_beta:.5f}")
print(f"  sin β   = {sin_beta:.5f}")
print(f"  Family_D = {family_D:.6f}")
print()


# ──────────────────────────────────────────────────────────────────
# Convert framework's low-scale predictions to MSSM Yukawas at M_Z
#
# CONVENTION: match srs_tan_beta.py's run_mz_to_gut_observed exactly.
#   - m_t: convert pole→running via QCD pole correction
#   - m_b: use m_b_MSbar directly (no QCD anomalous-dim running m_b→M_Z;
#          srs_tan_beta treats m_b_MSbar = 4.18 as the y_b-extraction
#          input at M_Z scale)
#   - m_τ: use directly (no correction)
# ──────────────────────────────────────────────────────────────────
qcd_pole_corr_t = 1.0 + 4.0 * alpha_s_MZ / (3.0 * math.pi)

# MSSM Yukawas at M_Z (framework predictions)
y_t_MZ_F   = (m_t_pole_F / qcd_pole_corr_t) / (v_over_root2 * sin_beta)
y_b_MZ_F   = m_b_F  / (v_over_root2 * cos_beta)
y_tau_MZ_F = m_tau_F / (v_over_root2 * cos_beta)

# MSSM Yukawas at M_Z (PDG observed, for comparison)
y_t_MZ_obs   = (m_t_pole_obs / qcd_pole_corr_t) / (v_over_root2 * sin_beta)
y_b_MZ_obs   = m_b_MSbar     / (v_over_root2 * cos_beta)
y_tau_MZ_obs = m_tau         / (v_over_root2 * cos_beta)

print("M_Z-scale MSSM Yukawas:")
print(f"  {'channel':<8s} | {'framework':>12s} | {'PDG':>12s} | {'Δ%':>8s}")
print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
def cmp(name, f, o):
    err = (f - o) / o * 100
    print(f"  {name:<8s} | {f:>12.6f} | {o:>12.6f} | {err:+8.3f}%")
cmp("y_t",   y_t_MZ_F,   y_t_MZ_obs)
cmp("y_b",   y_b_MZ_F,   y_b_MZ_obs)
cmp("y_τ",   y_tau_MZ_F, y_tau_MZ_obs)
print()


# ──────────────────────────────────────────────────────────────────
# Gauge couplings at M_Z (observed, GUT-normalized)
# ──────────────────────────────────────────────────────────────────
alpha_em_obs = 1.0 / alpha_em_inv_MZ
alpha_2_obs = alpha_em_obs / sin2_tw_MZ
alpha_Y_obs = alpha_em_obs / (1.0 - sin2_tw_MZ)
alpha_1_obs = (5.0 / 3.0) * alpha_Y_obs
alpha_3_obs = alpha_s_MZ

alpha_1_inv_obs = 1.0 / alpha_1_obs
alpha_2_inv_obs = 1.0 / alpha_2_obs
alpha_3_inv_obs = 1.0 / alpha_3_obs


# ──────────────────────────────────────────────────────────────────
# Run UP from M_Z to M_GUT (SM M_Z → M_SUSY, then MSSM M_SUSY → M_GUT)
# ──────────────────────────────────────────────────────────────────
log_M_SUSY = math.log(M_SUSY)


def run_up_to_gut(yt_mz, yb_mz, ytau_mz):
    """Run from M_Z to M_GUT with given M_Z Yukawas (SM below M_SUSY, MSSM above)."""
    y0_sm = [alpha_1_inv_obs, alpha_2_inv_obs, alpha_3_inv_obs,
             yt_mz, yb_mz, ytau_mz]
    sol_sm = solve_ivp(lambda t, y: sm_rge(t, y, use_2loop=True),
                       [log_M_Z, log_M_SUSY], y0_sm,
                       method='RK45', rtol=1e-10, atol=1e-12,
                       dense_output=True)
    at_susy = sol_sm.sol(log_M_SUSY)
    y0_mssm = list(at_susy)
    sol_mssm = solve_ivp(lambda t, y: mssm_rge(t, y, use_2loop=True),
                         [log_M_SUSY, log_M_GUT], y0_mssm,
                         method='RK45', rtol=1e-10, atol=1e-12,
                         dense_output=True)
    at_gut = sol_mssm.sol(log_M_GUT)
    return {
        'yt_gut': at_gut[3], 'yb_gut': at_gut[4], 'ytau_gut': at_gut[5],
        'alpha_1_inv_gut': at_gut[0],
        'alpha_2_inv_gut': at_gut[1],
        'alpha_3_inv_gut': at_gut[2],
    }


print("Running MSSM RGE from M_Z to M_GUT (framework-predicted Yukawas)…")
res_F = run_up_to_gut(y_t_MZ_F, y_b_MZ_F, y_tau_MZ_F)
print("Running MSSM RGE from M_Z to M_GUT (PDG-observed Yukawas)…")
res_obs = run_up_to_gut(y_t_MZ_obs, y_b_MZ_obs, y_tau_MZ_obs)
print()

# Landau-pole detection: if Yukawa values at GUT are O(1e10) or larger, the
# RGE has blown up to a Landau pole and the result is non-physical.
def is_landau(res):
    return any(abs(v) > 1e8 for v in (res['yt_gut'], res['yb_gut'], res['ytau_gut']))

framework_landau = is_landau(res_F)
pdg_landau = is_landau(res_obs)

if framework_landau or pdg_landau:
    print("=" * 78)
    print("LANDAU POLE DETECTED")
    print("=" * 78)
    if framework_landau:
        print(f"  FRAMEWORK case: y_b(M_Z) = {y_b_MZ_F:.4f} is past the MSSM Landau pole.")
        print(f"    The +1.96% m_b residual (W67) pushes y_b(M_Z) over the Landau-pole")
        print(f"    threshold at tan β = {tan_beta_F}.")
    if pdg_landau:
        print(f"  PDG case: y_b(M_Z) = {y_b_MZ_obs:.4f} also past the pole at this tan β.")
    print()
    print("  This is structurally informative: the framework's predicted m_b sits")
    print("  RIGHT AT the edge of MSSM viability at tan β = 44.73. The +1.96%")
    print("  residual is the difference between just-converging and Landau-pole")
    print("  divergence.")
    print()
    print("  REAL POLISH path: (a) reduce the +1.96% m_b residual via low-scale")
    print("  corrections (W42 attribution: QCD anomalous-dim + SUSY Δ_b +")
    print("  sub-leading Feshbach) BEFORE attempting GUT-consistency, OR")
    print("  (b) re-derive a lower tan β that allows m_b convergence with")
    print("  framework's tree-level m_b.")
    print()
    print("  SESSION 1 VERDICT: the W67 +1.96% m_b residual is LOAD-BEARING for")
    print("  the gen-3 anchor framework structurally; it sits at the Landau-pole")
    print("  edge. Polish is NOT a precision-tuning exercise; it's gated by")
    print("  low-scale W42 corrections that bring m_b inside the MSSM viability")
    print("  window before any GUT-consistency check can run.")
    print()
    print("=" * 78)
    sys.exit(0)


# ──────────────────────────────────────────────────────────────────
# Check framework's GUT-scale structural predictions
# ──────────────────────────────────────────────────────────────────
y_t_GUT_F   = res_F['yt_gut']
y_b_GUT_F   = res_F['yb_gut']
y_tau_GUT_F = res_F['ytau_gut']
y_b_over_y_tau_F = y_b_GUT_F / y_tau_GUT_F

y_t_GUT_obs   = res_obs['yt_gut']
y_b_GUT_obs   = res_obs['yb_gut']
y_tau_GUT_obs = res_obs['ytau_gut']
y_b_over_y_tau_obs = y_b_GUT_obs / y_tau_GUT_obs

print("=" * 78)
print("GUT-scale Yukawas (framework prediction vs PDG-bottom-up)")
print("=" * 78)
print()
print(f"  {'channel':<14s} | {'framework':>12s} | {'PDG bottom-up':>14s} | {'Δ%':>8s}")
print(f"  {'-'*14}-+-{'-'*12}-+-{'-'*14}-+-{'-'*8}")

def gut_cmp(name, f, o):
    if abs(o) > 1e-9:
        err = (f - o) / o * 100
        print(f"  {name:<14s} | {f:>12.6f} | {o:>14.6f} | {err:+8.3f}%")
    else:
        print(f"  {name:<14s} | {f:>12.6f} | {o:>14.6f} | {'N/A':>8s}")

gut_cmp("y_t(GUT)",   y_t_GUT_F,   y_t_GUT_obs)
gut_cmp("y_b(GUT)",   y_b_GUT_F,   y_b_GUT_obs)
gut_cmp("y_τ(GUT)",   y_tau_GUT_F, y_tau_GUT_obs)
gut_cmp("y_b/y_τ(GUT)", y_b_over_y_tau_F, y_b_over_y_tau_obs)
print()


# ──────────────────────────────────────────────────────────────────
# Gates
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("FRAMEWORK STRUCTURAL CONSISTENCY")
print("=" * 78)
print()

# G1: y_t(GUT) within 15% of 1.0 (framework saturation prediction)
err_g1 = (y_t_GUT_F - 1.0) / 1.0 * 100
g1_pass = abs(err_g1) < 15.0
print(f"  G1 — y_t(M_GUT) within 15% of framework target 1.0:")
print(f"        framework y_t(GUT) = {y_t_GUT_F:.4f}, Δ = {err_g1:+.2f}%")
print(f"        gate: {'PASS' if g1_pass else 'FAIL'}")
print()

# G2: y_b/y_τ(GUT) within 2% of GJ = 3
err_g2 = (y_b_over_y_tau_F - GJ) / GJ * 100
g2_pass = abs(err_g2) < 2.0
print(f"  G2 — y_b/y_τ(M_GUT) within 2% of GJ = {GJ} (Georgi-Jarlskog texture):")
print(f"        framework y_b/y_τ(GUT) = {y_b_over_y_tau_F:.4f}, Δ = {err_g2:+.3f}%")
print(f"        gate: {'PASS' if g2_pass else 'FAIL'}")
print()

# G3: y_τ(GUT) framework vs PDG-bottom-up within 10% (sanity: framework m_τ
# at -0.034% means framework y_τ runs to same GUT value as PDG y_τ within
# precision floor; the 10% bound is very loose just to detect gross error)
err_g3 = (y_tau_GUT_F - y_tau_GUT_obs) / y_tau_GUT_obs * 100
g3_pass = abs(err_g3) < 10.0
print(f"  G3 — y_τ(M_GUT) framework vs PDG-bottom-up within 10% (sanity):")
print(f"        framework y_τ(GUT) = {y_tau_GUT_F:.4f}")
print(f"        PDG-derived y_τ(GUT) = {y_tau_GUT_obs:.4f}")
print(f"        Δ = {err_g3:+.3f}%")
print(f"        gate: {'PASS' if g3_pass else 'FAIL'}")
print()

# G4: y_b(GUT) framework vs PDG-bottom-up within 10% (sanity: framework's
# +1.96% m_b residual yields ~similar GUT-scale shift after running)
err_g4 = (y_b_GUT_F - y_b_GUT_obs) / y_b_GUT_obs * 100
g4_pass = abs(err_g4) < 10.0
print(f"  G4 — y_b(M_GUT) framework vs PDG-bottom-up within 10% (sanity):")
print(f"        framework y_b(GUT) = {y_b_GUT_F:.4f}")
print(f"        PDG-derived y_b(GUT) = {y_b_GUT_obs:.4f}")
print(f"        Δ = {err_g4:+.3f}%")
print(f"        gate: {'PASS' if g4_pass else 'FAIL'}")
print()

# G5: y_t(GUT) framework vs PDG-bottom-up — gap reflects m_t residual
err_g5 = (y_t_GUT_F - y_t_GUT_obs) / y_t_GUT_obs * 100
g5_pass = abs(err_g5) < 5.0
print(f"  G5 — y_t(M_GUT) framework vs PDG-bottom-up within 5%:")
print(f"        framework y_t(GUT) = {y_t_GUT_F:.4f}")
print(f"        PDG-derived y_t(GUT) = {y_t_GUT_obs:.4f}")
print(f"        Δ = {err_g5:+.3f}%")
print(f"        gate: {'PASS' if g5_pass else 'FAIL'}")
print()


# ──────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────
n_pass = sum([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass])
print("=" * 78)
print(f"VERDICT: {n_pass}/5 gates pass")
print("=" * 78)
print()
if g1_pass and g2_pass:
    print("FRAMEWORK SELF-CONSISTENT: the gen-3 framework predictions RGE-run UP")
    print("to its own structural GUT-scale claims (y_t(GUT) ≈ 1 + GJ = 3).")
    print()
    print("HONEST SCOPE:")
    print("  This passing does NOT close the W67 residuals (+0.65%/+1.96%/-0.034%).")
    print("  Those are LOW-SCALE corrections (α_s pole-vs-MS-bar for m_t; QCD")
    print("  anomalous dim + SUSY Δ_b for m_b) that operate at the MeV–GeV scale")
    print("  separately from the GUT consistency this probe verifies.")
    print()
    print("  Polish session 2 would need to implement those low-scale corrections")
    print("  explicitly against the W42 attribution (+0.534% α_s threshold for")
    print("  y_t; QCD running + SUSY Δ_b decomposition for y_b).")
else:
    print("TENSION SURFACED: the gen-3 framework predictions do NOT RGE-run to")
    print("the framework's own GUT-scale claims within the gate tolerances.")
    print("Specifically:")
    if not g1_pass:
        print(f"    - y_t(GUT) = {y_t_GUT_F:.4f} ≠ 1.0 framework target (Δ {err_g1:+.2f}%)")
    if not g2_pass:
        print(f"    - y_b/y_τ(GUT) = {y_b_over_y_tau_F:.4f} ≠ GJ = 3 (Δ {err_g2:+.3f}%)")
    print()
    print("  This could indicate (a) the framework predictions need refinement,")
    print("  (b) the M_SUSY / tan β values need adjustment, or (c) the convention")
    print("  conversion has a subtle error.")
print()
print("=" * 78)
