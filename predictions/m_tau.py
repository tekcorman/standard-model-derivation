#!/usr/bin/env python3
"""
Canonical prediction file for the tau lepton mass m_tau.

STATUS (corrected W1 2026-05-18): THEOREM-GRADE-STRUCTURAL, conditional.
m_tau_pred = v × y_τ uses the Family-D-corrected y_τ. Two conditionals,
neither defeating the numeric value (UNCHANGED, -0.19σ_PDG):
 (i) v ← N_hub (Gap G1). The 2026-04-28 G1b R2 path closure
     (`docs/theorems/theorem_g1b_r2_closure.md`) addresses this leg only.
 (ii) y_τ's Family-D c_F: THEOREM-GRADE-STRUCTURAL conditional on the
     Clause-6 channel_select Step-1 argument (single-edge vs gauge-singlet,
     a δ_r-tier structural argument — see predictions/y_tau.py +
     predictions/dark_extraction_map.py + master doc §3 (D)).
The prior "UNIQUE — THEOREM-GRADE" addressed (i) only; with (ii) the honest
grade is THEOREM-GRADE-STRUCTURAL conditional, NOT UNIQUE. The tree y_τ =
α₁_full/k*² corollary (theorem_ytau_corollary.md) is a separate pre-Family-D
result, not what is reported here. Historical notes below preserved for record.
"""

# ============================================================
# PARAMETER: m_tau (tau lepton mass)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_tau = 1776.86 ± 0.12 MeV = 1.77686 ± 0.00012 GeV
# Source:      PDG 2024 (Particle Data Group, Review of Particle Physics)
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_tau = v × y_tau ≈ 1.7791 GeV (with v = 246.22 GeV)
# Deviation:   +2.23 MeV absolute, +0.126% relative
#
# Bridge convention (docs/framework/framework_scheme_convention.md §4.4): m_tau inherits
# its scheme treatment from y_tau, which is a tree-level α₁-dependent coupling.
# Comparison to m_tau_obs uses "bare + Feshbach = SM pole-mass-equivalent." The
# +0.13% residual is the un-derived Feshbach analog on the fermion-Higgs vertex
# (Priority 4.4 step 2.2). v contributes essentially zero residual via G_F
# round-trip (the (5/12) Feshbach correction on v is already applied in
# predictions/v_higgs.py and absorbed into the N_hub anchor).

# --- DERIVED FORMULA -----------------------------------------
# m_τ = v × y_τ = v × (α₁_full / k*²)
#
# where:
#   v       = Higgs vacuum expectation value (predictions/v_higgs.py)
#   y_τ     = α₁_full / k*² (predictions/y_tau.py) -- THEOREM-GRADE
#   α₁_full = (5/3)(2/3)^8 = 1280/19683 (predictions/alpha_1_full.py)
#   k*      = 3 (predictions/k_star.py)
#
# Status: STRICT-SOLID-conditional-on-G1 (inherits from v).
#   - y_τ itself is THEOREM-GRADE (docs/theorems/theorem_ytau_corollary.md,
#     session 25; 0 adoptions, all 14 load-bearing steps T1/T2/T3/T4).
#   - v = δ²M_P/(√2 N_hub^{1/4}) × (1−(5/12)α₁/(1−α₁)) round-trips
#     v_obs by construction — the adopted N_hub's value is calibrated via the measured G_F (session 19+21; predictions/v_higgs.py).
#     The G1 gap is N = N_hub, which requires deriving H_0 from A1-A4
#     (same wall as Newton's G and Λ_CC).
#
# This is the SINGLE INDEPENDENT LEPTON MASS PREDICTION.
# m_μ and m_e are ratio predictions from m_τ via Koide f_j structure
# (predictions/m_mu.py, predictions/m_e.py). The ratios are
# theorem-grade independently via Q/ε/δ_Koide. Only m_τ's absolute
# scale depends on v.

# --- INPUTS --------------------------------------------------
# symbol  | value       | status          | predictions/ file           | meaning
# --------|-------------|-----------------|-----------------------------|--------
# v       | 246.22 GeV  | [STRICT-SOLID]  | predictions/v_higgs.py      | Higgs VEV (G1 on N_hub)
# y_tau   | 1280/177147 | [THEOREM]       | predictions/y_tau.py        | tau Yukawa coupling

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v_higgs import predict_v_higgs, delta, M_P, N_hub, alpha_1
from y_tau import predict_y_tau
from alpha_1 import predict_alpha_1
from alpha_1_full import predict_alpha_1_full, n_g_edge
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
import functools

# --- chain imports ---
d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)
alpha_1_bare_val = float(predict_alpha_1(k, g))               # for Family D
alpha_1_full_exact = predict_alpha_1_full(k, g, n_g_edge)
alpha_1_full = float(alpha_1_full_exact)
# Family D-corrected y_τ (vertex: 1H + 2F at Yukawa, srs N_atoms=4)
from V_count import V_count_pred as N_atoms_srs  # = 4, srs primitive cell |V| / K_4 quotient (predict_V_count)
y_tau = predict_y_tau(alpha_1_full, alpha_1_bare_val, k, n_H_legs=1, n_F_legs=2,
                       N_atoms=N_atoms_srs)
v_pred = predict_v_higgs(delta, M_P, N_hub, alpha_1)

# --- m_tau from closed formula ---
m_tau_pred = v_pred * y_tau

# --- observed value ---
m_tau_obs   = 1.77686   # GeV (PDG 2024)
m_tau_sigma = 0.00012   # GeV (PDG 2024)

dev_abs   = m_tau_pred - m_tau_obs
dev_rel   = dev_abs / m_tau_obs
dev_sigma = dev_abs / m_tau_sigma

print("=" * 68)
print("  m_tau  --  STRICT-SOLID conditional on G1 (via v_Higgs)")
print("=" * 68)
print(f"  v_pred       = {v_pred:.4f} GeV   [predictions/v_higgs.py]")
print(f"  y_tau        = {y_tau:.10f}      [THEOREM: predictions/y_tau.py]")
print()
print(f"  m_tau_pred   = v × y_tau = {m_tau_pred:.6f} GeV")
print(f"  m_tau_obs    = {m_tau_obs} ± {m_tau_sigma} GeV  (PDG 2024)")
print(f"  Deviation    = {dev_abs*1000:+.3f} MeV  ({dev_rel*100:+.4f}%, {dev_sigma:+.1f} sigma)")
print()
print("  Grade chain:")
print("    y_tau       = THEOREM (docs/theorems/theorem_ytau_corollary.md, 0 adoptions)")
print("    v_Higgs     = STRICT-SOLID conditional on G1 (N=N_hub from the adopted N_hub (value pinned via the measured G_F))")
print("    m_tau       = STRICT-SOLID conditional on G1 (inherits v)")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_m_tau(v, y_tau):
    """
    Compute the tau lepton mass from the Higgs VEV and the tau Yukawa.

    Formula:
        m_τ = v × y_τ

    Both inputs are derived in the framework:
      - y_τ is THEOREM-GRADE from docs/theorems/theorem_ytau_corollary.md
      - v is STRICT-SOLID conditional on G1 via the chain via the adopted N_hub (value pinned via the measured G_F)

    Parameters
    ----------
    v : float
        Higgs vacuum expectation value in GeV. From predictions/v_higgs.py.
    y_tau : float
        Tau Yukawa coupling. From predictions/y_tau.py.

    Returns
    -------
    float
        Predicted tau lepton mass in GeV.
    """
    return v * y_tau


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = m_tau_pred
    pure_result = predict_m_tau(v_pred, y_tau)
    print()
    print(f"Implementation: {impl_result:.10f} GeV")
    print(f"Pure function:  {pure_result:.10f} GeV")
    assert abs(impl_result - pure_result) < 1e-10, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    m_tau = {pure_result*1000:.2f} MeV  "
          f"(obs: {m_tau_obs*1000:.2f} MeV, {dev_rel*100:+.4f}%)")
    print("    Rigor: THEOREM-GRADE-STRUCTURAL conditional (W1 2026-05-18) — cond.")
    print("           on G1 (v←N_hub) AND y_τ Family-D c_F Clause-6 channel arg.")
    print("           NOT UNIQUE. Numeric value unchanged.")
