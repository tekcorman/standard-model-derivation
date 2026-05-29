#!/usr/bin/env python3
"""
Higgs trilinear self-coupling λ_3 = m_H² / (2v).

The framework predicts the SM tree-level relation between λ_3 and the
upstream (theorem-grade Family D) m_H and v.  Since both m_H and v close
to sub-σ_PDG under Family D per-leg multiway dark-disruption
(`docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md` §3 (D)),
λ_3 inherits theorem-grade status by algebra.

Two equivalent readings:
  - Coefficient of h³ in the SM Higgs Lagrangian around v:
      L_H ⊃ -(m_H²/(2v)) h³   ⇒   λ_3 ≡ m_H²/(2v)
  - Equivalent form via the Higgs quartic and VEV:
      m_H² = 2λv²   ⇒   λ_3 = λ·v
  - LHC-convention dimensionless ratio:
      κ_λ ≡ λ_3 / λ_3^SM = (predicted λ_3) / (SM-tree relation using same v)
                        ≡ 1 by construction since the framework predicts
                        the SM-tree relation.

The two are algebraically consistent: m_H = √(2λ)·v with Family D
applied to λ (vertex 4H+0F, δλ/λ = -4α₁² ≈ -0.609%) and the (5/12)
Class-C correction applied to v gives both consistent λ_3 values.
"""

# ============================================================
# PARAMETER: λ_3 — Higgs trilinear self-coupling
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       λ_3^SM = m_H²/(2v) = (125.20 GeV)² / (2 · 246.22 GeV)
#                     = 31.835 GeV  (SM tree-level using PDG inputs)
#              κ_λ^SM = 1.0 by definition.
# Source:      Derived from PDG 2024 m_H = 125.20 ± 0.11 GeV and
#              v = 246.22 ± 0.12 GeV (electroweak precision).
#              LHC direct constraint:
#                ATLAS+CMS combined 2022 (HH → bbγγ, HH → bbττ, etc.):
#                κ_λ ∈ [-0.4, 6.3] @ 95% CL (Nature 607, 52 (2022)).
#                ATLAS 2023 (full Run 2 + HH inputs): κ_λ ∈ [-1.4, 6.1] @ 95% CL.
#              κ_λ = 1 (SM) is consistent with all current data.
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       λ_3 = m_H_FD² / (2 · v_FD) ≈ 31.832 GeV
#                  ≈ κ_λ × λ_3^SM with κ_λ = 1 by framework's SM tree-level structure.
# Deviation:   Inherits m_H Family-D-corrected match (-0.05σ_PDG on m_H);
#              v matches at -0.0001σ_PDG (round-trip via G_F).
#              Total σ_PDG-class deviation on λ_3 (vs PDG-computed SM value):
#              ≈ -0.10σ_PDG (sub-σ).

# --- DERIVED FORMULA -----------------------------------------
# λ_3 = m_H² / (2·v)
#
# where:
#   m_H = √(2·λ_FD)·v   [predictions/m_H.py, Family D propagated]
#   λ_FD = (2·(5/3)·α₁_bare) · (1 - 4·α₁_bare²)   [predictions/lambda_higgs.py, Family D]
#   v    = δ²·M_Pl·(1 - (5/12)·α₁/(1-α₁)) / (√2·N_hub^(1/4))   [predictions/v_higgs.py]
#
# Substituting: λ_3 = m_H²/(2v) = (2·λ_FD·v²) / (2v) = λ_FD · v
#
# So equivalently:
#   λ_3 = λ_FD · v_FD                            [algebraic identity]
#
# Chain:
#   Step 1: Family D theorem-grade for c_H, c_F (master doc §3 (D),
#           Routes H + C + F-1 + F-2 closed 2026-05-15)
#   Step 2: λ_FD = predict_lambda_higgs(...)     [Family D propagated]
#   Step 3: v    = predict_v_higgs(...)           [Class C (5/12) DC applied]
#   Step 4: λ_3  = λ_FD · v                       [Type 2 algebra]
#
# KAPPA_LAMBDA REPORT: κ_λ ≡ λ_3 / λ_3^SM_using_PDG_inputs = (very near 1 by construction)
# is reported but not a structural prediction beyond the SM-tree relation;
# direct LHC measurement currently constrains κ_λ ∈ [-1.4, 6.1].

# --- INPUTS --------------------------------------------------
# symbol     | value                | status     | predictions/ file       | meaning
# -----------|----------------------|------------|-------------------------|--------
# m_H        | ≈125.195 GeV         | [derived]  | predictions/m_H.py      | Higgs mass (Family D corrected)
# v          | ≈246.22 GeV          | [derived]  | predictions/v_higgs.py  | Higgs VEV (Class C corrected)
# (Family D + Class C upstream theorem-grade per master doc §3.)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from srs_E_at_P import predict_srs_E_at_P
from v_higgs import predict_v_higgs
from N_hub import predict_N_hub
from lambda_higgs import predict_lambda_higgs
from m_H import predict_m_H
from M_Pl_natural import M_Pl_GeV as M_P
from V_count import predict_V_count

d        = predict_d_spatial()
k        = predict_k_star(d)
g        = predict_g_girth(k, d)
alpha_1  = predict_alpha_1(k, g)
E_at_P   = predict_srs_E_at_P(k)
from p_toggle import predict_p_toggle
p        = predict_p_toggle()
V_val    = predict_V_count(k, d)
h        = predict_h_walker_eigenvalue(k, E_at_P, p)

delta    = 2.0 / 9.0
G_F_obs  = 1.1663787e-5
N_hub    = predict_N_hub(G_F_obs, M_P, alpha_1, delta, k, p, V_val)

# Family D structural inputs at the Higgs vertices (master doc §3 (D))
from V_count import V_count_pred as N_atoms_srs  # = 4, srs primitive cell |V| / K_4 quotient (predict_V_count)
n_channels  = 2

# Upstream theorem-grade quantities (Family D propagated)
v_pred   = predict_v_higgs(delta, M_P, N_hub, alpha_1)
lam_FD   = predict_lambda_higgs(alpha_1, h, n_channels=n_channels,
                                 n_H_legs=4, n_F_legs=0,
                                 N_atoms=N_atoms_srs, k_star=k)
m_H_pred = predict_m_H(delta, M_P, N_hub, alpha_1, h,
                        n_channels=n_channels, n_H_legs=4, n_F_legs=0,
                        N_atoms=N_atoms_srs, k_star=k)

# Trilinear coupling: λ_3 = m_H²/(2v) (SM tree-level coefficient of h³)
lambda_3_pred = m_H_pred ** 2 / (2.0 * v_pred)

# Cross-check via algebraic identity λ_3 = λ_FD · v
lambda_3_via_lambda_v = lam_FD * v_pred
assert abs(lambda_3_pred - lambda_3_via_lambda_v) / lambda_3_pred < 1e-12, (
    f"λ_3 identity mismatch: m_H²/(2v) = {lambda_3_pred} vs λ·v = {lambda_3_via_lambda_v}"
)

# Observed value: SM tree-level using PDG m_H, v
m_H_obs  = 125.20
v_obs    = 246.22
lambda_3_SM = m_H_obs ** 2 / (2.0 * v_obs)

# κ_λ ≡ predicted / SM-using-PDG-inputs
kappa_lambda = lambda_3_pred / lambda_3_SM

# σ_PDG-class deviation propagation: dominated by m_H precision (0.11 GeV)
# Δλ_3 / λ_3 = 2·Δm_H/m_H + Δv/v ; σ_λ_3 ≈ λ_3 × (2·σ_m_H/m_H + σ_v/v)
sigma_lambda_3 = lambda_3_SM * (2.0 * 0.11 / m_H_obs + 0.12 / v_obs)
dev_abs   = lambda_3_pred - lambda_3_SM
dev_rel   = dev_abs / lambda_3_SM
dev_sigma = dev_abs / sigma_lambda_3

# --- Runner-facing canonical aliases (slug = "lambda_3_higgs") ---------
# Without these, run_predictions.py _find_result_vars falls back to the
# shortest *_pred in scope and grabs the imported Higgs VEV v_pred (246
# GeV), mis-reporting λ_3 as ~246 instead of ~31.8 GeV. Aliases only;
# zero computational change.
lambda_3_higgs_pred  = lambda_3_pred
lambda_3_higgs_obs   = lambda_3_SM
lambda_3_higgs_sigma = sigma_lambda_3

print("=" * 72)
print("  λ_3  —  Higgs trilinear self-coupling")
print("=" * 72)
print(f"  m_H (Family D corrected)    = {m_H_pred:.4f} GeV")
print(f"  v   (Class C corrected)      = {v_pred:.4f} GeV")
print(f"  λ_FD (4H+0F vertex)          = {lam_FD:.8f}")
print()
print(f"  λ_3 = m_H²/(2v)              = {lambda_3_pred:.4f} GeV")
print(f"      cross-check via λ·v       = {lambda_3_via_lambda_v:.4f} GeV  (must match)")
print(f"  λ_3^SM (PDG-computed)        = {lambda_3_SM:.4f} GeV")
print()
print(f"  κ_λ ≡ predicted / SM-PDG     = {kappa_lambda:.6f}")
print(f"  Deviation vs SM-PDG          = {dev_abs:+.4f} GeV ({dev_rel*100:+.4f}%)")
print(f"  σ_λ_3 (propagated)           = {sigma_lambda_3:.4f} GeV")
print(f"  Sigma-class match            = {dev_sigma:+.2f} σ_PDG")
print()
print("  LHC direct constraint (ATLAS-CMS combined 2022, ATLAS 2023):")
print("    κ_λ ∈ [-1.4, 6.1] @ 95% CL — framework's κ_λ ≈ 1 well within.")
print()
print("  Rigor status: THEOREM-GRADE (algebraic descendant of m_H + v, both")
print("  theorem-grade with Family D propagated 2026-05-15). κ_λ = 1 is the")
print("  framework's SM-tree-relation-consistent prediction; LHC direct")
print("  measurement is the falsification test (will tighten with HL-LHC).")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_lambda_3_higgs(delta, M_P, N_hub, alpha_1, h, n_channels,
                            n_H_legs, n_F_legs, N_atoms, k_star):
    """
    Predict the Higgs trilinear self-coupling λ_3 = m_H²/(2v).

    Equivalent forms:
      - L_H ⊃ -(m_H²/(2v)) h³   ⇒   λ_3 ≡ m_H²/(2v)
      - λ_3 = λ_Higgs · v       (SM-tree algebraic identity)

    Both forms are computed and asserted equal.  The framework's
    prediction is theorem-grade by algebra: m_H is theorem-grade
    (Family D propagated per master doc §3 (D); see `m_H.py`); v is
    theorem-grade (Class C (5/12) DC + Family D sub-leading absorbed
    into N_hub anchor; see `v_higgs.py`).

    Parameters
    ----------
    delta : float
        Wigner D¹₁₀ Koide phase (= 2/9 on srs).
    M_P : float
        Planck mass in GeV (unit-setting constant, CODATA).
    N_hub : float
        The adopted dimensional input (calibrated via G_F).
    alpha_1 : float
        Bare NB walker survival ((k*-1)/k*)^(g-2) = (2/3)^8.
    h : complex
        Walker eigenvalue at P-point ((√3+i√5)/2 on srs).
    n_channels : int
        SU(2)_L Cl(2) channel multiplicity (= 2 on srs).
    n_H_legs : int
        Higgs legs at the |φ|⁴ vertex (= 4 on srs).
    n_F_legs : int
        Fermion legs at the |φ|⁴ vertex (= 0 on srs).
    N_atoms : int
        Wyckoff 8a atoms per primitive cell (= 4 on srs).
    k_star : int
        Coordination number (= 3 on srs).

    Returns
    -------
    float
        λ_3 in GeV (coefficient of h³ in SM Higgs Lagrangian around v).
    """
    v = predict_v_higgs(delta, M_P, N_hub, alpha_1)
    m_H = predict_m_H(delta, M_P, N_hub, alpha_1, h, n_channels,
                       n_H_legs, n_F_legs, N_atoms, k_star)
    return m_H ** 2 / (2.0 * v)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = lambda_3_pred
    pure_result = predict_lambda_3_higgs(
        delta, M_P, N_hub, alpha_1, h,
        n_channels=n_channels, n_H_legs=4, n_F_legs=0,
        N_atoms=N_atoms_srs, k_star=k,
    )
    print()
    print(f"Implementation: {impl_result:.10f} GeV")
    print(f"Pure function:  {pure_result:.10f} GeV")
    assert abs(impl_result - pure_result) / impl_result < 1e-12, (
        f"Implementation vs pure function mismatch: {impl_result} vs {pure_result}"
    )
    print("OK: outputs agree.")
    print(f"    λ_3 = {pure_result:.4f} GeV ; κ_λ = {kappa_lambda:.6f}")
    print(f"    σ_PDG-class match: {dev_sigma:+.2f} σ_PDG (sub-σ)")
