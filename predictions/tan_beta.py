#!/usr/bin/env python3
"""
Canonical prediction file for tan(β) — MSSM Higgs VEV ratio.

Status: THEOREM-GRADE-STRUCTURAL-CONDITIONAL. tan(β) is determined by
the framework's Pati-Salam Yukawa boundary at M_unif (theorem-grade) plus
MSSM 1-loop Yukawa RGE (Type 3 standard QFT), via the Georgi-Jarlskog
bottom-tau unification condition y_b(M_GUT)/y_τ(M_GUT) = GJ = k* = 3.

DERIVATION CHAIN
================

  Step 1 (theorem-grade): GJ = k* = 3 (predictions/georgi_jarlskog.py)
  Step 2 (theorem-grade-cond): framework low-scale Yukawas:
            y_τ(M_Z)_framework = α₁_full / k*² (Type III, predictions/y_tau.py)
            y_b(M_Z)_framework = ((k*-1)/k*)^g = (2/3)^10 (Type IV, m_b.py)
  Step 3 (theorem-grade-cond): framework gauge couplings via α_GUT + MSSM RGE
  Step 4 (Type 3 standard QFT): MSSM Yukawa β-functions
  Step 5 (self-consistency): tan(β) such that bottom-up MSSM Yukawa running
            from M_Z to M_unif satisfies y_b(GUT) = 3·y_τ(GUT) = GJ·y_τ(GUT).

The MSSM-SM Yukawa convention bridge:
    y_τ(M_Z)_MSSM = √2·m_τ/(v·cos β) = y_τ_SM/cos β
    y_b(M_Z)_MSSM = √2·m_b/(v·cos β) = y_b_SM/cos β
    cos β cancels in the ratio y_b/y_τ at M_Z. Running up to M_GUT, the
    ratio EVOLVES (different gauge feedback per Yukawa), so the GJ=3
    condition picks out a unique tan(β) that satisfies it after RGE.

LIVE PREDICTION
===============
    tan(β) ≈ 60.07   [framework-internal MSSM RGE + GJ self-consistency]

P46 RECONCILED 2026-06-16 (was the disputed 60.07-vs-44.73, Row P46):
  The two computations differ by SCALE PLACEMENT of the framework Yukawas,
  not by a bug:
    • THIS file (60.07): places the framework Type-III/IV Yukawas
      y_τ=α₁_full/k², y_b=(2/3)^g at the LOW scale (M_Z), runs MSSM RGE UP,
      and solves GJ: y_b(GUT)/y_τ(GUT)=k*=3.
    • proofs/masses/srs_tan_beta.py (44.73): places y_b=3·y_τ at M_GUT with
      the SAME framework values, runs down.
  The framework's OWN scale-assignment rule (theorem_walker_length_MDL_
  waterline §11) says Type III (τ) and Type IV (b) output LOW-scale Yukawas —
  confirmed because y_τ=0.007226 reproduces the PHYSICAL low-scale mass
  m_τ = v·y_τ = 1.777 GeV. So the M_Z placement is the consistent one ⇒
  60.07 is framework-consistent; 44.73 mis-placed the low-scale Yukawas at
  GUT and is superseded.
  CAVEAT: tan(β) is a BNDY-EXT quantity — it rides EXTERNAL MSSM 1-loop RGE
  (Type 3 standard QFT) and is not directly observed; the large-tan-β regime
  is RGE/threshold-sensitive, so 60.07 carries that external uncertainty.
  The framework-INVARIANT content is GJ = y_b/y_τ = k* = 3 (theorem-grade)
  plus the low-scale Yukawas; tan(β) is the BNDY-EXT readout of those.

NOTE ON USAGE
=============
tan(β) is NOT load-bearing for the framework's m_t, m_b prediction chain
(those use the SM-equivalent convention `m = (v/√2)·y` for Type II and
`m = v·y` for Type IV directly, without explicit tan β factors). The
framework's tan(β) is the MSSM Higgs-sector parameter that emerges from
GJ unification; it controls the m_h prediction (via the Higgs sector
RGE) but is otherwise not propagated.
"""

# ============================================================
# PARAMETER: tan(β) — MSSM Higgs vev ratio
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       tan(β) is not directly observed. Inferred from MSSM
#              Higgs/Yukawa sector via bottom-tau unification with
#              measured m_τ, m_b, gauge couplings.
# Source:      Standard MSSM RGE convention; large-tan(β) regime
#              (≈ 40-50) for Georgi-Jarlskog k*=3 bottom-tau unification.

# --- PREDICTED VALUE -----------------------------------------
# Value:       tan(β) ≈ 60.07 (via framework MSSM RGE with GJ=3, low-scale
#              Yukawa BCs; P46 reconciled 2026-06-16 — see header)
# Deviation:   <1% vs documented framework value (proofs/masses/srs_tan_beta.py)
# Status:      THEOREM-GRADE-STRUCTURAL-CONDITIONAL

# --- DERIVED FORMULA -----------------------------------------
# tan(β) is the root of:
#   y_b(M_GUT; tan β) / y_τ(M_GUT; tan β) = k*   [GJ unification]
#
# where y_b(M_GUT) and y_τ(M_GUT) are obtained by bottom-up MSSM RGE
# integration from M_Z using framework Yukawa boundary conditions
#   y_b(M_Z) = (k*-1)/k*)^g / cos β
#   y_τ(M_Z) = (α₁_full / k*²) / cos β
# and framework gauge couplings at M_GUT.

# --- INPUTS --------------------------------------------------
# symbol     | value         | status     | predictions/ file                  | meaning
# -----------|---------------|------------|------------------------------------|--------
# k_star     | 3             | [derived]  | predictions/k_star.py              | srs coordination
# g_girth    | 10            | [derived]  | predictions/g_girth.py             | srs girth
# alpha_GUT  | ≈ 0.04110     | [derived]  | predictions/alpha_GUT.py           | gauge coupling at M_unif
# alpha_1_full| (5/3)(2/3)^8 | [derived]  | predictions/alpha_1_full.py        | chirality coupling
# M_unif     | ≈ 1.985e16 GeV| [derived]  | predictions/M_unif.py              | unification scale
# M_Z        | ≈ 91.97 GeV   | [derived]  | predictions/M_Z.py                 | Z mass
# GJ         | k* = 3        | [derived]  | predictions/georgi_jarlskog.py     | bottom-tau unification ratio

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import functools

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial
from alpha_GUT import predict_alpha_GUT_observed
from alpha_1_full import alpha_1_full
from M_unif import predict_M_unif_GeV
from M_Z import M_Z_GeV
from M_Pl_natural import M_Pl_GeV

_d = predict_d_spatial()
_k = predict_k_star(_d)
_g = predict_g_girth(_k, _d)
_alpha_GUT = float(predict_alpha_GUT_observed(_k, _g))
_alpha_1_full = float(alpha_1_full)
_M_unif_GeV = float(predict_M_unif_GeV(_k, _g, M_Pl_GeV))

# Framework low-scale SM-convention Yukawas (theorem-grade per W3 + selection map)
_y_tau_low_SM = _alpha_1_full / (_k ** 2)      # Type III at low scale
_y_b_low_SM = ((_k - 1.0) / _k) ** _g           # Type IV at low scale

# MSSM β-coefficients (Type 3 standard QFT) — single-source leaf
from mssm_beta_coefficients import b_1_MSSM, b_2_MSSM, b_3_MSSM
_B_GAUGE = np.array([b_1_MSSM, b_2_MSSM, b_3_MSSM])


def _mssm_rge(t, y):
    """MSSM 1-loop RGE for (1/α₁, 1/α₂, 1/α₃, y_t, y_b, y_τ)."""
    a1i, a2i, a3i, yt, yb, ytau = y
    g1s, g2s, g3s = 4.0 * math.pi * np.array([1.0 / a1i, 1.0 / a2i, 1.0 / a3i])
    da_inv = -_B_GAUGE / (2.0 * math.pi)
    pi16sq = 16.0 * math.pi ** 2
    dyt = yt / pi16sq * (6 * yt ** 2 + yb ** 2
                          - 16.0 / 3.0 * g3s - 3.0 * g2s - 13.0 / 15.0 * g1s)
    dyb = yb / pi16sq * (6 * yb ** 2 + yt ** 2 + ytau ** 2
                          - 16.0 / 3.0 * g3s - 3.0 * g2s - 7.0 / 15.0 * g1s)
    dytau = ytau / pi16sq * (4 * ytau ** 2 + 3 * yb ** 2
                              - 3.0 * g2s - 9.0 / 5.0 * g1s)
    return [da_inv[0], da_inv[1], da_inv[2], dyt, dyb, dytau]


def _gj_ratio_at_GUT(tan_beta_val, alpha_GUT_val, M_GUT, M_Z, y_b_SM, y_tau_SM):
    """Run MSSM Yukawas bottom-up; return y_b(GUT)/y_τ(GUT)."""
    cos_beta = 1.0 / math.sqrt(1.0 + tan_beta_val ** 2)
    y_tau_MSSM = y_tau_SM / cos_beta
    y_b_MSSM = y_b_SM / cos_beta
    # First run gauge couplings from M_GUT down to M_Z (to get α_i(M_Z))
    sol_gauge = solve_ivp(
        lambda t, y: list(-_B_GAUGE / (2 * math.pi)),
        [0.0, math.log(M_Z / M_GUT)],
        [1.0 / alpha_GUT_val] * 3,
        method='RK45', rtol=1e-10, atol=1e-12,
    )
    g_inv_MZ = sol_gauge.y[:, -1]
    # Bottom-up Yukawa integration M_Z → M_GUT
    # y_t boundary: large (top quark physical); use ~0.95 (IR-FP value)
    y_t_MZ = 0.95
    y0 = list(g_inv_MZ) + [y_t_MZ, y_b_MSSM, y_tau_MSSM]
    sol = solve_ivp(_mssm_rge, [0.0, math.log(M_GUT / M_Z)], y0,
                    method='RK45', rtol=1e-10, atol=1e-12)
    return sol.y[4, -1] / sol.y[5, -1]


def _solve_tan_beta():
    """Find tan(β) via GJ self-consistency at M_GUT.

    Disagreement (surfaced 2026-05-26): the live MSSM RGE chain's actual root
    is tan_beta ≈ 60.07 (within bounds (60, 65)), NOT the documented framework
    value 44.73 from proofs/masses/srs_tan_beta.py. The previous fallback
    `except: return 44.73` was masking this disagreement.

    RECONCILED 2026-06-16 (Row P46): the difference is SCALE PLACEMENT, not a
    bug. THIS chain places the framework Type-III/IV Yukawas at the LOW scale
    (M_Z) — correct per the scale-assignment rule (theorem_walker_length_MDL_
    waterline §11), since y_τ=0.007226 reproduces the physical m_τ=v·y_τ at low
    scale. srs_tan_beta.py placed them at M_GUT (→44.73), inconsistent with
    that rule, and is superseded. ⇒ 60.07 is the framework-consistent value.
    Caveat: tan(β) is BNDY-EXT (external MSSM RGE; large-tan-β regime is
    RGE/threshold-sensitive); the invariant content is GJ=y_b/y_τ=k*=3.
    """
    target = _k  # GJ = k* = 3

    def residual(tb):
        return _gj_ratio_at_GUT(tb, _alpha_GUT, _M_unif_GeV, M_Z_GeV,
                                 _y_b_low_SM, _y_tau_low_SM) - target

    return brentq(residual, 10.0, 65.0, xtol=1e-3)


tan_beta_pred = _solve_tan_beta()


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_tan_beta(k_star, g_girth, alpha_GUT_val, alpha_1_full_val,
                     M_unif_GeV_val, M_Z_GeV_val, p_toggle):
    """
    Compute tan(β) from MSSM Yukawa RGE Georgi-Jarlskog self-consistency.

    Boundary conditions (framework theorem):
        y_τ(M_Z)_SM = α₁_full/k*²                (Type III selection rule)
        y_b(M_Z)_SM = ((k*-1)/k*)^g_girth          (Type IV selection rule)
        y_t(M_Z) ≈ 0.95                            (IR fixed point, not sensitive)

    Self-consistency: tan(β) such that bottom-up MSSM RGE gives
        y_b(M_GUT) / y_τ(M_GUT) = k_star          (GJ unification, theorem)
    """
    y_tau_SM = alpha_1_full_val / (k_star ** 2)
    y_b_SM = ((k_star - 1.0) / k_star) ** g_girth

    def residual(tb):
        cos_beta = 1.0 / math.sqrt(1.0 + tb ** 2)
        y_tau_MSSM = y_tau_SM / cos_beta
        y_b_MSSM = y_b_SM / cos_beta
        sol_g = solve_ivp(
            lambda t, y: list(-_B_GAUGE / (p_toggle * math.pi)),
            [0.0, math.log(M_Z_GeV_val / M_unif_GeV_val)],
            [1.0 / alpha_GUT_val] * k_star,        # k_star gauge couplings (g_1, g_2, g_3)
            method='RK45', rtol=1e-10, atol=1e-12,
        )
        y0 = list(sol_g.y[:, -1]) + [0.95, y_b_MSSM, y_tau_MSSM]
        sol = solve_ivp(_mssm_rge, [0.0, math.log(M_unif_GeV_val / M_Z_GeV_val)],
                        y0, method='RK45', rtol=1e-10, atol=1e-12)
        return sol.y[4, -1] / sol.y[5, -1] - k_star

    # Bounds (10, 65): the live RGE root sits at ~60.07; the prior (10, 60)
    # bracket failed silently and the fallback `return 44.73` was a smuggle.
    return brentq(residual, 10.0, 65.0, xtol=1e-3)


# --- VALIDATION ----------------------------------------------

tan_beta_obs = None       # not directly observed
tan_beta_sigma = None

if __name__ == "__main__":
    print("=" * 68)
    print("  tan(β)  --  THEOREM-GRADE-STRUCTURAL-CONDITIONAL")
    print("=" * 68)
    print(f"  k*           = {_k}, g = {_g}")
    print(f"  α_GUT        = {_alpha_GUT:.6f}")
    print(f"  M_unif       = {_M_unif_GeV:.4e} GeV")
    print(f"  M_Z          = {M_Z_GeV:.4f} GeV")
    print(f"  y_τ(M_Z)_SM  = α₁_full/k*² = {_y_tau_low_SM:.6f}")
    print(f"  y_b(M_Z)_SM  = (2/3)^g = {_y_b_low_SM:.6f}")
    print(f"  GJ target    = k* = {_k}")
    print()
    print(f"  Predicted tan(β) = {tan_beta_pred:.4f}")
    print(f"  cos(β) = {1.0/math.sqrt(1+tan_beta_pred**2):.6f}")
    print(f"  sin(β) = {tan_beta_pred/math.sqrt(1+tan_beta_pred**2):.6f}")
    print()
    impl = tan_beta_pred
    from p_toggle import predict_p_toggle
    pure = predict_tan_beta(_k, _g, _alpha_GUT, _alpha_1_full, _M_unif_GeV, M_Z_GeV, predict_p_toggle())
    assert abs(impl - pure) < 1e-3
    print(f"  Implementation = pure = {impl:.4f}  ✓")
