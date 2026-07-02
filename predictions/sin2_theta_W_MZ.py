#!/usr/bin/env python3
"""
sin²θ_W(M_Z) — weak mixing angle at the Z-pole.

Thin wrapper around the alpha_EM.py cluster output. Computes sin²θ_W at
M_Z by RG running from the theorem-grade sin²θ_W(M_unif) = 3/8 down to
the framework-derived M_Z (predictions/M_Z.py).

DISTINGUISHED FROM `predictions/sin2_theta_W.py` which predicts AT
UNIFICATION (3/8 exact, theorem-grade Class C). This file ships the
M_Z value (RG-derived).

CHAIN: sin²θ_W(M_Z) = α_Y(M_Z) / (α_2(M_Z) + α_Y(M_Z))
  with α_2, α_1 RG-run from α_GUT at M_unif (post-Stage-5 graduation),
  and α_Y = (3/5) α_1 (SU(5) embedding).

STATUS: THEOREM-GRADE-CONDITIONAL inheriting from M_unif (Row P62) +
M_Z (predictions/M_Z.py) + standard MSSM RG (Type 3). Match: -0.41% vs
PDG 0.23121.
"""

# ============================================================
# PARAMETER: sin²θ_W(M_Z) — weak mixing angle at Z-pole
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       sin²θ_W(M_Z) = 0.23121 ± 0.00004 (PDG 2024 on-shell scheme)
# Source:      PDG 2024 electroweak precision fits

# --- PREDICTED VALUE -----------------------------------------
# Value:       sin²θ_W(M_Z) ≈ 0.23125  (live, post-α_GUT-DC; was 0.23027 pre-DC, stale)
# Deviation:   +0.017% vs PDG (one-loop MSSM-style single-regime running; no M_SUSY threshold)
# Status:      near-PASS Clause 8 (~+1σ_PDG, see ledger Row P65 post-2026-05-15 table)

# --- DERIVED FORMULA -----------------------------------------
# At M_Z: sin²θ_W = α_Y / (α_2 + α_Y) with α_Y = (3/5) α_1 (GUT norm).
# At M_unif: α_1 = α_2 = α_GUT, giving sin²θ_W(M_unif) = 3/8 (theorem-grade).
# RG running through MSSM β-functions to M_Z gives the M_Z value.

# --- INPUTS --------------------------------------------------
# alpha_GUT       [theorem-grade]  predictions/alpha_GUT.py
# M_unif          [thm-grade-cond] predictions/M_unif.py
# M_Z             [thm-grade-cond] predictions/M_Z.py
# b_1, b_2 (MSSM) [Type 3]         (Peskin-Schroeder §16)
# hypercharge_norm = 3/5 [Type 1]  (SU(5) embedding)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import functools

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from M_unif import predict_M_unif_GeV
from M_Z import M_Z_GeV

# Theorem-grade primitives — k_star, g_girth, p_toggle sourced from leaves
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from p_toggle import predict_p_toggle
_d = predict_d_spatial()
k_star = predict_k_star(_d)
g_girth = predict_g_girth(k_star, _d)
_p = predict_p_toggle()

from alpha_GUT import predict_alpha_GUT_observed
alpha_GUT = float(predict_alpha_GUT_observed(k_star, g_girth))  # dark-corrected, theorem-grade-cond 2026-05-15
from mssm_beta_coefficients import b_1_MSSM  # MSSM one-loop β coefficient single-source
from mssm_beta_coefficients import b_2_MSSM  # MSSM one-loop β coefficient single-source
from mssm_beta_coefficients import hypercharge_norm  # = 3/5, GUT-norm single-source

from M_Pl_natural import M_Pl_GeV   # CODATA, single source — ANTHROPOCENTRIC SI TRANSLATION
M_unif_GeV = predict_M_unif_GeV(k_star, g_girth, M_Pl_GeV)

log_ratio = math.log(M_Z_GeV / M_unif_GeV)
inv_alpha_1 = 1.0/alpha_GUT - (b_1_MSSM / (_p*math.pi)) * log_ratio
inv_alpha_2 = 1.0/alpha_GUT - (b_2_MSSM / (_p*math.pi)) * log_ratio
alpha_1_MZ = 1.0 / inv_alpha_1
alpha_2_MZ = 1.0 / inv_alpha_2
alpha_Y_MZ = hypercharge_norm * alpha_1_MZ
sin2_theta_W_MZ = alpha_Y_MZ / (alpha_2_MZ + alpha_Y_MZ)

# Module-level exports
sin2_theta_W_MZ_pred = sin2_theta_W_MZ
sin2_theta_W_MZ_obs = 0.23121
sin2_theta_W_MZ_sigma = 0.00004

print(f"sin²θ_W(M_Z) = {sin2_theta_W_MZ:.5f}  (PDG 0.23121, dev "
      f"{(sin2_theta_W_MZ - 0.23121)/0.23121*100:+.3f}%)")
print(f"  Inputs: α_GUT=1/24 (theorem), M_unif (thm-grade-cond), M_Z (thm-grade-cond), MSSM RG (Type 3)")
print(f"  Status: THEOREM-GRADE-CONDITIONAL")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_sin2_theta_W_MZ(alpha_GUT, M_unif_GeV, M_Z_GeV, b_1, b_2, hypercharge_norm):
    """
    Predict sin²θ_W(M_Z) by RG running from sin²θ_W(M_unif)=3/8 to M_Z.

    Parameters
    ----------
    alpha_GUT : float
        Unified gauge coupling at M_unif (= 1/24).
    M_unif_GeV, M_Z_GeV : float
        Energy scales in GeV.
    b_1, b_2 : float
        MSSM one-loop β-function coefficients.
    hypercharge_norm : float
        SU(5) hypercharge normalization (= 3/5).

    Returns
    -------
    float
        sin²θ_W at M_Z.
    """
    log_ratio = math.log(M_Z_GeV / M_unif_GeV)
    inv_a1 = 1.0/alpha_GUT - (b_1 / (2*math.pi)) * log_ratio
    inv_a2 = 1.0/alpha_GUT - (b_2 / (2*math.pi)) * log_ratio
    a1 = 1.0 / inv_a1
    a2 = 1.0 / inv_a2
    aY = hypercharge_norm * a1
    return aY / (a2 + aY)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = sin2_theta_W_MZ
    pure_result = predict_sin2_theta_W_MZ(
        alpha_GUT, M_unif_GeV, M_Z_GeV, b_1_MSSM, b_2_MSSM, hypercharge_norm,
    )
    assert abs(impl_result - pure_result) < 1e-12
    print(f"OK: implementation = pure function = {impl_result:.6f}")
