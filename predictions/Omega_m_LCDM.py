#!/usr/bin/env python3
"""
Omega_m_LCDM — ΛCDM-fit total matter fraction, downstream of the ADOPTED
cosmology parameter z_eff (predictions/z_eff.py, N_hub-pattern).

Theorem-grade bias-function FORM Ω_m(z)=(u+1)/(u²+u+1), u=1+z (derived
from H_coast²=H_LCDM²; K-rational; no fitting), evaluated at the ADOPTED
z_eff. ONE adopted number replaces ΛCDM's free Ω_m. Grade:
MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff (same epistemic
class as H_0/t_0 on adopted N_hub).
"""

# ============================================================
# PARAMETER: Omega_m_LCDM
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       0.3153 +/- 0.0073   (Planck 2018, Aghanim+ 2020)
# Source:      Planck 2018 cosmological parameters
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       Om_bias(z_eff_adopted) ; z_eff~1.85 -> Om ~ 0.322
# Deviation:   ~ +0.9 sigma_obs (first-moment z_eff)
#              definitional band (bias-inverted z_eff) -> ~+3 sigma_obs

# --- DERIVED FORMULA -----------------------------------------
# Omega_m_LCDM = (u+1)/(u^2+u+1),  u = 1 + z_eff
# z_eff: predictions/z_eff.py (ADOPTED, N_hub-class).

# --- INPUTS --------------------------------------------------
# symbol | value      | status                 | predictions/ file
# -------|------------|------------------------|------------------
# z_eff  | ~1.85      | [adopted, N_hub-class] | predictions/z_eff.py

# --- IMPLEMENTATION ------------------------------------------

import functools
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from z_eff import predict_z_eff, BAO_ANCHORS, SN_MODEL

Z_EFF_K_RATIONAL = math.sqrt(3.0)  # exact-halving anchor (Om=1/3 exactly)


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_Omega_m_LCDM(z_eff):
    """
    ΛCDM-fit total matter fraction at the adopted effective redshift z_eff.

    Omega_m_LCDM = (u+1)/(u^2+u+1), u = 1 + z_eff  (theorem-grade form,
    derived from setting H_coast^2 = H_LCDM^2; K-rational; no fitting).

    Parameters
    ----------
    z_eff : float
        Adopted cosmology effective redshift (predictions/z_eff.py).

    Returns
    -------
    float
        Omega_m_LCDM (dimensionless).
    """
    u = 1.0 + z_eff
    return (u + 1.0) / (u * u + u + 1.0)


# --- INTROSPECTION (for run_predictions.py) ------------------
_z_eff = predict_z_eff(BAO_ANCHORS, SN_MODEL)
Omega_m_LCDM_pred = predict_Omega_m_LCDM(_z_eff)
Omega_m_LCDM_obs = 0.3153
Omega_m_LCDM_sigma = 0.0073


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    z_eff = _z_eff
    val = Omega_m_LCDM_pred
    anchor = predict_Omega_m_LCDM(Z_EFF_K_RATIONAL)
    obs, sig = Omega_m_LCDM_obs, Omega_m_LCDM_sigma
    print("=" * 72)
    print(" Omega_m_LCDM (downstream of ADOPTED z_eff)")
    print("=" * 72)
    print(f"  z_eff (adopted)            = {z_eff:.4f}")
    print(f"  Omega_m_LCDM @ z_eff       = {val:.4f}")
    print(f"  @ z=0 (substrate)          = {predict_Omega_m_LCDM(0.0):.4f} (=2/3)")
    print(f"  @ z=sqrt(3) (K-rational)   = {anchor:.4f} (=1/3 exactly)")
    print(f"  Planck 2018                = {obs} +/- {sig}")
    print(f"  deviation                  = {(val-obs)/sig:+.1f} sigma_obs")
    assert abs(anchor - 1.0/3.0) < 1e-12
    assert abs(predict_Omega_m_LCDM(0.0) - 2.0/3.0) < 1e-12
    print("  OK (K-rational anchor 1/3 exact; z=0 -> 2/3 exact)")
