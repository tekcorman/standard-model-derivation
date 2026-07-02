#!/usr/bin/env python3
"""
Omega_Lambda_LCDM — ΛCDM-fit dark-energy fraction. Type-4 inheritance:
= 1 - Omega_m_LCDM (flat-ΛCDM). Downstream of the ADOPTED z_eff.

The Λ_CC factor-of-2 of Row P24 is structurally Omega_Lambda_LCDM(z_eff) /
Omega_Lambda_substrate = Omega_Lambda_LCDM / (1/3); = 2 EXACTLY at the
K-rational anchor z=sqrt(3). Grade: MATHEMATICALLY-COMPLETE-CONDITIONAL-
ON-ADOPTED-z_eff (inherits Omega_m_LCDM).
"""

# ============================================================
# PARAMETER: Omega_Lambda_LCDM
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       0.6847 +/- 0.0073  (Planck 2018; = 1 - Omega_m_LCDM)
# Source:      Planck 2018
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       1 - Om_bias(z_eff_adopted) ; ~ 0.678 at z_eff~1.85
# Deviation:   ~ -0.9 sigma_obs (first-moment z_eff)

# --- DERIVED FORMULA -----------------------------------------
# Omega_Lambda_LCDM = 1 - Omega_m_LCDM = u^2/(u^2+u+1), u = 1+z_eff

# --- INPUTS --------------------------------------------------
# symbol | value | status                 | predictions/ file
# -------|-------|------------------------|--------------------------
# z_eff  | ~1.85 | [adopted, N_hub-class] | predictions/z_eff.py
# (via predictions/Omega_m_LCDM.py)

# --- IMPLEMENTATION ------------------------------------------

import functools
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from z_eff import predict_z_eff, BAO_ANCHORS, SN_MODEL
from Omega_m_LCDM import predict_Omega_m_LCDM


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_Omega_Lambda_LCDM(z_eff):
    """
    ΛCDM-fit dark-energy fraction = 1 - Omega_m_LCDM(z_eff) (flat-ΛCDM).

    Parameters
    ----------
    z_eff : float
        Adopted cosmology effective redshift (predictions/z_eff.py).

    Returns
    -------
    float
        Omega_Lambda_LCDM (dimensionless).
    """
    return 1.0 - predict_Omega_m_LCDM(z_eff)


# --- INTROSPECTION (for run_predictions.py) ------------------
_z_eff = predict_z_eff(BAO_ANCHORS, SN_MODEL)
Omega_Lambda_LCDM_pred = predict_Omega_Lambda_LCDM(_z_eff)
Omega_Lambda_LCDM_obs = 0.6847
Omega_Lambda_LCDM_sigma = 0.0073


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    z_eff = _z_eff
    val = Omega_Lambda_LCDM_pred
    anchor = predict_Omega_Lambda_LCDM(math.sqrt(3.0))
    obs, sig = Omega_Lambda_LCDM_obs, Omega_Lambda_LCDM_sigma
    print("=" * 72)
    print(" Omega_Lambda_LCDM (downstream of ADOPTED z_eff)")
    print("=" * 72)
    print(f"  z_eff (adopted)          = {z_eff:.4f}")
    print(f"  Omega_Lambda_LCDM @ z_eff= {val:.4f}")
    print(f"  @ z=sqrt(3) (K-rational) = {anchor:.4f} (=2/3 exactly)")
    print(f"  Planck 2018              = {obs} +/- {sig}")
    print(f"  deviation                = {(val-obs)/sig:+.1f} sigma_obs")
    print(f"  Lambda_CC factor-of-2: Om_L_LCDM/(1/3) @ z_eff = "
          f"{val/(1.0/3.0):.3f}; @ K-rational anchor = "
          f"{anchor/(1.0/3.0):.3f} (=2 exactly)")
    assert abs(anchor - 2.0/3.0) < 1e-12
    print("  OK (K-rational anchor 2/3 exact -> Lambda ratio 2 exact)")
