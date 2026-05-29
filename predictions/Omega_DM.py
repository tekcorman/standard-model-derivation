#!/usr/bin/env python3
"""
Omega_DM — absolute dark-matter fraction (ΛCDM-fit frame). Type-4
composition: Omega_m_LCDM(z_eff) * (Omega_DM/Omega_m), where the
visible/dark partition is the theorem-grade Row P22 Cl(2k*) Fock Poisson
tail. Downstream of the ADOPTED z_eff.

Grade: MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff. Supersedes
predictions/retracted/Omega_DM.py (which used external Omega_b).
"""

# ============================================================
# PARAMETER: Omega_DM
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       0.2645 +/- 0.0050  (Planck 2018)
# Source:      Planck 2018
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       Omega_m_LCDM(z_eff) * (1 - 61*e^-6) ; ~0.273 at z_eff~1.85
# Deviation:   ~ +1.8 sigma_obs (first-moment z_eff)

# --- DERIVED FORMULA -----------------------------------------
# Omega_DM = Omega_m_LCDM(z_eff) * (Omega_DM/Omega_m)
#   Omega_m_LCDM   : predictions/Omega_m_LCDM.py (theorem-grade form @ z_eff)
#   Omega_DM/Omega_m: predictions/Omega_DM_over_Omega_m.py (Row P22,
#                     UNIQUE-THEOREM-GRADE: 1 - P(k<=k*|Poisson(2k*)))

# --- INPUTS --------------------------------------------------
# symbol | value | status                 | predictions/ file
# -------|-------|------------------------|--------------------------------
# z_eff  | ~1.85 | [adopted, N_hub-class] | predictions/z_eff.py
# k_star | 3     | [derived]              | predictions/k_star.py

# --- IMPLEMENTATION ------------------------------------------

import functools
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from d_spatial import predict_d_spatial
from k_star import predict_k_star
from z_eff import predict_z_eff, BAO_ANCHORS, SN_MODEL
from Omega_m_LCDM import predict_Omega_m_LCDM
from Omega_DM_over_Omega_m import predict_Omega_DM_over_Omega_m


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_Omega_DM(z_eff, k_star):
    """
    Absolute dark-matter fraction (ΛCDM-fit frame) =
    Omega_m_LCDM(z_eff) * (Omega_DM/Omega_m)(k_star).

    Parameters
    ----------
    z_eff : float
        Adopted cosmology effective redshift (predictions/z_eff.py).
    k_star : int
        Substrate valence (= 3; Row 4 theorem-grade).

    Returns
    -------
    float
        Omega_DM (dimensionless).
    """
    return predict_Omega_m_LCDM(z_eff) * predict_Omega_DM_over_Omega_m(k_star)


# --- INTROSPECTION (for run_predictions.py) ------------------
_d = predict_d_spatial()
_k = predict_k_star(_d)
_z_eff = predict_z_eff(BAO_ANCHORS, SN_MODEL)
Omega_DM_pred = predict_Omega_DM(_z_eff, _k)
Omega_DM_obs = 0.2645
Omega_DM_sigma = 0.0050


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    d, k, z_eff = _d, _k, _z_eff
    val = Omega_DM_pred
    obs, sig = Omega_DM_obs, Omega_DM_sigma
    print("=" * 72)
    print(" Omega_DM (downstream of ADOPTED z_eff x Row-P22 ratio)")
    print("=" * 72)
    print(f"  z_eff (adopted)        = {z_eff:.4f}")
    print(f"  Omega_DM/Omega_m (P22) = {predict_Omega_DM_over_Omega_m(k):.4f}")
    print(f"  Omega_DM @ z_eff       = {val:.4f}")
    print(f"  Planck 2018            = {obs} +/- {sig}")
    print(f"  deviation              = {(val-obs)/sig:+.1f} sigma_obs")
    print("  OK")
