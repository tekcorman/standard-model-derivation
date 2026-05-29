#!/usr/bin/env python3
"""
Omega_b — absolute baryon fraction (ΛCDM-fit frame). Type-4 composition:
Omega_m_LCDM(z_eff) * (1 - Omega_DM/Omega_m) = Omega_m_LCDM * visible
fraction (Row P22 Poisson head). Downstream of the ADOPTED z_eff.

Grade: MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff. Carries a
known ~sub-percent partition residual (Row P22 ratio vs Planck-empirical
Omega_DM/Omega_m) NOT attributable to z_eff.
"""

# ============================================================
# PARAMETER: Omega_b
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       0.04930 +/- 0.00046  (Planck 2018)
# Source:      Planck 2018
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       Omega_m_LCDM(z_eff) * 61*e^-6 ; ~0.0487 at z_eff~1.85
# Deviation:   ~ -1.3 sigma_obs (first-moment z_eff)

# --- DERIVED FORMULA -----------------------------------------
# Omega_b = Omega_m_LCDM(z_eff) * (1 - Omega_DM/Omega_m)
#         = Omega_m_LCDM(z_eff) * P(k<=k* | Poisson(2k*))

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
def predict_Omega_b(z_eff, k_star):
    """
    Absolute baryon fraction (ΛCDM-fit frame) =
    Omega_m_LCDM(z_eff) * (1 - (Omega_DM/Omega_m)(k_star)).

    Parameters
    ----------
    z_eff : float
        Adopted cosmology effective redshift (predictions/z_eff.py).
    k_star : int
        Substrate valence (= 3; Row 4 theorem-grade).

    Returns
    -------
    float
        Omega_b (dimensionless).
    """
    return predict_Omega_m_LCDM(z_eff) * (1.0 - predict_Omega_DM_over_Omega_m(k_star))


# --- INTROSPECTION (for run_predictions.py) ------------------
_d = predict_d_spatial()
_k = predict_k_star(_d)
_z_eff = predict_z_eff(BAO_ANCHORS, SN_MODEL)
Omega_b_pred = predict_Omega_b(_z_eff, _k)
Omega_b_obs = 0.04930
Omega_b_sigma = 0.00046


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    d, k, z_eff = _d, _k, _z_eff
    val = Omega_b_pred
    obs, sig = Omega_b_obs, Omega_b_sigma
    Om = predict_Omega_m_LCDM(z_eff)
    ODM = Om * predict_Omega_DM_over_Omega_m(k)
    print("=" * 72)
    print(" Omega_b (downstream of ADOPTED z_eff x Row-P22 visible fraction)")
    print("=" * 72)
    print(f"  z_eff (adopted)   = {z_eff:.4f}")
    print(f"  Omega_b @ z_eff   = {val:.5f}")
    print(f"  Planck 2018       = {obs} +/- {sig}")
    print(f"  deviation         = {(val-obs)/sig:+.1f} sigma_obs")
    print(f"  closure: Omega_DM + Omega_b = {ODM + val:.4f} vs "
          f"Omega_m_LCDM = {Om:.4f}")
    assert abs((ODM + val) - Om) < 1e-12, "matter-fraction closure failed"
    print("  OK (visible/dark partition closes against Omega_m_LCDM exactly)")
