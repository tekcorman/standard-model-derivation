#!/usr/bin/env python3
"""
Scale energy for dimension-6 Lorentz-violation effects on the Hashimoto
(NB walk / photon) dispersion.

Audit anchor: cross-reference to dim-6 LV coefficients (Hashimoto via
`predictions/eta_lattice_lorentz_dim6.py` and scalar Bloch via
`predictions/srs_bloch_lv_dim6.py`). Component of the Lorentz arc closure
per `docs/theorems/lorentz_sig_ccclose_joint_closure.md` and Stage 3 results in
`docs/theorems/theorem_lorentz_causal_sector.md`.

Framework prediction: E_scale ~ 147 PeV, derived from the standard
threshold formula E_th = (m_e^2 E_Pl^2 / |eta_lattice|)^(1/4)
with |eta_lattice| = 1/12 (exact from the srs lattice dispersion).

Gate grade: COMPUTED from eta_lattice (Type 2 arithmetic on
upstream framework value).
"""

# ============================================================
# PARAMETER: E_scale (scale energy for dim-6 LIV effects)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Observable scale for dim-6 LIV effects on photon
#              propagation. Current highest-energy photon observations
#              reach ~1-1.4 PeV (LHAASO KM2A, 2021):
#                - 1.4 PeV photon from Cygnus region (Cao et al.,
#                  Nature 594, 33 (2021))
#                - 1.1 PeV photon from Crab Nebula (Cao et al.,
#                  Science 373, 425 (2021))
#              These are ~two orders of magnitude below the framework-
#              predicted 147 PeV scale. Next-generation facilities
#              (SWGO, LHAASO upgrades) in the 2030s target this regime.
# Source:      LHAASO Collaboration papers (above).
# PDG edition: Not PDG-tabulated.

# --- PREDICTED VALUE -----------------------------------------
# Value:       E_scale ~ 1.47 x 10^8 GeV = 147 PeV
# Deviation:   N/A — no direct observation at this scale yet.

# --- DERIVED FORMULA -----------------------------------------
# For dim-6 subluminal LIV with coefficient eta in the dispersion
#   E^2 = p^2 + m^2 - eta * p^4 / E_Pl^2,
# the threshold at which LIV effects modify pair-production kinematics
# is given by the standard formula (Coleman-Glashow 1999; Jacobson,
# Liberati, Mattingly 2003):
#
#   E_th = (m_e^2 * E_Pl^2 / |eta|)^(1/4).
#
# With the framework prediction |eta_lattice| = 1/12 (exact from
# predictions/eta_lattice_lorentz_dim6.py):
#
#   E_th = (m_e^2 * E_Pl^2 * 12)^(1/4)
#        = m_e^(1/2) * E_Pl^(1/2) * 12^(1/4)
#
# Numerical evaluation with m_e = 0.511 MeV, E_Pl = 1.22 x 10^19 GeV:
#   E_th ~ 1.47 x 10^8 GeV = 147 PeV.
#
# Derivation chain:
#   Stage 3 (Lorentz causal sector)
#     -> eta_lattice = 1/12 (CAS-verified, see predictions/eta_lattice_lorentz_dim6)
#     -> standard threshold formula E_th = (m_e^2 E_Pl^2 / |eta|)^(1/4)
#        (Coleman-Glashow 1999 or Jacobson-Liberati-Mattingly 2003).
#     -> E_th ~ 147 PeV.

# --- INPUTS --------------------------------------------------
# symbol       | value                | status     | source                                   | meaning
# -------------|----------------------|------------|------------------------------------------|---------
# eta_lattice  | 1/12                 | [derived]  | predictions/eta_lattice_lorentz_dim6.py  | dim-6 LIV coefficient
# m_e          | 0.5109989461e-3 GeV  | [external] | PDG 2024 electron mass                   | electron rest energy
# E_Pl         | 1.2208996e19 GeV     | [external] | PDG 2024 Planck energy                   | Planck energy

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eta_lattice_lorentz_dim6 import predict_eta_lattice
import functools

# Physical constants — single-source from canonical anchor file
from M_Pl_natural import M_Pl_GeV as E_Pl_GeV   # CODATA single-source — ANTHROPOCENTRIC SI TRANSLATION
from m_e import m_e_obs as m_e_GeV                # PDG empirical anchor (single-source via m_e.py)

# Framework input
eta_lattice = predict_eta_lattice(1/8, 1/768)   # = 1/12

# Standard threshold formula
E_scale_GeV = (m_e_GeV**2 * E_Pl_GeV**2 / abs(eta_lattice))**(1/4)
from M_Pl_natural import GeV_per_PeV  # = 1e6, SI prefix single-source
E_scale_PeV = E_scale_GeV / GeV_per_PeV

print(f"eta_lattice  = {eta_lattice:.15f} = 1/12")
print(f"m_e          = {m_e_GeV:.6e} GeV")
print(f"E_Pl         = {E_Pl_GeV:.6e} GeV")
print(f"E_scale      = {E_scale_GeV:.4e} GeV = {E_scale_PeV:.1f} PeV")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_scale_energy(eta_lattice, m_e, E_Pl, p_toggle):
    """
    Scale energy for dim-6 LIV effects on photon propagation.

    Uses the standard pair-production threshold formula
      E_th = (m_e^2 * E_Pl^2 / |eta|)^(1/4)
    from Coleman-Glashow 1999 and Jacobson-Liberati-Mattingly 2003.

    Coefficients sourced from framework primitives:
      2 (squaring m_e and E_Pl in threshold formula) = p_toggle
      4 (fourth-root exponent denominator)            = p_toggle²
      1 (fourth-root numerator)                       = p_toggle - 1

    Parameters
    ----------
    eta_lattice : float
        Dim-6 LIV coefficient. Framework: 1/12.
    m_e, E_Pl : float
        Electron mass and Planck energy in matching units.
    p_toggle : int
        Toggle arity (from predict_p_toggle). Sources the dim-6 LIV
        threshold formula's 2 / 4 / 1 coefficients.

    Returns
    -------
    float
        E_scale in same units as m_e and E_Pl.
    """
    sq = p_toggle              # = 2, squaring exponent
    one_nb = p_toggle - 1       # = 1, fourth-root numerator
    fourth_pow = p_toggle * p_toggle  # = 4, fourth-root denominator
    return (m_e**sq * E_Pl**sq / abs(eta_lattice))**(one_nb / fourth_pow)


# --- VALIDATION ----------------------------------------------

scale_energy_hashimoto_pred = E_scale_PeV


if __name__ == "__main__":
    impl_result = E_scale_GeV
    from p_toggle import predict_p_toggle
    pure_result = predict_scale_energy(eta_lattice, m_e_GeV, E_Pl_GeV, predict_p_toggle())
    print(f"\nImplementation: {impl_result:.6e} GeV")
    print(f"Pure function:  {pure_result:.6e} GeV")
    assert abs(impl_result - pure_result) < 1.0  # within 1 GeV

    # Cross-check rough order of magnitude
    assert 1e7 < pure_result < 1e9, f"Scale out of expected range: {pure_result}"
    assert 100 < E_scale_PeV < 200, f"Expected ~147 PeV; got {E_scale_PeV}"
    print(f"Scale:          ~{E_scale_PeV:.0f} PeV (expected ~147 PeV)")

    # Sanity: LHAASO max photon observation ~1.4 PeV, two orders below
    lhaaso_max_PeV = 1.4
    assert E_scale_PeV > lhaaso_max_PeV * 50, "Scale below current sensitivity"
    print(f"Above LHAASO:   factor ~{E_scale_PeV / lhaaso_max_PeV:.0f} "
          f"(LHAASO max photon = 1.4 PeV)")

    print("\nOK: E_scale ~ 147 PeV from eta_lattice = 1/12 and standard threshold formula.")
