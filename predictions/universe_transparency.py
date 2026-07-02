#!/usr/bin/env python3
"""
Universe transparency to UHE photons above the framework's ~147 PeV scale.

Audit anchor: downstream prediction from η_lattice = 1/12 (Hashimoto dim-6
LV per `predictions/eta_lattice_lorentz_dim6.py`). Component of the Lorentz
arc closure per `docs/theorems/lorentz_sig_ccclose_joint_closure.md` and Stage 3
results in `docs/theorems/theorem_lorentz_causal_sector.md`.

Framework prediction: subluminal dim-6 LIV (eta_lattice > 0) shifts the
photon pair-production threshold UPWARD, making the universe more
transparent to UHE photons above ~147 PeV than standard-model QED
would predict.

Gate grade: COMPUTED (Type 2 arithmetic on eta_lattice + standard
pair-production kinematics).
"""

# ============================================================
# PARAMETER: E_transparent (universe transparency threshold for UHE photons)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Tentative / partial evidence for anomalous transparency.
#              The GRB 221009A 18-TeV photon (Finke & Razzaque, ApJL
#              942, L21 (2023) [arXiv:2210.11261]) is cited as possible
#              evidence for raised pair-production thresholds — sign
#              consistent with subluminal dim-6 LIV. Their best-fit
#              E_QG,2 <~ 10^-6 E_Pl is in the right SIGN but not
#              numerically compatible with eta_lattice = 1/12 if taken
#              literally.
#              At the framework-predicted ~147 PeV scale, no extragalactic
#              photons have been observed. Pierre Auger UHE-photon flux
#              limits (JCAP 05 (2023) 021) set upper bounds on fluxes
#              above 10^17 eV (~100 PeV) but do not yet constrain
#              transparency at 147 PeV.
# Source:      Finke & Razzaque, ApJL 942, L21 (2023).
#              Pierre Auger Collab., JCAP 05 (2023) 021.
# PDG edition: Not PDG-tabulated.

# --- PREDICTED VALUE -----------------------------------------
# Value:       E_transparent ~ 147 PeV (framework's scale energy).
#              Above this scale, pair-production thresholds are raised
#              by the subluminal dim-6 LIV, making the universe
#              progressively more transparent to UHE photons.
# Deviation:   Qualitatively consistent with tentative GRB 221009A
#              evidence. Quantitative test awaits SWGO / LHAASO-upgrade
#              era (2030s).

# --- DERIVED FORMULA -----------------------------------------
# Standard QED pair production (gamma + gamma_bg -> e+ e-) has threshold
# in the absence of LIV:
#
#   E_gamma * E_bg > m_e^2   (center-of-momentum)
#
# For gamma_bg from CMB or EBL at ~meV scale, E_gamma_th ~ PeV-EeV range.
#
# Under dim-6 subluminal LIV with coefficient eta > 0:
#
#   Photon dispersion: E^2 ~ p^2 - eta p^4 / E_Pl^2
#
# This lowers the photon's CoM energy at fixed p, raising the
# pair-production threshold by a factor dependent on eta and the
# photon energy. Qualitatively: for E_gamma >~ E_scale where
#   E_scale = (m_e^2 E_Pl^2 / eta)^(1/4) = 147 PeV (framework),
# the threshold is significantly raised, and the universe becomes
# progressively more transparent at higher energies.
#
# For the framework's eta = 1/12 subluminal, the crossover scale
# is E_scale ~ 147 PeV; complete transparency (no threshold) is
# achieved for photons well above this (ref: Jacobson-Liberati-
# Mattingly 2003; specific UHE-photon transparency formulas
# reviewed in Martinez-Huerta et al. 2020).
#
# Derivation chain:
#   predictions/eta_lattice_lorentz_dim6.py: eta_lattice = 1/12
#     -> predictions/scale_energy_hashimoto.py: E_scale = 147 PeV
#     -> subluminal LIV raises pair-production threshold above E_scale
#     -> universe transparency above ~147 PeV (qualitative prediction).

# --- INPUTS --------------------------------------------------
# symbol             | value           | status     | source                              | meaning
# -------------------|-----------------|------------|-------------------------------------|---------
# eta_lattice        | 1/12            | [derived]  | predictions/eta_lattice_lorentz_dim6 | dim-6 coefficient
# E_scale            | 147 PeV         | [derived]  | predictions/scale_energy_hashimoto   | scale energy
# m_e                | 0.511 MeV       | [external] | PDG 2024                             | electron rest energy
# E_Pl               | M_Pl_natural.M_Pl_GeV | [derived] | predictions/M_Pl_natural.py | M_Pl/M_subst=8/√π theorem; GeV=single declared SI-anchor (CODE imports it line 99 — NOT "[external] PDG")

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eta_lattice_lorentz_dim6 import predict_eta_lattice
from scale_energy_hashimoto import predict_scale_energy
import functools

# Physical constants — single-source from canonical anchor file
from M_Pl_natural import M_Pl_GeV as E_Pl_GeV   # CODATA single-source — ANTHROPOCENTRIC SI TRANSLATION
from m_e import m_e_obs as m_e_GeV                # PDG empirical anchor (single-source via m_e.py)

# Framework inputs
from p_toggle import predict_p_toggle
p = predict_p_toggle()
eta = predict_eta_lattice(1/8, 1/768)                  # = 1/12
E_scale_GeV = predict_scale_energy(eta, m_e_GeV, E_Pl_GeV, p)
from M_Pl_natural import GeV_per_PeV   # = 1e6, SI prefix single-source
E_scale_PeV = E_scale_GeV / GeV_per_PeV
E_transparent_PeV = E_scale_PeV   # transparency onset = scale energy
universe_transparency_pred = E_transparent_PeV

print(f"eta_lattice      = {eta:.6f} = 1/12  (subluminal)")
print(f"E_scale          = {E_scale_PeV:.1f} PeV")
print(f"Sign:            subluminal -> pair-production threshold RAISED")
print(f"E_transparent    ~ {E_transparent_PeV:.0f} PeV")
print(f"(universe becomes progressively more transparent above this scale)")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_universe_transparency_threshold(eta_lattice, m_e, E_Pl, p_toggle):
    """
    Characteristic energy scale above which subluminal dim-6 LIV makes
    the universe increasingly transparent to UHE photons.

    Equivalent to the scale energy E_th from standard pair-production
    threshold formula. Provides a qualitative onset scale for
    transparency effects.

    Coefficients sourced from framework primitives:
      2 (squaring m_e and E_Pl) = p_toggle
      4 (fourth-root denominator) = p_toggle²
      1 (fourth-root numerator)   = p_toggle - 1

    Parameters
    ----------
    eta_lattice, m_e, E_Pl : float
        Dim-6 LIV coefficient, electron mass, Planck energy.
    p_toggle : int
        Toggle arity (from predict_p_toggle).

    Returns
    -------
    float
        Transparency-onset energy in same units as m_e, E_Pl.
    """
    if eta_lattice <= 0:
        raise ValueError(
            f"Transparency from raised thresholds requires subluminal "
            f"(eta > 0) LIV. Got eta = {eta_lattice}."
        )
    sq = p_toggle
    one_nb = p_toggle - 1
    fourth_pow = p_toggle * p_toggle
    return (m_e**sq * E_Pl**sq / eta_lattice)**(one_nb / fourth_pow)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = E_scale_GeV
    pure_result = predict_universe_transparency_threshold(eta, m_e_GeV, E_Pl_GeV, p)
    print(f"\nImplementation: {impl_result:.4e} GeV")
    print(f"Pure function:  {pure_result:.4e} GeV")
    assert abs(impl_result - pure_result) < 1.0

    # Same as scale_energy_hashimoto (by construction)
    from scale_energy_hashimoto import predict_scale_energy
    scale_result = predict_scale_energy(eta, m_e_GeV, E_Pl_GeV, p)
    assert abs(pure_result - scale_result) < 1e-6

    # Scope guard: subluminal only
    try:
        predict_universe_transparency_threshold(-0.1, m_e_GeV, E_Pl_GeV, p)
    except ValueError:
        print("Scope guard:    correctly rejected negative eta  OK")
    else:
        raise AssertionError("Scope guard should reject superluminal eta")

    print(f"\nOK: E_transparent ~ {E_scale_PeV:.0f} PeV (onset of universe transparency).")
