"""
The framework's one adopted dimensional input, unit-setting identifications,
and downgraded predictions.

The framework adopts EXACTLY ONE dimensional physical number: N_hub ≈ 8.394881e60
(the universe's worldline length / hub count — "which universe / how big"; see
`predictions/N_hub.py` and `simulator.axioms.n_hub_pivot()`). Everything dimensional
is DERIVED from it — including the Fermi constant G_F (G_F = 1/(√2 v²), v from the
BZJ cascade ← N_hub: G_F is a DOWNSTREAM PREDICTION, NOT an anchor — `predictions/G_F.py`).
Nothing in the framework "is tied to G_F". The earlier "N_hub anchored from G_F"
framing is RETRACTED (2026-05-12); the only role the measured G_F now plays is as a
ppm-precision check on N_hub's adopted value (`predictions/N_hub.py:n_hub_from_g_f_consistency`).

Also here:
- G_N: identification (Planck-units convention; G_N · M_Pl² = 1 by choice — a unit
  choice, not a physics anchor; M_substrate = 1 makes M_Pl nearly derived via
  M_substrate/M_Pl = √π/8).
- m_t: DOWNGRADED 2026-05-04 EOD+3 (Koide-waterfall computation needs PDG m_c + m_b
  as load-bearing inputs; cannot zero-input derive).
- G_F: returned here as the MEASURED value. G_F is the observable that CALIBRATES the
  adopted N_hub (predictions/N_hub.py), so its prediction matches by construction (a
  round-trip, like v_Higgs) — it is NOT the framework's input, and the genuine
  independent predictions from N_hub are H_0, t_0, and the particle masses.
"""

import math
from fractions import Fraction
import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from simulator.srs_engine.kernel import CountingKernel


# ============================================================================
# G_F — the OBSERVED Fermi constant (for comparison; G_F is a PREDICTION, not an anchor)
# ============================================================================

# PDG 2024 / MuLan 2011, 0.51 ppm — the MEASURED Fermi constant. Used by
# predictions/N_hub.py:n_hub_from_g_f_consistency to fix the value of the adopted
# N_hub to ppm precision (G_F is the calibrating observable). Because N_hub's value
# is fixed by exactly that chain, the predicted G_F (predictions/G_F.py) matches
# this measured value by construction (a round-trip, like v_Higgs) — G_F is NOT the
# framework's input; the genuine predictions from N_hub are H_0, t_0, the masses.
_G_F_OBS_GeV2 = 1.1663787e-5


def G_F(kernel=None):
    """G_F = 1.1663787e-5 GeV^-2 — the OBSERVED Fermi constant.

    NOT an anchor (the "N_hub anchored from G_F" framing was RETRACTED 2026-05-12).
    G_F is the observable that CALIBRATES the framework's one adopted dimensional
    input, N_hub (predictions/N_hub.py:n_hub_from_g_f_consistency fixes N_hub's value
    so the BZJ chain reproduces this measured G_F) — so the *predicted* G_F
    (G_F = 1/(√2 v²), v ← N_hub via BZJ; predictions/G_F.py) matches by construction,
    like v_Higgs. The genuine independent predictions from N_hub are H_0, t_0, and the
    particle masses. This function returns the *measured* value (the calibration target).
    """
    return _G_F_OBS_GeV2


# ============================================================================
# IDENTIFICATION — G_N (Newton's constant via Planck-units convention)
# ============================================================================

def G_N_dimensionless(kernel=None):
    """G_N · M_Pl² = 1 — Newton constant in framework natural units.
    IDENTIFICATION (unit-setting, not derived).

    Counting query: G_UV_lattice · M_substrate² = π / (16·N_atoms) = π/64
    on srs (N_atoms = 4); combined with M_Pl/M_substrate = 8/√π gives
    G_N · M_Pl² = (π/64)·(64/π) = 1 exactly.
    """
    kernel = kernel or CountingKernel()
    n_atoms = kernel.substrate.N_ATOMS
    G_UV_lattice = math.pi / (16 * n_atoms)
    M_Pl_over_substrate_sq = 64 / math.pi  # (8/√π)²
    return G_UV_lattice * M_Pl_over_substrate_sq


def G_N_SI(kernel=None):
    """G_N ≈ 6.674e-11 m³/(kg·s²) — Newton's constant in SI units.
    IDENTIFICATION via M_Pl_natural + ℏ + c + GeV→J conversions.
    """
    M_Pl_GeV = 1.220890e19  # framework's natural-unit Planck mass at the SI conversion point
    hbar_J_s = 1.054571817e-34
    c_m_s = 299792458.0
    GeV_to_J = 1.602176634e-10
    M_Pl_J = M_Pl_GeV * GeV_to_J
    M_Pl_kg = M_Pl_J / c_m_s ** 2
    return hbar_J_s * c_m_s / M_Pl_kg ** 2


# ============================================================================
# DOWNGRADED — m_t (zero-empirical-inputs not satisfiable)
# ============================================================================

def m_top(kernel=None):
    """m_t — DOWNGRADED 2026-05-04 EOD+3 (Row P38).

    The framework's prior Koide-waterfall computation uses observed m_c
    (PDG 1.27 GeV) and m_b (PDG 4.18 GeV) as load-bearing inputs to the
    prediction logic, so the framework cannot compute m_t without two
    PDG empirical inputs. Two reframing attempts NEGATIVE:
    - Σ(h) per-sector lift (sector-blind)
    - y_t(GUT) = 1 chain (fit-driven)

    Same R-14 (Pati-Salam quark/lepton differentiation) blocker as the
    rest of Row P39 quark masses (m_u, m_c, m_d, m_s, m_b).

    Returns None to flag explicitly; do NOT use this as a prediction.
    """
    return None
