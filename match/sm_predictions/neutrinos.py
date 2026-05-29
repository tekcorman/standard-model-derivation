"""
Neutrino sector predictions — counting-first queries.

Covers:
- Neutrino masses m_ν2, m_ν3 via global spectral-gap formula + Feshbach correction
- Mass-squared splitting ratio R_ν = Δm²₃₁/Δm²₂₁
- PMNS mixing angles θ_12, θ_13, θ_23

CHANNEL-SELECTED throughout: each PMNS angle's dark-extraction-map class
(1 amplitude / 2 mass² / 3 edge-local per `dark_extraction_map_derivation.md`)
is named as the structural channel; the masses' substrate-derivation channels
(global spectral gap, R-ratio splitting) are likewise named.
"""

import sys as _sys
from pathlib import Path as _Path
from fractions import Fraction
_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from predictions.m_nu2 import predict_m_nu2 as _predict_m_nu2
from predictions.m_nu3 import predict_m_nu3 as _predict_m_nu3
from predictions.R_nu_splitting import predict_R_nu_splitting as _predict_R_nu
from predictions.theta_12_PMNS import predict_theta_12_PMNS as _predict_theta_12
from predictions.theta_13_PMNS import predict_theta_13_PMNS as _predict_theta_13
from predictions.theta_23_PMNS import predict_theta_23_PMNS as _predict_theta_23

from simulator.srs_engine.kernel import CountingKernel
from .masses import _M_PL_GEV
from .gauge import V_us as _V_us
from .masses import alpha_1_bare as _alpha_1_bare


# ============================================================================
# Mass-squared splittings + neutrino masses
# ============================================================================

def R_nu_splitting(kernel=None):
    """R_ν = Δm²₃₁/Δm²₂₁ ≈ 32.57 — neutrino mass-squared splitting ratio.
    CHANNEL-SELECTED (spectral splitting via k*).

    Waterfilling-correct: spectral-splitting ratios on the substrate are
    each above-waterline; framework selects k*-controlled splitting.
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {'name': 'spectral splitting via k*',
         'channel': 'spectral_splitting_via_k_star',
         'fn': _predict_R_nu},
    ]
    selected = kernel.channel_select(
        candidates, channel='spectral_splitting_via_k_star')
    return selected['fn'](kernel.substrate.K_STAR)


def m_nu3(kernel=None):
    """m_ν3 ≈ 0.0506 eV — heaviest neutrino mass.
    CHANNEL-SELECTED (global spectral-gap formula).

    Waterfilling-correct: candidate neutrino-mass derivations exist for
    different substrate-cosmology cascades:
        global spectral gap:  m_ν3 = (k* · N_atoms) · M_Pl · N_hub^(-1/2)
                              (UNIQUE-THEOREM-GRADE-CONDITIONAL 2026-05-04)
        ADOPTED-PS chain:     RETRACTED (used m_t(GUT), MSSM RG, tan β)
        Cosmology-direct:     not yet enumerated as a closure path

    Framework selects the global spectral-gap channel.
    """
    kernel = kernel or CountingKernel()
    from .cosmology import N_HUB
    candidates = [
        {'name': 'global spectral gap m_ν3 = (k*·N_atoms)·M_Pl·N_hub^(-1/2)',
         'channel': 'global_spectral_gap_m_nu3',
         'fn': _predict_m_nu3},
    ]
    selected = kernel.channel_select(
        candidates, channel='global_spectral_gap_m_nu3')
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return selected['fn'](
            kernel.substrate.K_STAR,
            kernel.substrate.N_ATOMS,
            _M_PL_GEV,
            N_HUB,
        )


def m_nu2(kernel=None):
    """m_ν2 ≈ 0.00886 eV — second neutrino mass.
    CHANNEL-SELECTED (m_ν3 / √R_ν via splitting ratio).
    """
    kernel = kernel or CountingKernel()
    from .cosmology import N_HUB
    candidates = [
        {'name': 'splitting ratio: m_ν3 / √R_ν',
         'channel': 'm_nu_split_via_R_ratio',
         'fn': _predict_m_nu2},
    ]
    selected = kernel.channel_select(
        candidates, channel='m_nu_split_via_R_ratio')
    R_split = R_nu_splitting(kernel)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return selected['fn'](
            kernel.substrate.K_STAR,
            kernel.substrate.N_ATOMS,
            _M_PL_GEV,
            N_HUB,
            R_split,
        )


# ============================================================================
# PMNS angles — each is a dark-extraction-map class
# ============================================================================

def _dark_extraction_class_candidates():
    """The 3 above-waterline dark-extraction-map classes per
    `predictions/dark_extraction_map_derivation.md` §3."""
    return [
        {'name': 'Class 1 (amplitude): √5/4 · α_1',
         'channel': 'dark_class_1_amplitude'},
        {'name': 'Class 2 (mass²): 5/3 · α_1',
         'channel': 'dark_class_2_mass_squared'},
        {'name': 'Class 3 (edge-local): 1 · α_1',
         'channel': 'dark_class_3_edge_local'},
    ]


def theta_12_PMNS(kernel=None):
    """θ_12 ≈ 33.07° — PMNS solar mixing angle.
    CHANNEL-SELECTED (PS perp identity, TBM + V_us, edge-local class).

    Waterfilling-correct: θ_12 PMNS reads off the (-1)-eigenspace
    rotation between TBM and V_us; the relevant dark-extraction channel
    is Class 3 edge-local (vertex-symmetric, no Im(h) enhancement).
    """
    kernel = kernel or CountingKernel()
    selected = kernel.channel_select(
        _dark_extraction_class_candidates(),
        channel='dark_class_3_edge_local')
    v_us = float(_V_us(kernel))
    cos_TBM_sq = 2.0 / 3.0  # tribimaximal reference value
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return _predict_theta_12(v_us, cos_TBM_sq)


def theta_13_PMNS(kernel=None):
    """θ_13 ≈ 8.61° — PMNS reactor mixing angle.
    CHANNEL-SELECTED (TBM + edge-local dark Class 3).

    Waterfilling-correct: per dark-extraction map (R-9 closure pattern,
    2026-05-02 EOD+13), θ_13 is a Class-3 edge-local observable.
    V_us_bare = V_us / (1 + √5/4 · α_1) (strip Class-1 amplitude factor).
    """
    kernel = kernel or CountingKernel()
    selected = kernel.channel_select(
        _dark_extraction_class_candidates(),
        channel='dark_class_3_edge_local')
    v_us = float(_V_us(kernel))
    a1_bare = float(_alpha_1_bare(kernel))
    k_star = kernel.substrate.K_STAR
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return _predict_theta_13(v_us, a1_bare, k_star)


def theta_23_PMNS(kernel=None):
    """θ_23 ≈ 48.72° — PMNS atmospheric mixing angle.
    CHANNEL-SELECTED (mass-matrix 2×2 diagonalization, dark Class 2).

    Waterfilling-correct: θ_23 is a mass-mixing-diagonalization observable;
    its dark-extraction channel is Class 2 (mass², ν_m² = tan²(arg h) = 5/3).
    """
    kernel = kernel or CountingKernel()
    selected = kernel.channel_select(
        _dark_extraction_class_candidates(),
        channel='dark_class_2_mass_squared')
    a1_bare = float(_alpha_1_bare(kernel))
    h = kernel.substrate.ramanujan_eigenvalue_at_P
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return _predict_theta_23(a1_bare, h)
