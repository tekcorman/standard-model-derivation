"""
Symmetry-breaking cascade — scale-aware substrate exploration.

The framework's substrate (srs) is fixed; what varies as the universe cools
is which symmetries are unbroken in the low-energy effective theory at given
energy scale µ. This module exposes the breaking cascade as the simulator's
primary exploration mechanism.

Trajectory (by decreasing energy scale µ):

    µ ≥ M_unif      'unbroken_PS'       SU(4) × SU(2)_L × SU(2)_R     unified α_GUT
    M_unif > µ ≥ M_SUSY 'MSSM'              SU(3) × SU(2)_L × U(1)_Y      MSSM β-functions
    M_SUSY > µ ≥ M_EW   'SM_with_EW'        SU(3) × SU(2)_L × U(1)_Y      SM β-functions
    M_EW > µ ≥ M_QCD    'SM_broken_EW'      SU(3) × U(1)_EM               massive W/Z
    µ < M_QCD            'confined_QCD'      U(1)_EM (QCD confined)        hadrons

Within each regime the appropriate β-functions and matter content apply.

ANCHOR STATUS:
  - α_GUT = 1/24: substrate-native (theorem-grade, Class C Cl(6) Fock count)
  - M_unif ≈ 1.985e16 GeV: substrate-native (theorem-grade-conditional via
    cascade theorem; depends on M_Pl_natural in lattice units)
  - M_SUSY ≈ 1000 GeV: EXTERNAL ANCHOR (assumption; SUSY scale not yet
    derived from substrate)
  - v_Higgs ≈ 246 GeV: N_hub-anchored (← N_hub via the BZJ cascade; the cosmology layer is not yet
    substrate-native — see an internal working note)
  - M_Z ≈ 91.97 GeV: RG-derived self-consistently from v_Higgs; inherits
    v_Higgs's N_hub anchor
  - M_EW (= v_Higgs): inherits N_hub anchor
  - M_QCD ≈ 0.2 GeV: SET BY α_s confinement, RG-derived (inherits the
    same N_hub anchor cluster)

Numbers below M_Z are NOT YET fully substrate-native; they come from MSSM RG
running with v_Higgs as external input. Per user direction 2026-05-10:
"don't expect accurate numbers until we run purely on N_hub."

Predecessors:
- proofs/gauge/_mssm_rge.py — operational MSSM/SM RG machinery
- proofs/foundations/gauge_unification_full_RG_closure.py — 5-stage closure
- predictions/M_unif.py, predictions/M_Z.py, predictions/v_higgs.py — current
  scale-anchored values
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from proofs.gauge._mssm_rge import (
    run_down_from_gut,
    run_up_from_mz,
    couplings_at_mz_from_gut,
)


# ============================================================================
# SCALE CONSTANTS (with anchor annotations)
# ============================================================================

# Substrate-native (theorem-grade):
ALPHA_GUT = 1.0 / 24.0                # Class C, Cl(6) Fock count [SUBSTRATE]
M_UNIF_GeV = 1.985e16                   # substrate via cascade theorem [SUBSTRATE-CONDITIONAL]

# External anchors / inherits N_hub anchor cluster (NOT substrate-native):
M_SUSY_GeV = 1000.0                     # EXTERNAL — assumption
V_HIGGS_GeV = 246.22                    # ← N_hub (the adopted dimensional input) via the BZJ cascade
M_EW_GeV = V_HIGGS_GeV                  # = v_Higgs; inherits N_hub anchor
M_Z_GeV = 91.97                         # RG-derived from v_Higgs; inherits anchor
M_QCD_GeV = 0.2                         # set by α_s confinement; inherits anchor

# Hypercharge normalization (SU(5) embedding):
HYPERCHARGE_NORM = 3.0 / 5.0           # α_Y = (3/5) α_1 [SUBSTRATE]


# ============================================================================
# REGIMES — symmetry-breaking cascade
# ============================================================================

REGIMES = [
    {
        'name': 'unbroken_PS',
        'gauge_group': 'SU(4) × SU(2)_L × SU(2)_R',
        'matter': 'unified PS multiplets (4, 2, 1) + (4̄, 1, 2) per gen',
        'beta_function_set': 'unified α_GUT',
        'scale_lower_GeV': M_UNIF_GeV,
        'scale_upper_GeV': float('inf'),
        'breaking_at_lower': 'PS → SU(3)_c × SU(2)_L × U(1)_Y (via Cl(6) chirality-doubled edge qubit, G2-D theorem)',
    },
    {
        'name': 'MSSM',
        'gauge_group': 'SU(3)_c × SU(2)_L × U(1)_Y',
        'matter': 'SM fermions + superpartners (squarks, sleptons, gauginos, higgsinos)',
        'beta_function_set': 'MSSM two-loop (b = [33/5, 1, -3])',
        'scale_lower_GeV': M_SUSY_GeV,
        'scale_upper_GeV': M_UNIF_GeV,
        'breaking_at_lower': 'SUSY broken (superpartners decouple)',
    },
    {
        'name': 'SM_with_EW',
        'gauge_group': 'SU(3)_c × SU(2)_L × U(1)_Y',
        'matter': 'SM fermions (3 gens) + Higgs doublet',
        'beta_function_set': 'SM two-loop (b = [41/10, -19/6, -7])',
        'scale_lower_GeV': M_EW_GeV,
        'scale_upper_GeV': M_SUSY_GeV,
        'breaking_at_lower': 'EW broken via Higgs VEV → SU(3)_c × U(1)_EM (massive W, Z)',
    },
    {
        'name': 'SM_broken_EW',
        'gauge_group': 'SU(3)_c × U(1)_EM',
        'matter': 'SM fermions (massive) + massive W, Z + photon + Higgs',
        'beta_function_set': 'SM electroweak below v_Higgs',
        'scale_lower_GeV': M_QCD_GeV,
        'scale_upper_GeV': M_EW_GeV,
        'breaking_at_lower': 'QCD confines; chiral symmetry breaking',
    },
    {
        'name': 'confined_QCD',
        'gauge_group': 'U(1)_EM (QCD confined into hadrons)',
        'matter': 'hadrons + leptons + photon',
        'beta_function_set': 'low-energy effective theory (not RG-running in same sense)',
        'scale_lower_GeV': 0.0,
        'scale_upper_GeV': M_QCD_GeV,
        'breaking_at_lower': None,
    },
]


def active_regime_at(scale_GeV):
    """Return the active regime at given energy scale."""
    for regime in REGIMES:
        if regime['scale_lower_GeV'] <= scale_GeV < regime['scale_upper_GeV']:
            return regime
    raise ValueError(f"scale_GeV={scale_GeV} doesn't match any regime")


def active_gauge_group_at(scale_GeV):
    return active_regime_at(scale_GeV)['gauge_group']


def active_matter_content_at(scale_GeV):
    return active_regime_at(scale_GeV)['matter']


# ============================================================================
# GAUGE COUPLINGS AT SCALE (running)
# ============================================================================

class Undefined:
    """Sentinel for observables that are not defined at the given scale."""
    def __init__(self, reason):
        self.reason = reason
    def __repr__(self):
        return f"Undefined({self.reason!r})"
    def __bool__(self):
        return False


def _alpha_GUT_inv():
    return 1.0 / ALPHA_GUT  # = 24


def gauge_couplings_at(scale_GeV, alpha_gut=ALPHA_GUT, m_unif=M_UNIF_GeV,
                       m_susy=M_SUSY_GeV, m_z=M_Z_GeV):
    """Return running gauge couplings at scale_GeV.

    Above M_unif: only α_GUT defined (unified).
    M_unif → M_SUSY: α_1, α_2, α_3 via MSSM RG.
    M_SUSY → M_EW: α_1, α_2, α_3 via SM RG (matching at M_SUSY).
    M_EW → M_QCD: α_3 still RG-running; α_1, α_2 broken into α_EM, sin²θ_W.
    Below M_QCD: confined; running becomes non-perturbative.

    Returns dict with keys depending on regime. 'undefined' values are
    Undefined sentinels rather than raise.
    """
    regime = active_regime_at(scale_GeV)
    out = {'regime': regime['name'], 'scale_GeV': scale_GeV}

    if regime['name'] == 'unbroken_PS':
        out['alpha_GUT'] = alpha_gut
        out['alpha_1'] = Undefined("above M_unif: only unified α_GUT")
        out['alpha_2'] = Undefined("above M_unif: only unified α_GUT")
        out['alpha_3'] = Undefined("above M_unif: only unified α_GUT")
        out['alpha_EM'] = Undefined("above M_unif: EW symmetry unbroken")
        out['sin2_theta_W'] = Undefined("above M_unif: sin²θ_W = 3/8 exact (unification value)")
        return out

    # Below M_unif: run α_1, α_2, α_3 from M_unif (or M_SUSY) to scale
    # Use existing _mssm_rge machinery.
    import numpy as np
    from proofs.gauge._mssm_rge import _integrate, B_MSSM, BIJ_MSSM, B_SM, BIJ_SM

    alpha_gut_inv = 1.0 / alpha_gut
    y_gut = np.array([alpha_gut_inv, alpha_gut_inv, alpha_gut_inv])
    t_unif = math.log(m_unif)
    t_susy = math.log(m_susy)
    t_target = math.log(scale_GeV)

    if scale_GeV >= m_susy:
        # MSSM regime
        y = _integrate(t_unif, t_target, y_gut, B_MSSM, BIJ_MSSM)
    else:
        # Need to run through M_SUSY matching
        y_susy = _integrate(t_unif, t_susy, y_gut, B_MSSM, BIJ_MSSM)
        y = _integrate(t_susy, t_target, y_susy, B_SM, BIJ_SM)

    alpha_1, alpha_2, alpha_3 = 1.0 / y[0], 1.0 / y[1], 1.0 / y[2]
    out['alpha_1'] = float(alpha_1)
    out['alpha_2'] = float(alpha_2)
    out['alpha_3'] = float(alpha_3)
    out['alpha_GUT'] = Undefined("below M_unif: gauge group split")

    # Derived: sin²θ_W and α_EM (if EW unbroken at this scale)
    if regime['name'] in ('MSSM', 'SM_with_EW'):
        alpha_Y = HYPERCHARGE_NORM * alpha_1
        out['sin2_theta_W'] = float(alpha_Y / (alpha_2 + alpha_Y))
        out['alpha_EM'] = Undefined("EW unbroken at this scale; sin²θ_W defined separately")
    elif regime['name'] in ('SM_broken_EW', 'confined_QCD'):
        # Below EW breaking: α_EM = α_1·α_2 / ((3/5)α_1 + α_2) at EW scale, then
        # runs separately. For now compute it at scale assuming continuation.
        alpha_Y = HYPERCHARGE_NORM * alpha_1
        out['sin2_theta_W'] = float(alpha_Y / (alpha_2 + alpha_Y))
        # α_EM via SU(2) × U(1)_Y → U(1)_EM mixing
        out['alpha_EM'] = float(alpha_Y * alpha_2 / (alpha_Y + alpha_2))

    return out


# ============================================================================
# OBSERVABLE DEFINEDNESS AT SCALE
# ============================================================================

OBSERVABLE_REGIMES = {
    'alpha_GUT':        ['unbroken_PS'],
    'M_unif':           ['unbroken_PS'],
    'alpha_1':          ['MSSM', 'SM_with_EW', 'SM_broken_EW'],
    'alpha_2':          ['MSSM', 'SM_with_EW', 'SM_broken_EW'],
    'alpha_3':          ['MSSM', 'SM_with_EW', 'SM_broken_EW', 'confined_QCD'],
    'alpha_s':          ['MSSM', 'SM_with_EW', 'SM_broken_EW', 'confined_QCD'],
    'sin2_theta_W':     ['MSSM', 'SM_with_EW', 'SM_broken_EW'],
    'alpha_EM':         ['SM_broken_EW', 'confined_QCD'],
    'M_Z':              ['SM_broken_EW', 'confined_QCD'],
    'M_W':              ['SM_broken_EW', 'confined_QCD'],
    'M_Higgs':          ['SM_broken_EW', 'confined_QCD'],
    'v_Higgs':          ['SM_broken_EW', 'confined_QCD'],  # technically defined at EW transition
    'y_tau':            ['MSSM', 'SM_with_EW', 'SM_broken_EW', 'confined_QCD'],
    'V_us':             ['MSSM', 'SM_with_EW', 'SM_broken_EW', 'confined_QCD'],
    'V_cb':             ['MSSM', 'SM_with_EW', 'SM_broken_EW', 'confined_QCD'],
    'm_e':              ['SM_broken_EW', 'confined_QCD'],  # need EW broken for mass
    'm_mu':             ['SM_broken_EW', 'confined_QCD'],
    'm_tau':            ['SM_broken_EW', 'confined_QCD'],
    'eta_B':            ['MSSM', 'SM_with_EW', 'SM_broken_EW', 'confined_QCD'],  # cosmological
    'Q_Koide':          ['MSSM', 'SM_with_EW', 'SM_broken_EW', 'confined_QCD'],  # structural
    'epsilon_CP':       ['MSSM', 'SM_with_EW', 'SM_broken_EW', 'confined_QCD'],
}


def observable_defined_at(observable_name, scale_GeV):
    """Return True if observable is defined at the given scale."""
    regime = active_regime_at(scale_GeV)['name']
    if observable_name not in OBSERVABLE_REGIMES:
        return None  # unknown observable
    return regime in OBSERVABLE_REGIMES[observable_name]


# ============================================================================
# SNAPSHOT — the exploration interface
# ============================================================================

def exploration_at(scale_GeV, alpha_gut=ALPHA_GUT, m_unif=M_UNIF_GeV,
                   m_susy=M_SUSY_GeV, m_z=M_Z_GeV):
    """Return a snapshot of the framework's state at given energy scale.

    Includes:
      - regime, gauge group, active matter content
      - running coupling values (with Undefined sentinels where not applicable)
      - which named SM observables are defined at this scale
      - anchor-status disclaimer
    """
    regime = active_regime_at(scale_GeV)
    couplings = gauge_couplings_at(scale_GeV, alpha_gut, m_unif, m_susy, m_z)

    defined_observables = [
        name for name in OBSERVABLE_REGIMES
        if observable_defined_at(name, scale_GeV)
    ]
    undefined_observables = [
        name for name in OBSERVABLE_REGIMES
        if not observable_defined_at(name, scale_GeV)
    ]

    return {
        'scale_GeV': scale_GeV,
        'regime': regime['name'],
        'gauge_group': regime['gauge_group'],
        'matter_content': regime['matter'],
        'beta_function_set': regime['beta_function_set'],
        'breaking_at_lower_boundary': regime['breaking_at_lower'],
        'couplings': couplings,
        'defined_observables': defined_observables,
        'undefined_observables': undefined_observables,
        'anchor_status': {
            'substrate_native': ['alpha_GUT (=1/24)', 'M_unif (cascade theorem)'],
            'N_hub_anchored': ['v_Higgs (← N_hub via BZJ)', 'M_Z (self-consistent EW)',
                                'M_EW', 'M_QCD'],
            'externally_assumed': ['M_SUSY (~1 TeV; not derived)'],
            'caveat': 'Numerical values below M_unif inherit the N_hub-anchor '
                      'cluster. Per 2026-05-10 user direction: "don\'t expect '
                      'accurate numbers until we run purely on N_hub."'
        }
    }
