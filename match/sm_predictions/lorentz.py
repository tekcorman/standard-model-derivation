"""
Lorentz / Bloch-dispersion predictions — Family 3 extensions.

CHANNEL-SELECTED throughout: each prediction enumerates above-waterline
alternatives (different Bloch-Taylor orders, different screw-axis classes,
different cubic-moment orders) and channel_selects the observable-matching
one.

Iorio β / G_sub matter-loop coefficient is NOT wrapped here — G_sub is
STRUCTURALLY OPEN at the flat-band IR (cf.
proofs/foundations/lorentz_sig_g_sub_iorio_closure.py).
"""

import math
from fractions import Fraction

from simulator.srs_engine.kernel import CountingKernel


# ============================================================================
# Bloch-Taylor coefficients (4th-order at Γ)
# ============================================================================

# Bloch-Taylor candidate enumeration is shared between D4_iso/D4_aniso/η_NB
# — same kernel call, channel selects the Taylor coefficient.
def _bloch_taylor_candidates():
    return [
        {'name': 'order-2 (k²): D2',
         'channel': 'scalar_bloch_taylor_order_2',
         'key': 'D2'},
        {'name': 'order-4 (k⁴) isotropic: D4_iso',
         'channel': 'scalar_bloch_taylor_order_4_iso',
         'key': 'D4_iso'},
        {'name': 'order-4 (k⁴) anisotropic: D4_aniso',
         'channel': 'scalar_bloch_taylor_order_4_aniso',
         'key': 'D4_aniso'},
        {'name': 'dim-6 LV scalar walker: η_NB^H = D4_aniso/D2²',
         'channel': 'scalar_walker_dim6_LV',
         'key': 'eta_NB_H'},
    ]


def D4_iso_H(kernel=None):
    """D4_iso^H = -1/1024 — k⁴ isotropic Bloch-Taylor coefficient.
    CHANNEL-SELECTED (Bloch-Taylor order-4 isotropic).
    """
    kernel = kernel or CountingKernel()
    selected = kernel.channel_select(
        _bloch_taylor_candidates(), channel='scalar_bloch_taylor_order_4_iso')
    return kernel.bloch_taylor_at_gamma(order=4)[selected['key']]


def D4_aniso_H(kernel=None):
    """D4_aniso^H = +1/1536 — k⁴ cubic-anisotropic Bloch-Taylor coefficient.
    CHANNEL-SELECTED (Bloch-Taylor order-4 anisotropic).
    """
    kernel = kernel or CountingKernel()
    selected = kernel.channel_select(
        _bloch_taylor_candidates(), channel='scalar_bloch_taylor_order_4_aniso')
    return kernel.bloch_taylor_at_gamma(order=4)[selected['key']]


def eta_NB_H(kernel=None):
    """η_NB^H = D4_aniso/D2² = 1/6 — scalar-Bloch dim-6 LV coefficient.
    CHANNEL-SELECTED (scalar-walker dim-6 LV).

    Equals 2 × η_lattice (Hashimoto NB walker = 1/12) by the Ihara
    cross-walker identity h'(3) = 2.
    """
    kernel = kernel or CountingKernel()
    selected = kernel.channel_select(
        _bloch_taylor_candidates(), channel='scalar_walker_dim6_LV')
    return kernel.bloch_taylor_at_gamma(order=4)[selected['key']]


# ============================================================================
# Screw-axis Wigner content (I4_132 4_1 screw axis)
# ============================================================================

def _screw_axis_candidates(k_star):
    """Above-waterline screw-axis classes by space-group screw type.

    The srs lattice realizes I4_132's 4_1 screw axis (4-fold rotation +
    1/4 translation along body-diagonal). Other space groups support
    different screw-axes (2_1, 3_1, 4_1, 4_3, 6_1, 6_5, …) above-waterline
    for their respective Coxeter-quotient retentions.
    """
    return [
        {'name': '4_1 screw axis (I4_132, srs realization)',
         'channel': 'i4_132_screw_axis_4_1',
         'cos_beta': Fraction(k_star - 2, k_star)},  # = 1/k* = 1/3 on srs
        {'name': '2_1 screw axis (simpler space groups)',
         'channel': 'screw_axis_2_1',
         'cos_beta': Fraction(0)},
        {'name': '3_1 screw axis (hexagonal lattices)',
         'channel': 'screw_axis_3_1',
         'cos_beta': Fraction(-1, 2)},
    ]


def screw_wigner_cos_beta(kernel=None):
    """cos β = (k*-2)/k* = 1/3 on srs — body-diagonal C_3 screw-axis tilt.
    CHANNEL-SELECTED (I4_132 4_1 screw axis).
    """
    kernel = kernel or CountingKernel()
    k = kernel.substrate.K_STAR
    selected = kernel.channel_select(
        _screw_axis_candidates(k), channel='i4_132_screw_axis_4_1')
    return selected['cos_beta']


def screw_wigner_beta_deg(kernel=None):
    """β = arccos(1/3) ≈ 70.53° — screw-axis dihedral angle. DERIVED."""
    return math.degrees(math.acos(float(screw_wigner_cos_beta(kernel))))


def screw_wigner_d1_diag(kernel=None):
    """Wigner d¹ diagonal amplitudes (d¹_{±1,±1}, d¹_{00}) at the screw tilt.

    = ((1+cos β)/2, cos β) = (2/3, 1/3) on srs.
    """
    cos_b = screw_wigner_cos_beta(kernel)
    return ((Fraction(1) + cos_b) / 2, cos_b)


def screw_wigner_survival(kernel=None):
    """Survival probabilities (P_{±1}, P_0) = (4/9, 1/9) at the screw tilt.

    Counting query: |d¹_{m,m}|² for diagonal Wigner amplitudes; the harmonic
    mean HM(4/9, 1/9, 4/9) = 2/9 algebraically (matches δ_Koide value).
    """
    d_pm, d_0 = screw_wigner_d1_diag(kernel)
    return (d_pm * d_pm, d_0 * d_0)


# ============================================================================
# Cubic-axis directed-edge moments
# ============================================================================

def srs_cubic_moment(n=1, kernel=None):
    """⟨(e·ẑ)^{2n}⟩ = 1/(k* · 2^(n-1)) — directed-edge cubic-axis 2n-th moment.
    CHANNEL-SELECTED (I4_132 directed-edge moment, order n).

    Waterfilling-correct derivation: the 24 directed ⟨110⟩-edges per
    I4_132 conventional cell partition 8 + 16 under projection on a principal
    cubic axis (e·ẑ ∈ {0, ±1/√2}). Different moment orders n = 1, 2, 3, …
    are each above-waterline for different observables (η_5 vanishing,
    η_lattice, higher LV coefficients).
    """
    kernel = kernel or CountingKernel()
    k = kernel.substrate.K_STAR
    candidates = [
        {'name': f'directed-edge moment, n={ni}',
         'channel': f'i4_132_directed_edge_moment_order_{ni}',
         'value': Fraction(1, k * 2 ** (ni - 1))}
        for ni in range(1, 7)
    ]
    selected = kernel.channel_select(
        candidates, channel=f'i4_132_directed_edge_moment_order_{n}')
    return selected['value']
