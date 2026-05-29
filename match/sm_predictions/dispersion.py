"""
Dispersion / kinematics predictions — Family 3 (Bloch-gradient).

Counting-first queries for the framework's "travel speeds and bending"
family: Fermi velocities at Dirac points, Lorentz-violation bounds,
dim-5 and dim-6 LV coefficients.

All five predictions in this file are CHANNEL-SELECTED: each enumerates
above-waterline alternatives (different k-points for v_F, different
walker classes for η, different Taylor orders for D_H) and picks the
observable-matching channel.
"""

import math
from fractions import Fraction

from simulator.srs_engine.kernel import CountingKernel


def v_F_Gamma(kernel=None):
    """v_F at Γ-cone = 1/2 — Fermi velocity at the Γ-point Dirac cone.
    CHANNEL-SELECTED (Dirac cone at k = Γ).

    Waterfilling-correct derivation: the substrate adjacency Bloch operator
    supports Dirac cones at multiple high-symmetry k-points; each is
    above-waterline for a different observable:

        Γ-cone (k=0):  lower 3 bands triple-degenerate at λ=-1 (= K_4 spectrum,
                       Biggs 1993 §2.2); spin-1 Dirac splitting with v_F = 1/2.
        P-cone (k=P):  2-fold cluster at λ=±√3; 2-band Dirac with v_F = √3/6.
        N, H cones:    higher-symmetry k-points; not realized as cones on srs.

    v_F_Gamma selects the Γ-point cone; channel = `dirac_cone_at_gamma`.
    THEOREM-GRADE via `proofs/foundations/lorentz_sig_dirac_cone_refined.py`.
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {'name': 'Γ-cone (spin-1 Dirac, λ=-1 cluster)',
         'channel': 'dirac_cone_at_gamma',
         'k_point': 'Gamma',
         'deg_indices': (0, 1, 2)},
        {'name': 'P-cone (2-band Dirac, λ=±√3 cluster)',
         'channel': 'dirac_cone_at_p',
         'k_point': 'P',
         'deg_indices': (2, 3)},
    ]
    selected = kernel.channel_select(candidates, channel='dirac_cone_at_gamma')
    return kernel.dirac_cone_velocity(selected['k_point'], selected['deg_indices'])


def v_F_P(kernel=None):
    """v_F at P-cone = √3/6 — Fermi velocity at the P-point Dirac cone.
    CHANNEL-SELECTED (Dirac cone at k = P).

    Same enumeration as v_F_Gamma; channel = `dirac_cone_at_p`.
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {'name': 'Γ-cone (spin-1 Dirac, λ=-1 cluster)',
         'channel': 'dirac_cone_at_gamma',
         'k_point': 'Gamma',
         'deg_indices': (0, 1, 2)},
        {'name': 'P-cone (2-band Dirac, λ=±√3 cluster)',
         'channel': 'dirac_cone_at_p',
         'k_point': 'P',
         'deg_indices': (2, 3)},
    ]
    selected = kernel.channel_select(candidates, channel='dirac_cone_at_p')
    return kernel.dirac_cone_velocity(selected['k_point'], selected['deg_indices'])


def eta_5(kernel=None):
    """η_5 = 0 — dim-5 Lorentz-violation coefficient.
    CHANNEL-SELECTED (parity-even Bloch dispersion).

    Waterfilling-correct derivation: undirected-graph symmetry B(−k) = B(k)*
    forces the substrate Bloch dispersion to be REAL and EVEN in k. The
    two above-waterline candidates for the dispersion's parity class are:

        parity-even:  λ(-k) = λ(k); odd-power Taylor coefficients vanish.
        parity-odd:   would require directed-graph asymmetry; NOT realized
                      on srs (Hashimoto symmetry B(−k) = B(k)*).

    η_5 is the O(k³) coefficient in the photon dispersion ⇒ vanishes under
    parity-even; channel = `parity_even_substrate_dispersion`.
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {'name': 'parity-even (undirected-graph symmetry)',
         'channel': 'parity_even_substrate_dispersion',
         'odd_coeff': 0},
    ]
    selected = kernel.channel_select(
        candidates, channel='parity_even_substrate_dispersion')
    return selected['odd_coeff']


def eta_lattice(kernel=None):
    """η_lattice = 1/12 — dim-6 LV coefficient of the Hashimoto NB-walker.
    CHANNEL-SELECTED (NB walker on directed edges).

    Waterfilling-correct derivation: dim-6 LV coefficients are walker-class
    specific; each walker class is above-waterline for a different observable:

        scalar adjacency walker (on vertices):  η_NB^H = D4_aniso/D2² = 1/6
                                                (lorentz.eta_NB_H)
        Hashimoto NB walker (directed edges):   η_lattice = (Ihara cross-walker)
                                                D4_aniso_NB / D_NB² = 1/12

    The Ihara cross-walker theorem (Ihara 1966, Stark-Terras 1996) relates
    them via h'(3) = 2 for 3-regular graphs:
        D_NB        = h'(3) × D2          = 2 × 1/16   = 1/8
        D4_aniso_NB = h'(3) × D4_aniso    = 2 × 1/1536 = 1/768
        η_lattice   = (1/768) / (1/8)²    = 1/12

    η_lattice (photon LIV observable) selects the NB walker;
    channel = `nb_walker_dim6_LV`.
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {'name': 'scalar adjacency walker (η_NB^H)',
         'channel': 'scalar_adjacency_walker_dim6_LV',
         'h_prime': 1},
        {'name': 'Hashimoto NB walker (η_lattice)',
         'channel': 'nb_walker_dim6_LV',
         'h_prime': 2},
    ]
    selected = kernel.channel_select(candidates, channel='nb_walker_dim6_LV')
    coeffs = kernel.bloch_taylor_at_gamma(order=4)
    h_prime = selected['h_prime']
    D_walker = h_prime * coeffs['D2']
    D4_aniso_walker = h_prime * coeffs['D4_aniso']
    return D4_aniso_walker / (D_walker * D_walker)  # 1/12


def D_H(kernel=None):
    """D_H = 1/16 — substrate Bloch-Hamiltonian k² coefficient at Γ.
    CHANNEL-SELECTED (2nd-order Bloch-Taylor coefficient).

    Waterfilling-correct derivation: the Bloch-Taylor expansion of λ_max(k)
    around Γ yields a sequence of coefficients each above-waterline at its
    own Taylor order:

        order 2 (k²):   D2 = 1/16
        order 4 (k⁴):   D4_iso = -1/1024,  D4_aniso = +1/1536
        order 6 (k⁶):   higher-order, not yet enumerated

    D_H is the 2nd-order coefficient; channel = `scalar_bloch_taylor_order_2`.
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {'name': 'order 2 (k²): D2',
         'channel': 'scalar_bloch_taylor_order_2',
         'key': 'D2'},
        {'name': 'order 4 (k⁴) isotropic: D4_iso',
         'channel': 'scalar_bloch_taylor_order_4_iso',
         'key': 'D4_iso'},
        {'name': 'order 4 (k⁴) anisotropic: D4_aniso',
         'channel': 'scalar_bloch_taylor_order_4_aniso',
         'key': 'D4_aniso'},
    ]
    selected = kernel.channel_select(
        candidates, channel='scalar_bloch_taylor_order_2')
    return kernel.bloch_taylor_at_gamma(order=4)[selected['key']]
