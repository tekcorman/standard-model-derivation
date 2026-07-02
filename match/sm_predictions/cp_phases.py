"""
CP-phase + dark-coefficient predictions — counting-first queries.

Covers:
- δ_CP_CKM (Family 5 — tetrahedral dihedral on K_4 (-1)-eigenspace)
- θ_QCD = 0 (Family 5 — Z_3 holonomy flatness on srs)
- PMNS Majorana phases α_21, α_31 (Family 5 — Berry winding of arg(h))
- ε_CP, A_hemis (Family 6 — Bayesian birth/death asymmetry)
"""

import math
from fractions import Fraction
from simulator.srs_engine.kernel import CountingKernel
from simulator.srs_engine.utils import GeometricPhaseUtility


# ============================================================================
# CP-violating phases (Family 5 — geometric phases)
# ============================================================================

def delta_CP_CKM(kernel=None):
    """δ_CP_CKM = arccos(1/3) ≈ 70.53° — CKM CP-violating phase.
    CHANNEL-SELECTED (K_4 polytope dihedral class).

    Waterfilling-correct derivation: the K_4 adjacency matrix has spectrum
    {+3, -1, -1, -1}; its eigenspaces support distinct polytope geometries
    that are each above-waterline for different observables:

        (+3) Perron eigenspace:  1-dim, trivial — no dihedral
        (-1) eigenspace:         3-dim, regular-tetrahedron dihedrals
                                 with cos β = 1/3 (= 1/k* on srs's 3-coord)
        K_4 (+1) eigenvectors:   no such eigenvectors exist for K_4 alone

    The CKM CP-violating phase is the polytope dihedral on the K_4
    (-1)-eigenspace; channel = `k4_minus_eigenspace_dihedral`.
    Alternative readings (e.g., trivial Perron 1-dim) are physically
    realized but for different observables (Perron-dominant amplitudes).
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {
            'name': 'K_4 (-1)-eigenspace dihedral (regular tetrahedron, cos = 1/k*)',
            'channel': 'k4_minus_eigenspace_dihedral',
            'reading': GeometricPhaseUtility.k4_minus_eigenspace_dihedral,
        },
        {
            'name': 'K_4 Perron (+3)-eigenspace (trivial; no dihedral)',
            'channel': 'k4_perron_eigenspace_trivial',
            'reading': None,
        },
    ]
    selected = kernel.channel_select(
        candidates, channel='k4_minus_eigenspace_dihedral')
    return selected['reading']()['degrees']  # ≈ 70.53


def theta_QCD(kernel=None):
    """θ_QCD = 0 — strong CP problem solved. CHANNEL-SELECTED (Z_3 connection class).

    Waterfilling-correct derivation: Z_3 connections on srs cycles fall into
    physically distinct classes that are each above-waterline:

        flat connection on (3,10)-cage:   all girth-{10,12,14,...} holonomies
                                          vanish (R3 verified in
                                          proofs/flavor/z3_holonomy_cycles.py)
        curved Z_3 connection candidates: would give non-zero holonomy on
                                          some cycle class; NOT realized on
                                          srs because the bundle is globally
                                          trivializable.

    The strong CP channel is `flat_z3_connection_on_3_10_cage`. The
    framework's substrate admits only the flat connection above the
    waterline, so this channel selects the single attested candidate.
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {
            'name': 'flat Z_3 connection on srs (R3 theorem)',
            'channel': 'flat_z3_connection_on_3_10_cage',
            'reading': GeometricPhaseUtility.z3_holonomy_flat,
        },
    ]
    selected = kernel.channel_select(
        candidates, channel='flat_z3_connection_on_3_10_cage')
    return selected['reading']()['phase_rad']  # 0


def alpha_21_PMNS(kernel=None):
    """α_21 PMNS Majorana phase = g · arg(h) mod 360° ≈ 162.39°.
    CHANNEL-SELECTED (walker phase winding, n = g).

    Waterfilling-correct derivation: walker phase windings n × arg(h) are
    each above-waterline for different observables — Bloch-walker holonomies
    accumulate at every integer multiple of the Hashimoto Perron phase.
    Per Path B (UNIQUE-THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1):
    cycle-confined transfer operator forces the Majorana α_21 winding
    multiplier to be n = g (girth) — the closed-cycle walker holonomy.

    Channel: `walker_phase_winding_n_eq_g`. Other windings (single-step,
    double-step, etc.) couple to other observables (Berry phases on
    shorter cycles, propagators).
    """
    kernel = kernel or CountingKernel()
    g = kernel.substrate.GIRTH                # 10 — Galois winding = girth
    candidates = [
        {'name': 'n=1 single-step', 'channel': 'walker_phase_winding_n_eq_1', 'n': 1},
        {'name': 'n=2 double-step', 'channel': 'walker_phase_winding_n_eq_2', 'n': 2},
        {'name': f'n=g={g} girth winding (α_21)',
         'channel': 'walker_phase_winding_n_eq_g', 'n': g},
        {'name': f'n=2g={2*g} double-girth winding (α_31)',
         'channel': 'walker_phase_winding_n_eq_2g', 'n': 2 * g},
    ]
    selected = kernel.channel_select(
        candidates, channel='walker_phase_winding_n_eq_g')
    h = kernel.substrate.ramanujan_eigenvalue_at_P
    arg_h_deg = math.degrees(math.atan2(h.imag, h.real))
    return (selected['n'] * arg_h_deg) % 360


def alpha_31_PMNS(kernel=None):
    """α_31 PMNS Majorana phase = 2g · arg(h) mod 360° ≈ 324.78°.
    CHANNEL-SELECTED (walker phase winding, n = 2g).

    Same Path B chain as α_21 with winding multiplier n = 2g (double-girth);
    channel = `walker_phase_winding_n_eq_2g`.
    """
    kernel = kernel or CountingKernel()
    g = kernel.substrate.GIRTH
    candidates = [
        {'name': 'n=1 single-step', 'channel': 'walker_phase_winding_n_eq_1', 'n': 1},
        {'name': 'n=2 double-step', 'channel': 'walker_phase_winding_n_eq_2', 'n': 2},
        {'name': f'n=g={g} girth winding (α_21)',
         'channel': 'walker_phase_winding_n_eq_g', 'n': g},
        {'name': f'n=2g={2*g} double-girth winding (α_31)',
         'channel': 'walker_phase_winding_n_eq_2g', 'n': 2 * g},
    ]
    selected = kernel.channel_select(
        candidates, channel='walker_phase_winding_n_eq_2g')
    h = kernel.substrate.ramanujan_eigenvalue_at_P
    arg_h_deg = math.degrees(math.atan2(h.imag, h.real))
    return (selected['n'] * arg_h_deg) % 360


# ============================================================================
# Bayesian birth/death asymmetries (Family 6)
# ============================================================================

def epsilon_CP(kernel=None):
    """ε_CP = 1/5 — Bayesian birth-death asymmetry. CHANNEL-SELECTED.

    Waterfilling-correct derivation: Bayesian conjugate updates on a
    Bernoulli edge-existence parameter yield a family of (p_create,
    p_destroy) pairs, each above-waterline for a different observable:

        Beta(1,1) uniform prior:       p_c = 1/2, p_d = 1/2, ε = 0
                                        (no asymmetry — pre-observation)
        Beta(2,1) one-confirmation:    p_c = 1/2, p_d = 1/3, ε = 1/5
                                        (the cosmological arrow of time)
        Beta(1,2) one-disconfirmation: p_c = 1/3, p_d = 1/2, ε = -1/5
                                        (the time-reversal partner)

    ε_CP is the Sakharov CP-violation asymmetry — the framework's arrow
    of time. Its substrate channel is `beta_2_1_one_confirmation_posterior`:
    once an edge has been observed to exist (Beta(1,1) → Beta(2,1) update),
    the next observation's surprise asymmetry is set by the Beta(2,1)
    posterior. Result: p_c = 1/2, p_d = 1/k* = 1/3 (uniform over k* options
    after one confirmation), ε = (p_c − p_d)/(p_c + p_d) = (1/6)/(5/6) = 1/5.

    The Beta(1,1) and Beta(1,2) candidates are above-waterline Bayesian
    states physically realized at different points in the cosmic evolution
    (pre-observation, post-disconfirmation); they are not discarded.
    channel_select picks Beta(2,1) by name match — the cosmological-arrow
    state of the framework's substrate.
    """
    kernel = kernel or CountingKernel()
    rates = kernel.toggle_markov()
    k_star = kernel.substrate.K_STAR

    candidates = [
        {
            'name': 'Beta(1,1) uniform prior (pre-observation)',
            'channel': 'beta_1_1_uniform_prior',
            'p_create': Fraction(1, 2),
            'p_destroy': Fraction(1, 2),
        },
        {
            'name': 'Beta(2,1) one-confirmation posterior (cosmological arrow)',
            'channel': 'beta_2_1_one_confirmation_posterior',
            'p_create': rates['p_create'],            # 1/2
            'p_destroy': rates['p_destroy'],          # 1/k* = 1/3
        },
        {
            'name': 'Beta(1,2) one-disconfirmation posterior (time-reversal)',
            'channel': 'beta_1_2_one_disconfirmation_posterior',
            'p_create': Fraction(1, k_star),          # 1/3
            'p_destroy': Fraction(1, 2),
        },
    ]
    selected = kernel.channel_select(
        candidates,
        channel='beta_2_1_one_confirmation_posterior',
    )
    p_c = selected['p_create']
    p_d = selected['p_destroy']
    return (p_c - p_d) / (p_c + p_d)  # 1/5


def A_hemispherical(kernel=None):
    """A_hemis = 1/15 — CMB hemispherical asymmetry.

    Counting query: ε_CP × (1/k*) = (1/5) × (1/3) = 1/15.
    """
    kernel = kernel or CountingKernel()
    eps = epsilon_CP(kernel)
    return eps * Fraction(1, kernel.substrate.K_STAR)  # 1/15


# ============================================================================
# 2026-05-08 additions — δ_CP_PMNS revival + β cosmic birefringence
# ============================================================================

# α_EM(0) Thomson — used by β cosmic birefringence and R∞.
# Standard QED running below M_Z is a Type 3 framework anchor: α_EM(0)⁻¹ ≈
# α_EM(M_Z)⁻¹ + 9.91 from charged-fermion thresholds. The framework's α_EM(0)
# round-trips to ~1/137.036.
_DELTA_ALPHA_RUN = 9.91  # standard QED running M_Z → 0


def _alpha_EM_thomson(alpha_EM_MZ):
    return 1.0 / (1.0 / alpha_EM_MZ + _DELTA_ALPHA_RUN)


def delta_CP_PMNS(kernel=None):
    """δ_CP_PMNS = arccos(T_{B-L,lepton}) = arccos(-1) = 180°.
    CHANNEL-SELECTED (T_{B-L} eigenvalue axis on K_4 (-1)-eigenspace).

    Waterfilling-correct derivation (revival 2026-05-08): under the
    V_{−1}–T_{B-L} symmetry-breaking identity + SO(3)_{K4} → SO(2)_u bridge,
    the polar angle of a (B-L)-eigenvector on the K_4 (-1)-eigenspace is
    arccos(T_{B-L}). The three above-waterline (B-L)-eigenvector classes:

        T_{B-L} = -1 (lepton axis):   arccos(-1)  = 180°  ← δ_CP_PMNS
        T_{B-L} = +1/3 (color axis):  arccos(1/3) ≈ 70.5° ← δ_CP_CKM
        T_{B-L} = 0 (neutral):        arccos(0)   = 90°

    δ_CP_PMNS selects the LEPTON axis; channel = `pmns_t_bl_lepton_axis`.
    THEOREM-GRADE-STRUCTURAL; conditional on framework-wide CKM-↔-K_4-walks
    identification (Other-Smuggle, gated on Need-D-3 alone post-Need-A2
    closure 2026-05-08).
    """
    kernel = kernel or CountingKernel()
    k_star = kernel.substrate.K_STAR
    candidates = [
        {'name': 'lepton axis: T_BL = -1',  'channel': 'pmns_t_bl_lepton_axis',  'T_BL': -1},
        {'name': f'color axis: T_BL = 1/k* = 1/{k_star}',
         'channel': 'ckm_t_bl_color_axis',  'T_BL': Fraction(1, k_star)},
        {'name': 'neutral axis: T_BL = 0',  'channel': 'pmns_t_bl_neutral_axis', 'T_BL': Fraction(0)},
    ]
    selected = kernel.channel_select(candidates, channel='pmns_t_bl_lepton_axis')
    return math.degrees(math.acos(float(selected['T_BL'])))


def beta_cosmic_birefringence(kernel=None):
    """β cosmic birefringence ≈ 0.331° — CMB EB cross-correlation rotation.
    CHANNEL-SELECTED (h-functional coupling class, c=1).

    Waterfilling-correct derivation (UNIQUE-THEOREM-GRADE 2026-04-29):
    h-functional readings of the Hashimoto Perron eigenvalue are each
    above-waterline; the two relevant classes for EM phase observables are:

        c=1 walker-phase coupling:        β = sin(arg h) · α_EM
        c=arg(h)/π Berry-phase coupling:  alternative algebraic form

    The CMB EB cross-correlation observable selects c=1 (theorem-grade
    via uniqueness template + algebraicity meta-theorem). Channel:
    `walker_phase_h_functional_c_eq_1`.
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {'name': 'walker-phase coupling c=1', 'channel': 'walker_phase_h_functional_c_eq_1', 'c': 1.0},
        # Above-waterline alternative h-functional readings exist but
        # couple to different observables (Berry-phase corrections etc.)
    ]
    selected = kernel.channel_select(
        candidates, channel='walker_phase_h_functional_c_eq_1')
    h = kernel.substrate.ramanujan_eigenvalue_at_P
    sin_arg_h = h.imag / abs(h)
    # α_EM at framework CMB epoch — use Thomson limit (running below M_Z)
    from .rg_flow import alpha_EM as _alpha_EM_at_MZ
    alpha_EM_0 = _alpha_EM_thomson(_alpha_EM_at_MZ(kernel))
    beta_rad = selected['c'] * sin_arg_h * alpha_EM_0
    return math.degrees(beta_rad)
