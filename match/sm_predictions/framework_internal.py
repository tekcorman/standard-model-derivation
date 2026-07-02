"""
Framework-internal predictions — substrate observables.

These are constants the framework derives that aren't directly SM observables
but are load-bearing for derivation chains.

CHANNEL-SELECTED throughout: each prediction enumerates above-waterline
alternatives (Bayesian posteriors, Bloch eigenvalues at different fibers,
Feshbach n_fixed cases) and channel_selects the observable-matching one.
"""

import math
from fractions import Fraction

from simulator.srs_engine.kernel import CountingKernel


# ============================================================================
# Information-theoretic units (A1 axiom + natural-unit identification)
# ============================================================================

def p_toggle(kernel=None):
    """p_toggle = 2 — alphabet size of the binary toggle.
    CHANNEL-SELECTED (A1 binary-toggle axiom).

    Waterfilling-correct derivation: alphabet sizes |Σ| = 2, 3, 4, … are
    each above-waterline as substrate primitives; A1 (self-containment +
    finite observer + active reading) selects |Σ| = 2 structurally per
    `theorem_toggle_from_self_containment.md`.
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {'name': 'A1 binary toggle (|Σ|=2)',
         'channel': 'a1_binary_toggle', 'p': 2},
        {'name': 'ternary alphabet (|Σ|=3)',
         'channel': 'ternary_alphabet', 'p': 3},
        {'name': 'k-ary alphabet (general |Σ|=k)',
         'channel': 'k_ary_alphabet', 'p': None},
    ]
    selected = kernel.channel_select(candidates, channel='a1_binary_toggle')
    return selected['p']


def e_bit(kernel=None):
    """e_bit = 1.0 — natural-unit "edge bit" cost.
    CHANNEL-SELECTED (binary-toggle natural-unit convention).

    log₂(|Σ|) for |Σ| = 2 = 1 bit per toggle event. Definitional under A1.
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {'name': 'binary toggle log₂(2) = 1 bit',
         'channel': 'a1_binary_toggle', 'bits': 1.0},
        {'name': 'ternary toggle log₂(3)',
         'channel': 'ternary_alphabet', 'bits': math.log2(3.0)},
    ]
    selected = kernel.channel_select(candidates, channel='a1_binary_toggle')
    return selected['bits']


# ============================================================================
# Bayesian Beta-update channel family (Stage 2a)
# ============================================================================

def _beta_posterior_candidates(kernel):
    """Above-waterline Beta-posterior candidates per Stage 2a."""
    rates = kernel.toggle_markov()
    return [
        {'name': 'Beta(1,1) uniform prior (pre-observation)',
         'channel': 'beta_1_1_uniform_prior',
         'p_obs': Fraction(1, 2)},
        {'name': 'Beta(2,1) one-confirmation posterior',
         'channel': 'beta_2_1_one_confirmation_posterior',
         'p_obs': rates['p_create']},     # 1/2
        {'name': 'Beta(2,1) destruction = 1/(α+β) = 1/3',
         'channel': 'beta_2_1_one_disconfirm_predictive',
         'p_obs': rates['p_destroy']},    # 1/3
    ]


def S_fresh(kernel=None):
    """S_fresh = 1 bit — surprise of observing a fresh pattern.
    CHANNEL-SELECTED (Beta(1,1) uniform prior).

    Waterfilling-correct: Bayesian Beta(α,β) posteriors give different
    predictive probabilities — each posterior is above-waterline at a
    different stage of observation. S_fresh = -log₂(P_predictive) for
    Beta(1,1) (no prior data): P = 1/2 ⇒ S = 1 bit.
    """
    kernel = kernel or CountingKernel()
    selected = kernel.channel_select(
        _beta_posterior_candidates(kernel),
        channel='beta_1_1_uniform_prior')
    return -math.log2(float(selected['p_obs']))


def S_disconfirm(kernel=None):
    """S_disconfirm = log₂(3) bits — surprise of disconfirming a pattern.
    CHANNEL-SELECTED (Beta(2,1) one-disconfirmation predictive).

    After one confirmed observation Beta(1,1) → Beta(2,1), the predictive
    probability of "absent" is β/(α+β) = 1/3. Surprise = log₂(3).
    Channel: `beta_2_1_one_disconfirm_predictive`.
    """
    kernel = kernel or CountingKernel()
    selected = kernel.channel_select(
        _beta_posterior_candidates(kernel),
        channel='beta_2_1_one_disconfirm_predictive')
    return -math.log2(float(selected['p_obs']))


def asymmetry_bits(kernel=None):
    """Asymmetry = S_disconfirm - S_fresh = log₂(3/2) bits > 0.
    CHANNEL-SELECTED (Beta(2,1) − Beta(1,1) difference)."""
    return S_disconfirm(kernel) - S_fresh(kernel)


def lambda_toggle_rate(kernel=None):
    """λ_toggle = 2/5 — net toggle "on" rate of the renewal Markov.
    CHANNEL-SELECTED (Beta(2,1) one-confirmation posterior; harmonic mean).

    p_create = 1/2 (Beta(2,1)), p_destroy = 1/3 (Beta(2,1) on k* options).
    λ = 2·p_c·p_d / (p_c + p_d) = 2/5.
    """
    kernel = kernel or CountingKernel()
    rates = kernel.toggle_markov()
    candidates = [
        {'name': 'Beta(2,1) renewal Markov',
         'channel': 'beta_2_1_renewal_markov',
         'p_create': rates['p_create'],   # 1/2
         'p_destroy': rates['p_destroy']},  # 1/3
    ]
    selected = kernel.channel_select(
        candidates, channel='beta_2_1_renewal_markov')
    pc, pd = selected['p_create'], selected['p_destroy']
    return Fraction(2) * pc * pd / (pc + pd)


def xi_t_temporal_correlation(kernel=None):
    """ξ_t = 1/log(6) ≈ 0.558 ℓ_P — temporal correlation length.
    CHANNEL-SELECTED (Beta(2,1) renewal Markov, autocorrelation decay).

    Non-trivial eigenvalue r = tr(M) - 1 = 1/6 of the 2-state Markov on
    the per-edge toggle ⇒ exponential decay with ξ_t = 1/log(1/r) = 1/log(6).
    """
    kernel = kernel or CountingKernel()
    rates = kernel.toggle_markov()
    candidates = [
        {'name': 'Beta(2,1) renewal Markov, 2-state spectrum',
         'channel': 'beta_2_1_renewal_markov',
         'p_create': rates['p_create'],
         'p_destroy': rates['p_destroy']},
    ]
    selected = kernel.channel_select(
        candidates, channel='beta_2_1_renewal_markov')
    pc, pd = selected['p_create'], selected['p_destroy']
    r = (1 - pc) + (1 - pd) - 1  # = 1/6
    return 1.0 / math.log(1.0 / float(r))


# ============================================================================
# Bloch / spectral primitives (substrate eigenvalues)
# ============================================================================

def srs_E_at_P(kernel=None):
    """E_P = √k* = √3 — adjacency eigenvalue at the P-point.
    CHANNEL-SELECTED (adjacency Perron at fiber P).

    Waterfilling-correct: substrate adjacency Bloch operator has spectra
    at every BZ point; high-symmetry k-points are each above-waterline:

        Γ:   spectrum {+k*, -1, -1, -1} → Perron = k*
        P:   spectrum {+√k*, +√k*, -√k*, -√k*} → Perron = √k*
        N:   distinct spectrum
        H:   distinct spectrum

    srs_E_at_P selects fiber P; channel = `adjacency_perron_at_p`.
    """
    kernel = kernel or CountingKernel()
    k_star = kernel.substrate.K_STAR
    candidates = [
        {'name': 'adjacency Perron at Γ (= k*)',
         'channel': 'adjacency_perron_at_gamma', 'value': float(k_star)},
        {'name': 'adjacency Perron at P (= √k*)',
         'channel': 'adjacency_perron_at_p', 'value': math.sqrt(k_star)},
    ]
    selected = kernel.channel_select(
        candidates, channel='adjacency_perron_at_p')
    return selected['value']


def h_walker_eigenvalue(kernel=None):
    """h = (√3 + i√5)/2 — Hashimoto NB-walker eigenvalue at P (Ramanujan saddle).
    CHANNEL-SELECTED (Hashimoto Perron at fiber P, Ramanujan saturation).

    Waterfilling-correct: Ihara-Bass quadratic h² − E_P·h + (k*−1) = 0
    yields complex-conjugate pair (Ramanujan saddle) when discriminant is
    negative. Positive-imaginary root selected by I4_132 chirality;
    channel = `hashimoto_perron_at_p_ramanujan_positive_im`.
    """
    kernel = kernel or CountingKernel()
    k_star = kernel.substrate.K_STAR
    E_P = srs_E_at_P(kernel)
    re = E_P / 2
    im_mag = math.sqrt(4 * (k_star - 1) - E_P ** 2) / 2
    candidates = [
        {'name': 'positive-imaginary Hashimoto root (I4_132 chirality)',
         'channel': 'hashimoto_perron_at_p_ramanujan_positive_im',
         'h': complex(re, +im_mag)},
        {'name': 'negative-imaginary Hashimoto root (mirror chirality)',
         'channel': 'hashimoto_perron_at_p_ramanujan_negative_im',
         'h': complex(re, -im_mag)},
    ]
    selected = kernel.channel_select(
        candidates,
        channel='hashimoto_perron_at_p_ramanujan_positive_im')
    return selected['h']


# ============================================================================
# Planck-scale identification + cosmology anchor
# ============================================================================

def M_Pl_natural(kernel=None):
    """M_Pl = 8/√π — Planck mass in framework-natural lattice units.
    CHANNEL-SELECTED (G_N · M_Pl² = 1 unit-setting).

    Waterfilling-correct: dimensional unit-setting conventions for the
    Planck mass — each is above-waterline as a self-consistent natural-
    units choice:

        G_N · M_Pl² = 1 (Planck-units, framework default):
            G_UV · M_substrate² = π/(16·N_atoms) = π/64
            ⇒ M_Pl/M_substrate = √(64/π) = 8/√π
        G_N · M_Pl² = 8π (Cosmologist convention):  M_Pl' = √(8π/π·64) = different
        G_N · M_Pl² = 1/(8π) (Reduced Planck):       M_Pl_red = different

    Framework selects the Planck-units convention; channel =
    `g_n_planck_units_unit_setting`.
    """
    kernel = kernel or CountingKernel()
    n_atoms = kernel.substrate.N_ATOMS
    candidates = [
        {'name': 'G_N · M_Pl² = 1 (Planck units)',
         'channel': 'g_n_planck_units_unit_setting',
         'ratio_sq': 64.0 / math.pi},  # = (M_Pl / M_substrate)²
        {'name': 'G_N · M_Pl² = 8π (cosmologist)',
         'channel': 'g_n_cosmologist_unit_setting',
         'ratio_sq': 8 * math.pi * 64.0 / math.pi},
    ]
    selected = kernel.channel_select(
        candidates, channel='g_n_planck_units_unit_setting')
    return math.sqrt(selected['ratio_sq'])


def N_hub(kernel=None):
    """N_hub ≈ 8.4×10⁶⁰ — total toggle count between Planck and Hubble.
    CHANNEL-SELECTED (G_F-anchored cosmology cascade, EXTERNAL ANCHOR).

    Currently the framework anchors N_hub from G_F (Fermi constant). The
    above-waterline anchor candidates are each a different empirical
    input choice; the framework selects G_F by historical convention.

    HONEST: this is an empirical anchor, not a substrate derivation.
    """
    kernel = kernel or CountingKernel()
    from .cosmology import N_HUB
    candidates = [
        {'name': 'G_F (Fermi constant) — EXTERNAL ANCHOR',
         'channel': 'cosmology_anchor_G_F', 'value': N_HUB},
        # Other empirical-anchor candidates (M_Pl, R_∞, …) would each
        # produce a self-consistent N_hub if substituted as the anchor.
    ]
    selected = kernel.channel_select(
        candidates, channel='cosmology_anchor_G_F')
    return selected['value']


# ============================================================================
# Cubic-moment / Feshbach (re-exports for back-compat; see lorentz, masses)
# ============================================================================

def srs_cubic_moment_n1(kernel=None):
    """⟨(e·ẑ)²⟩ = 1/k* = 1/3 — first directed-edge cubic-axis moment.
    CHANNEL-SELECTED (I4_132 directed-edge moment, order n=1).

    See lorentz.srs_cubic_moment for the order-parameterized version.
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
        candidates, channel='i4_132_directed_edge_moment_order_1')
    return selected['value']


def feshbach_coupling(n_fixed=2, kernel=None):
    """Feshbach exponent principle: coupling = ((k*-1)/k*)^(g - n_fixed).
    CHANNEL-SELECTED (NB-walk by n_fixed class).

    Waterfilling-correct: n_fixed ∈ {0, 1, 2} are each above-waterline as
    distinct physical processes:
        n_fixed=0 self-energy   (closed loop)
        n_fixed=1 transition    (one pinned edge)
        n_fixed=2 scattering    (in+out pinned, = α_1_bare)
    """
    kernel = kernel or CountingKernel()
    k = kernel.substrate.K_STAR
    g = kernel.substrate.GIRTH
    candidates = [
        {'name': 'n_fixed=0 self-energy', 'channel': 'self_energy', 'n_fixed': 0},
        {'name': 'n_fixed=1 transition',  'channel': 'transition',  'n_fixed': 1},
        {'name': 'n_fixed=2 scattering',  'channel': 'scattering',  'n_fixed': 2},
    ]
    channel_map = {0: 'self_energy', 1: 'transition', 2: 'scattering'}
    selected = kernel.channel_select(candidates, channel=channel_map[n_fixed])
    return Fraction(k - 1, k) ** (g - selected['n_fixed'])
