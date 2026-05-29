"""
simulator.observables — exhaustive substrate-output catalog (physics-free).

Per the architectural separation:

    simulator/        substrate computer; entirely unaware of physics.
                      Computes observable features of the substrate
                      exhaustively to see what emerges.

    match/            optional layer pairing substrate outputs with SM
                      observables (V_us, m_τ, etc.) and PDG values.

This module provides the exhaustive substrate-output dump. Each top-level
function returns a structured table (dict / list / dataclass) of the
substrate quantities in its category. No SM observable names appear
anywhere in this module.

Categories:
- walk_survivals_table        — NB-walk survival fractions for relevant L
- multi_cycle_geometric_sums  — Σ over multi-cycle hosts (Hashimoto)
- bloch_eigenvalues_at_k      — adjacency / Hashimoto eigenvalues per k-point
- c3_isotypic_at_kpoints      — C₃-isotypic multiplicities at high-symmetry k
- bloch_taylor_table          — D2 / D4_iso / D4_aniso / η_NB^H
- closure_rates_at_saddle     — ν_amp / ν_mass² / ν_edge at h
- walker_phase_windings       — n × arg(h) mod 2π for n = 1..N
- bayesian_outputs            — toggle Markov + Beta-update quantities
- group_orbit_dimensions      — enumerated automorphism orbits
- clifford_grade_dims         — (Cl(6)-grade × Cl(0,2)-grade) dim table
- polytope_dihedrals          — K_4 (-1)-eigenspace + screw-axis angles
- structural_counts           — k*, |V|, |E|, g, d_spatial

Usage:
    from simulator.srs_engine.observables import all_substrate_outputs
    table = all_substrate_outputs()
    # table is a dict of dicts; inspect to see every substrate quantity
    # the simulator can derive from the kernel + utilities.
"""

import math
from fractions import Fraction
from itertools import combinations

from .kernel import CountingKernel
from .utils import (
    SpectralUtility,
    AlgebraicUtility,
    GroupOrbitUtility,
    GeometricPhaseUtility,
)


# ============================================================================
# Walk survivals (kernel.walk_count + branch_measure)
# ============================================================================

def walk_survivals_table(kernel=None, n_fixed_set=(0, 1, 2), L_max=20):
    """All NB-walk survival fractions ((k-1)/k)^L_eff for relevant (n_fixed, L_eff).

    Returns dict (n_fixed, L_eff) → Fraction.
    """
    kernel = kernel or CountingKernel()
    K = kernel.substrate.K_STAR
    table = {}
    for n_fixed in n_fixed_set:
        for L in range(1, L_max + 1):
            L_eff = L - n_fixed
            if L_eff <= 0:
                continue
            table[(n_fixed, L_eff)] = Fraction(K - 1, K) ** L_eff
    return table


def feshbach_at_girth(kernel=None, n_fixed_set=(0, 1, 2)):
    """Feshbach survival at girth: ((k-1)/k)^(g - n_fixed) for each n_fixed."""
    kernel = kernel or CountingKernel()
    K = kernel.substrate.K_STAR
    G = kernel.substrate.GIRTH
    return {n_fixed: Fraction(K - 1, K) ** (G - n_fixed) for n_fixed in n_fixed_set}


def nb_walk_geometric_sum(kernel=None, lengths=None):
    """α/(1-α) at each length L: NB-walk geometric sum over windings."""
    kernel = kernel or CountingKernel()
    K = kernel.substrate.K_STAR
    if lengths is None:
        lengths = list(range(2, 12))
    out = {}
    for L in lengths:
        a = Fraction(K - 1, K) ** (L - 1)
        out[L] = a / (1 - a)
    return out


def multi_cycle_geometric_sums(kernel=None, m_max=10, s_seam=2, n_fixed=2):
    """Σ_{m=2}^{m_max} α_m / (1 - α_m) for multi-cycle Hashimoto hosts.

    Returns the per-m contributions plus the cumulative sum.
    """
    kernel = kernel or CountingKernel()
    K = kernel.substrate.K_STAR
    G = kernel.substrate.GIRTH
    survival = Fraction(K - 1, K)
    contribs = {}
    cumulative = Fraction(0)
    for m in range(2, m_max + 1):
        L_eff = m * G - 2 * (m - 1) * s_seam - n_fixed
        a = survival ** L_eff
        contribs[m] = {'L_eff': L_eff, 'alpha_m': a, 'contribution': a / (1 - a)}
        cumulative += a / (1 - a)
    return {'per_m': contribs, 'cumulative': cumulative}


# ============================================================================
# Bloch spectra (substrate's adjacency_at_k / hashimoto_at_k)
# ============================================================================

def bloch_eigenvalues_at_k(kernel=None, k_label='P'):
    """Adjacency Bloch eigenvalues at named k-point (Γ, P, N, H)."""
    kernel = kernel or CountingKernel()
    return list(kernel.substrate.adjacency_spectrum_at_k(k_label))


def bloch_eigenvalues_at_all_high_symmetry(kernel=None):
    """Adjacency eigenvalues at all four high-symmetry k-points."""
    kernel = kernel or CountingKernel()
    return {label: list(kernel.substrate.adjacency_spectrum_at_k(label))
            for label in kernel.substrate.K_POINTS}


def ramanujan_saddle(kernel=None):
    """Ramanujan eigenvalue h at the P-saddle, plus its derived quantities.

    Returns dict with h, |h|², Re(h), Im(h), arg(h) in radians + degrees,
    sin/cos/tan of arg(h), and ν_amp / ν_m² / ν_edge closure rates.
    """
    kernel = kernel or CountingKernel()
    h = kernel.substrate.ramanujan_eigenvalue_at_P
    arg_h = math.atan2(h.imag, h.real)
    return {
        'h': h,
        'abs_h_sq': abs(h) ** 2,
        're': h.real,
        'im': h.imag,
        'arg_rad': arg_h,
        'arg_deg': math.degrees(arg_h),
        'sin_arg': h.imag / abs(h),
        'cos_arg': h.real / abs(h),
        'tan_arg': h.imag / h.real,
        'tan_arg_sq': (h.imag / h.real) ** 2,  # = ν_m² closure rate
        'closure_rate_amplitude': kernel.substrate.closure_rate_amplitude,    # √5/4
        'closure_rate_mass_squared': kernel.substrate.closure_rate_mass_squared,  # 5/3
        'closure_rate_edge_local': kernel.substrate.closure_rate_edge_local,  # 1
    }


def c3_isotypic_at_P(kernel=None):
    """V_Ram(P) C₃-isotypic multiplicity tuple, e.g., (4, 2, 2) on srs."""
    kernel = kernel or CountingKernel()
    return kernel.substrate.c3_isotypic_decomposition_at_P()


# ============================================================================
# Bloch-Taylor coefficients at Γ (4th order)
# ============================================================================

def bloch_taylor_at_gamma(kernel=None, order=4):
    """D2, D4_iso, D4_aniso, η_NB^H from kernel primitive."""
    kernel = kernel or CountingKernel()
    return kernel.bloch_taylor_at_gamma(order=order)


# ============================================================================
# Walker phase windings
# ============================================================================

def walker_phase_windings(kernel=None, n_set=None):
    """n × arg(h) mod 2π for n in n_set; default n_set covers small windings."""
    kernel = kernel or CountingKernel()
    if n_set is None:
        n_set = list(range(1, 21))
    return {n: GeometricPhaseUtility.walker_phase_winding(kernel, winding_number=n)
            for n in n_set}


# ============================================================================
# Bayesian / toggle outputs
# ============================================================================

def bayesian_outputs(kernel=None):
    """Toggle Markov rates + Beta-update surprise/asymmetry outputs."""
    kernel = kernel or CountingKernel()
    rates = kernel.toggle_markov()
    return {
        'p_create': rates['p_create'],          # 1/2 (Beta(1,1))
        'p_destroy': rates['p_destroy'],        # 1/k* = 1/3
        'asymmetry_bits': rates['asymmetry_bits'],  # log₂(3/2)
        's_fresh_bits': rates['s_fresh_bits'],   # 1
        's_disconfirm_bits': rates['s_disconfirm_bits'],  # log₂(3)
        'beta_class_D_asymmetry': Fraction(kernel.substrate.K_STAR - 2,
                                            kernel.substrate.K_STAR + 2),  # 1/5
    }


# ============================================================================
# Group orbits
# ============================================================================

def group_orbit_dimensions(kernel=None):
    """Enumerated automorphism / orbit dimensions on srs."""
    kernel = kernel or CountingKernel()
    return {
        'lattice_atoms_per_cell': kernel.orbit_count('lattice_atoms'),  # 4
        'c3_at_P_multiplicities': kernel.orbit_count('C_3_at_P'),  # (4, 2, 2)
        'fermion_states_per_gen': kernel.orbit_count('fermion_content_per_gen'),  # 8
        'gauge_boson_dim': kernel.orbit_count('gauge_bosons'),  # 12
        'galois_z3_orbit': kernel.orbit_count('generations'),  # 3
    }


# ============================================================================
# Equivalence-class counts
# ============================================================================

def equivalence_class_counts(kernel=None):
    """Substrate equivalence-class enumerations."""
    kernel = kernel or CountingKernel()
    return {
        'coupling_pair_per_girth_cycle': kernel.equiv_class_count(
            'coupling_pair_per_girth_cycle'),  # 9
        'site_stabilizer_orbit_at_vertex': kernel.equiv_class_count(
            'site_stabilizer_orbit_at_vertex'),  # 3
        'cl6_fock_label_slots': kernel.equiv_class_count('cl6_fock_label_slots'),  # 24
    }


# ============================================================================
# Clifford grade decomposition (Cl(6) ⊗ Cl(0,2))
# ============================================================================

def clifford_grade_dims():
    """Joint Cl(6) ⊗ Cl(0,2) dim table indexed by (m, n) grade pair."""
    cl6_dims = [math.comb(6, m) for m in range(7)]  # (1, 6, 15, 20, 15, 6, 1)
    cl02_dims = [1, 2, 1]
    return {(m, n): cl6_dims[m] * cl02_dims[n]
            for m in range(7) for n in range(3)}


def cl6_chirality_per_grade():
    """+1 for even Cl(6) grades, -1 for odd (γ_5 conjugation behaviour)."""
    return {m: (+1 if m % 2 == 0 else -1) for m in range(7)}


# ============================================================================
# Polytope dihedrals + cubic moments
# ============================================================================

def polytope_dihedrals(kernel=None):
    """Substrate polytope dihedral angles + screw-axis content."""
    kernel = kernel or CountingKernel()
    K = kernel.substrate.K_STAR
    cos_b = Fraction(K - 2, K)  # = 1/k* on srs
    beta_rad = math.acos(float(cos_b))
    return {
        'k4_minus_eigenspace_dihedral_cos': Fraction(1, K),  # = 1/3
        'k4_minus_eigenspace_dihedral_deg': math.degrees(math.acos(1.0 / K)),
        'screw_axis_cos_beta': cos_b,
        'screw_axis_beta_deg': math.degrees(beta_rad),
        'wigner_d1_pm1': (Fraction(1) + cos_b) / 2,  # 2/3
        'wigner_d1_00': cos_b,  # 1/3
        'survival_pm1': ((Fraction(1) + cos_b) / 2) ** 2,  # 4/9
        'survival_0': cos_b ** 2,  # 1/9
    }


def srs_cubic_moments(kernel=None, n_max=6):
    """Directed-edge 2n-th moments on principal cubic axis: 1/(k* · 2^(n-1))."""
    kernel = kernel or CountingKernel()
    K = kernel.substrate.K_STAR
    return {n: Fraction(1, K * 2 ** (n - 1)) for n in range(1, n_max + 1)}


# ============================================================================
# Structural counts (substrate primitives)
# ============================================================================

def structural_counts(kernel=None):
    """Substrate's fixed structural counts."""
    kernel = kernel or CountingKernel()
    s = kernel.substrate
    return {
        'k_star': s.K_STAR,
        'n_atoms_per_cell': s.N_ATOMS,
        'n_edges_per_cell': s.N_EDGES,
        'n_directed_edges': s.N_DIRECTED,
        'girth': s.GIRTH,
        'spatial_dim': s.D_SPATIAL,
        'cycle_space_dim': 2 * (s.N_EDGES - s.N_ATOMS) + 1,  # 5
        'tree_modes_dim': 2 * s.N_ATOMS - 2,  # 6 = Hashimoto trivial sector
    }


# ============================================================================
# Master dump — all substrate outputs in one call
# ============================================================================

def all_substrate_outputs(kernel=None):
    """Exhaustive dump of every substrate output category.

    Returns nested dict {category: outputs_dict}. Use this for the full
    catalog at a glance; use individual functions above for targeted
    queries.
    """
    kernel = kernel or CountingKernel()
    return {
        'structural_counts': structural_counts(kernel),
        'walk_survivals': walk_survivals_table(kernel),
        'feshbach_at_girth': feshbach_at_girth(kernel),
        'nb_geometric_sums': nb_walk_geometric_sum(kernel),
        'multi_cycle_sums': multi_cycle_geometric_sums(kernel),
        'bloch_eigenvalues': bloch_eigenvalues_at_all_high_symmetry(kernel),
        'ramanujan_saddle': ramanujan_saddle(kernel),
        'c3_isotypic_at_P': c3_isotypic_at_P(kernel),
        'bloch_taylor_gamma': bloch_taylor_at_gamma(kernel, order=4),
        'walker_phase_windings': walker_phase_windings(kernel),
        'bayesian_outputs': bayesian_outputs(kernel),
        'group_orbit_dimensions': group_orbit_dimensions(kernel),
        'equiv_class_counts': equivalence_class_counts(kernel),
        'clifford_grade_dims': clifford_grade_dims(),
        'cl6_chirality_per_grade': cl6_chirality_per_grade(),
        'polytope_dihedrals': polytope_dihedrals(kernel),
        'srs_cubic_moments': srs_cubic_moments(kernel),
    }
