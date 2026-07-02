"""
match/ — physics-naming layer over the substrate computer.

Architectural separation per the user's vision:

  simulator/   — entirely physics-free substrate computer.
                 Computes substrate observables exhaustively.
                 No SM observable names anywhere.

  match/       — optional layer that pairs substrate outputs with
                 SM observables, ADOPTED-B3 labelings, PDG values,
                 σ-deviation reports.

Top-level API:
    from match import (
        # Standard-Model predictions (each is a thin call into substrate
        # primitives, with the SM name as the function name)
        V_us, V_cb, V_ub, J_CKM, sin2_theta_W, alpha_GUT,
        y_tau, lambda_H, m_tau, m_H, M_Z, m_W,
        m_nu2, m_nu3, theta_12_PMNS, theta_13_PMNS, theta_23_PMNS,
        delta_CP_CKM, delta_CP_PMNS, alpha_21_PMNS, alpha_31_PMNS,
        eta_B, beta_cosmic_birefringence, A_s,
        # Particle aggregator
        Particle, get_particle, list_particles, particle_names,
        # Cosmology emulator
        CosmologyEmulator,
        # External anchors
        G_F, G_N_dimensionless, G_N_SI, m_top,
        # σ-deviation reporter
        sm_match_table,
    )

The canonical derivations (load-bearing, proof-grade) live in the
top-level `predictions/` package which is read by both this match
layer and (read-only) by the simulator. Neither this match package
nor the simulator modifies `predictions/`.
"""

from .sm_predictions import (
    # Gauge sector
    V_us, V_cb, V_ub, V_cd, V_cs, V_td, V_ts, V_tb, V_ud, J_CKM,
    sin2_theta_W, alpha_GUT, hypercharge,
    # Mass sector
    y_tau, lambda_H, alpha_1_bare, alpha_1_full,
    Q_Koide, epsilon_Koide, delta_Koide,
    v_higgs, m_tau, m_mu, m_e, m_H, M_Z, m_W, sin2_theta_W_MZ,
    koide_quark_ratio, georgi_jarlskog,
    # RG flow + atomic
    M_unif, g_1, g_2, g_3, alpha_s, alpha_EM,
    alpha_EM_thomson, R_infinity,
    # Neutrino sector
    m_nu2, m_nu3, R_nu_splitting,
    theta_12_PMNS, theta_13_PMNS, theta_23_PMNS,
    # Dispersion / Lorentz
    v_F_Gamma, v_F_P, eta_5, eta_lattice, D_H,
    D4_iso_H, D4_aniso_H, eta_NB_H,
    screw_wigner_cos_beta, screw_wigner_beta_deg,
    screw_wigner_d1_diag, screw_wigner_survival,
    srs_cubic_moment,
    # Framework-internal
    M_Pl_natural, srs_E_at_P, h_walker_eigenvalue,
    S_fresh, S_disconfirm, asymmetry_bits, N_hub,
    p_toggle, e_bit, lambda_toggle_rate, xi_t_temporal_correlation,
    srs_cubic_moment_n1, feshbach_coupling,
    # CP / dark phases
    delta_CP_CKM, theta_QCD, alpha_21_PMNS, alpha_31_PMNS,
    epsilon_CP, A_hemispherical,
    delta_CP_PMNS, beta_cosmic_birefringence,
    # Cosmology
    H_0, t_0, Lambda_CC, w_DE, Omega_DM_over_Omega_m, eta_B,
    A_s,
    # Structural
    k_star, d_spatial, g_girth, fermion_states_per_gen,
    n_generations, n_gauge_bosons, dark_feshbach_c,
)
from .anchors import G_F, G_N_dimensionless, G_N_SI, m_top
from .particle import Particle, get_particle, list_particles, particle_names
from .cosmology_emulator import CosmologyEmulator
from .pati_salam import PatiSalamUtility

__all__ = [
    # Gauge
    'V_us', 'V_cb', 'V_ub', 'V_cd', 'V_cs', 'V_td', 'V_ts', 'V_tb', 'V_ud', 'J_CKM',
    'sin2_theta_W', 'alpha_GUT', 'hypercharge',
    # Mass
    'y_tau', 'lambda_H', 'alpha_1_bare', 'alpha_1_full',
    'Q_Koide', 'epsilon_Koide', 'delta_Koide',
    'v_higgs', 'm_tau', 'm_mu', 'm_e', 'm_H', 'M_Z', 'm_W', 'sin2_theta_W_MZ',
    'koide_quark_ratio', 'georgi_jarlskog',
    # RG
    'M_unif', 'g_1', 'g_2', 'g_3', 'alpha_s', 'alpha_EM',
    'alpha_EM_thomson', 'R_infinity',
    # Neutrinos
    'm_nu2', 'm_nu3', 'R_nu_splitting',
    'theta_12_PMNS', 'theta_13_PMNS', 'theta_23_PMNS',
    # Dispersion / Lorentz
    'v_F_Gamma', 'v_F_P', 'eta_5', 'eta_lattice', 'D_H',
    'D4_iso_H', 'D4_aniso_H', 'eta_NB_H',
    'screw_wigner_cos_beta', 'screw_wigner_beta_deg',
    'screw_wigner_d1_diag', 'screw_wigner_survival',
    'srs_cubic_moment',
    # Framework-internal
    'M_Pl_natural', 'srs_E_at_P', 'h_walker_eigenvalue',
    'S_fresh', 'S_disconfirm', 'asymmetry_bits', 'N_hub',
    'p_toggle', 'e_bit', 'lambda_toggle_rate', 'xi_t_temporal_correlation',
    'srs_cubic_moment_n1', 'feshbach_coupling',
    # CP / dark
    'delta_CP_CKM', 'theta_QCD', 'alpha_21_PMNS', 'alpha_31_PMNS',
    'epsilon_CP', 'A_hemispherical',
    'delta_CP_PMNS', 'beta_cosmic_birefringence',
    # Cosmology
    'H_0', 't_0', 'Lambda_CC', 'w_DE', 'Omega_DM_over_Omega_m', 'eta_B',
    'A_s',
    # Structural
    'k_star', 'd_spatial', 'g_girth', 'fermion_states_per_gen',
    'n_generations', 'n_gauge_bosons', 'dark_feshbach_c',
    # Anchors
    'G_F', 'G_N_dimensionless', 'G_N_SI', 'm_top',
    # Aggregators / supporting
    'Particle', 'get_particle', 'list_particles', 'particle_names',
    'CosmologyEmulator',
    'PatiSalamUtility',
]
