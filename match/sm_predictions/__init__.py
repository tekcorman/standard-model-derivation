"""
Predictions layer — thin counting-first queries for framework observables.

Per the counting-first architecture: each prediction is a 5-15 line function
calling the kernel + utilities. This is the user-facing layer of the simulator.

Organized by mechanism family (per the 9-family table):
- gauge: family 1 (loop survival), family 4 (PS rep), family 8 (combinatorial)
- masses: mass cascade chain (family 1 + BZJ + structural)
- cp_phases: family 5 (geometric phases), family 6 (Bayesian asymmetries)
- cosmology: cosmology cascade (N_hub anchor + Friedmann), family 7 (MDL split)
- structural: family 9 (graph facts)
"""

from .gauge import (
    V_us, V_cb, V_ub, V_cd, V_cs, V_td, V_ts, V_tb, V_ud, J_CKM,
    sin2_theta_W, alpha_GUT, hypercharge,
)
from .masses import (
    y_tau, lambda_H, alpha_1_bare, alpha_1_full,
    Q_Koide, epsilon_Koide, delta_Koide,
    # Mass cascade (Phase 3b)
    v_higgs, m_tau, m_mu, m_e, m_H, M_Z, m_W, sin2_theta_W_MZ,
    # 3g additions (quark-sector Yukawa-texture identities)
    koide_quark_ratio, georgi_jarlskog,
)
from .rg_flow import (
    M_unif, g_1, g_2, g_3, alpha_s, alpha_EM,
    # 3g additions (atomic-precision constant)
    alpha_EM_thomson, R_infinity,
)
from .neutrinos import (
    m_nu2, m_nu3, R_nu_splitting,
    theta_12_PMNS, theta_13_PMNS, theta_23_PMNS,
)
from .dispersion import (
    v_F_Gamma, v_F_P, eta_5, eta_lattice, D_H,
)
from .lorentz import (
    D4_iso_H, D4_aniso_H, eta_NB_H,
    screw_wigner_cos_beta, screw_wigner_beta_deg,
    screw_wigner_d1_diag, screw_wigner_survival,
    srs_cubic_moment,
)
from .framework_internal import (
    M_Pl_natural, srs_E_at_P, h_walker_eigenvalue,
    S_fresh, S_disconfirm, asymmetry_bits, N_hub,
    # 3g additions (toggle constants + Feshbach exponents)
    p_toggle, e_bit, lambda_toggle_rate, xi_t_temporal_correlation,
    srs_cubic_moment_n1, feshbach_coupling,
)
from .cp_phases import (
    delta_CP_CKM, theta_QCD, alpha_21_PMNS, alpha_31_PMNS,
    epsilon_CP, A_hemispherical,
    # 3g additions (revival 2026-05-08)
    delta_CP_PMNS, beta_cosmic_birefringence,
)
from .cosmology import (
    H_0, t_0, Lambda_CC, w_DE, Omega_DM_over_Omega_m, eta_B,
    # 3g additions (primordial scalar amplitude)
    A_s,
)
from .structural import (
    k_star, d_spatial, g_girth, fermion_states_per_gen,
    n_generations, n_gauge_bosons, dark_feshbach_c,
)
# anchors (G_F, G_N, m_top) moved to match/anchors.py — they're top-level
# physics-anchored quantities, not "predictions". Reachable from
# `from match import G_F, G_N_dimensionless, ...`.

__all__ = [
    # Gauge sector (family 1, 4, 8)
    'V_us', 'V_cb', 'V_ub', 'V_cd', 'V_cs', 'V_td', 'V_ts', 'V_tb', 'J_CKM',
    'sin2_theta_W', 'alpha_GUT', 'hypercharge',
    # Mass sector (mass cascade)
    'y_tau', 'lambda_H', 'alpha_1_bare', 'alpha_1_full',
    'Q_Koide', 'epsilon_Koide', 'delta_Koide',
    'v_higgs', 'm_tau', 'm_mu', 'm_e', 'm_H', 'M_Z', 'm_W', 'sin2_theta_W_MZ',
    'koide_quark_ratio', 'georgi_jarlskog',
    # RG flow chain (Phase 3c)
    'M_unif', 'g_1', 'g_2', 'g_3', 'alpha_s', 'alpha_EM',
    'alpha_EM_thomson', 'R_infinity',
    # Neutrino sector (Phase 3d)
    'm_nu2', 'm_nu3', 'R_nu_splitting',
    'theta_12_PMNS', 'theta_13_PMNS', 'theta_23_PMNS',
    # Family 3 — dispersion / kinematics (Phase 3e)
    'v_F_Gamma', 'v_F_P', 'eta_5', 'eta_lattice', 'D_H',
    # Lorentz / Bloch dim-6 LV (Phase 3g.1)
    'D4_iso_H', 'D4_aniso_H', 'eta_NB_H',
    'screw_wigner_cos_beta', 'screw_wigner_beta_deg',
    'screw_wigner_d1_diag', 'screw_wigner_survival',
    'srs_cubic_moment',
    # Framework-internal (Phase 3f + 3g.2)
    'M_Pl_natural', 'srs_E_at_P', 'h_walker_eigenvalue',
    'S_fresh', 'S_disconfirm', 'asymmetry_bits', 'N_hub',
    'p_toggle', 'e_bit', 'lambda_toggle_rate', 'xi_t_temporal_correlation',
    'srs_cubic_moment_n1', 'feshbach_coupling',
    # CP/Dark (family 5, 6)
    'delta_CP_CKM', 'theta_QCD', 'alpha_21_PMNS', 'alpha_31_PMNS',
    'epsilon_CP', 'A_hemispherical',
    'delta_CP_PMNS', 'beta_cosmic_birefringence',
    # Cosmology
    'H_0', 't_0', 'Lambda_CC', 'w_DE', 'Omega_DM_over_Omega_m', 'eta_B',
    'A_s',
    # Structural (family 9)
    'k_star', 'd_spatial', 'g_girth', 'fermion_states_per_gen',
    'n_generations', 'n_gauge_bosons', 'dark_feshbach_c',
]
