#!/usr/bin/env python3
"""
srs_pmns_rg_running.py — RG running of PMNS angles from M_GUT to M_Z.

GOAL: Close the ~4.5% gap in theta_13:
    theta_13(GUT) = arcsin(V_us / sqrt(2)) ~ 8.96 deg  (Pati-Salam / srs)
    theta_13(obs, M_Z) = 8.57 deg
    Gap: ~4.5%, attributed to RG running in the MSSM.

APPROACH: Run the PHYSICAL y_tau (matching observed m_tau) from M_Z up to M_GUT,
recording y_tau(mu) at all scales. Then integrate the PMNS RGEs from M_GUT down
to M_Z using the physical y_tau profile.

FRAMEWORK INPUTS:
    tan(beta) = 44.73       (derived in srs_tan_beta.py)
    M_SUSY    = 1732 GeV    (= m_{3/2} = (2/3)^90 * M_P)
    M_GUT     = 2e16 GeV    (MSSM unification)
    alpha_GUT = 1/24.1      (Cl(6) normalization)
    M_R       = (2/3)^10 * M_GUT = 3.47e14 GeV  (seesaw scale)

PMNS RGEs: Antusch, Kersten, Lindner, Ratz (2003), hep-ph/0305273.
"""

import numpy as np
from numpy import sqrt, pi, log, exp, sin, cos, arcsin, arctan, tan
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# =============================================================================
# CONSTANTS
# =============================================================================

DEG = 180.0 / pi
RAD = pi / 180.0

# Graph topology
k = 3
g = 10
alpha_1 = (5.0 / 3.0) * (2.0 / 3.0)**8

# Energy scales
M_P = 1.22089e19
M_GUT = 2.0e16
M_Z = 91.1876
v_higgs = 246.22
v_over_root2 = v_higgs / sqrt(2)
alpha_GUT_inv = 24.1
alpha_GUT = 1.0 / alpha_GUT_inv

# SUSY scale
m_32 = (2.0 / 3.0)**(k**2 * g) * M_P
M_SUSY = m_32

# Seesaw scale
M_R = (2.0 / 3.0)**g * M_GUT

# tan(beta)
tan_beta = 44.73
cos_beta = 1.0 / sqrt(1.0 + tan_beta**2)
sin_beta = tan_beta * cos_beta

# Masses
m_tau = 1.7769
m_t_pole = 172.76
m_b_MSbar = 4.18
alpha_s_MZ = 0.1179
sin2_tw_MZ = 0.23122
alpha_em_inv_MZ = 127.95

# Derived: observed couplings at M_Z
alpha_em = 1.0 / alpha_em_inv_MZ
alpha_2_obs = alpha_em / sin2_tw_MZ
alpha_Y_obs = alpha_em / (1.0 - sin2_tw_MZ)
alpha_1_obs = (5.0/3.0) * alpha_Y_obs
alpha_1_inv_obs = 1.0 / alpha_1_obs
alpha_2_inv_obs = 1.0 / alpha_2_obs
alpha_3_inv_obs = 1.0 / alpha_s_MZ

# Yukawas at M_Z (MSSM convention)
qcd_corr = 1.0 + 4.0 * alpha_s_MZ / (3.0 * pi)
y_t_MZ = m_t_pole / (qcd_corr * v_over_root2 * sin_beta)
y_b_MZ = m_b_MSbar / (v_over_root2 * cos_beta)
y_tau_MZ = m_tau / (v_over_root2 * cos_beta)

# Beta function coefficients
b_MSSM = np.array([33.0/5.0, 1.0, -3.0])
bij_MSSM = np.array([
    [199.0/25.0, 27.0/5.0, 88.0/5.0],
    [  9.0/5.0,  25.0,     24.0     ],
    [ 11.0/5.0,   9.0,     14.0     ]
])
b_SM = np.array([41.0/10.0, -19.0/6.0, -7.0])
bij_SM = np.array([
    [199.0/50.0,  27.0/10.0, 44.0/5.0],
    [  9.0/10.0,  35.0/6.0,  12.0    ],
    [ 11.0/10.0,   9.0/2.0, -26.0    ]
])

# Neutrino data (NuFIT 5.3, NO)
dm2_21 = 7.53e-5
dm2_31 = 2.453e-3
m_nu3_obs = sqrt(dm2_31)
m_nu2_obs = sqrt(dm2_21)

# Observed PMNS
theta12_obs = 33.41
theta13_obs = 8.57
theta23_obs = 49.2
delta_CP_obs = 230.0

# GUT-scale PMNS
V_us = (2.0 / 3.0)**(2 + sqrt(3))
theta13_GUT = arcsin(V_us / sqrt(k - 1))  # radians
theta12_GUT = 35.26  # deg (TBM)
theta23_GUT = 45.0   # deg (TBM)

# Neutrino Yukawa
L_us = 2 + sqrt(3)
y_nu = (2.0 / 3.0) * sqrt(L_us / k)
R_ihara = 32.19


def pct(pred, obs):
    return (pred - obs) / obs * 100.0


def print_section(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


# =============================================================================
# GAUGE + YUKAWA RGE SYSTEM
# =============================================================================

def mssm_rge(t, y):
    """MSSM RGE: y = [1/a1, 1/a2, 1/a3, yt, yb, ytau]."""
    a1i, a2i, a3i, yt, yb, ytau = y
    a = np.array([1.0/a1i, 1.0/a2i, 1.0/a3i])
    g1_sq = 4*pi*a[0]; g2_sq = 4*pi*a[1]; g3_sq = 4*pi*a[2]

    da_inv = np.zeros(3)
    for i in range(3):
        da_inv[i] = -b_MSSM[i] / (2*pi)
        for j in range(3):
            da_inv[i] -= bij_MSSM[i,j] / (8*pi**2) * a[j]

    yt2, yb2, ytau2 = yt**2, yb**2, ytau**2
    beta_yt = yt/(16*pi**2) * (6*yt2 + yb2 - 16/3*g3_sq - 3*g2_sq - 13/15*g1_sq)
    beta_yb = yb/(16*pi**2) * (6*yb2 + yt2 + ytau2 - 16/3*g3_sq - 3*g2_sq - 7/15*g1_sq)
    beta_ytau = ytau/(16*pi**2) * (4*ytau2 + 3*yb2 - 3*g2_sq - 9/5*g1_sq)

    return [da_inv[0], da_inv[1], da_inv[2], beta_yt, beta_yb, beta_ytau]


def sm_rge(t, y):
    """SM RGE: y = [1/a1, 1/a2, 1/a3, yt, yb, ytau]."""
    a1i, a2i, a3i, yt, yb, ytau = y
    a = np.array([1.0/a1i, 1.0/a2i, 1.0/a3i])
    g1_sq = 4*pi*a[0]; g2_sq = 4*pi*a[1]; g3_sq = 4*pi*a[2]

    da_inv = np.zeros(3)
    for i in range(3):
        da_inv[i] = -b_SM[i] / (2*pi)
        for j in range(3):
            da_inv[i] -= bij_SM[i,j] / (8*pi**2) * a[j]

    yt2, yb2, ytau2 = yt**2, yb**2, ytau**2
    beta_yt = yt/(16*pi**2) * (9/2*yt2 + 3/2*yb2 - 8*g3_sq - 9/4*g2_sq - 17/12*g1_sq)
    beta_yb = yb/(16*pi**2) * (9/2*yb2 + 3/2*yt2 + ytau2 - 8*g3_sq - 9/4*g2_sq - 5/12*g1_sq)
    beta_ytau = ytau/(16*pi**2) * (5/2*ytau2 + 3*yb2 - 9/4*g2_sq - 15/4*g1_sq)

    return [da_inv[0], da_inv[1], da_inv[2], beta_yt, beta_yb, beta_ytau]


# =============================================================================
# PART 1: BUILD y_tau(mu) PROFILE
# =============================================================================

def build_ytau_profile():
    """
    Run gauge + Yukawa from M_Z up to M_GUT, building a continuous y_tau(mu).
    Uses SM below M_SUSY, MSSM above M_SUSY.
    """
    print_section("PART 1: BUILD y_tau(mu) PROFILE")

    t_MZ = log(M_Z)
    t_SUSY = log(M_SUSY)
    t_GUT = log(M_GUT)
    t_R = log(M_R)

    # SM y_tau at M_Z (no tan(beta) for SM)
    y_tau_SM_MZ = sqrt(2) * m_tau / v_higgs

    print(f"  Yukawas at M_Z:")
    print(f"    y_tau(SM, M_Z)   = {y_tau_SM_MZ:.6f}")
    print(f"    y_tau(MSSM, M_Z) = {y_tau_MZ:.6f}  (with tan(beta) = {tan_beta})")
    print(f"    y_t(MSSM, M_Z)  = {y_t_MZ:.6f}")
    print(f"    y_b(MSSM, M_Z)  = {y_b_MZ:.6f}")
    print()

    # Phase 1: SM from M_Z to M_SUSY
    y0_sm = [alpha_1_inv_obs, alpha_2_inv_obs, alpha_3_inv_obs,
             y_t_MZ * sin_beta,  # SM y_t = MSSM y_t * sin(beta) ... no
             y_b_MZ * cos_beta,  # SM y_b = MSSM y_b * cos(beta)
             y_tau_SM_MZ]

    # Actually: in the SM, y_t = sqrt(2)*m_t/(v), y_b = sqrt(2)*m_b/(v), y_tau = sqrt(2)*m_tau/(v)
    y_t_SM = m_t_pole / (qcd_corr * v_over_root2)
    y_b_SM = m_b_MSbar / v_over_root2
    y_tau_SM = m_tau / v_over_root2

    y0_sm = [alpha_1_inv_obs, alpha_2_inv_obs, alpha_3_inv_obs,
             y_t_SM, y_b_SM, y_tau_SM]

    t_eval_sm = np.linspace(t_MZ, t_SUSY, 500)
    sol_sm = solve_ivp(sm_rge, [t_MZ, t_SUSY], y0_sm,
                       method='RK45', rtol=1e-10, atol=1e-12,
                       dense_output=True, t_eval=t_eval_sm)

    # At SUSY threshold: convert SM -> MSSM Yukawas
    at_susy_sm = sol_sm.sol(t_SUSY)
    # SM: m_f = y_f * v / sqrt(2)
    # MSSM: m_t = y_t * v*sin(beta)/sqrt(2), m_b = y_b * v*cos(beta)/sqrt(2)
    # => y_t(MSSM) = y_t(SM) / sin(beta), y_b(MSSM) = y_b(SM) / cos(beta)
    y_t_MSSM_SUSY = at_susy_sm[3] / sin_beta
    y_b_MSSM_SUSY = at_susy_sm[4] / cos_beta
    y_tau_MSSM_SUSY = at_susy_sm[5] / cos_beta

    print(f"  At M_SUSY = {M_SUSY:.2f} GeV (after threshold):")
    print(f"    y_tau(MSSM) = {y_tau_MSSM_SUSY:.6f}")
    print(f"    y_b(MSSM)   = {y_b_MSSM_SUSY:.6f}")
    print(f"    y_t(MSSM)   = {y_t_MSSM_SUSY:.6f}")
    print()

    # Phase 2: MSSM from M_SUSY to M_GUT
    y0_mssm = [at_susy_sm[0], at_susy_sm[1], at_susy_sm[2],
               y_t_MSSM_SUSY, y_b_MSSM_SUSY, y_tau_MSSM_SUSY]

    t_eval_mssm = np.linspace(t_SUSY, t_GUT, 2000)
    sol_mssm = solve_ivp(mssm_rge, [t_SUSY, t_GUT], y0_mssm,
                         method='RK45', rtol=1e-10, atol=1e-12,
                         dense_output=True, t_eval=t_eval_mssm)

    at_gut = sol_mssm.sol(t_GUT)
    at_mr = sol_mssm.sol(t_R)

    print(f"  At M_R = {M_R:.4e} GeV:")
    print(f"    y_tau(MSSM) = {at_mr[5]:.6f}")
    print(f"    y_b(MSSM)   = {at_mr[4]:.6f}")
    print()

    print(f"  At M_GUT = {M_GUT:.2e} GeV:")
    print(f"    y_tau(MSSM) = {at_gut[5]:.6f}")
    print(f"    y_b(MSSM)   = {at_gut[4]:.6f}")
    print(f"    y_t(MSSM)   = {at_gut[3]:.6f}")
    print(f"    alpha_1_inv = {at_gut[0]:.4f}")
    print(f"    alpha_2_inv = {at_gut[1]:.4f}")
    print(f"    alpha_3_inv = {at_gut[2]:.4f}")
    print()
    print(f"    y_b/y_tau at GUT = {at_gut[4]/at_gut[5]:.4f} (expected GJ = 3)")
    print(f"    Framework y_tau_GUT = alpha_1/k^2 = {alpha_1/k**2:.6f}")
    print(f"    Physical y_tau_GUT = {at_gut[5]:.6f}")
    print(f"    Ratio: {at_gut[5]/(alpha_1/k**2):.2f}")

    # Build interpolators for MSSM y_tau (what matters for PMNS running)
    # Combine SM and MSSM segments
    # In the SM regime, the effective y_tau for kappa running is the SM y_tau
    # (no tan(beta) enhancement). But actually, below M_SUSY the kappa operator
    # runs with SM RGEs where C_e = -3/2 and y_tau ~ 0.01 (small).
    # The MSSM regime (M_SUSY to M_GUT) dominates the PMNS running.

    # MSSM y_tau interpolator (for running from M_GUT down to M_SUSY)
    t_mssm = sol_mssm.t
    ytau_mssm = np.array([sol_mssm.sol(t)[5] for t in t_mssm])
    yt_mssm = np.array([sol_mssm.sol(t)[3] for t in t_mssm])
    yb_mssm = np.array([sol_mssm.sol(t)[4] for t in t_mssm])
    a1_mssm = np.array([1.0/sol_mssm.sol(t)[0] for t in t_mssm])
    a2_mssm = np.array([1.0/sol_mssm.sol(t)[1] for t in t_mssm])

    ytau_interp = interp1d(t_mssm, ytau_mssm, kind='cubic', fill_value='extrapolate')
    yt_interp = interp1d(t_mssm, yt_mssm, kind='cubic', fill_value='extrapolate')
    a1_interp = interp1d(t_mssm, a1_mssm, kind='cubic', fill_value='extrapolate')
    a2_interp = interp1d(t_mssm, a2_mssm, kind='cubic', fill_value='extrapolate')

    # Print y_tau at several scales
    scales = [("M_GUT", t_GUT), ("10^15", log(1e15)), ("M_R", t_R),
              ("10^13", log(1e13)), ("10^10", log(1e10)), ("10^6", log(1e6)),
              ("M_SUSY", t_SUSY)]
    print(f"\n  y_tau profile across MSSM regime:")
    print(f"  {'Scale':<10} {'mu (GeV)':<12} {'y_tau':<12} {'y_tau^2':<12}")
    print(f"  {'-'*46}")
    for name, t in scales:
        if t_SUSY <= t <= t_GUT:
            yt = float(ytau_interp(t))
            print(f"  {name:<10} {exp(t):<12.2e} {yt:<12.6f} {yt**2:<12.6e}")

    return sol_mssm, sol_sm, ytau_interp, yt_interp, a1_interp, a2_interp


# =============================================================================
# PART 2: PMNS RG RUNNING WITH PHYSICAL y_tau
# =============================================================================

def run_pmns(sol_mssm, sol_sm, ytau_interp, yt_interp, a1_interp, a2_interp,
             alpha1_M=0.0, alpha2_M=0.0, verbose=True):
    """
    Integrate PMNS RGEs from M_GUT down to M_Z using the physical y_tau profile.
    """
    delta = delta_CP_obs * RAD

    # Neutrino masses at seesaw scale
    M_D = y_nu * v_higgs / sqrt(2)
    m3_init = M_D**2 / M_R * 1e9  # eV
    m2_init = m3_init / sqrt(R_ihara)
    m1_init = 0.0

    th12_init = theta12_GUT * RAD
    th13_init = theta13_GUT  # radians
    th23_init = theta23_GUT * RAD

    t_GUT = log(M_GUT)
    t_R = log(M_R)
    t_SUSY = log(M_SUSY)
    t_MZ = log(M_Z)

    C_e_mssm = 1.0
    C_e_sm = -3.0/2.0

    def pmns_rge(t, y, C_e, get_ytau):
        """PMNS + mass RGE at scale t."""
        th12, th13, th23, m1, m2, m3 = y

        ytau = get_ytau(t)
        ytau2 = ytau**2

        s12, c12 = sin(th12), cos(th12)
        s13, c13 = sin(th13), cos(th13)
        s23, c23 = sin(th23), cos(th23)

        dm2_21_loc = m2**2 - m1**2
        dm2_31_loc = m3**2 - m1**2
        dm2_32_loc = m3**2 - m2**2
        eps = 1e-40

        # theta_12 (vanishes for m1=0)
        if abs(dm2_21_loc) > eps and m1 > 1e-20:
            f12 = m1 * m2 * cos(alpha2_M - alpha1_M) / (dm2_21_loc + eps)
        else:
            f12 = 0.0
        dth12 = -C_e * ytau2 / (32*pi**2) * sin(2*th12) * s23**2 * f12

        # theta_13
        if abs(dm2_32_loc) > eps:
            term1 = 0.0 if m1 < 1e-20 else m1*m3*sin(alpha2_M - 2*delta) / (dm2_31_loc + eps)
            term2 = m2*m3*sin(alpha1_M - alpha2_M - 2*delta) / (dm2_32_loc + eps)
            f13 = term1 + term2
        else:
            f13 = 0.0
        dth13 = -C_e * ytau2 / (32*pi**2) * sin(2*th12) * sin(2*th23) / 2 * f13

        # theta_23
        if abs(dm2_32_loc) > eps:
            term_a = s12**2 * m2*m3*cos(alpha1_M - alpha2_M) / (dm2_32_loc + eps)
            term_b = 0.0 if m1 < 1e-20 else c12**2*m1*m3*cos(alpha2_M) / (dm2_31_loc + eps)
            f23 = term_a + term_b
        else:
            f23 = 0.0
        dth23 = -C_e * ytau2 / (32*pi**2) * sin(2*th23) * f23

        # Mass running
        Ut1_sq = (s12*s23)**2 + (c12*c23*s13)**2 - 2*s12*s23*c12*c23*s13*cos(delta)
        Ut2_sq = (c12*s23)**2 + (s12*c23*s13)**2 + 2*c12*s23*s12*c23*s13*cos(delta)
        Ut3_sq = c13**2 * c23**2

        # Universal running of kappa (gauge + yt contributions)
        # MSSM: alpha_kappa = 1/(16pi^2) * (6*yt^2 - 6/5*g1^2 - 6*g2^2)
        # SM:   alpha_kappa = 1/(16pi^2) * (-3*g2^2 + 6*yt^2 + lambda_H)
        if C_e > 0:  # MSSM
            try:
                yt = float(yt_interp(t))
                a1 = float(a1_interp(t))
                a2 = float(a2_interp(t))
            except:
                yt = 1.0; a1 = alpha_GUT; a2 = alpha_GUT
            g1_sq = 4*pi*a1
            g2_sq = 4*pi*a2
            alpha_kappa = 1.0/(16*pi**2) * (6*yt**2 - 6.0/5.0*g1_sq - 6.0*g2_sq)
        else:  # SM
            # Use approximate SM values
            g2_sq = 4*pi*alpha_2_obs
            g1_sq = 4*pi*alpha_1_obs
            yt_sm = sqrt(2)*m_t_pole/(qcd_corr*v_higgs)
            alpha_kappa = 1.0/(16*pi**2) * (-3*g2_sq + 6*yt_sm**2 + 0.13)

        dm1 = m1 * (C_e * ytau2/(16*pi**2) * Ut1_sq + alpha_kappa)
        dm2 = m2 * (C_e * ytau2/(16*pi**2) * Ut2_sq + alpha_kappa)
        dm3 = m3 * (C_e * ytau2/(16*pi**2) * Ut3_sq + alpha_kappa)

        return [dth12, dth13, dth23, dm1, dm2, dm3]

    # Initial conditions at M_GUT
    y0 = [th12_init, th13_init, th23_init, m1_init, m2_init, m3_init]

    # Phase 1: M_GUT -> M_SUSY (MSSM)
    def rge_mssm(t, y):
        return pmns_rge(t, y, C_e_mssm, lambda t: float(ytau_interp(t)))

    sol1 = solve_ivp(rge_mssm, [t_GUT, t_SUSY], y0,
                     method='RK45', rtol=1e-12, atol=1e-15,
                     dense_output=True, max_step=0.5)
    at_SUSY = sol1.sol(t_SUSY)

    if verbose:
        print(f"\n  At M_SUSY = {M_SUSY:.2f} GeV:")
        print(f"    theta_12 = {at_SUSY[0]*DEG:.4f} deg")
        print(f"    theta_13 = {at_SUSY[1]*DEG:.4f} deg")
        print(f"    theta_23 = {at_SUSY[2]*DEG:.4f} deg")
        print(f"    m_nu1    = {at_SUSY[3]:.6f} eV")
        print(f"    m_nu2    = {at_SUSY[4]:.6f} eV")
        print(f"    m_nu3    = {at_SUSY[5]:.6f} eV")

    # Phase 2: M_SUSY -> M_Z (SM)
    y_tau_SM = sqrt(2) * m_tau / v_higgs

    def rge_sm(t, y):
        return pmns_rge(t, y, C_e_sm, lambda t: y_tau_SM)

    sol2 = solve_ivp(rge_sm, [t_SUSY, t_MZ], list(at_SUSY),
                     method='RK45', rtol=1e-12, atol=1e-15,
                     dense_output=True)
    at_MZ = sol2.sol(t_MZ)

    if verbose:
        print(f"\n  At M_Z = {M_Z:.4f} GeV:")
        print(f"    theta_12 = {at_MZ[0]*DEG:.4f} deg")
        print(f"    theta_13 = {at_MZ[1]*DEG:.4f} deg")
        print(f"    theta_23 = {at_MZ[2]*DEG:.4f} deg")
        print(f"    m_nu1    = {at_MZ[3]:.6f} eV")
        print(f"    m_nu2    = {at_MZ[4]:.6f} eV")
        print(f"    m_nu3    = {at_MZ[5]:.6f} eV")

    return at_MZ


# =============================================================================
# PART 3: MAJORANA PHASE SCAN
# =============================================================================

def phase_scan(sol_mssm, sol_sm, ytau_interp, yt_interp, a1_interp, a2_interp):
    print_section("PART 3: MAJORANA PHASE SCAN")

    best_err = 999.0
    best_phases = None
    best_result = None

    n_scan = 36

    print(f"  Scanning {n_scan}x{n_scan} Majorana phase grid...")
    print(f"  Target: theta_13(M_Z) = {theta13_obs:.2f} deg")
    print()

    for i_a1 in range(n_scan):
        for i_a2 in range(n_scan):
            a1 = 2*pi*i_a1/n_scan
            a2 = 2*pi*i_a2/n_scan
            try:
                result = run_pmns(sol_mssm, sol_sm, ytau_interp, yt_interp,
                                  a1_interp, a2_interp,
                                  alpha1_M=a1, alpha2_M=a2, verbose=False)
                th13_deg = result[1] * DEG
                err = abs(th13_deg - theta13_obs)
                if err < best_err:
                    best_err = err
                    best_phases = (a1, a2)
                    best_result = result
            except Exception:
                pass

    if best_result is not None:
        print(f"  Best fit:")
        print(f"    alpha1_M = {best_phases[0]*DEG:.1f} deg")
        print(f"    alpha2_M = {best_phases[1]*DEG:.1f} deg")
        print(f"    theta_12(M_Z) = {best_result[0]*DEG:.4f} deg  (obs: {theta12_obs})")
        print(f"    theta_13(M_Z) = {best_result[1]*DEG:.4f} deg  (obs: {theta13_obs})")
        print(f"    theta_23(M_Z) = {best_result[2]*DEG:.4f} deg  (obs: {theta23_obs})")
        print(f"    m_nu2(M_Z)    = {best_result[4]:.6f} eV  (obs: {m_nu2_obs:.5f})")
        print(f"    m_nu3(M_Z)    = {best_result[5]:.6f} eV  (obs: {m_nu3_obs:.4f})")
        print(f"    Residual: {best_err:.4f} deg ({pct(best_result[1]*DEG, theta13_obs):+.2f}%)")

    return best_phases, best_result


# =============================================================================
# PART 4: ANALYTIC ESTIMATES
# =============================================================================

def analytic_estimates(ytau_interp):
    print_section("PART 4: ANALYTIC ESTIMATES")

    t_GUT = log(M_GUT)
    t_SUSY = log(M_SUSY)

    # Compute the integral of y_tau^2 over the MSSM range
    n_pts = 10000
    t_arr = np.linspace(t_SUSY, t_GUT, n_pts)
    ytau_arr = np.array([float(ytau_interp(t)) for t in t_arr])
    ytau2_arr = ytau_arr**2
    dt = (t_GUT - t_SUSY) / (n_pts - 1)
    integral_ytau2 = np.trapezoid(ytau2_arr, t_arr)

    print(f"  Integral of y_tau^2 * dt from M_SUSY to M_GUT:")
    print(f"    = {integral_ytau2:.6e}")
    print(f"    y_tau^2 at M_SUSY: {ytau_arr[0]**2:.6e}")
    print(f"    y_tau^2 at M_GUT:  {ytau_arr[-1]**2:.6e}")
    print(f"    Log range: {t_GUT - t_SUSY:.2f}")
    print()

    # Mass factor for theta_13
    M_D = y_nu * v_higgs / sqrt(2)
    m3 = M_D**2 / M_R * 1e9
    m2 = m3 / sqrt(R_ihara)
    mass_factor = m2 * m3 / (m3**2 - m2**2)

    # Angular factor
    ang_factor = sin(2*theta12_GUT*RAD) * sin(2*theta23_GUT*RAD) / 2

    # Maximum Delta_theta_13 (for |sin(phase)| = 1)
    delta_th13_max = 1.0/(32*pi**2) * ang_factor * mass_factor * integral_ytau2

    required = abs(theta13_GUT - theta13_obs*RAD)

    print(f"  Mass factor m2*m3/(m3^2-m2^2) = {mass_factor:.6f}")
    print(f"  Angular factor sin(2*th12)*sin(2*th23)/2 = {ang_factor:.6f}")
    print(f"  Max |Delta_theta_13| = {delta_th13_max:.6e} rad = {delta_th13_max*DEG:.4f} deg")
    print(f"  Required shift:      = {required:.6e} rad = {required*DEG:.4f} deg")
    print(f"  Ratio max/required:    {delta_th13_max/required:.4f}")
    print()

    if delta_th13_max > required:
        sin_needed = required / (1.0/(32*pi**2) * ang_factor * mass_factor * integral_ytau2)
        print(f"  SUFFICIENT: required |sin(phi)| = {sin_needed:.4f}")
    else:
        print(f"  INSUFFICIENT for pure RG to close the gap.")
        print(f"  Missing factor: {required/delta_th13_max:.2f}x")

    # =========================================================================
    # Check: what if we use a CONSTANT y_tau = y_tau(M_SUSY)?
    # =========================================================================
    ytau_susy = float(ytau_interp(t_SUSY))
    delta_const = ytau_susy**2 / (32*pi**2) * ang_factor * mass_factor * (t_GUT - t_SUSY)
    print(f"\n  With constant y_tau = y_tau(M_SUSY) = {ytau_susy:.6f}:")
    print(f"    Max shift = {delta_const:.6e} rad = {delta_const*DEG:.4f} deg")

    # =========================================================================
    # Alternative formula: Mei & Zhang (2005) for large tan(beta) MSSM
    # =========================================================================
    print(f"\n  --- ALTERNATIVE: Large tan(beta) radiative correction ---")

    # The dominant contribution from the renormalization of the kappa operator
    # at large tan(beta) is not from the RGE evolution of the angles, but from
    # the FINITE threshold correction at M_SUSY.
    #
    # Casas, Espinosa, Ibarra (2000); Chankowski, Pokorski, Wagner (1993):
    # The correction to the effective neutrino mass matrix from integrating out
    # sparticles gives:
    #
    #   delta(m_nu)_{ij} ~ -y_tau^2/(16*pi^2) * (m_nu)_{i3} * (m_nu)_{3j} / m_nu3
    #                      * ln(M_SUSY / m_stau)
    #
    # This is a rank-1 correction proportional to y_tau^2 that shifts the
    # diagonalization angles.

    # SUSY threshold correction to theta_13
    # From the shift in the (1,3) element of kappa:
    # delta_kappa_13 / kappa_33 ~ y_tau^2/(16pi^2) * (PMNS elements) * log factor
    ytau_susy2 = ytau_susy**2
    log_factor = log(M_SUSY / m_tau)  # ~ ln(1732/1.78) ~ 6.88
    # Actually the relevant log is ln(M_SUSY / m_{stau}) ~ ln(M_SUSY/M_SUSY) ~ 0
    # if staus are at M_SUSY. But for split spectrum, it could be O(1).
    # Use conservative estimate with ln(1) = 0, so no threshold correction.

    print(f"    y_tau(M_SUSY)^2 = {ytau_susy2:.6e}")
    print(f"    Note: SUSY threshold correction depends on sparticle spectrum details.")
    print(f"    For degenerate spectrum (m_sparticle ~ M_SUSY), log = 0 and correction vanishes.")
    print(f"    For split spectrum, correction can be O(y_tau^2/(16pi^2) * ln(ratio)) ~ O(0.01)")


# =============================================================================
# PART 5: SUMMARY
# =============================================================================

def summary(at_mz_zero, at_mz_best, best_phases):
    print_section("PART 5: COMPREHENSIVE SUMMARY")

    if at_mz_best is not None:
        zero_err = abs(at_mz_zero[1]*DEG - theta13_obs)
        best_err = abs(at_mz_best[1]*DEG - theta13_obs)
        if best_err < zero_err:
            at_mz = at_mz_best
            phase_note = f"alpha1_M={best_phases[0]*DEG:.0f}, alpha2_M={best_phases[1]*DEG:.0f}"
        else:
            at_mz = at_mz_zero
            phase_note = "alpha1_M=0, alpha2_M=0"
    else:
        at_mz = at_mz_zero
        phase_note = "alpha1_M=0, alpha2_M=0"

    th13_gut_deg = theta13_GUT * DEG

    print(f"\n  PMNS angles (GUT -> M_Z via MSSM+SM RG, physical y_tau profile):")
    print(f"  Majorana phases: {phase_note}")
    print()
    print(f"  {'Parameter':<16} {'GUT':<12} {'RG @ M_Z':<12} {'Observed':<12} {'Error':<10} {'Note'}")
    print(f"  {'-'*72}")

    items = [
        ("theta_12", theta12_GUT, at_mz[0]*DEG, theta12_obs),
        ("theta_13", th13_gut_deg, at_mz[1]*DEG, theta13_obs),
        ("theta_23", theta23_GUT, at_mz[2]*DEG, theta23_obs),
    ]
    for name, gut, mz, obs in items:
        err = pct(mz, obs)
        err_abs = abs(err)
        note = ""
        if name == "theta_12":
            note = "m1=0 kills RGE"
        elif name == "theta_13":
            if err_abs < 1:
                note = "THEOREM"
            elif err_abs < 5:
                note = "A- (RG partial)"
        elif name == "theta_23":
            note = "needs beyond-TBM"
        print(f"  {name:<16} {gut:<12.4f} {mz:<12.4f} {obs:<12.2f} {err:+.2f}%    {note}")

    print()
    M_D = y_nu * v_higgs / sqrt(2)
    m3_ss = M_D**2 / M_R * 1e9
    m2_ss = m3_ss / sqrt(R_ihara)

    print(f"  Neutrino masses:")
    print(f"  {'Parameter':<16} {'Seesaw':<12} {'RG @ M_Z':<12} {'Observed':<12} {'Error':<10}")
    print(f"  {'-'*60}")
    print(f"  {'m_nu3 (eV)':<16} {m3_ss:<12.6f} {at_mz[5]:<12.6f} {m_nu3_obs:<12.5f} {pct(at_mz[5], m_nu3_obs):+.2f}%")
    print(f"  {'m_nu2 (eV)':<16} {m2_ss:<12.6f} {at_mz[4]:<12.6f} {m_nu2_obs:<12.5f} {pct(at_mz[4], m_nu2_obs):+.2f}%")

    print()
    print(f"  ANALYSIS:")
    print(f"  ---------")

    th13_err = abs(pct(at_mz[1]*DEG, theta13_obs))
    th13_gut_err = abs(pct(th13_gut_deg, theta13_obs))
    improvement = th13_gut_err - th13_err

    print(f"  theta_13: GUT gap = {th13_gut_err:.2f}%, after RG = {th13_err:.2f}% "
          f"(improved by {improvement:.2f}%)")
    print()

    print(f"  PHYSICS CONCLUSIONS:")
    print(f"  1. The RG running of PMNS angles in the MSSM is driven by y_tau^2.")
    print(f"  2. For normal hierarchy with m1 = 0, the theta_12 RGE vanishes exactly.")
    print(f"  3. The theta_13 running depends on the Majorana phase combination")
    print(f"     sin(alpha1 - alpha2 - 2*delta_CP), with the m2*m3/dm2_32 mass factor.")
    print(f"  4. With large tan(beta) = {tan_beta}, y_tau is O(0.5) across the MSSM range:")
    print(f"     y_tau(M_GUT)={float(ytau_interp(log(M_GUT))):.4f}, y_tau(M_SUSY)={float(ytau_interp(log(M_SUSY))):.4f}.")
    print(f"  5. The integral of y_tau^2 is large (~7.7), but the mass factor")
    print(f"     m2*m3/(m3^2-m2^2) ~ 0.18 and the 1/(32*pi^2) loop suppression")
    print(f"     limit the total shift to ~0.12 deg (max) vs ~0.39 deg needed.")
    print(f"  6. theta_23 does not run to the observed 49.2 deg from TBM (45 deg);")
    print(f"     the GUT-scale value must already be non-TBM, or there are")
    print(f"     threshold/higher-order corrections.")
    print()

    if th13_err < 1.0:
        print(f"  VERDICT: theta_13 gap CLOSED. Upgrade to THEOREM.")
    elif th13_err < 5.0:
        print(f"  VERDICT: theta_13 remains A- (gap {th13_err:.2f}%).")
        print(f"  The RG running contributes {improvement:.2f}% improvement.")
        print(f"  Remaining gap may come from:")
        print(f"    - SUSY threshold corrections (spectrum-dependent)")
        print(f"    - 2-loop contributions")
        print(f"    - Precise Majorana phase values")
        print(f"    - GUT-scale threshold corrections at M_GUT")
    else:
        print(f"  VERDICT: theta_13 gap not closed ({th13_err:.2f}%).")

    return ytau_interp


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("  PMNS RG RUNNING: M_GUT -> M_Z (PHYSICAL y_tau PROFILE)")
    print("  Goal: close the ~4.5% gap in theta_13")
    print("=" * 78)

    print(f"\n  Framework inputs:")
    print(f"    tan(beta) = {tan_beta}")
    print(f"    M_SUSY    = {M_SUSY:.2f} GeV")
    print(f"    M_GUT     = {M_GUT:.2e} GeV")
    print(f"    M_R       = {M_R:.4e} GeV")
    print(f"    alpha_GUT = 1/{alpha_GUT_inv}")
    print(f"    V_us      = {V_us:.6f}")
    print(f"    theta_13(GUT) = {theta13_GUT*DEG:.4f} deg")
    print(f"    theta_13(obs) = {theta13_obs:.2f} deg")
    print(f"    Gap = {pct(theta13_GUT*DEG, theta13_obs):+.2f}%")

    sol_mssm, sol_sm, ytau_interp, yt_interp, a1_interp, a2_interp = build_ytau_profile()

    print_section("PART 2: PMNS RUNNING (zero Majorana phases)")
    at_mz_zero = run_pmns(sol_mssm, sol_sm, ytau_interp, yt_interp,
                          a1_interp, a2_interp, verbose=True)

    best_phases, at_mz_best = phase_scan(sol_mssm, sol_sm, ytau_interp,
                                          yt_interp, a1_interp, a2_interp)

    analytic_estimates(ytau_interp)

    ytau_interp = summary(at_mz_zero, at_mz_best, best_phases)
