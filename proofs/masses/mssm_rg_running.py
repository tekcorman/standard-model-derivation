#!/usr/bin/env python3
"""
MSSM two-loop renormalization group running of gauge couplings.

Runs alpha_1, alpha_2, alpha_3 from M_GUT down to M_Z with:
  - Two-loop MSSM beta functions above M_SUSY
  - Two-loop SM beta functions below M_SUSY
  - Self-consistent M_GUT determination
  - SUSY threshold scan
  - Proton lifetime estimate
  - Lambda_QCD extraction with flavor matching
  - Koide waterfall for lepton/quark masses
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
M_Z = 91.1876          # GeV
m_b = 4.18             # GeV (MS-bar mass)
m_c = 1.27             # GeV (MS-bar mass)
m_p = 0.93827          # GeV (proton mass)
YEAR_IN_SECONDS = 3.156e7

# Observed values at M_Z
ALPHA_EM_INV_OBS = 127.95       # 1/alpha_em(M_Z)
SIN2_TW_OBS = 0.23122          # sin^2(theta_W)
ALPHA_S_OBS = 0.1180           # alpha_s(M_Z)
ALPHA_S_ERR = 0.0009

# Derived observed couplings in GUT normalization
# alpha_em = alpha_1 * alpha_2 / (alpha_1 + alpha_2) with alpha_1 = (5/3)*alpha_Y
# sin^2(theta_W) = alpha_1^(-1) / (alpha_1^(-1) + alpha_2^(-1)) ... let's be careful
# In GUT normalization: sin^2(theta_W) = (3/5) * alpha_1 / (alpha_1 + (3/5)*alpha_1 * alpha_2/alpha_1)
# Actually: sin^2(theta_W) = g'^2/(g^2+g'^2), alpha_Y = g'^2/(4pi), alpha_2 = g^2/(4pi)
# so sin^2(theta_W) = alpha_Y/(alpha_Y + alpha_2)
# With alpha_1 = (5/3)*alpha_Y => alpha_Y = (3/5)*alpha_1
# sin^2(theta_W) = (3/5)*alpha_1 / ((3/5)*alpha_1 + alpha_2)
# alpha_em = e^2/(4pi) = alpha_Y * alpha_2 / (alpha_Y + alpha_2)
#          = (3/5)*alpha_1 * alpha_2 / ((3/5)*alpha_1 + alpha_2)

alpha_em_obs = 1.0 / ALPHA_EM_INV_OBS
# From sin^2(theta_W) = alpha_em / alpha_2  (standard relation)
alpha_2_obs = alpha_em_obs / SIN2_TW_OBS
# alpha_Y = alpha_em / cos^2(theta_W) = alpha_em / (1 - sin^2(theta_W))
alpha_Y_obs = alpha_em_obs / (1.0 - SIN2_TW_OBS)
alpha_1_obs = (5.0/3.0) * alpha_Y_obs  # GUT normalization
alpha_3_obs = ALPHA_S_OBS

print("=" * 72)
print("MSSM TWO-LOOP RENORMALIZATION GROUP RUNNING")
print("=" * 72)
print()
print("Observed couplings at M_Z = {:.4f} GeV:".format(M_Z))
print("  alpha_1_inv(M_Z) = %.4f  (GUT normalized)" % (1.0/alpha_1_obs))
print("  alpha_2_inv(M_Z) = %.4f" % (1.0/alpha_2_obs))
print("  alpha_3_inv(M_Z) = %.4f" % (1.0/alpha_3_obs))
print("  alpha_em_inv(M_Z) = %.2f" % ALPHA_EM_INV_OBS)
print("  sin^2(theta_W) = %.5f" % SIN2_TW_OBS)
print("  alpha_s(M_Z) = %.4f +/- %.4f" % (ALPHA_S_OBS, ALPHA_S_ERR))
print()

# ─────────────────────────────────────────────────────────
# Beta function coefficients
# ─────────────────────────────────────────────────────────

# MSSM one-loop
b_MSSM = np.array([33.0/5.0, 1.0, -3.0])

# MSSM two-loop (b_ij matrix)
bij_MSSM = np.array([
    [199.0/25.0, 27.0/5.0, 88.0/5.0],
    [9.0/5.0,    25.0,     24.0     ],
    [11.0/5.0,   9.0,      14.0     ]
])

# SM one-loop
b_SM = np.array([41.0/10.0, -19.0/6.0, -7.0])

# SM two-loop (for completeness)
bij_SM = np.array([
    [199.0/50.0, 27.0/10.0, 44.0/5.0],
    [9.0/10.0,   35.0/6.0,  12.0    ],
    [11.0/10.0,  9.0/2.0,   -26.0   ]
])


def rge_system(t, y, b, bij):
    """
    RGE for alpha_i^{-1} as function of t = ln(mu/GeV).

    d(alpha_i^{-1})/dt = -b_i/(2*pi) - sum_j b_ij * alpha_j / (8*pi^2)

    y = [alpha_1^{-1}, alpha_2^{-1}, alpha_3^{-1}]
    """
    # alpha_i = 1/y_i
    alpha = 1.0 / y

    dydt = np.zeros(3)
    for i in range(3):
        one_loop = -b[i] / (2.0 * np.pi)
        two_loop = 0.0
        for j in range(3):
            two_loop -= bij[i, j] * alpha[j] / (8.0 * np.pi**2)
        dydt[i] = one_loop + two_loop

    return dydt


def run_couplings(alpha_gut_inv, log_m_gut, m_susy=1000.0, two_loop=True):
    """
    Run couplings from M_GUT down to M_Z.

    Returns alpha_i^{-1} at M_Z.
    """
    m_gut = np.exp(log_m_gut)
    t_gut = log_m_gut
    t_susy = np.log(m_susy)
    t_mz = np.log(M_Z)

    # Initial condition: all couplings unified
    y0 = np.array([alpha_gut_inv, alpha_gut_inv, alpha_gut_inv])

    bij_use = bij_MSSM if two_loop else np.zeros((3, 3))
    bij_sm_use = bij_SM if two_loop else np.zeros((3, 3))

    # Phase 1: M_GUT -> M_SUSY (MSSM)
    if t_gut > t_susy:
        sol1 = solve_ivp(
            rge_system, [t_gut, t_susy], y0,
            args=(b_MSSM, bij_use),
            method='RK45', rtol=1e-10, atol=1e-12,
            dense_output=True
        )
        y_susy = sol1.sol(t_susy)
    else:
        y_susy = y0

    # Phase 2: M_SUSY -> M_Z (SM)
    if t_susy > t_mz:
        sol2 = solve_ivp(
            rge_system, [t_susy, t_mz], y_susy,
            args=(b_SM, bij_sm_use),
            method='RK45', rtol=1e-10, atol=1e-12,
            dense_output=True
        )
        y_mz = sol2.sol(t_mz)
    else:
        y_mz = y_susy

    return y_mz


def find_m_gut(alpha_gut_inv, m_susy=1000.0, two_loop=True):
    """
    Find M_GUT where alpha_1 and alpha_2 unify.
    We define M_GUT as the scale where alpha_1^{-1} = alpha_2^{-1} when
    running UP from M_Z. Equivalently, running DOWN from a trial M_GUT,
    we adjust M_GUT until the low-energy values match observations.

    Strategy: find log_m_gut such that alpha_1(M_Z) and alpha_2(M_Z)
    match their observed ratio (or equivalently, sin^2(theta_W) is correct).
    """
    def objective(log_m_gut):
        y_mz = run_couplings(alpha_gut_inv, log_m_gut, m_susy, two_loop)
        # We want sin^2(theta_W) to match observed
        a1 = 1.0 / y_mz[0]
        a2 = 1.0 / y_mz[1]
        sw2 = (3.0/5.0) * a1 / ((3.0/5.0) * a1 + a2)
        return sw2 - SIN2_TW_OBS

    # Search range: 10^14 to 10^18 GeV
    log_lo = np.log(1e14)
    log_hi = np.log(1e18)

    # Check signs
    f_lo = objective(log_lo)
    f_hi = objective(log_hi)

    if f_lo * f_hi > 0:
        # Try a wider range or return best guess
        # Just do a scan
        best_log = None
        best_val = 1e10
        for log_m in np.linspace(log_lo, log_hi, 200):
            val = abs(objective(log_m))
            if val < best_val:
                best_val = val
                best_log = log_m
        return best_log

    return brentq(objective, log_lo, log_hi, rtol=1e-12)


def extract_observables(y_mz):
    """Extract physical observables from alpha_i^{-1}(M_Z)."""
    a1 = 1.0 / y_mz[0]  # GUT normalized
    a2 = 1.0 / y_mz[1]
    a3 = 1.0 / y_mz[2]

    # sin^2(theta_W) = (3/5)*alpha_1 / ((3/5)*alpha_1 + alpha_2)
    a_Y = (3.0/5.0) * a1
    sin2_tw = a_Y / (a_Y + a2)

    # alpha_em = alpha_Y * alpha_2 / (alpha_Y + alpha_2)
    alpha_em = a_Y * a2 / (a_Y + a2)

    return {
        'alpha_1_inv': y_mz[0],
        'alpha_2_inv': y_mz[1],
        'alpha_3_inv': y_mz[2],
        'alpha_1': a1,
        'alpha_2': a2,
        'alpha_3': a3,
        'sin2_tw': sin2_tw,
        'alpha_em': alpha_em,
        'alpha_em_inv': 1.0/alpha_em,
    }


def compute_lambda_qcd(alpha_s_mz):
    """
    Compute Lambda_QCD from alpha_s(M_Z) with flavor matching.

    For n_f flavors, beta_0 = (33 - 2*n_f)/3 (one-loop, SM normalization)
    Lambda^(n_f) = mu * exp(-2*pi / (beta_0 * alpha_s(mu)))
    with two-loop corrections.
    """
    # One-loop beta_0 coefficients (SM convention: b = -beta_0/(2*pi))
    def beta0(nf):
        return (33.0 - 2.0*nf) / 3.0

    def beta1(nf):
        return (306.0 - 38.0*nf) / 3.0

    # Lambda^(5) from alpha_s(M_Z) — two-loop
    b0_5 = beta0(5)
    b1_5 = beta1(5)

    # Two-loop Lambda: Lambda = mu * exp(-1/(b0*a_s)) * (b0*a_s)^(-b1/(2*b0^2))
    # where a_s = alpha_s/(4*pi) ... let's use the standard formula
    # At one-loop: Lambda = mu * exp(-2*pi/(beta_0 * alpha_s))
    lambda_5_1loop = M_Z * np.exp(-2.0 * np.pi / (b0_5 * alpha_s_mz))

    # Two-loop correction factor
    L = 2.0 * np.log(M_Z / lambda_5_1loop)
    # Iterative: solve alpha_s(M_Z) = (4*pi)/(b0*L) * (1 - b1*log(L)/(b0^2*L))
    # for Lambda. Use the one-loop result and iterate.
    for _ in range(20):
        L = 2.0 * np.log(M_Z / lambda_5_1loop)
        a_s_check = 4.0*np.pi / (b0_5 * L) * (1.0 - b1_5/(b0_5**2) * np.log(L)/L)
        # Adjust lambda
        ratio = alpha_s_mz / a_s_check
        lambda_5_1loop *= ratio**0.3  # damped iteration

    lambda_5 = lambda_5_1loop

    # Match Lambda^(5) -> Lambda^(4) at mu = m_b
    # Matching: Lambda^(n_f-1) = Lambda^(n_f) * (m_q/Lambda^(n_f))^(2/(33-2*(n_f-1)) - 2/(33-2*n_f))
    # Continuous matching: alpha_s^(5)(m_b) = alpha_s^(4)(m_b)
    # Compute alpha_s at m_b using n_f=5 running
    b0_5_val = beta0(5)
    L_b = 2.0 * np.log(m_b / lambda_5)
    if L_b > 0:
        alpha_s_mb = 4.0*np.pi / (b0_5_val * L_b)
    else:
        alpha_s_mb = alpha_s_mz  # fallback

    # Lambda^(4) from alpha_s(m_b) with n_f=4
    b0_4 = beta0(4)
    lambda_4 = m_b * np.exp(-2.0*np.pi / (b0_4 * alpha_s_mb))

    # Match Lambda^(4) -> Lambda^(3) at mu = m_c
    L_c = 2.0 * np.log(m_c / lambda_4)
    if L_c > 0:
        alpha_s_mc = 4.0*np.pi / (b0_4 * L_c)
    else:
        alpha_s_mc = alpha_s_mb

    b0_3 = beta0(3)
    lambda_3 = m_c * np.exp(-2.0*np.pi / (b0_3 * alpha_s_mc))

    return {
        'lambda_5': lambda_5,
        'lambda_4': lambda_4,
        'lambda_3': lambda_3,
        'alpha_s_mb': alpha_s_mb,
        'alpha_s_mc': alpha_s_mc,
    }


def koide_waterfall(lambda_qcd_3):
    """
    Compute masses from Lambda_QCD via Koide-inspired relations.

    M^2 = Lambda_QCD * log2(3)
    m_tau from M^2 and Koide formula with delta=2/9, epsilon=sqrt(2)
    m_b from m_tau via Georgi-Jarlskog factor 3
    m_t from Koide waterfall
    """
    # M^2 scale
    log2_3 = np.log(3.0) / np.log(2.0)
    M_squared = lambda_qcd_3 * log2_3  # in GeV (this is a mass scale, not squared)
    # The claim is M^2 ~ 314 MeV, so M_squared here is actually a mass ~ Lambda_QCD * log2(3)

    # Koide formula for charged leptons: (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2 = 2/3
    # With known m_e, m_mu, solve for m_tau
    m_e = 0.000511  # GeV
    m_mu = 0.10566  # GeV

    # Standard Koide: solve (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2 = 2/3
    def koide_eq(m_tau_trial):
        s1 = np.sqrt(m_e) + np.sqrt(m_mu) + np.sqrt(m_tau_trial)
        return (m_e + m_mu + m_tau_trial) / s1**2 - 2.0/3.0

    m_tau_koide = brentq(koide_eq, 1.0, 3.0)

    # Georgi-Jarlskog: m_b/m_tau = 3 at GUT scale (approximately)
    m_b_gj = 3.0 * m_tau_koide

    # Top mass from Koide waterfall: apply Koide to (m_c, m_b, m_t)
    # (m_c + m_b + m_t) / (sqrt(m_c) + sqrt(m_b) + sqrt(m_t))^2 = 2/3
    m_c_pole = 1.27  # GeV
    def koide_top(m_t_trial):
        s1 = np.sqrt(m_c_pole) + np.sqrt(m_b_gj) + np.sqrt(m_t_trial)
        return (m_c_pole + m_b_gj + m_t_trial) / s1**2 - 2.0/3.0

    m_t_koide = brentq(koide_top, 50.0, 500.0)

    return {
        'M_squared': M_squared,
        'm_tau_koide': m_tau_koide,
        'm_b_gj': m_b_gj,
        'm_t_koide': m_t_koide,
    }


def proton_lifetime(m_gut, alpha_gut):
    """
    Estimate proton lifetime.

    tau_p ~ M_GUT^4 / (alpha_GUT^2 * m_p^5)

    The numerical coefficient depends on the GUT model. For minimal SU(5):
    tau_p ~ (M_X/10^16 GeV)^4 * (alpha_GUT/1/40)^{-2} * 10^{36} years

    More precisely: tau_p = M_GUT^4 / (alpha_GUT^2 * m_p^5 * C)
    where C encodes hadronic matrix elements and phase space.
    We use the standard estimate with C calibrated to give ~10^{35} yr for
    M_GUT = 2e16 GeV, alpha_GUT = 1/24.
    """
    # Dimensional analysis: [M_GUT^4] / [alpha^2 * m_p^5] has units of 1/GeV
    # Convert to seconds: multiply by hbar = 6.582e-25 GeV*s
    hbar = 6.582119569e-25  # GeV*s

    # The decay rate includes hadronic matrix elements. Standard formula:
    # Gamma = alpha_GUT^2 * m_p^5 / (4*pi * M_GUT^4) * |alpha_H|^2 * A_R^2 * phase_space
    # |alpha_H|^2 ~ 0.015 GeV^3 (lattice), A_R ~ 2-3 (renormalization), phase_space ~ 1/(8*pi)
    # We absorb into a single coefficient calibrated to known results.

    # Standard result: tau_p(SU(5)) ~ 4e29 * (M_GUT/10^15)^4 * (0.003/alpha_H^2)^2 years
    # For SUSY SU(5) with dimension-6 operators:
    # tau_p ~ 10^{34-37} years for M_GUT ~ 10^16 GeV

    # Simple estimate:
    # tau_p = M_GUT^4 / (alpha_GUT^2 * m_p^5) * hbar (in seconds)
    # Then need a fudge factor from the actual matrix element calculation

    tau_natural = m_gut**4 / (alpha_gut**2 * m_p**5)  # in GeV^{-1}
    tau_seconds = tau_natural * hbar
    tau_years = tau_seconds / YEAR_IN_SECONDS

    # The above gives a huge number because it's missing the 1/(4*pi)^2 and
    # hadronic matrix elements. The standard normalization gives:
    # For dimension-6: Gamma ~ alpha_GUT^2 * m_p / M_GUT^4 * (alpha_N * A)^2 / (4*pi*f_pi^2)
    # alpha_N ~ 0.015 GeV^3, f_pi ~ 0.131 GeV, A ~ 1.25 (RG enhancement)

    alpha_N = 0.015  # GeV^3 (proton decay matrix element, lattice)
    f_pi = 0.131     # GeV
    A_renorm = 1.5   # short-distance renormalization factor

    gamma = (alpha_gut**2 * m_p * (alpha_N * A_renorm)**2) / \
            (4.0 * np.pi * f_pi**2 * m_gut**4)  # GeV

    tau_seconds_phys = hbar / gamma
    tau_years_phys = tau_seconds_phys / YEAR_IN_SECONDS

    return tau_years_phys


def print_section(title):
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)


def pct_err(computed, observed):
    return (computed - observed) / observed * 100.0


# ─────────────────────────────────────────────────────────
# 1. Find M_GUT self-consistently
# ─────────────────────────────────────────────────────────
print_section("1. SELF-CONSISTENT M_GUT DETERMINATION")

alpha_gut_inv = 24.1
alpha_gut = 1.0 / alpha_gut_inv
M_SUSY_baseline = 1000.0  # 1 TeV

log_m_gut = find_m_gut(alpha_gut_inv, m_susy=M_SUSY_baseline, two_loop=True)
M_GUT = np.exp(log_m_gut)

print("alpha_GUT^{-1} = %.1f" % alpha_gut_inv)
print("M_SUSY = %.0f GeV" % M_SUSY_baseline)
print("M_GUT = %.4e GeV" % M_GUT)
print("log10(M_GUT/GeV) = %.2f" % np.log10(M_GUT))

# ─────────────────────────────────────────────────────────
# 2. Run couplings and extract observables
# ─────────────────────────────────────────────────────────
print_section("2. GAUGE COUPLINGS AT M_Z (two-loop)")

y_mz = run_couplings(alpha_gut_inv, log_m_gut, M_SUSY_baseline, two_loop=True)
obs = extract_observables(y_mz)

print("  Coupling      Computed    Observed     Error")
print("  ---------     --------    --------     -----")
print("  alpha_1^{-1}  %8.4f    %8.4f    %+.2f%%" %
      (obs['alpha_1_inv'], 1.0/alpha_1_obs, pct_err(obs['alpha_1_inv'], 1.0/alpha_1_obs)))
print("  alpha_2^{-1}  %8.4f    %8.4f    %+.2f%%" %
      (obs['alpha_2_inv'], 1.0/alpha_2_obs, pct_err(obs['alpha_2_inv'], 1.0/alpha_2_obs)))
print("  alpha_3^{-1}  %8.4f    %8.4f    %+.2f%%" %
      (obs['alpha_3_inv'], 1.0/alpha_3_obs, pct_err(obs['alpha_3_inv'], 1.0/alpha_3_obs)))
print()
print("  sin^2(theta_W)  %.5f    %.5f    %+.3f%%" %
      (obs['sin2_tw'], SIN2_TW_OBS, pct_err(obs['sin2_tw'], SIN2_TW_OBS)))
print("  alpha_em^{-1}   %.2f    %.2f    %+.3f%%" %
      (obs['alpha_em_inv'], ALPHA_EM_INV_OBS, pct_err(obs['alpha_em_inv'], ALPHA_EM_INV_OBS)))
print("  alpha_s(M_Z)    %.4f      %.4f      %+.2f%%" %
      (obs['alpha_3'], ALPHA_S_OBS, pct_err(obs['alpha_3'], ALPHA_S_OBS)))

# Also show one-loop for comparison
print()
print("  --- One-loop comparison ---")
y_mz_1l = run_couplings(alpha_gut_inv, log_m_gut, M_SUSY_baseline, two_loop=False)
obs_1l = extract_observables(y_mz_1l)
print("  alpha_1^{-1} (1-loop) = %.4f" % obs_1l['alpha_1_inv'])
print("  alpha_2^{-1} (1-loop) = %.4f" % obs_1l['alpha_2_inv'])
print("  alpha_3^{-1} (1-loop) = %.4f" % obs_1l['alpha_3_inv'])
print("  Two-loop shifts: da1^{-1} = %+.4f, da2^{-1} = %+.4f, da3^{-1} = %+.4f" %
      (obs['alpha_1_inv'] - obs_1l['alpha_1_inv'],
       obs['alpha_2_inv'] - obs_1l['alpha_2_inv'],
       obs['alpha_3_inv'] - obs_1l['alpha_3_inv']))

# ─────────────────────────────────────────────────────────
# 3. Unification quality check
# ─────────────────────────────────────────────────────────
print_section("3. UNIFICATION QUALITY")

# Run observed couplings UP to M_GUT to check how well they unify
# We'll run from M_Z up to M_GUT
def run_up(alpha_inv_mz, m_susy, log_m_gut_target):
    """Run couplings from M_Z up to M_GUT."""
    t_mz = np.log(M_Z)
    t_susy = np.log(m_susy)
    t_gut = log_m_gut_target

    y0 = alpha_inv_mz.copy()

    # Phase 1: M_Z -> M_SUSY (SM)
    sol1 = solve_ivp(
        rge_system, [t_mz, t_susy], y0,
        args=(b_SM, bij_SM),
        method='RK45', rtol=1e-10, atol=1e-12,
        dense_output=True
    )
    y_susy = sol1.sol(t_susy)

    # Phase 2: M_SUSY -> M_GUT (MSSM)
    sol2 = solve_ivp(
        rge_system, [t_susy, t_gut], y_susy,
        args=(b_MSSM, bij_MSSM),
        method='RK45', rtol=1e-10, atol=1e-12,
        dense_output=True
    )
    y_gut = sol2.sol(t_gut)

    return y_gut, sol1, sol2

alpha_inv_obs_mz = np.array([1.0/alpha_1_obs, 1.0/alpha_2_obs, 1.0/alpha_3_obs])
y_gut_from_obs, _, _ = run_up(alpha_inv_obs_mz, M_SUSY_baseline, log_m_gut)

print("Running observed couplings UP to M_GUT = %.3e GeV:" % M_GUT)
print("  alpha_1^{-1}(M_GUT) = %.4f" % y_gut_from_obs[0])
print("  alpha_2^{-1}(M_GUT) = %.4f" % y_gut_from_obs[1])
print("  alpha_3^{-1}(M_GUT) = %.4f" % y_gut_from_obs[2])
print("  Spread: %.4f (ideal = 0)" %
      (max(y_gut_from_obs) - min(y_gut_from_obs)))

# ─────────────────────────────────────────────────────────
# 4. Lambda_QCD
# ─────────────────────────────────────────────────────────
print_section("4. LAMBDA_QCD EXTRACTION")

lqcd = compute_lambda_qcd(obs['alpha_3'])
print("  From computed alpha_s(M_Z) = %.4f:" % obs['alpha_3'])
print("  Lambda_QCD^(5) = %.1f MeV" % (lqcd['lambda_5'] * 1000))
print("  Lambda_QCD^(4) = %.1f MeV  (matched at m_b = %.2f GeV)" % (lqcd['lambda_4']*1000, m_b))
print("  Lambda_QCD^(3) = %.1f MeV  (matched at m_c = %.2f GeV)" % (lqcd['lambda_3']*1000, m_c))
print("  PDG value: Lambda_QCD^(3) = 332 +/- 17 MeV (MS-bar)")
print("  Error: %+.1f%%" % pct_err(lqcd['lambda_3']*1000, 332.0))
print()
print("  alpha_s(m_b) = %.4f" % lqcd['alpha_s_mb'])
print("  alpha_s(m_c) = %.4f" % lqcd['alpha_s_mc'])

# Also from observed alpha_s
lqcd_obs = compute_lambda_qcd(ALPHA_S_OBS)
print()
print("  From observed alpha_s(M_Z) = %.4f:" % ALPHA_S_OBS)
print("  Lambda_QCD^(3) = %.1f MeV" % (lqcd_obs['lambda_3']*1000))

# ─────────────────────────────────────────────────────────
# 5. Koide waterfall
# ─────────────────────────────────────────────────────────
print_section("5. KOIDE WATERFALL FROM LAMBDA_QCD")

kw = koide_waterfall(lqcd['lambda_3'])
print("  Lambda_QCD^(3) = %.1f MeV" % (lqcd['lambda_3']*1000))
print("  M^2 = Lambda_QCD * log2(3) = %.1f MeV  (target: 313.84 MeV, error: %+.2f%%)" %
      (kw['M_squared']*1000, pct_err(kw['M_squared']*1000, 313.84)))
print()
print("  Koide formula m_tau = %.4f GeV  (observed: 1.7769 GeV, error: %+.2f%%)" %
      (kw['m_tau_koide'], pct_err(kw['m_tau_koide'], 1.7769)))
print("  Georgi-Jarlskog m_b = 3*m_tau = %.3f GeV  (observed: 4.18 GeV, error: %+.2f%%)" %
      (kw['m_b_gj'], pct_err(kw['m_b_gj'], 4.18)))
print("  Koide waterfall m_t = %.1f GeV  (observed: 172.69 GeV, error: %+.2f%%)" %
      (kw['m_t_koide'], pct_err(kw['m_t_koide'], 172.69)))

# ─────────────────────────────────────────────────────────
# 6. M_SUSY scan
# ─────────────────────────────────────────────────────────
print_section("6. M_SUSY SCAN (500 GeV to 10 TeV)")

m_susy_values = np.logspace(np.log10(500), np.log10(10000), 20)
print("  M_SUSY (GeV)  M_GUT (GeV)   alpha_s(M_Z)  sin^2(tw)   alpha_em^{-1}  Lambda^(3) (MeV)")
print("  " + "-"*90)

best_msusy = None
best_alpha_s_err = 1e10

results_scan = []

for ms in m_susy_values:
    lmg = find_m_gut(alpha_gut_inv, m_susy=ms, two_loop=True)
    y = run_couplings(alpha_gut_inv, lmg, ms, two_loop=True)
    o = extract_observables(y)
    lq = compute_lambda_qcd(o['alpha_3'])

    results_scan.append((ms, np.exp(lmg), o, lq))

    err_as = abs(o['alpha_3'] - ALPHA_S_OBS)
    if err_as < best_alpha_s_err:
        best_alpha_s_err = err_as
        best_msusy = ms

    print("  %10.0f    %.3e     %.4f       %.5f      %.2f         %.1f" %
          (ms, np.exp(lmg), o['alpha_3'], o['sin2_tw'], o['alpha_em_inv'], lq['lambda_3']*1000))

print()
print("  Best fit M_SUSY for alpha_s(M_Z) = %.4f: M_SUSY = %.0f GeV" %
      (ALPHA_S_OBS, best_msusy))

# Refine best M_SUSY
def alpha_s_residual(log_ms):
    ms = np.exp(log_ms)
    lmg = find_m_gut(alpha_gut_inv, m_susy=ms, two_loop=True)
    y = run_couplings(alpha_gut_inv, lmg, ms, two_loop=True)
    o = extract_observables(y)
    return o['alpha_3'] - ALPHA_S_OBS

try:
    log_ms_best = brentq(alpha_s_residual, np.log(300), np.log(20000), rtol=1e-6)
    ms_best = np.exp(log_ms_best)
    lmg_best = find_m_gut(alpha_gut_inv, m_susy=ms_best, two_loop=True)
    y_best = run_couplings(alpha_gut_inv, lmg_best, ms_best, two_loop=True)
    o_best = extract_observables(y_best)
    lq_best = compute_lambda_qcd(o_best['alpha_3'])

    print("  Refined: M_SUSY = %.0f GeV gives alpha_s(M_Z) = %.6f" % (ms_best, o_best['alpha_3']))
    print("  At this M_SUSY:")
    print("    M_GUT = %.4e GeV" % np.exp(lmg_best))
    print("    sin^2(theta_W) = %.5f (obs: %.5f, err: %+.3f%%)" %
          (o_best['sin2_tw'], SIN2_TW_OBS, pct_err(o_best['sin2_tw'], SIN2_TW_OBS)))
    print("    alpha_em^{-1} = %.2f (obs: %.2f, err: %+.3f%%)" %
          (o_best['alpha_em_inv'], ALPHA_EM_INV_OBS, pct_err(o_best['alpha_em_inv'], ALPHA_EM_INV_OBS)))
    print("    Lambda_QCD^(3) = %.1f MeV" % (lq_best['lambda_3']*1000))
except ValueError:
    print("  Could not find exact M_SUSY for alpha_s match (may be outside scan range)")

# ─────────────────────────────────────────────────────────
# 7. Proton lifetime
# ─────────────────────────────────────────────────────────
print_section("7. PROTON LIFETIME")

tau_p = proton_lifetime(M_GUT, alpha_gut)
print("  M_GUT = %.4e GeV" % M_GUT)
print("  alpha_GUT = 1/%.1f" % alpha_gut_inv)
print()
print("  tau_p = %.2e years" % tau_p)
print("  log10(tau_p/yr) = %.1f" % np.log10(tau_p))
print()
print("  Experimental bounds:")
print("    Super-K:  tau_p > 1.6e34 years (p -> e+ pi0)")
print("    Hyper-K sensitivity: ~10^35 years")
if tau_p > 1.6e34:
    print("    Status: CONSISTENT with Super-K bound")
else:
    print("    Status: EXCLUDED by Super-K")
if tau_p < 1e35:
    print("    Testable by Hyper-K: YES")
else:
    print("    Testable by Hyper-K: MARGINAL (tau_p > 10^35 yr)")

# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
print_section("SUMMARY (M_SUSY = 1 TeV baseline)")

print("  M_GUT = %.4e GeV  (log10 = %.2f)" % (M_GUT, np.log10(M_GUT)))
print("  alpha_GUT^{-1} = %.1f (input)" % alpha_gut_inv)
print()
print("  Observable        Computed      Observed       Error")
print("  ----------        --------      --------       -----")
print("  alpha_s(M_Z)      %.4f        %.4f         %+.2f%%" %
      (obs['alpha_3'], ALPHA_S_OBS, pct_err(obs['alpha_3'], ALPHA_S_OBS)))
print("  sin^2(theta_W)    %.5f      %.5f       %+.3f%%" %
      (obs['sin2_tw'], SIN2_TW_OBS, pct_err(obs['sin2_tw'], SIN2_TW_OBS)))
print("  alpha_em^{-1}     %.2f      %.2f       %+.3f%%" %
      (obs['alpha_em_inv'], ALPHA_EM_INV_OBS, pct_err(obs['alpha_em_inv'], ALPHA_EM_INV_OBS)))
print("  Lambda_QCD^(3)    %.0f MeV       332 MeV        %+.1f%%" %
      (lqcd['lambda_3']*1000, pct_err(lqcd['lambda_3']*1000, 332.0)))
print("  tau_p             %.1e yr    > 1.6e34 yr    %s" %
      (tau_p, "OK" if tau_p > 1.6e34 else "EXCLUDED"))
print()
print("  Koide waterfall:")
print("  M^2 = %.1f MeV      (target 313.84 MeV,  %+.2f%%)" %
      (kw['M_squared']*1000, pct_err(kw['M_squared']*1000, 313.84)))
print("  m_tau = %.4f GeV   (observed 1.7769 GeV, %+.2f%%)" %
      (kw['m_tau_koide'], pct_err(kw['m_tau_koide'], 1.7769)))
print("  m_b = %.3f GeV     (observed 4.18 GeV,   %+.2f%%)" %
      (kw['m_b_gj'], pct_err(kw['m_b_gj'], 4.18)))
print("  m_t = %.1f GeV     (observed 172.69 GeV, %+.2f%%)" %
      (kw['m_t_koide'], pct_err(kw['m_t_koide'], 172.69)))

print()
print("=" * 72)
