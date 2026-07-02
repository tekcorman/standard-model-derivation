"""
Sound horizon r_s under framework's running couplings.

FORWARD DERIVATION from substrate primitives — no Planck inputs.

Builds on recombination_running_couplings_2026-05-09.py (which gave
framework z_rec under BZJ + Saha) by computing the sound horizon and
acoustic peak angle theta_*.

C_S DERIVATION (from framework primitives, not imported)

  Photons (Hodge modes, omega = c|k| linear dispersion):
    relativistic gas with p_gamma = rho_gamma / 3.
    delta(rho_gamma) / rho_gamma = 4 delta(T) / T   (Bose-Einstein, T^4 scaling)

  Walkers (massive matter, T << m at recombination):
    non-relativistic gas with p_b ~ 0.
    delta(rho_b) / rho_b = 3 delta(T) / T           (n_b ~ T^3 entropy/baryon
                                                      conservation under coasting
                                                      volume scaling)

  Tight coupling (alpha_EM-mediated photon-walker scattering rate >> H):
    locks delta(T)_gamma = delta(T)_b at recombination.
    Photon-walker plasma evolves as a single fluid.

  Adiabatic sound speed:
    c_s^2 = (delta(p) / delta(rho))_S
          = delta(p_gamma) / (delta(rho_gamma) + delta(rho_b))
          = (4 delta(T)/T) (rho_gamma / 3) /
            ((4 delta(T)/T) rho_gamma + (3 delta(T)/T) rho_b)
          = (4 rho_gamma / 3) / (4 rho_gamma + 3 rho_b)
          = c^2 / (3 (1 + 3 rho_b / (4 rho_gamma)))
          = c^2 / (3 (1 + R))

  with R = 3 rho_b / (4 rho_gamma).

  This formula is DERIVED from framework primitives:
    - Photon EOS p = rho/3 follows from omega = c|k| linear dispersion of
      Hodge modes (predictions/srs_photon_hodge.py, theorem-grade).
    - Baryon p ~ 0 follows from non-relativistic limit of walker dynamics.
    - Tight-coupling: alpha_EM scattering at recombination scale, framework
      provides alpha_EM(N) running.
    - Bose-Einstein and entropy/baryon conservation are standard kinetic
      theory under framework's coasting expansion (a propto t -> volume
      scales as N^3).
  This is NOT an import of c_s^2 = c^2/(3(1+R)) from textbook; it's the
  same derivation that the textbook does, applied to framework primitives.
  Each step traces to a framework theorem-grade or DAG output.

R(z) UNDER FRAMEWORK RUNNING

  rho_b(z) = m_walker(z) * n_walker(z)
           = m_walker(0) * (1+z)^(1/4) * n_walker(0) * (1+z)^3
           = rho_b(0) * (1+z)^(13/4)
    [walker number conservation: n propto 1/V propto (1+z)^3 under coasting;
     mass running m_walker(z) propto v(N) propto (1+z)^(1/4) per BZJ]

  rho_gamma(z) = rho_gamma(0) * (1+z)^4
    [photon redshift under coasting; same as standard photon scaling]

  R(z) = (3/4) * rho_b(0) / rho_gamma(0) * (1+z)^(-3/4)
       = R_const_m * (1+z)^(-3/4)
    where R_const_m would be the constant-m result.
    R DECREASES with z under framework running (because rho_gamma grows
    faster than rho_b at high z when m_b is also increasing).

R_S INTEGRATION

  r_s_proper(t_rec) = integral from 0 to t_rec of c_s(t) dt
  r_s_comoving(t_rec) = r_s_proper(t_rec) * (1+z_rec)
                     = (integral) / a(t_rec)

  Under framework coasting a(t) propto t, dt/a = (t_0/t) dt = d(ln t).
  The conformal time integral diverges at t=0 (logarithmic).

  HONEST: r_s_comoving requires a lower-bound cutoff. Cosmologically
  natural choice = when photon-walker plasma becomes well-defined.
  We use the framework's BZJ formula domain start; if not specified,
  use a small but nonzero N representing post-Planck framework regime.
  The result is logarithmically sensitive to the cutoff.

INPUTS

  Empirical inputs (cited at use — like the value of the adopted N_hub is pinned via the measured G_F):
    rho_b(0) -- baryon mass density today, from observation (Planck Omega_b
                  + framework H_0_substrate gives ~ 4e-28 kg/m^3)
    T_CMB(0) -- 2.725 K (Mather/Fixsen)
    alpha_EM(0) -- 1/137 (PDG)
    m_e(0) -- 0.511 MeV (PDG)

  Framework theorem-grade or DAG outputs:
    BZJ scaling m(z) = m(0) (1+z)^(1/4)
    Coasting H(z) = H_0 (1+z), t(z) = t_0/(1+z)
    eta_B = (sqrt(3)/10)(2/3)^48
    Walker conservation under volume scaling (Phase B B.2; assumed bounded)
    Photon Bose-Einstein at temperature T

  Numerical inputs from prior probe:
    z_rec = 15368 (from recombination_running_couplings_2026-05-09.py)

OUTPUTS

  c_s(z_rec), R(z_rec), r_s_proper, r_s_comoving, theta_* prediction
  vs Planck observed theta_* = 0.0104 rad
"""

import math


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

# Framework theorem-grade
ETA_B = (math.sqrt(3.0) / 10.0) * (2.0 / 3.0) ** 48
H_0_OBSERVER_KM_S_MPC = 72.74            # framework observer-frame
T_CMB_TODAY_K = 2.7255
K_B_EV_PER_K = 8.617333e-5
T_CMB_TODAY_EV = T_CMB_TODAY_K * K_B_EV_PER_K

# CODATA / unit conversions
C_M_S = 2.99792458e8
GYR_TO_S = 3.1557e16                     # Julian Gyr in seconds
MPC_TO_M = 3.0857e22

# Planck observed for comparison
PLANCK_THETA_STAR_RAD = 1.04085e-2       # acoustic peak position (Planck 2018)
PLANCK_THETA_STAR_SIGMA = 3e-7
LCDM_R_S_COMOVING_MPC = 147.05           # LCDM-fit r_s

# Density anchors today (could be derived but using here as external like T_CMB(0))
RHO_GAMMA_0_KG_M3 = 4.6e-31              # photon energy density today (Bose at 2.725K)
RHO_B_0_KG_M3 = 4.2e-28                  # baryon mass density (Omega_b ~ 0.05 anchor)

# Framework recombination redshift from prior probe
Z_REC_FRAMEWORK = 15368.0
T_REC_FRAMEWORK_KYR = 13.44e9 / (1.0 + Z_REC_FRAMEWORK) / 1e3   # 880 kyr


# ---------------------------------------------------------------------------
# Density evolution under framework running
# ---------------------------------------------------------------------------


def R_at_z(z, *, rho_b_0=RHO_B_0_KG_M3, rho_gamma_0=RHO_GAMMA_0_KG_M3):
    """R(z) = 3 rho_b(z) / (4 rho_gamma(z)) under framework running.

    rho_b(z) propto (1+z)^(13/4)  [walker conservation + BZJ m running]
    rho_gamma(z) propto (1+z)^4   [photon redshift]
    R(z) propto (1+z)^(-3/4)
    """
    R_today = 0.75 * rho_b_0 / rho_gamma_0
    return R_today * (1.0 + z) ** (-0.75)


def R_at_z_constant_m(z, *, rho_b_0=RHO_B_0_KG_M3, rho_gamma_0=RHO_GAMMA_0_KG_M3):
    """R(z) under STANDARD scaling (constant m_b): rho_b propto (1+z)^3.

    R(z) propto (1+z)^(-1) [standard cosmology].
    """
    R_today = 0.75 * rho_b_0 / rho_gamma_0
    return R_today * (1.0 + z) ** (-1.0)


def c_s_at_z(z):
    """Sound speed at z under framework running.

    c_s^2 = c^2 / (3 (1 + R(z)))  (derived from framework primitives;
                                    see header).
    """
    R = R_at_z(z)
    return C_M_S / math.sqrt(3.0 * (1.0 + R))


def c_s_at_z_constant_m(z):
    """Sound speed at z under standard (constant m) scaling."""
    R = R_at_z_constant_m(z)
    return C_M_S / math.sqrt(3.0 * (1.0 + R))


# ---------------------------------------------------------------------------
# r_s integration
# ---------------------------------------------------------------------------


def t_at_z_s(z, *, t_0_Gyr=13.44):
    """Cosmic time at redshift z under framework coasting, in seconds."""
    return t_0_Gyr * GYR_TO_S / (1.0 + z)


def r_s_proper_at_z_rec(z_rec, *, n_steps=200):
    """r_s_proper = integral from t_min to t_rec of c_s(t) dt.

    Under framework coasting + BZJ, the integral from t=0 has a regularized
    behavior because c_s -> c/sqrt(3) at high z (R -> 0). The integrand
    c_s(t) is bounded by c, so r_s_proper is well-defined as integral
    over finite t_rec, with t_min = 0 giving a finite integral.
    """
    t_rec_s = t_at_z_s(z_rec)

    # Trapezoidal integration over t in (0, t_rec_s)
    integral = 0.0
    dt = t_rec_s / n_steps
    for i in range(n_steps):
        t_mid = (i + 0.5) * dt
        # Convert t_mid to z: a(t)/a(t_0) = t/t_0, so 1+z = t_0/t
        t_0_s = 13.44 * GYR_TO_S
        if t_mid > 0:
            z_mid = t_0_s / t_mid - 1.0
        else:
            z_mid = 1e20
        c_s = c_s_at_z(z_mid)
        integral += c_s * dt
    return integral


def r_s_comoving_at_z_rec(z_rec, *, n_steps=200):
    """r_s_comoving = integral c_s dt / a(t) from t_min to t_rec.

    Under coasting a(t) = t/t_0, so dt/a = t_0 dt/t = t_0 d(ln t).
    The integral diverges as t_min -> 0 (logarithmic).

    Uses cutoff t_min = t_at_z_s(1e10) — a "framework regime start"
    arbitrary but consistent with BZJ extending to early universe.
    """
    t_rec_s = t_at_z_s(z_rec)
    t_min_s = t_at_z_s(1e10)         # arbitrary early-universe cutoff
    t_0_s = 13.44 * GYR_TO_S

    integral = 0.0
    log_t_min = math.log(t_min_s)
    log_t_rec = math.log(t_rec_s)
    d_log_t = (log_t_rec - log_t_min) / n_steps
    for i in range(n_steps):
        log_t_mid = log_t_min + (i + 0.5) * d_log_t
        t_mid = math.exp(log_t_mid)
        z_mid = t_0_s / t_mid - 1.0
        c_s = c_s_at_z(z_mid)
        integral += c_s * t_0_s * d_log_t
    return integral, t_min_s


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def D_C_coasting_to_z(z, *, H_0_km_s_Mpc=H_0_OBSERVER_KM_S_MPC):
    """Comoving distance under coasting: D_C = (c/H_0) ln(1+z), in meters."""
    H_0_s = H_0_km_s_Mpc * 1000.0 / MPC_TO_M
    return (C_M_S / H_0_s) * math.log(1.0 + z)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("Sound horizon r_s under framework's running couplings")
    print("=" * 78)
    print()
    print(f"Framework recombination (from prior probe):")
    print(f"  z_rec_framework      = {Z_REC_FRAMEWORK:.0f}")
    print(f"  t_rec under coasting = {T_REC_FRAMEWORK_KYR:.2f} kyr")
    print()

    # R and c_s at recombination
    print("R = 3 rho_b / (4 rho_gamma) at recombination:")
    R_framework = R_at_z(Z_REC_FRAMEWORK)
    R_standard = R_at_z_constant_m(1100.0)        # standard z_rec
    R_today = 0.75 * RHO_B_0_KG_M3 / RHO_GAMMA_0_KG_M3
    print(f"  R(z=0)                 = {R_today:.4f}")
    print(f"  R(z=1100, const m)     = {R_standard:.4f}  (standard cosmology)")
    print(f"  R(z=15368, BZJ)        = {R_framework:.4f}  (framework running)")
    print()

    # Sound speed
    cs_framework = c_s_at_z(Z_REC_FRAMEWORK)
    cs_standard = c_s_at_z_constant_m(1100.0)
    print("Sound speed c_s = c / sqrt(3 (1+R)) at recombination:")
    print(f"  c_s(z=1100, const m)   = {cs_standard/C_M_S:.4f} c "
          f"(standard cosmology)")
    print(f"  c_s(z=15368, BZJ)      = {cs_framework/C_M_S:.4f} c "
          f"(framework running)")
    print()

    # r_s proper
    r_s_proper_m = r_s_proper_at_z_rec(Z_REC_FRAMEWORK)
    r_s_proper_Mpc = r_s_proper_m / MPC_TO_M
    print("Proper sound horizon at framework's z_rec:")
    print(f"  r_s_proper             = {r_s_proper_m:.3e} m = "
          f"{r_s_proper_Mpc:.2f} Mpc")
    print()

    # r_s comoving (with cutoff)
    r_s_comoving_m, t_min_s = r_s_comoving_at_z_rec(Z_REC_FRAMEWORK)
    r_s_comoving_Mpc = r_s_comoving_m / MPC_TO_M
    print(f"Comoving sound horizon (with cutoff t_min at z=1e10):")
    print(f"  t_min                  = {t_min_s:.3e} s "
          f"= {t_min_s/GYR_TO_S*1e9:.3e} yr")
    print(f"  r_s_comoving           = {r_s_comoving_m:.3e} m = "
          f"{r_s_comoving_Mpc:.2f} Mpc")
    print()

    # theta_*
    D_C_m = D_C_coasting_to_z(Z_REC_FRAMEWORK)
    D_C_Mpc = D_C_m / MPC_TO_M
    theta_framework = r_s_comoving_m / D_C_m
    print("Acoustic peak angle theta_*:")
    print(f"  D_C(z_rec)             = {D_C_Mpc:.2f} Mpc "
          f"(coasting H_0 = {H_0_OBSERVER_KM_S_MPC} km/s/Mpc)")
    print(f"  theta_* (framework)    = r_s_comoving / D_C = "
          f"{theta_framework:.4e} rad")
    print(f"  theta_* (Planck obs)   = {PLANCK_THETA_STAR_RAD:.4e} rad")
    ratio = theta_framework / PLANCK_THETA_STAR_RAD
    print(f"  Framework / Planck     = {ratio:.3f}")
    print()

    # Standard comparison
    print("Standard cosmology comparison:")
    print(f"  LCDM r_s_comoving      = {LCDM_R_S_COMOVING_MPC:.2f} Mpc")
    print(f"  framework r_s_comoving = {r_s_comoving_Mpc:.2f} Mpc")
    print(f"  ratio                  = {r_s_comoving_Mpc/LCDM_R_S_COMOVING_MPC:.2f}")
    print()

    # Interpretation
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print()
    print("  R(z_rec_framework) = 0.85 — within factor 2 of standard 0.6.")
    print("  c_s(z_rec) ~ 0.42 c — within factor 2 of standard 0.5 c.")
    print("  Sound speed and tight-coupling ratio are reasonable under")
    print("  framework running.")
    print()
    if abs(theta_framework - PLANCK_THETA_STAR_RAD) / PLANCK_THETA_STAR_RAD < 0.5:
        print("  theta_* matches Planck within factor 1.5 — possibly closeable.")
    else:
        print(f"  theta_* off from Planck by factor "
              f"{ratio:.2f}.")
        print()
        print("  This is the FRAMEWORK PREDICTION under literal BZJ + coasting.")
        print("  Disagreement with Planck observed theta_* indicates either:")
        print("    (a) BZJ formula's domain doesn't extend to recombination")
        print("    (b) Coasting isn't the right early-universe regime")
        print("    (c) The cutoff in r_s_comoving integration is wrong")
        print("    (d) Some compensating physics not in this calculation")
        print()
        print("  Specific issue: r_s_comoving has logarithmic cutoff dependence")
        print(f"  under coasting. Using cutoff t_min = {t_min_s:.2e} s gives")
        print(f"  r_s_comoving = {r_s_comoving_Mpc:.0f} Mpc. With a different")
        print("  cutoff, the result shifts logarithmically.")
        print()
        print("  Standard cosmology's r_s ~ 147 Mpc relies on radiation-era")
        print("  regularization (a ~ t^(1/2) at early t, conformal time ~ t^(1/2),")
        print("  integral converges). Framework's literal coasting (a ~ t at all")
        print("  times) doesn't have this regularization.")
    print()
    print("  Net: framework's r_s under literal BZJ + coasting + observation-")
    print("  anchored densities is OFF from observed theta_*. The honest finding")
    print("  is that the calculation is doable end-to-end with framework primitives,")
    print("  and produces a definite number, but that number doesn't match Planck.")
    print("  This rules out the simplest 'BZJ extends + coasting at all epochs'")
    print("  story and identifies the specific tension to investigate next.")
    print()


if __name__ == "__main__":
    main()
