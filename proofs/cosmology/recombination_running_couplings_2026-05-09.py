"""
Recombination redshift under framework's running couplings.

Concrete probe: where does the framework predict recombination
(Saha x_e = 0.5) to occur, given that the framework's m_e runs with N
per the BZJ formula (m_e propto v propto N^(-1/4))?

Standard cosmology: m_e is constant; Saha gives recombination at
z_recomb ~ 1370 (x_e = 0.5 criterion).

Framework: m_e(z) = m_e_today * (1+z)^(1/4). Rydberg(z) scales as m_e,
so binding energy is LARGER at higher z. Photon temperature T(z) =
T_CMB(0) * (1+z) scales linearly. Recombination requires T ~ Rydberg /
log(eta_B^-1), which gives a self-consistent equation in z.

This is a NEW concrete framework prediction that has not been computed
before. The result tells us:
  (a) at what redshift framework's recombination occurs;
  (b) at what cosmic TIME (under coasting) this corresponds to;
  (c) whether the framework's BZJ-formula scaling is consistent with
      standard recombination physics or implies a regime crossover.

Inputs (anchors / theorem-grade):
  - m_e_today           = 0.511 MeV (PDG; or via DAG from m_tau which
                                       runs with v - same N^(-1/4) scaling)
  - alpha_EM_today      = 1/137.036 (PDG; weak log running ignored at
                                       leading order)
  - T_CMB_today         = 2.725 K (Mather 1999; an empirical input — like the value of the adopted N_hub (pinned via the measured G_F)
                                    until framework derives T_CMB(0))
  - eta_B               = (sqrt(3)/10)(2/3)^48 = 6.11e-10 (theorem-grade)
  - BZJ scaling         = m_e(z) propto (1+z)^(1/4) (DAG output;
                                                      demonstrated 2026-05-09)
  - Coasting trajectory = N(z) = N_hub_now / (1+z) (theorem-grade D3)

NO new mechanisms. NO new modules. Pure composition of existing
framework primitives.
"""

import math

# ---------------------------------------------------------------------------
# Constants and anchors (cited at use)
# ---------------------------------------------------------------------------

# Framework theorem-grade
ETA_B = (math.sqrt(3.0) / 10.0) * (2.0 / 3.0) ** 48     # ~6.11e-10

# Empirical inputs (cited; comparable in role to the value of the adopted N_hub being pinned via the measured G_F)
M_E_TODAY_EV = 0.510998950e6                            # PDG 2024
ALPHA_EM_TODAY = 1.0 / 137.035999084                    # PDG (assume weak running)
T_CMB_TODAY_K = 2.7255                                  # Mather/Fixsen (COBE/FIRAS)
K_B_EV_PER_K = 8.617333e-5                              # CODATA Boltzmann
T_CMB_TODAY_EV = T_CMB_TODAY_K * K_B_EV_PER_K           # ~2.349e-4 eV

# CODATA/derived for n_b conversion
M_PROTON_EV = 938.272e6                                 # PDG
HBAR_C_EV_M = 1.97327e-7                                # eV.m (hbar*c)


# ---------------------------------------------------------------------------
# Framework running per BZJ (m_e propto v propto N^(-1/4); v_higgs.py)
# ---------------------------------------------------------------------------


def m_e_at_z_eV(z, *, m_e_today_eV=M_E_TODAY_EV):
    """m_e(z) under BZJ scaling. v_Higgs scales as N^(-1/4), masses propto v.

    With N(z) = N_hub_now / (1+z), m_e(z) = m_e_today * (1+z)^(1/4).
    """
    return m_e_today_eV * (1.0 + z) ** 0.25


def Rydberg_at_z_eV(z):
    """Rydberg(z) = (1/2) m_e(z) alpha_EM^2 in eV.

    alpha_EM running ignored (logarithmic; small effect compared to m_e
    quartic root scaling here).
    """
    return 0.5 * m_e_at_z_eV(z) * ALPHA_EM_TODAY ** 2


def T_CMB_at_z_eV(z):
    """T_CMB(z) = T_CMB(0) * (1+z) from photon energy redshift under coasting.

    Same as standard cosmology — photons redshift independently of BZJ
    running.
    """
    return T_CMB_TODAY_EV * (1.0 + z)


# ---------------------------------------------------------------------------
# Saha equation
# ---------------------------------------------------------------------------


def n_baryon_at_z_eV3(z, *, eta_B=ETA_B, T_today_eV=T_CMB_TODAY_EV):
    """Baryon number density in eV^3 (natural units, hbar=c=1).

    n_gamma(z) = (2 zeta(3) / pi^2) T(z)^3 = 0.2436 T(z)^3 (Bose-Einstein).
    n_b(z) = eta_B * n_gamma(z).
    """
    T_z = T_CMB_at_z_eV(z)
    n_gamma_natural = (2.0 * 1.20206 / math.pi ** 2) * T_z ** 3
    return eta_B * n_gamma_natural


def saha_x_e(z):
    """Solve Saha equation for ionization fraction x_e at redshift z.

    Saha:  x_e^2 / (1 - x_e) = (1/n_b) * (m_e T / 2pi)^(3/2) * exp(-Rydberg/T)
    """
    T_z = T_CMB_at_z_eV(z)
    m_e_z = m_e_at_z_eV(z)
    Ry_z = Rydberg_at_z_eV(z)
    n_b = n_baryon_at_z_eV3(z)

    pre = (m_e_z * T_z / (2.0 * math.pi)) ** 1.5
    boltz = math.exp(-Ry_z / T_z)
    rhs = pre * boltz / n_b      # = x_e^2 / (1 - x_e)

    # Solve quadratic: x_e^2 + rhs * x_e - rhs = 0
    # x_e = (-rhs + sqrt(rhs^2 + 4 rhs)) / 2
    if rhs > 1e30:
        return 1.0
    if rhs < 1e-30:
        return math.sqrt(rhs)
    disc = rhs * rhs + 4.0 * rhs
    return 0.5 * (-rhs + math.sqrt(disc))


def find_recombination_z(target_x_e=0.5, *, z_low=10.0, z_high=1.0e6,
                          tol=1.0e-3, max_iter=200):
    """Bisect for z where saha_x_e(z) = target_x_e.

    saha_x_e is monotone decreasing in z (since T_z grows with z, faster
    than Rydberg under any quartic-or-slower running).

    Wait — actually under BZJ, Rydberg ~ (1+z)^(1/4), T ~ (1+z), so
    T/Rydberg ~ (1+z)^(3/4) grows with z; exp(-Ry/T) -> 1 at high z;
    n_b ~ T^3 grows; the prefactor (m_e T/2pi)^(3/2) grows.

    saha_x_e is high (close to 1) at high z (everything ionized) and low
    at low z (everything recombined). We want the bisection in the right
    direction.
    """
    f_lo = saha_x_e(z_low) - target_x_e
    f_hi = saha_x_e(z_high) - target_x_e

    # Want target_x_e between low (smaller x_e) and high (larger x_e).
    # x_e is monotone INCREASING with z (more ionized at higher z).
    if f_lo > 0:
        return z_low, "Already at target_x_e or above at z_low"
    if f_hi < 0:
        return z_high, "Below target_x_e even at z_high"

    a, b = z_low, z_high
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        f_mid = saha_x_e(mid) - target_x_e
        if abs(f_mid) < tol:
            return mid, "converged"
        if f_mid > 0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b), "max_iter"


# ---------------------------------------------------------------------------
# Standard-cosmology comparison (no running)
# ---------------------------------------------------------------------------


def saha_x_e_no_running(z):
    """Same as saha_x_e but with constant m_e and Rydberg (standard cosmology)."""
    T_z = T_CMB_at_z_eV(z)
    Ry = 0.5 * M_E_TODAY_EV * ALPHA_EM_TODAY ** 2
    n_gamma_natural = (2.0 * 1.20206 / math.pi ** 2) * T_z ** 3
    n_b = ETA_B * n_gamma_natural
    pre = (M_E_TODAY_EV * T_z / (2.0 * math.pi)) ** 1.5
    boltz = math.exp(-Ry / T_z)
    rhs = pre * boltz / n_b
    if rhs > 1e30:
        return 1.0
    if rhs < 1e-30:
        return math.sqrt(rhs)
    disc = rhs * rhs + 4.0 * rhs
    return 0.5 * (-rhs + math.sqrt(disc))


def find_recombination_z_standard(target_x_e=0.5, *, z_low=10.0,
                                   z_high=1.0e6, tol=1.0e-3, max_iter=200):
    a, b = z_low, z_high
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        f_mid = saha_x_e_no_running(mid) - target_x_e
        if abs(f_mid) < tol:
            return mid, "converged"
        if f_mid > 0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b), "max_iter"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("Recombination redshift under framework's running couplings")
    print("=" * 78)
    print()

    # Anchors
    print("Anchors:")
    print(f"  m_e_today                = {M_E_TODAY_EV:.4e} eV  (PDG)")
    print(f"  alpha_EM_today           = {ALPHA_EM_TODAY:.6f}    (PDG)")
    print(f"  T_CMB_today              = {T_CMB_TODAY_K:.4f} K = "
          f"{T_CMB_TODAY_EV:.4e} eV  (Mather/Fixsen)")
    print(f"  eta_B                    = {ETA_B:.4e}    (theorem-grade,"
          f" Sakharov chain)")
    print()
    print("Framework running:")
    print(f"  m_e(z) = m_e_today * (1+z)^(1/4)        (BZJ scaling)")
    print(f"  Rydberg(z) = (1/2) m_e(z) alpha_EM^2    (alpha_EM running ignored)")
    print(f"  T(z) = T_CMB_today * (1+z)              (photon redshift)")
    print()

    # Sample x_e values at a few z to confirm direction
    print("x_e(z) sample under framework running:")
    print(f"  {'z':>10} {'T(z)/eV':>12} {'m_e(z)/MeV':>12} "
          f"{'Ry(z)/eV':>10} {'x_e':>14}")
    for z in (100, 300, 1000, 1100, 3000, 10000, 30000, 100000):
        T = T_CMB_at_z_eV(z)
        me = m_e_at_z_eV(z) / 1e6
        ry = Rydberg_at_z_eV(z)
        xe = saha_x_e(z)
        print(f"  {z:>10} {T:>12.4e} {me:>12.4f} {ry:>10.4f} {xe:>14.4e}")
    print()

    # Find recombination z
    z_rec_framework, status_f = find_recombination_z(target_x_e=0.5)
    z_rec_standard, status_s = find_recombination_z_standard(target_x_e=0.5)
    print("Recombination (Saha x_e = 0.5):")
    print(f"  Standard cosmology       : z_rec = {z_rec_standard:.0f}  "
          f"({status_s})")
    print(f"  Framework (BZJ running)  : z_rec = {z_rec_framework:.0f}  "
          f"({status_f})")
    print()

    # Cosmic time at framework's z_rec under coasting
    # Coasting: t(z) = t_0 / (1+z); t_0_observer ~ 13.44 Gyr
    t_0_observer_Gyr = 13.44
    t_rec_framework_Gyr = t_0_observer_Gyr / (1.0 + z_rec_framework)
    t_rec_standard_Gyr = t_0_observer_Gyr / (1.0 + z_rec_standard)
    t_rec_framework_kyr = t_rec_framework_Gyr * 1e6
    t_rec_standard_kyr = t_rec_standard_Gyr * 1e6
    print(f"Cosmic time at recombination (under framework coasting "
          f"with t_0_observer = {t_0_observer_Gyr} Gyr):")
    print(f"  Standard cosmology       : t_rec = {t_rec_standard_kyr:.2f} kyr")
    print(f"  Framework (BZJ running)  : t_rec = {t_rec_framework_kyr:.2f} kyr")
    print(f"  Standard cosmology has recombination at ~ 380 kyr")
    print()

    # Interpretation
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print()
    print(f"Framework predicts recombination (Saha x_e = 0.5) at z = "
          f"{z_rec_framework:.0f}")
    print(f"under BZJ running of m_e. Compare to standard {z_rec_standard:.0f}.")
    print()
    if z_rec_framework / z_rec_standard > 5:
        print(f"This is FACTOR {z_rec_framework/z_rec_standard:.0f}x DIFFERENT "
              f"from standard recombination redshift.")
        print("Indicates either (a) BZJ formula extends and framework predicts")
        print("recombination at much higher z than standard, or (b) BZJ formula")
        print("has a regime of applicability that does not reach recombination,")
        print("with some other physics taking over at intermediate N.")
        print()
        print("Cosmic-time perspective:")
        print(f"  Framework t_rec = {t_rec_framework_kyr:.0f} kyr; "
              f"standard ~380 kyr.")
        if abs(t_rec_framework_kyr - 380) / 380 < 1.0:
            print("  Framework's cosmic TIME for recombination is "
                  "order-of-magnitude consistent")
            print("  with standard ~380 kyr. The redshift differs because")
            print("  framework's coasting maps t<->z differently from LCDM.")
        else:
            print(f"  Framework's t_rec also differs significantly from "
                  f"standard ~380 kyr.")
    else:
        print(f"Framework prediction within order-of-magnitude of standard.")
    print()
    print("This is a NEW concrete framework prediction. Not previously")
    print("computed in the predictions/ DAG.")
    print()
    print("Implication for r_s closure: the framework's r_s would be")
    print("integrated to framework's z_rec, not to standard z=1100. That")
    print("changes both the upper limit and the integrand (running couplings)")
    print("and is what the substrate r_s coupled-mode derivation would")
    print("compose with.")


if __name__ == "__main__":
    main()
