#!/usr/bin/env python3
"""
bbn_network.py — BBN reaction-network harness with pluggable H(T) and η.

WHAT THIS IS
------------
A draft Big-Bang-nucleosynthesis harness whose cosmological inputs — the
expansion rate H(T) and the baryon-to-photon ratio η — are PLUGGABLE, so the
framework's substrate cosmology can be dropped in next to standard ΛCDM and
compared on the same nuclear physics.

It exists to answer "what do we need to run the full BBN reaction network?":
the nuclear/weak physics here is STANDARD and external; the framework-specific
content is entirely in the ExpansionModel (H normalization — the √g_* leading
factor, an internal note) and in η (whose
constancy across the window is the open question scoped in
an internal working note).

WHAT IS VALIDATED vs WHAT IS STUBBED
------------------------------------
  • VALIDATED: the weak n↔p sector → Y_p. Born-approximation rates (6 weak
    processes) normalized to the neutron lifetime, integrated with the chosen
    H(T). With ΛCDM inputs this reproduces Y_p ≈ 0.247 (see demo). This is the
    load-bearing piece for the framework's Y_p question because Y_p is set by
    the n/p ratio at freeze-out (∝ H) and the decay time to the bottleneck
    (∝ 1/H), both of which the ExpansionModel controls.

  • STUBBED / SCAFFOLD: the light-element network (D, ³He, ⁷Li). The reaction
    list and ⟨σv⟩(T) interface are present and clearly marked EXTERNAL nuclear
    physics, but the abundances are NOT precision-validated here. Wiring a full
    rate library (PArthENoPE / Kawano-style ~88 reactions) is the remaining
    engineering — it does not depend on the framework and is fenced off.

EXTERNAL INPUTS (out-of-scope-by-construction, like B_D/m_nucleon already are
in predictions/T_BBN_D_bottleneck.py): Q_np, m_e, τ_n, nuclear binding
energies, and all ⟨σv⟩ reaction-rate fits. These are measured nuclear physics,
not framework-derivable (Need-B / Clause-9 territory).

UNITS: temperatures in MeV; H in s⁻¹; time in s. Energies inside the weak-rate
integrals are in units of m_e (electron mass), which makes the neutron-lifetime
normalization exact and dimensionless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d

# ===========================================================================
# External constants (measured nuclear/particle physics — NOT framework)
# ===========================================================================
Q_NP_MeV = 1.293332      # m_n - m_p (PDG)
M_E_MeV = 0.510999       # electron mass (PDG)
TAU_N_s = 879.4          # neutron lifetime (PDG 2022)
B_D_MeV = 2.224566       # deuteron binding energy
M_NUCLEON_MeV = 938.918  # average nucleon mass
M_PL_GeV = 1.220890e19   # NON-reduced Planck mass
HBAR_GeV_s = 6.582119e-25  # ħ in GeV·s (natural→SI for H)

Q = Q_NP_MeV / M_E_MeV   # dimensionless n-p splitting ≈ 2.531
ZETA3 = 1.2020569

# Framework substrate primitive (theorem-grade upstream: predictions/k_star.py)
K_STAR = 3


# ===========================================================================
# Thermodynamics: g_*(T) and the neutrino temperature
# ===========================================================================
def _e_plus_minus_g(T_MeV: float) -> float:
    """Energy-density g-factor of one e± species pair via the FD integral.

    Returns the relativistic-equivalent g for e⁺+e⁻ (→ 7/2 when T≫m_e,
    →0 when T≪m_e). Standard fermion energy integral in units of m_e.
    """
    x = M_E_MeV / T_MeV
    if x > 50.0:
        return 0.0

    def integrand(u):  # u = E/T, energy E≥m_e
        E = u
        if E * E <= x * x:
            return 0.0
        p = math.sqrt(E * E - x * x)
        return E * E * p / (math.exp(E) + 1.0)

    val, _ = quad(integrand, x, x + 60.0, limit=100)
    # one fermion dof normalization: (g/2)·(7/8)·(π²/30) baseline → ratio form.
    # Relativistic single-fermion energy integral = (7/8)·(π⁴/15). e± has 4 dof.
    rel = (7.0 / 8.0) * (math.pi ** 4 / 15.0)
    return 4.0 * val / rel * (7.0 / 8.0)


def T_nu_over_T_gamma(T_MeV: float) -> float:
    """T_ν/T_γ. =1 before e± annihilation, →(4/11)^(1/3) after.

    Modeled by entropy dump: as e± annihilate they heat γ but not ν.
    Smooth interpolation in log-T across the annihilation (m_e scale).
    """
    asymptote = (4.0 / 11.0) ** (1.0 / 3.0)
    # blend: high T → 1, low T → asymptote, transition around T ~ m_e/3
    w = 0.5 * (1.0 + math.tanh((math.log(T_MeV) - math.log(M_E_MeV / 2.5)) / 0.6))
    return asymptote + (1.0 - asymptote) * w


def g_star_energy(T_MeV: float) -> float:
    """Energy-density g_*(T) over the BBN window (photons + e± + 3ν)."""
    g_gamma = 2.0
    g_e = _e_plus_minus_g(T_MeV)
    r = T_nu_over_T_gamma(T_MeV)
    g_nu = 3.0 * 2.0 * (7.0 / 8.0) * r ** 4   # 3 species, 2 dof, fermion 7/8
    return g_gamma + g_e + g_nu


# ===========================================================================
# Expansion models — THE framework-specific surface (pluggable)
# ===========================================================================
@dataclass
class ExpansionModel:
    """H(T) and the time-temperature relation. Plug ΛCDM or framework here.

    H_of_T : T[MeV] -> H[s⁻¹]
    name   : label for reporting
    """
    name: str
    H_of_T: Callable[[float], float]

    def dt_dT(self, T_MeV: float) -> float:
        """dt/dT [s/MeV] under adiabatic cooling T ∝ 1/a ⇒ dT/dt = -H·T.

        NOTE: adiabaticity is reading A/B of the η scoping doc (the default
        the first probe recommends). A genuinely non-adiabatic framework
        bath-cooling law would override this method.
        """
        return -1.0 / (self.H_of_T(T_MeV) * T_MeV)


def _H_natural_to_si(H_GeV: float) -> float:
    return H_GeV / HBAR_GeV_s


def lcdm_expansion() -> ExpansionModel:
    """Standard radiation-era Friedmann: H = 1.66·√g_*·T²/M_Pl."""
    def H(T_MeV: float) -> float:
        T_GeV = T_MeV * 1e-3
        g = g_star_energy(T_MeV)
        H_GeV = math.sqrt(8.0 * math.pi ** 3 * g / 90.0) * T_GeV ** 2 / M_PL_GeV
        return _H_natural_to_si(H_GeV)
    return ExpansionModel("ΛCDM (1.66·√g_*·T²/M_Pl)", H)


def framework_expansion(leading_factor: str = "candidate") -> ExpansionModel:
    """Framework substrate H = F·T²/M_Pl with F the leading-factor candidate.

    leading_factor:
      "bare"      F = 1            — N_hub cascade theorem prefactor (no √g_*).
                                     This is the literal framework H; gives the
                                     Y_p ≈ 0.05 falsification candidate.
      "candidate" F = √(k*·g_*)    — the leading-factor chase candidate
,
                                     +4.3% vs ΛCDM's √(8π³/90)·√g_*.
    """
    def F_of_T(T_MeV: float) -> float:
        if leading_factor == "bare":
            return 1.0
        if leading_factor == "candidate":
            return math.sqrt(K_STAR * g_star_energy(T_MeV))
        raise ValueError(f"unknown leading_factor {leading_factor!r}")

    def H(T_MeV: float) -> float:
        T_GeV = T_MeV * 1e-3
        H_GeV = F_of_T(T_MeV) * T_GeV ** 2 / M_PL_GeV
        return _H_natural_to_si(H_GeV)

    return ExpansionModel(f"framework substrate (F={leading_factor})", H)


# ===========================================================================
# Weak n↔p rates — Born approximation, 6 processes, normalized to τ_n
# ===========================================================================
def _fd(E: float, tau: float) -> float:
    """Fermi-Dirac occupation at energy E (in m_e units), temperature tau."""
    if tau <= 0.0:
        return 0.0
    arg = E / tau
    if arg > 60.0:
        return 0.0
    return 1.0 / (math.exp(arg) + 1.0)


_I0 = quad(lambda E: math.sqrt(E * E - 1.0) * E * (Q - E) ** 2, 1.0, Q)[0]
# _I0 is the T=0 decay phase space; it fixes the overall normalization so the
# decay process alone gives exactly 1/τ_n at T→0.


def weak_rates(T_gamma_MeV: float) -> tuple[float, float]:
    """Return (λ_{n→p}, λ_{p→n}) in s⁻¹ at photon temperature T_gamma.

    Six weak processes (energies in m_e units, q = Q_np/m_e):
      n→p:  n+ν→p+e⁻ | n+e⁺→p+ν̄ | n→p+e⁻ν̄ (decay)
      p→n:  p+e⁻→n+ν | p+ν̄→n+e⁺ | (inverse decay, negligible — omitted)
    Each integrand is p_e·E_e·E_ν²·(occupations), per the standard Born form.
    """
    tau_g = T_gamma_MeV / M_E_MeV
    tau_n = T_nu_over_T_gamma(T_gamma_MeV) * tau_g
    EMAX = Q + 60.0 * max(tau_g, tau_n) + 30.0

    def p_of(E):
        return math.sqrt(max(E * E - 1.0, 0.0))

    # --- n → p ---
    # (a) n + ν_e → p + e⁻ : integrate over ν energy E_ν≥0, E_e = E_ν + q
    def a_int(Enu):
        Ee = Enu + Q
        return Enu * Enu * Ee * p_of(Ee) * _fd(Enu, tau_n) * (1.0 - _fd(Ee, tau_g))

    # (b) n + e⁺ → p + ν̄ : integrate over e⁺ energy E_e≥1, E_ν = E_e + q
    def b_int(Ee):
        Enu = Ee + Q
        return p_of(Ee) * Ee * Enu * Enu * _fd(Ee, tau_g) * (1.0 - _fd(Enu, tau_n))

    # (c) n → p + e⁻ + ν̄ (decay): E_e∈[1,q], E_ν = q - E_e
    def c_int(Ee):
        Enu = Q - Ee
        return p_of(Ee) * Ee * Enu * Enu * (1.0 - _fd(Ee, tau_g)) * (1.0 - _fd(Enu, tau_n))

    Ja = quad(a_int, 0.0, EMAX, limit=120)[0]
    Jb = quad(b_int, 1.0, EMAX, limit=120)[0]
    Jc = quad(c_int, 1.0, Q, limit=80)[0]
    lam_np = (Ja + Jb + Jc) / (TAU_N_s * _I0)

    # --- p → n ---
    # (d) p + e⁻ → n + ν : E_e≥q, E_ν = E_e - q
    def d_int(Ee):
        Enu = Ee - Q
        return p_of(Ee) * Ee * Enu * Enu * _fd(Ee, tau_g) * (1.0 - _fd(Enu, tau_n))

    # (e) p + ν̄ → n + e⁺ : E_ν ≥ q+1, E_e = E_ν - q
    def e_int(Enu):
        Ee = Enu - Q
        return Enu * Enu * Ee * p_of(Ee) * _fd(Enu, tau_n) * (1.0 - _fd(Ee, tau_g))

    Jd = quad(d_int, Q, EMAX, limit=120)[0]
    Je = quad(e_int, Q + 1.0, EMAX, limit=120)[0]
    lam_pn = (Jd + Je) / (TAU_N_s * _I0)

    return lam_np, lam_pn


# ===========================================================================
# Deuterium bottleneck T_D(η) — where free neutrons get locked into ⁴He
# ===========================================================================
def deuterium_bottleneck_T_MeV(eta: float, n_iter: int = 40) -> float:
    """T_D where deuterium photo-dissociation stops winning (η-dependent).

    Same Saha-bottleneck structure as predictions/T_BBN_D_bottleneck.py:
    solve T = B_D / ln[(m_N T/2π)^{3/2} / (η·n_γ)] self-consistently.
    """
    T = B_D_MeV / 30.0
    for _ in range(n_iter):
        prefac = (M_NUCLEON_MeV * T / (2.0 * math.pi)) ** 1.5
        n_gamma = (2.0 * ZETA3 / math.pi ** 2) * T ** 3
        N_thermal = math.log(prefac / (eta * n_gamma))
        T = B_D_MeV / N_thermal
    return T


# ===========================================================================
# The weak-sector solve → Y_p (VALIDATED core)
# ===========================================================================
@dataclass
class BBNResult:
    Y_p: float
    X_n_freeze: float        # neutron fraction at the bottleneck (pre-decay-corrected by integration)
    T_bottleneck_MeV: float
    expansion: str
    eta: float


def run_weak_sector(
    expansion: ExpansionModel,
    eta: float,
    T_start_MeV: float = 10.0,
    n_grid: int = 400,
) -> BBNResult:
    """Integrate X_n through freeze-out + decay to the D bottleneck → Y_p.

    Y_p = 2·X_n(T_D): essentially all surviving neutrons end up in ⁴He once
    deuterium clears its bottleneck. Couples to H(T) twice — the freeze-out
    temperature (Γ_weak = H) and the decay time to T_D (∝ 1/H) — which is why
    Y_p is the sensitive probe of the framework's H normalization.
    """
    T_D = deuterium_bottleneck_T_MeV(eta)

    # tabulate weak rates on a log-T grid then interpolate (fast ODE RHS)
    T_grid = np.logspace(math.log10(T_start_MeV), math.log10(T_D * 0.95), n_grid)
    lam_np_tab = np.empty_like(T_grid)
    lam_pn_tab = np.empty_like(T_grid)
    for i, T in enumerate(T_grid):
        lam_np_tab[i], lam_pn_tab[i] = weak_rates(float(T))

    log_T = np.log(T_grid[::-1])
    f_np = interp1d(log_T, lam_np_tab[::-1], kind="cubic", fill_value="extrapolate")
    f_pn = interp1d(log_T, lam_pn_tab[::-1], kind="cubic", fill_value="extrapolate")

    # independent variable: T (decreasing). dX_n/dT = (dX_n/dt)·(dt/dT).
    def rhs(T, y):
        Xn = y[0]
        lam_np = float(f_np(math.log(T)))
        lam_pn = float(f_pn(math.log(T)))
        dXn_dt = lam_pn * (1.0 - Xn) - lam_np * Xn
        return [dXn_dt * expansion.dt_dT(T)]

    Xn0 = 1.0 / (1.0 + math.exp(Q_NP_MeV / T_start_MeV))  # equilibrium start
    sol = solve_ivp(
        rhs, (T_start_MeV, T_D), [Xn0],
        method="Radau", rtol=1e-8, atol=1e-12, dense_output=True,
    )
    Xn_final = float(sol.y[0, -1])
    return BBNResult(
        Y_p=2.0 * Xn_final,
        X_n_freeze=Xn_final,
        T_bottleneck_MeV=T_D,
        expansion=expansion.name,
        eta=eta,
    )


# ===========================================================================
# Light-element network SCAFFOLD (STUBBED — external rates, not validated)
# ===========================================================================
# The full network evolves Y_i for {n, p, D, T, ³He, ⁴He, ⁷Li, ⁷Be}. Below is
# the reaction LIST and the external-rate INTERFACE; the ⟨σv⟩ fits are where a
# real library (Kawano/PArthENoPE/AlterBBN) plugs in. NOT precision-validated.
KEY_REACTIONS = [
    # (reactants, products) — the standard minimal set that sets D/³He/⁷Li.
    ("p + n -> D + γ",        "deuteron formation; gates the whole network"),
    ("D + p -> ³He + γ",      "deuterium burning"),
    ("D + D -> ³He + n",      ""),
    ("D + D -> T + p",        ""),
    ("T + D -> ⁴He + n",      ""),
    ("³He + D -> ⁴He + p",    ""),
    ("³He + n -> T + p",      ""),
    ("³He + ⁴He -> ⁷Be + γ",  "⁷Be / ⁷Li production"),
    ("T + ⁴He -> ⁷Li + γ",    ""),
    ("⁷Be + n -> ⁷Li + p",    ""),
    ("⁷Li + p -> ⁴He + ⁴He",  "⁷Li destruction"),
]


def sigma_v_STUB(reaction: str, T9: float) -> float:
    """⟨σv⟩(T) interface for the light-element network — EXTERNAL nuclear data.

    A real run wires literature rate fits here (T9 = T/10⁹K). Left as an
    explicit stub so the harness's framework-specific surface (H, η) stays
    cleanly separated from the out-of-scope nuclear rate library.
    """
    raise NotImplementedError(
        "light-element ⟨σv⟩ fits are external nuclear physics; wire a rate "
        "library (Kawano/PArthENoPE) here. Y_p (weak sector) does not need it."
    )
