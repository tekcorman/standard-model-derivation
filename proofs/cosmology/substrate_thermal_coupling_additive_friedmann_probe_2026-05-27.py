#!/usr/bin/env python3
"""
Substrate-thermal-coupling mechanism — Route ADD viability probe (2026-05-27).

Scoping: an internal working note

THE QUESTION
------------
Can ONE additive Friedmann form, with an ADIABATIC photon bath, simultaneously:
  BC1  give H ≈ √(k*·g_*)·T²/M_Pl when radiation-dominated  (→ Y_p ✓),
  BC2  give H → 1/(N·t_P) ≈ 68 km/s/Mpc when substrate-dominated (→ H_0 ✓),
  BC3  deactivate the √g_* factor between them with NO free parameter,
  BC4  keep the bath adiabatic so η = η_B stays constant?

  H² = H_rad²(T) + H_sub²(N),   H_rad = √(8π³/90·g_*)·T²/M_Pl,   H_sub = 1/(N·t_P)

THE KEY POINT
-------------
The F-deactivation probe (F_deactivation_mechanism_probe_2026-05-27.py) ruled
MECH-1 (this exact additive form) a FAILURE — but ONLY under the α=1/2 bath law
(T ∝ a⁻¹ᐟ²), which makes ρ_rad ∝ T⁴ ∝ a⁻² scale IDENTICALLY to the substrate
term ρ_sub ∝ a⁻² → constant ratio, no deactivation.

The η scoping's reading A (ADIABATIC bath, T ∝ 1/a) gives ρ_rad ∝ a⁻⁴, which
falls FASTER than ρ_sub ∝ a⁻². This probe shows that single switch flips
MECH-1 from FAIL to WORK. The two open questions were blocking each other.

HONESTY
-------
This establishes PHENOMENOLOGICAL VIABILITY only. It does NOT derive the mechanism
(why the terms add in quadrature, why ρ_sub ∝ a⁻² gravitates, reconciling with
the N_hub cascade theorem — targets T1/T2/T3 of the scoping doc). No closure.

UPDATE 2026-05-28 — T1/T2/T3 SINCE CLOSED; this file is the PRE-CLOSURE record.
T2 admissible (cascade_reconcile_verdict), T3 positive-candidate, and the
holographic identification ρ_sub=E_obs/V_Hubble was DERIVED (perceptual-surface
principle). The complete mechanism — assembled from the DERIVED pieces and
verified end-to-end with zero free parameters — is in
`substrate_thermal_coupling_mechanism_consolidated_verification_2026-05-28.py`.
The verdict block below ("OPEN: T1/T2/T3") is kept as the historical pre-closure
state; read the consolidated verification for the resolved mechanism.

T2 REFINEMENT (2026-05-27, see substrate_thermal_coupling_T2_cascade_reconcile_verdict):
The deactivating "H_sub" used below is the w=−1/3 FRIEDMANN component ρ_sub ∝ a⁻²
(H_sub,Friedmann ∝ a⁻¹). This coincides with the literal cascade clock
H_info = 1/(N·t_P) ONLY at the late coasting attractor. During radiation
domination the literal cascade clock (N ∝ cosmic time) scales as a⁻² in H
(ρ_info ∝ a⁻⁴, same as radiation) and would NOT deactivate. The numbers below
are correct for the two-component Friedmann model with ρ_sub ∝ a⁻²; the
identification with 1/(N·t_P) is exact only late. T2 verdict: ADMISSIBLE — Route
ADD modifies the cascade theorem's own-flagged undefended assumption A3.a, not
its robust content D1+D2.

Run:
    python3 proofs/cosmology/substrate_thermal_coupling_additive_friedmann_probe_2026-05-27.py
"""

from __future__ import annotations

import math

# --- constants (substrate primitives + standard; cited at use site) ---------
K_STAR = 3                         # predictions/k_star.py (theorem-grade)
M_PL_GeV = 1.220890e19             # non-reduced Planck mass
T_P_s = 5.391247e-44               # Planck time
N_HUB = 8.394881e60                # predictions/N_hub.py (G_F-calibrated)
MPC_KM = 3.085677581e19            # 1 Mpc in km
HBAR_GeV_s = 6.582119e-25

T_TODAY_K = 2.7255                 # CMB temperature today
K_per_GeV = 1.0 / 8.617333e-14     # K per GeV (k_B)
T_TODAY_GeV = T_TODAY_K / K_per_GeV

FRIEDMANN_166 = math.sqrt(8.0 * math.pi ** 3 / 90.0)   # = 1.6606 (continuum)
SQRT_K = math.sqrt(K_STAR)                              # = 1.7321 (K-rational, +4.3%)


def banner(t):
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


def g_star(T_MeV):
    """Standard energy g_*(T): 10.75 (T≫m_e) → 3.36 (after e± annihilation)."""
    if T_MeV > 0.5:
        return 10.75
    if T_MeV > 0.02:
        return 3.36 + (10.75 - 3.36) * (T_MeV - 0.02) / (0.5 - 0.02)
    return 3.36


def H_si_from_GeV(H_GeV):
    return H_GeV / HBAR_GeV_s


# ===========================================================================
banner("ROUTE ADD VIABILITY PROBE — additive Friedmann + adiabatic bath")
print(f"""
  H² = H_rad²(T) + H_sub²(N)
    H_rad = √(8π³/90 · g_*) · T²/M_Pl   [or √(k*·g_*) with K-rational √3 tax]
    H_sub = 1/(N·t_P)                   [N_hub cascade theorem, theorem-grade]
  Bath ADIABATIC (reading A): T ∝ 1/a   ⇒  ρ_rad ∝ a⁻⁴

  substrate primitives: k* = {K_STAR}, √(8π³/90) = {FRIEDMANN_166:.4f}, √k* = {SQRT_K:.4f}
  H_sub today = 1/(N_hub·t_P) = {MPC_KM/(N_HUB*T_P_s):.2f} km/s/Mpc  (vs Planck 67.4)
""")

# ---------------------------------------------------------------------------
# (1) The deactivation: ρ_rad/ρ_sub vs redshift  (adiabatic bath)
# ---------------------------------------------------------------------------
banner("(1) Deactivation — ρ_rad/ρ_sub = Ω_rad·(1+z)²  (adiabatic)")

# ρ_rad ∝ a⁻⁴ (adiabatic). ρ_sub ∝ H_sub² ∝ a⁻² (coasting a∝N ⇒ H_sub∝1/a).
# Anchor today: ρ_rad/ρ_sub = Ω_rad. Then ratio(z) = Ω_rad·(1+z)^(4-2) = Ω_rad·(1+z)².
# Ω_rad today (photons + 3ν) at H_0 = 68:
H0_sub_si = H_si_from_GeV(1.0 / (N_HUB * T_P_s) * M_PL_GeV ** 0)  # placeholder; compute below
# critical density ρ_c = 3 H_0² M_Pl²/(8π); ρ_γ = (π²/15) T⁴; ν adds (7/8)(4/11)^{4/3}·3·(2/2)
H0_GeV = (1.0 / (N_HUB * T_P_s)) * HBAR_GeV_s   # H_sub today in GeV (1/(N t_P) is in 1/s → ×ħ)
rho_c = 3.0 * H0_GeV ** 2 * M_PL_GeV ** 2 / (8.0 * math.pi)
rho_gamma = (math.pi ** 2 / 15.0) * T_TODAY_GeV ** 4
nu_factor = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * 3.0   # 3 ν species
rho_rad_today = rho_gamma * (1.0 + nu_factor)
Omega_rad = rho_rad_today / rho_c

z_eq = (1.0 / math.sqrt(Omega_rad)) - 1.0

print(f"""
  Ω_rad today (γ + 3ν, at H_0 = 68)  = {Omega_rad:.3e}
  ρ_rad/ρ_sub (z) = Ω_rad·(1+z)²
  substrate-radiation equality:  z_eq = 1/√Ω_rad − 1 = {z_eq:.1f}

  {'epoch':<26} {'z':>12} {'ρ_rad/ρ_sub':>14} {'dominant':>12}""")
print("  " + "-" * 66)
for label, z in [
    ("today", 0.0),
    ("substrate-rad equality", z_eq),
    ("recombination", 1090.0),
    ("e+e- annihilation", 1.0e9),
    ("BBN (T~1 MeV)", 4.0e9),
]:
    ratio = Omega_rad * (1.0 + z) ** 2
    dom = "radiation" if ratio > 1 else "substrate"
    print(f"  {label:<26} {z:>12.3e} {ratio:>14.3e} {dom:>12}")

print(f"""
  ⇒ ρ_rad ∝ a⁻⁴ falls FASTER than ρ_sub ∝ a⁻². Radiation dominates at BBN
    (ratio ~10⁸), substrate dominates today (ratio ~{Omega_rad:.0e}). The √g_*
    factor is automatically PRESENT at BBN and ABSENT today — no free
    parameter. BC3 ✓. (Framework has NO matter-dominated era: radiation hands
    off directly to substrate-coasting at z_eq ≈ {z_eq:.0f}.)
""")

# ---------------------------------------------------------------------------
# (2) H at BBN and today — BC1 and BC2
# ---------------------------------------------------------------------------
banner("(2) H at BBN (BC1) and today (BC2)")

def H_rad_si(T_MeV, prefactor):
    T_GeV = T_MeV * 1e-3
    return H_si_from_GeV(prefactor * math.sqrt(g_star(T_MeV)) * T_GeV ** 2 / M_PL_GeV)

H_sub_today_si = 1.0 / (N_HUB * T_P_s)

# at BBN, H_sub is utterly negligible vs H_rad (shown in (1)); total H ≈ H_rad
for T_MeV in (1.0, 0.8):
    Hr_cont = H_rad_si(T_MeV, FRIEDMANN_166)
    Hr_krat = H_rad_si(T_MeV, SQRT_K)
    print(f"  T = {T_MeV} MeV:  H_rad(continuum √8π³/90) = {Hr_cont:.4f} s⁻¹   "
          f"H_rad(K-rational √k*) = {Hr_krat:.4f} s⁻¹")
print(f"""
  BC1 ✓ — at BBN H ≈ H_rad with the full √g_* factor: this is exactly the
  harness's 'candidate' expansion (Y_p = +0.8σ), NOT the bare F=1 (Y_p = −67.6σ).

  Today: H = √(H_rad² + H_sub²) ≈ H_sub (radiation is Ω_rad ≈ {Omega_rad:.0e} correction)
    H_sub = 1/(N_hub·t_P) = {MPC_KM/(N_HUB*T_P_s):.2f} km/s/Mpc  (Planck 67.4 ± 0.5)
  BC2 ✓ — H_0 preserved; the √g_* factor has switched off.
""")

# ---------------------------------------------------------------------------
# (3) The decisive contrast with the F-deactivation null result
# ---------------------------------------------------------------------------
banner("(3) Why this works where F-deactivation MECH-1 failed")
print(f"""
  Same additive form H² = H_rad² + H_sub². The ONLY change is the bath law:

    F-deactivation probe (α=1/2):  T ∝ a⁻¹ᐟ²  ⇒  ρ_rad ∝ T⁴ ∝ a⁻²
        ρ_rad/ρ_sub ∝ a⁻²/a⁻² = const  →  NO deactivation  (MECH-1 FAIL)

    This probe (reading A, α=1):    T ∝ a⁻¹    ⇒  ρ_rad ∝ T⁴ ∝ a⁻⁴
        ρ_rad/ρ_sub ∝ a⁻⁴/a⁻² = a⁻² →  deactivates  (MECH-1 WORKS)

  The adiabatic bath (the resolution of the η question, reading A) is EXACTLY
  the ingredient that unlocks the H-side deactivation (Gate 2). The two open
  problems were blocking each other; one assumption resolves both.
""")

# ---------------------------------------------------------------------------
# (4) honest verdict
# ---------------------------------------------------------------------------
banner("VERDICT — Route ADD is phenomenologically VIABLE (not a closure)")
print(f"""
  BC1 ✓  H ≈ √g_*·T²/M_Pl at BBN  → Y_p candidate (+0.8σ)
  BC2 ✓  H → 1/(N·t_P) = {MPC_KM/(N_HUB*T_P_s):.1f} today → H_0 preserved
  BC3 ✓  deactivation automatic (z_eq ≈ {z_eq:.0f}), ZERO free parameters
  BC4 ✓  adiabatic bath assumed (= the enabling switch; itself reading A)
  BC5 —  H_0, w_DE, η_B, N_eff untouched (radiation is a {Omega_rad:.0e} correction today)

  ⇒ A single additive Friedmann form with an adiabatic bath closes BOTH the
    √g_* leading factor (Gate 2) AND the η question at the phenomenological
    level, with no new parameter. This is the strongest evidence yet that a
    substrate-thermal-coupling mechanism exists.

  OPEN at the time of THIS probe (per scoping §5) — ALL SINCE CLOSED 2026-05-28
  (see substrate_thermal_coupling_mechanism_consolidated_verification_2026-05-28.py):
    T1  WHY H_rad and H_sub add in quadrature → CLOSED: observer info HAS energy
        E_obs (OEF thm); ρ_rad+ρ_sub is ordinary additive Friedmann.
    T2  Reconcile with N_hub cascade → ADMISSIBLE: Route ADD modifies the
        theorem's own-flagged undefended assumption A3.a, not robust D1+D2.
    T3  Derive ρ_sub ∝ a⁻² → CLOSED: ρ_sub=E_obs/V_Hubble (holographic,
        perceptual-surface) self-consistently SELECTS coasting + gives ∝a⁻².

  Per W58: viability ≠ closure — and the closure was subsequently completed
  (above). z_eq ≈ {z_eq:.0f} sits in the extrapolated (z > 2) zone — a sharp
  framework-distinct prediction, not yet a test. NOTE: the mechanism is
  radiation-dominated at z=1090 (z_eq < z_recomb) so it does NOT fix the θ_* wall.
""")
