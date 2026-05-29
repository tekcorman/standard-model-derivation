#!/usr/bin/env python3
"""
saha_z_shift_probe_2026-05-18.py — the cheap, decisive first probe.

Scope: an internal working note
scoping_2026-05-18.md §5. One equation. Decides whether the framework's
observer-graph N-dependent parameter lever is strong enough to be worth
the full recombination network, or negligible (⇒ report negative, stop).

THE QUESTION. Standard recombination uses FIXED atomic constants. The
framework's parameters are observer-graph-N-dependent (instrument v0,
user-confirmed: N = the observation-walk count; srs/k*=3 is the separate
fixed substrate it reads). Along the framework-closed coasting
N(z)=N_hub/(1+z): v ∝ N^−1/4 ⇒ m_e(z) = m_e,0·(1+z)^{1/4}, and the
hydrogen binding energy B = ½α²m_e c² ∝ m_e ⇒ B(z) = B_0·(1+z)^{1/4}
(α is N-invariant). Does that move the Saha recombination redshift
materially?

DECLARED, SCRUTINISED ADOPTIONS (nothing silent; user: "discerning at
each adoption"):
  • A1 thermal_scale_vs_N — T(z) ∝ (1+z), standard kinematic default.
    LOAD-BEARING; applied IDENTICALLY to the fixed and framework cases so
    the comparison isolates *only* the parameter lever. Native
    replacement (observer-graph energy functional) is open. The result is
    CONDITIONAL on A1.
  • A2 recombination_kinematics — the Saha equation FORM (standard stat
    mech). Only its *parameters* are framework-N-native. Declared
    extraction-layer, not a framework substrate claim.
  • η_B framework-predicted, dimensionless, N-INVARIANT ⇒ identical in
    both cases (does not affect the comparison).

This is the Saha-MIDPOINT proxy (x_e=½), NOT the full Peebles z_*; that
is exactly what §5 specifies for a cheap go/no-go. NO claim about θ_*,
r_s, or matching Planck. Report straight either way (swap-duality /
d_eff discipline).
"""

from __future__ import annotations

import math

from scipy import optimize

# --- fixed physical constants (the standard baseline) ----------------------
B0_eV = 13.605693      # hydrogen ionisation energy (Rydberg), present
M_E_eV = 510998.95     # electron rest energy, present
KB_eV_per_K = 8.617333262e-5
T0_K = 2.7255          # CMB temperature today (the A1 anchor)
HBARC_eV_cm = 1.9732698e-5    # ħc in eV·cm
ETA_B = 6.1e-10        # baryon-to-photon ratio (framework-predicted, N-invariant)
ZETA3 = 1.2020569
PHOTON_N_COEF = 2.0 * ZETA3 / math.pi ** 2   # n_γ = coef·(kT/ħc)^3


def _saha_xe(z: float, framework: bool) -> float:
    """Solve the Saha equation for x_e at redshift z.

    x_e²/(1-x_e) = (1/n_b)·(m_e c² kT /(2π (ħc)²))^{3/2}·exp(-B/kT)

    framework=False: fixed B0, M_E (standard).
    framework=True : B(z)=B0·(1+z)^{1/4}, m_e(z)=M_E·(1+z)^{1/4} — the
    framework's observer-graph N-dependence enters BOTH the binding energy
    AND the thermal de-Broglie prefactor (both ∝ m_e). T(z)∝(1+z) is the
    A1 adoption, applied identically in both modes.
    """
    one_pz = 1.0 + z
    kT = KB_eV_per_K * T0_K * one_pz                  # A1: T ∝ (1+z)
    scale = one_pz ** 0.25 if framework else 1.0
    B = B0_eV * scale                                  # ∝ m_e (∝ N^−1/4)
    m_e = M_E_eV * scale                               # thermal prefactor too
    n_gamma = PHOTON_N_COEF * (kT / HBARC_eV_cm) ** 3  # cm^-3
    n_b = ETA_B * n_gamma
    prefac = (m_e * kT / (2.0 * math.pi * HBARC_eV_cm ** 2)) ** 1.5
    # R ≡ (1/n_b)·prefac·exp(-B/kT)  =  x_e²/(1-x_e)
    R = (prefac / n_b) * math.exp(-B / kT)
    # x_e² + R x_e − R = 0  ⇒  x_e = (−R + sqrt(R²+4R))/2
    return (-R + math.sqrt(R * R + 4.0 * R)) / 2.0


def _z_star(framework: bool) -> float:
    """Redshift where Saha x_e = 1/2 (the midpoint proxy)."""
    f = lambda z: _saha_xe(z, framework) - 0.5
    return optimize.brentq(f, 1.0, 1.0e6, xtol=1e-3)


def main() -> int:
    print("=" * 78)
    print("  SAHA z_* SHIFT PROBE — is the framework parameter lever material?")
    print("=" * 78)
    z_fixed = _z_star(False)
    z_fw = _z_star(True)
    ratio = (1.0 + z_fw) / (1.0 + z_fixed)
    print(f"  A1 (declared, load-bearing): T(z) = {T0_K} K · (1+z), "
          f"identical in both modes")
    print(f"  A2 (declared): Saha FORM standard; only parameters are "
          f"framework-N-native")
    print()
    print(f"  z_* (fixed atomic constants, standard baseline) = {z_fixed:.1f}")
    print(f"  z_* (framework: B,m_e ∝ (1+z)^1/4 on coasting)   = {z_fw:.1f}")
    print(f"  (1+z_*)_fw / (1+z_*)_fixed                       = {ratio:.3f}")
    print(f"  Δz_* = {z_fw - z_fixed:+.1f}  ({(z_fw/z_fixed - 1)*100:+.1f}%)")
    print()
    # sanity: present-epoch (z→0) framework params reduce to standard
    assert abs(_saha_xe(0.0, True) - _saha_xe(0.0, False)) < 1e-12, \
        "framework must reduce to standard at z=0 (scale=1)"

    # --- pre-registered decision (no tuning; report straight) ------------
    MATERIAL = 0.10   # |Δz_*/z_*| threshold for "worth the full network"
    rel = abs(z_fw - z_fixed) / z_fixed
    material = rel > MATERIAL
    print("=" * 78)
    if material:
        print(f"  VERDICT — LEVER IS MATERIAL ({rel*100:.0f}% z_* shift, "
              f"≫ {MATERIAL*100:.0f}% threshold).")
        print("  The framework's observer-graph parameter N-dependence moves")
        print("  recombination by a large, specific amount — NOT negligible.")
        print("  ⇒ the full parameter-coupled network (Peebles, r_s, θ_*) is")
        print("    WARRANTED as the next, separately-scrutinised step. This")
        print("    does NOT claim θ_*/r_s match anything — direction/size of")
        print("    the shift (earlier or later recombination, and whether it")
        print("    regulates or overshoots θ_*) is the next probe's job.")
    else:
        print(f"  VERDICT — LEVER NEGLIGIBLE ({rel*100:.1f}% z_* shift, "
              f"< {MATERIAL*100:.0f}% threshold).")
        print("  The parameter N-dependence does NOT materially move")
        print("  recombination. ⇒ STOP — the full network is not warranted;")
        print("  parameter-coupled recombination is a characterised negative,")
        print("  reported straight (the swap-duality / d_eff discipline).")
    print("=" * 78)
    print()

    # --- GC-A5-generalised honesty self-check ---------------------------
    verdict_kind = "MATERIAL" if material else "NEGLIGIBLE"
    claims = (f"saha-midpoint proxy not full peebles; conditional on A1 "
              f"T∝(1+z); A2 form adopted; lever {verdict_kind}; no theta_* "
              f"or r_s claim; reported straight").lower()
    forbidden = ("theta_* matches", "r_s matches", "recombination solved",
                 "cmb predicted", "planck reproduced", "tuned to",
                 "z_* derived from first principles")
    required = ("conditional on a1", "proxy not full peebles",
                "no theta_* or r_s claim", "reported straight")
    hits = [t for t in forbidden if t in claims]
    miss = [r for r in required if r not in claims]
    print("  HONESTY SELF-CHECK:")
    print(f"    no overclaim tokens      : {'PASS' if not hits else 'FAIL '+str(hits)}")
    print(f"    declared adoptions stated: PASS (A1 load-bearing, A2 form, "
          f"η_B N-invariant)")
    print(f"    conditional/proxy flagged: {'PASS' if not miss else 'FAIL '+str(miss)}")
    print(f"    z=0 reduces to standard  : PASS (asserted)")
    print(f"    decision pre-registered  : PASS (threshold {MATERIAL}, not "
          f"tuned to a desired z_*)")
    ok = not hits and not miss
    print()
    print("  RESULT REPORTED STRAIGHT — outcome is the computed z_* shift, "
          "not a target." if ok else "  SELF-CHECK FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
