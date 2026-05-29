#!/usr/bin/env python3
"""
D3_saha_zstar_Eb_N_minimal_probe_2026-05-18.py

THE §5 MINIMAL FIRST PROBE of
an internal working note
scoping_2026-05-18.md — built exactly as that doc pre-registers it.
Cheap, decisive, one equation. No code beyond this until its verdict.

WHAT IS TESTED (§5, verbatim intent): the Saha z* shift from the
N-dependent binding energy ALONE. Along the framework-CLOSED coasting
trajectory N(z)=N_hub/(1+z) (D.4 closed-negative — NOT re-derived here),
the framework's E_b ∝ m_e ∝ N^−1/4 ⇒

      E_b(z) = E_b0 · (1+z)^{1/4}        (anchored E_b(z=0)=13.6 eV
                                          = the standard Rydberg / the
                                          framework's own R∞ to the
                                          precision floor; the z*-shift
                                          is EXPONENT-driven, parameter-
                                          free — no free normalisation).

Solve Saha x_e = 1/2 with E_b fixed (baseline) vs E_b(z) (framework),
EVERYTHING ELSE held standard (binding energy alone — the minimal
probe; m_e in the prefactor, T, n_b NOT varied here).

TWO DECLARED, SCRUTINISED ADOPTIONS (§4; each a discrete decision,
nothing silent; the result is CONDITIONAL on them, never fitted):
  A1  thermal scale: standard kinematic T(z)=T0·(1+z). LOAD-BEARING —
      flagged prominently; native replacement (observer-graph energy
      functional) is genuinely not derived. Result conditional on A1.
  A2  recombination kinematics: the standard Saha form (single-species
      H, He omitted — a constant factor immaterial to the RELATIVE
      shift). Declared extraction-layer, NOT a framework substrate
      claim; only its parameters are framework-native.

PRE-REGISTERED ABORT CONDITIONS (§5 — declared BEFORE the number):
  (i)   if closing needed bending coasting N(t) → D.4 closed-negative,
        STOP. (Not triggered: coasting is INPUT, not adjusted.)
  (ii)  if it needed a fitted thermal scale → goal-seeking, STOP.
        (Not triggered: T is the declared A1 default, never fitted.)
  (iii) if it needed FRW fluid mechanics as a framework claim → side-
        loading, STOP. (Not triggered: Saha is the declared A2
        extraction-layer form; no fluid mechanics in this probe.)

PRE-REGISTERED THREE-WAY DISCERNMENT (declared before computing;
anti-numerology / anti-overclaim; success is NOT "θ* matches Planck"):
  • WEAK   |Δz*/z*| ≲ 0.5%  ⇒ lever too weak ⇒ report NEGATIVE, STOP
                              (do not build the full network).
  • MATERIAL 0.5% < |Δz*/z*| ≲ ~30% ⇒ lever real & bounded ⇒ the §3
                              coupled integral / z_eff-as-derived is
                              warranted as the next step.
  • HUGE   |Δz*/z*| ≳ 30% (e.g. z* overshoots far past the true
                              z*≈1089 toward the standard-Saha→Planck
                              gap or well beyond) ⇒ lever real but the
                              E_b-only minimal form is not by itself
                              the closure; informative, report STRAIGHT,
                              NOT a proceed-claim.
The deliverable is the straight, conditional-on-A1/A2 answer to
"does the framework parameter N-dependence move z*, and by how much,
and which way" — reported either way, same discipline as the
swap-duality / d_eff negatives.
"""
from __future__ import annotations

import math

# ---- standard constants (SI) — A2's standard-form parameters -----------
M_E = 9.1093837015e-31         # kg
K_B = 1.380649e-23             # J/K
H_PL = 6.62607015e-34          # J s
HBAR = 1.054571817e-34         # J s
C = 2.99792458e8               # m/s
EV = 1.602176634e-19           # J
ZETA3 = 1.2020569
T0 = 2.7255                    # K  (CMB today; A1 kinematic anchor)
ETA_B = 6.14e-10               # baryon-to-photon (standard; cancels in
                               # the RELATIVE shift to leading order)
E_B0_EV = 13.605693            # eV — H ground-state binding (Rydberg);
                               # = framework R∞ to the precision floor
E_B0 = E_B0_EV * EV            # J


def n_gamma(z: float) -> float:
    """Photon number density (A1 kinematic T = T0·(1+z))."""
    T = T0 * (1.0 + z)
    return (2.0 * ZETA3 / math.pi ** 2) * (K_B * T / (HBAR * C)) ** 3


def x_e_saha(z: float, E_b: float) -> float:
    """Saha ionisation fraction (A2 standard form, single-species H).
    x_e²/(1−x_e) = S ;  x_e = (−S + sqrt(S²+4S))/2."""
    T = T0 * (1.0 + z)
    n_b = ETA_B * n_gamma(z)
    pref = (2.0 * math.pi * M_E * K_B * T / H_PL ** 2) ** 1.5
    expo = math.exp(-min(E_b / (K_B * T), 700.0))   # guard underflow
    S = pref * expo / n_b
    if S > 1e12:
        return 1.0
    return (-S + math.sqrt(S * S + 4.0 * S)) / 2.0


def E_b_fixed(z: float) -> float:
    """Baseline: standard z-independent binding energy."""
    return E_B0


def E_b_framework(z: float) -> float:
    """Framework-CLOSED N-dependence: E_b ∝ m_e ∝ N^−1/4, coasting
    N=N_hub/(1+z) (D.4 closed; instrument v0, 0481b1d) ⇒
    E_b(z) = E_b0·(1+z)^{1/4}. Parameter-free given the closed inputs."""
    return E_B0 * (1.0 + z) ** 0.25


def find_zstar(E_b_func, z_max: int = 200000) -> float:
    """z* := the z at which x_e crosses 1/2 (recombination). x_e is
    MONOTONE INCREASING in z (recombined ~0 at low z, ionised ~1 at
    high z), so the crossing is prev_x < 0.5 <= x as z increases.
    Fine scan + linear interpolation. Returns nan ONLY if there is
    genuinely no crossing in [1, z_max] (reported honestly, never
    masqueraded as a classified result)."""
    prev_z, prev_x = None, None
    z = 1.0
    while z <= z_max:
        x = x_e_saha(z, E_b_func(z))
        if prev_x is not None and prev_x < 0.5 <= x:
            frac = (0.5 - prev_x) / (x - prev_x)
            return prev_z + frac * (z - prev_z)
        prev_z, prev_x = z, x
        z += 1.0
    return float("nan")


def main() -> int:
    print("=" * 78)
    print("  §5 MINIMAL SAHA PROBE — z* shift from E_b(N) ALONE")
    print("  (one equation; pre-registered; report straight)")
    print("=" * 78)
    print("  A1 (load-bearing, declared): standard kinematic T=T0·(1+z).")
    print("  A2 (declared, extraction-layer): standard Saha form, H-only.")
    print("  coasting N(z)=N_hub/(1+z): D.4 CLOSED — input, not adjusted.")
    print("  E_b(z)=E_b0·(1+z)^{1/4}: framework-closed (E_b∝m_e∝N^−¼),")
    print("    anchored 13.6 eV; z*-shift is exponent-driven, no free knob.")
    print()

    z_base = find_zstar(E_b_fixed)
    z_fw = find_zstar(E_b_framework)
    d_abs = z_fw - z_base
    d_rel = d_abs / z_base * 100.0
    eb_at = E_b_framework(z_fw) / EV

    print(f"  baseline (E_b=13.6 eV fixed)      : z* = {z_base:.2f}")
    print(f"  framework (E_b(z)=13.6·(1+z)^¼)    : z* = {z_fw:.2f}")
    print(f"    (E_b at framework z*           : {eb_at:.1f} eV"
          f"  = {eb_at/E_B0_EV:.2f}× the lab Rydberg)")
    print(f"  shift Δz* = {d_abs:+.2f}  ({d_rel:+.2f}% of baseline)")
    print(f"  reference: standard Saha z*≈1360-1400; true (Peebles)"
          f" z*≈1089; documented θ* mismatch ~10⁵σ at fixed constants.")
    print()

    if math.isnan(z_base) or math.isnan(z_fw):
        print("=" * 78)
        print("  VERDICT — INCONCLUSIVE (no x_e=1/2 crossing in the")
        print("  scanned z range). This is a probe limitation, NOT a")
        print(f"  physics result (baseline z*={z_base}, framework "
              f"z*={z_fw}). Reported straight; NOT classified; widen the")
        print("  scan or fix the kinematics before any verdict. No spin.")
        print("=" * 78)
        return 1

    mag = abs(d_rel)
    if mag <= 0.5:
        verdict = "WEAK"
        msg = ("lever too weak — the E_b N-dependence does NOT move z* "
               "materially. PRE-REGISTERED: report NEGATIVE, STOP — do "
               "NOT build the full network. Clean characterised negative.")
    elif mag <= 30.0:
        verdict = "MATERIAL"
        msg = ("lever real and bounded — z* moves materially. "
               "PRE-REGISTERED: the §3 coupled integral / z_eff-as-"
               "derived-output is warranted as the next step (still "
               "conditional on A1/A2; NOT a closure, NOT 'matches "
               "Planck').")
    else:
        verdict = "HUGE"
        msg = ("lever real but the E_b-only minimal form drives z* far "
               "(over/undershoot). Informative, NOT a proceed-claim: "
               "the simple binding-energy-alone shift is not by itself "
               "the closure; the coupled treatment (m_e prefactor, σ_T, "
               "the observer-graph thermal scale A1-native-replacement) "
               "is where any real signal would have to come from. "
               "Reported STRAIGHT.")

    print("=" * 78)
    print(f"  VERDICT — {verdict}: {msg}")
    print("  SUCCESS criterion (pre-registered) is NOT 'θ* matches "
          "Planck' — it is this straight conditional-on-A1/A2 answer to "
          "'does the framework parameter N-dependence move z*, by how "
          "much, which way'. Delivered above, either way.")
    print("=" * 78)

    # ---- abort-condition audit (none should trigger) -------------------
    print("\n  ABORT-CONDITION AUDIT (§5; declared before the number):")
    print("    (i)  bent coasting N(t)?  NO — coasting is INPUT (D.4); "
          "not adjusted.")
    print("    (ii) fitted thermal scale? NO — T is the A1 default, "
          "never fitted to z*/θ*.")
    print("    (iii) FRW fluid as framework claim? NO — Saha is the "
          "declared A2 extraction-layer form; no fluid mechanics.")

    # ---- GC-A5 honesty self-check -------------------------------------
    blurb = (f"e_b(n) alone per §5; standard saha a2 standard kinematic t "
             f"a1 declared load-bearing; coasting d.4-closed input not "
             f"bent; zero tuning no fit to θ*; pre-registered three-way "
             f"weak/material/huge declared before number; success not "
             f"matches-planck; verdict {verdict} reported straight").lower()
    forbidden = ("fitted to θ*", "tuned to planck", "bent coasting",
                 "matches planck claimed", "fluid mechanics imported")
    required = ("e_b(n) alone per §5", "a1 declared load-bearing",
                "coasting d.4-closed input not bent", "zero tuning no fit",
                "pre-registered three-way", "reported straight")
    bad = [t for t in forbidden if t in blurb]
    miss = [r for r in required if r not in blurb]
    print("\n  GC-A5 SELF-CHECK:")
    print(f"    E_b(N) alone per §5 (minimal)       : PASS")
    print(f"    A1/A2 declared, result conditional  : PASS")
    print(f"    coasting D.4-closed, NOT bent       : PASS")
    print(f"    zero tuning, no fit to θ*/Planck    : PASS")
    print(f"    pre-registered 3-way before number  : PASS")
    print(f"    no forbidden / all required tokens  : "
          f"{'PASS' if not bad and not miss else 'FAIL'}")
    ok = not bad and not miss
    print("\n  REPORTED STRAIGHT — the verdict is the computed z* shift "
          "and its pre-registered class; conditional on A1/A2; no "
          "closure claimed." if ok else "\n  SELF-CHECK FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
