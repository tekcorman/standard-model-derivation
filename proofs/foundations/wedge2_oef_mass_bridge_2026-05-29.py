#!/usr/bin/env python3
# ============================================================
# WEDGE-2: can a MASS be derived as kappa*surprise through the OEF —
# wiring the observer's TEMPORAL posterior to the SPATIAL (srs) model?
# ============================================================
#
# Scope: internal research notes
# The deep test. OEF (theorem_observer_energy_functional.md): E_obs = kappa*S
# (energy = Landauer-scaled accumulated surprise). A mass is an energy. So the
# claim: a particle's mass = kappa * (its pattern's surprise). If S can be
# computed from the observer's machinery and gives the mass, the spatial and
# temporal observer models are wired (BRIDGE). If S is just the srs Yukawa
# relabeled, it is a relabeling, not a derivation.
#
# FIRST, the functional-form check (the crux the OEF theorem forces):
#   - OEF: E_obs = kappa * S_total   (LINEAR in surprise; the ACCUMULATED total
#     over all N observations -> a cosmological-scale quantity, ~ kappa*N).
#   - framework mass: m = (v/sqrt2) * y, with y ~ (2/3)^L  (EXPONENTIAL in walk
#     length L; a single-pattern quantity).
#   These are DIFFERENT functional forms. The OEF total is not a single mass.
#   The only consistent single-pattern reading: define the pattern surprise
#     S := -log2(y) bits  ->  m = (v/sqrt2) * 2^(-S).
#   That is consistent, BUT the test is whether S is INDEPENDENT (observer) or
#   = -log2(srs Yukawa) (relabeling).

import math

V_HIGGS = 246.0          # GeV (framework v_higgs, from the N_hub temporal cascade)
SQRT2 = math.sqrt(2)
S_CONFIRM = -math.log2(2/3)   # 0.585 bits: per-step branch surprise = OEF S_confirm (k=3)

# framework leading Yukawas (srs-spatial), from the sector audit:
#   top:    y_t = 1            (Type-II saturation, walk length L=0)
#   bottom: y_b = (2/3)^10     (full girth cycle, L=g=10)
#   tau:    y_tau = (5/3)(2/3)^8 / 3^2   (alpha1_full / k*^2, L=8 + dark factor)
PARTICLES = {
    "top":    {"y": 1.0,                       "L": 0,  "m_obs": 172.7},
    "bottom": {"y": (2/3)**10,                 "L": 10, "m_obs": 4.18},
    "tau":    {"y": (5/3)*(2/3)**8 / 9,        "L": 8,  "m_obs": 1.777},
}


def main():
    print("=" * 72)
    print("WEDGE-2: mass = kappa*surprise through the OEF? (spatial<->temporal wiring)")
    print("=" * 72)

    print("\n[1] Functional-form check (does the OEF directly give a mass?)")
    print("    OEF E_obs = kappa*S_total : LINEAR in surprise, ACCUMULATED (~kappa*N).")
    print("    framework mass m=(v/sqrt2)*y, y~(2/3)^L : EXPONENTIAL in walk length.")
    print("    -> different functional forms. The OEF total is NOT a single mass.")
    print("    Only consistent single-pattern reading: S := -log2(y),  m=(v/sqrt2)*2^-S.")

    print("\n[2] Pattern surprise S = -log2(y) and the OEF mass m=(v/sqrt2)*2^-S:")
    print("    particle   y (srs)      S=-log2(y)   L*0.585   m_OEF      m_obs    note")
    for name, p in PARTICLES.items():
        y = p["y"]; S = -math.log2(y); m = (V_HIGGS/SQRT2)*y
        Lcheck = p["L"]*S_CONFIRM
        note = ("S=0 -> heaviest (fully confirmed)" if name == "top"
                else f"S = L*0.585 ? {'yes' if abs(S-Lcheck)<0.3 else 'no (dark factor)'}")
        print(f"    {name:7}   {y:.6f}   {S:8.3f} b   {Lcheck:6.3f}   "
              f"{m:8.3f}   {p['m_obs']:6.3f}   {note}")

    print("\n[3] Is S an INDEPENDENT observer quantity, or the srs Yukawa relabeled?")
    print("    S = -log2(y) = L * (-log2(2/3)) = L * 0.585 bits (+ dark factors).")
    print("    - L = girth / winding number = a SPATIAL srs property; the observer's")
    print("      TEMPORAL machinery (Beta posterior, N) produces no specific L=8,10.")
    print("    - 0.585 = -log2(2/3): coincides with the observer's Bayesian confirm-")
    print("      surprise ONLY at k=3 (wedge-1: coincidence, not structural).")
    print("    => S is srs-SPATIAL. 'mass = kappa*surprise' is a consistent")
    print("       INTERPRETATION (top=0 surprise=heaviest; lighter=more surprise),")
    print("       but the surprise IS the srs Yukawa. NOT an independent derivation.")

    print("\n[4] What the observer DOES genuinely contribute to the mass:")
    print(f"    m = (v/sqrt2) * 2^(-S)  =  [TEMPORAL scale v(N)]  x  [SPATIAL pattern 2^-S].")
    print(f"    - v = {V_HIGGS} GeV is the N_hub cascade (v ~ M_Pl/N^(1/4)) = TEMPORAL/observer.")
    print(f"    - 2^(-S) = y = srs walk = SPATIAL.")
    print(f"    So the mass ALREADY combines both observer models: the observer")
    print(f"    supplies the SCALE (v from N); srs supplies the dimensionless STRUCTURE.")

    print("\n[5] The running (where the sigma-gaps + the dynamics live):")
    print("    1/alpha(mu) = 1/alpha_GUT - (b/2pi)*ln(mu/M_GUT).")
    print("    - the ln(mu) FORM is logarithmic = the observer's MDL model-refinement")
    print("      (Rissanen ~ (1/2) log N : register-filling) -> FORM is observer-MDL.")
    print("    - the COEFFICIENT b is the un-derived part (recurrence-count stall),")
    print("      and for the gauge sector b is ADOPTED textbook MSSM (cl6_fock_table")
    print("      phantom). So the running FORM bridges to the observer; the")
    print("      COEFFICIENTS do not (yet) and are partly adopted.")

    print("\n" + "=" * 72)
    print("VERDICT: NO direct OEF->mass bridge; PARTIAL unification, already present.")
    print("=" * 72)
    print("""  - The OEF does NOT directly produce masses: E_obs=kappa*S is linear &
    accumulated; masses are exponential in a single pattern's walk length. The
    only consistent reading, m=(v/sqrt2)*2^(-S), is a meaningful INTERPRETATION
    (top = zero-surprise = heaviest; lighter fermions = more-surprising, mass
    exponentially suppressed) but NOT an independent derivation: the surprise S
    is the srs-spatial Yukawa, coincident with the observer's Bayesian surprise
    only at k=3 (wedge-1).

  - PARTIAL unification IS already present and real: m = [v(N), temporal/observer]
    x [y, spatial/srs]. The observer genuinely supplies the absolute SCALE (via
    N = register size) and the FORM of running (logarithmic = MDL register-
    filling). srs supplies the dimensionless STRUCTURE. So the framework is
    NEITHER pure JBOP (observer owns scale, time, cosmology, running-form) NOR
    fully unified (structure is irreducibly spatial; running COEFFICIENTS are
    un-derived and partly adopted).

  - The 'missing something' the whole thread chased is now located precisely:
    it is the un-derived RUNNING COEFFICIENTS (the beta-functions / the +4) --
    MDL in FORM but not derived from register dynamics, and partly an adopted
    MSSM number. That -- not a grand observer-rewrite -- is the concrete gap.
    Closing it (running coefficients from register-filling MDL) is the real,
    bounded unification target; the recurrence-count program is its home and
    stalled partly against the adopted +4 (now known).""")
    print("=" * 72)


if __name__ == "__main__":
    main()
