#!/usr/bin/env python3
"""ERA-HANDOFF: is the cascade's ~50x thermal outlier exactly the radiation->matter->Lambda handoff?

The observer-flow dyadic ladder (theorem_observer_flow_dyadic_ladder) has ONE honest outlier:
the thermal rung T ~ N^-1/2. Inverting the radiation formula with TODAY's CMB T gives an N that
is ~50x (in N) too high (cascade_one_N_overdetermination). The probe's note: "TODAY is
Lambda-dominated, so the RADIATION bridge held in the radiation era." THIS probe tests whether
that ~50x is EXACTLY the era handoff -- i.e. the scale factor a grew FASTER than the radiation
N^1/2 track once the universe left radiation domination (matter a~N^2/3, then Lambda).

PHYSICS (framework spine + standard era structure -- the era transitions are IMPORTED standard
cosmology, NOT derived; the framework supplies only the spine H.N.t_P=1 and T~1/a):
  * spine:        H . N . t_P = 1  (exact, all eras; theorem)  =>  N = t / t_P  (t = proper time)
  * photons:      T ~ 1/a          (exact free-streaming, all eras)
  * radiation era: a ~ t^1/2 ~ N^1/2  =>  T ~ N^-1/2   (THE RUNG -- holds for z > z_eq)
  * matter era:    a ~ t^2/3 ~ N^2/3  =>  T ~ N^-2/3   (steeper; the rung breaks)
  * Lambda era:    a ~ exp(Ht)        =>  T falls faster still

TEST: anchor the rung at BBN (deep radiation), show it HOLDS to matter-radiation equality, then
show TODAY's overshoot = the actual a-growth (eq->today, = 1+z_eq) divided by the radiation
N^1/2 extrapolation -- and that this equals the cascade's ~50x (in N).
"""
import sys
import numpy as np

FAILURES = []
def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok: FAILURES.append(name)

# ---- constants (SI / standard cosmology; Planck-2018-ish) ----
t_P   = 5.391e-44          # Planck time, s
s_per_yr = 3.1557e7
Gyr   = 1e9 * s_per_yr
eV_K  = 8.617333e-5        # eV per kelvin
T0_K  = 2.7255            # CMB today, K
T0_eV = T0_K * eV_K       # 2.35e-4 eV
z_eq  = 3402.0            # matter-radiation equality (Planck 2018)
t0    = 13.797 * Gyr      # age of universe, s
# radiation-era anchor: BBN, T ~ 1 MeV at t ~ 1 s (deep radiation domination)
T_BBN_eV = 1.0e6
t_BBN    = 1.0           # s
# matter-radiation equality time (standard): ~ 51 kyr
t_eq  = 51100.0 * s_per_yr

# framework spine: N = t / t_P
N_BBN = t_BBN / t_P
N_eq  = t_eq  / t_P
N_0   = t0    / t_P
T_eq_eV = T0_eV * (1 + z_eq)     # T at equality = today's T blueshifted

print("=" * 84)
print(" ERA-HANDOFF: is the ~50x thermal outlier exactly the radiation->matter->Lambda handoff?")
print("=" * 84)
print(f"\n  spine N=t/t_P:   N_BBN={N_BBN:.2e}  N_eq={N_eq:.2e}  N_0={N_0:.2e}")
print(f"  temperatures:    T_BBN={T_BBN_eV:.2e} eV   T_eq={T_eq_eV:.3f} eV   T_0={T0_eV:.3e} eV")

# ---- EG1: the rung T~N^-1/2 HOLDS across the radiation era (BBN -> equality) ----
# if T ~ N^-1/2:  T_BBN/T_eq  should equal  (N_eq/N_BBN)^1/2
lhs = T_BBN_eV / T_eq_eV
rhs = (N_eq / N_BBN) ** 0.5
print(f"\n (EG1) rung in radiation era (BBN->equality): T_BBN/T_eq = {lhs:.3e} vs (N_eq/N_BBN)^1/2 = {rhs:.3e}")
gate("EG1 rung T~N^-1/2 HOLDS in radiation era (BBN->eq agree)", abs(lhs/rhs - 1) < 0.06,
     f"ratio {lhs/rhs:.3f}")

# ---- EG2: TODAY the rung OVERSHOOTS; quantify in T and in N ----
# extrapolate the radiation rung (anchored at BBN) to N_0:
T_rung_today = T_BBN_eV * (N_BBN / N_0) ** 0.5
overshoot_T = T_rung_today / T0_eV
overshoot_N = overshoot_T ** 2          # N ~ T^-2 in the rung
print(f"\n (EG2) extrapolate rung to today: T_rung(N_0)={T_rung_today:.3e} eV vs actual T_0={T0_eV:.3e} eV")
print(f"       overshoot = {overshoot_T:.2f}x in T  =  {overshoot_N:.1f}x in N  (cascade reported ~50x in N)")
gate("EG2 today's overshoot ~ 40-60x in N (matches cascade ~50x)", 30 < overshoot_N < 70,
     f"{overshoot_N:.1f}x in N")

# ---- EG3: the overshoot IS the era handoff (a-growth beyond the radiation track) ----
# actual a-growth equality->today:        a_0/a_eq = 1+z_eq
# radiation-extrapolated a-growth eq->today: (N_0/N_eq)^1/2  (continue a~N^1/2)
a_growth_actual = 1 + z_eq
a_growth_radextrap = (N_0 / N_eq) ** 0.5
handoff = a_growth_actual / a_growth_radextrap     # = how much faster a grew (matter+Lambda)
print(f"\n (EG3) a-growth equality->today:  actual (1+z_eq)={a_growth_actual:.0f}  vs  radiation-extrap"
      f" (N_0/N_eq)^1/2={a_growth_radextrap:.0f}")
print(f"       era-handoff factor = {handoff:.2f}x (in a = in T)  =  {handoff**2:.1f}x (in N)")
print(f"       => the overshoot ({overshoot_T:.2f}x in T) IS the matter+Lambda growth of a beyond N^1/2.")
gate("EG3 overshoot = era handoff (a grew faster than N^1/2 after eq)",
     abs(handoff / overshoot_T - 1) < 0.10, f"handoff {handoff:.2f} vs overshoot {overshoot_T:.2f}")

# ---- EG4: reconciliation -- use a RADIATION-ERA thermal anchor and the rung gives the right N ----
# at equality (radiation), invert the rung from BBN: predict N_eq, compare to true N_eq.
N_eq_from_rung = N_BBN * (T_BBN_eV / T_eq_eV) ** 2
print(f"\n (EG4) reconciliation: rung anchored at BBN predicts N_eq = {N_eq_from_rung:.2e} vs true {N_eq:.2e}")
gate("EG4 rung gives RIGHT N from a radiation-era anchor (use BBN/eq, not today's CMB)",
     abs(N_eq_from_rung / N_eq - 1) < 0.12, f"ratio {N_eq_from_rung/N_eq:.3f}")

print(f"""
{"="*84}
 VERDICT
{"="*84}
  The thermal rung T ~ N^-1/2 is RADIATION-ERA-LOCAL, and the cascade's ~50x (in N) outlier is
  EXACTLY the radiation->matter(->Lambda) handoff -- NOT a framework error:
   * EG1: the rung holds across the radiation era (BBN -> matter-radiation equality), to a few %.
   * EG2: extrapolated to today it overshoots ~{overshoot_T:.1f}x in T (~{overshoot_N:.0f}x in N) -- the cascade outlier.
   * EG3: that overshoot = the scale factor's actual growth (1+z_eq={a_growth_actual:.0f}) divided by the
     radiation N^1/2 extrapolation ({a_growth_radextrap:.0f}) = {handoff:.1f}x: after equality a grew as N^2/3
     (matter) then exp (Lambda), FASTER than N^1/2, so T fell faster and today's T is below the
     radiation extrapolation. The miss is the handoff, fully accounted.
   * EG4: anchored at a RADIATION-era point the rung gives the right N. So the over-determination
     should read the thermal rung at a radiation-era anchor (BBN / equality), where T~N^-1/2 holds
     -- NOT at today's Lambda-era CMB T. With that, the thermal row stops being an outlier.

  RECONCILED. The ladder is era-STRATIFIED below the spine: T ~ N^-1/2 (radiation), N^-2/3
  (matter), steeper (Lambda) -- one rung per era, the slope set by a(t) in that era. The spine
  H.N.t_P=1 and T~1/a are exact (all eras, framework); the era transitions are standard imported
  cosmology. HONEST: this is a CONSISTENCY/reconciliation result (the outlier is explained), not
  a new prediction; it imports z_eq, t_eq, the era a(t) laws.
""")
print("=" * 84)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}"); sys.exit(1)
print(" RESULT: ALL GATES PASS -- thermal outlier = radiation->matter->Lambda handoff, reconciled")
print("=" * 84); sys.exit(0)
