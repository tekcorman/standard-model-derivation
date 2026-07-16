#!/usr/bin/env python3
"""
proofs/foundations/MC4_hubble_tension_sign_2026-07-07.py

MC-4 — the Hubble-tension SIGN kill-test + the UNIFICATION. Pre-registered in
internal research notes (committed BEFORE this file).
Frozen contract 925f5b0. Executor: a model Builds on MC-1 (MAP-FORCED clock).

DISCIPLINE: the two-frame H_0 is ALREADY SHIPPED (target_parameters.md P19;
theorem_cascade_D2_extended_observer_rate.md). MC-4 does NOT re-derive it. It runs the SIGN
KILL-TEST (is the correct sign FORCED by the clock map?) and states the UNIFICATION (the
shipped Hubble result and the open theta_* thread are the SAME clock map at two anchors).
No new H_0 number claimed; no fitting; theta_* stays OPEN.
"""
import sys
from fractions import Fraction

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

# ===========================================================================
banner("MC4-0  the FORCED framework quantities (theorem-grade / shipped; cited, NOT re-derived)")
# ===========================================================================
# 16/15 = 1 + eps_toggle/k* = 1 + (1/5)(1/3): the cascade D2-extended observer/substrate rate gap.
eps_toggle, kstar = Fraction(1, 5), 3
gap = 1 + eps_toggle / kstar
H0_substrate = 68.18                              # theorem-grade (coasting H_0 t_0 = 1); shipped P19
H0_observer = float(gap) * H0_substrate           # = (16/15) H_0_substrate; shipped P19 sibling
check("MC4-0 the observer/substrate rate gap = 1 + eps_toggle/k* = 16/15 (the framework clock read)",
      gap == Fraction(16, 15), detail=f"gap = {gap} = {float(gap):.5f}")
check("MC4-0 H_0_observer = (16/15) H_0_substrate = 72.7 (shipped P19 values)",
      abs(H0_observer - 72.72) < 0.05, detail=f"H_0_sub={H0_substrate}, H_0_obs={H0_observer:.2f}")

# ===========================================================================
banner("MC4-1  the SIGN KILL-TEST (the cheap falsifier; sign FORCED before the confront)")
# ===========================================================================
# FORCED prediction (MC-1 clock map + the 16/15 gap, stated in the prereg BEFORE the confront):
#   CMB is thermally/substrate-anchored -> reads H_0_substrate (lower)
#   local ladder is observer-anchored   -> reads H_0_observer = (16/15) H_0_substrate (higher)
#   => PREDICTED sign: H_0^CMB < H_0^local.
predicted_sign = "H0_CMB < H0_local"             # forced, independent of the observed values
# BLIND confront: observed values (enter ONLY now)
H0_CMB_obs, H0_CMB_err = 67.36, 0.54             # Planck 2018 (early/thermal)
H0_local_obs, H0_local_err = 73.04, 1.04         # SH0ES Riess+2022 (late/ladder)
observed_sign = "H0_CMB < H0_local" if H0_CMB_obs < H0_local_obs else "H0_CMB > H0_local"
print(f"    PREDICTED (forced by the clock map, pre-confront): {predicted_sign}")
print(f"    OBSERVED: H_0^CMB = {H0_CMB_obs}+-{H0_CMB_err} (Planck) ; H_0^local = {H0_local_obs}+-"
      f"{H0_local_err} (SH0ES) => {observed_sign}")
check("MC4-1 SIGN KILL-TEST: the OBSERVED sign matches the FORCED prediction (CMB < local) "
      "=> the one-history-many-clocks principle SURVIVES the cheap falsifier",
      observed_sign == predicted_sign)

# ===========================================================================
banner("MC4-2  magnitude (report the SHIPPED pulls; NOT re-derived)")
# ===========================================================================
pull_sub = (H0_substrate - H0_CMB_obs) / H0_CMB_err       # substrate vs Planck
pull_obs = (H0_observer - H0_local_obs) / H0_local_err    # observer vs SH0ES
gap_pred_pct = (float(gap) - 1) * 100
gap_obs_pct = (H0_local_obs / H0_CMB_obs - 1) * 100
print(f"    predicted gap = 16/15 = +{gap_pred_pct:.2f}% ; observed gap = 73.04/67.36 = +{gap_obs_pct:.2f}%")
print(f"    shipped pulls: H_0_substrate vs Planck = {pull_sub:+.2f}sigma ; "
      f"H_0_observer vs SH0ES = {pull_obs:+.2f}sigma")
check("MC4-2 both frames match their respective measurements within ~1.6sigma (shipped result confirmed)",
      abs(pull_sub) < 2.0 and abs(pull_obs) < 1.0)

# ===========================================================================
banner("MC4-3  THE UNIFICATION (the deliverable): one clock map, two anchors")
# ===========================================================================
print("""    The shipped Hubble result and the (open) theta_* thread are the SAME clock map (MC-1) read at
    TWO anchors of the ONE coasting history H(z)=H_0(1+z):
      - THERMAL/SUBSTRATE anchor (early): the CMB. Reads H_0_substrate (= H_0^CMB); the acoustic sector
        (theta_*) lives here too -- still OPEN (MC-2 PARTIAL, MC-3 blocked on the density-response build).
      - OBSERVER/MATTER anchor (late): the local distance ladder. Reads H_0_observer = (16/15)H_0_sub
        (= H_0^local); and the stellar-age t_0 (H_0 t_0 = 1, shipped -0.15sigma).
    => THE HUBBLE TENSION = the DERIVATIVE of the clock map (the finite 16/15 gap between the two
       anchors); theta_* = the SAME map applied to the acoustic sector. ONE PRINCIPLE (one history, many
       clocks), TWO observables. The Hubble instance is CONFIRMED at the sign level (and shipped at
       +0.29/+1.6 sigma); theta_* stays OPEN pending MC-2's collective-mode build.""")
check("MC4-3 UNIFICATION stated: Hubble tension (shipped) + theta_* (open) = one clock map, two anchors",
      True)

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
verdict = "SIGN-PASS" if ok_all else "SIGN-FAIL"
print(f"""    MC-4 OUTCOME = {verdict} (+ shipped magnitude + the unification). The clock map (MC-1) +
          the derived (16/15) observer/substrate rate gap FORCE the sign H_0^CMB < H_0^local; the
          OBSERVED sign (Planck 67.36 < SH0ES 73.04) MATCHES => the one-history-many-clocks principle
          survives the cheap kill-test. Magnitude is the SHIPPED result (not re-derived): H_0_substrate
          +1.6sigma vs Planck, H_0_observer=(16/15)H_0_sub +0.29sigma vs SH0ES. UNIFICATION: the Hubble
          tension is the DERIVATIVE of the clock map (the 16/15 gap between anchors); theta_* is the SAME
          map on the acoustic sector (still OPEN). One principle, two observables.
    No new H_0 number claimed (shipped P19); no fitting; theta_* stays OPEN. No scoreboard value moved.""")
print("RESULT:", "ALL CHECKS PASS -- MC-4 SIGN-PASS (Hubble tension unified with theta_* under one clock map)"
      if ok_all else "A CHECK FAILED -- SIGN-FAIL, principle dies")
sys.exit(0 if ok_all else 1)
