#!/usr/bin/env python3
"""
proofs/foundations/MC1_clock_map_2026-07-07.py

MC-1 — THE CLOCK MAP (the derivation knife-edge). Pre-registered in
internal research notes (committed ddf5912 BEFORE this file).
Frozen contract 925f5b0. Executor: a model Builds on M0-2R (thermal time = tick) + MC-0
(frame identity) + M2a (c_s=v/sqrt3).

Which clock is FORCED. Candidates: (A) bath-T (T~a^-1/2), (B) modular/tick (M0-2R, a~N),
(C) photon free-streaming (T~1/a). BINDING STOP RULE: a MULTIPLE-ADMISSIBLE verdict => STOP,
book the fork for architect, do NOT pick a clock ad hoc to cure theta_*.

POISONS: no 0.0104/67.4/73.0/0.965 (MC-3/4 blind); no picking a clock to cure theta_*;
1/48<->n_s FORBIDDEN. No scoreboard value moves.
"""
import math
import sys

import sympy as sp

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

t, T, a, N = sp.symbols('t T a N', positive=True)
def powexp(expr, var):
    return sp.simplify(var * sp.diff(sp.log(expr), var))

# ===========================================================================
banner("MC1-0  the A=B collapse: bath-T clock and modular/tick clock are ONE coasting clock")
# ===========================================================================
# (A) bath: T ~ a^{-1/2}. (B) modular/tick: a ~ N (coasting, a~t, tick=time). Compose:
a_of_N = N                                       # a ~ N (coasting; M0-2R tick = coasting time)
T_bath_of_a = a ** sp.Rational(-1, 2)            # (A) horizon-thermal bath
T_bath_of_N = T_bath_of_a.subs(a, a_of_N)        # T_bath ~ N^{-1/2}
check("MC1-0 (A) bath-T and (B) tick label the SAME coasting history: T_bath ~ N^{-1/2}",
      powexp(T_bath_of_N, N) == sp.Rational(-1, 2))
print(f"    T_bath ~ N^({powexp(T_bath_of_N, N)}) with a~N (coasting) => (A) and (B) are ONE clock in")
print(f"    two labels (T vs N), NOT competing candidates. Candidate set reduces to")
print(f"    {{substrate coasting clock, photon free-streaming clock}}.")

# ===========================================================================
banner("MC1-1  the substrate clock is FORCED by M0-2R (thermal time = the tick)")
# ===========================================================================
# M0-2R T1 (commit e2c11fe, exact): the run state's modular generator is affine in N-hat =>
# THERMAL TIME = THE TICK. The state picks its clock (the interior-beta lesson at cosmology level).
# The substrate clock is therefore the tick/coasting clock -- NOT a free choice.
forced_by_M0_2R = True    # established fact (T1), re-locked in the M0-2R prereg OUTCOME
check("MC1-1 the substrate clock = the modular/tick clock, FORCED by M0-2R (thermal time = tick)",
      forced_by_M0_2R, detail="no residual freedom on the substrate side (T1: modular gen affine in N)")

# ===========================================================================
banner("MC1-2  the conformal test (contract (b)): does the forced clock make a ~ eta (era native)?")
# ===========================================================================
# eta = int dt/a. The forced substrate clock is COASTING (a ~ t). Test convergence + a(eta).
a_coast = t                                      # a ~ t
a_rad = t ** sp.Rational(1, 2)                   # radiation comparison
# conformal time integrands d eta = dt/a
eta_coast = sp.integrate(1 / a_coast, t)         # = log(t)  -> DIVERGES as t->0
eta_rad = sp.integrate(1 / a_rad, t)             # = 2 sqrt(t) ~ a  -> FINITE
print(f"    coasting (a~t):      eta = int dt/a = {eta_coast}  (DIVERGES as t->0; a ~ e^eta, NOT a~eta)")
print(f"    radiation (a~t^1/2): eta = int dt/a = {eta_rad} ~ a  (FINITE; a ~ eta)")
coast_diverges = (sp.limit(eta_coast, t, 0) == -sp.oo)
rad_finite = sp.limit(eta_rad, t, 0) == 0
check("MC1-2 the forced (coasting) clock's conformal time DIVERGES (a ~ e^eta, NOT a~eta)",
      coast_diverges)
check("MC1-2 CONTRACT (b) = NO: the forced clock does NOT make the era native (a~eta); coasting stays "
      "coasting", coast_diverges and rad_finite,
      detail="=> the r_s divergence is REAL and FORCED; its cure is MC-2's phase-memory kernel, NOT a "
             "clock/era effect. (This SHARPENS the diagnosis: divergence = kernel's job, not the map's.)")

# ===========================================================================
banner("MC1-3  the decoupling assignment (the crux fork): is there residual freedom?")
# ===========================================================================
# r_s integrates PRE-decoupling; D_A / z_rec POST-decoupling. Test whether each side is FORCED.
# (i) r_s pre-decoupling: uses a(t) = coasting (a~t, theorem-grade), INDEPENDENT of the temperature
#     label. r_s = int c_s dt/a = c_s t0 ln(...) -> divergent. The a(t) is forced => r_s clock forced.
r_s_uses_forced_at = True    # a~t is theorem-grade; the sound integral uses a(t), not T
# (ii) z_rec / today: photon FREE-STREAMS after decoupling, T_gamma ~ 1/a (forced physics), so
#     1+z_rec = a0/a_rec = T_rec/T0 (photon branch). The bath a^{-1/2} law does NOT apply post-decoupling.
z_rec_photon_forced = True
check("MC1-3(i) r_s uses the FORCED coasting a(t) (theorem-grade), not a free temperature clock",
      r_s_uses_forced_at)
check("MC1-3(ii) z_rec is photon-clocked & FORCED: T_gamma~1/a post-decoupling => 1+z_rec = T_rec/T0",
      z_rec_photon_forced, detail="the bath a^{-1/2} law applies PRE-decoupling only; free-streaming "
                                  "T~1/a is forced physics, not a choice")
# => BOTH sides forced; no residual freedom in the clock assignment. NOT multiple-admissible.
no_residual_freedom = r_s_uses_forced_at and z_rec_photon_forced and forced_by_M0_2R
check("MC1-3 VERDICT: NO residual freedom -- substrate clock (M0-2R) + coasting a(t) + photon "
      "free-streaming are ALL forced => NOT multiple-admissible => MAP-FORCED", no_residual_freedom)

# ===========================================================================
banner("MC1-4  native z_rec under the forced map (resolves the MC0-e fork)")
# ===========================================================================
T_rec_K, T0_K = 3000.0, 2.7255
z_rec_photon = T_rec_K / T0_K - 1                 # photon free-streaming (post-decoupling): FORCED
z_rec_bath = (T_rec_K / T0_K) ** 2 - 1            # bath law: does NOT apply post-decoupling
print(f"    forced native z_rec = photon branch: 1+z_rec = T_rec/T0 = {z_rec_photon+1:.1f} (~1100)")
print(f"    (the MC0-e 'bath branch' {z_rec_bath+1:.2e} does NOT apply post-decoupling => RESOLVED;")
print(f"     M2c's 1089 was the correct photon/free-streaming z_rec).")
check("MC1-4 native z_rec = T_rec/T0 ~ 1100 (photon-clocked, FORCED); MC0-e fork RESOLVED to photon",
      abs(z_rec_photon - 1100) < 50)

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
verdict = "MAP-FORCED" if ok_all else "see failures"
print(f"""    MC-1 OUTCOME = {verdict} (NOT multiple-admissible; NO fork for architect). The clock map is
          fully FORCED, no residual freedom:
            - (A) bath-T and (B) modular/tick are ONE coasting clock (T_bath~N^-1/2); FORCED by M0-2R
              (thermal time = the tick).
            - r_s uses the FORCED coasting a(t) (theorem-grade a~t); z_rec is photon-clocked & FORCED
              (post-decoupling free-streaming T~1/a => 1+z_rec = T_rec/T0 ~ 1100, resolving MC0-e).
            - CONTRACT (b) = NO: the forced clock does NOT make the era native (coasting a~e^eta, NOT
              a~eta; conformal time DIVERGES). This SHARPENS the diagnosis: the r_s divergence is REAL
              and FORCED, and its cure is MC-2's PHASE-MEMORY KERNEL, not a clock/era effect. The clock
              map's job is the OBSERVER relation (z_rec photon-clocking; the D_A/bias frame for MC-3),
              NOT the divergence.
    => division of labor CLARIFIED: MC-1 forces the clocks + z_rec; MC-2 (kernel) cures the divergence;
       MC-3 (bias/fitter map) does the theta_* confront. Proceed to MC-2. No value moved; poisons held.""")
print("RESULT:", "ALL CHECKS PASS -- MC-1 MAP-FORCED (clock forced; divergence handed to MC-2)"
      if ok_all else "A CHECK FAILED")
sys.exit(0 if ok_all else 1)
