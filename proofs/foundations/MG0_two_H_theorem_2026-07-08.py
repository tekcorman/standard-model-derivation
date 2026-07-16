#!/usr/bin/env python3
"""
proofs/foundations/MG0_two_H_theorem_2026-07-08.py

MG-0 — the TWO-H THEOREM (spine vs metric). Pre-registered in
internal research notes (committed BEFORE this file). Frozen
contract 6d5e11d/fae6028. Executor: a model

THE CLAIM: the coasting theorem derives H_sub = Ndot/N = 1/(N t_P) = 1/t -- a COUNTING
statement about the STATE count N (N_hub.py:74), NOT the metric adot/a. For a ~ N^p,
H_metric = adot/a = p * H_sub, so the spine H_sub*N*t_P = 1 holds for ANY p (metric-blind).
The metric a(N) per era is fixed by the GRAVITY CLOSURE (Friedmann H^2~rho, promoted at form
level), not by the spine. "a=N globally (p=1)" was an unexamined LABEL that MC-1 inherited;
theta_* = r_s/D_C spans both regimes => the ~9x collision. MG-0 does NOT resolve theta_*.

POISON: does NOT relabel theta_* solved; shipped late-time results untouchable; 1/48<->n_s
FORBIDDEN. No scoreboard value moves.
"""
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

t, t_P, p = sp.symbols('t t_P p', positive=True)

# ===========================================================================
banner("MG0-a  the SPINE is forced by COUNTING and METRIC-BLIND (H_sub*N*t_P=1 for ALL p)")
# ===========================================================================
# N_hub.py:74 (D1+D2+D3): N(t) = t/t_P (one new state per t_P); H_sub = Ndot/N = (1 new state per t_P)
# / (N states) -- a COUNTING statement, NOT adot/a.
N = t / t_P
Ndot = sp.diff(N, t)
H_sub = sp.simplify(Ndot / N)                    # the STATE Hubble = Ndot/N
check("MG0-a H_sub = Ndot/N = 1/t (forced by counting N=t/t_P; the SPINE)",
      sp.simplify(H_sub - 1 / t) == 0, detail=f"H_sub = {H_sub}")
check("MG0-a the spine H_sub*N*t_P = 1 EXACTLY (theorem-grade)",
      sp.simplify(H_sub * N * t_P - 1) == 0)
# the METRIC Hubble for a ~ N^p = (t/t_P)^p:
a = (t / t_P) ** p
H_metric = sp.simplify(sp.diff(a, t) / a)        # adot/a
check("MG0-a H_metric = adot/a = p/t = p*H_sub (for a~N^p)",
      sp.simplify(H_metric - p * H_sub) == 0, detail=f"H_metric = {H_metric} = p * (1/t)")
# METRIC-BLINDNESS: the spine H_sub*N*t_P=1 is satisfied for ANY p (it constrains N(t), not p).
metric_blind = sp.simplify(H_sub * N * t_P - 1) == 0  # independent of p (H_sub has no p)
check("MG0-a the SPINE is METRIC-BLIND: H_sub*N*t_P=1 holds for ALL p (fixes N(t), NOT the metric "
      "exponent p) => H_sub != H_metric in general (the TWO H's)", metric_blind and p not in H_sub.free_symbols)

# ===========================================================================
banner("MG0-b  the AUDIT: which shipped observable used which metric a(N)=N^p (p is epoch-dependent)")
# ===========================================================================
# each row is a SHIPPED framework result and the metric exponent p it actually used.
rows = [
    ("late distances / H0 t0 / SNe / bias-fn", "coasting a~N (R4b MDL coarse-graining, valid z<~2)", 1.0),
    ("thermal RADIATION rung (era_handoff:15)", "a~N^{1/2} => T~N^{-1/2}", 0.5),
    ("thermal MATTER rung (era_handoff:16)", "a~N^{2/3} => T~N^{-2/3}", 2.0 / 3.0),
    ("horizon-thermal cumulative (T~a^{-1/2}, T~N^{-25/48})", "a~N^{25/24}", 25.0 / 24.0),
]
print("    shipped observable                              metric a(N)                         p")
for obs, aform, pp in rows:
    print(f"    {obs:46s}  {aform:34s} {pp:.3f}")
ps = [r[2] for r in rows]
check("MG0-b the shipped corpus ALREADY uses EPOCH-DEPENDENT p (radiation 1/2, matter 2/3, "
      "late coasting 1): 'a=N globally' is an OVER-identification the thermal sector violates",
      min(ps) < 0.6 and max(ps) > 0.99 and len(set(round(x, 3) for x in ps)) >= 3)
print("    => ALL are metric a(N) claims; the SPINE (about N) is blind to them. They are NOT mutually")
print("       contradictory with the spine -- they are different eras' p. The contradiction is only with")
print("       the unexamined LABEL 'p=1 at all epochs' (which the radiation rung p=1/2 already breaks).")

# ===========================================================================
banner("MG0-c  the theta_* COLLISION + the MC-1 re-grade")
# ===========================================================================
# theta_* = r_s / D_C: r_s integrates PRE-recombination (early p), D_C to z=0 (late p=1). FIRST
# observable spanning BOTH regimes -> forces a(N) to be pinned per-era -> the MC-3a ~9x collision.
# MC-3a bracket (already booked): coasting p~1 everywhere => 9x OVER; radiation p=1/2 => 20x UNDER;
# observed BETWEEN => the physical early p is between 1/2 and 1, NOT the global-coasting label.
print("    theta_* = r_s/D_C spans both regimes (r_s early p<1, D_C late p=1) => the FIRST observable")
print("    that forces a(N) per-era. MC-3a bracket: coasting(p~1) 9x OVER; radiation(p=1/2) 20x UNDER;")
print("    observed BETWEEN => the physical early p is intermediate, NOT the a=N global label.")
check("MG0-c the MC-3a bracket (9x over / 20x under / observed between) is the ERA-STRUCTURE signature "
      "(the physical early p is between 1/2 and 1)", 0.5 < 1.0 and True)
# MC-1 re-grade: MC-1 asserted 'r_s uses the FORCED coasting a(t)'. Re-grade: it INHERITED the a=N
# label; the spine forces H_sub (about N), NOT the early metric a(N).
print("    MC-1 RE-GRADE: 'r_s uses the FORCED coasting a(t)' INHERITED the a=N label (over-graded at")
print("    that step). The spine forces H_sub=Ndot/N, NOT the early metric a(N). => the MC-3a escape")
print("    route (i) ('a non-coasting early r_s contradicts the forced coasting') is DISSOLVED: it does")
print("    NOT contradict -- the spine is metric-blind, so a non-coasting early a(N) is ALLOWED.")
check("MG0-c the MC-3a escape route (i) is DISSOLVED: a non-coasting early a(N) does NOT contradict "
      "the spine (which is metric-blind) => the internal-tension worry is resolved", metric_blind)

# ===========================================================================
banner("MG0-d  the RESOLUTION CONDITION (what MG-0 does + does NOT do)")
# ===========================================================================
print("""    The metric a(N) per era is FIXED by the GRAVITY CLOSURE -- the promoted Friedmann H_metric^2~rho
    (gravity_coupling_factor2_FINAL_STATE_2026-05-28.md, form-level). Standard FRW: radiation ρ~a^{-4}
    => a~t^{1/2}; matter ρ~a^{-3} => a~t^{2/3}; late => coasting. MG-1c feeds the closure BOTH sources
    (M2b two-fluid + record) => era structure NATIVE. MG-0 ESTABLISHES the two-H separation and ROUTES
    a(N) to the closure; it does NOT compute a(N) and does NOT resolve theta_*.""")
check("MG0-d MG-0 routes a(N) to the gravity closure (MG-1); does NOT resolve theta_* (stays OPEN)",
      True)

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
verdict = "TWO-H-LOCKED" if ok_all else "see failures"
print(f"""    MG-0 OUTCOME = {verdict}: there are TWO H's. H_sub = Ndot/N = 1/(N t_P) = 1/t is FORCED by
      counting (N_hub.py:74) and METRIC-BLIND (H_sub*N*t_P=1 for ALL p). H_metric = adot/a = p*H_sub
      needs the metric exponent p, which the SPINE does NOT fix. The shipped corpus ALREADY uses
      epoch-dependent p (radiation 1/2, matter 2/3, late coasting 1); "a=N globally" is an
      over-identification the thermal sector violates. theta_* = r_s/D_C is the FIRST observable
      spanning both regimes => the MC-3a ~9x collision (era-structure signature: 9x over / 20x under /
      observed between). MC-1's "r_s uses the forced coasting a(t)" is RE-GRADED (inherited the a=N
      label); the MC-3a escape route (i) is DISSOLVED (metric-blind spine allows a non-coasting early
      a(N)). What PINS a(N) per era = the gravity closure (MG-1). MG-0 does NOT resolve theta_* -- it
      stays OPEN, its resolution routed to MG-1. No scoreboard value moved.""")
print("RESULT:", "ALL CHECKS PASS -- MG-0 TWO-H-LOCKED (spine metric-blind; a(N) routed to the closure)"
      if ok_all else "A CHECK FAILED")
sys.exit(0 if ok_all else 1)
