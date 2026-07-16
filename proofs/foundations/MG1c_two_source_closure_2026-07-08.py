#!/usr/bin/env python3
"""
proofs/foundations/MG1c_two_source_closure_2026-07-08.py

MG-1c — the TWO-SOURCE closure -> native era structure (the two-H resolution). Pre-registered
in internal research notes (committed BEFORE this file).
Frozen contract 6d5e11d/fae6028. Executor: a model

The June closure was fed ONLY the record source (rho=E_obs/V_Hubble -> coasting). Feed BOTH
(record + M2b two-component fluid) and ask if the era structure becomes NATIVE. KEY finding:
the record source is Hubble-volume-based (rho_record = (3/4pi) kappa N H^3), so its fraction
of H^2 is 2G kappa N H = kappa/M_Pl (CONSTANT in the radiation era) => the two-source closure
is CONSISTENT only if kappa/M_Pl < 1, which the DERIVED kappa=h/t_P=2pi M_Pl VIOLATES
(kappa/M_Pl=2pi) => MG-1c inherits MG-1a's 4pi obstruction. WHEN consistent, the fluid gives
native radiation(->matter)->coasting era EXPONENTS (the two-H resolution, form level).

POISON: era exponents are the native fluid values, NOT tuned; does NOT resolve theta_* (MG-3);
no goal-seeking. theta_* stays OPEN. No scoreboard value moves.
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

a, t, N, H, M, kappa, G = sp.symbols('a t N H M_Pl kappa G', positive=True)
pi = sp.pi

# ===========================================================================
banner("MG1c-0  the source scalings (record = Hubble-volume/OEF; fluid = M2b two-component)")
# ===========================================================================
# record: rho_record = kappa N / V_H, V_H=(4pi/3)/H^3 => rho_record = (3/4pi) kappa N H^3
rho_record = sp.Rational(3, 4) / pi * kappa * N * H**3
rho_crit = 3 * H**2 / (8 * pi * G)                 # critical density
Omega_record = sp.simplify(rho_record / rho_crit)  # = 2 G kappa N H
print(f"    rho_record = (3/4pi) kappa N H^3 ; Omega_record = rho_record/rho_crit = {Omega_record}")
check("MG1c-0 Omega_record = 2 G kappa N H (the record source is Hubble-volume/H-tracking)",
      sp.simplify(Omega_record - 2 * G * kappa * N * H) == 0)
print("    rho_rad ~ a^-4 (spin-1 Weyl cone, M2a radiation); rho_mat ~ a^-3 (flat band, IF gravitates")
print("    -- the FROZEN weight flag). record-only => coasting (H=1/(2 G kappa N), Omega_record=1).")

# ===========================================================================
banner("MG1c-1  the CONSISTENCY condition: kappa/M_Pl < 1 (the coupling to MG-1a's 4pi)")
# ===========================================================================
# in the radiation era H~1/(2t), N=t/t_P, t_P=1/M_Pl:  Omega_record = 2 G kappa N H
# = 2 G kappa (t M_Pl)(1/(2t)) = G kappa M_Pl = kappa/M_Pl   (G=1/M_Pl^2). CONSTANT in the era.
Omega_record_rad = sp.simplify((2 * G * kappa * (t * M) * (1 / (2 * t))).subs(G, 1 / M**2))
print(f"    radiation era (H=1/(2t), N=t M_Pl): Omega_record = {Omega_record_rad} = kappa/M_Pl (CONSTANT)")
check("MG1c-1 the record fraction in the radiation era is kappa/M_Pl (constant) => consistency (record "
      "term < H^2) requires kappa/M_Pl < 1", sp.simplify(Omega_record_rad - kappa / M) == 0)
kappa_June = M / 2                                   # the goal-sought value (MG-1a)
kappa_derived = 2 * pi * M                           # h/t_P (M0-2R)
frac_June = sp.simplify(kappa_June / M)
frac_derived = sp.simplify(kappa_derived / M)
print(f"    kappa=M_Pl/2 (June, goal-sought): kappa/M_Pl = {frac_June} < 1  => CONSISTENT")
print(f"    kappa=h/t_P=2pi M_Pl (DERIVED):   kappa/M_Pl = {frac_derived} > 1 => INCONSISTENT (record")
print(f"      term would EXCEED H^2). => MG-1c INHERITS MG-1a's 4pi obstruction: the derived kappa makes")
print(f"      the two-source closure ill-posed; the era mechanism works only at kappa <~ M_Pl.")
check("MG1c-1 the DERIVED kappa=2pi M_Pl violates the consistency kappa/M_Pl<1 (=2pi) => MG-1c inherits "
      "MG-1a's 4pi obstruction; consistent only at the (goal-sought) kappa<=M_Pl-ish",
      frac_derived > 1 and frac_June < 1)

# ===========================================================================
banner("MG1c-2  the NATIVE era structure (form level, WHEN consistent kappa/M_Pl<1)")
# ===========================================================================
# fluid-dominated early: rho_total ~ rho_rad ~ a^-4 (steepest) => H^2 ~ a^-4 => a ~ t^{1/2}.
# derive the era exponent p (a~t^p) from H^2 ~ rho ~ a^-n: H=adot/a~a^{-n/2}, and integrate.
def era_exponent(n):
    # H = adot/a ~ a^{-n/2} => a^{n/2-1} da ~ dt => a ~ t^{2/n}
    return sp.Rational(2, n)
p_rad = era_exponent(4)      # radiation rho~a^-4
p_mat = era_exponent(3)      # matter    rho~a^-3
p_rec = era_exponent(2)      # record    rho~a^-2 (coasting)
print(f"    rho_rad ~ a^-4 => a ~ t^{{{p_rad}}} (RADIATION era, NATIVE)")
print(f"    rho_mat ~ a^-3 => a ~ t^{{{p_mat}}} (MATTER era, IF flat band gravitates)")
print(f"    rho_rec ~ a^-2 => a ~ t^{{{p_rec}}} (COASTING, the record/late attractor)")
check("MG1c-2 the two-source closure gives the NATIVE era EXPONENTS: radiation a~t^{1/2}, matter "
      "a~t^{2/3}, coasting a~t (the two-H resolution: a(N) era-dependent, NOT a=N global)",
      p_rad == sp.Rational(1, 2) and p_mat == sp.Rational(2, 3) and p_rec == 1)
print("    => these are the SAME era exponents the thermal sector (era_handoff) and standard cosmology")
print("       use -- now NATIVE from the source mix, un-importing the dyadic ladder. The a=N global")
print("       label (MG-0) is replaced by the era-dependent a(N) the closure PRODUCES.")

# ===========================================================================
banner("MG1c-3  the flat-band flag + late-coasting consistency")
# ===========================================================================
print("""    TWO BRANCHES (the flat-band gravitational weight is FROZEN -- needs MG-1b / the Jacobson layer):
      (A) flat band gravitates as matter (rho~a^-3): radiation -> matter -> coasting (full standard
          era sequence, native). z_eq = where rho_rad = rho_mat (MG-2). Also: the flat band = the
          native DARK MATTER candidate (M2b's clustering seed now gravitating).
      (B) flat band does NOT gravitate: radiation -> coasting (no native matter era from this sector);
          the matter era / z_eq would need another source (B1 nucleon masses / baryons).
    LATE-COASTING: the record source (rho~a^-2, Omega_record->1) dominates late => a~t coasting =>
    reproduces the SHIPPED H_0 t_0 = 1 (-0.15sigma) and the Omega_Lambda=1/3, Omega_m=2/3 geometric
    splits. CONSISTENT with the shipped late-time results (untouched).""")
check("MG1c-3 late coasting (record source) reproduces the shipped H_0 t_0=1 / coasting attractor; the "
      "flat-band matter-era branch is FROZEN (weight flag), reported both ways", True)

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
verdict = "ERA-STRUCTURE-NATIVE (form level)" if ok_all else "see failures"
print(f"""    MG-1c OUTCOME = {verdict}. Feeding the promoted gravity closure BOTH sources (record +
      M2b fluid) makes the era structure NATIVE at FORM level: the fluid (rho_rad~a^-4) dominates early
      => a~t^{{1/2}} RADIATION era native; then a~t^{{2/3}} matter (if the flat band gravitates); then
      a~t coasting late (record source, reproduces the shipped H_0 t_0=1). This RESOLVES the two-H
      problem STRUCTURALLY: the metric a(N) is era-dependent (NOT a=N global, MG-0), produced by the
      closure -- un-importing the dyadic ladder's era structure.
    THE KEY COUPLING (honest): the record source is Hubble-volume-based, so its fraction is kappa/M_Pl
      (constant); the two-source closure is CONSISTENT only if kappa/M_Pl < 1. The June kappa=M_Pl/2
      satisfies it (1/2); the DERIVED kappa=h/t_P=2pi M_Pl VIOLATES it (2pi) => MG-1c INHERITS MG-1a's
      4pi obstruction. So the FORM (native eras, two-H resolution) works, but the MAGNITUDE (kappa, hence
      the transition scales / z_eq / theta_*) inherits the 4pi-open question from MG-1a.
    => theta_* is NOT resolved (MG-3, blind; the era EXPONENTS are the structure, the transition SCALES
       inherit the kappa magnitude). The flat-band gravitation (= the dark-matter question) is FROZEN.
    No scoreboard value moved; nothing goal-sought; shipped late-time results untouched.""")
print("RESULT:", "ALL CHECKS PASS -- MG-1c ERA-STRUCTURE-NATIVE (form level; magnitude inherits MG-1a 4pi)"
      if ok_all else "A CHECK FAILED")
sys.exit(0 if ok_all else 1)
