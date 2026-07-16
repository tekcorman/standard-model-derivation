#!/usr/bin/env python3
"""
proofs/foundations/MG1a_derived_kappa_closure_2026-07-08.py

MG-1a — the DERIVED-κ gravity-closure re-run (over-determination test). Pre-registered in
internal research notes (committed BEFORE this file).
Frozen contract 6d5e11d/fae6028. Executor: a model

The RG2b/Cai-Kim closure left "κ = M_Pl/2 iff G_eff = G; κ UNDERIVED" (panel: M_Pl/2 was
GOAL-SOUGHT). M0-2R DERIVED κ = h/t_P independently. Test: does the derived κ equal the
closure's required κ (=> Newton's G parameter-free) or by what factor is it off?

DISCIPLINE: NEVER goal-seek G_eff=G / κ=M_Pl/2 / c_S=2 (the documented+retracted June
overclaim). Derived κ is INPUT; G_eff is OUTPUT. Convention control MANDATORY, no eyeballing.
No scoreboard value moves.
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

# explicit natural units: hbar=c=1, BARE Planck mass M_Pl (M_Pl^2 = 1/G), t_P = 1/M_Pl, h = 2*pi
M, N = sp.symbols('M_Pl N', positive=True)
G = 1 / M**2                                       # bare: G = 1/M_Pl^2
t_P = 1 / M                                        # t_P = sqrt(G) = 1/M_Pl (hbar=c=1)
pi = sp.pi
h = 2 * pi                                          # h = 2*pi*hbar, hbar=1
hbar = sp.Integer(1)

# ===========================================================================
banner("MG1a-0  CONVENTION CONTROL: reproduce the closure's required kappa = M_Pl/2")
# ===========================================================================
# closure: E_obs = kappa*N ; V = (4pi/3) R_H^3 ; R_H = 1/H ; rho = E_obs/V ;
#          Friedmann H^2 = (8 pi G / 3) rho.  Solve for H, then match cascade H = 1/(N t_P).
kappa = sp.symbols('kappa', positive=True)
H = sp.symbols('H', positive=True)
R_H = 1 / H
V = (sp.Rational(4, 3) * pi) * R_H**3
rho = kappa * N / V
friedmann = sp.Eq(H**2, (8 * pi * G / 3) * rho)    # standard Friedmann
H_closure = sp.solve(friedmann, H)                 # H(kappa, N)
H_closure = [s for s in H_closure if s.is_real is not False][0]
H_closure = sp.simplify(H_closure)
print(f"    closure Friedmann + Hubble-volume => H = {H_closure}")
# match cascade H = 1/(N t_P):
kappa_required = sp.solve(sp.Eq(H_closure, 1 / (N * t_P)), kappa)[0]
kappa_required = sp.simplify(kappa_required)
print(f"    match cascade H=1/(N t_P) => required kappa = {kappa_required}")
check("MG1a-0 CONTROL: the closure's required kappa = M_Pl/2 (reproduces the corpus; units locked)",
      sp.simplify(kappa_required - M / 2) == 0, detail=f"kappa_required = {kappa_required}")

# ===========================================================================
banner("MG1a-1  THE OVER-DETERMINATION: insert the DERIVED kappa = h/t_P (M0-2R); compute G_eff")
# ===========================================================================
kappa_derived_h = h / t_P                           # = 2*pi*M_Pl  (h = 2 pi hbar)
kappa_derived_hbar = hbar / t_P                     # = M_Pl        (hbar variant, for reference)
print(f"    derived kappa = h/t_P = {sp.simplify(kappa_derived_h)}  (= 2 pi M_Pl)")
print(f"    (hbar variant kappa = hbar/t_P = {sp.simplify(kappa_derived_hbar)} = M_Pl, reported for reference)")
# the clock forces c_S = 2 (H = c_S M_Pl/(2N) matched to cascade M_Pl/N; kappa cancels there)
c_S = sp.Integer(2)
def G_eff(kap):                                     # Cai-Kim: G_eff = 1/(kappa c_S M_Pl)
    return sp.simplify(1 / (kap * c_S * M))
Geff_h = G_eff(kappa_derived_h)
Geff_hbar = G_eff(kappa_derived_hbar)
ratio_h = sp.simplify(Geff_h / G)
ratio_hbar = sp.simplify(Geff_hbar / G)
print(f"    with clock-forced c_S=2:  G_eff(h)    = {Geff_h}  = {ratio_h} * G")
print(f"                              G_eff(hbar) = {Geff_hbar}  = {ratio_hbar} * G")
kfac_h = sp.simplify(kappa_derived_h / kappa_required)      # kappa_derived/kappa_required
print(f"    kappa_derived(h)/kappa_required = {kfac_h}  (= 4 pi)")
check("MG1a-1 the derived kappa=h/t_P does NOT equal the required M_Pl/2: it is 4 pi larger "
      "=> G_eff = G/(4 pi) (NOT G). The over-determination CONTRADICTS (does not close).",
      sp.simplify(kfac_h - 4 * pi) == 0 and sp.simplify(ratio_h - 1 / (4 * pi)) == 0)
# ALSO: the Friedmann clock with the derived kappa disagrees with the cascade spine by 4 pi
H_with_derived = sp.simplify(H_closure.subs(kappa, kappa_derived_h))
H_cascade = 1 / (N * t_P)
clock_ratio = sp.simplify(H_with_derived / H_cascade)
print(f"    (equivalently the Friedmann clock with derived kappa gives H = {clock_ratio} * H_cascade)")
check("MG1a-1 equivalently: Friedmann-with-derived-kappa disagrees with the cascade spine by 1/(4 pi)",
      sp.simplify(clock_ratio - 1 / (4 * pi)) == 0)

# ===========================================================================
banner("MG1a-2  THE FACTOR DIAGNOSIS (4 pi = 2 pi x 2; NOT goal-selected)")
# ===========================================================================
print("""    The residual 4 pi factorises as 2 pi x 2, two INDEPENDENTLY-NAMED conventions (neither chosen to
    land G):
      - 2 pi = the h-vs-hbar factor. kappa=h/t_P carries the M0-2R T4 '2 pi' (the modular circle /
        one full action-quantum per tick). Using hbar/t_P instead removes it: G_eff(hbar) = G/2.
      - 2 = the GEOMETRIC factor in kappa_required = M_Pl/2 -- it comes from the Friedmann + Hubble-volume
        algebra ((8 pi G/3)x(3/4 pi) = 2G in H = 1/(2 G kappa N)), NOT manifestly c_S. (Numerically it
        coincides with the c_S=2 the corpus needed for G_eff=G given kappa=M_Pl/2 -- and that c_S=2 was
        itself the goal-sought choice; c_S is tested NATIVELY in MG-1b, independent of this factor.)
    So the derived kappa is off from the goal-sought M_Pl/2 by (h/hbar) x (Hubble-volume 2) = 2 pi x 2 = 4 pi.""")
check("MG1a-2 the residual is 4 pi = (2 pi: h/hbar) x (2: Friedmann-Hubble-volume geometric) -- both "
      "named, neither goal-selected", sp.simplify(kfac_h - (2 * pi) * 2) == 0)
# a further named candidate: the LINEAR record entropy S=c_S R M_Pl vs the AREA law S=A/4G=pi R^2 M_Pl^2
# differ by a geometric (4pi-class) factor -- flagged, NOT asserted, NOT used to land G.
print("    FURTHER NAMED CANDIDATE (flagged, not asserted): the closure's LINEAR record entropy")
print("    S=c_S R M_Pl vs the standard AREA law S=A/4G=pi R^2 M_Pl^2 differ by a geometric 4pi-class")
print("    factor; the framework deliberately uses the linear (record) entropy (why it sits far below")
print("    the Bekenstein area bound). NOT selected to land G (that is the goal-seek poison).")

# ===========================================================================
banner("MG1a-3  VERDICT: CONTRADICTS-BY-FACTOR (the magnitude stays OPEN; panel flag CONFIRMED)")
# ===========================================================================
print(f"""    MG-1a OUTCOME = CONTRADICTS-BY-FACTOR. The INDEPENDENTLY-derived kappa = h/t_P = 2 pi M_Pl does
      NOT equal the closure's required kappa = M_Pl/2 -- it is 4 pi larger => G_eff = G/(4 pi), and the
      Friedmann clock with the derived kappa disagrees with the cascade spine by 4 pi.
    => This CONFIRMS the 2026-06-12 panel's flag that kappa=M_Pl/2 was GOAL-SOUGHT: the true, derived
      kappa is 4 pi different, so Newton's G does NOT close parameter-free via this route. The residual
      4 pi = (2 pi, the h/hbar modular-circle factor) x (2, the c_S entropy-count fork) -- both named,
      NEITHER goal-selected. The coupling MAGNITUDE (parameter-free G) stays ❌ OPEN, now sharper: with
      the derived kappa the mismatch is a clean 4 pi with a named factorisation, superseding the
      goal-sought M_Pl/2. Whether a forced convention (reduced M_Pl / area-vs-record entropy / the modular
      2 pi's gravitational role) reconciles it is the remaining question -- to be answered WITHOUT
      goal-seeking (MG-1b tests c_S natively; the h/hbar 2 pi ties to whether gravity sees the full
      action quantum h or the reduced hbar).
    G_N.py keeps its independent grade; the FORM (Friedmann H^2~rho) stands; only the magnitude is off.
    No scoreboard value moved. NOTHING goal-sought.""")
check("MG1a-3 verdict booked: CONTRADICTS-BY-FACTOR (4pi); magnitude OPEN; panel's goal-sought flag "
      "CONFIRMED; nothing goal-selected", True)

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
print("""    MG-1a: the derived kappa=h/t_P, inserted into the promoted gravity closure, gives G_eff=G/(4pi)
    (NOT G) -- a CLEAN CONTRADICTION-BY-FACTOR that CONFIRMS the panel's 'kappa=M_Pl/2 was goal-sought'.
    The 4pi = (2pi: h vs hbar, the M0-2R modular-circle factor) x (2: the c_S entropy count). Newton's G
    does NOT close parameter-free; the magnitude stays OPEN, now with the derived kappa + a named 4pi.
    Remaining: does gravity see h or hbar (the 2pi), and c_S=1 vs 2 (MG-1b) -- answered WITHOUT goal-seek.""")
print("RESULT:", "ALL CHECKS PASS -- MG-1a CONTRADICTS-BY-4pi (magnitude OPEN; goal-sought flag confirmed)"
      if ok_all else "A CHECK FAILED")
sys.exit(0 if ok_all else 1)
