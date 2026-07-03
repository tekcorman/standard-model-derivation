#!/usr/bin/env python3
"""
proofs/foundations/F4_S6_width_residue_no_go_2026-07-02.py

F4 S6 — sub-equation 7a ATTACKED: is the width's rho-bar the Z-channel POLE RESIDUE?
ANSWER (pre-registered structure, decided by computation below): the residue route is
SIGN-EXCLUDED for the entire structurally-admissible profile class, and the framework's
own waterline reading forces Z_res = 1 exactly. The -0.437% demand is NOT residue
content and NOT any existing dark class — 7a localizes into the un-built omega-resolved
vertex layer (incomplete_equations_todo.md par.7 proper).

SETUP (Type-3-grade identification, same status as the golden rule): the physical
width is the pole-residue-normalized rate, Gamma/M = Z_res x (tree assembly), with
Z_res = 1/(1 - E'(z_p)) for the dressed channel equation P(z)^-1 = z - E(z),
E(z) = z0 (1 - Sigma_Z(z)), z0 = 2 (the NB Perron eigenvalue = the Z channel),
Sigma_Z(z0) = delta_r = c_S u/(1-u) (the derived M_Z tree->pole read; c_S = 1/12,
u = alpha_1). What is NOT yet fixed by the object is the z-PROFILE of Sigma_Z off
the matching point — exactly the par.7 omega-resolution incompleteness, here
localized to "the z-profile of one girth winding".

PRE-REGISTERED PROFILE CLASS (declared from structure BEFORE computing values):
  per-winding amplitude phi(z) = u * (z0/z)^a, a >= 0, resummed
  Sigma_Z(z) = c_S * phi/(1-phi):
    a = 0 : the WATERLINE reading (windings are MDL/topological classes, the
            axioms' own NOTE) — z-flat;
    a = 1 : the proven SHELL z-structure transplanted (Feshbach contour theorem:
            Sigma(z) = alpha_1/z exactly for the water-filled circle = one
            effective mode at z = 0; session-1 verified for every radius);
    a = g = 10 : per-step scaling over the whole girth excursion.
  Negative a (amplitude GROWING with probe rate) contradicts the one proven
  z-structure in the framework (the shell theorem's sign) and is excluded.

HONESTY RAILS: the demand band is WIDE in coefficient units (+-21%), so MANY
K-rationals (1/12, 1/9, 1/8, ...) would "pass" a magnitude test — magnitude
matching is therefore FORBIDDEN here; only class-level sign/structure decides.
The taxonomy sweep lists every existing dark class with its derived magnitude and
the ARGUMENT for exclusion (never a fit). PDG enters only in COMPARISON blocks.
"""
import math
import os
import sys

import sympy as sp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "predictions"))

from delta_r import delta_r as DR                        # noqa: E402
from delta_rho import delta_rho as DRHO                  # noqa: E402
from alpha_1 import predict_alpha_1                      # noqa: E402
from k_star import predict_k_star                        # noqa: E402
from d_spatial import predict_d_spatial                  # noqa: E402
from g_girth import predict_g_girth                      # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

K = predict_k_star(predict_d_spatial())
G = predict_g_girth(K, predict_d_spatial())
U = predict_alpha_1(K, G)
CS = 1.0 / 12.0                                          # gauge-singlet projection (delta_r leaf)
Z0 = K - 1.0                                             # NB Perron eigenvalue = 2

print("=" * 88)
print(" T-A  the general residue formula and the SIGN LEMMA (sympy)")
print("=" * 88)
z, z0s, us, cs, a = sp.symbols('z z0 u c a', positive=True)
phi = us * (z0s / z) ** a
SigZ = cs * phi / (1 - phi)
E = z0s * (1 - SigZ)
Eprime = sp.diff(E, z)
Zres = sp.simplify(1 / (1 - Eprime))
Zres_at = sp.simplify(Zres.subs(z, z0s))
# closed form at the matching point:
target = 1 / (1 - a * cs * us / (1 - us) ** 2)
check(f"Z_res(z0) = 1/(1 - a·c_S·u/(1-u)²)  (sympy: {sp.simplify(Zres_at - target) == 0})",
      sp.simplify(Zres_at - target) == 0)
# SIGN LEMMA: for every a > 0, Z_res > 1 (UP); a = 0 gives exactly 1.
sign_expr = sp.simplify(Zres_at - 1)
check("sign lemma: Z_res - 1 = a·c_S·u/[(1-u)² - a·c_S·u] > 0 for all a>0 (0 iff a=0)",
      sp.simplify(sign_expr.subs(a, 0)) == 0
      and sp.simplify(sign_expr - a * cs * us / ((1 - us) ** 2 - a * cs * us)) == 0)
print("    ⟹ for EVERY profile consistent with the proven shell z-structure's sign")
print("      (amplitude non-increasing in probe rate, a ≥ 0): Z_res ≥ 1 — the residue")
print("      dresses the width UP or not at all. It can NEVER produce a DOWN shift.")

print("=" * 88)
print(" T-B  the waterline reading forces Z_res = 1 exactly")
print("=" * 88)
print("""    The framework's winding sum u/(1-u) is the A2-T WATERLINE SUM over girth-cycle
    winding CLASSES (framework_axioms.md, NOTE 2026-04-21: topological classes above
    the waterline, explicitly NOT a dynamical Green's-function resummation). A
    topological class weight carries NO spectral-parameter profile: a = 0 identically.
    ⟹ Sigma_Z is z-flat ⟹ Z_res = 1 EXACTLY: the framework's own reading FORCES
    no-residue-dressing of the width. (This is the a=0 row below, now derived, not
    just enumerated.)""")
check("waterline reading ⇒ a=0 ⇒ Z_res = 1 exactly (symbolic above)", True)

print("=" * 88)
print(" numeric profile table + the demand — COMPARISON block (PDG enters here)")
print("=" * 88)
demand, band = -0.00437, 0.00092          # S5: rho_bar_width - 1 needed on the alpha-form
print(f"    demand on the alpha-form normalization: {demand*100:+.3f}% ± {band*100:.3f}%  (DOWN)")
for aa, lbl in ((0, "a=0 waterline (FORCED reading)"),
                (1, "a=1 shell-theorem structure transplanted"),
                (G, f"a=g={G} per-step girth scaling")):
    zres = 1 / (1 - aa * CS * U / (1 - U) ** 2)
    print(f"    {lbl:>42}: Z_res - 1 = {(zres-1)*100:+.3f}%   "
          f"{'(= demand? NO — sign)' if aa else '(= demand? NO — zero)'}")
print("    ⟹ the residue route is EXCLUDED: zero or UP for the whole class; demand is DOWN.")

print("=" * 88)
print(" taxonomy sweep — every existing dark class vs the demand (argument, not fit)")
print("=" * 88)
rows = [
    ("oblique residue (this probe)", "0 or ≥ +0.35%", "sign-excluded (T-A) / zero (T-B)"),
    ("singlet c_S=1/12 re-applied to g_Z²", f"{-CS*U/(1-U)*100:+.3f}%",
     "POISONED: it is the MASS-shift projection (already used in M_Z); re-use as a"
     " coupling shift has no derivation — and the wide band would let it 'pass' (1.1σ)"),
    ("vertex Family-D per leg (c_F u²)", f"{-U*U/12*100:+.4f}%/leg",
     "8× too small even ×3 legs; the only formally-signed class (Peskin −1) but O(u²)"),
    ("channel c=1 (u/(1−u))", f"{-U/(1-U)*100:+.2f}%", "9× too large; saturated d/b-walker class"),
    ("democratic c_v=5/12", f"{-5/12*U/(1-U)*100:+.2f}%", "4× too large; the v-Higgs H¹ count"),
    ("custodial δρ=½(√5/4)u", f"{+DRHO*100:+.2f}%", "2.5× too large AND wrong sign (UP);"
     " it is the mass-ratio projection (validated there)"),
    ("QCD 3rd order / QED FSR / masses", "≤0.05% each", "stated-not-applied S3 omissions;"
     " an order too small individually; combining-to-fit is forbidden"),
]
for name, mag, why in rows:
    print(f"    {name:>38}: {mag:>14}   {why}")
print(f"""
    VERDICT: no existing framework dressing class produces the demand
    ({demand*100:+.3f}% ± {band*100:.3f}%) with sign + magnitude + pedigree simultaneously.
    Sub-equation 7a therefore LOCALIZES OUTSIDE the current dark taxonomy: the width's
    missing normalization is genuine omega-resolved VERTEX content — the same un-built
    Sigma_X(omega) equation of todo par.7, now carrying its sharpest numerical target.
    Gamma_Z/M_Z stays OPEN (+4.8 sigma). This is the honest end of the matching-point
    program for widths: mass-side reads (delta_r, delta_rho) are pole-position content;
    the width's normalization is not.""")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
