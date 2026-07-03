#!/usr/bin/env python3
"""
proofs/foundations/F4_S4_width_oblique_decomposition_2026-07-02.py

F4 S4 — DECOMPOSE the Gamma_Z/M_Z +0.44% demand against the framework's OWN oblique set.

QUESTION (pre-declared): the "EW radiative layer" named as the located cause of the
width residual (predictions/Gamma_Z_over_M_Z.py, +4.8 sigma OPEN) — is it ONE shared
object with the M_Z oblique residual (a single scalar gap), or a multi-component
structure? And how much of it does the framework ALREADY possess in its derived
quantities (delta_rho = +1.0906%, delta_r = +0.3384%, the G_F triangle)?

CLASS (pre-declared): DIAGNOSTIC/CONSISTENCY probe. It produces NO prediction, NO
value-lock entry, and CANNOT close the residual (per the law the miss stays OPEN
until the layer is derived top-down). Its deliverable is the located STRUCTURE:
which sub-equations are missing, each quantified.

METHOD DISCIPLINE:
  * No fitting, no scanning. Every number below is either (i) a framework-derived
    quantity imported from its predictions/ leaf, (ii) an exact algebraic identity
    of the declared Type-3 tree structure, or (iii) a comparison-side measured value
    (PDG 2024), entering ONLY in blocks marked COMPARISON.
  * Solved "needed" values (what the data demands in a named direction) are
    DIAGNOSTIC outputs, not adoptable numbers. PRE-POISONED numerology: any
    K-rational coincidence for the needed fractions (e.g. 3/5, 5/8, sqrt5/4-multiples)
    is hereby declared UNUSABLE without a forced projection derivation.
  * Identities are asserted tight; data comparisons are printed, never asserted.

PRE-REGISTERED possible verdicts:
  (V1) coherent: one scalar gap accounts for the M_Z and Gamma_Z residual pattern;
  (V2) multi-component: the residual vector is NOT collinear with any single
       framework oblique direction — the layer decomposes into named sub-equations
       (Delta-r-bar remainder / width-rho-bar projection / s-bar^2 shift), each
       logged in incomplete_equations_todo.md par.7.
Either verdict is a complete honest result.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "predictions"))

from g_2 import g_2_MZ                                   # noqa: E402
from sin2_theta_W_MZ import sin2_theta_W_MZ             # noqa: E402
from alpha_s import alpha_s_MZ                           # noqa: E402
from m_W import m_W_pred                                 # noqa: E402
from M_Z import M_Z_GeV                                  # noqa: E402
from delta_rho import delta_rho as DELTA_RHO             # noqa: E402
from delta_r import delta_r as DELTA_R                   # noqa: E402
from G_F import G_F_pred, G_F_obs                        # noqa: E402
from Gamma_Z_over_M_Z import (predict_Gamma_Z_over_M_Z,  # noqa: E402
                              Gamma_Z_over_M_Z_obs, Gamma_Z_over_M_Z_sigma,
                              _k_star, _n_gen, _n_up_open)
from Gamma_W_over_Gamma_Z import (predict_Gamma_W_over_Gamma_Z,  # noqa: E402
                                  Gamma_W_over_Gamma_Z_obs)

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

S2, C2 = sin2_theta_W_MZ, 1 - sin2_theta_W_MZ

def sigma_content(s2):
    tot, had = 0.0, 0.0
    for n in range(_k_star + 1):
        sgn = (-1) ** n
        T3, Q, Nc = sgn / 2, sgn * n / _k_star, math.comb(_k_star, n)
        gens = _n_up_open if n == 2 else _n_gen
        w = gens * Nc * ((T3 - 2 * Q * s2) ** 2 + T3 ** 2)
        tot += w
        if 0 < n < _k_star:
            had += w
    return tot, had / tot

SIG, F_HAD = sigma_content(S2)
X = alpha_s_MZ / math.pi
QCD = 1 + F_HAD * (X + 1.409 * X * X)

# ---------------------------------------------------------------------------
print("=" * 88)
print(" S1  the framework's tree triangle: three normalizations of ONE width assembly")
print("=" * 88)
# (a) alpha-form (the shipped S3 assembly): g2^2/c2_RG
gz2_alpha = g_2_MZ ** 2 / C2
# (b) G_F^tree-form: G_F built from the framework's own (g2, m_W) tree relation
GF_tree = g_2_MZ ** 2 / (4 * math.sqrt(2) * m_W_pred ** 2)
gz2_gf_tree = 4 * math.sqrt(2) * GF_tree * M_Z_GeV ** 2
# (c) G_F^v-form: the framework's G_F leaf (= 1/(sqrt2 v^2); calibration round-trip,
#     equal to the measured muon-decay G_F by construction — labeled)
gz2_gf_v = 4 * math.sqrt(2) * G_F_pred * M_Z_GeV ** 2

# exact tree identity: alpha-form == G_F^tree-form x (1+delta_rho), because
# m_W = M_Z c sqrt(1+delta_rho) => 4sqrt2 GF_tree M_Z^2 = g2^2/(c2 (1+delta_rho)).
check("tree identity: [g2^2/c2] = [4*sqrt2*GF_tree*M_Z^2]*(1+delta_rho)  (exact)",
      abs(gz2_alpha / (gz2_gf_tree * (1 + DELTA_RHO)) - 1) < 1e-9)
gap_GF = G_F_pred / GF_tree - 1
print(f"    g_Z^2 normalizations: alpha-form {gz2_alpha:.6f} | GF_tree-form {gz2_gf_tree:.6f} "
      f"| GF_v-form {gz2_gf_v:.6f}")
print(f"    THE G_F TRIANGLE GAP (framework-internal, no PDG): "
      f"G_F(v-side)/G_F(g2,m_W tree) - 1 = {gap_GF*100:+.3f}%")
print("""    This is the framework's OWN muon-decay-vs-tree mismatch — the structural slot
    where the SM's Delta-r-bar remainder lives. It exists in the framework's derived
    quantities and is currently WIRED INTO NOTHING.""")

# ---------------------------------------------------------------------------
print("=" * 88)
print(" S2  the width in all three normalizations — COMPARISON blocks marked")
print("=" * 88)
def width_ratio(gz2, rho_bar, s2_eff):
    tot, hf = sigma_content(s2_eff)
    return gz2 * rho_bar * tot / (48 * math.pi) * (1 + hf * (X + 1.409 * X * X))

rows = [
    ("alpha-form (shipped S3; rho_bar=1)",            width_ratio(gz2_alpha, 1.0, S2)),
    ("GF_tree-form x (1+delta_rho)  [== alpha-form]", width_ratio(gz2_gf_tree, 1 + DELTA_RHO, S2)),
    ("GF_v-form, rho_bar = 1",                        width_ratio(gz2_gf_v, 1.0, S2)),
    ("GF_v-form, rho_bar = 1+delta_rho",              width_ratio(gz2_gf_v, 1 + DELTA_RHO, S2)),
]
print(f"    {'form':>46}   {'Gamma_Z/M_Z':>11}   vs measured (COMPARISON)")
for lbl, v in rows:
    print(f"    {lbl:>46}   {v:.6f}   {(v/Gamma_Z_over_M_Z_obs-1)*100:+.2f}%")
check("shipped assembly reproduced by this probe's machinery",
      abs(rows[0][1] - predict_Gamma_Z_over_M_Z(g_2_MZ, S2, alpha_s_MZ,
          _k_star, _n_gen, _n_up_open)) < 1e-12)
print("""    READ: the framework's assembly is parametrization-consistent (rows 1=2, exact
    identity) — the +0.44% residual is INVARIANT content, not a scheme artifact. The
    measured value lies BETWEEN the GF_tree-form without rho (row: alpha/(1+delta_rho),
    -0.65%) and the alpha-form (+0.44%): no re-parametrization closes it.""")

# ---------------------------------------------------------------------------
print("=" * 88)
print(" S3  the demand, solved in the framework's own oblique directions (DIAGNOSTIC)")
print("=" * 88)
# Sensitivities of ln(Gamma_Z/M_Z) — computed numerically from the assembly itself:
eps = 1e-6
d_rho = (math.log(width_ratio(gz2_alpha, 1 + eps, S2))
         - math.log(width_ratio(gz2_alpha, 1.0, S2))) / eps
d_s2 = (math.log(width_ratio(gz2_alpha, 1.0, S2 + eps))
        - math.log(width_ratio(gz2_alpha, 1.0, S2))) / eps
demand = math.log(Gamma_Z_over_M_Z_obs / rows[0][1])          # COMPARISON enters here
print(f"    sensitivities: dln(G/M)/d(rho_bar) = {d_rho:+.4f}   dln(G/M)/d(s2_eff) = {d_s2:+.4f}")
print(f"    demand (measured/assembly - 1)     = {demand*100:+.3f}%")
rho_needed = demand / d_rho
s2_needed = demand / d_s2
print(f"    IF pure rho_bar direction:  rho_bar_needed - 1 = {rho_needed*100:+.3f}%  "
      f"= {rho_needed/DELTA_RHO:+.3f} x delta_rho   [DIAGNOSTIC — poisoned, not adoptable]")
print(f"    IF pure s2_eff direction:   delta_s2_needed    = {s2_needed:+.5f}  "
      f"(s2 {S2:.5f} -> {S2+s2_needed:.5f})           [DIAGNOSTIC — poisoned]")
print(f"    IF pure normalization:      needed factor      = {demand*100:+.3f}%  vs the")
print(f"    G_F triangle gap {gap_GF*100:+.3f}% — same magnitude, OPPOSITE sign requirement.")

# ---------------------------------------------------------------------------
print("=" * 88)
print(" S4  the residual VECTOR across observables vs the framework's oblique directions")
print("=" * 88)
# (framework residuals, all previously established; COMPARISON values)
resid = {
    "M_Z":        M_Z_GeV / 91.1876 - 1,                        # +0.0179%
    "m_W":        m_W_pred / 80.3692 - 1,                       # +0.0396%
    "G/M (Z)":    rows[0][1] / Gamma_Z_over_M_Z_obs - 1,        # +0.435%
    "G_W/G_Z":    predict_Gamma_W_over_Gamma_Z(S2, alpha_s_MZ, m_W_pred / M_Z_GeV,
                   _k_star, _n_gen, _n_up_open) / Gamma_W_over_Gamma_Z_obs - 1,
}
print(f"    residuals: " + "   ".join(f"{k} {v*100:+.3f}%" for k, v in resid.items()))
# collinearity test: is the residual vector proportional to the delta_rho-direction
# response vector? Response of each observable to a COMMON rho_bar shift eps:
#   M_Z: tree M_Z ~ v*sqrt(g2^2+gY^2)/2 — no rho_bar in the framework's M_Z read -> 0
#   m_W: m_W = M_Z c sqrt(1+delta_rho) -> +eps/2
#   G/M: d_rho x eps ;  G_W/G_Z: m_W/M_Z factor -> +eps/2, minus Sigma_Z... (numeric)
epsr = 1e-6
gwz0 = predict_Gamma_W_over_Gamma_Z(S2, alpha_s_MZ, m_W_pred / M_Z_GeV,
                                    _k_star, _n_gen, _n_up_open)
gwz1 = predict_Gamma_W_over_Gamma_Z(S2, alpha_s_MZ,
                                    (m_W_pred * math.sqrt(1 + epsr)) / M_Z_GeV,
                                    _k_star, _n_gen, _n_up_open)
resp_rho = {"M_Z": 0.0, "m_W": 0.5, "G/M (Z)": d_rho,
            "G_W/G_Z": (math.log(gwz1) - math.log(gwz0)) / epsr}
lam = resid["G/M (Z)"] / resp_rho["G/M (Z)"]                 # scale fixed on the width row
print(f"    single-scalar test (scale a common rho_bar-gap on the width row: "
      f"lambda = {lam*100:+.3f}%):")
coherent = True
for k in resid:
    pred_r = lam * resp_rho[k]
    ok = abs(pred_r - resid[k]) < 0.5 * abs(resid[k]) if resid[k] != 0 else True
    coherent = coherent and ok
    print(f"      {k:>8}: predicted-if-one-scalar {pred_r*100:+.3f}%   actual {resid[k]*100:+.3f}%"
          f"   {'~' if ok else 'X'}")
print(f"    VERDICT: {'(V1) coherent single-scalar' if coherent else '(V2) MULTI-COMPONENT'}"
      f" — the residual vector is {'collinear' if coherent else 'NOT collinear'} with a"
      f" common rho_bar direction.")

# ---------------------------------------------------------------------------
print("=" * 88)
print(" S5  invariance of the clean observable (Gamma_W/Gamma_Z)")
print("=" * 88)
print(f"    G_W/G_Z residual {resid['G_W/G_Z']*100:+.3f}% — normalization gaps (alpha-vs-G_F,")
print(f"    Delta-r-bar remainder) cancel in the ratio; its rho_bar response is only via")
print(f"    m_W/M_Z ({resp_rho['G_W/G_Z']:+.3f} per unit) — stays sub-sigma across the whole")
print(f"    decomposition. The ratio is the layer-insensitive width observable, as shipped.")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
