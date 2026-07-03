#!/usr/bin/env python3
"""
proofs/foundations/F4_S5_GF_triangle_identity_2026-07-02.py

F4 S5 — THE G_F TRIANGLE GAP IS AN IDENTITY, NOT A NEW OBJECT (chase-the-math-up).

CLAIM (T1, to be proven symbolically then verified on the live leaves): within the
framework's own EW chain,

    M_Z  = sqrt(pi) * v * sqrt(alpha_2 + alpha_Y) * (1 - delta_r)      [M_Z.py]
    m_W  = M_Z * c * sqrt(1 + delta_rho),  c^2 = alpha_2/(alpha_2+alpha_Y)
                                                                       [m_W.py, s^2 def]
    g_2^2 = 4*pi*alpha_2                                               [g_2.py]
    G_F^v = 1/(sqrt2 * v^2)                                            [G_F.py]
    G_F^tree = g_2^2/(4*sqrt2*m_W^2)                                   [tree relation]

the muon-decay-vs-tree mismatch is EXACTLY

    G_F^v / G_F^tree = (1 - delta_r)^2 * (1 + delta_rho)               (T1)

i.e. the S4 probe's "+0.410% triangle gap, wired into nothing" is WRONGLY framed:
it is the ALGEBRAIC IMAGE of the already-derived oblique pair (delta_rho, delta_r),
= delta_rho - 2*delta_r + O(delta^2) = +0.408%. Sub-equation 7b RETIRES. The S4
poison entry "gap ~ (3/8)*delta_rho" is thereby EXPLAINED: the true object is
delta_rho - 2*delta_r; the 3/8 was accidental.

COROLLARY (T2): the width's alpha-form equals its G_F^v-form divided by (1-delta_r)^2
exactly — so the ONE remaining open direction (sub-equation 7a) is the width's
G_F^v-parametrized dressing: data demands rho_bar_width - 1 = +0.24% +- 0.09%
(COMPARISON side). ☠ PRE-POISONED candidate dressings (do NOT select by fit):
{1, (1+delta_r), (1+delta_r)^2, (1+2 delta_r), (1+delta_rho), delta-combinations}
— the forced question is WHICH normalization the substrate's Z-pole residue/vertex
carries, to be DERIVED from the resolvent or left open.

CLASS: structural identity (algebra of existing derived reads) + diagnostic
restatement. NO closure: Gamma_Z/M_Z stays OPEN (+4.8 sigma).
"""
import math
import os
import sys

import sympy as sp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "predictions"))

from g_2 import g_2_MZ                                   # noqa: E402
from sin2_theta_W_MZ import sin2_theta_W_MZ             # noqa: E402
from m_W import m_W_pred                                 # noqa: E402
from M_Z import M_Z_GeV                                  # noqa: E402
from v_higgs import v_pred                               # noqa: E402
from delta_rho import delta_rho as DRHO                  # noqa: E402
from delta_r import delta_r as DR                        # noqa: E402
from G_F import G_F_pred                                 # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

print("=" * 88)
print(" T1  symbolic proof (sympy): G_F^v / G_F^tree == (1-delta_r)^2 (1+delta_rho)")
print("=" * 88)
v, a2, aY, dr, drho = sp.symbols('v alpha_2 alpha_Y delta_r delta_rho', positive=True)
MZ = sp.sqrt(sp.pi) * v * sp.sqrt(a2 + aY) * (1 - dr)
c2 = a2 / (a2 + aY)
mW = MZ * sp.sqrt(c2) * sp.sqrt(1 + drho)
g22 = 4 * sp.pi * a2
GF_tree = g22 / (4 * sp.sqrt(2) * mW ** 2)
GF_v = 1 / (sp.sqrt(2) * v ** 2)
identity = sp.simplify(GF_v / GF_tree - (1 - dr) ** 2 * (1 + drho))
check(f"symbolic: G_F^v/G_F^tree - (1-δ_r)²(1+δρ) == 0  (sympy: {identity})", identity == 0)
print("    ⟹ the triangle gap is DERIVED content: (1-δ_r)²(1+δρ) - 1 = δρ - 2δ_r + O(δ²).")

print("=" * 88)
print(" T1' numeric on the LIVE leaves (locates the cross-file slack)")
print("=" * 88)
GF_tree_live = g_2_MZ ** 2 / (4 * math.sqrt(2) * m_W_pred ** 2)
gap_live = G_F_pred / GF_tree_live - 1
gap_ident = (1 - DR) ** 2 * (1 + DRHO) - 1
print(f"    live triangle gap (S4's number)          = {gap_live*100:+.4f}%")
print(f"    identity value (1-δ_r)²(1+δρ) - 1        = {gap_ident*100:+.4f}%")
print(f"    first-order δρ - 2δ_r                    = {(DRHO - 2*DR)*100:+.4f}%")
slack = (1 + gap_live) / (1 + gap_ident) - 1
print(f"    residual chain slack                     = {slack*100:+.4f}%")
# locate the slack: the chain identity requires g_2^2 == 4*pi*alpha_2 with the SAME
# alpha_2 as M_Z.py's iteration, and s^2 == alpha_Y/(alpha_2+alpha_Y), and
# m_W == M_Z*sqrt(1-s^2)*sqrt(1+delta_rho), and G_F^v == 1/(sqrt2 v^2). Test each:
mW_chain = M_Z_GeV * math.sqrt(1 - sin2_theta_W_MZ) * math.sqrt(1 + DRHO)
GFv_chain = 1 / (math.sqrt(2) * v_pred ** 2)
print(f"    m_W leaf vs chain M_Z·c·√(1+δρ): {m_W_pred:.4f} vs {mW_chain:.4f} "
      f"({(m_W_pred/mW_chain-1)*100:+.4f}%)")
print(f"    G_F leaf vs 1/(√2 v²):           {G_F_pred:.6e} vs {GFv_chain:.6e} "
      f"({(G_F_pred/GFv_chain-1)*100:+.4f}%)")
check("slack < 0.01% and fully accounted by cross-file rounding of the SAME chain",
      abs(slack) < 1e-4)
check("identity reproduces the S4 'gap' to the slack level",
      abs(gap_live - gap_ident) < 1.5e-4)

print("=" * 88)
print(" T2  corollary: alpha-form width == G_F^v-form / (1-delta_r)^2 ; the lone demand")
print("=" * 88)
# Under T1: 4√2 G_F^v M_Z² = [g2²/c²]·(1-δ_r)²  ⟹  alpha-form/G_F^v-form = 1/(1-δ_r)²
ratio_forms = (g_2_MZ ** 2 / (1 - sin2_theta_W_MZ)) / (4 * math.sqrt(2) * G_F_pred * M_Z_GeV ** 2)
check(f"alpha-form/G_F^v-form = 1/(1-δ_r)² exactly under T1: numerically "
      f"{ratio_forms:.6f} vs {1/(1-DR)**2:.6f} (slack-level agreement)",
      abs(ratio_forms * (1 - DR) ** 2 - 1) < 1.5e-4)
print("""    With the (1+δρ) factor of the golden rule's ρ̄ slot made explicit:
      Γ/M(alpha-form)      = Γ/M(G_F^v, ρ̄=1) × 1/(1-δ_r)²        [+0.677% relative]
      Γ/M(G_F^v, ρ̄=1+δρ)  = Γ/M(G_F^v, ρ̄=1) × (1+δρ)            [+1.091%]
    COMPARISON (PDG, marked): measured sits at ρ̄_width - 1 = +0.24% ± 0.09% above the
    ρ̄=1 G_F^v-form. NEITHER framework normalization matches: 1/(1-δ_r)² gives +0.68%,
    (1+δρ) gives +1.09%, ρ̄=1 gives 0. THE LONE OPEN DIRECTION (sub-eq 7a): derive
    which dressing the substrate's Z-pole residue/current vertex forces on the
    G_F^v-normalized width. Candidate list PRE-POISONED in the docstring.""")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
