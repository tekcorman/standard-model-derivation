#!/usr/bin/env python3
"""
α_21_PMNS — first Majorana phase of the PMNS matrix.

THE FORMULA:

    α_21 = g · arg(h)  mod 360°  ≈ 162.388°

where g = 10 is the srs girth and h = (√3 + i√5)/2 is the Hashimoto walker
P-point eigenvalue. arg(h) = arctan(√5/√3) ≈ 52.239°.

DERIVATION CHAIN (grades in brackets):
  - h = (√3+i√5)/2, g = 10            [Type 4 theorem-grade:
        predictions/h_walker_eigenvalue.py, predictions/g_girth.py]
  - 3 generations = C³_gen modes      [Type 4: predictions/R3_observer_c3_generation.py,
        L2 theorem-grade; L3 via PDG mass non-degeneracy as A5(a) external input]
  - ν_R Majorana mass M_R^(m,m) = |M_R| · h_m^g on the C_3 generation channels;
    its PHASE factor h_m^g — one girth-ring's worth of walker holonomy per channel
    — is ADOPTED-NU-MAJ-PHASE  [IDENTIFICATION, A5(a)-adjacent — NOT derived;
        see docs/audits/registers/adoption_register.md].  The real-valued
        |M_R| = δ⁴·M_Pl/(2·k*·N_atoms) is theorem-grade-conditional (m_ν₃ closure)
        and is NOT the conditional here.
  - Type-I PS seesaw m_ν = M_D·M_R⁻¹·M_D^T  [Type 3: Mohapatra-Senjanović 1980]
    + Takagi diagonalization              [Type 2]   →  α_21 = g·arg(h) mod 360°.

STATUS: STRUCTURAL-DERIVATION-CONDITIONAL (re-graded 2026-05-12; was
"UNIQUE-THEOREM-GRADE-CONDITIONAL" 2026-05-04 EOD+1 — inflated).  The
structural chain is theorem-grade-rigour, but the load-bearing M_R phase
factor h_m^g is a bare identification (ADOPTED-NU-MAJ-PHASE), not derived.
A discharge was ATTEMPTED and FAILED 2026-05-12
(proofs/foundations/majorana_M_R_waterfilling.py):
  - Route 1 (A2-T-waterfilled loop sum Σ_{L≥g} 2^{-DL(L)}·h_m^L): does NOT
    converge — Ramanujan saturation |h|²=k*-1=2 makes every retained ring
    length contribute equal magnitude, no finite cutoff, phase drifts as
    ≈(g+L_max)/2·arg(h_m); g·arg(h_m) is only the L_max=g special case.
  - Route 2 (Path-B "cardinality-k orbit ↔ k girth rings"): broken at root —
    K_4 cycle-space generators (triangles) have nonzero Z³ voltages
    {(1,0,0),(0,-1,0),(0,0,1),(1,1,1)} ⇒ don't lift to closed srs cycles ⇒
    the factor `g` is unsourced.
Conditional on (ADOPTED-NU-MAJ-PHASE, C³_gen-L3 mass-ordering, ADOPTED-B3).
Same tier as R-9's γ.2 algebraic-K-complexity encoding choice.

OBSERVATIONALLY UNCONSTRAINED: α_21 is not directly measured (oscillation
experiments see only m_ν² differences; only 0νββ gives weak combined bounds).
The predicted 162.388° holds UNDER the ADOPTED-NU-MAJ-PHASE identification;
not falsified, but identification-conditional — a correct derivation of the
M_R phase might land on a different value.

Companion:
- predictions/alpha_21_PMNS_derivation.md
- proofs/foundations/majorana_M_R_waterfilling.py  (discharge-attempt probe + analysis)
- proofs/flavor/srs_hashimoto_seesaw_verify.py, proofs/foundations/path_b_M_R_upgrade.py
  (the existing M_R = h^g constructions this formalizes)
- docs/audits/registers/adoption_register.md  (ADOPTED-NU-MAJ-PHASE)
- docs/parameters/parameter_uniqueness_ledger.md Row P35
"""

# ============================================================
# PARAMETER: α_21_PMNS (first Majorana phase)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value: largely unconstrained by current data. Some global fits suggest
#        α_21 ~ 0–360° with broad uncertainty (no significant constraint
#        from oscillation data; only neutrinoless double-β decay would
#        constrain Majorana phases, currently null).
# Source: NuFIT 6.0 / PDG 2024 (no constraint on α_21).

# --- PREDICTED VALUE -----------------------------------------
# Value: α_21 = g · arg(h) mod 360° ≈ 162.388°
# Holds under ADOPTED-NU-MAJ-PHASE (M_R^(m,m) = |M_R|·h_m^g identification).
# Not measured; not falsified; identification-conditional.

# --- DERIVED FORMULA -----------------------------------------
# α_21 = g · arg(h) mod 360°
# arg(h) = arctan(Im h / Re h) at P-point
# Im h = √5/2, Re h = √3/2 (theorem-grade per h_walker_eigenvalue.py)

# --- INPUTS --------------------------------------------------
# h_walker eigenvalue            [theorem-grade, predictions/h_walker_eigenvalue.py]
# g_girth                        [theorem-grade, predictions/g_girth.py]
# M_R phase factor h^g           [IDENTIFICATION — ADOPTED-NU-MAJ-PHASE, not derived]

# --- IMPLEMENTATION ------------------------------------------

import sys, os, math, functools
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from h_walker_eigenvalue import h_walker_eigenvalue_pred
from g_girth import predict_g_girth
from k_star import predict_k_star
from d_spatial import predict_d_spatial

d_val = predict_d_spatial()
k_val = predict_k_star(d_val)
g_val = predict_g_girth(k_val, d_val)

h_re = h_walker_eigenvalue_pred.real
h_im = h_walker_eigenvalue_pred.imag
arg_h_rad = math.atan2(h_im, h_re)
arg_h_deg = math.degrees(arg_h_rad)

from M_Pl_natural import DEGREES_PER_CIRCLE   # = 360.0 (universal angle convention)
alpha_21_PMNS = (g_val * arg_h_deg) % DEGREES_PER_CIRCLE

# Module-level exports
alpha_21_PMNS_pred = alpha_21_PMNS
# No constraint from current data — leave _obs absent so scoreboard shows "unconstrained"

print(f"α_21_PMNS = g × arg(h) mod 360°")
print(f"           = {g_val} × {arg_h_deg:.6f}° mod 360°")
print(f"           = {alpha_21_PMNS:.6f}°")
print(f"  Inputs: h = ({h_re:.6f} + {h_im:.6f}i) [theorem-grade], g = {g_val} [theorem-grade]")
print(f"          M_R phase factor h^g = ADOPTED-NU-MAJ-PHASE [identification, not derived]")
print(f"  Status: STRUCTURAL-DERIVATION-CONDITIONAL (re-graded 2026-05-12)")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_alpha_21_PMNS(h_re, h_im, g_girth):
    """
    Predict α_21_PMNS from walker eigenvalue + girth.

    α_21 = g · arg(h) mod 360°

    Parameters
    ----------
    h_re : float
        Real part of Hashimoto walker eigenvalue at P (= √3/2).
    h_im : float
        Imaginary part (= √5/2).
    g_girth : int
        Substrate girth (= 10 for srs).

    Returns
    -------
    float
        α_21_PMNS in degrees, mod 360.
    """
    arg_h = math.atan2(h_im, h_re)
    return (g_girth * math.degrees(arg_h)) % DEGREES_PER_CIRCLE


if __name__ == "__main__":
    impl = alpha_21_PMNS
    pure = predict_alpha_21_PMNS(h_re, h_im, g_val)
    assert abs(impl - pure) < 1e-12
    print(f"OK: implementation = pure = {impl:.6f}°")
