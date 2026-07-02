#!/usr/bin/env python3
"""
α_31_PMNS — second Majorana phase of the PMNS matrix.

α_31 = 2g · arg(h)  mod 360°  ≈ 324.775°
     (= arg((h_ω/h_ω²)^g) mod 360°; the second non-trivial C_3 channel.)

*** INCONSISTENCY FLAG (2026-06-11 Majorana-sector panel; value NOT changed
*** while preregistration row 8 stands frozen): 2g·arg(h) = (φ_ω − φ_ω²)
*** mod 360 is the GENERATION-2-vs-3 relative phase under the adoption's own
*** form M_R = |M_R|·diag(1, h_ω^g, h_ω²^g); the adoption-consistent
*** α₃₁ = φ₃ − φ₁ = arg(h_ω²^g) = 197.612°. The in-repo m_ββ chain
*** (proofs/flavor/srs_unified_mixing.py §8) uses 197.612°. With m₁ = 0 only
*** one Majorana phase combination is physical (|α₃₁ − α₂₁| = 35.225°
*** adoption-consistent). See preregistration register, Annotations
*** 2026-06-11 (row-8 consistency defect).

Same derivation chain as α_21 — see alpha_21_PMNS.py for the full
documentation, including the load-bearing M_R phase factor h^g, which is
the ADOPTED-NU-MAJ-PHASE identification (NOT derived; discharge attempted
and FAILED 2026-05-12, proofs/foundations/majorana_M_R_waterfilling.py).

STATUS: STRUCTURAL-DERIVATION-CONDITIONAL (re-graded 2026-05-12; was
"UNIQUE-THEOREM-GRADE-CONDITIONAL" 2026-05-04 EOD+1 — inflated). Conditional
on (ADOPTED-NU-MAJ-PHASE, C³_gen-L3 mass-ordering, ADOPTED-B3). Unmeasured;
not falsified; identification-conditional. This file is a thin parallel for n=2.

Companion: predictions/alpha_31_PMNS_derivation.md;
predictions/alpha_21_PMNS_derivation.md (full chain);
proofs/foundations/majorana_M_R_waterfilling.py (discharge-attempt probe);
docs/audits/registers/adoption_register.md (ADOPTED-NU-MAJ-PHASE);
docs/parameters/parameter_uniqueness_ledger.md Row P36.
"""

# --- OBSERVED: unconstrained by current data
# --- PREDICTED: α_31 = 2g · arg(h) mod 360° ≈ 324.775°  (under ADOPTED-NU-MAJ-PHASE)

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

from p_toggle import predict_p_toggle
from M_Pl_natural import DEGREES_PER_CIRCLE   # = 360.0 universal angle convention
p_val = predict_p_toggle()
alpha_31_PMNS = (p_val * g_val * arg_h_deg) % DEGREES_PER_CIRCLE   # 2g·arg(h) with 2 = p_toggle

alpha_31_PMNS_pred = alpha_31_PMNS

print(f"α_31_PMNS = 2g × arg(h) mod 360°")
print(f"           = 2 × {g_val} × {arg_h_deg:.6f}° mod 360°")
print(f"           = {alpha_31_PMNS:.6f}°")
print(f"  Status: STRUCTURAL-DERIVATION-CONDITIONAL (re-graded 2026-05-12; M_R phase h^g = ADOPTED-NU-MAJ-PHASE)")


@functools.lru_cache(maxsize=None)
def predict_alpha_31_PMNS(h_re, h_im, g_girth, p_toggle):
    """Predict α_31_PMNS = (p_toggle·g) · arg(h) mod 360°.
    The pre-2026-05-26 literal `2` in `2 * g_girth` is sourced as p_toggle = 2;
    the `360.0` mod is the universal angle convention (M_Pl_natural.DEGREES_PER_CIRCLE).
    """
    arg_h = math.atan2(h_im, h_re)
    return (p_toggle * g_girth * math.degrees(arg_h)) % DEGREES_PER_CIRCLE


if __name__ == "__main__":
    impl = alpha_31_PMNS
    pure = predict_alpha_31_PMNS(h_re, h_im, g_val, p_val)
    assert abs(impl - pure) < 1e-12
    print(f"OK: implementation = pure = {impl:.6f}°")
