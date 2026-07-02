#!/usr/bin/env python3
"""
g_3(M_Z) — SU(3)_c gauge coupling at the Z-pole.

Thin wrapper. g_3 = √(4π α_3) with α_3 RG-run from α_3^observed at M_unif
under sector-specific dark correction c_color = 1/4.

STATUS (2026-05-26 EOD+1): sector-specific c_color = 1/4 brings g_3(M_Z)
prediction to within ~0.13σ of PDG (was -1.36σ under uniform c=1/3).
THEOREM-GRADE-NUMERICAL for the SU(3)_c sector per
docs/theorems/theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md.

Consistency: this file MUST use the same c_color path as predictions/alpha_s.py
since α_s = g_3²/(4π). Both pull from predict_alpha_GUT_observed_sector(...,'color').

PRIOR STATUS (pre-2026-05-26 EOD+1; kept for record): 🟡 OUT-OF-SCOPE-BY-
CONSTRUCTION per Row P68 re-grade 2026-05-17 Move-1 — under uniform c=1/3,
the -0.57% residual was attributed to omitted hadronic-VP/IR-threshold
matching. With c_color = 1/4, the residual closes structurally.
"""
# --- OBSERVED: g_3(M_Z) = 1.218 ± 0.005 (PDG 2024 derived from α_s)
# --- PREDICTED (live node, post-c_color=1/4): g_3(M_Z) ≈ 1.217 (−0.06%, ~−0.13σ)
# --- INPUTS: alpha_GUT_observed_sector(color), M_unif, M_Z, b_3 (MSSM)

import sys, os, math, functools
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from M_unif import predict_M_unif_GeV
from M_Z import M_Z_GeV
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from p_toggle import predict_p_toggle
from V_count import predict_V_count

_d = predict_d_spatial()
_k = predict_k_star(_d)
_g = predict_g_girth(_k, _d)
_p = predict_p_toggle()
_V = predict_V_count(_k, _d)

# Sector-specific α_GUT^observed (color sector, c=1/4) per
# theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md
from alpha_GUT import predict_alpha_GUT_observed_sector
alpha_GUT_color = float(predict_alpha_GUT_observed_sector(_k, _g, 'color', _p, _V))
from mssm_beta_coefficients import b_3_MSSM  # MSSM one-loop β coefficient single-source
from M_Pl_natural import M_Pl_GeV   # CODATA single-source — ANTHROPOCENTRIC SI TRANSLATION
M_unif_GeV = predict_M_unif_GeV(_k, _g, M_Pl_GeV)

log_ratio = math.log(M_Z_GeV / M_unif_GeV)
inv_alpha_3 = 1.0/alpha_GUT_color - (b_3_MSSM / (_p*math.pi)) * log_ratio
alpha_3_MZ = 1.0 / inv_alpha_3
g_3_MZ = math.sqrt(_p*_p * math.pi * alpha_3_MZ)   # √(4πα), 4=p²

g_3_pred = g_3_MZ
g_3_obs = 1.218
g_3_sigma = 0.005

print(f"g_3(M_Z) = {g_3_MZ:.4f}  (PDG 1.218 ± 0.005, dev "
      f"{(g_3_MZ - 1.218)/1.218*100:+.3f}%, {(g_3_MZ - 1.218)/0.005:+.2f}σ)")
print(f"  c_color = 1/4 (sector-specific, theorem-grade-numerical 2026-05-26 EOD+1)")


@functools.lru_cache(maxsize=None)
def predict_g_3(alpha_GUT, M_unif_GeV, M_Z_GeV, b_3, p_toggle):
    """g_3(M_Z) = √(4π α_3(M_Z)) via one-loop MSSM RG running from α_3^observed.

    The 2 and 4 in the loop coefficients source from p_toggle: 2π = p·π,
    4π = p²·π.

    Parameters
    ----------
    alpha_GUT : float    Sector-specific α_3^observed at M_unif (c_color = 1/4)
    M_unif_GeV : float   Unification scale in GeV
    M_Z_GeV : float      Z-pole mass in GeV
    b_3 : float          SU(3)_c MSSM one-loop β-coefficient (= -3)
    p_toggle : int       toggle arity (from predict_p_toggle)

    Returns
    -------
    float                Predicted g_3(M_Z)
    """
    log_r = math.log(M_Z_GeV / M_unif_GeV)
    inv_a3 = 1.0/alpha_GUT - (b_3 / (p_toggle * math.pi)) * log_r
    return math.sqrt(p_toggle * p_toggle * math.pi / inv_a3)


if __name__ == "__main__":
    impl = g_3_MZ
    pure = predict_g_3(alpha_GUT_color, M_unif_GeV, M_Z_GeV, b_3_MSSM, _p)
    assert abs(impl - pure) < 1e-12
    print(f"OK: implementation = pure = {impl:.6f}")
    # Consistency with alpha_s.py (same c_color path)
    from alpha_s import alpha_s_MZ as alpha_s_from_alpha_s_py
    alpha_s_from_g_3 = g_3_MZ**2 / (4*math.pi)
    assert abs(alpha_s_from_g_3 - alpha_s_from_alpha_s_py) < 1e-10, \
        f"Inconsistency: alpha_s from g_3 ({alpha_s_from_g_3}) != alpha_s.py ({alpha_s_from_alpha_s_py})"
    print(f"OK: α_s = g_3²/(4π) = {alpha_s_from_g_3:.6f} matches predictions/alpha_s.py.")
