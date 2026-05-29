#!/usr/bin/env python3
"""
g_2(M_Z) — SU(2)_L gauge coupling at the Z-pole.

Thin wrapper. g_2 = √(4π α_2) with α_2 RG-run from α_GUT at M_unif.

STATUS: THEOREM-GRADE-CONDITIONAL inheriting from M_unif + M_Z.
"""
# --- OBSERVED: g_2(M_Z) = 0.6520 ± 0.0001 (PDG 2024 derived)
# --- PREDICTED (live 2026-05-22, post-α_GUT-DC): g_2(M_Z) ≈ 0.65175 (−0.04%, near-PASS)
#     prior "0.6554 / +0.5%" was stale pre-α_GUT-DC drift; updated to live value
# --- INPUTS: alpha_GUT, M_unif, M_Z, b_2 (MSSM)

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

_d = predict_d_spatial()
_k = predict_k_star(_d)
_g = predict_g_girth(_k, _d)
_p = predict_p_toggle()

from alpha_GUT import predict_alpha_GUT_observed
alpha_GUT = float(predict_alpha_GUT_observed(_k, _g))  # dark-corrected, theorem-grade-cond 2026-05-15
from mssm_beta_coefficients import b_2_MSSM  # MSSM one-loop β coefficient single-source
from M_Pl_natural import M_Pl_GeV   # CODATA, single source — ANTHROPOCENTRIC SI TRANSLATION
M_unif_GeV = predict_M_unif_GeV(_k, _g, M_Pl_GeV)

log_ratio = math.log(M_Z_GeV / M_unif_GeV)
inv_alpha_2 = 1.0/alpha_GUT - (b_2_MSSM / (_p*math.pi)) * log_ratio
alpha_2_MZ = 1.0 / inv_alpha_2
g_2_MZ = math.sqrt(_p*_p * math.pi * alpha_2_MZ)   # √(4πα), 4=p²

g_2_pred = g_2_MZ
g_2_obs = 0.6520
g_2_sigma = 0.0001

print(f"g_2(M_Z) = {g_2_MZ:.4f}  (PDG 0.6520, dev "
      f"{(g_2_MZ - 0.6520)/0.6520*100:+.3f}%)")


@functools.lru_cache(maxsize=None)
def predict_g_2(alpha_GUT, M_unif_GeV, M_Z_GeV, b_2, p_toggle):
    """g_2 at M_Z via MSSM RG. 2 and 4 in loop coefs sourced from p_toggle."""
    log_r = math.log(M_Z_GeV / M_unif_GeV)
    inv_a2 = 1.0/alpha_GUT - (b_2 / (p_toggle * math.pi)) * log_r
    return math.sqrt(p_toggle * p_toggle * math.pi / inv_a2)


if __name__ == "__main__":
    impl = g_2_MZ
    pure = predict_g_2(alpha_GUT, M_unif_GeV, M_Z_GeV, b_2_MSSM, _p)
    assert abs(impl - pure) < 1e-12
    print(f"OK: implementation = pure = {impl:.6f}")
