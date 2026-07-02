#!/usr/bin/env python3
"""
g_1(M_Z) — U(1)_Y gauge coupling at the Z-pole, GUT-normalized.

Thin wrapper around alpha_EM.py cluster output. Computes g_1 = √(4π α_1)
where α_1 is RG-run from α_GUT at M_unif to M_Z using MSSM one-loop
β-functions.

GUT-NORMALIZATION: g_1 here means the SU(5)-embedded coupling, which
relates to the SM hypercharge coupling g' via g_1 = √(5/3) g'.

STATUS: THEOREM-GRADE-CONDITIONAL inheriting from M_unif (Row P62) + M_Z.
"""

# --- OBSERVED ------------------------------------------------
# Value: g_1(M_Z) (GUT-normalized) = 0.46144 ± 0.0001  (live g_1_obs)
# Source: derived from PDG α_EM(M_Z)=1/127.944 and sin²θ_W(M_Z)=0.23121
#         as g_1 = √(4π·(5/3)·α_Y),  α_Y = α_EM/(1−sin²θ_W)
# (Header refreshed W2 2026-05-18 — run-live, not docstring; prior
#  "≈0.4626" was stale and had leaked into run_predictions.py's manifest
#  fallback. The DAG runner uses this module's g_1_obs, not the manifest.)

# --- PREDICTED -----------------------------------------------
# Value: g_1(M_Z) = 0.46148  (live g_1_pred; α_1 RG-run α_GUT→M_Z)
# Deviation: +0.37σ_PDG (+0.008% vs derived PDG 0.46144)

# --- INPUTS --------------------------------------------------
# alpha_GUT, M_unif, M_Z, b_1 (MSSM)

# --- IMPLEMENTATION ------------------------------------------
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
from mssm_beta_coefficients import b_1_MSSM  # MSSM one-loop β coefficient single-source
from M_Pl_natural import M_Pl_GeV   # CODATA single-source — ANTHROPOCENTRIC SI TRANSLATION
M_unif_GeV = predict_M_unif_GeV(_k, _g, M_Pl_GeV)

log_ratio = math.log(M_Z_GeV / M_unif_GeV)
inv_alpha_1 = 1.0/alpha_GUT - (b_1_MSSM / (_p*math.pi)) * log_ratio
alpha_1_MZ = 1.0 / inv_alpha_1
g_1_MZ = math.sqrt(_p*_p * math.pi * alpha_1_MZ)   # √(4πα), 4=p²

# Derived "observed" g_1 from PDG α_EM and sin²θ_W:
# α_Y = α_EM/cos²θ_W; α_1_GUT = (5/3) α_Y
alpha_EM_obs = 1.0 / 127.944
sin2_theta_W_obs = 0.23121
alpha_Y_obs = alpha_EM_obs / (1 - sin2_theta_W_obs)
alpha_1_obs = (5.0/3.0) * alpha_Y_obs
g_1_obs = math.sqrt(4 * math.pi * alpha_1_obs)

g_1_pred = g_1_MZ
g_1_sigma = 0.0001

print(f"g_1(M_Z) GUT-norm = {g_1_MZ:.4f}  (derived PDG {g_1_obs:.4f}, "
      f"dev {(g_1_MZ - g_1_obs)/g_1_obs*100:+.3f}%)")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_g_1(alpha_GUT, M_unif_GeV, M_Z_GeV, b_1, p_toggle):
    """g_1 (GUT-normalized) at M_Z via one-loop MSSM RG running from α_GUT.

    The 2 in the 1/(2π) loop coefficient and the 4 in √(4πα) both source
    from p_toggle (= 2 = toggle arity): 2π = p·π, 4π = p²·π.
    """
    log_r = math.log(M_Z_GeV / M_unif_GeV)
    inv_a1 = 1.0/alpha_GUT - (b_1 / (p_toggle * math.pi)) * log_r
    return math.sqrt(p_toggle * p_toggle * math.pi / inv_a1)


if __name__ == "__main__":
    impl = g_1_MZ
    pure = predict_g_1(alpha_GUT, M_unif_GeV, M_Z_GeV, b_1_MSSM, _p)
    assert abs(impl - pure) < 1e-12
    print(f"OK: implementation = pure = {impl:.6f}")
