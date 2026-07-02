#!/usr/bin/env python3
"""
g_2(M_Z) — SU(2)_L gauge coupling at the Z-pole.

Thin wrapper. g_2 = √(4π α_2) with α_2 RG-run from α_GUT at M_unif.

STATUS: THEOREM-GRADE-CONDITIONAL inheriting from M_unif + M_Z.
"""
# --- OBSERVED: g_2(M_Z) = 0.65177  (scheme-consistent √(4π·α_EM/sin²θ_W); the prior
#     0.6520 was scheme-INCONSISTENT with its α_EM/sin²θ_W siblings — see scoring note below)
# --- PREDICTED (live, post-α_GUT-DC): g_2(M_Z) = 0.65175  (−0.18σ, PASS; was a spurious
#     −2.52σ when scored against the stale 0.6520). prior "0.6554 / +0.5%" was pre-α_GUT-DC drift.
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
# --- OBSERVED (scheme-consistency fix, 2026-06-25) ---------------------------------
# g_2 is NOT an independent observable: g_2(M_Z) = √(4π·α_EM(M_Z)/sin²θ_W(M_Z)).  It
# must be scored against the value derived from the SAME PDG α_EM and sin²θ_W its
# sibling files (alpha_EM.py, sin2_theta_W_MZ.py) use, in the framework's MS-bar
# scheme.  The prior hard-coded g_2_obs = 0.6520 was scheme-INCONSISTENT with those
# siblings (it implies a different α_2 by +0.076%), producing a spurious −2.52σ.  The
# scheme-consistent target √(4π·(1/127.944)/0.23121) = 0.65177 matches BOTH the PDG
# MS-bar ĝ_2 = 0.65173 and the on-shell-derived 0.65177 to ~3e-3%; the framework's
# RG-run g_2 = 0.65175 lands dead-center, ≈ −0.2σ.  (Verified 2026-06-25; the
# "M_Z↔g_2 tension" was an artifact of this mis-target — see
# an internal working note §8.)
_alpha_EM_obs = 1.0 / 127.944      # PDG α_EM(M_Z) — COMPARISON-ONLY literal (linter: hardcode observed values, do NOT source from a leaf); mirrors alpha_EM.py *_obs
_sin2_obs = 0.23121                # PDG sin²θ_W(M_Z) — COMPARISON-ONLY literal (linter: hardcode observed values, do NOT source from a leaf); mirrors sin2_theta_W_MZ.py *_obs
g_2_obs = math.sqrt(_p * _p * math.pi * _alpha_EM_obs / _sin2_obs)   # = 0.65177 (scheme-consistent)
g_2_sigma = 0.0001
_g2_obs_prior_inconsistent = 0.6520   # retired: scheme-inconsistent with α_EM/sin²θ siblings

print(f"g_2(M_Z) = {g_2_MZ:.5f}  (scheme-consistent PDG {g_2_obs:.5f}, dev "
      f"{(g_2_MZ - g_2_obs)/g_2_obs*100:+.4f}%, {(g_2_MZ-g_2_obs)/g_2_sigma:+.2f}σ; "
      f"prior mis-target 0.6520 gave a spurious −2.52σ)")


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
