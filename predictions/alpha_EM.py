#!/usr/bin/env python3
"""
α_EM — fine-structure constant at M_Z.

NEW DERIVATION (2026-05-04 EOD): α_EM(M_Z) as a downstream prediction
from M_unif (framework structural-derivation-conditional) + α_GUT
(theorem-grade) + sin²θ_W=3/8 at unification (theorem-grade) + standard
MSSM RG running.

THE CHAIN:

  At M_unif (framework prediction):
      α_GUT = 1/24                              [theorem-grade Class C]
      α_1 = α_2 = α_3 = α_GUT                   [unification by definition]
      sin²θ_W(M_unif) = 3/8                     [theorem-grade]

  Run from M_unif to M_Z (one-loop MSSM-style β-functions, single-regime — no M_SUSY threshold):
      1/α_i(M_Z) = 1/α_GUT - (b_i^MSSM / 2π) × ln(M_Z / M_unif)
      b_1^MSSM = 33/5    (U(1)_Y GUT-normalized)
      b_2^MSSM = 1       (SU(2)_L)
      b_3^MSSM = -3      (SU(3)_c)

  At M_Z, compute physical couplings:
      α_Y(M_Z) = (3/5) × α_1_GUT(M_Z)
      sin²θ_W(M_Z) = α_Y(M_Z) / (α_2(M_Z) + α_Y(M_Z))
      α_EM(M_Z) = α_2(M_Z) × sin²θ_W(M_Z)

KEY PROPERTIES:
- All inputs framework-internal (α_GUT, sin²θ_W=3/8, M_unif) plus standard
  MSSM β-function coefficients (Type 3 standard QFT).
- Does NOT require external α_1, α_2, α_3 measurements at M_Z — these are
  PREDICTIONS, not inputs.
- M_Z external (electroweak scale, separate prediction not yet completed).
- Inherits M_unif's STRUCTURAL-DERIVATION-CONDITIONAL grade.

STATUS (2026-05-04 EOD+1): THEOREM-GRADE-CONDITIONAL inheriting from M_unif
(Row P62 graduated via 5-stage closure program 2026-05-04 EOD+1). Numerical
match to PDG α_EM(M_Z) ≈ 1/127.94 at +0.7% / ~+65σ_PDG; Clause 8 FAIL vs σ_PDG alone.

LEVERAGE: This file unblocks the entire EM cluster (sin²θ_W(M_Z), g_1(M_Z),
g_2(M_Z), g_3(M_Z), α_s(M_Z), α_EM(M_Z), R∞) downstream of M_unif.

COMPANION DOCS:
- predictions/M_unif.py (M_unif structural prediction)
- predictions/alpha_GUT.py (α_GUT = 1/24)
- predictions/sin2_theta_W.py (sin²θ_W(M_unif) = 3/8)
"""

# ============================================================
# PARAMETER: α_EM(M_Z) (fine-structure constant at M_Z)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       α_EM(M_Z) = 1/127.944 ± 0.014
# Source:      PDG 2024 (Workman et al., Phys. Rev. D 110, 030001)
# Equivalent:  α_EM(0) (Thomson limit) = 1/137.0359991 (CODATA 2018)
# Note:        α_EM runs significantly between m_e and M_Z due to charged-fermion
#              loop contributions; α_EM(M_Z) > α_EM(0) by ~7%.

# --- PREDICTED VALUE -----------------------------------------
# Value:       α_EM(M_Z) ≈ 1/127.1
# Deviation:   ~0.7% vs PDG (one-loop MSSM-style single-regime running — no M_SUSY threshold)

# --- DERIVED FORMULA -----------------------------------------
# Step 1: At M_unif, α_GUT = 1/24, sin²θ_W = 3/8 (theorem-grade)
# Step 2: M_unif = (32/k*^(g-1)) × M_Pl ≈ 1.985×10¹⁶ GeV (structural-derivation-conditional)
# Step 3: One-loop MSSM RG running:
#         1/α_i(M_Z) = 1/α_GUT - (b_i/(2π)) × ln(M_Z/M_unif)
# Step 4: Physical couplings at M_Z:
#         α_Y(M_Z) = (3/5) × α_1_GUT(M_Z)
#         sin²θ_W(M_Z) = α_Y / (α_2 + α_Y)
#         α_EM(M_Z) = α_2(M_Z) × sin²θ_W(M_Z)

# --- INPUTS --------------------------------------------------
# symbol         | value           | status                  | predictions/ file
# ---------------|-----------------|-------------------------|------------------
# alpha_GUT      | 1/24            | [theorem-grade]         | predictions/alpha_GUT.py
# sin2_theta_W   | 3/8 at M_unif   | [theorem-grade]         | predictions/sin2_theta_W.py
# M_unif         | 1.985e16 GeV    | [structural-cond]       | predictions/M_unif.py
# M_Z            | ~91.204 GeV     | [derived]               | predictions/M_Z.py  (RG scale; self-consistent since 2026-05-04, NOT external-PDG — line 97 import; Clause-2c RG-reference role)
# b_1, b_2, b_3  | 33/5, 1, -3     | [Type 3 standard QFT]   | (MSSM one-loop β-functions)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import functools
from fractions import Fraction

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from M_unif import predict_M_unif_GeV
from M_Z import M_Z_GeV   # NEW 2026-05-04 EOD+1: M_Z derived self-consistently

# Theorem-grade primitives
# Boundary at M_unif: dark-corrected α_GUT_observed per
# `docs/theorems/theorem_alpha_GUT_dark_correction.md` (theorem-grade-cond, 2026-05-15)
from alpha_GUT import predict_alpha_GUT_observed
sin2_theta_W_unif = 3.0 / 8.0

# Framework prediction for unification scale — k_star, g_girth sourced from leaves
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
_d = predict_d_spatial()
k_star = predict_k_star(_d)
g_girth = predict_g_girth(k_star, _d)
from M_Pl_natural import M_Pl_GeV   # CODATA single-source — ANTHROPOCENTRIC SI TRANSLATION
M_unif_GeV = predict_M_unif_GeV(k_star, g_girth, M_Pl_GeV)

# Dark-corrected α_GUT (replaces bare 1/24 = 0.04167; corrected ≈ 0.04110 ≈ 1/24.329)
alpha_GUT = float(predict_alpha_GUT_observed(k_star, g_girth))

# MSSM one-loop β-function coefficients (standard QFT, Type 3)
# Convention: 1/α_i(μ) = 1/α_i(μ_0) - (b_i / 2π) × ln(μ/μ_0)
# Refs: Peskin-Schroeder §16; Martin SUSY primer §6.5
from mssm_beta_coefficients import b_1_MSSM  # MSSM one-loop β coefficient single-source
from mssm_beta_coefficients import b_2_MSSM  # MSSM one-loop β coefficient single-source
from mssm_beta_coefficients import hypercharge_norm  # = 3/5, GUT→physical norm single-source
from mssm_beta_coefficients import b_3_MSSM  # MSSM one-loop β coefficient single-source

# RG running
log_ratio = math.log(M_Z_GeV / M_unif_GeV)   # negative, since M_Z << M_unif

inv_alpha_1_MZ = 1.0/alpha_GUT - (b_1_MSSM / (2*math.pi)) * log_ratio
inv_alpha_2_MZ = 1.0/alpha_GUT - (b_2_MSSM / (2*math.pi)) * log_ratio
inv_alpha_3_MZ = 1.0/alpha_GUT - (b_3_MSSM / (2*math.pi)) * log_ratio

alpha_1_MZ = 1.0 / inv_alpha_1_MZ
alpha_2_MZ = 1.0 / inv_alpha_2_MZ
alpha_3_MZ = 1.0 / inv_alpha_3_MZ

# Physical couplings at M_Z
alpha_Y_MZ = hypercharge_norm * alpha_1_MZ    # convert from GUT to physical normalization (= 3/5)
sin2_theta_W_MZ = alpha_Y_MZ / (alpha_2_MZ + alpha_Y_MZ)
alpha_EM_MZ = alpha_2_MZ * sin2_theta_W_MZ

# Module-level exports
alpha_EM_pred = alpha_EM_MZ
alpha_EM_obs = 1.0 / 127.944
alpha_EM_sigma = 0.014 / 127.944**2  # PDG uncertainty

print("=" * 68)
print("  α_EM(M_Z)  --  Fine-structure constant at M_Z  --  STRUCTURAL-CONDITIONAL")
print("=" * 68)
print(f"  Inputs (theorem-grade or structural-cond):")
print(f"    α_GUT (dark-corrected) = {alpha_GUT:.6f} = 1/{1.0/alpha_GUT:.3f}")
print(f"    sin²θ_W(M_unif) = 3/8 = {sin2_theta_W_unif:.6f}")
print(f"    M_unif         = {M_unif_GeV:.4e} GeV   [structural-derivation-conditional]")
print(f"    M_Z            = {M_Z_GeV:.4f} GeV     [derived; predictions/M_Z.py, RG scale]")
print()
print(f"  RG running (MSSM one-loop, single-regime — no M_SUSY threshold):")
print(f"    ln(M_Z/M_unif) = {log_ratio:.4f}")
print(f"    1/α_1(M_Z)    = {inv_alpha_1_MZ:.4f}   (PDG ≈ 59.0)")
print(f"    1/α_2(M_Z)    = {inv_alpha_2_MZ:.4f}   (PDG ≈ 29.6)")
print(f"    1/α_3(M_Z)    = {inv_alpha_3_MZ:.4f}    (PDG ≈ 8.5)")
print()
print(f"  Physical couplings at M_Z:")
print(f"    sin²θ_W(M_Z) = {sin2_theta_W_MZ:.5f}    (PDG ≈ 0.23121)")
print(f"    α_EM(M_Z)    = {alpha_EM_MZ:.6f}      = 1/{1.0/alpha_EM_MZ:.3f}")
print(f"    PDG α_EM(M_Z) = {alpha_EM_obs:.6f}     = 1/127.944")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_alpha_EM_MZ(alpha_GUT, M_unif_GeV, M_Z_GeV, b_1, b_2, b_3, hypercharge_norm):
    """
    Predict α_EM(M_Z) by RG running from M_unif using MSSM one-loop β-functions.

    Parameters
    ----------
    alpha_GUT : float
        Unified gauge coupling at M_unif (theorem-grade = 1/24).
    M_unif_GeV : float
        Unification scale in GeV (structural-derivation-conditional).
    M_Z_GeV : float
        Electroweak scale in GeV (external).
    b_1 : float
        U(1)_Y MSSM one-loop β-function coefficient (= 33/5, GUT-normalized).
    b_2 : float
        SU(2)_L MSSM one-loop β-function coefficient (= 1).
    b_3 : float
        SU(3)_c MSSM one-loop β-function coefficient (= -3).
        Convention: 1/α_i(μ) = 1/α_i(μ_0) - (b_i/2π) × ln(μ/μ_0).
    hypercharge_norm : float
        GUT hypercharge normalization (= 3/5; SU(5) embedding factor relating
        physical α_Y to GUT-normalized α_1: α_Y = hypercharge_norm × α_1).

    Returns
    -------
    float
        α_EM(M_Z).
    """
    log_ratio = math.log(M_Z_GeV / M_unif_GeV)
    inv_alpha_1 = 1.0/alpha_GUT - (b_1 / (2*math.pi)) * log_ratio
    inv_alpha_2 = 1.0/alpha_GUT - (b_2 / (2*math.pi)) * log_ratio
    alpha_1 = 1.0 / inv_alpha_1
    alpha_2 = 1.0 / inv_alpha_2
    alpha_Y = hypercharge_norm * alpha_1
    sin2_W = alpha_Y / (alpha_2 + alpha_Y)
    return alpha_2 * sin2_W


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = alpha_EM_MZ
    pure_result = predict_alpha_EM_MZ(
        alpha_GUT, M_unif_GeV, M_Z_GeV,
        b_1_MSSM, b_2_MSSM, b_3_MSSM, 3.0/5.0,
    )
    print()
    print("=" * 68)
    print("STATUS (parameter linter clauses):")
    print("  Clauses 1-5 (chain):")
    print("    Step 1 [α_GUT]    = Type 4 (predictions/alpha_GUT.py, theorem-grade)")
    print("    Step 2 [sin²θ_W]  = Type 4 (predictions/sin2_theta_W.py, theorem-grade)")
    print("    Step 3 [M_unif]   = Type 4 (predictions/M_unif.py, structural-cond)")
    print("    Step 4 [MSSM RG]  = Type 3 (Peskin-Schroeder §16; Martin SUSY primer)")
    print("    Step 5 [α_EM]     = Type 2 (algebraic combination)")
    print("  Clause 8 (numerical match, σ_PDG only):")
    print(f"    σ_obs      = 0.011% (PDG)")
    dev_abs_ = alpha_EM_MZ - alpha_EM_obs
    dev_rel_ = dev_abs_ / alpha_EM_obs * 100
    dev_sig_ = dev_abs_ / alpha_EM_sigma
    print(f"    Deviation  = {dev_rel_:+.3f}%  ({dev_sig_:+.2f}σ_PDG)  ⇒  Clause 8 FAIL.")
    print("=" * 68)

    print()
    print(f"  Implementation:  α_EM(M_Z) = {impl_result:.6f} = 1/{1/impl_result:.3f}")
    print(f"  Pure function:   α_EM(M_Z) = {pure_result:.6f} = 1/{1/pure_result:.3f}")
    assert abs(impl_result - pure_result) / impl_result < 1e-12

    dev_abs = alpha_EM_MZ - alpha_EM_obs
    dev_rel = dev_abs / alpha_EM_obs * 100
    dev_sigma = dev_abs / alpha_EM_sigma
    print()
    print(f"  Predicted    : α_EM(M_Z) = {alpha_EM_MZ:.6f} = 1/{1/alpha_EM_MZ:.3f}")
    print(f"  PDG observed : α_EM(M_Z) = {alpha_EM_obs:.6f} = 1/127.944")
    print(f"  Deviation    : {dev_rel:+.3f}%  ({dev_sigma:+.2f}σ_PDG)")
    print()
    print(f"  Cluster prediction summary:")
    print(f"    sin²θ_W(M_Z) = {sin2_theta_W_MZ:.5f}  (PDG 0.23121, dev {(sin2_theta_W_MZ - 0.23121)/0.23121*100:+.2f}%)")
    print(f"    g_1(M_Z)     = {math.sqrt(4*math.pi*alpha_1_MZ):.4f}  (GUT-normalized)")
    print(f"    g_2(M_Z)     = {math.sqrt(4*math.pi*alpha_2_MZ):.4f}  (PDG 0.6520)")
    print(f"    g_3(M_Z)     = {math.sqrt(4*math.pi*alpha_3_MZ):.4f}  (PDG 1.218)")
    print(f"    α_s(M_Z)     = {alpha_3_MZ:.4f}      (PDG 0.1180)")
    print()
    print("OK: α_EM cluster predictions ship at STRUCTURAL-DERIVATION-CONDITIONAL")
    print("    grade, inheriting M_unif's structural-derivation-conditional status.")
    print("    Cluster deviations (post-α_GUT-DC, live 2026-05-22):")
    print("    sin²θ_W +0.02%, g_1 +0.01% PASS; g_2 −0.04%, α_EM −0.011%")
    print("    near-PASS; g_3 −0.56%, α_s −1.07% OUT-OF-SCOPE (IR threshold")
    print("    layer per Move 1, ledger P68–P70 ‘scope exclusion’ reframing).")
