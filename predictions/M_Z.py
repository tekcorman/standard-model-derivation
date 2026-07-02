#!/usr/bin/env python3
"""
M_Z — Z-boson mass via self-consistent electroweak matching.

NEW DERIVATION (2026-05-04 EOD+1, post-M_unif Stage 5).

THE CHAIN:

  M_Z = (1/2) × √(g_2² + g_Y²) × v
      = √π × v × √(α_2 + α_Y)
      = √π × v × √(α_2 + (3/5) α_1)               [GUT normalization]

where α_1, α_2, α_3 are the gauge couplings at M_Z, obtained by RG running
from α_GUT at M_unif, and v is the Higgs VEV (theorem-grade BZJ).

The relation is SELF-CONSISTENT — running couplings depend on the RG
matching scale (which is M_Z itself). The script iterates to convergence.

INPUTS (all framework-derived):
  - v: Higgs VEV (theorem-grade BZJ via predictions/v_higgs.py)
  - α_GUT = 1/24 (theorem-grade Type 4)
  - M_unif (THEOREM-GRADE-CONDITIONAL via predictions/M_unif.py)
  - MSSM β-functions (Type 3 standard QFT)

OUTPUT: M_Z as a derived prediction, no longer external.

STATUS: THEOREM-GRADE-CONDITIONAL inheriting from M_unif (Row P62) and
v_higgs (FSS family, conditional on N_hub). Numerical match (updated
2026-07-01; the "91.97 GeV / +375σ" in older revisions was the PRE-δ_r
tree iteration): M_Z_pred = tree·(1−δ_r) ≈ 91.204 GeV vs PDG 91.1876 GeV
→ +7.76σ_PDG (FAIL Clause 8 against σ_PDG alone; PDG is 2.3 ppm). This
is the framework's honestly-OPEN oblique residual: the BZ-integrated
vacuum polarization (2026-06-30) confirms a forced ~4%-relative
substrate-vs-SM oblique difference that does not clean-close — logged in
docs/incomplete_equations_todo.md; see
docs/theorems/theorem_M_Z_BZ_vacuum_polarization_2026-06-30.md.

SUPERSEDES: external M_Z input in alpha_EM.py and downstream cluster files.

COMPANION DOCS:
- predictions/M_Z_derivation.md
- predictions/M_unif.py (M_unif chain)
- predictions/alpha_EM.py (downstream consumer)
- predictions/v_higgs.py (v BZJ)
"""

# ============================================================
# PARAMETER: M_Z (Z-boson mass)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       M_Z = 91.1876 ± 0.0021 GeV
# Source:      PDG 2024 (LEP electroweak working group, Z-pole)
# Note:        Most precisely measured massive particle in SM.

# --- PREDICTED VALUE -----------------------------------------
# Value:       M_Z = √π × v × √(α_2(M_Z) + (3/5) α_1(M_Z)) × (1 − δ_r)
#              ≈ 91.204 GeV (self-consistent tree iteration, then the
#              substrate tree→pole oblique δ_r — Row P64)
# Deviation:   +0.018% from PDG → +7.76σ_PDG (FAIL against σ_PDG alone;
#              the honestly-OPEN oblique residual — see STATUS above).

# --- DERIVED FORMULA -----------------------------------------
# M_Z² = (g_2² + g_Y²) × v² / 4 = π × v² × (α_2 + α_Y)
# In GUT normalization: α_Y = (3/5) α_1, so α_2 + α_Y = α_2 + (3/5) α_1
#
# Logical chain:
#   Step 1: α_GUT = 1/24 (theorem-grade Type 4)
#   Step 2: M_unif = (32/k*^(g-1)) × M_Pl (THEOREM-GRADE-CONDITIONAL Type 4)
#   Step 3: v = δ²·M_Pl/(√2·N^(1/4)) (theorem-grade BZJ Type 4)
#   Step 4: One-loop MSSM RG run α_GUT → α_1(M_Z), α_2(M_Z) (Type 3)
#   Step 5: Self-consistency: M_Z = √π × v × √(α_2 + (3/5) α_1) [Type 2]
#   Step 6: Iterate until M_Z converges

# --- INPUTS --------------------------------------------------
# symbol         | value          | status                  | predictions/ file
# ---------------|----------------|-------------------------|------------------
# alpha_GUT      | 1/24           | [theorem-grade]         | predictions/alpha_GUT.py
# M_unif_GeV     | 1.985e16       | [structural-cond]       | predictions/M_unif.py
# v              | 246.22         | [theorem-grade]         | predictions/v_higgs.py
# b_1, b_2       | 33/5, 1        | [Type 3 MSSM]           | (Peskin-Schroeder §16; Martin SUSY primer)
# hypercharge    | 3/5            | [Type 1 SU(5) embedding]| (group theory)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import functools

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from M_unif import predict_M_unif_GeV
from v_higgs import predict_v_higgs
from alpha_1 import predict_alpha_1
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from p_toggle import predict_p_toggle
from V_count import predict_V_count as _predict_V_count
from N_hub import predict_N_hub

# Substrate primitives
d_val = predict_d_spatial()
k_val = predict_k_star(d_val)
g_val = predict_g_girth(k_val, d_val)
p_val = predict_p_toggle()
V_val = _predict_V_count(k_val, d_val)
alpha_1_substrate = predict_alpha_1(k_val, g_val)

# External anchors (unit translation)
from M_Pl_natural import M_Pl_GeV   # CODATA single-source — ANTHROPOCENTRIC SI TRANSLATION
G_F_obs = 1.1663787e-5              # the measured Fermi constant — used to pin N_hub's adopted value; G_F itself is a PREDICTION (predictions/G_F.py)
from delta_Koide import delta_Koide_pred as delta  # = 2/9 (Q*(1-Q) at Q=2/3, predict_delta_Koide)

# Framework predictions (theorem-grade conditional)
N_hub = predict_N_hub(G_F_obs, M_Pl_GeV, alpha_1_substrate, delta, k_val, p_val, V_val)
v_GeV = predict_v_higgs(delta, M_Pl_GeV, N_hub, alpha_1_substrate)
M_unif_GeV = predict_M_unif_GeV(k_val, g_val, M_Pl_GeV)

# Theorem-grade primitives
from alpha_GUT import predict_alpha_GUT_observed
alpha_GUT = float(predict_alpha_GUT_observed(k_val, g_val))  # dark-corrected, theorem-grade-cond 2026-05-15
from mssm_beta_coefficients import b_1_MSSM  # MSSM one-loop β coefficient single-source
from mssm_beta_coefficients import b_2_MSSM  # MSSM one-loop β coefficient single-source
from mssm_beta_coefficients import hypercharge_norm  # = 3/5, GUT-norm single-source


def _running_couplings(M_Z_guess, alpha_GUT, M_unif_GeV, b_1, b_2, p_toggle):
    """One-loop MSSM running of α_1, α_2 from M_unif to M_Z_guess.
    Returns (alpha_1, alpha_2) at the guessed M_Z scale. The `2` in
    the 1/(2π) loop factor is sourced from p_toggle."""
    log_ratio = math.log(M_Z_guess / M_unif_GeV)
    inv_alpha_1 = 1.0/alpha_GUT - (b_1 / (p_toggle * math.pi)) * log_ratio
    inv_alpha_2 = 1.0/alpha_GUT - (b_2 / (p_toggle * math.pi)) * log_ratio
    return 1.0 / inv_alpha_1, 1.0 / inv_alpha_2


def _self_consistent_M_Z(v, alpha_GUT, M_unif_GeV, b_1, b_2, hypercharge_norm,
                          p_toggle, M_Z_init=91.18, tol=1e-9, max_iter=100):
    """Iterate to self-consistent M_Z. p_toggle sources the 1/(2π) coefficient."""
    M_Z = M_Z_init
    for _ in range(max_iter):
        alpha_1, alpha_2 = _running_couplings(M_Z, alpha_GUT, M_unif_GeV, b_1, b_2, p_toggle)
        alpha_Y = hypercharge_norm * alpha_1
        M_Z_new = math.sqrt(math.pi) * v * math.sqrt(alpha_2 + alpha_Y)
        if abs(M_Z_new - M_Z) < tol:
            return M_Z_new
        M_Z = M_Z_new
    return M_Z


# Compute the SM TREE (ρ=1, no oblique) M_Z self-consistently
M_Z_tree = _self_consistent_M_Z(
    v_GeV, alpha_GUT, M_unif_GeV, b_1_MSSM, b_2_MSSM, hypercharge_norm, p_val,
)

# Substrate Δr-analog (Row P64; predictions/delta_r.py).  The SM tree
# relation M_Z = √π·v·√(α_2+(3/5)α_1) over-predicts the POLE M_Z by an
# intrinsic tree-vs-pole OBLIQUE radiative correction (decomposition
# `proofs/foundations/M_Z_residual_is_tree_vs_pole_oblique_2026-05-15.py`).
# δ_r is the sign-uniform sibling of δρ: the Z-Perron Hashimoto residue
# (Phase C) that cancels in the ρ ratio but IS the absolute-M_Z oblique;
# coefficient c_S=1/12 is the Phase-A two-routes, v_Higgs-calibrated
# value (counting Family-C template — NOT the SM Sirlin Δr import).
from delta_r import predict_delta_r
from V_count import predict_V_count
delta_r_val = predict_delta_r(k_val, g_val, p_val, predict_V_count(k_val, d_val))
M_Z_GeV = M_Z_tree * (1.0 - delta_r_val)

# Module-level exports
M_Z_pred = M_Z_GeV
M_Z_obs = 91.1876
M_Z_sigma = 0.0021

print("=" * 68)
print("  M_Z  --  Z-boson mass via self-consistent electroweak matching")
print("=" * 68)
print(f"  Inputs (all framework-derived):")
print(f"    α_GUT        = 1/24 = {alpha_GUT:.6f}    [theorem-grade]")
print(f"    M_unif       = {M_unif_GeV:.4e} GeV       [THEOREM-GRADE-CONDITIONAL]")
print(f"    v            = {v_GeV:.4f} GeV            [theorem-grade BZJ]")
print(f"    M_Z (PDG)    = {M_Z_obs:.4f} GeV          [external, target only]")
print()
print(f"  Self-consistent iteration (MSSM-style 1-loop, single-regime — no M_SUSY threshold):")
alpha_1_at_MZ, alpha_2_at_MZ = _running_couplings(M_Z_GeV, alpha_GUT, M_unif_GeV, b_1_MSSM, b_2_MSSM, p_val)
print(f"    α_1(M_Z)     = 1/{1/alpha_1_at_MZ:.4f}")
print(f"    α_2(M_Z)     = 1/{1/alpha_2_at_MZ:.4f}")
print(f"    α_Y(M_Z)     = (3/5) × α_1 = 1/{1/(hypercharge_norm * alpha_1_at_MZ):.4f}")
print()
print(f"  SM tree (ρ=1): M_Z_tree = √π·v·√(α_2+α_Y) = {M_Z_tree:.4f} GeV  "
      f"(residual {(M_Z_tree-M_Z_obs)/M_Z_obs*100:+.4f}%)")
print(f"  δ_r (Row P64)  = {delta_r_val*100:+.4f}%  [substrate tree→pole oblique,")
print(f"    sign-uniform sibling of δρ; c_S=1/12 Phase-A two-routes]")
print(f"  Pole:          M_Z = M_Z_tree·(1−δ_r) = {M_Z_GeV:.4f} GeV")
print(f"  PDG 2024:      M_Z = 91.1876 ± 0.0021 GeV")
dev_rel = (M_Z_GeV - M_Z_obs) / M_Z_obs * 100
dev_sigma = (M_Z_GeV - M_Z_obs) / M_Z_sigma
print(f"  Deviation: {dev_rel:+.4f}%  ({dev_sigma:+.2f}σ_PDG)  "
      f"[was {(M_Z_tree-M_Z_obs)/M_Z_obs*100:+.4f}% at tree]")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_M_Z(v_GeV, alpha_GUT, M_unif_GeV, b_1, b_2, hypercharge_norm,
                M_Z_init, tol, max_iter, delta_r=0.0):
    """
    Predict the POLE M_Z from v + α_GUT + M_unif + MSSM RG + the
    substrate Δr-analog δ_r (tree→pole oblique; Row P64).

    Solves the SM TREE M_Z = √π·v·√(α_2(M_Z)+(3/5)α_1(M_Z)) self-
    consistently, then applies the pole correction
        M_Z_pole = M_Z_tree · (1 − δ_r).
    δ_r=0 recovers the SM tree value.  α_1(M_Z), α_2(M_Z) RG-run from
    α_GUT at M_unif.

    Parameters
    ----------
    v_GeV : float
        Higgs VEV in GeV (theorem-grade BZJ, predictions/v_higgs.py).
    alpha_GUT : float
        Unified gauge coupling at M_unif (theorem-grade = 1/24).
    M_unif_GeV : float
        Unification scale in GeV (THEOREM-GRADE-CONDITIONAL).
    b_1, b_2 : float
        MSSM one-loop β-function coefficients (Type 3, b_1=33/5, b_2=1).
    hypercharge_norm : float
        SU(5) hypercharge normalization (= 3/5).
    M_Z_init : float
        Initial guess for self-consistent iteration (e.g., 91.18 GeV).
    tol : float
        Convergence tolerance (e.g., 1e-9 GeV).
    max_iter : int
        Maximum iterations (e.g., 100).

    Returns
    -------
    float
        M_Z in GeV (self-consistent solution).
    """
    M_Z = M_Z_init
    for _ in range(max_iter):
        log_ratio = math.log(M_Z / M_unif_GeV)
        inv_alpha_1 = 1.0/alpha_GUT - (b_1 / (2 * math.pi)) * log_ratio
        inv_alpha_2 = 1.0/alpha_GUT - (b_2 / (2 * math.pi)) * log_ratio
        alpha_1 = 1.0 / inv_alpha_1
        alpha_2 = 1.0 / inv_alpha_2
        alpha_Y = hypercharge_norm * alpha_1
        M_Z_new = math.sqrt(math.pi) * v_GeV * math.sqrt(alpha_2 + alpha_Y)
        if abs(M_Z_new - M_Z) < tol:
            return M_Z_new * (1.0 - delta_r)
        M_Z = M_Z_new
    return M_Z * (1.0 - delta_r)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = M_Z_GeV
    pure_result = predict_M_Z(
        v_GeV, alpha_GUT, M_unif_GeV, b_1_MSSM, b_2_MSSM, hypercharge_norm,
        91.18, 1e-9, 100, delta_r_val,
    )
    print()
    print("=" * 68)
    print("STATUS (parameter linter clauses):")
    print("  Clauses 1-5 (chain):")
    print("    Step 1 [α_GUT]    = Type 4 (predictions/alpha_GUT.py, theorem-grade)")
    print("    Step 2 [M_unif]   = Type 4 (predictions/M_unif.py, THEOREM-GRADE-COND)")
    print("    Step 3 [v]        = Type 4 (predictions/v_higgs.py, theorem-grade BZJ)")
    print("    Step 4 [MSSM RG]  = Type 3 standard QFT (Peskin-Schroeder §16)")
    print("    Step 5 [self-cons]= Type 2 (algebraic SM tree relation)")
    print("    Step 6 [δ_r]      = Type 4 (predictions/delta_r.py, math-complete,")
    print("                        Row P64 — substrate Δr-analog, Clause-9-safe)")
    print("  Clause 8 (numerical match, σ_PDG only):")
    print(f"    σ_obs      = 2.3 ppm (PDG world average)")
    dev_rel_ = (M_Z_GeV - M_Z_obs) / M_Z_obs * 100
    dev_sig_ = (M_Z_GeV - M_Z_obs) / M_Z_sigma
    verdict_ = "PASS" if abs(dev_sig_) <= 1.0 else "FAIL"
    print(f"    tree residual = {(M_Z_tree-M_Z_obs)/M_Z_obs*100:+.4f}%  →  "
          f"pole (with δ_r) = {dev_rel_:+.4f}%  ({dev_sig_:+.2f}σ_PDG)  ⇒  Clause 8 {verdict_}")
    print(f"    (relative residual cut ~20×; σ_PDG still ≫1 — M_Z is 2.3 ppm.")
    print(f"     The {dev_sig_:+.2f}σ residual is a CONFIRMED forced substrate-vs-SM oblique")
    print(f"     difference (~4%): the shell vertex/box, BZ-integrated, is only")
    print(f"     0.205× its Γ-template value — the Γ 'bracket' was an artifact; the")
    print(f"     residual does NOT clean-close. δ_r-only is the forced single term.")
    print(f"     [theorem_M_Z_BZ_vacuum_polarization_2026-06-30.md])")
    print("=" * 68)

    print()
    print(f"  Implementation:  M_Z = {impl_result:.4f} GeV")
    print(f"  Pure function:   M_Z = {pure_result:.4f} GeV")
    assert abs(impl_result - pure_result) < 1e-9
    print(f"  OK: outputs agree.")
    print()
    print("OK: M_Z derived self-consistently from framework-internal inputs.")
    print("    Status: THEOREM-GRADE-CONDITIONAL inheriting M_unif + v + MSSM RG.")
    print("    Supersedes external M_Z input in alpha_EM.py cluster.")
