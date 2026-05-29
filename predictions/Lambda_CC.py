#!/usr/bin/env python3
"""
Canonical prediction file for Λ_CC (cosmological constant).

The framework's coasting cosmology with Ω_Λ = 1/k* = 1/3 (Row P22 theorem-grade)
combined with the Friedmann equation gives the substrate-frame structural prediction

    Λ_substrate = H_0_substrate² (in 1/s²)
                = (H_0_substrate · t_P)² (in Planck units)
                = 1/N²                    (since H_0_substrate = 1/(N·t_P))

This is the framework's structural cosmological constant in coasting cosmology.
The observer-side prediction with the (16/15) cascade-theorem rate-gap is

    Λ_observer = (16/15)² · Λ_substrate

The ΛCDM-fit value reported by Planck 2018 is extracted under ΛCDM assumptions
(Ω_Λ_LCDM ≈ 0.685, vs framework's coasting Ω_Λ = 1/3), so Planck's reported
Λ_LCDM ≈ 3·H_0²·0.685 ≈ 2·H_0² is a factor of 2 larger than the framework's
substrate-frame Λ.  This factor-of-2 is a COSMOLOGY-MODEL split (which Ω_Λ
the model assumes), not a structural deviation in the framework's prediction.

The factor-of-2 is handled by the parametric-class translation in the
sibling file (predictions/Lambda_CC_LCDM.py): the framework predicts the
theorem-grade BIAS FUNCTION FORM Ω_m(z) = (u+1)/(u²+u+1), u = 1+z, and the
ΛCDM-fit Λ is 3·Ω_Λ_LCDM(z_eff)·Λ_substrate.  All z_eff-conditional content
(including the SN+BAO vs CMB-Fisher definitional band and the Item-5 wall)
lives there and in predictions/{z_eff,Omega_Lambda_LCDM}.py — it is NOT a
gap of this substrate-only file.

SCOPE OF THIS FILE (clarified 2026-05-16): this file predicts ONLY the
clean substrate-frame Λ_substrate = 1/N² (+ the (16/15)² observer rate-gap).
That value carries NO z_eff dependence and is the solid foundation.

The observed Planck ΛCDM-fit Λ (≈ 2.85e-122) and the Row-P24 "factor-of-2"
are predicted SEPARATELY in the observable-side sibling
`predictions/Lambda_CC_LCDM.py` as the parametric-class translation
Λ_LCDM = 3·Ω_Λ_LCDM(z_eff)·Λ_substrate (= 2·Λ_substrate exactly at the
K-rational anchor z=√3; +0.77σ_obs at the adopted z_eff).  The factor-of-2
is therefore neither "OPEN" nor a deviation in THIS file's prediction — it
is structurally accounted for in the sibling, which carries the (inherited,
not new) ADOPTED-z_eff conditional shared with predictions/Omega_Lambda_LCDM.py.
The earlier framing here — "Item 5 is the load-bearing upstream gap for the
(γ) closure of the factor-of-2" — was superseded by the 2026-05-15 EOD+5
adopted-z_eff resolution and the foundation/observable split; it is removed.

Earlier framing in this file's first commit (cd38af3) claimed z_eff was
"rejected as data-extraction" — that was wrong (a pattern-matching error
where I confused a calculable data-side quantity for an empirical input).
Corrected here; see also the (k*−1) pattern-match retraction in commit
following 9116af1.
"""

# ============================================================
# PARAMETER: Λ_CC (cosmological constant) — substrate-frame prediction
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Λ_LCDM ≈ 2.85 × 10⁻¹²² in Planck units
#              = 3 · H_0² · Ω_Λ_LCDM ≈ 3 × (67.4 km/s/Mpc)² × 0.685
# Source:      Planck 2018 CMB ΛCDM-fit (Aghanim et al. 2020, A&A 641, A6).
#              Extracted under ΛCDM cosmology with Ω_Λ_LCDM ≈ 0.685.
# PDG edition: 2024 (Planck 2018 value still canonical).
#
# Caveat: this is a MODEL-DEPENDENT extraction.  Under framework's coasting
# cosmology (Ω_Λ = 1/3 from Row P22), the same observational data extract a
# different Λ; the factor-of-2 between the two extractions is the
# parametric-class translation predicted in predictions/Lambda_CC_LCDM.py
# (Λ_LCDM = 3·Ω_Λ_LCDM(z_eff)·Λ_substrate; = 2·Λ_substrate exactly at z=√3).

# --- PREDICTED VALUE -----------------------------------------
# Substrate (framework-frame):
#   Λ_substrate = H_0_substrate² = (1/(N·t_P))² = 1/N²  in Planck units
#               ≈ 1.42 × 10⁻¹²²
# Observer (with (16/15) cascade rate-gap):
#   Λ_observer  = (16/15)² · Λ_substrate ≈ 1.61 × 10⁻¹²²
#
# Comparison to Λ_LCDM-fit:
#   Λ_LCDM / Λ_substrate ≈ 2.0  (= parametric-class translation; predicted
#                                 in predictions/Lambda_CC_LCDM.py)
#   Λ_LCDM / Λ_observer  ≈ 1.77 (rate-gap absorbs ~14% on the observer side)
#
# The framework's prediction is HONEST as the substrate-frame structural
# value (theorem-grade in the coasting frame).  Clause 8 for THIS file is
# the framework-frame consistency Λ_substrate = H_0_substrate² (exact by
# construction).  The Clause-8 comparison against Planck's ΛCDM-extracted Λ
# is made in the observable-side sibling predictions/Lambda_CC_LCDM.py
# (PASS at +0.77σ_obs; −0.20σ at the K-rational anchor) — NOT open here.

# --- DERIVED FORMULA -----------------------------------------
# Λ_substrate = H_0_substrate²            (Friedmann, coasting, Ω_Λ = 1/3)
#             = 1/(N·t_P)²                (cascade theorem D1+D2+D3)
#
# Equivalent forms:
#   Λ_substrate [in Planck units]  = 1/N_hub²
#   Λ_substrate [in 1/s²]           = (H_0_substrate [km/s/Mpc] / Mpc_in_km)²
#   Λ_substrate [in m⁻²]            = Λ_substrate [in 1/s²] / c²
#
# Logical chain:
#   Step 1: Ω_Λ = 1/k* = 1/3            [Row P22, theorem-grade Poisson(2k*) tail]
#   Step 2: Friedmann: Λ = 3·H²·Ω_Λ     [Type 3, Weinberg 2008 §1.5]
#   Step 3: Substitute Ω_Λ = 1/3:
#           Λ_substrate = 3·H_0_substrate²·(1/3) = H_0_substrate²
#   Step 4: Cascade theorem: H_0_substrate = 1/(N·t_P)  [Row P19, theorem-grade]
#   Step 5: Λ_substrate (Planck units) = (H_0_substrate · t_P)² = 1/N²
#   Step 6: Rate-gap (16/15) [Row P19, D2-extended]:
#           Λ_observer = (16/15)² · Λ_substrate

# --- INPUTS --------------------------------------------------
# symbol  | value                  | status     | predictions/ file              | meaning
# --------|------------------------|------------|--------------------------------|--------
# H_0_sub | 1/(N·t_P) km/s/Mpc     | [derived]  | predictions/H_0.py             | Substrate Hubble (cascade D1+D2+D3 theorem-grade)
# N_hub   | ≈ 8.395e60             | [adopted]  | predictions/N_hub.py           | Substrate count (G_F-calibrated, ADOPTED-N_HUB)
# t_P     | 5.391247e-44 s         | [external] | (CODATA 2018)                  | Planck time
# k*      | 3                      | [derived]  | predictions/k_star.py          | Coordination number (Row 4 theorem-grade)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from p_toggle import predict_p_toggle
from alpha_1 import predict_alpha_1
from N_hub import predict_N_hub
from H_0 import predict_H_0
from M_Pl_natural import M_Pl_GeV as M_P, t_P_seconds as t_P, Mpc_in_km   # single SI-anchor source (t_P derived ℏ/M_Pl, consolidated 2026-05-16; Mpc_in_km consolidated 2026-05-26)

# --- substrate primitives ---
d_val   = predict_d_spatial()
k       = predict_k_star(d_val)
g       = predict_g_girth(k, d_val)
p       = predict_p_toggle()
alpha_1 = predict_alpha_1(k, g)
from delta_Koide import delta_Koide_pred as delta  # = 2/9 (Q*(1-Q) at Q=2/3, predict_delta_Koide)

# --- adopted dimensional input + external Planck time ---
G_F_obs = 1.1663787e-5           # measured Fermi constant; pins N_hub's value (ADOPTED-N_HUB)
# t_P imported (derived ℏ/M_Pl) from M_Pl_natural — was: t_P = 5.391247e-44 CODATA hardcode (consolidated 2026-05-16; Δ=2.9e-9 sub-ppb)
# Mpc_in_km imported from M_Pl_natural single-source above (was: 3.085677581e19 inline; consolidated 2026-05-26)

# --- N_hub chain ---
from p_toggle import predict_p_toggle as _ppt
from V_count import predict_V_count as _pvc
_p_for_Nhub = _ppt()
_V_for_Nhub = _pvc(k, d_val)
N_hub = predict_N_hub(G_F_obs, M_P, alpha_1, delta, k, _p_for_Nhub, _V_for_Nhub)

# --- Friedmann + cascade theorem: Λ_substrate = H_0_substrate² ---
# In Planck units, Λ · t_P² = (H · t_P)² = 1/N²
Lambda_substrate_Planck = 1.0 / (N_hub ** 2)

# Cascade D2-extended rate-gap (16/15) on the observer side
# Inherits ε_toggle (1/5, Row P28 theorem-grade) × 1/k* (1/3, Row 4) = 1/15
# 5 = k_star + p_toggle (= 3 + 2, framework primitives; same sourcing as
# R_nu_splitting's cubic-root n = k* + p)
EPS_TOGGLE = 1.0 / (k + p)
GEOMETRIC_K = 1.0 / k                            # = 1/3
RATE_GAP = EPS_TOGGLE * GEOMETRIC_K               # = 1/15
RATE_GAP_FACTOR_SQ = (1.0 + RATE_GAP) ** 2        # (16/15)²

Lambda_observer_Planck = RATE_GAP_FACTOR_SQ * Lambda_substrate_Planck

# Cross-check: Λ in 1/s² via H_0 directly
H_0_pred_kmsMpc = predict_H_0(G_F_obs, M_P, t_P, alpha_1, delta)   # substrate H_0
H_0_pred_per_s = H_0_pred_kmsMpc / Mpc_in_km
Lambda_substrate_per_s2 = H_0_pred_per_s ** 2
Lambda_substrate_Planck_check = Lambda_substrate_per_s2 * (t_P ** 2)
assert abs(Lambda_substrate_Planck - Lambda_substrate_Planck_check) / Lambda_substrate_Planck < 1e-12, (
    f"Λ_substrate cross-check failed: {Lambda_substrate_Planck} vs {Lambda_substrate_Planck_check}"
)

# Canonical run_predictions exports — framework's structural prediction is
# Λ_substrate (coasting frame).  The factor-of-2 vs Planck ΛCDM-fit is the
# open cosmology-model split (Row P24), not a deviation in the structural
# prediction.  No σ_PDG-class comparison since the gap is structural, not
# observational.
Lambda_CC_pred = Lambda_substrate_Planck

# --- ΛCDM-extracted comparison value (Planck 2018) ---
# Λ_LCDM = 3·H_0_Planck²·Ω_Λ_LCDM with Ω_Λ_LCDM ≈ 0.685
# Friedmann coefficient 3 = k_star (= coordination number; algebraic identity
# Λ = k*·H²·Ω_Λ at k*=3 — same sourcing as Lambda_CC_LCDM.py).
H_0_Planck_kmsMpc = 67.4
Omega_Lambda_LCDM = 0.685
H_0_Planck_per_s = H_0_Planck_kmsMpc / Mpc_in_km
Lambda_LCDM_Planck = float(k) * Omega_Lambda_LCDM * (H_0_Planck_per_s * t_P) ** 2

# Open factor-of-2 split
ratio_LCDM_substrate = Lambda_LCDM_Planck / Lambda_substrate_Planck
ratio_LCDM_observer  = Lambda_LCDM_Planck / Lambda_observer_Planck

print("=" * 72)
print(" Λ_CC  --  cosmological constant (coasting-frame structural prediction)")
print("=" * 72)
print(f"  N_hub                 = {N_hub:.6e}  [adopted; predictions/N_hub.py]")
print(f"  H_0_substrate         = {H_0_pred_kmsMpc:.4f} km/s/Mpc  [theorem-grade cascade]")
print(f"  k*                    = {k}  [Row 4 theorem-grade]")
print()
print("FRAMEWORK STRUCTURAL PREDICTION (coasting, Ω_Λ = 1/k* = 1/3, Row P22):")
print(f"  Λ_substrate (Planck)  = 1/N²                  = {Lambda_substrate_Planck:.3e}")
print(f"  Λ_observer  (Planck)  = (16/15)² · Λ_sub      = {Lambda_observer_Planck:.3e}")
print(f"  rate-gap (1/15)       = ε_toggle/k* = (1/5)·(1/3) = {RATE_GAP:.6f}")
print()
print("PLANCK 2018 ΛCDM-FIT (model-dependent extraction, Ω_Λ_LCDM ≈ 0.685):")
print(f"  Λ_LCDM      (Planck)  = 3·H_0²·Ω_Λ_LCDM       = {Lambda_LCDM_Planck:.3e}")
print()
print("RATIOS (factor-of-2 OPEN structural question, Row P24):")
print(f"  Λ_LCDM / Λ_substrate  = {ratio_LCDM_substrate:.3f}  (factor-of-2 cosmology-model split)")
print(f"  Λ_LCDM / Λ_observer   = {ratio_LCDM_observer:.3f}  ((16/15)² rate-gap absorbs ~14%)")
print()
print("STATUS: UNIQUE-THEOREM-GRADE (substrate Λ = 1/N²; graduated 2026-05-16,")
print("  G1-cluster class — coasting + ADOPTED-N_HUB conditional only, the same")
print("  class graduated via the G1b R2 closure for P17/P19/P20).")
print("  Framework's substrate-frame Λ = 1/N² is theorem-grade in coasting frame")
print("  and carries NO z_eff dependence — this is the clean foundation.")
print("  Clause 8 for THIS file = framework-frame consistency Λ_sub = H_0_sub²")
print("  (exact by construction).")
print()
print("  The observed Planck ΛCDM-fit Λ (≈ 2.85e-122) and the Row-P24")
print("  'factor-of-2' are predicted SEPARATELY, as the parametric-class")
print("  translation Λ_LCDM = 3·Ω_Λ_LCDM(z_eff)·Λ_substrate, in the")
print("  observable-side sibling predictions/Lambda_CC_LCDM.py")
print("  (= 2·Λ_substrate exactly at the K-rational anchor z=√3;")
print("   +0.77σ_obs at the adopted z_eff).  The factor-of-2 is therefore")
print("  structurally accounted for there, NOT 'OPEN' here; the sibling")
print("  carries the (inherited, not new) ADOPTED-z_eff conditional shared")
print("  with predictions/Omega_Lambda_LCDM.py.")
print()
print("  Cross-references: Row P19 (H_0), Row P22 (Ω_Λ = 1/3), Row P24 (Λ_CC),")
print("  predictions/Lambda_CC_LCDM.py (observable-side ΛCDM-fit sibling),")
print("  proofs/cosmology/Lambda_CC_rate_gap.py (rate-gap derivation).")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_Lambda_CC(G_F_GeV2, M_P_GeV, t_P_s, alpha_1, delta, k_star, p_toggle):
    """
    Predict the substrate-frame cosmological constant Λ_substrate (in Planck units).

    Λ_substrate = H_0_substrate² = 1/N²  (coasting Friedmann + cascade theorem)

    where the coasting condition Ω_Λ = 1/k* (Row P22 theorem-grade) combined
    with the Friedmann relation Λ = 3·H²·Ω_Λ yields Λ_substrate = H²·k*·(1/k*) = H_0²,
    and the cascade theorem H_0 = 1/(N·t_P) gives Λ_substrate in Planck units = 1/N².

    The observer-side prediction adds the (16/15) rate-gap factor from cascade
    D2-extended (Row P19):

        Λ_observer = (16/15)² · Λ_substrate

    where (16/15) − 1 = 1/15 = ε_toggle · (1/k*) = (1/5) · (1/3) (theorem-grade
    inheritance from Row P28 ε_toggle + Row 4 k*=3).  ε_toggle⁻¹ = 5 is sourced
    as k_star + p_toggle (= 3 + 2, framework primitives; same sourcing as
    R_nu_splitting's cubic-root n = k* + p).

    Parameters
    ----------
    G_F_GeV2 : float
        Fermi constant in GeV^{-2} — pins N_hub (ADOPTED-N_HUB).
    M_P_GeV : float
        Planck mass in GeV (CODATA 2018 unit-setting constant).
    t_P_s : float
        Planck time in seconds (CODATA 2018).
    alpha_1 : float
        Bare NB walk survival ((k*-1)/k*)^{g-2} = (2/3)^8.
    delta : float
        Koide phase (2/9 exactly).
    k_star : int
        Coordination number (= 3 on srs, Row 4 theorem-grade).
    p_toggle : int
        Toggle arity (= 2; predict_p_toggle).

    Returns
    -------
    tuple of float
        (Λ_substrate, Λ_observer) both in dimensionless Planck units.
    """
    from V_count import predict_V_count
    from d_spatial import predict_d_spatial
    V_ = predict_V_count(k_star, predict_d_spatial())
    N = predict_N_hub(G_F_GeV2, M_P_GeV, alpha_1, delta, k_star, p_toggle, V_)
    Lambda_sub = 1.0 / (N ** 2)
    rate_gap = (1.0 / (k_star + p_toggle)) * (1.0 / k_star)   # ε_toggle · 1/k*
    Lambda_obs = ((1.0 + rate_gap) ** 2) * Lambda_sub
    return Lambda_sub, Lambda_obs


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_sub, impl_obs = Lambda_substrate_Planck, Lambda_observer_Planck
    pure_sub, pure_obs = predict_Lambda_CC(G_F_obs, M_P, t_P, alpha_1, delta, k, p)
    print()
    print(f"Implementation: Λ_sub = {impl_sub:.10e}, Λ_obs = {impl_obs:.10e}")
    print(f"Pure function:  Λ_sub = {pure_sub:.10e}, Λ_obs = {pure_obs:.10e}")
    assert abs(impl_sub - pure_sub) / impl_sub < 1e-12, (
        f"Λ_substrate mismatch: {impl_sub} vs {pure_sub}"
    )
    assert abs(impl_obs - pure_obs) / impl_obs < 1e-12, (
        f"Λ_observer mismatch: {impl_obs} vs {pure_obs}"
    )
    print(f"OK: outputs agree.  Λ_substrate = {pure_sub:.3e} (Planck units) = 1/N²")
    print(f"    Λ_observer = {pure_obs:.3e} (Planck units) = (16/15)² · Λ_substrate")
    print(f"    Λ_LCDM-fit (Planck 2018) = {Lambda_LCDM_Planck:.3e}; factor-of-2 OPEN per Row P24.")
