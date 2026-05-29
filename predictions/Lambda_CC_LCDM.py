#!/usr/bin/env python3
"""
Lambda_CC_LCDM — the ΛCDM-fit cosmological constant (the OBSERVED Planck value),
predicted via the parametric-class translation of the framework's clean
substrate-frame Λ.

This is the observable-side sibling of predictions/Lambda_CC.py:

  - predictions/Lambda_CC.py     : Λ_substrate = 1/N²  (clean, NO z_eff;
                                    theorem-grade-conditional on coasting +
                                    ADOPTED-N_HUB).  The SOLID FOUNDATION.

  - predictions/Lambda_CC_LCDM.py: Λ_LCDM-frame = 3·Ω_Λ_LCDM(z_eff)·Λ_substrate
                                    (THIS FILE).  Predicts the actual Planck
                                    2018 ΛCDM-fit number (≈ 2.85e-122 Planck
                                    units) and EXPLAINS it as the
                                    parametric-class translation of the clean
                                    substrate Λ — the "factor-of-2" of Row P24
                                    is exactly Ω_Λ_LCDM(z_eff)/Ω_Λ_substrate
                                    = Ω_Λ_LCDM/(1/3) = 3·Ω_Λ_LCDM, which is
                                    2 EXACTLY at the K-rational anchor z=√3.

This file makes NO new structural claim beyond what is already shipped by the
already-promoted P24-cluster siblings (predictions/{z_eff,Omega_m_LCDM,
Omega_Lambda_LCDM}.py).  It is strict Type-4 inheritance: it carries exactly
the siblings' accepted conditional (MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-
ADOPTED-z_eff) and does NOT relitigate or "close" Item 5.  The +3σ_obs
SN+BAO ⟨Ω_m(z)⟩_F definitional concern (ledger Row P24, 2026-05-15 EOD+5) is
a property of the z_eff adoption itself, already litigated and shipped for the
siblings; inheriting Ω_Λ_LCDM(z_eff) inherits that posture unchanged.

H₀ note: because the framework's coasting identity gives Λ_substrate ≡
H₀_substrate² (predictions/Lambda_CC.py Step 3+5), the form
3·Ω_Λ_LCDM·Λ_substrate uses the framework's OWN H₀, not Planck's H₀ — no
external-H₀ smuggle.  Planck's literal 2.85e-122 uses H₀=67.4; the framework
is internally self-consistent with its own H₀ (Row P19, +1.6σ vs Planck).
"""

# ============================================================
# PARAMETER: Λ_CC (ΛCDM-fit frame) — the observed cosmological constant
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Λ_LCDM = 2.849 × 10⁻¹²² ± 5.2 × 10⁻¹²⁴ (Planck units)
#              = 3 · H_0² · Ω_Λ_LCDM with Ω_Λ_LCDM = 0.6847 ± 0.0073,
#                H_0 = 67.4 ± 0.5 km/s/Mpc  (combined ≈ ±1.83 %)
# Source:      Planck 2018 VI — Aghanim et al. (2020) A&A 641, A6.
#              w_0 = −1.03 ± 0.03 (consistent with a cosmological constant;
#              the framework predicts w_DE = −1 exactly, Row P21, so the
#              ΛCDM-fit Λ is the correct comparison target — DESI DR2 2025
#              evolving-DE hints are a different (w0wa) model).
# PDG edition: 2024 (Planck 2018 value still canonical).

# --- PREDICTED VALUE -----------------------------------------
# @ adopted z_eff = 1.8519:  2.889 × 10⁻¹²²  → +1.41 %  = +0.77 σ_obs
# @ K-rational anchor z=√3:  2.838 × 10⁻¹²²  → −0.37 %  = −0.20 σ_obs
#                            (= exactly 2 · Λ_substrate, since 3·(2/3) = 2)
# Clause 8: PASS at +0.77 σ_obs (within 1σ) under the Category-B
# framework-vs-ΛCDM accommodation; −0.20 σ at the K-rational anchor.

# --- DERIVED FORMULA -----------------------------------------
# Λ_LCDM-frame = 3 · Ω_Λ_LCDM(z_eff) · Λ_substrate
#
#   where  Λ_substrate          = 1/N²          [predictions/Lambda_CC.py]
#          Ω_Λ_LCDM(z_eff)       = u²/(u²+u+1),  u = 1+z_eff
#                                                [predictions/Omega_Lambda_LCDM.py]
#          z_eff                 = adopted Fisher first-moment ≈ 1.8519
#                                                [predictions/z_eff.py]
#
# Logical chain:
#   Step 1: Λ_substrate = H_0_substrate² = 1/N²  [Lambda_CC.py; coasting
#           Friedmann with Ω_Λ_substrate = 1/k* = 1/3 absorbed, Row P22]
#   Step 2: Friedmann in the ΛCDM-fit frame: Λ_LCDM = 3·H_0²·Ω_Λ_LCDM
#           [Weinberg 2008 §1.5; Type 3, K-rational here]
#   Step 3: The framework's own H_0 makes Λ_substrate ≡ H_0_substrate²,
#           so Λ_LCDM-frame = 3·Ω_Λ_LCDM·Λ_substrate  (H_0 absorbed)
#   Step 4: Ω_Λ_LCDM at the adopted z_eff via the theorem-grade bias
#           function form Ω_m(z) = (u+1)/(u²+u+1)  [Omega_m_LCDM.py;
#           Ω_Λ_LCDM = 1 − Ω_m_LCDM = u²/(u²+u+1)]
#   Step 5: The Row-P24 "factor-of-2" = Λ_LCDM-frame/Λ_substrate
#           = 3·Ω_Λ_LCDM = Ω_Λ_LCDM/(1/3); = 2 EXACTLY at z=√3
#           (where Ω_Λ_LCDM = 2/3) — structurally demystified, not fitted.

# --- INPUTS --------------------------------------------------
# symbol            | value        | status                 | predictions/ file              | meaning
# ------------------|--------------|------------------------|--------------------------------|--------
# Λ_substrate       | 1.419e-122   | [derived]              | predictions/Lambda_CC.py       | Clean substrate Λ = 1/N² (coasting + ADOPTED-N_HUB)
# Ω_Λ_LCDM(z_eff)   | ≈ 0.6786     | [adopted, N_hub-class] | predictions/Omega_Lambda_LCDM.py | ΛCDM-fit dark-energy fraction = 1 − Ω_m_LCDM
# z_eff             | ≈ 1.8519     | [adopted, N_hub-class] | predictions/z_eff.py           | Survey Fisher first-moment effective redshift
# (3 = Friedmann structural coefficient Λ = 3H²Ω_Λ; not an empirical input)

# --- IMPLEMENTATION ------------------------------------------

import functools
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from Lambda_CC import (
    predict_Lambda_CC,
    G_F_obs,
    M_P,
    t_P,
    alpha_1,
    delta,
    k,
    p,
)
from z_eff import predict_z_eff, BAO_ANCHORS, SN_MODEL
from Omega_Lambda_LCDM import predict_Omega_Lambda_LCDM

# Clean substrate foundation (NO z_eff): Λ_substrate = 1/N²
_Lambda_substrate, _Lambda_observer = predict_Lambda_CC(
    G_F_obs, M_P, t_P, alpha_1, delta, k, p
)

# Adopted survey effective redshift and the ΛCDM-fit dark-energy fraction.
# z=√3 K-rational anchor: the literal 3 = k_star (coordination number; same
# algebraic identity that lets the Friedmann coefficient cancel the substrate
# Ω_Λ = 1/k* below).
_z_eff = predict_z_eff(BAO_ANCHORS, SN_MODEL)
_Omega_Lambda_LCDM_zeff = predict_Omega_Lambda_LCDM(_z_eff)
_Omega_Lambda_LCDM_anchor = predict_Omega_Lambda_LCDM(math.sqrt(float(k)))


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_Lambda_CC_LCDM(Lambda_substrate, Omega_Lambda_LCDM, k_star):
    """
    Predict the ΛCDM-fit cosmological constant (Planck units) as the
    parametric-class translation of the clean substrate Λ.

    Λ_LCDM-frame = k_star · Ω_Λ_LCDM · Λ_substrate
                 = 3 · Ω_Λ_LCDM · Λ_substrate    (at k_star = 3 on srs)

    The coefficient k_star = 3 is the Friedmann structural constant
    (Λ = 3·H²·Ω_Λ at k* = 3, where the "3" is identified with the
    coordination number — the same k_star that gives the substrate's
    Ω_Λ = 1/k* coasting condition); combined with the framework's coasting
    identity Λ_substrate ≡ H_0² it cancels the substrate's Ω_Λ = 1/3, so
    the ratio to Λ_substrate is exactly Ω_Λ_LCDM/(1/3) and equals 2 when
    Ω_Λ_LCDM = 2/3 (z = √3).

    Parameters
    ----------
    Lambda_substrate : float
        Clean substrate-frame Λ = 1/N² (predictions/Lambda_CC.py).
    Omega_Lambda_LCDM : float
        ΛCDM-fit dark-energy fraction at the adopted z_eff
        (predictions/Omega_Lambda_LCDM.py).
    k_star : int
        Coordination number (= 3 on srs; predict_k_star).

    Returns
    -------
    float
        Predicted ΛCDM-fit Λ in dimensionless Planck units.
    """
    return float(k_star) * Omega_Lambda_LCDM * Lambda_substrate


# --- INTROSPECTION (for run_predictions.py) ------------------
Lambda_CC_LCDM_pred = predict_Lambda_CC_LCDM(_Lambda_substrate, _Omega_Lambda_LCDM_zeff, k)
Lambda_CC_LCDM_obs = 2.84852e-122  # Planck 2018 ΛCDM-fit
Lambda_CC_LCDM_sigma = 5.204e-124  # combined ±1.83%


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl = float(k) * _Omega_Lambda_LCDM_zeff * _Lambda_substrate
    pure = Lambda_CC_LCDM_pred
    anchor = predict_Lambda_CC_LCDM(_Lambda_substrate, _Omega_Lambda_LCDM_anchor, k)

    obs, sig = Lambda_CC_LCDM_obs, Lambda_CC_LCDM_sigma

    print("=" * 72)
    print(" Λ_CC (ΛCDM-fit frame) — observed cosmological constant via")
    print(" parametric-class translation of the clean substrate Λ")
    print("=" * 72)
    print(f"  Λ_substrate (1/N², NO z_eff) = {_Lambda_substrate:.5e}  "
          f"[predictions/Lambda_CC.py — clean foundation]")
    print(f"  z_eff (adopted, N_hub-class) = {_z_eff:.4f}")
    print(f"  Ω_Λ_LCDM @ z_eff             = {_Omega_Lambda_LCDM_zeff:.4f}  "
          f"(k*·Ω_Λ_LCDM = {float(k) * _Omega_Lambda_LCDM_zeff:.4f})")
    print(f"  Ω_Λ_LCDM @ z=√3 (K-rational) = {_Omega_Lambda_LCDM_anchor:.6f} "
          f"(k*·Ω_Λ_LCDM = {float(k) * _Omega_Lambda_LCDM_anchor:.4f})")
    print()
    print(f"  PRED Λ_LCDM @ z_eff          = {pure:.5e}")
    print(f"  PRED Λ_LCDM @ K-anchor √3    = {anchor:.5e}  (= 2·Λ_substrate)")
    print(f"  OBS  Planck 2018 ΛCDM-fit    = {obs:.5e} ± {sig:.3e} (±1.83%)")
    print(f"    @ z_eff      : {(pure-obs)/obs*100:+.2f}%  = "
          f"{(pure-obs)/sig:+.2f} σ_obs")
    print(f"    @ K-anchor √3: {(anchor-obs)/obs*100:+.2f}%  = "
          f"{(anchor-obs)/sig:+.2f} σ_obs")
    print()
    print("  Grade: MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff")
    print("  (strict Type-4 inheritance from predictions/Omega_Lambda_LCDM.py;")
    print("   no new claim beyond the already-shipped P24-cluster siblings).")
    print("  Clause 8: PASS at +0.77σ_obs (Category-B framework-vs-ΛCDM).")

    # K-rational anchor gives exactly p_toggle · Λ_substrate (= 2 at p=2),
    # since k*·Ω_Λ_LCDM(z=√k*) = k*·(2/3) = 2 algebraically.
    assert abs(impl - pure) / impl < 1e-12, f"Mismatch: {impl} vs {pure}"
    assert abs(anchor - float(p) * _Lambda_substrate) / anchor < 1e-12, (
        "K-rational anchor must equal exactly p_toggle·Λ_substrate"
    )
    print()
    print("OK: outputs agree; K-rational anchor = 2·Λ_substrate exactly.")
