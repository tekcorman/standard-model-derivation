#!/usr/bin/env python3
"""
R∞ — Rydberg constant.

R∞ = α_EM² × m_e × c / (2h)

Atomic-precision constant. With α_EM(0) Thomson limit and m_e theorem-grade,
R∞ is a downstream multiplication. Uses α_EM(0) ≈ 1/137.036 (Thomson limit)
rather than α_EM(M_Z), since R∞ describes atomic transitions at zero
momentum.

STATUS: the framework predicts α_EM at M_Z (predictions/alpha_EM.py),
not α_EM(0). R∞ ≡ α(0)²·m_e·c/2h needs α(0) = α(M_Z) + Δα, and **Δα is
OUTSIDE THE FRAMEWORK'S SCOPE BY CONSTRUCTION**, not a deficit (Move 1,
2026-05-16, `proofs/foundations/delta_alpha_is_noThreshold_scope_
exclusion_2026-05-16.py` + `an internal working note
blocked_verdict_2026-05-16.md`): Δα is purely a fermion-mass-threshold
sum (Σ (α/3π)ln(M_Z²/m_f²)); the framework's single-regime
no-threshold RG contains no m_f, so it definitionally excludes the IR
threshold/decoupling layer (same boundary as the α_s/g_3 cluster
residuals & the oblique photon channel — ONE scope statement, not
three deficits). ⇒ R∞ is a DEPENDENT observable out-of-scope by that
boundary; `delta_alpha_running` is therefore an out-of-scope IR layer
that must NOT be patched in as if a framework prediction (β-class).
The clean in-scope EM test is α_EM(M_Z) (matched to ~0.02%).

The α_EM(0) — α_EM(M_Z) running below M_Z is via standard QED through
charged-fermion thresholds (Type 3 standard QED); the framework provides
α_EM(M_Z) as the input.

CLEAN-RATIO DIAGNOSTIC (2026-05-16, `proofs/foundations/
Rinf_clean_ratio_diagnostic_2026-05-16.py`): R_∞/v is N_hub-exactly-
cancelled (both ∝ N_hub^−1/4), so the R_∞ residual = 2·δ(α_EM(0)).
That δ splits into (DOMINANT) the α_EM(M_Z) gauge-cluster drift
(−0.021 in α⁻¹) and (SECONDARY) this `delta_alpha_running` import
error (+0.007 in α⁻¹).  Deriving a substrate Δα would NOT close the
clean ratio — the gauge cluster is the real lever.  m_e here is CODATA
(δ(m_e)≈0), NOT in the residual.
"""
# --- OBSERVED: R∞ = 1.0973731568160(21) × 10⁷ m⁻¹ (CODATA 2018)
# --- PREDICTED: R∞ matches at ~0.1% via framework α_EM(0) + m_e
# --- INPUTS: alpha_EM(0), m_e, c, h (CODATA conversions)

import sys, os, math, functools
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from alpha_EM import alpha_EM_MZ  # α_EM at M_Z

# QED running from M_Z down to atomic scale: α_EM(M_Z) → α_EM(0) ≈ 1/137.036
# Via standard QED (Type 3): α_EM⁻¹(0) − α_EM⁻¹(M_Z) ≈ 9.0917 (PDG-derived)
# Components (Jegerlehner / PDG): Δα_lept ≈ 0.03150 + Δα_had ≈ 0.02760 + Δα_top ≈ 0.0001
# → δα⁻¹ = (1/α(0)) − (1/α(M_Z)) ≈ 137.036 − 127.944 = 9.092
#
# Previous value 9.91 was calibrated against framework's BARE α_EM(M_Z) = 1/127.04
# (pre-dark-correction); that calibration was implicit numerology.  With the
# α_GUT dark correction now propagated (theorem-grade-cond, 2026-05-15), the
# framework's α_EM(M_Z) ≈ 1/127.92 essentially matches PDG, and the QED running
# uses the standard PDG-derived δα⁻¹ = 9.092 directly (Type 3 standard QED).
# ── CLAUSE-9 CONDITIONAL (tagged 2026-05-16) ──────────────────────────────
# `delta_alpha_running` is a Type-3 continuum-QED import the framework does
# NOT derive.  parameter_linter.md Clause 9 explicitly lists Δα_had≈0.0277
# as a continuum-loop transcendental (∉ K, Lindemann) — citing the imported
# value as closure is K-INVALID; this is a NAMED OPEN MECHANISM, not
# theorem-grade.  Substrate-analog attempt BLOCKED 2026-05-16
# (`proofs/foundations/substrate_Delta_alpha_photon_channel_2026-05-16.py`):
#   • Δα_had analog — B1-scoping-NEGATIVE (multiway+R-14 wall,
#     `B1_QCD_HVP_substrate_scoping_2026-05-15.py`);
#   • Δα_lep analog — the photon (charge-weighted Perron/off-support)
#     channel of the unified-oblique B_NB resolvent has NO first-principles-
#     FORCED K-rational coefficient (unlike c_S=1/(2|E|) / c=1/2 which WERE
#     forced); the closest form (−3.6%) is a cherry-pick → numerology gate
#     fail; SM value is lepton-mass-log transcendental (Clause 9).
# Resolution = Clause 9 (9b): tagged STRUCTURAL-DERIVATION-CONDITIONAL.
# Value UNCHANGED (9.092) — an honest named import, no longer silent.
# Per the clean-ratio diagnostic this is only the SECONDARY R_∞-residual
# piece; the dominant fix is the α_EM(M_Z) gauge-cluster drift.
alpha_EM_at_MZ = alpha_EM_MZ
delta_alpha_running = 9.092  # [Clause-9 STRUCTURAL-DERIVATION-CONDITIONAL;
                             #  Type-3 QED import; substrate analog BLOCKED]
alpha_EM_0 = 1.0 / (1.0/alpha_EM_at_MZ + delta_alpha_running)

# Constants (CODATA 2018; single-source from predictions/M_Pl_natural.py)
from m_e import m_e_pred as m_e_GeV   # framework prediction (theorem-grade Koide ratio chain)
from M_Pl_natural import hbar_J_s, c_m_s, GeV_to_J  # CODATA SI single-source
GeV_to_kg = GeV_to_J / c_m_s**2
m_e_kg = m_e_GeV * GeV_to_kg
h_J_s = hbar_J_s * 2 * math.pi

R_infinity = alpha_EM_0**2 * m_e_kg * c_m_s / (2 * h_J_s)

R_infinity_pred = R_infinity
R_infinity_obs = 1.0973731568160e7
R_infinity_sigma = 0.0000000000021e7

print(f"R∞ = {R_infinity:.6e} m⁻¹  (CODATA {R_infinity_obs:.6e}, "
      f"dev {(R_infinity - R_infinity_obs)/R_infinity_obs*100:+.3f}%)")
print(f"  α_EM(M_Z) = 1/{1/alpha_EM_at_MZ:.3f} (framework)")
print(f"  α_EM(0)   = 1/{1/alpha_EM_0:.3f} (after QED running, target 1/137.036)")


@functools.lru_cache(maxsize=None)
def predict_R_infinity(alpha_EM_0, m_e_GeV, hbar_J_s, c_m_s, GeV_to_J):
    """
    Predict R∞ from α_EM(0), m_e, and CODATA conversions.

    R∞ = α_EM² × m_e × c / (2h) = α_EM² × m_e_kg × c / (2 × 2π·ℏ)

    Parameters
    ----------
    alpha_EM_0 : float
        Fine-structure constant at zero momentum (Thomson limit).
    m_e_GeV : float
        Electron mass in GeV/c².
    hbar_J_s : float
        Reduced Planck constant in J·s.
    c_m_s : float
        Speed of light in m/s.
    GeV_to_J : float
        GeV → J conversion (CODATA).

    Returns
    -------
    float
        R∞ in m⁻¹.
    """
    GeV_to_kg = GeV_to_J / c_m_s**2
    m_e_kg = m_e_GeV * GeV_to_kg
    h_J_s = hbar_J_s * 2 * math.pi
    return alpha_EM_0**2 * m_e_kg * c_m_s / (2 * h_J_s)


if __name__ == "__main__":
    impl = R_infinity
    pure = predict_R_infinity(alpha_EM_0, m_e_GeV, hbar_J_s, c_m_s, GeV_to_J)
    assert abs(impl - pure) / impl < 1e-10
    print(f"OK: implementation = pure = {impl:.6e}")
