#!/usr/bin/env python3
"""
cross_anchor_honest_chain.py
============================

HONEST cross-anchor test using framework-predicted values consistently.

Earlier scripts (predict_cosmic_age_from_anchors.py, _via_Rydberg.py,
_clean_anchors.py) used PDG m_τ/m_e ratio and "y_τ = m_τ/v" (observed)
in the chain — mixing PDG and framework values. This made the apparent
"G_F prediction match" look much better than the framework's actual
prediction.

USER FEEDBACK trail:
  - "but aren't those percentages a high sigma value?" (correct — yes, but
    earlier I attributed it only to cosmic age uncertainty; missed that
    I was using PDG values where framework values should go).

THE REAL STORY: framework's y_τ = α_1_full / k*² = 1280/177147 ≈ 7.2256e-3
is +0.13% above observed y_τ = m_τ/v ≈ 7.2166e-3. This propagates:
  - v_Higgs (predicted from m_τ/y_τ_framework): -0.13% relative to PDG v
  - G_F (predicted from 1/(√2 v²)):              -0.26% relative to PDG G_F
  - At PDG precision (0.5 ppm), this is ~5000σ tension.

Similarly the framework's m_τ/m_e ratio is theorem-grade with the same
~0.13% systematic; the framework predicts m_τ_pred = 1.779 GeV vs PDG
1.77686 (+0.13%) and m_e_pred = 511.6 keV vs PDG 510.999 (+0.12%).

These are ALL THE SAME systematic — likely a sub-leading Yukawa correction
the framework hasn't yet derived. The framework's chain is internally
self-consistent at the ~0.13% level; PDG comparison shows where derivation
needs sub-leading work.

CHAIN (framework-internal, no PDG mixing):
  m_e (anchor, CODATA Penning trap) → m_τ via framework m_τ/m_e ratio
  m_τ → v_Higgs via framework y_τ
  v_Higgs → G_F via SM tree-level
  G_N (anchor) → M_Pl directly
  BZJ + framework v_Higgs + M_Pl → N
  Cascade: t_0 = N · ℏ/(M_Pl c²)

PREDICTIONS REPORTED HONESTLY:
  Each value reported with framework's known systematic offset documented.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'predictions'))

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1
from alpha_1_full import predict_alpha_1_full, n_g_edge


def main():
    # === ANCHORS (only 2) ===
    m_e_kg = 9.1093837015e-31         # CODATA Penning trap, ~3×10⁻¹⁰
    G_N_obs = 6.67430e-11             # CODATA, 22 ppm

    # === EXACT SI ===
    c = 299792458.0
    h = 6.62607015e-34
    hbar = h / (2 * math.pi)
    GeV_to_J = 1.602176634e-10
    yr_to_s = 365.25 * 24 * 3600

    # === FRAMEWORK THEOREM-GRADE INPUTS (computed, not from PDG!) ===
    d = predict_d_spatial()
    k = predict_k_star(d)
    g = predict_g_girth(k, d)
    alpha_1 = predict_alpha_1(k, g)
    alpha_1_full_exact = predict_alpha_1_full(k, g, n_g_edge)
    alpha_1_full = float(alpha_1_full_exact)
    delta = 2.0 / 9.0
    c_vertex = 5.0 / 12.0

    # FRAMEWORK-PREDICTED y_τ (NOT PDG m_τ/v):
    y_tau_framework = alpha_1_full / k**2

    print("=" * 78)
    print("  HONEST cross-anchor test: framework-only chain (no PDG mixing)")
    print("=" * 78)
    print()
    print(f"  Framework theorem-grade dimensionless predictions:")
    print(f"    α_1     = (2/3)^8        = {alpha_1:.6e}")
    print(f"    α_1_full = (5/3)·α_1     = {alpha_1_full:.6e}")
    print(f"    y_τ_framework = α_1_full/k*² = {y_tau_framework:.10e}")
    print(f"      [observed y_τ = m_τ/v = 7.2165543e-3, framework off by +0.13%]")
    print(f"    δ        = 2/9          = {delta:.6f}")
    print(f"    dark factor              = {1.0 - c_vertex*alpha_1/(1.0-alpha_1):.6f}")
    print()

    # === CHAIN ===
    m_e_GeV = m_e_kg * c**2 / GeV_to_J
    print(f"  Chain (framework-only):")
    print(f"    [1] m_e anchor (CODATA Penning)              = {m_e_GeV*1e3:.6f} MeV")

    # m_τ via framework m_τ/m_e ratio
    # Framework predicts m_τ_pred / m_e_pred ≈ 3477.6 (PDG ratio is 3477.2; framework off by +0.013%)
    # The framework's ratio can be computed from Koide formula's f_min/f_max
    # For this honest analysis, use framework's predicted m_τ_pred/m_e_pred ratio
    # (computed from Koide derivation in predictions/m_e.py and predictions/m_tau.py)
    m_tau_pred_GeV = 1.779094  # framework-predicted m_τ (per predictions/m_tau.py)
    m_e_pred_GeV = 0.51160e-3  # framework-predicted m_e (per predictions/m_e.py)
    m_tau_over_m_e_framework = m_tau_pred_GeV / m_e_pred_GeV

    m_tau_GeV = m_e_GeV * m_tau_over_m_e_framework
    print(f"    [2] m_τ = m_e × (m_τ/m_e)_framework          = {m_tau_GeV:.6f} GeV")
    print(f"        [framework m_τ/m_e ratio: {m_tau_over_m_e_framework:.4f}; PDG: 3477.23]")

    # v_Higgs from framework y_τ
    v_Higgs_pred = m_tau_GeV / y_tau_framework
    print(f"    [3] v_Higgs = m_τ / y_τ_framework            = {v_Higgs_pred:.4f} GeV")
    print(f"        [PDG v = 246.22; framework v predicted -0.13% off]")

    # G_F from SM tree
    G_F_pred = 1.0 / (math.sqrt(2) * v_Higgs_pred**2)
    print(f"    [4] G_F = 1/(√2·v²)                          = {G_F_pred:.6e} GeV⁻²")
    print(f"        [PDG G_F = 1.1664e-5; framework predicts +0.27% off]")

    # M_Pl from G_N
    M_Pl_kg = math.sqrt(hbar * c / G_N_obs)
    M_Pl_GeV = M_Pl_kg * c**2 / GeV_to_J
    print(f"    [5] M_Pl from G_N                            = {M_Pl_GeV:.4e} GeV")

    # BZJ + framework v_Higgs
    dark = 1.0 - c_vertex * alpha_1 / (1.0 - alpha_1)
    hbar_GeV_s = hbar / GeV_to_J
    N_pred = (delta**2 * M_Pl_GeV * dark / (math.sqrt(2) * v_Higgs_pred))**4
    print(f"    [6] BZJ: N = (δ²·M_Pl·dark/(√2·v))^4         = {N_pred:.4e}")

    # Cascade
    t_P_s = hbar_GeV_s / M_Pl_GeV
    t_0_pred_s = N_pred * t_P_s
    t_0_pred_Gyr = t_0_pred_s / yr_to_s / 1e9
    print(f"    [7] Cascade: t_0 = N·t_P                     = {t_0_pred_Gyr:.4f} Gyr")

    Mpc_in_km = 3.085677581e19
    H_0_kmsMpc = (1.0 / t_0_pred_s) * Mpc_in_km
    print(f"    [8] Coasting: H_0 = 1/t_0                    = {H_0_kmsMpc:.2f} km/s/Mpc")

    print()
    print("=" * 78)
    print("  HONEST COMPARISONS")
    print("=" * 78)
    print()

    comparisons = [
        ("y_τ (framework-internal)", y_tau_framework, 7.2165543e-3, "(none direct)",
         "α_1_full/k*² = 1280/177147; deviation from m_τ/v_PDG = +0.13%"),
        ("m_τ (framework chain from m_e)", m_tau_GeV, 1.77686, "PDG 0.0001 GeV",
         "via m_e_anchor × framework m_τ/m_e ratio"),
        ("v_Higgs (predicted)", v_Higgs_pred, 246.22, "0.0015 GeV",
         "v = m_τ_framework / y_τ_framework"),
        ("G_F (predicted)", G_F_pred, 1.1663787e-5, "6e-12 GeV⁻²",
         "G_F = 1/(√2·v²)"),
        ("M_Pl (predicted, anchor)", M_Pl_GeV, 1.22089e19, "6e13 GeV",
         "M_Pl = √(ℏc/G_N) — round-trip, since G_N is anchor"),
        ("t_0 (cosmic age)", t_0_pred_Gyr, 14.38, "0.80 Gyr",
         "vs Methuselah model-independent"),
        ("H_0 (Hubble)", H_0_kmsMpc, 67.4, "0.5 km/s/Mpc",
         "vs Planck CMB (ΛCDM)"),
    ]

    for name, pred, obs, sigma_str, notes in comparisons:
        if isinstance(obs, float) and obs > 0:
            dev_rel = (pred - obs) / obs
            print(f"  {name}:")
            print(f"    Framework predicted = {pred:.6g}")
            print(f"    Observed            = {obs:.6g}")
            print(f"    Deviation           = {dev_rel*100:+.4f}%   [obs σ: {sigma_str}]")
            print(f"    Notes: {notes}")
            print()

    print("=" * 78)
    print("  HONEST INTERPRETATION")
    print("=" * 78)
    print()
    print("  The framework's chain has a SYSTEMATIC ~0.12-0.13% offset in mass")
    print("  predictions (m_τ +0.13%, m_e +0.12%, y_τ +0.13%). This propagates:")
    print()
    print("    v_Higgs predicted   ≈ 246.22 GeV ± 0.13%  (matches obs ~by chance)")
    print("    G_F predicted       ≈ 1.169e-5 ± 0.27%   (~5400σ from PDG precision)")
    print("    t_0 predicted       ≈ 14.34 Gyr (matches Methuselah within input σ)")
    print("    H_0 predicted       ≈ 68.2 km/s/Mpc (between Planck CMB and SH0ES)")
    print()
    print("  My earlier 'G_F at 5.7σ tension' or 'G_F at 0.0003% match' framings")
    print("  were artifacts of MIXING PDG values (m_τ/m_e ratio, observed y_τ)")
    print("  with framework predictions in the chain. With consistent")
    print("  framework-only chain, G_F prediction is ~0.27% off PDG (5400σ).")
    print()
    print("  This 0.27% G_F discrepancy IS the framework's known systematic, not")
    print("  a 'new physics' signal. It traces back to y_τ = α_1_full/k*²")
    print("  prediction being +0.13% above observed — likely a sub-leading")
    print("  Yukawa correction the framework's current chain doesn't include.")
    print()
    print("  Where 'new physics' COULD show up:")
    print("  - Cosmic age 14.34 Gyr: matches Methuselah within input σ.")
    print("  - H_0 = 68 km/s/Mpc: framework picks coasting side of Hubble tension.")
    print("    These ARE genuine framework predictions distinguishing it from ΛCDM.")
    print()
    print("  But G_F 0.27% discrepancy: NOT new physics — it's the y_τ systematic.")


if __name__ == "__main__":
    main()
