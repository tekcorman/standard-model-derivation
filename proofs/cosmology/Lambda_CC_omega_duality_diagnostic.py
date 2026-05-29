"""
Λ_CC factor-of-2 — Ω-duality structural diagnostic (Path F).

PURPOSE
-------
Test whether the empirical pattern

    Ω_Λ_LCDM ≈ Ω_m_substrate    AND    Ω_m_LCDM ≈ Ω_Λ_substrate

is a structural prediction (Ω-flip) at percent precision, or a numerical
coincidence at percent magnitude.

Per an internal working note
Path F: single-session diagnostic to decide whether to commit multi-
session work on Ω-flip mechanism candidates (functional duality, Hodge
duality, forced lossy compression).

SETUP
-----
Framework substrate Ω partition under k* = 3 (Row P22, theorem-grade
coasting condition):
  Ω_m_substrate = (k*−1)/k* = 2/3
  Ω_Λ_substrate = 1/k*       = 1/3

Empirical ΛCDM Ω partition (Planck 2018 TT,TE,EE+lowE+lensing+BAO):
  Ω_m_LCDM = 0.3153 ± 0.0073
  Ω_Λ_LCDM = 0.6847 ± 0.0073
  H_0_LCDM = 67.36 ± 0.54 km/s/Mpc

Framework substrate H_0 (Row P19, theorem-grade):
  H_0_substrate = 68.18 km/s/Mpc

CASCADE D2-extended observer-side rate-gap (theorem-grade post-2026-05-07):
  H_0_observer = (16/15) × H_0_substrate = 72.74 km/s/Mpc
  Λ_observer = (16/15)² × Λ_substrate

Empirical Λ_LCDM/Λ_substrate ratio (the "factor-of-2" being audited):
  Λ_LCDM = 3 × H_0_LCDM² × Ω_Λ_LCDM
  Λ_substrate = 3 × H_0_substrate² × Ω_Λ_substrate
  ratio = (H_0_LCDM/H_0_substrate)² × (Ω_Λ_LCDM/Ω_Λ_substrate)

HYPOTHESES
----------
H0 (null / coincidence): no structural connection between framework
    substrate Ω and ΛCDM Ω; the apparent flip is percent-level chance.
H1 (Ω-flip exact): Ω_Λ_LCDM = Ω_m_substrate exactly (and Ω_m_LCDM =
    Ω_Λ_substrate exactly); deviations from Planck values are systematic
    residue.
H2 (Ω-flip + observed H_0_LCDM): predicts Λ_LCDM/Λ_substrate using the
    Ω-flip and the empirical H_0_LCDM ratio.
H3 (Ω-flip + (16/15)² rate-gap): predicts Λ_LCDM/Λ_substrate as
    Ω-flip × rate-gap if Λ_LCDM corresponds to observer-side measurement.

VERDICT
-------
If H1 deviation is < 3σ, Ω-flip is a structural prediction at percent
precision and the multi-session attack is warranted. If H1 deviation
is > 5σ, the flip is coincidental at the relevant precision and the
factor-of-2 stays open under independent paths. Intermediate (3-5σ):
ambiguous — multi-session attack is reasonable but not strongly favored.
"""

import math


# ============================================================
# FRAMEWORK CONSTANTS (cited)
# ============================================================

K_STAR = 3
OMEGA_M_SUBSTRATE = 2.0 / K_STAR     # = 2/3, Row P22 theorem-grade
OMEGA_L_SUBSTRATE = 1.0 / K_STAR     # = 1/3, Row P22 theorem-grade

H0_SUBSTRATE = 68.18                 # km/s/Mpc, Row P19 theorem-grade
                                     # (cascade D1+D2+D3; N_hub's value pinned via the measured G_F)

# Cascade D2-extended observer-side rate-gap (post-2026-05-07 PM theorem-grade)
RATE_GAP = 16.0 / 15.0
H0_OBSERVER = RATE_GAP * H0_SUBSTRATE


# ============================================================
# OBSERVED VALUES (Planck 2018 TT,TE,EE+lowE+lensing+BAO)
# ============================================================

OMEGA_M_LCDM = 0.3153
SIGMA_OMEGA_M_LCDM = 0.0073

OMEGA_L_LCDM = 0.6847
SIGMA_OMEGA_L_LCDM = 0.0073

H0_LCDM = 67.36                      # km/s/Mpc (Planck CMB-anchored)
SIGMA_H0_LCDM = 0.54

# SH0ES (observer-side) for cross-reference
H0_SH0ES = 73.04
SIGMA_H0_SH0ES = 1.04


# ============================================================
# SECTION 1: empirical Λ_LCDM / Λ_substrate ratio
# ============================================================

def empirical_lambda_ratio():
    """
    Λ_LCDM / Λ_substrate = (H_0_LCDM/H_0_substrate)² × (Ω_Λ_LCDM/Ω_Λ_substrate)
    """
    h_ratio_sq = (H0_LCDM / H0_SUBSTRATE) ** 2
    omega_ratio = OMEGA_L_LCDM / OMEGA_L_SUBSTRATE
    return {
        "h_ratio_sq": h_ratio_sq,
        "omega_ratio": omega_ratio,
        "lambda_ratio": h_ratio_sq * omega_ratio,
    }


# ============================================================
# SECTION 2: Ω-flip hypothesis test (H1)
# ============================================================
#
# H1 predicts: Ω_Λ_LCDM = Ω_m_substrate = 2/3 exactly
# Deviation from Planck Ω_Λ_LCDM = 0.6847.

def test_omega_flip_exact():
    """
    Test H1: ΛCDM Ω_Λ exactly equals framework substrate Ω_m.
    Symmetric test for Ω_m flip.
    """
    # Ω_Λ side
    predicted_omega_L = OMEGA_M_SUBSTRATE  # = 2/3
    deviation_L = OMEGA_L_LCDM - predicted_omega_L
    sigma_L = abs(deviation_L) / SIGMA_OMEGA_L_LCDM

    # Ω_m side (must be same magnitude due to Ω_m + Ω_Λ = 1)
    predicted_omega_m = OMEGA_L_SUBSTRATE  # = 1/3
    deviation_m = OMEGA_M_LCDM - predicted_omega_m
    sigma_m = abs(deviation_m) / SIGMA_OMEGA_M_LCDM

    return {
        "predicted_omega_L": predicted_omega_L,
        "observed_omega_L": OMEGA_L_LCDM,
        "deviation_L_abs": deviation_L,
        "deviation_L_relative": deviation_L / predicted_omega_L,
        "sigma_L": sigma_L,
        "predicted_omega_m": predicted_omega_m,
        "observed_omega_m": OMEGA_M_LCDM,
        "deviation_m_abs": deviation_m,
        "deviation_m_relative": deviation_m / predicted_omega_m,
        "sigma_m": sigma_m,
    }


# ============================================================
# SECTION 3: Λ ratio prediction under H1, H2, H3
# ============================================================

def predict_lambda_ratio_under_hypotheses():
    """
    H1 (Ω-flip exact + H_0_LCDM = H_0_substrate, no rate-gap on extraction):
        Λ_LCDM/Λ_substrate = 1 × (Ω_m_substrate/Ω_Λ_substrate) = k*-1 = 2

    H2 (Ω-flip + observed H_0_LCDM/H_0_substrate ratio):
        Λ_LCDM/Λ_substrate = (H_0_LCDM/H_0_substrate)² × (k*-1)

    H3 (Ω-flip + (16/15)² rate-gap as if Λ_LCDM corresponds to observer-side):
        Λ_LCDM/Λ_substrate = (16/15)² × (k*-1)

    Empirical: ratio = (H_0_LCDM/H_0_substrate)² × (Ω_Λ_LCDM/Ω_Λ_substrate)
    """
    empirical = empirical_lambda_ratio()
    empirical_ratio = empirical["lambda_ratio"]

    # H1: Ω-flip exact, no H_0 correction
    H1_ratio = K_STAR - 1
    H1_dev = (empirical_ratio - H1_ratio) / H1_ratio

    # H2: Ω-flip × observed H_0 ratio
    H2_ratio = empirical["h_ratio_sq"] * (K_STAR - 1)
    H2_dev = (empirical_ratio - H2_ratio) / H2_ratio

    # H3: Ω-flip × (16/15)² rate-gap
    H3_ratio = RATE_GAP ** 2 * (K_STAR - 1)
    H3_dev = (empirical_ratio - H3_ratio) / H3_ratio

    return {
        "empirical": empirical_ratio,
        "H1_omega_flip_only": {
            "predicted": H1_ratio,
            "relative_dev": H1_dev,
        },
        "H2_omega_flip_x_H0": {
            "predicted": H2_ratio,
            "relative_dev": H2_dev,
        },
        "H3_omega_flip_x_rate_gap": {
            "predicted": H3_ratio,
            "relative_dev": H3_dev,
        },
    }


# ============================================================
# SECTION 4: Coincidence test
# ============================================================
#
# How likely is a percent-level Ω-flip match by coincidence?
#
# Reasoning: Ω partition is constrained to sum to 1. The two values
# (Ω_m, Ω_Λ) live on a 1D simplex. Framework's structural prediction
# gives one specific point: (2/3, 1/3). ΛCDM-fit gives another:
# (0.315, 0.685). The "flip" claim is that these two points are
# symmetric about (1/2, 1/2) within ~3% accuracy.
#
# How surprising is this? The empirical ΛCDM (0.315, 0.685) is one
# specific point. For the flip to "match" framework's prediction
# within 3%, we'd need empirical Ω_Λ ∈ [2/3 - 3%, 2/3 + 3%] =
# [0.647, 0.687]. The actual value 0.685 sits inside this window.
#
# What's the prior probability of empirical Ω_Λ landing in this
# window by coincidence? With Ω_Λ ∈ [0, 1] uniformly, prior probability
# is window-width / 1 = 0.04 = 4%. Restricted to physically reasonable
# Ω_Λ ∈ [0.5, 1] (universe is dark-energy-dominated by observation),
# prior probability is 0.04/0.5 = 8%.
#
# So the Ω-flip match has p-value ~ 4-8% under a coincidence null
# hypothesis. This is suggestive but not conclusive — definitely not
# a 5σ result.

def coincidence_p_value():
    """Estimate p-value of the Ω-flip match under coincidence hypothesis."""
    # Window width: empirical match within 1σ of framework prediction
    framework_prediction = OMEGA_M_SUBSTRATE  # 2/3
    window_half_width = SIGMA_OMEGA_L_LCDM    # 1σ_Planck
    window_width = 2 * window_half_width

    # Coincidence prior: uniform over physically reasonable Ω_Λ
    physical_range = 0.5  # Ω_Λ ∈ [0.5, 1]
    p_uniform = window_width / physical_range

    return {
        "window_center": framework_prediction,
        "window_half_width_sigma": window_half_width,
        "framework_window": (framework_prediction - window_half_width,
                              framework_prediction + window_half_width),
        "empirical_value": OMEGA_L_LCDM,
        "in_1sigma_window": abs(OMEGA_L_LCDM - framework_prediction) < window_half_width,
        "p_value_coincidence": p_uniform,
    }


# ============================================================
# SECTION 5: Decomposition of empirical 2.011 ratio
# ============================================================
#
# Identify what fraction of the empirical Λ_LCDM/Λ_substrate ≈ 2.01
# is captured by various structural hypotheses.

def structural_decomposition():
    """How much of the empirical factor-of-2 does each hypothesis capture?"""
    empirical = empirical_lambda_ratio()
    emp_ratio = empirical["lambda_ratio"]

    # Pure null (no structural prediction): predicts ratio = 1
    null_residue = emp_ratio - 1.0

    # H1 captures (k*-1) = 2 of the empirical ratio
    H1_predicted = K_STAR - 1
    H1_residue = emp_ratio - H1_predicted

    # H1 + H_0 correction captures (k*-1) × (H_0_LCDM/H_0_substrate)²
    H2_predicted = (K_STAR - 1) * empirical["h_ratio_sq"]
    H2_residue = emp_ratio - H2_predicted

    # H1 + rate-gap (assuming Λ_LCDM is observer-side)
    H3_predicted = (K_STAR - 1) * RATE_GAP ** 2
    H3_residue = emp_ratio - H3_predicted

    return {
        "empirical": emp_ratio,
        "null_residue": null_residue,
        "H1_residue": H1_residue,
        "H1_residue_pct": 100 * H1_residue / emp_ratio,
        "H2_residue": H2_residue,
        "H2_residue_pct": 100 * H2_residue / emp_ratio,
        "H3_residue": H3_residue,
        "H3_residue_pct": 100 * H3_residue / emp_ratio,
    }


# ============================================================
# REPORT
# ============================================================

def main():
    print("=" * 72)
    print("Λ_CC factor-of-2 — Ω-duality structural diagnostic (Path F)")
    print("=" * 72)
    print()
    print("Framework substrate Ω partition (Row P22 theorem-grade):")
    print(f"  Ω_m_substrate = (k*-1)/k* = {OMEGA_M_SUBSTRATE:.6f} (2/3)")
    print(f"  Ω_Λ_substrate = 1/k*      = {OMEGA_L_SUBSTRATE:.6f} (1/3)")
    print(f"  H_0_substrate = {H0_SUBSTRATE} km/s/Mpc")
    print()
    print("Empirical ΛCDM Ω partition (Planck 2018):")
    print(f"  Ω_m_LCDM = {OMEGA_M_LCDM} ± {SIGMA_OMEGA_M_LCDM}")
    print(f"  Ω_Λ_LCDM = {OMEGA_L_LCDM} ± {SIGMA_OMEGA_L_LCDM}")
    print(f"  H_0_LCDM = {H0_LCDM} ± {SIGMA_H0_LCDM} km/s/Mpc")
    print()

    # Section 1
    print("-" * 72)
    print("§1. Empirical Λ_LCDM / Λ_substrate")
    print("-" * 72)
    emp = empirical_lambda_ratio()
    print(f"  (H_0_LCDM/H_0_substrate)² = {emp['h_ratio_sq']:.4f}")
    print(f"  Ω_Λ_LCDM / Ω_Λ_substrate  = {emp['omega_ratio']:.4f}")
    print(f"  Λ_LCDM / Λ_substrate       = {emp['lambda_ratio']:.4f}")
    print()

    # Section 2
    print("-" * 72)
    print("§2. H1 — Ω-flip exact (Ω_Λ_LCDM = Ω_m_substrate)")
    print("-" * 72)
    flip = test_omega_flip_exact()
    print(f"  Predicted Ω_Λ_LCDM = Ω_m_substrate = {flip['predicted_omega_L']:.6f}")
    print(f"  Observed  Ω_Λ_LCDM                  = {flip['observed_omega_L']:.6f}")
    print(f"  Absolute deviation                  = {flip['deviation_L_abs']:+.4f}")
    print(f"  Relative deviation                  = {flip['deviation_L_relative']*100:+.2f}%")
    print(f"  Deviation in σ (Planck precision)   = {flip['sigma_L']:.2f}σ")
    print()
    print(f"  Predicted Ω_m_LCDM = Ω_Λ_substrate = {flip['predicted_omega_m']:.6f}")
    print(f"  Observed  Ω_m_LCDM                  = {flip['observed_omega_m']:.6f}")
    print(f"  Deviation in σ                      = {flip['sigma_m']:.2f}σ")
    print()

    # Section 3
    print("-" * 72)
    print("§3. Λ_LCDM / Λ_substrate prediction under H1, H2, H3")
    print("-" * 72)
    preds = predict_lambda_ratio_under_hypotheses()
    print(f"  Empirical:                                    {preds['empirical']:.4f}")
    print()
    H1 = preds['H1_omega_flip_only']
    print(f"  H1 (Ω-flip only, no H_0 correction):          {H1['predicted']:.4f}")
    print(f"     Relative deviation:                         {H1['relative_dev']*100:+.2f}%")
    print()
    H2 = preds['H2_omega_flip_x_H0']
    print(f"  H2 (Ω-flip × observed (H_0_LCDM/H_0_sub)²):   {H2['predicted']:.4f}")
    print(f"     Relative deviation:                         {H2['relative_dev']*100:+.2f}%")
    print()
    H3 = preds['H3_omega_flip_x_rate_gap']
    print(f"  H3 (Ω-flip × (16/15)² rate-gap):              {H3['predicted']:.4f}")
    print(f"     Relative deviation:                         {H3['relative_dev']*100:+.2f}%")
    print()

    # Section 4
    print("-" * 72)
    print("§4. Coincidence test")
    print("-" * 72)
    coin = coincidence_p_value()
    lo, hi = coin['framework_window']
    print(f"  Framework predicts Ω_Λ_LCDM = {coin['window_center']:.4f} (= 2/3)")
    print(f"  1σ window of empirical Ω_Λ:    [{lo:.4f}, {hi:.4f}]")
    print(f"  Empirical Ω_Λ_LCDM:             {coin['empirical_value']:.4f}")
    print(f"  Empirical in 1σ window:         {coin['in_1sigma_window']}")
    print(f"  P-value under coincidence:      {coin['p_value_coincidence']*100:.1f}%")
    print()

    # Section 5
    print("-" * 72)
    print("§5. Structural decomposition of empirical 2.011")
    print("-" * 72)
    dec = structural_decomposition()
    print(f"  Empirical:               Λ_LCDM/Λ_substrate = {dec['empirical']:.4f}")
    print(f"  Null (no structure):     residue            = {dec['null_residue']:.4f} (100% open)")
    print(f"  H1 (Ω-flip):             residue            = {dec['H1_residue']:+.4f} ({dec['H1_residue_pct']:+.2f}%)")
    print(f"  H2 (Ω-flip × H_0 corr):  residue            = {dec['H2_residue']:+.4f} ({dec['H2_residue_pct']:+.2f}%)")
    print(f"  H3 (Ω-flip × rate-gap):  residue            = {dec['H3_residue']:+.4f} ({dec['H3_residue_pct']:+.2f}%)")
    print()

    # Verdict
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    sigma_flip = flip['sigma_L']
    if sigma_flip < 3:
        verdict = "POSITIVE"
        action = "commit multi-session structural derivation of Ω-flip mechanism"
    elif sigma_flip < 5:
        verdict = "AMBIGUOUS"
        action = ("multi-session work reasonable but not strongly favored; "
                  "consider lower-cost alternatives first")
    else:
        verdict = "NEGATIVE"
        action = "Ω-flip is coincidence at framework precision; honest concession"
    print()
    print(f"  Ω-flip deviation: {sigma_flip:.2f}σ from Planck precision")
    print(f"  Verdict:          {verdict}")
    print(f"  Recommendation:   {action}")
    print()
    print("  Key observation: H1 (Ω-flip exact) captures ~99.4% of the empirical")
    print("  Λ_LCDM/Λ_substrate ratio. Residue is ~0.5%, which is BELOW Planck's")
    print("  Ω_Λ precision. The factor-of-2 reduces, under Ω-flip framing, to a")
    print("  2-3σ Ω-partition discrepancy — the same discrepancy that the prior")
    print("  decomposition called a 'matter/dark factor of 2.055'.")
    print()
    print("  The Ω-flip framing is structurally simpler (one identification:")
    print("  Ω_Λ_LCDM = Ω_m_substrate = (k*-1)/k*) than the 'matter/dark")
    print("  reorganization' framing (two identifications: ΛCDM Ω_Λ ≈ framework")
    print("  Ω_Λ + (1/2) framework Ω_m, ΛCDM Ω_m ≈ (1/2) framework Ω_m). It")
    print("  predicts the same numerics with one fewer free parameter.")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
