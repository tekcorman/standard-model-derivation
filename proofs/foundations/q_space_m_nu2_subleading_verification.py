#!/usr/bin/env python3
"""
Investigation #2-followup verification — does M_2 subleading prediction
improve or degrade the m_ν2 PDG match?

CONTEXT: theorem_analytical_feshbach_ramanujan_boundary.md derived the
closed-form Σ(h) = (α_1/h)·[M_0 + Σ M_m e^{-imα}] at the Ramanujan-circle
saddle. Leading (M_0 = 1) reproduces the framework's m_ν dark coefficient
√5/4 EXACTLY. Investigation #3 measured empirical M_2 ≈ -0.27 across the
standard substrate family.

The M_2 subleading correction predicts:
  -Im(Σ)/α_1: 0.5590 (leading) → 0.4836 (with M_2 = -0.27)  i.e., -13.5% shift

If this M_2 correction reflects PHYSICAL Q-space density (not substrate
artifact), then including it should IMPROVE the m_ν2 prediction's PDG
match. If it makes the match WORSE, a separate private derivation by the author (uniform
density at MDL optimum, M_n = 0 for n≥1) IS structurally correct — and
the empirical M_n we measured is substrate-level noise that doesn't
propagate to SM observables.

This probe runs both predictions and reports σ deviations from PDG.
"""

import math, sys, os
from fractions import Fraction
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))

# Framework constants (from predictions/m_nu2.py)
ALPHA_1_BARE = (2/3)**8
M_NU3_BARE_EV = 0.048277       # ADOPTED-PS Pati-Salam seesaw scale
R_SPLITTING = Fraction(228, 7) # 228/7 splitting ratio (theorem-grade)
DARK_COEFF_LEADING = math.sqrt(5)/4   # = √5/4 ≈ 0.5590 (a separate private derivation by the author analytical leading)

# PDG / NuFIT 6.0 (Sept 2024), normal ordering
DM2_21_OBS = 7.49e-5
DM2_21_SIGMA = 0.19e-5
M_NU2_OBS = math.sqrt(DM2_21_OBS)              # ≈ 8.654 meV
M_NU2_SIGMA = 0.5 * DM2_21_SIGMA / M_NU2_OBS    # ≈ 0.110 meV

# Saddle h = (√3+i√5)/2; arg h
H_SADDLE = complex(math.sqrt(3)/2, math.sqrt(5)/2)
ARG_H = math.atan2(H_SADDLE.imag, H_SADDLE.real)


def predict_m_nu2(dark_coeff):
    """Compute m_nu2 with given dark coefficient on Im(Σ)/α_1 form."""
    correction = 1 + ALPHA_1_BARE * dark_coeff
    m_nu2_bare = M_NU3_BARE_EV / math.sqrt(float(R_SPLITTING))
    return m_nu2_bare * correction


def deviation_sigma(m_pred):
    return (m_pred - M_NU2_OBS) / M_NU2_SIGMA


def m_2_modulated_dark_coeff(M_2_val):
    """
    Apply M_2 modulation per theorem_analytical_feshbach_ramanujan_boundary.md.

    Σ_M2(h) = (α_1/h) · [1 + M_2 e^{-2iα}]
    -Im(Σ_M2)/α_1 = -Im[(1/h)·(1 + M_2 e^{-2iα})]
    With h = √2 e^{iα}, 1/h = e^{-iα}/√2:
    -Im(Σ_M2)/α_1 = -Im[(e^{-iα}/√2)·(1 + M_2 e^{-2iα})]
                  = (1/√2)·Im[(1 + M_2 e^{-2iα})·e^{iα}]   [conjugating 1/√2]
    Just compute numerically.
    """
    bracket = 1 + M_2_val * complex(math.cos(2*ARG_H), -math.sin(2*ARG_H))
    sigma_unit = bracket / H_SADDLE  # (1/h)·[1 + M_2·e^{-2iα}]
    return -sigma_unit.imag


def main():
    print("=" * 96)
    print("M_2 SUBLEADING VERIFICATION ON m_ν2 PREDICTION")
    print("=" * 96)
    print(f"\n  PDG / NuFIT 6.0 (normal ordering):")
    print(f"    Δm²_21 = {DM2_21_OBS:.2e} ± {DM2_21_SIGMA:.2e} eV²")
    print(f"    m_ν2_obs = {M_NU2_OBS*1e3:.4f} ± {M_NU2_SIGMA*1e3:.4f} meV")
    print(f"\n  Framework constants:")
    print(f"    α_1 = (2/3)^8 = {ALPHA_1_BARE:.10f}")
    print(f"    Bare scale (ADOPTED-PS): m_ν3_bare = {M_NU3_BARE_EV} eV")
    print(f"    Splitting (theorem): R = 228/7 = {float(R_SPLITTING):.6f}")
    print(f"    Saddle: arg h = {math.degrees(ARG_H):.4f}° = {ARG_H:.6f} rad")

    # ---- Leading-only (current framework) ----
    print("\n" + "-" * 96)
    print("Prediction 1: LEADING ONLY  (current framework, a separate private derivation by the author uniform density)")
    print("-" * 96)
    coeff_leading = DARK_COEFF_LEADING
    m_pred_leading = predict_m_nu2(coeff_leading)
    dev_leading = deviation_sigma(m_pred_leading)
    print(f"\n  Dark coefficient: -Im(Σ)/α_1 = √5/4 = {coeff_leading:.6f}")
    print(f"  Correction:       1 + α_1·√5/4 = {1 + ALPHA_1_BARE*coeff_leading:.10f}")
    print(f"  m_ν2 predicted:   {m_pred_leading*1e3:.4f} meV")
    print(f"  Deviation:        {(m_pred_leading-M_NU2_OBS)*1e3:+.4f} meV ({dev_leading:+.2f}σ)")

    # ---- M_2 modulated (theorem prediction) ----
    print("\n" + "-" * 96)
    print("Prediction 2: M_2 SUBLEADING  (theorem_analytical_feshbach with M_2 = -0.27)")
    print("-" * 96)
    M_2 = -0.27
    coeff_M2 = m_2_modulated_dark_coeff(M_2)
    m_pred_M2 = predict_m_nu2(coeff_M2)
    dev_M2 = deviation_sigma(m_pred_M2)
    print(f"\n  Empirical M_2 (Inv #3 standard family): {M_2}")
    print(f"  Dark coefficient: -Im(Σ_M2)/α_1 = {coeff_M2:.6f}")
    print(f"  Shift from leading: {(coeff_M2-coeff_leading):+.4f}  "
          f"({(coeff_M2/coeff_leading-1)*100:+.2f}%)")
    print(f"  Correction:       1 + α_1·{coeff_M2:.4f} = {1 + ALPHA_1_BARE*coeff_M2:.10f}")
    print(f"  m_ν2 predicted:   {m_pred_M2*1e3:.4f} meV")
    print(f"  Deviation:        {(m_pred_M2-M_NU2_OBS)*1e3:+.4f} meV ({dev_M2:+.2f}σ)")

    # ---- Sweep over plausible M_2 values ----
    print("\n" + "-" * 96)
    print("Sensitivity sweep: m_ν2 prediction vs M_2 input value")
    print("-" * 96)
    print(f"\n  {'M_2':>8s} {'-Im/α_1':>10s} {'m_ν2 (meV)':>12s} {'σ from PDG':>12s} {'verdict':<20s}")
    for M_2_test in [-0.40, -0.30, -0.27, -0.20, -0.10, 0.0, +0.10, +0.20]:
        c = m_2_modulated_dark_coeff(M_2_test)
        m_p = predict_m_nu2(c)
        dev = deviation_sigma(m_p)
        verdict = ("PDG-consistent" if abs(dev) < 1
                   else "1-2σ tension" if abs(dev) < 2
                   else "≥2σ tension")
        if M_2_test == 0.0: verdict += " ← a separate private derivation by the author leading"
        if M_2_test == -0.27: verdict += " ← Inv #3 empirical"
        print(f"  {M_2_test:>+8.2f} {c:>10.4f} {m_p*1e3:>12.4f} {dev:>+12.2f} {verdict:<20s}")

    # ---- Verdict ----
    print("\n" + "=" * 96)
    print("VERDICT — does M_2 subleading make the PDG match better or worse?")
    print("=" * 96)
    delta_sigma = dev_M2 - dev_leading
    print(f"\n  Leading-only deviation:  {dev_leading:+.2f}σ from PDG")
    print(f"  M_2-modulated deviation: {dev_M2:+.2f}σ from PDG")
    print(f"  Δσ from M_2 inclusion:   {delta_sigma:+.2f}σ")
    print()
    if abs(dev_M2) > abs(dev_leading) + 0.1:
        print("  ✗ M_2 SUBLEADING DEGRADES the PDG match.")
        print("    Suggests a separate private derivation by the author (uniform density at MDL optimum)")
        print("    IS structurally correct: empirical M_n we measured is substrate-level")
        print("    noise/structural fluctuation that does NOT propagate to SM observables.")
        print("    The framework's leading-order dark coefficients (√5/4, 5/12, ...) are")
        print("    the canonical physical predictions.")
    elif abs(dev_M2) < abs(dev_leading) - 0.1:
        print("  ✓ M_2 SUBLEADING IMPROVES the PDG match.")
        print("    Empirical M_2 modulation IS physical — should be incorporated into")
        print("    framework dark-correction predictions.")
    else:
        print("  ◐ NEITHER decisively better; both within precision.")
        print("    Difference {:.2f}σ is below decisive threshold; sub-leading status".format(delta_sigma))
        print("    indeterminate at current PDG precision (σ_obs = ±{:.4f} meV).".format(M_NU2_SIGMA*1e3))


if __name__ == '__main__':
    main()
