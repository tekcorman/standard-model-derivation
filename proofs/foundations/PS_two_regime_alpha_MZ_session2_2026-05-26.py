#!/usr/bin/env python3
"""
PS → SM two-regime α_i(M_Z) match — Session 2 probe (P2).

Scoping: an internal working note (P2)
Session 1: proofs/foundations/PS_beta_coefficients_substrate_2026-05-26.py

GOAL: under one-loop RG running, with PS β coefficients above M_R and SM β
coefficients below, does α_i(M_Z) match PDG at or near the precision currently
achieved with single-regime MSSM running?

Method:
  1. Start at M_unif with α_4 = α_2L = α_2R = α_GUT (framework theorem-grade).
  2. Run PS β coefficients (matter-only or with-Higgs) from M_unif down to M_R.
  3. Match at M_R: α_3 = α_4, α_2 = α_2L, 1/α_1_GUT = (9/25)/α_2R + (6/25)/α_4.
  4. Run SM β coefficients (NOT MSSM) from M_R down to M_Z.
  5. Compare 1/α_i(M_Z) to PDG.

Pre-declared aborts from scoping (§6):
  AB1 (PS β not derivable): not triggered — Session 1 derived matter-only.
  AB2 (α_i(M_Z) > 1σ worse than MSSM): primary test of this session.
  AB3 (c asymmetry must be put in by hand): emergent question of Session 3.
  AB4 (two M_R routes disagree): not tested here.
"""

import math

# ============================================================
# FRAMEWORK INPUTS (theorem-grade or theorem-grade-conditional)
# ============================================================
ALPHA_GUT = 0.0411027403      # framework-predicted, predictions/alpha_GUT.py
M_UNIF = 1.9849e16            # GeV, predictions/M_unif.py
M_R = 1.2406e15               # GeV, predictions/m_nu3.py (substrate spectral gap)
M_Z = 91.1876                 # GeV, PDG observed

# ============================================================
# β COEFFICIENTS
# ============================================================
# PS Session 1 (substrate-derived matter-only)
B4_MATTER, B2L_MATTER, B2R_MATTER = -32.0/3, -10.0/3, -10.0/3

# PS Session 1 (matter + minimal Higgs (1,2,2) + (4̄,1,2); literature)
B4_HIGGS, B2L_HIGGS, B2R_HIGGS = -31.0/3, -3.0, -7.0/3

# SUSY PS (matter + minimal Higgs as chiral supermultiplets)
B4_SUSY, B2L_SUSY, B2R_SUSY = -5.0, 1.0, 3.0

# SM β coefficients (non-SUSY, GUT-normalized for U(1))
B1_SM, B2_SM, B3_SM = 41.0/10, -19.0/6, -7.0

# MSSM β coefficients (current framework baseline)
B1_MSSM, B2_MSSM, B3_MSSM = 33.0/5, 1.0, -3.0


# ============================================================
# PDG TARGETS (GUT-normalized α_1)
# ============================================================
ALPHA_S_MZ = 0.1179            # PDG α_s(M_Z) = α_3(M_Z)
ALPHA_EM_INV_MZ = 127.94       # PDG α_EM^{-1}(M_Z)
SIN2_THETA_W = 0.23122         # PDG sin²θ_W(M_Z)

ALPHA_3_PDG = ALPHA_S_MZ
ALPHA_2_PDG = (1.0/ALPHA_EM_INV_MZ) / SIN2_THETA_W
ALPHA_Y_PDG = (1.0/ALPHA_EM_INV_MZ) / (1.0 - SIN2_THETA_W)
ALPHA_1_PDG_GUT = (5.0/3.0) * ALPHA_Y_PDG

INV_ALPHA_3_PDG = 1.0/ALPHA_3_PDG
INV_ALPHA_2_PDG = 1.0/ALPHA_2_PDG
INV_ALPHA_1_PDG = 1.0/ALPHA_1_PDG_GUT


# ============================================================
# RUNNING + MATCHING
# ============================================================
def run_inv_alpha(inv_alpha_init, mu_init, mu_final, b):
    """One-loop RG: 1/α(μ) = 1/α(μ₀) − (b/(2π))·ln(μ/μ₀)."""
    return inv_alpha_init - (b/(2.0*math.pi)) * math.log(mu_final/mu_init)


def match_PS_to_SM_at_MR(inv_a4, inv_a2L, inv_a2R):
    """
    At M_R: SU(4)×SU(2)_L×SU(2)_R → SU(3)×SU(2)_L×U(1)_Y.
      1/α_3 = 1/α_4
      1/α_2 = 1/α_2L
      1/α_Y_SM = (3/5)/α_2R + (2/5)/α_4   [from Y = T_3R + (B-L)/2]
      1/α_1_GUT = (3/5)/α_Y_SM = (9/25)/α_2R + (6/25)/α_4
    """
    inv_a3 = inv_a4
    inv_a2 = inv_a2L
    inv_a1_GUT = (9.0/25.0)*inv_a2R + (6.0/25.0)*inv_a4
    return inv_a3, inv_a2, inv_a1_GUT


def two_regime(b_PS, b_SM_below, label):
    """Run PS from M_unif to M_R, match, run SM below to M_Z."""
    b4, b2L, b2R = b_PS
    b1s, b2s, b3s = b_SM_below
    inv_a_GUT = 1.0/ALPHA_GUT

    inv_a4_MR = run_inv_alpha(inv_a_GUT, M_UNIF, M_R, b4)
    inv_a2L_MR = run_inv_alpha(inv_a_GUT, M_UNIF, M_R, b2L)
    inv_a2R_MR = run_inv_alpha(inv_a_GUT, M_UNIF, M_R, b2R)

    inv_a3_MR, inv_a2_MR, inv_a1_MR = match_PS_to_SM_at_MR(inv_a4_MR, inv_a2L_MR, inv_a2R_MR)

    inv_a1_MZ = run_inv_alpha(inv_a1_MR, M_R, M_Z, b1s)
    inv_a2_MZ = run_inv_alpha(inv_a2_MR, M_R, M_Z, b2s)
    inv_a3_MZ = run_inv_alpha(inv_a3_MR, M_R, M_Z, b3s)

    return label, (inv_a1_MZ, inv_a2_MZ, inv_a3_MZ)


def single_regime(b, label):
    """Single-regime running M_unif → M_Z with one set of β coefficients."""
    b1, b2, b3 = b
    inv_a_GUT = 1.0/ALPHA_GUT
    inv_a1_MZ = run_inv_alpha(inv_a_GUT, M_UNIF, M_Z, b1)
    inv_a2_MZ = run_inv_alpha(inv_a_GUT, M_UNIF, M_Z, b2)
    inv_a3_MZ = run_inv_alpha(inv_a_GUT, M_UNIF, M_Z, b3)
    return label, (inv_a1_MZ, inv_a2_MZ, inv_a3_MZ)


# ============================================================
# REPORT
# ============================================================
def report():
    print("=" * 78)
    print("  PS → SM two-regime α_i(M_Z) — Session 2 probe")
    print("=" * 78)
    print(f"\n  Framework inputs:")
    print(f"    α_GUT      = {ALPHA_GUT:.6f}  (1/α = {1/ALPHA_GUT:.4f})")
    print(f"    M_unif     = {M_UNIF:.3e} GeV")
    print(f"    M_R        = {M_R:.3e} GeV  (= M_unif/16)")
    print(f"    M_Z        = {M_Z:.4f} GeV")
    print(f"    PS regime  = {math.log10(M_UNIF/M_R):.2f} decades")
    print(f"    SM regime  = {math.log10(M_R/M_Z):.2f} decades")

    print(f"\n  PDG targets (GUT-normalized):")
    print(f"    1/α_1(M_Z) = {INV_ALPHA_1_PDG:.3f}")
    print(f"    1/α_2(M_Z) = {INV_ALPHA_2_PDG:.3f}")
    print(f"    1/α_3(M_Z) = {INV_ALPHA_3_PDG:.3f}  (α_s = {ALPHA_S_MZ})")

    results = []
    results.append(single_regime((B1_MSSM, B2_MSSM, B3_MSSM), "MSSM single-regime (current baseline)"))
    results.append(single_regime((B1_SM, B2_SM, B3_SM), "SM single-regime (no SUSY, no PS — for comparison)"))
    results.append(two_regime((B4_MATTER, B2L_MATTER, B2R_MATTER),
                              (B1_SM, B2_SM, B3_SM),
                              "PS(matter-only) → SM"))
    results.append(two_regime((B4_HIGGS, B2L_HIGGS, B2R_HIGGS),
                              (B1_SM, B2_SM, B3_SM),
                              "PS(with Higgs) → SM"))
    results.append(two_regime((B4_MATTER, B2L_MATTER, B2R_MATTER),
                              (B1_MSSM, B2_MSSM, B3_MSSM),
                              "PS(matter-only) → MSSM"))
    results.append(two_regime((B4_HIGGS, B2L_HIGGS, B2R_HIGGS),
                              (B1_MSSM, B2_MSSM, B3_MSSM),
                              "PS(with Higgs) → MSSM"))
    results.append(two_regime((B4_SUSY, B2L_SUSY, B2R_SUSY),
                              (B1_MSSM, B2_MSSM, B3_MSSM),
                              "SUSY-PS → MSSM"))

    print(f"\n  {'Scenario':<46} {'1/α_1':>9} {'1/α_2':>9} {'1/α_3':>9}")
    print(f"  {'':46} {'(PDG ' + f'{INV_ALPHA_1_PDG:.2f}'+')':>9} "
          f"{'(PDG ' + f'{INV_ALPHA_2_PDG:.2f}'+')':>9} "
          f"{'(PDG ' + f'{INV_ALPHA_3_PDG:.2f}'+')':>9}")
    print(f"  {'-'*46} {'-'*9} {'-'*9} {'-'*9}")
    for label, (i1, i2, i3) in results:
        print(f"  {label:<46} {i1:>9.3f} {i2:>9.3f} {i3:>9.3f}")

    print(f"\n  Deviations (Δ(1/α_i) from PDG):")
    print(f"  {'Scenario':<46} {'Δ1':>9} {'Δ2':>9} {'Δ3':>9}")
    print(f"  {'-'*46} {'-'*9} {'-'*9} {'-'*9}")
    for label, (i1, i2, i3) in results:
        d1 = i1 - INV_ALPHA_1_PDG
        d2 = i2 - INV_ALPHA_2_PDG
        d3 = i3 - INV_ALPHA_3_PDG
        print(f"  {label:<46} {d1:>+9.3f} {d2:>+9.3f} {d3:>+9.3f}")

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)

    mssm_devs = abs(results[0][1][0] - INV_ALPHA_1_PDG) + \
                abs(results[0][1][1] - INV_ALPHA_2_PDG) + \
                abs(results[0][1][2] - INV_ALPHA_3_PDG)

    # Find the two-regime variant with smallest total deviation
    best_label, best_dev = None, float('inf')
    for label, (i1, i2, i3) in results[2:]:
        dev = abs(i1 - INV_ALPHA_1_PDG) + abs(i2 - INV_ALPHA_2_PDG) + abs(i3 - INV_ALPHA_3_PDG)
        if dev < best_dev:
            best_dev = dev
            best_label = label

    print(f"\n  MSSM single-regime total |Δ| = {mssm_devs:.3f}")
    print(f"  Best two-regime variant: {best_label}")
    print(f"  Best two-regime total |Δ| = {best_dev:.3f}")
    print(f"  Improvement over MSSM = {mssm_devs - best_dev:+.3f}")

    print()
    if best_dev < mssm_devs - 0.5:
        print("  >>> Some two-regime variant CLEARLY BEATS MSSM single-regime.")
        print("  >>> AB2 NOT triggered. PS → SM scoping advances toward Outcome A.")
    elif best_dev < mssm_devs + 0.5:
        print("  >>> Two-regime variants are COMPARABLE to MSSM single-regime.")
        print("  >>> AB2 borderline. PS → SM scoping lands at Outcome B.")
    else:
        print("  >>> NO two-regime variant beats MSSM single-regime.")
        print("  >>> AB2 TRIGGERED. PS → SM mechanism in current form does NOT")
        print("      replace MSSM β-coefficient dependency.")

    print("\n  Honest structural reading:")
    print("    - MSSM β coefficients arise from MSSM matter content (SM + SUSY")
    print("      partners). The framework's substrate-derived PS multiplets")
    print("      contain only SM matter content, not SUSY partners.")
    print("    - Single-regime SM running gives 1/α_3(M_Z) ≈ -14 (catastrophic),")
    print("      which is why MSSM was historically invoked.")
    print("    - PS → SM Session 1 derived matter-only β coefficients but did")
    print("      not address whether the framework derives SUSY-partner-like")
    print("      matter content from substrate. If not, MSSM β values cannot be")
    print("      replaced by purely substrate-derived non-SUSY values.")
    print("    - The PS → SM scoping's proposal that PS thresholds replace SUSY")
    print("      scaffolding requires deriving the additional matter content")
    print("      (or a structurally equivalent mechanism) — Session 1 did not")
    print("      do this and Session 2 demonstrates the consequence.")
    print()
    print("  Open question: does Cl(6,0) Fock + Cl(0,2) decomposition contain")
    print("  additional 'SUSY-partner-like' multiplets that would soften the")
    print("  effective β coefficients toward MSSM-like values? If yes, the")
    print("  PS → SM scoping advances. If no, the MSSM matter content assertion")
    print("  is load-bearing in framework predictions and the suspect catalog's")
    print("  §3.1 graduation requires a separate substrate-side investigation.")
    print("=" * 78)

    return results


if __name__ == "__main__":
    report()
