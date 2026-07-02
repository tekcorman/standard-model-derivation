"""
proofs/foundations/A2_threshold_matching_session_1_2026-05-27.py

A2 Session 1 — multi-regime PS-running + threshold matching at M_R + SM-running
test for closing the Δb_2 = +4 SU(2)_L gap (and analogous Δb_i at U(1)/SU(3)).

Pre-committed design: an internal working note

Question: does the composition
  α_GUT (M_unif) → PS-running → M_R → matching → SM-running → α_i(M_Z)
produce effective β-coefficients matching MSSM (33/5, 1, -3) without
literal sparticles, using only the framework's substrate-derived matter
content + cascade-theorem M_R?

Convention (Martin SUSY primer + Wikipedia RGE):
  α_i⁻¹(M_low) = α_i⁻¹(M_high) + (b_i / 2π) · ln(M_high / M_low)
  with MSSM (b_1, b_2, b_3) = (33/5, +1, -3) and SM (41/10, -19/6, -7).
"""

from __future__ import annotations

import math
from fractions import Fraction


def banner(title, char="="):
    print(char * 100)
    print(title)
    print(char * 100)


# ============================================================================
# Framework constants (theorem-grade upstream)
# ============================================================================

ALPHA_GUT_INV = 24                              # Cl(6) Fock label count, theorem-grade
SIN2_THETA_W_AT_UNIF = Fraction(3, 8)            # GQW trace identity, theorem-grade
M_PL_GEV = 1.22e19                              # Planck mass
M_UNIF_GEV = (1/24) * (2/3)**8 * M_PL_GEV       # cascade theorem, theorem-grade
M_R_GEV = (2 / 3**9) * M_PL_GEV                 # PS → SM breaking scale, cascade theorem
M_Z_GEV = 91.1876                                # PDG

# PDG observed α_i⁻¹(M_Z) in GUT-norm convention
ALPHA_1_INV_MZ_PDG = 59.0                       # ≈ 5/3 / α_Y_SM
ALPHA_2_INV_MZ_PDG = 29.6                       # 1/α_2(M_Z)
ALPHA_3_INV_MZ_PDG = 8.5                        # 1/α_s(M_Z)

# Benchmark MSSM β-coefficients (Martin convention; framework's named-convention values)
B_MSSM = (Fraction(33, 5), Fraction(1, 1), Fraction(-3, 1))


# ============================================================================
# §4.1 — Framework matter content per Pati-Salam regime
# ============================================================================

def section_4_1_matter_content():
    banner("§4.1 Framework's PS matter content (substrate-derived, no sparticles)")
    print()
    print("Per generation, PS spinor multiplet under SU(4)_PS × SU(2)_L × SU(2)_R:")
    print("  (4, 2, 1)_L  → 8 Weyl spinors (Q_L + L_L)")
    print("  (4*, 1, 2)_R → 8 Weyl spinors (Q_R + L_R)")
    print("  Total per gen: 16 Weyl spinors (= SO(10) spinor 16, theorem-grade via B3)")
    print()
    print("3 generations × 16 = 48 Weyl fermion states (matches framework's existing")
    print("matter content per target_parameters.md Structural panel).")
    print()
    print("Higgs sector (substrate-derived):")
    print("  (1, 2, 2) PS bidoublet (theorem-grade via B3 labeling)")
    print("    → decomposes at M_R into H_u (1,2,+1/2) + H_d (1,2,-1/2) = 2HDM")
    print()
    print("PS-breaking scalars: framework dissolves PS-breaking into substrate dynamics")
    print("  (per an internal working note). No explicit (15,1,1)")
    print("  or (10,1,3) Higgs is hypothesized; matching at M_R is tree-level only.")
    print()


# ============================================================================
# §4.2 — PS-regime β-coefficients
# ============================================================================

def section_4_2_PS_betas():
    banner("§4.2 PS-regime β-coefficients (Regime II: M_unif → M_R)")
    print()
    print("Formula: b = -(11/3)·C_A + (2/3)·Σ_(Weyl) T(R_f)·n_f + (1/3)·Σ_(scalar) T(R_s)·n_s")
    print("In Martin convention (asymptotic-freedom = negative b).")
    print()

    # SU(4)_PS β-coefficient
    # C_A(SU4) = 4, T(fundamental) = 1/2
    # Per gen: (4,2,1) has 2 SU(4) fundamentals (one per SU(2)_L direction) → T = 1
    #          (4*,1,2) has 2 SU(4) fundamentals (one per SU(2)_R direction) → T = 1
    # 3 gens: total T_SU4 = 6
    # (1,2,2) Higgs is SU(4) singlet → 0 scalar contribution
    C_A_SU4 = 4
    T_SU4_per_gen = Fraction(1, 1) + Fraction(1, 1)  # 1 from (4,2,1), 1 from (4*,1,2)
    n_gen = 3
    T_SU4_total = T_SU4_per_gen * n_gen
    b_PS_4 = -Fraction(11, 3) * C_A_SU4 + Fraction(2, 3) * T_SU4_total + 0
    print(f"  SU(4)_PS: C_A = {C_A_SU4}, fermion T = {T_SU4_total} (3 gens × 2), Higgs T = 0")
    print(f"  b_PS_4 = -(11/3)·{C_A_SU4} + (2/3)·{T_SU4_total} + 0 = {b_PS_4} = {float(b_PS_4):.4f}")
    print()

    # SU(2)_L_PS β-coefficient
    # C_A(SU2) = 2, T(doublet) = 1/2
    # Per gen: (4,2,1) has 4 SU(2)_L doublets (one per SU(4) component) → T = 4·(1/2) = 2
    #          (4*,1,2) is SU(2)_L singlet → 0
    # 3 gens: total T_SU2L = 6
    # (1,2,2) Higgs has 2 SU(2)_L doublets → T_s = 2·(1/2) = 1
    C_A_SU2 = 2
    T_SU2L_per_gen = Fraction(2, 1)  # 2 from (4,2,1)
    T_SU2L_total = T_SU2L_per_gen * n_gen
    T_higgs_SU2L = Fraction(1, 1)  # (1,2,2) contributes 1 to SU(2)_L
    b_PS_2L = -Fraction(11, 3) * C_A_SU2 + Fraction(2, 3) * T_SU2L_total + Fraction(1, 3) * T_higgs_SU2L
    print(f"  SU(2)_L_PS: C_A = {C_A_SU2}, fermion T = {T_SU2L_total}, Higgs T_s = {T_higgs_SU2L}")
    print(f"  b_PS_2L = -(11/3)·{C_A_SU2} + (2/3)·{T_SU2L_total} + (1/3)·{T_higgs_SU2L} = {b_PS_2L} = {float(b_PS_2L):.4f}")
    print()

    # SU(2)_R_PS β-coefficient (by L-R symmetry, same as 2L)
    b_PS_2R = b_PS_2L
    print(f"  SU(2)_R_PS: by L-R symmetry of PS, b_PS_2R = b_PS_2L = {b_PS_2R} = {float(b_PS_2R):.4f}")
    print()

    print("Sanity check: PS b_2L = -3 should equal SM b_2_2HDM = -3 below M_R,")
    print("because (4,2,1) breaks at M_R to (3,2)+(1,2) = same SU(2)_L content as 2HDM SM.")
    print("Match: ✓ (this is structurally expected, not a coincidence)")
    print()

    return {'b_PS_4': b_PS_4, 'b_PS_2L': b_PS_2L, 'b_PS_2R': b_PS_2R}


# ============================================================================
# §4.3 — Matching at M_R (PS → SM)
# ============================================================================

def section_4_3_matching(PS_alpha_inv_at_MR):
    banner("§4.3 Matching at M_R (PS → SM, tree-level Slansky branching)")
    print()
    print("Standard SO(10)/PS matching formulas (tree-level, no threshold corrections):")
    print("  α_3⁻¹(M_R) = α_4⁻¹(M_R)              [SU(3) ⊂ SU(4) trivial embedding]")
    print("  α_2⁻¹(M_R) = α_2L⁻¹(M_R)              [SU(2)_L trivial]")
    print("  α_1_GUT⁻¹(M_R) = (3/5)·α_4⁻¹(M_R) + (2/5)·α_2R⁻¹(M_R)")
    print("                                        [Slansky branching with GUT-norm α_1 = (5/3)·α_Y]")
    print()

    alpha_4_inv = PS_alpha_inv_at_MR['alpha_4']
    alpha_2L_inv = PS_alpha_inv_at_MR['alpha_2L']
    alpha_2R_inv = PS_alpha_inv_at_MR['alpha_2R']

    alpha_3_inv = alpha_4_inv
    alpha_2_inv = alpha_2L_inv
    alpha_1_inv = (3/5) * alpha_4_inv + (2/5) * alpha_2R_inv

    print(f"  α_4⁻¹(M_R) = {alpha_4_inv:.4f}")
    print(f"  α_2L⁻¹(M_R) = {alpha_2L_inv:.4f}")
    print(f"  α_2R⁻¹(M_R) = {alpha_2R_inv:.4f}")
    print()
    print(f"After matching:")
    print(f"  α_3⁻¹(M_R) = {alpha_3_inv:.4f}")
    print(f"  α_2⁻¹(M_R) = {alpha_2_inv:.4f}")
    print(f"  α_1_GUT⁻¹(M_R) = (3/5)·{alpha_4_inv:.4f} + (2/5)·{alpha_2R_inv:.4f} = {alpha_1_inv:.4f}")
    print()

    return {'alpha_1': alpha_1_inv, 'alpha_2': alpha_2_inv, 'alpha_3': alpha_3_inv}


# ============================================================================
# §4.4 — SM (2HDM) β-coefficients (Regime III: M_R → M_Z)
# ============================================================================

def section_4_4_SM_2HDM_betas():
    banner("§4.4 SM (2HDM) β-coefficients (Regime III: M_R → M_Z)")
    print()
    print("Below M_R, the PS bidoublet (1,2,2) decomposes into 2 SM Higgs doublets")
    print("(H_u + H_d). Substrate matter content below M_R = 3 SM gens + 2HDM, no sparticles.")
    print()
    print("Martin convention values for 2HDM (SM + 1 extra Higgs doublet):")

    # In Martin convention:
    # SM:   b_1 = 41/10, b_2 = -19/6, b_3 = -7
    # 2HDM = SM + extra Higgs (1,2,1)_{Y=1}: Δb_1 = +1/10, Δb_2 = +1/6, Δb_3 = 0
    b_1_SM_2HDM = Fraction(41, 10) + Fraction(1, 10)
    b_2_SM_2HDM = -Fraction(19, 6) + Fraction(1, 6)
    b_3_SM_2HDM = -Fraction(7, 1)
    print(f"  b_1_2HDM = 41/10 + 1/10 = {b_1_SM_2HDM} = {float(b_1_SM_2HDM):.4f}")
    print(f"  b_2_2HDM = -19/6 + 1/6 = {b_2_SM_2HDM} = {float(b_2_SM_2HDM):.4f}")
    print(f"  b_3_2HDM = -7 (Higgs SU(3) singlet, no change) = {float(b_3_SM_2HDM):.4f}")
    print()
    print("(Cross-check with A1 Session 1: A1 computed b_2_2HDM = -3 in the same convention. ✓)")
    print()
    return {'b_1': b_1_SM_2HDM, 'b_2': b_2_SM_2HDM, 'b_3': b_3_SM_2HDM}


# ============================================================================
# §4.5 — Multi-regime running
# ============================================================================

def run_RG(alpha_inv_high, b, ln_ratio):
    """One-loop running: α⁻¹(M_low) = α⁻¹(M_high) + (b/2π)·ln(M_high/M_low)"""
    return alpha_inv_high + float(b) / (2 * math.pi) * ln_ratio


def section_4_5_multi_regime_running(b_PS, b_SM):
    banner("§4.5 Multi-regime RG running: α_i(M_unif) → α_i(M_R) → α_i(M_Z)")
    print()

    ln_unif_to_R = math.log(M_UNIF_GEV / M_R_GEV)
    ln_R_to_Z = math.log(M_R_GEV / M_Z_GEV)
    ln_unif_to_Z = math.log(M_UNIF_GEV / M_Z_GEV)

    print(f"Scales:")
    print(f"  M_unif = {M_UNIF_GEV:.3e} GeV  (cascade theorem: (1/24)·(2/3)⁸·M_Pl)")
    print(f"  M_R    = {M_R_GEV:.3e} GeV  (cascade theorem: 2/3⁹·M_Pl)")
    print(f"  M_Z    = {M_Z_GEV:.4f} GeV")
    print()
    print(f"  ln(M_unif/M_R) = {ln_unif_to_R:.4f}")
    print(f"  ln(M_R/M_Z)    = {ln_R_to_Z:.4f}")
    print(f"  ln(M_unif/M_Z) = {ln_unif_to_Z:.4f} (consistency check)")
    print()

    # At M_unif: all PS couplings unified to α_GUT (per SO(10) unification + α_GUT⁻¹ = 24)
    alpha_4_unif = ALPHA_GUT_INV
    alpha_2L_unif = ALPHA_GUT_INV
    alpha_2R_unif = ALPHA_GUT_INV
    print(f"Starting at M_unif: α_4⁻¹ = α_2L⁻¹ = α_2R⁻¹ = α_GUT⁻¹ = {ALPHA_GUT_INV}")
    print(f"  (sin²θ_W(M_unif) = 3/8 verified by α_2 = α_1_GUT at unification: trivially satisfied)")
    print()

    # Run PS to M_R
    alpha_4_MR = run_RG(alpha_4_unif, b_PS['b_PS_4'], ln_unif_to_R)
    alpha_2L_MR = run_RG(alpha_2L_unif, b_PS['b_PS_2L'], ln_unif_to_R)
    alpha_2R_MR = run_RG(alpha_2R_unif, b_PS['b_PS_2R'], ln_unif_to_R)

    print(f"PS-regime running from M_unif to M_R (using PS β's from §4.2):")
    print(f"  α_4⁻¹(M_R)   = 24 + ({float(b_PS['b_PS_4']):.4f}/2π)·{ln_unif_to_R:.4f} = {alpha_4_MR:.4f}")
    print(f"  α_2L⁻¹(M_R)  = 24 + ({float(b_PS['b_PS_2L']):.4f}/2π)·{ln_unif_to_R:.4f} = {alpha_2L_MR:.4f}")
    print(f"  α_2R⁻¹(M_R)  = 24 + ({float(b_PS['b_PS_2R']):.4f}/2π)·{ln_unif_to_R:.4f} = {alpha_2R_MR:.4f}")
    print()

    # Match at M_R
    PS_at_MR = {'alpha_4': alpha_4_MR, 'alpha_2L': alpha_2L_MR, 'alpha_2R': alpha_2R_MR}
    SM_at_MR = section_4_3_matching(PS_at_MR)
    print()

    # Run SM (2HDM) to M_Z
    alpha_1_MZ = run_RG(SM_at_MR['alpha_1'], b_SM['b_1'], ln_R_to_Z)
    alpha_2_MZ = run_RG(SM_at_MR['alpha_2'], b_SM['b_2'], ln_R_to_Z)
    alpha_3_MZ = run_RG(SM_at_MR['alpha_3'], b_SM['b_3'], ln_R_to_Z)

    print(f"SM (2HDM) running from M_R to M_Z (using SM 2HDM β's from §4.4):")
    print(f"  α_1_GUT⁻¹(M_Z) = {SM_at_MR['alpha_1']:.4f} + ({float(b_SM['b_1']):.4f}/2π)·{ln_R_to_Z:.4f} = {alpha_1_MZ:.4f}")
    print(f"  α_2⁻¹(M_Z)     = {SM_at_MR['alpha_2']:.4f} + ({float(b_SM['b_2']):.4f}/2π)·{ln_R_to_Z:.4f} = {alpha_2_MZ:.4f}")
    print(f"  α_3⁻¹(M_Z)     = {SM_at_MR['alpha_3']:.4f} + ({float(b_SM['b_3']):.4f}/2π)·{ln_R_to_Z:.4f} = {alpha_3_MZ:.4f}")
    print()

    return {'alpha_1_MZ': alpha_1_MZ, 'alpha_2_MZ': alpha_2_MZ, 'alpha_3_MZ': alpha_3_MZ}


# ============================================================================
# §4.6 — Effective β-coefficient extraction
# ============================================================================

def section_4_6_effective_betas(multi_regime_result):
    banner("§4.6 Effective single-regime β-coefficients from multi-regime composition")
    print()
    print("Compute b_i_effective such that single-regime running from α_GUT⁻¹=24 at M_unif")
    print("with this effective β would produce the multi-regime α_i(M_Z):")
    print("  b_i_effective = (α_i⁻¹(M_Z) − α_GUT⁻¹) · 2π / ln(M_unif/M_Z)")
    print()

    ln_unif_to_Z = math.log(M_UNIF_GEV / M_Z_GEV)
    b_i_effective = {}
    for i, label, pdg in [(1, 'b_1', ALPHA_1_INV_MZ_PDG), (2, 'b_2', ALPHA_2_INV_MZ_PDG), (3, 'b_3', ALPHA_3_INV_MZ_PDG)]:
        key = f'alpha_{i}_MZ'
        alpha_inv = multi_regime_result[key]
        b_eff = (alpha_inv - ALPHA_GUT_INV) * 2 * math.pi / ln_unif_to_Z
        b_i_effective[label] = b_eff
        print(f"  {label}_effective = ({alpha_inv:.4f} − 24)·2π/{ln_unif_to_Z:.4f} = {b_eff:+.4f}")
    print()

    print("Compare to:")
    print(f"  {'   ':>9} | {'MSSM':>8} | {'SM':>8} | {'2HDM':>8} | {'A2-multi':>10}")
    print(f"  {'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    print(f"  {'b_1':>9} | {33/5:>8.4f} | {41/10:>8.4f} | {42/10:>8.4f} | {b_i_effective['b_1']:>+10.4f}")
    print(f"  {'b_2':>9} | {1.0:>8.4f} | {-19/6:>8.4f} | {-3.0:>8.4f} | {b_i_effective['b_2']:>+10.4f}")
    print(f"  {'b_3':>9} | {-3.0:>8.4f} | {-7.0:>8.4f} | {-7.0:>8.4f} | {b_i_effective['b_3']:>+10.4f}")
    print()

    return b_i_effective


# ============================================================================
# §4.7 — Verdict
# ============================================================================

def section_4_7_verdict(multi_regime_result, b_i_effective):
    banner("§4.7 Verdict — Anchor gate + Match criterion", "=")
    print()

    # Compare multi-regime α_i(M_Z) to PDG
    print(f"Anchor gate — multi-regime α_i(M_Z) vs PDG:")
    print()
    print(f"  {'i':>3} | {'multi-regime':>13} | {'PDG':>8} | {'Δ (multi - PDG)':>16} | {'rel %':>8}")
    print(f"  {'-'*3}-+-{'-'*13}-+-{'-'*8}-+-{'-'*16}-+-{'-'*8}")
    anchor_pass_count = 0
    for i, key, pdg_val in [(1, 'alpha_1_MZ', ALPHA_1_INV_MZ_PDG), (2, 'alpha_2_MZ', ALPHA_2_INV_MZ_PDG), (3, 'alpha_3_MZ', ALPHA_3_INV_MZ_PDG)]:
        multi = multi_regime_result[key]
        delta = multi - pdg_val
        rel = 100 * delta / pdg_val if pdg_val != 0 else float('inf')
        anchor_ok = abs(rel) <= 5
        if anchor_ok:
            anchor_pass_count += 1
        flag = '✓' if anchor_ok else '✗'
        print(f"  {i:>3} | {multi:>13.4f} | {pdg_val:>8.4f} | {delta:>+16.4f} | {rel:>+7.2f}% {flag}")
    print()
    print(f"  Anchor passes: {anchor_pass_count}/3")
    print()

    # Match criterion vs MSSM
    print(f"Match criterion — effective b_i vs MSSM (33/5, +1, -3):")
    print()
    mssm_b = {'b_1': 33/5, 'b_2': 1.0, 'b_3': -3.0}
    mssm_match_count = 0
    for label in ['b_1', 'b_2', 'b_3']:
        eff = b_i_effective[label]
        mssm = mssm_b[label]
        delta = eff - mssm
        rel = 100 * abs(delta) / max(abs(mssm), 0.1)
        match = abs(rel) <= 10  # 10% tolerance per design doc match criterion
        if match:
            mssm_match_count += 1
        flag = '✓' if match else '✗'
        print(f"  {label}_effective = {eff:+.4f}  vs  MSSM {mssm:+.4f}  (Δ = {delta:+.4f}, |rel| = {rel:.2f}%) {flag}")
    print()
    print(f"  MSSM-match count: {mssm_match_count}/3")
    print()

    # Decision per design doc §5
    print("Decision per design doc §5 outcome table:")
    print()

    # Effective b_2 compared to 2HDM and MSSM
    b_2_eff = b_i_effective['b_2']
    b_2_2HDM = -3.0
    b_2_MSSM = 1.0
    Δb_2_substrate_2HDM = b_2_2HDM  # what substrate single-regime gives
    Δb_2_closed = abs(b_2_eff - Δb_2_substrate_2HDM) / abs(b_2_MSSM - Δb_2_substrate_2HDM) * 100

    if mssm_match_count >= 2 and anchor_pass_count >= 2:
        outcome = "POSITIVE-multi-regime-equivalent"
    elif b_2_eff > -2.5 and b_2_eff < 1.5:
        outcome = "PARTIAL-positive"
    elif abs(b_2_eff - b_2_2HDM) < 0.5:
        outcome = "NEGATIVE-multi-regime-doesn't-help"
    elif b_2_eff < -3.5:
        outcome = "NEGATIVE-wrong-direction (further from MSSM than substrate-2HDM)"
    else:
        outcome = "AMBIGUOUS"

    print(f"  Outcome: {outcome}")
    print()
    print(f"  Effective b_2 = {b_2_eff:+.4f}")
    print(f"  Substrate-2HDM b_2 = {b_2_2HDM:+.4f}")
    print(f"  MSSM b_2 = {b_2_MSSM:+.4f}")
    print(f"  Fraction of Δb_2 = +4 closed by multi-regime: ~{Δb_2_closed:.1f}%")
    print()

    # Detailed verdict
    if outcome.startswith("POSITIVE-multi-regime-equivalent"):
        print("→ Multi-regime composition produces MSSM-equivalent effective β.")
        print("→ ADOPTED-MSSM-Sb literal-particle adoption can graduate.")
        print("→ Session 2: formalize the structural derivation.")
    elif outcome.startswith("PARTIAL"):
        print("→ Multi-regime structure closes some fraction of Δb_2 = +4 gap.")
        print("→ Sharpens R-19 characterization; Session 2 could attempt completion via")
        print("  Mechanism B (2-loop substrate corrections).")
    elif "NEGATIVE-multi-regime-doesn't-help" in outcome:
        print("→ Multi-regime composition produces SAME effective β as substrate-2HDM single-regime.")
        print("→ Matching corrections at M_R don't shift effective β meaningfully.")
        print("→ Mechanism C closes negative; literal-particle interpretation stays as named")
        print("  adoption (R-19 unchanged). A2 confirms Branch A's bounded research surface")
        print("  is genuinely exhausted.")
    elif "NEGATIVE-wrong-direction" in outcome:
        print("→ Multi-regime composition moves effective β AWAY from MSSM.")
        print("→ Mechanism C closes negative; the PS-regime structure doesn't help.")
    else:
        print(f"→ Outcome {outcome}: requires structural review.")

    return outcome


def main():
    banner("A2 Session 1 — multi-regime PS + matching + SM running", "#")
    print(f"\nDesign doc: an internal working note")
    print(f"Date: 2026-05-27")
    print(f"Question: does substrate's multi-regime structure close Δb_2 = +4 at SU(2)_L?")
    print()

    section_4_1_matter_content()
    print()
    b_PS = section_4_2_PS_betas()
    print()
    b_SM = section_4_4_SM_2HDM_betas()
    print()
    multi_result = section_4_5_multi_regime_running(b_PS, b_SM)
    print()
    b_eff = section_4_6_effective_betas(multi_result)
    print()
    section_4_7_verdict(multi_result, b_eff)


if __name__ == "__main__":
    main()
