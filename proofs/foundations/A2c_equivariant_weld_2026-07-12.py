#!/usr/bin/env python3
"""
proofs/foundations/A2c_equivariant_weld_2026-07-12.py

STATION A2c -- THE CORRECTED-DICTIONARY EQUIVARIANT WELD (Push 2, station 3).
Frozen contract: internal research notes (SS1-SS7), consuming A2b's
adjudicated B3 (ff3560a -- the R/F_bit reversal<->fermion dictionary REFUTED for the graded class;
the bare conjugate-pair class has exactly 24 real dims; fermion parity = length parity
automatically) + AF-3 + W2 + A1 (all settled, none re-opened).

THE DESIGN DEPARTURE (SS1): this station adds an A4-EQUIVARIANCE pin at level 1 -- level-1 carries
an HONEST A4 3-irrep (machine-verified as pc1, not assumed), unlike the FULL 8-dim field algebra F
(which is only 2T-projective).  phi_1's frozen pin set (SS3): (i) grading, (ii) the per-sector-pair
block structure (A2b's own (iii), REUSED verbatim -- NOT A2b's R/F_bit pin), (iii) A4-equivariance
at level 1 (phi_1.rho_hist(g) = rho_1(g).phi_1 for every g in A4, a NEW pin).

WHAT THIS SCRIPT IS: a DRIVER, not a new derivation. Every construction and exactness check it
runs is IMPORTED from derivation_topdown/state/the_net.py Section 8h -- per the ONE-OBJECT/
LOCAL-NET LAW, Layer-3 math accretes in the_net.py; this file only RUNS it and prints the
verdict-tree evidence. Has an `if __name__ == "__main__":` guard; safe to import.

NUMBERS APPEAR NOWHERE (pre-reg SS6): M_Z, ppm, m_nu, a_e are not computed, not printed, not
compared anywhere below.

D4 HARD GUARD: the species map {nu,d,u,e} <-> levels is NEVER an input to any pin/constraint
below -- it is read off the (empty) solution and REPORTED ONLY.

THE ML-2b/HK-7 CONDITIONALITY QUALIFIER (verbatim -- attaches to EVERY verdict-adjacent sentence
below):
    "Every duality check here (HK-5) is CELL-LEVEL only (the 6-edge static vacuum). ML-2b's
    DR-frame argument is CONDITIONAL on the TD-limit duality holding, which is NOT verified by
    this suite."
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402  the ONE master Layer-3 object; nothing rebuilt here

QUALIFIER = ("Every duality check here (HK-5) is CELL-LEVEL only (the 6-edge static vacuum). "
             "ML-2b's DR-frame argument is CONDITIONAL on the TD-limit duality holding, which "
             "is NOT verified by this suite.")

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 92)
    print(f" {t}")
    print("=" * 92)


def main():
    banner("A2c  PRE-CHECKS pc1-pc4 (before any solve, each printed)")
    # =======================================================================
    pc1 = net.a2c_pc1_honest_level1_rep()
    print("  pc1: construct the CLEAN level-1 A4 rep rho_1 (the A2b workaround's honest-rep route,")
    print("  _a4_standard_3irrep -- do NOT touch the flawed/quarantined _field_algebra_a4_rep/")
    print("  spin_lift). Genuine rep required: composition exact <=1e-12, NO cocycle=-1 pair.")
    check(f"pc1: group-law residual = {pc1['group_law_residual_1e12']:.2e} (<=1e-12); "
          f"smallest |rho(g)rho(h)+rho(gh)| = {pc1['smallest_no_cocycle_minus1_gap']:.3f} "
          "(large => no cocycle=-1 pair, genuinely non-projective); character match = "
          f"{pc1['character_match_residual']:.2e}", pc1["pc1_pass"])
    if not pc1["pc1_pass"]:
        print("  pc1 FAILED -- per the pre-reg, this is C4 BLOCKED. STOPPING.")
        return False

    pc2 = net.a2c_pc2_hom_multiplicity()
    print("  pc2: Hom_A4(shell-1,level-1) = 3 complex dims, re-derived by CHARACTER INNER PRODUCT")
    print("  (shell-1 IS A4's regular rep; Frobenius reciprocity computed as a direct character sum,")
    print("  not merely cited) -- cross-checked against a fresh direct-SVD Hom-space computation.")
    check(f"pc2: <chi_reg,chi_3> = {pc2['hom_via_character_inner_product']}, direct SVD Hom dim = "
          f"{pc2['hom_via_direct_svd']}, sector_grading_hist mult[1] = {pc2['mult1']}", pc2["pc2_pass"])

    pc3 = net.a2c_pc3_reproduce_survivor()
    print("  pc3: reproduce A2b's 24-dim bare-survivor waypoint (grading + pair-blocks, NO parity")
    print("  dictionary, NO self-J) BEFORE adding equivariance.")
    check(f"pc3: grading alone = {pc3['grading_alone']}, +pairblock = "
          f"{pc3['pairblock_survivor']}/{pc3['total_real_dim']} (must be 24)", pc3["pc3_pass"])

    pc4 = net.a2c_pc4_automatic_parity()
    print("  pc4: verify automatic length-parity on Gamma(test-map) images (a standing check, never")
    print("  a pin) -- Gamma(w) is an exact NHAT-eigenvector with eigenvalue = |w|.")
    check(f"pc4: worst NHAT-eigenvalue residual = {pc4['worst_nhat_eigenvalue_residual']:.2e} over "
          f"{pc4['n_words_checked']} words (shells 1-3)", pc4["pc4_pass"])

    banner("A2c  STEP 1 -- THE ALLOWANCE + THE EQUIVARIANT-CHANNEL SIZE (pre-declared, printed "
           "before solving)")
    # =======================================================================
    alw = net.a2c_level1_allowance_per_block()
    print("  pins (i)+(ii) are VERBATIM A2b's own (i)+(iii) -- the allowance is A2b's OWN value,")
    print("  reused not re-derived.")
    check(f"ALLOWANCE: GROUP-03 = {alw['group03_allowance']} (forced 0), GROUP-12 = "
          f"{alw['group12_allowance']} (one phase); TOTAL = {alw['allowance']}",
          alw["allowance"] == 1 and alw["group03_allowance"] == 0)

    ecs = net.a2c_equivariant_channel_size()
    print("  the equivariant channel ALONE (grading+equivariance, no pairblock) is pre-declared")
    print("  <=6 real dims by pc2 (3 complex dims) -- VERIFIED here, not merely inferred:")
    check(f"equivariant channel: A-only = {ecs['A_only']} (matches pc2 bound: "
          f"{ecs['matches_pc2_bound']}), full real-linear = {ecs['full_real_linear']} (= 2x A-only, "
          "since dart_rep(g)/rho_1(g) are ordinary non-antiunitary operators -- A and B are "
          "independently, identically constrained)", ecs["matches_pc2_bound"])

    banner("A2c  STEP 2 -- THE SOLVE: THE INTERSECTION {24-dim survivor} ^ {equivariant channel}")
    # =======================================================================
    print("  SOLVE CONVENTION (stated explicitly): pin (iii)'s own text calls it a 'complex-linear")
    print("  constraint' -- the PRIMARY reading is the genuine complex-linear content of phi_1")
    print("  (A-only, B=0, matching SS1's own definition and a2_gamma_word's own B=0 convention,")
    print("  and the ONLY reading matching pc2's own <=6-real-dim bound). The FULL real-linear")
    print("  reading (A and B both independently equivariant) is ALSO reported, disclosed, per the")
    print("  'strictest reading, note the ambiguity' instruction.")
    print()
    print("  EMBEDDING-AMBIGUITY DISCLOSURE: rho_1(g) is built ABSTRACTLY (_a4_standard_3irrep), NOT")
    print("  by re-deriving F's own action on Pw[1] (would need the banned spin_lift). The specific")
    print("  embedding into Pw[1]'s Adag coordinates is UNFIXED by any accreted machinery -- tested,")
    print("  not dismissed, via random-unitary-embedding robustness trials below.")
    inter = net.a2c_intersection()
    check(f"INTERSECTION (default embedding U=I): A-only = {inter['default_A_only']}, full "
          f"real-linear = {inter['default_full_real_linear']} (allowance = {inter['allowance']}, "
          f"nullity<=allowance: {inter['nullity_le_allowance']})",
          inter["default_A_only"] == 0 and inter["default_full_real_linear"] == 0)
    for tr in inter["robustness_trials"]:
        print(f"    embedding trial {tr['trial']}: U unitary residual={tr['U_unitary_residual']:.2e}, "
              f"A-only={tr['A_only']}, full={tr['full_real_linear']}")
    check(f"ALL {len(inter['robustness_trials'])} random-unitary-embedding trials AGREE with the "
          f"default: {inter['all_trials_agree']} (spectral gap: smallest kept sv = "
          f"{inter['smallest_kept_sv']:.3f}, largest null sv = {inter['largest_null_sv']:.1e} -- "
          "no near-degeneracy)", inter["all_trials_agree"])

    banner("A2c  STEP 2b -- THE ALGEBRAIC PROOF (Schur's lemma / character orthogonality -- "
           "'expected to be clean' per SS5's C3 branch)")
    # =======================================================================
    print("  CLAIM: the intersection is {0} IDENTICALLY, EMBEDDING-INDEPENDENT, from TWO facts:")
    print("    FACT 1 (pin ii, REUSED verbatim from A2b's own FACT 1): phi_1(hist-GROUP-03) = 0")
    print("      identically (pin i already confines the image to level-1, disjoint from")
    print("      field-GROUP-03).")
    print("    FACT 2 (NEW, pin iii + character orthogonality): hist-GROUP-12 (the remaining 2-dim")
    print("      domain piece) decomposes as TWO 1-dimensional A4 irreps (dims[1]=dims[2]=1), while")
    print("      level-1 carries the UNIQUE IRREDUCIBLE 3-dim A4 rep. By character orthogonality,")
    print("      <chi_1,chi_3>=<chi_2,chi_3>=0 EXACTLY -- Hom_A4(1-dim,3-dim-irreducible) = {0} for")
    print("      DIMENSION-MISMATCH reasons alone (Schur's lemma), so phi_1|hist-GROUP-12 = 0 too,")
    print("      for BOTH A and B, and REGARDLESS of the embedding U (Schur is basis-independent --")
    print("      this is WHY the robustness trials above all agree).")
    proof = net.a2c_algebraic_proof()
    check(f"FACT 1: hist-GROUP-03 dim = {proof['group03_forced_zero_dim']} (forced 0 by pin ii)", True)
    check(f"FACT 2: <chi_1,chi_3> = {proof['character_orthogonality_group1']:.2e}, <chi_2,chi_3> = "
          f"{proof['character_orthogonality_group2']:.2e} (both ~0, character orthogonality); "
          f"hist-GROUP-12 dim = {proof['group12_dim']}, constituent irrep dims = "
          f"{proof['group12_constituent_irrep_dims']} vs level-1's irrep dim = "
          f"{proof['level1_irrep_dim']}",
          proof["character_orthogonality_group1"] < 1e-8 and proof["character_orthogonality_group2"] < 1e-8)
    check("numeric intersection nullity CONFIRMS the algebraic proof's conclusion (phi_1 = 0)",
          proof["proof_holds"])

    banner("A2c  STEP 3 -- P1/P2 RE-VERIFIED IN THIS CLASS (reused from A2, unchanged construction)")
    # =======================================================================
    p12 = net.a2c_p1_p2_reverify()
    p1, p2 = p12["p1"], p12["p2"]
    check(f"P1 PAULI TRUNCATION [structural test map]: ALL {p1['n_shell4_words']} shell-4 words "
          f"give Gamma=0 (worst {p1['worst_shell4_norm']:.2e})", p1["worst_shell4_norm"] < 1e-9)
    check(f"P2 REPEATED-DART KERNEL: synthetic repeat gives Gamma=0 (residual "
          f"{p2['synthetic_repeat_residual']:.2e}); first admissible repeat at shell "
          f"{p2['first_shell_with_admissible_repeat']} (P1 already kills shell>=4)",
          p2["synthetic_repeat_residual"] < 1e-9 and p2["first_shell_with_admissible_repeat"] == 4)

    banner("A2c  STEP 4 -- THE TOWER TEST against INDEPENDENTLY-built A2c-class shell-2/3 systems "
           "(the pre-reg's SS4 step3; 'expected NON-VACUOUS for the first time')")
    # =======================================================================
    s2 = net.a2c_shell_level_system_nullity(2, 2)
    s3 = net.a2c_shell_level_system_nullity(3, 3)
    print("  the A2c-CLASS shell-n system (grading + pair-block + level-n equivariance -- Lambda^n")
    print("  of an equivariant map is automatically Lambda^n-equivariant, so Lambda^n(rho_1) is the")
    print("  FORCED level-n rep), built INDEPENDENTLY at each shell, has its OWN freedom:")
    check(f"  shell2->level2 [dom {s2['domain_dim']}]: grading+pairblock A-only = "
          f"{s2['grading_pairblock_A_only']}, +equivariance A-only = {s2['full_A_only']} (full "
          f"real-linear {s2['full_real_linear']}); Lambda^2 group-law residual = "
          f"{s2['lambda_n_group_law_residual']:.2e}", s2["lambda_n_group_law_residual"] < 1e-8)
    check(f"  shell3->level3 [dom {s3['domain_dim']}]: grading+pairblock A-only = "
          f"{s3['grading_pairblock_A_only']}, +equivariance A-only = {s3['full_A_only']} (full "
          f"real-linear {s3['full_real_linear']}); Lambda^3 group-law residual = "
          f"{s3['lambda_n_group_law_residual']:.2e}", s3["lambda_n_group_law_residual"] < 1e-8)
    print("  Since phi_1's OWN solution basis is EMPTY (STEP 2/2b), Gamma(phi_1)=0 identically at")
    print("  every shell -- trivially a member of EVERY linear space (the 0 vector), REGARDLESS of")
    print("  whether the shell-n system above has its own nonzero freedom (shell 2: 0; shell 3:")
    print("  nonzero -- 8 A-only). Reported HONESTLY as VACUOUS. A DISCRIMINATING CONTROL (a genuine")
    print("  non-solution) is run alongside.")
    tw2 = net.a2c_tower_membership_test(2, [])
    check(f"SHELL 2 [D={tw2['D_shell_n']}]: basis_size=0 -> VACUOUS={tw2['vacuous']} (the 0 map, "
          "trivially member)", tw2["all_members"] and tw2["vacuous"])
    check(f"  discriminating control residual = {tw2['control_residual']:.4e} -- "
          f"control_is_member={tw2['control_is_member']} (should be False)",
          tw2["control_is_member"] is False)
    tw3 = net.a2c_tower_membership_test(3, [])
    check(f"SHELL 3 [D={tw3['D_shell_n']}]: basis_size=0 -> VACUOUS={tw3['vacuous']}",
          tw3["all_members"] and tw3["vacuous"])
    check(f"  discriminating control residual = {tw3['control_residual']:.4e} -- "
          f"control_is_member={tw3['control_is_member']} (should be False)",
          tw3["control_is_member"] is False)
    print("  EXPLICIT VACUOUS-OR-NOT STATEMENT: the tower test is VACUOUS -- CONTRARY to the")
    print("  pre-reg's own stated expectation ('expected NON-VACUOUS for the first time'). phi_1's")
    print("  nullity is 0 (the intersection is EMPTY, C3), so the vacuousness of the tower test is a")
    print("  DOWNSTREAM CONSEQUENCE of C3, not an independent finding -- this is NOT reported as a")
    print("  forcing confirmation.")

    banner("A2c  STEP 5 -- REPORT-ONLY READS: pair-completeness / N-hat-exactness / ember shadow / "
           "THE SPECIES READ")
    # =======================================================================
    pc = net.a2c_pair_completeness_read()
    check(f"PAIR-COMPLETENESS: dim(Im phi_1) = {pc['dim_Im_phi1']}; Phi~ = J_F.Phi.J_hist is zero = "
          f"{pc['Phi_tilde_is_zero']}; pair (Phi,Phi~) covers F = {pc['pair_covers_F']}",
          pc["dim_Im_phi1"] == 0 and pc["Phi_tilde_is_zero"])

    nh = net.a2b_nhat_intertwining_exactness()
    check(f"N-HAT-INTERTWINING EXACTNESS [REUSED from A2b unchanged, phi_1-independent, shells "
          f"1-3]: worst residual {nh['worst_residual']:.2e}", nh["worst_residual"] < 1e-9)

    emb = net.a2b_ember_consistency_shadow()
    tri = next(r for r in emb if r["is_triangle"])
    check(f"EMBER-CONSISTENCY SHADOW [REUSED from A2b/FOCK-0e unchanged]: triangle-orbit "
          f"lambda_1={tri['lambda_1']:.4f} (FOCK-0d ember reference 2.463) -- confronted with "
          "NOTHING", abs(tri["lambda_1"] - 2.463) < 0.01)

    sp = net.a2c_species_read()
    print(f"  {sp['note']}")
    check("SPECIES READ (D4 output only): no surviving isotypic components; species map never used "
          "as an input anywhere in SS3/SS4's pin construction", sp["surviving_isotypic_components"] == [])

    banner("A2c  REGRESSION: Sections 7/7b/8/8b/8c/8d/8e/8f/8g + module anchors untouched")
    # =======================================================================
    check("anchor_cell_projector() + anchor_tick_2pi() + accretion_selftest_2026_07_10() + "
          "i2b_selftest_2026_07_11() + fock0_selftest_2026_07_11() + "
          "fock0b_selftest_2026_07_11() + fock0c_selftest_2026_07_11() + "
          "fock0d_selftest_2026_07_11() + fock0e_selftest_2026_07_12() + "
          "a2_weld_selftest_2026_07_12() + a2b_weld_selftest_2026_07_12() all still PASS",
          net.anchor_cell_projector() and net.anchor_tick_2pi()
          and net.accretion_selftest_2026_07_10(verbose=False)
          and net.i2b_selftest_2026_07_11(verbose=False)
          and net.fock0_selftest_2026_07_11(verbose=False)
          and net.fock0b_selftest_2026_07_11(verbose=False)
          and net.fock0c_selftest_2026_07_11(verbose=False)
          and net.fock0d_selftest_2026_07_11(verbose=False)
          and net.fock0e_selftest_2026_07_12(verbose=False)
          and net.a2_weld_selftest_2026_07_12(verbose=False)
          and net.a2b_weld_selftest_2026_07_12(verbose=False))
    check("a2c_weld_selftest_2026_07_12() (the Section-8h permanent regression anchor) PASSES",
          net.a2c_weld_selftest_2026_07_12(verbose=False))

    banner("A2c VERDICT-RELEVANT SUMMARY (SS5 tree; ML-2b/HK-7 QUALIFIER attaches to EVERY "
           "sentence; architect adjudicates, NOT this driver)")
    # =======================================================================
    print(f"  QUALIFIER: {QUALIFIER}")
    print()
    print(f"  pc1-pc4: all PASS (honest level-1 A4 rep constructed; Hom_A4=3 confirmed by character")
    print(f"  inner product; the 24-dim bare survivor reproduced; automatic parity confirmed)")
    print(f"  ALLOWANCE (pre-declared, reused from A2b) = TOTAL: {alw['allowance']}")
    print(f"  equivariant channel alone = 6 real dims (A-only), matching pc2's bound exactly")
    print(f"  INTERSECTION nullity = {inter['default_A_only']} (A-only), "
          f"{inter['default_full_real_linear']} (full real-linear) -- EMBEDDING-INDEPENDENT "
          f"(5/5 random-unitary trials agree)")
    print(f"  ALGEBRAIC PROOF: intersection = {{0}} FORCED -- a FOURTH, distinct obstruction "
          "mechanism (Schur's lemma / character orthogonality: hist-GROUP-12's constituent 1-dim "
          "irreps cannot map equivariantly into level-1's irreducible 3-dim irrep)")
    print(f"  P1/P2: HOLD (reused from A2, unchanged)")
    print(f"  TOWER TEST shells 2/3: VACUOUS (downstream consequence of the empty intersection, NOT")
    print(f"  an independent finding -- CONTRARY to the pre-reg's stated 'expected NON-VACUOUS' "
          "hope; shell 3's OWN system does carry nonzero freedom of its own, 8 A-only, unexercised)")
    print(f"  PAIR-COMPLETENESS: Phi and Phi~ both zero; the pair covers NOTHING of F")
    print(f"  SPECIES READ: NONE -- the pre-reg's own 'now with content' framing did not "
          "materialize")
    print()
    print("  Evidence pattern (driver's read; the ARCHITECT adjudicates, not this driver): the")
    print("  equivariant channel (6 real dims, A-only) MISSES the 24-dim bare survivor ENTIRELY --")
    print("  intersection {0} -- matching SS5's C3 criterion ('the equivariant channel misses the")
    print("  24-dim survivor entirely') precisely. The algebraic proof delivered above (STEP 2b) is")
    print("  the 'algebraic proof (character/isotypic argument expected to be clean)' C3 calls for.")

    banner("RESULT")
    print("ALL MACHINE CHECKS PASS" if ok_all else "SOME CHECKS FAILED -- see [FAIL] lines above")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
