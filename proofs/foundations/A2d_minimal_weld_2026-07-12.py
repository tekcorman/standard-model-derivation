#!/usr/bin/env python3
"""
proofs/foundations/A2d_minimal_weld_2026-07-12.py

STATION A2d -- THE MINIMAL WELD CLASS (Push 2, THE FINAL ARC STATION).
Frozen contract: internal research notes (SS0-SS7), consuming A2c's
adjudicated C3 (the pair-block pin's Schur obstruction, sealed-confirmed) + the A2c checker's own
classification diagnostics (working notes/A2c_check_2026-07-12.md SS5-SS6 -- the pair-block-free
channel is nonzero, 3 complex/6 real dims, entirely in the 3-isotypic; tower membership at machine
precision; the pin's J-free justification = inheritance alone).

THE DESIGN DEPARTURE (SS1): the pair-block pin is REMOVED entirely (theorem/audit-backed, not a
convenience) -- its two frozen justifications are BOTH dead in this class (all J-pins refuted; the
(Phi,Phi~) pair-object motivation is moot since Phi~==0 everywhere upstream), and it discards
EXACTLY the 3-irrep content, the only content that can EVER reach level-1 equivariantly.  The
frozen pin set is now: (i) grading, (ii) A4-equivariance at level 1 ONLY.

THE STOPPING RULE (SS0): this is the LAST station of the weld arc, whatever it returns -- no
further pin modifications after this one.

WHAT THIS SCRIPT IS: a DRIVER, not a new derivation. Every construction and exactness check it
runs is IMPORTED from derivation_topdown/state/the_net.py Section 8i -- per the ONE-OBJECT/
LOCAL-NET LAW, Layer-3 math accretes in the_net.py; this file only RUNS it and prints the
verdict-tree evidence. Has an `if __name__ == "__main__":` guard; safe to import.

NUMBERS APPEAR NOWHERE (pre-reg SS6 poisons, inherited): M_Z, ppm, m_nu, a_e are not computed,
not printed, not compared anywhere below.

D4 HARD GUARD: the species map {nu,d,u,e} <-> levels is NEVER an input to any pin/constraint
below -- it is read off the actual (nonzero) solution and REPORTED ONLY, in a2d_species_read.

NO PAIR-BLOCK RE-IMPOSITION under any framing, anywhere below (SS1's removal is final per SS0).

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
    banner("A2d  STEP 1 -- THE PRE-DECLARED ALLOWANCE (printed BEFORE any solve)")
    # =======================================================================
    print("  the pair-block pin is REMOVED (SS1) -- the frozen pin set is now grading (i) + level-1")
    print("  A4-equivariance (ii) ONLY. Hom_A4(shell-1,level-1) is the multiplicity space of the")
    print("  3-irrep inside shell-1 (a2c_pc2_hom_multiplicity's own <chi_reg,chi_3>=3, REUSED). By")
    print("  Schur's lemma, the physically-unfixable freedom within any FIXED direction among the 3")
    print("  copies is ONE overall complex scale-phase (2 real) -- the direction itself is NOT fixed")
    print("  by grading+equivariance alone, so this allowance is DELIBERATELY smaller than the full")
    print("  6-real-dim multiplicity space.")
    alw = net.a2d_allowance()
    check(f"multiplicity space = {alw['multiplicity_space_complex_dim']} complex "
          f"({alw['multiplicity_space_real_dim']} real); ALLOWANCE = {alw['allowance_real']} real",
          alw["allowance_real"] == 2 and alw["multiplicity_space_complex_dim"] == 3)

    banner("A2d  STEP 2 -- THE SOLVE: grading (i) + level-1 A4-equivariance (ii) ONLY, NO pair-block")
    # =======================================================================
    print("  solved via TWO independent routes: (a) the F-embedded route (REUSED, UNCHANGED --")
    print("  a2c_equivariant_channel_size's own grading+equivariance-only construction, which never")
    print("  stacked pair-block rows in the first place); (b) the ABSTRACT route (a genuinely")
    print("  DIFFERENT Sylvester system, no F-embedding/grading-row Kronecker machinery at all --")
    print("  pure Hom_A4(shell-1,level-1) via the honest rho3 directly). Both MUST agree, since they")
    print("  are bijective via phi = E_1^dagger . A (E_1 = _a2c_level_embedding(1)).")
    solve = net.a2d_solve_channel()
    check(f"F-embedded waypoint: A-only={solve['waypoint_A_only']}, full real-linear="
          f"{solve['waypoint_full_real_linear']}", solve["waypoint_A_only"] == 6)
    check(f"abstract route: nullity={solve['abstract_nullity_complex']} complex "
          f"(group-law residual={solve['abstract_group_law_residual']:.2e}); matches waypoint: "
          f"{solve['abstract_matches_waypoint']}", solve["abstract_matches_waypoint"])
    check(f"NULLITY = {solve['nullity_real']} real vs ALLOWANCE = {solve['allowance_real']} real -- "
          f"EXCEEDS: {solve['nullity_exceeds_allowance']} (the pre-reg's own D2 anticipation)",
          solve["nullity_exceeds_allowance"])

    banner("A2d  STEP 3 -- WAYPOINT REPRODUCTION (the A2c checker's independent diagnostics, "
           "reproduced on THIS station's own basis)")
    # =======================================================================
    iso = net.a2d_waypoint_isotypic_locus()
    print("  A2c checker item 5(iii): the pair-block-free channel is annihilated by domain-side")
    print("  isotypic projectors P[0],P[1],P[2] and lives ENTIRELY in P[3] (the 3-irrep's own 9-dim")
    print("  isotypic block) -- checked directly on this station's OWN abstract basis:")
    check(f"vanish on P[0..2] = {iso['worst_vanish_P012']:.2e}; supported on P[3] residual = "
          f"{iso['worst_supported_P3_residual']:.2e}; entirely 3-isotypic: "
          f"{iso['entirely_3_isotypic']}", iso["entirely_3_isotypic"])

    s1 = net.a2d_waypoint_shell_level_nullity(1, 1)
    s2 = net.a2d_waypoint_shell_level_nullity(2, 2)
    s3 = net.a2d_waypoint_shell_level_nullity(3, 3)
    print("  A2c checker item 5(iv): the pair-block-FREE shell-2/3 systems' own independent nullity")
    print("  (grading + Lambda^n-equivariance, no pair-block) -- checker's own numbers: shell2 = 12")
    print("  real A-only (24 full); shell3 = 8 real A-only (16 full):")
    check(f"shell1->level1 [dom {s1['domain_dim']}]: full_A_only={s1['full_A_only']} "
          f"(cross-check vs STEP 2's waypoint 6)", s1["full_A_only"] == 6)
    check(f"shell2->level2 [dom {s2['domain_dim']}]: full_A_only={s2['full_A_only']} (expect 12), "
          f"full_real_linear={s2['full_real_linear']} (expect 24)",
          s2["full_A_only"] == 12 and s2["full_real_linear"] == 24)
    check(f"shell3->level3 [dom {s3['domain_dim']}]: full_A_only={s3['full_A_only']} (expect 8), "
          f"full_real_linear={s3['full_real_linear']} (expect 16)",
          s3["full_A_only"] == 8 and s3["full_real_linear"] == 16)

    banner("A2d  STEP 4 -- Gamma(phi_1) LIVE, NONZERO, ON A CHANNEL BASIS (shells 1-4) -- P1/P2 "
           "run on REAL images for the FIRST TIME in the arc")
    # =======================================================================
    live = net.a2d_gamma_live_behavior()
    for r in live["basis_rows"]:
        ps = r["per_shell"]
        print(f"    basis[{r['basis_index']}]: shell1 worst={ps[1]['worst']:.4f}, "
              f"shell2 worst={ps[2]['worst']:.4f}, shell3 worst={ps[3]['worst']:.4f}, "
              f"shell4 worst={ps[4]['worst']:.2e}, repeat-word norm={r['repeat_word_norm']:.2e}")
    check(f"P1 (Pauli truncation): worst shell-4 norm over {live['n_basis']} basis elements = "
          f"{live['worst_shell4_norm']:.2e} (vanishes, GENERAL fact of the realization)",
          live["worst_shell4_norm"] < 1e-9)
    check(f"P2 (repeated-dart kernel): worst repeat-word norm = {live['worst_repeat_norm']:.2e} "
          "(vanishes)", live["worst_repeat_norm"] < 1e-9)
    check("shell-1 image is genuinely NONZERO (not a degenerate all-zero basis)",
          live["basis_rows"][0]["per_shell"][1]["worst"] > 1e-6)

    banner("A2d  STEP 5 -- THE TOWER CONSISTENCY CHECK (labeled CONSISTENCY, NEVER forcing -- "
           "pre-reg SS3's honesty clause)")
    # =======================================================================
    print("  Lambda^n of an equivariant map is AUTOMATICALLY Lambda^n-equivariant -- so membership")
    print("  of Gamma(phi_1)'s ACTUAL basis in the independently-built pair-block-free shell-2/3")
    print("  systems is MATHEMATICALLY GUARANTEED, not a forcing result. Verified here (with a")
    print("  discriminating control that MUST fail) -- reported as CONSISTENCY, per the pre-reg.")
    tow = net.a2d_tower_consistency()
    for shell_n in (2, 3):
        t = tow[shell_n]
        check(f"shell {shell_n} [D={t['D_shell_n']}]: all_members_consistency="
              f"{t['all_members_consistency']} (member residuals: "
              f"{[f'{r:.2e}' for r in t['member_residuals']]}); discriminating control residual="
              f"{t['control_residual']:.3f} (control REJECTED: {not t['control_is_member']})",
              t["all_members_consistency"] and not t["control_is_member"])
    print("  ANY claim of forcedness may ONLY come from STEP 2's nullity==allowance comparison --")
    print("  this consistency finding does NOT, by itself, force anything.")

    banner("A2d  STEP 6 -- THE FOUR READS (report-only; the station's real content)")
    # =======================================================================
    print("  READ (a) THE MULTIPLICITY GEOMETRY")
    mg = net.a2d_multiplicity_geometry()
    check(f"psi (copy-embedding) group-law residual={mg['psi_group_law_residual']:.2e}; phi "
          f"(channel-basis) group-law residual={mg['phi_group_law_residual']:.2e}",
          mg["psi_group_law_residual"] < 1e-8 and mg["phi_group_law_residual"] < 1e-8)
    check(f"Schur scalar-check residual (phi_i . psi_k = d_ik . I_3 for every i,k): "
          f"{mg['schur_scalar_check_residual']:.2e}", mg["schur_scalar_check_residual"] < 1e-8)
    print(f"    D matrix (each row = one basis element's direction in the 3-copy multiplicity "
          f"space): {mg['D_matrix']}")
    print(f"    ker(phi_i) isotypic dims per basis element (expect [1,1,1,6] -- P[0..2] fully "
          f"killed, P[3]'s 9-dim block splits 3 image / 6 kernel): "
          f"{[r['ker_isotypic_dims'] for r in mg['ker_isotypic_dims']]}")
    print(f"    {mg['outlook_note']}")

    print()
    print("  READ (b) PAIR-COMPLETENESS -- what Gamma(phi_1)(H_hist) covers of F, what J_F maps it")
    print("  to, what (Phi,Phi~) jointly span (CONTRAST A2/A2b/A2c: Phi=0 everywhere, pair covered")
    print("  NOTHING -- this is the first time in the arc there is real content to report here)")
    pc = net.a2d_pair_completeness_and_coverage()
    for r in pc["per_basis"]:
        cov = r["shell_coverage"]
        print(f"    basis[{r['basis_index']}]: shell1(level-1) image rank={cov[1]['image_rank']}/"
              f"{cov[1]['target_dim']}, shell2(level-2) rank={cov[2]['image_rank']}/"
              f"{cov[2]['target_dim']}, shell3(level-3) rank={cov[3]['image_rank']}/"
              f"{cov[3]['target_dim']}; Phi~ rank={r['Phi_tilde_rank']}, confined to Pw[2] residual="
              f"{r['Phi_tilde_confined_to_Pw2_residual']:.2e}; joint(Phi,Phi~) rank="
              f"{r['joint_Phi_Phitilde_rank']}")
    check("Phi~ = J_F.Phi.J_hist is confined to Pw[2] for every basis element (AF-3's K-swap, "
          "tested not assumed)", all(r["Phi_tilde_confined_to_Pw2_residual"] < 1e-6 for r in pc["per_basis"]))

    print()
    print("  READ (c) THE EMBER/CLOCK SHADOW -- intersection of the channel's image (Pw[1]) with")
    print("  K_F's eigenspaces per region orbit (principal-angle cosines; 1=exact containment, "
          "0=orthogonal)")
    emb = net.a2d_ember_shadow()
    for r in emb["per_region"]:
        print(f"    region {r['region']} (triangle={r['is_triangle']}, epsilon={r['epsilon']:.4f}):")
        for eg in r["eigengroups"]:
            print(f"       eigenvalue={eg['eigenvalue']:+.4f} (dim {eg['eigenspace_dim']}): "
                  f"cosines={[round(x, 4) for x in eg['principal_angle_cosines_with_Pw1']]}")
    check(f"N-hat-intertwining exactness [REUSED from A2b unchanged]: worst residual="
          f"{emb['nhat_exactness']['worst_residual']:.2e}", emb["nhat_exactness"]["worst_residual"] < 1e-9)

    print()
    print("  READ (d) THE SPECIES READ (D4 output only)")
    sp = net.a2d_species_read()
    print(f"    species_sector_dims = {sp['species_sector_dims']}")
    print(f"    {sp['correspondence']}")
    print(f"    {sp['level3_asymmetry_note']}")
    print(f"    {sp['outlook_note']}")
    check("isotypic vanishing checks confirm P[0..2] exactly killed for every basis element",
          all(c["vanishes_on_P012"] < 1e-8 for c in sp["isotypic_vanishing_checks"]))

    banner("A2d  REGRESSION: Sections 7/7b/8/8b/8c/8d/8e/8f/8g/8h + module anchors untouched")
    # =======================================================================
    check("anchor_cell_projector() + anchor_tick_2pi() + accretion_selftest_2026_07_10() + "
          "i2b_selftest_2026_07_11() + fock0_selftest_2026_07_11() + "
          "fock0b_selftest_2026_07_11() + fock0c_selftest_2026_07_11() + "
          "fock0d_selftest_2026_07_11() + fock0e_selftest_2026_07_12() + "
          "a2_weld_selftest_2026_07_12() + a2b_weld_selftest_2026_07_12() + "
          "a2c_weld_selftest_2026_07_12() all still PASS",
          net.anchor_cell_projector() and net.anchor_tick_2pi()
          and net.accretion_selftest_2026_07_10(verbose=False)
          and net.i2b_selftest_2026_07_11(verbose=False)
          and net.fock0_selftest_2026_07_11(verbose=False)
          and net.fock0b_selftest_2026_07_11(verbose=False)
          and net.fock0c_selftest_2026_07_11(verbose=False)
          and net.fock0d_selftest_2026_07_11(verbose=False)
          and net.fock0e_selftest_2026_07_12(verbose=False)
          and net.a2_weld_selftest_2026_07_12(verbose=False)
          and net.a2b_weld_selftest_2026_07_12(verbose=False)
          and net.a2c_weld_selftest_2026_07_12(verbose=False))
    check("a2d_weld_selftest_2026_07_12() (the Section-8i permanent regression anchor) PASSES",
          net.a2d_weld_selftest_2026_07_12(verbose=False))

    banner("A2d VERDICT-RELEVANT SUMMARY (SS5 tree; ML-2b/HK-7 QUALIFIER attaches to EVERY "
           "sentence; architect adjudicates, NOT this driver)")
    # =======================================================================
    print(f"  QUALIFIER: {QUALIFIER}")
    print()
    print(f"  ALLOWANCE (pre-declared) = {alw['allowance_real']} real (one overall complex "
          "scale-phase)")
    print(f"  NULLITY = {solve['nullity_real']} real (3 complex, the FULL multiplicity space) -- "
          f"EXCEEDS the allowance")
    print("  the weld EXISTS as a mapped family (the pair-block-free channel is nonzero for the")
    print("  first time in the arc), but its DIRECTION in the 3-dim multiplicity space is NOT")
    print("  selected by grading+equivariance alone -- THE MULTIPLICITY SELECTOR is the arc's final")
    print("  named incomplete equation.")
    print("  TOWER: CONSISTENT (guaranteed by Lambda^n functoriality, verified with a discriminating")
    print("  control that correctly fails) -- explicitly NOT forcing language.")
    print("  P1/P2: HOLD, now exercised on genuinely nonzero images for the first time.")
    print("  Evidence pattern (driver's read; the ARCHITECT adjudicates, not this driver): nullity")
    print("  (6 real) > allowance (2 real) matches the pre-reg's own D2 criterion ('the weld EXISTS")
    print("  as a mapped family... THE MULTIPLICITY SELECTOR = the arc's final named incomplete")
    print("  equation... the arc closes').")

    banner("RESULT")
    print("ALL MACHINE CHECKS PASS" if ok_all else "SOME CHECKS FAILED -- see [FAIL] lines above")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
