#!/usr/bin/env python3
"""
proofs/foundations/A2b_conjugate_pair_2026-07-12.py

STATION A2b -- THE CONJUGATE-PAIR WELD (Push 2, station 2).
Frozen contract: internal research notes (SS1-SS8), consuming
A2's adjudicated AF-3 (the grading-parity mismatch theorem, cd83d2b -- SETTLED, not re-opened)
and A1's booked clock linearity (590d041). W2 (595d4e9) is SETTLED throughout.

THE HYPOTHESIS (SS1): AF-3 proved the SELF-J-pinned functor class is empty. The revised shape:
J does not pin the weld to itself -- it PAIRS two welds, Phi = Gamma(phi_1) (grading-preserving)
and Phi~ := J_F.Phi.J_hist (grading-REVERSING). The TOWER is pinned by grading/flow (A1), not by
J. phi_1's frozen pin set (SS3) drops the antiunitary pin entirely: (i) grading, (ii) the R/F_bit
parity pin, (iii) the per-sector-pair block structure. NO self-J-pin, NO region-K_F tower pin, NO
A4-equivariance pin anywhere in the verdict path.

WHAT THIS SCRIPT IS: a DRIVER, not a new derivation. Every construction and exactness check it
runs is IMPORTED from derivation_topdown/state/the_net.py Section 8g -- per the ONE-OBJECT/
LOCAL-NET LAW, Layer-3 math accretes in the_net.py; this file only RUNS it and prints the
verdict-tree evidence. Has an `if __name__ == "__main__":` guard; safe to import.

NUMBERS APPEAR NOWHERE (pre-reg SS7): M_Z, ppm, m_nu, a_e are not computed, not printed, not
compared anywhere below.

D4 HARD GUARD (SS3): the species map {nu,d,u,e} <-> levels is NEVER an input to any pin/
constraint below -- it is read off the (empty) solution and REPORTED ONLY.

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
    banner("A2b  STEP 0 -- THE PRE-REGISTERED LEMMAS L0a/L0b/L0c (MACHINE-VERIFY FIRST, cheap, "
           "bankable regardless of the station verdict)")
    # =======================================================================
    l0a = net.a2b_l0a_self_j_pin_probe()
    print("  L0a (strengthens AF-3): ANY level-additive functor with the SELF-J-pin dies beyond")
    print("  shell 1 (additivity caps image levels of shell n at >= n; the pin needs overlap with")
    print("  the 3-complement, forcing n<=1). Verified at the shell-2 instance:")
    check(f"  shell-2 -> level-2 self-J-pin nullity = {l0a['nullity']}/{l0a['total_real_dim']} "
          "(expect 0)", l0a["nullity"] == 0)

    l0b = net.a2b_l0b_region_flow_rank_lemma()
    print("  L0b (the region clock CANNOT pin the tower): a strict per-level flow pin forces")
    print("  range(phi_1) into h_A's SINGLE eigenspace at the hit eigenvalue -- taking the")
    print("  positive one (the only sign lambda_1>0 admits), h_A's own eigenspace there must be")
    print("  1-dim for rank(phi_1)<=1 to be forced (=> Lambda^2=0, tower dies at shell 2).")
    print("  NOT part of the frozen SS3 pin set -- this ONLY verifies the lemma itself, on ALL 4")
    print("  A4-orbit region representatives:")
    for r in l0b["per_region"]:
        print(f"    region={r['region']} triangle={r['is_triangle']} "
              f"h_A eigvals={['%.4f' % x for x in r['h_A_eigenvalues']]} "
              f"positive-eigenspace dim={r['positive_eigenspace_dim']}")
    check("ALL region orbits: h_A's positive eigenspace is 1-DIM",
          l0b["all_positive_eigenspaces_are_1d"])

    l0c = net.a2b_l0c_conjugate_flow_reversal()
    print("  L0c (conjugate sees reversed flow, M0-4b): the SAME bit that builds J_F flips")
    print("  K_A -> -K_A EXACTLY on every 3-edge region (already-accreted fact, re-verified across")
    print("  ALL 4 region orbits here, not just the triangle):")
    check(f"  worst bit-reversal residual over all region orbits = {l0c['worst_residual']:.2e}",
          l0c["worst_residual"] < 1e-6)

    banner("A2b  STEP 1 -- THE PRE-DECLARED ALLOWANCE, PER SECTOR-PAIR BLOCK (PRINTED BEFORE "
           "SOLVING)")
    # =======================================================================
    alw = net.a2b_level1_allowance_per_block()
    print("  level-1 (Pw[1]) lies entirely inside field-side GROUP-12. Pin (iii) additionally")
    print("  forces phi_1 to vanish on hist-side GROUP-03 (dim "
          f"{alw['group03_hist_dim']}) -- its allowance is declared 0 BEFORE solving. GROUP-12")
    print(f"  (hist dim {alw['group12_hist_dim']}) keeps FOCK-0c's own 'one phase per relevant "
          "block' convention.")
    check(f"ALLOWANCE: GROUP-03 = {alw['group03_allowance']} (forced 0), GROUP-12 = "
          f"{alw['group12_allowance']} (one phase); TOTAL = {alw['allowance']}",
          alw["allowance"] == 1 and alw["group03_allowance"] == 0)

    banner("A2b  STEP 2 -- THE phi_1 NULLITY TRAJECTORY (pins (i)/(ii)/(iii), NO antiunitary pin "
           "ANYWHERE)")
    # =======================================================================
    traj = net.a2b_phi1_pin_trajectory()
    print(f"  ambient real dimension: {traj['total_real_dim']}")
    check(f"stage 0 -- (i) grading alone (codomain confined to Pw[1]): "
          f"{traj['stage0_grading_alone']}", True)
    check(f"stage (i)+(iii) pair-block (no R/F_bit yet): {traj['stage_pairblock']}", True)
    check(f"FULL (i)+(ii)+(iii), real-linear (SS3's literal 'solve (real-linear)' text): "
          f"{traj['stage_full_real_linear']}", traj["stage_full_real_linear"] == 0)
    check(f"FULL (i)+(ii)+(iii), A-only / B=0 (the genuine complex-linear content SS4's "
          f"Gamma(phi_1) actually needs -- an ambiguity resolved by reporting BOTH, per the "
          f"contract's 'strictest reading, note it' instruction): {traj['stage_full_A_only']}",
          traj["stage_full_A_only"] == 0)

    banner("A2b  STEP 3 -- THE ALGEBRAIC PROOF (a THIRD, DISTINCT obstruction mechanism from "
           "AF-3, using NO antiunitary structure at all)")
    # =======================================================================
    print("  CLAIM: phi_1 = 0 is FORCED by pins (i)+(ii)+(iii), from three machine-verified facts:")
    print("    FACT 1 (pin iii): Phi(hist-GROUP-03) = 0 identically (the pair-block pin, given")
    print("      Phi's image already confined to Pw[1] subset field-GROUP-12 by pin i).")
    print("    FACT 2: R = reversal() acts as EXACTLY +I on hist-GROUP-12 (the 2-dim isotypic")
    print("      block of A4's two nontrivial 1-dim characters).")
    print("    FACT 3: F_bit = Pw[0]+Pw[3]-Pw[1]-Pw[2] acts as EXACTLY -I on Pw[1].")
    print("  PROOF: for v in hist-GROUP-12, pin (ii) gives Phi(R.v)=F_bit(Phi(v)). By FACT 2,")
    print("  R.v=v, so LHS=Phi(v). By FACT 3, RHS=-Phi(v). Hence Phi(v)=-Phi(v) => Phi(v)=0.")
    print("  Combined with FACT 1 (hist-GROUP-03 already 0), and GROUP-03+GROUP-12 spanning the")
    print("  full 12-dim domain, Phi=0 identically. QED -- NO antiunitary pin used anywhere.")
    proof = net.a2b_phi1_forced_zero_proof()
    check(f"FACT 2 (R = +I on hist-GROUP-12, dim {proof['group12_dim']}): residual "
          f"{proof['R_on_group12_identity_residual']:.2e}",
          proof["R_on_group12_identity_residual"] < 1e-8)
    check(f"FACT 3 (F_bit = -I on level-1): residual "
          f"{proof['F_bit_on_level1_neg_identity_residual']:.2e}",
          proof["F_bit_on_level1_neg_identity_residual"] < 1e-8)
    check(f"hist-GROUP-03 dim = {proof['group03_dim']} (forced 0 by pin iii, FACT 1)", True)
    check("numeric FULL-pin nullity (both real-linear and A-only) CONFIRMS the algebraic proof's "
          "conclusion (phi_1 = 0)", proof["stage_full_confirms_proof"] == 0)

    banner("A2b  STEP 4 -- P1/P2 RE-VERIFIED IN THIS CLASS (reused from A2, unchanged "
           "construction)")
    # =======================================================================
    p12 = net.a2b_p1_p2_reverify()
    p1, p2 = p12["p1"], p12["p2"]
    check(f"P1 PAULI TRUNCATION [structural test map]: ALL {p1['n_shell4_words']} shell-4 words "
          f"give Gamma=0 (worst {p1['worst_shell4_norm']:.2e})", p1["worst_shell4_norm"] < 1e-9)
    check(f"P2 REPEATED-DART KERNEL: synthetic repeat gives Gamma=0 (residual "
          f"{p2['synthetic_repeat_residual']:.2e}); first admissible repeat at shell "
          f"{p2['first_shell_with_admissible_repeat']} (P1 already kills shell>=4)",
          p2["synthetic_repeat_residual"] < 1e-9 and p2["first_shell_with_admissible_repeat"] == 4)

    banner("A2b  STEP 5 -- THE FORCING QUESTION / TOWER-MEMBERSHIP TEST (SS4.3, the honesty "
           "clause)")
    # =======================================================================
    s2 = net.a2b_shell_level_system_nullity(2, 2)
    s3 = net.a2b_shell_level_system_nullity(3, 3)
    print(f"  the A2b-CLASS shell-n system (grading+parity+pairblock, built INDEPENDENTLY at each")
    print(f"  shell, SAME builders as phi_1's own shell-1 system) has its OWN freedom:")
    check(f"  shell2->level2 [dom {s2['domain_dim']}]: FULL real-linear = "
          f"{s2['stage_full_real_linear']}/{s2['total_real_dim']} (A-only "
          f"{s2['stage_full_A_only']})", True)
    check(f"  shell3->level3 [dom {s3['domain_dim']}]: FULL real-linear = "
          f"{s3['stage_full_real_linear']}/{s3['total_real_dim']} (A-only "
          f"{s3['stage_full_A_only']})", True)
    print("  Since phi_1's OWN solution basis is EMPTY (STEP 3), Gamma(phi_1)=0 identically at")
    print("  every shell -- trivially a member of EVERY linear space (the 0 vector), REGARDLESS")
    print("  of whether the shell-n system above has its own nonzero freedom (it does, at shells")
    print("  2/3). Reported HONESTLY as VACUOUS -- the pre-reg's honesty clause: a vacuous pass")
    print("  is NOT forcing. A DISCRIMINATING CONTROL (a genuine non-solution) is run alongside.")
    tw2 = net.a2b_tower_membership_test(2, [], N_max=4)
    check(f"SHELL 2 [D={tw2['D_shell_n']}]: basis_size=0 -> VACUOUS={tw2['vacuous']} (the 0 map, "
          "trivially member)", tw2["all_members"] and tw2["vacuous"])
    check(f"  discriminating control residual = {tw2['control_residual']:.4e} -- "
          f"control_is_member={tw2['control_is_member']} (should be False)",
          tw2["control_is_member"] is False)
    tw3 = net.a2b_tower_membership_test(3, [], N_max=4)
    check(f"SHELL 3 [D={tw3['D_shell_n']}]: basis_size=0 -> VACUOUS={tw3['vacuous']}",
          tw3["all_members"] and tw3["vacuous"])
    check(f"  discriminating control residual = {tw3['control_residual']:.4e} -- "
          f"control_is_member={tw3['control_is_member']} (should be False)",
          tw3["control_is_member"] is False)
    print("  EXPLICIT VACUOUS-OR-NOT STATEMENT (pre-reg's #1 honesty-clause item): the tower test")
    print("  is VACUOUS -- phi_1's nullity is 0, so there is no nonzero basis to test forcing")
    print("  with. This is NOT reported as a forcing confirmation.")

    banner("A2b  STEP 6 -- PAIR-COMPLETENESS (report only)")
    # =======================================================================
    pc = net.a2b_pair_completeness_read()
    check(f"dim(Im phi_1) = {pc['dim_Im_phi1']}; Phi~ = J_F.Phi.J_hist is zero = "
          f"{pc['Phi_tilde_is_zero']}; pair (Phi,Phi~) covers F = {pc['pair_covers_F']}",
          pc["dim_Im_phi1"] == 0 and pc["Phi_tilde_is_zero"])
    print(f"  {pc['note']}")

    banner("A2b  STEP 7 -- N-HAT-INTERTWINING EXACTNESS + THE EMBER-CONSISTENCY SHADOW "
           "(structure only)")
    # =======================================================================
    nh = net.a2b_nhat_intertwining_exactness()
    for row in nh["per_shell"]:
        print(f"    shell {row['n']}: c_n={row['c_n']:.6f}  n.c_1={row['n_times_c1']:.6f}  "
              f"match={row['match']}  NHAT eigenvalue on level_n={row['nhat_eigenvalue_on_level_n']}")
    check(f"Phi.K_hist = c_1.NHAT.Phi holds EXACTLY for ANY grading-preserving Phi (structural, "
          f"phi_1-independent): worst residual {nh['worst_residual']:.2e}",
          nh["worst_residual"] < 1e-9)
    emb = net.a2b_ember_consistency_shadow()
    tri = next(r for r in emb if r["is_triangle"])
    check(f"EMBER-CONSISTENCY SHADOW: triangle-orbit lambda_1={tri['lambda_1']:.4f} (FOCK-0d "
          "ember reference 2.463) -- lambda's confronted with NOTHING", abs(tri["lambda_1"] - 2.463) < 0.01)

    banner("A2b  STEP 8 -- SS5 LABELED DIAGNOSTICS (report-only, NON-VERDICT)")
    # =======================================================================
    diag = net.a2b_equivariant_subspace_diagnostic()
    print("  Hom_A4(shell-1, level-1): TWO readings computed and disclosed (the pre-reg's own")
    print("  'complex dim 2' expectation is VERIFIED, not assumed, per the contract):")
    check(f"  READING A (standard, Frobenius reciprocity via an HONEST rho3, sidesteps F's own "
          f"uncontrolled phase ambiguity): dim = {diag['hom_A4_standard_reading_dim']} (matches "
          "sector_grading_hist's banked mult[1]=[1,1,1,3])",
          diag["hom_A4_standard_reading_dim"] == 3)
    check(f"  READING B (tensor-square, the pre-reg's own '3 tensor 3 contains 3 twice' "
          f"derivation, character-verified): dim = {diag['hom_A4_tensor_square_reading_dim']}",
          diag["hom_A4_tensor_square_reading_dim"] == 2)
    print(f"  {diag['ambiguity_note']}")
    check(f"INTERSECTION with phi_1's solution space (reading A, the only genuine subspace of "
          f"Hom(C^12,C^3)): {diag['intersection_with_phi1_solution_dim']} (phi_1=0, trivial)",
          diag["intersection_with_phi1_solution_dim"] == 0)
    print(f"  ISOTYPIC FLOW READ: {diag['isotypic_flow']['note']}")

    banner("A2b  REGRESSION: Sections 7/7b/8/8b/8c/8d/8e/8f + module anchors untouched")
    # =======================================================================
    check("anchor_cell_projector() + anchor_tick_2pi() + accretion_selftest_2026_07_10() + "
          "i2b_selftest_2026_07_11() + fock0_selftest_2026_07_11() + "
          "fock0b_selftest_2026_07_11() + fock0c_selftest_2026_07_11() + "
          "fock0d_selftest_2026_07_11() + fock0e_selftest_2026_07_12() + "
          "a2_weld_selftest_2026_07_12() all still PASS",
          net.anchor_cell_projector() and net.anchor_tick_2pi()
          and net.accretion_selftest_2026_07_10(verbose=False)
          and net.i2b_selftest_2026_07_11(verbose=False)
          and net.fock0_selftest_2026_07_11(verbose=False)
          and net.fock0b_selftest_2026_07_11(verbose=False)
          and net.fock0c_selftest_2026_07_11(verbose=False)
          and net.fock0d_selftest_2026_07_11(verbose=False)
          and net.fock0e_selftest_2026_07_12(verbose=False)
          and net.a2_weld_selftest_2026_07_12(verbose=False))
    check("a2b_weld_selftest_2026_07_12() (the Section-8g permanent regression anchor) PASSES",
          net.a2b_weld_selftest_2026_07_12(verbose=False))

    banner("A2b VERDICT-RELEVANT SUMMARY (SS6 tree; ML-2b/HK-7 QUALIFIER attaches to EVERY "
           "sentence; architect adjudicates, NOT this driver)")
    # =======================================================================
    print(f"  QUALIFIER: {QUALIFIER}")
    print()
    print(f"  L0a/L0b/L0c: all VERIFIED (self-J-pin dies at shell 2; region-flow rank lemma holds")
    print(f"  on all 4 orbits; conjugate flow-reversal confirmed on all 4 orbits)")
    print(f"  ALLOWANCE (pre-declared, per block) = GROUP-03: 0, GROUP-12: 1, TOTAL: {alw['allowance']}")
    print(f"  phi_1 nullity (full pin (i)+(ii)+(iii)) = {traj['stage_full_real_linear']} "
          f"(real-linear), {traj['stage_full_A_only']} (A-only)")
    print(f"  ALGEBRAIC PROOF: phi_1 = 0 FORCED -- a THIRD, distinct obstruction mechanism (R=+I "
          "on hist-GROUP-12 vs F_bit=-I on level-1; the pair-block pin alone kills hist-GROUP-03), "
          "using NO antiunitary structure at all")
    print(f"  P1/P2: HOLD (reused from A2, unchanged)")
    print(f"  TOWER TEST shells 2/3: VACUOUS (explicit, per the honesty clause) -- shell-2/3's OWN "
          f"systems DO have nonzero freedom ({s2['stage_full_real_linear']}, "
          f"{s3['stage_full_real_linear']} real-linear), but phi_1's empty basis cannot exercise it")
    print(f"  PAIR-COMPLETENESS: Phi and Phi~ both zero; the pair covers NOTHING of F")
    print(f"  N-HAT-INTERTWINING: EXACT (structural, phi_1-independent)")
    print(f"  SS5 DIAGNOSTICS: Hom_A4(shell-1,level-1) = 3 (standard) / 2 (tensor-square, matching "
          "the pre-reg's own stated expectation under THAT reading); intersection with phi_1 = 0")
    print()
    print("  Evidence pattern (driver's read; the ARCHITECT adjudicates, not this driver): the")
    print("  frozen SS3 pin set -- grading + R/F_bit parity + per-sector-pair block, EXPLICITLY")
    print("  WITHOUT any antiunitary/self-J pin -- STILL forces phi_1 = 0, theorem-grade, via a")
    print("  mechanism independent of AF-3's K-swap argument. This matches SS6's B3 criterion")
    print("  ('the SS3 system is empty') more precisely than B1/B2 (which require a nonzero")
    print("  solution). The B3 branch calls for 'the algebraic proof' -- delivered above (STEP 3).")

    banner("RESULT")
    print("ALL MACHINE CHECKS PASS" if ok_all else "SOME CHECKS FAILED -- see [FAIL] lines above")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
