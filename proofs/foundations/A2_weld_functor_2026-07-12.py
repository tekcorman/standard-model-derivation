#!/usr/bin/env python3
"""
proofs/foundations/A2_weld_functor_2026-07-12.py

STATION A2 -- THE WELD AS SECOND-QUANTIZATION FUNCTOR (Push 2, the pivotal station).
Frozen contract: internal research notes (SS1-SS7), extending
internal research notes (all amendments FOCK-0b/0c/0d/0e stand
unchanged; FOCK-0d's W2 verdict -- the clock-incommensurability obstruction theorem, 595d4e9 -- is
SETTLED and NOT re-opened). Consumes A1's booked theorem (FOCK-0e, 590d041: c_n = n.c_1 exact;
lambda_n = n.lambda_1 structure).

THE HYPOTHESIS (SS1): the weld Phi = Gamma(phi_1), phi_1 : shell-1 (the 12 darts) -> level-1 of F
(the 3 complex modes, Pw[1]), Gamma(phi_1)(w) = phi_1(d_1)^phi_1(d_2)^...^phi_1(d_n) -- the
fermionic second-quantization functor of a one-particle map. MS-1a's UNIQUE nontrivial fermion-
parity Z2 makes fermionic Gamma the only statistics available.

WHAT THIS SCRIPT IS: a DRIVER, not a new derivation. Every construction and exactness check it
runs is IMPORTED from derivation_topdown/state/the_net.py Section 8f -- per the ONE-OBJECT/
LOCAL-NET LAW, Layer-3 math accretes in the_net.py; this file only RUNS it and prints the
verdict-tree evidence. Has an `if __name__ == "__main__":` guard; safe to import.

NUMBERS APPEAR NOWHERE (pre-reg SS6): M_Z, ppm, m_nu, a_e are not computed, not printed, not
compared anywhere below.

D4 HARD GUARD (SS3): the species map {nu,d,u,e} <-> levels is NEVER an input to any pin/
constraint below -- it is read off the (empty) solution and REPORTED ONLY, at the very end.

THE ML-2b/HK-7 CONDITIONALITY QUALIFIER (verbatim, aqft_net.py:280-292 -- attaches to EVERY
verdict-adjacent sentence below):
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
    banner("A2  STEP 1 -- THE PRE-DECLARED ALLOWANCE (PRINTED BEFORE SOLVING, per FOCK-0c discipline)")
    # =======================================================================
    alw = net.a2_level1_allowance()
    print("  level-1 (Pw[1], dim 3) is PAIRED with level-2 (Pw[2]) under the SAME field-side bit-")
    print("  orbit K already established (orbit_12={1,2}) -- level-1 touches exactly ONE sector-")
    print("  pair block (GROUP-12), not two (contrast FOCK-0c's shell-1 -> full-F target).")
    check(f"ALLOWANCE = {alw['relevant_pair_blocks']} relevant block x "
          f"{alw['real_dims_per_block']} real dim (one phase) = {alw['allowance']}",
          alw["allowance"] == 1)
    print(f"    NAMED CAVEAT (stated up front): FOCK-0c's own shell-1 full-pin result already "
          f"found GROUP-12's own diagonal sub-block nullity = "
          f"{alw['precedent_group12_nullity_at_shell1']} (not the nominal 1) -- a 0 result below "
          "is consistent with that precedent, not a surprise.")

    banner("A2  STEP 2a -- THE NULLITY TRAJECTORY (pins (i)/(ii)/(iii), stacked incrementally)")
    # =======================================================================
    traj = net.a2_phi1_pin_trajectory()
    print(f"  ambient real dimension: {traj['total_real_dim']}")
    check(f"stage 0 -- (i) grading alone (codomain confined to Pw[1]): "
          f"{traj['stage0_grading_alone']}", True)
    check(f"stage 1 -- (i)+(ii) + antiunitary Tomita pin: "
          f"{traj['stage1_grading_plus_antiunitary']}", True)
    check(f"stage 2 -- FULL PIN (i)+(ii)+(iii) + R/F_bit: {traj['stage2_full_pin']}",
          traj["stage2_full_pin"] == 0)
    check(f"stage 3 (diagnostic, NOT the frozen pin) -- (i)+(iii), no antiunitary: "
          f"{traj['stage3_grading_plus_rfbit_no_antiunitary']}", True)
    check(f"the grading refinement (i) can only SHRINK FOCK-0c's shell-1 -> full-F nullity (16): "
          f"full pin {traj['stage2_full_pin']} <= 16", traj["shrinks_fock0c_16"])

    banner("A2  STEP 2b -- THE ALGEBRAIC PROOF (AF-3-grade; not SVD alone)")
    # =======================================================================
    print("  CLAIM: phi_1 = 0 is FORCED by pins (i)+(ii) alone, from three already-accreted facts:")
    print("    FACT 1: K = field_algebra_conjugation's M swaps range(Pw[1]) <-> range(Pw[2]) EXACTLY.")
    print("    FACT 2: range(Pw[1]) cap range(Pw[2]) = {0} (orthogonal NHAT eigenspaces).")
    print("    FACT 3: reversal() is an involution, hence v -> R@conj(v) is a BIJECTION of C^12.")
    print("  PROOF: if Phi's image subset range(Pw[1]) (pin i) and Phi(R.conj(v))=K.conj(Phi(v))")
    print("  (pin ii) for all v, then LHS in range(Pw[1]) while RHS in range(Pw[2]) whenever")
    print("  Phi(v)!=0 -- forcing Phi(R.conj(v))=0 for every v; since v->R.conj(v) is onto C^12,")
    print("  Phi = 0 identically. QED.")
    proof = net.a2_phi1_forced_zero_proof()
    check(f"FACT 1 (K swaps Pw[1]<->Pw[2]): residual {proof['K_swaps_Pw1_to_Pw2_residual']:.2e}",
          proof["K_swaps_Pw1_to_Pw2_residual"] < 1e-8)
    check(f"FACT 2 (Pw[1] cap Pw[2] = 0): residual {proof['Pw1_cap_Pw2_residual']:.2e}",
          proof["Pw1_cap_Pw2_residual"] < 1e-8)
    check(f"FACT 3 (R involution): residual {proof['R_involution_residual']:.2e}",
          proof["R_involution_residual"] < 1e-9)
    check("numeric stage-1 nullity CONFIRMS the algebraic proof's conclusion (phi_1 = 0)",
          proof["stage1_nullity_confirms_proof"] == 0)

    banner("A2  AMBIGUITY NOTE -- alternate-reading diagnostic (NOT the frozen pin)")
    # =======================================================================
    print("  Per the contract's ambiguity-handling rule ('take the strictest reading, note the")
    print("  ambiguity, never choose silently'): is the forced-zero result an artifact of the")
    print("  single-level codomain choice? Test with codomain = GROUP-12 (dim 6, level-1 UNION")
    print("  level-2, the FULL K-orbit) instead, domain UNCHANGED (still the full 12-dim shell-1).")
    alt = net.a2_alternate_reading_diagnostic()
    check(f"alt grading alone: {alt['stage_grading_alone']}/{alt['total_real_dim']}", True)
    check(f"alt grading + antiunitary: {alt['stage_plus_antiunitary']}/{alt['total_real_dim']} "
          "(NONZERO -- confirms the forced-zero result is SPECIFIC to a half-K-orbit codomain, "
          "not a generic collapse of this pin machinery)", alt["stage_plus_antiunitary"] > 0)
    check(f"alt full pin (+R/F_bit): {alt['stage_full_pin']}/{alt['total_real_dim']}", True)

    banner("A2  STEP 2c -- Gamma(phi_1) CONSTRUCTION: P1 (Pauli truncation) + P2 (repeated-dart kernel)")
    # =======================================================================
    print("  Gamma(phi_1) is realized fermionically via the 3 canonical creation operators")
    print("  Adag[0..2] on F (REUSED from _sector_projectors/field_algebra_conjugation, verified")
    print("  Adag[m]|vac> spans range(Pw[1]) exactly). P1/P2 are checked with a STRUCTURAL TEST")
    print("  MAP (a generic complex 12x3 matrix) since the frozen solution IS zero and would make")
    print("  a vacuous check -- this demonstrates the FUNCTOR FORM's guarantees hold regardless of")
    print("  the pin outcome (they are properties of the exterior-algebra/CAR realization itself).")
    p1 = net.a2_pauli_truncation_check(N_max=6)
    check(f"P1 PAULI TRUNCATION: ALL {p1['n_shell4_words']} shell-4 admissible words give "
          f"Gamma(phi_test)=0 (worst norm {p1['worst_shell4_norm']:.2e}) -- Lambda^4(3-dim)=0, "
          "the Pauli-truncation prediction, holds by construction", p1["worst_shell4_norm"] < 1e-9)
    check(f"  (shell<=3 sample norms nonzero, confirming the test map is generic, not "
          f"degenerately zero): {['%.3f' % x for x in p1['n_shell_le3_nonzero_sample']]}",
          all(x > 1e-6 for x in p1["n_shell_le3_nonzero_sample"]))

    p2 = net.a2_repeated_dart_kernel_check(N_max=10)
    check(f"P2 REPEATED-DART KERNEL: synthetic word (0,1,0) gives Gamma(phi_test)=0 "
          f"(residual {p2['synthetic_repeat_residual']:.2e})",
          p2["synthetic_repeat_residual"] < 1e-9)
    print(f"    HONEST DOMAIN FINDING: scanning admissible words to N_max={p2['n_max_scanned']}, "
          f"NO word of length <=3 contains a repeated dart at all -- counts by shell: "
          f"{p2['shell_le3_repeat_counts']}. The FIRST admissible repeated-dart word appears at "
          f"shell {p2['first_shell_with_admissible_repeat']} -- exactly where P1 ALREADY forces "
          "Gamma=0 for dimension reasons. P1 and P2 do not bite independently on this graph: P2's")
    print("    predicted kernel is EMPTY within the functor's live domain (shells 1-3); it holds")
    print("    VACUOUSLY there. Combinatorial-shadow cross-cite (mechanism, not re-run): T0-N-3's")
    print("    864/864 orientation-compatibility lemma + its 'coincidence collides 10/10' finding")
    print("    (internal research notes:30-50,105,114-116) -- a")
    print("    DIFFERENT construction (ground-state cycle pairs/triples under CAR) but the SAME")
    print("    mechanism: reusing a mode twice under a fermionic/CAR realization forces the state")
    print("    to zero.")
    check("no admissible repeated-dart word exists at shell<=3 (first appears at shell 4)",
          p2["first_shell_with_admissible_repeat"] == 4)

    banner("A2  STEP 3 -- THE FORCING QUESTION / TOWER-MEMBERSHIP TEST (D2 frozen)")
    # =======================================================================
    print("  Since phi_1 = 0 (STEP 2b), the phi_1 solution basis is EMPTY -- Gamma(phi_1) = 0")
    print("  identically at every shell, which trivially lies in EVERY linear space (the 0 vector).")
    print("  Reported HONESTLY as VACUOUS, not claimed as a nontrivial confirmation. A")
    print("  DISCRIMINATING CONTROL (the SAME structural test map from STEP 2c, a genuine")
    print("  non-solution) is run alongside to confirm the test machinery is not degenerately")
    print("  accepting everything.")
    tw2 = net.a2_tower_membership_test(2, [], N_max=4)
    check(f"SHELL 2 [D={tw2['D_shell_n']}]: basis_size=0 -> VACUOUSLY a member (the 0 map)",
          tw2["all_members"])
    check(f"  discriminating control (structural test map) residual = "
          f"{tw2['control_residual']:.4e} -- control_is_member={tw2['control_is_member']} "
          "(should be False: the control is NOT a real solution)",
          tw2["control_is_member"] is False)
    tw3 = net.a2_tower_membership_test(3, [], N_max=4)
    check(f"SHELL 3 [D={tw3['D_shell_n']}]: basis_size=0 -> VACUOUSLY a member (the 0 map)",
          tw3["all_members"])
    check(f"  discriminating control residual = {tw3['control_residual']:.4e} -- "
          f"control_is_member={tw3['control_is_member']} (should be False)",
          tw3["control_is_member"] is False)

    banner("A2  STEP 4 -- CLOCK READ (structure only; NO global-lambda pin, SS2 trap not attempted)")
    # =======================================================================
    clk = net.a2_clock_read(N_max=8)
    print(f"  phi_1 nonzero-eigenspace intersection: {clk['phi1_nonzero_eigenspace_intersection']}")
    check("no global-lambda pin attempted anywhere in this station (SS2's pre-registered trap is "
          "an EXPECTED theorem-let, never re-attempted or adjudicated as this station's verdict)",
          clk["global_lambda_pin_attempted"] is False)
    lam = clk["lambda_structure_inherited"]
    tri = next(r for r in lam if r["is_triangle"])
    print(f"  INHERITED (FOCK-0e, not re-derived): triangle-orbit lambda_1={tri['lambda_1']:.4f} "
          f"(lambda_n = n.lambda_1 structure) -- unconditionally true of K_hist/K_F's own spectra, "
          "independent of THIS station's forced-zero outcome.")

    banner("A2  STEP 5 -- SPECIES READ (output only; D4 hard guard)")
    # =======================================================================
    sp = net.a2_species_read()
    check(f"surviving isotypic components: {sp['surviving_isotypic_components']} (empty, phi_1=0)",
          sp["surviving_isotypic_components"] == [])
    print(f"  species correspondence: {sp['species_correspondence']}")
    print(f"  {sp['note']}")

    banner("A2  REGRESSION: Sections 7/7b/8/8b/8c/8d/8e + module anchors untouched")
    # =======================================================================
    check("anchor_cell_projector() + anchor_tick_2pi() + accretion_selftest_2026_07_10() + "
          "i2b_selftest_2026_07_11() + fock0_selftest_2026_07_11() + "
          "fock0b_selftest_2026_07_11() + fock0c_selftest_2026_07_11() + "
          "fock0d_selftest_2026_07_11() + fock0e_selftest_2026_07_12() all still PASS",
          net.anchor_cell_projector() and net.anchor_tick_2pi()
          and net.accretion_selftest_2026_07_10(verbose=False)
          and net.i2b_selftest_2026_07_11(verbose=False)
          and net.fock0_selftest_2026_07_11(verbose=False)
          and net.fock0b_selftest_2026_07_11(verbose=False)
          and net.fock0c_selftest_2026_07_11(verbose=False)
          and net.fock0d_selftest_2026_07_11(verbose=False)
          and net.fock0e_selftest_2026_07_12(verbose=False))
    check("a2_weld_selftest_2026_07_12() (the Section-8f permanent regression anchor) PASSES",
          net.a2_weld_selftest_2026_07_12(verbose=False))

    banner("A2 VERDICT-RELEVANT SUMMARY (SS5 tree; ML-2b/HK-7 QUALIFIER attaches to EVERY "
           "sentence; architect adjudicates, NOT this driver)")
    # =======================================================================
    print(f"  QUALIFIER: {QUALIFIER}")
    print()
    print(f"  ALLOWANCE (pre-declared) = {alw['allowance']}")
    print(f"  phi_1 nullity (full pin (i)+(ii)+(iii)) = {traj['stage2_full_pin']}")
    print(f"  ALGEBRAIC PROOF: phi_1 = 0 FORCED (AF-3-grade, K-swap/orthogonality/bijectivity, "
          "not SVD alone)")
    print(f"  P1 (Pauli truncation): HOLDS by construction (structural test map, shell-4 exact 0)")
    print(f"  P2 (repeated-dart kernel): HOLDS, but VACUOUSLY within the live domain (shells 1-3 "
          "have no admissible repeated-dart word on this graph)")
    print(f"  TOWER TEST shells 2/3: VACUOUSLY satisfied (Gamma(phi_1)=0); discriminating control "
          "correctly REJECTED at both shells")
    print(f"  SPECIES READ: empty (no nonzero image; D4 respected throughout)")
    print(f"  CLOCK READ: empty eigenspace intersection; lambda_n=n.lambda_1 inherited as structure")
    print()
    print("  Evidence pattern (driver's read; the ARCHITECT adjudicates, not this driver): the")
    print("  literal SS1 pin set (codomain = level-1 ALONE) is provably incompatible with the")
    print("  antiunitary Tomita pin (ii) -- phi_1 = 0 is FORCED, theorem-grade, independent of any")
    print("  numeric truncation. This matches SS5's AF-3 criterion ('the pin set forces phi_1 = 0')")
    print("  more precisely than AF-1/AF-2 (which require a nonzero solution). The ambiguity-note")
    print("  diagnostic shows this is NOT a generic collapse of the pin machinery: relaxing the")
    print("  codomain to the FULL K-orbit (GROUP-12, both levels) restores nonzero freedom.")

    banner("RESULT")
    print("ALL MACHINE CHECKS PASS" if ok_all else "SOME CHECKS FAILED -- see [FAIL] lines above")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
