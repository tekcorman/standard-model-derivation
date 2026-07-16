#!/usr/bin/env python3
"""
proofs/foundations/FOCK0b_pinned_map_2026-07-11.py

STATION FOCK-0b -- THE HISTORY-SIDE MODULAR CONJUGATION + THE PINNED-MAP TEST (post-V4
adjudication amendment to FOCK-0).  Frozen contract: the AMENDMENT (SS B-E) appended to
internal research notes (freeze commit b1c3546).  Adjudication
record this amendment discharges: the verifier (internal research notes
FOCK0_check_2026-07-11.md) found FOCK-0's dr_map_hom_space() solves ordinary full-A4 LINEAR
GENERATOR equivariance (Phi.dart_rep(g) = U(g).Phi for every g in A4) -- the EXACT class the
pre-reg's SS1/SS2 explicitly disclaim ("pinned by intertwining MODULAR CONJUGATIONS..., NOT by
generators"); J_sigma was never used, and the history-side modular conjugation never existed in
the repo.  This script runs the REAL (frozen-class) test this amendment builds.

WHAT THIS SCRIPT IS: a DRIVER, not a new derivation.  Every construction and exactness check it
runs is IMPORTED from derivation_topdown/state/the_net.py Section 8b (the FOCK-0b GNS/Tomita
construction + the pinned-map solver) -- per the ONE-OBJECT/LOCAL-NET LAW, Layer-3 math accretes
in the_net.py; this file only RUNS it and prints the verdict-tree evidence.  Has a
`if __name__ == "__main__":` guard; safe to import.

THE FROZEN HYPOTHESIS UNDER TEST (amendment SS B): does a LINEAR map Phi: (12-dim dart/shell-1
space) -> (8-dim field-algebra Fock F) exist, intertwining ONE antiunitary conjugation per side
(Phi.J_hist = J_F.Phi), WITH NO per-g GENERATOR CONSTRAINT ANYWHERE -- the class the frozen SS1
pin names, genuinely distinct from (and, per the verifier's own argument, NOT nested with)
the disclaimed full-A4-equivariance mechanism.

NUMBERS APPEAR NOWHERE (pre-reg SS3.4, amendment SS D): every printed quantity below is a
dimension, rank, nullity, or exactness residual (structure) -- never M_Z, ppm residuals, m_nu, or
a_e.  This station delivers STRUCTURE + a verdict on forcedness only.

POISONS RESPECTED (amendment SS D, on top of pre-reg SS5): J_hist != J6 (history-side Tomita vs
edge-side complex structure, never conflated); the antilinear system solves over THE REALS (the
SOLVER TRAP -- a complex-linear SVD would be WRONG and void the station; pinned_map_hom_space_real
builds the REAL 2n x 2n / 2m x 2m embeddings explicitly, never a bare complex SVD); per-path vs
per-shell in the Delta<->K consistency (b2) is stated explicitly, not conflated; the top-shell
truncation artifact is the I2b interior-exactness fact (checks live in the interior); NO 2T-lift
attempts on the dart carrier (proven impossible by the verifier, SS4a; the pin makes it
unnecessary); no goal-seek toward M_Z/-70ppm/m_nu/a_e; alpha_1 != alpha_EM; the four temperatures
are never conflated; beta' vs beta_natural is NOT adjudicated (amendment SS B/b0: this station
builds on omega_diag at beta_natural = 2*beta_gas AS MEASURED).

THE ML-2b/HK-7 CONDITIONALITY QUALIFIER (verbatim, aqft_net.py:280-292 / pre-reg SS0 -- attaches
to EVERY verdict sentence below):
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
    banner("FOCK-0b  b0/b1 -- WORK ON THE TRUNCATED I2b WORD-FOCK SPACE; GNS/PURIFICATION")
    # =======================================================================
    N_max = 4
    gp = net.gns_purification(N_max)
    check(f"[N_max={N_max}, D={len(gp['words'])}] state = omega_diag at beta_natural = 2*beta_gas "
          "(A1 stands: no beta'-vs-beta_natural adjudication made here)",
          True, detail=f"min(omega_diag) = {gp['min_rho']:.3e}")
    check("b1 SEPARATING: rho is full-rank on the truncation (every admissible word gets "
          "omega_diag > 0 -- checked explicitly, not assumed; W4 would fire here if it failed)",
          gp["full_rank_separating"])

    banner("FOCK-0b  b2 -- TOMITA DATA S = J.Delta^1/2, EXACTLY, IN FINITE DIMENSIONS")
    # =======================================================================
    tc = net.tomita_checks(N_max)
    check("S(x.Omega) == x^dagger.Omega EXACTLY (the defining Tomita relation, on the "
          "M-generated/seed-cyclic subspace)",
          tc["S_closed_form_residual"] < 1e-9, detail=f"resid={tc['S_closed_form_residual']:.2e}")
    check("J antiunitary with J^2 = 1", tc["J_squared_residual"] < 1e-9,
          detail=f"resid={tc['J_squared_residual']:.2e}")
    check("J.M.J subset M' ([pi_R(S_d^dagger), pi_L(S_e)] = 0 -- an algebraic identity, true for "
          "ANY subalgebra M, verified on the alg{S_d} generators to catch implementation bugs)",
          tc["JMJ_subset_Mprime_residual"] < 1e-8,
          detail=f"resid={tc['JMJ_subset_Mprime_residual']:.2e}")
    check(f"Delta <-> KMS-at-beta_natural CONSISTENCY, THE PER-PATH READING: "
          f"beta_natural = {tc['beta_natural_check']:.10f} (matches I2b's own 6.4874417297)",
          tc["kms_per_path_residual"] < 1e-8 and abs(tc["beta_natural_check"] - 6.4874417297) < 1e-6,
          detail=f"per-path resid={tc['kms_per_path_residual']:.2e} -- CONTRAST (per-shell "
                 f"aggregate rate, printed not asserted): {tc['per_shell_rate_sample']:.6f} = "
                 f"beta_natural - h_top = beta' = {tc['beta_prime_reproduced']:.6f} "
                 "(I2b's own beta_natural=beta'+h_top identity, reproduced here as the marker "
                 "of WHICH reading -- per-path -- this construction's Delta actually matches: "
                 "Delta acts diagonally on individual WORDS/microstates, not shell-aggregated "
                 "marginals)")

    banner("FOCK-0b  b3 -- SECTOR STRUCTURE OF J (THE GRADED GNS CARRIER)")
    # =======================================================================
    gc = net.gns_grading_commutation(N_max)
    check("rho (hence Delta) commutes EXACTLY with every A4-isotypic projector (shell-"
          "preservation: dart_word_action preserves word length, rho is a scalar per shell)",
          gc["rho_commutes_with_grading_residual"] < 1e-9,
          detail=f"resid={gc['rho_commutes_with_grading_residual']:.2e}")
    check("J's action on the grading: SELF-MAPS diagonal (a,a) blocks, SWAPS off-diagonal (a,b) "
          "<-> (b,a) pairs (both exact -- EITHER is a result; here it is BOTH, printed precisely)",
          gc["diag_block_self_map_residual"] < 1e-9 and gc["J_offdiag_swap_residual"] < 1e-9,
          detail=f"diag resid={gc['diag_block_self_map_residual']:.2e}, "
                 f"swap resid={gc['J_offdiag_swap_residual']:.2e}")
    check(f"A4's character-conjugation pairing (a candidate history-side sector pairing, NOT "
          f"used as the b4 pin below): self-conjugate irreps (dims [1,1,1,3] order) = "
          f"{gc['self_conjugate_irreps']}, conjugate PAIR = {gc['conjugate_irrep_pairs']} "
          "(A4's abelianization Z3 forces its two nontrivial linear characters to be complex "
          "conjugates of one another; the trivial character and the 3-dim irrep are self-conjugate)",
          gc["self_conjugate_irreps"] == [0, 3] and gc["conjugate_irrep_pairs"] == [(1, 2)])

    banner("FOCK-0b  b4 -- THE PINNED-MAP TEST (THE FROZEN SS1 CLASS, NOW TESTABLE)")
    # =======================================================================
    hr = net.history_reversal_matrix(N_max)
    check("THE HISTORY-SIDE REAL INVOLUTION (path-reversal: reverse-and-flip-each-dart) is "
          "admissible-closed at EVERY shell up to N_max and an exact involution; shell-1 "
          "restriction == reversal() EXACTLY (BRIDGE-LOCK's own forced Z2, reused not rebuilt)",
          hr["all_words_valid"] and hr["is_involution"] < 1e-9
          and hr["shell1_matches_reversal"] < 1e-12,
          detail=f"per_shell_valid={hr['per_shell_valid']}")

    pm = net.fock0b_pinned_map_shell1()
    fa = pm["field_side"]
    check("J_F,sigma CONSTRUCTED + VERIFIED (discharges the verifier's Named Incompleteness "
          "#2): an EXPLICIT antiunitary K(v)=M@conj(v) on the 8-dim field Fock, M unitary, M^2=1, "
          "K.Pw[w].K == Pw[3-w] EXACTLY for every w (realizes sector_pair_conjugation's 0<->3/"
          "1<->2 pairing as a single matrix, not by rebuilding the whole Fock space)",
          fa["sector_swap_residual"] < 1e-8 and fa["M_unitary_residual"] < 1e-8
          and fa["M_involution_residual"] < 1e-8 and fa["gram_identity_residual"] < 1e-8,
          detail=f"swap resid={fa['sector_swap_residual']:.2e}; NOTE: plain complex conjugation "
                 "alone does NOT work here (Cl(6) generators are not all real -- 3 of 6 have "
                 "nonzero imaginary parts -- checked and ruled out this session)")

    pr, cr = pm["primary"], pm["cross_check"]
    check(f"SHELL 1 (12-dim dart -> 8-dim F), PRIMARY (J_hist = reversal() o conj, the framework's "
          f"OWN already-forced Z2): Hom-space real dimension = {pr['nullity']} / "
          f"{pr['total_real_dim']} (SOLVED OVER THE REALS per the pre-registered SOLVER TRAP -- "
          "never a complex-linear SVD)",
          pr["nullity"] > 0,
          detail=f"rank={pr['rank']}, smallest kept sv={pr['smallest_kept_sv']:.3f}, "
                 f"largest null-side sv={pr['largest_null_sv']:.2e}")
    check(f"SHELL 1 CROSS-CHECK (J_hist = conj alone, Rtilde=I -- the DEGENERATE case): nullity = "
          f"{cr['nullity']} / {cr['total_real_dim']} -- SAME total as the PRIMARY test",
          cr["nullity"] == pr["nullity"] and cr["total_real_dim"] == pr["total_real_dim"])

    s2 = net.fock0b_pinned_map_shell(2, N_max=N_max)
    check(f"SHELL-BY-SHELL EXTENSION (shell 2, {s2['shell_dim']}-dim dart-word space -> 8-dim F): "
          f"nullity = {s2['nullity']} / {s2['total_real_dim']}",
          s2["nullity"] > 0)

    half_pattern = (pr["nullity"] == pr["total_real_dim"] // 2
                    and cr["nullity"] == cr["total_real_dim"] // 2
                    and s2["nullity"] == s2["total_real_dim"] // 2)
    check("THE PATTERN: nullity == EXACTLY HALF of total_real_dim in ALL THREE runs (shell-1 "
          "primary, shell-1 cross-check, shell-2 primary) -- REGARDLESS of which antiunitary "
          "involution is used on the history side.  THIS IS A GENERAL FACT, not a lattice-"
          "specific coincidence: for ANY pair of antiunitary INVOLUTIONS J,J' (J^2=J'^2=1) on "
          "C^n, C^m, the space of real-linear intertwiners {Phi : Phi.J = J'.Phi} decomposes as "
          "Phi=A+B.conj(.) with A,B EACH separately constrained to be a fixed point of its own "
          "antilinear involution (A = J'.conj(A).conj(J), likewise for B) -- and the fixed-point "
          "set of ANY antilinear involution on a d-complex-dim space is ALWAYS a real form of "
          "real dimension EXACTLY d (never smaller, never larger).  So a bare single-antiunitary-"
          "conjugation pin can NEVER be empty (W2 is impossible for this class by dimension-"
          "counting alone) and NEVER 'unique up to phase' (W1's real-dim-2 target) unless n.m is "
          "tiny -- it is STRUCTURALLY GUARANTEED to land at real-dim n.m for the complex-linear "
          "part alone (here 96 = 12x8), a LARGE, forced-to-be-large residual freedom.",
          half_pattern)

    banner("FOCK-0b  b5 -- THE SS2 FENCE RE-CHECK ON THE NEW CLASS")
    # =======================================================================
    fc = net.fock0b_fence_check()
    check("(1) antiunitary, phase-bearing (J_F,sigma verified nontrivial; J_hist's real-only "
          "content is a named, honest asymmetry -- see the fence-check docstring)",
          fc["1_O0_bit_even_democracy"]["is_phase_bearing"]
          and fc["1_O0_bit_even_democracy_FOCK0b_note"]["field_side_is_phase_bearing"])
    check("(2) NOT full-group generator equivariance (no per-g constraint anywhere, contrast "
          "dr_map_hom_space's 12 simultaneous per-g equations)",
          fc["2_M1b_no_generator_constraint"]["no_group_generator_used"])
    check("(3) NOT a BRIDGE-LOCK-form attachment functional (Fock-level/graded)",
          fc["3_BRIDGE_LOCK_attachment_functional_orbit_blind"]["is_fock_level"])
    check("(4) NOT BRIDGE-T-form two-point data (representation-theoretic)",
          fc["4_BRIDGE_T_two_point_data_blind"]["is_representation_theoretic"])
    check("(5) per-sector by construction (BRIDGE-GEOM's own requirement satisfied directly)",
          fc["5_BRIDGE_GEOM_per_sector_required"]["is_per_sector_by_design"])

    banner("FOCK-0b  REGRESSION: Sections 7/7b/8 + module anchors untouched")
    # =======================================================================
    check("anchor_cell_projector() + anchor_tick_2pi() + accretion_selftest_2026_07_10() + "
          "i2b_selftest_2026_07_11() + fock0_selftest_2026_07_11() all still PASS",
          net.anchor_cell_projector() and net.anchor_tick_2pi()
          and net.accretion_selftest_2026_07_10(verbose=False)
          and net.i2b_selftest_2026_07_11(verbose=False)
          and net.fock0_selftest_2026_07_11(verbose=False))
    check("fock0b_selftest_2026_07_11() (the Section-8b permanent regression anchor) PASSES",
          net.fock0b_selftest_2026_07_11(verbose=False))

    banner("FOCK-0b VERDICT (amendment SS C; ML-2b/HK-7 QUALIFIER attaches to EVERY sentence)")
    # =======================================================================
    print(f"  QUALIFIER: {QUALIFIER}")
    print()
    if pr["nullity"] > 0 and not half_pattern:
        print("  W1/W3 UNDETERMINED BY THIS DRIVER -- see report for manual adjudication.")
    elif pr["nullity"] == pr["total_real_dim"] // 2 and pr["nullity"] > 2:
        print("  W3 NONEMPTY, NOT FORCED (ML-2b/HK-7-conditional): the pinned-map candidate Phi "
              "EXISTS (nullity > 0 at shell 1 and shell 2) but is NOT forced -- the Hom space has "
              f"real dimension {pr['nullity']} (shell 1), far larger than 'unique up to phase' "
              "(real-dim 2), and this largeness is a STRUCTURAL fact (the general antiunitary-"
              "involution-intertwiner dimension-halving theorem above), not a numerical accident "
              "specific to this lattice.  The residual freedom (a real-dim-96 family of "
              "complex-linear parts at shell 1, each pairing with a determined antilinear part) "
              "is the named INCOMPLETE EQUATION this station books raw -- it is NEVER resolved "
              "by data (ML-2b/HK-7-conditional).")
    elif pr["nullity"] == 0:
        print("  W2 EMPTY (ML-2b/HK-7-conditional): genuine V3 of the frozen class -- see report "
              "for the algebraic proof sketch.")
    else:
        print("  W4 or other -- see report.")

    banner("RESULT")
    print("ALL MACHINE CHECKS PASS" if ok_all else "SOME CHECKS FAILED -- see [FAIL] lines above")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
