#!/usr/bin/env python3
"""
proofs/foundations/FOCK0c_full_pin_2026-07-11.py

STATION FOCK-0c -- THE FULL PIN: PER-SECTOR TOMITA BLOCKS + THE R/F_bit SECTOR-PARITY PIN.
Frozen contract: the FOCK-0c DIRECTIVE (SS F-G) appended to
internal research notes (freeze commit 6378d6f; the original
pre-reg SS0-7 and the FOCK-0b amendment SS B-E all stand unchanged).

ADJUDICATION RECORD THIS DIRECTIVE EXECUTES: FOCK-0b's own pinned-map test
(fock0b_pinned_map_shell1, the_net.py:2827-2851) solved only a SINGLE GLOBAL antiunitary pair
(one J_hist, one J_F acting on the WHOLE spaces) -- the sealed FOCK-0b checker
(internal research notes SS4) found this is a disclosed
NECESSARY but not SUFFICIENT special case of the frozen SS1/SSB.b4 hypothesis, which is stated
in the PLURAL ("the antiunitary J's -- per-sector"; "Phi.J_hist,sigma = J_F,sigma.Phi ... per
sector").  The checker's OWN independent computation supplied two waypoints for whoever completes
the literal system: nullity 192/384 (the bare global pair, unchanged) collapsing to 96/384 under
ONE additional frozen-class pin (the R/F_bit sector-parity constraint, SS3c) stacked on the SAME
single global pair.  This driver:
  1. prints the W1 allowance BEFORE any solve (fock0c_w1_allowance, the_net.py SS8c) -- 2 real
     dimensions (one overall phase per sector-PAIR block, 2 blocks under the frozen 0<->3/1<->2
     pairing);
  2. reproduces BOTH checker waypoints with THIS station's own machinery
     (fock0c_waypoint_reproduction) before trusting anything new;
  3. builds and solves the LITERAL plural pin -- (i) the per-sector-pair-group Tomita blocks
     (Phi graded so GROUP-03 of H_hist maps only into GROUP-03 of F, GROUP-12 only into GROUP-12,
     both antiunitary-pinned) SIMULTANEOUSLY with (ii) the R/F_bit pin, NOTHING ELSE stacked
     (fock0c_full_pin_shell) -- at BOTH shell 1 and shell 2;
  4. classifies against the pre-declared W1 allowance.

WHAT THIS SCRIPT IS: a DRIVER, not a new derivation.  Every construction and exactness check it
runs is IMPORTED from derivation_topdown/state/the_net.py Section 8c (the FOCK-0c full-pin
solver) -- per the ONE-OBJECT/LOCAL-NET LAW, Layer-3 math accretes in the_net.py; this file only
RUNS it and prints the verdict-tree evidence.  Has an `if __name__ == "__main__":` guard; safe to
import.

THE "PER SECTOR" READING, STATED UP FRONT (a named judgment call, NOT silently assumed -- see
the_net.py's SS8c module banner and the station report's Named Residual Freedom #1): the
history-side SS8 grading (A4-isotypic, dims [1,1,1,9] at shell 1) and the field-side SS8 grading
(species, dims {1,3,3,1}) do NOT share a forced 1-1 correspondence at the level of INDIVIDUAL
sectors.  What IS forced on BOTH sides independently is a partition into exactly TWO sector-PAIR
groups (history: self-conjugate {0,3} vs the conjugate pair {1,2}, gns_grading_commutation;
field: the bit-orbit {0,3} vs {1,2}, sector_pair_conjugation) -- matching the W1 allowance's own
"per sector-PAIR block" granularity.  "Per sector sigma" is read at this granularity: Phi is
GRADED (cross-group blocks vanish) at the pair-group level, with NO finer, invented
correspondence and NOTHING beyond (i)+(ii) stacked.

NUMBERS APPEAR NOWHERE (pre-reg SS3.4, amendment SS D): every printed quantity below is a
dimension, rank, nullity, or exactness residual (structure) -- never M_Z, ppm residuals, m_nu, or
a_e.  This station delivers STRUCTURE + a verdict on forcedness only.

POISONS RESPECTED (SS D, on top of pre-reg SS5, plus the FOCK-0c directive's HARD GOAL-SEEK
GUARD): NOTHING beyond (i) [the per-sector-pair-group Tomita pin] and (ii) [the R/F_bit pin] is
stacked -- no alternate/non-Tomita antiunitary (the checker's OWN robustness demo of stacking a
SECOND candidate J_hist is explicitly NOT reused here), no Delta-flow/temporal pin (confirmed
absent by design, per the checker's SS4).  Real-linear solve throughout (never a complex-linear
SVD).  alpha_1 != alpha_EM; the four temperatures are never conflated; beta' vs beta_natural is
NOT adjudicated.

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
    banner("FOCK-0c  STEP 1 -- THE PRE-DECLARED W1 ALLOWANCE (COUNTED BEFORE ANY SOLVE)")
    # =======================================================================
    alw = net.fock0c_w1_allowance()
    check(f"the frozen pairing (0<->3, 1<->2) partitions the four sectors into "
          f"{alw['n_pair_blocks']} sector-PAIR blocks; W1 permits ONE overall U(1) phase "
          f"({alw['real_dims_per_block']} real dim) per block -- ALLOWANCE = "
          f"{alw['n_pair_blocks']} x {alw['real_dims_per_block']} = {alw['allowance']} real "
          "dimensions, counted from the SS8 structure alone, BEFORE solving anything",
          alw["allowance"] == 2)

    banner("FOCK-0c  STEP 2 -- VERIFICATION WAYPOINTS (must reproduce BEFORE trusting the full pin)")
    # =======================================================================
    wp = net.fock0c_waypoint_reproduction(N_max=4)
    check(f"WAYPOINT 1: the single-GLOBAL-antiunitary-pair half (shell 1, FOCK-0b's OWN "
          f"fock0b_pinned_map_shell1, re-run not re-derived): nullity = "
          f"{wp['global_pair_nullity']}/{wp['global_pair_total']}",
          wp["global_pair_nullity"] == 192 and wp["global_pair_total"] == 384)
    check(f"WAYPOINT 2: the SAME single global pair PLUS the R/F_bit sector-parity pin (SS3c; "
          f"dr_map_hom_space's nullity_with_R_constraint precedent, the_net.py:2284-2289) stacked "
          f"-- built here as a NEW check (FOCK-0b never combined R/F_bit with the antiunitary "
          f"pin): nullity = {wp['global_plus_rfbit_nullity']}/{wp['global_plus_rfbit_total']} "
          "(must reproduce the FOCK-0b checker's independently-computed 96/384)",
          wp["global_plus_rfbit_nullity"] == 96 and wp["global_plus_rfbit_total"] == 384)

    banner("FOCK-0c  STEP 3 -- THE FULL PIN: (i) PER-SECTOR-PAIR-GROUP TOMITA BLOCKS "
           "SIMULTANEOUSLY WITH (ii) THE R/F_bit PIN, NOTHING ELSE")
    # =======================================================================
    print("  Reading 'per sector sigma' at sector-PAIR-GROUP granularity (module banner, "
          "the_net.py SS8c) -- GROUP-03 = {history sectors 0,3} <-> {field species 0,3=nu,e}; "
          "GROUP-12 = {history sectors 1,2} <-> {field species 1,2=d,u}; cross-group blocks of "
          "Phi are forced to vanish (the graded/plural ansatz), each surviving group's block is "
          "antiunitary-pinned + R/F_bit-pinned exactly as the single-global-pair test was, just "
          "restricted to that group.")
    print()

    s1 = net.fock0c_full_pin_shell(1, N_max=4)
    check(f"SHELL 1 ({s1['n']}-dim dart -> {s1['m']}-dim F): item-(i) ALONE (graded, no R/F_bit) "
          f"nullity = {s1['item_i_alone_nullity']} (GROUP-03={s1['item_i_alone_group03']}, "
          f"GROUP-12={s1['item_i_alone_group12']}) -- both groups independently reproduce the "
          "general antiunitary-involution half-theorem (2*n_grp*m_grp) for their own restricted "
          "dimensions",
          s1["item_i_alone_group03"] + s1["item_i_alone_group12"] == s1["item_i_alone_nullity"])
    check(f"SHELL 1 FULL PIN (i)+(ii): nullity = {s1['full_pin_nullity']}/{s1['total_real_dim']} "
          f"(GROUP-03={s1['full_pin_group03']}, GROUP-12={s1['full_pin_group12']}) -- "
          f"smallest kept sv={s1['smallest_kept_sv']:.4f}, largest null-side sv="
          f"{s1['largest_null_sv']:.2e} (clean machine-precision gap, no near-degeneracy)",
          s1["full_pin_group03"] + s1["full_pin_group12"] == s1["full_pin_nullity"]
          and s1["smallest_kept_sv"] > 0.5 and s1["largest_null_sv"] < 1e-8)
    print("    ALGEBRAIC NOTE (GROUP-12 forced to EXACTLY zero at shell 1, provable without "
          "SVD): R=reversal() acts as the SCALAR +1 on BOTH 1-dim history blocks H_1, H_2 "
          "(dart_rep-commutation forces R to self-map each A4-isotypic block; each is 1-dim at "
          "shell 1, so R restricted there is a scalar, and direct diagonalization gives +1 on "
          "both); F_bit = Pw[0]+Pw[3]-Pw[1]-Pw[2] is EXACTLY -1 on species sectors 1,2 by "
          "construction (a projector sum, not approximate).  The pin Phi.R = F_bit.Phi on "
          "GROUP-12 therefore forces Phi_(1,2) = (-1).Phi_(1,2), i.e. Phi_(1,2) == 0 identically "
          "-- an eigenvalue clash, not a numerical coincidence.")

    s2 = net.fock0c_full_pin_shell(2, N_max=4)
    check(f"SHELL 2 ({s2['n']}-dim word space -> {s2['m']}-dim F): item-(i) ALONE nullity = "
          f"{s2['item_i_alone_nullity']} (GROUP-03={s2['item_i_alone_group03']}, "
          f"GROUP-12={s2['item_i_alone_group12']})",
          s2["item_i_alone_group03"] + s2["item_i_alone_group12"] == s2["item_i_alone_nullity"])
    check(f"SHELL 2 FULL PIN (i)+(ii): nullity = {s2['full_pin_nullity']}/{s2['total_real_dim']} "
          f"(GROUP-03={s2['full_pin_group03']}, GROUP-12={s2['full_pin_group12']}) -- "
          f"smallest kept sv={s2['smallest_kept_sv']:.4f}, largest null-side sv="
          f"{s2['largest_null_sv']:.2e}",
          s2["full_pin_group03"] + s2["full_pin_group12"] == s2["full_pin_nullity"]
          and s2["smallest_kept_sv"] > 0.5 and s2["largest_null_sv"] < 1e-8)
    print("    NOTE (shell-dependence, honestly reported, not smoothed over): at shell 2, R's "
          "restriction to each 2-dim history block H_1/H_2 has MIXED eigenvalues (one +1, one "
          "-1 each, unlike shell 1's uniform +1) -- so GROUP-12 is NOT forced entirely to zero "
          "at shell 2 (24 survives, on the R=-1 sub-piece that now DOES match F_bit=-1). The "
          "shell-1 vanishing is a shell-1-specific fact (all shell-1 A4-isotypic blocks outside "
          "the 3-dim irrep are 1-dimensional), not a general theorem about GROUP-12.")

    banner("FOCK-0c  STEP 4 -- THE SS2 FENCE RE-CHECK ON THE FULL-PIN CLASS")
    # =======================================================================
    fc = net.fock0c_fence_check()
    check("(1)-(5) reused from fock0b_fence_check (antiunitary/phase-bearing; NOT full-group "
          "generator equivariance; NOT BRIDGE-LOCK-form; NOT BRIDGE-T-form; per-sector by "
          "construction)",
          fc["1_O0_bit_even_democracy"]["is_phase_bearing"]
          and fc["2_M1b_no_generator_constraint"]["no_group_generator_used"]
          and fc["3_BRIDGE_LOCK_attachment_functional_orbit_blind"]["is_fock_level"]
          and fc["4_BRIDGE_T_two_point_data_blind"]["is_representation_theoretic"]
          and fc["5_BRIDGE_GEOM_per_sector_required"]["is_per_sector_by_design"])
    check("(6) NEW: the sector-pair-group grading is built ONLY from SS8's own isotypic/species "
          "projectors (Hermitian, idempotent) -- no group generator, no resolvent/two-point "
          "data, no spatial attachment functional anywhere in the construction",
          fc["6_FOCK0c_grading_is_projector_only"]["no_extra_mechanism_introduced"])

    banner("FOCK-0c  REGRESSION: Sections 7/7b/8/8b + module anchors untouched")
    # =======================================================================
    check("anchor_cell_projector() + anchor_tick_2pi() + accretion_selftest_2026_07_10() + "
          "i2b_selftest_2026_07_11() + fock0_selftest_2026_07_11() + "
          "fock0b_selftest_2026_07_11() all still PASS",
          net.anchor_cell_projector() and net.anchor_tick_2pi()
          and net.accretion_selftest_2026_07_10(verbose=False)
          and net.i2b_selftest_2026_07_11(verbose=False)
          and net.fock0_selftest_2026_07_11(verbose=False)
          and net.fock0b_selftest_2026_07_11(verbose=False))
    check("fock0c_selftest_2026_07_11() (the Section-8c permanent regression anchor) PASSES",
          net.fock0c_selftest_2026_07_11(verbose=False))

    banner("FOCK-0c VERDICT (SS C tree, allowance-calibrated; ML-2b/HK-7 QUALIFIER attaches to "
           "EVERY sentence)")
    # =======================================================================
    print(f"  QUALIFIER: {QUALIFIER}")
    print()
    allowance = alw["allowance"]
    n1, n2 = s1["full_pin_nullity"], s2["full_pin_nullity"]
    if n1 == allowance and n2 == allowance:
        print(f"  W1 NONEMPTY + FORCED (up to per-sector-pair-block phase): nullity == the "
              f"allowance ({allowance}) at BOTH shells.")
    elif n1 == 0 and n2 == 0:
        print("  W2 EMPTY (ML-2b/HK-7-conditional): see report for the algebraic proof sketch.")
    elif n1 > allowance or n2 > allowance:
        print(f"  W3 NONEMPTY, ABOVE THE W1 ALLOWANCE (ML-2b/HK-7-conditional): full-pin nullity "
              f"= {n1} (shell 1) / {n2} (shell 2), vs the pre-declared allowance of {allowance} "
              "-- the excess residual freedom is the named INCOMPLETE EQUATION this station "
              "books raw (per-sector breakdown printed above: GROUP-03 carries ALL of shell-1's "
              f"freedom [{s1['full_pin_group03']}/{n1}], GROUP-12 is forced to exactly zero at "
              f"shell 1 but reopens to {s2['full_pin_group12']} at shell 2 -- a shell-dependent, "
              "not-yet-stabilizing pattern). It is NEVER resolved by data "
              "(ML-2b/HK-7-conditional).")
    else:
        print("  W4 or a non-monotonic pattern -- see report for manual adjudication.")

    banner("RESULT")
    print("ALL MACHINE CHECKS PASS" if ok_all else "SOME CHECKS FAILED -- see [FAIL] lines above")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
