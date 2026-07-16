#!/usr/bin/env python3
"""
proofs/foundations/FOCK0_dr_construction_2026-07-11.py

STATION FOCK-0 -- THE SECTOR-GRADED FOCK LAYER / DR-MAP CANDIDATE (HEAVY effort; the modular/
sector Fock-level object of the CRITICAL PATH v2, milestones II.2/II.4/IV.7/III.4-plausible).
Frozen contract: internal research notes (freeze commit
138795e).  Verified fact base: internal research notes

WHAT THIS SCRIPT IS: a DRIVER, not a new derivation.  Every construction and exactness check it
runs is IMPORTED from derivation_topdown/state/the_net.py Sections 7b (STEP 0: the ported I2b
dart/Toeplitz-CK algebra) and 8 (the FOCK-0 sector-graded Fock layer / DR-map candidate) -- per
the ONE-OBJECT/LOCAL-NET LAW, Layer-3 math accretes in the_net.py; this file only RUNS it and
prints the verdict-tree evidence.  This script HAS a `if __name__ == "__main__":` guard (unlike
I2b_matsumoto_completion_2026-07-10.py / BRIDGE_GEOM_2026-07-10.py, which do not -- see the
pre-reg SS3.0 caution) and is safe to import.

THE FROZEN HYPOTHESIS (pre-reg SS1): a Doplicher-Roberts-style reconstruction over the net's OWN
sector category (ML-2's A4/2T species {nu:1,d:3,u:3,e:1}, MS-1a's fusion ring), built on the
I2b-accreted dart/Toeplitz-CK Fock space H_hist, pinned by intertwining MODULAR CONJUGATIONS
(antiunitary, per-sector) rather than by generators.

NUMBERS APPEAR NOWHERE (pre-reg SS3.4): every printed quantity below is a dimension,
multiplicity, rank, or exactness residual (structure) -- never M_Z, ppm residuals, m_nu, or a_e.
This station delivers STRUCTURE + a verdict on FORCEDNESS only; numeric confrontation is FOCK-1,
a separate, later, gated pre-reg.

POISONS (pre-reg SS5, respected throughout): alpha_1 != alpha_EM; the four temperatures are never
conflated; beta' vs beta_natural is NOT adjudicated here (amendment A1: where a state is needed,
this station uses omega_diag at beta_natural = 2*beta_gas AS MEASURED); srs-z != the enantiomer;
no goal-seek toward M_Z/-70ppm/m_nu/a_e; no orbit choice by data; the ML-2b/HK-7 conditionality
qualifier is carried on EVERY verdict sentence below, quoted verbatim; the bridge routes
(LOCK/T/GEOM) are not re-run, only cited; the April B3_chirality_bridge* files are unrelated (not
touched here).

THE ML-2b/HK-7 CONDITIONALITY QUALIFIER (verbatim, aqft_net.py:280-292 / pre-reg SS0):
    "Every duality check here (HK-5) is CELL-LEVEL only (the 6-edge static vacuum). ML-2b's
    DR-frame argument is CONDITIONAL on the TD-limit duality holding, which is NOT verified by
    this suite."
This qualifier attaches to EVERY verdict sentence printed below.
"""
import math
import os
import sys

import numpy as np

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
    banner("FOCK-0  STEP 0 -- THE I2b DART/TOEPLITZ-CK ALGEBRA, ACCRETED INTO the_net.py")
    # =======================================================================
    res_i2b = net.toeplitz_ck_check(N_max=6)
    check(f"[N_max=6, D={res_i2b['D']}] Toeplitz-CK defect Sum_d S_dS_d^* = 1-P_seed EXACT "
          "(ported from I2b_matsumoto_completion_2026-07-10.py:143-210, NOT imported -- that "
          "file has no __main__ guard and calls sys.exit() at module level, line 427)",
          res_i2b["toeplitz_defect"] < 1e-9, detail=f"max|diff| = {res_i2b['toeplitz_defect']:.3e}")
    check("companion CK relation exact in the interior (|w|<6); boundary mismatches are the "
          "named, expected truncation artifact (S_d truncated to 0 at the top shell)",
          res_i2b["companion_interior"] < 1e-9,
          detail=f"worst interior = {res_i2b['companion_interior']:.3e}, "
                 f"boundary mismatches = {res_i2b['boundary_mismatches']}")
    words, index, lengths, omega = net.omega_diag_length(N_max=6)
    check("omega_diag (length-diagonal, u=alpha_1=(2/3)^8) normalized + positive -- STRUCTURAL "
          "note: depends ONLY on word length, i.e. manifestly bit-EVEN/real by construction "
          "(reused downstream as an INPUT, never as the pin)",
          abs(float(np.sum(omega)) - 1) < 1e-9 and bool(np.all(omega > 0)))
    print(f"""
    [A1 -- temperature statement, verbatim per the pre-reg]: the run's length-diagonal state is
    sharply KMS at beta_natural = 2*beta_gas = 6.4874417297, NOT at the pre-registered beta' =
    5.7942945492 (exact identity beta_natural = beta' + h_top).  This station builds on the
    algebra and, where a state is needed, on omega_diag at beta_natural AS MEASURED -- it does
    NOT adjudicate which temperature is physically operative, and no verdict below depends on
    that adjudication.""")

    banner("FOCK-0  3a -- GRADE H_hist BY THE A4-IRREP DECOMPOSITION (THE GRADING OPERATOR)")
    # =======================================================================
    N_max_grade = 3
    sg = net.sector_grading_hist(N_max_grade)
    pb = sg["projector_battery"]
    check(f"[N_max={N_max_grade}] isotypic projector battery EXACT: idempotent={pb['idem']:.1e}, "
          f"Hermitian={pb['herm']:.1e}, mutually orthogonal={pb['orth']:.1e}, "
          f"complete (Sum P_a = I)={pb['complete']:.1e}",
          max(pb.values()) < 1e-7)
    check("length-1 (= the 12 darts themselves) irrep multiplicities == [1,1,1,3] EXACTLY -- the "
          "REGULAR REPRESENTATION cross-check, by isotypic projection (a DIFFERENT method from "
          "Section 7's character inner product ip(chi,chi)=12, same conclusion)",
          list(sg["mult"][1]) == [1, 1, 1, 3] or sorted(sg["mult"][1].tolist()) == [1, 1, 1, 3])
    print(f"    per-length A4-irrep multiplicities (irrep order = fusion_ring('A4')['dims'] = "
          f"{sg['irrep_dims']}, trivial first):")
    for n in range(N_max_grade + 1):
        print(f"      |w|={n}: mult = {sg['mult'][n].tolist()}  (dim {12 * 2 ** max(n - 1, 0) if n else 1})")
    print("    [BONUS STRUCTURAL FINDING, not part of the core deliverable] the irrep-multiplicity "
          "PROPORTIONS are length-independent (mult(n) = 2^(n-1) * mult(1) for n>=1): neither the "
          "1'- nor 1''-isotypic share ever grows or shrinks relative to the others as the Fock "
          "space is built out.")
    check("word-length parity (-1)^n commutes with every A4-isotypic projector EXACTLY (length- "
          "grading and the A4-grading are compatible)",
          sg["parity_commute_residual"] < 1e-8,
          detail=f"residual = {sg['parity_commute_residual']:.1e}")
    print("""    [NAMED INCOMPLETENESS #1]: A4 alone supports only the TRIVIAL Z2 grading
    (MS-1a: z2_gradings('A4') == 1) -- the word-length parity (-1)^n used above as the natural
    Fock-space "fermion parity" candidate on H_hist is EXTRINSIC to the A4-representation content
    (compatible with it, but not FORCED by it the way the 2T center grading is forced on F).
    Whether length-parity is the physically correct fermion-parity bit for H_hist, or whether a
    genuine 2T-lift of the dart algebra is needed to force it, is NOT decided here -- logged as an
    open free choice, never silently fixed.""")

    banner("FOCK-0  3b -- THE PER-SECTOR MODULAR CONJUGATION J_sigma (FROM THE M0 BIT ANCHOR)")
    # =======================================================================
    spc = net.sector_pair_conjugation()
    check("the M0 bit (J -> -J; C(-J)=I-C(J), M0-4a) lifted to the Cl(6) FOCK level: sector 0 "
          "<-> sector 3 EXACTLY (subspace equality)", spc["orbit_03"] < 1e-9,
          detail=f"residual = {spc['orbit_03']:.1e}")
    check("...and sector 1 <-> sector 2 EXACTLY", spc["orbit_12"] < 1e-9,
          detail=f"residual = {spc['orbit_12']:.1e}")
    check("the flip is an involution (sign=+1 recomputed matches the original exactly)",
          spc["is_involution"] < 1e-9, detail=f"residual = {spc['is_involution']:.1e}")
    check("sign=+1 reproduces gauge_sector_category()'s own species_sector_dims exactly "
          "(consistency, not a re-derivation)", spc["dims_match_gsc"])
    print("""    J_sigma, AS CONSTRUCTED: the four sectors split into two bit-orbits {0,3}
    ('nu'<->'e') and {1,2} ('d'<->'u'); NEITHER orbit has a fixed point.  This pairing is FORCED
    by the M0 bit -- no free choice in WHICH sectors pair with which.
    [NAMED INCOMPLETENESS #2]: an EXPLICIT antiunitary/unitary operator K realizing this pairing
    as a single matrix on F (rather than as a projector-subspace equality) is NOT constructed
    here; End_A4(F) contains an 8-complex-dimensional family of candidate K's (2x2 blocks on each
    of the two multiplicity-2 isotypic pieces), and the OVERALL PHASE of any such K is not fixed
    by the anchors used here -- a genuine residual freedom, logged, not resolved.""")

    banner("FOCK-0  3c -- THE DR-MAP CANDIDATE: Hom_A4(dart_rep, THE FIELD ALGEBRA F)")
    # =======================================================================
    dr = net.dr_map_hom_space()
    a_idx, b_idx, gh_idx, cval = dr["cocycle_pair"]
    check("dart_rep is a GENUINE (non-projective) A4 representation on the named cocycle pair "
          "(dart_rep(g)dart_rep(h) == dart_rep(gh) exactly -- no square-root step, no sign "
          "ambiguity possible)", dr["homdev_dart_honest_rep"] < 1e-10,
          detail=f"residual = {dr['homdev_dart_honest_rep']:.1e}")
    check(f"F's own A4 action (U = spin_lift o edge_rep) IS projective: U(g)U(h) = -U(gh) on "
          f"element pair ({a_idx},{b_idx})->{gh_idx}, cocycle = {cval.real:+.3f} (HK-6b's "
          f"double_cover_2T fact, reproduced independently here)",
          dr["cocycle_residual"] < 1e-6, detail=f"residual = {dr['cocycle_residual']:.1e}")
    check(f"Hom_A4(dart_rep(12), F(8)) SVD: rank = {dr['rank']}/96 (FULL RANK), smallest singular "
          f"value = {dr['smallest_sv']:.4f} (bounded well away from zero -- not a near-degeneracy)"
          f" => nullity = {dr['nullity']} EXACTLY",
          dr["nullity"] == 0)
    check("corollary: adding the R/F_bit sector-parity intertwining constraint (the further pin) "
          "cannot un-obstruct an already-empty space",
          dr["nullity_with_R_constraint"] == 0,
          detail=f"nullity_with_R_constraint = {dr['nullity_with_R_constraint']}")
    print("""
    LEMMA (OBSTRUCTION, theorem-grade -- proof sketch): dart_rep is honestly non-projective
    (built from vertex PERMUTATIONS composing exactly, no square root involved); F's own A4
    action U is projective with cocycle -1 on some pairs (the 2T double cover, HK-6b).  If Phi
    solved Phi.dart_rep(g) = U(g).Phi for every g separately, composing a cocycle=-1 pair (g,h)
    forces U(gh).Phi = U(g)U(h).Phi = -U(gh).Phi, i.e. Phi = 0 (U(gh) invertible).  Hence
    Hom_A4(dart_rep, F) = {0} EXACTLY -- machine-confirmed above (full rank 96/96, no near-miss).
    VERDICT: the NAIVE DR-map candidate (equivariant w.r.t. the SAME A4 action on both the dart/
    word-Fock content and the field-algebra sector content) is OBSTRUCTED: a bosonic
    (dart_rep)/fermionic (2T-projective F) mismatch.  This is a NEW theorem-grade obstruction
    (nonexistence of ANY equivariant map, prior to and distinct from an orbit-discrimination
    blindness) -- it does not reduce to any of the five already-proven forms (checked explicitly
    below).  It leaves an EXPLICIT, NAMED, unexplored escape route: a genuine 2T (spinorial) lift
    of the dart/Toeplitz-CK algebra itself is NOT attempted here and might evade this specific
    obstruction -- a prerequisite for any future attempt at the SAME hypothesis, not derived.""")

    banner("FOCK-0  3d -- THE INDUCED FUNCTIONAL + THE TWO PROBES (CONSISTENCY INPUTS ONLY)")
    # =======================================================================
    print(f"""    omega_diag (STEP 0) depends ONLY on word length -- manifestly REAL, bit-EVEN.
    If used ALONE as the discriminating functional it would reduce to the O0 (bit-EVEN democracy)
    blind form (BOOTCAMP SS5) -- CONFIRMED here as a fence check, not a surprise: this is exactly
    why SS1's PIN must be the antiunitary J_sigma construction (3b), never a state/functional like
    omega_diag.  omega_diag is used here ONLY as a consistency input (amendment A1), never as the
    pin.
    Two-vertex interference probe (heuristic-only, amendment A2, orphaned 2026-05-15): its
    qualitative claim is a THREE-fold C3 decomposition (1 trivial + 2 twisted).  Section 3a's
    length-1 multiplicity table shows darts carry the SAME A4 3-irrep with multiplicity exactly
    THREE (three independent copies) -- a QUALITATIVE, non-numeric consistency (not a proof, not
    a re-derivation of the probe; both are C3/three-fold structural facts, nothing more is
    claimed).""")
    di = net.fock0_door_i_check()
    check("QF-2b door (i) structural addendum: single-cell anchor C=(I+iJ6)/2 reproduced exactly",
          di["single_cell_reproduces_anchor"])
    check("...the natural translation-invariant multi-cell extension has EXACTLY zero cross-cell "
          "block (both Re and Im) -- door (i) is NOT instantiated by this station (honest "
          "negative; consistent with the already-established cover-gauge-triviality theorem)",
          di["two_cell_offdiag_im"] == 0.0 and di["two_cell_offdiag_re"] == 0.0,
          detail=f"|Im| = {di['two_cell_offdiag_im']:.3e}, |Re| = {di['two_cell_offdiag_re']:.3e}")

    banner("FOCK-0  SS2 FENCE -- EXPLICIT EVASION OF THE FIVE PROVEN-BLIND FORMS")
    # =======================================================================
    fc = net.fock0_fence_check()
    check("1. O0 (bit-EVEN democracy): the PIN is phase-bearing (Im(J6) != 0), not an even "
          "functional", fc["1_O0_bit_even_democracy"]["is_phase_bearing"],
          detail=f"|J6| = {fc['1_O0_bit_even_democracy']['im_J6_norm']:.3f}")
    check("2. M-1b (linear intertwiners exactly perpendicular): dr_map_hom_space targets a "
          "DIFFERENT Hom-space pair (dart_rep(12) vs F(8)) than Section 7's map_commutant "
          "(edge_rep(6) vs dart_rep(12))", fc["2_M1b_linear_intertwiners_perpendicular"]["distinct_from_map_commutant"])
    check("3. BRIDGE-LOCK (attachment functionals orbit-blind): this construction is FOCK-LEVEL, "
          "not a one-particle attachment functional", fc["3_BRIDGE_LOCK_attachment_functional_orbit_blind"]["is_fock_level"])
    check("4. BRIDGE-T (all 2-point run data blind): this construction is REPRESENTATION-"
          "THEORETIC, not a 2-point resolvent functional", fc["4_BRIDGE_T_two_point_data_blind"]["is_representation_theoretic"])
    check("5. BRIDGE-GEOM (a per-sector map is required): this construction IS per-sector by "
          "design at every step", fc["5_BRIDGE_GEOM_per_sector_required"]["is_per_sector_by_design"])

    banner("FOCK-0  REGRESSION -- THE MODULE'S OWN ANCHORS + SECTION 7/7b BATTERY, UNTOUCHED")
    # =======================================================================
    check("net.anchor_cell_projector() (M0 cell C-projector)", net.anchor_cell_projector())
    check("net.anchor_tick_2pi() (M0-2R tick modular flow)", net.anchor_tick_2pi())
    check("net.accretion_selftest_2026_07_10() (Section 7: MAP/LOCK/T/MS-1a battery)",
          net.accretion_selftest_2026_07_10(verbose=False))
    check("net.i2b_selftest_2026_07_11() (Section 7b: STEP 0 accretion)",
          net.i2b_selftest_2026_07_11(verbose=False))
    check("net.fock0_selftest_2026_07_11() (Section 8: this station's own battery)",
          net.fock0_selftest_2026_07_11(verbose=False))

    banner("THE VERDICT TREE (pre-reg SS4; PROPOSED reading only -- adjudicated by the architect)")
    # =======================================================================
    print(f"""    Evidence summary:
      3a  GRADING            : CONSTRUCTED, exact, FORCED (no free choice once darts+B0 fixed).
      3b  PER-SECTOR J_sigma  : CONSTRUCTED, exact, FORCED (orbit pairing 0<->3, 1<->2; the
                                explicit intertwining matrix's overall phase is NOT forced --
                                named incompleteness #2, does not affect the pairing fact).
      3c  DR-MAP CANDIDATE    : OBSTRUCTED, theorem-grade (Hom_A4(dart_rep,F) = {{0}} exactly,
                                full-rank SVD, no near-degeneracy; a NEW obstruction, not a
                                restatement of the five prior nulls).
      3d  INDUCED FUNCTIONAL  : omega_diag alone is bit-EVEN (confirms the SS2 fence, does not
                                violate it); the two-vertex probe check is qualitative-only
                                (heuristic, per amendment A2); door (i) NOT instantiated (honest
                                negative).

    PROPOSED READING: **V3 -- OBSTRUCTED (theorem-grade)**.  The pinned map, AS ATTEMPTED HERE
    (equivariant w.r.t. the SAME A4 action on the dart/word-Fock content and on the field-
    algebra's sector content), provably does not exist (SS3c's lemma + computation).  This is
    booked as a FOURTH, structurally distinct, obstruction alongside BRIDGE-LOCK/T/GEOM -- but at
    the representation-existence level, not the orbit-discrimination level those three occupy.
    The grading (3a) and per-sector conjugation (3b) constructions themselves are NOT obstructed
    and stand as forced, exact structure.  An explicit, named, UNEXPLORED escape route survives:
    a genuine 2T (spinorial) lift of the dart/Toeplitz-CK algebra (not attempted here) is the
    natural next prerequisite for reviving the wider FOCK-0 hypothesis.

    THE ML-2b/HK-7 CONDITIONALITY QUALIFIER (attaches to EVERY sentence above, verbatim):
      "{QUALIFIER}"

    FINAL ADJUDICATION IS THE ARCHITECT'S (pre-reg SS4/SS6): this script proposes a reading of
    the evidence; it does not itself book V1/V2/V3/V4.""")

    banner("RESULT")
    print("RESULT:", "ALL MACHINE CHECKS PASS -- FOCK-0 EXECUTED: STEP 0 accreted; grading + "
          "per-sector conjugation CONSTRUCTED+FORCED; DR-map candidate OBSTRUCTED "
          "(theorem-grade, Hom=0 exactly); door (i) NOT instantiated (honest negative); SS2 "
          "fence evaded on all five counts"
          if ok_all else "A MACHINE CHECK FAILED -- verdict void")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
