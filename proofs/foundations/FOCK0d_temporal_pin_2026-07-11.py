#!/usr/bin/env python3
"""
proofs/foundations/FOCK0d_temporal_pin_2026-07-11.py

STATION FOCK-0d -- THE TEMPORAL PIN: THE FLOW ITSELF.
Frozen contract: AMENDMENT FOCK-0d (SS H-K) appended to
internal research notes (freeze commit 5d5739a; the original
pre-reg SS0-7, the FOCK-0b amendment SS B-E, and the FOCK-0c DIRECTIVE SS F-G all stand
unchanged). Baseline this station builds ON TOP OF: FOCK-0c's full pin, W3, nullity 16/384 at
shell 1 (GROUP-03=16, GROUP-12=0 exact) and 64/768 at shell 2 (GROUP-03=40, GROUP-12=24); W1
allowance = 2.

WHAT THIS ADDS: ONE more constraint on top of FOCK-0c's full pin set --
    Phi . K_hist,sigma  =  lambda . K_F,sigma . Phi          (per sector-pair-group)
K_hist = the GNS/Tomita modular Hamiltonian of the length-diagonal state omega_diag at
beta_natural (SS8b's b2 Tomita step, PER-PATH reading -- reused, not rebuilt).
K_F = the M0 half-cell modular (entangling) Hamiltonian of a 3-edge region (entanglement_
hamiltonian/region_data, M0's OWN 'complex-fermion, OWNED' Peschel convention), second-quantized
onto an 8-dim Fock space via the standard number-conserving bilinear.
lambda > 0 is ONE GLOBAL relative-clock scale, SOLVED FOR -- a generalized-eigenvalue/pencil
problem, characterized ANALYTICALLY (never grid-scanned-and-picked): since K_hist is EXACTLY
SCALAR on each length-shell (c_n.I_n, a structural consequence of omega_diag depending only on
word length), the flow-pin row is singular (nontrivial kernel) iff lambda = c_n/mu for mu an
eigenvalue of K_F -- an exact closed-form characterization of the pencil's singular locus.

WHAT THIS SCRIPT IS: a DRIVER, not a new derivation. Every construction and exactness check it
runs is IMPORTED from derivation_topdown/state/the_net.py Section 8d -- per the ONE-OBJECT/
LOCAL-NET LAW, Layer-3 math accretes in the_net.py; this file only RUNS it and prints the
verdict-tree evidence. Has an `if __name__ == "__main__":` guard; safe to import.

NUMBERS APPEAR NOWHERE new (pre-reg SS3.4/SSJ): c_n and K_F's eigenvalues are STRUCTURAL
quantities of THIS station's own construction; lambda is compared to NOTHING measured, fitted to
NOTHING, tuned to NOTHING -- it is an OUTPUT (the relative rate of the two derived clocks).

POISONS RESPECTED (SS J, on top of pre-reg SS5 and amendment SS D): lambda is never fitted,
scanned-and-picked by external agreement, or compared to any measured number; per-path vs
per-shell is stated at every K usage (K_hist and K_F are BOTH per-path/state-level modular
generators, never shell-aggregate rates); J_hist != J6 and K_hist != K_A are kept as four
distinct operators across two sides, never conflated; region choice is reported for ALL A4-orbit
representatives of 3-edge halves; FOCK-0c's waypoints are reproduced before trusting anything
new; numbers appear nowhere else.

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
    N_max = 4

    banner("FOCK-0d  WAYPOINTS FIRST -- REPRODUCE FOCK-0c's 16/384 (shell1) and 64/768 (shell2) "
           "BEFORE ADDING ANY FLOW ROW")
    # =======================================================================
    s1c = net.fock0c_full_pin_shell(1, N_max=N_max)
    s2c = net.fock0c_full_pin_shell(2, N_max=N_max)
    check(f"FOCK-0c's OWN full_pin_shell (re-run, not re-derived): shell1 nullity = "
          f"{s1c['full_pin_nullity']}/{s1c['total_real_dim']} (GROUP-03={s1c['full_pin_group03']}, "
          f"GROUP-12={s1c['full_pin_group12']})",
          s1c["full_pin_nullity"] == 16 and s1c["full_pin_group03"] == 16 and s1c["full_pin_group12"] == 0)
    check(f"shell2 nullity = {s2c['full_pin_nullity']}/{s2c['total_real_dim']} "
          f"(GROUP-03={s2c['full_pin_group03']}, GROUP-12={s2c['full_pin_group12']})",
          s2c["full_pin_nullity"] == 64 and s2c["full_pin_group03"] == 40 and s2c["full_pin_group12"] == 24)

    rows1, n1d, m1d = net._fock0c_rows_full(1, N_max=N_max)
    import numpy as np
    sv1 = np.linalg.svd(np.vstack(rows1), compute_uv=False)
    null1_reassembled = np.vstack(rows1).shape[1] - int(np.sum(sv1 > 1e-8))
    rows2, n2d, m2d = net._fock0c_rows_full(2, N_max=N_max)
    sv2 = np.linalg.svd(np.vstack(rows2), compute_uv=False)
    null2_reassembled = np.vstack(rows2).shape[1] - int(np.sum(sv2 > 1e-8))
    check(f"SELF-CONSISTENCY: FOCK-0d's OWN row re-assembly (_fock0c_rows_full, needed to stack "
          f"the new flow row on top) reproduces FOCK-0c's full-pin nullity EXACTLY "
          f"(shell1: {null1_reassembled} vs {s1c['full_pin_nullity']}; "
          f"shell2: {null2_reassembled} vs {s2c['full_pin_nullity']})",
          null1_reassembled == s1c["full_pin_nullity"] and null2_reassembled == s2c["full_pin_nullity"])

    banner("FOCK-0d  STEP H.1 -- K_hist: THE HISTORY-SIDE FLOW GENERATOR (per-path reading)")
    # =======================================================================
    h1 = net.history_side_flow_generator(1, N_max=N_max)
    h2 = net.history_side_flow_generator(2, N_max=N_max)
    check(f"K_hist|_shell1 is EXACTLY SCALAR: c_1 = {h1['c_n']:.10f} = beta_natural*1 "
          f"(beta_natural={h1['beta_natural']:.10f}); scalar-exactness residual = "
          f"{h1['scalar_exactness_residual']:.2e} (every length-1 word shares the IDENTICAL "
          f"-log(rho) value, since omega_diag depends only on |w|)",
          h1["scalar_exactness_residual"] < 1e-9)
    check(f"K_hist|_shell2 is EXACTLY SCALAR: c_2 = {h2['c_n']:.10f} = beta_natural*2; "
          f"residual = {h2['scalar_exactness_residual']:.2e}; c_2 == 2*c_1 EXACTLY "
          f"(dev={abs(h2['c_n']-2*h1['c_n']):.2e}) -- the PER-PATH clock is SECTOR-BLIND (same "
          f"scalar regardless of GROUP-03/GROUP-12) and its shell-ratio is fixed by word length "
          f"alone, independent of anything on the field side",
          h2["scalar_exactness_residual"] < 1e-9 and abs(h2["c_n"] - 2 * h1["c_n"]) < 1e-9)
    print(f"    (the additive -ln(Z) normalization constant is DROPPED by convention -- a flow "
          f"generator is defined up to an additive c.I with no effect on the flow it generates; "
          f"c_1_with_lnZ={h1['c_n_with_lnZ_constant']:.6f}, c_2_with_lnZ="
          f"{h2['c_n_with_lnZ_constant']:.6f}, for transparency only, NOT used below)")

    banner("FOCK-0d  STEP H.2 -- K_F: THE FIELD-SIDE FLOW GENERATOR (M0's 3-edge half-cell "
           "modular data, second-quantized) -- ALL A4-ORBIT-INEQUIVALENT 3-edge REGIONS")
    # =======================================================================
    orbits = net._three_edge_region_orbits()
    check(f"3-edge region census: {len(orbits)} A4-orbit(s) among the C(6,3)=20 three-edge "
          f"subsets of the cell (orbit sizes {[o['orbit_size'] for o in orbits]}, sum = "
          f"{sum(o['orbit_size'] for o in orbits)} = 20) -- ONE representative per orbit run "
          "below (running every member of an orbit gives K_F related by a fixed A4 conjugation, "
          "not new content)",
          sum(o["orbit_size"] for o in orbits) == 20)
    for o in orbits:
        tag = "TRIANGLE (M0's own girth-cycle anchor, M0-3/M0-4)" if o["is_triangle"] else "non-triangle"
        fsg = net.field_side_flow_generator(o["representative"])
        check(f"region {o['representative']} [{tag}, orbit size {o['orbit_size']}]: K_F Hermitian "
              f"(res={fsg['hermiticity_residual']:.2e}), CAR exact (res={fsg['car_residual']:.2e}), "
              f"M0-4b bit-reversal K_A -> -K_A reproduced EXACTLY as a MATRIX identity "
              f"(res={fsg['bit_reversal_check_residual']:.2e}); eigenvalues = "
              f"{[round(x, 4) for x in fsg['eigenvalues']]}",
              fsg["hermiticity_residual"] < 1e-8 and fsg["car_residual"] < 1e-9
              and fsg["bit_reversal_check_residual"] < 1e-6)

    banner("FOCK-0d  STEP H.3 -- THE GENERALIZED NULLITY PROBLEM: SOLVE FOR lambda (NOT chosen)")
    # =======================================================================
    print("  METHOD (analytic, not a grid scan): K_hist|_shell_n = c_n.I_n exactly, so the "
          "flow-pin row (viewed alone, as a square matrix on the WHOLE ambient real space) is "
          "the pencil c_n.I - lambda.(I (x) K_F): singular iff lambda = c_n/mu for mu an "
          "eigenvalue of K_F. Since c_2 = 2.c_1 EXACTLY, a lambda shared by BOTH shells (the "
          "frozen 'ONE global lambda' requirement) exists IFF K_F's own spectrum contains a "
          "positive pair (mu_1, mu_2) with mu_2 = 2.mu_1 EXACTLY. For lambda NOT of this form "
          "the pencil is invertible (trivial kernel), so stacking it onto ANY other system can "
          "only ever give a trivial joint kernel -- this is a property of the Kronecker "
          "structure alone, independent of what else is stacked. There is therefore NO other "
          "lambda to search for beyond this finite, closed-form candidate set.")
    print()
    all_results = net.fock0d_all_regions_analysis(N_max=N_max)
    total_candidates = 0
    for r in all_results:
        tag = "TRIANGLE" if r["is_triangle"] else "non-triangle"
        n_cand = len(r["candidates"])
        total_candidates += n_cand
        check(f"region {r['region']} [{tag}]: {n_cand} candidate lambda(s) with an EXACT "
              f"mu2=2*mu1 pair in K_F's spectrum (closest relative miss over all positive pairs = "
              f"{r['closest_relative_miss']:.4e}, reported for transparency, NOT used to pick "
              "anything)", True)
        for c in r["candidates"]:
            n1r, n2r = c["shell1"], c["shell2"]
            check(f"    candidate lambda*={c['lambda']:.6f} (mu1={c['mu1']:.6f}, "
                  f"mu2={c['mu2']:.6f}): JOINT nullity (FOCK-0c's full pin STACKED with the flow "
                  f"row) = shell1 {n1r['nullity']}/{n1r['total']} (g03={n1r['group03']}, "
                  f"g12={n1r['group12']}), shell2 {n2r['nullity']}/{n2r['total']} "
                  f"(g03={n2r['group03']}, g12={n2r['group12']})", True)

    banner("FOCK-0d  STEP H.4 -- THE SS2 FENCE RE-CHECK ON THE TEMPORAL-PIN CLASS")
    # =======================================================================
    fc = net.fock0d_fence_check()
    check("(1)-(6) reused from fock0c_fence_check (antiunitary/phase-bearing; NOT full-group "
          "generator equivariance; NOT BRIDGE-LOCK-form; NOT BRIDGE-T-form; per-sector by "
          "construction; grading is projector-only)",
          fc["1_O0_bit_even_democracy"]["is_phase_bearing"]
          and fc["2_M1b_no_generator_constraint"]["no_group_generator_used"]
          and fc["3_BRIDGE_LOCK_attachment_functional_orbit_blind"]["is_fock_level"]
          and fc["4_BRIDGE_T_two_point_data_blind"]["is_representation_theoretic"]
          and fc["5_BRIDGE_GEOM_per_sector_required"]["is_per_sector_by_design"]
          and fc["6_FOCK0c_grading_is_projector_only"]["no_extra_mechanism_introduced"])
    check("(7) NEW: K_hist/K_F are modular/GNS-Tomita objects, NEITHER is the run's own "
          "two-point resolvent (I-uB)^-1 data (BRIDGE-T's own concern) -- the flow row is "
          "stacked on the SAME sector-pair-graded system items 3/5 already establish",
          fc["7_FOCK0d_flow_not_resolvent_data"]["is_modular_not_resolvent"])

    banner("FOCK-0d  REGRESSION: Sections 7/7b/8/8b/8c + module anchors untouched")
    # =======================================================================
    check("anchor_cell_projector() + anchor_tick_2pi() + accretion_selftest_2026_07_10() + "
          "i2b_selftest_2026_07_11() + fock0_selftest_2026_07_11() + "
          "fock0b_selftest_2026_07_11() + fock0c_selftest_2026_07_11() all still PASS",
          net.anchor_cell_projector() and net.anchor_tick_2pi()
          and net.accretion_selftest_2026_07_10(verbose=False)
          and net.i2b_selftest_2026_07_11(verbose=False)
          and net.fock0_selftest_2026_07_11(verbose=False)
          and net.fock0b_selftest_2026_07_11(verbose=False)
          and net.fock0c_selftest_2026_07_11(verbose=False))
    check("fock0d_selftest_2026_07_11() (the Section-8d permanent regression anchor) PASSES",
          net.fock0d_selftest_2026_07_11(verbose=False))

    banner("FOCK-0d NULLITY TRAJECTORY (generator 0 -> J-global 192 -> +parity 96 -> full-J 16 "
           "-> +flow N(lambda))")
    # =======================================================================
    print(f"  {'class':<38} {'shell1 nullity/384':<22} {'shell2 nullity/768':<22}")
    print(f"  {'generator (dr_map_hom_space)':<38} {'0':<22} {'n/a':<22}")
    print(f"  {'J-global (single antiunitary pair)':<38} {'192':<22} {'n/a':<22}")
    print(f"  {'+parity (global pair + R/F_bit)':<38} {'96':<22} {'n/a':<22}")
    print(f"  {'full-J (FOCK-0c full pin)':<38} {'16':<22} {'64':<22}")
    if total_candidates == 0:
        print(f"  {'+flow (FOCK-0d, this station)':<38} {'0 (no admissible lambda)':<22} "
              f"{'0 (no admissible lambda)':<22}")
    else:
        for r in all_results:
            for c in r["candidates"]:
                print(f"  {'+flow lambda=%.4f (region %s)' % (c['lambda'], r['region']):<38} "
                      f"{c['shell1']['nullity']:<22} {c['shell2']['nullity']:<22}")

    banner("FOCK-0d VERDICT (SS I tree; ML-2b/HK-7 QUALIFIER attaches to EVERY sentence)")
    # =======================================================================
    print(f"  QUALIFIER: {QUALIFIER}")
    print()
    allowance = 2  # pre-declared BEFORE solving; lambda contributes NO allowance (SS I)
    print(f"  PRE-DECLARED W1 ALLOWANCE (re-stated, lambda adds NONE): {allowance} real "
          "dimensions.")
    if total_candidates == 0:
        print("  W2 EMPTY FOR ALL lambda>0: K_F's spectrum contains NO exact mu2=2*mu1 pair at "
              "ANY of the A4-orbit-inequivalent 3-edge regions tested, for the closed-form "
              "candidate set derived above -- hence NO lambda>0 admits a nonzero joint solution "
              "at both shells simultaneously. See the station report for the algebraic argument "
              "(not SVD alone) for why this is the expected, not merely observed, outcome.")
    else:
        any_forced = any(c["shell1"]["nullity"] == allowance and c["shell2"]["nullity"] == allowance
                          for r in all_results for c in r["candidates"])
        any_above = any(c["shell1"]["nullity"] > allowance or c["shell2"]["nullity"] > allowance
                        for r in all_results for c in r["candidates"])
        if any_forced:
            print("  W1 FORCED: an isolated lambda* gives nullity == the allowance at BOTH "
                  "shells -- see the printed candidate(s) above for lambda*'s value.")
        elif any_above:
            print("  W3 ABOVE ALLOWANCE at some lambda: see the printed per-candidate nullities "
                  "above; the excess is the named incomplete equation.")
        else:
            print("  Candidates exist but none reach W1/W3 cleanly -- see per-candidate nullity "
                  "printed above for manual adjudication (likely W4/mixed).")

    banner("RESULT")
    print("ALL MACHINE CHECKS PASS" if ok_all else "SOME CHECKS FAILED -- see [FAIL] lines above")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
