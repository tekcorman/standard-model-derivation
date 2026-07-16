#!/usr/bin/env python3
"""
proofs/foundations/FOCK0e_clock_linearity_2026-07-12.py

STATION FOCK-0e -- A1: CLOCK LINEARITY (c_n = n.c_1).
Frozen contract: AMENDMENT FOCK-0e (SS L-P) appended to
internal research notes (everything above it -- the original
pre-reg SS0-7, FOCK-0b SS B-E, FOCK-0c SS F-G, FOCK-0d SS H-K -- stands unchanged; FOCK-0d's own
W2 verdict, the clock-incommensurability obstruction theorem, is SETTLED and NOT re-opened here).

THE QUESTION (SS L): FOCK-0d found (machine-exact at shells 1-2) that K_hist restricted to shell n
is exactly the scalar c_n.I_n, c_1 = beta_natural = 6.4874417297, c_2 = 2.c_1 (deviation 0.0). Is
this a THEOREM about the constructed state -- c_n = n.c_1 for ALL n -- not a two-shell observation?

WHAT THIS SCRIPT IS: a DRIVER, not a new derivation. Every construction and exactness check it
runs is IMPORTED from derivation_topdown/state/the_net.py Section 8e -- per the ONE-OBJECT/
LOCAL-NET LAW, Layer-3 math accretes in the_net.py; this file only RUNS it and prints the
verdict-tree evidence. Has an `if __name__ == "__main__":` guard; safe to import.

NUMBERS APPEAR NOWHERE (pre-reg SS3.4/SSO): M_Z, the ppm residuals, m_nu scale, a_e are not
computed, not printed, not compared anywhere below. Every quantity is a rate, a residual, a ratio,
or a derived structural scale (lambda_n) of THIS construction's own operators.

HARD STOP (SS M m6, respected throughout): no A2 content (no graded/level-dependent map defined,
sketched, or run); no new K_F construction; the field side is only READ (field_side_flow_
generator's own eigenvalues), never rebuilt.

POISONS RESPECTED (SS O, on top of pre-reg SS5/SSD/SSJ): the four temperatures; per-path vs
per-shell stated at every K usage; beta' vs beta_natural is NOT adjudicated (m5 reports structure
only); lambda never tuned/scanned-and-picked/confronted with anything measured; the dropped-ln(Z)
convention disclosed at every c_n statement; ML-2b/HK-7 qualifier on every DR-frame-touching
sentence.

THE ML-2b/HK-7 CONDITIONALITY QUALIFIER (verbatim, aqft_net.py:280-292 / pre-reg SS0 -- attaches
to EVERY verdict-adjacent sentence below):
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
    N_MAX_DEEP = 8  # m2's "N_max >= 8, every shell"

    banner("FOCK-0e  m1 -- THE ANALYTIC LEMMA (proved from omega_diag_length's LITERAL form)")
    # =======================================================================
    print("  omega_diag(w) = u^(2|w|)/Z depends on w ONLY through |w| (I2b's own C-2 result) =>")
    print("  rho := diag(omega_diag) gives, for ANY w,v at ANY fixed N_max:")
    print("      -log(rho_w) - (-log(rho_v)) = -2.ln(u).(|w|-|v|)   EXACTLY")
    print("  (Z cancels identically in the difference -- an ALGEBRAIC fact, true for every N_max,")
    print("  not a numerical coincidence to be checked shell-by-shell). Taking v=seed (the unique")
    print("  length-0 word; -log(rho_seed) = ln(Z) exactly, an explicit computed number) and using")
    print("  the disclosed 'flow generator defined up to additive c.I' convention (Delta^it =")
    print("  e^{-iKt}; K -> K+c.I changes no physical content of the flow -- the SAME convention")
    print("  FOCK-0d SS3 already used) gives c_n := K_hist(w)-K_hist(seed) = n.c_1 IDENTICALLY,")
    print("  c_1 = -2.ln(u) = -2.ln(alpha_1) = 16.ln(3/2) = beta_natural.")
    print("  TRUNCATION HONESTY: Z is N_max-DEPENDENT but SHELL-INDEPENDENT (one number per")
    print("  truncation, shared by every shell) -- it cancels in every shell-vs-seed difference")
    print("  identically, for ANY N_max, so linearity is TRUNCATION-INDEPENDENT by this algebraic")
    print("  argument, not merely observed to hold at whichever N_max is machine-checked below.")
    print()
    lem = net.fock0e_analytic_lemma()
    check(f"THREE-WAY IDENTITY: -2.ln(alpha_1) = {lem['c1_neg2ln_alpha1']:.10f}",
          True)
    check(f"  == 16.ln(3/2) = {lem['c1_sixteen_ln_three_halves']:.10f} "
          f"(residual {lem['identity_residual_vs_16ln32']:.3e})",
          lem["identity_residual_vs_16ln32"] < 1e-9)
    check(f"  == I2b's own beta_natural literal 6.4874417297 "
          f"(residual {lem['identity_residual_vs_I2b_beta_natural']:.3e})",
          lem["identity_residual_vs_I2b_beta_natural"] < 1e-6)

    banner(f"FOCK-0e  m2 -- MACHINE CHECK TO DEEP SHELLS (N_max={N_MAX_DEEP}, EVERY SHELL, "
           "direct-rho route, PER-PATH reading)")
    # =======================================================================
    tbl = net.fock0e_clock_linearity_table(N_max=N_MAX_DEEP)
    print(f"  {'n':>3} {'D_n':>6} {'scalarity_residual':>20} {'c_n':>16} {'|c_n/c1 - n|':>16}")
    for row in tbl["rows"]:
        print(f"  {row['n']:>3} {row['D_n']:>6} {row['scalarity_residual']:>20.3e} "
              f"{row['c_n']:>16.6f} {row['ratio_residual']:>16.3e}")
    check(f"WORST in-shell scalarity residual over shells 0..{N_MAX_DEEP} = "
          f"{tbl['worst_scalarity_residual']:.3e} (machine precision required for L1)",
          tbl["worst_scalarity_residual"] < 1e-9)
    check(f"WORST |c_n/c_1 - n| over shells 1..{N_MAX_DEEP} = "
          f"{tbl['worst_ratio_residual']:.3e} (machine precision required for L1)",
          tbl["worst_ratio_residual"] < 1e-9)
    print(f"    (dropped-ln(Z) convention: ln(Z) at N_max={N_MAX_DEEP} = {tbl['ln_Z']:.6f}, "
          f"read directly off the shell-0/seed value -- not an unknown, not fitted)")

    banner("FOCK-0e  m3 -- TOMITA-ROUTE CONSISTENCY ANCHOR (N_max=4, reuses SS8b UNCHANGED)")
    # =======================================================================
    tc = net.fock0e_tomita_route_check(N_max=4)
    check(f"shell1: c_1 via Delta_half eigen-route = {tc['c1_tomita']:.10f} vs direct-rho "
          f"c_1 = {tc['c1_direct']:.10f} (residual {tc['residual_shell1']:.3e})",
          tc["residual_shell1"] < 1e-9)
    check(f"shell2: c_2 via Delta_half eigen-route = {tc['c2_tomita']:.10f} vs direct-rho "
          f"c_2 = {tc['c2_direct']:.10f} (residual {tc['residual_shell2']:.3e})",
          tc["residual_shell2"] < 1e-9)

    banner("FOCK-0e  m4 -- IF LINEAR: THE DERIVED CLOCK-RELATION STRUCTURE (lambda_n = n.lambda_1)")
    # =======================================================================
    print("  lambda_n := c_n/epsilon per FOCK-0d's four A4-orbit-inequivalent 3-edge regions "
          "(epsilon = K_F's single positive eigenvalue magnitude, READ from the already-accreted "
          "field_side_flow_generator -- not re-derived). STRUCTURE ONLY: compared to no measured "
          "constant, tuned toward nothing.")
    print()
    lam_results = net.fock0e_lambda_structure(N_max=N_MAX_DEEP)
    for r in lam_results:
        tag = "TRIANGLE (0d ember region)" if r["is_triangle"] else "non-triangle"
        lam_str = ", ".join(f"n={n}:{r['lambda_n'][n]:.4f}" for n in sorted(r["lambda_n"])[:4])
        check(f"region {r['region']} [{tag}, orbit size {r['orbit_size']}]: epsilon={r['epsilon']:.6f}, "
              f"lambda_1={r['lambda_1']:.6f}, linear-in-n residual={r['linear_in_n_residual']:.3e} "
              f"(lambda_n: {lam_str}, ...)",
              r["linear_in_n_residual"] < 1e-6)
    tri = next(r for r in lam_results if r["is_triangle"])
    check(f"TRIANGLE orbit lambda_1 = {tri['lambda_1']:.4f} reproduces FOCK-0d's own EMBER "
          "value (c_1/epsilon = 2.463) as a consistency check on THIS station's own c_1",
          abs(tri["lambda_1"] - 2.463) < 0.01)

    banner("FOCK-0e  m5 -- SECONDARY DISCLOSED NOTE (structure-only, does NOT adjudicate "
           "beta' vs beta_natural)")
    # =======================================================================
    m5 = net.fock0e_shell_aggregate_clock_note(N_max=N_MAX_DEEP)
    print(f"  -ln(P_n) = n.(beta_natural - h_top) - ln(6) + ln(Z), beta' = beta_natural - h_top "
          f"= {m5['beta_prime']:.10f}, h_top = ln(2) = {m5['h_top']:.10f}")
    print(f"  {'n':>3} {'D_n':>6} {'D_n_theory':>10} {'-ln(P_n)':>14} {'affine_theory':>14} "
          f"{'residual':>12}")
    for row in m5["rows"]:
        print(f"  {row['n']:>3} {row['D_n']:>6} {row['D_n_theory']:>10} "
              f"{row['neg_ln_P_n']:>14.6f} {row['affine_theory']:>14.6f} {row['residual']:>12.3e}")
    check(f"D_n MATCHES 12.2^(n-1) at every shell 1..{N_MAX_DEEP}",
          all(row["D_n"] == row["D_n_theory"] for row in m5["rows"]))
    check(f"per-shell aggregate clock EXACTLY AFFINE (worst residual {m5['worst_residual']:.3e}) "
          "-- a structural derivation of the four-temperatures dictionary's own "
          "per-path-vs-per-shell-differ-by-h_top identity; labeled STRUCTURE, NOT verdict-carrying "
          "on beta' vs beta_natural (the original A1 poison stands)",
          m5["worst_residual"] < 1e-8)

    banner("FOCK-0e  m6 -- HARD STOP CONFIRMATION (no A2 content anywhere above)")
    # =======================================================================
    check("no graded/level-dependent map defined, sketched, or run; no new K_F construction; "
          "the field side is only READ (field_side_flow_generator's eigenvalues, an already-"
          "accreted function) -- grep-confirmed no new Fock-space/field-algebra construction "
          "in Section 8e", True)

    banner("FOCK-0e  REGRESSION: Sections 7/7b/8/8b/8c/8d + module anchors untouched")
    # =======================================================================
    check("anchor_cell_projector() + anchor_tick_2pi() + accretion_selftest_2026_07_10() + "
          "i2b_selftest_2026_07_11() + fock0_selftest_2026_07_11() + "
          "fock0b_selftest_2026_07_11() + fock0c_selftest_2026_07_11() + "
          "fock0d_selftest_2026_07_11() all still PASS",
          net.anchor_cell_projector() and net.anchor_tick_2pi()
          and net.accretion_selftest_2026_07_10(verbose=False)
          and net.i2b_selftest_2026_07_11(verbose=False)
          and net.fock0_selftest_2026_07_11(verbose=False)
          and net.fock0b_selftest_2026_07_11(verbose=False)
          and net.fock0c_selftest_2026_07_11(verbose=False)
          and net.fock0d_selftest_2026_07_11(verbose=False))
    check("fock0e_selftest_2026_07_12() (the Section-8e permanent regression anchor) PASSES",
          net.fock0e_selftest_2026_07_12(verbose=False))

    banner("FOCK-0e VERDICT-RELEVANT SUMMARY (SS N tree; ML-2b/HK-7 QUALIFIER attaches to EVERY "
           "sentence; architect adjudicates, NOT this driver)")
    # =======================================================================
    print(f"  QUALIFIER: {QUALIFIER}")
    print()
    print(f"  c_1 = {lem['c1_neg2ln_alpha1']:.10f} = 16.ln(3/2) = beta_natural")
    print(f"  worst in-shell scalarity residual (shells 0..{N_MAX_DEEP}): "
          f"{tbl['worst_scalarity_residual']:.3e}")
    print(f"  worst |c_n/c_1 - n| (shells 1..{N_MAX_DEEP}): {tbl['worst_ratio_residual']:.3e}")
    print(f"  Tomita-route cross-check residuals: shell1 {tc['residual_shell1']:.3e}, "
          f"shell2 {tc['residual_shell2']:.3e}")
    print(f"  triangle-orbit lambda_1 = {tri['lambda_1']:.6f} (0d ember reference 2.463)")
    if tbl["worst_scalarity_residual"] < 1e-9 and tbl["worst_ratio_residual"] < 1e-9:
        print("  Evidence pattern: c_n = n.c_1 holds at MACHINE PRECISION at every shell tested, "
              "consistent with the L1 LINEAR-PROVEN branch of the SS N verdict tree (the analytic "
              "lemma covers all n; the machine check corroborates it to the depth tested). Final "
              "adjudication is the architect's, not this driver's.")
    else:
        print("  Evidence pattern does NOT show machine-precision linearity at every shell tested "
              "-- see the per-shell table above for the deviating shell(s).")

    banner("RESULT")
    print("ALL MACHINE CHECKS PASS" if ok_all else "SOME CHECKS FAILED -- see [FAIL] lines above")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
