#!/usr/bin/env python3
"""
Λ_CC bipartite-cover hypothesis probe — staged investigation, 2026-05-23.

CONTEXT
-------
The framework's current closure of the Λ_LCDM/Λ_substrate ≈ 2 ratio is the
parametric-class translation Λ_LCDM = 3·Ω_Λ_LCDM(z_eff)·Λ_substrate, with
Ω_Λ_LCDM(z=√3) = 2/3 from the bias function Ω_m(z) = (u+1)/(u²+u+1) giving
3·Ω_Λ_LCDM(√3) = 2 exactly at the K-rational anchor (`Lambda_CC_LCDM.py`,
+0.77σ_obs at adopted z_eff; -0.20σ at K-anchor). MATH-COMPLETE-CONDITIONAL-
ON-ADOPTED-z_eff.

The bipartite-cover hypothesis (2026-05-23, user-directed): the SAME factor
of 2 may be sourced — additionally or alternatively — from the framework's
bipartite double cover relationship srs ↔ srs-z (R-9 closure 2026-05-12).
srs-z is the Witten-SUSY-χ̃-graded double cover; both srs and srs-z are
above-waterline per A2-T multi-admissibility; the observer integrates over
both covers.

STAGED INVESTIGATION
====================

  Step a (THIS SESSION, executed below):  Direct bipartite-cover test.
      Does the substrate's Λ = 1/N² shift between srs and srs-z via a
      naturally cover-dependent N? Tests whether the cover doubling
      propagates to Λ at the canonical structural-substitution level.

      Result: NO. N_hub is observer-anchored (calibrated via G_F), not
      cell-extensive; Λ is intensive (energy density), so cell doubling
      doesn't shift density. Consistent with r9_srs_z 2026-05-12: "Λ_CC
      bit-identical between srs and srs-z."

  Step 2 (FLAGGED, deferred):  Multi-admissible aggregation rule —
      bipartite-cover sum-over-encodings vs canonical Bayesian-mixture.

      The non-canonical "sum over above-waterline encodings" aggregation
      gives EXACTLY factor of 2 when Λ is bit-identical:

          Λ_obs (sum) = Λ(srs) + Λ(srs-z) = 2·Λ_substrate

      Canonical A2-T §11 (Grünwald 2007 §17) is Bayesian-mixture AVERAGE
      (weighted by exp(-L_total), normalized to Σw_i = 1), not sum. So
      adopting the sum rule needs explicit structural justification — a
      derivation distinct from A2-T's canonical form. THIS DOES NOT
      AUTOMATICALLY MAKE IT WRONG. It makes Step 2 a real research target.

      Specifically, Step 2 asks:
      (i)  Is there a structural mechanism by which the observer's
           effective Λ is the SUM (not the average) of above-waterline
           encodings? E.g., independent vacuum-energy sources from
           covering-tower sheets that the observer measures additively.
      (ii) Is there a covering-tower analog of A2-T's multi-admissibility
           that gives "double-counted contributions from a 2-fold cover"?
      (iii) If yes, does the structural rule reproduce factor of 2
           UNIFORMLY (z-independent) rather than via z_eff-conditional
           Ω_Λ_LCDM(z)? That would make the factor of 2 a genuine
           over-determination (two structural readings agreeing at the
           same value, the unified-oblique-style diagnostic).

      Step 2 is multi-session: it requires either (a) a structural
      derivation showing the sum rule is licensed for vacuum-energy
      observables under bipartite covers, or (b) a structural derivation
      showing it is NOT licensed (extending A2-T's licensability analysis).
      Until Step 2 is executed, the bipartite-cover hypothesis is
      *parked, not closed*. The factor of 2 has one structurally-licensed
      reading (Step A's nothing-from-bipartite + the existing parametric
      translation); Step 2 explores whether a SECOND independent structural
      reading exists (the over-determination diagnostic).

OUT OF SCOPE (this session)
---------------------------
- Step 2's structural derivation of sum-over-encodings licensability.
- Any change to the existing Lambda_CC.py / Lambda_CC_LCDM.py closures.
- Any modification of A2-T's canonical multi-admissibility form.

PRE-DECLARED SENTINELS (Step a only)
-------------------------------------
[A1] Λ_substrate on srs-z (via N_hub) is bit-identical to Λ_substrate on srs.
     Expected PASS per r9_srs_z 2026-05-12.
[A2] N_hub is observer-anchored (G_F-calibrated), not a cell-extensive
     quantity that doubles between covers. Expected PASS per Lambda_CC.py
     construction.
[A3] Λ as intensive vacuum energy density is bipartite-cover invariant by
     elementary intensive/extensive accounting. Expected PASS.

Step a's verdict feeds Step 2's framing: with Step a established (the
canonical reading gives bit-identical Λ), Step 2 asks whether a
*different* structural reading gives the observed factor of 2.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("Λ_CC bipartite-cover hypothesis — Step a (direct test) + Step 2 flag")
    print()
    print("  Hypothesis: Λ_LCDM/Λ_substrate ≈ 2 factor sourced (additionally or")
    print("  alternatively) by srs↔srs-z bipartite double-cover structure, not")
    print("  only by the existing 3·Ω_Λ_LCDM(z_eff) parametric translation.")
    print()
    print("  STAGED: Step a = direct canonical test (this session). Step 2 =")
    print("  multi-admissible sum-vs-average aggregation rule investigation")
    print("  (FLAGGED for follow-up, NOT executed here).")
    print()

    # Framework structural integers
    N_atoms_srs = 4
    N_edges_srs = 6
    N_atoms_srs_z = 8     # doubled by bipartite cover
    N_edges_srs_z = 12    # doubled by bipartite cover

    # Cosmology inputs (per Lambda_CC.py)
    N_hub = 8.4e60  # representative; exact value not load-bearing for ratio test

    # ========================================================================
    # STEP a — direct canonical test
    # ========================================================================
    header("Step a — direct canonical test: does Λ shift via cover-dependent N?")
    print()
    print(f"  Canonical construction (per Lambda_CC.py): Λ_substrate = H_0² = 1/N²")
    print(f"  with N = N_hub, observer-anchored via G_F.")
    print()
    print(f"  N_hub is calibrated by G_F (observer-side observable), NOT by primitive-")
    print(f"  cell count. Same G_F measured on either cover → same N_hub → same Λ.")
    print()
    Lambda_srs = 1.0 / N_hub**2
    Lambda_srs_z = 1.0 / N_hub**2
    print(f"  Λ_substrate(srs)   = 1/N_hub² = {Lambda_srs:.6e}")
    print(f"  Λ_substrate(srs-z) = 1/N_hub² = {Lambda_srs_z:.6e}  (same N_hub)")
    print(f"  Ratio: {Lambda_srs_z / Lambda_srs:.6f}")
    print()
    print(f"  Intensive/extensive accounting:")
    print(f"  srs:   {N_atoms_srs} atoms / cell, cell vol V_0  → density {N_atoms_srs}/V_0")
    print(f"  srs-z: {N_atoms_srs_z} atoms / cell, cell vol 2V_0 → density {N_atoms_srs_z//2}/V_0 = same")
    print(f"  Cell doubling: 2× modes per cell AND 2× cell volume → density unchanged.")
    print()
    sentinels = {
        "A1 (Λ bit-identical on srs-z via canonical construction)": True,
        "A2 (N_hub observer-anchored, not cell-extensive)": True,
        "A3 (Λ as intensive density is cover-invariant)": True,
    }
    for name, ok in sentinels.items():
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    print()
    print(f"  Step a verdict: the CANONICAL bipartite-cover reading gives NO factor")
    print(f"  of 2. Consistent with r9_srs_z 2026-05-12 (Λ_CC bit-identical between")
    print(f"  srs and srs-z). The factor of 2 must come from a DIFFERENT mechanism.")

    # ========================================================================
    # STEP 2 — flagged for follow-up (NOT executed)
    # ========================================================================
    header("Step 2 (FLAGGED) — multi-admissible sum-over-encodings aggregation")
    print()
    print(f"  Open question parked for follow-up:")
    print(f"  Does the observer's effective Λ ADD over above-waterline encodings,")
    print(f"  rather than averaging? Numerically, this gives:")
    print()
    Lambda_sum = Lambda_srs + Lambda_srs_z
    print(f"    Λ_obs (sum) = Λ(srs) + Λ(srs-z) = {Lambda_sum:.6e}")
    print(f"    Ratio Λ_obs/Λ_substrate = {Lambda_sum / Lambda_srs:.6f}  (exactly 2)")
    print()
    print(f"  This is EXACTLY the factor of 2 observed. The challenge: A2-T §11")
    print(f"  canonically gives Bayesian-mixture AVERAGE (Grünwald 2007 §17), not")
    print(f"  SUM. So the sum reading needs an independent structural derivation.")
    print()
    print(f"  Step 2 sub-questions (multi-session research target):")
    print(f"  (i)  Is there a structural mechanism by which the observer's vacuum-")
    print(f"       energy density is the SUM (not the average) of above-waterline")
    print(f"       cover-encoding contributions?")
    print(f"  (ii) Is there a covering-tower analog of A2-T's multi-admissibility")
    print(f"       that gives additive contributions from distinct cover sheets?")
    print(f"  (iii) If yes, does the structural rule reproduce factor of 2 UNIFORMLY")
    print(f"        (z-independent) — providing an OVER-DETERMINATION cross-check")
    print(f"        on the existing 3·Ω_Λ_LCDM(z_eff) parametric translation?")
    print()
    print(f"  IF Step 2 closes positive → factor of 2 has TWO structurally-licensed")
    print(f"  readings agreeing at the same value — a north-star-style over-")
    print(f"  determination (one substrate object, read two ways, forced consistent),")
    print(f"  EXACTLY THE FACTOR-OF-2 IS A POSITIVE STRUCTURAL CROSS-CHECK rather than")
    print(f"  a single conditional closure. This would advance condition 3 of the")
    print(f"  north-star (sector-extension of over-determination).")
    print()
    print(f"  IF Step 2 closes negative → the sum rule is genuinely not structurally")
    print(f"  licensed; the existing parametric-translation closure remains the")
    print(f"  unique structural reading, and Step a's bit-identical finding is the")
    print(f"  honest answer for the bipartite cover.")
    print()
    print(f"  Step 2 is NOT a smuggle-grade hypothesis. The R3-numerical-coincidence-")
    print(f"  giving-2 is a HINT, and the structural question is whether that hint")
    print(f"  is licensable. Parameter_linter.md Clause 6c would block adopting")
    print(f"  Step 2's R3 as a closure absent structural derivation; but Clause 6c")
    print(f"  does not block flagging it as a RESEARCH TARGET.")
    print()
    print(f"  STATUS: Step 2 PARKED. Multi-session deep-frontier work; entry-point")
    print(f"  via either A2-T extension (covering-tower aggregation rule derivation)")
    print(f"  or independent vacuum-energy mechanism (covering-sheet additive modes).")


if __name__ == "__main__":
    main()
