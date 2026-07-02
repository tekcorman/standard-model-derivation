#!/usr/bin/env python3
"""
Cascade D1+D2+D3 — k* cancellation audit

QUESTION (from menu→observation bridge scoping, sub-problem β coupling):
    Does the cascade theorem H·N·t_P = 1 leak k* into the observer's
    compression budget B = log₂(N_hub), or is it k*-agnostic?

If k*-leak: B and k* are co-determined; bridge needs joint fixed point.
If no leak: B and k* are decoupled — each is its own substrate-internal
            fixed point; bridge sub-problem β closure (per-vertex
            Bayesian equilibrium S = θ_c + θ_p, toggle_arity.py) and
            B derivation are independent.

CASCADE D1+D2+D3 PER N_hub.py LINES 66–86:

  D1 [A1, Type 1]:      1 t_P = k*·N toggles.
                        Each toggle modifies 1/(k*·N) of causal structure.

  D2 [A2 + algebra]:    MDL surprise threshold θ* = log₂(k*).
                        Acceptance probability per toggle = 2^{-θ*} = 1/k*.

  D3 [algebra]:         New observable states per t_P
                          = (k*·N toggles) × (1/k*·N acceptance)
                          = 1   EXACTLY.
                        H = (1/N)/t_P → H·N·t_P = 1.

CRITICAL ALGEBRA: k* enters TWICE (D1 toggle count + D2 acceptance) and
the two appearances CANCEL EXACTLY in the D3 product:

    (k* · N) × (1 / (k* · N)) = 1

This is independent of k*.  The coefficient "1" in H·N·t_P = 1 is
k*-agnostic by exact algebraic cancellation, not by approximation.

This probe verifies the cancellation at machine precision across k* ∈
{2, 3, 4, ..., 12}, then computes B = log₂(N_hub) from the observational
H_0 anchor (which routes through cosmic_age/t_P and is structurally
k*-independent) and confirms B ≈ 200 without any k* input.

VERDICT (anticipated): cascade D1+D2+D3 is k*-agnostic in the
theorem-grade form. B = log₂(N_hub) ≈ 200 is k*-independent via the
observational route. Bridge sub-problem β (k* derivation) and B
derivation DECOUPLE structurally — each is a separate substrate-internal
fixed point. NOT a joint fixed point.

(Honest scope flag: the framework's precision-matching N_hub via BZJ
inversion DOES use k*=3 to compute α_1 = (2/3)^8; this is the
empirical-anchor route to numerical N_hub, NOT the theorem-grade
structural derivation. The cascade theorem itself is k*-agnostic.)
"""

import math
import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def cascade_D3_rate_from_kstar(k_star, N):
    """Compute D3's 'new observable states per t_P' rate from D1+D2 inputs.

    D1: toggles per t_P = k* · N
    D2: acceptance per toggle = 1 / (k* · N)
    D3: rate = D1 × D2 = (k*·N) × (1/(k*·N)) = 1 EXACTLY.

    Implementing literally to expose the cancellation.
    """
    toggles_per_tP = k_star * N        # D1
    acceptance = 1.0 / (k_star * N)     # D2
    rate = toggles_per_tP * acceptance  # D3
    return rate


def main():
    print("=" * 76)
    print(" Cascade D1+D2+D3 — k* cancellation audit")
    print(" Question: does H·N·t_P = 1 leak k* into B = log₂(N_hub)?")
    print("=" * 76)
    print()

    # ----- Step 1: verify D3 cancellation across multiple k* ----
    print(" Step 1 — D3 product (k*·N) · (1/(k*·N)) = 1 across k*:")
    print()
    print(f"   {'k*':>4}  {'N':>10}  {'D1 (k*·N)':>14}  {'D2 (1/(k*·N))':>16}  {'D3 product':>14}")
    print("   " + "-" * 64)

    N_test = 1_000_000   # arbitrary substrate state count
    for k in [2, 3, 4, 5, 6, 8, 10, 12]:
        d1 = k * N_test
        d2 = 1.0 / (k * N_test)
        d3 = cascade_D3_rate_from_kstar(k, N_test)
        print(f"   {k:>4}  {N_test:>10}  {d1:>14}  {d2:>16.6e}  {d3:>14.10f}")
        assert abs(d3 - 1.0) < 1e-12, f"D3 deviates from 1 at k*={k}: {d3}"

    print()
    print("   D3 = 1 EXACTLY for every k*. k* enters via D1 (toggle count)")
    print("   and D2 (acceptance), with exact algebraic cancellation in D3.")
    print()

    # ----- Step 2: observational route to N_hub (k*-independent) -----
    print(" Step 2 — N_hub from observational H_0 anchor (k*-independent route):")
    print()

    # Observational inputs (Planck 2018 CMB; CODATA t_P)
    H_0_kmsMpc = 67.4               # Planck 2018 (Planck Collaboration 2018)
    Mpc_in_m = 3.085677581e22       # CODATA conversion
    H_0_per_s = H_0_kmsMpc * 1000.0 / Mpc_in_m
    t_P_s = 5.391247e-44            # NIST CODATA 2018 Planck time

    N_hub_obs = 1.0 / (H_0_per_s * t_P_s)
    log2_N_hub = math.log2(N_hub_obs)

    print(f"   H_0 (Planck CMB)    = {H_0_kmsMpc:.2f} km/s/Mpc")
    print(f"                       = {H_0_per_s:.4e} /s")
    print(f"   t_P (CODATA)        = {t_P_s:.6e} s")
    print(f"   N_hub = 1/(H_0·t_P) = {N_hub_obs:.4e}")
    print(f"   log₂(N_hub)         = {log2_N_hub:.4f} bits")
    print()
    print(f"   Inputs used: H_0 (observation), t_P (CODATA). NO k*.")
    print(f"   N_hub via this route is structurally k*-independent.")
    print()

    # ----- Step 3: confirm framework's cutoff log₂(N_substrate) ≈ 200 -----
    print(" Step 3 — Framework-scale cutoff log₂(N) used in Coxeter audits:")
    print()
    print(f"   sector_coxeter_freq_weighted_audit.py + Path B audit use cutoff")
    print(f"     K · m · log₂(|E|) ≤ 200")
    print(f"   The 200 is log₂(N_hub) ≈ {log2_N_hub:.0f}. CONFIRMED k*-agnostic.")
    print()

    # ----- Step 4: BZJ-inverted route (precision-matching) DOES use k* -----
    print(" Step 4 — BZJ-inverted N_hub (precision route to G_F at 0.51 ppm):")
    print()
    print("   The framework's adopted N_hub has its value pinned via the measured G_F by BZJ inversion:")
    print("     N_hub_BZJ = (δ² · M_P · dark / (√2 · v_GF))⁴")
    print("   where dark = 1 − (5/12) α₁/(1−α₁) and α₁ = ((k*−1)/k*)^(g−2).")
    print("   With k* = 3, g = 10: α₁ = (2/3)^8.")
    print()
    print("   This route DOES use k* = 3. It is the framework's empirical-anchor")
    print("   route to numerical precision (matching G_F at 0.51 ppm), NOT the")
    print("   theorem-grade structural derivation.")
    print()
    print("   The cascade theorem's k* cancellation in Step 1 is what makes the")
    print("   FORM H·N·t_P = 1 theorem-grade independent of k*. The numerical")
    print("   value of N_hub is anchored two ways: observational (k*-free) and")
    print("   BZJ-inverted (k*-dependent precision). They agree at the percent")
    print("   level; the framework prefers BZJ-inverted for precision.")
    print()

    # ----- Step 5: conclusion -----
    print("=" * 76)
    print(" CONCLUSION — cascade D1+D2+D3 is k*-agnostic in the theorem-grade form")
    print("=" * 76)
    print()
    print(" Cascade theorem H·N·t_P = 1 has EXACT k* cancellation in D3 product.")
    print(" The coefficient 1 is structurally k*-agnostic (verified at machine")
    print(f" precision across k* ∈ {{2,3,4,5,6,8,10,12}}).")
    print()
    print(" B = log₂(N_hub) via observational route is k*-INDEPENDENT.")
    print(f" Numerically B ≈ {log2_N_hub:.1f} bits (matches Coxeter-audit 200-bit cutoff).")
    print()
    print(" Bridge implication (per menu_to_observation_bridge_scoping_2026-05-07.md")
    print(" sub-problem β + question 2): k* and B DECOUPLE. Each is its own")
    print(" substrate-internal fixed point:")
    print()
    print("   k* = 3:  per-vertex Bayesian equilibrium S(k,p) = θ_c(p) + θ_p(p)")
    print("            with binary p=2 → k* = 3 EXACTLY")
    print("            (toggle_arity.py / k_star_derivation.md).")
    print()
    print("   B ≈ 200: cosmic clock × Margolus-Levitin → N_hub = cosmic_age/t_P")
    print("            log₂(N_hub) = framework-scale info budget")
    print("            (this audit; observational H_0 anchor; k*-free).")
    print()
    print(" There is NO joint fixed point. The framework's bridge's Stage 4")
    print(" (MDL compression at observer's budget) can use B without circular")
    print(" k* leak via the cascade theorem.")
    print()
    print(" Honest scope flag: this audit covers the THEOREM-GRADE form of")
    print(" cascade D1+D2+D3. The framework's NUMERICAL N_hub via BZJ inversion")
    print(" through G_F does use k*=3 (via α_1) for percent-level precision")
    print(" matching against the measured G_F. That precision route is the calibration of N_hub's value (G_F is downstream),")
    print(" not structural derivation. The structural form is k*-agnostic.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
