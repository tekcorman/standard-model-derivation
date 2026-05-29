#!/usr/bin/env python3
"""
R15_session_2_yukawa_resign_asymmetry.py
========================================

Session 2 of Route E (R-15 scoping doc 2026-05-14):
test whether the existing Yukawa-vertex construction has a Re-sign-asymmetric
structural feature that vanishes on the trivial-C_3 sector of V_Ram.

Background.  Session 1 confirmed the dim-4 trivial-C_3 sector of V_Ram carries
all four B-eigenvalues {+h, +h̄, -h, -h̄} (Re-sign balanced), while ω and ω²
sectors carry only +Re and only -Re eigenvalues respectively. Route E (E.ii)
hypothesises: the Yukawa vertex is constructed as a sum/trace over Bloch
eigenvalues that requires Re-sign asymmetry to be non-zero. If yes, the
Yukawa on trivial-C_3 vanishes by trace cancellation → m_D^(gen 1) = 0 → m_ν1 = 0.

This probe checks the existing y_τ derivation (theorem_ytau_corollary.md):

  y_τ = α_1_full × (1/k*) × (1/k*) × 1 × 1
        |--------|   |-----|   |-----|   ↑           ↑
        cycle amp    fermion   fermion   Higgs edge  Cl(0,2)
        (5/3)(2/3)^8  edge_in  edge_out  factor 1    channel 1
                    [§4]      [§5 L3]    [§6]       [§7 L13-14]

Each factor's eigenvalue / Re-sign dependence:
- α_1_full = (5/3) × (2/3)^8 = tan²(arg h) × NB_survival
  tan²(arg h) = Im²(h)/Re²(h) = 5/3   ← depends on |Re|², |Im|² (SIGN-BLIND)
  NB_survival = ((k-1)/k)^(g-2) = (2/3)^8  ← combinatorial, no h-dependence
- (1/k*) factors come from L3 "uniform MDL distribution over k* incident edges"
  — these are C_3-blind (uniform across edges)
- Higgs factor 1 (deterministic complement)
- Cl(0,2) factor 1 (per-process selection)

Question: does ANY factor in y_τ as currently derived depend on sign(Re(h))
or sign(Im(h))?  If no, y_τ is sign-blind ⇒ the same y_τ applies on
trivial-C_3, ω, and ω² sectors ⇒ Route E (E.ii) is UNSUPPORTED by the
existing Yukawa-vertex derivation.

Steps:
  Part A — evaluate y_τ at h, -h, h̄, -h̄, and verify all four give the SAME y_τ.
           This is the explicit C_3-sector-blindness test.
  Part B — flag what would need to change in the y_τ derivation for Route E
           to be supported. Specifically: identify a candidate Re-sign-sensitive
           feature that's MISSING from the existing derivation.
  Part C — note Need-D-3 (Y_u vs Y_d eigenbasis on C³_gen) as the open
           framework question whose closure is a prerequisite for any
           Yukawa-Re-sign-sensitive refinement of Route E.

Sentinel pass means the C_3-sector-blindness has been numerically verified;
this is a NEGATIVE outcome for Route E (E.ii) as stated.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Bloch eigenvalue at k_P (Ramanujan)
h_re = math.sqrt(3) / 2
h_im = math.sqrt(5) / 2
h = complex(h_re, h_im)

# Framework constants
k_star = 3
g_girth = 10


def alpha_1_bare(k, g):
    """NB walk survival ((k-1)/k)^(g-2)."""
    return ((k - 1) / k) ** (g - 2)


def alpha_1_full_from_h(h_value, k, g):
    """α_1_full = tan²(arg h) × α_1_bare, evaluated for given h.

    This is the EXISTING framework derivation of α_1_full per
    theorem_alpha_1_full.md and predictions/alpha_1_full.py.

    Note: tan²(arg h) = Im²(h) / Re²(h).
    """
    if h_value.real == 0:
        return float('inf')
    tan2_arg_h = (h_value.imag / h_value.real) ** 2
    return tan2_arg_h * alpha_1_bare(k, g)


def y_tau_formula(h_value, k, g):
    """y_τ = α_1_full(h) / k²  (existing framework derivation)."""
    return alpha_1_full_from_h(h_value, k, g) / (k * k)


def part_A_sign_blindness_test():
    print("=" * 100)
    print("PART A — Evaluate y_τ derivation on all four V_Ram eigenvalues")
    print("=" * 100)

    eigvals = {
        '+h'  : +h,
        '+h̄'  : complex(h_re, -h_im),   # h̄
        '-h'  : -h,
        '-h̄'  : complex(-h_re, h_im),   # -h̄
    }

    print(f"\n  h = (√3 + i√5)/2 = ({h_re:.6f} + {h_im:.6f}i)")
    print(f"  tan²(arg h) = Im²/Re² = ({h_im}/{h_re})² = {(h_im/h_re)**2:.6f} = 5/3 = {5/3:.6f}\n")

    print(f"  Eigenvalue       Re       Im       tan²(arg)     α_1_full       y_τ")
    print(f"  " + "-" * 86)
    results = {}
    for label, e in eigvals.items():
        tan2 = (e.imag / e.real) ** 2
        a1f = alpha_1_full_from_h(e, k_star, g_girth)
        y = y_tau_formula(e, k_star, g_girth)
        results[label] = y
        print(f"  {label:6s}        {e.real:+.4f}   {e.imag:+.4f}   {tan2:.6f}    {a1f:.10f}   {y:.10f}")

    # Check: all four give identical y_τ?
    values = list(results.values())
    all_equal = all(abs(v - values[0]) < 1e-15 for v in values)
    print()
    print(f"  All four y_τ values equal at machine precision: {all_equal}")
    print(f"  Spread (max - min): {max(values) - min(values):.2e}")
    assert all_equal, "Expected all four y_τ to be identical"

    return all_equal


def part_B_flag_missing_resign_dependence():
    print("\n" + "=" * 100)
    print("PART B — Identify the missing Re-sign-sensitive feature")
    print("=" * 100)
    print(r"""
  The existing y_τ derivation depends on the Bloch eigenvalue h only through
  tan²(arg h) = Im²(h) / Re²(h), which is INVARIANT under sign flips of either
  Re(h) or Im(h). All four V_Ram eigenvalues {±h, ±h̄} give the SAME y_τ.

  Consequence for Route E (E.ii):
    The existing Yukawa-vertex amplitude CANNOT distinguish trivial-C_3
    (Re-sign balanced) from ω (only +Re) from ω² (only -Re). The hypothesised
    "trace cancellation on trivial-C_3 makes Yukawa vanish" is UNSUPPORTED
    by the current derivation.

  For Route E (E.ii) to be supported, the Yukawa vertex would need to involve
  a NEW structural feature that is:
    (i)  sensitive to sign(Re(h))  — vanishes when ΣRe(h_eigval) = 0,
    (ii) present in the neutrino-Yukawa channel but NOT in the charged-lepton
         Yukawa channel (since charged leptons have y_e ≠ 0 for gen 1),
    (iii) consistent with α_21/α_31 derivations (which use h^g phase on ω, ω²).

  Such a feature is NOT in the existing y_τ derivation. The candidate
  framework mechanism would be:
    Y_u (up-type) vs Y_d (down-type) eigenbasis distinction on C³_gen,
    via H̃ = iσ_2 H* vs H Cl(0,2) channel coupling.
  This is the framework's open "Need-D-3" question
.""")


def part_C_session2_status():
    print("\n" + "=" * 100)
    print("PART C — Session 2 status")
    print("=" * 100)
    print(r"""
  ROUTE E (E.ii) verdict: NEGATIVE as stated.

  The existing Yukawa-vertex construction (y_τ chain, theorem_ytau_corollary.md)
  is sign-blind in (sign(Re(h)), sign(Im(h))). It cannot mechanically produce a
  vanishing Yukawa on the Re-sign-balanced trivial-C_3 sector.

  CONSEQUENCES for the R-15 scoping doc:
    • Route E (E.i) closed POSITIVELY-WITH-RESIDUAL (Session 1).
    • Route E (E.ii) requires a NEW Yukawa-vertex feature, BLOCKED on Need-D-3
      closure (multi-session framework research, per memory 2026-05-09).
    • Route E AS A WHOLE is NOT closable in the bounded 3-session scope
      anticipated in the scoping doc.

  FALLBACKS per R-15 scoping doc §2:
    • Route A (M_D trivial-C_3 trace cancellation, 2 sessions) — still untested.
      The Re-sign-cancellation mechanism is the same as Route E (E.ii) — it
      requires the SAME structural feature in M_D that is missing from y_τ.
      Likely fails for the same reason. Untested but low priority.
    • Route D (ν_R decouples from trivial-C_3, 2 sessions) — different
      mechanism (decoupling rather than vertex-vanishing); has structural
      conflict with α_21 Step 1 ("ν_R lives on the C_3-trivial Bloch direction
      at the P-point for scale"). Needs disambiguation of "Bloch direction"
      vs "C_3 generation sector". HIGHER-PRIORITY fallback.

  R-15 STATUS:
    OPEN. Route E (E.i) confirmed structurally (Session 1). Route E (E.ii)
    BLOCKED on Need-D-3. Route D remains untested and the most promising
    bounded route, but requires reading α_21 Step 1 with care to avoid
    contradicting the existing m_ν3 scale-setting mechanism.""")


def main():
    print(r"""
==========================================================================================
R-15 ROUTE E — SESSION 2 — Yukawa vertex Re-sign-asymmetry check
==========================================================================================""")
    all_equal = part_A_sign_blindness_test()
    part_B_flag_missing_resign_dependence()
    part_C_session2_status()

    print("\n" + "=" * 100)
    print("SENTINEL VERDICT")
    print("=" * 100)
    print(r"""
  Session 2 closes NEGATIVELY on Route E (E.ii) as stated.

  The existing Yukawa-vertex derivation y_τ = (tan²(arg h)) × (2/3)^8 / k*²
  is sign-blind in (sign(Re(h)), sign(Im(h))) at machine precision. The
  hypothesised "Yukawa vertex requires Re-sign asymmetry" structural feature
  is ABSENT from the current y_τ derivation.

  Route E (E.ii) BLOCKED on Need-D-3 (Y_u vs Y_d eigenbasis on C³_gen, via
  H̃ vs H Cl(0,2) channel) — multi-session framework research.

  R-15 STATUS:
    Route E (E.i) closed POSITIVELY-WITH-RESIDUAL (Session 1).
    Route E (E.ii) BLOCKED on Need-D-3.
    Route E as a whole NOT closable in bounded 3-session scope.
    Route D (ν_R decoupling) is the recommended next bounded fallback,
    after disambiguating α_21 Step 1's "Bloch direction" vs "C_3 generation sector".

  Sentinel pass.""")


if __name__ == "__main__":
    main()
