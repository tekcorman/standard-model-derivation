#!/usr/bin/env python3
"""
W8 — Exponent principle internal consistency across y_τ, y_ν, y_t
==================================================================

Date: 2026-05-20
Question: Does the stated "Yukawa exponent principle"
    y_X = prefactor × (2/3)^(n_free·(g-2)) / k^(edge selections)
(per srs_tan_beta.py PART 1 + commit 66c8836) actually fit all 3 derived
Yukawa channels with a CONSISTENT (n_free, edge_sel, prefactor) assignment?
Or is it a post-hoc unification of separately-motivated derivations?

PRE-DECLARED CHANNELS + ACTUAL FRAMEWORK FORMS (from source files)
==================================================================
1. y_τ (predictions/y_tau.py + theorem_ytau_corollary.md):
     y_τ = α₁_full / k² = (5/3)(2/3)⁸ / 9 ≈ 0.007226
     Match: +0.13% vs y_τ_obs = m_τ_pole/v ≈ 0.007217

2. y_ν Dirac (srs_neutrino_mass_scale.py PART 3, the ACTUAL computation):
     y_ν = (k-1)/k · √(L_us/k)
         = (2/3) · √((2+√3)/3)
         ≈ 0.7436
     Note: the file's docstring identification "y_ν = α₁/k for delocalized
     states" gives ~0.0217 which the file EXPLICITLY annotates as
     "3 orders of magnitude too small". The actual computation uses
     L_us = 2 + √3 (spectral radius of srs Laplacian), NOT α₁.

3. y_t (commit 66c8836):
     y_t = 1 → m_t(tree) = v/√2 ≈ 174.10 GeV (+0.82%)
     Mechanism: "gen-3 limit n_free → 0, exponent → 1"

PRE-DECLARED VERDICT GATES
==========================
Apply the stated formula y_X = prefactor × (2/3)^(n_free·(g-2)) / k^(edge_sel)
with various (n_free, edge_sel, prefactor) assignments and check whether ALL
3 channels can be fit consistently:

- PASS: all 3 channels fit with the SAME prefactor (universal) at <5% precision
- PARTIAL: 2 of 3 channels fit consistently
- FAIL: the formula CANNOT cover all 3 channels with a single consistent reading

Pre-declared expectation: FAIL because y_ν's actual form has √(L_us/k), not
α₁'s (5/3)(2/3)⁸ — these are different STRUCTURAL OBJECTS, not different
(n_free, edge_sel) values within one formula.

USAGE
=====
    python3 proofs/foundations/W8_exponent_principle_consistency_2026-05-20.py
"""

from __future__ import annotations
import math
from fractions import Fraction

# Framework constants
K_STAR = 3
G_GIRTH = 10
ALPHA_1_BARE = Fraction(2, 3) ** 8       # = 256/6561
ALPHA_1_FULL = Fraction(5, 3) * ALPHA_1_BARE  # = 1280/19683
ALPHA_1_FULL_F = float(ALPHA_1_FULL)

L_US = 2 + math.sqrt(3)                  # srs Laplacian spectral radius (used in y_ν)


# ===========================================================================
# 1. The stated exponent principle formula
# ===========================================================================
def exponent_principle(prefactor: float, n_free: int, edge_sel: int) -> float:
    """y_X = prefactor × (2/3)^(n_free · (g-2)) / k^(edge_sel)"""
    return prefactor * (2/3)**(n_free * (G_GIRTH - 2)) / K_STAR**edge_sel


# ===========================================================================
# 2. Actual framework values per channel (from source files; not formula)
# ===========================================================================
# y_τ from theorem_ytau_corollary.md
y_tau_actual = ALPHA_1_FULL_F / K_STAR**2     # = (5/3)(2/3)^8 / 9 ≈ 0.007226

# y_ν from srs_neutrino_mass_scale.py PART 3 (actual computation, not docstring)
y_nu_actual = (K_STAR - 1)/K_STAR * math.sqrt(L_US / K_STAR)  # ≈ 0.7436

# y_t from commit 66c8836
y_t_actual = 1.0


# ===========================================================================
# 3. Pre-declared candidate (n_free, edge_sel, prefactor) assignments per channel
# ===========================================================================
# These are the values STATED in srs_tan_beta.py PART 1 + commit 66c8836
# and the docstring of srs_neutrino_mass_scale.py (the "α₁/k" identification)
candidate_assignments = {
    "y_τ":  {"n_free": 1, "edge_sel": 2, "prefactor_stated": 5/3,  "actual": y_tau_actual},
    "y_ν":  {"n_free": 1, "edge_sel": 1, "prefactor_stated": 5/3,  "actual": y_nu_actual},
    "y_t":  {"n_free": 0, "edge_sel": 0, "prefactor_stated": 1.0,  "actual": y_t_actual},
}

print("=" * 76)
print("W8 — Exponent principle internal consistency across y_τ, y_ν, y_t")
print("=" * 76)
print()
print(f"Stated formula: y_X = prefactor × (2/3)^(n_free·(g-2)) / k^(edge_sel)")
print(f"  with k = {K_STAR}, g-2 = {G_GIRTH-2}")
print()
print(f"  L_us = 2 + √3 ≈ {L_US:.6f} (srs Laplacian spectral radius, used in y_ν)")
print()


# ===========================================================================
# 4. Test each channel with STATED assignment
# ===========================================================================
print("=" * 76)
print("Test: predicted-by-formula vs actual-by-framework-computation")
print("=" * 76)
print()
print(f"  {'channel':<7}  {'n_free':>7}  {'edge_sel':>9}  {'prefactor':>11}  "
      f"{'formula gives':>15}  {'actual value':>14}  {'match?':<10}")
print(f"  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*11}  {'-'*15}  {'-'*14}  {'-'*10}")

results = {}
for ch, ass in candidate_assignments.items():
    predicted = exponent_principle(ass["prefactor_stated"], ass["n_free"], ass["edge_sel"])
    actual = ass["actual"]
    rel_err = abs(predicted - actual) / abs(actual) * 100
    match = "PASS" if rel_err < 5 else "FAIL"
    results[ch] = {"predicted": predicted, "actual": actual, "rel_err": rel_err, "match": match}
    print(f"  {ch:<7}  {ass['n_free']:>7}  {ass['edge_sel']:>9}  "
          f"{ass['prefactor_stated']:>11.5f}  {predicted:>15.6e}  "
          f"{actual:>14.6e}  {match:<10} ({rel_err:+.2f}%)")
print()


# ===========================================================================
# 5. Diagnose y_ν specifically: is α₁/k vs actual a 3-orders-of-magnitude gap?
# ===========================================================================
print("=" * 76)
print("y_ν diagnosis — α₁/k 'docstring identification' vs actual computation")
print("=" * 76)
print()
y_nu_docstring = ALPHA_1_FULL_F / K_STAR        # the docstring/commit-66c8836 reading
y_nu_actual_again = (K_STAR - 1)/K_STAR * math.sqrt(L_US / K_STAR)  # PART 3 actual

print(f"  Docstring identification (commit 66c8836 + srs_neutrino_mass_scale "
      f"L31): y_ν = α₁/k")
print(f"    y_ν_docstring = {ALPHA_1_FULL_F:.6f} / {K_STAR} = {y_nu_docstring:.6e}")
print()
print(f"  Actual computation (srs_neutrino_mass_scale L223): "
      f"y_ν = (k-1)/k · √(L_us/k)")
print(f"    y_ν_actual = (2/3) · √({L_US:.6f}/{K_STAR}) = {y_nu_actual_again:.6e}")
print()
print(f"  Ratio y_ν_actual / y_ν_docstring = {y_nu_actual_again / y_nu_docstring:.2f}")
print(f"  Log10 ratio = {math.log10(y_nu_actual_again / y_nu_docstring):+.2f}")
print()
print("  The framework's own annotation: \"Compare: alpha_1/k (naive) =")
print("  {0:.6e} (3 orders of magnitude too small)\".".format(y_nu_docstring))
print()
print("  ⟹ The docstring's α₁/k 'one less edge resolution' identification")
print("    is NOT the formula the framework actually uses for y_ν.")
print("    y_ν is structurally different — uses spectral radius L_us, not α₁.")
print()


# ===========================================================================
# 6. Could y_ν be salvaged by a different (n_free, edge_sel) within the formula?
# ===========================================================================
print("=" * 76)
print("Salvage attempt — can (n_free, edge_sel) make the formula give y_ν?")
print("=" * 76)
print()
print(f"  Target: y_ν_actual ≈ {y_nu_actual:.6f}")
print()
print(f"  If prefactor = 5/3 (universal across τ, ν), find (n_free, edge_sel):")
print()
print(f"  {'n_free':>7}  {'edge_sel':>9}  {'formula gives':>15}  {'rel err':<12}")
print(f"  {'-'*7}  {'-'*9}  {'-'*15}  {'-'*12}")
for n_f in range(0, 4):
    for e_s in range(0, 4):
        pred = exponent_principle(5/3, n_f, e_s)
        rel = (pred - y_nu_actual) / y_nu_actual * 100
        ok = "*" if abs(rel) < 5 else ""
        print(f"  {n_f:>7}  {e_s:>9}  {pred:>15.6e}  {rel:>+10.2f}% {ok}")
print()


# ===========================================================================
# 7. Could y_t's prefactor = 1 be consistent with y_τ's prefactor = 5/3?
# ===========================================================================
print("=" * 76)
print("y_t prefactor consistency — does prefactor depend on (n_free, edge_sel)?")
print("=" * 76)
print()
print(f"  y_τ assigned (n_free=1, edge_sel=2, prefactor=5/3) → {y_tau_actual:.6f} ✓")
print(f"  y_t assigned (n_free=0, edge_sel=0, prefactor=1.0)  → {y_t_actual:.6f} ✓")
print()
print(f"  Why does y_t's prefactor = 1 instead of 5/3?")
print()
print(f"  If y_t's prefactor were 5/3 (same as y_τ):")
y_t_with_53 = exponent_principle(5/3, 0, 0)
print(f"    formula would give: {y_t_with_53:.6f} (= 5/3) — NOT 1")
print(f"    inconsistent with the stated y_t = 1 result")
print()
print(f"  The framework's stated y_t = 1 derivation REQUIRES dropping the 5/3")
print(f"  prefactor at gen-3 limit. But the master template formula does NOT")
print(f"  encode this prefactor-dependence on n_free / edge_sel — it's an")
print(f"  EXTRA assumption (or absorbed into 'prefactor' by hand).")
print()
print(f"  ⟹ The y_t derivation requires an EXTRA assumption beyond the stated")
print(f"    exponent-principle formula: that the 5/3 chirality factor")
print(f"    'turns off' at the gen-3 limit. This is asserted, not derived.")
print()


# ===========================================================================
# 8. Verdict
# ===========================================================================
print("=" * 76)
print("W8 VERDICT")
print("=" * 76)
print()
print(f"  Channel matches under STATED assignments:")
for ch, r in results.items():
    print(f"    {ch}: predicted {r['predicted']:.6e}, actual {r['actual']:.6e},  {r['match']} ({r['rel_err']:+.2f}%)")
print()
print(f"  Universality of prefactor across channels: NO")
print(f"    y_τ requires prefactor = 5/3 (chirality factor)")
print(f"    y_ν actual form has NO α₁ at all — uses √(L_us/k); 3 orders of magnitude off the α₁/k identification")
print(f"    y_t requires prefactor = 1 (no chirality factor); inconsistent with y_τ's 5/3")
print()
print(f"  ⟹ FAIL: the exponent principle as a UNIFIED master formula is")
print(f"    POST-HOC UNIFICATION, not a derived master mechanism.")
print()
print(f"    What the framework actually has:")
print(f"      (1) y_τ via 4-factor template (theorem_ytau_corollary.md) — rigorous structural derivation")
print(f"      (2) y_ν via spectral seesaw (srs_neutrino_mass_scale.py PART 3) — DIFFERENT structural object (uses L_us, not α₁)")
print(f"      (3) y_t = 1 from 'gen-3 limit assertion' — asserted, not derived")
print()
print(f"    The three derivations share some structural motifs (k-factors,")
print(f"    (2/3)-factors) but they do not unify under a single (prefactor,")
print(f"    n_free, edge_sel) template. The 'exponent principle' formulation")
print(f"    in srs_tan_beta.py PART 1 + commit 66c8836 is a SUGGESTIVE PATTERN")
print(f"    that doesn't carry rigor across the 3 channels.")
print()
print(f"    Implication: the 'Yukawa master theory' as a single template")
print(f"    DOES NOT YET EXIST in the framework. What exists is three")
print(f"    separately-motivated derivations with shared structural motifs.")
print(f"    A genuine master theory requires either:")
print(f"      (α) A unified formula that derives y_τ, y_ν, y_t with consistent")
print(f"          (prefactor, n_free, edge_sel) assignments — currently absent.")
print(f"      (β) A higher-level mechanism that explains why these three")
print(f"          derivations are structurally related (e.g., A2-T waterline")
print(f"          retention applied per-sector with explicit Cl(6)-Fock content).")
print(f"      (γ) Honest acknowledgment that the 3 channels are derived")
print(f"          separately and the 'master theory' is aspirational.")
print()
print("=" * 76)
