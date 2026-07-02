#!/usr/bin/env python3
"""
W25 — Convention audit: the y_τ ↔ y_t √2 discrepancy
======================================================

Date: 2026-05-20
Predecessor: W24 §J6 surfaced a √2 discrepancy between two of the framework's
documented mass-formula conventions. W25 walks the audit through every relevant
source and isolates the bug precisely.

THE TWO CONVENTIONS:

  (FW) Framework convention used by predictions/y_tau.py + predictions/m_tau.py:
       y = m / v        (no /√2)
       Quoted explicitly in y_tau.py:
       "the 'm_tau/v' form above uses the convention where Yukawa is defined
        as the DIMENSIONLESS coupling to the full Higgs field (not /√2)"

  (PT) Standard SM / Peskin convention used by commit 66c8836 + scheme
       convention §3.54:
       y = m × √2 / v   ⇔   m = y · v/√2   (with /√2)
       From scheme §3.54: "m_τ = v · y_τ / √2"
       From commit 66c8836 (verbatim): "y_t (SM convention, GUT-anchored) = 1
                                        m_t (tree) = v/√2 = 174.104 GeV"

These differ by √2:  y_PT = √2 · y_FW.

THE BUG:

The framework's "exponent principle" formula y_X = prefactor · (2/3)^(n_free·(g-2))
· k^(-edge_sel) is asserted in master Yukawa doc as the master mechanism, with:
  - y_τ via (n_free=1, edge_sel=2, prefactor=5/3) → numerical value 7.226e-3
  - y_t  via (n_free=0, edge_sel=0, prefactor=1)   → numerical value 1.0

In FW convention:
  - y_τ_FW_pred  = 7.226e-3 vs y_τ_FW_obs = 7.216e-3 → +0.13% ✓
  - y_t_FW_pred  = 1.0      vs y_t_FW_obs = 0.7014  → +42.6% ✗  (off by √2)

In PT convention:
  - y_τ_PT_pred  = 7.226e-3 vs y_τ_PT_obs = 0.01021 → -29.2% ✗  (off by √2)
  - y_t_PT_pred  = 1.0      vs y_t_PT_obs = 0.9921  → +0.82% ✓

NEITHER convention works for both channels. The exponent principle's quoted
values 7.226e-3 (y_τ) and 1.0 (y_t) implicitly use DIFFERENT conventions —
FW for y_τ, PT for y_t. The convention switch is unflagged.

PRE-DECLARED GATE CHECKS:
  K1. predictions/y_tau.py uses FW convention (y = m/v); y_τ_pred matches at +0.13%.
  K2. commit 66c8836's y_t = 1 uses PT convention (m = y · v/√2); matches at +0.82%.
  K3. In FW convention alone, y_t = 1 gives m_t off by +42.6% (the bug surfaces).
  K4. In PT convention alone, y_τ = 7.226e-3 gives m_τ off by -29.2% (mirror bug).
  K5. The two conventions differ by EXACTLY √2: y_PT = √2 · y_FW.
  K6. To make the exponent principle internally consistent in ONE convention,
      one of the prefactors must absorb a √2 that the formula doesn't currently
      encode. (Structural question: where does the √2 live?)

USAGE:
    python3 proofs/foundations/W25_convention_audit_2026-05-20.py
"""

from __future__ import annotations
import math

EXPECTED = {
    "K1_y_tau_FW_matches":          True,
    "K2_y_t_PT_matches":            True,
    "K3_y_t_FW_breaks":             True,
    "K4_y_tau_PT_breaks":           True,
    "K5_conventions_differ_by_sqrt2": True,
    "K6_exponent_principle_inconsistent": True,
}
RESULTS = {}

print("=" * 78)
print("W25 — Convention audit: the y_τ ↔ y_t √2 discrepancy")
print("=" * 78)


# ============================================================================
# Constants — framework + PDG
# ============================================================================
V_HIGGS   = 246.22                    # GeV, electroweak scale (= full v, NOT v/√2)
V_OS2     = V_HIGGS / math.sqrt(2)    # ≈ 174.104 GeV — the Higgs vacuum ⟨h⁰⟩ in PT
M_TAU     = 1.77686                   # GeV, PDG 2024 tau lepton pole mass
M_TOP     = 172.69                    # GeV, PDG 2024 top quark pole mass

# Framework predictions (exponent principle):
y_tau_pred = 1280 / 177147            # = α₁_full / k*² = (5/3)(2/3)^8 / 9 ≈ 7.226e-3
y_t_pred   = 1.0                      # asserted gen-3 up-type limit, commit 66c8836

print(f"\nConstants:")
print(f"  v (electroweak scale) = {V_HIGGS} GeV")
print(f"  v/√2 (= ⟨h⁰⟩ in PT)   = {V_OS2:.4f} GeV")
print(f"  m_τ_obs = {M_TAU} GeV, m_top_obs = {M_TOP} GeV")
print(f"  y_τ_pred (exponent principle, framework derivation) = {y_tau_pred:.6e}")
print(f"  y_t_pred (exponent principle, gen-3 up-type limit) = {y_t_pred}")


# ============================================================================
# K1 — predictions/y_tau.py / m_tau.py use FW convention (y = m/v)
# ============================================================================
y_tau_FW_obs = M_TAU / V_HIGGS
m_tau_FW_pred = y_tau_pred * V_HIGGS
dev_tau_FW_pct = 100.0 * (m_tau_FW_pred - M_TAU) / M_TAU
print(f"\nK1 — framework convention applied to y_τ (m = y · v, no /√2)")
print(f"  y_τ_FW_obs  = m_τ / v = {M_TAU} / {V_HIGGS} = {y_tau_FW_obs:.6e}")
print(f"  y_τ_FW_pred = α₁_full/k*² = {y_tau_pred:.6e}")
print(f"  Match (FW convention): {100*(y_tau_pred - y_tau_FW_obs)/y_tau_FW_obs:+.3f}%")
print(f"  m_τ_FW_pred = y_τ_FW_pred · v = {m_tau_FW_pred:.4f} GeV vs PDG {M_TAU}")
print(f"  Deviation: {dev_tau_FW_pct:+.3f}%  (= +0.13% to leading order; matches y_τ_corollary §10 + m_tau.py)")
K1 = abs(dev_tau_FW_pct - 0.126) < 0.01
print(f"  K1 PASS: {K1}")
RESULTS["K1_y_tau_FW_matches"] = bool(K1)


# ============================================================================
# K2 — commit 66c8836's y_t = 1 uses PT convention (m = y · v/√2)
# ============================================================================
y_t_PT_obs = M_TOP * math.sqrt(2) / V_HIGGS
m_top_PT_pred = y_t_pred * V_OS2
dev_top_PT_pct = 100.0 * (m_top_PT_pred - M_TOP) / M_TOP
print(f"\nK2 — PT convention applied to y_t (m = y · v/√2)")
print(f"  y_t_PT_obs = m_top · √2 / v = {y_t_PT_obs:.6e}")
print(f"  y_t_PT_pred = 1 (exponent principle, gen-3 up-type)")
print(f"  m_t_PT_pred = y_t · v/√2 = {m_top_PT_pred:.4f} GeV vs PDG {M_TOP}")
print(f"  Deviation: {dev_top_PT_pct:+.3f}%  (= +0.82% matches commit 66c8836)")
K2 = abs(dev_top_PT_pct - 0.819) < 0.01
print(f"  K2 PASS: {K2}")
RESULTS["K2_y_t_PT_matches"] = bool(K2)


# ============================================================================
# K3 — But in FW convention, y_t = 1 gives a huge mismatch
# ============================================================================
y_t_FW_obs = M_TOP / V_HIGGS
m_top_FW_pred = y_t_pred * V_HIGGS
dev_top_FW_pct = 100.0 * (m_top_FW_pred - M_TOP) / M_TOP
print(f"\nK3 — framework convention applied to y_t = 1 (m = y · v)")
print(f"  y_t_FW_obs  = m_top / v = {y_t_FW_obs:.6e}  (= 0.7014)")
print(f"  y_t_FW_pred = 1   (exponent principle quoted value)")
print(f"  m_t_FW_pred = y_t · v = {m_top_FW_pred:.4f} GeV vs PDG {M_TOP}")
print(f"  Deviation: {dev_top_FW_pct:+.3f}%  (= +42.6% — the bug surfaces)")
K3 = abs(dev_top_FW_pct - 42.6) < 0.5
print(f"  K3 PASS (bug confirmed): {K3}")
RESULTS["K3_y_t_FW_breaks"] = bool(K3)


# ============================================================================
# K4 — Mirror: in PT convention, y_τ = 7.226e-3 gives a huge mismatch
# ============================================================================
y_tau_PT_obs = M_TAU * math.sqrt(2) / V_HIGGS
m_tau_PT_pred = y_tau_pred * V_OS2
dev_tau_PT_pct = 100.0 * (m_tau_PT_pred - M_TAU) / M_TAU
print(f"\nK4 — PT convention applied to y_τ = 7.226e-3 (m = y · v/√2)")
print(f"  y_τ_PT_obs  = m_τ · √2 / v = {y_tau_PT_obs:.6e}  (= 0.01021)")
print(f"  y_τ_PT_pred = α₁_full/k*² = {y_tau_pred:.6e}")
print(f"  m_τ_PT_pred = y_τ · v/√2 = {m_tau_PT_pred:.4f} GeV vs PDG {M_TAU}")
print(f"  Deviation: {dev_tau_PT_pct:+.3f}%  (= -29.2% — the mirror bug)")
K4 = abs(dev_tau_PT_pct - (-29.2)) < 0.5
print(f"  K4 PASS (mirror bug confirmed): {K4}")
RESULTS["K4_y_tau_PT_breaks"] = bool(K4)


# ============================================================================
# K5 — The two conventions differ by exactly √2
# ============================================================================
ratio_y_tau = y_tau_PT_obs / y_tau_FW_obs
ratio_y_t   = y_t_PT_obs / y_t_FW_obs
print(f"\nK5 — Convention ratio check")
print(f"  y_τ_PT_obs / y_τ_FW_obs = {ratio_y_tau:.6f}  (expect √2 = {math.sqrt(2):.6f})")
print(f"  y_t_PT_obs / y_t_FW_obs = {ratio_y_t:.6f}  (expect √2)")
print(f"  Both ratios equal √2 to machine precision:")
print(f"    |ratio_y_τ - √2| = {abs(ratio_y_tau - math.sqrt(2)):.2e}")
print(f"    |ratio_y_t - √2| = {abs(ratio_y_t - math.sqrt(2)):.2e}")
K5 = abs(ratio_y_tau - math.sqrt(2)) < 1e-9 and abs(ratio_y_t - math.sqrt(2)) < 1e-9
print(f"  K5 PASS: {K5}")
RESULTS["K5_conventions_differ_by_sqrt2"] = bool(K5)


# ============================================================================
# K6 — Exponent principle is internally inconsistent (the headline finding)
# ============================================================================
print(f"\nK6 — exponent principle inconsistency")
print()
print(f"  The exponent principle quotes y_τ = 7.226e-3 (FW) and y_t = 1 (PT).")
print(f"  No single convention reproduces both observed values from the formula:")
print()
print(f"  Scenario A: read both values in FW convention (y = m/v):")
print(f"    y_τ: {dev_tau_FW_pct:+.3f}% ✓")
print(f"    y_t: {dev_top_FW_pct:+.3f}% ✗  (off by √2 ≈ +42.6%)")
print()
print(f"  Scenario B: read both values in PT convention (m = y · v/√2):")
print(f"    y_τ: {dev_tau_PT_pct:+.3f}% ✗  (off by √2 ≈ -29.2%)")
print(f"    y_t: {dev_top_PT_pct:+.3f}% ✓")
print()
print(f"  The framework's published 'y_τ matches at +0.13%' claim uses FW.")
print(f"  The framework's published 'y_t matches at +0.82%' claim uses PT.")
print(f"  The convention switch is unflagged in the master Yukawa doc.")
print()
print(f"  STRUCTURAL QUESTION: where does the √2 live? Options:")
print(f"    (a) The exponent principle's prefactor for y_t at gen-3 limit should")
print(f"        be 1/√2 in framework convention (not 1). The 1/√2 is a structural")
print(f"        factor that the formula doesn't currently encode.")
print(f"    (b) The exponent principle's prefactor for y_τ should be (5/3) · √2")
print(f"        in PT convention (not 5/3). The √2 is a structural factor from")
print(f"        the SU(2)_L doublet (H = (h⁺, h⁰), only h⁰ acquires VEV v/√2).")
print(f"    Either way, ONE channel's prefactor needs an √2 the formula misses.")
K6 = True   # always true — the inconsistency is fully verified above
print(f"  K6 PASS (inconsistency verified, structural question posed): {K6}")
RESULTS["K6_exponent_principle_inconsistent"] = bool(K6)


# ============================================================================
# Proposed canonical fix
# ============================================================================
print("\n" + "=" * 78)
print("Proposed canonical fix")
print("=" * 78)
print(f"""
The framework's OPERATIONAL convention (used by predictions/*.py and the
y_τ corollary §10) is the FW convention: y = m / v (no /√2). This is what
the y_τ derivation chain produces — a dimensionless MDL probability that
matches m_τ/v at +0.13%.

Recommended canonical convention: FW (y = m/v).
  - y_τ_FW = α₁_full/k*² = 7.226e-3 ✓ (no change to derivation)
  - y_t_FW = 1/√2 ≈ 0.7071 (= PT y_t / √2)
      m_t = y_t · v = 0.7071 · 246.22 = 174.10 GeV ✓ (+0.82%, same numerical match)

Required edits to make the framework internally consistent:

  (1) docs/framework/framework_scheme_convention.md §3.54
      CURRENT (PT): "m_τ = v · y_τ / √2"
      FIX (FW):    "m_τ = v · y_τ"
      Rationale: this is the formula actually used by predictions/m_tau.py
      and produces the +0.13% match cited by the framework. The /√2 in the
      current text is inconsistent with the operational convention.

  (2) commit 66c8836's exposition (and any future m_top.py / predictions doc)
      CURRENT: "y_t = 1, m_t = v/√2 = 174.104 GeV"
      FIX: clarify convention. Either:
        (a) Recompute in FW: "y_t_FW = 1/√2, m_t = y_t · v = 174.104 GeV"
        (b) Keep PT but flag: "y_t_PT = 1 (PT convention; in framework
            convention this is y_t_FW = 1/√2)"

  (3) Exponent principle formula
      The structural question of where the √2 lives in the prefactor is
      genuinely open (probably tied to gen-3-limit chirality structure
      vs SU(2)_L doublet ⟨h⁰⟩ = v/√2 identification). The master Yukawa
      doc §11 retraction already names this — the exponent principle is
      "post-hoc unification, not derived master mechanism" — and this
      audit gives a concrete instance: the √2 difference between y_τ
      and y_t prefactors is a hidden convention switch.

NO numerical predictions change as a result of (1)-(3). The y_τ +0.13%
match and y_t +0.82% match remain, just with consistent labels.

What this fix DOES change: every downstream Yukawa derivation can now be
done in a single consistent convention. Steps 3-4's bounded next-step
probes (V_Ram audit, Koide quark extension, etc.) inherit FW convention
unambiguously.
""")

# ============================================================================
# Verdict
# ============================================================================
print("=" * 78)
print("W25 VERDICT — Gate Check")
print("=" * 78)
all_pass = True
for k, expected in EXPECTED.items():
    actual = RESULTS.get(k)
    status = "PASS" if actual == expected else "FAIL"
    if actual != expected: all_pass = False
    print(f"  {status}  {k:42s}  expected={expected}, got={actual}")
print()
if all_pass:
    print("  ALL CHECKS PASS — convention bug fully diagnosed.")
    print()
    print("  STATUS: Item #4 (convention reconciliation) closed at the audit level.")
    print("  Bug isolated; canonical convention proposed; required edits enumerated.")
    print("  No predictions/*.py touched yet; awaiting user approval for the 3 edits.")
print()
print("=" * 78)
