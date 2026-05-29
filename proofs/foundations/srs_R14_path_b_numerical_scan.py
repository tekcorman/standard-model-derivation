#!/usr/bin/env python3
"""
R-14 path (b) numerical scan: test natural h × formula × observable triples
at both P and N high-symmetry points, looking for clean sector-distinguishing
matches that would support a leptons-at-P / quarks-at-N (or vice versa)
reading.

CONTEXT
=======
an internal working note (this session's
scoping doc) opened R-14 path (b) with the structural finding that P and N are
both uniformly Ramanujan-saturated in the BCC BZ, with P C_3-fixed and N having
3-element C_3-orbit. Recommended next-session protocol §6.A: 1-session
numerical scan to test whether ANY natural h-rule at N reproduces observed
sector-distinguishing observables.

WHAT THIS SCAN COMPUTES
=======================
For each h-value at P (2 distinct) and N (4 distinct) × each natural
formula template the framework uses elsewhere (winding n·arg, conjugate
arg(h*), chirality factor tan²(arg), with windings n ∈ {1...12}) ×
each testable observable, compute the prediction and check whether it
matches observation within tolerance.

The "natural formula templates" come from existing framework derivations:
  - α_21 PMNS: g·arg(h_P) at theorem-grade
  - α_31 PMNS: 2g·arg(h_P) at theorem-grade
  - δ_CP_PMNS retired: (g-1)·arg(h_P*) ≈ 249.85° (RETIRED — observed ~177°)
  - δ_CP_CKM: arccos(1/3) ≈ 70.53° from K_4 geometry (currently sector-blind)
  - α₁_full chirality: (5/3) = tan²(arg(h_P)) factor

The scan evaluates these templates at h_P (control: should reproduce existing
matches) and at h_N (test: looking for sector-shifted matches).

TESTABLE OBSERVABLES
====================
- α_21 PMNS ≈ 162° (NuFIT 6.0; loose, data-anchored fit)
- α_31 PMNS ≈ 325° (NuFIT 6.0; loose, data-anchored fit)
- δ_CP_PMNS = 177° +19/-20° (NuFIT 6.0 NO best fit — what currently retired
  (g-1)·arg(h_P*) failed to predict)
- δ_CP_CKM ≈ 68.5° ± 5° (PDG 2024)
- y_τ via tan²(arg) chirality factor: 5/3 at P (already correct at +0.13%)

NB: Many observables (V_us, V_cb, V_ub magnitudes; individual quark masses)
are NOT formally arg(h)-derived in the framework's existing derivations
(they use multi-cycle host structure or ratios). Including them in this scan
would test reformulations that don't exist; we restrict to angles which DO
have arg(h)-based existing or retired derivations.

OUTCOME INTERPRETATION
======================
- POSITIVE: An h_N value × natural formula combination reproduces an open
  observable (e.g., δ_CP_PMNS = 177°) that's currently unsolved at h_P.
- PARTIAL: h_N matches some open observables but breaks closed ones.
- NEGATIVE: h_N either breaks closed observables (α_21 = 162° won't
  reproduce) or fails to match open ones. Path (b) likely dead under this
  reading.
"""

from __future__ import annotations

import math
from itertools import product
import numpy as np

# ============================================================================
# 1. h-values at P and N (from previous probe)
# ============================================================================
# At P (k* = 3, k* - 1 = 2; |h|² = 2 saturating Ramanujan):
#   E = +√3 (mult 2): h = (√3 ± i√5)/2
#   E = -√3 (mult 2): h = (-√3 ± i√5)/2
# At N (also |h|² = 2 saturating Ramanujan, all eigenvalues mult 1):
#   E = +√5: h = (√5 ± i√3)/2
#   E = -√5: h = (-√5 ± i√3)/2
#   E = +1:  h = (1 ± i√7)/2          (NB: √7 NOT in K = ℚ(√2,√3,√5))
#   E = -1:  h = (-1 ± i√7)/2         (NB: √7 NOT in K)

H_VALUES = {
    'P_E+sqrt3':  (math.sqrt(3) + 1j * math.sqrt(5)) / 2,   # arg ≈ 52.24°
    'P_E-sqrt3':  (-math.sqrt(3) + 1j * math.sqrt(5)) / 2,  # arg ≈ 127.76°
    'N_E+sqrt5':  (math.sqrt(5) + 1j * math.sqrt(3)) / 2,   # arg ≈ 37.76°  (K-rational)
    'N_E-sqrt5':  (-math.sqrt(5) + 1j * math.sqrt(3)) / 2,  # arg ≈ 142.24° (K-rational)
    'N_E+1':      (1 + 1j * math.sqrt(7)) / 2,              # arg ≈ 69.30°  (NOT K-rational)
    'N_E-1':      (-1 + 1j * math.sqrt(7)) / 2,             # arg ≈ 110.70° (NOT K-rational)
}

K_RATIONAL_H = {'P_E+sqrt3', 'P_E-sqrt3', 'N_E+sqrt5', 'N_E-sqrt5'}

# Verify each is Ramanujan-saturated
for name, h in H_VALUES.items():
    assert abs(abs(h)**2 - 2) < 1e-12, f"{name} not Ramanujan: |h|² = {abs(h)**2}"

# Print h-values + their arg's in degrees
print("=" * 80)
print("h-values at P and N, with arg in degrees and K-rationality flag")
print("=" * 80)
print(f"\n  {'name':<14}  {'h':<28}  {'arg(h) [°]':>11}  {'tan²(arg)':>12}  {'K-rat'}")
for name, h in H_VALUES.items():
    arg_deg = math.degrees(np.angle(h)) % 360
    tan_sq = (h.imag / h.real) ** 2
    krat = 'yes' if name in K_RATIONAL_H else 'no'
    print(f"  {name:<14}  {h.real:+.4f}{'+'+f'{h.imag:.4f}'+'i' if h.imag>=0 else f'{h.imag:.4f}i':<19}  {arg_deg:>11.4f}  {tan_sq:>12.6f}  {krat}")
print()


# ============================================================================
# 2. Observables and tolerances
# ============================================================================
# Format: name -> (value, tolerance, sigma_tolerance, kind)
#   kind: 'angle' (degrees, mod 360) or 'tan_sq' (dimensionless ratio)
# Tolerances are σ-class for angles (5° wide), tighter for sharp-peak.

OBSERVABLES = {
    # PMNS angles (currently all at P-point)
    'alpha_21_PMNS':  {'val': 162.39, 'tol': 5.0,  'kind': 'angle',
                       'note': 'Currently theorem-grade at P: g·arg(h_P) = 162.39° (data-anchored fit ~162°)'},
    'alpha_31_PMNS':  {'val': 324.78, 'tol': 5.0,  'kind': 'angle',
                       'note': 'Currently theorem-grade at P: 2g·arg(h_P) = 324.78°'},
    'delta_CP_PMNS':  {'val': 177.0,  'tol': 25.0, 'kind': 'angle',
                       'note': 'NuFIT 6.0 NO best fit: 177° +19/-20°. Retired prediction at P: (g-1)·arg(h_P*) ≈ 249.85°'},
    # CKM angles
    'delta_CP_CKM':   {'val': 68.5,   'tol': 5.0,  'kind': 'angle',
                       'note': 'PDG 2024 ≈ 68.5°. Currently from K_4 dihedral arccos(1/3) ≈ 70.53° (geometry, sector-blind).'},
    # Chirality factors
    'tan_sq_chirality': {'val': 5/3,  'tol': 0.05, 'kind': 'tan_sq',
                         'note': 'Currently at P: tan²(arg(h_P)) = 5/3 enters α₁_full; gives y_τ +0.13%'},
}


# ============================================================================
# 3. Formula templates
# ============================================================================
# Each formula takes (h, n) where n is an integer winding (or other integer).
# Returns the value mod 360° for angles or as a tan² for chirality.
def angle_formulas():
    """Yield (label, fn, kind) for angle templates."""
    yield ('arg(h)',         lambda h: math.degrees(np.angle(h)) % 360,                        'angle')
    yield ('arg(h*)',        lambda h: math.degrees(np.angle(h.conjugate())) % 360,            'angle')
    for n in range(1, 13):
        yield (f'{n}·arg(h)', lambda h, n=n: (n * math.degrees(np.angle(h))) % 360,            'angle')
        yield (f'{n}·arg(h*)',lambda h, n=n: (n * math.degrees(np.angle(h.conjugate()))) % 360,'angle')
    yield ('arg(h)+180',     lambda h: (math.degrees(np.angle(h)) + 180) % 360,                'angle')
    yield ('-arg(h)',        lambda h: (-math.degrees(np.angle(h))) % 360,                     'angle')

def chirality_formulas():
    """Yield (label, fn, kind) for chirality templates."""
    yield ('tan²(arg(h))',   lambda h: (h.imag / h.real) ** 2,    'tan_sq')


# ============================================================================
# 4. Match all (h, formula, observable) triples; tally
# ============================================================================
print("=" * 80)
print("Match scan: (h, formula) → observable, within tolerance")
print("=" * 80)

results = []
for h_name, h in H_VALUES.items():
    krat = h_name in K_RATIONAL_H

    # angle formulas → angle observables
    for fname, fn, kind in angle_formulas():
        try:
            value = fn(h)
        except Exception:
            continue
        for obs_name, obs in OBSERVABLES.items():
            if obs['kind'] != 'angle':
                continue
            # Compare modulo 360, with appropriate tolerance
            target = obs['val']
            tol = obs['tol']
            # Compute angular distance
            diff = abs((value - target + 180) % 360 - 180)
            if diff < tol:
                results.append({
                    'h_name': h_name,
                    'h_K_rat': krat,
                    'formula': fname,
                    'value': value,
                    'observable': obs_name,
                    'target': target,
                    'tol': tol,
                    'diff': diff,
                })

    # chirality formulas → tan_sq observables
    for fname, fn, kind in chirality_formulas():
        try:
            value = fn(h)
        except Exception:
            continue
        for obs_name, obs in OBSERVABLES.items():
            if obs['kind'] != 'tan_sq':
                continue
            target = obs['val']
            tol = obs['tol']
            diff = abs(value - target)
            if diff < tol:
                results.append({
                    'h_name': h_name,
                    'h_K_rat': krat,
                    'formula': fname,
                    'value': value,
                    'observable': obs_name,
                    'target': target,
                    'tol': tol,
                    'diff': diff,
                })


# ============================================================================
# 5. Output organized by observable
# ============================================================================
print()
for obs_name, obs in OBSERVABLES.items():
    print(f"--- {obs_name} (target {obs['val']}, tol ±{obs['tol']}, kind {obs['kind']}) ---")
    print(f"    Note: {obs['note']}")
    matches = [r for r in results if r['observable'] == obs_name]
    if not matches:
        print(f"    ✗ NO MATCH within tolerance for any (h, formula).")
    else:
        # Sort by (K-rat first, then by diff)
        matches.sort(key=lambda r: (not r['h_K_rat'], r['diff']))
        for r in matches:
            krat = '✓K' if r['h_K_rat'] else '·'
            print(f"    {krat} {r['h_name']:<14} {r['formula']:<14} = {r['value']:>9.4f}   "
                  f"|Δ| = {r['diff']:>7.4f}")
    print()


# ============================================================================
# 6. Verdict structure
# ============================================================================
print("=" * 80)
print("VERDICT STRUCTURE for R-14 path (b)")
print("=" * 80)
print()

# Group results by (h_name, observable) to see which h values match what
match_summary = {}
for r in results:
    key = (r['h_name'], r['observable'])
    if key not in match_summary or r['diff'] < match_summary[key]['diff']:
        match_summary[key] = r

print("  Q1: Does any K-rational h_N (= N_E+sqrt5 or N_E-sqrt5) match")
print("      observables that h_P does not, within tolerance?")
print()

P_matches = {(r['h_name'], r['observable']) for r in results if r['h_name'].startswith('P_')}
N_K_matches = {(r['h_name'], r['observable']) for r in results if r['h_name'].startswith('N_E') and r['h_K_rat']}
N_nonK_matches = {(r['h_name'], r['observable']) for r in results if r['h_name'].startswith('N_E') and not r['h_K_rat']}

# Per-observable: at P? at N (K-rat)? at N (non-K-rat)?
observables_with_P_match = {obs for h, obs in P_matches}
observables_with_NK_match = {obs for h, obs in N_K_matches}
observables_with_Nn_match = {obs for h, obs in N_nonK_matches}

print(f"  Observables matched at h_P (any formula):           {sorted(observables_with_P_match) or 'NONE'}")
print(f"  Observables matched at h_N K-rational (any formula): {sorted(observables_with_NK_match) or 'NONE'}")
print(f"  Observables matched at h_N non-K-rational (any formula): {sorted(observables_with_Nn_match) or 'NONE'}")
print()

# Path-(b) candidate: open observables that match at h_N but not at h_P
P_unmatched = set(OBSERVABLES.keys()) - observables_with_P_match
N_K_unique = observables_with_NK_match - observables_with_P_match
N_nonK_unique = observables_with_Nn_match - observables_with_P_match

print(f"  Currently UNRESOLVED at h_P (no match at P):        {sorted(P_unmatched) or 'NONE — all observables matched at P'}")
print(f"  Newly resolved at h_N K-rational (vs h_P unmatch):  {sorted(N_K_unique) or 'NONE'}")
print(f"  Newly resolved at h_N non-K-rational (vs h_P unmatch): {sorted(N_nonK_unique) or 'NONE'}")
print()

# Path-(b) penalty: observables that match at h_P but BREAK at h_N
P_unique = observables_with_P_match - observables_with_NK_match - observables_with_Nn_match
print(f"  Matched at h_P but NOT at any h_N:                  {sorted(P_unique) or 'NONE'}")
print()

# Conclusion
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
if N_K_unique:
    print(f"  Q1 = YES (K-rational): h_N K-rational matches {len(N_K_unique)} observable(s)")
    print(f"  that h_P does not. PATH (b) HAS POSITIVE EVIDENCE.")
    print(f"  Specific: {sorted(N_K_unique)}")
elif N_nonK_unique:
    print(f"  Q1 = YES (non-K-rational only): h_N non-K-rational matches")
    print(f"  {len(N_nonK_unique)} observable(s) that h_P does not, but ONLY using h-values")
    print(f"  outside the framework's K = ℚ(√2,√3,√5) ring. This requires extending K")
    print(f"  to include √7, propagating elsewhere — needs separate justification.")
    print(f"  Specific: {sorted(N_nonK_unique)}")
else:
    print(f"  Q1 = NO: h_N (K-rational or otherwise) does NOT match any open observable.")
    print(f"  PATH (b) under this scan setup is NEGATIVE.")
print()
if P_unique:
    print(f"  Path-(b) penalty: moving observables to h_N would BREAK")
    print(f"  {len(P_unique)} currently-matched observable(s) at h_P:")
    print(f"  {sorted(P_unique)}")
    print()
    print(f"  ⇒ Even if Q1 had positive evidence, the leptons-at-P/quarks-at-N")
    print(f"    reading would need to ALSO preserve P-matched observables for")
    print(f"    the leptons. The 'leptons-at-N' reading is excluded directly.")
print()
print("=" * 80)
print("END")
print("=" * 80)
