#!/usr/bin/env python3
"""
Unified-rule probe: δ_CP = arccos(Q_a + Q_b) for SU(2)_L-doublet mixing.

CONTEXT
=======
Probe 1 (`sector_charge_supplement_dCP.py`) found δ_CP_PMNS ≈ 180° = π
matches NuFIT 6.0 best-fit at 0.15σ. The framing was "K_4 dihedral +
supplement" (sector picks one or both K_4 angles).

Audit during attempt to derive the K_4 selection rule surfaced a tighter
UNIFIED pattern: both δ_CP_CKM and δ_CP_PMNS are reproduced by

    δ_CP_(matrix) = arccos(Q_a + Q_b)

where Q_a, Q_b are signed SM electric charges of the two species mixed
by the W-vertex within their SU(2)_L doublet:
- CKM mixes (u, d) within the up/down-type quark SU(2)_L doublet:
    Q_u + Q_d = +2/3 + (-1/3) = +1/3 → arccos(1/3) = 70.529°
    vs PDG 2024: 68.5° ± 3.0° — match at 0.68σ.
- PMNS mixes (ν, e) within the lepton SU(2)_L doublet:
    Q_ν + Q_e = 0 + (-1) = -1 → arccos(-1) = 180°
    vs NuFIT 6.0 NO best-fit: 177° ± 20° — match at 0.15σ.

The pattern is UNIFIED across sectors, depending only on charge sums.

WHAT THIS PROBE DOES
====================
- Verify the numerical pattern at machine precision.
- Audit the structural prerequisites needed to upgrade this from a
  numerical pattern to a theorem-grade derivation.

DOES NOT DERIVE
===============
The structural origin of "δ_CP = arccos(Q_a + Q_b)" is not derived here.
The pattern is observation-anchored. Structural prerequisites surfaced
by the audit:

1. Charge MAGNITUDES Q = n/k* are theorem-grade (charge-before-color).
2. Charge SIGNS depend on PS labeling = ADOPTED-B3 (R-14 territory).
3. The geometric identification "arccos(charge sum) = walk phase" is
   itself underived — no upstream theorem connects charge sums to
   K_4 walk phases.

So a structural derivation depends on closing all three prerequisites,
each multi-session research-level. This probe documents the pattern
honestly without overclaiming.
"""

from __future__ import annotations

import math
from fractions import Fraction

# ============================================================================
# 1. SM signed charges (PDG convention)
# ============================================================================
Q_E = Fraction(-1, 1)        # charged lepton (e-, μ-, τ-)
Q_NU = Fraction(0, 1)         # neutrino (ν_e, ν_μ, ν_τ)
Q_U = Fraction(2, 3)          # up-type quark (u, c, t)
Q_D = Fraction(-1, 3)         # down-type quark (d, s, b)

K_STAR = 3

# ============================================================================
# 2. Compute arccos(Q_a + Q_b) for each W-vertex doublet pair
# ============================================================================
print("=" * 78)
print("Unified-rule δ_CP = arccos(Q_a + Q_b) test")
print("=" * 78)
print()
print(f"  SM signed charges:")
print(f"    Q_e = {Q_E}, Q_ν = {Q_NU}, Q_u = {Q_U}, Q_d = {Q_D}")
print()

doublets = [
    ("CKM (up-down quark SU(2)_L doublet)", "u", "d", Q_U, Q_D),
    ("PMNS (lepton SU(2)_L doublet)",       "ν", "e", Q_NU, Q_E),
]

print(f"  {'doublet':<48}  {'Q_a + Q_b':>12}  {'arccos [°]':>12}")
print(f"  {'-'*48}  {'-'*12}  {'-'*12}")

predictions = {}
for name, label_a, label_b, Q_a, Q_b in doublets:
    Q_sum = Q_a + Q_b
    Q_sum_float = float(Q_sum)
    if -1 <= Q_sum_float <= 1:
        angle = math.degrees(math.acos(Q_sum_float))
    else:
        angle = float('nan')
    predictions[name.split(' ')[0]] = angle
    print(f"  {name:<48}  {str(Q_sum):>12}  {angle:>12.6f}")
print()


# ============================================================================
# 3. Compare to observed CKM and PMNS
# ============================================================================
print("=" * 78)
print("Numerical match against observation")
print("=" * 78)
print()

observations = [
    ("CKM",  "PDG 2024 (CKMfitter)",                 68.5,  3.0),
    ("PMNS", "NuFIT 6.0 NO best fit",               177.0, 20.0),
]

print(f"  {'matrix':<6}  {'predicted [°]':>14}  {'observed [°]':>14}  {'tolerance':>10}  {'|Δ|':>8}  {'σ':>6}")
print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*8}  {'-'*6}")

all_within_1sigma = True
for label, src, obs, tol in observations:
    pred = predictions[label]
    diff = abs((pred - obs + 180) % 360 - 180)
    sigma = diff / tol
    flag = "✓" if sigma <= 1 else ("close" if sigma <= 2 else "✗")
    print(f"  {label:<6}  {pred:>14.4f}  {obs:>14.1f}  {f'± {tol}':>10}  {diff:>8.4f}  {sigma:>6.3f}σ  {flag}")
    if sigma > 1:
        all_within_1sigma = False

print()


# ============================================================================
# 4. Structural prerequisites audit
# ============================================================================
print("=" * 78)
print("Structural prerequisites audit (what would need to close for theorem-grade)")
print("=" * 78)
print()

prereqs = [
    ("(1) Charge MAGNITUDES |Q| = n/k*",
     "THEOREM-GRADE",
     "charge_before_color theorem 2026-05-03; U(1) factor of U(3) ⊂ Spin(6)"),
    ("(2) Charge SIGNS",
     "ADOPTED",
     "PS labeling via ADOPTED-B3 / Furey 2018 §3 identifies Q signs with SM sign convention"),
    ("(3) Identification: arccos(Q_a + Q_b) = phase",
     "UNDERIVED",
     "no upstream theorem connects charge sums to K_4 walk phases"),
    ("(4) W-vertex 4-walk on K_4 (CKM/PMNS Jarlskog)",
     "ADOPTED (Other-Smuggle)",
     "per delta_CP_CKM_geometry §6 — Need-A2 + Need-D"),
]

for name, status, note in prereqs:
    print(f"  {name}")
    print(f"    Status: {status}")
    print(f"    Note: {note}")
    print()


# ============================================================================
# 5. Honest verdict
# ============================================================================
print("=" * 78)
print("VERDICT")
print("=" * 78)
print()
if all_within_1sigma:
    print("  Numerical pattern δ_CP = arccos(Q_a + Q_b) reproduces both CKM and PMNS")
    print("  observations within 1σ (CKM at 0.68σ, PMNS at 0.15σ).")
    print()
    print("  Structural origin requires closing:")
    print("  - (3) arccos(Q_a + Q_b) = K_4 walk phase identification (no current upstream)")
    print("  - (2) PS labeling adoption (= R-14 closure path)")
    print()
    print("  Both prerequisites are multi-session research-level. The pattern is")
    print("  documented as a candidate sector-dependent formula. Numerical match")
    print("  is necessary but NOT sufficient for theorem-grade closure.")
    print()
    print("  The unified rule is TIGHTER than 'K_4 dihedral / supplement' framing")
    print("  because it provides a single rule across sectors via charge sums,")
    print("  rather than per-sector K_4 angle selection.")
else:
    print("  Numerical pattern fails for at least one observable.")
    print("  Pattern not viable.")

print()
print("=" * 78)
print("END")
print("=" * 78)
