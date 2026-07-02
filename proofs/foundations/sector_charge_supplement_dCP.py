#!/usr/bin/env python3
"""
Probe 1: δ_CP charge-supplement test (Row P34 R-14 sector-dependence audit).

CONTEXT
=======
Per an internal working note §3.1, a
post-hoc numerical observation: δ_CP_PMNS − δ_CP_CKM ≈ 108.5° is suggestively
close to arccos(−1/3) ≈ 109.47° (within ~1°). If structural, this would
suggest δ_CP_PMNS = arccos(1/3) + arccos(−1/3) = 180° exactly (lepton sector
uses K_4 vertex-antipode angle while quark sector uses K_4 dihedral angle).

The K_4 tetrahedron has TWO structurally-natural CP-relevant angles per
`predictions/delta_CP_CKM_geometry_derivation.md` §3 (Coxeter 1973):
- DIHEDRAL angle (between face planes meeting at an edge): arccos(1/3) ≈ 70.53°
- VERTEX angle (at centroid, between two vertex directions): arccos(−1/3) ≈ 109.47°
- These are supplementary: arccos(x) + arccos(−x) = π for all x ∈ [−1, 1].

The framework currently uses the DIHEDRAL angle for δ_CP_CKM (theorem-grade
geometric value at +0.68σ; identification with physical CKM phase inherited
from Row P14 V_ub closure). δ_CP_PMNS was retired (Row P34) because the
substrate-geometric route gave the SAME prediction (~70°), conflicting with
observation (~177°).

WHAT THIS PROBE TESTS
=====================
Hypothesis H1 (charge-supplement): δ_CP_PMNS = arccos(1/3) + arccos(−1/3) = 180°
(lepton charge differs from quark, picks up the supplementary K_4 angle).

Hypothesis H2 (vertex-only for leptons): δ_CP_PMNS = arccos(−1/3) ≈ 109.47°
(lepton uses vertex angle directly, not sum).

Hypothesis H3 (some other natural angle).

Compare each candidate against:
- NuFIT 6.0 NO best-fit: δ_CP_PMNS = 177° (+19/−20°)
- NuFIT 6.0 ⊕ NuFIT 5.3 combined: 230° ± 36° (looser)

Outcome:
- POSITIVE: a candidate matches within 1σ AND has a structural reading.
- NEGATIVE: matches are within tolerance but no structural reading; or no
  matches within tolerance.

DOES NOT
========
- Establish a structural derivation of WHY leptons use supplement and
  quarks use dihedral. Numerical match is necessary but not sufficient
  for closure; a structural argument would be the next-session task.
"""

from __future__ import annotations

import math

K_STAR = 3
ARCCOS_1_3 = math.degrees(math.acos(1/3))      # ≈ 70.529°
ARCCOS_NEG_1_3 = math.degrees(math.acos(-1/3))  # ≈ 109.471°
SUM_OF_TWO = ARCCOS_1_3 + ARCCOS_NEG_1_3        # = 180° exactly

# Observed values (multiple confidence regions per scoping doc)
DELTA_CP_CKM_OBS = 68.5         # PDG 2024 ± 3.0°
DELTA_CP_CKM_TOL = 3.0
DELTA_CP_PMNS_NUFIT_BEST = 177.0  # NuFIT 6.0 NO best fit
DELTA_CP_PMNS_NUFIT_TOL = 20.0    # +19/−20° (1σ)

# ============================================================================
# 1. K_4 angles verification (sanity)
# ============================================================================
print("=" * 78)
print("K_4 tetrahedral angles — Coxeter 1973 / framework-derived")
print("=" * 78)
print()
print(f"  Dihedral angle: arccos(1/k*) = arccos(1/{K_STAR}) = {ARCCOS_1_3:.6f}°")
print(f"  Vertex angle:   arccos(-1/k*) = arccos(-1/{K_STAR}) = {ARCCOS_NEG_1_3:.6f}°")
print(f"  Sum:            {SUM_OF_TWO:.6f}° (exactly 180° by arccos(x)+arccos(-x)=π)")
print()


# ============================================================================
# 2. δ_CP_CKM control (already theorem-grade)
# ============================================================================
print("=" * 78)
print("δ_CP_CKM — theorem-grade dihedral, control for sanity")
print("=" * 78)
print()
diff_ckm = abs(ARCCOS_1_3 - DELTA_CP_CKM_OBS)
sigma_ckm = diff_ckm / DELTA_CP_CKM_TOL
print(f"  Predicted: arccos(1/k*) = {ARCCOS_1_3:.4f}°")
print(f"  Observed:  {DELTA_CP_CKM_OBS}° ± {DELTA_CP_CKM_TOL}°")
print(f"  |Δ| = {diff_ckm:.4f}° = {sigma_ckm:.2f}σ — PASS")
print()


# ============================================================================
# 3. δ_CP_PMNS candidate test
# ============================================================================
print("=" * 78)
print("δ_CP_PMNS candidates — test against NuFIT 6.0 NO best fit")
print("=" * 78)
print()
print(f"  Observed: {DELTA_CP_PMNS_NUFIT_BEST}° ± {DELTA_CP_PMNS_NUFIT_TOL}° (NuFIT 6.0 NO 1σ)")
print()

candidates = [
    ("H1: arccos(1/3) + arccos(-1/3) = 180° (charge-supplement)",
     SUM_OF_TWO,
     "lepton charge difference adds K_4 vertex angle to dihedral"),
    ("H2: arccos(-1/3) (K_4 vertex only)",
     ARCCOS_NEG_1_3,
     "lepton sector uses K_4 vertex angle instead of dihedral"),
    ("H3a: 2·arccos(1/3)",
     2 * ARCCOS_1_3,
     "double dihedral (two-edge winding around K_4)"),
    ("H3b: 2·arccos(-1/3)",
     2 * ARCCOS_NEG_1_3,
     "double vertex angle (no obvious geometric meaning)"),
    ("H3c: 360° - arccos(1/3)",
     360 - ARCCOS_1_3,
     "negative-direction dihedral (CP conjugate)"),
    ("H3d: 360° - arccos(-1/3)",
     360 - ARCCOS_NEG_1_3,
     "negative-direction vertex angle"),
    ("H3e: arccos(1/3) + 180° (= arccos(-1/3) + π trivially)",
     ARCCOS_1_3 + 180,
     "dihedral + π flat (= vertex + π by supplement, mod 360 = same)"),
]

print(f"  {'candidate':<54}  {'value [°]':>10}  {'|Δ|':>7}  {'σ':>6}  match?")
print(f"  {'-'*54}  {'-'*10}  {'-'*7}  {'-'*6}  ------")
matches_within_1sigma = []
for name, val, _ in candidates:
    val_mod = val % 360
    diff = abs((val_mod - DELTA_CP_PMNS_NUFIT_BEST + 180) % 360 - 180)
    sig = diff / DELTA_CP_PMNS_NUFIT_TOL
    is_match = sig <= 1.0
    flag = "MATCH" if is_match else ("close" if sig <= 2.0 else "")
    print(f"  {name:<54}  {val_mod:>10.4f}  {diff:>7.4f}  {sig:>6.2f}σ  {flag}")
    if is_match:
        matches_within_1sigma.append((name, val_mod, sig))
print()


# ============================================================================
# 4. Structural reading audit per matching candidate
# ============================================================================
print("=" * 78)
print("Structural reading audit (which 1σ matches have a structural argument?)")
print("=" * 78)
print()
for name, val, _ in candidates:
    val_mod = val % 360
    diff = abs((val_mod - DELTA_CP_PMNS_NUFIT_BEST + 180) % 360 - 180)
    if diff > DELTA_CP_PMNS_NUFIT_TOL:  # outside 1σ
        continue
    print(f"--- {name} ---")
    print(f"  Numerical match: {val_mod:.4f}° vs {DELTA_CP_PMNS_NUFIT_BEST}° (|Δ| = {diff:.4f}°)")

    # Structural reading per candidate (subjective audit)
    structural_readings = {
        "H1: arccos(1/3) + arccos(-1/3) = 180° (charge-supplement)": (
            "PARTIAL: 180° is the trivial sum of supplementary K_4 angles; "
            "matches observed within 0.15σ. Structural reading would need: "
            "(a) why CKM uses dihedral (one face traversal), and (b) why PMNS "
            "uses dihedral + vertex (= edge-around plus vertex-antipode = full π). "
            "Both K_4 angles ARE structurally natural per Coxeter 1973. The "
            "selection of which sector uses which is the missing piece — "
            "matches the form of R-14 generally."
        ),
        "H2: arccos(-1/3) (K_4 vertex only)": (
            "OUTSIDE 1σ — 109.47° vs observed 177°, diff 67.53° = 3.4σ. SKIP."
        ),
        "H3a: 2·arccos(1/3)": (
            "OUTSIDE 1σ — 141.06° vs 177°, diff 35.94° = 1.8σ. Borderline. "
            "Reading: 2-edge dihedral winding (going around two adjacent edges). "
            "No obvious privileged structural reason for double-winding to apply "
            "to PMNS specifically."
        ),
        "H3c: 360° - arccos(1/3)": (
            "MATCH at 289.47° vs 177°: diff = mod-360 |289.47-177|=112.47° "
            "or |289.47-180-177|=180-67.47=... wait recompute. Likely outside."
        ),
        "H3d: 360° - arccos(-1/3)": (
            "MATCH? 250.53° vs 177°: diff = 73.53° = 3.7σ. NO."
        ),
        "H3e: arccos(1/3) + 180°": (
            "MATCH at 250.53°. Same as H3d numerically (since arccos(1/3)+180 "
            "= 70.53+180 = 250.53). Outside 1σ."
        ),
    }
    print(f"  Structural reading: {structural_readings.get(name, 'NOT EVALUATED')}")
    print()


# ============================================================================
# 5. Verdict
# ============================================================================
print("=" * 78)
print("PROBE 1 VERDICT")
print("=" * 78)
print()
if not matches_within_1sigma:
    print("  NO candidate matches within 1σ. Probe 1 NEGATIVE.")
elif len(matches_within_1sigma) == 1:
    name, val, sig = matches_within_1sigma[0]
    print(f"  ONE candidate matches within 1σ: {name}")
    print(f"    Value: {val:.4f}°, σ = {sig:.2f}")
    print()
    if "H1" in name:
        print("  Specific to H1 (charge-supplement → δ_CP_PMNS = 180°):")
        print("  - Numerical: 180° vs observed 177° at 0.15σ. PASS Clause 8.")
        print("  - Structural: requires argument for sector-dependent K_4 angle")
        print("    selection (CKM dihedral vs PMNS supplement). The K_4 angles")
        print("    ARE structurally natural; missing piece is the SELECTION RULE.")
        print("  - For Probe 1 to upgrade to R-14 closure for Row P34, need:")
        print("    1. Structural argument for the selection rule")
        print("       (e.g., charge-Q dependent winding on K_4 traversal).")
        print("    2. Cross-check that the prescription doesn't break")
        print("       Row P15 (δ_CP_CKM identification) or other K_4-using rows.")
        print()
        print("  STATUS: HYPOTHESIS H1 CONSISTENT WITH OBSERVATION at 0.15σ;")
        print("          structural derivation pending (1-2 sessions to attempt).")
else:
    print(f"  MULTIPLE candidates match within 1σ:")
    for name, val, sig in matches_within_1sigma:
        print(f"    - {name} at {val:.4f}° (σ = {sig:.2f})")
    print()
    print("  Probe 1 PARTIAL — multiple consistent candidates without selection rule.")

print()
print("=" * 78)
print("END")
print("=" * 78)
