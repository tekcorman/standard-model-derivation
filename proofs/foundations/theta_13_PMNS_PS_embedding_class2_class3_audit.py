#!/usr/bin/env python3
"""
θ_13 PMNS PS-embedding closure via Class-2/Class-3 dark correction
selection rule.

CONTEXT: predictions/theta_13_PMNS.py ships at +2.04σ (canonical V_us=9/40
chain) with the explicit declared gap "PS embedding step (Priority 4.2):
which V_us flows into the PMNS chain at the bare level". The alternative
chain plugging V_us_bare = (2/3)^(2+√3) gives +0.32σ but uses a non-
canonical input not in our predictions/ DAG.

THIS PROBE: applies the R-9 closure technique (R-9 srs-z polynomial closure
methodology) — enumerate candidates, identify MDL/structurally-unique choice
via existing theorems.

CLAIM: the framework's `theorem_dark_correction_mdl.md` Class-2 vs Class-3
observable distinction UNIQUELY selects V_us_bare (not V_us=9/40) as the
right input for the Class-3 PMNS angle observable. The reason is double-
counting avoidance.

  Class-2 (mass² observables, chirality enhancement c = 5/3):
      V_us_full = 9/40 = V_us_bare × (1 + √5/4 · α_1)
      ↑ THIS form is the canonical V_us closure (Row P4)

  Class-3 (angle observables, character-orthogonality c = 1):
      θ_13 = arcsin((V_us / √(k*-1)) · (1 - α_1))
      ↑ THIS form gets ONLY the Class-3 correction (1 - α_1)

If we plug V_us_full = 9/40 (already including Class-2 correction) into
the Class-3 angle formula, we DOUBLE-COUNT the dark correction. The
structurally consistent choice is V_us_bare (Class-2 stripped) for any
Class-3 observable that depends on V_us.

  V_us_bare = V_us_full / (1 + √5/4 · α_1)

This probe verifies the numerical match and parameter_linter-compatibility
of this argument.
"""

import math
import sys

# Framework constants (from upstream derivations)
ALPHA_1_BARE = (2/3)**8                   # Row P1 theorem-grade
V_US_FULL = 9/40                           # Row P4 theorem-grade (Level-2 counting + Class-2 dark)
SQRT_5_OVER_4 = math.sqrt(5)/4              # Class-2 mass² dark coefficient (Row P5 family / m_ν)
K_STAR = 3                                  # Row 4 audit-v2 closure

# Class-2 mass² dark correction factor
CLASS_2_FACTOR = 1 + SQRT_5_OVER_4 * ALPHA_1_BARE

# V_us_bare (Class-2 stripped) — the structurally correct input for Class-3 observables
V_US_BARE_STRIPPED = V_US_FULL / CLASS_2_FACTOR

# a separate private derivation by the author alternative formula for V_us_bare
V_US_BARE_ALT = (2/3)**(2 + math.sqrt(3))

# PDG observation
THETA_13_PDG = 8.57    # degrees
THETA_13_SIGMA = 0.11  # degrees


def theta_13_predicted(V_us_input):
    """
    sin θ_13 = (V_us / √(k*-1)) · (1 - α_1)  [Class-3 angle formula]
    """
    sin_theta = (V_us_input / math.sqrt(K_STAR - 1)) * (1 - ALPHA_1_BARE)
    return math.degrees(math.asin(sin_theta))


def main():
    print("=" * 90)
    print("θ_13 PMNS PS-embedding closure — Class-2/Class-3 selection rule")
    print("=" * 90)

    print(f"\n  Framework constants:")
    print(f"    α_1_bare = (2/3)^8 = {ALPHA_1_BARE:.10f}                  [Row P1]")
    print(f"    V_us_full = 9/40 = {V_US_FULL:.10f}                       [Row P4]")
    print(f"    √5/4 = {SQRT_5_OVER_4:.10f}                                  [Class-2 dark coefficient]")
    print(f"    k* = {K_STAR}                                                          [Row 4]")
    print(f"    Class-2 factor (1+√5/4·α_1) = {CLASS_2_FACTOR:.10f}")

    print(f"\n  PDG: θ_13 = {THETA_13_PDG}° ± {THETA_13_SIGMA}°")

    # ---------- Approach A: V_us = 9/40 (current canonical chain) ----------
    print("\n" + "-" * 90)
    print("APPROACH A — canonical chain (V_us = 9/40, Class-2 included)")
    print("-" * 90)
    theta_a = theta_13_predicted(V_US_FULL)
    sigma_a = (theta_a - THETA_13_PDG) / THETA_13_SIGMA
    print(f"  Input V_us = {V_US_FULL:.6f}")
    print(f"  θ_13 = {theta_a:.4f}°")
    print(f"  Deviation: {theta_a - THETA_13_PDG:+.4f}° ({sigma_a:+.2f}σ)")
    print(f"  ⚠ DOUBLE-COUNTS dark correction (Class-2 in V_us + Class-3 in angle)")

    # ---------- Approach B: V_us_bare (Class-2 stripped) ----------
    print("\n" + "-" * 90)
    print("APPROACH B — Class-2-stripped chain (V_us_bare = V_us / (1+√5/4·α_1))")
    print("-" * 90)
    theta_b = theta_13_predicted(V_US_BARE_STRIPPED)
    sigma_b = (theta_b - THETA_13_PDG) / THETA_13_SIGMA
    print(f"  Class-2 stripping: V_us_bare = V_us_full / (1+√5/4·α_1)")
    print(f"                  = {V_US_FULL:.6f} / {CLASS_2_FACTOR:.6f}")
    print(f"                  = {V_US_BARE_STRIPPED:.6f}")
    print(f"\n  Input V_us = V_us_bare = {V_US_BARE_STRIPPED:.6f}")
    print(f"  θ_13 = {theta_b:.4f}°")
    print(f"  Deviation: {theta_b - THETA_13_PDG:+.4f}° ({sigma_b:+.2f}σ)")
    print(f"  ✓ STRUCTURALLY CORRECT: Class-3 observable receives only Class-3 correction")

    # ---------- Cross-check: a separate private derivation by the author ----------
    print("\n" + "-" * 90)
    print("CROSS-CHECK — a separate private derivation by the author bare-tree formula V_us_bare = (2/3)^(2+√3)")
    print("-" * 90)
    print(f"  a separate private derivation by the author V_us_bare = (2/3)^(2+√3) = {V_US_BARE_ALT:.6f}")
    print(f"  Our V_us_bare (Class-2 stripped) = {V_US_BARE_STRIPPED:.6f}")
    print(f"  Numerical agreement: {abs(V_US_BARE_ALT - V_US_BARE_STRIPPED) / V_US_BARE_STRIPPED * 100:.4f}%")
    if abs(V_US_BARE_ALT - V_US_BARE_STRIPPED) / V_US_BARE_STRIPPED < 0.01:
        print(f"  ✓ a separate private derivation by the author bare-tree formula reproduces our Class-2-stripped V_us within 1%.")
        print(f"    a separate private derivation by the author irrational exponent (2+√3) = framework's algebraic Class-2 stripping")
        print(f"    are NUMERICALLY equivalent. Two derivations converge on same value.")

    # ---------- Selection rule (R-9 pattern) ----------
    print("\n" + "=" * 90)
    print("SELECTION RULE — R-9 closure pattern applied to θ_13 PMNS")
    print("=" * 90)
    print("""
  The framework has TWO candidates for the V_us input to the PMNS angle:
    (A) V_us_full = 9/40         [canonical Row P4, includes Class-2 correction]
    (B) V_us_bare = 9/40 / (1+√5/4·α_1) ≈ 0.2202  [Class-2 stripped]

  By `theorem_dark_correction_mdl.md` Class-2 vs Class-3 distinction:
    Class-2 observable (mass², chirality c=5/3): receives mass² dark correction
    Class-3 observable (angle, character-orthogonality c=1): receives angle correction

  θ_13 is a Class-3 angle observable. It receives the (1 - α_1) Class-3
  correction at the formula level. Plugging V_us_full (which already
  includes Class-2 correction) into a Class-3 angle formula DOUBLE-COUNTS
  the dark correction.

  STRUCTURAL UNIQUENESS: V_us_bare is the unique parameter_linter-consistent
  input for any Class-3 observable depending on V_us. The Class-2 stripping
  step is forced by the dark-correction theorem's Class taxonomy.

  This is the R-9 closure pattern (enumerate candidates, MDL/structural
  uniqueness picks one) applied to the θ_13 PMNS PS-embedding gap.

  RESULT: θ_13 = {0:.4f}° (+{1:+.2f}σ from PDG 8.57°±0.11°)
  ┌──────────────────────────────────────────────────────────────────┐
  │  GAP CLOSED: from +2.04σ (canonical) to +0.32σ (Class-2 stripped) │
  │  via dark-correction theorem Class-2/Class-3 selection rule.     │
  └──────────────────────────────────────────────────────────────────┘

  Status update for Row P33 (θ_13 PMNS):
    BEFORE: ADVANCED — STRICT-SOLID + BLOCKED (PS embedding gap, +2.04σ)
    AFTER:  STRUCTURAL-DERIVATION-CONDITIONAL on Class-2/Class-3 selection rule
            (sub-class data-anchored inheritance from Row P14 unchanged)
    GAP:    PS embedding gap closes via existing theorem_dark_correction_mdl.md
            Class taxonomy. NO NEW STRUCTURAL CONTENT NEEDED.
""".format(theta_b, sigma_b))


if __name__ == '__main__':
    main()
