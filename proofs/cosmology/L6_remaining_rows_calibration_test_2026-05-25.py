#!/usr/bin/env python3
"""
L6 — testing remaining rows (n_s, r, σ_8, t_0) under the unified-process reframe.

CALIBRATION TEST per W58 discipline (beware numerology):

If the reframe is structurally correct, the remaining 4 L6-blocked rows
(n_s, r, σ_8, t_0 ΛCDM) should ALSO have clean framework-primitive
expressions matching Planck at sub-percent precision, with INDEPENDENTLY
chosen primitives (not the same n_g, N_local, N_atoms used for r_s, D_A,
θ_*).

If they fit cleanly → calibration evidence FOR the reframe (supports
the r_s/D_A matches as more than coincidence).

If they fit only with strained numerology or don't fit → supports the
EOD+9 numerology-suspect verdict on r_s/D_A.

DISCIPLINE:
  - For each row, identify the OBSERVED Planck value
  - Test framework-primitive candidates with explicit structural reading
  - Flag numerology risk: how many free primitive-product choices vs
    how many observable degrees of freedom
  - Honest verdict for each row
"""

from __future__ import annotations
import math
from fractions import Fraction


# Constants
c_light = 2.998e8
hbar = 1.054571817e-34
G_Newton = 6.6743e-11
t_P = math.sqrt(hbar * G_Newton / c_light ** 5)
Gyr_to_s = 3.156e16  # seconds per Gyr

# Framework primitives
k_star = 3
N_atoms = 4
n_E = 6
two_E = 2 * n_E
n_g = 15
g_girth = 10
N_local = 2 ** k_star * k_star
N_local_x_atoms = N_local * N_atoms
N_hub = 8.394881e60

alpha_GUT_bare = Fraction(1, 24)
alpha_1_bare = Fraction(2, 3) ** 8           # 256/6561
waterline = alpha_1_bare / (1 - alpha_1_bare)  # 256/6305
c_S = Fraction(1, two_E)

# Planck 2018 observations
n_s_obs = 0.9649
n_s_sigma = 0.0042
r_upper_95 = 0.07              # 95% CL upper limit
sigma_8_obs = 0.811
sigma_8_sigma = 0.006
t_0_obs_Gyr = 13.797
t_0_sigma_Gyr = 0.023


print("=" * 76)
print("L6 remaining rows — calibration test under unified-process reframe")
print("=" * 76)


# ===========================================================================
# Row 1: t_0 (age of universe under ΛCDM)
# ===========================================================================
print(f"\n{'#'*76}")
print("Row 1 — t_0 (age of universe under ΛCDM frame)")
print('#'*76)
print(f"""
Observed: Planck t_0 = {t_0_obs_Gyr} ± {t_0_sigma_Gyr} Gyr (ΛCDM-derived)

UNDER FRAMEWORK COASTING (a ∝ N, H = 1/(N·t_P)):
  Cosmological time = N × t_P  (one Planck-tick per observation)
  Today: t_0_coasting = N_hub × t_P
""")

t_0_framework_s = N_hub * t_P
t_0_framework_Gyr = t_0_framework_s / Gyr_to_s
t_0_residual = (t_0_framework_Gyr - t_0_obs_Gyr) / t_0_obs_Gyr * 100
t_0_sigma_distance = (t_0_framework_Gyr - t_0_obs_Gyr) / t_0_sigma_Gyr

print(f"  t_0_framework = N_hub × t_P = {N_hub:.3e} × {t_P:.3e} s")
print(f"                = {t_0_framework_s:.3e} s")
print(f"                = {t_0_framework_Gyr:.3f} Gyr")
print(f"  vs Planck     = {t_0_obs_Gyr} Gyr")
print(f"  Residual      = {t_0_residual:+.2f}% = {t_0_sigma_distance:+.1f}σ_Planck")
print(f"""
STRUCTURAL READING: t_0 = N_hub × t_P is the framework's NATURAL coasting
time. Coasting cosmology has H = 1/t exactly, so t_0 = 1/H_0 (the
"Hubble time" equals the age). Standard ΛCDM with matter+Λ has t_0 < 1/H_0
because of past deceleration. The +{t_0_residual:.1f}% disagreement
between coasting t_0 and ΛCDM-fit t_0 is EXPECTED structural difference,
not a numerical fit.

NUMEROLOGY ASSESSMENT: this is NOT a fit. t_0 = N_hub × t_P is forced
by coasting cosmology + N_hub adoption. The match quality reflects the
genuine difference between coasting and ΛCDM expansion histories.

VERDICT: STRUCTURAL PREDICTION at +{t_0_residual:.1f}% precision. Honest reading:
the framework's coasting cosmology predicts t_0 = 1/H_0 = 14.3 Gyr, which
is ~4% above ΛCDM-fit t_0 = 13.8 Gyr. This is a meaningful prediction
(not a fit), but doesn't match Planck at sub-percent.
""")


# ===========================================================================
# Row 2: n_s (scalar spectral index)
# ===========================================================================
print(f"\n{'#'*76}")
print("Row 2 — n_s (primordial scalar spectral index)")
print('#'*76)
print(f"""
Observed: Planck n_s = {n_s_obs} ± {n_s_sigma}

STANDARD READING: n_s = 1 - (slow-roll tilt). For scale-invariant
primordial power spectrum, n_s = 1; deviations measure inflationary
dynamics.

UNDER REFRAME: the primordial scalar power spectrum's tilt should come
from substrate-side primitives at the seed-fluctuation epoch.

CANDIDATE PRIMITIVE CHOICES:
""")

candidates_ns = [
    ("1 - α_1_bare = 1 - 256/6561", 1 - float(alpha_1_bare)),
    ("1 - α_1/(1-α_1) = 1 - 256/6305", 1 - float(waterline)),
    ("1 - α_GUT_bare = 1 - 1/24", 1 - float(alpha_GUT_bare)),
    ("1 - c_S = 11/12", 1 - float(c_S)),
    ("1 - 1/(2|E|·k*) = 35/36", 1 - 1/(two_E * k_star)),
    ("(2/3)^(1/9) = α_1_bare^(1/72)", (2/3)**(1/9)),
]

print(f"  {'candidate':<45} | {'value':<10} | {'σ distance':<12}")
print(f"  {'-'*45}-|-{'-'*10}-|-{'-'*12}")
for label, val in candidates_ns:
    sigma_off = (val - n_s_obs) / n_s_sigma
    print(f"  {label:<45} | {val:.6f} | {sigma_off:+8.2f}σ")

best_ns_candidate = 1 - float(alpha_1_bare)
print(f"""
BEST CANDIDATE: n_s = 1 - α_1_bare = 1 - 256/6561 = 0.96100
  Match: -0.93σ from Planck 0.9649 (within 1σ precision)
  Structural reading: scale-step tilt = α_1_bare (single dark winding)

NUMEROLOGY CHECK:
  - α_1_bare is theorem-grade primitive (NB walker survival probability)
  - The "tilt per scale-step" reading is suggestive but heuristic
  - With α_1_bare ≈ 0.039 vs needed 1-n_s ≈ 0.035, the match is at ~10%
    accuracy — not theorem-grade-numerical
  - Many primitive-products give values near 0.96 (1 - 1/24 = 0.958,
    1 - α_1/(1-α_1) = 0.959, etc.). The choice α_1_bare is structural
    BUT could be coincidence.

VERDICT: candidate-grade with moderate numerology risk. Within 1σ of
Planck but specific primitive choice not uniquely forced.
""")


# ===========================================================================
# Row 3: r (tensor-to-scalar ratio)
# ===========================================================================
print(f"\n{'#'*76}")
print("Row 3 — r (tensor-to-scalar ratio)")
print('#'*76)
print(f"""
Observed: r < {r_upper_95} at 95% CL (Planck+BICEP/Keck upper limit;
                                      no detection yet)

STANDARD READING: r = T/S = 16ε in slow-roll inflation, where ε is the
first slow-roll parameter. r relates tensor (gravitational wave) to
scalar perturbations.

UNDER REFRAME: r is the ratio of tensor to scalar primordial power.
Under substrate-side dynamics, tensor perturbations should be sub-leading
relative to scalar perturbations.

CANDIDATE FORMS:
""")

candidates_r = [
    ("α_1_bare = 256/6561", float(alpha_1_bare)),
    ("α_1/(1-α_1)", float(waterline)),
    ("α_GUT_bare = 1/24", float(alpha_GUT_bare)),
    ("α_1²_bare", float(alpha_1_bare) ** 2),
    ("c_S² = 1/144", float(c_S) ** 2),
    ("1/N_local_x_atoms = 1/96", 1.0 / N_local_x_atoms),
]

print(f"  {'candidate':<35} | {'value':<10} | {'< 0.07?':<8}")
print(f"  {'-'*35}-|-{'-'*10}-|-{'-'*8}")
for label, val in candidates_r:
    compat = "YES" if val < r_upper_95 else "NO"
    print(f"  {label:<35} | {val:.6f} | {compat:<8}")

print(f"""
COMPATIBILITY CHECK:
  Most framework primitives near 0.01-0.05 are compatible with r < 0.07.

LIMITATION: r is currently an UPPER LIMIT only (no detection). A
compatibility check is NOT a sub-percent fit. Until r is detected, we
cannot use it as calibration evidence.

VERDICT: r is consistent with several framework primitives (α_1_bare,
α_GUT_bare, α_1/(1-α_1) all under 0.07). This is only an UPPER-BOUND
COMPATIBILITY check, not a fit.
""")


# ===========================================================================
# Row 4: σ_8 (matter clustering amplitude at 8 h^-1 Mpc)
# ===========================================================================
print(f"\n{'#'*76}")
print("Row 4 — σ_8 (matter clustering amplitude)")
print('#'*76)
print(f"""
Observed: Planck σ_8 = {sigma_8_obs} ± {sigma_8_sigma}

STANDARD READING: σ_8 measures the rms amplitude of matter density
fluctuations smoothed on 8 h^-1 Mpc scale at z=0. Determined by
primordial power spectrum amplitude × structure-growth function.

UNDER REFRAME: σ_8 is a derived statistic of the observer's posterior
matter-density fluctuations integrated over a specific scale. Direct
expression in framework primitives is non-obvious.

CANDIDATE FORMS:
""")

candidates_sigma8 = [
    ("13/16", 13/16),
    ("(2/3)^(1/2) = √(2/3)", math.sqrt(2/3)),
    ("(5/6)·(1 - 1/N_local)", (5/6) * (1 - 1/N_local)),
    ("1 - α_1/(1-α_1)·(...)", 1 - float(waterline) * 4.65),  # would need fit
    ("c_S·(2|E|+something)", None),
    ("Constructed to match σ_8_obs = 0.811", None),
]

print(f"  {'candidate':<45} | {'value':<10} | {'σ distance':<12}")
print(f"  {'-'*45}-|-{'-'*10}-|-{'-'*12}")
for label, val in candidates_sigma8:
    if val is None:
        print(f"  {label:<45} | {'(no clean form)':<10} | N/A")
        continue
    sigma_off = (val - sigma_8_obs) / sigma_8_sigma
    print(f"  {label:<45} | {val:.4f} | {sigma_off:+8.2f}σ")

print(f"""
NUMEROLOGY CHECK:
  13/16 = 0.8125 matches σ_8 within +0.18% (well under 1σ Planck precision),
  but 13/16 has no obvious framework-primitive reading:
    - 13 is not 2|E|, k*², N_atoms × any factor, etc.
    - 16 = 2^(2k_star) but k_star = 3 gives 2^k* = 8, not 16
    - 16 = N_atoms² but that's not connected to σ_8 mechanism

  √(2/3) = 0.8165 is closer to a framework primitive (α_1_bare = (2/3)^8
  base), but +0.92σ from Planck.

VERDICT: σ_8 does NOT have a clean framework-primitive structural
reading. The "closest" matches (13/16, √(2/3)) are NUMEROLOGY-SUSPECT
without structural mechanism. σ_8 fails the calibration test.
""")


# ===========================================================================
# CALIBRATION TEST SUMMARY
# ===========================================================================
print(f"\n{'='*76}")
print("CALIBRATION TEST SUMMARY")
print('='*76)
print(f"""
Test: do the four remaining L6-blocked rows fit at sub-percent precision
with INDEPENDENTLY chosen framework primitives, providing calibration
evidence for the reframe?

RESULT BY ROW:

  t_0 (age):
    Status: STRUCTURAL PREDICTION (not fit) at +3.9% precision
    t_0_framework = N_hub × t_P = 14.34 Gyr vs Planck 13.797 Gyr
    The +4% reflects coasting vs ΛCDM expansion history (expected
    structural difference). Clean reading, no numerology.
    Verdict: structural-prediction-grade at 4%, NOT sub-percent.

  n_s (scalar spectral index):
    Status: candidate at -0.93σ Planck precision
    Best: n_s = 1 - α_1_bare = 0.961 vs Planck 0.9649
    Structural reading "tilt = single-dark-winding" is heuristic
    Numerology risk: moderate (several framework primitives give ~0.96)
    Verdict: candidate-grade within 1σ but not uniquely forced.

  r (tensor-to-scalar ratio):
    Status: COMPATIBILITY CHECK only (upper-bound)
    α_1_bare, α_GUT_bare, etc. all < 0.07 upper limit
    No sub-percent fit (no detection yet)
    Verdict: consistent but no calibration evidence.

  σ_8 (matter clustering):
    Status: NO CLEAN STRUCTURAL READING
    Closest 13/16 = 0.8125 is numerology-suspect (13 and 16 lack
    framework-primitive readings)
    Verdict: σ_8 does NOT calibrate.

OVERALL ASSESSMENT:

  Of 4 remaining L6-blocked rows:
  - 1 has clean structural prediction at +4% (t_0)
  - 1 is candidate within 1σ but specific primitive choice (n_s)
  - 1 is upper-bound compatible (r) — no fit
  - 1 has NO clean structural reading (σ_8)

  NONE of the 4 remaining rows fits at sub-percent with a clean
  independent framework primitive.

  Compare to r_s, D_A, θ_* (which DID claim sub-percent fits):
  - These three are reduced to TWO independent fits (D_A is derived
    from r_s and θ_*)
  - 2 free primitive choices fitting 2 observables is trivial χ²
  - The other 4 L6 rows do NOT replicate this pattern

CALIBRATION VERDICT:

  The calibration test does NOT provide independent evidence for the
  r_s/D_A/θ_* sub-percent matches. Across 4 remaining L6 rows:
  - t_0: clean structural reading at +4% (genuine evidence for the
    framework's structural framework, but at percent level, not sub-%)
  - n_s: candidate within 1σ (some evidence, with primitive-choice risk)
  - r: upper-bound only (no evidence either way)
  - σ_8: numerology-suspect (no evidence)

  Net: WEAK / MIXED calibration evidence. The reframe's specific
  numerical matches on r_s/D_A remain at "candidate-grade with elevated
  numerology risk" per EOD+9.

  The REFRAME'S CORE STRUCTURAL CONTENT (F4 η-dissolution, α=1/2 from
  beta-Bernoulli, T_today via cumulative-Perron, t_0 = N_hub × t_P
  coasting prediction) remains intact. These are STRUCTURAL findings
  not pinned to numerical fits.

  The reframe's NUMERICAL REACH (specific sub-percent fits) is more
  modest than the EOD+8 "3 sub-% matches" reading suggested. Per
  W58 discipline, the honest reading is:
  - Structural: F4, α=1/2, T_today, t_0 (4 substantive findings)
  - Numerical sub-percent fit: θ_* = 1/96 (single primitive choice)
  - Numerology-suspect: r_s = (c/H_0)/30, D_A derived
  - Failed calibration: σ_8 has no clean primitive form

NEXT STEPS:
  1. Accept the more modest scope: reframe has structural findings
     + 1 sub-% candidate (θ*) + several percent-level candidates (t_0,
     n_s, r compatibility)
  2. Multi-session structural work to rigorize the mechanisms
     (Bloch dispersion for r_s; angular-resolution mechanism for θ*)
  3. Or: pivot to a different open frontier given the bounded scope
     of what the reframe definitively closes
""")

print("=" * 76)
print("STATUS: Calibration evidence is MIXED. Reframe's core structural")
print("        content intact; numerical reach more modest than initial reading.")
print("=" * 76)
