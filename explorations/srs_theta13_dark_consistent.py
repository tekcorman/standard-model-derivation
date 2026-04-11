#!/usr/bin/env python3
"""
srs_theta13_dark_consistent.py — θ₁₃ with dark-corrected θ₂₃
=============================================================

GOAL: Replace cos(45°) = 1/√2 with cos(θ₂₃^dark) in the θ₁₃ formula,
      check which projection (sin vs cos) is correct, build full PMNS,
      explore spherical triangle on SU(4), verify self-consistency.

Physical argument:
  TBM column 3 = (0, 1/√2, 1/√2) when θ₂₃ = 45°.
  Dark-corrected column 3 = (0, sin(θ₂₃), cos(θ₂₃)) with θ₂₃ = 48.72°.
  The charged lepton correction projects V_us onto this column.
  Which element of U_l carries V_us determines sin vs cos.

Run: python3 explorations/srs_theta13_dark_consistent.py
"""

import numpy as np
from numpy import sqrt, pi, sin, cos, arcsin, arctan, arctan2
from numpy import linalg as la

DEG = 180.0 / pi
RAD = pi / 180.0

# ======================================================================
# FRAMEWORK CONSTANTS (all derived, zero free parameters)
# ======================================================================

k_star = 3
g = 10
n_g = 5
base = (k_star - 1) / k_star                    # 2/3
sqrt3 = sqrt(3.0)
L_us = 2 + sqrt3                                # spectral gap inverse
alpha1 = (n_g / k_star) * base**(g - 2)         # (5/3)(2/3)^8
V_us_framework = base ** L_us                    # (2/3)^{2+√3}
V_us_PDG = 0.2250
DELTA_KOIDE = 2.0 / 9.0

# Dark-corrected θ₂₃
theta23_dark_exact = arctan((1 + alpha1) / (1 - alpha1))  # radians
theta23_dark_deg = theta23_dark_exact * DEG

# PDG observations
theta12_obs = 33.41
theta13_obs = 8.54
theta23_obs = 49.2
theta23_obs_err = 1.3

print("=" * 78)
print("  θ₁₃ WITH DARK-CORRECTED θ₂₃: SELF-CONSISTENCY CHECK")
print("=" * 78)
print()
print(f"  Framework constants:")
print(f"    k* = {k_star}")
print(f"    α₁ = (5/3)(2/3)^8 = {alpha1:.6f}")
print(f"    V_us = (2/3)^{{2+√3}} = {V_us_framework:.6f}")
print(f"    θ₂₃(TBM) = 45.000°")
print(f"    θ₂₃(dark) = arctan((1+α₁)/(1-α₁)) = {theta23_dark_deg:.4f}°")
print(f"    θ₂₃(obs)  = {theta23_obs}° ± {theta23_obs_err}°")


# ======================================================================
# SECTION 1: sin(θ₂₃) vs cos(θ₂₃) — which projection?
# ======================================================================

print()
print("=" * 78)
print("  SECTION 1: sin(θ₂₃) vs cos(θ₂₃) — WHICH PROJECTION?")
print("=" * 78)

s23 = sin(theta23_dark_exact)
c23 = cos(theta23_dark_exact)

theta13_cos = arcsin(V_us_framework * c23) * DEG
theta13_sin = arcsin(V_us_framework * s23) * DEG
theta13_TBM = arcsin(V_us_framework / sqrt(2)) * DEG

print(f"""
  Dark-corrected TBM column 3: (0, sin(θ₂₃), cos(θ₂₃))
    sin(θ₂₃^dark) = {s23:.6f}
    cos(θ₂₃^dark) = {c23:.6f}
    1/√2           = {1/sqrt(2):.6f}

  U_PMNS(e,3) = (U_l†)_{{eμ}} × sin(θ₂₃) + (U_l†)_{{eτ}} × cos(θ₂₃)

  CASE A: (U_l†)_{{eμ}} = V_us, (U_l†)_{{eτ}} ≈ 0  [e↔μ mixing]
    sin(θ₁₃) = V_us × sin(θ₂₃^dark)
    θ₁₃ = {theta13_sin:.4f}°
    Error vs obs: {abs(theta13_sin - theta13_obs):.4f}° ({abs(theta13_sin - theta13_obs)/theta13_obs*100:.1f}%)
    Direction: WORSE than TBM (θ₂₃ > 45° means sin > cos)

  CASE B: (U_l†)_{{eμ}} ≈ 0, (U_l†)_{{eτ}} = V_us  [e↔τ mixing]
    sin(θ₁₃) = V_us × cos(θ₂₃^dark)
    θ₁₃ = {theta13_cos:.4f}°
    Error vs obs: {abs(theta13_cos - theta13_obs):.4f}° ({abs(theta13_cos - theta13_obs)/theta13_obs*100:.1f}%)
    Direction: BETTER than TBM

  CASE TBM: θ₂₃ = 45° (original formula)
    sin(θ₁₃) = V_us / √2
    θ₁₃ = {theta13_TBM:.4f}°
    Error vs obs: {abs(theta13_TBM - theta13_obs):.4f}° ({abs(theta13_TBM - theta13_obs)/theta13_obs*100:.1f}%)

  Observed: θ₁₃ = {theta13_obs}°
""")

# The physical question
print("  PHYSICAL ANALYSIS: Which case is correct?")
print("  " + "-" * 50)
print(f"""
  In the Pati-Salam quark-lepton correspondence:
    - (U_l)_{{eμ}} = V_us  (nearest-generation mixing: e↔μ)
    - (U_l)_{{eτ}} ~ V_ub ~ V_us × V_cb ≈ 0  (suppressed)

  For U_l real orthogonal:
    U_l†(e,μ) = U_l^T(e,μ) = U_l(μ,e)

  If U_l is a 1-2 rotation by angle θ_C (Cabibbo):
    U_l = [[cos θ_C, -sin θ_C, 0],
           [sin θ_C,  cos θ_C, 0],
           [0,        0,        1]]

  Then U_l†(e,μ) = U_l(μ,e) = sin θ_C ≈ V_us
  and  U_l†(e,τ) = U_l(τ,e) = 0

  This gives CASE A: sin(θ₁₃) = V_us × sin(θ₂₃)  →  {theta13_sin:.4f}°
  Which is WORSE than TBM.

  BUT WAIT: the column ordering matters. Let's check carefully.
""")


# ======================================================================
# SECTION 2: FULL PMNS MATRIX PRODUCT — dark-corrected U_ν
# ======================================================================

print("=" * 78)
print("  SECTION 2: FULL PMNS MATRIX — U_l† × U_ν(dark)")
print("=" * 78)

# Build dark-corrected TBM: standard TBM with θ₂₃ → θ₂₃^dark
# The TBM matrix is:
# U_TBM = R₂₃(θ₂₃) × R₁₂(θ₁₂^TBM) × diag(1,1,1)
# with θ₂₃ = 45° and θ₁₂ = arctan(1/√2) = 35.26°

theta12_TBM = arctan(1.0 / sqrt(2.0))  # 35.264°
theta23_TBM = pi / 4                     # 45°

def build_PMNS_angles(t12, t23, t13=0.0, dcp=0.0):
    """Build PMNS matrix from mixing angles (radians)."""
    c12, s12 = cos(t12), sin(t12)
    c23, s23 = cos(t23), sin(t23)
    c13, s13 = cos(t13), sin(t13)
    e_idcp = np.exp(1j * dcp)

    U = np.array([
        [ c12*c13,                s12*c13,               s13*np.conj(e_idcp)],
        [-s12*c23 - c12*s23*s13*e_idcp, c12*c23 - s12*s23*s13*e_idcp, s23*c13],
        [ s12*s23 - c12*c23*s13*e_idcp,-c12*s23 - s12*c23*s13*e_idcp, c23*c13],
    ])
    return U

# Standard TBM (θ₂₃ = 45°, θ₁₂ = 35.26°, θ₁₃ = 0°)
U_TBM = np.real(build_PMNS_angles(theta12_TBM, theta23_TBM, 0.0))

print(f"\n  U_TBM (standard, θ₂₃ = 45°):")
for i, fl in enumerate(['e  ', 'mu ', 'tau']):
    print(f"    {fl}: [{U_TBM[i,0]:+.6f}  {U_TBM[i,1]:+.6f}  {U_TBM[i,2]:+.6f}]")

# Dark-corrected TBM: θ₂₃ → θ₂₃^dark
U_nu_dark = np.real(build_PMNS_angles(theta12_TBM, theta23_dark_exact, 0.0))

print(f"\n  U_ν(dark) (θ₂₃ = {theta23_dark_deg:.4f}°):")
for i, fl in enumerate(['e  ', 'mu ', 'tau']):
    print(f"    {fl}: [{U_nu_dark[i,0]:+.6f}  {U_nu_dark[i,1]:+.6f}  {U_nu_dark[i,2]:+.6f}]")

print(f"\n  Third column comparison:")
print(f"    TBM:  (0, {U_TBM[1,2]:+.6f}, {U_TBM[2,2]:+.6f})")
print(f"    Dark: (0, {U_nu_dark[1,2]:+.6f}, {U_nu_dark[2,2]:+.6f})")
print(f"    sin(θ₂₃^dark) = {sin(theta23_dark_exact):.6f}, cos(θ₂₃^dark) = {cos(theta23_dark_exact):.6f}")

# Build U_l as Cabibbo rotation in 1-2 sector
theta_C = arcsin(V_us_framework)  # Cabibbo angle from V_us

U_l = np.eye(3)
U_l[0, 0] = cos(theta_C)
U_l[0, 1] = -sin(theta_C)
U_l[1, 0] = sin(theta_C)
U_l[1, 1] = cos(theta_C)

print(f"\n  U_l (Cabibbo rotation, θ_C = {theta_C*DEG:.4f}°):")
for i, fl in enumerate(['e  ', 'mu ', 'tau']):
    print(f"    {fl}: [{U_l[i,0]:+.6f}  {U_l[i,1]:+.6f}  {U_l[i,2]:+.6f}]")

# Compute U_PMNS = U_l† × U_ν
# Case 1: with TBM
U_PMNS_TBM = U_l.T @ U_TBM
# Case 2: with dark-corrected
U_PMNS_dark = U_l.T @ U_nu_dark

def extract_angles(U):
    """Extract mixing angles from PMNS matrix."""
    P = np.abs(U)**2
    s13_sq = P[0, 2]
    c13_sq = 1 - s13_sq
    s12_sq = P[0, 1] / c13_sq if c13_sq > 0 else 0
    s23_sq = P[1, 2] / c13_sq if c13_sq > 0 else 0
    t13 = arcsin(sqrt(max(s13_sq, 0))) * DEG
    t12 = arcsin(sqrt(max(min(s12_sq, 1), 0))) * DEG
    t23 = arcsin(sqrt(max(min(s23_sq, 1), 0))) * DEG
    return t12, t13, t23

t12_TBM, t13_TBM, t23_TBM_out = extract_angles(U_PMNS_TBM)
t12_dark, t13_dark, t23_dark_out = extract_angles(U_PMNS_dark)

print(f"\n  U_PMNS = U_l† × U_TBM:")
for i, fl in enumerate(['e  ', 'mu ', 'tau']):
    print(f"    {fl}: [{U_PMNS_TBM[i,0]:+.6f}  {U_PMNS_TBM[i,1]:+.6f}  {U_PMNS_TBM[i,2]:+.6f}]")
print(f"  Angles: θ₁₂ = {t12_TBM:.4f}°, θ₁₃ = {t13_TBM:.4f}°, θ₂₃ = {t23_TBM_out:.4f}°")

print(f"\n  U_PMNS = U_l† × U_ν(dark):")
for i, fl in enumerate(['e  ', 'mu ', 'tau']):
    print(f"    {fl}: [{U_PMNS_dark[i,0]:+.6f}  {U_PMNS_dark[i,1]:+.6f}  {U_PMNS_dark[i,2]:+.6f}]")
print(f"  Angles: θ₁₂ = {t12_dark:.4f}°, θ₁₃ = {t13_dark:.4f}°, θ₂₃ = {t23_dark_out:.4f}°")

print(f"\n  COMPARISON TABLE:")
print(f"  ┌─────────────────────────┬──────────┬──────────┬──────────┬──────────┐")
print(f"  │ Method                  │    θ₁₂   │    θ₁₃   │    θ₂₃   │  θ₁₃ err │")
print(f"  ├─────────────────────────┼──────────┼──────────┼──────────┼──────────┤")
print(f"  │ TBM (no correction)     │  35.264° │   0.000° │  45.000° │     --   │")
print(f"  │ U_l × TBM              │ {t12_TBM:7.3f}° │ {t13_TBM:7.3f}° │ {t23_TBM_out:7.3f}° │ {abs(t13_TBM - theta13_obs):6.2f}° │")
print(f"  │ U_l × U_ν(dark)        │ {t12_dark:7.3f}° │ {t13_dark:7.3f}° │ {t23_dark_out:7.3f}° │ {abs(t13_dark - theta13_obs):6.2f}° │")
print(f"  │ Observed                │ {theta12_obs:7.2f}° │ {theta13_obs:7.2f}° │ {theta23_obs:7.2f}° │     --   │")
print(f"  └─────────────────────────┴──────────┴──────────┴──────────┴──────────┘")


# ======================================================================
# SECTION 2b: Detailed element-level analysis
# ======================================================================

print()
print("=" * 78)
print("  SECTION 2b: ELEMENT-LEVEL ANALYSIS OF U_PMNS(e,3)")
print("=" * 78)

# U_PMNS(e,3) = sum_k U_l†(e,k) × U_ν(k,3)
# = U_l(0,0)×U_ν(0,3) + U_l(1,0)×U_ν(1,3) + U_l(2,0)×U_ν(2,3)
# (since U_l real: U_l†(e,k) = U_l^T(e,k) = U_l(k,e) = U_l(k,0) for e=0)

print(f"\n  For TBM:")
print(f"    U_PMNS(e,3) = U_l(0,0)×U_TBM(0,3) + U_l(1,0)×U_TBM(1,3) + U_l(2,0)×U_TBM(2,3)")
print(f"               = {U_l[0,0]:.6f}×{U_TBM[0,2]:.6f} + {U_l[1,0]:.6f}×{U_TBM[1,2]:.6f} + {U_l[2,0]:.6f}×{U_TBM[2,2]:.6f}")
term1_tbm = U_l[0,0]*U_TBM[0,2]
term2_tbm = U_l[1,0]*U_TBM[1,2]
term3_tbm = U_l[2,0]*U_TBM[2,2]
print(f"               = {term1_tbm:.6f} + {term2_tbm:.6f} + {term3_tbm:.6f}")
print(f"               = {term1_tbm + term2_tbm + term3_tbm:.6f}")
print(f"    Direct: U_PMNS(e,3) = {U_PMNS_TBM[0,2]:.6f}")
print(f"    |sin(θ₁₃)| = {abs(U_PMNS_TBM[0,2]):.6f}")
print(f"    θ₁₃ = {arcsin(abs(U_PMNS_TBM[0,2]))*DEG:.4f}°")

print(f"\n  For dark-corrected:")
print(f"    U_PMNS(e,3) = U_l(0,0)×U_ν(0,3) + U_l(1,0)×U_ν(1,3) + U_l(2,0)×U_ν(2,3)")
print(f"               = {U_l[0,0]:.6f}×{U_nu_dark[0,2]:.6f} + {U_l[1,0]:.6f}×{U_nu_dark[1,2]:.6f} + {U_l[2,0]:.6f}×{U_nu_dark[2,2]:.6f}")
term1_dk = U_l[0,0]*U_nu_dark[0,2]
term2_dk = U_l[1,0]*U_nu_dark[1,2]
term3_dk = U_l[2,0]*U_nu_dark[2,2]
print(f"               = {term1_dk:.6f} + {term2_dk:.6f} + {term3_dk:.6f}")
print(f"               = {term1_dk + term2_dk + term3_dk:.6f}")
print(f"    Direct: U_PMNS(e,3) = {U_PMNS_dark[0,2]:.6f}")
print(f"    |sin(θ₁₃)| = {abs(U_PMNS_dark[0,2]):.6f}")
print(f"    θ₁₃ = {arcsin(abs(U_PMNS_dark[0,2]))*DEG:.4f}°")

print(f"""
  KEY INSIGHT:
    U_l(1,0) = sin(θ_C) = V_us = {U_l[1,0]:.6f}  (the dominant term)
    U_l(2,0) = 0                                    (Cabibbo is 1-2 only)

    For TBM:  U_PMNS(e,3) = V_us × (1/√2) = {V_us_framework/sqrt(2):.6f}
    For dark: U_PMNS(e,3) = V_us × sin(θ₂₃^dark) = {V_us_framework * sin(theta23_dark_exact):.6f}

    Since θ₂₃ > 45°: sin(θ₂₃) > 1/√2 > cos(θ₂₃)
    So the dark correction makes θ₁₃ LARGER, not smaller.

    The cos(θ₂₃) case would require (U_l)_{{eτ}} = V_us instead of (U_l)_{{eμ}}.
    But the Cabibbo rotation is in the 1-2 sector (e↔μ), so (U_l)_{{eτ}} = 0.
""")


# ======================================================================
# SECTION 3: ALTERNATIVE — U_l with 2-3 rotation or full 3-generation
# ======================================================================

print("=" * 78)
print("  SECTION 3: ALTERNATIVE U_l STRUCTURES")
print("=" * 78)

# What if U_l has a 1-3 component (e↔τ mixing)?
# This would need a physical justification from srs.

print(f"\n  Scan: what U_l structure gives θ₁₃ closest to observation?")
print(f"  " + "-" * 60)

# Parameterize U_l = R₂₃(β) × R₁₃(γ) × R₁₂(α)
# We know α ≈ θ_C. Scan β and γ.
best_err = 999
best_params = None

for beta_deg in np.linspace(-10, 10, 201):
    for gamma_deg in np.linspace(-5, 5, 101):
        beta = beta_deg * RAD
        gamma = gamma_deg * RAD
        # R12
        R12 = np.eye(3)
        R12[0,0] = cos(theta_C); R12[0,1] = -sin(theta_C)
        R12[1,0] = sin(theta_C); R12[1,1] = cos(theta_C)
        # R13
        R13 = np.eye(3)
        R13[0,0] = cos(gamma); R13[0,2] = sin(gamma)
        R13[2,0] = -sin(gamma); R13[2,2] = cos(gamma)
        # R23
        R23 = np.eye(3)
        R23[1,1] = cos(beta); R23[1,2] = -sin(beta)
        R23[2,1] = sin(beta); R23[2,2] = cos(beta)

        U_l_test = R23 @ R13 @ R12
        U_PMNS_test = U_l_test.T @ U_nu_dark
        t12_t, t13_t, t23_t = extract_angles(U_PMNS_test)

        # Minimize total angular error
        err = (t13_t - theta13_obs)**2 + 0.1*(t23_t - theta23_obs)**2
        if err < best_err:
            best_err = err
            best_params = (beta_deg, gamma_deg, t12_t, t13_t, t23_t)

b_best, g_best, t12_b, t13_b, t23_b = best_params
print(f"  Best fit: β = {b_best:.2f}° (2-3 rot), γ = {g_best:.2f}° (1-3 rot)")
print(f"  Angles: θ₁₂ = {t12_b:.4f}°, θ₁₃ = {t13_b:.4f}°, θ₂₃ = {t23_b:.4f}°")
print(f"  Observed: θ₁₂ = {theta12_obs}°, θ₁₃ = {theta13_obs}°, θ₂₃ = {theta23_obs}°")
print()

# Check if γ is related to any framework quantity
print(f"  Is γ = {g_best:.4f}° a framework quantity?")
print(f"    α₁ in degrees: {alpha1 * DEG:.4f}°")
print(f"    V_us² × DEG: {V_us_framework**2 * DEG:.4f}°")
print(f"    arctan(α₁): {arctan(alpha1) * DEG:.4f}°")


# ======================================================================
# SECTION 4: SPHERICAL TRIANGLE ON SU(4) MANIFOLD
# ======================================================================

print()
print("=" * 78)
print("  SECTION 4: SPHERICAL TRIANGLE FORMULATION")
print("=" * 78)

# On SU(4)/U(1)^3, the mixing angles define a point on a sphere.
# θ_TBM and θ_C are perpendicular (proven in srs_theta12_perp.py).
# The dark correction adds a third direction.

theta_C_deg = theta_C * DEG
theta_TBM_12_deg = theta12_TBM * DEG

# Perpendicularity: θ_TBM in ν-sector, θ_C in l-sector
# On the unit sphere, these define two sides of a right spherical triangle.
# The hypotenuse gives the physical θ₁₃.

# Spherical law of cosines for right triangle (C = 90°):
# cos(c) = cos(a) × cos(b)
# where a, b are the two sides adjacent to the right angle,
# and c is the hypotenuse.

# What are the "sides"?
# Side a: the Cabibbo angle (charged lepton correction)
# Side b: the TBM θ₁₃ correction (= 0 for pure TBM)
# But θ₁₃(TBM) = 0, so this is degenerate.

# Better: think of θ₁₃ as the "out-of-plane" angle
# when combining perpendicular rotations in the 1-2 and 2-3 planes.

# For perpendicular rotations R₁₂(θ_C) and R₂₃(θ₂₃):
# The (e,3) element of R₁₂^T × R₂₃ is:
# [R₁₂^T]_{e,μ} × [R₂₃]_{μ,3} = sin(θ_C) × sin(θ₂₃)  [for pure 2-3 rot]
# But TBM involves BOTH θ₁₂ and θ₂₃.

# Spherical excess formula:
# For a right spherical triangle with legs a, b:
# tan(E/2) = tan(a/2) × tan(b/2)
# where E = excess = area of triangle

# The key is: what is the "distance" on the SU(4) manifold?
# The Fubini-Study metric gives geodesic distance = arccos(|<ψ₁|ψ₂>|)

# Let's compute the FS distances between TBM, dark-TBM, and PMNS

# States (columns of the mixing matrices)
def FS_distance(v1, v2):
    """Fubini-Study distance between two state vectors."""
    overlap = abs(np.dot(np.conj(v1), v2))
    overlap = min(overlap, 1.0)
    return np.arccos(overlap)

# Third columns (the ν₃ mass eigenstate)
col3_TBM = U_TBM[:, 2]
col3_dark = U_nu_dark[:, 2]
col3_PMNS_TBM = U_PMNS_TBM[:, 2]
col3_PMNS_dark = U_PMNS_dark[:, 2]
col3_obs = np.array([sin(theta13_obs*RAD),
                     sin(theta23_obs*RAD)*cos(theta13_obs*RAD),
                     cos(theta23_obs*RAD)*cos(theta13_obs*RAD)])

d_TBM_dark = FS_distance(col3_TBM, col3_dark)
d_TBM_PMNS = FS_distance(col3_TBM, col3_PMNS_TBM)
d_dark_PMNS = FS_distance(col3_dark, col3_PMNS_dark)
d_TBM_obs = FS_distance(col3_TBM, col3_obs)
d_dark_obs = FS_distance(col3_dark, col3_obs)

print(f"""
  Fubini-Study distances (on CP² for ν₃ column):

  d(TBM, dark-TBM) = {d_TBM_dark * DEG:.4f}°  [pure dark correction to θ₂₃]
  d(TBM, PMNS_TBM) = {d_TBM_PMNS * DEG:.4f}°  [pure U_l correction]
  d(dark, PMNS_dk) = {d_dark_PMNS * DEG:.4f}°  [U_l correction on dark basis]
  d(TBM, observed)  = {d_TBM_obs * DEG:.4f}°  [total deviation from TBM]
  d(dark, observed)  = {d_dark_obs * DEG:.4f}°  [deviation from dark-corrected]

  Spherical triangle check (right angle at TBM):
    If θ_C and δθ₂₃ are perpendicular, then:
    cos(hypotenuse) = cos(d_C) × cos(δθ₂₃)

    d_C = θ_C = {theta_C_deg:.4f}°
    δθ₂₃ = {(theta23_dark_deg - 45):.4f}°
    cos(d_C) × cos(δθ₂₃) = {cos(theta_C) * cos(theta23_dark_exact - pi/4):.6f}
    cos(d_TBM_obs) = {cos(d_TBM_obs):.6f}
    Predicted hypotenuse = {np.arccos(cos(theta_C) * cos(theta23_dark_exact - pi/4))*DEG:.4f}°
    Actual d(TBM, obs) = {d_TBM_obs * DEG:.4f}°
""")

# Pythagorean check (flat space limit of spherical)
d_flat = sqrt(theta_C**2 + (theta23_dark_exact - pi/4)**2) * DEG
print(f"  Flat-space Pythagorean: √(θ_C² + δθ₂₃²) = {d_flat:.4f}°")
print(f"  d(TBM, obs) = {d_TBM_obs * DEG:.4f}°")


# ======================================================================
# SECTION 5: SELF-CONSISTENCY — same perturbation h for θ₂₃ and θ₁₃?
# ======================================================================

print()
print("=" * 78)
print("  SECTION 5: SELF-CONSISTENCY OF DARK PERTURBATION")
print("=" * 78)

# The dark perturbation parameter is α₁ = (5/3)(2/3)^8.
# It enters θ₂₃ as: θ₂₃ = arctan((1+α₁)/(1-α₁))
# It enters θ₁₃ through U_l via V_us = (2/3)^{2+√3}, and through θ₂₃.
# The question: do both corrections come from the SAME h perturbation?

print(f"""
  The perturbation chain:

  1. Dark sector breaks C₃ at P → eigenvalue split ±α₁
     α₁ = (5/3)(2/3)^8 = {alpha1:.6f}

  2. This gives θ₂₃ = arctan((1+α₁)/(1-α₁)) = {theta23_dark_deg:.4f}°
     δθ₂₃ = {theta23_dark_deg - 45:.4f}°

  3. V_us = (2/3)^{{2+√3}} = {V_us_framework:.6f} is INDEPENDENT of dark sector
     (it comes from the NB walk on the COMPRESSED graph, not dark)

  4. θ₁₃ = arcsin(V_us × f(θ₂₃))
     where f depends on U_l structure:
       Pure Cabibbo: f = sin(θ₂₃)    → θ₁₃ = {theta13_sin:.4f}° (WORSE)
       e-τ mixing:   f = cos(θ₂₃)    → θ₁₃ = {theta13_cos:.4f}° (BETTER)
       TBM limit:    f = 1/√2         → θ₁₃ = {theta13_TBM:.4f}° (baseline)

  SELF-CONSISTENCY CHECK:
  ─────────────────────────
""")

# The dark perturbation modifies h → h(1 ± α₁).
# For θ₂₃: direct eigenvalue split in ν sector.
# For θ₁₃: the question is whether V_us also gets a dark correction.

# V_us comes from NB walks on the compressed graph (k ≤ k*).
# The dark sector (k > k*) should modify V_us as well, but at higher order.
# The first-order dark correction to V_us:
# V_us_dark = (2/3)^{2+√3} × (1 + δ_dark)
# where δ_dark ~ α₁ × (something)

# If both V_us and θ₂₃ get corrected by the SAME perturbation:
V_us_dark_enhanced = V_us_framework * (1 + alpha1)
V_us_dark_suppressed = V_us_framework * (1 - alpha1)

theta13_enhanced = arcsin(V_us_dark_enhanced / sqrt(2)) * DEG
theta13_suppressed = arcsin(V_us_dark_suppressed / sqrt(2)) * DEG

print(f"  If V_us also gets dark correction:")
print(f"    V_us × (1+α₁) = {V_us_dark_enhanced:.6f} → θ₁₃ = {theta13_enhanced:.4f}°")
print(f"    V_us × (1-α₁) = {V_us_dark_suppressed:.6f} → θ₁₃ = {theta13_suppressed:.4f}°")
print(f"    V_us (no corr) = {V_us_framework:.6f} → θ₁₃ = {theta13_TBM:.4f}°")
print(f"    Observed: θ₁₃ = {theta13_obs}°")
print()

# The combined correction: both V_us and θ₂₃ modified
# sin(θ₁₃) = V_us(1+α₁) × sin(θ₂₃^dark) — if both enhance
theta13_both_plus = arcsin(V_us_dark_enhanced * sin(theta23_dark_exact)) * DEG
# sin(θ₁₃) = V_us(1-α₁) × sin(θ₂₃^dark)
theta13_both_minus = arcsin(V_us_dark_suppressed * sin(theta23_dark_exact)) * DEG
# sin(θ₁₃) = V_us(1-α₁) × cos(θ₂₃^dark)
theta13_minus_cos = arcsin(V_us_dark_suppressed * cos(theta23_dark_exact)) * DEG
# sin(θ₁₃) = V_us(1+α₁) × cos(θ₂₃^dark)
theta13_plus_cos = arcsin(V_us_dark_enhanced * cos(theta23_dark_exact)) * DEG

print(f"  Combined dark corrections (V_us AND θ₂₃):")
print(f"  ┌──────────────────────────────────────────────┬──────────┬────────┐")
print(f"  │ Formula                                      │    θ₁₃   │ error  │")
print(f"  ├──────────────────────────────────────────────┼──────────┼────────┤")
print(f"  │ V_us/√2                      [TBM baseline] │ {theta13_TBM:7.4f}° │ {abs(theta13_TBM-theta13_obs):5.2f}°  │")
print(f"  │ V_us × sin(θ₂₃^dk)          [Cabibbo + dk] │ {theta13_sin:7.4f}° │ {abs(theta13_sin-theta13_obs):5.2f}°  │")
print(f"  │ V_us × cos(θ₂₃^dk)          [e-τ + dark]   │ {theta13_cos:7.4f}° │ {abs(theta13_cos-theta13_obs):5.2f}°  │")
print(f"  │ V_us(1+α₁) × sin(θ₂₃^dk)   [both +]       │ {theta13_both_plus:7.4f}° │ {abs(theta13_both_plus-theta13_obs):5.2f}°  │")
print(f"  │ V_us(1-α₁) × sin(θ₂₃^dk)   [Vus- sin]     │ {theta13_both_minus:7.4f}° │ {abs(theta13_both_minus-theta13_obs):5.2f}°  │")
print(f"  │ V_us(1-α₁) × cos(θ₂₃^dk)   [Vus- cos]     │ {theta13_minus_cos:7.4f}° │ {abs(theta13_minus_cos-theta13_obs):5.2f}°  │")
print(f"  │ V_us(1+α₁) × cos(θ₂₃^dk)   [Vus+ cos]     │ {theta13_plus_cos:7.4f}° │ {abs(theta13_plus_cos-theta13_obs):5.2f}°  │")
print(f"  │ V_us(1-α₁)/√2               [Vus- TBM]     │ {theta13_suppressed:7.4f}° │ {abs(theta13_suppressed-theta13_obs):5.2f}°  │")
print(f"  │ Observed                                     │ {theta13_obs:7.2f}° │   --   │")
print(f"  └──────────────────────────────────────────────┴──────────┴────────┘")


# ======================================================================
# SECTION 5b: What combination exactly reproduces θ₁₃(obs)?
# ======================================================================

print()
print("  INVERSE PROBLEM: what sin(θ₁₃)/V_us ratio gives θ₁₃ = 8.54°?")
print("  " + "-" * 60)

sin_t13_obs = sin(theta13_obs * RAD)
ratio_needed = sin_t13_obs / V_us_framework
print(f"    sin(θ₁₃^obs) = {sin_t13_obs:.6f}")
print(f"    sin(θ₁₃)/V_us = {ratio_needed:.6f}")
print(f"    1/√2           = {1/sqrt(2):.6f}")
print(f"    sin(θ₂₃^dark)  = {sin(theta23_dark_exact):.6f}")
print(f"    cos(θ₂₃^dark)  = {cos(theta23_dark_exact):.6f}")
print()
print(f"    The needed ratio {ratio_needed:.6f} is BETWEEN cos(θ₂₃^dk) and 1/√2:")
print(f"      cos(θ₂₃^dk) = {cos(theta23_dark_exact):.6f}")
print(f"      needed       = {ratio_needed:.6f}")
print(f"      1/√2         = {1/sqrt(2):.6f}")
print(f"      sin(θ₂₃^dk) = {sin(theta23_dark_exact):.6f}")

# What angle would give this ratio?
theta23_needed = np.arccos(ratio_needed)
print(f"\n    If the factor is cos(θ), then θ = {theta23_needed*DEG:.4f}°")
print(f"    This is NOT θ₂₃^dark ({theta23_dark_deg:.4f}°)")

theta23_needed_sin = arcsin(ratio_needed)
print(f"    If the factor is sin(θ), then θ = {theta23_needed_sin*DEG:.4f}°")
print(f"    This is NOT θ₂₃^dark either")

# Check: is it cos(θ₂₃^dark - δ) for some small δ?
# cos(θ₂₃ - δ) = cos(θ₂₃)cos(δ) + sin(θ₂₃)sin(δ) ≈ cos(θ₂₃) + δ sin(θ₂₃)
# ratio_needed = cos(θ₂₃) + δ sin(θ₂₃)
# δ = (ratio_needed - cos(θ₂₃)) / sin(θ₂₃)
delta_correction = (ratio_needed - cos(theta23_dark_exact)) / sin(theta23_dark_exact)
print(f"\n    If ratio = cos(θ₂₃^dk - δ): δ = {delta_correction:.6f} rad = {delta_correction*DEG:.4f}°")
print(f"    α₁ = {alpha1:.6f} rad = {alpha1*DEG:.4f}°")
print(f"    Ratio δ/α₁ = {delta_correction/alpha1:.4f}")


# ======================================================================
# SECTION 6: THE CORRECT PHYSICAL PICTURE
# ======================================================================

print()
print("=" * 78)
print("  SECTION 6: SYNTHESIS — THE CORRECT PHYSICAL PICTURE")
print("=" * 78)

print(f"""
  FINDINGS:

  1. The matrix product U_l† × U_ν gives unambiguous results:
     - With pure Cabibbo U_l and TBM U_ν:     θ₁₃ = {t13_TBM:.4f}°
     - With pure Cabibbo U_l and dark U_ν:     θ₁₃ = {t13_dark:.4f}°
     - Observed:                                θ₁₃ = {theta13_obs:.2f}°

  2. The dark correction to θ₂₃ makes θ₁₃ WORSE (farther from obs)
     when U_l is a pure Cabibbo rotation. This is because:
     - U_PMNS(e,3) = V_us × sin(θ₂₃), and sin(θ₂₃) > 1/√2 when θ₂₃ > 45°
     - So the dark-enhanced θ₂₃ pushes θ₁₃ UP, away from the observed value

  3. The cos(θ₂₃) factor would IMPROVE θ₁₃, but requires e↔τ mixing
     as the dominant term. This contradicts the Pati-Salam nearest-generation
     picture where e↔μ (Cabibbo) dominates.

  4. The ratio sin(θ₁₃^obs)/V_us = {ratio_needed:.6f} lies between
     cos(θ₂₃^dk) = {cos(theta23_dark_exact):.6f} and 1/√2 = {1/sqrt(2):.6f}.
     This suggests either:
     (a) The TBM formula sin(θ₁₃) = V_us/√2 is already the best zero-parameter
         result, and the dark correction to θ₂₃ does NOT feed into θ₁₃.
     (b) There is a SECOND dark correction (to V_us itself) that partially
         cancels the θ₂₃ effect.

  5. Self-consistency: The dark perturbation α₁ enters θ₂₃ at first order
     but enters θ₁₃ only at second order (through V_us × sin(θ₂₃)):
       δ(sin θ₁₃) = V_us × (sin(θ₂₃^dk) - 1/√2)
                   = V_us × (sin(45° + 3.7°) - sin(45°))
                   ≈ V_us × 3.7° × cos(45°) × π/180
                   ≈ {V_us_framework * (theta23_dark_deg - 45) * cos(pi/4) * RAD:.6f}
     This is a {V_us_framework * (theta23_dark_deg - 45) * cos(pi/4) * RAD / sin(theta13_obs*RAD) * 100:.1f}% correction to sin(θ₁₃).

  CONCLUSION:
  ───────────

  The TBM formula sin(θ₁₃) = V_us/√2 giving θ₁₃ = {theta13_TBM:.4f}° (4.9% error)
  is the correct zero-parameter prediction. The dark correction to θ₂₃
  does NOT improve θ₁₃ because:

  (a) The physical U_l is a Cabibbo (1-2) rotation → sin(θ₂₃) factor, not cos
  (b) sin(θ₂₃^dark) > 1/√2, pushing θ₁₃ further from observation
  (c) The remaining 4.9% discrepancy requires a DIFFERENT mechanism
      (possibly second-order dark corrections or Majorana phases)

  The dark correction to θ₂₃ and the θ₁₃ prediction are CONSISTENT
  but INDEPENDENT: they come from different sectors (ν vs l) of the
  same srs graph, perturbed at different orders by the dark coupling α₁.
""")

# Final summary
print("=" * 78)
print("  FINAL NUMERICAL SUMMARY")
print("=" * 78)
print()
print(f"  ┌────────────────────────────────┬──────────┬──────────┬────────┐")
print(f"  │ Prediction                     │  Value   │ Observed │ Error  │")
print(f"  ├────────────────────────────────┼──────────┼──────────┼────────┤")
print(f"  │ θ₂₃(dark) = arctan((1+α)/(1-α))│ {theta23_dark_deg:7.3f}° │ {theta23_obs:7.1f}° │ {abs(theta23_dark_deg-theta23_obs)/theta23_obs*100:5.1f}% │")
print(f"  │ θ₁₃(TBM)  = arcsin(V_us/√2)   │ {theta13_TBM:7.3f}° │ {theta13_obs:7.2f}° │ {abs(theta13_TBM-theta13_obs)/theta13_obs*100:5.1f}% │")
print(f"  │ θ₁₃(dark+Cab) = V_us×sin(θ₂₃) │ {theta13_sin:7.3f}° │ {theta13_obs:7.2f}° │ {abs(theta13_sin-theta13_obs)/theta13_obs*100:5.1f}% │")
print(f"  │ θ₁₃(dark+eτ)  = V_us×cos(θ₂₃) │ {theta13_cos:7.3f}° │ {theta13_obs:7.2f}° │ {abs(theta13_cos-theta13_obs)/theta13_obs*100:5.1f}% │")
print(f"  │ θ₁₃(Vus(1-α)/√2)              │ {theta13_suppressed:7.3f}° │ {theta13_obs:7.2f}° │ {abs(theta13_suppressed-theta13_obs)/theta13_obs*100:5.1f}% │")
print(f"  └────────────────────────────────┴──────────┴──────────┴────────┘")
print()
print(f"  STATUS: θ₁₃ = arcsin(V_us/√2) remains the best zero-parameter formula.")
print(f"  The dark-corrected θ₂₃ is self-consistent but does not improve θ₁₃.")
print(f"  The 4.9% residual is a target for second-order corrections.")
