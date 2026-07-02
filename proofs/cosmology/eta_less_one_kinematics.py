#!/usr/bin/env python3
"""
proofs/cosmology/eta_less_one_kinematics.py

NOTE (2026-05-05): The 1-η ≈ 0.066 fitted from SH0ES below received a
structural derivation in commit cdf9c71 as ε_toggle/k = 1/15 = 0.0667
(cascade D2-extended theorem). The "fishing flag" near §candidates below
is now superseded — 1/15 is theorem-grade-derived. See
`docs/theorems/theorem_cascade_D2_extended_observer_rate.md` and
`proofs/cosmology/cascade_observer_rate_gap.py`. This file is retained as
the kinematic-side derivation showing the η<1 → SH0ES connection; the
companion file gives the structural source of the η = 15/16 value.

PATH 2 STEP (b): Kinematics of the η<1 mechanism — what does it actually
predict for observer's H_0 and d_L vs z?

DERIVATION
----------
The cascade theorem with η<1 (data-processing-inequality deficit between
substrate-side D_sub = Λ and observer-side D_obs = η·Λ) modifies the
observer's clock relative to substrate-time.

R2 match equation:
    D_obs(t_obs_now) = 1/N_obs
    => η · Λ(t_obs_now) = 1/N_obs
    => η · t_P / t_obs_now = 1/N_obs    (using Λ(t) = t_P/t convention)
    => t_obs_now = η · N_obs · t_P

Observer's clock T relates to substrate-time t via dT/dt = η.
Observer's "now" in their own clock: T_now = η · t_obs_now = η² · N_obs · t_P.

DISTANCE MODIFICATION
---------------------
Observer's comoving distance integral:
    χ_obs(z) = c · ∫dT/a = c · ∫(η dt/a) = η · χ_sub(z)

Therefore:
    d_L_obs(z) = η · d_L_sub(z)

Observer measures distances smaller than substrate's by factor η.

OBSERVABLE PREDICTIONS
----------------------
1. SH0ES local-Hubble fit (c·z = H · d_L at low z):
       H_inferred = H_sub / η

   For framework H_sub = 68.19 km/s/Mpc and SH0ES H_inferred = 73.04:
       η = 68.19/73.04 = 0.9335
       1 - η = 0.0665 ≈ 7%

2. Pantheon+ d_L(z) shape:
       d_L_obs(z) = η · (c/H_sub)·(1+z)·ln(1+z)
                  = (c/H_inferred)·(1+z)·ln(1+z)

   This is STILL pure coasting shape — just with shifted H_0.
   The coasting-vs-ΛCDM shape mismatch at moderate z is INVARIANT under η.

CONCLUSIONS
-----------
- η<1 is a candidate mechanism for SH0ES H_0 residual (~7% target).
- η<1 does NOT close the moderate-z d_L shape mismatch (shape invariant
  under η rescaling).
- Step (c) becomes load-bearing for moderate-z shape: either SALT2 absorbs
  it via ΛCDM-trained light-curve standardization, or a separate framework
  mechanism is needed (substrate H(z) deviating from coasting).
- Step (a) target magnitude is 1 - η ≈ 0.07 from SH0ES alone.
"""

import math

# Framework prediction (from cascade theorem with η = 1)
H_sub = 68.19  # km/s/Mpc, framework H_0 = 1/t_0 with t_0 = 14.34 Gyr
H_SH0ES = 73.04  # km/s/Mpc, Riess+2022

# Solve for η from SH0ES kinematics
eta_SH0ES = H_sub / H_SH0ES
deviation = 1.0 - eta_SH0ES

# Comparison with framework-natural magnitudes
lambda_W = 0.225  # CKM Wolfenstein
candidates = {
    "λ (CKM Wolfenstein)":     lambda_W,
    "λ²":                      lambda_W**2,
    "λ² × 4/3":                lambda_W**2 * 4/3,
    "1/k* / 5 = 1/15":         1.0/15.0,
    "(2/3)^4·CKM-like factor": (2.0/3.0)**4,
}


def d_L_coasting(z, H0):
    """Coasting luminosity distance (a ∝ t)."""
    c = 2.99792458e5
    return (c / H0) * (1.0 + z) * math.log(1.0 + z)


if __name__ == "__main__":
    print("=" * 70)
    print("  Path 2 Step (b): Kinematics of η<1")
    print("=" * 70)
    print()
    print(f"  Framework substrate H_0 (cascade theorem, η=1):  {H_sub:.2f} km/s/Mpc")
    print(f"  SH0ES inferred H_0 (Riess+2022):                  {H_SH0ES:.2f} km/s/Mpc")
    print()
    print(f"  Observer's d_L(z) = η · d_L_substrate(z)")
    print(f"  SH0ES inferred H_0 = H_substrate / η")
    print()
    print(f"  Solving: η = H_sub/H_SH0ES = {eta_SH0ES:.4f}")
    print(f"  Magnitude target: 1 - η = {deviation:.4f} ≈ {deviation*100:.1f}%")
    print()
    print("  Comparison with framework-natural magnitudes:")
    print(f"  {'expression':28}  {'value':>9}  {'gap from target':>18}")
    for name, val in candidates.items():
        gap = (val - deviation) / deviation * 100
        print(f"  {name:28}  {val:>9.4f}  {gap:>+15.2f}%")
    print()
    print(f"  Note: λ² ≈ 0.051 is closest framework-natural; SH0ES needs ≈ 0.067.")
    print(f"  The 30%-ish gap could come from:")
    print(f"    - additional Yukawa-induced contributions beyond λ²")
    print(f"    - integrated cosmic-time effects")
    print(f"    - the actual Petz-recovery formula not being clean E²")
    print()
    print("  --- Pantheon+ d_L(z) shape check ---")
    print()
    print(f"  d_L_obs(z) under η-mechanism is COASTING SHAPE with shifted H_0.")
    print(f"  Compare to ΛCDM-fit Pantheon+ data demand at moderate z:")
    print()
    print(f"  {'z':>5}  {'coast(H=73)':>12}  {'coast(H=68)':>12}  {'ratio':>8}")
    for z in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]:
        d_obs = d_L_coasting(z, H_SH0ES)
        d_sub = d_L_coasting(z, H_sub)
        print(f"  {z:>5.2f}  {d_obs:>12.1f}  {d_sub:>12.1f}  {d_obs/d_sub:>8.4f}")
    print()
    print(f"  Ratio is constant = η ≈ 0.93 across all z — pure rescaling,")
    print(f"  not a shape change. So the η-mechanism shifts the H_0 calibration")
    print(f"  but leaves d_L(z) in coasting shape. The Pantheon+ shape mismatch")
    print(f"  (coasting vs ΛCDM at moderate z) is unchanged.")
    print()
    print("  CONCLUSION:")
    print("  - η<1 mechanism: candidate for SH0ES (1-η ≈ 0.07 needed)")
    print("  - η<1 mechanism: NOT a candidate for moderate-z shape")
    print("  - Path 2 splits: SH0ES via crack 1; moderate-z separately (step c)")
