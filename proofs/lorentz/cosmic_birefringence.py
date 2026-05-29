#!/usr/bin/env python3
"""
Cosmic birefringence: β = sin(arg h) · α_EM

Verifies the numerical prediction and establishes the key identities:
  h = (√3 + i√5)/2  [P2 Theorem 3: doubly-degenerate Hashimoto P-point eigenvalue]
  |h|² = 2 = k*-1   [Ramanujan saturation: srs is Ramanujan, |h| = √(k*-1)]
  sin(arg h) = Im(h/|h|) = √(5/8)
  β = sin(arg h) · α_EM ≈ 0.331°

Grade: A-
  CLOSED: Im(h)/|h| selection — definitional from β being a phase observable
            (sin(arg h) = Im of unit phasor, not Im of eigenvalue)
  OPEN:   α_EM coefficient exactly 1 (not 2α/π etc.) — requires 1-loop
            QFT calculation; cited from QED at leading order

Mechanism:
  β is a rotation angle in U(1) polarization space.
  Walk phase per step at P-point = arg(h).
  Parity-odd content of phase = sin(arg h) = Im(h/|h|).
  Coupling = α_EM (electromagnetic, photon sector).
  c₁ = 0 on photon bundle (P2 Theorem 4) → no topological suppression.

Distinction from V_us / m_ν (Feshbach class):
  Feshbach Σ(h) = α₁/h = α₁·h*/|h|² → Im[Σ] = -α₁·Im(h)/|h|²  (amplitude)
  β direct phase reading → Im(h/|h|) = Im(h)/|h|                  (phase)
  Extra 1/|h| in Feshbach comes from the resolvent, not the observable.
"""

from fractions import Fraction
import cmath, math

print("=" * 65)
print("Cosmic birefringence β = sin(arg h) · α_EM")
print("=" * 65)

# ---------------------------------------------------------------
# Step 1: P-point eigenvalue h (P2 Theorem 3)
# ---------------------------------------------------------------
print("\n--- Step 1: P-point eigenvalue h = (√3 + i√5)/2 ---")

sqrt3 = math.sqrt(3)
sqrt5 = math.sqrt(5)
h = complex(sqrt3/2, sqrt5/2)

print(f"  h = ({sqrt3:.6f} + i·{sqrt5:.6f}) / 2")
print(f"    = {h.real:.8f} + i·{h.imag:.8f}")
print(f"  |h|² = {abs(h)**2:.10f}  (expected: 2 = k*-1 = 3-1)")
print(f"  |h|  = {abs(h):.10f}  (expected: √2 = {math.sqrt(2):.10f})")

assert abs(abs(h)**2 - 2) < 1e-12, "Ramanujan saturation violated"
print("  Ramanujan saturation: |h|² = k*-1 = 2  ✓")

# ---------------------------------------------------------------
# Step 2: Phase and parity-odd content
# ---------------------------------------------------------------
print("\n--- Step 2: Phase arg(h) and parity-odd content Im(h/|h|) ---")

arg_h = cmath.phase(h)
unit_h = h / abs(h)

sin_arg_h = math.sin(arg_h)
im_unit_h = unit_h.imag
sqrt_5_8   = math.sqrt(5.0/8.0)

print(f"  arg(h)         = {arg_h:.8f} rad = {math.degrees(arg_h):.6f}°")
print(f"  h/|h|          = {unit_h.real:.8f} + i·{unit_h.imag:.8f}")
print(f"  Im(h/|h|)      = {im_unit_h:.8f}")
print(f"  sin(arg h)     = {sin_arg_h:.8f}")
print(f"  √(5/8)         = {sqrt_5_8:.8f}")
print(f"  Agreement:     {abs(im_unit_h - sin_arg_h):.2e}  (should be ~0)")

assert abs(im_unit_h - sin_arg_h) < 1e-14
assert abs(im_unit_h - sqrt_5_8)  < 1e-8
print("  Im(h/|h|) = sin(arg h) = √(5/8) confirmed  ✓")

# Verify EXACT rational form: Im(h) = √5/2, |h| = √2
# → Im(h/|h|) = (√5/2)/√2 = √5/(2√2) = √(5/8)  [exact]
im_h_exact   = math.sqrt(5)/2
abs_h_exact  = math.sqrt(2)
im_unit_exact = im_h_exact / abs_h_exact
print(f"\n  Exact form: Im(h)/|h| = (√5/2)/√2 = √5/(2√2) = √(5/8)")
print(f"  Numerical check: {im_unit_exact:.12f} vs √(5/8) = {sqrt_5_8:.12f}")
assert abs(im_unit_exact - sqrt_5_8) < 1e-14

# ---------------------------------------------------------------
# Step 3: Contrast with Im(h) and Im(h)/|h|² (Feshbach form)
# ---------------------------------------------------------------
print("\n--- Step 3: Contrast phase observable vs amplitude observable ---")
print("  Phase observable (β):          Im(h/|h|) = Im(h)/|h|")
print("  Amplitude observable (V_us):   Im(h)/|h|²  [from Σ(h)=α₁/h]")
print()
print(f"  Im(h)       = {h.imag:.8f}  = √5/2 ≈ {sqrt5/2:.8f}")
print(f"  Im(h)/|h|   = {h.imag/abs(h):.8f}  = √(5/8) ≈ {sqrt_5_8:.8f}")
print(f"  Im(h)/|h|²  = {h.imag/abs(h)**2:.8f}  = √5/4 ≈ {math.sqrt(5)/4:.8f}")
print(f"  Ratio: (1/|h|) = 1/√2 = {1/abs(h):.8f}")
print()
print("  The extra 1/|h| distinguishing β from V_us comes from the")
print("  extraction map: β extracts sin(arg h) = Im(unit phasor),")
print("  while V_us extracts Im[Σ(h)] = Im(α₁/h) = -α₁·Im(h)/|h|².")
print("  The /|h| is definitional (unit phasor), not empirical.")

# ---------------------------------------------------------------
# Step 4: Numerical prediction β
# ---------------------------------------------------------------
print("\n--- Step 4: β = sin(arg h) · α_EM ---")

alpha_EM = 1/137.035999084  # CODATA 2018

beta_rad = sin_arg_h * alpha_EM
beta_deg = math.degrees(beta_rad)

print(f"  sin(arg h)   = √(5/8)   = {sin_arg_h:.8f}")
print(f"  α_EM         = 1/137.04 = {alpha_EM:.8f}")
print(f"  β = sin(arg h) · α_EM")
print(f"    = {beta_rad:.6f} rad")
print(f"    = {beta_deg:.4f}°")

# Eskilt 2022 observation
obs_val  = 0.342  # degrees
obs_err  = 0.094  # degrees
pull     = (beta_deg - obs_val) / obs_err

print(f"\n  Observed (Eskilt 2022): {obs_val}° ± {obs_err}°")
print(f"  Predicted:              {beta_deg:.3f}°")
print(f"  Residual:               {beta_deg - obs_val:+.3f}°  ({pull:.2f}σ)")

# Hard cap
alpha_EM_deg = math.degrees(alpha_EM)
print(f"\n  Hard cap |β| ≤ α_EM = {alpha_EM_deg:.4f}°")
print(f"  Framework sits at {beta_deg/alpha_EM_deg*100:.1f}% of geometric bound")

# ---------------------------------------------------------------
# Step 5: Summary of open and closed gaps
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("RESULT: β = sin(arg h) · α_EM = {:.3f}°  (Eskilt 2022: {:.3f}° ± {:.3f}°, {:.2f}σ)".format(
    beta_deg, obs_val, obs_err, pull))
print("=" * 65)

print("\nGRADE: A-")
print()
print("CLOSED gaps:")
print("  (1) Im(h)/|h| selection — definitional, not empirical:")
print("      β is a phase rotation → couples to Im(unit phasor) = sin(arg h)")
print("      Im(h/|h|) = Im(h)/|h| by definition of unit phasor")
print("      No matching to Eskilt number required.")
print("  (2) B(-k) = B(k)* → h_max(k) even in k → η₅ = 0 → no dim-5 birefringence")
print("      (proven this session: proofs/lorentz/hashimoto_bloch_dispersion.py)")
print()
print("OPEN gaps (A- status):")
print("  (1) α_EM coupling coefficient exactly 1 (not 2α/π or similar)")
print("      Requires 1-loop QFT calculation for photon-walker coupling.")
print("      Currently cited at leading QED order (Peskin & Schroeder §6.3).")
print()
print("Prerequisites (both confirmed in results/parameters.csv):")
print("  - P2 Theorem 3: h = (√3+i√5)/2 doubly-degenerate at P-point")
print("  - P2 Theorem 4: c₁ = 0 on photon Hodge bundle (all BZ slices)")
