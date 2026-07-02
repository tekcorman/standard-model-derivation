#!/usr/bin/env python3
"""
P2.O1 step 1 — Saha recombination z* shift under framework N-dependent
parameters (2026-05-25).

This is THE bounded first probe of the D3 program per
an internal working note
§5: "Saha z* shift from the N-dependent binding energy alone — E_b(z) ∝
(1+z)^{1/4} raises the ionisation threshold as you go back; solve Saha x_e=½
with E_b(N), fixed vs N-dependent. One equation."

DECISION RULE (per the scoping doc):
  - If z* barely moves → parameter lever is too weak → report negative, stop.
  - If z* moves a lot → proceed to coupled integral.

DECLARED ADOPTIONS (per scoping doc §4, both flagged):
  - A1 (thermal scale): default kinematic T(z) = T_0·(1+z). Native replacement
    via observer-graph energy functional is OPEN.
  - A2 (recombination kinematics form): standard Saha equation for hydrogen
    (extraction-layer form, only parameters are framework-native).

The framework's N-dependent parameters at z ≈ 1089:
  - m_e ∝ N^(-1/4); since N(z) = N_hub/(1+z), m_e(z) = m_e_0 · (1+z)^{1/4}
  - E_b = ½·α²·m_e ∝ m_e ∝ (1+z)^{1/4}  (α is N-invariant)
  - σ_T ∝ 1/m_e²  ∝ (1+z)^{-1/2}        (not in this probe; for next step)

The Saha equation for hydrogen (extraction-layer form):
  x_e² / (1 - x_e) = (1/n_b) · ((m_e·k_B·T)/(2π·ℏ²))^(3/2) · exp(-E_b/(k_B·T))

NO TUNING. The only adoptions are A1 (T ∝ 1/a, kinematic) and A2 (Saha form).
Outcome reported straight.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

print("=" * 76)
print("P2.O1 step 1 — Saha z* under framework N-dependent parameters (2026-05-25)")
print("=" * 76)

# ------------------------------------------------------------------------
# Physical constants (SI)
# ------------------------------------------------------------------------
k_B = 1.380649e-23       # J/K
hbar = 1.054571817e-34   # J·s
c = 2.998e8              # m/s
eV = 1.602176634e-19     # J per eV
m_e_0 = 9.1093837015e-31 # kg, electron mass at z=0
E_b_0 = 13.605693122994 * eV  # J, hydrogen binding energy at z=0
alpha_em_0 = 7.2973525693e-3  # dimensionless (framework: N-invariant)

# CMB / cosmological constants
T_CMB_0 = 2.7255         # K, present CMB temperature
# Baryon density today (using Planck 2018 Ω_b·h² ≈ 0.0224; n_b_0 derived)
# n_b(z) = n_b_0 · (1+z)^3
# Standard value: n_b_0 ≈ 2.5e-7 cm^-3 = 2.5e-1 m^-3
n_b_0 = 2.503e-7 * 1e6   # m^-3  (Planck-derived baryon number density)

print(f"\nInputs (z=0):")
print(f"  T_CMB_0       = {T_CMB_0} K")
print(f"  m_e_0         = {m_e_0:.4e} kg")
print(f"  E_b_0         = 13.606 eV = {E_b_0:.4e} J")
print(f"  n_b_0         = {n_b_0:.3e} m^-3")
print(f"  α (N-invariant) = {alpha_em_0:.5e}")


# ------------------------------------------------------------------------
# Saha equation
# ------------------------------------------------------------------------
def saha_ratio(z, framework_native=False):
    """
    Compute x_e²/(1-x_e) at redshift z under the Saha approximation.

    Standard (framework_native=False): m_e and E_b fixed at z=0 values.
    Framework (framework_native=True):  m_e(z) = m_e_0 · (1+z)^{1/4};
                                         E_b(z) = E_b_0 · (1+z)^{1/4}.
    """
    T = T_CMB_0 * (1 + z)    # A1 adoption: kinematic T ∝ 1/a
    n_b = n_b_0 * (1 + z)**3  # baryon number density (matter conservation)

    if framework_native:
        m_e_z = m_e_0 * (1 + z)**(1/4)
        E_b_z = E_b_0 * (1 + z)**(1/4)
    else:
        m_e_z = m_e_0
        E_b_z = E_b_0

    thermal_de_broglie_inv = math.sqrt(m_e_z * k_B * T / (2 * math.pi * hbar**2))
    prefactor = thermal_de_broglie_inv**3 / n_b
    boltzmann = math.exp(-E_b_z / (k_B * T))
    return prefactor * boltzmann


def find_z_star(framework_native=False, target_x_e=0.5,
                z_search_lo=10.0, z_search_hi=200000.0):
    """Find z* where x_e = target_x_e (half-ionized).
    R(z) is monotone INCREASING with z (hot universe = more ionized).
    """
    target_ratio = target_x_e**2 / (1 - target_x_e)  # = 0.5 for x_e=0.5

    # Bisection: R(z) increasing in z. We want R(z*) = target.
    z_lo, z_hi = z_search_lo, z_search_hi
    # Check that the bracket is valid:
    r_lo, r_hi = saha_ratio(z_lo, framework_native), saha_ratio(z_hi, framework_native)
    if not (r_lo < target_ratio < r_hi):
        return None  # bracket fails — target outside [z_search_lo, z_search_hi]

    for _ in range(200):
        z_mid = (z_lo + z_hi) / 2
        ratio_mid = saha_ratio(z_mid, framework_native)
        if ratio_mid < target_ratio:
            # Too neutral; go HIGHER z (hotter)
            z_lo = z_mid
        else:
            # Too ionized; go LOWER z (cooler)
            z_hi = z_mid
        if z_hi - z_lo < 1e-6:
            break
    return (z_lo + z_hi) / 2


# ------------------------------------------------------------------------
# Compute and compare
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("Saha solutions")
print('='*76)

z_star_std = find_z_star(framework_native=False)
z_star_fw = find_z_star(framework_native=True)

print(f"\n  Standard (fixed m_e, E_b):           z* = {z_star_std:.2f}")
print(f"    (PDG/Planck-consistent: z* ≈ 1089 from Saha; full Peebles ≈ 1100)")
print(f"\n  Framework (m_e, E_b ∝ (1+z)^{{1/4}}):  z* = {z_star_fw:.2f}")

shift = z_star_fw - z_star_std
shift_rel = shift / z_star_std * 100
print(f"\n  Shift (framework - standard):  Δz* = {shift:+.2f}  ({shift_rel:+.2f}%)")

# Compare to the "needed" shift for θ*
# θ* mismatch in coasting was documented at ~10⁵σ. For r_s to match Planck
# θ* ≈ 0.0104 rad, what z* would be needed?
# r_s = ∫c_s/H da; θ* = r_s/D_A. The coasting D_A and H profiles fix a
# linear-ish relationship. Order-of-magnitude: a 10% shift in z* gives a few-%
# shift in r_s, comparable to θ*'s precision requirement.
# But the scoping doc says: success is NOT "θ* matches Planck"; success is a
# straight conditional answer to "does parameter N-dependence move it,
# materially, in the right direction?"

print(f"\n{'='*76}")
print("Diagnostic context")
print('='*76)
print(f"""
A 5.75× shift in E_b at z*≈1089 (per the scoping doc's tabulation) raises the
ionization threshold dramatically.

Interpretation of the sign of Δz*:
  - If Δz* > 0 (framework gives EARLIER recombination, higher z):
    E_b is larger at high z → harder to ionize → x_e drops sooner → larger z*.
  - If Δz* < 0 (framework gives LATER recombination, lower z):
    Saha prefactor (with N-dependent m_e increasing thermal de Broglie phase
    space) dominates → recombination happens later. This would partially
    cancel the binding-energy effect.
""")


# ------------------------------------------------------------------------
# Sensitivity: also report x_e trajectories to show the shift visually
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("x_e(z) trajectory comparison")
print('='*76)

# Solve x_e from Saha ratio: x_e² + R·x_e - R = 0 (R = ratio)
# → x_e = (-R + sqrt(R² + 4R)) / 2
def solve_x_e(z, framework_native=False):
    R = saha_ratio(z, framework_native)
    return (-R + math.sqrt(R*R + 4*R)) / 2

print(f"\n  z      |  x_e (std)  |  x_e (framework)")
print(f"  -------|-------------|-----------------")
# Wider range to capture both standard and framework recombination
for z in [500, 1089, 1200, 1379, 1500, 2000, 5000, 10000, 15365, 20000, 50000, 100000]:
    xe_s = solve_x_e(z, False)
    xe_f = solve_x_e(z, True)
    print(f"  {z:6d} |  {xe_s:8.4f}   |  {xe_f:8.4f}")

print(f"\n  Standard recombination (x_e=½): z* ≈ {z_star_std:.0f}")
print(f"  Framework recombination (x_e=½): z* ≈ {z_star_fw:.0f}")
print(f"  → framework recombination happens ~{z_star_fw/z_star_std:.1f}× EARLIER in z")


# ------------------------------------------------------------------------
# Verdict (per the scoping doc's decision rule)
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("VERDICT (per D3 scoping doc §5 decision rule)")
print('='*76)

# "If z* barely moves" — define "barely" as Δz*/z* < 1%
# "If z* moves a lot" — Δz*/z* > 10%
barely_threshold = 1.0
material_threshold = 10.0

if abs(shift_rel) < barely_threshold:
    verdict = "BARELY MOVES"
    msg = ("Framework's N-dependent E_b is NOT sufficient to materially shift "
           "recombination z*. The parameter lever (this sub-aspect of it) is "
           "too weak. Report negative — but check σ_T effect (next probe step) "
           "before fully closing the lever.")
elif abs(shift_rel) > material_threshold:
    verdict = "MATERIAL SHIFT"
    msg = ("Framework's N-dependent E_b DOES materially shift recombination "
           "z*. Proceed to the coupled Saha+Peebles integral with both E_b(N) "
           "and σ_T(N), and compute r_s, θ*.")
else:
    verdict = "INTERMEDIATE"
    msg = ("Framework gives a modest shift in z*. Worth proceeding to the "
           "coupled probe to see whether σ_T(N) amplifies or cancels the "
           "effect.")

print(f"\n  Shift: Δz*/z*_std = {shift_rel:+.2f}%")
print(f"  Verdict: {verdict}")
print(f"\n  {msg}")

print(f"\n{'='*76}")
print("Conditional on declared adoptions:")
print(f"  A1 (thermal scale): standard kinematic T(z) = T_0·(1+z)")
print(f"  A2 (Saha form):     standard hydrogen Saha equation (extraction-layer)")
print(f"")
print(f"Result is conditional on A1 + A2. If A1 native-replacement (observer-")
print(f"graph energy functional) changes T(z), the answer changes.")
print("=" * 76)
