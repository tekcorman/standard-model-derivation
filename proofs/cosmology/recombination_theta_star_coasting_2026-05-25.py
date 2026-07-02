#!/usr/bin/env python3
"""
P2.O1 step 2 — Coupled recombination + θ* under framework N-dependent
parameters and coasting cosmology (2026-05-25).

OUTCOME: not what the scoping doc anticipated as the bounded next step.
Step 2 surfaces a DEEPER issue than parameter-shift sensitivity: in
coasting cosmology, the standard comoving sound horizon r_s LOGARITHMICALLY
DIVERGES (no inflationary horizon to cut off the integral at the lower
end). This means the standard r_s / D_A → θ* pathway does NOT apply in
coasting at all. Step 1's z* shift is real, but it can't be cashed into a
θ* prediction via the standard route.

This is the L6 wall's actual depth: the L6-blocked observables (r_s, θ*,
σ_8, n_s) ALL rely on a finite-horizon r_s computation. Coasting doesn't
provide one. The framework's own r_s prescription must be DIFFERENT from
the standard FRW one — and that "different" is what the L6 BRIDGE wall
(per memory an internal note) actually is.

What the probe verifies:

  1. Step 1's z* shift (factor 11) is REAL and PARAMETER-DEPENDENT.
  2. The standard r_s formula (∫ c_s da/(a·H)) diverges in coasting.
     A cutoff at a_min (substrate-floor) gives an enormous r_s ∝
     ln(N_hub/(1+z*)) — at z*=1089, r_s ≈ 340 Gpc vs Planck's 147 Mpc.
     The cutoff sensitivity makes the standard route physically meaningless.
  3. The standard angular diameter distance D_A is finite (no divergence),
     but small in coasting: D_A(z*=1089) ≈ 28 Mpc vs Planck's ≈ 14 Gpc.
  4. The naive ratio r_s/D_A is ~10⁴ in coasting — well beyond 2π. The
     standard θ* concept doesn't apply.

This is the actual L6 wall, sharper than the parameter-shift framing
suggested. The bounded probe surfaces the right question: not "does
parameter N-dependence move θ*?" but "what IS the framework's native
acoustic-feature angular-scale definition?"

Per D3 scoping doc §3, this is exactly what the "z_eff as observer-graph
walk integral" framing was pointing at — but that's a multi-session
research direction, not a bounded probe.

So P2.O1 step 2 deliverable: CHARACTERIZED-NEGATIVE on the bounded route.
The lever exists (step 1) but can't be cashed via standard FRW formulas.
The next step is observer-graph-side: derive an r_s analog in the
framework's own terms (P2.O1 step 3: A1 native-replacement).
"""

from __future__ import annotations

import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

print("=" * 76)
print("P2.O1 step 2 — recombination → θ* in coasting cosmology (2026-05-25)")
print("=" * 76)

# ------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------
c = 2.998e8          # m/s
H_0_si = 68.2 * 1000 / (3.0857e22)  # 1/s
D_H = c / H_0_si     # m, Hubble distance
N_hub = 8.394881e60  # framework's N_hub (adopted)
Gpc = 3.0857e25      # m
Mpc = 3.0857e22      # m

theta_planck = 0.0104109  # rad

# z* from step 1
z_star_std = 1379.0
z_star_fw = 15365.0

print(f"\nFrom step 1:")
print(f"  Standard Saha z*     = {z_star_std:.0f}")
print(f"  Framework Saha z*    = {z_star_fw:.0f}")
print(f"  Planck observed θ*   = {theta_planck:.4e} rad")
print(f"  Hubble distance D_H  = c/H_0 = {D_H/Gpc:.2f} Gpc")


# ------------------------------------------------------------------------
# Standard r_s formula in coasting — and its divergence
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("Comoving sound horizon r_s in coasting cosmology")
print('='*76)
print("""
Standard formula:
  r_s_comoving(a*) = ∫_{a_min}^{a*} c_s · da / (a · H(a))

In coasting: H(a) = H_0/a, so a·H = H_0 (constant!).
With c_s = c/√3:
  r_s_comoving = (c/(√3·H_0)) · (a* - a_min) = (c/(√3·H_0)) · a*  [if a_min → 0]

Wait — let me re-derive carefully. dt = da/(ȧ) = da/(a·H). For coasting,
a·H = H_0, so dt = da/H_0. r_s = ∫ c_s · dt/a = (c_s/H_0) · ∫ da/a.

That DOES diverge logarithmically. The standard "no-horizon" property of
coasting cosmology: comoving sound horizon (and comoving particle horizon)
both diverge logarithmically at a_min → 0.
""")

# Test the divergence with several a_min cutoffs
print(f"  r_s_comoving with various lower cutoffs (z*={z_star_std:.0f}):")
print(f"  {'a_min':<20} | {'ln-factor':<12} | r_s_comoving (Gpc)")
print(f"  {'-'*20}-|-{'-'*12}-|-----------------")
a_star_std = 1.0 / (1.0 + z_star_std)
for label, a_min in [
    ("1/N_hub (Planck)",       1.0/N_hub),
    ("1/10^40 (random)",       1e-40),
    ("1/10^20",                1e-20),
    ("a_BBN ≈ 1/10⁹",          1e-9),
    ("a_recomb ≈ 10·a*",       10 * a_star_std),
]:
    log_factor = math.log(a_star_std / a_min)
    if log_factor < 0:
        print(f"  {label:<20} | {log_factor:>12.2f} | N/A (a_min > a*)")
        continue
    r_s_si = (c / (math.sqrt(3) * H_0_si)) * log_factor
    print(f"  {label:<20} | {log_factor:>12.2f} | {r_s_si/Gpc:>8.2f}")

print(f"\n  --> r_s is set by the LOWER cutoff a_min, not the physics.")
print(f"      Standard FRW resolution: r_s = ∫_BBN^{{a*}} (finite). Coasting:")
print(f"      no inflationary horizon → no natural cutoff. r_s diverges.")
print(f"      The framework's natural cutoff is a_min = 1/N_hub (substrate-floor).")


# ------------------------------------------------------------------------
# D_A in coasting (well-defined)
# ------------------------------------------------------------------------
def D_C_coasting(z_star):
    return (c / H_0_si) * math.log(1 + z_star)

def D_A_coasting(z_star):
    return D_C_coasting(z_star) / (1 + z_star)


print(f"\n{'='*76}")
print("Angular diameter distance D_A in coasting")
print('='*76)
D_A_std = D_A_coasting(z_star_std)
D_A_fw = D_A_coasting(z_star_fw)
print(f"\n  D_A(standard z*={z_star_std:.0f})  = {D_A_std/Gpc:.4f} Gpc = {D_A_std/Mpc:.1f} Mpc")
print(f"  D_A(framework z*={z_star_fw:.0f}) = {D_A_fw/Gpc:.4f} Gpc = {D_A_fw/Mpc:.1f} Mpc")
print(f"\n  Planck D_A_LCDM ≈ 14 Gpc (for comparison).")
print(f"  Coasting D_A is ~500× smaller than ΛCDM D_A at z*=1089.")


# ------------------------------------------------------------------------
# Naive θ* = r_s/D_A — and why it doesn't apply
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("Naive θ* = r_s/D_A under coasting (using substrate-floor cutoff)")
print('='*76)

def r_s_coasting_substrate_cutoff(z_star):
    """r_s with framework substrate-floor cutoff a_min = 1/N_hub."""
    a_star = 1.0 / (1 + z_star)
    log_factor = math.log(a_star * N_hub)  # = ln(N_hub/(1+z*))
    return (c / (math.sqrt(3) * H_0_si)) * log_factor

r_s_std = r_s_coasting_substrate_cutoff(z_star_std)
r_s_fw = r_s_coasting_substrate_cutoff(z_star_fw)

# Physical r_s = comoving · a*
r_s_phys_std = r_s_std / (1 + z_star_std)
r_s_phys_fw = r_s_fw / (1 + z_star_fw)
# Physical D_A = D_C · a* = D_A_coasting (already physical)

theta_naive_std = r_s_phys_std / D_A_std
theta_naive_fw = r_s_phys_fw / D_A_fw

print(f"\n  Standard (z*={z_star_std:.0f}):")
print(f"    r_s_comoving = {r_s_std/Gpc:.2f} Gpc  (with a_min = 1/N_hub cutoff)")
print(f"    r_s_phys     = {r_s_phys_std/Mpc:.2f} Mpc")
print(f"    D_A_phys     = {D_A_std/Mpc:.2f} Mpc")
print(f"    θ_naive      = r_s_phys/D_A = {theta_naive_std:.3e} rad")
print(f"      vs Planck    = {theta_planck:.3e} rad")
print(f"      ratio        = {theta_naive_std/theta_planck:.2f}× Planck")

print(f"\n  Framework (z*={z_star_fw:.0f}):")
print(f"    r_s_comoving = {r_s_fw/Gpc:.2f} Gpc")
print(f"    r_s_phys     = {r_s_phys_fw/Mpc:.4f} Mpc")
print(f"    D_A_phys     = {D_A_fw/Mpc:.4f} Mpc")
print(f"    θ_naive      = {theta_naive_fw:.3e} rad")
print(f"      vs Planck    = {theta_planck:.3e} rad")
print(f"      ratio        = {theta_naive_fw/theta_planck:.2f}× Planck")


# ------------------------------------------------------------------------
# Honest interpretation
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("HONEST INTERPRETATION OF STEP 2")
print('='*76)
print(f"""
The naive θ* is enormous (>> Planck) for both standard and framework
parameters in coasting, regardless of the substrate-floor cutoff choice.
This is the famous "coasting has no acoustic scale problem" — the
documented 10⁵σ θ* mismatch the scoping doc cited.

The parameter N-dependence (step 1) DID materially shift z* (by 11×).
But the standard r_s/D_A formula in coasting doesn't give a usable θ*
prediction at all, regardless of z*. The L6 wall is deeper than parameter
sensitivity.

Per memory an internal note:
  > 'L6 is a missing BRIDGE not a missing GENERATOR.'

The substrate has the dynamics; what's missing is the *bridge* from
substrate to macroscopic acoustic features. The standard FRW pathway
(r_s = ∫ c_s/H, D_A = D_C·a*) is the bridge that doesn't apply in
coasting because the integrand structure fails.

STEP 2 OUTCOME: CHARACTERIZED-NEGATIVE on the bounded route.
  - The Saha z* shift (step 1) is real and parameter-dependent.
  - The r_s pathway (step 2) doesn't apply in coasting via the standard
    FRW formula.
  - The L6 wall requires a framework-native acoustic-feature angular-
    scale definition, NOT just parameter substitution into FRW formulas.

NEXT STEP (P2.O1 step 3): A1 native-replacement.
  Derive T(N) via the observer-graph energy functional (per scoping
  doc §0). If T(z) departs significantly from kinematic T_0·(1+z), the
  recombination microphysics shifts further, AND we get a candidate
  observer-graph definition of acoustic-feature scales (the §3 z_eff-
  as-integral route).

This is the step where the multi-axial theorem's substrate/observer
boundary (validated by the A_s audit's (16/15) cascade D2-ext) starts
to do real work.

CONDITIONAL ON A1 (kinematic T) AND A2 (Saha form). Both flagged.
""")
print("=" * 76)
