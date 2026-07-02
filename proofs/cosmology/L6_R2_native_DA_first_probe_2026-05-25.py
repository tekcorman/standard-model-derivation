#!/usr/bin/env python3
"""
L6 closure direction R2 — first probe: native D_A.

Scope: an internal working note §4.

Tests candidate R2a (standard D_A formula in coasting at framework z*)
and characterizes the structural residual against Planck D_A ≈ 14 Gpc.

R2a derivation: in coasting (a ∝ N, a·H = H_0 constant):
    D_C(z) = ∫ c·dt/a = (c/H_0) × ln(1/a) = (c/H_0) × ln(1+z)  [FINITE]
    D_A(z) = D_C(z)/(1+z)

D_C is finite in coasting (no divergence, unlike r_s). But D_A in coasting
is small because (1+z) divides D_C — and D_C grows only logarithmically,
not linearly with comoving distance.
"""

from __future__ import annotations
import math


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
c = 2.998e8
Mpc = 3.0857e22
Gpc = 1000 * Mpc
H_0_si = 68.2 * 1000 / 3.0857e22

c_over_H0 = c / H_0_si       # m, Hubble distance ≈ 4.3 Gpc

# Cosmological anchors
z_star_standard = 1379.0
z_star_framework = 15365.0
D_A_planck_Gpc = 14.0


# ---------------------------------------------------------------------------
# R2a — coasting D_A at framework z*
# ---------------------------------------------------------------------------
print("=" * 72)
print("R2a — coasting D_A = D_C(z*) / (1+z*)")
print("=" * 72)
print(f"\n  D_C(z) = (c/H_0) × ln(1+z) [finite in coasting]")
print(f"  D_A(z) = D_C(z) / (1+z)")
print(f"  Hubble distance c/H_0 = {c_over_H0/Gpc:.3f} Gpc")
print(f"\n  Target: Planck D_A(z*=1089) ≈ {D_A_planck_Gpc} Gpc")
print()
print(f"  {'z*':<25} | {'D_C (Gpc)':>10} | {'D_A (Gpc)':>10} | {'D_A (Mpc)':>10} | {'ratio to Planck':>15}")
print(f"  {'-'*25}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}-|-{'-'*15}")

for label, z_star in [
    ("standard z*=1379", z_star_standard),
    ("framework z*=15366", z_star_framework),
    ("Planck z*=1089 (ΛCDM)", 1089.0),
]:
    log_term = math.log(1 + z_star)
    D_C = c_over_H0 * log_term
    D_A = D_C / (1 + z_star)
    ratio = D_A / (D_A_planck_Gpc * Gpc)
    print(f"  {label:<25} | {D_C/Gpc:>10.3f} | {D_A/Gpc:>10.5f} | {D_A/Mpc:>10.2f} | {ratio:>13.2e}×")


# ---------------------------------------------------------------------------
# What's the structural issue?
# ---------------------------------------------------------------------------
print(f"\n{'='*72}")
print("STRUCTURAL ISSUE — why coasting D_A is small")
print('='*72)
print(f"""
ΛCDM D_A at z*=1089 ≈ {D_A_planck_Gpc} Gpc.
Coasting D_A at z*=1089 = (c/H_0) × ln(1090)/1090 ≈ {c_over_H0/Gpc * math.log(1090)/1090:.3f} Gpc.

Ratio: ΛCDM is ~{D_A_planck_Gpc/(c_over_H0/Gpc * math.log(1090)/1090):.0f}× larger than coasting D_A.

The reason: in ΛCDM with matter+dark-energy domination, D_C grows
ROUGHLY LINEARLY with comoving lookback time, giving D_C(z*) ≈ ~14 Gpc.
In coasting, D_C only grows LOGARITHMICALLY with (1+z), giving D_C(z*)
≈ 31 Gpc (at standard z*). Then division by (1+z*) ≈ 1380 makes D_A
tiny.

This is the same FRW-formula-breakdown finding as for r_s — but it
manifests differently: D_C is finite (no divergence) but the (1+z)
normalization is enormous relative to D_C.

Framework z*=15366 makes it worse (D_A shrinks further by ratio ~ 11).
""")


# ---------------------------------------------------------------------------
# What D_A would close R2 + R1 jointly via θ* ~ 0.01 rad?
# ---------------------------------------------------------------------------
print(f"\n{'='*72}")
print("Would R1 + R2 closure jointly give Planck's θ* = 0.0104 rad?")
print('='*72)
print(f"""
θ* = r_s / D_A (small angle)
Planck observed: θ* = 0.0104 rad
  Planck r_s ≈ 147 Mpc
  Planck D_A ≈ 14 Gpc
  ratio = 147/14000 = 0.0105 ✓ (matches)

If we somehow had R1 closed (r_s_native ≈ 147 Mpc) but R2 stays at
coasting D_A ≈ 22 Mpc (standard z*) → θ* = 147/22 = 6.7 rad. Way over 2π.

So R1 closure alone is INSUFFICIENT — we also need R2 closure (D_A close
to 14 Gpc).

Conversely, if R2 closed (D_A ≈ 14 Gpc) but R1 stays at coasting r_s ≈
34 Gpc → θ* = 34/14 = 2.4 rad. Still way over 2π.

So R1 AND R2 BOTH need closure for θ* to be sensible.
""")


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*72}")
print("HONEST VERDICT — R2a (coasting D_A with native z*)")
print('='*72)
print(f"""
R2a: coasting D_A at standard z* ≈ 22.5 Mpc; at framework z* ≈ 2.7 Mpc.
Both off from Planck's 14 Gpc by factors 600 to 5000.

D_C in coasting is finite (no divergence) but only ~31 Gpc at z*=1379 —
which already exceeds Planck's 14 Gpc D_A. The issue is the division
by (1+z*) — D_A_coasting = D_C/(1+z*) = small.

R2a FAILS by the same FRW-formula-breakdown mechanism as R1a: the
standard formula in coasting gives wrong answers.

JOINT R1+R2 ASSESSMENT:
  Even if R1 closed (r_s_native = 147 Mpc), coasting D_A is too small
  to produce sensible θ*. The "first acoustic peak" Planck measures
  cannot be reconstructed via R1/R2 native replacements as currently
  framed.

STRUCTURAL IMPLICATION:
  R1 and R2 attacking the FRW formulas separately is the WRONG
  decomposition. The L6 wall is not "two formula breakdowns" but ONE
  structural mismatch: ΛCDM's recombination-era physics (RDE→MDE
  transition, finite particle horizon, matter-dominated D_A growth)
  is structurally absent from coasting.

  Solutions must be HOLISTIC: either
  (a) Derive the θ* observable directly from observer-graph dynamics,
      not via separate r_s and D_A. The observable IS the angular scale
      of the first acoustic feature in the observer's filtration — that
      MIGHT be derivable without invoking r_s/D_A separately.
  (b) Accept that coasting doesn't have a Planck-comparable θ*, and
      reframe what observable the framework actually predicts in this
      cluster.

VERDICT: R2a CHARACTERIZED-NEGATIVE.
  Same family as R1a: the standard FRW formula's coasting evaluation
  doesn't match observation. The deeper question is whether r_s, D_A,
  θ* are even the right framework-native observables.
""")
print("=" * 72)
