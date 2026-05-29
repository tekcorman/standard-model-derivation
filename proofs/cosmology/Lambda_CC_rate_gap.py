#!/usr/bin/env python3
"""
proofs/cosmology/Lambda_CC_rate_gap.py

Apply the cascade D2-extended (16/15) rate-gap correction to Λ_CC.

DERIVATION
----------
In coasting cosmology with Ω_Λ = 1/k* = 1/3:

    Λ_CC = 8πG · ρ_Λ = (8πG) · (1/3) · ρ_critical
         = (8πG) · (1/3) · (3 H_0² / 8πG)
         = H_0²

Substrate prediction: Λ_substrate = H_0_substrate²
Observer prediction: Λ_observer = H_0_observer² = (16/15)² · H_substrate²

OBSERVED VALUE
--------------
The "observed" Λ_CC (Planck CMB + SN Ia + BAO ΛCDM fit) is computed in
ΛCDM cosmology with Ω_Λ ≈ 0.685:

    Λ_ΛCDM = 3 H_0² · Ω_Λ ≈ 3 H_0² · 0.685 ≈ 2.055 H_0²

vs. framework's coasting Λ = H_0² · 1 = H_0² (Ω_Λ = 1/3, factor 3 absorbed).

So under SAME H_0 there is a built-in factor ≈ 2 tension between coasting
and ΛCDM-extracted Λ — flagged by `lambda_cc_coasting_scoping.md` and
unchanged by the rate-gap mechanism.

This file disentangles the two pieces:
  (a) Coasting/ΛCDM cosmology mismatch in Ω_Λ definition (factor 2)
  (b) Rate-gap correction (16/15)² ≈ 1.138

(a) is structural to the framework's coasting picture; (b) is the
observer-side correction. Neither closes the Λ_CC tension by itself.

CONCLUSION
----------
The rate-gap mechanism is ORTHOGONAL to the factor-of-2 Λ_CC tension.
Λ_CC remains an open framework problem with the rate-gap mechanism
contributing only a 14% piece of the total ≈100% tension.

This is honest disclosure: the rate-gap closes H_0 (SH0ES) and A_s but
NOT Λ_CC. The Λ_CC factor-of-2 has its own structural origin
(coasting Ω_Λ = 1/3 vs ΛCDM's 0.685) that requires a separate closure.
"""

import math

# Planck units conversions
t_P_s     = 5.391247e-44       # Planck time in seconds
Mpc_in_km = 3.085677581e19     # 1 Mpc in km
yr_in_s   = 3.15576e7

# Framework substrate H_0 (from cascade theorem)
H_0_substrate_kmsMpc = 68.19   # km/s/Mpc
H_0_observer_kmsMpc  = H_0_substrate_kmsMpc * (16.0/15.0)  # = 72.74

# H_0 in inverse seconds
H_0_substrate_per_s = H_0_substrate_kmsMpc / Mpc_in_km
H_0_observer_per_s  = H_0_observer_kmsMpc  / Mpc_in_km

# Λ_CC in Planck units = H_0² × t_P²
Lambda_substrate_Planck = (H_0_substrate_per_s * t_P_s)**2
Lambda_observer_Planck  = (H_0_observer_per_s  * t_P_s)**2

# Observed Λ_CC (Planck 2018 ΛCDM fit, in Planck units)
Lambda_observed_Planck = 1.105e-52 * (1.616e-35)**2  # ≈ 2.89e-122

# Alternatively: Λ_observed = 3 × H_0_Planck² × Ω_Λ_LCDM in (km/s/Mpc)²
H_0_Planck_kmsMpc = 67.4
Omega_Lambda_LCDM = 0.685
Lambda_obs_kmsMpc2 = 3 * H_0_Planck_kmsMpc**2 * Omega_Lambda_LCDM

# Convert observed to Planck units
H_0_Planck_per_s = H_0_Planck_kmsMpc / Mpc_in_km
Lambda_obs_recomputed_Planck = 3 * (H_0_Planck_per_s * t_P_s)**2 * Omega_Lambda_LCDM

print("=" * 72)
print(" Λ_CC under the cascade D2-extended (16/15) rate-gap correction")
print("=" * 72)
print()
print("FRAMEWORK COSMOLOGY (coasting, Ω_Λ = 1/k* = 1/3):")
print(f"  Λ_substrate = H_0_substrate² = ({H_0_substrate_kmsMpc} km/s/Mpc)²")
print(f"             = {Lambda_substrate_Planck:.3e}  (Planck units)")
print(f"  Λ_observer  = (16/15)² · Λ_substrate")
print(f"             = {Lambda_observer_Planck:.3e}  (Planck units)")
print()
print("OBSERVED (ΛCDM cosmology, Ω_Λ = 0.685):")
print(f"  Λ_LCDM     = 3 H_0² Ω_Λ = 3 × {H_0_Planck_kmsMpc}² × {Omega_Lambda_LCDM}")
print(f"             = {Lambda_obs_recomputed_Planck:.3e}  (Planck units)")
print()
print("RATIOS:")
print(f"  Λ_LCDM / Λ_substrate = {Lambda_obs_recomputed_Planck/Lambda_substrate_Planck:.3f}  (factor ~2)")
print(f"  Λ_LCDM / Λ_observer  = {Lambda_obs_recomputed_Planck/Lambda_observer_Planck:.3f}  (factor ~1.76)")
print()
print("TENSION DECOMPOSITION:")
factor_2 = 3 * Omega_Lambda_LCDM  # = 2.055 from coasting/ΛCDM Ω_Λ mismatch
factor_rate = (16.0/15.0)**2       # = 1.138 from rate-gap
print(f"  Factor (a): coasting Ω_Λ = 1/3 vs ΛCDM Ω_Λ = 0.685")
print(f"             3·Ω_Λ_LCDM/Ω_Λ_coast = {factor_2:.3f}")
print(f"  Factor (b): rate-gap (16/15)² = {factor_rate:.4f}")
print(f"  Combined predicted ratio: {factor_2/factor_rate:.3f}")
print()
print("CONCLUSION:")
print(f"  Rate-gap CLOSES {(factor_rate - 1.0) * 100 / (factor_2 - 1.0):.1f}% of the residual.")
print(f"  Remaining factor-of-2 tension is ORTHOGONAL to rate-gap mechanism.")
print(f"  Origin: framework coasting Ω_Λ = 1/k* = 1/3 vs ΛCDM-fit Ω_Λ = 0.685.")
print(f"  This pre-dates the rate-gap analysis (lambda_cc_coasting_scoping.md).")
print()
print("STATUS: rate-gap CONTRIBUTES ~14% to Λ_CC residual, but does NOT close it.")
print("       Full Λ_CC closure requires separate work on coasting/ΛCDM Ω_Λ split.")
