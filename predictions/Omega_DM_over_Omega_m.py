#!/usr/bin/env python3
"""
Canonical prediction file for Omega_DM/Omega_m (dark matter fraction).

RATE-GAP CLASSIFICATION (added 2026-05-05): Ω_DM/Ω_m is a GEOMETRIC
substrate-side ratio: 1 - P(k ≤ k* | Poisson(2k*)) is a combinatorial
fraction at the substrate level, independent of observer-side rates.
Per `docs/theorems/theorem_cascade_D2_extended_observer_rate.md` §3, no
(16/15) correction applies. (Note: the separate Λ_CC factor-of-2
decomposition under ΛCDM observational fitting could in principle reorganize
the matter/dark-energy split — but this affects the matter/Λ partition,
not the within-matter visible/dark partition that Ω_DM/Ω_m measures.)

NOTE (post-2026-04-26 demotion): A2 and A3 are derived theorems; structural
slate is {A1} + P1' + A5-mass per docs/framework/framework_axioms.md §10. The closure
chain referenced here is preserved; only the axiomatic-status labels change.
The Poisson-compression argument is broadly compatible with A3-T's
partial-trace reading of the dark sector, but the "compressibility threshold"
uses Gleason via d_spatial; G.1 and G.5 are now DERIVED via CDP 2011
(predictions/observer_hilbert_space.py).
"""

# ============================================================
# PARAMETER: Omega_DM / Omega_m (dark matter fraction of total matter)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       0.846 ± 0.016
# Source:      Planck 2018, Ω_DM = 0.265, Ω_b = 0.049, Ω_m = 0.315
#              Ω_DM/Ω_m = 0.265/0.315 = 0.841
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       1 - P(k ≤ 3 | Poisson(6)) = 0.8488
# Deviation:   0.5 sigma

# --- DERIVED FORMULA -----------------------------------------
# Omega_DM/Omega_m = 1 - P(k ≤ k* | Poisson(2k*))
#
# Derivation chain:
#   1. k* = 3 (from predictions/k_star.py)
#   2. The raw toggle dynamics at each node involve 2k* = 6 Fock
#      modes (Cl(2k*) = Cl(6): k* creation operators + k* annihilation
#      operators). Each mode toggles independently.
#   3. The degree distribution of the raw (uncompressed) toggle graph
#      is Poisson with mean d = 2k* = 6. This is the maximum-entropy
#      distribution for independent toggles with mean degree 2k*.
#      (Jaynes 1957: max-entropy for fixed mean on {0,1,2,...} is Poisson.)
#   4. The observer's MDL model accepts modes with k ≤ k* = 3 (the
#      compressible sector — Fisher rank ≤ d, Gleason applies).
#      Modes with k > k* are incompressible (dark sector).
#   5. Baryonic (visible) fraction:
#        Ω_b/Ω_m = P(k ≤ 3 | Poisson(6)) = e⁻⁶ Σ_{k=0}^{3} 6^k/k!
#   6. Dark matter fraction:
#        Ω_DM/Ω_m = 1 - P(k ≤ 3 | Poisson(6))
#
#   The Poisson assumption is self-consistent: any correlations
#   in the dark sector would themselves be compressible, shifting
#   modes into the visible sector. The residual is maximally random.
#   (paper_compression_physics.md §11, Remark on Poisson assumption.)

# --- INPUTS --------------------------------------------------
# symbol | value | status    | predictions/ file     | meaning
# -------|-------|-----------|----------------------|--------
# k_star | 3     | [derived] | predictions/k_star.py | compression threshold

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k_star import predict_k_star
from d_spatial import predict_d_spatial
import functools

d = predict_d_spatial()
k = predict_k_star(d)

# Poisson(2k*) CDF at k*
lam = 2 * k  # Poisson mean = 2k* = 6
P_visible = 0.0
for j in range(k + 1):  # j = 0, 1, 2, 3
    P_visible += (lam ** j) / math.factorial(j)
P_visible *= math.exp(-lam)

Omega_DM_over_Omega_m = 1 - P_visible

print(f"k* = {k}")
print(f"Poisson mean: 2k* = {lam}")
print(f"P(k ≤ {k} | Poisson({lam})):")
for j in range(k + 1):
    pj = math.exp(-lam) * lam**j / math.factorial(j)
    print(f"  P({j}) = {pj:.6f}")
print(f"  Sum = {P_visible:.6f}")
print(f"Ω_DM/Ω_m = 1 - {P_visible:.6f} = {Omega_DM_over_Omega_m:.6f}")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_Omega_DM_over_Omega_m(k_star):
    """
    Computes the dark matter fraction of total matter.

    The raw toggle graph has Poisson(2k*) degree distribution.
    The MDL observer compresses modes with k ≤ k* (visible sector).
    The remainder is the dark sector:
      Ω_DM/Ω_m = 1 - P(k ≤ k* | Poisson(2k*))

    Parameters
    ----------
    k_star : int
        Coordination number / compression threshold.

    Returns
    -------
    float
        Dark matter fraction Ω_DM/Ω_m.
    """
    lam = 2 * k_star
    cdf = sum(
        math.exp(-lam) * lam**j / math.factorial(j)
        for j in range(k_star + 1)
    )
    return 1 - cdf


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = Omega_DM_over_Omega_m
    pure_result = predict_Omega_DM_over_Omega_m(k)
    print(f"\nImplementation: {impl_result:.10f}")
    print(f"Pure function:  {pure_result:.10f}")
    assert abs(impl_result - pure_result) < 1e-15
    print("OK: outputs agree.")
    print(f"    Ω_DM/Ω_m = {pure_result:.4f} (obs: 0.842 ± 0.016, {abs(pure_result - 0.842)/0.016:.1f}σ)")
