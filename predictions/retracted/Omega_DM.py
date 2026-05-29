#!/usr/bin/env python3
"""
RETRACTED 2026-05-04 EOD+3 — moved to predictions/retracted/.

REASON FOR RETRACTION: this file uses observed Ω_b = 0.0493 from PDG/Planck
as a MULTIPLIER in the absolute-scale conversion. The framework derives the
DIMENSIONLESS RATIO Ω_DM/Ω_m = 2/3 structurally (theorem-grade, see
predictions/Omega_DM_over_Omega_m.py — that file STAYS, it is the genuine
prediction). The absolute Ω_DM value computed here multiplies the structural
ratio by the empirical Ω_b. Per the user's zero-empirical-inputs standard
(2026-05-04 EOD+3), this is not a framework prediction — it is the structural
ratio multiplied by an external Planck/BBN observation.

WHAT THE FRAMEWORK DOES PREDICT (clean, theorem-grade):
  Ω_DM/Ω_m = (k*-1)/k* = 2/3
  Ω_DM/Ω_b = r/(1-r) = 2

Both shipped in predictions/Omega_DM_over_Omega_m.py at theorem-grade. This
file's absolute-Ω_DM computation can be reproduced trivially from the ratio
once Ω_b is observed.

PATH FORWARD: a structural derivation of absolute Ω_b from substrate
combinatorics would close this. None currently exists in the framework;
Ω_b lives in baryogenesis territory (η_B is theorem-grade, but its
conversion to Ω_b requires standard cosmology which is anchored to
CMB observation).

LEDGER STATUS: Row P28 Ω_DM should be DOWNGRADED to OPEN-EMPIRICAL with
the structural ratio Ω_DM/Ω_m = 2/3 (predictions/Omega_DM_over_Omega_m.py)
as the genuine framework prediction.

Original docstring follows for historical reference:
=====================================================
Canonical prediction file for Omega_DM (dark matter density parameter).

Omega_DM = Omega_b * (Omega_DM/Omega_m) / (1 - Omega_DM/Omega_m)

where Omega_DM/Omega_m is derived in predictions/Omega_DM_over_Omega_m.py
from the Poisson(2k*) compression argument, and Omega_b is taken from
PDG / BBN measurement (external input).

NOTE (post-2026-04-26 demotion): A2 and A3 are derived theorems; structural
slate is {A1} + P1' + A5-mass per docs/framework/framework_axioms.md §10. The closure
chain referenced here is preserved; only the axiomatic-status labels change.
Omega_b remains an external input; A3-T does not address external physics
inputs. See an internal strict-gating audit Section 9 for updated status.
"""

# ============================================================
# PARAMETER: Omega_DM  (cold dark matter density fraction today)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       0.2645 +/- 0.0050
# Source:      Planck 2018 VI (Planck-only, flat LambdaCDM),
#              Omega_c h^2 = 0.1200 +/- 0.0012, h = 0.674 +/- 0.005
#              Omega_DM = Omega_c h^2 / h^2
#              arXiv:1807.06209
# PDG edition: 2024 (snapshot 2026-04-14)

# --- PREDICTED VALUE -----------------------------------------
# Value:       0.27675  (using predicted ratio 0.8488 and external Omega_b = 0.0493)
# Deviation:   +0.0123 absolute  =  +2.5 sigma (sigma_obs = 0.0050)
#
# The deviation is not a closure failure of the ratio derivation; it is
# dominated by the discrepancy between the pure Poisson(6) prediction
# Omega_DM/Omega_m = 0.8488 and the observed ratio 0.8398 (about 0.6
# sigma on the ratio itself; see predictions/Omega_DM_over_Omega_m.py).

# --- DERIVED FORMULA -----------------------------------------
# Algebraic identity (no physics):
#   Omega_m = Omega_b + Omega_DM
#   r := Omega_DM / Omega_m  =>  Omega_DM = r * (Omega_b + Omega_DM)
#                           =>  Omega_DM (1 - r) = r * Omega_b
#                           =>  Omega_DM = Omega_b * r / (1 - r).
#
# Chain, with gate-clearance citation for each step:
#   1. r = Omega_DM/Omega_m = 1 - P(k <= k* | Poisson(2k*))
#                                        [predictions/Omega_DM_over_Omega_m.py]
#   2. k* = 3                            [predictions/k_star.py]
#      => r = 1 - P(k <= 3 | Poisson(6)) = 1 - e^{-6} sum_{j=0}^{3} 6^j/j!
#                                         = 1 - 61 e^{-6}
#                                         ~= 0.848796.
#   3. Omega_b = 0.0493 +/- 0.0006       [EXTERNAL -- PDG 2024, Planck 2018]
#      Note: this is a measurement; no framework derivation exists.
#   4. Algebra:  Omega_DM = Omega_b * r / (1 - r).
#
# The only [external] input is Omega_b.  All other content is either
# an upstream prediction (item 1, itself depending on k* from item 2)
# or elementary algebra (item 4).

# --- INPUTS --------------------------------------------------
# symbol               | value   | status     | predictions/ file                      | meaning
# ---------------------|---------|------------|----------------------------------------|--------
# Omega_DM_over_Omega_m| 0.84880 | [derived]  | predictions/Omega_DM_over_Omega_m.py   | 1 - P(k<=k* | Poisson(2k*))
# Omega_b              | 0.0493  | [external] | (none -- PDG 2024 / Planck 2018)       | baryon density fraction

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Omega_DM_over_Omega_m import predict_Omega_DM_over_Omega_m
from k_star import predict_k_star
from d_spatial import predict_d_spatial
import functools

# Upstream values
d = predict_d_spatial()
k = predict_k_star(d)
ratio = predict_Omega_DM_over_Omega_m(k)

# External input: PDG 2024 / Planck 2018 baryon density
# (Planck Collaboration 2020, A&A 641 A6, arXiv:1807.06209).
Omega_b_external = 0.0493

# Algebra:  Omega_DM = Omega_b * r / (1 - r)
Omega_DM_value = Omega_b_external * ratio / (1.0 - ratio)

print(f"k*                       = {k}")
print(f"Omega_DM/Omega_m (ratio) = {ratio:.6f}  [derived, from Poisson({2*k}) compression]")
print(f"Omega_b                  = {Omega_b_external}    [EXTERNAL -- PDG 2024 / Planck 2018]")
print(f"Omega_DM = Omega_b * r / (1 - r)")
print(f"         = {Omega_b_external} * {ratio:.6f} / {1-ratio:.6f}")
print(f"         = {Omega_DM_value:.6f}")
print(f"Omega_m  = Omega_b + Omega_DM = {Omega_b_external + Omega_DM_value:.6f}")


# --- PURE FUNCTION -------------------------------------------
# No hardcoded physical constants: ratio r and Omega_b are both
# named parameters.  The function is elementary algebra.

@functools.lru_cache(maxsize=None)
def predict_Omega_DM(Omega_DM_over_Omega_m, Omega_b):
    """
    Compute the dark matter density fraction Omega_DM from the derived
    ratio Omega_DM/Omega_m and the (external) baryon density Omega_b.

    The identity Omega_m = Omega_b + Omega_DM and r = Omega_DM/Omega_m
    imply Omega_DM = Omega_b * r / (1 - r).

    Parameters
    ----------
    Omega_DM_over_Omega_m : float
        The derived ratio r = Omega_DM / Omega_m, in (0, 1).  Supplied
        upstream by predictions/Omega_DM_over_Omega_m.py.
    Omega_b : float
        The baryon density fraction today.  Currently an external input
        from PDG 2024 (Planck 2018); the framework has no closed
        derivation of Omega_b at the time of writing.

    Returns
    -------
    float
        Predicted value of Omega_DM.
    """
    return Omega_b * Omega_DM_over_Omega_m / (1.0 - Omega_DM_over_Omega_m)


# --- VALIDATION ----------------------------------------------

Omega_DM_pred = Omega_DM_value


if __name__ == "__main__":
    impl_result = Omega_DM_value
    pure_result = predict_Omega_DM(ratio, Omega_b_external)
    print(f"\nImplementation: {impl_result:.10f}")
    print(f"Pure function:  {pure_result:.10f}")
    assert abs(impl_result - pure_result) < 1e-12, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    obs = 0.2645
    sigma = 0.0050
    print(f"    Omega_DM = {pure_result:.4f}  "
          f"(obs: {obs} +/- {sigma}, "
          f"deviation {abs(pure_result - obs)/sigma:.1f} sigma)")
    print(f"    Omega_b used: {Omega_b_external} [EXTERNAL, PDG 2024 / Planck 2018]")
    print(f"    Grade: 'mathematically complete' -- Omega_b is not framework-derived.")
