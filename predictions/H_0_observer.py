#!/usr/bin/env python3
"""
H_0_observer — the framework's observer-side Hubble constant, tested
against the local distance-ladder (SH0ES) measurement.

Observable-side sibling of predictions/H_0.py, the H_0 analog of the
predictions/t_0.py / t_0_LCDM.py split and the established
Lambda_CC.py / Lambda_CC_LCDM.py convention:

  - predictions/H_0.py          : H_0_substrate ≈ 68.2 km/s/Mpc from the
                                  coasting identity H_0·t_0 = 1. Tested
                                  against the Planck CMB/ΛCDM value
                                  (67.4 ± 0.5) — the framework's Clause-8
                                  Category-B CMB-side anchor — at +1.6σ.

  - predictions/H_0_observer.py : THIS FILE. The framework's OBSERVER-side
                                  prediction H_0_observer = (16/15) ×
                                  H_0_substrate ≈ 72.7 km/s/Mpc, tested
                                  against the SH0ES distance-ladder value
                                  (73.04 ± 1.04, Riess et al. 2022).

The (16/15) rate gap is the D2-extended observer/substrate split
(RATE_GAP = ε_toggle·(1/k*) = (1/5)(1/3) = 1/15). The "Hubble tension"
is, in the framework, a STRUCTURAL PREDICTION, not an anomaly: the
substrate-side H_0 sits at the CMB value, the observer-side H_0 sits at
the distance-ladder value, and the gap between them is exactly the
predicted (16/15). This file makes that prediction an explicit,
first-class, visible row — symmetric with t_0_LCDM.py.
"""

# ============================================================
# PARAMETER: H_0 (observer-side) — local distance-ladder Hubble constant
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       73.04 ± 1.04 km/s/Mpc
# Source:      Riess et al. 2022, ApJL 934, L7 (SH0ES Cepheid–SN Ia
#              distance ladder).
# Note:        Local, late-time, largely model-independent (does NOT
#              assume ΛCDM or the sound horizon). This is the
#              observer-side anchor for the framework's H_0 prediction.

# --- PREDICTED VALUE -----------------------------------------
# Value:       H_0_observer = (16/15) × H_0_substrate ≈ 72.7 km/s/Mpc
# Deviation:   ≈ +0.3σ vs SH0ES — the framework's predicted resolution
#              of the Hubble tension PASSES the distance-ladder anchor.
#              (The substrate side simultaneously matches Planck-CMB at
#              +1.6σ via predictions/H_0.py — both sides accounted for.)

# --- DERIVED FORMULA -----------------------------------------
# H_0_observer = H_0_substrate × (1 + RATE_GAP),  RATE_GAP = (1/5)(1/k*)
#
# Logical chain:
#   Step 1: H_0_substrate from predictions/H_0.py (coasting H_0·t_0 = 1;
#           Type 4 upstream-closed file).
#   Step 2: D2-extended observer-rate correction: an observer measures
#           the toggle rate boosted by RATE_GAP = ε_toggle·(1/k*) =
#           (1/5)(1/3) = 1/15, so H_0_observer = (16/15)·H_0_substrate
#           (theorem_cascade_D2_extended_observer_rate.md).
#   Step 3: The Hubble tension = the (16/15) substrate↔observer gap is a
#           STRUCTURAL PREDICTION: substrate ≈ Planck-CMB, observer ≈
#           SH0ES distance ladder.

# --- INPUTS --------------------------------------------------
# symbol           | value    | status    | predictions/ file    | meaning
# -----------------|----------|-----------|----------------------|--------
# H_0_substrate    | ≈ 68.2   | [derived] | predictions/H_0.py   | coasting H_0 (km/s/Mpc)
# rate_gap         | 1/15     | [derived] | predictions/H_0.py   | ε_toggle·(1/k*) observer-rate boost

# --- IMPLEMENTATION ------------------------------------------

import functools
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from H_0 import H_0_pred as _H_0_substrate, RATE_GAP as _RATE_GAP  # Type-4


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_H_0_observer(H_0_substrate, rate_gap):
    """
    Predict the observer-side (distance-ladder) Hubble constant.

    H_0_observer = H_0_substrate × (1 + rate_gap)

    Parameters
    ----------
    H_0_substrate : float
        Coasting substrate-side H_0 in km/s/Mpc (predictions/H_0.py).
    rate_gap : float
        D2-extended observer-rate boost ε_toggle·(1/k*) = 1/15.

    Returns
    -------
    float
        Predicted observer-side H_0 in km/s/Mpc.
    """
    return H_0_substrate * (1.0 + rate_gap)


# --- Runner-facing scalars (slug = "H_0_observer") -----------
H_0_observer_pred  = predict_H_0_observer(_H_0_substrate, _RATE_GAP)
H_0_observer_obs   = 73.04     # km/s/Mpc  [Riess et al. 2022, SH0ES]
H_0_observer_sigma = 1.04      # km/s/Mpc
dev_abs   = H_0_observer_pred - H_0_observer_obs
dev_rel   = dev_abs / H_0_observer_obs
dev_sigma = dev_abs / H_0_observer_sigma

print("=" * 72)
print("  H_0 (observer-side)  --  framework's predicted Hubble-tension split")
print("=" * 72)
print(f"  H_0 substrate (predictions/H_0.py) = {_H_0_substrate:.4f} km/s/Mpc")
print(f"  RATE_GAP = ε_toggle·(1/k*)         = {_RATE_GAP:.6f}  (= 1/15)")
print(f"  H_0 observer = (16/15)·substrate   = {H_0_observer_pred:.4f} km/s/Mpc")
print(f"  SH0ES distance ladder (Riess 2022) = {H_0_observer_obs:.2f} ± {H_0_observer_sigma:.2f}")
print(f"  Deviation                          = {dev_abs:+.4f} "
      f"({dev_rel*100:+.2f}%, {dev_sigma:+.2f}σ)")
print()
print("  The substrate side simultaneously matches Planck-CMB at +1.6σ")
print("  (predictions/H_0.py). The (16/15) substrate↔observer gap IS the")
print("  framework's structural prediction of the Hubble tension.")


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl = _H_0_substrate * (1.0 + _RATE_GAP)
    pure = predict_H_0_observer(_H_0_substrate, _RATE_GAP)
    assert abs(impl - pure) < 1e-12, f"Mismatch: {impl} vs {pure}"
    assert abs(dev_sigma) < 1.0, f"Expected SH0ES PASS (<1σ), got {dev_sigma:+.2f}σ"
    print()
    print("OK: outputs agree; observer-side H_0 PASSES the SH0ES anchor.")
