#!/usr/bin/env python3
"""
RETIRED — redirect to predictions/delta_CP_CKM_geometry.py.

This file was the pre-A3 two-axiom derivation of the CKM CP phase. The
original derivation invoked B3 sector-universality, which made
V_us = V_cb = V_ub = 0 and killed the Jarlskog invariant identically.
That reading was retired 2026-04-25 / 2026-04-26 along with the B3
sector-universality argument; the canonical keeper is now:

    predictions/delta_CP_CKM_geometry.py

which derives δ_CP_CKM = arccos(1/3) ≈ 70.5288° from the regular-
tetrahedron dihedral angle of the K_4 (−1)-eigenspace at Γ.

This file is preserved as a redirect-tombstone so that:
  - the predictions DAG validator continues to accept the predictions/
    directory as a self-contained set;
  - any external reference to `predictions/delta_CP_CKM.py` resolves
    cleanly to the keeper rather than a stale derivation;
  - the historical retirement is documented in-line for archaeology.

Do NOT add new logic here. New work goes in delta_CP_CKM_geometry.py.

Audit anchor: Row P15 of `docs/parameters/parameter_uniqueness_ledger.md`. Status
is currently STRICT-SOLID conditional on ADOPTED-A5b-Sub3 (un-graduated
post 2026-04-29 retraction); see an internal working note.

Update 2026-04-29: rewritten as redirect-tombstone (was sentinel-stub
that still computed arccos(1/3), creating two scripts predicting the
same value — undesirable per parameter_linter §"On inconsistencies").
"""

# ============================================================
# REDIRECT TOMBSTONE — see delta_CP_CKM_geometry.py
# ============================================================

import sys
import os
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from delta_CP_CKM_geometry import predict_delta_CP_CKM_geometry


@functools.lru_cache(maxsize=None)
def predict_delta_CP_CKM(k_star):
    """
    RETIRED — delegates to delta_CP_CKM_geometry.predict_delta_CP_CKM_geometry.

    The original B3-sector-universality derivation in this file was
    retired 2026-04-25 / 2026-04-26. The canonical keeper is
    `predictions/delta_CP_CKM_geometry.py` which derives δ_CP_CKM as
    the regular-tetrahedron dihedral angle of the K_4 (−1)-eigenspace
    at Γ.

    Parameters
    ----------
    k_star : int
        Coordination number (= 3 for srs).

    Returns
    -------
    float
        δ_CP_CKM in degrees, delegating to the keeper.
    """
    return predict_delta_CP_CKM_geometry(k_star)


delta_CP_CKM_pred = predict_delta_CP_CKM(3)


if __name__ == "__main__":
    import math
    print("=" * 68)
    print("  delta_CP_CKM.py — RETIRED redirect tombstone")
    print("=" * 68)
    print()
    print("  This file is the pre-A3 two-axiom derivation, retired 2026-04-25.")
    print("  Canonical keeper: predictions/delta_CP_CKM_geometry.py")
    print()
    print("  Delegating to the keeper for backward compatibility:")
    result = predict_delta_CP_CKM(3)
    print(f"    delta_CP_CKM = {result:.4f}°  (from delta_CP_CKM_geometry.py)")
    print()
    print("  See docs/parameters/parameter_uniqueness_ledger.md Row P15 for canonical status.")
    print("  See an internal working note")
    print("  for the current STRICT-SOLID-conditional grade (post bridge-lemma retraction).")
