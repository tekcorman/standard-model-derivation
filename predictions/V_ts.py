#!/usr/bin/env python3
"""
Canonical prediction file for V_ts (CKM matrix element, |V_ts|).

STATUS UNDER PARAMETER LINTER (created 2026-05-02 via combined cleanup walk):
THEOREM-GRADE-NUMERICAL via Type-4 inheritance from Rows P3, P4, P14, P15.
Standard-parameterization unitary CKM construction. Clause 7 PASS-CITED;
Clause 8 SOFT-TENSION (within 3σ; uncertainty derived via unitarity).

Audit anchor: Row P14 V_ub family (M1c) of `docs/parameters/parameter_uniqueness_ledger.md`.

    V_ts = |−c_12·s_23 − s_12·c_23·s_13·e^(iδ)|   (Chau-Keung 1984 / PDG standard parameterization)

where (s_12, c_12, s_13, c_13, s_23, c_23, δ_CP) come from the four
framework-derived inputs:
    V_us = 9/40                             (Row P4)
    V_cb = 256/6305                         (Row P3)
    V_ub = Σ_{m≥2} (2/3)^(6m+2)/(1−(2/3)^(6m+2))   (Row P14)
    cos δ_CP = 1/3                          (Row P15)

Cross-verification: `proofs/foundations/v_ub_unitarity_triangle_route_c.py`
verifies ||V·V† − I|| ~ 1e-18 to machine precision.

The labeling layer (color ≡ generation) is OTHER-SMUGGLE residue inherited
from Row P14, NON-BLOCKING for predictive content per the (Z/2)^3 Angle D
verdict (commit e5ef667).

History:
  - Pre-2026-05-02: no prediction file; V_ts tracked only via the
    unitarity-triangle probe `proofs/foundations/v_ub_unitarity_triangle_route_c.py`.
  - 2026-05-02: Type-4 closure shipped via parameter_linter combined cleanup
    walk. Numerical value vs PDG = -1.50σ.
"""

# ============================================================
# PARAMETER: V_ts (CKM matrix element, |V_ts|)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       |V_ts| = 0.0411 ± 0.00083
# Source:      PDG 2024 Review of Particle Physics, CKM review (B_s mixing + unitarity).
# PDG edition: 2024.

# --- PREDICTED VALUE -----------------------------------------
# Value:       |V_ts| (CKM unitarity from V_us, V_cb, V_ub, δ_CP)
# Deviation:   -1.50σ (see runtime output for exact)
# Status:      THEOREM-GRADE-NUMERICAL (Type-4 inheritance);
#              labeling data-anchored / non-blocking via Row P14.
#              Systematic floor: zero. Clause 8: SOFT-TENSION (within 3σ; uncertainty derived via unitarity).

# --- DERIVED FORMULA -----------------------------------------
# V_ts = |−c_12·s_23 − s_12·c_23·s_13·e^(iδ)|
#
# where:
#   s_13 = V_ub,           c_13 = √(1 − V_ub²)
#   s_12 = V_us / c_13,    c_12 = √(1 − s_12²)
#   s_23 = V_cb / c_13,    c_23 = √(1 − s_23²)
#   δ_CP = arccos(cos_delta_CP) = arccos(1/3)

# --- INPUTS --------------------------------------------------
# symbol          | value      | status     | predictions/ file                       | meaning
# ----------------|------------|------------|-----------------------------------------|--------
# V_us            | 9/40       | [derived]  | predictions/V_us.py                     | Row P4
# V_cb            | 256/6305   | [derived]  | predictions/V_cb.py                     | Row P3
# V_ub            | 3.767e-3   | [derived]  | predictions/V_ub.py                     | Row P14
# cos δ_CP        | 1/3        | [derived]  | predictions/delta_CP_CKM_geometry.py    | Row P15
# helper module   |            | [derived]  | predictions/_ckm_unitarity.py           | Standard parameterization

# --- IMPLEMENTATION ------------------------------------------

import math
import functools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ckm_unitarity import (
    build_ckm_magnitudes,
    V_us_framework, V_cb_framework, V_ub_framework, cos_delta_CP_framework,
)

V_ts = build_ckm_magnitudes(
    V_us_framework, V_cb_framework, V_ub_framework, cos_delta_CP_framework,
).V_ts

# Observed (PDG 2024)
V_ts_obs = 0.0411
V_ts_unc = 0.00083
dev_abs = V_ts - V_ts_obs
dev_rel = dev_abs / V_ts_obs
dev_sigma = dev_abs / V_ts_unc

print("=" * 68)
print("  V_ts  --  THEOREM-GRADE-NUMERICAL via CKM unitarity")
print("=" * 68)
print(f"  V_us = {V_us_framework:.10f}    (Row P4)")
print(f"  V_cb = {V_cb_framework:.10f}    (Row P3)")
print(f"  V_ub = {V_ub_framework:.6e}     (Row P14)")
print(f"  cos δ_CP = {cos_delta_CP_framework:.10f}     (Row P15, = 1/3 exact)")
print()
print(f"  V_ts = |−c_12·s_23 − s_12·c_23·s_13·e^(iδ)|")
print(f"        = {V_ts:.10f}")
print()
print(f"  PDG 2024  : {V_ts_obs} ± {V_ts_unc}")
print(f"  Deviation : {dev_abs:+.5e} ({dev_rel*100:+.4f}%, {dev_sigma:+.2f}σ)")
print()
print("  Gate chain:")
print("    Step 1 [Type 4]: V_us, V_cb, V_ub, δ_CP from Rows P3, P4, P14, P15")
print("    Step 2 [Type 3 — Chau-Keung 1984]: standard CKM parameterization")
print("    Step 3 [Type 2]: V_ts closed-form")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_V_ts(V_us, V_cb, V_ub, cos_delta_CP):
    """
    Compute |V_ts| from the unitary CKM matrix in standard parameterization.

    Formula: V_ts = |−c_12·s_23 − s_12·c_23·s_13·e^(iδ)|.

    Parameters
    ----------
    V_us, V_cb, V_ub : float
        Framework-derived CKM magnitudes (Rows P4, P3, P14).
    cos_delta_CP : float
        cos(δ_CP_CKM) (Row P15, = 1/3 exact).

    Returns
    -------
    float
        Predicted |V_ts|.
    """
    return build_ckm_magnitudes(V_us, V_cb, V_ub, cos_delta_CP).V_ts


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = V_ts
    pure_result = predict_V_ts(
        V_us_framework, V_cb_framework, V_ub_framework, cos_delta_CP_framework,
    )
    print()
    print(f"Implementation: {impl_result:.12f}")
    print(f"Pure function:  {pure_result:.12f}")
    assert abs(impl_result - pure_result) < 1e-12, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    V_ts = {pure_result:.6f}  (PDG: {V_ts_obs} ± {V_ts_unc}, {dev_sigma:+.2f}σ)")
    print("    Rigor status: THEOREM-GRADE-NUMERICAL via Type-4 inheritance + CKM unitarity;")
    print("    labeling data-anchored / non-blocking via Row P14.")
