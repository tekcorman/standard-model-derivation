#!/usr/bin/env python3
"""
Shared helper for unitarity-derived CKM elements.

Provides the standard-parameterization 3x3 CKM construction from the four
framework-derived inputs (V_us, V_cb, V_ub, cos δ_CP_CKM), used by the six
unitarity-derived prediction files:

    predictions/V_ud.py    predictions/V_cs.py    predictions/V_tb.py
    predictions/V_cd.py    predictions/V_ts.py    predictions/V_td.py

This is NOT a standalone prediction file — it has no `predict_*` function
of its own and no observed value comparison. Each `V_ij.py` calls
`build_ckm_matrix(...)` with the framework-derived inputs and reads off
the appropriate magnitude.

Audit anchor: shared inheritance for Rows P3, P4, P14, P15 → unitarity
construction. The standard parameterization is (Chau-Keung 1984;
PDG CKM review):

    V_ud = c_12 c_13                                    (1,1)
    V_us = s_12 c_13                                    (1,2)
    V_ub = s_13 e^{−iδ}                                 (1,3)
    V_cd = −s_12 c_23 − c_12 s_23 s_13 e^{iδ}           (2,1)
    V_cs = c_12 c_23 − s_12 s_23 s_13 e^{iδ}            (2,2)
    V_cb = s_23 c_13                                    (2,3)
    V_td = s_12 s_23 − c_12 c_23 s_13 e^{iδ}            (3,1)
    V_ts = −c_12 s_23 − s_12 c_23 s_13 e^{iδ}           (3,2)
    V_tb = c_23 c_13                                    (3,3)

with positive-root branch:
    s_13 = V_ub,            c_13 = sqrt(1 − V_ub²)
    s_12 = V_us / c_13,     c_12 = sqrt(1 − s_12²)
    s_23 = V_cb / c_13,     c_23 = sqrt(1 − s_23²)
    δ_CP = arccos(cos_delta_CP)

Cross-verification: `proofs/foundations/v_ub_unitarity_triangle_route_c.py`
verifies ||V·V† − I|| ~ 1e-18 to machine precision (unitary by construction
in the standard parameterization).

Status: helper module, non-prediction-file. All six V_ij.py files that
import this module ship at THEOREM-GRADE-NUMERICAL or
THEOREM-GRADE-NUMERICAL with disclosed soft tension via Type-4 inheritance.
"""

import math
import functools
import sys
import os
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from V_us import predict_V_us, k_star as _k_us, g as _g_us, N_ATOMS as _N_us
from V_cb import predict_V_cb, k as _k_cb, g as _g_cb, n_fixed as _nfix_cb
from V_ub import predict_V_ub, k as _k_ub, g as _g_ub, s_seam as _s_ub, n_fixed as _nfix_ub, m_max as _m_ub
from delta_CP_CKM_geometry import predict_delta_CP_CKM_geometry, k as _k_dcp


class CKMMagnitudes(NamedTuple):
    V_ud: float
    V_us: float
    V_ub: float
    V_cd: float
    V_cs: float
    V_cb: float
    V_td: float
    V_ts: float
    V_tb: float


@functools.lru_cache(maxsize=None)
def build_ckm_magnitudes(V_us: float, V_cb: float, V_ub: float,
                         cos_delta_CP: float) -> CKMMagnitudes:
    """
    Construct the 3x3 CKM matrix magnitudes from four framework-derived inputs.

    Uses the PDG standard parameterization (Chau-Keung 1984):
        s_13 = V_ub,           c_13 = sqrt(1 − V_ub²)
        s_12 = V_us / c_13,    c_12 = sqrt(1 − s_12²)
        s_23 = V_cb / c_13,    c_23 = sqrt(1 − s_23²)
        δ_CP = arccos(cos_delta_CP)

    Each output is the absolute value of the corresponding CKM matrix element.
    The matrix is unitary by construction.

    Parameters
    ----------
    V_us : float
        |V_us| (Row P4, framework-derived).
    V_cb : float
        |V_cb| (Row P3, framework-derived).
    V_ub : float
        |V_ub| (Row P14, framework-derived).
    cos_delta_CP : float
        cos(δ_CP_CKM) (Row P15, framework-derived = 1/3 exact).

    Returns
    -------
    CKMMagnitudes
        Named tuple of all 9 |V_ij| values.
    """
    s_13 = V_ub
    c_13 = math.sqrt(1 - s_13**2)
    s_12 = V_us / c_13
    c_12 = math.sqrt(1 - s_12**2)
    s_23 = V_cb / c_13
    c_23 = math.sqrt(1 - s_23**2)

    delta = math.acos(cos_delta_CP)
    cos_d = math.cos(delta)
    sin_d = math.sin(delta)

    # Row 1: V_ud, V_us, V_ub
    V_ud = c_12 * c_13
    # Row 2: V_cd, V_cs, V_cb
    # |V_cd|² = (s_12 c_23 + c_12 s_23 s_13 cos_d)² + (c_12 s_23 s_13 sin_d)²
    re_cd = -s_12 * c_23 - c_12 * s_23 * s_13 * cos_d
    im_cd = -c_12 * s_23 * s_13 * sin_d
    V_cd_mag = math.sqrt(re_cd**2 + im_cd**2)
    re_cs = c_12 * c_23 - s_12 * s_23 * s_13 * cos_d
    im_cs = -s_12 * s_23 * s_13 * sin_d
    V_cs_mag = math.sqrt(re_cs**2 + im_cs**2)
    # Row 3: V_td, V_ts, V_tb
    re_td = s_12 * s_23 - c_12 * c_23 * s_13 * cos_d
    im_td = -c_12 * c_23 * s_13 * sin_d
    V_td_mag = math.sqrt(re_td**2 + im_td**2)
    re_ts = -c_12 * s_23 - s_12 * c_23 * s_13 * cos_d
    im_ts = -s_12 * c_23 * s_13 * sin_d
    V_ts_mag = math.sqrt(re_ts**2 + im_ts**2)
    V_tb = c_23 * c_13

    return CKMMagnitudes(
        V_ud=V_ud,
        V_us=s_12 * c_13,
        V_ub=s_13,
        V_cd=V_cd_mag,
        V_cs=V_cs_mag,
        V_cb=s_23 * c_13,
        V_td=V_td_mag,
        V_ts=V_ts_mag,
        V_tb=V_tb,
    )


# Convenience: framework-derived inputs cached at module load.
V_us_framework = predict_V_us(_k_us, _g_us, _N_us)
V_cb_framework = predict_V_cb(_k_cb, _g_cb, _nfix_cb)
V_ub_framework = predict_V_ub(_k_ub, _g_ub, _s_ub, _nfix_ub, _m_ub)
delta_CP_deg_framework = predict_delta_CP_CKM_geometry(_k_dcp)
cos_delta_CP_framework = math.cos(math.radians(delta_CP_deg_framework))

CKM_FRAMEWORK = build_ckm_magnitudes(
    V_us_framework, V_cb_framework, V_ub_framework, cos_delta_CP_framework,
)


if __name__ == "__main__":
    print("=" * 68)
    print("  Framework CKM matrix magnitudes (unitarity construction)")
    print("=" * 68)
    print(f"  Inputs:")
    print(f"    V_us  = {V_us_framework:.10f}     (Row P4)")
    print(f"    V_cb  = {V_cb_framework:.10f}     (Row P3)")
    print(f"    V_ub  = {V_ub_framework:.6e}      (Row P14)")
    print(f"    δ_CP  = {delta_CP_deg_framework:.6f}°    (Row P15)")
    print(f"    cos δ = {cos_delta_CP_framework:.10f}  = 1/3 exact")
    print()
    print(f"  CKM magnitudes:")
    for name, val in CKM_FRAMEWORK._asdict().items():
        print(f"    |{name}| = {val:.6e}")
    print()
    print("  Non-prediction-file: see V_ud.py, V_cs.py, V_tb.py, V_cd.py,")
    print("  V_ts.py, V_td.py for individual prediction files.")
