#!/usr/bin/env python3
"""
---
derives: g_1
inputs:
  - alpha_GUT
  - sin2_theta_W
  - v
script_version: 2.0.0
doc: standard-model-derivation/docs/parameters/derivations.md
doc_section: '§9.7 MSSM RG running'
doc_version_required: 0.0.1
mechanism: structural
rigor_status: rigor_route_specified
---

g_1 (SM hypercharge coupling g') at M_Z from top-down MSSM RG running.

Chain: alpha_GUT = 1/24.1 (Cl(6)) at M_GUT -> MSSM two-loop running to
M_SUSY -> SM two-loop running to M_Z -> extract alpha_1(M_Z) in GUT
normalization -> convert to SM g' = sqrt(4 pi alpha_Y).

M_Z is self-consistently determined: M_Z = v * g_2 / (2 cos theta_W)
where g_2 comes from the same running. The script iterates to convergence.

M_GUT = 2e16 GeV (framework), M_SUSY = 3000 GeV (from m_{3/2}).

Grade: A- (alpha_GUT is theorem; RG is standard math; v is A-).
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mssm_rge import run_down_from_gut  # noqa: E402

M_GUT = 2.0e16   # GeV, framework
M_SUSY = 3000.0  # GeV, from m_{3/2} = (2/3)^90 M_P


def derive(alpha_GUT: float, sin2_theta_W: float, v: float) -> dict:
    alpha_GUT_inv = 1.0 / alpha_GUT
    cos_theta_W = math.sqrt(1.0 - sin2_theta_W)

    # Self-consistent M_Z: iterate
    m_z = 91.0  # initial guess
    for _ in range(20):
        y = run_down_from_gut(alpha_GUT_inv, M_GUT, m_z, M_SUSY)
        a2_inv = y[1]
        g_2 = math.sqrt(4.0 * math.pi / a2_inv)
        m_z_new = v * g_2 / (2.0 * cos_theta_W)
        if abs(m_z_new - m_z) < 1e-6:
            break
        m_z = m_z_new

    a1_inv = y[0]
    # GUT normalization: alpha_1_GUT = (5/3) alpha_Y
    alpha_Y = (3.0 / 5.0) / a1_inv
    g_1 = math.sqrt(4.0 * math.pi * alpha_Y)

    return {
        'predicted': g_1,
        'checks': {
            'alpha_GUT_inv': alpha_GUT_inv,
            'M_Z_self_consistent': m_z,
            'g_2': g_2,
            'alpha_1_inv_GUT_norm': a1_inv,
            'alpha_Y': alpha_Y,
        },
    }


def main():
    # Framework inputs only
    alpha_GUT = 1.0 / 24.1
    sin2_theta_W = 3.0 / 13.0
    v = 245.64  # GeV, framework A-

    result = derive(alpha_GUT, sin2_theta_W, v)

    print(f"# PREDICT name=g_1 value={result['predicted']:.15f}")
    print()
    print("g_1 = g'(M_Z) from alpha_GUT top-down MSSM RG")
    print(f"  alpha_GUT^-1       = {result['checks']['alpha_GUT_inv']:.1f}")
    print(f"  M_Z (self-consist) = {result['checks']['M_Z_self_consistent']:.4f} GeV")
    print(f"  g_1 (SM g')        = {result['predicted']:.15f}")


if __name__ == '__main__':
    main()
