#!/usr/bin/env python3
"""
---
derives: g_3
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

g_3 (SU(3) coupling) at M_Z from top-down MSSM RG running.
Same chain as g_1_derivation.py — alpha_GUT at M_GUT run down to M_Z.

NOTE: The framework predicts alpha_s(M_Z) ~ 0.155 from alpha_GUT = 1/24.1,
which is 31% above observed 0.1180. This is a known gap — three-way gauge
unification does not hold in minimal MSSM. The honest prediction is reported.
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mssm_rge import run_down_from_gut  # noqa: E402

M_GUT = 2.0e16
M_SUSY = 3000.0


def derive(alpha_GUT: float, sin2_theta_W: float, v: float) -> dict:
    alpha_GUT_inv = 1.0 / alpha_GUT
    cos_theta_W = math.sqrt(1.0 - sin2_theta_W)

    m_z = 91.0
    for _ in range(20):
        y = run_down_from_gut(alpha_GUT_inv, M_GUT, m_z, M_SUSY)
        a2_inv = y[1]
        g_2 = math.sqrt(4.0 * math.pi / a2_inv)
        m_z_new = v * g_2 / (2.0 * cos_theta_W)
        if abs(m_z_new - m_z) < 1e-6:
            break
        m_z = m_z_new

    a3_inv = y[2]
    g_3 = math.sqrt(4.0 * math.pi / a3_inv)

    return {
        'predicted': g_3,
        'checks': {
            'alpha_3_inv': a3_inv,
            'alpha_s': 1.0 / a3_inv,
            'M_Z_self_consistent': m_z,
        },
    }


def main():
    result = derive(alpha_GUT=1.0/24.1, sin2_theta_W=3.0/13.0, v=245.64)
    print(f"# PREDICT name=g_3 value={result['predicted']:.15f}")
    print(f"  g_3 = {result['predicted']:.15f}")
    print(f"  alpha_s(M_Z) = {result['checks']['alpha_s']:.6f}  (obs: 0.1180)")
    print(f"  M_Z = {result['checks']['M_Z_self_consistent']:.4f} GeV")


if __name__ == '__main__':
    main()
