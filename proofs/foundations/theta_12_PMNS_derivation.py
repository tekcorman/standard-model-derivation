#!/usr/bin/env python3
"""
---
derives: theta_12_PMNS
inputs:
  - V_us_bare_tree
script_version: 1.0.0
doc: docs/parameters/derivations.md
doc_section: §9 PMNS chain item 2
doc_version_required: 1.0.0
mechanism: structural
rigor_status: closed
---

theta_12_PMNS = degrees(acos(cos(theta_TBM) / cos(theta_C)))

with
  theta_TBM = arctan(1/sqrt(2))  (TBM solar angle, exact)
  theta_C   = arcsin(V_us_bare_tree)  (Cabibbo angle from bare framework V_us)

This is the spherical Pythagorean identity:
  cos(theta_12) = cos(theta_TBM) * cos(theta_C)   (perpendicular composition)

on the SU(4) quotient manifold where T_TBM and T_Cabibbo are Killing-orthogonal
generators (15 = 8 + 1 + 3 + 3bar decomposition in PS). See derivations.md §9
PMNS chain item 2.

**Bare V_us only**: this script consumes V_us_bare_tree = (2/3)^(2+sqrt(3)) =
0.22020 directly, not the Feshbach-corrected composite V_us = 0.22527. This
matches the framework convention in derivations.md §9 — the PMNS 12-rotation
uses the bare Cabibbo angle because the Feshbach dark correction acts on the
walker amplitude, not on the TBM-Cabibbo angle composition.

Framework-internal: the only input is V_us_bare_tree (itself framework-derived
from the Kesten-McKay spectral gap walk length). No observed theta_12 enters.
"""

import math
import sys


def derive(V_us_bare_tree: float) -> dict:
    """Return theta_12_PMNS in degrees.

    Parameters
    ----------
    V_us_bare_tree : float
        Bare framework V_us = (2/3)^(2+sqrt(3)), piped from the V_us_bare_tree
        YAML row.

    Returns
    -------
    dict with 'predicted' (degrees) and 'checks'.
    """
    if V_us_bare_tree <= 0 or V_us_bare_tree >= 1:
        raise ValueError(f"V_us_bare_tree out of range: {V_us_bare_tree}")

    theta_TBM_rad = math.atan(1.0 / math.sqrt(2.0))  # ~35.2644°
    theta_C_rad = math.asin(V_us_bare_tree)           # ~12.72°

    cos_tbm = math.cos(theta_TBM_rad)
    cos_c = math.cos(theta_C_rad)
    ratio = cos_tbm / cos_c
    if not -1.0 <= ratio <= 1.0:
        raise ValueError(f"acos argument out of [-1, 1]: {ratio}")

    theta_12_rad = math.acos(ratio)
    theta_12_deg = math.degrees(theta_12_rad)

    return {
        'predicted': theta_12_deg,
        'checks': {
            'V_us_bare_tree': V_us_bare_tree,
            'theta_TBM_deg': math.degrees(theta_TBM_rad),
            'theta_C_deg': math.degrees(theta_C_rad),
            'cos_theta_TBM': cos_tbm,
            'cos_theta_C': cos_c,
            'cos_theta_12': ratio,
            'theta_12_rad': theta_12_rad,
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    # V_us_bare_tree = ((k*-1)/k*)^L_us = (2/3)^(2+sqrt(3))
    L_us_bare_tree = 2 + math.sqrt(3)
    V_us_bare_tree = (2.0 / 3.0) ** L_us_bare_tree

    inputs = {'V_us_bare_tree': V_us_bare_tree}
    result = derive(**inputs)
    c = result['checks']

    print(f"# PREDICT name=theta_12_PMNS value={result['predicted']:.12f}")
    print()
    print("theta_12_PMNS = degrees(acos(cos(theta_TBM) / cos(theta_C)))")
    print(f"  V_us_bare_tree  = {c['V_us_bare_tree']:.12f}")
    print(f"  theta_TBM       = {c['theta_TBM_deg']:.10f}° (= atan(1/sqrt(2)))")
    print(f"  theta_C         = {c['theta_C_deg']:.10f}° (= asin(V_us_bare_tree))")
    print(f"  cos(theta_TBM)  = {c['cos_theta_TBM']:.12f}")
    print(f"  cos(theta_C)    = {c['cos_theta_C']:.12f}")
    print(f"  cos(theta_12)   = {c['cos_theta_12']:.12f}")
    print(f"  theta_12_PMNS   = {result['predicted']:.10f}°")


if __name__ == '__main__':
    main()
