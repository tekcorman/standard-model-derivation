#!/usr/bin/env python3
"""
---
derives: theta_13_PMNS
inputs:
  - V_us_bare_tree
  - alpha_1
  - k_star
script_version: 1.0.0
doc: docs/parameters/target_parameters.md
doc_section: §4c.5b edge-local vertex-selection
doc_version_required: 1.0.0
mechanism: edge_local_vertex_selection
rigor_status: closed
---

theta_13_PMNS = degrees(arcsin((V_us_bare_tree / sqrt(k*-1)) * (1 - alpha_1_bare)))

Two structural ingredients:

1. **TBM baseline**: theta_13(TBM) = arcsin(V_us / sqrt(k*-1)). The
   sqrt(k*-1) factor is the TBM (2,3)/(3,3) element = 1/sqrt(k*-1) =
   1/sqrt(2), which is the maximal mixing of k*-1 = 2 states at the P-point
   under S_4(K_4) symmetry. The V_us factor comes from quark-lepton
   universality on srs: (U_l)_{12} = V_us because both sectors are Fock
   states on the same graph and the non-backtracking walk amplitude
   ((k*-1)/k*)^L_us is sector-blind. See srs_theta13_derivation.py §step 3b
   for the universality argument and srs_so10_embedding.py for the PS/SO(10)
   embedding framing.

2. **Edge-local dark absorption**: multiplies the amplitude sin(theta_13) by
   (1 - alpha_1_bare), with coefficient c_vertex = 1 (no chirality
   enhancement) derived from Tr(sigma_x) = 0 on the C_3-symmetric vertex.
   This kills the mass^2-class tan^2(arg h) enhancement that would otherwise
   apply to angle observables, reducing the correction to a trivial linear
   absorption — the "edge-local vertex-selection" class in the unified dark
   correction theorem (dark_correction_theorem_2026-04-14.md §4c.5b).

The correction acts on the AMPLITUDE sin(theta_13), not on the angle itself,
because the dark coupling absorbs walker flux edge-by-edge. Framework-internal:
all inputs come from the YAML (V_us_bare_tree, alpha_1, k_star). No observed
theta_13 enters.
"""

import math
import sys


def derive(V_us_bare_tree: float, alpha_1_bare: float, k_star: int) -> dict:
    """Return theta_13_PMNS in degrees.

    Parameters
    ----------
    V_us_bare_tree : float
        Bare framework V_us = (2/3)^(2+sqrt(3)), piped from V_us_bare_tree row.
    alpha_1_bare : float
        Bare chirality coupling = (2/3)^8, piped from alpha_1 row.
    k_star : int
        Vertex valence (= 3 for srs), piped from k_star row.

    Returns
    -------
    dict with 'predicted' (degrees) and 'checks'.
    """
    if V_us_bare_tree <= 0 or V_us_bare_tree >= 1:
        raise ValueError(f"V_us_bare_tree out of range: {V_us_bare_tree}")
    if alpha_1_bare <= 0 or alpha_1_bare >= 1:
        raise ValueError(f"alpha_1_bare out of range: {alpha_1_bare}")
    if k_star < 2:
        raise ValueError(f"k_star must be >= 2; got {k_star}")

    tbm_factor = 1.0 / math.sqrt(k_star - 1)           # 1/sqrt(2) for k*=3
    sin_theta_13_tbm = V_us_bare_tree * tbm_factor     # TBM baseline amplitude
    dark_factor = 1.0 - alpha_1_bare                   # edge-local absorption
    sin_theta_13 = sin_theta_13_tbm * dark_factor

    if not -1.0 <= sin_theta_13 <= 1.0:
        raise ValueError(f"arcsin argument out of [-1, 1]: {sin_theta_13}")

    theta_13_rad = math.asin(sin_theta_13)
    theta_13_deg = math.degrees(theta_13_rad)

    theta_13_tbm_deg = math.degrees(math.asin(sin_theta_13_tbm))

    return {
        'predicted': theta_13_deg,
        'checks': {
            'V_us_bare_tree': V_us_bare_tree,
            'alpha_1_bare': alpha_1_bare,
            'k_star': k_star,
            'tbm_factor_1_over_sqrt_k_minus_1': tbm_factor,
            'sin_theta_13_tbm': sin_theta_13_tbm,
            'theta_13_tbm_deg': theta_13_tbm_deg,
            'dark_factor_1_minus_alpha_1_bare': dark_factor,
            'sin_theta_13_dark': sin_theta_13,
            'theta_13_rad': theta_13_rad,
            'dark_shift_deg': theta_13_deg - theta_13_tbm_deg,
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    k_star = 3
    alpha_1_bare = (2.0 / 3.0) ** 8
    L_us_bare_tree = 2 + math.sqrt(3)
    V_us_bare_tree = (2.0 / 3.0) ** L_us_bare_tree

    inputs = {'V_us_bare_tree': V_us_bare_tree, 'alpha_1_bare': alpha_1_bare, 'k_star': k_star}
    result = derive(**inputs)
    c = result['checks']

    print(f"# PREDICT name=theta_13_PMNS value={result['predicted']:.12f}")
    print()
    print("theta_13_PMNS = degrees(arcsin((V_us_bare_tree / sqrt(k*-1)) * (1 - alpha_1_bare)))")
    print(f"  V_us_bare_tree              = {c['V_us_bare_tree']:.12f}")
    print(f"  alpha_1_bare                = {c['alpha_1_bare']:.15f}")
    print(f"  k_star                      = {c['k_star']}")
    print(f"  1/sqrt(k*-1)                = {c['tbm_factor_1_over_sqrt_k_minus_1']:.12f}")
    print(f"  sin(theta_13)_TBM           = {c['sin_theta_13_tbm']:.12f}")
    print(f"  theta_13(TBM, pre-dark)     = {c['theta_13_tbm_deg']:.10f}°")
    print(f"  (1 - alpha_1_bare)          = {c['dark_factor_1_minus_alpha_1_bare']:.15f}")
    print(f"  sin(theta_13)_dark          = {c['sin_theta_13_dark']:.12f}")
    print(f"  theta_13_PMNS               = {result['predicted']:.10f}°")
    print(f"  dark shift from TBM         = {c['dark_shift_deg']:+.6f}°")


if __name__ == '__main__':
    main()
