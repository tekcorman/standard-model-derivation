#!/usr/bin/env python3
"""
---
derives: V_us
inputs:
  - V_us_bare_tree
  - V_us_feshbach_correction
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: feshbach_amplitude
rigor_status: heuristic
rigor_route: gacs_formalization
---

V_us = V_us_bare_tree * (1 + V_us_feshbach_correction)

Composite theorem for V_us. Both factors are themselves YAML rows with
their own derive() functions:

- V_us_bare_tree = (2/3)^(2+sqrt(3)) — tree Green's function on srs
  at the Kesten-McKay spectral gap distance (vus_bare_tree.py).
- V_us_feshbach_correction = sqrt(5)/4 * alpha_1 — Feshbach self-energy
  on the water-filled ruliad Q-space at the walker-h P-point
  (vus_feshbach_correction.py).

This script does nothing but combine the two via the multiplicative
correction rule. The physics lives upstream; any change to the bare
amplitude or the correction factor automatically flows here.
"""

import sys


def derive(V_us_bare_tree: float, V_us_feshbach_correction: float) -> dict:
    """Return the composite V_us prediction.

    V_us = V_us_bare_tree * (1 + V_us_feshbach_correction)
    """
    predicted = V_us_bare_tree * (1.0 + V_us_feshbach_correction)
    return {
        'predicted': predicted,
        'checks': {
            'V_us_bare_tree': V_us_bare_tree,
            'V_us_feshbach_correction': V_us_feshbach_correction,
            'correction_factor': 1.0 + V_us_feshbach_correction,
        },
    }


def main():
    import math
    import os

    # Make repo root importable when this script is invoked directly
    # (python3 proofs/flavor/vus_derivation.py) rather than as a module.
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '..', '..'))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    # Compute upstream values from framework constants
    from proofs.flavor.vus_bare_tree import derive as derive_vus_bare
    from proofs.flavor.vus_feshbach_correction import derive as derive_feshbach

    k_star = 3
    L_us_bare_tree = 2 + math.sqrt(3)
    alpha_1 = (2.0 / 3.0) ** 8
    h_real = math.sqrt(3) / 2.0
    h_imag = math.sqrt(5) / 2.0

    V_us_bare_tree = derive_vus_bare(k_star, L_us_bare_tree)['predicted']
    V_us_feshbach_correction = derive_feshbach(alpha_1, h_real, h_imag)['predicted']

    inputs = {'V_us_bare_tree': V_us_bare_tree, 'V_us_feshbach_correction': V_us_feshbach_correction}
    result = derive(**inputs)

    print(f"# PREDICT name=V_us value={result['predicted']:.15f}")
    print()
    print("V_us = V_us_bare_tree * (1 + V_us_feshbach_correction)")
    print(f"  V_us_bare_tree            = {inputs['V_us_bare_tree']:.15f}")
    print(f"  V_us_feshbach_correction  = {inputs['V_us_feshbach_correction']:.15f}")
    print(f"  correction factor         = {result['checks']['correction_factor']:.15f}")
    print(f"  V_us                      = {result['predicted']:.15f}")


if __name__ == '__main__':
    main()
