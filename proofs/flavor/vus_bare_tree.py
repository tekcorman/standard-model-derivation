#!/usr/bin/env python3
"""
---
derives: V_us_bare_tree
inputs:
  - k_star
  - L_us_bare_tree
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: structural
rigor_status: heuristic
rigor_route: gacs_formalization
---

V_us_bare_tree = ((k*-1)/k*)^L_us_bare_tree

Bare tree-Green's-function walk amplitude for V_us on the srs graph. The
effective walk length L_us_bare_tree = 2+sqrt(3) is the framework's
GACS-derived spectral-gap distance at z* = 17/6 (see L_us_bare_tree YAML
row). The base (k*-1)/k* = 2/3 on trivalent srs is the NB walk step
survival probability.

The correction on top of this bare term is handled separately by
vus_feshbach_correction.py; the combined V_us is computed by vus_derivation.py
from (V_us_bare_tree, V_us_feshbach_correction).
"""

import sys


def derive(k_star: int, L_us_bare_tree: float) -> dict:
    """Return the bare tree prediction for V_us.

    Parameters
    ----------
    k_star : int
        Vertex valence of the srs graph (must be 3 for the current theorem).
    L_us_bare_tree : float
        Effective CKM walk distance from the Kesten-McKay spectral gap
        resolution, piped from the L_us_bare_tree YAML row.

    Returns
    -------
    dict with keys:
        predicted : float
        checks : dict of intermediate quantities
    """
    if k_star < 2:
        raise ValueError(f"k_star must be >= 2 (srs is trivalent); got {k_star}")
    base = (k_star - 1) / k_star
    predicted = base ** L_us_bare_tree
    return {
        'predicted': predicted,
        'checks': {
            'base': base,
            'L_us_bare_tree': L_us_bare_tree,
            'k_star': k_star,
        },
    }


def main():
    import math

    # Framework constants (hardcoded, no YAML dependency)
    k_star = 3
    L_us_bare_tree = 2 + math.sqrt(3)

    inputs = {'k_star': k_star, 'L_us_bare_tree': L_us_bare_tree}
    result = derive(**inputs)

    print(f"# PREDICT name=V_us_bare_tree value={result['predicted']:.15f}")
    print()
    print("V_us bare tree (Kesten-McKay) derivation")
    print(f"  inputs:")
    print(f"    k_star           = {inputs['k_star']}")
    print(f"    L_us_bare_tree   = {inputs['L_us_bare_tree']:.15f}")
    print(f"  base = (k*-1)/k*    = {result['checks']['base']:.15f}")
    print(f"  V_us_bare_tree = base^L = {result['predicted']:.15f}")


if __name__ == '__main__':
    main()
