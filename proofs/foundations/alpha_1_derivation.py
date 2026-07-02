#!/usr/bin/env python3
"""
---
derives: alpha_1
inputs:
  - k_star
  - g_girth
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: structural
rigor_status: closed
---

alpha_1 = ((k*-1)/k*)^(g-2)

Non-backtracking walk survival probability at walk length g-2 on a k*-regular
graph of girth g. For srs (k* = 3, g = 10): alpha_1 = (2/3)^8 = 256/6561
≈ 0.039018442310623.

This is the "bare" alpha_1 (not alpha_1_full = (5/3)*alpha_1 which carries the
additional tan²(arg h) = Im²(h)/Re²(h) factor from the Hermitian split of B).
Both coexist in the dark-correction framework: alpha_1 is the amplitude-class
coupling (V_us, V_cb, m_nu Feshbach), alpha_1_full is the mass²-class coupling
(Higgs quartic, theta_23, m_tau yukawa). Rows that need alpha_1_full apply the
(5/3) factor locally in their derive() from h_walker_eigenvalue.

See hashimoto_exponents.py (legacy) for the full Hashimoto-matrix derivation of
the alpha_1 exponent and fluctuation_spectrum.py for the related Koide-class
coupling fluctuation spectrum at k* = 3.
"""

import sys


def derive(k_star: int, g_girth: int) -> dict:
    """Return the bare alpha_1 = ((k_star-1)/k_star)^(g_girth-2).

    Parameters
    ----------
    k_star : int
        Vertex valence of the graph (must be >= 2).
    g_girth : int
        Graph girth (shortest cycle length, must be >= 2).

    Returns
    -------
    dict with keys:
        predicted : float — alpha_1
        checks : dict of intermediate quantities
    """
    if k_star < 2:
        raise ValueError(f"k_star must be >= 2; got {k_star}")
    if g_girth < 2:
        raise ValueError(f"g_girth must be >= 2; got {g_girth}")
    base = (k_star - 1) / k_star
    exponent = g_girth - 2
    predicted = base ** exponent
    return {
        'predicted': predicted,
        'checks': {
            'base': base,
            'exponent': exponent,
            'k_star': k_star,
            'g_girth': g_girth,
            'interpretation': 'NB walk survival probability at girth-2 on k*-regular graph',
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    k_star = 3
    g_girth = 10

    inputs = {'k_star': k_star, 'g_girth': g_girth}
    result = derive(**inputs)

    print(f"# PREDICT name=alpha_1 value={result['predicted']:.15f}")
    print()
    print("alpha_1 = ((k*-1)/k*)^(g-2)  NB walk survival at girth-2")
    print(f"  k_star      = {inputs['k_star']}")
    print(f"  g_girth     = {inputs['g_girth']}")
    print(f"  base        = (k*-1)/k* = {result['checks']['base']:.15f}")
    print(f"  exponent    = g-2 = {result['checks']['exponent']}")
    print(f"  alpha_1     = {result['predicted']:.15f}")


if __name__ == '__main__':
    main()
