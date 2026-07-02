#!/usr/bin/env python3
"""
---
derives: Q_Koide
inputs:
  - k_star
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: structural
rigor_status: closed
---

Q_Koide = (k* - 1) / k* = 2/3

Ratio Q = sum(m) / (sum sqrt(m))^2 for the Koide charged-lepton mass triplet,
derived from MDL equal allocation on the trivalent srs node: each of the k*=3
edges carries one unit of toggle activity, and two of the three edges
participate in any given non-backtracking step (the walker cannot immediately
reverse the edge it just arrived on). The active fraction is therefore
(k*-1)/k* = 2/3. Dressed with the symmetric Koide parametrization (see
epsilon_Koide), this fraction is exactly the Koide Q on the resulting mass
triplet.

Framework-internal: only input is k_star from the MDL + toggle derivation of
the srs vertex valence. No observed lepton masses enter the derivation.
"""

import sys


def derive(k_star: int) -> dict:
    """Return Q_Koide = (k*-1)/k*.

    Parameters
    ----------
    k_star : int
        Vertex valence of the srs graph (must be >= 2; srs has k*=3).

    Returns
    -------
    dict with 'predicted' and 'checks'.
    """
    if k_star < 2:
        raise ValueError(f"k_star must be >= 2; got {k_star}")
    Q = (k_star - 1) / k_star
    return {
        'predicted': Q,
        'checks': {
            'k_star': k_star,
            'active_edges': k_star - 1,
            'interpretation': 'non-backtracking fraction: k*-1 of k* edges active',
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    k_star = 3

    inputs = {'k_star': k_star}
    result = derive(**inputs)
    c = result['checks']

    print(f"# PREDICT name=Q_Koide value={result['predicted']:.15f}")
    print()
    print("Q_Koide = (k*-1)/k*  (non-backtracking active fraction)")
    print(f"  k_star        = {c['k_star']}")
    print(f"  active_edges  = k*-1 = {c['active_edges']}")
    print(f"  Q_Koide       = {result['predicted']:.15f}")


if __name__ == '__main__':
    main()
