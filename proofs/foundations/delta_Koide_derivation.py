#!/usr/bin/env python3
"""
---
derives: delta_Koide
inputs:
  - k_star
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: structural
rigor_status: closed
---

delta_Koide = HM(P_+, P_0, P_-) = 2/9

Koide phase delta, defined as the harmonic mean of the three Wigner D1-matrix
diagonal survival probabilities at the C3-protected angle beta = arccos(1/k*).
On the trivalent srs node, cos(beta) = 1/k* = 1/3 → beta = arccos(1/3); the
J=1 Wigner d-matrix diagonal entries are

    d^1_{+1,+1}(beta) = (1 + cos beta)/2 = (1 + 1/k*)/2
    d^1_{0,0}(beta)   = cos beta         = 1/k*
    d^1_{-1,-1}(beta) = (1 + cos beta)/2 = (1 + 1/k*)/2

The survival probabilities are the squares:

    P_+ = P_- = ((k*+1) / (2 k*))^2
    P_0       = 1 / k*^2

At k* = 3: P = (4/9, 1/9, 4/9). The harmonic mean

    HM(4/9, 1/9, 4/9) = 3 / (4/(4/9) + 1/(1/9)) = 3 / (9/4 + 9 + 9/4)
                      = 3 / (9/2 + 9)
                      = 3 / (27/2) = 6/27 = 2/9

is the unique power-mean of {4/9, 1/9, 4/9} yielding 2/9 (harmonic_mean_proof.py
establishes uniqueness among Holder means). Framework-internal: only input is
k_star — the Wigner D1 structure is standard SU(2) representation theory and
the C3-protected cos(beta) = 1/k* angle is a framework-structural consequence
of the trivalent-vertex geometry.
"""

import sys
from fractions import Fraction


def derive(k_star: int) -> dict:
    """Return delta_Koide = HM(P_+, P_0, P_-) at cos(beta) = 1/k*.

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
    k = Fraction(k_star)
    cos_beta = Fraction(1, k_star)
    P_pm = ((1 + cos_beta) / 2) ** 2
    P_0 = cos_beta ** 2
    inv_sum = 2 / P_pm + 1 / P_0
    HM = 3 / inv_sum
    return {
        'predicted': float(HM),
        'checks': {
            'k_star': k_star,
            'cos_beta': float(cos_beta),
            'P_plus_minus': float(P_pm),
            'P_zero': float(P_0),
            'HM_rational': f"{HM.numerator}/{HM.denominator}",
            'HM_float': float(HM),
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    k_star = 3

    inputs = {'k_star': k_star}
    result = derive(**inputs)
    c = result['checks']

    print(f"# PREDICT name=delta_Koide value={result['predicted']:.15f}")
    print()
    print("delta_Koide = HM(P_+, P_0, P_-)   (Wigner D1 harmonic mean)")
    print(f"  k_star        = {c['k_star']}")
    print(f"  cos(beta)     = 1/k*  = {c['cos_beta']:.15f}")
    print(f"  P_+, P_-      = ((1+1/k*)/2)^2 = {c['P_plus_minus']:.15f}")
    print(f"  P_0           = (1/k*)^2       = {c['P_zero']:.15f}")
    print(f"  HM (rational) = {c['HM_rational']}")
    print(f"  delta_Koide   = {result['predicted']:.15f}")


if __name__ == '__main__':
    main()
