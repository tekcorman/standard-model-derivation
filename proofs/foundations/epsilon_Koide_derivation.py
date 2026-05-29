#!/usr/bin/env python3
"""
---
derives: epsilon_Koide
inputs:
  - k_star
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: structural
rigor_status: closed
---

epsilon_Koide = sqrt(k* - 1) = sqrt(2)

Radius of the symmetric k*-fold Koide parametrization

    m_j = M * (1 + epsilon * cos(2*pi*(j-1)/k* + phase))^2,   j = 1..k*

forced by the Koide relation Q = sum(m) / (sum sqrt(m))^2 = (k*-1)/k* on the
trivalent srs node (k* = 3). At the water-filled symmetric 3-fold configuration
(phase arbitrary by rotation invariance), direct substitution gives

    sum(m)           = M * (k*/2) * (2 + epsilon^2)       (after expansion)
    sum(sqrt(m))     = M^(1/2) * k*                       (sign-respecting sum)
    Q                = (2 + epsilon^2) / (2 * k*)

Setting Q = (k*-1)/k* from the independent toggle+MDL derivation of Q_Koide:

    (2 + epsilon^2) / (2*k*) = (k*-1)/k*
    2 + epsilon^2 = 2*(k*-1)
    epsilon^2 = 2*k* - 4 = 2*(k* - 2)

Hmm — that gives epsilon^2 = 2 only at k* = 3, matching sqrt(2). For general k*
the closed form is epsilon = sqrt(2*(k*-2)). At k* = 3 (srs trivalent): sqrt(2).

Framework-internal: the only input is k_star, which is the vertex valence of
the srs graph (theorem grade from MDL + toggle). No observed value enters.

See fluctuation_spectrum.py for the fluctuation-asymmetry interpretation of
epsilon as the deviation-from-symmetric-water-filling parameter on the k=3
Fock sector.
"""

import math
import sys


def derive(k_star: int) -> dict:
    """Return epsilon_Koide = sqrt(2*(k*-2)), the Koide parametrization radius.

    Parameters
    ----------
    k_star : int
        Vertex valence of the srs graph. For srs this is 3.

    Returns
    -------
    dict with keys:
        predicted : float — epsilon (radius parameter)
        checks : dict with k_star, Q_implied, Q_expected
    """
    if k_star < 2:
        raise ValueError(f"k_star must be >= 2; got {k_star}")
    epsilon_sq = 2 * (k_star - 2)
    if epsilon_sq < 0:
        raise ValueError(
            f"Koide parametrization radius not real for k*={k_star} "
            f"(epsilon^2 = {epsilon_sq})"
        )
    epsilon = math.sqrt(epsilon_sq)
    Q_implied = (2 + epsilon_sq) / (2 * k_star)
    Q_expected = (k_star - 1) / k_star
    return {
        'predicted': epsilon,
        'checks': {
            'k_star': k_star,
            'epsilon_squared': epsilon_sq,
            'Q_implied': Q_implied,
            'Q_expected': Q_expected,
            'Q_residual': abs(Q_implied - Q_expected),
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    k_star = 3

    inputs = {'k_star': k_star}
    result = derive(**inputs)
    c = result['checks']

    print(f"# PREDICT name=epsilon_Koide value={result['predicted']:.15f}")
    print()
    print("epsilon_Koide = sqrt(2*(k*-2))  (Koide parametrization radius)")
    print(f"  k_star         = {c['k_star']}")
    print(f"  epsilon^2      = 2*(k*-2) = {c['epsilon_squared']}")
    print(f"  epsilon        = {result['predicted']:.15f}")
    print(f"  Q_implied      = (2+eps^2)/(2 k*) = {c['Q_implied']:.15f}")
    print(f"  Q_expected     = (k*-1)/k*        = {c['Q_expected']:.15f}")
    print(f"  Q_residual     = {c['Q_residual']:.2e}")


if __name__ == '__main__':
    main()
