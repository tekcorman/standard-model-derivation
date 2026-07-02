#!/usr/bin/env python3
"""
---
derives: m_mu
inputs:
  - m_tau
  - epsilon_Koide
  - delta_Koide
  - k_star
script_version: 1.0.0
doc: docs/parameters/target_parameters.md
doc_section: lepton Yukawa sector / Koide triplet
doc_version_required: 0.0.1
mechanism: structural
rigor_status: rigor_route_specified
---

m_mu from the symmetric Koide parametrization on k*=3, anchored to m_tau.

The C3-symmetric lepton mass triplet is
    sqrt(m_j) = M0 * (1 + eps*cos(2*pi*j/k* + delta)),  j=0,1,2
with (eps, delta) = (sqrt(2), 2/9) both theorem-grade from k_star alone
(see epsilon_Koide, delta_Koide rows). The three factors
    f_j = 1 + eps*cos(2*pi*j/k* + delta)
split into one dominant factor f_max (tau), one intermediate (mu), one small
(electron). The common scale M0 drops out of every ratio:
    m_mu = m_tau * (f_mid / f_max)^2.

Grade is inherited from m_tau (A-), because m_tau itself inherits v's A-
under strict grading (1b). The Koide (eps, delta) pair adds no free
parameters.
"""

import math


def derive(m_tau: float,
           epsilon_Koide: float,
           delta_Koide: float,
           k_star: float) -> dict:
    """m_mu = m_tau * (f_mid/f_max)^2 with Koide C_k* factors."""
    if m_tau <= 0:
        raise ValueError(f"m_tau must be > 0; got {m_tau}")
    if k_star < 2:
        raise ValueError(f"k_star must be >= 2; got {k_star}")

    k_int = int(round(k_star))
    factors = [
        1.0 + epsilon_Koide * math.cos(2.0 * math.pi * j / k_int + delta_Koide)
        for j in range(k_int)
    ]
    factors_sorted = sorted(factors)  # ascending
    f_min, f_mid, f_max = factors_sorted[0], factors_sorted[1], factors_sorted[2]
    ratio_sq = (f_mid / f_max) ** 2
    m_mu = m_tau * ratio_sq
    return {
        'predicted': m_mu,
        'checks': {
            'factors': factors_sorted,
            'ratio_sq_mu_over_tau': ratio_sq,
            'm_tau': m_tau,
        },
    }


def main():
    # Make repo root importable when this script is invoked directly
    # (python3 proofs/masses/m_mu_derivation.py) rather than as a module.
    import os, sys
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '..', '..'))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    # Compute m_tau from framework constants
    from proofs.masses.m_tau_derivation import derive as derive_m_tau

    k_star = 3.0
    v = 245.64
    alpha_1 = (2.0 / 3.0) ** 8
    h_real = math.sqrt(3) / 2.0
    h_imag = math.sqrt(5) / 2.0
    epsilon_Koide = math.sqrt(2)
    delta_Koide = 2.0 / 9.0

    m_tau = derive_m_tau(v, alpha_1, h_real, h_imag, k_star)['predicted']

    inputs = {'m_tau': m_tau, 'epsilon_Koide': epsilon_Koide, 'delta_Koide': delta_Koide, 'k_star': k_star}
    result = derive(**inputs)

    print(f"# PREDICT name=m_mu value={result['predicted']:.15f}")
    print()
    print("m_mu = m_tau * (f_mid / f_max)^2")
    print(f"  m_tau                = {inputs['m_tau']:.15f}  GeV   (framework A-)")
    print(f"  epsilon_Koide        = {inputs['epsilon_Koide']:.15f}   (= sqrt(2))")
    print(f"  delta_Koide          = {inputs['delta_Koide']:.15f}   (= 2/9)")
    print(f"  f_min, f_mid, f_max  = {result['checks']['factors']}")
    print(f"  (f_mid/f_max)^2      = {result['checks']['ratio_sq_mu_over_tau']:.15f}")
    print(f"  m_mu                 = {result['predicted']:.15f}  GeV")


if __name__ == '__main__':
    main()
