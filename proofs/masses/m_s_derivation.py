#!/usr/bin/env python3
"""
---
derives: m_s
inputs:
  - m_b
  - alpha_1
  - h_walker_eigenvalue
  - k_star
script_version: 1.0.0
doc: docs/parameters/target_parameters.md
doc_section: quark Yukawa sector / down-type Koide triplet (Stage 1)
doc_version_required: 0.0.1
mechanism: structural
rigor_status: rigor_route_specified
---

m_s from the down-type quark Koide triplet (n=1), anchored to m_b.

Down-sector Koide parameters (k*=3):
    eps_1^2 = 2 + 6 * alpha_1_full * 1 * f(1)   with f(1) = 1
    delta_1 = 2/(9*2) = 1/9

Here alpha_1_full = alpha_1_bare * (Im h / Re h)^2 = (2/3)^8 * 5/3.
The heaviest generation (b) sits at f_max; m_s is the middle factor:

    m_s = m_b * (f_mid / f_max)^2

Stage 1 (2026-04-15): script conversion only; grade inherits from m_b.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _quark_koide import koide_factors, alpha_1_full_from_inputs, shared_inputs


N_SECTOR = 1  # down-type quarks


def derive(m_b: float,
           alpha_1_bare: float,
           h_real: float,
           h_imag: float,
           k_star: float) -> dict:
    if m_b <= 0:
        raise ValueError(f"m_b must be > 0; got {m_b}")
    alpha_1_full = alpha_1_full_from_inputs(alpha_1_bare, h_real, h_imag)
    koide = koide_factors(N_SECTOR, alpha_1_full, k_star)
    ratio_sq = (koide['f_mid'] / koide['f_max']) ** 2
    m_s = m_b * ratio_sq
    return {
        'predicted': m_s,
        'checks': {
            'alpha_1_full': alpha_1_full,
            'eps_sq': koide['eps_sq'],
            'delta': koide['delta'],
            'f_n_of_1': koide['f_n'],
            'factors_sorted': koide['factors_sorted'],
            'ratio_sq_ms_over_mb': ratio_sq,
            'm_b': m_b,
        },
    }


def _load_inputs() -> dict:
    shared = shared_inputs()
    return {
        'm_b': shared['m_b'],
        'alpha_1_bare': shared['alpha_1_bare'],
        'h_real': shared['h_real'],
        'h_imag': shared['h_imag'],
        'k_star': shared['k_star'],
    }


def main():
    inputs = _load_inputs()
    result = derive(**inputs)

    print(f"# PREDICT name=m_s value={result['predicted']:.15f}")
    print()
    print("m_s = m_b * (f_mid / f_max)^2   [down sector, n=1]")
    print(f"  m_b                  = {inputs['m_b']:.6f}  GeV   (framework anchor)")
    print(f"  alpha_1_full         = {result['checks']['alpha_1_full']:.15f}")
    print(f"  f(1)                 = {result['checks']['f_n_of_1']:.15f}  (= 1)")
    print(f"  eps_1^2              = {result['checks']['eps_sq']:.15f}")
    print(f"  delta_1              = {result['checks']['delta']:.15f}  (= 1/9)")
    print(f"  f_min, f_mid, f_max  = {result['checks']['factors_sorted']}")
    print(f"  (f_mid/f_max)^2      = {result['checks']['ratio_sq_ms_over_mb']:.15f}")
    print(f"  m_s                  = {result['predicted']:.15f}  GeV")


if __name__ == '__main__':
    main()
