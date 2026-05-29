#!/usr/bin/env python3
"""
---
derives: m_d
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

m_d from the down-type quark Koide triplet (n=1), anchored to m_b.
Same sector machinery as m_s_derivation.py; m_d uses f_min:

    m_d = m_b * (f_min / f_max)^2
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
    ratio_sq = (koide['f_min'] / koide['f_max']) ** 2
    m_d = m_b * ratio_sq
    return {
        'predicted': m_d,
        'checks': {
            'alpha_1_full': alpha_1_full,
            'eps_sq': koide['eps_sq'],
            'delta': koide['delta'],
            'factors_sorted': koide['factors_sorted'],
            'ratio_sq_md_over_mb': ratio_sq,
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

    print(f"# PREDICT name=m_d value={result['predicted']:.15f}")
    print()
    print("m_d = m_b * (f_min / f_max)^2   [down sector, n=1]")
    print(f"  m_b                  = {inputs['m_b']:.6f}  GeV   (framework anchor)")
    print(f"  alpha_1_full         = {result['checks']['alpha_1_full']:.15f}")
    print(f"  eps_1^2              = {result['checks']['eps_sq']:.15f}")
    print(f"  delta_1              = {result['checks']['delta']:.15f}  (= 1/9)")
    print(f"  f_min, f_mid, f_max  = {result['checks']['factors_sorted']}")
    print(f"  (f_min/f_max)^2      = {result['checks']['ratio_sq_md_over_mb']:.15f}")
    print(f"  m_d                  = {result['predicted']:.15f}  GeV")


if __name__ == '__main__':
    main()
