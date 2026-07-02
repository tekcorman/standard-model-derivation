#!/usr/bin/env python3
"""
---
derives: m_c
inputs:
  - m_t
  - alpha_1
  - h_walker_eigenvalue
  - k_star
script_version: 1.0.0
doc: docs/parameters/target_parameters.md
doc_section: quark Yukawa sector / up-type Koide triplet (Stage 1)
doc_version_required: 0.0.1
mechanism: structural
rigor_status: rigor_route_specified
---

m_c from the up-type quark Koide triplet (n=2), anchored to m_t.

Up-sector Koide parameters (k*=3):
    eps_2^2 = 2 + 6 * alpha_1_full * 2 * f(2)    with f(2) = 1 + (g-2)/(2g)
    delta_2 = 2/(9*3) = 2/27
where alpha_1_full = alpha_1_bare * (Im h / Re h)^2 = (2/3)^8 * 5/3 and
g = 10 is the srs girth.

The three sqrt-mass factors sorted ascending are (f_min, f_mid, f_max)
with f_max the heaviest-generation (top). Propagation:
    m_c = m_t * (f_mid / f_max)^2

Stage 1 (2026-04-15): script-conversion-only. Grade inherits from m_t
(currently A-) because the Koide machinery adds no new free parameters
*conditional on* the conjecture-grade eps^2 formula. See
quark_koide_proof.py section 7 for the honest grade breakdown and
theory_open_items.md (Koide delta(n) first-principles) for the
Stage 2 upgrade path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _quark_koide import koide_factors, alpha_1_full_from_inputs, shared_inputs


N_SECTOR = 2  # up-type quarks


def derive(m_t: float,
           alpha_1_bare: float,
           h_real: float,
           h_imag: float,
           k_star: float) -> dict:
    if m_t <= 0:
        raise ValueError(f"m_t must be > 0; got {m_t}")
    alpha_1_full = alpha_1_full_from_inputs(alpha_1_bare, h_real, h_imag)
    koide = koide_factors(N_SECTOR, alpha_1_full, k_star)
    ratio_sq = (koide['f_mid'] / koide['f_max']) ** 2
    m_c = m_t * ratio_sq
    return {
        'predicted': m_c,
        'checks': {
            'alpha_1_bare': alpha_1_bare,
            'alpha_1_full': alpha_1_full,
            'eps_sq': koide['eps_sq'],
            'delta': koide['delta'],
            'f_n_of_2': koide['f_n'],
            'factors_sorted': koide['factors_sorted'],
            'ratio_sq_mc_over_mt': ratio_sq,
            'm_t': m_t,
        },
    }


def _load_inputs() -> dict:
    shared = shared_inputs()
    return {
        'm_t': shared['m_t'],
        'alpha_1_bare': shared['alpha_1_bare'],
        'h_real': shared['h_real'],
        'h_imag': shared['h_imag'],
        'k_star': shared['k_star'],
    }


def main():
    inputs = _load_inputs()
    result = derive(**inputs)

    print(f"# PREDICT name=m_c value={result['predicted']:.15f}")
    print()
    print("m_c = m_t * (f_mid / f_max)^2   [up sector, n=2]")
    print(f"  m_t                  = {inputs['m_t']:.6f}  GeV   (framework anchor)")
    print(f"  alpha_1_bare         = {result['checks']['alpha_1_bare']:.15f}  (= (2/3)^8)")
    print(f"  alpha_1_full         = {result['checks']['alpha_1_full']:.15f}  (= (5/3)*(2/3)^8)")
    print(f"  f(2)                 = {result['checks']['f_n_of_2']:.15f}  (= 1 + 8/20 = 7/5)")
    print(f"  eps_2^2              = {result['checks']['eps_sq']:.15f}")
    print(f"  delta_2              = {result['checks']['delta']:.15f}  (= 2/27)")
    print(f"  f_min, f_mid, f_max  = {result['checks']['factors_sorted']}")
    print(f"  (f_mid/f_max)^2      = {result['checks']['ratio_sq_mc_over_mt']:.15f}")
    print(f"  m_c                  = {result['predicted']:.15f}  GeV")


if __name__ == '__main__':
    main()
