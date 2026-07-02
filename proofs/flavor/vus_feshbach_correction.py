#!/usr/bin/env python3
"""
---
derives: V_us_feshbach_correction
inputs:
  - alpha_1
  - h_walker_eigenvalue
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: feshbach_amplitude
rigor_status: closed
---

V_us Feshbach dark amplitude correction: |Im[Sigma(h)]|
========================================================

Sigma(h) = alpha_1 / h is the Feshbach self-energy on the water-filled
ruliad Q-space (from contour integration with the pole at z = h/r
excluded as a P-space eigenvalue — a direct MDL consequence, not a
convention).

The observable fractional correction is extracted as |Im[Sigma(h)]|,
forced uniquely by walk-length independence across V_us and m_nu:

  |Im[Sigma(h)]| = alpha_1 * Im(h) / |h|^2
                 = alpha_1 * (sqrt(5)/2) / 2
                 = alpha_1 * sqrt(5) / 4

With alpha_1 = (2/3)^8 and h = (sqrt(3) + i*sqrt(5))/2 on srs (k* = 3),
this gives 0.02181... — the shared fractional dark correction that
v_us, m_nu2, m_nu3 all receive.

See vus_feshbach_derivation.py (legacy) for the full contour-integral
verification, MDL uniform-density argument, walk-length independence
check against m_nu, and self-consistent O(alpha_1^2) bound.

This script extracts only the closed-form correction factor from that
derivation so V_us_feshbach_correction stands as its own clean row in
the parameter DAG. The legacy monolithic script stays in place until
the neutrino sector is Phase-A-converted (which also consumes this
correction).
"""

import sys


def derive(alpha_1: float, h_real: float, h_imag: float) -> dict:
    """Return the V_us Feshbach fractional correction |Im[Sigma(h)]|.

    Parameters
    ----------
    alpha_1 : float
        Bare NB walk survival at girth-2 = (k*-1)/k* ^ (g-2). On srs
        with k*=3, g=10: (2/3)^8.
    h_real : float
        Re(h), the real part of the Hashimoto walk eigenvalue at the
        srs P-point.
    h_imag : float
        Im(h), the imaginary part of h.

    Returns
    -------
    dict with keys:
        predicted : float — |Im[Sigma(h)]|
        checks : dict of intermediate quantities (Sigma components,
                 alternative extractions ruled out by walk-length
                 independence).
    """
    abs_h_sq = h_real * h_real + h_imag * h_imag
    # Sigma(h) = alpha_1 / h
    # Im[Sigma] = alpha_1 * Im(1/h) = alpha_1 * (-h_imag / |h|^2)
    # |Im[Sigma]| = alpha_1 * |h_imag| / |h|^2
    correction = alpha_1 * abs(h_imag) / abs_h_sq
    return {
        'predicted': correction,
        'checks': {
            'alpha_1': alpha_1,
            'h_real': h_real,
            'h_imag': h_imag,
            'abs_h_sq': abs_h_sq,
            'Re_Sigma': alpha_1 * h_real / abs_h_sq,
            'Im_Sigma': -alpha_1 * h_imag / abs_h_sq,
            'abs_Sigma': alpha_1 / abs_h_sq ** 0.5,
        },
    }


def main():
    import math

    # Framework constants (hardcoded, no YAML dependency)
    alpha_1 = (2.0 / 3.0) ** 8
    h_real = math.sqrt(3) / 2.0
    h_imag = math.sqrt(5) / 2.0

    inputs = {'alpha_1': alpha_1, 'h_real': h_real, 'h_imag': h_imag}
    result = derive(**inputs)

    print(f"# PREDICT name=V_us_feshbach_correction value={result['predicted']:.15f}")
    print()
    print("V_us Feshbach dark amplitude correction")
    print(f"  inputs:")
    print(f"    alpha_1         = {inputs['alpha_1']:.15f}")
    print(f"    Re(h)           = {inputs['h_real']:.15f}")
    print(f"    Im(h)           = {inputs['h_imag']:.15f}")
    print(f"  |h|^2             = {result['checks']['abs_h_sq']:.15f}")
    print(f"  Sigma(h) components (ruled-out alternatives):")
    print(f"    Re(Sigma)       = {result['checks']['Re_Sigma']:.15f}")
    print(f"    Im(Sigma)       = {result['checks']['Im_Sigma']:.15f}")
    print(f"    |Sigma|         = {result['checks']['abs_Sigma']:.15f}")
    print(f"  |Im[Sigma(h)]|    = {result['predicted']:.15f}  (walk-length-independent; selected)")


if __name__ == '__main__':
    main()
