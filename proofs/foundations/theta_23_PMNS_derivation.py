#!/usr/bin/env python3
"""
---
derives: theta_23_PMNS
inputs:
  - alpha_1
  - h_walker_eigenvalue
script_version: 1.0.0
doc: docs/parameters/target_parameters.md
doc_section: §"Headline"
doc_version_required: 1.0.0
mechanism: mass_squared_2x2
rigor_status: closed
---

theta_23_PMNS = degrees(atan((1 + alpha_1_full) / (1 - alpha_1_full)))
             = 45° + degrees(atan(alpha_1_full))

with alpha_1_full = (Im(h)/Re(h))^2 * alpha_1_bare
                  = tan^2(arg h) * (2/3)^8
                  = (5/3) * (2/3)^8   at the srs P-point

The (5/3) factor is a spectral identity — NOT an ansatz. At the srs P-point,
the Hashimoto walker eigenvalue is h = (sqrt(3) + i*sqrt(5))/2, so
tan^2(arg h) = Im(h)^2 / Re(h)^2 = 5/3 exactly. This value is forced by
Ramanujan saturation |h|^2 = k-1: under any A(P)-eigenvalue perturbation,
d|h|^2/dlambda = 0 at first order, so the perturbative response dh is
tangent to the circle |h|^2 = 2 (perpendicular to h), and the ratio
(Re dh)^2 / (Im dh)^2 collapses to Im(h)^2 / Re(h)^2 = 5/3. Sympy-verified
(`theta23_ramanujan_closure_2026-04-15.md` §"Headline").

The structural chain:
  1. TBM 45° baseline from C_3-protected doubly-degenerate h eigenspace at
     P (P2 Theorem 3).
  2. Dark perturbation splits the {h, h*} sector symmetrically (sigma_z = 0
     from Tr theorem in srs_theta23_sigma_x.py), giving eigenvalue ratio
     (1 + alpha_1_full) / (1 - alpha_1_full).
  3. The (5/3) prefactor in alpha_1_full = (5/3) * alpha_1_bare is forced by
     Ramanujan saturation + perpendicularity (this script's Identity 1).

Framework-internal: inputs are alpha_1 (= alpha_1_bare = (2/3)^8) and
h_walker_eigenvalue (value_real, value_imag components). No observed theta_23
enters.
"""

import math
import sys


def derive(alpha_1_bare: float, h_re: float, h_im: float) -> dict:
    """Return theta_23_PMNS in degrees.

    Parameters
    ----------
    alpha_1_bare : float
        Bare chirality coupling = (2/3)^8 on srs (alpha_1 row in YAML).
    h_re, h_im : float
        Real and imaginary components of the Hashimoto walker eigenvalue
        h = (sqrt(3) + i*sqrt(5))/2 at the srs P-point.

    Returns
    -------
    dict with 'predicted' (degrees) and 'checks'.
    """
    if h_re == 0:
        raise ValueError("h must have non-zero real part at the P-point")
    if alpha_1_bare <= 0 or alpha_1_bare >= 1:
        raise ValueError(f"alpha_1_bare out of range: {alpha_1_bare}")

    spectral_ratio = (h_im / h_re) ** 2  # tan^2(arg h) = 5/3 at srs P-point
    alpha_1_full = spectral_ratio * alpha_1_bare
    if abs(alpha_1_full) >= 1:
        raise ValueError(f"alpha_1_full out of range for arctan form: {alpha_1_full}")

    tan_theta = (1 + alpha_1_full) / (1 - alpha_1_full)
    theta_rad = math.atan(tan_theta)
    theta_deg = math.degrees(theta_rad)

    # Rational check: spectral ratio should be exactly 5/3 at the P-point
    spectral_ratio_exact = 5.0 / 3.0
    spectral_ratio_err = abs(spectral_ratio - spectral_ratio_exact)

    return {
        'predicted': theta_deg,
        'checks': {
            'alpha_1_bare': alpha_1_bare,
            'h_re': h_re,
            'h_im': h_im,
            'spectral_ratio_computed': spectral_ratio,
            'spectral_ratio_exact_5_over_3': spectral_ratio_exact,
            'spectral_ratio_error': spectral_ratio_err,
            'alpha_1_full': alpha_1_full,
            'tan_ratio': tan_theta,
            'theta_23_rad': theta_rad,
            'tbm_baseline_deg': 45.0,
            'dark_shift_deg': theta_deg - 45.0,
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    alpha_1_bare = (2.0 / 3.0) ** 8
    h_re = math.sqrt(3) / 2.0
    h_im = math.sqrt(5) / 2.0

    inputs = {'alpha_1_bare': alpha_1_bare, 'h_re': h_re, 'h_im': h_im}
    result = derive(**inputs)
    c = result['checks']

    print(f"# PREDICT name=theta_23_PMNS value={result['predicted']:.12f}")
    print()
    print("theta_23_PMNS = degrees(atan((1 + alpha_1_full) / (1 - alpha_1_full)))")
    print(f"  alpha_1_bare            = {c['alpha_1_bare']:.15f}")
    print(f"  h                       = {c['h_re']:.12f} + {c['h_im']:.12f} i")
    print(f"  tan^2(arg h) = Im^2/Re^2 = {c['spectral_ratio_computed']:.15f}")
    print(f"  exact 5/3               = {c['spectral_ratio_exact_5_over_3']:.15f}")
    print(f"  ratio error             = {c['spectral_ratio_error']:.2e}")
    print(f"  alpha_1_full            = {c['alpha_1_full']:.15f}")
    print(f"  (1 + a)/(1 - a)         = {c['tan_ratio']:.15f}")
    print(f"  theta_23                = {result['predicted']:.10f}°")
    print(f"  shift from TBM 45°      = {c['dark_shift_deg']:+.6f}°")


if __name__ == '__main__':
    main()
