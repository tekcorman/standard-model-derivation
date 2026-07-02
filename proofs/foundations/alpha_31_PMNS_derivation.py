#!/usr/bin/env python3
"""
---
derives: alpha_31_PMNS
inputs:
  - h_walker_eigenvalue
  - g_girth
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: structural
rigor_status: closed
---

alpha_31_PMNS = arg((h / h*)^g) mod 360°

Second PMNS Majorana phase from the inter-band Hashimoto phase ratio at the
P-point. The walker eigenvalue h and its conjugate h* = (sqrt(3) -
i*sqrt(5))/2 live in the two C3-related bands at P; the ratio h/h* is a pure
phase of unit modulus with arg(h/h*) = 2*arg(h). Raising to the girth
exponent gives

    arg((h/h*)^g) = 2*g*arg(h)

At srs (g=10): 2*10*52.2388° = 1044.776° ≡ 324.78° (mod 360°).

Equivalently this is arg(h^g) - arg(h*^g) = g*(arg(h) - arg(h*)), which is
the relative holonomy of the two band walkers over one girth-cycle, making
alpha_31 a structural inter-band phase, not a free parameter.

Framework-internal: inputs are h_walker_eigenvalue and g_girth.
"""

import math
import sys


def derive(h_re: float, h_im: float, g_girth: int) -> dict:
    """Return arg((h/h*)^g) in degrees, wrapped to [0, 360).

    Parameters
    ----------
    h_re, h_im : float
        Real and imaginary components of h at the P-point.
    g_girth : int
        Graph girth (for srs: 10).

    Returns
    -------
    dict with 'predicted' (degrees) and 'checks'.
    """
    if g_girth < 1:
        raise ValueError(f"g_girth must be >= 1; got {g_girth}")
    if h_im == 0:
        raise ValueError("h must have non-zero imaginary part at a Ramanujan P-point")
    h = complex(h_re, h_im)
    h_conj = h.conjugate()
    ratio = h / h_conj
    ratio_g = ratio ** g_girth
    phase_rad = math.atan2(ratio_g.imag, ratio_g.real)
    phase_deg = math.degrees(phase_rad) % 360.0
    arg_ratio_deg = math.degrees(math.atan2(ratio.imag, ratio.real))
    return {
        'predicted': phase_deg,
        'checks': {
            'h_re': h_re,
            'h_im': h_im,
            'g_girth': g_girth,
            'arg_ratio_deg': arg_ratio_deg,
            'two_g_arg_h_deg': 2 * g_girth * math.degrees(math.atan2(h_im, h_re)),
            'ratio_modulus': abs(ratio),
            'ratio_g_re': ratio_g.real,
            'ratio_g_im': ratio_g.imag,
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    h_re = math.sqrt(3) / 2.0
    h_im = math.sqrt(5) / 2.0
    g_girth = 10

    inputs = {'h_re': h_re, 'h_im': h_im, 'g_girth': g_girth}
    result = derive(**inputs)
    c = result['checks']

    print(f"# PREDICT name=alpha_31_PMNS value={result['predicted']:.6f}")
    print()
    print("alpha_31_PMNS = arg((h/h*)^g) mod 360°")
    print(f"  h                = {c['h_re']:.12f} + {c['h_im']:.12f} i")
    print(f"  |h/h*|           = {c['ratio_modulus']:.15f}  (should be 1)")
    print(f"  arg(h/h*)        = {c['arg_ratio_deg']:.6f}° = 2·arg(h)")
    print(f"  g                = {c['g_girth']}")
    print(f"  2·g·arg(h)       = {c['two_g_arg_h_deg']:.6f}° (raw)")
    print(f"  alpha_31_PMNS    = {result['predicted']:.6f}°")


if __name__ == '__main__':
    main()
