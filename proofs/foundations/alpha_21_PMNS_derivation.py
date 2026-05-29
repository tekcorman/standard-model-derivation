#!/usr/bin/env python3
"""
---
derives: alpha_21_PMNS
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

alpha_21_PMNS = arg(h^g) mod 360°

The first PMNS Majorana phase is the argument of the Hashimoto walker
eigenvalue raised to the girth power, evaluated at the P-point where h is
complex. At srs (g=10), arg(h) = arctan(Im h / Re h) = arctan(sqrt(5)/sqrt(3))
= arctan(sqrt(5/3)) ≈ 52.2388°, and arg(h^g) = g * arg(h) = 522.39° ≡ 162.39°
(mod 360°).

Structurally, this is the phase accumulated by a non-backtracking closed walk
of length g on the srs graph at its Ramanujan P-point — the shortest-cycle
holonomy of the chirality-selected walker. Chirality (+Im branch) comes from
the framework parity convention; the exponent g is the structural girth, not
a free parameter.

Framework-internal: inputs are h_walker_eigenvalue (components Re(h), Im(h)
loaded from the YAML row) and g_girth. No observed phase data enters.
"""

import math
import sys


def derive(h_re: float, h_im: float, g_girth: int) -> dict:
    """Return arg(h^g) in degrees, wrapped to [0, 360).

    Parameters
    ----------
    h_re, h_im : float
        Real and imaginary components of the Hashimoto walker eigenvalue h
        at the P-point.
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
    h_g = h ** g_girth
    phase_rad = math.atan2(h_g.imag, h_g.real)
    phase_deg = math.degrees(phase_rad) % 360.0
    arg_h_deg = math.degrees(math.atan2(h_im, h_re))
    return {
        'predicted': phase_deg,
        'checks': {
            'h_re': h_re,
            'h_im': h_im,
            'g_girth': g_girth,
            'arg_h_deg': arg_h_deg,
            'raw_g_arg_h_deg': g_girth * arg_h_deg,
            'h_g_re': h_g.real,
            'h_g_im': h_g.imag,
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

    print(f"# PREDICT name=alpha_21_PMNS value={result['predicted']:.6f}")
    print()
    print("alpha_21_PMNS = arg(h^g) mod 360°")
    print(f"  h              = {c['h_re']:.12f} + {c['h_im']:.12f} i")
    print(f"  arg(h)         = {c['arg_h_deg']:.6f}°")
    print(f"  g              = {c['g_girth']}")
    print(f"  g * arg(h)     = {c['raw_g_arg_h_deg']:.6f}° (raw, before mod)")
    print(f"  alpha_21_PMNS  = {result['predicted']:.6f}°")


if __name__ == '__main__':
    main()
