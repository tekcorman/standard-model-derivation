#!/usr/bin/env python3
"""
---
derives: delta_CP_PMNS
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

delta_CP_PMNS = arg(h*^(g-1)) mod 360°

Dirac CP phase of the PMNS matrix at the P-point. The Hashimoto walker h at
the chirality-selected branch has complex-conjugate partner h* = (sqrt(3) -
i*sqrt(5))/2; raising h* to the transition exponent (g-1) — one fewer than
the closed-cycle exponent because one edge is held fixed across the CP
transition — and taking the argument gives the Dirac phase.

At srs (g = 10): arg(h*) = -arctan(sqrt(5/3)) ≈ -52.2388°; arg(h*^9) = -9 *
52.2388° = -470.149° ≡ 249.85° (mod 360°).

Option selection: exponent n = g-1 is favored over n = g (which was excluded
at 5.6σ by the Jarlskog cross-check in srs_dcp_exponent.py) because CP
transitions fix the initial/final edge, leaving g-1 free edges in the closed
walk. Not prong-2 numerology: the structural exponent comes from the CP
transition edge-counting argument, with the Jarlskog cross-check providing
an independent framework-internal consistency test (not an empirical fit).

Framework-internal: inputs are h_walker_eigenvalue and g_girth.
"""

import math
import sys


def derive(h_re: float, h_im: float, g_girth: int) -> dict:
    """Return arg(h*^(g-1)) in degrees, wrapped to [0, 360).

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
    if g_girth < 2:
        raise ValueError(f"g_girth must be >= 2 for (g-1) exponent; got {g_girth}")
    h = complex(h_re, h_im)
    h_conj = h.conjugate()
    exponent = g_girth - 1
    h_conj_n = h_conj ** exponent
    phase_rad = math.atan2(h_conj_n.imag, h_conj_n.real)
    phase_deg = math.degrees(phase_rad) % 360.0
    arg_h_conj_deg = math.degrees(math.atan2(h_conj.imag, h_conj.real))
    return {
        'predicted': phase_deg,
        'checks': {
            'h_re': h_re,
            'h_im': h_im,
            'g_girth': g_girth,
            'exponent_gm1': exponent,
            'arg_h_star_deg': arg_h_conj_deg,
            'raw_nm1_arg_h_star_deg': exponent * arg_h_conj_deg,
            'h_conj_n_re': h_conj_n.real,
            'h_conj_n_im': h_conj_n.imag,
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

    print(f"# PREDICT name=delta_CP_PMNS value={result['predicted']:.6f}")
    print()
    print("delta_CP_PMNS = arg(h*^(g-1)) mod 360°")
    print(f"  h                = {c['h_re']:.12f} + {c['h_im']:.12f} i")
    print(f"  arg(h*)          = {c['arg_h_star_deg']:.6f}°")
    print(f"  g-1              = {c['exponent_gm1']}")
    print(f"  (g-1)·arg(h*)    = {c['raw_nm1_arg_h_star_deg']:.6f}° (raw)")
    print(f"  delta_CP_PMNS    = {result['predicted']:.6f}°")


if __name__ == '__main__':
    main()
