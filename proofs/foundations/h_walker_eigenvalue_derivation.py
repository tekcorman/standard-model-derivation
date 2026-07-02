#!/usr/bin/env python3
"""
---
derives: h_walker_eigenvalue
inputs:
  - k_star
  - srs_E_at_P
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: structural
rigor_status: closed
---

h_walker_eigenvalue = |h| = sqrt(k* - 1) = sqrt(2)

The Hashimoto non-backtracking walk eigenvalue h solves the quadratic

    h^2 - E * h + (k* - 1) = 0

where E is an adjacency eigenvalue of the Bloch Hamiltonian at the selected
k-point. At the P-point of the srs primitive BZ, E = srs_E_at_P = sqrt(k*) =
sqrt(3), so the discriminant is E^2 - 4(k*-1) = 3 - 8 = -5, giving

    h = (sqrt(3) + i*sqrt(5)) / 2        (positive-imaginary branch)
    |h|^2 = (3 + 5)/4 = 2 = k* - 1       (Ramanujan saturation)

The sentinel reports |h| = sqrt(2); the components Re(h) = sqrt(3)/2 and
Im(h) = sqrt(5)/2 are stored in the YAML row as value_real and value_imag and
reprinted here as auxiliary output for consumers that need them (the PMNS
CP-phase scripts downstream).

Chirality / branch choice: the positive-imaginary branch is the walker state;
the negative-imaginary branch is its conjugate h' = (sqrt(3) - i*sqrt(5))/2.
Both are needed for alpha_31_PMNS (inter-band phase ratio (h/h')^g); the
conjugate is reconstructed locally from Re/Im in each downstream consumer.

Framework-internal: inputs are k_star (MDL + toggle) and srs_E_at_P (A(P)
char poly). No observed values enter.
"""

import math
import sys


def derive(k_star: int, srs_E_at_P: float) -> dict:
    """Solve h^2 - E*h + (k*-1) = 0 for the Hashimoto walker eigenvalue.

    Parameters
    ----------
    k_star : int
        Vertex valence (>= 2; srs has k* = 3).
    srs_E_at_P : float
        Adjacency eigenvalue at the walker's k-point (srs: +sqrt(k*)).

    Returns
    -------
    dict with keys:
        predicted : float — |h|
        checks : dict with real/imag components, discriminant, Ramanujan
                 saturation test
    """
    if k_star < 2:
        raise ValueError(f"k_star must be >= 2; got {k_star}")

    E = float(srs_E_at_P)
    disc = E * E - 4.0 * (k_star - 1)
    if disc >= 0:
        raise RuntimeError(
            f"Hashimoto discriminant is non-negative (disc={disc}); expected "
            f"complex h at the Ramanujan-saturated P-point."
        )

    Re_h = E / 2.0
    Im_h = math.sqrt(-disc) / 2.0       # positive branch
    mod_h_sq = Re_h * Re_h + Im_h * Im_h
    mod_h = math.sqrt(mod_h_sq)

    # Ramanujan saturation: |h|^2 = k* - 1 exactly for the P-point choice.
    ramanujan_residual = abs(mod_h_sq - (k_star - 1))
    if ramanujan_residual > 1e-12:
        raise RuntimeError(
            f"Ramanujan saturation |h|^2 = k*-1 failed: "
            f"|h|^2 = {mod_h_sq}, k*-1 = {k_star - 1}"
        )

    return {
        'predicted': mod_h,
        'checks': {
            'k_star': k_star,
            'E_at_P': E,
            'discriminant': disc,
            'Re_h': Re_h,
            'Im_h': Im_h,
            'mod_h_squared': mod_h_sq,
            'ramanujan_residual': ramanujan_residual,
            'arg_h_rad': math.atan2(Im_h, Re_h),
            'arg_h_deg': math.degrees(math.atan2(Im_h, Re_h)),
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    k_star = 3
    srs_E_at_P = math.sqrt(k_star)  # sqrt(3), from srs_E_at_P derivation

    inputs = {'k_star': k_star, 'srs_E_at_P': srs_E_at_P}
    result = derive(**inputs)
    c = result['checks']

    print(f"# PREDICT name=h_walker_eigenvalue value={result['predicted']:.15f}")
    print()
    print("h = (E + i*sqrt(4(k*-1) - E^2)) / 2   (Hashimoto at P)")
    print(f"  k_star         = {c['k_star']}")
    print(f"  E_P            = {c['E_at_P']:.15f}")
    print(f"  discriminant   = E^2 - 4(k*-1) = {c['discriminant']:.15f}")
    print(f"  Re(h)          = E/2 = {c['Re_h']:.15f}")
    print(f"  Im(h)          = sqrt(-disc)/2 = {c['Im_h']:.15f}")
    print(f"  |h|^2          = Re^2 + Im^2 = {c['mod_h_squared']:.15f}")
    print(f"  k* - 1         = {c['k_star'] - 1}")
    print(f"  Ramanujan res. = {c['ramanujan_residual']:.2e}")
    print(f"  arg(h)         = {c['arg_h_deg']:.6f}°")
    print(f"  |h|            = {result['predicted']:.15f}")


if __name__ == '__main__':
    main()
