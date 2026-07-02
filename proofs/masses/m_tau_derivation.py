#!/usr/bin/env python3
"""
---
derives: m_tau
inputs:
  - v
  - alpha_1
  - h_walker_eigenvalue
  - k_star
script_version: 1.0.0
doc: docs/parameters/target_parameters.md
doc_section: lepton Yukawa sector / tau mass corollary
doc_version_required: 0.0.1
mechanism: structural
rigor_status: rigor_route_specified
---

m_tau = v * y_tau with y_tau = alpha_1_full / k*^2 and alpha_1_full
= tan^2(arg h) * alpha_1_bare.

Derivation chain (ytau_corollary):

  1. lambda = 2 * alpha_1_bare (Higgs quartic, Cl(2) channels, edge-transitive)
  2. y_tau = alpha_1_full / k*^2 where alpha_1_full = (5/3) * alpha_1_bare
     with 5/3 = tan^2(arg h) the chirality class factor.
     Each of the two fermion fields in the Yukawa vertex contributes 1/k*
     from edge-mode projection on the trivalent vertex.
  3. m_tau = v * y_tau (standard-model mass from Yukawa and vev).

Inputs (all framework rows):
  v                    - Higgs vev (A-, srs_delta_sq_theorem; FSS + vertex self-energy)
  alpha_1              - bare NB walk survival (2/3)^8 on k=3 g=10 graph
  h_walker_eigenvalue  - complex Hashimoto eigenvalue at the P-point;
                         tan^2(arg h) = (Im h / Re h)^2 = (3*k*-4)/k* = 5/3
  k_star               - srs vertex valence, 3

Because v is A- (not theorem), m_tau inherits A- under the strict grading
rule (1b): no coefficient is tuned to observation, but the v chain still has
the unproven MF -> Curie-Weiss FSS equivalence step. The entire Koide triplet
inherits the same grade through m_tau.
"""

import sys


def derive(v: float,
           alpha_1: float,
           h_real: float,
           h_imag: float,
           k_star: float) -> dict:
    """m_tau from v, alpha_1_bare, h_walker_eigenvalue components, k_star."""
    if v <= 0:
        raise ValueError(f"v must be > 0; got {v}")
    if alpha_1 <= 0:
        raise ValueError(f"alpha_1 must be > 0; got {alpha_1}")
    if h_real == 0:
        raise ValueError("Re(h) must be non-zero")
    if k_star <= 0:
        raise ValueError(f"k_star must be > 0; got {k_star}")

    tan_sq_arg_h = (h_imag / h_real) ** 2
    alpha_1_full = tan_sq_arg_h * alpha_1
    y_tau = alpha_1_full / (k_star ** 2)
    m_tau = v * y_tau
    return {
        'predicted': m_tau,
        'checks': {
            'tan_sq_arg_h': tan_sq_arg_h,
            'alpha_1_bare': alpha_1,
            'alpha_1_full': alpha_1_full,
            'y_tau': y_tau,
            'v': v,
        },
    }


def main():
    import math

    # Framework constants (hardcoded, no YAML dependency)
    v = 245.64           # GeV, framework A-
    alpha_1 = (2.0 / 3.0) ** 8
    h_real = math.sqrt(3) / 2.0
    h_imag = math.sqrt(5) / 2.0
    k_star = 3.0

    inputs = {'v': v, 'alpha_1': alpha_1, 'h_real': h_real, 'h_imag': h_imag, 'k_star': k_star}
    result = derive(**inputs)

    print(f"# PREDICT name=m_tau value={result['predicted']:.15f}")
    print()
    print("m_tau = v * alpha_1_full / k*^2,  alpha_1_full = tan^2(arg h) * alpha_1_bare")
    print(f"  v                    = {inputs['v']:.6f}  GeV  (framework A-)")
    print(f"  alpha_1_bare         = {inputs['alpha_1']:.15f}")
    print(f"  Re(h), Im(h)         = {inputs['h_real']:.15f}, {inputs['h_imag']:.15f}")
    print(f"  tan^2(arg h)         = {result['checks']['tan_sq_arg_h']:.15f}  (= 5/3)")
    print(f"  alpha_1_full         = {result['checks']['alpha_1_full']:.15f}")
    print(f"  y_tau                = {result['checks']['y_tau']:.15f}")
    print(f"  k*                   = {inputs['k_star']:.0f}")
    print(f"  m_tau                = {result['predicted']:.15f}  GeV")


if __name__ == '__main__':
    main()
