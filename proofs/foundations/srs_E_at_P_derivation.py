#!/usr/bin/env python3
"""
---
derives: srs_E_at_P
inputs:
  - k_star
script_version: 1.0.0
doc: TODO
doc_section: TODO
doc_version_required: 0.0.1
mechanism: structural
rigor_status: closed
---

srs_E_at_P = +sqrt(k*) = sqrt(3)

Positive adjacency eigenvalue of the 4x4 Bloch Hamiltonian H(k) at the P-point
k_P = (1/4, 1/4, 1/4) of the srs primitive BZ. The srs net (space group
I4_132) has 4 atoms per cell at Wyckoff 8a base positions; find_bonds() from
proofs/common.py returns the NN connectivity, and bloch_H(k_P, bonds) builds
the tight-binding 4x4 matrix.

Diagonalizing H(k_P) produces eigenvalues {+sqrt(3), +sqrt(3), -sqrt(3),
-sqrt(3)} — doubly degenerate ±sqrt(k*). The doubly-degenerate multiplicity
is a separate C3-protected theorem (B_P_doubly_degenerate_h / P2 Theorem 3).
This script produces the *value* of the positive branch, which feeds the
Hashimoto equation h^2 - E*h + (k*-1) = 0 downstream in h_walker_eigenvalue.

No existing-physics inputs: the Bloch Hamiltonian is built from the srs graph
alone (k* = 3, Wyckoff 8a positions, I4_132 primitive vectors — all framework
structural). The numerical diagonalization is standard linear algebra.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, bloch_H, K_STAR


K_P = (0.25, 0.25, 0.25)


def derive(k_star: int) -> dict:
    """Return E_P = +sqrt(k*) by diagonalizing H(k_P) on the srs Bloch basis.

    Parameters
    ----------
    k_star : int
        Vertex valence of the srs graph (must equal 3 for srs).

    Returns
    -------
    dict with keys:
        predicted : float — positive adjacency eigenvalue at P
        checks : dict with eigenvalues, degeneracy, char poly residual
    """
    if k_star != K_STAR:
        raise ValueError(
            f"srs_E_at_P is specialized to the srs lattice (k*={K_STAR}); "
            f"got k_star={k_star}"
        )

    bonds = find_bonds()
    H_P = bloch_H(K_P, bonds)
    evals = np.sort(np.real(la.eigvalsh(H_P)))

    expected = np.array([-np.sqrt(k_star), -np.sqrt(k_star),
                         +np.sqrt(k_star), +np.sqrt(k_star)])
    residual = float(la.norm(evals - expected))
    if residual > 1e-9:
        raise RuntimeError(
            f"H(k_P) eigenvalues {evals} do not match expected "
            f"±sqrt(k*) doubly-degenerate (residual {residual:.2e})"
        )

    E_P = float(evals[-1])
    return {
        'predicted': E_P,
        'checks': {
            'k_star': k_star,
            'k_point': K_P,
            'eigenvalues_sorted': evals.tolist(),
            'multiplicity_pos': 2,
            'multiplicity_neg': 2,
            'char_poly_residual': residual,
            'interpretation': 'positive branch A(P) eigenvalue, feeds Hashimoto h',
        },
    }


def main():
    # Framework constants (hardcoded, no YAML dependency)
    k_star = 3

    inputs = {'k_star': k_star}
    result = derive(**inputs)

    print(f"# PREDICT name=srs_E_at_P value={result['predicted']:.15f}")
    print()
    print("srs_E_at_P = sqrt(k*) from A(P) char poly (lambda^2 - k*)^2")
    print(f"  k_star               = {inputs['k_star']}")
    print(f"  k_point              = {result['checks']['k_point']}")
    print(f"  H(k_P) eigenvalues   = {result['checks']['eigenvalues_sorted']}")
    print(f"  multiplicity (+E_P)  = {result['checks']['multiplicity_pos']}")
    print(f"  multiplicity (-E_P)  = {result['checks']['multiplicity_neg']}")
    print(f"  char poly residual   = {result['checks']['char_poly_residual']:.2e}")
    print(f"  E_P = +sqrt(k*)      = {result['predicted']:.15f}")


if __name__ == '__main__':
    main()
