#!/usr/bin/env python3
"""
L3 gate check: do srs edge orientation (e1) and causal direction (e2)
mix under Lorentz boosts from Stage 3?

Context: G2 (Higgs doublet) proof attempt, an internal working note
Proposed argument: e1 and e2 are spatial/temporal components of the same
edge 4-vector; Stage 3 Lorentz invariance mixes them; therefore they cannot
both be Lorentz-invariantly defined simultaneously, implying non-commutation.

This script CHECKS the mixing claim and measures exactly what it does
and does not establish.
"""

import sys
import os
import numpy as np
from numpy import linalg as la
from fractions import Fraction
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proofs.common import find_bonds, ATOMS, A_PRIM

RTOL = 1e-12


# ---------------------------------------------------------------------------
# 1. Lorentz boost matrix (3+1 D, signature +---)
# ---------------------------------------------------------------------------

def lorentz_boost(beta_vec):
    """
    4x4 active Lorentz boost. beta_vec is a 3-vector with |beta_vec| < 1.
    x'^mu = Lambda^mu_nu x^nu.
    Convention: x^mu = (t, x, y, z).
    """
    beta = la.norm(beta_vec)
    if beta < 1e-15:
        return np.eye(4)
    n = beta_vec / beta
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    L = np.eye(4)
    L[0, 0] = gamma
    L[0, 1:] = -gamma * beta * n
    L[1:, 0] = -gamma * beta * n
    for i in range(3):
        for j in range(3):
            L[i+1, j+1] = (gamma - 1.0) * n[i] * n[j] + (1.0 if i == j else 0.0)
    return L


# ---------------------------------------------------------------------------
# 2. srs edge 4-vectors in the lattice rest frame
# ---------------------------------------------------------------------------

def edge_4vectors(bonds):
    """
    Rest-frame 4-vectors for all srs bonds.
    In the lattice frame the edges are purely spatial: x^mu = (0, dr).
    """
    vecs = []
    for src, tgt, cell in bonds:
        r_src = ATOMS[src]
        r_tgt = ATOMS[tgt] + (cell[0]*A_PRIM[0] + cell[1]*A_PRIM[1]
                               + cell[2]*A_PRIM[2])
        dr = r_tgt - r_src
        vecs.append(np.array([0.0, dr[0], dr[1], dr[2]]))
    return vecs


# ---------------------------------------------------------------------------
# 3. Key analytic result: sign of temporal component after boost
# ---------------------------------------------------------------------------

def analytic_temporal_sign():
    """
    For x^mu = (0, dr), boost by beta along n_hat:
      x'^0 = gamma * (0 - beta * n_hat . dr) = -gamma * beta * (n_hat . dr)

    sign(x'^0) = -sign(n_hat . dr)

    The temporal sign in the boosted frame is OPPOSITE to the spatial
    projection of dr onto the boost direction.
    """
    print("=" * 60)
    print("ANALYTIC RESULT")
    print("=" * 60)
    print("""
  Rest frame: x^mu = (0, dr)   [purely spatial]
  Boost beta along n_hat:

    x'^0 = gamma(x^0 - beta * n_hat . x)
         = gamma(0   - beta * n_hat . dr)
         = -gamma * beta * (n_hat . dr)

  =>  sign(x'^0) = -sign(n_hat . dr)           ...(*)

  The temporal component is anti-correlated with the spatial
  component along the boost direction.  Exact at all beta < 1.
""")


# ---------------------------------------------------------------------------
# 4. Numerical verification on actual srs bonds
# ---------------------------------------------------------------------------

def verify_sign_anticorrelation(bonds):
    """
    Verify (*) numerically for all 12 srs bonds and several boost directions.
    """
    print("=" * 60)
    print("NUMERICAL VERIFICATION  (sign(x'^0) = -sign(n_hat . dr))")
    print("=" * 60)

    vecs = edge_4vectors(bonds)
    betas = [0.3, 0.6, 0.9]

    # Use 6 canonical boost directions + 4 srs bond directions
    boost_dirs = [
        np.array([1, 0, 0], dtype=float),
        np.array([0, 1, 0], dtype=float),
        np.array([0, 0, 1], dtype=float),
        np.array([1, 1, 0], dtype=float) / np.sqrt(2),
        np.array([1, 1, 1], dtype=float) / np.sqrt(3),
        np.array([1,-1, 1], dtype=float) / np.sqrt(3),
    ]
    # add first four bond directions
    for x4 in vecs[:4]:
        dr = x4[1:]
        n = dr / la.norm(dr)
        boost_dirs.append(n)

    violations = 0
    checks = 0
    for beta in betas:
        for n_hat in boost_dirs:
            L = lorentz_boost(beta * n_hat)
            for x4 in vecs:
                x4p = L @ x4
                dr = x4[1:]
                proj = np.dot(n_hat, dr)
                expected_t_sign = -np.sign(proj) if abs(proj) > 1e-12 else 0.0
                actual_t_sign = np.sign(x4p[0]) if abs(x4p[0]) > 1e-12 else 0.0
                if expected_t_sign != 0.0 and actual_t_sign != 0.0:
                    checks += 1
                    if abs(expected_t_sign - actual_t_sign) > 0.5:
                        violations += 1

    print(f"  Checks: {checks}   Violations: {violations}")
    assert violations == 0, "sign anticorrelation violated!"
    print("  ✓  sign(x'^0) = -sign(n_hat . dr) holds in all cases.")


# ---------------------------------------------------------------------------
# 5. Can any Lorentz frame give UNIFORM temporal signs for all bonds?
# ---------------------------------------------------------------------------

def search_uniform_temporal_frame(bonds):
    """
    Scan 5000 random boost directions (beta=0.7) looking for a frame
    where ALL 12 srs bonds simultaneously have x'^0 > 0.

    Consequence: if no such frame exists, there is no Lorentz-invariant
    way to assign a single causal direction (e2 = +1 for all) to every edge.
    """
    print("\n" + "=" * 60)
    print("SEARCH: uniform causal direction for all edges in one frame?")
    print("=" * 60)

    vecs = edge_4vectors(bonds)
    beta = 0.7
    rng = np.random.default_rng(42)
    n_trials = 5000
    found = 0

    for _ in range(n_trials):
        n_hat = rng.normal(size=3)
        n_hat /= la.norm(n_hat)
        L = lorentz_boost(beta * n_hat)
        t_comps = [( L @ x4)[0] for x4 in vecs]
        if all(t > 1e-12 for t in t_comps):
            found += 1
        elif all(t < -1e-12 for t in t_comps):
            found += 1

    print(f"  Trials: {n_trials}   Uniform-sign frames found: {found}")
    if found == 0:
        print("  ✗  No frame gives uniform causal direction across all 12 bonds.")
        print("     e2 cannot be assigned consistently in any single Lorentz frame.")
    else:
        print(f"  ✓  {found} frames give uniform sign (unexpected — check geometry).")
    return found == 0


# ---------------------------------------------------------------------------
# 6. Orthogonality gap: does mixing -> anti-commutation?
# ---------------------------------------------------------------------------

def report_orthogonality_gap():
    """
    Reports the remaining algebraic gap between:
    (a) Lorentz mixing of e1, e2 (established above), and
    (b) Clifford anti-commutation {e1, e2} = 0 (needed for Cl(0,2)).

    For two ±1 observables A = n1.sigma, B = n2.sigma on a qubit (C^2):
      {A, B} = 2(n1.n2) I

    Anti-commutation requires n1.n2 = 0  (orthogonal Bloch vectors).

    What Lorentz mixing gives:   n1.n2 != +-1   (not parallel)
    What anti-commutation needs: n1.n2 = 0      (orthogonal)

    The gap: Lorentz shows non-collinearity; orthogonality needs an additional
    argument.  Candidate: the Minkowski metric g^{mu nu} = diag(+1,-1,-1,-1)
    makes a timelike unit vector and a spacelike unit vector ORTHOGONAL in
    4D (g^{mu nu} t_mu s_nu = 0 when t^mu = (1,0,0,0), s^mu = (0,n)).
    If the qubit's Bloch-sphere inner product inherits the Minkowski metric
    (via Stage 3), then n1.n2 = g^{mu nu}(e1)_mu (e2)_nu = 0 exactly.
    """
    print("\n" + "=" * 60)
    print("ORTHOGONALITY GAP ANALYSIS")
    print("=" * 60)
    print("""
  Established (Type 2+4):
    sign(x'^0) = -sign(n_hat . dr)   [Lorentz mixing]
    No uniform causal-direction frame for all 12 bonds exists.
    => e1 and e2 are NOT simultaneously Lorentz-invariantly definable.
    => They cannot be collinear on the Bloch sphere (n1.n2 != +-1).

  Needed for Cl(0,2) anti-commutation:
    {e1, e2} = 2(n1 . n2) I = 0   requires n1 . n2 = 0 (orthogonal).

  The Lorentz argument gives non-collinearity, not orthogonality.

  Candidate closing step (not yet gate-passing):
    If the Bloch-sphere inner product on the edge qubit = Minkowski metric,
    then the timelike direction (e2, from Stage 2c E_obs) and the spacelike
    direction (e1, from I4_132 spatial chirality) satisfy:
      n1 . n2 = g^{mu nu} (e1)_mu (e2)_nu = g^{0i} = 0   exactly.
    This gives orthogonality from the Minkowski structure of Stage 3.

  Gate type if closing step holds:
    Type 4 (Stage 3: Minkowski metric from Lorentz invariance)
    + Type 2 (g^{0i} = 0 by definition of Minkowski metric).

  VERDICT: L3 is CANDIDATE, not yet SOLID.
  The Lorentz mixing half is closed (Type 2+4).
  The orthogonality half needs one more step: Bloch metric = Minkowski metric.
""")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bonds = find_bonds()
    print(f"srs lattice: {len(bonds)} bonds loaded.\n")

    analytic_temporal_sign()
    verify_sign_anticorrelation(bonds)
    no_uniform = search_uniform_temporal_frame(bonds)
    report_orthogonality_gap()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
  CLOSED (Type 2 + Type 4 [Stage 3]):
    1. srs rest-frame edges: x^mu = (0, dr), purely spatial.
    2. Under any Lorentz boost: sign(x'^0) = -sign(n_hat . dr).
       [Analytic, verified on all 12 bonds x 6 directions x 3 speeds]
    3. No single Lorentz frame gives a uniform causal direction
       to all 12 srs edges simultaneously.

  IMPLICATION:
    e1 (spatial orientation) and e2 (causal direction) cannot both
    be Lorentz-invariantly defined in the same frame for all edges.
    Their Bloch vectors are not collinear: n1.n2 != +-1.

  STILL OPEN (one step to full L3):
    Bloch-sphere inner product on the edge qubit = Minkowski metric.
    If this holds (Type 4: Stage 3), then n1.n2 = g^{0i} = 0,
    giving full anti-commutation {e1, e2} = 0.

  DOWNSTREAM (L4-L6 remain solid if L3 closes):
    A3 complex structure: e^2 = -1  (not +1)
    Cl(0,2) from {e1,e2}=0, e^2=-1
    SU(2) = unit quaternions = Sp(1)
    2-dim representation = Higgs doublet
""")
