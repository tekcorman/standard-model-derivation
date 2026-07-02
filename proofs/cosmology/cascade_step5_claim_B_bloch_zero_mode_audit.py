#!/usr/bin/env python3
"""
Cascade Step 5 Claim B — Bloch zero-mode persistence: STRUCTURAL AUDIT

The Session-1 scoping doc (cascade_step5_compression_integral_session1_scoping_2026-05-06.md)
identified B-route 2 (Bloch zero-mode persistence) as the most promising
structural rescue for Claim B after the Model 1 audit (cascade_step5_claim_B_persistence_audit.py)
went NEGATIVE.

This file AUDITS B-route 2 STRUCTURALLY before committing to a numerical
implementation. The claim under audit:

    The cosmological IC directional anisotropy persists in the spatial
    k = 0 Bloch mode of the substrate's per-vertex direction distribution
    under translation-invariant Markov dynamics, because λ(k = 0) = 0.

The audit's key observation: Bloch decomposition addresses spatial mode
preservation. Whether the k = 0 mode preserves DIRECTIONAL (per-vertex
internal-state) anisotropy depends entirely on the LOCAL on-site dynamics
at each lattice vertex.

LOCAL STATE SPACE
-----------------
At each lattice vertex, the substrate has 24 srs directed bonds, each with
a binary Beta-state label (F = Beta(1,1), P = Beta(2,1)). The per-vertex
state space is therefore 2^24-dimensional. The local generator L_local is
the renewal Markov chain on this space:
    at each Planck time, sample one direction d ∈ {1, ..., 24} uniformly;
    if state[d] = F, flip to P with prob P_fresh = 1/2;
    if state[d] = P, flip to F with prob P_disconfirm = 1/3.

Bloch zero-mode dynamics is exactly L_local.

STRUCTURAL ARGUMENT
-------------------
The local generator L_local has a SYMMETRY: it commutes with permutations
of the 24 direction labels. (At each step, the SAME uniform sampling weight
1/24 applies to every direction; the F↔P transition rates depend only on
the local state, not on which direction is sampled.)

Under this S_24 permutation symmetry, the unique stationary distribution
of L_local must be ALSO S_24-invariant. The symmetry forces each direction's
marginal Beta state to have IDENTICAL distribution at stationary.

By detailed balance on the per-direction (F, P) chain:
    π_F · P_fresh = π_P · P_disconfirm
    π_F / π_P = (1/3) / (1/2) = 2/3
    π_F = 2/5, π_P = 3/5.

Therefore the stationary per-direction event rate is
    r_∞ = π_F · P_fresh + π_P · P_disconfirm
        = (2/5)(1/2) + (3/5)(1/3)
        = 1/5 + 1/5  =  2/5
INDEPENDENT of direction.

The k = 0 Bloch mode at stationary has ZERO directional anisotropy. The
preservation guaranteed by λ(k = 0) = 0 is preservation of the spatial
ZERO-MODE, not of the directional anisotropy embedded in the IC.

VERIFICATION
------------
This script verifies the symmetry argument with TWO checks:

  Check 1 — small-substrate stationary (4-direction toy model). Build
    the full 2^4 = 16-dim local generator; find its stationary state by
    eigendecomposition; verify that the per-direction event rate is
    direction-INDEPENDENT.

  Check 2 — symmetry-projected detailed balance (analytic). Apply the
    standard 2-state detailed balance to (F, P) under direction-uniform
    sampling; verify (π_F, π_P) = (2/5, 3/5) and stationary rate 2/5.

Both checks confirm: under direction-uniform local sampling, the
per-vertex stationary state is DIRECTION-UNIFORM; Bloch zero-mode does
NOT preserve IC directional anisotropy.

VERDICT
-------
B-route 2 (Bloch zero-mode persistence) is STRUCTURALLY UNABLE to save
Claim B under the framework's standard renewal Markov dynamics.

For Claim B persistence to hold, the local dynamics would need to either:
    (a) BREAK the S_24 direction-permutation symmetry, e.g., by
        direction-anisotropic sampling (Model 2 from the persistence
        audit) — but the source of the anisotropic sampling is NOT in
        the framework's current axioms;
    (b) Give a degenerate manifold of stationary states parametrized
        by direction (Model 3 NESS) — but the local renewal Markov has
        a UNIQUE stationary by detailed balance, NOT a degenerate
        manifold.

Both routes need structural input the framework does not currently supply.

Combined with the Model 1 negative result (cascade_step5_claim_B_persistence_audit.py),
this audit closes TWO independent rescue routes for Claim B. The empirical
match α = 0.207 ± 0.036 (joint A_dilution + cascade rate-gap, +0.18σ from
ε_toggle) remains UNEXPLAINED at the structural level under direction-
uniform renewal Markov dynamics.

This is an HONEST audit-before-implement finding. The cleaner conclusion
is: the framework's α = ε_toggle identification at the substrate stationary
level is empirically anchored, NOT structurally derived from current axioms.
"""

import os
import sys
from fractions import Fraction
import itertools
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ----------------------------------------------------------------------------
# Check 2 — analytic detailed balance on the per-direction (F, P) chain
# ----------------------------------------------------------------------------

def detailed_balance_per_direction():
    """
    Per-direction (F, P) Markov chain at stationary.

    Conditional on direction being sampled (prob 1/n_dirs per step):
        F → P : prob P_fresh = 1/2
        P → F : prob P_disconfirm = 1/3

    Detailed balance:
        π_F · P_fresh = π_P · P_disconfirm
        π_F / π_P = P_disconfirm / P_fresh = (1/3)/(1/2) = 2/3

    Normalize: π_F + π_P = 1 → π_F = 2/5, π_P = 3/5.
    """
    P_fresh = Fraction(1, 2)
    P_disconfirm = Fraction(1, 3)

    pi_P = Fraction(1) / (Fraction(1) + (P_disconfirm / P_fresh))
    pi_F = (P_disconfirm / P_fresh) * pi_P

    assert pi_F + pi_P == Fraction(1)
    assert pi_F == Fraction(2, 5)
    assert pi_P == Fraction(3, 5)

    # stationary rate = π_F · P_fresh + π_P · P_disconfirm
    r_inf = pi_F * P_fresh + pi_P * P_disconfirm
    assert r_inf == Fraction(2, 5)

    return pi_F, pi_P, r_inf


# ----------------------------------------------------------------------------
# Check 1 — small-substrate full-state stationary (4-direction toy)
# ----------------------------------------------------------------------------

def build_full_state_generator(n_dirs, P_fresh=0.5, P_disconfirm=1.0/3.0):
    """
    Build the LOCAL generator on the FULL 2^n_dirs joint state space
    {F, P}^n_dirs.

    Each state is a tuple (s_0, ..., s_{n_dirs-1}) ∈ {0, 1}^n_dirs (0=F, 1=P).

    Per Planck time, one direction d is sampled uniformly (prob 1/n_dirs).
    State (s_0, ..., s_d, ..., s_{n_dirs-1}) transitions to
    (s_0, ..., 1−s_d, ..., s_{n_dirs-1}) with probability P_flip[s_d]
    where P_flip[F] = P_fresh, P_flip[P] = P_disconfirm.

    This is the FULL local generator — captures the joint dynamics over
    all n_dirs binary state labels.
    """
    n_states = 2 ** n_dirs
    L = np.zeros((n_states, n_states))

    for i in range(n_states):
        bits = [(i >> b) & 1 for b in range(n_dirs)]
        for d in range(n_dirs):
            sample_weight = 1.0 / n_dirs
            P_flip = P_fresh if bits[d] == 0 else P_disconfirm
            j = i ^ (1 << d)  # flip bit d
            rate = sample_weight * P_flip
            L[j, i] += rate
            L[i, i] -= rate

    return L


def find_unique_stationary(L):
    """Find the unique normalized non-negative right null-vector of L."""
    eigvals, eigvecs = np.linalg.eig(L)
    abs_vals = np.abs(eigvals)
    idx = np.argsort(abs_vals)
    # the local generator is irreducible for n_dirs ≥ 2, so null space is 1D
    candidate = np.real(eigvecs[:, idx[0]])
    if candidate.sum() < 0:
        candidate = -candidate
    candidate = candidate / candidate.sum()
    return candidate, float(np.real(eigvals[idx[0]])), float(np.real(eigvals[idx[1]]))


def per_direction_marginals(rho, n_dirs):
    """Compute per-direction P(s_d = P) = π_P(d) from joint stationary ρ."""
    pi_P_per_direction = np.zeros(n_dirs)
    for i in range(2 ** n_dirs):
        for d in range(n_dirs):
            if (i >> d) & 1:
                pi_P_per_direction[d] += rho[i]
    return pi_P_per_direction


def per_direction_event_rate(rho, n_dirs, P_fresh=0.5, P_disconfirm=1.0/3.0):
    """
    Per-direction effective event rate at stationary.

    r(d) = E[P_flip(s_d)] = π_F(d) · P_fresh + π_P(d) · P_disconfirm.
    """
    pi_P_d = per_direction_marginals(rho, n_dirs)
    pi_F_d = 1.0 - pi_P_d
    return pi_F_d * P_fresh + pi_P_d * P_disconfirm


def main():
    print("=" * 76)
    print(" Cascade Step 5 Claim B — Bloch zero-mode persistence: STRUCTURAL AUDIT")
    print("=" * 76)
    print()
    print(" Question: does Bloch zero-mode (k = 0 spatial Fourier mode) preserve")
    print(" the cosmological IC DIRECTIONAL anisotropy under translation-invariant")
    print(" renewal Markov dynamics?")
    print()
    print(" Approach: at k = 0 the spatial coupling vanishes; the dynamics reduces")
    print(" to the LOCAL on-site renewal Markov on {F, P}^n_dirs. We check whether")
    print(" the local stationary state has DIRECTIONAL structure.")
    print()

    # ---- Check 2: analytic detailed balance ----
    print(" Check 2 — analytic detailed balance on per-direction (F, P):")
    pi_F, pi_P, r_inf = detailed_balance_per_direction()
    print(f"   π_F = {pi_F},  π_P = {pi_P}  (detailed balance with P_fresh=1/2, P_disc=1/3)")
    print(f"   stationary per-direction rate r_∞ = π_F · P_fresh + π_P · P_disconfirm")
    print(f"                                    = (2/5)(1/2) + (3/5)(1/3)  =  {r_inf}")
    print(f"   INDEPENDENT of direction (S_n permutation symmetry of the dynamics).")
    print()

    # ---- Check 1: small-substrate full-state numerical stationary ----
    print(" Check 1 — small-substrate full-state stationary (n_dirs = 4):")
    n_dirs = 4
    L = build_full_state_generator(n_dirs)
    rho_star, lam0, lam1 = find_unique_stationary(L)
    print(f"   state space: {2 ** n_dirs} joint states {{F,P}}^{n_dirs}")
    print(f"   eigenvalue λ_0 (null space)        = {lam0:.3e}  (should be 0)")
    print(f"   eigenvalue λ_1 (slowest decay)     = {lam1:.4f}  (should be < 0)")
    assert abs(lam0) < 1e-12, f"L not in null at 0: {lam0}"
    assert lam1 < -1e-3, f"Spectral gap missing: {lam1}"
    print(f"   spectral gap |λ_1 − λ_0| > 0  →  unique stationary by Perron-Frobenius.")

    # per-direction marginals
    pi_P_d = per_direction_marginals(rho_star, n_dirs)
    rates_d = per_direction_event_rate(rho_star, n_dirs)
    print()
    print(f"   per-direction P(state=P) at stationary:")
    for d in range(n_dirs):
        print(f"      d={d}:  π_P(d) = {pi_P_d[d]:.6f}  (analytic π_P = 0.6)")
    print(f"   per-direction event rate at stationary:")
    for d in range(n_dirs):
        print(f"      d={d}:  r(d)   = {rates_d[d]:.6f}  (analytic r_∞ = 0.4)")
    err_pi = float(np.max(np.abs(pi_P_d - 0.6)))
    err_r = float(np.max(np.abs(rates_d - 0.4)))
    print(f"   max |π_P(d) − 0.6| = {err_pi:.3e}  (machine precision check)")
    print(f"   max |r(d) − 2/5|   = {err_r:.3e}  (machine precision check)")
    assert err_pi < 1e-12, f"π_P(d) deviates from 0.6: {err_pi}"
    assert err_r < 1e-12, f"r(d) deviates from 2/5: {err_r}"
    print(f"   per-direction marginal IS direction-uniform at stationary.  CONFIRMED.")
    print()

    # ---- Symmetry argument ----
    print(" Symmetry argument (closes the structural verdict):")
    print(f"   Local generator L_local on {{F,P}}^n_dirs commutes with the S_n")
    print(f"   group of direction permutations (uniform sampling weight 1/n_dirs at")
    print(f"   every step; F↔P transition rates depend only on local state, not on")
    print(f"   direction label).")
    print(f"   By Perron-Frobenius + S_n invariance, the unique stationary ρ_∞ is")
    print(f"   S_n-invariant. Therefore each direction's marginal Beta state has")
    print(f"   identical (π_F, π_P) = (2/5, 3/5), and per-direction event rate at")
    print(f"   stationary is direction-independent at exactly r_∞ = 2/5.")
    print()

    # ---- Verdict ----
    print("=" * 76)
    print(" VERDICT — B-route 2 (Bloch zero-mode) does NOT save Claim B")
    print("=" * 76)
    print()
    print(" Bloch decomposition isolates spatial-mode preservation. λ(k = 0) = 0")
    print(" is real and ensures the SPATIAL k = 0 mode survives mixing. But the")
    print(" k = 0 mode's INTERNAL state is governed by L_local, whose unique")
    print(" stationary is direction-uniform under the framework's standard")
    print(" direction-uniform renewal Markov dynamics.")
    print()
    print(" The IC's directional anisotropy is NOT a Bloch zero-mode quantity.")
    print(" It lives in the per-vertex internal-state space, which mixes to a")
    print(" direction-uniform stationary regardless of spatial-mode structure.")
    print()
    print(" For Claim B persistence to hold, the local dynamics needs structural")
    print(" input that BREAKS S_n direction-permutation symmetry. The two")
    print(" candidates are:")
    print()
    print("   (a) direction-anisotropic local sampling (Model 2 in the persistence")
    print("       audit) — source of anisotropy NOT in framework's current axioms;")
    print()
    print("   (b) NESS with degenerate stationary manifold (Model 3) — the local")
    print("       renewal Markov has UNIQUE stationary by detailed balance, NOT")
    print("       a degenerate manifold.")
    print()
    print(" Combined with the Model 1 negative (cascade_step5_claim_B_persistence_audit.py),")
    print(" two independent rescue routes for Claim B are now structurally closed.")
    print()
    print(" Cascade Step 5 amplitude α = ε_toggle: STRUCTURALLY OPEN under standard")
    print(" renewal Markov dynamics. Empirical match α = 0.207 ± 0.036 (+0.18σ)")
    print(" remains unexplained at the structural level.")
    print()
    print(" Pre-emptive structural audit prevented a multi-session dead-end")
    print(" implementation of B-route 2. Honest audit-before-ansatz finding.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
