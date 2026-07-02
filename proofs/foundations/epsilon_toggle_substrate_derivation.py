#!/usr/bin/env python3
"""
ε_toggle = 1/5 — substrate-primitives-only derivation

Derives the Bayesian-toggle posterior asymmetry

    ε_toggle = (P_fresh − P_persist) / (P_fresh + P_persist) = 1/5

from substrate primitives only (A1 + A2-T + Jaynes 1957 MaxEnt + Bayesian
conjugate update), with no geometric or observable-channel factor mixed
in. Composes the two theorem-grade upstream pieces:

  - P_fresh   = 1/2     (predictions/S_fresh.py;       Beta(1,1) MaxEnt prior)
  - P_persist = 1/3     (predictions/S_disconfirm.py;  Beta(2,1) posterior after
                         one confirmation)

ε_toggle is the unique scalar invariant of the (P_fresh, P_persist) pair under
linear normalization to [-1, 1]. Equivalently, in posterior-ratio form,

    p_creation = P_fresh / (P_fresh + P_persist) = 3/5
    ε_toggle   = 2 · p_creation − 1               = 1/5

Both forms agree exactly. The derivation is Type-1 + Type-2 algebra on top of
S_fresh.py + S_disconfirm.py; no observable channel and no srs geometric
factor enters this file.

Why isolate this in its own probe
---------------------------------
ε_toggle = 1/5 is the SHARED structural source for three independent observable
channels, each composing it with a different geometric / process factor:

  Row P27 (A_hemispherical):          A = ε_toggle · ⟨(ê·ẑ)²⟩
                                        = (1/5)(1/3) = 1/15
                                      (geometric factor: srs cubic moment, 1/k*)

  Row P28 (ε_CP baryon-CP):           |ε_CP| = ε_toggle = 1/5
                                      (no geometric factor; per-process directly)

  Rows P19/P20 (cascade D2-extended): H_obs/H_sub = 1 + ε_toggle / k*
                                                 = 1 + (1/5)(1/3) = 16/15
                                      (geometric factor: same chiral-cubic 1/k*)

The shared origin of these three observables is a non-trivial cross-prediction
connection: agreement of all three with experiment is partial confirmation of
the Bayesian-toggle setup. Centralizing ε_toggle = 1/5 in a single probe makes
this shared origin auditable and prevents drift if the upstream Beta-posterior
arithmetic is ever revised.

Open structural conditional (cascade D2-ext use only)
-----------------------------------------------------
For Rows P19/P20, the inheritance coefficient c in α = c · ε_toggle (where α
is the substrate stationary-anisotropy amplitude) is empirically anchored at
c = 1 to within ~1σ jointly across H_0 + A_dilution but not yet structurally
derived. See proofs/cosmology/cascade_step5_amplitude_via_A_dilution.py and
an internal working note (Route 4
compression integral). This conditional applies ONLY to the cascade D2-ext
observables; A_hemis and ε_CP do not inherit it because they apply ε_toggle
directly without a stationary-anisotropy amplitude step.

Gate grade: THEOREM (Type 1 + Type 2 + upstream theorem-grade S_fresh.py +
S_disconfirm.py).
"""

import os
import sys
from fractions import Fraction

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def derive_epsilon_toggle():
    """
    Compose ε_toggle = (P_fresh − P_persist) / (P_fresh + P_persist)
    from the theorem-grade upstreams S_fresh and S_disconfirm.

    Returns
    -------
    epsilon : Fraction
        Exact value 1/5.
    p_fresh : Fraction
        Upstream value 1/2.
    p_persist : Fraction
        Upstream value 1/3.
    """
    # Upstream 1: P_fresh = 1/2 — Jaynes 1957 MaxEnt prior on the Bernoulli
    # parameter gives Beta(1, 1); predictive probability of either outcome
    # equals alpha/(alpha+beta) = 1/2.
    alpha_prior = Fraction(1)
    beta_prior = Fraction(1)
    p_fresh = alpha_prior / (alpha_prior + beta_prior)

    # Upstream 2: P_persist = 1/3 — after one observation of "exists" the
    # Bayesian conjugate update gives posterior Beta(2, 1); predictive
    # probability of disconfirmation (next observation = "absent") equals
    # beta/(alpha+beta) = 1/3 = 1/k*.
    alpha_post = Fraction(2)
    beta_post = Fraction(1)
    p_persist = beta_post / (alpha_post + beta_post)

    # Compose: linear-normalization asymmetry
    epsilon = (p_fresh - p_persist) / (p_fresh + p_persist)

    return epsilon, p_fresh, p_persist


def main():
    print("=" * 76)
    print(" ε_toggle = 1/5 — substrate-primitives-only derivation")
    print("=" * 76)
    print()

    epsilon, p_fresh, p_persist = derive_epsilon_toggle()

    print(" Upstream 1 (predictions/S_fresh.py — Jaynes MaxEnt Beta(1,1)):")
    print(f"   P_fresh     = α/(α+β) = 1/2 = {p_fresh}")
    print()
    print(" Upstream 2 (predictions/S_disconfirm.py — Bayesian conjugate Beta(2,1)):")
    print(f"   P_persist   = β/(α+β) = 1/3 = {p_persist}")
    print()
    print(" Composition (linear-normalization asymmetry):")
    print(f"   ε_toggle    = (P_fresh − P_persist) / (P_fresh + P_persist)")
    print(f"               = ({p_fresh} − {p_persist}) / ({p_fresh} + {p_persist})")
    print(f"               = {p_fresh - p_persist} / {p_fresh + p_persist}")
    print(f"               = {epsilon}")
    print()

    # Equivalent form via posterior ratio
    p_creation = p_fresh / (p_fresh + p_persist)
    epsilon_alt = 2 * p_creation - Fraction(1)
    print(" Equivalent form (posterior-ratio):")
    print(f"   p_creation  = P_fresh / (P_fresh + P_persist) = {p_creation}")
    print(f"   ε_toggle    = 2 · p_creation − 1              = {epsilon_alt}")
    print()

    # Hard checks
    assert epsilon == Fraction(1, 5), f"Expected 1/5, got {epsilon}"
    assert epsilon_alt == Fraction(1, 5), f"Equivalent form mismatch: {epsilon_alt}"
    assert epsilon == epsilon_alt, "Two derivation forms disagree"

    # Sympy exact cross-check
    import sympy as sp
    a_pr, b_pr, a_po, b_po = sp.symbols("a_pr b_pr a_po b_po", positive=True)
    pf_sym = a_pr / (a_pr + b_pr)
    pp_sym = b_po / (a_po + b_po)
    eps_sym = (pf_sym - pp_sym) / (pf_sym + pp_sym)
    eps_val = sp.nsimplify(eps_sym.subs({a_pr: 1, b_pr: 1, a_po: 2, b_po: 1}))
    assert eps_val == sp.Rational(1, 5), f"Sympy mismatch: {eps_val}"
    print(f" Sympy exact check: ε_toggle = {eps_val}  OK")
    print()

    # Float cross-check
    eps_float = float(epsilon)
    assert abs(eps_float - 0.2) < 1e-15
    print(f" Float cross-check: ε_toggle = {eps_float:.15f}  (expected 0.200000000000000)")
    print()

    # Cross-import consistency check against S_fresh.py + S_disconfirm.py
    print(" Cross-import consistency with predictions/S_fresh.py and S_disconfirm.py:")
    from predictions.S_fresh import predict_S_fresh
    from predictions.S_disconfirm import predict_S_disconfirm
    s_fresh_bits = predict_S_fresh(1.0, 1.0)
    s_disc_bits = predict_S_disconfirm(2.0, 1.0)
    p_fresh_via_S = 2.0 ** (-s_fresh_bits)
    p_persist_via_S = 2.0 ** (-s_disc_bits)
    eps_via_S = (p_fresh_via_S - p_persist_via_S) / (p_fresh_via_S + p_persist_via_S)
    assert abs(p_fresh_via_S - 1/2) < 1e-15
    assert abs(p_persist_via_S - 1/3) < 1e-15
    assert abs(eps_via_S - 0.2) < 1e-14
    print(f"   2^(-S_fresh)        = {p_fresh_via_S:.15f}  (matches 1/2)")
    print(f"   2^(-S_disconfirm)   = {p_persist_via_S:.15f}  (matches 1/3)")
    print(f"   ε_toggle (via S)    = {eps_via_S:.15f}  (matches 1/5)")
    print()

    print("=" * 76)
    print(" Downstream channels using this ε_toggle")
    print("=" * 76)
    print()
    print(" A_hemis  (Row P27): A = ε · ⟨(ê·ẑ)²⟩ = (1/5)(1/3) = 1/15")
    print("                    geometric factor = srs cubic moment 1/k*")
    print(" ε_CP     (Row P28): |ε_CP| = ε = 1/5")
    print("                    (no geometric factor; per-process directly)")
    print(" cascade  (P19/P20): H_obs/H_sub = 1 + ε/k* = 1 + 1/15 = 16/15")
    print("                    conditional on c=1 inheritance for substrate")
    print("                    stationary-anisotropy amplitude (Route 4 open)")
    print()
    print(" The shared substrate-primitives origin (this probe) makes the three")
    print(" channels structurally co-determined: any revision to S_fresh or")
    print(" S_disconfirm propagates uniformly to A_hemis, ε_CP, and the cascade")
    print(" D2-ext rate-gap.")
    print()
    print(" OK: ε_toggle = 1/5 derived from substrate primitives only.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
