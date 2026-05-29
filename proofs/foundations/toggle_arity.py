#!/usr/bin/env python3
"""
Toggle arity independence test.

Does k* = 3 regardless of whether the toggle is binary, ternary, or p-ary?
If yes: arity is irrelevant, the framework is robust.
If no: binary must be uniquely selected, or the framework is fragile.

A p-ary toggle: each edge has p states {0, 1, ..., p-1}, cycling or random.
The observer uses a Dirichlet(1,...,1) prior (maximum ignorance over p states).

Thresholds:
  theta_create: surprise of first confirmation of a new edge state.
    Under Dirichlet(1,...,1) with p categories, P(state_i) = 1/p.
    Surprise = -log2(1/p) = log2(p).

  theta_persist: surprise of first disconfirmation of a confirmed state.
    After one confirmation: Dirichlet with alpha_i = 2 for confirmed state,
    alpha_j = 1 for others. Total alpha = p + 1.
    P(different state) = (p-1)/(p+1).
    Surprise = -log2((p-1)/(p+1)) = log2((p+1)/(p-1)).

Equilibrium: S(k) = theta_create + theta_persist
where S(k) is the contextual surprise of a toggle at degree k.

For a p-ary toggle, S(k) = log2(p) + log2(k) [specifying which edge + which state]
Wait: this needs more care.

Actually: a toggle event at a degree-k node in a p-ary system:
  - Specifying which edge was toggled: log2(k) bits
  - Specifying the new state: log2(p-1) bits (one of p-1 other states)
  - Binary surprise of "something happened": 1 bit? No — this is already counted.

Let me think about this from first principles.

The observer sees: "edge e at node v changed from state s to state s'."
Information content:
  1. Which edge: log2(k) bits (k edges at the node)
  2. Which new state: log2(p-1) bits (p-1 alternatives to current state)
  Total contextual surprise: S(k,p) = log2(k) + log2(p-1)

But for binary (p=2): S(k,2) = log2(k) + log2(1) = log2(k) + 0 = log2(k).
The paper has S(k) = 1 + log2(k). Where does the 1 come from?

The 1 bit is the surprise of the toggle event itself (something changed vs nothing).
So: S(k,p) = 1 + log2(k) + log2(p-1)

For p=2: S(k,2) = 1 + log2(k) + 0 = 1 + log2(k). ✓ Matches paper.

Equilibrium: S(k*,p) = theta_create(p) + theta_persist(p)

theta_create(p) = log2(p)
theta_persist(p) = log2((p+1)/(p-1))

So: 1 + log2(k*) + log2(p-1) = log2(p) + log2((p+1)/(p-1))

Solve for k*:
log2(k*) = log2(p) + log2((p+1)/(p-1)) - 1 - log2(p-1)
         = log2(p) + log2(p+1) - log2(p-1) - 1 - log2(p-1)
         = log2(p) + log2(p+1) - 2*log2(p-1) - 1
         = log2(p(p+1)/(p-1)^2) - 1
         = log2(p(p+1)/(2(p-1)^2))

k* = p(p+1) / (2(p-1)^2)

Check p=2: k* = 2*3 / (2*1) = 6/2 = 3. ✓

General formula: k*(p) = p(p+1) / (2(p-1)^2)
"""

import math
import numpy as np

def compute_k_star(p):
    """Compute equilibrium degree k* for a p-ary toggle."""
    if p < 2:
        return float('inf')

    # Thresholds
    theta_create = math.log2(p)
    theta_persist = math.log2((p + 1) / (p - 1))

    # Contextual surprise at degree k: S(k,p) = 1 + log2(k) + log2(p-1)
    # Equilibrium: S(k*,p) = theta_create + theta_persist
    # 1 + log2(k*) + log2(p-1) = log2(p) + log2((p+1)/(p-1))

    log2_k = math.log2(p) + math.log2((p + 1) / (p - 1)) - 1 - math.log2(p - 1)
    k_star = 2 ** log2_k

    # Closed form: k* = p(p+1) / (2(p-1)^2)
    k_star_formula = p * (p + 1) / (2 * (p - 1) ** 2)

    return k_star, k_star_formula, theta_create, theta_persist


def main():
    print("Toggle Arity Independence Test")
    print("=" * 65)
    print()
    print("Formula: k*(p) = p(p+1) / (2(p-1)²)")
    print()
    print(f"  {'p':>3}  {'k* (exact)':>12}  {'k* (rounded)':>12}  {'θ_create':>10}  {'θ_persist':>10}")
    print("  " + "-" * 57)

    for p in range(2, 20):
        k, k_f, tc, tp = compute_k_star(p)
        print(f"  {p:>3}  {k:>12.6f}  {round(k):>12d}  {tc:>10.4f}  {tp:>10.4f}")

    print()

    # Key results
    print("KEY RESULTS:")
    print()
    k2, _, _, _ = compute_k_star(2)
    k3, _, _, _ = compute_k_star(3)
    k4, _, _, _ = compute_k_star(4)
    print(f"  p=2 (binary):   k* = {k2:.6f} = 3 exactly")
    print(f"  p=3 (ternary):  k* = {k3:.6f} = 3/2 × 1 = 1.5")
    print(f"  p=4 (quaternary): k* = {k4:.6f}")
    print()

    # Limit
    print(f"  p→∞: k*(p) → p(p+1)/(2p²) → 1/2")
    print()

    # Analysis
    print("ANALYSIS:")
    print()
    print("  k* = 3 ONLY for p = 2 (binary toggle).")
    print()
    print("  For p = 3: k* = 1.5, which rounds to 2.")
    print("  → Cl(4) Fock space: 2² = 4 states")
    print("  → Weyl group S₂ → A₁ → su(2)")
    print("  → Gauge group: SU(2) × U(1)")
    print("  → No color! No quarks! Not our universe.")
    print()
    print("  For p ≥ 3: k* < 2, rounding to 1 or 2.")
    print("  → No trivalent structure, no SU(3), no Standard Model.")
    print()
    print("  VERDICT: Binary toggle is UNIQUELY SELECTED.")
    print("  k* = 3 requires p = 2. The toggle arity is not free.")
    print("  Binary is the only arity that produces three edges,")
    print("  three colors, three generations, and the SM gauge group.")
    print()

    # What about the creation threshold model?
    # Alternative: theta_create = 1 bit always (something changed)
    # independent of p
    print("=" * 65)
    print("ALTERNATIVE: θ_create = 1 bit (arity-independent)")
    print("=" * 65)
    print()
    print("  If θ_create = 1 bit regardless of p (just 'something changed'),")
    print("  and θ_persist = log₂(p+1) - log₂(p-1):")
    print()
    print(f"  {'p':>3}  {'k* (formula)':>14}  {'k* (rounded)':>12}")
    print("  " + "-" * 35)

    for p in range(2, 12):
        tp = math.log2((p + 1) / (p - 1))
        # S(k,p) = 1 + log2(k) + log2(p-1) = 1 + tp
        # log2(k*) = tp - log2(p-1)
        log2_k = tp - math.log2(p - 1)
        k = 2 ** log2_k
        print(f"  {p:>3}  {k:>14.6f}  {round(k):>12d}")

    print()
    print("  Under this model, k* = 3 still only works for p = 2.")
    print()

    # The REAL alternative: what if theta_create depends on arity differently?
    # The paper's argument: theta_create = -log2(P(ON|Beta(1,1))) = -log2(1/2) = 1
    # For p-ary: P(state_i | Dirichlet(1,...,1)) = 1/p
    # theta_create = -log2(1/p) = log2(p)
    # This IS the natural extension.
    print("=" * 65)
    print("ROBUSTNESS CHECK: Self-inverse constraint")
    print("=" * 65)
    print()
    print("  The toggle is SELF-INVERSE: toggle² = identity.")
    print("  For p-ary states: cycling s → s+1 mod p is self-inverse")
    print("  ONLY when p = 2 (since 2 mod 2 = 0).")
    print("  For p = 3: s → s+1 → s+2 ≠ s. NOT self-inverse!")
    print()
    print("  A p-ary toggle that IS self-inverse must be an involution.")
    print("  On p states, involutions pair states: p must be even.")
    print("  For p = 2: the unique involution is 0 ↔ 1. ✓")
    print("  For p = 4: involutions exist (0↔1, 2↔3) but k* = 10/9 ≈ 1.")
    print()
    print("  SELF-INVERSE + k*=3 ⟹ p = 2.")
    print("  Binary is the UNIQUE toggle that is self-inverse AND")
    print("  produces trivalent equilibrium.")


if __name__ == '__main__':
    main()
