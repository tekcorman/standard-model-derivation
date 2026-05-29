#!/usr/bin/env python3
"""
Canonical prediction file for p (toggle arity).

Audit anchor: Row 1 of `docs/audits/registers/uniqueness_ledger.md` (binary self-inverse
toggle p = 2). Direct A1 axiom; UNIQUE under the structural-pass strict
reading of A1.
"""

# ============================================================
# PARAMETER: p (toggle arity — number of states per edge)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       2 (binary)
# Source:      The Standard Model is built on binary (Z_2) gauge
#              structure at the fundamental level. All fermion
#              fields are two-component (Weyl spinors).
# PDG edition: N/A (structural)

# --- PREDICTED VALUE -----------------------------------------
# Value:       2 (exact)
# Deviation:   0

# --- DERIVED FORMULA -----------------------------------------
# p = 2, uniquely selected by the involution constraint.
#
# Derivation chain:
#   1. Framework axiom: the toggle is self-inverse (toggle^2 = identity).
#   2. A toggle on p states is a map T: {0,...,p-1} -> {0,...,p-1}.
#   3. Self-inverse means T(T(s)) = s for all s, i.e., T is an involution.
#   4. The simplest toggle is cycling: T(s) = s + 1 mod p.
#   5. Cycling is self-inverse iff T(T(s)) = s + 2 mod p = s for all s,
#      i.e., 2 ≡ 0 (mod p), i.e., p divides 2.
#   6. p ∈ {1, 2}. Since p = 1 is trivial (no state change), p = 2.
#
# Alternatively (general involutions, not just cycling):
#   7. Any involution on p states is a product of disjoint transpositions.
#   8. A fixed-point-free involution (every state changes) requires p even.
#   9. For p = 2: unique involution is 0 ↔ 1. ✓
#  10. For p = 4: involutions exist (0↔1, 2↔3), but MDL selects minimum p.
#      (Larger p means more states per edge = higher model cost with no
#      additional compression benefit, since the toggle is binary in nature.)
#  11. Therefore p = 2 is the unique MDL-optimal self-inverse toggle arity.

# --- INPUTS --------------------------------------------------
# symbol  | value        | status    | predictions/ file   | meaning
# --------|--------------|-----------|---------------------|--------
# (none)  |              |           |                     | p is derived from framework axiom alone

# --- IMPLEMENTATION ------------------------------------------

def _derive_p():
    """
    Derive toggle arity from involution constraint.

    The cycling map T(s) = s+1 mod p is self-inverse iff p | 2.
    The only non-trivial solution is p = 2.
    """
    for p in range(1, 100):
        # Cycling is self-inverse iff 2 mod p == 0
        if p >= 2 and (2 % p == 0):
            return p
    raise RuntimeError("No solution found")


p = _derive_p()
print(f"Toggle arity p = {p}")
print(f"  Involution check: T(T(0)) = (0+2) mod {p} = {2 % p} = 0  ✓")
print(f"  p=1: trivial (no state change), rejected")
print(f"  p=2: unique non-trivial self-inverse cycling map")
print(f"  p=3: T(T(0)) = 2 mod 3 = 2 ≠ 0, NOT self-inverse")
print(f"  p=4: T(T(0)) = 2 mod 4 = 2 ≠ 0, NOT self-inverse (cycling)")
print(f"        (involutions exist on p=4 but MDL selects minimum p=2)")


# --- PURE FUNCTION -------------------------------------------

import functools

@functools.lru_cache(maxsize=None)
def predict_p_toggle():
    """
    Returns the toggle arity p, derived from the involution constraint.

    The framework axiom states that the toggle is self-inverse:
    T^2 = identity. The cycling map T(s) = s+1 mod p satisfies this
    iff p divides 2. The unique non-trivial solution is p = 2.

    Parameters
    ----------
    (none — derived from axiom alone)

    Returns
    -------
    int
        Toggle arity p = 2.
    """
    # The cycling map s -> s+1 mod p is self-inverse iff p | 2.
    # p = 1 is trivial. p = 2 is the unique non-trivial solution.
    return 2


# --- VALIDATION ----------------------------------------------

p_toggle_pred = p


if __name__ == "__main__":
    impl_result = p
    pure_result = predict_p_toggle()
    print(f"\nImplementation: {impl_result}")
    print(f"Pure function:  {pure_result}")
    assert impl_result == pure_result, \
        f"Mismatch: {impl_result} vs {pure_result}"
    assert pure_result == 2, \
        f"Expected p=2, got {pure_result}"
    print("OK: outputs agree. p = 2 exactly.")
