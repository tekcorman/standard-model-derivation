#!/usr/bin/env python3
"""
Canonical prediction file for ε_CP_baryon (baryon-CP asymmetry per process).

The framework's per-process CP asymmetry ε_CP equals the Bayesian-toggle
posterior asymmetry ε_toggle directly, with no geometric or process factor
between them:

    ε_CP = ε_toggle = (P_fresh − P_persist) / (P_fresh + P_persist) = 1/5

where P_fresh = 1/2 (Beta(1,1) MaxEnt prior, predictions/S_fresh.py) and
P_persist = 1/3 (Beta(2,1) posterior after one confirmation,
predictions/S_disconfirm.py).

This is the SHARED structural source for three independent observable
channels (Row P27 A_hemispherical, this row P28, Rows P19/P20 cascade
D2-extended), each composing ε_toggle with a different geometric factor;
ε_CP is the direct application with NO geometric factor.

Until 2026-05-15 EOD+1 this value was only embedded inside the η_B chain
(predictions/eta_B.py uses eps_CP = Fraction(1,5) inline) and the
cross-prediction script proofs/foundations/epsilon_toggle_substrate_derivation.py.
This file surfaces it in the predictions/ DAG as its own row (P28) per the
parameter linter's "expose theorem-grade structural predictions" discipline.
"""

# ============================================================
# PARAMETER: ε_CP_baryon (baryon CP asymmetry per process)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       not directly observed; feeds η_B at 6.12 × 10⁻¹⁰ (Planck 2018).
# Source:      Per-process ε_CP is not directly measured; it enters the
#              Sakharov-Hashimoto baryogenesis chain together with substrate
#              tree-amplitude factors to predict η_B (= 6.11×10⁻¹⁰, Row P29,
#              theorem-grade −0.20σ vs Planck).
# PDG edition: 2024 (η_B value from Planck 2018 / PDG cosmology table).
#
# Note: ε_CP cannot be measured cleanly in isolation from the chain; only
# the product ε_CP · Re(h_P) · α₁^M emerges as the observable η_B.  This
# row predicts the per-process factor structurally.

# --- PREDICTED VALUE -----------------------------------------
# Value:       ε_CP = 1/5 (exact)
# Status:      UNIQUE — THEOREM-GRADE under A1 + A2-T + A3-T + Bayesian
#              Beta(2,1) update + Jaynes MaxEnt; conditional inherited from
#              S_fresh + S_disconfirm + Row 4 (Bayesian setup applies under
#              k* = 3 substrate).

# --- DERIVED FORMULA -----------------------------------------
# ε_CP = (P_fresh − P_persist) / (P_fresh + P_persist)
#
# where:
#   P_fresh   = 1/2   (Bayesian conjugate, Beta(1,1) MaxEnt prior:
#                     P(toggle present | no prior data) = 1/2; equivalently
#                     S_fresh = -log₂(P_fresh) = 1 bit, predictions/S_fresh.py)
#   P_persist = 1/3   (Beta(2,1) posterior after one confirmation:
#                     P(toggle absent | one prior confirmation) = 1/3;
#                     equivalently S_disconfirm = -log₂(P_persist) = log₂(3),
#                     predictions/S_disconfirm.py)
#
# Logical chain:
#   Step 1: P_fresh = 1/2 from Beta(1,1) Jaynes MaxEnt [Type 3 + Type 1,
#                     S_fresh.py theorem-grade]
#   Step 2: P_persist = 1/3 from Beta(2,1) Bayesian conjugate update
#                     [Type 3 + Type 1, S_disconfirm.py theorem-grade]
#   Step 3: ε_CP = (P_fresh − P_persist) / (P_fresh + P_persist)
#                = (1/2 − 1/3) / (1/2 + 1/3) = (1/6) / (5/6) = 1/5
#                [Type 2 algebra]
#
# Cross-check via the (k − 2)/(k + 2) class formula at k = k* = 3:
#   ε_CP = (3 − 2)/(3 + 2) = 1/5  ✓
#   (See Row P28 *Class A audit note* in parameter ledger — this formula
#   gives 1/5 at k=3 specifically; at qtz's k=4 it would be 1/3.)

# --- INPUTS --------------------------------------------------
# symbol     | value | status     | predictions/ file           | meaning
# -----------|-------|------------|-----------------------------|--------
# P_fresh    | 1/2   | [derived]  | predictions/S_fresh.py      | Beta(1,1) MaxEnt prior
# P_persist  | 1/3   | [derived]  | predictions/S_disconfirm.py | Beta(2,1) Bayesian posterior
# k*         | 3     | [derived]  | predictions/k_star.py       | coordination (for cross-check (k-2)/(k+2))

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import functools
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from S_fresh import predict_S_fresh
from S_disconfirm import predict_S_disconfirm
from k_star import predict_k_star
from d_spatial import predict_d_spatial

# --- substrate primitives ---
d_val = predict_d_spatial()
k = predict_k_star(d_val)

# Step 1: P_fresh from Beta(1,1) prior
# S_fresh = -log₂(P_fresh) so P_fresh = 2^(-S_fresh).  But S_fresh.py computes
# both — we recover P_fresh as the exact rational 1/2 by construction (Beta(1,1)
# alpha=1, beta=1).
S_fresh_pred = predict_S_fresh(1.0, 1.0)         # = 1 bit
P_fresh = Fraction(1, 2)
assert abs(S_fresh_pred - 1.0) < 1e-12, f"S_fresh should be 1 bit, got {S_fresh_pred}"

# Step 2: P_persist from Beta(2,1) posterior
import math
S_disconfirm_pred = predict_S_disconfirm(2.0, 1.0)  # Beta(2,1) posterior; = log₂(3) ≈ 1.585 bits
P_persist = Fraction(1, 3)
assert abs(S_disconfirm_pred - math.log2(3)) < 1e-12, f"S_disconfirm should be log₂(3), got {S_disconfirm_pred}"

# Step 3: ε_CP = (P_fresh − P_persist) / (P_fresh + P_persist) at exact rational
epsilon_CP_exact = (P_fresh - P_persist) / (P_fresh + P_persist)
assert epsilon_CP_exact == Fraction(1, 5), f"ε_CP should be 1/5, got {epsilon_CP_exact}"

# Float for downstream
epsilon_CP_pred = float(epsilon_CP_exact)         # = 0.2

# Cross-check via (k − 2)/(k + 2) at k = k* = 3
epsilon_CP_class_A = Fraction(k - 2, k + 2)
assert epsilon_CP_class_A == epsilon_CP_exact, (
    f"Class-A cross-check failed: (k−2)/(k+2) = {epsilon_CP_class_A} vs Bayesian = {epsilon_CP_exact}"
)

print("=" * 72)
print(" ε_CP_baryon  --  per-process baryon CP asymmetry (= ε_toggle)")
print("=" * 72)
print(f"  k*           = {k}  [Row 4 theorem-grade]")
print(f"  P_fresh      = {P_fresh} = 1/2  [Beta(1,1) MaxEnt; S_fresh = 1 bit]")
print(f"  P_persist    = {P_persist} = 1/3  [Beta(2,1) posterior; S_disconfirm = log₂(3) bits]")
print()
print(f"  ε_CP = (P_fresh − P_persist) / (P_fresh + P_persist)")
print(f"       = ({P_fresh} − {P_persist}) / ({P_fresh} + {P_persist})")
print(f"       = ({P_fresh - P_persist}) / ({P_fresh + P_persist})")
print(f"       = {epsilon_CP_exact}")
print()
print(f"  Cross-check via Class A (k−2)/(k+2) at k=3: {epsilon_CP_class_A} ✓")
print()
print("STATUS: UNIQUE — THEOREM-GRADE.")
print("  Inherits S_fresh + S_disconfirm theorem-grade (Bayesian conjugate updates).")
print("  Conditional on Row 4 (k*=3 selection structurally closes the (k−2)/(k+2)")
print("  cross-check; at qtz k=4 the Class-A formula gives 1/3).")
print("  ε_CP feeds η_B (Row P29, theorem-grade −0.20σ vs Planck) via")
print("    η_B = ε_CP · Re(h_P) · α₁^M = (1/5)·(√3/2)·(2/3)⁴⁸ ≈ 6.11×10⁻¹⁰.")
print()
print("Cross-references: Row P28 ledger; Row P27 A_hemispherical (ε_CP · 1/k* = 1/15);")
print("                  Row P19 H_0 (cascade D2-extended uses 1 + ε_CP/k* = 16/15).")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_epsilon_CP(P_fresh_in, P_persist_in):
    """
    Predict the per-process baryon CP asymmetry ε_CP from the Bayesian-toggle
    posterior asymmetry.

    ε_CP = (P_fresh − P_persist) / (P_fresh + P_persist)

    For framework inputs P_fresh = 1/2 (Beta(1,1) MaxEnt prior) and
    P_persist = 1/3 (Beta(2,1) Bayesian posterior after one confirmation),
    ε_CP = 1/5 exactly.

    Parameters
    ----------
    P_fresh_in : float
        Prior toggle probability under Beta(1,1) MaxEnt (= 1/2 in framework).
    P_persist_in : float
        Posterior toggle-absent probability under Beta(2,1) (= 1/3 in framework).

    Returns
    -------
    float
        ε_CP = (P_fresh − P_persist) / (P_fresh + P_persist).
    """
    return (P_fresh_in - P_persist_in) / (P_fresh_in + P_persist_in)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = epsilon_CP_pred
    pure_result = predict_epsilon_CP(float(P_fresh), float(P_persist))
    print()
    print(f"Implementation: ε_CP = {impl_result}")
    print(f"Pure function:  ε_CP = {pure_result}")
    assert abs(impl_result - pure_result) < 1e-12, (
        f"Mismatch: {impl_result} vs {pure_result}"
    )
    print(f"OK: outputs agree.  ε_CP = {pure_result} = 1/5 exactly.")
    print(f"    Class A cross-check (k−2)/(k+2) at k=3: matches Bayesian derivation.")
