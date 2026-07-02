#!/usr/bin/env python3
"""
Canonical prediction file for E_P (adjacency eigenvalue at the P-point).

Audit anchor: foundational. Conditional on Rows 4, 6 of
`docs/audits/registers/uniqueness_ledger.md` (k* = 3 + srs identification). Theorem-grade
spectral-decomposition result; underpins the Hashimoto eigenvalue h
(`predictions/h_walker_eigenvalue.py`) and the C₃-isotypic decomposition
on the 8-dim Ramanujan subspace at the P-point (per
`docs/theorems/theorem_bloch_lift_mu.md`).
"""

# ============================================================
# PARAMETER: E_P (adjacency matrix eigenvalue at P-point of BZ)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       sqrt(3) ≈ 1.7321 (exact algebraic)
# Source:      Mathematical property of srs Bloch Hamiltonian.
# PDG edition: N/A (structural/mathematical)

# --- PREDICTED VALUE -----------------------------------------
# Value:       sqrt(3) (exact)
# Deviation:   0

# --- DERIVED FORMULA -----------------------------------------
# E_P = sqrt(k*)
#
# Derivation chain:
#   1. k* = 3 (from predictions/k_star.py)
#   2. The srs lattice has 4 atoms per primitive cell (Wyckoff 8a
#      in I4_132, with the BCC conventional cell containing 8 atoms
#      and the primitive cell containing 4).
#   3. The Bloch Hamiltonian H(k) is 4×4. At the P-point
#      k_P = (π/2a)(1,1,1), the matrix H(k_P) has a special form
#      dictated by the I4_132 symmetry.
#   4. The characteristic polynomial of H(k_P) factors as
#      (λ² - k*)² = (λ² - 3)², giving eigenvalues ±sqrt(k*)
#      each with multiplicity 2.
#      (Computed in proofs/foundations/srs_E_at_P_derivation.py
#       by explicit matrix diagonalization.)
#   5. E_P = +sqrt(k*) = sqrt(3).
#
# The factorization (λ²-k*)² at the P-point is a consequence of
# the C₃ site symmetry: the stabilizer of P in the 432 point group
# is C₃, which forces the 4×4 matrix to decompose into 2×2 blocks,
# each with eigenvalues ±sqrt(k*).
# (Standard result in band theory of crystal nets — see Sunada 2012.)

# --- INPUTS --------------------------------------------------
# symbol | value | status    | predictions/ file      | meaning
# -------|-------|-----------|------------------------|--------
# k_star | 3     | [derived] | predictions/k_star.py  | coordination number

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k_star import predict_k_star
from d_spatial import predict_d_spatial
import functools

d = predict_d_spatial()
k = predict_k_star(d)

E_P = math.sqrt(k)

print(f"k* = {k}")
print(f"E_P = sqrt(k*) = sqrt({k}) = {E_P:.15f}")
print(f"  Eigenvalues of H(k_P): ±sqrt({k}), each with multiplicity 2")
print(f"  Char poly: (λ² - {k})² = 0")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_srs_E_at_P(k_star):
    """
    Computes the adjacency eigenvalue at the P-point of the srs BZ.

    The Bloch Hamiltonian H(k) of the srs lattice is 4×4 (4 atoms
    per primitive cell). At the P-point k_P = (π/2a)(1,1,1), the
    C₃ site symmetry forces the characteristic polynomial to factor
    as (λ² - k*)², giving eigenvalues ±sqrt(k*) with multiplicity 2.

    Parameters
    ----------
    k_star : int
        Coordination number (from predict_k_star).

    Returns
    -------
    float
        E_P = sqrt(k_star), the positive adjacency eigenvalue at P.
    """
    return math.sqrt(k_star)


# --- VALIDATION ----------------------------------------------

srs_E_at_P_pred = E_P


if __name__ == "__main__":
    impl_result = E_P
    pure_result = predict_srs_E_at_P(k)
    print(f"\nImplementation: {impl_result:.15f}")
    print(f"Pure function:  {pure_result:.15f}")
    print(f"sqrt(3):        {math.sqrt(3):.15f}")
    assert abs(impl_result - pure_result) < 1e-15, \
        f"Mismatch: {impl_result} vs {pure_result}"
    assert abs(pure_result - math.sqrt(3)) < 1e-15, \
        f"Expected sqrt(3), got {pure_result}"
    print("OK: outputs agree. E_P = sqrt(3) exactly.")
