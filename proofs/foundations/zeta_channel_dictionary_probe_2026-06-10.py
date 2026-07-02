#!/usr/bin/env python3
"""Phase 1.2 — zeta-functional re-expression of the counting/winding channels.

Built on the Phase 1.1 infrastructure (zeta_factorization_srs_srsz_2026-06-10.py).
The crystal Ihara zeta has  log zeta(u) = sum_L N_L u^L / L  where

    N_L = INT_BZ Tr[B(k)^L] d^3k     (closed NB walks of net-zero translation,
                                      per primitive cell, length L)

The integrand is a trigonometric polynomial in k (bounded cell displacements),
so a uniform grid above Nyquist computes the integral EXACTLY (up to fp).

Gates:
  Z1  Girth from the zeta: N_L = 0 for L < 10 (g = 10 read off log zeta).
  Z2  N_10 = 120: 12 oriented girth cycles/cell x 10 basepoint edges
      == "10 oriented girth cycles per directed bond" (120/12).
  Z3  V_us = k*^3 / N_10 = 27/120 = 9/40  -- the Level-2 counting density
      re-expressed as a ZETA COEFFICIENT functional (zero new constants).
  Z4  V_cb = zeta_C(8)(u) - 1 = u^8/(1-u^8) at u = 2/3  == 256/6305 exactly
      (single cycle-class zeta, period L_eff = 8, evaluated by analytic
      continuation past the crystal-zeta convergence radius 1/2).
  Z5  V_ub = sum_{m>=2} [zeta_C(6m+2)(u) - 1] at u = 2/3  == repo value
      (Lambert-type sum over the 6m+2 cycle-length family).

Honest channel notes (not gates): the effective lengths (8 = g-2; 6m+2) and
the evaluation point u = 2/3 = (k*-1)/k* are CHANNEL DATA the zeta organization
does not yet force; they are exactly the freedom the self-MDL ledger prices
under "A5(b) channel levels" and that Phase 1.3+ aims to force.
"""
import os
import sys
from fractions import Fraction

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, K_STAR  # noqa: E402

TOL = 1e-9
FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def directed_edges(bonds):
    edges = [(i, j, tuple(cell)) for (i, j, cell) in bonds]
    rev = {}
    for a, (i, j, c) in enumerate(edges):
        target = (j, i, tuple(-x for x in c))
        for b, e2 in enumerate(edges):
            if e2 == target:
                rev[a] = b
    return edges, rev


def bloch_B(k, edges, rev):
    n = len(edges)
    B = np.zeros((n, n), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        for b, (i2, j2, c2) in enumerate(edges):
            if i2 == j and b != rev[a]:
                B[b, a] = np.exp(2j * np.pi * np.dot(k, c2))
    return B


def cycle_counts(edges, rev, L_max, n_grid):
    """N_L = INT_BZ Tr B(k)^L dk via uniform grid (exact for trig polys)."""
    N = np.zeros(L_max + 1)
    pts = (np.arange(n_grid) + 0.5) / n_grid
    for k1 in pts:
        for k2 in pts:
            for k3 in pts:
                B = bloch_B(np.array([k1, k2, k3]), edges, rev)
                P = np.eye(len(edges), dtype=complex)
                for L in range(1, L_max + 1):
                    P = P @ B
                    N[L] += np.trace(P).real
    return N / n_grid**3


def main():
    print("=" * 72)
    print(" PHASE 1.2 -- zeta-functional channels: counting + winding sectors")
    print("=" * 72)
    bonds = find_bonds()
    edges, rev = directed_edges(bonds)

    # Cell displacements are small; degree per axis over 12 steps stays well
    # below Nyquist for n_grid = 31 (verified by grid-doubling check below).
    L_max = 12
    N = cycle_counts(edges, rev, L_max, n_grid=31)
    N_check = cycle_counts(edges, rev, L_max, n_grid=37)
    grid_ok = np.allclose(N, N_check, atol=1e-6)
    gate("Z0 BZ quadrature converged (31^3 vs 37^3 grids)", grid_ok,
         f"max diff {np.abs(N - N_check).max():.2e}")

    print("\n  Closed-NB-cycle counts per primitive cell (zeta coefficients):")
    for L in range(1, L_max + 1):
        print(f"    N_{L:<2} = {N[L]:+.6f}")

    gate("Z1 girth from zeta: N_L = 0 for L < 10",
         np.allclose(N[1:10], 0.0, atol=1e-6), f"max |N_<10| {np.abs(N[1:10]).max():.2e}")

    n10 = N[10]
    gate("Z2 N_10 = 120 (= 10 oriented girth cycles per directed bond x 12)",
         abs(n10 - 120) < 1e-6, f"N_10 = {n10:.8f}")

    # Z3: V_us as a zeta-coefficient functional
    V_us_zeta = K_STAR**3 / n10
    gate("Z3 V_us = k*^3 / N_10 = 9/40 (counting density == zeta coefficient)",
         abs(V_us_zeta - 9 / 40) < 1e-12, f"k*^3/N_10 = {V_us_zeta:.12f}")

    # Z4: V_cb as a cycle-class zeta special value
    u = Fraction(2, 3)
    V_cb = u**8 / (1 - u**8)
    gate("Z4 V_cb = zeta_C8(2/3) - 1 = 256/6305 exactly",
         V_cb == Fraction(256, 6305), f"= {V_cb}")

    # Z5: V_ub as the Lambert-type sum over the 6m+2 family
    uf = 2.0 / 3.0
    # repo convention truncates at m=10 (predictions/V_ub.py); full-series tail
    # beyond that is ~(2/3)^68 ~ 1e-12 (sub-precision, noted not gated)
    V_ub = sum(uf ** (6 * m + 2) / (1 - uf ** (6 * m + 2)) for m in range(2, 11))
    gate("Z5 V_ub = sum_{m=2..10} [zeta_C(6m+2)(2/3) - 1] = repo 3.767023820519e-3",
         abs(V_ub - 3.767023820519e-3) < 1e-15, f"= {V_ub:.12e}")

    print("\n  Channel notes (the residual freedom, priced in the self-MDL ledger):")
    print("   - effective winding period 8 = g - 2 (V_cb) and family 6m+2 (V_ub)")
    print("     are channel data; the zeta organization does not yet FORCE them.")
    print("   - evaluation point u = 2/3 = (k*-1)/k* sits outside the crystal-zeta")
    print("     convergence radius 1/2; cycle-class factors are evaluated by")
    print("     analytic continuation (rational functions).")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- counting + winding channels are zeta functionals")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
