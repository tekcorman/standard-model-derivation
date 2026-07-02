#!/usr/bin/env python3
"""Phase 1.3 — translation-resolved zeta sectors: parity theorem + mirror girth.

The crystal zeta refines over closing translations R (primitive coords):
N_L(R) = #closed NB walks of length L with net translation exactly R
       = INT_BZ Tr[B(k)^L] e^{-2pi i k.R} dk    (exact on a uniform grid).

Gates (all counts per primitive cell):
  P1 PARITY THEOREM: every nonzero N_L(R) has (-1)^L = (-1)^{sum R}.
     Walk-length parity == body-center-coset parity (combinatorial form of
     the F1 antiperiod / the bipartite character of the srs-z cover).
  P2 MIRROR GIRTH = 3 = k*: the through-the-mirror sector's minimal cycles
     are NB TRIANGLES: N_3(R) = 3 for each of the 8 odd nearest translations
     (the <111> body-diagonal directions -- the srs helix axes); nothing
     odd below L = 3.
  P3 SCREW SECTOR: N_4(R) = 4 for each of the 6 cubic-axis translations
     (the 4_1 screw pitches: (1,1,0)-type in primitive = (0,0,1)-type cubic),
     and N_8(same R) = 8 -- the L = 8 channel lives in the SCREW sector
     (second harmonic of the 4_1 helix), not at R = 0 and not in the mirror.
  P4 HOME SECTOR: N_10(0) = 120 (girth cycles; consistency with Phase 1.2).

Sector map (established here):
  R = 0 even bulk: girth 10 (120 cycles)         -> V_us / counting channel
  mirror (odd) sector: girth 3 = k*, <111> axes  -> C3/generation geometry
  screw (cubic-axis) sector: girth 4, harmonics 8 -> candidate home of the
     L_eff = 8 channel (V_cb winding series, lepton anchor) -- CANDIDATE,
     promotion requires forcing the harmonic selection (Phase 1.3+).
"""
import os
import sys
from itertools import product as iproduct

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds  # noqa: E402

TOL = 1e-6
FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 1.3 -- translation-resolved zeta sectors (parity + mirror girth)")
    print("=" * 72)
    bonds = find_bonds()
    edges = [(i, j, tuple(c)) for (i, j, c) in bonds]
    rev = {}
    for a, (i, j, c) in enumerate(edges):
        target = (j, i, tuple(-x for x in c))
        for b, e2 in enumerate(edges):
            if e2 == target:
                rev[a] = b

    L_max, n = 12, 27
    pts = (np.arange(n) + 0.5) / n
    traces = np.zeros((L_max + 1, n, n, n), dtype=complex)
    for i1, k1 in enumerate(pts):
        for i2, k2 in enumerate(pts):
            for i3, k3 in enumerate(pts):
                k = np.array([k1, k2, k3])
                B = np.zeros((12, 12), dtype=complex)
                for a, (i, j, c) in enumerate(edges):
                    for b, (i2e, j2, c2) in enumerate(edges):
                        if i2e == j and b != rev[a]:
                            B[b, a] = np.exp(2j * np.pi * np.dot(k, c2))
                P = np.eye(12, dtype=complex)
                for L in range(1, L_max + 1):
                    P = P @ B
                    traces[L, i1, i2, i3] = np.trace(P)

    K1, K2, K3 = np.meshgrid(pts, pts, pts, indexing="ij")

    def N(L, R):
        ph = np.exp(-2j * np.pi * (K1 * R[0] + K2 * R[1] + K3 * R[2]))
        return (traces[L] * ph).mean()

    # P1: parity theorem over the |n_i|<=2 window
    bad = []
    for L in range(1, L_max + 1):
        for R in iproduct(range(-2, 3), repeat=3):
            v = N(L, R)
            if abs(v) > TOL and (L % 2) != (sum(R) % 2):
                bad.append((L, R, v.real))
    gate("P1 parity theorem: (-1)^L = (-1)^{sum R} for every nonzero sector",
         not bad, f"violations: {bad[:3] if bad else 'none'}")

    # P2: mirror girth = 3 = k* on the 8 <111> odd nearest translations
    odd_nn = [(1, 1, 1), (-1, -1, -1), (1, 0, 0), (-1, 0, 0),
              (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    vals3 = [N(3, R).real for R in odd_nn]
    below = max(abs(N(L, R)) for L in (1, 2) for R in iproduct(range(-2, 3), repeat=3))
    gate("P2 mirror girth = 3 = k*: N_3 = 3 on all 8 <111> directions; none below",
         np.allclose(vals3, 3.0, atol=TOL) and below < TOL,
         f"N_3 = {np.round(vals3, 6)}, max N_(L<3) = {below:.1e}")

    # P3: screw sector girth 4 + harmonic 8 on the 6 cubic-axis translations
    screws = [(1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, -1, -1)]
    v4 = [N(4, R).real for R in screws]
    v8 = [N(8, R).real for R in screws]
    gate("P3 screw sector: N_4 = 4 and N_8 = 8 on all 6 cubic-axis pitches",
         np.allclose(v4, 4.0, atol=TOL) and np.allclose(v8, 8.0, atol=TOL),
         f"N_4 = {np.round(v4, 6)}, N_8 = {np.round(v8, 6)}")

    # P4: home sector girth (consistency with Phase 1.2)
    v10 = N(10, (0, 0, 0)).real
    gate("P4 home sector: N_10(0) = 120", abs(v10 - 120) < 1e-4, f"N_10(0) = {v10:.6f}")

    print("\n  Sector map: R=0 bulk girth 10 | mirror(odd) girth 3=k* on <111> |")
    print("  screw girth 4 + harmonic 8 on cubic axes. Speculative (NOT gated):")
    print("  mirror triangles on [111] share the C3/generation axis; Koide-phase")
    print("  link is a hypothesis for Phase 1.3 S3, priced before use.")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- sector structure of the crystal zeta established")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
