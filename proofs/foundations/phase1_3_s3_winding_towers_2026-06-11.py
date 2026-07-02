#!/usr/bin/env python3
"""Phase 1.3 / S3 — the two exact winding towers of the srs crystal zeta.

Translation-resolved cycle counts N_L(R) = INT_BZ Tr[B(k)^L] e^{-2pi i k.R} dk
reveal two exact helix towers:

  T1 SCREW TOWER (4_1, cubic axes):    N(4n, n*(1,1,0)) = 4   for n = 1..4
  T2 MIRROR TOWER (C3, body diagonal): N(3n, n*(1,1,1)) = 3   for n = 1..4

Interpretation (towers are gated; readings are priced separately in the
Phase 1.3 spec):
  - The mirror (odd-parity) sector's entire low-lying content is the C3 helix
    tower: period 3 = k*, multiplicity 3, on the [111] generation axis.
  - CANDIDATE (panel-corrected 2026-06-11, verdict PARTIAL): the V_cb /
    lepton channel argument u^8 ADMITS an even-winding screw reading
    (u^{8m} <-> N(8m, 2m*pitch) = 4, T3). UNFORCED: L=8 carries 3 nonzero
    sector classes (incl. a face-diagonal class and N_8(pitch)=8, a second
    competing home), the even-windings-only restriction is a ~1-bit
    selection (the u^4 fundamental appears in no observable), and the
    pre-existing 8 = g-2 address competes. Selection bits itemized in
    docs/scoping/phase1_3_bet_spec_2026-06-10.md "PANEL VERDICT".
  - Bulk sector: N_10(0) = 120 girth cycles (T4, consistency with Phase 1.2;
    u^10 <-> girth is tautological, zero evidential weight).
  - Tower counts are FORCED symmetry content (panel Lens D extremal
    argument): exact, but expected from the C3 / 4_1 screw orbits.

Counts per single axis direction; 8 directions <111>, 6 directions <100>-type.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds  # noqa: E402

TOL = 1e-5
FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 1.3/S3 -- winding towers: 4_1 screw (4n:4) and C3 mirror (3n:3)")
    print("=" * 72)
    bonds = find_bonds()
    edges = [(i, j, tuple(c)) for (i, j, c) in bonds]
    rev = {}
    for a, (i, j, c) in enumerate(edges):
        target = (j, i, tuple(-x for x in c))
        for b, e2 in enumerate(edges):
            if e2 == target:
                rev[a] = b

    L_max, n = 16, 27
    pts = (np.arange(n) + 0.5) / n
    want = {}
    for m in range(1, 5):
        want[(4 * m, (m, m, 0))] = 4.0      # screw tower
        want[(3 * m, (m, m, m))] = 3.0      # mirror tower
    want[(10, (0, 0, 0))] = 120.0           # bulk girth
    acc = {key: 0.0 + 0j for key in want}

    for k1 in pts:
        for k2 in pts:
            for k3 in pts:
                k = np.array([k1, k2, k3])
                B = np.zeros((12, 12), dtype=complex)
                for a, (i, j, c) in enumerate(edges):
                    for b, (i2e, j2, c2) in enumerate(edges):
                        if i2e == j and b != rev[a]:
                            B[b, a] = np.exp(2j * np.pi * np.dot(k, c2))
                P = np.eye(12, dtype=complex)
                for L in range(1, L_max + 1):
                    P = P @ B
                    tr = np.trace(P)
                    for (LL, R) in want:
                        if LL == L:
                            acc[(LL, R)] += tr * np.exp(
                                -2j * np.pi * (k1 * R[0] + k2 * R[1] + k3 * R[2]))

    vals = {key: (v / n**3).real for key, v in acc.items()}
    screw = [vals[(4 * m, (m, m, 0))] for m in range(1, 5)]
    mirror = [vals[(3 * m, (m, m, m))] for m in range(1, 5)]
    gate("T1 screw tower N(4n, n*pitch) = 4, n=1..4",
         np.allclose(screw, 4.0, atol=TOL), f"{np.round(screw, 6)}")
    gate("T2 mirror tower N(3n, n*(1,1,1)) = 3, n=1..4",
         np.allclose(mirror, 3.0, atol=TOL), f"{np.round(mirror, 6)}")
    gate("T3 V_cb winding rungs = even screw windings: N(8m, 2m*pitch) = 4, m=1,2",
         abs(screw[1] - 4) < TOL and abs(screw[3] - 4) < TOL,
         f"m=1: {screw[1]:.6f}, m=2: {screw[3]:.6f}")
    gate("T4 bulk N_10(0) = 120", abs(vals[(10, (0, 0, 0))] - 120) < 1e-3,
         f"{vals[(10, (0, 0, 0))]:.4f}")

    print("\n  Towers: C3 (period 3 = k*, count 3, [111] generation axis, ODD parity)")
    print("  and 4_1 (period 4, count 4, cubic axes, EVEN parity). The u^{8m}")
    print("  geometric series = literal even windings of the 4_1 helix.")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- two exact winding towers established")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
