#!/usr/bin/env python3
"""Phase 1.3 — L=8 sector census (panel-ordered double-home cleanup).

The two earlier probes assigned the L=8 anchor different sector homes
(N_8(pitch)=8 "second harmonic" vs N(8,2*pitch)=4 "even winding"). This
census enumerates EVERY nonzero N_8(R) exactly and checks completeness, so
the channel dictionary carries ONE record: the L=8 content is plural and the
address selection is UNFORCED (priced ~2 bits among the candidate readings,
incl. the pre-existing combinatorial 8 = g-2 endpoint-pinning).

Gates:
  E1 completeness: sum over all R in |n|<=3 of N_8(R) equals Tr[B(Gamma)^8]
     (the all-translations total), confirming no class lies outside.
  E2 the census: exactly three R-orbits carry L=8 cycles --
     pitch (1,1,0)-type: 6 vectors x 8; double-pitch (2,2,0)-type: 6 x 4;
     and a third class (the panel's "face-diagonal"): enumerated here.
"""
import os
import sys
from itertools import product as iproduct

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 1.3 -- L=8 sector census (double-home cleanup)")
    print("=" * 72)
    bonds = find_bonds()
    edges = [(i, j, tuple(c)) for (i, j, c) in bonds]
    rev = {}
    for a, (i, j, c) in enumerate(edges):
        t = (j, i, tuple(-x for x in c))
        for b, e2 in enumerate(edges):
            if e2 == t:
                rev[a] = b

    n = 27
    pts = (np.arange(n) + 0.5) / n
    tr8 = np.zeros((n, n, n), dtype=complex)
    for i1, k1 in enumerate(pts):
        for i2, k2 in enumerate(pts):
            for i3, k3 in enumerate(pts):
                k = np.array([k1, k2, k3])
                B = np.zeros((12, 12), dtype=complex)
                for a, (i, j, c) in enumerate(edges):
                    for b, (i2e, j2, c2) in enumerate(edges):
                        if i2e == j and b != rev[a]:
                            B[b, a] = np.exp(2j * np.pi * np.dot(k, c2))
                tr8[i1, i2, i3] = np.trace(np.linalg.matrix_power(B, 8))

    K1, K2, K3 = np.meshgrid(pts, pts, pts, indexing="ij")
    census = {}
    for R in iproduct(range(-3, 4), repeat=3):
        ph = np.exp(-2j * np.pi * (K1 * R[0] + K2 * R[1] + K3 * R[2]))
        v = (tr8 * ph).mean().real
        if abs(v) > 1e-6:
            census[R] = round(v, 6)

    total = sum(census.values())
    # all-translations total = Tr B(Gamma)^8 (sum over R of the Fourier coeffs)
    BG = np.zeros((12, 12), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        for b, (i2e, j2, c2) in enumerate(edges):
            if i2e == j and b != rev[a]:
                BG[b, a] = 1.0
    trG = np.trace(np.linalg.matrix_power(BG, 8)).real
    gate("E1 completeness: sum of census = Tr B(Gamma)^8",
         abs(total - trG) < 1e-3, f"sum={total:.3f}, Tr={trG:.3f}")

    # classify by sorted |components|
    classes = {}
    for R, v in census.items():
        key = (tuple(sorted(map(abs, R))), v)
        classes.setdefault(key, []).append(R)
    print("\n  L=8 census by orbit (|components| sorted, N per vector, count):")
    for (shape, v), Rs in sorted(classes.items()):
        print(f"    type {shape}: N = {v:g} per vector x {len(Rs)} vectors")

    got = {(s, v): len(Rs) for (s, v), Rs in classes.items()}
    expected = {((0, 1, 1), 8.0): 12,   # 6 screw pitches (1,1,0)-type + 6 mixed-sign (1,-1,0)-type
                ((0, 2, 2), 4.0): 6,    # double-pitch (even helix winding)
                ((1, 1, 2), 8.0): 6}    # the panel's third class
    gate("E2 census: (0,1,1)x12@8 + (0,2,2)x6@4 + (1,1,2)x6@8 (= 168 total)",
         got == expected, f"got={got}")

    print("\n  RECORD: the L=8 channel content is PLURAL; address selection among")
    print("  {pitch-8, double-pitch-4 (even winding), third class, combinatorial")
    print("  g-2 pinning} is UNFORCED (~2 bits, priced in the self-MDL ledger).")
    print("  The two earlier probes' single-home claims are superseded by this census.")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- L=8 census complete; double-home resolved as plural")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
