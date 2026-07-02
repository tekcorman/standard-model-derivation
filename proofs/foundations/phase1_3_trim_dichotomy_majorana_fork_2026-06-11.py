#!/usr/bin/env python3
"""Phase 1.3 — the TRIM dichotomy and the Majorana-phase fork (the nu_R seam).

A momentum-diagonal Majorana coupling pairs a mode at k with its conjugate at
-k, so it requires k == -k (mod Z^3): a time-reversal-invariant momentum
(TRIM). Exact arithmetic on the four Ramanujan saddles:

  M1  Gamma, H, N are TRIM (-k == k mod Z^3): a mirror-DIAGONAL Majorana
      coupling can live there.
  M2  P is the UNIQUE non-TRIM saddle, and its conjugation partner is its
      MIRROR IMAGE: -P == P + DELTA (mod Z^3). A Majorana coupling involving
      the P modes is necessarily MIRROR-CROSSING. Spectrally:
      spec conj(B(P)) = spec B(P + DELTA).
  M3  The fork's two exact phase candidates (g = 10 girth holonomy):
      P-reading (current mass chain / ADOPTED-NU-MAJ-PHASE):
        arg(lambda^10) = +/-162.3876 deg  (lambda = (sqrt3 +/- i sqrt5)/2)
      Gamma/H-reading (walker dictionary's nu address):
        arg(lambda^10) = +/-27.0481 deg   (lambda = (+/-1 +/- i sqrt7)/2;
        Gamma and H give the SAME set since (-lambda)^10 = lambda^10 --
        the parity theorem makes the mirror invisible in even powers).
  M4  |lambda^10| = 2^5 = 32 at ALL Ramanujan saddles (|lambda|^2 = 2 is the
      Ramanujan property): holonomy MAGNITUDES are address-blind; only the
      PHASES discriminate.

THE FORK (documented, NOT resolved here): alpha_21 = 162.39 deg
(preregistration row 7, frozen; the P/mirror-crossing reading) vs 27.05 deg
(the H/mirror-diagonal reading). Neither address is derived. Named
discriminator for the next session: the mirror parity of the Majorana
condensate -- mirror-EVEN condensate -> TRIM/H reading; mirror-ODD
condensate (nontrivial mirror holonomy, cf. grade-blind mass
classification) -> P/mirror-crossing reading. The register row STANDS until
a forcing rule exists; any supersession follows the preregistration
protocol (new dated row, old row preserved).
"""
import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, A_PRIM  # noqa: E402

TOL = 1e-10
FAILURES = []
DELTA = np.array([0.5, 0.5, -0.5])


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def is_integer(v):
    return np.allclose(np.mod(v + 0.5, 1.0) - 0.5, 0.0, atol=1e-12)


def main():
    print("=" * 72)
    print(" PHASE 1.3 -- TRIM dichotomy: the Majorana-phase fork at the nu_R seam")
    print("=" * 72)
    bonds = find_bonds()
    edges = [(i, j, tuple(c)) for (i, j, c) in bonds]
    rev = {}
    for a, (i, j, c) in enumerate(edges):
        target = (j, i, tuple(-x for x in c))
        for b, e2 in enumerate(edges):
            if e2 == target:
                rev[a] = b

    def B_of(k):
        B = np.zeros((12, 12), dtype=complex)
        for a, (i, j, c) in enumerate(edges):
            for b, (i2, j2, c2) in enumerate(edges):
                if i2 == j and b != rev[a]:
                    B[b, a] = np.exp(2j * np.pi * np.dot(k, np.asarray(c2, float)))
        return B

    G = np.zeros(3)
    H = A_PRIM @ np.array([0.0, 0.0, 1.0])
    P = np.array([0.25, 0.25, 0.25])
    N = A_PRIM @ np.array([0.0, 0.5, 0.5])

    # M1: TRIM status
    trims = {nm: is_integer(2 * k) for nm, k in [("Gamma", G), ("H", H), ("N", N)]}
    p_trim = is_integer(2 * P)
    gate("M1 Gamma, H, N are TRIM; P is NOT",
         all(trims.values()) and not p_trim, f"{trims}, P TRIM={p_trim}")

    # M2: P's conjugation partner is its mirror image
    ok_arith = is_integer(-P - (P + DELTA))
    sC = np.sort_complex(np.round(np.conj(la.eigvals(B_of(P))), 9))
    sM = np.sort_complex(np.round(la.eigvals(B_of(P + DELTA)), 9))
    gate("M2 -P == P + DELTA (mod Z3); spec conj(B(P)) = spec B(P+DELTA)",
         ok_arith and np.allclose(sC, sM, atol=1e-8), f"arith={ok_arith}")

    # M3: the two exact phase candidates
    def ram_phases(k):
        ev = la.eigvals(B_of(k))
        ram = ev[np.abs(np.abs(ev) - np.sqrt(2)) < 1e-8]
        return np.unique(np.round(np.abs(np.degrees(np.angle(ram ** 10))), 4))

    phP = ram_phases(P)
    phG = ram_phases(G)
    phH = ram_phases(H)
    a21_P = (10 * np.degrees(np.arctan(np.sqrt(5) / np.sqrt(3)))) % 360
    a21_GH = abs(((10 * np.degrees(np.arctan(np.sqrt(7)))) % 360) - 360)
    okP = np.allclose(phP, round(a21_P, 4), atol=1e-3)
    okGH = np.allclose(phG, round(a21_GH, 4), atol=1e-3) and np.allclose(phH, phG, atol=1e-6)
    gate("M3 fork phases exact: P -> 162.3876; Gamma/H -> 27.0481 (same set)",
         okP and okGH, f"P={phP}, Gamma={phG}, H={phH}")

    # M4: magnitudes are address-blind
    mags = []
    for k in (P, G, H):
        ev = la.eigvals(B_of(k))
        ram = ev[np.abs(np.abs(ev) - np.sqrt(2)) < 1e-8]
        mags.append(np.abs(ram ** 10))
    gate("M4 |lambda^10| = 32 at P, Gamma, H (Ramanujan: magnitudes address-blind)",
         all(np.allclose(m, 32.0, atol=1e-8) for m in mags),
         f"means={[float(np.round(m.mean(), 8)) for m in mags]}")

    print("\n  THE FORK: alpha_21 = 162.39 deg (P / mirror-CROSSING Majorana,")
    print("  preregistered row 7) vs 27.05 deg (H / mirror-DIAGONAL Majorana,")
    print("  walker-dictionary address). Discriminator to test: mirror parity")
    print("  of the Majorana condensate. Register row STANDS pending a forcing rule.")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- TRIM dichotomy exact; fork posed precisely")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
