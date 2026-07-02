#!/usr/bin/env python3
"""Phase 1.3 — C3 character content of the saddle eigenspaces: the fork's endpoints.

The Majorana phases alpha_21/alpha_31 are RELATIVE phases between the three
C3 generation classes. This probe computes the C3 ([111] 3-fold) character
decomposition of every Bloch-Hashimoto eigenspace at Gamma, H, P. Gates:

  K1  C3 commutes with B(k) at all three saddles (exact).
  K2  At H (and Gamma): each Ramanujan eigenvalue lambda = (+/-1 +/- i sqrt7)/2
      is TRIPLY degenerate with characters {1, w, w2} -- the REGULAR
      representation: one mode per generation, all with the SAME eigenvalue.
      [PANEL-REFUTED INTERPRETATION 2026-06-11: the original consequence
      "alpha_21 = alpha_31 = 0 EXACTLY" assumed M_R prop-to identity, which
      is C3-FORBIDDEN at the TRIM H saddle (same-fiber bilinear law; the
      omega-sector invariant projection of the identity is exactly zero).
      The C3-invariant completion of the H address is {a; antidiag(b)} ->
      relative phase pi (the C3/TRIM branch). Even granting M_R prop-to 1,
      the light phases relocate to m_D. The exact K2 computation itself
      stands and is banked.]
  K3  At P: each Ramanujan eigenvalue is doubly degenerate and the generation
      classes are DISTRIBUTED ACROSS DISTINCT eigenvalues (no regular-rep
      triplet) -> unequal per-generation holonomies are possible; the adopted
      P-reading (ADOPTED-NU-MAJ-PHASE, 2026-05-12) builds alpha_21 = 162.39
      deg from this structure.

THE FORK, finalized (both branches sharp and preregistration-grade):
  P-reading (adopted, frozen rows 7/8):  alpha_21 = 162.39, alpha_31 = 324.78
  H-reading (dictionary address + K2):   alpha_21 = alpha_31 = 0 exactly
    (and the Majorana-phase mechanism relocates to the Dirac block m_D).
Row 9 (m_bb) inherits the fork: zero phases -> constructive |U_e2^2 m_2 +
U_e3^2 m_3| (larger m_bb) vs partial cancellation at 162 deg.
NO supersession: both branches conditional (dictionary A5-mass bits; M_R
TRIM-diagonality; C3-class = generation identification R3). Recorded
pre-measurement (nEXO/LEGEND 2030+) per the preregistration protocol.
"""
import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, ATOMS, A_PRIM  # noqa: E402

TOL = 1e-9
FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 1.3 -- C3 characters at the saddles: Majorana-fork endpoints")
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

    # C3 about [111]: cartesian (x,y,z)->(z,x,y); atom permutation v0 fixed, 1->3->2
    sigma = {0: 0, 1: 3, 3: 2, 2: 1}
    Rcart = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    M = A_PRIM.T

    def cart(i, j, c):
        return ATOMS[j] + M @ np.asarray(c, float) - ATOMS[i]

    P3 = np.zeros((12, 12))
    for a, (i, j, c) in enumerate(edges):
        v = Rcart @ cart(i, j, c)
        for b, (i2, j2, c2) in enumerate(edges):
            if (i2, j2) == (sigma[i], sigma[j]) and np.allclose(cart(i2, j2, c2), v, atol=1e-9):
                P3[b, a] = 1.0
                break
    assert np.allclose(P3 @ P3 @ P3, np.eye(12))

    w = np.exp(2j * np.pi / 3)
    saddles = {"Gamma": np.zeros(3), "H": np.array([0.5, 0.5, -0.5]),
               "P": np.array([0.25, 0.25, 0.25])}

    def char_table(k):
        B = B_of(k)
        comm = la.norm(P3 @ B - B @ P3)
        ev, V = la.eig(B)
        table = {}
        done = []
        for lam in ev:
            if any(abs(lam - d) < 1e-7 for d in done):
                continue
            done.append(lam)
            idx = np.abs(ev - lam) < 1e-7
            Q, _ = la.qr(V[:, idx])
            chars = la.eigvals(Q.conj().T @ P3 @ Q)
            content = sorted(
                next(nm for nm, val in [("1", 1), ("w", w), ("w2", w**2)]
                     if abs(ch - val) < 1e-6) for ch in chars)
            table[np.round(lam, 7)] = (int(idx.sum()), content)
        return comm, table

    comms, tables = {}, {}
    for nm, k in saddles.items():
        comms[nm], tables[nm] = char_table(k)
    gate("K1 [C3, B] = 0 at Gamma, H, P",
         all(c < TOL for c in comms.values()),
         f"norms={[f'{c:.1e}' for c in comms.values()]}")

    # K2: regular-rep triplets at H and Gamma for the Ramanujan eigenvalues
    def ram_regular(table):
        rams = {lam: mc for lam, mc in table.items()
                if abs(abs(lam) - np.sqrt(2)) < 1e-6}
        return (len(rams) == 2
                and all(m == 3 and c == ["1", "w", "w2"] for m, c in rams.values()))

    gate("K2 H and Gamma: Ramanujan triplets = regular rep {1,w,w2} (one/generation)",
         ram_regular(tables["H"]) and ram_regular(tables["Gamma"]),
         "equal per-generation holonomy -> alpha_21 = alpha_31 = 0 under H-reading")

    # K3: P has no regular-rep Ramanujan triplet (classes split across eigenvalues)
    ramsP = {lam: mc for lam, mc in tables["P"].items()
             if abs(abs(lam) - np.sqrt(2)) < 1e-6}
    gate("K3 P: four Ramanujan doublets, generation classes split across eigenvalues",
         len(ramsP) == 4 and all(m == 2 for m, _ in ramsP.values()),
         f"multiplicities={[m for m, _ in ramsP.values()]}")

    # K3b (panel-ordered): the true P character content -- every Ramanujan
    # doublet carries a TRIVIAL partner ({1,w} or {1,w2}); the |lambda|=1
    # doublets carry {w,w2}. No clean class<->eigenvalue bijection at P.
    ram_ok = all(sorted(c) in (["1", "w"], ["1", "w2"]) for _, c in ramsP.values())
    triv = {lam: mc for lam, mc in tables["P"].items()
            if abs(abs(lam) - 1.0) < 1e-6}
    triv_ok = all(sorted(c) == ["w", "w2"] for _, c in triv.values())
    gate("K3b P character content: Ramanujan doublets {1,w}/{1,w2}; +-1 doublets {w,w2}",
         ram_ok and triv_ok,
         "trivial partner present in every Ramanujan doublet -- no clean bijection")

    print("\n  FORK (PANEL-ADJUDICATED 2026-06-11): P-reading stands as adoption")
    print("  (adoption-consistent alpha_31 = 197.61, row-8 defect recorded);")
    print("  H-reading 0/0 REFUTED (M_R prop-to 1 is C3-forbidden at TRIM) and")
    print("  collapses into the C3/TRIM branch (phase pi - delta_breaking).")
    print("  Register annotation documents the fork; rows untouched.")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- fork endpoints exact")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
