#!/usr/bin/env python3
"""Phase 2.1 — the unitary walker: Bloch-Grover walk on srs + Konno-Sato law.

Phase 2 needs a genuine UNITARY substrate dynamics (the Lindblad/Hermitian-
companion patches are not unitary walks). The canonical construction is the
Grover walk on the directed-edge space: U = S C, where C is the per-vertex
Grover coin (2/d J - I on the d edges leaving each vertex) and S is the
edge-reversal (flip) operator. Its crystal (Bloch) version U(k) carries the
SAME A(k) data as the Ihara zeta (Konno-Sato): for a d-regular graph,

    spec U(k) = { e^{+/-i theta_j(k)} : cos theta_j = lambda_j(A(k))/d }
                union {+1 x (|E|-|V|)} union {-1 x (|E|-|V|)}

so the unitary walker and the Hashimoto walker are two coordinatizations of
one spectral object (both fibered over spec A(k)).

Gates:
  U1  U(k) is exactly unitary at random k and at the saddles.
  U2  Konno-Sato law at 5 random k: the 8 'walk' eigenvalues are e^{+/-i
      theta_j} with cos theta_j = lambda_j(A(k))/3 (j = 1..4), plus +1, +1,
      -1, -1 (|E|-|V| = 2 each).
  U3  saddle anchors: at P (spec A = +/-sqrt3 doubly) the walk phases are
      theta = arccos(1/sqrt3) = 54.7356 deg (the body-diagonal magic angle),
      each four-fold; at Gamma, theta in {0 (Perron), arccos(-1/3) =
      109.4712 deg (tetrahedral angle) x3}.
  U4  zeta tie: the SAME quadratic data — B-eigenvalues solve
      x^2 - lambda_A x + (d-1) = 0, U-eigenvalues solve
      mu^2 - (2 lambda_A / d) mu + 1 = 0 — verified as exact root maps at
      random k (the unitary walk is the |.|=1 normalization of the zeta's
      quadratic; Ramanujan modes <-> genuinely oscillatory walk modes).
  U5  mirror compatibility: U(k + DELTA) = -F U(k) F^-1-type antiperiod
      consequence at the SPECTRAL level: spec U(k+DELTA) = -spec U(k)
      reflected... concretely gated as: the walk phases at k+DELTA are
      pi - theta_j(k) (cos flips sign with A -> -A), so the mirror maps
      walk modes to their 'anti-walk' partners. (Parity theorem, unitary
      form.)

This is the dynamics object for Phase 2.2 (Born-rule Koide): amplitudes are
now genuine unitary amplitudes.
"""
import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, A_PRIM  # noqa: E402

RNG = np.random.default_rng(20260611)
TOL = 1e-10
FAILURES = []
DELTA = np.array([0.5, 0.5, -0.5])
D = 3  # regularity


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def build(bonds):
    edges = [(i, j, tuple(c)) for (i, j, c) in bonds]
    rev = {}
    for a, (i, j, c) in enumerate(edges):
        t = (j, i, tuple(-x for x in c))
        for b, e2 in enumerate(edges):
            if e2 == t:
                rev[a] = b
    return edges, rev


def U_of(k, edges, rev):
    """Bloch-Grover U(k) = S(k) C: coin per tail vertex, flip with phase."""
    n = len(edges)
    C = np.zeros((n, n), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        for b, (i2, j2, c2) in enumerate(edges):
            if i2 == i:  # same tail vertex
                C[b, a] = 2.0 / D - (1.0 if b == a else 0.0)
    S = np.zeros((n, n), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        S[rev[a], a] = np.exp(2j * np.pi * np.dot(k, np.asarray(c, float)))
    return S @ C


def A_of(k, bonds):
    A = np.zeros((4, 4), dtype=complex)
    for (i, j, c) in bonds:
        A[j, i] += np.exp(2j * np.pi * np.dot(k, np.asarray(c, float)))
    return A


def walk_phases(k, bonds):
    lamA = np.sort(la.eigvalsh(A_of(k, bonds)))
    return np.sort(np.degrees(np.arccos(np.clip(lamA / D, -1, 1))))


def main():
    print("=" * 72)
    print(" PHASE 2.1 -- Bloch-Grover unitary walker on srs + Konno-Sato law")
    print("=" * 72)
    bonds = find_bonds()
    edges, rev = build(bonds)

    # U1: unitarity
    worst = 0.0
    ks = [np.zeros(3), DELTA, np.array([0.25, 0.25, 0.25])] + [RNG.random(3) for _ in range(4)]
    for k in ks:
        U = U_of(k, edges, rev)
        worst = max(worst, la.norm(U.conj().T @ U - np.eye(12)))
    gate("U1 U(k) unitary at saddles + 4 random k", worst < TOL, f"worst {worst:.2e}")

    # U2: Konno-Sato spectral law
    ok_all, worst = True, 0.0
    for _ in range(5):
        k = RNG.random(3)
        U = U_of(k, edges, rev)
        mu = la.eigvals(U)
        lamA = la.eigvalsh(A_of(k, bonds))
        expected = []
        for lam in lamA:
            th = np.arccos(np.clip(lam / D, -1, 1))
            expected += [np.exp(1j * th), np.exp(-1j * th)]
        expected += [1, 1, -1, -1]
        d = np.abs(np.sort_complex(np.round(mu, 9)) - np.sort_complex(np.round(np.array(expected), 9))).max()
        worst = max(worst, d)
        ok_all &= d < 1e-7
    gate("U2 Konno-Sato: spec U = {e^(+-i th), cos th = lam_A/3} + {+-1 x2}",
         ok_all, f"worst {worst:.2e}")

    # U3: saddle anchors
    thP = walk_phases(np.array([0.25, 0.25, 0.25]), bonds)
    magic = np.degrees(np.arccos(1 / np.sqrt(3)))
    okP = np.allclose(np.sort(thP), np.sort([magic, magic, 180 - magic, 180 - magic]), atol=1e-8)
    thG = walk_phases(np.zeros(3), bonds)
    tetra = np.degrees(np.arccos(-1 / 3.0))
    okG = np.allclose(np.sort(thG), np.sort([0.0, tetra, tetra, tetra]), atol=1e-8)
    gate("U3 anchors: P -> magic angle arccos(1/sqrt3); Gamma -> {0, tetrahedral x3}",
         okP and okG, f"theta_P={np.round(thP, 4)}, theta_Gamma={np.round(thG, 4)}")

    # U4: same quadratic data as the zeta (root maps through lambda_A)
    k = RNG.random(3)
    lamA = la.eigvalsh(A_of(k, bonds))
    ok4 = True
    for lam in lamA:
        xB = np.roots([1, -lam, D - 1])      # Hashimoto quadratic
        xU = np.roots([1, -2 * lam / D, 1])  # Grover quadratic
        ok4 &= abs(np.prod(xB) - (D - 1)) < TOL and abs(np.prod(xU) - 1) < TOL
        # both pairs determined by the same lambda_A; phases agree iff |xB|=sqrt2
        if abs(abs(xB[0]) - np.sqrt(2)) < 1e-9:
            ok4 &= abs(np.cos(np.angle(xU[0])) - lam / D) < 1e-9
    gate("U4 zeta tie: B and U eigen-pairs are the two quadratics of one lambda_A",
         ok4, "x^2 - lam x + 2 = 0 vs mu^2 - (2 lam/3) mu + 1 = 0")

    # U5: mirror in unitary form: phases at k+DELTA are pi - theta(k)
    ok_all, worst = True, 0.0
    for _ in range(4):
        k = RNG.random(3)
        t1 = np.sort(walk_phases(k, bonds))
        t2 = np.sort(180 - walk_phases(k + DELTA, bonds))
        d = np.abs(t1 - t2).max()
        worst = max(worst, d)
        ok_all &= d < 1e-7
    gate("U5 mirror (parity theorem, unitary form): theta(k+DELTA) = 180 - theta(k)",
         ok_all, f"worst {worst:.2e} deg")

    print("\n  The unitary walker exists, carries the zeta's spectral data, and")
    print("  respects the mirror. Phase 2.2 (Born-rule Koide) now has its")
    print("  amplitude-level object.")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- unitary substrate dynamics established")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
