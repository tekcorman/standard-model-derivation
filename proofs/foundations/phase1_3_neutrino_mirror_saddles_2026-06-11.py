#!/usr/bin/env python3
"""Phase 1.3 — the saddle orbit map of the body-centering mirror.

[PANEL-CORRECTED 2026-06-11: interpretation reworded per ultracode
adjudication (PARTIAL); gates unchanged — the arithmetic passed unanimously
and is in fact a matrix identity B(k+DELTA) = -B(k). See
internal research notes "PANEL VERDICT".]

The body-centering mirror shift DELTA = (1/2,1/2,-1/2)_prim acts on the four
Ramanujan saddles of the srs Bloch-Hashimoto structure as:

    Gamma + DELTA = H        (exact; the mirror EXPLAINS the Gamma/H
                              antipodality A(H) = -A(Gamma) documented
                              2026-05-21, on which the 2026-05-27 walker
                              dictionary's nu assignment was built — one
                              fact, two readings; NOT independent
                              corroboration)
    P + DELTA == -P (mod Z3) (mirror acts at P as complex conjugation:
                              self-conjugate)
    N + DELTA ~ N            (self-conjugate; spectrum +/-{1, sqrt5};
                              N hosts dark/inert + 4 chir-7 NEUTRINO
                              spillover modes per the dictionary)

Conditional on the 2026-05-27 dictionary (A5-mass freedom priced 15.3 bits),
the neutrino's PRIMARY saddle pair (Gamma,H) is the unique mirror-exchanged
fermionic pair — CANDIDATE-grade consilience consistent with a Majorana
reading; NOT forced (forcing rule open). NG4 below is a CONSISTENCY CHECK of
ADOPTED-NU-MAJ-PHASE (undischarged adoption), not a derivation chain; the
4-trivial-mode count is the fiber-generic Ihara-Bass (1-u^2)^{|E|-|V|}
factor (4 at every generic k, 5 at Gamma/H), and the phase is read at the
self-conjugate P saddle (the dictionary-vs-mass-chain nu_R address seam is
OPEN).

Gates:
  NG1  H_prim = DELTA exactly, and det(I-uB(DELTA)) = det(I-uB_sgn(Gamma)):
       the nu_R fiber is the sign-twisted nu_L fiber (the L(u,sgn) fiber).
  NG2  P + DELTA + P in Z3, and spec B(P+DELTA) = conj spec B(P) (multisets).
  NG3  spec A(N+DELTA) = spec A(N) = +/-{1, sqrt5} (self-conjugate, mirror-
       invisible like P).
  NG4  massless criterion exact: exactly 4 eigenvalues of B(P) with
       lambda^g = 1 (the trivial +/-1 doublets -> m_nu1 = 0 via rank-2
       seesaw), and the 8 Ramanujan modes carry arg(lambda^g) =
       +/-162.388deg = +/-(g*arg h mod 360) = the alpha_21 Majorana phase
       (preregistration register row 7) as GIRTH HOLONOMY.
  NG5  spec A(H) = -spec A(Gamma): nu_L / nu_R adjacency spectra are exact
       mirror negatives ({3,-1,-1,-1} vs {-3,1,1,1}).
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
G_GIRTH = 10


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 1.3 -- neutrino/mirror leg: the seesaw is the mirror swap")
    print("=" * 72)
    bonds = find_bonds()
    edges = [(i, j, tuple(c)) for (i, j, c) in bonds]
    rev = {}
    for a, (i, j, c) in enumerate(edges):
        target = (j, i, tuple(-x for x in c))
        for b, e2 in enumerate(edges):
            if e2 == target:
                rev[a] = b

    def A_of(k):
        A = np.zeros((4, 4), dtype=complex)
        for (i, j, c) in bonds:
            A[j, i] += np.exp(2j * np.pi * np.dot(k, np.asarray(c, float)))
        return A

    def B_of(k, sign=+1):
        B = np.zeros((12, 12), dtype=complex)
        for a, (i, j, c) in enumerate(edges):
            for b, (i2, j2, c2) in enumerate(edges):
                if i2 == j and b != rev[a]:
                    B[b, a] = sign * np.exp(2j * np.pi * np.dot(k, np.asarray(c2, float)))
        return B

    def poly_eq(fL, fR, n=25):
        us = 1.2 * (RNG.random(n) - 0.5 + 1j * (RNG.random(n) - 0.5))
        worst = max(abs(fL(u) - fR(u)) / max(1.0, abs(fL(u))) for u in us)
        return worst < TOL, worst

    # NG1: H = mirror image of Gamma
    H_prim = A_PRIM @ np.array([0.0, 0.0, 1.0])
    ok1 = np.allclose(H_prim, DELTA, atol=1e-12)
    BH = B_of(DELTA)
    Bsg = B_of(np.zeros(3), sign=-1)
    ok2, worst = poly_eq(lambda u: la.det(np.eye(12) - u * BH),
                         lambda u: la.det(np.eye(12) - u * Bsg))
    gate("NG1 H = Gamma + DELTA; nu_R fiber = sign-twisted nu_L fiber",
         ok1 and ok2, f"H_prim={np.round(H_prim, 6)}, det worst {worst:.2e}")

    # NG2: P self-conjugate up to inversion; mirror acts as conjugation
    P = np.array([0.25, 0.25, 0.25])
    integral = np.allclose((P + DELTA + P) % 1.0, 0.0, atol=1e-12)
    sP = np.sort_complex(np.round(la.eigvals(B_of(P)), 9))
    sPD = np.sort_complex(np.round(la.eigvals(B_of(P + DELTA)), 9))
    conj_match = np.allclose(np.sort_complex(np.conj(sP)), sPD, atol=1e-8)
    gate("NG2 P + DELTA == -P (mod Z3); spec B(P+DELTA) = conj spec B(P)",
         integral and conj_match, f"P+DELTA+P integral={integral}")

    # NG3: N self-conjugate, spectrum +/-{1, sqrt5}
    N = A_PRIM @ np.array([0.0, 0.5, 0.5])
    sN = np.sort(la.eigvalsh(A_of(N)))
    sND = np.sort(la.eigvalsh(A_of(N + DELTA)))
    expect = np.sort([-np.sqrt(5), -1.0, 1.0, np.sqrt(5)])
    gate("NG3 spec A(N+DELTA) = spec A(N) = +/-{1, sqrt5}",
         np.allclose(sN, sND, atol=TOL) and np.allclose(sN, expect, atol=TOL),
         f"spec={np.round(sN, 6)}")

    # NG4: exact massless criterion + Majorana phase as girth holonomy
    evP = la.eigvals(B_of(P))
    hol = evP ** G_GIRTH
    trivial = np.abs(hol - 1) < 1e-8
    n_triv = int(trivial.sum())
    ram = evP[~trivial]
    args = np.degrees(np.angle(ram ** G_GIRTH))
    alpha21 = (G_GIRTH * np.degrees(np.arctan(np.sqrt(5) / np.sqrt(3)))) % 360
    phase_ok = np.allclose(np.sort(np.abs(args)), alpha21, atol=1e-6)
    gate("NG4 exactly 4 modes with lambda^g = 1 (m_nu1=0); 8 with arg = +/-alpha_21",
         n_triv == 4 and phase_ok,
         f"trivial={n_triv}, |arg(l^g)|={np.round(np.sort(np.abs(args))[0], 4)} vs alpha21={alpha21:.4f}")

    # NG5: nu_L / nu_R adjacency spectra are mirror negatives
    sG = np.sort(la.eigvalsh(A_of(np.zeros(3))))
    sH = np.sort(la.eigvalsh(A_of(DELTA)))
    gate("NG5 spec A(H) = -spec A(Gamma) ({3,-1,-1,-1} vs {-3,1,1,1})",
         np.allclose(sH, -sG[::-1], atol=TOL), f"A(Gamma)={np.round(sG, 6)}")

    print("\n  Saddle orbit map (exact): Gamma <-> H exchanged; P -> conj(P),")
    print("  N -> N self-conjugate. Neutrino reading: dictionary-conditional")
    print("  CANDIDATE consilience (panel verdict PARTIAL); NG4 = consistency")
    print("  check of ADOPTED-NU-MAJ-PHASE, not a derivation chain.")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- saddle orbit map exact (interpretation: see spec)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
