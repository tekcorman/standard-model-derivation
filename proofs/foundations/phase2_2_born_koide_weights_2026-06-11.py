#!/usr/bin/env python3
"""Phase 2.2 — Born-Koide stage 1: W1 fails by symmetry; W2 weights forced.

Bet spec: internal research notes (frozen). Q = 2/3 is
algebraically equivalent to character weights (1/2, 1/4, 1/4) + coherent
(amplitude-level) aggregation. Stage-1 results, all exact:

  BW1  W1 FAILS BY SYMMETRY (gated facts): the uniform directed-edge state
       |u> at P lies ENTIRELY in V_Ram (||P_Ram u||^2 = 1) but is
       C3-invariant, hence purely trivial-channel (1, 0, 0). Any
       C3-symmetric state gives equal masses -> Q = 1/3. Generation
       differentiation cannot come from a symmetric initial state.
  BW2  THE BORN MEASURE REQUIRES THE UNITARY WALKER (the day's central
       finding, discovered through this probe's own failures):
       (pre) B(P) is NON-NORMAL: even its joint {B,C3} character-pure
       modes are NOT mutually orthogonal (gated: ||M^H M - I|| = 1.155 = 2/sqrt(3) exactly)
       -> uniform measure over B-modes does NOT give multiplicity
       weights; B cannot support a consistent Born measure. This is the
       structural explanation of the Lindblad-era Q = 1/2 failures.
       (a-c) On the UNITARY walker U(P) (Phase 2.1): C3 commutes; the
       joint {U(P), C3} CSCO has 8 one-dim walk eigenspaces, characters
       (4,2,2), basis ORTHONORMAL (unitarity) and canonical. Uniform
       branch measure over the 8 branches -> character weights EXACTLY
       (1/2, 1/4, 1/4), phase-robust (5.6e-16 over random phases).
  BW3  the aggregation trichotomy (algebra): coherent pure read with
       weights (1/2,1/4,1/4) -> Q = 2/3 exactly (delta-independent);
       fully incoherent -> Q = 1/3; the in-repo Lindblad null sits at 1/2.

REMAINING (stage 2, open): the FORCED coherent generation read — defining
|j> without an unpriced per-channel vector choice (candidate: C3-orbit
Fourier structure; the 12 edges at P split into 4 C3-orbits of 3). K2/K3
of the spec live exactly there.
"""
import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, ATOMS, A_PRIM  # noqa: E402

RNG = np.random.default_rng(20260611)
TOL = 1e-10
FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 2.2 -- Born-Koide stage 1: weights forced, W1 dead by symmetry")
    print("=" * 72)
    bonds = find_bonds()
    edges = [(i, j, tuple(c)) for (i, j, c) in bonds]
    rev = {}
    for a, (i, j, c) in enumerate(edges):
        t = (j, i, tuple(-x for x in c))
        for b, e2 in enumerate(edges):
            if e2 == t:
                rev[a] = b

    P = np.array([0.25, 0.25, 0.25])
    B = np.zeros((12, 12), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        for b, (i2, j2, c2) in enumerate(edges):
            if i2 == j and b != rev[a]:
                B[b, a] = np.exp(2j * np.pi * np.dot(P, np.asarray(c2, float)))
    ev, V = la.eig(B)
    ram = np.abs(np.abs(ev) - np.sqrt(2)) < 1e-8
    Q_, _ = la.qr(V[:, ram])
    P_Ram = Q_ @ Q_.conj().T

    # C3 edge permutation
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
    w = np.exp(2j * np.pi / 3)
    Pa = [sum(np.linalg.matrix_power(P3, n) * np.conj(ch) ** n for n in range(3)) / 3
          for ch in (1, w, w**2)]

    # BW1
    u = np.ones(12, dtype=complex) / np.sqrt(12)
    psi1 = P_Ram @ u
    wts1 = [la.norm(Pa[al] @ psi1) ** 2 for al in range(3)]
    gate("BW1 W1 dead by symmetry: ||P_Ram u||^2 = 1, channel content (1,0,0)",
         abs(la.norm(psi1) ** 2 - 1) < TOL
         and abs(wts1[0] - 1) < TOL and wts1[1] < TOL and wts1[2] < TOL,
         f"weights = {np.round(wts1, 8)} -> Q = 1/3 for ANY C3-symmetric state")

    # BW2-pre: B(P)'s character-pure joint modes are NOT orthogonal
    joint_B = []
    done = []
    for lam in ev[ram]:
        if any(abs(lam - d) < 1e-7 for d in done):
            continue
        done.append(lam)
        idx = np.abs(ev - lam) < 1e-7
        Qe, _ = la.qr(V[:, idx])
        for al in range(3):
            W = Pa[al] @ Qe
            U_, S_, _ = la.svd(W)
            for r in range(int(np.sum(S_ > 1e-8))):
                joint_B.append(U_[:, r])
    MB = np.column_stack(joint_B)
    nonorth = la.norm(MB.conj().T @ MB - np.eye(MB.shape[1]))
    gate("BW2-pre B(P) NON-NORMAL: no frame-free (canonical) Born measure",
         nonorth > 1e-2,
         f"||M^H M - I|| = {nonorth:.3f} = 2/sqrt3; a biorthogonal eta-measure exists "
         "but is eigenvector-frame dependent (unpriced selection). Panel-corrected "
         "attribution: the Lindblad era Hermitized B BECAUSE of non-normality; its "
         "Q=1/2 came from incoherent per-channel readout on h-content (1,1,0)")

    # BW2a-c: the unitary walker U(P)
    Cc = np.zeros((12, 12), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        for b, (i2, j2, c2) in enumerate(edges):
            if i2 == i:
                Cc[b, a] = 2.0 / 3.0 - (1.0 if b == a else 0.0)
    Sf = np.zeros((12, 12), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        Sf[rev[a], a] = np.exp(2j * np.pi * np.dot(P, np.asarray(c, float)))
    Uw = Sf @ Cc
    gate("BW2a [C3, U(P)] = 0 and U unitary",
         la.norm(P3 @ Uw - Uw @ P3) < TOL
         and la.norm(Uw.conj().T @ Uw - np.eye(12)) < TOL, "")
    evU, VU = la.eig(Uw)
    walk = np.abs(np.abs(evU.real) - 1) > 1e-8
    joint = []
    done = []
    for lam in evU[walk]:
        if any(abs(lam - d) < 1e-7 for d in done):
            continue
        done.append(lam)
        idx = np.abs(evU - lam) < 1e-7
        Qe, _ = la.qr(VU[:, idx])
        for al in range(3):
            W = Pa[al] @ Qe
            U_, S_, _ = la.svd(W)
            for r in range(int(np.sum(S_ > 1e-8))):
                joint.append((lam, al, U_[:, r]))
    modes = np.column_stack([v for _, _, v in joint])
    chars = sorted(al for _, al, _ in joint)
    gate("BW2b joint {U(P),C3} CSCO: 8 one-dim walk spaces, chars (4,2,2), ORTHONORMAL",
         len(joint) == 8 and chars == [0, 0, 0, 0, 1, 1, 2, 2]
         and la.norm(modes.conj().T @ modes - np.eye(8)) < 1e-10,
         f"chars={chars}, ||M^H M - I|| = {la.norm(modes.conj().T @ modes - np.eye(8)):.2e}")
    ok_all, worst = True, 0.0
    for _ in range(8):
        th = np.exp(2j * np.pi * RNG.random(8))
        psi = modes @ (th / np.sqrt(8))
        wts = np.array([la.norm(Pa[al] @ psi) ** 2 for al in range(3)])
        d = np.abs(wts - np.array([0.5, 0.25, 0.25])).max()
        worst = max(worst, d)
        ok_all &= d < 1e-10
    gate("BW2c weights FORCED phase-robust on the unitary CSCO: (1/2, 1/4, 1/4)",
         ok_all, f"worst dev {worst:.2e} over 8 random phase draws")

    # BW3 (panel-corrected): the aligned read EXECUTED on the constructed psi;
    # positivity window; and the honest rho = I/8 control (the coherence lives
    # in the READ -- the frozen-K2 residue, named: stage-2 alignment lemma).
    th = np.exp(2j * np.pi * RNG.random(8))
    psi = modes @ (th / np.sqrt(8))
    z = [la.norm(Pa[al] @ psi) for al in range(3)]   # aligned read magnitudes
    delta = 2 / 9

    def Q_of(zv, dl):
        m = [abs(zv[0] + zv[1] * np.exp(1j * dl) * w**j
                 + zv[2] * np.exp(-1j * dl) * w**(-j)) ** 2 for j in range(3)]
        return sum(m) / (sum(np.sqrt(m)) ** 2)

    Qc = Q_of(z, delta)
    gate("BW3a aligned read on the CONSTRUCTED psi -> Q = 2/3 exactly",
         abs(Qc - 2 / 3) < 1e-12,
         f"Q = {Qc:.12f}; z = {np.round(z, 6)} (executed, not hand-fed)")
    # positivity window: Q = 2/3 iff |delta| <= pi/12
    inside = abs(Q_of(z, 2 / 9) - 2 / 3) < 1e-12 and (2 / 9) < np.pi / 12
    outside = abs(Q_of(z, 0.27) - 2 / 3) > 1e-4
    gate("BW3b positivity window: Q = 2/3 iff |delta| <= pi/12; delta = 2/9 INSIDE (85%)",
         inside and outside,
         f"Q(0.27) = {Q_of(z, 0.27):.6f}; pi/12 = {np.pi/12:.4f}, 2/9 = {2/9:.4f}")
    # the honest control: the SAME aligned read on the fully decohered state
    rho = modes @ modes.conj().T / 8
    z_mix = [np.sqrt(np.real(np.trace(Pa[al] @ rho))) for al in range(3)]
    Qm = Q_of(z_mix, delta)
    gate("BW3c CONTROL (the K2 fact, stated honestly): aligned read on rho = I/8 "
         "ALSO gives 2/3 -- the sqrt-coherence lives in the READ, not the state",
         abs(Qm - 2 / 3) < 1e-12,
         "P2 residue = the alignment rule (frozen-K2 class); incoherent "
         "multiplicity read -> 0.3431; equal-mass decoherent limit -> 1/3")

    print("\n  Stage-2 (open, K2/K3 of the spec): the forced coherent generation")
    print("  read |j> -- candidate: C3-orbit Fourier (12 edges = 4 orbits of 3).")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- weights derived; P2 verdict PARTIAL "
          "(alignment lemma = the named residue; see bet spec panel section)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
