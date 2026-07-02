#!/usr/bin/env python3
"""Phase 1.1 — Ihara/Bass zeta infrastructure for srs and its Z2 mirror cover.

Establishes, at machine precision, the algebraic identities the zeta
unification program (Phase 1) is built on:

  G0  ANCHOR: the Bloch-Hashimoto fiber B(P) on srs reproduces the framework's
      central eigenvalue h = (sqrt(3) + i*sqrt(5))/2, doubly degenerate.
  G1  Bass identity, finite quotient (Gamma fiber = K4):
        det(I12 - u B) = (1-u^2)^2 det(I4 - A u + 2 u^2)
      cross-checked against the classical closed form
        zeta_K4(u)^-1 = (1-u^2)^2 (1-u)(1-2u)(1+u+2u^2)^3.
  G2  Twisted (Bloch) Bass identity at generic k:
        det(I12 - u B(k)) = (1-u^2)^2 det(I4 - A(k) u + 2 u^2)
  G3  Z2 mirror-cover factorization (Stark-Terras), per Bloch fiber:
        det(I24 - u B_cover(k)) = det(I12 - u B(k)) * L(u, sgn; k)^-1,
        L(u, sgn; k)^-1 = (1-u^2)^2 det(I4 + A(k) u + 2 u^2)
      The sign character IS the srs<->srs-z mirror; the L-factor is the
      mirror-holonomy-weighted side where the mass sector lives.
  G4  Finite-level Stark-Terras: zeta_cover^-1 = zeta_K4^-1 * L^-1 against
      exact closed forms (cover adjacency spectrum = +/-{3,-1,-1,-1}).

The cover fiber is built by the voltage construction sigma(e) = -1 on every
edge (bipartite double cover), i.e. B_cover(k) = the Hashimoto operator on
the 8-atom mirror-doubled cell at the same k. Identification of this object
with the repo's crystallographic srs-z fibers is the Phase 1.3 follow-up
(flagged, not assumed here).

All gates PASS/FAIL; any FAIL exits nonzero.
"""
import os
import sys
import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, N_ATOMS  # noqa: E402

RNG = np.random.default_rng(20260610)
TOL = 1e-10
FAILURES = []


def gate(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Directed-edge basis and Bloch operators on the srs primitive cell
# ---------------------------------------------------------------------------
def directed_edges(bonds):
    """bonds: (src, tgt, cell) for all 12 directed bonds (3 per atom)."""
    edges = [(i, j, tuple(cell)) for (i, j, cell) in bonds]
    assert len(edges) == 12
    rev = {}
    for a, (i, j, c) in enumerate(edges):
        cbar = tuple(-x for x in c)
        for b, (i2, j2, c2) in enumerate(edges):
            if (i2, j2, c2) == (j, i, cbar):
                rev[a] = b
    assert len(rev) == 12 and all(rev[rev[a]] == a for a in rev)
    return edges, rev


def bloch_A(k, bonds):
    """4x4 Bloch adjacency: A(k)[j,i] += exp(2*pi*i k.cell) per bond i->j."""
    A = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    for (i, j, cell) in bonds:
        A[j, i] += np.exp(2j * np.pi * np.dot(k, cell))
    return A


def bloch_B(k, edges, rev, sign=+1):
    """12x12 Bloch-Hashimoto: B[e',e] = sgn(e') ph(e') if head(e)=tail(e'),
    e' != reverse(e). sign=-1 applies the Z2 voltage (sign character) to
    every edge -- the twisted operator whose determinant is L(u,sgn)^-1."""
    n = len(edges)
    B = np.zeros((n, n), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        for b, (i2, j2, c2) in enumerate(edges):
            if i2 == j and b != rev[a]:
                B[b, a] = sign * np.exp(2j * np.pi * np.dot(k, c2))
    return B


def cover_B(k, edges, rev):
    """24x24 Bloch-Hashimoto of the Z2 (bipartite/mirror) cover at the same k:
    vertices (v,s); every edge flips s. Directed cover-edges = (e, s_tail)."""
    n = len(edges)
    B = np.zeros((2 * n, 2 * n), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        for b, (i2, j2, c2) in enumerate(edges):
            if i2 == j and b != rev[a]:
                ph = np.exp(2j * np.pi * np.dot(k, c2))
                for s in (0, 1):
                    # edge a enters layer s^1; edge b departs layer s^1
                    B[b * 2 + (s ^ 1), a * 2 + s] = ph
    return B


def poly_eq(fL, fR, n_samples=40, tol=TOL):
    """Compare two entire functions of u on random complex samples in |u|<1.2."""
    us = 1.2 * (RNG.random(n_samples) - 0.5 + 1j * (RNG.random(n_samples) - 0.5))
    worst = 0.0
    for u in us:
        L, R = fL(u), fR(u)
        scale = max(1.0, abs(L), abs(R))
        worst = max(worst, abs(L - R) / scale)
    return worst < tol, worst


def main():
    print("=" * 72)
    print(" PHASE 1.1 -- Ihara/Bass zeta infrastructure: srs + Z2 mirror cover")
    print("=" * 72)
    bonds = find_bonds()
    edges, rev = directed_edges(bonds)

    # --- G0: anchor -- h at the P point --------------------------------------
    kP = np.array([0.25, 0.25, 0.25])
    h = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
    evP = la.eigvals(bloch_B(kP, edges, rev))
    d = np.abs(evP - h)
    mult = int(np.sum(d < 1e-8))
    gate("G0 anchor: h=(sqrt3+i*sqrt5)/2 in spec B(P), multiplicity 2",
         mult == 2 and d.min() < TOL,
         f"min|lambda-h|={d.min():.2e}, mult={mult}")

    # --- G1: Bass identity at Gamma (= classical K4) --------------------------
    kG = np.zeros(3)
    BG = bloch_B(kG, edges, rev)
    AG = bloch_A(kG, bonds)
    ok, worst = poly_eq(
        lambda u: la.det(np.eye(12) - u * BG),
        lambda u: (1 - u**2) ** 2 * la.det(np.eye(4) - u * AG + 2 * u**2 * np.eye(4)))
    gate("G1a Bass identity at Gamma (12x12 vs 4x4)", ok, f"worst rel err {worst:.2e}")
    ok, worst = poly_eq(
        lambda u: la.det(np.eye(12) - u * BG),
        lambda u: (1 - u**2) ** 2 * (1 - u) * (1 - 2 * u) * (1 + u + 2 * u**2) ** 3)
    gate("G1b classical closed form zeta_K4^-1", ok, f"worst rel err {worst:.2e}")

    # --- G2: twisted Bass identity at random generic k ------------------------
    worst_all = 0.0
    for _ in range(5):
        k = RNG.random(3)
        Bk = bloch_B(k, edges, rev)
        Ak = bloch_A(k, bonds)
        ok, worst = poly_eq(
            lambda u: la.det(np.eye(12) - u * Bk),
            lambda u: (1 - u**2) ** 2 * la.det(np.eye(4) - u * Ak + 2 * u**2 * np.eye(4)))
        worst_all = max(worst_all, worst)
        if not ok:
            break
    gate("G2 Bloch Bass identity at 5 random k", ok, f"worst rel err {worst_all:.2e}")

    # --- G3: Z2 mirror-cover factorization per fiber --------------------------
    worst_all = 0.0
    for trial_k in [kG, kP] + [RNG.random(3) for _ in range(3)]:
        Bc = cover_B(trial_k, edges, rev)
        Bk = bloch_B(trial_k, edges, rev)
        Ak = bloch_A(trial_k, bonds)
        ok, worst = poly_eq(
            lambda u: la.det(np.eye(24) - u * Bc),
            lambda u: la.det(np.eye(12) - u * Bk)
            * (1 - u**2) ** 2 * la.det(np.eye(4) + u * Ak + 2 * u**2 * np.eye(4)))
        worst_all = max(worst_all, worst)
        if not ok:
            break
    gate("G3 cover factorization det(I-uB_cover) = zeta-fiber * L(u,sgn)-fiber",
         ok, f"worst rel err {worst_all:.2e}, k in {{Gamma,P,3 random}}")

    # Twisted-B route to the same L-factor (sign voltage on B directly)
    k = RNG.random(3)
    Bs = bloch_B(k, edges, rev, sign=-1)
    Ak = bloch_A(k, bonds)
    ok, worst = poly_eq(
        lambda u: la.det(np.eye(12) - u * Bs),
        lambda u: (1 - u**2) ** 2 * la.det(np.eye(4) + u * Ak + 2 * u**2 * np.eye(4)))
    gate("G3b sign-twisted Hashimoto gives the same L-factor", ok,
         f"worst rel err {worst:.2e}")

    # --- G4: finite-level Stark-Terras with exact closed forms ----------------
    BcG = cover_B(kG, edges, rev)
    ok, worst = poly_eq(
        lambda u: la.det(np.eye(24) - u * BcG),
        lambda u: ((1 - u**2) ** 4 * (1 - u) * (1 - 2 * u) * (1 + u) * (1 + 2 * u)
                   * (1 - u + 2 * u**2) ** 3 * (1 + u + 2 * u**2) ** 3))
    gate("G4a closed form zeta_cover^-1 (spectrum +/-{3,-1^3})", ok,
         f"worst rel err {worst:.2e}")
    # cover adjacency spectrum = spec(A) U spec(-A)
    Acov = np.block([[np.zeros((4, 4)), bloch_A(kG, bonds)],
                     [bloch_A(kG, bonds).conj().T, np.zeros((4, 4))]])
    sc = np.sort(la.eigvalsh(Acov))
    expect = np.sort(np.array([-3, -1, -1, -1, 1, 1, 1, 3], dtype=float))
    gate("G4b cover adjacency spectrum = +/- spec(K4)",
         np.allclose(sc, expect, atol=TOL), f"spec={np.round(sc, 6)}")

    # --- Report: where the mirror lives --------------------------------------
    print("\n  Structural notes (no gates):")
    AP = bloch_A(kP, bonds)
    sP = np.sort(np.real(la.eigvals(AP)))
    print(f"   - A(P) eigenvalues {np.round(sP, 6)}: symmetric multiset, so at P the")
    print("     base zeta-factor and the mirror L-factor carry IDENTICAL spectra --")
    print("     the mirror is spectrally invisible at P, maximally visible at Gamma")
    print("     (Perron root u=1/2 in zeta vs u=-1/2 in L: alternating-walk side).")
    print("   - Phase 1.3 target: M_persistence mass data as data of L(u, sgn).")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- zeta/L-function infrastructure established")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
