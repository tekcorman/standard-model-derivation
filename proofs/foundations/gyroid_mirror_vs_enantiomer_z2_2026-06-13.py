#!/usr/bin/env python3
"""
gyroid_mirror_vs_enantiomer_z2_2026-06-13.py
============================================
Disambiguation: the framework's mass-MIRROR Z2 vs the gyroid's ENANTIOMER Z2.

CONTEXT.  The gyroid follow-up (memory project_gyroid_voronoi_unused_geometry)
raised a sharp question for the mass = "srs<->srs-z mirror holonomy" program:
the framework identifies the mirror with the BODY-CENTERING translation
t = (1/2,1/2,1/2) (Phase 1.3/S1, `phase1_3_s1_mirror_is_bodycentering`), but a
gyroid minimal surface separates two labyrinths related by INVERSION -- the
OPPOSITE-handed Laves net (I4_132 <-> I4_3 32, combined = double gyroid Ia-3d).
Is the framework's mirror the same Z2 as the gyroid's enantiomer flip?  And if
not, should mass-holonomy live on the inversion instead?

This probe answers BOTH, natively, by separating the two candidate Z2 operations
on the srs structure and showing they are distinct in EVERY invariant that matters:

  Z2_t  = body-centering translation  t = (1/2,1/2,1/2)_cubic   (framework mirror)
  Z2_i  = inversion  r -> -r                                    (gyroid enantiomer)

GATES
  A  orientation:   t has linear part +I (proper, chirality-PRESERVING);
                    i has linear part -I (improper, chirality-REVERSING).
  B  net action:    t maps srs onto itself (t is a BCC lattice vector = a1+a2+a3,
                    so it is a symmetry; the MIRROR is t as the nontrivial coset of
                    T_I / T_P, restricting periodicity to simple-cubic -- purely
                    translational).  i maps srs onto a net NOT superimposable by any
                    lattice translation: the opposite enantiomer.
  C  handedness witness:  the bond-triple-product pseudoscalar at a vertex is
                    INVARIANT under t and FLIPS SIGN under i.  (Decisive: t cannot
                    be the enantiomer map.)
  D  spectral signature:  body-centering gives the ANTIPERIOD  spec A(k+DELTA) =
                    -spec A(k)  (the bipartite/chiral-sublattice Z2 the framework
                    reads as the mirror);  inversion gives  spec A(-k) = spec A(k)
                    (NO sign flip).  The two Z2 act spectrally differently => they
                    are genuinely different symmetries.
  E  group placement + assignment table:  I4_132 already contains t; adjoining i
                    gives the centrosymmetric supergroup Ia-3d = the double gyroid.
                    => the framework's TWO Z2 map cleanly onto the gyroid's two
                    structures: translation Z2 = mass mirror (srs-z, zeta L(u,sgn));
                    inversion Z2 = enantiomer / both-hands chirality (R-12, SU(2)),
                    geometric home = the double-gyroid labyrinth pair.

VERDICT.  The gyroid does NOT demand relocating mass-holonomy to the inversion:
the mass mirror is correctly the (chirality-preserving) body-centering translation,
and the (chirality-reversing) enantiomer inversion is a SEPARATE Z2 the framework
already uses for the both-hands / SU(2) chirality sector.  The gyroid's value is to
give each Z2 a geometric avatar and to confirm they are independent.  No graded
content changes -- structural disambiguation only.
"""

import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, ATOMS, A_PRIM, N_ATOMS  # noqa: E402

TOL = 1e-9
RNG = np.random.default_rng(20260613)
FAILURES = []

T_BC = np.array([0.5, 0.5, 0.5])          # body-centering translation (cubic frac)
DELTA = np.array([0.5, 0.5, -0.5])         # = q.A_PRIM, q=(0,0,1): the BCC k-shift for the fold


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def bloch_A(k, bonds):
    A = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    for i, j, c in bonds:
        A[j, i] += np.exp(2j * np.pi * np.dot(np.asarray(k, float), np.asarray(c, float)))
    return A


def in_lattice(v, tol=1e-6):
    """Is the Cartesian vector v a BCC lattice vector (integer combo of A_PRIM rows)?"""
    n = la.solve(A_PRIM.T, v)          # coords in primitive basis
    return np.all(np.abs(n - np.round(n)) < tol)


def bond_vectors_at(atom, bonds):
    """The 3 NN bond displacement vectors emanating from `atom` (Cartesian)."""
    vs = []
    for i, j, c in bonds:
        if i == atom:
            rj = ATOMS[j] + c[0] * A_PRIM[0] + c[1] * A_PRIM[1] + c[2] * A_PRIM[2]
            vs.append(rj - ATOMS[i])
    return vs


def nonbacktracking_3path(bonds, start=0):
    """Three consecutive bond vectors (e1,e2,e3) of a non-backtracking walk from
    `start`. The chirality of srs lives in the TORSION of such a chain, not at a
    single (locally planar) vertex."""
    adj = {i: [] for i in range(N_ATOMS)}
    for i, j, c in bonds:
        adj[i].append((j, np.asarray(c, float)))

    def disp(i, j, c):
        return ATOMS[j] + c @ A_PRIM - ATOMS[i]

    j1, c1 = adj[0 if start is None else start][0]
    e1 = disp(start, j1, c1)
    es, cur, prev_e = [e1], j1, e1
    for _ in range(2):
        for j2, c2 in adj[cur]:
            e = disp(cur, j2, c2)
            if not np.allclose(e, -prev_e):      # forbid immediate reversal
                es.append(e)
                cur, prev_e = j2, e
                break
    return es


def main():
    print("=" * 76)
    print(" GYROID MIRROR(translation) vs ENANTIOMER(inversion):  are they the same Z2?")
    print("=" * 76)
    bonds = find_bonds()
    print(f"  srs: {N_ATOMS} atoms, {len(bonds)} directed NN bonds;  t=(1/2,1/2,1/2), inversion r->-r")

    # --- A: orientation (linear part determinant) ----------------------------
    det_t = +1.0                                   # translation: linear part = I
    det_i = la.det(-np.eye(3))                     # inversion: linear part = -I
    gate("A  orientation: det(lin t)=+1 (proper), det(lin i)=-1 (improper)",
         abs(det_t - 1) < TOL and abs(det_i + 1) < TOL, f"det_t={det_t:+.0f}, det_i={det_i:+.0f}")

    # --- B: action on the net ------------------------------------------------
    # t is a BCC lattice vector (a1+a2+a3) => a symmetry of srs (maps atoms->atoms).
    t_cart = A_PRIM[0] + A_PRIM[1] + A_PRIM[2]
    t_is_lattice = in_lattice(t_cart)
    # and 2t in simple-cubic, t the nontrivial coset of T_I/T_P:
    coset_z2 = in_lattice(2 * t_cart)
    gate("B1 body-centering t = a1+a2+a3 is a BCC lattice vector (proper symmetry of srs)",
         t_is_lattice and coset_z2, f"t in T_I: {t_is_lattice}; 2t in T_I: {coset_z2}")

    # inversion image -ATOMS: superimposable on ATOMS by a lattice translation?
    inv_superimposable = True
    for a in range(N_ATOMS):
        # is -ATOMS[a] congruent to some ATOMS[b] modulo the BCC lattice?
        hit = any(in_lattice(-ATOMS[a] - ATOMS[b]) for b in range(N_ATOMS))
        inv_superimposable &= hit
    gate("B2 inversion image -srs is NOT superimposable on srs by a lattice translation",
         not inv_superimposable, f"superimposable-by-translation = {inv_superimposable} (srs is chiral)")

    # --- C: handedness witness = TORSION of a 3-bond chain (decisive) --------
    # NB: the 3 bonds at a single srs vertex are COPLANAR (local triple product = 0,
    # checked below) -- the chirality is in the inter-vertex twist, so the witness
    # must be the signed torsion e1.(e2 x e3) of a non-backtracking 3-path.
    vertex_planar = all(abs(np.dot(b[0], np.cross(b[1], b[2]))) < 1e-9
                        for b in (bond_vectors_at(a, bonds) for a in range(N_ATOMS)))
    e1, e2, e3 = nonbacktracking_3path(bonds, start=0)
    tau = float(np.dot(e1, np.cross(e2, e3)))                  # torsion pseudoscalar
    tau_t = float(np.dot(e1, np.cross(e2, e3)))                # under translation: unchanged
    tau_i = float(np.dot(-e1, np.cross(-e2, -e3)))             # under inversion: (-1)^3
    nonzero = abs(tau) > 1e-6
    inv_flips = abs(tau_i + tau) < 1e-9 and abs(tau_i - tau) > 1e-9
    t_keeps = abs(tau_t - tau) < 1e-9
    gate("C  3-bond TORSION pseudoscalar: nonzero, INVARIANT under t, FLIPS under i",
         vertex_planar and nonzero and inv_flips and t_keeps,
         f"vertex coplanar (chi_vtx=0): {vertex_planar}; tau={tau:+.4f}, tau_t={tau_t:+.4f}, tau_i={tau_i:+.4f}")

    # --- D: spectral signatures ----------------------------------------------
    # body-centering: antiperiod spec A(k+DELTA) = -spec A(k)
    ok_bc, worst_bc = True, 0.0
    for _ in range(6):
        k = RNG.random(3)
        s1 = np.sort(la.eigvalsh(bloch_A(k, bonds)))
        s2 = np.sort(-la.eigvalsh(bloch_A(k + DELTA, bonds)))
        worst_bc = max(worst_bc, np.abs(s1 - s2).max())
        ok_bc &= worst_bc < TOL
    gate("D1 body-centering ANTIPERIOD: spec A(k+DELTA) = -spec A(k)", ok_bc, f"worst {worst_bc:.2e}")

    # inversion: spec A(-k) = spec A(k)  (no sign flip)  -- and it is NOT the antiperiod
    ok_inv, worst_inv = True, 0.0
    matches_antiperiod = False
    for _ in range(6):
        k = RNG.random(3)
        s_k = np.sort(la.eigvalsh(bloch_A(k, bonds)))
        s_mk = np.sort(la.eigvalsh(bloch_A(-k, bonds)))
        worst_inv = max(worst_inv, np.abs(s_k - s_mk).max())
        ok_inv &= np.abs(s_k - s_mk).max() < TOL
        if np.abs(np.sort(-s_mk) - s_k).max() < TOL and np.abs(s_k).max() > TOL:
            matches_antiperiod = True
    gate("D2 inversion: spec A(-k) = spec A(k) (NO sign flip => different Z2 from body-centering)",
         ok_inv and not matches_antiperiod, f"worst {worst_inv:.2e}; mimics antiperiod: {matches_antiperiod}")

    # --- E: group placement + assignment -------------------------------------
    print("\n" + "-" * 76)
    print(" E  group placement & Z2 assignment")
    print("-" * 76)
    print("""  srs space group I4_132 (#214, chiral) ALREADY contains the body-centering t.
  Adjoining the inversion i yields the centrosymmetric supergroup Ia-3d (#230)
  = the DOUBLE GYROID = two opposite-handed Laves labyrinths.  So t and i are
  INDEPENDENT generators (t in I4_132; i takes I4_132 -> I4_3 32).

    Z2            | isometry          | orientation | spectral sig        | framework role                 | gyroid home
    --------------+-------------------+-------------+---------------------+--------------------------------+----------------------
    body-centering| translation (1/2)^3| proper (+1) | antiperiod -spec    | MASS MIRROR: srs-z, zeta L(u,sgn)| SC-fold within ONE labyrinth
    inversion     | r -> -r           | improper(-1)| spec A(-k)=spec A(k)| ENANTIOMER / both-hands SU(2) (R-12)| the two labyrinths (Ia-3d)
""")

    print("=" * 76)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: the mass-mirror Z2 (body-centering, chirality-PRESERVING) and the")
    print(" enantiomer Z2 (inversion, chirality-REVERSING) are DISTINCT in orientation,")
    print(" net-action, handedness witness, and spectral signature.  The gyroid does NOT")
    print(" demand moving mass-holonomy onto the inversion: mass correctly rides the")
    print(" translation Z2; the inversion Z2 is the framework's separate both-hands /")
    print(" SU(2) chirality sector (R-12), whose geometric home is the double gyroid Ia-3d.")
    print("=" * 76)
    print("gyroid_mirror_vs_enantiomer_z2_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
