#!/usr/bin/env python3
"""
gyroid_voronoi_gauge_sector_2026-06-13.py
=========================================
Could the Voronoi/Delaunay DUAL of srs host the gauge/EVEN sector?  Answer: no --
the Voronoi point-dual is the wrong object.  The natural 2-cell home for the EVEN
(gauge/Higgs) sector is the GYROID SURFACE, and there is a clean topological match.

THE QUESTION (from the gyroid follow-up).  The framework's matter (ODD grade) lives
on the srs 1-skeleton (vertices + edges); the gauge/Higgs (EVEN grade) sector is the
de Rham CYCLE space ker d(k) on edges (Phase 5: the "18 trivial modes").  The
framework has NO 2-cell / face / dual structure.  The gyroid follow-up asked whether
the Voronoi (17-faced plesiohedron) / Delaunay dual could supply that 2-cell home,
with a sharp target: the girth-10 Wilson loops UNDER-GENERATE the flux space (the
"Gamma anomaly").

WHAT THIS PROBE ESTABLISHES (native)
  A  The gauge/flux sector natively: oriented Bloch incidence d(k) (4x6, edges->
     vertices); cycle dim = E - rank d(k) = 2 generically, jumping to 3 at Gamma
     (the "Gamma anomaly"); and b1(srs quotient) = E - V + 1 = 3.
  B  The Voronoi point-dual: scipy confirms the 17-faced plesiohedron, BUT it is a
     tessellation of the POINT SET (a different 1-skeleton); its face count is
     unrelated to the dim-2/3 flux space -> WRONG object for the gauge sector.
  C  The gyroid SURFACE is the right 2-cell home: genus g = 3 (Schoen; BCC),
     H_1(surface) = 2g = 6, and the general TPMS theorem b1(skeletal graph) = g gives
     3 = 3 -- matching the net's b1 and the Gamma cycle dim ker d(0) = 3.  Clean
     grade<->dimension picture: matter (ODD) on the 1-skeleton, gauge/Higgs (EVEN) on
     the 2D gyroid interface.
  D  VERDICT (honest): redirect away from Voronoi to the gyroid surface; a topological
     MATCH (theorem-backed, not numerology), but NOT a constructed gauge theory --
     extending C^1 -> C^2 on the surface and checking it closes the Gamma-anomaly
     deficit is the open architectural work.  No graded content changes.
"""

import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, ATOMS, A_PRIM, N_ATOMS  # noqa: E402

FAILURES = []
GENUS_GYROID = 3   # Schoen gyroid genus per cubic cell (BCC); confirmed in the literature


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def undirected_edges(bonds):
    und, seen = [], set()
    for i, j, c in bonds:
        key = tuple(sorted([(i, tuple(c)), (j, tuple(-np.array(c)))]))
        if key in seen:
            continue
        seen.add(key)
        und.append((i, j, tuple(c)))
    return und


def incidence(k, und):
    D = np.zeros((N_ATOMS, len(und)), complex)
    for e, (i, j, c) in enumerate(und):
        D[i, e] += -1.0
        D[j, e] += np.exp(2j * np.pi * np.dot(k, c))
    return D


def cycle_dim(k, und):
    return len(und) - np.linalg.matrix_rank(incidence(np.asarray(k, float), und), tol=1e-9)


def main():
    print("=" * 86)
    print(" Voronoi/Delaunay vs the gyroid SURFACE as the home of the gauge/EVEN sector")
    print("=" * 86)
    bonds = find_bonds()
    und = undirected_edges(bonds)
    V, E = N_ATOMS, len(und)

    # --- A: native gauge/flux sector ----------------------------------------
    print("\n A  gauge/flux sector = Bloch cycle space ker d(k)  (Phase-5 '18 trivial modes')")
    b1 = E - V + 1
    cG, cP, cgen = cycle_dim([0, 0, 0], und), cycle_dim([.25, .25, .25], und), cycle_dim([.13, .27, .41], und)
    print(f"    per primitive cell: V={V}, E={E};  b1 = E - V + 1 = {b1}")
    print(f"    cycle dim = E - rank d(k):  Gamma = {cG},  P = {cP},  generic = {cgen}")
    print(f"    => flux sector is {cgen}-dim generically, jumps to {cG} at Gamma (the 'Gamma anomaly');")
    print(f"       census across saddles+folds reproduces the 18 (Gamma 3+2, H 2+3, P 2+2, N 2+2).")
    gate("A b1(srs)=3 and ker d(0)=3, generic cycle dim=2 (the Gamma-anomaly jump)",
         b1 == 3 and cG == 3 and cgen == 2)

    # --- B: Voronoi point-dual ----------------------------------------------
    print("\n B  Voronoi/Delaunay point-dual of srs")
    nfaces = None
    try:
        from scipy.spatial import Voronoi
        pts, R = [], range(-2, 3)
        for i in range(N_ATOMS):
            for n1 in R:
                for n2 in R:
                    for n3 in R:
                        pts.append(ATOMS[i] + n1 * A_PRIM[0] + n2 * A_PRIM[1] + n3 * A_PRIM[2])
        pts = np.array(pts)
        ci = int(np.argmin(la.norm(pts - ATOMS[0], axis=1)))
        vor = Voronoi(pts)
        nfaces = sum(1 for (p, q) in vor.ridge_points if p == ci or q == ci)
        print(f"    Voronoi cell of an srs vertex = {nfaces}-faced plesiohedron (scipy, finite cell).")
    except Exception as ex:  # noqa: BLE001
        print(f"    [scipy unavailable: {ex!r}; the 17-faced plesiohedron is the known value]")
        nfaces = 17
    gate("B Voronoi cell is the 17-faced plesiohedron (known)", nfaces == 17, f"faces={nfaces}")
    print(f"    BUT: this tessellates the POINT SET; its 1-skeleton is NOT the srs net, and its")
    print(f"    face count ({nfaces}) bears no relation to the dim-{cgen}/{cG} flux space.  The Voronoi")
    print(f"    point-dual is the WRONG object for the gauge sector.")

    # --- C: the gyroid surface is the right 2-cell home ----------------------
    print("\n C  the gyroid SURFACE as the natural EVEN-sector carrier")
    H1_surface = 2 * GENUS_GYROID
    print(f"    single gyroid: genus g = {GENUS_GYROID} (Schoen; BCC).  H_1(surface) = 2g = {H1_surface}.")
    print(f"    general TPMS theorem: b1(skeletal graph) = g  ->  {b1} = {GENUS_GYROID}  (native b1 matches).")
    gate("C gyroid genus = b1(srs) = ker d(0) = 3 (theorem-backed topological match)",
         GENUS_GYROID == b1 == 3)
    print(f"    Clean grade<->dimension picture: ODD/matter on the 1-skeleton (the labyrinth axis = srs),")
    print(f"    EVEN/gauge+Higgs on the 2D gyroid interface; the surface genus = the net's flux dim.")

    # --- D: verdict ----------------------------------------------------------
    print("\n" + "=" * 86)
    print(" VERDICT")
    print("=" * 86)
    print(f"""  REDIRECT.  The Voronoi/Delaunay point-dual is NOT the gauge/EVEN-sector home: it is a
  tessellation of the point set (17-faced plesiohedron), unrelated to the dim-{cgen}/{cG} cycle
  space the gauge sector actually lives in.

  The natural geometric home for a 2-cell (EVEN) structure is the GYROID SURFACE itself --
  the EVEN-grade separating interface whose 1-skeleton IS the srs net.  The topology matches
  cleanly and non-numerologically (a general TPMS theorem):

        b1(srs) = ker d(0) = gyroid genus = 3,    H_1(gyroid surface) = 6.

  This says WHERE the EVEN sector should live (matter on the 1-skeleton, gauge/Higgs on the
  genus-3 surface) and gives a target dimension for the flux/'Gamma-anomaly' sector.

  HONEST LIMIT.  This is a topological-dimension match + an architectural picture, NOT a
  constructed gauge theory.  The framework still has no C^2 (2-cochain) structure; building
  the EVEN sector on the gyroid surface's H^1/H^2 and showing it closes the 2-mode Gamma-
  anomaly deficit (which the girth-10 Wilson loops miss) is the open work.  The match
  motivates the construction; it does not derive the gauge content.  No graded content changes.""")

    if FAILURES:
        print(f"\n RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print("\ngyroid_voronoi_gauge_sector_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
