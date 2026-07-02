#!/usr/bin/env python3
"""
gyroid_surface_c2_flux_homology_2026-06-13.py
=============================================
Thread A (the C^1 -> C^2 construction): build the 2-cochain (face) structure the
framework's gauge sector lacks, and resolve the "Gamma anomaly" / flux deficit.

SETUP.  Phase 5.3/B2 placed the gauge/EVEN sector as the Bloch cycle space
Z_1(k) = ker d_1(k) (oriented incidence, edges->vertices), and found:
  * the 6 girth-10 rings (shortest Wilson loops) SPAN Z_1(k) at H, P, N, generic k
    (cycle dim 2), but
  * at Gamma every oriented ring chain VANISHES while dim ker d_1(0) = 3
    -- the "Gamma anomaly", a flux deficit the minimal Wilson loops miss.
The earlier follow-up (gyroid_voronoi_gauge_sector) argued the natural 2-cell home
is the GYROID SURFACE (genus 3 = b_1(srs)).  This probe makes that concrete: promote
the 10-rings to 2-cells, build the chain complex C_2 -> C_1 -> C_0, compute H_1, and
identify what the anomaly IS.

CONSTRUCTION.  d_2(k): C_2 (6 rings) -> C_1 (6 edges), columns = the ring boundary
1-chains ring_chain(rep, k).  Each is a cycle, so d_1 d_2 = 0 (verified).  Then
  H_1(k) = dim ker d_1(k) - rank d_2(k) = cycle_dim(k) - (#rings the boundaries span).

RESULT (computed below).  H_1(k) = 0 at H, P, N, generic (the 10-ring plaquettes cap
ALL local flux), and H_1(Gamma) = 3 = b_1(K4 quotient) = E - V + 1 = the gyroid genus.
So the anomaly is NOT a gap to be closed: it is genuine first homology of the maximal
abelian cover = the genus.  The gauge/flux sector splits cleanly as

    [ local plaquette flux: 6 girth-10 Wilson loops, fully captured, H_1=0 generic ]
  (+)
    [ 3 TOPOLOGICAL fluxes = the gyroid genus = the Gamma residual ]

matching the gyroid SURFACE H_1 = 2*genus = 6 = 3 "longitudes" (the net's b_1, the
topological fluxes) + 3 "meridians" (capped by the local plaquettes).  No local 2-cell
can cap the 3 topological fluxes -- they are the irreducible gauge content.

HONEST SCOPE.  This characterises the flux TOPOLOGY and resolves the anomaly's
identity; it does NOT build the full gauge+Higgs dynamics on the surface (matter
coupling, the C^2 inner product, Higgs as a 2-cochain) -- that remains the open
construction.  The framework's GF(2)-torus deficit was 2; the C-fiber residual at
Gamma is 3 (the extra mode is the zero-phase global/Perron-linked cycle); both are
reported.  No graded content changes.
"""

import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, A_PRIM  # noqa: E402

FAILURES = []
RNG = np.random.default_rng(20260613)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- srs combinatorics (reuse the Phase-5.3/B2 conventions) ------------------
bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
NE = len(EDGES)
E_INDEX = {e: a for a, e in enumerate(EDGES)}
REV = {a: E_INDEX[(j, i, tuple(-x for x in c))] for a, (i, j, c) in enumerate(EDGES)}
SEEN, UBONDS = set(), []
for (i, j, c) in EDGES:
    if (j, i, tuple(-x for x in c)) in SEEN:
        continue
    SEEN.add((i, j, tuple(c)))
    UBONDS.append((i, j, tuple(c)))
NU = len(UBONDS)                              # 6 undirected edges / cell
NV = 4
FOLLOW = {a: [b for b in range(NE) if EDGES[b][0] == EDGES[a][1] and b != REV[a]]
          for a in range(NE)}


def d1(k):
    """C_1 -> C_0 oriented Bloch incidence (4 x 6). ker = cycle space."""
    D = np.zeros((NV, NU), complex)
    for b, (i, j, c) in enumerate(UBONDS):
        D[i, b] += -1.0
        D[j, b] += np.exp(2j * np.pi * np.dot(k, np.asarray(c, float)))
    return D


def cycle_dim(k):
    return NU - np.linalg.matrix_rank(d1(np.asarray(k, float)), tol=1e-9)


def closed_nb_walks(length):
    walks = []

    def step(path, a, d):
        ca = EDGES[a][2]
        d2 = (d[0] + ca[0], d[1] + ca[1], d[2] + ca[2])
        if len(path) == length:
            if path[0][0] in FOLLOW[a] and d2 == (0, 0, 0):
                walks.append(list(path))
            return
        for b in FOLLOW[a]:
            step(path + [(b, d2)], b, d2)
    for a0 in range(NE):
        step([(a0, (0, 0, 0))], a0, (0, 0, 0))
    return walks


def cycle_class(walk):
    inst = set()
    for a, d in walk:
        i, j, c = EDGES[a]
        ra = REV[a]
        d2 = (d[0] + c[0], d[1] + c[1], d[2] + c[2])
        inst.add(min((a, d), (ra, d2)))
    base = min(d for _, d in inst)
    return frozenset((a, (d[0] - base[0], d[1] - base[1], d[2] - base[2])) for a, d in inst)


def ring_chain(walk, k):
    """The ring's boundary 1-chain in C_1 (dim 6) at Bloch k = d_2(ring)."""
    v = np.zeros(NU, complex)
    for a, d in walk:
        i, j, c = EDGES[a]
        if (i, j, tuple(c)) in SEEN:
            v[UBONDS.index((i, j, tuple(c)))] += np.exp(2j * np.pi * np.dot(k, np.asarray(d, float)))
        else:
            i2, j2, c2 = EDGES[REV[a]]
            d2 = (d[0] + c[0], d[1] + c[1], d[2] + c[2])
            v[UBONDS.index((i2, j2, tuple(c2)))] -= np.exp(2j * np.pi * np.dot(k, np.asarray(d2, float)))
    return v


def d2(reps, k):
    """C_2 (rings) -> C_1 (edges); columns = ring boundary chains. (6 x 6)"""
    return np.array([ring_chain(w, np.asarray(k, float)) for w in reps]).T


def main():
    print("=" * 88)
    print(" THREAD A: C^2 (gyroid-surface) flux homology -- resolving the Gamma anomaly")
    print("=" * 88)

    walks = closed_nb_walks(10)
    classes = {cycle_class(w): w for w in walks}
    reps = list(classes.values())
    gate("setup: 120 closed NB 10-walks -> 6 ring 2-cells (10 edges each)",
         len(walks) == 120 and len(reps) == 6 and all(len(c) == 10 for c in classes))

    DELTA = np.array([0.5, 0.5, -0.5])
    SADDLES = {"Gamma": np.zeros(3), "H": DELTA, "P": np.array([.25, .25, .25]),
               "N": A_PRIM @ np.array([0.0, .5, .5]), "generic": RNG.uniform(-.5, .5, 3)}

    # --- A: valid chain complex (d1 d2 = 0) ---------------------------------
    print("\n A  chain complex C_2 -> C_1 -> C_0 :  d_1 d_2 = 0  (ring boundaries are cycles)")
    worst = max(la.norm(d1(k) @ d2(reps, k)) for k in SADDLES.values())
    gate("A d_1 d_2 = 0 at all saddles + generic (valid 2-complex)", worst < 1e-9,
         f"max |d1 d2| = {worst:.1e}")

    # --- B: H_1 = cycle_dim - rank d_2 --------------------------------------
    print("\n B  H_1(k) = dim ker d_1(k) - rank d_2(k)   (flux not captured by the 6 plaquettes)")
    print(f"    {'k':>8} | {'cycle dim':>9} | {'rank d_2':>8} | {'H_1':>4}")
    print("    " + "-" * 40)
    H1 = {}
    for nm, k in SADDLES.items():
        cd = cycle_dim(k)
        r2 = np.linalg.matrix_rank(d2(reps, k), tol=1e-9)
        H1[nm] = cd - r2
        print(f"    {nm:>8} | {cd:>9} | {r2:>8} | {H1[nm]:>4}")
    gate("B1 the 6 girth-10 plaquettes CAP all local flux off Gamma (H_1 = 0 at H,P,N,generic)",
         all(H1[nm] == 0 for nm in ("H", "P", "N", "generic")))
    gate("B2 GAMMA RESIDUAL H_1 = 3 (the plaquettes cap none of ker d_1(0) = 3)",
         H1["Gamma"] == 3)

    # --- C: identify the residual -------------------------------------------
    print("\n C  what the Gamma residual IS")
    b1_quotient = NU - NV + 1                      # E - V + 1 for the K4 quotient
    GENUS = 3
    print(f"    b_1(K4 quotient) = E - V + 1 = {NU} - {NV} + 1 = {b1_quotient}")
    print(f"    gyroid genus g = {GENUS};  H_1(gyroid surface) = 2g = {2*GENUS}")
    gate("C Gamma residual = b_1(quotient) = gyroid genus = 3 (irreducible topological flux)",
         H1["Gamma"] == b1_quotient == GENUS == 3)
    print(f"    => the 3 residual modes are the cycles of the K4 quotient -- closed only at zero")
    print(f"       Bloch phase (nonzero net offset in the cover, why srs girth is 10 not 3); no")
    print(f"       LOCAL (contractible) plaquette can cap them.  They are the genus-3 topological flux.")

    # --- D: the split + surface match ---------------------------------------
    print("\n" + "=" * 88)
    print(" VERDICT  (Thread A: resolved)")
    print("=" * 88)
    print(f"""  The gauge/EVEN flux sector splits cleanly, exactly as the gyroid surface predicts:

     gauge flux  =  [ LOCAL plaquette flux ]              (+)  [ TOPOLOGICAL flux ]
                    6 girth-10 Wilson loops                    3 = gyroid genus = b_1
                    cap all of ker d_1 off Gamma (H_1=0)       the Gamma residual H_1=3

  and on the gyroid SURFACE (H_1 = 2g = 6):  6 = 3 "longitudes" (= the net's b_1, the
  topological fluxes that survive as the Gamma residual) + 3 "meridians" (bound disks in
  the labyrinth = the local plaquettes).  So the "Gamma anomaly" is NOT an accounting gap
  to be closed by adding faces -- it is GENUINE first homology, equal to the gyroid genus,
  the irreducible topological gauge content.  Adding the 10-ring 2-cells caps precisely the
  contractible flux and leaves exactly the genus, as it must.

  HONEST SCOPE.  This resolves the anomaly's IDENTITY (= genus) and builds the flux
  2-complex; it does NOT yet build the gauge+Higgs DYNAMICS on the surface (matter
  coupling, a C^2 Higgs cochain, the inner product) -- that is the remaining construction.
  Framework's GF(2)-torus deficit was 2; the C-fiber residual at Gamma is 3 (the extra is
  the zero-phase global/Perron-linked cycle). No graded content changes.""")

    if FAILURES:
        print(f"\n RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print("\ngyroid_surface_c2_flux_homology_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
