#!/usr/bin/env python3
"""
gyroid_surface_twisted_bundle_2026-06-13.py
===========================================
THE LEAD (step 4): the escape hatch from step 3 -- does a NON-TRIVIAL (Z2-twisted)
gauge bundle make the spatial topology interact with the gauge structure?

Step 3 showed that with a TRIVIAL gauge bundle the genus (spatial, =3) and the EW
breaking (group) FACTORISE, so the "3 = 3" was coincidental.  It named one escape: a
non-trivial bundle, where H^1(complex; twisted) != genus.  The framework already carries
a natural Z2 bundle -- the body-centering "mirror" (Phase 1.3/S1), whose sign-twist is
equivalent to the momentum shift k -> k + Delta (the antiperiod
spec A(k+Delta) = -spec A(k); F4 of phase1_3_s1).  So the Z2-TWISTED harmonic content at
Gamma = the UNTWISTED content at Delta.

THIS PROBE computes the gauge harmonic content untwisted (at Gamma) vs Z2-twisted
(at Delta), at both levels -- the cycle space (no 2-cells) and H_1 (with the girth-10
2-cells of Thread A) -- and decides whether the twist changes it.

RESULT (computed)
  cycle space (ker d_1):   untwisted Gamma = 3 (= genus);  twisted = 2.
  H_1 (with ring 2-cells): untwisted Gamma = 3 (= genus);  twisted = 0.
So a non-trivial Z2 twist DOES change the gauge harmonic content (3 -> 2, or 3 -> 0 with
the surface 2-cells).  The step-3 factorisation is SPECIAL to the trivial bundle: a
non-trivial bundle makes the spatial topology and the gauge structure interact.

HONEST LIMIT (the real open question).  The Z2 used here is the body-centering MASS-mirror
(exploit #2), NOT shown to be the EW-breaking Z2 -- those are distinct Z2's (mass-mirror =
translation, chirality-preserving; EW breaking lives in the group).  So step 4 establishes
the MECHANISM (a non-trivial bundle breaks the genus<->breaking factorisation) but not the
physical identification.  Turning the step-2 coincidence into physics needs a Z2 bundle
that (a) is non-trivial over the complex AND (b) embeds in the EW gauge group -- the named
frontier.  No graded content changes.
"""

import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, A_PRIM  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- srs combinatorics + Thread-A 2-cells ------------------------------------
bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
NE = len(EDGES)
REV = {a: EDGES.index((j, i, tuple(-x for x in c))) for a, (i, j, c) in enumerate(EDGES)}
SEEN, UBONDS = set(), []
for (i, j, c) in EDGES:
    if (j, i, tuple(-x for x in c)) in SEEN:
        continue
    SEEN.add((i, j, tuple(c)))
    UBONDS.append((i, j, tuple(c)))
NU, NV = len(UBONDS), 4
FOLLOW = {a: [b for b in range(NE) if EDGES[b][0] == EDGES[a][1] and b != REV[a]] for a in range(NE)}
DELTA = np.array([0.5, 0.5, -0.5])     # body-centering Z2 = sign-twist (phase1_3_s1 F4)


def d1(k):
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


REPS = list({cycle_class(w): w for w in closed_nb_walks(10)}.values())


def H1(k):
    cd = cycle_dim(k)
    D2 = np.array([ring_chain(w, np.asarray(k, float)) for w in REPS]).T
    return cd - np.linalg.matrix_rank(D2, tol=1e-9)


def main():
    print("=" * 88)
    print(" THE LEAD (step 4): does a Z2-twisted bundle make genus interact with the gauge?")
    print("=" * 88)

    GAMMA = np.zeros(3)
    print("\n  Z2 twist = body-centering 'mirror' = sign-twist ~ momentum shift k -> k+Delta")
    print(f"  (phase1_3_s1 F4: shifted fiber == sign-twisted fiber).  Delta = {DELTA}")

    # --- A: cycle space (no 2-cells) ----------------------------------------
    cyc_unt, cyc_tw = cycle_dim(GAMMA), cycle_dim(GAMMA + DELTA)
    print("\n A  cycle space (ker d_1) harmonic content")
    print(f"    untwisted (Gamma)        = {cyc_unt}   (= genus = b1)")
    print(f"    Z2-twisted (Gamma+Delta) = {cyc_tw}")
    gate("A the twist CHANGES the cycle-space harmonic content (3 -> 2)",
         cyc_unt == 3 and cyc_tw == 2)

    # --- B: H_1 with the surface 2-cells ------------------------------------
    h1_unt, h1_tw = H1(GAMMA), H1(GAMMA + DELTA)
    print("\n B  H_1 with the girth-10 surface 2-cells (Thread A)")
    print(f"    untwisted (Gamma)        = {h1_unt}   (= genus, the topological gauge modes)")
    print(f"    Z2-twisted (Gamma+Delta) = {h1_tw}")
    gate("B the twist CHANGES the physical (H_1) gauge content (3 -> 0)",
         h1_unt == 3 and h1_tw == 0)

    # --- C: so factorization is bundle-trivial-specific ----------------------
    print("\n C  consequence for step 3")
    print(f"    trivial bundle  : harmonic = genus = {h1_unt}  -> factorises from breaking (step 3).")
    print(f"    Z2-twisted bundle: harmonic = {h1_tw}        -> genus content is REMOVED by the twist.")
    gate("C a non-trivial bundle breaks the genus<->breaking factorisation (harmonic != genus)",
         h1_tw != h1_unt)

    # --- D: verdict ----------------------------------------------------------
    print("\n" + "=" * 88)
    print(" VERDICT  (the lead, step 4)")
    print("=" * 88)
    print(f"""  YES -- a non-trivial Z2 bundle DOES make the spatial topology interact with the gauge
  structure.  Twisting by the framework's body-centering Z2 changes the gauge harmonic
  content from the genus (3 at Gamma) to {cyc_tw} (cycle space) / {h1_tw} (with the surface
  2-cells).  So step 3's clean genus<->breaking FACTORISATION is special to the TRIVIAL
  bundle; the escape hatch is real.

  HONEST LIMIT (the genuine open question).  The Z2 used here is the body-centering
  MASS-mirror (exploit #2: a translation, chirality-preserving) -- NOT the EW-breaking Z2
  (which lives in the gauge group).  They are distinct.  So step 4 establishes the
  MECHANISM (a non-trivial bundle breaks factorisation) but not the physical link to EW
  breaking.  Turning step 2's coincidence into physics needs a Z2 bundle that is BOTH
  non-trivial over the complex AND embedded in the EW gauge group -- the named frontier
  for any next step.  No graded content changes.""")

    if FAILURES:
        print(f"\n RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print("\ngyroid_surface_twisted_bundle_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
