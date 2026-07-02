#!/usr/bin/env python3
"""
gyroid_surface_z2_ew_embedding_2026-06-13.py
============================================
THE LEAD (step 5): is there a Z2 that BOTH twists the complex AND sits in the EW group --
so the genus could genuinely drive electroweak breaking?

Step 4 showed a non-trivial Z2 bundle changes the gauge harmonic content (escape hatch
real), but used the body-centering MASS-mirror Z2 -- a spatial translation, not in the
gauge group.  This probe asks the decisive question: does the EW group itself contain a
Z2 that acts non-trivially over the complex, connecting genus to breaking?

KEY GROUP FACT.  SU(2) has a UNIQUE order-2 element: -I, the CENTER.  (Any g in SU(2)
with g^2=I has eigenvalues +-1 and det=1, forcing g=I or g=-I.)  The center acts:
  * on the ADJOINT (gauge bosons, spin 1) TRIVIALLY:  Ad(-I) X = (-I) X (-I)^-1 = X;
  * on the FUNDAMENTAL (Higgs doublet, left fermions) by -1.

CONSEQUENCE (computed below).
  C  Twisting the GAUGE (adjoint) bundle by the EW Z2 (= center) does NOTHING to the
     harmonic content -- it stays = genus.  So unlike the spatial body-centering twist of
     step 4, the EW-group Z2 does NOT change the gauge sector.  => genus does NOT drive
     GAUGE breaking via a Z2 bundle.  (No single Z2 is both non-trivial on the adjoint AND
     in the EW group: the center is in EW but blind to the adjoint; the body-centering
     hits the adjoint but is spatial, not in EW.)
  D  But the EW Z2 (center) acts by -1 on the HIGGS / MATTER DOUBLET.  A non-trivial Z2
     flux then twists the doublet bundle, and the constant (Perron) VEV mode is OBSTRUCTED
     (the signed/"frustrated" scalar Laplacian loses its zero mode: ker 1 -> 0).  So the
     Z2's genuine EW lever is on the FUNDAMENTAL (Higgs/matter), not the adjoint (gauge).

VERDICT: the lead's central frontier resolves.  The genus-3 surface topology does NOT
drive electroweak GAUGE breaking through a Z2 bundle (the EW Z2 is the SU(2) center,
which the adjoint cannot feel).  The real topology<->EW coupling runs through the
HIGGS/MATTER doublet (the fundamental), where a non-trivial Z2 flux can obstruct the
condensate.  This redirects the program from "genus -> gauge" to "Z2 flux -> the matter
1-skeleton (fundamental)" -- where the framework's fermions live.  VEV still posited;
this characterises the bundle structure, not a derivation. No graded content changes.
"""

import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- srs graph at Gamma (K4 quotient) ----------------------------------------
bonds = find_bonds()
SEEN, UB = set(), []
for (i, j, c) in bonds:
    if (j, i, tuple(-x for x in c)) in SEEN:
        continue
    SEEN.add((i, j, tuple(c)))
    UB.append((i, j, tuple(c)))
NU, NV = len(UB), 4


def signed_incidence(signs):
    """At Gamma: D[i,e]=-1, D[j,e]=signs[e].  signs in {+1,-1} = a Z2 flux on edges."""
    D = np.zeros((NV, NU))
    for e, (i, j, c) in enumerate(UB):
        D[i, e] += -1.0
        D[j, e] += signs[e]
    return D


def scalar_ker(signs):
    D = signed_incidence(signs)
    L = D @ D.T
    return int(np.sum(np.abs(la.eigvalsh(L)) < 1e-9))


def flux_through_triangles(signs):
    """Z2 holonomy (product of edge signs) around each 3-cycle of K4 -- non-trivial if any -1."""
    tris = []
    for e1, (i1, j1, _) in enumerate(UB):
        for e2, (i2, j2, _) in enumerate(UB):
            for e3, (i3, j3, _) in enumerate(UB):
                if e1 < e2 < e3 and len({i1, j1, i2, j2, i3, j3}) == 3:
                    tris.append(signs[e1] * signs[e2] * signs[e3])
    return tris


def main():
    print("=" * 90)
    print(" THE LEAD (step 5): does an EW-group Z2 twist the complex? (genus -> EW breaking?)")
    print("=" * 90)

    # --- A: Z2 flux sectors = H^1(complex; Z2) = genus ----------------------
    b1_z2 = NU - (NV - 1)        # GF(2) cycle rank of the connected K4 quotient = E - (V-1)
    print(f"\n A  Z2 flux sectors of the complex = H^1(K4 quotient; Z2), dim = b1 = {b1_z2} = genus")
    print(f"    => 2^{b1_z2} = {2**b1_z2} distinct Z2 flux sectors")
    gate("A Z2 flux sector count = 2^genus = 8", b1_z2 == 3)

    # --- B: the EW group's unique order-2 element = SU(2) center -I ----------
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    center = -I2
    # adjoint action by conjugation on the su(2) generators:
    adj_trivial = all(np.allclose(center @ S @ la.inv(center), S) for S in (sx, sy, sz))
    # fundamental action on a doublet:
    doublet = np.array([0.0, 1.0], complex)
    fund_sign = np.vdot(doublet, center @ doublet)   # = -1
    print("\n B  EW order-2 element = SU(2) center -I")
    print(f"    acts on ADJOINT (gauge bosons) trivially: Ad(-I)=id ? {adj_trivial}")
    print(f"    acts on FUNDAMENTAL (Higgs doublet) by {fund_sign.real:+.0f}")
    gate("B center -I is trivial on the adjoint and -1 on the doublet",
         adj_trivial and abs(fund_sign + 1) < 1e-9)

    # --- C: gauge (adjoint) bundle twisted by the EW Z2 = UNCHANGED ----------
    print("\n C  GAUGE (adjoint) bundle twisted by the EW Z2 (center): harmonic content")
    print(f"    the center acts trivially on the adjoint, so the adjoint twist is the IDENTITY:")
    print(f"    gauge harmonic = genus = 3 regardless of the EW-Z2 flux sector (UNCHANGED).")
    print(f"    contrast step 4: the spatial body-centering Z2 DID change it (3->0) -- but that")
    print(f"    Z2 is not in the gauge group.  No single Z2 is both in EW AND felt by the adjoint.")
    gate("C the EW-group Z2 leaves the gauge (adjoint) harmonic content = genus (no genus->gauge link)",
         adj_trivial)

    # --- D: Higgs/matter (doublet) bundle twisted by the EW Z2 = OBSTRUCTED --
    print("\n D  HIGGS/MATTER (doublet) bundle twisted by a non-trivial EW-Z2 flux")
    triv = [1] * NU                                   # trivial flux
    nontriv = [1] * NU; nontriv[0] = -1               # flip one edge -> unbalanced (non-trivial)
    k_triv, k_nt = scalar_ker(triv), scalar_ker(nontriv)
    holo_triv = flux_through_triangles(triv)
    holo_nt = flux_through_triangles(nontriv)
    print(f"    trivial flux   : triangle holonomies all +1 ({all(h > 0 for h in holo_triv)});  "
          f"doublet Perron zero modes = {k_triv}")
    print(f"    non-trivial flux: some triangle holonomy -1 ({any(h < 0 for h in holo_nt)});      "
          f"doublet Perron zero modes = {k_nt}")
    gate("D1 trivial Z2 flux: the doublet keeps its Perron VEV mode (ker=1)", k_triv == 1)
    gate("D2 non-trivial Z2 flux OBSTRUCTS the Higgs VEV: doublet Perron mode gone (ker=0)",
         k_nt == 0)
    print(f"    => the EW Z2's genuine lever is on the FUNDAMENTAL (Higgs/matter): a non-trivial")
    print(f"       flux frustrates the condensate.  (The adjoint, a center-singlet, never feels it.)")

    # --- E: verdict ----------------------------------------------------------
    print("\n" + "=" * 90)
    print(" VERDICT  (the lead, step 5 -- the central frontier resolves)")
    print("=" * 90)
    print("""  genus -> electroweak GAUGE breaking via a Z2 bundle: NO.  The EW group's only Z2 is the
  SU(2) center -I, which acts TRIVIALLY on the adjoint -- so it cannot change the gauge
  harmonic content (= genus).  The Z2 that DID change it (step 4) is the spatial
  body-centering mirror, which is not in the gauge group.  No single Z2 is both in EW and
  felt by the gauge sector, so the surface topology does not drive gauge breaking.

  The topology <-> EW coupling that DOES exist runs through the HIGGS/MATTER DOUBLET (the
  fundamental): the center acts by -1 there, so a non-trivial Z2 flux twists the doublet
  bundle and OBSTRUCTS the constant (Perron) VEV -- the signed scalar Laplacian loses its
  zero mode (ker 1 -> 0; a frustrated condensate).  So the right question is not "genus ->
  gauge" but "Z2 flux -> the matter 1-skeleton (fundamental)", where the framework's
  fermions live and the Higgs condenses.

  This redirects the lead cleanly: the surface gauge dynamics (steps 1-4) is sound but
  genus-blind to EW breaking; the live coupling is the Z2 flux acting on the fundamental
  (Higgs/matter), to be pursued on the matter 1-skeleton.  VEV still posited; bundle
  structure characterised, not derived.  No graded content changes.""")

    if FAILURES:
        print(f"\n RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print("\ngyroid_surface_z2_ew_embedding_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
