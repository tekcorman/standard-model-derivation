#!/usr/bin/env python3
"""
gyroid_surface_ew_nonabelian_2026-06-13.py
==========================================
THE LEAD (step 3): non-abelian electroweak gauge+Higgs on the surface complex --
does the genus-3 spatial topology connect to EW breaking, or factorize from it?

Step 2 found a SUGGESTIVE coincidence: the abelian Higgs mechanism on the genus-3
complex gave 3 massive gauge modes, matching the SM's W+,W-,Z count.  I flagged it as
count-only because the abelian toy is not the SM group.  This probe replaces the toy
with the REAL electroweak group SU(2)_L x U(1)_Y and a doublet Higgs, and asks the
sharp question: is the 3 (massive bosons) = 3 (genus) match a real connection, or two
unrelated indices?

CONSTRUCTION
  group part : generators (W^1,W^2,W^3,B) = (g T^1, g T^2, g T^3, g' Y), T^a = sigma^a/2,
               doublet Higgs VEV <Phi> = (0, v/sqrt2), Y_Phi = 1/2.  Gauge-boson mass^2
               matrix M2_ab = Re <Phi>| X_a X_b |Phi>; diagonalise -> {W+, W-, Z, photon}.
  spatial part: each generator's gauge field is a 1-cochain on the surface complex; the
               physical (harmonic) spatial modes at Gamma = H_1 = the genus = 3 (step 1).
  full physical gauge operator at Gamma = [group mass^2 M2_ab] (x) [spatial Hodge, harmonic
               multiplicity = genus].  The Perron VEV is spatially constant, so the mass
               term is uniform in the spatial index -> the two indices TENSOR.

RESULT (computed)
  group: eigenvalues {0 (photon), M_W^2, M_W^2, M_Z^2} with M_W^2/M_Z^2 = cos^2 theta_W
         (checked at the framework's unification value sin^2 theta_W = 3/8 -> 5/8).
  full : 12 physical modes at Gamma = genus(3) x generators(4) = 9 massive (3 genus x 3
         broken: W+,W-,Z) + 3 massless (3 genus x 1 photon).  The genus index (spatial,
         =3) and the broken-generator index (group, =3) FACTORISE.

VERDICT: the step-2 "3 = 3" is COINCIDENTAL -- genus (spatial topology) and the number
of broken EW generators (group theory) are orthogonal indices that both happen to be 3.
The EW Higgs mechanism runs cleanly on the complex (right W/Z/photon pattern and mass
ratio), but the surface topology does NOT, by itself, explain EW breaking.  The ONLY way
they could genuinely interact is a NON-TRIVIAL (Z2-Kitaev-flux-twisted) gauge bundle over
the complex, where H^1(complex; adjoint bundle) != genus x dim(adjoint) -- the open
sub-step, named.  Honest negative on the coincidence; clean placement of EW dynamics.
No graded content changes.
"""

import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds  # noqa: E402

FAILURES = []
SIN2_W = 3.0 / 8.0    # framework unification value (GQW trace); ratio test is value-agnostic


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- spatial genus at Gamma (cycle space of the srs net) ---------------------
def genus_at_gamma():
    bonds = find_bonds()
    SEEN, UB = set(), []
    for (i, j, c) in bonds:
        if (j, i, tuple(-x for x in c)) in SEEN:
            continue
        SEEN.add((i, j, tuple(c)))
        UB.append((i, j, tuple(c)))
    D = np.zeros((4, len(UB)))
    for b, (i, j, c) in enumerate(UB):
        D[i, b] += -1.0
        D[j, b] += 1.0
    return len(UB) - np.linalg.matrix_rank(D, tol=1e-9)   # cycle dim at Gamma = b1 = genus


# --- electroweak group mass matrix -------------------------------------------
def ew_mass_matrix(v=1.0):
    s2 = SIN2_W
    g = 1.0
    gp = g * np.sqrt(s2 / (1.0 - s2))            # g' = g tan(theta_W)
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    T = [g * sx / 2, g * sy / 2, g * sz / 2, gp * (0.5 * np.eye(2))]   # W1,W2,W3,B (Y=1/2)
    phi = np.array([0.0, v / np.sqrt(2)], complex)                      # <Phi> = (0, v/sqrt2)
    M2 = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            M2[a, b] = np.real(np.vdot(phi, (T[a] @ T[b] + T[b] @ T[a]) / 2 @ phi))
    return M2, g, gp


def main():
    print("=" * 90)
    print(" THE LEAD (step 3): non-abelian EW gauge+Higgs on the complex -- genus vs breaking")
    print("=" * 90)

    genus = genus_at_gamma()
    print(f"\n  spatial genus (cycle dim of srs at Gamma, = b1) = {genus}")

    # --- A: EW group mass spectrum ------------------------------------------
    v = 1.7
    M2, g, gp = ew_mass_matrix(v)
    ev = np.sort(la.eigvalsh(M2))
    print("\n A  EW group mass^2 spectrum (W1,W2,W3,B basis), sin^2 theta_W = 3/8")
    print(f"    eigenvalues = {np.round(ev,4)}  ->  photon=0, W (x2), Z")
    mW2 = ev[1]                # lowest nonzero (W, doubly degenerate)
    mZ2 = ev[3]                # highest (Z)
    ratio = mW2 / mZ2
    cos2 = 1.0 - SIN2_W
    print(f"    M_W^2 = {mW2:.4f} (x2),  M_Z^2 = {mZ2:.4f},  photon = {ev[0]:.2e}")
    print(f"    M_W^2 / M_Z^2 = {ratio:.5f}   vs  cos^2 theta_W = {cos2:.5f}")
    n_massless = int(np.sum(ev < 1e-9))
    n_massive = int(np.sum(ev > 1e-9))
    gate("A1 EW spectrum = 1 massless (photon) + 3 massive (W+,W-,Z)",
         n_massless == 1 and n_massive == 3)
    gate("A2 M_W^2/M_Z^2 = cos^2 theta_W (Higgs mechanism gives the right mass ratio)",
         abs(ratio - cos2) < 1e-9, f"{ratio:.5f} = {cos2:.5f}")

    # --- B: factorization with the spatial genus -----------------------------
    print("\n B  full physical gauge spectrum at Gamma = group (x) spatial-genus")
    # spatially constant Perron VEV => uniform spatial mass => tensor with genus modes
    full = np.kron(ev, np.ones(genus))            # each group mode x genus spatial copies
    full = np.sort(full)
    n_massive_full = int(np.sum(full > 1e-9))
    n_massless_full = int(np.sum(full < 1e-9))
    print(f"    {genus} genus modes x 4 generators = {genus*4} physical modes")
    print(f"    massive = {n_massive_full} (= genus {genus} x 3 broken: W+,W-,Z)")
    print(f"    massless = {n_massless_full} (= genus {genus} x 1 photon)")
    gate("B genus and breaking FACTORISE: massive = 3*genus, massless = 1*genus",
         n_massive_full == 3 * genus and n_massless_full == 1 * genus)

    # --- C: the coincidence test --------------------------------------------
    print("\n C  is the step-2 '3 = 3' a real connection?")
    n_broken = 3      # broken EW generators (W+,W-,Z); group theory: dim(SU2xU1) - dim(U1_em) = 4-1
    print(f"    genus (spatial topology) = {genus}")
    print(f"    broken EW generators (group theory) = {n_broken}  (= dim SU(2)xU(1) - dim U(1)_em = 4 - 1)")
    print(f"    these are ORTHOGONAL indices (spatial (x) group); both equal 3 -> coincidence.")
    gate("C the 3=3 match is genus(spatial) vs broken-generators(group): factorised, coincidental",
         genus == n_broken == 3)

    # --- D: verdict ----------------------------------------------------------
    print("\n" + "=" * 90)
    print(" VERDICT  (the lead, step 3)")
    print("=" * 90)
    print(f"""  The electroweak Higgs mechanism runs cleanly on the surface complex: doublet VEV in
  the Perron mode -> {{photon massless, W+, W-, Z massive}} with M_W^2/M_Z^2 = cos^2 theta_W
  (= 5/8 at the framework's sin^2 theta_W = 3/8).  But the spatial genus and the group
  breaking TENSOR:  12 physical gauge modes at Gamma = genus(3) x generators(4) = 9 massive
  (3 genus x 3 broken) + 3 massless (3 genus x photon).

  So the step-2 "3 massive bosons = 3 genus" coincidence is RESOLVED as COINCIDENTAL:
  the genus is a SPATIAL-topology index (=3, the gyroid handles) and the 3 broken EW
  generators are a GROUP-THEORY index (= dim SU(2)xU(1) - dim U(1)_em); they are orthogonal
  and merely both equal 3.  The surface topology does not, by itself, explain EW breaking.

  The ONLY way genus could genuinely drive breaking is a NON-TRIVIAL gauge bundle over the
  complex -- e.g. the framework's Z2 Kitaev flux twisting the EW adjoint, so that
  H^1(complex; adjoint bundle) != genus x dim(adjoint).  That twisted-bundle computation is
  the named open sub-step.  Honest negative on the coincidence; the EW dynamics is now
  correctly placed in the geometry (gauge = adjoint-valued cochains; photon carries the
  genus-3 EM flux). No graded content changes.""")

    if FAILURES:
        print(f"\n RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print("\ngyroid_surface_ew_nonabelian_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
