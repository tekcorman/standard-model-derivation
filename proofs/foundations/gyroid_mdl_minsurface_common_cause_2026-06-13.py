#!/usr/bin/env python3
"""
gyroid_mdl_minsurface_common_cause_2026-06-13.py
================================================
The "MDL => minimal-surface" conjecture, examined: are the three srs-selection
handles (Sunada strong-isotropy / MDL / the geometric L^3/V = 27/sqrt(2) extremum)
logically INDEPENDENT, or co-extremized by a common cause?

BACKGROUND.  `gyroid_mdl_minimal_surface_uniqueness_2026-06-13.py` (exploit #3)
showed srs saturates L^3/V = 27/sqrt(2) and billed it a "third INDEPENDENT handle".
The attractive open conjecture was "least description => least area" (MDL-min implies
the geometric minimum).  This probe tests the mechanism and, honestly, REFINES the
independence claim.

THE FINDING (and the link back to exploit #2).  srs vertices are 120-degree COPLANAR:
the three unit bond vectors at every vertex sum to ZERO (this is exactly why the
single-vertex chirality triple product vanished in `gyroid_mirror_vs_enantiomer_z2`).
Zero vertex force = the symmetric realization is a CRITICAL POINT of the total edge
length L at fixed cell volume -> L^3/V is *minimised* there.  And that same edge- and
vertex-transitivity is what zeroes srs's MDL coordinate- and edge-bits (dl_comparison:
coordinates=0, edges=0).  So all three handles are co-extremised by ONE property --
maximal local symmetry -- not three independent facts.

WHAT THIS PROBE DOES (native)
  A  vertex force balance: the 3 unit bonds at each vertex sum to ~0 (=> symmetric
     realization is a stationary point of L; links to exploit #2's coplanarity).
  B  perturbation test: random symmetry-BREAKING displacements of the basis (cell
     fixed) raise L^3/V above 27/sqrt(2), with a quadratic minimum at the symmetric
     point -> the geometric extremum IS the maximal-symmetry realization.
  C  MDL decomposition (dl_comparison): srs's winning bits (coordinates=0, edges=0)
     are exactly edge+vertex transitivity = the same symmetry; ths pays edges=2.
  D  VERDICT: the three handles share maximal symmetry as common cause -> they are
     different KINDS of functional that CONVERGE on srs (robustness), NOT causally
     independent evidence.  The "MDL => L^3/V" conjecture reduces to "the information-
     optimum and the geometry-optimum are both the maximally-symmetric net" -- true
     here via the common cause, but NOT a direct implication and NOT proved.  This
     honestly refines exploit #3's "independent" wording.  No graded content changes.
"""

import os
import sys
from math import sqrt

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, ATOMS, A_PRIM, NN_DIST, N_ATOMS  # noqa: E402
from proofs.foundations.dl_comparison import dl_srs, dl_ths  # noqa: E402

TOL = 1e-9
RNG = np.random.default_rng(20260613)
FAILURES = []
TARGET = 27.0 / sqrt(2.0)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def total_edge_length(atoms, bonds):
    """L = total UNDIRECTED edge length per primitive cell (bonds is directed -> /2)."""
    s = 0.0
    for i, j, c in bonds:
        rj = atoms[j] + c[0] * A_PRIM[0] + c[1] * A_PRIM[1] + c[2] * A_PRIM[2]
        s += la.norm(rj - atoms[i])
    return s / 2.0


def L3_over_V(atoms, bonds):
    return total_edge_length(atoms, bonds) ** 3 / abs(la.det(A_PRIM))


def main():
    print("=" * 86)
    print(" MDL <-> minimal-surface: common cause (maximal symmetry), not independence")
    print("=" * 86)
    bonds = find_bonds()

    # --- A: vertex force balance --------------------------------------------
    print("\n A  vertex force balance (3 unit bond vectors sum to 0 => L is stationary)")
    max_resid = 0.0
    for a in range(N_ATOMS):
        us = []
        for i, j, c in bonds:
            if i == a:
                rj = ATOMS[j] + c[0] * A_PRIM[0] + c[1] * A_PRIM[1] + c[2] * A_PRIM[2]
                v = rj - ATOMS[a]
                us.append(v / la.norm(v))
        resid = la.norm(np.sum(us, axis=0))
        max_resid = max(max_resid, resid)
    gate("A sum of 3 unit bonds = 0 at every vertex (120-deg coplanar; cf. exploit #2)",
         max_resid < 1e-9, f"max |sum of unit bonds| = {max_resid:.2e}")

    # --- B: perturbation test ------------------------------------------------
    print("\n B  symmetry-breaking perturbation raises L^3/V above 27/sqrt(2) (quadratic min)")
    base = L3_over_V(ATOMS, bonds)
    print(f"    symmetric srs:  L^3/V = {base:.10f}   (27/sqrt2 = {TARGET:.10f})")
    print(f"    {'displacement delta':>20} | {'mean L^3/V':>16} | {'min over trials':>16} | all >= 27/sqrt2?")
    print("    " + "-" * 78)
    all_above = True
    quad_ok = True
    prev = None
    for delta in (0.001, 0.005, 0.01, 0.02, 0.05):
        vals = []
        for _ in range(200):
            pert = ATOMS + delta * RNG.normal(size=ATOMS.shape)
            vals.append(L3_over_V(pert, bonds))
        vals = np.array(vals)
        above = bool(np.all(vals >= TARGET - 1e-9))
        all_above &= above
        excess = vals.mean() - TARGET
        print(f"    {delta:>20.3f} | {vals.mean():>16.8f} | {vals.min():>16.8f} | {above}")
        # quadratic: excess(delta)/delta^2 ~ const across the small deltas
        if prev is not None:
            ratio = (excess / delta**2) / prev
            quad_ok &= 0.3 < ratio < 3.0
        prev = excess / delta**2
    gate("B1 every symmetry-breaking perturbation has L^3/V >= 27/sqrt(2) (symmetric = the min)",
         all_above)
    gate("B2 the rise is quadratic in displacement (stationary minimum, not a saddle/edge)",
         quad_ok, "excess/delta^2 ~ const across small deltas")

    # --- C: MDL decomposition ------------------------------------------------
    print("\n C  MDL decomposition (dl_comparison): srs's winning bits = the same transitivity")
    tot_srs, b_srs = dl_srs()
    tot_ths, b_ths = dl_ths()
    print(f"    srs bits: {{{', '.join(f'{k}:{v:.2f}' for k,v in b_srs.items())}}}  total {tot_srs:.2f}")
    print(f"    ths bits: {{{', '.join(f'{k}:{v:.2f}' for k,v in b_ths.items())}}}  total {tot_ths:.2f}")
    win_coords = b_ths['coordinates'] - b_srs['coordinates']
    win_edges = b_ths['edges'] - b_srs['edges']
    gate("C srs coordinate-bits = edge-bits = 0 (Wyckoff 8a fixed + edge-transitive)",
         b_srs['coordinates'] == 0.0 and b_srs['edges'] == 0.0,
         f"vs ths edges={b_ths['edges']:.1f}; the same transitivity that gives A's force balance")
    print(f"    => the symmetry that zeroes srs's (coordinate,edge) MDL bits is the SAME edge+vertex")
    print(f"       transitivity that forces the 120-deg vertex (A) and hence the L^3/V minimum (B).")

    # --- D: verdict ----------------------------------------------------------
    print("\n" + "=" * 86)
    print(" VERDICT")
    print("=" * 86)
    print(f"""  The three srs-selection handles are CO-EXTREMISED BY ONE PROPERTY -- maximal local
  symmetry (edge + vertex transitivity, the I4_132 site group D_3):

    * Sunada strong-isotropy : symmetry, directly.
    * MDL minimum            : symmetry zeroes the coordinate- and edge-bits (C).
    * L^3/V = 27/sqrt(2)      : symmetry => 120-deg coplanar vertex (A) => the
                               symmetric realization is the minimum of L (B).

  So they are different KINDS of functional (symmetry / information / metric geometry)
  that CONVERGE on srs -- valuable as robustness (if one handle is contested, the
  others still point to srs) -- but they are NOT causally independent evidence: they
  share maximal symmetry as a common cause.

  The "MDL => minimal-surface" conjecture ("least description = least area") therefore
  reduces to: the information-optimal net and the geometry-optimal net are both the
  maximally-symmetric net.  That is TRUE for srs, via the common cause -- but it is NOT
  a direct implication and is NOT proved here (a general theorem would have to show the
  MDL-optimal net is always the symmetry-optimal net across all candidates).

  HONEST REFINEMENT of exploit #3: "third INDEPENDENT handle" overstated the logical
  status.  Accurate: a third, independently-PROVED extremal characterisation whose
  minimiser coincides with the other two THROUGH shared maximal symmetry.  No graded
  content changes.""")

    if FAILURES:
        print(f"\n RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print("\ngyroid_mdl_minsurface_common_cause_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
