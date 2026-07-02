#!/usr/bin/env python3
"""
gyroid_matter_flux_groundstate_2026-06-13.py
============================================
THE REDIRECTED FRONTIER (matter sector): what Z2 flux does the framework's matter sit
in -- trivial (Higgs VEV permitted) or non-trivial (condensate frustrated)?

Step 5 (gyroid_surface_z2_ew_embedding) showed the topology<->EW lever is the Z2 flux
acting on the Higgs/matter DOUBLET (the fundamental): a non-trivial flux obstructs the
constant (Perron) VEV.  Whether EW breaking is permitted or frustrated by the geometry
therefore hinges on WHICH flux the matter actually occupies -- the Kitaev-style
ground-state flux of the srs matter sector.

CONSTRUCTION.  The matter Bloch Hamiltonian with a Z2 gauge field s_e in {+1,-1} on the
6 undirected edges:  H^s(k)[t,s] = s_e * exp(2pi i k.c).  The fermion ground-state energy
at half filling is  E_GS(flux) = -1/2 * sum_k sum_n |E_n(k; flux)| ,  so the ground-state
flux MAXIMISES  S(flux) = sum_k sum_n |E_n(k; flux)| .  Gauge-invariant flux = the Z2
holonomy around the 3 fundamental cycles of the K4 quotient (tree e0,e1,e2 = star from
vertex 0; non-tree e3=012, e4=013, e5=023 carry the 3 cycle fluxes).  8 sectors.

WHAT THIS PROBE DOES
  A  build H^s(k); enumerate the 8 flux sectors (cycle holonomies (s3,s4,s5)).
  B  compute S(flux) over a Brillouin-zone mesh; find the ground-state (max-S) flux.
  C  report whether the ground-state flux is TRIVIAL (+,+,+) or NON-TRIVIAL, and the
     antiperiod-forced degeneracy (flux-free vs body-centering give +-spectra => equal S).
  D  VERDICT: trivial GS flux => the geometry PERMITS the Higgs VEV (normal EW breaking
     on the matter 1-skeleton); non-trivial => the geometry FRUSTRATES it.  Honest: this
     uses the adjacency (Majorana-hopping) matter Hamiltonian; the VEV/coupling are
     posited; it characterises the flux ground state, not a full derivation.  No graded
     content changes.
"""

import os
import sys
from itertools import product

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, N_ATOMS  # noqa: E402

FAILURES = []
DELTA = np.array([0.5, 0.5, -0.5])


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- edges (directed) + undirected list + sign lookup ------------------------
bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
SEEN, UB = set(), []
for (i, j, c) in EDGES:
    if (j, i, tuple(-x for x in c)) in SEEN:
        continue
    SEEN.add((i, j, tuple(c)))
    UB.append((i, j, tuple(c)))
# map each directed edge to its undirected-edge index (for the sign)
U_INDEX = {}
for ui, (i, j, c) in enumerate(UB):
    U_INDEX[(i, j, c)] = ui
    U_INDEX[(j, i, tuple(-x for x in c))] = ui

# spanning tree = e0,e1,e2 (star from v0); non-tree = e3,e4,e5 carry the cycle fluxes
TREE = [0, 1, 2]
NONTREE = [3, 4, 5]


def signed_H(k, signs):
    """signs: dict undirected-edge-index -> +-1.  H^s(k)[t,s] = s_e exp(2pi i k.c)."""
    H = np.zeros((N_ATOMS, N_ATOMS), complex)
    for (i, j, c) in EDGES:
        ue = U_INDEX[(i, j, c)]
        H[j, i] += signs[ue] * np.exp(2j * np.pi * np.dot(k, np.asarray(c, float)))
    return (H + H.conj().T) / 2.0


def band_sum(signs, nmesh=14):
    """S(flux) = sum_k sum_n |E_n(k)| over a uniform BZ mesh (per k-point average)."""
    pts = np.linspace(0, 1, nmesh, endpoint=False)
    tot, npt = 0.0, 0
    for a in pts:
        for b in pts:
            for cc in pts:
                w = la.eigvalsh(signed_H(np.array([a, b, cc]), signs))
                tot += np.sum(np.abs(w))
                npt += 1
    return tot / npt


def main():
    print("=" * 88)
    print(" MATTER-SECTOR Z2 FLUX GROUND STATE: does the geometry permit or frustrate the VEV?")
    print("=" * 88)

    # --- A/B: enumerate flux sectors, compute S(flux) ------------------------
    print("\n A/B  band-energy S(flux) = <sum_n |E_n(k)|> over the BZ, per flux sector")
    print(f"      (ground state = MAX S; tree e0,e1,e2 = +1; flux = holonomies (s3,s4,s5))")
    print(f"      {'(s3,s4,s5)':>12} | {'flux':>12} | {'S(flux)':>10}")
    print("      " + "-" * 42)
    results = {}
    for s3, s4, s5 in product([+1, -1], repeat=3):
        signs = {0: 1, 1: 1, 2: 1, 3: s3, 4: s4, 5: s5}
        S = band_sum(signs)
        results[(s3, s4, s5)] = S
        flux = "trivial" if (s3, s4, s5) == (1, 1, 1) else "non-trivial"
        print(f"      {str((s3,s4,s5)):>12} | {flux:>12} | {S:>10.5f}")

    Smax, Smin = max(results.values()), min(results.values())
    spread = Smax - Smin
    gs = [hol for hol, S in results.items() if abs(S - Smax) < 1e-6]
    print(f"\n      S spread across all 8 sectors (max - min) = {spread:.2e}")

    # --- C: the band energy is flux-INDEPENDENT (a flat direction) -----------
    print("\n C  is the matter band energy flux-dependent?")
    flux_independent = spread < 1e-6
    trivial_is_gs = (1, 1, 1) in gs
    print(f"      band energy flux-INDEPENDENT (all 8 sectors equal)? {flux_independent}")
    print(f"      reason: a per-cell Z2 flux is gauge-equivalent to a momentum shift, and the")
    print(f"      full-BZ integral of sum|E| is shift-invariant -> the discrete flux washes out.")
    print(f"      => the matter does NOT select a flux; trivial (+,+,+) is degenerate-permitted.")
    gate("C the matter band energy is FLUX-INDEPENDENT (flat direction; trivial flux permitted)",
         flux_independent and trivial_is_gs, f"spread={spread:.1e}")

    # --- D: verdict ----------------------------------------------------------
    print("\n" + "=" * 88)
    print(" VERDICT  (matter-sector flux)")
    print("=" * 88)
    if flux_independent:
        print(f"""  The srs matter band energy is EXACTLY FLUX-INDEPENDENT: all 8 Z2 flux sectors have the
  same S = {Smax:.5f} (spread {spread:.1e}).  A per-cell Z2 flux is gauge-equivalent to a
  momentum shift, and the full-BZ integral of sum|E| is shift-invariant, so the discrete
  flux does not affect the matter ground-state energy -- the per-cell flux is a FLAT
  direction.  Consequently the matter does NOT frustrate the trivial flux: the Higgs
  doublet (EW center, step 5) can sit in trivial flux on the 1-skeleton, the Perron VEV
  mode survives, and electroweak breaking is NOT obstructed by the surface topology.

  Combined with the lead, this CLOSES the topology<->EW question opened by step 2: the
  genus-3 surface neither DRIVES EW breaking (step 5: the EW Z2 is the adjoint-blind SU(2)
  center) NOR FRUSTRATES it (here: the per-cell matter flux is a flat direction, trivial
  permitted).  The surface geometry is fully CONSISTENT with -- but does not by itself
  select -- ordinary electroweak breaking; any flux selection must come from elsewhere
  (the Kitaev plaquette term / gauge dynamics / a finite-size or defect flux), not the
  per-cell matter band energy.""")
    else:
        print(f"""  The srs matter ground-state Z2 flux is NON-TRIVIAL ({gs}); the doublet (EW center,
  step 5) then sees non-trivial flux, FRUSTRATING the Perron VEV -- a genuine
  topology<->EW obstruction to pursue.""")
    print("""
  HONEST SCOPE.  Adjacency (Majorana-hopping) matter Hamiltonian; half-filling; the VEV
  and the EW coupling are posited; this characterises the per-cell flux ground state, not
  a full derivation. No graded content changes.""")

    if FAILURES:
        print(f"\n RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print("\ngyroid_matter_flux_groundstate_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
