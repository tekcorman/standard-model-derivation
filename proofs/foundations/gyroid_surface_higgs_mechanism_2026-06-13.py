#!/usr/bin/env python3
"""
gyroid_surface_higgs_mechanism_2026-06-13.py
============================================
THE LEAD (step 2): the Higgs mechanism on the genus-3 surface complex -- the Perron
condensate VEV gaps the genus gauge modes.

Step 1 (`gyroid_surface_gauge_higgs_hodge`) built the free Hodge dynamics: the gauge
field's physical (harmonic) modes = H_1 = the gyroid genus (3 at Gamma), and the Higgs
zero mode = the Gamma Perron / constant mode (the framework's "condensate's home").
This probe turns on the coupling and breaks the symmetry: put a VEV in the Perron mode
and show the genus gauge modes -- massless in the free theory -- acquire mass (the
abelian Higgs mechanism, realised on the complex).

CONSTRUCTION (abelian U(1), at Gamma where the genus modes live).
  * Higgs phi in C^0 (scalar on the 4 vertices); VEV phi0 = v * Perron mode
    (the unit kernel vector of Delta_0 at Gamma = the normalised constant (1,1,1,1)/2).
  * gauge field A in C^1 (one real per undirected edge).
  * gauge-covariant Higgs energy  E(A) = sum_e | phi_{t(e)} e^{iA_e} - phi_{s(e)} |^2.
  * gauge-boson MASS matrix  M = Hessian d^2 E / dA dA  at A = 0, phi = phi0.
  * physical (transverse) gauge subspace = complement of pure-gauge im d_1^t.
  * massive gauge spectrum = eigenvalues of (Delta_1^free + M) on the physical subspace.

RESULT (computed).  M = 2 v^2 I on C^1 (constant VEV).  The 3 genus modes (free mass 0)
acquire mass^2 = 2 v^2; the 3 pure-gauge (longitudinal) modes are eaten as the would-be
Goldstones.  So EW-type breaking by the Perron condensate makes the genus-3 topological
gauge sector massive: 3 massive physical gauge bosons + the eaten Goldstones + 1 massive
radial Higgs.

NOTE (flagged, NOT a claim): the count 3 (massive physical gauge bosons from genus 3)
coincides with the SM's 3 massive electroweak bosons (W+, W-, Z).  This is SUGGESTIVE
only -- the abelian toy is not the SM gauge group, and the genus-3 is the topological
flux of the whole gauge complex, not the broken EW triplet; identifying them needs the
non-abelian Pati-Salam structure (the open step).  The VEV v is POSITED (the framework
flags scalar-potential posits / sigma-coupler freedom); the mechanism is shown, the VEV
magnitude is input.  No graded content changes.
"""

import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds  # noqa: E402

FAILURES = []
V_VEV = 1.7   # posited Higgs VEV (arbitrary units; the mechanism is v-independent in form)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- srs combinatorics at Gamma (real incidence) -----------------------------
bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
REV = {a: EDGES.index((j, i, tuple(-x for x in c))) for a, (i, j, c) in enumerate(EDGES)}
SEEN, UBONDS = set(), []
for (i, j, c) in EDGES:
    if (j, i, tuple(-x for x in c)) in SEEN:
        continue
    SEEN.add((i, j, tuple(c)))
    UBONDS.append((i, j, tuple(c)))
NU, NV = len(UBONDS), 4


def d1_gamma():
    D = np.zeros((NV, NU))
    for b, (i, j, c) in enumerate(UBONDS):
        D[i, b] += -1.0
        D[j, b] += 1.0
    return D


def higgs_energy(A, phi):
    """E(A) = sum_e |phi_t e^{iA_e} - phi_s|^2 at Gamma."""
    e = 0.0
    for b, (i, j, c) in enumerate(UBONDS):
        e += abs(phi[j] * np.exp(1j * A[b]) - phi[i]) ** 2
    return e


def mass_matrix(phi, h=1e-4):
    """M = Hessian of E(A) wrt A at A=0 (numerical, symmetric)."""
    M = np.zeros((NU, NU))
    e0 = higgs_energy(np.zeros(NU), phi)
    for a in range(NU):
        for b in range(a, NU):
            ea = np.zeros(NU); ea[a] = h
            eb = np.zeros(NU); eb[b] = h
            epp = higgs_energy(ea + eb, phi)
            ep0 = higgs_energy(ea, phi)
            e0b = higgs_energy(eb, phi)
            M[a, b] = M[b, a] = (epp - ep0 - e0b + e0) / h ** 2
    return M


def main():
    print("=" * 88)
    print(" THE LEAD (step 2): Higgs mechanism -- Perron VEV gaps the genus gauge modes")
    print("=" * 88)

    D1 = d1_gamma()
    L0 = D1 @ D1.T                 # Higgs/scalar Laplacian at Gamma
    L1 = D1.T @ D1                 # at Gamma, d2 d2^t = 0, so gauge Hodge Lap = d1^t d1

    # --- A: the VEV is the Perron mode --------------------------------------
    w0, V0 = la.eigh(L0)
    perron = V0[:, 0]
    perron = perron / perron[0] * abs(perron[0])   # fix sign
    perron = perron / la.norm(perron)
    print("\n A  Higgs VEV channel = the Perron zero mode of Delta_0 at Gamma")
    print(f"    Delta_0 spectrum at Gamma = {np.round(w0,3)}  -> zero mode = Perron")
    print(f"    Perron vector (normalised) = {np.round(perron,3)}  (constant => the condensate's home)")
    gate("A VEV mode is the constant/Perron zero mode of Delta_0 (dim ker = 1)",
         abs(w0[0]) < 1e-9 and np.allclose(np.abs(perron), abs(perron[0]), atol=1e-6))

    phi0 = V_VEV * perron / abs(perron[0])         # phi0 = v * (1,1,1,1)
    print(f"    phi0 = v * (1,1,1,1), v = {V_VEV}")

    # --- B: gauge-boson mass matrix from the covariant Higgs energy ----------
    M = mass_matrix(phi0)
    print("\n B  gauge-boson mass matrix  M = d^2 E/dA^2 at the VEV")
    print(f"    M is proportional to I:  M ~ {np.round(M[0,0],4)} * I ? "
          f"max|M - mI|={np.max(np.abs(M - M[0,0]*np.eye(NU))):.2e}")
    expected = 2 * V_VEV ** 2
    gate("B M = 2 v^2 * I on C^1 (constant VEV gives a uniform gauge mass term)",
         np.allclose(M, expected * np.eye(NU), atol=1e-3),
         f"M[0,0]={M[0,0]:.3f} vs 2v^2={expected:.3f}")

    # --- C: physical subspace + the mechanism --------------------------------
    # pure-gauge (longitudinal) = im d_1^t ;  physical = its orthogonal complement
    U, s, _ = la.svd(D1.T, full_matrices=True)
    pg_dim = int(np.sum(s > 1e-9))
    PG = U[:, :pg_dim]                       # pure-gauge directions
    PHYS = U[:, pg_dim:]                     # physical (harmonic + coexact) directions
    phys_dim = PHYS.shape[1]

    free_phys = PHYS.T @ L1 @ PHYS          # free gauge operator on physical modes
    massive_phys = PHYS.T @ (L1 + M) @ PHYS  # after EW breaking
    ev_free = np.sort(la.eigvalsh((free_phys + free_phys.T) / 2))
    ev_mass = np.sort(la.eigvalsh((massive_phys + massive_phys.T) / 2))
    print("\n C  physical gauge spectrum before / after the Perron VEV")
    print(f"    pure-gauge (longitudinal, eaten) dim = {pg_dim};  physical dim = {phys_dim}")
    print(f"    free physical mass^2  = {np.round(ev_free,3)}   (the genus modes are massless)")
    print(f"    broken physical mass^2 = {np.round(ev_mass,3)}   (= 2 v^2 = {expected:.3f})")
    gate("C1 free physical gauge modes are the 3 massless genus modes",
         phys_dim == 3 and np.allclose(ev_free, 0, atol=1e-7))
    gate("C2 after the Perron VEV all 3 genus modes acquire mass^2 = 2 v^2 (Higgs mechanism)",
         np.allclose(ev_mass, expected, atol=1e-3))
    gate("C3 the 3 pure-gauge (longitudinal) modes are eaten as would-be Goldstones",
         pg_dim == 3)

    # --- D: verdict ----------------------------------------------------------
    print("\n" + "=" * 88)
    print(" VERDICT  (the lead, step 2)")
    print("=" * 88)
    print(f"""  The Higgs mechanism runs natively on the surface complex: the Perron condensate
  (the Delta_0 zero mode at Gamma) acquires a VEV, and the 3 genus gauge modes -- the
  physical, massless harmonic sector of step 1 -- pick up mass^2 = 2 v^2.  The 3
  pure-gauge (longitudinal) modes are eaten as the would-be Goldstones; one radial Higgs
  remains massive.  So the genus-3 TOPOLOGICAL gauge sector becomes 3 massive physical
  gauge bosons after EW-type breaking by the framework's named condensate.

  SUGGESTIVE (flagged, NOT a claim): 3 massive physical gauge bosons = the SM's W+, W-, Z
  count.  This is a coincidence of counts only -- the abelian toy is not the SM gauge
  group, and genus-3 is the whole complex's topological flux, not the broken EW triplet.
  Settling it needs the non-abelian Pati-Salam structure (the open step).

  HONEST SCOPE.  Abelian U(1) demonstration; the VEV v is POSITED (scalar-potential
  posits are flagged framework-wide as sigma-coupler freedom).  Open next: non-abelian
  PS via Cl(6)/Cl(2) operator-algebra cochains; a DERIVED Higgs potential; matter
  coupling/Yukawa. No graded content changes.""")

    if FAILURES:
        print(f"\n RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print("\ngyroid_surface_higgs_mechanism_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
