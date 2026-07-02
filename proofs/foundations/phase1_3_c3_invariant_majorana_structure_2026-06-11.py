#!/usr/bin/env python3
"""Phase 1.3 — C3-invariance of the Majorana bilinear: the third branch.

Pure group theory + seesaw algebra; no saddle data, no holonomy values.

Setup: generations = C3 characters {1, w, w2} (R3 identification). Under
C3, nu_m -> chi_m nu_m. A MAJORANA mass term is the symmetric bilinear
(1/2) M^{mm'} nu_m nu_{m'}: invariance requires chi_m * chi_{m'} = 1
entrywise. A DIRAC term nu_L,m nubar_R,m' transforms with chi_m chi_{m'}^*:
diagonal allowed.

Gates (exact, basis = character basis (1, w, w2)):
  J1  The C3-invariant symmetric M_R is EXACTLY { a on (1,1); b on
      (w,w2)+(w2,w) } — in particular the class-diagonal entries (w,w) and
      (w2,w2) are FORBIDDEN. (The adopted form M_R^(m,m) = |M_R| h_m^g,
      diagonal across all three classes, is NOT C3-invariant as a Majorana
      bilinear unless h_w-type entries vanish.)
  J2  Eigenvalues of the invariant M_R: {a, +b, -b} — the omega-sector pair
      is split by SIGN only: relative Majorana phase pi, equal magnitudes.
  J3  Seesaw closure: with C3-invariant DIAGONAL m_D = diag(d1, dw, dw2),
      m_nu = -m_D M_R^{-1} m_D^T again has the invariant form
      { d1^2/a ; antidiag(dw dw2 / b) } -> massive pair = +/- (dw dw2 / b):
      the RELATIVE PHASE PI SURVIVES THE SEESAW (common phases cancel in
      the ratio), independent of all holonomy/saddle data.
  J4  m_nu1 = 0 iff the trivial-channel Dirac coupling d1 = 0 (rank-2
      without any condition on M_R's a-entry).

[PANEL-CORRECTED 2026-06-11 (Majorana-sector panel): J1-J4 verified exactly
by all seven refuters and STRENGTHENED — diagonal m_D is FORCED by exact C3
(chi*_i chi_j = 1 admits only diagonal), not assumed; and pi survives O(1)
zero-diagonal Majorana-side breaking that reproduces R_nu = 228/7 (Takagi
invariant: [[z,1],[1,0]] = P_L (real sym) P_L). THREE OVERCLAIMS REMOVED:
(a) NOT saddle-independent — the invariant form follows from the SAME-FIBER
law, valid at TRIM saddles (Gamma/H/N) only; at the non-TRIM P the bilinear
pairs P with -P = P+Delta (conjugated characters) under the crossing law
chi*chibar' = 1, which ALLOWS the class-diagonal form (the strike against
the adopted M_R is WITHDRAWN as mis-aimed; properly aimed it kills the
H-reading's M_R prop-to 1). (b) "C3 breaking REQUIRED" is overdrawn as
Majorana-side: degeneracy breaking is required, and may enter DIRAC-side
via the framework's W49 EWSB C3-mixing vacuum with M_R exactly invariant.
(c) pi is the class-basis / zero-diagonal TAKAGI invariant (flavor-row
relative phases have O(2) gauge freedom at exact degeneracy).]

Consequence (the C3/TRIM branch of the Majorana fork): with m_nu1 = 0 there
is ONE physical Majorana phase; exact C3 at a TRIM saddle forces it to pi,
with |m_2| = |m_3| — contradicted by R_nu = 228/7, so the physical phase =
pi - delta_breaking with delta underived. The minimal N-spillover anchor
(17.612 deg) was REFUTED over 19 breaking placements (0 passes; see
phase1_3_od_nspillover_takagi_2026-06-11.py). Conditional on: R3 +
same-fiber nu-nu bilinear (TRIM). Rows 7/8/9 untouched; panel annotation in
the preregistration register documents the fork.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

TOL = 1e-12
FAILURES = []
RNG = np.random.default_rng(20260611)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 1.3 -- C3-invariant Majorana structure: the third branch (pi)")
    print("=" * 72)
    w = np.exp(2j * np.pi / 3)
    U = np.diag([1, w, w**2])

    # J1: enumerate invariant symmetric matrices: U^T M U = M
    basis = []
    for i in range(3):
        for j in range(i, 3):
            E = np.zeros((3, 3), dtype=complex)
            E[i, j] = E[j, i] = 1
            if np.allclose(U.T @ E @ U, E, atol=TOL):
                basis.append((i, j))
    gate("J1 invariant symmetric entries are exactly {(0,0), (1,2)}",
         basis == [(0, 0), (1, 2)], f"allowed={basis} (class-diagonal (w,w),(w2,w2) forbidden)")

    # J2: eigenvalues {a, +b, -b}
    a, b = (RNG.standard_normal(2) + 1j * RNG.standard_normal(2))
    M = np.zeros((3, 3), dtype=complex)
    M[0, 0] = a
    M[1, 2] = M[2, 1] = b
    # Takagi-type physical masses: for the antidiag block the two Majorana
    # eigenvalues are +b and -b (equal magnitude, relative phase pi)
    ev = np.linalg.eigvals(M)
    ev_sorted = sorted(ev, key=lambda z: (abs(z - a) > 1e-9, z.real))
    pair = [z for z in ev if abs(z - a) > 1e-9]
    gate("J2 eigenvalues {a, +b, -b}: massive pair equal |.|, relative phase pi",
         abs(pair[0] + pair[1]) < TOL and abs(abs(pair[0]) - abs(b)) < TOL,
         f"pair sum = {abs(pair[0] + pair[1]):.1e}")

    # J3: seesaw preserves the structure and the pi
    d1, dw, dw2 = RNG.standard_normal(3) + 1j * RNG.standard_normal(3)
    mD = np.diag([d1, dw, dw2])
    mnu = -mD @ np.linalg.inv(M) @ mD.T
    ok_form = (abs(mnu[0, 1]) < TOL and abs(mnu[0, 2]) < TOL
               and abs(mnu[1, 1]) < TOL and abs(mnu[2, 2]) < TOL
               and abs(mnu[1, 2] - mnu[2, 1]) < TOL)
    evl = np.linalg.eigvals(mnu)
    pairl = sorted(evl, key=lambda z: -abs(z))[:2]
    gate("J3 seesaw: m_nu inherits the invariant form; massive pair = +/-mu (phase pi)",
         ok_form and abs(pairl[0] + pairl[1]) < 1e-9,
         f"pair sum = {abs(pairl[0] + pairl[1]):.1e}")

    # J4: m_nu1 = 0 iff d1 = 0
    mD0 = np.diag([0, dw, dw2])
    mnu0 = -mD0 @ np.linalg.inv(M) @ mD0.T
    ev0 = sorted(np.abs(np.linalg.eigvals(mnu0)))
    gate("J4 d1 = 0 => m_nu1 = 0 exactly (rank-2, no condition on M_R)",
         ev0[0] < TOL and ev0[1] > 1e-6, f"|m| = {np.round(ev0, 6)}")

    print("\n  C3/TRIM BRANCH (panel-corrected): the single physical Majorana")
    print("  phase = pi at TRIM saddles under exact C3 (same-fiber law; the")
    print("  crossing law at non-TRIM P allows class-diagonal instead).")
    print("  Diagonal m_D is FORCED by exact C3; pi survives zero-diagonal")
    print("  O(1) breaking; splitting may be Dirac-side (W49 EWSB vacuum).")
    print("  Rows 7/8/9 untouched; panel annotation in the register.")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- C3-invariant Majorana structure exact")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
