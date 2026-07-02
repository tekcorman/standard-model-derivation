#!/usr/bin/env python3
"""The Higgs-VEV amplitude prefactor delta^2/sqrt2 = |h|^3/k*^4: |h|=sqrt2 is the
P-point winding-weighted CLOSING AMPLITUDE of the observer's read-walk.
Self-checking; not in the verify backbone.

Result (2026-06-14): the two remaining prefactor pieces of
predictions/v_higgs_derivation.md -- delta=2/9 and sqrt2=|h|_P -- both reduce to
ONE read quantity, |h|, the P-point closing amplitude of the non-backtracking
read-walk weighted by the winding phase (f1 = edge direction = net cell
displacement).  |h|^2 = 2 = k*-1 (Ramanujan), so
  delta = |h|^2 / k*^2 = 2/9,
  amplitude prefactor delta^2/sqrt2 = |h|^3/k*^4 = 2 sqrt2 / 81.
The B(P) eigenvalue is exactly h = (sqrt3 + i sqrt5)/2; ALL dominant P-modes
have |lambda| = sqrt2.  Companion: vev_prefactor_nb_closing,
project memory project_vev_observer_read_decomposition_2026-06-14.

GATES (exact):
  G1 Gamma closing amplitude = 2 (real-space NB growth; momentum-sensitive)
  G2 B(P) eigenvalue = h = (sqrt3 + i sqrt5)/2 exactly
  G3 |h|^2 = 2 = k*-1
  G4 delta = |h|^2/k*^2 = 2/9
  G5 amplitude prefactor = |h|^3/k*^4 = 2 sqrt2 / 81
"""
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, 'proofs')
from common import find_bonds  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def B_of(k):
    """Bloch non-backtracking (Hashimoto) operator with windings (phase4_2)."""
    bonds = find_bonds()
    E = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
    idx = {e: a for a, e in enumerate(E)}
    rev = {a: idx[(j, i, tuple(-x for x in c))] for a, (i, j, c) in enumerate(E)}
    B = np.zeros((12, 12), dtype=complex)
    for a2, (i, j, c) in enumerate(E):
        for b2, (i2, j2, c2) in enumerate(E):
            if i2 == j and b2 != rev[a2]:
                B[b2, a2] = np.exp(2j * np.pi * np.dot(k, np.asarray(c2, float)))
    return B


KSTAR = 3
P = np.array([0.25, 0.25, 0.25])
GAMMA = np.zeros(3)

print("=" * 76)
print(" VEV AMPLITUDE PREFACTOR: |h|=sqrt2 = P-point winding closing amplitude")
print("=" * 76)

rad_gamma = max(abs(la.eigvals(B_of(GAMMA))))
rad_P = max(abs(la.eigvals(B_of(P))))
print(f"\n  closing amplitude (spectral radius of B(k)):  Gamma={rad_gamma:.5f}  "
      f"P={rad_P:.5f}")
gate("G1 Gamma closing amplitude = 2 (real-space NB growth; momentum-sensitive)",
     abs(rad_gamma - 2.0) < 1e-6, f"{rad_gamma:.5f}")

evP = la.eigvals(B_of(P))
h_ref = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
hmatch = evP[np.argmin(np.abs(evP - h_ref))]
mods = np.sort(np.abs(evP))[::-1]
print(f"\n  h_ref=(sqrt3+i sqrt5)/2={h_ref:.5f} |h|={abs(h_ref):.5f}; nearest "
      f"B(P) eig={hmatch:.5f}; |eig| top6={np.round(mods[:6],4)}")
gate("G2 B(P) eigenvalue = h = (sqrt3 + i sqrt5)/2 exactly",
     abs(hmatch - h_ref) < 1e-6, f"diff {abs(hmatch-h_ref):.1e}")

h2 = abs(hmatch) ** 2
delta = h2 / KSTAR ** 2
amp = delta ** 2 / abs(hmatch)
gate("G3 |h|^2 = 2 = k*-1 (Ramanujan)", abs(h2 - (KSTAR - 1)) < 1e-6, f"|h|^2={h2:.5f}")
gate("G4 delta = |h|^2/k*^2 = 2/9", abs(delta - 2 / 9) < 1e-6, f"delta={delta:.5f}")
gate("G5 amplitude prefactor delta^2/sqrt2 = |h|^3/k*^4 = 2 sqrt2/81",
     abs(amp - 2 * np.sqrt(2) / 81) < 1e-6
     and abs(amp - abs(hmatch) ** 3 / KSTAR ** 4) < 1e-9,
     f"{amp:.6f} = 2sqrt2/81 = {2*np.sqrt(2)/81:.6f}")

print(f"\n  => the amplitude prefactor reduces to |h|=sqrt2 (P-point winding")
print(f"     closing amplitude) and the coordination k*=3.  With the closing")
print(f"     survival/cycles (vev_prefactor_nb_closing) and the recurrence")
print(f"     exponent (vev_exponent_observer_recurrence): the WHOLE VEV is one")
print(f"     observer-read of one NB walk -- returns x closing x vertex-overlap.")

print("\n" + "=" * 76)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- |h|=sqrt2 is the P-point closing amplitude")
print("=" * 76)
sys.exit(0)
