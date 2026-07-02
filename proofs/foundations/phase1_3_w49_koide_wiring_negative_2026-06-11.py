#!/usr/bin/env python3
"""Phase 1.3 — NEGATIVE (gated): the W49/Koide zero-parameter wiring fails.

Hypothesis tested (pre-declared, zero new parameters): the neutrino Dirac
block shares the charged-lepton Koide C3-breaking geometry (epsilon = sqrt2,
delta_e = 2/9 — both theorem-grade, EWSB sector; nu_L and e_L share the
SU(2) doublet), while M_R stays exactly C3-invariant ({a; antidiag(b)},
a -> inf decoupling the trivial channel, m_nu1 = 0 exact). Then BOTH
R_nu = m3^2/m2^2 and the physical Majorana phase are forced with NO freedom.

Pre-declared wirings (the only two; no variants tried):
  W1: m_D = Koide circulant at sqrt-m level: in the character basis
      C = I + (sqrt2/2) e^{i 2/9} S + (sqrt2/2) e^{-i 2/9} S^2.
  W2: mass level, C -> C^2.
Seesaw: m_nu = -C X C^T, X = exchange E_{w,w2} + E_{w2,w}.

RESULT — both wirings REFUTED on magnitude (gated below):
  W1: R = 529.54   (needed 228/7 = 32.571)
  W2: R = 142724   (needed 32.571)
The Koide breaking is O(1) and splits the seesaw pair ~16x (W1) too hard.
delta_breaking remains UNDERIVED; the Majorana-phase content of Phase 1.3
stays K3-PARTIAL per the 2026-06-11 panel ruling. Tried-pattern ledger: +2.
"""
import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 1.3 -- W49/Koide zero-parameter wiring: gated NEGATIVE")
    print("=" * 72)
    S = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    X = np.zeros((3, 3))
    X[1, 2] = X[2, 1] = 1.0
    kappa, delta = np.sqrt(2) / 2, 2 / 9

    def R_of(level):
        C = (np.eye(3, dtype=complex)
             + kappa * np.exp(1j * delta) * S
             + kappa * np.exp(-1j * delta) * (S @ S))
        if level == 2:
            C = C @ C
        mnu = -(C @ X @ C.T)
        sv = np.sqrt(np.abs(np.sort(la.eigvalsh(mnu @ mnu.conj().T))))
        return (sv[2] / sv[1]) ** 2, sv[0]

    R1, m1a = R_of(1)
    R2, m1b = R_of(2)
    target = 228 / 7
    gate("W1 (sqrt-m level): R = 529.54, NOT 228/7 -- refuted",
         abs(R1 - 529.5447) < 0.01 and abs(R1 - target) > 1,
         f"R = {R1:.4f}, m_nu1 = {m1a:.2e} (rank-2 OK)")
    gate("W2 (mass level): R = 142724, NOT 228/7 -- refuted",
         abs(R2 - 142724.38) < 1.0 and abs(R2 - target) > 1,
         f"R = {R2:.2f}, m_nu1 = {m1b:.2e}")
    gate("hypothesis dead: no pre-declared wiring reproduces R_nu",
         abs(R1 - target) > 1 and abs(R2 - target) > 1,
         "delta_breaking remains underived; K3-PARTIAL stands")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- negative result gated (panel standard)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
