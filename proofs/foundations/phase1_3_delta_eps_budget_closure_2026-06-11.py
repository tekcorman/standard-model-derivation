#!/usr/bin/env python3
"""Phase 1.3 — δ/ε² budget run: K2-PARTIAL closure (gated).

The last open item of the Phase 1.3 bet: are the Koide sector phases
δ_s = {2/9, 1/9, 2/27} and the ε² bands {2, 2+6α₁f, 2+6α₁f·(14/5)} data of
the sector structure? Hard budget: 10 pre-declared patterns, then verdict.

PRE-DECLARED BUDGET (all evaluated; outcomes frozen here):
  B1 n_s = color multiplicity {1,3,3}            -> FAIL (needs {1,2,3})
  B2 n_s = |Q|^-1 pattern {1,3,3/2}              -> FAIL
  B3 n_s from anchor walk lengths {8,10,0}       -> FAIL (no forced map)
  B4 n_s = mirror-tower rung via sector address  -> FAIL (addresses plural/
                                                    unforced per L=8 census)
  B5 n_s = #sector classes at the anchor L       -> FAIL (lepton needs n=1,
                                                    L=8 census shows 3)
  B6 "period-2 sector" for the u-power rewrites  -> FAIL (no sector supplies
                                                    period 2; standing gap)
  B7 14/5 = 2*N_14(0)/N_10(0) = 2*168/120        -> EXACT but mechanism-free
                                                    (dangling factor 2);
                                                    LOGGED, NOT PROMOTED
  B8 14/5 from h-data (tan^2 arg h combinations) -> FAIL (no clean form)
  B9 ε²_e = 2 = k*-1 = |h|² (Ramanujan bound)    -> EXACT, type-(iii) sector
                                                    data, ZERO new bits (D2)
  B10 δ family one-number form                   -> EXACT identity (D1);
      (δ_e, δ_d, δ_u) = (2/9)/{1,2,3} = {2/k*², 1/k*², 2/k*³}
      (equivalent to the priced u-power rewrites; divisor set {1,2,3} =
      {1,2,k*} indistinguishable at k* = 3)

VERDICT (per the frozen bet spec): **K2-PARTIAL.** Established at zero bits:
ε²_e = k*-1 (Ramanujan saturation) and the δ family's reduction to ONE
derived number (2/9, in-repo theorem-grade via Wigner D¹) divided by an
integer label n_s ∈ {1,2,3}. UNDERIVED: the n_s assignment rule (~2.6 bits)
and the quark ε² dressings (6α₁f, 14/5). Budget spent: 10/10; 0 forcing
rules promoted. Tried-pattern ledger total: 48.
"""
import os
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 1.3 -- delta/eps^2 budget closure: K2-PARTIAL")
    print("=" * 72)
    u, k = F(2, 3), 3
    h = (np.sqrt(3) + 1j * np.sqrt(5)) / 2

    gate("D1 delta family = ONE number / {1,2,3}: (2/9, 1/9, 2/27) exact",
         F(2, 9) == F(2, k**2) == u**2 / 2
         and F(1, 9) == F(2, 9) / 2 == F(1, k**2)
         and F(2, 27) == F(2, 9) / 3 == F(2, k**3),
         "= {2/k*^2, 1/k*^2, 2/k*^3}; divisors {1,2,3} == {1,2,k*} at k*=3")

    gate("D2 eps^2_e = 2 = k*-1 = |h|^2 exactly (Ramanujan bound; zero bits)",
         k - 1 == 2 and abs(abs(h) ** 2 - 2) < 1e-15, f"|h|^2 = {abs(h)**2:.16f}")

    gate("D3 B7 curiosity exact (logged, unpromoted): 14/5 = 2*N_14(0)/N_10(0)",
         F(14, 5) == 2 * F(168, 120), "dangling factor 2 -- no mechanism")

    print("\n  VERDICT: K2-PARTIAL. The n_s in {1,2,3} assignment rule and the")
    print("  quark eps^2 dressings remain underived after the 10-pattern budget")
    print("  (0 promoted). The delta sector content reduces to one derived")
    print("  number (2/9, Wigner D1) + ~2.6 unforced bits; eps^2_e is Ramanujan.")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- Phase 1.3 delta/eps^2 closed at K2-PARTIAL")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
