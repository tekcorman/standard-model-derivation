#!/usr/bin/env python3
"""
proofs/foundations/OMEGA_S2_Q2_internal_a4_gauge_row_2026-07-02.py

OMEGA SESSION 2, STATION 2, ENTRY QUESTION Q2 -- is the object's gauge fluctuation a
VECTOR MULTIPLET (graded pair => -3 C2) or a BARE VECTOR (=> -11/3 C2)? Decided by
COMPUTING the grading structure, never by inheriting it from the N=1 target shape
(kickoff par.2, "the fit trap in grade clothing").

WHAT IS AT STAKE (todo par.5): the beta VALUES {33/5, 1, -3} are in hand (2HDM Dynkin
sums + the computed completion, read_gauge_running); OMEGA_T1 proved b_4d == -3C2 +
T_f + T_H (N=1 index form) and localized the gauge row OUT of the band sector. The
OPEN equation is the beta FORMULA -- the (-11/3, 2/3, 1/3) structure, still "standard
QFT typed" (Layer-2). This probe attacks the formula's native origin AND the grading.

PRE-REGISTERED CLAIMS AND CLASSES:
  T-A/T-B  machinery validation on EXACT spectra (calibration; no class): the two
           universal Seeley-DeWitt coefficients a4 in (1/12) tr Omega^2 + (1/2) tr E^2,
           checked against closed-form torus/Landau traces.
  T-C      SM-REPRODUCTION: the three one-loop row-values {+1/3 complex scalar,
           +2/3 Weyl, -11/3 vector+ghost} DERIVED from those two coefficients with
           the magnetic-moment endomorphism E = -2 F.S -- per helicity pair
           b = -(-1)^{2s} [(2 s_z)^2 - 1/3]. Seeley-DeWitt is the DECLARED Type-3
           import (published math, same status as Ihara-Bass); the "one-loop QFT
           formula" import of read_gauge_running is thereby REPLACED by the same
           spectral-action layer zeta_{D4}(0) lives in.
  T-D      exact-rational THEOREM: in any grading that pairs each helicity pair with
           an opposite-statistics partner of spin |s - 1/2| and equal internal
           content, the orbital (-1/3) parts cancel pairwise and the graded totals
           are -3 C2 (vector pair) and +T (chiral pair): b_graded = -3 C2 + T_f + T_H
           -- OMEGA_T1's identity re-derived from the formula layer.
  T-E      STRUCTURAL (the Q2 decision): does the OBJECT supply the pairing?
           Computed: (i) on every NONZERO mode the pairing operator EXISTS and is D3
           itself (the supercharge: parity-flipping, isospectral, commuting with the
           internal charges); (ii) the FLATS -- exactly the spatial gauge modes --
           are UNPAIRED (D3.flat = 0): the gaugino/higgsino shadow content CANNOT
           come from the spatial complex; it localizes to the time-leg (gamma_t d_N)
           fluctuation complex, which is UN-BUILT.

KILL CRITERIA (pre-registered): K1 the exact-spectrum checks of (1/12)/(1/2) fail;
K2 the three row-values do not reproduce from the dictionary (no refitting allowed;
one unit normalization, two forced outcomes); K3 the graded cancellation fails;
K4 the object-side pairing fails (D3-partners wrong parity/energy, or flats paired).
OUTCOME EITHER WAY: Q2 is answered by computation; what does not close is NAMED and
logged (todo par.5), not absorbed.

Falsification surface (kickoff par.6.4): the SAME machinery must reproduce the matter
row -- the Weyl row here IS the +2/3 T content the 06-25 cone probe measured
(1 Weyl per cone, spacelike log); no per-row tuning exists anywhere below.
"""
import math
import os
import sys
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

print("=" * 88)
print(" T-A  exact-spectrum check of the E-coefficient: a4 contains (1/2) tr E^2")
print("=" * 88)
# H = -d^2 + E on T^3 (L = 2pi) with a constant Hermitian E: the heat trace factorizes
# exactly: Tr e^{-tH} = theta3(t)^3 . tr e^{-tE}; the (4 pi t)^{-3/2} V prefactor is the
# a0 layer, and tr e^{-tE} = m - t tr E + (t^2/2) tr E^2 - ... supplies a4 = (1/2) tr E^2.
t0 = 0.5
th3 = sum(math.exp(-t0 * n * n) for n in range(-40, 41))
pref = (4 * math.pi * t0) ** -0.5 * (2 * math.pi)
check(f"torus theta-normalization: theta3(t)/[(4 pi t)^(-1/2) L] - 1 = "
      f"{th3/pref-1:.2e} (Poisson-exact)", abs(th3 / pref - 1) < 1e-8)
E = np.array([[0.3, 0.1 + 0.2j], [0.1 - 0.2j, -0.4]])
evE = np.linalg.eigvalsh(E)
for tt in (0.02, 0.01):
    c = sum(math.exp(-tt * e) for e in evE)
    series = (2 - tt * np.trace(E).real + tt ** 2 / 2 * np.trace(E @ E).real
              - tt ** 3 / 6 * np.trace(E @ E @ E).real)
    check(f"tr e^(-tE) at t={tt}: matches 2 - t trE + (t^2/2) trE^2 - (t^3/6) trE^3 "
          f"to {abs(c-series):.1e}", abs(c - series) < 1e-6)
print("    => the endomorphism enters a4 as +(1/2) tr E^2 (with a2 = -tr E): standard")
print("       Seeley-DeWitt, verified on an exact spectrum, conventions locked.")

print("=" * 88)
print(" T-B  exact-spectrum check of the curvature coefficient: a4 contains (1/12) tr Omega^2")
print("=" * 88)
# Landau levels on the plane (per unit area): Tr e^{-tH} = (1/(4 pi t)) . tB/sinh(tB).
# Expansion: 1 - (tB)^2/6 + 7(tB)^4/360 - ...; Omega_{12} = iB (U(1) curvature) gives
# (1/12) Omega_mn Omega^mn = (1/12)(2)(iB)^2 = -B^2/6: the t^2 coefficient. Exact match:
B = 0.7
okB = True
for tt in (0.05, 0.1, 0.2):
    x = tt * B
    exact = x / math.sinh(x)
    series = 1 - x * x / 6 + 7 * x ** 4 / 360
    okB &= abs(exact - series) < 3e-6 * max(1, 1)
check("Landau trace tB/sinh(tB) = 1 - (tB)^2/6 + 7(tB)^4/360 - ... (exact spectrum: "
      "the t^2 coefficient IS (1/12) tr Omega^2 = -B^2/6)", okB)
# and the Zeeman factor on top (spin-1/2): 2 cosh(tB) supplies +(1/2) tr E^2 = B^2:
okZ = True
for tt in (0.05, 0.1):
    x = tt * B
    exact = (x / math.sinh(x)) * 2 * math.cosh(x)
    series = 2 * (1 + x * x * (1 / 2 - 1 / 6)) + 2 * x ** 4 * (7 / 360 + 1 / 24 - 1 / 12)
    okZ &= abs(exact - series) < 5e-6
check("with E = -sigma.B (spin-1/2 Zeeman): net t^2 content per pair = "
      "[(2 s_z)^2 - 1/3] . B^2/2 . t^2-normalized -- the PARAMAGNETIC term dominates "
      "the DIAMAGNETIC -1/3 (Nielsen-Hughes structure, from the same two coefficients)", okZ)

print("=" * 88)
print(" T-C  the three row-values DERIVED from the two coefficients (exact rationals;")
print("      one unit normalization, two forced outcomes)  [SM-REPRODUCTION]")
print("=" * 88)
# per helicity pair (+-s_z), statistics F in {0,1}:  b_pair = -(-1)^{2s} [(2 s_z)^2 - 1/3]
def b_pair(two_sz, fermion):
    orb = Fraction(-1, 3)                          # (1/12) tr Omega^2 per complex pair
    zee = Fraction(two_sz * two_sz)                # (1/2) tr E^2, E = -2 F.S per pair
    return (1 if fermion else -1) * (zee + orb)
rows = {
    "complex scalar": b_pair(0, False),            # s_z = 0 boson pair
    "Weyl fermion": b_pair(1, True),               # s_z = +-1/2 pair
    "vector + ghosts": b_pair(2, False),           # transverse s_z = +-1 pair (ghosts
                                                   # = -2 real scalar dof, absorbed by
                                                   # the pair convention; component
                                                   # check below)
}
KNOWN = {"complex scalar": Fraction(1, 3), "Weyl fermion": Fraction(2, 3),
         "vector + ghosts": Fraction(-11, 3)}
for name, val in rows.items():
    print(f"    {name:>16}: b = {val}   (known {KNOWN[name]})")
check("all three rows reproduce {+1/3, +2/3, -11/3} from (1/12)Omega^2 + (1/2)E^2 "
      "with E = -2F.S -- the beta FORMULA is the heat kernel's two universal "
      "coefficients + the spin dictionary, NOT an independent QFT import",
      all(rows[k] == KNOWN[k] for k in rows))
# component-level cross-check of the vector row (no pair shortcut). In b-units the
# scalar row fixes the orbital content: +1/3 per complex bosonic pair = +1/6 per REAL
# bosonic dof (bosons contribute orbital POSITIVELY: b_pair(0, boson) = +1/3);
# fermionic loops flip the sign.
orb_real = Fraction(1, 6)                           # orbital per REAL bosonic dof
orb_vec = 4 * orb_real                              # +2/3: all four components orbit
zee_vec = Fraction(-4)                              # Zeeman: E = -2F_mn acts on the
                                                    # transverse (1,2) block only: one
                                                    # +-1 pair, boson sign: -(2 s_z)^2 = -4
orb_ghost = -2 * orb_real                           # ghosts: 2 REAL Grassmann scalars,
                                                    # opposite loop sign: -1/3
vec_total = orb_vec + zee_vec + orb_ghost
print(f"    vector by components: orbital(4 real dof) {orb_vec} + Zeeman(transverse) "
      f"{zee_vec} + ghosts {orb_ghost} = {vec_total}")
check("component bookkeeping (4 dof + ghosts) agrees with the helicity-pair shortcut: "
      f"-11/3 (got {vec_total})", vec_total == Fraction(-11, 3))
# assemble b_2HDM from the derived rows + the registered content (regression against
# read_gauge_running's Dynkin sums -- same content table, no retuning):
def gauge_dynkin(fields, mult):
    T3 = {1: Fraction(0), 3: Fraction(1, 2), 8: Fraction(3)}
    T2 = {1: Fraction(0), 2: Fraction(1, 2), 3: Fraction(2)}
    s = {1: Fraction(0), 2: Fraction(0), 3: Fraction(0)}
    for c, w, Y in fields:
        s[3] += T3[c] * w * mult
        s[2] += T2[w] * c * mult
        s[1] += Fraction(3, 5) * Y * Y * c * w * mult
    return s
K = 3
sgn = lambda n: 1 if n % 2 == 0 else -1
Qn = lambda n: Fraction(sgn(n) * n, K)
fermions = [(3, 2, Qn(2) - Fraction(1, 2)), (1, 2, Qn(0) - Fraction(1, 2)),
            (3, 1, Qn(2)), (3, 1, Qn(1)), (1, 1, Qn(3))]
higgs = [(1, 2, Fraction(1, 2)), (1, 2, Fraction(-1, 2))]
Tf, TH = gauge_dynkin(fermions, 3), gauge_dynkin(higgs, 1)
C2G = {1: Fraction(0), 2: Fraction(2), 3: Fraction(3)}
b2hdm = {i: rows["vector + ghosts"] * C2G[i] + rows["Weyl fermion"] * Tf[i]
         + rows["complex scalar"] * TH[i] for i in (1, 2, 3)}
check(f"b_2HDM assembled from the DERIVED rows = {dict((i, str(b2hdm[i])) for i in (1,2,3))} "
      "= read_gauge_running's {21/5, -3, -7} (same content, no per-row tuning)",
      b2hdm[1] == Fraction(21, 5) and b2hdm[2] == -3 and b2hdm[3] == -7)
print("    [matter-row regression: the Weyl row +2/3 T IS the content the 06-25 cone")
print("     probe measured natively (1 Weyl per cone, spacelike log) -- same machinery.]")

print("=" * 88)
print(" T-D  the GRADED theorem (exact): pairing kills the orbital -1/3; only Zeeman")
print("      survives => -3 C2 + T_f + T_H  [the N=1 form, re-derived from the rows]")
print("=" * 88)
# shadow of a helicity pair (spin s, statistics F) = spin |s - 1/2| pair, statistics 1-F,
# same internal content (the completion's own accounting: sfermion/higgsino/gaugino):
pairs = {
    "vector multiplet (gauge + shadow)": b_pair(2, False) + b_pair(1, True),
    "chiral multiplet (Weyl + shadow)": b_pair(1, True) + b_pair(0, False),
    "Higgs multiplet (scalar + shadow)": b_pair(0, False) + b_pair(1, True),
}
for name, val in pairs.items():
    print(f"    {name:>36}: b_graded = {val}")
check("graded totals: vector pair = -3, chiral/Higgs pair = +1 (per T unit): the "
      "orbital -1/3's cancel PAIRWISE (opposite statistics), only the paramagnetic "
      "content survives -- b_graded = -3 C2 + T_f + T_H exactly",
      pairs["vector multiplet (gauge + shadow)"] == -3
      and pairs["chiral multiplet (Weyl + shadow)"] == 1
      and pairs["Higgs multiplet (scalar + shadow)"] == 1)
badd = {i: -3 * C2G[i] + Tf[i] + TH[i] - b2hdm[i] for i in (1, 2, 3)}
addf = {i: Fraction(1, 3) * Tf[i] + Fraction(2, 3) * TH[i] + Fraction(2, 3) * C2G[i]
        for i in (1, 2, 3)}
check("the completion 'add' == the SHADOW rows exactly: (1/3)T_f (sfermion) + "
      "(2/3)T_H (higgsino) + (2/3)C2 (gaugino) == b_graded - b_2HDM, all groups",
      all(badd[i] == addf[i] for i in (1, 2, 3)))

print("=" * 88)
print(" T-E  THE Q2 DECISION -- does the OBJECT supply the pairing? (computed)")
print("=" * 88)
def D_q(q):
    d = np.zeros((4, 6), complex)
    for e, (i, j, v) in enumerate(srs.EDGES):
        d[i, e] = -1.0
        d[j, e] = np.exp(1j * np.dot(q, v))
    return np.block([[np.zeros((4, 4)), d], [d.conj().T, np.zeros((6, 6))]])
GT = np.diag([1.0] * 4 + [-1.0] * 6)
rng = np.random.default_rng(17)
ok_pair, ok_flat = True, True
for _ in range(6):
    q = rng.uniform(-math.pi, math.pi, 3)
    D3 = D_q(q)
    ev, V = np.linalg.eigh(D3)
    for i in range(10):
        psi = V[:, i]
        par = float(psi.conj() @ GT @ psi)
        if abs(ev[i]) > 1e-8:
            phi = D3 @ psi / ev[i]
            # partner: normalized, SAME |energy| pairing via gamma_t psi (E -> -E) and
            # definite-parity split: use the SUSY-QM pair (psi_even, psi_odd) of D3^2:
            # project psi onto even/odd; both components are D3^2-eigenmodes at ev^2.
            pe = (psi + GT @ psi) / 2
            po = (psi - GT @ psi) / 2
            ne, no = np.linalg.norm(pe), np.linalg.norm(po)
            ok_pair &= ne > 1e-6 and no > 1e-6           # both parities present
            for comp, nrm in ((pe, ne), (po, no)):
                r = D3 @ D3 @ comp - ev[i] ** 2 * comp
                ok_pair &= np.linalg.norm(r) < 1e-8       # isospectral pair of D3^2
        else:
            # zero modes: parity-definite and D3-UNPAIRED
            ok_flat &= abs(abs(par) - 1) < 1e-9
check("every NONZERO mode is a SUSY-QM pair: its even and odd components are separate "
      "D3^2-eigenmodes at the same energy (the supercharge D3 maps one to the other; "
      "internal Cl(6) charges commute trivially -- different tensor factor): the "
      "shadow pairing EXISTS for all massive/cone content", ok_pair)
check("the ZERO modes (H1 flats + Gamma-extras) are parity-DEFINITE and D3-unpaired: "
      "the spatial complex supplies NO shadow for the gauge sector", ok_flat)
# the internal gauge directions are Cl-even and gamma_t-transparent (Wedderburn tie-in):
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402
g6 = AlgebraicUtility.cl6_generators()
biv = g6[0] @ g6[1]                                   # a grade-2 (gauge) direction
g5 = g6[0] @ g6[1] @ g6[2] @ g6[3] @ g6[4] @ g6[5]
check("internal gauge directions are grade-2 = chirality-preserving ([bivector, "
      f"gamma5] = 0: {np.max(np.abs(biv @ g5 - g5 @ biv)):.1e}) and act on the "
      "internal factor only (gamma_t-transparent by tensor structure)",
      np.max(np.abs(biv @ g5 - g5 @ biv)) < 1e-12)

print("=" * 88)
print(" VERDICT -- Q2 ANSWERED (mixed, computed):")
print("=" * 88)
print("""    THE FORMULA LAYER CLOSES: the (-11/3, 2/3, 1/3) structure is DERIVED from the
    heat kernel's two universal coefficients (1/12) tr Omega^2 + (1/2) tr E^2 with
    E = -2F.S -- validated on exact spectra (T-A/T-B), one unit normalization, two
    forced outcomes, component-level ghost bookkeeping agreeing (T-C). Seeley-DeWitt
    replaces "one-loop QFT" as the declared Type-3 import: the beta FORMULA now lives
    in the SAME spectral-action layer as zeta_{D4}(0). todo par.5's Layer-2 tag can be
    upgraded accordingly (wording user-gated).

    THE GRADING IS DERIVED WHERE THE OBJECT PAIRS: D3 is the supercharge; every
    massive/cone mode comes in charge-preserving opposite-parity pairs (T-E), and for
    paired content the orbital terms cancel exactly, leaving the N=1 totals (T-D).
    Conditional step, NAMED: form-parity <-> statistics (the KO 2->6 identification;
    A4/CAR gives the Cl factor fermionic statistics; equating the two gradings is the
    KO-theory step, not yet derived).

    THE GAUGE ROW ITSELF DOES NOT CLOSE -- and now we know exactly why: the spatial
    gauge modes ARE the flats, and the flats are D3-UNPAIRED (T-E; the same fact as
    T1's index/beta separation). The gaugino (2/3)C2 and higgsino (2/3)T_H shadows
    can only come from the TIME-LEG (gamma_t d_N) fluctuation complex -- which is
    UN-BUILT. That is the sharpened todo par.5 equation: build the time-leg
    fluctuation complex for the flat/Higgs sector; its graded a4 must supply
    (2/3)C2 + (2/3)T_H. The multiplet reading is NOT inherited: it is derived for
    paired content, and localized (not asserted) for the unpaired sector.

    KILLS: none fired on the formula/theorem layers; the gauge-row closure kill fired
    exactly where pre-registered (the object forces no spatial shadow for flats).
    beta VALUES unchanged (they were never in question); no prediction touched.""")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
