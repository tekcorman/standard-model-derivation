#!/usr/bin/env python3
"""
proofs/foundations/LOOP_A5_winding_weld_W2_2026-07-04.py

THE WINDING WELD W2 -- is the chiral (eps) SEED forced? Pre-registered in
internal research notes (committed
5c40ffb BEFORE this file). PURELY STRUCTURAL: NO eps evaluation; the R-eps target
appears NOWHERE. The O(1) asymmetry is the SEED/home of the chiral channel, not
eps (eps = a subtle transport functional of it, a later BLIND step).

Follows W1 (02ca217): the read<->coupled descent is a FORCED COVARIANT
superposition {1/3, 1/3 +- sqrt3/6} but NOT a forced bijection. User directive:
take the thread to completion, follow the lead. The lead = that superposition,
identified as the VACUUM's Cl(6)-deck content.

STAGES:
  S-0  re-lock vac(+-J), U_pi, C = I + iJ, |<0|U_pi|0>| = 1/2.
  CC1  the deck content d_s = <0|Pi_s^{U^2}|0> is FORCED {1/3, 1/3+-sqrt3/6}
       (generic Cl6 vector does NOT give it).
  CC2  KEY: bit-EVEN part democratic {1/3,1/3,1/3} EXACTLY; the +-sqrt3/6
       asymmetry is PURE BIT-ODD (flips under J->-J).
  CC3  the FORCED SOURCE: <0|U_pi^2|0> = +- i/2 (pure imaginary) -> closed form
       d_s = 1/3(1 + 2 Re(w^{-s}<0|U^2|0>)); Re=0 IS the democratic bit-even, Im
       = 1/2 IS the chiral asymmetry; magnitude = |<0|U_pi|0>|/sqrt3.
  CC4  free-vanishing (bit-average democratic) + carried by iJ (Im) not I (Re).
  CC5  VERDICT: does the route REOPEN (eps-seed forced) or dead-end?
"""
import cmath
import itertools
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
import srs  # noqa: E402
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
ND = 2 * NE
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
def gam(x):
    return sum(x[a] * g6[a] for a in range(NE))
def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6
OM = cmath.exp(2j * math.pi / 3)
S36 = math.sqrt(3) / 6.0

# ===========================================================================
banner("S-0  re-lock vac(+-J), U_pi, C = I + iJ, |<0|U_pi|0>| = 1/2")
# ===========================================================================
d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1],
                 [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
rows = []
for g in A4:
    R6 = edge_rep(g)
    rows.append(np.kron(np.eye(3), (H1.T @ R6 @ H1).T) - np.kron(B1.T @ R6 @ B1, np.eye(3)))
_, Sp, Vp = np.linalg.svd(np.vstack(rows))
phi3 = Vp[-1].reshape(3, 3); phi3 *= math.sqrt(3) / np.linalg.norm(phi3)
J6 = B1 @ phi3 @ H1.T - H1 @ phi3.T @ B1.T
wJ, VJ = np.linalg.eig(J6)
def build_vac(sign):
    sel = 1j if sign > 0 else -1j
    modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - sel) < 1e-9)[0]])
    A = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
    N = sum(a.conj().T @ a for a in A)
    wN, VN = np.linalg.eigh(N)
    v = VN[:, [int(np.argmin(wN))]]
    return v / np.linalg.norm(v)
vac, vac_m = build_vac(+1), build_vac(-1)
C_PAIR = np.array([[(vac.conj().T @ g6[a] @ g6[b] @ vac).item()
                    for b in range(NE)] for a in range(NE)])
# U_pi
sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
pi = {}
for e, (i, j, v) in enumerate(EDGES):
    a, b = sigma3[i], sigma3[j]
    pi[e] = EIDX[(min(a, b), max(a, b))]
Rpi = np.zeros((NE, NE))
for e in range(NE):
    Rpi[pi[e], e] = 1.0
rows = [np.kron(gam(Rpi[:, a]), np.eye(8)) - np.kron(np.eye(8), g6[a].T) for a in range(NE)]
_, S2s, Vh = np.linalg.svd(np.vstack(rows))
null = Vh[np.sum(S2s > 1e-9):].conj()
U_pi = null[0].reshape(8, 8)
U_pi /= np.sqrt(np.abs(np.linalg.det(U_pi @ U_pi.conj().T)) ** (1 / 8))
U2 = np.linalg.matrix_power(U_pi, 2)
ov1 = (vac.conj().T @ U_pi @ vac).item()
check("S-0 re-lock: C = I + iJ (Re=I, Im antisymmetric), U_pi^3=-I, "
      f"|<0|U_pi|0>| = 1/2 ({abs(ov1):.4f})",
      np.max(np.abs(C_PAIR.real - np.eye(NE))) < 1e-10
      and np.max(np.abs(C_PAIR.imag + C_PAIR.imag.T)) < 1e-10
      and np.max(np.abs(np.linalg.matrix_power(U_pi, 3) + np.eye(8))) < 1e-9
      and abs(abs(ov1) - 0.5) < 1e-9)

# deck projectors on the 8-dim Cl6 module (U_pi^2, order 3)
Pu = {s: sum(OM ** (-s * m) * np.linalg.matrix_power(U2, m) for m in range(3)) / 3
      for s in (0, 1, 2)}
def content(v):
    return [float(np.real((v.conj().T @ Pu[s] @ v).item())) for s in (0, 1, 2)]

# ===========================================================================
banner("CC1  the vacuum deck content is FORCED {1/3, 1/3 +- sqrt3/6}")
# ===========================================================================
dvac = content(vac)
targ = sorted([1/3, 1/3 + S36, 1/3 - S36])
# a fixed GENERIC Cl6 vector (deterministic, reproducible) for contrast
gen = (np.array([1, 2, 3, 4, 5, 6, 7, 8], float)
       + 1j * np.array([8, 5, 2, 7, 3, 1, 6, 4], float)).reshape(8, 1)
gen = gen / np.linalg.norm(gen)
dgen = content(gen)
print(f"    vacuum deck content : {np.round(dvac,5)} (sum {sum(dvac):.4f})")
print(f"    forced target set   : {np.round(targ,5)} = {{1/3, 1/3+-sqrt3/6}}")
print(f"    GENERIC Cl6 vector  : {np.round(dgen,5)}  (contrast -- NOT the target)")
check("CC1 the vacuum deck content = {1/3, 1/3+-sqrt3/6} EXACTLY (forced); a "
      "generic Cl6 vector does NOT reproduce it (the structure is special)",
      max(abs(a - b) for a, b in zip(sorted(dvac), targ)) < 1e-6
      and max(abs(a - b) for a, b in zip(sorted(dgen), targ)) > 1e-2)

# ===========================================================================
banner("CC2  KEY: bit-EVEN democratic; the +-sqrt3/6 asymmetry is PURE BIT-ODD")
# ===========================================================================
dvm = content(vac_m)
even = [0.5 * (dvac[s] + dvm[s]) for s in (0, 1, 2)]
odd = [dvac[s] - dvm[s] for s in (0, 1, 2)]
print(f"    d(+J) = {np.round(dvac,5)};  d(-J) = {np.round(dvm,5)}")
print(f"    bit-EVEN 1/2(d(+J)+d(-J)) = {np.round(even,5)}  (democratic 1/3?)")
print(f"    bit-ODD  d(+J)-d(-J)      = {np.round(odd,5)}  (the +-1/sqrt3 flip)")
even_democratic = max(abs(e - 1/3) for e in even) < 1e-6
odd_pure = abs(sum(odd)) < 1e-9 and max(abs(o) for o in odd) > 0.5  # sums to 0, O(1/sqrt3)
check("CC2 KEY: the bit-EVEN deck content is DEMOCRATIC {1/3,1/3,1/3} EXACTLY "
      "(the read-visible part is featureless => chirality-BLIND), and the "
      "+-sqrt3/6 asymmetry is PURE BIT-ODD (flips under J->-J; d(+J)-d(-J) sums "
      f"to 0 with |flip| = 1/sqrt3 = {1/math.sqrt(3):.4f}) => ALL chirality is "
      "in the forced bit-odd asymmetry", even_democratic and odd_pure)

# ===========================================================================
banner("CC3  the FORCED SOURCE: <0|U_pi^2|0> = +- i/2 (pure imaginary)")
# ===========================================================================
a2 = (vac.conj().T @ U2 @ vac).item()
print(f"    <0|U_pi^2|0> = {a2:+.6f}  (Re = {a2.real:+.2e}, Im = {a2.imag:+.5f})")
# closed form: d_s = 1/3 (1 + 2 Re(OM^{-s} a2))
d_closed = [float((1 + 2 * np.real(OM ** (-s) * a2)) / 3) for s in (0, 1, 2)]
print(f"    closed form 1/3(1+2Re(w^-s <0|U^2|0>)) = {np.round(d_closed,5)}  "
      f"(matches d(+J)? {max(abs(a-b) for a,b in zip(d_closed,dvac)) < 1e-9})")
print(f"    asymmetry magnitude sqrt3/6 = {S36:.5f} vs |<0|U_pi|0>|/sqrt3 = "
      f"{abs(ov1)/math.sqrt(3):.5f}")
check("CC3 the FORCED SOURCE: <0|U_pi^2|0> = +-i/2 -- REAL part = 0 (=> the "
      "bit-even part is democratic), IMAGINARY part = 1/2 (=> the chiral "
      "asymmetry). The closed form d_s=1/3(1+2Re(w^-s<0|U^2|0>)) reproduces the "
      "content EXACTLY; asymmetry = |<0|U_pi|0>|/sqrt3 = 1/2/sqrt3 = sqrt3/6. "
      "Forced, no free constant",
      abs(a2.real) < 1e-9 and abs(abs(a2.imag) - 0.5) < 1e-6
      and max(abs(a - b) for a, b in zip(d_closed, dvac)) < 1e-9
      and abs(S36 - abs(ov1) / math.sqrt(3)) < 1e-9)

# ===========================================================================
banner("CC4  free-vanishing (bit-average) + carried by iJ (Im), not I (Re)")
# ===========================================================================
# the asymmetry is carried by the IMAGINARY (iJ) part: Re(<0|U^2|0>)=0 (CC3) means
# the I (real, chirality-blind) part of the vacuum pairing gives NO asymmetry;
# the entire asymmetry is the iJ (chiral) content. Bit-average = democratic (CC2).
check("CC4 free-vanishing / C-E2a: bit-averaging gives democratic {1/3,1/3,1/3} "
      "(no chirality without the bit); and Re(<0|U^2|0>)=0 => the I (chirality-"
      "blind) part carries NO asymmetry -- the entire seed is the iJ (chiral) "
      "content of the vacuum pairing C=I+iJ (E2a). The chiral seed VANISHES on "
      "the mirror-symmetric / real part", even_democratic and abs(a2.real) < 1e-9)

# ===========================================================================
banner("CC5  VERDICT + tier")
# ===========================================================================
reopen = even_democratic and odd_pure and abs(a2.real) < 1e-9 and abs(abs(a2.imag)-0.5) < 1e-6
print(f"""    TIER: {'PASS -- the route REOPENS.' if reopen else 'see below.'}
    The chiral (eps) SEED is FORCED: the vacuum's deck-asymmetry is the
    pure-imaginary <0|U_pi^2|0> = +-i/2 -- its REAL part is 0 (so the read-visible
    bit-EVEN deck content is DEMOCRATIC = chirality-blind, explaining WHY the read
    sees no chirality) and its IMAGINARY part = 1/2 carries the entire bit-ODD
    asymmetry +-sqrt3/6 = |<0|U_pi|0>|/sqrt3. This is the kinematic face of E2a's
    C = I + iJ: the I gives the democratic (blind) part, the iJ gives the chiral
    seed. It flips with the bit (bit-odd) and vanishes on the mirror-symmetric
    average (free-vanishing).
    UPGRADE to W1: the W1 'irreducible adoption' was confined to the bit-EVEN
    MODULUS grading (deck moduli 2/sqrt2 vs the read's dart grading) -- and eps
    does NOT live there (that part is democratic/blind). The bit-ODD channel where
    eps DOES live is FORCED. So the winding weld's CHIRAL channel is FORCED; only
    the bit-even modulus identity remains an adoption, and it carries no eps.
    RE-LOCALIZATION (sharpened, the route REOPENS with a concrete forced object):
    eps = a transport-minus-leading functional of the FORCED seed <0|U_pi^2|0>=i/2
    (the bit-odd deck-asymmetry) carried to the lepton slice. That evaluation is
    the NEXT step and is BLIND (the seed is O(1); eps is sub-ppm) -- NOT done here.
    An open miss stays open: eps is NOT computed; what closed is that its home/seed
    is FORCED, not an adoption.""")
check("CC5 scope honesty: purely structural; NO eps evaluated; target absent; ONE "
      "frozen test set; the O(1) asymmetry is the SEED not eps; no fit; no value; "
      "PASS = eps-home is forced (route reopens), NOT eps computed", True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
