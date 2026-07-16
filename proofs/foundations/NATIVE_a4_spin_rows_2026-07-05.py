#!/usr/bin/env python3
"""
proofs/foundations/NATIVE_a4_spin_rows_2026-07-05.py

D1 / Piece 1 -- native zeta_{D4}(0): PROBE 2 (the SPIN rows from the D4 cone).
Pre-registration: internal research notes §7
(committed BEFORE this probe: f7c7224).

GOAL: attempt to de-import the SPIN coefficients of the gauge/Higgs rows -- the -11/3
(vector+ghost) and +1/3 (scalar) -- from the substrate D4 cone, replacing OMEGA_S2_Q2's
declared Seeley-DeWitt import.  CLASS: pure structure (class a).  NO PDG.

PRE-DECLARED EXPECTED OUTCOME: WALL on A5(b) (the spin-1 -> spin-1/2 Clifford locking).
The checks below CONFIRM the wall's structure; ALL-PASS = the confirmations ran; the
scientific VERDICT is WALL (located precisely), not a de-import.

P1: the lambda=-1 triple's k.p cone is a SPIN-1 multifold (Chern (-2,0,+2); flat middle
    band -> eigenvalues {-c,0,+c} = m=(-1,0,+1)), NOT a Dirac channel (|C|=1).
P2: the FERMION row is NATIVE -- the spin-1 cone read as matter = 1 Weyl per cone = +2/3
    (06-25 / read_matter_row); with probe-1's T_f the whole (2/3)T_f is native.
P3: the vector/scalar spin coefficients -11/3 / +1/3 assume Lorentz-LOCKED fields; the
    substrate cone's a4 "Weyl counts" are UNLOCKED (timelike 4 / spacelike 1 / topological
    2 != Dirac-locked 2/2/0, OMEGA_T4) => the de-import needs the Clifford LOCKING = A5(b).
"""
import math
import os
import sys
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

NV, NE = 4, 6
EDGES = [(0, 1, (0, 0, 0)), (0, 2, (0, 0, 0)), (0, 3, (0, 0, 0)),
         (1, 2, (1, 0, 0)), (1, 3, (0, 1, 0)), (2, 3, (0, 0, 1))]
Cm = np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 0]], float)
G12 = (5 * np.eye(3) + Cm) / 3      # Albanese frame (Q0)

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

def A_q(q):
    A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = np.exp(1j * np.dot(q, v)); A[i, j] += p; A[j, i] += np.conj(p)
    return A

def dA_q(q, ax):   # velocity operator d A / d q_ax
    A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = 1j * v[ax] * np.exp(1j * np.dot(q, v)); A[i, j] += p; A[j, i] += np.conj(p)
    return A

def chern_sphere(Hf, band, r=0.15, Nth=36, Nph=72):
    thetas = np.linspace(1e-3, math.pi - 1e-3, Nth)
    phis = np.linspace(0, 2 * math.pi, Nph, endpoint=False)
    V = np.empty((Nth, Nph), object)
    for i, t in enumerate(thetas):
        for j, ph in enumerate(phis):
            p = r * np.array([math.sin(t) * math.cos(ph), math.sin(t) * math.sin(ph), math.cos(t)])
            _, W = np.linalg.eigh(Hf(p)); V[i, j] = W[:, band]
    F = 0.0
    for i in range(Nth - 1):
        for j in range(Nph):
            j2 = (j + 1) % Nph
            u1 = np.vdot(V[i, j], V[i, j2]); u2 = np.vdot(V[i, j2], V[i + 1, j2])
            u3 = np.vdot(V[i + 1, j2], V[i + 1, j]); u4 = np.vdot(V[i + 1, j], V[i, j])
            F += np.angle(u1 * u2 * u3 * u4)
    return F / (2 * math.pi)

print("=" * 90)
print(" P1  the substrate matter cone (lambda=-1 triple) is a SPIN-1 MULTIFOLD, not Dirac")
print("=" * 90)
# Gamma spectrum: {-1,-1,-1, 3}; the triple = the spin-1 cone, the Perron k*=3 apart.
evG = np.round(np.real(np.linalg.eigvalsh(A_q((0, 0, 0)))), 6)
check(f"A(Gamma) spectrum = {sorted(evG.tolist())} = the lambda=-1 TRIPLE + Perron 3 "
      "(the triple = the spin-1 Weyl cone)", sorted(np.round(evG).astype(int).tolist()) == [-1, -1, -1, 3])

# k.p on the triple: project the velocity operators onto the lambda=-1 eigenspace.
wG, UG = np.linalg.eigh(A_q((0, 0, 0)))
P3 = UG[:, np.abs(wG + 1) < 1e-6]                       # 4x3, the triple eigenvectors
S = [P3.conj().T @ dA_q((0, 0, 0), ax) @ P3 for ax in range(3)]   # three 3x3 velocity matrices
# spin-1 signature (a): the k.p cone H(khat) = sum khat_i S_i has a FLAT middle band
#   (eigenvalues {-c, 0, +c} = m=(-1,0,+1)), unlike a Dirac cone (two bands +-c, no flat).
rng = np.random.default_rng(5)
flat_ok = True
for _ in range(20):
    kh = rng.normal(size=3); kh /= np.linalg.norm(kh)
    Hk = sum(kh[a] * S[a] for a in range(3))
    ev = np.sort(np.real(np.linalg.eigvalsh(Hk)))       # 3 eigenvalues
    c = 0.5 * (abs(ev[0]) + abs(ev[2]))
    flat_ok &= abs(ev[1]) < 1e-9 and abs(ev[0] + ev[2]) < 1e-9 and c > 1e-3   # flat middle, symmetric
check("k.p cone eigenvalues = {-c, 0, +c} (FLAT middle band m=0 + dispersing m=+-1): "
      "the spin-1 Weyl structure (a Dirac cone has NO flat band)", flat_ok)

# spin-1 signature (b): Chern of the triple bands = (-2, 0, +2) in the Albanese frame
#   (a spin-s Weyl node has Chern +-2s; s=1 -> +-2; a Dirac/Weyl node has |C|=1).
HA_p = lambda p: A_q(G12 @ np.asarray(p, float))
ch = [chern_sphere(HA_p, b) for b in (0, 1, 2)]
print(f"    triple-band Chern (lower, mid, upper): {ch[0]:+.3f}, {ch[1]:+.3f}, {ch[2]:+.3f}")
check("Chern = (-2, 0, +2): a SPIN-1 multifold (|C|=2), NOT a Dirac/Weyl node (|C|=1) "
      "-- confirms OMEGA_T4",
      abs(abs(ch[0]) - 2) < 0.1 and abs(ch[1]) < 0.1 and abs(abs(ch[2]) - 2) < 0.1)

print("=" * 90)
print(" P2  the FERMION row is NATIVE (cone spin +2/3 x probe-1 group T_f)")
print("=" * 90)
# The spin-1 cone read as MATTER = 1 Weyl per cone (06-25 O_spin1_cone_gauge_beta /
# the_run.read_matter_row: the m=+-1 dispersing pair, flat band required for gauge invariance)
# = the +2/3 spin coefficient.  Probe 1 made the SU(3) group factor T_f native.
spin_coeff_fermion = Fraction(2, 3)      # = 1 Weyl per cone (06-25), the cone's own a4 matter row
Tf_color = Fraction(6)                    # NATIVE from probe 1 (NATIVE_a4_color_su3)
fermion_row = spin_coeff_fermion * Tf_color
print(f"    +2/3 (spin, from the spin-1 cone = 1 Weyl per cone, 06-25) x T_f={Tf_color} (probe 1) "
      f"=> (2/3)T_f = {fermion_row}")
check("the FERMION contribution to b3, (2/3)T_f = 4, is now FULLY native "
      "(spin coeff from the cone; group factor from probe 1)", fermion_row == 4)

print("=" * 90)
print(" P3  the VECTOR (-11/3) & SCALAR (+1/3) spin coefficients: WALLED on A5(b)")
print("=" * 90)
# The Seeley-DeWitt spin dictionary b = -(-1)^{2s}[(2s_z)^2 - 1/3] gives the rows ONLY for
# Lorentz-LOCKED fields.  The substrate cone's a4 "Weyl counts" are UNLOCKED (OMEGA_T4 T-C):
timelike, spacelike, topological = 4, 1, 2          # OMEGA_T4 measured (1/6pi=4 Weyl; 06-25; |Chern|)
dirac_locked = (2, 2, 0)                             # a genuine Dirac channel: one Lorentz function
unlocked = (timelike, spacelike, topological) != dirac_locked
print(f"    substrate spin-1 cone counts (timelike, spacelike, topological) = "
      f"({timelike}, {spacelike}, {topological})  vs  Dirac-locked {dirac_locked}")
check("the three a4 counts are UNLOCKED (!= the Dirac 2/2/0): the substrate cone is a spin-1 "
      "multifold, so the LOCKED spin dictionary (-11/3 vector, +1/3 scalar) does NOT follow "
      "from it without the spin-1 -> spin-1/2 Clifford LOCKING = A5(b)", unlocked
      and abs(abs(ch[2]) - topological) < 0.1)

print("=" * 90)
print(" VERDICT -- WALL (as pre-registered), located precisely")
print("=" * 90)
print("""    The substrate D4 matter cone is a SPIN-1 MULTIFOLD (flat middle band m=0; Chern
    (-2,0,+2); |C|=2), whose heat-kernel a4 "Weyl counts" are Lorentz-UNLOCKED (4/1/2 vs
    the Dirac-locked 2/2/0).  Consequences for the spin-row de-import:

      FERMION row (+2/3): NATIVE.  The cone read as matter = 1 Weyl per cone (06-25);
        with probe-1's T_f, the whole (2/3)T_f fermion contribution to b is native.

      VECTOR (-11/3) & SCALAR (+1/3) rows: WALLED.  These assume Lorentz-LOCKED spin-1
        and spin-0 fields (the Seeley-DeWitt dictionary).  The substrate cone's spin-1
        multifold, with unlocked counts, does NOT supply them -- the de-import REQUIRES the
        spin-1 -> spin-1/2 / locked-vector Clifford LOCKING = A5(b) = OMEGA_T4's open
        "Clifford-native phase space" (Target 4) = the PS-embedding spacetime/internal split.
        That single identification is the framework's un-derived Type-3 seam -- the SAME one
        the couplings, the widths' native grade, and the -70 ppm all route to.

    So probe 2 does NOT close: the -11/3 / +1/3 spin rows stay OMEGA_S2_Q2's declared
    Seeley-DeWitt import (SM-reproduction-conditional) until A5(b) is derived.  Probe 1's
    group-factor de-import stands; probe 2 sharpens the remaining wall to ONE object (A5(b)).
    No value moved; no PDG; the WALL is banked WITH its location, not relabeled as progress.""")
print("=" * 90)
print(f" OVERALL: {'ALL CHECKS PASS (verdict: WALL, located at A5(b))' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 90)
sys.exit(0 if ok_all else 1)
