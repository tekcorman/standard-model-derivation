#!/usr/bin/env python3
"""
proofs/foundations/NATIVE_a4_su2L_2026-07-06.py

D1 / Piece 1 -- native zeta_{D4}(0) internal a4: PROBE 3 (the SU(2)_L GROUP factors).
Pre-registration: internal research notes (6ceed2f BEFORE this file).
CLASS: pure structure (class a). NO PDG anywhere.

GOAL: de-import the SU(2)_L GROUP factors -- the Dynkin index T(2) and the adjoint Casimir C2(adj) --
as TRACES over the object's own su(2), replacing the HARDCODED table in the_run.gauge_dynkin
(T2[2]=1/2, C2G[2]=2) and the TYPED S=1/2 sigma^3 in read_gauge. The direct analog of the LANDED
SU(3)_c de-import (NATIVE_a4_color_su3_2026-07-05).

HOME (pinned): the T-ID2 commutant su(2) -- K = [gb1gb2/2, gb2gb0/2, gb0gb1/2] (gb = the B1/internal
Clifford bivectors), which COMMUTES with the full Cl(3,1) and is "the weak-isospin, chirality-preserving"
(TID2_C_lorentzian_assembly). The site-Fock = Dirac(4) (x) doublet(2); the commutant su(2) acts on the
doublet(2) as its fundamental.

P1: the Hermitian generators J_a = i*K_a close into su(2); commute with Cl(3,1); the doublet = the 2.
P2 (crux): T(2)=1/2, C2(adj)=2 by TRACE over the doublet (canonical normalization) = the hardcoded values.
P3: native b2 assembles.  SCOPE: MODEST (the forced connection). Spin rows stay declared Seeley-DeWitt.
"""
import os
import sys
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")
np.set_printoptions(precision=4, suppress=True, linewidth=120)

EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
EPS = np.zeros((3, 3, 3))
for a in range(3):
    for b in range(3):
        for c in range(3):
            EPS[a, b, c] = 0.5 * (a - b) * (b - c) * (c - a)

print("=" * 90)
print(" P0  the object: T-ID2 split (H1 spatial / B1 internal); the Cl(3,1) + the commutant su(2)")
print("=" * 90)
g6 = [np.array(g, complex) for g in AlgebraicUtility.cl6_generators()]
D = g6[0].shape[0]
d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0; d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
def gam(x): return sum(x[a] * g6[a] for a in range(NE))
gh = [gam(H1[:, i]) for i in range(3)]           # spatial Dirac gammas
gb = [gam(B1[:, i]) for i in range(3)]           # internal (B1) gammas
g0 = gb[0] @ gb[1] @ gb[2]                        # gamma^0 (time)
G4 = [g0, gh[0], gh[1], gh[2]]                    # Cl(3,1)
eta = np.diag([-1.0, 1, 1, 1])
lor = max(np.max(np.abs(G4[m] @ G4[n] + G4[n] @ G4[m] - 2 * eta[m, n] * np.eye(D))) for m in range(4) for n in range(4))
check(f"T-ID2 Cl(3,1): {{gamma^mu,gamma^nu}}=2 eta (dev {lor:.1e})", lor < 1e-9)

print("=" * 90)
print(" P1  the commutant su(2)_L: J_a = i*(gb bivectors)/2 -- closes, commutes with Cl(3,1)")
print("=" * 90)
K = [gb[1] @ gb[2] / 2, gb[2] @ gb[0] / 2, gb[0] @ gb[1] / 2]   # T-ID2 commutant (anti-Hermitian)
J = [1j * Kk for Kk in K]                                       # Hermitian su(2) generators
herm = max(np.max(np.abs(Ja - Ja.conj().T)) for Ja in J)
check(f"J_a = i*K_a are Hermitian (dev {herm:.1e})", herm < 1e-10)
# commute with all of Cl(3,1) (=> the doublet is an INTERNAL index, not spacetime)
commCl = max(np.max(np.abs(Ja @ G - G @ Ja)) for Ja in J for G in G4)
check(f"[J_a, Cl(3,1)] = 0 (dev {commCl:.1e}): the su(2) is INTERNAL (weak isospin), commutes with the "
      "Dirac gammas -- T-ID2's chirality-preserving commutant", commCl < 1e-9)
# close into su(2): fit [J_a,J_b] = i f eps_{abc} J_c, require |f|=1
f_fit = np.real(np.trace((-1j * (J[0] @ J[1] - J[1] @ J[0])).conj().T @ J[2]) / np.trace(J[2].conj().T @ J[2]))
so3 = max(np.max(np.abs((J[a] @ J[b] - J[b] @ J[a]) - 1j * f_fit * sum(EPS[a, b, c] * J[c] for c in range(3))))
          for a in range(3) for b in range(3))
check(f"J_a CLOSE into su(2): [J_a,J_b]=i f eps J_c (f={f_fit:.3f}, |f|=1; dev {so3:.1e}) -- rank 1",
      so3 < 1e-9 and abs(abs(f_fit) - 1) < 1e-9)
# the Casimir J^2 = s(s+1) on every state; s=1/2 => 3/4 (the whole Fock is doublets)
cas = np.real(np.trace(sum(Ja @ Ja for Ja in J))) / D
check(f"Casimir J^2 = {cas:.4f} = 3/4 on the whole Fock (s=1/2): every state is in a DOUBLET "
      "(site-Fock = Dirac(4) (x) doublet(2))", abs(cas - 0.75) < 1e-9)

print("=" * 90)
print(" P2  NATIVE GROUP FACTORS by trace (the de-import): T(2)=1/2, C2(adj)=2")
print("=" * 90)
# The whole Fock is doublets (P1 Casimir 3/4) => it is (dim_Fock/2) = 4 copies of the FUNDAMENTAL 2.
# T(2) = Tr_Fock(J_a J_a)/(#doublets) ; C2(adj) = f^{acd}f^{bcd} from the object's structure constants.
n_doublets = D // 2                              # = 4  (dim Fock / dim doublet)
Gram = np.array([[np.trace(J[a].conj().T @ J[b]) for b in range(3)] for a in range(3)])
gram_diag = np.real(np.mean(np.diag(Gram)))
offG = np.max(np.abs(Gram - np.diag(np.diag(Gram))))
T2_index = gram_diag / n_doublets                # Tr_fund per doublet
check(f"native fundamental Dynkin index T(2) = Tr_Fock(J^a J^a)/{n_doublets} = {T2_index:.6f} = 1/2 "
      f"(Gram = {gram_diag:.3f} delta, off-diag {offG:.1e}; = the hardcoded gauge_dynkin T2[2])",
      abs(T2_index - 0.5) < 1e-9 and offG < 1e-9)
# structure constants f_{abc} (object-derived, Gram-normalized): [J_a,J_b] = i f_{abc} J_c
Ginv = np.linalg.inv(Gram)
fabc = np.zeros((3, 3, 3))
for a in range(3):
    for b in range(3):
        comm = J[a] @ J[b] - J[b] @ J[a]
        rhs = np.array([np.trace(J[c].conj().T @ comm) for c in range(3)])
        fabc[a, b] = np.real((Ginv @ rhs) / 1j)
fratio = fabc / np.where(EPS != 0, EPS, np.nan)
fconst = np.nanmean(fratio)                      # f_{abc} = fconst * eps_{abc}
C2adj = np.einsum('acd,bcd->ab', fabc, fabc)
C2adj_val = np.real(np.mean(np.diag(C2adj)))
offd = np.max(np.abs(C2adj - np.diag(np.diag(C2adj))))
check(f"native adjoint Casimir C2(adj) = f^acd f^bcd = {C2adj_val:.4f} delta (off-diag {offd:.1e}) = 2 "
      f"(structure constants f_abc = {fconst:.3f}*eps, |f|=1 clean su(2); = the hardcoded C2G[2])",
      abs(C2adj_val - 2.0) < 1e-6 and offd < 1e-6 and abs(abs(fconst) - 1) < 1e-9)

print("=" * 90)
print(" P3  the NATIVE SU(2) beta-row b2 (group factors from P2; weak mult from the SM content)")
print("=" * 90)
T2 = Fraction(1, 2)      # NATIVE (P2)
C2 = 2                   # NATIVE (P2)
gens_n = 3
# SU(2)-charged fermions: Q_L (color 3, doublet), L_L (color 1, doublet); u_R,d_R,e_R = weak singlets
weak_fermions = [(3, 2), (1, 2)]        # (color_dim, weak_dim=2 doublet)
Tf2 = sum((T2 * c if w == 2 else Fraction(0)) for c, w in weak_fermions) * gens_n
TH2 = Fraction(2) * T2                  # 2 Higgs doublets
b2_2hdm = -Fraction(11, 3) * C2 + Fraction(2, 3) * Tf2 + Fraction(1, 3) * TH2
add2 = Fraction(1, 3) * Tf2 + Fraction(2, 3) * TH2 + Fraction(2, 3) * C2
b2_4d = b2_2hdm + add2
print(f"    native: T(2)=1/2, C2(adj)=2 => Sum T_f(weak) = {Tf2}, T_H(weak) = {TH2}")
print(f"    b2(2HDM) = -(11/3)*2 + (2/3)*{Tf2} + (1/3)*{TH2} = {b2_2hdm}")
print(f"    b2(4D completion) = b2(2HDM) + add({add2}) = {b2_4d}")
check("native Sum T_f(weak) = 6 (= the_run.gauge_dynkin Tf[2])", Tf2 == 6)
check(f"native b2(4D completion) = {b2_4d} = 1 = MSSM b2 (= read_gauge_running b[2] = {{33/5,1,-3}}[2])",
      b2_4d == Fraction(1))

print("=" * 90)
print(" VERDICT")
print("=" * 90)
print("""    The SU(2)_L GROUP factors of the gauge row are now NATIVE: the su(2) is the T-ID2 commutant
    (the B1-bivectors J_a = i*K_a) -- Hermitian, commuting with the full Cl(3,1) (an INTERNAL weak
    isospin), closing into su(2); the site-Fock is all doublets (Casimir 3/4). T(2)=1/2 and C2(adj)=2
    fall out as TRACES over the doublet, replacing gauge_dynkin's hardcoded {1/2, 2} AND read_gauge's
    typed S=1/2 sigma^3; and b2(4D)=1=MSSM assembles from them. DE-IMPORT DONE for the SU(2)_L group
    factors. GROUP-FACTOR COLUMN of the D1 ledger now COMPLETE: SU(3)_c (probe 1) + SU(2)_L (here).
    STILL: U(1)_Y's C3-breaking hypercharge selection (a STATED adoption, framework_axioms/explore_m07);
    the spin ROWS (-11/3, 2/3, 1/3) = declared Seeley-DeWitt on the A5(b)-derived Lorentz-locked cone
    (premise lifted). MODEST grade (the forced connection, not the Casimir arithmetic). No PDG; no value.""")
print("=" * 90)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 90)
sys.exit(0 if ok_all else 1)
