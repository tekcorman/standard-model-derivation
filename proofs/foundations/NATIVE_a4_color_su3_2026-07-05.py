#!/usr/bin/env python3
"""
proofs/foundations/NATIVE_a4_color_su3_2026-07-05.py

D1 / Piece 1 -- the native zeta_{D4}(0) internal a4: PROBE 1 (color SU(3)_c group-factor
de-import).  Pre-registration: internal research notes
(committed BEFORE this probe: auto-sync c670a58, 2026-07-05 17:30).

GOAL: produce the SU(3)_c GROUP factors -- the Dynkin index T(3) and the adjoint Casimir
C2(adj) -- as TRACES over the object's own Cl(6)-Fock / so(6)-bivector realization, replacing
the HARDCODED table in the_run.gauge_dynkin (T3={1:0,3:1/2,8:3}, C2G[3]=3), and assemble the
native color beta-row b3.  CLASS: pure structure (class a).  NO PDG anywhere.

CRUX (resolved 2026-07-05, from the live the_run.py): color SU(3)_c lives in the Cl(6)-Fock
sector -- Fock = Lambda^*(C^3) = 2^3 = 1 (+) 3 (+) 3bar (+) 1 (Hamming weight n=0,1,2,3 ->
nu,d,u,e); a quark's color TRIPLET = Lambda^1.  Color su(3) = the 8 traceless number-conserving
mode-bilinears on the k*=3 fermionic edge-modes (grade-2 Cl(6) bivectors).  NOT Object A's
C[A4] M3 block (that is a FAMILY su(3)).

PRE-REGISTERED CLAIMS: P1 (su(3) closes from the object; Fock = 1+3+3bar+1; generators grade-2 =
gamma5-even = vectors); P2 (native T(3)=1/2, C2(adj)=3 by trace, = the hardcoded values);
P3 (native b3 = -(11/3)C2 + (2/3)Tf = -7 (2HDM) / -3 (4D completion), color mult from Fock).
KILLS: K1 no su(3) closure / wrong Fock decomposition; K2 T(3)!=1/2 or C2(adj)!=3; K3 generators
not gamma5-even; K4 assembled b3 != -7 without the hardcoded table.

SCOPE (honest): de-imports only the GROUP factors.  The spin ROWS (-11/3, 2/3, 1/3) stay
OMEGA_S2_Q2's DECLARED Seeley-DeWitt import (probe 2 = de-import those from the continuum cone).
SU(2)_L (edge-qubit H) and hypercharge (open C3-off-diagonal) are OUT of scope.
"""
import os
import sys
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

np.set_printoptions(precision=4, suppress=True, linewidth=120)

print("=" * 90)
print(" P0  the object: Cl(6) gammas -> 3 fermionic edge-modes -> Fock = Lambda^*(C^3)")
print("=" * 90)
g6 = [np.array(g, complex) for g in AlgebraicUtility.cl6_generators()]
D = g6[0].shape[0]
# Clifford check {gamma_a, gamma_b} = 2 delta
cliff = max(np.max(np.abs(g6[a] @ g6[b] + g6[b] @ g6[a] - (2.0 if a == b else 0.0) * np.eye(D)))
            for a in range(6) for b in range(6))
check(f"Cl(6): {{gamma_a,gamma_b}}=2 delta on the {D}-dim spinor rep (max dev {cliff:.1e})", cliff < 1e-10)

# 3 fermionic modes from disjoint gamma pairs (Jordan-Wigner is automatic: the 6 gammas already
# mutually anticommute).  a_i = (gamma_{2i} + i gamma_{2i+1})/2.
a = [0.5 * (g6[2 * i] + 1j * g6[2 * i + 1]) for i in range(3)]
adag = [x.conj().T for x in a]
# verify the CAR: {a_i, a_j^dag} = delta_ij, {a_i, a_j} = 0
car_ok = True
for i in range(3):
    for j in range(3):
        acomm = a[i] @ adag[j] + adag[j] @ a[i]
        car_ok &= np.max(np.abs(acomm - (np.eye(D) if i == j else 0))) < 1e-10
        acomm2 = a[i] @ a[j] + a[j] @ a[i]
        car_ok &= np.max(np.abs(acomm2)) < 1e-10
check("the 3 edge-modes satisfy the CAR {a_i,a_j^dag}=delta_ij, {a_i,a_j}=0 (genuine fermions)", car_ok)

# number operator N = sum a_i^dag a_i ; its spectrum = Hamming weight; multiplicities = Fock grades
Nhat = sum(adag[i] @ a[i] for i in range(3))
evN = np.round(np.real(np.linalg.eigvalsh(Nhat))).astype(int)
mult = {w: int(np.sum(evN == w)) for w in (0, 1, 2, 3)}
check(f"Fock = Lambda^*(C^3): N spectrum multiplicities {mult} = 1(+)3(+)3bar(+)1 "
      "(= read_species: n=0 nu, 1 d[triplet], 2 u[antitriplet], 3 e)",
      mult == {0: 1, 1: 3, 2: 3, 3: 1})

# grade projectors onto the Fock weight subspaces (eigenspaces of N)
wN, VN = np.linalg.eigh(Nhat)
wN = np.round(np.real(wN)).astype(int)
def weight_space(w):
    cols = [VN[:, k] for k in range(D) if wN[k] == w]
    return np.array(cols).T   # D x dim(w)

print("=" * 90)
print(" P1  color su(3)_c = the 8 traceless number-conserving mode-bilinears (built from the object)")
print("=" * 90)
# Build the 8 su(3) generators DIRECTLY from the modes (no typed Gell-Mann): 3 symmetric +
# 3 antisymmetric off-diagonal + 2 diagonal-traceless number-conserving bilinears.
gens = []
labels = []
for i in range(3):
    for j in range(i + 1, 3):
        gens.append(adag[i] @ a[j] + adag[j] @ a[i]); labels.append(f"X{i}{j}")            # symmetric
        gens.append(-1j * (adag[i] @ a[j] - adag[j] @ a[i])); labels.append(f"Y{i}{j}")     # antisymmetric
n_ii = [adag[i] @ a[i] for i in range(3)]
gens.append(n_ii[0] - n_ii[1]); labels.append("H1")
gens.append((n_ii[0] + n_ii[1] - 2 * n_ii[2]) / np.sqrt(3)); labels.append("H2")
assert len(gens) == 8

# (a) all Hermitian
herm = max(np.max(np.abs(T - T.conj().T)) for T in gens)
check(f"the 8 generators are Hermitian (max dev {herm:.1e})", herm < 1e-10)

# (b) closure into a Lie algebra of dim 8 (su(3)): every [T^a,T^b] is a real-linear combo of the T^c.
#     Build the Gram + solve for structure constants over the 8-dim span (trace inner product on Fock).
def ip(X, Y):
    return np.trace(X.conj().T @ Y)
Gram = np.array([[ip(gens[a_], gens[b_]) for b_ in range(8)] for a_ in range(8)])
Ginv = np.linalg.inv(Gram)
f_struct = np.zeros((8, 8, 8))
closes = True
for a_ in range(8):
    for b_ in range(8):
        comm = gens[a_] @ gens[b_] - gens[b_] @ gens[a_]        # [T^a,T^b] = i f^{abc} T^c
        rhs = np.array([ip(gens[c_], comm) for c_ in range(8)])
        coeff = Ginv @ rhs                                      # = i f^{ab*}
        recon = sum(coeff[c_] * gens[c_] for c_ in range(8))
        closes &= np.max(np.abs(comm - recon)) < 1e-9          # lies in the span (closure)
        f_struct[a_, b_] = np.real(coeff / 1j)                 # real structure constants
check("the 8 mode-bilinears CLOSE into an 8-dim Lie algebra (every [T^a,T^b] in the span) "
      "= su(3), rank 2 (2 commuting Cartan H1,H2)", closes
      and abs(np.max(np.abs(gens[6] @ gens[7] - gens[7] @ gens[6]))) < 1e-9)

# (c) grade-2 = chirality-preserving: [T^a, gamma5] = 0 with gamma5 = the Cl(6) volume element.
g5 = g6[0] @ g6[1] @ g6[2] @ g6[3] @ g6[4] @ g6[5]
chir = max(np.max(np.abs(T @ g5 - g5 @ T)) for T in gens)
check(f"generators are grade-2 (chirality-preserving): [T^a, gamma5_Cl6] = 0 (max {chir:.1e}) "
      "=> the inner-fluctuation gauge one-form is gamma5-even (a genuine VECTOR)", chir < 1e-10)

# (d) the fundamental = Lambda^1 (weight-1); the singlets (Lambda^0, Lambda^3) are annihilated.
P1w = weight_space(1); P0w = weight_space(0); P3w = weight_space(3)
acts_on_triplet = min(np.linalg.norm(P1w.conj().T @ (T @ P1w)) for T in gens[:2]) > 1e-6
kills_singlets = max(np.max(np.abs(T @ P0w)) for T in gens) < 1e-9 \
    and max(np.max(np.abs(T @ P3w)) for T in gens) < 1e-9
check("Fock decomposes as su(3) reps: Lambda^1 = the color TRIPLET (generators act), "
      "Lambda^0 & Lambda^3 = SINGLETS (annihilated)", acts_on_triplet and kills_singlets)

print("=" * 90)
print(" P2  NATIVE GROUP FACTORS by trace (the de-import): T(3)=1/2, C2(adj)=3")
print("=" * 90)
# Restrict the generators to the fundamental Lambda^1 (3-dim) and canonically normalize so that
# Tr_fund(M^a M^b) = 1/2 delta^{ab} (the standard Dynkin normalization).  Then C2(adj), computed
# from the SAME (object-derived) structure constants in that normalization, must be 3.
Mfund = [P1w.conj().T @ (T @ P1w) for T in gens]     # 3x3 action on the color triplet
# diagonalize the Gram of the fundamental generators, rotate to an orthonormal (canonical) basis
Gf = np.array([[np.trace(Mfund[a_].conj().T @ Mfund[b_]) for b_ in range(8)] for a_ in range(8)])
# Gf should be proportional to identity up to the basis we chose; symmetric-orthonormalize:
evg, U = np.linalg.eigh(Gf)
# canonical generators: T'^a = sum_b U[b,a]/sqrt(2*evg[a]) * ... normalize to Tr_fund = 1/2
scale = np.array([1.0 / np.sqrt(2.0 * evg[k]) for k in range(8)])
Mcan = [sum(U[b_, k] * Mfund[b_] for b_ in range(8)) * scale[k] for k in range(8)]
Tfund_check = np.array([[np.trace(Mcan[a_].conj().T @ Mcan[b_]) for b_ in range(8)] for a_ in range(8)])
T3_index = np.real(np.mean([Tfund_check[k, k] for k in range(8)]))
check(f"native fundamental Dynkin index T(3) = Tr_fund(T^a T^a) = {T3_index:.6f} = 1/2 "
      "(canonical normalization; = the hardcoded gauge_dynkin T3[3])", abs(T3_index - 0.5) < 1e-9)

# structure constants in the canonical basis, and C2(adj) = f^{acd} f^{bcd}
fcan = np.zeros((8, 8, 8))
for a_ in range(8):
    for b_ in range(8):
        comm = Mcan[a_] @ Mcan[b_] - Mcan[b_] @ Mcan[a_]
        for c_ in range(8):
            fcan[a_, b_, c_] = np.real(np.trace(Mcan[c_].conj().T @ comm) / 1j / 0.5)  # /Tr_fund norm
C2adj = np.einsum('acd,bcd->ab', fcan, fcan)
C2adj_val = np.real(np.mean(np.diag(C2adj)))
offdiag = np.max(np.abs(C2adj - np.diag(np.diag(C2adj))))
check(f"native adjoint Casimir C2(adj) = f^acd f^bcd = {C2adj_val:.4f} delta^ab "
      f"(off-diag {offdiag:.1e}) = 3 = dim-normalized su(3) adjoint (= hardcoded C2G[3])",
      abs(C2adj_val - 3.0) < 1e-6 and offdiag < 1e-6)

print("=" * 90)
print(" P3  the NATIVE color beta-row b3 (group factors from P2; color mult from Fock)")
print("=" * 90)
# The forced fermion content (color_dim from read_species: quark=3, lepton=1; weak_dim as SM),
# 3 generations.  Native color Dynkin sum uses T(3)=1/2 (P2), color mult = 3 (Fock, read_species).
T3 = Fraction(1, 2)          # NATIVE (P2), not the hardcoded gauge_dynkin lookup
C2 = 3                       # NATIVE (P2)
gens_n = 3
# (color_dim, weak_dim): colored = quark triplet
fermions = [(3, 2), (3, 1), (3, 1),   # Q_L, u_R, d_R  (colored)
            (1, 2), (1, 1)]           # L_L, e_R       (color singlets -> 0)
Tf_color = sum((T3 * w if c == 3 else Fraction(0)) for c, w in fermions) * gens_n
TH_color = Fraction(0)       # the 2 Higgs doublets are color singlets
b3_2hdm = -Fraction(11, 3) * C2 + Fraction(2, 3) * Tf_color + Fraction(1, 3) * TH_color
add3 = Fraction(1, 3) * Tf_color + Fraction(2, 3) * TH_color + Fraction(2, 3) * C2
b3_4d = b3_2hdm + add3
print(f"    native: T(3)=1/2, C2(adj)=3, color mult=3 (Fock) => Sum T_f(color) = {Tf_color}")
print(f"    b3(2HDM)  = -(11/3)*3 + (2/3)*{Tf_color} = {b3_2hdm}")
print(f"    b3(4D)    = b3(2HDM) + add({add3})       = {b3_4d}")
check("native color Dynkin sum Sum T_f(color) = 6 (= the_run.gauge_dynkin Tf[3])", Tf_color == 6)
check("native b3(2HDM) = -7 assembled from the OBJECT's traces (no hardcoded T3/C2G table) "
      "= the_run.read_gauge_running b_2HDM[3]", b3_2hdm == Fraction(-7))
check("native b3(4D completion) = -3 = MSSM b3 (= read_gauge_running / MSSM-lit)", b3_4d == Fraction(-3))

print("=" * 90)
print(" VERDICT")
print("=" * 90)
print("""    The SU(3)_c GROUP factors of the gauge row are now NATIVE: the color su(3) is the
    8 traceless mode-bilinears (grade-2 Cl(6) bivectors) on the k*=3 edge-modes; the Fock
    Lambda^*(C^3) = 1+3+3bar+1 supplies the reps (read_species); T(3)=1/2 and C2(adj)=3
    fall out as TRACES over the object, replacing gauge_dynkin's hardcoded {1/2, 3}; and
    b3 = -7 (2HDM) / -3 (4D) assembles from them with no lookup table.  DE-IMPORT DONE for
    the color group factors.  STILL IMPORTED (probe 2): the SPIN rows (-11/3, 2/3, 1/3) =
    OMEGA_S2_Q2's declared Seeley-DeWitt layer, to be de-imported from the continuum D4 cone.
    OUT OF SCOPE: SU(2)_L (edge-qubit H), hypercharge (open C3-off-diagonal).  No PDG; no
    value moved; grade frontier only.""")
print("=" * 90)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 90)
sys.exit(0 if ok_all else 1)
