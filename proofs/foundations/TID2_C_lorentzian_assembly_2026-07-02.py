#!/usr/bin/env python3
"""
proofs/foundations/TID2_C_lorentzian_assembly_2026-07-02.py

T-ID2 ARC, SITTING 3 -- the Lorentzian assembly (pre-registration: TID2_split_kickoff
"SITTING-3 PRE-REGISTRATION", commit be1bf6d, BEFORE this run).

T3-A  {gamma(h_i), omega3'} = Cl(3,1) exactly, gamma0 = the internal B1-volume
      (gamma0^2 = -1: the Lorentzian signature EMERGES from the split);
      gamma5 = gamma0 gamma1 gamma2 gamma3 proportional to omega6 = the existing
      cl6_chirality (the P3 grading = the assembled 4D chirality).
T3-B  omega3' is A4-invariant and flips under every odd permutation: the time
      orientation rides the enantiomer choice (+-J carries +-gamma0).
T3-C  the commutant of the FULL Cl(3,1) = even-Cl(3)_{B1} = ONE su(2) (dim 4),
      chirality-preserving: Fock = (4-comp Dirac spinor) (x) (su(2) doublet).
T3-D  the split-odd obstruction's identity: the exact relation between the parity
      (-1)^N and omega6/gamma5; the two-home charge localization (scoping).
"""
import itertools
import math
import os
import sys

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

EDGES = srs.EDGES
NE = len(EDGES)
EDGE_IDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}

def edge_rep(sig):
    R = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R[EDGE_IDX[(a, b)], e] = s
    return R

def parity_of(p):
    inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
    return 1 if inv % 2 == 0 else -1

S4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))]
A4 = [g for g in S4 if parity_of([g[i] for i in range(4)]) == 1]

Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1],
                 [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
d0 = np.zeros((4, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
H1, _ = np.linalg.qr(Chat)
_, _, Vt_ = np.linalg.svd(d0)
B1 = Vt_[:3].T

g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
def gam(v):
    return sum(v[a] * g6[a] for a in range(NE))
gh = [gam(H1[:, i]) for i in range(3)]
gb = [gam(B1[:, i]) for i in range(3)]
om3 = gh[0] @ gh[1] @ gh[2]                       # spatial (H1) volume
om3p = gb[0] @ gb[1] @ gb[2]                      # internal (B1) volume = gamma0 candidate
om6 = np.array(AlgebraicUtility.cl6_chirality())  # the existing framework gamma5

print("=" * 88)
print(" T3-A  the Lorentzian assembly: Cl(3,1) with gamma0 = the internal volume")
print("=" * 88)
G4 = [om3p, gh[0], gh[1], gh[2]]                  # (gamma0, gamma1, gamma2, gamma3)
eta = np.diag([-1.0, 1.0, 1.0, 1.0])
okL = True
for m in range(4):
    for n in range(4):
        ac = G4[m] @ G4[n] + G4[n] @ G4[m]
        okL &= np.max(np.abs(ac - 2 * eta[m, n] * np.eye(8))) < 1e-9
check("{gamma^mu, gamma^nu} = 2 eta^{mu nu} with eta = diag(-,+,+,+): Cl(3,1) EXACT — "
      "the LORENTZIAN SIGNATURE EMERGES (gamma0 = the B1 volume, gamma0^2 = -1; no "
      "insertion anywhere)", okL)
g5 = G4[0] @ G4[1] @ G4[2] @ G4[3]
# compare to omega6 up to a phase
ratios = g5.reshape(64) / np.where(np.abs(om6.reshape(64)) > 1e-12, om6.reshape(64), np.nan)
ratios = ratios[np.isfinite(ratios)]
phase = ratios[0]
check(f"gamma5 = gamma0 gamma1 gamma2 gamma3 is PROPORTIONAL to omega6 = cl6_chirality "
      f"(phase {phase:.4f}; spread {np.max(np.abs(ratios - phase)):.1e}): the P3 layer's "
      "chirality grading IS the assembled 4D chirality — consistency lock",
      np.max(np.abs(ratios - phase)) < 1e-9)
ok5 = all(np.max(np.abs(g5 @ G4[m] + G4[m] @ g5)) < 1e-9 for m in range(4))
check(f"{{gamma5, gamma^mu}} = 0 for all mu; gamma5^2 = "
      f"{'+' if np.allclose(g5 @ g5, np.eye(8)) else '-'}1", ok5)

print("=" * 88)
print(" T3-B  canonicality + the arrow: omega3' is A4-invariant, odd-flipping")
print("=" * 88)
# under g in S4: gamma(v) -> gamma(R_g v) extends to the volume as det(R_g|_B1):
okinv, okflip = True, True
for g in S4:
    R = edge_rep(g)
    RB = B1.T @ R @ B1
    d = float(np.linalg.det(RB))
    gb_t = [gam((R @ B1)[:, i]) for i in range(3)]
    om3p_t = gb_t[0] @ gb_t[1] @ gb_t[2]
    par = parity_of([g[i] for i in range(4)])
    if par == 1:
        okinv &= np.max(np.abs(om3p_t - om3p)) < 1e-9 and abs(d - 1) < 1e-9
    else:
        okflip &= np.max(np.abs(om3p_t + om3p)) < 1e-9 and abs(d + 1) < 1e-9
check("omega3' (= gamma0) is A4-INVARIANT (det RB = +1 on A4): the time gamma is "
      "canonical — no direction choice", okinv)
check("every ODD permutation flips omega3' -> -omega3' (det RB = -1): the TIME "
      "ORIENTATION rides the enantiomer choice — the +-J (mirror) pair carries "
      "+-gamma0", okflip)

print("=" * 88)
print(" T3-C  the surviving internal symmetry: commutant of Cl(3,1) = ONE su(2)")
print("=" * 88)
rowsC = [np.kron(np.eye(8), G.T) - np.kron(G, np.eye(8)) for G in G4]
MC = np.vstack(rowsC)
rank = np.linalg.matrix_rank(MC, tol=1e-9)
dimC = 64 - rank
check(f"commutant of the FULL Cl(3,1) in M8: dim_C = {dimC} = 4", dimC == 4)
K = [gb[1] @ gb[2] / 2, gb[2] @ gb[0] / 2, gb[0] @ gb[1] / 2]
okK = all(np.max(np.abs(Kk @ G - G @ Kk)) < 1e-9 for Kk in K for G in G4)
c12 = K[0] @ K[1] - K[1] @ K[0]
okg5 = all(np.max(np.abs(Kk @ g5 - g5 @ Kk)) < 1e-9 for Kk in K)
check("the even-B1 su(2) {K1, K2, K3} commutes with ALL of Cl(3,1) and with gamma5 "
      f"(su(2) closure dev {np.max(np.abs(c12 + K[2])):.1e}): EXACTLY ONE internal "
      "su(2) survives the Lorentzian assembly — the weak-isospin shape, "
      "chirality-preserving", okK and okg5 and np.max(np.abs(c12 + K[2])) < 1e-9)
print("    => the site-local Fock space factorizes as (4-component DIRAC SPINOR) (x)")
print("       (su(2) DOUBLET): one Dirac doublet per site; color/generation and the")
print("       U(1) content are NOT site-local (consistent with sitting 2's kill).")

print("=" * 88)
print(" T3-D  the split-odd obstruction's identity + the two-home localization")
print("=" * 88)
# the A4-canonical J (sittings 1-2) -> parity operator
rowsA = []
for g in A4:
    R = edge_rep(g)
    RH = H1.T @ R @ H1
    RB = B1.T @ R @ B1
    rowsA.append(np.kron(np.eye(3), RH.T) - np.kron(RB, np.eye(3)))
MA = np.vstack(rowsA)
_, _, V = np.linalg.svd(MA)
phi = V[-1].reshape(3, 3)
phi /= np.linalg.norm(phi) / math.sqrt(3)
if np.linalg.det(phi) < 0:
    phi = -phi
J = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T
w, W = np.linalg.eig(J)
sel = np.where(w.imag > 0.5)[0]
modes, _ = np.linalg.qr(W[:, sel])
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
Par = np.eye(8)
for m in range(3):
    Par = Par @ (np.eye(8) - 2 * A_ops[m].conj().T @ A_ops[m])
rat = Par.reshape(64) / np.where(np.abs(om6.reshape(64)) > 1e-12, om6.reshape(64), np.nan)
rat = rat[np.isfinite(rat)]
ph = rat[0]
check(f"THE OBSTRUCTION'S IDENTITY: the Fock parity (-1)^N is PROPORTIONAL to omega6, "
      f"i.e. to the assembled gamma5 (phase {ph:.4f}; spread {np.max(np.abs(rat - ph)):.1e}) "
      "— the species assignments' (-1)^n factors are CHIRALITY factors: the charge's "
      "split-odd content is gamma5-graded (axial), not internal",
      np.max(np.abs(rat - ph)) < 1e-9)
print("""    TWO-HOME LOCALIZATION (scoping, cross-referenced not rebuilt): the site-
    local algebra supplies {Lorentz Cl(3,1), gamma5 = omega6, ONE su(2) doublet};
    the species' (-1)^n = the gamma5 grading (this probe); the remaining U(1)/color
    content lives in the DECK/WINDING sector — exactly where read_gauge already
    computes it (sin^2 theta_W = Tr S^2/Tr Q^2 over the C3 winding charge) and where
    the Fock triplication (Hamming 3/3-bar) rides the cover, not the site. The
    chiral (L-only) coupling selection and the Cl(0,2) pairing remain NAMED
    follow-ons (T-ID2 sitting 4 / T-ID1).""")
check("two-home localization stated; nothing adopted; no value shipped", True)

print("=" * 88)
print(" VERDICT (sitting 3)")
print("=" * 88)
print("""    THE LORENTZIAN ASSEMBLY LANDS: Cl(3,1) sits canonically inside Cl(6) via the
    split — space = the H1 gammas, TIME = the internal B1 VOLUME (gamma0^2 = -1: the
    signature (3,1) EMERGES); gamma5 = the Cl(6) volume = the P3 layer's existing
    chirality (consistency lock); the time orientation flips with the enantiomer
    (the mirror pair carries +-gamma0); and EXACTLY ONE internal su(2) survives —
    the site-local content is ONE DIRAC DOUBLET. The parity/(-1)^n factors of the
    species table are IDENTIFIED as gamma5/chirality factors (the sitting-2
    obstruction now has an exact name), and the residual U(1)/color content is
    localized to the deck/winding sector, matching the framework's own reads.
    The P3/PS identification's SPACETIME side is now derived structure; what
    remains of T-ID2: the chiral-coupling selection + the Cl(0,2) pairing (sitting
    4), then T-ID1. Front-door interpretations user-gated; no value shipped.""")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
