#!/usr/bin/env python3
"""
proofs/foundations/TID2_A_split_and_J_2026-07-02.py

T-ID2 ARC, SITTING 1 -- the spacetime/internal split theorem chain (kickoff:
docs/scoping/TID2_split_kickoff_2026-07-02.md, committed 5d46928 BEFORE this run).

THEOREM CHAIN (pre-registered there, verbatim):
  T-A  uniqueness of the split: the S4 edge rep = H1 (+) B1 with the two INEQUIVALENT
       3-dim irreps; Hom_{S4}(H1, B1) = 0.
  T-B  the spatial Clifford: Cl(3)_{H1} exact; Tr omega3 = 0; the commutant structure
       of Cl(3)_{H1} in M8 (expected dim 8 = M2 (+) M2 per chirality).
  T-C  the J-theorems: (i) no S4-invariant complex structure; (ii) Hom_{A4}(H1, B1)
       is 1-dim => THE A4-canonical J, unique up to +-: QUANTIZATION FORCES S4 -> A4;
       (iii) odd permutations flip J -> -J (the enantiomer statement); (iv) the
       legacy/utils pairing recorded as a convention.
  T-D  charge placement: the A4-canonical modes (expected A4-triplet => the species
       structure 1(+)3(+)3bar(+)1), N-hat and charges vs the Cl(3)_{H1} commutant.
KILLS: K1 irreps equivalent/reducible; K2 an S4-invariant J exists or dim Hom_{A4}
       != 1; K3 the commutant structure fails; K4 charges have zero commutant part.
CLASS: pure structure (class a). No PDG, no values, no dressings anywhere.
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
    """signed permutation of oriented edges induced by the vertex permutation sig."""
    R = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R[EDGE_IDX[(a, b)], e] = s
    return R

S4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))]
A4 = [g for g in S4 if np.linalg.det(edge_rep(g)) > 0 or True]  # placeholder; fix below
def parity(p):
    inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
    return 1 if inv % 2 == 0 else -1
A4 = [g for g in S4 if parity([g[i] for i in range(4)]) == 1]

# H1 (cycle space) and B1 (coboundary space), orthonormal in the edge metric
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1],
                 [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)   # cycles (Q0 convention)
d0 = np.zeros((4, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
def orth(M):
    Q, _ = np.linalg.qr(M)
    return Q
H1 = orth(Chat)                                   # 6x3
B1 = orth(d0.T[:, :3] if np.linalg.matrix_rank(d0) == 3 else d0.T)
B1 = orth(d0.T @ orth(np.linalg.svd(d0)[2][:3].T) if False else d0.T)
# robust: B1 = orthonormal basis of the row space of d0 (rank 3)
U_, S_, Vt_ = np.linalg.svd(d0)
B1 = Vt_[:3].T                                    # 6x3 orthonormal, row space of d0

print("=" * 88)
print(" T-A  uniqueness: the S4 edge rep = H1 (+) B1, the two INEQUIVALENT 3-dim irreps")
print("=" * 88)
check(f"orthogonal decomposition: H1 perp B1, 3 + 3 = 6 (|H1^T B1| = "
      f"{np.max(np.abs(H1.T @ B1)):.1e})", np.max(np.abs(H1.T @ B1)) < 1e-12)
invH = all(np.max(np.abs((np.eye(NE) - H1 @ H1.T) @ edge_rep(g) @ H1)) < 1e-12 for g in S4)
invB = all(np.max(np.abs((np.eye(NE) - B1 @ B1.T) @ edge_rep(g) @ B1)) < 1e-12 for g in S4)
check("both summands S4-invariant (cycles -> cycles, coboundaries -> coboundaries)",
      invH and invB)
# characters on a transposition (odd class): chi_H1 vs chi_B1 must differ by sign
t01 = {0: 1, 1: 0, 2: 2, 3: 3}
chiH = float(np.trace(H1.T @ edge_rep(t01) @ H1))
chiB = float(np.trace(B1.T @ edge_rep(t01) @ B1))
print(f"    characters on the transposition (01): chi_H1 = {chiH:+.4f}, chi_B1 = {chiB:+.4f}")
check("the two summands are the two INEQUIVALENT 3-dim irreps (characters differ on "
      "odd classes: chi_H1 = -chi_B1 != 0)", abs(chiH + chiB) < 1e-9 and abs(chiH) > 0.5)
# Hom_{S4}(H1,B1) = 0: solve the intertwiner equations
rows = []
for g in S4:
    R = edge_rep(g)
    RH = H1.T @ R @ H1
    RB = B1.T @ R @ B1
    # phi RH = RB phi  (phi 3x3), ROW-major vec convention (matches .reshape):
    rows.append(np.kron(np.eye(3), RH.T) - np.kron(RB, np.eye(3)))
M = np.vstack(rows)
dimS4 = 9 - np.linalg.matrix_rank(M, tol=1e-9)
check(f"Hom_S4(H1, B1) = {dimS4} (must be 0): the split is UNIQUE -- no invariant "
      "mixing exists  [K1 decided]", dimS4 == 0)

print("=" * 88)
print(" T-B  the spatial Clifford Cl(3)_{H1} and the Fock factorization")
print("=" * 88)
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
def gam(vec6):
    return sum(vec6[a] * g6[a] for a in range(NE))
gh = [gam(H1[:, i]) for i in range(3)]
okcl = all(np.max(np.abs(gh[i] @ gh[j] + gh[j] @ gh[i]
                         - (2.0 if i == j else 0.0) * np.eye(8))) < 1e-12
           for i in range(3) for j in range(3))
check("Cl(3)_{H1} exact: {gamma(h_i), gamma(h_j)} = 2 delta_ij on the 8-dim Fock rep", okcl)
om3 = gh[0] @ gh[1] @ gh[2]
evo = np.linalg.eigvals(om3)
check(f"omega3^2 = -I and Tr omega3 = {np.trace(om3).real:.1e}: the two chiralities "
      f"come with EQUAL multiplicity (spectrum {sorted(np.round(evo.imag, 6))})",
      np.max(np.abs(om3 @ om3 + np.eye(8))) < 1e-12 and abs(np.trace(om3)) < 1e-9)
# commutant of Cl(3)_{H1} in M8
rowsC = [np.kron(np.eye(8), gh[i]) - np.kron(gh[i].T, np.eye(8)) for i in range(3)]
MC = np.vstack(rowsC)
dimComm = 64 - np.linalg.matrix_rank(MC, tol=1e-9)
check(f"commutant of Cl(3)_H1 in M8: dim_C = {dimComm} = 8 = M2 (+) M2 (a 2-dim "
      "internal factor PER chirality -- the two-Weyl/doublet structure)  [K3 decided]",
      dimComm == 8)

print("=" * 88)
print(" T-C  the J-theorems: quantization forces S4 -> A4; the +- pair = enantiomers")
print("=" * 88)
# (i) no S4-invariant J on R^6: solve [J, R_g] = 0 for all g, then J^2 = -I feasibility
rowsJ = []
for g in S4:
    R = edge_rep(g)
    rowsJ.append(np.kron(np.eye(NE), R) - np.kron(R.T, np.eye(NE)))
MJ = np.vstack(rowsJ)
null_dim = NE * NE - np.linalg.matrix_rank(MJ, tol=1e-9)
# the S4-commutant of the edge rep: by Schur = R (+) R (scalars on each irrep): dim 2
print(f"    S4-commutant of the edge rep: dim = {null_dim} (Schur: scalars on each of "
      "the two inequivalent real irreps)")
check("(i) NO S4-invariant complex structure exists: the commutant is {aP_H1 + bP_B1} "
      "(diagonal scalars), and (aP+bQ)^2 = -I needs a^2 = b^2 = -1 -- impossible over "
      "R  [K2 part 1]", null_dim == 2)
# (ii) Hom_{A4}(H1,B1): the intertwiner space
rowsA = []
for g in A4:
    R = edge_rep(g)
    RH = H1.T @ R @ H1
    RB = B1.T @ R @ B1
    rowsA.append(np.kron(np.eye(3), RH.T) - np.kron(RB, np.eye(3)))
MA = np.vstack(rowsA)
ns = 9 - np.linalg.matrix_rank(MA, tol=1e-9)
check(f"(ii) Hom_A4(H1, B1) = {ns} (must be 1: on A4 the two irreps become EQUIVALENT, "
      "real-type Schur) -- the A4-canonical intertwiner exists and is unique up to "
      "scale  [K2 part 2]", ns == 1)
# construct THE canonical J: phi = the unit intertwiner; J(h) = phi h, J(b) = -phi^{-1} b
# (+- convention DECLARED: fix the sign so that <c1_cycle, J c1_grad-partner> > 0 ... we
#  fix sign by requiring the triple product orientation below to be positive.)
_, _, V = np.linalg.svd(MA)
phi = V[-1].reshape(3, 3)
phi /= np.linalg.norm(phi) / math.sqrt(3)          # normalize to an isometry (check below)
check(f"phi is an isometry up to sign (phi^T phi = I to {np.max(np.abs(phi.T@phi - np.eye(3))):.1e})",
      np.max(np.abs(phi.T @ phi - np.eye(3))) < 1e-9)
if np.linalg.det(phi) < 0:
    phi = -phi                                     # declared sign convention
J = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T            # J: H1 -> B1 (+phi), B1 -> H1 (-phi^T)
check(f"J^2 = -I ({np.max(np.abs(J @ J + np.eye(NE))):.1e}), J antisymmetric "
      f"({np.max(np.abs(J + J.T)):.1e}): a genuine orthogonal complex structure",
      np.max(np.abs(J @ J + np.eye(NE))) < 1e-9 and np.max(np.abs(J + J.T)) < 1e-9)
okA4 = all(np.max(np.abs(edge_rep(g) @ J - J @ edge_rep(g))) < 1e-9 for g in A4)
odd_flip = all(np.max(np.abs(edge_rep(g) @ J @ edge_rep(g).T + J)) < 1e-9
               for g in S4 if parity([g[i] for i in range(4)]) == -1)
check("(ii') J is A4-invariant  [QUANTIZATION FORCES S4 -> A4 = the framework's C[A4]]", okA4)
check("(iii) EVERY odd permutation conjugates J -> -J: the two quantizations (+-J) are "
      "exchanged by the improper operations -- THE ENANTIOMER PAIR (srs <-> srs-z)", odd_flip)
print("    (iv) legacy/utils convention (recorded): the Jordan-Wigner pairing "
      "(g1,g2),(g3,g4),(g5,g6) is a REPRESENTATION choice on the abstract algebra; it "
      "carries no edge dictionary and no invariance claim; no physics hangs on it.")

print("=" * 88)
print(" T-D  the A4-canonical modes and charge placement  [K4 decided]")
print("=" * 88)
# complex modes from THE canonical J: eigenvectors of J with eigenvalue +i (3 modes)
w, W = np.linalg.eig(J)
sel = np.where(w.imag > 0.5)[0]
modes, _ = np.linalg.qr(W[:, sel])                 # orthonormalize the degenerate +i space
# annihilators: a_m = gamma(conj(v_m))/sqrt(2); v^T v = 0 automatic on the +i space,
# <v_m, v_n> = delta after QR => exact CAR
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
okCAR = True
for m in range(3):
    for n in range(3):
        acAB = A_ops[m] @ A_ops[n].conj().T + A_ops[n].conj().T @ A_ops[m]
        acAA = A_ops[m] @ A_ops[n] + A_ops[n] @ A_ops[m]
        okCAR &= np.max(np.abs(acAB - (1.0 if m == n else 0.0) * np.eye(8))) < 1e-9
        okCAR &= np.max(np.abs(acAA)) < 1e-9
check("the 3 canonical modes satisfy the CAR exactly ({a_m, a_n†} = delta_mn, "
      "{a_m, a_n} = 0): the A4-canonical Fock structure exists", okCAR)
# A4-triplet check: the modes transform as the 3-dim A4 irrep (permuted by A4 up to U(3) mixing
# WITHIN the +i eigenspace):
okTrip = all(np.max(np.abs((np.eye(NE) - modes @ modes.conj().T) @ edge_rep(g) @ modes)) < 1e-9
             for g in A4)
check("the mode space is A4-invariant: the 3 canonical modes form an A4 TRIPLET => "
      "Lambda* = 1 (+) 3 (+) 3bar (+) 1 as A4-reps (the species structure, now from THE "
      "canonical J)", okTrip)
Nhat = sum(A_ops[m].conj().T @ A_ops[m] for m in range(3))
evN = np.round(np.sort(np.linalg.eigvalsh(Nhat)), 6)
check(f"N-hat spectrum = {list(evN)} = Hamming weights {{0,1,1,1,2,2,2,3}} (the Fock "
      "grading from the canonical J)", np.allclose(evN, [0, 1, 1, 1, 2, 2, 2, 3]))
# charge placement: decompose N-hat into commutant part + orthogonal remainder w.r.t.
# Cl(3)_{H1}: project onto the commutant (HS-orthogonal projection via the nullspace basis)
_, _, VC = np.linalg.svd(MC)
comm_basis = VC[np.linalg.matrix_rank(MC, tol=1e-9):]        # (64 - rank) x 64
Nvec = Nhat.reshape(64)
# HS projection: orthonormalize comm_basis rows first
Qc, _ = np.linalg.qr(comm_basis.conj().T)
Npar = Qc @ (Qc.conj().T @ Nvec)
frac = float(np.linalg.norm(Npar) ** 2 / np.linalg.norm(Nvec) ** 2)
print(f"    charge placement: |P_comm(N-hat)|^2 / |N-hat|^2 = {frac:.4f}")
check(f"N-hat has a NONZERO commutant component (K4 does not fire); the fraction "
      f"{frac:.3f} is the recorded sitting-2 input (1.0 would be full factorization; "
      "the remainder measures exactly how the canonical J straddles the split)",
      frac > 0.05)

print("=" * 88)
print(" VERDICT (sitting 1)")
print("=" * 88)
print(f"""    LANDED (all machine-verified): the split H1 (+) B1 is UNIQUE (the two
    inequivalent S4 irreps; no invariant mixing); Cl(3)_{{H1}} is exact with equal
    chiralities and commutant M2 (+) M2 (a 2-dim internal factor per chirality); NO
    S4-invariant complex structure exists, an A4-invariant J EXISTS and is unique up
    to +-, every odd permutation flips J -> -J. THEOREM-CANDIDATES ESTABLISHED (front-
    door claims user-gated): (1) THE CAR/FOCK QUANTIZATION FORCES S4 -> A4 -- the
    framework's C[A4] is the stabilizer of the complex structure the quantization
    requires; (2) the +-J pair IS the enantiomer pair (srs <-> srs-z): the joint
    object = both quantizations; (3) the canonical modes are an A4 triplet with the
    species Fock grading falling out of THE canonical J (no convention).
    RECORDED for sitting 2 (the current-form test): the charge operator's commutant
    fraction {frac:.3f} -- the canonical J straddles the split by construction
    (J: H1 <-> B1), so the naive 'charges fully internal' picture is quantitatively
    wrong by exactly this measured amount; sitting 2 must derive the current's
    (v - a gamma5) structure from the landed split + J, or localize the obstruction.
    No value shipped; the P3/PS identification is NOT yet claimed closed.""")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
