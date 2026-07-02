#!/usr/bin/env python3
"""Phase 5.2 follow-up (PANEL-ORDERED identity check) -- the P-sign bit IS
the omega/omega2 character-naming convention.

Order: 5.2 panel verdict 2026-06-11 (spec phase5_2_repricing_spec, post-
panel append): "P-sign proven identical to the omega/omega2 channel-
labeling convention (identity-check ORDERED, open) -> row 2.0."  The
sensitivity was FROZEN with the row: if this check passes, the A5-mass row
prints 2.0 (in-row residual = Higgs placement 1.585 only); the P-sign bit
single-homes at its priced home -- the omega/omega2 channel-labeling
convention, ~1 bit in the Majorana-panel ledger amendment (self_mdl_ledger
2026-06-11), itself part of the R3/M1.B basis-match + ADOPTED-B3 labeling
complex. If any gate fails, the row STAYS at 3.0.

THE CLAIM, precisely: the Sec-2.2 grain's in-row freedom "which P-sign
family is the up-type/CKM source" is not a new binary choice -- it is the
SAME Z2 as the (already-homed) choice of which C3 character is named
omega. Logic: (I1+I2) the sign partition of the 8 Ramanujan P modes
coincides exactly with the little-group irrep-class partition; (I3) the
two families' C3 contents are exact conjugates ({1,w} vs {1,w2}), so the
relabeling w <-> w2 exchanges the families' character-type names; (I4) the
families are otherwise INTRINSICALLY INDISTINGUISHABLE -- the mirror
composite lambda -> -conj(lambda) (antiperiod, exact) maps one onto the
other, and every convention-independent invariant (|lambda| = sqrt2,
arg(lambda^10) = +-162.3876 deg, two forced 2-dim projective irreps each)
is identical. Hence the ONLY datum that distinguishes "h_P" from
"h_P_neg" is the character name attached to it: assigning a physical role
to a sign family IS assigning it to a character name. One convention, one
home, zero new bits.

Gates:
  I1 sign partition: 8 Ramanujan modes at P split (4,4) by sign(Re),
     Re = +-sqrt3/2 exactly (the Sec-2.2 h_P / h_P_neg families).
  I2 the full T(12) projective-irrep class partition of the four
     Ramanujan doublets has exactly 2 classes x 2 clusters and EQUALS the
     sign partition.
  I3 C3 content (P3 convention, banked): +Re clusters {1,w}; -Re clusters
     {1,w2} -- exact conjugate multisets.
  I4 mirror pairing + invariants: multiset(-conj(spec_+)) = spec_- at
     machine zero; |lambda| = sqrt2 both; arg(lambda^10) multisets both
     {+-162.3876 deg}; both families consist of two FORCED 2-dim blocks.
  I5 conclusion (logical composition of I1-I4, gated so verify.py locks
     the ordered finding): the P-sign bit = the omega/omega2 naming
     convention; frozen sensitivity fires -> row prints 2.0.
"""
import os
import sys
from itertools import permutations, product

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, ATOMS, A_PRIM  # noqa: E402

FAILURES = []
M_CART = A_PRIM.T
M_INV = la.inv(M_CART)
SQ2, SQ3 = np.sqrt(2), np.sqrt(3)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def mod1(v):
    w = np.asarray(v, float) % 1.0
    w[np.abs(w - 1.0) < 1e-9] = 0.0
    return w


def is_bcc(v):
    w = np.asarray(v, float)
    if np.max(np.abs(w - np.round(w))) < 1e-9:
        return True
    w2 = w - 0.5
    return np.max(np.abs(w2 - np.round(w2))) < 1e-9


def canon_tau(v):
    cands = [mod1(v), mod1(np.asarray(v, float) + 0.5)]
    keys = [tuple(np.round(c, 6)) for c in cands]
    return cands[keys.index(min(keys))]


def prim_int(v_cart):
    d = M_INV @ np.asarray(v_cart, float)
    di = np.round(d)
    assert np.max(np.abs(d - di)) < 1e-9
    return di.astype(int)


bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
NE = len(EDGES)
E_INDEX = {e: a for a, e in enumerate(EDGES)}
REV = {a: E_INDEX[(j, i, tuple(-x for x in c))] for a, (i, j, c) in enumerate(EDGES)}


def B_of(k):
    B = np.zeros((NE, NE), dtype=complex)
    for a, (i, j, c) in enumerate(EDGES):
        for b, (i2, j2, c2) in enumerate(EDGES):
            if i2 == j and b != REV[a]:
                B[b, a] = np.exp(2j * np.pi * np.dot(k, np.asarray(c2, float)))
    return B


def atom_of(pos):
    for j in range(4):
        L = pos - ATOMS[j]
        if is_bcc(L):
            return j, L
    return None, None


def op_preserves(R, tau):
    for i in range(4):
        if atom_of(R @ ATOMS[i] + tau)[0] is None:
            return False
    for (i, j, c) in EDGES:
        i2, Li = atom_of(R @ ATOMS[i] + tau)
        j2, Lj = atom_of(R @ (ATOMS[j] + M_CART @ np.asarray(c, float)) + tau)
        if (i2, j2, tuple(prim_int(Lj) - prim_int(Li))) not in E_INDEX:
            return False
    return True


OPS = []
for perm in permutations(range(3)):
    for signs in product((1, -1), repeat=3):
        R = np.zeros((3, 3))
        for row, (col, s) in enumerate(zip(perm, signs)):
            R[row, col] = s
        if abs(la.det(R) - 1.0) > 1e-9:
            continue
        seen = set()
        for j in range(4):
            tau = canon_tau(ATOMS[j] - R @ ATOMS[0])
            key = tuple(np.round(tau, 6))
            if key in seen:
                continue
            seen.add(key)
            if op_preserves(R, tau):
                OPS.append((R, tau))
assert len(OPS) == 24
R_PRIM = [np.round(M_INV @ R @ M_CART).astype(int) for R, _ in OPS]

EMAP = []
for (R, t) in OPS:
    rows = []
    for (i, j, c) in EDGES:
        i2, Li = atom_of(R @ ATOMS[i] + t)
        j2, Lj = atom_of(R @ (ATOMS[j] + M_CART @ np.asarray(c, float)) + t)
        rows.append((E_INDEX[(i2, j2, tuple(prim_int(Lj) - prim_int(Li)))],
                     prim_int(Li)))
    EMAP.append(rows)


def k_image(g, k):
    return la.inv(R_PRIM[g]).T @ np.asarray(k, float)


def U_edge(g, k):
    k = np.asarray(k, float)
    kp = k_image(g, k)
    U = np.zeros((NE, NE), dtype=complex)
    for a, (i, j, c) in enumerate(EDGES):
        a2, di = EMAP[g][a]
        c2 = np.asarray(EDGES[a2][2], float)
        U[a2, a] = np.exp(2j * np.pi * (kp @ (di + c2) - k @ np.asarray(c, float)))
    return U


# P3: the banked phase-free C3 edge permutation (c3-characters convention)
w = np.exp(2j * np.pi / 3)
C3_R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
sigma = {0: 0, 1: 3, 3: 2, 2: 1}
P3 = np.zeros((NE, NE))
for a, (i, j, c) in enumerate(EDGES):
    v = C3_R @ (ATOMS[j] + M_CART @ np.asarray(c, float) - ATOMS[i])
    for b, (i2, j2, c2) in enumerate(EDGES):
        if (i2, j2) == (sigma[i], sigma[j]) and np.allclose(
                ATOMS[j2] + M_CART @ np.asarray(c2, float) - ATOMS[i2], v, atol=1e-9):
            P3[b, a] = 1.0
            break
assert np.allclose(P3 @ P3 @ P3, np.eye(NE))


def c3_label(z):
    for nm, val in (("1", 1), ("w", w), ("w2", w ** 2)):
        if abs(z - val) < 1e-6:
            return nm
    return "?"


print("=" * 72)
print(" PHASE 5.2 ORDERED CHECK -- P-sign bit == omega/omega2 convention?")
print("=" * 72)

kP = np.array([0.25, 0.25, 0.25])
BP = B_of(kP)
evP, VP = la.eig(BP)
VPi = la.inv(VP)

ram_idx = [i for i in range(NE) if abs(abs(evP[i]) - SQ2) < 1e-6]
plus = sorted(i for i in ram_idx if evP[i].real > 0)
minus = sorted(i for i in ram_idx if evP[i].real < 0)
gate("I1 sign partition (4,4); Re = +-sqrt3/2 exactly",
     len(ram_idx) == 8 and len(plus) == 4 and len(minus) == 4
     and all(abs(abs(evP[i].real) - SQ3 / 2) < 1e-9 for i in ram_idx))

# I2: full T(12) irrep-class partition of the Ramanujan doublets
lgP = [g for g in range(24)
       if np.max(np.abs(k_image(g, kP) - kP - np.round(k_image(g, kP) - kP))) < 1e-9]
Us = {g: U_edge(g, kP) for g in lgP}
rng = np.random.default_rng(11)
H0 = rng.normal(size=(NE, NE)) + 1j * rng.normal(size=(NE, NE))
H0 = H0 + H0.conj().T
Hb = sum(U @ H0 @ U.conj().T for U in Us.values()) / len(lgP)
evH, VH = la.eigh(Hb)
blocks, i = [], 0
while i < NE:
    grp = [i]
    while i + 1 < NE and abs(evH[i + 1] - evH[i]) < 1e-8:
        i += 1
        grp.append(i)
    blocks.append(VH[:, grp])
    i += 1
chars = [tuple(np.round(np.trace(Q.conj().T @ Us[g] @ Q), 6) for g in lgP)
         for Q in blocks]
class_keys = []
class_of_block = []
for chi in chars:
    for ci, ck in enumerate(class_keys):
        if max(abs(np.array(chi) - np.array(ck))) < 1e-5:
            class_of_block.append(ci)
            break
    else:
        class_keys.append(chi)
        class_of_block.append(len(class_keys) - 1)
Piso = {}
for ci, ck in enumerate(class_keys):
    bi = class_of_block.index(ci)
    d = blocks[bi].shape[1]
    Piso[ci] = (d / len(lgP)) * sum(np.conj(ck[ig]) * Us[g]
                                    for ig, g in enumerate(lgP))


def cluster_class(lam):
    idx = [i for i in range(NE) if abs(evP[i] - lam) < 1e-7]
    Pc = VP[:, idx] @ VPi[idx, :]
    content = [ci for ci, Pa in Piso.items()
               if np.round(np.real(np.trace(Pa @ Pc))) > 0]
    assert len(content) == 1
    return content[0], len(idx)


ram_lams = sorted({np.round(evP[i], 7) for i in ram_idx},
                  key=lambda z: (round(z.real, 6), round(z.imag, 6)))
cls = {lam: cluster_class(lam) for lam in ram_lams}
cls_plus = {cls[lam][0] for lam in ram_lams if lam.real > 0}
cls_minus = {cls[lam][0] for lam in ram_lams if lam.real < 0}
gate("I2 irrep-class partition == sign partition (one class per sign family, "
     "2 clusters each, all dims 2)",
     len(cls_plus) == 1 and len(cls_minus) == 1 and cls_plus != cls_minus
     and all(d == 2 for _, d in cls.values()))

# I3: C3 content conjugacy (P3 convention)
def c3_content(lam):
    idx = [i for i in range(NE) if abs(evP[i] - lam) < 1e-7]
    Vc = VP[:, idx]
    return sorted(c3_label(z) for z in la.eigvals(la.pinv(Vc) @ P3 @ Vc))


cont_plus = {tuple(c3_content(lam)) for lam in ram_lams if lam.real > 0}
cont_minus = {tuple(c3_content(lam)) for lam in ram_lams if lam.real < 0}
gate("I3 C3 content: +Re family {1,w} x2; -Re family {1,w2} x2 -- exact "
     "conjugate types (relabeling w<->w2 exchanges the family names)",
     cont_plus == {("1", "w")} and cont_minus == {("1", "w2")})

# I4: mirror pairing + convention-independent invariants identical
sp = sorted((evP[i] for i in plus), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
sm = sorted((evP[i] for i in minus), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
mirror_img = sorted((-np.conj(z) for z in sp),
                    key=lambda z: (round(z.real, 6), round(z.imag, 6)))
pair_ok = all(abs(a - b) < 1e-12 for a, b in zip(mirror_img, sm))
arg10 = lambda z: round(abs(np.degrees(np.angle(z ** 10))), 4)   # noqa: E731
inv_plus = sorted(arg10(z) for z in sp)
inv_minus = sorted(arg10(z) for z in sm)
gate("I4 mirror composite -conj(.) maps +family onto -family exactly; "
     "invariants identical (|l|=sqrt2; |arg l^10|=162.3876 all 8)",
     pair_ok and inv_plus == inv_minus
     and all(abs(v - 162.3876) < 1e-3 for v in inv_plus)
     and all(abs(abs(z) - SQ2) < 1e-12 for z in sp + sm))

# I5: the ordered conclusion, locked
gate("I5 ORDERED IDENTITY ESTABLISHED: the in-row P-sign bit is the "
     "omega/omega2 channel-labeling convention (single-homed at its "
     "priced ~1-bit Majorana-panel ledger line, R3/M1.B+B3 complex); "
     "frozen sensitivity fires -> A5-mass row prints 2.0",
     not FAILURES)

print("\n  Consequence (panel-frozen sensitivity): in-row residual = Higgs")
print("  placement log2(3) = 1.585 -> row prints 2.0 (spent rounds up).")
print("  Remaining frozen sensitivity: R1 ratified as framework-derived")
print("  removes the placement freedom -> in-row 0 -> row prints 0")
print("  (the panel's 'both -> 0' band; residuals live at cross-ref homes).")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES} -- row STAYS 3.0")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- ordered identity check DISCHARGED")
print("=" * 72)
sys.exit(0)
