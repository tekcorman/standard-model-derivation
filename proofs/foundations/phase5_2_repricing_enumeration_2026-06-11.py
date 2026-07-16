#!/usr/bin/env python3
"""Phase 5.2 -- A5-mass re-pricing: the enumeration behind the decision tree.

Spec: internal research notes (FROZEN before
this probe existed; SHA-256 9f54c854..., register row in the freezing
commit). The ledger row does NOT move on this probe -- it moves only after
the ultracode panel rules on R1-R3. This probe verifies, at machine
precision, every count and every zero-cost fact the frozen tree invokes:

  E1  size-compatibility: of the 8! = 40320 family->role bijections,
      exactly 12 preserve mode budgets  (N1 = 3.58 bits; the -11.72 is a
      LEDGER CORRECTION, not an EBR result).
  E2  exactly one 2-mode family has a Gamma component (Perron) -- family
      census recomputed fresh from B(saddle) spectra, not asserted.
  E3  ruling R1 (Higgs role needs a Gamma component): 12 -> 4  (2.0 bits).
  E4  Bass-branch tie EXACT: the N-spill eigenvalue sets equal the roots
      of lambda^2 - a*lambda + 2 for a = +1 / a = -1 and coincide with the
      H / Gamma Ramanujan sets respectively (1e-12).
  E5  the two spills carry DISTINCT D2 characters at N (5.1 corroboration).
  E6  the Perron mode at Gamma is exactly A1: U_g v = v for all 24 ops.
  E7  ruling R2 (spill follows its branch parent): 4 -> 2  (1.0 bit).
  E8  the surviving Z2 is the joint nu_L/nu_R + spill swap -- i.e.
      EXACTLY the orientation convention priced ~1 bit in Phase 1.3
      (ruling R3: cross-reference, 0 new bits here; one home only).
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
SQ2, SQ5, SQ7 = np.sqrt(2), np.sqrt(5), np.sqrt(7)


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


SADDLES = {
    "Gamma": np.zeros(3),
    "H": np.array([0.5, 0.5, -0.5]),
    "P": np.array([0.25, 0.25, 0.25]),
    "N": A_PRIM @ np.array([0.0, 0.5, 0.5]),
}


def family_of(saddle, lam):
    m = abs(lam)
    if abs(m - 2.0) < 1e-6:
        return "Perron"
    if abs(m - 1.0) < 1e-6:
        return "trivial"
    if saddle == "P":
        return "h_P"
    if saddle == "Gamma":
        return "h_Gamma"
    if saddle == "H":
        return "h_H"
    if abs(abs(lam.real) - SQ5 / 2) < 1e-6:
        return "h_N"
    return "h_H-spill" if lam.real > 0 else "h_Gamma-spill"


print("=" * 72)
print(" PHASE 5.2 -- the enumeration behind the frozen decision tree")
print("=" * 72)

# fresh family census + per-family saddle sets
fam_size, fam_saddles = {}, {}
spec = {nm: la.eigvals(B_of(k)) for nm, k in SADDLES.items()}
for nm, evs in spec.items():
    for lam in evs:
        f = family_of(nm, lam)
        fam_size[f] = fam_size.get(f, 0) + 1
        fam_saddles.setdefault(f, set()).add(nm)

FAMILIES = ["h_P", "h_Gamma", "h_H", "h_Gamma-spill", "h_H-spill",
            "h_N", "Perron", "trivial"]
ROLES = {"charged": 8, "nu_L": 6, "nu_R": 6, "spill_L": 2, "spill_R": 2,
         "higgs": 2, "dark": 4, "nonmatter": 18}
ROLE_LIST = list(ROLES)

# E1: size filter over all 8! bijections
adm1 = []
for pi in permutations(ROLE_LIST):
    f = dict(zip(FAMILIES, pi))
    if all(ROLES[f[fam]] == fam_size[fam] for fam in FAMILIES):
        adm1.append(f)
gate("E1 size-compatible bijections = 12 of 40320 -> N1 = 3.58 bits",
     len(adm1) == 12, f"count={len(adm1)}, bits={np.log2(max(len(adm1),1)):.2f}")

# E2: unique 2-mode family with a Gamma component
two_with_gamma = [fam for fam in FAMILIES
                  if fam_size[fam] == 2 and "Gamma" in fam_saddles[fam]]
gate("E2 exactly ONE 2-mode family has a Gamma component (Perron)",
     two_with_gamma == ["Perron"], f"{two_with_gamma}")

# E3: ruling R1
adm2 = [f for f in adm1 if "Gamma" in fam_saddles[
    next(fam for fam in FAMILIES if f[fam] == "higgs")]]
gate("E3 R1 (Higgs needs Gamma component): 12 -> 4 -> 2.0 bits",
     len(adm2) == 4, f"count={len(adm2)}")

# E4: Bass-branch tie, exact
def uniq(vals, tol=1e-9):
    out = []
    for z in vals:
        if not any(abs(z - w) < tol for w in out):
            out.append(complex(z))
    return out


def close_sets(a, b, tol=1e-9):
    key = lambda z: (round(z.real, 6), round(z.imag, 6))   # noqa: E731
    a, b = sorted(a, key=key), sorted(b, key=key)
    return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))


roots_p = [(1 + 1j * SQ7) / 2, (1 - 1j * SQ7) / 2]      # lambda^2 - lambda + 2
roots_m = [(-1 + 1j * SQ7) / 2, (-1 - 1j * SQ7) / 2]    # lambda^2 + lambda + 2
spill_H = [z for z in spec["N"] if family_of("N", z) == "h_H-spill"]
spill_G = [z for z in spec["N"] if family_of("N", z) == "h_Gamma-spill"]
ram_H = [z for z in spec["H"] if family_of("H", z) == "h_H"]
ram_G = [z for z in spec["Gamma"] if family_of("Gamma", z) == "h_Gamma"]
gate("E4 branch tie EXACT: spill_H = roots(a=+1) = H-Ramanujan set; "
     "spill_G = roots(a=-1) = Gamma-Ramanujan set",
     close_sets(spill_H, roots_p) and close_sets(spill_G, roots_m)
     and close_sets(uniq(ram_H), roots_p) and close_sets(uniq(ram_G), roots_m))

# E5: the two spills carry distinct D2 characters at N
kN = SADDLES["N"]
lgN = [g for g in range(24)
       if np.max(np.abs(k_image(g, kN) - kN - np.round(k_image(g, kN) - kN))) < 1e-9]
BN = B_of(kN)
evN, VN = la.eig(BN)


def char_of(lam):
    idx = int(np.argmin(np.abs(evN - lam)))
    v = VN[:, idx] / la.norm(VN[:, idx])
    return tuple(np.round(np.conj(v) @ U_edge(g, kN) @ v, 6) for g in lgN)


chi_H = {char_of(z) for z in spill_H}
chi_G = {char_of(z) for z in spill_G}
gate("E5 spill D2 characters: each spill one class, the two DISTINCT",
     len(chi_H) == 1 and len(chi_G) == 1 and chi_H != chi_G)

# E6: Perron at Gamma is A1 (U_g v = v, all 24 ops)
kG = SADDLES["Gamma"]
BG = B_of(kG)
evG, VG = la.eig(BG)
vP = VG[:, int(np.argmin(np.abs(evG - 2.0)))]
vP = vP / la.norm(vP)
worst = max(la.norm(U_edge(g, kG) @ vP - vP) for g in range(24))
gate("E6 Perron(Gamma) is exactly A1: U_g v = v for all 24 ops",
     worst < 1e-9, f"max dev={worst:.1e}")

# E7: ruling R2 -- spill role follows its branch parent's role
PARENT_FAM = {"h_H-spill": "h_H", "h_Gamma-spill": "h_Gamma"}
PARENT_ROLE = {"spill_R": "nu_R", "spill_L": "nu_L"}
adm3 = [f for f in adm2
        if all(f[PARENT_FAM[s]] == PARENT_ROLE[f[s]]
               for s in ("h_H-spill", "h_Gamma-spill"))]
gate("E7 R2 (branch tie): 4 -> 2 -> 1.0 bit", len(adm3) == 2,
     f"count={len(adm3)}")

# E8: the surviving Z2 is exactly the joint nu-orientation swap
def as_pairs(f):
    return tuple(sorted(f.items()))


sw = {as_pairs(f) for f in adm3}
ident = {"h_P": "charged", "h_Gamma": "nu_L", "h_H": "nu_R",
         "h_Gamma-spill": "spill_L", "h_H-spill": "spill_R",
         "h_N": "dark", "Perron": "higgs", "trivial": "nonmatter"}
swap = dict(ident)
swap.update({"h_Gamma": "nu_R", "h_H": "nu_L",
             "h_Gamma-spill": "spill_R", "h_H-spill": "spill_L"})
gate("E8 surviving Z2 = {dictionary, joint nu_L/nu_R+spill swap} "
     "= the Phase-1.3 orientation bit (R3: 0 new bits, one home only)",
     sw == {as_pairs(ident), as_pairs(swap)})

print("\n--- the verified decision tree (panel rules on R1-R3) ---")
print("  N0 log2(8!) = 15.30   (the ESTIMATE)")
print("  N1 size-compatibility -> 12  = 3.58 bits   [(a) LEDGER CORRECTION]")
print("  N2 + R1 (Higgs@Gamma) -> 4   = 2.00 bits   [(b2) pre-5.1 physics]")
print("  N3 + R2 (branch tie)  -> 2   = 1.00 bit    [(b2); 5.1 corroborates]")
print("  N4 + R3 (1.3 cross-ref) -> 1 = 0 new bits  [(c) lives in 1.3 row]")
print("  (b1) 5.1-specific kills on THIS row: 0 bits -- 5.1's value =")
print("       kinematic forcing (P doublets etc.), hardening other rows.")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- tree verified; ledger row awaits the panel")
print("=" * 72)
sys.exit(0)
