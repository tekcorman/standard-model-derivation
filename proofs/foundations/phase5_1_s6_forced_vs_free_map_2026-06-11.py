#!/usr/bin/env python3
"""Phase 5.1 S6 -- the forced-vs-free map over all 48 saddle modes.

Spec: docs/scoping/phase5_1_ebr_spec_2026-06-11.md (S6, high). Inputs are
FINAL per S5: forcing mechanisms = {little-group irreps (S1-S3),
Ihara-Bass +/-1 content}; the EBR layer adds nothing (gated negative).

This probe assembles the deliverable the spec froze: the 48-row table
(saddle, eigenvalue, walker family, irrep class, dim, forcing mechanism)
and the MECHANICAL residual-freedom count at the family level. The walker
families are the 8 rows of the 2026-05-27 dictionary
(docs/theorems/theorem_walker_matter_unification_2026-05-27.md sec 2.3),
identified per mode by the dictionary's own (saddle, |lambda|, Re lambda)
classes -- Re lambda in {+-sqrt3/2 (h_P), +-1/2 (h_Gamma/h_H), +-sqrt5/2
(h_N)} for the Ramanujan modes.

Gates:
  T1 family census matches the dictionary table exactly:
     h_P 8, h_Gamma 6, h_H 6, h_Gamma-spill 2, h_H-spill 2, h_N 4,
     Perron 2, trivial 18 (total 48).
  T2 cluster forcing census: 26 eigenclusters across the 4 saddles;
     24 FORCED as single little-group irreps (Gamma 5, H 5, P 6, N 8);
     the 2 remaining (the +-1 doublets at N) are Ihara-Bass-only.
     Dual-mechanism: the +-1 content is ALSO little-group-forced at
     Gamma/H/P (irrep dims 3/2 there), IB-only at N.
  T3 family symmetry signatures (multiset of (saddle, irrep class, dim)
     over the family's modes) are PAIRWISE DISTINCT for all 8 families
     -> NO family-level permutation respects the space-group labels:
     the mechanical residual family-permutation freedom is log2(1) = 0
     bits UNDER SPACE-GROUP LABELS ALONE.
  T4 intra-family / same-class freedoms inventoried (these are NOT in the
     A5-mass 15.3 = log2(8!) scope, listed for 5.2 completeness):
     the two h_Gamma triplets are copies of ONE O-irrep class (their
     eigenvalue assignment is symmetry-free); likewise h_H; at P the two
     {1,w}-type doublets are one class, the two {1,w2}-type another.

WHAT THIS DOES NOT SETTLE (5.2 territory, xhigh + ultracode per the
adjudicated effort map): the family -> PHYSICAL-CONTENT assignment
freedom. Symmetry distinguishing the 8 families is not the same as
symmetry naming which family is "charged fermions". The 5.2 re-pricing
must decompose the 15.3-bit estimate into (a) bookkeeping correction
(permutations the estimate counted that were never structurally
available), (b) symmetry-forced reduction (this phase's labels), and
(c) residual content-assignment bits (priced, possibly closed by other
banked facts each priced separately). The mirror identity
B(k+Delta) = -B(k) (Gamma<->H exchange) is NOT a space-group element and
is deliberately NOT used here; its status for the count is a 5.2
methodology decision (it relates h_Gamma<->h_H and the two Perron modes:
the nu_L/nu_R orientation bit already priced ~1 in Phase 1.3).
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


# --- space group + edge rep (native, as S1; duplicated by repo pattern)
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
LG = {nm: [g for g in range(24)
           if np.max(np.abs(k_image(g, k) - k - np.round(k_image(g, k) - k))) < 1e-9]
      for nm, k in SADDLES.items()}

SQ2, SQ3, SQ5 = np.sqrt(2), np.sqrt(3), np.sqrt(5)


def family_of(saddle, lam):
    """Walker family per the 2026-05-27 dictionary classes."""
    m = abs(lam)
    if abs(m - 2.0) < 1e-6:
        return "Perron"
    if abs(m - 1.0) < 1e-6:
        return "trivial"
    assert abs(m - SQ2) < 1e-6, lam
    re = abs(lam.real)
    if saddle == "P":
        return "h_P"
    if saddle == "Gamma":
        return "h_Gamma"
    if saddle == "H":
        return "h_H"
    # N: spillover and dark families split by Re lambda
    if abs(re - SQ5 / 2) < 1e-6:
        return "h_N"
    assert abs(re - 0.5) < 1e-6, lam
    return "h_H-spill" if lam.real > 0 else "h_Gamma-spill"


# --- per-saddle irrep classes (group-average; as S3) and B clusters
def saddle_analysis(nm, k):
    lg = LG[nm]
    Us = {g: U_edge(g, k) for g in lg}
    rng_ = np.random.default_rng(11)
    H0 = rng_.normal(size=(NE, NE)) + 1j * rng_.normal(size=(NE, NE))
    H0 = H0 + H0.conj().T
    Hb = sum(U @ H0 @ U.conj().T for U in Us.values()) / len(lg)
    ev, V = la.eigh(Hb)
    blocks, i = [], 0
    while i < NE:
        grp = [i]
        while i + 1 < NE and abs(ev[i + 1] - ev[i]) < 1e-8:
            i += 1
            grp.append(i)
        blocks.append(V[:, grp])
        i += 1
    chars, classes = [], []
    for Q in blocks:
        chi = tuple(np.round(np.trace(Q.conj().T @ Us[g] @ Q), 6) for g in lg)
        chars.append(chi)
    class_of_block = []
    class_keys = []
    for chi in chars:
        for ci, ck in enumerate(class_keys):
            if max(abs(np.array(chi) - np.array(ck))) < 1e-5:
                class_of_block.append(ci)
                break
        else:
            class_keys.append(chi)
            class_of_block.append(len(class_keys) - 1)
    # isotypic projectors
    Piso = {}
    for ci, ck in enumerate(class_keys):
        bi = class_of_block.index(ci)
        d = blocks[bi].shape[1]
        Pa = (d / len(lg)) * sum(np.conj(ck[ig]) * Us[g]
                                 for ig, g in enumerate(lg))
        Piso[ci] = (d, Pa)
    # B spectral clusters
    Bk = B_of(k)
    evb, Vb = la.eig(Bk)
    key = np.round(evb, 7)
    order = np.lexsort((key.imag, key.real))
    evb, Vb = evb[order], Vb[:, order]
    Vbi = la.inv(Vb)
    clusters, i = [], 0
    while i < NE:
        grp = [i]
        while i + 1 < NE and abs(evb[i + 1] - evb[grp[0]]) < 1e-7:
            i += 1
            grp.append(i)
        clusters.append((evb[grp[0]], len(grp),
                         Vb[:, grp] @ Vbi[grp, :]))
        i += 1
    out = []
    for lam, dim_c, Pc in clusters:
        content = []
        copies = 0.0
        for ci, (d, Pa) in Piso.items():
            t = np.real(np.trace(Pa @ Pc))
            if np.round(t) > 0:
                content.append((ci, d, int(np.round(t))))
                copies += np.round(t) / d
        out.append(dict(lam=lam, dim=dim_c, content=content,
                        forced=abs(copies - 1.0) < 1e-9))
    return out


print("=" * 72)
print(" PHASE 5.1 S6 -- the forced-vs-free map over the 48 saddle modes")
print("=" * 72)

TABLE = []           # one row per cluster, modes counted by dim
fam_counts = {}
forced_census = {}
for nm, k in SADDLES.items():
    rows = saddle_analysis(nm, k)
    forced_census[nm] = sum(1 for r in rows if r["forced"])
    for r in rows:
        fam = family_of(nm, r["lam"])
        fam_counts[fam] = fam_counts.get(fam, 0) + r["dim"]
        is_pm1 = abs(abs(r["lam"]) - 1.0) < 1e-6 and abs(r["lam"].imag) < 1e-6
        mech = ("LG-irrep + IB" if (r["forced"] and is_pm1)
                else "LG-irrep" if r["forced"]
                else "IB only" if is_pm1 else "NONE")
        TABLE.append(dict(saddle=nm, lam=r["lam"], dim=r["dim"], fam=fam,
                          content=r["content"], forced=r["forced"], mech=mech))

print("\n--- the 48-mode table (one row per eigencluster) ---")
print(f"  {'saddle':>6} {'eigenvalue':>22} {'dim':>4} {'family':>14} "
      f"{'irrep(class:dim)':>18} {'mechanism':>14}")
for r in TABLE:
    cstr = "+".join(f"c{ci}:d{d}" for ci, d, n in r["content"])
    print(f"  {r['saddle']:>6} {np.round(r['lam'], 6)!s:>22} {r['dim']:>4} "
          f"{r['fam']:>14} {cstr:>18} {r['mech']:>14}")

print("\n--- gates ---")
expected = {"h_P": 8, "h_Gamma": 6, "h_H": 6, "h_Gamma-spill": 2,
            "h_H-spill": 2, "h_N": 4, "Perron": 2, "trivial": 18}
gate("T1 family census = dictionary table (8,6,6,2,2,4,2,18; total 48)",
     fam_counts == expected and sum(fam_counts.values()) == 48,
     f"{fam_counts}")

n_clusters = len(TABLE)
n_forced = sum(forced_census.values())
gate("T2 cluster census: 26 clusters; 24 LG-forced (5/5/6/8); "
     "2 IB-only (the +-1 doublets at N)",
     n_clusters == 26 and n_forced == 24
     and forced_census == {"Gamma": 5, "H": 5, "P": 6, "N": 8}
     and sum(1 for r in TABLE if r["mech"] == "IB only") == 2
     and all(r["mech"] != "NONE" for r in TABLE),
     f"clusters={n_clusters}, forced={forced_census}")

# T3: family symmetry signatures pairwise distinct
sig = {}
for fam in expected:
    s = sorted((r["saddle"], tuple(sorted(r["content"])), r["dim"])
               for r in TABLE if r["fam"] == fam)
    sig[fam] = tuple(s)
pairs_equal = [(f1, f2) for i, f1 in enumerate(sorted(sig))
               for f2 in sorted(sig)[i + 1:] if sig[f1] == sig[f2]]
gate("T3 all 8 family signatures pairwise DISTINCT -> residual "
     "family-permutation freedom under space-group labels = log2(1) = 0 bits",
     not pairs_equal, f"colliding pairs: {pairs_equal}")

# T4: intra-family same-class freedoms (outside the 15.3 scope; for 5.2)
print("\n--- T4: intra-family same-irrep-class freedoms (5.2 inventory) ---")
intra = []
for nm in SADDLES:
    rows = [r for r in TABLE if r["saddle"] == nm]
    byclass = {}
    for r in rows:
        for ci, d, n in r["content"]:
            byclass.setdefault(ci, []).append(np.round(r["lam"], 4))
    for ci, lams in byclass.items():
        if len(lams) > 1:
            intra.append((nm, ci, lams))
            print(f"  {nm}: class c{ci} hosts {len(lams)} clusters {lams} "
                  f"-> their eigenvalue assignment is symmetry-FREE (dynamics)")
intra_census = {}
for nm, ci, lams in intra:
    intra_census[nm] = intra_census.get(nm, 0) + 1
gate("T4 same-class multi-cluster inventory frozen: Gamma 1 (conj Ram "
     "pair), H 1, P 3 (both Ram pairs + the +-1 pair), N 4 -- 9 entries",
     len(intra) == 9
     and intra_census == {"Gamma": 1, "H": 1, "P": 3, "N": 4},
     f"{intra_census}")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- 5.1 forced-vs-free map COMPLETE")
print("=" * 72)
sys.exit(0)
