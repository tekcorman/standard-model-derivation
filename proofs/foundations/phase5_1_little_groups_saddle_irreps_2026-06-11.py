#!/usr/bin/env python3
"""Phase 5.1 (S1-S3) -- native I4_132 little groups at the saddles.

Spec: internal research notes (FROZEN, SHA-256
79419b8e..., registered in docs/audits/registers/frozen_spec_hashes.md).

THE QUESTION: which parts of the 48-saddle-mode degeneracy structure
(12-dim Bloch-Hashimoto fibers at Gamma, P, N, H) are SYMMETRY-FORCED by
space group I4_132 (No. 214)? Forced degeneracy = modes inside one
little-group irrep block; any symmetric operator must keep them degenerate.

NATIVE CONSTRUCTION (no table imports):
  - the 24 point operations are enumerated as signed permutation matrices
    (det +1); the fractional translation tau of each is SOLVED from the
    crystal itself (atom set + bond set preservation mod the BCC lattice);
  - the Bloch edge representation U_g(k) is derived in the repo's exact
    convention (B[b,a] = e^{2pi i k.c_b}; Bloch sum e^{-2pi i k.(d+c_a)}):
        [U_g(k)]_{g.a, a} = exp{2pi i [k'.(d_i(g,a) + c_{g.a}) - k.c_a]},
        k' = R_prim^{-T} k,
    where d_i(g,a) is the (integer, primitive-basis) cell offset of the
    image edge's source vertex;
  - factor systems omega(g,h) on the little groups are extracted from
    U_g U_h = omega U_{gh}; their (non)triviality is decided constructively
    (a 1-dim block trivializes; all-even irreducible blocks with no even
    ordinary irrep dimension available proves nontrivial -- the ordinary
    dimension menu is computed natively from the regular representation of
    the abstract little co-group, not cited).

GATES (construction, all machine precision):
  G1 space group: exactly 24 coset classes, one tau class per rotation;
     closure; atom+bond preservation; point parts = 432 trace multiset;
     site stabilizers (atom: 6; directed edge: single orbit, order 2).
  G2 R_prim integer; U_g(k) unitary; d_i integer.
  G3 intertwining U_g(k) B(k) U_g(k)^dag = B(R_prim^{-T} k) for all 24 ops
     at the 4 saddles + 3 random k, < 1e-12.
  G4 little co-group orders (Gamma 24, H 24, P 12, N 4); subgroup closure;
     factor system scalar; 2-cocycle identity.
  G5 block decomposition stable across 2 random averages; every block
     irreducible (projective character norm 1); isotypic projectors
     idempotent and complete.
  G6 B-eigencluster spectral projectors idempotent/complete/commuting with
     the little group; integer irrep content; C3 regression: P3-convention
     content of every cluster reproduces the banked phase1_3 tables.

FINDINGS GATES (frozen after first run, per spec: hypothesis failure is a
finding, not a probe failure -- what is TRUE gets gated):
  F-P   P: little co-group T (order 12), factor system NONTRIVIAL, all six
        blocks 2-dim irreducible -> every B(P) doublet (4 Ramanujan + 2
        (+/-1)) is a single projective irrep: FORCED. B(P) cannot have a
        nondegenerate eigenvalue. (Pre-registered H-P.)
  F-GH  Gamma/H: ordinary O irreps; Perron = 1-dim; each Ramanujan triplet
        = a single 3-dim irrep: FORCED. (Pre-registered H-GammaH.)
  F-N   N: little co-group D2 (order 4), trivial factor system, all-1-dim
        irreps -> NOTHING little-group-forced at N; the (+/-1) doublets
        there are Ihara-Bass (zeta-structural) content, a distinct forcing
        mechanism. (Pre-registered H-N.)
"""
import os
import sys
from itertools import permutations, product

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, ATOMS, A_PRIM  # noqa: E402

TOL = 1e-12
CLUSTER_TOL = 1e-7
FAILURES = []
M_CART = A_PRIM.T          # primitive-integer coords -> cartesian (cubic a=1)
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
    """Is cartesian vector v a BCC lattice translation (a=1 cubic)?"""
    w = np.asarray(v, float)
    if np.max(np.abs(w - np.round(w))) < 1e-9:
        return True
    w2 = w - 0.5
    return np.max(np.abs(w2 - np.round(w2))) < 1e-9


def canon_tau(v):
    """Canonical representative of tau mod the BCC lattice."""
    cands = [mod1(v), mod1(np.asarray(v, float) + 0.5)]
    keys = [tuple(np.round(c, 6)) for c in cands]
    return cands[keys.index(min(keys))]


def prim_int(v_cart):
    """Cartesian lattice vector -> integer primitive coords (gated)."""
    d = M_INV @ np.asarray(v_cart, float)
    di = np.round(d)
    assert np.max(np.abs(d - di)) < 1e-9, f"non-integer primitive coords {d}"
    return di.astype(int)


# ----------------------------------------------------------------------
# Section 0 -- edges and the Bloch-Hashimoto fiber (repo convention)
# ----------------------------------------------------------------------
bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
NE = len(EDGES)
E_INDEX = {e: a for a, e in enumerate(EDGES)}
REV = {}
for a, (i, j, c) in enumerate(EDGES):
    REV[a] = E_INDEX[(j, i, tuple(-x for x in c))]


def B_of(k):
    B = np.zeros((NE, NE), dtype=complex)
    for a, (i, j, c) in enumerate(EDGES):
        for b, (i2, j2, c2) in enumerate(EDGES):
            if i2 == j and b != REV[a]:
                B[b, a] = np.exp(2j * np.pi * np.dot(k, np.asarray(c2, float)))
    return B


SADDLES = {
    "Gamma": np.zeros(3),
    "H": np.array([0.5, 0.5, -0.5]),
    "P": np.array([0.25, 0.25, 0.25]),
    "N": A_PRIM @ np.array([0.0, 0.5, 0.5]),   # = (1/2, 0, 0) primitive
}


# ----------------------------------------------------------------------
# Section 1 (S1) -- the space group, natively
# ----------------------------------------------------------------------
def rotations24():
    """All proper rotations of the cube: signed permutations, det +1."""
    rots = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                R[row, col] = s
            if abs(la.det(R) - 1.0) < 1e-9:
                rots.append(R)
    return rots


def atom_of(pos):
    """Which atom (mod BCC lattice) sits at cartesian pos? -> (index, L)."""
    for j in range(4):
        L = pos - ATOMS[j]
        if is_bcc(L):
            return j, L
    return None, None


def op_preserves(R, tau):
    """Does (R|tau) map the atom set and the bond set to themselves?"""
    amap = {}
    for i in range(4):
        j, _ = atom_of(R @ ATOMS[i] + tau)
        if j is None:
            return False, None
        amap[i] = j
    for (i, j, c) in EDGES:
        src = R @ ATOMS[i] + tau
        tgt = R @ (ATOMS[j] + M_CART @ np.asarray(c, float)) + tau
        i2, Li = atom_of(src)
        j2, Lj = atom_of(tgt)
        c2 = tuple(prim_int(Lj) - prim_int(Li))
        if (i2, j2, c2) not in E_INDEX:
            return False, None
    return True, amap


def build_space_group():
    ops = []
    for R in rotations24():
        found = []
        seen = set()
        for j in range(4):
            tau = canon_tau(ATOMS[j] - R @ ATOMS[0])
            key = tuple(np.round(tau, 6))
            if key in seen:
                continue
            seen.add(key)
            ok, _ = op_preserves(R, tau)
            if ok:
                found.append(tau)
        if len(found) == 1:
            ops.append((R, found[0]))
        else:
            ops.append(None)
    return ops


print("=" * 72)
print(" PHASE 5.1 (S1-S3) -- native I4_132 little groups at the saddles")
print("=" * 72)
print("\n--- S1: space group construction ---")

OPS = build_space_group()
gate("G1a exactly one tau class per rotation, 24/24 ops",
     all(o is not None for o in OPS), f"found={sum(o is not None for o in OPS)}")
OPS = [o for o in OPS if o is not None]
N_OPS = len(OPS)

# closure + multiplication table
OP_KEY = {}
for idx, (R, t) in enumerate(OPS):
    OP_KEY[(tuple(R.astype(int).ravel()), tuple(np.round(t, 6)))] = idx


def op_index(R, tau):
    return OP_KEY.get((tuple(np.round(R).astype(int).ravel()),
                       tuple(np.round(canon_tau(tau), 6))))


MULT = np.full((N_OPS, N_OPS), -1, dtype=int)
for a1, (R1, t1) in enumerate(OPS):
    for a2, (R2, t2) in enumerate(OPS):
        MULT[a1, a2] = op_index(R1 @ R2, R1 @ t2 + t1) if op_index(R1 @ R2, R1 @ t2 + t1) is not None else -1
gate("G1c group closure (24x24 multiplication table total)",
     np.all(MULT >= 0) and all(len(set(MULT[i, :])) == N_OPS for i in range(N_OPS)))

traces = sorted(int(np.round(np.trace(R))) for R, _ in OPS)
gate("G1d point parts = 432 (trace multiset E:3, 8C3:0, 9C2:-1, 6C4:+1)",
     traces == sorted([3] + [0] * 8 + [-1] * 9 + [1] * 6), f"{traces}")

stab0 = [g for g, (R, t) in enumerate(OPS)
         if atom_of(R @ ATOMS[0] + t)[0] == 0]
gate("G1e atom-0 site stabilizer order 6 (Wyckoff 8a, .32)",
     len(stab0) == 6, f"order={len(stab0)}")

# edge action: emap[g][a] = (a', d_i) with d_i integer primitive offset
EMAP = []
for (R, t) in OPS:
    rows = []
    for (i, j, c) in EDGES:
        src = R @ ATOMS[i] + t
        tgt = R @ (ATOMS[j] + M_CART @ np.asarray(c, float)) + t
        i2, Li = atom_of(src)
        j2, Lj = atom_of(tgt)
        di, dj = prim_int(Li), prim_int(Lj)
        rows.append((E_INDEX[(i2, j2, tuple(dj - di))], di))
    EMAP.append(rows)
gate("G1b bond set preserved; edge action is a permutation for all ops",
     all(sorted(r[0] for r in rows) == list(range(NE)) for rows in EMAP))

orbit_reached = {EMAP[g][0][0] for g in range(N_OPS)}
edge_stab = [g for g in range(N_OPS) if EMAP[g][0][0] == 0]
gate("G1f directed edges = ONE orbit, stabilizer C2 (band rep = Ind_C2(triv))",
     len(orbit_reached) == NE and len(edge_stab) == 2,
     f"orbit={len(orbit_reached)}/12, stab={len(edge_stab)}")

R_PRIM = []
ok_int = True
for (R, t) in OPS:
    Rp = M_INV @ R @ M_CART
    Rpi = np.round(Rp)
    ok_int &= bool(np.max(np.abs(Rp - Rpi)) < 1e-9)
    R_PRIM.append(Rpi.astype(int))
gate("G2a R_prim integer for all 24 ops", ok_int)


# ----------------------------------------------------------------------
# Section 2 -- the Bloch edge representation U_g(k)
# ----------------------------------------------------------------------
def k_image(g, k):
    return la.inv(R_PRIM[g]).T @ np.asarray(k, float)


def U_of(g, k):
    k = np.asarray(k, float)
    kp = k_image(g, k)
    U = np.zeros((NE, NE), dtype=complex)
    for a, (i, j, c) in enumerate(EDGES):
        a2, di = EMAP[g][a]
        c2 = np.asarray(EDGES[a2][2], float)
        U[a2, a] = np.exp(2j * np.pi * (kp @ (di + c2) - k @ np.asarray(c, float)))
    return U


print("\n--- S1 gates: representation ---")
rng = np.random.default_rng(12345)
test_ks = list(SADDLES.values()) + [rng.uniform(-0.5, 0.5, 3) for _ in range(3)]
worst_int, worst_uni = 0.0, 0.0
for k in test_ks:
    Bk = B_of(k)
    for g in range(N_OPS):
        U = U_of(g, k)
        worst_uni = max(worst_uni, la.norm(U @ U.conj().T - np.eye(NE)))
        worst_int = max(worst_int, la.norm(U @ Bk @ U.conj().T - B_of(k_image(g, k))))
gate("G2b U_g(k) unitary at saddles + 3 random k", worst_uni < TOL, f"max={worst_uni:.1e}")
gate("G3 intertwining U B(k) U^dag = B(R^-T k), all 24 ops x 7 k", worst_int < TOL,
     f"max={worst_int:.1e}")


# ----------------------------------------------------------------------
# Section 3 (S2) -- little groups and factor systems
# ----------------------------------------------------------------------
def little_group(k):
    return [g for g in range(N_OPS)
            if np.max(np.abs(k_image(g, k) - k - np.round(k_image(g, k) - k))) < 1e-9]


print("\n--- S2: little co-groups + factor systems ---")
LG = {nm: little_group(k) for nm, k in SADDLES.items()}
orders = {nm: len(v) for nm, v in LG.items()}
gate("G4a little co-group orders Gamma=24, H=24, P=12, N=4",
     orders == {"Gamma": 24, "H": 24, "P": 12, "N": 4}, f"{orders}")
gate("G4a' little groups closed under multiplication",
     all(all(MULT[a, b] in set(lg) for a in lg for b in lg) for lg in LG.values()))

OMEGA, U_SADDLE = {}, {}
ok_scalar, ok_cocycle = True, True
for nm, k in SADDLES.items():
    lg = LG[nm]
    Us = {g: U_of(g, k) for g in lg}
    U_SADDLE[nm] = Us
    om = {}
    for a in lg:
        for b in lg:
            W = Us[a] @ Us[b] @ Us[MULT[a, b]].conj().T
            s = W[0, 0] if abs(W[0, 0]) > 0.5 else W.ravel()[np.argmax(np.abs(W.ravel()))]
            ok_scalar &= bool(la.norm(W - s * np.eye(NE)) < 1e-9 and abs(abs(s) - 1) < 1e-9)
            om[(a, b)] = s
    for a in lg:
        for b in lg:
            for cc in lg:
                lhs = om[(a, b)] * om[(MULT[a, b], cc)]
                rhs = om[(a, MULT[b, cc])] * om[(b, cc)]
                ok_cocycle &= bool(abs(lhs - rhs) < 1e-8)
    OMEGA[nm] = om
gate("G4b factor system scalar, |omega|=1, all saddles", ok_scalar)
gate("G4c 2-cocycle identity, all saddles", ok_cocycle)


# ----------------------------------------------------------------------
# Section 4 (S3) -- irrep blocks, isotypic projectors, B alignment
# ----------------------------------------------------------------------
def group_average_blocks(Us, seed):
    rng_ = np.random.default_rng(seed)
    H0 = rng_.normal(size=(NE, NE)) + 1j * rng_.normal(size=(NE, NE))
    H0 = H0 + H0.conj().T
    Hb = sum(U @ H0 @ U.conj().T for U in Us.values()) / len(Us)
    ev, V = la.eigh(Hb)
    blocks, i = [], 0
    while i < NE:
        grp = [i]
        while i + 1 < NE and abs(ev[i + 1] - ev[i]) < 1e-8:
            i += 1
            grp.append(i)
        blocks.append(V[:, grp])
        i += 1
    return blocks


def ordinary_irrep_dims(lg):
    """Distinct ordinary irrep dims of the abstract little co-group,
    computed natively from its regular representation."""
    n = len(lg)
    pos = {g: i for i, g in enumerate(lg)}
    regs = {}
    for a in lg:
        Lm = np.zeros((n, n))
        for b in lg:
            Lm[pos[MULT[a, b]], pos[b]] = 1.0
        regs[a] = Lm
    rng_ = np.random.default_rng(7)
    H0 = rng_.normal(size=(n, n)) + 1j * rng_.normal(size=(n, n))
    H0 = H0 + H0.conj().T   # complex Hermitian: splits conjugate irrep pairs
    Hb = sum(Lm @ H0 @ Lm.T for Lm in regs.values()) / n
    ev = la.eigvalsh(Hb)
    dims, i = [], 0
    while i < n:
        m = 1
        while i + m < n and abs(ev[i + m] - ev[i]) < 1e-8:
            m += 1
        dims.append(m)
        i += m
    return sorted(set(dims))


def spectral_projectors(B):
    ev, V = la.eig(B)
    order = np.argsort(np.round(ev, 7).view(float).reshape(-1, 2)[:, 0] * 1e6
                       + np.round(ev, 7).view(float).reshape(-1, 2)[:, 1])
    ev, V = ev[order], V[:, order]
    Vinv = la.inv(V)
    clusters, i = [], 0
    while i < NE:
        grp = [i]
        while i + 1 < NE and abs(ev[i + 1] - ev[grp[0]]) < CLUSTER_TOL:
            i += 1
            grp.append(i)
        clusters.append((ev[grp[0]], grp))
        i += 1
    projs = [(lam, V[:, grp] @ Vinv[grp, :]) for lam, grp in clusters]
    recon = la.norm(V @ np.diag(ev) @ Vinv - B)
    return projs, recon


# banked phase1_3 C3 content (regression targets), P3 convention
w = np.exp(2j * np.pi / 3)
C3_R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)


def build_P3():
    sigma = {0: 0, 1: 3, 3: 2, 2: 1}
    P3 = np.zeros((NE, NE))
    for a, (i, j, c) in enumerate(EDGES):
        v = C3_R @ (ATOMS[j] + M_CART @ np.asarray(c, float) - ATOMS[i])
        for b, (i2, j2, c2) in enumerate(EDGES):
            if (i2, j2) == (sigma[i], sigma[j]) and np.allclose(
                    ATOMS[j2] + M_CART @ np.asarray(c2, float) - ATOMS[i2], v, atol=1e-9):
                P3[b, a] = 1.0
                break
    return P3


P3 = build_P3()


def c3_label(z):
    for nm, val in (("1", 1), ("w", w), ("w2", w ** 2)):
        if abs(z - val) < 1e-6:
            return nm
    return "?"


BANKED = {
    "P": {"ram": (["1", "w"], ["1", "w2"]), "pm1": ["w", "w2"]},
    "Gamma": {"ram": ["1", "w", "w2"]},
    "H": {"ram": ["1", "w", "w2"]},
}

print("\n--- S3: irrep blocks vs B eigenclusters, per saddle ---")
SUMMARY = {}
for nm, k in SADDLES.items():
    lg, Us = LG[nm], U_SADDLE[nm]
    nG = len(lg)
    print(f"\n  === {nm}  (little co-group order {nG}) ===")

    # irrep blocks from two independent group averages
    blocks_a = group_average_blocks(Us, 11)
    blocks_b = group_average_blocks(Us, 12)
    dims_a = sorted(b.shape[1] for b in blocks_a)
    dims_b = sorted(b.shape[1] for b in blocks_b)
    gate(f"G5a [{nm}] block dims stable across 2 random averages",
         dims_a == dims_b, f"dims={dims_a}")
    blocks = blocks_a

    # projective characters per block; irreducibility norm
    chars, ok_irr = [], True
    for Q in blocks:
        chi = np.array([np.trace(Q.conj().T @ Us[g] @ Q) for g in lg])
        nrm = np.sum(np.abs(chi) ** 2) / nG
        ok_irr &= bool(abs(nrm - 1.0) < 1e-8)
        chars.append(chi)
    gate(f"G5b [{nm}] every block irreducible (projective char norm 1)", ok_irr)

    # group equivalent blocks into classes; isotypic projectors
    classes = []
    for bi, chi in enumerate(chars):
        for cl in classes:
            if la.norm(chi - chars[cl[0]]) < 1e-6:
                cl.append(bi)
                break
        else:
            classes.append([bi])
    Piso, ok_proj = [], True
    for cl in classes:
        d = blocks[cl[0]].shape[1]
        chi = chars[cl[0]]
        Pa = (d / nG) * sum(np.conj(chi[ig]) * Us[g] for ig, g in enumerate(lg))
        ok_proj &= bool(la.norm(Pa @ Pa - Pa) < 1e-8)
        Piso.append((d, len(cl), Pa))
    ok_proj &= bool(la.norm(sum(p for _, _, p in Piso) - np.eye(NE)) < 1e-8)
    gate(f"G5c [{nm}] isotypic projectors idempotent + complete", ok_proj,
         f"classes (dim x mult): {[(d, m) for d, m, _ in Piso]}")

    # factor-system class, constructively
    odims = ordinary_irrep_dims(lg)
    bdims = sorted(set(b.shape[1] for b in blocks))
    if 1 in bdims:
        cocycle = "TRIVIAL (1-dim block exists -> explicit trivializing cochain)"
        nontrivial = False
    elif any(d not in odims for d in bdims):
        cocycle = (f"NONTRIVIAL (irreducible block dims {bdims} impossible "
                   f"ordinarily: ordinary dims {odims})")
        nontrivial = True
    else:
        cocycle = f"INCONCLUSIVE (block dims {bdims} within ordinary menu {odims})"
        nontrivial = None
    print(f"  factor system: {cocycle}")

    # B eigenclusters and their irrep content
    Bk = B_of(k)
    projs, recon = spectral_projectors(Bk)
    gate(f"G6a [{nm}] B diagonalizable (eigvec reconstruction)", recon < 1e-9,
         f"resid={recon:.1e}")
    ok_comm = all(la.norm(Pc @ Us[g] - Us[g] @ Pc) < 1e-8
                  for _, Pc in projs for g in lg)
    ok_sum = la.norm(sum(Pc for _, Pc in projs) - np.eye(NE)) < 1e-8
    gate(f"G6a' [{nm}] spectral projectors commute with LG + sum to I",
         ok_comm and ok_sum)

    rows, ok_int_content, all_forced = [], True, True
    for lam, Pc in projs:
        dim_c = int(np.round(np.real(np.trace(Pc))))
        copies = 0.0
        content = []
        for ci, (d, m, Pa) in enumerate(Piso):
            t = np.real(np.trace(Pa @ Pc))
            ok_int_content &= bool(abs(t - np.round(t)) < 1e-7)
            if np.round(t) > 0:
                content.append((ci, d, int(np.round(t))))
                copies += np.round(t) / d
        forced = (abs(copies - 1.0) < 1e-9)
        all_forced &= forced
        # C3 content in the banked (P3) convention, where applicable
        c3c = ""
        if nm in ("Gamma", "H", "P"):
            ev_b, V = la.eig(Bk)
            idx = np.where(np.abs(ev_b - lam) < CLUSTER_TOL)[0]
            Vc = V[:, idx]
            Wr = la.pinv(Vc) @ P3 @ Vc
            c3c = sorted(c3_label(z) for z in la.eigvals(Wr))
        rows.append((lam, dim_c, content, forced, c3c))
    gate(f"G6b [{nm}] irrep content of every cluster is integer", ok_int_content)

    print(f"  {'eigenvalue':>22} {'dim':>4} {'irrep content (class:dxn)':>28} "
          f"{'verdict':>10}  C3(P3-conv)")
    for lam, dim_c, content, forced, c3c in rows:
        cstr = "+".join(f"cl{ci}:d{d}" + (f"x{n // d}" if n // d > 1 else "")
                        for ci, d, n in content)
        tag = "FORCED" if forced else "COMPOSITE"
        ib = "  [IB +/-1]" if abs(abs(lam) - 1.0) < 1e-6 and abs(lam.imag) < 1e-6 else ""
        print(f"  {np.round(lam, 6)!s:>22} {dim_c:>4} {cstr:>28} {tag:>10}"
              f"  {c3c}{ib}")

    # C3 regression against banked phase1_3 content
    if nm in BANKED:
        ok_reg = True
        for lam, dim_c, content, forced, c3c in rows:
            if abs(abs(lam) - np.sqrt(2)) < 1e-6:
                if nm == "P":
                    ok_reg &= (c3c in (sorted(BANKED["P"]["ram"][0]),
                                       sorted(BANKED["P"]["ram"][1])))
                else:
                    ok_reg &= (c3c == sorted(BANKED[nm]["ram"]))
            elif nm == "P" and abs(abs(lam) - 1.0) < 1e-6:
                ok_reg &= (c3c == sorted(BANKED["P"]["pm1"]))
        gate(f"G6c [{nm}] C3 content reproduces banked phase1_3 tables", ok_reg)

    SUMMARY[nm] = dict(order=nG, nontrivial=nontrivial,
                       dims=[(d, m) for d, m, _ in Piso],
                       all_forced=all_forced,
                       n_clusters=len(rows), rows=rows)

# ----------------------------------------------------------------------
# Findings gates (frozen after first run -- see docstring)
# ----------------------------------------------------------------------
print("\n--- findings vs pre-registered hypotheses ---")
sP = SUMMARY["P"]
gate("F-P  P: T(12) co-group, NONTRIVIAL cocycle, six 2-dim irreps, "
     "all six B(P) doublets FORCED (H-P confirmed)",
     sP["order"] == 12 and sP["nontrivial"] is True
     and sorted(d for d, m in sP["dims"] for _ in range(m)) == [2] * 6
     and sP["all_forced"] and sP["n_clusters"] == 6)
sG, sH = SUMMARY["Gamma"], SUMMARY["H"]
gate("F-GH Gamma/H: ordinary O irreps; Perron 1-dim; Ramanujan triplets "
     "= single 3-dim irreps; every cluster FORCED (H-GammaH confirmed)",
     sG["nontrivial"] is False and sH["nontrivial"] is False
     and sG["all_forced"] and sH["all_forced"])
sN = SUMMARY["N"]
gate("F-N  N: D2(4) co-group, TRIVIAL cocycle, all 1-dim irreps -> "
     "little-group forces NOTHING at N; +/-1 doublets are Ihara-Bass "
     "content (H-N confirmed)",
     sN["order"] == 4 and sN["nontrivial"] is False
     and all(d == 1 for d, m in sN["dims"]))

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS")
print("=" * 72)
sys.exit(0)
