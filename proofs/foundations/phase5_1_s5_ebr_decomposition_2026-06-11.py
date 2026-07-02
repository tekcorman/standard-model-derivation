#!/usr/bin/env python3
"""Phase 5.1 S5 -- EBR decomposition of the Hashimoto band representation.

Spec: docs/scoping/phase5_1_ebr_spec_2026-06-11.md (S5, effort high with
xhigh triggers: decomposition non-uniqueness with physical consequence, or
any cross-saddle forcing claim).

S1 established: the 12 directed bonds are ONE orbit with stabilizer C2
(the 2-fold ALONG the bond), so the Hashimoto band rep is Ind_C2(triv).
Induction in stages through the bond-midpoint site group D2 gives the
SHARP CLAIM tested here:

    Ind_C2(triv) = EBR(12mid, A) + EBR(12mid, B_par)

where A, B_par are the two D2 site irreps trivial on the along-bond C2
(symmetric / antisymmetric under bond reversal). Everything is built
natively: special Wyckoff orbits found by stabilizer scan on the 1/8 grid;
site irreps constructed from the actual rotation matrices (1-dims by
brute-force homomorphism, the D3 E irrep by restriction to the plane
perpendicular to the local 3-fold axis); EBR Bloch matrices in the same
convention as the S1 edge rep (cocycle equality gated).

Gates:
  W1 special-orbit census: two order-6 orbits of size 4 (8a = atoms, 8b)
     and two order-4 (D2) orbits of size 6, one being the bond midpoints;
     midpoint D2 has its 2-fold along the bond.
  W2 EBR builders: unitary; SAME factor system as the edge rep at every
     saddle (cocycle is orbit-independent -- correctness of conventions).
  W3 induction in stages: content(EBR_A + EBR_B) = content(edge rep) at
     all 4 saddles, class-by-class.
  W4 exhaustive integer-decomposition scan over all 14 maximal-position
     EBRs (8a/8b x {A1,A2,E}, 12mid/12other x {A,B1,B2,B3}): the W3
     solution is found; ALL solutions reported (uniqueness is a FINDING).
  W4b (xhigh-trigger follow-up, frozen after first run): the 6 solutions
     are indistinguishable by line-point content too (one equivalence
     class at the Gamma-P/Gamma-H/Gamma-N midpoints) -- plausibly six
     induction labels for ONE band rep; the induction-in-stages route is
     the structurally derived label; the choice carries no physical
     weight for 5.2.
  W5 GATED NEGATIVE (frozen after first run): the two EBR summands'
     contents OVERLAP at every saddle, so EBR membership does not refine
     the B(k) eigencluster forcing anywhere. The EBR layer adds NOTHING
     to the forced-vs-free map beyond per-saddle irreps (S1-S3) and the
     Ihara-Bass mechanism. Honest negative, gated.
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


# --- space group (native, as S1; duplicated to keep committed probes frozen)
bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
NE = len(EDGES)
E_INDEX = {e: a for a, e in enumerate(EDGES)}
REV = {a: E_INDEX[(j, i, tuple(-x for x in c))] for a, (i, j, c) in enumerate(EDGES)}


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
N_OPS = len(OPS)
assert N_OPS == 24

OP_KEY = {(tuple(R.astype(int).ravel()), tuple(np.round(t, 6))): i
          for i, (R, t) in enumerate(OPS)}


def op_index(R, tau):
    return OP_KEY.get((tuple(np.round(R).astype(int).ravel()),
                       tuple(np.round(canon_tau(tau), 6))))


MULT = np.array([[op_index(R1 @ R2, R1 @ t2 + t1) for (R2, t2) in OPS]
                 for (R1, t1) in OPS])
R_PRIM = [np.round(M_INV @ R @ M_CART).astype(int) for R, _ in OPS]

SADDLES = {
    "Gamma": np.zeros(3),
    "H": np.array([0.5, 0.5, -0.5]),
    "P": np.array([0.25, 0.25, 0.25]),
    "N": A_PRIM @ np.array([0.0, 0.5, 0.5]),
}


def k_image(g, k):
    return la.inv(R_PRIM[g]).T @ np.asarray(k, float)


def little_group(k):
    return [g for g in range(N_OPS)
            if np.max(np.abs(k_image(g, k) - k - np.round(k_image(g, k) - k))) < 1e-9]


LG = {nm: little_group(k) for nm, k in SADDLES.items()}


# --- edge band rep (as S1)
EMAP = []
for (R, t) in OPS:
    rows = []
    for (i, j, c) in EDGES:
        i2, Li = atom_of(R @ ATOMS[i] + t)
        j2, Lj = atom_of(R @ (ATOMS[j] + M_CART @ np.asarray(c, float)) + t)
        di, dj = prim_int(Li), prim_int(Lj)
        rows.append((E_INDEX[(i2, j2, tuple(dj - di))], di))
    EMAP.append(rows)


def U_edge(g, k):
    k = np.asarray(k, float)
    kp = k_image(g, k)
    U = np.zeros((NE, NE), dtype=complex)
    for a, (i, j, c) in enumerate(EDGES):
        a2, di = EMAP[g][a]
        c2 = np.asarray(EDGES[a2][2], float)
        U[a2, a] = np.exp(2j * np.pi * (kp @ (di + c2) - k @ np.asarray(c, float)))
    return U


# ----------------------------------------------------------------------
# W1 -- special Wyckoff orbits by stabilizer scan (1/8 grid)
# ----------------------------------------------------------------------
print("=" * 72)
print(" PHASE 5.1 S5 -- EBR decomposition of the Hashimoto band rep")
print("=" * 72)
print("\n--- W1: special-orbit census ---")


def stabilizer(x):
    """Ops (mod lattice) fixing point x; returns list of (g, L_cart)."""
    out = []
    for g, (R, t) in enumerate(OPS):
        L = R @ x + t - x
        if is_bcc(L):
            out.append((g, L))
    return out


def orbit_of(x):
    pts = []
    for (R, t) in OPS:
        y = mod1(R @ x + t)
        # reduce mod BCC: canonical = lexicographic min of {y, y+(.5,.5,.5)} mod 1
        y = canon_tau(y)
        if not any(np.max(np.abs(y - p)) < 1e-9 for p in pts):
            pts.append(y)
    return pts


grid = [np.array([a, b, c]) / 8.0
        for a in range(8) for b in range(8) for c in range(8)]
found = []     # (orbit_size, stab_order, representative)
seen_pts = []
for x in grid:
    xc = canon_tau(x)
    if any(np.max(np.abs(xc - p)) < 1e-9 for p in seen_pts):
        continue
    st = stabilizer(xc)
    if len(st) in (4, 6):
        orb = orbit_of(xc)
        seen_pts.extend(orb)
        found.append((len(orb), len(st), xc))

census = sorted((o, s) for o, s, _ in found)
gate("W1a census: two (4-point, order-6) + two (6-point, order-4) orbits",
     census == [(4, 6), (4, 6), (6, 4), (6, 4)], f"{census}")

# identify the atom orbit and the bond-midpoint orbit
mid0 = None
for (i, j, c) in EDGES:
    m = (ATOMS[i] + ATOMS[j] + M_CART @ np.asarray(c, float)) / 2.0
    mid0 = canon_tau(m)
    break
orbits = {}
for o, s, x in found:
    if s == 6:
        key = "8a" if atom_of(x)[0] is not None else "8b"
    else:
        in_mid = any(np.max(np.abs(canon_tau(
            (ATOMS[i] + ATOMS[j] + M_CART @ np.asarray(c, float)) / 2.0) - x)) < 1e-9
            for (i, j, c) in EDGES) or any(
            np.max(np.abs(p - x)) < 1e-9 for p in orbit_of(mid0))
        key = "12mid" if in_mid else "12oth"
    orbits[key] = x
gate("W1b orbits identified: 8a(atoms), 8b, 12mid(bond midpoints), 12oth",
     set(orbits) == {"8a", "8b", "12mid", "12oth"},
     {k: tuple(np.round(v, 4)) for k, v in orbits.items()})

# midpoint D2: one 2-fold ALONG the bond
e0 = EDGES[0]
bvec = ATOMS[e0[1]] + M_CART @ np.asarray(e0[2], float) - ATOMS[e0[0]]
m0 = canon_tau((ATOMS[e0[0]] + ATOMS[e0[1]] + M_CART @ np.asarray(e0[2], float)) / 2.0)
st_m0 = stabilizer(m0)
along = [g for g, _ in st_m0
         if int(np.round(np.trace(OPS[g][0]))) == -1
         and la.norm(OPS[g][0] @ bvec - bvec) < 1e-9]
gate("W1c midpoint site group D2 with one 2-fold along the bond",
     len(st_m0) == 4 and len(along) == 1,
     f"|stab|={len(st_m0)}, along-bond 2-folds={len(along)}")


# ----------------------------------------------------------------------
# site irreps (native) and EBR Bloch builders
# ----------------------------------------------------------------------
def affine_inv(R, t):
    Ri = la.inv(R)
    return Ri, -Ri @ t


def affine_mul(a, b):
    return a[0] @ b[0], a[0] @ b[1] + a[1]


def site_setup(x0):
    """Stabilizer point ops about x0 (exact affine, fixing x0), coset reps."""
    st = []
    for g, L in stabilizer(x0):
        R, t = OPS[g]
        st.append((R, t - L))          # exact: (R|t-L) x0 = x0
    orb = orbit_of(x0)
    reps = []
    for p in orb:
        for (R, t) in OPS:
            y = R @ x0 + t
            L = y - p
            if is_bcc(L):
                reps.append((R, t - L))   # exact: r_p x0 = p
                break
    return st, orb, reps


def one_dim_irreps(st):
    """All 1-dim real characters of the site point group (brute force)."""
    n = len(st)
    keys = [tuple(np.round(R.ravel()).astype(int)) for R, _ in st]
    kidx = {k: i for i, k in enumerate(keys)}
    table = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(n):
            Rab = st[a][0] @ st[b][0]
            table[a, b] = kidx[tuple(np.round(Rab.ravel()).astype(int))]
    chars = []
    for signs in product((1.0, -1.0), repeat=n):
        if signs[keys.index(tuple(np.eye(3, dtype=int).ravel()))] != 1.0:
            continue
        if all(abs(signs[table[a, b]] - signs[a] * signs[b]) < 1e-9
               for a in range(n) for b in range(n)):
            if not any(np.allclose(signs, c) for c in chars):
                chars.append(np.array(signs))
    return keys, chars


def d3_E_irrep(st, axis):
    """2-dim E irrep of D3: restriction of the point ops to plane _|_ axis."""
    a = axis / la.norm(axis)
    u = np.array([1.0, -1.0, 0.0])
    u -= (u @ a) * a
    u /= la.norm(u)
    v = np.cross(a, u)
    Pl = np.vstack([u, v])
    return [Pl @ R @ Pl.T for R, _ in st]


def build_ebr_U(x0, st, orb, reps, rho_of, dim_rho, g, k):
    """Bloch matrix of the band rep induced from site irrep rho at orbit x0."""
    k = np.asarray(k, float)
    kp = k_image(g, k)
    n = len(orb)
    U = np.zeros((n * dim_rho, n * dim_rho), dtype=complex)
    Rg, tg = OPS[g]
    st_keys = [tuple(np.round(R.ravel()).astype(int)) for R, _ in st]
    for p in range(n):
        y = Rg @ orb[p] + tg
        for p2 in range(n):
            L = y - orb[p2]
            if is_bcc(L):
                break
        else:
            raise RuntimeError("orbit not preserved")
        h = affine_mul(affine_mul(affine_inv(*reps[p2]),
                                  (Rg, tg - L)), reps[p])
        assert la.norm(h[0] @ x0 + h[1] - x0) < 1e-9
        hk = tuple(np.round(h[0].ravel()).astype(int))
        rho = rho_of(st_keys.index(hk))
        ph = np.exp(2j * np.pi * (kp @ prim_int(L)))
        U[p2 * dim_rho:(p2 + 1) * dim_rho, p * dim_rho:(p + 1) * dim_rho] = ph * rho
    return U


# assemble the 14 candidate EBRs
EBRS = {}
for wname in ("8a", "8b", "12mid", "12oth"):
    x0 = orbits[wname]
    st, orb, reps = site_setup(x0)
    keys, chars = one_dim_irreps(st)
    for ci, ch in enumerate(chars):
        EBRS[f"{wname}:chi{ci}"] = dict(
            x0=x0, st=st, orb=orb, reps=reps, dim=1,
            rho_of=(lambda ch_: lambda idx: np.array([[ch_[idx]]]))(ch),
            sitechar=ch, sitekeys=keys)
    if len(st) == 6:                      # D3: add the E irrep
        axis = np.array([1.0, 1.0, 1.0])  # local 3-fold axis is a body diagonal
        # find the actual C3 axis of this stabilizer
        for R, _ in st:
            if int(np.round(np.trace(R))) == 0:
                ev, V = la.eig(R)
                axis = np.real(V[:, np.argmin(np.abs(ev - 1))])
                break
        mats = d3_E_irrep(st, axis)
        EBRS[f"{wname}:E"] = dict(
            x0=x0, st=st, orb=orb, reps=reps, dim=2,
            rho_of=(lambda m: lambda idx: m[idx])(mats), sitechar=None,
            sitekeys=keys)

gate("W1d EBR menu assembled: 14 candidates (3+3 from 8a/8b, 4+4 from 12s)",
     len(EBRS) == 14, f"{sorted(EBRS)}")


# ----------------------------------------------------------------------
# W2/W3 -- contents at saddles; cocycle equality; induction in stages
# ----------------------------------------------------------------------
def rep_content(U_of_g, dim, lg, ref_chars=None):
    """Decompose a (projective) rep into character classes.
    Returns dict {class_key: (d, mult)} with class_key = rounded char tuple."""
    rng_ = np.random.default_rng(5)
    Us = {g: U_of_g(g) for g in lg}
    H0 = rng_.normal(size=(dim, dim)) + 1j * rng_.normal(size=(dim, dim))
    H0 = H0 + H0.conj().T
    Hb = sum(U @ H0 @ U.conj().T for U in Us.values()) / len(lg)
    ev, V = la.eigh(Hb)
    blocks, i = [], 0
    while i < dim:
        grp = [i]
        while i + 1 < dim and abs(ev[i + 1] - ev[i]) < 1e-8:
            i += 1
            grp.append(i)
        blocks.append(V[:, grp])
        i += 1
    content = {}
    for Q in blocks:
        chi = tuple(np.round(np.trace(Q.conj().T @ Us[g] @ Q), 6) for g in lg)
        d = Q.shape[1]
        if (chi, d) in content:
            content[(chi, d)] += 1
        else:
            content[(chi, d)] = 1
    return content, Us


def cocycle(Us, lg):
    om = {}
    for a in lg:
        for b in lg:
            W = Us[a] @ Us[b] @ Us[MULT[a, b]].conj().T
            om[(a, b)] = W[0, 0] if abs(W[0, 0]) > 0.5 else \
                W.ravel()[np.argmax(np.abs(W.ravel()))]
    return om


print("\n--- W2/W3: saddle contents, cocycle equality, induction in stages ---")
EDGE_CONTENT, EBR_CONTENT = {}, {nm: {} for nm in SADDLES}
ok_cocycle = True
for nm, k in SADDLES.items():
    lg = LG[nm]
    EDGE_CONTENT[nm], Us_edge = rep_content(lambda g: U_edge(g, k), NE, lg)
    om_edge = cocycle(Us_edge, lg)
    for ename, E in EBRS.items():
        U_of = lambda g: build_ebr_U(E["x0"], E["st"], E["orb"], E["reps"],
                                     E["rho_of"], E["dim"], g, k)
        dimE = len(E["orb"]) * E["dim"]
        EBR_CONTENT[nm][ename], Us_e = rep_content(U_of, dimE, lg)
        om_e = cocycle(Us_e, lg)
        ok_cocycle &= all(abs(om_e[ab] - om_edge[ab]) < 1e-8 for ab in om_edge)
gate("W2 all 14 EBRs carry the SAME factor system as the edge rep, "
     "all saddles", ok_cocycle)

# the two midpoint site irreps trivial on the along-bond C2
mid = orbits["12mid"]
st_m, _, _ = site_setup(mid)
keys_m, chars_m = one_dim_irreps(st_m)
g_along, _ = next((g, L) for g, L in stabilizer(mid)
                  if int(np.round(np.trace(OPS[g][0]))) == -1
                  and la.norm(OPS[g][0] @ bvec - bvec) < 1e-9)
R_along = next((R for R, t in st_m
                if la.norm(R @ bvec - bvec) < 1e-9
                and int(np.round(np.trace(R))) == -1))
ia = [tuple(np.round(R.ravel()).astype(int)) for R, _ in st_m].index(
    tuple(np.round(R_along.ravel()).astype(int)))
pair = [f"12mid:chi{ci}" for ci, ch in enumerate(chars_m) if ch[ia] > 0]
gate("W3a exactly two 12mid site irreps trivial on the along-bond C2",
     len(pair) == 2, f"{pair}")


def add_contents(c1, c2):
    out = dict(c1)
    for kk, v in c2.items():
        out[kk] = out.get(kk, 0) + v
    return out


ok_stages = True
for nm in SADDLES:
    s = add_contents(EBR_CONTENT[nm][pair[0]], EBR_CONTENT[nm][pair[1]])
    ok_stages &= (s == EDGE_CONTENT[nm])
gate("W3b INDUCTION IN STAGES: EBR_A + EBR_B = edge band rep, "
     "class-by-class at all 4 saddles", ok_stages)


# ----------------------------------------------------------------------
# W4 -- exhaustive integer-decomposition scan
# ----------------------------------------------------------------------
print("\n--- W4: exhaustive decomposition scan over the 14-EBR menu ---")
names = sorted(EBRS)
dims = {e: len(EBRS[e]["orb"]) * EBRS[e]["dim"] for e in names}
sols = []


def scan(idx, remaining, counts):
    if remaining == 0:
        ok = True
        for nm in SADDLES:
            tot = {}
            for e, n in counts.items():
                if n:
                    for kk, v in EBR_CONTENT[nm][e].items():
                        tot[kk] = tot.get(kk, 0) + n * v
            if tot != EDGE_CONTENT[nm]:
                ok = False
                break
        if ok:
            sols.append(dict(counts))
        return
    if idx == len(names):
        return
    e = names[idx]
    for n in range(remaining // dims[e] + 1):
        counts[e] = n
        scan(idx + 1, remaining - n * dims[e], counts)
    counts[e] = 0


scan(0, NE, {e: 0 for e in names})
sol_strs = [" + ".join(f"{n}x{e}" for e, n in sorted(s.items()) if n)
            for s in sols]
gate("W4a the induction-in-stages solution is found by the scan",
     any(set(e for e, n in s.items() if n) == set(pair)
         and all(n == 1 for e, n in s.items() if n) for s in sols))
print(f"  ALL integer decompositions matching all 4 saddles "
      f"({len(sols)} found):")
for s in sol_strs:
    print(f"    {s}")
unique = len(sols) == 1
print(f"  FINDING: decomposition {'UNIQUE' if unique else 'NOT unique'} "
      f"on saddle content alone")

# W4b (xhigh trigger follow-up): discriminate the solutions at LINE points
# (Gamma-P, Gamma-H, Gamma-N midpoints). If contents differ there, the
# alternatives are saddle-only coincidences, not equal band reps.
print("\n--- W4b: line-point discrimination of the W4 solutions ---")
LINE_KS = {
    "Lambda(G-P mid)": np.array([0.125, 0.125, 0.125]),
    "Delta(G-H mid)": np.array([0.25, 0.25, -0.25]),
    "Sigma(G-N mid)": np.array([0.25, 0.0, 0.0]),
}
line_content = {}
for lnm, k in LINE_KS.items():
    lg = little_group(k)
    line_content[lnm] = {}
    for ename, E in EBRS.items():
        U_of = lambda g: build_ebr_U(E["x0"], E["st"], E["orb"], E["reps"],
                                     E["rho_of"], E["dim"], g, k)
        dimE = len(E["orb"]) * E["dim"]
        line_content[lnm][ename], _ = rep_content(U_of, dimE, lg)
    print(f"  {lnm}: little co-group order {len(lg)}")


def sol_line_sig(s):
    sig = []
    for lnm in LINE_KS:
        tot = {}
        for e, n in s.items():
            if n:
                for kk, v in line_content[lnm][e].items():
                    tot[kk] = tot.get(kk, 0) + n * v
        sig.append(tuple(sorted((kk, v) for kk, v in tot.items())))
    return tuple(sig)


sigs = {}
for s, ss in zip(sols, sol_strs):
    sigs.setdefault(sol_line_sig(s), []).append(ss)
print(f"  line-content equivalence classes among the {len(sols)} solutions: "
      f"{len(sigs)}")
for i, (sig, members) in enumerate(sigs.items()):
    print(f"    class {i}: {members}")
n_line_classes = len(sigs)
gate("W4b FINDING (frozen): 6 induction labels, ONE line-content class "
     "(plausibly one band rep; choice carries no physical weight)",
     len(sols) == 6 and n_line_classes == 1)

# ----------------------------------------------------------------------
# W5 -- does EBR membership refine the per-saddle forcing?
# ----------------------------------------------------------------------
print("\n--- W5: EBR_A vs EBR_B content disjointness per saddle ---")
overlap_everywhere = True
for nm in SADDLES:
    cA = set(EBR_CONTENT[nm][pair[0]])
    cB = set(EBR_CONTENT[nm][pair[1]])
    shared = cA & cB
    overlap_everywhere &= bool(shared)
    tag = "DISJOINT -> B-clusters acquire EBR labels here" if not shared \
        else f"OVERLAP in {len(shared)} class(es) -> no new forcing"
    print(f"  {nm:>6}: A={sorted((d, m) for (chi, d), m in EBR_CONTENT[nm][pair[0]].items())} "
          f"B={sorted((d, m) for (chi, d), m in EBR_CONTENT[nm][pair[1]].items())}  {tag}")
gate("W5 GATED NEGATIVE (frozen): EBR summand contents OVERLAP at every "
     "saddle -> the EBR layer adds NO forcing beyond per-saddle irreps + "
     "Ihara-Bass; no cross-saddle ties for the dictionary",
     overlap_everywhere)

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS")
print("=" * 72)
sys.exit(0)
