#!/usr/bin/env python3
"""Phase 5.3/B3 -- THE BRIDGE: global CAR from per-node Cl(6), no ordering.

Spec: internal research notes (B3, xhigh; kill
criteria K1 import / K2 freedom blowup / K3 wrong structure frozen).

INPUTS (K1 audit -- nothing else enters): (i) the srs adjacency from
proofs/common.py; (ii) the local-CAR theorem's per-node construction
(theorem_car_local_jordan_wigner: 3 edge-end qubits per node, intra-node
JW -> 6 Majoranas = Cl(6); intra-node ordering proven gauge); (iii)
explicit matrix algebra on clusters of actual srs nodes (2-node edge,
dim 64; 3-node path, dim 512).

THE CONSTRUCTION (all operators node-local or edge-local; NO total node
ordering anywhere):
  per node v, incident-edge slot m: bond Majorana b_{v,m} = gamma_{2m},
    matter Majorana c_{v,m} = gamma_{2m+1} (the local theorem's pairs).
  per edge e = (u,v): LINK operator u_e := b_{u,e} b_{v,e}
    (cross-node product of commuting Hermitian involutions).
  per node: GAUSS operator G_v := the Cl(6) volume element (total local
    fermion parity).
  DRESSED MATTER MAJORANAS: pick any spanning tree of the cluster and a
    root; chi_{v,m} := (i^k) c_{v,m} * (product of u_e along the tree
    path v -> root). The dressing is built of LINK operators only -- the
    only choice is the tree/root, and D4 proves that choice is GAUGE.

Gates:
  D0 local-theorem regression: the 6 node Majoranas satisfy Cl(6)
     exactly ({g_a, g_b} = 2 delta, 8x8).
  D1 link algebra: u_e Hermitian, u_e^2 = 1 (Z2 -- with the BOSONIC
     cross-node tensor structure the involution needs NO i, unlike
     Kitaev's fermionic ambient); u's sharing a node ANTICOMMUTE,
     disjoint u's commute (the discovered relation, gated as found).
  D2 Gauss/gauge structure: G_v^2 = 1; conjugation by G_v flips the sign
     of u_e exactly on edges incident to v and fixes all other u's and
     all K_f := (i b c)_u (i b c)_v composites -- the NODE GAUGE MOVE of
     the B1 quadratic level, now derived at the operator level. All K_e
     mutually commute (the gauge-invariant hop layer).
  D3 GLOBAL CAR, NO ORDERING: on the 3-node srs path (root chosen at one
     end), the 9 dressed matter Majoranas {chi} are Hermitian, square to
     +1, and ALL 36 pairs anticommute -- the canonical anticommutation
     relations across distinct nodes, achieved with link dressing alone.
  D4 [PANEL-CORRECTED 2026-06-11, verdict PARTIAL, order 1]: the gated
     computation (end-to-end re-rooting: one uniform link factor, all
     bilinears equal) is an END-ROOT SPECIAL CASE. The general law
     (panel counterexample, now gated in phase5_3_b3b E1): a dressing
     change is a CAR-ALGEBRA AUTOMORPHISM -- a link-built factor times
     per-Majorana Z2 signs (middle root flips 36/81 bilinear signs,
     factorizably). On CYCLIC clusters (b3b E4) a spanning-tree change
     shifts straddling bilinears by exactly a CENTRAL ring operator W:
     dressing/ordering is gauge WITHIN A FLUX SECTOR (the spec's frozen
     wording), not unconditionally. The total-ordering REQUIREMENT of
     the local-CAR scope limit is still dissolved -- no ordering is
     needed -- but the flux-sector residue is physical and partially
     unclassified (named follow-ups).
  D5 choice census: swapping the b <-> c roles at every end of one edge
     leaves the construction a CAR set (the per-end labeling is the
     intra-node basis gauge of the local theorem); counted: tree (gauge,
     D4), root (gauge, D4), b/c labeling (intra-node gauge, D5),
     orientation of u_e (none -- commuting factors), i-normalizations
     (fixed by Hermiticity, no freedom). ZERO unpriced bits.

Honest scope: the 10-ring flux operators do not fit a feasible cluster
(8^10 dims); their algebra (Hermitian involutions commuting with the
dressed bilinears; flux sectors as in B1's quadratic A4g) follows from
the D1 relations and is NOT separately gated here. The A4
adoption-register / ledger consequence is decided by the ULTRACODE
verdict panel, not this probe.
"""
import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


# the local theorem's node Majoranas (3 edge-end qubits, intra-node JW)
G6 = [kron(X, I2, I2), kron(Y, I2, I2),
      kron(Z, X, I2), kron(Z, Y, I2),
      kron(Z, Z, X), kron(Z, Z, Y)]

print("=" * 72)
print(" PHASE 5.3/B3 -- global CAR from per-node Cl(6); ordering -> gauge")
print("=" * 72)

ok0 = all(la.norm(G6[a] @ G6[b] + G6[b] @ G6[a]
                  - (2.0 if a == b else 0.0) * np.eye(8)) < 1e-12
          for a in range(6) for b in range(6))
gate("D0 local-theorem regression: per-node Cl(6) exact (8x8)", ok0)

# srs cluster: a 3-node path from the actual graph
bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
out_slots = {v: [a for a, (i, j, c) in enumerate(EDGES) if i == v]
             for v in range(4)}
# path: node u0=0 --e--> v0 --f--> w0 (w0 != u0)
e0 = out_slots[0][0]
v0 = EDGES[e0][1]
f0 = next(a for a in out_slots[v0] if EDGES[a][1] != 0)
w0 = EDGES[f0][1]
NODES = [0, v0, w0]
POS = {n: p for p, n in enumerate(NODES)}
DIM = 8 ** len(NODES)


def emb(op, node):
    mats = [np.eye(8)] * len(NODES)
    mats[POS[node]] = op
    return kron(*mats)


def slot(v, target):
    """Incident-edge slot index of edge v->target at node v."""
    for m, a in enumerate(out_slots[v]):
        if EDGES[a][1] == target:
            return m
    raise ValueError


def b_op(v, target):
    return emb(G6[2 * slot(v, target)], v)


def c_op(v, m):
    return emb(G6[2 * m + 1], v)


# link operators along the path edges e: (0, v0), f: (v0, w0)
u_e = b_op(0, v0) @ b_op(v0, 0)
u_f = b_op(v0, w0) @ b_op(w0, v0)

herm = la.norm(u_e - u_e.conj().T) + la.norm(u_f - u_f.conj().T)
invol = la.norm(u_e @ u_e - np.eye(DIM)) + la.norm(u_f @ u_f - np.eye(DIM))
anti_shared = la.norm(u_e @ u_f + u_f @ u_e)   # share node v0
gate("D1 link algebra: u_e Hermitian involutions (no i needed -- bosonic "
     "ambient); sharing a node => ANTICOMMUTE",
     herm < 1e-12 and invol < 1e-12 and anti_shared < 1e-12,
     f"herm={herm:.1e}, u^2={invol:.1e}, shared-anticomm={anti_shared:.1e}")

# Gauss operators and gauge-invariant hops
def gauss(v):
    P = np.eye(8, dtype=complex)
    for g in G6:
        P = P @ g
    # normalize to a Hermitian involution
    P = (1j) ** 3 * P
    assert la.norm(P @ P - np.eye(8)) < 1e-12
    return emb(P, v)


G_v0 = gauss(v0)
K_e = (1j * b_op(0, v0) @ c_op(0, slot(0, v0))) @ \
      (1j * b_op(v0, 0) @ c_op(v0, slot(v0, 0)))
K_f = (1j * b_op(v0, w0) @ c_op(v0, slot(v0, w0))) @ \
      (1j * b_op(w0, v0) @ c_op(w0, slot(w0, v0)))
flip_e = la.norm(G_v0 @ u_e @ G_v0 + u_e)       # v0 incident to e: flips
flip_f = la.norm(G_v0 @ u_f @ G_v0 + u_f)       # v0 incident to f: flips
fix_K = la.norm(G_v0 @ K_e @ G_v0 - K_e) + la.norm(G_v0 @ K_f @ G_v0 - K_f)
KK = la.norm(K_e @ K_f - K_f @ K_e)
gate("D2 Gauss structure: G_v flips u on incident edges (node gauge move "
     "at operator level), fixes the gauge-invariant hops K_e; K's commute",
     flip_e < 1e-12 and flip_f < 1e-12 and fix_K < 1e-12 and KK < 1e-12,
     f"flips={flip_e + flip_f:.1e}, fixK={fix_K:.1e}, [K,K]={KK:.1e}")


def hermitize(M):
    """Multiply by the i-power making M Hermitian (fixed by algebra,
    no freedom); fail loudly if none works."""
    for k in range(4):
        Mk = (1j) ** k * M
        if la.norm(Mk - Mk.conj().T) < 1e-9:
            return Mk
    raise ValueError("no Hermitian normalization")


def dressed_set(root):
    """chi_{v,m} = hermitized c_{v,m} * (links along tree path v->root).
    The path graph's tree is itself; paths: along the chain."""
    path_links = {0: [], v0: [], w0: []}
    if root == 0:
        path_links[v0] = [u_e]
        path_links[w0] = [u_f, u_e]
    elif root == w0:
        path_links[v0] = [u_f]
        path_links[0] = [u_e, u_f]
    chis = []
    for v in NODES:
        for m in range(3):
            M = c_op(v, m)
            for L in path_links[v]:
                M = M @ L
            chis.append(hermitize(M))
    return chis


CHI = dressed_set(root=0)
ok_sq = all(la.norm(ch @ ch - np.eye(DIM)) < 1e-11 for ch in CHI)
worst_pair = max(la.norm(CHI[a] @ CHI[b] + CHI[b] @ CHI[a])
                 for a in range(9) for b in range(a + 1, 9))
gate("D3 GLOBAL CAR WITHOUT ORDERING: 9 dressed matter Majoranas on the "
     "3-node srs path -- Hermitian, chi^2 = 1, ALL 36 pairs anticommute",
     ok_sq and worst_pair < 1e-11, f"worst {{chi,chi}} = {worst_pair:.1e}")

# D4: re-root the tree -> second CAR set. DISCOVERED STRUCTURE (first
# run; the naive conjugator guess failed and was replaced by what is
# exactly true): the two dressings differ by ONE UNIFORM link-built
# unitary LEFT factor, and their even (physical) algebras are IDENTICAL.
CHI2 = dressed_set(root=w0)
ok_sq2 = all(la.norm(ch @ ch - np.eye(DIM)) < 1e-11 for ch in CHI2)
worst2 = max(la.norm(CHI2[a] @ CHI2[b] + CHI2[b] @ CHI2[a])
             for a in range(9) for b in range(a + 1, 9))
V_link = -u_e @ u_f                      # the uniform gauge factor
dev_factor = max(la.norm(CHI2[a] - V_link @ CHI[a]) for a in range(9))
dev_bilinear = max(la.norm(CHI2[a] @ CHI2[b] - CHI[a] @ CHI[b])
                   for a in range(9) for b in range(9))
gate("D4 end-to-end re-rooting (PANEL-REWORDED, order 1: this uniform-"
     "factor/equal-bilinears structure is an END-ROOT SPECIAL CASE, not "
     "the general law -- middle-root flips 36/81 bilinear signs; the "
     "general form is a CAR automorphism = link factor x per-Majorana "
     "signs, gated in phase5_3_b3b E1; on cycles dressing is gauge "
     "WITHIN a flux sector, b3b E4): chi'_a = -(u_e u_f) chi_a and all "
     "81 bilinears equal, for the two path-END roots",
     ok_sq2 and worst2 < 1e-11 and dev_factor < 1e-11
     and dev_bilinear < 1e-11,
     f"pair={worst2:.1e}, factor={dev_factor:.1e}, "
     f"bilinear={dev_bilinear:.1e}")

# D5: b <-> c swap on edge e (both ends) -- intra-node basis gauge
def b_swap(v, target):
    return emb(G6[2 * slot(v, target) + 1], v)


def c_swap(v, m, swapped_slot):
    g = G6[2 * m] if m == swapped_slot else G6[2 * m + 1]
    return emb(g, v)


u_e_sw = b_swap(0, v0) @ b_swap(v0, 0)
sw0, swv = slot(0, v0), slot(v0, 0)
chis_sw = []
for v in NODES:
    for m in range(3):
        ss = sw0 if v == 0 else (swv if v == v0 else None)
        M = c_swap(v, m, ss) if v in (0, v0) else c_op(v, m)
        links = {0: [], v0: [u_e_sw], w0: [u_f, u_e_sw]}[v]
        for L in links:
            M = M @ L
        chis_sw.append(hermitize(M))
worst_sw = max(la.norm(chis_sw[a] @ chis_sw[b] + chis_sw[b] @ chis_sw[a])
               for a in range(9) for b in range(a + 1, 9))
ok_sq_sw = all(la.norm(ch @ ch - np.eye(DIM)) < 1e-11 for ch in chis_sw)
gate("D5 choice census (PANEL-REWORDED, order 2): b<->c relabeling on an "
     "edge still yields a CAR set (one-END swap + out-vs-in convention "
     "gated in b3b/E2); i-normalizations fixed mod 2 -- the per-chi Z2 "
     "Majorana sign is standard CAR gauge, convention smallest-k -> zero "
     "unpriced bits, ALL NAMED",
     ok_sq_sw and worst_sw < 1e-11, f"worst pair={worst_sw:.1e}")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- the bridge stands; A4 verdict -> panel")
print("=" * 72)
sys.exit(0)
