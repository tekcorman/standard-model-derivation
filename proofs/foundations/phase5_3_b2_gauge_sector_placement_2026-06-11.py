#!/usr/bin/env python3
"""Phase 5.3/B2 -- the Kitaev gauge sector placed in framework terms.

Spec: docs/scoping/phase5_3_kitaev_spec_2026-06-11.md (B2, high).

Three placements, each gated:

  C1  THE GAUGE SECTOR IS THE CYCLE SPACE, BLOCH-RESOLVED: the Z2 flux
      degrees of freedom of the Kitaev construction live on the graph's
      cycle space; the Hashimoto operator's +1 eigenvalue multiplicity at
      every k equals dim ker d(k) of the oriented Bloch incidence map
      (cycle space at k), and by the banked antiperiod the -1
      multiplicity equals dim ker d(k+Delta). Verified at the 4 saddles
      + random k.
  C2  THE DICTIONARY'S 18 "TRIVIAL" MODES = THE GAUGE SECTOR: the
      48-table's trivial census per saddle decomposes exactly as
      [dim ker d(k)] + [dim ker d(k+Delta)] -- Gamma (3+2), H (2+3),
      P (2+2), N (2+2) = 18. The non-matter walker content IS the Z2
      gauge sector of the Kitaev construction.
  C3  FLUX RINGS = THE GIRTH CONTENT: closed NB 10-walks from cell-0
      edges = 120 (the banked zeta N_10); they organize into exactly
      120/(10*2) = 6 translation classes of 10-rings (the shortest
      gauge-invariant Wilson loops).
  C3' (FINDING; the naive "rings generate the flux space" conjecture is
      REFUTED and the true structure gated): fiber-wise the ring chains
      span ker d(k) at every tested k EXCEPT Gamma, where all six
      oriented ring chains VANISH IDENTICALLY (rank 0 vs ker dim 3);
      over GF(2) on tori the ring span misses the local cycle space by a
      STABLE deficit of exactly 2 (L = 2 and L = 3). The flux sector
      carries invariants beyond the minimal Wilson loops, localized at
      the Gamma anomaly (ker d(0) = 3, the a = 3 Bass-branch crossing).
      Named for B3.
  C4  MAJORANA BOOKKEEPING: the local-CAR theorem gives 6 Majoranas per
      node (2 per edge-end); the Kitaev pairing consumes 1 per edge-end
      (3 bond Majoranas) leaving 3 MATTER Majoranas per node = 12 per
      primitive cell = dim of the directed-edge (Hashimoto) space.
      Reported as a NAMED, UNPROMOTED observation together with the D3
      site-group action (permutes the 3 edges, hence the 3 matter
      Majoranas) -- exploration for B3, no claim gated beyond the counts.
"""
import os
import sys
from itertools import product

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, A_PRIM  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
NE = len(EDGES)
E_INDEX = {e: a for a, e in enumerate(EDGES)}
REV = {a: E_INDEX[(j, i, tuple(-x for x in c))] for a, (i, j, c) in enumerate(EDGES)}
seen = set()
UBONDS = []
for (i, j, c) in EDGES:
    if (j, i, tuple(-x for x in c)) in seen:
        continue
    seen.add((i, j, tuple(c)))
    UBONDS.append((i, j, tuple(c)))
NU = len(UBONDS)


def B_of(k):
    B = np.zeros((NE, NE), dtype=complex)
    for a, (i, j, c) in enumerate(EDGES):
        for b, (i2, j2, c2) in enumerate(EDGES):
            if i2 == j and b != REV[a]:
                B[b, a] = np.exp(2j * np.pi * np.dot(k, np.asarray(c2, float)))
    return B


def d_of(k):
    """Oriented Bloch incidence: vertex x undirected-bond; source -1,
    target +phase. ker = the cycle space at k."""
    D = np.zeros((4, NU), dtype=complex)
    for b, (i, j, c) in enumerate(UBONDS):
        D[i, b] += -1.0
        D[j, b] += np.exp(2j * np.pi * np.dot(k, np.asarray(c, float)))
    return D


def mults(k):
    ev = la.eigvals(B_of(k))
    return (int(np.sum(np.abs(ev - 1.0) < 1e-7)),
            int(np.sum(np.abs(ev + 1.0) < 1e-7)))


def ker_dim(k):
    s = la.svd(d_of(k), compute_uv=False)
    return NU - int(np.sum(s > 1e-9))


DELTA = np.array([0.5, 0.5, -0.5])
SADDLES = {
    "Gamma": np.zeros(3),
    "H": np.array([0.5, 0.5, -0.5]),
    "P": np.array([0.25, 0.25, 0.25]),
    "N": A_PRIM @ np.array([0.0, 0.5, 0.5]),
}

print("=" * 72)
print(" PHASE 5.3/B2 -- the gauge sector placed in framework terms")
print("=" * 72)

# C1: mult_{+1}(B(k)) = dim ker d(k); mult_{-1}(B(k)) = dim ker d(k+Delta)
rng = np.random.default_rng(17)
test_ks = list(SADDLES.values()) + [rng.uniform(-0.5, 0.5, 3) for _ in range(4)]
ok1 = True
rows = []
for k in test_ks:
    mp, mm = mults(k)
    kp, km = ker_dim(k), ker_dim(k + DELTA)
    ok1 &= (mp == kp and mm == km)
    rows.append((np.round(k, 3), mp, kp, mm, km))
gate("C1 Bloch cycle space = Hashimoto +-1 content: mult_+1(B(k)) = "
     "dim ker d(k) AND mult_-1(B(k)) = dim ker d(k+Delta), saddles + 4 "
     "random k", ok1)
for k, mp, kp, mm, km in rows[:4]:
    print(f"      k={k!s:>24}: mult_+1={mp} ker={kp} | mult_-1={mm} ker(+D)={km}")

# C2: the 18 trivial modes = the gauge sector, saddle by saddle
census = {nm: (ker_dim(k), ker_dim(k + DELTA)) for nm, k in SADDLES.items()}
total = sum(a + b for a, b in census.values())
gate("C2 the dictionary's 18 'trivial' modes = the Z2 gauge sector: "
     "Gamma(3+2) + H(2+3) + P(2+2) + N(2+2) = 18",
     census == {"Gamma": (3, 2), "H": (2, 3), "P": (2, 2), "N": (2, 2)}
     and total == 18, f"{census}, total={total}")

# C3: 10-ring census vs the banked N_10 = 120
FOLLOW = {a: [b for b, (i2, j2, c2) in enumerate(EDGES)
              if i2 == EDGES[a][1] and b != REV[a]] for a in range(NE)}


def closed_nb_walks(length):
    walks = []

    def step(path, a, d):
        ca = EDGES[a][2]
        d2 = (d[0] + ca[0], d[1] + ca[1], d[2] + ca[2])
        if len(path) == length:
            # closure step: the first edge must FOLLOW the last, landing
            # at cumulative offset zero
            if path[0][0] in FOLLOW[a] and d2 == (0, 0, 0):
                walks.append(list(path))
            return
        for b in FOLLOW[a]:
            step(path + [(b, d2)], b, d2)

    for a0 in range(NE):
        step([(a0, (0, 0, 0))], a0, (0, 0, 0))
    return walks


walks10 = closed_nb_walks(10)
n10 = len(walks10)


def cycle_class(walk):
    inst = set()
    for a, d in walk:
        i, j, c = EDGES[a]
        ra = REV[a]
        d2 = (d[0] + c[0], d[1] + c[1], d[2] + c[2])
        inst.add(min((a, d), (ra, d2)))
    base = min(d for _, d in inst)
    return frozenset((a, (d[0] - base[0], d[1] - base[1], d[2] - base[2]))
                     for a, d in inst)


classes = {cycle_class(w) for w in walks10}
gate("C3 N_10 = 120 closed NB 10-walks reproduced; exactly 6 translation "
     "classes of 10-rings (= the shortest Wilson loops), 10 edges each",
     n10 == 120 and len(classes) == 6
     and all(len(cl) == 10 for cl in classes),
     f"N_10={n10}, classes={len(classes)}")

# C3' (FINDING, first-run conjecture "rings generate" REFUTED as stated;
# the true structure is gated): fiber-wise over C the rings generate the
# cycle space at every k EXCEPT Gamma, where ALL oriented ring chains
# vanish identically; over GF(2) on tori the ring span misses the local
# cycle space by a STABLE deficit of exactly 2 (L=2 and L=3) -- the
# minimal Wilson loops under-generate by a 2-dim sector tied to the
# Gamma anomaly (where ker d(0) = 3, the a=3 Bass-branch crossing).
reps = []
for cl in classes:
    for w in walks10:
        if cycle_class(w) == cl:
            reps.append(w)
            break


def ring_chain(walk, k):
    v = np.zeros(NU, dtype=complex)
    for a, d in walk:
        i, j, c = EDGES[a]
        if (i, j, tuple(c)) in seen:
            v[UBONDS.index((i, j, tuple(c)))] += np.exp(
                2j * np.pi * np.dot(k, np.asarray(d, float)))
        else:
            i2, j2, c2 = EDGES[REV[a]]
            d2 = (d[0] + c[0], d[1] + c[1], d[2] + c[2])
            v[UBONDS.index((i2, j2, tuple(c2)))] -= np.exp(
                2j * np.pi * np.dot(k, np.asarray(d2, float)))
    return v


ok_fiber = True
for nm, k in [("H", DELTA), ("P", np.array([0.25, 0.25, 0.25])),
              ("N", SADDLES["N"]), ("rand", rng.uniform(-0.5, 0.5, 3))]:
    V = np.array([ring_chain(w, k) for w in reps])
    ok_fiber &= (max(la.norm(d_of(k) @ v) for v in V) < 1e-9)
    ok_fiber &= (int(np.sum(la.svd(V, compute_uv=False) > 1e-9)) == ker_dim(k))
V0 = np.array([ring_chain(w, np.zeros(3)) for w in reps])
gamma_vanish = float(np.max(np.abs(V0)))
gate("C3'a fiber-wise: oriented ring chains lie in ker d(k) and SPAN it "
     "at H, P, N, random k (rank = ker dim = 2)", ok_fiber)
gate("C3'b GAMMA ANOMALY: at k = 0 every oriented 10-ring chain vanishes "
     "IDENTICALLY (each ring net-cancels on every primitive bond class) "
     "-- rings see none of ker d(0) = 3",
     gamma_vanish < 1e-12, f"max |chain(Gamma)|={gamma_vanish:.1e}")


def gf2_ring_deficit(L):
    cells_ = list(product(range(L), repeat=3))
    cidx_ = {c: i for i, c in enumerate(cells_)}
    NEdge_ = NU * len(cells_)

    def slot(a, d):
        i, j, c = EDGES[a]
        if (i, j, tuple(c)) in seen:
            b, base = UBONDS.index((i, j, tuple(c))), d
        else:
            i2, j2, c2 = EDGES[REV[a]]
            b = UBONDS.index((i2, j2, tuple(c2)))
            base = (d[0] + c[0], d[1] + c[1], d[2] + c[2])
        return cidx_[tuple(x % L for x in base)] * NU + b

    vecs_ = []
    for cl in classes:
        for shift in cells_:
            v = np.zeros(NEdge_, dtype=np.int8)
            for a, d in cl:
                v[slot(a, (d[0] + shift[0], d[1] + shift[1],
                           d[2] + shift[2]))] ^= 1
            vecs_.append(v)
    Mw = (np.array(vecs_) % 2).copy()
    rank = 0
    for col in range(Mw.shape[1]):
        piv = next((r for r in range(rank, Mw.shape[0]) if Mw[r, col]), None)
        if piv is None:
            continue
        Mw[[rank, piv]] = Mw[[piv, rank]]
        for r in range(Mw.shape[0]):
            if r != rank and Mw[r, col]:
                Mw[r] ^= Mw[rank]
        rank += 1
    return (NEdge_ - 4 * len(cells_)) - rank   # E - V minus ring rank


defs = {L: gf2_ring_deficit(L) for L in (2, 3)}
gate("C3'c GF(2) torus deficit STABLE at exactly 2 (L = 2 and L = 3): "
     "the minimal Wilson loops under-generate the flux space by a fixed "
     "2-dim sector (NAMED for B3: extra gauge invariants beyond 10-ring "
     "fluxes)", defs == {2: 2, 3: 2}, f"deficits={defs}")

# C4: Majorana bookkeeping (named, unpromoted)
n_node_majoranas = 6      # local-CAR theorem: Cl(6) per node
n_bond_per_node = 3       # one per incident edge-end consumed by u_e
n_matter = n_node_majoranas - n_bond_per_node
gate("C4 bookkeeping: 6 Majoranas/node (local-CAR Cl(6)) = 3 bond + 3 "
     "matter; matter total per cell = 12 = dim of the directed-edge "
     "(Hashimoto) space", n_matter * 4 == NE, f"{n_matter}*4 = {n_matter*4}")
print("  NAMED, UNPROMOTED (B3/B2 exploration, no gate): (i) the 3 matter")
print("  Majoranas per node carry the D3 site-group edge permutation --")
print("  the same 3-fold structure the framework reads as generations;")
print("  (ii) 12 matter Majoranas per cell match the Hashimoto space dim;")
print("  whether the gauged matter dynamics REPRODUCES B(k) is a separate")
print("  bet (K3 discipline: no silent target swap).")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- gauge sector placed")
print("=" * 72)
sys.exit(0)
