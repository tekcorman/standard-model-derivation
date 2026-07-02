#!/usr/bin/env python3
"""
THEOREM (substrate selection: srs, from MDL-dominance + the Yukawa mass channel)
================================================================================

Combines the two mechanisms that select the substrate, each machine-verified:
  (I)  DOMINANCE   — minimal description length = topological crystal (b1 = d).
  (II) MASS CHANNEL — the Yukawa/chirality walk exists iff the quotient is
       non-bipartite; observed massive fermions therefore require it.

Together they select srs and exclude every competitor in the trivalent family —
including the two that defeated weaker arguments: srs-z (survives nothing but
b1) and the achiral, structurally-CHEAPER nets ths/diamond (survive (I) as
topological crystals but fail (II): bipartite quotient -> massless).

PREMISES
--------
(P1) [MDL / A2-T] observer model weight ∝ 2^(-L); L = minimal generating description.
(P2) [Sunada, topological crystallography] a d-periodic net is the maximal abelian
     cover of a finite quotient Q; canonical (Q alone determines it, period-map cost 0)
     IFF b1(Q) = |E(Q)|-|V(Q)|+1 = d, else +(b1-d) period vectors. [published]
(P3) [derived upstream] d = 3 (Gleason); substrate k-regular (no-privilege).
(P4) [framework Yukawa, walker_dynamics + srs↔srs-z] the fermion mass / chirality
     coupling is the walk from the quotient Q to its BIPARTITE DOUBLE COVER (the χ̃
     operator). That cover is a nontrivial connected distinct graph IFF Q is
     NON-bipartite (a bipartite graph's double cover = two disjoint copies = trivial).
(P5) [observation] fermions are massive.

LEMMA A (dominance).  L(net) = K(Q) + L(period map); minimal ⟺ b1(Q)=d (topological
   crystal).  k-regular + one orbit + b1=3 ⟹ |V|=4/(k-2) ∈ Z+ ⟺ k∈{3,4,6}.
   At k=3 the minimal quotient is K_4 (complete, max symmetry); its cover is srs.
   srs-z (Q_3 quotient, b1=5) is over-determined ⟹ strictly subdominant.

LEMMA B (mass channel).  Yukawa channel exists ⟺ quotient non-bipartite.
   srs: quotient K_4 is non-bipartite (triangle) ⟹ channel ✓ ⟹ massive.
   diamond, ths: quotient bipartite ⟹ channel ✗ ⟹ MASSLESS.

THEOREM.  Under (P1)-(P5), the substrate is srs:
   - srs is the k=3, b1=3, NON-bipartite-quotient topological crystal (cover of K_4).
   - srs-z excluded by (I): b1=5, over-determined cover (subdominant; it is the
     bipartite partner where the boson/Higgs sector lives — see note).
   - ths, diamond excluded by (II): bipartite quotient ⟹ massless, contradicting (P5).
   K_4 is the unique 4-vertex cubic graph and is non-bipartite ⟹ srs is unique.

   COROLLARY (mass selects chirality): the excluded massless nets (ths, diamond) are
   achiral; srs (non-bipartite K_4) is chiral. So requiring mass picks the chiral net —
   one mechanism (the χ̃ walk), not a separate CP postulate. (The spectral CP-penalty
   was shown to NOT derive; this replaces it.)

STATUS: THEOREM-GRADE-CONDITIONAL on (P1 MDL + P2 Sunada[published] + P3 derived +
P4 framework-Yukawa).  Residual: "achiral ⟺ bipartite-quotient" holds 3/3 (computed,
not proven general); the selection rests on (P5)+(P4), not on that equivalence.

------------------------------------------------------------------------------
Machine verification (b1 from primitive cells; quotient bipartiteness; ths built from
genuine #141 ops and validated against its catalogued coordination sequence).
"""
import json, os, re
import numpy as np
from collections import deque, Counter
import spglib
from rcsr_net_assessment import get_space_group_ops, orbit_of, reconstruct_bonds
from arc_transitivity_ground_truth import build_net, get_symmetry_and_orbit, nearest_neighbor_bonds

SNAP = json.load(open(os.path.join(os.path.dirname(__file__), '..', '..',
        'simulator', 'menus', 'data', 'rcsr_candidates_snapshot.json')))['entries']


def centering(sg): return {'I': 2, 'F': 4, 'R': 3, 'C': 2, 'A': 2, 'B': 2, 'P': 1}.get(sg[0], 1)


def b1_primitive(name):
    e = SNAP[name]; vo = e['vertex_orbits'][0]; k = vo['coord']
    V = int(re.match(r'(\d+)', vo['wyckoff_label']).group(1)) // centering(e['sg_name'])
    E = k * V // 2
    return k, V, E, E - V + 1


def _ops_db(num):
    for h in range(1, 531):
        t = spglib.get_spacegroup_type(h)
        if (t.number if hasattr(t, 'number') else t['number']) == num:
            yield h


def net_bonds(name):
    """Return (positions, bonds) for the primitive/conventional cell. ths built from
    genuine #141 ops at the origin that reproduces its catalogued coordination sequence."""
    if name == 'dia':
        lat, seed, _ = build_net('dia'); r, t, pos, _, _ = get_symmetry_and_orbit(lat, seed)
        arcs, _, _ = nearest_neighbor_bonds(lat, pos)
        bonds = [(i, j, s) for i, j, s in arcs if i <= j]
        return pos, bonds, 'F'
    if name == 'ths':
        e = SNAP['ths']; seed = np.array(e['vertex_orbits'][0]['cartesian'])
        cat = e['coordination_sequence'][:5]
        for h in _ops_db(141):
            s = spglib.get_symmetry_from_database(h)
            R = s.rotations if hasattr(s, 'rotations') else s['rotations']
            T = s.translations if hasattr(s, 'translations') else s['translations']
            for sh in [(0,0,0),(0,0,.5),(.5,.5,.5),(0,.25,.125),(0,-.25,.125),(.125,.125,.125)]:
                shv = np.array(sh)
                pos = orbit_of((seed+shv) % 1.0, R, T)
                mids = np.vstack([orbit_of((np.array(eo['cartesian'])+shv) % 1.0, R, T)
                                  for eo in e['edge_orbits']])
                bonds = [b for b in reconstruct_bonds(pos, mids, 1e-3, 3) if b]
                if not bonds: continue
                deg = Counter()
                for i, j, sc in bonds: deg[i] += 1; deg[j] += 1
                if sorted(set(deg.values())) != [3]: continue
                if _coord_seq(pos, bonds) == cat:
                    return pos, bonds, 'I'
        raise RuntimeError("ths build failed to validate against catalogued CS")
    e = SNAP[name]; r, t, _, _ = get_space_group_ops(e['sg_name'])
    pos = orbit_of(np.array(e['vertex_orbits'][0]['cartesian']), r, t)
    mids = np.vstack([orbit_of(np.array(eo['cartesian']), r, t) for eo in e['edge_orbits']])
    bonds = [b for b in reconstruct_bonds(pos, mids, 1e-3, 3) if b]
    return pos, bonds, e['sg_name'][0]


def _supercell_adj(pos, bonds, N):
    from itertools import product
    arcs = []
    for i, j, s in bonds:
        arcs.append((i, j, tuple(s))); arcs.append((j, i, tuple(-x for x in s)))
    cells = list(product(range(N), repeat=3)); adj = {}
    for (i, j, s) in arcs:
        for c in cells:
            nc = tuple(c[t]+s[t] for t in range(3))
            if all(0 <= q < N for q in nc):
                adj.setdefault((i, c), set()).add((j, nc))
    return adj


def _coord_seq(pos, bonds, depth=5, N=11):
    adj = _supercell_adj(pos, bonds, N)
    for u in list(adj): adj.setdefault(u, set())
    # symmetrize
    for u in list(adj):
        for v in adj[u]: adj.setdefault(v, set()).add(u)
    cen = N // 2; src = (0, (cen, cen, cen)); dist = {src: 0}; dq = deque([src]); sh = [0]*(depth+1)
    while dq:
        u = dq.popleft()
        if dist[u] < depth:
            for v in adj.get(u, ()):
                if v not in dist: dist[v] = dist[u]+1; sh[dist[v]] += 1; dq.append(v)
    return sh[1:]


def quotient_bipartite(name):
    pos, bonds, cent = net_bonds(name)
    n = len(pos)
    if cent == 'I':
        bc = np.array([.5, .5, .5]); cmap = {}; nxt = 0
        for i in range(n):
            if i in cmap: continue
            cmap[i] = nxt
            for j in range(n):
                if j != i and j not in cmap and np.linalg.norm(((pos[j]-pos[i]-bc) % 1.0+.5) % 1.0-.5) < 1e-4:
                    cmap[j] = nxt
            nxt += 1
    else:
        cmap = {i: i for i in range(n)}; nxt = n
    A = np.zeros((nxt, nxt), int)
    for i, j, s in bonds:
        a, b = cmap[i], cmap[j]
        if a == b: A[a, a] += 1
        else: A[a, b] += 1; A[b, a] += 1
    # bipartite?
    col = [-1]*nxt; bip = True
    for st in range(nxt):
        if col[st] != -1: continue
        col[st] = 0; q = [st]
        while q:
            u = q.pop()
            if A[u, u] > 0: bip = False
            for v in range(nxt):
                if A[u, v] > 0:
                    if col[v] == -1: col[v] = 1-col[u]; q.append(v)
                    elif col[v] == col[u]: bip = False
    tri = any(A[i, j] > 0 and A[j, k] > 0 and A[i, k] > 0
              for i in range(nxt) for j in range(nxt) for k in range(nxt) if i < j < k)
    return nxt, bip, tri


def main():
    fails = []
    print("LEMMA A — dominance (b1 = d = 3 ⟺ canonical topological crystal):")
    print(f"  {'net':<7}{'k':>3}{'|V|':>4}{'|E|':>4}{'b1':>4}  status")
    for n, exp in [('srs', 3), ('srs-z', 5), ('dia', 3), ('pcu', 3)]:
        k, V, E, b1 = b1_primitive(n)
        if b1 != exp: fails.append(f"b1({n})={b1}≠{exp}")
        print(f"  {n:<7}{k:>3}{V:>4}{E:>4}{b1:>4}  {'topological crystal' if b1==3 else 'over-determined (subdominant)'}")
    iso = [k for k in range(3, 13) if 4 % (k-2) == 0]
    print(f"  single-orbit topological crystals: k∈{iso} (expect [3,4,6])")
    if iso != [3, 4, 6]: fails.append(f"iso={iso}")

    print("\nLEMMA B — mass channel (Yukawa exists ⟺ quotient NON-bipartite):")
    print(f"  {'net':<7}{'|Vq|':>5}{'bipartite':>10}{'triangle':>9}  channel / mass")
    expect_bip = {'srs': False, 'srs-z': True, 'dia': True, 'ths': True}
    for n, eb in expect_bip.items():
        nq, bip, tri = quotient_bipartite(n)
        if bip != eb: fails.append(f"bipartite({n})={bip}≠{eb}")
        chan = "NON-bip → Yukawa channel ✓ → MASSIVE" if not bip else "bipartite → no channel → MASSLESS"
        print(f"  {n:<7}{nq:>5}{str(bip):>10}{str(tri):>9}  {chan}")

    print("\n" + ("ALL CHECKS PASS — substrate = srs (b1=3 + non-bipartite K_4 quotient); "
                  "srs-z subdominant; ths, diamond massless-excluded."
                  if not fails else "FAILURES: " + "; ".join(fails)))
    return not fails


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
