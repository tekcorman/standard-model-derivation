#!/usr/bin/env python3
"""
admitted_set_validated.py
=========================

Firm foundation for "which nets are ADMITTED" (= arc-transitive trivalent 3D
crystal nets), with every bond reconstruction VALIDATED against the authoritative
RCSR topological invariants (coordination sequence + girth from the vertex
symbol) before any arc-transitivity verdict is trusted.

Why: neither bond method is reliable alone — Euclidean-nearest fails for some
nets (mixed degree), and reconstruct_bonds (shortest-midpoint) mis-assigned
srs-z (gave girth 14 vs the catalogued 10). So we reconstruct by BOTH methods,
keep only the bond set whose computed (coordination sequence, girth) matches
RCSR, and compute arc-transitivity on that.

Admission filters (RCSR-catalogued, no reconstruction needed):
  F1  genuinely 3D     : coordination sequence must grow super-linearly
                         (hcb-c4 has CS [3,6,9,12,15,18] -> linear -> excluded).
  F2  edge-transitive  : #edge_orbits == 1 (necessary for arc-transitivity)
                         (lou/lov/okw have 2 -> excluded).
Then for survivors: reconstruct+validate bonds, compute arc-orbits, edge-
reversibility, local vertex action (full S_k = Sunada strong isotropy).

Cross-check: by Sunada (independently reproduced in arc_transitivity_ground_truth.py),
at most ONE net is strongly isotropic; any validated strongly-isotropic candidates
must therefore be isomorphic (= srs in different RCSR symmetry settings).
"""

import json, os, sys
import numpy as np
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    get_space_group_ops, orbit_of, reconstruct_bonds, build_directed_edges,
)
from arc_transitivity_ground_truth import (
    op_atom_map, orbit_count, edge_reversible, local_vertex_action,
)

SNAP = json.load(open(os.path.join(os.path.dirname(__file__), '..', '..',
        'simulator', 'menus', 'data', 'rcsr_candidates_snapshot.json')))['entries']
CANDS = ['srs', 'srs-z', 'srs-c4', 'srs-c8', 'srs-c27', 'lou', 'lov', 'okw', 'hcb-c4']


def girth_from_vertex_symbol(vs):
    """vertex symbol like '10(5).10(5).10(5)' -> girth 10."""
    if not vs:
        return None
    import re
    nums = [int(x) for x in re.findall(r'(\d+)\(', vs)]
    return min(nums) if nums else None


def is_3d(cs):
    """Super-linear coordination-sequence growth => genuinely 3D. Linear (constant
    second difference ~0, i.e. arithmetic) => lower-dimensional."""
    d1 = [cs[i+1] - cs[i] for i in range(len(cs)-1)]
    d2 = [d1[i+1] - d1[i] for i in range(len(d1)-1)]
    return any(abs(x) > 1 for x in d2)  # 3D nets have growing first differences


def distance_bonds(positions, lattice=None):
    """Undirected nearest-neighbour bonds (i<=j, integer shift)."""
    if lattice is None:
        lattice = np.eye(3)
    n = len(positions); P = np.array(positions)
    from itertools import product
    cells = list(product([-1, 0, 1], repeat=3))
    dd = []
    for i in range(n):
        for j in range(n):
            for c in cells:
                s = np.array(c)
                d = np.linalg.norm(lattice @ (P[j] + s - P[i]))
                if d > 1e-6:
                    dd.append((d, i, j, tuple(c)))
    dd.sort()
    d0 = dd[0][0]
    bonds = set()
    for d, i, j, s in dd:
        if abs(d - d0) < 1e-3:
            key = (i, j, s) if i <= j else (j, i, tuple(-x for x in s))
            bonds.add(key)
    return list(bonds)


def rcsr_bonds(positions, entry, rots, trans):
    mids = np.vstack([orbit_of(np.array(eo['cartesian']), rots, trans)
                      for eo in entry['edge_orbits']])
    b = reconstruct_bonds(positions, mids, tol=1e-3, max_shift=3)
    return [x for x in b if x is not None]


def supercell_graph(positions, bonds, N):
    arcs = build_directed_edges(list(bonds))
    from itertools import product
    cells = list(product(range(N), repeat=3))
    adj = {}
    for (i, j, s) in arcs:
        for c in cells:
            nc = tuple(c[t] + s[t] for t in range(3))
            if all(0 <= q < N for q in nc):
                adj.setdefault((i, c), set()).add((j, nc))
                adj.setdefault((j, nc), set()).add((i, c))
    return adj


def coord_seq(positions, bonds, depth=5, N=13):
    adj = supercell_graph(positions, bonds, N)
    cen = N // 2
    src = (0, (cen, cen, cen))
    shells = [0] * (depth + 1)
    dist = {src: 0}; dq = deque([src])
    while dq:
        u = dq.popleft()
        if dist[u] < depth:
            for w in adj.get(u, ()):
                if w not in dist:
                    dist[w] = dist[u] + 1
                    shells[dist[w]] += 1
                    dq.append(w)
    return shells[1:]


def girth(positions, bonds, N=7):
    adj = supercell_graph(positions, bonds, N)
    cen = N // 2
    best = float('inf')
    for i in range(len(positions)):
        src = (i, (cen, cen, cen))
        dist = {src: 0}; par = {src: None}; dq = deque([src])
        while dq:
            u = dq.popleft()
            for w in adj.get(u, ()):
                if w not in dist:
                    dist[w] = dist[u] + 1; par[w] = u; dq.append(w)
                elif par[u] != w:
                    best = min(best, dist[u] + dist[w] + 1)
    return best


def main():
    print("=" * 100)
    print("VALIDATED ADMITTED SET — bonds gated on RCSR coordination sequence + girth")
    print("=" * 100)
    admitted = []
    strong = []
    for name in CANDS:
        e = SNAP[name]
        cs_rcsr = e['coordination_sequence']
        g_rcsr = girth_from_vertex_symbol(e.get('vertex_symbol'))
        n_eo = len(e['edge_orbits'])
        # F1: 3D?
        if not is_3d(cs_rcsr[:6]):
            print(f"{name:<9s} EXCLUDED (F1 not-3D): CS={cs_rcsr[:6]} grows linearly")
            continue
        # F2: edge-transitive?
        if n_eo != 1:
            print(f"{name:<9s} EXCLUDED (F2 not edge-transitive): {n_eo} edge orbits, girth~{g_rcsr} -> cannot be arc-transitive")
            continue
        # reconstruct + validate
        rots, trans, _, _ = get_space_group_ops(e['sg_name'])
        pos = orbit_of(np.array(e['vertex_orbits'][0]['cartesian']), rots, trans)
        chosen = None
        for label, bonds in [('rcsr-midpoint', rcsr_bonds(pos, e, rots, trans)),
                             ('distance', distance_bonds(pos))]:
            deg = Counter(i for b in bonds for i in (b[0], b[1]))
            if not bonds or any(deg[v] != 3 for v in range(len(pos))):
                continue
            cs = coord_seq(pos, bonds, depth=5)
            g = girth(pos, bonds)
            ok_cs = (cs == cs_rcsr[:5])
            ok_g = (g_rcsr is None or g == g_rcsr)
            if ok_cs and ok_g:
                chosen = (label, bonds, cs, g)
                break
        if chosen is None:
            print(f"{name:<9s} UNVALIDATED: no bond set reproduces RCSR CS={cs_rcsr[:5]} / girth={g_rcsr}")
            continue
        label, bonds, cs, g = chosen
        arcs = build_directed_edges(list(bonds))
        maps = op_atom_map(rots, trans, pos)
        n_arc, _ = orbit_count(arcs, maps, directed=True)
        n_und, _ = orbit_count(arcs, maps, directed=False)
        rev = edge_reversible(arcs, maps)
        k, loc, full = local_vertex_action(arcs, maps)
        at = (n_arc == 1)
        si = (loc == full and n_und == 1)
        verdict = "ADMITTED" if at else "not arc-transitive"
        if at:
            admitted.append(name)
        if si:
            strong.append(name)
        print(f"{name:<9s} {e['sg_name']:<9s} bonds={label:<13s} girth={g} CS={cs}  "
              f"E-orb={n_und} arc-orb={n_arc} localS={loc}/{full}  "
              f"-> {verdict}{' + STRONGLY ISOTROPIC' if si else ''}")

    print("\n" + "=" * 100)
    print("RESULT")
    print("=" * 100)
    print(f"  ADMITTED (arc-transitive, trivalent, 3D): {admitted}")
    print(f"  strongly isotropic among them: {strong}")
    print(f"\n  By Sunada (reproduced), >=2 strongly-isotropic nets must be ISOMORPHIC")
    print(f"  (= srs in different RCSR symmetry settings). So up to isomorphism the")
    print(f"  admitted superposition basis is:  srs  +  the arc-transitive-but-NOT-")
    print(f"  strongly-isotropic distinct net(s) {[n for n in admitted if n not in strong]}.")


if __name__ == '__main__':
    main()
