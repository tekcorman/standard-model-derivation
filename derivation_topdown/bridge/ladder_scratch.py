"""
SCRATCH (walled): the bond-combination ladder on the K4 cell.
Reads only ../dirac_srs_mdl/srs.py for the native EDGES/voltages. Pure math, no physics.
Enumerate all 2^6 subsets of the 6 K4 edges, graded by size; for each:
  - vertex boundary (loose ends) -> closed (cycle) vs open
  - cycle-space dimension (b1 of the occupied subgraph)
  - non-backtracking (Hashimoto) recurrence of the occupied sub-network:
       shortest closed NB walk (girth), closed-walk counts N_m = Tr B^m, NB spectrum |h|.
Also the Z^3 cover: do the voltages of the occupied edges close to 0 (cell loop) or carry homology?
"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

EDGES = srs.EDGES                      # 6 edges: (tail, head, Z^3 voltage)
NE = len(EDGES)                        # 6
NV = srs.NV                            # 4
w = np.exp(2j*np.pi/3)

def boundary(subset):
    """Vertex boundary = vertices of ODD degree in the occupied subgraph (mod 2 d-image)."""
    deg = np.zeros(NV, int)
    for e in subset:
        i, j, _ = EDGES[e]; deg[i] += 1; deg[j] += 1
    return tuple(v for v in range(NV) if deg[v] % 2 == 1)

def occupied_b1(subset):
    """b1 of occupied subgraph = E - V_touched + C (components among touched vertices)."""
    if not subset: return 0, 0, 0
    verts = set()
    for e in subset:
        i, j, _ = EDGES[e]; verts.add(i); verts.add(j)
    # union-find for components
    parent = {v: v for v in verts}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in subset:
        i, j, _ = EDGES[e]; parent[find(i)] = find(j)
    comps = len({find(v) for v in verts})
    E = len(subset); V = len(verts)
    return E - V + comps, V, comps   # b1, V_touched, components

def cover_homology(subset):
    """Sum of Z^3 voltages over a chosen orientation; whether the GF2 cycle space has any
    cycle carrying nonzero net voltage (cell-loop vs covering-loop). We report the rank of the
    voltage map restricted to the occupied cycle space over the rationals (how many homology
    directions the occupied loops span)."""
    # Build occupied cycle space over GF2 is awkward with voltages; instead: build the
    # integer cycle matrix of the occupied subgraph and push voltages.
    if not subset: return 0
    verts = sorted({v for e in subset for v in EDGES[e][:2]})
    vidx = {v: a for a, v in enumerate(verts)}
    # incidence (oriented) over occupied edges
    Inc = np.zeros((len(verts), len(subset)), int)
    Volt = np.zeros((3, len(subset)), int)
    for c, e in enumerate(subset):
        i, j, v = EDGES[e]
        Inc[vidx[i], c] = -1; Inc[vidx[j], c] = +1
        Volt[:, c] = np.array(v)
    # cycle space = kernel of Inc over Q
    from numpy.linalg import svd
    u, s, vt = svd(Inc.astype(float))
    null = vt[np.sum(s > 1e-9):]        # rows span the (right) null space = cycles
    if null.shape[0] == 0: return 0
    # net voltage of each cycle
    netvolt = Volt.astype(float) @ null.T   # 3 x (#cycles)
    # rank of homology directions spanned
    return int(np.linalg.matrix_rank(netvolt, tol=1e-9))

def nb_recurrence(subset, k=(0.0, 0.0, 0.0), use_cover_phase=True):
    """Non-backtracking operator of the OCCUPIED sub-network (darts of occupied edges only).
    Returns (dim, girth_or_None, N_m list, sorted |h| of nonzero NB spectrum)."""
    darts = []
    for e in subset:
        i, j, v = EDGES[e]
        darts.append((i, j, np.array(v)))
        darts.append((j, i, -np.array(v)))
    n = len(darts)
    if n == 0:
        return 0, None, [], []
    B = np.zeros((n, n), complex)
    kk = np.asarray(k, float)
    for b, (tb, hb, vb) in enumerate(darts):
        for a, (ta, ha, va) in enumerate(darts):
            # NB: dart a then dart b; head(a)=tail(b) and not exact reverse
            if ha == tb and not (hb == ta and np.array_equal(vb, -va)):
                B[b, a] = np.exp(2j*np.pi*(kk @ vb)) if use_cover_phase else 1.0
    ev = np.linalg.eigvals(B)
    nz = sorted(abs(l) for l in ev if abs(l) > 1e-7)
    # closed NB walk counts (with cover phases on the cell, k=0 gives the cell-quotient counts)
    Nm = [int(round(np.trace(np.linalg.matrix_power(B, m)).real)) for m in range(1, 13)]
    girth = next((m for m in range(1, 13) if Nm[m-1] != 0), None)
    return n, girth, Nm, [round(x, 4) for x in nz]

# ---------------------------------------------------------------------------------------------
print("="*90)
print("K4 EDGES (tail, head, Z^3 voltage):")
for e, (i, j, v) in enumerate(EDGES):
    print(f"  e{e}: {i}-{j}  voltage {v}")
print("Spanning tree = {e0,e1,e2} (voltage 0); cotree {e3,e4,e5} carry e_1,e_2,e_3.")
print("="*90)

# group all 64 subsets by size; record boundary, b1, cover-homology rank, NB data
from collections import defaultdict
by_size = defaultdict(list)
for r in range(NE+1):
    for sub in itertools.combinations(range(NE), r):
        bdy = boundary(list(sub))
        b1, Vt, comps = occupied_b1(list(sub))
        homrank = cover_homology(list(sub))
        by_size[r].append((sub, bdy, b1, Vt, comps, homrank))

print("\nLADDER: closed (cycle) content per size rung")
print(f"{'size':>4} {'#subsets':>9} {'#closed(bdy=0)':>14} {'max b1':>7} {'cycle-subsets b1>=1':>20}")
for r in range(NE+1):
    rows = by_size[r]
    closed = [x for x in rows if len(x[1]) == 0]   # empty boundary
    withcycle = [x for x in rows if x[2] >= 1]
    maxb1 = max((x[2] for x in rows), default=0)
    print(f"{r:>4} {len(rows):>9} {len(closed):>14} {maxb1:>7} {len(withcycle):>20}")

print("\nTotal closed (empty-boundary, nonempty) subsets = cycle elements of the cell:")
allclosed = [x for r in range(1, NE+1) for x in by_size[r] if len(x[1]) == 0]
for sub, bdy, b1, Vt, comps, homrank in allclosed:
    edges_str = "+".join(f"e{e}" for e in sub)
    print(f"  size {len(sub)}: {{{edges_str}}}  b1={b1}  cover-homology-rank={homrank}")
print(f"  => {len(allclosed)} nonempty closed subsets total. (Cycle space b1(K4)=3 ; #nonzero GF2 cycles=2^3-1=7.)")
