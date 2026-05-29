"""
Cayley-graph construction + structural invariants for the Coxeter GROUP W(M).

For a finite Coxeter system M = (S, m), Cay(W(M), S) is the concrete finite
graph on |W(M)| vertices, |S|-regular, with edges {(w, w·s) : w ∈ W, s ∈ S}.
We build it via the (faithful, for finite W) geometric reflection
representation: s_i acts on ℝⁿ (n = |S|) as the reflection in the hyperplane
with normal α_i, B(α_i, α_j) = −cos(π/m_ij), B(α_i, α_i) = 1. BFS from the
identity matrix generates exactly |W| group elements; the Cayley graph + its
structural invariants (|V|, degree, girth, diameter, adjacency spectrum,
closed-walk counts) follow.

⚠️ THIS IS NOT THE FRAMEWORK'S SUBSTRATE. Read this before using `cayley`
output as if it were a substrate catalog:

  - Cay(W(M), S) is an abstract group-theoretic graph — a *sanity / structural-
    invariants tool* for the Coxeter-quotient menu (`menus/coxeter.py`). It is
    NEITHER the toggle-stream model (that's the Coxeter QUOTIENT of F_inv(|E|),
    scored in `gating/mdl.py`) NOR the framework's spatial substrate.

  - The framework's spatial substrate is a CRYSTAL NET. The realization
    candidate set is the V+E-transitive 3-connected chiral 3D 3-regular RCSR
    nets — srs, srs-z, srs-c4, srs-c8, srs-c27, lou, lov, okw, hcb-c4 — plus
    the centrosymmetric (ths, dia) and d>3 alternatives for the non-chiral
    channels. That layer ALREADY EXISTS and is mature:
        proofs/foundations/rcsr_net_assessment.py        (parse RCSR DB, build nets)
        proofs/foundations/rcsr_per_substrate_fingerprint.py  (per-net fingerprint)
        proofs/foundations/rcsr_candidate_sweep.py       (χ̃/bipartiteness sweep)
        proofs/foundations/dl_comparison.py              (DL minimization → srs)
        proofs/foundations/srs_vs_srs_z_dl_audit.py + qtz_vs_srs_dl_comparison.py + lov_dl_audit.py
        proofs/foundations/substrate_lattice_waterfilling_batch.py  (A2-T waterfilling)
        an internal working note    (channel map)
        an internal working note        (Row-4 k* / R-9 program)
    `simulator.menus.crystal_nets` is the rebuild's thin INDEX of /
    bridge to that layer (the framework's substrate IS srs, the MDL-minimum
    chiral 3D 3-regular net; R-9 = the open srs-vs-srs-z 2.56-bit gap, being
    closed). DO NOT reimplement the crystal-net layer here.

  - srs is NOT Cay(W(H_3), S). srs is the (10,3)-a / Laves graph — an infinite
    3-periodic net in space group I4_132 with girth 10 — whereas Cay(W(H_3), S)
    is a finite 120-vertex graph with girth 4. They are different objects.

So: use `cayley.structural_catalog(M)` to ask "what does the Coxeter GROUP
W(M) look like as a graph" (legit), NOT "what is the substrate for slice M"
(use `menus.crystal_nets` / the RCSR apparatus for that, where it's defined —
which is only for the crystal-net realization candidates, not arbitrary |E|
Coxeter quotients).

For non-finite (affine / hyperbolic / free) Coxeter slices the GROUP is
infinite; a finite truncation (ball of radius R from the identity) is built,
with the truncation radius reported alongside the (truncation-dependent) counts.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .menus.coxeter import CoxeterSystem


# Size caps (keep the builder fast & memory-bounded):
_BFS_MAX_ELEMENTS = 60_000      # don't BFS finite W larger than this
_SPECTRUM_MAX_V = 2_000         # don't dense-diagonalize larger adjacency
_WALK_MAX_V = 50_000            # don't trace(A^L) for larger graphs
_ROUND_DECIMALS = 9             # matrix-entry rounding for element dedup
_TRUNCATION_RADIUS = 8          # ball radius for infinite-group truncation


@dataclass
class CayleyGraph:
    """A built Cayley graph + provenance.

    Attributes:
        coxeter        : the CoxeterSystem
        n_vertices     : |V| in the built graph (= |W| for finite & fully built)
        degree         : |S| (regular degree)
        adjacency      : (n_vertices × n_vertices) numpy adjacency, or None if
                         the graph exceeded the build cap
        truncated      : True iff this is a radius-R ball of an infinite group
        truncation_radius : R if truncated, else None
        capped         : True iff the build was skipped because |W| > _BFS_MAX_ELEMENTS
    """
    coxeter: CoxeterSystem
    n_vertices: Optional[int]
    degree: int
    adjacency: Optional[np.ndarray]
    truncated: bool = False
    truncation_radius: Optional[int] = None
    capped: bool = False


# ---------------------------------------------------------------------------
# Reflection-representation generators
# ---------------------------------------------------------------------------

def _bilinear_form(coxeter: CoxeterSystem) -> np.ndarray:
    """n×n Gram matrix B with B[i][i]=1, B[i][j]=−cos(π/m_ij) (m_ij default 2)."""
    n = coxeter.generators
    B = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            m = coxeter.m_pairs.get((i + 1, j + 1), 2)
            val = 0.0 if m == float('inf') else -math.cos(math.pi / m)
            B[i, j] = B[j, i] = val
    return B


def _generator_matrices(coxeter: CoxeterSystem) -> list[np.ndarray]:
    """The |S| reflection matrices S_i on ℝⁿ: S_i(α_j) = α_j − 2 B[j][i] α_i."""
    n = coxeter.generators
    B = _bilinear_form(coxeter)
    gens = []
    for i in range(n):
        S = np.eye(n)
        # row i becomes [−2 B[i][0], …, −2 B[i][i]+? ]: S(α_j)=α_j−2B[j][i]α_i
        # ⇒ column j of S is e_j − 2 B[j][i] e_i ⇒ S[i][j] = δ_ij − 2 B[i][j].
        for j in range(n):
            S[i, j] = (1.0 if j == i else 0.0) - 2.0 * B[i, j]
        gens.append(S)
    return gens


def _key(mat: np.ndarray) -> tuple:
    return tuple(np.round(mat, _ROUND_DECIMALS).ravel())


# ---------------------------------------------------------------------------
# BFS construction
# ---------------------------------------------------------------------------

def build_cayley_graph(coxeter: CoxeterSystem) -> CayleyGraph:
    """Build Cay(W(M), S).

    Finite W: full graph (unless |W| > cap, in which case `capped=True` and
    only |V|=|W| / degree are filled). Infinite W (affine/hyperbolic/free):
    a radius-_TRUNCATION_RADIUS ball from the identity, `truncated=True`.
    """
    n = coxeter.generators
    degree = n
    if n == 0:
        return CayleyGraph(coxeter, 0, 0, np.zeros((0, 0)))

    finite = (coxeter.growth_class == 'finite')
    if finite and coxeter.order is not None and coxeter.order > _BFS_MAX_ELEMENTS:
        return CayleyGraph(coxeter, coxeter.order, degree, None, capped=True)

    gens = _generator_matrices(coxeter)
    identity = np.eye(n)
    index = {_key(identity): 0}
    elements = [identity]
    frontier = [0]
    edges = set()  # undirected (u, v) with u < v
    radius = 0
    max_radius = math.inf if finite else _TRUNCATION_RADIUS

    while frontier and radius < max_radius:
        radius += 1
        new_frontier = []
        for u in frontier:
            g = elements[u]
            for S in gens:
                h = S @ g
                k = _key(h)
                v = index.get(k)
                if v is None:
                    v = len(elements)
                    index[k] = v
                    elements.append(h)
                    new_frontier.append(v)
                a, b = (u, v) if u < v else (v, u)
                if a != b:
                    edges.add((a, b))
        frontier = new_frontier
        if finite and len(elements) > _BFS_MAX_ELEMENTS:
            # safety: representation not closing as expected — bail to capped
            return CayleyGraph(coxeter, coxeter.order, degree, None, capped=True)

    nV = len(elements)
    A = np.zeros((nV, nV))
    for a, b in edges:
        A[a, b] = A[b, a] = 1.0
    return CayleyGraph(
        coxeter, nV, degree, A,
        truncated=(not finite),
        truncation_radius=(None if finite else _TRUNCATION_RADIUS))


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

def _analytic_girth(coxeter: CoxeterSystem) -> Optional[int]:
    """2·min over distinct pairs of m_ij (default 2). ∞ if |S|≤1 or no finite m."""
    n = coxeter.generators
    if n <= 1:
        return None  # K_2 / single edge / point — acyclic
    ms = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            m = coxeter.m_pairs.get((i, j), 2)
            if m != float('inf'):
                ms.append(m)
    if not ms:
        return None  # all pairs m=∞ (free baseline) — acyclic (tree)
    return 2 * min(ms)


def _bfs_girth(A: np.ndarray) -> Optional[int]:
    """Girth via BFS from each vertex (simple-graph shortest cycle)."""
    nV = A.shape[0]
    if nV == 0:
        return None
    adj = [np.nonzero(A[i])[0].tolist() for i in range(nV)]
    best = math.inf
    for src in range(nV):
        dist = {src: 0}
        parent = {src: -1}
        queue = [src]
        qi = 0
        while qi < len(queue):
            u = queue[qi]; qi += 1
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
                elif parent[u] != w:
                    cyc = dist[u] + dist[w] + 1
                    if cyc < best:
                        best = cyc
        if best <= 4:  # can't do better in a simple graph with degree ≥ 2
            break
    return None if best == math.inf else int(best)


def _bfs_diameter(A: np.ndarray) -> Optional[int]:
    """Graph diameter (max over all pairs of shortest-path distance)."""
    nV = A.shape[0]
    if nV == 0:
        return None
    adj = [np.nonzero(A[i])[0].tolist() for i in range(nV)]
    diam = 0
    for src in range(nV):
        dist = {src: 0}
        queue = [src]; qi = 0
        while qi < len(queue):
            u = queue[qi]; qi += 1
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        if dist:
            diam = max(diam, max(dist.values()))
    return diam


def _closed_walk_counts(A: np.ndarray, max_len: int = 8) -> dict:
    """{L: trace(A^L)} for L = 2..max_len. (L=1 is 0 — simple graph, no loops.)"""
    nV = A.shape[0]
    out = {}
    P = np.eye(nV)
    for L in range(1, max_len + 1):
        P = P @ A
        out[L] = int(round(float(np.trace(P))))
    return out


def structural_catalog(coxeter: CoxeterSystem) -> dict:
    """Coxeter-Cayley-graph structural catalog for a slice.

    Always: name, growth_class, |E| (= |S| = degree), |W| (= |V| if finite),
    girth (analytic), the realization-gap note. When the graph is built and
    small enough: BFS girth (cross-checks analytic), diameter, adjacency
    spectrum (sorted, with Perron eigenvalue = |S|), closed-walk counts.
    """
    g = build_cayley_graph(coxeter)
    cat = {
        'name': coxeter.name,
        'growth_class': coxeter.growth_class,
        'generators_E': coxeter.generators,
        'cayley_degree': coxeter.generators,
        'group_order_W': coxeter.order,
        'girth_analytic': _analytic_girth(coxeter),
        'n_vertices_built': g.n_vertices,
        'truncated': g.truncated,
        'truncation_radius': g.truncation_radius,
        'build_capped': g.capped,
        'realization_note': (
            'Cayley graph of W(M); NOT a crystal-net spatial realization — '
            'see simulator.cayley module docstring (realization gap).'),
    }
    A = g.adjacency
    if A is not None and A.shape[0] > 0:
        nV = A.shape[0]
        if nV <= _SPECTRUM_MAX_V:
            evals = np.sort(np.linalg.eigvalsh(A))
            cat['adjacency_spectrum'] = [float(round(x, 9)) for x in evals]
            cat['adjacency_perron'] = float(round(evals[-1], 9))
        else:
            cat['adjacency_perron'] = float(coxeter.generators)  # regular ⇒ Perron = degree
        cat['girth_bfs'] = _bfs_girth(A)
        if nV <= _WALK_MAX_V:
            cat['diameter'] = _bfs_diameter(A)
            cat['closed_walk_counts'] = _closed_walk_counts(A, 8)
    else:
        cat['adjacency_perron'] = float(coxeter.generators)  # |S|-regular
    return cat


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo():
    from .menus import coxeter as cm
    print("=" * 92)
    print(" simulator.cayley — Coxeter-Cayley-graph structural catalogs")
    print("=" * 92)
    for cs in [cm.enumerate_finite()[0],  # I_2(2) = V_4
               next(c for c in cm.enumerate_finite() if 'S_3' in c.name),
               next(c for c in cm.enumerate_finite() if c.name.startswith('A_3')),
               next(c for c in cm.enumerate_finite() if 'H_3' in c.name),
               next(c for c in cm.enumerate_finite() if 'F_4' in c.name),
               next(c for c in cm.enumerate_affine() if 'Ã_2' in c.name),
               next(c for c in cm.enumerate_finite() if c.name.startswith('E_8'))]:
        cat = structural_catalog(cs)
        spec = cat.get('adjacency_spectrum')
        spec_s = (f"[{spec[0]:.3f} … {spec[-1]:.3f}], {len(spec)} evals" if spec
                  else f"Perron={cat['adjacency_perron']:.0f} (not diagonalized)")
        print(f"\n {cs.name}")
        print(f"   |V|={cat['n_vertices_built']}  deg={cat['cayley_degree']}  "
              f"|W|={cat['group_order_W']}  girth(analytic)={cat['girth_analytic']}"
              + (f"  girth(BFS)={cat['girth_bfs']}" if 'girth_bfs' in cat else "")
              + (f"  diam={cat['diameter']}" if 'diameter' in cat else "")
              + (f"  capped={cat['build_capped']}" if cat['build_capped'] else "")
              + (f"  truncated@R={cat['truncation_radius']}" if cat['truncated'] else ""))
        print(f"   spectrum: {spec_s}")
        if 'closed_walk_counts' in cat:
            print(f"   closed walks: {cat['closed_walk_counts']}")
    print("\n" + "=" * 92)


if __name__ == '__main__':
    _demo()
