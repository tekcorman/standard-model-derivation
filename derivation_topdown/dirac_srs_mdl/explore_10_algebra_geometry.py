"""
explore_10 — (A) EMERGENT ALGEBRA and (B) GEOMETRY of the srs net.  Pure math, walled off.

(A) Does a Clifford / recognizable algebra flow WITHOUT being imposed?
    - Clifford-like anticommutation among D=[[0,d],[d*,0]] and natural operators.
    - The 12-dim dart space as an A4-module = 1 + 1' + 1'' + 3 + 3 + 3.  By Schur the
      COMMUTANT of the A4-action is C + C + C + M_3(C)  (dim 1+1+1+9 = 12).  Verify
      numerically; report what the M_3 block (the three copies of the 3-irrep) organizes.

(B) GEOMETRY:
    - GIRTH by BFS/cycle search on a finite patch of the Z^3 cover.  Confirm girth = 10,
      count length-10 cycles through a vertex, report their A4-orbit structure.
    - CHIRALITY: is A(-k) ~ A(k) by a PERMUTATION (proper point-group element)?  Is there
      any IMPROPER (orientation-reversing) symmetry?  No improper symmetry => chiral.
"""
import numpy as np, itertools, srs
np.set_printoptions(precision=4, suppress=True)

# =====================================================================================
# A4 = even permutations of {0,1,2,3}; helpers for permutation reps on V/E/darts.
# =====================================================================================
def parity(p):
    seen = [False]*4; par = 0
    for i in range(4):
        if not seen[i]:
            j = i; c = 0
            while not seen[j]: seen[j] = True; j = p[j]; c += 1
            par += c-1
    return par % 2
S4 = list(itertools.permutations(range(4)))
A4 = [p for p in S4 if parity(p) == 0]
ODD = [p for p in S4 if parity(p) == 1]

DARTS = srs._darts()                      # 12 darts (tail, head, vec)
DUV = [(d[0], d[1]) for d in DARTS]       # underlying (tail,head) pairs
def pd(p):                                # permutation rep on the 12-dim dart space
    M = np.zeros((12, 12))
    for d, (i, j) in enumerate(DUV):
        t = (p[i], p[j])
        for f, (a, b) in enumerate(DUV):
            if (a, b) == t: M[f, d] = 1; break
    return M

print("="*86)
print("(A)  EMERGENT ALGEBRA")
print("="*86)

# -------------------------------------------------------------------------------------
# A.1  Clifford-like anticommutation of the Hodge-Dirac and natural grading operator.
# -------------------------------------------------------------------------------------
print("\n[A.1] Hodge-Dirac D=[[0,d],[d*,0]] and the natural Z2 grading G=diag(+I_C0,-I_C1).")
k = (0.2, 0.25, 0.3)                      # generic point
D = srs.hodge_dirac(k)
n0, n1 = srs.NV, len(srs.EDGES)
G = np.diag([1.0]*n0 + [-1.0]*n1)         # chirality / grading operator
antiDG = D @ G + G @ D
print(f"   D is Hermitian:                 {np.allclose(D, D.conj().T)}")
print(f"   G^2 = I:                        {np.allclose(G@G, np.eye(n0+n1))}")
print(f"   {{D, G}} = DG + GD = 0:           {np.allclose(antiDG, 0)}   (D is ODD for the grading)")
print(f"   D^2 is block-diagonal (even):   {np.allclose((D@D)[:n0, n0:], 0) and np.allclose((D@D)[n0:, :n0], 0)}")
print("   => (D, G) is a Clifford-like ODD/EVEN pair: D anticommutes with the grading G,")
print("      D^2 = Laplacian is even.  This is the SPECTRAL-TRIPLE (Clifford) grading, and it")
print("      flows from the bipartite C0(+)C1 structure WITHOUT being imposed.")

# A.1b  Genuine Clifford generators?  Look for Hermitian unitaries that pairwise anticommute.
# The two natural sign operators are G (above) and any 'reflection' from the point group.
# Test a concrete pair built from the structure: G and a dart-conjugation J (reversal).
print("\n[A.1b] Dart reversal J (a -> a-bar) is a natural involution on C1; build C0(+)C1 version.")
rev = np.zeros((n1, n1))                   # edge space is unoriented (6 edges) -> reversal = I there;
for e in range(n1): rev[e, e] = 1.0        # so on C1 reversal acts trivially (edges, not darts).
# Instead test the dart-space reversal (12-dim) which IS a nontrivial involution:
Jdart = np.zeros((12, 12))
for a, (i, j, v) in enumerate(DARTS):
    for b, (p, q, w) in enumerate(DARTS):
        if (p, q) == (j, i) and np.array_equal(w, -v): Jdart[b, a] = 1.0; break
print(f"   dart-reversal J on 12-dim dart space: J^2 = I:  {np.allclose(Jdart@Jdart, np.eye(12))}")
print(f"   J is a (real) permutation/orthogonal involution: {np.allclose(Jdart@Jdart.T, np.eye(12))}")
print("   => J is a genuine Z2 involution (orientation reversal of darts); pairs with the A4-action.")

# -------------------------------------------------------------------------------------
# A.2  The A4-module structure of the 12-dim dart space and ITS COMMUTANT.
# -------------------------------------------------------------------------------------
print("\n[A.2] A4-module decomposition of the 12-dim dart space (characters).")
def chi3(p):
    fx = sum(1 for i in range(4) if p[i] == i)
    return 3 if fx == 4 else (-1 if fx == 0 else 0)
chs = [np.trace(pd(p)).real for p in A4]
m1  = sum(chs)/12
m3  = sum(c*chi3(p) for c, p in zip(chs, A4))/12
m1p = (12 - m1 - 3*m3)/2                   # m(1') = m(1'') by reality
print(f"   multiplicities:  1:{round(m1)}   1':{round(m1p)}   1'':{round(m1p)}   3:{round(m3)}")
print(f"   => 12 = 1 + 1' + 1'' + 3+3+3   ( {round(m1)} + {round(m1p)} + {round(m1p)} + {round(m3)}x3 )")

# Commutant: all 12x12 matrices commuting with every pd(p), p in A4.
# Solve the linear system  pd(p) X = X pd(p)  for ALL p in A4 (a few generators do NOT
# suffice here — using only 2 elements under-determines it and overcounts the nullspace).
# vectorise (vec(AXB)=(B^T (x) A)vec(X)):  (I (x) pd(p) - pd(p)^T (x) I) vec(X) = 0.
rows = []
I12 = np.eye(12)
for g in A4:
    Pg = pd(g).astype(complex)
    rows.append(np.kron(I12, Pg) - np.kron(Pg.T, I12))
Mbig = np.vstack(rows)
u, s, vh = np.linalg.svd(Mbig)
null = vh[np.sum(s > 1e-9):].conj().T      # columns = basis of the commutant (as vec(X))
dim_comm = null.shape[1]
print(f"\n   dim of the COMMUTANT of the A4-action on the 12-dim dart space = {dim_comm}")
print(f"   Schur prediction for 1+1'+1''+3+3+3:  1^2 x (3 distinct singlets) + 3^2 (M_3 on the")
print(f"      multiplicity space of the triple-3)  = 1+1+1+9 = 12")
print(f"   match: {dim_comm == 12}")

# Confirm the M_3 block: restrict to the 9-dim isotypic component of the 3-irrep and show
# the commutant there is exactly M_3(C) (dim 9) = End of the multiplicity space C^3.
# Build the projector onto the 3-isotypic component via the central idempotent of A4.
def inv_perm(p):
    q = [0]*4
    for i in range(4): q[p[i]] = i
    return tuple(q)
def proj_irrep(chi_fn, dim_irrep):
    P = np.zeros((12, 12))
    for p in A4:
        P += chi_fn(p)*pd(inv_perm(p))       # central idempotent uses g^{-1}; chars real
    return (dim_irrep/12.0)*P
def chi1(p):  return 1.0
P3 = proj_irrep(chi3, 3)
P1 = proj_irrep(chi1, 1)
r3 = round(np.trace(P3).real)
r1 = round(np.trace(P1).real)
print(f"\n   rank of projector onto 3-isotypic component  = {r3}   (expect 9 = 3 copies x dim 3)")
print(f"   rank of projector onto trivial-isotypic comp. = {r1}   (expect 1)")
print("   => the THREE copies of the 3-irrep span a 9-dim isotypic block; the A4-commutant")
print("      there is M_3(C) (dim 9) acting on the C^3 MULTIPLICITY space = the 'triality'/")
print("      generation index that mixes the three 3-irreps.  The full commutant algebra is")
print("      C + C + C + M_3(C)  (the three A4-singlets each give a scalar C).")

# Identify the M_3 multiplicity space concretely: decompose the 3-isotypic block as
# (C^3 multiplicity) (x) (C^3 irrep). The commutant = M_3(C) (x) I_3.
ev3, U3 = np.linalg.eigh(P3)
B3 = U3[:, ev3 > 0.5]                         # 9 orthonormal vectors spanning 3-isotypic block
# Project the commutant basis onto this block and report its dimension there.
comm_on_block = []
for c in range(dim_comm):
    X = null[:, c].reshape(12, 12)
    Xb = B3.conj().T @ X @ B3
    comm_on_block.append(Xb)
# dimension of span of {Xb}: stack and rank
stack = np.array([Xb.reshape(-1) for Xb in comm_on_block])
rk = np.linalg.matrix_rank(stack, tol=1e-9)
print(f"   commutant restricted to the 9-dim 3-isotypic block has dimension {rk}  (expect 9 = M_3).")

# DECISIVE structural check: the CENTER of the commutant algebra has dim = # of simple
# blocks.  C(+)C(+)C(+)M_3(C) has 4 simple blocks => center dim must be 4.
basis = [null[:, c].reshape(12, 12) for c in range(dim_comm)]
con = []
for Bj in basis:
    blk = np.array([(basis[i] @ Bj - Bj @ basis[i]).reshape(-1) for i in range(dim_comm)]).T
    con.append(blk)
uu, ss, vv = np.linalg.svd(np.vstack(con))
center_dim = dim_comm - int(np.sum(ss > 1e-8))
print(f"   CENTER of the commutant algebra has dimension {center_dim}  (= number of simple")
print(f"      blocks; expect 4 for C(+)C(+)C(+)M_3 ).   match: {center_dim == 4}")

print("\n[A.2 verdict] What genuinely flows: the dart space is an A4-module 1+1'+1''+3+3+3,")
print("   and its symmetry algebra (commutant) is C(+)C(+)C(+)M_3(C).  The M_3(C) is the")
print("   matrix algebra acting on the 3-dim MULTIPLICITY space of the 3-irrep (the three")
print("   copies) — a genuine, un-imposed M_3 ('three-generation mixing') structure.  No")
print("   full Clifford algebra is forced on the darts; what flows is the spectral-triple")
print("   grading (A.1) plus this M_3(C) endomorphism algebra (A.2).")

# =====================================================================================
# (B) GEOMETRY
# =====================================================================================
print("\n" + "="*86)
print("(B)  GEOMETRY")
print("="*86)

# -------------------------------------------------------------------------------------
# B.1  GIRTH via BFS on a finite patch of the Z^3 cover.
# A vertex of the cover = (sublattice s in {0,1,2,3}, cell c in Z^3).  Two vertices are
# adjacent iff some EDGE (i,j,vec) (or its reverse) connects them: (i,c)~(j,c+vec).
# -------------------------------------------------------------------------------------
print("\n[B.1] GIRTH: BFS for the shortest cycle through the basepoint (0,(0,0,0)).")
RNG = 3                                       # cells in [-RNG, RNG]^3 (patch radius)
cells = [(a, b, c) for a in range(-RNG, RNG+1) for b in range(-RNG, RNG+1) for c in range(-RNG, RNG+1)]
def vid(s, c): return (s, c)
adj = {}                                      # adjacency list on the patch
for c in cells:
    for s in range(4):
        adj.setdefault(vid(s, c), [])
def add_edge(u, v):
    if u in adj and v in adj:
        adj[u].append(v); adj[v].append(u)
for (i, j, vec) in srs.EDGES:
    vec = tuple(int(x) for x in vec)
    for c in cells:
        cj = (c[0]+vec[0], c[1]+vec[1], c[2]+vec[2])
        if cj in [tuple(x) for x in cells] or cj in set(cells):
            add_edge(vid(i, c), vid(j, cj))

# Girth = shortest cycle through the basepoint: BFS tree, first non-tree edge closes a cycle.
def girth_through(src):
    from collections import deque
    dist = {src: 0}; par = {src: None}; best = 10**9
    dq = deque([src])
    # track parent EDGE (the neighbor we came from) to forbid immediate backtrack on same edge
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if w not in dist:
                dist[w] = dist[u]+1; par[w] = u; dq.append(w)
            elif par[u] != w:                  # non-tree edge -> a cycle
                # cycle length = dist[u]+dist[w]+1, but only counts if it's a real cycle through src
                best = min(best, dist[u]+dist[w]+1)
    return best
base = vid(0, (0, 0, 0))
g = girth_through(base)
print(f"   patch radius {RNG} cells, |V| on patch = {len(adj)}")
print(f"   shortest cycle length through basepoint (girth) = {g}")
print(f"   girth == 10 ?  {g == 10}")

# -------------------------------------------------------------------------------------
# B.1b  Count distinct shortest (length-10) cycles through the basepoint and their A4 orbits.
# Enumerate closed non-backtracking walks of length 10 from base that return to base,
# canonicalise each as a set of edges to avoid double counting (direction + start).
# -------------------------------------------------------------------------------------
print("\n[B.1b] Enumerate shortest cycles through the basepoint and their A4-orbit structure.")
L = g                                         # = 10
def neighbors_d(u):                           # neighbors with the EDGE used (for non-backtracking)
    s, c = u
    out = []
    for (i, j, vec) in srs.EDGES:
        vec = tuple(int(x) for x in vec)
        if i == s:
            w = (j, (c[0]+vec[0], c[1]+vec[1], c[2]+vec[2]))
            if w in adj: out.append((w, ('E', i, j, vec, c)))      # edge id token
        if j == s:
            w = (i, (c[0]-vec[0], c[1]-vec[1], c[2]-vec[2]))
            if w in adj: out.append((w, ('E', i, j, vec, (c[0]-vec[0], c[1]-vec[1], c[2]-vec[2]))))
    return out
def edge_token(u, w):
    # canonical undirected edge token between adjacent cover-vertices
    return frozenset([u, w])

cycles = set()
def dfs(u, last_edge, depth, visited_edges, edge_path):
    if depth == L:
        if u == base:
            cyc = frozenset(edge_path)
            if len(cyc) == L:                  # a genuine 10-cycle (no repeated edges)
                cycles.add(cyc)
        return
    if depth > 0 and u == base:                # returned early -> not a length-L cycle through base only
        return
    for w, _tok in neighbors_d(u):
        et = edge_token(u, w)
        if et == last_edge:                    # non-backtracking
            continue
        if et in visited_edges:                # simple cycle (no edge repeat)
            continue
        visited_edges.add(et); edge_path.append(et)
        dfs(w, et, depth+1, visited_edges, edge_path)
        visited_edges.discard(et); edge_path.pop()

dfs(base, None, 0, set(), [])
print(f"   number of distinct simple {L}-cycles through the basepoint = {len(cycles)}")

# A4-orbit structure.  A4 permutes the 4 sublattices; the stabiliser of the basepoint
# cover-vertex (0, origin) is the C3 generated by sigma=(123).  As a CRYSTAL automorphism
# this C3 acts on a cover-vertex (s, c) by BOTH permuting the sublattice AND rotating the
# cell:  (s, c) -> (sigma[s], M c), where M is sigma's induced action on H_1=Z^3
# (e1->e3, e2->-e1, e3->-e2; cf. explore_06).  (Keeping the cell fixed is NOT an
# automorphism — it fails to map net-edges to net-edges.)  Verify M is an automorphism.
sigma = {0: 0, 1: 2, 2: 3, 3: 1}               # the 3-cycle (1 2 3); fixes sublattice 0
M = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])   # sigma_* on H_1
edgemap = {}
for (i, j, vec) in srs.EDGES:
    vec = np.array(vec); edgemap[(i, j)] = vec; edgemap[(j, i)] = -vec
autom_ok = all((sigma[i], sigma[j]) in edgemap and np.array_equal(edgemap[(sigma[i], sigma[j])], M @ np.array(vec))
               for (i, j, vec) in srs.EDGES)
print(f"   basepoint-stabiliser in A4 = C3 = <sigma=(123)>, acting as (s,c)->(sigma[s], M c).")
print(f"   sigma+M is a genuine graph automorphism of the cover (maps net-edges to net-edges): {autom_ok}")

def act_vertex(u):
    s, c = u; cc = M @ np.array(c)
    return (sigma[s], (int(cc[0]), int(cc[1]), int(cc[2])))
def act_cycle(cyc):
    return frozenset(frozenset([act_vertex(a), act_vertex(b)]) for a, b in (tuple(e) for e in cyc))

# group cycles into C3-orbits (orbit = {cycle, sigma.cycle, sigma^2.cycle})
seen = set(); orbits = []
for cy in cycles:
    if cy in seen: continue
    orb = set()
    x = cy
    for _ in range(3): orb.add(x); x = act_cycle(x)
    orbits.append(orb); seen |= orb
sizes = sorted(len(o) for o in orbits)
all_valid = all(o2 in cycles for o in orbits for o2 in o)   # images are genuine 10-cycles
print(f"   under the C3 basepoint-stabiliser: {len(orbits)} orbits, orbit sizes = {sizes}")
print(f"   all C3-images are valid 10-cycles: {all_valid};  orbit sizes sum to {sum(sizes)} (= {len(cycles)} cycles)")
print(f"   => the {len(cycles)} shortest cycles organize as {len(orbits)} free C3-triples (size-3 orbits);")
print(f"      no C3-symmetric (size-1) cycle through the basepoint.")

# -------------------------------------------------------------------------------------
# B.2  CHIRALITY.  Is A(-k) ~ A(k) by a PERMUTATION? (proper symmetry, always true by
# time-reversal/conjugation).  Is there an IMPROPER (orientation-reversing) symmetry that
# realises k -> -k as a point-group element acting as a graph automorphism on the COVER
# (i.e. an odd permutation realising the homology inversion)?  No improper => chiral.
# -------------------------------------------------------------------------------------
print("\n[B.2] CHIRALITY of srs.")
# (i) A(-k) = conj(A(k)) always (Hermitian, real-edge), so A(-k) ~ A(k) trivially by complex
#     conjugation (time reversal). The real question: is k->-k induced by a POINT-GROUP element?
ks = [(0.2, 0.25, 0.3), (0.1, 0.37, 0.42), (0.15, 0.05, 0.31)]
print("   (i) A(-k) vs conj(A(k)) (time reversal, always a symmetry of spectrum):")
for k in ks:
    same = np.allclose(srs.adjacency(tuple(-np.array(k))), srs.adjacency(k).conj())
    print(f"       k={k}:  A(-k) == conj(A(k)) ?  {same}")

# (ii) Does any permutation p of {0,1,2,3} (proper A4 or improper odd) realise the homology
#      inversion M = -I on Z^3, i.e. for every edge (i,j,vec) there is an edge between p[i],p[j]
#      with homology vector -vec (up to the spanning-tree gauge)?  Build the induced action on
#      H_1 for each permutation and test whether ANY gives M = -I.
# Homology of an edge in the chosen basis: cotree edges 12,13,23 carry e1,e2,e3; tree edges 0.
def homology_of_edge(i, j):
    # in K4 with tree {01,02,03}: the directed edge i->j has homology = (loop i->j->0->i style)
    # Simplify: only cotree edges carry homology.  Map unordered {i,j} to its homology vector.
    table = {frozenset([1, 2]): np.array([1, 0, 0]),
             frozenset([1, 3]): np.array([0, 1, 0]),
             frozenset([2, 3]): np.array([0, 0, 1])}
    return table.get(frozenset([i, j]), np.array([0, 0, 0]))
# The action of a permutation p on H_1: it permutes edges; cotree edges may map to tree edges,
# so the induced map on H_1 = Z^3 is obtained by expressing p(cotree generator) in the cycle basis.
# Build it via the cycle space: for each generator cotree edge e, its fundamental cycle is
# e + (tree path).  Apply p, re-express in the 3 fundamental cycles.
def fundamental_cycle_vector(i, j):
    # fundamental cycle of cotree edge {i,j} (i,j in 1,2,3): i -> j -> 0 -> i, as a
    # signed edge vector over the 6 edges; but for H_1 we only need its class = the basis vec.
    return homology_of_edge(i, j)
EDGE_SET = [frozenset([e[0], e[1]]) for e in srs.EDGES]
def induced_on_H1(p):
    cols = []
    for gen in [(1, 2), (2, 3) if False else (1, 3), (2, 3)]:
        # apply p to the cotree edge {i,j}; its image is some edge {p[i],p[j]}
        i, j = gen
        img = frozenset([p[i], p[j]])
        # express the image edge's fundamental cycle in the basis.  An image that is a TREE
        # edge {0,x} has homology 0; a cotree image gives that basis vector (with a sign from
        # orientation, which we fix by the head-tail convention head-tail).
        if img in [frozenset([1, 2]), frozenset([1, 3]), frozenset([2, 3])]:
            cols.append(homology_of_edge(*tuple(img)))
        else:
            # tree edge -> its fundamental class is 0 only if it stays a tree edge; but a
            # permuted tree edge can become a cotree edge.  Handle generally below.
            cols.append(np.array([0, 0, 0]))
    return np.array(cols).T

# The simple table above is gauge-dependent; do it ROBUSTLY via the Bloch matrix instead:
# p is a symmetry realising k -> M k iff  P_v(p) A(k) P_v(p)^T == A(M k)  for all k, where
# P_v(p) is the 4x4 vertex permutation.  Test M = -I (inversion) for every p in S4.
def Pv(p):
    M = np.zeros((4, 4))
    for i in range(4): M[p[i], i] = 1
    return M
test_ks = [np.array([0.13, 0.27, 0.41]), np.array([0.22, 0.05, 0.34]), np.array([0.09, 0.31, 0.18])]
def realises_inversion(p):
    Pp = Pv(p)
    for k in test_ks:
        lhs = Pp @ srs.adjacency(tuple(k)) @ Pp.T
        rhs = srs.adjacency(tuple(-k))           # M = -I  => A(Mk)=A(-k)
        if not np.allclose(lhs, rhs): return False
    return True
proper_inv = [p for p in A4 if realises_inversion(p)]
improper_inv = [p for p in ODD if realises_inversion(p)]
print("\n   (ii) Is the homology inversion k -> -k realised by a PERMUTATION automorphism?")
print(f"        proper (A4) permutations realising A(-k)=P A(k) P^T:   {len(proper_inv)}  {proper_inv}")
print(f"        improper (odd) permutations realising it:              {len(improper_inv)}  {improper_inv}")

# (iii) General point-group test: which permutations are symmetries AT ALL (realise SOME M),
#       and are any of them orientation-reversing (det M = -1)?  A net is achiral iff its
#       full symmetry group contains an orientation-reversing element.
def induced_M(p):
    # find integer M with A(Mk) = Pv(p) A(k) Pv(p)^T for all k, if it exists; else None.
    Pp = Pv(p)
    # M acts on the 3 homology basis vectors; read off columns by matching phases.
    # Use three independent k's and solve; verify.
    # Build map: for basis e_m, the permuted lattice vector is read from how edge-phases move.
    # Simpler: try all M in {-1,0,1}^{3x3} with det != 0 and |entries|<=1 (point group of cubic
    # crystals has entries in {-1,0,1}); test the Bloch identity.
    cand = []
    vals = [-1, 0, 1]
    # restrict to signed permutation matrices (the only orthogonal integer matrices) -> 48
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            M = np.zeros((3, 3), int)
            for r in range(3): M[r, perm[r]] = signs[r]
            ok = True
            for k in test_ks:
                lhs = Pp @ srs.adjacency(tuple(k)) @ Pp.T
                rhs = srs.adjacency(tuple(M @ k))
                if not np.allclose(lhs, rhs): ok = False; break
            if ok: cand.append(M)
    return cand
print("\n   (iii) Full point-group scan: for each permutation, find integer signed-perm M with")
print("         A(Mk)=Pv(p)A(k)Pv(p)^T; collect orientation-reversing (det=-1) symmetries.")
orient_reversing = []
proper_syms = 0
for p in S4:
    Ms = induced_M(p)
    for M in Ms:
        d = int(round(np.linalg.det(M)))
        if d == -1: orient_reversing.append((p, M))
        if d == +1: proper_syms += 1
print(f"        # (permutation, M) proper symmetries (det M = +1):       {proper_syms}")
print(f"        # (permutation, M) IMPROPER symmetries (det M = -1):     {len(orient_reversing)}")
if orient_reversing:
    print(f"        example improper: p={orient_reversing[0][0]}, M=\n{orient_reversing[0][1]}")

print("\n[B.2 verdict]")
if len(orient_reversing) == 0:
    print("   NO orientation-reversing (improper) point-group symmetry exists.")
    print("   => srs is CHIRAL: the net and its mirror image srs* are distinct (not superposable).")
    print("   (A(-k)~A(k) holds only by complex conjugation = time reversal, which is NOT a")
    print("    spatial/point-group operation, so it does not make the net achiral.)")
else:
    print("   An orientation-reversing symmetry WAS found => srs would be achiral.  (Investigate.)")

print("\n" + "="*86)
print("DONE.")
print("="*86)
