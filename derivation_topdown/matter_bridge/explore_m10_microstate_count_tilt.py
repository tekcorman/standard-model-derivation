"""
explore_m10 -- MICROSTATE-COUNT x INVERSION-AXIS TILT over edge-configurations (sealed reading-sheet).

PURE MATH, walled. Builds ONLY on the verified bare object (../dirac_srs_mdl/srs.py = Sunada's K_4 crystal)
and the forced inversion axis derived in m07 (the crystallographic vacuum = inversion -I; srs-z = conj(srs
net) = A(-k); on the chosen Z^3 voltage assignment this is the homology map v -> -v).

A CONFIGURATION = a subset of the 6 base edges of K_4 (the object MDL forces). For each configuration we
derive TWO independent quantities from the object and combine them:

 (1) MICROSTATE-COUNT  (a multiplicity, NOT the recurrence modulus/phase already done in m09).
     We let the object choose. Candidates compared:
       (a) tau(S)   = number of spanning trees of the occupied sub-network (Kirchhoff complexity)
                      = the natural count of "maximal admissible realizations" / weighted graph complexity.
       (b) Ncw_m(S) = number (count, not modulus) of closed non-backtracking walks = Tr B_S^m (the
                      multiplicity the Ihara/Ruelle zeta enumerates: prime-orbit counts).
       (c) dim ker / b_1 of the occupied sub-network (the admissible 1-cycle state-space dimension =
                      exterior-algebra grade of harmonic 1-forms on S).
     We report how each scales with the NUMBER of occupied edges and whether distinct configurations of
     the same size carry distinct counts.

 (2) ORIENTATION under the forced inversion axis. The inversion -I acts on the homology label of each
     edge by v -> -v (tree edges v=0 are FIXED; cotree edges {12,13,23} carry v=e1,e2,e3 -> -e1,-e2,-e3).
     A configuration S (a set of phased edges) is SYMMETRIC if the inversion maps the occupied-edge set to
     itself, ASYMMETRIC otherwise. We classify ALL 2^6 = 64 subsets.

 (3) COMBINE: the object's own pairing of a multiplicity with the axis is the inversion-PROJECTED count:
       symmetric part  P+ = (1/2)(count(S) + count(-S))  and  asymmetric part P- = (1/2)(count - count(-S)).
     For a count that is itself a graph invariant (tau, b_1, Tr B^m are isomorphism invariants and the
     inversion is a graph automorphism of the LABELLED object), count(-S) is the count of the inverted
     configuration. The natural combined quantity is the count WEIGHTED by the symmetry class indicator
     (the inversion eigenvalue +-1 of the configuration), i.e. count(S) carried with its axis-parity.

 (4) DISTRIBUTION: tabulate combined quantity over all 64 configs, by edge-count and by symmetry class.
"""
import numpy as np
from itertools import combinations
from collections import defaultdict
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs  # verified bare object; EDGES carry the Z^3 homology labels

# The 6 base edges of K_4 with their forced homology labels (from srs.EDGES):
#   tree {01,02,03} -> v=0 (inversion-FIXED);  cotree {12,13,23} -> v=e1,e2,e3 (inversion-FLIPPED).
EDGES = srs.EDGES            # [(i,j,v), ...], len 6
NE = len(EDGES)              # 6
assert NE == 6

# ---- the forced inversion axis: v -> -v on each edge label ----
def inverted_edge(e):
    i, j, v = e
    return (i, j, tuple(-np.array(v)))

# An edge's IDENTITY for set-membership: (unordered endpoints, homology direction up to overall sign?).
# The inversion is a genuine map on the PHASED edges. Two phased edges are "the same occupied bond" iff
# same endpoints AND same homology label v. Inversion sends label v -> -v. For tree edges v=0=-0 (fixed);
# for cotree edges v=e_a -> -e_a, which is a DIFFERENT label on the SAME endpoints. So on the base K_4
# the inversion FIXES every edge as an UNDIRECTED BOND (same endpoints) but FLIPS its homology phase.
#
# => Two natural readings of "configuration set under inversion":
#   (A) configuration = set of BONDS (endpoints only). Then inversion fixes EVERY bond-set (all symmetric):
#       the axis is invisible at bond level -- this is the "spectator" collapse the framework warns about.
#   (B) configuration = set of PHASED edges (endpoints + homology label). Then inversion v->-v genuinely
#       moves cotree edges, and SYMMETRIC means the phase-set is inversion-closed.
# Reading (B) is the one that engages the forced axis (m07: the phase arg(M) IS the inversion registration).
# We use (B): occupancy is of PHASED darts-as-bonds; a config is inversion-symmetric iff for every occupied
# cotree edge with label v, the edge with label -v is ALSO occupied (or it is a tree edge, auto-fixed).
#
# But on the bare K_4 each cotree pair {i,j} has ONLY ONE edge (label +e_a); the label -e_a lives on the
# SAME bond. So inversion maps the single phased edge (i,j,+e_a) to (i,j,-e_a): NOT in the original edge
# list. The honest statement: the inversion acts on the configuration by conjugating its Bloch phase. A
# configuration's INVERSION IMAGE is the same bond-subgraph carried with conjugated phases. Its microstate
# COUNTS (tau, b_1, Tr B^m) are phase-INDEPENDENT graph invariants, so count(-S)=count(S) ALWAYS.
# Therefore the axis parity is read NOT from the count but from how the occupied COTREE labels sit under
# v->-v: a config is SYMMETRIC under the axis iff its occupied set is fixed as a LABELLED voltage graph,
# i.e. iff it contains NO cotree edge (all labels 0) OR its cotree labels are inversion-paired -- which on
# the single-copy base means iff it has ZERO or (the impossible) paired cotree edges.
#
# The clean, object-honest formalization that does NOT collapse: classify by the NUMBER c of occupied
# COTREE (flipped, v!=0) edges. The inversion fixes the config's voltage class iff the flipped-set is
# inversion-invariant. On the base each flip is its own orbit (e_a and -e_a never both occur), so:
#     c = 0  -> SYMMETRIC (purely tree / inversion-fixed labels)
#     c >= 1 -> ASYMMETRIC (carries a net, unpaired homology orientation flipped by the axis)
# This is exactly the m07 statement that the inversion FIXES the gap direction (tree, v=0) and FLIPS the
# axial/phase direction (cotree, v!=0). c is the "tilt": # of occupied edges that the forced axis moves.

def cotree_count(S):
    """number of occupied edges with nonzero homology label = edges the inversion axis FLIPS = TILT c."""
    return sum(1 for e in S if any(np.array(e[2]) != 0))

# ---- the FORCED axis: inversion -I (m07), tested as a LABELLED-EDGE automorphism --------------------
# m07 forces the inversion -I (v -> -v) as the canonical axis. TWO honest groups can act:
#   (A) the GRAPH-AUTOMORPHISM group A_4 (rotations of K_4): each g permutes vertices and hence the
#       labelled edges; combine with v->-v.  This acts directly on the configuration (a labelled-edge set).
#   (B) the FULL SPATIAL point group O (order 24, STRUCTURE.md Bucket-A: the embedded net's lattice
#       symmetry, including the 4_1 screws A_4 lacks).  O acts on the homology lattice Z^3 only.
# Under (A): inversion (combined with any A_4 vertex-rotation) restores the occupied PHASED-edge set iff
#   the occupied cotree-label set is fixed by v->-v up to a vertex relabelling -> ONLY c=0 (empty cotree).
#   => SYM = {c=0} (8 configs), ASYM = {c>=1} (56).   [the conservative, directly-justified dichotomy]
# Under (B): a proper rotation in O maps {e1,e2,e3} -> {-e1,-e2,-e3} (verified separately), so the FULL
#   cotree triple c=3 is ALSO restored by inversion+O-rotation -> SYM = {c=0, c=3} (16), ASYM = {c=1,c=2}
#   (48). This is the EMERGENT full/empty=symmetric vs partial=asymmetric split, but it needs the lattice
#   point group O (the 4_1 screws), which is a symmetry of the NET'S LATTICE, not of the bare K_4 vertices.
# We report BOTH. The primary `axis_class` uses (A) (the labelled-edge automorphism = most conservative).
import itertools as _it
def _parity(p):
    p = list(p); seen = [False] * len(p); par = 0
    for i in range(len(p)):
        if not seen[i]:
            j = i; c = 0
            while not seen[j]:
                seen[j] = True; j = p[j]; c += 1
            par += c - 1
    return par % 2
_A4 = [p for p in _it.permutations(range(4)) if _parity(p) == 0]   # 12 rotations of K_4
def _key(e):
    i, j, v = e
    return (frozenset((i, j)), tuple(v))
def _img(g, e):                      # apply rotation g on vertices AND inversion v->-v on the label
    i, j, v = e
    return (frozenset((g[i], g[j])), tuple(-np.array(v)))
def axis_class(S):
    """(A) SYMMETRIC iff (inversion v->-v) combined with SOME A_4 vertex-rotation restores the occupied
    PHASED-edge set.  Conservative axis (labelled-edge automorphism).  Gives SYM={c=0}."""
    target = frozenset(_key(e) for e in S)
    for g in _A4:
        if frozenset(_img(g, e) for e in S) == target:
            return "SYM"
    return "ASYM"
def axis_class_O(S):
    """(B) EMERGENT lattice dichotomy: SYM iff the occupied cotree-direction set is EMPTY (c=0) or the FULL
    triple (c=3) -- both fixed by inversion combined with the lattice point group O (the 4_1 screw mapping
    {e1,e2,e3}->{-e1,-e2,-e3}).  PARTIAL (c=1,2) = ASYM.  Needs the net's lattice symmetry O, not just A_4."""
    c = cotree_count(S)
    return "SYM" if c in (0, 3) else "ASYM"

# ---------- microstate-count notions, computed on the occupied sub-network ----------
def adjacency_and_verts(S):
    vs = sorted({a for (i, j, v) in S for a in (i, j)})
    idx = {v: t for t, v in enumerate(vs)}
    n = len(vs)
    A = np.zeros((n, n))
    for (i, j, v) in S:
        A[idx[i], idx[j]] += 1
        A[idx[j], idx[i]] += 1   # multigraph adjacency (counts parallel edges)
    return A, vs

def spanning_trees(S):
    """(a) Kirchhoff weighted complexity tau(S) = # spanning trees of the occupied sub-network.
    For a DISCONNECTED sub-network tau is defined as 0 (no spanning tree of the whole) -- we instead
    report the PRODUCT of spanning-tree counts of the components (the natural multiplicity of maximal
    realizations), which equals the standard tau for connected S and is the proper multiplicity otherwise."""
    A, vs = adjacency_and_verts(S)
    n = len(vs)
    if n == 0:
        return 0
    # components
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    deg = A.sum(1)
    for a in range(n):
        for b in range(a + 1, n):
            if A[a, b] > 0:
                ra, rb = find(a), find(b)
                if ra != rb: parent[ra] = rb
    comp = defaultdict(list)
    for a in range(n): comp[find(a)].append(a)
    prod = 1
    for nodes in comp.values():
        if len(nodes) == 1:
            prod *= 1   # isolated vertex: one (empty) tree
            continue
        sub = A[np.ix_(nodes, nodes)]
        L = np.diag(sub.sum(1)) - sub
        # Kirchhoff: any cofactor = # spanning trees of this component
        M = L[1:, 1:]
        t = round(float(np.linalg.det(M)))
        prod *= t
    return prod

def b1(S):
    """(c) first Betti number of the occupied sub-network = dim of admissible 1-cycle space (harmonic
    1-form grade). b_1 = E - V + C."""
    A, vs = adjacency_and_verts(S)
    n = len(vs)
    if n == 0:
        return 0
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (i, j, v) in S:
        pass
    # components by edges
    idx = {v: t for t, v in enumerate(vs)}
    for (i, j, v) in S:
        ra, rb = find(idx[i]), find(idx[j])
        if ra != rb: parent[ra] = rb
    C = len({find(x) for x in range(n)})
    E = len(S); V = n
    return E - V + C

def nb_closedwalk_counts(S, mmax=12):
    """(b) the COUNT (multiplicity) of closed non-backtracking walks of each length =
    Tr B_S^m (m=1..mmax). This is what the Ihara/Ruelle zeta enumerates (prime-orbit counts), a pure
    multiplicity, NOT the |h| modulus. Built on the unphased sub-network (phase-independent counts)."""
    darts = []
    for (i, j, v) in S:
        darts.append((i, j)); darts.append((j, i))
    nd = len(darts)
    if nd == 0:
        return [0] * mmax, 0
    out_of = defaultdict(list)
    for d, (t, h) in enumerate(darts): out_of[t].append(d)
    B = np.zeros((nd, nd))
    for x, (tx, hx) in enumerate(darts):
        for y in out_of[hx]:
            ty, hy = darts[y]
            if not (hy == tx):   # forbid immediate reversal
                B[x, y] = 1.0
    counts = []
    P = np.eye(nd)
    for m in range(1, mmax + 1):
        P = P @ B
        counts.append(int(round(np.trace(P))))
    return counts, nd

# ============================================================
print("=" * 92)
print(" MICROSTATE-COUNT x INVERSION-AXIS TILT over the 2^6=64 edge-configurations of K_4 (srs base)")
print("=" * 92)
print(f"\n base object: K_4, {NE} edges; tree {{01,02,03}} v=0 (axis-FIXED), cotree {{12,13,23}} v=e1,e2,e3 (axis-FLIPPED).")
print(" forced axis = inversion -I (m07): v -> -v.  TILT c(S) = # occupied edges with v!=0 (axis-flipped).")
print(" SYMMETRIC <=> c=0 (labels inversion-fixed);  ASYMMETRIC <=> c>=1 (net unpaired flipped homology).")

# enumerate all subsets
all_subsets = []
for r in range(NE + 1):
    for combo in combinations(range(NE), r):
        S = [EDGES[t] for t in combo]
        all_subsets.append((combo, S))

# ---- PART 1: which microstate-count does the object pick? Compare tau, b1, Tr B^m ----
print("\n" + "=" * 92)
print(" PART 1 -- compare candidate microstate-counts; how each scales with edge-count & whether")
print("           equal-size configs carry DISTINCT counts (a genuine spread).")
print("=" * 92)
print(f"\n {'E':>2} {'#configs':>8}  {'tau: distinct vals':>34}  {'b1: distinct vals':>22}")
by_E = defaultdict(list)
for combo, S in all_subsets:
    by_E[len(S)].append((combo, S))
for E in range(NE + 1):
    taus = sorted(set(spanning_trees(S) for _, S in by_E[E]))
    b1s = sorted(set(b1(S) for _, S in by_E[E]))
    print(f" {E:>2} {len(by_E[E]):>8}  {str(taus):>34}  {str(b1s):>22}")

print("""
 READING (forced):  tau (spanning-tree complexity) gives a genuine SPREAD that grows with edge-count
   and DISTINGUISHES equal-size configs (e.g. a triangle has tau=3, a 3-edge path tau=1). b_1 is the
   coarse cycle-rank (0,1,2,3). The closed-NB-walk counts Tr B^m are an infinite multiplicity vector;
   its first nonzero entry is Tr B^g (g=girth) and equals 2 x (#shortest cycles). We use tau as the
   object's natural single-number microstate-count (Kirchhoff = the # of maximal admissible realizations,
   a true multiplicity), and cross-check with Tr B^m below.""")

# Cross-check: smallest cycle multiplicity from Tr B^m for a few configs
print(" cross-check (closed-NB-walk COUNT Tr B^m, m=1..8) for representative configs:")
reps = {
    "triangle {01,02,12}": [EDGES[0], EDGES[1], EDGES[3]],
    "3-path  {01,02,03}":  [EDGES[0], EDGES[1], EDGES[2]],
    "full K4 (6 edges)":   list(EDGES),
}
for name, S in reps.items():
    cw, nd = nb_closedwalk_counts(S, 8)
    print(f"   {name:24s}: tau={spanning_trees(S):>3}  b1={b1(S)}  TrB^m={cw}")

# ---- PART 2: the axis classification of ALL 64 configs, by edge-count ----
print("\n" + "=" * 92)
print(" PART 2 -- inversion-axis class (SYM c=0 / ASYM c>=1) across all 64 configs, by edge-count & tilt c")
print("=" * 92)
print(f"\n {'E':>2}  {'A:#SYM':>7} {'A:#ASYM':>8} | {'O:#SYM':>7} {'O:#ASYM':>8}   tilt-c histogram")
for E in range(NE + 1):
    cs = defaultdict(int)
    a_s = a_a = o_s = o_a = 0
    for _, S in by_E[E]:
        cs[cotree_count(S)] += 1
        if axis_class(S) == "SYM": a_s += 1
        else: a_a += 1
        if axis_class_O(S) == "SYM": o_s += 1
        else: o_a += 1
    hist = " ".join(f"{c}:{cs[c]}" for c in sorted(cs))
    print(f" {E:>2}  {a_s:>7} {a_a:>8} | {o_s:>7} {o_a:>8}   {hist}")
print("""
 TWO HONEST DICHOTOMIES (by tilt c = # occupied edges the inversion axis flips):
   AXIS (A) -- inversion + A_4 GRAPH automorphism (acts directly on the labelled-edge config):
     SYMMETRIC = { c=0 } (8 configs: subsets of the 3 tree edges, labels v=0, inversion-fixed).
     ASYMMETRIC = { c>=1 } (56 configs: any occupied cotree edge carries an unpaired flipped label).
     [the conservative, directly-justified split: a tree-only config is the ONLY kind A_4+inversion fixes.]
   AXIS (B) -- inversion + the net's LATTICE point group O (the 4_1 screws; STRUCTURE Bucket-A):
     SYMMETRIC = { c=0 (EMPTY) } U { c=3 (FULL cotree triple) }  -> 8 + 8 = 16 configs.
       c=3 is restored because a proper rotation in O maps {e1,e2,e3} -> {-e1,-e2,-e3} (verified), so the
       COMPLETE homology triple is axis-symmetric while PARTIAL triples are not.
     ASYMMETRIC = { c=1, c=2 (PARTIAL) }  -> 24 + 24 = 48 configs.
     => FULL-or-EMPTY = symmetric, PARTIAL = asymmetric. EMERGENT from the object (A_4/O on the homology
        3-irrep + inversion); needs the net's lattice symmetry O, NOT just the bare K_4 vertex group A_4.""")

# ---- PART 3: COMBINE -- the microstate-count carried with its axis parity; the distribution ----
print("\n" + "=" * 92)
print(" PART 3 -- COMBINED quantity = microstate-count tau, resolved by axis class (SYM vs ASYM),")
print("           tabulated by edge-count.  THE DELIVERABLE DISTRIBUTION.")
print("=" * 92)
print(" (using AXIS (B): SYM = full-or-empty cotree {c=0,c=3}; ASYM = partial {c=1,c=2} -- the emergent split)")
print(f"\n {'E':>2} | {'SYM (c=0 or c=3)':>26} | {'ASYM (c=1 or c=2)':>40}")
print(f" {'':>2} | {'tau values (count:#cfg)':>26} | {'tau values (count:#cfg)':>40}")
print(" " + "-" * 78)
sym_all = []
asym_all = []
for E in range(NE + 1):
    symv = defaultdict(int); asymv = defaultdict(int)
    for _, S in by_E[E]:
        t = spanning_trees(S)
        if axis_class_O(S) == "SYM":
            symv[t] += 1; sym_all.append(t)
        else:
            asymv[t] += 1; asym_all.append(t)
    symstr = " ".join(f"{k}:{v}" for k, v in sorted(symv.items())) or "-"
    asymstr = " ".join(f"{k}:{v}" for k, v in sorted(asymv.items())) or "-"
    print(f" {E:>2} | {symstr:>26} | {asymstr:>40}")

print("\n  Aggregate over ALL configs:")
def summarize(name, vals):
    dv = sorted(set(vals))
    print(f"   {name:12s}: {len(vals):>3} configs, tau in {dv}, distinct={len(dv)}, "
          f"max={max(vals)}, mean={np.mean(vals):.3f}")
summarize("SYM(c0,c3)", sym_all)
summarize("ASYM(c1,c2)", asym_all)

# Distribution character: does tau SPREAD (unlike the flat modulus/phase of m09)?
all_tau = sym_all + asym_all
print(f"\n  WHOLE distribution of tau over 64 configs: distinct values = {sorted(set(all_tau))}")
print(f"    => a genuine integer SPREAD (NOT a single locked value, NOT a continuum): {len(set(all_tau))} distinct counts.")

# Tilt-resolved tau (the branching x tilt product): for each config report tau weighted by axis parity.
# The object's natural combination (m07): the count tau is the inversion-EVEN multiplicity; the tilt c is
# the inversion-ODD orientation. The combined per-config quantity:  (tau, parity=+ if c=0 else -).
print("\n" + "=" * 92)
print(" PART 4 -- tilt-resolved spread: tau by tilt-c (does the cycle-count grow with how many edges")
print("           the axis flips? does the SYMMETRIC class differ from the ASYMMETRIC class?)")
print("=" * 92)
by_c = defaultdict(list)
for _, S in all_subsets:
    by_c[cotree_count(S)].append(spanning_trees(S))
print(f"\n {'tilt c':>6} {'#configs':>9} {'tau values (distinct)':>40} {'max tau':>8} {'mean tau':>9}")
for c in sorted(by_c):
    vals = by_c[c]
    print(f" {c:>6} {len(vals):>9} {str(sorted(set(vals))):>40} {max(vals):>8} {np.mean(vals):>9.3f}")

print("""
 SYMMETRIC vs ASYMMETRIC (explicit, AXIS (B) = inversion + lattice point group O; SYM={c=0,c=3}):
   SYM class (c=0 OR c=3, 16 configs): carries BOTH extremes of branching --
     c=0 (empty cotree, 8 forests): tau=1, b_1=0 (no cycle) -- the LOW end.
     c=3 (full cotree triple, 8 configs): tau in {3,8,16} -- the HIGH end (these are the densest cyclic
       sub-networks, incl. full K_4 tau=16). So the symmetric class occupies the EXTREMES.
   ASYM class (c=1 or c=2, 48 configs): tau in {1,3,4,8} -- the INTERMEDIATE, partial-occupation spread.
 => The branching count is NOT flat across the classes: the SYMMETRIC class holds the extreme low (full-
    empty, tau=1) and extreme high (full-occupied, tau up to 16); the ASYMMETRIC (partial) class holds the
    intermediate spread. Unlike the m09 modulus (locked to sqrt2) and phase (flat continuum), the
    microstate-COUNT tau genuinely SPREADS over discrete integers {0,1,3,4,8,16}, and the spread tracks
    the tilt c monotonically (mean tau: c0=0.875, c1=1.5, c2=2.75, c3=6.5).""")

# ---- PART 5: same-clock; forced vs choice ----
print("\n" + "=" * 92)
print(" PART 5 -- same-clock check; FORCED vs CHOICE")
print("=" * 92)
print("""
 SAME CLOCK: tau (Kirchhoff) and Tr B^m (closed-NB-walk counts) are BOTH determinants/traces of the SAME
   operators the object already uses -- the graph Laplacian L=3I-A (whose Bloch det is the srs spanning-
   tree entropy, STRUCTURE.md sec.3 Mahler measure) and the Hashimoto B (the geodesic-flow generator =
   the Ihara=Ruelle zeta, sec.7). Both microstate-counts read off the ONE clock (the NB/zeta operator).
   The tilt c reads off the SAME object's forced inversion axis (m07). One object, one clock.
 FORCED:
   - the configuration alphabet = the 6 base edges (MDL forces K_4: explore_05). [FORCED]
   - the partition tree(v=0)/cotree(v!=0) = the chosen Z^3 voltage basis (b_1=3 spanning tree); the
     inversion v->-v fixes the tree, flips the cotree. The 8 SYM = subsets of the 3 tree edges. [FORCED
     given the voltage basis; the SPECIFIC tree {01,02,03} is a basis CHOICE, but the COUNT 2^3=8 of
     inversion-fixed configs and the dichotomy 'cycle <=> occupies cotree' are basis-independent because
     ANY spanning tree has 3 edges and any cycle must use >=1 cotree edge.]
   - tau=1 & b_1=0 for every forest, tau>1 only with a cycle: forced (Kirchhoff).
   - the inversion -I as the canonical (unique central orientation-reversing) axis: FORCED (m07).
 CHOICE:
   - WHICH count to call 'the' microstate-count: we let the object pick tau (Kirchhoff complexity =
     # maximal admissible realizations) as the single-number multiplicity, cross-checked by Tr B^m.
     Both are phase-independent invariants, so neither distinguishes S from its inversion image by VALUE;
     the axis enters only through the tilt c (which edges are occupied), exactly as m07 says (the count is
     inversion-EVEN, the orientation/phase is inversion-ODD).
   - WHICH edges are occupied (the configuration) is the free input, not forced by {D,srs,MDL} (as in m09).
 FLAG (needs structure beyond these 3 dirs): combining 'branching x tilt' into a SINGLE scalar with a
   definite weight (e.g. tau^alpha or tau * f(c)) is NOT forced by the bare object -- the object forces the
   two factors and their parity classes, but the exponent/weight of the product would need an external
   principle. Reported here as the two factors + their joint distribution, NOT a fitted product.
""")
print("[m10 done]")
