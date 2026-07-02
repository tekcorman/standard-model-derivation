"""
explore_m09 -- RECURRENCE-RATE SPREAD OF OCCUPIED-EDGE PATTERNS (sealed reading-sheet extension).

THE HYPOTHESIS (pure math, no physics): a PATTERN = a set of occupied edges (a sub-network) of
the srs object. Its RECURRENCE RATE = the non-backtracking (Hashimoto) spectral radius restricted
to the occupied edges = |h| of the restricted Ihara zeta. Question: do equal-edge-count / equal-b1
patterns carry DISTINCT rates (a genuine spread that could host distinct masses), or does the rate
COLLAPSE to a function of b1 alone, or LOCK to the Ramanujan shell with variation only in phase?

We work in finite (Z_N)^3 covers of K_4 (Sunada's K_4 crystal = srs), built explicitly. Multi-cell
patterns then have b1 that can exceed 3 and equal-edge-count patterns need not be isomorphic.

NB / Hashimoto restricted to a sub-network S of occupied edges: build the directed-edge (dart)
operator B_S where dart a -> dart b allowed iff head(a)=tail(b) and not a reversal, restricted to
darts of S. rho(B_S) = recurrence rate of the pattern.  rho=0 (forest/no NB cycle), rho>=1 once a
cycle exists; on a k-regular shell the Ramanujan value is sqrt(k-1)=sqrt2.
"""
import numpy as np
from itertools import combinations
import sys

# ---------- build a finite (Z_N)^3 cover of K_4 explicitly ----------
# K_4 spanning tree {01,02,03} voltage 0; cotree {12,13,23} voltages e1,e2,e3.
K4_EDGES = [(0,1,(0,0,0)),(0,2,(0,0,0)),(0,3,(0,0,0)),
            (1,2,(1,0,0)),(1,3,(0,1,0)),(2,3,(0,0,1))]

def build_cover(N):
    """Vertices = (v, cell) with cell in (Z_N)^3. Returns (vertices, edges) of the finite cover.
    edge between (i,c) and (j, c+voltage). Undirected edge list (no duplicates)."""
    def add(c, v):
        return tuple((np.array(c)+np.array(v)) % N)
    verts = [(v, (a,b,cc)) for v in range(4) for a in range(N) for b in range(N) for cc in range(N)]
    vidx = {vv:i for i,vv in enumerate(verts)}
    edges = []
    seen = set()
    for c in [(a,b,cc) for a in range(N) for b in range(N) for cc in range(N)]:
        for (i,j,vol) in K4_EDGES:
            u = (i, tuple(c)); w = (j, add(c, vol))
            key = frozenset([u,w]) if u!=w else None
            if key is None:  # self-loop impossible here
                continue
            if key in seen:
                continue
            seen.add(key)
            edges.append((vidx[u], vidx[w]))
    return verts, edges

# ---------- non-backtracking spectral radius of a sub-network ----------
def nb_radius_and_b1(edge_subset, edges):
    """edge_subset: list of (a,b) undirected edges (subset of `edges`).
    Build Hashimoto B on the darts of the sub-network; return (rho(B), b1, n_components, n_verts)."""
    # incident vertices
    vs = sorted({a for e in edge_subset for a in e})
    if not edge_subset:
        return 0.0, 0, 0, 0
    # darts: each undirected edge -> two directed darts
    darts = []
    for (a,b) in edge_subset:
        darts.append((a,b)); darts.append((b,a))
    nd = len(darts)
    # adjacency of darts: dart x=(t,h) -> dart y=(t2,h2) allowed iff h==t2 and not (t2==h and h2==t)
    # build head index
    from collections import defaultdict
    out_of = defaultdict(list)  # tail -> list of dart indices
    for idx,(t,h) in enumerate(darts):
        out_of[t].append(idx)
    B = np.zeros((nd,nd))
    for x,(tx,hx) in enumerate(darts):
        for y in out_of[hx]:
            ty,hy = darts[y]
            if ty==hx and not (hy==tx):  # forbid immediate reversal
                B[x,y] = 1.0
    rho = max(abs(np.linalg.eigvals(B))) if nd else 0.0
    # b1 of sub-network = E - V + C
    E = len(edge_subset); V = len(vs)
    # components via union-find
    parent = {v:v for v in vs}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for (a,b) in edge_subset:
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    C = len({find(v) for v in vs})
    b1 = E - V + C
    return rho, b1, C, V

# ============================================================
print("="*70)
print("RECURRENCE-RATE SPREAD: occupied-edge patterns on (Z_N)^3 cover of K_4 (srs)")
print("="*70)

# Use N=2 cover: 32 vertices, plenty of multi-cell patterns; edges:
verts, edges = build_cover(2)
print(f"\n(Z_2)^3 cover: {len(verts)} vertices, {len(edges)} edges, 3-regular check:")
deg = {}
for (a,b) in edges:
    deg[a]=deg.get(a,0)+1; deg[b]=deg.get(b,0)+1
print(f"  degrees: {sorted(set(deg.values()))}  (expect [3])")

# Sanity: full-cover NB radius should be ~ k-1 = 2 (Perron of full 3-regular graph)
rho_full,b1_full,C,V = nb_radius_and_b1(edges, edges)
print(f"  FULL cover NB radius rho(B) = {rho_full:.4f}  (expect k-1 = 2);  b1 = {b1_full}")

# ============================================================
# PART 1: THE COVER. Multi-cell sub-patterns graded by edge-count & b1.
#   For each b1 class, collect the set of distinct rho values across MANY patterns
#   of that class. If rho is a function of b1 alone -> one value per class (collapse).
#   If patterns of same b1 carry different rho -> spread.
# ============================================================
print("\n" + "="*70)
print("PART 1: rate vs (edge-count, b1) for multi-cell sub-patterns")
print("="*70)

rng = np.random.default_rng(0)

# Enumerate CONNECTED sub-patterns by growing edge sets randomly, then bin by (E, b1).
# We focus on patterns that actually carry NB cycles (rho>0), grouped by b1.
from collections import defaultdict
by_b1 = defaultdict(list)      # b1 -> list of rho
by_E_b1 = defaultdict(list)    # (E,b1) -> list of rho

E_all = edges
nE = len(E_all)

# Random connected-ish samples: random subsets of increasing size; keep those with a cycle.
samples = 0
target = 4000
for _ in range(40000):
    if samples >= target: break
    size = rng.integers(6, 30)
    idx = rng.choice(nE, size=size, replace=False)
    sub = [E_all[i] for i in idx]
    rho,b1,C,V = nb_radius_and_b1(sub, E_all)
    if b1 <= 0:   # forest, no NB cycle
        continue
    by_b1[b1].append(round(rho,6))
    by_E_b1[(len(sub),b1)].append(round(rho,6))
    samples += 1

print(f"  collected {samples} cyclic sub-patterns.\n")
print(f"  {'b1':>4} {'#patterns':>10} {'#distinct rho':>14} {'min rho':>9} {'max rho':>9} {'spread':>8}")
for b1 in sorted(by_b1):
    vals = by_b1[b1]
    dv = sorted(set(vals))
    print(f"  {b1:>4} {len(vals):>10} {len(dv):>14} {min(vals):>9.4f} {max(vals):>9.4f} {max(vals)-min(vals):>8.4f}")

print("\n  Same (edge-count E, b1) classes -- do equal-(E,b1) patterns share one rho (collapse)")
print("  or carry distinct rho (spread)?  Showing classes with >=8 samples:")
print(f"  {'(E,b1)':>10} {'#':>5} {'#distinct rho':>14} {'min':>8} {'max':>8} {'spread':>8}")
shown=0
for key in sorted(by_E_b1, key=lambda k:(k[1],k[0])):
    vals = by_E_b1[key]
    if len(vals) < 8: continue
    dv = sorted(set(vals))
    print(f"  {str(key):>10} {len(vals):>5} {len(dv):>14} {min(vals):>8.4f} {max(vals):>8.4f} {max(vals)-min(vals):>8.4f}")
    shown+=1
    if shown>=20: break

# ============================================================
# PART 2: THE CLOSED-WALK / IHARA-BASS |h| SPECTRUM.
#   Read the rate the other way: the modulus |h| over the object's NB closed walks.
#   On the FULL cover the spectrum of the Hashimoto B has eigenvalues h; the closed-walk
#   recurrence rates are |h|. The Ihara-Bass relation h^2 - lam h + (k-1) = 0 says:
#   complex roots (|lam|<2sqrt(q)) ALL have |h|^2 = q = k-1 (the Ramanujan shell, MODULUS LOCKED);
#   real roots only for |lam|>=2sqrt(q) (the Perron/trivial band).
#   Compute the actual |h| distribution over the full Bloch torus + finite cover.
# ============================================================
print("\n" + "="*70)
print("PART 2: closed-walk |h| spectrum (Ihara-Bass) over the object")
print("="*70)

q = 2  # k-1

# 2a) finite (Z_2)^3 cover: full Hashimoto spectrum
def full_hashimoto(edges):
    darts=[]
    for (a,b) in edges:
        darts.append((a,b)); darts.append((b,a))
    nd=len(darts)
    out_of=defaultdict(list)
    for idx,(t,h) in enumerate(darts): out_of[t].append(idx)
    B=np.zeros((nd,nd))
    for x,(tx,hx) in enumerate(darts):
        for y in out_of[hx]:
            ty,hy=darts[y]
            if not (hy==tx): B[x,y]=1.0
    return B
B=full_hashimoto(edges)
ev=np.linalg.eigvals(B)
mods=np.abs(ev)
# classify: trivial (real-root band, |h|=q or 1) vs Ramanujan shell |h|=sqrt(q)
ram=mods[np.abs(mods-np.sqrt(q))<1e-6]
print(f"  (Z_2)^3 cover Hashimoto: {len(ev)} eigenvalues h.")
print(f"    |h| distribution (rounded): {sorted(set(np.round(mods,4)))}")
print(f"    on Ramanujan shell |h|=sqrt2={np.sqrt(q):.4f}: {len(ram)} of {len(ev)} eigenvalues ({100*len(ram)/len(ev):.0f}%)")
# the rest:
nonram = sorted(set(np.round(mods[np.abs(mods-np.sqrt(q))>=1e-6],4)))
print(f"    off-shell |h| values: {nonram}  (the trivial/Perron band: |h|=q={q} and |h|=1 and 0)")

# 2b) Bloch torus of the infinite srs: |h| over many k via Ihara-Bass roots of A(k)
import importlib.util, os
spec=importlib.util.spec_from_file_location("srs", os.path.join(os.path.dirname(__file__),"..","dirac_srs_mdl","srs.py"))
srs=importlib.util.module_from_spec(spec); spec.loader.exec_module(srs)
ks=np.random.default_rng(1).random((4000,3))
allmod=[]; offshell=[]
for k in ks:
    lam=np.linalg.eigvalsh(srs.adjacency(k))
    for l in lam:
        for h in np.roots([1,-l,q]):   # h^2 - l h + q = 0
            allmod.append(abs(h))
            if abs(abs(h)-np.sqrt(q))>=1e-6: offshell.append((round(l,4),round(abs(h),4)))
allmod=np.array(allmod)
on=np.sum(np.abs(allmod-np.sqrt(q))<1e-6)
print(f"\n  infinite srs Bloch torus ({len(ks)} random k, 4 bands, 2 roots each):")
print(f"    fraction of closed-walk roots with |h| EXACTLY on Ramanujan shell sqrt2: {on}/{len(allmod)} = {100*on/len(allmod):.1f}%")
print(f"    modulus range of the ON-SHELL part: [{allmod[np.abs(allmod-np.sqrt(q))<1e-6].min():.6f}, {allmod[np.abs(allmod-np.sqrt(q))<1e-6].max():.6f}]  (locked)")
offmods=np.array([m for (l,m) in offshell])
if len(offmods):
    print(f"    OFF-shell roots (real-root band, |lam|>=2sqrt2): |h| in [{offmods.min():.4f},{offmods.max():.4f}], product of the pair = q (so geometric mean still sqrt2)")

# ============================================================
# PART 3: WHERE DOES THE SPREAD LIVE -- modulus or phase?
#   Take the Ramanujan-shell roots h (|h|=sqrt2 locked) over the Bloch torus and look at arg(h).
#   If modulus is locked but arg spreads -> the variation is PURELY in the phase.
# ============================================================
print("\n" + "="*70)
print("PART 3: modulus locked vs phase spread (Ramanujan-shell roots over Bloch torus)")
print("="*70)
phases=[]; modspread=[]
for k in ks:
    lam=np.linalg.eigvalsh(srs.adjacency(k))
    for l in lam:
        for h in np.roots([1,-l,q]):
            if abs(abs(h)-np.sqrt(q))<1e-6:
                phases.append(np.angle(h)); modspread.append(abs(h))
phases=np.array(phases); modspread=np.array(modspread)
print(f"  on-shell roots: {len(phases)}")
print(f"    MODULUS |h|: min={modspread.min():.8f} max={modspread.max():.8f}  -> std={modspread.std():.2e} (LOCKED)")
print(f"    PHASE arg(h): min={phases.min():.4f} max={phases.max():.4f} rad  -> std={phases.std():.4f} (SPREADS)")
print(f"    phase covers a continuous interval; modulus is a single point.")
print(f"  CONCLUSION: the recurrence-rate MODULUS does NOT spread; the variation lives in the PHASE.")

# ============================================================
# PART 4: VERDICT support -- the bounded discrete sub-pattern rates.
#   The PART-1 spread (rho in [1, ~1.21..1.42]) is the finite-truncation Perron radius of small
#   sub-networks, NOT a shell modulus. As b1->large / pattern -> the full shell, rho -> sqrt2 (and
#   the full cover -> 2 = Perron). Characterize the discrete rate set that actually appears.
# ============================================================
print("\n" + "="*70)
print("PART 4: the discrete sub-pattern rate set (small-pattern Perron radii)")
print("="*70)
allrho=sorted(set(r for L in by_b1.values() for r in L))
print(f"  distinct sub-pattern rates rho that appeared: {len(allrho)} values in [{min(allrho):.4f},{max(allrho):.4f}]")
print(f"  smallest few: {allrho[:8]}")
print(f"  largest few:  {allrho[-8:]}")
print(f"  Note rho=1.0 is the UNIQUE rate of EVERY single-cycle (b1=1) pattern (a bare NB loop:")
print(f"  one orbit, period = its length, Perron radius exactly 1 regardless of length or shape).")
print(f"  Multi-cycle patterns interpolate up toward the shell sqrt2={np.sqrt(2):.4f} / Perron 2.")

# Are equal-b1 sub-pattern rates a SPREAD or do they collapse? Quantify b1=2 explicitly.
v2=sorted(set(by_b1[2]))
print(f"\n  b1=2 distinct rates ({len(v2)}): they form a near-CONTINUUM in [1.0, {max(v2):.4f}],")
print(f"    bounded above by the two-loop coupling max (figure-eight / theta graph), NOT a clean shell.")
print(f"    => same-b1 patterns DO carry different rates, but as a bounded continuum, not discrete masses.")

# ============================================================
# PART 5: SAME-CLOCK / FORCED-vs-CHOICE
# ============================================================
print("\n" + "="*70)
print("PART 5: same-clock check; forced vs choice")
print("="*70)
print("  SAME CLOCK: rho(B_S) and |h| are BOTH read from the one operator B (Hashimoto = the geodesic")
print("    NB-walk generator = the SAME object whose Ihara zeta = the Ruelle zeta). One flow, one clock.")
print("  FORCED:  q=k-1=2 (so shell |h|=sqrt2) is forced by 3-regularity (MDL->K_4). The modulus-lock")
print("    (Ihara-Bass discriminant: complex roots => |h|^2=product=q) is forced, k-independent.")
print("  FORCED:  rho=1 for every b1=1 pattern is forced (a single NB cycle's transfer matrix is a")
print("    permutation, Perron 1). The sub-pattern continuum [1, sqrt2) is forced by truncation.")
print("  CHOICE:  WHICH edges are occupied (the pattern S) is a choice/input, not forced by {D,srs,MDL}.")
print("  FLAG: nothing used beyond the three sealed dirs. srs.py imported only for adjacency(k).")

# ============================================================
# PART 6: CRUX -- same (E,b1) non-isomorphic patterns: distinct rates, but discrete or continuum?
#   Take all b1=2 patterns at a fixed small edge-count; list the distinct rho and whether they
#   correspond to combinatorial sub-graph TYPES (theta vs dumbbell vs ...) -> discrete -- or vary
#   continuously with embedding -> continuum.
# ============================================================
print("\n" + "="*70)
print("PART 6: crux -- are equal-(E,b1) rates a few combinatorial values or a continuum?")
print("="*70)
# Enumerate ALL minimal b1=2 connected subgraphs: take pairs of short cycles sharing structure.
# Simpler: classify by the SHAPE invariant (degree sequence of the b1=2 core). For b1=2 there are
# exactly two homeomorphism types: THETA (two deg-3 verts, 3 paths) and DUMBBELL/FIGURE-8.
# rho of a NB walk depends only on the metric (path lengths). Show rho is a function of those lengths.
def theta_nb_radius(a,b,c):
    """theta graph: two hubs joined by 3 internal paths of lengths a,b,c edges. NB spectral radius."""
    # build explicit small graph
    n=0; E=[]; 
    def chain(L, start, end):
        nonlocal n
        prev=start
        for _ in range(L-1):
            n+=1; node=('m',n); E.append((prev,node)); prev=node
        E.append((prev,end))
    H1=('h',1); H2=('h',2)
    chain(a,H1,H2); chain(b,H1,H2); chain(c,H1,H2)
    rho,b1,C,V=nb_radius_and_b1(E,E)
    return rho,b1
print("  THETA graphs (two hubs, 3 connecting paths of lengths a,b,c) -- rho varies CONTINUOUSLY")
print("  with the path lengths (a genuine arithmetic function, not a fixed shell):")
for (a,b,c) in [(1,1,1),(2,1,1),(2,2,1),(3,2,1),(2,2,2),(3,3,2),(4,3,2),(5,5,5),(10,10,10)]:
    rho,b1=theta_nb_radius(a,b,c)
    print(f"    theta({a},{b},{c}): b1={b1}, rho={rho:.5f}")
print("  => rho -> 1 as paths lengthen (sparser = weaker recurrence), and is LARGEST for the densest")
print("     theta(1,1,1) (= the doubled triangle).  Distinct path-length triples give distinct rho:")
print("     a real arithmetic SPREAD over the pattern, but governed by lengths, bounded by the shell.")
