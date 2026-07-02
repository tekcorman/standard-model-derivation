"""
SCRATCH (walled): non-backtracking recurrence per bond-combination.
For each subset that CONTAINS at least one cycle, compute the NB recurrence of the occupied
sub-network: girth (shortest closed NB walk), N_m=Tr B^m, NB spectrum |h|, and the Ihara-Bass
shell. Compare across combinations of the same size and different sizes.
Also: at k=0 (cell quotient) and on the cover (does the cover phase / homology change |h|?).
"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs
from collections import defaultdict

EDGES = srs.EDGES; NE = len(EDGES); NV = srs.NV

def occ_info(sub):
    verts = set()
    for e in sub:
        i, j, _ = EDGES[e]; verts.add(i); verts.add(j)
    deg = {v: 0 for v in verts}
    for e in sub:
        i, j, _ = EDGES[e]; deg[i] += 1; deg[j] += 1
    parent = {v: v for v in verts}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in sub:
        i, j, _ = EDGES[e]; parent[find(i)] = find(j)
    comps = len({find(v) for v in verts})
    b1 = len(sub) - len(verts) + comps
    bdy = tuple(v for v in verts if deg[v] % 2 == 1)
    degseq = tuple(sorted(deg.values()))
    return b1, bdy, degseq, comps, len(verts)

def nb(sub, k=(0,0,0)):
    darts = []
    for e in sub:
        i, j, v = EDGES[e]
        darts.append((i, j, np.array(v))); darts.append((j, i, -np.array(v)))
    n = len(darts)
    if n == 0: return None
    B = np.zeros((n, n), complex); kk = np.asarray(k, float)
    for b,(tb,hb,vb) in enumerate(darts):
        for a,(ta,ha,va) in enumerate(darts):
            if ha==tb and not (hb==ta and np.array_equal(vb,-va)):
                B[b,a]=np.exp(2j*np.pi*(kk@vb))
    ev = np.linalg.eigvals(B)
    nz = sorted((round(abs(l),4) for l in ev if abs(l)>1e-7))
    Nm = [int(round(np.trace(np.linalg.matrix_power(B,m)).real)) for m in range(1,9)]
    girth = next((m for m in range(1,9) if Nm[m-1]!=0), None)
    # |h|^2 multiset of the nonzero NB eigenvalues
    hh2 = sorted(round(abs(l)**2,3) for l in ev if abs(l)>1e-7)
    specrad = max((abs(l) for l in ev), default=0.0)
    return dict(n=n, girth=girth, Nm=Nm, nz=nz, hh2=hh2, specrad=round(specrad,4))

# Enumerate subsets WITH a cycle, group by size, report recurrence signature
print("="*94)
print("NON-BACKTRACKING RECURRENCE PER BOND-COMBINATION (cell quotient k=0)")
print("="*94)
sig_by_size = defaultdict(lambda: defaultdict(list))
for r in range(1, NE+1):
    for sub in itertools.combinations(range(NE), r):
        b1, bdy, degseq, comps, V = occ_info(sub)
        if b1 < 1:  # acyclic (tree/forest) -> no closed walks, NB nilpotent
            continue
        rec = nb(sub)
        # signature: what distinguishes recurrence = (girth, sorted Nm up to 6, spectral radius, |h|^2 multiset)
        sig = (b1, rec['girth'], tuple(rec['Nm'][:6]), rec['specrad'], tuple(rec['hh2']))
        sig_by_size[r][sig].append((sub, degseq, bdy))

for r in range(1, NE+1):
    if not sig_by_size[r]: continue
    print(f"\n----- SIZE {r} (combinations containing >=1 cycle) -----")
    print(f"  distinct recurrence signatures at this size: {len(sig_by_size[r])}")
    for sig, members in sig_by_size[r].items():
        b1, girth, Nm6, specrad, hh2 = sig
        ex = members[0]
        edges_str = "+".join(f"e{e}" for e in ex[0])
        print(f"   * b1={b1} girth={girth} specrad={specrad}  N_m(1..6)={list(Nm6)}")
        print(f"       |h|^2 multiset={list(hh2)}   ({len(members)} combos, e.g. {{{edges_str}}}, degseq {ex[1]})")

# Cross-size: do different sizes give different recurrence? Tabulate the unique (girth, specrad, |h|^2)
print("\n" + "="*94)
print("CROSS-SIZE: unique recurrence types over the whole ladder")
print("="*94)
seen = {}
for r in range(1, NE+1):
    for sig, members in sig_by_size[r].items():
        b1, girth, Nm6, specrad, hh2 = sig
        key = (b1, girth, specrad, hh2)
        seen.setdefault(key, []).append((r, len(members)))
for key, occ in sorted(seen.items()):
    b1, girth, specrad, hh2 = key
    sizes = ", ".join(f"size{r}x{c}" for r,c in occ)
    print(f"  b1={b1} girth={girth} specrad={specrad} |h|^2={list(hh2)}  at [{sizes}]")

# THE COVER: does the Z^3 cover phase change the recurrence of a single cell-cycle?
print("\n" + "="*94)
print("COVER CHECK: NB spectrum of one triangle cycle as Bloch phase varies (does |h| move?)")
print("="*94)
tri = (0,1,3)  # e0+e1+e3 = a triangle carrying homology (cover-rank 1)
for k in [(0,0,0),(.25,0,0),(.5,0,0),(.2,.3,.1)]:
    rec = nb(tri, k)
    print(f"  k={k}: girth={rec['girth']} |h|={rec['nz']} specrad={rec['specrad']}")
print("  (A single 3-cycle's NB operator is a 6x6; its closed walks wind the triangle. Phase = winding.)")

# The FULL cell (size 6) NB = the established srs Hashimoto. Confirm.
print("\n" + "="*94)
print("FULL CELL (size 6) = the established Hashimoto B(k):")
rec = nb(tuple(range(6)), (0,0,0))
print(f"  girth={rec['girth']} (expect cell-quotient first nonzero N_m), N_m={rec['Nm']}")
print(f"  |h|^2 multiset={rec['hh2']}  specrad={rec['specrad']} (expect Perron 2, Ramanujan shell |h|^2=2)")
