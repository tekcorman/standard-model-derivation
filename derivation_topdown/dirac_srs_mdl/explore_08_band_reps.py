"""
explore_08 — BAND STRUCTURE and A4 REPRESENTATION CONTENT of the K4-cover.

Pure math, walled off (imports only srs + numpy/itertools).  We walk the
Brillouin-zone path  Gamma=(0,0,0) -> P=(1/4,1/4,1/4) -> H=(1/2,1/2,1/2) -> Gamma
and report, for BOTH the Bloch adjacency A(k) and the non-backtracking B(k):

  (1) the bands (eigenvalues) and their DEGENERACY multiplicities at Gamma, P, H;
  (2) the A4-irrep content {1, 1', 1'', 3} of each eigenspace at Gamma
      (where the full rotation group A4 = even perms of {0,1,2,3} is a symmetry);
  (3) the compatibility / splitting of each Gamma multiplet as k -> P and k -> H.

A4 facts used:  |A4|=12; irreps 1, 1', 1'' (the three C3 characters) and 3 (standard).
chi_3(g) = 3 (id), -1 (double transposition), 0 (3-cycle).
Orthogonality:  mult of an irrep R in a rep with character chi is  (1/12) sum_g chi(g) conj(chi_R(g)).
"""
import numpy as np, itertools, srs

np.set_printoptions(suppress=True)

# ====================================================================== #
#  A4 = the 12 even permutations of {0,1,2,3}  (rotation group of K4).    #
# ====================================================================== #
def parity(p):
    seen = [False]*4; par = 0
    for i in range(4):
        if not seen[i]:
            j = i; c = 0
            while not seen[j]: seen[j] = True; j = p[j]; c += 1
            par += c-1
    return par % 2
A4 = [p for p in itertools.permutations(range(4)) if parity(p) == 0]

def cls(p):                       # conjugacy class of g in A4
    fx = sum(1 for i in range(4) if p[i] == i)
    if fx == 4: return 'e'        # identity
    if fx == 0: return 'd'        # double transposition  (3 of them)
    return '3'                    # 3-cycle  (8 of them, two classes 3+ / 3-)

def chi3(p):                      # character of the standard 3-irrep
    return {'e': 3, 'd': -1, '3': 0}[cls(p)]

# the three 1-dim irreps are the characters of A4/V4 = C3.  A 3-cycle gets
# 1 (triv), omega (1'), omega^2 (1'').  We only need to split 1' vs 1'' when
# a 1-dim multiplet is complex; for the real adjacency spaces 1'+1'' pair up.
om = np.exp(2j*np.pi/3)
# The 8 three-cycles split into TWO A4-conjugacy classes (4 each); 1' assigns
# omega to one class and omega^2 to the other.  Build the class of a chosen
# representative by actual conjugation h g h^{-1} (verified: <1',1'>=1, <1',1>=0).
def _conj(h, p):
    hinv = tuple(int(x) for x in np.argsort(h))
    return tuple(h[p[hinv[i]]] for i in range(4))
REP3 = next(p for p in A4 if cls(p) == '3')
_CLASS_A = {_conj(h, REP3) for h in A4}          # the omega-class
def chi1p(p):                     # character of the irrep 1'
    if cls(p) != '3': return 1.0  # trivial on e and on the double-transpositions
    return om if p in _CLASS_A else om.conjugate()

def mult_of(chs, chi_R):
    """multiplicity of irrep R (character chi_R: A4->C) in rep with characters chs."""
    return sum(c*np.conjugate(chi_R(p)) for c, p in zip(chs, A4))/12.0

# ---- A4 acting on vertices (4), on darts (12) ----
def Pv(p):
    M = np.zeros((4, 4))
    for i in range(4): M[p[i], i] = 1.0
    return M
DUV = [(d[0], d[1]) for d in srs._darts()]
def Pd(p):
    M = np.zeros((12, 12))
    for d, (i, j) in enumerate(DUV):
        t = (p[i], p[j])
        for f, (a, b) in enumerate(DUV):
            if (a, b) == t: M[f, d] = 1.0; break
    return M

# ====================================================================== #
#  helpers                                                               #
# ====================================================================== #
def cluster(vals, tol=1e-6):
    """group (sorted) scalar values into [representative, multiplicity, members]."""
    out = []
    for v in vals:
        for cl in out:
            if abs(v - cl[0]) < tol: cl[1] += 1; cl[2].append(v); break
        else:
            out.append([v, 1, [v]])
    return out

def proj_onto_eigenspace(M_eigvecs, idx):
    """orthonormal columns spanning a chosen set of eigenvectors -> projector chars are
    computed by restricting the group action; we return the column matrix Q (12 x m)."""
    return M_eigvecs[:, idx]

def irrep_content(Q, group_mats):
    """Given orthonormal columns Q spanning a g-invariant subspace and the list of
    group representation matrices, return (m1, m1p, m1pp, m3) by characters."""
    chs = []
    for g, P in zip(A4, group_mats):
        # character on the subspace = Tr( Q^dagger P Q )
        chs.append(np.trace(Q.conj().T @ P @ Q))
    chs = np.array(chs)
    m1   = mult_of(chs, lambda p: 1.0).real
    m3   = mult_of(chs, chi3).real
    m1p  = mult_of(chs, chi1p)
    m1pp = mult_of(chs, lambda p: np.conjugate(chi1p(p)))
    return m1, m1p.real, m1pp.real, m3

def fmt_irreps(m1, m1p, m1pp, m3):
    parts = []
    R = lambda x: int(round(x))
    if R(m1):  parts.append(f"{R(m1)}·1" if R(m1) > 1 else "1")
    if R(m1p): parts.append(f"{R(m1p)}·1'" if R(m1p) > 1 else "1'")
    if R(m1pp):parts.append(f"{R(m1pp)}·1''" if R(m1pp) > 1 else "1''")
    if R(m3):  parts.append(f"{R(m3)}·3" if R(m3) > 1 else "3")
    return " + ".join(parts) if parts else "(none)"

# ====================================================================== #
#  (1) BAND STRUCTURE along Gamma -> P -> H -> Gamma                     #
# ====================================================================== #
G = np.array([0., 0., 0.]); P = np.array([.25, .25, .25]); H = np.array([.5, .5, .5])
SPECIAL = [('Gamma', G), ('P', P), ('H', H)]

def path(p0, p1, n):              # n samples, endpoints included
    return [p0 + (p1-p0)*t for t in np.linspace(0, 1, n)]

print("="*72)
print(" (1) BAND STRUCTURE  on the path  Gamma -> P -> H -> Gamma")
print("="*72)
print("\n  Bloch adjacency A(k): eigenvalues (sampled along each segment)")
for (na, a), (nb, b) in [(('Gamma', G), ('P', P)), (('P', P), ('H', H)), (('H', H), ('Gamma', G))]:
    print(f"\n   segment {na} -> {nb}:")
    for k in path(a, b, 11):
        ev = np.sort(np.linalg.eigvalsh(srs.adjacency(k)))
        print(f"     k={np.round(k,3)}   A-eigs = {np.round(ev,4)}")

print("\n  Non-backtracking B(k): |h|^2 of the 12 eigenvalues (sampled)")
for (na, a), (nb, b) in [(('Gamma', G), ('P', P)), (('P', P), ('H', H)), (('H', H), ('Gamma', G))]:
    print(f"\n   segment {na} -> {nb}:")
    for k in path(a, b, 11):
        h2 = np.sort(np.abs(np.linalg.eigvals(srs.hashimoto(k)))**2)
        print(f"     k={np.round(k,3)}   |h|^2 = {np.round(h2,3)}")

print("\n  DEGENERACY MULTIPLICITIES at the special points:")
for nm, k in SPECIAL:
    evA = np.sort(np.linalg.eigvalsh(srs.adjacency(k)))
    clA = cluster(evA)
    h2  = np.sort(np.abs(np.linalg.eigvals(srs.hashimoto(k)))**2)
    clB = cluster(h2)
    sA = "  ".join(f"{cl[0]:+.3f}(x{cl[1]})" for cl in clA)
    sB = "  ".join(f"{cl[0]:.3f}(x{cl[1]})" for cl in clB)
    print(f"   {nm:6s}:  A-eigs  {sA}")
    print(f"           |h|^2   {sB}")

# ====================================================================== #
#  (2) A4-IRREP CONTENT at Gamma (full A4 symmetry there)                #
# ====================================================================== #
print("\n" + "="*72)
print(" (2) A4-IRREP CONTENT at Gamma   (irreps 1, 1', 1'', 3)")
print("="*72)

# sanity: total content of the two carrier spaces
Vmats = [Pv(p) for p in A4]; Dmats = [Pd(p) for p in A4]
chV = np.array([np.trace(M) for M in Vmats]); chD = np.array([np.trace(M) for M in Dmats])
print("\n  carrier-space decompositions (independent of k):")
print(f"   vertices C0 (dim 4):  " +
      fmt_irreps(mult_of(chV, lambda p:1.).real, mult_of(chV, chi1p).real,
                 mult_of(chV, lambda p:np.conjugate(chi1p(p))).real, mult_of(chV, chi3).real))
print(f"   darts       (dim 12): " +
      fmt_irreps(mult_of(chD, lambda p:1.).real, mult_of(chD, chi1p).real,
                 mult_of(chD, lambda p:np.conjugate(chi1p(p))).real, mult_of(chD, chi3).real))

# --- A(Gamma): the adjacency is real symmetric; eigenvectors are real ---
print("\n  --- adjacency A(Gamma) ---")
A0 = srs.adjacency(G).real
wA, UA = np.linalg.eigh(A0)
for cl in sorted(cluster(np.round(wA, 9)), key=lambda c: -c[0]):
    val = cl[0]; idx = [i for i in range(len(wA)) if abs(wA[i]-val) < 1e-6]
    Q = UA[:, idx]
    m = irrep_content(Q.astype(complex), Vmats)
    print(f"   eigval {val:+.3f} (mult {len(idx)}):  A4-content = {fmt_irreps(*m)}")

# --- B(Gamma): non-backtracking, generally non-normal. Use generalized
#     eigen-decomposition; group eigenvectors by |h|^2 shell and read content.
print("\n  --- non-backtracking B(Gamma) ---")
B0 = srs.hashimoto(G)
wB, UB = np.linalg.eig(B0)
h2 = np.abs(wB)**2
# B commutes with the dart-permutation rep, so eigenspaces are A4-invariant.
# Cluster by |h|^2 (the physical Ramanujan/tree shells); within a shell the
# distinct eigenvalues may differ by phase but share the |h|^2 label.
shells = cluster(np.round(h2, 6))
for cl in sorted(shells, key=lambda c: -c[0]):
    val = cl[0]; idx = [i for i in range(len(wB)) if abs(h2[i]-val) < 1e-6]
    raw = UB[:, idx]
    # orthonormalize the (possibly non-orthogonal) eigenvectors spanning this shell
    Q, _ = np.linalg.qr(raw)
    Q = Q[:, :len(idx)]
    m = irrep_content(Q, Dmats)
    # also list the distinct h-values in the shell
    hs = sorted({complex(np.round(wB[i].real, 3), np.round(wB[i].imag, 3)) for i in idx},
                key=lambda z: (round(z.real, 3), round(z.imag, 3)))
    hs_s = ", ".join(f"{z.real:+.2f}{z.imag:+.2f}i" for z in hs)
    print(f"   |h|^2={val:.3f} (mult {len(idx)}):  A4-content = {fmt_irreps(*m)}    h in {{{hs_s}}}")

# ====================================================================== #
#  (3) CONNECTIVITY / COMPATIBILITY  Gamma -> P  and  Gamma -> H         #
# ====================================================================== #
print("\n" + "="*72)
print(" (3) CONNECTIVITY: how Gamma multiplets SPLIT toward P and H")
print("="*72)

def trace_split(op, p0, p1, label):
    """Follow each Gamma cluster a small step toward p1 and report how the
    (sorted) values split.  op='A' (eigvalsh) or 'B' (|h|^2).
    We use a small finite step (resolved adaptively) since the splitting of B's
    tree modes is second-order in |k| near Gamma (degenerate perturbation)."""
    if op == 'A':
        get = lambda k: np.sort(np.linalg.eigvalsh(srs.adjacency(k)))
        eps = 1e-3                 # A splits linearly: a tiny step resolves it
    else:
        get = lambda k: np.sort(np.abs(np.linalg.eigvals(srs.hashimoto(k)))**2)
        eps = 0.05                 # B's |h|^2 moves at O(eps^2): need a real step
    dst = 'P' if (p1 == P).all() else 'H'
    print(f"\n   {label}:  Gamma  --(step {eps})-->  {dst}")
    cl0 = cluster(np.round(get(p0), 6))
    vals_eps = get(p0 + (p1-p0)*eps)
    used = [False]*len(vals_eps)
    for cl in sorted(cl0, key=lambda c: -c[0]):
        v0 = cl[0]; mult = cl[1]
        order = sorted(range(len(vals_eps)), key=lambda i: abs(vals_eps[i]-v0))
        take = []
        for i in order:
            if not used[i]:
                take.append(vals_eps[i]); used[i] = True
                if len(take) == mult: break
        sub = cluster(np.round(np.sort(take), 4))
        split = " + ".join(f"{c[1]}" for c in sub)
        split_vals = ", ".join(f"{c[0]:+.4f}(x{c[1]})" for c in sub)
        arrow = f"{mult} (unsplit)" if len(sub) == 1 else f"{mult} -> {split}"
        print(f"     {v0:+.4f}(x{mult})  ->  {arrow}   [{split_vals}]")

for label, fn in [("adjacency A", 'A'), ("non-backtracking B (|h|^2)", 'B')]:
    trace_split(fn, G, P, label)
    trace_split(fn, G, H, label)

# --- B: the structurally important fact is the ENDPOINT shell membership, not
#     the local split.  Report how the Gamma |h|^2-shells map onto the P/H shells.
print("\n   B(k) shell EVOLUTION (endpoint membership, the Ramanujan story):")
def shells_of(k):
    h2 = np.abs(np.linalg.eigvals(srs.hashimoto(k)))**2
    return {c[0]: c[1] for c in cluster(np.round(np.sort(h2), 4))}
sG, sP, sH = shells_of(G), shells_of(P), shells_of(H)
def shellstr(s): return "  ".join(f"|h|^2={v:.0f}:x{m}" for v, m in sorted(s.items(), key=lambda kv:-kv[0]))
print(f"     Gamma : {shellstr(sG)}")
print(f"     P     : {shellstr(sP)}")
print(f"     H     : {shellstr(sH)}")
print("     => Gamma & H carry a TREE pair {|h|^2=4 (x1)} + one of the {|h|^2=1} fivefold;")
print("        moving to P these two NON-Ramanujan (tree) modes collapse ONTO the Ramanujan")
print("        shell |h|^2=k-1=2, giving the maximal 8-fold Ramanujan degeneracy at P.")
print("        The genuine 6-fold Ramanujan shell (= 2 copies of the A4 3-irrep) is RIGID")
print("        all along the path; only the tree modes move.")

# also show the FULL evolution of multiplicities at the three endpoints side by side
print("\n   summary of degeneracy patterns:")
for op, getter in [('A-eigs', lambda k: np.linalg.eigvalsh(srs.adjacency(k)).real),
                   ('|h|^2 ', lambda k: np.abs(np.linalg.eigvals(srs.hashimoto(k)))**2)]:
    print(f"     {op}:  " + "   |   ".join(
        f"{nm}: " + ",".join(f"{c[0]:+.2f}x{c[1]}" for c in
                              sorted(cluster(np.round(np.sort(getter(k)), 4)), key=lambda c:-c[0]))
        for nm, k in SPECIAL))

print("\n[done]")
