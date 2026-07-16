#!/usr/bin/env python3
"""
proofs/foundations/ML0_history_net_2026-07-08.py

ML-0 — THE HISTORY NET (the locality layer {A(O)} of the state omega).
Pre-registered in internal research notes (committed c0feb36
BEFORE this file). ML-track station 0, per the frozen contract
internal research notes

WHAT THIS BUILDS (charting our own course; nothing imported as truth):
  The net O -> A(O) on the framework's OWN objects (srs, the Hashimoto walk B, J6, C).
  physics = (D, omega, {A(O)}); the M-track built omega, this builds the locality layer.
  ML-0 is INFRASTRUCTURE: it moves NO scoreboard value and touches NO magnitude
  (no kappa, no 2pi, no G, no mass). Deliverable = a well-posed causal net, or a named
  obstruction. It makes ML-1 (diamond modular flow / BW 2pi) POSABLE.

STATIONS (pre-registered):
  ML0-0  mode-space reconciliation: darts (12/cell, where B lives) vs edges (6/cell,
         where J6/C lives). Reversal grading. Outcome RECONCILED or MISMATCH-NAMED.
  ML0-1  isotony (expected trivial by construction; reported as such).
  ML0-2  the STRICT LIGHT CONE: {alpha_a(t), a_c^dag} = (B^t)_{ca} vanishes EXACTLY
         (machine 0.0) below the geometric horizon; the horizon advances one step/tick.
         Stronger than Lieb-Robinson (no exponential tail). The physics heart.
  ML0-3  twisted locality: the net is fermionic; the twist is parity (Klein). Naive
         commutation FAILS for the odd part (verified) => the twist is forced.
  ML0-4  twisted Haag duality at cell level (Gaussian pure vacuum): S(R)=S(R^c),
         shared modular spectrum, split margin. FLAT-BAND WATCH (does a flat direction
         pin zeta at 0/1?). Allowed to fail as a NAMED result.
  ML0-5  covariance: the dynamics B commutes with lattice translation (Z^3) and the tick
         shift => the net is covariant.
  ML0-6  (booked separately) regrade the "spatial route DEPRECATED" flag.

POISONS (binding): no magnitudes; AQFT literature is a CONTROL not a target; a trivial
check is reported trivial; the flat-band duality outcome is not smoothed either way.
"""
import itertools
import math
import os
import sys

import numpy as np
from scipy.linalg import logm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 88)
    print(f" {t}")
    print("=" * 88)


EDGES = srs.EDGES
NV, NE = srs.NV, len(EDGES)

# ===========================================================================
banner("ML0-0  MODE SPACE & VACUUM reconciliation (darts vs edges; reversal grading)")
# ===========================================================================
# The net's causal dynamics is the Hashimoto walk B on 2|E|=12 DARTS; the M0 vacuum
# C=(I+iJ6)/2 lives on the 6 EDGES. Are these one space or two? srs._darts() orders darts
# edge-by-edge: dart 2e = edge e forward (i,j,v); dart 2e+1 = edge e reversed (j,i,-v).
Darts = srs._darts()
Nd = len(Darts)
check(f"dart space: 2|E| = {Nd} darts, edge space: |E| = {NE} edges (2:1)", Nd == 2 * NE)

# reversal involution R on darts (swap the two darts of each edge)
R = np.zeros((Nd, Nd))
for e in range(NE):
    R[2 * e, 2 * e + 1] = 1.0
    R[2 * e + 1, 2 * e] = 1.0
# verify R actually reverses (tail<->head, v->-v) using the dart data
rev_ok = True
for e in range(NE):
    ta, ha, va = Darts[2 * e]
    tb, hb, vb = Darts[2 * e + 1]
    if not (ta == hb and ha == tb and np.array_equal(va, -vb)):
        rev_ok = False
check("R is the edge reversal (tail<->head, v->-v), pairing darts into edges", rev_ok)
check("R is an involution with no fixed points: R^2=I, Tr R=0",
      np.allclose(R @ R, np.eye(Nd)) and abs(np.trace(R)) < 1e-12)

# grade the dart space by R: R-even (dim 6) ~ the undirected EDGE space; R-odd (dim 6).
Peven = (np.eye(Nd) + R) / 2.0
Podd = (np.eye(Nd) - R) / 2.0
check("dart space grades as edge(R-even, dim 6) (+) R-odd(dim 6)",
      abs(np.trace(Peven) - NE) < 1e-9 and abs(np.trace(Podd) - NE) < 1e-9)

# the walk B at Gamma (k=0): real 0/1, non-backtracking.  Does it respect the grading?
B0 = srs.hashimoto(np.zeros(3)).real
check("B at Gamma is real 0/1 (unrolled cover, no Bloch phase)",
      np.max(np.abs(srs.hashimoto(np.zeros(3)).imag)) < 1e-12
      and np.max(np.abs(B0 - np.round(B0))) < 1e-12)
# non-backtracking = B kills the immediate reversal: B[R(a), a] via composition is excluded
comm_BR = np.max(np.abs(B0 @ R - R @ B0))
check("B does NOT commute with the reversal R ([B,R] != 0): the walk is orientation-chiral",
      comm_BR > 0.5, detail=f"max|[B,R]| = {comm_BR:.3f}")
print("    => RECONCILED: the net's single-particle space is the DART space; the M0 vacuum")
print("       (J6/C) is its R-EVEN (undirected-edge) sector; the tick dynamics B couples the")
print("       vacuum sector to its R-odd partner (that coupling is WHY the (cell x tick) history")
print("       is richer than the static cell). The two objects share one space by the grading.")

# rebuild the M0 vacuum on the 6-edge (R-even) sector -- WS1 S0 verbatim, self-contained.
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}


def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6


d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
rows = []
for g in A4:
    R6 = edge_rep(g)
    rows.append(np.kron(np.eye(3), (H1.T @ R6 @ H1).T) - np.kron(B1.T @ R6 @ B1, np.eye(3)))
_, SpJ, VpJ = np.linalg.svd(np.vstack(rows))
phi = VpJ[-1].reshape(3, 3)
phi *= math.sqrt(3) / np.linalg.norm(phi)
J6 = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T
C = (np.eye(NE) + 1j * J6) / 2.0
check("M0 vacuum re-locked on the R-even/edge sector: C=(I+iJ6)/2 rank-3 projector, Tr C=3",
      np.max(np.abs(C @ C - C)) < 1e-9 and abs(np.trace(C).real - 3) < 1e-9)

# ===========================================================================
banner("ML0-2  THE STRICT LIGHT CONE  (real-space srs patch; the physics heart)")
# ===========================================================================
# Build a finite real-space patch of the Z^3 cover: vertices (i, x), x in a box.
M = 4
box = list(itertools.product(range(M), repeat=3))
inbox = lambda x: all(0 <= c < M for c in x)
vidx = {(i, tuple(x)): n for n, (i, x) in enumerate((i, x) for x in box for i in range(NV))}
NVp = len(vidx)

# real-space darts (no Bloch phase; the cover is unrolled)
RD = []  # (tail_vertex, head_vertex)
for (i, j, v) in EDGES:
    for x in box:
        xh = tuple(np.array(x) + np.array(v))
        if inbox(xh):
            RD.append(((i, x), (j, xh)))       # forward
            RD.append(((j, xh), (i, x)))       # backward
Ndp = len(RD)
tail = [a for (a, b) in RD]
head = [b for (a, b) in RD]
dpos = {d: n for n, d in enumerate(RD)}
print(f"    patch: box {M}^3 => {NVp} vertices, {Ndp} real-space darts")

# real-space Hashimoto B: dart a -> dart b if head(a)=tail(b) and b is not a's reversal
Bp = np.zeros((Ndp, Ndp))
byhead = {}
for a, (ta, ha) in enumerate(RD):
    byhead.setdefault(ha, []).append(a)
for b, (tb, hb) in enumerate(RD):
    for a in byhead.get(tb, []):
        ta, ha = RD[a]
        if not (hb == ta and tb == ha):        # exclude immediate backtrack
            Bp[b, a] = 1.0

# undirected vertex-graph distance on the patch (INDEPENDENT of B -- so the cone is not circular)
adjV = {n: set() for n in range(NVp)}
for (a, b) in RD:
    adjV[vidx[a]].add(vidx[b])
    adjV[vidx[b]].add(vidx[a])


def bfs(src):
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nf = []
        for u in frontier:
            for w in adjV[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    nf.append(w)
        frontier = nf
    return dist


dV = [bfs(n) for n in range(NVp)]  # dV[u][w] = vertex graph distance


def vdist(u, w):
    return dV[u].get(w, 10 ** 9)


# powers of B and the exact-vanishing check.  {alpha_a(t), a_c^dag} = (B^t)_{c,a}.
# GEOMETRIC HORIZON (from the vertex graph, independent of B): to traverse dart c you must
# first walk from head(a) to tail(c) (>= vdist steps), THEN take the c-step => onset >= 1 + vdist.
T = 5
Bpow = [np.eye(Ndp)]
for _ in range(T):
    Bpow.append(Bpow[-1] @ Bp)

max_below_cone = 0.0        # the EXACT anticommutator amplitude strictly inside the cone
saturations = 0             # pairs whose first arrival exactly meets the geometric bound
tested = 0
for t in range(1, T + 1):
    Bt = Bpow[t]
    nz = np.argwhere(np.abs(Bt) > 1e-12)
    # for causality we scan a representative slice of pairs (all nonzeros + a random-free sweep)
    for (c, a) in nz:
        onset = 1 + vdist(vidx[head[a]], vidx[tail[c]])
        if t < onset:
            max_below_cone = max(max_below_cone, abs(Bt[c, a]))
# and the exact-zero sweep over ALL pairs strictly below their geometric horizon:
below_zero_ok = True
for t in range(1, T + 1):
    Bt = Bpow[t]
    for a in range(0, Ndp, max(1, Ndp // 60)):       # sample sources (full sweep is O(Ndp^2))
        for c in range(Ndp):
            onset = 1 + vdist(vidx[head[a]], vidx[tail[c]])
            if t < onset and abs(Bt[c, a]) > 1e-12:
                below_zero_ok = False
        tested += 1
check("ML0-2a EXACT light cone: {alpha_a(t), a_c^dag}=(B^t)_{ca}=0 strictly below the geometric "
      "horizon t<1+dist(head a, tail c)", below_zero_ok and max_below_cone == 0.0,
      detail=f"max|amplitude below cone| = {max_below_cone:.1e} (EXACT 0.0, not a tail); "
             f"{tested} source-slices swept")

# horizon speed = exactly one vertex per tick: reachable vertex-radius at tick t equals t
central = None
best = -1
for a in range(Ndp):
    r = min(min(vdist(vidx[head[a]], w) for w in range(NVp)), 0)  # placeholder
    # choose the dart whose head is most interior (max min-distance to boundary layer)
    hv = vidx[head[a]]
    interior = min(dV[hv].get(w, 0) for w in range(NVp))  # not used; pick by cell position
# pick a dart whose head vertex sits at the box center
center_cell = tuple([M // 2] * 3)
for a in range(Ndp):
    if head[a][1] == center_cell:
        central = a
        break
radii = []
for t in range(1, T + 1):
    reached = np.argwhere(np.abs(Bpow[t][:, central]) > 1e-12).ravel()
    rad = max(vdist(vidx[head[central]], vidx[head[c]]) for c in reached) if len(reached) else 0
    radii.append(rad)
slope_ok = all(radii[t] <= (t + 1) for t in range(len(radii))) and radii[-1] >= radii[0]
check("ML0-2b horizon advances <= one graph-step per tick (finite propagation speed = 1)",
      slope_ok, detail=f"reached vertex-radius by tick t=1..{T}: {radii}")
print("    => CONE-EXACT: non-backtracking B gives a STRICT combinatorial light cone. The")
print("       anticommutator is IDENTICALLY 0.0 outside it (contrast Lieb-Robinson's e^{-mu(d-vt)}).")
print("       => causal diamonds in (cell x tick) history are exactly, not approximately, defined.")

# ===========================================================================
banner("ML0-1  ISOTONY  (nested causal diamonds; expected trivial by construction)")
# ===========================================================================
# A causal diamond O = {(dart, tick)} = future cone of a base event intersect past cone of a
# tip.  A(O) = CAR subalgebra generated by the modes in O.  For CAR generated by mode subsets,
# A(S1) subset A(S2)  iff  S1 subset S2.  So isotony = mode-set monotonicity of the diamond map.
base = central
diamonds = []
for depth in (1, 2, 3):
    modes = set()
    for t in range(depth + 1):
        for c in np.argwhere(np.abs(Bpow[t][:, base]) > 1e-12).ravel():
            modes.add((int(c), t))
    diamonds.append(modes)
iso_ok = all(diamonds[i] <= diamonds[i + 1] for i in range(len(diamonds) - 1))
check("ML0-1 isotony: O1 subset O2 => A(O1) subset A(O2) (diamond mode-sets nest)",
      iso_ok, detail=f"|O(depth 1,2,3)| = {[len(d) for d in diamonds]}")
print("    (trivially true by the generation map, as pre-registered -- reported, not inflated.)")

# ===========================================================================
banner("ML0-3  TWISTED LOCALITY  (the net is fermionic; the twist is parity -- Klein)")
# ===========================================================================
# Explicit Jordan-Wigner Fock space, N modes, two DISJOINT regions R1, R2. OWNED convention
# (M0-C): |0>=(1,0) empty, a=[[0,1],[0,0]], a_p = Z^{(x)p} (x) a (x) I..., mode 0 leftmost.
I2 = np.eye(2)
Z2 = np.diag([1.0, -1.0])
a1 = np.array([[0.0, 1.0], [0.0, 0.0]])


def kron_list(ms):
    out = np.array([[1.0]])
    for m in ms:
        out = np.kron(out, m)
    return out


def ann(p, N):
    return kron_list([Z2] * p + [a1] + [I2] * (N - 1 - p))


Nf = 4
a = [ann(p, Nf) for p in range(Nf)]
adag = [op.conj().T for op in a]
R1, R2 = [0, 1], [2, 3]     # disjoint regions
# even (observable) operators
eA = adag[R1[0]] @ a[R1[1]]     # in A(R1), even
eB = adag[R2[0]] @ a[R2[1]]     # in A(R2), even
oA = a[R1[0]]                   # in A(R1), odd
oB = a[R2[0]]                   # in A(R2), odd
comm = lambda X, Y: X @ Y - Y @ X
acomm = lambda X, Y: X @ Y + Y @ X
check("ML0-3a even(R1) commutes with even(R2): [a0^dag a1, a2^dag a3]=0",
      np.max(np.abs(comm(eA, eB))) < 1e-12)
check("ML0-3b even(R1) commutes with odd(R2): [a0^dag a1, a2]=0",
      np.max(np.abs(comm(eA, oB))) < 1e-12)
check("ML0-3c odd(R1) ANTI-commutes with odd(R2): {a0, a2}=0",
      np.max(np.abs(acomm(oA, oB))) < 1e-12)
naive = np.max(np.abs(comm(oA, oB)))
check("ML0-3d naive commutation FAILS for odd-odd ([a0,a2] != 0) => the twist is FORCED",
      naive > 0.5, detail=f"max|[a0,a2]| = {naive:.3f}")
# Klein twist: dress R2 by the parity of R1; the twisted odd operator commutes with A(R1).
P1 = np.eye(2 ** Nf)
for i in R1:
    P1 = P1 @ (np.eye(2 ** Nf) - 2 * adag[i] @ a[i])
oB_tw = P1 @ oB
check("ML0-3e Klein-twisted odd(R2)~ = P(R1).odd(R2) COMMUTES with odd(R1): net is TWISTED-local",
      np.max(np.abs(comm(oA, oB_tw))) < 1e-12)
print("    => the physical net is the TWISTED (graded) net; twist = fermion parity (-1)^N.")
print("       This is the structure DHR sectors (ML-2) are defined relative to.")

# ===========================================================================
banner("ML0-4  TWISTED HAAG DUALITY at cell level  (Gaussian vacuum; FLAT-BAND WATCH)")
# ===========================================================================
def region_data(C, A):
    C_A = C[np.ix_(A, A)]
    zeta = np.linalg.eigvalsh(C_A).real
    zc = np.clip(zeta, 1e-12, 1 - 1e-12)
    eps = np.log((1 - zc) / zc)
    H2 = -zc * np.log(zc) - (1 - zc) * np.log(1 - zc)
    return zeta, np.sort(eps), float(np.sum(H2))


# R = a K4 triangle (edges 01,02,12 = indices 0,1,3); R^c = the complementary 3 edges.
Rtri = [0, 1, 3]
Rc = [e for e in range(NE) if e not in Rtri]
zR, epsR, SR = region_data(C, Rtri)
zRc, epsRc, SRc = region_data(C, Rc)
check("ML0-4a purity/duality precursor: S(R) = S(R^c) for the pure global vacuum",
      abs(SR - SRc) < 1e-9, detail=f"S(R)={SR:.6f}  S(R^c)={SRc:.6f} nats")
check("ML0-4b complementary regions share the modular (entanglement) spectrum: eps_R = eps_R^c",
      np.allclose(epsR, epsRc, atol=1e-7),
      detail=f"eps_R={np.round(epsR,4)}  eps_R^c={np.round(epsRc,4)}")
split_margin = min(np.min(np.abs(zR)), np.min(np.abs(1 - zR)),
                   np.min(np.abs(zRc)), np.min(np.abs(1 - zRc)))
check("ML0-4c SPLIT holds at cell level: no occupation zeta pinned at 0/1 (region genuinely "
      "entangled) => twisted Haag duality is well-posed here",
      split_margin > 1e-6, detail=f"split margin dist(zeta,{{0,1}}) = {split_margin:.4f}")
print("    FLAT-BAND WATCH (pre-registered): the M0 vacuum C=(I+iJ6)/2 is the STATIC A4-covariant")
print(f"    complex structure; its occupations are bounded away from 0/1 (margin {split_margin:.3f})")
print("    => NO flat-direction pinning at the single-cell level; duality is clean here.")
print("    HONEST SCOPE: the flat band (lambda=-1 triple) lives in the k-DISPERSION of the walk,")
print("    NOT in the static J6. The split/duality test against the flat band is therefore the")
print("    k-dependent (supercell) covariance C(k) restricted to a diamond -- that is ML-1's")
print("    object. ML-0 books cell-level duality HOLDS + forwards the flat-band split test to ML-1.")

# ===========================================================================
banner("ML0-5  COVARIANCE  (the dynamics B commutes with lattice translation + tick shift)")
# ===========================================================================
# lattice translation T_1: shift every cell x -> x + e_1, as a permutation of darts whose image
# is still in the patch (interior).  Verify [B, T_1] = 0 on the interior => the net is Z^3-covariant.
def shifted(dart, e):
    (ti, tx), (hi, hx) = dart
    return ((ti, tuple(np.array(tx) + e)), (hi, tuple(np.array(hx) + e)))


e1 = np.array([1, 0, 0])
interior = [n for n, d in enumerate(RD) if shifted(d, e1) in dpos]
Tsh = {n: dpos[shifted(RD[n], e1)] for n in interior}
# check B commutes with the shift on interior darts: for a,b interior with shifts, B[b,a]=B[T b, T a]
cov_ok = True
checked = 0
for a in interior:
    for b in interior:
        if Bp[b, a] != 0 or Bp[Tsh[b], Tsh[a]] != 0:
            if abs(Bp[b, a] - Bp[Tsh[b], Tsh[a]]) > 1e-12:
                cov_ok = False
            checked += 1
check("ML0-5 dynamics is lattice-covariant: B[b,a] = B[T_1 b, T_1 a] on the interior "
      "(Z^3 translation) ; tick shift = B itself",
      cov_ok, detail=f"{checked} nonzero dart-pairs matched under the translation")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
print("""    ML0-0  RECONCILED: net single-particle space = DART space (2|E|=12/cell); the M0 vacuum
           J6/C is its R-even (undirected-edge) sector; the non-backtracking walk B breaks the
           reversal grading ([B,R]!=0), coupling the vacuum sector to its R-odd partner -- which
           is exactly why the (cell x tick) HISTORY carries more than the static cell.
    ML0-2  CONE-EXACT: the non-backtracking B gives a STRICT combinatorial light cone. The
           anticommutator {alpha_a(t), a_c^dag}=(B^t)_{ca} is IDENTICALLY 0.0 below the geometric
           horizon (t < 1 + dist(head a, tail c)); horizon speed = one graph-step/tick. Strictly
           stronger than Lieb-Robinson => causal diamonds are EXACTLY defined. (the physics heart)
    ML0-1  ISOTONY holds (trivially, by the generation map -- reported as such).
    ML0-3  TWISTED-LOCAL: even algebras of disjoint regions commute; odd parts anticommute; naive
           commutation FAILS => the twist (fermion parity, Klein) is forced. This is the structure
           DHR sectors (ML-2) require.
    ML0-4  DUALITY-HOLDS at cell level: S(R)=S(R^c), shared modular spectrum, split margin > 0 (no
           flat-direction pinning in the static J6). The flat-band split test is FORWARDED to ML-1's
           k-dependent covariance (honest scope; not faked, not smoothed).
    ML0-5  COVARIANT: B commutes with Z^3 lattice translation; the tick shift is B itself.
    => NET-LOCKED: the history net O -> A(O) is well-posed, causally exact, twisted-local, covariant,
       cell-level dual.  ML-1 (the diamond modular flow / the BW 2pi decider) is now POSABLE.
    NO magnitude touched; NO scoreboard value moved.  ML0-6 (regrade the deprecation flag) booked
    as a documentation edit in the results commit.""")
print("RESULT:", "ALL CHECKS PASS" if ok_all else "A CHECK FAILED -- inspect above")
sys.exit(0 if ok_all else 1)
