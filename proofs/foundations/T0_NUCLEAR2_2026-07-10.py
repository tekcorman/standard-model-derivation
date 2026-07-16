#!/usr/bin/env python3
# ============================================================================
# T0-NUCLEAR-2 — the 3-body kinetic correction, V-hat from the DeltaS
#                convention ITSELF (IV.4's remaining half, re-attempted)
# ============================================================================
#
# PRE-REGISTRATION (binding, frozen):
#   internal research notes  (the ONE change:
#     V-hat's depth function)
#   internal research notes             (everything
#     else, inherited UNCHANGED: T-hat, boxes/grids, B2 import, mirror-mean
#     confrontation, verdict tree, poisons)
#
# LINEAGE / GATE FINDING THIS STATION REPAIRS (887152d / 5f87e96):
#   proofs/foundations/T0_NUCLEAR_2026-07-10.py gated CONSTRUCTION-MISMATCH:
#   the co-information functional ALONE (I for pairs, C3 = I12+I13+I23-II3
#   for triples) gives 5 at the ground pair and 15 at the ground triple,
#   where the rungs are 3 and 13 -- the gap in BOTH cases being EXACTLY the
#   vertex-branching cost Sum_v max(deg_v-2, 0) = 2, which the stage3a rung
#   convention subtracts and the bare co-information functional does not
#   carry.  This station's V-hat uses the FULL stage3a formula (compression
#   MINUS branch), so the gate is rung-reproduction BY CONSTRUCTION.
#
# MACHINERY REPLICATED (not edited; same objects, cited), from
#   proofs/foundations/T0_NUCLEAR_2026-07-10.py (this station's own prior
#   file) and, through it, from:
#   proofs/foundations/IV4_T0_class_2026-07-10.py
#     - Dirac D(k) 32x32, eps_low = lowest POSITIVE band, offset mesh
#       q_i = (i+0.5)/n_q (T-hat: IDENTICAL constituents, untouched by the
#       amendment).
#   proofs/foundations/bound_state_edge_resolved_kernel_2026-05-29.py
#     - build_prim_adjacency / one_girth_cycle / translate_vertex /
#       translate_edgeset: the girth-cycle self-translation configuration-
#       overlap machinery (primitive BCC frame), used to build the
#       CONTINUOUS V-hat(x,y) over the Jacobi relative-cell box.
#   proofs/foundations/BOUND_stage3a_dS_spectrum_2026-07-03.py (lines
#     64-78, VERBATIM) / I0b_RATIO_stage_BC_2026-07-10.py
#     - THE DEPTH FUNCTION (this station's V-hat, per the amendment):
#         DeltaS = [ Sum_e(mult_e - 1) - Sum_v max(deg_v - 2, 0) ] * b_edge,
#         b_edge = log2(k*-1) = 1 bit.
#       Also: the supercell(3) cycle/pair/triple enumeration + frozen rungs
#       (2-body ground DeltaS = 3, 648 pairs; 3-body ground DeltaS = 13,
#       216 triples) used for THE GATE (Step 1) and the histogram re-locks.
#   The 2026-06-01 co-information vertex file is NOT used in this station
#   (that was the prior station's V-hat; superseded by the amendment).
#
# ---------------------------------------------------------------------------
# FROZEN A PRIORI (nothing below was chosen in response to a binding number):
#
# STEP 1 — THE GATE (pre-reg amendment stop-clause; adjudicated FIRST):
#   Evaluate the stage3a DeltaS formula (verbatim) at the SAME reference
#   configurations used for the frozen rungs on supercell(3): the DeltaS=3
#   ground pairs (648 of them) and the DeltaS=13 ground triples (216 of
#   them).  Because DeltaS as computed IS how those rungs are defined (the
#   top of the frozen positive ladders {1,3} and {1,2,3,4,6,13}), this gate
#   is rung-reproduction BY CONSTRUCTION; it can only fail via a genuine
#   regression in the graph/cycle machinery (guarded by histogram re-lock
#   asserts against the frozen 2026-07-03 counts).  If it fails: STOP ->
#   CONSTRUCTION-MISMATCH-2, with the computed values printed (a deeper
#   finding -- not a license to reweight).  On the mismatch path NO 3-body
#   number is computed and the naive/measured values are NOT printed.
#
# STEP 2 — THE 3-BODY SOLVE (runs ONLY if the gate passes; conventions
# inherited verbatim from proofs/foundations/T0_NUCLEAR_2026-07-10.py):
#   - Constituents: three IDENTICAL bare walkers (nucleon-analog stand-ins;
#     composite-of-composites unresolved -- declared limitation, named not
#     fixed).  The DISTINGUISHABLE relative problem is solved (declared).
#   - Jacobi relative coordinates x = r1-r3, y = r2-r3, total quasimomentum
#     K = 0 sector.
#   - T-hat: E3(q1,q2) = eps(q1)+eps(q2)+eps(-q1-q2), eps = the frozen
#     lowest positive band of the 32x32 D(k); real-space hoppings by the
#     same offset-mesh quadrature as the 2-body station; eps(-k) = eps(k)
#     exactly (asserted on the mesh) so t3 = t.
#   - V-hat (THE AMENDED PIECE): V(x,y) = -e_bit * DeltaS(config), config =
#     the 3 edge-sets {E_0, E_x, E_y} of ONE canonical girth-10 cycle
#     self-translated by the relative Jacobi displacements x, y in the
#     primitive periodic frame (supercell L = 4*box+3) -- the SAME
#     self-translation ansatz the prior (gated) station used for its
#     co-information V, with the depth function replaced per the amendment.
#     Used RAW (the gate IS the depth normalization; no rescaling, no cap).
#   - Box/grid sizes (DECLARED HERE, never changed after a result; inherited
#     from the original pre-reg docstring):
#       boxes = {1, 2, 3}  (relative cells x,y in [-box,box]^3;
#                           M = (2*box+1)^6 = 729 / 15,625 / 117,649)
#       n_q   = 14 primary, with a duplicate at (box = 2, n_q = 18) as the
#               dispersion-grid convergence control.
#     Convergence shown across the three boxes at n_q = 14 (>= 2 sizes) plus
#     the n_q duplicate.  Primary number = (box = 3, n_q = 14).
#   - Solver: dense eigh at box = 1 (cross-check) + ARPACK eigsh
#     (LinearOperator, structured matvec; k = 1, which='SA', tol = 1e-10,
#     deterministic v0) at all boxes.  Eigen identity E0 = <T>+<V> asserted
#     to 1e-7.  Algebra validated by an ALWAYS-RUN self-test on a labeled
#     TOY operator (no frozen-construction number produced by the test).
#   - Feasibility (printed BEFORE any diagonalization, and before each V
#     build): M per box; V-build wall time; one measured matvec per box;
#     projected eigsh cost = 400 matvecs x safety 3; hard station cap
#     ~20 min (T_CAP = 1140 s).  If a projection exceeds the remaining
#     budget: print INFEASIBLE-AT-CAP and STOP (a bookable outcome).
#   - Outputs: B3 = E_th3 - E0;  T0^(3) = 13 - B3;
#              R_kin = B3 / B2,  B2 = 2.6689 FROZEN (T0-CLASS pole,
#              n_q = 26, IMPORTED, never re-solved).
#
# FROZEN CONFRONTATIONS (declared context, never optimization targets;
# printed ONLY in the final confrontation block on the gate-pass path):
#   measured B(3H)/B_d = 3.81279, B(3He)/B_d = 3.46946, mirror MEAN
#   3.64112 (needs f = 0.8403); the naive sealed prediction 13/3 = 4.333.
#   T0 CANNOT split the mirror pair (identical constituents) -- the
#   confrontation object is the MIRROR MEAN; the +-4.95 % split is E_odd's
#   (0.381876 MeV, registered row, NEVER absorbed here).
#   Stage-C bins (reporting conventions): EXACT (<1 %), NEAR (1-10 %,
#   quantified OPEN), OFF (>10 %, OPEN).
#
# FROZEN VERDICT TREE (inherited unchanged):
#   CONSTRUCTION-MISMATCH-2 : the gate fails (per above; a deeper finding).
#   KIN-WRONG-WAY : R_kin moves AWAY from the mirror mean relative to the
#                   naive 13/3 (|R_kin - mean| > |13/3 - mean|).
#   KIN-CLOSES    : R_kin lands EXACT/NEAR (<= 10 %) on the mirror mean --
#                   the RATIO-MISS is adjudicated as the dropped kinetic
#                   term (residual named if NEAR).
#   KIN-PARTIAL   : R_kin moves toward the mean but lands OFF (> 10 %) --
#                   deviation quantified and booked; remaining legs named
#                   (the EP-2 dictionary adoption, the bare-walker stand-in).
#
# POISONS (carried verbatim): no kappa; no per-system reweighting; the
# rungs 3/13 and B2 = 2.6689 imported frozen and untouched; E_odd never
# absorbed; box/band/functional unchanged after (indeed: before) any
# 3-body number; measured values only in the final confrontation printout;
# bare-walker limitation declared not fixed; an open miss stays open; no
# post-output convention changes.
# ---------------------------------------------------------------------------
# Standalone: python3 proofs/foundations/T0_NUCLEAR2_2026-07-10.py ; exit 0.
# Asserts fire ONLY on machine-checkable regressions + proven algebra
# (Dirac validation, histogram re-locks, solver self-test); ALL station
# verdicts are PRINTED, never asserted.
# ============================================================================

import math
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations, product

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from proofs.common import find_bonds  # noqa: E402
import srs_graph_analysis as srs      # noqa: E402

T_START = time.monotonic()
T_CAP_S = 1140.0                      # ~19 min hard budget (cap ~20 min)

# ------------------------- frozen constants -------------------------------
E_BIT = 1.0                           # e_bit = t (DISCLOSED, carried)
GIRTH = 10
K_STAR = 3
B_EDGE = math.log2(K_STAR - 1)        # = 1 bit (stage3a convention)
DS2_RUNG = 3                          # frozen 2-body ground rung
DS3_RUNG = 13                         # frozen 3-body ground rung
B2_FROZEN = 2.6689                    # T0-CLASS grid-converged pole (FROZEN)

# stage3a / I0b frozen re-lock anchors (I0b_RATIO_stage_BC lines 240-246;
# re-verified against T0_NUCLEAR_2026-07-10.py's own re-lock)
LADDER2 = [1, 3]
LADDER3 = [1, 2, 3, 4, 6, 13]
HIST2_STAGE0 = {-1: 4212, 0: 2592, 1: 648, 3: 648}
HIST3_20260703 = {-3: 108, -2: 76464, -1: 82512, 0: 54432,
                  1: 20736, 2: 19224, 3: 16848, 4: 3888, 6: 2592, 13: 216}

W = 78
ok_all = True


def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")


def banner(t):
    print("=" * W)
    print(f" {t}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Block 1 — Dirac D(k) + lowest positive band (T0_NUCLEAR_2026-07-10.py
# Block 1, itself IV4 Blocks 1/3, verbatim) — T-hat, UNTOUCHED by the
# amendment (identical constituents throughout)
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
k3 = lambda a, b, c: np.kron(np.kron(a, b), c)
GAMMAS = [k3(X, I2, I2), k3(Y, I2, I2), k3(Z, X, I2),
          k3(Z, Y, I2), k3(Z, Z, X), k3(Z, Z, Y)]
BONDS = find_bonds()


def undirected_edges():
    seen = {}
    for src, tgt, cell in BONDS:
        cell = tuple(int(c) for c in cell)
        key = (src, tgt, cell) if src < tgt else (tgt, src, tuple(-c for c in cell))
        seen[key] = True
    e = sorted(seen.keys())
    assert len(e) == 6
    return e


EDGES = undirected_edges()


def L_e(edge, k):
    a, b, n = edge
    L = np.zeros((4, 4), dtype=complex)
    ph = np.exp(2j * np.pi * np.dot(k, n))
    L[b, a], L[a, b] = ph, np.conj(ph)
    for c in range(4):
        if c not in (a, b):
            L[c, c] = 1.0
    return L


def D_of_k(k):
    D = np.zeros((32, 32), dtype=complex)
    for i, e in enumerate(EDGES):
        D += np.kron(GAMMAS[i], L_e(e, k))
    return D


def validate_dirac():
    kk = np.array([0.17, 0.31, 0.53])
    D = D_of_k(kk)
    R = np.zeros((32, 32), dtype=complex)
    Ls = [L_e(e, kk) for e in EDGES]
    for i in range(6):
        for j in range(6):
            if i != j:
                R += 0.5 * np.kron(GAMMAS[i] @ GAMMAS[j], Ls[i] @ Ls[j] - Ls[j] @ Ls[i])
    return np.allclose(D @ D, 6 * np.eye(32) + R, atol=1e-9) and np.allclose(D, D.conj().T)


def eps_low(k):
    """The frozen 'lowest Dirac band' convention (2026-05-29 lines 110-112)."""
    ev = np.linalg.eigvalsh(D_of_k(k))
    return ev[ev > 1e-9].min()


def eps_on_mesh(n_q, offset=0.5):
    """eps_low over the mesh q_i = (i+offset)/n_q (offset=0.5: the frozen
    2-body quadrature mesh; offset=0.0: the integer mesh, needed for
    eps(q1+q2) which lands on integer fractions)."""
    qs = (np.arange(n_q) + offset) / n_q
    eps = np.empty((n_q, n_q, n_q))
    for i, j, l in product(range(n_q), repeat=3):
        eps[i, j, l] = eps_low(np.array([qs[i], qs[j], qs[l]]))
    return eps


# ---------------------------------------------------------------------------
# Block 2 — THE DEPTH FUNCTION: the stage3a DeltaS convention ITSELF
# (BOUND_stage3a_dS_spectrum_2026-07-03.py lines 64-78, VERBATIM).  Per the
# amendment this is now BOTH the gate criterion AND V-hat's depth function
# directly -- the bare co-information functional (I / C3 alone, no branch
# term) is NOT used anywhere in this station.
# ---------------------------------------------------------------------------
def cycle_edges(cycle):
    n = len(cycle)
    return frozenset(frozenset((cycle[i], cycle[(i + 1) % n])) for i in range(n))


def dS_union_parts(edgesets):
    """DeltaS = [Sum_e(mult_e-1) - Sum_v max(deg_v-2,0)] * b_edge, in units
    of b_edge = 1 bit; compression and branch returned separately (same
    formula as stage3a's dS_of_union, terms exposed only for printing)."""
    mult = defaultdict(int)
    for es in edgesets:
        for e in es:
            mult[e] += 1
    deg = defaultdict(int)
    for e in mult:
        for v in e:
            deg[v] += 1
    compression = sum(m - 1 for m in mult.values())
    branch = sum(max(d - 2, 0) for d in deg.values())
    return (compression - branch) * B_EDGE, compression, branch


# ---------------------------------------------------------------------------
# Block 3 — primitive-frame self-translation machinery (bound_state_edge_
# resolved_kernel_2026-05-29.py lines 115-194, verbatim; used to build the
# CONTINUOUS V-hat(x,y) over the Jacobi relative-cell box)
# ---------------------------------------------------------------------------
def build_prim_adjacency(L):
    adj = defaultdict(list)

    def vid(n, iv):
        return (n[0] % L, n[1] % L, n[2] % L, iv)

    for src, tgt, cell in BONDS:
        cell = np.array(cell)
        for n in product(range(L), repeat=3):
            n = np.array(n)
            a = vid(n, src)
            b = vid(n + cell, tgt)
            if b not in adj[a]:
                adj[a].append(b)
            if a not in adj[b]:
                adj[b].append(a)
    return adj


def one_girth_cycle(adj, start):
    found = []

    def dfs(path):
        if len(found):
            return
        cur = path[-1]
        if len(path) == GIRTH:
            if start in adj[cur]:
                found.append(list(path))
            return
        for w in adj[cur]:
            if w == start or w in path:
                continue
            path.append(w)
            dfs(path)
            path.pop()
            if found:
                return
    dfs([start])
    return found[0] if found else None


def translate_vertex(v, d, L):
    return ((v[0] + d[0]) % L, (v[1] + d[1]) % L, (v[2] + d[2]) % L, v[3])


def translate_edgeset(E0, d, L):
    return frozenset(frozenset((translate_vertex(u, d, L), translate_vertex(w, d, L)))
                     for u, w in (tuple(e) for e in E0))


# ---------------------------------------------------------------------------
# Block 4 — the 3-body Jacobi solver (T0_NUCLEAR_2026-07-10.py Block 5,
# verbatim; runs only on the gate-pass path; algebra validated by the toy
# self-test below)
# ---------------------------------------------------------------------------
def t_from_eps(eps, box):
    """t(Delta) = (1/n^3) sum_q eps(q) e^{2 pi i q.Delta} over the mesh eps
    was built on, for Delta in [-2box, 2box]^3 (exact phase factorization,
    the IV4 kinetic_real_space algebra applied to the SINGLE-particle band)."""
    n_q = eps.shape[0]
    qs = (np.arange(n_q) + 0.5) / n_q
    rng = np.arange(-2 * box, 2 * box + 1)
    px = np.exp(2j * np.pi * np.outer(rng, qs))
    A = np.einsum('ri,ijk->rjk', px, eps.astype(complex))
    Bm = np.einsum('sj,rjk->rsk', px, A)
    T = np.einsum('tk,rsk->rst', px, Bm) / n_q ** 3
    assert np.abs(T.imag).max() < 1e-9, "t(Delta) not real (evenness broken)"
    return T.real


def threshold3(eps_off, eps_int):
    """E_th3 = min over offset-mesh (q1, q2) of eps(q1)+eps(q2)+eps(q1+q2).
    q1+q2 = ((i+j+1) mod n)/n per axis lands on the INTEGER mesh (proven
    index algebra, asserted in the self-test); eps(-p) = eps(p) used."""
    n = eps_off.shape[0]
    off = eps_off.reshape(-1)
    ints = eps_int.reshape(-1)
    I3 = np.array(list(product(range(n), repeat=3)))
    S = (np.arange(n)[:, None] + np.arange(n)[None, :] + 1) % n
    best = np.inf
    for a in range(len(off)):
        ia = I3[a]
        sidx = (S[ia[0], I3[:, 0]] * n + S[ia[1], I3[:, 1]]) * n + S[ia[2], I3[:, 2]]
        e3 = off[a] + off + ints[sidx]
        m = float(e3.min())
        if m < best:
            best = m
    return best


class Jacobi3Body:
    """H = T1 + T2 + T3 + V on relative cells (x, y) in [-box, box]^3 x
    [-box, box]^3 (K = 0 Jacobi sector):
      T1: t(x-x') delta_{y,y'}   T2: t(y-y') delta_{x,x'}
      T3: t3(x-x') delta_{x-x', y-y'}   V: diagonal V(x,y).
    Structured matvec: T1/T2 as nC x nC matmuls; T3 via the (u = x-y)-
    diagonal gather -> batched matmul -> bijective scatter."""

    def __init__(self, box, t_tensor, t3_tensor, Vmat):
        self.box = box
        cells = list(product(range(-box, box + 1), repeat=3))
        self.cells = cells
        nC = len(cells)
        self.nC = nC
        self.M = nC * nC
        Rv = np.array(cells)
        d = Rv[:, None, :] - Rv[None, :, :]
        w = 4 * box + 1
        idxD = ((d[..., 0] + 2 * box) * w + (d[..., 1] + 2 * box)) * w \
            + (d[..., 2] + 2 * box)                     # (nC,nC) flat Delta idx
        self.Tx = t_tensor.reshape(-1)[idxD]            # pair-hop matrix (sym)
        self.Tx3 = t3_tensor.reshape(-1)[idxD]          # third-walker hop
        assert np.allclose(self.Tx, self.Tx.T, atol=1e-12)
        self.V = Vmat                                   # (nC, nC) diagonal vals
        # gather/scatter structure for T3: u index = idxD[a, b]
        U = w ** 3
        G = np.zeros((U, nC), dtype=np.int64)
        MASK = np.zeros((U, nC), dtype=bool)
        for a in range(nC):
            for b in range(nC):
                u = idxD[a, b]
                G[u, a] = a * nC + b
                MASK[u, a] = True
        self.G, self.MASK = G, MASK

    def matvec(self, psi):
        P = psi.reshape(self.nC, self.nC)
        out = self.Tx @ P + P @ self.Tx.T + self.V * P
        Yg = psi.reshape(-1)[self.G] * self.MASK
        Zg = (Yg @ self.Tx3.T) * self.MASK
        o3 = np.zeros(self.M)
        o3[self.G[self.MASK]] = Zg[self.MASK]
        return out.reshape(-1) + o3

    def matvec_T(self, psi):                            # kinetic only (for <T>)
        P = psi.reshape(self.nC, self.nC)
        out = self.Tx @ P + P @ self.Tx.T
        Yg = psi.reshape(-1)[self.G] * self.MASK
        Zg = (Yg @ self.Tx3.T) * self.MASK
        o3 = np.zeros(self.M)
        o3[self.G[self.MASK]] = Zg[self.MASK]
        return out.reshape(-1) + o3

    def dense(self):
        """Dense assembly by an INDEPENDENT construction (kron + broadcast
        equality for T3) — the self-test cross-check."""
        nC = self.nC
        H = np.kron(self.Tx, np.eye(nC)) + np.kron(np.eye(nC), self.Tx)
        box, w = self.box, 4 * self.box + 1
        Rv = np.array(self.cells)
        d = Rv[:, None, :] - Rv[None, :, :]
        idxD = ((d[..., 0] + 2 * box) * w + (d[..., 1] + 2 * box)) * w \
            + (d[..., 2] + 2 * box)
        eq = idxD[:, None, :, None] == idxD[None, :, None, :]
        T3 = (self.Tx3[:, None, :, None] * eq).reshape(self.M, self.M)
        H += T3
        H += np.diag(self.V.reshape(-1))
        return H

    def ground(self, dense_check=False):
        from scipy.sparse.linalg import LinearOperator, eigsh
        v0 = np.zeros(self.M)
        i0x = self.cells.index((0, 0, 0))
        v0[i0x * self.nC + i0x] = 1.0                   # deterministic start
        op = LinearOperator((self.M, self.M), matvec=self.matvec, dtype=float)
        ev, evec = eigsh(op, k=1, which='SA', tol=1e-10, v0=v0, maxiter=5000)
        E0 = float(ev[0])
        psi = evec[:, 0]
        T_exp = float(psi @ self.matvec_T(psi))
        V_exp = float(np.sum(self.V.reshape(-1) * psi ** 2))
        assert abs(E0 - (T_exp + V_exp)) < 1e-7, "eigen identity broken"
        if dense_check:
            Ed = float(np.linalg.eigvalsh(self.dense())[0])
            assert abs(E0 - Ed) < 1e-8, f"eigsh vs dense mismatch: {E0} vs {Ed}"
        return E0, T_exp, V_exp


def dS_potential(box):
    """V(x,y) = -e_bit * DeltaS(config) on the 3-walker self-translation
    configuration overlap (girth-cycle primitive frame, Block 3 machinery),
    where DeltaS is the AMENDED depth function (Block 2, dS_union_parts)
    evaluated on the 3 edge-sets {E_0, E_x, E_y} -- one canonical girth-10
    cycle self-translated by the relative Jacobi displacements x and y.
    This is the prior (gated) station's coinf_potential(box) with the depth
    function replaced per the amendment; everything else (L = 4*box+3,
    build_prim_adjacency, one_girth_cycle, translate_edgeset) is identical."""
    L = 4 * box + 3
    adj = build_prim_adjacency(L)
    start = (L // 2, L // 2, L // 2, 0)
    cyc = one_girth_cycle(adj, start)
    assert cyc is not None and len(cyc) == GIRTH, "no girth-10 cycle found"
    E0 = frozenset(cycle_edges(tuple(cyc)))
    cells = list(product(range(-box, box + 1), repeat=3))
    Ts = {c: translate_edgeset(E0, c, L) for c in cells}
    nC = len(cells)
    V = np.zeros((nC, nC))
    dS_min, dS_max, br_max = math.inf, -math.inf, 0
    for ix, xv in enumerate(cells):
        Ex = Ts[xv]
        for iy, yv in enumerate(cells):
            Ey = Ts[yv]
            dS, comp, br = dS_union_parts([E0, Ex, Ey])
            V[ix, iy] = -dS * E_BIT
            dS_min = min(dS_min, dS)
            dS_max = max(dS_max, dS)
            br_max = max(br_max, br)
    return V, {"dS_min": dS_min, "dS_max": dS_max, "br_max": br_max}


# ---------------------------------------------------------------------------
# Block 5 — solver algebra SELF-TEST (TOY operator; always runs; produces NO
# frozen-construction number) — T0_NUCLEAR_2026-07-10.py Block 6, verbatim
# ---------------------------------------------------------------------------
def solver_self_test():
    n_q = 6
    qs = (np.arange(n_q) + 0.5) / n_q
    Qi, Qj, Ql = np.meshgrid(qs, qs, qs, indexing='ij')
    eps_toy = 3.0 - np.cos(2 * np.pi * Qi) - np.cos(2 * np.pi * Qj) \
        - np.cos(2 * np.pi * Ql)                         # separable toy band
    t_toy = t_from_eps(eps_toy, box=1)
    rng = np.random.default_rng(20260710)
    Vtoy = -3.0 * rng.random((27, 27))
    op = Jacobi3Body(1, t_toy, t_toy, Vtoy)
    H = op.dense()
    assert np.allclose(H, H.T, atol=1e-12), "toy dense H not symmetric"
    for _ in range(3):                                   # matvec vs dense
        v = rng.standard_normal(op.M)
        assert np.allclose(op.matvec(v), H @ v, atol=1e-9), \
            "structured matvec != dense H"
    E0, T_exp, V_exp = op.ground(dense_check=True)
    # threshold index algebra: ((i+.5)/n + (j+.5)/n) mod 1 == ((i+j+1)%n)/n
    for i in range(n_q):
        for j in range(n_q):
            lhs = ((i + 0.5) / n_q + (j + 0.5) / n_q) % 1.0
            rhs = ((i + j + 1) % n_q) / n_q
            assert abs(lhs - rhs) < 1e-12, "sum-mesh index algebra broken"
    # threshold3 vs brute force on the toy
    qs0 = np.arange(n_q) / n_q
    Q0i, Q0j, Q0l = np.meshgrid(qs0, qs0, qs0, indexing='ij')
    eps_int_toy = 3.0 - np.cos(2 * np.pi * Q0i) - np.cos(2 * np.pi * Q0j) \
        - np.cos(2 * np.pi * Q0l)
    th = threshold3(eps_toy, eps_int_toy)
    brute = np.inf
    for i1 in product(range(n_q), repeat=3):
        q1 = (np.array(i1) + 0.5) / n_q
        e1 = 3.0 - np.cos(2 * np.pi * q1).sum()
        for i2 in product(range(n_q), repeat=3):
            q2 = (np.array(i2) + 0.5) / n_q
            e2 = 3.0 - np.cos(2 * np.pi * q2).sum()
            e3 = 3.0 - np.cos(2 * np.pi * (-(q1 + q2))).sum()
            brute = min(brute, e1 + e2 + e3)
    assert abs(th - brute) < 1e-10, f"threshold3 {th} != brute {brute}"
    # amended-depth-function sanity: dS_union_parts on two IDENTICAL toy
    # edge-sets of a 4-cycle reduces to the textbook s_run - 2 case
    toy_cyc = frozenset(frozenset((i, (i + 1) % 4)) for i in range(4))
    dS_id, comp_id, br_id = dS_union_parts([toy_cyc, toy_cyc])
    assert (comp_id, br_id) == (4, 0) and dS_id == 4.0, \
        f"dS_union_parts toy check failed: {(dS_id, comp_id, br_id)}"
    return E0


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    banner("T0-NUCLEAR-2 — 3-body kinetic correction, V-hat from the "
           "DeltaS convention itself")
    print("pre-reg: internal research notes "
          "(FROZEN; inherits\n         internal research notes "
          "verbatim except V-hat)")
    print(f"units: substrate energy (t = e_bit = 1, disclosed); "
          f"B2 = {B2_FROZEN} FROZEN (T0-CLASS pole, n_q = 26; not re-solved)")

    ok = validate_dirac()
    print(f"\n[validation] Dirac D(k)^2 = 6I + R_sub : {'PASS' if ok else 'FAIL'}")
    assert ok, "Dirac validation regression"

    E0_toy = solver_self_test()
    print(f"[validation] solver algebra self-test (TOY band + seeded random "
          f"well;\n             dense == structured matvec == eigsh; threshold "
          f"index algebra;\n             dS_union_parts toy identity; eigen "
          f"identity): PASS  (toy E0 = {E0_toy:.6f} — a TOY number, not a "
          f"result)")

    # ================= STEP 1 — THE GATE =================
    banner("STEP 1 — THE GATE: rung-reproduction BY CONSTRUCTION "
           "(V-hat's depth function IS the DeltaS convention)")
    print("    depth function (BOUND_stage3a_dS_spectrum_2026-07-03.py "
          "lines 64-78, VERBATIM):")
    print("      DeltaS = [ Sum_e(mult_e-1) - Sum_v max(deg_v-2,0) ] * "
          "b_edge,  b_edge = 1 bit")
    print("    (the prior station's CONSTRUCTION-MISMATCH used the bare "
          "co-information\n     functional I / C3 ALONE -- missing EXACTLY "
          "this branch term: ground pair\n     I=5, branch=2 -> dS=3; "
          "ground triple C3=15, branch=2 -> dS=13)")

    t0 = time.monotonic()
    positions, edges, adjacency, cell_indices = srs.build_supercell(3)
    n_verts = len(positions)
    g = srs.find_girth(adjacency, n_verts, max_length=14)
    assert g == GIRTH, f"girth {g} != {GIRTH}"
    seen = set()
    for v in range(n_verts):
        for cyc in srs.enumerate_cycles_dfs(adjacency, v, GIRTH):
            seen.add(cyc)
    cycles = [tuple(c) for c in seen]
    edgesets = [cycle_edges(c) for c in cycles]
    print(f"\n    supercell(3): {n_verts} vertices, girth {g}, "
          f"{len(cycles)} girth-{GIRTH} cycles")

    edge_to_cyc = defaultdict(set)
    for ci, es in enumerate(edgesets):
        for e in es:
            edge_to_cyc[e].add(ci)
    overlap_nbr = defaultdict(set)
    pairs = set()
    for e, cs in edge_to_cyc.items():
        for a, b in combinations(sorted(cs), 2):
            pairs.add((a, b))
            overlap_nbr[a].add(b)
            overlap_nbr[b].add(a)

    # ---- 2-body: rung re-lock + gate (ground pairs -> dS == 3) ----
    hist2 = Counter()
    ground_pair_dS_vals = set()
    for a, b in pairs:
        dS, comp, br = dS_union_parts([edgesets[a], edgesets[b]])
        hist2[round(dS)] += 1
        if round(dS) == DS2_RUNG:
            ground_pair_dS_vals.add(round(dS))
    assert dict(hist2) == HIST2_STAGE0, \
        f"2-body histogram re-lock FAILED: {dict(hist2)}"
    pos2 = sorted(k for k in hist2 if k > 0)
    assert pos2 == LADDER2, f"2-body spectrum {pos2} != {LADDER2}"
    check("gate re-lock: 2-body histogram == Stage-0 {-1:4212,0:2592,1:648,3:648}",
          dict(hist2) == HIST2_STAGE0)
    gate_2body = bool(ground_pair_dS_vals) and ground_pair_dS_vals == {DS2_RUNG} \
        and max(pos2) == DS2_RUNG
    check(f"GATE: all {hist2[DS2_RUNG]} ground-pair configurations give "
          f"DeltaS == {DS2_RUNG}", gate_2body)

    # ---- 3-body: rung re-lock + gate (ground triples -> dS == 13) ----
    triples = set()
    for b in range(len(cycles)):
        nbrs = sorted(overlap_nbr[b])
        for a, c in combinations(nbrs, 2):
            triples.add(frozenset((a, b, c)))
    hist3 = Counter()
    ground_tri_dS_vals = set()
    for tri in triples:
        i, j, k = sorted(tri)
        dS, comp, br = dS_union_parts([edgesets[i], edgesets[j], edgesets[k]])
        hist3[round(dS)] += 1
        if round(dS) == DS3_RUNG:
            ground_tri_dS_vals.add(round(dS))
    pos3 = sorted(k for k in hist3 if k > 0)
    assert pos3 == LADDER3, f"3-body spectrum {pos3} != {LADDER3}"
    assert dict(hist3) == HIST3_20260703, \
        f"3-body histogram re-lock FAILED: {dict(hist3)}"
    check("gate re-lock: 3-body histogram == recorded 2026-07-03 run "
          f"(ladder {LADDER3}, #(dS=13)=216)", dict(hist3) == HIST3_20260703)
    gate_3body = bool(ground_tri_dS_vals) and ground_tri_dS_vals == {DS3_RUNG} \
        and max(pos3) == DS3_RUNG
    check(f"GATE: all {hist3[DS3_RUNG]} ground-triple configurations give "
          f"DeltaS == {DS3_RUNG}", gate_3body)
    print(f"\n    gate timing: {time.monotonic() - t0:.1f} s")

    gate_pass = gate_2body and gate_3body
    print(f"\n    GATE CRITERION: ground pairs -> DeltaS == {DS2_RUNG}; "
          f"ground triples -> DeltaS == {DS3_RUNG}.")
    print(f"      2-body: {'PASS' if gate_2body else 'FAIL'}   "
          f"3-body: {'PASS' if gate_3body else 'FAIL'}")

    if not gate_pass:
        # ================= VERDICT: CONSTRUCTION-MISMATCH-2 =================
        banner("VERDICT (frozen tree): CONSTRUCTION-MISMATCH-2")
        print(f"""
  Even the full stage3a DeltaS formula (compression MINUS branch) does NOT
  reproduce its own frozen rungs at the reference configurations it was
  used to DEFINE (computed values, supercell(3), all histogram re-locks
  checked above):

    ground pairs   : DeltaS in {ground_pair_dS_vals or '{}'}   REQUIRED {{{DS2_RUNG}}}
    ground triples : DeltaS in {ground_tri_dS_vals or '{}'}   REQUIRED {{{DS3_RUNG}}}

  This would be a DEEPER finding than the prior CONSTRUCTION-MISMATCH: it
  would mean the graph/cycle enumeration or the dS_union_parts replication
  itself has regressed relative to BOUND_stage3a_dS_spectrum_2026-07-03.py,
  since DeltaS AS DEFINED is how those rungs were originally identified.
  Per the pre-reg stop-clause: no re-weighted functional was built, no rung
  was re-assigned, no alternative depth was tried.

  NOT COMPUTED (stop-clause honored): B3, T0^(3), R_kin.  The 3-body solve
  did not run; no confrontation printout exists; the RATIO-MISS stays OPEN
  exactly as booked.

  Poisons honored: no kappa; no per-system reweighting; rungs 3/13 and
  B2 = {B2_FROZEN} imported frozen and untouched; E_odd never absorbed;
  box/band/functional unchanged after (indeed: before) any 3-body number —
  none was produced; bare-walker limitation moot on this path; the open
  miss stays open.""")
        check("station verdict adjudicated: CONSTRUCTION-MISMATCH-2 "
              "(stop-clause path)", True)
        print("=" * W)
        print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}"
              f"   |   STATION VERDICT: CONSTRUCTION-MISMATCH-2")
        print("=" * W)
        sys.exit(0 if ok_all else 1)

    print("\n    GATE PASSED (by construction, as expected): V-hat's depth "
          "function IS the rung\n    definition itself; the prior station's "
          "branch-cost gap is repaired by using the\n    full formula "
          "directly, not by reweighting.")

    # ================= STEP 2 — THE 3-BODY SOLVE (gate-pass path) ==========
    banner("STEP 2 — THE 3-BODY JACOBI SOLVE (gate PASSED; frozen sizes, "
           "inherited unchanged)")
    BOXES = (1, 2, 3)
    NQ_PRIMARY, NQ_DUP = 14, 18

    # ---- feasibility print BEFORE any diagonalization ----
    print("\n    feasibility (declared cap ~20 min; projection = 400 matvecs "
          "x safety 3):")
    sizes = {b: (2 * b + 1) ** 6 for b in BOXES}
    for b in BOXES:
        print(f"      box {b}: M = (2*{b}+1)^6 = {sizes[b]:,}")
    remaining = T_CAP_S - (time.monotonic() - T_START)
    print(f"      remaining budget: {remaining:.0f} s")

    eps14 = eps_on_mesh(NQ_PRIMARY)
    assert np.allclose(eps14, eps14[::-1, ::-1, ::-1], atol=1e-10), \
        "eps evenness broken on the offset mesh"
    eps14_int = eps_on_mesh(NQ_PRIMARY, offset=0.0)
    E_th14 = threshold3(eps14, eps14_int)
    eps_min = float(eps14.min())
    print(f"\n    E_th3 (n_q = {NQ_PRIMARY}) = {E_th14:.6f}   "
          f"[context: 3*eps_min = {3 * eps_min:.6f}; the K=0 frustration "
          f"gap = {E_th14 - 3 * eps_min:+.6f}]")

    results = {}
    projected_ok = True
    for b in BOXES:
        remaining = T_CAP_S - (time.monotonic() - T_START)
        tV0 = time.monotonic()
        Vmat, diag = dS_potential(b)
        tV = time.monotonic() - tV0
        print(f"\n    box {b}: V-hat built in {tV:.2f} s (DeltaS-based, "
              f"depth range on the grid\n             = "
              f"[{diag['dS_min']:.0f}, {diag['dS_max']:.0f}], branch up to "
              f"{diag['br_max']:.0f}); V(0,0) = "
              f"{Vmat[len(Vmat) // 2, len(Vmat) // 2]:.1f}")
        remaining = T_CAP_S - (time.monotonic() - T_START)
        if remaining <= 0:
            print(f"      -> INFEASIBLE-AT-CAP after building V at box {b}: "
                  f"no budget remains; stopping.")
            projected_ok = False
            break
        t_tensor = t_from_eps(eps14, b)
        op = Jacobi3Body(b, t_tensor, t_tensor, Vmat)
        v = np.random.default_rng(1).standard_normal(op.M)
        t1 = time.monotonic()
        op.matvec(v)
        mv_s = time.monotonic() - t1
        proj = 400 * mv_s * 3
        remaining = T_CAP_S - (time.monotonic() - T_START)
        print(f"      matvec = {mv_s * 1e3:.1f} ms; projected eigsh ~ "
              f"{proj:.0f} s; remaining {remaining:.0f} s")
        if proj > remaining:
            print(f"      -> INFEASIBLE-AT-CAP at box {b}: projected "
                  f"{proj:.0f} s > remaining {remaining:.0f} s; stopping "
                  f"(bookable outcome; smaller boxes reported).")
            projected_ok = False
            break
        E0, T_exp, V_exp = op.ground(dense_check=(b == 1))
        B3 = E_th14 - E0
        results[(b, NQ_PRIMARY)] = (E0, B3, T_exp, V_exp)
        print(f"      E0 = {E0:.6f}   B3 = E_th3 - E0 = {B3:.6f}   "
              f"<T> = {T_exp:.6f}   <V> = {V_exp:.6f}"
              + ("   [dense cross-check PASS]" if b == 1 else ""))

    # dispersion-grid duplicate (declared): box = 2, n_q = 18
    if projected_ok and (2, NQ_PRIMARY) in results:
        remaining = T_CAP_S - (time.monotonic() - T_START)
        print(f"\n    duplicate control (box 2, n_q = {NQ_DUP}); remaining "
              f"budget {remaining:.0f} s")
        eps18 = eps_on_mesh(NQ_DUP)
        eps18_int = eps_on_mesh(NQ_DUP, offset=0.0)
        E_th18 = threshold3(eps18, eps18_int)
        Vmat2, _ = dS_potential(2)
        op18 = Jacobi3Body(2, t_from_eps(eps18, 2), t_from_eps(eps18, 2), Vmat2)
        E0_18, T18, V18 = op18.ground()
        B3_18 = E_th18 - E0_18
        results[(2, NQ_DUP)] = (E0_18, B3_18, T18, V18)
        print(f"      E_th3 = {E_th18:.6f}  E0 = {E0_18:.6f}  "
              f"B3 = {B3_18:.6f}")

    if not results:
        banner("VERDICT: INFEASIBLE-AT-CAP (bookable; no 3-body number produced)")
        print("=" * W)
        sys.exit(0 if ok_all else 1)

    # ---- convergence table + primary numbers ----
    print("\n    convergence (B3 across the declared sizes):")
    for key in sorted(results):
        b, nq = key
        print(f"      box {b}, n_q {nq}:  B3 = {results[key][1]:.6f}")
    prim_key = (max(b for (b, nq) in results if nq == NQ_PRIMARY), NQ_PRIMARY)
    E0_p, B3_p, _, _ = results[prim_key]
    T0_3 = DS3_RUNG - B3_p
    R_kin = B3_p / B2_FROZEN
    print(f"\n    PRIMARY (box {prim_key[0]}, n_q {NQ_PRIMARY}):")
    print(f"      B3      = {B3_p:.6f}")
    print(f"      T0^(3)  = {DS3_RUNG} - B3 = {T0_3:.6f}")
    print(f"      R_kin   = B3 / B2 = {B3_p:.6f} / {B2_FROZEN} = {R_kin:.6f}")

    # ================= FINAL CONFRONTATION (the ONLY place measured values
    # and the naive 13/3 appear) =================
    banner("FINAL CONFRONTATION (declared context; never optimization targets)")
    R_3H, R_3HE = 3.81279, 3.46946
    MEAN = 3.64112
    NAIVE = 13.0 / 3.0
    dev_mean = 100.0 * (R_kin / MEAN - 1.0)
    dev_naive_mean = 100.0 * (NAIVE / MEAN - 1.0)

    def bin_of(d):
        a = abs(d)
        return "EXACT (<1%)" if a < 1 else \
               ("NEAR (1-10%) -> quantified OPEN" if a <= 10 else
                "OFF (>10%) -> OPEN")

    print(f"\n      naive sealed prediction 13/3 = {NAIVE:.6f}   "
          f"(vs mean: {dev_naive_mean:+.2f}%)")
    print(f"      measured B(3H)/B_d  = {R_3H}   B(3He)/B_d = {R_3HE}")
    print(f"      mirror MEAN = {MEAN}  (the confrontation object; the "
          f"+-4.95% split is E_odd's,\n       0.381876 MeV, registered row, "
          f"NEVER absorbed here)")
    print(f"\n      R_kin = {R_kin:.6f}  vs mean {MEAN}:  deviation "
          f"{dev_mean:+.2f}%  -> {bin_of(dev_mean)}")
    print(f"      individual (context): vs B(3H)/B_d "
          f"{100 * (R_kin / R_3H - 1):+.2f}%,  vs B(3He)/B_d "
          f"{100 * (R_kin / R_3HE - 1):+.2f}%")

    moved_away = abs(R_kin - MEAN) > abs(NAIVE - MEAN)
    if moved_away:
        verdict = "KIN-WRONG-WAY"
    elif abs(dev_mean) <= 10.0:
        verdict = "KIN-CLOSES"
    else:
        verdict = "KIN-PARTIAL"
    banner(f"VERDICT (frozen tree): {verdict}")
    if verdict == "KIN-CLOSES":
        print(f"  R_kin lands {bin_of(dev_mean)} on the mirror mean: the "
              f"RATIO-MISS is adjudicated\n  as the dropped kinetic term; "
              f"the 13/3 booking is amended to CLOSED-BY-\n  COMPLETION with "
              f"the residual named ({dev_mean:+.2f}%).")
    elif verdict == "KIN-PARTIAL":
        print(f"  R_kin moves toward the mean but lands OFF "
              f"({dev_mean:+.2f}%): quantified and booked;\n  remaining legs "
              f"named: the EP-2 dictionary adoption, the bare-walker "
              f"stand-in.")
    else:
        print(f"  R_kin moves AWAY from the mean ({dev_mean:+.2f}%): a sharp "
              f"finding against the\n  kinetic-completion reading; booked raw.")
    print(f"\n  Poisons honored: no kappa; rungs and B2 frozen; E_odd never "
          f"absorbed;\n  no post-output convention change; open misses stay "
          f"open.")
    print("=" * W)
    print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}"
          f"   |   STATION VERDICT: {verdict}")
    print("=" * W)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
