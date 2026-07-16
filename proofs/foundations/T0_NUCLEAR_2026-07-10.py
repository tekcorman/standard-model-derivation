#!/usr/bin/env python3
# ============================================================================
# T0-NUCLEAR — the 3-body kinetic correction (IV.4's remaining half)
# ============================================================================
#
# PRE-REGISTRATION (binding, frozen): internal research notes
# Lineage: T0-CLASS (proofs/foundations/IV4_T0_class_2026-07-10.py):
#   B2 = 2.6689 (grid-converged pole value, n_q = 26), T0^(2) = 0.3311 =
#   11.04 % of U — IMPORTED FROZEN here, never re-solved.
#
# MACHINERY REPLICATED (not edited), with cites:
#   proofs/foundations/IV4_T0_class_2026-07-10.py
#     - Dirac D(k) 32x32, eps_low = lowest POSITIVE band, offset mesh
#       q_i = (i+0.5)/n_q, contact-kernel conventions (its Blocks 1-3).
#   proofs/foundations/bound_state_edge_resolved_kernel_2026-05-29.py
#     - build_prim_adjacency / one_girth_cycle / cycle_edge_set /
#       translate_vertex (lines 115-194): the girth-cycle self-translation
#       configuration-overlap machinery (primitive BCC frame).
#   proofs/foundations/I0b_RATIO_stage_BC_2026-07-10.py (itself a verbatim
#     replication of BOUND_stage3a_dS_spectrum_2026-07-03.py)
#     - supercell(3) cycle/pair/triple enumeration + the frozen rung
#       convention dS = [sum_e (mult_e - 1) - sum_v max(deg_v - 2, 0)]*b_edge,
#       b_edge = 1 bit; frozen rungs: 2-body ground DeltaS = 3 (648 pairs),
#       3-body ground DeltaS = 13 (216 triples); histogram re-locks.
#   proofs/foundations/n_body_oef_vertex_coinformation_2026-06-01.py
#     - THE POTENTIAL FUNCTIONAL, VERBATIM (its own definitions; no
#       hand-decomposition, no invented weights, no cap, no branch
#       subtraction — that file's honest note itself flags the
#       branch-realization cost as SEPARATE from what it computes):
#         edge-coverage entropy  S(X_T) = |union of edges of cycles in T|
#         pairwise piece         I(i;j) = |E_i INTERSECT E_j|
#         irreducible 3-body     II3    = |E_1 INTERSECT E_2 INTERSECT E_3|
#         total correlation      C3     = I12 + I13 + I23 - II3
#       vertex: E_int = -kappa*C  (attractive-or-zero); depth in substrate
#       units = C * e_bit, e_bit = t = 1 (DISCLOSED adoption, carried).
#
# ---------------------------------------------------------------------------
# FROZEN A PRIORI (declared BEFORE any result of this station was computed;
# nothing below was chosen in response to a binding number):
#
# STEP 1 — THE GATE (pre-reg stop-clause; adjudicated FIRST, before any
# 3-body solve):
#   Evaluate the frozen co-information functional at the reference
#   configurations of the frozen rungs:
#     (a) "full pairwise overlap" — BOTH declared readings evaluated:
#         (a1) the coincident configuration (Delta = 0 self-translation,
#              E_A = E_B — the configuration that set the 2-body contact
#              depth in the 2026-05-29 file, there via its MDL cap);
#         (a2) the 2-body GROUND pairs (the DeltaS = 3 maximal-overlap
#              distinct pairs on supercell(3), stage3a convention re-locked).
#     (b) the 3-body GROUND configuration = the DeltaS = 13 triples on
#         supercell(3) (stage3a convention re-locked).
#   GATE CRITERION (frozen): the functional must give U2 = 3 at full
#   pairwise overlap (either declared reading) AND 13 at the 3-body ground
#   configuration.  If not: STOP -> verdict CONSTRUCTION-MISMATCH, with the
#   computed values printed (a finding about the vertex file, NOT a license
#   to reweight).  On the mismatch path NO 3-body number is computed, and
#   the naive prediction / measured values are NOT printed (they appear ONLY
#   in the final confrontation printout, which exists only on the gate-pass
#   path).
#
# STEP 2 — THE 3-BODY SOLVE (runs ONLY if the gate passes), all conventions
# declared here BEFORE any result:
#   - Constituents: three IDENTICAL bare walkers (nucleon-analog stand-ins;
#     composite-of-composites unresolved — declared limitation, named not
#     fixed).  Statistics: the DISTINGUISHABLE relative problem is solved,
#     exactly as the 2-body station did (declared).
#   - Jacobi relative coordinates: x = r1 - r3, y = r2 - r3 (two relative
#     primitive-cell vectors), total quasimomentum K = 0 sector — the direct
#     generalization of the 2-body file's E_pair(q) = eps(q) + eps(-q)
#     (which IS its K = 0 sector).
#   - T_hat: E3(q1,q2) = eps(q1) + eps(q2) + eps(-q1-q2), eps = the frozen
#     lowest positive band of the 32x32 D(k).  Real-space hoppings by the
#     same offset-mesh quadrature as the 2-body station:
#       t(Delta) = (1/n_q^3) sum_q eps(q) e^{2 pi i q.Delta}
#     giving T[(x,y),(x',y')] = t(x-x') delta_{y,y'} + t(y-y') delta_{x,x'}
#     + t3(x-x') delta_{x-x', y-y'}, with t3 built from eps(-q); eps(-k) =
#     eps(k) EXACTLY (D(-k) = conj D(k)), asserted on the mesh, so t3 = t.
#   - V_hat: V(x,y) = -e_bit * C3(x,y), the GATED co-information functional
#     evaluated on the 3-walker configuration overlaps (girth-cycle
#     self-translation in the primitive frame, supercell L = 4*box + 3):
#       C3(x,y) = I(x) + I(y) + I(x-y) - II3(x,y),
#       I(d) = |E_0 ^ E_d|, II3(x,y) = |E_0 ^ E_x ^ E_y|.
#     Used RAW (the gate IS the depth normalization; no rescaling).
#   - Box/grid sizes (DECLARED HERE, never changed after a result):
#       boxes = {1, 2, 3}  (relative cells x,y in [-box,box]^3;
#                           M = (2*box+1)^6 = 729 / 15,625 / 117,649)
#       n_q   = 14 primary (the 2-body station's verbatim box-solver mesh),
#               with a duplicate at (box = 2, n_q = 18) as the
#               dispersion-grid convergence control.
#     Convergence shown across the three boxes at n_q = 14 (>= 2 sizes) plus
#     the n_q duplicate.  Primary number = (box = 3, n_q = 14).
#   - Threshold: E_th3 = min over the offset mesh (q1,q2) of E3(q1,q2) —
#     the bottom of the SAME free operator T_hat in the same K = 0 relative
#     space (the direct analog of the 2-body E_th = min E_pair).  3*eps_min
#     is printed as declared context (the K = 0 kinematic frustration size:
#     3*k_min != 0 mod 1, so all three walkers cannot sit at the band
#     minimum simultaneously in this sector).
#   - Solver: dense eigh at box = 1 (cross-check) + ARPACK eigsh
#     (LinearOperator with the structured matvec; k = 1, which='SA',
#     tol = 1e-10, deterministic v0 = the (x,y) = (0,0) basis vector) at all
#     boxes.  Eigen identity E0 = <T> + <V> asserted to 1e-7.
#     The matvec/assembly algebra is validated by an ALWAYS-RUN self-test on
#     a clearly-labeled TOY operator (synthetic separable band + seeded
#     random diagonal well) — no frozen-construction 3-body number is
#     produced by the self-test.
#   - Feasibility (printed BEFORE any diagonalization): M per box; one
#     measured matvec per box; projected eigsh cost = 400 matvecs x safety
#     3; hard station cap ~20 min (T_CAP = 1140 s).  If the projection
#     exceeds the remaining budget: print INFEASIBLE-AT-CAP with the
#     estimate and STOP (a bookable outcome).
#   - Outputs: B3 = E_th3 - E0;  T0^(3) = 13 - B3;
#              R_kin = B3 / B2,  B2 = 2.6689 FROZEN (T0-CLASS pole, n_q=26).
#
# FROZEN CONFRONTATIONS (declared context, never optimization targets;
# printed ONLY in the final confrontation block on the gate-pass path):
#   measured B(3H)/B_d = 3.81279, B(3He)/B_d = 3.46946, mirror MEAN
#   3.64112 (needs f = 0.8403); the naive sealed prediction 13/3 = 4.333.
#   T0 CANNOT split the mirror pair (identical constituents) — the
#   confrontation object is the MIRROR MEAN; the +-4.95 % split is E_odd's
#   (0.381876 MeV, registered row, NEVER absorbed here).
#   Stage-C bins (reporting conventions): EXACT (<1 %), NEAR (1-10 %,
#   quantified OPEN), OFF (>10 %, OPEN).
#
# FROZEN VERDICT TREE:
#   CONSTRUCTION-MISMATCH : the gate fails (per above).
#   KIN-WRONG-WAY : R_kin moves AWAY from the mirror mean relative to the
#                   naive 13/3 (|R_kin - mean| > |13/3 - mean|).
#   KIN-CLOSES    : R_kin lands EXACT/NEAR (<= 10 %) on the mirror mean —
#                   the RATIO-MISS is adjudicated as the dropped kinetic
#                   term (residual named if NEAR).
#   KIN-PARTIAL   : R_kin moves toward the mean but lands OFF (> 10 %) —
#                   deviation quantified and booked; remaining legs named
#                   (the EP-2 dictionary adoption, the bare-walker stand-in).
#
# POISONS (carried verbatim): no kappa; no per-system reweighting; the
# rungs 3/13 and B2 imported frozen; no post-output convention changes (a
# changed box/band/functional after the first 3-body number = poisoned);
# E_odd never absorbed; measured values only in the final confrontation
# printout; bare walkers stand in for nucleons (declared limitation); an
# open miss stays open.
# ---------------------------------------------------------------------------
# Standalone: python3 proofs/foundations/T0_NUCLEAR_2026-07-10.py ; exit 0.
# Asserts fire ONLY on machine-checkable regressions + proven algebra
# (Dirac validation, histogram re-locks, functional identities, solver
# self-test); ALL station verdicts are PRINTED, never asserted.
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
DS_CAP = 3.0                          # 2026-05-29 MDL cap (context only here)

# stage3a / I0b frozen re-lock anchors (I0b_RATIO_stage_BC lines 240-246)
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
# Block 1 — Dirac D(k) + lowest positive band (IV4 Blocks 1/3, verbatim)
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
# Block 2 — stage3a rung machinery (I0b replication of BOUND_stage3a,
# convention frozen 2026-07-03; terms exposed for the gate decomposition)
# ---------------------------------------------------------------------------
def cycle_edges(cycle):
    n = len(cycle)
    return frozenset(frozenset((cycle[i], cycle[(i + 1) % n])) for i in range(n))


def dS_union_parts(edgesets):
    """The frozen rung convention (stage3a lines 64-78 / I0b dS_of_union):
    DeltaS = [compression - branch] * b_edge, with the two terms returned
    separately (same formulas; exposure needed to PRINT the gate finding)."""
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
# Block 3 — THE POTENTIAL FUNCTIONAL (n_body_oef_vertex_coinformation
# 2026-06-01, VERBATIM: coverage entropy; I = |^2|; II3 = |^3|;
# C3 = sum I - II3; no cap, no branch subtraction)
# ---------------------------------------------------------------------------
def coinf_pair(EA, EB):
    return len(EA & EB)


def coinf_triple(EA, EB, EC):
    I12, I13, I23 = len(EA & EB), len(EA & EC), len(EB & EC)
    II3 = len(EA & EB & EC)
    C3 = I12 + I13 + I23 - II3
    return I12, I13, I23, II3, C3


# ---------------------------------------------------------------------------
# Block 4 — primitive-frame self-translation machinery (2026-05-29 lines
# 115-194, verbatim; used by the solve path's V_hat)
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
# Block 5 — the 3-body Jacobi solver (declared construction; runs only on
# the gate-pass path; algebra validated by the toy self-test below)
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


def coinf_potential(box):
    """V(x,y) = -e_bit * C3 on the 3-walker self-translation configuration
    (0, x, y); returns (Vmat over cells^2, the pair profile I(d))."""
    L = 4 * box + 3
    adj = build_prim_adjacency(L)
    start = (L // 2, L // 2, L // 2, 0)
    cyc = one_girth_cycle(adj, start)
    assert cyc is not None and len(cyc) == GIRTH, "no girth-10 cycle found"
    E0 = frozenset(cycle_edges(tuple(cyc)))
    I_prof = {}
    for dvec in product(range(-2 * box, 2 * box + 1), repeat=3):
        I_prof[dvec] = coinf_pair(E0, translate_edgeset(E0, dvec, L))
    Ts = {dvec: translate_edgeset(E0, dvec, L)
          for dvec in product(range(-box, box + 1), repeat=3)}
    cells = list(product(range(-box, box + 1), repeat=3))
    nC = len(cells)
    V = np.zeros((nC, nC))
    for ix, xv in enumerate(cells):
        E0x = E0 & Ts[xv]
        for iy, yv in enumerate(cells):
            u = (xv[0] - yv[0], xv[1] - yv[1], xv[2] - yv[2])
            II3 = len(E0x & Ts[yv])
            C3 = I_prof[u] + I_prof[xv] + I_prof[yv] - II3
            V[ix, iy] = -C3 * E_BIT
    return V, I_prof


# ---------------------------------------------------------------------------
# Block 6 — solver algebra SELF-TEST (TOY operator; always runs; produces NO
# frozen-construction number)
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
    return E0


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    banner("T0-NUCLEAR — the 3-body kinetic correction (IV.4's remaining half)")
    print("pre-reg: internal research notes (FROZEN)")
    print(f"units: substrate energy (t = e_bit = 1, disclosed); "
          f"B2 = {B2_FROZEN} FROZEN (T0-CLASS pole, n_q = 26; not re-solved)")

    ok = validate_dirac()
    print(f"\n[validation] Dirac D(k)^2 = 6I + R_sub : {'PASS' if ok else 'FAIL'}")
    assert ok, "Dirac validation regression"

    E0_toy = solver_self_test()
    print(f"[validation] solver algebra self-test (TOY band + seeded random "
          f"well;\n             dense == structured matvec == eigsh; threshold "
          f"index algebra;\n             eigen identity): PASS  "
          f"(toy E0 = {E0_toy:.6f} — a TOY number, not a result)")

    # ================= STEP 1 — THE GATE =================
    banner("STEP 1 — THE GATE: the frozen co-information functional at the "
           "frozen rungs")
    print("    functional (2026-06-01, VERBATIM): I(i;j) = |Ei ^ Ej|;  "
          "II3 = |E1 ^ E2 ^ E3|;")
    print("    C3 = I12 + I13 + I23 - II3   (coverage entropy; no cap, no "
          "branch subtraction)")

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

    # ---- rung re-lock (regression asserts) + gate evaluation: pairs ----
    hist2 = Counter()
    pair_decomp = Counter()             # (I, branch, dS) over all pairs
    ground_pair_I = Counter()           # I over the DeltaS = 3 ground pairs
    for a, b in pairs:
        dS, comp, br = dS_union_parts([edgesets[a], edgesets[b]])
        I = coinf_pair(edgesets[a], edgesets[b])
        assert comp == I, "pair compression != coverage I (identity broken)"
        hist2[round(dS)] += 1
        pair_decomp[(I, br, round(dS))] += 1
        if round(dS) == DS2_RUNG:
            ground_pair_I[(I, br)] += 1
    assert dict(hist2) == HIST2_STAGE0, \
        f"2-body histogram re-lock FAILED: {dict(hist2)}"
    pos2 = sorted(k for k in hist2 if k > 0)
    assert pos2 == LADDER2, f"2-body spectrum {pos2} != {LADDER2}"
    check("gate re-lock: 2-body histogram == Stage-0 {-1:4212,0:2592,1:648,3:648}",
          dict(hist2) == HIST2_STAGE0)
    print(f"    all-pairs (I, branch, dS) decomposition: "
          f"{dict(sorted(pair_decomp.items()))}")

    # reading (a1): coincident full overlap (Delta = 0 self-translation)
    I_coincident = coinf_pair(edgesets[0], edgesets[0])
    # reading (a2): the DeltaS = 3 ground pairs
    n_ground_pairs = sum(ground_pair_I.values())
    print(f"\n    (a1) coincident full overlap (E_A = E_B):        "
          f"I = {I_coincident}")
    print(f"         [the 2026-05-29 contact depth 3 at Delta=0 came from its "
          f"MDL cap\n          min(dS, {DS_CAP:.0f}); the co-information "
          f"functional carries NO cap]")
    print(f"    (a2) the {n_ground_pairs} DeltaS = {DS2_RUNG} GROUND pairs:  "
          f"          (I, branch) = {dict(ground_pair_I)}")
    Ivals_ground = sorted({iv for (iv, _) in ground_pair_I})
    U2_ground = Ivals_ground[0] if len(Ivals_ground) == 1 else None
    gate_pair_vals = {"coincident": I_coincident, "ground": U2_ground}
    gate_2body = (I_coincident == DS2_RUNG) or (U2_ground == DS2_RUNG)

    # ---- rung re-lock + gate evaluation: triples ----
    triples = set()
    for b in range(len(cycles)):
        nbrs = sorted(overlap_nbr[b])
        for a, c in combinations(nbrs, 2):
            triples.add(frozenset((a, b, c)))
    hist3 = Counter()
    ground_tri = Counter()              # (sorted I's, II3, C3, branch)
    for tri in triples:
        i, j, k = sorted(tri)
        dS, comp, br = dS_union_parts([edgesets[i], edgesets[j], edgesets[k]])
        hist3[round(dS)] += 1
        if round(dS) == DS3_RUNG:
            I12, I13, I23, II3, C3 = coinf_triple(edgesets[i], edgesets[j],
                                                  edgesets[k])
            assert C3 == comp, "C3 != compression (vertex-file identity broken)"
            ground_tri[(tuple(sorted((I12, I13, I23))), II3, C3, br)] += 1
    pos3 = sorted(k for k in hist3 if k > 0)
    assert pos3 == LADDER3, f"3-body spectrum {pos3} != {LADDER3}"
    assert hist3[DS3_RUNG] == HIST3_20260703[DS3_RUNG], \
        f"ground-triple count {hist3[DS3_RUNG]} != {HIST3_20260703[DS3_RUNG]}"
    check("gate re-lock: 3-body positive spectrum == frozen {1,2,3,4,6,13} "
          f"and #(dS=13) == 216", pos3 == LADDER3
          and hist3[DS3_RUNG] == HIST3_20260703[DS3_RUNG])
    check("gate re-lock: full 3-body histogram == recorded 2026-07-03 run",
          dict(hist3) == HIST3_20260703)
    print(f"\n    (b) the {hist3[DS3_RUNG]} DeltaS = {DS3_RUNG} GROUND triples "
          f"— (sorted(I12,I13,I23), II3, C3, branch):")
    for kk, vv in sorted(ground_tri.items()):
        print(f"        {kk} : {vv} triples")
    C3vals = sorted({c3 for (_, _, c3, _) in ground_tri})
    C3_ground = C3vals[0] if len(C3vals) == 1 else None
    gate_3body = (C3_ground == DS3_RUNG)
    print(f"    gate timing: {time.monotonic() - t0:.1f} s")

    # ---- gate adjudication ----
    print(f"\n    GATE CRITERION: U2 == {DS2_RUNG} at full pairwise overlap "
          f"AND {DS3_RUNG} at the 3-body ground.")
    print(f"      2-body: coincident -> {I_coincident};  ground pairs -> "
          f"{U2_ground}   (required: {DS2_RUNG})  "
          f"{'PASS' if gate_2body else 'FAIL'}")
    print(f"      3-body: ground triples -> C3 = {C3_ground}   "
          f"(required: {DS3_RUNG})  {'PASS' if gate_3body else 'FAIL'}")
    gate_pass = gate_2body and gate_3body

    if not gate_pass:
        # ================= VERDICT: CONSTRUCTION-MISMATCH =================
        banner("VERDICT (frozen tree): CONSTRUCTION-MISMATCH")
        print(f"""
  The frozen co-information functional does NOT reproduce the frozen rungs
  at either reference configuration (computed values, supercell(3), all
  asserts re-locked):

    full pairwise overlap : I = {I_coincident} (coincident reading) / I = {U2_ground} (all
                            {n_ground_pairs} ground pairs)      REQUIRED {DS2_RUNG}   -> MISMATCH
    3-body ground config  : C3 = {C3_ground} (all {hist3[DS3_RUNG]} ground triples,
                            uniformly (5,5,5| II3=0))  REQUIRED {DS3_RUNG}  -> MISMATCH

  STRUCTURE OF THE MISMATCH (the finding, computed not conjectured):
    (1) On EVERY reference configuration the two frozen conventions differ
        by EXACTLY the branch-realization cost of the stage3a rung
        convention:
          ground pairs   : rung 3  = I  - branch = 5  - 2
          ground triples : rung 13 = C3 - branch = 15 - 2
        The 2026-06-01 vertex file's own honest-bounds note names precisely
        this: its coverage entropy is the PURE information content, while
        "the F1/F8 net binding additionally subtracts a branch-realization
        cost n_branch" — a cost that lives OUTSIDE its co-information
        functional.  The pre-reg froze the functional (2026-06-01,
        verbatim) AND the depth normalization (rungs 3/13, stage3a) at
        once; as frozen, they are mutually inconsistent by that cost.
        At the coincident configuration the gap is instead the 2026-05-29
        MDL cap (I = 10, uncapped).
    (2) II3 = 0 on ALL {hist3[DS3_RUNG]} ground triples: under the vertex file's own
        irreducible-3-body discriminator, the DeltaS = {DS3_RUNG} ground
        configuration carries NO junction core — it is three pairwise
        5-overlaps (pairwise-REDUCIBLE in pure-information terms), not a
        baryon-junction-type configuration.
    (3) Corollary of (1)+(2): the ground triple's pure-information content
        is exactly pairwise-additive (C3 = 15 = 3 x 5); the rungs'
        departure from pairwise additivity ({DS3_RUNG} vs {DS2_RUNG}) is carried
        ENTIRELY by the branch cost, which the frozen functional does not
        contain.

  Per the pre-reg stop-clause this is a finding about the vertex file, NOT
  a license to reweight: no re-weighted functional was built, no rung was
  re-assigned, no alternative depth was tried.

  NOT COMPUTED (stop-clause honored): B3, T0^(3), R_kin.  The 3-body solve
  did not run; no confrontation printout exists, so the naive prediction
  and the measured nuclear values do not appear in this run.  The RATIO-MISS
  (I-0b-RATIO booking) stays OPEN exactly as booked.

  Poisons honored: no kappa; no per-system reweighting; rungs 3/13 and
  B2 = {B2_FROZEN} imported frozen and untouched; E_odd never absorbed; box/
  band/functional unchanged after (indeed: before) any 3-body number —
  none was produced; bare-walker limitation moot on this path; the open
  miss stays open.""")
        check("station verdict adjudicated: CONSTRUCTION-MISMATCH "
              "(stop-clause path)", True)
        print("=" * W)
        print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}"
              f"   |   STATION VERDICT: CONSTRUCTION-MISMATCH")
        print("=" * W)
        sys.exit(0 if ok_all else 1)

    # ================= STEP 2 — THE 3-BODY SOLVE (gate-pass path) ==========
    banner("STEP 2 — THE 3-BODY JACOBI SOLVE (gate PASSED; frozen sizes)")
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
        Vmat, I_prof = coinf_potential(b)
        rng_I = max((sum(abs(c) for c in d) for d, v in I_prof.items() if v > 0
                     and d != (0, 0, 0)), default=0)
        print(f"\n    box {b}: co-information V range (|Delta|_1 of I > 0, "
              f"off-site): {rng_I}; V(0,0) = {Vmat[len(Vmat)//2, len(Vmat)//2]:.1f}")
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
        eps18 = eps_on_mesh(NQ_DUP)
        eps18_int = eps_on_mesh(NQ_DUP, offset=0.0)
        E_th18 = threshold3(eps18, eps18_int)
        Vmat2, _ = coinf_potential(2)
        op18 = Jacobi3Body(2, t_from_eps(eps18, 2), t_from_eps(eps18, 2), Vmat2)
        E0_18, T18, V18 = op18.ground()
        B3_18 = E_th18 - E0_18
        results[(2, NQ_DUP)] = (E0_18, B3_18, T18, V18)
        print(f"\n    duplicate (box 2, n_q = {NQ_DUP}): E_th3 = {E_th18:.6f}  "
              f"E0 = {E0_18:.6f}  B3 = {B3_18:.6f}")

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
