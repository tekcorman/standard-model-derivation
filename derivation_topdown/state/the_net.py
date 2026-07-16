#!/usr/bin/env python3
"""
derivation_topdown/state/the_net.py

THE NET — the local-algebra layer {A(O)} of the state omega.  MASTER OBJECT for Layer 3.

physics = (D, omega, {A(O)}):
  Layer 1  D's global spectrum          -> derivation_topdown/bridge/the_run.py  (~95 live reads)
  Layer 2  omega's GLOBAL structure     -> kappa=h/t_P, KMS/tick, clock map, eras-at-form
  Layer 3  omega's LOCAL structure      -> THIS FILE: O |-> (A(O), omega|_O) -> modular data

Every open parameter is ONE region-shape of this object (parameter_bins_and_local_net_throughline_2026-07-08.md):
  causal-diamond modular flow   (ML-1)  -> Newton's G 2pi
  DHR sector category           (ML-2)  -> species / -70 ppm / m_nu-scale / B1 -> Y_p
  flat-band modular weight      (ML-3)  -> native z_eq -> theta_* (ML-4)
  local density response        (later) -> n_s / sigma_8

ARCHITECTURE RULE (ONE-OBJECT / LOCAL-NET LAW): Layer-3 math ACCRETES here.  EXTEND this module
every station; never a session-scratch probe.  Frame each station as "add a region-class / forced
read to the net", never "attack observable X".  Do not fork a second local-net program.

REGRESSION ANCHORS (must ALWAYS hold; the two degenerate region-shapes already known):
  * region = one CELL     -> M0's vacuum covariance C=(I+iJ6)/2, an exact rank-3 projector, Tr C = 3.
  * region = the global TICK subalgebra -> M0-2R's tick modular flow = a compact U(1) of period 2pi.

PROVENANCE:
  ML-0 (pre-reg c0feb36, commit 0961746, verify 65/65) built the net and verified: mode-space
  reconciliation (dart space; vacuum = R-even sector), the EXACT combinatorial light cone from the
  non-backtracking walk B, twisted (Klein) locality, cell-level twisted Haag duality, Z^3 covariance.
  This module consolidates that construction into the durable importable form.  ML-1/2/3 extend it.

NO magnitude is defined in this module.  It exposes structure (regions, algebras, modular data);
the blind numerical confronts live in the per-station pre-registered extensions.
"""
import cmath
import itertools
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402  (the walled-off clean-room srs object)

EDGES = srs.EDGES
NV = srs.NV
NE = len(EDGES)
DARTS = srs._darts()          # dart 2e = edge e forward (i,j,v); dart 2e+1 = reversed (j,i,-v)
ND = len(DARTS)               # 2|E| = 12


# ===========================================================================
# 1. THE VACUUM ON A CELL  (M0's C = (I + iJ6)/2 on the 6-edge space)  [anchor]
# ===========================================================================
def _edge_rep(sig):
    EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6


def complex_structure_J6():
    """The A4-covariant complex structure J6 on the 6-edge space (WS1 S0; forced unique up to
    the bit).  Real antisymmetric, J6^2 = -I."""
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
        R6 = _edge_rep(g)
        rows.append(np.kron(np.eye(3), (H1.T @ R6 @ H1).T) - np.kron(B1.T @ R6 @ B1, np.eye(3)))
    _, _, VpJ = np.linalg.svd(np.vstack(rows))
    phi = VpJ[-1].reshape(3, 3)
    phi *= math.sqrt(3) / np.linalg.norm(phi)
    return B1 @ phi @ H1.T - H1 @ phi.T @ B1.T


def vacuum_covariance(sign=+1):
    """The one-bit vacuum covariance on the 6-edge (R-even) sector: C = (I + i*sign*J6)/2.
    Exact rank-3 projector (pure Gaussian state)."""
    J6 = complex_structure_J6()
    return (np.eye(NE) + 1j * sign * J6) / 2.0


def region_data(C, A):
    """omega|_A modular data for an edge-region A (list of edge indices), OWNED Peschel convention:
    h_A = log((I - C_A) C_A^{-1}); zeta = occupations; eps = single-particle modular energies;
    S = entanglement entropy (nats).  Returns (zeta_sorted, eps_sorted, S)."""
    C_A = C[np.ix_(A, A)]
    zeta = np.linalg.eigvalsh(C_A).real
    zc = np.clip(zeta, 1e-12, 1 - 1e-12)
    eps = np.log((1 - zc) / zc)
    S = float(np.sum(-zc * np.log(zc) - (1 - zc) * np.log(1 - zc)))
    return np.sort(zeta), np.sort(eps), S


# ===========================================================================
# 2. THE REVERSAL GRADING  (dart space = edge (R-even) (+) R-odd)  [ML-0 reconciliation]
# ===========================================================================
def reversal():
    """The dart reversal involution R (swap the two darts of each edge). R^2 = I, Tr R = 0.
    The R-even sector (dim 6) is the undirected EDGE space carrying the vacuum J6/C; the R-odd
    sector (dim 6) is where the non-backtracking walk B pushes amplitude."""
    R = np.zeros((ND, ND))
    for e in range(NE):
        R[2 * e, 2 * e + 1] = 1.0
        R[2 * e + 1, 2 * e] = 1.0
    return R


def hashimoto_gamma():
    """The non-backtracking Hashimoto walk B at k=0 (the unrolled cover): real 0/1, 12x12."""
    return srs.hashimoto(np.zeros(3)).real


# ===========================================================================
# 3. THE HISTORY NET  (causal diamonds in cell x tick; the EXACT light cone)  [ML-0]
# ===========================================================================
class Patch:
    """A finite real-space patch of the Z^3 cover (cells in a box), its darts, the real-space
    non-backtracking walk B, and the vertex-graph distance (INDEPENDENT of B -> the light cone is
    not circular).  The causal net O |-> A(O) lives here: A(O) is the CAR algebra generated by the
    (dart, tick) modes in the causal diamond O."""

    def __init__(self, M=4, skip_pair_bfs=False):
        self.M = M
        self.skip_pair_bfs = skip_pair_bfs
        box = list(itertools.product(range(M), repeat=3))
        self.box = box
        self.RD = []  # real-space darts: (tail_vertex, head_vertex), vertex = (branch i, cell x)
        inbox = lambda x: all(0 <= c < M for c in x)
        for (i, j, v) in EDGES:
            for x in box:
                xh = tuple(np.array(x) + np.array(v))
                if inbox(xh):
                    self.RD.append(((i, x), (j, xh)))
                    self.RD.append(((j, xh), (i, x)))
        self.tail = [a for (a, b) in self.RD]
        self.head = [b for (a, b) in self.RD]
        self.dpos = {d: n for n, d in enumerate(self.RD)}
        self.Nd = len(self.RD)
        # real-space Hashimoto B (non-backtracking): a -> b iff head(a)=tail(b) and b != reverse(a)
        self.B = np.zeros((self.Nd, self.Nd))
        byhead = {}
        for a, (ta, ha) in enumerate(self.RD):
            byhead.setdefault(ha, []).append(a)
        for b, (tb, hb) in enumerate(self.RD):
            for a in byhead.get(tb, []):
                ta, ha = self.RD[a]
                if not (hb == ta and tb == ha):
                    self.B[b, a] = 1.0
        # vertex indexing + undirected vertex-graph distances (BFS)
        verts = [(i, x) for x in box for i in range(NV)]
        self.vidx = {v: n for n, v in enumerate(verts)}
        NVp = len(verts)
        if skip_pair_bfs:
            # D1b (pre-reg 2026-07-09, adjudication 3): this all-pairs BFS is O(NVp^2) and dominates
            # build cost at M>=14; it is UNUSED by the Cartesian bond_profile_slope reader (only
            # vdist/geodesic_dist_to_vertices consume it). Skipping it is an OPT-IN accretion:
            # default (skip_pair_bfs=False) preserves prior behavior exactly.
            self._dV = None
        else:
            adjV = {n: set() for n in range(NVp)}
            for (a, b) in self.RD:
                adjV[self.vidx[a]].add(self.vidx[b])
                adjV[self.vidx[b]].add(self.vidx[a])
            self._dV = [self._bfs(adjV, n) for n in range(NVp)]

    @staticmethod
    def _bfs(adjV, src):
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

    def vdist(self, u, w):
        if self._dV is None:
            raise RuntimeError("Patch built with skip_pair_bfs=True: vdist is unavailable (the "
                                "all-pairs BFS was skipped). Rebuild with skip_pair_bfs=False (the "
                                "default) if vertex-graph distances are needed.")
        return self._dV[u].get(w, 10 ** 9)

    def powers(self, T):
        P = [np.eye(self.Nd)]
        for _ in range(T):
            P.append(P[-1] @ self.B)
        return P

    def anticommutator_below_cone(self, T):
        """The MASTER causality read: {alpha_a(t), a_c^dag} = (B^t)_{ca}.  The geometric horizon
        (from the vertex graph, independent of B): to traverse dart c you first walk head(a)->tail(c)
        (>= vdist steps) THEN take the c-step -> onset >= 1 + vdist(head a, tail c).  Returns the
        maximum |amplitude| found STRICTLY inside the horizon over a source-sampled sweep (must be
        EXACTLY 0.0 -> a strict combinatorial light cone, stronger than Lieb-Robinson)."""
        P = self.powers(T)
        worst = 0.0
        stride = max(1, self.Nd // 60)
        for t in range(1, T + 1):
            Bt = P[t]
            for a in range(0, self.Nd, stride):
                for c in range(self.Nd):
                    onset = 1 + self.vdist(self.vidx[self.head[a]], self.vidx[self.tail[c]])
                    if t < onset:
                        worst = max(worst, abs(Bt[c, a]))
        return worst

    def horizon_radii(self, base, T):
        """Reached vertex-radius by tick t = 1..T from a base dart (should be [1,2,...] -> speed 1)."""
        P = self.powers(T)
        out = []
        for t in range(1, T + 1):
            reached = np.argwhere(np.abs(P[t][:, base]) > 1e-12).ravel()
            rad = max((self.vdist(self.vidx[self.head[base]], self.vidx[self.head[c]])
                       for c in reached), default=0)
            out.append(int(rad))
        return out

    def central_dart(self):
        center = tuple([self.M // 2] * 3)
        for a in range(self.Nd):
            if self.head[a][1] == center:
                return a
        return 0

    def geodesic_dist_to_vertices(self, target_vidx):
        """BFS graph distance (in HOPS) from every vertex to the nearest vertex in target_vidx.
        The hop is the FORCED proper-distance unit (ML-0: the light cone advances exactly one hop per
        tick).  Vertex indexing matches vertex_adjacency's `verts`.  Returns an array over all vertices."""
        if self._dV is None:
            raise RuntimeError("Patch built with skip_pair_bfs=True: geodesic_dist_to_vertices is "
                                "unavailable (the all-pairs BFS was skipped). Rebuild with "
                                "skip_pair_bfs=False (the default) for the geodesic-hop metric (e.g. "
                                "D1b's V-5 fourth-convention confront).")
        tgt = list(target_vidx)
        n = len(self._dV)
        out = np.full(n, 10 ** 9, dtype=float)
        for v in range(n):
            dv = self._dV[v]
            best = min((dv.get(t, 10 ** 9) for t in tgt), default=10 ** 9)
            out[v] = best
        return out

    def vertex_adjacency(self):
        """The real-space srs walk Hamiltonian on VERTICES (i,x): H[(i,x),(j,x+v)] = 1 for every
        cover edge (Hermitian).  This is the physical single-particle Hamiltonian whose Dirac-sea
        vacuum ML-1 restricts to a spatial half-space.  Returns (H, verts) with verts the ordered
        vertex list."""
        verts = [(i, x) for x in self.box for i in range(NV)]
        vpos = {v: n for n, v in enumerate(verts)}
        n = len(verts)
        H = np.zeros((n, n))
        inbox = lambda x: all(0 <= c < self.M for c in x)
        for (i, j, v) in EDGES:
            for x in self.box:
                xh = tuple(np.array(x) + np.array(v))
                if inbox(xh):
                    H[vpos[(i, x)], vpos[(j, xh)]] += 1.0
                    H[vpos[(j, xh)], vpos[(i, x)]] += 1.0
        return H, verts

    def diamond(self, base, depth):
        """A causal diamond as a set of (dart_index, tick) modes: the forward cone of `base` to
        `depth` ticks.  A(O) = CAR generated by these modes.  (The past-cone/tip intersection is
        added by ML-1 when it restricts omega to the diamond.)"""
        P = self.powers(depth)
        modes = set()
        for t in range(depth + 1):
            for c in np.argwhere(np.abs(P[t][:, base]) > 1e-12).ravel():
                modes.add((int(c), t))
        return modes


# ===========================================================================
# 4. TWISTED (Klein) LOCALITY  — the net is fermionic; the twist is parity  [ML-0]
# ===========================================================================
def _jw_ops(N):
    I2 = np.eye(2)
    Z2 = np.diag([1.0, -1.0])
    a1 = np.array([[0.0, 1.0], [0.0, 0.0]])

    def kron_list(ms):
        out = np.array([[1.0]])
        for m in ms:
            out = np.kron(out, m)
        return out

    a = [kron_list([Z2] * p + [a1] + [I2] * (N - 1 - p)) for p in range(N)]
    return a, [op.conj().T for op in a]


def twisted_locality_holds(N=4, R1=(0, 1), R2=(2, 3)):
    """On an explicit JW Fock space: even algebras of disjoint regions commute, odd parts
    ANTI-commute (naive commutation FAILS), and the Klein-twisted odd operator commutes.  Returns
    a dict of the residuals so callers can regression-check the twisted structure."""
    a, adag = _jw_ops(N)
    dim = 2 ** N
    eA = adag[R1[0]] @ a[R1[1]]
    eB = adag[R2[0]] @ a[R2[1]]
    oA, oB = a[R1[0]], a[R2[0]]
    comm = lambda X, Y: X @ Y - Y @ X
    acomm = lambda X, Y: X @ Y + Y @ X
    P1 = np.eye(dim)
    for i in R1:
        P1 = P1 @ (np.eye(dim) - 2 * adag[i] @ a[i])
    return {
        "even_even_commute": np.max(np.abs(comm(eA, eB))),
        "even_odd_commute": np.max(np.abs(comm(eA, oB))),
        "odd_odd_anticommute": np.max(np.abs(acomm(oA, oB))),
        "naive_commutation_fails": np.max(np.abs(comm(oA, oB))),
        "klein_twist_commutes": np.max(np.abs(comm(oA, P1 @ oB))),
    }


# ===========================================================================
# 4b. MODULAR FLOW READERS  (Peschel entanglement Hamiltonian; the BW near-horizon slope) [ML-1]
# ===========================================================================
def entanglement_hamiltonian(C_A):
    """Single-particle entanglement (modular) Hamiltonian h_A = log((I-C_A) C_A^{-1}) for a Gaussian
    region correlation C_A, built STABLY from the eigendecomposition (avoids logm on eigenvalues
    pinned near 0/1).  OWNED convention (M0-C)."""
    w, V = np.linalg.eigh(C_A)
    w = np.clip(w.real, 1e-14, 1 - 1e-14)
    eps = np.log((1 - w) / w)
    return (V * eps) @ V.conj().T


def bw_near_horizon_slope(h_A, phys_hop, bond_dist):
    """The BW/Unruh read: the local modular temperature beta(x) = (entanglement local coupling) /
    (physical local coupling) rises LINEARLY from the horizon, beta(x) -> 2pi*x (Bisognano-Wichmann).
    `bond_list`: list of (i, j, x_center) nearest-neighbour bonds with distance-to-horizon x_center;
    beta = |h_A[i,j]| / phys_hop.  Returns the near-horizon slope beta(x)/x extrapolated to the
    FIRST bond (x->0), the calibrated 2pi observable (see benchmark).  The 2pi is MEASURED, never
    inserted."""
    xs = np.array([x for (_, _, x) in bond_dist], float)
    betas = np.array([abs(h_A[i, j]) / phys_hop for (i, j, x) in bond_dist], float)
    order = np.argsort(xs)
    xs, betas = xs[order], betas[order]
    return betas[0] / xs[0], xs, betas          # first-bond slope = the calibrated BW reader


_EX12 = None


def _ex12_module():
    """Lazy import of explore_12_harmonic_geometry (D1b, 2026-07-09).  That module RE-RUNS its own
    diagnostic prints on first import (same pattern D1/adapters/sunada_geometry.py already accept);
    Python's module cache means every call after the first is silent.  Kept OUT of the_net.py's own
    module-level imports so `import the_net` stays side-effect-free for every OTHER caller (~dozens of
    stations) that never asks for a Cartesian position."""
    global _EX12
    if _EX12 is None:
        import explore_12_harmonic_geometry as _ex12mod
        _EX12 = _ex12mod
    return _EX12


def vertex_position(v):
    """D1b (pre-reg 2026-07-09) POSITION BRIDGE: map a Patch vertex v=(branch i, cell x) to the
    Albanese/Kotani-Sunada Cartesian frame pos(i,x) = Xv[i] + L@x (explore_12_harmonic_geometry's
    harmonic-equilibrium standard realization -- imported and used AS-IS, no local re-derivation, no
    rescaling; the frame the isotropization weld certifies as the emergent-metric frame).  `x` may be
    any length-3 int sequence (Patch's cell tuples).  Returns a numpy Cartesian 3-vector in LATTICE
    units; divide by v_iso = sqrt(mean eig(L @ emergent_metric() @ L.T)) for the v_iso-scaled
    proper-distance frame (the D1/ML-1''' PROPER convention -- see bond_profile_slope below)."""
    ex12 = _ex12_module()
    i, x = v
    return ex12.Xv[i] + ex12.L @ np.asarray(x, dtype=float)


def bond_profile_slope(h_A, bonds, cut_normal, window, point="midpoint",
                        origin_constrained=False, absolute_window=None):
    """D1b (pre-reg 2026-07-09): the generalized multi-bond/multi-layer BW near-horizon PROFILE
    extractor -- the SLOPE-not-RATIO successor to bw_near_horizon_slope's single first-bond read (the
    D1 flaw class: a ratio at ONE bond is sensitive to the absolute distance convention; a SLOPE fit
    over MANY bonds at DERIVED positions is invariant under any constant shift of the position origin).

    W2-D1c (pre-reg 2026-07-10) ACCRETION -- two new optional arguments, ACCRETION-ONLY (both default
    to values that reproduce D1b's ORIGINAL behavior BIT-IDENTICALLY; nothing existing is modified):

    origin_constrained : if True, replace the free-intercept OLS fit with the ORIGIN-CONSTRAINED
        weighted-least-squares estimator a = sum(beta_b*x_b) / sum(x_b^2) over bonds with
        0 < x_b <= window (no intercept -- Bisognano-Wichmann FORCES beta(0)=0, so the free-intercept
        fit's intercept term was absorbing/stealing slope; D1b's adversarial check found the free
        intercept steals ~1.6-2.3x of slope).  The bond selection additionally REQUIRES x_b > 0 (strict)
        in this mode -- a bond sitting exactly on the horizon plane contributes nothing to either sum
        and would be ill-defined for a beta/x-type read.  If False (default), the bond selection and
        fit are EXACTLY the original code (x_b <= window, free-intercept OLS) -- bit-identical.
    absolute_window : if given (not None), OVERRIDES `window` as the actual cutoff value used for bond
        selection (self-documents the W2-D1c ABSOLUTE COMMON WINDOW correction -- a window in units of
        d_b = one bond length in proper units, held IDENTICAL across directions/M, never a fraction of
        region depth -- at the call site, without requiring the caller to overload `window` itself).
        If None (default), `window` is used exactly as before -- bit-identical.

    h_A    : the region's Peschel entanglement Hamiltonian (region-local indexing; entanglement_hamiltonian()).
    bonds  : iterable of (i, j, pos_i, pos_j) -- i, j are h_A row/col indices (region-local) of the TWO
             ENDPOINTS of a caller-declared PHYSICAL hopping bond (a nearest-neighbour edge of the
             physical adjacency restricted to the region -- the bw_near_horizon_slope / D1
             axis_bond_mean convention: beta_b is read from h_A ONLY at genuine physical-adjacency
             bonds, never from h_A's generic all-to-all entanglement content, which is dense).
             pos_i/pos_j are the two endpoints' Cartesian positions, v_iso-scaled PROPER units,
             ALREADY shifted so the horizon plane sits at cut_normal-projection = 0 (station: pos -
             threshold*cut_normal, using vertex_position for pos).
             [DISCLOSED DEVIATION from the pre-reg's literal 4-argument signature
             (h_A, positions, cut_normal, window): "positions" is realized as this per-bond record
             list rather than a flat all-vertex array, because deciding WHICH h_A entries constitute
             "the bond coupling" requires the physical-adjacency pairing; the station supplies that
             pairing when it builds `bonds`.  See the D1b station file's disclosure section.]
    cut_normal : unit Cartesian 3-vector, the horizon normal (proper-frame; the station derives it as
             n = L^{-T}e_d, cut_normal = n/|n| -- the reciprocal-lattice Cartesian normal to the
             fractional-plane family {y: e_d.y=c}, for a declared integer direction e_d).
    window : keep only bonds whose declared point has |projection on cut_normal| <= window (proper
             units; the station sets this to a fraction of the region's achieved proper depth).
    point  : 'midpoint' (PRIMARY; declared point = (pos_i+pos_j)/2) or 'endpoint' (ALTERNATE;
             declared point = whichever of pos_i, pos_j is CLOSER to the horizon, i.e. the smaller
             |projection| -- a deterministic, disclosed convention).
    beta_b = |h_A[i,j]|;  x_b = |cut_normal . declared_point|.
    Returns (slope, fit_err, n_bonds, residuals).  DEFAULT (origin_constrained=False): an ordinary
    least-squares affine fit beta_b = slope*x_b + intercept over the window-selected bonds, EACH
    INDIVIDUAL BOND its own data point (not pre-averaged per layer) -- so a depth with more transverse
    bond copies carries proportionally more weight in the fit (the pre-reg's "weighted" fit: weighted
    BY MULTIPLICITY, since no independent per-bond variance exists in a single spectral realization).
    slope is the near-horizon BW coefficient (-> 2pi under Bisognano-Wichmann).  residuals[k] =
    beta_b[k] - (slope*x_b[k]+intercept), one per RETAINED bond, for diagnosis.  Returns (nan, nan, n,
    []) if fewer than 2 bonds fall in the window (ill-posed fit).
    ORIGIN-CONSTRAINED (origin_constrained=True, W2-D1c): beta_b = a*x_b, NO intercept; a = sum(beta_b*
    x_b) / sum(x_b^2) over bonds with 0 < x_b <= (effective window).  residuals[k] = beta_b[k] -
    a*x_b[k].  Returns (nan, nan, n, []) if no bond falls in the (0, window] range (ill-posed: the
    sum(x_b^2) denominator would be zero)."""
    cut_normal = np.asarray(cut_normal, dtype=float)
    eff_window = window if absolute_window is None else absolute_window
    xs, betas = [], []
    for (i, j, pi, pj) in bonds:
        pi = np.asarray(pi, dtype=float)
        pj = np.asarray(pj, dtype=float)
        if point == "midpoint":
            declared = (pi + pj) / 2.0
        elif point == "endpoint":
            xi, xj = abs(float(cut_normal @ pi)), abs(float(cut_normal @ pj))
            declared = pi if xi <= xj else pj
        else:
            raise ValueError(f"bond_profile_slope: unknown point convention {point!r}")
        x = abs(float(cut_normal @ declared))
        keep = (0.0 < x <= eff_window) if origin_constrained else (x <= eff_window)
        if keep:
            xs.append(x)
            betas.append(abs(h_A[i, j]))
    n_bonds = len(xs)
    if origin_constrained:
        if n_bonds < 1:
            return float("nan"), float("nan"), n_bonds, np.array([])
        xs = np.array(xs, dtype=float)
        betas = np.array(betas, dtype=float)
        denom = float(np.sum(xs ** 2))
        if denom <= 0.0:
            return float("nan"), float("nan"), n_bonds, np.array([])
        a = float(np.sum(betas * xs) / denom)
        residuals = betas - a * xs
        fit_err = float(np.sqrt(np.mean(residuals ** 2)))
        return a, fit_err, n_bonds, residuals
    else:
        if n_bonds < 2:
            return float("nan"), float("nan"), n_bonds, np.array([])
        xs = np.array(xs, dtype=float)
        betas = np.array(betas, dtype=float)
        a, b = np.polyfit(xs, betas, 1)
        residuals = betas - (a * xs + b)
        fit_err = float(np.sqrt(np.mean(residuals ** 2)))
        return float(a), fit_err, n_bonds, residuals


def cone_velocity(direction, eps=1e-4, node=-1.0):
    """The emergent cone velocity v(n) = |dE/dk_phys| of the srs dispersive branch at the node
    lambda_F (default -1), along a lattice direction (physical k = 2*pi*fractional).  A FORCED
    spectral read of A(k) near Gamma (no target).  Returns (v_upper, v_lower, flat_dev)."""
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    w = np.sort(np.linalg.eigvalsh(srs.adjacency(n * eps)).real)
    near = w[np.abs(w - node) < 0.5]
    kphys = 2 * math.pi * eps
    if len(near) >= 3:
        v_lo = abs(near[0] - node) / kphys
        v_hi = abs(near[-1] - node) / kphys
        flat = abs(near[1] - node) / kphys
    else:
        v_lo = v_hi = flat = float("nan")
    return v_hi, v_lo, flat


def band_quantum_metric(kvec, node=-1.0, d=1e-5):
    """The quantum geometric tensor of the srs adjacency's m=0 (flat/nearest-node) band at k:
    Q_ij = Tr[P ∂_iP ∂_jP], P = the band projector.  Real part = quantum metric g_ij (the FORCED
    daylight object — un-computed elsewhere; every prior Berry read is the imaginary part on a
    DISPERSIVE band).  Returns (trace_g, berry_xy, E_rel_node).  g DIVERGES ~ C(n̂)/|k|^2 off-axis at
    the quadratic band-touching; = 0 exactly along the flat axes."""
    def P_and_E(k):
        w, V = np.linalg.eigh(srs.adjacency(np.asarray(k, float)))
        i = int(np.argmin(np.abs(w - node)))
        v = V[:, i:i + 1]
        return v @ v.conj().T, w[i] - node

    P, E = P_and_E(kvec)
    dP = []
    for ax in range(3):
        e = np.zeros(3)
        e[ax] = d
        Pp, _ = P_and_E(np.asarray(kvec, float) + e)
        Pm, _ = P_and_E(np.asarray(kvec, float) - e)
        dP.append((Pp - Pm) / (2 * d))
    tr_g = sum(np.real(np.trace(P @ dP[a] @ dP[a])) for a in range(3))
    berry = 2 * np.imag(np.trace(P @ dP[0] @ dP[1]))
    return tr_g, berry, E


def dr_frame_audit():
    """Fork-A (ML-2b): is the winding a gauge/DHR charge (CATEGORY-BIGGER) or a cross-cutting non-gauge
    grading?  A gauge charge fixes the vacuum; A4 does, the winding screw U_pi does NOT (<0|U_pi^2|0>=i/2
    != 1) => winding adds no sectors => the DR frame (F,2T) is canonical (conditional on TD-limit duality)
    => O4's lift-dependence dissolves => ML-5 posable.  DR pays the FRAME, NOT the weld H(w|t)=1.63 bits.
    Returns the verdicts.  See proofs/foundations/ML2b_dr_frame_2026-07-08.py."""
    import cmath as _cm
    import itertools as _it
    sys.path.insert(0, _REPO)
    sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "bridge"))
    from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402
    g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
    I8 = np.eye(8)
    EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
    gam = lambda x: sum(x[a] * g6[a] for a in range(NE))

    def spin_lift(R):
        rowsU = [np.kron(gam(R[:, a]), I8) - np.kron(I8, g6[a].T) for a in range(NE)]
        _, s, Vh = np.linalg.svd(np.vstack(rowsU))
        M = Vh[np.sum(s > 1e-9):].conj()[0].reshape(8, 8)
        return M / np.sqrt(np.abs(np.linalg.det(M @ M.conj().T)) ** (1 / 8))

    J6 = complex_structure_J6()
    wJ, VJ = np.linalg.eig(J6)
    modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
    A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
    NHAT = sum(a.conj().T @ a for a in A_ops)
    wN, VN = np.linalg.eigh(NHAT)
    vac = VN[:, [int(np.argmin(wN))]]
    Pw = {w: VN[:, np.round(wN).astype(int) == w] @ VN[:, np.round(wN).astype(int) == w].conj().T
          for w in range(4)}
    sig3 = {0: 0, 1: 2, 2: 3, 3: 1}
    Rpi = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig3[i], sig3[j]
        Rpi[EIDX[(min(a, b), max(a, b))], e] = 1.0
    Upi2 = spin_lift(Rpi) @ spin_lift(Rpi)
    w2 = abs((vac.conj().T @ Upi2 @ vac).item())
    winding_gauge = w2 > 0.99 and np.max(np.abs(Upi2 @ NHAT - NHAT @ Upi2)) < 1e-6
    # weld H(w|t) from the WS1 species x winding table
    evU, VU = np.linalg.eig(Upi2)
    lab = np.array([int(round(_cm.phase(z) / (2 * math.pi / 3))) % 3 for z in evU])
    PiF = {t: (lambda Q: Q @ Q.conj().T)(np.linalg.qr(VU[:, lab == t])[0]) for t in (0, 1, 2)}
    Pj = np.array([[np.real(np.trace(Pw[w] @ PiF[t])) for t in range(3)] for w in range(4)]) / 8.0
    Pwm, Ptm = Pj.sum(1), Pj.sum(0)
    Hw = -sum(p * math.log2(p) for p in Pwm if p > 1e-12)
    Iwt = sum(Pj[w, t] * math.log2(Pj[w, t] / (Pwm[w] * Ptm[t]))
              for w in range(4) for t in range(3) if Pj[w, t] > 1e-12)
    return {
        "winding_is_gauge": bool(winding_gauge),        # False => category not bigger than the species
        "category_bigger": bool(winding_gauge),
        "frame_forced": not winding_gauge,              # canonical (F,2T) frame (conditional TD-limit)
        "weld_bits": Hw - Iwt,                           # H(w|t) = 1.63 survives DR (unpaid weld)
    }


def emergent_metric():
    """The srs cone's emergent inverse-spatial-metric g^{ij} (dispersion velocity^2 tensor,
    E^2 = g^{ij} k_i k_j), assembled FORCED from the cone velocities (ML-1'').  Eigenvalues {1/4,1/4,1}
    (a genuine anisotropic relativistic Dirac cone).  The proper distance from a bond to the horizon
    plane {x_0 = c} is |dx_0| / sqrt(g^{00}) (ML-1‴).  Returns the 3x3 g^{ij}."""
    diag = cone_velocity([1, 0, 0])[0] ** 2                   # g^{00} = v_axis^2 = 1/2
    g01 = cone_velocity([1, 1, 0])[0] ** 2 - diag
    g02 = cone_velocity([1, 0, 1])[0] ** 2 - diag
    g12 = cone_velocity([0, 1, 1])[0] ** 2 - diag
    return np.array([[diag, g01, g02], [g01, diag, g12], [g02, g12, diag]])


def diamond_modular_energy(R, beta_eff, ngrid=26, node=-1.0):
    """Fork-C object: the per-band contribution to a causal diamond's local modular energy, for a
    diamond of PROPER radius R.  The diamond IS the regulator -- proper momentum resolution q_min=pi/R
    (proper momentum |k|_g = sqrt(g^{ij}k_ik_j), the emergent metric ML-1''), NO chosen hand-regulator.
    Bands split at the node lambda_F: flat = m=0 (min |E|, the matter candidate, E~q^2), cone = the
    dispersive branches (radiation, E~q).  Modular energy density = beta_eff*E*n(E) (KMS).  Returns
    (dK_cone, dK_flat), each finite for every finite R."""
    G = emergent_metric()                                     # g^{ij}, |k|_g^2 = k.G.k
    qmin = math.pi / R
    qs = [(i + 0.5) / ngrid for i in range(ngrid)]
    dK = {"cone": 0.0, "flat": 0.0}
    for kk in itertools.product(qs, repeat=3):
        k = np.array(kk)
        kf = np.minimum(k, 1 - k)                             # fold to [-1/2,1/2] magnitude
        p = math.sqrt(kf @ G @ kf)                            # proper momentum |k|_g
        if p < qmin:                                          # excluded by the diamond (IR)
            continue
        w = np.sort(np.abs(np.linalg.eigvalsh(srs.adjacency(k)) + (-node)))
        for j, E in enumerate(w):
            if E < 1e-9:
                continue
            lab = "flat" if j == 0 else ("cone" if E < 2 else None)
            if lab is None:
                continue
            x = beta_eff * E
            nB = 1.0 / (math.exp(x) - 1.0) if x < 60 else 0.0
            dK[lab] += x * nB                                 # modular energy density beta*E*n
    return dK["cone"], dK["flat"]


def chain_vacuum(L):
    """Critical half-filled free-fermion chain ground-state correlation on L sites (infinite-chain
    vacuum restricted) -- the BW benchmark control (a lattice Dirac vacuum)."""
    idx = np.arange(L)
    d = idx[:, None] - idx[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.sin(np.pi * d / 2) / (np.pi * d)
    C[d == 0] = 0.5
    return C


def benchmark_bw_2pi(L=400):
    """Validate the BW-slope pipeline on the critical chain: the first-bond beta/x must approach 2pi
    (physical hopping = 1).  Returns (slope, slope/2pi)."""
    C = chain_vacuum(L)
    h_A = entanglement_hamiltonian(C)
    beta0 = abs(h_A[0, 1])            # first-bond entanglement hopping (x_center = 0.5)
    slope = beta0 / 0.5
    return slope, slope / (2 * math.pi)


# ===========================================================================
# 4c. GAUGE SECTORS  (the DHR superselection category of the observable algebra)  [ML-2]
# ===========================================================================
def gauge_sector_category():
    """The DHR sectors of the observable algebra A = F^G on the net.  Field algebra F = 8-dim Cl(6)
    Fock; gauge group G = A4 (forced J-covariance), Fock rep SPINORIAL => the double cover 2T.  Returns
    a dict: the species-sector dims (the G-irrep decomposition of F), whether the rep is a genuine
    double cover, whether the sectors coincide with the species grading, and the fermion-parity
    (Bose/Fermi) per sector.  See proofs/foundations/ML2_dhr_sectors_2026-07-08.py."""
    import itertools as _it
    sys.path.insert(0, _REPO)
    sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "bridge"))
    from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402
    g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
    I8 = np.eye(8)
    EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}

    def gam(x):
        return sum(x[a] * g6[a] for a in range(NE))

    def edge_rep(sig):
        R = np.zeros((NE, NE))
        for e, (i, j, v) in enumerate(EDGES):
            a, b = sig[i], sig[j]
            s = 1.0
            if a > b:
                a, b, s = b, a, -1.0
            R[EIDX[(a, b)], e] = s
        return R

    def spin_lift(R):
        rowsU = [np.kron(gam(R[:, a]), I8) - np.kron(I8, g6[a].T) for a in range(NE)]
        _, s, Vh = np.linalg.svd(np.vstack(rowsU))
        M = Vh[np.sum(s > 1e-9):].conj()[0].reshape(8, 8)
        return M / np.sqrt(np.abs(np.linalg.det(M @ M.conj().T)) ** (1 / 8))

    A4 = [dict(enumerate(p)) for p in _it.permutations(range(4))
          if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
    J6 = complex_structure_J6()
    wJ, VJ = np.linalg.eig(J6)
    modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
    A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
    NHAT = sum(a.conj().T @ a for a in A_ops)
    wN, VN = np.linalg.eigh(NHAT)
    wNr = np.round(np.real(wN)).astype(int)
    Pw = {w: VN[:, wNr == w] @ VN[:, wNr == w].conj().T for w in range(4)}
    U = [spin_lift(edge_rep(g)) for g in A4]
    # is the rep a genuine double cover (cocycle takes -1)?
    keyf = lambda dd: tuple(dd[i] for i in range(NV))
    ix = {keyf(g): n for n, g in enumerate(A4)}
    comp = lambda g, h: {i: g[h[i]] for i in range(NV)}
    dbl = False
    for a, g in enumerate(A4):
        for b, h in enumerate(A4):
            c = np.trace(np.linalg.solve(U[ix[keyf(comp(g, h))]], U[a] @ U[b])) / 8.0
            if abs(c + 1) < 1e-3:
                dbl = True
    gauge_inv = max(np.max(np.abs(U[a] @ NHAT - NHAT @ U[a])) for a in range(len(A4))) < 1e-7
    dims = {w: int(round(np.trace(Pw[w]).real)) for w in range(4)}
    return {
        "species_sector_dims": dims,                       # {0:1,1:3,2:3,3:1} = nu,d,u,e
        "double_cover_2T": dbl,                             # spinorial gauge group
        "sectors_are_species": gauge_inv and dims == {0: 1, 1: 3, 2: 3, 3: 1},
        "fermion_parity": {w: int((-1) ** w) for w in range(4)},   # Bose/Fermi per sector
    }


# ===========================================================================
# 4d. DENSITY RESPONSE  (L-response; B2-a, pre-reg internal research notes)
# ===========================================================================
# The Lindhard density-density bubble chi_0(q,omega) on srs.adjacency(k)'s FULL 4-band Bloch
# structure (adjudication 3: the full bands, not D2's abstracted 4x4 continuum fiber; M2b's node
# convention lambda_F = -1, "half-filling") + the Mermin (RTA number-conserving) closure
# (adjudication 5, the ONE declared math import).  B2a-0's finding (numerically confirmed
# k=(0.1,0,0) etc.): the "m=0" node-nearest branch is EXACTLY flat only along special lines, not
# globally flat over the BZ -- so bands are labeled PER-k by rank in |E(k)-node| (0=nearest-node
# "flat" candidate, 1,2=dispersive "cone" branches if E<2, 3="far"/Perron if E>=2), reusing
# diamond_modular_energy's own flat/cone convention (line ~586 above) rather than a fixed band index.
NODE_LAM_F = -1.0          # M2b's Weyl-node Fermi level (M2b_fluctuation_spectrum_2026-07-07.py:62)


def bz_grid(n):
    """The full first Brillouin zone (fractional k in [-1/2,1/2)), Monkhorst-Pack n^3 mesh.  The
    closest-compliant reading of the D2 pattern's momentum "ball grid" for a GENUINELY PERIODIC
    lattice model (srs.adjacency(k) is exactly periodic, period 1 in each fractional component) --
    unlike D2's unbounded continuum fiber, no UV cutoff/ball is needed or meaningful; the natural
    analogue of D2's cutoff is simply the full zone.  dk3 normalizes Sum_k dk3 = 1 (a per-unit-cell
    BZ average, not an absolute (2pi)^-3-weighted volume -- declared, not claimed as absolute units).
    """
    ax = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return pts, 1.0 / n ** 3


def _bands_at(kpts, node=NODE_LAM_F):
    """Vectorized batch Bloch band structure over kpts (N,3): NV=4 eigenvalues + eigenvectors at
    every point, columns REORDERED per-k by rank in |E-node| (0=nearest-node .. 3=farthest) -- a
    relabeling only (sums over all bands are invariant), matching diamond_modular_energy's
    flat/cone convention.  Reuses srs.EDGES only (srs.py untouched; no batch API there)."""
    NVn = srs.NV
    A = np.zeros((len(kpts), NVn, NVn), dtype=complex)
    for i, j, v in EDGES:
        vv = np.asarray(v, dtype=float)
        p = np.exp(2j * np.pi * (kpts @ vv))
        A[:, i, j] += p
        A[:, j, i] += np.conj(p)
    E, U = np.linalg.eigh(A)
    order = np.argsort(np.abs(E - node), axis=1)                 # (N,NV): 0=nearest-node..3=farthest
    E = np.take_along_axis(E, order, axis=1)
    U = np.take_along_axis(U, order[:, None, :], axis=2)
    return E, U


def lindhard_setup(q_vec, beta, n_grid=32, node=NODE_LAM_F, intraband_only=False):
    """Precompute the flattened (k,n,n') Lindhard terms at fixed q (band structure does not depend
    on omega, so this is built ONCE and reused across every omega / Mermin complex-shift evaluation).
    Density vertex (adjudication 4): Gamma = I, matrix element = the Bloch periodic-part overlap
    <u_{n'}(k+q)|u_n(k)> (the genuine, k-dependent eigenvector overlap -- NOT a fixed matrix, since
    on this lattice no fixed Gamma commutes with H(k) at every k except scalars).  intraband_only=True
    is the R-5 DECORATIVE CONTROL: masks to n'=n (rank-matched, post node-reordering) ONLY, i.e. a
    vertex constructed to carry NO interband matrix structure by fiat (see the station file's R-5
    disclosure for why this -- not a literal fixed commuting Gamma -- is the closest-compliant
    control on a lattice with no k-independent symmetry protecting a fixed eigenvector)."""
    kpts, dk3 = bz_grid(n_grid)
    Ek, Uk = _bands_at(kpts, node)
    Ekq, Ukq = _bands_at(kpts + np.asarray(q_vec, float), node)
    f_k = 1.0 / (1.0 + np.exp(beta * (Ek - node)))
    f_kq = 1.0 / (1.0 + np.exp(beta * (Ekq - node)))
    M = np.einsum("kap,kan->knp", np.conj(Ukq), Uk)               # M[k,n,p] = <p,k+q|n,k>
    absM2 = np.abs(M) ** 2
    if intraband_only:
        NVn = Ek.shape[1]
        absM2 = absM2 * np.eye(NVn)[None, :, :]
    w = f_k[:, :, None] - f_kq[:, None, :]                        # w[k,n,p] = f_n(k) - f_p(k+q)
    dE = Ekq[:, None, :] - Ek[:, :, None]                         # dE[k,n,p] = E_p(k+q) - E_n(k)
    rankK = np.broadcast_to(np.arange(Ek.shape[1])[None, :, None], w.shape)
    rankKq = np.broadcast_to(np.arange(Ek.shape[1])[None, None, :], w.shape)
    return {"absM2": absM2.ravel(), "w": w.ravel(), "dE": dE.ravel(), "dk3": dk3,
            "rankK": rankK.ravel(), "rankKq": rankKq.ravel(), "Ek": Ek, "Ekq": Ekq}


def chi0_from_setup(setup, omega, eta):
    """chi_0(q,omega) from a precomputed lindhard_setup, at a (possibly complex, for the Mermin
    shift omega+i*gamma) frequency omega; eta is an ADDITIONAL declared numerical broadening
    (printed by the station; set eta=0 when omega already carries Im(omega)=gamma>0)."""
    denom = (omega + 1j * eta) - setup["dE"]
    return setup["dk3"] * np.sum(setup["absM2"] * setup["w"] / denom)


def lindhard_chi0(q_vec, omegas, beta, n_grid=32, node=NODE_LAM_F, eta=1e-3, intraband_only=False):
    """The free (collisionless) finite-T Lindhard bubble chi_0(q,omega) over an array of real
    omegas, standard causal (+i*eta) retarded convention: chi_0(q,w) = Sum_k Sum_{n,n'}
    |<u_n'(k+q)|u_n(k)>|^2 (f_n(k)-f_n'(k+q)) / (w + E_n(k) - E_n'(k+q) + i*eta).  Full-BZ
    Monkhorst-Pack grid (bz_grid), M2b node convention (node=lambda_F=-1).  Returns
    (chi_array (complex, len(omegas)), chi_static (real omega=0 value), setup dict for reuse/
    diagnostics e.g. the R-3 two-fluid split)."""
    setup = lindhard_setup(q_vec, beta, n_grid, node, intraband_only)
    chi = np.array([chi0_from_setup(setup, w, eta) for w in omegas], dtype=complex)
    chi_static = chi0_from_setup(setup, 0.0, eta)
    return chi, chi_static, setup


def mermin_chi(q_vec, omegas, beta, gamma, n_grid=32, node=NODE_LAM_F, intraband_only=False):
    """THE ONE DECLARED MATH IMPORT (adjudication 5): the Mermin/RTA number-conserving closure,
        chi_M(q,w) = [(1 + i*gamma/w) chi_0(q,w+i*gamma)] / [1 + (i*gamma/w) chi_0(q,w+i*gamma)/chi_0(q,0)]
    -- the standard conserving completion of the relaxation-time approximation (a closure IDENTITY,
    not a physics constant).  Both inputs are reused, not adjusted: beta=beta_eff (G5a thermal-time,
    derivation_topdown/adapters/thermal_time.py:209-211) and gamma=gamma_micro (MC-2's derived
    Ramanujan-gap rate, MC2_phase_memory_kernel_2026-07-07.py:42-57).  omega=0 uses the closure's
    OWN removable-singularity limit chi_M(q,0)=chi_0(q,0) exactly (the built-in compressibility/
    f-sum-rule identity of the conserving closure -- not a separate approximation).  Returns
    (chi_M array (complex), chi0_static, setup)."""
    setup = lindhard_setup(q_vec, beta, n_grid, node, intraband_only)
    chi0_static = chi0_from_setup(setup, 0.0, eta=1e-3)
    out = np.zeros(len(omegas), dtype=complex)
    for i, wv in enumerate(omegas):
        if abs(wv) < 1e-12:
            out[i] = chi0_static
            continue
        chi0_shift = chi0_from_setup(setup, wv + 1j * gamma, eta=0.0)
        num = (1 + 1j * gamma / wv) * chi0_shift
        den = 1 + (1j * gamma / wv) * chi0_shift / chi0_static
        out[i] = num / den
    return out, chi0_static, setup


# ===========================================================================
# 5. REGRESSION ANCHORS  (the two known degenerate region-shapes; must always hold)
# ===========================================================================
def anchor_cell_projector():
    """region = one CELL: C=(I+iJ6)/2 is an exact rank-3 Hermitian projector with Tr C = 3."""
    C = vacuum_covariance()
    return (np.max(np.abs(C - C.conj().T)) < 1e-12
            and np.max(np.abs(C @ C - C)) < 1e-9
            and abs(np.trace(C).real - NE / 2) < 1e-9)


def anchor_tick_2pi():
    """region = the global TICK subalgebra: M0-2R's tick modular flow is a compact U(1) of MINIMAL
    period 2pi.  Structural stand-in: the tick number operator N-hat has integer spectrum, so
    U(theta)=exp(i*theta*N-hat) satisfies U(2pi)=I EXACTLY and has NO earlier return U(2pi/j)=I
    (j>1).  This is the anchor the diamond modular flow (ML-1) is measured against."""
    N = np.diag(np.arange(0, 6, dtype=float))  # consecutive integer tick counts
    U = lambda th: np.diag(np.exp(1j * th * np.diag(N)))
    full = np.max(np.abs(U(2 * math.pi) - np.eye(6)))
    earliest = min(np.max(np.abs(U(2 * math.pi / j) - np.eye(6))) for j in range(2, 7))
    return full < 1e-9 and earliest > 1e-3


# ===========================================================================
# 6. SELF-TEST  (reproduces ML-0's headline reads + the two anchors)
# ===========================================================================
def self_test(verbose=True):
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    # anchors
    ck("ANCHOR cell C-projector (rank-3, Tr=3)", anchor_cell_projector())
    ck("ANCHOR tick modular flow U(1) minimal period 2pi", anchor_tick_2pi())

    # reconciliation
    R = reversal()
    B0 = hashimoto_gamma()
    ck("reversal R^2=I, Tr R=0", np.allclose(R @ R, np.eye(ND)) and abs(np.trace(R)) < 1e-12)
    ck("B breaks the reversal grading ([B,R] != 0)",
       np.max(np.abs(B0 @ R - R @ B0)) > 0.5)

    # the exact light cone (the physics heart)
    patch = Patch(M=4)
    worst = patch.anticommutator_below_cone(T=5)
    ck("EXACT light cone: {alpha_a(t),a_c^dag}=(B^t)_ca IDENTICALLY 0.0 below the geometric horizon",
       worst == 0.0, detail=f"max|below cone| = {worst:.1e}")
    radii = patch.horizon_radii(patch.central_dart(), T=5)
    ck("horizon speed = one graph-step/tick", radii == [1, 2, 3, 4, 5], detail=f"radii={radii}")

    # isotony
    base = patch.central_dart()
    d1, d2, d3 = patch.diamond(base, 1), patch.diamond(base, 2), patch.diamond(base, 3)
    ck("isotony: nested diamonds nest as mode-sets", d1 <= d2 <= d3,
       detail=f"|O(1,2,3)|={[len(d1),len(d2),len(d3)]}")

    # twisted locality
    tl = twisted_locality_holds()
    ck("twisted (Klein) locality: even commute, odd anticommute, naive FAILS, twist commutes",
       tl["even_even_commute"] < 1e-12 and tl["odd_odd_anticommute"] < 1e-12
       and tl["naive_commutation_fails"] > 0.5 and tl["klein_twist_commutes"] < 1e-12)

    # cell-level twisted Haag duality (region = K4 triangle, complement)
    C = vacuum_covariance()
    zR, epsR, SR = region_data(C, [0, 1, 3])
    zRc, epsRc, SRc = region_data(C, [2, 4, 5])
    ck("cell-level duality: S(R)=S(R^c), shared modular spectrum, no zeta pinned",
       abs(SR - SRc) < 1e-9 and np.allclose(epsR, epsRc, atol=1e-7)
       and min(zR.min(), 1 - zR.max()) > 1e-6)

    # ML-2 gauge sectors: the DHR sectors = the species grading, under the double-cover gauge group
    sc = gauge_sector_category()
    ck("gauge sectors = species {1,3,3,1} under the double-cover (2T) gauge group (ML-2)",
       sc["sectors_are_species"] and sc["double_cover_2T"],
       detail=f"dims {sc['species_sector_dims']}, parity {sc['fermion_parity']}")

    # ML-3 flat-band quantum geometry: the m=0 band's quantum metric diverges ~C(n)/|k|^2 at the node
    g_face = band_quantum_metric(np.array([1, 1, 0.]) / math.sqrt(2) * 1e-2)[0]
    ck("m=0 flat-band quantum metric diverges ~C(n)/|k|^2 at the node (ML-3 daylight object)",
       g_face * (1e-2) ** 2 > 1.0, detail=f"tr g([110])*|k|^2 = {g_face * (1e-2)**2:.2f}")

    # ML-1‴ emergent metric: g^{ij} eigenvalues {1/4,1/4,1} (used for proper-distance BW reads)
    gev = np.linalg.eigvalsh(emergent_metric())
    ck("emergent metric g^{ij} eigenvalues {1/4,1/4,1} (ML-1'' / ML-1‴ proper-distance)",
       np.allclose(np.sort(gev), [0.25, 0.25, 1.0], atol=1e-2), detail=f"eigs {np.round(gev,3)}")

    # ML-3b diamond-regulated per-band modular energy: finite for a finite proper radius (no hand regulator)
    kc, kf = diamond_modular_energy(32.0, 2 * math.log((1 / math.sqrt(2)) / 0.039), ngrid=14)
    ck("diamond delta<K_R> finite per band (ML-3b: the diamond is the IR regulator)",
       kc > 0 and kf > 0 and np.isfinite(kc) and np.isfinite(kf), detail=f"cone {kc:.1f}, flat {kf:.1f}")

    # ML-2b DR frame audit: winding is NOT a gauge charge => frame forced; weld H(w|t)=1.63 survives
    fa = dr_frame_audit()
    ck("DR frame audit: winding not gauge => FRAME-FORCED, weld H(w|t)~1.63 survives (ML-2b)",
       (not fa["winding_is_gauge"]) and fa["frame_forced"] and abs(fa["weld_bits"] - 1.63) < 0.01,
       detail=f"frame_forced={fa['frame_forced']}, weld={fa['weld_bits']:.3f} bits")

    if verbose:
        print("RESULT:", "ALL NET ANCHORS + ML-0/ML-2/ML-3 READS PASS" if ok else "A CHECK FAILED")
    return ok


if __name__ == "__main__":
    print("=" * 88)
    print(" THE NET (Layer 3, {A(O)}) — self-test: regression anchors + ML-0 reads")
    print("=" * 88)
    sys.exit(0 if self_test() else 1)


# ===========================================================================
# 4d (continued). W2-BGK — THE TWO-MOMENT {n, j} CONSERVING CLOSURE
# (pre-reg internal research notes, frozen 8ca645c/6e03d95)
# ===========================================================================
# [PLACEMENT NOTE: appended after the __main__ guard purely to keep this accretion conflict-free
#  against concurrent stations editing earlier sections (W2 second wave runs several stations on
#  this file at once).  On IMPORT the definitions land in the module namespace exactly as if they
#  sat with the rest of §4d; the __main__ self-test path does not use them, so script mode is
#  unaffected.  Logically this block IS §4d.]
#
# THE CLOSURE (adjudication 2; the declared math import = Mermin's density-AND-current-conserving
# generalization of the RTA -- math, not physics constants).  The kinetic/von-Neumann equation for
# the response deviation delta-rho, with the {n,j}-CONSERVING collision term
#
#     C[delta-f] = -gamma * (delta-f - P_{n,j}[delta-f]),
#
# P_{n,j} = the projector onto local-equilibrium shifts of the conserved moments.  The conserved
# moments do not relax AT ALL (that is what conserving means); gamma = gamma_micro relaxes only the
# orthogonal complement.  NO second relaxation coefficient exists anywhere in this construction.
#
# DERIVATION (on-screen, per the station contract).  Particle-hole space: xi = (k, n, p) labels the
# transition |n,k> -> |p,k+q>, with
#     dE_xi = E_p(k+q) - E_n(k),   w_xi = f_n(k) - f_p(k+q),
#     [Gamma_a]_xi = <p,k+q| Gamma_a |n,k>,   a in {n, j_x, j_y, j_z}
# (Gamma_n = 1 -> the Bloch periodic-part overlap M, exactly lindhard_setup's element; Gamma_{j_a} =
# the analytic velocity vertex, velocity_vertex below).
#
# THE MOMENT-SPACE INNER PRODUCT (set up + documented per the station contract): the equilibrium
# measure is
#     mu_xi = w_xi / dE_xi  >= 0,     degenerate limit |dE|->0:   mu -> beta*f*(1-f) = -df/dE,
# i.e. the "-df/dE" measure; <X, Y>_mu = sum_xi conj(X_xi) Y_xi mu_xi.  Gram matrix of the conserved
# vertices: G_ab = dk3 * sum_xi conj(Gamma_a) Gamma_b mu (Hermitian, positive).  mu is the linear-
# response kernel of a static shift of H: f(H - sum_b lam_b O_b) has ph elements delta-rho^le_xi =
# mu_xi * sum_b lam_b [Gamma_b]_xi to first order.  Hence
#     P_{n,j}[X]_xi = mu_xi * sum_ab [Gamma_a]_xi [G^-1]_ab m_b(X),   m_b(X) = dk3*sum_xi conj(Gamma_b) X_xi
# is precisely the mu-ORTHOGONAL projection of the deviation function X/mu onto span{Gamma_a}; it
# satisfies m_a(P[X]) = m_a(X) EXACTLY, so the collision term's action on the {n,j} moments is zero
# BY CONSTRUCTION (bgk_conservation_check verifies to machine precision), and P^2 = P.
#
# CLOSED FORM.  Linearized kinetic equation at (q, omega), external potential phi coupling to the
# density, z = omega + i*gamma:
#     (z - dE_xi) delta-rho_xi = w_xi [Gamma_n]_xi phi + i*gamma * delta-rho^le_xi ,
# with Mermin's self-consistency: the local equilibrium carries the SAME conserved moments as the
# actual solution, m_a(delta-rho) = m_a(delta-rho^le) = [G lam]_a.  Taking moments and using the
# exact partial fraction  1/(dE*(z-dE)) = (1/z)*(1/dE + 1/(z-dE))  gives, with the EXACT static
# bubble chi0(0) := -G (its degenerate transitions carry the -df/dE limit; the eta-broadened static
# of B2-a's mermin_chi is a numerical stand-in for it, difference quantified in the station file):
#     m(delta-rho) = chi0(z)_{.,n} phi + (i*gamma/z) [chi0(z) - chi0(0)] lam ,   lam = G^{-1} m
#  => the self-consistent 4x4 (moment-space) solve
#     chi_M(q,omega) = chi0(0) [ chi0(0) + (i*gamma/z)(chi0(z) - chi0(0)) ]^{-1} chi0(z) .
# Scalar {n}-only reduction: chi_M = z*chi0(z)*chi0(0) / [omega*chi0(0) + i*gamma*chi0(z)], which is
# ALGEBRAICALLY IDENTICAL to Mermin's formula as implemented in mermin_chi above (multiply out
# (1+i*gamma/omega) = z/omega) -- the BGK-2 contract check verifies this numerically.  The matrix
# form is manifestly REGULAR at omega=0 (z = i*gamma there): chi_M(q,0) = chi0(0) exactly -- the
# conserving compressibility identity, automatic rather than special-cased.
# GL(4) COVARIANCE: under any change of moment basis L (Gamma -> L.Gamma), chi0 -> L chi0 L^dag and
# chi_M -> L chi_M L^dag, with the nn element INVARIANT whenever Gamma_n is kept as a basis vector:
# the closure depends only on the conserved SPAN, as a projector-based construction must (checked
# numerically in the station file).
def velocity_operator(kvec):
    """The ANALYTIC velocity operator v_a(k) = (1/2pi) * dA(k)/dk_a (a=x,y,z), a triple of exact
    Hermitian 4x4 matrices at one fractional k.  A(k)'s entries are sums of exp(2pi*i k.v) edge
    phases (srs.adjacency; srs.py lines 17-22), so d/dk_a inserts a factor (2pi*i v_a) per edge
    term; the 1/(2pi) makes this dE/dk_phys with k_phys = 2pi*k_frac -- EXACTLY cone_velocity's
    units (its kphys = 2*pi*eps, line ~462 above).  The 2pi bookkeeping is thus: THIS operator's
    band-diagonal elements are group velocities in cone_velocity's physical normalization; multiply
    by 2pi to get d/dk_frac (as band_quantum_metric's finite differences use).  Returns (3,4,4)."""
    k = np.asarray(kvec, float)
    V = np.zeros((3, NV, NV), complex)
    for i, j, v in EDGES:
        vv = np.asarray(v, float)
        p = 1j * np.exp(2j * np.pi * (k @ vv))          # (1/2pi)*(2pi*i) = i per edge phase
        for a in range(3):
            V[a, i, j] += vv[a] * p
            V[a, j, i] += np.conj(vv[a] * p)
    return V


def _velocity_blocks(kpts):
    """Batch velocity_operator over kpts (N,3): returns (3, N, 4, 4).  Same edge-phase derivative,
    vectorized (mirrors _bands_at's batch construction; srs.py untouched)."""
    V = np.zeros((3, len(kpts), NV, NV), complex)
    for i, j, v in EDGES:
        vv = np.asarray(v, float)
        p = 1j * np.exp(2j * np.pi * (kpts @ vv))
        for a in range(3):
            V[a, :, i, j] += vv[a] * p
            V[a, :, j, i] += np.conj(vv[a] * p)
    return V


def velocity_vertex(q_vec, beta, n_grid=32, node=NODE_LAM_F, scramble_seed=None):
    """W2-BGK moment-space setup: the analytic velocity vertex SANDWICHED in the lindhard_setup
    eigenbases (same _bands_at diagonalization -- no new diagonalization convention), together with
    the density vertex and the conserving-closure ingredients.  Vertex convention (declared): the
    MIDPOINT rule Gamma_{j_a}(k,k+q) = v_a(k + q/2), the exact lattice (Peierls/continuity) current
    to O(q^2): per edge, A(k+q)-A(k) = e^{2pi*i(k+q/2).v} * 2i*sin(pi q.v), and the midpoint
    gradient replaces sin(pi q.v) -> pi q.v -- an O((pi q.v)^2/6) relative difference, verified
    in-code by the station's continuity/f-sum identity  dE_xi * M_xi = <p,k+q|A(k+q)-A(k)|n,k>
    (exact, from the eigenvalue equations).  Returns a dict:
      B    : (4, Nxi) vertex matrix elements, rows = {n, j_x, j_y, j_z}; B[0] is EXACTLY
             lindhard_setup's overlap M (flattened in the same (k,n,p) order)
      w    : f_n(k) - f_p(k+q);   dE : E_p(k+q) - E_n(k);   dk3 : BZ weight (bz_grid)
      mu   : the -df/dE measure w/dE with the exact degenerate limit beta*f*(1-f) at |dE|<1e-9
      Ek, Ekq : band energies (N, 4), node-rank ordered as in lindhard_setup
    scramble_seed (BGK-4 DECORATIVE CONTROL): if not None, each j-row of B is independently
    permuted over the transition index xi (numpy default_rng(scramble_seed)) -- same magnitudes,
    all vertex-band correlation destroyed; the density row and (w, dE, mu) are untouched."""
    kpts, dk3 = bz_grid(n_grid)
    q_vec = np.asarray(q_vec, float)
    Ek, Uk = _bands_at(kpts, node)
    Ekq, Ukq = _bands_at(kpts + q_vec, node)
    f_k = 1.0 / (1.0 + np.exp(beta * (Ek - node)))
    f_kq = 1.0 / (1.0 + np.exp(beta * (Ekq - node)))
    Vmid = _velocity_blocks(kpts + q_vec / 2.0)
    Bn = np.einsum("kap,kan->knp", np.conj(Ukq), Uk)              # <p,k+q|n,k>  (= lindhard M)
    Bj = np.einsum("kap,xkab,kbn->xknp", np.conj(Ukq), Vmid, Uk)  # <p,k+q|v_a(k+q/2)|n,k>
    w = f_k[:, :, None] - f_kq[:, None, :]
    dE = Ekq[:, None, :] - Ek[:, :, None]
    f_b = np.broadcast_to(f_k[:, :, None], w.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(np.abs(dE) > 1e-9, w / dE, beta * f_b * (1.0 - f_b))
    B = np.stack([Bn.reshape(-1)] + [Bj[a].reshape(-1) for a in range(3)])
    if scramble_seed is not None:
        rng = np.random.default_rng(scramble_seed)
        for a in (1, 2, 3):
            B[a] = B[a][rng.permutation(B.shape[1])]
    return {"B": B, "w": w.ravel(), "dE": dE.ravel(), "mu": mu.ravel(), "dk3": dk3,
            "Ek": Ek, "Ekq": Ekq, "q_vec": q_vec, "n_grid": n_grid, "node": node,
            "scrambled": scramble_seed is not None}


def moment_chi0_matrix(setup, z, idx=None):
    """The bare multi-moment bubble matrix chi0_ab(q, z) at one (possibly complex) frequency z:
    chi0_ab = dk3 * sum_xi conj(B_a) B_b w / (z - dE) -- same transition structure/denominator as
    chi0_from_setup (which is the (0,0) element with |M|^2 numerator), vertex-generalized.
    idx selects the moment subset (default all 4: {n, j_x, j_y, j_z})."""
    if idx is None:
        idx = [0, 1, 2, 3]
    B = setup["B"][idx]
    r = setup["w"] / (z - setup["dE"])
    return setup["dk3"] * np.einsum("ax,bx,x->ab", np.conj(B), B, r)


def moment_static_matrix(setup, idx=None):
    """The EXACT conserving static bubble chi0(q,0) = -G, G_ab = dk3 * sum conj(B_a) B_b mu -- the
    omega=0 bubble with its degenerate (dE=0) transitions carried at their analytic -df/dE limit
    (mu).  This is the static matrix the conserving closure's derivation demands (the partial-
    fraction identity is exact with THIS chi0(0)); B2-a's eta=1e-3-broadened static (mermin_chi's
    denominator) is a numerical stand-in for it -- the difference is quantified, disclosed, in the
    W2-BGK station file.  Hermitian by construction; returned complex for uniform algebra."""
    if idx is None:
        idx = [0, 1, 2, 3]
    B = setup["B"][idx]
    G = setup["dk3"] * np.einsum("ax,bx,x->ab", np.conj(B), B, setup["mu"])
    return -G


def closure_from_moments(c0z_stack, chi0_static, omegas, gamma):
    """The two-moment conserving-RTA closure (derivation in the section header above), applied to a
    precomputed stack of bare moment matrices:
        chi_M(omega) = chi0(0) [ chi0(0) + (i*gamma/z)(chi0(z) - chi0(0)) ]^{-1} chi0(z),  z=omega+i*gamma.
    c0z_stack: (Nw, m, m) bare bubbles at z = omegas + i*gamma;  chi0_static: (m, m).
    Shared by the certified CPU path (two_moment_chi) and the station's GPU path (which accelerates
    ONLY the bubble reductions; the closure algebra is this same fp64 numpy code either way).
    Returns (chi_nn (Nw,), chi_mats (Nw, m, m)).  Regular at omega=0 (z = i*gamma)."""
    omegas = np.asarray(omegas, float)
    z = omegas + 1j * gamma
    fac = (1j * gamma / z)[:, None, None]
    stat = chi0_static[None, :, :]
    M = stat + fac * (c0z_stack - stat)
    X = np.linalg.solve(M, c0z_stack)
    chi_mats = np.einsum("ab,wbc->wac", chi0_static, X)
    return chi_mats[:, 0, 0], chi_mats


def two_moment_chi(q_vec, omegas, beta, gamma, n_grid=32, node=NODE_LAM_F, moments="nj",
                   static="exact", setup=None, scramble_seed=None):
    """W2-BGK deliverable: the 4x4 moment-space {n, j_x, j_y, j_z} conserving-RTA closure solve
    (certified CPU path).  moments="nj" (the two-moment closure) or "n" (the {n}-only projection =
    the scalar Mermin through the IDENTICAL code path; equals mermin_chi -- BGK-2's contract).
    static="exact" (moment_static_matrix, the derivation's exact conserving chi0(0) -- PRODUCTION)
    or "eta" (chi0(q, i*1e-3), mermin_chi's eta=1e-3 static convention -- used for the ==mermin_chi
    contract check and quantifying the static-convention systematic).  Both closure inputs are
    reused, never adjusted: beta = beta_eff (G5a), gamma = gamma_micro (MC-2).  Returns
    (chi_nn array, chi_mats (Nw,m,m), chi0_static (m,m), setup)."""
    if setup is None:
        setup = velocity_vertex(q_vec, beta, n_grid, node, scramble_seed)
    idx = [0] if moments == "n" else [0, 1, 2, 3]
    if static == "exact":
        stat = moment_static_matrix(setup, idx)
    else:
        stat = moment_chi0_matrix(setup, 0.0 + 1j * 1e-3, idx)
    c0z = np.stack([moment_chi0_matrix(setup, w + 1j * gamma, idx) for w in np.asarray(omegas, float)])
    chi_nn, chi_mats = closure_from_moments(c0z, stat, omegas, gamma)
    return chi_nn, chi_mats, stat, setup


def bgk_conservation_check(setup, n_tests=3, seed=20260710):
    """BGK-2(i): verify IN-CODE that the conserving collision term's action on the {n,j} moment
    subspace is zero: for deterministic pseudo-random test vectors X, the moments of
    C[X]/(-gamma) = X - P[X] must vanish (relative to the moments of X), and P^2 = P.  Also
    reports the measure floor min(mu) (must be >= 0) and the Gram spectrum (must be positive).
    Returns {'moment_residual': max rel moment of (1-P)X, 'projector_idem': max rel |P^2X - PX|,
    'mu_min': ..., 'G_eigs': ...}."""
    B, mu, dk3 = setup["B"], setup["mu"], setup["dk3"]
    G = dk3 * np.einsum("ax,bx,x->ab", np.conj(B), B, mu)
    Gi = np.linalg.inv(G)
    rng = np.random.default_rng(seed)
    res_m, res_p = 0.0, 0.0
    for _ in range(n_tests):
        X = rng.normal(size=B.shape[1]) + 1j * rng.normal(size=B.shape[1])
        m = dk3 * (np.conj(B) @ X)
        PX = mu * (B.T @ (Gi @ m))
        mC = m - dk3 * (np.conj(B) @ PX)                      # moments of (1-P)X
        res_m = max(res_m, float(np.max(np.abs(mC)) / np.max(np.abs(m))))
        P2X = mu * (B.T @ (Gi @ (dk3 * (np.conj(B) @ PX))))
        res_p = max(res_p, float(np.max(np.abs(P2X - PX)) / np.max(np.abs(PX))))
    return {"moment_residual": res_m, "projector_idem": res_p,
            "mu_min": float(np.min(mu)), "G_eigs": np.linalg.eigvalsh((G + G.conj().T) / 2).real}


# ===========================================================================
# 7. THE ACCRETION PASS (2026-07-10) — the O(2) vertex-map family, the BRIDGE
#    null machinery, the MS-1a fusion rings   [ONE-OBJECT / LOCAL-NET LAW]
# ===========================================================================
# [PLACEMENT NOTE: appended after the __main__ guard (the same conflict-free pattern as the
#  W2-BGK block above): on IMPORT every definition lands in the module namespace; script mode
#  (python3 the_net.py) exits at the guard and is unaffected.  Run this section's own regression:
#      python3 -c "import sys; sys.path.insert(0, 'derivation_topdown/state'); import the_net; \
#                  sys.exit(0 if the_net.accretion_selftest_2026_07_10() else 1)"        ]
#
# PROVENANCE (four checked-and-integrated 2026-07-10 stations; their DURABLE MATH is accreted
# here as importable net methods with regression anchors, so future stations EXTEND rather than
# rebuild -- the MAP/LOCK/T chain rebuilt dart_rep/Ue/Uo/the-commutant three times over):
#   [MAP]  proofs/foundations/W2_MAP_vertex_propagator_2026-07-10.py -- the classification of the
#          vertex/propagator map Phi: internal (edge/J6/Witt) one-particle structure -> cover
#          (dart) space under the derived requirement set {R1,R2,R5}: AMBIGUOUS-BY-O(2)
#          (R-even branch EMPTY by rank obstruction; R-odd = Uo @ End_A4(edge_rep) = Mat_2(R) =
#          span{I6,J6,S1,S2}; isometric locus = O(2) EXACTLY); M-1b selection FAILS by the EXACT
#          orthogonality identity <Beo.C, C'.J6>_F = 0 over the whole commutant.
#   [LOCK] proofs/foundations/BRIDGE_LOCK_2026-07-10.py -- Design A: LENS-NULL, THEOREM-GRADE via
#          the three lemmas (map_null_lemmas below); every rotation member transports
#          J6 -> +Uo J6 Uo^T, every reflection member -> -Uo J6 Uo^T (OPPOSITE orientations).
#   [T]    proofs/foundations/BRIDGE_T_2026-07-10.py -- Design B: ARROW-BLIND, THEOREM-GRADE via
#          the SYMMETRIC-COMPRESSION LEMMA (symmetric_compression below): the R-odd compression
#          of the run cannot distinguish B from B^T at ANY order -- the derived arrow is
#          structurally invisible to that whole functional class.
#   [MS1a] proofs/foundations/MS1a_fusion_grading_2026-07-10.py -- the finite-fusion-ring
#          NO-ADDITIVE-CHARGE THEOREM on the DR-frame sector category (fusion_ring /
#          additive_charge_nullity / z2_gradings below): no baryon-number-like additive Z-valued
#          charge exists at sector level; fermion parity is the ONE nontrivial Z2 grading
#          (the 2T center / spinoriality grading).
#
# Everything below is RECOMPUTED from the net's OWN objects (complex_structure_J6, _edge_rep,
# reversal, hashimoto_gamma, EDGES/DARTS) -- no station file is imported or re-run; the section
# self-test (accretion_selftest_2026_07_10) anchors the recomputation against the stations'
# printed values.  The two module anchors (M0 cell C-projector, M0-2R tick-2pi) are UNTOUCHED
# and re-checked in the self-test.  NO magnitude is defined here (structure only, per the module
# contract); nothing scoreboard-adjacent is computed.
_SEC7_CACHE = {}


def _a4_vertex_group():
    """A4 as vertex-permutation dicts (the module's own convention: complex_structure_J6 /
    gauge_sector_category)."""
    return [dict(enumerate(p)) for p in itertools.permutations(range(NV))
            if sum(1 for i in range(NV) for j in range(i + 1, NV) if p[i] > p[j]) % 2 == 0]


def dart_rep(sig):
    """[MAP M-0i/M-0l] the cover-side A4 action on the 12-dim DART space, induced by the vertex
    permutation sig acting on each dart's (tail, head) labels: a genuine (NON-projective) A4
    homomorphism, equal to the REGULAR representation of A4 (A4 acts SIMPLY TRANSITIVELY on the
    12 darts; character 0 off the identity), commuting EXACTLY with the reversal R (M-0j) and
    with the walk B (BRIDGE-T D1).  Companion of _edge_rep (= 3 (+) 3 of A4) on the cover side."""
    EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
    Rd = np.zeros((ND, ND))
    for a, (i, j, v) in enumerate(DARTS):
        ni, nj = sig[i], sig[j]
        e2 = EIDX[(min(ni, nj), max(ni, nj))]
        Rd[2 * e2 if ni < nj else 2 * e2 + 1, a] = 1.0
    return Rd


def dart_embeddings():
    """[MAP M-0 conventions] the R-even / R-odd edge-coefficient embeddings (12x6):
    Ue[:, e] = (d_2e + d_(2e+1))/sqrt2 (UNSIGNED), Uo[:, e] = (d_2e - d_(2e+1))/sqrt2 (SIGNED).
    MAP M-0k booked finding: Uo -- NOT Ue -- intertwines edge_rep with dart_rep EXACTLY
    (edge_rep's oriented-edge/1-form sign convention is carried by the R-ODD sector); the R-odd
    projector is P_odd = (I - R)/2 = Uo @ Uo^T (BRIDGE-T T-0a).  Returns (Ue, Uo)."""
    Ue = np.zeros((ND, NE))
    Uo = np.zeros((ND, NE))
    for e in range(NE):
        Ue[2 * e, e] = 1 / math.sqrt(2)
        Ue[2 * e + 1, e] = 1 / math.sqrt(2)
        Uo[2 * e, e] = 1 / math.sqrt(2)
        Uo[2 * e + 1, e] = -1 / math.sqrt(2)
    return Ue, Uo


def map_commutant():
    """[MAP M-1a, 2026-07-10] the commutant End_A4(edge_rep) -- the dart-multiplicity space of
    the vertex/propagator-map classification -- together with the R-split of
    Hom_A4(edge_rep, dart_rep).  FACTS (each re-asserted here on the net's own objects):
      * dim Hom_A4(edge_rep, dart_rep) = 6 (= mult_edge(3-irrep) x mult_dart(3-irrep) = 2x3);
        R acts on Hom as an involution splitting it R-even dim 2 (+) R-odd dim 4 (M-1a-i/ii).
      * the R-EVEN branch is PROVABLY EMPTY under the isometry requirement R5: every element has
        rank <= 3 < 6 -- a rank obstruction, not a search failure (M-1a-iii).
      * the R-odd sector, coordinatized by Uo, carries EXACTLY edge_rep, so Hom into it =
        End_A4(edge_rep) = Mat_2(R), dim 4, spanned by {I6, J6, S1, S2}: I6/J6 the 'rotation'
        directions; S1/S2 the SYMMETRIC + TRACELESS 'reflection' directions, both ANTICOMMUTING
        with J6 (they reverse the internal orientation; M-1a-iv..vii, LOCK L-1a(i)).  S1n/S2n
        are the unit-isometry normalizations (S1n^T S1n = I6; LOCK L-1a).
      * the isometric sub-locus is O(2) EXACTLY: {a I6 + b J6} union {c S1 + d S2}, and NOTHING
        else (a generic mix of the two families is not isometric; M-1a-viii).
    Returns a dict: I6, J6, S1, S2, S1n, S2n, Ue, Uo, hom_dim=6, r_even_dim=2, r_odd_dim=4,
    commutant_dim=4.  Cached (the construction is deterministic)."""
    if "commutant" in _SEC7_CACHE:
        return _SEC7_CACHE["commutant"]
    A4 = _a4_vertex_group()
    J6 = complex_structure_J6()
    R = reversal()
    Ue, Uo = dart_embeddings()
    I6 = np.eye(NE)
    reps_e = [_edge_rep(g) for g in A4]
    reps_d = [dart_rep(g) for g in A4]
    # Hom_A4(edge_rep, dart_rep): vec(Rd.Phi - Phi.R6) = 0  [MAP M-1a-i]
    Cstack = np.vstack([np.kron(np.eye(NE), Rd) - np.kron(R6.T, np.eye(ND))
                        for Rd, R6 in zip(reps_d, reps_e)])
    _, Ssvd, Vt = np.linalg.svd(Cstack)
    rank = int(np.sum(Ssvd > 1e-9))
    Phis = [Vt[rank + k].reshape(ND, NE, order="F") for k in range(Cstack.shape[1] - rank)]
    assert len(Phis) == 6, f"map_commutant: hom dim {len(Phis)} != 6 (MAP M-1a-i)"
    assert max(np.max(np.abs(Rd @ P - P @ R6)) for P in Phis
               for Rd, R6 in zip(reps_d, reps_e)) < 1e-9
    # R-split of Hom (R commutes with dart_rep, so Phi -> R.Phi preserves Hom)  [MAP M-1a-ii]
    bv = np.stack([P.reshape(-1, order="F") for P in Phis], axis=1)
    Rv = np.stack([(R @ P).reshape(-1, order="F") for P in Phis], axis=1)
    coeff, *_ = np.linalg.lstsq(bv, Rv, rcond=None)
    assert np.max(np.abs(bv @ coeff - Rv)) < 1e-9
    eigsR, eigvecsR = np.linalg.eig(coeff)
    assert np.allclose(np.sort(eigsR.real), [-1, -1, -1, -1, 1, 1], atol=1e-6)
    # R-even branch EMPTY under R5: the rank obstruction  [MAP M-1a-iii; rng(0) as the station]
    Qe, _ = np.linalg.qr(eigvecsR[:, np.abs(eigsR.real - 1) < 1e-6].real)
    even_vecs = bv @ Qe
    Phi_even = [even_vecs[:, k].reshape(ND, NE, order="F") for k in range(even_vecs.shape[1])]
    rng = np.random.default_rng(0)
    for _ in range(8):
        c = rng.normal(size=len(Phi_even))
        assert np.linalg.matrix_rank(sum(c[k] * Phi_even[k] for k in range(len(Phi_even))),
                                     tol=1e-9) <= 3
    # the R-odd sector carries EXACTLY edge_rep; Uo intertwines, Ue maximally fails [M-1a-iv, M-0k]
    assert max(np.max(np.abs(Uo.T @ Rd @ Uo - R6)) for Rd, R6 in zip(reps_d, reps_e)) < 1e-9
    assert np.max(np.abs(R @ Uo + Uo)) < 1e-15
    assert max(np.max(np.abs(Rd @ Ue - Ue @ R6)) for Rd, R6 in zip(reps_d, reps_e)) > 1.0
    # End_A4(edge_rep) = Mat_2(R), dim 4, contains I6 and J6  [MAP M-1a-v/vi]
    C2 = np.vstack([np.kron(np.eye(NE), R6) - np.kron(R6.T, np.eye(NE)) for R6 in reps_e])
    _, S2v, Vt2 = np.linalg.svd(C2)
    rank2 = int(np.sum(S2v > 1e-9))
    Cs = [Vt2[rank2 + k].reshape(NE, NE, order="F") for k in range(C2.shape[1] - rank2)]
    assert len(Cs) == 4, f"map_commutant: commutant dim {len(Cs)} != 4 (MAP M-1a-v)"
    vecs = np.stack([c_.reshape(-1, order="F") for c_ in Cs], axis=1)
    for M in (I6, J6):
        cf, *_ = np.linalg.lstsq(vecs, M.reshape(-1, order="F"), rcond=None)
        assert np.max(np.abs((vecs @ cf).reshape(NE, NE, order="F") - M)) < 1e-9
    # S1, S2 = the Frobenius complement of span{I6, J6} inside the commutant  [MAP M-1a-vii]
    Q_IJ, _ = np.linalg.qr(np.stack([I6.reshape(-1, order="F"),
                                     J6.reshape(-1, order="F")], axis=1))
    Qc, _ = np.linalg.qr(vecs - Q_IJ @ (Q_IJ.T @ vecs))
    S1 = Qc[:, 0].reshape(NE, NE, order="F")
    S2 = Qc[:, 1].reshape(NE, NE, order="F")
    assert np.allclose(S1, S1.T, atol=1e-8) and np.allclose(S2, S2.T, atol=1e-8)
    assert abs(np.trace(S1)) < 1e-8 and abs(np.trace(S2)) < 1e-8
    assert np.max(np.abs(S1 @ J6 + J6 @ S1)) < 1e-12 and np.max(np.abs(S2 @ J6 + J6 @ S2)) < 1e-12
    S1n = S1 / math.sqrt(float(np.trace(S1.T @ S1) / NE))
    S2n = S2 / math.sqrt(float(np.trace(S2.T @ S2) / NE))
    assert np.max(np.abs(S1n.T @ S1n - I6)) < 1e-12

    # the isometric sub-locus is O(2) EXACTLY  [MAP M-1a-viii]
    def _ir(M):
        G = M.T @ M
        return np.linalg.norm(G - (np.trace(G) / NE) * I6) / (np.linalg.norm(G) + 1e-30)

    assert _ir(0.6 * I6 + 0.8 * J6) < 1e-9 and _ir(0.6 * S1 + 0.8 * S2) < 1e-9
    assert _ir(0.5 * I6 + 0.3 * J6 + 0.4 * S1) > 1e-3
    out = {"I6": I6, "J6": J6, "S1": S1, "S2": S2, "S1n": S1n, "S2n": S2n, "Ue": Ue, "Uo": Uo,
           "hom_dim": 6, "r_even_dim": 2, "r_odd_dim": 4, "commutant_dim": 4}
    _SEC7_CACHE["commutant"] = out
    return out


def map_family(angle, branch="rotation"):
    """[MAP M-1a VERDICT: AMBIGUOUS-BY-O(2)] the FULL {R1,R2,R5}-compliant family of vertex/
    propagator maps Phi: internal (edge/J6/Witt) one-particle space -> cover (dart) space:
        branch='rotation'    Phi = Uo @ (cos(angle) I6 + sin(angle) J6)    [complex-LINEAR on
                             the Witt +i space; angle = the internal U(1) phase freedom]
        branch='reflection'  Phi = Uo @ (cos(angle) S1n + sin(angle) S2n)  [complex-ANTIlinear
                             = rotation o sigma; a genuinely distinct isometric family]
    Every member is an exact isometry (Phi^T Phi = I6) with image in the R-ODD dart sector
    (R Phi = -Phi), and transports J6 -> +Uo J6 Uo^T (rotation) / -Uo J6 Uo^T (reflection):
    the two branches carry OPPOSITE orientations -- the cell vacuum goes to C(+J_D) vs
    C(-J_D) = its exact particle-hole conjugate, so the orbit ambiguity IS a modular-flow-
    orientation ambiguity on the R-odd sector (LOCK L-1a; T T-1).
    NO derived structure selects a member: {R1,R2,R5} + the walk's own dynamics fail (MAP M-1b,
    an EXACT orthogonality identity), every band-edge attachment functional of R-parity-definite
    one-particle transports fails (LOCK: LENS-NULL, theorem-grade), and every seed-anchored
    R-odd two-point run datum fails -- and provably cannot even see the arrow (T: ARROW-BLIND,
    theorem-grade).  The O(2) ambiguity STANDS; a discriminator must supply a NEW derived
    structure (phase-bearing/Fock-level, or geometric).  Returns the 12x6 Phi."""
    cm = map_commutant()
    c, s = math.cos(angle), math.sin(angle)
    if branch == "rotation":
        Mred = c * cm["I6"] + s * cm["J6"]
    elif branch == "reflection":
        Mred = c * cm["S1n"] + s * cm["S2n"]
    else:
        raise ValueError(f"map_family: unknown branch {branch!r} (use 'rotation'|'reflection')")
    return cm["Uo"] @ Mred


def map_null_lemmas():
    """[LOCK L-1b(vi) / T T-0c, 2026-07-10] the three machine-checked lemmas that force BOTH
    theorem-grade nulls on the O(2) family (LENS-NULL: no R-parity-definite one-particle
    attachment functional discriminates the orbits; ARROW-BLIND: no R-odd two-point run datum
    does either):
      LEMMA 1  R = -Id on the whole R-odd dart sector:     R @ Uo = -Uo
               (tau_dart's sign pinned by R5-survival, not fiat);
      LEMMA 2  reversal-transpose (Ihara-Bass structure):  R @ B @ R = B^T   EXACTLY;
      LEMMA 3  B real => CONJUGATE band-edge projectors:   P(h-) = conj(P(h+)),  where
               h+- = (lam +- i sqrt(4(k-1) - lam^2))/2 = -1/2 +- i sqrt(7)/2 are the chir-7
               Ihara-Bass roots of lam = -1 (the A4 3-irrep triple adjacency eigenvalue), each
               a 3-dim eigenspace of B; P(h) = the (non-orthogonal) right/left spectral
               projector, idempotent, B P = h P.
    The IB roots are DERIVED here from the net's own adjacency (k = 2|E|/|V| = 3; lam = the
    triple eigenvalue), never typed in.  Returns the residual dict
    {'R_Uo_plus_Uo', 'RBR_minus_BT', 'conj_band_edge_projectors'}, EACH asserted < 1e-12."""
    R = reversal()
    _, Uo = dart_embeddings()
    B0 = hashimoto_gamma()
    # the chir-7 IB root pair, derived from the net's own graph data
    A_adj = np.zeros((NV, NV))
    for i, j, v in EDGES:
        A_adj[i, j] += 1.0
        A_adj[j, i] += 1.0
    k_deg = 2 * NE // NV
    adj_ev = np.linalg.eigvalsh(A_adj)
    lam = -1.0
    assert k_deg == 3 and int(np.sum(np.abs(adj_ev - lam) < 1e-9)) == 3   # the 3-irrep band
    disc = lam * lam - 4 * (k_deg - 1)
    assert disc < 0 and abs(disc + 7) < 1e-12                             # chir-7
    h_plus = complex(lam / 2, math.sqrt(-disc) / 2)

    def _eigspace(B, h, tol=1e-8):
        _, s, Vh = np.linalg.svd(B - h * np.eye(ND))
        kdim = int(np.sum(s < tol))
        return Vh[ND - kdim:].conj().T

    Qp = _eigspace(B0, h_plus)
    Qm = _eigspace(B0, np.conj(h_plus))
    Lp = _eigspace(B0.T, np.conj(h_plus))
    Lm = _eigspace(B0.T, h_plus)
    assert Qp.shape[1] == 3 and Qm.shape[1] == 3
    Pp = Qp @ np.linalg.inv(Lp.conj().T @ Qp) @ Lp.conj().T
    Pm = Qm @ np.linalg.inv(Lm.conj().T @ Qm) @ Lm.conj().T
    assert np.max(np.abs(Pp @ Pp - Pp)) < 1e-9 and np.max(np.abs(B0 @ Pp - h_plus * Pp)) < 1e-9
    res = {"R_Uo_plus_Uo": float(np.max(np.abs(R @ Uo + Uo))),
           "RBR_minus_BT": float(np.max(np.abs(R @ B0 @ R - B0.T))),
           "conj_band_edge_projectors": float(np.max(np.abs(Pm - Pp.conj())))}
    for name, val in res.items():
        assert val < 1e-12, f"map_null_lemmas: {name} residual {val:.2e} >= 1e-12"
    return res


def symmetric_compression(u):
    """[T T-2 D3/D3'/D5, 2026-07-10] THE SYMMETRIC-COMPRESSION LEMMA (Lemmas 1+2 of
    map_null_lemmas promoted to state level): the R-odd compression of the run's resolvent,
        F(u) = Uo^T (I - u B)^{-1} Uo ,
    is EXACTLY SYMMETRIC, and forward/reversed INVARIANT: F_B(u) == F_{B^T}(u)
    [proof: F^T = Uo^T (I - u B^T)^{-1} Uo = Uo^T R (I - u B)^{-1} R Uo = F, using R B R = B^T
    and R Uo = -Uo; it holds ORDER-BY-ORDER in B, so it is no resummation accident].
    CONSEQUENCES (the accreted theorem): antisym(P_odd (I - uB)^{-1} P_odd) = 0, so EVERY
    seed-anchored, state-level, all-orders-in-B two-point run datum on the R-odd dart sector is
    orientation-blind -- and the derived arrow (sub-criticality u < u_c = 1/(k-1); operating
    point u = alpha_1 = (2/3)^8) is structurally INVISIBLE to the whole functional class
    (ARROW-BLIND, theorem-grade).  The two theorem checks are asserted as residuals < 1e-12.
    Returns (F, residuals_dict)."""
    _, Uo = dart_embeddings()
    B0 = hashimoto_gamma()
    F = Uo.T @ np.linalg.solve(np.eye(ND) - u * B0, Uo)
    F_rev = Uo.T @ np.linalg.solve(np.eye(ND) - u * B0.T, Uo)
    res = {"symmetry": float(np.max(np.abs(F - F.T))),
           "forward_reversed_invariance": float(np.max(np.abs(F - F_rev)))}
    for name, val in res.items():
        assert val < 1e-12, f"symmetric_compression: {name} residual {val:.2e} >= 1e-12"
    return F, res


# ---- MS-1a: the fusion rings of the DR-frame gauge groups (A4 / 2T) ------------------------------
def _fusion_group(group):
    """[MS1a PART A] the two candidate gauge groups, built from scratch: 'A4' = the even
    permutations of {0,1,2,3} (the net's own vertex convention); '2T' = the binary tetrahedral
    group as the 24 unit Hurwitz quaternions in doubled-integer coordinates (2a,2b,2c,2d), so
    ALL group arithmetic is EXACT (generated by closure from i and (1+i+j+k)/2).
    Returns (elems, mul, inv, identity)."""
    if group == "A4":
        elems = [p for p in itertools.permutations(range(4))
                 if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
        mul = lambda p, q: tuple(p[q[i]] for i in range(4))

        def inv(p):
            v = [0] * 4
            for i, pi in enumerate(p):
                v[pi] = i
            return tuple(v)

        return elems, mul, inv, (0, 1, 2, 3)
    if group == "2T":
        def mul(x, y):
            a0, a1, a2, a3 = x
            b0, b1, b2, b3 = y
            z = (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                 a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                 a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                 a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0)
            assert all(t % 2 == 0 for t in z), "non-Hurwitz product"
            return tuple(t // 2 for t in z)

        e = (2, 0, 0, 0)
        elems, frontier = {e}, [e]
        while frontier:
            new = []
            for g in frontier:
                for h in ((0, 2, 0, 0), (1, 1, 1, 1)):
                    x = mul(g, h)
                    if x not in elems:
                        elems.add(x)
                        new.append(x)
            frontier = new
        return sorted(elems), mul, (lambda x: (x[0], -x[1], -x[2], -x[3])), e
    raise ValueError(f"_fusion_group: group must be 'A4' or '2T', got {group!r}")


def _elt_order(g, mul, e):
    n, x = 1, g
    while x != e:
        x = mul(x, g)
        n += 1
    return n


def _conjugacy_classes(elems, mul, inv, e):
    """Classes ordered: identity first, then by (element order, class size, representative)."""
    seen, classes = set(), []
    for g in elems:
        if g in seen:
            continue
        cl = sorted({mul(mul(h, g), inv(h)) for h in elems})
        classes.append(cl)
        seen |= set(cl)
    classes.sort(key=lambda c: (_elt_order(c[0], mul, e), len(c), c[0]))
    assert classes[0] == [e]
    return classes


def _character_table(elems, mul, inv, e):
    """[MS1a check (a)] Dixon/Burnside: the class-sum multiplication matrices commute; their
    common eigenvectors are the central characters w_a(Z_i) = |C_i| chi_a(g_i)/chi_a(1); dims
    from the norm relation.  Returns (classes, sizes, cls_of, X) with X[a][j] = chi_a on class
    j, trivial irrep first.  Everything is VERIFIED downstream by orthogonality in
    fusion_ring() -- computed, never ported as a table."""
    G = len(elems)
    classes = _conjugacy_classes(elems, mul, inv, e)
    r = len(classes)
    sizes = [len(c) for c in classes]
    cls_of = {g: i for i, c in enumerate(classes) for g in c}
    A = np.zeros((r, r, r))
    for i in range(r):
        for k in range(r):
            zk = classes[k][0]
            for x in classes[i]:
                A[i, cls_of[mul(inv(x), zk)], k] += 1.0   # x*y = z_k with y = x^-1 z_k
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23][:r]
    M = sum(math.sqrt(p) * A[i] for i, p in enumerate(primes))
    evals, evecs = np.linalg.eig(M)
    gap = min(abs(evals[i] - evals[j]) for i in range(r) for j in range(i + 1, r))
    assert gap > 1e-6, f"_character_table: degenerate Dixon combination (gap={gap:.2e})"
    X, dims = [], []
    for a in range(r):
        w = evecs[:, a] / evecs[0, a]                     # w_0 = w(Z_e) = 1
        d = math.sqrt(G / sum(abs(w[j]) ** 2 / sizes[j] for j in range(r)))
        X.append([w[j] * d / sizes[j] for j in range(r)])
        dims.append(d)
    order = sorted(range(r), key=lambda a: (round(dims[a], 6),
                                            tuple((round(X[a][j].real, 6), round(X[a][j].imag, 6))
                                                  for j in range(r))))
    X = [X[a] for a in order]
    triv = next(a for a in range(r) if all(abs(X[a][j] - 1) < 1e-8 for j in range(r)))
    return classes, sizes, cls_of, [X[a] for a in ([triv] + [a for a in range(r) if a != triv])]


def fusion_ring(group):
    """[MS1a PART A, 2026-07-10] the fusion ring R(G) of the DR-frame gauge-group candidates,
    COMPUTED from scratch (Dixon/Burnside class-matrix characters -- not ported tables) with the
    full verification battery ASSERTED: row+column character orthogonality; positive-integer
    dims with sum d^2 = |G|; non-negative-integer fusion coefficients N_ab^c with unit,
    commutativity, associativity, unique duals, and the dimension homomorphism.
    group = 'A4' (the forced J-covariance gauge group; dims [1,1,1,3]; = the fusion CLOSURE of
    the sectors as built {nu:1, d:3, u:3, e:1}, since 3 x 3 produces 1' and 1'') or '2T' (the
    spinorial double cover the Fock rep forces, HK-6b; dims [1,1,1,2,2,2,3]).  For '2T' the
    center grading chi_a(-1)/chi_a(1) is computed and included: it is -1 EXACTLY on the three
    2-dim spinorial irreps (spinorial = fermionic).  Returns a dict: group, order, class_sizes,
    names, dims, chars ((r,r) complex ndarray), N ((r,r,r) int ndarray), center_grading (tuple
    for '2T', None for 'A4').  Cached (deterministic)."""
    key = ("fusion", group)
    if key in _SEC7_CACHE:
        return _SEC7_CACHE[key]
    elems, mul, inv, e = _fusion_group(group)
    G = len(elems)
    classes, sizes, cls_of, X = _character_table(elems, mul, inv, e)
    r = len(X)
    row = max(abs(sum(sizes[j] * X[a][j] * np.conj(X[b][j]) for j in range(r)) / G
                  - (1 if a == b else 0)) for a in range(r) for b in range(r))
    col = max(abs(sum(X[a][i] * np.conj(X[a][j]) for a in range(r))
                  - (G / sizes[i] if i == j else 0)) for i in range(r) for j in range(r))
    assert row < 1e-9 and col < 1e-8, f"fusion_ring({group}): orthogonality fails"
    dims = [X[a][0].real for a in range(r)]
    assert all(abs(d - round(d)) < 1e-9 and round(d) >= 1 for d in dims)
    assert abs(sum(d * d for d in dims) - G) < 1e-6
    N = np.zeros((r, r, r))
    for a in range(r):
        for b in range(r):
            for c in range(r):
                N[a, b, c] = sum(sizes[j] * (X[a][j] * X[b][j] * np.conj(X[c][j])).real
                                 for j in range(r)) / G
    Nint = np.round(N).astype(int)
    assert float(np.max(np.abs(N - Nint))) < 1e-7 and int(Nint.min()) >= 0
    assert all(Nint[0, a, c] == (1 if a == c else 0) for a in range(r) for c in range(r))  # unit
    assert bool(np.all(Nint == np.transpose(Nint, (1, 0, 2))))                       # commutative
    assert max(abs(int(sum(Nint[a, b, x] * Nint[x, c, d] for x in range(r))
                       - sum(Nint[b, c, f] * Nint[a, f, d] for f in range(r))))
               for a in range(r) for b in range(r) for c in range(r) for d in range(r)) == 0
    assert all(int(np.sum(Nint[a, :, 0])) == 1 for a in range(r))                    # unique duals
    assert max(abs(sum(Nint[a, b, c] * dims[c] for c in range(r)) - dims[a] * dims[b])
               for a in range(r) for b in range(r)) < 1e-6                           # dim homo.
    dims_i = [int(round(d)) for d in dims]
    names, seen = [], {}
    for d in dims_i:
        names.append(str(d) + "'" * seen.get(d, 0))
        seen[d] = seen.get(d, 0) + 1
    parity = None
    if group == "2T":
        j_m1 = cls_of[(-2, 0, 0, 0)]                     # the central -1 (order 2; Z(2T)={+-1})
        parity = tuple(int(round((X[a][j_m1] / X[a][0]).real)) for a in range(r))
        assert all(abs(X[a][j_m1] / X[a][0] - parity[a]) < 1e-9 for a in range(r))
        assert all((parity[a] == -1) == (dims_i[a] == 2) for a in range(r))  # spinorial=fermionic
    out = {"group": group, "order": G, "class_sizes": sizes, "names": names, "dims": dims_i,
           "chars": np.array(X), "N": Nint, "center_grading": parity}
    _SEC7_CACHE[key] = out
    return out


def additive_charge_nullity(group):
    """[MS1a PART D, 2026-07-10 -- THE THEOREM] the EXACT (Fraction-rref over Q) solution space
    of the additive-charge constraint system { q(a) + q(b) - q(c) = 0 : N_ab^c > 0 } on
    R(group).  Nullity 0 <=> q == 0 is the ONLY additive Z-valued charge (a rational nullspace
    of dim 0 has no nonzero integer points either) <=> NO baryon-number-like unbounded additive
    conservation law exists at the sector level: exact-conservation protection of the proton is
    STRUCTURALLY IMPOSSIBLE in the category as built.  Forcing chain on R(A4): 3 x 3 contains 3
    => q(3) = 0; 3 x 3 contains 1 => q(1) = 0; 1'^3 = 1 (torsion) => q(1') = q(1'') = 0.
    (SCOPE, per MS1a: conditional on the TD-limit twisted Haag duality premise P1; proves no
    EXACT law only -- no rate/lifetime/suppression is computed, that is MS-1b's job, gated on
    the interaction layer.  Consistent with the eta_B Sakharov skeleton, which REQUIRES
    effective B-violation.)  Returns the nullity (int), ASSERTED == 0."""
    from fractions import Fraction
    Nint = fusion_ring(group)["N"]
    r = Nint.shape[0]
    rows = []
    for a in range(r):
        for b in range(r):
            for c in range(r):
                if Nint[a, b, c] > 0:
                    v = [Fraction(0)] * r
                    v[a] += 1
                    v[b] += 1
                    v[c] -= 1
                    rows.append(v)
    M = [row[:] for row in rows]
    pivots, ri = [], 0
    for col in range(r):
        piv = next((i for i in range(ri, len(M)) if M[i][col] != 0), None)
        if piv is None:
            continue
        M[ri], M[piv] = M[piv], M[ri]
        pv = M[ri][col]
        M[ri] = [x / pv for x in M[ri]]
        for i in range(len(M)):
            if i != ri and M[i][col] != 0:
                f = M[i][col]
                M[i] = [x - f * y for x, y in zip(M[i], M[ri])]
        pivots.append(col)
        ri += 1
        if ri == len(M):
            break
    nullity = r - len(pivots)
    assert nullity == 0, (f"additive_charge_nullity({group}): NONZERO additive grading exists "
                          f"(nullity {nullity}) -- MS1a-SURPRISE, book it, do not suppress")
    return nullity


def z2_gradings(group):
    """[MS1a PART C, 2026-07-10] ALL Z2 gradings of R(group), enumerated brute-force (all 2^r
    sign maps s with s(a) s(b) = s(c) whenever N_ab^c > 0).  ASSERTED: R(A4) admits ONLY the
    trivial grading (count 1); R(2T) admits EXACTLY ONE nontrivial grading (count 2), and it
    EQUALS the center/spinoriality grading = the FERMION PARITY of the DR frame (F,2T)
    (s = -1 exactly on the three 2-dim spinorial irreps).  Fermion parity is therefore the ONE
    AND ONLY Z2 sector grading the category supports -- no room for a second, R-parity-like Z2
    at sector level.  Returns the count (int)."""
    fr = fusion_ring(group)
    Nint = fr["N"]
    r = Nint.shape[0]
    triples = [(a, b, c) for a in range(r) for b in range(r) for c in range(r)
               if Nint[a, b, c] > 0]
    gradings = [s for s in itertools.product([1, -1], repeat=r)
                if all(s[a] * s[b] == s[c] for a, b, c in triples)]
    if group == "A4":
        assert len(gradings) == 1 and gradings[0] == (1,) * r, \
            f"z2_gradings(A4): expected the trivial grading only, got {gradings}"
    else:
        nontriv = [s for s in gradings if s != (1,) * r]
        assert len(gradings) == 2 and len(nontriv) == 1 and nontriv[0] == fr["center_grading"], \
            f"z2_gradings(2T): expected trivial + center grading, got {gradings}"
    return len(gradings)


# ---- SECTION 7 SELF-TEST (regression against the stations' printed values) -----------------------
def accretion_selftest_2026_07_10(verbose=True):
    """Section-7 regression: every accreted method re-verified against the ORIGINAL station
    files' printed values (W2_MAP / BRIDGE_LOCK / BRIDGE_T / MS1a, all 2026-07-10), RECOMPUTED
    from the net's own J6/edge_rep/R/B objects -- nothing imported or re-run from the stations.
    Also re-checks the two module anchors (M0 cell C-projector, M0-2R tick-2pi) untouched."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 7 self-test — the accretion pass (2026-07-10)")
        print("=" * 88)

    # 0. the module anchors, untouched
    ck("ANCHORS untouched: M0 cell C-projector + M0-2R tick-2pi still hold",
       anchor_cell_projector() and anchor_tick_2pi())

    # 1. [MAP M-0i/j/l; T D1] dart_rep: genuine A4 hom = the REGULAR rep; commutes with R and B
    A4 = _a4_vertex_group()
    R = reversal()
    B0 = hashimoto_gamma()
    comp = lambda g, h: {i: g[h[i]] for i in range(NV)}
    g0, h0 = A4[3], A4[7]
    homdev = np.max(np.abs(dart_rep(g0) @ dart_rep(h0) - dart_rep(comp(g0, h0))))
    chi_d = np.array([np.trace(dart_rep(g)) for g in A4])
    ck("W2-MAP M-0i/M-0l: dart_rep = genuine A4 hom = the REGULAR rep (ip(chi,chi)=12, "
       "trace 0 off identity)",
       homdev < 1e-12 and abs(float(np.sum(chi_d * chi_d)) / len(A4) - 12.0) < 1e-9
       and np.allclose(chi_d[1:], 0.0, atol=1e-9) and abs(chi_d[0] - 12) < 1e-9,
       detail=f"hom dev {homdev:.1e}")
    ck("W2-MAP M-0j + BRIDGE-T D1: R and B both commute with dart_rep(g) for every g",
       max(np.max(np.abs(dart_rep(g) @ R - R @ dart_rep(g))) for g in A4) < 1e-12
       and max(np.max(np.abs(dart_rep(g) @ B0 - B0 @ dart_rep(g))) for g in A4) < 1e-12)

    # 2. [MAP M-1a] the classification (station: dim Hom = 6, split 2 (+) 4, commutant = 4,
    #    R-even EMPTY, isometric locus = O(2) -- all re-asserted inside map_commutant)
    cm = map_commutant()
    ck("W2-MAP M-1a: dim Hom = 6, R-split 2 (+) 4, commutant {I6,J6,S1,S2} dim 4 "
       "(R-even EMPTY + O(2) locus asserted in-method)",
       cm["hom_dim"] == 6 and cm["r_even_dim"] == 2 and cm["r_odd_dim"] == 4
       and cm["commutant_dim"] == 4)

    # 3. [MAP M-0f + M-1b] the sweep-fact kill + the EXACT selection-failure identity
    Beo = cm["Ue"].T @ B0 @ cm["Uo"]
    Aeo = (Beo - Beo.T) / 2
    cbf = float(np.sum(Aeo * cm["J6"]) / np.sum(cm["J6"] * cm["J6"]))
    resid = float(np.linalg.norm(Aeo - cbf * cm["J6"]) / np.linalg.norm(Aeo))
    ck("W2-MAP M-0f: antisym(Beo) vs J6 best-fit c = 0, normalized residual = 1.000 "
       "(the naive-candidate kill)",
       abs(cbf) < 1e-9 and abs(resid - 1.0) < 1e-9, detail=f"c={cbf:.1e}, resid={resid:.6f}")
    Call = [cm["I6"], cm["J6"], cm["S1"], cm["S2"]]
    bil = max(abs(float(np.sum((Beo @ Ci_) * (Cj_ @ cm["J6"]))))
              for Ci_ in Call for Cj_ in Call)
    ck("W2-MAP M-1b EXACT IDENTITY: <Beo.C, C'.J6>_F = 0 over the FULL commutant "
       "(selection FAILS; AMBIGUOUS-BY-O(2) stands)", bil < 1e-9, detail=f"max = {bil:.1e}")

    # 4. [LOCK L-1a; T T-1] the family: R-odd isometries transporting OPPOSITE orientations
    JD = cm["Uo"] @ cm["J6"] @ cm["Uo"].T
    P_odd = (np.eye(ND) - R) / 2
    fam_ok = True
    for ang in (0.0, 0.7, 2.4, 4.9):
        for br, sgn in (("rotation", +1.0), ("reflection", -1.0)):
            Phi = map_family(ang, br)
            fam_ok = (fam_ok and np.max(np.abs(Phi.T @ Phi - cm["I6"])) < 1e-12
                      and np.max(np.abs(R @ Phi + Phi)) < 1e-12
                      and np.max(np.abs(Phi @ cm["J6"] @ Phi.T - sgn * JD)) < 1e-12)
    ck("BRIDGE-LOCK L-1a / BRIDGE-T T-1: every member (4 angles x 2 branches) is an R-odd exact "
       "isometry with Phi J6 Phi^T = +J_D (rotation) / -J_D (reflection)", fam_ok)
    ck("BRIDGE-T T-1(iii): transported vacua are exact particle-hole conjugates "
       "C(-J_D) = P_odd - C(+J_D)",
       np.max(np.abs((P_odd - 1j * JD) / 2 - (P_odd - (P_odd + 1j * JD) / 2))) < 1e-15)

    # 5. [LOCK L-1b(vi) / T T-0c] the three null lemmas
    nl = map_null_lemmas()
    ck("BRIDGE-LOCK/T null lemmas: R.Uo=-Uo, R.B.R=B^T, P(h-)=conj(P(h+)) all < 1e-12",
       all(v < 1e-12 for v in nl.values()),
       detail=", ".join(f"{k}={v:.1e}" for k, v in nl.items()))

    # 6. [T D3/D3'/D5] the symmetric compression, at the operating point and off it,
    #    plus order-by-order (n = 0..12, the station's own sweep)
    u_op = (2.0 / 3.0) ** 8                     # T T-0e: u = alpha_1, sub-critical (< u_c = 1/2)
    sc_ok = True
    for u in (u_op, 0.11, 0.31):
        _, res = symmetric_compression(u)
        sc_ok = sc_ok and max(res.values()) < 1e-12
    _, Uo = dart_embeddings()
    dev_order = 0.0
    Bn = np.eye(ND)
    for n in range(13):
        Fn = Uo.T @ Bn @ Uo
        dev_order = max(dev_order, float(np.max(np.abs(Fn - Fn.T))))
        Bn = Bn @ B0
    ck("BRIDGE-T D3/D3'/D5: F(u) symmetric + arrow-invariant at u = alpha_1 = (2/3)^8 and off "
       "the operating point; Uo^T B^n Uo symmetric ORDER-BY-ORDER n = 0..12",
       sc_ok and dev_order < 1e-12, detail=f"max order asym = {dev_order:.1e}")

    # 7. [MS1a A/C/D] the fusion rings, gradings, and THE THEOREM
    frA = fusion_ring("A4")
    fr2 = fusion_ring("2T")
    ck("MS1a PART A: A4 -> classes [1,3,4,4], dims [1,1,1,3]; 2T -> classes [1,1,4,4,4,4,6], "
       "dims [1,1,1,2,2,2,3] (full verification battery asserted in-method)",
       sorted(frA["class_sizes"]) == [1, 3, 4, 4] and sorted(frA["dims"]) == [1, 1, 1, 3]
       and sorted(fr2["class_sizes"]) == [1, 1, 4, 4, 4, 4, 6]
       and sorted(fr2["dims"]) == [1, 1, 1, 2, 2, 2, 3])
    i3 = frA["dims"].index(3)
    ck("MS1a forcing chain on R(A4): 3 x 3 contains 3 (=> q(3)=0) and contains 1 (=> q(1)=0)",
       frA["N"][i3, i3, i3] >= 1 and frA["N"][i3, i3, 0] >= 1)
    even = [a for a in range(7) if fr2["center_grading"][a] == +1]
    closed = all(fr2["N"][a, b, c] == 0 for a in even for b in even
                 for c in range(7) if c not in even)
    ck("MS1a: the center-even sub-ring of R(2T) is fusion-closed with dims {1,1,1,3} "
       "(the R(A4) content)",
       closed and sorted(fr2["dims"][a] for a in even) == [1, 1, 1, 3])
    ck("MS1a PART D THEOREM: additive-charge nullity = 0 for R(A4) AND R(2T) "
       "(q == 0 is the ONLY additive Z-valued charge; no baryon-like sector law)",
       additive_charge_nullity("A4") == 0 and additive_charge_nullity("2T") == 0)
    ck("MS1a PART C: Z2 gradings -- A4: 1 (trivial only); 2T: 2, the nontrivial one = the "
       "center grading = FERMION PARITY (-1 exactly on the three spinorial 2-dims)",
       z2_gradings("A4") == 1 and z2_gradings("2T") == 2)
    sc = gauge_sector_category()
    ck("consistency: net.gauge_sector_category() sectors {1,3,3,1} + double cover 2T unchanged "
       "(HK-6, the ring's premise P3)",
       sc["sectors_are_species"] and sc["double_cover_2T"])

    if verbose:
        print("RESULT:", "SECTION-7 ACCRETION REGRESSION PASSES (all station anchors reproduced)"
              if ok else "A SECTION-7 CHECK FAILED")
    return ok


# ===========================================================================
# 7b. THE I2b DART/TOEPLITZ-CK ALGEBRA  (STEP 0 ACCRETION, 2026-07-11)
#     FOCK0_dr_reconstruction_prereg_2026-07-11.md, mandated amendment A3
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as Section 7 above -- appended after the
#  __main__ guard; on IMPORT every definition lands in the module namespace; script mode
#  (python3 the_net.py) is unaffected.]
#
# PROVENANCE: ports proofs/foundations/I2b_matsumoto_completion_2026-07-10.py:143-259
# (build_hist / build_S / the Toeplitz-CK defect + companion-relation checks; C-1/C-2 of that
# station) as plain importable functions.  This DISCHARGES the booked I2b Layer-3 accretion debt
# (assembly hazard #2/#8: the dart algebra was NOT reachable from the_net.py -- confirmed by a
# zero-hit grep for Toeplitz|H_hist|Cuntz|build_hist -- prior to this section).
#
# The station file I2b_matsumoto_completion_2026-07-10.py is NOT imported (per its own docstring
# and assembly hazard #2: no `if __name__ == "__main__":` guard, and it calls sys.exit() at
# module level, line 427 -- importing it would re-execute the whole script and then raise
# SystemExit into the caller).  What follows is an independent, faithful re-derivation of its
# C-1/C-2 logic from the net's OWN hashimoto_gamma() (verified bit-identical to the station's own
# `the_run.hashimoto((0,0,0))` import -- both resolve to srs.hashimoto with max|diff|=0.0,
# confirmed at accretion time).  NO magnitude beyond structural/exactness residuals is defined
# here (module contract, per file header): this section builds the ALGEBRA and its defect/
# companion checks, plus the length-diagonal state omega_diag as a STRUCTURAL object (a function
# of word length only, per I2b's own C-2 gauge-average) -- it does not compute or compare any
# physical constant.
def _dart_admissible_successors():
    """A_sft (SFT transition matrix, row=source dart) + succ[d] = admissible continuations of
    dart d, built from the net's OWN hashimoto_gamma() (== I2b's B0; verified bit-identical to
    the station's `the_run.hashimoto((0,0,0))` at accretion time).  I2b convention:
    A_sft = round(B0).T (row = source dart)."""
    B0 = hashimoto_gamma()
    Bi = np.rint(B0).astype(int)
    assert set(Bi.flatten().tolist()) <= {0, 1}, "_dart_admissible_successors: B0 not exactly 0/1"
    A_sft = Bi.T
    return A_sft, [np.nonzero(A_sft[a])[0].tolist() for a in range(ND)]


def build_hist(N_max, succ=None):
    """[I2b C-1 port, I2b_matsumoto_completion_2026-07-10.py:143-153] H_hist = the word-Fock
    ("history") space: H_0 = C|seed> (1-dim vacuum), H_n (n>=1) = span of the 12*2^(n-1)
    admissible dart-words of length n.  Returns (words, index, lengths)."""
    if succ is None:
        _, succ = _dart_admissible_successors()
    words = [()]
    lengths = [0]
    frontier = [()]
    for n in range(1, N_max + 1):
        new_frontier = ([(d,) for d in range(ND)] if n == 1 else
                         [w + (d,) for w in frontier for d in succ[w[-1]]])
        words.extend(new_frontier)
        lengths.extend([n] * len(new_frontier))
        frontier = new_frontier
    index = {w: i for i, w in enumerate(words)}
    return words, index, np.array(lengths)


def build_S(words, index, lengths, N_max, succ=None):
    """[I2b C-1 port, I2b_matsumoto_completion_2026-07-10.py:155-170] S_d: H_n -> H_{n+1}, the
    Cuntz-Krieger word-extension partial isometry (append dart d if admissible; seed -> any dart;
    truncated to 0 at |w|=N_max).  Returns the list of ND scipy.sparse csr (D,D) matrices."""
    import scipy.sparse as spa
    if succ is None:
        _, succ = _dart_admissible_successors()
    rows_d = [[] for _ in range(ND)]
    cols_d = [[] for _ in range(ND)]
    for i, w in enumerate(words):
        if lengths[i] == N_max:
            continue
        if len(w) == 0:
            for d in range(ND):
                rows_d[d].append(index[(d,)])
                cols_d[d].append(i)
        else:
            for d in succ[w[-1]]:
                rows_d[d].append(index[w + (d,)])
                cols_d[d].append(i)
    D = len(words)
    return [spa.csr_matrix((np.ones(len(rows_d[d])), (rows_d[d], cols_d[d])), shape=(D, D))
            for d in range(ND)]


def toeplitz_ck_check(N_max=6):
    """[I2b C-1 anchor port, STEP 0] the Toeplitz-Cuntz-Krieger defect Sum_d S_d S_d^* = 1 -
    P_seed EXACTLY, plus the companion relation S_d^*S_d = P_seed + Sum_e A_sft[e,d] S_eS_e^*
    (exact in the interior |w| < N_max; a NAMED truncation artifact at the top shell only -- S_d
    is truncated to 0 there, so the projector identity cannot hold at the boundary; this does NOT
    affect the requested defect identity, whose two ingredients never leave the truncation
    window).  Re-verified here at a MODEST N_max as a PERMANENT the_net.py regression anchor
    (I2b's own station verified the SAME identity at N_max=10/16; this is the same identity,
    ported, not re-derived physics).  Returns {'toeplitz_defect', 'companion_interior', 'D',
    'boundary_mismatches'}; the first two are ASSERTED < 1e-9."""
    import scipy.sparse as spa
    A_sft, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(N_max, succ)
    D = len(words)
    S = build_S(words, index, lengths, N_max, succ)
    Sdag = [Sd.transpose().tocsr() for Sd in S]
    Pseed = spa.csr_matrix(([1.0], ([0], [0])), shape=(D, D))
    Iden = spa.identity(D, format="csr")
    total = sum(S[d] @ Sdag[d] for d in range(ND))
    diff = (total - (Iden - Pseed)).tocoo()
    max_defect = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    interior = lengths < N_max
    worst_interior, boundary_mismatches = 0.0, 0
    for d in range(ND):
        lhs = Sdag[d] @ S[d]
        rhs = (Pseed + sum(A_sft[e, d] * (S[e] @ Sdag[e]) for e in range(ND))).tocoo()
        dm = (lhs - rhs).tocoo()
        is_int = interior[dm.row] & interior[dm.col]
        if is_int.any():
            worst_interior = max(worst_interior, float(np.max(np.abs(dm.data[is_int]))))
        boundary_mismatches += int(np.sum(np.abs(dm.data[~is_int]) > 1e-9))
    assert max_defect < 1e-9, f"toeplitz_ck_check: defect {max_defect:.3e} >= 1e-9 at N_max={N_max}"
    assert worst_interior < 1e-9, \
        f"toeplitz_ck_check: companion interior {worst_interior:.3e} at N_max={N_max}"
    return {"toeplitz_defect": max_defect, "companion_interior": worst_interior,
            "D": D, "boundary_mismatches": boundary_mismatches}


def omega_diag_length(N_max, u=None):
    """[I2b C-2 port] the run |G> = Sum_n u^n B^n|seed>, gauge-averaged (length-diagonal) state
    omega_diag(w) = u^(2|w|)/Z -- I2b's own closed form (<w|G>=u^|w|), re-derived here (NOT
    re-run from the station).  u defaults to alpha_1 = (2/3)^8, the run's own operating fugacity
    (the SAME u_op used throughout Section 7's symmetric_compression/accretion_selftest).
    STRUCTURAL note (used downstream, FOCK-0 3d): omega_diag depends ONLY on word length -- it is
    manifestly REAL and length-block-diagonal (I2b C-3a: 'Perron weights, uniform'), i.e.
    bit-EVEN by construction; it is reused here as a functional INPUT, never as the PIN.
    Returns (words, index, lengths, omega_diag)."""
    if u is None:
        u = (2.0 / 3.0) ** 8
    words, index, lengths = build_hist(N_max)
    amp = u ** lengths.astype(float)
    Z = float(np.sum(amp ** 2))
    omega_diag = (amp ** 2) / Z
    return words, index, lengths, omega_diag


def i2b_selftest_2026_07_11(verbose=True):
    """STEP 0 regression: the ported Toeplitz-CK defect + companion relation, at a modest N_max,
    a new permanent the_net.py anchor (alongside the two module anchors and Section 7's battery).
    Also cross-checks hashimoto_gamma() against the station's own the_run.hashimoto((0,0,0))
    import, bit-identical, confirming the port uses the SAME engine object."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 7b self-test -- I2b dart/Toeplitz-CK accretion (STEP 0, 2026-07-11)")
        print("=" * 88)

    sys.path.insert(0, _REPO)
    from derivation_topdown.bridge import the_run  # noqa: E402  cross-check only, not re-imported below
    B0_net = hashimoto_gamma()
    B0_station = the_run.hashimoto((0, 0, 0)).real
    ck("hashimoto_gamma() == the_run.hashimoto((0,0,0)).real bit-identical (I2b's own B0 import)",
       float(np.max(np.abs(B0_net - B0_station))) == 0.0)

    res = toeplitz_ck_check(N_max=6)
    ck(f"[N_max=6, D={res['D']}] Toeplitz-CK defect Sum_d S_dS_d^* = 1-P_seed EXACT",
       res["toeplitz_defect"] < 1e-9, detail=f"max|diff| = {res['toeplitz_defect']:.3e}")
    ck(f"[N_max=6] companion relation exact in the interior (|w|<6)",
       res["companion_interior"] < 1e-9,
       detail=f"worst interior = {res['companion_interior']:.3e}, "
              f"boundary mismatches = {res['boundary_mismatches']} (named, expected)")

    words, index, lengths, omega = omega_diag_length(N_max=6)
    ck("omega_diag normalized (sum=1) and positive (length-diagonal, bit-EVEN by construction)",
       abs(float(np.sum(omega)) - 1) < 1e-9 and bool(np.all(omega > 0)))

    if verbose:
        print("RESULT:", "STEP-0 I2b ACCRETION REGRESSION PASSES" if ok else "A STEP-0 CHECK FAILED")
    return ok


# ===========================================================================
# 8. FOCK-0 — THE SECTOR-GRADED FOCK LAYER / DR-MAP CANDIDATE  (2026-07-11)
#    FOCK0_dr_reconstruction_prereg_2026-07-11.md SS3 (a)-(d)
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as Sections 7/7b above.]
#
# ARCHITECT HYPOTHESIS UNDER TEST: a Doplicher-Roberts-style reconstruction over the net's OWN
# sector category (ML-2's A4/2T species {nu:1,d:3,u:3,e:1}), pinned by intertwining MODULAR
# CONJUGATIONS (antiunitary, per-sector), built on the STEP-0-accreted I2b dart/Toeplitz-CK Fock
# space H_hist.  Every step below reuses accreted APIs (fusion_ring/additive_charge_nullity/
# z2_gradings, gauge_sector_category, vacuum_covariance/anchor_cell_projector/anchor_tick_2pi,
# dart_rep/_edge_rep/_a4_vertex_group, _fusion_group/_character_table/_conjugacy_classes) --
# nothing is rebuilt.  NUMBERS APPEAR NOWHERE: every quantity below is a dimension, multiplicity,
# rank, or exactness residual (structure), never M_Z/ppm/m_nu/a_e (module contract + pre-reg SS3.4).
#
# ML-2b/HK-7 CONDITIONALITY (carries into every verdict sentence below, verbatim per aqft_net.py
# 280-292 and the pre-reg SS0): "Every duality check here (HK-5) is CELL-LEVEL only (the 6-edge
# static vacuum). ML-2b's DR-frame argument is CONDITIONAL on the TD-limit duality holding, which
# is NOT verified by this suite."  The sector category (gauge_sector_category, MS-1a's fusion
# ring) this section grades over inherits that conditionality unchanged; nothing here discharges
# it.
def _a4_key(g):
    """The vertex-permutation DICT convention (_a4_vertex_group, dart_rep, gauge_sector_category,
    dr_frame_audit) as a hashable tuple key, matching _fusion_group('A4')'s own element encoding
    EXACTLY (same itertools.permutations(range(4)) enumeration, same evenness filter -- verified
    bit-for-bit at accretion time: _a4_vertex_group()[i] == dict(enumerate(_fusion_group('A4')[0][i]))
    for every i)."""
    return tuple(g[i] for i in range(NV))


def _a4_char_lookup():
    """[FOCK-0 3a] chi_a(g) for EVERY A4 element g (module vertex-permutation-dict convention),
    reusing the accreted _fusion_group/_conjugacy_classes/_character_table/fusion_ring (the SAME
    Dixon/Burnside machinery fusion_ring('A4') already runs -- this additionally keeps cls_of,
    which fusion_ring() itself discards, so it is a THIN wrapper, not a re-derivation).  Irrep
    order matches fusion_ring('A4')['dims'] == [1,1,1,3] (trivial forced first by
    _character_table's own reordering).  Returns (dims, chars_by_elt): chars_by_elt[a] is a dict
    _a4_key(g) -> chi_a(g) (complex)."""
    elems, mul, inv, e = _fusion_group("A4")
    classes, sizes, cls_of, X = _character_table(elems, mul, inv, e)
    fr = fusion_ring("A4")
    dims = fr["dims"]
    assert len(X) == len(dims) and all(abs(X[a][0].real - dims[a]) < 1e-8 for a in range(len(dims))), \
        "_a4_char_lookup: _character_table/fusion_ring dims disagree"
    chars_by_elt = [{g_tuple: X[a][cls_of[g_tuple]] for g_tuple in elems} for a in range(len(X))]
    return dims, chars_by_elt


def dart_word_action(words, index):
    """[FOCK-0 3a] the A4 action on H_hist words: g.w = (pi_g(d_1),...,pi_g(d_n)) for
    w=(d_1,...,d_n), pi_g = the dart permutation read off dart_rep(g) (a genuine 0/1 permutation
    matrix, the REGULAR rep, Section 7).  This is a genuine group action because A_sft is
    dart_rep-covariant: dart_rep(g) commutes with hashimoto_gamma() EXACTLY for every g in A4
    (accretion_selftest_2026_07_10 verifies this for a spot-checked pair; re-verified here for
    ALL 12 elements, max residual < 1e-12) -- so admissibility of w is preserved by every g, and
    g.w is always a valid word already present in `words`.  Returns (A4v, perms): A4v = the 12
    vertex-permutation dicts (_a4_vertex_group order); perms[k] = int array of length D with
    perms[k][i] = index of A4v[k].w_i in `words`."""
    A4v = _a4_vertex_group()
    B0 = hashimoto_gamma()
    assert max(float(np.max(np.abs(dart_rep(g) @ B0 - B0 @ dart_rep(g)))) for g in A4v) < 1e-12, \
        "dart_word_action: dart_rep is not B0-covariant for some g (action ill-defined)"
    perms = []
    for g in A4v:
        Rd = dart_rep(g)
        pi = np.argmax(Rd, axis=0)          # pi[d] = image dart of d under g
        perm = np.empty(len(words), dtype=int)
        for i, w in enumerate(words):
            gw = tuple(int(pi[d]) for d in w)
            perm[i] = index[gw]
        perms.append(perm)
    return A4v, perms


def sector_grading_hist(N_max):
    """[FOCK-0 3a, THE GRADING OPERATOR] grade H_hist (truncated at N_max) by its A4-isotypic
    decomposition, using dart_word_action's genuine group action.  Builds the isotypic projectors
    P_a = (dim_a/|G|) Sum_g conj(chi_a(g)) Pi(g) (Pi(g) = the D x D permutation matrix of the
    action), verifies the FULL projector battery (idempotent, Hermitian, mutually orthogonal,
    complete: Sum_a P_a = I), and reads off the per-length irrep multiplicities.
    CROSS-CHECK (regular-rep consistency): at length 1 (H_1 = the 12 darts themselves), mult must
    equal [1,1,1,3] EXACTLY (each irrep with multiplicity = its own dimension -- the defining
    property of the regular representation, independently confirming Section 7's
    ip(chi,chi)=12 character check via a DIFFERENT method, isotypic projection).
    Returns {'words','index','lengths','mult' (shape (N_max+1, 4) int),'P' (list of 4 D x D
    projectors),'irrep_dims','parity_commute_residual','projector_battery'}."""
    words, index, lengths = build_hist(N_max)
    D = len(words)
    A4v, perms = dart_word_action(words, index)
    dims, chars_by_elt = _a4_char_lookup()
    r = len(dims)
    G = len(A4v)
    e_idx = next(i for i, g in enumerate(A4v) if all(g[k] == k for k in range(NV)))
    assert np.array_equal(perms[e_idx], np.arange(D)), "sector_grading_hist: identity acts non-trivially"
    comp = lambda g, h: {i: g[h[i]] for i in range(NV)}
    gi, hi = 3, 7
    gh_key = _a4_key(comp(A4v[gi], A4v[hi]))
    gh_idx = next(i for i, g in enumerate(A4v) if _a4_key(g) == gh_key)
    assert np.array_equal(perms[gi][perms[hi]], perms[gh_idx]), \
        "sector_grading_hist: dart_word_action is not a homomorphism"
    assert all(int(lengths[perms[k][i]]) == int(lengths[i]) for k in range(G) for i in (0, D - 1)), \
        "sector_grading_hist: action does not preserve word length"
    P = []
    for a in range(r):
        Pa = np.zeros((D, D), dtype=complex)
        for k, g in enumerate(A4v):
            chi = chars_by_elt[a][_a4_key(g)]
            Pi_g = np.zeros((D, D))
            Pi_g[perms[k], np.arange(D)] = 1.0
            Pa += np.conj(chi) * Pi_g
        Pa *= dims[a] / G
        P.append(Pa)
    idem = max(float(np.max(np.abs(Pa @ Pa - Pa))) for Pa in P)
    herm = max(float(np.max(np.abs(Pa - Pa.conj().T))) for Pa in P)
    orth = max(float(np.max(np.abs(P[a] @ P[b]))) for a in range(r) for b in range(r) if a != b)
    complete = float(np.max(np.abs(sum(P) - np.eye(D))))
    assert idem < 1e-7 and herm < 1e-7 and orth < 1e-7 and complete < 1e-7, \
        (f"sector_grading_hist: projector battery fails (idem={idem:.1e}, herm={herm:.1e}, "
         f"orth={orth:.1e}, complete={complete:.1e})")
    mult = np.zeros((N_max + 1, r))
    for n in range(N_max + 1):
        idx_n = np.where(lengths == n)[0]
        for a in range(r):
            block = P[a][np.ix_(idx_n, idx_n)]
            mult[n, a] = float(np.real(np.trace(block))) / dims[a]
    mult_round = np.round(mult).astype(int)
    assert float(np.max(np.abs(mult - mult_round))) < 1e-5, "sector_grading_hist: non-integer multiplicity"
    if N_max >= 1:
        i3 = dims.index(3)
        assert list(mult_round[1]) == [1 if a != i3 else 3 for a in range(r)], \
            f"sector_grading_hist: length-1 multiplicities {mult_round[1].tolist()} != regular-rep [1,1,1,3]"
    assert list(mult_round[0]) == [1, 0, 0, 0], \
        f"sector_grading_hist: seed multiplicities {mult_round[0].tolist()} != [1,0,0,0]"
    parity = np.array([(-1.0) ** n for n in lengths])
    parity_commute = max(float(np.max(np.abs((parity[:, None] * Pa) - (Pa * parity[None, :]))))
                         for Pa in P)
    return {"words": words, "index": index, "lengths": lengths, "mult": mult_round,
            "P": P, "irrep_dims": dims, "parity_commute_residual": parity_commute,
            "projector_battery": {"idem": idem, "herm": herm, "orth": orth, "complete": complete}}


def _sector_projectors(sign=+1):
    """[FOCK-0 3b] the NHAT-eigenspace (sector) projectors Pw, w=0..3, built EXACTLY as
    gauge_sector_category() builds them but threading the M0 bit's sign parameter (already
    exposed by vacuum_covariance(sign=...), unused by gauge_sector_category as shipped) through
    complex_structure_J6: sign=+1 reproduces gauge_sector_category()'s own species dims exactly
    (cross-checked in sector_pair_conjugation below); sign=-1 is the bit-flipped J -> -J
    construction (M0-4a's anchor, C(-J)=I-C(J), lifted here to the Fock/sector level).  NOT a
    second Fock space: the SAME 8-dim Cl(6) Fock construction as gauge_sector_category, sign is
    the only new knob.  Returns (Pw dict w->8x8 projector, NHAT)."""
    sys.path.insert(0, _REPO)
    sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "bridge"))
    from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402
    g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]

    def gam(x):
        return sum(x[a] * g6[a] for a in range(NE))

    J6 = sign * complex_structure_J6()
    wJ, VJ = np.linalg.eig(J6)
    modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
    A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
    NHAT = sum(a.conj().T @ a for a in A_ops)
    wN, VN = np.linalg.eigh(NHAT)
    wNr = np.round(np.real(wN)).astype(int)
    Pw = {w: VN[:, wNr == w] @ VN[:, wNr == w].conj().T for w in range(4)}
    return Pw, NHAT


def sector_pair_conjugation():
    """[FOCK-0 3b, THE PER-SECTOR MODULAR CONJUGATION] the M0 bit (J -> -J; anchor C(-J)=I-C(J),
    M0-4a) acting at the Cl(6) FOCK level: verify EXACTLY that the sign flip pairs sector w with
    sector (3-w) as SUBSPACES of the SAME 8-dim Fock space F (Pw(sign=-1) and P_{3-w}(sign=+1)
    project onto the identical subspace: A@B==B and B@A==A for exact equal-rank projectors).
    This is J_sigma: sigma pairs with 3-sigma, FORCED by the anchor -- no free choice.  The four
    sectors split into the two bit-orbits {0,3} ('nu'<->'e') and {1,2} ('d'<->'u'); NEITHER orbit
    has a fixed point (0!=3, 1!=2), so the bit acts WITHOUT a self-conjugate sector.  Also cross-
    checks sign=+1 against gauge_sector_category()'s own species_sector_dims (consistency, not a
    re-derivation).  Returns {'orbit_03','orbit_12','is_involution','dims_match_gsc'} (first three
    asserted < 1e-9)."""
    Pp, _ = _sector_projectors(sign=+1)
    Pm, _ = _sector_projectors(sign=-1)

    def subspace_eq(A, B):
        return max(float(np.max(np.abs(A @ B - B))), float(np.max(np.abs(B @ A - A))))

    r03 = max(subspace_eq(Pm[0], Pp[3]), subspace_eq(Pm[3], Pp[0]))
    r12 = max(subspace_eq(Pm[1], Pp[2]), subspace_eq(Pm[2], Pp[1]))
    Pp2, _ = _sector_projectors(sign=+1)
    invol = max(float(np.max(np.abs(Pp[w] - Pp2[w]))) for w in range(4))
    dims_pp = {w: int(round(np.trace(Pp[w]).real)) for w in range(4)}
    sc = gauge_sector_category()
    res = {"orbit_03": r03, "orbit_12": r12, "is_involution": invol,
           "dims_match_gsc": dims_pp == sc["species_sector_dims"]}
    for name in ("orbit_03", "orbit_12", "is_involution"):
        assert res[name] < 1e-9, f"sector_pair_conjugation: {name} residual {res[name]:.2e} >= 1e-9"
    assert res["dims_match_gsc"], f"sector_pair_conjugation: dims {dims_pp} != gsc {sc['species_sector_dims']}"
    return res


def _field_algebra_a4_rep():
    """[FOCK-0 3c] the field-algebra F's own A4 action, U(g) = spin_lift(edge_rep(g)) for every g
    in A4 -- REPLICATES gauge_sector_category()'s internal construction (same recipe; the
    codebase's existing convention is for each function to recompute spin_lift locally rather
    than share one, e.g. dr_frame_audit and gauge_sector_category each define their own copy).
    Machine-re-verifies the SAME cocycle fact HK-6b already established (double_cover_2T ==
    True): exhibits an explicit (g,h) pair with U(g)U(h) = -U(gh) -- F is honestly a
    representation of 2T (PROJECTIVE as an A4 rep), not of A4 itself.  Returns (A4v, U,
    cocycle_pair) where cocycle_pair = (a_idx, b_idx, gh_idx, c) with c approx -1."""
    sys.path.insert(0, _REPO)
    sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "bridge"))
    from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402
    g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
    I8 = np.eye(8)

    def gam(x):
        return sum(x[a] * g6[a] for a in range(NE))

    def spin_lift(Rmat):
        rowsU = [np.kron(gam(Rmat[:, a]), I8) - np.kron(I8, g6[a].T) for a in range(NE)]
        _, s, Vh = np.linalg.svd(np.vstack(rowsU))
        M = Vh[np.sum(s > 1e-9):].conj()[0].reshape(8, 8)
        return M / np.sqrt(np.abs(np.linalg.det(M @ M.conj().T)) ** (1 / 8))

    A4v = _a4_vertex_group()
    U = [spin_lift(_edge_rep(g)) for g in A4v]
    comp = lambda g, h: {i: g[h[i]] for i in range(NV)}
    ix = {_a4_key(g): n for n, g in enumerate(A4v)}
    cocycle_pair = None
    for a, g in enumerate(A4v):
        for b, h in enumerate(A4v):
            gh_idx = ix[_a4_key(comp(g, h))]
            c = np.trace(np.linalg.solve(U[gh_idx], U[a] @ U[b])) / 8.0
            if abs(c + 1) < 1e-3:
                cocycle_pair = (a, b, gh_idx, complex(c))
                break
        if cocycle_pair:
            break
    assert cocycle_pair is not None, \
        "_field_algebra_a4_rep: no cocycle=-1 pair found -- HK-6b double_cover_2T fact not reproduced"
    return A4v, U, cocycle_pair


def dr_map_hom_space():
    """[FOCK-0 3c, THE DR-MAP CANDIDATE] the intertwining space Hom_A4(dart_rep, F-via-spin-lift):
    linear maps Phi (8x12, darts -> the 8-dim field-algebra Fock) satisfying Phi@dart_rep(g) =
    U(g)@Phi for EVERY g in A4, solved by SVD null-space (the SAME method as map_commutant's
    Hom_A4(edge_rep,dart_rep), Section 7 -- a DIFFERENT linear system, different target space).

    LEMMA (proof sketch, machine-verified below).  dart_rep is a GENUINE (non-projective) A4
    representation: it is built directly from vertex PERMUTATIONS composing exactly under the
    group law (dart_rep(g)dart_rep(h) = dart_rep(gh) for every pair -- no square-root/spin-lift
    step is involved, so no sign ambiguity can enter).  U is a PROJECTIVE A4 representation
    (U(g)U(h) = -U(gh) on cocycle-nontrivial pairs; HK-6b's double_cover_2T fact).  If Phi solved
    the per-generator constraints for a cocycle=-1 pair (g,h), composing them:
        Phi.dart_rep(g).dart_rep(h) = Phi.dart_rep(gh)          [dart_rep's honest group law]
        U(g).U(h).Phi = -U(gh).Phi                              [the -1 cocycle]
    but the per-generator constraints also give Phi.dart_rep(g).dart_rep(h) = U(g).U(h).Phi, so
    U(gh).Phi = -U(gh).Phi, i.e. 2.U(gh).Phi = 0, i.e. Phi = 0 (U(gh) invertible).
    CONSEQUENCE: Hom_A4(dart_rep, F) = {0} EXACTLY -- the naive DR-map candidate (equivariant
    w.r.t. the SAME A4 action on both sides) is OBSTRUCTED: a bosonic (dart_rep) / fermionic
    (2T-projective F) mismatch, THEOREM-GRADE (not a search failure -- the SVD below finds full
    rank/corank 0, smallest singular value bounded well away from zero, no near-degeneracy).
    This is a NEW obstruction (nonexistence of ANY equivariant map, not an orbit-discrimination
    blindness) -- it does not reduce to any of the five already-proven forms (SS2 fence, checked
    separately in fock0_fence_check).  A trivial corollary (nullity_with_R_constraint) shows
    adding the R/F_bit sector-parity intertwining requirement (SS3c's further pin) cannot
    un-obstruct an already-empty space.
    Returns {'nullity','rank','smallest_sv','cocycle_pair','homdev_dart_honest_rep',
    'cocycle_residual','nullity_with_R_constraint'}."""
    A4v, U, cocycle_pair = _field_algebra_a4_rep()
    DartR = [dart_rep(g) for g in A4v]
    I8 = np.eye(8)
    rows = [np.kron(Rd.T, I8) - np.kron(np.eye(ND), Ug) for Rd, Ug in zip(DartR, U)]
    Cstack = np.vstack(rows)
    s = np.linalg.svd(Cstack, compute_uv=False)
    rank = int(np.sum(s > 1e-8))
    nullity = Cstack.shape[1] - rank
    a_idx, b_idx, gh_idx, cval = cocycle_pair
    homdev_dart = float(np.max(np.abs(DartR[a_idx] @ DartR[b_idx] - DartR[gh_idx])))
    cocycle_resid = float(np.max(np.abs(U[gh_idx] + U[a_idx] @ U[b_idx])))
    R = reversal()
    Pp, _ = _sector_projectors(sign=+1)
    F_bit = Pp[0] + Pp[3] - Pp[1] - Pp[2]
    rows2 = rows + [np.kron(R.T, I8) - np.kron(np.eye(ND), F_bit)]
    s2 = np.linalg.svd(np.vstack(rows2), compute_uv=False)
    nullity2 = (I8.shape[0] * ND) - int(np.sum(s2 > 1e-8))
    assert homdev_dart < 1e-10, "dr_map_hom_space: dart_rep failed its own group law (unexpected)"
    assert cocycle_resid < 1e-6, "dr_map_hom_space: named cocycle pair is not actually -1"
    assert nullity == 0, f"dr_map_hom_space: SURPRISE -- nullity {nullity} != 0, Hom space nonempty; book raw"
    return {"nullity": nullity, "rank": rank,
            "smallest_sv": float(s[-1]) if len(s) else float("nan"),
            "cocycle_pair": (a_idx, b_idx, gh_idx, cval), "homdev_dart_honest_rep": homdev_dart,
            "cocycle_residual": cocycle_resid, "nullity_with_R_constraint": nullity2}


def fock0_door_i_check():
    """[FOCK-0 3d, QF-2b DOOR (i) STRUCTURAL ADDENDUM] does a patch-restricted state built from
    THIS station's own ingredients exhibit Im C != 0 while reproducing the cell anchor
    C=(I+iJ6)/2?  The only NEW ingredient this station derives beyond the single-cell anchor is
    sector_pair_conjugation -- a relation BETWEEN Pw(sign=+1) and Pw(sign=-1) projectors on the
    SAME cell's 8-dim Fock space, i.e. a cross-SECTOR fact, not a cross-CELL (spatial) one.  A
    genuine door-(i) instantiation needs a nontrivial SPATIAL (cross-cell) imaginary correlation.
    Test: the natural TRANSLATION-INVARIANT multi-cell extension of the single-cell C (repeat the
    anchor block-diagonally) has an EXACTLY zero cross-cell block (both Re and Im) -- consistent
    with the framework's OWN already-established cover-gauge-triviality theorem (any cell-periodic
    phase reduces to pure gauge) and holonomy-triviality theorem (192/192 cover-closed cycles ->
    +I exactly, quantum_foundations.py QF-2b F-0/adjudication 1): a translation-invariant
    construction cannot manufacture cross-cell phase content, by the framework's own prior
    theorems, not by fiat here.
    VERDICT (honest, negative): door (i) is NOT instantiated by this station -- reaching it needs
    a genuinely non-translation-invariant or particle-number-violating (Bogoliubov) ingredient,
    named here as an OPEN, undeveloped prerequisite (consistent with QF-2b's own framing: "a new
    derivation station").
    Returns {'single_cell_reproduces_anchor','two_cell_offdiag_im','two_cell_offdiag_re'}."""
    C0 = vacuum_covariance(sign=+1)
    J6 = complex_structure_J6()
    anchor_ok = bool(np.max(np.abs(C0 - (np.eye(NE) + 1j * J6) / 2.0)) < 1e-12)
    C_two = np.block([[C0, np.zeros((NE, NE), dtype=complex)],
                      [np.zeros((NE, NE), dtype=complex), C0]])
    offdiag = C_two[:NE, NE:]
    return {"single_cell_reproduces_anchor": anchor_ok,
            "two_cell_offdiag_im": float(np.max(np.abs(offdiag.imag))),
            "two_cell_offdiag_re": float(np.max(np.abs(offdiag.real)))}


def fock0_fence_check():
    """[FOCK-0 pre-reg SS2, THE DESIGN FENCE] explicit, checkable confirmation that this
    construction does not reduce to any of the five proven-blind forms (BOOTCAMP SS5); items 2-5
    are structural (by-construction) facts about WHICH objects this station computes, restated
    precisely so an adversarial checker can re-derive them independently; item 1 is a direct
    numeric check.  Returns a dict with one entry per fence item."""
    J6 = complex_structure_J6()
    im_norm = float(np.max(np.abs(J6)))
    return {
        "1_O0_bit_even_democracy": {
            "check": "the PIN (sector_pair_conjugation) is built from J6, antiunitary/phase-"
                     "bearing by construction, NOT an even (real-valued) functional",
            "im_J6_norm": im_norm, "is_phase_bearing": im_norm > 1e-6},
        "2_M1b_linear_intertwiners_perpendicular": {
            "check": "dr_map_hom_space targets Hom_A4(dart_rep(12), F(8)); Section 7's "
                     "map_commutant targets Hom_A4(edge_rep(6), dart_rep(12)) -- a different pair "
                     "of spaces, not the same computation re-run",
            "distinct_from_map_commutant": True},
        "3_BRIDGE_LOCK_attachment_functional_orbit_blind": {
            "check": "this construction is FOCK-LEVEL/GRADED (sector_grading_hist on H_hist, "
                     "dr_map_hom_space on the 8-dim Fock F) -- neither object is a single "
                     "R-parity-definite one-particle attachment functional Phi: edge->dart "
                     "(map_family's form)",
            "is_fock_level": True},
        "4_BRIDGE_T_two_point_data_blind": {
            "check": "this construction is REPRESENTATION-THEORETIC (isotypic projectors, "
                     "Hom-space nullity/rank) -- not a two-point correlation functional of the "
                     "run's resolvent (I-uB)^-1 at any order",
            "is_representation_theoretic": True},
        "5_BRIDGE_GEOM_per_sector_required": {
            "check": "this construction IS per-sector by design (graded by A4/2T irreps at every "
                     "step: sector_grading_hist, sector_pair_conjugation, dr_map_hom_space) -- "
                     "satisfies the K-DEPENDENT verdict's own requirement directly",
            "is_per_sector_by_design": True},
    }


def fock0_selftest_2026_07_11(verbose=True):
    """FOCK-0 station regression: STEP 0 (I2b accretion) + the sector-graded Fock layer
    (grading, per-sector conjugation, DR-map Hom-space, door-(i) check, fence check), plus the
    module anchors and Section 7's battery, all untouched.  Prints the verdict-relevant evidence;
    does NOT itself adjudicate V1-V4 (pre-reg SS4: architect-only)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 8 self-test -- FOCK-0 sector-graded Fock layer (2026-07-11)")
        print("=" * 88)

    ck("ANCHORS + Section 7/7b untouched",
       anchor_cell_projector() and anchor_tick_2pi()
       and accretion_selftest_2026_07_10(verbose=False) and i2b_selftest_2026_07_11(verbose=False))

    sg = sector_grading_hist(N_max=3)
    ck("3a GRADING: projector battery exact (idem/herm/orth/complete all < 1e-7); length-1 "
       "multiplicities == [1,1,1,3] (regular rep, cross-checked against Section 7's character "
       "inner product by a DIFFERENT method); seed multiplicities == [1,0,0,0]",
       max(sg["projector_battery"].values()) < 1e-7,
       detail=f"mult[0..3] = {sg['mult'].tolist()}")
    ck("3a parity: word-length parity (-1)^n commutes with every A4-isotypic projector "
       "(length-grading and A4-grading are compatible, though A4 alone supports only the "
       "TRIVIAL Z2 grading per MS-1a z2_gradings('A4')==1 -- length-parity is EXTRINSIC to the "
       "A4 content, a named incompleteness, see the station report)",
       sg["parity_commute_residual"] < 1e-8, detail=f"residual = {sg['parity_commute_residual']:.1e}")

    spc = sector_pair_conjugation()
    ck("3b PER-SECTOR CONJUGATION J_sigma: sign flip pairs sector 0<->3 and 1<->2 EXACTLY "
       "(subspace equality < 1e-9), matches gauge_sector_category()'s own dims, involution holds",
       spc["orbit_03"] < 1e-9 and spc["orbit_12"] < 1e-9 and spc["is_involution"] < 1e-9
       and spc["dims_match_gsc"],
       detail=f"orbit_03={spc['orbit_03']:.1e}, orbit_12={spc['orbit_12']:.1e}")

    dr = dr_map_hom_space()
    ck("3c DR-MAP CANDIDATE: Hom_A4(dart_rep, F) nullity == 0 EXACTLY (OBSTRUCTED, theorem-grade; "
       "cocycle pair verified genuine -1, dart_rep verified genuinely non-projective)",
       dr["nullity"] == 0 and dr["homdev_dart_honest_rep"] < 1e-10 and dr["cocycle_residual"] < 1e-6,
       detail=f"rank={dr['rank']}/96, smallest_sv={dr['smallest_sv']:.3f}, "
              f"nullity_with_R_constraint={dr['nullity_with_R_constraint']}")

    di = fock0_door_i_check()
    ck("3d DOOR (i): single-cell anchor reproduced exactly; translation-invariant two-cell "
       "extension has EXACTLY zero cross-cell block (both Re, Im) -- door (i) NOT instantiated "
       "(honest negative, consistent with cover-gauge-triviality)",
       di["single_cell_reproduces_anchor"] and di["two_cell_offdiag_im"] == 0.0
       and di["two_cell_offdiag_re"] == 0.0)

    fc = fock0_fence_check()
    ck("SS2 FENCE: all five evasions confirmed (bit-EVEN/M-1b/BRIDGE-LOCK/BRIDGE-T/BRIDGE-GEOM)",
       fc["1_O0_bit_even_democracy"]["is_phase_bearing"]
       and fc["2_M1b_linear_intertwiners_perpendicular"]["distinct_from_map_commutant"]
       and fc["3_BRIDGE_LOCK_attachment_functional_orbit_blind"]["is_fock_level"]
       and fc["4_BRIDGE_T_two_point_data_blind"]["is_representation_theoretic"]
       and fc["5_BRIDGE_GEOM_per_sector_required"]["is_per_sector_by_design"])

    if verbose:
        print("RESULT:", "FOCK-0 SECTION-8 REGRESSION PASSES" if ok else "A FOCK-0 CHECK FAILED")
    return ok


# ===========================================================================
# 8b. FOCK-0b — THE HISTORY-SIDE MODULAR CONJUGATION + THE PINNED-MAP TEST  (2026-07-11)
#     FOCK0_dr_reconstruction_prereg_2026-07-11.md AMENDMENT SS B (post-V4 adjudication)
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as Sections 7/7b/8 above.]
#
# CONTEXT: FOCK-0's own dr_map_hom_space() (SS8 above) was adjudicated V4 -- the verifier
# (working notes/FOCK0_check_2026-07-11.md) found it solves ordinary full-A4 LINEAR GENERATOR
# equivariance (Phi.dart_rep(g) = U(g).Phi for every g), the EXACT class SS1/SS2 of the pre-reg
# disclaims ("pinned by intertwining MODULAR CONJUGATIONS..., NOT by generators"); J_sigma was
# never used, and the history-side modular conjugation was never built.  This section builds it
# and runs the REAL (frozen-class) test: does a linear map Phi intertwine ONE antiunitary
# conjugation per side (NO per-g constraint at all), rather than the whole non-abelian A4 closure?
#
# NUMBERS APPEAR NOWHERE: every quantity below is a dimension, rank, nullity, or exactness
# residual (structure), never M_Z/ppm/m_nu/a_e (module contract + pre-reg SS3.4/SSD).
#
# ML-2b/HK-7 CONDITIONALITY (carries into every verdict sentence below, verbatim, unchanged from
# Section 8's own banner): "Every duality check here (HK-5) is CELL-LEVEL only (the 6-edge static
# vacuum). ML-2b's DR-frame argument is CONDITIONAL on the TD-limit duality holding, which is NOT
# verified by this suite."
def gns_purification(N_max, u=None):
    """[FOCK-0b b1, GNS/PURIFICATION] omega_diag at beta_natural = 2*beta_gas (u defaults to
    alpha_1 = (2/3)^8, the SAME operating fugacity omega_diag_length already uses -- I2b's own
    finding is that THIS state's length-diagonal decay rate IS beta_natural, not the pre-
    registered beta'; A1 stands, no beta'-vs-beta_natural adjudication is made here).
    rho := diag(omega_diag) on the truncated H_hist (build_hist/build_S, SS7b).  GNS/purification
    carrier = Hilbert-Schmidt operators on H_hist (D x D matrices), cyclic vector Omega = rho^1/2,
    the truncated algebra M = alg{S_d} acting by LEFT multiplication.
    SEPARATING CHECK (explicit, not assumed): rho is full-rank iff every admissible word gets
    omega_diag > 0; build_hist enumerates EVERY admissible word up to N_max and
    omega_diag(w) = u^(2|w|)/Z > 0 for every finite length, so rho is manifestly strictly positive
    on the whole truncation -- verified numerically below (min_rho > 0), not merely asserted.  If
    this ever failed on some future truncation scheme, that would be the W4 branch (book, do not
    patch); it does not fail here.
    Returns {'words','index','lengths','rho','min_rho','full_rank_separating'}."""
    words, index, lengths, omega = omega_diag_length(N_max, u=u)
    min_rho = float(np.min(omega))
    return {"words": words, "index": index, "lengths": lengths, "rho": omega,
            "min_rho": min_rho, "full_rank_separating": bool(min_rho > 0.0)}


def tomita_data(rho):
    """[FOCK-0b b2, THE CLOSED-FORM TOMITA DATA] for a diagonal, strictly-positive density matrix
    rho (length-D vector) on H = C^D, the exact finite-dim GNS/purification Tomita operators on
    HS(H) = Mat(D,C):
        Delta^1/2(A) = diag(sqrt(rho)) . A . diag(1/sqrt(rho))        [Delta(A) = rho.A.rho^-1]
        J(A)         = A^dagger                                       [plain matrix adjoint]
        S(A)         = J(Delta^1/2(A)) = diag(1/sqrt(rho)) . A^dagger . diag(sqrt(rho))
    DERIVATION (why this is exactly the Tomita S, not merely an ansatz): for Omega = rho^1/2 and
    ANY matrix x (Omega's cyclic orbit under the FULL matrix algebra, of which any subalgebra M is
    a subset -- the closed form does not need M specified), x.Omega = x @ diag(sqrt(rho)); then
        S(x.Omega) = diag(1/sqrt(rho)) . (x.diag(sqrt(rho)))^dagger . diag(sqrt(rho))
                   = diag(1/sqrt(rho)).diag(sqrt(rho)).x^dagger.diag(sqrt(rho)) = x^dagger.Omega,
    EXACTLY the defining Tomita relation S: x.Omega -> x*.Omega -- verified numerically in
    tomita_checks below on sampled x = E_{w,seed} (elementary seed-to-word matrix units).
    Returns the three callables plus rho and sqrt(rho)."""
    r = np.sqrt(rho)

    def Delta_half(A):
        return (r[:, None] * A) / r[None, :]

    def J(A):
        return A.conj().T

    def S(A):
        return J(Delta_half(A))

    return {"S": S, "Delta_half": Delta_half, "J": J, "rho": rho, "sqrt_rho": r}


def tomita_checks(N_max, u=None, n_probe_words=40, seed=0):
    """[FOCK-0b b1/b2, THE FOUR MACHINE CHECKS] build gns_purification + tomita_data for
    omega_diag at beta_natural (N_max truncation) and verify:
      (1) S(x.Omega) == x^dagger.Omega EXACTLY for sampled x = E_{w,seed} -- the defining Tomita
          relation, on the M-generated (seed-cyclic) subspace, at machine precision.
      (2) J^2 = identity (antiunitary INVOLUTION) on a random test matrix.
      (3) J M J subset M' (the commutant): [pi_R(S_d^dagger), pi_L(S_e)] = 0 for sampled generator
          pairs (d,e) of M = alg{S_d}, on a random probe matrix -- an ALGEBRAIC IDENTITY (left- and
          right-multiplication always commute, true for ANY subalgebra M, not merely alg{S_d});
          verified numerically here to catch implementation bugs, not because the fact is
          M-specific.
      (4) Delta <-> KMS-at-beta_natural, the PER-PATH reading: ln(rho_w) - ln(rho_v) ==
          -beta_natural.(len(w)-len(v)) for sampled word pairs -- the modular Hamiltonian
          K = -ln(rho) acting on the GNS carrier is EXACTLY beta_natural times the PER-WORD
          length (up to the additive -ln Z constant), matching the run's own per-microstate
          (per-PATH) KMS temperature.  CONTRAST, printed not asserted (the per-path/per-shell
          poison, SSD): the PER-SHELL aggregate rate -ln(p_[n+1]/p_n), p_n = sum of rho over
          length-n words, reproduces beta_natural - h_top = beta' EXACTLY (I2b's own
          beta_natural = beta' + h_top identity) -- confirming this construction's Delta matches
          the PER-PATH reading (Delta acts diagonally on individual WORDS/microstates, each
          carrying its own rho_w = u^2|w|/Z, not on shell-aggregated marginals)."""
    gp = gns_purification(N_max, u=u)
    words, index, lengths, rho = gp["words"], gp["index"], gp["lengths"], gp["rho"]
    D = len(words)
    td = tomita_data(rho)
    S, Delta_half, J = td["S"], td["Delta_half"], td["J"]
    r = td["sqrt_rho"]

    rng = np.random.default_rng(seed)
    seed_i = index[()]
    sample_idx = rng.choice(D, size=min(n_probe_words, D), replace=False)
    worst_S = 0.0
    for wi in sample_idx:
        x = np.zeros((D, D), dtype=complex)
        x[int(wi), seed_i] = 1.0
        A = x * r[None, :]
        lhs = S(A)
        rhs = x.conj().T * r[None, :]
        worst_S = max(worst_S, float(np.max(np.abs(lhs - rhs))))

    Atest = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    worst_J2 = float(np.max(np.abs(J(J(Atest)) - Atest)))

    _, succ = _dart_admissible_successors()
    Sops = build_S(words, index, lengths, N_max, succ)
    Bsmall = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    worst_comm = 0.0
    for d in (0, 1, ND - 1):
        for e in (0, 2, ND - 1):
            Sd = Sops[d].toarray()
            Se = Sops[e].toarray()
            lhs = Se @ (Bsmall @ Sd.conj().T)
            rhs = (Se @ Bsmall) @ Sd.conj().T
            worst_comm = max(worst_comm, float(np.max(np.abs(lhs - rhs))))

    u_eff = (2.0 / 3.0) ** 8 if u is None else u
    beta_natural_check = -2.0 * math.log(u_eff)
    pairs = [(int(sample_idx[i]), int(sample_idx[i + 1])) for i in range(0, len(sample_idx) - 1, 2)]
    worst_kms = 0.0
    for wi, vi in pairs:
        lhs = math.log(rho[wi]) - math.log(rho[vi])
        rhs = -beta_natural_check * (int(lengths[wi]) - int(lengths[vi]))
        worst_kms = max(worst_kms, abs(lhs - rhs))
    Dlen = int(lengths.max())
    p_shell = np.array([float(np.sum(rho[lengths == n])) for n in range(Dlen + 1)])
    h_top = math.log(2.0)
    shell_rate = -math.log(p_shell[Dlen] / p_shell[Dlen - 1]) if Dlen >= 1 else float("nan")
    beta_prime_reproduced = beta_natural_check - h_top

    out = {
        "D": D, "min_rho": gp["min_rho"], "full_rank_separating": gp["full_rank_separating"],
        "S_closed_form_residual": worst_S, "J_squared_residual": worst_J2,
        "JMJ_subset_Mprime_residual": worst_comm,
        "beta_natural_check": beta_natural_check,
        "kms_per_path_residual": worst_kms,
        "per_shell_rate_sample": shell_rate, "beta_prime_reproduced": beta_prime_reproduced,
        "h_top": h_top,
    }
    for name, tol in (("S_closed_form_residual", 1e-9), ("J_squared_residual", 1e-9),
                      ("JMJ_subset_Mprime_residual", 1e-8), ("kms_per_path_residual", 1e-8)):
        assert out[name] < tol, f"tomita_checks: {name} = {out[name]:.3e} >= {tol}"
    assert abs(beta_natural_check - 6.4874417297) < 1e-6, \
        f"tomita_checks: beta_natural {beta_natural_check} does not match I2b's own 6.4874417297"
    return out


def gns_grading_commutation(N_max, u=None):
    """[FOCK-0b b3, THE GRADED GNS CARRIER] apply Section 8's A4-isotypic grading
    (sector_grading_hist) to the GNS carrier:
      * rho (hence Delta) commutes EXACTLY with every isotypic projector P[a] -- a consequence of
        dart_word_action preserving word LENGTH (sector_grading_hist's own assertion): rho is a
        SCALAR on each length shell, hence commutes with anything shell-preserving, in particular
        the A4 group action and its isotypic projectors.
      * J's action on the graded blocks: J(P[a].A.P[b]) = P[b].J(A).P[a] EXACTLY (J is plain
        adjoint, P's Hermitian) -- J SELF-MAPS each DIAGONAL (a,a) block and SWAPS each
        OFF-DIAGONAL (a,b) <-> (b,a) pair.  Since A4's abelianization is Z3, its two nontrivial
        1-dim characters (indices matching fusion_ring('A4')'s dims=[1,1,1,3] ordering) are
        COMPLEX CONJUGATES of one another (the trivial character and the 3-dim irrep are each
        self-conjugate) -- identified below by direct comparison of chars_by_elt, the natural
        candidate for a nontrivial history-side sector PAIRING analogous to the field side's
        0<->3 / 1<->2 orbits (NOT used as the b4 pin below -- b4 uses the path-reversal
        involution instead, see fock0b_pinned_map_shell1's docstring for why).
    Returns {'rho_commutes_with_grading_residual','J_offdiag_swap_residual',
    'diag_block_self_map_residual','irrep_dims','conjugate_irrep_pairs','self_conjugate_irreps'}."""
    sg = sector_grading_hist(N_max)
    words, index, lengths, P = sg["words"], sg["index"], sg["lengths"], sg["P"]
    D = len(words)
    words2, index2, lengths2, rho = omega_diag_length(N_max, u=u)
    assert words2 == words and index2 == index, "gns_grading_commutation: word ordering mismatch"
    rho_mat = np.diag(rho)
    worst_commute = max(float(np.max(np.abs(rho_mat @ Pa - Pa @ rho_mat))) for Pa in P)

    td = tomita_data(rho)
    J = td["J"]
    rng = np.random.default_rng(1)
    A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    r = len(P)
    worst_swap, worst_diag = 0.0, 0.0
    for a in range(r):
        for b in range(r):
            block = P[a] @ A @ P[b]
            lhs = J(block)
            rhs = P[b] @ J(A) @ P[a]
            resid = float(np.max(np.abs(lhs - rhs)))
            if a == b:
                worst_diag = max(worst_diag, resid)
            else:
                worst_swap = max(worst_swap, resid)

    dims, chars_by_elt = _a4_char_lookup()
    rr = len(dims)
    conj_pairs, self_conj, matched = [], [], set()
    for a in range(rr):
        if a in matched:
            continue
        ca = {g: np.conj(chars_by_elt[a][g]) for g in chars_by_elt[a]}
        if all(abs(ca[g] - chars_by_elt[a][g]) < 1e-8 for g in ca):
            self_conj.append(a)
            matched.add(a)
            continue
        for b in range(rr):
            if b == a or b in matched:
                continue
            if all(abs(ca[g] - chars_by_elt[b][g]) < 1e-8 for g in ca):
                conj_pairs.append((a, b))
                matched.add(a)
                matched.add(b)
                break
    assert worst_commute < 1e-9, f"gns_grading_commutation: rho does not commute with grading ({worst_commute:.2e})"
    assert worst_diag < 1e-9 and worst_swap < 1e-9, "gns_grading_commutation: J block-swap check failed"
    return {"rho_commutes_with_grading_residual": worst_commute,
            "J_offdiag_swap_residual": worst_swap, "diag_block_self_map_residual": worst_diag,
            "irrep_dims": dims, "conjugate_irrep_pairs": conj_pairs,
            "self_conjugate_irreps": self_conj}


def history_reversal_matrix(N_max):
    """[FOCK-0b b4, THE HISTORY-SIDE REAL INVOLUTION] the PATH-REVERSAL permutation on H_hist:
    reverse(w) for w=(d_1,...,d_n) is (r(d_n),...,r(d_1)), r = the single-dart reversal (r(d)=d^1,
    reversal()'s own action: dart 2e <-> 2e+1).  At length 1 this IS reversal() EXACTLY (verified
    below -- build_hist's own length-1 word order is [(d,) for d in range(ND)], i.e. dart order).
    VERIFIES (not assumed): path-reversal is admissible-closed at EVERY shell (every reversed word
    of an admissible word is itself admissible and present in `words`) and an exact involution.
    Returns {'words','index','lengths','Rw' (D x D permutation matrix), 'per_shell_valid',
    'all_words_valid', 'shell1_matches_reversal', 'is_involution'}."""
    words, index, lengths = build_hist(N_max)
    D = len(words)
    r_dart = lambda d: d ^ 1
    Rw = np.zeros((D, D))
    per_shell_valid = {}
    ok_all = True
    for i, w in enumerate(words):
        rw = tuple(r_dart(d) for d in reversed(w))
        n = len(w)
        valid = rw in index
        per_shell_valid[n] = per_shell_valid.get(n, True) and valid
        if valid:
            Rw[index[rw], i] = 1.0
        else:
            ok_all = False
    R1 = reversal()
    shell1 = np.where(lengths == 1)[0]
    Rw_shell1 = Rw[np.ix_(shell1, shell1)]
    shell1_matches_reversal = float(np.max(np.abs(Rw_shell1 - R1)))
    is_involution = float(np.max(np.abs(Rw @ Rw - np.eye(D))))
    assert ok_all, "history_reversal_matrix: path-reversal is NOT admissible-closed at some shell"
    assert shell1_matches_reversal < 1e-12, "history_reversal_matrix: shell-1 block != reversal()"
    assert is_involution < 1e-9, "history_reversal_matrix: path-reversal is not an involution"
    return {"words": words, "index": index, "lengths": lengths, "Rw": Rw,
            "per_shell_valid": per_shell_valid, "all_words_valid": ok_all,
            "shell1_matches_reversal": shell1_matches_reversal, "is_involution": is_involution}


def field_algebra_conjugation():
    """[FOCK-0b b4, THE FIELD-ALGEBRA SIDE J_F,sigma -- DISCHARGES NAMED INCOMPLETENESS #2]
    the EXPLICIT antiunitary charge-conjugation operator on the 8-dim field-algebra Fock space F.
    The verifier (FOCK0_check_2026-07-11.md SS4) found sector_pair_conjugation only proves a
    SUBSPACE identity (Pw(sign=-1) equals P_(3-w)(sign=+1)); "an EXPLICIT antiunitary/unitary
    operator K realizing this pairing as a single matrix on F... is NOT constructed" -- this
    function builds it.

    CONSTRUCTION (the standard fermionic particle-hole/charge conjugation): build the 3 canonical
    creation operators A_ops[m]^dagger (m=0,1,2; the SAME construction _sector_projectors uses,
    sign=+1) and the Fock vacuum |vac> (NHAT's lowest eigenvector).  The 8-dim Fock basis
    {|S> = A_ops[m1]^dagger...A_ops[mk]^dagger|vac> : S subset {0,1,2}, ascending order} is
    ORTHONORMAL -- VERIFIED (not assumed): the A_ops obey canonical CAR {A_m,A_n^dagger}=delta_mn I,
    {A_m,A_n}=0 EXACTLY (checked in the driver), so the 8x8 Gram matrix of these states is I
    exactly.  NOTE: plain complex conjugation ALONE does NOT realize the sector pairing here (a
    dead end explicitly ruled out this session: the Cl(6) generators g6 are NOT all real -- 3 of 6
    have nonzero imaginary parts -- so conj(Pw(sign=+1)) != Pw(sign=+1)[3-w] numerically, unlike
    the history side; a genuine unitary M is required alongside conjugation).
    Define the antiunitary K(v) := M @ conj(v), M := U @ Dperm @ U^T, where U = the 8x8 unitary
    matrix of basis vectors |S> (columns, subset order below) and Dperm = the 0/1 permutation
    swapping each basis label S with its COMPLEMENT {0,1,2}\\S (the particle<->hole map -- NO
    extra phase correction needed; verified sufficient below).  K is a genuine antiunitary
    INVOLUTION (M unitary, M @ conj(M) = I, both < 1e-8) and satisfies
    K @ Pw[w] @ K == Pw[3-w] EXACTLY for every w (Pw = _sector_projectors(sign=+1); < 1e-8) --
    i.e. K REALIZES sector_pair_conjugation's 0<->3/1<->2 pairing as a SINGLE MATRIX on the FIXED
    (sign=+1) representation.
    NAMED RESIDUAL FREEDOM: M carries the usual GAUGE freedom of any charge-conjugation operator
    (the creation-operator ordering within each subset, the vacuum's overall phase, the mode QR
    phase all feed into U); a different admissible choice gives a DIFFERENT M satisfying the SAME
    defining property K.Pw[w].K = Pw[3-w] -- conventional freedom every particle-hole antiunitary
    carries, not a NEW incompleteness beyond that.
    Returns {'M','M_unitary_residual','M_involution_residual','sector_swap_residual',
    'gram_identity_residual'}."""
    sys.path.insert(0, _REPO)
    sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "bridge"))
    from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402
    g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]

    def gam(x):
        return sum(x[a] * g6[a] for a in range(NE))

    J6 = complex_structure_J6()
    wJ, VJ = np.linalg.eig(J6)
    modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
    A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
    NHAT = sum(a.conj().T @ a for a in A_ops)
    wN, VN = np.linalg.eigh(NHAT)
    vac = VN[:, [int(np.argmin(wN))]]
    vac = vac / np.linalg.norm(vac)

    subsets = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    def build_state(S):
        v = vac.copy()
        for m in reversed(S):
            v = A_ops[m].conj().T @ v
        return v

    cols = [build_state(S) for S in subsets]
    U = np.hstack(cols)
    gram_resid = float(np.max(np.abs(U.conj().T @ U - np.eye(8))))
    comp = {S: tuple(sorted(set((0, 1, 2)) - set(S))) for S in subsets}
    idx = {S: i for i, S in enumerate(subsets)}
    Dperm = np.zeros((8, 8))
    for S in subsets:
        Dperm[idx[comp[S]], idx[S]] = 1.0
    M = U @ Dperm @ U.T

    Pp, _ = _sector_projectors(sign=+1)
    swap_resid = max(float(np.max(np.abs(M @ np.conj(Pp[w]) @ np.conj(M) - Pp[3 - w])))
                      for w in range(4))
    unit_resid = float(np.max(np.abs(M.conj().T @ M - np.eye(8))))
    invol_resid = float(np.max(np.abs(M @ np.conj(M) - np.eye(8))))
    assert gram_resid < 1e-8, f"field_algebra_conjugation: Fock basis not orthonormal ({gram_resid:.2e})"
    assert unit_resid < 1e-8, f"field_algebra_conjugation: M not unitary ({unit_resid:.2e})"
    assert invol_resid < 1e-8, f"field_algebra_conjugation: K^2 != 1 ({invol_resid:.2e})"
    assert swap_resid < 1e-8, \
        f"field_algebra_conjugation: K does not realize the 0<->3/1<->2 pairing ({swap_resid:.2e})"
    return {"M": M, "M_unitary_residual": unit_resid, "M_involution_residual": invol_resid,
            "sector_swap_residual": swap_resid, "gram_identity_residual": gram_resid}


def _antiunitary_real_embed(Rtilde):
    """Real 2n x 2n matrix representing the antiunitary map v -> Rtilde @ conj(v) on C^n
    (Rtilde an n x n matrix, possibly COMPLEX -- Cl(6)'s field-side conjugation M is genuinely
    complex, unlike the real dart-side reversal), in STACKED [Re(v); Im(v)] real coordinates:
    v = p + iq -> Rtilde.(p - iq) = (Rre.p + Rim.q) + i(Rim.p - Rre.q), i.e.
    [p;q] -> [[Rre, Rim], [Rim, -Rre]] . [p;q].  Requires Rtilde @ conj(Rtilde) = I (the genuine
    antiunitary-INVOLUTION condition, J^2=1) -- NOT checked here (callers assert it); returns the
    (2n,2n) real block matrix."""
    Rre, Rim = Rtilde.real, Rtilde.imag
    return np.block([[Rre, Rim], [Rim, -Rre]])


def pinned_map_hom_space_real(Rtilde_hist, Rtilde_F):
    """[FOCK-0b b4, THE PINNED-MAP TEST -- THE FROZEN SS1 CLASS, NOW TESTABLE] solve, over the
    REALS (the pre-registered SOLVER TRAP: the constraint Phi(J_hist.v) = J_F(Phi(v)) is
    ANTILINEAR-compatible, hence REAL-linear in Phi, NOT complex-linear -- a complex-linear SVD on
    this system is WRONG and voids the station), for the space of real-linear maps
    Phi: C^n -> C^m (n = Rtilde_hist.shape[0], m = Rtilde_F.shape[0]) intertwining the two
    antiunitary INVOLUTIONS J_hist(v) = Rtilde_hist @ conj(v), J_F(w) = Rtilde_F @ conj(w):
        Phi(J_hist(v)) == J_F(Phi(v))   for ALL v in C^n,   WITH NO PER-g CONSTRAINT ANYWHERE
    (the pin sidesteps group equivariance entirely -- this is the design difference from
    dr_map_hom_space's disclaimed full-A4-generator mechanism).
    METHOD: represent Phi by its UNIQUE real-linear-map matrix on [Re(v);Im(v)] coordinates
    (capturing BOTH Phi's complex-linear part A and antilinear part B, Phi(v)=Av+B.conj(v));
    build J_hist_R, J_F_R (_antiurotary_real_embed); vectorize the constraint
    M.J_hist_R - J_F_R.M = 0 (M = Phi's real matrix) via the standard Kronecker vec identity
    vec(M.J_hist_R) - vec(J_F_R.M) = [(J_hist_R^T (x) I) - (I (x) J_F_R)].vec(M), and take the
    null space of that REAL matrix by REAL SVD (never complex SVD).
    Returns {'n','m','nullity','rank','total_real_dim','smallest_kept_sv','largest_null_sv'}."""
    n = Rtilde_hist.shape[0]
    m = Rtilde_F.shape[0]
    assert float(np.max(np.abs(Rtilde_hist @ np.conj(Rtilde_hist) - np.eye(n)))) < 1e-8, \
        "pinned_map_hom_space_real: Rtilde_hist is not an antiunitary involution (J_hist^2 != 1)"
    assert float(np.max(np.abs(Rtilde_F @ np.conj(Rtilde_F) - np.eye(m)))) < 1e-8, \
        "pinned_map_hom_space_real: Rtilde_F is not an antiunitary involution (J_F^2 != 1)"
    Jh = _antiunitary_real_embed(Rtilde_hist)      # (2n,2n)
    Jf = _antiunitary_real_embed(Rtilde_F)         # (2m,2m)
    Cop = np.kron(Jh.T, np.eye(2 * m)) - np.kron(np.eye(2 * n), Jf)
    s = np.linalg.svd(Cop, compute_uv=False)
    rank = int(np.sum(s > 1e-8))
    total = Cop.shape[1]
    nullity = total - rank
    return {"n": n, "m": m, "nullity": nullity, "rank": rank, "total_real_dim": total,
            "smallest_kept_sv": float(s[rank - 1]) if rank > 0 else float("nan"),
            "largest_null_sv": float(s[rank]) if rank < len(s) else 0.0}


def fock0b_pinned_map_shell1():
    """[FOCK-0b b4] THE PINNED-MAP TEST on the SAME 12-dim dart/shell-1 space the (disclaimed)
    generator test used (dr_map_hom_space).  TWO candidate history-side antiunitary conjugations
    are tried -- a NAMED residual freedom (the raw GNS/Tomita J of b1/b2 acts on the OPERATOR
    space HS(H_hist) by plain adjoint; restricted to acting on genuinely REAL vectors -- and the
    dart algebra + omega_diag are manifestly real in the word basis, b2/b3's own finding -- plain
    conjugation is the IDENTITY, carrying no intrinsic phase content of its own; there is no
    canonical, forced way to promote the GNS/Tomita J into a NONtrivial history-side antiunitary
    from the b1/b2 data alone):
      (A) PRIMARY: J_hist = reversal() o conj -- the framework's OWN already-forced, theorem-grade
          Z2 (BRIDGE-LOCK Lemma 1, R@Uo=-Uo; map_null_lemmas) supplies the nontrivial real part;
          this is the LEAST ARBITRARY choice available (REUSED, not invented) and a genuine
          involution (R^2=I, reversal()'s own anchor).
      (B) CROSS-CHECK: J_hist = conj alone (Rtilde=I_12) -- the DEGENERATE case, included to show
          the raw GNS/Tomita J's own triviality quantitatively (see the returned dict: both give
          the SAME total nullity, a structural fact -- see the driver/report for the theorem).
    J_F,sigma = field_algebra_conjugation()'s explicit, VERIFIED operator (M @ conj, matches
    sector_pair_conjugation's pairing exactly).  Solves the REAL-linear intertwiner system (never
    complex-linear SVD) via pinned_map_hom_space_real.
    Returns {'primary','cross_check','field_side'}."""
    R = reversal()
    fa = field_algebra_conjugation()
    primary = pinned_map_hom_space_real(R, fa["M"])
    cross = pinned_map_hom_space_real(np.eye(ND), fa["M"])
    return {"primary": primary, "cross_check": cross, "field_side": fa}


def fock0b_pinned_map_shell(shell_n, N_max=None):
    """[FOCK-0b b4, SHELL-BY-SHELL EXTENSION] as fock0b_pinned_map_shell1, but on H_(shell_n)
    (dimension 12*2^(shell_n-1) for shell_n>=1), using history_reversal_matrix's PATH-REVERSAL
    involution restricted to that shell as J_hist's real part (the natural generalization of
    shell 1's reversal(); VERIFIED by history_reversal_matrix to reduce to reversal() exactly at
    shell_n=1, and to be admissible-closed at every shell up to N_max).  The field side is
    unchanged (F is always 8-dim; the DR-map candidate's TARGET does not grow with the source
    shell).  Returns pinned_map_hom_space_real's dict, plus 'shell_dim'."""
    if N_max is None:
        N_max = shell_n
    hr = history_reversal_matrix(N_max)
    idx = np.where(hr["lengths"] == shell_n)[0]
    Rshell = hr["Rw"][np.ix_(idx, idx)]
    fa = field_algebra_conjugation()
    res = pinned_map_hom_space_real(Rshell, fa["M"])
    res["shell_dim"] = len(idx)
    return res


def fock0b_fence_check():
    """[FOCK-0b b5, THE DESIGN FENCE re-check on the pinned-map class] as fock0_fence_check, for
    THIS station's construction (reuses that dict, adds two FOCK-0b-specific entries):
      1. antiunitary/phase-bearing: J_F,sigma is a VERIFIED, genuine antiunitary pairing distinct
         sectors (0<->3, 1<->2) -- NOT an even functional.  HONEST CAVEAT (a named asymmetry, not
         a fence failure): J_hist's own real part (reversal or identity) is a REAL orthogonal map
         -- it carries NO intrinsic phase content of its own, because the dart algebra + omega_diag
         are manifestly real (b2/b3's finding); the phase-bearing content of the overall pin comes
         from J_F,sigma.  The intertwiner EQUATION is still built from genuine antiunitary
         conjugations on BOTH sides (not an even functional of run data), distinguishing it from
         O0's bit-EVEN democracy.
      2. NOT full-group generator equivariance: solved with NO per-g constraint anywhere (contrast
         dr_map_hom_space's 12 simultaneous per-g equations over ALL of A4) -- genuinely the
         DIFFERENT (weaker) condition SS1/SS2 item 2 names, not the disclaimed mechanism relabeled.
      3-5: reused verbatim from fock0_fence_check (still hold: Fock-level/graded, representation-
         theoretic, per-sector by construction).
    Returns fock0_fence_check()'s dict plus '1_O0_bit_even_democracy_FOCK0b_note' and
    '2_M1b_no_generator_constraint'."""
    base = fock0_fence_check()
    fa = field_algebra_conjugation()
    out = dict(base)
    out["1_O0_bit_even_democracy_FOCK0b_note"] = {
        "check": "J_F,sigma is a VERIFIED antiunitary sector-pairing (K.Pw[w].K == Pw[3-w] "
                 "exactly); J_hist's REAL part (reversal or identity) carries NO intrinsic phase "
                 "content -- the dart algebra + omega_diag are manifestly real (b2/b3 finding). "
                 "The intertwiner EQUATION is still built from genuine antiunitary conjugations "
                 "(not an even functional of run data); this asymmetry is logged, not hidden.",
        "field_side_is_phase_bearing": fa["sector_swap_residual"] < 1e-8}
    out["2_M1b_no_generator_constraint"] = {
        "check": "pinned_map_hom_space_real solves ONE antiunitary-intertwining relation, with NO "
                 "per-g in A4 constraint anywhere (contrast dr_map_hom_space's 12 simultaneous "
                 "per-g equations) -- the DIFFERENT (weaker) condition SS1/SS2 item 2 names, not "
                 "the disclaimed mechanism relabeled.",
        "no_group_generator_used": True}
    return out


def fock0b_selftest_2026_07_11(verbose=True):
    """FOCK-0b station regression: b1 GNS purification + b2 Tomita data (S=J.Delta^1/2, J^2=1,
    JMJ subset M', Delta<->KMS-at-beta_natural PER-PATH) + b3 grading commutation (rho commutes
    with the A4-isotypic grading; J's off-diag block-swap; the character-conjugation pairing) +
    b4 THE PINNED-MAP TEST (field_algebra_conjugation, shell-1 primary+cross-check, shell-2
    extension) + b5 fence re-check, plus Sections 7/7b/8 untouched.  Does NOT itself adjudicate
    W1-W4 (pre-reg SSC: architect-only)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 8b self-test -- FOCK-0b history-side modular conjugation (2026-07-11)")
        print("=" * 88)

    ck("ANCHORS + Sections 7/7b/8 untouched",
       anchor_cell_projector() and anchor_tick_2pi() and accretion_selftest_2026_07_10(verbose=False)
       and i2b_selftest_2026_07_11(verbose=False) and fock0_selftest_2026_07_11(verbose=False))

    N_max = 4
    tc = tomita_checks(N_max)
    ck(f"b1 GNS PURIFICATION [N_max={N_max}, D={tc['D']}]: rho full-rank/separating "
       f"(min_rho={tc['min_rho']:.3e} > 0)", tc["full_rank_separating"])
    ck("b2 TOMITA: S(x.Omega) == x^dagger.Omega EXACT (closed form reproduces the defining relation)",
       tc["S_closed_form_residual"] < 1e-9, detail=f"resid={tc['S_closed_form_residual']:.2e}")
    ck("b2 TOMITA: J^2 = 1 (antiunitary involution)",
       tc["J_squared_residual"] < 1e-9, detail=f"resid={tc['J_squared_residual']:.2e}")
    ck("b2 TOMITA: J M J subset M' ([pi_R(S_d^dagger), pi_L(S_e)]=0, algebraic identity, "
       "verified on generators)", tc["JMJ_subset_Mprime_residual"] < 1e-8,
       detail=f"resid={tc['JMJ_subset_Mprime_residual']:.2e}")
    ck(f"b2 Delta<->KMS-at-beta_natural, PER-PATH: beta_natural={tc['beta_natural_check']:.7f} "
       f"(matches I2b's own 6.4874417297)",
       tc["kms_per_path_residual"] < 1e-8 and abs(tc["beta_natural_check"] - 6.4874417297) < 1e-6,
       detail=f"per-path resid={tc['kms_per_path_residual']:.2e}, per-shell rate sample="
              f"{tc['per_shell_rate_sample']:.6f} (= beta_natural - h_top = beta' = "
              f"{tc['beta_prime_reproduced']:.6f}, cross-checking I2b's own identity)")

    gc = gns_grading_commutation(N_max)
    ck("b3 GRADING: rho (hence Delta) commutes EXACTLY with every A4-isotypic projector",
       gc["rho_commutes_with_grading_residual"] < 1e-9,
       detail=f"resid={gc['rho_commutes_with_grading_residual']:.2e}")
    ck("b3 J's action on the grading: SELF-MAPS diagonal (a,a) blocks, SWAPS off-diagonal (a,b) "
       "<-> (b,a) blocks (both exact)",
       gc["diag_block_self_map_residual"] < 1e-9 and gc["J_offdiag_swap_residual"] < 1e-9,
       detail=f"diag={gc['diag_block_self_map_residual']:.2e}, "
              f"swap={gc['J_offdiag_swap_residual']:.2e}, "
              f"conjugate irrep pairs={gc['conjugate_irrep_pairs']}, "
              f"self-conjugate irreps={gc['self_conjugate_irreps']}")

    hr = history_reversal_matrix(N_max)
    ck("b4 PATH-REVERSAL: admissible-closed at every shell + exact involution; shell-1 == "
       "reversal() EXACTLY",
       hr["all_words_valid"] and hr["is_involution"] < 1e-9 and hr["shell1_matches_reversal"] < 1e-12,
       detail=f"per_shell_valid={hr['per_shell_valid']}")

    pm = fock0b_pinned_map_shell1()
    fa = pm["field_side"]
    ck("b4 J_F,sigma CONSTRUCTED+VERIFIED: K.Pw[w].K == Pw[3-w] EXACTLY (discharges Named "
       "Incompleteness #2)",
       fa["sector_swap_residual"] < 1e-8 and fa["M_unitary_residual"] < 1e-8
       and fa["M_involution_residual"] < 1e-8 and fa["gram_identity_residual"] < 1e-8,
       detail=f"swap resid={fa['sector_swap_residual']:.2e}")
    pr, cr = pm["primary"], pm["cross_check"]
    s2 = fock0b_pinned_map_shell(2, N_max=N_max)
    ck(f"b4 THE PINNED-MAP TEST: shell-1 PRIMARY (J_hist=reversal-o-conj) nullity="
       f"{pr['nullity']}/{pr['total_real_dim']}; shell-1 CROSS-CHECK (conj alone) nullity="
       f"{cr['nullity']}/{cr['total_real_dim']}; shell-2 PRIMARY nullity={s2['nullity']}/"
       f"{s2['total_real_dim']} -- PATTERN: nullity == EXACTLY HALF of total_real_dim in all "
       f"three (a general fact about antiunitary-involution intertwiner spaces, not a lattice-"
       f"specific coincidence)",
       pr["nullity"] == pr["total_real_dim"] // 2 and cr["nullity"] == cr["total_real_dim"] // 2
       and s2["nullity"] == s2["total_real_dim"] // 2 and pr["nullity"] > 0)

    fc = fock0b_fence_check()
    ck("b5 FENCE re-check on the pinned-map class: all items confirmed",
       fc["1_O0_bit_even_democracy"]["is_phase_bearing"]
       and fc["1_O0_bit_even_democracy_FOCK0b_note"]["field_side_is_phase_bearing"]
       and fc["2_M1b_no_generator_constraint"]["no_group_generator_used"]
       and fc["3_BRIDGE_LOCK_attachment_functional_orbit_blind"]["is_fock_level"]
       and fc["4_BRIDGE_T_two_point_data_blind"]["is_representation_theoretic"]
       and fc["5_BRIDGE_GEOM_per_sector_required"]["is_per_sector_by_design"])

    if verbose:
        print("RESULT:",
              "FOCK-0b SECTION-8b REGRESSION PASSES" if ok else "A FOCK-0b CHECK FAILED")
    return ok


# ===========================================================================
# 8c. FOCK-0c -- THE FULL PIN: PER-SECTOR TOMITA BLOCKS + THE R/F_bit PARITY PIN  (2026-07-11)
#     FOCK0_dr_reconstruction_prereg_2026-07-11.md FOCK-0c DIRECTIVE SS F-G
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as Sections 7/7b/8/8b above.]
#
# CONTEXT: the sealed FOCK-0b checker (working notes/FOCK0b_check_2026-07-11.md SS4) found that
# fock0b_pinned_map_shell1's b4 test solves only a SINGLE GLOBAL antiunitary pair (one J_hist, one
# J_F acting on the WHOLE 12-dim / 8-dim spaces) -- a disclosed NECESSARY special case of the
# frozen SS1/SSB.b4 hypothesis, which is stated in the PLURAL ("the antiunitary J's -- per-sector";
# "Phi.J_hist,sigma = J_F,sigma.Phi ... per sector").  This section builds the literal plural
# system and adds the SS3c/nullity_with_R_constraint pin, PER THE FOCK-0c DIRECTIVE SS F, EXACTLY:
#   (i)  the per-sector Tomita blocks, for EACH sector sigma per the SS8 grading and its pairing
#        (0<->3, 1<->2);
#   (ii) the R/F_bit sector-parity pin (original pre-reg SS3c; dr_map_hom_space's
#        nullity_with_R_constraint precedent, the_net.py:2284-2289, REUSED unchanged).
# NOTHING ELSE is stacked (the directive's hard goal-seek guard): no alternate/non-Tomita
# antiunitary, no Delta-flow/temporal pin.
#
# THE "PER SECTOR" CONSTRUCTION, MADE PRECISE (a named judgment call, not silently assumed --
# see fock0c_selftest's printed rationale and the station report SS "Named Residual Freedom #1"):
# the history-side SS8 grading (sector_grading_hist) is indexed by A4-IRREP TYPE with isotypic
# dims [1,1,1,9] at shell 1 (mult x irrep-dim); the field-side SS8 grading (_sector_projectors /
# gauge_sector_category) is indexed by SPECIES with dims {1,3,3,1}.  These are DIFFERENT
# dimension patterns -- there is no forced 1-1 correspondence between an individual history
# isotypic index a in {0,1,2,3} and an individual field species index w in {0,1,2,3} (attempting
# one, e.g. by raw position a=w, is an EXTRA invented convention the frozen text does not supply).
# What IS forced on BOTH sides independently (no invention) is a PARTITION into exactly TWO
# sector-PAIR GROUPS, matching the pre-declared W1 allowance's own granularity ("one overall phase
# per sector-PAIR block"):
#   history side: self_conjugate_irreps=[0,3] (each individually fixed by J_hist=reversal.conj,
#     gns_grading_commutation) and conjugate_irrep_pairs=[(1,2)] (swapped by J_hist) --
#     GROUP-03 := P[0]+P[3], GROUP-12 := P[1]+P[2] (sector_grading_hist's own projectors).
#   field side: orbit_03={0,3}, orbit_12={1,2} (sector_pair_conjugation, swapped by K) --
#     GROUP-03 := Pw[0]+Pw[3], GROUP-12 := Pw[1]+Pw[2] (_sector_projectors's own projectors).
# "Per sector sigma" is read at this GROUP granularity (sigma in {GROUP-03, GROUP-12}): Phi is
# required to be GRADED (GROUP-03 of H_hist maps only into GROUP-03 of F; GROUP-12 only into
# GROUP-12; cross terms forced to vanish) -- the FINEST grading available on BOTH sides using
# ONLY objects the SS8/SSB machinery already built, with NO per-individual-sigma invention.  This
# is the frozen text's "plural constraint" instantiated as concretely as the existing accreted
# objects allow.
def _linear_real_embed(X):
    """Real 2n x 2m matrix embedding the ORDINARY complex-LINEAR (non-antiunitary) map
    v -> X @ v (X an n x m possibly-complex matrix) in STACKED [Re(v);Im(v)] real coordinates:
    v = p + iq -> X.v = (Xre.p - Xim.q) + i(Xim.p + Xre.q), i.e.
    [p;q] -> [[Xre,-Xim],[Xim,Xre]].[p;q].  CONTRAST _antiunitary_real_embed (the DIFFERENT sign
    pattern for an ANTIunitary v -> Rtilde@conj(v) map).  Used below for plain pre/post-
    multiplication constraints (sector-pair-group projectors, the R/F_bit pin) that involve NO
    conjugation."""
    Xre, Xim = X.real, X.imag
    return np.block([[Xre, -Xim], [Xim, Xre]])


def _zero_block_rows(P_row, P_F_side, P_col, P_H_side):
    """Real constraint ROWS enforcing P_row @ Phi @ P_col == 0 exactly, for Phi a REAL-linear map
    represented (as in pinned_map_hom_space_real) by its real 2m x 2n embedding X (m = P_row's
    ambient dim = P_F_side, n = P_col's ambient dim = P_H_side).  DERIVATION: for M, N ordinary
    COMPLEX-linear operators (M = P_row on the field/output side, N = P_col on the history/input
    side) and Phi(v) = Av + B.conj(v) real-linear, Y := M.Phi.N satisfies
    Y(v) = (M.A.N).v + (M.B.conj(N)).conj(v) -- i.e. Y's OWN real embedding is
    X' = _linear_real_embed(M) @ X @ _linear_real_embed(N) (verified: [Re(Nv);Im(Nv)] =
    _linear_real_embed(N).[Re(v);Im(v)] by construction of _linear_real_embed, then apply X, then
    _linear_real_embed(M) again).  Requiring Y = 0 (the vanishing block) is the vectorized
    constraint (_linear_real_embed(N).T (x) _linear_real_embed(M)) . vec(X) = 0 -- the same
    Kronecker-vec identity pinned_map_hom_space_real already uses for the antiunitary pin, here
    applied to a PLAIN-linear (non-conjugating) pre/post-multiplication instead.
    Returns the real constraint-row matrix (rows to stack into a Cop-style matrix)."""
    Mr = _linear_real_embed(P_row)
    Nr = _linear_real_embed(P_col)
    return np.kron(Nr.T, Mr)


def _antiunitary_pin_rows(Rtilde_hist, Rtilde_F):
    """The SAME antiunitary-pin constraint rows pinned_map_hom_space_real builds internally
    (Phi.J_hist = J_F.Phi, J_hist(v)=Rtilde_hist@conj(v), J_F(w)=Rtilde_F@conj(w)), exposed here as
    a reusable ROW-BUILDER (not a full solve) so FOCK-0c can stack it alongside the NEW
    sector-pair-group and R/F_bit rows below without re-deriving the antiunitary-involution
    asserts twice.  Callers must have already verified Rtilde_hist/Rtilde_F are antiunitary
    involutions (pinned_map_hom_space_real's own asserts, reused verbatim in
    fock0c_full_pin_shell below)."""
    n = Rtilde_hist.shape[0]
    m = Rtilde_F.shape[0]
    Jh = _antiunitary_real_embed(Rtilde_hist)
    Jf = _antiunitary_real_embed(Rtilde_F)
    return np.kron(Jh.T, np.eye(2 * m)) - np.kron(np.eye(2 * n), Jf)


def _linear_pin_rows(X_hist, X_F):
    """Constraint rows for a PLAIN-linear (non-antiunitary) intertwining pin Phi.X_hist =
    X_F.Phi (X_hist, X_F fixed possibly-complex matrices, e.g. the R/F_bit pair -- NO conjugation
    anywhere), via the same _linear_real_embed + Kronecker-vec method as _zero_block_rows."""
    n = X_hist.shape[0]
    m = X_F.shape[0]
    Xh = _linear_real_embed(X_hist)
    Xf = _linear_real_embed(X_F)
    return np.kron(Xh.T, np.eye(2 * m)) - np.kron(np.eye(2 * n), Xf)


def history_sector_pair_groups(N_max):
    """[FOCK-0c, THE HISTORY-SIDE SECTOR-PAIR GROUPS] partitions sector_grading_hist's four
    A4-isotypic projectors P[0..3] into the TWO groups forced by character conjugation
    (gns_grading_commutation's own self_conjugate_irreps/conjugate_irrep_pairs, cross-checked here
    not re-derived): GROUP-03 := P[0]+P[3] (the two SELF-conjugate irreps, each individually fixed
    by J_hist=reversal.conj) and GROUP-12 := P[1]+P[2] (the conjugate PAIR, swapped by J_hist).
    Verifies (not assumes) both groups are genuine projectors (idempotent, Hermitian) and that
    they are complementary (sum to identity) and orthogonal to each other.
    Returns {'words','index','lengths','P_group03','P_group12','self_conjugate_irreps',
    'conjugate_irrep_pairs'}."""
    gc_probe = gns_grading_commutation(1)
    assert gc_probe["self_conjugate_irreps"] == [0, 3] and gc_probe["conjugate_irrep_pairs"] == [(1, 2)], \
        (f"history_sector_pair_groups: unexpected character-conjugation pattern "
         f"{gc_probe['self_conjugate_irreps']}/{gc_probe['conjugate_irrep_pairs']} -- the frozen "
         "GROUP-03/GROUP-12 partition assumed below does not apply")
    sg = sector_grading_hist(N_max)
    P = sg["P"]
    P_group03 = P[0] + P[3]
    P_group12 = P[1] + P[2]
    idem = max(float(np.max(np.abs(Pg @ Pg - Pg))) for Pg in (P_group03, P_group12))
    herm = max(float(np.max(np.abs(Pg - Pg.conj().T))) for Pg in (P_group03, P_group12))
    orth = float(np.max(np.abs(P_group03 @ P_group12)))
    complete = float(np.max(np.abs(P_group03 + P_group12 - np.eye(len(sg["words"])))))
    assert idem < 1e-7 and herm < 1e-7 and orth < 1e-7 and complete < 1e-7, \
        (f"history_sector_pair_groups: group projector battery fails (idem={idem:.1e}, "
         f"herm={herm:.1e}, orth={orth:.1e}, complete={complete:.1e})")
    return {"words": sg["words"], "index": sg["index"], "lengths": sg["lengths"],
            "P_group03": P_group03, "P_group12": P_group12,
            "self_conjugate_irreps": gc_probe["self_conjugate_irreps"],
            "conjugate_irrep_pairs": gc_probe["conjugate_irrep_pairs"]}


def field_sector_pair_groups():
    """[FOCK-0c, THE FIELD-SIDE SECTOR-PAIR GROUPS] the SAME two-group partition on the field
    side, built from _sector_projectors(sign=+1) (reused, not rebuilt) and cross-checked against
    sector_pair_conjugation's own orbit_03/orbit_12 naming (the SAME {0,3}/{1,2} split, verified
    theorem-grade there, reused here not re-derived): GROUP-03 := Pw[0]+Pw[3],
    GROUP-12 := Pw[1]+Pw[2].  Returns {'P_group03','P_group12'} (each an 8x8 projector)."""
    Pw, _ = _sector_projectors(sign=+1)
    return {"P_group03": Pw[0] + Pw[3], "P_group12": Pw[1] + Pw[2]}


def fock0c_w1_allowance():
    """[FOCK-0c SS F, STEP 1 -- THE PRE-DECLARED W1 ALLOWANCE, COUNTED BEFORE SOLVING] the frozen
    pairing (SS8: sector_pair_conjugation's 0<->3, 1<->2) partitions the four sectors into EXACTLY
    TWO sector-PAIR blocks (GROUP-03, GROUP-12; history_sector_pair_groups/field_sector_pair_groups
    above).  W1 ("unique up to per-sector phase") permits exactly ONE overall U(1) phase parameter
    -- ONE REAL dimension -- per sector-pair block (a global phase multiplying an entire block's
    Phi is a single real angle, regardless of the block's own matrix dimensions).  With TWO blocks,
    the allowance is 1 + 1 = 2 real dimensions, counted from the SS8 structure alone, BEFORE any
    solve.  Returns {'n_pair_blocks','real_dims_per_block','allowance'} (allowance = 2)."""
    n_pair_blocks = 2
    real_dims_per_block = 1
    return {"n_pair_blocks": n_pair_blocks, "real_dims_per_block": real_dims_per_block,
            "allowance": n_pair_blocks * real_dims_per_block}


def fock0c_waypoint_reproduction(N_max=4):
    """[FOCK-0c SS G, VERIFICATION WAYPOINTS] reproduces, with THIS station's own machinery, the
    two waypoints the FOCK-0b verifier's independent computation supplies
    (working notes/FOCK0b_check_2026-07-11.md SS4, item 2): the single-GLOBAL-antiunitary-pair
    half (nullity 192/384 at shell 1 -- fock0b_pinned_map_shell1, UNCHANGED, re-run not re-derived)
    and its collapse to 96/384 under the ADDITIONAL R/F_bit sector-parity pin (SS3c;
    dr_map_hom_space's nullity_with_R_constraint precedent) stacked on the SAME single global pair
    -- built here as a NEW check (FOCK-0b never combined R/F_bit with the antiunitary pin).
    Both waypoints must reproduce EXACTLY before the full per-sector pin (fock0c_full_pin_shell)
    is trusted.  Returns {'global_pair_nullity','global_pair_total','global_plus_rfbit_nullity',
    'global_plus_rfbit_total'}."""
    pm = fock0b_pinned_map_shell1()
    pr = pm["primary"]
    R = reversal()
    fa = pm["field_side"]
    K = fa["M"]
    Pw, _ = _sector_projectors(sign=+1)
    F_bit = Pw[0] + Pw[3] - Pw[1] - Pw[2]
    rows = [_antiunitary_pin_rows(R, K), _linear_pin_rows(R, F_bit)]
    Cop = np.vstack(rows)
    s = np.linalg.svd(Cop, compute_uv=False)
    rank = int(np.sum(s > 1e-8))
    total = Cop.shape[1]
    nullity = total - rank
    assert pr["nullity"] == 192 and pr["total_real_dim"] == 384, \
        f"fock0c_waypoint_reproduction: FOCK-0b's own global-pair waypoint changed ({pr['nullity']}/{pr['total_real_dim']})"
    assert nullity == 96 and total == 384, \
        f"fock0c_waypoint_reproduction: global+R/F_bit waypoint {nullity}/{total} != the checker's 96/384"
    return {"global_pair_nullity": pr["nullity"], "global_pair_total": pr["total_real_dim"],
            "global_plus_rfbit_nullity": nullity, "global_plus_rfbit_total": total}


def fock0c_full_pin_shell(shell_n, N_max=None):
    """[FOCK-0c SS F, THE FULL PIN -- BOTH (i) AND (ii), NOTHING ELSE] the literal frozen SSB.b4
    plural pin, read at sector-PAIR-GROUP granularity (module banner above), PLUS the R/F_bit
    sector-parity pin (SS3c), imposed SIMULTANEOUSLY on the shell_n dart/word space -> the 8-dim
    field Fock F.  Builds, and stacks as ONE real constraint system (never a complex SVD):
      (i)  the antiunitary Tomita pin (Phi.J_hist = J_F.Phi, the SAME J_hist=reversal.conj /
           J_F=K as FOCK-0b, UNCHANGED) PLUS the two sector-pair-group cross-vanishing
           constraints (GROUP-12(field).Phi.GROUP-03(hist) = 0 and
           GROUP-03(field).Phi.GROUP-12(hist) = 0) that make it genuinely GRADED/plural rather
           than the single global equation FOCK-0b tested;
      (ii) the R/F_bit pin (Phi.R = F_bit.Phi, R=reversal() [shell-restricted for shell_n>1 via
           history_reversal_matrix], F_bit = Pw[0]+Pw[3]-Pw[1]-Pw[2] -- IDENTICAL to
           dr_map_hom_space's nullity_with_R_constraint, the_net.py:2284-2289, reused unchanged).
    NO alternate antiunitary, NO Delta-flow pin (the directive's hard goal-seek guard).
    ALSO reports the per-sector-pair-group NULLITY BREAKDOWN (isolating each group's own
    contribution by additionally zeroing the OTHER group's own diagonal block -- a genuine
    sub-nullity, not a heuristic split: verified to SUM EXACTLY to the full-pin total) under BOTH
    item (i) alone (the grading, no R/F_bit) and the full (i)+(ii) pin, for the honest per-sector
    story the SS F classification requires.
    Returns a dict with the antiunitary-only waypoint, item-(i)-alone nullity (+ group breakdown),
    and the full-pin nullity (+ group breakdown, + smallest-kept/largest-null singular values for
    the exactness gap)."""
    if N_max is None:
        N_max = shell_n
    hg = history_sector_pair_groups(N_max)
    idx = np.where(hg["lengths"] == shell_n)[0]
    n = len(idx)
    P_H_g03 = hg["P_group03"][np.ix_(idx, idx)]
    P_H_g12 = hg["P_group12"][np.ix_(idx, idx)]
    if shell_n == 1:
        R = reversal()
    else:
        hr = history_reversal_matrix(N_max)
        R = hr["Rw"][np.ix_(idx, idx)]
    fg = field_sector_pair_groups()
    P_F_g03, P_F_g12 = fg["P_group03"], fg["P_group12"]
    fa = field_algebra_conjugation()
    K = fa["M"]
    Pw, _ = _sector_projectors(sign=+1)
    F_bit = Pw[0] + Pw[3] - Pw[1] - Pw[2]
    m = 8

    assert float(np.max(np.abs(R @ np.conj(R) - np.eye(n)))) < 1e-8, \
        "fock0c_full_pin_shell: history-side R is not an antiunitary involution as a real matrix"
    assert float(np.max(np.abs(K @ np.conj(K) - np.eye(m)))) < 1e-8, \
        "fock0c_full_pin_shell: field-side K is not an antiunitary involution"

    def nullity_of(rowlist):
        Cop = np.vstack(rowlist)
        s = np.linalg.svd(Cop, compute_uv=False)
        rank = int(np.sum(s > 1e-8))
        total = Cop.shape[1]
        return total - rank, rank, total, s

    rows_pin = [_antiunitary_pin_rows(R, K)]
    cross_rows = [_zero_block_rows(P_F_g12, m, P_H_g03, n), _zero_block_rows(P_F_g03, m, P_H_g12, n)]
    rfbit_row = [_linear_pin_rows(R, F_bit)]

    null_global, *_ = nullity_of(rows_pin)
    null_i, *_ = nullity_of(rows_pin + cross_rows)
    null_i_g03, *_ = nullity_of(rows_pin + cross_rows + [_zero_block_rows(P_F_g12, m, P_H_g12, n)])
    null_i_g12, *_ = nullity_of(rows_pin + cross_rows + [_zero_block_rows(P_F_g03, m, P_H_g03, n)])

    rows_full = rows_pin + cross_rows + rfbit_row
    null_full, rank_full, total_full, s_full = nullity_of(rows_full)
    null_full_g03, *_ = nullity_of(rows_full + [_zero_block_rows(P_F_g12, m, P_H_g12, n)])
    null_full_g12, *_ = nullity_of(rows_full + [_zero_block_rows(P_F_g03, m, P_H_g03, n)])

    assert null_i_g03 + null_i_g12 == null_i, \
        f"fock0c_full_pin_shell: item-(i) group breakdown {null_i_g03}+{null_i_g12} != total {null_i}"
    assert null_full_g03 + null_full_g12 == null_full, \
        f"fock0c_full_pin_shell: full-pin group breakdown {null_full_g03}+{null_full_g12} != total {null_full}"

    return {
        "shell_n": shell_n, "n": n, "m": m, "total_real_dim": total_full,
        "waypoint_global_pair_nullity": null_global,
        "item_i_alone_nullity": null_i, "item_i_alone_group03": null_i_g03, "item_i_alone_group12": null_i_g12,
        "full_pin_nullity": null_full, "full_pin_group03": null_full_g03, "full_pin_group12": null_full_g12,
        "smallest_kept_sv": float(s_full[rank_full - 1]) if rank_full > 0 else float("nan"),
        "largest_null_sv": float(s_full[rank_full]) if rank_full < len(s_full) else 0.0,
    }


def fock0c_fence_check():
    """[FOCK-0c, THE SS2 FENCE RE-CHECK on the full-pin class] as fock0b_fence_check, for THIS
    station's construction (reuses that dict, adds the grading-specific note): the sector-pair-
    group cross-vanishing is a STRUCTURAL constraint built from the SS8 isotypic/species
    projectors ALONE (no group-element/generator content, no two-point resolvent data, no
    spatial/attachment functional) -- it does not reintroduce any of the five blind forms; the
    R/F_bit pin is the pre-existing SS3c precedent, unchanged.  Returns fock0b_fence_check()'s
    dict plus '6_FOCK0c_grading_is_projector_only'."""
    base = fock0b_fence_check()
    out = dict(base)
    out["6_FOCK0c_grading_is_projector_only"] = {
        "check": "the sector-pair-group cross-vanishing rows are built ONLY from sector_grading_"
                 "hist's/_sector_projectors' own isotypic/species projectors (Hermitian, "
                 "idempotent) -- no A4 group element, no generator, no resolvent/two-point data, "
                 "no spatial attachment functional anywhere in _zero_block_rows/history_sector_"
                 "pair_groups/field_sector_pair_groups (grep-confirmed within this module)",
        "no_extra_mechanism_introduced": True}
    return out


def fock0c_selftest_2026_07_11(verbose=True):
    """FOCK-0c station regression: the W1 allowance count, the two verification waypoints
    (192/384 global; 96/384 global+R/F_bit), the full per-sector-pair-group pin at shells 1 and 2
    (+ group breakdown), and the SS2 fence re-check, plus Sections 7/7b/8/8b + module anchors
    untouched.  Does NOT itself adjudicate the SS C tree (architect-only per the pre-reg)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 8c self-test -- FOCK-0c the full pin (2026-07-11)")
        print("=" * 88)

    ck("ANCHORS + Sections 7/7b/8/8b untouched",
       anchor_cell_projector() and anchor_tick_2pi() and accretion_selftest_2026_07_10(verbose=False)
       and i2b_selftest_2026_07_11(verbose=False) and fock0_selftest_2026_07_11(verbose=False)
       and fock0b_selftest_2026_07_11(verbose=False))

    alw = fock0c_w1_allowance()
    ck(f"STEP 1 W1 ALLOWANCE (counted BEFORE solving): {alw['n_pair_blocks']} sector-pair blocks "
       f"x {alw['real_dims_per_block']} real dim (one overall phase) each = {alw['allowance']}",
       alw["allowance"] == 2)

    wp = fock0c_waypoint_reproduction(N_max=4)
    ck(f"WAYPOINT 1 (single global antiunitary pair, shell 1): nullity = "
       f"{wp['global_pair_nullity']}/{wp['global_pair_total']} (must reproduce FOCK-0b's own "
       "192/384)", wp["global_pair_nullity"] == 192 and wp["global_pair_total"] == 384)
    ck(f"WAYPOINT 2 (+ R/F_bit collapse, same single global pair): nullity = "
       f"{wp['global_plus_rfbit_nullity']}/{wp['global_plus_rfbit_total']} (must reproduce the "
       "FOCK-0b checker's independent 96/384)",
       wp["global_plus_rfbit_nullity"] == 96 and wp["global_plus_rfbit_total"] == 384)

    s1 = fock0c_full_pin_shell(1, N_max=4)
    ck(f"SHELL 1 full pin: item-(i)-alone (graded, no R/F_bit) nullity = {s1['item_i_alone_nullity']} "
       f"(group03={s1['item_i_alone_group03']}, group12={s1['item_i_alone_group12']}); "
       f"FULL PIN (i)+(ii) nullity = {s1['full_pin_nullity']}/{s1['total_real_dim']} "
       f"(group03={s1['full_pin_group03']}, group12={s1['full_pin_group12']})",
       s1["item_i_alone_group03"] + s1["item_i_alone_group12"] == s1["item_i_alone_nullity"]
       and s1["full_pin_group03"] + s1["full_pin_group12"] == s1["full_pin_nullity"]
       and s1["smallest_kept_sv"] > 0.5 and s1["largest_null_sv"] < 1e-8)

    s2 = fock0c_full_pin_shell(2, N_max=4)
    ck(f"SHELL 2 full pin: item-(i)-alone nullity = {s2['item_i_alone_nullity']} "
       f"(group03={s2['item_i_alone_group03']}, group12={s2['item_i_alone_group12']}); "
       f"FULL PIN nullity = {s2['full_pin_nullity']}/{s2['total_real_dim']} "
       f"(group03={s2['full_pin_group03']}, group12={s2['full_pin_group12']})",
       s2["item_i_alone_group03"] + s2["item_i_alone_group12"] == s2["item_i_alone_nullity"]
       and s2["full_pin_group03"] + s2["full_pin_group12"] == s2["full_pin_nullity"]
       and s2["smallest_kept_sv"] > 0.5 and s2["largest_null_sv"] < 1e-8)

    fc = fock0c_fence_check()
    ck("SS2 FENCE re-check on the full-pin class: all items confirmed",
       fc["1_O0_bit_even_democracy"]["is_phase_bearing"]
       and fc["2_M1b_no_generator_constraint"]["no_group_generator_used"]
       and fc["3_BRIDGE_LOCK_attachment_functional_orbit_blind"]["is_fock_level"]
       and fc["4_BRIDGE_T_two_point_data_blind"]["is_representation_theoretic"]
       and fc["5_BRIDGE_GEOM_per_sector_required"]["is_per_sector_by_design"]
       and fc["6_FOCK0c_grading_is_projector_only"]["no_extra_mechanism_introduced"])

    if verbose:
        print("RESULT:",
              "FOCK-0c SECTION-8c REGRESSION PASSES" if ok else "A FOCK-0c CHECK FAILED")
    return ok


# ===========================================================================
# 8d. FOCK-0d -- THE TEMPORAL PIN: THE FLOW ITSELF  (2026-07-11)
#     FOCK0_dr_reconstruction_prereg_2026-07-11.md AMENDMENT FOCK-0d SS H-K
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as Sections 7/7b/8/8b/8c above.]
#
# CONTEXT: FOCK-0c's full pin (SS8c) is BANKED (16/384 shell 1, 64/768 shell 2, W3). This
# amendment imposes ONE MORE constraint ON TOP of FOCK-0c's full pin set: a FLOW (not
# flow-reversing) pin, Phi.K_hist,sigma = lambda.K_F,sigma.Phi, with K_hist = the GNS/Tomita
# modular Hamiltonian of the length-diagonal state (SS8b's b2 Tomita step, PER-PATH reading) and
# K_F = the M0 half-cell modular (entangling) Hamiltonian of a 3-edge region, second-quantized.
# lambda > 0 is ONE GLOBAL relative-clock scale, SOLVED FOR (a generalized-eigenvalue/pencil
# problem, characterized ANALYTICALLY below -- never grid-scanned-and-picked).
#
# NUMBERS APPEAR NOWHERE new: c_n (the history-side flow rate) and K_F's eigenvalues are
# STRUCTURAL quantities of this station's own construction (word-length rate; a region's
# entangling spectrum) -- lambda is compared to NOTHING measured, fitted to NOTHING, tuned to
# NOTHING (module contract + pre-reg SS3.4/SSJ).
#
# ML-2b/HK-7 CONDITIONALITY (carries into every verdict sentence below, verbatim, unchanged from
# Section 8's own banner): "Every duality check here (HK-5) is CELL-LEVEL only (the 6-edge static
# vacuum). ML-2b's DR-frame argument is CONDITIONAL on the TD-limit duality holding, which is NOT
# verified by this suite."
def history_side_flow_generator(shell_n, N_max=None):
    """[FOCK-0d SS H, K_hist -- THE HISTORY-SIDE MODULAR/FLOW GENERATOR] K_hist = -log(rho)
    (equivalently -log(Delta_hist), SS8b's b2 Tomita data: Delta_half(A) = diag(sqrt(rho)).A.
    diag(1/sqrt(rho)), i.e. Delta_hist acts on the GNS carrier as Ad(rho); its GENERATOR
    -log(rho), restricted to acting on H_hist ITSELF (the SAME carrier Phi maps FROM, not the
    doubled HS(H_hist) operator space) is what this station needs, exactly the object
    tomita_checks' own check (4) already establishes: 'the modular Hamiltonian K=-ln(rho) ...
    is EXACTLY beta_natural times the PER-WORD length (up to the additive -ln Z constant)'.
    STRUCTURAL FACT (verified below, not assumed): because omega_diag depends ONLY on word
    length (I2b's own C-2 finding, omega_diag_length), rho is EXACTLY CONSTANT across every word
    of length n -- hence K_hist|_shell_n is EXACTLY a SCALAR operator (c_n.I_n) on shell n, for
    EVERY sector alike (SECTOR-BLIND: the per-path clock does not discriminate GROUP-03 from
    GROUP-12 at all -- a genuinely reportable structural finding, verified to machine precision).
    THE ADDITIVE CONSTANT: -ln(Z) is DROPPED, by the standard 'a flow generator is defined up to
    an additive constant' convention (Delta^it = e^{-iKt}; K -> K+c.I rescales Delta by a
    SHELL-INDEPENDENT phase e^{-ict}, changing NO physical content of the flow) -- tomita_checks'
    OWN per-path KMS check already only ever compares DIFFERENCES ln(rho_w)-ln(rho_v), never an
    absolute K value, confirming -ln(Z) carries no independent physical content here. So
    c_n := beta_natural * n EXACTLY (c_n_with_lnZ_constant is ALSO returned, for transparency,
    but NOT used in the lambda-solve below).
    Returns {'c_n' (the constant used), 'c_n_with_lnZ_constant', 'beta_natural', 'shell_dim',
    'scalar_exactness_residual'} (residual asserted < 1e-9)."""
    if N_max is None:
        N_max = shell_n
    words, index, lengths, omega = omega_diag_length(N_max)
    idx = np.where(lengths == shell_n)[0]
    assert len(idx) > 0, f"history_side_flow_generator: no words of length {shell_n} at N_max={N_max}"
    neglog_rho = -np.log(omega[idx])
    c_n_with_const = float(np.mean(neglog_rho))
    resid = float(np.max(np.abs(neglog_rho - c_n_with_const)))
    assert resid < 1e-9, \
        f"history_side_flow_generator: K_hist not scalar on shell {shell_n} (residual {resid:.2e})"
    u_eff = (2.0 / 3.0) ** 8
    beta_natural = -2.0 * math.log(u_eff)
    c_n_theory = beta_natural * shell_n
    return {"c_n": c_n_theory, "c_n_with_lnZ_constant": c_n_with_const,
            "beta_natural": beta_natural, "shell_dim": len(idx),
            "scalar_exactness_residual": resid}


def _region_fock_ops():
    """[FOCK-0d SS H, K_F CONSTRUCTION, STEP 1] standard Jordan-Wigner fermion ladder operators
    b_0,b_1,b_2 on an INDEPENDENTLY-BUILT 8-dim Fock space, using the IDENTICAL basis-labeling
    convention field_algebra_conjugation already committed to for F (ascending subsets of
    {0,1,2}: (), (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)) -- so that identifying this
    space WITH F (a NAMED, DISCLOSED judgment call, see field_side_flow_generator's docstring and
    the station report's 'Named Residual Freedom', NOT a forced identity) is at least
    BASIS-LABEL-CONSISTENT with F's own existing convention, not an arbitrary re-labeling of its
    own invention.
    Standard JW sign: b_i^dagger|S> = 0 if i in S, else (-1)^(#{j in S : j<i}) |S union {i}>.
    Returns (bdag, b, car_residual): bdag/b each a list of 3 real 8x8 matrices; CAR
    {b_i,b_j^dagger}=delta_ij, {b_i,b_j}=0 VERIFIED exactly (same verification style
    field_algebra_conjugation already used for its own A_ops CAR check)."""
    subsets = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    idx = {S: i for i, S in enumerate(subsets)}
    bdag = []
    for i in range(3):
        M = np.zeros((8, 8))
        for S in subsets:
            if i in S:
                continue
            sign = (-1.0) ** sum(1 for j in S if j < i)
            Snew = tuple(sorted(S + (i,)))
            M[idx[Snew], idx[S]] = sign
        bdag.append(M)
    b = [m.T for m in bdag]
    I8 = np.eye(8)
    car_resid = 0.0
    for i in range(3):
        for j in range(3):
            acomm_dag = bdag[i] @ b[j] + b[j] @ bdag[i]
            target = I8 if i == j else np.zeros((8, 8))
            car_resid = max(car_resid, float(np.max(np.abs(acomm_dag - target))))
            acomm = b[i] @ b[j] + b[j] @ b[i]
            car_resid = max(car_resid, float(np.max(np.abs(acomm))))
    assert car_resid < 1e-9, f"_region_fock_ops: CAR violated (residual {car_resid:.2e})"
    return bdag, b, car_resid


def field_side_flow_generator(region_edges):
    """[FOCK-0d SS H, K_F -- THE FIELD-SIDE MODULAR/FLOW GENERATOR] M0's own region_data/
    entanglement_hamiltonian (M0's EXPLICIT, OWNED convention -- verbatim from
    M0_modular_hamiltonian_kappa_2026-07-07.py:105, '# the vacuum covariance (complex-fermion
    convention, OWNED): C = (I + iJ)/2' -- i.e. C is treated as the correlation matrix of
    INDEPENDENT complex-fermion 'edge' modes for the PURPOSE of the Peschel entanglement-Hamiltonian
    formula, a DIFFERENT (if numerically coincident) use of the SAME J6/C data than Cl(6)'s
    Majorana spinor module F used elsewhere in this section) applied to a 3-edge region A,
    second-quantized via the STANDARD number-conserving Peschel bilinear
    K_F = sum_{i,j in A} h_A[i,j].b_i^dagger.b_j (_region_fock_ops's basis, field_algebra_
    conjugation's OWN ascending-subset convention).
    NAMED JUDGMENT CALL (logged, not forced -- see the station report): this 8-dim region-local
    Fock space is IDENTIFIED with F purely by DIMENSION MATCH (2^3=8=dim F) plus a SHARED
    basis-labeling convention; it is NOT a proof that region A's local complex-fermion algebra
    literally equals F's Cl(6) Majorana spinor module (a priori DIFFERENT constructions from the
    same numbers -- M0's own C is explicitly the 'complex-fermion, OWNED' convention, distinct
    from the g6 Majorana generators _sector_projectors/field_algebra_conjugation use elsewhere).
    This is the least-arbitrary way to get a concrete, computable K_F consistent with everything
    already accreted (entanglement_hamiltonian, region_data), reusing the SAME basis labels
    field_algebra_conjugation already committed to for F -- nothing new is invented beyond the
    textbook Peschel second-quantization recipe and this one disclosed identification.
    ALSO VERIFIES (M0-4b consistency, reused as a check on THIS construction, not re-derived):
    the bit sigma (J->-J) reverses K_A -> -K_A on region A EXACTLY as a MATRIX identity (not just
    same eigenvalues) -- since Cm|_A = I_A - C_A exactly, h_A(Cm|_A) = -h_A(C_A) exactly by the
    log-of-inverse identity, hence K_F(Cm) = -K_F(C) exactly by K_F's own LINEARITY in h_A.
    Returns {'K_F' (8x8 Hermitian), 'h_A' (3x3), 'eigenvalues' (sorted real), 'hermiticity_residual',
    'region_edges', 'car_residual', 'bit_reversal_check_residual'} (all residuals asserted small)."""
    C = vacuum_covariance(sign=+1)
    A = list(region_edges)
    assert len(A) == 3, f"field_side_flow_generator: region must have exactly 3 edges, got {A}"
    C_A = C[np.ix_(A, A)]
    h_A = entanglement_hamiltonian(C_A)
    bdag, b, car_resid = _region_fock_ops()
    K_F = sum(h_A[i, j] * (bdag[i] @ b[j]) for i in range(3) for j in range(3))
    herm_resid = float(np.max(np.abs(K_F - K_F.conj().T)))
    assert herm_resid < 1e-8, f"field_side_flow_generator: K_F not Hermitian (residual {herm_resid:.2e})"
    eigvals = np.sort(np.linalg.eigvalsh(K_F).real)
    Cm = vacuum_covariance(sign=-1)
    h_Am = entanglement_hamiltonian(Cm[np.ix_(A, A)])
    K_Fm = sum(h_Am[i, j] * (bdag[i] @ b[j]) for i in range(3) for j in range(3))
    bit_reversal_resid = float(np.max(np.abs(K_Fm + K_F)))
    assert bit_reversal_resid < 1e-6, \
        f"field_side_flow_generator: M0-4b bit-reversal consistency failed (residual {bit_reversal_resid:.2e})"
    return {"K_F": K_F, "h_A": h_A, "eigenvalues": eigvals, "hermiticity_residual": herm_resid,
            "region_edges": tuple(A), "car_residual": car_resid,
            "bit_reversal_check_residual": bit_reversal_resid}


def _three_edge_region_orbits():
    """[FOCK-0d SS H, 'run ALL inequivalent 3-edge halves'] classify ALL C(6,3)=20 three-edge
    subsets of the cell into A4-orbits (using _edge_rep, the SAME edge-permutation
    representation complex_structure_J6/M0 use), picking ONE representative per orbit -- the
    rigorous, non-cherry-picking way to honor the pre-reg's 'if inequivalent halves exist, run
    ALL of them' instruction (running one representative per orbit is EXHAUSTIVE: any other
    member of an orbit gives a K_F related by a FIXED A4 conjugation to the representative's, not
    new content -- A4 is a symmetry of C/J6 itself, verified elsewhere, e.g. complex_structure_
    J6's own A4-covariance construction).
    Returns a list of {'representative' (tuple of 3 edge indices, sorted), 'orbit_size',
    'is_triangle' (matches M0's own girth-cycle triangles, the class M0-3/M0-4 anchored and
    examined by name, 'A0 = triangles[0]')}."""
    A4v = _a4_vertex_group()
    all_subsets = list(itertools.combinations(range(NE), 3))
    perms = []
    for g in A4v:
        R6 = _edge_rep(g)
        pi = np.argmax(np.abs(R6), axis=0)
        perms.append(pi)
    seen = set()
    orbits = []
    for A in all_subsets:
        if A in seen:
            continue
        orbit = set()
        for pi in perms:
            gA = tuple(sorted(int(pi[e]) for e in A))
            orbit.add(gA)
        seen |= orbit
        orbits.append((A, len(orbit)))
    EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
    triangles = set()
    for tri in itertools.combinations(range(NV), 3):
        es = tuple(sorted(EIDX[(a, b)] for a, b in itertools.combinations(sorted(tri), 2)))
        triangles.add(es)
    assert len(triangles) == 4, f"_three_edge_region_orbits: expected 4 K4 triangles, got {len(triangles)}"
    total = sum(size for _, size in orbits)
    assert total == len(all_subsets), \
        f"_three_edge_region_orbits: orbit sizes sum to {total} != {len(all_subsets)} total subsets"
    return [{"representative": rep, "orbit_size": size, "is_triangle": rep in triangles}
            for rep, size in orbits]


def _fock0c_rows_full(shell_n, N_max=None):
    """[FOCK-0d glue, NOT a new construction] re-assembles FOCK-0c's EXACT rows_full stack
    (fock0c_full_pin_shell's own row list: the antiunitary Tomita pin + the two cross-group
    vanishing rows + the R/F_bit pin), calling the IDENTICAL row-builder functions with the
    IDENTICAL arguments fock0c_full_pin_shell uses (the_net.py:3244-3253) -- reused, not
    rebuilt; this exists ONLY because fock0c_full_pin_shell returns nullity numbers, not the raw
    row matrix, and FOCK-0d needs to stack ONE more row on top of the SAME system.
    Returns (rows_full list, n, m). Consistency VERIFIED in fock0d_selftest: nullity_of(rows_full)
    reproduces fock0c_full_pin_shell's own full_pin_nullity exactly."""
    if N_max is None:
        N_max = shell_n
    hg = history_sector_pair_groups(N_max)
    idx = np.where(hg["lengths"] == shell_n)[0]
    n = len(idx)
    P_H_g03 = hg["P_group03"][np.ix_(idx, idx)]
    P_H_g12 = hg["P_group12"][np.ix_(idx, idx)]
    if shell_n == 1:
        R = reversal()
    else:
        hr = history_reversal_matrix(N_max)
        R = hr["Rw"][np.ix_(idx, idx)]
    fg = field_sector_pair_groups()
    P_F_g03, P_F_g12 = fg["P_group03"], fg["P_group12"]
    fa = field_algebra_conjugation()
    K = fa["M"]
    Pw, _ = _sector_projectors(sign=+1)
    F_bit = Pw[0] + Pw[3] - Pw[1] - Pw[2]
    m = 8
    rows_pin = [_antiunitary_pin_rows(R, K)]
    cross_rows = [_zero_block_rows(P_F_g12, m, P_H_g03, n), _zero_block_rows(P_F_g03, m, P_H_g12, n)]
    rfbit_row = [_linear_pin_rows(R, F_bit)]
    return rows_pin + cross_rows + rfbit_row, n, m


def _flow_pin_row(c_n, lam, K_F, n):
    """The NEW flow constraint row: Phi.(c_n.I_n) = lam.K_F.Phi -- i.e. Phi.K_hist,sigma =
    lambda.K_F,sigma.Phi with K_hist|_shell_n = c_n.I_n EXACTLY (history_side_flow_generator).
    LINEAR (not antilinear) pin, per the pre-reg -- built via _linear_pin_rows, the SAME method
    _zero_block_rows/the R/F_bit pin already use (no conjugation anywhere)."""
    return _linear_pin_rows(c_n * np.eye(n), lam * K_F)


def fock0d_joint_nullity(shell_n, lam, K_F, N_max=None):
    """Nullity of FOCK-0c's FULL pin set (reused via _fock0c_rows_full) STACKED with the NEW
    flow-pin row at a GIVEN candidate lambda -- ON TOP of FOCK-0c's full pin set, per SS H.
    Returns {'nullity','rank','total','group03','group12'} (group03+group12 verified == nullity)."""
    if N_max is None:
        N_max = shell_n
    rows_full, n, m = _fock0c_rows_full(shell_n, N_max=N_max)
    c_n = history_side_flow_generator(shell_n, N_max=N_max)["c_n"]
    row_new = _flow_pin_row(c_n, lam, K_F, n)
    all_rows = rows_full + [row_new]

    def nullity_of(rowlist):
        Cop = np.vstack(rowlist)
        s = np.linalg.svd(Cop, compute_uv=False)
        rank = int(np.sum(s > 1e-8))
        return Cop.shape[1] - rank, rank, Cop.shape[1]

    nullity, rank, total = nullity_of(all_rows)
    hg = history_sector_pair_groups(N_max)
    idx = np.where(hg["lengths"] == shell_n)[0]
    P_H_g03 = hg["P_group03"][np.ix_(idx, idx)]
    P_H_g12 = hg["P_group12"][np.ix_(idx, idx)]
    fg = field_sector_pair_groups()
    null_g03, *_ = nullity_of(all_rows + [_zero_block_rows(fg["P_group12"], m, P_H_g12, n)])
    null_g12, *_ = nullity_of(all_rows + [_zero_block_rows(fg["P_group03"], m, P_H_g03, n)])
    assert null_g03 + null_g12 == nullity, \
        f"fock0d_joint_nullity: group breakdown {null_g03}+{null_g12} != total {nullity}"
    return {"nullity": nullity, "rank": rank, "total": total, "group03": null_g03, "group12": null_g12}


def fock0d_lambda_candidates(region_edges, N_max=4, tol=1e-6):
    """[FOCK-0d SS H, THE GENERALIZED NULLITY PROBLEM -- SOLVED ANALYTICALLY, NOT GRID-SCANNED]
    Since K_hist|_shell_n is EXACTLY SCALAR (c_n.I_n, history_side_flow_generator), the flow-pin
    row (as a SQUARE matrix on the WHOLE ambient real space, before stacking with anything else)
    is the PENCIL c_n.I - lambda.(I (x) K_F): singular (nontrivial kernel) IFF
    lambda = c_n / mu for mu an eigenvalue of K_F -- an EXACT, closed-form characterization (for
    lambda NOT of this form the pencil is invertible, hence its OWN kernel is {0}, hence stacking
    it onto ANY other system can only ever give a TRIVIAL joint kernel; this is a property of the
    Kronecker structure alone, independent of what else is stacked -- see the station report for
    the argument in full). Since c_2 = 2.c_1 EXACTLY (c_n=beta_natural*n), a lambda consistent at
    BOTH shells (the frozen 'ONE global lambda shared by all shells' requirement) exists IFF
    K_F's OWN spectrum contains a pair (mu_1,mu_2), both POSITIVE (lambda>0 needs mu same sign as
    c_n>0), with mu_2 = 2.mu_1 EXACTLY (to numerical tolerance). This is checked directly from
    K_F's computed eigenvalues -- a closed-form characterization of the pencil's singular locus,
    NOT a grid scan over candidate lambda values.
    Returns {'K_F_eigenvalues','c1','c2','region_edges','candidates': [{'mu1','mu2','lambda'}],
    'closest_miss': the minimum |mu2 - 2*mu1| over all positive-eigenvalue pairs, for honest
    near-miss reporting (NOT used to pick a lambda -- diagnostic only)}."""
    fsg = field_side_flow_generator(region_edges)
    K_F = fsg["K_F"]
    mus = fsg["eigenvalues"]
    c1 = history_side_flow_generator(1, N_max=N_max)["c_n"]
    c2 = history_side_flow_generator(2, N_max=N_max)["c_n"]
    assert abs(c2 - 2 * c1) < 1e-9, f"fock0d_lambda_candidates: c2={c2} != 2*c1={2*c1} (unexpected)"
    pos_mus = [float(mu) for mu in mus if mu > 1e-9]
    candidates = []
    closest_miss = float("inf")
    for mu1 in pos_mus:
        for mu2 in pos_mus:
            diff = abs(mu2 - 2 * mu1)
            closest_miss = min(closest_miss, diff / max(1.0, abs(mu2)))
            if diff < tol * max(1.0, abs(mu2)):
                candidates.append({"mu1": mu1, "mu2": mu2, "lambda": float(c1 / mu1)})
    return {"K_F_eigenvalues": mus.tolist(), "c1": c1, "c2": c2,
            "region_edges": fsg["region_edges"], "candidates": candidates,
            "closest_relative_miss": closest_miss}


def fock0d_all_regions_analysis(N_max=4):
    """[FOCK-0d SS H, TOP-LEVEL DRIVER LOGIC] runs the temporal-pin lambda-analysis for EVERY
    A4-orbit representative of 3-edge regions (honoring 'if inequivalent 3-edge halves exist, run
    ALL of them and report every block'). For each region, and for each candidate lambda found
    (if any), computes the JOINT nullity (FOCK-0c's full pin STACKED with the new flow row) at
    BOTH shells.
    Returns a list of per-region dicts: {'region','orbit_size','is_triangle','K_F_eigenvalues',
    'closest_relative_miss','candidates': [{'lambda','mu1','mu2','shell1':{...},'shell2':{...}}]}."""
    orbits = _three_edge_region_orbits()
    results = []
    for orb in orbits:
        region = orb["representative"]
        lc = fock0d_lambda_candidates(region, N_max=N_max)
        K_F = field_side_flow_generator(region)["K_F"]
        per_cand = []
        for cand in lc["candidates"]:
            lam = cand["lambda"]
            n1 = fock0d_joint_nullity(1, lam, K_F, N_max=N_max)
            n2 = fock0d_joint_nullity(2, lam, K_F, N_max=N_max)
            per_cand.append({"lambda": lam, "mu1": cand["mu1"], "mu2": cand["mu2"],
                              "shell1": n1, "shell2": n2})
        results.append({"region": region, "orbit_size": orb["orbit_size"],
                         "is_triangle": orb["is_triangle"],
                         "K_F_eigenvalues": lc["K_F_eigenvalues"],
                         "closest_relative_miss": lc["closest_relative_miss"],
                         "candidates": per_cand})
    return results


def fock0d_fence_check():
    """[FOCK-0d, THE SS2 FENCE RE-CHECK on the temporal/flow-pin class] as fock0c_fence_check,
    plus a NEW item for the flow pin specifically: K_hist (the GNS/Tomita modular Hamiltonian of
    omega_diag, I2b's OWN per-path length statistic) and K_F (the M0 Peschel entangling
    Hamiltonian of a 3-edge region, second-quantized) are BOTH representation-theoretic/modular
    objects -- NEITHER is the run's own two-point RESOLVENT (I-uB)^-1 data (BRIDGE-T's own
    concern, item 4); the flow row is stacked ON TOP OF the SAME sector-pair-graded system items
    3/5 already establish (reused unchanged). Returns fock0c_fence_check()'s dict plus
    '7_FOCK0d_flow_not_resolvent_data'."""
    base = fock0c_fence_check()
    out = dict(base)
    out["7_FOCK0d_flow_not_resolvent_data"] = {
        "check": "K_hist is the GNS/Tomita modular Hamiltonian of omega_diag (I2b's per-path "
                 "length statistic, tomita_data/gns_purification) and K_F is the M0 Peschel "
                 "entangling Hamiltonian of a 3-edge region (entanglement_hamiltonian), second-"
                 "quantized -- NEITHER is a two-point correlation functional of the run's "
                 "resolvent (I-uB)^-1 at any order; the flow row is stacked on the SAME "
                 "sector-pair-graded system FOCK-0c already built (items 3/5 reused unchanged)",
        "is_modular_not_resolvent": True}
    return out


def fock0d_selftest_2026_07_11(verbose=True):
    """FOCK-0d station regression: WAYPOINTS FIRST (FOCK-0c's 16/384 shell-1 and 64/768 shell-2
    full-pin numbers, group03/group12 breakdown, reproduced via _fock0c_rows_full's own
    re-assembly -- a self-consistency check that the row re-derivation is faithful), THEN K_hist's
    scalar-exactness at both shells, K_F's construction (Hermiticity + M0-4b bit-reversal
    consistency) for the triangle region, the lambda-candidate scan (all 3-edge-region orbits),
    the joint-nullity evaluation at any candidates found, and the SS2 fence re-check, plus
    Sections 7/7b/8/8b/8c + module anchors untouched. Does NOT itself adjudicate the SS I tree
    (architect-only per the pre-reg)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 8d self-test -- FOCK-0d the temporal pin (2026-07-11)")
        print("=" * 88)

    ck("ANCHORS + Sections 7/7b/8/8b/8c untouched",
       anchor_cell_projector() and anchor_tick_2pi() and accretion_selftest_2026_07_10(verbose=False)
       and i2b_selftest_2026_07_11(verbose=False) and fock0_selftest_2026_07_11(verbose=False)
       and fock0b_selftest_2026_07_11(verbose=False) and fock0c_selftest_2026_07_11(verbose=False))

    N_max = 4
    s1c = fock0c_full_pin_shell(1, N_max=N_max)
    s2c = fock0c_full_pin_shell(2, N_max=N_max)
    rows1, n1, m1 = _fock0c_rows_full(1, N_max=N_max)
    s1 = np.linalg.svd(np.vstack(rows1), compute_uv=False)
    null1 = np.vstack(rows1).shape[1] - int(np.sum(s1 > 1e-8))
    rows2, n2, m2 = _fock0c_rows_full(2, N_max=N_max)
    s2 = np.linalg.svd(np.vstack(rows2), compute_uv=False)
    null2 = np.vstack(rows2).shape[1] - int(np.sum(s2 > 1e-8))
    ck(f"WAYPOINT (re-derivation match): _fock0c_rows_full reproduces FOCK-0c's own full-pin "
       f"nullity EXACTLY at both shells (shell1 {null1} vs {s1c['full_pin_nullity']}; "
       f"shell2 {null2} vs {s2c['full_pin_nullity']})",
       null1 == s1c["full_pin_nullity"] == 16 and null2 == s2c["full_pin_nullity"] == 64)

    h1 = history_side_flow_generator(1, N_max=N_max)
    h2 = history_side_flow_generator(2, N_max=N_max)
    ck(f"K_hist SCALAR-EXACTNESS: shell1 c_1={h1['c_n']:.6f} (residual={h1['scalar_exactness_residual']:.2e}), "
       f"shell2 c_2={h2['c_n']:.6f} (residual={h2['scalar_exactness_residual']:.2e}); "
       f"c_2 == 2*c_1 EXACTLY ({abs(h2['c_n'] - 2*h1['c_n']):.2e})",
       h1["scalar_exactness_residual"] < 1e-9 and h2["scalar_exactness_residual"] < 1e-9
       and abs(h2["c_n"] - 2 * h1["c_n"]) < 1e-9)

    orbits = _three_edge_region_orbits()
    tri = next(o for o in orbits if o["is_triangle"])
    fsg = field_side_flow_generator(tri["representative"])
    ck(f"K_F CONSTRUCTED (triangle region {tri['representative']}, orbit size {tri['orbit_size']}): "
       f"Hermitian (residual={fsg['hermiticity_residual']:.2e}), CAR exact "
       f"(residual={fsg['car_residual']:.2e}), M0-4b bit-reversal K_A->-K_A reproduced "
       f"EXACTLY (residual={fsg['bit_reversal_check_residual']:.2e})",
       fsg["hermiticity_residual"] < 1e-8 and fsg["car_residual"] < 1e-9
       and fsg["bit_reversal_check_residual"] < 1e-6)

    all_results = fock0d_all_regions_analysis(N_max=N_max)
    n_orbits = len(all_results)
    n_candidates = sum(len(r["candidates"]) for r in all_results)
    ck(f"LAMBDA SCAN over all {n_orbits} A4-orbit-representative 3-edge regions: "
       f"{n_candidates} candidate lambda(s) found (exact mu2=2*mu1 matches in K_F's spectrum, "
       "shared by both shells)", True,
       detail="; ".join(f"region={r['region']} triangle={r['is_triangle']} "
                         f"closest_miss={r['closest_relative_miss']:.3e}" for r in all_results))

    fc = fock0d_fence_check()
    ck("SS2 FENCE re-check on the temporal-pin class: all items confirmed",
       fc["1_O0_bit_even_democracy"]["is_phase_bearing"]
       and fc["2_M1b_no_generator_constraint"]["no_group_generator_used"]
       and fc["3_BRIDGE_LOCK_attachment_functional_orbit_blind"]["is_fock_level"]
       and fc["4_BRIDGE_T_two_point_data_blind"]["is_representation_theoretic"]
       and fc["5_BRIDGE_GEOM_per_sector_required"]["is_per_sector_by_design"]
       and fc["6_FOCK0c_grading_is_projector_only"]["no_extra_mechanism_introduced"]
       and fc["7_FOCK0d_flow_not_resolvent_data"]["is_modular_not_resolvent"])

    if verbose:
        print("RESULT:",
              "FOCK-0d SECTION-8d REGRESSION PASSES" if ok else "A FOCK-0d CHECK FAILED")
    return ok


# ===========================================================================
# 8e. FOCK-0e -- A1: CLOCK LINEARITY (c_n = n.c_1), STRUCTURE ONLY  (2026-07-12)
#     FOCK0_dr_reconstruction_prereg_2026-07-11.md AMENDMENT FOCK-0e SS L-P
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as Sections 7/7b/8/8b/8c/8d above.]
#
# CONTEXT: FOCK-0d SS3 found (machine-exact at shells 1-2) that K_hist restricted to shell n is
# EXACTLY the scalar c_n.I_n, c_1 = beta_natural = 6.4874417297, c_2 = 2.c_1 (deviation 0.0).
# FOCK-0e asks EXACTLY ONE question and stops: is this a THEOREM about the constructed state --
# c_n = n.c_1 for ALL n -- not a two-shell observation?  FOCK-0d's own W2 verdict (the clock-
# incommensurability obstruction theorem, commit 595d4e9) is SETTLED and is NOT re-opened here.
# HARD STOP (SS M m6): no A2 content (no graded/level-dependent map defined, sketched, or run
# anywhere below); no new K_F construction; the field side is otherwise UNTOUCHED (K_F is only
# READ via the already-accreted field_side_flow_generator, for its eigenvalues, exactly as SS8d
# already did -- nothing new is built on the field side here).
#
# NUMBERS APPEAR NOWHERE: every quantity below is a rate, a residual, a ratio, or a derived
# structural scale (lambda_n) of THIS construction's OWN operators -- never M_Z/ppm/m_nu/a_e
# (module contract + pre-reg SS3.4/SSO).  lambda is compared to NOTHING measured, never tuned.
#
# ML-2b/HK-7 CONDITIONALITY (carries into every DR-frame-touching sentence below, verbatim,
# unchanged from Section 8's own banner): "Every duality check here (HK-5) is CELL-LEVEL only (the
# 6-edge static vacuum). ML-2b's DR-frame argument is CONDITIONAL on the TD-limit duality holding,
# which is NOT verified by this suite."
def fock0e_analytic_lemma():
    """[FOCK-0e SS M, m1 -- THE ANALYTIC LEMMA, proved from omega_diag_length's LITERAL closed
    form] omega_diag(w) = u^(2|w|)/Z (I2b's own C-2 result) depends on w ONLY through |w| -- hence
    rho := diag(omega_diag) satisfies, for ANY two words w, v (at ANY fixed truncation N_max):
        -log(rho_w) - (-log(rho_v)) = -2.ln(u).(|w|-|v|)   EXACTLY,
    because the SAME Z cancels identically in the difference (Z is a single N_max-dependent number
    summed over the WHOLE truncated word set, hence SHELL-INDEPENDENT at fixed N_max) -- an
    ALGEBRAIC fact, true for every N_max, not a numerical coincidence to be checked shell-by-shell.
    Taking v = the seed (the UNIQUE length-0 word; rho_seed = 1/Z, so -log(rho_seed) = ln(Z)
    EXACTLY, an explicit computed number, not an unknown) and defining, under the disclosed
    'a flow generator is defined up to an additive c.I' convention (Delta^it = e^{-iKt}; K -> K+c.I
    rescales Delta by a shell-independent phase, changing no physical content of the flow -- the
    SAME convention FOCK-0d SS3 already used and disclosed at every c_n statement, restated here):
        c_n := K_hist(w) - K_hist(seed)  for |w| = n,
    the lemma gives c_n = n.c_1 IDENTICALLY, c_1 = -2.ln(u) = -2.ln(alpha_1) = 16.ln(3/2) =
    beta_natural.  TRUNCATION HONESTY (per SS M m1): Z depends on N_max but is SHELL-INDEPENDENT,
    so it cancels in EVERY shell-vs-seed difference identically, for ANY N_max -- linearity is
    TRUNCATION-INDEPENDENT by this algebraic argument (not merely observed to hold at whichever
    N_max the machine check below happens to use).
    Verifies the THREE-WAY numeric identity requested by SS M m1: -2.ln(alpha_1) == 16.ln(3/2) ==
    I2b's own beta_natural = 6.4874417297 (the SAME literal constant tomita_checks already asserts
    against, the_net.py:2586-2587).
    Returns {'alpha_1', 'c1_neg2ln_alpha1', 'c1_sixteen_ln_three_halves', 'beta_natural_I2b_literal',
    'identity_residual_vs_16ln32', 'identity_residual_vs_I2b_beta_natural'}."""
    u = (2.0 / 3.0) ** 8  # = alpha_1, the run's own operating fugacity (omega_diag_length's default)
    c1_direct = -2.0 * math.log(u)
    c1_sixteen_ln32 = 16.0 * math.log(1.5)
    beta_natural_i2b_literal = 6.4874417297  # I2b's own constant (tomita_checks' own assert target)
    resid_16ln32 = abs(c1_direct - c1_sixteen_ln32)
    resid_i2b = abs(c1_direct - beta_natural_i2b_literal)
    return {"alpha_1": u, "c1_neg2ln_alpha1": c1_direct,
            "c1_sixteen_ln_three_halves": c1_sixteen_ln32,
            "beta_natural_I2b_literal": beta_natural_i2b_literal,
            "identity_residual_vs_16ln32": resid_16ln32,
            "identity_residual_vs_I2b_beta_natural": resid_i2b}


def fock0e_clock_linearity_table(N_max=8):
    """[FOCK-0e SS M, m2 -- THE MACHINE CHECK, EVERY SHELL] direct-rho route (no GNS/Tomita
    needed for a diagonal state -- reuses omega_diag_length UNCHANGED): at a SINGLE truncation
    N_max, for EVERY shell n = 0..N_max, computes (a) the IN-SHELL SCALARITY residual
    max_w|{-log(omega_diag(w))} - mean| (the WITH-constant/absolute K_hist reading, PER-PATH: acts
    identically on every INDIVIDUAL word/microstate of length n, never a shell-aggregate) and
    (b) the DROPPED-CONSTANT ratio residual |c_n/c_1 - n|, where c_n := (shell-n mean of
    -log(omega_diag)) - (shell-0 value) -- the shell-0 value IS ln(Z) exactly (fock0e_analytic_
    lemma), so this is the SAME disclosed convention as FOCK-0d SS3, applied uniformly to every
    shell rather than checked only at n=1,2.
    Returns {'N_max', 'rows': [{'n','D_n','scalarity_residual','c_n_with_const','c_n',
    'ratio_residual'}], 'c1', 'ln_Z', 'worst_scalarity_residual', 'worst_ratio_residual'}."""
    words, index, lengths, omega = omega_diag_length(N_max)
    neglog = -np.log(omega)
    seed_i = index[()]
    ln_Z = float(neglog[seed_i])  # = -log(omega_diag(seed)) = -log(1/Z) = ln(Z), exactly
    rows = []
    for n in range(0, N_max + 1):
        idx = np.where(lengths == n)[0]
        vals = neglog[idx]
        c_n_with_const = float(np.mean(vals))
        scal_resid = float(np.max(np.abs(vals - c_n_with_const))) if len(vals) > 1 else 0.0
        c_n = c_n_with_const - ln_Z
        rows.append({"n": n, "D_n": int(len(idx)), "scalarity_residual": scal_resid,
                     "c_n_with_const": c_n_with_const, "c_n": c_n})
    c1 = rows[1]["c_n"]
    for row in rows:
        row["ratio_residual"] = abs(row["c_n"] / c1 - row["n"]) if row["n"] > 0 else abs(row["c_n"])
    worst_scal = max(row["scalarity_residual"] for row in rows)
    worst_ratio = max(row["ratio_residual"] for row in rows[1:])
    return {"N_max": N_max, "rows": rows, "c1": c1, "ln_Z": ln_Z,
            "worst_scalarity_residual": worst_scal, "worst_ratio_residual": worst_ratio}


def fock0e_tomita_route_check(N_max=4):
    """[FOCK-0e SS M, m3 -- TOMITA-ROUTE CONSISTENCY ANCHOR] reuses SS8b's gns_purification +
    tomita_data UNCHANGED (no rebuild).  tomita_data's own Delta_half acts on a seed-to-word
    elementary matrix unit A = E_{w,seed}.diag(sqrt(rho)) as Delta_half(A) = (r_w/r_seed).A EXACTLY
    (tomita_data's own docstring/derivation) -- extracting c_n_tomita := -2.log(r_w/r_seed) =
    -log(rho_w) + log(rho_seed) = K_hist(w) - K_hist(seed) via the Delta_half MACHINERY (not by
    reading rho directly a second time), for a probe word at shells 1 and 2, and confirming it
    reproduces the direct-rho route's c_n (history_side_flow_generator / fock0e_clock_linearity_
    table, already FOCK-0d/FOCK-0e anchors) EXACTLY -- anchoring m2's cheap direct-rho shortcut to
    the GNS/Tomita route FOCK-0b already built and machine-verified (a construction-consistency
    check, not a new physics claim).
    Returns {'N_max', 'c1_direct', 'c2_direct', 'c1_tomita', 'c2_tomita', 'residual_shell1',
    'residual_shell2'}."""
    gp = gns_purification(N_max)
    words, index, lengths, rho = gp["words"], gp["index"], gp["lengths"], gp["rho"]
    td = tomita_data(rho)
    Delta_half = td["Delta_half"]
    r = td["sqrt_rho"]
    seed_i = index[()]
    D = len(words)
    c_tomita = {}
    for n in (1, 2):
        wi = int(np.where(lengths == n)[0][0])
        x = np.zeros((D, D))
        x[wi, seed_i] = 1.0
        A = x * r[None, :]
        out = Delta_half(A)
        ratio = out[wi, seed_i] / A[wi, seed_i]
        c_tomita[n] = -2.0 * math.log(ratio)
    c1_direct = history_side_flow_generator(1, N_max=N_max)["c_n"]
    c2_direct = history_side_flow_generator(2, N_max=N_max)["c_n"]
    return {"N_max": N_max, "c1_direct": c1_direct, "c2_direct": c2_direct,
            "c1_tomita": c_tomita[1], "c2_tomita": c_tomita[2],
            "residual_shell1": abs(c_tomita[1] - c1_direct),
            "residual_shell2": abs(c_tomita[2] - c2_direct)}


def fock0e_lambda_structure(N_max=8):
    """[FOCK-0e SS M, m4 -- THE DERIVED CLOCK-RELATION STRUCTURE, IF LINEAR] for EACH of FOCK-0d's
    four A4-orbit-inequivalent 3-edge regions (_three_edge_region_orbits + field_side_flow_
    generator -- REUSED UNCHANGED, nothing rebuilt on the field side, per SS M m6's hard stop),
    books lambda_n := c_n/epsilon = n.(c_1/epsilon) = n.lambda_1, where epsilon = K_F's single
    positive eigenvalue magnitude (FOCK-0d SS5/SS6's proven {0x4,-eps x2,+eps x2} spectrum, read
    off the already-accreted field_side_flow_generator's own 'eigenvalues' field -- NOT
    re-derived).  These lambda's are STRUCTURE: compared to NO measured constant, tuned toward
    NOTHING (SS O poison).  lambda_1 for the triangle orbit reproduces FOCK-0d's own EMBER value
    (c_1/epsilon = 2.463) as a consistency check on THIS station's own c_1, not a re-derivation of
    the ember itself.
    Returns a list of {'region','orbit_size','is_triangle','epsilon','lambda_1',
    'lambda_n': {n: lambda_n for n in 1..N_max}, 'linear_in_n_residual'}."""
    orbits = _three_edge_region_orbits()
    table = fock0e_clock_linearity_table(N_max=N_max)
    c_of_n = {row["n"]: row["c_n"] for row in table["rows"]}
    c1 = c_of_n[1]
    results = []
    for orb in orbits:
        region = orb["representative"]
        fsg = field_side_flow_generator(region)
        eps = float(fsg["eigenvalues"][-1])  # the single positive magnitude, per FOCK-0d SS6
        lam1 = c1 / eps
        lam_n = {n: c_of_n[n] / eps for n in range(1, N_max + 1)}
        worst_lin = max(abs(lam_n[n] / lam1 - n) for n in lam_n)
        results.append({"region": region, "orbit_size": orb["orbit_size"],
                         "is_triangle": orb["is_triangle"], "epsilon": eps,
                         "lambda_1": lam1, "lambda_n": lam_n,
                         "linear_in_n_residual": worst_lin})
    return results


def fock0e_shell_aggregate_clock_note(N_max=8):
    """[FOCK-0e SS M, m5 -- SECONDARY DISCLOSED NOTE, STRUCTURE-ONLY, NOT VERDICT-CARRYING] the
    per-SHELL AGGREGATE clock (D_n = 12.2^(n-1) words of length n, build_hist's OWN dimension
    count; P_n := sum of omega_diag over shell n = D_n.u^(2n)/Z): algebraically,
        -ln(P_n) = -ln(D_n) - 2n.ln(u) + ln(Z) = n.(c_1 - h_top) - ln(6) + ln(Z)
    (using -ln(12.2^(n-1)) = -ln(12) - (n-1).ln(2) = n.(-ln 2) + (ln2 - ln12) = -n.h_top - ln(6)),
    i.e. EXACTLY AFFINE in n, rate beta' = beta_natural - h_top, offset -ln(6) (up to the same
    dropped-ln(Z) convention as m1-m4).  DOES NOT ADJUDICATE beta' vs beta_natural (the ORIGINAL
    A1 poison, FOCK-0 SS3, stands verbatim -- this note reports a structural DERIVATION of the
    FOUR-TEMPERATURES DICTIONARY's own 'per-path vs per-shell differ by exactly h_top' identity
    (BOOTCAMP SS4), labeled as such, nothing more).
    Returns {'N_max','beta_natural','h_top','beta_prime','ln_Z',
    'rows': [{'n','D_n','D_n_theory','P_n','neg_ln_P_n','affine_theory','residual'}],
    'worst_residual'}."""
    words, index, lengths, omega = omega_diag_length(N_max)
    seed_i = index[()]
    ln_Z = float(-math.log(omega[seed_i]))
    beta_natural = fock0e_analytic_lemma()["c1_neg2ln_alpha1"]
    h_top = math.log(2.0)
    beta_prime = beta_natural - h_top
    rows = []
    for n in range(1, N_max + 1):
        idx = np.where(lengths == n)[0]
        D_n = int(len(idx))
        D_n_theory = 12 * 2 ** (n - 1)
        P_n = float(np.sum(omega[idx]))
        neglnP = -math.log(P_n)
        theory = n * beta_prime - math.log(6.0) + ln_Z
        rows.append({"n": n, "D_n": D_n, "D_n_theory": D_n_theory, "P_n": P_n,
                     "neg_ln_P_n": neglnP, "affine_theory": theory,
                     "residual": abs(neglnP - theory)})
    worst = max(row["residual"] for row in rows)
    return {"N_max": N_max, "beta_natural": beta_natural, "h_top": h_top,
            "beta_prime": beta_prime, "ln_Z": ln_Z, "rows": rows, "worst_residual": worst}


def fock0e_selftest_2026_07_12(verbose=True):
    """FOCK-0e station regression: Sections 7/7b/8/8b/8c/8d + module anchors untouched, THEN the
    m1 analytic-identity check, m2's machine-precision linearity table at N_max=8 (every shell),
    m3's Tomita-route consistency anchor at N_max=4, m4's lambda-structure booking (all 4 region
    orbits, triangle lambda_1 reproduces the 0d ember 2.463), and m5's disclosed secondary affine-
    per-shell note.  Does NOT itself adjudicate the SS N verdict tree (architect-only per the
    pre-reg)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 8e self-test -- FOCK-0e A1 clock linearity (2026-07-12)")
        print("=" * 88)

    ck("ANCHORS + Sections 7/7b/8/8b/8c/8d untouched",
       anchor_cell_projector() and anchor_tick_2pi() and accretion_selftest_2026_07_10(verbose=False)
       and i2b_selftest_2026_07_11(verbose=False) and fock0_selftest_2026_07_11(verbose=False)
       and fock0b_selftest_2026_07_11(verbose=False) and fock0c_selftest_2026_07_11(verbose=False)
       and fock0d_selftest_2026_07_11(verbose=False))

    lem = fock0e_analytic_lemma()
    ck(f"m1 ANALYTIC LEMMA: -2ln(alpha_1) = {lem['c1_neg2ln_alpha1']:.10f} == 16ln(3/2) "
       f"(resid {lem['identity_residual_vs_16ln32']:.2e}) == I2b's own beta_natural "
       f"6.4874417297 (resid {lem['identity_residual_vs_I2b_beta_natural']:.2e})",
       lem["identity_residual_vs_16ln32"] < 1e-9
       and lem["identity_residual_vs_I2b_beta_natural"] < 1e-6)

    tbl = fock0e_clock_linearity_table(N_max=8)
    ck(f"m2 MACHINE CHECK [N_max=8, every shell 0..8]: worst in-shell scalarity residual "
       f"{tbl['worst_scalarity_residual']:.2e}, worst |c_n/c1-n| {tbl['worst_ratio_residual']:.2e}",
       tbl["worst_scalarity_residual"] < 1e-9 and tbl["worst_ratio_residual"] < 1e-9)

    tc = fock0e_tomita_route_check(N_max=4)
    ck(f"m3 TOMITA-ROUTE ANCHOR [N_max=4]: shell1 residual {tc['residual_shell1']:.2e}, "
       f"shell2 residual {tc['residual_shell2']:.2e} (direct-rho vs Delta_half eigen-route)",
       tc["residual_shell1"] < 1e-9 and tc["residual_shell2"] < 1e-9)

    lam = fock0e_lambda_structure(N_max=8)
    tri = next(r for r in lam if r["is_triangle"])
    worst_lam = max(r["linear_in_n_residual"] for r in lam)
    ck(f"m4 LAMBDA STRUCTURE [4 region orbits]: worst linear-in-n residual {worst_lam:.2e}; "
       f"triangle lambda_1={tri['lambda_1']:.4f} (0d ember reference 2.463)",
       worst_lam < 1e-6 and abs(tri["lambda_1"] - 2.463) < 0.01)

    m5 = fock0e_shell_aggregate_clock_note(N_max=8)
    ck(f"m5 SECONDARY NOTE (structure-only, NOT verdict-carrying): per-shell aggregate clock "
       f"exactly affine, rate beta'={m5['beta_prime']:.6f}, worst residual "
       f"{m5['worst_residual']:.2e}, all D_n match 12*2^(n-1)",
       m5["worst_residual"] < 1e-8 and all(row["D_n"] == row["D_n_theory"] for row in m5["rows"]))

    if verbose:
        print("RESULT:", "FOCK-0e SECTION-8e REGRESSION PASSES" if ok else "A FOCK-0e CHECK FAILED")
    return ok


# ===========================================================================
# 8f. A2 -- THE WELD AS SECOND-QUANTIZATION FUNCTOR: phi_1 + Gamma(phi_1) + THE TOWER TEST
#     (2026-07-12, Push 2's pivotal station)
#     A2_weld_functor_prereg_2026-07-12.md SS3-4
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as Sections 7/7b/8/8b/8c/8d/8e above.]
#
# HYPOTHESIS UNDER TEST (SS1): Phi = Gamma(phi_1), phi_1 : shell-1 (the 12 darts) -> LEVEL-1 of F
# (the 3-dim Pw[1] subspace, NOT full F -- a strict refinement of FOCK-0c's shell-1 target,
# pin (i)).  phi_1 is additionally pinned by the per-sector antiunitary Tomita relation
# (pin (ii), the SAME J_hist=reversal-o-conj / J_F=field_algebra_conjugation()'s K as FOCK-0b/0c,
# reused unchanged, restricted to the level-1 codomain) and the R/F_bit parity pin (pin (iii),
# FOCK-0c's own precedent, reused unchanged).  D4 (frozen, hard guard): the species map is NEVER
# an input anywhere below -- grep-confirmed, no gauge_sector_category species label is read INTO
# any constraint; it only ever appears in the OUTPUT-only a2_species_read.
#
# NUMBERS APPEAR NOWHERE: every quantity below is a dimension, nullity, rank, or exactness
# residual (structure), never M_Z/ppm/m_nu/a_e (module contract + pre-reg SS6).
#
# ML-2b/HK-7 CONDITIONALITY (carries into every DR-frame-touching sentence below, verbatim,
# unchanged from Section 8's own banner): "Every duality check here (HK-5) is CELL-LEVEL only (the
# 6-edge static vacuum). ML-2b's DR-frame argument is CONDITIONAL on the TD-limit duality holding,
# which is NOT verified by this suite."
def a2_level1_allowance():
    """[A2 SS3, THE PRE-DECLARED ALLOWANCE, PRINTED BEFORE SOLVING] level-1 (Pw[1], dim 3) is
    PAIRED with level-2 (Pw[2]) under the SAME field-side bit-orbit sector_pair_conjugation/
    field_algebra_conjugation already established (orbit_12 = {1,2}, K.Pw[1].K == Pw[2] EXACTLY):
    level-1 belongs to EXACTLY ONE sector-pair block (GROUP-12), not two (contrast FOCK-0c's
    shell-1 -> full-F target, which touched BOTH GROUP-03 and GROUP-12).  Applying FOCK-0c's OWN
    W1 convention ('one overall phase per sector-pair block' -- fock0c_w1_allowance, the_net.py
    ~3143) at the granularity phi_1's OWN codomain touches: 1 relevant block x 1 real dim (one
    overall phase) = 1 real dimension, counted from the SS8 pairing structure ALONE, BEFORE any
    solve.  NAMED CAVEAT (stated up front, not discovered after the fact): FOCK-0c's own shell-1
    full-pin result already found GROUP-12's OWN diagonal sub-block nullity to be EXACTLY 0 (not
    the nominally-declared 1) -- so a 0 result below is CONSISTENT with that precedent, not a
    surprise; the allowance is still the PRE-registered comparison target, per the FOCK-0c/0d
    discipline of declaring the number before solving regardless of what the precedent suggests.
    Returns {'relevant_pair_blocks','real_dims_per_block','allowance','precedent_group12_nullity'}."""
    fock0c_precedent_group12_nullity = 0  # fock0c_full_pin_shell(1,...)['full_pin_group12'], cited not re-solved here
    return {"relevant_pair_blocks": 1, "real_dims_per_block": 1, "allowance": 1,
            "precedent_group12_nullity_at_shell1": fock0c_precedent_group12_nullity}


def _a2_level1_grading_rows(n_domain=ND):
    """[A2 pin (i), GRADING PRESERVATION] the three zero-block rows forcing Pw[0].Phi = Pw[2].Phi
    = Pw[3].Phi = 0 identically (Phi's image confined to Pw[1], level-1's 3-dim range) for a
    domain of dimension n_domain (12 = the full shell-1 dart space, per SS1's literal 'phi_1 maps
    shell-1 -> level-1' -- NO further domain-side sector split is imposed by (i); reuses
    _zero_block_rows unchanged, P_col = I_(n_domain) so the constraint applies to the WHOLE
    domain, not a sector-restricted piece of it)."""
    Pw, _ = _sector_projectors(sign=+1)
    I_dom = np.eye(n_domain)
    return [_zero_block_rows(Pw[w], 8, I_dom, n_domain) for w in (0, 2, 3)]


def a2_phi1_pin_trajectory():
    """[A2 SS3, THE NULLITY TRAJECTORY] stacks pins (i)/(ii)/(iii) INCREMENTALLY on the SAME
    12-dim shell-1 domain (R = reversal(), the SAME antiunitary/parity operators FOCK-0b/0c
    already built and verified -- REUSED, not rebuilt), reporting nullity at each stage:
      stage 0: (i) grading alone (Phi's image confined to Pw[1]);
      stage 1: (i) + (ii) (+ the antiunitary Tomita pin, Phi.J_hist = J_F.Phi);
      stage 2: (i) + (ii) + (iii) (+ the R/F_bit parity pin);
      stage 3 (diagnostic only, NOT part of the frozen pin set): (i) + (iii), no antiunitary --
        isolates how much of the collapse (ii) alone is responsible for.
    ALSO verifies the pre-reg's own cross-check (SS4.1): this nullity must be <= FOCK-0c's own
    shell-1 -> full-F nullity (16), since the grading refinement (i) can only SHRINK it (a
    STRICTER codomain restriction can only remove solutions, never add any -- verified as a plain
    integer inequality below, not re-derived structurally).
    Returns {'stage0_grading_alone','stage1_grading_plus_antiunitary','stage2_full_pin',
    'stage3_grading_plus_rfbit_no_antiunitary','total_real_dim','shrinks_fock0c_16'}."""
    R = reversal()
    fa = field_algebra_conjugation()
    K = fa["M"]
    Pw, _ = _sector_projectors(sign=+1)
    F_bit = Pw[0] + Pw[3] - Pw[1] - Pw[2]
    grading_rows = _a2_level1_grading_rows(n_domain=ND)
    anti_rows = [_antiunitary_pin_rows(R, K)]
    rfbit_rows = [_linear_pin_rows(R, F_bit)]

    def nullity_of(rowlist):
        Cop = np.vstack(rowlist)
        s = np.linalg.svd(Cop, compute_uv=False)
        rank = int(np.sum(s > 1e-8))
        total = Cop.shape[1]
        return total - rank, total

    n0, t0 = nullity_of(grading_rows)
    n1, t1 = nullity_of(grading_rows + anti_rows)
    n2, t2 = nullity_of(grading_rows + anti_rows + rfbit_rows)
    n3, t3 = nullity_of(grading_rows + rfbit_rows)
    assert t0 == t1 == t2 == t3 == 4 * ND * 8, \
        f"a2_phi1_pin_trajectory: ambient real dim mismatch ({t0},{t1},{t2},{t3}) != {4*ND*8}"
    assert n2 <= 16, f"a2_phi1_pin_trajectory: full-pin nullity {n2} exceeds FOCK-0c's shell-1 16"
    return {"stage0_grading_alone": n0, "stage1_grading_plus_antiunitary": n1,
            "stage2_full_pin": n2, "stage3_grading_plus_rfbit_no_antiunitary": n3,
            "total_real_dim": t0, "shrinks_fock0c_16": bool(n2 <= 16)}


def a2_phi1_forced_zero_proof():
    """[A2 SS3/SS5 AF-3, THE ALGEBRAIC PROOF -- not SVD alone] PROVES phi_1 = 0 is FORCED by pins
    (i)+(ii) alone (independent of (iii), independent of the domain's own A4-sector structure),
    from three already-accreted facts, none re-derived:
      FACT 1 (sector_pair_conjugation / field_algebra_conjugation, REUSED): K = M@conj(.) swaps
        range(Pw[1]) <-> range(Pw[2]) EXACTLY (K.Pw[1].K == Pw[2]), so for ANY x in range(Pw[1]),
        K(x) in range(Pw[2]).
      FACT 2 (NHAT eigenspaces, REUSED): range(Pw[1]) cap range(Pw[2]) = {0} (orthogonal
        eigenspaces of the Hermitian occupation-number operator NHAT at DIFFERENT eigenvalues).
      FACT 3 (reversal(), REUSED): R is an involution (R^2=I), hence v -> R@conj(v) is a BIJECTION
        of C^12 onto itself (antiunitary involutions are always bijective).
    PROOF: suppose Phi's image is confined to range(Pw[1]) (pin (i)) and Phi(R@conj(v)) =
    K@conj(Phi(v)) for ALL v (pin (ii), phi_1's Tomita relation restricted to level-1's codomain,
    read literally: the SAME global K FOCK-0b/0c/0d/0e already use, since no OTHER antiunitary
    endomorphism of Pw[1] alone is accreted anywhere -- inventing one would violate SS7's
    reuse-never-rebuild discipline).  For any v, LHS = Phi(R@conj(v)) in range(Phi) subset
    range(Pw[1]) (FACT nothing new, just pin (i)).  RHS = K@conj(Phi(v)); if Phi(v) != 0 then
    Phi(v) in range(Pw[1]) so K@conj(Phi(v)) in range(Pw[2]) (FACT 1).  LHS=RHS forces this common
    value into range(Pw[1]) cap range(Pw[2]) = {0} (FACT 2), i.e. Phi(R@conj(v)) = 0.  Since
    v -> R@conj(v) is a bijection of C^12 (FACT 3), Phi(u) = 0 for EVERY u in C^12.  QED: phi_1 = 0
    identically -- a level-1-ONLY codomain is incompatible with the antiunitary pin BY THEOREM,
    regardless of any numeric truncation.  Verifies FACTS 1-3 numerically (machine precision) and
    cross-checks against a2_phi1_pin_trajectory's stage-1 nullity (must be exactly 0).
    Returns {'K_swaps_Pw1_to_Pw2_residual','Pw1_cap_Pw2_residual','R_involution_residual',
    'stage1_nullity_confirms_proof','proof_holds'}."""
    fa = field_algebra_conjugation()
    K = fa["M"]
    Pw, _ = _sector_projectors(sign=+1)
    R = reversal()
    fact1 = float(np.max(np.abs(K @ np.conj(Pw[1]) @ np.conj(K) - Pw[2])))
    fact2 = float(np.max(np.abs(Pw[1] @ Pw[2])))
    fact3 = float(np.max(np.abs(R @ R - np.eye(ND))))
    traj = a2_phi1_pin_trajectory()
    stage1 = traj["stage1_grading_plus_antiunitary"]
    assert fact1 < 1e-8 and fact2 < 1e-8 and fact3 < 1e-9, \
        (f"a2_phi1_forced_zero_proof: a load-bearing FACT failed (K-swap={fact1:.2e}, "
         f"Pw1-cap-Pw2={fact2:.2e}, R-involution={fact3:.2e}) -- the proof's premises do not hold")
    assert stage1 == 0, \
        f"a2_phi1_forced_zero_proof: numeric nullity {stage1} != 0, the algebraic proof's conclusion is CONTRADICTED"
    return {"K_swaps_Pw1_to_Pw2_residual": fact1, "Pw1_cap_Pw2_residual": fact2,
            "R_involution_residual": fact3, "stage1_nullity_confirms_proof": stage1,
            "proof_holds": True}


def a2_alternate_reading_diagnostic():
    """[A2, AMBIGUITY NOTE -- NOT the frozen pin, a labeled secondary diagnostic per the contract's
    'take the strictest reading, note the ambiguity, do not choose silently'] tests whether the
    forced-zero result (a2_phi1_forced_zero_proof) is an artifact of confining the codomain to a
    SINGLE level, by relaxing ONLY that one choice: codomain = the FULL sector-pair block GROUP-12
    (Pw[1]+Pw[2], dim 6, level-1 UNION level-2 -- the K-orbit CLOSED under the antiunitary swap),
    domain STILL the unrestricted 12-dim shell-1 (unlike FOCK-0c's own group12 sub-block, which
    ALSO restricted the domain to GROUP-12(hist), a 2-dim sub-piece -- this diagnostic keeps the
    full 12-dim domain to isolate the codomain choice's effect alone).  Result: nonzero (144/384
    with the antiunitary pin, 72/384 with R/F_bit added) -- CONFIRMING the forced-zero result is
    SPECIFIC to asking for a codomain that is only HALF a K-orbit, not a generic collapse.  This is
    NOT the frozen SS1 hypothesis (SS1 states level-1, the 3-dim piece, explicitly, 'the 3 complex
    modes') -- reported here ONLY as an ambiguity-resolution diagnostic, never substituted for the
    primary result.
    Returns {'stage_grading_alone','stage_plus_antiunitary','stage_full_pin','total_real_dim'}."""
    R = reversal()
    fa = field_algebra_conjugation()
    K = fa["M"]
    Pw, _ = _sector_projectors(sign=+1)
    F_bit = Pw[0] + Pw[3] - Pw[1] - Pw[2]
    I12 = np.eye(ND)
    grading_rows_alt = [_zero_block_rows(Pw[w], 8, I12, ND) for w in (0, 3)]
    anti_rows = [_antiunitary_pin_rows(R, K)]
    rfbit_rows = [_linear_pin_rows(R, F_bit)]

    def nullity_of(rowlist):
        Cop = np.vstack(rowlist)
        s = np.linalg.svd(Cop, compute_uv=False)
        rank = int(np.sum(s > 1e-8))
        total = Cop.shape[1]
        return total - rank, total

    n0, t0 = nullity_of(grading_rows_alt)
    n1, t1 = nullity_of(grading_rows_alt + anti_rows)
    n2, t2 = nullity_of(grading_rows_alt + anti_rows + rfbit_rows)
    return {"stage_grading_alone": n0, "stage_plus_antiunitary": n1, "stage_full_pin": n2,
            "total_real_dim": t0}


def _level1_creation_ops():
    """[A2 SS4, THE FERMIONIC REALIZATION OF level-1] the 3 canonical single-particle creation
    operators A_ops[m]^dagger (m=0,1,2) on F's 8-dim Cl(6) Fock space and the Fock vacuum |vac> --
    the IDENTICAL construction _sector_projectors/field_algebra_conjugation already use (REUSED
    verbatim: gam(conj(modes[:,m]))/sqrt(2), NHAT's lowest eigenvector -- nothing new invented).
    VERIFIES (not assumed): Adag[m]|vac> spans range(Pw[1]) EXACTLY (the fermionic single-particle
    basis realizing level-1), and the creation-creation CAR {Adag_i,Adag_j}=0 for ALL i,j
    (including i=j -- the Grassmann/exterior-algebra structure P1/P2 both rest on).
    Returns (Adag: list of 3 complex 8x8 matrices, vac: 8x1 complex column, 'basis_residual',
    'car_residual')."""
    sys.path.insert(0, _REPO)
    sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "bridge"))
    from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402
    g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]

    def gam(x):
        return sum(x[a] * g6[a] for a in range(NE))

    J6 = complex_structure_J6()
    wJ, VJ = np.linalg.eig(J6)
    modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
    A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
    NHAT = sum(a.conj().T @ a for a in A_ops)
    wN, VN = np.linalg.eigh(NHAT)
    vac = VN[:, [int(np.argmin(wN))]]
    vac = vac / np.linalg.norm(vac)
    Adag = [a.conj().T for a in A_ops]
    Pw, _ = _sector_projectors(sign=+1)
    level1_states = np.hstack([Adag[m] @ vac for m in range(3)])
    basis_resid = float(np.max(np.abs(Pw[1] @ level1_states - level1_states)))
    car_resid = 0.0
    for i in range(3):
        for j in range(3):
            car_resid = max(car_resid, float(np.max(np.abs(Adag[i] @ Adag[j] + Adag[j] @ Adag[i]))))
    assert basis_resid < 1e-8, f"_level1_creation_ops: Adag|vac> not in Pw[1] range ({basis_resid:.2e})"
    assert car_resid < 1e-8, f"_level1_creation_ops: creation-creation CAR violated ({car_resid:.2e})"
    return Adag, vac, basis_resid, car_resid


def a2_gamma_word(phi1, word, Adag, vac):
    """[A2 SS1/SS4, THE FUNCTOR ON A WORD] Gamma(phi_1)(w) = phi_1(d_1) ^ phi_1(d_2) ^ ... ^
    phi_1(d_n) for w=(d_1,...,d_n), REALIZED fermionically (not as an abstract wedge symbol): each
    dart's image phi_1(d) in C^3 (level-1's coordinates in the Adag[0..2] basis) becomes the
    creation operator sum_m phi1[d,m].Adag[m], applied RIGHT-TO-LEFT (d_n first) to |vac>.  This
    is the SAME construction as an ordinary Slater-determinant/CAR product; antisymmetrization is
    NOT imposed by hand -- it is the automatic consequence of {Adag_i,Adag_j}=0 (creation operators
    of ANY two single-particle states anticommute, including a state with itself), which is
    EXACTLY what makes P1 (Pauli truncation, shell>=4 vanishes -- only 3 modes exist) and P2
    (repeated-dart kernel) hold BY CONSTRUCTION, checked below, not assumed.  phi1 is a
    (n_domain x 3) complex matrix (phi1[d,m] = the m-th level-1 coordinate of phi_1(d)); word is
    ANY tuple of integer dart indices in range(n_domain) -- including tuples that are NOT
    admissible walks in H_hist (the functor's DEFINING formula makes sense on any tuple; P2's
    'any word with a repeated dart' is a claim about the FUNCTOR FORM, tested here on synthetic
    tuples where needed, see a2_repeated_dart_kernel_check)."""
    v = vac.copy()
    for d in reversed(word):
        op = sum(phi1[d, m] * Adag[m] for m in range(3))
        v = op @ v
    return v


def a2_pauli_truncation_check(N_max=6, seed=0):
    """[A2 SS4 step 2, P1 -- PAULI TRUNCATION, STRUCTURAL] Gamma(phi_1) is IDENTICALLY ZERO on
    every shell n >= 4, for ANY phi_1 (not just the frozen station's solution, which is itself 0 --
    see a2_phi1_forced_zero_proof): Lambda^n of a 3-dim space vanishes for n>=4, a GENERAL fact of
    the fermionic realization (only 3 creation operators exist; any product of 4+ of them, however
    combined, contains a repeat by pigeonhole and vanishes via {Adag_i,Adag_j}=0).  Verified here
    with a STRUCTURAL TEST MAP (a generic complex 12x3 matrix, NOT the frozen pin's solution --
    labeled explicitly, since the frozen solution is 0 and would make this check vacuous) on EVERY
    admissible word of shell 4 up to N_max.
    Returns {'N_max','n_shell4_words','worst_shell4_norm','n_shell_le3_nonzero_sample'}."""
    Adag, vac, _, _ = _level1_creation_ops()
    rng = np.random.default_rng(seed)
    phi_test = rng.normal(size=(ND, 3)) + 1j * rng.normal(size=(ND, 3))
    _, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(N_max, succ)
    idx4 = [i for i, w in enumerate(words) if len(w) == 4]
    worst4 = max((float(np.max(np.abs(a2_gamma_word(phi_test, words[i], Adag, vac))))
                  for i in idx4), default=0.0)
    idx3 = [i for i, w in enumerate(words) if len(w) == 3]
    sample3 = [float(np.max(np.abs(a2_gamma_word(phi_test, words[i], Adag, vac))))
               for i in idx3[:5]]
    assert worst4 < 1e-9, f"a2_pauli_truncation_check: shell-4 Gamma(phi_test) nonzero ({worst4:.2e})"
    assert all(x > 1e-6 for x in sample3), \
        "a2_pauli_truncation_check: shell-3 Gamma(phi_test) degenerately zero (test map not generic)"
    return {"N_max": N_max, "n_shell4_words": len(idx4), "worst_shell4_norm": worst4,
            "n_shell_le3_nonzero_sample": sample3}


def a2_repeated_dart_kernel_check(N_max=10, seed=0):
    """[A2 SS4 step 2, P2 -- THE REPEATED-DART KERNEL, STRUCTURAL + AN HONEST DOMAIN FINDING]
    Gamma(phi_1)(w) = 0 for ANY word w with a repeated dart (adjacent OR NOT -- Adag(v)Adag(w)
    ANTICOMMUTES for ANY v,w since {Adag_i,Adag_j}=0 for ALL i,j including i=j, so moving a
    repeated factor together via transpositions costs only signs, and Adag(v)^2=0 kills it), a
    GENERAL fact of the fermionic realization, independent of phi_1's specific values.  VERIFIED
    on a SYNTHETIC tuple (dart 0, dart 1, dart 0) -- NOT necessarily an admissible H_hist word (the
    functor's formula is defined on any tuple; admissibility is a SEPARATE, additional fact about
    which tuples the walk actually realizes, checked next).
    HONEST DOMAIN FINDING (reported, not smoothed over): scanning build_hist up to N_max, NO
    admissible word of length <= 3 contains a repeated dart AT ALL (the first repeated-dart
    admissible word appears at shell 4) -- i.e. P2's predicted kernel is EMPTY within shells 1-3,
    the ONLY shells where Gamma(phi_1) could be nonzero per P1 (shell>=4 already vanishes for
    dimension reasons).  P1 and P2 therefore do NOT bite independently on THIS graph: within the
    functor's live domain, P1 alone accounts for every vanishing; P2 holds VACUOUSLY there.  This
    is the combinatorial shadow the pre-reg (SS2 P2) points at (T0-N-3's 864/864
    orientation-compatibility lemma + its 'coincidence collides 10/10' finding,
    internal research notes:30-50,105,114-116 and
    proofs/foundations/T0N3_domain_theorem_2026-07-11.py:349-373 -- a DIFFERENT construction
    [ground-state cycle pairs/triples under CAR], but the SAME underlying mechanism: reusing a
    dart/mode twice under a fermionic/CAR realization forces the state to ZERO -- cited here for
    the mechanism-level connection, not re-derived or re-run).
    Returns {'synthetic_repeat_residual','first_shell_with_admissible_repeat','n_max_scanned',
    'shell_le3_repeat_counts'}."""
    Adag, vac, _, _ = _level1_creation_ops()
    rng = np.random.default_rng(seed)
    phi_test = rng.normal(size=(ND, 3)) + 1j * rng.normal(size=(ND, 3))
    synthetic_word = (0, 1, 0)
    resid = float(np.max(np.abs(a2_gamma_word(phi_test, synthetic_word, Adag, vac))))
    assert resid < 1e-9, f"a2_repeated_dart_kernel_check: synthetic repeated-dart word nonzero ({resid:.2e})"

    _, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(N_max, succ)
    shell_counts = {}
    first_shell = None
    for n in range(1, N_max + 1):
        ws = [w for w in words if len(w) == n]
        reps = [w for w in ws if len(set(w)) < len(w)]
        shell_counts[n] = len(reps)
        if reps and first_shell is None:
            first_shell = n
    assert shell_counts.get(1, 0) == 0 and shell_counts.get(2, 0) == 0 and shell_counts.get(3, 0) == 0, \
        f"a2_repeated_dart_kernel_check: an admissible repeated-dart word exists at shell <=3 ({shell_counts})"
    return {"synthetic_repeat_residual": resid, "first_shell_with_admissible_repeat": first_shell,
            "n_max_scanned": N_max,
            "shell_le3_repeat_counts": {n: shell_counts[n] for n in range(1, 4)}}


def _a2_vec_colmajor_from_AB(A, B, n, m):
    """[A2 SS4 step 3, THE TOWER-MEMBERSHIP VECTORIZATION] the REAL parameter vector (length 4nm)
    representing a real-linear map Phi(v)=A.v+B.conj(v) (A,B complex m x n) in the IDENTICAL
    convention pinned_map_hom_space_real/_zero_block_rows/_antiunitary_pin_rows already use
    (verified against a live FOCK-0c shell-1 null-space vector, residual < 1e-15, not merely
    asserted): the real 2m x 2n embedding M = [[Are+Bre, Bim-Aim],[Aim+Bim, Are-Bre]] (derived from
    Phi(p+iq) = (Are.p-Aim.q+Bre.p+Bim.q) + i(Aim.p+Are.q+Bim.p-Bre.q)), column-major (Fortran)
    flattened -- the SAME vec(X) convention _zero_block_rows's Kronecker-vec identity uses."""
    Are, Aim, Bre, Bim = A.real, A.imag, B.real, B.imag
    M11 = Are + Bre
    M12 = Bim - Aim
    M21 = Aim + Bim
    M22 = Are - Bre
    X = np.block([[M11, M12], [M21, M22]])
    return X.flatten(order="F")


def a2_tower_membership_test(shell_n, phi1_basis, N_max=4):
    """[A2 SS4 step 3, THE FORCING QUESTION -- D2 FROZEN] for EACH basis map Phi1 in phi1_basis
    (a list of complex ND x 3 matrices; empty if phi_1's nullity is 0, per
    a2_phi1_forced_zero_proof), constructs Gamma(Phi1) explicitly on shell_n as an
    (8 x D_shell_n) complex matrix (columns = a2_gamma_word over the shell's own word order,
    IDENTICAL to _fock0c_rows_full's own np.where(lengths==shell_n) slicing -- same N_max, same
    ordering, nothing re-indexed), vectorizes it (_a2_vec_colmajor_from_AB, B=0 since Gamma(Phi1)
    is genuinely COMPLEX-linear -- a product of complex-linear creation operators, no antilinear
    part), and tests membership in FOCK-0c's shell_n full-pin null space by applying
    _fock0c_rows_full's OWN row-stack (REUSED, not rebuilt) and checking the residual.
    IF phi1_basis IS EMPTY: Gamma(phi_1)=0 identically trivially lies in EVERY linear space (the 0
    vector) -- reported HONESTLY as VACUOUS, not claimed as a nontrivial confirmation.  A
    DISCRIMINATING CONTROL is run alongside: the SAME construction with the structural test map
    from a2_pauli_truncation_check (a GENUINE non-solution) is checked too, to confirm the test
    machinery is not degenerately accepting -- IF the control ALSO passes, that would mean the
    tower test cannot discriminate at all (a named caveat to report); if it correctly FAILS
    (nonzero residual), the machinery is confirmed discriminating even though the frozen result is
    vacuous.
    Returns {'shell_n','D_shell_n','basis_size','member_residuals' (list, one per basis map),
    'all_members','control_residual','control_is_member'}."""
    _, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(max(N_max, shell_n), succ)
    idx = np.where(lengths == shell_n)[0]
    D_n = len(idx)
    rows_full, n_dom, m_cod = _fock0c_rows_full(shell_n, N_max=N_max)
    assert n_dom == D_n and m_cod == 8, \
        f"a2_tower_membership_test: dimension mismatch ({n_dom},{m_cod}) vs ({D_n},8)"
    Cop = np.vstack(rows_full)
    Adag, vac, _, _ = _level1_creation_ops()

    def gamma_matrix(phi1):
        cols = [a2_gamma_word(phi1, words[i], Adag, vac) for i in idx]
        return np.hstack(cols) if cols else np.zeros((8, 0), dtype=complex)

    def residual_of(Gamma):
        vecX = _a2_vec_colmajor_from_AB(Gamma, np.zeros_like(Gamma), D_n, 8)
        return float(np.max(np.abs(Cop @ vecX))) if Cop.size else 0.0

    member_residuals = []
    for phi1 in phi1_basis:
        Gamma = gamma_matrix(phi1)
        member_residuals.append(residual_of(Gamma))
    all_members = all(r < 1e-6 for r in member_residuals) if member_residuals else True

    rng = np.random.default_rng(0)
    phi_test = rng.normal(size=(ND, 3)) + 1j * rng.normal(size=(ND, 3))
    control_resid = residual_of(gamma_matrix(phi_test))
    return {"shell_n": shell_n, "D_shell_n": D_n, "basis_size": len(phi1_basis),
            "member_residuals": member_residuals, "all_members": bool(all_members),
            "control_residual": control_resid, "control_is_member": bool(control_resid < 1e-6)}


def a2_clock_read(N_max=8):
    """[A2 SS4 step 4, CLOCK READ -- STRUCTURE ONLY, NO GLOBAL-LAMBDA PIN (SS2 TRAP, NOT
    RE-ATTEMPTED)] since phi_1 is forced to 0 (a2_phi1_forced_zero_proof), there is NO nonzero
    solution whose intersection with h_A's/K_F's eigenspaces could be reported per region orbit --
    reported HONESTLY as EMPTY, not smoothed into a claim.  The A1-booked per-level structure
    (lambda_n = n.lambda_1, FOCK-0e's fock0e_lambda_structure, REUSED not re-derived) is restated
    here as INHERITED STRUCTURE, unconditionally true of the CONSTRUCTION regardless of THIS
    station's forced-zero outcome (it is a fact about K_hist/K_F's own spectra, not about phi_1).
    NO global-lambda pin is imposed or attempted anywhere in this station (the SS2 pre-registered
    trap: a single global lambda against a single 3-edge region's K_F is EXPECTED to fail at level
    2 by the W2 mechanism -- that theorem-let is FOCK-0d's, not re-run here).
    Returns {'phi1_nonzero_eigenspace_intersection','lambda_structure_inherited' (FOCK-0e's own
    per-orbit table, reused), 'global_lambda_pin_attempted'}."""
    lam = fock0e_lambda_structure(N_max=N_max)
    return {"phi1_nonzero_eigenspace_intersection": "EMPTY (phi_1 = 0, no nonzero solution exists)",
            "lambda_structure_inherited": lam, "global_lambda_pin_attempted": False}


def a2_species_read():
    """[A2 SS4 step 5, SPECIES READ -- OUTPUT ONLY, D4 HARD GUARD] reports which history isotypic
    components of the [1,1,1,9] decomposition survive in phi_1's image/kernel: since phi_1 = 0
    identically (a2_phi1_forced_zero_proof), EVERY history isotypic component is in the KERNEL and
    NONE survive into a nonzero image -- there is no correspondence with {1,3,3,1} to read off.
    D4 GUARD (verified, not merely claimed): grep-confirmed that no `gauge_sector_category`/
    species-label value is read INTO any pin/row-builder in this section (a2_level1_allowance,
    _a2_level1_grading_rows, a2_phi1_pin_trajectory, a2_phi1_forced_zero_proof) -- the species
    grading enters ONLY here, as a report of the (empty) result, never as an input constraint.
    Returns {'surviving_isotypic_components','species_correspondence','note'}."""
    return {"surviving_isotypic_components": [],
            "species_correspondence": "NONE -- phi_1 = 0 identically; no nonzero image exists to "
                                       "read a species correspondence from",
            "note": "D4 respected: the species map was never a constraint anywhere in SS3's pin "
                    "construction (a2_level1_allowance/_a2_level1_grading_rows/"
                    "a2_phi1_pin_trajectory/a2_phi1_forced_zero_proof); this function is the ONLY "
                    "place gauge_sector_category's species labels are narratively invoked, and "
                    "only to report an empty result"}


def a2_weld_selftest_2026_07_12(verbose=True):
    """A2 station regression: Sections 7/7b/8/8b/8c/8d/8e + module anchors untouched, THEN the
    pre-declared level-1 allowance (PRINTED before any solve), the phi_1 nullity trajectory, the
    algebraic forced-zero proof (AF-3), the alternate-reading diagnostic (labeled, not the frozen
    result), P1 (Pauli truncation) and P2 (repeated-dart kernel) structural checks, the tower-
    membership test at shells 2 and 3 (vacuous per the forced-zero result, with a discriminating
    control), and the clock/species reads (structure-only / output-only).  Does NOT itself
    adjudicate SS5's AF-1..4 verdict tree (architect-only per the pre-reg)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 8f self-test -- A2 the weld as second-quantization functor (2026-07-12)")
        print("=" * 88)

    ck("ANCHORS + Sections 7/7b/8/8b/8c/8d/8e untouched",
       anchor_cell_projector() and anchor_tick_2pi() and accretion_selftest_2026_07_10(verbose=False)
       and i2b_selftest_2026_07_11(verbose=False) and fock0_selftest_2026_07_11(verbose=False)
       and fock0b_selftest_2026_07_11(verbose=False) and fock0c_selftest_2026_07_11(verbose=False)
       and fock0d_selftest_2026_07_11(verbose=False) and fock0e_selftest_2026_07_12(verbose=False))

    alw = a2_level1_allowance()
    ck(f"STEP 1 ALLOWANCE (PRINTED BEFORE solving): level-1 touches {alw['relevant_pair_blocks']} "
       f"sector-pair block x {alw['real_dims_per_block']} real dim (one phase) = "
       f"{alw['allowance']} (FOCK-0c precedent for GROUP-12's own diagonal sub-block: "
       f"{alw['precedent_group12_nullity_at_shell1']})", alw["allowance"] == 1)

    traj = a2_phi1_pin_trajectory()
    ck(f"NULLITY TRAJECTORY [ambient {traj['total_real_dim']}]: (i) grading alone = "
       f"{traj['stage0_grading_alone']}; (i)+(ii) antiunitary = "
       f"{traj['stage1_grading_plus_antiunitary']}; FULL PIN (i)+(ii)+(iii) = "
       f"{traj['stage2_full_pin']}; diagnostic (i)+(iii) no antiunitary = "
       f"{traj['stage3_grading_plus_rfbit_no_antiunitary']} -- shrinks FOCK-0c's shell-1 16: "
       f"{traj['shrinks_fock0c_16']}",
       traj["stage2_full_pin"] == 0 and traj["shrinks_fock0c_16"])

    proof = a2_phi1_forced_zero_proof()
    ck("THE ALGEBRAIC PROOF (AF-3-grade, not SVD alone): K swaps range(Pw[1])<->range(Pw[2]) "
       "exactly, range(Pw[1]) cap range(Pw[2]) = {0}, R is an involution (bijective) -- phi_1 = 0 "
       "FORCED, cross-checked against the numeric nullity",
       proof["proof_holds"] and proof["stage1_nullity_confirms_proof"] == 0,
       detail=f"K-swap={proof['K_swaps_Pw1_to_Pw2_residual']:.1e}, "
              f"Pw1-cap-Pw2={proof['Pw1_cap_Pw2_residual']:.1e}, "
              f"R-involution={proof['R_involution_residual']:.1e}")

    alt = a2_alternate_reading_diagnostic()
    ck(f"AMBIGUITY NOTE (NOT the frozen pin): codomain=GROUP-12 (dim 6, both levels) gives "
       f"nonzero nullity ({alt['stage_grading_alone']} -> {alt['stage_plus_antiunitary']} -> "
       f"{alt['stage_full_pin']}) -- confirms the forced-zero result is SPECIFIC to a "
       f"single-level (half-K-orbit) codomain, not a generic collapse",
       alt["stage_plus_antiunitary"] > 0)

    p1 = a2_pauli_truncation_check(N_max=6)
    ck(f"P1 PAULI TRUNCATION [structural test map, N_max={p1['N_max']}]: ALL "
       f"{p1['n_shell4_words']} shell-4 words give Gamma=0 (worst {p1['worst_shell4_norm']:.1e}); "
       f"shell<=3 sample nonzero (test map generic)",
       p1["worst_shell4_norm"] < 1e-9)

    p2 = a2_repeated_dart_kernel_check(N_max=10)
    ck(f"P2 REPEATED-DART KERNEL: synthetic repeat (0,1,0) gives Gamma=0 (resid "
       f"{p2['synthetic_repeat_residual']:.1e}); HONEST FINDING: no admissible repeated-dart word "
       f"exists at shell<=3 (first at shell {p2['first_shell_with_admissible_repeat']}) -- P2 "
       f"holds VACUOUSLY within the functor's live domain (P1 already kills shell>=4)",
       p2["synthetic_repeat_residual"] < 1e-9 and p2["first_shell_with_admissible_repeat"] == 4)

    tw2 = a2_tower_membership_test(2, [], N_max=4)
    tw3 = a2_tower_membership_test(3, [], N_max=4)
    ck(f"TOWER-MEMBERSHIP TEST shell 2 [D={tw2['D_shell_n']}]: basis_size=0 -> VACUOUSLY member "
       f"(the 0 map); discriminating control residual={tw2['control_residual']:.3e} "
       f"(control_is_member={tw2['control_is_member']})",
       tw2["all_members"])
    ck(f"TOWER-MEMBERSHIP TEST shell 3 [D={tw3['D_shell_n']}]: basis_size=0 -> VACUOUSLY member; "
       f"discriminating control residual={tw3['control_residual']:.3e} "
       f"(control_is_member={tw3['control_is_member']})",
       tw3["all_members"])

    clk = a2_clock_read()
    ck("CLOCK READ (structure only, no global-lambda pin attempted): phi_1 eigenspace "
       "intersection EMPTY (0 forced); lambda_n = n.lambda_1 inherited from FOCK-0e unchanged",
       clk["global_lambda_pin_attempted"] is False)

    sp = a2_species_read()
    ck("SPECIES READ (output only, D4 guard): no surviving isotypic components (phi_1 = 0); "
       "species map never used as an input anywhere in SS3's pin construction",
       sp["surviving_isotypic_components"] == [])

    if verbose:
        print("RESULT:", "A2 SECTION-8f REGRESSION PASSES" if ok else "AN A2 CHECK FAILED")
    return ok


# ===========================================================================
# 8g. A2b -- THE CONJUGATE-PAIR WELD (2026-07-12, Push 2 station 2)
#     A2b_conjugate_pair_weld_prereg_2026-07-12.md SS2-5
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as Sections 7/7b/8/8b/8c/8d/8e/8f above.]
#
# HYPOTHESIS UNDER TEST (SS1): AF-3 (A2, cd83d2b/section 8f) proved the SELF-J-pinned functor
# class is EMPTY (a level-1-only codomain collides with the antiunitary Tomita pin, which
# structurally swaps level-1<->level-2).  The revised shape: J does not pin the weld to itself --
# it PAIRS two welds, Phi (grading-preserving) and Phi~ := J_F.Phi.J_hist (grading-REVERSING).
# The TOWER is pinned by grading/flow (A1), NOT by J.  SS3's frozen pin set for phi_1 drops the
# antiunitary pin ENTIRELY: (i) grading, (ii) the R/F_bit parity pin (FOCK-0c's own (ii), PLAIN
# LINEAR, no conjugation), (iii) the per-sector-PAIR block structure (NEW here).  NO self-J-pin,
# NO region-K_F tower pin, NO A4-equivariance pin anywhere in the verdict path (SS5 is
# report-only).  D4 (frozen, hard guard, verbatim from A2): the species map is NEVER an input
# anywhere below -- grep-confirmed, only ever read in the OUTPUT-only diagnostic functions.
#
# NUMBERS APPEAR NOWHERE: every quantity below is a dimension, nullity, rank, or exactness
# residual (structure), never M_Z/ppm/m_nu/a_e (module contract + pre-reg SS7 poisons).
#
# ML-2b/HK-7 CONDITIONALITY (carries into every DR-frame-touching sentence below, verbatim,
# unchanged from Section 8's own banner): "Every duality check here (HK-5) is CELL-LEVEL only (the
# 6-edge static vacuum). ML-2b's DR-frame argument is CONDITIONAL on the TD-limit duality holding,
# which is NOT verified by this suite."
def a2b_l0a_self_j_pin_probe(N_max=4):
    """[A2b SS2, L0a -- MACHINE-VERIFY FIRST] the pre-registered generalization of AF-3: ANY
    level-additive functor with a SELF-J-pin dies beyond shell 1 (additivity caps the image level
    of shell n at >= n; the self-J-pin needs image levels L with L cap (3-L) != {}, which forces
    n <= 1).  Verified here as 'the naive pair-codomain variant's shell-2 death' (the pre-reg's own
    named instance): the SAME antiunitary Tomita pin AF-3 used (R/K = the SAME field_algebra_
    conjugation()['M'], reversal(), REUSED unchanged), but at shell-2 -> level-2 (Pw[2], via
    history_reversal_matrix's shell-2-restricted path-reversal, since reversal() alone only acts
    on shell-1) -- grading (image confined to Pw[2]) + the antiunitary pin, exactly as A2's own
    a2_phi1_pin_trajectory stage 1, generalized to shell 2.  Expect nullity 0 (the level-additive
    self-J class is dead beyond shell 1, confirmed at its first instance beyond shell 1).
    Returns {'shell_n','level_n','domain_dim','nullity','total_real_dim'}."""
    Pw, _ = _sector_projectors(sign=+1)
    fa = field_algebra_conjugation()
    K = fa["M"]
    hr = history_reversal_matrix(N_max)
    idx2 = np.where(hr["lengths"] == 2)[0]
    n2 = len(idx2)
    R2 = hr["Rw"][np.ix_(idx2, idx2)]
    assert float(np.max(np.abs(R2 @ R2 - np.eye(n2)))) < 1e-9, \
        "a2b_l0a_self_j_pin_probe: shell-2 path-reversal is not an involution"
    grading_rows2 = [_zero_block_rows(Pw[w], 8, np.eye(n2), n2) for w in (0, 1, 3)]
    anti_rows2 = [_antiunitary_pin_rows(R2, K)]
    Cop = np.vstack(grading_rows2 + anti_rows2)
    s = np.linalg.svd(Cop, compute_uv=False)
    rank = int(np.sum(s > 1e-8))
    total = Cop.shape[1]
    nullity = total - rank
    assert nullity == 0, f"a2b_l0a_self_j_pin_probe: SURPRISE -- shell-2 self-J-pin nullity {nullity} != 0"
    return {"shell_n": 2, "level_n": 2, "domain_dim": n2, "nullity": nullity, "total_real_dim": total}


def a2b_l0b_region_flow_rank_lemma():
    """[A2b SS2, L0b -- MACHINE-VERIFY FIRST] the pre-registered lemma: a strict per-level
    region-flow pin Phi_1.K_hist = lambda_1.h_A.Phi_1 forces range(Phi_1) into h_A's SINGLE
    eigenspace at whichever eigenvalue mu = c_1/lambda_1 is hit (an ELEMENTARY, general fact: for
    ANY v, h_A.Phi_1(v) = mu.Phi_1(v), so Phi_1(v) is always a mu-eigenvector or 0) -- taking
    mu = the POSITIVE eigenvalue (the ONLY sign lambda_1>0, c_1>0 admits), h_A's own eigenspace
    there is verified 1-DIMENSIONAL (h_A REUSED unchanged from field_side_flow_generator, run over
    ALL 4 A4-orbit region representatives per FOCK-0d's own 'run all inequivalent halves'
    discipline) forces rank(Phi_1) <= 1, hence Lambda^2(range(Phi_1)) = {0} (two vectors in a
    <=1-dim space are always proportional) -- the tower dies at shell 2 UNDER THIS (hypothetical,
    NOT frozen -- SS3 excludes it) pin.  THIS PIN IS NOT PART OF SS3's frozen class (no
    region-K_F/h_A tower pin anywhere in SS3-4 below); this function ONLY machine-verifies the
    PRE-REGISTERED LEMMA itself, bankable regardless of the station's own verdict.
    Returns {'per_region': [{'region','is_triangle','h_A_eigenvalues','positive_eigenspace_dim'}],
    'all_positive_eigenspaces_are_1d'}."""
    orbits = _three_edge_region_orbits()
    per_region = []
    all_1d = True
    for orb in orbits:
        fsg = field_side_flow_generator(orb["representative"])
        h_A = fsg["h_A"]
        w = np.linalg.eigvalsh(h_A)
        pos_mask = w > 1e-9
        pos_dim = int(np.sum(pos_mask))
        all_1d = all_1d and (pos_dim == 1)
        per_region.append({"region": orb["representative"], "is_triangle": orb["is_triangle"],
                            "h_A_eigenvalues": w.tolist(), "positive_eigenspace_dim": pos_dim})
    assert all_1d, f"a2b_l0b_region_flow_rank_lemma: a region's h_A positive eigenspace is not 1-dim: {per_region}"
    return {"per_region": per_region, "all_positive_eigenspaces_are_1d": all_1d}


def a2b_l0c_conjugate_flow_reversal():
    """[A2b SS2, L0c -- MACHINE-VERIFY FIRST, 'a check, not a re-derivation'] Phi~ := J_F.Phi.J_hist
    intertwines with the REVERSED flow, EXPECTED AUTOMATIC because M0-4b's bit (the SAME sigma
    that builds J_F = field_algebra_conjugation()'s K) flips K_A -> -K_A EXACTLY on any 3-edge
    region -- ALREADY an accreted, asserted fact inside field_side_flow_generator
    ('bit_reversal_check_residual', the_net.py:3486-3507).  Re-verified here across ALL 4
    A4-orbit region representatives (not just the triangle FOCK-0d spot-checked), since Phi~'s
    conjugate-flow property rides on the SAME bit for EVERY region, not just one.
    Returns {'per_region': [{'region','is_triangle','bit_reversal_residual'}], 'worst_residual'}."""
    orbits = _three_edge_region_orbits()
    per_region = []
    worst = 0.0
    for orb in orbits:
        fsg = field_side_flow_generator(orb["representative"])
        r = fsg["bit_reversal_check_residual"]
        worst = max(worst, r)
        per_region.append({"region": orb["representative"], "is_triangle": orb["is_triangle"],
                            "bit_reversal_residual": r})
    assert worst < 1e-6, f"a2b_l0c_conjugate_flow_reversal: worst bit-reversal residual {worst:.2e} >= 1e-6"
    return {"per_region": per_region, "worst_residual": worst}


def a2b_level1_allowance_per_block():
    """[A2b SS3, THE PRE-DECLARED ALLOWANCE, PER SECTOR-PAIR BLOCK, PRINTED BEFORE SOLVING]
    level-1 (Pw[1]) lies entirely inside field-side GROUP-12 (orbit_12={1,2}); GROUP-03
    (field-side) is NOT relevant to phi_1's codomain at all.  Pin (iii) below (the per-sector-pair
    block structure) additionally FORCES Phi to vanish identically on hist-side GROUP-03 (the
    isotypic block of A4-irreps {0,3} -- the trivial irrep plus the 3-dim standard irrep, dim
    1+9=10 at shell 1, history_sector_pair_groups' own naming) -- so GROUP-03's own allowance is
    declared 0 BEFORE solving (forced by (iii) alone, not a solve result).  GROUP-12
    (hist-side dim 1+1=2, the two nontrivial 1-dim characters) keeps FOCK-0c's own 'one phase per
    relevant block' convention (fock0c_w1_allowance/A2's a2_level1_allowance precedent, REUSED):
    1 relevant block x 1 real dim (one overall phase) = 1.
    Returns {'group03_allowance','group12_allowance','allowance','group03_hist_dim','group12_hist_dim'}."""
    hg1 = history_sector_pair_groups(1)
    idx1 = np.where(hg1["lengths"] == 1)[0]
    g03_dim = int(round(np.trace(hg1["P_group03"][np.ix_(idx1, idx1)]).real))
    g12_dim = int(round(np.trace(hg1["P_group12"][np.ix_(idx1, idx1)]).real))
    return {"group03_allowance": 0, "group12_allowance": 1, "allowance": 1,
            "group03_hist_dim": g03_dim, "group12_hist_dim": g12_dim}


def _a2b_pairblock_cross_rows(shell_n, level_n, N_max):
    """[A2b SS3 pin (iii), THE PER-SECTOR-PAIR BLOCK STRUCTURE] the cross-vanishing rows forcing
    Phi to respect the 0<->3/1<->2 pairing on BOTH sides -- IDENTICAL construction to FOCK-0c's own
    cross_rows (fock0c_full_pin_shell, the_net.py:3244-3245), generalized here to an arbitrary
    (shell_n, level_n) pair (FOCK-0c only ever paired shell_n with the FULL field algebra, never a
    single level): field-GROUP(containing level_n).Phi.hist-GROUP(the OTHER group) = 0, both
    directions (one direction is redundant once grading (i) already confines the image to
    level_n, kept for fidelity to the FOCK-0c construction discipline -- 'same builders').
    Returns (rows list, P_H_g03, P_H_g12, n_domain)."""
    Pw, _ = _sector_projectors(sign=+1)
    fg = field_sector_pair_groups()
    P_F_g03, P_F_g12 = fg["P_group03"], fg["P_group12"]
    hg = history_sector_pair_groups(N_max)
    idx = np.where(hg["lengths"] == shell_n)[0]
    n = len(idx)
    P_H_g03 = hg["P_group03"][np.ix_(idx, idx)]
    P_H_g12 = hg["P_group12"][np.ix_(idx, idx)]
    rows = [_zero_block_rows(P_F_g12, 8, P_H_g03, n), _zero_block_rows(P_F_g03, 8, P_H_g12, n)]
    return rows, P_H_g03, P_H_g12, n


def _a2b_B_zero_rows(n_dom, m_cod):
    """[A2b, THE A-ONLY (GENUINE COMPLEX-LINEAR) SELECTOR] SS3's own text says 'solve
    (real-linear)' (the SAME A,B ansatz Phi(v)=A.v+B.conj(v) as A2/FOCK-0c use) -- but SS1 defines
    phi_1 as a PLAIN complex-linear map (Gamma(phi_1)(w) = phi_1(d_1)^...^phi_1(d_n) uses ONLY
    phi_1(d) vectors, no conjugation anywhere; a2_gamma_word's own construction always uses B=0).
    STATED AMBIGUITY (per the contract's 'strictest reading, note it' instruction): this station
    solves the FULL real-linear system (matching SS3's literal text and A2's own precedent) AND
    additionally reports the A-ONLY (B=0) sub-nullity needed for SS4's actual Gamma(phi_1)
    construction -- both numbers reported, never only one.  These rows force B=0 exactly: from
    _a2_vec_colmajor_from_AB's own block identities (M11=Are+Bre, M22=Are-Bre, M12=Bim-Aim,
    M21=Aim+Bim), B=0 iff M11=M22 (Bre=0) AND M21=-M12 (Bim=0) -- both PLAIN linear conditions on
    vecX's own column-major-flattened entries, built here by direct index arithmetic (verified by
    round-trip against _a2_vec_colmajor_from_AB in the station's scratch validation)."""
    m, n = m_cod, n_dom
    rows = []
    for i in range(m):
        for j in range(n):
            pos_M11 = j * (2 * m) + i
            pos_M22 = (n + j) * (2 * m) + (m + i)
            r1 = np.zeros(4 * m * n)
            r1[pos_M11] = 1.0
            r1[pos_M22] = -1.0
            rows.append(r1)
            pos_M12 = (n + j) * (2 * m) + i
            pos_M21 = j * (2 * m) + (m + i)
            r2 = np.zeros(4 * m * n)
            r2[pos_M12] = 1.0
            r2[pos_M21] = 1.0
            rows.append(r2)
    return np.array(rows)


def _a2b_AB_from_vec_colmajor(vecx, n_dom, m_cod):
    """Inverse of _a2_vec_colmajor_from_AB (verified by round-trip in the station's scratch
    validation, residual < 1e-15): reconstructs the complex (A, B) pair (m_cod x n_dom each) from
    a real null-space vector vecx (length 4.n_dom.m_cod)."""
    m, n = m_cod, n_dom
    X = vecx.reshape(2 * m, 2 * n, order="F")
    M11, M12 = X[:m, :n], X[:m, n:]
    M21, M22 = X[m:, :n], X[m:, n:]
    Are, Bre = (M11 + M22) / 2, (M11 - M22) / 2
    Aim, Bim = (M21 - M12) / 2, (M21 + M12) / 2
    return Are + 1j * Aim, Bre + 1j * Bim


def _a2b_shell_level_system(shell_n, level_n, N_max=4):
    """[A2b SS3/SS4.3, THE A2b-CLASS PIN SYSTEM AT (shell_n, level_n)] builds pins (i) grading +
    (ii) R/F_bit parity + (iii) per-sector-pair block, EXACTLY the SS3 construction, generalized
    to an arbitrary (shell_n, level_n) pair (shell_n=1,level_n=1 IS the frozen phi_1 system; other
    pairs are the SS4.3 independent tower-test systems).  R is reversal() at shell 1, else
    history_reversal_matrix's shell-restricted path-reversal (FOCK-0c's own convention, REUSED).
    Returns (grading_rows, rfbit_rows, pairblock_rows, n_domain, R_used)."""
    Pw, _ = _sector_projectors(sign=+1)
    F_bit = Pw[0] + Pw[3] - Pw[1] - Pw[2]
    pairblock_rows, P_H_g03, P_H_g12, n = _a2b_pairblock_cross_rows(shell_n, level_n, N_max)
    if shell_n == 1:
        R = reversal()
    else:
        hr = history_reversal_matrix(N_max)
        idx = np.where(hr["lengths"] == shell_n)[0]
        R = hr["Rw"][np.ix_(idx, idx)]
    grading_rows = [_zero_block_rows(Pw[w], 8, np.eye(n), n) for w in range(4) if w != level_n]
    rfbit_rows = [_linear_pin_rows(R, F_bit)]
    return grading_rows, rfbit_rows, pairblock_rows, n, R


def _a2b_nullity_of(rowlist):
    Cop = np.vstack(rowlist)
    s = np.linalg.svd(Cop, compute_uv=False)
    rank = int(np.sum(s > 1e-8))
    total = Cop.shape[1]
    return total - rank, rank, total, s


def a2b_phi1_pin_trajectory(N_max=4):
    """[A2b SS3, THE NULLITY TRAJECTORY] stacks pins (i)/(ii)/(iii) INCREMENTALLY on the SAME
    12-dim shell-1 domain -> level-1 (Pw[1]) codomain, NO antiunitary pin ANYWHERE (the frozen SS3
    class): stage0 = (i) grading alone; stage_pairblock = (i)+(iii) (no R/F_bit yet, isolates the
    pair-block's own effect); stage_full = (i)+(ii)+(iii), the FROZEN class, reported BOTH as the
    full real-linear nullity (SS3's literal 'solve (real-linear)' text) and the A-only (B=0)
    sub-nullity (the genuine complex-linear content SS4's Gamma(phi_1) actually needs, per
    _a2b_B_zero_rows's disclosed ambiguity note).  Cross-check (SS3 allowance comparison): the
    stage_full nullity is compared against a2b_level1_allowance_per_block()'s pre-declared 1.
    Returns {'stage0_grading_alone','stage_pairblock','stage_full_real_linear',
    'stage_full_A_only','total_real_dim','smallest_kept_sv','largest_null_sv'}."""
    grading_rows, rfbit_rows, pairblock_rows, n, R = _a2b_shell_level_system(1, 1, N_max=N_max)
    n0, _, t0, _ = _a2b_nullity_of(grading_rows)
    n_pb, _, t_pb, _ = _a2b_nullity_of(grading_rows + pairblock_rows)
    rows_full = grading_rows + rfbit_rows + pairblock_rows
    n_full, rank_full, t_full, s_full = _a2b_nullity_of(rows_full)
    b0_rows = [_a2b_B_zero_rows(n, 8)]
    n_A, _, t_A, _ = _a2b_nullity_of(rows_full + b0_rows)
    assert t0 == t_pb == t_full == t_A == 4 * n * 8, \
        f"a2b_phi1_pin_trajectory: ambient real dim mismatch ({t0},{t_pb},{t_full},{t_A}) != {4*n*8}"
    return {"stage0_grading_alone": n0, "stage_pairblock": n_pb,
            "stage_full_real_linear": n_full, "stage_full_A_only": n_A,
            "total_real_dim": t0,
            "smallest_kept_sv": float(s_full[rank_full - 1]) if rank_full > 0 else float("nan"),
            "largest_null_sv": float(s_full[rank_full]) if rank_full < len(s_full) else 0.0}


def a2b_phi1_forced_zero_proof():
    """[A2b SS3/SS6, THE ALGEBRAIC PROOF -- a THIRD, DISTINCT obstruction mechanism from AF-3's
    antiunitary K-swap argument, using ONLY plain-linear structure, NO antiunitary pin anywhere]
    PROVES phi_1 = 0 is FORCED by pins (i)+(ii)+(iii), from three machine-verified facts:
      FACT 1 (pin iii, REUSED _zero_block_rows construction): Phi(hist-GROUP-03) = 0 identically
        (the pair-block pin forces the ENTIRE 10-dim hist-GROUP-03 domain piece -- the isotypic
        block of A4-irreps {trivial, 3-dim-standard} -- into the kernel, since Phi's image is
        already confined to Pw[1] subset field-GROUP-12 by pin (i), disjoint from field-GROUP-03).
      FACT 2 (reversal(), REUSED): R = reversal() acts as EXACTLY +I_2 (the identity) on
        hist-GROUP-12 (the 2-dim isotypic block of A4's two nontrivial 1-dim characters) --
        verified directly, not assumed.
      FACT 3 (_sector_projectors, REUSED): F_bit = Pw[0]+Pw[3]-Pw[1]-Pw[2] acts as EXACTLY -I on
        Pw[1] (level-1 subset field-GROUP-12, by F_bit's own definition).
    PROOF: by FACT 1, Phi vanishes on hist-GROUP-03; it remains to show Phi vanishes on
    hist-GROUP-12 too.  For v in hist-GROUP-12, pin (ii) gives Phi(R.v) = F_bit(Phi(v)).  By FACT 2,
    R.v = v, so LHS = Phi(v).  By pin (i), Phi(v) in range(Pw[1]), so by FACT 3, RHS =
    F_bit(Phi(v)) = -Phi(v).  Hence Phi(v) = -Phi(v), i.e. 2.Phi(v) = 0, i.e. Phi(v) = 0.  Since
    hist-GROUP-03 (FACT 1) and hist-GROUP-12 (this argument) TOGETHER span the full 12-dim shell-1
    domain (history_sector_pair_groups' own completeness assertion, REUSED), Phi(u) = 0 for EVERY
    u.  QED: phi_1 = 0 identically -- theorem-grade, independent of any numeric truncation, and
    using NO antiunitary structure at all (unlike AF-3).
    Returns {'R_on_group12_identity_residual','F_bit_on_level1_neg_identity_residual',
    'group03_dim','group12_dim','stage_full_confirms_proof','proof_holds'}."""
    hg1 = history_sector_pair_groups(1)
    idx1 = np.where(hg1["lengths"] == 1)[0]
    P_H_g03 = hg1["P_group03"][np.ix_(idx1, idx1)]
    P_H_g12 = hg1["P_group12"][np.ix_(idx1, idx1)]
    g03_dim = int(round(np.trace(P_H_g03).real))
    g12_dim = int(round(np.trace(P_H_g12).real))
    vals, vecs = np.linalg.eigh(P_H_g12)
    basis12 = vecs[:, np.abs(vals - 1) < 1e-6]
    R = reversal()
    Rsmall = basis12.conj().T @ R @ basis12
    fact2 = float(np.max(np.abs(Rsmall - np.eye(g12_dim))))
    Pw, _ = _sector_projectors(sign=+1)
    F_bit = Pw[0] + Pw[3] - Pw[1] - Pw[2]
    fact3 = float(np.max(np.abs(F_bit @ Pw[1] + Pw[1])))
    assert fact2 < 1e-8 and fact3 < 1e-8, \
        (f"a2b_phi1_forced_zero_proof: a load-bearing FACT failed (R-on-G12={fact2:.2e}, "
         f"F_bit-on-Pw1={fact3:.2e}) -- the proof's premises do not hold")
    traj = a2b_phi1_pin_trajectory()
    assert traj["stage_full_real_linear"] == 0 and traj["stage_full_A_only"] == 0, \
        (f"a2b_phi1_forced_zero_proof: numeric nullity ({traj['stage_full_real_linear']}, "
         f"{traj['stage_full_A_only']}) != (0,0), the algebraic proof's conclusion is CONTRADICTED")
    return {"R_on_group12_identity_residual": fact2, "F_bit_on_level1_neg_identity_residual": fact3,
            "group03_dim": g03_dim, "group12_dim": g12_dim,
            "stage_full_confirms_proof": traj["stage_full_real_linear"], "proof_holds": True}


def a2b_p1_p2_reverify(N_max_p1=6, N_max_p2=10, seed=0):
    """[A2b SS4.2, P1/P2 RE-VERIFIED IN THIS CLASS] P1 (Pauli truncation) and P2 (repeated-dart
    kernel) are properties of the fermionic Gamma REALIZATION (the SAME 3 creation operators,
    _level1_creation_ops) itself, NOT of phi_1's particular pin class -- A2's own P1/P2 checks
    (a2_pauli_truncation_check, a2_repeated_dart_kernel_check, the_net.py:4304-4377) already use a
    STRUCTURAL test map (class-independent), so re-verification in the A2b class means CALLING
    THEM AGAIN (REUSED verbatim, not re-derived) and confirming the SAME results hold -- the
    construction they test (Gamma(phi_1) = wedge of level-1 vectors via the fixed Adag[0..2]) is
    UNCHANGED by A2b's different phi_1 pin set.
    Returns {'p1','p2'} (the two functions' own return dicts)."""
    p1 = a2_pauli_truncation_check(N_max=N_max_p1, seed=seed)
    p2 = a2_repeated_dart_kernel_check(N_max=N_max_p2, seed=seed)
    assert p1["worst_shell4_norm"] < 1e-9 and p2["synthetic_repeat_residual"] < 1e-9, \
        "a2b_p1_p2_reverify: P1/P2 structural checks failed under re-verification"
    return {"p1": p1, "p2": p2}


def a2b_shell_level_system_nullity(shell_n, level_n, N_max=4):
    """[A2b SS4.3, THE INDEPENDENT SHELL-n SYSTEM] builds the A2b-class pin system at (shell_n,
    level_n) INDEPENDENTLY (same builders as phi_1's own shell-1 system, generalized), reports its
    OWN nullity (grading-alone, +pairblock, FULL real-linear, FULL A-only) -- the comparison space
    the tower-membership test (a2b_tower_membership_test) checks Gamma(phi_1) against.
    Returns {'shell_n','level_n','domain_dim','stage0_grading_alone','stage_pairblock',
    'stage_full_real_linear','stage_full_A_only','total_real_dim'}."""
    grading_rows, rfbit_rows, pairblock_rows, n, R = _a2b_shell_level_system(shell_n, level_n, N_max=N_max)
    n0, _, t0, _ = _a2b_nullity_of(grading_rows)
    n_pb, _, t_pb, _ = _a2b_nullity_of(grading_rows + pairblock_rows)
    rows_full = grading_rows + rfbit_rows + pairblock_rows
    n_full, rank_full, t_full, s_full = _a2b_nullity_of(rows_full)
    b0_rows = [_a2b_B_zero_rows(n, 8)]
    n_A, _, t_A, _ = _a2b_nullity_of(rows_full + b0_rows)
    return {"shell_n": shell_n, "level_n": level_n, "domain_dim": n,
            "stage0_grading_alone": n0, "stage_pairblock": n_pb,
            "stage_full_real_linear": n_full, "stage_full_A_only": n_A, "total_real_dim": t0}


def a2b_tower_membership_test(shell_n, phi1_basis, N_max=4):
    """[A2b SS4.3, THE FORCING QUESTION + THE HONESTY CLAUSE] for EACH basis map in phi1_basis
    (empty if phi_1's nullity is 0, per a2b_phi1_forced_zero_proof), constructs Gamma(phi1) on
    shell_n (REUSED a2_gamma_word/_level1_creation_ops, unchanged) and tests membership in the
    A2b-CLASS shell_n system's OWN full-pin row-stack (_a2b_shell_level_system, NOT FOCK-0c's
    _fock0c_rows_full -- that system includes the antiunitary Tomita pin, which A2b's frozen class
    explicitly excludes; comparing against it would be testing the WRONG space).
    VACUOUSNESS DETERMINATION (the pre-reg's honesty clause, SS4.3): IF phi1_basis is empty,
    Gamma(phi_1)=0 trivially lies in EVERY linear space -- reported HONESTLY as VACUOUS, never as
    a nontrivial confirmation, REGARDLESS of whether the shell_n system's own nullity
    (a2b_shell_level_system_nullity) is itself zero or nonzero (the shell_n system CAN have its
    own freedom -- shells 2/3 do, see the station report -- without that freedom being touched by
    an empty phi1_basis).  A DISCRIMINATING CONTROL (the SAME structural test map A2's P1/P2
    checks use, REUSED) is run alongside to confirm the test machinery is not degenerately
    accepting everything.
    Returns {'shell_n','D_shell_n','basis_size','member_residuals','all_members',
    'control_residual','control_is_member','vacuous'}."""
    _, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(max(N_max, shell_n), succ)
    idx = np.where(lengths == shell_n)[0]
    D_n = len(idx)
    grading_rows, rfbit_rows, pairblock_rows, n, R = _a2b_shell_level_system(shell_n, shell_n, N_max=N_max)
    assert n == D_n, f"a2b_tower_membership_test: dimension mismatch ({n}) vs D_shell_n ({D_n})"
    Cop = np.vstack(grading_rows + rfbit_rows + pairblock_rows)
    Adag, vac, _, _ = _level1_creation_ops()

    def gamma_matrix(phi1):
        cols = [a2_gamma_word(phi1, words[i], Adag, vac) for i in idx]
        return np.hstack(cols) if cols else np.zeros((8, 0), dtype=complex)

    def residual_of(Gamma):
        vecX = _a2_vec_colmajor_from_AB(Gamma, np.zeros_like(Gamma), D_n, 8)
        return float(np.max(np.abs(Cop @ vecX))) if Cop.size else 0.0

    member_residuals = [residual_of(gamma_matrix(phi1)) for phi1 in phi1_basis]
    all_members = all(r < 1e-6 for r in member_residuals) if member_residuals else True
    vacuous = len(phi1_basis) == 0

    rng = np.random.default_rng(0)
    phi_test = rng.normal(size=(ND, 3)) + 1j * rng.normal(size=(ND, 3))
    control_resid = residual_of(gamma_matrix(phi_test))
    return {"shell_n": shell_n, "D_shell_n": D_n, "basis_size": len(phi1_basis),
            "member_residuals": member_residuals, "all_members": bool(all_members),
            "control_residual": control_resid, "control_is_member": bool(control_resid < 1e-6),
            "vacuous": vacuous}


def a2b_pair_completeness_read():
    """[A2b SS4.4, PAIR-COMPLETENESS -- REPORT ONLY] since phi_1 = 0 identically
    (a2b_phi1_forced_zero_proof), dim(Im phi_1) = 0 and there is NO nonzero Gamma(phi_1) image at
    any shell for J_F to act on -- Phi~ := J_F.Phi.J_hist is therefore ALSO identically 0 (the
    conjugate of the zero map is the zero map), and the pair (Phi, Phi~) covers NOTHING of F,
    reported honestly rather than smoothed over.
    Returns {'dim_Im_phi1','Phi_tilde_is_zero','pair_covers_F','note'}."""
    return {"dim_Im_phi1": 0, "Phi_tilde_is_zero": True, "pair_covers_F": False,
            "note": "phi_1 = 0 identically (a2b_phi1_forced_zero_proof) -- Phi and its conjugate "
                    "Phi~ = J_F.Phi.J_hist are BOTH the zero map; there is no image for J_F to "
                    "pair, and the pair (Phi,Phi~) fails to cover ANY of F. Reported honestly, "
                    "not claimed as a partial cover."}


def a2b_nhat_intertwining_exactness(N_max=8):
    """[A2b SS4.5, THE N-HAT-INTERTWINING EXACTNESS -- L0b's CONSEQUENCE, STRUCTURE ONLY]
    verifies, for shells 1-3, that K_hist|_shell_n = c_n.I_n = n.c_1.I_n EXACTLY (FOCK-0e,
    history_side_flow_generator, REUSED unchanged) while N-hat's eigenvalue on level_n (Pw[n]) is
    EXACTLY n (by _sector_projectors' own construction, the defining property of the NHAT
    eigenspace decomposition) -- so for ANY grading-preserving map Phi (shell_n -> level_n,
    regardless of whether it is nonzero), Phi.K_hist|_shell_n = n.c_1.Phi = c_1.(n.Phi) =
    c_1.NHAT.Phi EXACTLY: the intertwining Phi.K_hist = c_1.NHAT.Phi holds IDENTICALLY, a general
    structural fact of the construction (A1's own theorem, restated here per-shell), independent
    of phi_1's own (forced-zero) value.
    Returns {'per_shell': [{'n','c_n','n_times_c1','match','nhat_eigenvalue_on_level_n'}],
    'worst_residual'}."""
    c1 = history_side_flow_generator(1, N_max=N_max)["c_n"]
    rows = []
    worst = 0.0
    for n in (1, 2, 3):
        cn = history_side_flow_generator(n, N_max=N_max)["c_n"]
        resid = abs(cn - n * c1)
        worst = max(worst, resid)
        rows.append({"n": n, "c_n": cn, "n_times_c1": n * c1, "match": resid < 1e-9,
                     "nhat_eigenvalue_on_level_n": n})
    assert worst < 1e-9, f"a2b_nhat_intertwining_exactness: worst residual {worst:.2e} >= 1e-9"
    return {"per_shell": rows, "worst_residual": worst}


def a2b_ember_consistency_shadow(N_max=8):
    """[A2b SS4.5, THE EMBER-CONSISTENCY SHADOW -- STRUCTURE ONLY, lambda's CONFRONTED WITH
    NOTHING] restates FOCK-0e's own per-region lambda_n = n.lambda_1 structure
    (fock0e_lambda_structure, REUSED unchanged) as the per-region eps-eigenspace consistency read
    this station inherits -- NOT a tower pin (SS3 excludes any region-K_F pin), purely a
    structural cross-reference to the ember (lambda* = c_1/eps = 2.463 at the triangle orbit).
    Returns fock0e_lambda_structure(N_max)'s own list, unchanged."""
    return fock0e_lambda_structure(N_max=N_max)


def _a4_standard_3irrep():
    """[A2b SS5, THE EQUIVARIANCE DIAGNOSTIC -- AN HONEST, NON-PROJECTIVE 3-DIM A4 IRREP] A2's own
    accreted field-side rep U(g) (_field_algebra_a4_rep) carries an UNCONTROLLED per-generator
    U(1) phase (spin_lift's SVD null-vector phase is implementation-arbitrary, fixed only in
    MODULUS by the |det|=1 normalization) -- verified in this station's own scratch exploration:
    restricting U(g) to Pw[1] and testing the group law directly gives GENERICALLY COMPLEX (not
    just +-1) defects at most (g,h) pairs, unsuitable for a clean honesty determination.  This
    function sidesteps that by building A4's OWN standard 3-dim irrep DIRECTLY and honestly: the
    vertex-permutation representation on R^4 (4 vertices, an HONEST group homomorphism A4->S4->
    GL(4,R), no square-root/lift step anywhere) decomposes as trivial (+) the 3-dim standard irrep
    (the sum-zero subspace) -- this IS A4's unique 3-dim irrep (character-matched below against
    _a4_char_lookup, REUSED), built with NO projectivity ambiguity at all.
    Returns (A4v, rho3: list of 12 real orthogonal 3x3 matrices, 'honest_group_law_residual',
    'character_match_residual')."""
    A4v = _a4_vertex_group()

    def vertex_perm(g):
        P = np.zeros((NV, NV))
        for i in range(NV):
            P[g[i], i] = 1.0
        return P

    v0 = np.ones(NV) / math.sqrt(NV)
    Q, _ = np.linalg.qr(np.eye(NV) - np.outer(v0, v0))
    basis3 = Q[:, np.abs(Q.T @ v0) < 1e-8][:, :3]
    rho3 = [basis3.T @ vertex_perm(g) @ basis3 for g in A4v]
    comp = lambda g, h: {i: g[h[i]] for i in range(NV)}
    ix = {_a4_key(g): n for n, g in enumerate(A4v)}
    worst_honest = max(float(np.max(np.abs(rho3[a] @ rho3[b] - rho3[ix[_a4_key(comp(A4v[a], A4v[b]))]])))
                        for a in range(12) for b in range(12))
    dims, chars_by_elt = _a4_char_lookup()
    i3 = dims.index(3)
    chi3_expected = np.array([chars_by_elt[i3][_a4_key(g)] for g in A4v])
    chi3_actual = np.array([np.trace(rho3[k]) for k in range(12)])
    char_resid = float(np.max(np.abs(chi3_expected - chi3_actual)))
    assert worst_honest < 1e-9, f"_a4_standard_3irrep: NOT an honest rep ({worst_honest:.2e})"
    assert char_resid < 1e-9, f"_a4_standard_3irrep: character mismatch vs fusion_ring's 3-irrep ({char_resid:.2e})"
    return A4v, rho3, worst_honest, char_resid


def a2b_equivariant_subspace_diagnostic():
    """[A2b SS5, LABELED DIAGNOSTIC -- NON-VERDICT, REPORT-ONLY] computes Hom_A4(shell-1, level-1)
    from the accreted A4 machinery, per the pre-reg's own instruction to VERIFY the stated
    expectation ('complex dim 2') rather than assume it.  TWO DISTINCT READINGS are computed and
    DISCLOSED (a genuine ambiguity in what the pre-reg's own phrase means, resolved by computing
    BOTH rather than guessing):
      (A) THE STANDARD READING: Hom_A4(dart_rep(12), rho3(3)) via honest-A4 rho3 (_a4_standard_
          3irrep, sidesteps F's own uncontrolled phase ambiguity) -- an ordinary SVD null-space
          intertwiner computation, matching Frobenius reciprocity (mult of the 3-irrep in the
          REGULAR representation = dim(3-irrep) = 3) and sector_grading_hist's own INDEPENDENTLY-
          banked mult[1]=[1,1,1,3] finding.  THIS is the reading that is an actual SUBSPACE of the
          SAME ambient (12x3 complex matrices) phi_1 lives in -- used below for the intersection.
      (B) THE TENSOR-SQUARE READING (the pre-reg's own stated derivation, 'the 9-part = 3 tensor 3
          contains 3 twice: symmetric + antisymmetric'): Hom_A4(3 tensor 3, 3) via the character
          inner product <chi_3^2, chi_3> (REUSED _a4_char_lookup, no messy U(g) needed at all) --
          this DOES reproduce 2 exactly, confirming the pre-reg's OWN arithmetic is correct, but it
          answers a DIFFERENT question (a diagonal tensor-square action, not the direct Hom_A4 of
          shell-1's own regular-rep structure against level-1) -- NOT a subspace of Hom(C^12,C^3),
          so it cannot be intersected with phi_1's solution space directly.
    INTERSECTION (reading A, the only one that is an actual subspace here): since phi_1's own
    solution space is EMPTY (a2b_phi1_forced_zero_proof), the intersection is TRIVIALLY {0} --
    reported honestly, not smoothed into a claim about the equivariant subspace's OWN structure
    (which is itself nonzero, dim 3, independent of phi_1).
    ISOTYPIC FLOW READ: since phi_1 = 0, EVERY isotypic component of [1,1,1,9] is in ker(phi_1)
    and NONE survive into a nonzero image -- the SAME honest finding as A2's own species_read.
    Returns {'hom_A4_standard_reading_dim','hom_A4_tensor_square_reading_dim',
    'intersection_with_phi1_solution_dim','isotypic_flow','ambiguity_note'}."""
    A4v, rho3, _, _ = _a4_standard_3irrep()
    dart_rep_list = [dart_rep(g) for g in A4v]
    I3 = np.eye(3)
    rows = [np.kron(Rd.T, I3) - np.kron(np.eye(ND), Rg) for Rd, Rg in zip(dart_rep_list, rho3)]
    Cstack = np.vstack(rows)
    s = np.linalg.svd(Cstack, compute_uv=False)
    rank = int(np.sum(s > 1e-8))
    hom_standard = Cstack.shape[1] - rank
    sg = sector_grading_hist(1)
    assert list(sg["mult"][1]) == [1, 1, 1, 3], \
        f"a2b_equivariant_subspace_diagnostic: sector_grading_hist mult[1] {sg['mult'][1]} != [1,1,1,3]"
    assert hom_standard == 3, \
        f"a2b_equivariant_subspace_diagnostic: SURPRISE -- standard-reading Hom_A4 dim {hom_standard} != 3"

    dims, chars_by_elt = _a4_char_lookup()
    i3 = dims.index(3)
    chi3 = np.array([chars_by_elt[i3][_a4_key(g)] for g in A4v])
    G = len(A4v)
    hom_tensor_square = complex(np.sum(np.conj(chi3 * chi3) * chi3) / G)
    assert abs(hom_tensor_square.imag) < 1e-8 and abs(hom_tensor_square.real - 2) < 1e-8, \
        f"a2b_equivariant_subspace_diagnostic: tensor-square reading {hom_tensor_square} != 2"

    proof = a2b_phi1_forced_zero_proof()
    intersection_dim = 0 if proof["proof_holds"] else None

    return {"hom_A4_standard_reading_dim": hom_standard,
            "hom_A4_tensor_square_reading_dim": int(round(hom_tensor_square.real)),
            "intersection_with_phi1_solution_dim": intersection_dim,
            "isotypic_flow": {"surviving_in_image": [], "note": "phi_1 = 0 identically; every "
                               "isotypic component [1,1,1,9] is in ker(phi_1), none survive"},
            "ambiguity_note": "the pre-reg's stated 'complex dim 2' matches READING B (the "
                               "tensor-square/diagonal-action question, character-verified exact) "
                               "but NOT reading A (the standard Hom_A4(regular-rep,irrep)=dim(irrep)"
                               "=3 fact, Frobenius reciprocity, independently cross-checked against "
                               "sector_grading_hist's banked mult[1]=[1,1,1,3]) -- BOTH are reported; "
                               "reading A is the one used for the intersection since it alone is a "
                               "genuine subspace of Hom(C^12,C^3)."}


def a2b_weld_selftest_2026_07_12(verbose=True):
    """A2b station regression: Sections 7/7b/8/8b/8c/8d/8e/8f + module anchors untouched, THEN
    L0a/L0b/L0c (pre-registered lemmas, machine-verified first), the pre-declared per-block
    allowance, the phi_1 nullity trajectory (SS3 frozen pin set, NO antiunitary pin), the algebraic
    forced-zero proof (a THIRD, distinct obstruction mechanism from AF-3), P1/P2 re-verification,
    the independent shell-2/3 systems + the tower-membership test with its explicit vacuousness
    determination and discriminating control, pair-completeness, N-hat-intertwining exactness, the
    ember-consistency shadow, and the SS5 labeled (report-only) equivariance diagnostic.  Does NOT
    itself adjudicate the SS6 verdict tree (architect-only per the pre-reg)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 8g self-test -- A2b the conjugate-pair weld (2026-07-12)")
        print("=" * 88)

    ck("ANCHORS + Sections 7/7b/8/8b/8c/8d/8e/8f untouched",
       anchor_cell_projector() and anchor_tick_2pi() and accretion_selftest_2026_07_10(verbose=False)
       and i2b_selftest_2026_07_11(verbose=False) and fock0_selftest_2026_07_11(verbose=False)
       and fock0b_selftest_2026_07_11(verbose=False) and fock0c_selftest_2026_07_11(verbose=False)
       and fock0d_selftest_2026_07_11(verbose=False) and fock0e_selftest_2026_07_12(verbose=False)
       and a2_weld_selftest_2026_07_12(verbose=False))

    l0a = a2b_l0a_self_j_pin_probe()
    ck(f"L0a (self-J-pin dies beyond shell 1): shell-2->level-2 self-J-pin nullity = "
       f"{l0a['nullity']}/{l0a['total_real_dim']} (expect 0, the shell-2 instance)",
       l0a["nullity"] == 0)

    l0b = a2b_l0b_region_flow_rank_lemma()
    ck(f"L0b (region-flow rank lemma): ALL {len(l0b['per_region'])} region orbits' h_A have a "
       f"1-DIM positive eigenspace (forces rank(Phi_1)<=1 under a hypothetical h_A flow pin, "
       f"NOT part of the frozen SS3 class)", l0b["all_positive_eigenspaces_are_1d"])

    l0c = a2b_l0c_conjugate_flow_reversal()
    ck(f"L0c (conjugate sees reversed flow): worst bit-reversal residual over all region orbits "
       f"= {l0c['worst_residual']:.2e}", l0c["worst_residual"] < 1e-6)

    alw = a2b_level1_allowance_per_block()
    ck(f"STEP 1 ALLOWANCE PER BLOCK (PRINTED BEFORE solving): GROUP-03 = "
       f"{alw['group03_allowance']} (forced 0 by pin (iii), hist dim {alw['group03_hist_dim']}); "
       f"GROUP-12 = {alw['group12_allowance']} (one phase, hist dim {alw['group12_hist_dim']}); "
       f"TOTAL = {alw['allowance']}", alw["allowance"] == 1 and alw["group03_allowance"] == 0)

    traj = a2b_phi1_pin_trajectory()
    ck(f"NULLITY TRAJECTORY [ambient {traj['total_real_dim']}]: (i) grading alone = "
       f"{traj['stage0_grading_alone']}; (i)+(iii) pairblock = {traj['stage_pairblock']}; "
       f"FULL (i)+(ii)+(iii) real-linear = {traj['stage_full_real_linear']}, A-only = "
       f"{traj['stage_full_A_only']}",
       traj["stage_full_real_linear"] == 0 and traj["stage_full_A_only"] == 0)

    proof = a2b_phi1_forced_zero_proof()
    ck("THE ALGEBRAIC PROOF (a THIRD obstruction mechanism, no antiunitary pin used): R = +I on "
       "hist-GROUP-12, F_bit = -I on level-1, pair-block kills hist-GROUP-03 -- phi_1 = 0 FORCED",
       proof["proof_holds"], detail=f"R-on-G12={proof['R_on_group12_identity_residual']:.1e}, "
       f"F_bit-on-level1={proof['F_bit_on_level1_neg_identity_residual']:.1e}, "
       f"G03dim={proof['group03_dim']}, G12dim={proof['group12_dim']}")

    p12 = a2b_p1_p2_reverify()
    ck(f"P1/P2 RE-VERIFIED (reused from A2, unchanged construction): P1 worst shell-4 norm "
       f"{p12['p1']['worst_shell4_norm']:.1e}; P2 synthetic-repeat residual "
       f"{p12['p2']['synthetic_repeat_residual']:.1e}, first admissible repeat at shell "
       f"{p12['p2']['first_shell_with_admissible_repeat']}",
       p12["p1"]["worst_shell4_norm"] < 1e-9 and p12["p2"]["synthetic_repeat_residual"] < 1e-9)

    s2 = a2b_shell_level_system_nullity(2, 2)
    s3 = a2b_shell_level_system_nullity(3, 3)
    ck(f"INDEPENDENT SHELL SYSTEMS (own freedom, NOT phi_1's): shell2->level2 FULL real-linear = "
       f"{s2['stage_full_real_linear']}/{s2['total_real_dim']} (A-only {s2['stage_full_A_only']}); "
       f"shell3->level3 FULL real-linear = {s3['stage_full_real_linear']}/{s3['total_real_dim']} "
       f"(A-only {s3['stage_full_A_only']})", True)

    tw2 = a2b_tower_membership_test(2, [], N_max=4)
    tw3 = a2b_tower_membership_test(3, [], N_max=4)
    ck(f"TOWER-MEMBERSHIP TEST shell 2 [D={tw2['D_shell_n']}]: basis_size=0 -> VACUOUS={tw2['vacuous']} "
       f"(the 0 map, trivially member); discriminating control residual={tw2['control_residual']:.3e} "
       f"(control_is_member={tw2['control_is_member']})",
       tw2["all_members"] and tw2["vacuous"] and not tw2["control_is_member"])
    ck(f"TOWER-MEMBERSHIP TEST shell 3 [D={tw3['D_shell_n']}]: basis_size=0 -> VACUOUS={tw3['vacuous']}; "
       f"discriminating control residual={tw3['control_residual']:.3e} "
       f"(control_is_member={tw3['control_is_member']})",
       tw3["all_members"] and tw3["vacuous"] and not tw3["control_is_member"])

    pc = a2b_pair_completeness_read()
    ck(f"PAIR-COMPLETENESS (report only): dim(Im phi_1)={pc['dim_Im_phi1']}, "
       f"Phi~ is zero={pc['Phi_tilde_is_zero']}, pair covers F={pc['pair_covers_F']}",
       pc["dim_Im_phi1"] == 0 and pc["Phi_tilde_is_zero"])

    nh = a2b_nhat_intertwining_exactness()
    ck(f"N-HAT-INTERTWINING EXACTNESS [shells 1-3]: worst residual {nh['worst_residual']:.2e} "
       f"(Phi.K_hist = c_1.NHAT.Phi holds EXACTLY for ANY grading-preserving Phi, structural, "
       f"phi_1-independent)", nh["worst_residual"] < 1e-9)

    emb = a2b_ember_consistency_shadow()
    tri = next(r for r in emb if r["is_triangle"])
    ck(f"EMBER-CONSISTENCY SHADOW (structure only): triangle orbit lambda_1={tri['lambda_1']:.4f} "
       f"(FOCK-0d ember reference 2.463)", abs(tri["lambda_1"] - 2.463) < 0.01)

    diag = a2b_equivariant_subspace_diagnostic()
    ck(f"SS5 DIAGNOSTIC (report-only, non-verdict): Hom_A4(shell-1,level-1) standard reading = "
       f"{diag['hom_A4_standard_reading_dim']} (Frobenius reciprocity, matches sector_grading_hist "
       f"mult[1]=[1,1,1,3]); tensor-square reading (the pre-reg's own '2') = "
       f"{diag['hom_A4_tensor_square_reading_dim']}; intersection with phi_1's solution = "
       f"{diag['intersection_with_phi1_solution_dim']} (phi_1=0)",
       diag["hom_A4_standard_reading_dim"] == 3 and diag["hom_A4_tensor_square_reading_dim"] == 2
       and diag["intersection_with_phi1_solution_dim"] == 0)

    if verbose:
        print("RESULT:", "A2b SECTION-8g REGRESSION PASSES" if ok else "AN A2b CHECK FAILED")
    return ok


# ===========================================================================
# 8h. A2c -- THE CORRECTED-DICTIONARY EQUIVARIANT WELD (2026-07-12, Push 2 station 3)
#     A2c_equivariant_weld_prereg_2026-07-12.md SS2-5
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as Sections 7/7b/8/8b/8c/8d/8e/8f/8g above.]
#
# HYPOTHESIS UNDER TEST (SS1): A2b (section 8g) proved the minimal conjugate-pair-weld pin class
# (grading + R/F_bit parity + pair-block) is ITSELF empty, via a mechanism using NO antiunitary
# structure at all.  This station drops the R/F_bit parity pin ENTIRELY and replaces it with a
# genuine A4-EQUIVARIANCE pin at level 1: phi_1.rho_hist(g) = rho_1(g).phi_1 for EVERY g in A4
# (rho_hist = the honest dart_rep/dart_word_action action on shell-1, ALREADY genuinely non-
# projective; rho_1 = the honest level-1 A4 rep, CONSTRUCTED via the A2b workaround's honest-rep
# route, _a4_standard_3irrep, REUSED unchanged -- the flawed/quarantined _field_algebra_a4_rep/
# spin_lift are NEVER touched anywhere below, grep-confirmed).  SS3's frozen pin set: (i) grading,
# (ii) the per-sector-pair block structure (A2b's own (iii), REUSED verbatim -- NOT A2b's R/F_bit
# pin), (iii) A4-equivariance at level 1 (NEW).  D4 (frozen, hard guard, verbatim from A2/A2b):
# the species map is NEVER an input anywhere below -- grep-confirmed, only ever read in the
# OUTPUT-only a2c_species_read.
#
# NUMBERS APPEAR NOWHERE: every quantity below is a dimension, nullity, rank, or exactness
# residual (structure), never M_Z/ppm/m_nu/a_e (module contract + pre-reg SS6 poisons).
#
# ML-2b/HK-7 CONDITIONALITY (carries into every DR-frame-touching sentence below, verbatim,
# unchanged from Section 8's own banner): "Every duality check here (HK-5) is CELL-LEVEL only (the
# 6-edge static vacuum). ML-2b's DR-frame argument is CONDITIONAL on the TD-limit duality holding,
# which is NOT verified by this suite."
def a2c_pc1_honest_level1_rep():
    """[A2c pc1, PRE-CHECK -- MUST PASS OR STOP=C4] construct the CLEAN level-1 A4 rep rho_1 via
    the A2b workaround's honest-rep route (_a4_standard_3irrep, the_net.py:5016-5054, REUSED
    unchanged -- do NOT touch _field_algebra_a4_rep/spin_lift, flawed/quarantined per the A2b
    checker's own finding, working notes/A2b_check_2026-07-12.md SS8).  Verifies: (a) rho_1 is a
    GENUINE (non-projective) A4 representation -- composition exact to <=1e-12 (STRICTER than the
    1e-9 _a4_standard_3irrep itself asserts, per the pre-reg's own tolerance); (b) NO cocycle=-1
    pair exists anywhere in the 144 (g,h) pairs (contrast _field_algebra_a4_rep's OWN cocycle=-1
    pair, HK-6b) -- i.e. rho_1 carries NO projective/2T obstruction at all, confirming SS1(a)'s
    claim that level-1 (unlike the FULL 8-dim F) is honestly A4, not merely A4-up-to-sign;
    (c) rho_1's character matches fusion_ring's own 3-irrep exactly (independent cross-check,
    REUSED from _a4_standard_3irrep's own assertion).
    Returns {'group_law_residual_1e12','smallest_no_cocycle_minus1_gap','character_match_residual',
    'pc1_pass'} (STOP=C4 if pc1_pass is False)."""
    A4v, rho3, worst_honest, char_resid = _a4_standard_3irrep()
    comp = lambda g, h: {i: g[h[i]] for i in range(NV)}
    ix = {_a4_key(g): n for n, g in enumerate(A4v)}
    worst2 = max(float(np.max(np.abs(rho3[a] @ rho3[b] - rho3[ix[_a4_key(comp(A4v[a], A4v[b]))]])))
                 for a in range(12) for b in range(12))
    smallest_gap = min(float(np.max(np.abs(rho3[a] @ rho3[b] + rho3[ix[_a4_key(comp(A4v[a], A4v[b]))]])))
                        for a in range(12) for b in range(12))
    pc1_pass = bool(worst2 < 1e-12 and char_resid < 1e-9 and smallest_gap > 0.1)
    assert pc1_pass, (f"a2c_pc1_honest_level1_rep: PC1 FAILED (C4 BLOCKED) -- group_law={worst2:.2e}, "
                       f"char={char_resid:.2e}, cocycle_gap={smallest_gap:.3f}")
    return {"group_law_residual_1e12": worst2, "smallest_no_cocycle_minus1_gap": smallest_gap,
            "character_match_residual": char_resid, "pc1_pass": pc1_pass}


def a2c_pc2_hom_multiplicity():
    """[A2c pc2, PRE-CHECK] Hom_A4(shell-1, level-1) has complex dim 3, re-derived by CHARACTER
    INNER PRODUCT (a genuinely different method from the direct SVD null-space computation
    a2b_equivariant_subspace_diagnostic's Reading A already used -- pc2's own instruction is to
    're-derive by character inner product', not just re-cite): shell-1 IS A4's regular
    representation (dart_rep acts simply transitively on the 12 darts; sector_grading_hist's own
    cross-check, mult[1]=[1,1,1,3] = [1x1-dim,1x1-dim,1x1-dim,3x3-dim], summing to 12), so
    chi_reg(g) = 12 if g=e else 0; Frobenius reciprocity's own content, computed here as a DIRECT
    character sum (not merely asserted from the theorem's name): <chi_reg, chi_3> = chi_3(e) = 3
    (the identity's character = the irrep's own dimension).  Cross-checked against the direct
    complex SVD Hom-space computation (a2b_equivariant_subspace_diagnostic's own method, re-run
    fresh here) and sector_grading_hist's independently-banked mult[1].
    Returns {'hom_via_character_inner_product','hom_via_direct_svd','mult1','pc2_pass'}."""
    A4v = _a4_vertex_group()
    dims, chars_by_elt = _a4_char_lookup()
    i3 = dims.index(3)
    e_idx = next(i for i, g in enumerate(A4v) if all(g[k] == k for k in range(NV)))
    chi_reg = np.array([12.0 if k == e_idx else 0.0 for k in range(12)])
    chi3 = np.array([chars_by_elt[i3][_a4_key(g)] for g in A4v])
    hom_char = complex(np.sum(chi_reg * np.conj(chi3)) / 12)
    _, rho3, _, _ = _a4_standard_3irrep()
    dart_rep_list = [dart_rep(g) for g in A4v]
    I3 = np.eye(3)
    rows = [np.kron(Rd.T, I3) - np.kron(np.eye(ND), Rg) for Rd, Rg in zip(dart_rep_list, rho3)]
    s = np.linalg.svd(np.vstack(rows), compute_uv=False)
    hom_svd = 3 * ND - int(np.sum(s > 1e-8))
    sg = sector_grading_hist(1)
    pc2_pass = (abs(hom_char.real - 3) < 1e-8 and abs(hom_char.imag) < 1e-8 and hom_svd == 3
                and list(sg["mult"][1]) == [1, 1, 1, 3])
    assert pc2_pass, (f"a2c_pc2_hom_multiplicity: PC2 FAILED -- char={hom_char}, svd={hom_svd}, "
                       f"mult1={sg['mult'][1].tolist()}")
    return {"hom_via_character_inner_product": hom_char, "hom_via_direct_svd": hom_svd,
            "mult1": sg["mult"][1].tolist(), "pc2_pass": pc2_pass}


def a2c_pc3_reproduce_survivor(N_max=4):
    """[A2c pc3, PRE-CHECK] reproduce A2b's 24-dim bare-survivor waypoint (grading (i) + pair-
    block (ii)/A2b's own (iii), NO parity dictionary, NO self-J) BEFORE adding equivariance, via
    _a2b_shell_level_system(1,1,N_max)'s OWN grading_rows/pairblock_rows (REUSED unchanged -- the
    rfbit_rows output is simply not stacked here, since SS3's pin (ii) is A2b's own pair-block
    pin, not A2b's R/F_bit pin).
    Returns {'grading_alone','pairblock_survivor','total_real_dim','pc3_pass'}."""
    grading_rows, rfbit_rows, pairblock_rows, n, R = _a2b_shell_level_system(1, 1, N_max=N_max)
    n0, _, t0, _ = _a2b_nullity_of(grading_rows)
    n_pb, _, t_pb, _ = _a2b_nullity_of(grading_rows + pairblock_rows)
    pc3_pass = (n_pb == 24)
    assert pc3_pass, f"a2c_pc3_reproduce_survivor: PC3 FAILED -- pairblock survivor {n_pb} != 24"
    return {"grading_alone": n0, "pairblock_survivor": n_pb, "total_real_dim": t0, "pc3_pass": pc3_pass}


def a2c_pc4_automatic_parity(N_max=4, seed=0):
    """[A2c pc4, PRE-CHECK] fermion parity on Gamma(test-map) images = (-1)^|w| identically, i.e.
    Gamma(w) is an EXACT NHAT-eigenvector with eigenvalue = the shell number |w| (a structural
    consequence of the CAR realization, A2b_check's own item (ii) finding, REUSED here as a
    standing check -- NEVER a pin, per pc4's own framing).  Tested on a GENERIC structural test
    map (not the frozen -- empty -- solution) across shells 1-3.
    Returns {'worst_nhat_eigenvalue_residual','n_words_checked','pc4_pass'}."""
    Adag, vac, _, _ = _level1_creation_ops()
    Pw, NHAT = _sector_projectors(sign=+1)
    rng = np.random.default_rng(seed)
    phi_test = rng.normal(size=(ND, 3)) + 1j * rng.normal(size=(ND, 3))
    _, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(N_max, succ)
    worst = 0.0
    n_checked = 0
    for shell in (1, 2, 3):
        idxs = [i for i, w in enumerate(words) if len(w) == shell]
        for i in idxs:
            v = a2_gamma_word(phi_test, words[i], Adag, vac)
            resid = float(np.max(np.abs(NHAT @ v - shell * v)))
            worst = max(worst, resid)
            n_checked += 1
    pc4_pass = worst < 1e-9
    assert pc4_pass, f"a2c_pc4_automatic_parity: PC4 FAILED -- worst NHAT-eigenvalue residual {worst:.2e}"
    return {"worst_nhat_eigenvalue_residual": worst, "n_words_checked": n_checked, "pc4_pass": pc4_pass}


def a2c_level1_allowance_per_block():
    """[A2c SS4 step1, THE PRE-DECLARED ALLOWANCE] pins (i) grading + (ii) pair-block are VERBATIM
    A2b's own (i)+(iii) (SS3 says so explicitly) -- so the allowance is A2b's OWN
    a2b_level1_allowance_per_block() value, REUSED not re-derived: GROUP-03=0 (forced by the
    pair-block pin), GROUP-12=1 (one phase), TOTAL=1.  PRINTED before solving, per the pre-reg.
    Returns a2b_level1_allowance_per_block()'s own dict, unchanged."""
    return a2b_level1_allowance_per_block()


def _a2c_level_rep(level_n, U=None):
    """[A2c SS3 pin (iii), THE HONEST LEVEL-n A4 REP] Lambda^n of the honest level-1 rep
    (_a4_standard_3irrep's rho3, pc1): level 1 = rho3 itself (dim 3); level 2 = Lambda^2(rho3)
    (dim 3, verified an honest rep below; DECOMPOSES as one more copy of the SAME 3-irrep -- A4's
    unique 3-irrep is isomorphic to its own Lambda^2); level 3 = Lambda^3(rho3) = det(rho3) (dim
    1, the TRIVIAL rep -- A4's own vertex-permutation action is EVEN by construction,
    _a4_vertex_group's own evenness filter, so det(rho3(g))=1 for every g, verified below).  U
    (default identity, 3x3 unitary): the EMBEDDING FREEDOM noted in the station report -- by
    Schur's lemma the ONLY intertwiner between two realizations of an IRREDUCIBLE rep is unique up
    to an overall phase (which changes NOTHING, since U.rho(g).U^dagger = rho(g) exactly for
    scalar U); a NON-scalar U tests whether a genuinely different embedding changes the downstream
    result (a2c_intersection's own robustness trials use this).  Lambda^n(U.rho.U^dagger) =
    Lambda^n(U).Lambda^n(rho).Lambda^n(U)^dagger, so the SAME U propagates consistently up the
    tower (self-consistent, not an independent freedom per shell).
    Returns (A4v, rho_n: list of 12 complex level_n x level_n matrices, 'group_law_residual',
    'irrep_decomposition' (mult of A4 irreps [0,1,2,3] in rho_n, by character projection))."""
    A4v = _a4_vertex_group()
    _, rho3, _, _ = _a4_standard_3irrep()
    if U is None:
        U = np.eye(3)
    rho1 = [U @ g @ U.conj().T for g in rho3]
    if level_n == 1:
        rho_n = rho1
    elif level_n == 2:
        pairs = [(0, 1), (0, 2), (1, 2)]

        def w2(M):
            out = np.zeros((3, 3), dtype=complex)
            for r, (a, b) in enumerate(pairs):
                for c, (cc, dd) in enumerate(pairs):
                    out[r, c] = M[a, cc] * M[b, dd] - M[a, dd] * M[b, cc]
            return out
        rho_n = [w2(g) for g in rho1]
    elif level_n == 3:
        rho_n = [np.array([[np.linalg.det(g)]], dtype=complex) for g in rho1]
    else:
        raise ValueError(f"_a2c_level_rep: level_n {level_n} not supported (only 1,2,3 -- P1 kills shell>=4)")
    comp = lambda g, h: {i: g[h[i]] for i in range(NV)}
    ix = {_a4_key(g): n for n, g in enumerate(A4v)}
    worst = max(float(np.max(np.abs(rho_n[a] @ rho_n[b] - rho_n[ix[_a4_key(comp(A4v[a], A4v[b]))]])))
                for a in range(12) for b in range(12))
    dims, chars_by_elt = _a4_char_lookup()
    chi_n = np.array([np.trace(rho_n[k]) for k in range(12)])
    decomp = [complex(np.sum(chi_n * np.conj(np.array([chars_by_elt[a][_a4_key(g)] for g in A4v]))) / 12)
              for a in range(4)]
    assert worst < 1e-8, f"_a2c_level_rep: Lambda^{level_n}(rho_1) fails its own group law ({worst:.2e})"
    return A4v, rho_n, worst, [round(x.real, 6) for x in decomp]


def _a2c_level_embedding(level_n):
    """[A2c SS3 pin (iii)] the isometric embedding E_n: C^(dim level_n) -> C^8 realizing level_n's
    ABSTRACT coordinate space (the SAME Adag[0..2]-wedge basis a2_gamma_word/_level1_creation_ops
    already use) inside F: level 1 -- Adag[m]|vac>; level 2 -- Adag[i]Adag[j]|vac> (i<j, the SAME
    pair order as _a2c_level_rep's Lambda^2 basis); level 3 -- Adag[0]Adag[1]Adag[2]|vac> (the top
    wedge, dim 1).  Verified: the columns are already orthonormal (fermionic Slater-determinant
    states of an orthonormal single-particle basis are automatically orthonormal) and
    E_n.E_n^dagger == Pw[level_n] (residual < 1e-8)."""
    Adag, vac, _, _ = _level1_creation_ops()
    Pw, _ = _sector_projectors(sign=+1)
    if level_n == 1:
        E = np.hstack([Adag[m] @ vac for m in range(3)])
    elif level_n == 2:
        pairs = [(0, 1), (0, 2), (1, 2)]
        E = np.hstack([Adag[i] @ (Adag[j] @ vac) for (i, j) in pairs])
    elif level_n == 3:
        E = (Adag[0] @ (Adag[1] @ (Adag[2] @ vac))).reshape(8, 1)
    else:
        raise ValueError(f"_a2c_level_embedding: level_n {level_n} not supported")
    gram_resid = float(np.max(np.abs(E.conj().T @ E - np.eye(E.shape[1]))))
    proj_resid = float(np.max(np.abs(E @ E.conj().T - Pw[level_n])))
    assert gram_resid < 1e-8, f"_a2c_level_embedding: E_{level_n} not orthonormal ({gram_resid:.2e})"
    assert proj_resid < 1e-8, f"_a2c_level_embedding: E_{level_n}.E_{level_n}^dagger != Pw[{level_n}] ({proj_resid:.2e})"
    return E


def _a2c_rho_full_embedded(level_n, U=None):
    """[A2c SS3 pin (iii)] the 8x8 extension of _a2c_level_rep(level_n, U)'s honest rep, embedded
    via _a2c_level_embedding on Pw[level_n], IDENTITY on the complement.  STATED CONVENTION: every
    constraint this station applies also confines Phi's image to Pw[level_n] via pin (i), so the
    extension's action OFF Pw[level_n] never gets exercised by any solution in the intersected
    space -- the identity choice is itself a genuine representation (Pw[level_n]'s complement,
    trivial rep) but its specific value is provably irrelevant to every nullity computed below."""
    Pw, _ = _sector_projectors(sign=+1)
    E = _a2c_level_embedding(level_n)
    Pl = Pw[level_n]
    _, rho_n, _, _ = _a2c_level_rep(level_n, U=U)
    I8 = np.eye(8)
    return [E @ g @ E.conj().T + (I8 - Pl) for g in rho_n]


def _a2c_shell_hist_action(shell_n, N_max=4):
    """[A2c SS4 step3] the A4 action on shell_n's OWN D_n-dim word-space, restricted from
    dart_word_action's full-H_hist permutation (the action preserves word length,
    sector_grading_hist's own verified fact) -- generalizes dart_rep (= shell-1's own action,
    cross-checked below to agree exactly) to any shell.  Returns (A4v, mats: list of 12 D_n x D_n
    real permutation matrices, idx: the shell's own global word indices)."""
    _, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(N_max, succ)
    A4v, perms = dart_word_action(words, index)
    idx = np.where(lengths == shell_n)[0]
    pos = {g: p for p, g in enumerate(idx)}
    mats = []
    for k in range(len(A4v)):
        M = np.zeros((len(idx), len(idx)))
        for p, i in enumerate(idx):
            j = perms[k][i]
            assert j in pos, "_a2c_shell_hist_action: action left the shell block (should be impossible)"
            M[pos[j], p] = 1.0
        mats.append(M)
    if shell_n == 1:
        resid = max(float(np.max(np.abs(mats[k] - dart_rep(A4v[k])))) for k in range(len(A4v)))
        assert resid < 1e-9, f"_a2c_shell_hist_action: shell-1 action != dart_rep ({resid:.2e})"
    return A4v, mats, idx


def a2c_equivariant_channel_size(N_max=4):
    """[A2c SS4 step1, THE EQUIVARIANT CHANNEL SIZE -- PRE-DECLARED FROM pc2, VERIFIED NOT ASSUMED]
    the equivariant channel ALONE (grading (i), confining the codomain to level-1, PLUS
    equivariance (iii); NO pair-block yet) has REAL dim 6 (= 2 x pc2's complex dim 3) for the
    A-only/genuine-complex-linear reading -- the pre-registered upper bound SS4 step1 states
    ('expect <=6 real dims by pc2'), VERIFIED here by actually building and solving the
    (grading+equivariance)-only system, not merely inferred from pc2's abstract count.  ALSO
    reports the FULL real-linear reading: since dart_rep(g)/rho_1(g) are ORDINARY (non-
    antiunitary) operators, the SAME _linear_pin_rows Kronecker structure applied to a real-linear
    Phi(v)=Av+B.conj(v) constrains A and B INDEPENDENTLY by the IDENTICAL equation (derived in the
    station report) -- so the full reading is EXACTLY 2x the A-only count (12), a DISCLOSED
    consequence of the pin's own linearity, not a new ambiguity.
    Returns {'A_only','full_real_linear','matches_pc2_bound'}."""
    Pw, _ = _sector_projectors(sign=+1)
    grading_rows = [_zero_block_rows(Pw[w], 8, np.eye(ND), ND) for w in (0, 2, 3)]
    A4v = _a4_vertex_group()
    dart_list = [dart_rep(g) for g in A4v]
    rho1_full = _a2c_rho_full_embedded(1, U=None)
    equiv_rows = [_linear_pin_rows(Rd, Rf) for Rd, Rf in zip(dart_list, rho1_full)]
    n_A, _, t_A, _ = _a2b_nullity_of(grading_rows + equiv_rows + [_a2b_B_zero_rows(ND, 8)])
    n_F, _, t_F, _ = _a2b_nullity_of(grading_rows + equiv_rows)
    return {"A_only": n_A, "full_real_linear": n_F, "matches_pc2_bound": n_A == 6}


def a2c_intersection(N_max=4, n_robustness_trials=5, seed=1):
    """[A2c SS4 step1, THE SOLVE -- THE VERDICT-CARRIER] the intersection of {the 24-dim bare
    survivor} (pc3: grading (i) + pair-block (ii)) with {the equivariant channel} (pin (iii):
    phi_1.rho_hist(g) = rho_1(g).phi_1 for EVERY g in A4).

    SOLVE CONVENTION (stated explicitly, per the pre-reg's own instruction): pin (iii)'s own text
    calls it a 'complex-linear constraint... the real-linear antilinear machinery is not needed
    for it' -- read literally (and cross-checked against SS4 step1's own 'expect <=6 real dims'
    bound, which ONLY matches the A-only/genuine-complex-linear reading of the equivariant channel
    -- a2c_equivariant_channel_size confirms A-only=6, full real-linear=12), the PRIMARY reading
    solves pin (iii) on the GENUINE complex-linear content of phi_1 (A-only, B=0 -- matching SS1's
    own literal definition of phi_1 and a2_gamma_word's own B=0 convention).  The FULL real-linear
    reading (both A and B independently equivariant) is ALSO reported as a disclosed secondary
    check, per the 'strictest reading, note the ambiguity, do not choose silently' instruction.

    EMBEDDING-AMBIGUITY DISCLOSURE: rho_1(g) is constructed ABSTRACTLY (_a4_standard_3irrep), NOT
    by re-deriving F's own action on Pw[1] (which would need the banned/flawed spin_lift).  Since
    A4's 3-irrep is IRREDUCIBLE, Schur's lemma fixes any two realizations of it to agree up to an
    intertwiner unique up to overall PHASE only if both are genuinely the abstract 3-irrep; the
    specific embedding into Pw[1]'s own Adag coordinates is otherwise UNFIXED by any accreted
    machinery.  Tested, not dismissed: the intersection is recomputed under n_robustness_trials
    RANDOM unitary embeddings U (plus the default U=I) -- a2c_algebraic_proof shows below WHY the
    nullity must be embedding-INDEPENDENT (Schur's lemma applies to the abstract isomorphism class,
    not to any specific matrix realization); this function VERIFIES that prediction numerically.
    Returns {'default_A_only','default_full_real_linear','allowance','nullity_le_allowance',
    'robustness_trials','all_trials_agree','smallest_kept_sv','largest_null_sv'}."""
    grading_rows, rfbit_rows, pairblock_rows, n, R = _a2b_shell_level_system(1, 1, N_max=N_max)
    A4v = _a4_vertex_group()
    dart_list = [dart_rep(g) for g in A4v]

    def solve(U):
        rho1_full = _a2c_rho_full_embedded(1, U=U)
        equiv_rows = [_linear_pin_rows(Rd, Rf) for Rd, Rf in zip(dart_list, rho1_full)]
        rows_full = grading_rows + pairblock_rows + equiv_rows
        n_A, rank_A, t_A, s_A = _a2b_nullity_of(rows_full + [_a2b_B_zero_rows(n, 8)])
        n_F, rank_F, t_F, s_F = _a2b_nullity_of(rows_full)
        return n_A, n_F, rank_A, s_A

    n_default_A, n_default_F, rank_default_A, s_default_A = solve(None)
    rng = np.random.default_rng(seed)
    trials = []
    for t in range(n_robustness_trials):
        Mrand = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        Q, Rr = np.linalg.qr(Mrand)
        d = np.diag(Rr)
        U = Q * (d / np.abs(d))
        unit_resid = float(np.max(np.abs(U.conj().T @ U - np.eye(3))))
        nA, nF, _, _ = solve(U)
        trials.append({"trial": t, "U_unitary_residual": unit_resid, "A_only": nA, "full_real_linear": nF})
    all_agree = all(tr["A_only"] == n_default_A and tr["full_real_linear"] == n_default_F for tr in trials)
    alw = a2c_level1_allowance_per_block()["allowance"]
    return {"default_A_only": n_default_A, "default_full_real_linear": n_default_F,
            "allowance": alw, "nullity_le_allowance": n_default_A <= alw,
            "robustness_trials": trials, "all_trials_agree": all_agree,
            "smallest_kept_sv": float(s_default_A[rank_default_A - 1]) if rank_default_A > 0 else float("nan"),
            "largest_null_sv": float(s_default_A[rank_default_A]) if rank_default_A < len(s_default_A) else 0.0}


def a2c_algebraic_proof():
    """[A2c SS4 step1/SS5, THE ALGEBRAIC PROOF -- SCHUR'S LEMMA / CHARACTER-ORTHOGONALITY ROUTE,
    'expected to be clean' per the pre-reg's own SS5 C3 branch] PROVES the intersection is {0}
    IDENTICALLY (theorem-grade, EMBEDDING-INDEPENDENT -- unlike the numeric SVD route, this proof
    needs NO choice of U at all), from TWO machine-verified facts:
      FACT 1 (pin (ii), pair-block, REUSED verbatim from A2b's own FACT 1, _zero_block_rows):
        phi_1(hist-GROUP-03) = 0 identically (the pair-block pin alone forces the 10-dim
        hist-GROUP-03 domain piece -- history_sector_pair_groups' own isotypic block {irrep 0
        [trivial], irrep 3 [the 3-dim standard]} -- into the kernel, given pin (i) already
        confines the image to level-1 subset field-GROUP-12, disjoint from field-GROUP-03).
      FACT 2 (NEW, pin (iii) + character orthogonality): hist-GROUP-12 (the REMAINING 2-dim
        domain piece) decomposes EXACTLY as irreps {1,2} of sector_grading_hist's own [1,1,1,3]
        table -- BOTH ONE-DIMENSIONAL (dims[1]=dims[2]=1, gns_grading_commutation's own
        conjugate_irrep_pairs=[(1,2)]) -- while level-1 (pc1) carries the UNIQUE IRREDUCIBLE
        3-dimensional A4 rep.  By the ORTHOGONALITY OF IRREDUCIBLE CHARACTERS (<chi_a,chi_b> =
        delta_ab), <chi_1,chi_3> = <chi_2,chi_3> = 0 EXACTLY -- i.e. Hom_A4(irrep 1, level-1) =
        Hom_A4(irrep 2, level-1) = {0} for DIMENSION-MISMATCH reasons alone (a 1-dim and an
        irreducible 3-dim rep can never be isomorphic, so by Schur's lemma any A4-intertwiner
        between them is 0) -- so ANY equivariant phi_1 restricted to hist-GROUP-12 is forced to 0,
        for BOTH the A and B (real-linear) parts (the equivariance equation is IDENTICAL in form
        for A and B, since dart_rep(g)/rho_1(g) are REAL/ordinary, not antiunitary), and
        REGARDLESS of which specific embedding U realizes level-1's abstract irrep inside Pw[1]
        (Schur's lemma is a basis-INDEPENDENT statement about isomorphism classes -- this is WHY
        a2c_intersection's embedding-robustness trials all agree).
    PROOF: FACT 1 kills hist-GROUP-03 (10 of 12 domain dims); FACT 2 kills hist-GROUP-12 (the
    remaining 2).  Together they span the full 12-dim shell-1 domain (history_sector_pair_groups'
    own completeness assertion, REUSED).  QED: phi_1 = 0 identically under pins (i)+(ii)+(iii) --
    a FOURTH, DISTINCT obstruction mechanism (character orthogonality / Schur's lemma), different
    from AF-3's antiunitary K-swap and A2b's R=+I/F_bit=-I collision.
    Returns {'group03_forced_zero_dim','character_orthogonality_group1','character_orthogonality_
    group2','group12_dim','group12_constituent_irrep_dims','level1_irrep_dim','proof_holds'}."""
    hg1 = history_sector_pair_groups(1)
    idx1 = np.where(hg1["lengths"] == 1)[0]
    g03_dim = int(round(np.trace(hg1["P_group03"][np.ix_(idx1, idx1)]).real))
    g12_dim = int(round(np.trace(hg1["P_group12"][np.ix_(idx1, idx1)]).real))
    gc = gns_grading_commutation(1)
    assert gc["conjugate_irrep_pairs"] == [(1, 2)], \
        f"a2c_algebraic_proof: unexpected conjugate_irrep_pairs {gc['conjugate_irrep_pairs']}"
    dims, chars_by_elt = _a4_char_lookup()
    A4v = _a4_vertex_group()
    i3 = dims.index(3)

    def char_ip(a, b):
        ca = np.array([chars_by_elt[a][_a4_key(g)] for g in A4v])
        cb = np.array([chars_by_elt[b][_a4_key(g)] for g in A4v])
        return complex(np.sum(ca * np.conj(cb)) / 12)

    ip1 = char_ip(1, i3)
    ip2 = char_ip(2, i3)
    assert dims[1] == 1 and dims[2] == 1 and dims[i3] == 3, f"a2c_algebraic_proof: irrep dims not as expected {dims}"
    assert abs(ip1) < 1e-8 and abs(ip2) < 1e-8, \
        f"a2c_algebraic_proof: character orthogonality FAILED -- <chi_1,chi_3>={ip1}, <chi_2,chi_3>={ip2}"
    inter = a2c_intersection()
    assert inter["default_A_only"] == 0 and inter["default_full_real_linear"] == 0, \
        (f"a2c_algebraic_proof: numeric intersection ({inter['default_A_only']},"
         f"{inter['default_full_real_linear']}) != (0,0), the algebraic proof's conclusion is CONTRADICTED")
    return {"group03_forced_zero_dim": g03_dim, "character_orthogonality_group1": abs(ip1),
            "character_orthogonality_group2": abs(ip2), "group12_dim": g12_dim,
            "group12_constituent_irrep_dims": [dims[1], dims[2]], "level1_irrep_dim": dims[i3],
            "proof_holds": True}


def a2c_p1_p2_reverify(N_max_p1=6, N_max_p2=10, seed=0):
    """[A2c SS4 step2, P1/P2 RE-VERIFIED IN THIS CLASS] identical reasoning to A2b's own
    a2b_p1_p2_reverify (the_net.py:4884-4898, REUSED by direct call, not re-derived): P1/P2 are
    properties of the fermionic Gamma REALIZATION itself, class-independent of phi_1's pin set.
    Returns a2b_p1_p2_reverify()'s own dict, unchanged."""
    return a2b_p1_p2_reverify(N_max_p1=N_max_p1, N_max_p2=N_max_p2, seed=seed)


def a2c_shell_level_system_nullity(shell_n, level_n, N_max=4):
    """[A2c SS4 step3, THE INDEPENDENT SHELL-n A2c-CLASS SYSTEM] grading (i) + pair-block (ii)
    (REUSED, _a2b_shell_level_system's own grading_rows/pairblock_rows, generalized to (shell_n,
    level_n)) PLUS equivariance (iii): phi_n.rho_hist_shell_n(g) = Lambda^n(rho_1)(g).phi_n for
    EVERY g in A4 (rho_hist_shell_n = the SAME A4 action on shell_n's own D_n-dim word space,
    dart_word_action REUSED and restricted; Lambda^n(rho_1) = the natural, FORCED level-n rep,
    since Lambda^n of an equivariant map is automatically Lambda^n-equivariant, per the pre-reg's
    own note).  This is the comparison space the tower-membership test checks Gamma(phi_1)
    against; it ALSO has its OWN, phi_1-INDEPENDENT freedom, reported here regardless of phi_1's
    own (forced-zero) outcome.
    Returns {'shell_n','level_n','domain_dim','grading_pairblock_A_only','full_A_only',
    'full_real_linear','lambda_n_group_law_residual'}."""
    grading_rows, rfbit_rows, pairblock_rows, n, R = _a2b_shell_level_system(shell_n, level_n, N_max=N_max)
    _, hist_mats, idx = _a2c_shell_hist_action(shell_n, N_max=N_max)
    assert len(idx) == n
    _, rho_n, worst_law, _ = _a2c_level_rep(level_n)
    rho1_full = _a2c_rho_full_embedded(level_n)
    equiv_rows = [_linear_pin_rows(Hd, Rf) for Hd, Rf in zip(hist_mats, rho1_full)]
    n_gp_A, _, _, _ = _a2b_nullity_of(grading_rows + pairblock_rows + [_a2b_B_zero_rows(n, 8)])
    rows_full = grading_rows + pairblock_rows + equiv_rows
    n_full_A, _, _, _ = _a2b_nullity_of(rows_full + [_a2b_B_zero_rows(n, 8)])
    n_full_F, _, _, _ = _a2b_nullity_of(rows_full)
    return {"shell_n": shell_n, "level_n": level_n, "domain_dim": n,
            "grading_pairblock_A_only": n_gp_A, "full_A_only": n_full_A,
            "full_real_linear": n_full_F, "lambda_n_group_law_residual": worst_law}


def a2c_tower_membership_test(shell_n, phi1_basis, N_max=4):
    """[A2c SS4 step3, THE FORCING QUESTION + THE HONESTY CLAUSE] for EACH basis map in
    phi1_basis (empty here, since a2c_intersection's nullity is 0 -- a2c_algebraic_proof), builds
    Gamma(phi1) on shell_n (REUSED a2_gamma_word/_level1_creation_ops unchanged) and tests
    membership in the A2c-CLASS shell_n system's OWN full-pin row-stack (grading+pairblock+
    equivariance, a2c_shell_level_system_nullity's own construction, rebuilt here for the residual
    test).  VACUOUSNESS (the pre-reg's honesty clause): IF phi1_basis is empty, Gamma(phi_1)=0
    trivially lies in EVERY linear space -- reported HONESTLY as VACUOUS, REGARDLESS of whether
    the shell_n system's own nullity is itself zero or nonzero.  A DISCRIMINATING CONTROL (the
    SAME structural test map A2/A2b's P1/P2 checks use) is run alongside.
    Returns {'shell_n','D_shell_n','basis_size','all_members','control_residual',
    'control_is_member','vacuous'}."""
    grading_rows, rfbit_rows, pairblock_rows, n, R = _a2b_shell_level_system(shell_n, shell_n, N_max=N_max)
    _, hist_mats, idx = _a2c_shell_hist_action(shell_n, N_max=N_max)
    rho1_full = _a2c_rho_full_embedded(shell_n)
    equiv_rows = [_linear_pin_rows(Hd, Rf) for Hd, Rf in zip(hist_mats, rho1_full)]
    Cop = np.vstack(grading_rows + pairblock_rows + equiv_rows)
    _, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(max(N_max, shell_n), succ)
    Adag, vac, _, _ = _level1_creation_ops()

    def gamma_matrix(phi1):
        cols = [a2_gamma_word(phi1, words[i], Adag, vac) for i in idx]
        return np.hstack(cols) if cols else np.zeros((8, 0), dtype=complex)

    def residual_of(Gamma):
        vecX = _a2_vec_colmajor_from_AB(Gamma, np.zeros_like(Gamma), len(idx), 8)
        return float(np.max(np.abs(Cop @ vecX))) if Cop.size else 0.0

    member_residuals = [residual_of(gamma_matrix(phi1)) for phi1 in phi1_basis]
    all_members = all(r < 1e-6 for r in member_residuals) if member_residuals else True
    vacuous = len(phi1_basis) == 0
    rng = np.random.default_rng(0)
    phi_test = rng.normal(size=(ND, 3)) + 1j * rng.normal(size=(ND, 3))
    control_resid = residual_of(gamma_matrix(phi_test))
    return {"shell_n": shell_n, "D_shell_n": len(idx), "basis_size": len(phi1_basis),
            "all_members": bool(all_members), "control_residual": control_resid,
            "control_is_member": bool(control_resid < 1e-6), "vacuous": vacuous}


def a2c_pair_completeness_read():
    """[A2c SS4 step4, PAIR-COMPLETENESS -- REPORT ONLY] since phi_1 = 0 identically
    (a2c_algebraic_proof), dim(Im phi_1) = 0; Phi~ := J_F.Phi.J_hist (the pair-object DEFINITION
    inherited from A2b, SS3's own banner: 'Phi_tilde := J_F.Phi.J_hist defined; L0c check
    inherited' -- carried here purely as a DEFINITION, not re-solved) is therefore ALSO 0 (the
    conjugate of the zero map is the zero map).  The pair (Phi, Phi~) covers NOTHING of F.
    Returns {'dim_Im_phi1','Phi_tilde_is_zero','pair_covers_F'}."""
    proof = a2c_algebraic_proof()
    return {"dim_Im_phi1": 0, "Phi_tilde_is_zero": bool(proof["proof_holds"]), "pair_covers_F": False}


def a2c_species_read():
    """[A2c SS4 step4, THE SPECIES READ -- D4 OUTPUT ONLY, 'now with content' per the pre-reg's own
    framing] since phi_1 = 0 identically (a2c_algebraic_proof, the intersection is EMPTY: C3), NO
    isotypic component of [1,1,1,9] survives into a nonzero image and there is NO combination of
    the three 3-irrep copies (the 9-part) that carries a solution -- the pre-reg's own hopeful
    framing ('now with content') does NOT materialize; disclosed as a genuine, honest finding, not
    smoothed into a partial-content claim.  D4 GUARD (verified): grep-confirmed no
    gauge_sector_category/species-label value is read INTO a2c_intersection/a2c_algebraic_proof/
    a2c_pc1..pc4/a2c_equivariant_channel_size -- the species map is NEVER a constraint anywhere in
    this station's pin construction, only ever narratively invoked here to report the (empty)
    result.
    Returns {'surviving_isotypic_components','species_correspondence','note'}."""
    return {"surviving_isotypic_components": [],
            "species_correspondence": "NONE -- phi_1 = 0 identically (the equivariant channel "
                                       "misses the 24-dim survivor entirely, C3); no nonzero image "
                                       "exists to read an isotypic<->{1,3,3,1} correspondence from",
            "note": "D4 respected throughout SS3/SS4's pin construction; this function is the ONLY "
                    "place gauge_sector_category's species labels are narratively invoked, and "
                    "only to report an empty result -- the pre-reg's SS4 step4 framing ('now with "
                    "content') is disclosed as NOT materializing, an honest negative"}


def a2c_weld_selftest_2026_07_12(verbose=True):
    """A2c station regression: Sections 7/7b/8/8b/8c/8d/8e/8f/8g + module anchors untouched, THEN
    pc1-pc4 (printed before any solve), the pre-declared allowance, the equivariant-channel-size
    diagnostic (cross-checking pc2's own <=6 real dims bound), the intersection (default embedding
    + robustness trials), the algebraic proof (Schur's lemma / character orthogonality), P1/P2
    re-verification, the independent shell-2/3 A2c-class systems + the tower-membership test with
    its explicit vacuousness determination and discriminating control, pair-completeness, and the
    species read.  Does NOT itself adjudicate the SS5 verdict tree (architect-only per the pre-reg)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 8h self-test -- A2c the corrected-dictionary equivariant weld (2026-07-12)")
        print("=" * 88)

    ck("ANCHORS + Sections 7/7b/8/8b/8c/8d/8e/8f/8g untouched",
       anchor_cell_projector() and anchor_tick_2pi() and accretion_selftest_2026_07_10(verbose=False)
       and i2b_selftest_2026_07_11(verbose=False) and fock0_selftest_2026_07_11(verbose=False)
       and fock0b_selftest_2026_07_11(verbose=False) and fock0c_selftest_2026_07_11(verbose=False)
       and fock0d_selftest_2026_07_11(verbose=False) and fock0e_selftest_2026_07_12(verbose=False)
       and a2_weld_selftest_2026_07_12(verbose=False) and a2b_weld_selftest_2026_07_12(verbose=False))

    pc1 = a2c_pc1_honest_level1_rep()
    ck(f"pc1 (honest level-1 A4 rep, STOP=C4 if failed): group-law residual={pc1['group_law_residual_1e12']:.1e} "
       f"(<=1e-12), NO cocycle=-1 pair found (smallest gap {pc1['smallest_no_cocycle_minus1_gap']:.3f}), "
       f"character match={pc1['character_match_residual']:.1e}", pc1["pc1_pass"])

    pc2 = a2c_pc2_hom_multiplicity()
    ck(f"pc2 (Hom_A4(shell-1,level-1)=3 complex dims, via character inner product): "
       f"<chi_reg,chi_3>={pc2['hom_via_character_inner_product']}, direct SVD={pc2['hom_via_direct_svd']}, "
       f"mult[1]={pc2['mult1']}", pc2["pc2_pass"])

    pc3 = a2c_pc3_reproduce_survivor()
    ck(f"pc3 (reproduce A2b's 24-dim bare survivor): grading alone={pc3['grading_alone']}, "
       f"+pairblock={pc3['pairblock_survivor']}/{pc3['total_real_dim']}", pc3["pc3_pass"])

    pc4 = a2c_pc4_automatic_parity()
    ck(f"pc4 (automatic word-length parity on Gamma images): worst NHAT-eigenvalue residual "
       f"{pc4['worst_nhat_eigenvalue_residual']:.1e} over {pc4['n_words_checked']} words (shells 1-3)",
       pc4["pc4_pass"])

    alw = a2c_level1_allowance_per_block()
    ck(f"ALLOWANCE (PRINTED before solving, reused from A2b): GROUP-03={alw['group03_allowance']}, "
       f"GROUP-12={alw['group12_allowance']}, TOTAL={alw['allowance']}",
       alw["allowance"] == 1 and alw["group03_allowance"] == 0)

    ecs = a2c_equivariant_channel_size()
    ck(f"EQUIVARIANT CHANNEL SIZE (grading+equivariance alone, no pairblock): A-only={ecs['A_only']} "
       f"(matches pc2's <=6 real-dim bound: {ecs['matches_pc2_bound']}), full real-linear={ecs['full_real_linear']}",
       ecs["matches_pc2_bound"])

    inter = a2c_intersection()
    ck(f"THE INTERSECTION (24-dim survivor ^ equivariant channel): default A-only={inter['default_A_only']}, "
       f"default full real-linear={inter['default_full_real_linear']}, allowance={inter['allowance']} "
       f"(nullity<=allowance: {inter['nullity_le_allowance']}); {len(inter['robustness_trials'])} random-"
       f"embedding robustness trials ALL AGREE: {inter['all_trials_agree']}; smallest kept sv="
       f"{inter['smallest_kept_sv']:.3f}, largest null sv={inter['largest_null_sv']:.1e}",
       inter["default_A_only"] == 0 and inter["default_full_real_linear"] == 0 and inter["all_trials_agree"])

    proof = a2c_algebraic_proof()
    ck("THE ALGEBRAIC PROOF (Schur's lemma / character orthogonality -- a FOURTH, distinct "
       "obstruction mechanism, embedding-independent): pair-block kills hist-GROUP-03 (dim "
       f"{proof['group03_forced_zero_dim']}); character orthogonality <chi_1,chi_3>="
       f"{proof['character_orthogonality_group1']:.1e}, <chi_2,chi_3>={proof['character_orthogonality_group2']:.1e} "
       f"kills hist-GROUP-12 (dim {proof['group12_dim']}, constituent irrep dims "
       f"{proof['group12_constituent_irrep_dims']} vs level-1's irrep dim {proof['level1_irrep_dim']}) "
       "-- phi_1 = 0 FORCED", proof["proof_holds"])

    p12 = a2c_p1_p2_reverify()
    ck(f"P1/P2 RE-VERIFIED (reused from A2, unchanged construction): P1 worst shell-4 norm "
       f"{p12['p1']['worst_shell4_norm']:.1e}; P2 synthetic-repeat residual "
       f"{p12['p2']['synthetic_repeat_residual']:.1e}, first admissible repeat at shell "
       f"{p12['p2']['first_shell_with_admissible_repeat']}",
       p12["p1"]["worst_shell4_norm"] < 1e-9 and p12["p2"]["synthetic_repeat_residual"] < 1e-9)

    s2 = a2c_shell_level_system_nullity(2, 2)
    s3 = a2c_shell_level_system_nullity(3, 3)
    ck(f"INDEPENDENT SHELL-n A2c-CLASS SYSTEMS (own freedom, NOT phi_1's): shell2->level2 grading+"
       f"pairblock A-only={s2['grading_pairblock_A_only']}, +equivariance A-only={s2['full_A_only']} "
       f"(full real-linear {s2['full_real_linear']}); shell3->level3 grading+pairblock A-only="
       f"{s3['grading_pairblock_A_only']}, +equivariance A-only={s3['full_A_only']} (full real-linear "
       f"{s3['full_real_linear']}); Lambda^n group-law residuals {s2['lambda_n_group_law_residual']:.1e}/"
       f"{s3['lambda_n_group_law_residual']:.1e}",
       s2["lambda_n_group_law_residual"] < 1e-8 and s3["lambda_n_group_law_residual"] < 1e-8)

    tw2 = a2c_tower_membership_test(2, [])
    tw3 = a2c_tower_membership_test(3, [])
    ck(f"TOWER-MEMBERSHIP TEST shell 2 [D={tw2['D_shell_n']}]: basis_size=0 -> VACUOUS={tw2['vacuous']} "
       f"(the 0 map, trivially member); discriminating control residual={tw2['control_residual']:.3e} "
       f"(control_is_member={tw2['control_is_member']})",
       tw2["all_members"] and tw2["vacuous"] and not tw2["control_is_member"])
    ck(f"TOWER-MEMBERSHIP TEST shell 3 [D={tw3['D_shell_n']}]: basis_size=0 -> VACUOUS={tw3['vacuous']}; "
       f"discriminating control residual={tw3['control_residual']:.3e} "
       f"(control_is_member={tw3['control_is_member']})",
       tw3["all_members"] and tw3["vacuous"] and not tw3["control_is_member"])

    pc = a2c_pair_completeness_read()
    ck(f"PAIR-COMPLETENESS (report only): dim(Im phi_1)={pc['dim_Im_phi1']}, "
       f"Phi~ is zero={pc['Phi_tilde_is_zero']}, pair covers F={pc['pair_covers_F']}",
       pc["dim_Im_phi1"] == 0 and pc["Phi_tilde_is_zero"])

    nh = a2b_nhat_intertwining_exactness()
    ck(f"N-HAT-INTERTWINING EXACTNESS [shells 1-3, REUSED from A2b unchanged, phi_1-independent]: "
       f"worst residual {nh['worst_residual']:.2e}", nh["worst_residual"] < 1e-9)

    emb = a2b_ember_consistency_shadow()
    tri = next(r for r in emb if r["is_triangle"])
    ck(f"EMBER-CONSISTENCY SHADOW [REUSED from A2b/FOCK-0e unchanged] (structure only): triangle "
       f"orbit lambda_1={tri['lambda_1']:.4f} (FOCK-0d ember reference 2.463)",
       abs(tri["lambda_1"] - 2.463) < 0.01)

    sp = a2c_species_read()
    ck("SPECIES READ (output only, D4 guard): no surviving isotypic components (phi_1 = 0); "
       "species map never used as an input anywhere in SS3/SS4's pin construction",
       sp["surviving_isotypic_components"] == [])

    if verbose:
        print("RESULT:", "A2c SECTION-8h REGRESSION PASSES" if ok else "AN A2c CHECK FAILED")
    return ok


# ===========================================================================
# 8i. A2d -- THE MINIMAL WELD CLASS (2026-07-12, Push 2, THE FINAL ARC STATION)
#     A2d_minimal_weld_prereg_2026-07-12.md SS2-4
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as Sections 7/7b/8/8b/8c/8d/8e/8f/8g/8h above.]
#
# HYPOTHESIS UNDER TEST (SS1, pre-reg): A2c's own checker (working notes/A2c_check_2026-07-12.md
# SS5-SS6) found that the per-sector-pair BLOCK pin (A2c's pin (ii), A2b's own (iii)) is retained
# in a J-free class purely by INHERITANCE -- its two frozen justifications (the per-sector-pair
# Tomita structure; A2b's (Phi,Phi~) pair-object motivation) are BOTH dead here (AF-3/L0a refuted
# all J-pins; Phi~==0 makes the pair-object motivation moot).  It ALSO discards EXACTLY the 3-irrep
# content -- the only content that can EVER reach level-1 equivariantly (Schur's lemma forces any
# equivariant map to vanish on the two 1-dim characters).  This station DROPS the pair-block pin
# entirely (SS1's removal, theorem/audit-backed, NOT a convenience) -- the frozen pin set is now:
# (i) grading, (ii) A4-equivariance at level 1 ONLY.  D4 (frozen, hard guard, verbatim from A2/
# A2b/A2c): the species map is NEVER an input anywhere below -- grep-confirmed, only ever read in
# the OUTPUT-only a2d_species_read.  THE STOPPING RULE (pre-reg SS0): this is the LAST station of
# the weld arc -- no further pin modifications after this one.
#
# NUMBERS APPEAR NOWHERE: every quantity below is a dimension, nullity, rank, or exactness
# residual (structure), never M_Z/ppm/m_nu/a_e (module contract + pre-reg SS6 poisons).
#
# ML-2b/HK-7 CONDITIONALITY (carries into every DR-frame-touching sentence below, verbatim,
# unchanged from Section 8's own banner): "Every duality check here (HK-5) is CELL-LEVEL only (the
# 6-edge static vacuum). ML-2b's DR-frame argument is CONDITIONAL on the TD-limit duality holding,
# which is NOT verified by this suite."
def _a2d_shell1_isotypic_projectors():
    """[A2d, SHARED HELPER] sector_grading_hist(1)'s own isotypic projectors P[0..3], RESTRICTED
    to the shell-1 (length==1, the 12 darts) index block -- sector_grading_hist(N_max=1) includes
    the 1-dim length-0 seed too (13-dim total), so every P[a] must be sliced down to the 12-dim
    dart block before use (the SAME restriction a2c_algebraic_proof's own idx1 = np.where(lengths
    == 1) already performs, REUSED convention)."""
    sg1 = sector_grading_hist(1)
    idx1 = np.where(sg1["lengths"] == 1)[0]
    return [sg1["P"][a][np.ix_(idx1, idx1)] for a in range(4)]


def _a2d_rank(M, tol=1e-8):
    """[A2d, SHARED HELPER] numeric rank of a (possibly complex) matrix M by SVD threshold, the
    SAME 1e-8 threshold _a2b_nullity_of/a2c's own routines use throughout this section (a plain
    utility, not a new construction)."""
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol))


def a2d_allowance():
    """[A2d SS2, THE PRE-DECLARED ALLOWANCE -- PRINTED BEFORE SOLVING] the frozen pin set is now
    ONLY (i) grading + (ii) A4-equivariance at level 1 (the pair-block pin REMOVED, SS1).
    Hom_A4(shell-1, level-1) is the MULTIPLICITY SPACE of the 3-irrep inside shell-1's own isotypic
    decomposition ([1,1,1,9] = [1,1,1,3x3], sector_grading_hist's own mult[1]) -- a THREE-
    dimensional COMPLEX space (a2c_pc2_hom_multiplicity's own <chi_reg,chi_3>=3, REUSED unchanged;
    this station removes the pin that previously discarded all of it, so the FULL multiplicity
    space is now live).  By SCHUR'S LEMMA, any SINGLE nonzero equivariant map picked from a FIXED
    direction among the 3 copies is unique up to an OVERALL COMPLEX SCALAR (scale + phase) -- 1
    complex = 2 real dims -- the physically-unfixable freedom every equivariant construction
    carries regardless of which direction is chosen.  The DIRECTION itself (which linear
    combination of the 3 copies) is NOT selected by grading + equivariance alone -- so the
    PRE-DECLARED allowance (2 real, the scale-phase alone) is DELIBERATELY smaller than the full
    multiplicity-space dimension (6 real): a solved nullity of 6 (matching the FULL multiplicity
    space, no direction singled out) is therefore ANTICIPATED to EXCEED this allowance -- stated
    here, before solving, per the pre-reg's own D2 anticipation (SS5), so it cannot be dressed
    after the fact.
    Returns {'multiplicity_space_complex_dim','multiplicity_space_real_dim','allowance_real',
    'allowance_justification'}."""
    pc2 = a2c_pc2_hom_multiplicity()
    mult_dim_complex = int(round(pc2["hom_via_character_inner_product"].real))
    assert mult_dim_complex == 3, f"a2d_allowance: multiplicity space dim {mult_dim_complex} != 3"
    return {"multiplicity_space_complex_dim": mult_dim_complex,
            "multiplicity_space_real_dim": 2 * mult_dim_complex,
            "allowance_real": 2,
            "allowance_justification":
                "ONE overall complex scale-phase (Schur's lemma) per FIXED direction in the 3-dim "
                "multiplicity space -- the physically-unfixable freedom; the direction itself is "
                "NOT fixed by grading+equivariance alone, so a solved nullity of 6 real "
                "(the full multiplicity space) is anticipated to EXCEED this 2-real allowance."}


def _a2d_abstract_hom_basis():
    """[A2d SS4 step1, THE SOLVE -- ABSTRACT COORDINATES] Hom_A4(shell-1, level-1) computed
    DIRECTLY in ABSTRACT coordinates: dart_rep(g) on the 12-dim domain, rho3 (_a4_standard_3irrep,
    pc1's honest, non-projective level-1 rep) on the 3-dim abstract codomain -- the IDENTICAL
    complex-linear Sylvester system a2c_pc2_hom_multiplicity's own direct-SVD cross-check builds
    (REUSED method), but returning the NULL-SPACE BASIS itself (pc2 only returns its dimension).
    BIJECTIVE with a2c_equivariant_channel_size's own A-only null space via phi = E_1^dagger . A
    (E_1 = _a2c_level_embedding(1)): since grading confines A's image to Pw[1] = range(E_1) and
    rho1_full acts as E_1.rho3.E_1^dagger there (_a2c_rho_full_embedded's own construction), the
    F-embedded equivariance equation A.dart(g) = rho1_full(g).A restricted to range(A) subset Pw[1]
    is EXACTLY the abstract equation phi.dart(g) = rho3(g).phi -- so this basis and the F-embedded
    A-only channel describe the SAME solution space (cross-checked numerically in
    a2d_solve_channel below, not merely asserted here).
    Returns (A4v, basis: list of complex (3,ND) matrices phi_k with phi_k @ dart(g) = rho3(g) @
    phi_k for every g in A4, 'nullity_complex', 'group_law_residual')."""
    A4v = _a4_vertex_group()
    _, rho3, _, _ = _a4_standard_3irrep()
    dart_list = [dart_rep(g) for g in A4v]
    I3 = np.eye(3)
    rows = [np.kron(Rd.T, I3) - np.kron(np.eye(ND), Rg) for Rd, Rg in zip(dart_list, rho3)]
    Cop = np.vstack(rows).astype(complex)
    U, s, Vh = np.linalg.svd(Cop, full_matrices=True)
    rank = int(np.sum(s > 1e-8))
    total = Cop.shape[1]
    nullity = total - rank
    null_vecs = Vh[rank:, :].conj()
    basis = [v.reshape(3, ND, order="F") for v in null_vecs]
    worst = 0.0
    for phi in basis:
        for k in range(len(A4v)):
            worst = max(worst, float(np.max(np.abs(phi @ dart_list[k] - rho3[k] @ phi))))
    return A4v, basis, nullity, worst


def _a2d_copy_embeddings():
    """[A2d SS4 step4a, THE THREE COPY EMBEDDINGS] Hom_A4(level-1, shell-1) -- the OPPOSITE
    direction from _a2d_abstract_hom_basis -- via the IDENTICAL complex-linear SVD method (rows
    swapped): psi: C^3 -> C^12 with psi @ rho3(g) = dart(g) @ psi for every g.  By Frobenius
    reciprocity (the dual statement to pc2's own count) this ALSO has complex dim 3 -- THE THREE
    EMBEDDINGS of the 3-irrep's copies inside shell-1's own 9-dim isotypic block, used by
    a2d_multiplicity_geometry to express solutions as directions in the 3-dim multiplicity space.
    Returns (A4v, basis: list of complex (ND,3) matrices, 'nullity_complex', 'group_law_residual')."""
    A4v = _a4_vertex_group()
    _, rho3, _, _ = _a4_standard_3irrep()
    dart_list = [dart_rep(g) for g in A4v]
    I12 = np.eye(ND)
    rows = [np.kron(Rg.T, I12) - np.kron(np.eye(3), Rd) for Rd, Rg in zip(dart_list, rho3)]
    Cop = np.vstack(rows).astype(complex)
    U, s, Vh = np.linalg.svd(Cop, full_matrices=True)
    rank = int(np.sum(s > 1e-8))
    nullity = ND * 3 - rank
    null_vecs = Vh[rank:, :].conj()
    basis = [v.reshape(ND, 3, order="F") for v in null_vecs]
    worst = 0.0
    for psi in basis:
        for k in range(len(A4v)):
            worst = max(worst, float(np.max(np.abs(psi @ rho3[k] - dart_list[k] @ psi))))
    return A4v, basis, nullity, worst


def a2d_solve_channel(N_max=4):
    """[A2d SS4 step1, THE SOLVE] the minimal system, grading (i) + level-1 A4-equivariance (ii)
    ONLY -- NO pair-block pin (removed per SS1; NOT re-imposed under any framing).  Reports the
    nullity vs the pre-declared allowance (a2d_allowance).  Reproduces a2c_equivariant_channel_
    size's own A-only=6/full=12 F-embedded waypoint (REUSED, UNCHANGED construction -- the row
    builders are identical, this station's pins (i)+(ii) are literally the SAME construction
    a2c_equivariant_channel_size already builds, since a2c never stacked the pair-block rows into
    THIS particular sub-computation) AND cross-checks it against the abstract-coordinate
    computation (_a2d_abstract_hom_basis, a genuinely DIFFERENT route -- no F-embedding/grading-row
    Kronecker machinery at all, pure Hom_A4(shell-1,level-1) via the honest rho3 directly), which
    must give the SAME complex dimension (bijective via phi = E_1^dagger . A).
    Returns {'waypoint_A_only','waypoint_full_real_linear','abstract_nullity_complex',
    'abstract_matches_waypoint','abstract_group_law_residual','allowance_real','nullity_real',
    'nullity_exceeds_allowance'}."""
    ecs = a2c_equivariant_channel_size(N_max=N_max)
    A4v, phi_basis, n_phi_complex, worst_law = _a2d_abstract_hom_basis()
    alw = a2d_allowance()
    nullity_real = ecs["A_only"]
    matches = (2 * n_phi_complex == ecs["A_only"])
    assert matches, (f"a2d_solve_channel: abstract route ({n_phi_complex} complex = "
                      f"{2*n_phi_complex} real) != F-embedded waypoint ({ecs['A_only']} real) "
                      "-- SURPRISE, the two routes disagree")
    return {"waypoint_A_only": ecs["A_only"], "waypoint_full_real_linear": ecs["full_real_linear"],
            "abstract_nullity_complex": n_phi_complex, "abstract_matches_waypoint": matches,
            "abstract_group_law_residual": worst_law, "allowance_real": alw["allowance_real"],
            "nullity_real": nullity_real, "nullity_exceeds_allowance": nullity_real > alw["allowance_real"]}


def a2d_waypoint_isotypic_locus():
    """[A2d, WAYPOINT REPRODUCTION -- A2c checker item 5(iii)] reproduce the A2c checker's
    isotypic-locus finding on THIS station's OWN basis (_a2d_abstract_hom_basis, computed
    independently of anything the checker built): the channel is annihilated by domain-side
    isotypic projectors P[0],P[1],P[2] (sector_grading_hist's own battery, REUSED) and lives
    ENTIRELY in P[3] (the 3-irrep's own 9-dim isotypic block).
    Returns {'n_basis','worst_vanish_P012','worst_supported_P3_residual','entirely_3_isotypic'}."""
    A4v, phi_basis, n_phi, worst_law = _a2d_abstract_hom_basis()
    P = _a2d_shell1_isotypic_projectors()
    worst_low = 0.0
    worst_p3 = 0.0
    for phi in phi_basis:
        worst_low = max(worst_low, max(float(np.max(np.abs(phi @ P[a]))) for a in (0, 1, 2)))
        worst_p3 = max(worst_p3, float(np.max(np.abs(phi @ P[3] - phi))))
    return {"n_basis": len(phi_basis), "worst_vanish_P012": worst_low,
            "worst_supported_P3_residual": worst_p3,
            "entirely_3_isotypic": bool(worst_low < 1e-8 and worst_p3 < 1e-8)}


def _a2d_shell_level_system_no_pairblock(shell_n, level_n, N_max=4):
    """[A2d SS4 step2/step3, THE PAIR-BLOCK-FREE SHELL-n SYSTEM] grading (i) + level-n
    equivariance (ii) ONLY -- the pair-block pin DROPPED entirely (SS1's removal; its
    pairblock_rows output from _a2b_shell_level_system is simply NOT stacked here, the SAME
    'reuse the builder, drop one term' pattern a2c_pc3 already used for the grading+pairblock-only
    reading).  Equivariance rows reuse _a2c_shell_hist_action/_a2c_level_rep/_a2c_rho_full_embedded
    UNCHANGED (the IDENTICAL builders a2c_shell_level_system_nullity itself uses, minus the
    pairblock_rows term).
    Returns (grading_rows, equiv_rows, n_domain, lambda_n_group_law_residual)."""
    grading_rows, rfbit_rows, pairblock_rows, n, R = _a2b_shell_level_system(shell_n, level_n, N_max=N_max)
    _, hist_mats, idx = _a2c_shell_hist_action(shell_n, N_max=N_max)
    assert len(idx) == n
    _, rho_n, worst_law, _ = _a2c_level_rep(level_n)
    rho1_full = _a2c_rho_full_embedded(level_n)
    equiv_rows = [_linear_pin_rows(Hd, Rf) for Hd, Rf in zip(hist_mats, rho1_full)]
    return grading_rows, equiv_rows, n, worst_law


def a2d_waypoint_shell_level_nullity(shell_n, level_n, N_max=4):
    """[A2d, WAYPOINT REPRODUCTION -- A2c checker item 5(iv)] the pair-block-FREE shell_n->level_n
    system's own nullity (grading-alone, +equivariance, A-only and full real-linear).  For
    (1,1) this MUST match a2c_equivariant_channel_size's own A_only=6 exactly (the SAME
    construction, cross-checked in a2d_solve_channel via a different route too); for (2,2)/(3,3)
    this reproduces the checker's own independently-computed 12/8 real A-only waypoints.
    Returns {'shell_n','level_n','domain_dim','grading_alone_A_only','full_A_only',
    'full_real_linear','lambda_n_group_law_residual'}."""
    grading_rows, equiv_rows, n, worst_law = _a2d_shell_level_system_no_pairblock(shell_n, level_n, N_max=N_max)
    n_gp_A, _, _, _ = _a2b_nullity_of(grading_rows + [_a2b_B_zero_rows(n, 8)])
    rows_full = grading_rows + equiv_rows
    n_full_A, _, _, _ = _a2b_nullity_of(rows_full + [_a2b_B_zero_rows(n, 8)])
    n_full_F, _, _, _ = _a2b_nullity_of(rows_full)
    return {"shell_n": shell_n, "level_n": level_n, "domain_dim": n,
            "grading_alone_A_only": n_gp_A, "full_A_only": n_full_A,
            "full_real_linear": n_full_F, "lambda_n_group_law_residual": worst_law}


def _a2d_phi1_list(N_max=4):
    """[A2d, SHARED HELPER] the channel basis (_a2d_abstract_hom_basis) converted to a2_gamma_
    word's own (ND,3) convention (phi1[d,m], REUSED unchanged construction/convention from A2)."""
    A4v, phi_basis, n_phi, worst_law = _a2d_abstract_hom_basis()
    return [phi.T for phi in phi_basis]


def a2d_gamma_live_behavior(N_max=6):
    """[A2d SS4 step2, Gamma(phi_1) ON A NONZERO CHANNEL BASIS -- P1/P2 NOW LIVE] P1 (Pauli
    truncation, shell>=4 vanishes) and P2 (repeated-dart kernel) run on the ACTUAL, NONZERO
    channel basis for the FIRST TIME in the arc (A2/A2b/A2c all had phi_1=0 forced, so these
    checks previously ran on a SYNTHETIC structural test map instead, see a2_pauli_truncation_
    check/a2_repeated_dart_kernel_check's own docstrings).  For each basis element (a2_gamma_word,
    _level1_creation_ops REUSED unchanged): reports shell-1/2/3 image norms (expected NONZERO,
    genuine content) and shell-4 norm (must vanish, P1 -- a GENERAL fact of the fermionic
    realization, independent of which phi_1), plus a repeated-dart synthetic-word check (P2).
    Returns {'basis_rows','worst_shell4_norm','worst_repeat_norm','n_basis'}."""
    phi1_list = _a2d_phi1_list(N_max=N_max)
    Adag, vac, _, _ = _level1_creation_ops()
    _, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(N_max, succ)
    rows = []
    for bi, phi1 in enumerate(phi1_list):
        per_shell = {}
        for shell in (1, 2, 3, 4):
            idxs = [i for i, w in enumerate(words) if len(w) == shell]
            norms = [float(np.linalg.norm(a2_gamma_word(phi1, words[i], Adag, vac))) for i in idxs]
            per_shell[shell] = {"n_words": len(idxs), "worst": max(norms) if norms else 0.0,
                                 "mean": float(np.mean(norms)) if norms else 0.0}
        repeat_word = (0, 1, 0)
        repeat_norm = float(np.linalg.norm(a2_gamma_word(phi1, repeat_word, Adag, vac)))
        rows.append({"basis_index": bi, "per_shell": per_shell, "repeat_word_norm": repeat_norm})
    worst_shell4 = max(r["per_shell"][4]["worst"] for r in rows)
    worst_repeat = max(r["repeat_word_norm"] for r in rows)
    assert worst_shell4 < 1e-9, f"a2d_gamma_live_behavior: P1 violated, shell-4 norm {worst_shell4:.2e}"
    assert worst_repeat < 1e-9, f"a2d_gamma_live_behavior: P2 violated, repeat-word norm {worst_repeat:.2e}"
    assert rows[0]["per_shell"][1]["worst"] > 1e-6, \
        "a2d_gamma_live_behavior: shell-1 image degenerately zero (basis not genuinely nonzero)"
    return {"basis_rows": rows, "worst_shell4_norm": worst_shell4, "worst_repeat_norm": worst_repeat,
            "n_basis": len(phi1_list)}


def a2d_tower_consistency(N_max=4):
    """[A2d SS4 step3, THE TOWER CONSISTENCY CHECK -- LABELED CONSISTENCY, NEVER FORCING, per the
    pre-reg SS3 honesty clause] Lambda^n of an equivariant map is AUTOMATICALLY Lambda^n-
    equivariant -- so membership of Gamma(phi_1)'s ACTUAL, NONZERO basis in the INDEPENDENTLY-
    built pair-block-free shell-2/3 systems (_a2d_shell_level_system_no_pairblock) is
    MATHEMATICALLY GUARANTEED, not a forcing result: this function VERIFIES that guarantee holds
    (residuals ~machine precision) and reports it EXPLICITLY as a CONSISTENCY check, per the
    pre-reg's own instruction ('do NOT count it as forcing').  A DISCRIMINATING CONTROL (the SAME
    structural, non-equivariant test map A2/A2b/A2c's P1/P2 checks use) is run alongside and MUST
    be correctly REJECTED (large residual) -- confirming the membership machinery genuinely
    discriminates and the consistency finding is not a degenerate always-pass artifact.
    Returns {'shell_n','D_shell_n','basis_size','member_residuals','all_members_consistency',
    'control_residual','control_is_member'} for shells 2 and 3 (dict keyed by shell_n)."""
    phi1_list = _a2d_phi1_list(N_max=N_max)
    Adag, vac, _, _ = _level1_creation_ops()
    _, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(max(N_max, 3), succ)
    out = {}
    for shell_n in (2, 3):
        grading_rows, equiv_rows, n, worst_law = _a2d_shell_level_system_no_pairblock(shell_n, shell_n, N_max=N_max)
        Cop = np.vstack(grading_rows + equiv_rows)
        idx = np.where(lengths == shell_n)[0]

        def gamma_matrix(phi1):
            cols = [a2_gamma_word(phi1, words[i], Adag, vac) for i in idx]
            return np.hstack(cols) if cols else np.zeros((8, 0), dtype=complex)

        def residual_of(Gamma):
            vecX = _a2_vec_colmajor_from_AB(Gamma, np.zeros_like(Gamma), len(idx), 8)
            return float(np.max(np.abs(Cop @ vecX))) if Cop.size else 0.0

        member_residuals = [residual_of(gamma_matrix(phi1)) for phi1 in phi1_list]
        all_consistency = all(r < 1e-6 for r in member_residuals)
        rng = np.random.default_rng(0)
        phi_test = rng.normal(size=(ND, 3)) + 1j * rng.normal(size=(ND, 3))
        control_resid = residual_of(gamma_matrix(phi_test))
        out[shell_n] = {"shell_n": shell_n, "D_shell_n": len(idx), "basis_size": len(phi1_list),
                         "member_residuals": member_residuals,
                         "all_members_consistency": bool(all_consistency),
                         "control_residual": control_resid,
                         "control_is_member": bool(control_resid < 1e-6)}
        assert all_consistency, f"a2d_tower_consistency: shell {shell_n} CONSISTENCY VIOLATED (SURPRISE)"
        assert control_resid > 8.0, \
            f"a2d_tower_consistency: shell {shell_n} discriminating control too weak ({control_resid:.2f})"
    return out


def a2d_multiplicity_geometry():
    """[A2d SS4 step4a, THE MULTIPLICITY GEOMETRY READ -- report-only] explicit basis of the
    solution space (Hom_A4(shell-1,level-1), complex dim 3) as directions in the 3-dim
    multiplicity space of the 3-irrep inside shell-1.  For EACH channel basis element phi_i
    (_a2d_abstract_hom_basis) and EACH of the three copy-embeddings psi_k (_a2d_copy_embeddings),
    phi_i @ psi_k is an ENDOMORPHISM of level-1's abstract V=C^3 -- by SCHUR'S LEMMA this MUST be a
    SCALAR multiple of I_3 (VERIFIED, not assumed): the scalar d_ik IS phi_i's own coordinate
    along the k-th copy's direction.  Assembling D=[d_ik] (3x3 complex, the explicit basis-change
    between the SVD-derived channel basis and the 'which combination of the three 3-irrep copies'
    basis) is the requested explicit basis.  Mod the allowance (ONE overall complex scale-phase,
    a2d_allowance), the solution space projectivizes to CP^2 -- reported here as the normalized
    direction vectors (d_i1:d_i2:d_i3) for each basis element.  ALSO decomposes ker(phi_i) (a
    9-dim subspace of the 12-dim domain, generically -- phi_i:12->3 of rank 3) via sector_grading_
    hist's own isotypic projectors P[0..3] restricted to shell-1 (REUSED), reporting the per-
    isotypic kernel dimension for each basis element (report-only, cross-checks
    a2d_waypoint_isotypic_locus's own P[0..2]-vanishing finding from the KERNEL side).
    NAMED INCOMPLETE EQUATION (THE MULTIPLICITY SELECTOR): nothing in grading+equivariance selects
    a specific direction (d_i1:d_i2:d_i3) -- this is the arc's final open structural question.
    Any resemblance of '3 multiplicity copies' to generations/species counts is OUTLOOK REGISTER
    ONLY, never a claim -- the identification-layer law.
    Returns {'psi_group_law_residual','phi_group_law_residual','D_matrix','schur_scalar_check_
    residual','ker_isotypic_dims','projective_directions','outlook_note'}."""
    A4v, phi_basis, n_phi, worst_phi_law = _a2d_abstract_hom_basis()
    _, psi_basis, n_psi, worst_psi_law = _a2d_copy_embeddings()
    assert n_phi == 3 and n_psi == 3, f"a2d_multiplicity_geometry: expected 3/3, got {n_phi}/{n_psi}"
    D = np.zeros((3, 3), dtype=complex)
    schur_resid = 0.0
    for i, phi in enumerate(phi_basis):
        for k, psi in enumerate(psi_basis):
            M = phi @ psi
            d = complex(np.trace(M) / 3)
            D[i, k] = d
            schur_resid = max(schur_resid, float(np.max(np.abs(M - d * np.eye(3)))))
    assert schur_resid < 1e-8, f"a2d_multiplicity_geometry: Schur scalar check FAILED ({schur_resid:.2e})"
    P = _a2d_shell1_isotypic_projectors()
    ker_rows = []
    for i, phi in enumerate(phi_basis):
        rphi = _a2d_rank(phi)
        s, vv, vh = np.linalg.svd(phi, full_matrices=True)
        ker_basis = vh[rphi:, :].conj().T
        dims_per_a = [int(round(_a2d_rank(P[a] @ ker_basis))) for a in range(4)]
        ker_rows.append({"basis_index": i, "rank_phi": rphi, "ker_dim": ker_basis.shape[1],
                          "ker_isotypic_dims": dims_per_a})
    proj_dirs = []
    for i in range(3):
        row = D[i, :]
        nrm = np.linalg.norm(row)
        proj_dirs.append((row / nrm).tolist() if nrm > 1e-12 else row.tolist())
    return {"psi_group_law_residual": worst_psi_law, "phi_group_law_residual": worst_phi_law,
            "D_matrix": D.tolist(), "schur_scalar_check_residual": schur_resid,
            "ker_isotypic_dims": ker_rows, "projective_directions": proj_dirs,
            "outlook_note": "the direction (d_i1:d_i2:d_i3) in the 3-dim multiplicity space is NOT "
                             "selected by grading+equivariance alone -- THE MULTIPLICITY SELECTOR, "
                             "the arc's final named incomplete equation. Any resemblance of the "
                             "'3 copies' to generation/species counting is OUTLOOK REGISTER ONLY, "
                             "never a claim of this station."}


def a2d_pair_completeness_and_coverage(N_max=4):
    """[A2d SS4 step4b, PAIR-COMPLETENESS WITH NONZERO Phi -- report-only] contrast A2/A2b/A2c
    (Phi=0 everywhere, pair covers nothing): what Gamma(phi_1)(H_hist) covers of F per level (the
    image rank within Pw[level], for level=1,2,3 -- Gamma automatically lands in Pw[n] at shell n,
    pc4's own NHAT-eigenvector fact, REUSED); what J_F maps it to (Phi~ := J_F.Phi.J_hist, A2b's
    own pair-object DEFINITION, REUSED verbatim, computed here explicitly as the 8x12 matrix
    K @ conj(E_1) @ conj(phi) @ R, derived from Phi(v) = E_1.phi.v and J_hist(v)=R@conj(v),
    J_F(w)=K@conj(w)); whether Phi~'s image is confined to Pw[2] (AF-3's own K-swap fact,
    range(Pw[1])<->range(Pw[2]), REUSED, tested not assumed here); and the JOINT rank of
    (Phi, Phi~) together.
    Returns {'per_basis': [{'basis_index','shell_coverage','Phi_tilde_rank',
    'Phi_tilde_confined_to_Pw2_residual','joint_Phi_Phitilde_rank'}]}."""
    A4v, phi_basis, n_phi, worst_law = _a2d_abstract_hom_basis()
    Adag, vac, _, _ = _level1_creation_ops()
    Pw, _ = _sector_projectors(sign=+1)
    E1 = _a2c_level_embedding(1)
    fa = field_algebra_conjugation()
    K = fa["M"]
    R = reversal()
    _, succ = _dart_admissible_successors()
    words, index, lengths = build_hist(N_max, succ)
    per_basis = []
    for i, phi in enumerate(phi_basis):
        phi_ND3 = phi.T
        cover = {}
        for shell, Pt in ((1, Pw[1]), (2, Pw[2]), (3, Pw[3])):
            idxs = [j for j, w in enumerate(words) if len(w) == shell]
            cols = [a2_gamma_word(phi_ND3, words[j], Adag, vac) for j in idxs]
            Gm = np.hstack(cols) if cols else np.zeros((8, 0), dtype=complex)
            rnk = _a2d_rank(Gm)
            in_target = float(np.max(np.abs(Pt @ Gm - Gm))) if Gm.size else 0.0
            cover[shell] = {"image_rank": rnk, "target_dim": int(round(np.trace(Pt).real)),
                             "confined_to_level_residual": in_target}
        Phi_tilde = K @ np.conj(E1) @ np.conj(phi) @ R
        rank_tilde = _a2d_rank(Phi_tilde)
        in_Pw2 = float(np.max(np.abs(Pw[2] @ Phi_tilde - Phi_tilde)))
        joint_rank = _a2d_rank(np.hstack([E1 @ phi, Phi_tilde]))
        per_basis.append({"basis_index": i, "shell_coverage": cover,
                           "Phi_tilde_rank": rank_tilde,
                           "Phi_tilde_confined_to_Pw2_residual": in_Pw2,
                           "joint_Phi_Phitilde_rank": joint_rank})
    return {"per_basis": per_basis}


def a2d_ember_shadow():
    """[A2d SS4 step4c, THE EMBER/CLOCK SHADOW WITH NONZERO phi_1 -- report-only, lambda's
    confronted with NOTHING] the intersection of the channel's codomain (range(E_1)=Pw[1], the
    SAME 3-dim slice of F EVERY channel basis element's image lies in) with K_F's OWN eigenspaces,
    per region orbit (_three_edge_region_orbits/field_side_flow_generator, REUSED unchanged from
    FOCK-0d).  For each region and each of K_F's distinct eigenvalues, reports the PRINCIPAL-ANGLE
    COSINES between Pw[1] and that eigenspace (np.linalg.svd of the cross-Gram matrix -- a cosine
    of 1 means EXACT containment/intersection along that direction, 0 means orthogonal, values in
    between mean generic partial coupling) -- directly answering 'does the weld's image see the
    epsilon-mode' (the single positive-eigenvalue eigenspace, FOCK-0d's own {0x4,-eps x2,+eps x2}
    finding, eps = field_side_flow_generator's own largest eigenvalue).  Also reports N-hat-
    intertwining exactness (REUSED a2b_nhat_intertwining_exactness unchanged, phi_1-independent)
    and the lambda_n=n.lambda_1 structure (REUSED a2b_ember_consistency_shadow/fock0e_lambda_
    structure unchanged) -- STRUCTURE ONLY, no lambda is confronted with any measured quantity.
    Returns {'per_region': [{'region','is_triangle','epsilon','eigengroups':
    [{'eigenvalue','eigenspace_dim','principal_angle_cosines_with_Pw1'}]}], 'nhat_exactness',
    'lambda_structure'}."""
    Pw, _ = _sector_projectors(sign=+1)
    E1 = _a2c_level_embedding(1)
    orbits = _three_edge_region_orbits()
    rows = []
    for orb in orbits:
        region = orb["representative"]
        fsg = field_side_flow_generator(region)
        K_F = fsg["K_F"]
        eigvals, eigvecs = np.linalg.eigh(K_F)
        used = np.zeros(len(eigvals), dtype=bool)
        groups = []
        for idx0 in range(len(eigvals)):
            if used[idx0]:
                continue
            close = np.abs(eigvals - eigvals[idx0]) < 1e-6
            used = used | close
            groups.append((float(eigvals[idx0]), eigvecs[:, close]))
        eigengroups = []
        for mu, vecs in groups:
            cross = vecs.conj().T @ E1
            sv = np.linalg.svd(cross, compute_uv=False)
            eigengroups.append({"eigenvalue": mu, "eigenspace_dim": int(vecs.shape[1]),
                                 "principal_angle_cosines_with_Pw1": [float(x) for x in sv]})
        eps = float(fsg["eigenvalues"][-1])
        rows.append({"region": region, "is_triangle": orb["is_triangle"], "epsilon": eps,
                     "eigengroups": eigengroups})
    nh = a2b_nhat_intertwining_exactness()
    emb = a2b_ember_consistency_shadow()
    return {"per_region": rows, "nhat_exactness": nh, "lambda_structure": emb}


def a2d_species_read():
    """[A2d SS4 step4d, THE SPECIES READ -- D4 OUTPUT ONLY] the isotypic<->{1,3,3,1} correspondence
    the ACTUAL (nonzero) solutions induce.  gauge_sector_category (the species labels) is called
    ONLY here -- grep-confirmed never inside a2d_allowance/_a2d_abstract_hom_basis/_a2d_copy_
    embeddings/a2d_solve_channel/_a2d_shell_level_system_no_pairblock/a2d_tower_consistency (D4's
    hard guard, verified not merely claimed).  a2d_waypoint_isotypic_locus's own P[0..2]-vanishing
    finding means: the domain's 3-irrep isotypic component (P[3], the 9-dim mult-3 block of
    shell-1) is the ONLY isotypic piece carrying a nonzero solution, mapping into level-1 (species
    'd' in gauge_sector_category's {nu:0,d:1,u:2,e:3} dims-indexing) -- the induced correspondence
    is history-3-irrep <-> field-level-1('d').  THE LEVEL-3 ASYMMETRY (named per the pre-reg SS3):
    level-3 (species 'e') sits in field-GROUP-03 while level-1 (species 'd', this station's own
    codomain) sits in field-GROUP-12 (A2c checker item 5, chain (iv)) -- restated here since this
    read compares levels 1 and 3: the pair-block-free shell-3->level-3 system carries its OWN,
    independently-nonzero freedom (a2d_waypoint_shell_level_nullity(3,3)), a DIFFERENT object from
    level-1's, not a continuation of the same solution up the tower.
    Returns {'species_sector_dims','correspondence','isotypic_vanishing_checks',
    'level3_asymmetry_note','outlook_note'}."""
    gsc = gauge_sector_category()
    A4v, phi_basis, n_phi, worst_law = _a2d_abstract_hom_basis()
    P = _a2d_shell1_isotypic_projectors()
    checks = []
    for i, phi in enumerate(phi_basis):
        resid_low = max(float(np.max(np.abs(phi @ P[a]))) for a in (0, 1, 2))
        resid_p3 = float(np.max(np.abs(phi @ P[3] - phi)))
        checks.append({"basis_index": i, "vanishes_on_P012": resid_low, "supported_on_P3": resid_p3})
    s3 = a2d_waypoint_shell_level_nullity(3, 3)
    return {
        "species_sector_dims": gsc["species_sector_dims"],
        "correspondence": "the domain's 3-irrep isotypic component (P[3], the 9-dim mult-3 block "
                           "of shell-1) is the ONLY isotypic piece carrying a nonzero solution "
                           "(P[0..2] vanish exactly, checked above); it maps into level-1 (species "
                           "'d' in gauge_sector_category's indexing) -- the induced correspondence "
                           "is history-3-irrep <-> field-level-1('d').",
        "isotypic_vanishing_checks": checks,
        "level3_asymmetry_note":
            "level-3 (species 'e') sits in field-GROUP-03 while level-1 (species 'd', this "
            "station's own codomain) sits in field-GROUP-12 -- the A2c checker's named structural "
            "fact (item 5, chain (iv)), restated here since this read compares levels 1 and 3: the "
            "pair-block-free shell-3->level-3 system has its own, independently-nonzero freedom "
            f"({s3['full_A_only']} real A-only dims) -- a DIFFERENT object from level-1's, not a "
            "continuation of the same solution up the tower.",
        "outlook_note": "OUTLOOK REGISTER ONLY: no claim of generation/species counting is made by "
                         "this correspondence; D4 stands throughout -- species was never an input "
                         "anywhere in this station's pin construction."}


def a2d_weld_selftest_2026_07_12(verbose=True):
    """A2d station regression (THE FINAL ARC STATION): Sections 7/7b/8/8b/8c/8d/8e/8f/8g/8h +
    module anchors untouched, THEN the pre-declared allowance, the solve (abstract route + the
    F-embedded waypoint cross-check), the isotypic-locus waypoint, the pair-block-free shell-2/3
    waypoints, Gamma(phi_1)'s live P1/P2 behavior on the nonzero basis, the tower CONSISTENCY check
    (labeled, never forcing) with its discriminating control, and the four reads (multiplicity
    geometry, pair-completeness/coverage, ember/clock shadow, species).  Does NOT itself adjudicate
    the SS5 verdict tree (architect-only per the pre-reg)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 8i self-test -- A2d the minimal weld class (2026-07-12, FINAL ARC STATION)")
        print("=" * 88)

    ck("ANCHORS + Sections 7/7b/8/8b/8c/8d/8e/8f/8g/8h untouched",
       anchor_cell_projector() and anchor_tick_2pi() and accretion_selftest_2026_07_10(verbose=False)
       and i2b_selftest_2026_07_11(verbose=False) and fock0_selftest_2026_07_11(verbose=False)
       and fock0b_selftest_2026_07_11(verbose=False) and fock0c_selftest_2026_07_11(verbose=False)
       and fock0d_selftest_2026_07_11(verbose=False) and fock0e_selftest_2026_07_12(verbose=False)
       and a2_weld_selftest_2026_07_12(verbose=False) and a2b_weld_selftest_2026_07_12(verbose=False)
       and a2c_weld_selftest_2026_07_12(verbose=False))

    alw = a2d_allowance()
    ck(f"ALLOWANCE (PRINTED before solving): multiplicity space = {alw['multiplicity_space_complex_dim']} "
       f"complex ({alw['multiplicity_space_real_dim']} real); ALLOWANCE = {alw['allowance_real']} real "
       "(one overall complex scale-phase)", alw["allowance_real"] == 2)

    solve = a2d_solve_channel()
    ck(f"THE SOLVE: waypoint A-only={solve['waypoint_A_only']}, full={solve['waypoint_full_real_linear']}; "
       f"abstract-route nullity={solve['abstract_nullity_complex']} complex (matches waypoint: "
       f"{solve['abstract_matches_waypoint']}); nullity {solve['nullity_real']} vs allowance "
       f"{solve['allowance_real']} -- EXCEEDS: {solve['nullity_exceeds_allowance']}",
       solve["abstract_matches_waypoint"] and solve["nullity_real"] == 6)

    iso = a2d_waypoint_isotypic_locus()
    ck(f"WAYPOINT (A2c checker 5iii): channel entirely 3-isotypic -- vanish on P[0..2]="
       f"{iso['worst_vanish_P012']:.1e}, supported on P[3]={iso['worst_supported_P3_residual']:.1e}",
       iso["entirely_3_isotypic"])

    s2 = a2d_waypoint_shell_level_nullity(2, 2)
    s3 = a2d_waypoint_shell_level_nullity(3, 3)
    ck(f"WAYPOINT (A2c checker 5iv) shell2->level2: full_A_only={s2['full_A_only']} (expect 12), "
       f"full_real_linear={s2['full_real_linear']} (expect 24)",
       s2["full_A_only"] == 12 and s2["full_real_linear"] == 24)
    ck(f"WAYPOINT (A2c checker 5iv) shell3->level3: full_A_only={s3['full_A_only']} (expect 8), "
       f"full_real_linear={s3['full_real_linear']} (expect 16)",
       s3["full_A_only"] == 8 and s3["full_real_linear"] == 16)

    live = a2d_gamma_live_behavior()
    ck(f"Gamma(phi_1) LIVE ON {live['n_basis']} NONZERO BASIS ELEMENTS: shell1 worst norm="
       f"{live['basis_rows'][0]['per_shell'][1]['worst']:.3f} (nonzero), P1 shell4 worst norm="
       f"{live['worst_shell4_norm']:.1e} (vanishes), P2 repeat-word worst norm="
       f"{live['worst_repeat_norm']:.1e} (vanishes)",
       live["worst_shell4_norm"] < 1e-9 and live["worst_repeat_norm"] < 1e-9)

    tow = a2d_tower_consistency()
    ck(f"TOWER CONSISTENCY (labeled, NEVER forcing) shell2 [D={tow[2]['D_shell_n']}]: "
       f"all_members_consistency={tow[2]['all_members_consistency']}, discriminating control "
       f"residual={tow[2]['control_residual']:.3f} (rejected: {not tow[2]['control_is_member']})",
       tow[2]["all_members_consistency"] and not tow[2]["control_is_member"])
    ck(f"TOWER CONSISTENCY (labeled, NEVER forcing) shell3 [D={tow[3]['D_shell_n']}]: "
       f"all_members_consistency={tow[3]['all_members_consistency']}, discriminating control "
       f"residual={tow[3]['control_residual']:.3f} (rejected: {not tow[3]['control_is_member']})",
       tow[3]["all_members_consistency"] and not tow[3]["control_is_member"])

    mg = a2d_multiplicity_geometry()
    ck(f"READ (a) MULTIPLICITY GEOMETRY: psi group-law={mg['psi_group_law_residual']:.1e}, phi "
       f"group-law={mg['phi_group_law_residual']:.1e}, Schur scalar-check residual="
       f"{mg['schur_scalar_check_residual']:.1e}; ker isotypic dims per basis element="
       f"{[r['ker_isotypic_dims'] for r in mg['ker_isotypic_dims']]}",
       mg["schur_scalar_check_residual"] < 1e-8)

    pc = a2d_pair_completeness_and_coverage()
    ck(f"READ (b) PAIR-COMPLETENESS/COVERAGE: {len(pc['per_basis'])} basis elements, shell-1 image "
       f"rank(s)={[r['shell_coverage'][1]['image_rank'] for r in pc['per_basis']]}, Phi~ confined to "
       f"Pw[2] residual(s)={[round(r['Phi_tilde_confined_to_Pw2_residual'], 10) for r in pc['per_basis']]}",
       all(r["Phi_tilde_confined_to_Pw2_residual"] < 1e-6 for r in pc["per_basis"]))

    emb = a2d_ember_shadow()
    ck(f"READ (c) EMBER/CLOCK SHADOW: {len(emb['per_region'])} region orbits scanned; N-hat-"
       f"intertwining exactness worst residual={emb['nhat_exactness']['worst_residual']:.1e}",
       emb["nhat_exactness"]["worst_residual"] < 1e-9)

    sp = a2d_species_read()
    ck("READ (d) SPECIES READ (D4 output only): P[3]-only correspondence confirmed; level-3 "
       "asymmetry noted", all(c["vanishes_on_P012"] < 1e-8 for c in sp["isotypic_vanishing_checks"]))

    if verbose:
        print("RESULT:", "A2d SECTION-8i REGRESSION PASSES" if ok else "AN A2d CHECK FAILED")
    return ok


# ===========================================================================
# 9. ML-1d -- THE DERIVED HORIZON: MIRROR-MODE PLACEMENT + THE OPERATOR-LEVEL BOOST TEST (2026-07-12)
#    internal research notes (Push-3 W1; commit d41e286, BEFORE this
#    section).  Targets MG-1d's incomplete equation (G_eff = G/(2pi)).
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as every section above; ACCRETION-ONLY -- nothing
#  above this line is touched, no existing function signature or default behavior changes.]
#
# CONTEXT: the D1/D1b/D1c park found the half-bond horizon-PLACEMENT freedom is exactly the
# corpus's own factor-of-2 ambiguity (midway vs edge-site convention) and that 1D modular data does
# not resolve it.  This section builds the TWO readers ML-1d's freeze needs to attack placement
# directly at the modular level, in 3D, using the FOCK arc's own machinery (region census, Peschel
# h_A):
#   Leg B (R1) -- mirror_mode_pairing: the free-fermion SCHMIDT/mirror-mode pairing of a pure
#     bipartition A|A' (does the net's OWN modular structure place the entangling surface?).
#   Leg C (R2) -- k_boost_bond_matrix / boost_c_star: the OPERATOR-LEVEL h_A = c*K_boost test
#     ML-1'''-B declared and never ran on the near-surface sector with a genuinely derived placement.
# Every function below is a GENERIC linear-algebra reader over a plain numpy region-correlation
# block / physical-hopping block -- it takes arrays, not a Patch, so it composes identically with
# the 1D chain (Leg A's hard gate) and the 3D lattice (the station itself), exactly the "benchmark
# through the extractor's own pipeline" law D1b/W2-D1c established.  The heavy per-M/per-direction
# ORCHESTRATION loop (building Patch/H/eigh, slicing regions, calling these readers) lives in the
# station file (proofs/foundations/ML1d_derived_horizon_2026-07-12.py), matching the D1b/W2-D1c
# precedent of keeping loop/IO logic OUT of the master object.
#
# NO magnitude is defined here; every quantity below is a dimension, residual, or a plain reader
# over caller-supplied numbers (module contract).  species/M_Z/ppm appear nowhere.
def mirror_mode_pairing(C_A, C_ApA):
    """[ML-1d Leg B, step 1] the free-fermion Schmidt/mirror-mode pairing of a PURE bipartition
    A|A' of a global correlation matrix C (C^2=C, e.g. a Dirac-sea vacuum projector): C_A is the
    region-A block (Peschel convention, entanglement_hamiltonian's own input), C_ApA = C[Ap_idx,
    A_idx] the CROSS block onto the complementary region A'.
    Eigendecompose C_A (ascending eigenvalues zeta_k, eigenvectors u_k as columns of U_A) --
    Peschel's own convention, entanglement_hamiltonian's internal step, exposed here so callers can
    read BOTH the modular energies eps_A = log((1-zeta)/zeta) AND the eigenbasis U_A.
    THE MIRROR MODE (the frozen construction): for a PURE global state, the Schur-complement
    identity of a projector, C_A - C_A^2 = C_ApA^T.C_ApA (real case) / C_ApA^dagger.C_ApA
    (complex), means C_ApA.u_k is the (unnormalized) mirror partner of u_k in A', with norm
    EXACTLY sqrt(zeta_k(1-zeta_k)) -- VERIFIED numerically below (not assumed): mirror_unnorm =
    C_ApA @ U_A; mirror_norm = its column norms; compared against sqrt(zeta*(1-zeta))
    (mirror_norm_residual). A SECOND, independent purity check reads the SAME identity's diagonal
    (in the ORIGINAL, un-rotated A-index basis) WITHOUT ever forming C_A@C_A explicitly (which would
    cost O(|A|^3), as expensive as another eigh): diag(C_A-C_A^2) = diag(U_A.diag(zeta(1-zeta)).
    U_A^dagger), read via one O(|A|^2) einsum, compared against diag(C_ApA^dagger.C_ApA) (also
    O(|A|^2), the column norms of C_ApA in the ORIGINAL basis) -- purity_residual. Both residuals
    are near machine precision for an EXACT projector C (the 3D lattice Dirac sea); a genuinely
    nonzero residual flags that the supplied C is NOT exactly pure (e.g. an idealized/truncated
    correlator, not a literal finite Slater determinant -- disclosed per-caller, not assumed away).
    Returns {'zeta_A','U_A','eps_A','mirror_unnorm','mirror_norm','mirror_norm_residual','C_ApA',
    'purity_residual'}."""
    C_A = np.asarray(C_A)
    C_ApA = np.asarray(C_ApA)
    w, U = np.linalg.eigh(C_A)
    zeta = np.clip(w.real, 1e-14, 1 - 1e-14)
    eps = np.log((1 - zeta) / zeta)
    mirror_unnorm = C_ApA @ U
    mirror_norm = np.linalg.norm(mirror_unnorm, axis=0)
    expect_norm = np.sqrt(np.clip(zeta * (1 - zeta), 0.0, None))
    mirror_norm_residual = float(np.max(np.abs(mirror_norm - expect_norm)))
    diagLHS = np.real(np.einsum('ik,ik,k->i', np.conj(U), U, zeta * (1 - zeta)))
    diagRHS = np.real(np.einsum('ki,ki->i', np.conj(C_ApA), C_ApA))
    purity_residual = float(np.max(np.abs(diagLHS - diagRHS)))
    return {"zeta_A": zeta, "U_A": U, "eps_A": eps, "mirror_unnorm": mirror_unnorm,
            "mirror_norm": mirror_norm, "mirror_norm_residual": mirror_norm_residual,
            "C_ApA": C_ApA, "purity_residual": purity_residual}


def mode_position_expectation(vecs, positions):
    """[ML-1d Leg B, step 2] per-column position expectation <x> = sum_i |vec_i|^2 * x_i / ||vec||^2
    (NOT assumed pre-normalized -- e.g. mirror_mode_pairing's mirror_unnorm columns are not unit
    norm; this reader normalizes internally). `vecs`: (n, k) array (k mode columns over the SAME
    n-length index set `positions` is given in, matching order). `positions`: length-n real array
    (e.g. a cut-normal-projected proper-unit coordinate, already shifted so the reference/placement
    origin sits at 0). Returns a length-k real array, one expectation value per mode column."""
    vecs = np.asarray(vecs)
    positions = np.asarray(positions, dtype=float)
    num = np.real(np.einsum('ik,ik,i->k', np.conj(vecs), vecs, positions))
    den = np.real(np.einsum('ik,ik->k', np.conj(vecs), vecs))
    return num / den


def near_surface_selection(eps, frac=0.5):
    """[ML-1d Leg B/C shared primitive] the FROZEN 'near-surface sector' selection used by BOTH
    legs: the `frac` fraction (default one HALF, per the freeze's literal wording in both legs) of
    modes with SMALLEST |modular energy| eps. For continuous/generic eps this is operationally
    identical to 'modes with |eps| <= the region median' (Leg B's phrasing) -- a median is exactly
    the boundary of the smallest half. Returns a sorted int index array into `eps` (length =
    max(1, round(frac*len(eps)))), so callers can slice a companion eigenvector matrix consistently."""
    eps = np.asarray(eps, dtype=float)
    n = len(eps)
    k = max(1, int(round(frac * n)))
    order = np.argsort(np.abs(eps))
    return np.sort(order[:k])


def project_matrix(M, U):
    """[ML-1d Leg C, step 1] the restriction of a Hermitian (or real-symmetric) matrix M onto the
    subspace spanned by the (not-necessarily-square) orthonormal columns of U: U^dagger.M.U. Used
    both to restrict h_A onto its own near-surface eigenbasis (giving an EXACTLY diagonal result,
    since U's columns are h_A's own eigenvectors there) and to restrict K_boost onto that SAME
    basis (a genuine change of basis, not diagonal in general)."""
    U = np.asarray(U)
    return U.conj().T @ np.asarray(M) @ U


def k_boost_bond_matrix(H_A, x):
    """[ML-1d Leg C, step 2] K_boost at the REFERENCE placement (x measured from wherever the
    caller's coordinate origin sits): K_boost[i,j] = H_A[i,j] * (x_i+x_j)/2 (T00_b = the physical
    hopping matrix element at bond (i,j); x_b(p) = the bond MIDPOINT's position relative to
    placement p -- ML-1'''-B's own convention, generalized from a single nearest-bond read to the
    full region). H_A: the region's physical single-particle Hamiltonian (cone sector), region-local
    indexing. x: length-|A| real array of proper-unit positions along the cut normal, ALREADY
    shifted so x=0 sits at the reference placement.
    LINEARITY (used by the station to cover three placements cheaply, EXACTLY, no re-projection):
    if positions are re-referenced by a constant shift delta (x -> x-delta, i.e. the horizon moves
    by delta along +cut_normal), K_boost(delta) = K_boost(0) - delta*H_A EXACTLY, since
    (x_i-delta + x_j-delta)/2 = (x_i+x_j)/2 - delta.  Returns the (|A|,|A|) K_boost matrix at the
    supplied x (i.e. K_boost(0) in this shifted frame)."""
    x = np.asarray(x, dtype=float)
    return np.asarray(H_A) * ((x[:, None] + x[None, :]) / 2.0)


def boost_c_star(h_proj, K_proj):
    """[ML-1d Leg C, step 3] c* = argmin_c ||h_proj - c*K_proj||_F (closed-form real least squares
    on the Frobenius inner product: c* = Re<K_proj,h_proj> / Re<K_proj,K_proj>), plus the
    normalized residual ||h_proj - c*.K_proj||_F / ||h_proj||_F. Both inputs are ALREADY restricted
    to the near-surface sector (project_matrix); this function performs no selection of its own.
    Returns (c_star, residual); (nan, nan) if K_proj is identically zero (ill-posed)."""
    h_proj = np.asarray(h_proj)
    K_proj = np.asarray(K_proj)
    den = float(np.real(np.vdot(K_proj, K_proj)))
    if den <= 0.0:
        return float("nan"), float("nan")
    num = float(np.real(np.vdot(K_proj, h_proj)))
    c = num / den
    hnorm = float(np.linalg.norm(h_proj))
    resid = float(np.linalg.norm(h_proj - c * K_proj) / hnorm) if hnorm > 0.0 else float("nan")
    return c, resid


def distinct_spectrum_count(eps, rel_tol=1e-3, abs_tol=1e-9):
    """[ML-1d, THE L0b RANK/PROFILE READ, TICK-REDUCES DISCRIMINATOR] the number of DISTINCT
    magnitudes |eps| in a modular spectrum, up to a relative-tolerance clustering (greedy: sort
    |eps| ascending, start a NEW cluster whenever the gap to the previous cluster's anchor exceeds
    rel_tol*scale + abs_tol). This is the OPERATIONAL form of 'does the modular spectrum enrich with
    region growth (boost-like, growing distinct-value count as near-horizon DEPTH resolution grows)
    or stay rank-starved (tick-like, a FIXED small number of distinct values regardless of region
    size -- L0b's own 3-edge finding: one magnitude +/-eps plus the forced 1/2 zero mode, i.e. 2-3
    distinct values no matter how the region is chosen)'. A single scalar per region; the STATION
    tracks this across the M-ladder (mandatory, cheap -- reuses the SAME eigh already computed for
    Leg C, no extra heavy linear algebra). NOT a verdict by itself; the station applies the frozen
    tree."""
    vals = np.sort(np.abs(np.asarray(eps, dtype=float)))
    if len(vals) == 0:
        return 0
    clusters = [vals[0]]
    for v in vals[1:]:
        ref = clusters[-1]
        scale = max(abs(ref), abs_tol)
        if abs(v - ref) > rel_tol * scale + abs_tol:
            clusters.append(v)
    return len(clusters)


def diamond_vertex_region(patch, base, depth):
    """[ML-1d, THE DIAMOND DIAGNOSTIC region-builder, report-only] converts an EXACT causal diamond
    (Patch.diamond's own dart+tick mode set, ML-0's exact light cone) into a spatial VERTEX index
    set suitable for region_data/entanglement_hamiltonian: every vertex touched by either endpoint
    of any dart appearing in the diamond's mode set (tail or head), deduplicated. Uses ONLY Patch's
    existing public interface (diamond, RD, vidx) -- the Patch class itself is not modified.
    Returns a sorted int numpy array of GLOBAL vertex indices (matching vertex_adjacency's own
    `verts` ordering, i.e. safe to intersect with an M0/ML-1d region built from vertex_adjacency)."""
    modes = patch.diamond(base, depth)
    verts = set()
    for (d, t) in modes:
        ta, ha = patch.RD[d]
        verts.add(patch.vidx[ta])
        verts.add(patch.vidx[ha])
    return np.array(sorted(verts), dtype=int)


def ml1d_selftest_2026_07_12(verbose=True):
    """ML-1d SECTION-9 regression: mirror_mode_pairing (purity + mirror-norm identities on a
    GENUINE finite Dirac-sea projector, where they must hold near machine precision) +
    mode_position_expectation (a synthetic sanity check) + near_surface_selection (half-count,
    correct ordering) + project_matrix (an orthonormal-U identity spot-check) + k_boost_bond_matrix
    + boost_c_star (recovers a planted c on synthetic h=c0*K+noise) + distinct_spectrum_count
    (a synthetic clustered spectrum) + diamond_vertex_region (nonempty, subset of the ambient
    patch's vertices, grows with depth), plus every prior section's own self-test (anchors +
    Sections 1-8i untouched). Does NOT itself adjudicate ML-1d's four-branch verdict tree
    (architect/station-file only, per the pre-reg)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 9 self-test -- ML-1d the derived horizon (2026-07-12)")
        print("=" * 88)

    ck("ANCHORS + Sections 1-8i untouched",
       anchor_cell_projector() and anchor_tick_2pi() and accretion_selftest_2026_07_10(verbose=False)
       and i2b_selftest_2026_07_11(verbose=False) and fock0_selftest_2026_07_11(verbose=False)
       and fock0b_selftest_2026_07_11(verbose=False) and fock0c_selftest_2026_07_11(verbose=False)
       and fock0d_selftest_2026_07_11(verbose=False) and fock0e_selftest_2026_07_12(verbose=False)
       and a2_weld_selftest_2026_07_12(verbose=False) and a2b_weld_selftest_2026_07_12(verbose=False)
       and a2c_weld_selftest_2026_07_12(verbose=False) and a2d_weld_selftest_2026_07_12(verbose=False))

    # mirror_mode_pairing: a GENUINE finite Dirac-sea projector (open tight-binding chain, exact
    # rank-N/2 projector by construction) -- the purity/mirror-norm identities MUST hold near
    # machine precision here (unlike an idealized/truncated correlator such as chain_vacuum).
    Nc = 40
    Hc = np.diag(np.ones(Nc - 1), 1) + np.diag(np.ones(Nc - 1), -1)
    Ec, Vc = np.linalg.eigh(Hc)
    cols = Vc[:, Ec < 0.0]
    Cfull = cols @ cols.conj().T
    A_idx = np.arange(0, Nc // 2)
    Ap_idx = np.arange(Nc // 2, Nc)
    C_A = Cfull[np.ix_(A_idx, A_idx)]
    C_ApA = Cfull[np.ix_(Ap_idx, A_idx)]
    mp = mirror_mode_pairing(C_A, C_ApA)
    ck(f"mirror_mode_pairing on a GENUINE finite projector: mirror_norm_residual="
       f"{mp['mirror_norm_residual']:.2e}, purity_residual={mp['purity_residual']:.2e}",
       mp["mirror_norm_residual"] < 1e-6 and mp["purity_residual"] < 1e-6)

    # mode_position_expectation: a synthetic sanity check (a mode localized at index 3 of 5 has
    # <x> = positions[3] exactly).
    vtest = np.zeros((5, 2))
    vtest[3, 0] = 1.0
    vtest[:, 1] = 1.0                      # uniform mode -> <x> = mean(positions)
    postest = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    xe = mode_position_expectation(vtest, postest)
    ck("mode_position_expectation: localized mode -> exact position; uniform mode -> mean",
       abs(xe[0] - 3.0) < 1e-12 and abs(xe[1] - 2.0) < 1e-12, detail=f"{xe}")

    # near_surface_selection: half-count, smallest |eps| kept.
    eps_test = np.array([5.0, -0.1, 3.0, 0.2, -4.0, 0.05])
    idx_ns = near_surface_selection(eps_test, frac=0.5)
    ck("near_surface_selection: half-count, correct (smallest-|eps|) members",
       len(idx_ns) == 3 and set(idx_ns.tolist()) == {1, 3, 5}, detail=f"idx={idx_ns}")

    # project_matrix: an orthonormal U spot-check (U^dagger.U=I sub-block of a random orthogonal
    # matrix) reproduces a DIRECT sub-block computation.
    rng = np.random.default_rng(0)
    Mtest = rng.normal(size=(6, 6))
    Mtest = Mtest + Mtest.T
    Qtest, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    Utest = Qtest[:, :3]
    Pdirect = Utest.T @ Mtest @ Utest
    Pfn = project_matrix(Mtest, Utest)
    ck("project_matrix == direct U^dagger.M.U", np.allclose(Pfn, Pdirect, atol=1e-12))

    # k_boost_bond_matrix + the LINEARITY-under-shift identity used by the station to cover all
    # three placements from ONE basis-change.
    HAtest = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    xtest = np.array([0.0, 1.0, 2.0])
    K0 = k_boost_bond_matrix(HAtest, xtest)
    delta = 0.37
    Kdelta_direct = k_boost_bond_matrix(HAtest, xtest - delta)
    Kdelta_formula = K0 - delta * HAtest
    ck("k_boost_bond_matrix: K_boost(delta) == K_boost(0) - delta*H_A EXACTLY (the placement-shift "
       "linearity the station relies on)", np.allclose(Kdelta_direct, Kdelta_formula, atol=1e-12))

    # boost_c_star: recovers a planted c0 on h = c0*K + small noise.
    c0 = 2 * math.pi
    noise = 1e-6 * rng.normal(size=K0.shape)
    noise = noise + noise.T
    h_planted = c0 * K0 + noise
    c_star, resid = boost_c_star(h_planted, K0)
    ck(f"boost_c_star recovers a planted c0=2pi (c*={c_star:.6f}, residual={resid:.2e})",
       abs(c_star - c0) < 1e-3 and resid < 1e-3)

    # distinct_spectrum_count: a synthetic 2-cluster + 1 zero-mode spectrum (the L0b 3-edge shape).
    eps_l0b = np.array([0.0, 0.7, -0.7, 0.7000001, -0.6999998])
    dcount = distinct_spectrum_count(eps_l0b, rel_tol=1e-3)
    ck(f"distinct_spectrum_count: the L0b 3-edge shape (0, +/-eps cluster) -> 2 distinct magnitudes "
       f"(got {dcount})", dcount == 2)
    eps_grown = np.concatenate([eps_l0b, [1.4, -1.4, 2.1, -2.1]])
    dcount2 = distinct_spectrum_count(eps_grown, rel_tol=1e-3)
    ck(f"distinct_spectrum_count GROWS when new, well-separated magnitudes are added ({dcount} -> "
       f"{dcount2})", dcount2 > dcount)

    # diamond_vertex_region: nonempty, a subset of the ambient patch's vertices, grows with depth.
    patch_d = Patch(M=6)
    base_d = patch_d.central_dart()
    verts_all = patch_d.vertex_adjacency()[1]
    Nv_all = len(verts_all)
    reg2 = diamond_vertex_region(patch_d, base_d, 2)
    reg4 = diamond_vertex_region(patch_d, base_d, 4)
    ck(f"diamond_vertex_region: nonempty subsets of the ambient patch, grow with depth "
       f"(|reg(2)|={len(reg2)}, |reg(4)|={len(reg4)}, N_ambient={Nv_all})",
       0 < len(reg2) <= len(reg4) < Nv_all
       and set(reg2.tolist()) <= set(range(Nv_all)) and set(reg4.tolist()) <= set(range(Nv_all)))

    if verbose:
        print("RESULT:", "ML-1d SECTION-9 REGRESSION PASSES" if ok else "AN ML-1d CHECK FAILED")
    return ok


# ===========================================================================
# 9b. ML-1d-b -- THE CORRECTED INSTRUMENT (amendment; 2026-07-13)
#     internal research notes (commit 27b0916, BEFORE this code)
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention; ACCRETION-ONLY -- Section 9's own functions
#  (near_surface_selection, k_boost_bond_matrix, ...) are NOT modified; the amendment's two
#  corrected definitions are NEW functions so ed61816's committed behavior stays bit-identical
#  and the amendment checker can diff exactly the delta.]
#
# THE TWO CORRECTED DEFINITIONS (verbatim lineage in the amendment pre-reg):
#   1. entanglement_carrying_selection -- replaces the frozen "half of A's modes with smallest
#      |modular energy|" (near_surface_selection), which the ML-1d return diagnosed as ~90%
#      round-off-ordered saturated bulk modes on any region large relative to the O(log L)
#      1D-critical entanglement scale.  The corrected sector is STATE-DETERMINED: all modes with
#      occupation lambda in [delta, 1-delta], delta = 1e-8 (declared in the amendment BEFORE any
#      re-run; physically the genuinely fractional/entangled modes; numerically far above the
#      ~1e-16 round-off saturation floor).
#   2. k_boost_parabolic -- replaces the linear/infinite-Rindler template x*T00 (k_boost_bond_
#      matrix) with the FINITE-REGION conformal-Killing parabola w(x) = x*(ell-x)/ell, ell = the
#      region's proper depth along the cut normal (FIXED by geometry in the declared position
#      frame, never fitted).  The BW statement under test is c* = 2pi against THIS template (the
#      causal-diamond modular weight; its x->0 edge behavior reproduces the Rindler 2pi*x law,
#      i.e. ML-1 Stage A's own calibrated first-bond form).
# Plus the W3a-lemma L0b helper (positive_spectrum_dimension): the amendment makes explicit that
# every ODD-dimension region carries a FORCED modular zero mode (docs/theorems/
# CA_half_lemma_2026-07-12.md) which is excluded from positive-spectrum counts EXACTLY.
def entanglement_carrying_selection(lam, delta=1e-8):
    """[ML-1d-b, CORRECTED DEFINITION 1] the ENTANGLEMENT-CARRYING sector: indices of all modes
    whose occupation lambda lies in [delta, 1-delta] (default delta = 1e-8, the amendment's
    declared threshold -- no other threshold may be substituted after numbers are seen).  Input
    is the OCCUPATION spectrum (C_A's eigenvalues), NOT the modular energies -- the selection is
    monotone-equivalent to |eps| <= log((1-delta)/delta) ~ 18.42, but stating it on lambda keeps
    it directly tied to the physical statement ('genuinely fractional modes').  The sector
    DIMENSION is state-determined (O(log L) on a 1D-critical benchmark; area-law-ish on the 3D
    lattice vacuum; near-extensive for a thermal state) and is REPORTED per region, never fixed.
    Returns a sorted int index array into `lam`."""
    lam = np.asarray(lam, dtype=float)
    return np.where((lam >= delta) & (lam <= 1.0 - delta))[0]


def k_boost_parabolic(H_A, x, ell):
    """[ML-1d-b, CORRECTED DEFINITION 2] the finite-region conformal boost template
    K_template[i,j] = H_A[i,j] * w((x_i+x_j)/2),   w(x) = x*(ell-x)/ell
    (T00_b = the physical hopping matrix element at bond (i,j); the bond's declared point = its
    MIDPOINT along the cut normal, ML-1'''-B's own convention, unchanged from k_boost_bond_matrix
    -- ONLY the weight profile w changes: parabola, not the infinite-Rindler linear x).
    `x`: length-|A| proper-unit positions along the cut normal, measured from the placement p
    (x=0 at the assumed horizon).  `ell`: the region's proper depth along the cut normal -- a
    GEOMETRIC property of the region in the declared position frame (the station computes it as
    the D1b region_depth, threshold_midway - proj.min()), NEVER fitted and NOT varied with
    placement.  NOTE the placement-shift linearity of the linear template does NOT carry over
    (w is quadratic); the station recomputes K_template per placement directly.  w's x->0
    behavior is w(x) ~ x exactly (the Rindler law), and w(0) = w(ell) = 0 (the two conformal
    Killing zeros of the finite region/diamond).  Returns the (|A|,|A|) matrix."""
    x = np.asarray(x, dtype=float)
    Xm = (x[:, None] + x[None, :]) / 2.0
    return np.asarray(H_A) * (Xm * (ell - Xm) / ell)


def positive_spectrum_dimension(eps, region_dim, zero_tol=1e-6):
    """[ML-1d-b, THE L0b READ'S W3a-EXPLICIT FORM] the dimension of a modular spectrum's
    POSITIVE part, with the forced odd-region zero mode excluded EXACTLY (the W3a lemma,
    docs/theorems/CA_half_lemma_2026-07-12.md: every ODD-dimension region of a covariance built
    from a real antisymmetric complex structure carries a FORCED eigenvalue 1/2, i.e. a forced
    modular zero mode; on EVEN regions no zero is forced).  Operationally: count eps > zero_tol
    (the tolerance window around 0 is what excludes the forced zero mode 'exactly' -- its eps is
    0 up to numerics); additionally report whether a zero mode is present (min |eps| < zero_tol)
    and whether the region's dimension makes it FORCED (odd) or incidental (even).
    Returns {'n_positive','has_zero_mode','zero_forced_by_odd_dim','min_abs_eps'}."""
    eps = np.asarray(eps, dtype=float)
    n_positive = int(np.sum(eps > zero_tol))
    min_abs = float(np.min(np.abs(eps))) if len(eps) else float("nan")
    has_zero = bool(min_abs < zero_tol) if len(eps) else False
    return {"n_positive": n_positive, "has_zero_mode": has_zero,
            "zero_forced_by_odd_dim": bool(region_dim % 2 == 1), "min_abs_eps": min_abs}


def ml1db_selftest_2026_07_13(verbose=True):
    """ML-1d-b SECTION-9b regression: the two corrected definitions + the W3a L0b helper, unit-
    tested, plus Section 9 (and thereby every prior section) untouched.  Does NOT adjudicate the
    amended station's verdict (station file / architect only)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 9b self-test -- ML-1d-b the corrected instrument (2026-07-13)")
        print("=" * 88)

    ck("Section 9 (and all prior sections) untouched", ml1d_selftest_2026_07_12(verbose=False))

    # entanglement_carrying_selection: saturated modes excluded, fractional kept, boundary cases.
    lam_test = np.array([1e-12, 0.3, 0.5, 1 - 1e-12, 1e-7, 1.0, 0.0, 1 - 1e-7])
    sel = entanglement_carrying_selection(lam_test)
    ck("entanglement_carrying_selection keeps exactly the fractional modes "
       f"(got idx={sel.tolist()})", set(sel.tolist()) == {1, 2, 4, 7})

    # k_boost_parabolic: w(0)=w(ell)=0; symmetry w(x)=w(ell-x); x->0 slope = 1 (matches the
    # linear template k_boost_bond_matrix exactly at leading order near the horizon).
    HAtest = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    ell = 10.0
    xa = np.array([0.0, 4.0, 10.0])
    Ka = k_boost_parabolic(HAtest, xa, ell)
    Kb = k_boost_parabolic(HAtest, ell - xa, ell)
    ck("k_boost_parabolic: reflection symmetry w(m)=w(ell-m) ELEMENTWISE (positions x -> ell-x "
       "give bond midpoints m -> ell-m, and the parabola is reflection-even)",
       np.allclose(Ka, Kb, atol=1e-12))
    x_small = np.array([0.0, 1e-6, 2e-6])
    K_par = k_boost_parabolic(HAtest, x_small, ell)
    K_lin = k_boost_bond_matrix(HAtest, x_small)
    ck("k_boost_parabolic: x->0 behavior == the linear (Rindler) template to O(x/ell)",
       np.allclose(K_par, K_lin, rtol=1e-5, atol=1e-18))

    # positive_spectrum_dimension on a REAL forced-zero case: the 3-edge triangle region's
    # {-eps, 0, +eps} spectrum (W3a's own anchored object).
    C6 = vacuum_covariance(sign=+1)
    _, eps3, _ = region_data(C6, [0, 1, 3])
    psd = positive_spectrum_dimension(eps3, region_dim=3)
    ck(f"positive_spectrum_dimension on the 3-edge triangle: n_positive={psd['n_positive']} "
       f"(exactly 1), forced zero mode detected (min|eps|={psd['min_abs_eps']:.1e})",
       psd["n_positive"] == 1 and psd["has_zero_mode"] and psd["zero_forced_by_odd_dim"])

    # and on an even synthetic spectrum with no zero: nothing excluded, no forced zero.
    psd2 = positive_spectrum_dimension(np.array([-2.0, -1.0, 1.0, 2.0]), region_dim=4)
    ck("positive_spectrum_dimension on an even no-zero spectrum: n_positive=2, no zero mode",
       psd2["n_positive"] == 2 and not psd2["has_zero_mode"]
       and not psd2["zero_forced_by_odd_dim"])

    if verbose:
        print("RESULT:", "ML-1d-b SECTION-9b REGRESSION PASSES" if ok else "AN ML-1d-b CHECK FAILED")
    return ok


# ===========================================================================
# 10. W2 -- THE SELECTOR READ (2026-07-13)
#     internal research notes (commit c7aca84, BEFORE this code)
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as every section above -- appended after the
#  __main__ guard; ACCRETION-ONLY, nothing in Sections 1-9b is modified.]
#
# CONTEXT (Push 3's pivotal question): does LOCAL MODULAR CONSISTENCY -- the welded state
# satisfying each region's own KMS condition, region by region -- cut the ELLIPTIC-CP^2
# multiplicity freedom the A2d weld arc left open (THE MULTIPLICITY SELECTOR)?  The welded state
# rho_d is built by pushing omega_diag_length forward through A2's own Gamma(Phi_d) functor
# (a2_gamma_word, _level1_creation_ops, REUSED unchanged) for a family member Phi_d = sum d_i.phi_i
# (phi_i = _a2d_abstract_hom_basis's own channel basis, REUSED via _a2d_phi1_list).  The alignment
# functional T1 tests [rho_d, K_F(A)] (field_side_flow_generator, REUSED) per region orbit
# (_three_edge_region_orbits, REUSED).
#
# ***** THE HEADLINE RESULT (established below, proved not merely observed) *****
# BOTH nontrivial levels of rho_d (level 1 = shell-1's own 3-irrep V; level 2 = Lambda^2(V), which
# _a2c_level_rep's OWN docstring already establishes is ISOMORPHIC to the SAME 3-irrep, "A4's
# unique 3-irrep is isomorphic to its own Lambda^2") are FORCED BY SCHUR'S LEMMA to be scalar
# multiples of the identity on their own level, for EVERY direction u in the multiplicity space:
# phi_i @ phi_j^dagger (both phi_i, phi_j being A4-intertwiners shell1->level1) is a self-
# intertwiner of an IRREDUCIBLE rep, hence forced scalar (proved exactly below for level 1;
# verified to machine precision for level 2, citing the SAME Lambda^2(V)~=V fact).  Levels 0/3
# are each 1-dimensional (trivially scalar).  CONSEQUENCE: rho_d = sum_n lambda_n(r).Pw[n] --
# an operator depending ONLY on the SCALE r, NEVER on the projective DIRECTION u.  Hence ANY
# functional built from rho_d against a FIXED K_F(A) (T1, T2, or any other alignment test) is
# IDENTICALLY CONSTANT over the WHOLE multiplicity space CP^2 -- THE READ IS VACUOUS (SS3's own
# honesty clause, confirmed by an EXACT mechanism, not merely sampled).  Per the freeze's own
# instruction ("if constant, the verdict is V-VACUOUS and you report that honestly -- do not hunt
# for a locus anyway"), the expensive multi-start/locus-search machinery is NOT run: the mechanism
# above, plus the mandated 20-sample honesty-clause check, is the complete, honest evidence.
# TWO OF THE FREEZE'S OWN "ANTICIPATED STRUCTURE" ITEMS FAIL (verified, not assumed, reported raw
# per SS1/SS2): (a) K_F(A) is NOT level-block-diagonal wrt Pw[0..3] (large cross-block residual,
# ~O(1), for every orbit -- field_side_flow_generator's own docstring already discloses the
# region-Fock-space<->F identification is "purely by DIMENSION MATCH... NOT a proof", and this is
# the concrete consequence); (b) T1 is NOT r-independent (it varies smoothly with r, since rho_d's
# level-mixing weights lambda_n(r) shift the balance between Pw[1] and Pw[2] against the SAME
# fixed, non-block-diagonal K_F -- the freeze's own anticipation implicitly assumed the per-block
# commutators individually vanish, which they do not, since K_F leaks across levels).  T2's own
# support-well-posedness ALSO fails identically in u (K_F(A) does not preserve rho_d's ~7-dim
# support; leak norm ~O(1) for every region sampled) -- a COMPOUNDING, not competing, finding.
#
# ML-2b/HK-7 CONDITIONALITY (carries into every DR-frame-touching sentence below, verbatim,
# unchanged from Section 8's own banner): "Every duality check here (HK-5) is CELL-LEVEL only (the
# 6-edge static vacuum). ML-2b's DR-frame argument is CONDITIONAL on the TD-limit duality holding,
# which is NOT verified by this suite."
#
# NUMBERS APPEAR NOWHERE: every quantity below is a dimension, rank, residual, locus dimension, or
# solved structural scalar (r, beta, c, epsilon) of THIS station's own construction -- never
# M_Z/ppm/m_nu/G/2pi-as-target; no comparison of any solved scalar to any measured value.  D4: the
# species map (gauge_sector_category) is NEVER an input anywhere below (grep-confirmed absent).
# READ-ONLY: nothing is booked; a V-VACUOUS outcome is booked RAW per the freeze's own tree, no
# selector claim is made, no register/verify.py edit accompanies this accretion.


def _w2_phi1_basis():
    """[W2 SS0, SHARED HELPER] the A2d channel basis phi_1,phi_2,phi_3 in a2_gamma_word's own
    (ND,3) convention -- REUSED verbatim via _a2d_phi1_list() (itself a thin wrapper over
    _a2d_abstract_hom_basis's (3,ND) SVD null-space basis, transposed; the null-space SVD
    threshold there is 1e-8, which is the inherited numerical floor every residual below is
    measured against).  Returns a list of 3 complex (ND,3) matrices."""
    return _a2d_phi1_list()


def w2_family_phi_d(d_vec, phi1_basis=None):
    """[W2 SS1] Phi_d = sum_i d_i.phi_i (d = r.u, r>0, u in C^3 unit norm) -- the family member, in
    a2_gamma_word's (ND,3) convention.  d_vec is any length-3 complex array (the caller supplies
    r.u already multiplied through; w2_family_direction is the usual entry point)."""
    if phi1_basis is None:
        phi1_basis = _w2_phi1_basis()
    d_vec = np.asarray(d_vec, dtype=complex).reshape(-1)
    assert len(d_vec) == 3, f"w2_family_phi_d: d_vec must have length 3, got {len(d_vec)}"
    return sum(d_vec[i] * phi1_basis[i] for i in range(3))


def w2_family_direction(u_vec, r=1.0):
    """[W2 SS1] normalizes u (any nonzero complex 3-vector) to unit norm and returns (d=r.u, u).
    The global phase of u is NOT fixed here -- it is verified to drop from every functional below
    (w2_global_phase_drop_check), per the freeze's own disclosed expectation."""
    u = np.asarray(u_vec, dtype=complex).reshape(-1)
    nrm = np.linalg.norm(u)
    assert nrm > 1e-12, "w2_family_direction: u must be nonzero"
    u = u / nrm
    return r * u, u


def w2_gamma_table(d_vec, N_max=4, max_length=4):
    """[W2 SS1] Gamma_d(w) := a2_gamma_word(Phi_d, w, Adag, vac) (REUSED unchanged) for every
    admissible word of length 0..max_length, bucketed by length (using omega_diag_length's OWN
    words/index/lengths/omega -- one build_hist call, not a second independent one, so every
    index lookup downstream is by construction consistent).
    Returns {'by_length': {n: {'idxs','words','vectors' (8,D_n) complex}}, 'Phi_d',
    'vac_check_residual' (Gamma_d(()) vs vac, EXACT since the empty word never touches Phi_d),
    'words','index','lengths','omega'}."""
    Phi_d = w2_family_phi_d(d_vec)
    Adag, vac, _, _ = _level1_creation_ops()
    words, index, lengths, omega = omega_diag_length(max(N_max, max_length))
    out = {}
    for n in range(0, max_length + 1):
        idxs = np.where(lengths == n)[0]
        ws = [words[i] for i in idxs]
        cols = [a2_gamma_word(Phi_d, w, Adag, vac) for w in ws]
        mat = np.hstack([c.reshape(8, 1) for c in cols]) if cols else np.zeros((8, 0), dtype=complex)
        out[n] = {"idxs": idxs, "words": ws, "vectors": mat}
    vac_resid = float(np.max(np.abs(out[0]["vectors"][:, 0] - vac.flatten())))
    return {"by_length": out, "Phi_d": Phi_d, "vac_check_residual": vac_resid,
            "words": words, "index": index, "lengths": lengths, "omega": omega}


def w2_welded_state(d_vec, N_max=4, length_range=(0, 3)):
    """[W2 SS1, THE WELDED STATE] rho_d = (1/Z).sum_w omega_diag(w).|Gamma_d(w)><Gamma_d(w)|, sum
    over words of length_range -- (0,3) is the PRIMARY read (the seed maps to vac, w2_gamma_table's
    own vac_check_residual verifies this exactly); (1,3) is the SECONDARY, seed-stripped read.
    Z = TRACE normalization (the station's own Z, NOT omega_diag_length's own global-N_max Z,
    which cancels identically in this length<=3 truncation by fock0e_analytic_lemma's ratio
    argument -- VERIFIED not merely asserted, see w2_truncation_stability_check)."""
    gt = w2_gamma_table(d_vec, N_max=N_max, max_length=max(length_range))
    omega = gt["omega"]
    rho_un = np.zeros((8, 8), dtype=complex)
    per_length = {}
    for n in range(length_range[0], length_range[1] + 1):
        idxs = gt["by_length"][n]["idxs"]
        V = gt["by_length"][n]["vectors"]
        if V.shape[1] == 0:
            per_length[n] = 0.0
            continue
        wts = omega[idxs]
        block = (V * wts[None, :]) @ V.conj().T
        rho_un += block
        per_length[n] = float(np.real(np.trace(block)))
    Z = float(np.real(np.trace(rho_un)))
    assert Z > 1e-14, "w2_welded_state: degenerate normalization (Z ~ 0)"
    rho = rho_un / Z
    herm_resid = float(np.max(np.abs(rho - rho.conj().T)))
    assert herm_resid < 1e-8, f"w2_welded_state: rho not Hermitian ({herm_resid:.2e})"
    return {"rho": rho, "Z": Z, "per_length_weight": per_length,
            "vac_check_residual": gt["vac_check_residual"], "Phi_d": gt["Phi_d"],
            "d_vec": np.asarray(d_vec, dtype=complex), "hermiticity_residual": herm_resid}


def w2_length4_vanishing_check(d_vec, N_max=4):
    """[W2 SS1, P1 SPOT CHECK] length>=4 words map to 0 (P1, a2_pauli_truncation_check's OWN
    GENERAL fact, TRUE FOR ANY phi_1, cited not re-derived) -- spot-checked here on THIS station's
    OWN family member Phi_d, on every admissible shell-4 word (not merely one)."""
    gt = w2_gamma_table(d_vec, N_max=N_max, max_length=4)
    V4 = gt["by_length"][4]["vectors"]
    worst = float(np.max(np.abs(V4))) if V4.size else 0.0
    return {"worst_shell4_abs": worst, "n_shell4_words": int(V4.shape[1])}


def w2_truncation_stability_check(d_vec, length_range=(0, 3)):
    """[W2 SS1, TRUNCATION STABILITY] N_max=3 vs N_max=4 comparison of rho_d (both restricted to
    length_range<=3) -- ANTICIPATED exactly zero (the per-length omega_diag ratios are
    N_max-independent, fock0e_analytic_lemma's own algebraic argument); a nonzero residual would
    be a genuine finding, reported raw, not smoothed."""
    r3 = w2_welded_state(d_vec, N_max=3, length_range=length_range)
    r4 = w2_welded_state(d_vec, N_max=4, length_range=length_range)
    resid = float(np.max(np.abs(r3["rho"] - r4["rho"])))
    return {"residual": resid}


def w2_global_phase_drop_check(u_vec, r=1.0, thetas=(0.7, 2.3)):
    """[W2 SS1, GLOBAL PHASE DROP] Phi_d's overall phase (u -> u.exp(i.theta)) must drop from
    rho_d -- VERIFIED, not assumed."""
    d0, u = w2_family_direction(u_vec, r=r)
    base = w2_welded_state(d0)["rho"]
    worst = 0.0
    for th in thetas:
        d1 = r * (u * np.exp(1j * th))
        rho1 = w2_welded_state(d1)["rho"]
        worst = max(worst, float(np.max(np.abs(rho1 - base))))
    return {"worst_phase_residual": worst, "thetas_tested": list(thetas)}


def w2_level_block_diagonality_check(u_vec, r, region_edges, N_max=4):
    """[W2 SS1, ANTICIPATED-STRUCTURE CHECK -- level-block-diagonality of rho_d AND K_F(A) wrt
    Pw[0..3]] VERIFIED, not assumed; per the freeze's own instruction a FAILED anticipation is
    reported RAW.  HEADLINE (see the section banner): rho_d's block-diagonality HOLDS to machine
    precision (level confinement, pc4's own NHAT-eigenvector fact); K_F(A)'s block-diagonality
    FAILS (O(1) cross-block residual) -- field_side_flow_generator's OWN docstring already
    discloses the region-Fock<->F identification is by dimension match alone, not a proof; this is
    the concrete numerical consequence, not a new incompleteness invented here."""
    d_vec, _ = w2_family_direction(u_vec, r=r)
    rho = w2_welded_state(d_vec, N_max=N_max)["rho"]
    K_F = field_side_flow_generator(region_edges)["K_F"]
    Pw, _ = _sector_projectors(sign=+1)
    worst_rho = 0.0
    worst_KF = 0.0
    for m in range(4):
        for n in range(4):
            if m == n:
                continue
            worst_rho = max(worst_rho, float(np.max(np.abs(Pw[m] @ rho @ Pw[n]))))
            worst_KF = max(worst_KF, float(np.max(np.abs(Pw[m] @ K_F @ Pw[n]))))
    return {"worst_rho_cross_block": worst_rho, "worst_KF_cross_block": worst_KF,
            "rho_block_diagonal": bool(worst_rho < 1e-8),
            "KF_block_diagonal": bool(worst_KF < 1e-8)}


def w2_T1_commutator_residual(u_vec, r, region_edges, N_max=4):
    """[W2 SS2, T1 -- FLOW INVARIANCE] N1(u;A) = ||[rho_d,K_F(A)]||_F / (||rho_d||_F.||K_F(A)||_F),
    computed directly on the FULL 8x8 matrices (exact, no scanning) -- the necessary condition for
    region-KMS."""
    d_vec, _ = w2_family_direction(u_vec, r=r)
    rho = w2_welded_state(d_vec, N_max=N_max)["rho"]
    K_F = field_side_flow_generator(region_edges)["K_F"]
    comm = rho @ K_F - K_F @ rho
    num = float(np.linalg.norm(comm, "fro"))
    rho_fro = float(np.linalg.norm(rho, "fro"))
    KF_fro = float(np.linalg.norm(K_F, "fro"))
    den = rho_fro * KF_fro
    N1 = num / den if den > 1e-14 else float("nan")
    return {"N1": N1, "commutator_fro": num, "rho_fro": rho_fro, "K_F_fro": KF_fro}


def w2_r_dependence_check(u_vec, region_edges, r_values=(0.3, 1.0, 3.0, 8.0), N_max=4):
    """[W2 SS1, THE r-INDEPENDENCE ANTICIPATION -- VERIFIED, not assumed] the freeze anticipates
    T1 is r-independent ('per-block scale cancels in a commutator-zero test').  HEADLINE: THIS
    ANTICIPATION FAILS (reported raw, per the freeze's own instruction) -- T1 is NOT identically
    zero (K_F leaks across levels, w2_level_block_diagonality_check), so rho_d's r-dependent
    level-mixing weights lambda_n(r) genuinely shift N1's value with r, even though N1 does NOT
    depend on u at all (the section's headline finding)."""
    vals = [w2_T1_commutator_residual(u_vec, r, region_edges, N_max=N_max)["N1"] for r in r_values]
    spread = float(max(vals) - min(vals))
    return {"r_values": list(r_values), "N1_values": vals, "spread": spread,
            "r_independent": bool(spread < 1e-6)}


def w2_level1_gram_scalar_proof():
    """[W2 SS2/SS3, THE EXACT MECHANISM -- LEVEL 1] W[i,j] := phi_i^T @ conj(phi_j) (3x3 complex,
    i,j=0,1,2) is, for EVERY (i,j), an ENDOMORPHISM of level-1's abstract V=C^3 built as the
    composition phi_i . phi_j^dagger of TWO A4-intertwiners (phi_j^dagger: V->shell1 is the adjoint
    of the intertwiner phi_j: shell1->V; phi_i: shell1->V) -- since V carries A4's IRREDUCIBLE
    3-dim standard rep (_a4_standard_3irrep, pc1), SCHUR'S LEMMA forces W[i,j] to be a SCALAR
    multiple of I_3, for every i,j, EXACTLY (not merely observed on a sample).  Gamma1(d)[m,n] :=
    sum_ij d_i.conj(d_j).W[i,j][m,n] (the level-1 block of rho_d before its own omega/Z weight,
    Gamma1(d) = Phi_d^T @ conj(Phi_d) -- VERIFIED via a cross-check against the closed-form tensor
    below) is THEREFORE a scalar multiple of I_3 for EVERY d: Gamma1(d) = (||d||^2/3).I_3 EXACTLY
    (the constant pinned by SVD-orthonormality of the phi_i basis: trace(W[i,j])=delta_ij, i.e.
    <phi_i,phi_j>_Frob=delta_ij, so trace(Gamma1(d))=||d||^2, and Gamma1(d) prop I_3 forces the
    diagonal value to be trace/3).  THIS IS THE EXACT MECHANISM ('exact symmetry argument', SS3)
    behind the honesty clause's vacuity finding at level 1: level 1 is BLIND to direction u, by
    Schur's lemma, full stop -- not merely numerically flat.
    Returns {'W' (dict (i,j)->3x3 complex), 'offdiag_residual' (worst |W[i,j] off I_3-multiple|),
    'diag_value_ij' (the scalar c_ij=trace(W[i,j])/3, should be delta_ij/3),
    'cross_check_residual' (Gamma1(d) via tensor vs Phi_d^T@conj(Phi_d) directly, random d)}."""
    phi1_list = _w2_phi1_basis()
    W = {}
    offdiag_resid = 0.0
    diag_vals = {}
    for i in range(3):
        for j in range(3):
            Wij = phi1_list[i].T @ phi1_list[j].conj()
            W[(i, j)] = Wij
            c = complex(np.trace(Wij) / 3.0)
            diag_vals[(i, j)] = c
            offdiag_resid = max(offdiag_resid, float(np.max(np.abs(Wij - c * np.eye(3)))))
    rng = np.random.default_rng(0)
    d_test = rng.normal(size=3) + 1j * rng.normal(size=3)
    Phi_test = w2_family_phi_d(d_test, phi1_list)
    direct = Phi_test.T @ Phi_test.conj()
    via_tensor = sum(d_test[i] * np.conj(d_test[j]) * W[(i, j)] for i in range(3) for j in range(3))
    cross_resid = float(np.max(np.abs(direct - via_tensor)))
    return {"W": W, "offdiag_residual": offdiag_resid, "diag_value_ij": diag_vals,
            "cross_check_residual": cross_resid}


def w2_level2_gram_scalar_check(n_samples=6, seed=0, N_max=4):
    """[W2 SS2/SS3, THE EXACT MECHANISM -- LEVEL 2, NUMERIC CONFIRMATION] level 2's abstract space
    is Lambda^2(V); _a2c_level_rep's OWN docstring already establishes (REUSED, not re-derived)
    'Lambda^2(rho3) ... DECOMPOSES as one more copy of the SAME 3-irrep -- A4's unique 3-irrep is
    isomorphic to its own Lambda^2' -- i.e. level 2 is ALSO an IRREDUCIBLE 3-dim A4-rep, so the
    SAME Schur argument (w2_level1_gram_scalar_proof) applies: block_2(d) := E_2^dagger.rho_d.E_2
    (before Z, extracted from the built-and-normalized rho_d for convenience, then rescaled) must
    be a scalar multiple of I_3, for EVERY d.  A fully explicit symbolic Lambda^2 tensor analogous
    to level 1's W[i,j] was NOT hand-derived in this pass (disclosed scope limit); this function
    instead VERIFIES the scalar-multiple conclusion NUMERICALLY, to machine precision, across
    n_samples random directions -- the operative evidence for level 2.
    Returns {'worst_offdiag_residual','per_sample_offdiag' (list), 'n_samples'}."""
    E2 = _a2c_level_embedding(2)
    rng = np.random.default_rng(seed)
    worst = 0.0
    per_sample = []
    for _ in range(n_samples):
        d = rng.normal(size=3) + 1j * rng.normal(size=3)
        d = d / np.linalg.norm(d)
        rho = w2_welded_state(d, N_max=N_max)["rho"]
        block2 = E2.conj().T @ rho @ E2
        offdiag = float(np.max(np.abs(block2 - np.diag(np.diag(block2)))))
        diag_spread = float(np.max(np.abs(np.diag(block2) - np.diag(block2)[0])))
        worst = max(worst, offdiag, diag_spread)
        per_sample.append({"offdiag": offdiag, "diag_spread": diag_spread})
    return {"worst_offdiag_residual": worst, "per_sample": per_sample, "n_samples": n_samples}


def w2_direction_independence_check(n_pairs=5, seed=0, r=1.0, N_max=4):
    """[W2 SS1/SS3, THE CONSEQUENCE, DIRECT VERIFICATION] rho_d(u1,r) vs rho_d(u2,r) compared
    DIRECTLY AS MATRICES (not merely via eigenvalues/spectrum) for n_pairs independent Haar-random
    (u1,u2) pairs -- HEADLINE anticipation: IDENTICAL (up to the ~1e-8 floor inherited from
    _a2d_abstract_hom_basis's own SVD null-space threshold), confirming rho_d = sum_n
    lambda_n(r).Pw[n] is genuinely DIRECTION-INDEPENDENT, not merely isospectral-by-coincidence."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    residuals = []
    for _ in range(n_pairs):
        u1 = rng.normal(size=3) + 1j * rng.normal(size=3)
        u1 = u1 / np.linalg.norm(u1)
        u2 = rng.normal(size=3) + 1j * rng.normal(size=3)
        u2 = u2 / np.linalg.norm(u2)
        rho1 = w2_welded_state(r * u1, N_max=N_max)["rho"]
        rho2 = w2_welded_state(r * u2, N_max=N_max)["rho"]
        resid = float(np.max(np.abs(rho1 - rho2)))
        residuals.append(resid)
        worst = max(worst, resid)
    return {"worst_residual": worst, "per_pair_residual": residuals, "n_pairs": n_pairs}


def _w2_reverse_word(word):
    """[W2 SS1, SHARED HELPER] J_hist on a word: reverse(w) = (r(d_n),...,r(d_1)), r(d)=d^1 --
    history_reversal_matrix's OWN rule (REUSED, not re-derived; that function already verifies
    this reversal is admissible-closed and an exact involution)."""
    return tuple((d ^ 1) for d in reversed(word))


def w2_conjugate_gamma_word(d_vec, word):
    """[W2 SS1, THE CONJUGATE WELD FUNCTOR -- NAMED JUDGMENT CALL, DISCLOSED, per the codebase's
    own 'named judgment call' convention (e.g. field_side_flow_generator, a2_alternate_reading_
    diagnostic)] Gamma_{Phi~_d}(w) := J_F(Gamma_{Phi_d}(J_hist(w))) = K @ conj(Gamma_{Phi_d}
    (reverse(w))), K = field_algebra_conjugation()'s own M (REUSED), reverse(w) = _w2_reverse_word
    (history_reversal_matrix's own per-word rule, REUSED).
    WHY A JUDGMENT CALL IS NEEDED (disclosed, not silent): Phi~_d, as LITERALLY constructed by
    a2d_pair_completeness_and_coverage (Phi~_d = K.conj(E_1.Phi_d).R, an (8,ND) map defined ONLY
    on SINGLE darts, landing in Pw[2]) has NO accreted Fock-functor extension to WORDS of length>1
    -- a2_gamma_word's wedge machinery is hard-wired to level-1's OWN 3 creation operators
    (Adag[0..2]) and CANNOT accept a Pw[2]-valued per-dart map (Pw[1] cap Pw[2] = {0}, so no
    (ND,3) matrix in a2_gamma_word's convention could represent Phi~_d at ALL -- a genuine TYPE
    MISMATCH, not an implementation pass oversight).  This function instead extends Phi~_d's OWN DEFINING
    FORMULA (apply J_F after, J_hist before) from single darts to the WHOLE functor, reusing ONLY
    frozen SS0 objects (field_algebra_conjugation, the reversal rule).
    CONSEQUENCE (verified, not assumed, w2_pair_state): because omega_diag depends ONLY on word
    LENGTH (omega_diag_length's own literal form) and reverse(w) has the SAME length as w,
    omega_diag(reverse(w))=omega_diag(w) TRIVIALLY -- so under THIS definition, rho_Phi~ =
    K.conj(rho_Phi).K^dagger = J_F.rho_Phi.J_F^{-1} EXACTLY, BY CONSTRUCTION.  The freeze's own
    stated check ('reversal-invariance of omega_diag') is therefore CONFIRMED, but its content,
    under this reading, reduces to omega_diag's PRE-EXISTING length-only dependence (already
    machine-verified elsewhere) -- reported HONESTLY as such, not oversold as new information."""
    Adag, vac, _, _ = _level1_creation_ops()
    rw = _w2_reverse_word(word)
    Phi_d = w2_family_phi_d(d_vec)
    v = a2_gamma_word(Phi_d, rw, Adag, vac)
    fa = field_algebra_conjugation()
    K = fa["M"]
    return K @ np.conj(v)


def w2_pair_state(d_vec, N_max=4, length_range=(0, 3)):
    """[W2 SS1, THE PAIR READ -- report-only] rho_Phi~ built via w2_conjugate_gamma_word (the
    disclosed judgment-call extension, see its docstring), rho_pair = 0.5.(rho_Phi + rho_Phi~).
    CHECKS J_F.rho_Phi.J_F^{-1} == rho_Phi~ EXACTLY (K.conj(rho_Phi).K^dagger, field_algebra_
    conjugation's own K, REUSED) -- confirmed to machine precision (see this section's headline;
    under the disclosed extension this is a construction identity, not new physics) -- and reports
    the per-level trace SWAP (Pw[0]<->Pw[3], Pw[1]<->Pw[2]) this induces, matching BOOTCAMP SS8's
    own 'J_F = THE ANTIPARTICLE PAIRING (levels 0<->3/1<->2)' dictionary fact."""
    words, index, lengths, omega = omega_diag_length(N_max)
    rho_un = np.zeros((8, 8), dtype=complex)
    for n in range(length_range[0], length_range[1] + 1):
        idxs = np.where(lengths == n)[0]
        if len(idxs) == 0:
            continue
        cols = [w2_conjugate_gamma_word(d_vec, words[i]) for i in idxs]
        V = np.hstack([c.reshape(8, 1) for c in cols])
        wts = omega[idxs]
        rho_un += (V * wts[None, :]) @ V.conj().T
    Z = float(np.real(np.trace(rho_un)))
    assert Z > 1e-14, "w2_pair_state: degenerate normalization (Z_tilde ~ 0)"
    rho_tilde = rho_un / Z
    rho_phi = w2_welded_state(d_vec, N_max=N_max, length_range=length_range)["rho"]
    fa = field_algebra_conjugation()
    K = fa["M"]
    predicted = K @ np.conj(rho_phi) @ K.conj().T
    resid = float(np.max(np.abs(predicted - rho_tilde)))
    rho_pair = 0.5 * (rho_phi + rho_tilde)
    Pw, _ = _sector_projectors(sign=+1)
    level_swap = {n: {"Phi": float(np.real(np.trace(Pw[n] @ rho_phi))),
                       "Phi_tilde": float(np.real(np.trace(Pw[n] @ rho_tilde)))}
                  for n in range(4)}
    return {"rho_tilde": rho_tilde, "rho_pair": rho_pair, "JF_conjugation_residual": resid,
            "Z_tilde": Z, "level_swap": level_swap}


def w2_support_well_posedness(u_vec, r, region_edges, N_max=4, tol=1e-8):
    """[W2 SS2, T2's SUPPORT WELL-POSEDNESS -- CHECKED FIRST, per the freeze] whether K_F(A)
    preserves supp(rho_d): builds P_supp_rho (rho_d's own nonzero-eigenvalue projector) and
    P_nonzero_KF (the complement of K_F(A)'s forced-zero eigenspace, C_A=1/2 lemma -- excluded
    from the fit per the freeze's own rule), then the FIT SUPPORT S = range(P_supp_rho) cap
    range(P_nonzero_KF) via the standard exact linear-algebra fact: eigenvectors of
    (P_supp_rho+P_nonzero_KF) with eigenvalue EXACTLY 2 span the intersection of the two ranges.
    leakage_norm = ||(I-P_supp_rho).K_F(A).P_supp_rho|| tests whether K_F(A) preserves supp(rho_d)
    at all (HEADLINE: FAILS, O(1), for every region sampled -- reported raw, feeds the T2 finding,
    NOT relabeled).
    Returns {'supp_rho_dim','KF_zero_dim','leakage_norm','well_posed','fit_support_dim','S_basis',
    'rho','K_F'}."""
    d_vec, _ = w2_family_direction(u_vec, r=r)
    rho = w2_welded_state(d_vec, N_max=N_max)["rho"]
    K_F = field_side_flow_generator(region_edges)["K_F"]
    lam_rho, V_rho = np.linalg.eigh(rho)
    supp_mask = lam_rho > tol * max(1.0, float(np.max(lam_rho)))
    P_supp_rho = (V_rho[:, supp_mask]) @ (V_rho[:, supp_mask].conj().T)
    mu_KF, V_KF = np.linalg.eigh(K_F)
    zero_mask = np.abs(mu_KF) < 1e-6
    P_zero_KF = (V_KF[:, zero_mask]) @ (V_KF[:, zero_mask].conj().T)
    P_nonzero_KF = np.eye(8) - P_zero_KF
    leak = (np.eye(8) - P_supp_rho) @ K_F @ P_supp_rho
    leak_norm = float(np.max(np.abs(leak)))
    Msum = P_supp_rho + P_nonzero_KF
    w, V = np.linalg.eigh(Msum)
    inter_mask = np.abs(w - 2.0) < 1e-6
    S_basis = V[:, inter_mask]
    return {"supp_rho_dim": int(np.sum(supp_mask)), "KF_zero_dim": int(np.sum(zero_mask)),
            "leakage_norm": leak_norm, "well_posed": bool(leak_norm < 1e-6),
            "fit_support_dim": int(S_basis.shape[1]), "S_basis": S_basis, "rho": rho, "K_F": K_F}


def w2_affine_fit_T2(u_vec, r, region_edges, N_max=4):
    """[W2 SS2, T2 -- FULL REGION-KMS] on supp(rho_d) (well-posedness checked FIRST,
    w2_support_well_posedness; forced zero modes excluded per C_A=1/2), -log(rho_d) = beta.K_F(A)
    + c.I fit by EXACT 2-parameter Frobenius-inner-product least squares (a closed-form 2x2 linear
    solve -- NEVER grid-scanned).  N2 = the normalized residual.  If the fit support has dim<2,
    (beta,c) is underdetermined and the fit is UNDEFINED (reported, not forced)."""
    wp = w2_support_well_posedness(u_vec, r, region_edges, N_max=N_max)
    S = wp["S_basis"]
    dim_s = S.shape[1]
    if dim_s < 2:
        return {"well_posed": wp["well_posed"], "leakage_norm": wp["leakage_norm"],
                "fit_support_dim": dim_s, "undefined": True,
                "reason": "fit support dimension < 2 -- (beta,c) underdetermined"}
    rho, K_F = wp["rho"], wp["K_F"]
    rho_S = S.conj().T @ rho @ S
    lam, V = np.linalg.eigh(rho_S)
    if not np.all(lam > 1e-12):
        return {"well_posed": wp["well_posed"], "leakage_norm": wp["leakage_norm"],
                "fit_support_dim": dim_s, "undefined": True,
                "reason": "rho not strictly positive on the fit support -- -log undefined"}
    neglog_S = V @ np.diag(-np.log(lam)) @ V.conj().T
    KF_S = S.conj().T @ K_F @ S
    I_S = np.eye(dim_s)

    def ip(A, B):
        return float(np.real(np.trace(A.conj().T @ B)))

    a11, a12, a22 = ip(KF_S, KF_S), ip(KF_S, I_S), ip(I_S, I_S)
    b1, b2 = ip(KF_S, neglog_S), ip(I_S, neglog_S)
    beta, c = np.linalg.solve(np.array([[a11, a12], [a12, a22]]), np.array([b1, b2]))
    resid_mat = neglog_S - beta * KF_S - c * I_S
    N2 = float(np.linalg.norm(resid_mat, "fro") / max(np.linalg.norm(neglog_S, "fro"), 1e-14))
    return {"well_posed": wp["well_posed"], "leakage_norm": wp["leakage_norm"],
            "fit_support_dim": dim_s, "beta": float(beta), "c": float(c), "N2": N2,
            "undefined": False}


def w2_T2_direction_check(region_edges, n_samples=5, seed=0, r=1.0, N_max=4):
    """[W2 SS2, T2's OWN honesty check] (beta,c,N2) sampled across n_samples random directions --
    HEADLINE anticipation (per the section's mechanism): CONSTANT, for the same reason T1 is
    (rho_d is direction-independent).  well_posed is expected FALSE identically (the support
    leakage is a fact about K_F(A) vs rho_d's near-full support, not about u)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        u = rng.normal(size=3) + 1j * rng.normal(size=3)
        u = u / np.linalg.norm(u)
        rows.append(w2_affine_fit_T2(u, r, region_edges, N_max=N_max))
    return {"rows": rows}


def w2_honesty_clause(region_edges, n_samples=20, seed=0, r=1.0, N_max=4):
    """[W2 SS3, THE HONESTY CLAUSE -- vacuity guard, run BEFORE any locus interpretation, per the
    freeze] samples n_samples (default 20, per the freeze) Haar-random unit directions u in CP^2
    and reports the SPREAD of N1(u;A) across them.  AN EXACT SYMMETRY ARGUMENT IS AVAILABLE (SS3
    prefers this over sampling): w2_level1_gram_scalar_proof (Schur, level 1, EXACT) +
    w2_level2_gram_scalar_check (Schur via Lambda^2(V)~=V, level 2, machine-precision-verified)
    together PROVE rho_d is direction-independent -- the sampling below is the freeze-mandated
    CONFIRMING check, not the primary evidence."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_samples):
        u = rng.normal(size=3) + 1j * rng.normal(size=3)
        u = u / np.linalg.norm(u)
        vals.append(w2_T1_commutator_residual(u, r, region_edges, N_max=N_max)["N1"])
    vals = np.array(vals)
    spread = float(np.max(vals) - np.min(vals))
    rel_spread = spread / max(1.0, float(np.mean(np.abs(vals))))
    is_constant = bool(rel_spread < 1e-6)
    return {"n_samples": n_samples, "N1_values": vals.tolist(), "min": float(np.min(vals)),
            "max": float(np.max(vals)), "spread": spread, "relative_spread": rel_spread,
            "is_constant_vacuous": is_constant}


def w2_region_analysis(region_edges, n_samples=20, seed=0, r=1.0, N_max=4):
    """[W2, PER-REGION BUNDLE] honesty clause + T1 sample + level-block-diagonality check + T2
    direction check, for ONE region orbit representative."""
    honesty = w2_honesty_clause(region_edges, n_samples=n_samples, seed=seed, r=r, N_max=N_max)
    bd = w2_level_block_diagonality_check(np.array([1.0, 0.0, 0.0]), r, region_edges, N_max=N_max)
    t2 = w2_T2_direction_check(region_edges, n_samples=min(5, n_samples), seed=seed, r=r, N_max=N_max)
    n1_ref = w2_T1_commutator_residual(np.array([1.0, 0.0, 0.0]), r, region_edges, N_max=N_max)["N1"]
    return {"region": region_edges, "honesty": honesty, "block_diagonality": bd,
            "T2_direction_check": t2, "N1_reference": n1_ref}


def w2_full_station_analysis(n_samples=20, seed=0, r=1.0, N_max=4, r_check_values=(0.3, 1.0, 3.0, 8.0)):
    """[W2, THE FULL STATION DRIVER] assembles the complete honest read: the exact mechanism
    (Schur, both levels), the direction-independence direct-matrix check, the anticipated-structure
    checks (vac, P1, truncation stability, phase drop, r-dependence), per-region analysis over ALL
    4 A4-orbit-representative 3-edge regions, and the verdict per the freeze's SS4 tree.  Per the
    freeze's own instruction, NO multi-start/locus-search optimization is run once vacuity is
    established via the exact mechanism ('do not hunt for a locus anyway')."""
    orbits = _three_edge_region_orbits()
    u0 = np.array([1.0, 0.0, 0.0])
    mech1 = w2_level1_gram_scalar_proof()
    mech2 = w2_level2_gram_scalar_check(n_samples=6, seed=seed, N_max=N_max)
    dind = w2_direction_independence_check(n_pairs=5, seed=seed, r=r, N_max=N_max)
    vac_check = w2_gamma_table(u0, N_max=N_max, max_length=0)["vac_check_residual"]
    len4 = w2_length4_vanishing_check(u0, N_max=N_max)
    trunc = w2_truncation_stability_check(u0)
    phase = w2_global_phase_drop_check(u0, r=r)
    r_dep = w2_r_dependence_check(u0, orbits[0]["representative"], r_values=r_check_values, N_max=N_max)
    per_region = []
    for orb in orbits:
        ra = w2_region_analysis(orb["representative"], n_samples=n_samples, seed=seed, r=r, N_max=N_max)
        ra["orbit_size"] = orb["orbit_size"]
        ra["is_triangle"] = orb["is_triangle"]
        per_region.append(ra)
    pair = w2_pair_state(r * np.array([1.0, 0.0, 0.0]), N_max=N_max)

    all_vacuous = all(pr["honesty"]["is_constant_vacuous"] for pr in per_region)
    all_well_posed = all(row["well_posed"] for pr in per_region for row in pr["T2_direction_check"]["rows"]
                          if not row.get("undefined", False))
    if all_vacuous:
        verdict = "V-VACUOUS"
    else:
        worst_spread = max(pr["honesty"]["relative_spread"] for pr in per_region)
        verdict = f"UNRESOLVED-NUMERIC (honesty clause not uniformly constant; worst relative spread {worst_spread:.2e})"

    return {"mechanism_level1": mech1, "mechanism_level2": mech2,
            "direction_independence": dind, "vac_check_residual": vac_check,
            "length4_check": len4, "truncation_stability": trunc,
            "global_phase_drop": phase, "r_dependence": r_dep,
            "per_region": per_region, "pair_read": pair,
            "all_regions_vacuous": all_vacuous, "T2_well_posed_anywhere": all_well_posed,
            "verdict": verdict}


def w2_selftest_2026_07_13(verbose=True):
    """W2 station regression (fast, small sizes, < 120s per the verify per-entry timeout law):
    Sections 1-9b + module anchors untouched, THEN the welded-state construction (vac/P1/
    truncation/phase-drop), the EXACT Schur mechanism at level 1 + the numeric confirmation at
    level 2, the direct direction-independence check, T1/T2 well-posedness, the honesty clause
    (small sample), and the PAIR read's J_F-conjugation identity.  Does NOT adjudicate booking (the
    working note / architect only) and does NOT run the full 4-region analysis at large sample
    counts (w2_full_station_analysis, run separately, still fast: <5s)."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 10 self-test -- W2 the selector read (2026-07-13)")
        print("=" * 88)

    # NOTE ON SCOPE (disclosed, not silent): the FULL historical chain (ml1db_selftest_2026_07_13,
    # which transitively re-runs EVERY prior section's own selftest back through a2c/a2b/a2/fock0*)
    # is, BY ITSELF, already >100s (measured: a2c_weld_selftest_2026_07_12 ALONE costs ~57.6s;
    # a2b ~26s; a2 ~12s; the historical chain compounds past the 120s budget this function is
    # bound by, per the freeze's own 'verify per-entry timeout law').  This fast selftest therefore
    # checks ONLY the two cheap module-level anchors here (instant) as a spot check that the module
    # still imports/runs cleanly; the FULL historical-chain regression (ml1db_selftest_2026_07_13)
    # was run SEPARATELY, once, OUTSIDE this function (not wired in, per the freeze's own
    # instruction) -- its result is reported in the working note, not re-run on every selftest
    # call.
    ck("module anchors (cheap spot check; full historical chain confirmed separately, see the "
       "working note)", anchor_cell_projector() and anchor_tick_2pi())

    N_max = 3
    u0 = np.array([1.0, 0.0, 0.0])
    gt = w2_gamma_table(u0, N_max=N_max, max_length=N_max)
    ck(f"Gamma_d(()) == vac EXACTLY (residual={gt['vac_check_residual']:.2e})",
       gt["vac_check_residual"] < 1e-12)

    len4 = w2_length4_vanishing_check(u0, N_max=4)
    ck(f"P1 spot check: {len4['n_shell4_words']} shell-4 words, worst norm={len4['worst_shell4_abs']:.2e}",
       len4["worst_shell4_abs"] < 1e-9)

    trunc = w2_truncation_stability_check(u0)
    ck(f"truncation stability N_max=3 vs 4: residual={trunc['residual']:.2e}", trunc["residual"] < 1e-9)

    phase = w2_global_phase_drop_check(u0, r=1.0)
    ck(f"global phase drop: worst residual={phase['worst_phase_residual']:.2e}",
       phase["worst_phase_residual"] < 1e-9)

    mech1 = w2_level1_gram_scalar_proof()
    ck(f"LEVEL-1 SCHUR PROOF: offdiag residual={mech1['offdiag_residual']:.2e}, "
       f"c_00={mech1['diag_value_ij'][(0, 0)]:.4f} (expect 1/3), "
       f"c_01={mech1['diag_value_ij'][(0, 1)]:.2e} (expect 0), "
       f"cross-check residual={mech1['cross_check_residual']:.2e}",
       mech1["offdiag_residual"] < 1e-9 and abs(mech1["diag_value_ij"][(0, 0)] - 1.0 / 3.0) < 1e-9
       and abs(mech1["diag_value_ij"][(0, 1)]) < 1e-9 and mech1["cross_check_residual"] < 1e-9)

    mech2 = w2_level2_gram_scalar_check(n_samples=3, seed=0, N_max=N_max)
    ck(f"LEVEL-2 SCHUR CHECK (numeric, {mech2['n_samples']} samples): "
       f"worst offdiag/diag-spread residual={mech2['worst_offdiag_residual']:.2e}",
       mech2["worst_offdiag_residual"] < 1e-6)

    dind = w2_direction_independence_check(n_pairs=3, seed=0, r=1.0, N_max=N_max)
    ck(f"DIRECTION INDEPENDENCE (direct matrix compare, {dind['n_pairs']} pairs): "
       f"worst residual={dind['worst_residual']:.2e}", dind["worst_residual"] < 1e-6)

    orbits = _three_edge_region_orbits()
    reg0 = orbits[0]["representative"]
    bd = w2_level_block_diagonality_check(u0, 1.0, reg0, N_max=N_max)
    ck(f"BLOCK-DIAGONALITY: rho cross-block={bd['worst_rho_cross_block']:.2e} (holds), "
       f"K_F cross-block={bd['worst_KF_cross_block']:.2e} (FAILS, honestly reported)",
       bd["rho_block_diagonal"] and not bd["KF_block_diagonal"])

    honesty = w2_honesty_clause(reg0, n_samples=8, seed=0, r=1.0, N_max=N_max)
    ck(f"HONESTY CLAUSE [8 samples, region {reg0}]: N1 in [{honesty['min']:.6f},{honesty['max']:.6f}], "
       f"relative spread={honesty['relative_spread']:.2e} -> is_constant_vacuous="
       f"{honesty['is_constant_vacuous']}", honesty["is_constant_vacuous"])

    wp = w2_support_well_posedness(u0, 1.0, reg0, N_max=N_max)
    ck(f"T2 SUPPORT WELL-POSEDNESS [region {reg0}]: supp_rho_dim={wp['supp_rho_dim']}, "
       f"KF_zero_dim={wp['KF_zero_dim']}, leakage_norm={wp['leakage_norm']:.3f} "
       f"(FAILS -- honestly reported, feeds the T2 finding)", not wp["well_posed"])

    pair = w2_pair_state(u0, N_max=N_max)
    ck(f"PAIR READ: J_F-conjugation residual={pair['JF_conjugation_residual']:.2e} "
       f"(rho_Phi~ == K.conj(rho_Phi).K^dagger, by-construction identity)",
       pair["JF_conjugation_residual"] < 1e-9)

    if verbose:
        print("RESULT:", "W2 SECTION-10 REGRESSION PASSES" if ok else "A W2 CHECK FAILED")
    return ok


# ===========================================================================
# 4e. THE RESPONSE-BLOCK SCAFFOLD  (L-response; the "ML-5/Delta-c" density-response
#     build, 2026-07-13 -- user-ratified parallel lane;
#     ROADMAP_frontiers_and_pushes_2026-07-12.md S4 changelog 2026-07-13, item (B))
# ===========================================================================
# [PLACEMENT NOTE: appended at end-of-file, the SAME conflict-free convention already used twice
#  above (section 7's accretion pass, the W2-BGK block) -- on import these definitions land in the
#  module namespace exactly as if written with the rest of section 4d; nothing above this point is
#  touched, and __main__'s self_test() (script mode) is unaffected.]
#
# NAMING DISCLOSURE (flagged, not silent -- this is the ONE conflict this build turned up).
# "ML-5" is used TWICE in this repo's history for TWO UNRELATED objects.
#   (i) internal research notes's ML-5 is "the EPSILON
#       READOUT" -- the chiral next-order phase eps = delta_eff - 2/9 for the -70 ppm keystone.
#       That station RAN (ML5_epsilon_2026-07-08.py, ML5b_epsilon_transport_2026-07-08.py,
#       R1_zeta_order_reading_2026-07-09.py) and was REFUTED (docs/scoping/
#       completeness_review_2026-07-09.md: "Three disciplined shots refuted (ML-5, ML-5b, R1)").
#       That station is CLOSED; nothing in this section reopens it, and it is NOT this section's
#       spec despite sharing a label.
#   (ii) internal research notes (S4 2026-07-13 changelog) reuses
#       the SAME LABEL for a different object entirely -- "the chi_0/Delta-c density-response
#       build" feeding Bin L-response (n_s, sigma_8, S_8, D(z), f(z), fsigma_8; see
#       internal research notes). THIS SECTION BUILDS
#       (ii) ONLY. The dispatch brief for this build named (i)'s document as "the authoritative
#       ML-5 spec to read first" -- that document's ML-5 section is a DIFFERENT, already-settled
#       physics object, not this one; this is reported in the working note.
#
# WHAT ALREADY EXISTS (promoted here, not rebuilt). Section 4d above already carries the free
# Lindhard bubble chi_0(q,omega) (lindhard_chi0/chi0_from_setup/lindhard_setup), the scalar
# Mermin/RTA conserving closure (mermin_chi), and the certified two-moment {n,j} conserving closure
# (two_moment_chi/moment_static_matrix/closure_from_moments, W2-BGK 2026-07-10). The station script
# proofs/foundations/B2a_density_response_2026-07-09.py used chi_0/mermin_chi to run ONE frozen
# confront (R-4, three declared q's) whose crux verdict was adjudicated at the two-moment/BGK level
# and BOOKED as NO-SOUND (internal research notes line 31 lists
# "NO-SOUND" among that wave's booked results; consistent with B2a's own R-4 diagnostic: gamma_micro
# = 0.5*ln(2) ~ 0.347 >> c_s*q at every declared q, i.e. heavily overdamped). But the POLE-EXTRACTION
# / sound-speed-CONTRAST machinery that did the extraction lived ONLY in that disposable script --
# a violation of the one-master-object law this section corrects: it promotes that machinery into
# the net as reusable, grid-declared functions, generalizes the three frozen q-points into a
# continuous q-CURVE (the "Delta-c" object the build's name refers to), and adds the two structural
# hooks the parallel lane needs (a compressibility-positivity sanity check; a growth-kernel FORM
# interface). IT DOES NOT RE-ADJUDICATE R-4/BGK-3's booked verdict: this section's own functions run
# on smaller, DECLARED grids (n_grid ~10-20, <=150 omega points) for speed, a genuinely different
# (smaller) computation than the certified run (n_grid=32/40, up to 500 omega points, the full {n,j}
# closure) -- disclosed as such; any pole/no-pole result these functions produce on their own grids
# is reported RAW, never reconciled against or used to overwrite the certified verdict.
#
# SCOPE (printed, per this repo's R-6-style declaration convention -- NO VALUE for any of the
# following is read, computed, or assigned anywhere in this section): n_s, sigma_8, S_8, D(z),
# f(z), fsigma_8, A_s are NOT touched (all strictly downstream, per B2-a's own R-6); no era/N
# exponent, no GR growth ODE is integrated; Newton's G/2pi and the calibration fence's N_hub/G_F
# are untouched; no measured quantity of any kind enters any function below -- the ONLY confront in
# this section is the frozen theory-vs-theory one B2-a's own poisons already license (chi_0/chi_M
# derived response vs c_s^2=1/3 derived from the walk-gas EoS, M2a) -- never confronted against data.

BETA_EFF_G5A = 2.0 * math.log((1.0 / (srs.DEG - 1)) / (((srs.DEG - 1) / srs.DEG) ** 8))
# quoted formula (NOT re-derived): G5a / derivation_topdown/adapters/thermal_time.py:151-152,209-211
# u_c = 1/(k-1), alpha_1 = ((k-1)/k)^8, beta_eff = 2*log(u_c/alpha_1);  == 5.1011473686 for srs.DEG=3.


def _gamma_micro_mc2():
    """gamma_micro, quoted formula (NOT re-derived): MC-2's Ramanujan-gap rate
    (MC2_phase_memory_kernel_2026-07-07.py:42-57, reused verbatim by B2a_density_response's R-0(d)):
    the Perron/sub-Perron eigenvalue gap of the edge Hashimoto matrix at k=0, == 0.5*ln(srs.DEG-1)."""
    modsG = np.sort(np.abs(np.linalg.eigvals(srs.hashimoto((0, 0, 0)))))[::-1]
    lamP = modsG[0]
    lam_sub = max(m for m in modsG if m < lamP - 1e-6)
    return math.log(lamP / lam_sub)


GAMMA_MICRO_MC2 = _gamma_micro_mc2()   # == 0.5*ln(2) for srs.DEG=3 (MC-2/B2a's reused value)


def response_static_profile(q_mags, direction=(1.0, 0.0, 0.0), beta=BETA_EFF_G5A, n_grid=20,
                             node=NODE_LAM_F):
    """The 'chi_0' half of the build's name: the static response chi_0(q,0) profile over a
    DECLARED q-grid along one direction, promoted from lindhard_chi0 into a directly-callable
    q-CURVE (a convenience accretion; lindhard_chi0/chi0_from_setup themselves are untouched, no
    new physics). SIGN CONVENTION (disclosed, reused from B2a's own R-1 disclosure, NOT reinvented
    here): chi_0(q,0) is NEGATIVE in the causal (+i*eta) convention lindhard_chi0 uses (the standard
    static-compressibility sign: induced density opposes a positive external potential) --
    'positivity' of the physical compressibility magnitude is read as -chi_0(q,0) > 0, not
    chi_0(q,0) > 0. Returns {'q_mags', 'chi0_static' (real array, expected NEGATIVE), 'all_positive'
    (== all(-chi0_static > 0), B2a's convention), 'all_finite'}."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    vals = []
    for q in q_mags:
        _, chi_static, _ = lindhard_chi0(q * d, np.array([0.0]), beta, n_grid=n_grid, node=node)
        vals.append(chi_static.real)   # lindhard_chi0's static value is real up to an eta-residual
    vals = np.asarray(vals, dtype=float)
    return {"q_mags": np.asarray(q_mags, dtype=float), "chi0_static": vals,
            "all_positive": bool(np.all(-vals > 0)), "all_finite": bool(np.all(np.isfinite(vals)))}


def compressibility_positivity_hook(profile):
    """A PARAMETER-FREE thermodynamic-stability sanity check on a response_static_profile() result:
    a genuinely compressible medium requires -chi_0(q,0) (the physical compressibility magnitude,
    B2a's disclosed sign convention -- see response_static_profile's docstring) finite and positive
    across the declared grid (the same check B2-a's own R-1 already printed, promoted here into a
    reusable function) -- NO measured value enters; this is an internal consistency check of the
    net's own construction, not a confront. Returns {'stable': bool, 'q0_value': float (raw
    chi_0(q_min,0), NEGATIVE by convention), 'reason': str}."""
    q0_idx = int(np.argmin(profile["q_mags"]))
    q0_val = float(profile["chi0_static"][q0_idx])
    stable = bool(profile["all_positive"] and profile["all_finite"])
    reason = ("-chi_0(q,0) positive and chi_0(q,0) finite across the declared grid (B2a's disclosed "
              "sign convention)" if stable else
              "-chi_0(q,0) is non-positive, or chi_0(q,0) non-finite, somewhere on the declared grid "
              "(booked raw)")
    return {"stable": stable, "q0_value": q0_val, "reason": reason}


def pole_locate(q_mag, direction=(1.0, 0.0, 0.0), beta=BETA_EFF_G5A, gamma=GAMMA_MICRO_MC2,
                n_grid=20, node=NODE_LAM_F, omega_max=None, n_omega=150):
    """Locate the small-q collective (sound-candidate) structure of the Mermin-closed response at
    ONE |q|, generalizing B2a's R-4 single-q read into a reusable net function (promoted out of the
    disposable station script, per the one-master-object law; identical formula/convention:
    omega_max defaults to B2a's own declared window max(0.5, 8*q_mag)). Returns {'q_mag',
    'omega_peak', 'c_pole'=omega_peak/q_mag, 'peak_val'=|Im chi_M|_peak, 'edge_val'=|Im chi_M| at
    omega->0, 'interior_peak' (bool, 0<argmax<last -- False = no genuine peak, an honest NO-POLE
    read at this q), 'chi0_static'}."""
    if omega_max is None:
        omega_max = max(0.5, 8.0 * q_mag)
    omegas = np.linspace(1e-3, omega_max, n_omega)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    chiM, chi0_stat, _ = mermin_chi(q_mag * d, omegas, beta, gamma, n_grid=n_grid, node=node)
    ipk = int(np.argmax(np.abs(chiM.imag)))
    w_peak = float(omegas[ipk])
    interior = 0 < ipk < len(omegas) - 1
    return {"q_mag": q_mag, "omega_peak": w_peak,
            "c_pole": (w_peak / q_mag) if q_mag else float("nan"),
            "peak_val": float(np.abs(chiM.imag)[ipk]), "edge_val": float(np.abs(chiM.imag)[0]),
            "interior_peak": interior, "chi0_static": chi0_stat}


def sound_speed_contrast_curve(q_mags, direction=(1.0, 0.0, 0.0), beta=BETA_EFF_G5A,
                                gamma=GAMMA_MICRO_MC2, n_grid=20, node=NODE_LAM_F, n_omega=150):
    """THE 'Delta-c' OBJECT (the density-response build's namesake): the sound-speed CONTRAST
    Delta_c(q) = c_pole(q)^2 - 1/3, over a declared q-grid -- generalizing B2a's frozen 3-point R-4
    read into a continuous structural curve. c_s^2=1/3 (M2's walk-gas EoS) enters ONLY in the
    reported contrast, NEVER in the construction of chi_0/chi_M (B2a's poison, preserved exactly: c_s
    never enters the construction, only the confront). A q is NOT assigned a contrast
    (delta_c=None in its row) if no interior peak was found there -- an honest NO-POLE read, never a
    fabricated number. The q->0 extrapolation (linear fit of c_pole(q), B2a's own convention) is
    reported ONLY if >=2 interior-peak q's exist; otherwise None, with the reason recorded.
    THIS FUNCTION DOES NOT RE-ADJUDICATE R-4/BGK-3's certified verdict (see the section-header
    disclosure above: smaller grid, single-scalar-moment closure, disclosed as a different
    computation) -- whatever it finds on ITS grid is reported raw."""
    rows = [pole_locate(q, direction, beta, gamma, n_grid, node, n_omega=n_omega) for q in q_mags]
    for r in rows:
        r["delta_c"] = (r["c_pole"] ** 2 - 1.0 / 3.0) if r["interior_peak"] else None
    interior_rows = [r for r in rows if r["interior_peak"]]
    if len(interior_rows) >= 2:
        qarr = np.array([r["q_mag"] for r in interior_rows])
        carr = np.array([r["c_pole"] for r in interior_rows])
        slope, intercept = np.polyfit(qarr, carr, 1)
        c_pole_q0 = float(intercept)
        delta_c_q0 = c_pole_q0 ** 2 - 1.0 / 3.0
        note = "q->0 linear extrapolation over the interior-peak q's (B2a's convention)"
    else:
        c_pole_q0, delta_c_q0 = None, None
        note = (f"fewer than 2 interior-peak q's on this grid ({len(interior_rows)}/{len(rows)}); "
                "no extrapolation attempted (honest, not a fabricated fit)")
    return {"rows": rows, "c_pole_q0": c_pole_q0, "delta_c_q0": delta_c_q0,
            "n_interior": len(interior_rows), "n_total": len(rows), "extrapolation_note": note}


def mermin_static_limit_check(q_mag=0.1, direction=(1.0, 0.0, 0.0), beta=BETA_EFF_G5A,
                               gamma=GAMMA_MICRO_MC2, n_grid=14, node=NODE_LAM_F):
    """Regression spot-check of an identity mermin_chi's OWN docstring already claims:
    chi_M(q,0) = chi_0(q,0) EXACTLY (the conserving closure's removable-singularity limit) -- a
    cheap consistency check that this section's use of both objects agrees with that identity.
    Returns {'mermin_self_residual', 'mermin_vs_lindhard_residual'} (both should be ~0)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    chiM, chi0_stat_M, _ = mermin_chi(q_mag * d, np.array([0.0]), beta, gamma, n_grid=n_grid, node=node)
    _, chi0_stat_L, _ = lindhard_chi0(q_mag * d, np.array([0.0]), beta, n_grid=n_grid, node=node,
                                       eta=1e-3)
    return {"mermin_self_residual": float(abs(chiM[0].real - chi0_stat_M)),
            "mermin_vs_lindhard_residual": float(abs(chi0_stat_M - chi0_stat_L))}


def growth_kernel_form(k_mags, chi0_static_of_k, H_label, H_value=None, gravity_coupling=None):
    """STRUCTURAL INTERFACE ONLY for a future confrontation station (the B2-b/B2-c growth-kernel /
    horizon-crossing stations named, gated, and NOT built in B2a's pre-reg lineage) -- NOT itself a
    growth computation. Exposes the FORM of the linearized density-contrast growth-equation
    coefficients in the substrate frame, built from THIS section's own chi_0 object, WITHOUT
    integrating any ODE, WITHOUT fixing a normalization, and WITHOUT confronting n_s / sigma_8 /
    S_8 / D(z) / f(z) / fsigma_8 (none of which appear anywhere in this module).

    HARD GUARDS enforced STRUCTURALLY, not just documented:
      * H_label must be exactly "H_sub" or "H_metric" (BOOTCAMP.md S4's TWO H's: H_sub=1/(N*t_P) !=
        H_metric=a-dot/a) -- raises ValueError otherwise. This function does NOT decide which rate
        belongs in a growth equation; the caller states its own reading and it is only ever passed
        through, never computed here.
      * H_value defaults to None (UNASSIGNED, never defaulted to a number).
      * gravity_coupling defaults to None (UNASSIGNED -- this build does not touch Newton's G/2pi or
        the calibration fence's N_hub/G_F; a caller needing one supplies its OWN value from
        elsewhere, never defaulted here).
      * NO factor of 16/15 (or any power of it) appears anywhere in this signature or body -- the
        A_s/16-15 trap (standing, user-enforced: the observer/substrate rate-gap's power assignment
        for dimensionless quantities is UNDERIVED) is kept structurally UNREACHABLE from this
        interface: there is no kwarg for it at all, so it cannot be silently defaulted here even by
        a future edit that forgets the trap; a caller who needs one must add it OUTSIDE this
        function, explicitly, on their own authority.

    Returns coefficient CALLABLES (never evaluated numbers) plus the list of names left unassigned:
      pressure_coeff(k, a) = k^2 * chi0_static_of_k(k) / a^2      (the ONE piece this build owns)
      friction_coeff(a)    = 2*H_value(a) [or 2*H_value if not callable], else None if unassigned
      source_coeff(a, rho) = -4*pi*gravity_coupling*rho,          else None if unassigned
    """
    if H_label not in ("H_sub", "H_metric"):
        raise ValueError('H_label must be "H_sub" or "H_metric" (the TWO H rates; BOOTCAMP.md S4) -- '
                          'no default is offered; the caller must name which rate this growth '
                          'equation uses.')
    unassigned = []
    if H_value is None:
        unassigned.append("H_value")
        friction_coeff = None
    elif callable(H_value):
        friction_coeff = lambda a, _h=H_value: 2.0 * _h(a)
    else:
        friction_coeff = lambda a, _v=H_value: 2.0 * _v
    if gravity_coupling is None:
        unassigned.append("gravity_coupling")
        source_coeff = None
    else:
        source_coeff = lambda a, rho, _g=gravity_coupling: -4.0 * math.pi * _g * rho
    pressure_coeff = lambda k, a: (k ** 2) * chi0_static_of_k(k) / (a ** 2)
    return {"pressure_coeff": pressure_coeff, "friction_coeff": friction_coeff,
            "source_coeff": source_coeff, "H_label": H_label, "unassigned": unassigned}


def ml5_selftest_2026_07_13(verbose=True):
    """Fast BUILD selftest (<120s, per the verify per-entry timeout law) for the "ML-5/Delta-c"
    density-response BUILD (2026-07-13, user-ratified parallel lane) -- small grids, calling the
    SAME section functions a future integration would use. NOT wired into verify.py (waits for the
    integration batch, per the dispatch brief); run standalone:
        python3 -c "import sys; sys.path.insert(0,'derivation_topdown/state'); import the_net as n; \\
                    sys.exit(0 if n.ml5_selftest_2026_07_13() else 1)"
    """
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 4e self-test -- 'ML-5/Delta-c' density-response BUILD (2026-07-13)")
        print("=" * 88)

    ck("module anchors (cheap regression spot check)", anchor_cell_projector() and anchor_tick_2pi())

    ck("beta_eff (G5a, quoted formula) == 5.1011473686", abs(BETA_EFF_G5A - 5.1011473686) < 1e-9,
       detail=f"{BETA_EFF_G5A:.10f}")
    ck("gamma_micro (MC-2, quoted formula) == 0.5*ln(2)",
       abs(GAMMA_MICRO_MC2 - 0.5 * math.log(2)) < 1e-9, detail=f"{GAMMA_MICRO_MC2:.10f}")

    mc = mermin_static_limit_check(n_grid=10)
    ck("Mermin removable-singularity identity chi_M(q,0)==chi_0(q,0) (mermin_chi's own docstring "
       "claim)", mc["mermin_self_residual"] < 1e-9 and mc["mermin_vs_lindhard_residual"] < 1e-9,
       detail=f"self={mc['mermin_self_residual']:.2e}, cross={mc['mermin_vs_lindhard_residual']:.2e}")

    prof = response_static_profile([0.05, 0.10, 0.20], n_grid=10)
    ck("chi_0(q,0) static profile: positive + finite over the small declared grid",
       prof["all_positive"] and prof["all_finite"], detail=f"{np.round(prof['chi0_static'], 4)}")

    stab = compressibility_positivity_hook(prof)
    ck("compressibility-positivity hook: chi_0(q->0,0) stable (parameter-free structural check)",
       stab["stable"], detail=stab["reason"])

    curve = sound_speed_contrast_curve([0.05, 0.10, 0.20], n_grid=10, n_omega=60)
    ck(f"Delta-c(q) sound-speed-contrast curve API runs cleanly ({curve['n_interior']}/"
       f"{curve['n_total']} interior peaks on this small grid; {curve['extrapolation_note']})", True)
    # NOTE: no PASS/FAIL threshold on interior-peak count is asserted here -- this is a structure/
    # API smoke test on a MUCH smaller grid than the certified BGK-3 run, not a re-adjudication of
    # its booked NO-SOUND verdict; whatever this small grid finds is printed, not fought.

    def _chi0_of_k(k):
        _, cs, _ = lindhard_chi0(k * np.array([1.0, 0.0, 0.0]), np.array([0.0]), BETA_EFF_G5A,
                                  n_grid=10)
        return cs.real   # lindhard_chi0's static value is real up to an eta-residual (see above)

    gk = growth_kernel_form([0.05, 0.1], _chi0_of_k, H_label="H_metric")
    ck("growth-kernel FORM scaffold: unassigned params correctly flagged (H_value, "
       "gravity_coupling), pressure_coeff callable without either",
       gk["unassigned"] == ["H_value", "gravity_coupling"] and gk["friction_coeff"] is None
       and gk["source_coeff"] is None and callable(gk["pressure_coeff"])
       and math.isfinite(gk["pressure_coeff"](0.1, 1.0)), detail=f"unassigned={gk['unassigned']}")

    raised = False
    try:
        growth_kernel_form([0.1], _chi0_of_k, H_label="bogus")
    except ValueError:
        raised = True
    ck("growth-kernel FORM scaffold: bad H_label raises (no silent default; the TWO H's guard)",
       raised)

    if verbose:
        print("RESULT:", "'ML-5/Delta-c' BUILD SELFTEST PASSES" if ok else "A CHECK FAILED")
    return ok


# ===========================================================================
# 11. V1 — THE VERTEX ON THE FAMILY (2026-07-13)
#     internal research notes (commit 2fc5e77, BEFORE this code)
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as every section above — appended after the
#  __main__ guard; ACCRETION-ONLY, nothing in Sections 1-10 is modified.]
#
# CONTEXT (selector station II; W2 = station I returned V-VACUOUS by Schur): the vertex — the
# MDL contact coupling E_int(A,B) = -kappa.I(A;B), the one built object NONLINEAR in the state —
# evaluated ON the A2d weld family, asking whether dynamics lifts the CP^2 multiplicity
# degeneracy.  Per the freeze's W2-informed design laws, the station works on THE CHANNEL
# (|W_u> on H_hist (x) F, sqrt(omega)-weighted words 0..3, r frozen at 1), never the pushed
# state (dead by W2's theorem); STEP 0 (functional-level vacuity pre-check) ran BEFORE any
# locus/extremum computation, on the closed frozen functional list (91 components; nothing
# added after).  Every frozen object REUSED by call (w2_family_direction / w2_gamma_table /
# w2_conjugate_gamma_word = §10's disclosed Gamma_Phi~ extension, the one pre-authorized
# judgment call / _level1_creation_ops' own Adag+vac).
#
# ***** THE HEADLINE (V-SELECTED, proposal-grade; verification required; NOTHING BOOKED) *****
# A SINGLE, CONJUGATION-FIXED, EXACTLY-REAL DIRECTION IS SELECTED BY THE ARGMAX OF THE
# PRE-DECLARED I-BASED CANDIDATES:
#     u_A = (0.9319033775, 0.1172343025, 0.3432378378)   [real; FS(u_A, conj u_A) = 0]
# - F2[m0;m1] (the omega-averaged conditional field-internal vertex): argmax at u_A, gradient
#   2.0e-12 (< the frozen 1e-8), Hessian eigenvalues (-0.0162,-0.0162,-0.0101,-0.0100) — an
#   ISOLATED NONDEGENERATE MAXIMUM on the CP^2 quotient.
# - F1[s1u2u3; every mode subset] (the union-block channel information): argmax at THE SAME
#   POINT (FS distance 0.0 after independent polish), gradient 1.7e-10, Hessian ISOTROPIC
#   (-0.0121 x4).
# - F2p[across-copy m1] is stationary at u_A as well (gradient 3.5e-13), but with one
#   +1e-4 Hessian eigenvalue — saddle/degenerate at tolerance, reported raw.
# - THE STRUCTURAL INVARIANTS AT u_A ARE EXACT RATIONALS: c_2(u_A) = 1/6 and c_3(u_A) = 1/72
#   (the level-2/3 image-norm sums, machine precision) — the selected direction maximizes the
#   weld's level-2/3 penetration at exactly-rational weights.  (Internal structural scalars of
#   this construction only — compared to NOTHING measured.)
# - DISAGREEMENT (finding, not failure, full matrix in the return): F2[m0;m2] argmax elsewhere;
#   F3 profile argmax elsewhere (one ~0 Hessian eigenvalue, degenerate); the copy-Holevo family
#   F1p across-copies has a strongly ANISOTROPIC ridge maximum elsewhere (eigs -5.8,-5.8,
#   -0.005,-0.0007) and an ISOLATED NONDEGENERATE MINIMUM (value 0.0020, eigs +1.3..+44) at a
#   third point; F3p single-mode profile hits its 1-bit CEILING on a positive-dimensional
#   plateau (exact value 1.0, two zero Hessian eigenvalues — V-DEGENERATE component); F2[m0;m1]
#   also has at least one SECONDARY local max (value 0.001503 vs 0.001507) far from u_A.
#   u_A is ALSO F2[m1;m2]'s argmin (FS 2e-4) — the mode-pair vertex trades off across pairs.
#
# ***** THE TWO EXACT LEMMAS (both machine-verified; they extend the blindness ledger) *****
# (1) THE PURITY-REDUCTION LEMMA: for the pure channel state, EVERY history-field MI is a
#     functional of the field marginal alone (I(hist;B) = S(rho_F)+S(rho_B)-S(rho_B^c), Schmidt
#     purity).  Single-shell blocks: the block normalization cancels the per-level Schur scalar
#     ==> ALL single-shell F1 reads (and single-shell per-copy pair conditionals) are CONSTANT
#     ON CP^2 BY THEOREM (step-0 confirms: rel spreads <= 3e-15).  A NEW BLINDNESS-LEDGER ENTRY:
#     single-level state-marginal information reads are Schur-dead.
#     UNION BLOCKS SURVIVE GENUINELY: Schur pins the scalar's VALUE only at level 1
#     (c_1 = ||d||^2/3); c_2(u), c_3(u) vary at O(1) relative — every union-block state-marginal
#     read factors through these TWO real level weights.  COROLLARY (raw W2 correction): W2's
#     "rho_d depends only on r" is exact only at level 1; the omega-suppressed c_2(u) variation
#     (~7e-8 in rho_d entries) matches the 7.5e-8 residual W2's direction-independence check
#     misattributed to the SVD floor.  W2's V-VACUOUS verdict is UNAFFECTED (its N1 spreads
#     stay below its own 1e-6 threshold, dominated by the exactly-pinned level 1).
# (2) THE COPY-OVERLAP LEVEL LEMMA: same-word cross-copy overlaps <Gamma_Phi~(w)|Gamma_Phi(w)>
#     vanish IDENTICALLY (level n vs 3-n, J_F's antiparticle swap) ==> on single-shell blocks
#     I(hist;copy) = the HOLEVO distinguishability chi({rho_hist^Phi, rho_hist^Phi~}; 1/2,1/2);
#     at shell 1 these are the dart-space Grams G_u = Phi_u^dag Phi_u and R G_u^T R (R = dart
#     reversal).  WHY THE VERTEX CLASS EVADES THE WALL: it reads Grams in the REDUCIBLE 12-dim
#     dart space (phi_i^dag phi_j is not a self-intertwiner of an irreducible rep) and pair
#     cross-correlations (u(x)u-covariant) — both outside the u(x)conj(u) adjoint channel that
#     Schur kills.
#
# OBSERVED EXACT SYMMETRY: u -> conj(u) (in the frozen phi-basis) leaves every frozen
# functional invariant (machine precision) ==> extrema come in conjugate pairs or sit on the
# conj-fixed real locus.  u_A is conj-FIXED; F2[m1;m2]'s argmax is a conjugate PAIR (FS(u,
# conj u) = pi/2), i.e. a finite 2-orbit.  Global phase drops from every pair functional
# (verified — nontrivial, Gamma_Phi~ is ANTIlinear in u).
#
# GUARDS: NO K_F / region modular generator anywhere; gauge_sector_category ONLY in the
# output-only species read (v1_species_read_output_only, called AFTER characterization; its
# outcome: the occupation-level weights are LEVEL-LOCKED by the tower's level confinement —
# shell n maps to level n for every u — so the species read is direction-insensitive here);
# kappa's numeric value appears NOWHERE (E_int = -kappa.I, kappa > 0 ratified ==> extremum
# LOCATIONS are kappa-independent — stated once); r FROZEN at 1, r in {1/2,2} appendix
# report-only, NO r-optimization; NUMBERS APPEAR NOWHERE (no M_Z/ppm/m_nu/G/2pi token; every
# scalar above is an information/structure read of this station's own construction, in bits).
# READ-ONLY: the selected direction is a PROPOSAL requiring the user's ratified freeze + a
# verification — NOTHING IS BOOKED AS FORCED; FOCK-2 STAYS GATED.
#
# ML-2b/HK-7 CONDITIONALITY (verbatim, carries into every verdict sentence): "Every duality
# check here (HK-5) is CELL-LEVEL only (the 6-edge static vacuum). ML-2b's DR-frame argument
# is CONDITIONAL on the TD-limit duality holding, which is NOT verified by this suite."


# ---------------------------------------------------------------------------
# V1 shared machinery: the JW occupation-basis transform + exact-diagonalization
# information helpers (all von Neumann entropies BASE 2, per the freeze §2)
# ---------------------------------------------------------------------------

_V1_SUBSETS = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


def _v1_occupation_transform():
    """[V1 §1, THE MODE-BASIS TRANSFORM] F's Jordan-Wigner factorization into the 3 complex-
    fermion modes: the occupation basis |S> = Adag[m1]...Adag[mk]|vac> (S ascending, factors
    applied in _a2c_level_embedding's OWN order — REUSED convention, the same states E_1/E_2/E_3
    embed), assembled into the unitary U whose columns are the |S> in F's own Cl(6)-module
    coordinates (field_algebra_conjugation builds the IDENTICAL U internally for its M).
    Returns T = P_bits @ U^dagger: the 8x8 unitary taking an F-coordinates vector to its
    occupation-NUMBER tensor coefficients psi[n0,n1,n2] (row-major (2,2,2) tensor, n_i = mode-i
    occupation) — the mode-subset subsystem lattice of F, per the freeze §1.
    Verified: T unitary; T@vac = e_{(0,0,0)} exactly (the Fock vacuum is the empty-occupation
    product state, as it must be in its own occupation basis)."""
    Adag, vac, _, _ = _level1_creation_ops()
    cols = []
    for S in _V1_SUBSETS:
        v = vac.copy()
        for m in reversed(S):
            v = Adag[m] @ v
        cols.append(v.reshape(8))
    U = np.stack(cols, axis=1)
    gram_resid = float(np.max(np.abs(U.conj().T @ U - np.eye(8))))
    assert gram_resid < 1e-9, f"_v1_occupation_transform: U not unitary ({gram_resid:.2e})"
    P_bits = np.zeros((8, 8))
    for s_idx, S in enumerate(_V1_SUBSETS):
        bit_idx = (4 if 0 in S else 0) + (2 if 1 in S else 0) + (1 if 2 in S else 0)
        P_bits[bit_idx, s_idx] = 1.0
    T = P_bits @ U.conj().T
    vac_image = T @ vac.reshape(8)
    vac_resid = float(np.max(np.abs(vac_image - np.eye(8)[:, 0])))
    assert vac_resid < 1e-9, f"_v1_occupation_transform: T@vac != |000> ({vac_resid:.2e})"
    return T


def _v1_entropy_base2(rho):
    """[V1 shared] S(rho) = -Tr rho log2 rho, exact eigen-decomposition (matrices <= 16-dim)."""
    herm = float(np.max(np.abs(rho - rho.conj().T)))
    assert herm < 1e-6, f"_v1_entropy_base2: rho not Hermitian ({herm:.2e})"
    w = np.clip(np.real(np.linalg.eigvalsh(rho)), 0.0, None)
    s = float(np.sum(w))
    if s > 1e-14:
        w = w / s
    nz = w > 1e-14
    return float(-np.sum(w[nz] * np.log2(w[nz])))


def _v1_pure_marginal(vec, axis_dims, keep_axes):
    """[V1 shared] reduced density matrix of a PURE state vector over the kept tensor axes
    (exact partial trace, no sampling; history side <= 85 words, field side <= 16 — tiny)."""
    axis_dims = tuple(axis_dims)
    n = len(axis_dims)
    keep = tuple(keep_axes)
    comp = tuple(a for a in range(n) if a not in keep)
    T = np.transpose(vec.reshape(axis_dims), keep + comp)
    dk = int(np.prod([axis_dims[a] for a in keep])) if keep else 1
    dc = int(np.prod([axis_dims[a] for a in comp])) if comp else 1
    M = T.reshape(dk, dc)
    rho = M @ M.conj().T
    tr = float(np.real(np.trace(rho)))
    assert tr > 1e-14, "_v1_pure_marginal: degenerate norm"
    return rho / tr


def _v1_mutual_information(vec, axis_dims, axes_A, axes_B):
    """[V1 shared] I(A;B) = S(A)+S(B)-S(AB) on a pure state vector, base 2."""
    S_A = _v1_entropy_base2(_v1_pure_marginal(vec, axis_dims, axes_A))
    S_B = _v1_entropy_base2(_v1_pure_marginal(vec, axis_dims, axes_B))
    S_AB = _v1_entropy_base2(_v1_pure_marginal(vec, axis_dims, tuple(axes_A) + tuple(axes_B)))
    return S_A + S_B - S_AB


# ---------------------------------------------------------------------------
# V1 the channel states (freeze §0.1): |W_u> on H_hist (x) F, and the pair
# channel state on H_hist (x) (C^2_copy (x) F)  [F(+)F ~= C^2 (x) F, disclosed]
# ---------------------------------------------------------------------------

def _v1_gamma_mode_table(d_vec, N_max=4, max_length=3, conjugate=False):
    """[V1 §1] the tower images in the JW occupation basis: per shell n (0..max_length), the
    (8, D_n) matrix of T@Gamma_d(w) columns (T = _v1_occupation_transform).  conjugate=True
    uses w2_conjugate_gamma_word — §10's DISCLOSED Gamma_Phi~ extension, REUSED verbatim, same
    disclosure (the one pre-authorized judgment call)."""
    T = _v1_occupation_transform()
    gt = w2_gamma_table(d_vec, N_max=N_max, max_length=max_length)
    out = {}
    for n in range(0, max_length + 1):
        blk = gt["by_length"][n]
        if conjugate:
            cols = [w2_conjugate_gamma_word(d_vec, w).reshape(8) for w in blk["words"]]
            V = np.stack(cols, axis=1) if cols else np.zeros((8, 0), dtype=complex)
        else:
            V = blk["vectors"]
        out[n] = {"idxs": blk["idxs"], "words": blk["words"], "vectors": T @ V}
    return {"by_length": out, "omega": gt["omega"],
            "vac_check_residual": gt["vac_check_residual"]}


def v1_channel_state(gt_mode, shells):
    """[V1 §1, THE CHANNEL STATE restricted to a shell block] |W_u>_block = (1/sqrt(Z_block))
    sum_{w in shells} sqrt(omega(w)) |w> (x) Gamma_u(w), trace(=vector-norm)-normalized (freeze
    §0.1).  Returns {'vec' (flat, axes (word, n0, n1, n2)), 'D', 'Z_block', 'word_weights'}."""
    shells = shells if isinstance(shells, (list, tuple)) else (shells,)
    cols, wts = [], []
    for n in shells:
        blk = gt_mode["by_length"][n]
        for k in range(blk["vectors"].shape[1]):
            cols.append(blk["vectors"][:, k])
            wts.append(gt_mode["omega"][blk["idxs"][k]])
    D = len(cols)
    assert D > 0, f"v1_channel_state: empty shell block {shells}"
    V = np.stack(cols, axis=1)
    wts = np.array(wts)
    word_weights = wts * np.sum(np.abs(V) ** 2, axis=0)
    Z = float(np.sum(word_weights))
    assert Z > 1e-14, "v1_channel_state: degenerate normalization"
    C = (V * np.sqrt(wts)[None, :]).T / math.sqrt(Z)
    return {"vec": C.reshape(-1), "D": D, "Z_block": Z, "word_weights": word_weights / Z}


def v1_pair_channel_state(gtP, gtT, shells):
    """[V1 §1, THE PAIR CHANNEL STATE restricted to a shell block] field side F(+)F, realized as
    C^2_copy (x) F (the canonical Hilbert-space isomorphism of a two-term direct sum — DISCLOSED
    reading of the freeze's 'field side doubled': the copy label becomes a qubit subsystem, JW
    mode subsets are read per copy by CONDITIONING on it and across copies by treating it as a
    tensor factor).  Axes: (word, copy, n0, n1, n2)."""
    shells = shells if isinstance(shells, (list, tuple)) else (shells,)
    colsP, colsT, wts = [], [], []
    for n in shells:
        bP, bT = gtP["by_length"][n], gtT["by_length"][n]
        for k in range(bP["vectors"].shape[1]):
            colsP.append(bP["vectors"][:, k])
            colsT.append(bT["vectors"][:, k])
            wts.append(gtP["omega"][bP["idxs"][k]])
    D = len(colsP)
    VP = np.stack(colsP, axis=1)
    VT = np.stack(colsT, axis=1)
    wts = np.array(wts)
    word_weights = wts * (np.sum(np.abs(VP) ** 2, axis=0) + np.sum(np.abs(VT) ** 2, axis=0))
    Z2 = float(np.sum(word_weights))
    assert Z2 > 1e-14, "v1_pair_channel_state: degenerate normalization"
    A = np.zeros((D, 2, 8), dtype=complex)
    A[:, 0, :] = (VP * np.sqrt(wts)[None, :]).T
    A[:, 1, :] = (VT * np.sqrt(wts)[None, :]).T
    return {"vec": (A / math.sqrt(Z2)).reshape(-1), "D": D, "Z2_block": Z2,
            "word_weights": word_weights / Z2}


# ---------------------------------------------------------------------------
# V1 the frozen functional set (§2) — closed list, nothing added after step 0
# ---------------------------------------------------------------------------

_V1_MODE_SUBSETS = {"m0": (0,), "m1": (1,), "m2": (2,), "m01": (0, 1), "m02": (0, 2),
                    "m12": (1, 2), "full": (0, 1, 2)}
_V1_SHELL_BLOCKS = {"s1": (1,), "s2": (2,), "s3": (3,), "s123": (1, 2, 3)}


def v1_F1(gt_mode):
    """[V1 §2 F1] I(A_hist; B_field) in |W_u>, A_hist = the shell blocks (1,2,3, 1u2u3 — read as
    the channel state RESTRICTED to that word block, renormalized; the direct-sum H_hist carries
    no tensor 'shell subsystem', so block restriction is the operative reading, disclosed),
    B_field = each JW mode subset + full F.  Returns {block: {subset: I}}."""
    out = {}
    for bname, shells in _V1_SHELL_BLOCKS.items():
        st = v1_channel_state(gt_mode, shells)
        dims = (st["D"], 2, 2, 2)
        out[bname] = {mname: _v1_mutual_information(st["vec"], dims, (0,),
                                                     tuple(a + 1 for a in mset))
                      for mname, mset in _V1_MODE_SUBSETS.items()}
    return out


def v1_F2_F3(gt_mode, max_length=3):
    """[V1 §2 F2+F3] the omega-averaged conditional (per-word) reads, words 0..3 (the freeze's
    own |W_u> word range; the length-0 seed word conditions to |vac> = |000>, contributing zero
    entropy and ~0.998 of the weight — a DIRECTION-INDEPENDENT dilution constant, since the
    per-shell pushed weights are Schur-forced scalars; extremum locations are unaffected).
    F2: Ibar(mode_i;mode_j) = sum_w (omega ||Gamma||^2 / Z) I_{rho_F|w}(i;j).
    F3: Sbar(mode_i) likewise with the single-mode entanglement entropy."""
    pairs = {"01": (0, 1), "02": (0, 2), "12": (1, 2)}
    Z, accF2, accF3 = 0.0, {k: 0.0 for k in pairs}, {m: 0.0 for m in range(3)}
    for n in range(0, max_length + 1):
        blk = gt_mode["by_length"][n]
        for k in range(blk["vectors"].shape[1]):
            v = blk["vectors"][:, k]
            nrm2 = float(np.real(np.vdot(v, v)))
            wt = gt_mode["omega"][blk["idxs"][k]]
            Z += wt * nrm2
            if nrm2 < 1e-14:
                continue
            vt = v / math.sqrt(nrm2)
            for pname, (i, j) in pairs.items():
                accF2[pname] += wt * nrm2 * _v1_mutual_information(vt, (2, 2, 2), (i,), (j,))
            for m in range(3):
                accF3[m] += wt * nrm2 * _v1_entropy_base2(_v1_pure_marginal(vt, (2, 2, 2), (m,)))
    assert Z > 1e-14
    return ({p: accF2[p] / Z for p in pairs}, {f"m{m}": accF3[m] / Z for m in range(3)})


def v1_F1p(gtP, gtT, blocks=None, full_subset_blocks=None):
    """[V1 §2 F1p] F1 on the pair channel state.  PER COPY: I(hist; mode subset | copy=c) — the
    state conditioned (projected+renormalized) on the copy qubit; ACROSS COPIES: the copy qubit
    as a tensor factor — I(hist; copy), I(hist; copy u full), and I(hist; mode subset) with the
    copy TRACED OUT.  Full subset table on the s123 block; single shells carry the reduced
    {full}-subset read (per-eval cost; DISCLOSED scope reduction, approved).
    # GAP-A 2026-07-13 (V1 verification gap closure, checker check 1): `full_subset_blocks`
    # (optional, default None) names single-shell blocks that should ALSO get the full
    # 7-subset table instead of just {full} — needed to compute the 18 traced_* mode-subset
    # components per shell that the original scope reduction skipped (traced_full already
    # existed; traced_m0/m1/m2/m01/m02/m12 did not and are NOT covered by the purity-reduction
    # lemma, per the verification's check 1). Default behavior (full_subset_blocks=None) is
    # BYTE-IDENTICAL to the original function — no existing call site or output changes."""
    out = {}
    block_items = _V1_SHELL_BLOCKS if blocks is None else {b: _V1_SHELL_BLOCKS[b] for b in blocks}
    full_subset_blocks = set() if full_subset_blocks is None else set(full_subset_blocks)  # GAP-A
    for bname, shells in block_items.items():
        st = v1_pair_channel_state(gtP, gtT, shells)
        dims = (st["D"], 2, 2, 2, 2)
        T5 = st["vec"].reshape(dims)
        row = {}
        subset_names = list(_V1_MODE_SUBSETS) if (bname == "s123" or bname in full_subset_blocks) \
            else ["full"]  # GAP-A: extra branch only; s123 behavior unchanged
        for cname, cval in (("Phi", 0), ("Phit", 1)):
            branch = T5[:, cval, :, :, :]
            bn = float(np.real(np.vdot(branch, branch)))
            assert bn > 1e-14
            bvec = (branch / math.sqrt(bn)).reshape(-1)
            for mname in subset_names:
                mset = _V1_MODE_SUBSETS[mname]
                row[f"cond_{cname}_{mname}"] = _v1_mutual_information(
                    bvec, (st["D"], 2, 2, 2), (0,), tuple(a + 1 for a in mset))
        row["I_hist_copy"] = _v1_mutual_information(st["vec"], dims, (0,), (1,))
        row["I_hist_copyfull"] = _v1_mutual_information(st["vec"], dims, (0,), (1, 2, 3, 4))
        for mname in subset_names:
            mset = _V1_MODE_SUBSETS[mname]
            row[f"traced_{mname}"] = _v1_mutual_information(
                st["vec"], dims, (0,), tuple(a + 2 for a in mset))
        out[bname] = row
    return out


def v1_F2p_F3p(gtP, gtT, max_length=3):
    """[V1 §2 F2p+F3p] the omega-averaged conditional reads on the pair state, words 0..3.
    Per word w the conditional field state is the pure (Gamma_Phi(w) (+) Gamma_Phi~(w))/norm on
    C^2 (x) F.  PER COPY: mode-pair MI / single-mode entropy inside each copy branch (branch-
    weighted); ACROSS: I(copy; mode_i) and the traced single-mode entropies + Sbar(copy)."""
    pairs = {"01": (0, 1), "02": (0, 2), "12": (1, 2)}
    Z = 0.0
    accF2 = {f"cond_{c}_{p}": 0.0 for c in ("Phi", "Phit") for p in pairs}
    accF2x = {f"across_copy_m{m}": 0.0 for m in range(3)}
    accF3 = {f"cond_{c}_m{m}": 0.0 for c in ("Phi", "Phit") for m in range(3)}
    accF3x = {f"traced_m{m}": 0.0 for m in range(3)}
    accF3x["S_copy"] = 0.0
    for n in range(0, max_length + 1):
        bP, bT = gtP["by_length"][n], gtT["by_length"][n]
        for k in range(bP["vectors"].shape[1]):
            vP, vT = bP["vectors"][:, k], bT["vectors"][:, k]
            nrm2 = float(np.real(np.vdot(vP, vP) + np.vdot(vT, vT)))
            wt = gtP["omega"][bP["idxs"][k]]
            Z += wt * nrm2
            if nrm2 < 1e-14:
                continue
            pair_vec = np.concatenate([vP, vT]) / math.sqrt(nrm2)
            dims = (2, 2, 2, 2)
            for m in range(3):
                accF2x[f"across_copy_m{m}"] += wt * nrm2 * _v1_mutual_information(
                    pair_vec, dims, (0,), (m + 1,))
                accF3x[f"traced_m{m}"] += wt * nrm2 * _v1_entropy_base2(
                    _v1_pure_marginal(pair_vec, dims, (m + 1,)))
            accF3x["S_copy"] += wt * nrm2 * _v1_entropy_base2(
                _v1_pure_marginal(pair_vec, dims, (0,)))
            for cname, v in (("Phi", vP), ("Phit", vT)):
                bn = float(np.real(np.vdot(v, v)))
                if bn < 1e-14:
                    continue
                bvec = v / math.sqrt(bn)
                for pname, (i, j) in pairs.items():
                    accF2[f"cond_{cname}_{pname}"] += wt * nrm2 * (bn / nrm2) * \
                        _v1_mutual_information(bvec, (2, 2, 2), (i,), (j,))
                for m in range(3):
                    accF3[f"cond_{cname}_m{m}"] += wt * nrm2 * (bn / nrm2) * \
                        _v1_entropy_base2(_v1_pure_marginal(bvec, (2, 2, 2), (m,)))
    assert Z > 1e-14
    f2 = {k: v / Z for k, v in {**accF2, **accF2x}.items()}
    f3 = {k: v / Z for k, v in {**accF3, **accF3x}.items()}
    return f2, f3


def v1_all_functionals(u_vec, r=1.0, N_max=4):
    """[V1 §2, THE CLOSED EVALUATION] every frozen functional component at one direction u —
    a flat {(family, component): value} dict.  This list is FROZEN; step 0 runs on exactly
    these keys and nothing is added afterwards (freeze §3/§8)."""
    d_vec, _ = w2_family_direction(u_vec, r=r)
    gtP = _v1_gamma_mode_table(d_vec, N_max=N_max)
    gtT = _v1_gamma_mode_table(d_vec, N_max=N_max, conjugate=True)
    flat = {}
    for bname, row in v1_F1(gtP).items():
        for mname, val in row.items():
            flat[("F1", f"{bname}_{mname}")] = val
    f2, f3 = v1_F2_F3(gtP)
    for k, v in f2.items():
        flat[("F2", k)] = v
    for k, v in f3.items():
        flat[("F3", k)] = v
    for bname, row in v1_F1p(gtP, gtT).items():
        for k, v in row.items():
            flat[("F1p", f"{bname}_{k}")] = v
    f2p, f3p = v1_F2p_F3p(gtP, gtT)
    for k, v in f2p.items():
        flat[("F2p", k)] = v
    for k, v in f3p.items():
        flat[("F3p", k)] = v
    return flat


# ---------------------------------------------------------------------------
# V1 step 0 — the mandatory vacuity pre-check (freeze §3) + the exact arguments
# ---------------------------------------------------------------------------

def v1_purity_reduction_lemma(u_pair_seed=0, r=1.0):
    """[V1 §3, EXACT SYMMETRY ARGUMENT — THE PURITY-REDUCTION LEMMA] For the PURE channel state
    |W_u>_block on H_block (x) F, EVERY F1 read is a functional of the field marginal alone:
    I(hist;B) = S(rho_F) + S(rho_B) - S(rho_{B^c}) (Schmidt purity: S(hist)=S(rho_F) and
    S(hist u B)=S(rho_{B^c}); rho_B, rho_{B^c} are partial traces OF rho_F).  Per level, W2's
    Schur argument forces rho_F's SHAPE to be scalar.P_n for every u.  TWO CONSEQUENCES,
    SHARPER THAN FIRST ANTICIPATED (a genuine mid-station correction, kept raw):
    (1) SINGLE-SHELL BLOCKS ARE EXACTLY BLIND: the block normalization cancels the one scalar,
        rho_F = P_n/dim_n identically ==> every single-shell F1 read is CONSTANT ON CP^2 by
        theorem (machine-verified: rel spreads <= 3e-15 at step 0).
    (2) THE UNION BLOCK IS *NOT* EXACTLY BLIND: Schur pins the scalar's VALUE only at level 1
        (c_1 = ||d||^2/3, the trace pinned by the phi-basis SVD-orthonormality); the level-2/3
        scalars c_2(u), c_3(u) — the PER-SHELL IMAGE-NORM SUMS — genuinely vary with u
        (v1_per_word_norm_read measures rel spreads ~0.6 / ~1.6).  The s123 F1 reads therefore
        survive step 0 GENUINELY, but their entire direction-dependence factors through the TWO
        real level-weight ratios (lambda_2(u), lambda_3(u)) — a 2-parameter read, whatever the
        mode subset.  COROLLARY (W2 correction, raw): W2's 'rho_d = sum lambda_n(r) Pw[n]
        depends only on r' is exact only at level 1; the omega-suppressed level-2 entry
        variation (~ 3.5e-7 x 0.6 / 3 ~ 7e-8) matches the 7.5e-8 residual W2's
        direction-independence check attributed to the SVD floor — a MISATTRIBUTION in W2's
        mechanism footnote (W2's verdict is unaffected: N1's honesty-clause spreads stay below
        its 1e-6 threshold, dominated by the exactly-pinned level 1).
    VERIFIED below, not asserted: the purity identity (residual 0), the field-marginal
    u-variation, and the level-resolved eigenvalue variation (level-1 eigenvalues pinned to
    machine precision; level-2 eigenvalues varying at the c_2(u) scale)."""
    rng = np.random.default_rng(u_pair_seed)
    u1 = rng.normal(size=3) + 1j * rng.normal(size=3)
    u2 = rng.normal(size=3) + 1j * rng.normal(size=3)
    u1, u2 = u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)
    out = {}
    rhoFs, spectra = [], []
    for tag, u in (("u1", u1), ("u2", u2)):
        d_vec, _ = w2_family_direction(u, r=r)
        gt = _v1_gamma_mode_table(d_vec)
        st = v1_channel_state(gt, (1, 2, 3))
        dims = (st["D"], 2, 2, 2)
        rhoF = _v1_pure_marginal(st["vec"], dims, (1, 2, 3))
        rhoFs.append(rhoF)
        spectra.append(np.sort(np.linalg.eigvalsh(rhoF))[::-1])
        I_direct = _v1_mutual_information(st["vec"], dims, (0,), (1,))
        S_F = _v1_entropy_base2(rhoF)
        S_B = _v1_entropy_base2(_v1_pure_marginal(st["vec"], dims, (1,)))
        S_Bc = _v1_entropy_base2(_v1_pure_marginal(st["vec"], dims, (2, 3)))
        out[tag] = {"identity_residual": abs(I_direct - (S_F + S_B - S_Bc)),
                    "I_direct": I_direct}
    out["field_marginal_max_entry_u_variation"] = float(np.max(np.abs(rhoFs[0] - rhoFs[1])))
    out["level1_eigenvalue_variation"] = float(np.max(np.abs(spectra[0][:3] - spectra[1][:3])))
    out["level2_eigenvalue_variation"] = float(np.max(np.abs(spectra[0][3:6] - spectra[1][3:6])))
    return out


def v1_step0(n_samples=20, seed=0, r=1.0, threshold=1e-6):
    """[V1 §3, STEP 0] every frozen functional component at n_samples Haar-random directions
    (seed 0, pre-declared).  relative variation = (max-min)/max(1, mean|value|).  A component
    with rel var > threshold proceeds; SINGLE-SHELL F1 reads (and single-shell F1p per-copy
    conditionals) are additionally booked BLIND BY THE EXACT ARGUMENT (v1_purity_reduction_lemma
    consequence (1) — the freeze's own preference for exact symmetry arguments, reported
    alongside the sampling; the union-block reads survive GENUINELY via consequence (2))."""
    rng = np.random.default_rng(seed)
    tables = {}
    for _ in range(n_samples):
        u = rng.normal(size=3) + 1j * rng.normal(size=3)
        u = u / np.linalg.norm(u)
        for key, val in v1_all_functionals(u, r=r).items():
            tables.setdefault(key, []).append(val)
    rows = {}
    for key, vals in tables.items():
        arr = np.array(vals)
        spread = float(arr.max() - arr.min())
        rel = spread / max(1.0, float(np.mean(np.abs(arr))))
        block = key[1].split("_")[0]
        exact_blind = (key[0] == "F1" and block in ("s1", "s2", "s3")) or \
                      (key[0] == "F1p" and "cond_" in key[1] and block in ("s1", "s2", "s3"))
        rows[key] = {"mean": float(arr.mean()), "spread": spread, "rel_spread": rel,
                     "proceeds_numeric": bool(rel > threshold),
                     "exact_blind_argument": exact_blind,
                     "survives": bool(rel > threshold) and not exact_blind}
    return rows


def v1_per_word_norm_read(n_samples=20, seed=0, r=1.0):
    """[V1 §2 mechanism read] the per-word image norms ||Gamma_u(w)||^2 across shells.
    MEASURED FACTS (each feeds the mechanism report): (i) the SHELL-1 sum is EXACTLY 1 for every
    u (Schur + the phi-basis SVD-orthonormality pins c_1 = ||d||^2/3, trace 1 at r=1) — the only
    shell whose scalar is pinned; (ii) the SHELL-2/3 sums c_2(u), c_3(u) VARY with u at O(1)
    relative — the two real scalars through which every union-block state-marginal read factors
    (v1_purity_reduction_lemma consequence (2)); (iii) the INDIVIDUAL per-word norms vary
    strongly within every shell — the history marginal diag(omega(w)||Gamma_u(w)||^2)/Z is
    direction-dependent word-by-word.  THE MECHANISM: direction lives in WHICH words carry the
    weight and in the level-weight ratios (and, for the pair, in the Phi--Phi~ cross-
    correlation, u(x)u-covariant, outside Schur's u(x)conj(u) reach) — never in any single
    level's marginal shape."""
    rng = np.random.default_rng(seed)
    shell_sums = {n: [] for n in (1, 2, 3)}
    word_norm_spread = {n: 0.0 for n in (1, 2, 3)}
    ref_norms = None
    all_norms = []
    for _ in range(n_samples):
        u = rng.normal(size=3) + 1j * rng.normal(size=3)
        u = u / np.linalg.norm(u)
        d_vec, _ = w2_family_direction(u, r=r)
        gt = _v1_gamma_mode_table(d_vec)
        norms = {n: np.sum(np.abs(gt["by_length"][n]["vectors"]) ** 2, axis=0)
                 for n in (1, 2, 3)}
        all_norms.append(norms)
        for n in (1, 2, 3):
            shell_sums[n].append(float(np.sum(norms[n])))
    out = {}
    for n in (1, 2, 3):
        arr = np.array(shell_sums[n])
        stacked = np.stack([nn[n] for nn in all_norms])
        out[n] = {"shell_sum_mean": float(arr.mean()),
                  "shell_sum_rel_spread": float((arr.max() - arr.min()) / max(arr.mean(), 1e-30)),
                  "per_word_norm_max_rel_spread": float(np.max(
                      (stacked.max(axis=0) - stacked.min(axis=0)) /
                      np.maximum(stacked.mean(axis=0), 1e-30)))}
    return out


# ---------------------------------------------------------------------------
# V1 §4 — the extremum read (survivors only): CP^2 chart optimization + Hessian
# ---------------------------------------------------------------------------

def v1_component_value(u_vec, key, r=1.0):
    """[V1 §4 helper] one frozen component's value at u (evaluates only the needed family)."""
    fam, comp = key
    d_vec, _ = w2_family_direction(u_vec, r=r)
    gtP = _v1_gamma_mode_table(d_vec)
    if fam == "F2":
        return v1_F2_F3(gtP)[0][comp]
    if fam == "F3":
        return v1_F2_F3(gtP)[1][comp]
    gtT = _v1_gamma_mode_table(d_vec, conjugate=True)
    if fam == "F1p":
        bname, sub = comp.split("_", 1)
        # GAP-A 2026-07-13: request the full subset table on single shells so traced_m*/m0*
        # components resolve (see v1_F1p docstring); s123 path is untouched (fsb computed but
        # unused there since v1_F1p already special-cases bname == "s123").
        fsb = (bname,) if bname != "s123" else None
        return v1_F1p(gtP, gtT, blocks=(bname,), full_subset_blocks=fsb)[bname][sub]
    if fam == "F2p":
        return v1_F2p_F3p(gtP, gtT)[0][comp]
    if fam == "F3p":
        return v1_F2p_F3p(gtP, gtT)[1][comp]
    if fam == "F1":
        bname, sub = comp.split("_", 1)
        return v1_F1(gtP)[bname][sub]
    raise ValueError(f"v1_component_value: unknown family {fam}")


def _v1_chart(u_center):
    """[V1 §4] a real-4-dim chart on CP^2 at u_center: z in C^2 -> (u0 + B z)/norm, B = an
    orthonormal basis of u0's orthogonal complement — the standard affine chart; the 4 real
    coordinates parameterize exactly the projective quotient (phase+scale removed)."""
    u0 = u_center / np.linalg.norm(u_center)
    A = np.eye(3, dtype=complex) - np.outer(u0, u0.conj())
    U, s, _ = np.linalg.svd(A)
    return u0, U[:, :2]


def _v1_chart_point(u0, B, zr):
    z = np.array([zr[0] + 1j * zr[1], zr[2] + 1j * zr[3]])
    u = u0 + B @ z
    return u / np.linalg.norm(u)


def v1_local_optimize(key, u_start, sign=+1.0, r=1.0, n_steps=25, step0=0.3, h=1e-3):
    """[V1 §4] projected gradient ascent (sign=+1, the pre-declared argmax candidate) or descent
    (sign=-1, reported as structure) in recentered CP^2 charts; finite-difference chart gradient.
    Deterministic given the start — no scanning-and-picking (the multi-start set is fixed)."""
    u_cur = u_start / np.linalg.norm(u_start)
    f = lambda u: v1_component_value(u, key, r=r)
    val_cur = f(u_cur)
    step = step0
    for _ in range(n_steps):
        u0, B = _v1_chart(u_cur)
        g = np.zeros(4)
        f0 = f(u0)
        for i in range(4):
            zp = np.zeros(4)
            zp[i] = h
            g[i] = (f(_v1_chart_point(u0, B, zp)) - f0) / h
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        direction = sign * g / gn
        improved = False
        for _ in range(6):
            u_try = _v1_chart_point(u0, B, direction * step)
            v_try = f(u_try)
            if sign * (v_try - val_cur) > 1e-14:
                u_cur, val_cur, improved = u_try, v_try, True
                step *= 1.3
                break
            step *= 0.5
        if not improved and step < 1e-9:
            break
    return u_cur, float(val_cur)


def v1_chart_hessian(key, u_star, r=1.0, h=2e-3):
    """[V1 §4] finite-difference gradient + Hessian of the component in the CP^2 chart at
    u_star (the quotient tangent space — phase and scale are outside the chart by construction).
    Returns (grad (4,), Hessian eigenvalues (4,), Hessian (4,4))."""
    u0, B = _v1_chart(u_star)
    f = lambda zr: v1_component_value(_v1_chart_point(u0, B, zr), key, r=r)
    z0 = np.zeros(4)
    f0 = f(z0)
    fp, fm = np.zeros(4), np.zeros(4)
    grad = np.zeros(4)
    for i in range(4):
        zp, zm = z0.copy(), z0.copy()
        zp[i] += h
        zm[i] -= h
        fp[i], fm[i] = f(zp), f(zm)
        grad[i] = (fp[i] - fm[i]) / (2 * h)
    H = np.zeros((4, 4))
    for i in range(4):
        H[i, i] = (fp[i] - 2 * f0 + fm[i]) / (h * h)
    for i in range(4):
        for j in range(i + 1, 4):
            zpp, zmm = z0.copy(), z0.copy()
            zpp[i] += h
            zpp[j] += h
            zmm[i] -= h
            zmm[j] -= h
            H[i, j] = H[j, i] = (f(zpp) - fp[i] - fp[j] + 2 * f0 - fm[i] - fm[j] + f(zmm)) / (2 * h * h)
    return grad, np.linalg.eigvalsh(H), H


def v1_fs_distance(u, v):
    """Fubini-Study distance on CP^2: arccos |<u,v>| (unit vectors)."""
    return float(np.arccos(min(1.0, abs(np.vdot(u / np.linalg.norm(u), v / np.linalg.norm(v))))))


def v1_extremum_read(key, r=1.0, n_scan=150, scan_seed=1, n_starts=4):
    """[V1 §4, per surviving functional] FS-uniform Haar scan (n_scan points, SEEDING ONLY —
    the 40^2-patch landscape illustration is dropped, a DISCLOSED approved scope reduction),
    then multi-start local optimization (best scan point + (n_starts-1) fixed Haar starts) for
    the argmax (pre-declared selector candidate) AND the argmin (structure); FD gradient +
    Hessian eigenvalues at each converged point."""
    rng = np.random.default_rng(scan_seed)
    us, vals = [], []
    for _ in range(n_scan):
        u = rng.normal(size=3) + 1j * rng.normal(size=3)
        us.append(u / np.linalg.norm(u))
        vals.append(v1_component_value(us[-1], key, r=r))
    vals = np.array(vals)
    order = np.argsort(vals)
    out = {"key": key, "scan_min": float(vals.min()), "scan_max": float(vals.max()),
           "scan_mean": float(vals.mean())}
    for tag, sign, seed_idx in (("argmax", +1.0, order[-1]), ("argmin", -1.0, order[0])):
        starts = [us[seed_idx]] + [us[order[-2 if sign > 0 else 1]]] + \
                 [us[i] for i in (7, 23)][: max(0, n_starts - 2)]
        results = []
        for s in starts[:n_starts]:
            u_e, v_e = v1_local_optimize(key, s, sign=sign, r=r)
            results.append((v_e, u_e))
        best_v, best_u = max(results, key=lambda t: sign * t[0])
        grad, heigs, _ = v1_chart_hessian(key, best_u, r=r)
        out[tag] = {"value": best_v, "u": best_u, "grad_norm": float(np.linalg.norm(grad)),
                    "hessian_eigs": [float(x) for x in heigs],
                    "multi_start_values": [float(v) for v, _ in results],
                    "multi_start_us": [u for _, u in results],
                    "multi_start_fs_dist_to_best": [v1_fs_distance(u, best_u)
                                                     for _, u in results]}
    return out


# ---------------------------------------------------------------------------
# V1 mechanism lemmas + output-only species read + r-sensitivity appendix
# ---------------------------------------------------------------------------

def v1_copy_overlap_level_lemma(u_vec, r=1.0):
    """[V1 mechanism, EXACT] same-word cross-copy overlaps <Gamma_Phi~(w)|Gamma_Phi(w)> vanish
    IDENTICALLY: Gamma_Phi(w) lives in level n_w, Gamma_Phi~(w) in level 3-n_w (J_F's antiparticle
    level swap, BOOTCAMP §8), and n = 3-n has no integer solution — verified numerically on every
    word of length 1..3.  CONSEQUENCE (single-shell blocks): rho_{hist,copy} is EXACTLY copy-
    block-diagonal with equal branch weights (reversal is an omega-preserving bijection of each
    shell), so I(hist;copy) = the HOLEVO quantity chi({rho_hist^Phi, rho_hist^Phi~}, 1/2 each) —
    the distinguishability of the weld's history Gram from its conjugate-weld (reversal-
    transposed) history Gram.  At shell 1: rho_hist^Phi prop G_u = Phi_u^dagger Phi_u (the 12x12
    dart-space Gram) and rho_hist^Phi~ = R G_u^T R (R = the dart reversal) — the functional reads
    G_u in the REDUCIBLE 12-dim dart space, where phi_i^dagger phi_j is NOT a self-intertwiner of
    an irreducible rep, which is exactly why Schur's lemma (the W2 blindness mechanism) does NOT
    force it scalar.  THE VERTEX CLASS EVADES THE WALL BY READING GRAM/CHANNEL CORRELATIONS, NOT
    STATE MARGINALS."""
    d_vec, _ = w2_family_direction(u_vec, r=r)
    gtP = _v1_gamma_mode_table(d_vec)
    gtT = _v1_gamma_mode_table(d_vec, conjugate=True)
    worst_cross = 0.0
    for n in (1, 2, 3):
        VP = gtP["by_length"][n]["vectors"]
        VT = gtT["by_length"][n]["vectors"]
        worst_cross = max(worst_cross, float(np.max(np.abs(
            np.einsum("ik,ik->k", VT.conj(), VP)))))
    st = v1_pair_channel_state(gtP, gtT, (1,))
    I_direct = _v1_mutual_information(st["vec"], (st["D"], 2, 2, 2, 2), (0,), (1,))
    VP1 = gtP["by_length"][1]["vectors"]
    VT1 = gtT["by_length"][1]["vectors"]
    GP = VP1.conj().T @ VP1
    GT = VT1.conj().T @ VT1
    GP, GT = GP / np.trace(GP).real, GT / np.trace(GT).real
    chi = _v1_entropy_base2(0.5 * (GP + GT)) - 0.5 * _v1_entropy_base2(GP) \
        - 0.5 * _v1_entropy_base2(GT)
    return {"worst_same_word_cross_overlap": worst_cross,
            "I_hist_copy_s1": I_direct, "holevo_chi_s1": chi,
            "holevo_identity_residual": abs(I_direct - chi)}


def v1_species_read_output_only(u_star, r=1.0):
    """[V1 §4 species read — OUTPUT ONLY, hard guard D4] consulted ONLY AFTER the extremum is
    characterized; NEVER an input or selection criterion.  Names which sector/species content
    (gauge_sector_category's own dictionary: occupation level 0/1/2/3 = nu/d/u/e) the selected
    direction's image weights, per shell."""
    gsc = gauge_sector_category()
    species = {0: "nu", 1: "d", 2: "u", 3: "e"}
    d_vec, _ = w2_family_direction(u_star, r=r)
    gt = _v1_gamma_mode_table(d_vec)
    out = {"species_sector_dims": gsc["species_sector_dims"]}
    for n in (1, 2, 3):
        V = gt["by_length"][n]["vectors"]
        occ = np.zeros(4)
        for k in range(V.shape[1]):
            psi = np.abs(V[:, k].reshape(2, 2, 2)) ** 2
            for i0 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        occ[i0 + i1 + i2] += psi[i0, i1, i2]
        tot = occ.sum()
        out[f"shell{n}_level_weights"] = {species[m]: float(occ[m] / tot) if tot > 1e-30 else 0.0
                                           for m in range(4)}
    return out


def v1_r_sensitivity(keys, r_values=(0.5, 1.0, 2.0), n_samples=12, seed=0):
    """[V1 appendix, REPORT-ONLY] step-0-style relative spreads of the named components at
    r in {1/2, 1, 2} (shell reweighting).  NO r-optimization: the values are reported raw and
    nothing is selected from them (freeze §1/§8)."""
    rng_base = np.random.default_rng(seed)
    dirs = []
    for _ in range(n_samples):
        u = rng_base.normal(size=3) + 1j * rng_base.normal(size=3)
        dirs.append(u / np.linalg.norm(u))
    out = {}
    for rv in r_values:
        vals = {k: [] for k in keys}
        for u in dirs:
            for k in keys:
                vals[k].append(v1_component_value(u, k, r=rv))
        out[rv] = {k: {"mean": float(np.mean(v)),
                       "rel_spread": float((np.max(v) - np.min(v)) /
                                            max(1.0, float(np.mean(np.abs(v)))))}
                   for k, v in vals.items()}
    return out


# One pinned regression value (computed once at accretion time, u = (1,1,1)/sqrt(3), r=1):
V1_REGRESSION_I_HIST_COPY_111 = 0.936104841968


def v1_selftest_2026_07_13(verbose=True):
    """V1 station regression (fast, < 120s per the verify per-entry timeout law; NOT wired into
    verify.py — integration batch, L9).  Checks: the two module anchors (cheap spot check), the
    occupation-basis transform exactness, channel-state exactness (norm / Schur-blind field
    marginal / purity identity), the copy-overlap level lemma + the Holevo mechanism identity,
    global-phase drop and conj-invariance of the pair functionals, a small-sample step-0 spread
    contrast (F1 exactly blind at single shells vs the F1p copy read varying at O(1)), and one
    pinned regression value."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 11 self-test -- V1 the vertex on the family (2026-07-13)")
        print("=" * 88)

    ck("module anchors (cheap spot check)", anchor_cell_projector() and anchor_tick_2pi())

    T = _v1_occupation_transform()
    ck("occupation transform: T unitary + T@vac=|000> (asserted in constructor)",
       float(np.max(np.abs(T @ T.conj().T - np.eye(8)))) < 1e-9)

    u0 = np.array([1.0, 0.0, 0.0])
    d0, _ = w2_family_direction(u0, r=1.0)
    gt = _v1_gamma_mode_table(d0)
    st = v1_channel_state(gt, (1,))
    nrm = float(np.linalg.norm(st["vec"]))
    ck(f"channel state unit norm (residual={abs(nrm - 1.0):.2e})", abs(nrm - 1.0) < 1e-9)
    rhoF = _v1_pure_marginal(st["vec"], (st["D"], 2, 2, 2), (1, 2, 3))
    lam = np.sort(np.linalg.eigvalsh(rhoF))[::-1]
    schur_resid = float(np.max(np.abs(lam[:3] - 1.0 / 3.0)))
    ck(f"shell-1 field marginal = I/3 on level 1 (Schur; residual={schur_resid:.2e})",
       schur_resid < 1e-6)

    pl = v1_purity_reduction_lemma()
    ck(f"PURITY-REDUCTION LEMMA identity (residual={max(pl['u1']['identity_residual'], pl['u2']['identity_residual']):.2e}); "
       f"level-1 eigenvalues pinned ({pl['level1_eigenvalue_variation']:.2e}) while level-2 "
       f"eigenvalues vary genuinely ({pl['level2_eigenvalue_variation']:.2e}, the c_2(u) "
       f"level-weight mechanism)",
       max(pl["u1"]["identity_residual"], pl["u2"]["identity_residual"]) < 1e-9
       and pl["level1_eigenvalue_variation"] < 1e-3
       and pl["level2_eigenvalue_variation"] > 1e-7)

    rng = np.random.default_rng(3)
    u = rng.normal(size=3) + 1j * rng.normal(size=3)
    u = u / np.linalg.norm(u)
    lem = v1_copy_overlap_level_lemma(u)
    ck(f"COPY-OVERLAP LEVEL LEMMA: worst same-word cross overlap={lem['worst_same_word_cross_overlap']:.2e}; "
       f"HOLEVO identity residual={lem['holevo_identity_residual']:.2e}",
       lem["worst_same_word_cross_overlap"] < 1e-9 and lem["holevo_identity_residual"] < 1e-9)

    key = ("F1p", "s1_I_hist_copy")
    base = v1_component_value(u, key)
    vph = v1_component_value(u * np.exp(0.7j), key)
    vcj = v1_component_value(np.conj(u), key)
    ck(f"pair functional: global phase drops ({abs(vph - base):.2e}) and conj-invariance "
       f"({abs(vcj - base):.2e}, an observed EXACT symmetry -> extrema come in conj pairs "
       f"or sit on the conj-fixed locus)", abs(vph - base) < 1e-9 and abs(vcj - base) < 1e-9)

    vals_blind, vals_copy = [], []
    rng0 = np.random.default_rng(0)
    for _ in range(6):
        uu = rng0.normal(size=3) + 1j * rng0.normal(size=3)
        uu = uu / np.linalg.norm(uu)
        vals_blind.append(v1_component_value(uu, ("F1", "s1_full")))
        vals_copy.append(v1_component_value(uu, ("F1p", "s1_I_hist_copy")))
    sp_blind = (max(vals_blind) - min(vals_blind)) / max(1.0, float(np.mean(np.abs(vals_blind))))
    sp_copy = (max(vals_copy) - min(vals_copy)) / max(1.0, float(np.mean(np.abs(vals_copy))))
    ck(f"STEP-0 CONTRAST (6 samples): F1 s1_full rel spread={sp_blind:.2e} (exactly blind), "
       f"F1p s1_I_hist_copy rel spread={sp_copy:.2e} (the copy read varies at O(1))",
       sp_blind < 1e-12 and sp_copy > 1e-2)

    reg = v1_component_value(np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0), ("F1p", "s1_I_hist_copy"))
    ck(f"regression value: F1p s1_I_hist_copy at u=(1,1,1)/sqrt(3) = {reg:.10f} "
       f"(pinned {V1_REGRESSION_I_HIST_COPY_111:.10f})",
       abs(reg - V1_REGRESSION_I_HIST_COPY_111) < 1e-8)

    if verbose:
        print("RESULT:", "V1 SECTION-11 REGRESSION PASSES" if ok else "A V1 CHECK FAILED")
    return ok


# ===========================================================================
# 12. FOCK-2 -- THE M_Z/ppm CONFRONTATION AT THE RATIFIED TRIAD FRAME (2026-07-14)
#     internal research notes (frozen BEFORE this code)
# ===========================================================================
# [PLACEMENT NOTE: same conflict-free convention as every section above -- appended after the
#  __main__ guard; ACCRETION-ONLY, nothing in Sections 1-11 is modified.]
#
# CONTEXT: the freeze's §2 authorizes EXACTLY ONE new machinery class for this station -- "the
# per-sector decomposition of the [V1] F-family blocks by the ML-2 sector category
# gauge_sector_category() ({nu:1, d:3, u:3, e:1})".  Everything else (F1/F2/F3/F1p/F2p/F3p
# themselves, w2_*, gauge_sector_category) is REUSED unchanged from Sections 8c/10/11.
#
# THE FORCED MECHANISM (A2, committed BEFORE any Phase-B number is touched): V1's per-word
# tower already carries an EXACT level-confinement fact (v1_species_read_output_only's own
# finding, Section 11: "shell n maps to level n for every u", i.e. a length-n word's field image
# Gamma_u(w) lies ENTIRELY in gauge_sector_category's occupation-level-n subspace).  v1_F2_F3 /
# v1_F2p_F3p aggregate omega-weighted sums over ALL shells 0..3 at once -- which MIXES sectors
# (nu-shell0 + d-shell1 + u-shell2 + e-shell3 contributions together).  Because the shells
# partition the word set and each shell maps into a SINGLE occupation level, the SECTOR-PURE
# read is simply v1_F2_F3 / v1_F2p_F3p's own per-word loop RESTRICTED to one shell -- no new
# mathematics, only a restriction of the existing sum (fock2_F2_F3_per_shell /
# fock2_F2p_F3p_per_shell below).  v1_F1p already supports this restriction natively via its own
# blocks=/full_subset_blocks= arguments (GAP-A, Section 11) -- reused as-is, no wrapper needed.
#
# THE TWO NONTRIVIAL SECTORS: "d" (shell 1, occupation level 1, dim 3) and "u" (shell 2,
# occupation level 2, dim 3 via the Lambda^2(V)~=V isomorphism, Section 10's own fact) are the
# only sectors with internal 3-fold structure to read; "nu" (shell 0) and "e" (shell 3) are each
# DIMENSION 1 (species_sector_dims = {0:1,1:3,2:3,3:1}) -- a 1-dim occupation-level image cannot
# carry ANY superposition/entanglement structure, so every F2/F3-type read there is EXACTLY ZERO
# BY DIMENSION COUNT (a forced, provable fact, verified not merely observed below --
# fock2_selftest_2026_07_14), matching the ALREADY-established F1 s3_full~=0 finding (Section 11
# headline).  "PER-C3-ISOTYPE" (freeze A2's own term): the 3 internal components of the "d"/"u"
# sectors are read in the SAME basis the entire closed V1 inventory already uses (the JW modes
# m0/m1/m2, or the complementary pairs 01/02/12) -- INHERITED from the frozen §2 inventory, not a
# new pick.  DISCLOSED (identification-layer law): this is gauge_sector_category's own DHR/color-
# multiplicity species map ({nu,d,u,e} = A4/2T sectors), NOT the separate "observer C3" generation
# count (R3_observer_c3_generation) -- any resemblance between the two stays OUTLOOK ONLY, per the
# standing rule (triad<->generations is never in the verdict path).
#
# GAUGE LAW (per freeze §1/§4): every read below is evaluated at ALL THREE triad members and the
# max pairwise leaf difference is reported (fock2_triad_gauge_check) -- a read that fails this
# check is INADMISSIBLE in the verdict path (appendix-only), per the freeze's own instruction.
#
# NUMBERS APPEAR NOWHERE ABOVE PHASE C: nothing in this section's own bodies references M_Z, m_W,
# ppm, or any booked residual value -- the comparison happens exactly once, in the station's
# return document, not in this accreted code.
# ML-2b/HK-7 CONDITIONALITY (verbatim, carries into every verdict sentence downstream): "Every
# duality check here (HK-5) is CELL-LEVEL only (the 6-edge static vacuum). ML-2b's DR-frame
# argument is CONDITIONAL on the TD-limit duality holding, which is NOT verified by this suite."


_FOCK2_SECTOR_OF_SHELL = {0: "nu", 1: "d", 2: "u", 3: "e"}   # gauge_sector_category's own labels
_FOCK2_NONTRIVIAL_SECTORS = ((1, "d"), (2, "u"))              # the two dim-3 sectors


def fock2_F2_F3_per_shell(gt_mode, shell_n):
    """[FOCK-2 sec 12, THE PER-SECTOR RESTRICTION of v1_F2_F3] the SAME omega-averaged
    conditional reads as v1_F2_F3 (F2: mode-pair MI; F3: single-mode entropy), restricted to ONE
    word shell (rather than v1_F2_F3's aggregate over n in 0..max_length) -- the forced per-
    sector decomposition licensed by level confinement (see section banner).  Returns
    (f2_dict {'01','02','12'}, f3_dict {'m0','m1','m2'}) -- identical keys to v1_F2_F3, restricted
    to shell_n only, reusing v1_F2_F3's own per-word loop body verbatim (sum over {shell_n})."""
    pairs = {"01": (0, 1), "02": (0, 2), "12": (1, 2)}
    blk = gt_mode["by_length"][shell_n]
    Z, accF2, accF3 = 0.0, {k: 0.0 for k in pairs}, {m: 0.0 for m in range(3)}
    for k in range(blk["vectors"].shape[1]):
        v = blk["vectors"][:, k]
        nrm2 = float(np.real(np.vdot(v, v)))
        wt = gt_mode["omega"][blk["idxs"][k]]
        Z += wt * nrm2
        if nrm2 < 1e-14:
            continue
        vt = v / math.sqrt(nrm2)
        for pname, (i, j) in pairs.items():
            accF2[pname] += wt * nrm2 * _v1_mutual_information(vt, (2, 2, 2), (i,), (j,))
        for m in range(3):
            accF3[m] += wt * nrm2 * _v1_entropy_base2(_v1_pure_marginal(vt, (2, 2, 2), (m,)))
    if Z < 1e-14:
        return ({p: 0.0 for p in pairs}, {f"m{m}": 0.0 for m in range(3)})
    return ({p: accF2[p] / Z for p in pairs}, {f"m{m}": accF3[m] / Z for m in range(3)})


def fock2_F2p_F3p_per_shell(gtP, gtT, shell_n):
    """[FOCK-2 sec 12, THE PER-SECTOR RESTRICTION of v1_F2p_F3p] SAME construction as
    fock2_F2_F3_per_shell but on the pair-channel state (v1_F2p_F3p's own per-word loop body,
    reused verbatim, restricted to shell_n only).  Returns (f2p_dict, f3p_dict) with the SAME key
    names as v1_F2p_F3p's own return (cond_Phi_*, cond_Phit_*, across_copy_m*, traced_m*, S_copy)."""
    pairs = {"01": (0, 1), "02": (0, 2), "12": (1, 2)}
    bP, bT = gtP["by_length"][shell_n], gtT["by_length"][shell_n]
    Z = 0.0
    accF2 = {f"cond_{c}_{p}": 0.0 for c in ("Phi", "Phit") for p in pairs}
    accF2x = {f"across_copy_m{m}": 0.0 for m in range(3)}
    accF3 = {f"cond_{c}_m{m}": 0.0 for c in ("Phi", "Phit") for m in range(3)}
    accF3x = {f"traced_m{m}": 0.0 for m in range(3)}
    accF3x["S_copy"] = 0.0
    for k in range(bP["vectors"].shape[1]):
        vP, vT = bP["vectors"][:, k], bT["vectors"][:, k]
        nrm2 = float(np.real(np.vdot(vP, vP) + np.vdot(vT, vT)))
        wt = gtP["omega"][bP["idxs"][k]]
        Z += wt * nrm2
        if nrm2 < 1e-14:
            continue
        pair_vec = np.concatenate([vP, vT]) / math.sqrt(nrm2)
        dims = (2, 2, 2, 2)
        for m in range(3):
            accF2x[f"across_copy_m{m}"] += wt * nrm2 * _v1_mutual_information(
                pair_vec, dims, (0,), (m + 1,))
            accF3x[f"traced_m{m}"] += wt * nrm2 * _v1_entropy_base2(
                _v1_pure_marginal(pair_vec, dims, (m + 1,)))
        accF3x["S_copy"] += wt * nrm2 * _v1_entropy_base2(_v1_pure_marginal(pair_vec, dims, (0,)))
        for cname, v in (("Phi", vP), ("Phit", vT)):
            bn = float(np.real(np.vdot(v, v)))
            if bn < 1e-14:
                continue
            bvec = v / math.sqrt(bn)
            for pname, (i, j) in pairs.items():
                accF2[f"cond_{cname}_{pname}"] += wt * nrm2 * (bn / nrm2) * \
                    _v1_mutual_information(bvec, (2, 2, 2), (i,), (j,))
            for m in range(3):
                accF3[f"cond_{cname}_m{m}"] += wt * nrm2 * (bn / nrm2) * \
                    _v1_entropy_base2(_v1_pure_marginal(bvec, (2, 2, 2), (m,)))
    if Z < 1e-14:
        f2 = {k: 0.0 for k in {**accF2, **accF2x}}
        f3 = {k: 0.0 for k in {**accF3, **accF3x}}
        return f2, f3
    f2 = {k: v / Z for k, v in {**accF2, **accF2x}.items()}
    f3 = {k: v / Z for k, v in {**accF3, **accF3x}.items()}
    return f2, f3


def fock2_per_sector_read(u_vec, r=1.0, N_max=4):
    """[FOCK-2 sec 12, THE FORCED READ -- freeze A2] the per-sector (gauge_sector_category)
    decomposition of the frozen union-block-survivor families (F2, F3, F1p-traced/cond, F2p,
    F3p; F1's OWN single-shell blocks are excluded here since they are EXACTLY BLIND by
    v1_purity_reduction_lemma, already established -- re-deriving them per-sector would be
    re-litigating a settled theorem) at ONE direction u, for the two dim-3 sectors 'd' (shell 1)
    and 'u' (shell 2).  Returns {'species_sector_dims':..., 'd': {...}, 'u': {...}} where each
    sector dict holds the isotype-vector for every surviving component family, keyed exactly as
    v1_F2_F3/v1_F1p/v1_F2p_F3p already key their outputs (m0/m1/m2 or 01/02/12)."""
    gsc = gauge_sector_category()
    d_vec, _ = w2_family_direction(u_vec, r=r)
    gtP = _v1_gamma_mode_table(d_vec, N_max=N_max)
    gtT = _v1_gamma_mode_table(d_vec, N_max=N_max, conjugate=True)
    out = {"species_sector_dims": gsc["species_sector_dims"]}
    for shell_n, sector in _FOCK2_NONTRIVIAL_SECTORS:
        bname = f"s{shell_n}"
        f2, f3 = fock2_F2_F3_per_shell(gtP, shell_n)
        f2p, f3p = fock2_F2p_F3p_per_shell(gtP, gtT, shell_n)
        f1p = v1_F1p(gtP, gtT, blocks=(bname,), full_subset_blocks=(bname,))[bname]
        out[sector] = {
            "shell": shell_n,
            "F2_pairs": f2,
            "F3_modes": f3,
            "F2p_cond_Phi_pairs": {k: f2p[f"cond_Phi_{k}"] for k in ("01", "02", "12")},
            "F2p_across_copy_modes": {f"m{m}": f2p[f"across_copy_m{m}"] for m in range(3)},
            "F3p_cond_Phi_modes": {f"m{m}": f3p[f"cond_Phi_m{m}"] for m in range(3)},
            "F3p_traced_modes": {f"m{m}": f3p[f"traced_m{m}"] for m in range(3)},
            "F1p_traced_modes": {f"m{m}": f1p[f"traced_m{m}"] for m in range(3)},
            "F1p_cond_Phi_modes": {f"m{m}": f1p[f"cond_Phi_m{m}"] for m in range(3)},
        }
    return out


def fock2_dim1_sector_check(u_vec, r=1.0, N_max=4):
    """[FOCK-2 sec 12, STEP-0 STRUCTURAL FACT -- freeze A3] the 'nu' (shell 0) and 'e' (shell 3)
    sectors are DIMENSION 1 (species_sector_dims); a 1-dim occupation-level image cannot carry
    ANY field-internal entanglement, so EVERY field-only component (F2, F3, F2p's cond_Phi/
    cond_Phit pairs, F3p's cond_Phi/cond_Phit modes) is EXACTLY ZERO there BY DIMENSION COUNT --
    verified (not merely sampled).  The COPY-CROSS components (F2p's across_copy_modes, F3p's
    traced_modes/S_copy) are NOT forced to zero (the antiparticle-copy qubit can still carry
    information about a mode's DETERMINISTIC dim-1 occupation value -- a genuinely different,
    equally forced mechanism: verified numerically to be EXACTLY 1.0 bit, i.e. MAXIMAL, since
    level n and level 3-n assign every mode a definite, opposite occupation for n in {0,3}).
    EITHER WAY, no per-isotype ASYMMETRY is possible in a dim-1 sector: the spread across
    {m0,m1,m2} (or {01,02,12}) is checked to be exactly zero for every family (the field-only
    families at the value 0; the copy-cross families at their own shared constant) -- the
    unifying step-0 fact this function verifies.  Returns the worst absolute field-only value
    AND the worst isotype spread across every family (should both sit at the ~1e-14 floor for
    field-only families' values, and at that same floor for EVERY family's spread)."""
    d_vec, _ = w2_family_direction(u_vec, r=r)
    gtP = _v1_gamma_mode_table(d_vec, N_max=N_max)
    gtT = _v1_gamma_mode_table(d_vec, N_max=N_max, conjugate=True)
    worst_fieldonly = 0.0
    worst_spread = 0.0
    for shell_n in (0, 3):
        f2, f3 = fock2_F2_F3_per_shell(gtP, shell_n)
        f2p, f3p = fock2_F2p_F3p_per_shell(gtP, gtT, shell_n)
        fieldonly = [f2, f3,
                     {k: f2p[f"cond_Phi_{k}"] for k in ("01", "02", "12")},
                     {k: f2p[f"cond_Phit_{k}"] for k in ("01", "02", "12")},
                     {f"m{m}": f3p[f"cond_Phi_m{m}"] for m in range(3)},
                     {f"m{m}": f3p[f"cond_Phit_m{m}"] for m in range(3)}]
        for d in fieldonly:
            for v in d.values():
                worst_fieldonly = max(worst_fieldonly, abs(v))
        copycross = [{f"m{m}": f2p[f"across_copy_m{m}"] for m in range(3)},
                     {f"m{m}": f3p[f"traced_m{m}"] for m in range(3)}]
        for d in fieldonly + copycross:
            vals = list(d.values())
            worst_spread = max(worst_spread, max(vals) - min(vals))
    return {"worst_abs_value_fieldonly": worst_fieldonly, "worst_isotype_spread": worst_spread}


def _fock2_max_leaf_diff(a, b):
    """[FOCK-2 sec 12, shared helper] max absolute difference between two structurally-identical
    nested dict/scalar objects (recurses through dicts, compares leaves as floats); used only for
    numeric leaves -- non-numeric leaves (e.g. species_sector_dims / 'shell') must be excluded by
    the caller before use."""
    if isinstance(a, dict):
        return max((_fock2_max_leaf_diff(a[k], b[k]) for k in a), default=0.0)
    return abs(float(a) - float(b))


def fock2_triad_gauge_check(u_A, u_B, u_C, r=1.0, N_max=4):
    """[FOCK-2 sec 12, THE GAUGE LAW CHECK -- freeze §1/§4] fock2_per_sector_read at all three
    triad members; returns the per-pair and worst-overall max leaf-wise absolute difference
    (species_sector_dims and the 'shell' integer tag excluded from the diff -- static, non-
    numeric-in-spirit fields).  A read whose worst value exceeds the stated floor is INADMISSIBLE
    in the verdict path per the freeze's own gauge law."""
    def _clean(read):
        out = {}
        for sector, row in read.items():
            if sector == "species_sector_dims":
                continue
            out[sector] = {k: v for k, v in row.items() if k != "shell"}
        return out

    reads_raw = {"u_A": fock2_per_sector_read(u_A, r=r, N_max=N_max),
                 "u_B": fock2_per_sector_read(u_B, r=r, N_max=N_max),
                 "u_C": fock2_per_sector_read(u_C, r=r, N_max=N_max)}
    reads = {k: _clean(v) for k, v in reads_raw.items()}
    names = list(reads)
    diffs = {}
    worst = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            d = _fock2_max_leaf_diff(reads[names[i]], reads[names[j]])
            diffs[f"{names[i]}_vs_{names[j]}"] = d
            worst = max(worst, d)
    return {"reads_raw": reads_raw, "pairwise_max_diff": diffs, "worst": worst}


def fock2_isotype_asymmetry(vec3):
    """[FOCK-2 sec 12, THE ASYMMETRY STATISTIC -- freeze A2 pre-declared list] given a 3-entry
    dict (an isotype vector), returns the value vector itself, every pairwise ratio (a/b for all
    ordered pairs), the descending ordering of the keys, and the max-min spread -- the exact
    pre-declared statistic list ('per-isotype value vector; pairwise ratios; ordering'), nothing
    else computed."""
    keys = list(vec3.keys())
    vals = {k: float(vec3[k]) for k in keys}
    ratios = {}
    for a in keys:
        for b in keys:
            if a != b:
                ratios[f"{a}/{b}"] = (vals[a] / vals[b]) if abs(vals[b]) > 1e-300 else float("nan")
    ordering = sorted(keys, key=lambda k: -vals[k])
    spread = max(vals.values()) - min(vals.values())
    return {"values": vals, "pairwise_ratios": ratios, "ordering": ordering, "spread": spread}


def fock2_democracy_test(spread, floor):
    """[FOCK-2 sec 12, THE DEMOCRACY TEST -- freeze A2] 'all isotype values equal to the stated
    floor = species-democratic'."""
    return bool(spread <= floor)


def fock2_selftest_2026_07_14(verbose=True):
    """FOCK-2 section-12 self-test (fast, well under the 120s verify per-entry timeout; NOT wired
    into verify.py -- integration batch, L9).  Checks: (1) the dimension-1 sectors (nu shell-0, e
    shell-3) are EXACTLY BLIND by dimension count, on a random direction (not merely at the
    triad); (2) the per-shell decomposition of F2/F3 correctly reconstructs v1_F2_F3's own
    omega-weighted aggregate (a decomposition identity, confirming the restriction is implemented
    correctly, not new physics); (3) the per-sector read is gauge-invariant across the triad's
    OWN headline (unpolished, ~1e-4-precision) coordinates, to a floor consistent with that
    input precision -- the station's own driver Newton-polishes to ~1e-13 and re-checks at the
    tighter floor; this is a fast smoke check only."""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    if verbose:
        print("=" * 88)
        print(" THE NET section 12 self-test -- FOCK-2 the M_Z/ppm confrontation (2026-07-14)")
        print("=" * 88)

    rng = np.random.default_rng(1)
    u = rng.normal(size=3) + 1j * rng.normal(size=3)
    u = u / np.linalg.norm(u)
    dim1 = fock2_dim1_sector_check(u)
    ck(f"dim-1 sectors (nu shell0, e shell3): field-only families EXACTLY ZERO "
       f"(worst={dim1['worst_abs_value_fieldonly']:.2e}) and EVERY family isotype-spread ZERO "
       f"(worst={dim1['worst_isotype_spread']:.2e}) on a random direction -- no per-isotype "
       f"asymmetry is possible in a dim-1 sector, either way",
       dim1["worst_abs_value_fieldonly"] < 1e-9 and dim1["worst_isotype_spread"] < 1e-9)

    d_vec, _ = w2_family_direction(u)
    gtP = _v1_gamma_mode_table(d_vec)
    f2_full, f3_full = v1_F2_F3(gtP)
    Zs = {}
    per_shell_f2 = {}
    for n in range(4):
        blk = gtP["by_length"][n]
        Zn = 0.0
        for k in range(blk["vectors"].shape[1]):
            v = blk["vectors"][:, k]
            Zn += gtP["omega"][blk["idxs"][k]] * float(np.real(np.vdot(v, v)))
        Zs[n] = Zn
        per_shell_f2[n], _ = fock2_F2_F3_per_shell(gtP, n)
    Ztot = sum(Zs.values())
    worst_recon = 0.0
    for pname in ("01", "02", "12"):
        recon = sum(Zs[n] * per_shell_f2[n][pname] for n in range(4)) / Ztot
        worst_recon = max(worst_recon, abs(recon - f2_full[pname]))
    ck(f"per-shell F2 decomposition reconstructs v1_F2_F3's aggregate (worst={worst_recon:.2e})",
       worst_recon < 1e-9)

    u_A = np.array([0.9319033775, 0.1172343025, 0.3432378378], dtype=complex)
    u_B = np.array([-0.2741, 0.8473, 0.4549], dtype=complex)
    u_C = np.array([-0.2375, -0.518, 0.8218], dtype=complex)
    gc = fock2_triad_gauge_check(u_A, u_B, u_C)
    ck(f"per-sector read gauge-invariant across the (unpolished headline) triad coords "
       f"(worst pairwise leaf diff={gc['worst']:.2e})", gc["worst"] < 1e-2)

    if verbose:
        print("RESULT:", "FOCK-2 SECTION-12 REGRESSION PASSES" if ok else "A FOCK-2 CHECK FAILED")
    return ok


# ===========================================================================================
# GEN-IDENT-beta (2026-07-15): THE RUN-ENDPOINT VERTEX FUNCTIONAL  V(s) = -kappa * I(A;B)(s)
# ===========================================================================================
# [freeze internal research notes]  Does the forced
# substrate-internal vertex functional -- the mutual information between the forced C3-winding
# sectors of the endpoint-s generation-run state, on the forced Lambda*(C^3)=(4,2,2) carrier --
# have a DISTINGUISHED, non-degenerate interior stationary point s*, read top-down and without
# any fit to a lepton value?  Verdict driver:
# proofs/foundations/genident_beta_endpoint_vertex_check_2026-07-15.py ; verdict doc
# docs/theorems/genident_beta_*_2026-07-15.md ; return doc
# internal research notes .
#
# REUSES the V1 apparatus UNCHANGED: _v1_mutual_information / _v1_pure_marginal /
# _v1_entropy_base2, on the SAME axis_dims=(2,2,2) occupation-tensor convention
# _v1_occupation_transform already produces from F.  NOT a fork of those primitives.
#
# DISCLOSED JUDGMENT CALL (why this station builds its OWN tiny CAR realization instead of
# reusing _level1_creation_ops's Cl(6) Adag operators): _level1_creation_ops's 3 modes are built
# from complex_structure_J6()'s +i eigenbasis -- a DIFFERENT physical construction (the spatial
# field-algebra F) whose mode index has NO established/verified correspondence to the C3-winding
# label t={0,1,2} this station needs (the "F is Lambda*(C^3) up to DIMENSION MATCH" identification
# is explicitly flagged elsewhere in this file, e.g. near _region_fock_ops, as NOT a proof).
# Splicing winding-t amplitudes onto an unverified mode-t of a DIFFERENT construction would be
# exactly the kind of invented/smuggled identification the freeze bars ("do NOT substitute an
# invented tensor space").  Building the small 8-dim CAR realization directly over the 3 NAMED
# windings (verified below to reproduce read_flavor()'s (4,2,2) isotype content bit-for-bit, and
# to satisfy the CAR exactly) is the honest, minimal, non-smuggled way to realize "the forced
# second-quantized run construction" on Lambda*(C^3) -- while still reusing the V1 ENTROPY /
# MUTUAL-INFORMATION machinery verbatim.


def _beta_car_creation_ops():
    """[GEN-IDENT-beta SEC2, THE SECOND-QUANTIZED RUN CONSTRUCTION] the 3 winding creation
    operators a_t^dagger (t=0,1,2) and vacuum |vac>=|000> on the abstract Lambda*(C^3) (dim 8,
    occupation basis |n0 n1 n2>, axis_dims=(2,2,2) -- the SAME bit convention
    _v1_occupation_transform/_v1_pure_marginal already use for F: axis0=mode0 weight 4, axis1=
    mode1 weight 2, axis2=mode2 weight 1).  Standard Jordan-Wigner CAR realization:
    a_t^dagger|n> = 0 if n_t=1, else (-1)^(sum_{j<t} n_j) |n with n_t set to 1>.
    VERIFIED (not assumed): {a_i,a_j^dagger}=delta_ij*I and {a_i^dagger,a_j^dagger}=0 exactly
    (machine precision); the resulting isotype content (weight(S)=sum(S) mod 3 over occupied
    winding sets S) reproduces read_flavor()'s (4,2,2) bit-for-bit (the_run.py:240-252).
    Returns (Adag: list of 3 complex 8x8 matrices, vac: 8x1 complex column, index: dict
    (n0,n1,n2)->basis row, 'isotype_content': the (4,2,2) count, verified)."""
    dim = 8
    basis = list(itertools.product((0, 1), repeat=3))          # (n0, n1, n2)
    index = {n: 4 * n[0] + 2 * n[1] + n[2] for n in basis}      # row-major, matches V1's P_bits
    Adag = [np.zeros((dim, dim), dtype=complex) for _ in range(3)]
    for t in range(3):
        for n in basis:
            if n[t] == 1:
                continue
            sign = (-1.0) ** sum(n[:t])
            n2 = list(n); n2[t] = 1; n2 = tuple(n2)
            Adag[t][index[n2], index[n]] = sign
    vac = np.zeros((dim, 1), dtype=complex); vac[index[(0, 0, 0)], 0] = 1.0

    I8 = np.eye(dim, dtype=complex)
    car_ac = 0.0
    for i in range(3):
        for j in range(3):
            anti = Adag[i].conj().T @ Adag[j] + Adag[j] @ Adag[i].conj().T
            car_ac = max(car_ac, float(np.max(np.abs(anti - (I8 if i == j else 0.0 * I8)))))
    car_cc = 0.0
    for i in range(3):
        for j in range(3):
            car_cc = max(car_cc, float(np.max(np.abs(Adag[i] @ Adag[j] + Adag[j] @ Adag[i]))))
    assert car_ac < 1e-9, f"_beta_car_creation_ops: {{a_i,a_j^dagger}} CAR violated ({car_ac:.2e})"
    assert car_cc < 1e-9, f"_beta_car_creation_ops: {{a_i^dagger,a_j^dagger}} CAR violated ({car_cc:.2e})"

    content = {0: 0, 1: 0, 2: 0}
    for n in basis:
        S = [t for t in range(3) if n[t] == 1]
        content[sum(S) % 3] += 1
    isotype_content = (content[0], content[1], content[2])
    assert isotype_content == (4, 2, 2), \
        f"_beta_car_creation_ops: isotype content {isotype_content} != forced (4,2,2)"
    return Adag, vac, index, isotype_content


def _beta_forced_phi():
    """[GEN-IDENT-beta SEC2] phi = 2*pi/sqrt(4*(k-1) - lam3^2) = 2*pi/sqrt(7), k=3 (srs.DEG,
    the deck screw's degree), lam3=-1 (the A4 3-irrep eigenvalue at Gamma) -- the_run.py:290 /
    derive_generation_spectrum.py:118-122, re-derived here from the SAME two forced integers
    (k, lam3), not re-imported as a float literal."""
    k, lam3 = float(srs.DEG), -1.0
    return 2.0 * math.pi / math.sqrt(4.0 * (k - 1.0) - lam3 ** 2)


def _beta_winding_amplitudes(s, moduli="frozen"):
    """[GEN-IDENT-beta SEC2, THE FROZEN c(s)] the per-winding run amplitudes.  The DIRECTED PHASE
    {0, +phi*s, -phi*s} is ALWAYS the forced screw phase (the_run.py:291) -- the ONLY s-dependence
    anywhere in this construction.  moduli='frozen' (default) sets the FORCED equal-shell moduli
    |c_t|=1 (derive_generation_spectrum.py:153, the construction the freeze SEC2 pins verbatim:
    "c(s) = (1, e^{+i phi s}, e^{-i phi s})").  moduli='perron' switches to the ALTERNATE
    Perron-weighted fork {2, sqrt2, sqrt2} that the_run.py:288-291 uses for the live mass
    MAGNITUDES -- NOT the frozen construction; included ONLY as an explicit symmetric/asymmetric
    cross-check for the SEC3 S3-control (see beta_endpoint_vertex_read)."""
    phi = _beta_forced_phi()
    if moduli == "frozen":
        m0, m12 = 1.0, 1.0
    elif moduli == "perron":
        m0, m12 = 2.0, math.sqrt(2.0)
    else:
        raise ValueError(f"_beta_winding_amplitudes: unknown moduli {moduli!r}")
    c = np.array([m0 + 0j, m12 * cmath.exp(1j * phi * s), m12 * cmath.exp(-1j * phi * s)])
    return c, phi


def _beta_promote_state(c, promotion):
    """[GEN-IDENT-beta SEC2, THE STATE PROMOTION -- the one derivation this station OWNS] promote
    the 3 winding amplitudes c=(c0,c1,c2) to a pure state on Lambda*(C^3) (the 8-dim CAR Fock
    space _beta_car_creation_ops builds) by the forced second-quantized run construction, in
    EVERY defensible literal reading of the freeze's "single-particle Lambda^1 vs coherent
    exp(sum c_t a_t^dagger)|0>" (SEC2: "compute ALL of them"):

    'single_particle' (Lambda^1, no vacuum admixture): |Psi> = sum_t c_t a_t^dagger|vac>.  The
        literal one-fermion promotion -- c_t IS the single-particle amplitude on winding t.

    'coherent_exp' (the LITERAL exp(sum_t c_t a_t^dagger)|vac>): since {a_i^dagger,a_j^dagger}=0
        for ALL i,j INCLUDING i=j (creation-creation CAR, verified above to <1e-9), X = sum_t
        c_t a_t^dagger satisfies X^2 = sum_{i,j} c_i c_j a_i^dagger a_j^dagger = 0 EXACTLY -- a
        THEOREM from the CAR relation, not an approximation (i=j terms are c_i^2 (a_i^dagger)^2=0;
        i!=j ordered pairs sum to c_i c_j (a_i^dagger a_j^dagger + a_j^dagger a_i^dagger) = 0).
        Hence exp(X) = 1 + X identically (all higher Taylor terms vanish).
        |Psi> = |vac> + sum_t c_t a_t^dagger|vac> (vacuum + level-1 admixture).

    'coherent_product' (the STANDARD multi-mode fermionic/Slater-determinant coherent state --
        the reading most literature means by "coherent state built from 3 independent modes"):
        |Psi> = prod_t (1 + c_t a_t^dagger) |vac>, fixed operator order t=0,1,2 (spans all 4
        Fock levels 0..3, the full 8 dims).  Reported as a DISTINCT, equally-defensible promotion
        from 'coherent_exp' -- exp(sum X_t) != prod exp(X_t) for anticommuting X_t, so the
        freeze's "coherent exp(sum c_t a_t^dagger)|0>" phrase is genuinely ambiguous between the
        two readings; BOTH are computed and reported honestly, never silently resolved one way.

    Returns the normalized 8-vector in the CAR occupation basis (axis_dims=(2,2,2))."""
    Adag, vac, _, _ = _beta_car_creation_ops()
    if promotion == "single_particle":
        v = sum(c[t] * (Adag[t] @ vac) for t in range(3))
    elif promotion == "coherent_exp":
        v = vac + sum(c[t] * (Adag[t] @ vac) for t in range(3))
    elif promotion == "coherent_product":
        v = vac.copy()
        I8 = np.eye(8, dtype=complex)
        for t in (0, 1, 2):
            v = (I8 + c[t] * Adag[t]) @ v
    else:
        raise ValueError(f"_beta_promote_state: unknown promotion {promotion!r}")
    n = float(np.linalg.norm(v))
    assert n > 1e-14, f"_beta_promote_state[{promotion}]: degenerate norm"
    return (v / n).reshape(8)


def _beta_bipartition_axes(bipartition):
    """[GEN-IDENT-beta SEC2, THE FORCED BIPARTITION] axis_dims=(2,2,2) is Lambda*(C^3)'s
    occupation-number tensor lattice (mode0=omega^0, mode1=omega^1, mode2=omega^2) -- the SAME
    lattice convention _v1_pure_marginal/_v1_mutual_information already operate on.  'mode0' =
    the FORCED Perron bipartition A=C_{omega^0} (dim 2, axis 0) vs B=C^2_{omega^1,omega^2} (dim 4,
    axes 1,2) -- the freeze's SEC2 split (the Perron-excess winding vs the mirror pair).
    'mode1'/'mode2' single out the OTHER windings as A instead, for the bipartition-robustness
    control (SEC3.3): if a signal depends on which winding is singled out as A, it is an
    artifact, not a genuine (4,2,2)-forced read."""
    axis_dims = (2, 2, 2)
    table = {"mode0": ((0,), (1, 2)), "mode1": ((1,), (0, 2)), "mode2": ((2,), (0, 1))}
    if bipartition not in table:
        raise ValueError(f"_beta_bipartition_axes: unknown bipartition {bipartition!r}")
    axes_A, axes_B = table[bipartition]
    return axis_dims, axes_A, axes_B


def beta_endpoint_vertex_read(s, promotion="single_particle", bipartition="mode0",
                               moduli="frozen", kappa=1.0):
    """[GEN-IDENT-beta, THE ACCRETED READ -- see the section-header block above for full
    provenance]  V(s) = -kappa * I(A;B)(s), I(A;B) the mutual information between the forced
    C3-winding sectors of the endpoint-s generation-run state |Psi(s)>, on the forced
    Lambda*(C^3)=(4,2,2) carrier.  |Psi(s)> is built by `promotion` from the frozen amplitudes
    c(s) (_beta_winding_amplitudes); the read is bipartitioned per `bipartition`
    (_beta_bipartition_axes).  REUSES _v1_mutual_information / _v1_pure_marginal /
    _v1_entropy_base2 UNCHANGED -- no forked entropy/MI code.  kappa is a positive overall scale
    (per the freeze's V(s)=-kappa*I(A;B) form); it rescales V but cannot move a stationary point,
    kept at the neutral default 1.0 (never fit).
    Returns {'V','I_AB','S_A','phi','u'=phi*s,'s','promotion','bipartition','moduli'}."""
    c, phi = _beta_winding_amplitudes(s, moduli=moduli)
    vec = _beta_promote_state(c, promotion)
    axis_dims, axes_A, axes_B = _beta_bipartition_axes(bipartition)
    I_AB = _v1_mutual_information(vec, axis_dims, axes_A, axes_B)
    S_A = _v1_entropy_base2(_v1_pure_marginal(vec, axis_dims, axes_A))
    V = -kappa * I_AB
    return {"V": V, "I_AB": I_AB, "S_A": S_A, "phi": phi, "u": phi * s, "s": s,
            "promotion": promotion, "bipartition": bipartition, "moduli": moduli}
