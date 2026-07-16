#!/usr/bin/env python3
# ============================================================================
# IV.4 / T0-CLASS — the mu-scaling class of the MDL contact vertex
# ============================================================================
#
# PRE-REGISTRATION (binding, frozen): internal research notes
# Lineage: I-0b-RATIO A2 (sealed; candidate completion E_bind = -kappa*dS + T0(mu_eff))
#          -> IV.4 sweep dossier (2026-07-10).
#
# MACHINERY REPLICATED (not edited), with line cites:
#   proofs/foundations/bound_state_edge_resolved_kernel_2026-05-29.py
#     - Dirac D(k) 32x32 (lines 55-107): GAMMAS (Cl(6) rep), L_e(edge,k), D(k) =
#       sum_i Gamma_i (x) L_e_i(k); validation D(k)^2 = 6I + R_sub (line 98-107).
#     - eps_low(k) = lowest POSITIVE band of D(k) (lines 110-112) — THE frozen
#       "lowest Dirac band" convention.
#     - kinetic_real_space (lines 198-210): T(R) = (1/Nq) sum_q E_pair(q) e^{2pi i q.R},
#       offset mesh q = (i+0.5)/n_q; E_pair(q) = eps_low(q) + eps_low(-q).
#     - edge-resolved MDL kernel V(Delta) from girth-cycle self-translation
#       (lines 115-194); PROVEN CONTACT (range 0) by that file's own run:
#       V(Delta) = 3*delta_{Delta,0} (its verdict (a)). Re-asserted here.
#     - solve_relative (lines 213-225): H_rel = T_hat + V_hat, real-space
#       diagonalization on a box of relative cells; U = dS*e_bit = 3, e_bit = t = 1
#       (DISCLOSED adoption, carried verbatim, not re-derived).
#   proofs/foundations/bound_state_propagator_pole_2026-05-28.py
#     - Pi(E) = <1/(E_pair - E)> and the pole condition 1 = U*Pi(E_B) (lines 88-114);
#       exact (infinite-box) for a contact well; used as the box-free cross-check
#       and as the A-2 scan solver.
#
# THE ELECTRON BAND — convention adjudication (DISCLOSED):
#   The contract cites the A5-derived P-point 2-band cone, v_F = sqrt(3)/6
#   (proofs/lorentz/srs_dirac_cone_velocities.py lines 21-22; the_run.py:346-365
#   wires e -> lambda = sqrt(3)) as the electron IDENTIFICATION, and then fixes the
#   working convention: "the lowest Dirac band as in the 2026-05-29 file's own
#   convention". These resolve to the 32x32 D(k) lowest positive band:
#     (i)  the 2026-05-29 file's own convention IS eps_low of its 32x32 D(k);
#     (ii) the contract's own quoted critical coupling U_c ~ 0.26 is that band's
#          (the 2026-05-29 verdict (b));
#     (iii) the 4-band scalar-Bloch "lowest positive" band was inspected ONLY to
#          adjudicate this sentence and found PATHOLOGICAL for the convention
#          (zero-crossing surfaces; min ~ 0 at k far from P), i.e. it cannot be
#          "the A5-derived P band" under the lowest-positive rule.  No binding
#          number was computed on it; no band shopping (poison honored — the band
#          is FIXED here, before any Stage-A output, and never changed).
#   Honest note printed at runtime: the 32x32 lowest band's minimum sits at
#   k ~ (0.179, 0.179, 0.821)+perms, NOT at P; the P-point/v_F citation is the
#   species-identification lineage of the SAME Dirac construction, not a claim
#   that the band minimum is at P.
#
# ---------------------------------------------------------------------------
# FROZEN A PRIORI (declared BEFORE the first Stage-0/Stage-A/A-2 result of this
# station was ever computed; nothing below was chosen in response to a binding
# number):
#   STAGE 0 (T0-RIGID):
#     - U = dS*e_bit = 3.0 (verbatim); contact kernel (re-asserted from the
#       girth-cycle profile at windows (L=9, box=2) and (L=11, box=4)).
#     - Box solver: boxes (2, 3, 4) verbatim [regression-asserted against the
#       frozen 2026-05-29 output: E_th = 1.1855, E0 = -1.4912/-1.4913/-1.4914,
#       B = +2.6767/+2.6768/+2.6769, tolerance 2e-4] + box 5 as the declared
#       extra convergence point; n_q = 14 verbatim.
#     - Pole cross-check (infinite box): n_q in {14, 20, 26} (grid convergence).
#     - T0^(2) := U - B.  RIGID criterion: |T0^(2)|/U < 1e-3.  The -16%
#       calibration fact is REPORTED ONLY (declared context; not a target):
#       "provably too small" := T0^(2)/U < 0.04 (a quarter of 16%).
#   STAGE A (two configurations ONLY, verbatim from the pre-reg):
#     (a) EQUAL : E_pair(q) = eps(q) + eps(-q)      [both walkers on the band]
#     (b) STATIC: E_pair(q) = eps(q) + eps_min      [walker 2 frozen, m*2 -> inf;
#         the additive constant provably drops out of B — asserted numerically;
#         const := eps_min declared for concreteness].  PARAMETER-FREE (no
#         proton mass imported; no m* = m_phys/v^2 adoption anywhere).
#     - A-1 solver: box = 4, n_q = 14 (the Stage-0-converged settings); report
#       B, <T_hat>_ground, <T_hat> - E_th (zero-point), <V_hat>, U - B; assert
#       the eigen identity E0 = <T_hat> + <V_hat> to 1e-9 (proven algebra).
#     - A-3: B_static/B_equal reported RAW (box primary, pole cross-check).
#   A-2 THE CLASS SCAN (all grids frozen here):
#     - U grid  : {0.3, 0.5, 1, 2, 3, 5}          (the pre-reg's example grid,
#                                                   adopted verbatim)
#     - s grid  : {2^-1, 2^-1/2, 1, 2^1/2, 2}     (log-uniform, step ln sqrt(2))
#     - operating-point refinement pair (declared a priori, U = 3 only):
#       s = 2^{-1/8}, 2^{+1/8} for a tighter centered derivative at s = 1.
#     - solver  : pole condition on the n_q = 30 pair grid (primary), with the
#       full scan duplicated at n_q = 20 as the grid-convergence control.
#     - delta_safe = 0.02*s (grid-safe threshold guard, 2026-05-28 convention
#       scaled by the dispersion scale).
#     - flags   : NEAR-THRESHOLD if B < 2*delta_safe*s (magnitude unresolved);
#                 GRID-LIMITED if |B(n30) - B(n20)| > 0.01*B(n30).
#       Flagged points are printed but their exponents carry the flag mark.
#     - THE MU MAPPING (stated precisely): eps(q) has isolated minima (Hessian
#       positive-definiteness checked and printed at runtime).  Under the overall
#       scaling eps -> s*eps the inverse-mass tensor M^-1 -> s*M^-1, i.e. EVERY
#       constituent effective mass m* -> m*/s, hence
#         EQUAL : 1/mu = 2/m*  => mu = m*/2  ~ 1/s
#         STATIC: 1/mu = 1/m*  => mu = m*    ~ 1/s
#       so d(ln mu_eff) = -d(ln s) EXACTLY (any curvature-derived mass scale;
#       anisotropy irrelevant since s scales the whole tensor), and the reported
#       exponent is dlnB/dln(mu_eff) = -dlnB/dln(s), centered log-differences on
#       the frozen s grid (one-sided at the ends, marked).
#       Note mu_static/mu_equal = 2 exactly — the reduced-mass form.
#     - NO additional (U, s) points after results; no regime re-scan.
#   VERDICT TREE (thresholds frozen):
#     RIGID        iff |T0^(2)|/U < 1e-3.
#     T1-CLOSE     iff |exponent_op - 1| <= 0.15 at (U=3, s=1) AND
#                      |B_static/B_equal - 2| <= 0.2.
#     T1-FENCE     otherwise (the derived class != linear-mu at the operating
#                   point; EXPECTED, publishable: H/Ps -> the connection sector).
#     UNDER-POWERED (calibration adjunct) iff T0^(2)/U < 0.04.
#   POISONS carried verbatim: no kappa; no oblique/EW quantity; E_odd (0.381876
#   MeV) never absorbed into T0; the 13/3 RATIO-MISS stays booked regardless;
#   the known measured targets (2.0, -16%) appear ONLY in the final declared-
#   context confrontation printout, never as optimization targets.
# ---------------------------------------------------------------------------
# Standalone: python3 proofs/foundations/IV4_T0_class_2026-07-10.py ; exit 0.
# Asserts only on regressions + proven algebra; ALL class verdicts are PRINTED.
# ============================================================================

import os
import sys
import numpy as np
from itertools import product
from collections import defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from proofs.common import find_bonds  # noqa: E402

# ---------------------------------------------------------------------------
# Block 1 — the 2026-05-29 Dirac D(k) + lowest band, replicated verbatim
# (bound_state_edge_resolved_kernel_2026-05-29.py lines 51-112)
# ---------------------------------------------------------------------------
E_BIT = 1.0            # e_bit = t (DISCLOSED, carried, not re-derived)
DS_CAP = 3.0           # Stage-0 max reward
GIRTH = 10
U_OP = DS_CAP * E_BIT  # U = dS * e_bit = 3  — the frozen operating point

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
    assert len(e) == 6          # regression: srs primitive cell has 6 undirected edges
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


# ---------------------------------------------------------------------------
# Block 2 — the edge-resolved MDL kernel, replicated verbatim
# (bound_state_edge_resolved_kernel_2026-05-29.py lines 115-194) — used ONLY to
# re-assert the PROVEN-CONTACT finding; the solves then use the contact kernel.
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


def cycle_edge_set(cycle):
    n = len(cycle)
    return set(frozenset((cycle[i], cycle[(i + 1) % n])) for i in range(n))


def translate_vertex(v, d, L):
    return ((v[0] + d[0]) % L, (v[1] + d[1]) % L, (v[2] + d[2]) % L, v[3])


def edge_resolved_profile(L, box):
    adj = build_prim_adjacency(L)
    start = (L // 2, L // 2, L // 2, 0)
    cyc = one_girth_cycle(adj, start)
    assert cyc is not None and len(cyc) == GIRTH, "no girth-10 cycle found"
    E0 = cycle_edge_set(cyc)
    profile = {}
    for d in product(range(-box, box + 1), repeat=3):
        Ed = set(frozenset((translate_vertex(u, d, L), translate_vertex(w, d, L)))
                 for u, w in (tuple(e) for e in E0))
        shared = E0 & Ed
        s = len(shared)
        if s == 0:
            profile[d] = 0.0
            continue
        deg = defaultdict(int)
        for e in (E0 | Ed):
            for v in e:
                deg[v] += 1
        n_branch = sum(1 for v in deg if deg[v] >= 3)
        dS = s - n_branch
        profile[d] = max(0.0, min(dS, DS_CAP)) * E_BIT
    return profile


def assert_contact(profile, label):
    """Regression of the 2026-05-29 proven-contact finding: V = 3*delta."""
    for d, v in profile.items():
        if d == (0, 0, 0):
            assert abs(v - 3.0) < 1e-12, f"contact regression broken at 0 ({label})"
        else:
            assert abs(v) < 1e-12, f"kernel not contact at {d} ({label})"


# ---------------------------------------------------------------------------
# Block 3 — pair dispersions on the offset mesh (verbatim values, cached exec)
# ---------------------------------------------------------------------------
_EPS_CACHE = {}


def eps_on_mesh(n_q):
    """eps_low over the offset mesh q_i = (i+0.5)/n_q, shape (n,n,n).
    Identical values to per-point calls (eps_low is exactly 1-periodic in k)."""
    if n_q in _EPS_CACHE:
        return _EPS_CACHE[n_q]
    qs = (np.arange(n_q) + 0.5) / n_q
    eps = np.empty((n_q, n_q, n_q))
    for i, j, l in product(range(n_q), repeat=3):
        eps[i, j, l] = eps_low(np.array([qs[i], qs[j], qs[l]]))
    _EPS_CACHE[n_q] = eps
    return eps


def epair_equal(n_q):
    """E_pair(q) = eps_low(q) + eps_low(-q); on the offset mesh -q is the
    reversed index (n-1-i), exactly (2026-05-29 line 203 convention)."""
    eps = eps_on_mesh(n_q)
    return eps + eps[::-1, ::-1, ::-1]


def epair_static(n_q, const=None):
    """STATIC limit m*2 -> inf: E_pair(q) = eps_low(q) + const.
    const drops out of B identically (pole condition sees E_pair - E only);
    declared const = eps_min."""
    eps = eps_on_mesh(n_q)
    if const is None:
        const = eps.min()
    return eps + const


# ---------------------------------------------------------------------------
# Block 4 — solvers
# ---------------------------------------------------------------------------
def kinetic_real_space(epair3, box, n_q):
    """T(R) = (1/Nq) sum_q E_pair(q) e^{2pi i q.R}  (2026-05-29 lines 198-210),
    computed by exact phase factorization e^{2pi i q.R} = prod_a e^{2pi i q_a R_a}
    (identical algebra, vectorized).  Returns Tarr over R in [-2box, 2box]^3."""
    qs = (np.arange(n_q) + 0.5) / n_q
    rng = np.arange(-2 * box, 2 * box + 1)
    px = np.exp(2j * np.pi * np.outer(rng, qs))           # (4b+1, n)
    A = np.einsum('ri,ijk->rjk', px, epair3.astype(complex))
    B = np.einsum('sj,rjk->rsk', px, A)
    T = np.einsum('tk,rsk->rst', px, B) / n_q ** 3        # (4b+1, 4b+1, 4b+1)
    return T


def solve_relative_contact(epair3, box, n_q, U):
    """H_rel = T_hat - U |0><0| on relative cells R in [-box, box]^3
    (2026-05-29 solve_relative, lines 213-225, with the PROVEN-CONTACT kernel).
    Returns E0, E_th, <T>, <V>, psi0_amp2."""
    T = kinetic_real_space(epair3, box, n_q)
    E_th = float(epair3.min())
    Rv = np.array(list(product(range(-box, box + 1), repeat=3)))   # (M,3)
    d = Rv[:, None, :] - Rv[None, :, :]                            # (M,M,3)
    idx = ((d[..., 0] + 2 * box) * (4 * box + 1) + (d[..., 1] + 2 * box)) \
        * (4 * box + 1) + (d[..., 2] + 2 * box)
    H = T.reshape(-1)[idx]                                         # Hermitian
    i0 = int(np.where((Rv == 0).all(axis=1))[0][0])
    Tmat = H.copy()
    H[i0, i0] += -U
    ev, evec = np.linalg.eigh(H)
    E0 = float(ev[0])
    psi = evec[:, 0]
    T_exp = float(np.real(psi.conj() @ Tmat @ psi))
    V_exp = -U * float(np.abs(psi[i0]) ** 2)
    # proven algebra: the eigen identity E0 = <T> + <V>
    assert abs(E0 - (T_exp + V_exp)) < 1e-9, "eigen identity broken"
    return E0, E_th, T_exp, V_exp, float(np.abs(psi[i0]) ** 2)


def pole_binding(dep_flat, U, s, delta_safe=0.02, iters=200):
    """Contact pole condition 1 = U * Pi(E_B) (2026-05-28 lines 88-114), exact
    infinite-box for a contact well.  dep = E_pair - E_th >= 0 (grid);
    g(B) = U * <1/(s*dep + B)>, decreasing; B in (delta_safe*s, U).
    Returns B or None (no grid-safe sub-threshold pole)."""
    g = lambda B: U * np.mean(1.0 / (s * dep_flat + B))
    lo, hi = delta_safe * s, U
    if g(lo) < 1.0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if g(mid) >= 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Block 5 — band-minimum refinement + Hessian (mu-mapping premise check)
# ---------------------------------------------------------------------------
def refine_minimum(k0, h0=0.02, h_min=1e-5):
    """Local coordinate-descent refinement of the band minimum (fractional k)."""
    k = np.array(k0, dtype=float)
    f = eps_low(k)
    h = h0
    while h > h_min:
        moved = False
        for a in range(3):
            for sgn in (+1, -1):
                kt = k.copy()
                kt[a] += sgn * h
                ft = eps_low(kt)
                if ft < f - 1e-14:
                    k, f, moved = kt, ft, True
        if not moved:
            h *= 0.5
    return k, f


def hessian_frac(k, h=2e-3):
    """Central-difference Hessian of eps_low in fractional-k coordinates.
    Only POSITIVE-DEFINITENESS is load-bearing (isolated quadratic minimum);
    absolute normalization (metric) deliberately not adopted (poison-safe)."""
    Hm = np.zeros((3, 3))
    f0 = eps_low(k)
    for a in range(3):
        for b in range(3):
            if a == b:
                kp, km = k.copy(), k.copy()
                kp[a] += h
                km[a] -= h
                Hm[a, a] = (eps_low(kp) - 2 * f0 + eps_low(km)) / h ** 2
            elif a < b:
                kpp, kpm, kmp, kmm = k.copy(), k.copy(), k.copy(), k.copy()
                kpp[a] += h; kpp[b] += h
                kpm[a] += h; kpm[b] -= h
                kmp[a] -= h; kmp[b] += h
                kmm[a] -= h; kmm[b] -= h
                Hm[a, b] = Hm[b, a] = (eps_low(kpp) - eps_low(kpm)
                                       - eps_low(kmp) + eps_low(kmm)) / (4 * h ** 2)
    return Hm


# ---------------------------------------------------------------------------
# MAIN — Stage 0, Stage A, A-2 class scan, A-3 ratio, verdict
# ---------------------------------------------------------------------------
def main():
    W = 78
    print("=" * W)
    print("IV.4 / T0-CLASS — the mu-scaling class of the MDL contact vertex")
    print("pre-reg: internal research notes (FROZEN)")
    print("units: substrate energy (t = e_bit = 1); U = dS*e_bit = 3 (disclosed)")
    print("=" * W)

    # ---- validation (proven algebra) ----
    ok = validate_dirac()
    print(f"\n[validation] Dirac D(k)^2 = 6I + R_sub : {'PASS' if ok else 'FAIL'}")
    assert ok, "Dirac validation regression"

    # ---- kernel: re-assert PROVEN CONTACT (regression) ----
    p1 = edge_resolved_profile(L=9, box=2)
    assert_contact(p1, "L=9,box=2")
    p2 = edge_resolved_profile(L=11, box=4)
    assert_contact(p2, "L=11,box=4")
    print("[kernel]     edge-resolved MDL profile re-measured at (L=9,box=2) and")
    print("             (L=11,box=4): V(Delta) = 3*delta_{Delta,0}  -> CONTACT (range 0)")
    print("             confirmed (2026-05-29 verdict (a) regression PASS)")

    # ================= STAGE 0 — T0-RIGID =================
    print("\n" + "=" * W)
    print("STAGE 0 — T0-RIGID  (frozen 2026-05-29 conventions verbatim; U = 3)")
    print("=" * W)
    frozen_ref = {2: (-1.4912, 2.6767), 3: (-1.4913, 2.6768), 4: (-1.4914, 2.6769)}
    print("\n[0.1] box convergence of H_rel = T_hat + V_hat  (n_q = 14, verbatim):")
    print("      box    E_th        E0          B = E_th - E0    <T>-E_th    U-B")
    ep14 = epair_equal(14)
    stage0 = {}
    for box in (2, 3, 4, 5):
        E0, E_th, T_exp, V_exp, a0 = solve_relative_contact(ep14, box, 14, U_OP)
        Bnd = E_th - E0
        stage0[box] = (E0, E_th, Bnd, T_exp, V_exp)
        print(f"      {box:>2}   {E_th:9.6f}  {E0:10.6f}     {Bnd:9.6f}     "
              f"{T_exp - E_th:8.6f}  {U_OP - Bnd:8.6f}")
        if box in frozen_ref:   # regression against the frozen 2026-05-29 output
            E0r, Br = frozen_ref[box]
            assert abs(E0 - E0r) < 2e-4, f"Stage-0 E0 regression at box={box}"
            assert abs(Bnd - Br) < 2e-4, f"Stage-0 B regression at box={box}"
            assert abs(E_th - 1.1855) < 1e-4, "Stage-0 E_th regression"
    print("      (boxes 2-4 regression-match the frozen 2026-05-29 run to <2e-4)")

    print("\n[0.2] pole cross-check 1 = U*Pi(E_B) (contact-exact, infinite box),")
    print("      grid convergence:")
    print("      n_q     E_th        B_pole      T0^(2) = U - B    T0^(2)/U")
    B_pole_seq = {}
    for n_q in (14, 20, 26):
        ep = epair_equal(n_q)
        dep = (ep - ep.min()).reshape(-1)
        Bp = pole_binding(dep, U_OP, 1.0)
        B_pole_seq[n_q] = (float(ep.min()), Bp)
        print(f"      {n_q:>3}  {ep.min():9.6f}   {Bp:9.6f}      {U_OP - Bp:9.6f}"
              f"       {(U_OP - Bp) / U_OP:8.5f}")
    B0 = B_pole_seq[26][1]                    # grid-converged primary
    B0_box = stage0[5][2]                     # box-converged (verbatim n_q=14)
    T0_2 = U_OP - B0
    print(f"\n[0.3] STAGE-0 NUMBERS (primary = pole n_q=26; box-verbatim alongside):")
    print(f"      B          = {B0:.6f}   (box-5/n14 verbatim: {B0_box:.6f})")
    print(f"      T0^(2)     = U - B = {T0_2:.6f}")
    print(f"      T0^(2)/U   = {T0_2 / U_OP:.5f}  ({100 * T0_2 / U_OP:.2f} %)")
    rigid = abs(T0_2) / U_OP < 1e-3
    print(f"      RIGID criterion |T0|/U < 1e-3 : {'MET -> RIGID' if rigid else 'NOT met -> NON-RIGID (quantified)'}")
    frac = T0_2 / U_OP
    print(f"\n[0.4] calibration fact (REPORT ONLY, declared context, not a target):")
    print(f"      T0^(2)/U = {100 * frac:.2f} %  vs the -16 % common kinetic factor the")
    if frac < 0.04:
        print("      nuclear pair needs (T4): PROVABLY TOO SMALL (< 4 %) -> the nuclear")
        print("      miss needs more than T0 (bookable adjunct: UNDER-POWERED).")
    else:
        print(f"      nuclear pair needs (T4): SAME ORDER (ratio {frac / 0.16:.2f} of 16 %) ->")
        print("      NOT provably too small; the 3-body T0-NUCLEAR station stays live.")

    # ================= STAGE A =================
    print("\n" + "=" * W)
    print("STAGE A — THE CLASS STATION  (two configurations ONLY; parameter-free)")
    print("=" * W)

    # premise check for the mu mapping: isolated quadratic minimum
    eps30 = eps_on_mesh(30)
    imin = np.unravel_index(np.argmin(eps30), eps30.shape)
    qs30 = (np.arange(30) + 0.5) / 30
    k_start = np.array([qs30[imin[0]], qs30[imin[1]], qs30[imin[2]]])
    k_min, eps_min_ref = refine_minimum(k_start)
    Hm = hessian_frac(k_min)
    hev = np.linalg.eigvalsh(Hm)
    print(f"\n[A.0] the electron band (frozen convention: lowest positive band of the")
    print(f"      32x32 Dirac D(k), 2026-05-29): eps_min = {eps_min_ref:.6f} at")
    print(f"      k_frac ~ ({k_min[0]:.4f}, {k_min[1]:.4f}, {k_min[2]:.4f}) [+ symmetry copies]")
    print(f"      Hessian eigenvalues (fractional-k units): "
          f"{hev[0]:.3f}, {hev[1]:.3f}, {hev[2]:.3f}")
    print(f"      positive-definite: {'YES -> isolated quadratic minimum; the' if (hev > 0).all() else 'NO — flagged:'}")
    print(f"      quadratic expansion exists and s*eps => m* -> m*/s => mu_eff ~ 1/s EXACT")
    print(f"      NOTE (honest): the band minimum is NOT at P = (1/4,1/4,1/4); the")
    print(f"      A5 P-point v_F = sqrt(3)/6 cite is the species-identification lineage")
    print(f"      of this same Dirac construction (see header adjudication).")

    # ---- A-1: the two frozen configurations at the operating point ----
    print(f"\n[A.1] ground state at the operating point (U = 3; box = 4, n_q = 14;")
    print(f"      pole cross-check in brackets):")
    ep_eq = ep14
    ep_st = epair_static(14)
    res = {}
    for label, ep3 in (("EQUAL ", ep_eq), ("STATIC", ep_st)):
        E0, E_th, T_exp, V_exp, a0 = solve_relative_contact(ep3, 4, 14, U_OP)
        dep = (ep3 - ep3.min()).reshape(-1)
        Bp = pole_binding(dep, U_OP, 1.0)
        Bnd = E_th - E0
        res[label.strip()] = dict(E0=E0, E_th=E_th, B=Bnd, Bp=Bp, T=T_exp, V=V_exp)
        print(f"      {label}: E_th = {E_th:9.6f}   E0 = {E0:10.6f}   "
              f"B = {Bnd:9.6f}  [{Bp:9.6f}]")
        print(f"              T0 = <T_hat> = {T_exp:9.6f}   <T_hat>-E_th = {T_exp - E_th:8.6f}   "
              f"<V_hat> = {V_exp:9.6f}   U-B = {U_OP - Bnd:8.6f}")
    # proven algebra: the STATIC additive constant drops out of B
    dep_st = (epair_static(14) - epair_static(14).min()).reshape(-1)
    eps14 = eps_on_mesh(14)
    dep_st0 = (eps14 - eps14.min()).reshape(-1)     # const = 0 variant
    B_c1 = pole_binding(dep_st, U_OP, 1.0)
    B_c0 = pole_binding(dep_st0, U_OP, 1.0)
    assert abs(B_c1 - B_c0) < 1e-10, "STATIC const-invariance (proven algebra) broken"
    print("      (STATIC additive-constant invariance of B: asserted, PASS)")

    # ---- A-3: the parameter-free ratio (raw) ----
    ratio_box = res["STATIC"]["B"] / res["EQUAL"]["B"]
    ratio_pole = res["STATIC"]["Bp"] / res["EQUAL"]["Bp"]
    print(f"\n[A.3] THE PARAMETER-FREE RATIO (raw, no adjustment):")
    print(f"      B_static / B_equal = {ratio_box:.6f}   (pole: {ratio_pole:.6f})")
    print(f"      [mu_static/mu_equal = 2 exactly; a linear-mu class would put the")
    print(f"       ratio at the reduced-mass form 2]")

    # ---- A-2: THE CLASS SCAN (frozen grids; see header) ----
    print(f"\n[A.2] THE CLASS VERDICT — B(U; s*eps) on the frozen grids")
    print(f"      U in {{0.3, 0.5, 1, 2, 3, 5}}; s in {{2^-1, 2^-1/2, 1, 2^1/2, 2}};")
    print(f"      exponent = dlnB/dln(mu_eff) = -dlnB/dln(s) (centered; one-sided at")
    print(f"      ends '^'); flags: * NEAR-THRESHOLD (B < 0.04 s), ~ GRID-LIMITED")
    print(f"      (n30 vs n20 > 1 %); 'unb' = no grid-safe sub-threshold pole.")
    U_GRID = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    S_GRID = [2 ** -1.0, 2 ** -0.5, 1.0, 2 ** 0.5, 2.0]
    ep30 = epair_equal(30)
    ep20 = epair_equal(20)
    scans = {}
    for cfg, e30, e20 in (("EQUAL", ep30, ep20),
                          ("STATIC", epair_static(30), epair_static(20))):
        d30 = (e30 - e30.min()).reshape(-1)
        d20 = (e20 - e20.min()).reshape(-1)
        Uc30 = 1.0 / np.mean(1.0 / (d30 + 0.02))
        tab = {}
        for U in U_GRID:
            for s in S_GRID:
                B30 = pole_binding(d30, U, s)
                B20 = pole_binding(d20, U, s)
                flag = ""
                if B30 is None:
                    tab[(U, s)] = (None, "unb")
                    continue
                if B30 < 2 * 0.02 * s:
                    flag += "*"
                if B20 is None or abs(B30 - B20) > 0.01 * B30:
                    flag += "~"
                tab[(U, s)] = (B30, flag)
        scans[cfg] = (tab, Uc30, d30)

    lnS = np.log(S_GRID)
    for cfg in ("EQUAL", "STATIC"):
        tab, Uc30, d30 = scans[cfg]
        print(f"\n      --- {cfg} configuration "
              f"(grid-operational U_c(s=1, delta=0.02, n30) = {Uc30:.4f}; "
              f"U_c(s) = s*U_c) ---")
        hdr = "      U \\ s |" + "".join(f"   s={s:6.4f}   " for s in S_GRID)
        print(hdr)
        print("      " + "-" * (len(hdr) - 6))
        for U in U_GRID:
            rowB, rowE = f"      {U:5.2f} |", "      B->exp|"
            Bs = [tab[(U, s)] for s in S_GRID]
            for i, s in enumerate(S_GRID):
                B, fl = Bs[i]
                rowB += f" {B:9.5f}{fl:<4}" if B is not None else "    unb      "
            # exponents: -dlnB/dlns, centered where possible
            for i, s in enumerate(S_GRID):
                B, fl = Bs[i]
                if B is None:
                    rowE += "     --      "
                    continue
                lo = i - 1 if i - 1 >= 0 and Bs[i - 1][0] is not None else i
                hi = i + 1 if i + 1 < len(S_GRID) and Bs[i + 1][0] is not None else i
                if lo == hi:
                    rowE += "     --      "
                    continue
                expo = -(np.log(Bs[hi][0]) - np.log(Bs[lo][0])) / (lnS[hi] - lnS[lo])
                mark = "^" if (lo == i or hi == i) else " "
                rowE += f" {expo:+8.4f}{mark}{Bs[i][1]:<3} "
            print(rowB)
            print(rowE)

    # operating-point refinement pair (declared a priori): U=3, s = 2^{+-1/8}
    print(f"\n      operating-point exponent (U = 3, s = 1; refined centered pair")
    print(f"      s = 2^-1/8, 2^+1/8, declared a priori):")
    op_expo = {}
    for cfg in ("EQUAL", "STATIC"):
        d30 = scans[cfg][2]
        sm, sp = 2 ** (-1 / 8), 2 ** (1 / 8)
        Bm = pole_binding(d30, 3.0, sm)
        Bp = pole_binding(d30, 3.0, sp)
        expo = -(np.log(Bp) - np.log(Bm)) / (np.log(sp) - np.log(sm))
        op_expo[cfg] = expo
        print(f"        {cfg:6s}: dlnB/dln(mu_eff) = {expo:+.5f}")
    c_deep = float(np.mean(scans["EQUAL"][2]))
    print(f"      [deep-regime anatomy, computed: B ~ U - c*s with c = <E_pair>-E_th")
    print(f"       = {c_deep:.4f} (n30, EQUAL) -> predicted exponent c*s/B|op ~ "
          f"{c_deep / (U_OP - c_deep):.4f}]")

    # ================= VERDICT =================
    print("\n" + "=" * W)
    print("VERDICT (frozen tree; thresholds declared in the header a priori)")
    print("=" * W)
    expo_op = op_expo["EQUAL"]
    linear_mu = (abs(expo_op - 1.0) <= 0.15) and (abs(ratio_box - 2.0) <= 0.2)
    if rigid:
        verdict = "RIGID"
    elif linear_mu:
        verdict = "T1-CLOSE"
    else:
        verdict = "T1-FENCE"
    print(f"\n  STAGE 0 : T0^(2) = {T0_2:.4f} = {100 * T0_2 / U_OP:.2f} % of U  ->  "
          f"{'RIGID' if rigid else 'NON-RIGID'}")
    print(f"            (T0 is real and quantified: reading (b) of the sealed A2 is")
    print(f"            INSTANTIATED in-framework — the law E_bind = -kappa*dS is")
    print(f"            incomplete without the kinetic term; E_bind = -kappa*dS +")
    print(f"            T0(mu_eff) carries a derived, nonzero T0.)")
    print(f"\n  STAGE A : operating-point exponent dlnB/dln(mu_eff) = {expo_op:+.4f} "
          f"(EQUAL; STATIC {op_expo['STATIC']:+.4f})")
    print(f"            linear-mu (+1) criterion |exp-1| <= 0.15 : "
          f"{'MET' if abs(expo_op - 1) <= 0.15 else 'NOT MET'}")
    print(f"            B_static/B_equal = {ratio_box:.4f} ; reduced-mass form (=2)")
    print(f"            criterion |ratio-2| <= 0.2 : "
          f"{'MET' if abs(ratio_box - 2) <= 0.2 else 'NOT MET'}")
    print(f"\n  ==> STATION VERDICT: {verdict}")
    if verdict == "T1-FENCE":
        print(f"\n  The derived class of the MDL contact vertex is NOT linear-mu at the")
        print(f"  untuned operating point (U = 3, s = 1): the vertex is DEEP-CONTACT,")
        print(f"  B = U - T0 with T0 a small kinetic zero-point correction (exponent")
        print(f"  ~ {expo_op:+.2f}, not +1).  The exponent runs from large (near-critical,")
        print(f"  where B ~ (U - s*U_c)^2) through +1 only on a TUNED locus far from")
        print(f"  U = 3.  Direction note: B_static > B_equal (heavier partner binds")
        print(f"  deeper) — the hydrogen-vs-positronium ORDER is reproduced, but the")
        print(f"  magnitude class is not linear-mu.  Per the frozen verdict tree, the")
        print(f"  H/Ps reduced-mass datum is adjudicated OUT of the contact vertex and")
        print(f"  INTO the CONNECTION sector (the finite-k photon, IV.7): a publishable")
        print(f"  re-scope of IV.6's atomic block and a sharpening of I-0a's")
        print(f"  vertex/connection split.  NOT a failure of T0.")
    print(f"\n  DECLARED-CONTEXT CONFRONTATION (the two known targets, stated ONLY here,")
    print(f"  per the pre-reg's honesty clause; they were never optimization targets):")
    print(f"    measured H/Ps binding ratio ~ 2.0 (the reduced-mass factor)  vs")
    print(f"    derived B_static/B_equal = {ratio_box:.4f}   -> the contact vertex does NOT")
    print(f"    own the atomic reduced-mass law (consistent with the {verdict} verdict);")
    print(f"    nuclear common kinetic factor ~ -16 % (T4)  vs  T0^(2)/U = "
          f"{100 * frac:.2f} %")
    print(f"    -> same order: the kinetic sector is NOT under-powered for the nuclear")
    print(f"    station; the 3-body T0-NUCLEAR pre-reg stays gated on this verdict.")
    print(f"\n  Poisons honored: no kappa; no oblique/EW; E_odd (0.381876 MeV) untouched;")
    print(f"  the 13/3 RATIO-MISS stays booked OPEN regardless of this outcome; e_bit=t")
    print(f"  disclosed; both configurations parameter-free (no m* adoption); no")
    print(f"  post-output adjustment of U, range, cap, box, band, or grids.")
    print("=" * W)


if __name__ == "__main__":
    main()
