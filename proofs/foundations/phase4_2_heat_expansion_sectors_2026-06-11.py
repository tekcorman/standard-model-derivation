#!/usr/bin/env python3
"""Phase 4.2 -- heat expansion of Tr f(D_z^2/Lambda^2): the induced sectors.

Spec: docs/scoping/phase4_bet_spec_2026-06-11.md (FROZEN b4bb97b; S1 banked
7d25109). DESIGN NOTE (first-run gate corrections, recorded in the spec
log): the sector table is computed at MOMENT level -- exact t-polynomials
of Tr(D+tV)^{2n}, the graph Seeley-analog coefficients, which are f-FREE --
because in the spectral action each Seeley coefficient carries its own
f-moment (f0, f2, f4, ...), so fixed-Lambda fits mix them and sharp-cutoff
finite differences are distributional. f enters only in the ASSEMBLY
section (smooth f's, two-route gated). Gauge universality is gated within
space-group EDGE-PAIR ORBITS (this Dirac ties the Clifford index to the
edge index; the quotient graph is K4: 12 adjacent + 3 disjoint pairs), with
S4-invariant atom-indicator profile sums on a symmetric 4^3 BZ grid.

Frozen commitments executed: f-class {exp(-x), exp(-sqrt x), sharp cutoff};
inner-fluctuation one-forms A = a[D,b] with internal factor = the Cl(6)
bivectors (Spin(6) = SU(4)_PS EXACT per k4_pati_salam_cl8.py -- all 15);
HIGGS direction = the mirror-off-diagonal sigma ~ M; two routes; D1/D3
controls on comparable rows.

Gates:
  W1  two-route identity (f1 eigenvalue sum == moment Taylor, L^2 = 24).
  W2  Lambda/topological block: a0 = 64, m2 = 384, m4 = 3456 exact k-flat.
  W3  curvature sector: m6(k) k-VARYING (EH-analog seat); harmonics.
  W4  HIGGS sector, moment level (all EXACT, k-flat, f-free):
      {D_z, M} = 2 M Phi_cov;  m2(t) = 384 + 64 t^2;
      m4(t) = 3456 + 1536 t^2 + 64 t^4  -- the quartic coefficient = dim
      and the quadratic SPLITS 1536 = 768 (universal t^2 I) + 768
      (condensate 4 Tr Phi^2): the sigma-potential is integer-exact.
      m6(t) coefficients recorded (flatness pattern).
  W5  GAUGE sector: g4(T) = t^2-coefficient of <Tr(D_pair + tV)^4> with
      S4-summed profiles on the symmetric grid: EXACTLY constant within
      each edge-pair orbit (12 adjacent | 3 disjoint); kinetic moments
      (q^2-slopes, axis-averaged) nonzero for representatives + U(1) +
      chirality-projected row; su(4) adjoint sum recorded (the 4.3 input).
  W6  sigma kinetic moment nonzero (M-direction one-form q-response).
  W7  K4 controls: U(1) one-form g4 SIGN and M-coupler quartic SIGN agree
      across D2/D1/D3 (comparable rows); the Phi-condensate row is
      D2-specific (S1 scorecard, restated -- not a K4 trigger since D2 is
      the pre-registered default carrying the full triple).
  W8  f-assembly: smooth f's assemble the moment table (f1 route-B
      regression on the sigma family); sharp cutoff = counting function
      (staircase recorded, no derivative gates -- distributional).
"""
import os
import sys
from math import factorial

import numpy as np
from numpy import linalg as la

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from proofs.common import find_bonds, A_PRIM  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


GAMMAS = [kron3(X, I2, I2), kron3(Y, I2, I2), kron3(Z, X, I2),
          kron3(Z, Y, I2), kron3(Z, Z, X), kron3(Z, Z, Y)]
GAMMA7 = ((-1j) ** 3) * np.linalg.multi_dot(GAMMAS)


def undirected_edges():
    bonds = find_bonds()
    seen = {}
    for src, tgt, cell in bonds:
        cell = tuple(int(c) for c in cell)
        key = (src, tgt, cell) if src < tgt else (tgt, src, tuple(-c for c in cell))
        seen[key] = True
    edges = sorted(seen.keys())
    assert len(edges) == 6
    return edges


EDGES = undirected_edges()


def L_e(edge, k):
    a, b, n = edge
    L = np.zeros((4, 4), dtype=complex)
    ph = np.exp(2j * np.pi * np.dot(k, n))
    L[b, a] = ph
    L[a, b] = np.conj(ph)
    for c in range(4):
        if c not in (a, b):
            L[c, c] = 1.0
    return L


def D_of_k(k):
    D = np.zeros((32, 32), dtype=complex)
    for i, e in enumerate(EDGES):
        D += np.kron(GAMMAS[i], L_e(e, k))
    return D


DELTA = np.array([0.5, 0.5, -0.5])


def D_cover(k):
    Dz = np.zeros((64, 64), dtype=complex)
    Dz[:32, :32] = D_of_k(k)
    Dz[32:, 32:] = D_of_k(np.asarray(k) + DELTA)
    return Dz


_DCACHE = {}


def D_cover_c(k):
    key = tuple(np.round(np.asarray(k, float), 9))
    if key not in _DCACHE:
        _DCACHE[key] = D_cover(k)
    return _DCACHE[key]


PHI = np.zeros((32, 32), dtype=complex)
for i, (a, b, n) in enumerate(EDGES):
    F = np.zeros((4, 4))
    for c in range(4):
        if c not in (a, b):
            F[c, c] = 1.0
    PHI += np.kron(GAMMAS[i], F)
PHI_COV = np.kron(np.eye(2), PHI)
M_SWAP = np.kron(np.array([[0, 1], [1, 0]]), np.eye(32)).astype(complex)

SADDLES = {"Gamma": np.zeros(3), "H": np.array([0.5, 0.5, -0.5]),
           "P": np.array([0.25, 0.25, 0.25]), "N": A_PRIM @ np.array([0.0, 0.5, 0.5])}
GRID6 = [np.array([i, j, l]) / 6.0 for i in range(6) for j in range(6)
         for l in range(6)]
GRID4 = [np.array([i, j, l]) / 4.0 for i in range(4) for j in range(4)
         for l in range(4)]          # cubic-symmetric as a set (mod 1)
KSAMP = list(SADDLES.values()) + GRID6[::72]

print("=" * 72)
print(" PHASE 4.2 -- heat expansion: the induced sector structure")
print("=" * 72)

# ---- W1: two-route identity ----
L2, N_TERMS = 24.0, 14


def S_route_A(Dz, f, L2):
    ev = la.eigvalsh(Dz @ Dz)
    return float(np.sum(f(ev / L2)))


def moments(Dz, nmax):
    M2 = Dz @ Dz
    out, P = [], np.eye(Dz.shape[0], dtype=complex)
    for _ in range(nmax + 1):
        out.append(float(np.trace(P).real))
        P = P @ M2
    return out


devs = []
for k in KSAMP:
    Dz = D_cover(k)
    a = S_route_A(Dz, lambda x: np.exp(-x), L2)
    m = moments(Dz, N_TERMS)
    b = sum(((-1.0) ** n / factorial(n)) * m[n] / L2 ** n
            for n in range(N_TERMS + 1))
    devs.append(abs(a - b) / abs(a))
gate("W1 two-route identity (f1, L^2 = 24): eigenvalue sum == moment Taylor",
     max(devs) < 1e-6, f"max rel dev={max(devs):.1e}")

# ---- W2: flat block ----
m_chk = [moments(D_cover(k), 2) for k in KSAMP]
gate("W2 Lambda/topological block: a0 = 64, m2 = 384, m4 = 3456 exact k-flat",
     max(abs(m[0] - 64) + abs(m[1] - 384) + abs(m[2] - 3456) for m in m_chk) < 1e-8)

# ---- W3: curvature at m6 ----
m6 = np.array([moments(D_cover(k), 3)[3] for k in GRID6]).reshape(6, 6, 6)
m6_mean, m6_std = float(m6.mean()), float(m6.std())
spec_f = np.abs(np.fft.fftn(m6 - m6_mean)) / m6.size
idx = np.dstack(np.unravel_index(np.argsort(spec_f.ravel())[::-1], spec_f.shape))[0]
top = [(tuple(int(x) if x <= 3 else int(x) - 6 for x in idx[i]),
        float(spec_f[tuple(idx[i])])) for i in range(0, 6, 2)]
gate("W3 curvature sector EXISTS at m6 (k-varying; EH-analog seat)",
     m6_std > 1.0, f"mean={m6_mean:.1f}, std={m6_std:.2f}, top {top[:3]}")

# ---- W4: Higgs sector at moment level (exact) ----
dev_id = max(la.norm((D_cover(k) @ M_SWAP + M_SWAP @ D_cover(k))
                     - 2 * M_SWAP @ PHI_COV) for k in KSAMP)
tr_phi2 = float(np.trace(PHI_COV @ PHI_COV).real)


def sigma_polys(k):
    """Exact even-coefficient extraction of m2(t), m4(t), m6(t) for D+tM."""
    ts = np.arange(-3, 4, dtype=float)
    vals = {2: [], 4: [], 6: []}
    for t in ts:
        Dt = D_cover(k) + t * M_SWAP
        m = moments(Dt, 3)
        vals[2].append(m[1])
        vals[4].append(m[2])
        vals[6].append(m[3])
    V = np.vander(ts, 7, increasing=True)
    out = {}
    for n in (2, 4, 6):
        out[n] = la.solve(V, np.array(vals[n]))  # coeffs c0..c6
    return out


pol = sigma_polys(SADDLES["P"])
pol2 = sigma_polys(GRID6[100])
m2_ok = (abs(pol[2][0] - 384) < 1e-8 and abs(pol[2][2] - 64) < 1e-8
         and max(abs(pol[2][i]) for i in (1, 3, 4, 5, 6)) < 1e-8)
m4_ok = (abs(pol[4][0] - 3456) < 1e-7 and abs(pol[4][2] - 1536) < 1e-7
         and abs(pol[4][4] - 64) < 1e-7
         and max(abs(pol[4][i]) for i in (1, 3, 5, 6)) < 1e-7)
flat_ok = max(abs(pol[4][i] - pol2[4][i]) for i in range(7)) < 1e-7
gate("W4a HIGGS moment block EXACT + k-flat + f-free: {D,M} = 2MPhi; "
     "m2(t) = 384 + 64t^2; m4(t) = 3456 + 1536t^2 + 64t^4",
     dev_id < 1e-12 and m2_ok and m4_ok and flat_ok,
     f"id dev={dev_id:.1e}; 4*TrPhi^2 = {4 * tr_phi2:.1f} (condensate share "
     f"of the 1536 = 768 + 768 split)")
gate("W4b sigma quartic coefficient = 64 = a0 (dim) > 0 -- the Higgs-shaped "
     "quartic EXISTS at moment level, integer-exact",
     abs(pol[4][4] - 64) < 1e-7 and abs(pol[2][2] - 64) < 1e-8)
m6c = pol[6]
m6c2 = pol2[6]
m6_coef_flat = [abs(m6c[i] - m6c2[i]) < 1e-6 for i in (2, 4, 6)]
print(f"      m6(t) coeffs at P: t^0={m6c[0]:.1f} (k-varying), "
      f"t^2={m6c[2]:.1f}, t^4={m6c[4]:.1f}, t^6={m6c[6]:.1f}; "
      f"t^2/t^4/t^6 k-flat: {m6_coef_flat}")

# ---- W5: gauge sector (edge-pair orbits; S4-summed profiles; exact traces) ----
BIVS = {}
for e in range(6):
    for f_ in range(e + 1, 6):
        BIVS[(e, f_)] = 1j * (GAMMAS[e] @ GAMMAS[f_])


def edge_pair_class(e, f_):
    (a1, b1, _), (a2, b2, _) = EDGES[e], EDGES[f_]
    return "adjacent" if len({a1, b1} & {a2, b2}) > 0 else "disjoint"


CLASSES = {p: edge_pair_class(*p) for p in BIVS}
n_adj = sum(1 for v in CLASSES.values() if v == "adjacent")
print(f"      edge-pair orbits (quotient graph K4): adjacent {n_adj}, "
      f"disjoint {15 - n_adj}")

INDICATORS = [np.diag((np.arange(4) == c).astype(float)) for c in range(4)]


def one_form(T, k, prof):
    """A = (prof (x) T)[D_z(k), prof (x) I] -- atom-indicator profile."""
    Dk = D_cover(k)
    Bm = np.kron(np.eye(2), np.kron(np.eye(8), prof))
    Am = np.kron(np.eye(2), np.kron(T, prof))
    return Am @ (Dk @ Bm - Bm @ Dk)


def g4_of(T, q, ks):
    """t^2-coefficient of <Tr (D_pair + t V)^4>: 4Tr(D^2V^2) + 2Tr(DVDV),
    summed over the 4 S4-invariant indicator profiles."""
    tot = 0.0
    for k in ks:
        Dk, Dkq = D_cover(k), D_cover(np.asarray(k) + q)
        Dp = np.zeros((128, 128), dtype=complex)
        Dp[:64, :64], Dp[64:, 64:] = Dk, Dkq
        for prof in INDICATORS:
            V = one_form(T, k, prof)
            Vp = np.zeros((128, 128), dtype=complex)
            Vp[64:, :64], Vp[:64, 64:] = V, V.conj().T
            DV = Dp @ Vp
            tot += float((4 * np.trace(Dp @ DV @ Vp) + 2 * np.trace(DV @ DV)).real)
    return tot / len(ks)


def g4_fast(T, ks):
    """q = 0 shortcut in the single 64-dim fiber:
    g4 = 4[Tr(D^2 V^dag V) + Tr(D^2 V V^dag)] + 4 Re Tr(D V^dag D V)."""
    tot = 0.0
    for k in ks:
        D = D_cover_c(k)
        D2 = D @ D
        for prof in INDICATORS:
            Bm = np.kron(np.eye(2), np.kron(np.eye(8), prof))
            Xp = D @ Bm - Bm @ D
            V = np.kron(np.eye(2), np.kron(T, prof)) @ Xp
            Vd = V.conj().T
            t1 = float(np.sum((D2 @ Vd) * V.T).real)
            t2 = float(np.sum((D2 @ V) * np.conj(V)).real)
            t3 = float(((D @ Vd) * (D @ V).T).sum().real)
            tot += 4 * t1 + 4 * t2 + 4 * t3
    return tot / len(ks)


# one-shot consistency: the fast single-fiber form == the pair form at q=0
_chk_T = BIVS[(0, 1)]
_chk = abs(g4_fast(_chk_T, [GRID4[5]]) - g4_of(_chk_T, np.zeros(3), [GRID4[5]]))
assert _chk < 1e-8, f"g4_fast inconsistent with pair form: {_chk}"

# orbit-constancy at q = 0 on the symmetric grid (exact statement)
g4_vals = {p: g4_fast(T, GRID4) for p, T in BIVS.items()}
adj = [v for p, v in g4_vals.items() if CLASSES[p] == "adjacent"]
dis = [v for p, v in g4_vals.items() if CLASSES[p] == "disjoint"]
sprd_a = (max(adj) - min(adj)) / max(abs(np.mean(adj)), 1e-12)
sprd_d = (max(dis) - min(dis)) / max(abs(np.mean(dis)), 1e-12)
gate("W5a gauge response constant WITHIN each edge-pair orbit (12 adjacent "
     "| 3 disjoint), S4-summed, symmetric grid",
     sprd_a < 1e-9 and sprd_d < 1e-9,
     f"adjacent g4 = {np.mean(adj):+.4f} (spread {sprd_a:.1e}), disjoint "
     f"g4 = {np.mean(dis):+.4f} (spread {sprd_d:.1e})")
su4_sum = sum(g4_vals.values())
print(f"      Sum over the 15 DISTINGUISHED bivectors: {su4_sum:+.4f} = "
      f"12*({np.mean(adj):+.4f}) + 3*({np.mean(dis):+.4f})")
print("      PANEL DEMOTION (2026-06-12, binding): equality across both "
      "orbits is a DISTINGUISHED-BIVECTOR-BASIS statement, NOT su(4)-"
      "invariant (the g4 form varies on random Lie-algebra combinations, "
      "9971-10836); '157440' is a normalization input, UNUSED in 4.3.")

# kinetic moments: m4 level is EXACTLY q-flat (co-located with curvature at
# m6, like the continuum a4 containing F^2) -- gate the forced flatness,
# then compute the kinetic moments at the m6 level.
QMAG = 0.15
GRID2 = [np.array([i, j, l]) / 2.0 for i in range(2) for j in range(2)
         for l in range(2)]
GRID3 = [np.array([i, j, l]) / 3.0 for i in range(3) for j in range(3)
         for l in range(3)]
# panel correction C2: the m4 q-flatness is a POINTWISE identity -- gate it
# at single random k with random q directions (no grid sum, no aliasing)
RNG2 = np.random.default_rng(20260612)
m4_flat = []
for nm, T in (("adj(0,1)", BIVS[(0, 1)]), ("U(1)", np.eye(8, dtype=complex))):
    for _ in range(3):
        k1 = RNG2.uniform(0, 1, 3)
        q1 = RNG2.uniform(-0.3, 0.3, 3)
        m4_flat.append(abs(g4_of(T, q1, [k1]) - g4_of(T, np.zeros(3), [k1])))
gate("W5b-pre m4-level kinetic response = 0 POINTWISE (panel-verified "
     "identity: per fixed k, any q -- no grid involved; kinetic co-locates "
     "with curvature at m6, as continuum a4 contains F^2)",
     max(m4_flat) < 1e-6, f"max pointwise |g4(k;q)-g4(k;0)| = {max(m4_flat):.1e}")


def t2_coeff_m(pows, Vp, m):
    """EXACT t^2-coefficient of Tr (Dp + t Vp)^m:
    (m/2) * sum_{j=0..m-2} Tr(Dp^j Vp Dp^(m-2-j) Vp); pows precomputed."""
    W = [pows[j] @ Vp for j in range(m - 1)]
    s = sum(float(np.sum(W[j] * W[m - 2 - j].T).real) for j in range(m - 1))
    return (m / 2.0) * s


def kin_coeff(T_or_V, q, ks, m, sigma=False):
    """t^2-coefficient of <Tr (D_pair + t V_pair)^m>, S4-summed profiles."""
    tot = 0.0
    for k in ks:
        Dk, Dkq = D_cover_c(k), D_cover_c(np.asarray(k) + q)
        Dp = np.zeros((128, 128), dtype=complex)
        Dp[:64, :64], Dp[64:, 64:] = Dk, Dkq
        pows = [np.eye(128, dtype=complex)]
        for _ in range(m - 2):
            pows.append(pows[-1] @ Dp)
        for prof in INDICATORS:
            if sigma:
                V = M_SWAP @ np.kron(np.eye(2), np.kron(np.eye(8), prof))
            else:
                V = one_form(T_or_V, k, prof)
            Vp = np.zeros((128, 128), dtype=complex)
            Vp[64:, :64], Vp[:64, 64:] = V, V.conj().T
            tot += t2_coeff_m(pows, Vp, m)
    return tot / len(ks)


def m6_t2_coeff(T_or_V, q, ks, sigma=False):
    return kin_coeff(T_or_V, q, ks, 6, sigma=sigma)


kin6 = {}
g0_adj = m6_t2_coeff(BIVS[(0, 1)], np.zeros(3), GRID4)
adj_slopes = []
for ax in range(3):
    q = np.zeros(3)
    q[ax] = QMAG
    adj_slopes.append((m6_t2_coeff(BIVS[(0, 1)], q, GRID4) - g0_adj) / QMAG ** 2)
kin6["adj(0,1)"] = (g0_adj, float(np.mean(adj_slopes)))
T_u1 = np.eye(8, dtype=complex)
g0_u1 = m6_t2_coeff(T_u1, np.zeros(3), GRID4)
kin6["U(1) T=I"] = (g0_u1, (m6_t2_coeff(T_u1, np.array([QMAG, 0, 0]), GRID4)
                            - g0_u1) / QMAG ** 2)   # isotropic: one axis
for nm, v in kin6.items():
    print(f"      m6-level kinetic {nm:10s}: g6(0)={v[0]:+.2f}, "
          f"c2 = {v[1]:+.4f} (4^3 grid)")
print("      (2^3 subgroup grid ALIASES the kinetic harmonics to exactly 0 "
      "-- the kinetic content lives outside the 2Z^3 harmonic lattice)")
print("      PANEL ANISOTROPY DISCLOSURE (2026-06-12): the adjoint value is "
      "an AXIS AVERAGE of one direction-locked kinetic quantum (-21105.4 on "
      "the locked axis, ~0 off-axis); the U(1) response is isotropic "
      "(referee-verified).")
gate("W5b gauge kinetic moments EXIST at the m6 level (q^2-slope nonzero: "
     "su(4) representative axis-avg + U(1))",
     all(abs(v[1]) > 1e-6 for v in kin6.values()))
g4_chiral = g4_fast(BIVS[(0, 1)] @ ((np.eye(8) + GAMMA7) / 2.0), GRID4)
gate("W5c chirality-projected direction carries EXCESS over half its parent "
     "(the L/R split has moment-level content)",
     abs(g4_chiral - np.mean(adj) / 2) > 1.0,
     f"g4(adj P+) = {g4_chiral:+.1f} vs parent/2 = {np.mean(adj) / 2:+.1f} "
     f"(excess {g4_chiral - np.mean(adj) / 2:+.1f} = 2*dim)")

# ---- W6: sigma kinetic moment -- TIERED: zero at m6, appears at m8 ----
gs0 = m6_t2_coeff(None, np.zeros(3), GRID4, sigma=True)
gs_dev = max(abs(m6_t2_coeff(None, np.array([QMAG, 0, 0]), GRID4, sigma=True)
                 - gs0),
             abs(m6_t2_coeff(None, np.array([QMAG] * 3), GRID2, sigma=True)
                 - m6_t2_coeff(None, np.zeros(3), GRID2, sigma=True)))
# panel ERRATUM (2026-06-12): the m8 value must be computed ALIAS-FREE --
# the original GRID2 figure +89362.23 was a 3.18x aliasing artifact (see
# phase4_e2_sigma_census_aliasfree_2026-06-12.py: GRID3 == GRID5 == MC).
g8_0 = kin_coeff(None, np.zeros(3), GRID3, 8, sigma=True)
g8_slopes = [(kin_coeff(None, np.array([QMAG if ax == a2 else 0.0
                                        for a2 in range(3)]), GRID3, 8,
                        sigma=True) - g8_0) / QMAG ** 2 for ax in range(3)]
cs2 = float(np.mean(g8_slopes))
gate("W6 sigma kinetic TIERED (alias-free, panel erratum applied): "
     "machine-zero at m6 (forced; any q-direction) but NONZERO at m8 -- "
     "the Higgs-direction coupler propagates one Seeley rung deeper than "
     "the gauge fields; corrected value ~+28140 (was +89362, GRID2 artifact)",
     gs_dev < 1e-6 and abs(cs2) > 1e-3 and abs(cs2 - 28140.53) < 1.0,
     f"m6 dev={gs_dev:.1e}; m8 c2_sigma={cs2:+.2f} (isotropic, E2-certified)")

# ---- W7: K4 controls on comparable rows ----
def D1_of(k):
    d = np.zeros((6, 4), dtype=complex)
    for i, (a, b, n) in enumerate(EDGES):
        d[i, a] = -1.0
        d[i, b] = np.exp(2j * np.pi * np.dot(k, n))
    D1 = np.zeros((10, 10), dtype=complex)
    D1[4:, :4] = d
    D1[:4, 4:] = d.conj().T
    return D1


def B_of(k):
    bonds = find_bonds()
    E = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
    idx = {e: a for a, e in enumerate(E)}
    rev = {a: idx[(j, i, tuple(-x for x in c))] for a, (i, j, c) in enumerate(E)}
    B = np.zeros((12, 12), dtype=complex)
    for a2, (i, j, c) in enumerate(E):
        for b2, (i2, j2, c2) in enumerate(E):
            if i2 == j and b2 != rev[a2]:
                B[b2, a2] = np.exp(2j * np.pi * np.dot(k, np.asarray(c2, float)))
    return B


def D3_of(k):
    B = B_of(k)
    D3 = np.zeros((24, 24), dtype=complex)
    D3[12:, :12] = B
    D3[:12, 12:] = B.conj().T
    return D3


def ctrl_u1_g4(D_fn, dim, atom_diag_fn, ks=GRID4[::4]):
    """U(1) one-form g4 for a control Dirac (atom-profile one-forms)."""
    tot = 0.0
    for k in ks:
        Dk = D_fn(k)
        for prof_v in INDICATORS:
            Bm = atom_diag_fn(prof_v)
            V = Bm @ (Dk @ Bm - Bm @ Dk)  # a[D,b] with a = b = indicator
            DV = Dk @ V
            tot += float((4 * np.trace(Dk @ DV @ V) + 2 * np.trace(DV @ DV)).real)
    return tot / len(ks)


# D2 same-convention row (a = b = indicator, T = I, single fiber)
g4_d2 = ctrl_u1_g4(D_cover, 64,
                   lambda p: np.kron(np.eye(2), np.kron(np.eye(8), p)))
g4_d1 = ctrl_u1_g4(D1_of, 10,
                   lambda p: la.block_diag(p, np.zeros((6, 6))) if hasattr(la, "block_diag")
                   else np.block([[p, np.zeros((4, 6))],
                                  [np.zeros((6, 4)), np.zeros((6, 6))]]))


def d3_atom_diag(p):
    # directed edges inherit the SOURCE atom's profile value
    bonds = find_bonds()
    E = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
    vals = [p[i, i] for (i, j, c) in E]
    return np.diag(vals + vals).astype(complex)  # block diag over the two chiral copies


g4_d3 = ctrl_u1_g4(D3_of, 24, d3_atom_diag)


def ctrl_m4_quartic(D_fn, dim, ks=GRID4[::4]):
    Mc = np.kron(np.array([[0, 1], [1, 0]]), np.eye(dim)).astype(complex)
    ts = np.arange(-2, 3, dtype=float)
    vals = []
    for t in ts:
        s = 0.0
        for k in ks:
            Dc = np.zeros((2 * dim, 2 * dim), dtype=complex)
            Dc[:dim, :dim] = D_fn(k)
            Dc[dim:, dim:] = D_fn(np.asarray(k) + DELTA)
            M2 = (Dc + t * Mc) @ (Dc + t * Mc)
            s += float(np.trace(M2 @ M2).real)
        vals.append(s / len(ks))
    V = np.vander(ts, 5, increasing=True)
    return la.solve(V, np.array(vals))[4]


q4_d1 = ctrl_m4_quartic(lambda k: D_of_k(k), 32)   # srs fiber of D2 (control basis)
q4_d1_dr = ctrl_m4_quartic(D1_of, 10)
q4_d3 = ctrl_m4_quartic(D3_of, 24)
gate("W7 K4 controls: U(1) one-form g4 sign agrees across D2/D1/D3 "
     "(the comparable row; K4 does not fire). PANEL NOTE: the M-coupler "
     "quartic-sign control is VACUOUS as evidence (t^4 coefficient = cover "
     "dim for ANY Hermitian block Dirac) -- recorded, carries no weight",
     np.sign(g4_d2) == np.sign(g4_d1) == np.sign(g4_d3),
     f"U(1) g4: D2 {g4_d2:+.3f}, D1 {g4_d1:+.3f}, D3 {g4_d3:+.3f}; "
     f"quartic (vacuous: = dim): D2 {q4_d1:+.3f}, D1 {q4_d1_dr:+.3f}, "
     f"D3 {q4_d3:+.3f}")
print("      (Phi-condensate share remains D2-specific -- S1 scorecard row "
      "restated, not a K4 trigger: D2 is the pre-registered default)")

# ---- W8: f-assembly ----
# f1 assembles the sigma moment family: S(t) = sum_n (-1)^n/n! m_{2n}(t)/L^2n
def S_sigma_assembled(k, t, L2, nmax=14):
    Dt = D_cover(k) + t * M_SWAP
    m = moments(Dt, nmax)
    return sum(((-1.0) ** n / factorial(n)) * m[n] / L2 ** n
               for n in range(nmax + 1))


dev_asm = []
for t in (0.5, 1.0):
    for k in (SADDLES["P"], GRID6[100]):
        a = S_route_A(D_cover(k) + t * M_SWAP, lambda x: np.exp(-x), L2)
        b = S_sigma_assembled(k, t, L2)
        dev_asm.append(abs(a - b) / abs(a))
ev0 = la.eigvalsh(D_cover(SADDLES["P"]) @ D_cover(SADDLES["P"]))
stair = [int(np.sum(ev0 <= s)) for s in (4.0, 6.0, 8.0, 12.0)]
gate("W8 f-assembly: f1 (smooth) reassembles the moment table (route B "
     "== route A on the sigma family); sharp cutoff = counting staircase "
     "(recorded, distributional -- no derivative gates)",
     max(dev_asm) < 1e-6,
     f"max rel dev={max(dev_asm):.1e}; N(L^2) staircase at P: "
     f"{dict(zip((4, 6, 8, 12), stair))}")

print("\n--- 4.2 SECTOR TABLE (moment level, f-free; panel-corrected "
      "2026-06-12) ---")
print("  Lambda/topological : a0=64, m2=384, m4=3456 (k-flat, exact)")
print(f"  curvature (EH seat): m6 k-varying, std {m6_std:.2f}; harmonics "
      f"2nd-order cubic-axis")
print(f"  gauge              : g4 = {np.mean(adj):+.1f} on the 15 "
      f"DISTINGUISHED bivectors (basis-relative, both orbits); kinetic = "
      f"one direction-locked quantum, axis-averaged")
print(f"  Higgs (mirror dir) : m4(t) = 3456 + 1536 t^2 + 64 t^4 integer-"
      f"exact k-flat; quadratic split 768 universal + 768 condensate "
      f"(4TrPhi^2); kinetic TIERED (0 at m6, c2 = {cs2:+.2f} at m8, "
      f"alias-free)")
print("  moment ladder      : m2/m4 topological | m6 curvature + gauge "
      "kinetic | m8 sigma kinetic")
print("  sector status      : THREE sectors INDUCED (volume, gauge, "
      "curvature); the sigma/Higgs sector is PROPAGATED for a posited "
      "external coupler -- the frozen algebra (A and JAJ^-1 alike) "
      "generates NO mirror-crossing one-forms")
print("  scorecard          : JAJ^-1 gauge-leg dressing = OPEN obligation "
      "E3 (priced 1 bit until executed); f2 not entire (route A only); "
      "cutoff distributional (staircase)")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- sector structure banked (panel-corrected); "
      "see erratum probe E2")
print("=" * 72)
sys.exit(0)
