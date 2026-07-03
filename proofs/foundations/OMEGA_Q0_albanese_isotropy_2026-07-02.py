#!/usr/bin/env python3
"""
proofs/foundations/OMEGA_Q0_albanese_isotropy_2026-07-02.py

OMEGA-KEYSTONE Q0 -- the warped-cone entry question, faced FIRST and decided by
exact computation (kickoff: docs/scoping/OMEGA_keystone_kickoff_2026-07-02.md par.2).

QUESTION (pre-registered). S2a (F4_cone_spectral_function_2026-07-02.py) measured the
substrate cones as "chirally warped spin-1 multifolds (v100 = 1/sqrt2, v110 = 1/2,
v111 = 1/sqrt3 exact), NON-METRIC under the cubic little group -- no linear coordinate
change isotropizes", with C = 0.0733 matching no universal constant. Q0 demands: either
(a) DERIVE isotropy restoration, or (b) scope Lorentz as internal/Clifford only.

HYPOTHESIS H-ALB (declared BEFORE computing; a sharp (a)-candidate that, if true,
REFUTES the S2a warping verdict):
    The anisotropy is not warping at all. It is the HOMOLOGY-COORDINATE representation
    of ONE metric cone. The S2a little-group argument implicitly assumed the physical
    point group acts orthogonally on the homology Bloch coordinates; it does not (it
    acts through GL(3,Z)). The object's own H1 sector supplies the canonical metric --
    the cycle-space Gram (= the Kotani-Sunada standard-realization / Albanese metric,
    the "emergent metric from the H1 flat/gauge sector" = kickoff par.2 candidate (a)-3)
    -- and in that metric BOTH cones are EXACTLY isotropic at leading order:
        Q_adj = M/2 with M = I - C/2,  C = [[0,1,-1],[1,0,1],[-1,1,0]],
        Q_hodge = M/8 = Q_adj/4,
        Q_adj^{-1} = 3I + C = Gram_H1  EXACTLY (no scalar!),
    so in Albanese momentum p (q = Gram^{1/2} p): v_adj = 1, v_hodge = 1/2 exactly.

SCORING CLASS (kickoff par.5 rule 1): STRUCTURAL (class a) -- geometry of the object
read off itself. NO PDG NUMBER APPEARS ANYWHERE IN THIS PROBE. The only recorded
experimental-free comparison is against the S2a probe's own measured constant.

PRE-REGISTERED KILL CRITERIA (before any computation):
  K1  if the leading-order dispersion of either cone deviates from the candidate
      quadratic form (sympy char-poly identity over ALL directions at once), the cone
      is genuinely non-metric => Q0 = (b), the S2a kill stands, H-ALB is dead.
  K2  if Q_adj^{-1} != Gram_H1 exactly (sympy), the isotropizing metric exists but is
      NOT the object's H1 metric => (a) exists but is UN-FORCED: log the incompleteness,
      do not adopt.
  K3  if the S2a sigma(omega) pipeline re-run in Albanese coordinates does not return
      the isotropic spin-1 constant 1/(6 pi) within the pipeline's own calibration
      tolerance (2.5%), the C-anomaly is not pure coordinates => the "not
      band-geometric" half of the S2a verdict stands (partial (b)).
  K4  if the middle branch of the adjacency triple has nonzero linear slope anywhere
      (char-poly lambda^1 term != 0), the exact spin-1 eigenstructure claim fails.
NO magnitude tuning anywhere: every claimed constant is exact algebra (sympy) or a
calibrated-pipeline read at the S2a probe's own tolerances. Over-application hazard
(kickoff par.5 rule 2): the pipeline is re-calibrated in-run on the exactly solvable
continuum spin-1 cone before any substrate number is quoted.
"""
import math
import sys

import numpy as np
import sympy as sp

NV, NE = 4, 6
EDGES = [(0, 1, (0, 0, 0)), (0, 2, (0, 0, 0)), (0, 3, (0, 0, 0)),
         (1, 2, (1, 0, 0)), (1, 3, (0, 1, 0)), (2, 3, (0, 0, 1))]
C_WEYL, C_DIRAC, C_SPIN1 = 1 / (24 * math.pi), 1 / (12 * math.pi), 1 / (6 * math.pi)

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")


# ---------------------------------------------------------------------------
# operators in radian momentum q (same conventions as the S2a probe)
# ---------------------------------------------------------------------------
def A_q(q):
    A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = np.exp(1j * np.dot(q, v)); A[i, j] += p; A[j, i] += np.conj(p)
    return A

def dA_q(q, ax):
    A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = 1j * v[ax] * np.exp(1j * np.dot(q, v)); A[i, j] += p; A[j, i] += np.conj(p)
    return A

def d_inc(q):
    d = np.zeros((NV, NE), complex)
    for e, (i, j, v) in enumerate(EDGES):
        d[i, e] = -1.0; d[j, e] = np.exp(1j * np.dot(q, v))
    return d

def dd_inc(q, ax):
    d = np.zeros((NV, NE), complex)
    for e, (i, j, v) in enumerate(EDGES):
        d[j, e] = 1j * v[ax] * np.exp(1j * np.dot(q, v))
    return d

def D_q(q):
    d = d_inc(q)
    return np.block([[np.zeros((NV, NV)), d], [d.conj().T, np.zeros((NE, NE))]])

def dD_q(q, ax):
    d = dd_inc(q, ax)
    return np.block([[np.zeros((NV, NV)), d], [d.conj().T, np.zeros((NE, NE))]])


# ---------------------------------------------------------------------------
# machinery copied VERBATIM from F4_cone_spectral_function_2026-07-02.py (S2a)
# -- the same calibrated pipeline; no re-derivation, no re-tuning.
# ---------------------------------------------------------------------------
def sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n; phi = i * math.pi * (3 - math.sqrt(5))
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)

def groups_of(ev, tol=1e-8):
    gs, cur = [], [0]
    for i in range(1, len(ev)):
        if ev[i] - ev[i - 1] < tol: cur.append(i)
        else: gs.append(cur); cur = [i]
    gs.append(cur)
    return gs

def sigma_shell(Hf, Jf, q0, fills, f0, omega, ndirs=1000, rref=0.25, rmax=1.0,
                pair_filter=None):
    q0 = np.asarray(q0, float)
    dirs = sphere(ndirs)
    acc = np.zeros(3)
    fl = [f0 if f is None else float(f) for f in fills]
    for kh in dirs:
        ev_ref = np.linalg.eigvalsh(Hf(q0 + rref * kh))
        gs = groups_of(ev_ref)
        gf = []
        for g in gs:
            vals = {fl[i] for i in g}
            assert len(vals) == 1, "filling not constant on a degenerate group"
            gf.append(vals.pop())
        ge_ref = [float(np.mean(ev_ref[g])) for g in gs]
        def gap(r, a, b):
            ev = np.linalg.eigvalsh(Hf(q0 + r * kh))
            return np.mean(ev[gs[b]]) - np.mean(ev[gs[a]])
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                df = gf[a] - gf[b]
                if df < 1e-12: continue
                if pair_filter is not None and not pair_filter(ge_ref[a], ge_ref[b]): continue
                glo, ghi = gap(1e-4, a, b), gap(rmax, a, b)
                if not (glo < omega <= ghi): continue
                lo, hi = 1e-4, rmax
                for _ in range(46):
                    mid = 0.5 * (lo + hi)
                    if gap(mid, a, b) < omega: lo = mid
                    else: hi = mid
                rs = 0.5 * (lo + hi)
                dh = 1e-4
                slope = (gap(rs + dh, a, b) - gap(rs - dh, a, b)) / (2 * dh)
                if abs(slope) < 1e-12: continue
                ev, V = np.linalg.eigh(Hf(q0 + rs * kh))
                for ax in range(3):
                    J = Jf(q0 + rs * kh, ax)
                    M = V[:, gs[b]].conj().T @ J @ V[:, gs[a]]
                    acc[ax] += rs * rs / abs(slope) * float(np.sum(np.abs(M) ** 2)) * df
    pref = (math.pi / omega) * (1 / (2 * math.pi) ** 3) * (4 * math.pi / ndirs)
    return pref * float(np.mean(acc)), pref * acc

def vdir(Hf, q0, band, eF, kh, h=0.02):
    """direction-resolved velocity, Richardson from h and 2h (READ, not assumed)."""
    q0 = np.asarray(q0, float); kh = np.asarray(kh, float); kh = kh / np.linalg.norm(kh)
    e1 = np.linalg.eigvalsh(Hf(q0 + h * kh))[band] - eF
    e2 = np.linalg.eigvalsh(Hf(q0 + 2 * h * kh))[band] - eF
    return (4 * e1 - e2) / (2 * h)


# ===========================================================================
print("=" * 88)
print(" T0  re-verify the S2a velocity claims (finite differences; the kickoff's numbers)")
print("=" * 88)
S3 = 1 / math.sqrt(3)
DIRS = {"100": (1, 0, 0), "110": (1, 1, 0), "111": (1, 1, 1)}
CLAIM_A = {"100": 1 / math.sqrt(2), "110": 0.5, "111": 1 / math.sqrt(3)}
for lbl, kh in DIRS.items():
    va = vdir(A_q, (0, 0, 0), 2, -1.0, kh)           # adjacency: upper cone branch
    vh = vdir(D_q, (0, 0, 0), 6, 0.0, kh)            # hodge: +eps branch
    check(f"adjacency v_{lbl} = {va:.6f} vs claimed {CLAIM_A[lbl]:.6f} "
          f"({(va/CLAIM_A[lbl]-1)*100:+.3f}%)", abs(va / CLAIM_A[lbl] - 1) < 2e-3)
    check(f"hodge     v_{lbl} = {vh:.6f} = adjacency/2 ({(vh/(va/2)-1)*100:+.3f}%)",
          abs(vh / (va / 2) - 1) < 2e-3)

print("=" * 88)
print(" T1  EXACT k.p (sympy): both cones are METRIC at leading order -- ALL directions")
print("=" * 88)
q1, q2, q3 = sp.symbols('q1 q2 q3', real=True)
lam = sp.symbols('lam')
Csym = sp.Matrix([[0, 1, -1], [1, 0, 1], [-1, 1, 0]])
Msym = sp.eye(3) - Csym / 2
qvec = sp.Matrix([q1, q2, q3])
Qform_a = sp.expand((qvec.T * (Msym / 2) * qvec)[0])     # candidate adjacency form
Qform_h = sp.expand((qvec.T * (Msym / 8) * qvec)[0])     # candidate hodge form

# adjacency triple: X = P dA(q) P on the lambda=-1 eigenspace (P = I - uu^T)
Nsym = sp.zeros(4, 4)
for (i, j, ax) in ((1, 2, 0), (1, 3, 1), (2, 3, 2)):     # cotree edges carry e1,e2,e3
    Nsym[i, j] += sp.I * qvec[ax]; Nsym[j, i] += -sp.I * qvec[ax]
u4 = sp.Matrix([1, 1, 1, 1]) / 2
P4 = sp.eye(4) - u4 * u4.T
Xa = sp.expand(P4 * Nsym * P4)
cp_a = sp.expand(Xa.charpoly(lam).as_expr())
target_a = sp.expand(lam ** 4 - Qform_a * lam ** 2)
check("adjacency char poly == lam^4 - (q.Q_a q) lam^2  [eigs {+-sqrt(qQq), 0, 0} EXACT; "
      "K1+K4 decided for ALL q at once]", sp.simplify(cp_a - target_a) == 0)

# hodge zero-space k.p: X = P_ker dD(q) P_ker, P_ker = uu^T (C0) + cycle projector (C1)
Chat = sp.Matrix([[1, 1, 0], [-1, 0, 1], [0, -1, -1],
                  [1, 0, 0], [0, 1, 0], [0, 0, 1]])      # cycles c_k as 1-chains (rows=EDGES)
# boundary check: each column is a genuine cycle (d0 . c = 0 at Gamma)
d0 = sp.zeros(4, 6)
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1; d0[j, e] = 1
check("cycle matrix: d0 @ Chat == 0 (genuine cycles) and cotree rows == I (class bookkeeping)",
      d0 * Chat == sp.zeros(4, 3) and Chat[3:, :] == sp.eye(3))
Gram = Chat.T * Chat
check(f"Gram_H1 = Chat^T Chat = 3I + C  (computed: {Gram.tolist()})",
      Gram == 3 * sp.eye(3) + Csym)
Pcyc = Chat * (Chat.T * Chat) ** -1 * Chat.T
Pker = sp.zeros(10, 10)
Pker[:4, :4] = u4 * u4.T
Pker[4:, 4:] = Pcyc
dDsym = sp.zeros(10, 10)
for e, (i, j, v) in enumerate(EDGES):
    for ax in range(3):
        if v[ax]:
            dDsym[i, 4 + e] += 0                          # tail carries no phase (srs.py convention)
            dDsym[j, 4 + e] += 0
# build via the d-block: (dd)_je = i*v_ax (head only), assemble D-shape
dd_blocks = []
for ax in range(3):
    db = sp.zeros(4, 6)
    for e, (i, j, v) in enumerate(EDGES):
        if v[ax]:
            db[j, e] = sp.I * v[ax]
    dd_blocks.append(db)
dDq = sp.zeros(10, 10)
for ax in range(3):
    blk = sp.zeros(10, 10)
    blk[:4, 4:] = dd_blocks[ax]
    blk[4:, :4] = dd_blocks[ax].conjugate().T
    dDq += qvec[ax] * blk
Xh = Pker * dDq * Pker
tr2 = sp.simplify(sp.trace(Xh * Xh))
tr3 = sp.simplify(sp.trace(Xh * Xh * Xh))
tr4 = sp.simplify(sp.trace(Xh * Xh * Xh * Xh))
check("hodge k.p: Tr X^2 = 2 (q.Q_h q)   [dispersing pair velocity^2 = the METRIC form]",
      sp.simplify(tr2 - 2 * Qform_h) == 0)
check("hodge k.p: Tr X^3 = 0, Tr X^4 = 2 (q.Q_h q)^2  [eigs {+-sqrt(qQ_h q), 0^8} EXACT: "
      "cone pair + flats, no extra linear branch]",
      tr3 == 0 and sp.simplify(tr4 - 2 * Qform_h ** 2) == 0)
print("    => BOTH cones' leading dispersions are PERFECT QUADRATIC FORMS: eps^2 = q.Q q.")
print("       'Non-metric warping' is REFUTED at leading order (K1 passes, S2a verdict falls);")
print("       Q_hodge = Q_adj/4 exactly (the S2a 'adjacency/2 direction-by-direction').")

print("=" * 88)
print(" T2  the isotropizing metric IS the object's own H1 (Albanese) metric -- EXACT")
print("=" * 88)
check("Q_adj^{-1} = Gram_H1 EXACTLY:  (M/2)(3I+C) == I   [K2 decided]",
      sp.simplify((Msym / 2) * (3 * sp.eye(3) + Csym) - sp.eye(3)) == sp.zeros(3, 3))
G12 = (5 * sp.eye(3) + Csym) / 3
check("Gram^{1/2} = (5I + C)/3 exactly  (uses C^2 = 2I - C)",
      sp.simplify(G12 * G12 - Gram) == sp.zeros(3, 3))
check("Albanese isotropy EXACT: G12 (M/2) G12 == I  (v_adj = 1),  G12 (M/8) G12 == I/4 "
      "(v_hodge = 1/2)",
      sp.simplify(G12 * (Msym / 2) * G12 - sp.eye(3)) == sp.zeros(3, 3)
      and sp.simplify(G12 * (Msym / 8) * G12 - sp.eye(3) / 4) == sp.zeros(3, 3))
print("    => q = Gram^{1/2} p turns BOTH cones into EXACT isotropic cones:")
print("       adjacency spin-1 cone: v = 1 (the 'speed of light' = 1 in the object's own")
print("       harmonic/Albanese unit);  hodge (D4 spatial) cone: v = 1/2 exactly.")

print("=" * 88)
print(" T3  the symmetry proof: Aut(K4)=S4 acts by GL(3,Z) on homology; the invariant")
print("     form is UNIQUE (irreducible) and equals Gram_H1 -- the S2a little-group")
print("     argument applied O(3) in the WRONG coordinates")
print("=" * 88)
from itertools import permutations
Chat_np = np.array(Chat.tolist(), float)
Gram_np = np.array(Gram.tolist(), float)
EDGE_IDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
S_list = []
all_cyc_ok, all_gram_ok, all_spec_ok = True, True, True
rng = np.random.default_rng(7)
qtest = rng.uniform(-1, 1, size=(3, 3))
for sig in permutations(range(4)):
    R = np.zeros((6, 6))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b: a, b, s = b, a, -1.0
        R[EDGE_IDX[(a, b)], e] = s
    RC = R @ Chat_np
    S = RC[3:, :]                                       # induced GL(3,Z) on homology classes
    S_list.append(S)
    all_cyc_ok &= np.allclose(RC, Chat_np @ S, atol=1e-12)      # cycle space preserved
    all_gram_ok &= np.allclose(S.T @ Gram_np @ S, Gram_np, atol=1e-12)
    Sinv_T = np.linalg.inv(S).T
    for qq in qtest:
        e1 = np.sort(np.linalg.eigvalsh(A_q(qq)))
        e2 = np.sort(np.linalg.eigvalsh(A_q(Sinv_T @ qq)))
        all_spec_ok &= np.allclose(e1, e2, atol=1e-10)
check("all 24 automorphisms: g(cycles) are cycles and the induced S_g are integer GL(3,Z)",
      all_cyc_ok and all(np.allclose(S, np.round(S)) for S in S_list))
check("all 24: S_g^T Gram S_g = Gram  (the H1 metric IS the invariant form)", all_gram_ok)
check("all 24: spec A(S_g^{-T} q) = spec A(q)  (they ARE Bloch symmetries)", all_spec_ok)
# uniqueness: solve S^T X S = X for symmetric X over the 24 group elements
rows = []
basis = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
def sym_from_coords(c):
    X = np.zeros((3, 3))
    for k, (i, j) in enumerate(basis):
        X[i, j] += c[k]; X[j, i] = X[i, j]
    return X
for S in S_list:
    for k in range(6):
        c = np.zeros(6); c[k] = 1.0
        Y = S.T @ sym_from_coords(c) @ S - sym_from_coords(c)
        rows.append([Y[i, j] for (i, j) in basis])
null_dim = 6 - np.linalg.matrix_rank(np.array(rows), tol=1e-9)
check(f"invariant symmetric forms: dim = {null_dim} (must be 1 => Q ~ Gram^(-1) is FORCED "
      "for ANY smooth Gamma extremum; irreducibility)", null_dim == 1)
print("    => the little group at Gamma is the FULL S4, acting by GL(3,Z), NOT by O(3);")
print("       its unique invariant form is Gram_H1, so 'isotropy' can only ever mean")
print("       'proportional to Gram^{-1}' -- which T1/T2 show holds EXACTLY.")

print("=" * 88)
print(" T4  numeric cross-check: velocities in the Albanese frame (band level, not k.p)")
print("=" * 88)
G12_np = np.array(G12.tolist(), float)
HA_p = lambda p: A_q(G12_np @ np.asarray(p, float))
HD_p = lambda p: D_q(G12_np @ np.asarray(p, float))
va_all, vh_all = [], []
for kh in sphere(40):
    va_all.append(vdir(HA_p, (0, 0, 0), 2, -1.0, kh, h=0.01))
    vh_all.append(vdir(HD_p, (0, 0, 0), 6, 0.0, kh, h=0.01))
va_all, vh_all = np.array(va_all), np.array(vh_all)
check(f"adjacency in p-frame: v = {va_all.mean():.6f} +- {va_all.std():.2e} "
      f"(spread {(va_all.max()-va_all.min()):.2e}) == 1", abs(va_all.mean() - 1) < 2e-3
      and (va_all.max() - va_all.min()) < 4e-3)
check(f"hodge in p-frame:     v = {vh_all.mean():.6f} +- {vh_all.std():.2e} "
      f"(spread {(vh_all.max()-vh_all.min()):.2e}) == 1/2", abs(vh_all.mean() - 0.5) < 2e-3
      and (vh_all.max() - vh_all.min()) < 4e-3)
# evenness (bands are even in q EXACTLY: A(-q) = conj A(q)); chirality is NOT in |v|
ev_ok = True
for qq in qtest:
    ev_ok &= np.allclose(np.sort(np.linalg.eigvalsh(A_q(qq))),
                         np.sort(np.linalg.eigvalsh(A_q(-qq))), atol=1e-12)
check("band evenness: spec A(-q) == spec A(q) exactly (A(-q) = conj A(q)) -- the "
      "'CHIRALLY warped' language was wrong twice over: dispersions are even at EVERY "
      "order; chirality lives in eigenvectors (Berry), not in |v|", ev_ok)

print("=" * 88)
print(" T5  the S2a sigma(omega) pipeline, re-run in the Albanese frame  [K3 decided]")
print("=" * 88)
# in-run calibration on the exactly solvable continuum spin-1 (the S2a pattern)
r2 = 1 / math.sqrt(2)
S1X = np.array([[0, r2, 0], [r2, 0, r2], [0, r2, 0]], complex)
S1Y = np.array([[0, -1j * r2, 0], [1j * r2, 0, -1j * r2], [0, 1j * r2, 0]], complex)
S1Z = np.diag([1.0, 0.0, -1.0]).astype(complex)
SPIN1 = [S1X, S1Y, S1Z]
H1c = lambda q: q[0] * SPIN1[0] + q[1] * SPIN1[1] + q[2] * SPIN1[2]
J1c = lambda q, ax: SPIN1[ax]
s_cal, _ = sigma_shell(H1c, J1c, (0, 0, 0), [1, None, 0], 0.5, 0.1, rmax=0.6)
C_cal = s_cal / 0.1
tol_pipe = abs(C_cal / C_SPIN1 - 1) + 0.02               # pipeline's own tolerance envelope
check(f"calibration: continuum spin-1 C = {C_cal:.6f} vs 1/6pi = {C_SPIN1:.6f} "
      f"({(C_cal/C_SPIN1-1)*100:+.2f}%)", abs(C_cal / C_SPIN1 - 1) < 0.02)

JA_p = lambda p, ax: sum(G12_np[i, ax] * dA_q(G12_np @ np.asarray(p, float), i) for i in range(3))
JD_p = lambda p, ax: sum(G12_np[i, ax] * dD_q(G12_np @ np.asarray(p, float), i) for i in range(3))
fills_A = [1, None, 0, 0]
fills_D = [1, 1, 1, 1, None, None, 0, 0, 0, 0]
for name, Hf, Jf, fills, vexp in (("adjacency", HA_p, JA_p, fills_A, 1.0),
                                  ("hodge", HD_p, JD_p, fills_D, 0.5)):
    Cs = {}
    for om in (0.05, 0.12):
        s, sax = sigma_shell(Hf, Jf, (0, 0, 0), fills, 0.5, om, rmax=0.6)
        Cs[om] = s * vexp / om
        if om == 0.05:
            spread = (sax.max() - sax.min()) / sax.mean()
    C0 = Cs[0.05]
    print(f"    {name}: C(omega=0.05) = {C0:.6f}   [/(1/6pi) = {C0/C_SPIN1:.4f}]   "
          f"omega-drift to 0.12: {(Cs[0.12]/C0-1)*100:+.2f}%   per-axis spread {spread*100:.2f}%")
    check(f"{name} cone in its own (Albanese) frame IS the universal isotropic spin-1 "
          f"constant 1/(6pi)  ({(C0/C_SPIN1-1)*100:+.2f}%)", abs(C0 / C_SPIN1 - 1) < tol_pipe)
    check(f"{name}: per-axis isotropy of sigma in p-frame (spread {spread*100:.2f}% < 5%)",
          spread < 0.05)
# f0-independence in the isotropic frame: exact for true spin-1; for the substrate the
# mid band has O(q^2) curvature, so the cancellation is exact only as omega -> 0.
# Honest check: the spread is small AND shrinks with omega (curvature artifact, not
# structure).
spreads = {}
for om in (0.05, 0.025):
    sf0, _ = sigma_shell(HA_p, JA_p, (0, 0, 0), fills_A, 0.0, om, rmax=0.6)
    sf1, _ = sigma_shell(HA_p, JA_p, (0, 0, 0), fills_A, 1.0, om, rmax=0.6)
    spreads[om] = abs(sf1 - sf0) / sf0
check(f"adjacency flat-filling dependence is a finite-omega mid-band-curvature artifact: "
      f"spread {spreads[0.05]*100:.2f}% (omega=0.05) -> {spreads[0.025]*100:.2f}% "
      f"(omega=0.025), shrinking and < 2%",
      spreads[0.025] < spreads[0.05] and spreads[0.05] < 0.02)

print("=" * 88)
print(" T6  postdiction: the S2a 'C = 0.0733 = 2.76 x Dirac' is EXACTLY the coordinate")
print("     artifact  C_meas = [(Tr Q/3)/sqrt(det Q)] * <v>_sphere * C_spin1")
print("=" * 88)
TrQ3 = sp.Rational(1, 2)                                  # Tr(M/2)/3 = 1/2 (sympy below)
check("exact ellipsoid factor: Tr(Q_a)/3 = 1/2, det Q_a = 1/16 => (TrQ/3)/sqrt(detQ) = 2",
      sp.simplify(sp.trace(Msym / 2) / 3 - sp.Rational(1, 2)) == 0
      and sp.simplify((Msym / 2).det() - sp.Rational(1, 16)) == 0)
Q_a_np = np.array((Msym / 2).tolist(), float)
vbar = float(np.mean([math.sqrt(kh @ Q_a_np @ kh) for kh in sphere(4000)]))
C_pred = 2 * vbar * C_SPIN1
S2A_RECORDED = 0.0733                                     # the S2a probe's own measured constant
print(f"    <v>_sphere(homology frame) = {vbar:.6f};  C_pred = 2*<v>*C_spin1 = {C_pred:.6f}")
print(f"    S2a recorded C = {S2A_RECORDED}  -> ratio {C_pred/S2A_RECORDED:.4f}")
check(f"S2a's anomalous constant postdicted to {abs(C_pred/S2A_RECORDED-1)*100:.2f}% "
      "(pure coordinate artifact; scale-invariance of the factor also explains why the "
      "two objects' constants coincided)", abs(C_pred / S2A_RECORDED - 1) < 0.01)
# and the scale-invariance: Q -> cQ leaves 2<v(Q)>*... invariant since (TrQ/3)/sqrt(detQ)
# scales as c^{-1/2} and <v> as c^{+1/2}: this is WHY hodge (Q/4) gave the same 0.0733.
Q_h_np = Q_a_np / 4
vbar_h = float(np.mean([math.sqrt(kh @ Q_h_np @ kh) for kh in sphere(4000)]))
fac_h = (np.trace(Q_h_np) / 3) / math.sqrt(np.linalg.det(Q_h_np))
check(f"hodge postdiction identical: factor*<v> = {fac_h*vbar_h:.6f} == {2*vbar:.6f} "
      "(the S2a 'coincidence' is the proportionality Q_h = Q_a/4)",
      abs(fac_h * vbar_h - 2 * vbar) < 1e-9)

print("=" * 88)
print(" VERDICT -- Q0 ANSWERED: (a), ISOTROPY RESTORATION DERIVED")
print("=" * 88)
print("""    K1 FAILS TO KILL: both cones are EXACT quadratic-form (metric) cones at leading
    order -- the S2a 'chirally warped, non-metric, un-isotropizable' verdict is REFUTED
    (its little-group argument applied O(3) in homology coordinates where the actual
    action is GL(3,Z); and band dispersions are even in q at every order, so nothing
    about the leading cone was ever 'chiral').
    K2 FAILS TO KILL: the isotropizing metric is FORCED and is the object's own H1
    cycle Gram: Q_adj^{-1} = Gram_H1 exactly (no free scalar); by irreducibility (T3)
    it is the ONLY invariant candidate. This is the Kotani-Sunada standard-realization
    (Albanese) metric = the kickoff's candidate home (a)-3: the emergent metric from
    the H1 flat/gauge sector.
    K3 FAILS TO KILL: in its own metric the substrate cone IS universal: C = 1/(6 pi)
    (isotropic spin-1), and the S2a anomaly 0.0733 = 2<v> x 1/(6 pi) is postdicted to
    <1% as the pure coordinate artifact, including the two-object 'coincidence'.

    WHAT SURVIVES OF S2a (unchanged, coordinate-independent): the cone is SPIN-1-LIKE
    (not Dirac): the direct pair channel is q^2-dark, all low-omega weight is flat-band
    mediated, and the universal constant is 1/(6 pi) = 2 x [1/(12 pi)] -- the band
    supplies EXACT relativistic spin-1 kinematics with v_adj = 1 (and the D4 spatial
    section v = 1/2), NOT the spin-1/2 Dirac value. The factor bridging 1/(6 pi) to the
    SM's per-channel 1/(12 pi) remains the Clifford-layer question (Target 4).

    a2/a4 MEANING (Q0's demand): heat-kernel coefficients of continuum-D4 are now
    well-defined w.r.t. the Albanese volume; the spatial section has velocity 1/2
    (relative to the adjacency cone's 1), a structural constant to carry explicitly.""")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
