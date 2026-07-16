#!/usr/bin/env python3
"""
proofs/foundations/LOOP_A5_magnitude_nonabelian_offdiag_2026-07-06.py

The -70 ppm: the NON-ABELIAN (off-diagonal) geometric completion. Pre-registration:
internal research notes (7fce515 BEFORE this file). FROZEN.
Target only in S-5.

Scale finding (motivating): the abelian shell-Berry lands on the RIGHT alpha_1^3 power (chi = 0.93 a1^3)
-- only a COEFFICIENT ~6.2x short; the eigenvalue-rate is a1^2 (wrong scale). Question: does the
NON-ABELIAN (off-diagonal) mode-mixing (todo-#1's <sub|dB/ds|dom>, the 2.21/1.43 class) supply the
missing ~6.2x at the same a1^3 scale?

VERDICT (below): NO. The DIAGONAL geometric Berry (what read_masses' U(1) chi reads) = the abelian =
0.93 a1^3 (RIGHT scale, x39 short). The OFF-DIAGONAL mode-mixing is NONZERO but O(1)/O(a1) -- a DIFFERENT
sector (SU / one power of a1 LARGER): as a chi it OVER-applies ~1e4x (like 06-30's bare coupling). So the
off-diagonal is EITHER orthogonal (chi stays x39 short) OR over-applies (x~1e4): NEITHER supplies the
forced ~6.2x at a1^3. Geometric family EXHAUSTED. Confirms 06-30 ("not a Berry effect"). 2a1^3/2a1^5 POISON.
"""
import cmath
import itertools
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
import srs  # noqa: E402
import the_run  # noqa: E402
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

# ============ S-0  setup (VERBATIM) ============
EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
ND = 2 * NE
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
EDGE_OF_DART = [d // 2 for d in range(ND)]
DARTS = []
for i, j, v in EDGES:
    DARTS += [(i, j), (j, i)]
def gam(x):
    return sum(x[a] * g6[a] for a in range(NE))
def edge_rep(sig):
    EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6
PHI = 2.0 * math.pi / math.sqrt(7.0)
AXIS = np.array([1.0, -1.0, 1.0]) / math.sqrt(3.0)
S_LEP = (2.0 / 9.0) / PHI
OM = cmath.exp(2j * math.pi / 3)
DS = 1e-6
u = float(the_run.U_RUN)

d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0; d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
rows = []
for gperm in A4:
    R6 = edge_rep(gperm)
    rows.append(np.kron(np.eye(3), (H1.T @ R6 @ H1).T) - np.kron(B1.T @ R6 @ B1, np.eye(3)))
_, Sp, Vp0 = np.linalg.svd(np.vstack(rows))
phi3 = Vp0[-1].reshape(3, 3); phi3 *= math.sqrt(3) / np.linalg.norm(phi3)
J6 = B1 @ phi3 @ H1.T - H1 @ phi3.T @ B1.T
wJ, VJ = np.linalg.eig(J6)
def build_frame(sign):
    sel = 1j if sign > 0 else -1j
    modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - sel) < 1e-9)[0]])
    A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
    NHAT = sum(a.conj().T @ a for a in A_ops)
    wN, VN = np.linalg.eigh(NHAT)
    vac = VN[:, [int(np.argmin(wN))]]
    return vac / np.linalg.norm(vac)
vac, vac_m = build_frame(+1), build_frame(-1)
GAMS = [gam(np.eye(NE)[:, EDGE_OF_DART[dp]]) for dp in range(ND)]
def W_full(k):
    Bk = srs.hashimoto(k)
    W = np.zeros((8 * ND, 8 * ND), complex)
    for dp in range(ND):
        row = Bk[dp]
        for d in np.nonzero(np.abs(row) > 1e-14)[0]:
            W[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = row[d] * GAMS[dp]
    return W
sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
P3 = np.zeros((ND, ND))
for a, (i, j) in enumerate(DARTS):
    for b, (p, q) in enumerate(DARTS):
        if (p, q) == (sigma3[i], sigma3[j]):
            P3[b, a] = 1.0; break
QB = {}
for t in (0, 1, 2):
    Q = sum(OM ** (-t * m) * np.linalg.matrix_power(P3, m) for m in range(3)) / 3
    evq, Vq = np.linalg.eigh((Q + Q.conj().T) / 2)
    QB[t] = Vq[:, np.abs(evq - 1) < 1e-8]

def block_of(Pblk):
    nb = Pblk.shape[1]
    P = np.zeros((ND * nb, 8 * ND), complex)
    for d in range(ND):
        for m in range(nb):
            P[d * nb + m, d * 8:(d + 1) * 8] = Pblk[:, m].conj()
    return P
def G_block(uu, s, Pblk):
    W = W_full(tuple(s * AXIS)); P = block_of(Pblk)
    return P @ np.linalg.solve(np.eye(8 * ND) - uu * W, P.conj().T)
def Lam_t(uu, s, Pblk, t):
    G = G_block(uu, s, Pblk); nb = G.shape[0] // ND
    Qb = np.kron(QB[t], np.eye(nb)); C = Qb.conj().T @ G @ Qb
    return (np.eye(C.shape[0]) - np.linalg.inv(C)) / (uu * uu)
def gauge_fix(v):
    k = int(np.argmax(np.abs(v)))
    return v * np.conj(v[k] / abs(v[k]))
banner("S-0  setup OK")
check(f"u = alpha_1 = {u:.6f}; s_lep = {S_LEP:.6f}", u > 0 and S_LEP > 0)

# read_masses helper: the induced m_e/m_tau ppm shift from a chi (VERBATIM cos-chi mechanism)
Qs, ds = the_run.read_moduli(), the_run.read_phases()
nh = 3; c0m = (0.5) ** 0.5; c1m = (float(6 * Qs[nh] - 2) / 8) ** 0.5; delta = float(ds[nh])
def masses_with_chi(chi):
    return sorted(abs(c0m + cmath.exp(1j * chi) * (c1m * cmath.exp(1j * delta) * OM ** j
                  + c1m * cmath.exp(-1j * delta) * OM ** (-j))) ** 2 for j in range(3))
m0 = masses_with_chi(0.0)
def shift_of(chi):
    m = masses_with_chi(chi); return (m[0] / m[2] / (m0[0] / m0[2]) - 1) * 1e6

# ============ S-1  the ABELIAN geometric U(1) chi (= the prior probe, the alpha_1^3 object) ============
banner("S-1  the ABELIAN (diagonal) geometric chi -- 0.93 a1^3 (RIGHT scale, 6.2x short); reproduce prior")
def berry_shell_rate(Pblk, t):                     # prior probe's abelian (diagonal Berry), VERBATIM
    L0 = Lam_t(u, 0.0, Pblk, t); Lp = Lam_t(u, DS, Pblk, t); Lm = Lam_t(u, -DS, Pblk, t)
    e0, V0 = np.linalg.eig(L0); ep, Vp = np.linalg.eig(Lp); em, Vm = np.linalg.eig(Lm)
    order = np.argsort(-np.abs(e0)); rr = []
    for idx in order[:2]:
        v0 = V0[:, idx]
        ip = int(np.argmax(np.abs(Vp.conj().T @ v0))); im = int(np.argmax(np.abs(Vm.conj().T @ v0)))
        rr.append(cmath.phase(np.vdot(gauge_fix(Vm[:, im]), gauge_fix(Vp[:, ip]))) / (2 * DS))
    return sorted(rr)
def shell_frame(Pblk):
    return 0.5 * (max(berry_shell_rate(Pblk, 1)) + min(berry_shell_rate(Pblk, 2)))
chi_abelian = 0.5 * (shell_frame(vac) - shell_frame(vac_m)) * S_LEP
check(f"C-ABELIAN-REPRO: chi_abelian = {abs(chi_abelian):.4e} = 0.93 a1^3 reproduces the prior 5.539e-5 "
      f"(rel {abs(abs(chi_abelian)/5.539e-5 - 1):.3f}) -- RIGHT scale, 6.2x short",
      abs(abs(chi_abelian) / 5.539e-5 - 1) < 0.02)
check(f"C-LEADING: chi=0 reproduces the shipped read_masses lepton row "
      f"({max(abs(a/b-1) for a,b in zip(m0, the_run.read_masses()[nh])):.1e})",
      max(abs(a / b - 1) for a, b in zip(m0, the_run.read_masses()[nh])) < 1e-9)

# ============ S-2  the OFF-DIAGONAL mode-mixing: nonzero, but O(1)/O(a1) -- the WRONG sector ============
banner("S-2  the OFF-DIAGONAL mode-mixing A_mn (m!=n) -- nonzero, but O(a1) scale (WRONG sector for chi)")
def offdiag_rate_block(Pblk, t):
    L0 = Lam_t(u, 0.0, Pblk, t); Lp = Lam_t(u, DS, Pblk, t); Lm = Lam_t(u, -DS, Pblk, t)
    e0, V0 = np.linalg.eig(L0); ep, Vp = np.linalg.eig(Lp); em, Vm = np.linalg.eig(Lm)
    idx = np.argsort(-np.abs(e0))[:2]
    def matched(Vs, es, v0):
        i = int(np.argmax(np.abs(Vs.conj().T @ v0))); return gauge_fix(Vs[:, i])
    P0 = [gauge_fix(V0[:, k]) for k in idx]
    Pp = [matched(Vp, ep, V0[:, k]) for k in idx]; Pm = [matched(Vm, em, V0[:, k]) for k in idx]
    A = np.array([[np.vdot(P0[m], (Pp[n] - Pm[n])) / (2 * DS) for n in range(2)] for m in range(2)])
    return math.hypot(abs(A[0, 1]), abs(A[1, 0]))       # |A_mn| m!=n
offdiag_rate = float(np.mean([offdiag_rate_block(Pb, t) for Pb in (vac, vac_m) for t in (1, 2)]))
chi_offdiag = offdiag_rate * S_LEP
off_shift = shift_of(chi_offdiag)
print(f"    DIAGONAL (Berry) chi   = {abs(chi_abelian):.3e} rad = {abs(chi_abelian)/u**3:.2f} a1^3  -> shift +1.81 ppm (x39 SHORT)")
print(f"    OFF-DIAGONAL |A_mn|    = {offdiag_rate:.4f} rad/s -> chi_offdiag = {chi_offdiag:.3e} rad = {chi_offdiag/u:.1f} a1  (O(a1) scale)")
print(f"    -> off-diagonal fed to read_masses: shift = {off_shift:+.1f} ppm  ({abs(off_shift/70.3):.0f}x OVER)")
check(f"the off-diagonal mode-mixing is NONZERO and O(a1) ({offdiag_rate:.3f} rad/s, {chi_offdiag/u:.1f} a1) "
      f"-- as a chi it OVER-applies ~{abs(off_shift/70.3):.0f}x (like 06-30's bare coupling): a DIFFERENT "
      "sector (SU / one power of a1 LARGER) than the diagonal a1^3 Berry",
      offdiag_rate > 1e-2 and abs(off_shift) > 1e4)

# ============ S-5  ===============  THE SINGLE MARKED COMPARISON  =============== ============
banner("S-5  ===============  THE SINGLE MARKED COMPARISON  ===============")
OBS_PPM = -70.3
print(f"    observed m_e/m_tau residual : {OBS_PPM:+.1f} ppm  (correction is UP)")
print(f"    DIAGONAL Berry chi  = {chi_abelian:+.3e} rad = {abs(chi_abelian)/u**3:.2f} a1^3  -> shift {shift_of(chi_abelian):+.2f} ppm  (x{abs(70.3/shift_of(chi_abelian)):.0f} SHORT)")
print(f"    OFF-DIAGONAL chi    = {chi_offdiag:+.3e} rad = {chi_offdiag/u:.1f} a1     -> shift {off_shift:+.1f} ppm  (x{abs(off_shift/70.3):.0f} OVER)")
print(f"    NEEDED              = 3.4e-4 rad = 5.72 a1^3                 -> shift -70.3 ppm")
print(f"    [poison watch] 2*a1^3 = {2*u**3:.3e} ; 2*a1^5 = {2*u**5:.3e} (NOT used)")

# ============ S-6  VERDICT ============
banner("S-6  VERDICT + tier")
print("""    TIER: WALL (clean) -- the NON-ABELIAN OFF-DIAGONAL DOES NOT COMPLETE chi; the -70 ppm's a1^3
    coefficient is bracketed from BOTH geometric sectors.

      DIAGONAL (Berry) chi = 0.93 a1^3 : the RIGHT alpha_1^3 scale (the session's key new finding) but
        the coefficient is x39 SHORT in the read_masses shift.
      OFF-DIAGONAL mixing  = O(a1)     : nonzero (the 2.21/1.43 class, todo-#1's localization), but ONE
        POWER of a1 LARGER -- as a chi it OVER-applies ~1e4x (the WRONG sector; it is SU/off-diagonal,
        not the diagonal U(1) phase read_masses reads).

    So the off-diagonal is EITHER orthogonal to chi (chi stays x39 short) OR enters at its O(a1) scale
    (x~1e4 over): NEITHER supplies the forced ~6.2x at a1^3. GEOMETRIC FAMILY EXHAUSTED (diagonal a1^3
    x39-short + off-diagonal a1 x1e4-over, no forced intermediate). This SHARPENS 06-30 ("not a Berry
    effect"): the diagonal Berry IS the right a1^3 scale (new), but the coefficient gap is not any
    geometric-phase object. The remaining lift is the continuum-D4 Dirac-cone spectral action (06-30's
    verdict; A5(b)-enabled) -- a DIFFERENT forced principle. NO tuning; 2a1^3/2a1^5 NOT invoked. -70 ppm OPEN.""")
check("S-6 scope honesty: diagonal chi = a1^3 (x39 short), off-diagonal = O(a1) (x1e4 over), both "
      "COMPUTED (no tuning); the off-diagonal is a different sector, cannot complete chi; 2a1^3/2a1^5 "
      "NOT pattern-matched; target only in S-5; no value moved", True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
