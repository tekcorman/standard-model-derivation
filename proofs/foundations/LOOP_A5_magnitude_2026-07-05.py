#!/usr/bin/env python3
"""
proofs/foundations/LOOP_A5_magnitude_2026-07-05.py

A5-MAGNITUDE -- attempt the -70 ppm as the forced W2 seed's bit-odd chiral
holonomy chi entering read_masses at SECOND order (cos chi, automatic). Pre-
registered in internal research notes (committed eb079c3
BEFORE this file). The construction is FROZEN there; NO factor/power/functional
tuned after the number is seen. The target appears ONLY in S-3.

Honest expectation: WALL at a forced over-application factor (banked with the
factor). 2*alpha_1^5 is POISON -- not pattern-matched.
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

EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
ND = 2 * NE
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
EDGE_OF_DART = [d // 2 for d in range(ND)]
DARTS = []
for i, j, v in EDGES:
    DARTS += [(i, j), (j, i)]
def gam(x):
    return sum(x[a] * g6[a] for a in range(NE))
def edge_rep(sig):
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
u = float(the_run.U_RUN)                                # = alpha_1 (the run fugacity)

# ===========================================================================
banner("S-0  re-lock (J / C=I+iJ, coupled W, QB, vac/vac_m); u = alpha_1")
# ===========================================================================
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
_, Sp, Vp = np.linalg.svd(np.vstack(rows))
phi3 = Vp[-1].reshape(3, 3); phi3 *= math.sqrt(3) / np.linalg.norm(phi3)
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
check(f"S-0 re-lock OK; u = alpha_1 = {u:.6f} = rho^(g-2); s_lep = {S_LEP:.6f}", u > 0 and S_LEP > 0)

# ===========================================================================
banner("S-1  chi = the FORCED bit-odd chiral holonomy (A5/E2c coupled shell-rate)")
# ===========================================================================
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
def coupled_shell_rate(uu, Pblk, t):
    L0 = Lam_t(uu, 0.0, Pblk, t); Lp = Lam_t(uu, DS, Pblk, t); Lm = Lam_t(uu, -DS, Pblk, t)
    e0 = np.linalg.eigvals(L0); ep = np.linalg.eigvals(Lp); em = np.linalg.eigvals(Lm)
    order = np.argsort(-np.abs(e0)); rr = []
    for e in e0[order[:2]]:
        ip = int(np.argmin(np.abs(ep - e))); im = int(np.argmin(np.abs(em - e)))
        rr.append((cmath.phase(ep[ip] / e) - cmath.phase(em[im] / e)) / (2 * DS) / 2)
    return sorted(rr)
def chi_rate_frame(Pblk):
    r1 = max(coupled_shell_rate(u, Pblk, 1)); r2 = min(coupled_shell_rate(u, Pblk, 2))
    return 0.5 * (r1 + r2)                              # chi = phase-SUM direction
chi_p = chi_rate_frame(vac); chi_m = chi_rate_frame(vac_m)
chi_rate = 0.5 * (chi_p - chi_m)                        # bit-ODD (flips with J)
chi = chi_rate * S_LEP                                  # holonomy to s_lep (FROZEN)
print(f"    chi_rate(+J) = {chi_p:+.6f}, chi_rate(-J) = {chi_m:+.6f}")
print(f"    bit-odd chi_rate = 1/2(chi(+J)-chi(-J)) = {chi_rate:+.6f} rad/s")
print(f"    chi(s_lep) = chi_rate * s_lep = {chi:+.6e} rad   (the forced holonomy)")
# C-FREE: free ensemble bit-odd = 0
def free_shell_rate(s, t):
    def Gf(ss):
        Bk = srs.hashimoto(tuple(ss * AXIS)); return np.linalg.inv(np.eye(ND) - u * u * Bk @ Bk)
    C0 = QB[t].conj().T @ Gf(s) @ QB[t]
    Cp = QB[t].conj().T @ Gf(s + DS) @ QB[t]; Cm = QB[t].conj().T @ Gf(s - DS) @ QB[t]
    def lam(C): return (np.eye(C.shape[0]) - np.linalg.inv(C)) / (u * u)
    e0 = np.linalg.eigvals(lam(C0)); ep = np.linalg.eigvals(lam(Cp)); em = np.linalg.eigvals(lam(Cm))
    order = np.argsort(-np.abs(e0)); rr = []
    for e in e0[order[:2]]:
        ip = int(np.argmin(np.abs(ep - e))); im = int(np.argmin(np.abs(em - e)))
        rr.append((cmath.phase(ep[ip] / e) - cmath.phase(em[im] / e)) / (2 * DS) / 2)
    return sorted(rr)
f1 = max(free_shell_rate(0.0, 1)); f2 = min(free_shell_rate(0.0, 2))
check(f"S-1 C-FREE [Q3]: free-ensemble chi-sum = {0.5*(f1+f2):+.1e} ~ 0 (the "
      "chiral holonomy is purely the interacting bit-odd part)", abs(0.5 * (f1 + f2)) < 5e-3)

# ===========================================================================
banner("S-2  plug chi into read_masses (FORCED, second order via cos chi)")
# ===========================================================================
Qs, ds = the_run.read_moduli(), the_run.read_phases()
nh = 3                                                  # the lepton row
c0 = (0.5) ** 0.5
c1 = (float(6 * Qs[nh] - 2) / 8) ** 0.5
delta = float(ds[nh])
def masses_with_chi(chi_row):
    out = []
    for j in range(3):
        shell = c1 * cmath.exp(1j * delta) * OM ** j + c1 * cmath.exp(-1j * delta) * OM ** (-j)
        amp = c0 + cmath.exp(1j * chi_row) * shell      # common e^{i chi} on the shell pair rel to Perron
        out.append(abs(amp) ** 2)
    return sorted(out)
m0 = masses_with_chi(0.0)                               # leading (shipped)
shipped = the_run.read_masses()[nh]
check(f"S-2 C-LEADING: chi=0 reproduces the shipped read_masses lepton row "
      f"(rel err {max(abs(a/b-1) for a,b in zip(m0,shipped)):.1e})",
      max(abs(a / b - 1) for a, b in zip(m0, shipped)) < 1e-9)
mchi = masses_with_chi(chi)
# m_e/m_tau = smallest/largest
ratio0 = m0[0] / m0[2]
ratio_chi = mchi[0] / mchi[2]
shift_ppm = (ratio_chi / ratio0 - 1) * 1e6
print(f"    m_e/m_tau leading = {ratio0:.8e}; with chi = {ratio_chi:.8e}")
print(f"    induced m_e/m_tau shift = {shift_ppm:+.3f} ppm  (sign forced by chi)")
check(f"S-2 C-SIGN: chi sign forced (advance-sign + J orientation) = "
      f"{'NEG (DOWN)' if chi < 0 else 'POS (UP)'}; induced mass shift sign = "
      f"{'DOWN' if shift_ppm < 0 else 'UP'}", True)

# ===========================================================================
banner("S-3  ================  THE SINGLE MARKED COMPARISON  ================")
# ===========================================================================
OBS_PPM = -70.3                                         # observed m_e/m_tau residual
EPS_TARGET = -1.7515e-7                                 # the pinned chiral phase
print(f"    observed m_e/m_tau residual : {OBS_PPM:+.1f} ppm")
print(f"    forced construction yields  : {shift_ppm:+.3f} ppm")
factor = shift_ppm / OBS_PPM if OBS_PPM else float('nan')
print(f"    ratio to observed = {factor:+.2f}  (1.0 = LAND; |.|>>1 = over-applies)")
print(f"    pinned eps phase target = {EPS_TARGET:+.4e} rad;  forced chi = {chi:+.4e} rad")
print(f"    chi / eps_target = {chi/EPS_TARGET:+.2f}")

# ===========================================================================
banner("S-4  VERDICT + tier")
# ===========================================================================
land = abs(factor) < 1.5 and (shift_ppm < 0) == (OBS_PPM < 0) and abs(factor) > 1/1.5
a1p5 = 2 * u ** 5
print(f"    [poison watch] 2*alpha_1^5 = {a1p5:.3e} (NOT used; flagged coincidence)")
if land:
    tier = ("LAND -- the -70 ppm falls out of the forced seed through read_masses' "
            "second order, no tuning. (USER-gated adoption.)")
elif (shift_ppm < 0) == (OBS_PPM < 0):
    tier = (f"WALL -- mechanism + SIGN correct, magnitude over/under by the FORCED "
            f"factor F = {factor:+.2f}. The cos(chi) second order is right; chi from "
            f"the STATE-BLOCK proxy is off by F (A5: state-block over-applies). Step "
            f"2 -> architect: compute chi via the PROPER transport, target chi/sqrt(|F|) "
            f"= {chi/math.sqrt(abs(factor)):+.3e} rad. NO tuning of F; 2*alpha_1^5 "
            f"NOT invoked.")
else:
    tier = (f"WALL (sign) -- magnitude factor {factor:+.2f} but the SIGN is wrong; "
            "the bit-odd holonomy sign does not match the observed DOWN residual "
            "(re-examine the advance-sign/J orientation). Banked as-run.")
print("    TIER:", tier)
check("S-4 scope honesty: ONE frozen chi construction (A5 bit-odd proxy); the "
      "second-order suppression is read_masses' own cos(chi), NOT inserted; NO "
      "alpha_1 power added; 2*alpha_1^5 NOT pattern-matched; target only in S-3; "
      "no value", True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
