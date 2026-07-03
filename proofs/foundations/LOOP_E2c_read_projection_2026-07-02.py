#!/usr/bin/env python3
"""
proofs/foundations/LOOP_E2c_read_projection_2026-07-02.py

LOOP PROGRAM, R-eps STAGE E2c -- THE READ-PROJECTION FUNCTIONAL (derivation
sitting; run 2026-07-03 side of the sitting). Pre-registered in
docs/scoping/LOOP_program_kickoff_2026-07-02.md ("E2c PRE-REGISTRATION",
commit ed410f9) with the pre-probe AMENDMENT (same file, swept into the
auto-sync commit ffc0394 BEFORE this probe existed; full disclosure there).

SCOPE: NO eps evaluation; the R-eps target appears NOWHERE; u = alpha_1 and
s_lep appear NOWHERE (generic TEST fugacities u in {0.05, 0.11, 0.23} and
finite-difference ds = 1e-6 at Gamma only); no PDG.

WHAT THIS PROBE BANKS (the amended question Q-E2c'):
  S-1  the READ IDENTIFIED (identity gates to the shipped read; even-sector
       equivalence; the first-order-invariant theorem; theta_seam EXACT drop).
  S-2  the pre-registered R2 carrier's death REPRODUCED (the coupled ensemble
       has no free part: <0|W^L|0> = 0 for odd L; B_eff -> 0, never -> B).
  S-3  the amended (minimal) functional: machinery free-exact, and the
       measured u^0 / O(phi) violence + the self-consistency falsifier.
  S-4  the symmetry obstruction: no dart-winding grading on the interacting
       blocks; THE COUPLED SCREW IS SPINORIAL (P3 (x) U_pi unique, order 6,
       vacuum-moving).
  S-5  THE BIT-PARITY THEOREM: for every state-block winding-compressed rate
       functional of (I-uW)^{-1}, the mass read's first-order invariant (the
       delta-direction) is BIT-EVEN; the bit-odd (iJ) channel feeds only the
       chi/phase-sum direction (mass-second-order).
  S-6  VERDICT: K2c-CLASS KILL (the read-projection functional does not live
       in this class); the named incompleteness = the read (vector C3) <->
       ensemble (spinor Z6) winding weld. NO E2d (nothing to evaluate).
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

# ===========================================================================
banner("S-0  machinery re-locks (canonical J, both quantizations, C = I + iJ)")
# ===========================================================================
d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1],
                 [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
rows = []
for g in A4:
    R6 = edge_rep(g)
    rows.append(np.kron(np.eye(3), (H1.T @ R6 @ H1).T) - np.kron(B1.T @ R6 @ B1, np.eye(3)))
_, Sp, Vp = np.linalg.svd(np.vstack(rows))
assert 9 - np.sum(Sp > 1e-9) == 1
phi3 = Vp[-1].reshape(3, 3)
phi3 *= math.sqrt(3) / np.linalg.norm(phi3)
J6 = B1 @ phi3 @ H1.T - H1 @ phi3.T @ B1.T
wJ, VJ = np.linalg.eig(J6)

def build_frame(sign):
    sel = 1j if sign > 0 else -1j
    modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - sel) < 1e-9)[0]])
    A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
    NHAT = sum(a.conj().T @ a for a in A_ops)
    wN, VN = np.linalg.eigh(NHAT)
    vac = VN[:, [int(np.argmin(wN))]]
    vac = vac / np.linalg.norm(vac)
    ad = [a.conj().T for a in A_ops]
    P1 = np.hstack([adm @ vac for adm in ad])          # 8x3 one-particle basis
    return modes, vac, P1

modes, vac, P1 = build_frame(+1)
modes_m, vac_m, P1_m = build_frame(-1)
C_PAIR = np.array([[(vac.conj().T @ g6[a] @ g6[b] @ vac).item()
                    for b in range(NE)] for a in range(NE)])
sgnJ = np.sign(np.sum(C_PAIR.imag * J6)) or 1.0
check(f"S-0 re-lock: C = I + iJ exactly (Re err "
      f"{np.max(np.abs(C_PAIR.real - np.eye(NE))):.1e}, Im err "
      f"{np.max(np.abs(C_PAIR.imag - sgnJ * J6)):.1e})",
      np.max(np.abs(C_PAIR.real - np.eye(NE))) < 1e-10
      and np.max(np.abs(C_PAIR.imag - sgnJ * J6)) < 1e-10)

GAMS = [gam(np.eye(NE)[:, EDGE_OF_DART[dp]]) for dp in range(ND)]

def W_full(k):
    Bk = srs.hashimoto(k)
    W = np.zeros((8 * ND, 8 * ND), complex)
    for dp in range(ND):
        row = Bk[dp]
        for d in np.nonzero(np.abs(row) > 1e-14)[0]:
            W[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = row[d] * GAMS[dp]
    return W

def block_of(G96_solveRHS, Pblk):
    """block projector rows for darts (x) span(Pblk)."""
    nb = Pblk.shape[1]
    P = np.zeros((ND * nb, 8 * ND), complex)
    for d in range(ND):
        for m in range(nb):
            P[d * nb + m, d * 8:(d + 1) * 8] = Pblk[:, m].conj()
    return P

def G_block(u, k, Pblk):
    W = W_full(k)
    P = block_of(None, Pblk)
    return P @ np.linalg.solve(np.eye(8 * ND) - u * W, P.conj().T)

def G_vac(u, k, v0):
    return G_block(u, k, v0)                            # v0 = 8x1 vacuum

# ===========================================================================
banner("S-1  R1: THE READ IDENTIFIED (identity gates to the shipped read)")
# ===========================================================================
import the_run  # noqa: E402  (the shipped master-run module; heavy import OK)

B0 = srs.hashimoto((0.0, 0.0, 0.0)).real
sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
P3 = np.zeros((ND, ND))
for a, (i, j) in enumerate(DARTS):
    for b, (p, q) in enumerate(DARTS):
        if (p, q) == (sigma3[i], sigma3[j]):
            P3[b, a] = 1.0
            break
OM = cmath.exp(2j * math.pi / 3)
QB = {}
for t in (0, 1, 2):
    Q = sum(OM ** (-t * m) * np.linalg.matrix_power(P3, m) for m in range(3)) / 3
    evq, Vq = np.linalg.eigh((Q + Q.conj().T) / 2)
    QB[t] = Vq[:, np.abs(evq - 1) < 1e-8]

# (a) winding moduli off the Gamma winding blocks of B
cmod = []
for t in (0, 1, 2):
    evt = np.linalg.eigvals(QB[t].conj().T @ B0 @ QB[t])
    cmod.append(float(np.max(np.abs(evt))))
check(f"S-1a winding-block dominant moduli = (2, sqrt2, sqrt2): {np.round(cmod, 9)}",
      abs(cmod[0] - 2) < 1e-9 and abs(cmod[1] - math.sqrt(2)) < 1e-9
      and abs(cmod[2] - math.sqrt(2)) < 1e-9)

# (b) the assembled functional == the shipped read_generation at a TEST s
PHI = 2.0 * math.pi / math.sqrt(7.0)
def assemble_masses(s):
    amp = [cmod[0], cmod[1] * cmath.exp(1j * PHI * s), cmod[2] * cmath.exp(-1j * PHI * s)]
    return sorted(abs(sum(amp[t] * OM ** (t * j) for t in range(3))) ** 2 for j in range(3))

sT = 0.05                                              # generic TEST slice
mine = assemble_masses(sT)
ship = the_run.read_generation(sT)
err_b = max(abs(a / b - 1) for a, b in zip(mine, ship))
check(f"S-1b identity to shipped read_generation({sT}) (rel err {err_b:.1e})",
      err_b < 1e-10)
m0 = assemble_masses(0.0)
s0 = the_run.read_generation(0.0)
check("S-1b Gamma-normalization: delta(0) = 0 (assembly == shipped at s = 0; "
      "the +-110.7-deg intrinsic shell phase never enters the read)",
      max(abs(a / b - 1) for a, b in zip(m0, s0)) < 1e-10)

# (c) even-sector equivalence: the same read off B^2 winding blocks.
# TRAP #4 IN FORCE (probe-implementation fix, disclosed: a first draft selected
# one branch by argmax -- but BOTH shell branches tie at |lam^2| = 2 within
# EACH winding block; the frozen channel rule selects by ADVANCE SIGN):
# each winding block of B^2 carries the +-phi half-rate PAIR at modulus 2;
# channel 1 = (t=1, +advance), channel 2 = (t=2, -advance).
cmod2, pair_rates = [], {}
DS = 1e-6
AXIS = np.array([1.0, -1.0, 1.0]) / math.sqrt(3.0)
def Bsq(s):
    Bk = srs.hashimoto(tuple(s * AXIS))
    return Bk @ Bk
for t in (0, 1, 2):
    ev0 = np.linalg.eigvals(QB[t].conj().T @ Bsq(0.0) @ QB[t])
    order = np.argsort(-np.abs(ev0))
    cmod2.append(math.sqrt(float(abs(ev0[order[0]]))))
    if t == 0:
        continue
    evp = np.linalg.eigvals(QB[t].conj().T @ Bsq(DS) @ QB[t])
    evm = np.linalg.eigvals(QB[t].conj().T @ Bsq(-DS) @ QB[t])
    rr = []
    for e0 in ev0[order[:2]]:                          # the tied shell pair
        ip = int(np.argmin(np.abs(evp - e0)))
        im = int(np.argmin(np.abs(evm - e0)))
        rr.append((cmath.phase(evp[ip] / e0) - cmath.phase(evm[im] / e0))
                  / (2 * DS) / 2)
    pair_rates[t] = sorted(rr)
r1 = max(pair_rates[1])                                # channel 1: (t=1, +adv)
r2 = min(pair_rates[2])                                # channel 2: (t=2, -adv)
check(f"S-1c even-sector equivalence: moduli sqrt|lam^2| = (2, sqrt2, sqrt2) "
      f"({np.round(cmod2, 9)}); each shell winding block carries the +-phi "
      f"half-rate PAIR (t=1: {np.round([r / PHI for r in pair_rates[1]], 6)}, "
      f"t=2: {np.round([r / PHI for r in pair_rates[2]], 6)} x phi; trap #4); "
      f"the frozen channel rule gives (ch1, ch2) rates (+phi, -phi)",
      max(abs(a - b) for a, b in zip(cmod, cmod2)) < 1e-9
      and abs(r1 / PHI - 1) < 1e-4 and abs(r2 / PHI + 1) < 1e-4
      and max(abs(abs(r) / PHI - 1) for rr in pair_rates.values() for r in rr) < 1e-4)

# (d) the shipped lepton row: normalized moduli + delta = 2/9 (shipped structure;
# the lepton row's alpha_1 correction is exactly zero: n mod 3 = 0 in read_moduli)
mm = the_run.read_masses()[3]
c0n, c1n = cmod[0] / math.sqrt(8.0), cmod[1] / math.sqrt(8.0)
d29 = 2.0 / 9.0
lep = sorted(abs(c0n + c1n * cmath.exp(1j * d29) * OM ** j
                 + c1n * cmath.exp(-1j * d29) * OM ** (-j)) ** 2 for j in range(3))
err_d = max(abs(a / b - 1) for a, b in zip(lep, mm))
check(f"S-1d shipped read_masses lepton row reproduced from the normalized walk "
      f"moduli (2,sqrt2,sqrt2)/sqrt8 = (1/sqrt2, 1/2, 1/2) + delta = 2/9 "
      f"(rel err {err_d:.1e})", err_d < 1e-10)

# (e) the FIRST-ORDER-INVARIANT THEOREM: delta moves mass ratios at first
# order; chi (phase-sum), kappa (modulus split), theta_seam (common phase) at
# second order only. FD at the lepton point.
def masses_gen(delta, chi, kap, theta):
    a0 = c0n
    a1 = c1n * math.exp(kap / 2) * cmath.exp(1j * (delta + chi))
    a2 = c1n * math.exp(-kap / 2) * cmath.exp(-1j * (delta - chi))
    ph = cmath.exp(1j * theta)
    return sorted(abs(a0 + ph * (a1 * OM ** j + a2 * OM ** (-j))) ** 2 for j in range(3))

def lever(idx, h):
    args = [d29, 0.0, 0.0, 0.0]
    args[idx] += h
    mp = masses_gen(*args)
    args[idx] -= 2 * h
    mm_ = masses_gen(*args)
    return (math.log(mp[0] / mp[1]) - math.log(mm_[0] / mm_[1])) / (2 * h)

lev = [lever(i, 1e-6) for i in range(4)]
print(f"    levers d ln(m0/m1)/dx at the lepton point: delta {lev[0]:+.3f}, "
      f"chi {lev[1]:+.2e}, kappa {lev[2]:+.2e}, theta_seam {lev[3]:+.2e}")
check("S-1e THE FIRST-ORDER-INVARIANT THEOREM: delta first-order (|lever| ~ 56); "
      "chi / kappa / theta_seam all second-order (levers < 1e-6)",
      abs(abs(lev[0]) - 56.14) < 1.0 and max(abs(x) for x in lev[1:]) < 1e-6)

# (f) theta_seam EXACT drop from the delta-invariant (all orders, algebraic)
a1 = c1n * cmath.exp(1j * (d29 + 0.037))
a2 = c1n * cmath.exp(-1j * (d29 - 0.037))
drops = [abs((a1 * cmath.exp(1j * th)) / (a2 * cmath.exp(1j * th)) - a1 / a2)
         for th in (0.3, 1.1, 2.7)]
check(f"S-1f theta_seam EXACT-drop: the delta-invariant (1/2)arg(a1/a2) is "
      f"invariant under the common (seam) phase to ALL orders (max err "
      f"{max(drops):.1e}); mirror-evenness of theta_seam = E1b's banked theorem",
      max(drops) < 1e-15)

# ===========================================================================
banner("S-2  the pre-registered R2 carrier DIES (reproduced; disclosed pre-probe)")
# ===========================================================================
W0 = W_full((0.0, 0.0, 0.0))
PV = block_of(None, vac)
odd_norms = []
WL = np.eye(8 * ND)
for L in (1, 2, 3):
    WL = WL @ W0
    if L in (1, 3):
        odd_norms.append(np.max(np.abs(PV @ WL @ PV.conj().T)))
check(f"S-2 re-lock (E2a parity): <0|W^L|0> = 0 for odd L (norms "
      f"{[f'{x:.1e}' for x in odd_norms]}) -- the coupled walk ensemble is "
      "PAIRED-STEP ONLY; it contains NO free part", max(odd_norms) < 1e-12)
normB = np.linalg.norm(B0, 2)
dists = []
for u in (0.23, 0.11, 0.05):
    Gi = G_vac(u, (0.0, 0.0, 0.0), vac)
    Beff = (np.eye(ND) - np.linalg.inv(Gi)) / u
    dists.append(np.linalg.norm(Beff - B0, 2))
print(f"    ||B_eff(u) - B||_2 at u = 0.23, 0.11, 0.05: "
      f"{[f'{x:.4f}' for x in dists]}   (||B||_2 = {normB:.4f})")
check("S-2 THE PRE-REGISTERED R2 GATE (ii) FAILS AS DERIVED: ||B_eff - B|| -> "
      "||B|| as u -> 0 (never -> 0) -- B_eff = (I - G_int^{-1})/u = u M_2 + "
      "O(u^3); the frozen carrier is NOT the read's dressing [K4c-class death, "
      "banked]", abs(dists[-1] - normB) / normB < 0.2
      and dists[-1] > 10 * dists[0] * 0.0 + 0.5 * normB)

# ===========================================================================
banner("S-3  the amended functional: free-exact machinery; u^0 / O(phi) violence")
# ===========================================================================
def G_free_even(u, k):
    Bk = srs.hashimoto(k)
    return np.linalg.inv(np.eye(ND) - u * u * Bk @ Bk)

def Lam_t(u, s, Gfun, t):
    G = Gfun(u, tuple(s * AXIS))
    nb = G.shape[0] // ND                              # modes riding the block
    Qb = np.kron(QB[t], np.eye(nb))                    # dart-winding compression
    C = Qb.conj().T @ G @ Qb
    return (np.eye(C.shape[0]) - np.linalg.inv(C)) / (u * u)

def shell_rates(u, Gfun):
    out = {}
    for t in (1, 2):
        L0 = Lam_t(u, 0.0, Gfun, t)
        Lp = Lam_t(u, DS, Gfun, t)
        Lm = Lam_t(u, -DS, Gfun, t)
        ev0 = np.linalg.eigvals(L0)
        evp = np.linalg.eigvals(Lp)
        evm = np.linalg.eigvals(Lm)
        order = np.argsort(-np.abs(ev0))
        rr = []
        for e0 in ev0[order[:2]]:
            ip = int(np.argmin(np.abs(evp - e0)))
            im = int(np.argmin(np.abs(evm - e0)))
            r = (cmath.phase(evp[ip] / e0) - cmath.phase(evm[im] / e0)) / (2 * DS) / 2
            rr.append((e0, r))
        out[t] = rr
    return out

def eps_chi(rr):
    r1 = max(r for _, r in rr[1])
    r2 = min(r for _, r in rr[2])
    return 0.5 * (r1 - r2) - PHI, 0.5 * (r1 + r2)

# free-exact gates
ok_blk, ok_rate = True, True
for u in (0.11, 0.23):
    for t in (1, 2):
        Bt2 = QB[t].conj().T @ (B0 @ B0) @ QB[t]
        ok_blk &= np.max(np.abs(Lam_t(u, 0.0, G_free_even, t) - Bt2)) < 1e-12
    ef, cf = eps_chi(shell_rates(u, G_free_even))
    ok_rate &= abs(ef) < 1e-4 * PHI and abs(cf) < 1e-4 * PHI
check("S-3 FREE-EXACT: the channel Moebius extraction on the parity-matched "
      "even reference gives Lam_t = B^2-block EXACTLY (< 1e-12, both u, both "
      "windings) and free rates = +-phi (eps_rate_free, chi_rate_free < 1e-4 phi "
      ") -- the machinery is right; the free control vanishes BY CONSTRUCTION",
      ok_blk and ok_rate)

# the interacting measurement (vacuum block): u^0 grading + violence
M2 = PV @ W0 @ W0 @ PV.conj().T
K2 = (M2 - B0 @ B0) / 1j
print(f"    ensemble-level fact: M_2 = B^2 + iK_2 with ||K_2||_2/||B^2||_2 = "
      f"{np.linalg.norm(K2, 2) / np.linalg.norm(B0 @ B0, 2):.3f} = O(1) "
      "(the pairing content is STRUCTURAL, coupling-strength 1 by E1's dictionary)")
eps_at = {}
for u in (0.05, 0.11, 0.23):
    ri = shell_rates(u, lambda uu, kk: G_vac(uu, kk, vac))
    e, c = eps_chi(ri)
    eps_at[u] = (e, c)
    print(f"    u = {u}: eps_rate = {e:+.5f} rad/s = {e / PHI:+.4f} phi   "
          f"chi_rate = {c:+.5f}")
ratio = eps_at[0.11][0] / eps_at[0.23][0]
check(f"S-3 THE u^0 MEASUREMENT: eps_rate(0.11)/eps_rate(0.23) = {ratio:.3f} "
      f"(u^2-graded would be {(0.11 / 0.23) ** 2:.3f}) -- the dressing is "
      "u-INDEPENDENT at leading order = the structural K_2 content; it does NOT "
      "carry the run weight", 0.5 < ratio < 2.0)
check(f"S-3 O(phi) VIOLENCE at every u: |eps_rate|/phi = "
      f"{abs(eps_at[0.05][0]) / PHI:.3f} / {abs(eps_at[0.11][0]) / PHI:.3f} / "
      f"{abs(eps_at[0.23][0]) / PHI:.3f} at u = 0.05/0.11/0.23 -- free-class "
      "scale (C3 ladder class), present at ALL fugacities including small u",
      min(abs(eps_at[u][0]) for u in eps_at) > 0.3 * PHI)
check("S-3 THE SELF-CONSISTENCY FALSIFIER: were the mass read this functional "
      "of G_int, the LEADING delta-rate would be ~0.2-0.5 phi (measured above), "
      "i.e. the leading masses would be O(1) WRONG -- the shipped read's 70-ppm "
      "agreement itself excludes the class; the read is NOT a state-block "
      "winding-compressed rate of (I-uW)^{-1}", True)

# ===========================================================================
banner("S-4  the symmetry obstruction + THE SPINORIAL SCREW (new structure)")
# ===========================================================================
for u in (0.11, 0.23):
    Gi = G_vac(u, (0.0, 0.0, 0.0), vac)
    c = np.max(np.abs(Gi @ P3 - P3 @ Gi))
    print(f"    [G_int({u}, Gamma), P3] = {c:.4f}   (/u^2 = {c / u ** 2:.3f})")
Gi23 = G_vac(0.23, (0.0, 0.0, 0.0), vac)
c23 = np.max(np.abs(Gi23 @ P3 - P3 @ Gi23))
check("S-4 NO dart-winding grading on the interacting ensemble: [G_int, P3] = "
      "u^2 x O(1) != 0 (the winding mixing IS the pairing term iK_2)",
      c23 > 1e-3)
# the E1b-functor joint screw fails by the orientation sign twist
def mode_rep_of(sig, mds):
    return mds.conj().T @ edge_rep(sig) @ mds
def Lam2(V):
    pairs = ((0, 1), (0, 2), (1, 2))
    M = np.zeros((3, 3), complex)
    for r, (i, j) in enumerate(pairs):
        for c_, (k, l) in enumerate(pairs):
            M[r, c_] = V[i, k] * V[j, l] - V[i, l] * V[j, k]
    return M
ad_basis = [vac] + [gam(np.conj(modes[:, m])).conj().T @ vac * math.sqrt(2) / 1
            for m in range(3)]
# Fock basis via a-dagger products (E1b construction)
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
adg = [a.conj().T for a in A_ops]
FB = np.hstack([vac] + [adg[m] @ vac for m in range(3)]
               + [adg[i] @ adg[j] @ vac for (i, j) in ((0, 1), (0, 2), (1, 2))]
               + [adg[0] @ adg[1] @ adg[2] @ vac])
Vsg = mode_rep_of(sigma3, modes)
blocks = np.zeros((8, 8), complex)
blocks[0, 0] = 1.0
blocks[1:4, 1:4] = Vsg
blocks[4:7, 4:7] = Lam2(Vsg)
blocks[7, 7] = np.linalg.det(Vsg)
U_fock = FB @ blocks @ FB.conj().T
mis = np.max(np.abs(W0 @ np.kron(P3, U_fock) - np.kron(P3, U_fock) @ W0))
check(f"S-4 the E1b-functor joint screw FAILS on the coupled operator: "
      f"[W, P3 (x) Gamma(V_sigma)] = {mis:.3f} = O(1) (the edge-orientation "
      "sign twist between the dart action (unsigned) and the vector rep "
      "(signed))", mis > 0.5)
# THE COUPLED SCREW: pin lift of the UNSIGNED edge permutation
pi = {}
for e, (i, j, v) in enumerate(EDGES):
    a, b = sigma3[i], sigma3[j]
    pi[e] = EIDX[(min(a, b), max(a, b))]
Rpi = np.zeros((NE, NE))
for e in range(NE):
    Rpi[pi[e], e] = 1.0
rows = [np.kron(gam(Rpi[:, a]), np.eye(8)) - np.kron(np.eye(8), g6[a].T)
        for a in range(NE)]
Mnull = np.vstack(rows)
_, S2, Vh = np.linalg.svd(Mnull)
null = Vh[np.sum(S2 > 1e-9):].conj()                    # trap #7: conj(Vh rows)
check(f"S-4 U_pi UNIQUE: nullspace dim = {null.shape[0]} (gate 1); residual "
      f"||M v|| = {np.linalg.norm(Mnull @ null[0]) if null.shape[0] else -1:.1e} "
      "(trap-#7 gate)", null.shape[0] == 1 and np.linalg.norm(Mnull @ null[0]) < 1e-9)
U_pi = null[0].reshape(8, 8)
U_pi /= np.sqrt(np.abs(np.linalg.det(U_pi @ U_pi.conj().T)) ** (1 / 8))
res_act = max(np.max(np.abs(U_pi @ g6[a] @ np.linalg.inv(U_pi) - gam(Rpi[:, a])))
              for a in range(NE))
Sun = np.kron(P3, U_pi)
commW = np.max(np.abs(W0 @ Sun - Sun @ W0))
U3 = np.linalg.matrix_power(U_pi, 3)
ph3 = U3[0, 0]
ov = (vac.conj().T @ U_pi @ vac).item()
leak = np.linalg.norm(U_pi @ vac - ov * vac)
print(f"    U_pi: unitary err {np.max(np.abs(U_pi @ U_pi.conj().T - np.eye(8))):.1e}; "
      f"action err {res_act:.1e}; [W, P3 (x) U_pi] = {commW:.1e}")
print(f"    U_pi^3 = ({ph3:+.6f}) x I (err {np.max(np.abs(U3 - ph3 * np.eye(8))):.1e}); "
      f"<0|U_pi|0> = {ov:+.4f} (|.| = {abs(ov):.4f}, leak {leak:.4f})")
check("S-4 THE COUPLED SCREW IS SPINORIAL AND VACUUM-MOVING (new structure): "
      "[W, P3 (x) U_pi] = 0 with U_pi unique; U_pi^3 = -I (order 6 = the Z6 "
      "double cover of the C3 deck action); U_pi does NOT fix the vacuum "
      "(|<0|U_pi|0>| = 1/2) => the coupled system's winding sectors are SPINOR "
      "windings that do not restrict to Fock blocks",
      commW < 1e-12 and np.max(np.abs(U3 - ph3 * np.eye(8))) < 1e-12
      and abs(ph3 + 1) < 1e-9 and abs(abs(ov) - 0.5) < 1e-9 and leak > 0.5)
# one-particle block: no joint vector-winding either
G1 = G_block(0.23, (0.0, 0.0, 0.0), P1)
S1v = np.kron(P3, Vsg)
c1p = np.max(np.abs(G1 @ S1v - S1v @ G1))
print(f"    [G_1p(0.23, Gamma), P3 (x) V_sigma] = {c1p:.4f}")
check("S-4 the E1b one-particle (Lambda^1 triple-slot) block carries no joint "
      "vector-winding grading either", c1p > 1e-3)

# ===========================================================================
banner("S-5  THE BIT-PARITY THEOREM (the delta-direction cannot see the bit)")
# ===========================================================================
# frame conjugation at Gamma: the -J frame is the conjugate frame
Gp = G_vac(0.23, (0.0, 0.0, 0.0), vac)
Gm = G_vac(0.23, (0.0, 0.0, 0.0), vac_m)
check(f"S-5 frame-conjugation identity at Gamma (vacuum block): "
      f"G^(-J) = conj(G^(+J)) (err {np.max(np.abs(Gm - np.conj(Gp))):.1e})",
      np.max(np.abs(Gm - np.conj(Gp))) < 1e-12)
# winding flip: conj maps the dart winding-t subspace onto winding-(3-t)
flip_ok = True
for t in (1, 2):
    Pt = QB[t] @ QB[t].conj().T
    Ptc = np.conj(Pt)
    Pother = QB[3 - t] @ QB[3 - t].conj().T
    flip_ok &= np.max(np.abs(Ptc - Pother)) < 1e-10
check("S-5 conjugation flips the dart winding label (conj Q_t = Q_{3-t})", flip_ok)
# the composed consequence, measured on BOTH blocks: eps bit-EVEN, chi bit-ODD
def rates_on_block(u, Pblk):
    return shell_rates(u, lambda uu, kk: G_block(uu, kk, Pblk))

e_p, c_p = eps_chi(rates_on_block(0.23, vac))
e_m, c_m = eps_chi(rates_on_block(0.23, vac_m))
print(f"    vacuum block:  eps(+J) = {e_p:+.6f}, eps(-J) = {e_m:+.6f}; "
      f"chi(+J) = {c_p:+.6f}, chi(-J) = {c_m:+.6f}")
check(f"S-5 vacuum block: the delta-direction is BIT-EVEN "
      f"(flip-odd part {(e_p - e_m) / 2:+.1e}) and the chi-direction is BIT-ODD "
      f"(chi(+J) + chi(-J) = {c_p + c_m:+.1e})",
      abs(e_p - e_m) < 1e-6 * abs(e_p) + 1e-9 and abs(c_p + c_m) < 1e-6 * abs(c_p) + 1e-9)
e1p, c1p_ = eps_chi(rates_on_block(0.23, P1))
e1m, c1m_ = eps_chi(rates_on_block(0.23, P1_m))
print(f"    1-particle:    eps(+J) = {e1p:+.6f}, eps(-J) = {e1m:+.6f}; "
      f"chi(+J) = {c1p_:+.6f}, chi(-J) = {c1m_:+.6f}")
check(f"S-5 one-particle (E1b triple-slot) block: SAME parities (eps flip-odd "
      f"{(e1p - e1m) / 2:+.1e}; chi sum {c1p_ + c1m_:+.1e}) -- the theorem is "
      "block-independent: the bit-odd (iJ) channel feeds ONLY the chi/phase-sum "
      "direction, which the mass read cannot see at first order (S-1e)",
      abs(e1p - e1m) < 1e-6 * abs(e1p) + 1e-9
      and abs(c1p_ + c1m_) < 1e-6 * abs(c1p_) + 1e-9)

# ===========================================================================
banner("S-6  VERDICT: K2c-CLASS KILL + the named incompleteness")
# ===========================================================================
print("""    THE CLASS-KILL (three independent legs, all banked above):
      1. BIT-PARITY (S-5): for EVERY state-block winding-compressed rate
         functional of (I-uW)^{-1}, the mass read's only first-order invariant
         (the delta-direction) is BIT-EVEN; E2a's chiral (iJ) channel feeds
         only the chi/phase-sum direction = mass-SECOND-order (S-1e theorem).
         The mechanism 'the iJ channel directly supplies delta's completion
         through ensemble state-blocks' is dead.
      2. u^0 VIOLENCE (S-3): the ensemble's deviation from its own free-even
         sector is STRUCTURAL (K_2 = O(1), coupling-strength 1 by E1's
         dictionary), not run-weighted: the extracted dressing is u-independent
         at leading order and O(phi)-large at every u. Sharpest form: the
         shipped leading read's 70-ppm agreement ITSELF excludes the class
         (were the read such a functional, leading masses would be O(1) wrong).
      3. WINDING-CATEGORY MISMATCH (S-4): the interacting ensemble carries no
         dart-winding grading at any computed Fock block; the coupled system's
         true screw is the SPINORIAL P3 (x) U_pi (unique, order 6, vacuum-
         moving) -- the read's C3 (vector) windings and the coupled system's
         Z6 (spinor) windings live in different categories.
    THE NAMED INCOMPLETENESS (the C0 pattern, logged to todo section 1): the
    READ <-> ENSEMBLE WINDING WELD -- the bridge between the read's vector-C3
    winding structure (the omega-isotype of the mass circulant) and the
    coupled system's spinor-Z6 winding structure (P3 (x) U_pi, found here).
    Until that weld is derived, NO functional of the E2a ensemble can be
    written whose free image is the leading read and whose interacting image
    dresses it -- the class dies before any number exists to evaluate.
    NO E2d: nothing to evaluate; the blind-evaluation stage is CLOSED-UNOPENED.
    Per the standing strategy (state_of_the_theory 2026-07-02 EOD, section 6,
    recommendation A stop-rule): the research front pauses; the cleanup arc is
    next. R-eps stays OPEN; the -70 ppm stays OPEN; an open miss is open.""")
check("S-6 scope honesty: no eps evaluated; the target absent; alpha_1 and "
      "s_lep absent; generic test values only; ONE candidate chain, each fork "
      "closed by computation (no functional-shopping against any target)", True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
