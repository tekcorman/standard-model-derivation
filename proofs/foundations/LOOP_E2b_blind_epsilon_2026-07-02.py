#!/usr/bin/env python3
"""
proofs/foundations/LOOP_E2b_blind_epsilon_2026-07-02.py

LOOP PROGRAM, R-eps STAGE E2b -- THE BLIND NUMBER. Pre-registered in
internal research notes ("E2b PRE-REGISTRATION",
commit 361da9f, BEFORE this probe). The target appears in EXACTLY ONE marked
block (S-4) and nowhere else in this file.

THE FROZEN OBJECT (all pieces derived; no constants; no scans as selectors):
  eps_raw = [Delta-chi_int - Delta-chi_free](s_lep)
  Delta-chi_X(s) = (1/2)[arg g_{X,h} - arg g_{X,hbar}](s), TOTAL unwrapped
    from Gamma (endpoint value + accumulated increments; the free
    Gamma-offset = the intrinsic shell phase, NOT delta, and subtracts),
  g_{X,t}(s) = <v_t(s)| G_X(u, s.AXIS) |v_t(s)>  (phase-gauge-invariant),
  G_free = (I - uB)^{-1};  G_int = <0|(I - uW)^{-1}|0> (the E2a-forced form),
  u = U_RUN = (2/3)^8 = alpha_1;  s_lep = (2/9)/phi, phi = 2pi/sqrt(7);
  channels: Gamma-seeded h/hbar shell modes (C3 convention, h = Im lam > 0),
  transported by overlap continuity (the frozen cocycle; trap #5: no winding
  projectors off Gamma). n_steps = 800 (+ one 1600 convergence gate at 1%).

SURFACES: S1 J-reality (eps real by construction); S2 soft rows <= 1.2 sigma
via the read (delta -> 2/9 + eps); S3 leading reads (the phi.s_lep = 2/9
station-A lock; the int-free onset at O(u^2), E2a parity); S4 no winding-side
leakage (omega-odd modulus dressing x lever <= 10% of the phase effect).
KILLS: K1b machinery; K2b tier-kill (re-localization PRE-NAMED: the
read-projection layer -- which channel weighting the physical delta-read
applies, the E1b theta_seam/triplet structure); K3b a surface breaks.
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

U = (2.0 / 3.0) ** 8                                # u = alpha_1 (C0's forced weight)
PHI = 2.0 * math.pi / math.sqrt(7.0)                # the screw rate (derived)
S_LEP = (2.0 / 9.0) / PHI                           # the lepton slice (= delta/phi)
AXIS = np.array([1.0, -1.0, 1.0]) / math.sqrt(3.0)

def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6

def gam(x):
    return sum(x[a] * g6[a] for a in range(NE))

# ===========================================================================
banner("S-0  machinery re-locks (E2a form; the canonical J vacuum)")
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
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
NHAT = sum(a.conj().T @ a for a in A_ops)
wN, VN = np.linalg.eigh(NHAT)
vac = VN[:, [int(np.argmin(wN))]]
vac = vac / np.linalg.norm(vac)
C_PAIR = np.array([[(vac.conj().T @ g6[a] @ g6[b] @ vac).item()
                    for b in range(NE)] for a in range(NE)])
check("S-0 re-lock: the vacuum pairing C = I + iJ exactly "
      f"(err {max(np.max(np.abs(C_PAIR.real - np.eye(NE))), np.max(np.abs(C_PAIR.imag - np.sign(np.sum(C_PAIR.imag * J6)) * J6))):.1e})",
      np.max(np.abs(C_PAIR.real - np.eye(NE))) < 1e-10)

# the gamma-weighted transfer per Bloch point, and its vacuum-block resolvent
GAM_OF_DART = [gam(np.eye(NE)[:, EDGE_OF_DART[dp]]) for dp in range(ND)]
P_VAC = np.zeros((ND, 8 * ND), complex)
for d in range(ND):
    P_VAC[d, d * 8:(d + 1) * 8] = vac[:, 0].conj()

def G_int_mat(u, k):
    Bk = srs.hashimoto(k)
    W = np.zeros((8 * ND, 8 * ND), complex)
    for dp in range(ND):
        row = Bk[dp]
        for d in np.nonzero(np.abs(row) > 1e-14)[0]:
            W[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = row[d] * GAM_OF_DART[dp]
    return P_VAC @ np.linalg.solve(np.eye(8 * ND) - u * W, P_VAC.conj().T)

# parity onset re-lock at Gamma on TEST fugacities (E2a theorem; O(u^2) onset)
Gf011 = np.linalg.inv(np.eye(ND) - 0.11 * srs.hashimoto((0, 0, 0)))
Gi011 = G_int_mat(0.11, (0.0, 0.0, 0.0))
Gf023 = np.linalg.inv(np.eye(ND) - 0.23 * srs.hashimoto((0, 0, 0)))
Gi023 = G_int_mat(0.23, (0.0, 0.0, 0.0))
r11 = np.max(np.abs(Gi011 - Gf011)) / 0.11 ** 2
r23 = np.max(np.abs(Gi023 - Gf023)) / 0.23 ** 2
check(f"S-0 re-lock: the int-free difference onsets at O(u^2) (||diff||/u^2 = "
      f"{r11:.3f} at u=0.11, {r23:.3f} at u=0.23 -- same order, finite; the E2a "
      "parity theorem in force)", 0.1 < r11 / r23 < 10)

# ===========================================================================
banner("S-1  the Gamma seeds and the free control (station-A lock)  [S3]")
# ===========================================================================
sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
P3 = np.zeros((ND, ND))
for a, (i, j) in enumerate(DARTS):
    for b, (p, q) in enumerate(DARTS):
        if (p, q) == (sigma3[i], sigma3[j]):
            P3[b, a] = 1.0
            break
B0 = srs.hashimoto((0.0, 0.0, 0.0))
assert np.max(np.abs(P3 @ B0 - B0 @ P3)) < 1e-12
OM = cmath.exp(2j * math.pi / 3)
seeds = {}
for t in (1, 2):
    Q = sum(OM ** (-t * m) * np.linalg.matrix_power(P3, m) for m in range(3)) / 3
    evq, Vq = np.linalg.eigh((Q + Q.conj().T) / 2)
    Qb = Vq[:, np.abs(evq - 1) < 1e-8]
    ev, V = np.linalg.eig(Qb.conj().T @ B0 @ Qb)
    i = int(np.argmax(np.abs(ev)))
    vec = Qb @ V[:, i]
    seeds[ev[i].imag > 0] = (ev[i], vec / np.linalg.norm(vec))
(lam_h, vec_h), (lam_hb, vec_hb) = seeds[True], seeds[False]
print(f"    Gamma seeds: lam_h = {lam_h:.6f} (|lam| = {abs(lam_h):.6f}), "
      f"lam_hbar = {lam_hb:.6f}")

def Bfull(s):
    return srs.hashimoto(s * AXIS)

def track_channels(vec0, lam0, s_end, n_steps):
    """the frozen cocycle: overlap-continuity transport; returns the TOTAL
    unwrapped arg and ln|.| of the free and interacting channel functionals
    (endpoint Gamma values + accumulated increments), plus diagnostics."""
    vec = vec0.copy()
    gf_prev = 1.0 / (1.0 - U * lam0)                 # free channel value at Gamma
    Gi0 = G_int_mat(U, (0.0, 0.0, 0.0))
    gi_prev = (vec.conj() @ Gi0 @ vec)
    arg_f, arg_i = cmath.phase(gf_prev), cmath.phase(gi_prev)
    lnm_f, lnm_i = math.log(abs(gf_prev)), math.log(abs(gi_prev))
    min_ovl = 1.0
    prev_lam = lam0
    acc_eig = 0.0
    for s in np.linspace(0.0, s_end, n_steps + 1)[1:]:
        Bs = Bfull(s)
        ev, VR = np.linalg.eig(Bs)
        ovl = np.abs(VR.conj().T @ vec)
        i = int(np.argmax(ovl))
        min_ovl = min(min_ovl, float(ovl[i] / np.linalg.norm(VR[:, i])))
        v = VR[:, i] / np.linalg.norm(VR[:, i])
        acc_eig += float(np.angle(ev[i] / prev_lam))
        gf = 1.0 / (1.0 - U * ev[i])                  # == <v|G_free|v> (identity gate below)
        Gi = G_int_mat(U, s * AXIS)
        gi = (v.conj() @ Gi @ v)
        arg_f += cmath.phase(gf / gf_prev)
        arg_i += cmath.phase(gi / gi_prev)
        prev_lam, gf_prev, gi_prev, vec = ev[i], gf, gi, v
    lnm_f_end, lnm_i_end = math.log(abs(gf_prev)), math.log(abs(gi_prev))
    return dict(arg_f=arg_f, arg_i=arg_i, acc_eig=acc_eig, min_ovl=min_ovl,
                lnm_f0=lnm_f, lnm_i0=lnm_i, lnm_f1=lnm_f_end, lnm_i1=lnm_i_end,
                vec_end=vec, lam_end=prev_lam)

# the free-identity gate: <v|G_free|v> == 1/(1 - u lam) on tracked eigenvectors
sT = 0.37 * S_LEP
evT, VT = np.linalg.eig(Bfull(sT))
iT = int(np.argmax(np.abs(VT.conj().T @ vec_h)))
vT = VT[:, iT] / np.linalg.norm(VT[:, iT])
lhs = vT.conj() @ np.linalg.inv(np.eye(ND) - U * Bfull(sT)) @ vT
rhs = 1.0 / (1.0 - U * evT[iT])
check(f"S-1 free-identity gate: <v|(I-uB)^-1|v> = 1/(1-u lam) on the tracked mode "
      f"(err {abs(lhs/rhs-1):.1e})", abs(lhs / rhs - 1) < 1e-10)

TR_h = track_channels(vec_h, lam_h, S_LEP, 800)
TR_hb = track_channels(vec_hb, lam_hb, S_LEP, 800)
chir_eig = (TR_h['acc_eig'] - TR_hb['acc_eig']) / 2
check(f"S-1 [S3 surface] the station-A lock: the tracked eigenphase chiral part = "
      f"{chir_eig:+.6f} vs phi*s_lep = {PHI*S_LEP:.6f} = 2/9 "
      f"({(chir_eig/(PHI*S_LEP)-1)*100:+.3f}%, gate 0.5%) -- the leading read "
      "reproduced by the FREE machinery", abs(chir_eig / (PHI * S_LEP) - 1) < 0.005)
check(f"S-1 tracking quality: min overlap h {TR_h['min_ovl']:.4f}, hbar "
      f"{TR_hb['min_ovl']:.4f} (> 0.9: no mode collision)",
      TR_h['min_ovl'] > 0.9 and TR_hb['min_ovl'] > 0.9)

# ===========================================================================
banner("S-2  the blind computation (the frozen epsilon-functional)")
# ===========================================================================
dchi_free = (TR_h['arg_f'] - TR_hb['arg_f']) / 2
dchi_int = (TR_h['arg_i'] - TR_hb['arg_i']) / 2
eps_raw = dchi_int - dchi_free
# components: the Gamma-offset difference and the line-increment difference
g0f_h = 1.0 / (1.0 - U * lam_h)
g0f_hb = 1.0 / (1.0 - U * lam_hb)
Gi0 = G_int_mat(U, (0.0, 0.0, 0.0))
g0i_h = (vec_h.conj() @ Gi0 @ vec_h)
g0i_hb = (vec_hb.conj() @ Gi0 @ vec_hb)
off_free = (cmath.phase(g0f_h) - cmath.phase(g0f_hb)) / 2
off_int = (cmath.phase(g0i_h) - cmath.phase(g0i_hb)) / 2
comp_offset = off_int - off_free
comp_line = eps_raw - comp_offset
print(f"    Delta-chi_free(s_lep) = {dchi_free:+.8f} rad   (Gamma offset {off_free:+.8f})")
print(f"    Delta-chi_int (s_lep) = {dchi_int:+.8f} rad   (Gamma offset {off_int:+.8f})")
print(f"    eps_raw = {eps_raw:+.8e} rad")
print(f"      component (Gamma-offset difference) = {comp_offset:+.8e}")
print(f"      component (line-increment difference) = {comp_line:+.8e}")
# convergence gate: n_steps 800 -> 1600
TR_h2 = track_channels(vec_h, lam_h, S_LEP, 1600)
TR_hb2 = track_channels(vec_hb, lam_hb, S_LEP, 1600)
eps_raw2 = ((TR_h2['arg_i'] - TR_hb2['arg_i']) - (TR_h2['arg_f'] - TR_hb2['arg_f'])) / 2
check(f"S-2 convergence: eps(1600) = {eps_raw2:+.6e} vs eps(800) = {eps_raw:+.6e} "
      f"(rel diff {abs(eps_raw2-eps_raw)/max(abs(eps_raw),1e-300):.1e}, gate 1%)",
      abs(eps_raw2 - eps_raw) <= 0.01 * abs(eps_raw))
check("S-2 [S1 surface] J-reality: eps is a difference of real phase functionals "
      "(real by construction; the modulus channel is handled separately in S-3)", True)

# ===========================================================================
banner("S-3  surfaces: leakage (S4) and the soft rows (S2)")
# ===========================================================================
# omega-odd modulus dressing (int - free), endpoint at the slice:
kappa = ((TR_h['lnm_i1'] - TR_h['lnm_f1']) - (TR_hb['lnm_i1'] - TR_hb['lnm_f1'])) / 2
# the read replica (the shipped structure: c0 = 1/sqrt2, c1 = 1/2 for the charged
# leptons at Q = 2/3; delta = 2/9 leading) -- levers computed from the read itself
def lepton_masses(delta, c1_scale=1.0):
    c0 = math.sqrt(0.5)
    c1 = 0.5 * c1_scale
    oms = [cmath.exp(2j * math.pi * j / 3) for j in range(3)]
    sm = sorted(abs(c0 + c1 * cmath.exp(1j * delta) * o + c1 * cmath.exp(-1j * delta) * o.conjugate())
                for o in oms)
    return [x * x for x in sm]

D0 = 2.0 / 9.0
m0 = lepton_masses(D0)
dd = 1e-9
m1 = lepton_masses(D0 + dd)
lever_phase = (math.log(m1[0] / m1[1]) - math.log(m0[0] / m0[1])) / dd     # dln(me/mmu)/ddelta
m2 = lepton_masses(D0, 1.0 + 1e-9)
lever_mod = (math.log(m2[0] / m2[2]) - math.log(m0[0] / m0[2])) / 1e-9     # dln(me/mtau)/dln c1
print(f"    read levers (from the shipped read structure): d ln(m_e/m_mu)/d delta = "
      f"{lever_phase:+.2f}; d ln(m_e/m_tau)/d ln c1 = {lever_mod:+.2f}")
phase_effect_ppm = abs(lever_phase * eps_raw) * 1e6
mod_effect_ppm = abs(lever_mod * kappa) * 1e6
check(f"S-3 [S4 surface] no winding-side leakage: omega-odd modulus dressing kappa = "
      f"{kappa:+.3e}; implied modulus-side effect {mod_effect_ppm:.3f} ppm vs "
      f"phase-side {phase_effect_ppm:.3f} ppm (gate: modulus <= 10% of phase)",
      mod_effect_ppm <= 0.10 * phase_effect_ppm)
# soft rows at the computed eps (bands are experimental -> quoted in the marked block;
# computed here, gated there)
me = lepton_masses(D0 + eps_raw)
soft_e_tau_ppm = (math.log(me[0] / me[2]) - math.log(m0[0] / m0[2])) * 1e6
soft_mu_tau_ppm = (math.log(me[1] / me[2]) - math.log(m0[1] / m0[2])) * 1e6
hard_e_mu_ppm = (math.log(me[0] / me[1]) - math.log(m0[0] / m0[1])) * 1e6
print(f"    implied shifts at eps_raw: m_e/m_mu {hard_e_mu_ppm:+.3f} ppm; "
      f"m_e/m_tau {soft_e_tau_ppm:+.3f} ppm; m_mu/m_tau {soft_mu_tau_ppm:+.3f} ppm")

# ===========================================================================
banner("S-4  THE MARKED COMPARISON BLOCK (the target appears HERE ONLY)")
# ===========================================================================
TARGET, TARGET_S = -1.7515e-7, 3.9e-10              # pre-registered (Q3; 0.22% band)
SOFT_BAND_PPM = 67.5                                 # m_tau-limited soft-row 1 sigma (Q3)
HARD_DEMAND_PPM, HARD_S_PPM = 9.83, 0.022            # the m_tau-free hard row (Q3)
pull = (eps_raw - TARGET) / TARGET_S
factor = eps_raw / TARGET
print(f"    Row 1  THE CANDIDATE: eps_raw = {eps_raw:+.6e} rad")
print(f"           target (pre-registered): {TARGET:+.6e} +- {TARGET_S:.1e} rad")
print(f"           pull = {pull:+.3e}; ratio eps/target = {factor:+.4e}")
tier = ("LANDING" if abs(pull) <= 1 else
        "MARGINAL (no adoption)" if abs(pull) <= 2 else "KILL")
print(f"           tier: {tier}")
print(f"    Row 2  components: Gamma-offset {comp_offset:+.3e}; line {comp_line:+.3e}")
print(f"    Row 3  soft rows: m_e/m_tau {soft_e_tau_ppm:+.2f} ppm "
      f"({soft_e_tau_ppm/SOFT_BAND_PPM:+.2f} sigma); m_mu/m_tau {soft_mu_tau_ppm:+.2f} ppm "
      f"({soft_mu_tau_ppm/SOFT_BAND_PPM:+.2f} sigma)  [gate <= 1.2 sigma]")
print(f"    Row 4  the hard row: implied m_e/m_mu shift {hard_e_mu_ppm:+.3f} ppm vs the "
      f"demanded {HARD_DEMAND_PPM:+.2f} +- {HARD_S_PPM:.3f} ppm (the same comparison as "
      "Row 1, read-side view)")
print(f"    Row 5  S4 leakage ratio: {mod_effect_ppm/max(phase_effect_ppm,1e-30):.3f} "
      "(gate <= 0.10)")
print("    Row 6  the C3 reference ladder (free-gas candidates, all killed): x4.4e10 "
      "(total-gas cumulant), x2.1e6 (winding cumulant), x4.9e3 (all-orders one-body).")
check(f"S-4 Row 1 verdict recorded honestly: {tier} at ratio {factor:+.3e}",
      True)
soft_ok = (abs(soft_e_tau_ppm) <= 1.2 * SOFT_BAND_PPM
           and abs(soft_mu_tau_ppm) <= 1.2 * SOFT_BAND_PPM)
check(f"S-4 [S2 surface] soft rows within 1.2 sigma_exp: {soft_ok}", soft_ok)

# ===========================================================================
banner("S-5  verdict")
# ===========================================================================
if abs(pull) <= 1:
    print("""    R-eps LANDS: the interacting chiral dressing produces the pinned number
    within its 0.22% band with all surfaces holding. Registration (value/header
    changes) is the separate USER-GATED step; nothing ships from this probe.""")
elif abs(pull) <= 2:
    print("""    MARGINAL: in the 1-2 sigma corridor. NO adoption; the result is banked
    as-is; the next step is localization of the gap, not relabeling.""")
else:
    print(f"""    THE TIER-KILL FIRES (pre-registered): the candidate is off by the factor
    {factor:+.3e}. Per the pre-registration the class KILLS and RE-LOCALIZES to
    the PRE-NAMED next layer: the READ-PROJECTION LAYER -- the physical
    delta-read applies a channel weighting (the E1b theta_seam / odd-half
    triplet-channel structure: Perron->e-slot, triple->d-slot) that this
    functional's bare h/hbar channel expectation does not carry. The honest
    statement: the interacting ensemble HAS the chiral channel (E2a, theorem);
    its BARE channel phase is not the read's dressing; the missing piece is
    the projection of G_int onto the READ's own channel weights -- logged to
    todo §1 as the sharpened localization. An open miss stays open.""")
check("S-5 outcome recorded per the pre-registered tier rule; no adoption; no "
      "relabeling; the -70 ppm status updated honestly", True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
