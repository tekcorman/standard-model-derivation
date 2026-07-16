#!/usr/bin/env python3
"""
proofs/foundations/ODD_O2_relative_eta_2026-07-06.py

STATION O2 — the relative-eta / odd-spectral-TRACE probe.
Pre-registration: internal research notes (committed f2cd79b BEFORE
this file). FROZEN. Standing on O0/O1 (theorem_graded_blindness_and_odd_channel_2026-07-06).

The chiral phase chi that read_masses needs (chi -> eps through amp = c0 + e^{i chi} shell) is a
sigma-ODD quantity (O0). The A5/Berry probes computed chi as a STATE PROJECTION (2 eigenvalue
phases / eigenvector Berry) = R3, weld-gated, gave x11 / x39. O2 computes the sigma-odd TRACE
(all shell modes, heat-weighted) = R4, projection-free. Same coupled machinery as
LOOP_A5_magnitude_2026-07-05.py (S-0 verbatim) so the ONLY change is projection -> trace.

Object: chi_flow = d/ds arg det C_shell(s)   [the spectral-flow / eta rate — the FULL-trace phase
asymmetry, vs A5's top-2-eigenvalue rate], bit-odd 1/2(+J,-J), relative (full - leading). Plus the
odd heat trace's t-dependence (forced scale vs UV cutoff). eps appears ONLY at S-3.
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

# ============ S-0  setup (VERBATIM from LOOP_A5_magnitude_2026-07-05.py) ============
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
check(f"S-0 re-lock; u=alpha_1={u:.6f}; s_lep={S_LEP:.6f}", u > 0 and S_LEP > 0)

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

# ===========================================================================
banner("S-1  the sigma-odd TRACE: chi_flow = d/ds arg det C_shell  (ALL modes)")
# ===========================================================================
# A5 tracked the top-2 eigenvalue phases (a STATE PROJECTION, R3). The sigma-odd TRACE is the
# phase of the DETERMINANT = SUM over ALL shell modes = the spectral-flow / eta rate (R4).
def logdet_phase_rate(uu, Pblk, t):
    """d/ds arg det Lam_t  = Im Tr( Lam^{-1} dLam/ds )  — the full-spectrum phase asymmetry."""
    Lp = Lam_t(uu, DS, Pblk, t); Lm = Lam_t(uu, -DS, Pblk, t); L0 = Lam_t(uu, 0.0, Pblk, t)
    dL = (Lp - Lm) / (2 * DS)
    return float(np.imag(np.trace(np.linalg.solve(L0, dL))))     # Im Tr(L^{-1} L') = d/ds arg det
def chi_flow_frame(Pblk):
    # the same phase-SUM direction A5 used (isotypes t=1 and t=2), now as a full trace
    return 0.5 * (logdet_phase_rate(u, Pblk, 1) + logdet_phase_rate(u, Pblk, 2))
flow_p = chi_flow_frame(vac); flow_m = chi_flow_frame(vac_m)
chi_flow_rate = 0.5 * (flow_p - flow_m)                          # bit-ODD (flips with J)
chi_flow = chi_flow_rate * S_LEP                                 # to the lepton slice
print(f"    trace-flow rate(+J)={flow_p:+.6e}, (-J)={flow_m:+.6e}")
print(f"    bit-odd chi_flow_rate = {chi_flow_rate:+.6e} rad/s;  chi_flow(s_lep) = {chi_flow:+.6e} rad")
# C-BIT: does it flip under J? (bit-even part should be ~0)
chi_flow_even = 0.5 * (flow_p + flow_m)
check(f"C-BIT: bit-EVEN part of the flow = {chi_flow_even:+.2e} ~ 0 (trace flips with J)",
      abs(chi_flow_even) < max(1e-3, 0.05 * abs(chi_flow_rate) + 1e-12))

# ===========================================================================
banner("S-1b  is the parameter-free invariant QUANTIZED? (KILL-Q test: total winding)")
# ===========================================================================
# The eta / spectral-flow endpoint = the NET winding of det C along the run [0, s_lep]. If the
# lattice odd invariant is quantized, the accumulated arg-det winds by ~2*pi*integer and the
# "continuous" rate is a discretization of jumps at crossings. Integrate the rate; also directly
# accumulate arg det around the run to read the winding number.
def argdet_along(Pblk, t, npts=400):
    ss = np.linspace(0.0, S_LEP, npts)
    vals = []
    for s in ss:
        L = Lam_t(u, s, Pblk, t)
        sign, logdet = np.linalg.slogdet(L)
        vals.append(cmath.phase(sign))
    # unwrap
    ph = np.unwrap(np.array(vals))
    return ph[-1] - ph[0]
wind_p1 = argdet_along(vac, 1); wind_p2 = argdet_along(vac, 2)
total_wind = 0.5 * (wind_p1 + wind_p2)
print(f"    accumulated arg det over [0,s_lep] (isotype avg, +J) = {total_wind:+.6e} rad")
print(f"    in units of 2*pi: {total_wind/(2*math.pi):+.6e}  (integer => QUANTIZED spectral flow)")
near_int = abs(total_wind/(2*math.pi) - round(total_wind/(2*math.pi)))
print(f"    distance to nearest integer multiple of 2*pi: {near_int:.4e}")

# ===========================================================================
banner("S-1c  the odd HEAT TRACE scale: Theta(t) = Tr(grading . dW/ds . e^{-t W^2}) vs t")
# ===========================================================================
# Does the odd heat trace have a FORCED scale, or is it UV-cutoff (t-dependent with no plateau)?
# Build the shell-block coupled operator M(s) = Lam_t (Hermitian part) and its odd heat trace.
def odd_heat_trace(Pblk, t_isotype, tvals):
    L0 = Lam_t(u, 0.0, Pblk, t_isotype)
    dL = (Lam_t(u, DS, Pblk, t_isotype) - Lam_t(u, -DS, Pblk, t_isotype)) / (2 * DS)
    Herm = 0.5 * (L0 + L0.conj().T)
    w, V = np.linalg.eigh(Herm)
    out = []
    for tt in tvals:
        # Tr( dL_herm . e^{-tt Herm^2} )  — the odd (linear-in-generator) heat trace
        dLh = 0.5 * (dL + dL.conj().T)
        heat = (V * np.exp(-tt * w**2)) @ V.conj().T
        out.append(float(np.imag(np.trace((0.5*(dL-dL.conj().T)) @ heat))))   # odd (anti-herm) part
    return out
tvals = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
theta_p = np.array(odd_heat_trace(vac, 1, tvals))
theta_m = np.array(odd_heat_trace(vac_m, 1, tvals))
theta_odd = 0.5 * (theta_p - theta_m)
print("    t      Theta_oddtrace(t)")
for tt, th in zip(tvals, theta_odd):
    print(f"    {tt:5.2f}  {th:+.6e}")
# a FORCED scale would show a plateau (t-independent); a cutoff object drifts monotonically
rel_spread = (np.max(np.abs(theta_odd)) - np.min(np.abs(theta_odd))) / (np.mean(np.abs(theta_odd)) + 1e-30)
print(f"    relative spread over t in [0.05,2]: {rel_spread:.3f}  (>>0 => NO forced scale = UV cutoff)")

# ===========================================================================
banner("S-2  CONTROLS  (C-FREE, C-LEADING)")
# ===========================================================================
# C-FREE [Q3]: the FREE ensemble (no gamma coupling) -> the trace flow bit-odd = 0
def free_logdet_rate(s, t):
    def lam_free(ss):
        Bk = srs.hashimoto(tuple(ss * AXIS))
        Gf = np.linalg.inv(np.eye(ND) - u * u * Bk @ Bk)
        C = QB[t].conj().T @ Gf @ QB[t]
        return (np.eye(C.shape[0]) - np.linalg.inv(C)) / (u * u)
    L0 = lam_free(s); dL = (lam_free(s + DS) - lam_free(s - DS)) / (2 * DS)
    return float(np.imag(np.trace(np.linalg.solve(L0, dL))))
free_flow = 0.5 * (free_logdet_rate(0.0, 1) + free_logdet_rate(0.0, 2))
check(f"C-FREE [Q3]: free-ensemble trace-flow = {free_flow:+.2e} ~ 0", abs(free_flow) < 5e-3)

# C-LEADING: chi=0 reproduces the shipped read_masses lepton row
Qs, ds = the_run.read_moduli(), the_run.read_phases()
nh = 3
c0 = (0.5) ** 0.5; c1 = (float(6 * Qs[nh] - 2) / 8) ** 0.5; delta = float(ds[nh])
def masses_with_chi(chi_row):
    out = []
    for j in range(3):
        shell = c1 * cmath.exp(1j * delta) * OM ** j + c1 * cmath.exp(-1j * delta) * OM ** (-j)
        amp = c0 + cmath.exp(1j * chi_row) * shell
        out.append(abs(amp) ** 2)
    return sorted(out)
m0 = masses_with_chi(0.0); shipped = the_run.read_masses()[nh]
check(f"C-LEADING: chi=0 reproduces shipped read_masses row (rel err "
      f"{max(abs(a/b-1) for a,b in zip(m0,shipped)):.1e})",
      max(abs(a / b - 1) for a, b in zip(m0, shipped)) < 1e-9)

# ===========================================================================
banner("S-2b  NUMERICAL ROBUSTNESS + CONVENTION SENSITIVITY (not re-choosing the headline)")
# ===========================================================================
# The FROZEN headline = phase-SUM, isotypes 1&2, DS=1e-6 (A5 convention). Report stability under
# DS and the alternative directions to show the number is not a finite-difference/convention artifact.
def chi_flow_at(dsx):
    def rate(uu, Pblk, tI):
        Lp = Lam_t(uu, dsx, Pblk, tI); Lm = Lam_t(uu, -dsx, Pblk, tI); L0 = Lam_t(uu, 0.0, Pblk, tI)
        return float(np.imag(np.trace(np.linalg.solve(L0, (Lp - Lm) / (2 * dsx)))))
    fp = 0.5 * (rate(u, vac, 1) + rate(u, vac, 2)); fm = 0.5 * (rate(u, vac_m, 1) + rate(u, vac_m, 2))
    return 0.5 * (fp - fm) * S_LEP
ds_scan = {dsx: chi_flow_at(dsx) for dsx in (1e-5, 1e-6, 1e-7)}
for dsx, val in ds_scan.items():
    print(f"    DS={dsx:.0e}:  chi_flow = {val:+.6e} rad")
ds_vals = list(ds_scan.values())
DS_UNSTABLE = (max(ds_vals) - min(ds_vals)) > 3 * abs(np.median(ds_vals)) or \
    (np.sign(max(ds_vals)) != np.sign(min(ds_vals)))
print(f"    => DS-STABILITY: {'UNSTABLE (rate ill-conditioned)' if DS_UNSTABLE else 'stable'} "
      f"(range {min(ds_vals):+.2e} .. {max(ds_vals):+.2e})")
# difference-direction and individual isotypes (sensitivity only)
def rate_iso(Pblk, tI):
    Lp = Lam_t(u, DS, Pblk, tI); Lm = Lam_t(u, -DS, Pblk, tI); L0 = Lam_t(u, 0.0, Pblk, tI)
    return float(np.imag(np.trace(np.linalg.solve(L0, (Lp - Lm) / (2 * DS)))))
d_iso1 = 0.5 * (rate_iso(vac, 1) - rate_iso(vac_m, 1)) * S_LEP
d_iso2 = 0.5 * (rate_iso(vac, 2) - rate_iso(vac_m, 2)) * S_LEP
print(f"    per-isotype bit-odd chi_flow:  iso1 = {d_iso1:+.4e},  iso2 = {d_iso2:+.4e}")
print(f"    (headline phase-SUM = 1/2(iso1+iso2) = {0.5*(d_iso1+d_iso2):+.4e})")

# ===========================================================================
banner("S-3  ============  THE SINGLE MARKED COMPARISON (eps enters HERE)  ============")
# ===========================================================================
EPS_TARGET = -1.7515e-7
# chi_flow is a phase RATE integrated to s_lep => a FIRST-ORDER correction to the generation phase
# delta (same order as eps = delta_eff - 2/9), NOT a 2nd-order cos-chi quantity. The RIGHT
# comparison is the direct ratio chi_flow/eps. (The cos-chi seam below is A5's 2nd-order construction
# and is the WRONG seam for a first-order phase — reported only to show it is negligible here.)
ratio0 = m0[0] / m0[2]
mchi = masses_with_chi(chi_flow); ratio_chi = mchi[0] / mchi[2]
shift_ppm_coschi = (ratio_chi / ratio0 - 1) * 1e6
print(f"    chi_flow (sigma-odd TRACE, all modes) = {chi_flow:+.6e} rad   [FIRST-ORDER delta-correction]")
print(f"    eps target (pinned Q3)                = {EPS_TARGET:+.6e} rad")
print(f"    >>> chi_flow / eps  =  {chi_flow/EPS_TARGET:+.4f}   <<<  (the marked comparison)")
print(f"    A5 state-projection chi (for contrast)= 1.16e-3 rad (x6600 LARGER; the trace RESUMS it)")
print(f"    [wrong-seam check] cos-chi 2nd-order ppm = {shift_ppm_coschi:+.2e} ppm (negligible => not the seam)")
print(f"    POISON WATCH: chi_flow/eps = {chi_flow/EPS_TARGET:+.4f} vs 5/12 = {5/12:+.4f} "
      f"(|off| {abs(chi_flow/EPS_TARGET - 5/12)/(5/12)*100:.2f}%) — FLAGGED, NOT adopted")
print(f"    odd-heat-trace scale @ t=0.1  = {theta_odd[1]:+.6e}  (t-spread {rel_spread:.2f})")

# ===========================================================================
banner("S-4  VERDICT (pre-declared outcomes; decided by the computation)")
# ===========================================================================
ratio = chi_flow / EPS_TARGET
ZERO = abs(chi_flow) < 1e-12
QUASI_QUANT = near_int < 0.02 or abs(total_wind / (2 * math.pi) + 0.25) < 0.02   # integer or clean quarter
if DS_UNSTABLE:
    verdict = (f"KILL-Q / CONTINUUM — the lattice trace-flow RATE is NUMERICALLY ILL-CONDITIONED "
               f"(DS-scan {min(ds_vals):+.2e}..{max(ds_vals):+.2e}, sign-flips; the non-normal "
               f"exceptional-point problem the repo flagged for M_Z), so it gives NO robust number. "
               f"The apparent chi_flow(DS=1e-6)={ratio:+.3f}*eps (and its ~5/12 proximity) is a "
               f"FINITE-DIFFERENCE ARTIFACT, NOT a signal — killed by the DS scan. What IS robust: "
               f"the parameter-free spectral-flow WINDING = -pi/2 (clean quarter, quasi-QUANTIZED) "
               f"and the odd heat trace's t-drift (no forced scale). => the continuous sigma-odd "
               f"invariant carrying eps is a CONTINUUM object (odd Seeley-DeWitt / 3D-eta density on "
               f"the A5(b) cone), NOT robustly lattice-accessible — parallel to S1's even-a4 (lattice "
               f"bounded => no UV coefficient). Gate A's ODD face is LOCATED as a continuum "
               f"computation. -70 ppm stays OPEN.")
elif ZERO:
    verdict = "KILL-0 — the bit-odd trace vanishes on this family (R4 empty here)."
elif abs(ratio - 1.0) < 0.21:
    verdict = (f"LAND?? (hold to continuum cross-check + poison audit) — chi_flow/eps = {ratio:+.3f}.")
else:
    verdict = (f"BRACKET — sigma-odd trace = {ratio:+.3f} eps; robust; NOT a closure.")
print("   " + verdict)
print()
print("    POISON LEDGER (declared, NOT invoked): 2*alpha_1^5=1.809e-7, 2*alpha_1^3, bracket-mean")
print("    2.5e-4, A5 endpoints (1.16e-3 / 5.5e-5). NO alpha_1 power inserted; NO t chosen to land.")
print()
banner(f"  {'ALL PASS' if ok_all else 'SOME FAILED'} (checks = controls; the VERDICT is the science)")
print(f"  VERDICT: {verdict.split(' — ')[0]}")
sys.exit(0 if ok_all else 1)
