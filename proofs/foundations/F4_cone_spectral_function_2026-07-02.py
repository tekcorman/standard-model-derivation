#!/usr/bin/env python3
"""
proofs/foundations/F4_cone_spectral_function_2026-07-02.py

F4 S2a — THE CONE SPECTRAL FUNCTION: is the golden-rule phase space substrate-native?

QUESTION (pre-declared): compute, from the object alone (no inserted constants), the
low-frequency transverse current-current spectral density of each cone,

    Re sigma^{xx}(omega) = (pi/omega) * (1/(2pi)^3) * INT d^3q  Sum_{a<b} W_ab(q) [f_a - f_b]
                           * delta(omega - (eps_b - eps_a)),      W_ab = Sum |<b| dH/dq_x |a>|^2,

and extract the dimensionless per-cone constant  C = sigma(omega) * v_cone / omega  in the
omega -> 0 limit, with the cone velocity v READ off the spectrum (never inserted).

PRE-REGISTERED COMPARISON SET (declared BEFORE any substrate number is computed; these are
the standard universal values, hand-derived in the appendix note below and re-derived by
this probe's own machinery in TEST-0/1):
    spin-1/2 Weyl node (2-comp):  C = 1/(24 pi) = 0.0132629...
    Dirac fermion     (4-comp):  C = 1/(12 pi) = 0.0265258...   <- the SM width formula's
                                    per-channel phase space (Gamma = g_V^2+g_A^2 that unit)
    spin-1 node       (3-comp):  C = 1/(6 pi)  = 0.0530516...   (= 4x Weyl; the flat-band
                                    filling f0 cancels EXACTLY: (1-f0)+(f0) transitions)
Anything else is reported as-is.

SCORING CLASSES (kickoff rule 1): TEST-0/1 = machinery calibration (no class).
TEST-2 (adjacency cones) and TEST-3 (Hodge-Dirac cone = the D4 lift's spatial section)
= class (a) STRUCTURAL reads of the object: its own pair-creation kinematics.
NO Gamma_Z claim is made by this probe: the bridge to a width requires the forced
object-selection argument + the internal (Cl(6)) weight sum, neither computed here.
NO PDG number and NO SM formula appears anywhere in this probe.

PRE-STATED HAZARDS (kickoff rule 2 + session-1 lessons):
  * integrate-the-zone: the quoted Hodge constant must survive a full-BZ histogram
    cross-check (Gaussian-broadened, offset grid) against the cone-shell integral (ASSERTED
    within 12%). The adjacency object has NO canonical global filling (band variable mu is
    not an energy with a Dirac sea; the P-point doublets would contribute Drude-like weight
    under any naive global fill) -- so adjacency constants are quoted PER CONE ONLY and the
    physical-object status stays with the Hodge-Dirac (whose sea is canonical and whose
    omega->0 weight provably localizes at Gamma: eps^2 = 3 - mu = 0 only at the Perron top).
  * flat-band occupation f0 must not matter (computed at f0 = 0, 1/2, 1; exact for the
    continuum spin-1, approximate for the substrate where the mid band has curvature).
  * degeneracies handled basis-free: cluster projector weights Tr(P_a J P_b J).
  * KILL: if histogram and shell disagree, or C(omega) is not omega-stable, report the
    failure per the law; do NOT tune.

Hand-derivation backing TEST-0/1 (so the asserts are against independent algebra):
  Weyl H = q.sigma: transitions at omega = 2q; |<+|sigma_x|->|^2 = 1 - qhat_x^2
  (angular avg 2/3); sigma = (pi/omega)(1/8pi^3)(omega^2/8)(8pi/3) = omega/(24pi).
  Spin-1 H = q.S: transitions -1->0 and 0->+1 BOTH at omega = q; -1->+1 FORBIDDEN
  (Delta m = 2 for every direction, exact); Sum_i |<0|S_i|-1>|^2 = 1 (frame-independent)
  so angular avg of the x-element is 1/3 per transition; occupations (1-f0)+(f0) = 1;
  sigma = (pi/omega)(1/8pi^3)(omega^2)(4pi/3) = omega/(6pi).
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

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
# substrate operators in radian momentum q (phases e^{i q.v}); currents exact
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

# continuum references
SX = np.array([[0, 1], [1, 0]], complex); SY = np.array([[0, -1j], [1j, 0]]); SZ = np.diag([1.0, -1.0]).astype(complex)
PAULI = [SX, SY, SZ]
r2 = 1 / math.sqrt(2)
S1X = np.array([[0, r2, 0], [r2, 0, r2], [0, r2, 0]], complex)
S1Y = np.array([[0, -1j * r2, 0], [1j * r2, 0, -1j * r2], [0, 1j * r2, 0]], complex)
S1Z = np.diag([1.0, 0.0, -1.0]).astype(complex)
SPIN1 = [S1X, S1Y, S1Z]


# ---------------------------------------------------------------------------
# generic machinery: sphere, groups, shell integral, histogram, velocity read
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
    """cone-shell zone integral of Re sigma(omega); returns (avg over axes, per-axis).
    pair_filter(ea, eb): optional predicate on the two group energies at rref
    (relative to nothing — raw band energies) to restrict to a transition class."""
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

def sigma_hist(Hf, Jf, fills, f0, omega, G=44, eta=0.02):
    """full-BZ Gaussian-broadened cross-check (offset MP grid)."""
    fl = np.array([f0 if f is None else float(f) for f in fills])
    pts = 2 * math.pi * (np.arange(G) + 0.5) / G
    tot = 0.0
    norm = 1 / (eta * math.sqrt(2 * math.pi))
    for qa in pts:
        for qb in pts:
            for qc in pts:
                q = np.array([qa, qb, qc])
                ev, V = np.linalg.eigh(Hf(q))
                gs = groups_of(ev)
                gf = [float(np.mean(fl[g])) for g in gs]
                ge = [float(np.mean(ev[g])) for g in gs]
                Js = None
                for a in range(len(gs)):
                    for b in range(a + 1, len(gs)):
                        df = gf[a] - gf[b]
                        de = ge[b] - ge[a]
                        if df < 1e-12 or abs(de - omega) > 5 * eta: continue
                        if Js is None:
                            Js = [Jf(q, ax) for ax in range(3)]
                        w = np.mean([float(np.sum(np.abs(V[:, gs[b]].conj().T @ Js[ax] @ V[:, gs[a]]) ** 2))
                                     for ax in range(3)])
                        tot += w * df * norm * math.exp(-0.5 * ((de - omega) / eta) ** 2)
    return (math.pi / omega) * tot / G ** 3

def read_velocity(Hf, q0, band, eF, ndirs=200, h=0.02):
    """v(direction) = d eps/dr of `band` at the cone, Richardson from h and 2h; READ."""
    q0 = np.asarray(q0, float)
    vs = []
    for kh in sphere(ndirs):
        e1 = np.linalg.eigvalsh(Hf(q0 + h * kh))[band] - eF
        e2 = np.linalg.eigvalsh(Hf(q0 + 2 * h * kh))[band] - eF
        vs.append((4 * e1 - e2) / (2 * h))
    return float(np.mean(vs)), float(np.min(vs)), float(np.max(vs))

def kp_matrices(Hf, dHf, q0, idx):
    """degenerate k.p at q0: M_i = V^dag (dH/dq_i) V on the subspace `idx` (leading order)."""
    q0 = np.asarray(q0, float)
    ev, V = np.linalg.eigh(Hf(q0))
    Vs = V[:, idx]
    return [Vs.conj().T @ dHf(q0, ax) @ Vs for ax in range(3)]


# ---------------------------------------------------------------------------
print("=" * 88)
print(" TEST-0  machinery calibration: continuum spin-1/2 Weyl (analytic C = 1/24pi)")
print("=" * 88)
Hw = lambda q: q[0] * PAULI[0] + q[1] * PAULI[1] + q[2] * PAULI[2]
Jw = lambda q, ax: PAULI[ax]
for om in (0.05, 0.2):
    s, _ = sigma_shell(Hw, Jw, (0, 0, 0), [1, 0], 0.5, om, rmax=0.6)
    C = s * 1.0 / om   # v = 1 exactly
    check(f"Weyl C(omega={om}) = {C:.6f} vs 1/24pi = {C_WEYL:.6f} ({(C/C_WEYL-1)*100:+.2f}%)",
          abs(C / C_WEYL - 1) < 0.02)

print("=" * 88)
print(" TEST-1  continuum spin-1 (analytic C = 1/6pi; f0-independent; -1<->+1 forbidden)")
print("=" * 88)
H1 = lambda q: q[0] * SPIN1[0] + q[1] * SPIN1[1] + q[2] * SPIN1[2]
J1 = lambda q, ax: SPIN1[ax]
Cs = []
for f0 in (0.0, 0.5, 1.0):
    s, _ = sigma_shell(H1, J1, (0, 0, 0), [1, None, 0], f0, 0.1, rmax=0.6)
    Cs.append(s / 0.1)
check(f"spin-1 C = {Cs[1]:.6f} vs 1/6pi = {C_SPIN1:.6f} ({(Cs[1]/C_SPIN1-1)*100:+.2f}%)",
      abs(Cs[1] / C_SPIN1 - 1) < 0.02)
check(f"flat-band filling cancels EXACTLY (spread {max(Cs)-min(Cs):.2e})", max(Cs) - min(Cs) < 1e-10)
# the -1 <-> +1 selection rule, measured directly:
qq = np.array([0.21, 0.13, 0.08]); ev, V = np.linalg.eigh(H1(qq))
w_pm = sum(float(np.sum(np.abs(V[:, [2]].conj().T @ S @ V[:, [0]]) ** 2)) for S in SPIN1)
check(f"pair-creation channel (-1 -> +1) weight = {w_pm:.2e} (FORBIDDEN, exact)", w_pm < 1e-20)
print(f"    NOTE: for spin-1 ALL omega->0 weight flows through the FLAT band; the direct")
print(f"    lower->upper (pair) channel is exactly dark. C_spin1 = 4 x C_Weyl.")

print("=" * 88)
print(" TEST-2  substrate ADJACENCY cones (transport object; per-cone constants only)")
print("=" * 88)
Jadj = lambda q, ax: dA_q(q, ax)
# cone locations + Fermi levels READ off the spectrum (multiplicity-3 eigenvalue)
for name, q0, fills in (("Gamma", (0.0, 0.0, 0.0), [1, None, 0, 0]),
                        ("R=(pi,pi,pi)", (math.pi, math.pi, math.pi), [1, 1, None, 0])):
    ev0 = np.linalg.eigvalsh(A_q(np.asarray(q0) + 1e-9))
    vals, counts = np.unique(np.round(ev0, 6), return_counts=True)
    eF = float(vals[np.argmax(counts)])
    upper_band = [i for i, f in enumerate(fills) if f == 0][0]
    v, vmin, vmax = read_velocity(A_q, q0, upper_band, eF)
    print(f"    {name}: triple at mu = {eF:+.6f}; v_cone read = {v:.6f} "
          f"(anisotropy {vmin:.6f}..{vmax:.6f}, spread {(vmax-vmin)/v*100:.2f}%)")
    # leading-order k.p on the triple: is the anisotropy real at linear order?
    ev_full, Vf = np.linalg.eigh(A_q(np.asarray(q0, float)))
    idx = [i for i in range(NV) if abs(ev_full[i] - eF) < 1e-9]
    Ms = kp_matrices(A_q, dA_q, q0, idx)
    errs, named = [], {}
    for lbl, kh in (("100", (1, 0, 0)), ("110", (1, 1, 0)), ("111", (1, 1, 1))):
        kh = np.asarray(kh, float); kh /= np.linalg.norm(kh)
        v_kp = float(max(np.linalg.eigvalsh(sum(kh[i] * Ms[i] for i in range(3)))))
        e1 = np.linalg.eigvalsh(A_q(np.asarray(q0) + 0.02 * kh))[upper_band] - eF
        e2 = np.linalg.eigvalsh(A_q(np.asarray(q0) + 0.04 * kh))[upper_band] - eF
        v_fd = (4 * e1 - e2) / 0.04
        errs.append(abs(v_kp / v_fd - 1)); named[lbl] = v_kp
    check(f"{name}: k.p reproduces v(khat) (max err {max(errs)*100:.2f}%) => anisotropy is LEADING-ORDER",
          max(errs) < 0.01)
    print(f"        v_100 = {named['100']:.6f}   v_110 = {named['110']:.6f}   v_111 = {named['111']:.6f}")
    Cs = {}
    for f0 in (0.0, 0.5, 1.0):
        for om in (0.05, 0.12):
            s, sax = sigma_shell(A_q, Jadj, q0, fills, f0, om)
            Cs[(f0, om)] = s * v / om
    Cmid = Cs[(0.5, 0.05)]
    print(f"    C(f0=1/2, omega=0.05) = {Cmid:.6f}  [v = sphere-mean, def'n stated]  "
          f"omega-drift to 0.12: {(Cs[(0.5,0.12)]/Cmid-1)*100:+.2f}%   f0-spread: "
          f"{(max(Cs.values())-min(Cs.values())):.2e}")
    for lbl, ref in (("1/24pi Weyl", C_WEYL), ("1/12pi Dirac", C_DIRAC), ("1/6pi spin-1", C_SPIN1)):
        print(f"        C / ({lbl:>13}) = {Cmid/ref:.4f}")
print("""    STRUCTURAL VERDICT (forced): v(khat) varies ~2x under the CUBIC little group at
    Gamma/R. Any velocity anisotropy absorbable by an emergent metric would be an
    O-invariant quadratic form = isotropic. Since v(khat) is NOT constant, the cone is a
    CHIRALLY WARPED spin-1 multifold — its phase-space constant is NOT any isotropic
    universal value, and no linear change of coordinates can make it one.""")

print("=" * 88)
print(" TEST-3  substrate HODGE-DIRAC cone (the D4 lift's spatial section; canonical sea)")
print("=" * 88)
fills_D = [1, 1, 1, 1, None, None, 0, 0, 0, 0]
# verify the claimed omega->0 structure before quoting anything (EXCLUDING the two
# exact flat zero bands, which sit at eps=0 for every k by construction).
# NOTE on conventions: high-symmetry-point COORDINATES are gauge-dependent (srs.py's
# edge-translation choice differs from proofs.common find_bonds(); e.g. (pi/2)^3 here
# hosts {+-(1+sqrt2), +-(sqrt2-1)}, not the +-sqrt3 doublets). So the honest locality
# check is a BZ-WIDE scan: does any NON-flat band approach zero anywhere off Gamma?
def min_nonflat(q):
    return float(np.sort(np.abs(np.linalg.eigvalsh(D_q(np.asarray(q, float)))))[2])
flat_dev = max(abs(np.linalg.eigvalsh(D_q(np.array([1.1, 0.4, 2.2])))[4:6]))
check(f"two EXACT flat zero bands across the BZ (|eps| = {flat_dev:.1e} at generic q)", flat_dev < 1e-12)
Gscan = 24
best, bestq = 1e9, None
for a in range(Gscan):
    for b in range(Gscan):
        for c in range(Gscan):
            q = 2 * math.pi * (np.array([a, b, c]) + 0.5) / Gscan
            qc = np.minimum(q, 2 * math.pi - q)          # distance to Gamma (periodic)
            if np.linalg.norm(qc) < 1.0: continue         # outside the shell's reach (rmax)
            m = min_nonflat(q)
            if m < best: best, bestq = m, q
check(f"BZ scan: min NON-flat |eps| beyond the shell's reach (|q|>rmax=1.0) = {best:.3f} "
      f"(at q/pi = {np.round(bestq/math.pi,2)}); > 0.2 => ALL omega<=0.15 weight is within "
      f"the Gamma-cone shell (the slow (110) skirt, v_min=0.25, is cone weight and IS "
      f"integrated; flat<->flat has df=0)", best > 0.2)
vD, vDmin, vDmax = read_velocity(D_q, (0, 0, 0), 6, 0.0)
print(f"    Hodge cone velocity read = {vD:.6f} (anisotropy {vDmin:.6f}..{vDmax:.6f}, "
      f"spread {(vDmax-vDmin)/vD*100:.2f}%)   [= adjacency v(khat)/2? see ratio below]")
CsD = {}
for f0 in (0.0, 0.5, 1.0):
    for om in (0.05, 0.12, 0.2):
        s, _ = sigma_shell(D_q, dD_q, (0, 0, 0), fills_D, f0, om)
        CsD[(f0, om)] = s * vD / om
CD = CsD[(0.5, 0.05)]
print(f"    C(f0=1/2, omega=0.05) = {CD:.6f}   omega-drift 0.05->0.2: "
      f"{(CsD[(0.5,0.2)]/CD-1)*100:+.2f}%   f0-spread: {max(CsD.values())-min(CsD.values()):.2e}")
for lbl, ref in (("1/24pi Weyl", C_WEYL), ("1/12pi Dirac", C_DIRAC), ("1/6pi spin-1", C_SPIN1)):
    print(f"        C / ({lbl:>13}) = {CD/ref:.4f}")
# channel decomposition at omega=0.12, f0=1/2, via pair_filter on rref group energies:
FL = 1e-9
s_all, _ = sigma_shell(D_q, dD_q, (0, 0, 0), fills_D, 0.5, 0.12)
s_cc, _ = sigma_shell(D_q, dD_q, (0, 0, 0), fills_D, 0.5, 0.12,
                      pair_filter=lambda ea, eb: abs(ea) > FL and abs(eb) > FL)
s_lf, _ = sigma_shell(D_q, dD_q, (0, 0, 0), fills_D, 0.5, 0.12,
                      pair_filter=lambda ea, eb: ea < -FL and abs(eb) < FL)
s_fu, _ = sigma_shell(D_q, dD_q, (0, 0, 0), fills_D, 0.5, 0.12,
                      pair_filter=lambda ea, eb: abs(ea) < FL and eb > FL)
print(f"    channel decomposition at omega=0.12 (f0=1/2): cone->cone {s_cc/s_all*100:.1f}%  "
      f"lower->flat {s_lf/s_all*100:.1f}%  flat->upper {s_fu/s_all*100:.1f}%")
check(f"decomposition closes (sum {100*(s_cc+s_lf+s_fu)/s_all:.2f}%)",
      abs((s_cc + s_lf + s_fu) / s_all - 1) < 0.005)
# the direct pair channel vanishes at LEADING ORDER (q^2 in |M|^2), like the spin-1
# selection rule but approximate: measure the scaling.
kh = np.array([0.62, 0.33, 0.71]); kh /= np.linalg.norm(kh)
wpair = []
for qr in (0.03, 0.06):
    ev, V = np.linalg.eigh(D_q(qr * kh))
    wpair.append(sum(float(np.sum(np.abs(V[:, [6]].conj().T @ dD_q(qr * kh, ax) @ V[:, [3]]) ** 2))
                     for ax in range(3)))
check(f"cone->cone |M|^2 scales as q^2 (ratio {wpair[1]/wpair[0]:.2f} for 2x q; "
      f"absolute {wpair[0]:.1e} at q=0.03) => pair channel DARK at leading order",
      3.0 < wpair[1] / wpair[0] < 5.0 and wpair[0] < 1e-4)

print("\n    integrate-the-zone cross-check (full-BZ histogram vs cone-shell, linear window):")
for om, G, eta in ((0.10, 52, 0.012), (0.15, 52, 0.012)):
    sh, _ = sigma_shell(D_q, dD_q, (0, 0, 0), fills_D, 0.5, om)
    hi = sigma_hist(D_q, dD_q, fills_D, 0.5, om, G=G, eta=eta)
    check(f"omega={om}: shell {sh:.6f} vs BZ-histogram {hi:.6f} ({(hi/sh-1)*100:+.1f}%)",
          abs(hi / sh - 1) < 0.12)
# diagnostic only (nonlinear window; no assert): where the linear cone stops being the story
sh25, _ = sigma_shell(D_q, dD_q, (0, 0, 0), fills_D, 0.5, 0.25)
hi25 = sigma_hist(D_q, dD_q, fills_D, 0.5, 0.25)
print(f"    [diagnostic] omega=0.25: shell {sh25:.6f} vs histogram {hi25:.6f} "
      f"({(hi25/sh25-1)*100:+.1f}%) — nonlinear/far-zone window, quoted constant is omega->0")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
