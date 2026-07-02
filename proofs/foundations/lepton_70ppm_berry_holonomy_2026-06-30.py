#!/usr/bin/env python3
"""
proofs/foundations/lepton_70ppm_berry_holonomy_2026-06-30.py

THE BERRY-HOLONOMY ATTACK on the open charged-lepton -70 ppm miss
(an internal working note Step 2b flagged it as a concrete UNTRIED next probe;
docs/incomplete_equations_todo.md item 1).

LEADING READ (the_run.read_generation): the 3 generation modes are the C3-winding
PF-dominant poles, moduli FROZEN at the Gamma values {2,sqrt2,sqrt2}, evolved by the
LINEAR dynamical phase exp(+- i phi s), phi = 2pi/sqrt7.  At the lepton slice s_lep
(phi*s_lep = 2/9, the Koide phase -- FALLS OUT) this leaves the -70.3/-60.5 ppm
winding-asymmetric residual.  The geometric next-order the leading read DROPS is the
Berry connection / holonomy of D4 = d_s + B(s*AXIS), AXIS=(1,-1,1)/sqrt3.

HYPOTHESIS (build_dN Step 2b): the open-path transport over-applies (~1e4 ppm) but the
gauge-invariant CLOSED-LOOP Berry holonomy may cancel it, leaving the O(alpha1^3) ~60 ppm.

RESULT (this script): FALSIFIED -- the Berry route OVER-APPLIES, 10th route ruled out.
  * The genuine OPERATOR period along the screw is sqrt3 (B(sqrt3*AXIS)=B(0) to 1e-16),
    NOT sqrt7 (sqrt7 is the eigenvalue-PHASE period; the doc conflated them).  The true
    closed loop is s in [0,sqrt3]; over it arg(h) advances 2pi*sqrt(3/7) (the {0,+-} gen phase).
  * Abelian closed-loop Berry phase per winding = EXACTLY {-pi, 0, 0} -- purely
    TOPOLOGICAL (Z2: the Perron winding flips sign; the two shells get 0).  Carries NO
    continuous ~60 ppm content.
  * Open-path geometric phase to s_lep = O(0.1-1 rad) ~ 1e4-1e5 ppm, ~1e4x the target,
    and gauge-dependent.  Shell relative phase 1.03 rad = 17000x the ~6e-5 rad needed.
  * Non-abelian (inter-winding) holonomy BREAKS DOWN at the Perron->shell mode crossings
    (det W -> 0, shell amplitudes -> 0) -- the same crossing pathology that over-applies
    every transport/curvature route.
  CONCLUSION: the closed loop DOES cancel the over-application (build_dN's conjecture) --
  but to EXACTLY 0/-pi (topological), not to ~60 ppm.  The -70 ppm is NOT a band-geometric
  (Berry) effect.  The spectral/run/geometric ROUTE to it is now FULLY exhausted (10 routes);
  only the continuum-D4 Dirac-cone spectral action remains (research-level, unbuilt).
  The miss stays OPEN; what is ruled out is the Berry route.  No fit was made.
"""
import sys, cmath
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "derivation_topdown" / "dirac_srs_mdl"))
import srs

K = 3; TWO_PI = 2*np.pi; LAM_3IRREP = -1.0
AXIS = np.array([1.0, -1.0, 1.0]) / np.sqrt(3)
D = srs._darts(); NB = len(D); hashimoto = srs.hashimoto
om = cmath.exp(TWO_PI*1j/3)
phi = TWO_PI / np.sqrt(4*(K-1) - LAM_3IRREP**2)        # 2pi/sqrt7


def c3_winding_bases():
    sigma = {0:0, 1:2, 2:3, 3:1}; P = np.zeros((NB, NB))
    for a, (i, j, v) in enumerate(D):
        for b, (p, q, w) in enumerate(D):
            if (p, q) == (sigma[i], sigma[j]): P[b, a] = 1; break
    wom = cmath.exp(TWO_PI*1j/3); out = []
    for t in (0, 1, 2):
        Pc = sum(wom**(-t*m) * np.linalg.matrix_power(P, m) for m in range(3)) / 3
        ev, V = np.linalg.eigh(Pc); out.append(V[:, np.abs(ev - 1) < 1e-6])
    return out


bases = c3_winding_bases()
B0 = hashimoto((0, 0, 0))
c0 = [abs(np.linalg.eigvals(Q.conj().T @ B0 @ Q)[np.argmax(np.abs(np.linalg.eigvals(Q.conj().T @ B0 @ Q)))])
      for Q in bases]


def leading_masses(s):
    amp = [c0[0], c0[1]*cmath.exp(1j*phi*s), c0[2]*cmath.exp(-1j*phi*s)]
    return [abs(sum(amp[t]*om**(t*j) for t in range(3)))**2 for j in range(3)]


def abelian_berry(ns, smax):
    """per-winding-dominant abelian Berry phase (Wilson product of biorthogonal links)."""
    sgrid = np.linspace(0, smax, ns+1); res = []
    for Q in bases:
        prev_r = None; rs = []; ls = []
        for s in sgrid:
            M = Q.conj().T @ hashimoto(s*AXIS) @ Q
            ev, VR = np.linalg.eig(M); VL = np.linalg.inv(VR)
            idx = (np.argmax(np.abs(ev)) if prev_r is None
                   else int(np.argmax([abs(np.vdot(prev_r, VR[:, m])) for m in range(len(ev))])))
            r = VR[:, idx]; l = VL[idx, :] / (VL[idx, :] @ VR[:, idx]); prev_r = r
            rs.append(r); ls.append(l)
        links = [cmath.phase(ls[i] @ rs[i+1]) for i in range(len(sgrid)-1)]
        res.append((-np.sum(links), np.array(links)))
    return sgrid, res


def nonabelian_wilson(ns, smax, s_lep):
    sgrid = np.linspace(0, smax, ns+1); idx_lep = int(round(s_lep/smax*ns))
    Rprev = None; Rs = []
    for s in sgrid:
        cols = []
        for Q in bases:
            M = Q.conj().T @ hashimoto(s*AXIS) @ Q
            ev, VR = np.linalg.eig(M); cols.append(Q @ VR[:, np.argmax(np.abs(ev))])
        Rf = np.column_stack(cols)
        if Rprev is not None:
            for c in range(3):
                ph = np.vdot(Rprev[:, c], Rf[:, c])
                if abs(ph) > 1e-12: Rf[:, c] *= np.conj(ph)/abs(ph)
        Rs.append(Rf); Rprev = Rf
    W = np.eye(3, dtype=complex); L_lep = np.eye(3, dtype=complex)
    for i in range(ns):
        A, Bn = Rs[i], Rs[i+1]
        Llink = np.linalg.solve(A.conj().T @ A, A.conj().T @ Bn)
        W = Llink @ W
        if i < idx_lep: L_lep = Llink @ L_lep
    return W, L_lep


if __name__ == "__main__":
    print("=" * 78)
    print("  Berry-holonomy attack on the charged-lepton -70 ppm miss")
    print("=" * 78)

    # (0) leading read + slice
    me, mmu, mtau = 0.51099895, 105.6583755, 1776.86
    Re, Rmu = me/mtau, mmu/mtau
    best = None
    for s in np.linspace(1e-4, 0.3, 60000):
        m = sorted(leading_masses(s)); re, rm = m[0]/m[2], m[1]/m[2]
        err = (re/Re-1)**2 + (rm/Rmu-1)**2
        if best is None or err < best[0]: best = (err, s)
    s_lep = best[1]
    m = sorted(leading_masses(s_lep))
    print(f"\n(0) leading read: moduli {np.round(c0,4)}, phi*s_lep = {phi*s_lep:.6f} = Koide delta (2/9={2/9:.6f})")
    print(f"    m_e/m_tau resid {(m[0]/m[2]/Re-1)*1e6:+.1f} ppm, m_mu/m_tau resid {(m[1]/m[2]/Rmu-1)*1e6:+.1f} ppm")
    print(f"    (canonical live baseline -70.3 / -60.5 ppm; same O(100 ppm) winding-asym class)")

    # (1) operator period
    print("\n(1) operator period along the screw:")
    for s in (np.sqrt(3), np.sqrt(7)):
        d = np.max(np.abs(hashimoto(s*AXIS) - B0))
        print(f"    ||B({s:.4f}*AXIS)-B(0)|| = {d:.1e}   {'== B(0): CLOSED LOOP' if d<1e-9 else '!= B(0)'}")
    print("    => genuine closed loop is s in [0, sqrt3]  (sqrt7 = eigenvalue-phase period only)")

    # (2) abelian closed-loop holonomy + open path
    sgrid, res = abelian_berry(2400, np.sqrt(3))
    idx_lep = int(round(s_lep/np.sqrt(3)*2400))
    print("\n(2) abelian Berry holonomy:")
    for t, (gclosed, links) in enumerate(res):
        g = (gclosed+np.pi) % TWO_PI - np.pi
        gopen = -np.sum(links[:idx_lep])
        print(f"    winding t={t}: closed-loop = {np.degrees(g):+8.3f} deg  |  open-path to s_lep = {gopen:+.4e} rad")
    print("    => closed loop = {-pi, 0, 0} TOPOLOGICAL; open path O(0.1-1 rad) ~ 1e4-1e5 ppm (over-applies)")

    # (3) non-abelian
    W, L_lep = nonabelian_wilson(3600, np.sqrt(3), s_lep)
    evW = np.linalg.eigvals(W)
    rel = cmath.phase(np.diag(L_lep)[1]) - cmath.phase(np.diag(L_lep)[2])
    print("\n(3) non-abelian (inter-winding) holonomy:")
    print(f"    |det W| = {abs(np.linalg.det(W)):.3e}  (->0: subspace collapses at the Perron->shell crossings)")
    print(f"    shell relative open-path phase = {abs(rel):.3e} rad = {abs(rel)/6e-5:.0f}x the ~6e-5 rad target")

    print("\n" + "=" * 78)
    print("VERDICT: Berry route RULED OUT (10th).  Closed loop cancels the over-application")
    print("to EXACTLY 0/-pi (topological), NOT to ~60 ppm; open path & non-abelian over-apply")
    print("at the mode crossings.  The -70 ppm is NOT a band-geometric effect -- it is the")
    print("O(alpha1^3) Dyson diagram (conjecture-grade).  Spectral/geometric ROUTE exhausted;")
    print("only the continuum-D4 cone remains (research-level).  Miss stays OPEN; no fit made.")
    print("=" * 78)
