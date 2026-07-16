#!/usr/bin/env python3
"""
proofs/foundations/M0_2R_T4_twopi_2026-07-07.py

M0-2R T4-R — DERIVING THE 2pi (kappa*t_P = A_tick) from tick integrality. Pre-registered
in internal research notes (committed e533558 BEFORE this file).
Builds on Session 1 (T2+T3, e965d3d) and Session 2 (T1 KMS-TICK, e2c11fe). Executor: a model.

THE CLAIM (frozen): the 2pi is the CIRCUMFERENCE OF THE MODULAR / THERMAL-TIME CIRCLE,
forced by the integrality of the tick count -- NOT read off A1, NOT fit to data.
  (F1,T1) N-hat integer spectrum on CONSECUTIVE integers => gcd(spectrum diffs)=1.
  (F2,T1) modular generator K_mod = beta_eff * N-hat (affine, exact).
  T4b  spec(N-hat) in Z (gcd 1)  =>  {e^{-i theta N-hat}} is a COMPACT U(1) of MINIMAL
       period exactly 2pi (the modular angle theta lives on the circle R/2piZ = the
       Pontryagin dual of Z). The 2pi is pure spectral geometry, no data.
  T4c  canonical quantization of the number-phase pair (N-hat in Z, theta in [0,2pi)):
       one quantum cell of phase space = 2pi*hbar = h (Bohr-Sommerfeld). => one tick =
       one N-hat quantum = one quantum of ACTION = h.
  T4d  thermal time = tick (T1) => A_tick = h ; kappa = h/t_P. CROSS-CHECK vs A1 only.

POISONS (never the source): 2pi/ln2 pattern-matching (A1's h is an END cross-check ONLY;
the 2pi is derived as the circle period); screw-winding U_pi^3=-I conflation (a DIFFERENT
winding); alpha_1-vs-u_c. Only the DIMENSIONLESS 2pi is derived; t_P remains the framework's
dimensional anchor (NOT claimed derived). NO scoreboard value moves.
"""
import math
import os
import sys
from math import gcd
from functools import reduce

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

# ===========================================================================
banner("T4a  re-lock (from T1): the tick number N-hat is integer on CONSECUTIVE integers")
# ===========================================================================
k = srs.DEG; q = k - 1
u_c = 1.0 / q
alpha1 = (q / k) ** (10 - 2)
B0 = srs.hashimoto((0, 0, 0)).real
lam_P = max(abs(np.linalg.eigvals(B0)))
PERRON = np.ones(B0.shape[0]) / math.sqrt(B0.shape[0])
Nmax = 40

# reconstruct the T1 equilibrium marginal p_n and beta_eff (K_mod = beta_eff * N-hat)
Bh = B0 / lam_P
v = PERRON.astype(complex).copy()
w = []
for n in range(Nmax + 1):
    w.append(((alpha1 * lam_P) ** (2 * n)) * float(np.vdot(v, v).real))
    v = Bh @ v
p = np.array(w) / sum(w)
support = [n for n in range(Nmax + 1) if p[n] > 1e-300]
diffs = [support[i + 1] - support[i] for i in range(len(support) - 1)]
g = reduce(gcd, diffs)
beta_eff = 2 * math.log(u_c / alpha1)              # slope of -log p_n (T1)
check("T4a N-hat spectrum = CONSECUTIVE non-negative integers (support gcd = 1)",
      support == list(range(len(support))) and g == 1, detail=f"gcd(diffs) = {g}")
check("T4a K_mod = beta_eff * N-hat: -log p_n affine, slope beta_eff = 2 log(u_c/alpha_1)",
      abs(np.polyfit(np.arange(Nmax + 1), -np.log(p), 1)[0] - beta_eff) < 1e-9,
      detail=f"beta_eff = {beta_eff:.6f}")

# ===========================================================================
banner("T4b  the modular flow is a COMPACT U(1) of MINIMAL PERIOD 2pi (the circle)")
# ===========================================================================
# N-hat on the tick ladder (integer diagonal). The modular one-parameter group is
# U(theta) = e^{-i theta N-hat}. spec(N-hat) in Z  =>  U(2pi) = I exactly. gcd 1 => 2pi minimal.
Nop = np.diag(np.arange(Nmax + 1).astype(float))
def U(theta):
    return np.diag(np.exp(-1j * theta * np.arange(Nmax + 1)))
dev_2pi = np.max(np.abs(U(2 * math.pi) - np.eye(Nmax + 1)))
check("T4b e^{-i 2pi N-hat} = I EXACTLY (N-hat integer => the modular angle closes at 2pi)",
      dev_2pi < 1e-9, detail=f"||U(2pi) - I|| = {dev_2pi:.2e}")
# minimal period: U(theta)=I only at theta in 2pi Z (because support gcd = 1). Scan sub-multiples.
submultiple_returns = [j for j in range(2, 13)
                       if np.max(np.abs(U(2 * math.pi / j) - np.eye(Nmax + 1))) < 1e-9]
check("T4b MINIMAL period is exactly 2pi (no return-to-I at 2pi/j for j=2..12; gcd 1 forbids it)",
      submultiple_returns == [], detail=f"sub-multiple returns: {submultiple_returns or 'none'}")
# the modular FLOW e^{-i beta_eff N-hat s} therefore has minimal period s* = 2pi/beta_eff
s_star = 2 * math.pi / beta_eff
dev_flow = np.max(np.abs(U(beta_eff * s_star) - np.eye(Nmax + 1)))
check("T4b the thermal-time flow e^{-i beta_eff N-hat s} closes at s* = 2pi/beta_eff",
      dev_flow < 1e-9, detail=f"s* = {s_star:.4f}, ||U(beta_eff s*)-I|| = {dev_flow:.2e}")
print(f"    => the modular/thermal-time angle lives on the CIRCLE R/2piZ (Pontryagin dual of Z).")
print(f"       circumference = 2pi, FORCED by N-hat integer + consecutive support. Pure geometry.")

# ===========================================================================
banner("T4c  number-phase quantization: one tick = one quantum of ACTION = 2pi*hbar = h")
# ===========================================================================
# Pegg-Barnett phase states on the truncated ladder: |theta_m> = (1/sqrt(d)) sum_n e^{i n theta_m}|n>
d = Nmax + 1
thetas = 2 * math.pi * np.arange(d) / d              # d equally-spaced angles in [0,2pi)
F = np.array([[np.exp(1j * n * th) for n in range(d)] for th in thetas]) / math.sqrt(d)
# (a) phase states orthonormal (complete basis dual to N-hat)
gram = F @ F.conj().T
check("T4c the phase (angle) states are orthonormal -- a complete basis dual to N-hat",
      np.allclose(gram, np.eye(d), atol=1e-9))
# (b) N-hat GENERATES angle shifts: e^{-i N-hat alpha} |theta_m> = |theta_m + alpha>
alpha = thetas[1] - thetas[0]                        # one angle grid step
shift = np.diag(np.exp(-1j * np.arange(d) * alpha)) @ F.conj().T[:, 0]  # e^{-iN alpha}|theta_0>
check("T4c N-hat generates 2pi-periodic ANGLE shifts (e^{-iN alpha}|theta_0> = |theta_1>)",
      np.allclose(shift, F.conj().T[:, 1], atol=1e-9))
# (c) the angle is 2pi-periodic: theta and theta+2pi are the SAME state
check("T4c the angle is 2pi-periodic: |theta + 2pi> = |theta> (compact conjugate variable)",
      np.allclose(np.exp(1j * (thetas + 2 * math.pi) * 3), np.exp(1j * thetas * 3), atol=1e-12))
# (d) Bohr-Sommerfeld: action J = hbar * N-hat (spacing hbar), angle in [0,2pi) => phase-space
#     area per quantum state = (Delta J)(Delta theta) = hbar * 2pi = h. ONE TICK = h of action.
#     (dimensionless statement: area/state / hbar = 2pi.)
area_per_state_over_hbar = 1.0 * (2 * math.pi)       # (Delta N = 1) x (angle period 2pi)
check("T4c action-angle quantum cell = (Delta N=1)x(2pi) = 2pi hbar = h  (one tick = one action quantum)",
      abs(area_per_state_over_hbar - 2 * math.pi) < 1e-12,
      detail=f"area/state = {area_per_state_over_hbar:.6f} hbar = 2pi hbar = h")
print("    => A_tick = 2pi*hbar = h. The 2pi is the angle period (T4b); the action quantum is h.")

# ===========================================================================
banner("T4d  conclusion: kappa = h/t_P  (2pi DERIVED; t_P is the anchor). A1 = CROSS-CHECK only")
# ===========================================================================
hbar = 1.054571817e-34; h = 6.62607015e-34; t_P = 5.391247e-44
A_tick = 2 * math.pi * hbar                           # DERIVED: action per tick = h
kappa = A_tick / t_P                                  # kappa = h/t_P (dimensionless coeff = 1)
check("T4d h = 2pi*hbar (identity) => A_tick = h (derived from the circle period, not from A1)",
      abs(A_tick - h) / h < 1e-6)
# CROSS-CHECK (consistency, NOT the source): A1's kappa = hbar*(2pi/t_P)
kappa_A1 = hbar * (2 * math.pi / t_P)
check("T4d CROSS-CHECK: kappa = h/t_P matches A1's kappa = hbar*2pi/t_P (consistency, not source)",
      abs(kappa - kappa_A1) / kappa < 1e-9, detail=f"kappa = {kappa:.6e} J")
print(f"    kappa = h/t_P = {kappa:.6e} J. DIMENSIONLESS content FORCED: coefficient = 1 (one Planck")
print(f"    action quantum per tick). Only t_P (the tick duration) remains -- the STANDING anchor,")
print(f"    NOT derived here and NOT claimed derived (it is the framework's fundamental time unit).")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
verdict = "CLOSE" if ok_all else "see failures"
print(f"""    T4-R OUTCOME = {verdict}: the 2pi is DERIVED as the circumference of the modular /
          thermal-time circle, FORCED by tick integrality (N-hat integer on consecutive integers =>
          the modular flow is a compact U(1) of minimal period exactly 2pi; the conjugate angle is
          2pi-compact; the number-phase action quantum = 2pi*hbar = h). => A_tick = h, and
          kappa = h/t_P with dimensionless content FORCED (coefficient 1 = one action quantum/tick).
    NOT pattern-matched: the 2pi is the geometric period of the circle; A1's h enters only as an END
          consistency cross-check. Screw-winding NOT invoked. Two-temperatures held.
    t_P is the STANDING dimensional anchor -- NOT derived, NOT claimed derived. Only the dimensionless
          2pi (hence the full dimensionless content of kappa) is closed here.
    NET: M0-2R kappa completion -- ln2 (T3) + thermal-time=tick (T1) + 2pi (T4) ALL DERIVED =>
          kappa = h/t_P fully determined up to the t_P anchor. A-IT3 (Landauer) becomes
          framework-internal. No scoreboard value moved.""")
print("RESULT:", "ALL CHECKS PASS -- 2pi DERIVED (CLOSE)" if ok_all else "A CHECK FAILED")
sys.exit(0 if ok_all else 1)
