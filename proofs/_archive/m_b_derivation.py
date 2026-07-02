#!/usr/bin/env python3
"""
---
derives: m_b
inputs:
  - m_tau
  - m_t
  - alpha_GUT
  - sin2_theta_W
  - v
script_version: 2.0.0
doc: standard-model-derivation/docs/parameters/derivations.md
doc_section: '§9.6 quark Koide + PS Fock'
doc_version_required: 0.0.1
mechanism: structural
rigor_status: rigor_route_specified
---

m_b from Georgi-Jarlskog bottom-tau unification + MSSM/SM Yukawa RG running.

Uses ONLY framework quantities — no external anchors. Gauge couplings at M_Z
are obtained by top-down MSSM RG from alpha_GUT = 1/24.1, with M_Z
self-consistently determined from v and the resulting g_2.

Method:
  1. Run alpha_GUT top-down to get gauge couplings at M_Z.
  2. Fix tan(beta) from y_t(GUT) = 1 using m_t (anchor).
  3. Find m_b such that y_b(GUT)/y_tau(GUT) = GJ = 3.

NOTE: Framework alpha_s(M_Z) ~ 0.155 (vs observed 0.118). This affects the
Yukawa RG running and the m_b prediction. The honest framework prediction is
reported, including whatever error this introduces.
"""

import os
import sys
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gauge'))
from _mssm_rge import run_down_from_gut  # noqa: E402

GJ = 3
M_GUT = 2.0e16
M_SUSY = 3000.0
PI = math.pi

# ---------- MSSM / SM Yukawa beta functions ----------

B_MSSM = np.array([33.0 / 5.0, 1.0, -3.0])
BIJ_MSSM = np.array([
    [199.0 / 25.0, 27.0 / 5.0, 88.0 / 5.0],
    [9.0 / 5.0,    25.0,        24.0],
    [11.0 / 5.0,   9.0,         14.0],
])
B_SM = np.array([41.0 / 10.0, -19.0 / 6.0, -7.0])
BIJ_SM = np.array([
    [199.0 / 50.0, 27.0 / 10.0, 44.0 / 5.0],
    [9.0 / 10.0,   35.0 / 6.0,  12.0],
    [11.0 / 10.0,  9.0 / 2.0,  -26.0],
])


def _mssm_rge(t, y):
    a1i, a2i, a3i, yt, yb, ytau = y
    a = np.array([1.0 / a1i, 1.0 / a2i, 1.0 / a3i])
    g1s, g2s, g3s = 4 * PI * a[0], 4 * PI * a[1], 4 * PI * a[2]
    da = np.zeros(3)
    for i in range(3):
        da[i] = -B_MSSM[i] / (2 * PI)
        for j in range(3):
            da[i] -= BIJ_MSSM[i, j] / (8 * PI**2) * a[j]
    yt2, yb2, ytau2 = yt**2, yb**2, ytau**2
    return [da[0], da[1], da[2],
            yt / (16 * PI**2) * (6 * yt2 + yb2 - (16 / 3) * g3s - 3 * g2s - (13 / 15) * g1s),
            yb / (16 * PI**2) * (6 * yb2 + yt2 + ytau2 - (16 / 3) * g3s - 3 * g2s - (7 / 15) * g1s),
            ytau / (16 * PI**2) * (4 * ytau2 + 3 * yb2 - 3 * g2s - (9 / 5) * g1s)]


def _sm_rge(t, y):
    a1i, a2i, a3i, yt, yb, ytau = y
    a = np.array([1.0 / a1i, 1.0 / a2i, 1.0 / a3i])
    g1s, g2s, g3s = 4 * PI * a[0], 4 * PI * a[1], 4 * PI * a[2]
    da = np.zeros(3)
    for i in range(3):
        da[i] = -B_SM[i] / (2 * PI)
        for j in range(3):
            da[i] -= BIJ_SM[i, j] / (8 * PI**2) * a[j]
    yt2, yb2, ytau2 = yt**2, yb**2, ytau**2
    return [da[0], da[1], da[2],
            yt / (16 * PI**2) * ((9 / 2) * yt2 + (3 / 2) * yb2 - 8 * g3s - (9 / 4) * g2s - (17 / 12) * g1s),
            yb / (16 * PI**2) * ((9 / 2) * yb2 + (3 / 2) * yt2 + ytau2 - 8 * g3s - (9 / 4) * g2s - (5 / 12) * g1s),
            ytau / (16 * PI**2) * ((5 / 2) * ytau2 + 3 * yb2 - (9 / 4) * g2s - (15 / 4) * g1s)]


def _run_mz_to_gut(alpha_inv_mz, yt_mz, yb_mz, ytau_mz,
                   log_mz, log_msusy, log_mgut):
    y0 = list(alpha_inv_mz) + [yt_mz, yb_mz, ytau_mz]
    s1 = solve_ivp(_sm_rge, [log_mz, log_msusy], y0,
                   method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
    s2 = solve_ivp(_mssm_rge, [log_msusy, log_mgut], list(s1.sol(log_msusy)),
                   method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
    return s2.sol(log_mgut)


def derive(m_tau: float, m_t: float, alpha_GUT: float,
           sin2_theta_W: float, v: float) -> dict:

    alpha_GUT_inv = 1.0 / alpha_GUT
    cos_tw = math.sqrt(1.0 - sin2_theta_W)

    # Step 0: self-consistent M_Z from top-down gauge running
    m_z = 91.0
    for _ in range(20):
        y_gauge = run_down_from_gut(alpha_GUT_inv, M_GUT, m_z, M_SUSY)
        a2_inv = y_gauge[1]
        g_2 = math.sqrt(4.0 * math.pi / a2_inv)
        m_z_new = v * g_2 / (2.0 * cos_tw)
        if abs(m_z_new - m_z) < 1e-6:
            break
        m_z = m_z_new

    # Framework gauge couplings at M_Z
    alpha_inv_mz = np.array([y_gauge[0], y_gauge[1], y_gauge[2]])
    alpha_s_mz = 1.0 / y_gauge[2]

    log_mz = math.log(m_z)
    log_msusy = math.log(M_SUSY)
    log_mgut = math.log(M_GUT)
    v2 = v / math.sqrt(2.0)
    qcd_corr_t = 1.0 + 4.0 * alpha_s_mz / (3.0 * PI)

    # Step 1: find tan(beta) from y_t(GUT) = 1 (seed m_b = 4.18)
    m_b_seed = 4.18

    def yt_gut_residual(tan_beta):
        sb = tan_beta / math.sqrt(1.0 + tan_beta**2)
        cb = 1.0 / math.sqrt(1.0 + tan_beta**2)
        at = _run_mz_to_gut(alpha_inv_mz,
                            m_t / (qcd_corr_t * v2 * sb),
                            m_b_seed / (v2 * cb),
                            m_tau / (v2 * cb),
                            log_mz, log_msusy, log_mgut)
        return at[3] - 1.0

    tan_beta = brentq(yt_gut_residual, 30.0, 55.0, xtol=1e-4)

    # Step 2: find m_b from GJ = 3
    sb = tan_beta / math.sqrt(1.0 + tan_beta**2)
    cb = 1.0 / math.sqrt(1.0 + tan_beta**2)
    yt_mz = m_t / (qcd_corr_t * v2 * sb)
    ytau_mz = m_tau / (v2 * cb)

    def gj_residual(m_b_trial):
        yb_mz = m_b_trial / (v2 * cb)
        at = _run_mz_to_gut(alpha_inv_mz, yt_mz, yb_mz, ytau_mz,
                            log_mz, log_msusy, log_mgut)
        return at[4] / at[5] - GJ

    m_b = brentq(gj_residual, 1.0, 10.0, xtol=1e-8)

    # Verify
    yb_mz_final = m_b / (v2 * cb)
    at_gut = _run_mz_to_gut(alpha_inv_mz, yt_mz, yb_mz_final, ytau_mz,
                            log_mz, log_msusy, log_mgut)

    return {
        'predicted': m_b,
        'checks': {
            'tan_beta': tan_beta,
            'yt_gut': float(at_gut[3]),
            'ratio_yb_ytau_gut': float(at_gut[4] / at_gut[5]),
            'alpha_s_mz_framework': alpha_s_mz,
            'M_Z_self_consistent': m_z,
            'm_tau_input': m_tau,
            'm_t_input': m_t,
        },
    }


def main():
    # Framework inputs only
    import sys
    try:
        result = derive(
            m_tau=1.774905586885468,  # from m_tau_derivation.py
            m_t=172.71,              # anchor (Stage 2 target)
            alpha_GUT=1.0 / 24.1,
            sin2_theta_W=3.0 / 13.0,
            v=245.64,
        )
    except Exception as e:
        # Framework gauge couplings (alpha_s ~ 0.155) make the Yukawa RGE
        # inconsistent: y_t(GUT) reaches only ~0.4 (not 1.0) and y_b/y_tau
        # reaches only ~1.2 (not 3.0). The alpha_s gap (theory_open_items A.9)
        # must be closed before m_b can be self-consistently derived.
        print(f"# BLOCKED name=m_b reason=alpha_s_gap")
        print()
        print("m_b derivation BLOCKED: framework alpha_s(M_Z) = 0.155 (obs: 0.118)")
        print("  The 31% alpha_s gap makes GJ=3 and y_t(GUT)=1 unreachable.")
        print("  Closing theory_open_items A.9 (alpha_s from framework) unblocks this.")
        print(f"  Technical: {e}")
        sys.exit(1)

    chk = result['checks']
    print(f"# PREDICT name=m_b value={result['predicted']:.15f}")
    print()
    print("m_b from GJ=3 + y_t(GUT)=1 with framework gauge couplings (top-down)")
    print(f"  tan(beta)           = {chk['tan_beta']:.4f}")
    print(f"  y_t(GUT)            = {chk['yt_gut']:.6f}  (target: 1.0)")
    print(f"  y_b/y_tau(GUT)      = {chk['ratio_yb_ytau_gut']:.6f}  (target: {GJ})")
    print(f"  alpha_s(M_Z) fw     = {chk['alpha_s_mz_framework']:.4f}  (obs: 0.1180)")
    print(f"  M_Z (self-consist)  = {chk['M_Z_self_consistent']:.4f} GeV  (obs: 91.19)")
    print(f"  m_b                 = {result['predicted']:.15f}  GeV  (obs: 4.18)")


if __name__ == '__main__':
    main()
