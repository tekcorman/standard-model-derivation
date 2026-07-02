#!/usr/bin/env python3
"""
theorem_beta_coefficients_derived_two_loop_check.py
====================================================

Two-loop refinement of the β-coefficient derivation theorem.

The one-loop version of the theorem (`theorem_beta_coefficients_derived.md`)
inverts the one-loop running:
    b_i = (2π / ln(M_unif/M_Z)) × (1/α_i(M_Z) − 1/α_i(M_unif))
and finds (b_1, b_2, b_3) = (6.66, 1.06, −2.95), with the b_2 deviation at
+6.22% as the dominant gap.

This script tests whether the gap closes when two-loop running is used
forward (the "honest" inverse extraction at two-loop is operational —
holding b_ij fixed at MSSM values and asking what b_i makes the running
hit PDG endpoints).

Methodology — for each (M_SUSY, loop-order) configuration:
  (a) Run forward from M_unif with MSSM b_i = (33/5, 1, −3) and the
      corresponding b_ij (MSSM above M_SUSY, SM below).  Get
      1/α_i^pred(M_Z).
  (b) Compute residual gap δ_i = 1/α_i^PDG − 1/α_i^pred(M_Z).
  (c) The "effective b_i correction" (treating δ_i as a leading-order
      pull on the slope from M_unif to M_Z) is
          Δb_i ≈ δ_i × (2π / ln(M_unif/M_Z))
      so the inverse-extracted b_i is b_i^MSSM + Δb_i.
  (d) Compare to MSSM b_i and report the deviation.

The output answers: does the two-loop refinement tighten the b_i
extraction relative to the one-loop algebra?

CONVENTION: x_i = 1/α_i (GUT-normalized for i=1), t = ln(µ).
    dx_i/dt = −b_i/(2π) − Σ_j b_ij/(8π²) × (1/x_j)
"""

from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Framework upstream (theorem-grade) + PDG endpoints
# ---------------------------------------------------------------------------

ALPHA_GUT_INV = 24                          # 1/α_GUT at M_unif (theorem-grade)
M_UNIF_GeV = 1.985e16                       # theorem-grade-conditional
M_Z_GeV = 91.1876                           # external

# PDG 2024 at M_Z
ALPHA_EM_INV_MZ = 127.94
SIN_SQ_THETA_W_MZ = 0.23121
ALPHA_S_MZ = 0.1180

ALPHA_1_INV_MZ = (3.0 / 5.0) * ALPHA_EM_INV_MZ * (1 - SIN_SQ_THETA_W_MZ)
ALPHA_2_INV_MZ = ALPHA_EM_INV_MZ * SIN_SQ_THETA_W_MZ
ALPHA_3_INV_MZ = 1.0 / ALPHA_S_MZ

ALPHA_INV_PDG = np.array([ALPHA_1_INV_MZ, ALPHA_2_INV_MZ, ALPHA_3_INV_MZ])

LN_RATIO = math.log(M_UNIF_GeV / M_Z_GeV)   # ≈ 33.01

# MSSM reference values
MSSM_B_FRAC = (Fraction(33, 5), Fraction(1), Fraction(-3))
B_MSSM = np.array([33.0 / 5.0, 1.0, -3.0])

# MSSM two-loop matrix b_ij (Martin SUSY primer §6.5)
B_IJ_MSSM = np.array([
    [199.0 / 25.0, 27.0 / 5.0, 88.0 / 5.0],
    [9.0 / 5.0,    25.0,       24.0      ],
    [11.0 / 5.0,   9.0,        14.0      ],
])

# SM reference + b_ij (Machacek-Vaughn; gauge part only)
B_SM = np.array([41.0 / 10.0, -19.0 / 6.0, -7.0])
B_IJ_SM = np.array([
    [199.0 / 50.0, 27.0 / 10.0, 44.0 / 5.0],
    [9.0 / 10.0,   35.0 / 6.0,  12.0      ],
    [11.0 / 10.0,  9.0 / 2.0,  -26.0      ],
])


# ---------------------------------------------------------------------------
# Running primitive
# ---------------------------------------------------------------------------

def rge_rhs(t, x, b, b_ij, two_loop):
    x = np.asarray(x, dtype=float)
    dxdt = -b / (2.0 * math.pi)
    if two_loop:
        alpha = 1.0 / x
        dxdt -= (b_ij @ alpha) / (8.0 * math.pi ** 2)
    return dxdt


def integrate(x_start, t_start, t_end, b, b_ij, two_loop):
    if t_start == t_end:
        return np.asarray(x_start, dtype=float)
    sol = solve_ivp(
        lambda t, x: rge_rhs(t, x, b, b_ij, two_loop),
        (t_start, t_end), x_start,
        method='RK45', rtol=1e-12, atol=1e-14,
    )
    return sol.y[:, -1]


def forward_run(M_SUSY_GeV, two_loop, b_mssm=B_MSSM, b_sm=B_SM):
    """Run x = 1/α from M_unif to M_Z through M_SUSY threshold."""
    t_unif = math.log(M_UNIF_GeV)
    t_susy = math.log(M_SUSY_GeV)
    t_MZ = math.log(M_Z_GeV)

    x_unif = np.array([float(ALPHA_GUT_INV)] * 3)
    x_susy = integrate(x_unif, t_unif, t_susy, b_mssm, B_IJ_MSSM, two_loop)
    if M_SUSY_GeV == M_Z_GeV:
        return x_susy
    x_MZ = integrate(x_susy, t_susy, t_MZ, b_sm, B_IJ_SM, two_loop)
    return x_MZ


# ---------------------------------------------------------------------------
# Inverse extraction of effective b_i
# ---------------------------------------------------------------------------

def extract_effective_b(M_SUSY_GeV, two_loop):
    """Forward 2-loop run with MSSM b_i; back out 'effective b_i' from PDG gap.

    Forward gives x^pred at M_Z.  Residual gap δ = x^PDG − x^pred.
    Treating δ as a slope correction over ln(M_unif/M_Z):
        Δb_i ≈ δ × 2π / ln(M_unif/M_Z)
    so the effective extracted b_i = b_i^MSSM + Δb_i.
    """
    x_pred = forward_run(M_SUSY_GeV, two_loop)
    delta = ALPHA_INV_PDG - x_pred
    Delta_b = delta * 2.0 * math.pi / LN_RATIO
    b_eff = B_MSSM + Delta_b
    return {
        'x_pred': x_pred,
        'x_pdg': ALPHA_INV_PDG,
        'delta_x': delta,
        'Delta_b': Delta_b,
        'b_eff': b_eff,
    }


def pure_one_loop_inverse():
    """The current theorem's one-line algebra."""
    b = (ALPHA_INV_PDG - ALPHA_GUT_INV) * 2.0 * math.pi / LN_RATIO
    return b


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_pct(num, ref):
    if ref == 0:
        return '   ∞'
    return f'{100 * (num - ref) / abs(ref):+7.2f}%'


def main():
    print('=' * 88)
    print(' theorem_beta_coefficients_derived_two_loop_check.py')
    print('=' * 88)
    print()
    print(f'  M_unif = {M_UNIF_GeV:.3e} GeV;  M_Z = {M_Z_GeV} GeV;')
    print(f'  ln(M_unif/M_Z) = {LN_RATIO:.4f};  2π/ln(M_unif/M_Z) = {2*math.pi/LN_RATIO:.5f}')
    print(f'  1/α_GUT at M_unif = {ALPHA_GUT_INV}  (theorem-grade upstream)')
    print(f'  PDG endpoints: 1/α_1 = {ALPHA_1_INV_MZ:.3f}, 1/α_2 = {ALPHA_2_INV_MZ:.3f}, '
          f'1/α_3 = {ALPHA_3_INV_MZ:.3f}')
    print()
    print(f'  MSSM b_i reference: 33/5 = {33/5}, 1, −3')
    print()

    # --- (1) Pure one-loop algebra (existing theorem) ---
    print('-' * 88)
    print(' (1) PURE ONE-LOOP ALGEBRA — what the current theorem computes')
    print('-' * 88)
    b_one_loop = pure_one_loop_inverse()
    print(f'  b_i_extracted = (2π/ln) × (1/α_i^PDG − 24):')
    for i in range(3):
        print(f'    b_{i+1} = {b_one_loop[i]:+8.4f}   (MSSM {float(MSSM_B_FRAC[i]):+8.4f})   '
              f'dev {fmt_pct(b_one_loop[i], float(MSSM_B_FRAC[i]))}')
    max_dev_1loop = max(abs(100 * (b_one_loop[i] - float(MSSM_B_FRAC[i])) /
                            abs(float(MSSM_B_FRAC[i])))
                        for i in range(3))
    print(f'  Max deviation: {max_dev_1loop:.2f}%')
    print()

    # --- (2) Two-loop forward with M_SUSY=M_Z (pure MSSM) ---
    print('-' * 88)
    print(' (2) TWO-LOOP, M_SUSY = M_Z (pure MSSM all the way, no threshold)')
    print('     This isolates the pure two-loop b_ij correction.')
    print('-' * 88)
    r = extract_effective_b(M_Z_GeV, two_loop=True)
    print(f'  Forward 2-loop run with MSSM b_i predicts at M_Z:')
    print(f'    1/α_1^pred = {r["x_pred"][0]:7.4f}  vs PDG {r["x_pdg"][0]:7.4f}  '
          f'(Δ = {r["delta_x"][0]:+.4f})')
    print(f'    1/α_2^pred = {r["x_pred"][1]:7.4f}  vs PDG {r["x_pdg"][1]:7.4f}  '
          f'(Δ = {r["delta_x"][1]:+.4f})')
    print(f'    1/α_3^pred = {r["x_pred"][2]:7.4f}  vs PDG {r["x_pdg"][2]:7.4f}  '
          f'(Δ = {r["delta_x"][2]:+.4f})')
    print()
    print(f'  Effective b_i extracted (MSSM b_i + δ × 2π/ln correction):')
    for i in range(3):
        b_eff = r['b_eff'][i]
        b_mssm = float(MSSM_B_FRAC[i])
        print(f'    b_{i+1}^eff = {b_eff:+8.4f}   (MSSM {b_mssm:+8.4f})   dev {fmt_pct(b_eff, b_mssm)}')
    max_dev_2loop_pure = max(abs(100 * (r['b_eff'][i] - float(MSSM_B_FRAC[i])) /
                                 abs(float(MSSM_B_FRAC[i])))
                             for i in range(3))
    print(f'  Max deviation: {max_dev_2loop_pure:.2f}%')
    print()

    # --- (3) Two-loop scan over M_SUSY ---
    print('-' * 88)
    print(' (3) M_SUSY SCAN, TWO-LOOP — physically realistic with SM below threshold')
    print('-' * 88)
    print(f'  {"M_SUSY (GeV)":>12}  {"b_1^eff":>10}  {"dev":>9}  {"b_2^eff":>10}  '
          f'{"dev":>9}  {"b_3^eff":>10}  {"dev":>9}  {"max":>8}')
    M_SUSY_grid = [91.1876, 120.0, 150.0, 180.0, 200.0, 250.0, 300.0, 500.0,
                   1000.0, 2000.0, 5000.0, 10000.0]
    best_max_dev = float('inf')
    best_M_SUSY = None
    best_b_eff = None
    for M_SUSY in M_SUSY_grid:
        r = extract_effective_b(M_SUSY, two_loop=True)
        b_eff = r['b_eff']
        devs = [100 * (b_eff[i] - float(MSSM_B_FRAC[i])) / abs(float(MSSM_B_FRAC[i]))
                for i in range(3)]
        max_dev = max(abs(d) for d in devs)
        if max_dev < best_max_dev:
            best_max_dev = max_dev
            best_M_SUSY = M_SUSY
            best_b_eff = b_eff
        print(f'  {M_SUSY:>12g}  {b_eff[0]:>+10.4f}  {devs[0]:>+8.2f}%  '
              f'{b_eff[1]:>+10.4f}  {devs[1]:>+8.2f}%  '
              f'{b_eff[2]:>+10.4f}  {devs[2]:>+8.2f}%  {max_dev:>7.2f}%')
    print()
    print(f'  Best M_SUSY (smallest max-deviation): {best_M_SUSY} GeV → max dev {best_max_dev:.2f}%')
    print(f'  Effective b_i at best: ({best_b_eff[0]:+.4f}, {best_b_eff[1]:+.4f}, {best_b_eff[2]:+.4f})')
    print()

    # --- (4) Net headline ---
    print('=' * 88)
    print(' NET — does two-loop refinement tighten the extraction?')
    print('=' * 88)
    print()
    print(f'  1-loop pure algebra        max b_i deviation: {max_dev_1loop:6.2f}%')
    print(f'  2-loop pure MSSM (no thr): max b_i deviation: {max_dev_2loop_pure:6.2f}%')
    print(f'  2-loop best M_SUSY ({best_M_SUSY:g} GeV):  max b_i deviation: {best_max_dev:6.2f}%')
    print()
    if best_max_dev < max_dev_1loop:
        ratio = max_dev_1loop / best_max_dev
        print(f'  TIGHTENING: two-loop + threshold reduces max deviation by '
              f'×{ratio:.1f} (from {max_dev_1loop:.2f}% to {best_max_dev:.2f}%).')
    else:
        print(f'  NO TIGHTENING: two-loop refinement does not reduce the maximum')
        print(f'                 b_i deviation below the one-loop pure-algebra value.')
        print(f'                 The 1-loop figure {max_dev_1loop:.2f}% is the honest')
        print(f'                 envelope (M_SUSY threshold variation widens it).')
    print()
    print('=' * 88)
    print()
    print('theorem_beta_coefficients_derived_two_loop_check.py: sentinel done.')


if __name__ == '__main__':
    main()
