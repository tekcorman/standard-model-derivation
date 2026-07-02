#!/usr/bin/env python3
"""
Refined Dirac-cone diagnostic — confirm or reject the apparent linear band
splitting at Gamma, H, P found by lorentz_sig_dirac_cone_scan.py.

The first scan reported log-log slope p_fit ~ 0.995 with pure-linear residual
~7% over eps in [1e-5, 1e-2]. That is consistent either with a true Dirac
cone (slope = 1 exactly, residual from O(eps^2) sub-leading terms inside the
fit window) or with a near-but-not-exactly-linear dispersion (e.g. spread
~ eps * log(eps) or eps^(1+small)).

To decide, we compute lim_{eps -> 0} spread(eps)/eps along several specific
directions. If the limit exists and is direction-dependent, the local
splitting is **linear with anisotropic Fermi velocities** -- a Dirac cone.
If spread/eps -> 0, the splitting is sub-linear (e.g. eps^(3/2)). If
spread/eps grows, it is super-linear (suggesting eps^p with p<1, unphysical
for a smooth Bloch operator).

Verdict criterion:
  spread(eps)/eps over eps in {1e-3, 1e-4, ..., 1e-9}, shown alongside
  spread(eps)/eps^2. If spread/eps converges to a finite non-zero value
  while spread/eps^2 diverges, the splitting is linear.

Test sites:
  Gamma (triple degeneracy of lower 3 bands at lambda=-1)
  H     (triple degeneracy of upper 3 bands at lambda=+1)
  P     (two double-degeneracies at lambda=+/-sqrt(3))

Test directions (unit vectors in fractional reciprocal coords):
  e1 = (1, 0, 0)
  e2 = (0, 1, 0)
  e3 = (0, 0, 1)
  d_iso = (1, 1, 1)/sqrt(3)
  d_off = (1, 0.7, 0.3)/||...||  (generic interior direction)
"""

import os
import sys
import numpy as np
from numpy import linalg as la

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

from proofs.common import bloch_H, find_bonds, N_ATOMS

bonds = find_bonds()


def eigs_at(k_frac):
    H = bloch_H(k_frac, bonds)
    w = np.sort(np.real(la.eigvalsh(H)))
    return w


def spread(k_star, deg_idx, direction, eps):
    k = k_star + eps * np.asarray(direction)
    w = eigs_at(k)
    sub = w[deg_idx]
    return float(sub.max() - sub.min())


# -------------------------------------------------------------------------
# Test sites
# -------------------------------------------------------------------------

GAMMA = np.array([0.0, 0.0, 0.0])
H_PT  = np.array([-0.5, 0.5, 0.5])
P_PT  = np.array([0.25, 0.25, 0.25])

SITES = [
    {
        'name': 'Gamma (lower 3 bands)',
        'k':    GAMMA,
        'deg':  [0, 1, 2],
        'level': -1.0,
    },
    {
        'name': 'H (upper 3 bands)',
        'k':    H_PT,
        'deg':  [1, 2, 3],
        'level': +1.0,
    },
    {
        'name': 'P (lower 2 bands)',
        'k':    P_PT,
        'deg':  [0, 1],
        'level': -np.sqrt(3),
    },
    {
        'name': 'P (upper 2 bands)',
        'k':    P_PT,
        'deg':  [2, 3],
        'level': +np.sqrt(3),
    },
]


def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / la.norm(v)


DIRS = [
    ('e1   = (1,0,0)',           normalize([1, 0, 0])),
    ('e2   = (0,1,0)',           normalize([0, 1, 0])),
    ('e3   = (0,0,1)',           normalize([0, 0, 1])),
    ('111  = (1,1,1)/sqrt(3)',   normalize([1, 1, 1])),
    ('11-1 = (1,1,-1)/sqrt(3)',  normalize([1, 1, -1])),
    ('110  = (1,1,0)/sqrt(2)',   normalize([1, 1, 0])),
    ('off  = (1,0.7,0.3)/||..||', normalize([1, 0.7, 0.3])),
]

EPS_GRID = [10**(-p) for p in [3, 4, 5, 6, 7, 8, 9]]


# -------------------------------------------------------------------------
# Per-site test
# -------------------------------------------------------------------------

def test_site(site):
    print(f"\n{'-'*78}")
    print(f"  Site: {site['name']}  k_star = {site['k']}  level = {site['level']:+.6f}")
    print(f"  Spectrum at site:    {eigs_at(site['k'])}")
    print(f"{'-'*78}")

    # For each direction, table eps, spread, spread/eps, spread/eps^2
    headers = (f"{'eps':>10s} | "
               + " | ".join(f"{n:>22s}" for n, _ in DIRS))
    print()
    print("  spread(eps)/eps  (should converge to direction-dependent v_F if Dirac)")
    print("  " + headers)
    for eps in EPS_GRID:
        row = f"  {eps:10.1e} | "
        for _, d in DIRS:
            s = spread(site['k'], site['deg'], d, eps)
            row += f"{s/eps:22.10e} | "
        print(row[:-3])

    print()
    print("  spread(eps)/eps^2  (should diverge if Dirac, converge if quadratic)")
    print("  " + headers)
    for eps in EPS_GRID:
        row = f"  {eps:10.1e} | "
        for _, d in DIRS:
            s = spread(site['k'], site['deg'], d, eps)
            row += f"{s/eps**2:22.6e} | "
        print(row[:-3])

    # Overall verdict per direction: ratio of spread/eps at smallest two eps values
    print()
    print("  Per-direction limits (smallest two eps -> v_F estimate):")
    final_v = []
    for n, d in DIRS:
        s_small = spread(site['k'], site['deg'], d, EPS_GRID[-1])
        s_smaller = spread(site['k'], site['deg'], d, EPS_GRID[-1]/10)
        v_small = s_small / EPS_GRID[-1]
        v_smaller = s_smaller / (EPS_GRID[-1]/10)
        rel_change = abs(v_small - v_smaller) / max(abs(v_small), 1e-30)
        verdict = "LINEAR (Dirac)" if rel_change < 0.01 and v_small > 1e-6 else \
                  "sub-linear" if v_smaller < 0.5 * v_small else \
                  "super-linear" if v_smaller > 2 * v_small else "approx-linear"
        print(f"    {n:30s}  v_F ~ {v_small:.6e}   rel_change = {rel_change:.2e}   "
              f"{verdict}")
        final_v.append((n, v_small, verdict))

    return final_v


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    print("=" * 78)
    print(" Refined Dirac-cone diagnostic at Gamma / H / P")
    print(" (verifying or rejecting the linear band-splitting result)")
    print("=" * 78)

    site_results = {}
    for site in SITES:
        site_results[site['name']] = test_site(site)

    # Final summary
    print("\n" + "=" * 78)
    print(" Summary")
    print("=" * 78)
    any_dirac = False
    for site_name, dir_results in site_results.items():
        print(f"\n  {site_name}:")
        for n, v_F, verdict in dir_results:
            mark = "  *  " if "LINEAR" in verdict else "     "
            print(f"  {mark}{n:30s}  v_F = {v_F:.6e}   {verdict}")
            if "LINEAR" in verdict:
                any_dirac = True

    print("\n" + "=" * 78)
    if any_dirac:
        print(" Verdict: AT LEAST ONE direction with linear (Dirac) local dispersion.")
        print(" Route C-iii Step 2 PASSED on at least one site/direction.")
        print(" Next: Step 3 (Lorentzian cone reading from anisotropic v_F tensor),")
        print("       Step 4 (local-to-global signature lift -- research-level).")
    else:
        print(" Verdict: NO direction shows linear local dispersion at any test site.")
        print(" Route C-iii Step 2 FAILED -- pivot to Route C-i.")
    print("=" * 78)


if __name__ == "__main__":
    main()
