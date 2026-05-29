#!/usr/bin/env python3
"""
Lorentzian-signature Route C-iii diagnostic — Dirac-cone scan of the srs
4-band scalar Bloch Hamiltonian H(k) over the BCC reduced Brillouin zone.

Purpose
-------
The handoff an internal note proposes Route
C-iii: derive Lorentzian signature from a Dirac-cone (linear band-touching)
of the substrate dispersion. This script executes its minimum-effort first
move: scan the BZ for any k_* where two or more bands touch with linear
local dispersion.

Two known features of H(k) on srs (4-atom primitive cell, NN bond list from
proofs/common.find_bonds()):
  - At Gamma: H(0) is the K_4 adjacency matrix; spectrum {3, -1, -1, -1}.
    The TOP band's local dispersion is quadratic: lambda_0(q) = 3 - |q|^2/16
    (predictions/srs_bloch_dispersion_gamma.py). The LOWER 3 bands have a
    triple degeneracy at lambda = -1 at Gamma -- candidate band-touching.
  - At P = (1/4,1/4,1/4)_frac: Hashimoto B(P) spectrum {+/-h, +/-h*, +/-1}
    with h = (sqrt 3 + i sqrt 5)/2 (predictions/B_P_doubly_degenerate_h.py).
    Scalar H(P) spectrum is computed below; Ramanujan saturation of the
    Hashimoto does not directly give Hermitian band-touching.

The scan tests Gamma (lower 3 bands), P (any band pair), and high-symmetry
lines Gamma-H, Gamma-P, Gamma-N, H-P, N-P, N-H. At each near-degeneracy,
a local dispersion fit decides linear (Dirac) vs quadratic (parabolic).

Decisive output
---------------
For each candidate band-touching at some k_*:
  - residual_linear  = ||delta_E - v*|delta_k|      || / ||delta_E||
  - residual_quadratic = ||delta_E - alpha*|delta_k|^2|| / ||delta_E||
The verdict at k_* is `linear` if residual_linear is much smaller, `quadratic`
if the converse holds, `mixed` otherwise.

If no candidate has residual_linear small AND residual_quadratic large, then
Route C-iii is structurally unavailable on srs's scalar Bloch dispersion, and
the handoff's recommended first move terminates as a clean negative finding.
"""

import os
import sys
import numpy as np
from numpy import linalg as la

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

from proofs.common import bloch_H, find_bonds, N_ATOMS

# =============================================================================
# Conventions
#
# We use the proofs/common.bloch_H convention: phase = exp(2*pi*i*k_frac . cell)
# where cell = (n1, n2, n3) is the integer offset on the BCC primitive lattice.
# Therefore k_frac = (k1, k2, k3) lives in the primitive reciprocal basis with
# 2*pi already absorbed into the phase. The fundamental BZ unit cell is the
# unit cube in k_frac.
#
# High-symmetry points of the bcc reduced BZ (truncated octahedron) in this
# fractional convention:
#   Gamma = (0, 0, 0)
#   H     = (-1/2, 1/2, 1/2)   <- center of square face along Cartesian (2pi,0,0)
#   P     = ( 1/4, 1/4, 1/4)   <- triple point at Cartesian (pi, pi, pi)
#   N     = ( 0,   0,   1/2)   <- center of hex-face edge at Cartesian (pi, pi, 0)
# Lines tested:
#   Gamma-H, Gamma-P, Gamma-N, H-P, N-P, N-H, plus a generic interior probe.
# =============================================================================

GAMMA = np.array([0.0, 0.0, 0.0])
H_PT  = np.array([-0.5, 0.5, 0.5])
P_PT  = np.array([0.25, 0.25, 0.25])
N_PT  = np.array([0.0, 0.0, 0.5])

LINES = [
    ('Gamma-H', GAMMA, H_PT),
    ('Gamma-P', GAMMA, P_PT),
    ('Gamma-N', GAMMA, N_PT),
    ('H-P',     H_PT,  P_PT),
    ('N-P',     N_PT,  P_PT),
    ('N-H',     N_PT,  H_PT),
]

NSAMPLES = 401   # samples per line
GAP_TOL  = 1e-3  # threshold for "near-degeneracy"

bonds = find_bonds()


def eigs_at(k_frac):
    """Return the 4 sorted real eigenvalues of the 4x4 scalar Bloch H."""
    H = bloch_H(k_frac, bonds)
    # H must be Hermitian for the scalar adjacency-Bloch case.
    assert np.max(np.abs(H - H.conj().T)) < 1e-10, "H not Hermitian"
    w = np.sort(np.real(la.eigvalsh(H)))
    return w


def all_pair_gaps(w):
    """Return the gaps |w[i+1]-w[i]| for i = 0..2."""
    return np.array([w[i+1] - w[i] for i in range(N_ATOMS - 1)])


# -----------------------------------------------------------------------------
# Step 1: Sample all lines and record (k, eigenvalues, min_gap, gap_pair_idx).
# -----------------------------------------------------------------------------

def scan_line(name, k0, k1, n=NSAMPLES):
    ts = np.linspace(0.0, 1.0, n)
    ks = [k0 + t*(k1 - k0) for t in ts]
    spectra = np.array([eigs_at(k) for k in ks])
    gaps = np.array([all_pair_gaps(w) for w in spectra])
    min_gap_per_t = gaps.min(axis=1)
    return ts, np.array(ks), spectra, gaps, min_gap_per_t


# -----------------------------------------------------------------------------
# Step 2: Local dispersion fit at a candidate k_*.
#
# Sample radius eps around k_* in `n_dirs` random directions. Track the
# smallest (band_i, band_{i+1}) gap as a function of |delta_k|. Fit the
# spread of the band cluster to:
#     spread(eps) = c1 * eps + c2 * eps^2 + ...
# Linear-vs-quadratic verdict: compare residuals of linear-only and
# quadratic-only models on a log-log fit.
# -----------------------------------------------------------------------------

def local_dispersion_fit(k_star, deg_idx, n_dirs=12, n_eps=20, eps_min=1e-5,
                         eps_max=1e-2, seed=0):
    """At k_star with a band-cluster degeneracy at index range deg_idx
    (a list of band indices that are degenerate), fit the local spread.

    Returns dict with linear-fit residual, quadratic-fit residual, and verdict.
    """
    rng = np.random.default_rng(seed)
    eps_grid = np.geomspace(eps_min, eps_max, n_eps)

    # spread[e] = max - min over the deg_idx eigenvalues, averaged over dirs
    spreads = []
    for eps in eps_grid:
        spread_dirs = []
        for _ in range(n_dirs):
            d = rng.standard_normal(3)
            d /= la.norm(d)
            k = k_star + eps * d
            w = eigs_at(k)
            sub = w[deg_idx]
            spread_dirs.append(sub.max() - sub.min())
        spreads.append(np.mean(spread_dirs))
    spreads = np.array(spreads)

    # Linear-only fit: spread = a * eps. Use log-log slope.
    # Quadratic-only fit: spread = b * eps^2.
    log_eps = np.log(eps_grid)
    log_spr = np.log(np.maximum(spreads, 1e-300))

    # Fit log_spr = log_a + p * log_eps  (where p == 1 for linear, 2 for quad)
    A = np.vstack([np.ones_like(log_eps), log_eps]).T
    coef, *_ = la.lstsq(A, log_spr, rcond=None)
    log_a_fit, p_fit = coef[0], coef[1]
    log_a = np.exp(log_a_fit)

    # Residual against pure linear model
    a_lin = np.exp(log_a_fit)  # only meaningful if p_fit ~ 1
    pred_lin = log_a * eps_grid
    pred_quad = log_a * eps_grid**2

    # Better: fit a, b in spread = a*eps + b*eps^2 directly
    M = np.vstack([eps_grid, eps_grid**2]).T
    abm, *_ = la.lstsq(M, spreads, rcond=None)
    a_full, b_full = abm[0], abm[1]
    pred_full = a_full*eps_grid + b_full*eps_grid**2

    # Residuals as relative L2
    def rel_err(pred):
        denom = la.norm(spreads)
        if denom < 1e-30:
            return 0.0
        return la.norm(spreads - pred) / denom

    # Pure-model fits
    c_lin, *_ = la.lstsq(eps_grid.reshape(-1, 1), spreads, rcond=None)
    a_pure_lin = float(c_lin[0])
    res_lin = rel_err(a_pure_lin * eps_grid)

    c_quad, *_ = la.lstsq((eps_grid**2).reshape(-1, 1), spreads, rcond=None)
    b_pure_quad = float(c_quad[0])
    res_quad = rel_err(b_pure_quad * eps_grid**2)

    # Verdict
    if res_lin < 0.05 and res_quad > 0.5:
        verdict = "linear (Dirac-cone)"
    elif res_quad < 0.05 and res_lin > 0.5:
        verdict = "quadratic (parabolic)"
    else:
        verdict = f"mixed (slope p_fit ~ {p_fit:.3f})"

    return {
        'eps_grid': eps_grid,
        'spread':   spreads,
        'p_fit':    p_fit,
        'a_pure_linear':  a_pure_lin,
        'b_pure_quadratic': b_pure_quad,
        'res_pure_linear':    res_lin,
        'res_pure_quadratic': res_quad,
        'a_combined': a_full,
        'b_combined': b_full,
        'verdict':    verdict,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 76)
    print(" Lorentzian-signature Route C-iii diagnostic")
    print(" Dirac-cone scan of srs scalar Bloch H(k) on the BCC reduced BZ")
    print("=" * 76)

    # -- 1. Anchor: spectra at the high-symmetry points
    print("\n--- Anchor spectra at high-symmetry points ---")
    for name, k in [('Gamma', GAMMA), ('H', H_PT), ('P', P_PT), ('N', N_PT)]:
        w = eigs_at(k)
        print(f"  H({name:5s}) eigenvalues = {[f'{x:+.6f}' for x in w]}")

    # -- 2. Scan the lines, find candidate band-touchings
    print("\n--- Scan along high-symmetry lines (search for near-degeneracies) ---")
    candidates = []  # (line_name, t, k_frac, gap_idx, gap_value)

    for name, k0, k1 in LINES:
        ts, ks, spectra, gaps, min_gap = scan_line(name, k0, k1)
        # Find local minima of gap profile per band-pair
        for j in range(N_ATOMS - 1):
            g_j = gaps[:, j]
            # endpoints excluded; find interior minima below tolerance
            interior_min_idx = np.argmin(g_j)
            interior_min_val = g_j[interior_min_idx]
            # Always report endpoints + minimum in scan summary
            print(f"  Line {name:8s} pair ({j},{j+1}): "
                  f"min gap = {interior_min_val:.3e} at t={ts[interior_min_idx]:.4f} "
                  f"(k_frac={ks[interior_min_idx]})")
            if interior_min_val < GAP_TOL:
                candidates.append({
                    'line':  name,
                    't':     float(ts[interior_min_idx]),
                    'k':     ks[interior_min_idx].copy(),
                    'pair':  (j, j+1),
                    'gap':   float(interior_min_val),
                })

    # -- 3. Always evaluate Gamma for the lower 3 bands (3-fold degeneracy)
    #    This is the most physically interesting candidate.
    print("\n--- Special candidate: Gamma, lower 3 bands (triple degeneracy) ---")
    w_gamma = eigs_at(GAMMA)
    print(f"  spec H(Gamma) = {w_gamma}  (lower 3 bands degenerate at lambda = -1)")
    fit_gamma = local_dispersion_fit(GAMMA, deg_idx=[0, 1, 2])
    print(f"  log-log slope of spread(eps): p_fit = {fit_gamma['p_fit']:.4f}")
    print(f"     pure linear   model: spread ~ {fit_gamma['a_pure_linear']:.4f}*eps,"
          f"   residual = {fit_gamma['res_pure_linear']:.4f}")
    print(f"     pure quadratic model: spread ~ {fit_gamma['b_pure_quadratic']:.4f}*eps^2,"
          f" residual = {fit_gamma['res_pure_quadratic']:.4f}")
    print(f"     verdict at Gamma (lower 3 bands): {fit_gamma['verdict']}")

    # -- 4. Special candidate: P, any near-degenerate pair
    print("\n--- Special candidate: P, scalar Bloch H(P) ---")
    w_P = eigs_at(P_PT)
    print(f"  spec H(P) = {w_P}")
    P_pairs = []
    for i in range(N_ATOMS - 1):
        if abs(w_P[i+1] - w_P[i]) < 0.05:
            P_pairs.append((i, i+1))
    if P_pairs:
        for pr in P_pairs:
            print(f"  near-degenerate pair {pr} at lambda ~ {w_P[pr[0]]:.6f}")
            fit_P = local_dispersion_fit(P_PT, deg_idx=list(pr))
            print(f"     verdict at P pair {pr}: {fit_P['verdict']}")
            print(f"        (p_fit={fit_P['p_fit']:.3f},"
                  f" res_lin={fit_P['res_pure_linear']:.3f},"
                  f" res_quad={fit_P['res_pure_quadratic']:.3f})")
    else:
        print("  no band pair within 0.05 at P -> no Dirac candidate at P")

    # -- 5. Local fits at every interior candidate
    print("\n--- Local dispersion fits at interior candidates (gap < 1e-3) ---")
    if not candidates:
        print("  No interior near-degeneracies found below tolerance 1e-3.")
    else:
        for cand in candidates:
            print(f"\n  Candidate: {cand['line']} t={cand['t']:.4f}"
                  f" k_frac={cand['k']} pair={cand['pair']} gap={cand['gap']:.3e}")
            fit = local_dispersion_fit(cand['k'], deg_idx=list(cand['pair']))
            print(f"     verdict: {fit['verdict']}")
            print(f"        (p_fit={fit['p_fit']:.3f},"
                  f" res_lin={fit['res_pure_linear']:.3f},"
                  f" res_quad={fit['res_pure_quadratic']:.3f})")

    # -- 6. Final verdict
    print("\n" + "=" * 76)
    print(" Summary verdict for Route C-iii")
    print("=" * 76)
    g_verdict = fit_gamma['verdict']
    p_summary = "no candidate" if not P_pairs else "see above"
    interior_summary = (
        "no interior candidates" if not candidates
        else f"{len(candidates)} interior candidates fit above"
    )
    print(f"  Gamma (lower 3 bands): {g_verdict}")
    print(f"  P:                     {p_summary}")
    print(f"  interior:              {interior_summary}")

    # Decide whether ANY linear (Dirac) verdict was found
    linear_found = False
    if 'linear' in g_verdict and 'mixed' not in g_verdict:
        linear_found = True
    for cand in candidates:
        fit = local_dispersion_fit(cand['k'], deg_idx=list(cand['pair']))
        if 'linear' in fit['verdict'] and 'mixed' not in fit['verdict']:
            linear_found = True
            break
    for pr in P_pairs:
        fit = local_dispersion_fit(P_PT, deg_idx=list(pr))
        if 'linear' in fit['verdict'] and 'mixed' not in fit['verdict']:
            linear_found = True
            break

    print()
    if linear_found:
        print("  => At least one Dirac-cone candidate found.")
        print("     Route C-iii's Step-2 premise is partially open;")
        print("     proceed to Step 3 (lightcone reading) and Step 4 (local-to-global).")
    else:
        print("  => NO Dirac-cone candidate found anywhere in this scan.")
        print("     Route C-iii is pre-falsified on the scalar Bloch H of srs.")
        print("     Recommended pivot: Route C-i (Krein-space NCG).")
    print()


if __name__ == "__main__":
    main()
