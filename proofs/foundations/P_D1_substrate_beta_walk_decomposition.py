#!/usr/bin/env python3
"""
P_D1_substrate_beta_walk_decomposition.py
==========================================

P-D1 session 1: decompose closed-walk counts by Bloch-visibility.

Key conceptual move:
  Closed walks of length L on srs are classified by winding class W ∈ Z³.
  Bloch averaging over the BZ acts on a walk count via:
    ∫_BZ N(L, k) d³k  =  Σ_W N(L, W) · ∫_BZ e^{2πi k·W} d³k
                      =  N(L, W = 0)
  (the Fourier kernel δ_{W,0} kills all walks with non-zero winding).

  So:
    Tr_full B^L = Σ_W N(L, W)       (all walks)
    Tr_Bloch B^L = N(L, W=0)         (Bloch-averaged; keeps only zero-winding)
    ΔN(L) = Tr_full B^L − N(L, W=0) = "Layer-1 content at length L"
                                       (walks with non-trivial winding,
                                        ESCAPE Bloch averaging)

  Per the standard CC spectral action, the β-function contribution from
  loops of length L sees only the Bloch-averaged content N(L, W=0).
  The full substrate β would additionally see ΔN(L) — these are
  candidate Layer-1 contributions per NA-4.

This probe:
  1. Computes N(L, W) for all closed NB walks of length L up to L_max,
     classified by W ∈ Z³.
  2. Reports Tr B^L (= total) vs N(L, W=0) (= Bloch-visible).
  3. Reports ΔN(L) = Layer-1 content per length.
  4. Decomposes ΔN(L) by Z_2 sector (Probe C: L even ↔ W even-sum).
  5. Comments on implications for β-function contributions.

No graded content changes.  P-D1 session 1 setup probe.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import find_bonds  # noqa: E402
from proofs.foundations.theorem_B5_3_core import (  # noqa: E402
    build_directed_edges, bloch_hashimoto,
)
from proofs.foundations.probe_C_winding_invariants_srs import (  # noqa: E402
    enumerate_NB_closed_walks_by_winding,
)


np.set_printoptions(precision=6, suppress=True, linewidth=140)
TOL = 1e-9


# ---------------------------------------------------------------------------
# Tr B(k)^L at multiple k-points — cross-check with walk enumeration
# ---------------------------------------------------------------------------

def trace_B_at_k(L, k, directed):
    B_k = bloch_hashimoto(k, directed)
    B_L = np.linalg.matrix_power(B_k, L)
    return np.trace(B_L)


def bloch_avg_trace(L, directed, n_grid=8):
    """Compute (1/V_BZ) ∫_BZ Tr B(k)^L d³k via Monte Carlo / grid sum.
    Result should equal N(L, W=0) (zero-winding closed walks of length L)."""
    total = 0.0j
    n_pts = 0
    for ix in range(n_grid):
        for iy in range(n_grid):
            for iz in range(n_grid):
                k = (ix / n_grid, iy / n_grid, iz / n_grid)
                total += trace_B_at_k(L, k, directed)
                n_pts += 1
    return total / n_pts


# ---------------------------------------------------------------------------
# Main decomposition
# ---------------------------------------------------------------------------

def part_A_compute(L_max):
    print("=" * 100)
    print(f"PART A — Enumerate N(L, W) for L ≤ {L_max}")
    print("=" * 100)
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    counts = enumerate_NB_closed_walks_by_winding(directed, L_max)
    print(f"  Done. Total walks across all lengths: {sum(counts.values())}")
    return counts, directed


def part_B_decomposition(counts):
    print("\n" + "=" * 100)
    print("PART B — Bloch-visible (W=0) vs Layer-1 content (W≠0) per length L")
    print("=" * 100)
    # Aggregate by length
    by_length = defaultdict(lambda: {'total': 0, 'W_zero': 0, 'W_nonzero': 0,
                                     'sum_W_even': 0, 'sum_W_odd': 0})
    for (L, W), c in counts.items():
        d = by_length[L]
        d['total'] += c
        if W == (0, 0, 0):
            d['W_zero'] += c
        else:
            d['W_nonzero'] += c
        sumW = W[0] + W[1] + W[2]
        if sumW % 2 == 0:
            d['sum_W_even'] += c
        else:
            d['sum_W_odd'] += c

    print(f"\n  {'L':>3s}  {'Tr B^L':>8s}   {'N(L,0)':>8s}  {'ΔN(L)':>10s}   "
          f"{'%Layer-1':>9s}   {'Even-sec':>9s}  {'Odd-sec':>9s}")
    for L in sorted(by_length):
        d = by_length[L]
        if d['total'] == 0: continue
        pct = 100 * d['W_nonzero'] / d['total']
        print(f"  {L:>3d}  {d['total']:>8d}   {d['W_zero']:>8d}  {d['W_nonzero']:>10d}   "
              f"{pct:>7.1f}%   {d['sum_W_even']:>9d}  {d['sum_W_odd']:>9d}")

    print(r"""
  Tr B^L  = total closed walks of length L on srs (= Σ_W N(L, W))
  N(L, 0) = zero-winding walks (Bloch-visible; survive ∫_BZ averaging)
  ΔN(L)   = N(L, W≠0) (Layer-1 content; killed by Bloch averaging)
  Even-sec = walks with Σ W_i even (per Probe C);  Odd-sec = Σ W_i odd""")
    return by_length


def part_C_bloch_avg_sanity(directed, by_length):
    print("\n" + "=" * 100)
    print("PART C — Bloch-average sanity: (1/V_BZ) ∫ Tr B(k)^L d³k = N(L, 0)?")
    print("=" * 100)
    # For each length, compute Bloch average and compare to N(L, W=0)
    n_grid = 6
    print(f"\n  Grid size: {n_grid}³ = {n_grid**3} k-points (uniform Bloch averaging)")
    print(f"\n  {'L':>3s}  {'∫ Tr d³k / V':>15s}   {'N(L, 0)':>10s}   match?")
    for L in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if L not in by_length:
            continue
        avg = bloch_avg_trace(L, directed, n_grid=n_grid)
        N_zero = by_length[L]['W_zero']
        # The trace should be real for unitary-symmetric edge enumeration
        match = abs(avg.real - N_zero) < 0.5  # tolerance for grid discretization
        print(f"  {L:>3d}  {avg.real:>13.2f}{('  '):2s}   {N_zero:>10d}    "
              f"{'✓' if match else 'MISMATCH'}")


def part_D_layer1_content(by_length):
    print("\n" + "=" * 100)
    print("PART D — Layer-1 content fraction by length")
    print("=" * 100)
    print(r"""
  Layer-1 content = closed walks with non-trivial winding W ≠ (0,0,0).
  These are killed by Bloch averaging (they "escape Layer-2").
  Per NA-4 framing, they're the candidate observables for substrate
  Layer-1 escape from Bloch decomposition.""")
    print(f"\n  {'L':>3s}  {'Total':>8s}  {'Layer-1':>10s}  {'% Layer-1':>10s}  {'Parity':>8s}")
    for L in sorted(by_length):
        d = by_length[L]
        if d['total'] == 0: continue
        pct = 100 * d['W_nonzero'] / d['total']
        parity = "even" if L % 2 == 0 else "odd"
        marker = ' ←' if pct == 100 else ''
        print(f"  {L:>3d}  {d['total']:>8d}  {d['W_nonzero']:>10d}  {pct:>8.1f}%   "
              f"{parity:>8s}{marker}")
    print(r"""
  KEY OBSERVATION: at L < girth = 10, 100% of closed walks have non-trivial
  winding.  ALL of these walks are Layer-1 content.  Zero-winding walks
  first appear at L = girth = 10 (when a walk can "close back to its
  origin" in the same cell).

  For L < 10, the framework's existing derivations (e.g., λ_Higgs at L=8)
  are using ENTIRELY Layer-1 content already — they're not Bloch-averaged
  traces but specific spectral eigenvalue components that retain
  winding-phase information.

  For L ≥ 10, both Layer-2 (W=0) and Layer-1 (W≠0) content coexist.""")


def part_E_implications(by_length):
    print("\n" + "=" * 100)
    print("PART E — Implications for substrate β")
    print("=" * 100)
    print(r"""
  Standard CC spectral-action β picks up matter-loop contributions of
  the form  Σ_L (b_L / L) where b_L counts how strongly the gauge field
  is coupled to closed matter loops of length L.

  For walks on srs:
    b_L^{Bloch}     = c · N(L, W=0)         (Bloch-averaged content)
    b_L^{Layer-1}   = c · N(L, W≠0)         (winding-non-trivial content)

  The framework's CURRENT β-running uses standard QFT β-functions with
  ADOPTED MSSM matter — it implicitly absorbs all walk contributions
  into the matter-loop calculation, then assumes the matter content is
  MSSM.

  P-D1's question: can we compute β DIRECTLY from substrate walks,
  splitting the Bloch and Layer-1 pieces?  If we do, does Layer-1
  contribute exactly the missing  Δb = (+5/2, +25/6, +4) ?

  Path forward (multi-session):

  P-D1.a — Derive the spectral action's a_4 coefficient from substrate
           walks explicitly, with both Bloch and Layer-1 pieces.
  P-D1.b — Identify the gauge-charge weighting of each walk (depends on
           which fermion species the walk threads through).
  P-D1.c — Compute b_i_Bloch and b_i_Layer-1 separately per gauge factor.
  P-D1.d — Compare b_i_Layer-1 to MSSM Δb target.""")


# ---------------------------------------------------------------------------
def main():
    print(r"""
==========================================================================================
P-D1 session 1 — Substrate β-function from spectral action: walk-class decomposition
==========================================================================================""")
    L_max = 14
    counts, directed = part_A_compute(L_max)
    by_length = part_B_decomposition(counts)
    part_C_bloch_avg_sanity(directed, by_length)
    part_D_layer1_content(by_length)
    part_E_implications(by_length)
    print("\n" + "=" * 100)
    print("P-D1 session 1 sentinel: done.")
    print("=" * 100)


if __name__ == "__main__":
    main()
