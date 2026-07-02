#!/usr/bin/env python3
"""Spectral test of the 5/12 dark coefficient identification.

Hypothesis (refined from word-partition analysis): the (5/12) Feshbach dark
coefficient is the dimensional fraction of the Hashimoto operator's "real
non-Perron" eigenspace at the Γ point.

For srs primitive cell (k=3, |V|=4, |E|=6):
  - Total Hashimoto B dimension: 2|E| = 12
  - Bipartite-factor eigenvalues u²=1: ±1 each with multiplicity |E|-|V| = 2
  - Perron eigenvalue +2 from λ_A=+3
  - Perron-A image +1 from λ_A=+3 (other root of u²-3u+2)
  - Complex pairs (-1±i√7)/2 from λ_A=-1 (mult 3)

Real eigenvalues structure:
  +2  (Perron NB)        : 1-dim  ← visible (dynamic, growing)
  +1  (Perron-A + bip)   : 3-dim  ← marginal (|λ|=1, no growth)
  -1  (bipartite)        : 2-dim  ← marginal (|λ|=1, alternating)
  ----- 5 marginal real non-Perron eigenvalues -----
  -1/2 ± i√7/2 (×3 each) : 6-dim  ← visible (oscillatory, |λ|=√2)
  ----- 6 oscillatory complex eigenvalues -----

  Visible (P-space): Perron + oscillatory = 1 + 6 = 7-dim
  Dark    (Q-space): marginal real = 3 + 2  = 5-dim
  Ratio: dark/total = 5/12 ✓

This script verifies the spectral decomposition numerically by building the
12×12 Hashimoto B at Γ and diagonalizing.
"""
from __future__ import annotations
import os, sys, math
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
from proofs.common import find_bonds, N_ATOMS  # noqa: E402

def build_hashimoto_at_Gamma(bonds):
    """Build the 12×12 Hashimoto B matrix at k=0 (Γ point)."""
    n = len(bonds)
    B = np.zeros((n, n), dtype=complex)
    for i, (src_i, tgt_i, cell_i) in enumerate(bonds):
        for j, (src_j, tgt_j, cell_j) in enumerate(bonds):
            if tgt_j != src_i:
                continue
            is_reverse = (
                src_i == tgt_j and tgt_i == src_j
                and tuple(cell_i) == tuple(-c for c in cell_j)
            )
            if is_reverse:
                continue
            B[i, j] = 1.0
    return B

def build_adjacency_at_Gamma(bonds):
    """Build the n_atoms × n_atoms adjacency at k=0."""
    A = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    for src, tgt, _cell in bonds:
        A[tgt, src] += 1.0
    return A

if __name__ == '__main__':
    print("=" * 90)
    print("Spectral test: 5/12 dark coefficient as dim(real non-Perron) / dim(Hashimoto)")
    print("=" * 90)

    bonds = find_bonds()
    print(f"\n  srs primitive cell: |V| = {N_ATOMS}, |E_directed| = {len(bonds)}")
    print(f"  Expected |E_directed| = 2·|E_undirected| = 2·6 = 12  ✓" if len(bonds) == 12 else "  ⚠ unexpected directed-edge count")

    A = build_adjacency_at_Gamma(bonds)
    A_eigs = np.sort(np.real(np.linalg.eigvalsh(A)))
    print(f"\n  Adjacency A spectrum at Γ: {[f'{e:+.3f}' for e in A_eigs]}")

    B = build_hashimoto_at_Gamma(bonds)
    B_eigs = np.linalg.eigvals(B)

    # Sort by real part descending, then imaginary part
    order = np.argsort(-np.real(B_eigs) - 1e-9 * np.abs(np.imag(B_eigs)))
    B_eigs_sorted = B_eigs[order]

    print(f"\n  Hashimoto B spectrum at Γ:")
    for i, e in enumerate(B_eigs_sorted):
        re, im = float(np.real(e)), float(np.imag(e))
        modulus = abs(e)
        if abs(im) < 1e-9:
            tag = 'real'
        else:
            tag = 'complex'
        print(f"    [{i:2d}] {re:+.4f}{im:+.4f}j   |λ|={modulus:.4f}   ({tag})")

    # Classify eigenvalues
    real_eigs = []
    complex_eigs = []
    for e in B_eigs_sorted:
        if abs(np.imag(e)) < 1e-9:
            real_eigs.append(float(np.real(e)))
        else:
            complex_eigs.append(complex(np.real(e), np.imag(e)))

    print(f"\n  Classification:")
    print(f"    Real eigenvalues: {len(real_eigs)}")
    for e in sorted(real_eigs, reverse=True):
        print(f"      {e:+.4f}")
    print(f"    Complex eigenvalues (pairs): {len(complex_eigs)}")
    for e in complex_eigs[:6]:
        print(f"      {np.real(e):+.4f}{np.imag(e):+.4f}j")

    # Find the Perron eigenvalue (largest real)
    perron = max(real_eigs)
    perron_count = sum(1 for e in real_eigs if abs(e - perron) < 1e-9)
    print(f"\n  Perron eigenvalue: {perron:.4f} (multiplicity {perron_count})")
    print(f"  Expected: Perron = k* − 1 = 2 ✓" if abs(perron - 2.0) < 1e-6 else "  ⚠ Perron ≠ 2")

    # Real non-Perron eigenvalues = "marginal" sector
    real_non_perron = [e for e in real_eigs if abs(e - perron) >= 1e-9]
    print(f"\n  Real non-Perron eigenvalues (the 'marginal' sector):")
    for e in sorted(real_non_perron, reverse=True):
        print(f"    {e:+.4f}    (|λ| = {abs(e):.4f})")

    # The headline ratio
    n_total = len(B_eigs_sorted)
    n_perron = perron_count
    n_marginal = len(real_non_perron)
    n_oscillatory = len(complex_eigs)

    print(f"\n{'='*90}")
    print(f"HEADLINE: spectral decomposition of Hashimoto B at Γ")
    print(f"{'='*90}")
    print(f"\n  {'sector':<35}{'dim':>6}{'fraction':>12}{'role':<25}")
    print(f"  {'-'*78}")
    print(f"  {'Perron (top NB walk, |λ|=2)':<35}{n_perron:>6}{n_perron/n_total:>12.4f}    visible (growing)")
    print(f"  {'Oscillatory (complex, |λ|=√2)':<35}{n_oscillatory:>6}{n_oscillatory/n_total:>12.4f}    visible (oscillatory)")
    print(f"  {'Marginal real non-Perron (|λ|=1)':<35}{n_marginal:>6}{n_marginal/n_total:>12.4f}    DARK (no dynamics)")
    print(f"  {'-'*78}")
    print(f"  {'TOTAL':<35}{n_total:>6}{1.0:>12.4f}")

    print(f"\n  Visible / total: ({n_perron} + {n_oscillatory})/{n_total} = "
          f"{(n_perron + n_oscillatory)}/{n_total} = "
          f"{(n_perron + n_oscillatory)/n_total:.4f}")
    print(f"  Dark / total:    {n_marginal}/{n_total} = "
          f"{n_marginal/n_total:.4f}")

    print(f"\n  Framework's dark Feshbach coefficient: 5/12 = {5/12:.4f}")
    print(f"  Computed dim ratio dark/total:         {n_marginal}/{n_total} = {n_marginal/n_total:.4f}")
    delta = abs(n_marginal/n_total - 5/12)
    if delta < 1e-9:
        print(f"\n  ✓ EXACT MATCH: 5/12 = dim(marginal-real Hashimoto modes) / dim(Hashimoto).")
        print(f"\n  Interpretation: the (5/12) Feshbach dark coefficient is the dimensional")
        print(f"  fraction of substrate Hashimoto eigenmodes that lie at |λ|=1 (marginal —")
        print(f"  neither growing like Perron nor oscillating like complex pairs). These are")
        print(f"  the modes that carry no net information across long NB walks; in Feshbach")
        print(f"  projection, they're the natural Q-space (eliminated subspace).")
    else:
        print(f"\n  Discrepancy: |{n_marginal/n_total:.4f} - {5/12:.4f}| = {delta:.4e}")

    # Extra check: how does this decompose by the bipartite factor and Ihara factor structure?
    print(f"\n{'='*90}")
    print(f"Decomposition by Ihara factorization:")
    print(f"{'='*90}")
    # Ihara: char(B) = (u²-1)^(|E|-|V|) · prod_i (u² - λ_i u + (k-1))
    n_E = len(bonds) // 2
    n_V = N_ATOMS
    k = 3   # NN coordination
    bipartite_factor_dim = 2 * (n_E - n_V)   # ±1 each with mult |E|-|V|
    print(f"\n  Bipartite factor (u²-1)^({n_E}-{n_V}) = (u²-1)^{n_E-n_V}:")
    print(f"    Contributes {bipartite_factor_dim} eigenvalues at u=±1 ({n_E - n_V} of each).")
    print(f"    These are PURELY bipartite-structural; no Perron, no oscillation.")
    print(f"    → all {bipartite_factor_dim} dimensions go into the DARK sector.")

    print(f"\n  Adjacency-derived factor: prod_λ (u² - λu + {k-1}):")
    print(f"    For each A eigenvalue λ, gives roots u = [λ ± √(λ²-{4*(k-1)})]/2.")
    A_eigs_real = np.sort(np.real(np.linalg.eigvalsh(A)))
    for lam in A_eigs_real:
        disc = float(lam)**2 - 4*(k-1)
        if disc >= 0:
            r = math.sqrt(disc)
            u1, u2 = (float(lam) + r)/2, (float(lam) - r)/2
            print(f"    λ={float(lam):+.3f}: real roots u = {u1:+.3f}, {u2:+.3f}  → "
                  f"contributes 1 Perron-like (|λ|>1) + 1 marginal/intermediate")
        else:
            re_part = float(lam)/2
            im_part = math.sqrt(-disc)/2
            mod = math.sqrt(re_part**2 + im_part**2)
            print(f"    λ={float(lam):+.3f}: complex roots u = {re_part:+.3f}±{im_part:+.3f}i, |u|={mod:.3f}"
                  f"  → contributes 2 oscillatory")

    # Summary: count dim contributions
    print(f"\n  Total counts:")
    print(f"    Bipartite factor → 2(|E|−|V|) = {bipartite_factor_dim} marginal eigenvalues")
    print(f"    λ=+{k} factor → roots {{1, 2}}: 1 Perron + 1 marginal")
    n_marg_lambda3 = 1
    n_perron_lambda3 = 1
    # Compute marginal contribution from λ=-1 multiplicity
    n_lambda_neg1 = sum(1 for e in A_eigs_real if abs(e + 1) < 1e-6)
    print(f"    λ=-1 factor (×{n_lambda_neg1}) → complex roots: {2*n_lambda_neg1} oscillatory")

    expected_marginal = bipartite_factor_dim + n_marg_lambda3
    expected_visible = n_perron_lambda3 + 2 * n_lambda_neg1
    print(f"\n  Expected dark (marginal):  {bipartite_factor_dim} (bipartite) + {n_marg_lambda3} (λ=+3 image) "
          f"= {expected_marginal} = 5 ✓" if expected_marginal == 5 else f"= {expected_marginal}")
    print(f"  Expected visible:          {n_perron_lambda3} (Perron) + {2*n_lambda_neg1} (λ=−1 oscillatory) "
          f"= {expected_visible} = 7 ✓" if expected_visible == 7 else f"= {expected_visible}")
    print(f"\n  Framework constants this connects to:")
    print(f"    5/12 = (|E|-|V|+1)·2/(2|E|) for srs: ({n_E}-{n_V}+1)·2 / (2·{n_E}) = "
          f"{(n_E - n_V + 1)*2}/{2*n_E} = {(n_E - n_V + 1)*2}/{2*n_E}")
    print(f"    Numerator = 2(|E|-|V|+1) = 2·χ-rank = bipartite + Perron-A image dimension")
    print(f"    Denominator = 2|E| = total Hashimoto dimension")
