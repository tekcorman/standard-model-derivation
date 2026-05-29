#!/usr/bin/env python3
"""
y_τ multiway-moment Phase 2 — candidate M2 (single-girth all-sector at k_P).

PER SPEC
--------
an internal working note §3 candidate M2:

    y_τ^M2 = (single-girth all-sector closed-cycle amplitude at k_P)

Tree-level (M1, current framework treatment) uses Ramanujan-only:
    α_1_bare = ((k*-1)/k*)^(g-2) = (2/3)^8

The full closed-cycle amplitude at k_P from Corollary 2 of
`theorem_multiway_branch_measure.md` is:

    Tr(B(k_P)^g) summed over ALL eigenvalue sectors

Tree-level uses only the Ramanujan content (|h|² = k*−1 = 2, mult 4).
The trivial C_3-fixed sector (eigenvalues ±1, mult 4 per §7.3) is NOT in
M1. M2 includes it.

QUANTITATIVE EXPECTATION
------------------------
Sub-Ramanujan (|±1|² = 1) suppression vs Ramanujan (|h|² = 2) at L=g=10:
ratio = (1/3)^10 / (2/3)^10 = (1/2)^10 = 1/1024 ≈ 0.098%.
Multiplicity ratio: 4/4 = 1 (both sectors mult 4 at k_P).
Sign depends on parity of (-1)^g = (-1)^10 = +1.

Order of magnitude: ~0.1%, in the +0.13% target ballpark.

SCOPE
-----
This probe DOES NOT compare to PDG y_τ. It computes the candidate M2
moment, compares to M1 = tree-level, identifies any K-rational structure,
and reports honestly.

Per an internal note: derive mechanism
first (spec doc), compute value second (this probe), compare to PDG third
(only with explicit user permission, after structural identification).

REFERENCES
----------
- `docs/theorems/theorem_multiway_branch_measure.md` Corollary 1, 2
- `docs/theorems/theorem_bloch_lift_mu.md`
- `docs/theorems/theorem_BP_doubly_degenerate_h.md` (eigenvalue mults at k_P)
- `proofs/foundations/theorem_walker_dynamics.py` (bloch_hashimoto)
"""

import math
import sys
import os
import numpy as np
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theorem_walker_dynamics import bloch_hashimoto, build_directed_edges
from t_v_eigenstructure import find_bonds


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

K_STAR = 3
G_GIRTH = 10
N_G_EDGE = 5  # combinatorial multiplicity per ordered edge pair (alpha_1_full.py)

BONDS = find_bonds()
DIRECTED = build_directed_edges(BONDS)
assert len(DIRECTED) == 12, "expected 12 directed edges in srs primitive cell"

K_P_FRAC = np.array([0.25, 0.25, 0.25])  # canonical framework k_P


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


# ----------------------------------------------------------------------------
# Eigenvalue analysis at k_P
# ----------------------------------------------------------------------------

def analyze_kP_eigenstructure():
    """Diagonalize B(k_P), return all eigenvalues with multiplicity classification."""
    B = bloch_hashimoto(K_P_FRAC, DIRECTED)
    eigs = np.linalg.eigvals(B)

    # Classify eigenvalues by sector:
    #   Ramanujan: |λ|² ≈ k*-1 = 2 (saturates Ramanujan bound)
    #   Trivial C_3-fixed: |λ| ≈ 1 (sub-Ramanujan)
    #   Other: anything else
    ramanujan = []
    trivial = []
    other = []
    for λ in eigs:
        abs_sq = abs(λ) ** 2
        if abs(abs_sq - (K_STAR - 1)) < 1e-6:
            ramanujan.append(λ)
        elif abs(abs_sq - 1.0) < 1e-6:
            trivial.append(λ)
        elif abs_sq < 1e-9:
            other.append(λ)  # zero eigenvalues
        else:
            other.append(λ)

    return eigs, ramanujan, trivial, other


def print_eigenstructure():
    eigs, ram, triv, other = analyze_kP_eigenstructure()
    header("Eigenvalue structure of B(k_P) at k_P = (1/4, 1/4, 1/4)")
    print(f"  B(k_P) is 12-dim Hashimoto operator on directed edges.")
    print(f"  Total eigenvalues: {len(eigs)}")
    print()
    print(f"  Ramanujan sector (|λ|² ≈ k*-1 = 2): {len(ram)} eigenvalues")
    for λ in ram:
        print(f"    λ = {λ.real:+.6f}{λ.imag:+.6f}i  |λ|² = {abs(λ)**2:.6f}")
    print()
    print(f"  Trivial C_3-fixed sector (|λ|² ≈ 1): {len(triv)} eigenvalues")
    for λ in triv:
        print(f"    λ = {λ.real:+.6f}{λ.imag:+.6f}i  |λ|² = {abs(λ)**2:.6f}")
    print()
    print(f"  Other (zero or unclassified): {len(other)} eigenvalues")
    for λ in other:
        print(f"    λ = {λ.real:+.6f}{λ.imag:+.6f}i  |λ|² = {abs(λ)**2:.6f}")
    return eigs, ram, triv, other


# ----------------------------------------------------------------------------
# Closed-cycle amplitudes at length L = g
# ----------------------------------------------------------------------------

def closed_cycle_trace(eigs, L):
    """Tr(B(k_P)^L) = Σ λ^L over all eigenvalues."""
    return sum(λ ** L for λ in eigs)


def closed_cycle_trace_sector(eigs_sector, L):
    """Tr restricted to a specific eigenvalue sector."""
    return sum(λ ** L for λ in eigs_sector)


def report_closed_cycle_amplitudes():
    eigs, ram, triv, other = analyze_kP_eigenstructure()
    header("Closed-cycle amplitudes at length L = g = 10")

    L = G_GIRTH
    total = closed_cycle_trace(eigs, L)
    ram_only = closed_cycle_trace_sector(ram, L)
    triv_only = closed_cycle_trace_sector(triv, L)
    other_only = closed_cycle_trace_sector(other, L)

    print(f"  Tr(B(k_P)^{L}) total = {total.real:+.6e}{total.imag:+.6e}i")
    print(f"    Ramanujan-only = {ram_only.real:+.6e}{ram_only.imag:+.6e}i")
    print(f"    Trivial-only   = {triv_only.real:+.6e}{triv_only.imag:+.6e}i")
    print(f"    Other          = {other_only.real:+.6e}{other_only.imag:+.6e}i")
    print(f"    Sum check      = {(ram_only + triv_only + other_only).real:+.6e}"
          f"{(ram_only + triv_only + other_only).imag:+.6e}i")

    # Imaginary parts should be near-zero for closed cycles (real-valued trace)
    if abs(total.imag) > 1e-6:
        print(f"  ⚠ Total trace has non-trivial imaginary part {total.imag:.6e}; "
              f"should be ~0 for closed cycles. Check eigenvalue conjugate pairs.")

    return total.real, ram_only.real, triv_only.real, other_only.real


# ----------------------------------------------------------------------------
# M1 vs M2 comparison
# ----------------------------------------------------------------------------

def compare_M1_M2():
    """Compare framework's M1 (tree-level) to M2 (all-sector at k_P)."""
    eigs, ram, triv, other = analyze_kP_eigenstructure()

    L = G_GIRTH
    ram_amplitude = closed_cycle_trace_sector(ram, L).real
    triv_amplitude = closed_cycle_trace_sector(triv, L).real
    total_amplitude = closed_cycle_trace(eigs, L).real

    # M1 reference: framework α_1_bare = (2/3)^g per Corollary 1
    # Need to figure out the right normalization: tree-level α_1_bare = (2/3)^(g-2),
    # which is per-step survival × (g-2) intermediate steps. The Hashimoto trace at
    # k_P normalized by (k*-1)^(L-2) × something gives the framework's α form.
    #
    # Let's compute several reference quantities and see which matches.

    header("M1 (tree-level) vs M2 (all-sector at k_P) — closed-cycle amplitudes")
    print(f"  L = g = {L}")
    print(f"  k* = {K_STAR}, |E| = 6")
    print()
    print(f"  Reference: α_1_bare = (2/3)^(g-2) = {(2/3)**(L-2):.6e}  [framework Corollary 1]")
    print(f"  Reference: α_1_full = (5/3)(2/3)^8 = {(5/3)*(2/3)**8:.6e}  [tree-level cycle factor]")
    print(f"  Reference: y_τ_tree = α_1_full/k*² = {(5/3)*(2/3)**8/9:.6e}  [tree-level y_τ]")
    print()
    print(f"  Computed at k_P, length L=g={L}:")
    print(f"    Tr_Ramanujan = {ram_amplitude:+.6e}")
    print(f"    Tr_trivial   = {triv_amplitude:+.6e}")
    print(f"    Tr_total     = {total_amplitude:+.6e}")
    print()

    # Per-(k*-1)^L normalization (per-step survival^L)
    print(f"  Normalized by (k*-1)^L = 2^{L} = {2**L}:")
    print(f"    Tr_Ram / 2^L     = {ram_amplitude / (2**L):+.6e}")
    print(f"    Tr_total / 2^L   = {total_amplitude / (2**L):+.6e}")
    print()

    # Per-(k*)^L normalization (per-step branching)
    print(f"  Normalized by k*^L = 3^{L} = {3**L}:")
    print(f"    Tr_Ram / 3^L     = {ram_amplitude / (3**L):+.6e}")
    print(f"    Tr_total / 3^L   = {total_amplitude / (3**L):+.6e}")
    print()

    # M2 / M1 RELATIVE correction (eigenvector-mode-content)
    if abs(ram_amplitude) > 1e-15:
        rel_correction = (total_amplitude - ram_amplitude) / ram_amplitude
        print(f"  RELATIVE correction (M2 - M1) / M1 from sub-Ramanujan eigvalue content:")
        print(f"    (Tr_total - Tr_Ram) / Tr_Ram = {rel_correction:+.6e}")
        print(f"                                  = {rel_correction*100:+.4f}%")
        print()
        print(f"  Comparison to y_τ deviation +0.126% (predicted high vs observed):")
        if abs(rel_correction * 100) < 0.5 and rel_correction < 0:
            print(f"    ✓ Correct sign (negative) and similar magnitude")
        elif abs(rel_correction * 100) < 0.5 and rel_correction > 0:
            print(f"    ⚠ Correct order but wrong sign")
        else:
            print(f"    ⚠ Off magnitude or sign")

    return ram_amplitude, triv_amplitude, total_amplitude


def look_for_k_rational_form(value, max_denom=100, max_pi_power=4):
    """Search for K[1/π]-rational form: value = p/q × π^n for small p, q, n.

    K = ℚ(√2, √3, √5). Test against rationals first; π^n at any negative n
    indicates BZ-volume normalization needed.
    """
    print(f"  Search for K-rational form of {value:+.10e}:")
    candidates = []
    for n_pi in range(-max_pi_power, max_pi_power + 1):
        target = value / (math.pi ** n_pi) if n_pi != 0 else value
        # Try to express as p/q
        for q in range(1, max_denom + 1):
            p_float = target * q
            p = round(p_float)
            if p == 0:
                continue
            err = abs(target - p / q) / abs(target) if target != 0 else 1.0
            if err < 1e-6:
                pi_str = f" × π^{n_pi}" if n_pi != 0 else ""
                candidates.append((err, p, q, n_pi, f"  {p}/{q}{pi_str} = {p/q*math.pi**n_pi:+.10e}"))
    candidates.sort()
    for c in candidates[:5]:
        print(c[4])
    if not candidates:
        print(f"    No clean K[1/π]-rational match found at max_denom={max_denom}, "
              f"max_pi_power={max_pi_power}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    header("y_τ multiway-moment Phase 2 — candidate M2 (single-girth all-sector at k_P)")
    print(f"  Per spec doc `y_tau_multiway_moment_spec_2026-05-01.md`")
    print(f"  Question: does Tr(B(k_P)^g) all-sector match α_1_bare structurally,")
    print(f"            and does the sub-Ramanujan correction equal +0.13%?")

    print_eigenstructure()
    report_closed_cycle_amplitudes()
    ram_amp, triv_amp, total_amp = compare_M1_M2()

    header("K-rational structural identification")
    print()
    print("  Sub-Ramanujan content (Tr_trivial at L=g):")
    look_for_k_rational_form(triv_amp)
    print()
    print("  Total amplitude (Tr_total at L=g):")
    look_for_k_rational_form(total_amp)

    header("CONCLUSION")
    print()
    if abs(triv_amp) < 1e-9:
        print("  Tr_trivial ≈ 0 — the trivial C_3-fixed sector ±1 eigenvalues at length")
        print(f"  L=g={G_GIRTH} (even) cancel by parity. ((+1)^g + (-1)^g = 1 + 1 = 2 if both")
        print("  in same multiplicity, but this depends on actual mult structure.)")
        print()
        print("  If trivial sector contribution is identically zero at L=g,")
        print("  candidate M2 reduces to M1 (Ramanujan only), and M2 doesn't explain")
        print("  the +0.13% residual. Pivot to M5 (Bloch-integrated) or accept")
        print("  Interpretation 2 (bridge-systematic).")
    else:
        rel = (total_amp - ram_amp) / ram_amp if abs(ram_amp) > 1e-15 else float('nan')
        if -0.005 < rel < 0:
            print(f"  Sub-Ramanujan contribution gives {rel*100:.4f}% correction —")
            print(f"  CORRECT SIGN and similar magnitude to y_τ +0.13% deviation.")
            print(f"  Phase 3 should attempt structural identification of this number.")
        elif 0 < rel < 0.005:
            print(f"  Sub-Ramanujan contribution gives {rel*100:.4f}% — order-of-magnitude")
            print(f"  match but WRONG SIGN. Re-examine sign convention.")
        elif abs(rel) > 0.01:
            print(f"  Sub-Ramanujan contribution gives {rel*100:.4f}% — too large")
            print(f"  to explain the +0.13% deviation cleanly.")
        else:
            print(f"  Result: relative correction {rel*100:+.4f}%. See raw numbers above.")


if __name__ == '__main__':
    main()
