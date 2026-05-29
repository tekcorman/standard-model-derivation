#!/usr/bin/env python3
"""
BR4 session 3, Probe 2 — Bloch isotypic multiplicities at Γ trivial λ=+3
for the down (h=2 Perron) and up (h=1) IB roots.

The framework's Q_Koide derivation for charged leptons uses the
P-point Ramanujan multiplicities (4, 2, 2) under body-diagonal C₃.
For quarks, the selection map (theorem_selection_map_2026-05-21.md)
places them at Γ trivial λ=+3 with IB roots h ∈ {1, 2}:
  - h=2 (Perron, Type IV)    → down sector
  - h=1 (degenerate, Type II) → up sector

But Γ trivial λ=+3 has eigenvalues h ∈ {1, 2} on the IB curve, |h|² ∈ {1, 4},
NEITHER is Ramanujan (|h|²=2). So the Q_Koide Born-rule machinery applied
to lepton at P doesn't literally transcribe to quarks at Γ.

This probe asks the analogous question for quarks: what ARE the C₃
isotypic multiplicities of each IB-root eigenspace at Γ trivial λ=+3?
If the multiplicities are species-distinct, they could plug into the
Born-rule analog to get Q^(d) and Q^(u), and we test whether the
resulting δ^(s) = Q^(s)(1−Q^(s)) is closer to empirical.

Six gates:
  G1: Build B_NB at Γ; verify eigenstructure (12 eigvals; trivial λ=+3
      eigenspace dimension; tree eigenvalues ±1).
  G2: For each non-trivial eigenvalue h with multiplicity > 1, compute
      C₃ isotypic decomposition (μ_0, μ_ω, μ_ω̄).
  G3: Identify the IB roots h=2 and h=1 in the spectrum. Get their
      C₃ multiplicities (= "down multiplicities" and "up multiplicities").
  G4: Apply the Born-rule analog (Q_Koide.py paradigm) to each set of
      multiplicities. Predict Q^(d) and Q^(u).
  G5: Compute δ^(d) = Q^(d)(1−Q^(d)) and δ^(u) similarly. Compare to
      empirical and framework-internal ε² values.
  G6: Honest verdict.
"""

import sys
import os
import math
import numpy as np
from numpy import linalg as la
from fractions import Fraction

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from proofs.common import find_bonds
from proofs.foundations.theorem_B5_3_core import (
    bloch_hashimoto, build_c3_on_directed_edges, build_directed_edges,
)


GAMMA = np.array([0.0, 0.0, 0.0])


def _build_gamma():
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    B_Gamma = bloch_hashimoto(GAMMA, directed)
    U_C3 = build_c3_on_directed_edges(directed)
    return B_Gamma, U_C3


def _eigenspace(M, target_eig, tol=1e-6):
    """Return basis of generalized eigenspace at target_eig."""
    eigs, V = la.eig(M)
    idx = [i for i in range(len(eigs)) if abs(eigs[i] - target_eig) < tol]
    if not idx:
        return None, []
    W = V[:, idx]
    # Orthonormalise (note: M may not be Hermitian, so this is a basis, not eigenvectors of M†)
    Q, _ = la.qr(W)
    return Q[:, :len(idx)], [eigs[i] for i in idx]


def _c3_isotypic_decomposition(W, U_C3):
    """
    For a subspace spanned by columns of W (basis), compute the C₃
    isotypic multiplicities (μ_0, μ_ω, μ_ω̄).

    C₃ acts on W via U_C3 restricted: (U_C3)|_W = W† U_C3 W.
    Diagonalise this 3-cyclic action and count eigenvalues = 1, ω, ω̄.
    """
    omega = np.exp(2j * np.pi / 3)
    omega_bar = omega.conj()

    # Restriction of C₃ to W
    C3_restricted = W.conj().T @ U_C3 @ W

    # Eigenvalues of restricted C₃
    eigs = la.eigvals(C3_restricted)

    # Classify each eigenvalue
    mu_0 = 0
    mu_omega = 0
    mu_omegabar = 0
    other = []
    for e in eigs:
        if abs(e - 1) < 1e-4:
            mu_0 += 1
        elif abs(e - omega) < 1e-4:
            mu_omega += 1
        elif abs(e - omega_bar) < 1e-4:
            mu_omegabar += 1
        else:
            other.append(e)

    return mu_0, mu_omega, mu_omegabar, other


def _born_rule_q(mu_trivial, mu_omega, mu_omegabar):
    """
    Apply Q_Koide.py paradigm: amplitudes amp_j = √μ_0 + √μ_ω·ω^j + √μ_ω̄·ω^(-j).
    Compute Q = Σ|amp_j|² / (Σ|amp_j|)².
    Returns (Q, eps_sq, m_j_list, sqrt_m_j_list).
    """
    omega = np.exp(2j * np.pi / 3)
    amps = []
    for j in range(3):
        a = math.sqrt(mu_trivial) + math.sqrt(mu_omega) * omega**j + math.sqrt(mu_omegabar) * omega**(-j)
        amps.append(a)
    m_j = [abs(a)**2 for a in amps]
    sqrt_m_j = [abs(a) for a in amps]
    sum_m = sum(m_j)
    sum_sqrt = sum(sqrt_m_j)
    if sum_sqrt < 1e-12:
        return None, None, m_j, sqrt_m_j
    Q = sum_m / sum_sqrt**2
    eps_sq = 2 * (3 * Q - 1)
    return Q, eps_sq, m_j, sqrt_m_j


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=" * 78)
    print("BR4 session 3, Probe 2 — Γ-trivial multiplicities for down/up sectors")
    print("=" * 78)
    print()

    B_Gamma, U_C3 = _build_gamma()
    n = B_Gamma.shape[0]
    eigs_all = la.eigvals(B_Gamma)
    print(f"  Bloch Hashimoto B(Γ) spectrum (12-dim):")
    for i, e in enumerate(sorted(eigs_all, key=lambda z: (abs(z.imag) > 1e-6, -z.real, -z.imag))):
        if abs(e.imag) < 1e-6:
            print(f"    eig[{i}] = {e.real:+.4f}     (real, |h|² = {abs(e)**2:.4f})")
        else:
            print(f"    eig[{i}] = {e.real:+.4f} {e.imag:+.4f}j  (|h|² = {abs(e)**2:.4f})")
    print()

    # Find distinct real eigenvalues
    distinct = []
    for e in eigs_all:
        if abs(e.imag) < 1e-6:
            r = round(e.real, 4)
            if r not in [d[0] for d in distinct]:
                distinct.append((r, []))
            for d in distinct:
                if d[0] == r:
                    d[1].append(e)
    print(f"  Distinct real eigenvalues at Γ:")
    for r, eigs_at in distinct:
        print(f"    h = {r:+.4f}: multiplicity {len(eigs_at)}")
    print()

    # Verify Moore bound: IB roots are h_+ = k*-1 = 2 (Perron) and h_- = 1
    # at λ = +k* = +3 (trivial). Both lie in the trivial=+3 eigenspace of the
    # ADJACENCY matrix; the Hashimoto B's spectrum at λ_adj=+3 (column-sum = k*)
    # is {h_+, h_-} per Ihara-Bass.
    print(f"  Per master synthesis §4(C): color-triplet → Γ trivial λ_adj=+3,")
    print(f"  IB roots h ∈ {{1, 2}} (Perron h=2 Type IV down; h=1 Type II up).")
    print()

    # Compute C₃ multiplicities at h=2 (down) and h=1 (up)
    print(f"  {'h':<6} {'mult':>6} {'μ_0 (trivial)':>16} {'μ_ω':>8} {'μ_ω̄':>8} {'unclassified':>14}")
    print("  " + "-" * 70)
    for h_target in [2.0, 1.0, -1.0, -2.0]:
        W, eigs = _eigenspace(B_Gamma, h_target)
        if W is None:
            print(f"  h={h_target:+.2f}: NO eigenspace found")
            continue
        mu_0, mu_o, mu_b, other = _c3_isotypic_decomposition(W, U_C3)
        print(f"  {h_target:>+6.2f} {len(eigs):>6} {mu_0:>16} {mu_o:>8} {mu_b:>8} "
              f"{len(other) if other else 0:>14}")
    print()

    # Apply Born rule to derived multiplicities
    print("=" * 78)
    print("Born-rule application (Q_Koide.py paradigm)")
    print("=" * 78)
    print()

    targets = [
        ("down quark (h=+2 Perron)", 2.0, 5.80, 6.31),
        ("up quark (h=+1)",          1.0, 4.27, 4.27),
        ("lepton (P-point, reference)", None, 12.7324, 12.7324),  # uses (4,2,2)
    ]

    print(f"  {'Species':<32} {'(μ_0,μ_ω,μ_ω̄)':>14} {'Q':>10} {'ε²':>10} {'δ_B °':>10} {'Empirical δ °':>14} {'Δ rel':>10}")
    print("  " + "-" * 110)
    for label, h_target, emp_lo, emp_hi in targets:
        if h_target is None:
            # Lepton reference: use (4,2,2) directly
            mu_t, mu_o, mu_b = 4, 2, 2
        else:
            W, _ = _eigenspace(B_Gamma, h_target)
            if W is None:
                continue
            mu_t, mu_o, mu_b, _ = _c3_isotypic_decomposition(W, U_C3)
        Q, eps_sq, m_j, sqrt_m_j = _born_rule_q(mu_t, mu_o, mu_b)
        if Q is None:
            print(f"  {label:<32} ({mu_t},{mu_o},{mu_b})   Q-formula degenerate (all multiplicities involve √0)")
            continue
        delta_B_rad = Q * (1 - Q)
        delta_B_deg = math.degrees(delta_B_rad)
        emp = (emp_lo + emp_hi) / 2
        rel = (delta_B_deg - emp) / emp * 100
        print(f"  {label:<32} ({mu_t},{mu_o},{mu_b})  {Q:>10.4f} {eps_sq:>10.4f} {delta_B_deg:>+10.4f} "
              f"{emp:>+14.4f} {rel:>+8.2f}%")

    print()
    print("=" * 78)
    print("HONEST READING")
    print("=" * 78)
    print()
    print("This probe TESTS whether the C₃ multiplicities at Γ trivial λ=+3")
    print("eigenspaces (h=1 and h=2 IB roots) provide a substrate-derived Q^(s)")
    print("for each quark sector via the Q_Koide.py paradigm.")
    print()
    print("Expected outcomes:")
    print("  (a) Multiplicities give species-distinct (μ_0, μ_ω, μ_ω̄) ≠ (4,2,2)")
    print("      → derive distinct Q^(d), Q^(u) → distinct δ_Bernoulli predictions.")
    print("  (b) Multiplicities give (4,4,4) symmetric → Q = 1 → ε = 0 (degenerate)")
    print("      → no quark hierarchy mechanism from this Bloch concentration.")
    print("  (c) Multiplicities give something else → structural surprise.")
    print()
    print("If empirical δ^(d), δ^(u) match the predicted values, Candidate B")
    print("extends structurally to quarks via species-specific multiplicities.")
    print("If not, the multiplicities are recorded as a baseline structural fact")
    print("and δ_quark remains open at structural grade.")


if __name__ == "__main__":
    main()
