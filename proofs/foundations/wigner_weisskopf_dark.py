#!/usr/bin/env python3
"""
wigner_weisskopf_dark.py — Subspace-projected Lindblad on the srs Ramanujan
============================================================================

GOAL: Numerically verify the closure of α₁_bare = (2/3)^8 via the
subspace-projected Wigner-Weisskopf decay route described in
an internal working note §9.4b.

The construction:
  1. Build B(k_P), the 12×12 Bloch Hashimoto at the P-point of srs.
  2. Find V_Ram = span{eigenvectors of B(k_P) with eigenvalues ±h, ±h̄}.
  3. Build projector P_R onto V_Ram.
  4. Construct projected jump operators L_e = P_R |reverse(e)⟩⟨e| P_R
     for each directed edge e.
  5. Verify the Lindblad rate γ that emerges from Σ_e L_e† L_e on V_Ram.
  6. Compute coherence decay over L = g - 2 = 8 steps.

EXPECTED RESULT:
  - The rate operator Σ_e L_e† L_e |_{V_Ram} should be proportional to I_R
    (identity on V_Ram), with proportionality constant γ ≤ 1.
  - The visible coherence after L steps decays as (1 - γ)^L.
  - For γ = 1/k = 1/3 and L = 8: decay = (2/3)^8 = α₁_bare ✓.

If the rate is exactly 1/k, the chain closes. If not, we need to understand
why the projection modifies the rate, and whether that modification is
itself derivable from A1+A2+A3.

Status: Numerical verification of Step 9.4b of the attempt document.
"""

import math
import os
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import find_bonds, ATOMS, A_PRIM, N_ATOMS  # noqa: E402
from proofs.foundations.theorem_B5_3_core import (  # noqa: E402
    build_directed_edges,
    bloch_hashimoto,
)
from proofs.foundations.cocycle_check_vram import (  # noqa: E402
    find_vram_basis,
    project_to_subspace,
)


# ===========================================================================
# CONSTANTS
# ===========================================================================

K_STAR = 3
GIRTH = 10
G_MINUS_2 = GIRTH - 2  # = 8
K_P = (0.25, 0.25, 0.25)
H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
ALPHA_1_BARE_EXACT = Fraction(2, 3) ** G_MINUS_2  # = 256/6561


def header(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print()


# ===========================================================================
# BUILD INFRASTRUCTURE
# ===========================================================================

def reverse_edge_map(directed):
    """Map each directed-edge index to its reverse index.

    `directed` is a list of (src_atom, tgt_atom, cell_offset_tuple).
    The reverse of (src, tgt, cell) is (tgt, src, -cell).
    """
    edge_to_idx = {de: i for i, de in enumerate(directed)}
    rev = {}
    for i, (src, tgt, cell) in enumerate(directed):
        neg_cell = tuple(-c for c in cell)
        reverse_de = (tgt, src, neg_cell)
        if reverse_de in edge_to_idx:
            rev[i] = edge_to_idx[reverse_de]
        else:
            # Reverse not in same primitive cell — try modular arithmetic
            # (some directed edges have reverses in adjacent cells)
            # For our 12-edge primitive cell of srs, all reverses must be present
            raise KeyError(f"Reverse of {(src, tgt, cell)} not in edge set")
    return rev


def build_jump_operators(directed):
    """
    Construct the unprojected per-edge backtrack jump operators on the
    full 12-dim directed-edge space.

    L_e = √(1/k) × |reverse(e)⟩⟨e|

    Returns a list of n complex matrices (n = 12).
    """
    n = len(directed)
    rev = reverse_edge_map(directed)
    L_list = []
    coef = math.sqrt(1.0 / K_STAR)
    for e in range(n):
        L = np.zeros((n, n), dtype=complex)
        L[rev[e], e] = coef
        L_list.append(L)
    return L_list, rev


def project_jump(L, P_R):
    """Compute the V_Ram-projected jump: P_R L P_R."""
    return P_R @ L @ P_R


def build_P_R(B_P, h_exact):
    """
    Build the projector onto the Ramanujan subspace V_Ram.

    V_Ram = span of eigenvectors of B(k_P) with eigenvalues ±h, ±h̄.
    Returns a 12×12 projector P_R.
    """
    V_basis = find_vram_basis(B_P, h_exact)  # 12 × 8 column matrix
    Q, _ = la.qr(V_basis)
    Q = Q[:, :V_basis.shape[1]]  # ensure orthonormal columns
    P_R = Q @ Q.conj().T
    return P_R, Q


# ===========================================================================
# WIGNER-WEISSKOPF / LINDBLAD CHECK
# ===========================================================================

def compute_rate_operator(L_list_proj):
    """
    Compute R = Σ_e L_e† L_e over all projected jump operators.

    For Wigner-Weisskopf decay, the visible-state survival amplitude
    decays under the non-Hermitian Hamiltonian H_eff = H - i/2 R.
    If R = γ I_R (proportional to identity on V_Ram), then γ is the
    Lindblad rate per step and survival probability after L steps
    is exp(-γ L) ≈ (1 - γ)^L for small γ.
    """
    R = sum(L.conj().T @ L for L in L_list_proj)
    return R


def lindblad_evolution(rho_0, H_R, L_list_proj, n_steps):
    """
    Evolve a density matrix under the discrete Lindblad equation:
      ρ(t+1) = U ρ(t) U† + Σ_e (L_e ρ L_e† - 1/2 {L_e†L_e, ρ})

    Where U = exp(-i H_R) is the visible Hamiltonian step.

    Returns the trajectory of the off-diagonal coherence between the
    two principal Ramanujan eigenstates.
    """
    from scipy.linalg import expm
    U = expm(-1j * H_R)

    rho = rho_0.copy()
    coherence_trajectory = [rho.copy()]

    for t in range(n_steps):
        # Coherent unitary part
        rho_new = U @ rho @ U.conj().T
        # Dissipative part
        for L in L_list_proj:
            rho_new = rho_new + L @ rho @ L.conj().T \
                              - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
        rho = rho_new
        coherence_trajectory.append(rho.copy())

    return coherence_trajectory


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    header("WIGNER-WEISSKOPF DECAY ON THE srs RAMANUJAN SUBSPACE")

    # ---- Step 1: Build B(k_P) ----
    print("Step 1: Build B(k_P) = 12×12 Bloch Hashimoto at P-point.")
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    print(f"  Found {len(bonds)} undirected bonds, {len(directed)} directed edges")
    assert len(directed) == 12, f"Expected 12 directed edges, got {len(directed)}"

    B_P = bloch_hashimoto(K_P, directed)
    print(f"  B(k_P) shape: {B_P.shape}")

    evals = la.eigvals(B_P)
    print(f"  Eigenvalues of B(k_P) (sorted by |λ|):")
    for ev in sorted(evals, key=lambda z: -abs(z)):
        print(f"    λ = {ev:.6f}   |λ| = {abs(ev):.6f}")
    print()

    # ---- Step 2-3: V_Ram and projector ----
    print("Step 2-3: Find V_Ram and build projector P_R.")
    P_R, Q_R = build_P_R(B_P, H_EXACT)
    rank_R = int(round(np.trace(P_R).real))
    print(f"  V_Ram dimension: {rank_R}")
    print(f"  Tr(P_R) = {np.trace(P_R).real:.6f}  (expected 8)")
    print(f"  ||P_R^2 - P_R|| = {la.norm(P_R @ P_R - P_R):.2e}  (idempotency)")
    print()

    # ---- Step 4: Build projected jump operators ----
    print("Step 4: Build projected jump operators L_e = P_R |ē⟩⟨e| P_R.")
    L_list_full, rev = build_jump_operators(directed)
    L_list_proj = [project_jump(L, P_R) for L in L_list_full]
    print(f"  Built {len(L_list_proj)} projected jump operators")
    print()

    # ---- Step 5: Rate operator ----
    print("Step 5: Compute rate operator R = Σ_e L_e† L_e on V_Ram.")
    R_full = compute_rate_operator(L_list_proj)

    # Restrict R to V_Ram coordinates
    R_R = Q_R.conj().T @ R_full @ Q_R
    print(f"  R restricted to V_Ram (8×8):")
    print(f"  ||R_R - diag||_F = {la.norm(R_R - np.diag(np.diag(R_R))):.6e}")
    eigvals_R = sorted(la.eigvalsh(R_R).real)
    print(f"  Eigenvalues of R_R: {[f'{v:.6f}' for v in eigvals_R]}")
    avg_rate = np.trace(R_R).real / rank_R
    print(f"  Average rate γ = Tr(R_R)/dim(V_R) = {avg_rate:.10f}")
    print(f"  Expected (if R = γ I): γ = 1/k = 1/3 = {1.0/K_STAR:.10f}")
    rate_uniform = la.norm(R_R - avg_rate * np.eye(rank_R)) < 1e-6
    print(f"  R_R proportional to I_R? {rate_uniform}")
    print()

    if rate_uniform:
        gamma = avg_rate
        print(f"  ✓ R_R = γ × I_R with γ = {gamma:.10f}")
    else:
        gamma = avg_rate
        print(f"  ✗ R_R is NOT proportional to identity.")
        print(f"    Spread of eigenvalues: max-min = "
              f"{max(eigvals_R) - min(eigvals_R):.6e}")
        print(f"    Using avg γ = {gamma:.10f} for decay estimate")

    # ---- Step 6: Coherence decay over L = g-2 = 8 steps ----
    header(f"Step 6: COHERENCE DECAY OVER L = g-2 = {G_MINUS_2} STEPS")

    print(f"  Wigner-Weisskopf survival probability: (1 - γ)^L")
    print(f"  Or equivalently:                       exp(-γ L)")
    print()

    # Initialize ρ as a coherent superposition of two distinct Ramanujan eigenstates
    # The most physical choice: two C₃-degenerate Ramanujan eigenvectors
    evals_full, evecs_full = la.eig(B_P)
    h_indices = [i for i, ev in enumerate(evals_full) if abs(ev - H_EXACT) < 1e-6]
    print(f"  Found {len(h_indices)} eigenvectors with eigenvalue h = {H_EXACT:.4f}")

    if len(h_indices) >= 2:
        v1 = evecs_full[:, h_indices[0]]
        v2 = evecs_full[:, h_indices[1]]
        v1 = v1 / la.norm(v1)
        v2 = v2 / la.norm(v2)
        # Equal superposition
        psi = (v1 + v2) / la.norm(v1 + v2)
        rho_0 = np.outer(psi, psi.conj())
        # Project to V_Ram (should be invariant since psi is in V_Ram)
        rho_0_R = P_R @ rho_0 @ P_R

        # H_R = projection of B_P onto V_Ram (Hermitian part only, since B is not Hermitian)
        # For Wigner-Weisskopf we use the Hermitian H_eff = (B + B†)/2 restricted to V_R
        H_R = P_R @ (B_P + B_P.conj().T) / 2 @ P_R

        traj = lindblad_evolution(rho_0_R, H_R, L_list_proj, G_MINUS_2)

        # Extract coherence: ⟨v1|ρ|v2⟩
        coherences = []
        for rho in traj:
            c = v1.conj() @ rho @ v2
            coherences.append(abs(c))

        c_initial = coherences[0]
        c_final = coherences[-1]
        decay_observed = c_final / c_initial if c_initial > 1e-12 else 0.0

        decay_expected_lindblad = (1 - gamma) ** G_MINUS_2
        decay_expected_alpha1 = float(ALPHA_1_BARE_EXACT)

        print(f"  Initial coherence |⟨v1|ρ_0|v2⟩| = {c_initial:.6e}")
        print(f"  After {G_MINUS_2} steps:           {c_final:.6e}")
        print(f"  Decay ratio:                  {decay_observed:.10f}")
        print()
        print(f"  Expected (1 - γ)^L with γ={gamma:.6f}: {decay_expected_lindblad:.10f}")
        print(f"  Expected α₁_bare = (2/3)^8:           {decay_expected_alpha1:.10f}")
        print()

        # Compare to alpha_1_bare
        rel_err = abs(decay_observed - decay_expected_alpha1) / decay_expected_alpha1
        print(f"  Relative error vs (2/3)^8: {rel_err*100:.4f}%")
    else:
        print(f"  WARNING: only {len(h_indices)} eigenvector(s) for h, expected ≥ 2")

    # ---- Summary ----
    header("SUMMARY")
    print(f"  α₁_bare = (2/3)^{G_MINUS_2} = {ALPHA_1_BARE_EXACT}")
    print(f"          = {float(ALPHA_1_BARE_EXACT):.10f}")
    print()
    print(f"  Lindblad rate from projected jumps: γ = {gamma:.10f}")
    print(f"  Expected (Jaynes uniform): 1/k = {1.0/K_STAR:.10f}")
    print(f"  Ratio: γ / (1/k) = {gamma * K_STAR:.6f}")
    print()
    if rate_uniform:
        print("  STATUS: Step 9.4b verified — rate operator IS proportional to identity")
        print("          on V_Ram. Wigner-Weisskopf form is exact.")
    else:
        print("  STATUS: Rate operator is NOT proportional to identity on V_Ram.")
        print("          The projection modifies the structure. Need further analysis.")
    print()


if __name__ == "__main__":
    main()
