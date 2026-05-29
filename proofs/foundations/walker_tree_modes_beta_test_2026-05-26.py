#!/usr/bin/env python3
"""
Walker tree modes as candidate β contribution — test.

CONTEXT (returning to β issue post T1 closure):
  Hashimoto B(P) on srs primitive cell has 12-dim spectrum:
    - 8 Ramanujan eigenvalues (|λ|² = 2) — V_Ram, used for gauge content
    - 4 tree eigenvalues (±1 each mult 2) — currently "trivial gauge content"

  HYPOTHESIS (Reading 3 from prior analysis): the 4 tree modes might
  contribute to β coefficients as SCALAR-LIKE walker modes — providing
  the (1/3)·T scalar-partner content that MSSM β needs without literal
  SUSY partners.

QUESTION:
  1. What is the C_3 (and SU(4)_PS, if available) decomposition of the
     4-dim tree subspace?
  2. Under what attribution do tree modes contribute to β?
  3. Does this attribution close the SM→MSSM β gap (Δb_1 = 2.4, Δb_2 = 4,
     Δb_3 = 4)?

PROBE STRATEGY:
  Step 1: Extract 4-dim tree subspace from B(P) at P-point
  Step 2: Compute σ-action on tree subspace → C_3 isotypes
  Step 3: Test β-contribution attributions
  Step 4: Report which (if any) attribution closes the MSSM gap

This is bounded and falsifiable.
"""

import sys
import os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

TOL = 1e-9

# ============================================================
# 1. BUILD B(P), EXTRACT TREE SUBSPACE (complement of V_Ram)
# ============================================================
from proofs.common import find_bonds, K_STAR

bonds = find_bonds()
N_arcs = len(bonds)

P_POINT = np.array([1/4, 1/4, 1/4])

def build_BNB(arc_list, k_frac):
    n = len(arc_list)
    M = np.zeros((n, n), dtype=complex)
    for j, (sj, tj, cj) in enumerate(arc_list):
        for i, (si, ti, ci) in enumerate(arc_list):
            if sj != ti:
                continue
            dc = tuple(int(ci[d]) + int(cj[d]) for d in range(3))
            if tj == si and dc == (0, 0, 0):
                continue
            M[j, i] = np.exp(2j * np.pi * np.dot(k_frac, ci))
    return M


B_P = build_BNB(bonds, P_POINT)
eigs_BP, vecs_BP = np.linalg.eig(B_P)
eig_mags2 = np.abs(eigs_BP)**2

# Tree subspace: |λ|² ≈ 1 (4 eigenvalues at ±1)
tree_mask = np.abs(eig_mags2 - 1.0) < TOL
ramanujan_mask = np.abs(eig_mags2 - 2.0) < TOL
n_tree = int(tree_mask.sum())
n_ram = int(ramanujan_mask.sum())

print(f"  B(P) spectrum:")
print(f"    Tree modes (|λ|²=1):      {n_tree} eigenvalues")
print(f"    Ramanujan modes (|λ|²=2): {n_ram} eigenvalues")
tree_eigs_vals = eigs_BP[tree_mask].tolist()
print(f"    Tree eigenvalues: {sorted(tree_eigs_vals, key=lambda z: (z.real, z.imag))}")

V_tree_raw = vecs_BP[:, tree_mask]
Q_tree, _ = np.linalg.qr(V_tree_raw)
V_tree_basis = Q_tree[:, :n_tree]   # 12×4 orthonormal


# ============================================================
# 2. σ-ACTION ON TREE SUBSPACE → C_3 ISOTYPES
# ============================================================
sigma_vertex_map = {0: 0, 1: 3, 2: 1, 3: 2}
def sigma_cell(c): return (c[2], c[0], c[1])

def sigma_arc_perm(arc_list):
    n = len(arc_list)
    P = np.zeros((n, n), dtype=complex)
    for i, (s, t, c) in enumerate(arc_list):
        sigma_arc = (sigma_vertex_map[s], sigma_vertex_map[t], sigma_cell(c))
        j = arc_list.index(sigma_arc)
        P[j, i] = 1.0
    return P


U_sigma_arcs = sigma_arc_perm(bonds)
U_sigma_tree = V_tree_basis.conj().T @ U_sigma_arcs @ V_tree_basis

# Verify σ³ = I on tree subspace
sigma_cubed_tree = U_sigma_tree @ U_sigma_tree @ U_sigma_tree
assert np.allclose(sigma_cubed_tree, np.eye(n_tree), atol=1e-7), \
    f"σ³ ≠ I on tree subspace: max|·-I| = {np.max(np.abs(sigma_cubed_tree - np.eye(n_tree)))}"

# Compute C_3 isotypic decomposition of tree subspace
tree_eigs = np.linalg.eigvals(U_sigma_tree)
omega = np.exp(2j * np.pi / 3)
omega_bar = np.exp(-2j * np.pi / 3)

def classify(z):
    if abs(z - 1) < 1e-5: return 'trivial'
    if abs(z - omega) < 1e-5: return 'omega'
    if abs(z - omega_bar) < 1e-5: return 'omega_bar'
    return f'other({z:.4f})'

tree_iso = Counter(classify(z) for z in tree_eigs)
print(f"\n  Tree subspace C_3 isotypic decomposition: {dict(tree_iso)}")


# ============================================================
# 3. β-COEFFICIENT TEST UNDER ATTRIBUTION HYPOTHESES
# ============================================================
# SM baseline (3 SM gens + 2 Higgs doublets, no partners):
b1_SM = 41.0/10.0   # GUT-normalized
b2_SM = -19.0/6.0
b3_SM = -7.0

# MSSM target:
b1_MSSM = 33.0/5.0
b2_MSSM = 1.0
b3_MSSM = -3.0

# Gap to close: b_MSSM - b_SM (or 2HDM)
b1_2HDM = 21.0/5.0
b2_2HDM = -3.0
b3_2HDM = -7.0

gap_b1 = b1_MSSM - b1_2HDM
gap_b2 = b2_MSSM - b2_2HDM
gap_b3 = b3_MSSM - b3_2HDM

print(f"\n  β coefficient gap (MSSM - 2HDM): Δb_1 = {gap_b1}, Δb_2 = {gap_b2}, Δb_3 = {gap_b3}")
print(f"    Need contribution: (Δb_1, Δb_2, Δb_3) = ({gap_b1:.3f}, {gap_b2:.3f}, {gap_b3:.3f})")

# Hypothesis 1: tree modes are SU(4) singlets (no gauge representation)
print(f"\n  HYPOTHESIS 1: Tree modes are SU(4) singlets")
print(f"    Contribution to β: 0 (no gauge coupling)")
print(f"    Result: β unchanged from 2HDM → catastrophic")

# Hypothesis 2: tree modes are 4 scalar partners with specific gauge reps
# Suppose 4 tree modes carry T(R) = 1/2 each in some natural gauge rep
# This is highly model-dependent. Various attributions possible:

# H2a: 4 modes act as 4 squark/slepton-like complex scalars per generation
# For 3 generations: 12 total scalar contributions, T(R)_total per gen = 2
# Contribution to β: (1/3) · 12 ... but this doesn't match the count

# H2b: 4 tree modes = 4 "gauge-mode partners" = singlets for U(1), in fund of SU(2), SU(3)?
# Hard to motivate

# H2c: 4 tree modes = 3 gauge-multiplet shadows + 1 singlet
# 3 gauginos (Weyl in adjoint), each contributing (2/3)·C_2(G)
# For SU(3): (2/3) × 3 = 2 → close to MSSM Δb_3 = 4? Not quite enough
# For SU(2): (2/3) × 2 = 4/3 ≈ 1.33 → close to MSSM Δb_2 = 4? No

# Even with optimistic gaugino-equivalent interpretation:
# Maximum β contribution from 4 modes acting as gauginos in adjoint of SU(3):
# 4 × (2/3) × 3 = 8 — but this is per cell, and contributes to b_3 only
# Doesn't match Δb_3 = 4 (would overshoot)

# Realistic assessment:
# - 4 tree modes don't naturally give 6 chiral supermultiplet contributions
# - The count 4 doesn't match the 6 (gauginos + Higgsinos = 12 + 4 = ~16 in MSSM)
# - Tree modes can't naturally fill the partner-content gap

# Even if we attempt: 4 tree modes ↦ 4 sfermion equivalents (1/3·T each)
# For 3 SM gens of matter, T_SU(3) = 6 (per-gen sum × 3), so 1/3 × 6 = 2 to b_3
# That gives b_3 = -7 + 2 = -5, NOT -3 (MSSM).
# To get -3 we'd need additional (gaugino-like) contribution of +2 to b_3

# So under any attribution of 4 tree modes as scalars, b_3 can rise to -5 max.
# To reach -3 we'd need ALSO ~2-equivalent gaugino content. 4 modes total <
# enough for both sfermion + gaugino contributions.


# ============================================================
# 4. β CONTRIBUTION CHECK — most generous attribution
# ============================================================
def beta_with_tree_attribution(scalar_T_3, scalar_T_2, scalar_T_1,
                                fermion_T_3, fermion_T_2, fermion_T_1):
    """β contribution from tree modes interpreted as
    scalars (T_S) + extra fermions (T_F) in given gauge reps."""
    # On top of 2HDM baseline:
    b_3 = b3_2HDM + (1/3)*scalar_T_3 + (2/3)*fermion_T_3
    b_2 = b2_2HDM + (1/3)*scalar_T_2 + (2/3)*fermion_T_2
    b_1 = b1_2HDM + (1/3)*scalar_T_1 + (2/3)*fermion_T_1
    return (b_1, b_2, b_3)


# Most-generous attribution: 4 tree modes split into 2 scalar + 2 fermion contributions
# Try a few natural splits
scenarios = [
    ("All 4 tree modes as singlets (H1)",   (0, 0, 0), (0, 0, 0)),
    ("All 4 modes as scalars in 3 SM gen partner reps",
        (6, 6, 6), (0, 0, 0)),
    ("Half (2 modes) as scalars + half as fermions in 3 gen reps",
        (3, 3, 3), (3, 3, 3)),
    ("All 4 as gauginos (adjoint of SU(3) only)",
        (0, 0, 0), (3, 0, 0)),
    ("Optimistic MSSM-mimicking: scalars + gauginos + Higgsinos at MSSM counts",
        (6, 7, 33/5 - 4), (3, 4, 0)),
]

print(f"\n  ATTRIBUTION SCENARIOS (b_1, b_2, b_3 results):")
print(f"    {'Scenario':<60} {'b_1':>8} {'b_2':>8} {'b_3':>8} {'|Δ_MSSM|':>10}")
print(f"    {'MSSM target':<60} {b1_MSSM:>8.3f} {b2_MSSM:>8.3f} {b3_MSSM:>8.3f}     —")
for name, scalars, fermions in scenarios:
    b = beta_with_tree_attribution(*scalars, *fermions)
    total_dev = abs(b[0] - b1_MSSM) + abs(b[1] - b2_MSSM) + abs(b[2] - b3_MSSM)
    print(f"    {name:<60} {b[0]:>8.3f} {b[1]:>8.3f} {b[2]:>8.3f} {total_dev:>10.3f}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 78)
print("  VERDICT — Walker tree modes as β-contribution candidates")
print("=" * 78)
print(f"""
  STRUCTURAL FACT (verified by this probe):
    Tree subspace of B(P) has dim 4, with C_3 isotypic decomposition
    {dict(tree_iso)}.

  CRITICAL ISSUE FOR β:
    The 4 tree modes are NOT ENOUGH to provide the full MSSM partner
    content. MSSM requires:
      - 48 sfermion scalars (3 gens × 16)
      - 12 gauginos
      - 4 Higgsinos
    Total: ~64 partner states. Tree subspace has only 4 modes per cell.

    Even with maximally generous attribution (treating all 4 tree modes
    as multiple gauge representations simultaneously), the maximum
    β-contribution per cell is far short of MSSM's full partner content.

  KEY MISMATCH:
    Per-cell tree modes: 4 (an EXTENSIVE count — scales with cell size)
    Per-generation MSSM partners: ~20 fields (an INTENSIVE count)

    For 3 generations across N cells, we'd need ~60 partner equivalents.
    Tree modes give 4N (linear in cells), MSSM needs ~60 (fixed by gens).

  CONCLUSION:
    Walker tree modes CANNOT supply MSSM β-coefficient contributions.
    The count mismatch is fundamental: 4 tree modes per cell vs ~60
    MSSM partner equivalents per 3-generation set. Even with full
    cell-extensive counting, the structural attribution of tree modes
    to specific gauge representations would require additional input
    not in the framework's current axioms.

  WHAT THIS RULES OUT:
    "Walker tree resonances" as a non-MSSM β mechanism is closed-as-
    negative. The framework's gauge β coefficients cannot reach MSSM
    values via this mechanism alone.

  WHAT REMAINS:
    - Composite particle scenarios (preons, technicolor): require
      multi-session research, structurally different from current framework
    - Higher-derivative gauge effects from full Hashimoto dynamics:
      speculative, hard to probe
    - Non-perturbative substrate effects: speculative

  HONEST READING:
    The β coefficient gap (SM → MSSM) is genuinely structural in the
    framework. ADOPTED-MSSM-Sb remains the settled endpoint. Walker
    tree modes don't bridge the gap; future β work would need to
    explore mechanisms outside the current substrate axioms.
""")
print("=" * 78)
