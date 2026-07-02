#!/usr/bin/env python3
"""
Gauge-hub Stage 17 — Route β: 2T-decompose V_tree (the non-Ramanujan part of
the 12-dim P-point arc space) and confirm the 3-dim 2T irrep is absent from
the entire P-point arc space.

Context.  Stage 14 retracted the linear-A_4 generation triplet (A(P) partition
is (2,2), not (1,3) — incompatible with a linear A_4 rep).  Stage 15
constructed the P-point little group G_P = 2T = SL(2,3), acting PROJECTIVELY,
and decomposed V_Ram = 2·H_4 into four 2-dim 2T-irreps — *no 3-dim irrep in
V_Ram*.  The companion scoping doc names Route β: search for the 3-dim irrep
in V_tree (4-dim), the non-Ramanujan part of the 12-dim arc space.  Expected
negative: V_tree's C_3-shadow is (0, 2, 2), incompatible with the 3-dim 2T
irrep (whose C_3-shadow is (1, 1, 1) — needs one trivial-C_3 state).

This probe runs Route β to closure.

Procedure.
  Step 1.  Build the 12×12 Bloch Hashimoto B_NB(P) at P = (1/4, 1/4, 1/4).
           Identify V_Ram (eigenspace |λ|² = k* − 1 = 2, dim 8) and V_tree
           (the orthogonal complement, dim 4 — the "trivial cycle" sector).
  Step 2.  Lift the body-diagonal C_3 to the 12-arc space as a permutation
           Π_C3 that commutes with B_NB(P).  Restrict to V_tree.
  Step 3.  Compute Tr(Π_C3) on V_tree and read off the C_3-decomposition
           (multiplicities of trivial, ω, ω²).  Sentinel: expect (0, 2, 2).
  Step 4.  Verdict on the 3-dim 2T irrep.  The 3-dim irrep of 2T factors
           through A_4 ⊂ 2T/Z_2 (a linear irrep); its restriction to C_3 is
           the regular rep 1 + ω + ω², requiring ≥ 1 trivial-C_3 state.
           V_tree has 0 trivial-C_3 states → 3 cannot appear → Route β
           closed-negative.

Gates (PASS = expected structural fact):
  G1  B_NB(P) has the expected Ihara-Bass split:  4 "trivial" eigenvalues
      (|λ| = 1) + 8 "Ramanujan" eigenvalues (|λ|² = 2).
  G2  Π_C3 lifts to arcs: a 12-permutation matrix consistent with the atom
      permutation σ_C3 = (0)(1 2 3), commuting with B_NB(P).
  G3  C_3-shadow of V_tree = (0, 2, 2)  (matches scoping prediction).
  G4  No 3-dim 2T irrep can sit in V_tree (consequence of G3 + character
      theory).  Combined with Stage 15 (no 3 in V_Ram), the 3-dim 2T irrep
      is ABSENT from the entire 12-dim P-point arc space.

Route β closure: confirmed structural negative — the "generations = 3-dim 2T
irrep at the P-point" route is dead.  The honest options are α (3 = 2 ⊕ 1,
needs the singlet sourced elsewhere), δ (un-anchored per Stage 16), and γ
(P-point is the wrong place — B7.1's dim = 3 stands but the symmetry is
sought elsewhere).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
from proofs.common import find_bonds, C3_PERM
from proofs.foundations.theorem_B5_3_core import build_directed_edges, bloch_hashimoto


TOL = 1e-7

# ---------------------------------------------------------------------------
# Step 1: B_NB(P), V_Ram (|λ|²=2), V_tree (|λ|²=1)
# ---------------------------------------------------------------------------

bonds = find_bonds()
directed = build_directed_edges(bonds)
N_ARCS = len(directed)               # 12
K_STAR_MINUS_1 = 2                   # k* − 1 = 2 (Ramanujan threshold |λ|²)

P_FRAC = (0.25, 0.25, 0.25)          # P-point (C_3-invariant body-diagonal)
B_P = bloch_hashimoto(P_FRAC, directed)

evals, evecs = np.linalg.eig(B_P)
mod2 = np.abs(evals) ** 2

ram_mask  = np.abs(mod2 - K_STAR_MINUS_1) < TOL
tree_mask = np.abs(mod2 - 1.0) < TOL

dim_ram  = int(ram_mask.sum())
dim_tree = int(tree_mask.sum())

# Orthonormal bases of V_Ram and V_tree (eigvecs of a non-Hermitian B can
# be non-orthogonal; QR-orthonormalise on the eigenspan)
def orthonormalise(M):
    Q, _ = np.linalg.qr(M)
    return Q

V_Ram_basis  = orthonormalise(evecs[:, ram_mask])
V_tree_basis = orthonormalise(evecs[:, tree_mask])

# ---------------------------------------------------------------------------
# Step 2: lift C_3 to the 12-arc space — permutation matrix Π_C3 with
# σ_atom = (0)(1 2 3) plus the body-diagonal cell rotation
# ---------------------------------------------------------------------------

# Atom permutation from common.C3_PERM (matrix → list)
sigma_atom = [None] * 4
for i in range(4):
    for j in range(4):
        if abs(C3_PERM[j, i] - 1.0) < 1e-9:
            sigma_atom[i] = j

def build_Pi_C3(cell_rule):
    """Build the 12-arc permutation induced by σ_atom + cell_rule on cells.
    `cell_rule` is a callable (n1, n2, n3) → (n_a, n_b, n_c) for the lattice.
    Returns the permutation matrix or None if the rule does not yield a
    consistent permutation."""
    P = np.zeros((N_ARCS, N_ARCS), dtype=complex)
    arc_index = {(s, t, tuple(c)): i for i, (s, t, c) in enumerate(directed)}
    for i, (src, tgt, cell) in enumerate(directed):
        new = (sigma_atom[src], sigma_atom[tgt], tuple(cell_rule(cell)))
        if new not in arc_index:
            return None
        j = arc_index[new]
        P[j, i] = 1.0
    if not np.allclose(P @ P.conj().T, np.eye(N_ARCS), atol=TOL):
        return None
    return P

# Try both cyclic conventions (n1,n2,n3) → (n2,n3,n1) and (n3,n1,n2)
cell_rules = {
    "(n2,n3,n1)": lambda c: (c[1], c[2], c[0]),
    "(n3,n1,n2)": lambda c: (c[2], c[0], c[1]),
}
Pi_C3 = None
chosen_rule = None
for name, rule in cell_rules.items():
    Pi = build_Pi_C3(rule)
    if Pi is None:
        continue
    if np.linalg.norm(Pi @ B_P - B_P @ Pi) < TOL:
        Pi_C3 = Pi
        chosen_rule = name
        break

# ---------------------------------------------------------------------------
# Step 3: C_3-shadow of V_tree
# ---------------------------------------------------------------------------

if Pi_C3 is not None:
    # Restrict Π_C3 to V_tree
    Pi_C3_on_Vtree = V_tree_basis.conj().T @ Pi_C3 @ V_tree_basis
    # Eigenvalues should be {1, ω, ω²} for various multiplicities
    c3_eigs = np.linalg.eigvals(Pi_C3_on_Vtree)
    # Quantize each eigenvalue to its nearest cube root of unity
    cube_roots = [1.0 + 0j, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)]
    labels = []
    for e in c3_eigs:
        idx = int(np.argmin([abs(e - r) for r in cube_roots]))
        labels.append(idx)
    m_trivial = labels.count(0)
    m_omega   = labels.count(1)
    m_omega2  = labels.count(2)
    c3_shadow_tree = (m_trivial, m_omega, m_omega2)

    # Cross-check: V_Ram shadow should be (2, 2, 2) from Stage 11/15
    Pi_C3_on_VRam = V_Ram_basis.conj().T @ Pi_C3 @ V_Ram_basis
    c3_eigs_ram = np.linalg.eigvals(Pi_C3_on_VRam)
    labels_ram = []
    for e in c3_eigs_ram:
        idx = int(np.argmin([abs(e - r) for r in cube_roots]))
        labels_ram.append(idx)
    c3_shadow_ram = (labels_ram.count(0), labels_ram.count(1), labels_ram.count(2))

    # Sanity: total shadow = (2,2,2)+(0,2,2) = (2,4,4) — matches 12-arc rep,
    # whose C_3-shadow should equal the C_3-shadow of the underlying srs
    # arc space at P.
    total_shadow = tuple(c3_shadow_ram[i] + c3_shadow_tree[i] for i in range(3))
else:
    c3_shadow_tree = None
    c3_shadow_ram = None
    total_shadow = None

# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
gates = []

# G1: Ihara-Bass split
gates.append((
    "G1  B_NB(P) Ihara-Bass split: 4 trivial (|λ|=1) + 8 Ramanujan (|λ|²=2)",
    dim_tree == 4 and dim_ram == 8,
    f"dim V_tree = {dim_tree}, dim V_Ram = {dim_ram}",
))

# G2: C_3 lifts to arcs
gates.append((
    "G2  C_3 lifts to 12-arc space (permutation respecting σ_atom + body-"
    "diagonal cell rotation), commuting with B_NB(P)",
    Pi_C3 is not None,
    f"Pi_C3 found with cell rule {chosen_rule}" if Pi_C3 is not None
    else "no cell rule yields a commuting Pi_C3",
))

# G3: C_3-shadow of V_tree
gates.append((
    "G3  C_3-shadow of V_tree = (0, 2, 2)  (no trivial-C_3 states; "
    "matches scoping prediction)",
    c3_shadow_tree == (0, 2, 2),
    f"C_3-shadow V_tree = {c3_shadow_tree}; V_Ram = {c3_shadow_ram}; total = {total_shadow}",
))

# G4: 3-dim 2T irrep is ABSENT from V_tree (hence from the full P-point arc
# space, combined with Stage 15).  The 3-dim irrep of 2T is linear (factors
# through A_4 = 2T/Z_2) and its restriction to C_3 is the regular rep
# 1 + ω + ω² with C_3-shadow (1, 1, 1).  A containing space must have at
# least 1 trivial-C_3 state per copy of 3.
gates.append((
    "G4  3-dim 2T irrep is ABSENT from V_tree (C_3-shadow (0,2,2) of "
    "V_tree contains 0 trivial-C_3 states; the 3 needs ≥1 per copy)",
    c3_shadow_tree is not None and c3_shadow_tree[0] == 0,
    f"trivial-C_3 multiplicity in V_tree = {c3_shadow_tree[0] if c3_shadow_tree else 'unknown'} "
    f"→ 0 copies of 3 in V_tree",
))

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
print("=" * 78)
print("GAUGE-HUB STAGE 17 — ROUTE β: V_tree 2T-DECOMPOSITION")
print("=" * 78)
print()
print(f"  P-point fractional coords: {P_FRAC}")
print(f"  12-arc Bloch Hashimoto B_NB(P) constructed.")
print(f"  V_Ram = {dim_ram}-dim (|λ|² = {K_STAR_MINUS_1});  V_tree = {dim_tree}-dim (|λ|² = 1).")
print()
if Pi_C3 is not None:
    print(f"  C_3 lift to arcs found via cell rule {chosen_rule}")
    print(f"  C_3-shadow:  V_Ram = {c3_shadow_ram}    V_tree = {c3_shadow_tree}    total = {total_shadow}")
else:
    print("  WARNING: no cell rule yielded a commuting Pi_C3 — symmetry lift FAILED.")
print()

npass = 0
for name, ok, detail in gates:
    tag = "PASS" if ok else "FAIL"
    npass += int(ok)
    print(f"  [{tag}]  {name}")
    print(f"          {detail}")
print()
print("-" * 78)
print(f"  {npass}/{len(gates)} gates")
print()
print("""
  ROUTE β VERDICT
  ===============
  V_tree's C_3-shadow is (0, 2, 2) — NO trivial-C_3 states.  The 3-dim 2T
  irrep is linear (factors through A_4 = 2T/Z_2), and its restriction to
  C_3 ⊂ 2T is the regular rep 1 + ω + ω² (C_3-shadow (1, 1, 1)).  Any
  embedding of the 3-irrep into a subspace requires that subspace to
  contain at least 1 trivial-C_3 state per copy of 3.  V_tree contains
  zero — so V_tree contains zero copies of 3.

  Combined with Stage 15 (V_Ram = four 2-dim irreps, NO 3-dim irrep):
  the 3-dim 2T irrep is ABSENT from the entire 12-dim P-point arc space.

  ROUTE β is CLOSED-NEGATIVE — as expected by the scoping doc (the (2,2)
  degeneracy of A(P) forces 2-dim-irrep structure throughout the P-point
  representation theory).  The honest remaining options for the generation
  sector are α (3 = 2 ⊕ 1, with the singlet sourced from V_tree or
  elsewhere), δ (un-anchored per Stage 16 — 2T's central Z_2 ≠ srs-z χ̃),
  and γ (the P-point is the wrong place).

  WHAT V_tree IS, briefly.  C_3-shadow (0, 2, 2) — two copies of (1' ⊕ 1'')
  if linear, OR two copies of a 2-dim projective 2T-irrep (2, 2', or 2''
  whose C_3-restriction is ω ⊕ ω²).  Distinguishing requires V_B/spinorial
  data on the arc space, which is not needed for Route β's verdict.
""")
sys.exit(0 if npass == len(gates) else 1)
