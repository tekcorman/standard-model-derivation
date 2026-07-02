#!/usr/bin/env python3
"""(E) Spectral identification for ε_CP = 1/5 (baryon CP asymmetry, Row P28).

Existing derivation (Bayesian-toggle):
    ε_CP = (P_create − P_disrupt) / (P_create + P_disrupt)
         = (1/2 − 1/3) / (1/2 + 1/3) = (1/6)/(5/6) = 1/5

The 1/2 is the binary-toggle creation probability; 1/3 is the disruption rate
(= 1/k*). So in fact ε_CP = (1/2 − 1/k*) / (1/2 + 1/k*).

Hypothesis: ε_CP has a spectral identification on Hashimoto/adjacency at Γ:
    ε_CP = (λ_max(A) − λ_max(B)) / (λ_max(A) + λ_max(B))
         = (k* − (k*−1)) / (k* + (k*−1))
         = 1 / (2k* − 1)

For srs k*=3: ε_CP = 1/5 ✓ exact.
"""
from __future__ import annotations
import os, sys, math
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
from proofs.common import find_bonds, N_ATOMS  # noqa: E402

bonds = find_bonds()
n_bonds = len(bonds)

# Build B at Γ
B = np.zeros((n_bonds, n_bonds), dtype=complex)
for i, (src_i, tgt_i, cell_i) in enumerate(bonds):
    for j, (src_j, tgt_j, cell_j) in enumerate(bonds):
        if tgt_j == src_i and not (src_i == tgt_j and tgt_i == src_j
                                     and tuple(cell_i) == tuple(-c for c in cell_j)):
            B[i, j] = 1.0

A = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
for src, tgt, _cell in bonds:
    A[tgt, src] += 1.0

A_eigs = np.real(np.linalg.eigvalsh(A))
B_eigs = np.linalg.eigvals(B)

lam_A = max(np.real(A_eigs))
lam_B = max(np.real(B_eigs))

print("=" * 90)
print("(E) Spectral identification for ε_CP = 1/5")
print("=" * 90)

print(f"\n  λ_max(A) = {lam_A:.6f}  (= k* = 3)")
print(f"  λ_max(B) = {lam_B:.6f}  (= k*−1 = 2)")

eps_spectral = (lam_A - lam_B) / (lam_A + lam_B)
eps_bayesian = (1/2 - 1/3) / (1/2 + 1/3)
eps_target = 1/5

print(f"\n  Spectral asymmetry: (λ_A − λ_B)/(λ_A + λ_B) = (3−2)/(3+2) = 1/5")
print(f"    Computed:                                     {eps_spectral:.6f}")
print(f"  Bayesian-toggle:    (P_c − P_d)/(P_c + P_d) = (1/2 − 1/3)/(1/2 + 1/3) = 1/5")
print(f"    Computed:                                     {eps_bayesian:.6f}")
print(f"  Target ε_CP = 1/5:                              {eps_target:.6f}")

if abs(eps_spectral - eps_target) < 1e-9 and abs(eps_bayesian - eps_target) < 1e-9:
    print(f"\n  ✓ EXACT MATCH: both routes give 1/5.")

print(f"\n  General formula for k-regular graph:")
print(f"    ε_CP(k*) = (k* − (k*−1)) / (k* + (k*−1)) = 1 / (2k* − 1)")
print(f"\n  For various k*:")
print(f"    k* = 2: ε_CP = 1/3 = 0.3333")
print(f"    k* = 3: ε_CP = 1/5 = 0.2000  ← srs FRAMEWORK")
print(f"    k* = 4: ε_CP = 1/7 = 0.1429")
print(f"    k* = 5: ε_CP = 1/9 = 0.1111")
print(f"    k* = 6: ε_CP = 1/11 = 0.0909")

# Connection to marginal-sector dim
print(f"\n  CONNECTION TO MARGINAL HASHIMOTO SECTOR (5/12 derivation):")
n_marg = 5  # for srs
n_total = 12
print(f"    For srs (|V|=4, k*=3): marginal Hashimoto dim = 2(|E|−|V|)+1 = 5")
print(f"    Coincidence: 2k* − 1 = 2·3 − 1 = 5 = marginal dim")
print(f"    → For srs, ε_CP = 1/(marginal Hashimoto sector dim)")
print(f"    → 1/5 = 1/(dim of dark Q-projector)")
print(f"    This is structurally meaningful: ε_CP is the INVERSE of the")
print(f"    dark-sector dimension. As the dark sector grows (more marginal modes),")
print(f"    the per-process CP asymmetry SHRINKS.")
print(f"\n  But the equality 2k*-1 = 2(|E|-|V|)+1 (for k-regular) requires")
print(f"    |V|(k-2)+1 = 2k-1, i.e. |V|(k-2) = 2k-2, so |V| = 2(k-1)/(k-2).")
print(f"    For k=3: |V| = 4 ✓ (srs).  Coincidence specific to srs's cell.")
print(f"    For k=4: |V| = 3 (different cell).  For k=5: not integer.")

# Yet another spectral identification: ε_CP via |λ| differences
mass_perron = max(abs(e)**2 for e in B_eigs)   # 4
mass_marg_b = sum(abs(e)**2 for e in B_eigs if abs(np.imag(e)) < 1e-6
                   and abs(abs(e) - 2.0) > 1e-6)   # 5 (1+1+1+1+1)
mass_total = sum(abs(e)**2 for e in B_eigs)        # 21
print(f"\n  Alternative spectral-mass identifications (NOT clean):")
print(f"    Perron mass / total = {mass_perron/mass_total:.4f} (= 4/21)")
print(f"    Marginal mass / total = {mass_marg_b/mass_total:.4f} (= 5/21)")
print(f"    Marginal/(total+1) = {mass_marg_b/(mass_total+1):.4f} (5/22)")
print(f"  None match 1/5 directly. The Perron-asymmetry route is the canonical one.")

print(f"\n{'='*90}")
print(f"HEADLINE: ε_CP = 1/5 has a spectral identification on the Hashimoto operator")
print(f"{'='*90}")
print(f"""
  The framework's ε_CP = 1/5 (baryon CP asymmetry, Row P28) was derived via
  a Bayesian-toggle posterior update. The same number falls out of a spectral
  asymmetry computation on the substrate's Hashimoto / adjacency operators:

    ε_CP = (λ_max(A) − λ_max(B)) / (λ_max(A) + λ_max(B))
         = 1/(2k* − 1)
         = 1/5  for k* = 3

  This adds ε_CP to the list of dark/visible coefficients with spectral
  identifications:
    1. q_NB = 2/3 = λ_max(B) / λ_max(A)              [Perron ratio]
    2. α_1_bare = (2/3)^8 = q_NB^(g−2)                [cumulative survival]
    3. c = 5/12 = (2(|E|−|V|)+1) / (2|E|)             [Q-projector dim fraction]
    4. ε_CP = 1/5 = (λ_A − λ_B)/(λ_A + λ_B)           [Perron asymmetry]

  All four are different observables of the same Hashimoto / adjacency
  operator pair at Γ. The framework's dark-sector physics is reduced to
  a spectral structure on a single operator family.

  ε_CP's specific value 1/5 is the ratio of (A's and B's Perron gap) to
  (their Perron sum). Structurally, this is the "spectral asymmetry of
  Perron eigenvalues" — visible-channel speed (λ_B) vs raw walks (λ_A).

  For srs's specific cell (|V|=4, k*=3), 2k*−1 = 5 = marginal Hashimoto
  dimension, giving the additional identity ε_CP = 1/dim(Q-projector).
  This is a srs-specific coincidence (only k=3 gives |V|=4 integer
  solution to the equality).
""")
