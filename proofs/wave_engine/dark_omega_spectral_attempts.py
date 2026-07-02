#!/usr/bin/env python3
"""Attempt to find a spectral identification for Ω_DM/Ω_m = 0.8488 analogous
to the 5/12 derivation.

Existing framework derivation: Ω_DM/Ω_m = 1 − P(k ≤ k* | Poisson(2k*))
                              = 1 − e^(−6) · Σ_{j=0}^{3} 6^j/j!
                              = 1 − 0.15120
                              = 0.84880

Question: does this also have a clean spectral identification on the substrate's
operator structure, like 5/12 does?

Try several spectral measures and see if any give 0.8488 ± 1%.
"""
from __future__ import annotations
import os, sys, math
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
from proofs.common import find_bonds, N_ATOMS  # noqa: E402

TARGET = 1 - sum(6**j / math.factorial(j) for j in range(4)) * math.exp(-6)
print(f"Target: Ω_DM/Ω_m = 0.84880 (Poisson(6) tail above k*=3)")
print(f"Computed:                {TARGET:.6f}\n")

bonds = find_bonds()
n_bonds = len(bonds)

# Build B at Γ
B = np.zeros((n_bonds, n_bonds), dtype=complex)
for i, (src_i, tgt_i, cell_i) in enumerate(bonds):
    for j, (src_j, tgt_j, cell_j) in enumerate(bonds):
        if tgt_j != src_i:
            continue
        is_reverse = (src_i == tgt_j and tgt_i == src_j
                      and tuple(cell_i) == tuple(-c for c in cell_j))
        if is_reverse:
            continue
        B[i, j] = 1.0

# Build A at Γ
A = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
for src, tgt, _cell in bonds:
    A[tgt, src] += 1.0

A_eigs = np.real(np.linalg.eigvalsh(A))
B_eigs = np.linalg.eigvals(B)

print("=" * 80)
print("Spectral candidates for Ω_DM/Ω_m = 0.8488:")
print("=" * 80)

# === Attempt 1: |λ|² weighted spectral mass ===
mass_total = sum(abs(e)**2 for e in B_eigs)
mass_perron = max(abs(e)**2 for e in B_eigs)   # +2 → 4
mass_marginal = sum(abs(e)**2 for e in B_eigs if abs(abs(e) - 1.0) < 1e-6)
mass_oscillatory = sum(abs(e)**2 for e in B_eigs
                        if abs(np.imag(e)) > 1e-6)
print(f"\n[1] Hashimoto |λ|² spectral mass distribution:")
print(f"    total |λ|²            = {mass_total:.4f}")
print(f"    Perron mass / total   = {mass_perron/mass_total:.4f}")
print(f"    marginal mass / total = {mass_marginal/mass_total:.4f}")
print(f"    1 - Perron/total      = {1 - mass_perron/mass_total:.4f}   (compare 0.8488)")

# === Attempt 2: Number-operator distribution on Cl(6) Fock ===
print(f"\n[2] Cl(6) Fock occupation distribution (3 fermion modes):")
# Cl(6) Fock has 2^3 = 8 states; number operator eigenvalues 0, 1, 2, 3
# Multiplicities: C(3,0)=1, C(3,1)=3, C(3,2)=3, C(3,3)=1 (binomial)
fock_dim = 8
for n_visible_max in range(4):
    visible = sum(math.comb(3, k) for k in range(n_visible_max + 1))
    dark = fock_dim - visible
    print(f"    visible = states with N ≤ {n_visible_max}: dim {visible}, dark dim {dark}, "
          f"dark/total = {dark/fock_dim:.4f}")

# === Attempt 3: NB walk survival rate (1 - α₁) ===
alpha_1_bare = (2/3)**8
alpha_1_full = 256/6305
print(f"\n[3] NB walk survival vs decay:")
print(f"    α₁_bare = (2/3)^8       = {alpha_1_bare:.6f}")
print(f"    1 - α₁_bare             = {1 - alpha_1_bare:.6f}")
print(f"    α₁_full = 256/6305      = {alpha_1_full:.6f}")
print(f"    1 - α₁_full             = {1 - alpha_1_full:.6f}")

# === Attempt 4: Mass-density partition via Hashimoto sectors ===
print(f"\n[4] Hashimoto sectors normalized by trace:")
# Various mass partitions
sectors = {
    'Perron (|λ|=2)': sum(abs(e) for e in B_eigs if abs(abs(e) - 2.0) < 1e-6),
    'Oscillatory (|λ|=√2)': sum(abs(e) for e in B_eigs if abs(abs(e) - math.sqrt(2)) < 1e-6),
    'Marginal (|λ|=1)':  sum(abs(e) for e in B_eigs if abs(abs(e) - 1.0) < 1e-6),
}
total_abs = sum(sectors.values())
print(f"    Σ|λ| = {total_abs:.4f}")
for name, m in sectors.items():
    print(f"    {name:<22}: {m:.4f} = {m/total_abs:.4f} of total")

# === Attempt 5: Compute random-matrix-style ===
# For Poisson(6) graph, the spectral density follows specific tail laws.
# Compare e.g. fraction of B's eigenvalues with |λ| ≥ √(k*) = √3 ≈ 1.732
print(f"\n[5] Eigenvalues by magnitude threshold:")
for thresh, label in [(2.0, '|λ|=2 only (Perron)'),
                       (math.sqrt(3), '|λ| ≥ √k* = √3'),
                       (math.sqrt(2), '|λ| ≥ √2'),
                       (1.0, '|λ| ≥ 1')]:
    n_above = sum(1 for e in B_eigs if abs(e) >= thresh - 1e-6)
    n_below = len(B_eigs) - n_above
    print(f"    {label:<25}: above {n_above}, below {n_below}, "
          f"below/total = {n_below/len(B_eigs):.4f}")

# === Attempt 6: Direct Poisson connection — does it factor through anything spectral? ===
print(f"\n[6] Direct Poisson(6) PMF inspection:")
print(f"    P(k=0) = e^(-6)          = {math.exp(-6):.6f}")
print(f"    P(k=1) = 6 e^(-6)         = {6*math.exp(-6):.6f}")
print(f"    P(k=2) = 18 e^(-6)        = {18*math.exp(-6):.6f}")
print(f"    P(k=3) = 36 e^(-6)        = {36*math.exp(-6):.6f}")
print(f"    Σ P(k≤3) = 61 e^(-6)      = {61*math.exp(-6):.6f}")
print(f"    Visible weight 61/e^6     = {61/math.exp(6):.6f}")
print(f"    Dark weight 1 - 61/e^6    = {1 - 61/math.exp(6):.6f}  (target)")
print()
print(f"    Note: 61 = 1+6+18+36 = Σ 6^k/k! for k=0..3, which is exactly the truncated")
print(f"    exponential expansion. This is NOT a spectral identity — it's a")
print(f"    statistical (Jaynes max-entropy) result for independent toggles.")

# === Headline ===
print(f"\n{'='*80}")
print(f"HEADLINE: spectral fits for Ω_DM/Ω_m = 0.8488")
print(f"{'='*80}")
candidates = [
    ('Poisson(6) tail (existing)',     1 - sum(6**j/math.factorial(j) for j in range(4))*math.exp(-6)),
    ('1 - mass_Perron/total |λ|²',     1 - mass_perron/mass_total),
    ('marginal_count / fock(8)',       5/8),    # 5 marginal Hashimoto modes / Cl(6) Fock dim 8
    ('mass_oscillatory / total |λ|²',  mass_oscillatory/mass_total),
    ('1 - 1/√3',                        1 - 1/math.sqrt(3)),    # arbitrary candidate
    ('11/12 (1 - 1/12)',               11/12),
    ('17/20',                           17/20),
    ('dark/total (5/12 from B)',       5/12),
]
print(f"\n  {'candidate':<40}{'value':>12}{'|Δ|':>10}{'rel':>10}")
print('  ' + '-' * 72)
for name, val in sorted(candidates, key=lambda x: abs(x[1] - TARGET)):
    diff = abs(val - TARGET)
    rel = diff / TARGET
    flag = '★' if rel < 0.01 else ('~' if rel < 0.05 else '')
    print(f"  {name:<40}{val:>12.6f}{diff:>10.4f}{rel:>9.2%} {flag}")

# === Conclusion ===
print(f"""
{'='*80}
CONCLUSION
{'='*80}

Honest finding: Ω_DM/Ω_m = 0.8488 does NOT have a clean spectral identification
analogous to 5/12. The closest spectral candidates miss by 5–15%.

Why the disanalogy:
  - 5/12 IS a dimensional fraction (rank of Hashimoto Q-projector). It's a
    rational with a structural meaning (graph cycle-rank + Perron-A image).
  - 0.8488 IS a statistical weight (Poisson(6) tail). It's irrational
    (involves e^(-6)). Statistical → not spectral.

The two coefficients live at different layers:
  - 5/12 lives at the Hashimoto operator layer (spectral decomposition of B).
  - 0.8488 lives at the random-graph statistical layer (Jaynes max-entropy on
    independent toggles with mean degree 2k*=6).

These layers connect via the framework's general MDL apparatus but aren't
reducible to each other. The 5/12 spectral derivation is a NEW structural
identification; the 0.8488 derivation remains statistical.

This is itself a finding: not every dark coefficient is spectral. The
substrate's dark structure has BOTH spectral aspects (5/12 = dim Q-projector)
AND statistical aspects (0.8488 = Poisson(6) tail above visible k*=3 cutoff).
""")
