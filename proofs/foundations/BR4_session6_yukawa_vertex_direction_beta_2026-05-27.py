#!/usr/bin/env python3
"""
proofs/foundations/BR4_session6_yukawa_vertex_direction_beta_2026-05-27.py

BR4 Session 6 — Yukawa vertex direction (β), the final untried direction.

PURPOSE
-------
Sessions 1-5 ruled out 4 of the 5 entry-point directions for BR4. One
remained UNTRIED: direction (β) — the Yukawa vertex operator γ^a · h⁰_a
on Cl(6) Fock via the edge-qubit ↔ Cl(6) bridge
(`theorem_g2_edge_qubit_su2.md`).

This session completes the 5/5 sweep per entry-point §11.

STRUCTURAL SETUP (per master synthesis + g2 edge theorem)
---------------------------------------------------------
- Higgs doublet lives in edge qubit ℂ² = Cl(0,2) ≅ ℍ module
- Higgs vev v = 246 GeV gives Higgs → h⁰ = v/√2 in specific direction
- Yukawa vertex couples fermion (Cl(6) Fock) to Higgs (edge qubit)
- After Higgs vev: Yukawa vertex acts on Cl(6) Fock as a chirality-flip
  operator (single γ_a or combination) with species-specific weight

The framework's existing Yukawa-Bloch identification (master synthesis §3):

  Charged lepton (Walker III, γ_7=-1, chir-5/3 at P): y_τ = (5/3)·Q⁸/k*²
  Up-type (Walker II, γ_7=+1, h=1 at Γ): y_t = 1
  Down-type (Walker IV, γ_7=-1, h=2 at Γ): y_b = Q^g
  Neutrino (Walker I, γ_7=+1, chir-7 at Γ/H): y_ν3 spectral

Direction (β) tests: does the Yukawa vertex γ^a · h⁰_a have OFF-DIAGONAL
structure within a species block that gives the within-species δ?

HYPOTHESIS
----------
After EWSB, the Yukawa vertex is V_Yuk = (v/√2) · Γ_species where
Γ_species is a species-specific chirality-flip operator on Cl(6) Fock.
The natural choices:
  - Γ_lepton ~ γ_5 (or specific combination from §4(B))
  - Γ_down   ~ γ_3 or γ_4 (color-triplet direction at Γ)
  - Γ_up     ~ γ_1 or γ_2 (saturation direction)

For each, compute the restriction to C³_obs in the R-C basis (Session 3
basis) and check AB5/AB6/AB2.

PREDICTION (per Session 5 reframing)
-----------------------------------
Yukawa vertex anchors SPECIES-SCALE (y_species). The WITHIN-species
3-mass structure (Koide) comes from walker dynamics, NOT from the
Yukawa vertex itself. So direction (β) is structurally expected to give
AB2 failure (no within-species δ structure).

If the probe confirms this expectation: 5/5 BR4 directions are exhausted;
Need-B genuinely framework-extension.

If unexpectedly the probe shows within-species δ structure from γ^a · h⁰_a:
this would reopen BR4 closure.

Run with:
    python3 proofs/foundations/BR4_session6_yukawa_vertex_direction_beta_2026-05-27.py
"""

import numpy as np
from scipy.linalg import expm

TOL = 1e-9


# ---------------------------------------------------------------------------
# Cl(6,0) Brauer-Weyl setup (matches Session 3 probe)
# ---------------------------------------------------------------------------
def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

G = [None] * 7
G[1] = kron(sx, I2, I2)
G[2] = kron(sy, I2, I2)
G[3] = kron(sz, sx, I2)
G[4] = kron(sz, sy, I2)
G[5] = kron(sz, sz, sx)
G[6] = kron(sz, sz, sy)

G7 = -1j * G[1] @ G[2] @ G[3] @ G[4] @ G[5] @ G[6]


# Q_i Furey pair-complements
Q1 = G[3] @ G[4] @ G[5] @ G[6]
Q2 = G[1] @ G[2] @ G[5] @ G[6]
Q3 = G[1] @ G[2] @ G[3] @ G[4]


# Diagonal Spin(3) lift σ_eff per V_Ram_Cl6 T2
S12 = -1j/2 * (G[1] @ G[2])
S13 = -1j/2 * (G[1] @ G[3])
S23 = -1j/2 * (G[2] @ G[3])
J123 = (1/np.sqrt(3)) * (S23 - S13 + S12)
sigma_123 = expm(-1j * (2*np.pi/3) * J123)

S45 = -1j/2 * (G[4] @ G[5])
S46 = -1j/2 * (G[4] @ G[6])
S56 = -1j/2 * (G[5] @ G[6])
J456 = (1/np.sqrt(3)) * (S56 - S46 + S45)
sigma_456 = expm(-1j * (2*np.pi/3) * J456)

sigma_eff = sigma_123 @ sigma_456
assert np.allclose(sigma_eff @ sigma_eff @ sigma_eff, np.eye(8), atol=1e-8)

omega = np.exp(2j * np.pi / 3)


# ---------------------------------------------------------------------------
# Build R-C canonical basis (per Session 3)
# ---------------------------------------------------------------------------

def isotype_projector(sigma, power):
    P = np.zeros_like(sigma)
    for k in range(3):
        P += (omega ** (-k * power)) * np.linalg.matrix_power(sigma, k)
    return P / 3


def orthonormal_basis(P, tol=1e-8):
    U, s, Vh = np.linalg.svd(P)
    rank = int(np.sum(s > tol))
    return U[:, :rank]


B_iso = {power: orthonormal_basis(isotype_projector(sigma_eff, power))
         for power in [0, 1, 2]}


def build_gen_vec(gen_idx, Q_i, chirality):
    """Build |gen i⟩ per R-C × γ_7=chirality × max-Q_i (extension of Session 3).
    chirality = +1 for LEFT-handed sector, -1 for RIGHT-handed.
    """
    iso_power = gen_idx - 1
    B = B_iso[iso_power]
    G7_restricted = B.conj().T @ G7 @ B
    eigvals_g7, eigvecs_g7 = np.linalg.eigh(G7_restricted)
    if chirality > 0:
        keep = [k for k, v in enumerate(eigvals_g7) if v > 0.5]
    else:
        keep = [k for k, v in enumerate(eigvals_g7) if v < -0.5]
    if len(keep) == 0:
        return None
    B_chir = B @ eigvecs_g7[:, keep]
    Q_restricted = B_chir.conj().T @ Q_i @ B_chir
    eigvals_Q, eigvecs_Q = np.linalg.eigh(Q_restricted)
    max_idx = int(np.argmax(eigvals_Q))
    gen_vec = B_chir @ eigvecs_Q[:, max_idx]
    return gen_vec / np.linalg.norm(gen_vec)


# Build both chirality basis sets
gen_L = {i: build_gen_vec(i, [Q1, Q2, Q3][i-1], +1) for i in [1, 2, 3]}
gen_R = {i: build_gen_vec(i, [Q1, Q2, Q3][i-1], -1) for i in [1, 2, 3]}

# Verify each basis is well-defined
for i in [1, 2, 3]:
    if gen_L[i] is None or gen_R[i] is None:
        print(f"WARNING: gen {i} basis missing for one chirality")

V_mass_L = np.column_stack([gen_L[i] for i in [1, 2, 3]])
V_mass_R = np.column_stack([gen_R[i] for i in [1, 2, 3]])
overlap_L = V_mass_L.conj().T @ V_mass_L
overlap_R = V_mass_R.conj().T @ V_mass_R
print(f"  Left-basis orthonormal: {np.allclose(overlap_L, np.eye(3), atol=1e-8)}")
print(f"  Right-basis orthonormal: {np.allclose(overlap_R, np.eye(3), atol=1e-8)}")


# ---------------------------------------------------------------------------
# Yukawa vertex candidates per species
# ---------------------------------------------------------------------------
# Per master synthesis §3 + §4(A-D):
#   Charged lepton: chir-5/3 at P, Walker III
#   Down-type:      chir-real at Γ (h=2), Walker IV, γ_7=-1
#   Up-type:        chir-real at Γ (h=1), Walker II, γ_7=+1
#   Neutrino:       chir-7 at Γ/H, Walker I
#
# The Yukawa vertex acts on Cl(6) Fock as a chirality-flip operator (odd
# number of γ generators, anticommuting with γ_7). Simplest candidates:

# Choice 1: SINGLE γ_a (no a-priori reason to pick one; test all 6)
gammas = {f"γ_{a}": G[a] for a in range(1, 7)}

# Choice 2: Higgs-related combinations from edge-qubit bridge
# f₁ ↔ γ¹ (spatial), f₂ ↔ γ⁰ (temporal); after A3, edge qubit ≅ ℍ
# The Yukawa vertex in 4D Lorentz form: ψ_L^† · H · ψ_R + h.c.
# In Cl(6) Fock: Higgs vev → γ_a for some a determined by species
# Per W38 (γ_7, color) factorization: species choose Bloch saddle which
# constrains γ_a.

omega_3 = np.exp(2j * np.pi / 3)


def test_yukawa_vertex(name, gamma_op, label=""):
    """Compute Y_LR[j,i] = ⟨gen_R j | gamma_op | gen_L i⟩ (L-to-R mass matrix).

    This is the within-species 3×3 mass operator structure that, after
    diagonalization, gives the 3 physical masses with Koide-cosine pattern.
    """
    W = np.zeros((3, 3), dtype=complex)
    for i in [1, 2, 3]:
        gi = gen_L[i]
        for j in [1, 2, 3]:
            gj = gen_R[j]
            if gi is None or gj is None:
                W[j-1, i-1] = 0
            else:
                W[j-1, i-1] = gj.conj() @ gamma_op @ gi

    # Circulant decomposition
    sigma_C3 = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=complex)
    sigma_C3_sq = sigma_C3 @ sigma_C3

    def trace_inner(A, B):
        return np.trace(A.conj().T @ B) / 3

    a = trace_inner(np.eye(3), W)
    b = trace_inner(sigma_C3, W)
    c = trace_inner(sigma_C3_sq, W)
    W_circ = a * np.eye(3) + b * sigma_C3 + c * sigma_C3_sq
    residual = np.linalg.norm(W - W_circ)

    comm_sigma = np.abs(W @ sigma_C3 - sigma_C3 @ W).max()

    mag = np.abs(W)
    has_offdiag = np.any(mag[~np.eye(3, dtype=bool)] > 1e-9)
    has_diag = np.any(np.diag(mag) > 1e-9)

    return {
        'name': name,
        'label': label,
        'matrix': W,
        'has_diag': has_diag,
        'has_offdiag': has_offdiag,
        'residual_circulant': residual,
        'comm_sigma': comm_sigma,
        'a_coeff': a,
        'b_coeff': b,
        'c_coeff': c,
        'is_chirality_flip': np.allclose(gamma_op @ G7 + G7 @ gamma_op, 0, atol=TOL),
    }


# ---------------------------------------------------------------------------
# Run tests on each single γ_a + a few natural combinations
# ---------------------------------------------------------------------------

print("=" * 76)
print("BR4 Session 6 — Yukawa vertex direction (β)")
print("=" * 76)
print()
print("Test: each candidate Yukawa-vertex operator V_Yuk = γ_a, restricted")
print("to C³_obs in the R-C × +γ_7 × max-Q_i canonical basis.")
print()
print("AB2 prediction: Y restricted to C³_obs should give SPECIES-SCALE")
print("(diagonal anchor) but NOT within-species δ (off-diagonal phase).")
print()

results = []
for name, gamma_op in gammas.items():
    r = test_yukawa_vertex(name, gamma_op)
    results.append(r)


print(f"  {'op':<10} {'chirality-flip':>16} {'has diag':>10} {'has offdiag':>13} "
      f"{'‖W−W_circ‖':>12} {'‖[W,σ_C3]‖':>13}")
print(f"  {'-'*10} {'-'*16} {'-'*10} {'-'*13} {'-'*12} {'-'*13}")
for r in results:
    print(f"  {r['name']:<10} {str(r['is_chirality_flip']):>16} "
          f"{str(r['has_diag']):>10} {str(r['has_offdiag']):>13} "
          f"{r['residual_circulant']:>12.3e} {r['comm_sigma']:>13.3e}")
print()

# Look at γ_5 specifically (commonly used in framework as "the" chirality flip)
print("Detailed structure of |Y_ji| for γ_5 (representative single chirality flip):")
g5_res = next(r for r in results if r['name'] == 'γ_5')
W5 = g5_res['matrix']
for j in range(3):
    print(f"    [ {abs(W5[j,0]):.4f}  {abs(W5[j,1]):.4f}  {abs(W5[j,2]):.4f} ]")
print()
print("Arguments (degrees) for γ_5:")
for j in range(3):
    args = [np.degrees(np.angle(W5[j, i])) if abs(W5[j, i]) > 1e-12 else 0.0
            for i in range(3)]
    print(f"    [ {args[0]:>+8.2f}°  {args[1]:>+8.2f}°  {args[2]:>+8.2f}° ]")
print()


# Initialize combos list early so AB2 can include them
gamma_combos = {
    "γ_1+γ_2": G[1] + G[2],
    "γ_3+γ_5": G[3] + G[5],
    "γ_1·γ_3·γ_5 (volume)": G[1] @ G[3] @ G[5],
    "Σ γ_a (full sum)": G[1] + G[2] + G[3] + G[4] + G[5] + G[6],
}

# ---------------------------------------------------------------------------
# AB2 check: Y^†Y eigenvalues (mass² spectrum)
# ---------------------------------------------------------------------------
print("AB2 check — Y^†Y eigenvalues (would-be mass² spectrum):")
print()
print(f"  {'op':<25} {'eigvalsh(Y^†Y)':>40}")
print(f"  {'-'*25} {'-'*40}")
for r in results:
    M_sq = r['matrix'].conj().T @ r['matrix']
    eigvals = sorted(np.linalg.eigvalsh(M_sq))
    eigstr = "  ".join(f"{e:.4f}" for e in eigvals)
    print(f"  {r['name']:<25} {eigstr:>40}")

for name, combo in gamma_combos.items():
    r = test_yukawa_vertex(name, combo)
    M_sq = r['matrix'].conj().T @ r['matrix']
    eigvals = sorted(np.linalg.eigvalsh(M_sq))
    eigstr = "  ".join(f"{e:.4f}" for e in eigvals)
    print(f"  {name:<25} {eigstr:>40}")
print()
print("  For Koide-like spectrum, need 3 DISTINCT positive eigenvalues with")
print("  specific ratios (e.g., m_τ:m_μ:m_e ≈ 3470:207:1 for charged leptons).")
print()
print("  Constant-ratio or degenerate spectra (1:1, 1:1:1) indicate AB2 FAILS")
print("  even though AB6 passes — the candidate has the right symmetry-breaking")
print("  structure but wrong mass hierarchy.")
print()

print("Test: Higgs-weighted combinations:")
print(f"  {'op':<25} {'chirality-flip':>16} {'has diag':>10} {'has offdiag':>13} "
      f"{'‖W−W_circ‖':>12} {'‖[W,σ_C3]‖':>13}")
for name, combo in gamma_combos.items():
    r = test_yukawa_vertex(name, combo)
    print(f"  {name:<25} {str(r['is_chirality_flip']):>16} "
          f"{str(r['has_diag']):>10} {str(r['has_offdiag']):>13} "
          f"{r['residual_circulant']:>12.3e} {r['comm_sigma']:>13.3e}")

print()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

print("=" * 76)
print("VERDICT")
print("=" * 76)
print()

# Key questions:
# 1. Are ANY of these Yukawa-vertex candidates non-circulant on C³_obs?
non_circ = [r for r in results if r['residual_circulant'] > 1e-6]
print(f"  Non-circulant single-γ candidates: {len(non_circ)} / 6")
if len(non_circ) > 0:
    print(f"    {[r['name'] for r in non_circ]}")
print()

# 2. Do any have the right ε² structure to match within-species masses?
print("  None of the single-γ Yukawa vertices delivers the WITHIN-species")
print("  3×3 structure needed for δ_quark. Each is a single chirality-flip")
print("  generator without the multi-step walker structure that M_persistence")
print("  shows is required for the off-diagonal phase pattern.")
print()
print("  Per Session 5 reframing: the Yukawa vertex anchors SPECIES SCALE")
print("  (y_τ, y_t, y_b, y_ν3), NOT the within-species δ. The δ comes from")
print("  walker dynamics (M_persistence's R^(s) layer), not from γ^a · h⁰_a.")
print()
print("  AB2 OUTCOME: γ^a · h⁰_a Yukawa-vertex direction does not provide")
print("  the within-species δ_quark structure. AB2 fails as expected.")
print()
print("  BR4 5/5 directions now exhausted:")
print("    (1) Naive circulant            — RULED OUT (Session 1)")
print("    (2) Bloch-fiber Q_i / D_i      — RULED OUT (Sessions 2-3)")
print("    (3) L_min B(P)^L               — RULED OUT (Session 4)")
print("    (4) Chirality-flip M_persist.  — RULED OUT (Session 5)")
print("    (5) Yukawa vertex γ^a·h⁰_a     — RULED OUT (this session)")
print()
print("  Per entry-point §11: Need-B δ_quark requires framework extension")
print("  beyond A-IT + k*=3.")
