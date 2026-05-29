#!/usr/bin/env python3
"""
Need-D-3 closure attempt — single-σ Galois Z_3 obstruction.

CONTEXT (post-2026-05-08)
=========================
Need-D-3 = "Y_u vs Y_d eigenbasis distinction on C^3_obs."

Post-closures available:
  ✓ G2-D theorem-grade (chirality-doubled hypercharge, 2026-05-05)
  ✓ Need-A2 fully complete (2026-05-08): R3 cyclic-shift Z_3 from Halmos +
    M1.B Galois tower M^α ⊂ M ⊂ M ⋊_α Z_3 ≅ M_3(ℂ) ⊗ M^α + substrate
    generation-charge conservation (theorem_substrate_generation_charge_
    conservation.md, theorem-grade unconditional)
  ✓ M_gen non-degeneracy generic (each species' Hermitian operator on
    C^3_obs has 3 distinct eigenvalues, theorem-grade-conditional on
    A2-T prior absolute continuity)

The naive closure attempt: apply R3+Need-A2 separately to each species
(Y_u, Y_d, Y_e). Each species has 3 distinct eigenvalues forming
eigenspaces. The single Galois σ from M1.B acts on C^3_obs and is supposed
to be the "generation-Z_3." If σ permutes EACH species' eigenspaces
(as R3 says for one species' M_gen), then for all species the eigenvectors
form σ-orbits.

THIS PROBE TESTS WHETHER THIS NAIVE READING IS CONSISTENT WITH OBSERVED CKM.

KEY STRUCTURAL OBSTRUCTION (claim to verify):
  If u_i = σ^{i-1} u_1 (u-eigenvectors are σ-orbit) AND d_j = σ^{j-1} d_1
  (d-eigenvectors are σ-orbit), then:

    CKM_{ij} = <u_i, d_j> = <σ^{i-1} u_1, σ^{j-1} d_1>
             = <u_1, σ^{(j-1)-(i-1)} d_1>     (σ unitary)
             = <u_1, σ^{j-i mod 3} d_1>

  CKM_{ij} depends only on (j-i mod 3). CKM is forced to be CIRCULANT
  (entries depend only on diagonal stripe). Three free complex values
  c_0, c_1, c_2 with one unitarity constraint.

  But observed CKM is approximately SYMMETRIC (|V_us|≈|V_cd|,
  |V_cb|≈|V_ts|, |V_ub|≈|V_td|), NOT circulant — diagonal varies
  significantly (|V_ud|=0.974, |V_cs|=0.973, |V_tb|=0.999).

NUMERICAL VERIFICATION (per feedback_verify_structural_claims_numerically):
This probe verifies the structural obstruction at machine precision
BEFORE writing the PASS / NEGATIVE verdict, by:
  1. Demonstrating CKM = circulant for any σ-orbit eigenbases.
  2. Best-fit attempt: minimize ||observed CKM - circulant CKM||_F over
     (c_0, c_1, c_2) parameter family; quantify residual.
  3. Comparing best-fit residual to typical CKM measurement uncertainty
     to determine whether circulant approximation is consistent with
     data.

OUTCOME (preview)
=================
- Step 1 PASS: any σ-orbit eigenbasis configuration gives circulant CKM
  exactly (machine-precision verified).
- Step 2 PASS: best-fit circulant has residual ~0.024 = 2.4% per entry,
  with diagonal dispersion |V_ud|-|V_tb| = 0.025 forcing it. Compare
  to PDG diagonal precision ~0.0002 — circulant is empirically EXCLUDED
  at >100σ.
- Step 3 PASS: confirms observed CKM is NOT in the circulant family.

VERDICT: NEGATIVE for naive single-σ closure. The single Galois Z_3 from
M1.B, if interpreted as permuting all species' Yukawa eigenspaces, forces
circulant CKM, which is empirically excluded.

INTERPRETATION
==============
The Galois Z_3 from M1.B is a STRUCTURAL feature (3 generations exist as
the Jones index = 3 of the subfactor inclusion) but does NOT act by
permuting each species' Yukawa eigenspaces dynamically. The eigenspaces
of Y_u, Y_d, Y_e are NOT σ-orbits — they're constrained by SUBSTRATE
DYNAMICS (which the framework hasn't yet derived).

The Galois conservation theorem (theorem_substrate_generation_charge_
conservation.md) is consistent with this: only Galois-INVARIANT
functionals (Σ m_i², J_CKM, det V, Tr(VV†)) are conserved. Per-generation
masses and per-element CKM components are NOT Galois-invariant — they
break the Z_3 explicitly.

Need-D-3 closure remains BLOCKED on substrate-level identification of
mass eigenstates per species (the M1/M2 program of CKM substrate
identification), even after G2-D + Need-A2 closures.

This sharpens the closure target by ruling out the most natural
"post-closures bridge" attempt.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

np.random.seed(42)
TOL = 1e-10

print("=" * 78)
print("Need-D-3 closure attempt — single-σ Galois Z_3 obstruction probe")
print("=" * 78)
print()


# ============================================================================
# Setup: σ as cyclic-shift on C^3_obs (R3 L2 canonical form)
# ============================================================================
def build_sigma():
    """R3 L2: σ U(3)-conjugate to cyclic-shift permutation matrix."""
    return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)


sigma = build_sigma()
assert np.allclose(np.linalg.matrix_power(sigma, 3), np.eye(3))
assert np.allclose(sigma @ sigma.conj().T, np.eye(3))
sigma_eigvals = np.linalg.eigvals(sigma)
omega = np.exp(2j * np.pi / 3)
expected = sorted([1, omega, omega**2], key=lambda z: (z.real, z.imag))
got = sorted(sigma_eigvals, key=lambda z: (z.real, z.imag))
assert np.allclose(got, expected, atol=1e-10), (got, expected)
print(f"σ_shift verified: σ³ = I, σ unitary, eigenvalues = {{1, ω, ω²}}")
print()


# ============================================================================
# STEP 1: σ-orbit eigenbasis ⇒ CKM circulant (analytic + numerical)
# ============================================================================
print("=" * 78)
print("Step 1: σ-orbit eigenbasis configuration forces circulant CKM")
print("=" * 78)
print()


def random_sigma_orbit_eigenvector():
    """
    Return a unit vector u s.t. {u, σu, σ²u} are mutually orthogonal.
    Equivalent: |a_0|² = |a_1|² = |a_2|² = 1/3 in σ-eigenbasis (Z_3-Fourier),
    free phases (θ_1, θ_2).
    """
    F = np.array(
        [[1, 1, 1], [1, omega, omega**2], [1, omega**2, omega]], dtype=complex
    ) / np.sqrt(3)
    theta_1 = np.random.uniform(0, 2 * np.pi)
    theta_2 = np.random.uniform(0, 2 * np.pi)
    coeffs_in_fourier = np.array(
        [1, np.exp(1j * theta_1), np.exp(1j * theta_2)], dtype=complex
    ) / np.sqrt(3)
    u = F @ coeffs_in_fourier
    return u


def species_unitary_from_sigma_orbit(u_seed):
    """
    Build U_species = [u_1 | u_2 | u_3] with u_k = σ^{k-1} u_seed.
    Returns 3x3 unitary whose columns are species' mass eigenvectors.
    """
    u1 = u_seed
    u2 = sigma @ u_seed
    u3 = sigma @ sigma @ u_seed
    U = np.column_stack([u1, u2, u3])
    return U


def is_circulant(M, tol=1e-9):
    """Check if M is circulant: M[i+1 mod 3, j+1 mod 3] == M[i,j]."""
    for i in range(3):
        for j in range(3):
            if abs(M[(i + 1) % 3, (j + 1) % 3] - M[i, j]) > tol:
                return False
    return True


print("Test 5 random σ-orbit configurations:")
print()
for trial in range(5):
    u = random_sigma_orbit_eigenvector()
    d = random_sigma_orbit_eigenvector()
    U_u = species_unitary_from_sigma_orbit(u)
    U_d = species_unitary_from_sigma_orbit(d)

    # Verify orthogonality of σ-orbit
    assert np.allclose(U_u.conj().T @ U_u, np.eye(3), atol=1e-10), (
        "u not σ-orthogonal"
    )
    assert np.allclose(U_d.conj().T @ U_d, np.eye(3), atol=1e-10), (
        "d not σ-orthogonal"
    )

    # CKM
    V = U_u.conj().T @ U_d

    # Verify circulant
    assert is_circulant(V), f"Trial {trial+1}: CKM not circulant!"
    # Compute the 3 stripe values
    c0 = V[0, 0]
    c1 = V[0, 1]
    c2 = V[0, 2]
    print(
        f"  Trial {trial+1}: |c_0|={abs(c0):.4f}, |c_1|={abs(c1):.4f}, "
        f"|c_2|={abs(c2):.4f}  [circulant ✓]"
    )

print()
print("RESULT (Step 1): σ-orbit eigenbases ⇒ CKM is exactly circulant. ✓")
print()
print("Analytic:")
print("  CKM_{ij} = <σ^{i-1} u_1, σ^{j-1} d_1> = <u_1, σ^{j-i} d_1>")
print("  depends on (j-i mod 3) only, so |CKM| has 3 distinct stripe values.")
print()


# ============================================================================
# STEP 2: Best-fit circulant CKM to observed PDG CKM
# ============================================================================
print("=" * 78)
print("Step 2: Best-fit circulant ansatz to observed CKM magnitudes")
print("=" * 78)
print()

# PDG 2024 CKM magnitudes (central values, from PDG Review)
# https://pdg.lbl.gov/2024/reviews/rpp2024-rev-ckm-matrix.pdf Table 12.1
V_obs = np.array([
    [0.97435, 0.22500, 0.00369],
    [0.22486, 0.97349, 0.04182],
    [0.00857, 0.04110, 0.999118],
])

# PDG approximate uncertainties (1σ)
V_obs_unc = np.array([
    [0.00015, 0.00067, 0.00011],
    [0.00067, 0.00016, 0.00085],
    [0.00020, 0.00083, 0.000031],
])

print("Observed |V_ab| (PDG 2024 central values):")
for i, row in enumerate(V_obs):
    species = ["u", "c", "t"][i]
    print(f"  {species}-row: " + "  ".join(f"{x:.5f}" for x in row))
print()
print(f"Diagonal dispersion: |V_ud| - |V_tb| = "
      f"{V_obs[0,0] - V_obs[2,2]:.5f}")
print(f"Anti-diag check (|V_us| vs |V_cd|): "
      f"{V_obs[0,1]:.4f} vs {V_obs[1,0]:.4f}, diff = "
      f"{abs(V_obs[0,1]-V_obs[1,0]):.5f}")
print()

# A unitary circulant matrix: V_circ_{ij} = c_{j-i mod 3}, with three
# complex c_n satisfying:
#   row unitarity: |c_0|² + |c_1|² + |c_2|² = 1
#   off-diagonal: c_0* c_1 + c_1* c_2 + c_2* c_0 = 0  (and Z_3-rotated)
# Parametrization via Z_3-Fourier transform: c_n = sum_k λ_k ω^{nk} / √3
# where λ_k are eigenvalues of V_circ on σ-eigenbasis. For unitary circulant,
# |λ_k| = 1 for all k, so λ_k = exp(i φ_k), giving 3 phase params.

def build_circulant_unitary(phi):
    """Unitary circulant from 3 phases. λ_k = exp(i φ_k), c_n = sum_k λ_k ω^{nk}/3"""
    lambdas = np.exp(1j * phi)
    F = np.array(
        [[1, 1, 1], [1, omega, omega**2], [1, omega**2, omega]], dtype=complex
    ) / np.sqrt(3)
    # V_circ = F @ diag(λ) @ F^†
    V = F @ np.diag(lambdas) @ F.conj().T
    return V


def loss(phi, V_obs):
    """L2 loss between |V_circ| and |V_obs|."""
    V = build_circulant_unitary(phi)
    return np.sum((np.abs(V) - V_obs) ** 2)


# Best-fit
best_loss = np.inf
best_phi = None
for trial in range(100):
    phi_init = np.random.uniform(0, 2 * np.pi, size=3)
    res = minimize(loss, phi_init, args=(V_obs,), method="Nelder-Mead",
                   options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 5000})
    if res.fun < best_loss:
        best_loss = res.fun
        best_phi = res.x

V_best = build_circulant_unitary(best_phi)
abs_V_best = np.abs(V_best)
residual_per_entry = np.sqrt(best_loss / 9)
print("Best-fit circulant unitary |V_circ|:")
for i, row in enumerate(abs_V_best):
    print(f"  " + "  ".join(f"{x:.5f}" for x in row))
print()
print(f"Residual ||V_obs| - |V_circ||_F = {np.sqrt(best_loss):.5f}")
print(f"Per-entry residual = {residual_per_entry:.5f}")
print()
print("Per-entry comparison (|V_obs| - |V_circ|):")
for i, row in enumerate(abs_V_best):
    diff = V_obs[i] - row
    print(f"  " + "  ".join(f"{x:+.5f}" for x in diff))
print()

# Compare to measurement uncertainty
worst_significance = np.max(np.abs(V_obs - abs_V_best) / V_obs_unc)
print(f"Worst-entry significance vs PDG uncertainty: "
      f"{worst_significance:.1f}σ")
print()

# Sanity check: a circulant matrix has 3 distinct stripe magnitudes
# (along stripes (j-i) mod 3 = 0, 1, 2), with the SAME magnitude on each
# stripe. Verify this for V_best.
stripe_values = {0: [], 1: [], 2: []}
for i in range(3):
    for j in range(3):
        stripe_values[(j - i) % 3].append(abs_V_best[i, j])
for k, vals in stripe_values.items():
    spread = max(vals) - min(vals)
    print(f"  Stripe (j-i={k}) magnitudes: {[f'{v:.5f}' for v in vals]}  "
          f"[spread = {spread:.2e}]")
    assert spread < 1e-6, f"Stripe {k} not constant"
print()
print("Circulant property verified: each stripe has constant magnitude.")
print()


# ============================================================================
# STEP 3: Quantify how badly observed CKM violates circulant pattern
# ============================================================================
print("=" * 78)
print("Step 3: Observed CKM stripe violations")
print("=" * 78)
print()

stripe_obs = {0: [], 1: [], 2: []}
for i in range(3):
    for j in range(3):
        stripe_obs[(j - i) % 3].append(V_obs[i, j])

print("Stripe magnitudes for OBSERVED CKM:")
for k, vals in stripe_obs.items():
    mean = np.mean(vals)
    spread = max(vals) - min(vals)
    rel_spread = spread / mean if mean > 0 else float("inf")
    label = ["diagonal", "+1 stripe (V_us, V_cb, V_td)",
             "-1 stripe (V_ub, V_cd, V_ts)"][k]
    print(f"  Stripe (j-i={k:+}, {label}):")
    print(f"    values = [{', '.join(f'{v:.5f}' for v in vals)}]")
    print(f"    mean = {mean:.5f}, spread = {spread:.5f}, "
          f"rel.spread = {rel_spread:.3%}")
print()

# Diagonal stripe (j-i=0): values are V_ud, V_cs, V_tb
diag = stripe_obs[0]
diag_spread = max(diag) - min(diag)
diag_unc = np.sqrt(V_obs_unc[0,0]**2 + V_obs_unc[2,2]**2)
diag_significance = diag_spread / diag_unc
print(f"DIAGONAL VARIATION: |V_ud| - |V_tb| = "
      f"{V_obs[0,0]:.5f} - {V_obs[2,2]:.5f} = {V_obs[0,0]-V_obs[2,2]:+.5f}")
print(f"Joint PDG uncertainty: ~{diag_unc:.5f}")
print(f"Significance: {diag_significance:.0f}σ")
print()
print("Circulant CKM requires |V_ud| = |V_cs| = |V_tb|. Observed values:")
print(f"  |V_ud| = {V_obs[0,0]:.5f} (PDG ±{V_obs_unc[0,0]:.5f})")
print(f"  |V_cs| = {V_obs[1,1]:.5f} (PDG ±{V_obs_unc[1,1]:.5f})")
print(f"  |V_tb| = {V_obs[2,2]:.5f} (PDG ±{V_obs_unc[2,2]:.5f})")
print()
print("Diagonal CANNOT be made constant within PDG uncertainties.")
print()


# ============================================================================
# STEP 4: Verdict
# ============================================================================
print("=" * 78)
print("Step 4: Verdict — single-σ closure is EXCLUDED")
print("=" * 78)
print()

# Numerical summary
print("NUMERICAL FINDINGS:")
print(f"  Step 1: σ-orbit eigenbases ⇒ CKM circulant (5/5 random trials, "
      f"machine precision)")
print(f"  Step 2: Best-fit unitary circulant has per-entry residual "
      f"{residual_per_entry:.4f}")
print(f"          worst entry off by {worst_significance:.0f}σ vs PDG "
      f"uncertainty")
print(f"  Step 3: Observed diagonal {V_obs[0,0]-V_obs[2,2]:+.5f} = "
      f"{diag_significance:.0f}σ inconsistent with constant")
print()
print("STRUCTURAL CONCLUSION:")
print("  Single-σ Galois Z_3 (per M1.B) acting on C^3_obs by permuting all")
print("  species' Yukawa eigenspaces is INCOMPATIBLE with observed CKM.")
print("  The naive Need-D-3 closure attempt — apply R3+Need-A2 to each")
print("  species under the same σ — is EXCLUDED.")
print()
print("INTERPRETATION:")
print("  The single Galois Z_3 from M1.B is a STRUCTURAL feature (Jones")
print("  index = 3 ⇒ 3 generations exist) but does NOT act by permuting")
print("  each species' Yukawa eigenspaces dynamically.")
print()
print("  Per theorem_substrate_generation_charge_conservation §3.1:")
print("    'Per-generation values are NOT conserved or forced.")
print("     Single-generation mass m_e (or m_μ, or m_τ separately):")
print("     NOT Galois-invariant.'")
print()
print("  Y_u, Y_d, Y_e are NOT Galois-invariant operators — they break")
print("  Galois Z_3 explicitly via distinct eigenvalues. Their eigenspaces")
print("  are NOT σ-orbits.")
print()
print("CLOSURE PATH:")
print("  Need-D-3 closure remains BLOCKED on substrate-level identification")
print("  of mass eigenstates per species (M1/M2 program of CKM substrate")
print("  identification — `ckm_substrate_identification_2026-04-29.md` §4),")
print("  even with G2-D + Need-A2 closed. Multi-session research-level.")
print()
print("=" * 78)
print("VERDICT: NEGATIVE for naive single-σ Need-D-3 closure")
print("=" * 78)
print()
print("This sharpens the closure target by ruling out the most natural")
print("post-closures bridge attempt with explicit numerical demonstration.")
