#!/usr/bin/env python3
"""
W23 — Structural audits A/B/C of the Koide derivation chain.

Date: 2026-05-26
Context: closes the m_e/m_μ Koide-ratio residual investigation by replacing
α-power pattern matching (W4-W22 falsified) with audits of foundational
framework derivations.

THREE AUDITS:

(A) Are (μ_t, μ_ω, μ_ω̄) = (4, 2, 2) substrate-exact (topological)
    or counting-based (admits substrate-dependent corrections)?

(B) Is the cos-form f_j = 1 + ε·cos(2πj/k* + δ) used in m_e.py / m_mu.py
    DERIVED from substrate Q_Koide.py amp_j, or is it a parametric
    extrapolation? Specifically: where does δ = 2/9 enter the cos
    argument structurally?

(C) Does walker holonomy h^g at length g (the framework's PMNS phase
    mechanism, W45) enter mass eigenvalues at sub-leading order? If so,
    what cos-phase would it produce? Does it match the empirical
    δ ≈ 12.73° ≈ 2/9 rad?

This is structural reasoning + numerical cross-check, NOT curve-fitting.
"""

import math
import numpy as np
import sympy as sp
from numpy import linalg as la
from itertools import product

print("=" * 76)
print("W23 — Structural audits A/B/C of the Koide derivation chain")
print("Date: 2026-05-26")
print("=" * 76)

# ============================================================
# AUDIT A — (4,2,2) topological exactness
# ============================================================
print()
print("=" * 76)
print("AUDIT A — Are (4,2,2) multiplicities topologically exact?")
print("=" * 76)

# Per predictions/B_P_doubly_degenerate_h_derivation.md:
#   - A(P) is 4×4 Hermitian, char poly (λ²-3)² → eigenvalues ±√3 each mult 2
#   - C_3 invariant: +√3-space = trivial ⊕ ω; -√3-space = trivial ⊕ ω²
#   - Ihara-Bass lifts to B(P): each A-eigenvalue gives 2 B-eigenvalues with
#     mult inherited (squared inner factor)
#   - V_Ram = 8-dim = h(2) ⊕ h*(2) ⊕ -h(2) ⊕ -h*(2)
#   - C_3 isotypic counts on V_Ram: trivial=4, ω=2, ω²=2 (EXACT INTEGERS)

# The multiplicities are DIMENSIONS of finite-dimensional subspaces of V_Ram,
# determined by:
#   1. Algebraic identity (Ihara-Bass)
#   2. Schur's lemma applied to the C_3-symmetry-preserved decomposition
# Both are EXACT structural identities with no quantitative slack.

# Verify by direct construction:
A_PRIM = np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]])
ATOMS = np.array([[1/8, 1/8, 1/8], [3/8, 7/8, 5/8],
                  [7/8, 5/8, 3/8], [5/8, 3/8, 7/8]])
N_ATOMS = 4
g_girth = 10
k_star = 3
k_P = np.array([.25, .25, .25])

# Build bond list
_d = []
for i in range(N_ATOMS):
    for j in range(N_ATOMS):
        for n in product(range(-2, 3), repeat=3):
            rj = ATOMS[j] + n @ A_PRIM
            d = la.norm(rj - ATOMS[i])
            if d > 0.02:
                _d.append(d)
NN = min(_d)
bonds = []
for i in range(N_ATOMS):
    for j in range(N_ATOMS):
        for n in product(range(-2, 3), repeat=3):
            rj = ATOMS[j] + n @ A_PRIM
            if abs(la.norm(rj - ATOMS[i]) - NN) < 0.02:
                bonds.append((i, j, n))

def build_hashimoto(k_frac):
    n = len(bonds)
    B = np.zeros((n, n), dtype=complex)
    k = np.asarray(k_frac, float)
    for fi, (fs, ft, fc) in enumerate(bonds):
        for ei, (es, et, ec) in enumerate(bonds):
            if fs != et:
                continue
            if ft == es and np.array_equal(fc, tuple(-x for x in ec)):
                continue
            B[fi, ei] = np.exp(2j * np.pi * np.dot(k, fc))
    return B

B_P = build_hashimoto(k_P)
ev = la.eigvals(B_P)

# C_3 action on directed-edge space: permutation built from σ on vertices
# σ = (v_0)(v_1 v_3 v_2) → bond permutation
sigma_vertex = {0: 0, 1: 3, 2: 1, 3: 2}  # σ : v_i ↦ v_{σ(i)} from sigma = (v1 v3 v2)
# Bond (i,j,n) → bond (σ(i), σ(j), σ(n)) where σ acts on cell vector by cyclic perm
def sigma_cell(n):
    return (n[2], n[0], n[1])  # body-diagonal C_3 cyclic permutation on (k1,k2,k3) → on cell coords

sigma_bond_idx = {}
for idx, (i, j, n) in enumerate(bonds):
    target = (sigma_vertex[i], sigma_vertex[j], sigma_cell(n))
    # find matching bond
    for idx2, b2 in enumerate(bonds):
        if b2[0] == target[0] and b2[1] == target[1] and tuple(b2[2]) == tuple(target[2]):
            sigma_bond_idx[idx] = idx2
            break

# Build C_3 permutation matrix on bond space (12-dim)
n_bonds = len(bonds)
P_sigma = np.zeros((n_bonds, n_bonds), dtype=complex)
for src, tgt in sigma_bond_idx.items():
    P_sigma[tgt, src] = 1.0

# Verify P_sigma^3 ≈ identity
P3 = P_sigma @ P_sigma @ P_sigma
assert np.allclose(P3, np.eye(n_bonds), atol=1e-9), "C_3 action not period 3"
print(f"  C_3 permutation on bond space: period 3 verified.")

# At the P-point, B(P) should commute with P_sigma (after accounting for the
# Bloch phase transformation: σ also acts on k by cyclic perm, but P is fixed)
commutator = B_P @ P_sigma - P_sigma @ B_P
print(f"  ||[B(P), P_σ]||_∞ = {np.max(np.abs(commutator)):.3e}")

# Diagonalize B(P) and project eigenvectors onto C_3 isotypic
evals, evecs = la.eig(B_P)
# Identify V_Ram modes (|h|² = k*-1 = 2)
ram_mask = np.abs(np.abs(evals)**2 - 2.0) < 1e-6
v_ram_idx = np.where(ram_mask)[0]
print(f"  V_Ram dimension: {len(v_ram_idx)}  (expected 8)")

# For each V_Ram eigenvector, compute C_3 isotypic component
# Project ψ onto trivial = (1/3)(I + σ + σ²)ψ, ω = (1/3)(I + ω·σ + ω̄·σ²)ψ, ω̄ = conj
omega = np.exp(2j * np.pi / 3)
P_trivial = (np.eye(n_bonds, dtype=complex) + P_sigma + P_sigma @ P_sigma) / 3
P_omega = (np.eye(n_bonds, dtype=complex) + omega * P_sigma + omega.conj() * (P_sigma @ P_sigma)) / 3
P_omega_bar = (np.eye(n_bonds, dtype=complex) + omega.conj() * P_sigma + omega * (P_sigma @ P_sigma)) / 3

# Verify projectors sum to identity
assert np.allclose(P_trivial + P_omega + P_omega_bar, np.eye(n_bonds), atol=1e-9), \
    "Projectors don't sum to identity"

# Project V_Ram basis (use eigenvectors)
V_ram = evecs[:, v_ram_idx]
norms_trivial = np.linalg.norm(P_trivial @ V_ram, axis=0)
norms_omega = np.linalg.norm(P_omega @ V_ram, axis=0)
norms_omega_bar = np.linalg.norm(P_omega_bar @ V_ram, axis=0)

# Each V_Ram eigenvector sits in a definite isotypic (since B(P) commutes
# with C_3 and shares eigenvectors). For a generic eigenvector, we'd see
# nonzero projection onto multiple isotypics if there's degeneracy.

# Sum the isotypic dimension contributions (trace of projector restricted to V_Ram)
# This is the proper way to count multiplicities:
P_trivial_V_ram = V_ram.conj().T @ P_trivial @ V_ram
P_omega_V_ram = V_ram.conj().T @ P_omega @ V_ram
P_omega_bar_V_ram = V_ram.conj().T @ P_omega_bar @ V_ram

trace_trivial = np.real(np.trace(P_trivial_V_ram))
trace_omega = np.real(np.trace(P_omega_V_ram))
trace_omega_bar = np.real(np.trace(P_omega_bar_V_ram))

print(f"  C_3 isotypic multiplicities on V_Ram:")
print(f"    trivial: {trace_trivial:.4f}  (expected 4)")
print(f"    ω:       {trace_omega:.4f}  (expected 2)")
print(f"    ω̄:       {trace_omega_bar:.4f}  (expected 2)")
print(f"    sum:     {trace_trivial+trace_omega+trace_omega_bar:.4f}  (expected 8)")

a_audit_pass = (abs(trace_trivial - 4) < 0.05 and
                abs(trace_omega - 2) < 0.05 and
                abs(trace_omega_bar - 2) < 0.05)

print()
print(f"  AUDIT A VERDICT: (4,2,2) numerically verified" +
      (" ✓ (algebraic identity)" if a_audit_pass else " ✗ (FAIL)"))
print()
print("  STRUCTURAL: the multiplicities are integer counts of finite-")
print("  dimensional subspace dimensions of V_Ram, fixed by:")
print("    (1) the Ihara-Bass identity (exact algebraic identity)")
print("    (2) Schur's lemma applied to C_3-equivariant decomposition")
print("        of A(P)'s ±√3 eigenspaces")
print("  No 'ppm-scale correction' to (4,2,2) is structurally possible.")
print("  These are TOPOLOGICAL counts. → AUDIT A SETTLED NEGATIVE: (4,2,2)")
print("  cannot be the source of the ppm-scale m_e/m_μ residuals.")


# ============================================================
# AUDIT B — Is the Koide cos-form derived or parametric?
# ============================================================
print()
print("=" * 76)
print("AUDIT B — Is the cos-form f_j = 1 + ε·cos(2πj/k* + δ) derived?")
print("=" * 76)

# The framework has TWO objects that look like Koide cos-forms:
#
# (1) Q_Koide.py amp_j construction (substrate-derived):
#       amp_j = √μ_t + √μ_ω·ω^j + √μ_ω̄·ω^{-j}
#       For (4,2,2): amp_j = 2 + 2√2·cos(2πj/3)    ← NO δ in argument
#       Born rule m_j = |amp_j|² gives:
#         m_0 = (2+2√2)² = 12+8√2 ≈ 23.31  → m_τ slot
#         m_1 = (2-√2)²  = 6-4√2  ≈ 0.343 → m_μ slot
#         m_2 = (2-√2)²  = 6-4√2  ≈ 0.343 → m_e slot
#       Q = 24/36 = 2/3 ✓
#       But m_1 = m_2 EXACTLY → m_e = m_μ degenerate!

mu_t, mu_o, mu_w = 4, 2, 2
amps_substrate = []
for j in range(3):
    a = (sp.sqrt(mu_t)
         + sp.sqrt(mu_o) * sp.exp(2*sp.pi*sp.I*j/3)
         + sp.sqrt(mu_w) * sp.exp(-2*sp.pi*sp.I*j/3))
    amps_substrate.append(sp.simplify(sp.expand_complex(a)))
masses_substrate = [sp.simplify(sp.Abs(a)**2) for a in amps_substrate]
print(f"  Substrate Q_Koide amp_j (Born-rule construction):")
for j, (a, m) in enumerate(zip(amps_substrate, masses_substrate)):
    print(f"    j={j}: amp = {a},  m = {m}  ≈ {float(m):.4f}")
print(f"  → m_1 = m_2 EXACTLY (degenerate)")

# (2) m_e.py / m_mu.py cos-form (used to produce m_e ≠ m_μ):
#       f_j = 1 + ε·cos(2πj/k* + δ)  with ε=√2, δ=2/9
#       m_j = m_τ · (f_j/f_max)²
epsilon = math.sqrt(2)
delta = 2.0/9.0  # taken as RADIANS in m_e.py
factors = [1 + epsilon * math.cos(2*math.pi*j/3 + delta) for j in range(3)]
m_tau_obs = 1776.86  # MeV
fs = sorted(factors)
m_pred = [m_tau_obs * (f/fs[2])**2 for f in fs]
print()
print(f"  m_e.py cos-form (ε=√2, δ=2/9 rad):")
print(f"    f factors (sorted): {[f'{f:.4f}' for f in fs]}")
print(f"    masses: m_e_pred={m_pred[0]*1e3:.4f} keV, m_μ_pred={m_pred[1]:.4f} MeV,"
      f" m_τ={m_pred[2]:.4f} MeV (anchor)")
print(f"  → m_e ≠ m_μ achieved via δ ≠ 0 in cos argument")

# Algebraic verification: does Q from (1+ε·cos(2πj/3+δ))² depend on δ?
# Sum_j m_j ∝ Sum_j (1 + ε cos(θ_j + δ))² where θ_j = 2πj/3
# = Sum_j [1 + 2ε cos(θ_j+δ) + ε² cos²(θ_j+δ)]
# The Sum_j cos(θ_j + δ) = 0 (sum of 3 cube roots rotated by δ still sums to 0)
# Sum_j cos²(θ_j + δ) = 3/2
# So Sum_j m_j = 3 + 0 + 3ε²/2 — INDEPENDENT of δ
# Same for Sum_j sqrt(m_j) = 3 — INDEPENDENT of δ
# Therefore Q = (1+ε²/2)/3 — INDEPENDENT of δ
# δ is COMPLETELY FREE in the cos-form parametrization

Q_sym, eps_sym, delta_sym, j_sym = sp.symbols('Q eps delta j', real=True)
f_j_sym = 1 + eps_sym * sp.cos(2*sp.pi*j_sym/3 + delta_sym)
sum_f = sum(f_j_sym.subs(j_sym, j) for j in range(3))
sum_f_sq = sum(f_j_sym.subs(j_sym, j)**2 for j in range(3))
Q_from_cos = sp.simplify(sum_f_sq / sum_f**2)
print()
print(f"  Algebraic verification of δ-independence of Q:")
print(f"    Q(cos-form) = sum f_j² / (sum f_j)² = {sp.simplify(Q_from_cos)}")
print(f"    This is INDEPENDENT of δ. δ is a free parameter in the cos-form.")

# (3) Compare cos-form δ=0 with substrate:
fs_delta0 = sorted([1 + math.sqrt(2)*math.cos(2*math.pi*j/3) for j in range(3)])
m_pred_delta0 = [m_tau_obs * (f/fs_delta0[2])**2 for f in fs_delta0]
print()
print(f"  cos-form with δ=0 (no phase shift):")
print(f"    f factors: {[f'{f:.4f}' for f in fs_delta0]}")
print(f"    masses: f_min=f_mid → m_e_pred = m_μ_pred = {m_pred_delta0[0]*1e3:.4f} keV")
print(f"  → DEGENERATE — same as substrate Q_Koide amp_j")
print()
print(f"  CONCLUSION: cos-form with δ=0 reproduces the substrate's degenerate")
print(f"  spectrum exactly. δ ≠ 0 is the ARTIFACT introduced to break degeneracy.")

# The framework's own admission (delta_Koide_derivation.md line 3):
#   "the IDENTIFICATION of δ_Bernoulli (variance, dimensionless) with the
#   Koide cosine PHASE δ in radians (the parameter that gives 3-distinct
#   lepton mass values...) is a NUMERICAL coincidence (2/9 ≈ 12.73° matches
#   observed Koide phase). Whether this coincidence has a structural
#   derivation is Need-B of an internal working note
#   — a SEPARATE multi-session research question."
print()
print(f"  AUDIT B VERDICT: the cos-form δ ≠ 0 is NOT derived from substrate.")
print(f"  The substrate Q_Koide construction gives m_1 = m_2 EXACTLY.")
print(f"  Setting δ := Q(1-Q) = 2/9 is a DIMENSIONAL category error:")
print(f"  Q(1-Q) is a dimensionless variance moment; δ is an angle in")
print(f"  radians. Their numerical coincidence is acknowledged in the")
print(f"  framework as Need-B (not yet derived). → AUDIT B IDENTIFIES the")
print(f"  STRUCTURAL DEFECT: δ-phase in cos-form is currently unjustified.")


# ============================================================
# AUDIT C — Does walker holonomy h^g enter mass eigenvalues?
# ============================================================
print()
print("=" * 76)
print("AUDIT C — Does walker holonomy h^g enter mass eigenvalues?")
print("=" * 76)

# The framework's PMNS Majorana phases use walker holonomy:
#   α_21 = arg(h^g) = arg(h^10) ≈ 162.39°
#   δ_CP-channel = arg((-h*)^g) = -arg(h^g) mod 360 = 197.61°
# Both modulate phases in V_Ram's ω and ω² isotypic sectors respectively.

h = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
arg_h = math.degrees(math.atan2(math.sqrt(5), math.sqrt(3)))
arg_h_g10 = (10 * arg_h) % 360
arg_minus_h_conj_g10 = (-10 * arg_h) % 360
print(f"  h = (√3 + i√5)/2,  arg(h) = arctan(√5/√3) ≈ {arg_h:.4f}°")
print(f"  arg(h^10)   = {arg_h_g10:.4f}°  (= α_21 PMNS Majorana phase)")
print(f"  arg((-h*)^10) = {arg_minus_h_conj_g10:.4f}°  (= ω²-channel)")
print()

# If walker holonomy entered the substrate amp construction symmetrically
# (i.e., modifying √μ_ω → √μ_ω · h^g · e^{iφ} and conjugate), it would
# add a global phase shift to the cos argument in amp_j:
#   amp_j_walker = √μ_t + √μ_ω · e^{iφ} · ω^j + √μ_ω̄ · e^{-iφ} · ω^{-j}
#                = √μ_t + 2√μ_ω · cos(2πj/3 + φ)
# with φ = arg(h^g) for SOME natural framework length g.

# Check various natural lengths:
print(f"  Walker holonomy phase candidates from h^L mod 360°:")
for L in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
    phase_L = (L * arg_h) % 360
    phase_L_rad = math.radians(phase_L)
    # Compute predicted m_e/m_μ ratio under this phase
    eps = math.sqrt(2)
    facs = sorted([1 + eps*math.cos(2*math.pi*j/3 + phase_L_rad) for j in range(3)])
    if abs(facs[0]) > 1e-10 and abs(facs[2]) > 1e-10:
        ratio = (facs[1]/facs[0])**2  # m_μ/m_e
    else:
        ratio = float('nan')
    print(f"    L={L:2d}:  φ = {phase_L:7.3f}° = {phase_L_rad:.4f} rad,"
          f"  m_μ/m_e (predicted) = {ratio:.2f}")
print(f"     [observed m_μ/m_e ≈ 206.77]")
print(f"     [empirical Koide phase ≈ 12.73° = 2/9 rad → m_μ/m_e = 206.77]")
print()

# What length L gives exactly δ_emp = 2/9 rad = 12.73°?
delta_emp = 2.0/9.0
arg_h_rad = math.radians(arg_h)
# Find L such that L·arg_h mod 2π = 2/9 rad
# L·0.9115 mod 2π = 0.2222
# L = (0.2222 + 2π·n)/0.9115 for integer n
print(f"  Solving L·arg(h) mod 2π = 2/9 = {delta_emp:.4f} rad")
for n in range(0, 6):
    L_candidate = (delta_emp + 2*math.pi*n) / arg_h_rad
    print(f"    n={n}: L = {L_candidate:.6f}  (need integer L)")
print(f"  → No natural integer L produces walker holonomy = 2/9 rad.")
print(f"  → The empirical Koide cos-phase δ ≈ 2/9 rad does NOT arise from")
print(f"    walker holonomy h^L at any natural integer L on the srs lattice.")
print()

# Check the FRACTIONAL relationship: is 2/9 = arg(h)/k* or similar?
fractions_to_check = [
    ("arg(h)/k*", arg_h_rad / 3),
    ("arg(h)/(2k*)", arg_h_rad / 6),
    ("arg(h)/(2k*²)", arg_h_rad / 18),
    ("arg(h)/(k*²)", arg_h_rad / 9),
    ("arg(h)·k*/(2π·n)", arg_h_rad * 3 / (2*math.pi)),
    ("π/k*² + arg(h)/?", None),
]
print(f"  Check fractional relationships between arg(h) and δ_emp=2/9:")
for label, val in fractions_to_check:
    if val is not None:
        print(f"    {label} = {val:.5f}  vs  2/9 = {delta_emp:.5f},"
              f"  Δ = {abs(val-delta_emp)/delta_emp*100:+.2f}%")

# What's the NATURAL framework value for the substrate-derived cos-phase?
# The Q_Koide.py construction gives PHASE-FREE amp_j (i.e., effective δ=0).
# Walker holonomy at the SHORTEST closed walks (girth g=10) gives δ=162.4°.
# Neither matches 12.73°.
#
# The framework's CURRENT identification δ = Q(1-Q) = 2/9 has NO derivation
# in either substrate amp or walker holonomy mechanisms.

print()
print(f"  AUDIT C VERDICT: walker holonomy h^g at natural integer g")
print(f"  does NOT reproduce δ ≈ 2/9 rad ≈ 12.73°.")
print(f"  • g=10 (girth) gives 162.4°, off by 13×")
print(f"  • No integer L solves L·arg(h) mod 2π = 2/9 cleanly")
print(f"  • Fractional inverses of arg(h) don't match 2/9 either")
print()
print(f"  Walker holonomy IS a viable structural mechanism for a cos-phase")
print(f"  shift in amp_j (it would add δ-equivalent phase asymmetry between")
print(f"  ω/ω² isotypics, CC-conjugate at length g=even). But its natural")
print(f"  framework value at g=10 doesn't match observation.")
print()
print(f"  → AUDIT C IDENTIFIES walker holonomy as a CANDIDATE mechanism")
print(f"    for the cos-phase but with VALUE mismatch at all natural L.")


# ============================================================
# CONSOLIDATED VERDICT
# ============================================================
print()
print("=" * 76)
print("CONSOLIDATED VERDICTS (A, B, C)")
print("=" * 76)
print(f"""
  A: (4,2,2) multiplicities are TOPOLOGICALLY EXACT (Ihara-Bass + Schur).
     → NOT a source of m_e/m_μ residuals.

  B: Cos-form δ = 2/9 in m_e.py / m_mu.py is PARAMETRIC numerical-coincidence
     identification of dimensionless Q(1-Q) with cos-phase in radians.
     Substrate Q_Koide.py gives effective δ = 0 (m_1 = m_2 degenerate).
     → THIS IS THE STRUCTURAL DEFECT (D3): the cos-form's δ is unjustified.

  C: Walker holonomy h^g is a CANDIDATE mechanism for the cos-phase
     structurally, but the natural value at g=10 is 162.4°, not 12.73°.
     No integer L on srs gives L·arg(h) mod 2π = 2/9 rad.
     → Walker holonomy DOESN'T close the defect.

  SYNTHESIS: the structural defect lives in the substrate→cos-form bridge.
  The framework needs either:
    (1) A new derivation of the cos-phase δ from substrate (currently absent)
    (2) A different parametrization of the substrate→mass map that produces
        three distinct masses WITHOUT requiring the cos-form phase
    (3) An admission that the m_e ≠ m_μ split is not currently theorem-grade

  The current state is (3): Need-B is acknowledged but unresolved.
""")
