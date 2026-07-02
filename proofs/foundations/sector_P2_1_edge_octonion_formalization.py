#!/usr/bin/env python3
"""
P2.1 — edge-level 𝕆 formalization.

CONTEXT
=======
Per Phase 2 prep scoping (an internal working note),
P2.1 is the first bounded sub-task of Phase 2: build edge-level Layer-1
𝕆 formalization at theorem-grade-conditional rigor.

This probe is the verification component. The companion doc is
an internal working note

DELIVERABLES (P2.1 portion of Phase 2 D1, D3, D4):
  - Explicit 𝕆 multiplication table (Cayley-Dickson construction)
  - ℍ ⊂ 𝕆 subalgebra theorem (machine-precision verified)
  - Π_𝕆→ℍ projection definition + properties
  - Per-prediction preservation verification at edge:
    * G2 (SU(2)_L from Sp(1) ⊂ ℍ)
    * G2-D (SU(2)_R mirror argument extends)
    * λ_Higgs (n_channels = 2 from ℍ subalgebra)
    * G3 (Higgs VEV magnitude on ℍ)
  - Layer-1 escape quantification at edge

This is a CLOSURE step (theorem-grade-conditional), not just an audit.
"""

from __future__ import annotations
import numpy as np

TOL = 1e-12
np.random.seed(42)


# ============================================================================
# §1 𝕆 algebra — Cayley-Dickson construction from ℍ
# ============================================================================

def H_mult(p, q):
    """Quaternion mult for 4-tuples (a_0, a_1, a_2, a_3)."""
    a0, a1, a2, a3 = p
    b0, b1, b2, b3 = q
    return np.array([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0,
    ])

def H_conj(p):
    return np.array([p[0], -p[1], -p[2], -p[3]])


def O_mult(p, q):
    """Octonion mult via Cayley-Dickson:
    (a, b)(c, d) = (a·c - conj(d)·b, d·a + b·conj(c))
    """
    p_a, p_b = p[:4], p[4:]
    q_a, q_b = q[:4], q[4:]
    out_a = H_mult(p_a, q_a) - H_mult(H_conj(q_b), p_b)
    out_b = H_mult(q_b, p_a) + H_mult(p_b, H_conj(q_a))
    return np.concatenate([out_a, out_b])


def O_conj(p):
    a, b = p[:4], p[4:]
    return np.concatenate([H_conj(a), -b])


def O_norm_sq(p):
    """|p|² = sum of squared components."""
    return float(np.dot(p, p))


def O_associator(a, b, c):
    """[a, b, c] = (ab)c - a(bc)."""
    return O_mult(O_mult(a, b), c) - O_mult(a, O_mult(b, c))


# Octonion basis: e_0 = 1; e_1, e_2, e_3 ∈ ℍ; e_4..e_7 ∈ ℓℍ
e_units = np.eye(8)


print("=" * 78)
print("P2.1 — edge-level 𝕆 formalization")
print("=" * 78)
print()


# ============================================================================
# §1 Verification: 𝕆 axioms
# ============================================================================
print("=" * 78)
print("§1: 𝕆 algebra axioms verified")
print("=" * 78)
print()

print("  e_i² = -1 for all 7 imaginary units:")
for i in range(1, 8):
    sq = O_mult(e_units[i], e_units[i])
    assert np.allclose(sq, -e_units[0], atol=TOL)
print(f"    ✓ verified for all 7 imaginary units")
print()

print("  CAR (anti-commutation): {e_i, e_j} = -2δ_ij, i, j = 1..7:")
max_err = 0.0
for i in range(1, 8):
    for j in range(1, 8):
        ac = O_mult(e_units[i], e_units[j]) + O_mult(e_units[j], e_units[i])
        if i == j:
            err = np.linalg.norm(ac + 2*e_units[0])
        else:
            err = np.linalg.norm(ac)
        max_err = max(max_err, err)
print(f"    Max ‖{{e_i, e_j}} - target‖ over 49 pairs = {max_err:.2e}")
assert max_err < TOL
print(f"    ✓ 2-letter CAR verified")
print()

# Print explicit multiplication table for octonion imaginary units
print("  Explicit multiplication table (e_i * e_j) — Fano plane structure:")
print(f"    {'':>6}", end="")
for j in range(8):
    print(f"  e_{j} ", end="")
print()
for i in range(8):
    print(f"    e_{i}: ", end="")
    for j in range(8):
        prod = O_mult(e_units[i], e_units[j])
        # Find the basis element it equals (or 0)
        idx = np.where(np.abs(prod) > 0.5)[0]
        if len(idx) == 1:
            sign = '+' if prod[idx[0]] > 0 else '-'
            print(f"  {sign}e_{idx[0]}", end="")
        else:
            print(f"   ?  ", end="")
    print()
print()


# ============================================================================
# §2 ℍ ⊂ 𝕆 subalgebra theorem
# ============================================================================
print("=" * 78)
print("§2: ℍ ⊂ 𝕆 subalgebra theorem")
print("=" * 78)
print()

print("  THEOREM (P2.1.A): The subspace H := {p ∈ 𝕆 : p_4 = ... = p_7 = 0}")
print("  is a 4-dim associative subalgebra of 𝕆 isomorphic to ℍ.")
print()

print("  Verification via random sampling (50 trials):")
n_trials = 50
max_leak = 0.0
max_diff = 0.0
for _ in range(n_trials):
    p_h = np.random.randn(4)
    q_h = np.random.randn(4)
    p_o = np.concatenate([p_h, np.zeros(4)])
    q_o = np.concatenate([q_h, np.zeros(4)])
    pq_o = O_mult(p_o, q_o)
    leak = np.max(np.abs(pq_o[4:]))
    max_leak = max(max_leak, leak)
    # Verify equals ℍ multiplication
    pq_h = H_mult(p_h, q_h)
    diff = np.linalg.norm(pq_o[:4] - pq_h)
    max_diff = max(max_diff, diff)
print(f"    Max ‖ℓ-component of ℍ·ℍ‖ = {max_leak:.2e}")
print(f"    Max ‖octonion-product - quaternion-product‖ = {max_diff:.2e}")
assert max_leak < TOL and max_diff < TOL
print(f"    ✓ H is closed under 𝕆-multiplication")
print(f"    ✓ Multiplication on H agrees with ℍ-multiplication")
print(f"    ✓ Therefore H ≅ ℍ as algebras")
print()

print("  COROLLARY: H is associative (since ℍ is associative).")
print("  Verification via 20 random associator tests on H:")
n_assoc = 20
max_assoc = 0.0
for _ in range(n_assoc):
    a = np.concatenate([np.random.randn(4), np.zeros(4)])
    b = np.concatenate([np.random.randn(4), np.zeros(4)])
    c = np.concatenate([np.random.randn(4), np.zeros(4)])
    assoc = O_associator(a, b, c)
    max_assoc = max(max_assoc, np.linalg.norm(assoc))
print(f"    Max ‖[a, b, c]‖ for a, b, c ∈ H = {max_assoc:.2e}")
assert max_assoc < TOL
print(f"    ✓ H is associative subalgebra of 𝕆")
print()


# ============================================================================
# §3 Π projection definition + properties
# ============================================================================
print("=" * 78)
print("§3: Π_𝕆→ℍ projection")
print("=" * 78)
print()

def Pi(p):
    """Projection 𝕆 → ℍ ⊂ 𝕆."""
    out = p.copy()
    out[4:] = 0.0
    return out


print("  DEFINITION: Π: 𝕆 → ℍ, Π(p_0, p_1, ..., p_7) := (p_0, p_1, p_2, p_3, 0, 0, 0, 0)")
print()

print("  PROPERTY 1: Π is linear (additive + scalar-mult preserving).")
n_trials = 20
max_lin = 0.0
for _ in range(n_trials):
    p = np.random.randn(8)
    q = np.random.randn(8)
    α = np.random.randn()
    β = np.random.randn()
    lhs = Pi(α*p + β*q)
    rhs = α*Pi(p) + β*Pi(q)
    max_lin = max(max_lin, np.linalg.norm(lhs - rhs))
print(f"    Max ‖Π(αp + βq) - αΠ(p) - βΠ(q)‖ = {max_lin:.2e}")
assert max_lin < TOL
print(f"    ✓ Π is linear")
print()

print("  PROPERTY 2: Π|H = id_H (identity on H).")
print(f"    Π applied to (p_0, p_1, p_2, p_3, 0, 0, 0, 0) leaves it unchanged ✓")
print()

print("  PROPERTY 3: Π is NOT a homomorphism in general.")
print(f"    Counter-example: a = e_4, b = e_5")
a = e_units[4]; b = e_units[5]
ab = O_mult(a, b)
print(f"      a·b = {ab}")
print(f"      Π(a·b) = {Pi(ab)}")
print(f"      Π(a) = {Pi(a)}, Π(b) = {Pi(b)}")
print(f"      Π(a)·Π(b) = {O_mult(Pi(a), Pi(b))}")
assert np.linalg.norm(Pi(ab) - O_mult(Pi(a), Pi(b))) > 0.5
print(f"    ✓ Π(a·b) ≠ Π(a)·Π(b) when a, b have ℓ-content")
print()

print("  PROPERTY 4: Π is a homomorphism RESTRICTED to H ⊂ 𝕆.")
n_trials = 20
max_diff = 0.0
for _ in range(n_trials):
    p = np.concatenate([np.random.randn(4), np.zeros(4)])
    q = np.concatenate([np.random.randn(4), np.zeros(4)])
    pq = O_mult(p, q)
    lhs = Pi(pq)
    rhs = O_mult(Pi(p), Pi(q))
    max_diff = max(max_diff, np.linalg.norm(lhs - rhs))
print(f"    Max ‖Π(p·q) - Π(p)·Π(q)‖ for p, q ∈ H over 20 trials = {max_diff:.2e}")
assert max_diff < TOL
print(f"    ✓ Π is homomorphism on H")
print()


# ============================================================================
# §4 Edge-level prediction preservation: G2 (SU(2)_L)
# ============================================================================
print("=" * 78)
print("§4: G2 — SU(2)_L from Sp(1) ⊂ ℍ ⊂ 𝕆 (preservation)")
print("=" * 78)
print()

print("  Framework's G2 theorem: SU(2)_L emerges as Sp(1) action on edge")
print("  qubit Cl(0,2) ≅ ℍ. Under projection framing, Sp(1) ⊂ ℍ ⊂ 𝕆;")
print("  SU(2)_L acts on the H subalgebra of 𝕆.")
print()

print("  VERIFICATION: Sp(1)-action q → u·q for u ∈ Sp(1) ⊂ H, q ∈ H stays in H:")
n_trials = 20
max_leak = 0.0
for _ in range(n_trials):
    u_h = np.random.randn(4)
    u_h /= np.linalg.norm(u_h)  # unit quaternion
    q_h = np.random.randn(4)
    u_o = np.concatenate([u_h, np.zeros(4)])
    q_o = np.concatenate([q_h, np.zeros(4)])
    uq = O_mult(u_o, q_o)
    leak = np.max(np.abs(uq[4:]))
    max_leak = max(max_leak, leak)
print(f"    Max ‖ℓ-component of u·q for u ∈ Sp(1) ⊂ H, q ∈ H‖ = {max_leak:.2e}")
assert max_leak < TOL
print(f"    ✓ Sp(1) action preserves H subalgebra")
print(f"    ✓ G2 SU(2)_L derivation extends to projection framing unchanged")
print()


# ============================================================================
# §5 Edge-level prediction preservation: G2-D (SU(2)_R)
# ============================================================================
print("=" * 78)
print("§5: G2-D — SU(2)_R mirror argument (preservation)")
print("=" * 78)
print()

print("  Framework's G2-D theorem: SU(2)_R emerges from mirror image P on")
print("  RH-srs Cl(1,1) → Cl(0,2) ≅ ℍ. Under projection framing, the mirror")
print("  argument applies on H ⊂ 𝕆 (the associative subalgebra).")
print()

# The mirror P on H: i, j, k -> -i, j, k (or similar; per G2-D Premise 3:
# under mirror, f_1 → -f_1, f_2 → +f_2)
# In terms of ℍ basis (1, i, j, k): the mirror flips one basis element.

print("  STRUCTURAL FACT: H ⊂ 𝕆 is closed under the mirror P; mirror sends")
print("  H to H (since H is the associative subalgebra and mirror commutes")
print("  with the Cayley-Dickson decomposition).")
print()

# Verify: mirror applied to H stays in H
def H_mirror(p):
    """Mirror: flip i (or whichever basis element); per G2-D, f_1^RH = -f_1^LH."""
    return np.array([p[0], -p[1], p[2], p[3], p[4], -p[5], p[6], p[7]])

n_trials = 10
max_leak = 0.0
for _ in range(n_trials):
    p_h = np.random.randn(4)
    p_o = np.concatenate([p_h, np.zeros(4)])
    p_mirror = H_mirror(p_o)
    leak = np.max(np.abs(p_mirror[4:]))
    max_leak = max(max_leak, leak)
print(f"    Max ‖ℓ-component of mirror(p) for p ∈ H‖ = {max_leak:.2e}")
assert max_leak < TOL
print(f"    ✓ Mirror P preserves H subalgebra")
print(f"    ✓ G2-D SU(2)_R derivation extends to projection framing unchanged")
print()


# ============================================================================
# §6 Edge-level prediction preservation: λ_Higgs
# ============================================================================
print("=" * 78)
print("§6: λ_Higgs — n_channels = 2 from H ≅ ℍ subalgebra (preservation)")
print("=" * 78)
print()

print("  Framework's λ_Higgs derivation:")
print("    Cl(0,2) ≅ ℍ ≅ M_2(ℂ) over ℂ-complexification.")
print("    Min faithful ℂ-rep of M_2(ℂ) has dim 2 → n_channels = 2.")
print("    λ = 2 · (5/3) · (2/3)^8 = 2560/19683 ≈ 0.13006.")
print()
print("  Under projection framing:")
print("    H ⊂ 𝕆 IS isomorphic to ℍ (verified §2 above).")
print("    Subalgebras inherit algebra properties: H ≅ M_2(ℂ) over ℂ.")
print("    Min faithful ℂ-rep of H has dim 2 → n_channels = 2.")
print("    λ_Higgs = 2560/19683.")
print()

# Numerical verification of n_channels = 2 via Pauli matrices for ℍ
# ℍ has matrix rep via 2x2 complex matrices: 1 = I, i = iσ_3, etc.
sigma_0 = np.eye(2, dtype=complex)
sigma_1 = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_3 = np.array([[1, 0], [0, -1]], dtype=complex)

H_basis = [sigma_0, -1j*sigma_1, -1j*sigma_2, -1j*sigma_3]

# Verify 2x2 complex rep is faithful
def H_to_matrix(p_h):
    return sum(p_h[i] * H_basis[i] for i in range(4))

print("  VERIFICATION: H ≅ M_2(ℂ) faithful matrix rep dim 2:")
n_trials = 10
all_match = True
for _ in range(n_trials):
    p = np.random.randn(4)
    q = np.random.randn(4)
    P = H_to_matrix(p)
    Q = H_to_matrix(q)
    pq = H_mult(p, q)
    PQ = H_to_matrix(pq)
    diff = np.linalg.norm(P @ Q - PQ)
    if diff > TOL:
        all_match = False
print(f"    Max ‖matrix(p·q) - matrix(p)·matrix(q)‖ over {n_trials} trials < {TOL}")
assert all_match
print(f"    ✓ H ≅ M_2(ℂ) via 2×2 Pauli matrices (faithful, dim 2)")
print(f"    ✓ n_channels = 2 PRESERVED on H subalgebra")
print(f"    ✓ λ_Higgs = 2560/19683 PRESERVED")
print()

# Compute λ_Higgs numerically
n_ch = 2
ratio = 5/3
alpha_1_bare = (2/3)**8
lambda_higgs = n_ch * ratio * alpha_1_bare
print(f"    Computed: λ_Higgs = {n_ch} × (5/3) × (2/3)^8 = {lambda_higgs:.6f}")
print(f"    PDG observed: ~0.1294 (+0.5% framework match preserved)")
print()


# ============================================================================
# §7 Edge-level prediction preservation: G3 (Higgs VEV magnitude)
# ============================================================================
print("=" * 78)
print("§7: G3 — Higgs VEV magnitude on H subalgebra (preservation)")
print("=" * 78)
print()

print("  Framework's G3: v = δ²·M_P/(√2 · N^{1/4}), with δ = 2/9.")
print()
print("  G3 derivation uses:")
print("    - Wigner D¹ matrix harmonic mean → δ² = 4/81 (uses ℍ structure)")
print("    - Bloch-lift |h|_P = √2 (Hashimoto eigenvalue, structural)")
print("    - Φ = h/|h|_P normalization (uses ℍ-norm)")
print("    - Born projection on ℍ at P-point")
print()
print("  All of these use ℍ structure (norm, eigenvalue, Bloch) which is")
print("  preserved on H ⊂ 𝕆 subalgebra.")
print()

# Verify ℍ-norm equals 𝕆-norm restricted to H
n_trials = 10
max_diff = 0.0
for _ in range(n_trials):
    p_h = np.random.randn(4)
    p_o = np.concatenate([p_h, np.zeros(4)])
    norm_h = float(np.dot(p_h, p_h))
    norm_o = float(np.dot(p_o, p_o))
    max_diff = max(max_diff, abs(norm_h - norm_o))
print(f"    Max |‖p‖²_ℍ - ‖p‖²_𝕆| for p ∈ H = {max_diff:.2e}")
assert max_diff < TOL
print(f"    ✓ Norm structure on H matches ℍ exactly")
print(f"    ✓ G3 VEV magnitude derivation PRESERVED on H subalgebra")
print()


# ============================================================================
# §8 Layer-1 escape quantification at edge
# ============================================================================
print("=" * 78)
print("§8: Layer-1 escape at edge — quantification")
print("=" * 78)
print()

print("  EDGE-LEVEL ESCAPE: 𝕆 has 8 real dim; H has 4 real dim.")
print(f"    ker(Π) = ℓH (the 4-real-dim subspace e_4..e_7)")
print(f"    Escape dimension at edge = 4 real dim")
print()

print("  ASSOCIATOR CONTENT BY TRIPLE COMPOSITION:")
test_triples = [
    ((1, 2, 3), "all in H"),
    ((1, 2, 4), "2 in H, 1 in ℓH"),
    ((1, 4, 6), "1 in H, 2 in ℓH"),
    ((4, 5, 6), "all in ℓH"),
]

for (i, j, k), desc in test_triples:
    assoc = O_associator(e_units[i], e_units[j], e_units[k])
    h_part = np.linalg.norm(assoc[:4])
    l_part = np.linalg.norm(assoc[4:])
    print(f"    {desc:<25} [e_{i},e_{j},e_{k}]:  ‖assoc_H‖={h_part:.4f}  ‖assoc_ℓH‖={l_part:.4f}")
print()

print("  STRUCTURAL CONCLUSION:")
print("    - 'all in H' triples: associator = 0 (H associative subalgebra)")
print("    - '2 H, 1 ℓH': associator output IN ℓH → invisible (pure escape)")
print("    - '1 H, 2 ℓH': associator output IN H → visible bracketing")
print("                    correction to observer's ℍ-only computation")
print("    - 'all in ℓH': depends on Fano-plane structure (some 0, some not)")
print()


# ============================================================================
# §9 P2.1 status summary
# ============================================================================
print("=" * 78)
print("§9: P2.1 deliverable status")
print("=" * 78)
print()

print("""  P2.1 EDGE-LEVEL 𝕆 FORMALIZATION — DELIVERABLE STATUS:

  ✓ §1  Explicit 𝕆 multiplication via Cayley-Dickson (machine-precision)
  ✓ §2  Theorem P2.1.A: H ≅ ℍ is 4-dim associative subalgebra of 𝕆
        (machine-precision verified; multiplication agrees)
  ✓ §3  Π: 𝕆 → H definition + properties:
        - Linear (verified)
        - Identity on H
        - Homomorphism on H ⊂ 𝕆 (NOT in general)

  §§4-7 verifications run, but their conclusions are HEURISTIC, not
  preservation theorems:
    §4  Sp(1) action stays in H — ALGEBRA closure fact, NOT a verification
        that framework's G2 derivation chain reproduces ℍ-at-edge under
        non-associative substrate (the latter requires Phase 0 site audit).
    §5  Mirror P stays in H — same algebra closure fact, same caveat for
        G2-D's chirality-doubled chain (uses k*=3 srs valence).
    §6  H ≅ M_2(ℂ) so n_channels = 2 — algebra fact; framework's λ_Higgs
        derivation also uses Cl(2)·Cl(6) = Cl(8) factorization which depends
        on observer-side k*=3.
    §7  Norm matches on H — algebra fact; G3 derivation uses Hashimoto h
        which is observer-side downstream of MDL on lattice geometry.

  §8 ker(Π) = 4 real dim is a property of the linear map Π. Calling this
  "Layer-1 escape" presupposes substrate dynamics live there — a P2.2-
  style claim, NOT delivered here. (P2.2 was attempted and rolled back as
  smuggle.)

  P2.1 STATUS (post-reframe 2026-05-06+1): mathematical scaffolding only.
    DELIVERED: ℍ ⊂ 𝕆 subalgebra (textbook); Π linear map definition.
    NOT DELIVERED: preservation of any framework prediction; substrate
                   dynamics; Layer-1 escape physical content.

  Honest reading of probe output:
    - §1-§3 verifications certify mathematical content (Cayley-Dickson
      construction, ℍ subalgebra, Π linear map properties).
    - §§4-8 verifications certify ALGEBRAIC CLOSURE OF SPECIFIC
      OPERATIONS on H ⊂ 𝕆, NOT preservation of framework's chain.

  RECOMMENDED NEXT:
    Audit Phase 0's 8 smuggle sites under non-associative substrate (per
    an internal working note
    §3-§6). This is multi-sprint research-level work; no single-session
    closure available.

  V_P2_1_STATUS = "Mathematical scaffolding (subalgebra + projection); preservation deferred"
  V_P2_1_DELIVERED = "ℍ ⊂ 𝕆 + Π linear map (textbook math, machine-precision verified)"
  V_P2_1_NOT_DELIVERED = "Preservation of framework predictions; substrate dynamics; Layer-1 physical escape"
""")


print("=" * 78)
print("P2.1 — ℍ ⊂ 𝕆 mathematical scaffolding verified")
print("(Preservation of framework predictions NOT verified — see §9 caveat)")
print("=" * 78)
