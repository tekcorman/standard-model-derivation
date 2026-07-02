#!/usr/bin/env python3
"""
Theorem G2: Cl(0,2) boolean edge structure from A1+A3+A4.

Each undirected edge {u,v} of the srs lattice carries a Cl(0,2) algebra
from A1 (toggle involution) + A4 (CAR at k*-valent nodes) + A3 (F=C).
The minimal faithful complex representation has dimension 2.
Therefore n_channels = 2 is STRICT-SOLID under A1+A3+A4.

Derivation chain:
  A1: T_{(u,v)}^2 = T_{(v,u)}^2 = I      (toggle involutions)
  A4: {T_{(u,v)}, T_{(v,u)}} = 0           (CAR at shared vertices u and v)
  A3: field F = C (complex); set gamma_j = i*T_j
      => gamma_j^2 = (i T_j)^2 = -T_j^2 = -I   (signature -1)
  => gamma_1, gamma_2: gamma_j^2 = -I, {gamma_1, gamma_2} = 0
  => Cl(0,2) algebra generators

  Cl(0,2) over R  is isomorphic to H (quaternions, dim 4 over R)
  Cl(0,2) over C  is isomorphic to M_2(C) (2x2 matrices, dim 4 over C)

  Minimal faithful C-rep of M_2(C) has dim = 2.
  => n_channels = 2  (STRICT-SOLID under A1+A3+A4)

Adoption residual:
  Identifying C^2 with the SU(2)_L Higgs doublet requires
  ADOPTED-B3 (spinor-fermion identification, sprint_9_kickoff.md).

Closes: BLOCK-2 of an internal working note (sub-claims A, B, C).
Sub-claim D (hypercharge Y=+1/2) still requires ADOPTED-B3.

References:
  A1: docs/framework/framework_axioms.md §2
  A3: docs/framework/framework_axioms.md §4
  A4: docs/framework/framework_axioms.md §5
  Clifford algebra M_2(C): Lounesto 2001, §15.3
  Quaternion isomorphism: Porteous 1995, §15.1
"""

import numpy as np
import numpy.linalg as la

# Pauli matrices and identity
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)


# ── Jordan-Wigner realisation of the two directed edges ──────────────────────
#
# Physical setup: undirected edge {u,v} in the srs K_4-quotient.
#   e1 = directed edge (u,v)   e2 = directed edge (v,u)
#
# Both e1 and e2 are incident to vertex u (and to vertex v).
# A4 (CAR at k*=3-valent vertex u):
#   {gamma_{e1}, gamma_{e2}} = 2 * delta_{e1,e2} * I = 0  (e1 != e2)
#
# Majorana-form toggle operators in 2-mode fermionic Fock space:
#   T1 = a + a†  (mode 1; no Jordan-Wigner string needed for first mode)
#   T2 = b + b†  (mode 2; Jordan-Wigner string of sz for mode 1)
#
# Explicit 4x4 form using Kronecker products (basis: |00>,|10>,|01>,|11>):
T1 = np.kron(sx, I2)   # T_{(u,v)} = sx ⊗ I
T2 = np.kron(sz, sx)   # T_{(v,u)} = sz ⊗ sx  (with J-W string)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: A1 verification — T_e^2 = I
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART 1: A1 — T_e^2 = I ===")

T1_sq = T1 @ T1
T2_sq = T2 @ T2
err1 = la.norm(T1_sq - I4)
err2 = la.norm(T2_sq - I4)
print(f"  ||T1^2 - I4|| = {err1:.2e}  (tolerance 1e-14)")
print(f"  ||T2^2 - I4|| = {err2:.2e}  (tolerance 1e-14)")
assert err1 < 1e-14, f"T1^2 != I: residual {err1}"
assert err2 < 1e-14, f"T2^2 != I: residual {err2}"
print("  PASS\n")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: A4 verification — {T1, T2} = 0  (CAR for edges at same vertex)
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART 2: A4 — {T_{(u,v)}, T_{(v,u)}} = 0 ===")

anticomm_T = T1 @ T2 + T2 @ T1
err_A4 = la.norm(anticomm_T)
print(f"  ||{{T1, T2}}|| = {err_A4:.2e}  (tolerance 1e-14)")
assert err_A4 < 1e-14, f"CAR failed: residual {err_A4}"
print("  PASS\n")


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Cl(0,2) generators from A3 (F = C)
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART 3: Cl(0,2) generators via A3 (field C) ===")

# A3: F = C allows multiplication by i.
# gamma_j = i * T_j  =>  gamma_j^2 = -T_j^2 = -I  (signature -1)
g1 = 1j * T1   # gamma_1
g2 = 1j * T2   # gamma_2

err_g1_sq = la.norm(g1 @ g1 + I4)
err_g2_sq = la.norm(g2 @ g2 + I4)
anticomm_g = g1 @ g2 + g2 @ g1
err_anti = la.norm(anticomm_g)

print(f"  ||gamma_1^2 + I|| = {err_g1_sq:.2e}  (signature -1)")
print(f"  ||gamma_2^2 + I|| = {err_g2_sq:.2e}  (signature -1)")
print(f"  ||{{gamma_1, gamma_2}}|| = {err_anti:.2e}  (anticommute)")
assert err_g1_sq < 1e-14
assert err_g2_sq < 1e-14
assert err_anti < 1e-14
print("  PASS: gamma_1, gamma_2 are Cl(0,2) generators\n")

# Derived element: gamma_12 = gamma_1 * gamma_2  (acts as 'k' in quaternions)
g12 = g1 @ g2
err_g12_sq = la.norm(g12 @ g12 + I4)
err_anti_12_1 = la.norm(g12 @ g1 + g1 @ g12)
err_anti_12_2 = la.norm(g12 @ g2 + g2 @ g12)
print(f"  gamma_12 = gamma_1*gamma_2:")
print(f"  ||gamma_12^2 + I||         = {err_g12_sq:.2e}  (k^2 = -1)")
print(f"  ||{{gamma_12, gamma_1}}||  = {err_anti_12_1:.2e}  (ij+ji = 0)")
print(f"  ||{{gamma_12, gamma_2}}||  = {err_anti_12_2:.2e}  (kj+jk = 0)")
assert err_g12_sq < 1e-14
assert err_anti_12_1 < 1e-14
assert err_anti_12_2 < 1e-14
print("  PASS: quaternion algebra H satisfied (i^2=j^2=k^2=-1, ij+ji=0, etc.)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Algebra dimension — Cl(0,2)_C has C-dimension 4, isomorphic to M_2(C)
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART 4: Cl(0,2)_C ≅ M_2(C), dim_C = 4 ===")

# Basis: {I, gamma_1, gamma_2, gamma_12}  should be C-linearly independent.
basis_4d = [I4, g1, g2, g12]
labels = ["I", "gamma_1", "gamma_2", "gamma_12"]

# Gram matrix under the Hilbert-Schmidt inner product (A,B) = tr(A†B)/4
G4 = np.zeros((4, 4), dtype=complex)
for i, A in enumerate(basis_4d):
    for j, B in enumerate(basis_4d):
        G4[i, j] = np.trace(A.conj().T @ B) / 4.0

evals_G4 = np.sort(np.abs(la.eigvalsh(0.5*(G4 + G4.conj().T))))
rank_4 = np.linalg.matrix_rank(G4, tol=1e-10)
print(f"  Gram matrix rank = {rank_4}  (need 4 for linear independence)")
print(f"  Gram eigenvalues (abs): {evals_G4}")
assert rank_4 == 4, f"Basis not 4-dimensional: rank {rank_4}"
print("  PASS: {I, gamma_1, gamma_2, gamma_12} are C-linearly independent")
print("  => Cl(0,2)_C is 4-dimensional over C, isomorphic to M_2(C)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Part 5: Minimal faithful C-representation has dimension 2
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART 5: Minimal faithful C-rep of Cl(0,2)_C has dim = 2 ===")

# Lower bound: dim >= 2.
# If dim = 1, then gamma_1 and gamma_2 are scalars in C with gamma_j^2 = -1
# and {gamma_1, gamma_2} = 0.  On C^1: {a,b} = 2ab = 0 requires a=0 or b=0,
# so one generator acts as 0 — not faithful.  Therefore dim >= 2.

# Construct the explicit 2-dim faithful representation:
#   gamma_1 ↦ i*sigma_x     gamma_2 ↦ i*sigma_z
# Verification: (i sx)^2 = -I, (i sz)^2 = -I, {i sx, i sz} = -(sx sz + sz sx) = 0.
g1_2d = 1j * sx
g2_2d = 1j * sz
I2c   = np.eye(2, dtype=complex)
g12_2d = g1_2d @ g2_2d

err_2d_g1sq   = la.norm(g1_2d @ g1_2d + I2c)
err_2d_g2sq   = la.norm(g2_2d @ g2_2d + I2c)
err_2d_anti   = la.norm(g1_2d @ g2_2d + g2_2d @ g1_2d)
print(f"  2-dim rep: gamma_1 = i*sigma_x, gamma_2 = i*sigma_z")
print(f"  ||gamma_1^2 + I2|| = {err_2d_g1sq:.2e}")
print(f"  ||gamma_2^2 + I2|| = {err_2d_g2sq:.2e}")
print(f"  ||{{gamma_1,gamma_2}}|| = {err_2d_anti:.2e}")
assert err_2d_g1sq < 1e-14
assert err_2d_g2sq < 1e-14
assert err_2d_anti < 1e-14

# Verify faithfulness: {I2, g1_2d, g2_2d, g12_2d} must be C-linearly independent.
basis_2d = [I2c, g1_2d, g2_2d, g12_2d]
G2 = np.zeros((4, 4), dtype=complex)
for i, A in enumerate(basis_2d):
    for j, B in enumerate(basis_2d):
        G2[i, j] = np.trace(A.conj().T @ B) / 2.0

rank_2d = np.linalg.matrix_rank(G2, tol=1e-10)
print(f"  Faithful check: Gram rank = {rank_2d}  (need 4)")
assert rank_2d == 4, f"2-dim rep not faithful: rank {rank_2d}"
print("  PASS: 2-dim rep is faithful\n")

print("  Lower bound: dim=1 impossible (scalar CAR forces a generator to 0).")
print("  Upper bound: 2-dim faithful rep achieved by gamma_1↦iσₓ, gamma_2↦iσᶻ.")
print()
print("  *** n_channels = dim(minimal faithful C-rep of Cl(0,2)_C) = 2 ***")
print("      STRICT-SOLID under A1 + A3 + A4\n")


# ─────────────────────────────────────────────────────────────────────────────
# Part 6: Fock-space verification (A4 context: where T1, T2 live)
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART 6: Fock-space representation decomposes as C^4 = 2 × C^2 ===")

# The 4-dim Fock space is a reducible rep of Cl(0,2)_C ≅ M_2(C).
# A 4-dim faithful rep of M_2(C) must decompose as 2+2 (since the only
# irrep of M_2(C) over C has dim 2).
#
# Verify: commutant of the algebra {g1, g2} in End(C^4) has C-dim = 4.
# (By the double centralizer theorem, commutant ≅ M_2(C) on multiplicity space.)
#
# Method: [A, M] = 0  ⟺  (I⊗A - A^T⊗I) vec(M) = 0
# where vec(M) is the column-stacking of M (16-vector).
# Build the 16×16 linear system and find its null space.

print("  Finding commutant of Cl(0,2) via null-space of commutator map...")

n = 4
I16 = np.eye(n*n, dtype=complex)

# Commutator [g, M] = 0  ⟺  (I_4 ⊗ g - g^T ⊗ I_4) vec(M) = 0
constraint_rows = []
for G in [g1, g2]:
    K = np.kron(I4, G) - np.kron(G.T, I4)   # 16×16 matrix
    constraint_rows.append(K)

# Stack both constraint matrices (32×16 total)
A_sys = np.vstack(constraint_rows)

# SVD to find null space
_, s, Vh = la.svd(A_sys)
# Null space = rows of Vh corresponding to singular values < tol
tol_sv = 1e-10
null_dim = np.sum(s < tol_sv) + (Vh.shape[0] - len(s))
# More carefully: A_sys is 32×16, so we need the null space in C^16
# The rank of A_sys gives null dim = 16 - rank.
rank_A = np.sum(s > tol_sv)
commutant_dim = n*n - rank_A

print(f"  Constraint matrix rank = {rank_A}  (out of 16)")
print(f"  Commutant C-dimension  = {commutant_dim}  (expect 4 for 2+2 decomposition)")
assert commutant_dim == 4, \
    f"Commutant has wrong dimension {commutant_dim}, expected 4"
print("  PASS: commutant is 4-dimensional => Fock rep = 2 × (2-dim irrep)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("THEOREM G2 — SUMMARY")
print("=" * 62)
print("""
  A1 + A4 + A3  ⟹  n_channels = 2  (STRICT-SOLID)

  Step 1 (A1):   T_{(u,v)}² = T_{(v,u)}² = I   [toggle involutions]
  Step 2 (A4):   {T_{(u,v)}, T_{(v,u)}} = 0    [CAR at shared vertex]
  Step 3 (A3):   γ_j = i T_j  →  γ_j² = -I     [complex field, sig −1]
  Step 4:        γ₁, γ₂ generate Cl(0,2) ≅ ℍ over ℝ, ≅ M₂(ℂ) over ℂ
  Step 5:        min faithful ℂ-rep of M₂(ℂ) has dim = 2  [shown above]
  ⟹             n_channels = 2  STRICT-SOLID

  Adoption residual:
    Identifying ℂ² with the SU(2)_L Higgs doublet
    requires ADOPTED-B3 (spinor-fermion identification).

  Closes: BLOCK-2 sub-claims A, B, C of theorem_higgs_vev_scoping.md.
  Sub-claim D (hypercharge Y = +1/2) still requires ADOPTED-B3.
""")
