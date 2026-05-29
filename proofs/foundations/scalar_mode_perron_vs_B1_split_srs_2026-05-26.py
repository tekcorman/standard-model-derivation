#!/usr/bin/env python3
"""
Split the 2 J=+1 scalar modes in the u=±1 eigenspace of B on srs (K_4) into
"Perron-adjacency-derived scalar" vs "B¹ vertex-coboundary residue."

CONTEXT
-------
The Wilson-loop probe (`H1_sub_bundle_mode_count_srs_2026-05-26.py`) found:
- 3 cycle modes at u=±1 (all J=-1), Wilson-loop carrying.
- 2 scalar modes at u=±1 (both J=+1), Wilson-loop kernel.

Route H asserts the 4-dim "bipartite-factor sector" = 3 cycle + 1 scalar
(B¹ vertex-coboundary residue), plus 1 separate "Perron-adjacency scalar"
that's excluded from gauge 1-point coupling.

For the structural picture (c_EM = 4/12 = 1/3, c_color = 3/12 = 1/4) to be
substrate-derived, we need to identify the 1 B¹ scalar mode that's IN the
EM-block bipartite-factor sector vs the 1 Perron scalar that's NOT.

The standard Bass formula T(φ)(e) = u·φ(h(e)) - φ(t(e)) applied to the
Perron adjacency eigenvector at u=+1 gives zero on K_4 (verified by hand).
So we need a different structural decomposition.

This probe uses the SYMMETRIC vertex-lift S(f)(e) = f(h(e)) + f(t(e)),
which is injective on K_4 (image dim = |V| = 4 within the 6-dim J=+1
sector). The 2 J=+1 scalar modes both lie in image(S) since they have
no Wilson-loop holonomy. The probe asks:

  Does projecting the 2-dim scalar subspace onto image(S_Perron) (the
  1-dim subspace lifted from the Perron adjacency eigenvector) give
  a clean (1, 1) split?

If YES: one mode is "Perron-derived" (= image of uniform vertex
function under S), the other is "B¹ residue" (= image of non-Perron
vertex content under S). Route H's structural split is then derivable.

If NO: the 2 scalar modes are an inseparable pair, and Route H's
"4 + 1" decomposition is conventional, not structural.
"""

import math
import numpy as np
from itertools import permutations

# ------------------------------------------------------------------
# 1. K_4 setup
# ------------------------------------------------------------------
N_V = 4
vertices = list(range(N_V))
directed_edges = [(u, v) for u in vertices for v in vertices if u != v]
N_E = len(directed_edges) // 2     # 6
N_DE = len(directed_edges)         # 12
e2i = {e: i for i, e in enumerate(directed_edges)}

# Hashimoto B
B = np.zeros((N_DE, N_DE), dtype=int)
for i, (u, v) in enumerate(directed_edges):
    for w in vertices:
        if w == u or w == v:
            continue
        B[i, e2i[(v, w)]] = 1

# Edge-reversal J
J = np.zeros((N_DE, N_DE), dtype=int)
for i, (u, v) in enumerate(directed_edges):
    J[i, e2i[(v, u)]] = 1

# Adjacency A
A = np.zeros((N_V, N_V), dtype=int)
for u in vertices:
    for v in vertices:
        if u != v:
            A[u, v] = 1

# Triangles for Wilson-loop check
triangles = []
for omit in range(N_V):
    others = [v for v in range(N_V) if v != omit]
    a, b, c = others
    triangles.append([(a, b), (b, c), (c, a)])

# ------------------------------------------------------------------
# 2. Find the 5-dim u=±1 eigenspace
# ------------------------------------------------------------------
eigvals, eigvecs = np.linalg.eig(B.astype(float))
mask_plus  = np.abs(eigvals - 1.0) < 1e-8
mask_minus = np.abs(eigvals + 1.0) < 1e-8
V_plus  = np.real_if_close(eigvecs[:, mask_plus], tol=1000)
V_minus = np.real_if_close(eigvecs[:, mask_minus], tol=1000)
if np.iscomplexobj(V_plus):
    V_plus = np.real(V_plus)
if np.iscomplexobj(V_minus):
    V_minus = np.real(V_minus)
# Orthonormalize
V_plus,  _ = np.linalg.qr(V_plus)
V_minus, _ = np.linalg.qr(V_minus)
V_pm = np.concatenate([V_plus, V_minus], axis=1)
print("="*78)
print(" Scalar-mode split: Perron-adjacency vs B¹ vertex-coboundary on srs (K_4)")
print("="*78)
print(f"  dim(u = +1) = {V_plus.shape[1]}, dim(u = -1) = {V_minus.shape[1]}, total = {V_pm.shape[1]}")

# ------------------------------------------------------------------
# 3. Wilson-loop split: image = cycle modes, kernel = scalar modes
# ------------------------------------------------------------------
H_mat = np.array([[sum(V_pm[e2i[e], k] for e in C) for C in triangles]
                  for k in range(V_pm.shape[1])])
U_h, S_h, _ = np.linalg.svd(H_mat, full_matrices=True)
rank_h = int(np.sum(S_h > 1e-8))
V_BM       = V_pm @ U_h[:, :rank_h]              # cycle modes (dim 3)
V_scalar   = V_pm @ U_h[:, rank_h:]              # scalar modes (dim 2)
print(f"  cycle modes (Wilson-loop carrying): dim {V_BM.shape[1]}")
print(f"  scalar modes (Wilson-loop kernel):   dim {V_scalar.shape[1]}")
print()

# ------------------------------------------------------------------
# 4. Symmetric vertex-lift map S: ℂ^V → ℂ^(2|E|), S(f)(e) = f(h(e)) + f(t(e))
# ------------------------------------------------------------------
S = np.zeros((N_DE, N_V))
for i, (u, v) in enumerate(directed_edges):
    S[i, u] += 1     # tail contribution
    S[i, v] += 1     # head contribution
# Columns of S span image(S). Verify dim:
S_rank = np.linalg.matrix_rank(S)
print(f"  Symmetric vertex-lift S: image dim = {S_rank} (expected {N_V} = injective on K_4)")
# Orthonormal basis of image(S)
Q_S, _ = np.linalg.qr(S)
V_Slift = Q_S[:, :S_rank]
print(f"  image(S) is a {S_rank}-dim subspace within the J=+1 sector (dim 6).")
print()

# ------------------------------------------------------------------
# 5. Are the 2 scalar modes inside image(S)?
# ------------------------------------------------------------------
projector_Slift = V_Slift @ V_Slift.T
inside_check = []
for k in range(V_scalar.shape[1]):
    v = V_scalar[:, k]
    v_proj = projector_Slift @ v
    overlap_sq = np.dot(v_proj, v_proj) / np.dot(v, v)
    inside_check.append(overlap_sq)
    print(f"  scalar mode {k}: |proj onto image(S)|² / |v|² = {overlap_sq:.6f}  "
          f"({'IN image(S)' if overlap_sq > 0.99 else 'PARTIAL' if overlap_sq > 0.01 else 'NOT in image(S)'})")
print()

# ------------------------------------------------------------------
# 6. Split image(S) by Perron vs non-Perron vertex content
# ------------------------------------------------------------------
# Adjacency eigendecomposition
A_eigvals, A_eigvecs = np.linalg.eigh(A.astype(float))
order = np.argsort(-A_eigvals)
A_eigvals = A_eigvals[order]
A_eigvecs = A_eigvecs[:, order]
print(f"  Adjacency spectrum of K_4: {np.round(A_eigvals, 4).tolist()}")
phi_Perron = A_eigvecs[:, 0]      # λ = +3
phi_nonPerron = A_eigvecs[:, 1:]  # λ = -1 eigenspace (dim 3)

# Lift Perron via S
S_phi_Perron = S @ phi_Perron
S_phi_Perron_norm = S_phi_Perron / np.linalg.norm(S_phi_Perron)
print(f"  S(φ_Perron)(e) = {np.round(S_phi_Perron, 4).tolist()[:6]}... (uniform direction)")
# It should be the uniform vector since Perron is uniform on K_4
is_uniform = np.allclose(S_phi_Perron / S_phi_Perron[0], np.ones(N_DE))
print(f"  Is S(φ_Perron) ∝ uniform-on-edges? {is_uniform}  (expected YES on K_4)")
print()

# Lift non-Perron via S
S_phi_nonPerron = S @ phi_nonPerron   # 12 × 3
Q_nonPerron, _ = np.linalg.qr(S_phi_nonPerron)
V_S_nonPerron = Q_nonPerron[:, :3]
print(f"  Image of non-Perron vertex content under S has dim 3 (within image(S) of dim 4).")
print()

# ------------------------------------------------------------------
# 7. KEY TEST: do the 2 scalar modes split as (Perron-derived, non-Perron-derived)?
# ------------------------------------------------------------------
# Project each scalar mode onto S(φ_Perron) direction
proj_Perron = []
for k in range(V_scalar.shape[1]):
    v = V_scalar[:, k]
    coeff = np.dot(v, S_phi_Perron_norm)
    overlap_sq = coeff**2 / np.dot(v, v)
    proj_Perron.append(overlap_sq)
    print(f"  scalar mode {k}: |proj onto S(φ_Perron)|² / |v|² = {overlap_sq:.6f}")
print()

# What we expect for a clean (Perron, B¹) split:
# - One scalar mode has overlap_sq ≈ 1 with S(φ_Perron) → this is the Perron-adjacency scalar
# - The other has overlap_sq ≈ 0 (orthogonal) → this is the B¹ residue from non-Perron content
# A clean split requires these two overlaps to sum to ~1 and be ~(1, 0) or near it.

# Alternative: diagonalize the projection operator on V_scalar
# P_Perron_on_scalar = V_scalar.T @ |S_phi_Perron⟩⟨S_phi_Perron|/||²  @ V_scalar
M = V_scalar.T @ np.outer(S_phi_Perron_norm, S_phi_Perron_norm) @ V_scalar
M_eigvals, M_eigvecs = np.linalg.eigh(M)
print(f"  Diagonalize S(φ_Perron)-projection on V_scalar:")
print(f"    eigenvalues = {np.round(M_eigvals, 6).tolist()}")
print()

# ------------------------------------------------------------------
# 8. Honest verdict
# ------------------------------------------------------------------
print("="*78)
print(" VERDICT")
print("="*78)
sum_overlaps = sum(proj_Perron)
max_eig = max(M_eigvals)
print(f"  Total projection of V_scalar onto S(φ_Perron): {sum_overlaps:.6f}")
print(f"  Max eigenvalue of projection (in V_scalar basis): {max_eig:.6f}")
print()
if max_eig > 0.99:
    print("  ✓ CLEAN SPLIT: one scalar mode aligns ~exactly with S(φ_Perron)")
    print("    (Perron-adjacency-derived scalar), the other is orthogonal")
    print("    (B¹ vertex-coboundary residue from non-Perron vertex content).")
    print("  → Route H's '4 + 1' decomposition IS substrate-derivable.")
elif max_eig > 0.5:
    print(f"  PARTIAL: the projection has max eigenvalue {max_eig:.3f} < 1.")
    print("    The Perron-direction picks out a preferred axis but doesn't")
    print("    give a unit-rank split. Structural identification incomplete.")
else:
    print(f"  NO CLEAN SPLIT: max eigenvalue {max_eig:.3f} of the Perron-projection")
    print("    on V_scalar is far from 1. The 2 scalar modes are NOT naturally")
    print("    split into Perron-derived vs non-Perron-derived by S(φ_Perron)")
    print("    alone — Route H's '4 + 1' is conventional, not structural.")
    print()
    print("  → The structural derivation of c_EM = 1/3 vs c_color = 1/4 via")
    print("    B¹-vs-Perron scalar mode separation is NOT achieved here.")
print("="*78)

# Sanity: what IS the structure of V_scalar then?
print()
print(" Additional diagnostic — what V_scalar IS:")
# Each scalar mode in basis of {S_phi_Perron, V_S_nonPerron}
print(" Project each scalar mode onto Perron-lift vs non-Perron-lift basis:")
for k in range(V_scalar.shape[1]):
    v = V_scalar[:, k]
    cP = np.dot(v, S_phi_Perron_norm)
    cNP_proj = V_S_nonPerron.T @ v
    cNP_sq = np.dot(cNP_proj, cNP_proj)
    total_sq = cP**2 + cNP_sq
    print(f"   mode {k}: c_Perron² = {cP**2:.4f}, |c_nonPerron|² = {cNP_sq:.4f},  "
          f"sum = {total_sq:.4f}  (1.0 means fully in image(S))")
