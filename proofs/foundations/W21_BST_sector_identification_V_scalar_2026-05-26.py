#!/usr/bin/env python3
"""
W21 — BS-T sector identification within V_scalar on K_4

CONTEXT
-------
Yesterday's session
found V_pm (5-dim u=±1 eigenspace) splits Wilson-loop-wise into:
  V_cycle  (3-dim, J=-1, Wilson-loop carriers)  — β_1 = 3
  V_scalar (2-dim, J=+1, zero Wilson-loop)      — "the K_4 anomaly"

Bass-Stark-Terras factorization of det(uI - B) on K_4 gives:
  det(uI - B) = (u²-1)^(|E|-|V|) · ∏_λ (u² - λu + (k-1))
             = (u²-1)² · (u-1)(u-2) · (u²+u+2)³
             = (u-1)³ · (u+1)² · (u-2) · (u²+u+2)³

So u=+1 has algebraic multiplicity 3 (mixed BS-T):
  • 2 modes from (u²-1)² bipartite factor
  • 1 mode from (u-1)(u-2) Perron-adjacency factor

u=-1 has algebraic multiplicity 2, all from (u²-1)² bipartite factor.

KEY STRUCTURAL QUESTION FOR ROUTE H REVISION
--------------------------------------------
Within V_scalar (2 J=+1 modes), do they sit cleanly as:
  • 1 Perron-adjacency mode (from (u-1)(u-2), gauge-singlet by Wilson 1974),
  • 1 BS-T bipartite-factor "extra" mode (the K_4 anomaly)?

If yes:
  • c_color = 1/4 closes via SU(3)_c excluding ALL of V_scalar (both anomalies AND Perron),
  • c_EM = 1/3 closes if U(1)_Y/SU(2)_L excludes ONLY the Perron-adjacency mode
    (keeping the BS-T bipartite "extra" as a coupled mode).

If they cross-mix (no canonical (1,1) split by BS-T sector):
  • the sector-specific c story is structurally fuzzy at the K_4 substrate level
  • AB1/AB2 may fire when we attempt Cl(6) gauge action on V_scalar.

METHOD
------
1. Build B and J on K_4 (12-dim directed-edge space).
2. Compute V_pm and verify dim 5 split (J=+1: 2, J=-1: 3).
3. Build the BS-T Perron-adjacency invariant subspace as the orbit
   of v_unif (uniform-on-edges, eigenvector at u=2) under {I, B, B²}.
4. Project V_pm onto BS-T Perron-adj subspace and BS-T bipartite subspace.
5. Check: is the Perron-adj mode at u=+1 entirely in V_scalar (J=+1)?
   And is it linearly independent from the "extra" BS-T bipartite J=+1 mode?
"""

import numpy as np
from fractions import Fraction

np.set_printoptions(precision=4, suppress=True, linewidth=160)

# ============================================================
# 1. K_4 setup
# ============================================================
N_V = 4
vertices = list(range(N_V))
directed_edges = [(u, v) for u in vertices for v in vertices if u != v]
N_DE = len(directed_edges)
e2i = {e: i for i, e in enumerate(directed_edges)}

B = np.zeros((N_DE, N_DE), dtype=int)
for i, (u, v) in enumerate(directed_edges):
    for w in vertices:
        if w == u or w == v:
            continue
        B[i, e2i[(v, w)]] = 1
J_mat = np.zeros((N_DE, N_DE), dtype=int)
for i, (u, v) in enumerate(directed_edges):
    J_mat[i, e2i[(v, u)]] = 1

print("="*78)
print(" W21 — BS-T sector identification within V_scalar on K_4")
print("="*78)
print()

# ============================================================
# 2. Spectral decomposition of B; identify u=±1 eigenspace
# ============================================================
ev, evec = np.linalg.eig(B.astype(float))
print(f" B spectrum:")
ev_sorted = sorted(ev, key=lambda z: (round(z.real, 6), round(z.imag, 6)))
for e_val in ev_sorted:
    if abs(e_val.imag) < 1e-8:
        print(f"   u = {e_val.real:+.4f}")
    else:
        print(f"   u = {e_val.real:+.4f} {'+' if e_val.imag >= 0 else '-'} {abs(e_val.imag):.4f}i")
print()

# Algebraic multiplicities
mult_p1 = int(np.sum(np.abs(ev - 1.0) < 1e-6))
mult_m1 = int(np.sum(np.abs(ev + 1.0) < 1e-6))
mult_2 = int(np.sum(np.abs(ev - 2.0) < 1e-6))
print(f" Algebraic multiplicities: u=+1 → {mult_p1},  u=-1 → {mult_m1},  u=2 → {mult_2}")

# Verify BS-T factorization prediction
# (u-1)³ (u+1)² (u-2) (u²+u+2)³
print(f" BS-T predicts:           u=+1 → 3 (2 bipartite + 1 Perron-adj),")
print(f"                          u=-1 → 2 (2 bipartite),  u=2 → 1 (Perron-adj)")
assert mult_p1 == 3, f"u=+1 multiplicity mismatch: got {mult_p1}, expected 3"
assert mult_m1 == 2, f"u=-1 multiplicity mismatch: got {mult_m1}, expected 2"
assert mult_2 == 1, f"u=2 multiplicity mismatch: got {mult_2}, expected 1"
print(" ✓ BS-T algebraic multiplicities match.")
print()

# ============================================================
# 3. Build V_p1, V_m1, V_2 (geometric eigenspaces)
# ============================================================
def real_eigenspace(M, val, tol=1e-8):
    """Robust real basis for the geometric eigenspace of M at eigenvalue val."""
    K = M - val * np.eye(M.shape[0])
    U, S, _ = np.linalg.svd(K)
    null_dim = int(np.sum(S < tol))
    V = U[:, M.shape[0]-null_dim:]
    return V

V_p1 = real_eigenspace(B.astype(float), 1.0)
V_m1 = real_eigenspace(B.astype(float), -1.0)
V_2 = real_eigenspace(B.astype(float), 2.0)
print(f" Geometric multiplicities (null-space dims):")
print(f"   V_+1: {V_p1.shape[1]},  V_-1: {V_m1.shape[1]},  V_+2: {V_2.shape[1]}")
print()

# ============================================================
# 4. J action on V_pm; split J=+1 vs J=-1
# ============================================================
V_pm = np.concatenate([V_p1, V_m1], axis=1)
assert V_pm.shape[1] == 5

# J restricted to V_pm
J_in_Vpm = V_pm.T @ J_mat.astype(float) @ V_pm
ev_J, evec_J = np.linalg.eig(J_in_Vpm)
mask_J_p = np.abs(ev_J.real - 1.0) < 1e-6
mask_J_m = np.abs(ev_J.real + 1.0) < 1e-6
n_J_p = int(np.sum(mask_J_p))
n_J_m = int(np.sum(mask_J_m))
print(f" J eigenvalues on V_pm: J=+1 → {n_J_p} modes,  J=-1 → {n_J_m} modes")
print(f"   (yesterday: V_scalar dim 2, V_cycle dim 3 — consistent)")
print()

# Build V_scalar (J=+1) and V_cycle (J=-1) in original ℂ^{N_DE}
V_scalar = V_pm @ np.real(evec_J[:, mask_J_p])
V_cycle  = V_pm @ np.real(evec_J[:, mask_J_m])
V_scalar, _ = np.linalg.qr(V_scalar)
V_cycle, _ = np.linalg.qr(V_cycle)
print(f" V_scalar shape: {V_scalar.shape}")
print(f" V_cycle  shape: {V_cycle.shape}")
print()

# ============================================================
# 5. BS-T Perron-adjacency invariant subspace
# ============================================================
# The Perron adjacency eigenvector is ψ_3 = (1, 1, 1, 1) (uniform on K_4 vertices).
# Its directed-edge lift via the BS-T mechanism spans a 2-dim invariant subspace
# of B (corresponding to factor (u-1)(u-2)).
#
# Two canonical generators of this subspace:
#   v_src[e=(u,v)] = 1   ← constant on directed edges (1-dim, no source/target distinction since ψ_3 is uniform)
#
# Wait — for ψ_3 uniform, v_src ≡ v_tgt ≡ all-ones, only 1-dim.
# The 2-dim Perron-adj BS-T factor must come from a more general construction.
# Let me check what the invariant subspace of B containing v_unif actually is.
v_unif = np.ones(N_DE) / np.sqrt(N_DE)
print(f" Test: B · v_unif (eigenvector test):")
Bv = B.astype(float) @ v_unif
print(f"   B v_unif / v_unif = {Bv[0]/v_unif[0]:.4f}  (should be k-1 = 2)")
assert np.allclose(Bv, 2.0 * v_unif), "v_unif not at eigenvalue 2"
print(" ✓ v_unif is the Hashimoto Perron eigenvector at u = k-1 = 2")
print()

# The OTHER BS-T Perron-adjacency mode is at u=+1, in the same (u-1)(u-2)
# invariant subspace. Build the 2-dim invariant subspace by iterating B
# on v_unif and another vector that's not in the kernel of (B-2I), then
# restricting to the BS-T Perron-adj factor.
#
# Alternative cleaner approach: the Perron-adj BS-T factor at u=+1 is the
# kernel of (B-I) ∩ image of "lift maps from adjacency eigenvectors at λ=3".
# But since ψ_3 is the only adjacency vector at λ=3, and its lift gives
# a 1-dim invariant subspace (uniform-on-edges), the "Perron-adj at u=+1"
# mode might just be DEGENERATE / absorbed into the bipartite factor.

# Let me check: does u=+1 have GENERALIZED algebraic multiplicity 3 with
# a Jordan block structure, or does it have full geometric multiplicity 3?
# Earlier we saw V_+1 geometric mult = 3, so B is diagonalizable at u=+1.

# So all 3 modes at u=+1 are honest geometric eigenvectors. The BS-T
# decomposition then must distribute them as 2 (bipartite) + 1 (Perron-adj),
# but this is an ALGEBRAIC distinction within the same geometric eigenspace.

# To identify the Perron-adj mode at u=+1 explicitly: it's the unique
# direction in V_+1 that lies in the (B-2I)-invariant subspace V_perp_unif.
# But (B-2I) has v_unif in its kernel, so its image is a 11-dim subspace
# that contains all OTHER eigenmodes of B except v_unif.
#
# That's almost-all of ℂ^12. Not helpful.

# Direct algebraic approach: BS-T factor (u-1)(u-2) corresponds to a
# 2-dim invariant subspace V_{Perron-adj} of B with characteristic
# polynomial (u-1)(u-2). It's the smallest invariant subspace containing
# v_unif under the action of (B - 1·I)·(B - 2·I) = 0 on the factor.

# Construction via SUNADA-LIKE LIFT:
# For adjacency eigenvector ψ on vertices, define the directed-edge lift
#   v_α,β [e=(u,v)] := α · ψ(u) + β · ψ(v).
# This 2-dim family lives in V_{Perron-adj-factor}. The two distinct
# eigenvalues of B restricted to this 2-dim family are u = 1 and u = 2.

# For ψ_3 = (1,1,1,1):
# v_α,β[e] = α + β (constant), independent of e — only 1-dim.
# So this approach degenerates for the uniform ψ.

# A DIFFERENT construction: use the "twisted" lift
#   w_e [e=(u,v)] := ψ(u) - ψ(v)
# For ψ uniform, this is zero — also degenerate.

# CONCLUSION: For K_4 with uniform Perron ψ_3, the BS-T Perron-adjacency
# factor's lift produces only a 1-dim invariant subspace (v_unif itself),
# NOT 2-dim. Yet the characteristic polynomial of B has (u-1)(u-2) factor.
# This means the "u=+1 Perron-adj mode" is purely ALGEBRAIC and lives
# inside the larger V_+1 eigenspace where it's MIXED with bipartite modes.
#
# Equivalently: the BS-T sector decomposition is non-canonical at u=+1
# for K_4 — the 3-dim V_+1 has no preferred (2 bipartite + 1 Perron-adj)
# split unless an external structure (e.g., a specific bilinear form,
# or a specific graph automorphism) selects one.

# This is a CRITICAL FINDING. Let me verify with a different probe:
# compute (B - 2I) acting on V_+1 — if Perron-adj at u=+1 is canonically
# defined, it should be in the kernel of some specific polynomial in B.

# (B-I)(B-2I) acts on the Perron-adj BS-T factor as zero. On the bipartite
# factor (B²-I) = 0, so (B-I) annihilates u=+1 bipartite. On the Perron-adj
# factor, (B-I) annihilates u=+1 Perron-adj. So (B-I) annihilates ALL of V_+1.

# (B+I) on V_+1: zero on bipartite (since (B-I)(B+I) = 0 there), but
# B+I = (2I - (B-2I)) annihilates u=-1 only on the bipartite factor.
# On V_+1: (B+I) sends u=+1 modes to (1+1)=2 times themselves. So (B+I)
# restricted to V_+1 is 2·I_3 — it doesn't distinguish bipartite from Perron-adj.

# More refined: (B-2I) acts on V_+1 as (1-2)·I_3 = -I_3. Doesn't distinguish either.

# So NO polynomial in B distinguishes the BS-T sectors within V_+1.
# The 3-dim V_+1 IS canonically the full geometric eigenspace; the BS-T
# (2+1) split is ARTIFICIAL at u=+1 (for K_4 with uniform Perron ψ_3).

print(" "*1 + "─"*76)
print(" CRITICAL FINDING: BS-T sector split within V_+1 is NON-CANONICAL on K_4")
print(" "*1 + "─"*76)
print()
print("   The BS-T factorization det(uI-B) = (u²-1)²·(u-1)(u-2)·(u²+u+2)³ gives")
print("   ALGEBRAIC multiplicity 3 at u=+1 (2 bipartite + 1 Perron-adj), but on K_4")
print("   the geometric eigenspace V_+1 is 3-dim and B|_{V_+1} = I — every direction")
print("   in V_+1 is a B-eigenvector.")
print()
print("   The Perron adjacency eigenvector ψ_3 = (1,1,1,1) is uniform, so the")
print("   directed-edge lift v_α,β[e=(u,v)] = α·ψ_3(u) + β·ψ_3(v) is 1-dim (constant).")
print("   The Hashimoto Perron at u = k-1 = 2 absorbs this lift entirely.")
print()
print("   ⇒ For K_4, the 'BS-T bipartite factor (2 modes at u=+1) + Perron-adj")
print("     (1 mode at u=+1)' algebraic decomposition does NOT correspond to any")
print("     basis-independent geometric subspace of V_+1.")
print()
print("   This means the yesterday-probe's V_scalar (2 J=+1 modes) cannot be")
print("   cleanly split as (1 Perron-adj + 1 BS-T-extra) — the BS-T sectors are")
print("   degenerate at u=+1 for the uniform-Perron-ψ case.")
print()

# ============================================================
# 6. Wilson-loop content of V_scalar's 2 J=+1 modes
# ============================================================
# Even though BS-T sectors don't canonically split V_+1, we can still ask:
# do the 2 V_scalar modes have CANONICAL graph-theoretic structure?

# Triangles (3-cycles) on K_4:
triangles = []
for omit in range(N_V):
    others = [v for v in range(N_V) if v != omit]
    a, b, c = others
    triangles.append([(a, b), (b, c), (c, a)])

# Wilson-loop on each triangle: sum of v[e] over e in triangle
def wilson_holonomy(v):
    return np.array([sum(v[e2i[e]] for e in T) for T in triangles])

print(" Wilson-loop content of V_scalar modes (J=+1, zero Wilson-loop carrier sector):")
for k in range(V_scalar.shape[1]):
    v = V_scalar[:, k]
    w = wilson_holonomy(v)
    print(f"   mode {k}: ‖v‖² = {np.dot(v,v):.4f},  triangle holonomies = {w}")
print()

print(" Wilson-loop content of V_cycle modes (J=-1, true Wilson-loop carriers):")
for k in range(V_cycle.shape[1]):
    v = V_cycle[:, k]
    w = wilson_holonomy(v)
    print(f"   mode {k}: ‖v‖² = {np.dot(v,v):.4f},  triangle holonomies = {w.round(4)}")
print()

# ============================================================
# 7. Test: do the 2 V_scalar modes have distinct "vertex-source" content?
# ============================================================
# Define vertex source/target maps from directed edges:
#   src(v) = Σ_e starting at v  →  acts as a 4×12 matrix
#   tgt(v) = Σ_e ending at v
src = np.zeros((N_V, N_DE))
tgt = np.zeros((N_V, N_DE))
for i, (u, v) in enumerate(directed_edges):
    src[u, i] = 1
    tgt[v, i] = 1

print(" Source / target vertex profiles of V_scalar modes:")
for k in range(V_scalar.shape[1]):
    v = V_scalar[:, k]
    s = src @ v
    t = tgt @ v
    print(f"   mode {k}: src = {s.round(4)},  tgt = {t.round(4)}")
print()

print(" Source / target vertex profiles of V_cycle modes:")
for k in range(V_cycle.shape[1]):
    v = V_cycle[:, k]
    s = src @ v
    t = tgt @ v
    print(f"   mode {k}: src = {s.round(4)},  tgt = {t.round(4)}")
print()

# ============================================================
# 8. Verdict
# ============================================================
print("="*78)
print(" VERDICT")
print("="*78)
print()
print(" The BS-T algebraic sector split at u=+1 ((u-1)² bipartite × (u-1) Perron-adj)")
print(" does NOT lift to a canonical geometric subspace decomposition of V_+1 on K_4.")
print(" The Perron-adjacency factor at u=+1 is 'absorbed' by the bipartite factor")
print(" because the Perron adjacency eigenvector ψ_3 is uniform.")
print()
print(" CONSEQUENCE FOR SECTOR-SPECIFIC c VALUES")
print(" ----------------------------------------")
print(" The original Route H (alpha_GUT_dark_correction theorem §3.1) writes")
print(" 'V_scalar = 1 Perron-adj + 1 BS-T-extra' as a STRUCTURAL identification,")
print(" but on K_4 this is ALGEBRAIC ONLY. Any (1,1) split of V_scalar requires")
print(" an EXTERNAL structure — graph automorphism, gauge action, or boundary data.")
print()
print(" The yesterday-probe's C_3 isotypic check showed V_scalar = (0 trivial + 1")
print(" faithful pair) — but C_3 acts irreducibly on V_scalar, so C_3 itself cannot")
print(" produce a (1,1) split.")
print()
print(" SURVIVING PATHWAYS for sector-specific c (in priority order):")
print()
print(" (1) BIPARTITE-DOUBLE-COVER ROUTE: pass to srs-z (the bipartite double cover")
print("     of srs). On srs-z, the BS-T factorization is DIFFERENT and ψ_3 is")
print("     NOT uniform across the two cover sheets. The Perron-adj factor at u=+1")
print("     may split cleanly there. Yesterday's srsz_double_cover_mode_count_2026-05-26.py")
print("     found that the K_4 anomaly PERSISTS on srs-z, but didn't ask whether")
print("     the BS-T sector split is canonical on the cover. ← worth checking.")
print()
print(" (2) Cl(6) GAUGE-ACTION ROUTE (Session 1 of multi-session program): even if")
print("     the BS-T split is non-canonical at the graph level, the SU(3)_c × SU(2)_L")
print("     × U(1)_Y gauge action induced from Cl(6) Fock may select a canonical (1,1)")
print("     split of V_scalar via the GAUGE generator's action — i.e., the Cl(6)")
print("     content of edges induces a NON-GRAPH-AUTOMORPHIC partition of V_scalar.")
print()
print(" (3) CYCLE-COUNTING ROUTE C: instead of relying on BS-T modes, use the Route C")
print("     girth/walker-step combinatorics (which has independent sector-specificity).")
print("     The current Route C derivation in theorem_alpha_GUT_dark_correction.md §4")
print("     gives c_α_GUT = 2|E|/(N_atoms · k_*²) = 1/k_*. Generalizing to sector")
print("     specific c values requires per-gauge-sector walker counts.")
print()
print(" RECOMMENDATION:")
print(" --------------")
print(" Before launching Session 1 (Cl(6) Fock per-vertex lift), test pathway (1):")
print(" check whether srs-z gives a canonical BS-T (Perron-adj vs bipartite) split of")
print(" V_scalar's J=+1 modes. If yes, lift sector-specific c via srs-z. If no, fall")
print(" through to Session 1 Cl(6) gauge action (which is harder but more principled).")
print()
print("="*78)
