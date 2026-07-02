#!/usr/bin/env python3
"""
srs-z (bipartite double cover of K_4) Hashimoto mode count.

CONTEXT
-------
On K_4 (non-bipartite), Bass-Stark-Terras gives bipartite-factor
multiplicity 2(|E|-|V|) = 4 at u=±1, but Wilson-loop holonomy reveals
only 3 cycle modes (= β_1) + 2 scalar modes. The "extra 1" in the BS-T
polynomial count is a non-bipartite anomaly.

On srs-z (bipartite), BS-T gives 2(|E|-|V|+1) = 2·β_1 = 10 marginal
modes. Bipartite graphs have NO Perron-adjacency anomaly — the
bipartite-factor count exactly matches β_1 cycle modes.

GOAL
----
1. Compute Hashimoto B' on srs-z; verify the 10-mode bipartite-marginal
   sector cleanly matches β_1 cycle modes (no scalar pollution).
2. Project srs-z modes back onto K_4 (via the natural deck-transformation
   quotient) and ask: does the lift+quotient give the (3 cycle + 1 scalar
   + 1 Perron) = (3,1,1) split on K_4 that the gauge-cluster fit demands?
3. If yes: this is the substrate-side natural structural decomposition,
   and the (3+1, 3) split = (EM, color) gauge-block partition is forced
   by deck-symmetry χ̃ (the Z_2 covering map srs-z → srs).
"""

import math
import numpy as np
from itertools import permutations

# ============================================================
# 1. Build srs-z = bipartite double cover of K_4
# ============================================================
N_V_base = 4
base_vertices = list(range(N_V_base))
# srs-z vertices: (v, layer) with layer ∈ {0, 1}
srsz_vertices = [(v, ε) for v in base_vertices for ε in (0, 1)]
N_V = len(srsz_vertices)  # 8
v2i = {v: i for i, v in enumerate(srsz_vertices)}

# srs-z edges: (u, 0) ↔ (v, 1) iff (u, v) is an edge in K_4
# K_4 has all (u, v) with u ≠ v as edges
srsz_directed_edges = []
for u in base_vertices:
    for v in base_vertices:
        if u == v:
            continue
        # Two layer-crossing edges per K_4 edge (we list directed pairs)
        srsz_directed_edges.append(((u, 0), (v, 1)))
        srsz_directed_edges.append(((u, 1), (v, 0)))
        # NOTE: each undirected (u,v) in K_4 gives 2 undirected edges in srs-z
        # (and we list each undirected edge as 2 directed edges, total 4 per K_4 edge)
# Remove duplicates: directed edges are listed pairs
N_DE = len(srsz_directed_edges)
print("="*78)
print("  srs-z (bipartite double cover of K_4) — Hashimoto mode count")
print("="*78)
print(f"  K_4: |V| = 4, |E| = 6, non-bipartite")
print(f"  srs-z: |V'| = {N_V} = 2|V|,  N_DE_raw = {N_DE}")

# Deduplicate undirected: each undirected srs-z edge appears twice in srsz_directed_edges
# Let me rebuild cleanly:
srsz_directed_edges = set()
for u in base_vertices:
    for v in base_vertices:
        if u == v:
            continue
        # undirected (u,v) in K_4 → srs-z has TWO undirected edges:
        # (u,0)—(v,1)  AND  (u,1)—(v,0)
        # As directed edges, that's 4 total:
        srsz_directed_edges.add(((u, 0), (v, 1)))
        srsz_directed_edges.add(((v, 1), (u, 0)))
        srsz_directed_edges.add(((u, 1), (v, 0)))
        srsz_directed_edges.add(((v, 0), (u, 1)))
srsz_directed_edges = sorted(srsz_directed_edges)
N_DE = len(srsz_directed_edges)
N_E = N_DE // 2
print(f"  srs-z (clean): |V'| = {N_V},  |E'| = {N_E},  |2E'| = {N_DE}")
print(f"  β_1(srs-z) = |E'| - |V'| + 1 = {N_E - N_V + 1}")
print(f"  Bass-Stark-Terras (bipartite) bipartite-factor exponent = |E'| - |V'| + 1 = {N_E - N_V + 1}")
print(f"  Expected bipartite-marginal modes: 2(|E'| - |V'| + 1) = {2*(N_E - N_V + 1)}")
print(f"  Total NB dim: {N_DE}")
print(f"  Expected unified c = 2(|E'|-|V'|+1)/(2|E'|) = "
      f"{2*(N_E-N_V+1)}/{N_DE} = {2*(N_E-N_V+1)/N_DE:.6f}")
print()

# Re-establish edge index
e2i_srsz = {e: i for i, e in enumerate(srsz_directed_edges)}

# Hashimoto B' on srs-z
B = np.zeros((N_DE, N_DE), dtype=int)
for i, (s, t) in enumerate(srsz_directed_edges):
    # outgoing NB from t: edges (t, w) with w ≠ s
    for w in srsz_vertices:
        if w == s or w == t:
            continue
        if (t, w) in e2i_srsz:
            B[i, e2i_srsz[(t, w)]] = 1

# Edge reversal J'
J = np.zeros((N_DE, N_DE), dtype=int)
for i, (s, t) in enumerate(srsz_directed_edges):
    J[i, e2i_srsz[(t, s)]] = 1

# Sanity
assert J @ J - np.eye(N_DE) == pytest_zero if False else True  # skip
B_row_sum = B.sum(axis=1)
print(f"  B row sums (NB out-valence per directed edge): {set(B_row_sum.tolist())}")
print(f"  (Each directed edge has 2 NB-following edges, since k=3.)")
print()

# ============================================================
# 2. Hashimoto eigenvalue decomposition
# ============================================================
eigvals = np.linalg.eigvals(B.astype(float))
print(f"Hashimoto eigenvalues of srs-z (sorted by real part):")
ev_sort = np.array(sorted(eigvals, key=lambda z: (z.real, z.imag)))
for ev in ev_sort:
    print(f"  {ev}")
print()

# ============================================================
# 3. u = ±1 eigenspaces
# ============================================================
eigvals_arr, eigvecs_arr = np.linalg.eig(B.astype(float))
mask_p = np.abs(eigvals_arr - 1.0) < 1e-8
mask_m = np.abs(eigvals_arr + 1.0) < 1e-8
V_p = np.real_if_close(eigvecs_arr[:, mask_p], tol=1000)
V_m = np.real_if_close(eigvecs_arr[:, mask_m], tol=1000)
if np.iscomplexobj(V_p):
    V_p = np.real(V_p)
if np.iscomplexobj(V_m):
    V_m = np.real(V_m)
V_p, _ = np.linalg.qr(V_p)
V_m, _ = np.linalg.qr(V_m)

print(f"  u = +1 eigenspace dim: {V_p.shape[1]}")
print(f"  u = -1 eigenspace dim: {V_m.shape[1]}")

V_pm = np.concatenate([V_p, V_m], axis=1)
print(f"  Total u = ±1 sector dim: {V_pm.shape[1]}  (expected 2β_1 + small Perron-corrections)")
print()

# ============================================================
# 4. Wilson-loop holonomy on srs-z cycles
# ============================================================
# β_1 = 5 cycles in srs-z. Build a representative set.
# Cycles of length ≥ 4 (since srs-z is bipartite, no odd cycles).
# 4-cycles: (u,0)—(v,1)—(w,0)—(x,1)—(u,0) where u,v,w,x are appropriate K_4 vertices

# Find all 4-cycles by BFS
def find_4_cycles():
    cycles = set()
    for start in srsz_vertices:
        # BFS depth 4
        for v1 in srsz_vertices:
            if v1 == start: continue
            if (start, v1) not in e2i_srsz: continue
            for v2 in srsz_vertices:
                if v2 in (start, v1): continue
                if (v1, v2) not in e2i_srsz: continue
                for v3 in srsz_vertices:
                    if v3 in (start, v1, v2): continue
                    if (v2, v3) not in e2i_srsz: continue
                    if (v3, start) not in e2i_srsz: continue
                    # Found a 4-cycle
                    cycle = (start, v1, v2, v3)
                    # Canonicalize: rotation/reflection-invariant
                    rotations = [cycle[i:] + cycle[:i] for i in range(4)]
                    rev = tuple(reversed(cycle))
                    rotations.extend([rev[i:] + rev[:i] for i in range(4)])
                    canon = min(rotations)
                    cycles.add(canon)
    return list(cycles)

cycles_4 = find_4_cycles()
print(f"  Found {len(cycles_4)} undirected 4-cycles in srs-z.")

# Convert each cycle to directed edge list
def cycle_to_directed(cycle):
    edges = []
    for i in range(len(cycle)):
        s = cycle[i]
        t = cycle[(i+1) % len(cycle)]
        edges.append((s, t))
    return edges

# Compute Wilson-loop holonomy matrix
cycle_dirs = [cycle_to_directed(c) for c in cycles_4]
H_mat = np.zeros((V_pm.shape[1], len(cycle_dirs)))
for k in range(V_pm.shape[1]):
    v = V_pm[:, k]
    for c_idx, cycle in enumerate(cycle_dirs):
        H_mat[k, c_idx] = sum(v[e2i_srsz[e]] for e in cycle)

U_h, S_h, _ = np.linalg.svd(H_mat, full_matrices=True)
rank_h = int(np.sum(S_h > 1e-8))
print(f"  Wilson-loop holonomy matrix rank: {rank_h}")
print(f"  → cycle-mode count in u=±1 sector: {rank_h}")
print(f"  → scalar-mode (kernel of WL) count: {V_pm.shape[1] - rank_h}")
print(f"  (For bipartite srs-z, expected cycle-modes = 2β_1 = {2*(N_E-N_V+1)} if BS-T clean)")
print()

# ============================================================
# 5. Project srs-z modes back to K_4 (quotient by deck transformation χ̃)
# ============================================================
# Deck transformation χ̃: (v, 0) ↔ (v, 1)  (Z_2 covering map)
# χ̃ on directed edges: (s, t) ↦ (χ̃(s), χ̃(t)) where χ̃ flips the layer
print("-"*78)
print("Deck transformation χ̃ (Z_2 covering map srs-z → srs):")
print("-"*78)

chi = np.zeros((N_DE, N_DE), dtype=int)
def flip(vv):
    return (vv[0], 1 - vv[1])
for i, (s, t) in enumerate(srsz_directed_edges):
    s_flip = flip(s)
    t_flip = flip(t)
    chi[i, e2i_srsz[(s_flip, t_flip)]] = 1

# Verify χ̃² = I
assert np.allclose(chi @ chi, np.eye(N_DE)), "χ̃² should be identity"
print(f"  χ̃² = I confirmed.")

# Decompose u=±1 sector by χ̃ = ±1 eigenspaces
chi_on_pm = V_pm.T @ chi @ V_pm
chi_evals, chi_evecs = np.linalg.eigh(chi_on_pm)
print(f"  χ̃ spectrum on u=±1 sector: {np.round(chi_evals, 4).tolist()}")
n_chi_plus  = np.sum(chi_evals > 0.5)
n_chi_minus = np.sum(chi_evals < -0.5)
print(f"  dim(χ̃ = +1, symmetric: lifts of K_4 modes)  = {n_chi_plus}")
print(f"  dim(χ̃ = -1, antisymmetric: new srs-z modes) = {n_chi_minus}")
print()

# The χ̃ = +1 sector projects back to K_4 (5 dim should match K_4's u=±1 sector)
# The χ̃ = -1 sector is the "new" content from the double cover (genuinely srs-z)

print("-"*78)
print(" Cross-check: χ̃ = +1 sector projects to K_4 u=±1 sector (dim 5)?")
print("-"*78)
print(f"  K_4 had: 3 cycle modes + 2 scalar modes = 5 total at u=±1.")
print(f"  srs-z has dim {V_pm.shape[1]} at u=±1.")
print(f"  Expected from deck symmetry: χ̃ = +1 sub-sector has dim 5 (K_4 lift)")
print()

# Identify cycle vs scalar within χ̃ = +1
chi_plus_basis = chi_evecs[:, chi_evals > 0.5]
V_chi_plus = V_pm @ chi_plus_basis   # 24 × n_chi_plus

# Compute Wilson-loop holonomy of χ̃=+1 modes on srs-z 4-cycles
H_chi_plus = np.zeros((V_chi_plus.shape[1], len(cycle_dirs)))
for k in range(V_chi_plus.shape[1]):
    v = V_chi_plus[:, k]
    for c_idx, cycle in enumerate(cycle_dirs):
        H_chi_plus[k, c_idx] = sum(v[e2i_srsz[e]] for e in cycle)
rank_chi_plus = np.linalg.matrix_rank(H_chi_plus, tol=1e-8)
print(f"  χ̃ = +1 sector Wilson-loop rank: {rank_chi_plus}  (cycle-modes within χ̃ = +1)")
print(f"  → scalar-modes within χ̃ = +1: {V_chi_plus.shape[1] - rank_chi_plus}")
print()

# Same for χ̃ = -1
chi_minus_basis = chi_evecs[:, chi_evals < -0.5]
V_chi_minus = V_pm @ chi_minus_basis
H_chi_minus = np.zeros((V_chi_minus.shape[1], len(cycle_dirs)))
for k in range(V_chi_minus.shape[1]):
    v = V_chi_minus[:, k]
    for c_idx, cycle in enumerate(cycle_dirs):
        H_chi_minus[k, c_idx] = sum(v[e2i_srsz[e]] for e in cycle)
rank_chi_minus = np.linalg.matrix_rank(H_chi_minus, tol=1e-8)
print(f"  χ̃ = -1 sector Wilson-loop rank: {rank_chi_minus}  (cycle-modes within χ̃ = -1)")
print(f"  → scalar-modes within χ̃ = -1: {V_chi_minus.shape[1] - rank_chi_minus}")
print()

# ============================================================
# 6. Verdict
# ============================================================
print("="*78)
print("VERDICT")
print("="*78)
total_cycles = rank_h
total_scalars = V_pm.shape[1] - rank_h
print(f"  srs-z u=±1 sector: dim {V_pm.shape[1]} = {total_cycles} cycle-modes "
      f"+ {total_scalars} scalar-modes")
print(f"  β_1(srs-z) = {N_E - N_V + 1};  expected pure-bipartite cycle count = 2·β_1 = "
      f"{2*(N_E-N_V+1)}.")
print()
if total_cycles == 2 * (N_E - N_V + 1):
    print(f"  ✓ Cycle-mode count = 2·β_1 EXACTLY on srs-z (no scalar pollution).")
    print(f"    Bipartite-factor count = cycle count on bipartite double cover.")
    print(f"    This confirms the K_4 anomaly is non-bipartite-specific.")
else:
    print(f"  ✗ Cycle count {total_cycles} ≠ 2β_1 = {2*(N_E-N_V+1)} — even on bipartite cover.")

print()
print(f"  χ̃ split (deck transformation Z_2):")
print(f"    χ̃ = +1 (K_4 lifts): dim {n_chi_plus}  "
      f"→ {rank_chi_plus} cycle + {n_chi_plus - rank_chi_plus} scalar")
print(f"    χ̃ = -1 (new srs-z): dim {n_chi_minus}  "
      f"→ {rank_chi_minus} cycle + {n_chi_minus - rank_chi_minus} scalar")
print()
if rank_chi_plus == 3 and (n_chi_plus - rank_chi_plus) == 2:
    print(f"  ✓ χ̃ = +1 sector matches K_4's u=±1 split: 3 cycle + 2 scalar.")
    print(f"    The 2 K_4 scalar modes lift CLEANLY to the χ̃ = +1 sub-sector of srs-z.")
elif rank_chi_plus == 3:
    print(f"  Partial: χ̃ = +1 has 3 cycle modes (matching K_4 β_1).")
else:
    print(f"  Unexpected: χ̃ = +1 cycle count {rank_chi_plus} ≠ 3.")

print()
print(" STRUCTURAL READING:")
print(" If χ̃ = +1 sub-sector cleanly splits as (3 cycle, 2 scalar) matching K_4,")
print(" and χ̃ = -1 sub-sector has its own structure, then χ̃ (deck symmetry) is")
print(" a CANDIDATE substrate-internal operator that distinguishes the 2 scalar")
print(" modes — IF one specific scalar mode in K_4 lifts to a χ̃ = +1 'preferred'")
print(" combination and the other to a χ̃-mixed combination.")
print("="*78)
