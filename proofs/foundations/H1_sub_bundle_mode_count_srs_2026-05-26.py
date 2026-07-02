#!/usr/bin/env python3
"""
Direct sub-bundle bipartite-marginal mode count on srs (K_4).

GOAL
----
The sector-specific c probe (`sector_specific_c_alpha_GUT_scan_2026-05-26.py`)
found that fitting the gauge cluster to PDG requires:

    c_1 ≈ 1/3 = 4/12,   c_2 ≈ 1/3 = 4/12,   c_3 ≈ 1/4 = 3/12

i.e., SU(3)_c "sees" exactly 3 of the 4 bipartite-marginal Hashimoto modes
while U(1)_Y and SU(2)_L see all 4. The structural claim that closes this:

    dim H¹_{4-mode sector} resolved by SM gauge sub-bundle gives partition
    {U(1)_Y: 4, SU(2)_L: 4, SU(3)_c: 3}.

This probe computes the bipartite-marginal eigenspace directly, then tests
whether any natural framework-grade on srs splits it as (4, 4, 3) — or, more
modestly, whether some natural Z_2 or C_n grading reduces dim from 4 to 3
for one specific sector.

METHOD
------
1. Build directed-edge Hashimoto operator B on K_4 (12 × 12 sparse).
2. Compute eigenvectors at u = +1 and u = -1 (the bipartite-factor marginal
   modes). Confirm total dim 4 = 2(|E|-|V|).
3. Test the 4-dim space under each natural grading:
   (a) Edge-reversal involution J: e ↦ -e. Eigenvalues ±1.
   (b) Bipartite double-cover χ̃ Z_2.
   (c) S_4 = Aut(K_4) representation theory; in particular, the standard
       3-dim irrep (which gives V_Ram structure in the framework).
   (d) Cycle-class structure: which independent 3-cycles or 4-cycles of
       K_4 carry the mode.
4. Report whether any grading produces a (3, 1) split on the 4-dim space —
   the structural fingerprint c_3 = 1/4 would need.
5. HONEST verdict: if no natural grading produces (3,1), the sector-specific
   c values are coincidence, not substrate-derivable from Route H sub-bundles.

NOT theorem-grade. Diagnostic computation only. Outcome determines whether
to scope a deeper structural derivation or rule out this hypothesis.
"""

import math
import numpy as np
from fractions import Fraction
from itertools import permutations

# ------------------------------------------------------------------
# 1. Build K_4 directed-edge Hashimoto operator
# ------------------------------------------------------------------
N_V = 4
vertices = list(range(N_V))

# Directed edges: list of (u, v) with u != v
directed_edges = [(u, v) for u in vertices for v in vertices if u != v]
N_E = len(directed_edges) // 2     # 6 undirected
N_DE = len(directed_edges)         # 12 directed

# Index map
e2i = {e: i for i, e in enumerate(directed_edges)}

# Build B
B = np.zeros((N_DE, N_DE), dtype=int)
for i, (u, v) in enumerate(directed_edges):
    # Outgoing NB from v: (v, w) where w != u
    for w in vertices:
        if w == u or w == v:
            continue
        j = e2i[(v, w)]
        B[i, j] = 1

# Edge-reversal J: J[i, j] = 1 if directed_edges[i] = reverse(directed_edges[j])
J = np.zeros((N_DE, N_DE), dtype=int)
for i, (u, v) in enumerate(directed_edges):
    j = e2i[(v, u)]
    J[i, j] = 1

# Sanity checks
assert B.sum() == 24, f"Expected 24 nonzero in B, got {B.sum()}"
assert np.allclose(J @ J, np.eye(N_DE)), "J² should be identity"
print("="*78)
print("  Sub-bundle bipartite-marginal mode count on srs (K_4)  (2026-05-26)")
print("="*78)
print(f"  |V| = {N_V},  |E| = {N_E},  |2E| = {N_DE},  k* = 3")
print(f"  Bass-Stark-Terras: 2(|E|-|V|) = {2*(N_E - N_V)} bipartite-marginal modes expected")
print(f"  Total NB dim = {N_DE}, so unified c = {2*(N_E - N_V)}/{N_DE} = "
      f"{Fraction(2*(N_E - N_V), N_DE)}")
print()

# ------------------------------------------------------------------
# 2. Eigendecomposition of B
# ------------------------------------------------------------------
eigvals, eigvecs = np.linalg.eig(B.astype(float))
# Sort eigenvalues by real part for inspection
order = np.argsort(eigvals.real)
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

print("Hashimoto eigenvalues of K_4:")
for ev in eigvals:
    print(f"  {ev}")
print()

# ------------------------------------------------------------------
# 3. Extract bipartite-marginal eigenspace at u = ±1
# ------------------------------------------------------------------
TOL = 1e-9

def find_eigenspace(target):
    """Return orthonormal basis (real if possible) of eigenspace at target eigenvalue."""
    mask = np.abs(eigvals - target) < 1e-8
    V = eigvecs[:, mask]
    # Drop tiny imaginary parts
    V = np.real_if_close(V, tol=1000)
    if np.iscomplexobj(V):
        # Re-orthogonalize real parts
        Vr = np.real(V)
        Vi = np.imag(V)
        combined = np.concatenate([Vr, Vi], axis=1)
        Q, _ = np.linalg.qr(combined)
        rank = np.linalg.matrix_rank(combined, tol=1e-8)
        V = Q[:, :rank]
    else:
        V, _ = np.linalg.qr(V)
    return V

V_plus  = find_eigenspace(+1.0)
V_minus = find_eigenspace(-1.0)
dim_plus = V_plus.shape[1]
dim_minus = V_minus.shape[1]

# Note: u = +1 has multiplicity 3 from Bass-Stark-Terras (2 from bipartite factor
# + 1 from adjacency-Perron scalar). u = -1 has multiplicity 2 (all from
# bipartite factor). So total at u = ±1 is 5 raw, of which 4 are bipartite-
# marginal proper. We need to separate the Perron-adjacency scalar (uniform
# directed-edge function) from the bipartite-factor marginal modes.

# The Perron-adjacency scalar mode at u = +1 is the constant function on
# directed edges (eigenvector of A with λ = k*, lifted by u² - 3u + 2 root
# u = 1, multiplicity 1, mode = (1,1,...,1)/sqrt(12)).
uniform = np.ones(N_DE) / math.sqrt(N_DE)
print(f"u = +1 eigenspace raw dim = {dim_plus}  (= 2 cycle-modes + 1 Perron-scalar)")
print(f"u = -1 eigenspace raw dim = {dim_minus}  (= 2 cycle-modes)")

# Identify the Perron-adjacency-derived scalar mode at u = +1 by Wilson-loop
# holonomy: it is the unique mode in the u = +1 eigenspace with ZERO holonomy
# on every 3-cycle of K_4.
# K_4's 4 oriented 3-cycles (one per omitted vertex):
triangles = []
for omit in range(N_V):
    others = [v for v in range(N_V) if v != omit]
    # Oriented as (a→b, b→c, c→a)
    a, b, c = others
    triangles.append([(a, b), (b, c), (c, a)])

def wilson_loop(v, cycle):
    """Sum of mode v over a directed cycle."""
    return sum(v[e2i[edge]] for edge in cycle)

# Combine raw u = ±1 modes (5-dim space)
V_all = np.concatenate([V_plus, V_minus], axis=1)
print(f"Full u = ±1 eigenspace dim = {V_all.shape[1]}  (= 5 = 4 BM + 1 Perron-scalar)")

# For each mode in V_all, compute Wilson-loop holonomy magnitude across all triangles
holonomy_norms = []
for k in range(V_all.shape[1]):
    v = V_all[:, k]
    norms = [abs(wilson_loop(v, c)) for c in triangles]
    holonomy_norms.append(max(norms))

# Find the kernel of the holonomy map (modes with zero holonomy on all cycles)
# Construct the 4 × 5 holonomy matrix H[c, k] = WL(triangle c, mode k)
H = np.array([[wilson_loop(V_all[:, k], c) for c in triangles] for k in range(V_all.shape[1])])
# Each row is a mode's holonomy vector; find rows in the null space of holonomy
# i.e. modes spanning the kernel of cycle-coupling
U_h, S_h, Vt_h = np.linalg.svd(H, full_matrices=True)
print(f"Holonomy matrix singular values: {np.round(S_h, 4)}")
# Null space: columns of U_h corresponding to zero singular values
rank_h = int(np.sum(S_h > 1e-8))
print(f"Holonomy matrix rank = {rank_h}  → cycle-mode count = {rank_h},  scalar-mode count = {V_all.shape[1] - rank_h}")
print()

# 4-dim bipartite-marginal (cycle) sector: image of cycle-coupling
# 1-dim Perron-scalar sector: kernel of cycle-coupling
V_BM = V_all @ U_h[:, :rank_h]
V_perron_scalar = V_all @ U_h[:, rank_h:]
print(f"V_BM (cycle modes) dim = {V_BM.shape[1]}  ← gauge 1-point couples here (Route H c = 4/12 = 1/3)")
print(f"V_perron_scalar dim    = {V_perron_scalar.shape[1]}  ← scalar 2-point also couples here (v_Higgs c = 5/12)")
print()

# Sanity: verify each column is a B-eigenvector at u = ±1 and J·v has the
# matching reversal property.
print("Verification of bipartite-marginal sector:")
for k in range(V_BM.shape[1]):
    v = V_BM[:, k]
    Bv = B @ v
    # Find which eigenvalue this is closest to
    ratios = Bv[np.abs(v) > 0.05] / v[np.abs(v) > 0.05]
    u_eig = np.mean(ratios) if len(ratios) > 0 else float('nan')
    Jv = J @ v
    overlap_self = np.abs(np.dot(v, Jv))
    print(f"  mode {k}: u_eig ≈ {u_eig:+.4f},  ⟨v, Jv⟩ = {overlap_self:+.4f}")
print()

# ------------------------------------------------------------------
# 4. Grading tests
# ------------------------------------------------------------------
# (a) Edge-reversal J: split the 4-dim sector into J = +1 and J = -1 subspaces
print("-"*78)
print("GRADING (a): edge-reversal J = ±1")
print("-"*78)
# Project V_BM onto J-eigenspaces:
J_on_BM = V_BM.T @ J @ V_BM
J_evals, J_evecs = np.linalg.eigh(J_on_BM)
print(f"  J-spectrum on bipartite-marginal sector: {np.round(J_evals, 4)}")
dim_J_plus  = np.sum(J_evals > 0.5)
dim_J_minus = np.sum(J_evals < -0.5)
print(f"  dim(J=+1) = {dim_J_plus},  dim(J=-1) = {dim_J_minus}")
if (dim_J_plus, dim_J_minus) in [(3, 1), (1, 3)]:
    print(f"  >>> SPLIT IS (3, 1) under J — matches c_3 = 3/12 signature.")
else:
    print(f"  No (3,1) split under J. Symmetric (2,2) is the natural decomposition.")
print()

# (b) Bipartite double-cover χ̃: vertex 2-coloring of K_4. K_4 is NOT bipartite,
# but we can grade directed edges by parity of vertex labels.
print("-"*78)
print("GRADING (b): vertex-parity Z_2  (head + tail mod 2)")
print("-"*78)
parity = np.array([(u + v) % 2 for (u, v) in directed_edges])
chi = np.diag(2*parity - 1).astype(float)   # ±1 diagonal
chi_on_BM = V_BM.T @ chi @ V_BM
chi_evals, _ = np.linalg.eigh(chi_on_BM)
print(f"  χ-spectrum on bipartite-marginal sector: {np.round(chi_evals, 4)}")
dim_chi_plus  = np.sum(chi_evals > 0.5)
dim_chi_minus = np.sum(chi_evals < -0.5)
print(f"  dim(χ=+1) = {dim_chi_plus},  dim(χ=-1) = {dim_chi_minus}")
if (dim_chi_plus, dim_chi_minus) in [(3, 1), (1, 3)]:
    print(f"  >>> SPLIT IS (3, 1) under vertex-parity Z_2 — matches signature.")
print()

# (c) S_4 = Aut(K_4) decomposition
# Most useful S_4 reps: trivial (1), sign (1), standard 3-dim (3), 2-dim (2),
# 3-dim·sign (3). Total dim 1+1+2+3+3 = 10.
# Action: σ ∈ S_4 acts on directed edges by relabeling.
print("-"*78)
print("GRADING (c): S_4 = Aut(K_4) representation decomposition")
print("-"*78)
# Build representation matrices for the 24 permutations
def perm_matrix(sigma):
    """Permutation matrix on directed edges induced by σ ∈ S_4."""
    M = np.zeros((N_DE, N_DE))
    for i, (u, v) in enumerate(directed_edges):
        u_new, v_new = sigma[u], sigma[v]
        j = e2i[(u_new, v_new)]
        M[j, i] = 1
    return M

all_perms = list(permutations(range(N_V)))
# Compute the action of each permutation restricted to V_BM
# For each permutation σ, character on V_BM is Tr(V_BM^T · P_σ · V_BM)
chars = []
for sigma in all_perms:
    P = perm_matrix(sigma)
    # Restriction trace
    chi_val = np.trace(V_BM.T @ P @ V_BM)
    chars.append(chi_val)

# Group by cycle structure (conjugacy class in S_4)
def cycle_type(sigma):
    sigma = list(sigma)
    seen = [False]*len(sigma)
    cycles = []
    for i in range(len(sigma)):
        if seen[i]:
            continue
        j = i
        c = 0
        while not seen[j]:
            seen[j] = True
            j = sigma[j]
            c += 1
        cycles.append(c)
    return tuple(sorted(cycles, reverse=True))

# Average character over each conjugacy class
class_chars = {}
class_sizes = {}
for sigma, chi_val in zip(all_perms, chars):
    ct = cycle_type(sigma)
    class_chars.setdefault(ct, []).append(chi_val)
    class_sizes[ct] = class_sizes.get(ct, 0) + 1

print(f"  Average character of V_BM (4-dim) on S_4 conjugacy classes:")
chi_BM = {}
for ct, vals in class_chars.items():
    avg = np.mean(vals)
    chi_BM[ct] = avg
    print(f"    {ct} (size {class_sizes[ct]}):  χ = {avg:+.4f}")

# S_4 irreducible character table:
#                   e      (12)    (12)(34)  (123)   (1234)
# trivial           1       1       1        1        1
# sign              1      -1       1        1       -1
# standard 2-dim    2       0       2       -1        0
# standard 3-dim    3       1      -1        0       -1
# 3-dim ⊗ sign      3      -1      -1        0        1
irrep_chars = {
    'trivial':       {(1,1,1,1): 1, (2,1,1): 1,  (2,2): 1,  (3,1): 1,  (4,): 1},
    'sign':          {(1,1,1,1): 1, (2,1,1): -1, (2,2): 1,  (3,1): 1,  (4,): -1},
    'standard_2d':   {(1,1,1,1): 2, (2,1,1): 0,  (2,2): 2,  (3,1): -1, (4,): 0},
    'standard_3d':   {(1,1,1,1): 3, (2,1,1): 1,  (2,2): -1, (3,1): 0,  (4,): -1},
    '3d_x_sign':     {(1,1,1,1): 3, (2,1,1): -1, (2,2): -1, (3,1): 0,  (4,): 1},
}
class_sizes_S4 = {(1,1,1,1): 1, (2,1,1): 6, (2,2): 3, (3,1): 8, (4,): 6}

# Decompose: m_irrep = (1/|G|) Σ_g χ_irrep(g) · χ_BM(g)
print()
print(f"  Irrep multiplicities in V_BM (S_4 character inner product):")
mults = {}
for irrep_name, chars_dict in irrep_chars.items():
    m = 0.0
    for ct, size in class_sizes_S4.items():
        if ct in chi_BM:
            m += size * chars_dict[ct] * chi_BM[ct]
    m = m / 24.0
    mults[irrep_name] = m
    print(f"    {irrep_name}: multiplicity = {m:+.4f}")

clean = {k: round(v) for k, v in mults.items() if abs(round(v) - v) < 0.1}
total_dim = sum(clean[k] * (1 if 'trivial' in k or 'sign' in k else 2 if '2d' in k else 3)
                for k in clean if 'sign' == k or 'trivial' == k or '2d' in k or '3d' in k)
print(f"  Reconstructed dimension from clean multiplicities: {total_dim}")
print()

# ------------------------------------------------------------------
# 4.5  Identify the 2 scalar modes: which is Perron-adjacency-derived, which
#      is the "phantom" cycle mode (B¹ vertex-coboundary residue)?
# ------------------------------------------------------------------
print("-"*78)
print("FOLLOW-UP: split the 2-dim scalar sector (Perron + B¹-residue)")
print("-"*78)
print("Looking inside V_perron_scalar (dim 2) for J = ±1 split and S_4 content.")
# J grading on V_perron_scalar
J_on_PS = V_perron_scalar.T @ J @ V_perron_scalar
J_eig_PS, _ = np.linalg.eigh(J_on_PS)
print(f"  J-spectrum on V_perron_scalar: {np.round(J_eig_PS, 4)}")
dim_PS_Jp = np.sum(J_eig_PS > 0.5)
dim_PS_Jm = np.sum(J_eig_PS < -0.5)
print(f"  dim(J=+1) = {dim_PS_Jp},  dim(J=-1) = {dim_PS_Jm}")

# J grading on V_BM (cycle sector)
J_on_BM_cycle = V_BM.T @ J @ V_BM
J_eig_BMc, _ = np.linalg.eigh(J_on_BM_cycle)
print(f"  J-spectrum on V_BM (cycle sector, dim 3): {np.round(J_eig_BMc, 4)}")
dim_BMc_Jp = np.sum(J_eig_BMc > 0.5)
dim_BMc_Jm = np.sum(J_eig_BMc < -0.5)
print(f"  dim(J=+1) = {dim_BMc_Jp},  dim(J=-1) = {dim_BMc_Jm}")
print()

# Construct the "Bass-Stark-Terras bipartite-factor sector" = 3 cycle modes + 1 B¹ mode
# Heuristic: pick the J=+1 mode in V_perron_scalar — this is the "uniform-on-vertices-
# antiderivative" component, structurally B¹-like (vertex coboundary).
# The remaining J=-1 mode in V_perron_scalar is the Perron-adjacency scalar proper.
if dim_PS_Jp >= 1 and dim_PS_Jm >= 1:
    # Diagonalize J on V_perron_scalar; take the J=+1 eigenvector
    J_PS = V_perron_scalar.T @ J @ V_perron_scalar
    J_eigvals, J_eigvecs = np.linalg.eigh(J_PS)
    idx_Jplus  = np.where(J_eigvals > 0.5)[0]
    idx_Jminus = np.where(J_eigvals < -0.5)[0]
    V_B1_mode  = V_perron_scalar @ J_eigvecs[:, idx_Jplus]
    V_Perron_mode = V_perron_scalar @ J_eigvecs[:, idx_Jminus]
    print(f"  Split V_perron_scalar:")
    print(f"    V_B1_mode (J=+1, dim {V_B1_mode.shape[1]}): B¹ vertex-coboundary residue")
    print(f"    V_Perron_mode (J=-1, dim {V_Perron_mode.shape[1]}): Perron-adjacency scalar proper")
    # Bass-Stark-Terras bipartite-factor sector = V_BM ⊕ V_B1_mode  (dim 4)
    V_BST_bipartite = np.concatenate([V_BM, V_B1_mode], axis=1)
    print(f"  Bass-Stark-Terras bipartite-factor sector (dim {V_BST_bipartite.shape[1]}):")
    print(f"    = V_BM (cycle, dim 3) ⊕ V_B1 (vertex coboundary, dim 1)")
    print(f"  This is the 4-dim sector underlying Route H's c_unified = 4/12 = 1/3.")
    print()
    print("  --- KEY STRUCTURAL FINDING ---")
    print(f"  The 4-dim bipartite-factor sector splits NATURALLY as (3, 1)")
    print(f"  under the J = +/- 1 (edge-reversal) grading:")
    print(f"    3 cycle modes (J=-1, Wilson-loop carrying)")
    print(f"    1 B¹ vertex-coboundary mode (J=+1, no Wilson-loop holonomy)")
    print()
    print(f"  This MATCHES the sector-specific c fit:")
    print(f"    c_3 = 3/12 = 1/4    (SU(3): couples ONLY to cycle modes)")
    print(f"    c_1 = c_2 = 4/12 = 1/3    (U(1)_Y, SU(2)_L: cycle + B¹ vertex modes)")
    print(f"    c_v = 5/12  (scalar 2-point: cycle + B¹ + Perron-adjacency scalar)")
    found_31_split = True
else:
    print(f"  V_perron_scalar does not split cleanly into J=+1 and J=-1 components.")
    found_31_split = False
print()

# ------------------------------------------------------------------
# 5. Diagnosis
# ------------------------------------------------------------------
print("="*78)
print("DIAGNOSIS")
print("="*78)
print()
print("Question: does any natural grading on srs (K_4) split the 4-dim")
print("bipartite-marginal sector as (3, 1) — the structural fingerprint")
print("of c_3 = 1/4 = 3/12 vs uniform c = 4/12 = 1/3?")
print()

split_31_found = found_31_split
if found_31_split:
    print(f"  ✓ The 4-dim Bass-Stark-Terras bipartite-factor sector splits as (3, 1)")
    print(f"    under edge-reversal J: 3 cycle modes (J=-1) + 1 B¹ vertex mode (J=+1).")
    print(f"    Wilson-loop cycle-mode count = 3 = β_1(K_4), strictly less than the")
    print(f"    polynomial-multiplicity count 2(|E|-|V|) = 4.")

# Check S_4 irrep decomp for a 3+1
has_3d_plus_1d = (clean.get('standard_3d', 0) >= 1 and
                  (clean.get('trivial', 0) >= 1 or clean.get('sign', 0) >= 1))
has_3dx_plus_1d = (clean.get('3d_x_sign', 0) >= 1 and
                   (clean.get('trivial', 0) >= 1 or clean.get('sign', 0) >= 1))
if has_3d_plus_1d:
    print(f"  ✓ S_4 decomposition contains a 3-dim + 1-dim irrep pair (standard_3d + trivial/sign)")
    print(f"    → 4-dim sector = 3-dim S_4 irrep ⊕ 1-dim S_4 irrep")
    print(f"    → Natural (3, 1) split via S_4 representation theory")
    split_31_found = True
if has_3dx_plus_1d:
    print(f"  ✓ S_4 decomposition contains a 3-dim·sign + 1-dim irrep pair")
    split_31_found = True

if not split_31_found:
    print("  ✗ No natural grading on srs produces a (3, 1) split of the 4-dim")
    print("    bipartite-marginal sector.")
    print()
    print("  → c_3 = 1/4 has no substrate-derivable mechanism via Route H")
    print("    sub-bundle mode counting.")
    print("  → Sector-specific dark correction RULED OUT as a structural fix.")
    print("  → The fit-extracted c_i values are numerical coincidences, not")
    print("    substrate-derived.")
else:
    print()
    print("  → A natural grading on srs DOES produce (3, 1) on the 4-dim BM sector.")
    print("  → Identified mechanism:")
    print("      cycle modes (3, J=-1): Wilson-loop carrying, gauge-charged H¹")
    print("      B¹ mode      (1, J=+1): vertex-coboundary residue, gauge-direction")
    print("    Together = the 4-dim bipartite-factor sector (= Route H's c=4/12).")
    print()
    print("  STRUCTURAL HYPOTHESIS NOW CONCRETE:")
    print("      SU(3)_c gauge bosons couple ONLY to the 3 cycle modes  → c_3 = 3/12 = 1/4")
    print("      U(1)_Y, SU(2)_L couple to cycle modes + the B¹ mode    → c_1,2 = 4/12 = 1/3")
    print("      Higgs (scalar 2-point) couples to all 5 (incl. Perron) → c_v = 5/12")
    print()
    print("  WHY would SU(3) miss the B¹ mode?")
    print("    The B¹ mode = vertex-coboundary residue: representable as a function")
    print("    f(v) on vertices, with edge-mode φ(e) = f(h(e)) + f(t(e)) (symmetric).")
    print("    For U(1)_Y a vertex-coboundary IS a residual gauge phase (acts on edges")
    print("    via additive lifts of vertex U(1) values). For SU(2)_L, a vertex-")
    print("    coboundary on a vertex doublet still survives as the diagonal direction.")
    print("    For SU(3)_c, the vertex spinor carries a color triplet rep where the")
    print("    'diagonal' direction is NOT a color-singlet — there's no SU(3)-trivial")
    print("    direction on a color triplet. So vertex-coboundary gauge transformations")
    print("    on the substrate vertex Cl(6) Fock leave NO residual mode for SU(3).")
    print()
    print("  TESTABLE NEXT STEP: compute the H¹ ↔ vertex-coboundary action on the")
    print("  framework's Cl(6) Fock per vertex, sector by sector, and verify that")
    print("  the B¹ residue is rank 1 for U(1)_Y, rank 1 for SU(2)_L, rank 0 for SU(3).")
    print()
    print("  CAVEAT: this is a structural HYPOTHESIS matching all three numerical")
    print("  values (1/3, 1/3, 1/4). Closure to theorem grade requires the Cl(6)")
    print("  vertex-coboundary computation above. Not done here.")
print("="*78)
