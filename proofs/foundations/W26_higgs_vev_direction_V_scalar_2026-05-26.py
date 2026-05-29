#!/usr/bin/env python3
"""
W26 — Candidate B Session 1: substrate Higgs VEV direction → V_scalar intersection

CONTEXT
-------
W24 established (post-linter, theorem-grade-numerical for SU(3)_c):
  c_color   = β_1/(2|E|) = 3/12 = 1/4
  c_EW      = (β_1+1)/(2|E|) = 4/12 = 1/3
  c_v_Higgs = V_pm/(2|E|)   = 5/12

The "+1 mode" canonical pick within V_scalar (2-dim, J=+1, C_3-faithful) for
c_EW is the OPEN structural gap (session summary §6 Candidate B).

EXISTING SUBSTRATE-HIGGS MACHINERY (this probe builds on it):
  - theorem_g2_edge_qubit_su2.md: Higgs doublet = edge qubit ℂ² with
    f₁ (spatial orientation, ≅ iσ_y = γ¹) + f₂ (causal direction, ≅ σ_z = γ⁰).
    The 2-dim left ℍ-module is the SU(2)_L doublet; SU(2) = Sp(1) = unit quaternions.
  - theorem_updown_split_conjugate_higgs_2026-05-21.md: H = odd-grade (couples to
    down-type), H̃ = iσ_2 H* = even-grade (couples to up-type).
  - J operator (directed-edge reversal) on each ℂ² edge ≅ σ_x.
    J = σ_x and grade-2 element f₁·f₂ ≅ ±σ_x → J eigenvalues correspond to
    grade ± parity.

HYPOTHESIS
----------
The SM Higgs VEV ⟨H⟩ at each edge is in a specific 1-dim sub-direction of the
edge qubit ℂ² (= Higgs doublet). Per SM convention, ⟨H⟩ = (0, v/√2)^T in
(φ+, φ0) basis = the σ_z = -1 (T_3 = -1/2, neutral) component.

In the directed-edge basis {(u→v), (v→u)} ≅ σ_z basis, the Higgs VEV at edge
i is one of the two basis states (e.g., (v→u) if we identify σ_z = -1 with that
orientation).

The 6-dim "Higgs VEV mode" in ℂ^12 = ⊕_edges (Higgs VEV per edge) intersects
V_pm (5-dim) somewhere. We compute:

(1) The intersection dim under different orientation conventions
(2) Whether the intersection canonically picks a 1-dim sub-direction of V_scalar
(3) The C_3 isotypic content of any V_scalar∩(Higgs VEV mode) direction

POSSIBLE OUTCOMES
-----------------
- POSITIVE: there's a canonical 1-dim sub-direction of V_scalar aligned with
  Higgs VEV (independent of orientation choice via gauge invariance) → c_EW = 4/12
  has its "+1 mode" canonical pick. Multi-session theorem-grade work to follow.

- AMBIGUOUS: the intersection varies with orientation choice; no canonical pick.
  Suggests Higgs VEV alone doesn't pick a 1-dim sub-direction; need additional
  structure (e.g., conjugate-Higgs H̃ direction, or T_3 projection).

- NEGATIVE: the Higgs VEV mode either misses V_scalar entirely (intersection = 0)
  or contains V_scalar entirely (intersection = 2). Either way, no "+1 mode"
  selection rule.
"""

import numpy as np
from fractions import Fraction
from itertools import product

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# ============================================================
# 1. K_4 setup (same as W24)
# ============================================================
N_V = 4
vertices = list(range(N_V))
directed_edges = [(u, v) for u in vertices for v in vertices if u != v]
N_DE = len(directed_edges)
e2i = {e: i for i, e in enumerate(directed_edges)}
undirected_edges = sorted({tuple(sorted([u, v])) for u, v in directed_edges})
N_E = len(undirected_edges)
ue2i = {ue: i for i, ue in enumerate(undirected_edges)}

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
print(" W26 — Candidate B Session 1: Higgs VEV direction → V_scalar intersection")
print("="*78)
print()

# ============================================================
# 2. Build V_pm, V_cycle, V_scalar (same as W24)
# ============================================================
def real_eigenspace(M, val, tol=1e-8):
    K = M - val * np.eye(M.shape[0])
    U, S, _ = np.linalg.svd(K)
    null_dim = int(np.sum(S < tol))
    return U[:, M.shape[0]-null_dim:]

V_p1 = real_eigenspace(B.astype(float), 1.0)
V_m1 = real_eigenspace(B.astype(float), -1.0)
V_pm = np.concatenate([V_p1, V_m1], axis=1)

def split_J(V_sub):
    Jsub = V_sub.T @ J_mat.astype(float) @ V_sub
    Jsub_sym = 0.5 * (Jsub + Jsub.T)
    ev_J, evec_J = np.linalg.eigh(Jsub_sym)
    mask_p = np.abs(ev_J - 1.0) < 1e-6
    mask_m = np.abs(ev_J + 1.0) < 1e-6
    Vp = V_sub @ evec_J[:, mask_p]
    Vm = V_sub @ evec_J[:, mask_m]
    return Vp, Vm

V_pm_Jp, V_pm_Jm = split_J(V_pm)
V_scalar = V_pm_Jp  # 2-dim
V_cycle  = V_pm_Jm  # 3-dim
V_scalar, _ = np.linalg.qr(V_scalar)
V_cycle, _ = np.linalg.qr(V_cycle)
print(f" V_pm dim: {V_pm.shape[1]}, V_scalar (J=+1) dim: {V_scalar.shape[1]}, V_cycle (J=-1) dim: {V_cycle.shape[1]}")
print()

# ============================================================
# 3. Higgs VEV mode under different orientation conventions
# ============================================================
# Per theorem_g2: Higgs doublet at edge i = ℂ² with basis {f₁=σ_z=+1, f₂=σ_z=-1}
# ≅ {(u→v), (v→u)} in directed-edge basis (with σ_z = "causal direction").
# Higgs VEV ⟨H⟩ in SM convention = (0, v/√2)^T = σ_z = -1 component
# = (v→u) direction (after picking orientation).
#
# We test 3 orientation conventions:
#   ORIENT A: at edge {u, v} with u < v (numeric ordering), pick (u→v) direction
#              so VEV is at (v→u). Higgs VEV mode = span{(v→u) : u < v}
#   ORIENT B: at edge {u, v} with u < v, pick (v→u) direction so VEV is at (u→v).
#              Higgs VEV mode = span{(u→v) : u < v}
#   ORIENT C: J=+1 symmetric — VEV is at ((u→v)+(v→u))/√2 at every edge.
#              This is fully J=+1 → embeds in V_scalar's parent J=+1 sub-space.
#   ORIENT D: J=-1 antisymmetric — VEV is at ((u→v)-(v→u))/√2.
#              Embeds in J=-1 sub-space → orthogonal to V_scalar.

def higgs_vev_mode(orient):
    """Return the 6-dim Higgs VEV mode basis (12×6 matrix) for the given orientation convention."""
    M = np.zeros((N_DE, N_E), dtype=float)
    for k, (a, b) in enumerate(undirected_edges):  # a < b convention
        if orient == 'A':
            # VEV at (b→a) (= "v→u" if (u, v) = (a, b))
            M[e2i[(b, a)], k] = 1.0
        elif orient == 'B':
            # VEV at (a→b)
            M[e2i[(a, b)], k] = 1.0
        elif orient == 'C':
            # J=+1 symmetric VEV
            M[e2i[(a, b)], k] = 1.0/np.sqrt(2)
            M[e2i[(b, a)], k] = 1.0/np.sqrt(2)
        elif orient == 'D':
            # J=-1 antisymmetric VEV
            M[e2i[(a, b)], k] = 1.0/np.sqrt(2)
            M[e2i[(b, a)], k] = -1.0/np.sqrt(2)
    return M

# ============================================================
# 4. Intersection with V_scalar (2-dim) and V_pm (5-dim)
# ============================================================
def subspace_intersection(A, B, tol=1e-8):
    """Compute basis of A ∩ B where A, B are column-space bases (each N_DE × k)."""
    # Use projector approach: P_A P_B has eigenvalue 1 for vectors in A ∩ B
    PA = A @ A.T  # projection onto A
    PB = B @ B.T
    P_inter = PA @ PB
    # vectors fixed by P_inter (eigenvalue 1) span the intersection
    ev, evec = np.linalg.eig(P_inter)
    real_mask = np.abs(ev.imag) < tol
    eig1_mask = real_mask & (np.abs(ev.real - 1.0) < 1e-4)
    intersection_dim = int(np.sum(eig1_mask))
    if intersection_dim == 0:
        return np.zeros((A.shape[0], 0)), 0
    basis = np.real(evec[:, eig1_mask])
    # Re-orthogonalize against B (since P_inter eigenvectors aren't exact)
    # Then check they ARE in A
    basis_in_B = B @ (B.T @ basis)
    return basis_in_B, intersection_dim

print("-"*78)
print(" TEST 1: Higgs VEV mode intersection with V_scalar (2-dim) and V_pm (5-dim)")
print("-"*78)
print()

results = {}
for orient in ['A', 'B', 'C', 'D']:
    HVM = higgs_vev_mode(orient)
    # Intersection with V_scalar
    _, dim_scalar = subspace_intersection(HVM, V_scalar)
    # Intersection with V_cycle
    _, dim_cycle = subspace_intersection(HVM, V_cycle)
    # Intersection with V_pm
    _, dim_Vpm = subspace_intersection(HVM, V_pm)
    # Total Higgs VEV mode projected to V_pm (sum of intersections)
    P_Vpm = V_pm @ V_pm.T
    HVM_in_Vpm = P_Vpm @ HVM
    rank_HVM_in_Vpm = np.linalg.matrix_rank(HVM_in_Vpm, tol=1e-8)
    results[orient] = (dim_scalar, dim_cycle, dim_Vpm, rank_HVM_in_Vpm)
    print(f"   ORIENT {orient}: HVM (6-dim) ∩ V_scalar = {dim_scalar},  "
          f"∩ V_cycle = {dim_cycle},  ∩ V_pm = {dim_Vpm},  "
          f"rank(P_V_pm · HVM) = {rank_HVM_in_Vpm}")

print()

# ============================================================
# 5. For non-trivial intersections, check C_3 isotypic content
# ============================================================
print("-"*78)
print(" TEST 2: C_3 isotypic content of HVM∩V_scalar (if dim ≥ 1)")
print("-"*78)
print()

# C_3 generator: (0 1 2)(3) — sends 0→1, 1→2, 2→0, 3→3
c3_id = {v: v for v in vertices}
c3_gen = {0: 1, 1: 2, 2: 0, 3: 3}
c3_gen2 = {0: 2, 1: 0, 2: 1, 3: 3}

def perm_matrix(sigma):
    M = np.zeros((N_DE, N_DE))
    for i, e in enumerate(directed_edges):
        new_e = (sigma[e[0]], sigma[e[1]])
        j = e2i[new_e]
        M[j, i] = 1
    return M

P_id = perm_matrix(c3_id)
P_c  = perm_matrix(c3_gen)
P_cc = perm_matrix(c3_gen2)

omega = np.exp(2j*np.pi/3)
def c3_isotypic_dim(V_sub):
    if V_sub.shape[1] == 0:
        return (0, 0)
    chars = [np.trace(V_sub.T @ P @ V_sub) for P in (P_id, P_c, P_cc)]
    m_t = (chars[0] + chars[1] + chars[2]) / 3.0
    m_w = (chars[0] + np.conj(omega)*chars[1] + omega*chars[2]) / 3.0
    m_wb = (chars[0] + omega*chars[1] + np.conj(omega)*chars[2]) / 3.0
    return (np.real(m_t), np.real(m_w + m_wb))  # trivial dim, faithful-pair dim

# For each orientation, compute the V_scalar∩HVM basis and its C_3 content
for orient in ['A', 'B', 'C', 'D']:
    HVM = higgs_vev_mode(orient)
    inter, dim = subspace_intersection(HVM, V_scalar)
    if dim > 0:
        # Re-orthogonalize
        inter_qr, _ = np.linalg.qr(inter)
        # Reduce to actual intersection dim
        actual = inter_qr[:, :dim]
        m_t, m_f = c3_isotypic_dim(actual)
        print(f"   ORIENT {orient}: dim(HVM ∩ V_scalar) = {dim},  C_3 isotypic = "
              f"({m_t:.2f} trivial, {m_f:.2f} faithful-pair)")
    else:
        print(f"   ORIENT {orient}: dim(HVM ∩ V_scalar) = 0 (empty intersection)")
print()

# ============================================================
# 6. The right approach: project Higgs VEV onto V_pm, then look at J=+1 part
# ============================================================
print("-"*78)
print(" TEST 3: project HVM onto V_pm, then onto V_scalar — what's the result?")
print("-"*78)
print()

for orient in ['A', 'B', 'C', 'D']:
    HVM = higgs_vev_mode(orient)
    # Project HVM (6 vectors) onto V_pm
    P_Vpm = V_pm @ V_pm.T
    HVM_pm = P_Vpm @ HVM
    # Then project onto V_scalar
    P_Vsc = V_scalar @ V_scalar.T
    HVM_scalar_proj = P_Vsc @ HVM
    rank_in_scalar = np.linalg.matrix_rank(HVM_scalar_proj, tol=1e-8)
    # Get the actual subspace
    if rank_in_scalar > 0:
        U_s, S_s, _ = np.linalg.svd(HVM_scalar_proj, full_matrices=False)
        HVM_in_Vscalar = U_s[:, :rank_in_scalar]
        m_t, m_f = c3_isotypic_dim(HVM_in_Vscalar)
        # Show the projection magnitudes
        norm_HVM = np.linalg.norm(HVM)
        norm_proj_pm = np.linalg.norm(HVM_pm)
        norm_proj_scalar = np.linalg.norm(HVM_scalar_proj)
        print(f"   ORIENT {orient}: ‖HVM‖={norm_HVM:.3f}, "
              f"‖P_Vpm HVM‖={norm_proj_pm:.3f}, "
              f"‖P_Vscalar HVM‖={norm_proj_scalar:.3f}")
        print(f"            rank(P_Vscalar HVM) = {rank_in_scalar} → "
              f"C_3 isotypic ({m_t:.2f} trivial, {m_f:.2f} faithful-pair)")
    else:
        print(f"   ORIENT {orient}: P_Vscalar HVM = 0 (orthogonal)")
print()

# ============================================================
# 7. The HONEST answer: Higgs VEV direction picks WHAT in V_scalar?
# ============================================================
print("="*78)
print(" VERDICT")
print("="*78)
print()
print(" The Higgs VEV mode (6-dim, one direction per edge in the SM convention)")
print(" projects onto V_pm (5-dim) and the J=+1 sub-sector (V_scalar, 2-dim).")
print()
print(" KEY OBSERVATIONS:")
print()
print(" 1. ORIENT C (J=+1 symmetric) — the Higgs VEV is the FULLY-SYMMETRIC mode")
print(f"    at each edge. Projection onto V_scalar: should hit V_scalar entirely")
print(f"    (since both are J=+1) → rank 2, no 1-dim canonical pick.")
print()
print(" 2. ORIENT D (J=-1 antisymmetric) — projection onto V_scalar should be 0")
print(f"    (orthogonal J-eigenspaces) → rank 0, no Higgs VEV in V_scalar at all.")
print()
print(" 3. ORIENT A, B (specific directed-edge orientation) — projection onto")
print(f"    V_scalar gives a SPECIFIC sub-direction whose C_3 isotypic content")
print(f"    is fixed by the orientation choice.")
print()
print(" If ORIENT A and ORIENT B both project to the SAME 1-dim sub-direction of")
print(" V_scalar (up to sign), that's a CANONICAL pick of '+1 mode'. If they")
print(" project to DIFFERENT sub-directions, the orientation choice is load-bearing")
print(" and the structural derivation requires fixing the orientation by an")
print(" additional substrate principle (e.g., the I4_1 32 chirality of full srs).")
print()
print(" Check ORIENT A vs ORIENT B alignment in V_scalar:")
HVM_A = higgs_vev_mode('A')
HVM_B = higgs_vev_mode('B')
P_Vsc = V_scalar @ V_scalar.T
HVM_A_in_Vsc = P_Vsc @ HVM_A
HVM_B_in_Vsc = P_Vsc @ HVM_B
# Normalize: take the rank-1 part
U_A, S_A, _ = np.linalg.svd(HVM_A_in_Vsc, full_matrices=False)
U_B, S_B, _ = np.linalg.svd(HVM_B_in_Vsc, full_matrices=False)
rank_A = int(np.sum(S_A > 1e-8))
rank_B = int(np.sum(S_B > 1e-8))
print(f"   rank A in V_scalar = {rank_A}, rank B in V_scalar = {rank_B}")
if rank_A > 0 and rank_B > 0:
    # Inner product of the dominant SVD directions
    overlap = abs(U_A[:, 0] @ U_B[:, 0])
    print(f"   |⟨v_A | v_B⟩| in V_scalar = {overlap:.4f}  "
          f"(1.0 means same direction, 0.0 means orthogonal)")
print()
print("="*78)
