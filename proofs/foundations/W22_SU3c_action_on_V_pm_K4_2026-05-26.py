#!/usr/bin/env python3
"""
W22 — Session 1, Cut 1: SU(3)_c color action on V_pm via K_4 edge-coloring

CONTEXT
-------
Yesterday's session + W21 established that:
  - V_pm (5-dim, u=±1 eigenspace of B on K_4) splits into V_cycle (3, J=-1)
    + V_scalar (2, J=+1).
  - V_scalar is C_3-irreducible (faithful 2-pair).
  - The BS-T (1 Perron-adj + 1 bipartite-extra) split within V_scalar is
    NON-CANONICAL at the graph level (W21 finding).

For sector-specific c values (c_color = 1/4, c_EM = 1/3, c_v_Higgs = 5/12),
we need an EXTERNAL gauge structure to canonically partition V_pm modes.

THIS PROBE
----------
Defines SU(3)_c on K_4 via the canonical 3-edge-coloring:
  Color R = {(0,1), (2,3)}     ← perfect matching 1
  Color G = {(0,2), (1,3)}     ← perfect matching 2
  Color B = {(0,3), (1,2)}     ← perfect matching 3

At each vertex v, the 3 incident edges are exactly one of each color, so
SU(3)_c global rotation g ∈ SU(3) acts on directed edges via the fundamental
representation on the 3 outgoing colors at each source vertex.

TESTS
-----
(T1) Does the SU(3)_c action commute with the Hashimoto matrix B?
     - If YES, SU(3)_c preserves V_pm and the gauge action is well-defined.
     - If NO, this SU(3)_c definition is inconsistent with substrate dynamics
       and we need a different lift.

(T2) Restricted to V_pm, what is the kernel of each SU(3)_c generator?
     - Generators tested: the 8 Gell-Mann matrices λ^a (a=1,...,8).
     - Common kernel dim = dim of SU(3)_c-singlet sector of V_pm.

(T3) Does the SU(3)_c-singlet sector of V_pm coincide with V_scalar (yesterday's
     J=+1 sector, dim 2)?
     - If YES: c_color = V_cycle / 12 = 3/12 = 1/4 closes structurally.
     - If NO: either V_scalar mixes with V_cycle under SU(3)_c, or the singlet
       has different dimension. In either case, c_color = 1/4 needs a different
       derivation.
"""

import numpy as np
from fractions import Fraction
import itertools

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# ============================================================
# 1. K_4 setup (same conventions as yesterday)
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
print(" W22 — SU(3)_c color action on V_pm via K_4 edge-coloring")
print("="*78)
print()

# ============================================================
# 2. Canonical 3-edge-coloring of K_4 via perfect matchings
# ============================================================
# K_4 is class-1 (3-edge-colorable). The 3 perfect matchings:
#   Color 0 (R): {(0,1), (2,3)}
#   Color 1 (G): {(0,2), (1,3)}
#   Color 2 (B): {(0,3), (1,2)}
color_classes = {
    0: [(0, 1), (2, 3)],
    1: [(0, 2), (1, 3)],
    2: [(0, 3), (1, 2)],
}
edge_color = {}  # undirected edge → color
for c, edges in color_classes.items():
    for (a, b) in edges:
        edge_color[(a, b)] = c
        edge_color[(b, a)] = c

# Sanity check: each vertex sees one of each color
print(" Edge coloring check: each vertex sees one of each color")
for v in vertices:
    colors_at_v = set()
    for u in vertices:
        if u == v:
            continue
        colors_at_v.add(edge_color[(v, u)])
    assert colors_at_v == {0, 1, 2}, f"vertex {v} doesn't see all 3 colors: {colors_at_v}"
    print(f"   vertex {v}: edges → {[(u, edge_color[(v,u)]) for u in vertices if u != v]}")
print(" ✓ 3-edge-coloring is consistent.")
print()

# ============================================================
# 3. Build the "source-vertex color basis": at each vertex v, the 3 outgoing
#    edges form an ordered triple by color (0, 1, 2). This gives a natural
#    basis isomorphism: ℂ^12 ≅ ⊕_v (ℂ^3 = color rep at v).
# ============================================================
# Order: for each vertex v, the 3 outgoing edges sorted by color (0, 1, 2).
def vertex_color_index(v, c):
    """Return the directed-edge index of the outgoing edge at v with color c."""
    for u in vertices:
        if u == v:
            continue
        if edge_color[(v, u)] == c:
            return e2i[(v, u)]
    raise ValueError(f"no edge of color {c} at vertex {v}")

# Build the permutation P that re-orders directed_edges into
# (vertex 0 color 0, v0 c1, v0 c2, v1 c0, ..., v3 c2).
P_reorder = np.zeros((N_DE, N_DE), dtype=int)
new_basis = []  # list of original e2i indices in new order
for v in vertices:
    for c in (0, 1, 2):
        new_basis.append(vertex_color_index(v, c))
for new_i, orig_i in enumerate(new_basis):
    P_reorder[new_i, orig_i] = 1

print(" Source-vertex-color basis (re-ordering): for each v ∈ {0..3}, color c ∈ {0..2},")
print(" the outgoing edge at v with color c:")
for v in vertices:
    print(f"   v={v}: " + ", ".join(f"c={c}→{directed_edges[vertex_color_index(v, c)]}" for c in (0, 1, 2)))
print()

# B in the new basis
B_new = P_reorder @ B.astype(float) @ P_reorder.T
J_new = P_reorder @ J_mat.astype(float) @ P_reorder.T

# ============================================================
# 4. SU(3)_c action: global SU(3) rotation g acts on each vertex's color triplet
# ============================================================
# Gell-Mann matrices λ^a (a = 1..8), normalized so tr(λ^a λ^b) = 2 δ^{ab}
GM = {}
GM[1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
GM[2] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
GM[3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
GM[4] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
GM[5] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
GM[6] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
GM[7] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
GM[8] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3)

# Block-diagonal SU(3)_c action on ℂ^12 (in the new basis)
def lambda_to_full(GMa):
    """Embed SU(3) generator on ℂ^12 (new basis) as I_4 ⊗ GMa."""
    return np.kron(np.eye(N_V), GMa)

T_a = {a: lambda_to_full(GM[a]) for a in range(1, 9)}

# Map back to original basis: T^a_orig = P^T T^a_new P
T_a_orig = {a: P_reorder.T @ T_a[a] @ P_reorder for a in range(1, 9)}

# ============================================================
# 5. Test (T1): does SU(3)_c commute with B?
# ============================================================
print("-"*78)
print(" TEST T1: does SU(3)_c commute with B?")
print("-"*78)
print()
commutator_norms = {}
for a in range(1, 9):
    comm = T_a_orig[a] @ B.astype(complex) - B.astype(complex) @ T_a_orig[a]
    nrm = np.linalg.norm(comm)
    commutator_norms[a] = nrm
    print(f"   ‖[T^{a}, B]‖_F = {nrm:.6f}")
print()
all_commute = all(nrm < 1e-8 for nrm in commutator_norms.values())
if all_commute:
    print(" ✓ All 8 SU(3)_c generators commute with B.")
    print("   ⇒ SU(3)_c preserves V_pm and the gauge action is well-defined on V_pm.")
else:
    print(" ✗ Not all SU(3)_c generators commute with B.")
    print("   ⇒ This SU(3)_c definition is inconsistent with substrate dynamics.")
    print("   The natural global SU(3)_c via color-rotation-at-source-vertex does NOT")
    print("   define a symmetry of the Hashimoto matrix.")
    max_nrm = max(commutator_norms.values())
    print(f"   Max commutator norm: {max_nrm:.4f}")
print()

# Even if SU(3)_c doesn't commute with B, project to V_pm to see what the structure looks like
# ============================================================
# 6. Build V_pm (5-dim u=±1 eigenspace of B)
# ============================================================
ev, evec = np.linalg.eig(B.astype(float))
mp = np.abs(ev - 1.0) < 1e-8
mm = np.abs(ev + 1.0) < 1e-8
V_p = np.real_if_close(evec[:, mp], tol=1000)
V_m = np.real_if_close(evec[:, mm], tol=1000)
if np.iscomplexobj(V_p): V_p = np.real(V_p)
if np.iscomplexobj(V_m): V_m = np.real(V_m)
V_p, _ = np.linalg.qr(V_p)
V_m, _ = np.linalg.qr(V_m)
V_pm = np.concatenate([V_p, V_m], axis=1)
assert V_pm.shape[1] == 5, f"V_pm dim = {V_pm.shape[1]}, expected 5"

# Build V_scalar (J=+1) and V_cycle (J=-1)
J_in_Vpm = V_pm.T @ J_mat.astype(float) @ V_pm
ev_J, evec_J = np.linalg.eig(J_in_Vpm)
mask_J_p = np.abs(ev_J.real - 1.0) < 1e-6
mask_J_m = np.abs(ev_J.real + 1.0) < 1e-6
V_scalar = V_pm @ np.real(evec_J[:, mask_J_p])
V_cycle  = V_pm @ np.real(evec_J[:, mask_J_m])
V_scalar, _ = np.linalg.qr(V_scalar)
V_cycle, _ = np.linalg.qr(V_cycle)

# ============================================================
# 7. Test (T2): Restrict each T^a to V_pm and find the common kernel
# ============================================================
print("-"*78)
print(" TEST T2: SU(3)_c action on V_pm — common kernel structure")
print("-"*78)
print()

# T^a restricted to V_pm: 5×5 matrix per a
T_a_Vpm = {a: V_pm.T @ np.real(T_a_orig[a]) @ V_pm for a in range(1, 9)}

# Print restricted matrices
print(" T^a restricted to V_pm (5×5, in V_pm orthonormal basis):")
for a in (3, 8):  # The 2 Cartan generators (diagonal)
    print(f"   T^{a} (Cartan, diagonal in color basis):")
    print(f"   {T_a_Vpm[a].round(4)}")
    print()

# Compute the common kernel: vectors v ∈ V_pm with T^a v = 0 for all a
# Equivalently: kernel of the 40×5 matrix stacking all T^a|V_pm
stack = np.vstack([T_a_Vpm[a] for a in range(1, 9)])
print(f" Stacked T^a|V_pm matrix shape: {stack.shape}")
# Use SVD to find the kernel
U_s, S_s, Vt_s = np.linalg.svd(stack)
print(f" Singular values of stacked T^a|V_pm:")
print(f"   {S_s.round(4)}")
print()

kernel_dim = int(np.sum(S_s < 1e-8))
kernel_basis = Vt_s[-kernel_dim:].T if kernel_dim > 0 else np.zeros((5, 0))
print(f" Common kernel dim (SU(3)_c-singlet subspace of V_pm): {kernel_dim}")
print()

# ============================================================
# 8. Test (T3): does the SU(3)_c-singlet subspace = V_scalar?
# ============================================================
print("-"*78)
print(" TEST T3: does ker(SU(3)_c |_{V_pm}) coincide with V_scalar?")
print("-"*78)
print()
# Coordinates of V_scalar in V_pm basis:
V_scalar_coords = V_pm.T @ V_scalar  # 5x2
V_cycle_coords = V_pm.T @ V_cycle    # 5x3
print(f" V_scalar in V_pm coords ({V_scalar_coords.shape}):")
print(V_scalar_coords.round(4))
print()
print(f" V_cycle in V_pm coords ({V_cycle_coords.shape}):")
print(V_cycle_coords.round(4))
print()

if kernel_dim > 0:
    print(f" Kernel basis in V_pm coords ({kernel_basis.shape}):")
    print(kernel_basis.round(4))
    print()

    # Check: is the kernel basis a subspace of V_scalar?
    # Project kernel onto V_scalar and check if norm is preserved
    P_scalar = V_scalar_coords @ V_scalar_coords.T  # 5×5 projector onto V_scalar in V_pm coords
    P_cycle = V_cycle_coords @ V_cycle_coords.T
    for k in range(kernel_dim):
        ker_v = kernel_basis[:, k]
        proj_scalar = P_scalar @ ker_v
        proj_cycle = P_cycle @ ker_v
        norm_scalar = np.dot(proj_scalar, proj_scalar)
        norm_cycle = np.dot(proj_cycle, proj_cycle)
        total = np.dot(ker_v, ker_v)
        print(f"   kernel vector {k}:")
        print(f"     ‖proj_V_scalar‖² / ‖v‖² = {norm_scalar/total:.4f}")
        print(f"     ‖proj_V_cycle‖² / ‖v‖²  = {norm_cycle/total:.4f}")
print()

# ============================================================
# 9. VERDICT
# ============================================================
print("="*78)
print(" VERDICT")
print("="*78)
print()

if all_commute:
    print(f" ✓ SU(3)_c (canonical edge-coloring lift) is a symmetry of B.")
    if kernel_dim == 2:
        # Check if kernel = V_scalar
        # Compute Frobenius distance between kernel-projection and V_scalar-projection
        P_ker = kernel_basis @ kernel_basis.T
        diff = P_ker - P_scalar
        print(f"   SU(3)_c-singlet dim = 2, matching V_scalar dim.")
        if np.linalg.norm(diff) < 1e-6:
            print(f"   ✓ ker(SU(3)_c |_{{V_pm}}) = V_scalar EXACTLY.")
            print(f"   ⇒ c_color = V_cycle / (2|E|) = 3/12 = 1/4 closes structurally.")
        else:
            print(f"   ‖P_ker − P_V_scalar‖_F = {np.linalg.norm(diff):.4f}")
            print(f"   The kernel has dim 2 but is NOT exactly V_scalar.")
    elif kernel_dim == 1:
        print(f"   SU(3)_c-singlet dim = 1.")
        print(f"   ⇒ Only 1 SU(3)_c-singlet mode in V_pm (likely the Perron-adj mode).")
        print(f"   ⇒ c_color = (V_pm - 1)/12 = 4/12 = 1/3 (same as current uniform value).")
    elif kernel_dim == 0:
        print(f"   SU(3)_c-singlet dim = 0 in V_pm.")
        print(f"   ⇒ All V_pm modes are SU(3)_c-charged. c_color = 5/12 = c_v_Higgs.")
        print(f"   ⇒ Doesn't match the empirical c_color = 1/4.")
    elif kernel_dim == 5:
        print(f"   SU(3)_c acts trivially on V_pm. ⇒ c_color = 0 (no V_pm coupling).")
        print(f"   ⇒ Doesn't match the empirical c_color = 1/4.")
    else:
        print(f"   SU(3)_c-singlet dim = {kernel_dim} (unexpected).")
else:
    print(f" ✗ SU(3)_c (canonical edge-coloring lift) does NOT commute with B.")
    print(f"   The natural color-rotation-at-source-vertex is NOT a symmetry of substrate")
    print(f"   dynamics. Either:")
    print(f"     - The SU(3)_c lift needs to include B-conjugate-acting components,")
    print(f"     - Or the gauge connection structure is more intricate (link variables),")
    print(f"     - Or the substrate SU(3)_c is defined differently (e.g., via")
    print(f"       Cl(6) Fock at each vertex with a specific edge labeling).")
    print()
    print(f"   Next step: re-derive SU(3)_c from Cl(6) Fock per theorem_charge_before_color.md")
    print(f"   §9 and check if the resulting action commutes with B.")
print()
print("="*78)
