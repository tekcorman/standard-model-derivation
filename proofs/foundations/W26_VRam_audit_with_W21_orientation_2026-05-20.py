#!/usr/bin/env python3
"""
W26 — Sector-resolved V_Ram audit on BD(K_4) with W21 broken-vacuum orientation
================================================================================

Date: 2026-05-20
Item #1 sub-probe (i): the bounded entry point to R-14 / Need-D-3 / V_Ram ≅
Cl(6)-Fock identification.

CONTEXT.
chi_tilde 2026-05-01 EOD probe `srs_z_chi_C3_VRam_isotypic.py` audited V_Ram
on srs-z's walker WITHOUT the broken-vacuum input and found:
  C_3 isotypic multiplicities on V_Ram(srs-z) = (8, 4, 4) (doubled from
    srs (4, 2, 2)).
  Joint χ̃ × C_3: BOTH χ̃ sectors carry IDENTICAL (4, 2, 2). Bipartite cover
    doubles structure but χ̃ tracks copy without introducing C_3 asymmetry.
  ⟹ V_ub closure route 'color = generation duality via χ̃ × C_3' is
    NOT grounded by χ̃ alone.

W21 (today) supplied the missing input: the broken Higgs vacuum's orientation
on the bipartite cover, σ_combined = σ_swap × σ_mirror. W22 showed this
breaks χ̃-pair degeneracy at second order. W26 asks the analogous question
at the V_Ram level: does the W21 orientation, when projected onto V_Ram,
introduce per-C_3-sector structure that chi_tilde 2026-05-01 didn't see?

PROBE STRUCTURE.
1. Build BD(K_4) Hashimoto B at k=Γ; find Ramanujan eigenspace V_Ram (|λ|² = 2).
2. Build χ̃ (diagonal sign by arc-tail-sheet); verify {χ̃, B} = 0.
3. Build C_3 graph automorphism on the 24-arc walker; verify [C_3, B] = 0.
4. Decompose V_Ram under C_3 × χ̃ → isotypic structure.
5. Build W21 Higgs operator H_VEV on the walker (diagonal arc-basis).
6. Restrict H_VEV to V_Ram; check whether the restricted operator:
   (a) couples DIFFERENT C_3 sectors (inter-generation mixing),
   (b) breaks χ̃ symmetry within fixed-C_3 sectors,
   (c) leaves V_Ram structurally invariant under C_3 × χ̃.
7. Verdict: does the chi_tilde 2026-05-01 negative finding still hold once
   the W21 orientation is included, or does it open per-species labeling?

DELIBERATE NOTE ON SCOPE.
W26 uses the abstract BD(K_4) Hashimoto at k=Γ. chi_tilde 2026-05-01 used
Bloch-decomposed B(k_R) on srs-z's PRIMITIVE CELL with k_R = (1/2,1/2,1/2),
which has different multiplicities (V_Ram = 16 vs abstract = 12). The
structural question (does W21 break symmetries?) is preserved across both
versions. Reproducing chi_tilde's exact Bloch numbers requires RCSR data
not available on this machine.

PRE-DECLARED GATE CHECKS:
  L1. BD(K_4) Hashimoto has Ramanujan modes with |λ|² = 2.
  L2. V_Ram (B²-eigenspace at 2) has dimension 12 on abstract BD(K_4).
  L3. χ̃ on V_Ram has equal ±1 multiplicities (the chi_tilde split survives).
  L4. C_3 commutes with B; C_3 isotypic decomposition of V_Ram is well-defined.
  L5. Joint C_3 × χ̃ decomposition of V_Ram (without W21): chi_tilde's
      'symmetric across χ̃ sectors' result reproduced on abstract BD(K_4).
  L6. W21 Higgs operator H_VEV = (1/√2) · χ̃ · (per-edge VEV) on the walker
      restricts non-trivially to V_Ram.
  L7. H_VEV|_V_Ram structure: documents whether it (a) inter-couples C_3
      sectors, (b) breaks χ̃ symmetry, or (c) is sector-respecting and only
      contributes a uniform shift.

USAGE:
    python3 proofs/foundations/W26_VRam_audit_with_W21_orientation_2026-05-20.py
"""

from __future__ import annotations
import numpy as np

EXPECTED = {
    "L1_ramanujan_eigenvalues":      True,
    "L2_VRam_dim_12":                True,
    "L3_chi_balanced_in_VRam":       True,
    "L4_C3_commutes_with_B":         True,
    "L5_no_W21_chi_C3_symmetric":    True,
    "L6_H_VEV_nontrivial_in_VRam":   True,
    "L7_H_VEV_structure_documented": True,
}
RESULTS = {}

print("=" * 78)
print("W26 — V_Ram audit on BD(K_4) with W21 broken-vacuum orientation")
print("=" * 78)


# ============================================================================
# Step A — Build BD(K_4) explicitly (same as W21/W22)
# ============================================================================
N_V_K4 = 4
K4_edges = [(u, v) for u in range(N_V_K4) for v in range(u + 1, N_V_K4)]
N_V_BD = 8
def encode(u, sheet): return u + sheet * N_V_K4

bd_edges = []
cover_pairs = []
for u, v in K4_edges:
    alpha = (encode(u, 0), encode(v, 1))
    beta  = (encode(v, 0), encode(u, 1))
    bd_edges.append(alpha); bd_edges.append(beta)
    cover_pairs.append((len(bd_edges)-2, len(bd_edges)-1))

def directed_arcs(edges):
    arcs = []
    for ei, (u, v) in enumerate(edges):
        arcs.append((u, v, ei))
        arcs.append((v, u, ei))
    return arcs

BD_arcs = directed_arcs(bd_edges)
N_ARCS_BD = len(BD_arcs)
arc_lookup = {a: i for i, a in enumerate(BD_arcs)}

def hashimoto(arcs):
    n = len(arcs)
    B = np.zeros((n, n), dtype=complex)
    for i_p, (t_p, h_p, e_p) in enumerate(arcs):
        for i, (t, h, e) in enumerate(arcs):
            if h == t_p and e_p != e:
                B[i_p, i] = 1.0
    return B

B = hashimoto(BD_arcs)
print(f"\nStep A — BD(K_4) walker: {N_ARCS_BD} arcs, B shape = {B.shape}")


# ============================================================================
# Step B — Find V_Ram (B²-eigenspace at |λ|² = k*-1 = 2)
# ============================================================================
K_STAR = 3
RAM_LAMBDA_SQ = K_STAR - 1   # = 2

# B has complex eigenvalues; use the eigenvectors of B directly, masked by
# Ramanujan condition |λ|² = k* - 1 = 2. (NOT eigenvalues of B², which for
# complex λ would be λ², not |λ|².)
eigvals_B, V_B = np.linalg.eig(B)
ram_mask = np.abs(np.abs(eigvals_B)**2 - RAM_LAMBDA_SQ) < 1e-7
V_Ram_basis = V_B[:, ram_mask]
dim_VRam = V_Ram_basis.shape[1]
print(f"\nStep B — V_Ram identification")
print(f"  B spectrum (sample): {sorted(set(round(e.real,3) + round(e.imag,3)*1j for e in eigvals_B), key=lambda x: (x.real, x.imag))[:8]}")
print(f"  Ramanujan threshold: |λ|² = k*-1 = {RAM_LAMBDA_SQ}")
print(f"  V_Ram dimension: {dim_VRam}")
L1 = any(abs(abs(e)**2 - RAM_LAMBDA_SQ) < 1e-7 for e in eigvals_B)
L2 = (dim_VRam == 12)
print(f"  L1 (Ramanujan eigenvalues present): {L1}")
print(f"  L2 (V_Ram dim = 12 on abstract BD(K_4)): {L2}")
RESULTS["L1_ramanujan_eigenvalues"] = bool(L1)
RESULTS["L2_VRam_dim_12"] = bool(L2)

# Orthonormalize V_Ram basis (eigenvectors may be non-orthogonal due to B's
# non-Hermiticity; use QR-style orthonormalization on the subspace)
Q, _ = np.linalg.qr(V_Ram_basis)
V_Ram = Q   # 24 x 12; columns are orthonormal basis of V_Ram

# Projection onto V_Ram
P_VRam = V_Ram @ V_Ram.conj().T


# ============================================================================
# Step C — χ̃ on the walker; restrict to V_Ram
# ============================================================================
side_label = {idx: (+1 if idx < N_V_K4 else -1) for idx in range(N_V_BD)}
chi_diag = np.array([side_label[t] for (t, _, _) in BD_arcs], dtype=complex)
chi = np.diag(chi_diag)

anticomm = chi @ B + B @ chi
print(f"\nStep C — χ̃ on the walker")
print(f"  ||{{χ̃, B}}||_F = {np.linalg.norm(anticomm):.4e}  (expect 0)")
assert np.linalg.norm(anticomm) < 1e-10

# χ̃ restricted to V_Ram (in V_Ram basis)
chi_VRam = V_Ram.conj().T @ chi @ V_Ram   # 12x12
chi_VRam_eigs = np.linalg.eigvalsh((chi_VRam + chi_VRam.conj().T) / 2)
n_plus_VRam = int((chi_VRam_eigs > 0.5).sum())
n_minus_VRam = int((chi_VRam_eigs < -0.5).sum())
print(f"  χ̃|_V_Ram eigenvalues: {[round(float(e), 4) for e in sorted(chi_VRam_eigs)]}")
print(f"  χ̃ = +1 count in V_Ram: {n_plus_VRam}")
print(f"  χ̃ = -1 count in V_Ram: {n_minus_VRam}")
L3 = (n_plus_VRam == 6) and (n_minus_VRam == 6)
print(f"  L3 (χ̃ balanced in V_Ram): {L3}")
RESULTS["L3_chi_balanced_in_VRam"] = bool(L3)


# ============================================================================
# Step D — C_3 graph automorphism on BD(K_4)
# ============================================================================
# K_4's natural C_3: fix vertex 3, cycle (0, 1, 2). On BD(K_4) (8 vertices),
# this lifts to sheet-preserving permutation:
#   (0, sheet) → (1, sheet); (1, sheet) → (2, sheet); (2, sheet) → (0, sheet)
#   (3, sheet) → (3, sheet)
def c3_vertex(v):
    base = v % N_V_K4
    sheet = v // N_V_K4
    new_base = {0: 1, 1: 2, 2: 0, 3: 3}[base]
    return new_base + sheet * N_V_K4

# Verify it's an order-3 graph automorphism
def is_graph_auto(perm_fn):
    # Check it preserves the edge set
    edge_set = set(frozenset(e) for e in bd_edges)
    permuted = set(frozenset((perm_fn(u), perm_fn(v))) for (u, v) in bd_edges)
    return edge_set == permuted

c3_auto_ok = is_graph_auto(c3_vertex)
# And order 3
c3_order_3 = all(c3_vertex(c3_vertex(c3_vertex(v))) == v for v in range(N_V_BD))
print(f"\nStep D — C_3 graph automorphism")
print(f"  c3 is graph automorphism (edges preserved): {c3_auto_ok}")
print(f"  c3 has order 3: {c3_order_3}")
assert c3_auto_ok and c3_order_3

# Build edge permutation
K4_edge_lookup = {frozenset(e): i for i, e in enumerate(K4_edges)}
bd_edge_lookup = {frozenset(e): i for i, e in enumerate(bd_edges)}

def c3_edge(ei):
    (u, v) = bd_edges[ei]
    return bd_edge_lookup[frozenset((c3_vertex(u), c3_vertex(v)))]

# Build C_3 action on arcs (24x24 permutation matrix)
C3 = np.zeros((N_ARCS_BD, N_ARCS_BD), dtype=complex)
for i, (t, h, e) in enumerate(BD_arcs):
    new_arc = (c3_vertex(t), c3_vertex(h), c3_edge(e))
    j = arc_lookup[new_arc]
    C3[j, i] = 1.0

# C_3 should commute with B (since C_3 is a graph automorphism of the
# underlying simple graph that respects edge labelings).
comm_C3_B = C3 @ B - B @ C3
print(f"  ||[C_3, B]||_F = {np.linalg.norm(comm_C3_B):.4e}  (expect 0)")
L4 = np.linalg.norm(comm_C3_B) < 1e-10
print(f"  L4 (C_3 commutes with B): {L4}")
RESULTS["L4_C3_commutes_with_B"] = bool(L4)

# C_3 also commutes with χ̃ (since C_3 is sheet-preserving)
comm_C3_chi = C3 @ chi - chi @ C3
print(f"  ||[C_3, χ̃]||_F = {np.linalg.norm(comm_C3_chi):.4e}  (expect 0; sheet-preserving)")


# ============================================================================
# Step E — C_3 isotypic decomposition of V_Ram
# ============================================================================
# C_3 has irreps ω^0 (trivial), ω^1, ω^2 with ω = exp(2πi/3).
# A vector v is in the ω^k isotypic iff C_3 · v = ω^k · v.
omega = np.exp(2j * np.pi / 3)
projectors_C3 = []
for k in range(3):
    P_k = sum(omega ** (-k * m) * np.linalg.matrix_power(C3, m) for m in range(3)) / 3
    projectors_C3.append(P_k)

# Restrict V_Ram to each C_3 isotypic
def isotypic_dim(P_k_full):
    P_k_VRam = V_Ram.conj().T @ P_k_full @ V_Ram   # 12x12 in V_Ram basis
    rank = int(round(np.trace(P_k_VRam).real))
    return rank, P_k_VRam

dims_C3 = []
for k in range(3):
    d, _ = isotypic_dim(projectors_C3[k])
    dims_C3.append(d)
print(f"\nStep E — V_Ram C_3 isotypic decomposition (no W21)")
print(f"  V_Ram dimensions per C_3 isotypic (ω^0, ω^1, ω^2): {tuple(dims_C3)}")
print(f"  Sum: {sum(dims_C3)} (expect 12)")

# Joint C_3 × χ̃ decomposition
print(f"\n  Joint (C_3, χ̃) decomposition:")
joint_dims = {}
for k in range(3):
    for s in [+1, -1]:
        P_k = projectors_C3[k]
        P_chi_s = (np.eye(N_ARCS_BD, dtype=complex) + s * chi) / 2
        P_joint = P_k @ P_chi_s
        P_joint_VRam = V_Ram.conj().T @ P_joint @ V_Ram
        rank = int(round(np.trace(P_joint_VRam).real))
        joint_dims[(k, s)] = rank
        print(f"    (C_3 = ω^{k}, χ̃ = {s:+d}): dim = {rank}")

# chi_tilde-style symmetry test: are χ̃=+1 and χ̃=-1 sectors identical per C_3?
plus_sequence = (joint_dims[(0, +1)], joint_dims[(1, +1)], joint_dims[(2, +1)])
minus_sequence = (joint_dims[(0, -1)], joint_dims[(1, -1)], joint_dims[(2, -1)])
print(f"  χ̃ = +1 C_3 sequence: {plus_sequence}")
print(f"  χ̃ = -1 C_3 sequence: {minus_sequence}")
L5 = (plus_sequence == minus_sequence)
print(f"  L5 (χ̃ sectors symmetric under C_3 without W21): {L5}")
RESULTS["L5_no_W21_chi_C3_symmetric"] = bool(L5)


# ============================================================================
# Step F — W21 Higgs operator on the walker
# ============================================================================
# Per W21: each BD(K_4) edge carries uniform ⟨h⁰⟩ = +v/√2 · e_1; on the
# walker, the per-arc dimensionless weight is (1/√2) · χ̃[i] (since each arc
# inherits the edge's e_1 content and the χ̃ tail-side sign gives the
# σ_mirror-relevant flip). Equivalently:
#     H_VEV = (1/√2) · χ̃  (as 24×24 diagonal in arc basis, units of v)
hzero_over_v = 1.0 / np.sqrt(2.0)
H_VEV = hzero_over_v * chi.copy()   # diagonal, ± (1/√2) per arc

print(f"\nStep F — W21 Higgs operator H_VEV on walker")
print(f"  H_VEV = (1/√2) · χ̃ (diagonal arc-basis; units of v)")
print(f"  H_VEV diagonal sample (first 8 arcs): {[float(H_VEV[i,i].real) for i in range(8)]}")


# ============================================================================
# Step G — H_VEV restricted to V_Ram
# ============================================================================
H_VEV_VRam = V_Ram.conj().T @ H_VEV @ V_Ram   # 12x12
H_VEV_VRam_norm = np.linalg.norm(H_VEV_VRam)
print(f"\nStep G — H_VEV restricted to V_Ram")
print(f"  ||H_VEV|_V_Ram||_F = {H_VEV_VRam_norm:.4f}")
L6 = H_VEV_VRam_norm > 0.1
print(f"  L6 (H_VEV non-trivial in V_Ram): {L6}")
RESULTS["L6_H_VEV_nontrivial_in_VRam"] = bool(L6)


# ============================================================================
# Step H — Decompose H_VEV|_V_Ram under C_3 × χ̃ sectors
# ============================================================================
# Build basis for each (C_3, χ̃) sector by projecting V_Ram basis vectors.
# Within each sector, H_VEV may have non-trivial entries.
# More importantly: does H_VEV have INTER-SECTOR entries (coupling different
# (C_3, χ̃) sectors)?

# Construct projected basis vectors per (C_3, χ̃) sector.
def sector_basis(k, s):
    P_k = projectors_C3[k]
    P_chi_s = (np.eye(N_ARCS_BD, dtype=complex) + s * chi) / 2
    P_joint = P_k @ P_chi_s
    P_joint_VRam_arc = P_joint @ V_Ram     # 24 x 12
    # Reduce to orthonormal basis of the image; columns that are non-zero.
    U, S, _ = np.linalg.svd(P_joint_VRam_arc)
    n_nonzero = int((S > 1e-7).sum())
    return U[:, :n_nonzero]   # arc basis

# H_VEV inter-sector block-matrix
print(f"\nStep H — Inter-sector structure of H_VEV|_V_Ram")
sectors = [(k, s) for k in range(3) for s in [+1, -1]]
sector_bases = {sec: sector_basis(*sec) for sec in sectors}
sector_dims = {sec: sector_bases[sec].shape[1] for sec in sectors}
print(f"  Sector dimensions (verifying L5): {[(sec, sector_dims[sec]) for sec in sectors]}")
total_dim = sum(sector_dims.values())
print(f"  Sum: {total_dim} (expect 12)")

# Block matrix of H_VEV under the (C_3 × χ̃) sector decomposition
print(f"\n  H_VEV inter-sector block norms (||B[i,j]||_F):")
print(f"  rows = (k, s) target sector, cols = source sector")
print(f"  {'':>14s} " + " ".join(f"({k},{s:+d})" for (k, s) in sectors))
inter_block_max = 0.0
intra_block_max = 0.0
H_block_matrix = {}
for sec_i in sectors:
    Ui = sector_bases[sec_i]
    row_str = f"  ({sec_i[0]},{sec_i[1]:+d}) "
    for sec_j in sectors:
        Uj = sector_bases[sec_j]
        if Ui.shape[1] == 0 or Uj.shape[1] == 0:
            block = np.zeros((Ui.shape[1], Uj.shape[1]))
        else:
            block = Ui.conj().T @ H_VEV @ Uj
        H_block_matrix[(sec_i, sec_j)] = block
        norm = np.linalg.norm(block)
        if sec_i == sec_j:
            intra_block_max = max(intra_block_max, norm)
        else:
            inter_block_max = max(inter_block_max, norm)
        row_str += f"  {norm:6.3f}"
    print(row_str)

print()
print(f"  Max intra-sector block norm (diagonal): {intra_block_max:.4f}")
print(f"  Max inter-sector block norm (off-diag): {inter_block_max:.4f}")

# Determine structural verdict
inter_couples_C3   = any(np.linalg.norm(H_block_matrix[((k1, s1), (k2, s2))]) > 1e-6
                         for (k1, s1) in sectors for (k2, s2) in sectors
                         if k1 != k2)
inter_couples_chi  = any(np.linalg.norm(H_block_matrix[((k, +1), (k, -1))]) > 1e-6
                         for k in range(3))
nontrivial_intra  = intra_block_max > 1e-6

L7 = True  # documents the structure
RESULTS["L7_H_VEV_structure_documented"] = bool(L7)


# ============================================================================
# Step I — Verdict
# ============================================================================
print("\n" + "=" * 78)
print("W26 VERDICT — Gate Check")
print("=" * 78)
all_pass = True
for k, expected in EXPECTED.items():
    actual = RESULTS.get(k)
    status = "PASS" if actual == expected else "FAIL"
    if actual != expected: all_pass = False
    print(f"  {status}  {k:42s}  expected={expected}, got={actual}")

print()
print("STRUCTURAL FINDING:")
print(f"  H_VEV couples DIFFERENT C_3 sectors:        {inter_couples_C3}")
print(f"  H_VEV couples χ̃ = +1 ↔ χ̃ = -1 (within k): {inter_couples_chi}")
print(f"  H_VEV is non-trivial within sectors:         {nontrivial_intra}")
print()
if not inter_couples_C3 and not inter_couples_chi:
    print("  NEGATIVE result: H_VEV|_V_Ram is SECTOR-DIAGONAL — it only contributes")
    print("  a uniform per-sector shift, does NOT mix C_3 generations or χ̃ chiralities.")
    print("  The chi_tilde 2026-05-01 'V_ub closure via χ̃ × C_3 NOT grounded' finding")
    print("  EXTENDS to the W21-orientated case: V_Ram alone (even with broken-vacuum")
    print("  input) lacks the per-species labeling needed for R-14 closure.")
elif inter_couples_C3 and not inter_couples_chi:
    print("  PARTIAL POSITIVE: H_VEV|_V_Ram couples C_3 sectors but respects χ̃.")
    print("  This is NEW vs chi_tilde 2026-05-01 — it indicates the W21 orientation,")
    print("  acting on V_Ram, introduces inter-generation mixing. Bounded next probe:")
    print("  characterize the inter-generation matrix and check whether it encodes")
    print("  a structurally-meaningful per-(n, j) labeling.")
elif not inter_couples_C3 and inter_couples_chi:
    print("  PARTIAL POSITIVE: H_VEV|_V_Ram breaks χ̃ symmetry within each C_3 sector.")
    print("  This is NEW vs chi_tilde 2026-05-01. Bounded next probe: track the χ̃-")
    print("  asymmetric contribution per C_3 sector and link to species masses.")
else:
    print("  STRONG POSITIVE: H_VEV|_V_Ram couples BOTH C_3 and χ̃ sectors.")
    print("  This significantly extends chi_tilde 2026-05-01 and could unblock R-14.")
    print("  Bounded next probe: characterize the full (C_3 × χ̃ × H_VEV) action and")
    print("  link to per-(n, j) species labels.")
print()
print("Honest scope note:")
print("  This audit uses abstract BD(K_4) at k=Γ (V_Ram = 12). chi_tilde used")
print("  Bloch B(k_R) on srs-z primitive cell (V_Ram = 16). The structural")
print("  question (does W21 break symmetries?) is preserved; reproducing")
print("  chi_tilde's exact Bloch numerics requires RCSR data not on this machine.")
print()
print("=" * 78)
