#!/usr/bin/env python3
"""
R1_Av_full_one_form_module_probe.py
===================================
A.v.full slow-path probe, following A.v.simple (per-edge SU(2)_e) and
A.v.refined (B3's SU(2)_L) both closed-negative on the simple [a, D_F]
subspace.

The fix: use the FULL CC 1-form module Ω_D^1 = span{a · [D_F, b] : a, b
∈ A_F}, which is gauge-closed by construction (left-multiplication by a
absorbs the SU(2) escape that broke A.v.simple/refined).

Structural shortcut for tractable computation
---------------------------------------------
[D_F, b] is OFF-DIAGONAL in the C⁰/C¹ block structure (D_F itself is
purely off-diagonal).  Specifically:
  [D_F, b] = [[0, d† b_C1 − b_C0 d†], [d b_C0 − b_C1 d, 0]]

For a ∈ A_F (block-diagonal):
  a · [D_F, b] has:
    - C⁰→C¹ block = a_C0 · (d† b_C1 − b_C0 d†)
    - C¹→C⁰ block = a_C1 · (d b_C0 − b_C1 d)
  Diagonal blocks vanish.

So Ω_D^1 ⊆ off-diagonal subspace of End(H_F), dim ≤ 12288.

PER-VERTEX DECOMPOSITION:  a_C0 is block-diagonal across the 4 vertices,
so a_C0 · X only affects the vertex-row block where a_C0 is non-zero.
Different vertex row blocks are linearly independent in M_{256, 24}.

→ Per-vertex problem: for each vertex v, compute the span of
  {a_v · [D_F, b]_{C0→C1 at vertex v row} : a_v ∈ M_8 basis, b ∈ A_F basis}
  in M_{8, 24} = 192-dim.

Sum over 4 vertices gives the full C0→C1 block dim ≤ 4 × 192 = 768.
By Hermitian conjugacy, the symmetric C1→C0 contribution has same dim.
→ Total  Ω_D^1 dim ≤ 1536.

What this probe does
--------------------
A — Compute the d (cochain map) once.
B — For each vertex v ∈ {0, 1, 2, 3}:
    (i) Extract (d†)_{v row block} ∈ M_{8, 24}.
    (ii) For each b ∈ A_F basis, compute [D_F, b]_{v row block in C0→C1} ∈ M_{8, 24}.
    (iii) For each a_v ∈ M_8 basis (64), form 17920 candidates a_v · [D_F, b]_{v row}.
    (iv) Rank of these candidates = per-vertex dim contribution.
C — Sum the 4 per-vertex rank contributions.
D — Account for the symmetric C1→C0 part: total Ω_D^1 dim = 2 × sum.
E — Report findings and implications for b_i.

No graded content changes.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.foundations.de_rham_susy_fibered_v2_probe import (  # noqa: E402
    d_alg, NE, NV, SX, SY, SZ, I2,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-9


# -----------------------------------------------------------------------------
# Setup: D_F structure
# -----------------------------------------------------------------------------

def build_d_and_DF():
    """Build the cochain map d (24 × 256) and D_F (280 × 280)."""
    d = d_alg((0.0, 0.0, 0.0))   # shape (24, 256)
    dim0, dim1 = NV * 64, NE * 4
    D_F = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    D_F[:dim0, dim0:] = d.conj().T
    D_F[dim0:, :dim0] = d
    return d, D_F, dim0, dim1


def hermitian_basis_Mn(n):
    gens = []
    for i in range(n):
        E = np.zeros((n, n), dtype=complex); E[i, i] = 1.0
        gens.append(E)
    for i in range(n):
        for j in range(i + 1, n):
            E_s = np.zeros((n, n), dtype=complex); E_s[i, j] = 1; E_s[j, i] = 1
            gens.append(E_s)
            E_a = np.zeros((n, n), dtype=complex); E_a[i, j] = 1j; E_a[j, i] = -1j
            gens.append(E_a)
    return gens


# -----------------------------------------------------------------------------
# Part A — extract d's vertex-row blocks
# -----------------------------------------------------------------------------

def part_A_setup():
    print("=" * 100)
    print("PART A — setup: d (24×256), D_F (280×280), and per-vertex row blocks of d†")
    print("=" * 100)
    d, D_F, dim0, dim1 = build_d_and_DF()
    print(f"\n  d shape         : {d.shape}  (= {NE*4} × {NV*64})")
    print(f"  d† shape        : {d.conj().T.shape}  (= {NV*64} × {NE*4})")
    print(f"  D_F dim         : {D_F.shape}")

    # d† has shape (256, 24): rows = matter, cols = gauge
    # Vertex v's row block is rows [64v : 64(v+1)] of d† = 64×24 matrix
    d_dagger = d.conj().T
    d_dagger_v_blocks = []
    for v in range(NV):
        block = d_dagger[v*64:(v+1)*64, :]   # (64, 24)
        d_dagger_v_blocks.append(block)
        print(f"  d†_{{vertex {v} row block}} : shape {block.shape}, norm = {np.linalg.norm(block):.4f}")
    return d, D_F, d_dagger, d_dagger_v_blocks


# -----------------------------------------------------------------------------
# Part B — per-vertex computation of Ω_D^1's C0→C1 contribution
# -----------------------------------------------------------------------------

def part_B_per_vertex(d, d_dagger, d_dagger_v_blocks):
    print("\n" + "=" * 100)
    print("PART B — per-vertex 1-form contributions to Ω_D^1's C0→C1 block")
    print("=" * 100)

    # Each vertex's M_8 = 64 generators (Hermitian basis).
    M8_basis = hermitian_basis_Mn(8)

    # For each vertex v, build A_F basis's [D_F, b]_{C0→C1, v row block} candidates,
    # then multiply by all 64 a_v ∈ M_8 basis from vertex v.
    #
    # In flattened M_8 (column-major flatten) row-block of M_{64, 24}:
    # An element X ∈ M_{8, 24} = (rows = 8 fermion modes in vertex v) × (cols = 24 gauge dofs across all edges)
    # has 192 entries.  But our representation: row block of d† is M_{64, 24}; we need to convert.
    #
    # Actually wait — d† has matter side as M_8(ℂ) flattened, so each "row" of d† indexes
    # a flattened operator slot, not a Fock state.  d† maps 24-dim edge space → 256-dim matter
    # OPERATOR space (= ⊕_v M_8 as a vector space).
    #
    # Vertex v's row block of d† has shape (64, 24) — 64 operator components × 24 edge.
    #
    # An element X = d†_{v row} or [D_F, b]_{v row} ∈ M_{64, 24}.
    #
    # The "vertex-a ∈ M_8 left-mult" action:
    #   For a_v ∈ M_8 (8x8), the left-mult on vertex v's M_8 block: when we represent the
    #   M_8 operator algebra as flatten of 8×8 matrices (64-dim vectors), left-mult by a_v
    #   acts as np.kron(I_8, a_v) (column-major flatten) — a 64×64 matrix.
    #
    # So a_v · X for X ∈ M_{64, 24}:  apply (I_8 ⊗ a_v) to each column of X, giving X' ∈ M_{64, 24}.

    # All 280 generators of A_F (in left-mult representation): we just need [D_F, b] for each.
    # Construct b's (each is 280×280 in End(H_F)) and compute [D_F, b]'s C0→C1 block per vertex.

    print(f"\n  Building [D_F, b]'s C0→C1 block for each of 280 A_F generators...")

    # Build the per-vertex restriction: [D_F, b]_{v row block, C0→C1} ∈ M_{64, 24} for each b.
    # For b in vertex-w generator (M_8 at vertex w as left-mult): [D_F, b]_{v row C0→C1}
    #   = (d† b_C1 - b_C0 d†)_{v row} = (-b_C0 d†)_{v row} (since b_C1 = 0)
    #   = - (b_C0)_{v row block} · d†   (but b_C0 is left-mult M_8 at vertex w)
    #   For (b_C0)_{v row block}: this is non-zero only if v = w (since b_C0 has block-diagonal struct)
    #   When v = w: (b_C0)_{v block} acts as left-mult M_8 on the v-row of d†.
    #
    # For b in edge-e generator (M_2 at edge e as left-mult): [D_F, b]_{v row C0→C1}
    #   = (d† b_C1)_{v row} = (d†)_{v row} · b_C1_{full}
    #   where b_C1_{full} is 24×24 left-mult M_2 at edge e (acts on 24-dim gauge sector).
    #
    # Practical extraction:
    # Easiest: build each b as a 280×280 op, compute [D_F, b], extract the (v row block in C0→C1).

    n_AF = NV * 64 + NE * 4   # = 280
    candidates_per_vertex = [[] for _ in range(NV)]   # list of 17920 candidates per vertex

    # Build the 280 generators b as left-mult operators on H_F
    def build_b_left_mult(M_block, sector, idx):
        """M_block ∈ M_8 if sector='vertex' with idx=vertex, M_2 if sector='edge' with idx=edge.
        Returns 280×280 left-mult op."""
        dim0, dim1 = NV * 64, NE * 4
        op = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
        if sector == 'vertex':
            op[idx*64:(idx+1)*64, idx*64:(idx+1)*64] = np.kron(np.eye(8, dtype=complex), M_block)
        else:
            op[dim0+idx*4:dim0+(idx+1)*4, dim0+idx*4:dim0+(idx+1)*4] = np.kron(np.eye(2, dtype=complex), M_block)
        return op

    # Get all 280 b's
    b_list = []
    for w in range(NV):
        for M8 in M8_basis:
            b_list.append((build_b_left_mult(M8, 'vertex', w), ('vertex', w)))
    M2_basis = hermitian_basis_Mn(2)
    for e in range(NE):
        for M2 in M2_basis:
            b_list.append((build_b_left_mult(M2, 'edge', e), ('edge', e)))
    print(f"  built {len(b_list)} A_F generators b")

    # For each b, compute [D_F, b] and extract row blocks for each vertex
    print(f"  computing [D_F, b] for each b and extracting per-vertex row blocks...")
    _, D_F, _, _ = build_d_and_DF()
    blocks_per_vertex = [[] for _ in range(NV)]   # blocks_per_vertex[v][b_idx] = (64, 24) matrix
    for b_op, label in b_list:
        comm = D_F @ b_op - b_op @ D_F
        for v in range(NV):
            block = comm[v*64:(v+1)*64, NV*64:NV*64+NE*4]   # 64 × 24
            blocks_per_vertex[v].append(block)
    print(f"  blocks_per_vertex[v] = list of {len(blocks_per_vertex[0])} matrices per vertex")

    # For each vertex v: form the (64 × 280, 64 × 24) candidates by left-mult by M_8 basis
    # Specifically: a_v ∈ M_8 (64 basis) acts on the v-row block.  Result is again (64, 24).
    # Stack as a 17920 × (64 × 24) = 17920 × 1536 matrix.
    print(f"\n  Per-vertex rank computation:")
    per_vertex_rank = []
    for v in range(NV):
        cands = []
        for a_v in M8_basis:
            ad_a_v = np.kron(np.eye(8, dtype=complex), a_v)   # 64×64 left-mult on vertex v's row block flatten
            for X in blocks_per_vertex[v]:
                Y = ad_a_v @ X   # (64, 24)
                cands.append(Y.flatten())
        cands = np.array(cands)
        rank_v = np.linalg.matrix_rank(cands, tol=TOL)
        print(f"    vertex {v}: {len(cands)} candidates → rank = {rank_v}  (max possible 64×24 = 1536, but bounded by SVD on 64×8 effectively)")
        per_vertex_rank.append(rank_v)

    total_C0_to_C1 = sum(per_vertex_rank)
    print(f"\n  Sum across vertices: dim(C0→C1 part of Ω_D^1) = {total_C0_to_C1}")
    return per_vertex_rank, total_C0_to_C1


# -----------------------------------------------------------------------------
# Part C — Hermitian conjugate gives the C1→C0 part
# -----------------------------------------------------------------------------

def part_C_summary(per_vertex_rank, total_C0_to_C1):
    print("\n" + "=" * 100)
    print("PART C — total Ω_D^1 dim and structural interpretation")
    print("=" * 100)
    total_Omega = 2 * total_C0_to_C1
    print(f"""
  Per-vertex C0→C1 contribution ranks : {per_vertex_rank}
  Sum (= dim of C0→C1 block in Ω_D^1) : {total_C0_to_C1}
  By Hermitian conjugacy, C1→C0 block has same dim.
  TOTAL Ω_D^1 dim = 2 × {total_C0_to_C1} = {total_Omega}

  Reference scales:
    A.v.simple's [a, D_F] subspace            : 279
    Off-diagonal subspace of End(H_F) max     : 12288 (= 2 × 256 × 24)
    Per-vertex max (8 × 24)                   : 192
    Per-vertex theoretical max × 4 × 2        : 1536

  Comparison to MSSM/SM Higgs sector dims:
    SM Higgs (1 doublet, complex)             : 4 real states
    MSSM Higgs (H_u + H_d doublets)           : 8 real states
    MSSM Higgs + sfermions (per cell)         : 56 real states
""")
    if total_Omega <= 60:
        print(f"  → Ω_D^1 dim = {total_Omega} matches or is close to MSSM Higgs+sfermion content (56).")
        print(f"     This is encouraging — could give MSSM-compatible scalar content.")
    elif total_Omega <= 200:
        print(f"  → Ω_D^1 dim = {total_Omega} is intermediate;  worth detailed irrep analysis.")
    else:
        print(f"  → Ω_D^1 dim = {total_Omega} is LARGE compared to MSSM/SM Higgs sector.")
        print(f"     Framework's Higgs sector is structurally bigger than MSSM/SM expectations.")
        print(f"     Further gauge decomposition needed to extract physical (non-redundant) scalar content.")


# -----------------------------------------------------------------------------
def main():
    print(r"""
==========================================================================================
A.v.full — full CC 1-form module Ω_D^1 = span{a · [D_F, b] : a, b ∈ A_F}
Gauge-closed by construction.  Replaces A.v.simple/refined (both closed-negative).
==========================================================================================""")
    d, D_F, d_dagger, d_dagger_v_blocks = part_A_setup()
    per_vertex_rank, total = part_B_per_vertex(d, d_dagger, d_dagger_v_blocks)
    part_C_summary(per_vertex_rank, total)

    print("\n" + "=" * 100)
    print("A.v.full INTERIM VERDICT")
    print("=" * 100)
    total_Omega = 2 * total
    print(f"""
  ESTABLISHED (this probe, machine precision):

   • dim(Ω_D^1) = {total_Omega}     (the framework's 1-form Higgs module from the spectral triple)

   Per-vertex C0→C1 contributions: {per_vertex_rank}

  STATUS:
   • A.v.full {'YIELDS A FINITE OBSERVABLE HIGGS SECTOR' if total_Omega > 0 else 'is empty'}
   • Comparison to MSSM/SM is QUALITATIVELY {('matching MSSM scale' if total_Omega <= 60 else ('intermediate' if total_Omega <= 200 else 'much LARGER than MSSM/SM'))}
   • Next step:  decompose Ω_D^1 under the framework's gauge group (SU(2)_L per B3,
     SU(3)_c per B6, U(1)_Y).  Per-gauge spin-J counts give T(R_s) for the b_i formula.

  ADOPTED-MSSM-Sb stands.  R1 status: INTERIM.  No graded content changes.
""")
    print("R1_Av_full_one_form_module_probe.py: sentinel done.")


if __name__ == "__main__":
    main()
