#!/usr/bin/env python3
"""
Z_CC_reductions_omega_D1_probe.py
=================================

Refined Z probe of the M-arc: apply the standard CC reductions
(Hermitian, J-twin, order-1) to the framework's raw inner-fluctuation
1-form module Ω_D¹ = 1536, then count surviving scalar DOF.

Motivation.  The parameter ledger pattern (`docs/parameters/target_parameters.md`
+ `parameter_uniqueness_ledger.md`) shows that EVERY MSSM-dependent parameter
in the framework is downstream of the gauge-coupling RG flow from M_unif to
M_Z.  Particle masses (m_e, m_μ, m_τ, m_ν₂, m_ν₃, m_H, m_W), CKM, neutrino
mixing — all theorem-grade-without-MSSM.  Only the gauge couplings at M_Z
need MSSM β-coefficients (33/5, 1, −3) to flow from the framework's
substrate-native boundary (α_GUT⁻¹ = 24, sin²θ_W = 3/8).

That tells us the SUSY partners' role in the framework is GAUGE-RUNNING-MEDIATED,
not mass-mediated.  Their contribution to the β-coefficients is the load-bearing
thing; their existence as literal particles is one possible interpretation but
not the only one.

This probe tests whether the framework's spectral-triple inner-fluctuation
machinery produces the right scalar DOF count + gauge-rep content to reproduce
MSSM Δb = (+5/2, +25/6, +4) as the β contribution between SM and MSSM.

Steps (this probe = Phase 1):
   1. Build Ω_D¹ upper-right block basis (target dim 768, per A.v.full).
   2. Apply Hermitian condition (parametrize Φ by upper-right B alone).
   3. Apply J-real condition (J^α and J^β separately, both KO-dim 0 sign).
   4. Report real-dim of J-real Hermitian Ω_D¹ for both J variants.

Phase 2 (separate probe, after Phase 1):
   5. Check order-1 condition.
   6. Decompose under SU(2)_L × SU(2)_R × U(1)_Y.
   7. Compute Δb_i and compare to MSSM (+5/2, +25/6, +4).

No graded content changes.  This is a structural probe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.foundations.de_rham_susy_fibered_v2_probe import (  # noqa: E402
    d_alg, NV, NE, EDGES, incident_edges,
)
from proofs.foundations.M1_J_real_structure_probe import (  # noqa: E402
    J_alpha_64, J_beta_64, J_alpha_4, J_beta_4,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-9


# ---------------------------------------------------------------------------
# Framework data
# ---------------------------------------------------------------------------

def build_D_F():
    d = d_alg((0.0, 0.0, 0.0))
    dim0, dim1 = NV * 64, NE * 4
    D_F = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    D_F[:dim0, dim0:] = d.conj().T
    D_F[dim0:, :dim0] = d
    return D_F, dim0, dim1


def part_A_build_omega_D1_upper_right():
    """Build basis of upper-right block of Ω_D¹.

    Per A.v.full: per-vertex rank = 192, total upper-right = 4 × 192 = 768.
    We build per-vertex and aggregate.
    """
    print("=" * 100)
    print("PART A — build Ω_D¹ upper-right block basis (target dim 768)")
    print("=" * 100)
    D_F, dim0, dim1 = build_D_F()
    dim_tot = dim0 + dim1
    print(f"  H_F dim = {dim_tot} = {dim0} (matter) + {dim1} (gauge)")
    print(f"  Upper-right block shape = {dim0} × {dim1} = {dim0 * dim1} complex DOF")

    # Per-vertex Ω_D¹ contribution: (a, b) ∈ M_8(v) × M_8(v).
    # Each (a, b) gives G = a · [D_F, b] = a D_F b - a b D_F.
    # Extract upper-right rows of vertex v = rows [64v, 64(v+1)) of upper-right block.
    # That sub-block has shape (64, 24) = 1536 complex per vector.
    #
    # 64 × 64 = 4096 pairs per vertex.  Per-vertex rank = 192 expected.

    # Pre-build all 280 b-generators as left-mult operators on H_F.
    # b ∈ M_8(w):  280×280 with np.kron(I_8, E^{ij}) on row/col block w.
    # b ∈ M_2(e):  280×280 with np.kron(I_2, E^{ij}) on row/col block e.
    print(f"\n  Pre-building 280 A_F left-mult b-generators...")
    b_list = []  # each entry is a 280×280 complex op
    for w in range(NV):
        for i in range(8):
            for j in range(8):
                E = np.zeros((8, 8), dtype=complex); E[i, j] = 1
                op = np.zeros((dim_tot, dim_tot), dtype=complex)
                op[w*64:(w+1)*64, w*64:(w+1)*64] = np.kron(np.eye(8, dtype=complex), E)
                b_list.append(op)
    for e in range(NE):
        for i in range(2):
            for j in range(2):
                E = np.zeros((2, 2), dtype=complex); E[i, j] = 1
                op = np.zeros((dim_tot, dim_tot), dtype=complex)
                op[dim0+e*4:dim0+(e+1)*4, dim0+e*4:dim0+(e+1)*4] = np.kron(np.eye(2, dtype=complex), E)
                b_list.append(op)
    n_b = len(b_list)
    print(f"  built {n_b} = 4×64 + 6×4 b-generators")

    # Pre-compute [D_F, b] for each b, then extract upper-right block (256×24).
    print(f"  computing [D_F, b] commutators and extracting upper-right blocks...")
    comm_blocks = []  # each is (256, 24) = the upper-right block of [D_F, b]
    for b in b_list:
        c = D_F @ b - b @ D_F
        comm_blocks.append(c[:dim0, dim0:dim0+dim1])
    print(f"  computed {len(comm_blocks)} commutators")

    bases_per_vertex: list[np.ndarray] = []
    for v in range(NV):
        # For each a ∈ M_8(v) basis (64), each b (280): candidate = a · [D_F, b]
        # restricted to upper-right row-block-v (64 rows × 24 cols = 1536 complex).
        # Since a is non-zero only in vertex-v's matter block, a · [D_F, b] also has
        # support in row-block-v of the upper-right.
        # Specifically: (a · M)_{row v, col} = (a_block-v) · M_{row v, col}.
        n_pairs = 64 * n_b
        candidates = np.zeros((1536, n_pairs), dtype=complex)
        col = 0
        for i_a in range(8):
            for j_a in range(8):
                E_a = np.zeros((8, 8), dtype=complex); E_a[i_a, j_a] = 1
                a_64 = np.kron(np.eye(8, dtype=complex), E_a)  # left-mult by E_a on M_8 flatten
                for cb in comm_blocks:
                    # cb is (256, 24).  Vertex v's rows of cb are cb[64v:64(v+1), :].
                    # a · cb has row-block-v = a_64 · cb[64v:64(v+1), :].
                    block_v = a_64 @ cb[v*64:(v+1)*64, :]
                    candidates[:, col] = block_v.flatten()
                    col += 1
        U, S, Vh = np.linalg.svd(candidates, full_matrices=False)
        rank_v = int(np.sum(S > TOL * S[0])) if S[0] > TOL else 0
        print(f"  Vertex v={v}: rank of {col} candidates = {rank_v}")
        bases_per_vertex.append(U[:, :rank_v])

    # Aggregate: full upper-right block is 256 × 24 = 6144 complex.
    # Per-vertex basis lives in 64 × 24 = 1536-complex sub-block (rows [64v, 64v+64)).
    # Place each vertex's basis into the appropriate row-block of the 6144-complex flatten.
    full_upper_dim = dim0 * dim1
    total_rank = sum(b.shape[1] for b in bases_per_vertex)
    print(f"\n  Total upper-right Ω_D¹ rank = {total_rank} (expected 768)")

    # Stack per-vertex bases into a 6144 × 768 complex basis matrix.
    # Per-vertex local flatten is (64, 24) → 1536; embed into (256, 24) → 6144.
    Omega_basis = np.zeros((full_upper_dim, total_rank), dtype=complex)
    col_offset = 0
    for v in range(NV):
        bv = bases_per_vertex[v]
        n_v = bv.shape[1]
        # Each column of bv is a 1536-vector representing 64×24 local block.
        # Embed into 256×24 global block: rows [64v, 64(v+1)) of (256, 24).
        for k in range(n_v):
            local = bv[:, k].reshape(64, 24)  # 64×24
            global_block = np.zeros((dim0, dim1), dtype=complex)
            global_block[v*64:(v+1)*64, :] = local
            Omega_basis[:, col_offset + k] = global_block.flatten()
        col_offset += n_v

    # Sanity: orthonormalise (different vertices already orthogonal; intra-vertex from SVD already orthonormal)
    Q, R = np.linalg.qr(Omega_basis)
    rank_final = int(np.sum(np.abs(np.diag(R)) > TOL))
    print(f"  After QR orthonormalisation: rank = {rank_final}")

    return Q[:, :rank_final], dim0, dim1


# ---------------------------------------------------------------------------
# Part B: J-real reduction
# ---------------------------------------------------------------------------

def part_B_J_real(Omega_basis_complex: np.ndarray, dim0: int, dim1: int):
    """For Φ Hermitian with upper-right block B ∈ Omega_basis_complex span,
    apply J-real condition  P_m · conj(B) · P_g = B  (KO-dim 0 sign +1).
    Return real-dim of J-real Hermitian Ω_D¹ for both J^α and J^β.
    """
    print("\n" + "=" * 100)
    print("PART B — apply Hermitian + J-real reductions")
    print("=" * 100)
    # Hermitian: parametrize Φ by B ∈ upper-right Ω_D¹.  Real-dim of Hermitian
    # Ω_D¹ = 2 × complex-dim of upper-right Ω_D¹ = 2 × 768 = 1536 real DOF.
    n_complex = Omega_basis_complex.shape[1]
    print(f"  Hermitian Ω_D¹ real-dim = 2 × {n_complex} = {2 * n_complex} (B free; lower-left = B†)")

    # Build B-space basis: span of n_complex columns viewed as real vectors of
    # dim 2 × 6144 = 12288.  Each column gives 2 real basis vectors (re, im).
    flat_dim = dim0 * dim1
    real_basis = np.zeros((2 * flat_dim, 2 * n_complex), dtype=float)
    for k in range(n_complex):
        vec = Omega_basis_complex[:, k]
        # Real component: B = vec.real → real_vec = (real(vec), imag(vec)) = (vec.real, vec.imag)
        # We use the natural real-isomorphism C^flat → R^{2*flat}: v → (re(v), im(v)).
        real_basis[:flat_dim, 2 * k] = vec.real
        real_basis[flat_dim:, 2 * k] = vec.imag
        # iB: B → iB has (re, im) → (-im, re)
        real_basis[:flat_dim, 2 * k + 1] = -vec.imag
        real_basis[flat_dim:, 2 * k + 1] = vec.real
    # Orthonormalise
    Qr, _ = np.linalg.qr(real_basis)
    rank_real = int(np.linalg.matrix_rank(Qr, tol=TOL))
    print(f"  Hermitian Ω_D¹ real-dim (verified by QR): {rank_real}")
    # ^ This should equal 2 * n_complex = 1536.

    # Now apply J-real for each J variant.
    for variant in ('alpha', 'beta'):
        print(f"\n  --- J^({variant}) (KO-dim 0 sign +1) ---")
        # Build P_matter (256×256) and P_gauge (24×24) for this variant.
        if variant == 'alpha':
            P_64 = J_alpha_64(); P_4 = J_alpha_4()
        else:
            P_64 = J_beta_64(); P_4 = J_beta_4()
        P_matter = np.zeros((dim0, dim0), dtype=complex)
        for v in range(NV):
            P_matter[v*64:(v+1)*64, v*64:(v+1)*64] = P_64
        P_gauge = np.zeros((dim1, dim1), dtype=complex)
        for e in range(NE):
            P_gauge[e*4:(e+1)*4, e*4:(e+1)*4] = P_4

        # For each B in B-space, the J-real condition is:
        #   conj(B) - P_matter @ B @ P_gauge = 0      (J real with +1 sign in KO-dim 0)
        # Equivalently: conj(B) = P_matter B P_gauge.
        # In real coords: let B = X + iY (X, Y real).  Then conj(B) = X - iY.
        # P_matter B P_gauge = P_matter (X + iY) P_gauge = P_matter X P_gauge + i P_matter Y P_gauge.
        # Equating: X - iY = P_matter X P_gauge + i P_matter Y P_gauge.
        # So  X = P_matter X P_gauge  and  Y = -P_matter Y P_gauge.
        # I.e., X is symmetric under conjugation by P (with eigenvalue +1)
        # and Y is anti-symmetric (eigenvalue -1).
        # For real (P_matter, P_gauge) the action B → P_m B P_g is real-linear on real-flatten.

        # Practical: project each Hermitian basis element to the J-real subspace.
        # Compute, for each column of `real_basis`, the projection onto J-real eigenspace.
        # J-real action on a B-vector (split into real (X) and imag (Y)) is:
        #   (X, Y) → (P_m X P_g, -P_m Y P_g)
        # The +1 eigenspace consists of (X, Y) with X = P_m X P_g and Y = -P_m Y P_g.
        # We compute the projector onto this eigenspace by averaging:  P_+ = (I + J)/2.

        # Build the action of "J on real-flatten" as a (2*flat_dim, 2*flat_dim) real matrix.
        # Costly: flat_dim = 6144 → matrix is 12288×12288 = 1.5e8 entries (1.2 GB).  Too big.
        # Use rank-tractable approach: apply J to each basis vector of real_basis,
        # then form (b + J b)/2 to get the +1 eigenspace projection.

        # Action of J on a B-vector encoded as 12288 real:
        #   given (X, Y) (each 6144 real = 256×24), return (P_m X P_g, -P_m Y P_g) flattened.
        # We just apply it directly.

        def apply_J_real(vec_real: np.ndarray) -> np.ndarray:
            X = vec_real[:flat_dim].reshape(dim0, dim1)
            Y = vec_real[flat_dim:].reshape(dim0, dim1)
            X_new = (P_matter.real @ X @ P_gauge.real)
            Y_new = -(P_matter.real @ Y @ P_gauge.real)
            out = np.empty_like(vec_real)
            out[:flat_dim] = X_new.flatten()
            out[flat_dim:] = Y_new.flatten()
            return out

        # Apply (I + J)/2 to each column of real_basis, then SVD to find rank.
        n_cols = real_basis.shape[1]
        proj = np.zeros_like(real_basis)
        for k in range(n_cols):
            proj[:, k] = 0.5 * (real_basis[:, k] + apply_J_real(real_basis[:, k]))
        # Rank of projection = dim of J-real subspace
        U, S, _ = np.linalg.svd(proj, full_matrices=False)
        rank_J_real = int(np.sum(S > TOL * S[0])) if S[0] > TOL else 0
        print(f"    J-real Hermitian Ω_D¹ real-dim = {rank_J_real}")
        print(f"    (= {rank_J_real // 2} real-scalar DOF if grouped in complex pairs)")
        # Reference counts for comparison
        print(f"    Compare to:")
        print(f"      SM Higgs (1 doublet)       = 4 real DOF")
        print(f"      2HDM       (2 doublets)    = 8 real DOF")
        print(f"      MSSM Higgs + sfermions/gen × 3 gens ≈ 98 real DOF (no ν_R)")
        print(f"      MSSM with ν̃_R              ≈ 104 real DOF")


# ---------------------------------------------------------------------------
def main():
    print(r"""
==========================================================================================
Z — CC reductions of Ω_D¹ = 1536 → physical scalar count
Refined per parameter-ledger insight: MSSM-dependence is gauge-RG-only;
sfermions/gauginos contribute Δb = (+5/2, +25/6, +4).
==========================================================================================""")
    Omega_basis_complex, dim0, dim1 = part_A_build_omega_D1_upper_right()
    part_B_J_real(Omega_basis_complex, dim0, dim1)
    print("\n" + "=" * 100)
    print("Z probe Phase 1: sentinel done.  Phase 2 (gauge decomposition + β contribution)")
    print("is a separate session contingent on Phase 1's reduced count.")
    print("=" * 100)


if __name__ == "__main__":
    main()
