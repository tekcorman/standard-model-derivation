#!/usr/bin/env python3
"""
Theorem B5.3-core — C_3-equivariant decomposition of the srs Bloch Hashimoto
operator B(k) along the Γ-P fixed axis.

STATEMENT (partial, as verified here).

Let B(k) be the Bloch Hashimoto NB-walk operator on the srs primitive cell
(12 × 12 on directed edges, per ../../predictions/walker_dynamics_derivation.md §W3). Let
C_3 ⊂ 432 be the body-diagonal 3-fold rotation,

    C_3: (k_1, k_2, k_3) ↦ (k_3, k_1, k_2)     (base),
         σ = (v_0)(v_1 v_3 v_2)  on vertices   (fibre),

induced on directed edges as the 12 × 12 permutation matrix U_{C_3}. Then:

(a) [B(k), U_{C_3}] = 0 whenever C_3·k = k, i.e. on the Γ–P fixed axis
    F := {k : k_1 = k_2 = k_3}.

(b) On F, B(k) and U_{C_3} admit a common eigenbasis. The character
    (χ_trivial(k), χ_ω(k), χ_{ω²}(k)) of the C_3 action on the 12-dim
    fibre, restricted to F, is **pointwise constant**:
        (χ_trivial, χ_ω, χ_{ω²})(k ∈ F) = (4, 4, 4)
    and does not depend on k ∈ F.  (Proof: σ-permutation-character
    computation, independent of k, since U_{C_3} is k-independent.)

(c) Each of the three C_3-isotypic subspaces at k ∈ F therefore has
    constant fibre dimension 4 (= multiplicity) — a locally trivial
    equivariant subbundle over F.

(d) Restricted to the 8-dim Ramanujan subspace of B(P) (spectrum
    {±h, ±h*} each with multiplicity 2, |±h|² = |±h*|² = k−1 = 2, per
    ../../predictions/B_P_doubly_degenerate_h_derivation.md), the C_3-isotypic dimensions
    are (4, 2, 2) — matching theorem BP §Step 3.

(e) On the 4-dim tree ±1 subspace of B(P), the C_3-isotypic dimensions
    are (0, 2, 2).

(f) Hence the (4, 4, 4) isotypic decomposition of the full 12-dim fibre
    decomposes as ((4, 2, 2) on Ramanujan) ⊕ ((0, 2, 2) on ±1 tree)
    — pointwise on F, with constant fibre dimensions along F.

WHAT THIS SCRIPT VERIFIES

 Step 0:  Build srs primitive cell, 12 directed edges, U_{C_3} permutation.
 Step 1:  U_{C_3}^3 = I  (order-3 check).
 Step 2:  χ(e) = 12,  χ(U_{C_3}) = χ(U_{C_3}^2) = 0.  Orbit structure:
          4 orbits of length 3, 0 fixed directed edges.
          ⇒  multiplicities  (m_1, m_ω, m_ω²) = (4, 4, 4).
 Step 3:  [B(k), U_{C_3}] = 0  along F := {k : k_1 = k_2 = k_3},
          numerically at multiple points of F (Γ, P/2, P, intermediate).
 Step 4:  At each k ∈ F, simultaneously diagonalize B(k) and U_{C_3}; the
          three C_3-isotypic subspace dimensions are (4, 4, 4) — constant.
 Step 5:  At k = P, restrict to Ramanujan subspace (|eig|² = 2), recover
          (4, 2, 2); restrict to tree ±1 subspace, recover (0, 2, 2).
 Step 6:  Off-axis sanity: [B(k), U_{C_3}] ≠ 0 generically; the C_3 does
          NOT act on a single fibre away from F, but induces an
          intertwiner B(k) ↔ B(C_3·k), consistent with the equivariant
          bundle structure.

RIGOR NOTES

- The (4, 4, 4) character is purely combinatorial on the σ-permutation of
  directed edges (Step 2). It is k-independent and therefore automatic.
- On F, continuity of B(k) plus integrality of isotypic dimensions force
  the (4, 4, 4) multiplicities to be locally constant along F.
- The off-F extension to a constant-fibre-dimension decomposition over
  the full BZ/C_3 quotient is the equivariant-bundle statement of
  Atiyah & Segal 1968; the rigorous version on F is what this script
  verifies directly. See the companion doc
  docs/theorem_B5_3_core.md for the full statement.

Prints "OK:" on success.
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import (
    find_bonds, N_ATOMS, C3_PERM, omega3,
)


# ----------------------------------------------------------------------
# Infrastructure: directed edges, B(k), C_3 on directed edges
# ----------------------------------------------------------------------

H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2        # P-point Hashimoto eigenvalue
K_P = (0.25, 0.25, 0.25)
K_GAMMA = (0.0, 0.0, 0.0)


def build_directed_edges(bonds):
    """find_bonds() returns 12 directed-edge records (src, tgt, cell).
    Verify the count and return as tuples."""
    directed = [tuple(b) for b in bonds]
    assert len(directed) == 12, f"expected 12 directed edges, got {len(directed)}"
    return directed


def bloch_hashimoto(k_frac, directed):
    """12×12 Bloch Hashimoto B(k) on directed edges.

    B(k)[e', e] = exp(2πi · k · cell_{e'})   if e → e' is a valid NB step
                  0                           otherwise.

    Valid NB: target(e) = source(e'), and e' ≠ reverse(e).
    Reverse of (src, tgt, cell) is (tgt, src, −cell).
    """
    n = len(directed)
    B = np.zeros((n, n), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for i_p, (src_p, tgt_p, cell_p) in enumerate(directed):
        for i_e, (src_e, tgt_e, cell_e) in enumerate(directed):
            if tgt_e != src_p:
                continue
            is_reverse = (tgt_p == src_e and
                          tuple(np.array(cell_p) + np.array(cell_e)) == (0, 0, 0))
            if is_reverse:
                continue
            phase = np.exp(2j * np.pi * np.dot(k, cell_p))
            B[i_p, i_e] += phase
    return B


def c3_vertex_perm():
    """C_3 vertex action σ = (v_0)(v_1 v_3 v_2), read from common.C3_PERM.
    Returns a dict  j ↦ σ(j)."""
    perm = {}
    for i in range(4):
        for j in range(4):
            if abs(C3_PERM[i, j] - 1.0) < 1e-12:
                perm[j] = i
    # sanity: σ(v0)=v0, σ(v1)=v3, σ(v2)=v1, σ(v3)=v2
    assert perm == {0: 0, 1: 3, 2: 1, 3: 2}, f"unexpected σ: {perm}"
    return perm


def c3_cell_perm(cell):
    """C_3 on the primitive-cell displacement vector (integer lattice).

    The BCC primitive vectors a_1, a_2, a_3 are permuted by C_3 as
    a_1 → a_2, a_2 → a_3, a_3 → a_1 (one cyclic orbit on the three
    axes under the body-diagonal C_3; see common.A_PRIM). A cell
    label (n_1, n_2, n_3) = n_1 a_1 + n_2 a_2 + n_3 a_3 transforms as
    (n_1, n_2, n_3) → (n_3, n_1, n_2)  (this is σ^{-1} on cells, equivalently
    the contragredient action consistent with σ on vertices)."""
    return (cell[2], cell[0], cell[1])


def build_c3_on_directed_edges(directed):
    """12×12 permutation matrix U_{C_3} representing the C_3 action on
    directed edges. C_3 fixes P and Γ, but U_{C_3} is k-independent
    (pure permutation of the directed-edge basis)."""
    vp = c3_vertex_perm()
    n = len(directed)
    edge_to_idx = {de: i for i, de in enumerate(directed)}
    U = np.zeros((n, n), dtype=complex)
    for i, (src, tgt, cell) in enumerate(directed):
        new_edge = (vp[src], vp[tgt], c3_cell_perm(cell))
        j = edge_to_idx.get(new_edge)
        if j is None:
            raise RuntimeError(
                f"C_3 mapped {(src,tgt,cell)} ↦ {new_edge}, not in directed set"
            )
        U[j, i] = 1.0
    return U


# ----------------------------------------------------------------------
# Step 2: permutation character of U_{C_3} on directed edges
# ----------------------------------------------------------------------

def character_multiplicities(U):
    """Given a permutation matrix U of order 3, compute the multiplicity
    of each C_3 irrep (1, ω, ω²) on the representation it generates.

    m_ρ = (1/|G|) Σ_g χ_ρ(g)^* · χ(g).

    For cyclic C_3 with irreps of dim 1 and χ_{1}(g)=1, χ_ω(g)=ω^k,
    χ_{ω²}(g)=ω^{2k} at g = C_3^k:

        m_1   = (χ(e) + χ(c) + χ(c²))/3
        m_ω   = (χ(e) + ω̄·χ(c) + ω̄²·χ(c²))/3
        m_ω²  = (χ(e) + ω²̄·χ(c) + ω̄·χ(c²))/3
    """
    I = np.eye(U.shape[0])
    chi_e  = np.trace(I).real
    chi_c  = np.trace(U).real + 1j * np.trace(U).imag
    chi_c2 = np.trace(U @ U).real + 1j * np.trace(U @ U).imag
    om = omega3
    m_1   = (chi_e + chi_c + chi_c2) / 3
    m_w   = (chi_e + np.conj(om)    * chi_c + np.conj(om)**2 * chi_c2) / 3
    m_w2  = (chi_e + np.conj(om)**2 * chi_c + np.conj(om)    * chi_c2) / 3
    return {
        'chi_e':   complex(chi_e),
        'chi_c':   complex(chi_c),
        'chi_c2':  complex(chi_c2),
        'm_1':     complex(m_1),
        'm_omega': complex(m_w),
        'm_omega2': complex(m_w2),
    }


def c3_orbit_structure_on_directed_edges(directed):
    """Decompose the 12 directed edges into σ-orbits and report lengths."""
    vp = c3_vertex_perm()
    edge_to_idx = {de: i for i, de in enumerate(directed)}
    visited = set()
    orbits = []
    for i, de in enumerate(directed):
        if i in visited:
            continue
        orbit = [i]
        cur = de
        while True:
            nxt = (vp[cur[0]], vp[cur[1]], c3_cell_perm(cur[2]))
            j = edge_to_idx[nxt]
            if j in orbit:
                break
            orbit.append(j)
            cur = nxt
        visited.update(orbit)
        orbits.append(orbit)
    return orbits


# ----------------------------------------------------------------------
# Step 3/4: Simultaneous diagonalization on the Γ-P axis
# ----------------------------------------------------------------------

def commutator_norm(A, B):
    return la.norm(A @ B - B @ A)


def isotypic_dimensions(B_k, U, tol=1e-6):
    """Simultaneously diagonalize B(k) and U on the subspace where they
    commute. Return dimensions of each C_3 isotypic subspace (m_1, m_ω,
    m_ω²), plus the character triple (χ_e, χ_c, χ_c²) on the total space.

    Caller must ensure [B_k, U] ≈ 0 within numerical tolerance.
    """
    n = U.shape[0]
    # Eigendecompose U (pure permutation; eigenvalues ∈ {1, ω, ω²}).
    evalsU, evecsU = la.eig(U)
    # Classify each U-eigenvector by its U-eigenvalue
    labels = []
    for i, ev in enumerate(evalsU):
        if abs(ev - 1.0) < 0.1:
            labels.append('1')
        elif abs(ev - omega3) < 0.1:
            labels.append('w')
        elif abs(ev - omega3 ** 2) < 0.1:
            labels.append('w2')
        else:
            labels.append(f'?{ev}')
    dim_1 = labels.count('1')
    dim_w = labels.count('w')
    dim_w2 = labels.count('w2')

    # For a non-trivial check that [B, U] has B-preserving U-eigenspace
    # structure, diagonalize B on each U-eigenspace and collect B-eigs.
    isotypic_B_spectrum = {'1': [], 'w': [], 'w2': []}
    for label in ['1', 'w', 'w2']:
        idx = [i for i, l in enumerate(labels) if l == label]
        if not idx:
            continue
        basis = evecsU[:, idx]   # n × m_label
        # Orthonormalize via QR (U eigenvectors at distinct eigs are orthogonal
        # for unitary U; for same eigenvalue block, re-orthonormalize).
        Q, _ = la.qr(basis)
        B_block = Q.conj().T @ B_k @ Q
        ev_block = la.eigvals(B_block)
        isotypic_B_spectrum[label] = list(ev_block)

    return {
        'dims': (dim_1, dim_w, dim_w2),
        'total': dim_1 + dim_w + dim_w2,
        'isotypic_B_spectrum': isotypic_B_spectrum,
    }


def classify_eigs_by_modulus(eigs, tol=1e-6):
    """Classify eigenvalues by |μ|² ∈ {0, 1, 2} and by sign of Re/Im."""
    ram = []     # |μ|² ≈ 2 (Ramanujan: h, h*, -h, -h*)
    tree = []    # |μ|² ≈ 1 (tree: ±1)
    other = []
    for mu in eigs:
        m2 = abs(mu) ** 2
        if abs(m2 - 2.0) < tol:
            ram.append(mu)
        elif abs(m2 - 1.0) < tol:
            tree.append(mu)
        else:
            other.append(mu)
    return ram, tree, other


# ----------------------------------------------------------------------
# Main verification
# ----------------------------------------------------------------------

def main():
    print("=" * 72)
    print("THEOREM B5.3-core — C_3-equivariant decomposition of B(k) on Γ-P axis")
    print("=" * 72)

    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    U = build_c3_on_directed_edges(directed)
    assert U.shape == (12, 12)

    # ─── Step 1: U has order 3 ──────────────────────────────────────────
    print()
    print("Step 1 — U_{C_3} has order 3")
    U3 = U @ U @ U
    res = la.norm(U3 - np.eye(12))
    print(f"  ||U^3 - I|| = {res:.2e}")
    assert res < 1e-10, f"U is not of order 3: ||U^3 - I|| = {res}"

    # ─── Step 2: Character of U on 12-dim directed-edge space ──────────
    print()
    print("Step 2 — Character and multiplicities of C_3 on directed edges")
    ch = character_multiplicities(U)
    print(f"  χ(e)   = {ch['chi_e'].real:.3f}   (expected 12)")
    print(f"  χ(c)   = {ch['chi_c'].real:.3f}{ch['chi_c'].imag:+.3f}i   (expected 0)")
    print(f"  χ(c²)  = {ch['chi_c2'].real:.3f}{ch['chi_c2'].imag:+.3f}i   (expected 0)")
    print(f"  m_1    = {ch['m_1'].real:.3f}{ch['m_1'].imag:+.3f}i   (expected 4)")
    print(f"  m_ω    = {ch['m_omega'].real:.3f}{ch['m_omega'].imag:+.3f}i   (expected 4)")
    print(f"  m_ω²   = {ch['m_omega2'].real:.3f}{ch['m_omega2'].imag:+.3f}i   (expected 4)")
    assert abs(ch['chi_e'].real - 12) < 1e-10
    assert abs(ch['chi_c']) < 1e-10
    assert abs(ch['chi_c2']) < 1e-10
    assert abs(ch['m_1'].real - 4) < 1e-10
    assert abs(ch['m_omega'].real - 4) < 1e-10
    assert abs(ch['m_omega2'].real - 4) < 1e-10

    orbits = c3_orbit_structure_on_directed_edges(directed)
    orbit_lens = sorted([len(o) for o in orbits], reverse=True)
    print(f"  σ-orbits on directed edges: {orbit_lens}   (expected [3, 3, 3, 3])")
    assert orbit_lens == [3, 3, 3, 3], f"orbit lengths {orbit_lens}"

    # ─── Step 3: [B(k), U] = 0 along Γ-P axis ──────────────────────────
    print()
    print("Step 3 — Commutator [B(k), U_{C_3}] along Γ-P axis")
    axis_points = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.49]
    for t in axis_points:
        k = (t, t, t)
        B_k = bloch_hashimoto(k, directed)
        cn = commutator_norm(B_k, U)
        print(f"  k = ({t:.2f}, {t:.2f}, {t:.2f})   ||[B, U]|| = {cn:.2e}")
        assert cn < 1e-10, f"[B(k),U] nonzero at t={t}: {cn}"

    # ─── Step 3b: sanity — off-axis [B(k), U] ≠ 0 ──────────────────────
    print()
    print("Step 3b — Off-axis commutator (sanity: should be nonzero)")
    off_axis = [(0.1, 0.2, 0.3), (0.0, 0.0, 0.5), (0.25, 0.25, 0.5)]
    for k in off_axis:
        B_k = bloch_hashimoto(k, directed)
        cn = commutator_norm(B_k, U)
        print(f"  k = {k}   ||[B, U]|| = {cn:.2e}   (nonzero expected)")
        assert cn > 1e-6, f"[B(k),U] accidentally zero at off-axis k={k}: {cn}"

    # ─── Step 4: constant fibre dimensions (4, 4, 4) along Γ-P ─────────
    print()
    print("Step 4 — Isotypic dimensions along Γ-P axis (constant fibre dim)")
    axis_samples = [0.00, 0.10, 0.20, 0.25, 0.30, 0.40, 0.49]
    for t in axis_samples:
        k = (t, t, t)
        B_k = bloch_hashimoto(k, directed)
        info = isotypic_dimensions(B_k, U)
        dims = info['dims']
        print(f"  t = {t:.2f}   dims (m_1, m_ω, m_ω²) = {dims}   total = {info['total']}")
        assert dims == (4, 4, 4), f"dims at t={t}: {dims}"

    # ─── Step 5: At k=P, match Ramanujan (4,2,2) and tree (0,2,2) ──────
    print()
    print("Step 5 — At k = P: split by Ramanujan vs tree subspaces")
    B_P = bloch_hashimoto(K_P, directed)
    info_P = isotypic_dimensions(B_P, U)
    # For each isotypic block, classify its B-eigenvalues into Ramanujan
    # (|μ|²=2) vs tree (|μ|²=1).
    ram_counts = {'1': 0, 'w': 0, 'w2': 0}
    tree_counts = {'1': 0, 'w': 0, 'w2': 0}
    for label, spec in info_P['isotypic_B_spectrum'].items():
        ram, tree, other = classify_eigs_by_modulus(spec)
        ram_counts[label] = len(ram)
        tree_counts[label] = len(tree)
        # Verify the Ramanujan spectrum is a subset of {h, h*, -h, -h*}
        for mu in ram:
            matches_any = any(
                abs(mu - target) < 1e-7
                for target in (H_EXACT, H_EXACT.conjugate(),
                               -H_EXACT, -H_EXACT.conjugate())
            )
            assert matches_any, f"Ramanujan eig {mu} in block {label} not ±h or ±h*"
        # Verify tree eigs are ±1
        for mu in tree:
            assert abs(mu - 1.0) < 1e-7 or abs(mu + 1.0) < 1e-7, (
                f"tree eig {mu} in block {label} not ±1"
            )
    ram_tuple  = (ram_counts['1'], ram_counts['w'], ram_counts['w2'])
    tree_tuple = (tree_counts['1'], tree_counts['w'], tree_counts['w2'])
    print(f"  Ramanujan (|μ|²=2) fibre dims:  {ram_tuple}   (expected (4, 2, 2))")
    print(f"  Tree      (|μ|²=1) fibre dims:  {tree_tuple}   (expected (0, 2, 2))")
    assert ram_tuple == (4, 2, 2), f"Ramanujan dims {ram_tuple}"
    assert tree_tuple == (0, 2, 2), f"tree dims {tree_tuple}"

    # Cross-check total at P:
    total = tuple(r + tr for r, tr in zip(ram_tuple, tree_tuple))
    print(f"  Total  (Ramanujan + tree)    :  {total}   (expected (4, 4, 4))")
    assert total == (4, 4, 4)

    # ─── Step 5b: same structure at Γ ──────────────────────────────────
    print()
    print("Step 5b — At k = Γ: isotypic decomposition")
    B_G = bloch_hashimoto(K_GAMMA, directed)
    info_G = isotypic_dimensions(B_G, U)
    dims_G = info_G['dims']
    print(f"  dims (m_1, m_ω, m_ω²) at Γ = {dims_G}   (expected (4, 4, 4))")
    assert dims_G == (4, 4, 4)
    # Γ is NOT Ramanujan-saturated for all eigenvalues; spectrum includes
    # ±3 (flat), 0 (triply degenerate triplet factor from Ihara-Bass), etc.
    # We do not enforce a specific Ramanujan / tree split at Γ.

    # ─── Step 5c: Orbit-averaged bundle over BZ/C_3 — off-axis case ────
    # For a generic k, the C_3 orbit {k, C·k, C²·k} consists of 3 distinct
    # points. The equivariant bundle restricted to this 3-point set has
    # total dimension 3 × 12 = 36. The C_3 action on this 36-dim space is
    # well defined: it sends a fibre element at k to the corresponding
    # element at C·k via U_{C_3}, and cyclically. This induces a new 36 × 36
    # "total permutation + fibre permutation" operator whose character can
    # be computed rigorously, with the same multiplicities (12, 12, 12)
    # for (m_1, m_ω, m_ω²) on the direct-sum representation ⊕_{k ∈ orbit}
    # B(k). The key equivariant statement is:
    #
    #   m_ρ(⊕_{k ∈ orbit} B(k)) = m_ρ(Ind_{Stab(k)}^{C_3} ρ_fibre|Stab(k))
    #
    # which by Frobenius reciprocity equals
    #
    #   m_ρ(B(k) at a fixed k) · [C_3 : Stab(k)]
    #
    # For a generic k (Stab(k) = {e}), this gives (m_1, m_ω, m_ω²) =
    # (4, 4, 4) on the single-k fibre level (which is meaningless because
    # C_3 does not act on it) — the correct statement is on the induced
    # representation, which has dimensions 12, 12, 12 (= 4 · 3 each).
    #
    # Here we verify this: combine three off-axis fibres into a 36-dim
    # space, construct the combined C_3 operator, verify it commutes
    # with the block-diag B, and extract its isotypic dimensions.

    print()
    print("Step 5c — Off-axis induced representation (Frobenius reciprocity)")
    k0 = (0.1, 0.2, 0.3)       # generic off-axis k
    k1 = (0.3, 0.1, 0.2)       # C_3 · k0
    k2 = (0.2, 0.3, 0.1)       # C_3^2 · k0
    B0 = bloch_hashimoto(k0, directed)
    B1 = bloch_hashimoto(k1, directed)
    B2 = bloch_hashimoto(k2, directed)
    B_total = np.block([
        [B0,                 np.zeros((12,12)), np.zeros((12,12))],
        [np.zeros((12,12)),  B1,                np.zeros((12,12))],
        [np.zeros((12,12)),  np.zeros((12,12)), B2],
    ])
    # Combined C_3 acts as a cyclic shift across the 3 k-blocks,
    # simultaneously applying U_{C_3} inside each fibre:
    # U_combined sends (ψ_0, ψ_1, ψ_2) ↦ (U·ψ_2, U·ψ_0, U·ψ_1)
    # so that after applying it, the ψ_0' at k_0 comes from ψ_2 at k_2 via U
    # (matching how B(k_0) = U B(k_2) U^{-1} for a C_3-equivariant bundle).
    U_combined = np.zeros((36, 36), dtype=complex)
    U_combined[0:12,   24:36] = U
    U_combined[12:24,  0:12]  = U
    U_combined[24:36,  12:24] = U
    # Verify order 3
    res = la.norm(U_combined @ U_combined @ U_combined - np.eye(36))
    print(f"  ||U_combined^3 - I||  = {res:.2e}")
    assert res < 1e-10
    # Verify [B_total, U_combined] = 0
    cn = commutator_norm(B_total, U_combined)
    print(f"  ||[B_total, U_combined]|| = {cn:.2e}   (must be 0 for equivariance)")
    assert cn < 1e-8, f"equivariance fails off-axis: ||[B,U]|| = {cn}"
    # Character and isotypic dimensions
    ch_c = character_multiplicities(U_combined)
    dims_c = (
        int(round(ch_c['m_1'].real)),
        int(round(ch_c['m_omega'].real)),
        int(round(ch_c['m_omega2'].real)),
    )
    print(f"  Character χ(e, c, c²) = ({ch_c['chi_e'].real:.0f}, "
          f"{ch_c['chi_c'].real:+.2f}, {ch_c['chi_c2'].real:+.2f})   (expected (36, 0, 0))")
    print(f"  Isotypic dims (m_1, m_ω, m_ω²) on 36-dim induced = {dims_c}   "
          f"(expected (12, 12, 12))")
    assert dims_c == (12, 12, 12), f"induced isotypic dims {dims_c}"
    print(f"  Per-k-orbit-slice: (12, 12, 12)/3 = (4, 4, 4), matching F-axis.")

    # ─── Step 6: match to theorem_BP Step 3 (h-eigenspace) ─────────────
    print()
    print("Step 6 — Cross-check: C_3 content of h-eigenspace of B(P)")
    # Use the same machinery: project U onto the h-eigenspace of B(P).
    evalsB, evecsB = la.eig(B_P)
    h_idx = [i for i, ev in enumerate(evalsB) if abs(ev - H_EXACT) < 1e-7]
    assert len(h_idx) == 2, f"h-mult at P = {len(h_idx)}, expected 2"
    h_basis_raw = evecsB[:, h_idx]
    Q, _ = la.qr(h_basis_raw)
    U_sub = Q.conj().T @ U @ Q
    u_evals = la.eigvals(U_sub)
    print(f"  U|h-eigenspace eigenvalues: {[f'{e.real:+.3f}{e.imag:+.3f}i' for e in u_evals]}")
    labels_h = []
    for e in u_evals:
        if abs(e - 1.0) < 0.1:
            labels_h.append('1')
        elif abs(e - omega3) < 0.1:
            labels_h.append('w')
        elif abs(e - omega3 ** 2) < 0.1:
            labels_h.append('w2')
        else:
            labels_h.append('?')
    print(f"  C_3 content of h-eigenspace = {sorted(labels_h)}   (theorem BP: [1, ω])")
    assert sorted(labels_h) == ['1', 'w'], f"h-eigenspace C_3 content = {labels_h}"

    # ─── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("  • U_{C_3} has order 3 on directed edges.")
    print("  • Character (12, 0, 0) ⇒ multiplicities (m_1, m_ω, m_ω²) = (4, 4, 4).")
    print("  • [B(k), U_{C_3}] = 0 everywhere on Γ-P axis F = {k_1=k_2=k_3}.")
    print("  • Isotypic fibre dimensions are constant (4, 4, 4) along F.")
    print("  • At k = P: Ramanujan subspace decomposes as (4, 2, 2),")
    print("    tree subspace as (0, 2, 2), matching theorem BP §Step 3.")
    print("  • Off F, [B(k), U_{C_3}] ≠ 0 on a single fibre: C_3 is a bundle")
    print("    automorphism covering the base action. On a C_3-orbit {k, Ck,")
    print("    C²k}, the combined operator B_total = B(k) ⊕ B(Ck) ⊕ B(C²k)")
    print("    on C^36 commutes with U_combined (numerically verified); its")
    print("    C_3 character is (36, 0, 0), giving isotypic dims (12, 12, 12)")
    print("    = 3 × (4, 4, 4). By Frobenius reciprocity this matches the")
    print("    Γ-P axis multiplicities — the Atiyah-Segal 1968 equivariant")
    print("    K-theory statement, now verified on a single orbit.")
    print()
    print("OK: theorem_B5_3_core verified on Γ-P fixed axis, matches theorem BP at P.")


if __name__ == "__main__":
    main()
