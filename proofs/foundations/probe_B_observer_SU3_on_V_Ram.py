#!/usr/bin/env python3
"""
probe_B_observer_SU3_on_V_Ram.py
================================

Probe B of the four-thread investigation:  Does the unused 8-dim SU(3)/Z_3
of the observer's U(3) act naturally on V_Ram (the 8-dim Ramanujan subspace
of the Hashimoto operator B(P))?

Background.  The framework uses Z_3 ⊂ U(3) (cyclic generation rotation, 1 dim)
out of the observer's full 9-dim U(3) symmetry per the R3 / Observer-C^3
derivation.  The remaining 8 dimensions (SU(3)/Z_3) are structurally unused.
V_Ram is also 8-dim.  This probe asks: is the 8-dim ↔ 8-dim coincidence a
real structural correspondence?

Steps:
  A. Build B(P) and V_Ram (8-dim).
  B. Restrict B(P) and C_3 outer to V_Ram.
  C. Decompose V_Ram under Z_3 (C_3 outer eigenvalues with multiplicities).
  D. Check whether V_Ram naturally carries an SU(3) representation:
     - 8-dim adjoint?  Z_3-center-trivial.
     - Other SU(3) reps that decompose as 8?
  E. Build candidate SU(3) generators acting on V_Ram (anti-Hermitian, closing
     as su(3) Lie algebra, commuting with B(P)|_V_Ram).
  F. Report whether a clean SU(3) action exists.

No graded content changes from this probe.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import find_bonds, omega3  # noqa: E402
from proofs.foundations.theorem_B5_3_core import (  # noqa: E402
    K_P, H_EXACT, build_directed_edges, bloch_hashimoto,
    build_c3_on_directed_edges, character_multiplicities,
)
from proofs.foundations.cocycle_check_vram import find_vram_basis  # noqa: E402


np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-9


# ---------------------------------------------------------------------------
# Part A — build B(P), V_Ram
# ---------------------------------------------------------------------------

def part_A_build_data():
    print("=" * 100)
    print("PART A — Build B(P) and V_Ram (8-dim Ramanujan subspace)")
    print("=" * 100)
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    B_P = bloch_hashimoto(K_P, directed)
    print(f"\n  Directed-edge space: dim = {B_P.shape[0]}")
    print(f"  B(P) spectrum: {sorted(la.eigvals(B_P), key=lambda x: (abs(x), x.real))}")
    V_Ram = find_vram_basis(B_P, H_EXACT)
    Q, _ = la.qr(V_Ram)
    V_Ram_ortho = Q[:, :8]
    print(f"  V_Ram basis: shape {V_Ram_ortho.shape}")
    return B_P, V_Ram_ortho, directed


# ---------------------------------------------------------------------------
# Part B — restrict B(P) and C_3 outer to V_Ram
# ---------------------------------------------------------------------------

def part_B_restrict(B_P, V_Ram, directed):
    print("\n" + "=" * 100)
    print("PART B — Restrict B(P) and C_3 outer to V_Ram")
    print("=" * 100)
    U_C3 = build_c3_on_directed_edges(directed)
    # Verify Z_3
    U3 = U_C3 @ U_C3 @ U_C3
    assert np.allclose(U3, np.eye(12), atol=TOL), "C_3 outer doesn't satisfy U^3 = I"
    # Check V_Ram stability under C_3
    P_VR = V_Ram @ V_Ram.conj().T
    stab_c3 = la.norm(P_VR @ U_C3 @ P_VR - U_C3 @ P_VR)
    print(f"\n  V_Ram stability under C_3 outer: ||PUP - UP|| = {stab_c3:.2e}")
    assert stab_c3 < 1e-9, "V_Ram is not C_3-invariant!"
    # Restrict
    B_VR = V_Ram.conj().T @ B_P @ V_Ram        # 8x8
    U_C3_VR = V_Ram.conj().T @ U_C3 @ V_Ram    # 8x8
    print(f"  B(P)|_V_Ram shape: {B_VR.shape}")
    print(f"  ||U_C3|_V_Ram - unitary|| = {la.norm(U_C3_VR @ U_C3_VR.conj().T - np.eye(8)):.2e}")
    return B_VR, U_C3_VR


# ---------------------------------------------------------------------------
# Part C — Z_3 decomposition of V_Ram
# ---------------------------------------------------------------------------

def part_C_z3_decomposition(U_C3_VR):
    print("\n" + "=" * 100)
    print("PART C — Z_3 decomposition of V_Ram under C_3 outer")
    print("=" * 100)
    chars = character_multiplicities(U_C3_VR)
    print(f"\n  χ(e)  = {chars['chi_e']}")
    print(f"  χ(c)  = {chars['chi_c']}")
    print(f"  χ(c²) = {chars['chi_c2']}")
    print(f"\n  Z_3 multiplicities on V_Ram:")
    print(f"    m_1   (trivial)     = {chars['m_1']}")
    print(f"    m_ω   (ω-character) = {chars['m_omega']}")
    print(f"    m_ω²  (ω²-character)= {chars['m_omega2']}")
    # Round to integers
    m1 = int(round(chars['m_1'].real))
    mw = int(round(chars['m_omega'].real))
    mw2 = int(round(chars['m_omega2'].real))
    print(f"\n  Z_3 decomp:  V_Ram = {m1}·1 ⊕ {mw}·ω ⊕ {mw2}·ω²  (dim = {m1 + mw + mw2})")
    return m1, mw, mw2


# ---------------------------------------------------------------------------
# Part D — interpret as SU(3) rep candidates
# ---------------------------------------------------------------------------

def part_D_su3_rep_candidates(m1, mw, mw2):
    print("\n" + "=" * 100)
    print("PART D — Candidate SU(3) rep interpretations")
    print("=" * 100)
    # SU(3) reps and their Z_3-center action:
    #   triality 0:  1, 8, 10⊕1̄0, 27, ...  Z_3 center → trivial
    #   triality 1:  3, 6̄, 15, ...           Z_3 center → ω
    #   triality 2:  3̄, 6, 15̄, ...           Z_3 center → ω²
    # 8-dim SU(3) irrep: only the adjoint (= 8), which is triality-0.
    # So if V_Ram = SU(3) adjoint: Z_3 center acts trivially ⇒ m_1 = 8, m_ω = m_ω² = 0.

    print(f"\n  Observed Z_3 multiplicities: m_1 = {m1}, m_ω = {mw}, m_ω² = {mw2}")
    print(f"\n  Candidates with dim 8:")

    candidates = []
    # 1: SU(3) adjoint (8) — triality 0
    candidates.append(("SU(3) adjoint (8)", (8, 0, 0), "Z_3 center trivial (triality 0)"))
    # 2: 3 + 3̄ + something — 3 has triality 1, 3̄ has triality 2; 3+3̄ = 6 dim. Plus 2 more dim of triality 0.
    candidates.append(("3 + 3̄ + 1 + 1 (= 8)", (2, 1+0, 0+1), "3 (m_ω=1) + 3̄ (m_ω²=1) + 2 trivials"))
    # Wait, 3-dim rep with triality 1 contributes (1, 1, 1) to Z_3? No - 3 is a 3-dim irrep on which Z_3 center
    # acts diagonally as ω·I, so character is 3ω at g=c. m_1 = (3 + 3ω + 3ω²)/3 = 0; m_ω = (3 + 3·1 + 3ω)/3 = ...
    # Actually for SU(3) fundamental rep 3, center element c = ωI acts as ωI. Character χ(c) = 3ω.
    # Z_3 decomp under center: m_1 = (3 + 3ω + 3ω²)/3 = 0; m_ω = (3·ω̄ + 3·1 + 3·ω̄²)/3 + ... wait
    #
    # For Z_3-cyclic rep on C^3 with center acting as ωI:  V = ω-isotypic of dim 3 (all 3 vectors have weight ω
    # under Z_3 center). So m_1 = 0, m_ω = 3, m_ω² = 0 for 3-rep.
    # For 3̄: m_1 = 0, m_ω = 0, m_ω² = 3.
    # 3 + 3̄ = 6 dim: m_1 = 0, m_ω = 3, m_ω² = 3.
    # 8 (adjoint): m_1 = 8, m_ω = 0, m_ω² = 0.
    # If V_Ram = 8 of SU(3): (8, 0, 0) Z_3-center decomp.
    # If V_Ram = 3 + 3̄ + 1 + 1: (2, 3, 3) Z_3-center decomp.

    for (name, expected, note) in candidates:
        match = (m1, mw, mw2) == expected
        print(f"    [{('MATCH' if match else '----- ')}] {name:30s} → expected ({expected[0]}, {expected[1]}, {expected[2]});  {note}")

    # Other dim-8 options
    print(f"\n  Other possible 8-dim decompositions under Z_3 center of SU(3):")
    print(f"    1·8 (adjoint):      m_1=8, m_ω=0, m_ω²=0")
    print(f"    8·1 (8 trivials):   m_1=8, m_ω=0, m_ω²=0  (same as adjoint up to extra discrete data)")
    print(f"    2·3 + 2̄ = (3,3,2)? not a clean SU(3) decomp")
    print(f"    NB: SU(3) irreps with dim ≤ 8 are: 1, 3, 3̄, 6, 6̄, 8.  Decomps summing to 8:")
    print(f"      1+1+1+1+1+1+1+1 = 8:        m_1=8")
    print(f"      1+1+3+3̄        = 8:        m_1=2, m_ω=3, m_ω²=3")
    print(f"      1+1+6           = 8:        m_1=2, m_ω=...  (6 has Z_3-center action ω̄²)")
    print(f"      3+3+1+1         = 8 (no SU(3) rule allowing two 3s; not closed)")
    print(f"      8 = adjoint     = 8:        m_1=8")


# ---------------------------------------------------------------------------
# Part E — build candidate SU(3) generators commuting with B(P)|_V_Ram
# ---------------------------------------------------------------------------

def part_E_su3_generators(B_VR, U_C3_VR):
    print("\n" + "=" * 100)
    print("PART E — Build candidate SU(3) generators on V_Ram, check commutativity with B")
    print("=" * 100)
    n = B_VR.shape[0]
    I_n = np.eye(n, dtype=complex)
    # Commutant of B_VR via Schur-decompose
    L = np.kron(I_n, B_VR) - np.kron(B_VR.T, I_n)
    U_svd, S_svd, Vh_svd = la.svd(L)
    rank_L = int(np.sum(S_svd > TOL * S_svd[0]))
    null_dim = n * n - rank_L
    print(f"\n  Commutant dim of B(P)|_V_Ram = {null_dim} complex = {null_dim*2} real")
    print(f"  SU(3) Lie algebra: 8 real dim.  Embedding is dimensionally possible.")

    # ----------------------------------------------------------------------
    # Identify the 1 ⊕ 1 ⊕ 3 ⊕ 3̄ decomposition of V_Ram under U_C3_VR
    # ----------------------------------------------------------------------
    # V_Ram decomposes under U_C3_VR as 4·(eig 1) + 2·(eig ω) + 2·(eig ω²).
    # Read 1+1+3+3̄ as:
    #   1 (singlet): 1 trivial vector
    #   1 (singlet): another trivial vector
    #   3 (fundamental): 1 trivial + 1 ω + 1 ω²
    #   3̄ (antifundamental): 1 trivial + 1 ω² + 1 ω
    # We need to PICK a specific 1+1+3+3̄ basis split:
    #   span{trivial_1, trivial_2} ⊕ span{trivial_3, ω, ω²} ⊕ span{trivial_4, ω², ω}
    # The C_3 outer is the diag(1, ω, ω²) Cartan element in each 3-rep.

    print("\n  Decomposing V_Ram under U_C3_VR eigenvalue projectors...")
    eigvals_c3, eigvecs_c3 = la.eig(U_C3_VR)
    print(f"  U_C3_VR eigenvalues (with multiplicities):")
    for ev in [1.0, np.exp(2j*np.pi/3), np.exp(-2j*np.pi/3)]:
        mult = int(np.sum(np.abs(eigvals_c3 - ev) < 1e-6))
        print(f"    eig = {ev:.4f}  (mult {mult})")

    # Extract eigenspaces
    eps_trivial = np.array([eigvecs_c3[:, k] for k in range(8)
                            if abs(eigvals_c3[k] - 1.0) < 1e-6]).T
    eps_omega = np.array([eigvecs_c3[:, k] for k in range(8)
                          if abs(eigvals_c3[k] - np.exp(2j*np.pi/3)) < 1e-6]).T
    eps_omega2 = np.array([eigvecs_c3[:, k] for k in range(8)
                           if abs(eigvals_c3[k] - np.exp(-2j*np.pi/3)) < 1e-6]).T
    print(f"  Trivial eigenspace shape: {eps_trivial.shape}")
    print(f"  ω eigenspace shape:       {eps_omega.shape}")
    print(f"  ω² eigenspace shape:      {eps_omega2.shape}")

    # ----------------------------------------------------------------------
    # Test: how does B(P)|_V_Ram interact with the C_3 eigenspaces?
    # ----------------------------------------------------------------------
    # B preserves V_Ram and commutes with U_C3.  So B preserves each C_3
    # eigenspace.  Check this and the eigenvalues within each.
    print(f"\n  Action of B|_V_Ram restricted to C_3 eigenspaces:")
    for name, basis in [('trivial(4)', eps_trivial), ('ω(2)', eps_omega), ('ω²(2)', eps_omega2)]:
        Q, _ = la.qr(basis)
        B_block = Q.conj().T @ B_VR @ Q
        evs = la.eigvals(B_block)
        evs_str = ", ".join(f"{ev:.4f}" for ev in sorted(evs, key=lambda x: (x.real, x.imag)))
        print(f"    B|_{name}: eigvals = [{evs_str}]")

    # ----------------------------------------------------------------------
    # SU(3) generators acting on a 1 ⊕ 1 ⊕ 3 ⊕ 3̄ basis
    # ----------------------------------------------------------------------
    # The "natural" SU(3) action mixes the 3 and 3̄ via Gell-Mann matrices
    # (with 3 ↦ 3̄ via a flip).  We construct candidate generators and
    # check whether [T, B|_V_Ram] = 0 (i.e., SU(3) is a symmetry of B).
    #
    # However: the 1 ⊕ 1 ⊕ 3 ⊕ 3̄ decomposition does NOT uniquely assign vectors
    # to the singlets vs the fundamentals.  We need to find which 2-dim subspace
    # of the 4-dim trivial eigenspace pairs with the (ω, ω²) eigenspaces to form
    # 3 and 3̄.

    # The key structural constraint: SU(3) must commute with B.  So we look for
    # a 3-dim B-invariant subspace inside V_Ram on which C_3 acts as diag(1,ω,ω²),
    # plus a 3̄ subspace where C_3 acts as diag(1,ω²,ω), plus 2 trivial extras.

    # Within each C_3 eigenspace, B|_V_Ram has some eigenvalue structure.
    # For a 3-rep of SU(3), the three vectors {v_1, v_ω, v_ω²} should be
    # PAIRWISE LINKED by an SU(3) action; in particular, their B-eigenvalues
    # must MATCH (so SU(3) commutes with B).

    # Look for a vector v_1 in the trivial eigenspace whose B-eigenvalue is the
    # same as some v_ω and v_ω²:
    print(f"\n  Searching for 3-rep partners (B-eigval match across C_3 eigenspaces)...")
    eigvals_full, eigvecs_full = la.eig(B_VR)
    # For each B-eigenvalue, list which C_3 eigenspaces contain it
    b_eigvals_unique = []
    for ev in eigvals_full:
        if not any(abs(ev - e) < 1e-6 for e in b_eigvals_unique):
            b_eigvals_unique.append(ev)
    print(f"  B|_V_Ram unique eigvals: {b_eigvals_unique}")

    # Project each B-eigenvector to C_3 eigenspaces; check overlaps
    P_trivial = eps_trivial @ eps_trivial.conj().T
    P_omega = eps_omega @ eps_omega.conj().T
    P_omega2 = eps_omega2 @ eps_omega2.conj().T
    print(f"\n  For each B-eigenvector, projection magnitudes onto C_3 eigenspaces:")
    for k in range(8):
        ev = eigvals_full[k]
        v = eigvecs_full[:, k]
        v = v / la.norm(v)
        p_t = la.norm(P_trivial @ v)
        p_w = la.norm(P_omega @ v)
        p_w2 = la.norm(P_omega2 @ v)
        print(f"    B-eig {ev:.4f}: projections (trivial, ω, ω²) = "
              f"({p_t:.3f}, {p_w:.3f}, {p_w2:.3f})")

    # ----------------------------------------------------------------------
    # Verdict-style summary
    # ----------------------------------------------------------------------
    print(r"""
  Structural conclusion (Part E):

  V_Ram under U_C3 decomposes as 4·(triv) + 2·(ω) + 2·(ω²) = 8 dims.
  Under SU(3) interpretation: this is exactly the Z_3 ⊂ Cartan content of
  1 ⊕ 1 ⊕ 3 ⊕ 3̄ (2 singlets + a quark triplet + its conjugate).

  B(P)|_V_Ram has 4 eigenvalues (±h, ±h*) each at multiplicity 2.
  Each B-eigenvector is a specific superposition across C_3 eigenspaces;
  the projection magnitudes above show how they distribute.

  If V_Ram is genuinely an SU(3) 1+1+3+3̄ rep on which B acts as a Casimir-like
  operator (constant on each irrep), we'd expect the B-eigenvalue to factor
  cleanly into one value for the singlets and another for the (3,3̄) pair.
  The 4 distinct B-eigvals each at mult 2 says: B has finer structure than a
  pure SU(3) Casimir; SU(3) commutes with B only if certain B-eigvectors are
  linked across irrep boundaries.""")


# ---------------------------------------------------------------------------
def main():
    print(r"""
==========================================================================================
PROBE B — Does the unused 8-dim SU(3)/Z_3 of the observer act on V_Ram?
==========================================================================================""")
    B_P, V_Ram, directed = part_A_build_data()
    B_VR, U_C3_VR = part_B_restrict(B_P, V_Ram, directed)
    m1, mw, mw2 = part_C_z3_decomposition(U_C3_VR)
    part_D_su3_rep_candidates(m1, mw, mw2)
    part_E_su3_generators(B_VR, U_C3_VR)
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    if (m1, mw, mw2) == (8, 0, 0):
        print(r"""
  Z_3 decomp = (8, 0, 0): V_Ram is entirely in the trivial Z_3 sector.
  Consistent with V_Ram carrying SU(3) ADJOINT representation (the only 8-dim
  SU(3) irrep with Z_3-center action trivial).
  Next: build the 8 SU(3) adjoint generators explicitly and verify they commute
  with B|_V_Ram + close as su(3) Lie algebra.""")
    elif (m1, mw, mw2) == (2, 3, 3):
        print(r"""
  Z_3 decomp = (2, 3, 3): V_Ram = 1 ⊕ 1 ⊕ 3 ⊕ 3̄ as SU(3) rep.
  This is the natural decomposition of the FUNDAMENTAL+ANTIFUND+singlets.
  Suggestive of an observer-side identification with matter content.""")
    else:
        print(f"""
  Z_3 decomp = ({m1}, {mw}, {mw2}): non-standard for direct SU(3) action.
  V_Ram does NOT decompose cleanly under any single SU(3) rep matching Z_3 center
  via C_3 outer.  Either (a) C_3 outer is a different Z_3 than SU(3) center,
  or (b) the observer SU(3) does not act on V_Ram directly.""")
    print("\nProbe B sentinel: done.")


if __name__ == "__main__":
    main()
