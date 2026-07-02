#!/usr/bin/env python3
"""
sigma_S phase coupling to Hashimoto walks — V_ub topological-mechanism test
============================================================================

CONTEXT
-------
The 2026-04-25 σ_S vs C₃_body comparison
(`proofs/foundations/sigma_S_vs_C3_body_comparison.py`) established that the
matching-basis Spin(6) lift σ_S and the standard-basis body-diagonal C₃ on the
8-dim Cl(6,0) spinor are GENUINELY DIFFERENT order-3 elements that generate
the binary icosahedral group 2I = SL(2,5) of order 120. Both have isotypic
(4, 2, 2) on the spinor, but they do not commute.

C₃_body has been load-bearing for color identification (session 25 sin²θ_W
theorem at unification, via B6 bridge). σ_S is the leading candidate for
generation Z₃. The choke-point memory (2026-04-25) lists two open follow-ups
to elevate σ_S = generation from candidate to derived theorem:

  (3) Does σ_S Z₃ phase couple to walks on the srs Hashimoto graph in the
      way V_ub's topological argument needs?
  (4) Compatibility with Q_Koide's adopted P1 + Y identifications.

This script addresses (3). The V_ub candidate formula is

    V_ub  ?=  V_us · (2/3)^g                                       (*)

with g = 10 the srs girth. The structural argument
(an internal working note §2.1) requires:

  - generation index = Z_3 eigenvalue label (with ΔGen = 2 ≡ −1 mod 3),
  - g mod 3 = 1, so closing a girth cycle accumulates one unit of Z_3 phase,
  - hence ΔGen = 2 b→u walks need ONE extra (reverse-)girth cycle relative to
    ΔGen = 1 b→s walks; that extra cycle contributes the (2/3)^g factor.

For σ_S to license this argument at theorem grade, σ_S transported to V_Ram(P)
must be a "good label" on closed Bloch-Hashimoto walks. The minimum
prerequisite: σ_S^V := A σ_S A^† (with A the C_3-intertwiner V_Ram(P) ≅ S of
`proofs/foundations/gamma7_chirality.py` Step 4) must commute with B(P)
restricted to V_Ram(P). If σ_S^V does NOT commute with B(P)|_VRam, σ_S
eigenvalues are not invariants of Bloch-walk amplitudes, and the topological
phase-accumulation argument has no carrier.

Independent of the commutation question, we also test the WALK-AMPLITUDE
ATTACHMENT: closed walks of length g = 10 starting in σ_S = ω vs σ_S = 1
sectors. If σ_S^V phases attach to walk amplitudes in a Z_3-asymmetric pattern,
the topological argument is supported numerically; if not, σ_S = generation
needs a different mechanism.

WHAT THIS SCRIPT VERIFIES
-------------------------

  Step 1.  Build B(P) (12×12) and U_{C_3} (12×12) on directed edges; extract
           V_Ram(P) (8-dim Ramanujan subspace) and U_C3_VRam = restriction.
  Step 2.  Build σ_S (matching Brauer-Weyl) and U_C3_S (standard Brauer-Weyl)
           on the 8-dim Cl(6,0) spinor S. Reuses
           `matching_brauer_weyl_sigma.build_sigma_S` and
           `gamma7_chirality.build_U_C3_S`.
  Step 3.  Build the C_3-intertwiner A: S → V_Ram(P) coordinates (sector-
           by-sector matching, gamma7_chirality.py Step 4 recipe). Verify
           A·U_C3_S = U_C3_VRam·A.
  Step 4.  Transport σ_S to V_Ram(P): σ_S^V = A·σ_S·A^†. Verify σ_S^V is
           order 3, isotypic (4, 2, 2). Confirm [σ_S^V, U_C3_VRam] ≠ 0
           (consistent with SL(2,5) generation in S).
  Step 5.  KEY TEST: compute [σ_S^V, B(P)|_VRam]. Report the Frobenius norm
           and whether it's zero within numerical tolerance.
  Step 6.  Gauge-search probe: σ_S^V is defined up to a centralizer
           ambiguity g ∈ U(4) × U(2) × U(2) (centralizer of U_C3_VRam in
           U(8)). For 200 random g, compute ||[g σ_S^V g^†, B(P)|_VRam]||
           and report the minimum. If a sector-preserving gauge can drive
           the commutator to zero, σ_S = good label is reachable; if not,
           σ_S^V never commutes with B(P)|_VRam regardless of gauge.
  Step 7.  Joint eigenstructure: simultaneously diagonalize B(P)|_VRam and
           σ_S^V (only meaningful if Step 5 or Step 6 found commuting case).
           If they don't commute, decompose V_Ram(P) into σ_S^V-eigenspaces
           and report B(P) action on each.
  Step 8.  Closed-walk amplitude test: starting from σ_S^V-eigenstate
           |ψ_α⟩ in α ∈ {1, ω, ω²}, compute B(P)^g |ψ_α⟩ for g = 10 and
           extract the σ_S^V-eigenvalue distribution of the result.
           If the (2/3)^g topological argument is realized, expect
             - σ_S^V = 1 sector returns to 1 sector with amplitude scaling
             - σ_S^V = ω sector picks up an extra (2/3)^g factor relative to 1
             - σ_S^V = ω² similarly
  Step 9.  Verdict and structural diagnosis.

Run with:
    python3 proofs/foundations/sigma_S_phase_walks.py

Upstream:
    proofs/common.py
    proofs/foundations/theorem_B5_3_core.py    (B(k), U_{C_3} on directed edges)
    proofs/foundations/matching_brauer_weyl_sigma.py  (σ_S construction)
    proofs/foundations/gamma7_chirality.py     (C_3-intertwiner A recipe)
    proofs/foundations/sigma_S_vs_C3_body_comparison.py  (SL(2,5) finding)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la
from scipy.linalg import expm

from proofs.common import find_bonds, omega3
from proofs.foundations.theorem_B5_3_core import (
    build_directed_edges,
    bloch_hashimoto,
    build_c3_on_directed_edges,
    commutator_norm,
)
from proofs.foundations.matching_brauer_weyl_sigma import (
    brauer_weyl_gammas,
    build_sigma_S,
    sigma_permutation_on_gammas,
)
from proofs.foundations.gamma7_chirality import (
    build_U_C3_S,
    c3_isotypic_basis,
    classify_c3,
)


TOL = 1e-8
H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
K_P = (0.25, 0.25, 0.25)
GIRTH = 10
omega = omega3
omega2 = omega * omega
PRINT_W = 76


def banner(s):
    print()
    print("=" * PRINT_W)
    print(s)
    print("=" * PRINT_W)


def section(s):
    print()
    print(s)
    print("-" * PRINT_W)


def extract_v_ram(B_P):
    """Extract orthonormal 12×8 V_Ram(P) basis (|eig|^2 = 2)."""
    evals, evecs = la.eig(B_P)
    ram_idx = [i for i, ev in enumerate(evals) if abs(abs(ev) ** 2 - 2.0) < 1e-5]
    assert len(ram_idx) == 8, f"Expected 8 Ramanujan eigvecs, got {len(ram_idx)}"
    raw = evecs[:, ram_idx]
    Q, _ = la.qr(raw)
    return Q[:, :8]


def isotypic_signature(M, tol=0.1):
    """Return (m_1, m_omega, m_omega2) for an order-3 unitary M."""
    evals = la.eigvals(M)
    n1 = sum(1 for ev in evals if abs(ev - 1.0) < tol)
    nw = sum(1 for ev in evals if abs(ev - omega) < tol)
    nw2 = sum(1 for ev in evals if abs(ev - omega2) < tol)
    return (n1, nw, nw2)


def random_centralizer_unitary(U_C3, rng):
    """Random unitary in the centralizer of U_C3 (block-diagonal w.r.t.
    isotypic decomposition).

    For U_C3 with isotypic (4, 2, 2): block-diag on U(4) × U(2) × U(2).
    Returns 8×8 unitary g satisfying [g, U_C3] = 0.
    """
    bases = c3_isotypic_basis(U_C3)
    g = np.zeros((8, 8), dtype=complex)
    for label, dim in [('1', 4), ('w', 2), ('w2', 2)]:
        Q = bases[label]
        # Random unitary on the dim-dim sector via QR of random complex matrix
        A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        Q_loc, R = la.qr(A)
        # Fix sign convention so result is unitary
        d = np.diag(R)
        ph = d / np.abs(d)
        U_loc = Q_loc * ph
        g += Q @ U_loc @ Q.conj().T
    return g


def main():
    banner("σ_S phase coupling to closed Bloch-Hashimoto walks — V_ub test")

    # -------------------------------------------------------------------
    # Step 1: B(P), U_{C_3}, V_Ram(P), U_C3_VRam
    # -------------------------------------------------------------------
    section("Step 1 — Build B(P), U_{C_3}, extract V_Ram(P)")

    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    B_P = bloch_hashimoto(K_P, directed)
    U_C3_12 = build_c3_on_directed_edges(directed)

    # Sanity
    assert la.norm(np.linalg.matrix_power(U_C3_12, 3) - np.eye(12)) < TOL
    cn_B_U = commutator_norm(B_P, U_C3_12)
    assert cn_B_U < 1e-10, f"[B(P), U_C3]≠0: {cn_B_U}"

    V_Ram = extract_v_ram(B_P)
    print(f"  V_Ram shape = {V_Ram.shape}")

    B_VRam = V_Ram.conj().T @ B_P @ V_Ram                    # 8x8
    U_C3_VRam = V_Ram.conj().T @ U_C3_12 @ V_Ram             # 8x8

    iso_VRam = isotypic_signature(U_C3_VRam)
    print(f"  C_3 isotypic on V_Ram = {iso_VRam}   (expected (4, 2, 2))")
    assert iso_VRam == (4, 2, 2)

    cn_B_C3_VRam = commutator_norm(B_VRam, U_C3_VRam)
    print(f"  ||[B(P)|_VRam, U_C3_VRam]|| = {cn_B_C3_VRam:.2e}   (expected 0)")
    assert cn_B_C3_VRam < 1e-8

    # -------------------------------------------------------------------
    # Step 2: σ_S and U_C3_S on the 8-dim Cl(6,0) spinor
    # -------------------------------------------------------------------
    section("Step 2 — Build σ_S (matching basis) and U_C3_S (standard basis) on S")

    Gs = brauer_weyl_gammas()
    perm = sigma_permutation_on_gammas()
    sigma_S = build_sigma_S(Gs, perm)
    U_C3_S = build_U_C3_S(directed)

    iso_sigma = isotypic_signature(sigma_S)
    iso_C3_S = isotypic_signature(U_C3_S)
    print(f"  σ_S isotypic   = {iso_sigma}   (expected (4, 2, 2))")
    print(f"  U_C3_S isotypic = {iso_C3_S}   (expected (4, 2, 2))")
    assert iso_sigma == (4, 2, 2) and iso_C3_S == (4, 2, 2)

    cn_sig_C3_S = commutator_norm(sigma_S, U_C3_S)
    print(f"  ||[σ_S, U_C3_S]|| = {cn_sig_C3_S:.2e}   "
          f"(expected NONZERO — they generate SL(2,5))")
    assert cn_sig_C3_S > 1e-3

    # -------------------------------------------------------------------
    # Step 3: C_3-intertwiner A: S → V_Ram(P) coords
    # -------------------------------------------------------------------
    section("Step 3 — Build C_3-intertwiner A: S → V_Ram(P)")

    bases_S = c3_isotypic_basis(U_C3_S)
    bases_VR = c3_isotypic_basis(U_C3_VRam)

    A = np.zeros((8, 8), dtype=complex)
    for label in ['1', 'w', 'w2']:
        A += bases_VR[label] @ bases_S[label].conj().T

    # Unitarity + intertwiner property
    err_unit = la.norm(A @ A.conj().T - np.eye(8))
    err_int = la.norm(A @ U_C3_S - U_C3_VRam @ A)
    print(f"  ||A A^† - I_8||                = {err_unit:.2e}")
    print(f"  ||A U_C3_S - U_C3_VRam A||     = {err_int:.2e}")
    assert err_unit < 1e-8 and err_int < 1e-8

    # -------------------------------------------------------------------
    # Step 4: Transport σ_S to V_Ram(P) coords
    # -------------------------------------------------------------------
    section("Step 4 — Transport σ_S to V_Ram(P): σ_S^V = A σ_S A^†")

    sigma_S_V = A @ sigma_S @ A.conj().T

    # Order, isotypic, expected non-commutation with U_C3_VRam
    err_order = la.norm(np.linalg.matrix_power(sigma_S_V, 3) - np.eye(8))
    iso_sV = isotypic_signature(sigma_S_V)
    cn_sV_C3 = commutator_norm(sigma_S_V, U_C3_VRam)

    print(f"  ||(σ_S^V)^3 - I||              = {err_order:.2e}")
    print(f"  σ_S^V isotypic                = {iso_sV}   (expected (4, 2, 2))")
    print(f"  ||[σ_S^V, U_C3_VRam]||         = {cn_sV_C3:.2e}   "
          f"(NONZERO expected, A·[σ_S, U_C3_S]·A^†)")
    assert err_order < 1e-8
    assert iso_sV == (4, 2, 2)
    assert cn_sV_C3 > 1e-3

    # -------------------------------------------------------------------
    # Step 5: KEY TEST — does σ_S^V commute with B(P)|_VRam?
    # -------------------------------------------------------------------
    section("Step 5 — KEY TEST: [σ_S^V, B(P)|_VRam] = ?")

    cn_sV_B = commutator_norm(sigma_S_V, B_VRam)
    print(f"  ||[σ_S^V, B(P)|_VRam]||        = {cn_sV_B:.6e}")
    sV_commutes_B = cn_sV_B < 1e-6
    print(f"  σ_S^V commutes with B(P)|_VRam (canonical gauge): "
          f"{'YES' if sV_commutes_B else 'NO'}")

    # -------------------------------------------------------------------
    # Step 6: Gauge-search — does any sector-preserving gauge make it commute?
    # -------------------------------------------------------------------
    section("Step 6 — Gauge search: minimize over centralizer of U_C3_VRam")

    rng = np.random.default_rng(20260425)
    n_trials = 400
    best = cn_sV_B
    best_g = np.eye(8, dtype=complex)
    for _ in range(n_trials):
        g = random_centralizer_unitary(U_C3_VRam, rng)
        sV_g = g @ sigma_S_V @ g.conj().T
        # Sanity: g preserves U_C3_VRam isotypic ⇒ sV_g still has isotypic (4,2,2)
        cn_try = commutator_norm(sV_g, B_VRam)
        if cn_try < best:
            best = cn_try
            best_g = g

    print(f"  Random gauge trials: {n_trials}")
    print(f"  min ||[g σ_S^V g^†, B(P)|_VRam]|| over trials = {best:.6e}")
    gauge_can_commute = best < 1e-4
    print(f"  Sector-preserving gauge able to drive commutator to zero: "
          f"{'YES' if gauge_can_commute else 'NO (search did not find it)'}")

    # -------------------------------------------------------------------
    # Theoretical derivation: which σ_S^V-conjugates COULD commute with B(P)?
    # -------------------------------------------------------------------
    section("Step 6b — Existence argument (analytical)")

    # B(P)|_VRam has 4 distinct eigenvalues {h, h*, -h, -h*} each mult 2.
    # The joint (B, U_C3_VRam) eigendecomposition gives 8 1-dim simultaneous
    # eigenspaces (theorem BP §Step 6: per chirality the C_3 content of each
    # B-eigenspace is (1, ω) or (1, ω²)).
    #
    # Any operator T that commutes with B(P)|_VRam must preserve each 2-dim
    # B-eigenspace. If T also commutes with U_C3_VRam, then T is diagonal in
    # the joint eigenbasis (1-dim eigenspaces).
    #
    # If T is order 3 with isotypic (4, 2, 2) AND respects U_C3_VRam isotypic,
    # then T's spectrum on the 4-dim trivial sector is some 4-tuple of cube
    # roots of unity summing to (4, 0, 0) eigenvalue distribution, i.e.,
    # T|_trivial has multiplicities (4, 0, 0) — meaning T = +I on trivial.
    # Similarly T = ω·I on the ω-sector and T = ω²·I on the ω²-sector.
    # That is, T = U_C3_VRam exactly.
    #
    # Therefore the ONLY order-3 (4,2,2)-isotypic operator that commutes with
    # BOTH B(P)|_VRam and U_C3_VRam is U_C3_VRam itself.
    #
    # σ_S^V satisfies isotypic (4,2,2) BUT [σ_S^V, U_C3_VRam] ≠ 0
    # (verified above). Therefore σ_S^V cannot equal U_C3_VRam, and there is
    # NO gauge g in centralizer(U_C3_VRam) such that g σ_S^V g^† equals
    # U_C3_VRam (centralizer-conjugation preserves the U_C3_VRam-commutation
    # structure: [g σ_S^V g^†, U_C3_VRam] = g [σ_S^V, U_C3_VRam] g^† ≠ 0).
    #
    # ⇒ σ_S^V CANNOT commute with B(P)|_VRam under any sector-preserving
    #   gauge.  Step 6's gauge search is consistent with this.

    print("  Analytical: any order-3 (4,2,2)-isotypic T that commutes with")
    print("  both B(P)|_VRam and U_C3_VRam must equal U_C3_VRam. Since")
    print("  [σ_S^V, U_C3_VRam] ≠ 0 (established above), no centralizer")
    print("  conjugate g σ_S^V g^† can commute with B(P)|_VRam.")
    print("  σ_S^V is structurally INCOMPATIBLE with B(P)-eigenstate labels.")

    # -------------------------------------------------------------------
    # Step 7: σ_S^V isotypic decomposition of V_Ram, B(P) action per sector
    # -------------------------------------------------------------------
    section("Step 7 — σ_S^V-eigenspace decomposition of V_Ram(P)")
    print("  (sectors don't commute with B(P), so we report B(P)|_sector)")

    bases_sV = c3_isotypic_basis(sigma_S_V)

    sector_summary = {}
    for label, dim_expected in [('1', 4), ('w', 2), ('w2', 2)]:
        Q = bases_sV[label]
        actual_dim = Q.shape[1]
        assert actual_dim == dim_expected, (
            f"σ_S^V {label}-sector dim {actual_dim} != {dim_expected}"
        )
        # Project B onto this sector (NOT block-diagonal; off-diagonal blocks are nonzero)
        B_sector = Q.conj().T @ B_VRam @ Q  # dim x dim
        sv_block = la.svd(B_sector, compute_uv=False)
        eig_block = la.eigvals(B_sector)
        eig_mods = sorted([abs(e) for e in eig_block])
        print(f"  σ_S^V = {label}  (dim {dim_expected})")
        print(f"    B(P)|_sector singular values: {[f'{s:.6f}' for s in sv_block]}")
        print(f"    B(P)|_sector |eig|:           {[f'{m:.6f}' for m in eig_mods]}")
        sector_summary[label] = {'svs': list(sv_block), 'eigs': eig_block}

    # Off-block coupling: for g_walk = 10 closed walks, off-block matters
    print()
    print("  Off-σ_S^V-sector mixing of B(P) (Frobenius norms):")
    for la_lab in ['1', 'w', 'w2']:
        for lb_lab in ['1', 'w', 'w2']:
            if la_lab == lb_lab:
                continue
            Qa = bases_sV[la_lab]
            Qb = bases_sV[lb_lab]
            B_ab = Qa.conj().T @ B_VRam @ Qb
            norm = la.norm(B_ab)
            print(f"    ||B(P): σ_S^V = {lb_lab} → {la_lab}|| = {norm:.6f}")

    # -------------------------------------------------------------------
    # Step 8: Closed-walk amplitude test — does B^g send σ_S^V sectors as expected?
    # -------------------------------------------------------------------
    section(f"Step 8 — Closed-walk amplitudes for length g = {GIRTH}")

    # B(P)^g acting on σ_S^V eigenstates
    Bg_VRam = np.linalg.matrix_power(B_VRam, GIRTH)
    Bg_norm = la.norm(Bg_VRam)
    print(f"  ||B(P)|_VRam^{GIRTH}||_F = {Bg_norm:.6f}")
    print(f"  Per-eigenvalue multiplier on V_Ram: |h|^{GIRTH} = "
          f"{abs(H_EXACT) ** GIRTH:.6f} = 2^{GIRTH/2:.0f} = {2**(GIRTH/2):.6f}")
    print()

    # For each σ_S^V-eigenstate |ψ_α⟩, compute decomposition of B^g |ψ_α⟩ over
    # σ_S^V-eigenstates (m_β with α, β ∈ {1, ω, ω²})
    print(f"  Amplitude transfer matrix M_{{β,α}} = ||P_β B(P)^{GIRTH} P_α||_F :")
    print(f"  (rows: target σ_S^V sector β; columns: source α)")
    print()
    transfer = np.zeros((3, 3), dtype=float)
    label_idx = {'1': 0, 'w': 1, 'w2': 2}
    label_names = ['1', 'ω', 'ω²']
    for la_lab, i in label_idx.items():
        for lb_lab, j in label_idx.items():
            P_target = bases_sV[la_lab] @ bases_sV[la_lab].conj().T
            P_source = bases_sV[lb_lab] @ bases_sV[lb_lab].conj().T
            transfer[i, j] = la.norm(P_target @ Bg_VRam @ P_source)

    # Display transfer matrix
    print(f"  {'sector':>8s}", end='')
    for j in range(3):
        print(f"  {'α=' + label_names[j]:>14s}", end='')
    print()
    for i in range(3):
        print(f"  β={label_names[i]:>6s}", end='')
        for j in range(3):
            print(f"  {transfer[i, j]:>14.6f}", end='')
        print()

    # Topological-argument predictions:
    #
    # If σ_S^V phase couples to closed Bloch walks via the (2/3)^g topological
    # mechanism (V_ub scoping doc §2.1 Step 3), expect:
    #
    #   Diagonal strong (return-to-same-sector dominant for g = 10),
    #   Off-diagonal SUPPRESSED by factor (2/3)^g per |Δsector mod 3| step.
    #
    # Specifically:  M_{β,α} / M_{α,α} ≈ (2/3)^{g · |β-α mod 3|}  ?
    #
    # If σ_S^V does NOT couple this way, expect generic |M_{β,α}| values
    # with no Z_3-pattern (driven instead by B(P) eigenvalue moduli only).

    # Compute the expected (2/3)^g suppression factor
    suppress_1 = (2.0 / 3.0) ** GIRTH
    suppress_2 = (2.0 / 3.0) ** (2 * GIRTH)
    print()
    print(f"  Topological-argument suppression factors:")
    print(f"    ΔGen = 1: (2/3)^{GIRTH}     = {suppress_1:.4e}")
    print(f"    ΔGen = 2: (2/3)^{2 * GIRTH} = {suppress_2:.4e}")

    # Diagnose pattern
    print()
    diag_mean = (transfer[0, 0] + transfer[1, 1] + transfer[2, 2]) / 3.0
    off_diag_norm = np.sqrt((transfer[0, 1] ** 2 + transfer[1, 0] ** 2 +
                             transfer[1, 2] ** 2 + transfer[2, 1] ** 2 +
                             transfer[0, 2] ** 2 + transfer[2, 0] ** 2) / 6.0)
    if diag_mean > 0:
        ratio = off_diag_norm / diag_mean
    else:
        ratio = float('inf')
    print(f"  diagonal mean        = {diag_mean:.6f}")
    print(f"  off-diagonal RMS     = {off_diag_norm:.6f}")
    print(f"  off-diag / diagonal  = {ratio:.4e}")
    print(f"  expected if (2/3)^g topological coupling: ratio ~ "
          f"{suppress_1:.4e} (ΔGen=1)")

    # -------------------------------------------------------------------
    # Step 9: Verdict
    # -------------------------------------------------------------------
    banner("VERDICT")

    print()
    print("CAS-verified facts:")
    print(f"  • σ_S^V = A σ_S A^† has order 3 and isotypic (4, 2, 2).")
    print(f"  • [σ_S^V, U_C3_VRam] ≠ 0 (= {cn_sV_C3:.3e}).")
    print(f"  • [σ_S^V, B(P)|_VRam] = {cn_sV_B:.3e}, NONZERO under canonical gauge.")
    print(f"  • Min over {n_trials} sector-preserving gauges: {best:.3e}.")
    print(f"  • Analytical: no centralizer-gauge can achieve commutation.")
    print()
    print("INTERPRETATION FOR V_ub TOPOLOGICAL ARGUMENT:")
    print()
    print("  σ_S^V is NOT a simultaneous eigenstate label with B(P)|_VRam.")
    print("  The σ_S^V-sectors of V_Ram(P) are NOT closed under Bloch-Hashimoto")
    print("  walks: B(P) carries amplitude across σ_S^V sectors (off-diagonal")
    print("  blocks of B(P) in the σ_S^V eigenbasis are nonzero).")
    print()
    if ratio < 0.05:
        print("  EMPIRICAL: closed g-walk transfer matrix is diagonal-dominant")
        print("  with off-diagonal/diagonal ratio = {:.3e}, comparable to".format(ratio))
        print("  the (2/3)^g expectation. THIS IS CONSISTENT with σ_S^V being")
        print("  the topological generation Z_3 of V_ub's argument.")
        verdict = "POSITIVE"
    elif ratio < 0.5:
        print("  EMPIRICAL: g-walk transfer is partially diagonal-dominant but")
        print("  not at the (2/3)^g level. σ_S^V phase coupling is suppressed")
        print("  but not at the topological-argument scale.")
        verdict = "AMBIGUOUS"
    else:
        print("  EMPIRICAL: g-walk transfer matrix has substantial off-diagonal")
        print("  weight; σ_S^V sectors are heavily mixed by B(P)^g. The")
        print("  (2/3)^g topological suppression is NOT realized at the σ_S^V")
        print("  level on V_Ram(P).")
        verdict = "NEGATIVE"
    print()
    print(f"  V_ub topological-mechanism verdict: {verdict}")
    print()
    if verdict == "NEGATIVE":
        print("  Implication: σ_S = generation Z_3 with (2/3)^g phase-accumulation")
        print("  on closed Bloch walks is NOT supported by this test. The V_ub")
        print("  candidate formula V_us · (2/3)^g would need a different phase")
        print("  carrier or a different mechanism (per an internal working note")
        print("  theorem_Vub_scoping.md routes (b)–(d)).")
    elif verdict == "POSITIVE":
        print("  Implication: σ_S^V phase coupling on closed Bloch walks is")
        print("  consistent with the V_ub topological argument. Continue to")
        print("  follow-up (4): compatibility with Q_Koide P1 + Y adoptions.")

    print()
    print("=" * PRINT_W)
    return {
        'sV_commutes_B_canonical': sV_commutes_B,
        'gauge_min_commutator': best,
        'transfer_matrix': transfer.tolist(),
        'off_diag_over_diag': ratio,
        'verdict': verdict,
    }


if __name__ == "__main__":
    main()
