#!/usr/bin/env python3
"""
proofs/foundations/path_b_T0_T1_z2_symmetry.py

PURPOSE
-------
Analytical proof that at the Bloch point N1 = (0, 0, 1/2), the K_4 cycle-
transfer operators U_T_0 = P_T_0 · B(N1)^3 · P_T_0 and
U_T_1 = P_T_1 · B(N1)^3 · P_T_1 give equal expectation values on every
B(N1)-eigenmode in V_Ram(N1).

This was the "T_0 ≡ T_1 sub-symmetry" item flagged in
an internal working note as a
~1-session pending piece for Doc 1 closure (Majorana phases α_21, α_31).

MECHANISM
---------
The structural reason is a unitary Z_2 symmetry σ on the 12-dim directed-
edge basis with the following properties:

  1. σ is the unique lift of the K_4 vertex transposition (2 3) to
     directed edges, defined by σ(s, t, c) = (σ_v(s), σ_v(t), c'),
     where σ_v = (2 3) and c' is the unique cell-shift such that
     (σ_v(s), σ_v(t), c') is a directed bond in the srs primitive cell.
     (Each ordered (s', t') vertex pair has exactly one bond, by the
     structural property of srs.)

  2. σ^2 = I (involution).

  3. [σ, B(N1)] = 0 — the N1-specific commutation. At general k, σ does
     NOT commute with B(k); this is a Bloch-point-stabilizer ("little
     group") symmetry that emerges only at N1 because every directed
     edge i and its image σ(i) have cells with the same n_3 (mod 2),
     and the N1 phase factor exp(2πi · N1 · cell) = (-1)^{n_3} only
     depends on n_3 mod 2.

  4. σ swaps the cycle-incidence projectors: σ^† P_T_0 σ = P_T_1,
     σ^† P_T_2 σ = P_T_2. (Because σ_v = (2 3) on K_4 vertices swaps
     the triangles {0,1,2} ↔ {0,1,3} and fixes {0,2,3}.)

THEOREM AND PROOF (chain)
-------------------------
By (3) and (4), σ^† U_T_0 σ = U_T_1.

V_Ram(N1) has 8 distinct B(N1)-eigenvalues, so each V_Ram mode ψ_λ is
unique up to phase as a B(N1)-eigenvector. By (3), σ commutes with B(N1),
so σ acts on each ψ_λ as a scalar c_λ. By (2), σ^2 = I forces c_λ ∈ {±1}.

Therefore for any V_Ram mode ψ_λ:
    ⟨ψ_λ| U_T_1 |ψ_λ⟩
        = ⟨ψ_λ| σ^† U_T_0 σ |ψ_λ⟩
        = c̄_λ · c_λ · ⟨ψ_λ| U_T_0 |ψ_λ⟩
        = |c_λ|² · ⟨ψ_λ| U_T_0 |ψ_λ⟩
        = ⟨ψ_λ| U_T_0 |ψ_λ⟩.   ∎

The off-diagonal entries of (U_T_0 - U_T_1) on V_Ram are NOT zero —
the equality is *diagonal-on-eigenbasis*, not full operator equality.
This is exactly what σ produces: σ swaps the operators globally but
fixes each V_Ram eigenvector individually (up to ±1 sign).

WHAT THIS PROBE VERIFIES
------------------------
  P1. The σ-permutation is well-defined and an involution.
  P2. [σ, B(N1)] = 0 to machine precision.
  P3. σ does NOT commute with B(k) for generic k (N1-specificity).
  P4. σ^† P_T_0 σ = P_T_1, σ^† P_T_2 σ = P_T_2 (exact identity).
  P5. σ^† U_T_0 σ = U_T_1 to machine precision.
  P6. V_Ram has 8 distinct eigenvalues; each V_Ram mode is a σ-
      eigenvector with c_λ ∈ {±1}.
  P7. Diagonal expectation equality: ⟨ψ|U_T_0|ψ⟩ = ⟨ψ|U_T_1|ψ⟩ for
      every V_Ram mode ψ, to machine precision.
  P8. Diagonal-only equality (the operators differ off-diagonally
      on V_Ram).

GATE STATUS
-----------
This probe converts the CAS-verified observation "T_0 ≡ T_1 at N1"
(see Section 4 / Observation 1 of `path_b_cycle_transfer_operator_
2026-05-03.md`) into an analytical proof grounded in (a) explicit
σ-permutation construction from K_4 graph automorphism + lift
uniqueness, (b) finite parity check for [σ, B(N1)] = 0, (c) standard
unique-eigenvector argument given non-degenerate spectrum.

This is one of the three pending sub-pieces flagged in §6 of the
2026-05-03 cycle-transfer-operator doc. Closing it tightens the Doc 1
closure path; the cycle-space basis effectively reduces to 2
inequivalent classes at N1 (the T_0 = T_1 class + the T_2 class)
which combined with eigenvalue (sign, conj, type) bits gives the
observed 8 distinct V_Ram modes.

CROSS-REFERENCES
----------------
    (analytical writeup)
    (operator-level finding; this probe closes its §6 item 1)
    (Z_3 gauge equivalence on cycle-space subsets)
    (Doc 1 master)
  - `proofs/foundations/path_b_v_ram_cycle_space_bijection.py`
    (Path B1.a probe)
"""

import os
import sys

import numpy as np
from numpy import linalg as la

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from proofs.common import find_bonds
from proofs.foundations.theorem_B5_3_core import (
    bloch_hashimoto, build_directed_edges,
)


# =============================================================================
# CONFIG
# =============================================================================

N1 = np.array([0.0, 0.0, 0.5])

# Triangle definitions: (frozenset of K_4 edge pairs)
T_PAIRS = [
    {frozenset((0, 1)), frozenset((0, 2)), frozenset((1, 2))},  # T_0 = T_012
    {frozenset((0, 1)), frozenset((0, 3)), frozenset((1, 3))},  # T_1 = T_013
    {frozenset((0, 2)), frozenset((0, 3)), frozenset((2, 3))},  # T_2 = T_023
]

H_MOD_SQ_RAM = 2.0  # |h|² = k* - 1 = 2 (Ramanujan saturation at h_P)


# =============================================================================
# σ — vertex-swap (2 3) lifted to directed edges
# =============================================================================

def vertex_swap_23(v):
    """K_4 vertex transposition (2 3); fixes 0, 1."""
    if v == 2:
        return 3
    if v == 3:
        return 2
    return v


def build_sigma_permutation(directed):
    """
    Construct σ: directed edges → directed edges, the unique lift of (2 3).

    For each directed edge (s, t, c), the image under σ is the unique
    directed edge with vertex pair (σ_v(s), σ_v(t)). The cell shift is
    determined by the bond geometry (NOT just the same c).

    Returns: list of length 12, with sigma_perm[i] = σ(i).

    Asserts uniqueness (one bond per ordered vertex pair) and involution.
    """
    target_to_idx = {}
    for i, (s, t, c) in enumerate(directed):
        key = (s, t)
        assert key not in target_to_idx, (
            f"srs primitive cell has multiple bonds with vertex pair {key} "
            f"— σ-uniqueness fails"
        )
        target_to_idx[key] = i

    sigma_perm = [None] * len(directed)
    for i, (s, t, c) in enumerate(directed):
        s2, t2 = vertex_swap_23(s), vertex_swap_23(t)
        assert (s2, t2) in target_to_idx, (
            f"image vertex pair ({s2}, {t2}) has no bond — σ undefined"
        )
        sigma_perm[i] = target_to_idx[(s2, t2)]

    # Involution check
    for i in range(len(directed)):
        assert sigma_perm[sigma_perm[i]] == i, (
            f"σ not involutive at edge {i}"
        )
    return sigma_perm


def sigma_matrix(sigma_perm, n=12):
    """12x12 unitary permutation matrix for σ."""
    M = np.zeros((n, n), dtype=complex)
    for i, j in enumerate(sigma_perm):
        M[j, i] = 1.0
    return M


def projector_T(t_pairs, directed):
    """Diagonal projector onto directed edges with K_4 edge pair in t_pairs."""
    diag = np.array(
        [1.0 if frozenset((s, t)) in t_pairs else 0.0 for (s, t, _) in directed]
    )
    return np.diag(diag.astype(complex))


# =============================================================================
# Verifications P1-P8
# =============================================================================

def main():
    print("=" * 76)
    print("Path B — analytical proof: T_0 ≡ T_1 sub-symmetry at N1 (Z_2 σ)")
    print("=" * 76)

    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    n = len(directed)
    assert n == 12

    # ---- P1: σ-permutation construction ----
    print("\n[P1] σ-permutation construction (lift of K_4 (2 3) to directed edges)")
    sigma_perm = build_sigma_permutation(directed)
    print(f"     ✓ σ well-defined and σ² = I (involution)")
    print(f"     σ-cycles on directed edges (showing fixed points + transpositions):")
    visited = set()
    cycles = []
    for i in range(n):
        if i in visited:
            continue
        if sigma_perm[i] == i:
            cycles.append((i,))
            visited.add(i)
        else:
            j = sigma_perm[i]
            cycles.append((i, j))
            visited.update({i, j})
    fixed = [c for c in cycles if len(c) == 1]
    swaps = [c for c in cycles if len(c) == 2]
    print(f"       fixed points: {len(fixed)} ({[c[0] for c in fixed]})")
    print(f"       transpositions: {len(swaps)} ({[c for c in swaps]})")
    print()
    print(f"     Edge table (i, src, tgt, cell, n_3, σ(i), n_3 of σ(i)):")
    print(f"       {'i':>3} {'(s,t)':>7} {'cell':>13} {'n_3':>4} | "
          f"{'σ(i)':>4} {'σ(s,t)':>7} {'σcell':>13} {'σn_3':>5}")
    for i in range(n):
        s, t, c = directed[i]
        c_int = tuple(int(x) for x in c)
        j = sigma_perm[i]
        s2, t2, c2 = directed[j]
        c2_int = tuple(int(x) for x in c2)
        print(f"       {i:3d} ({s},{t})  {str(c_int):>13} {c_int[2]:>4} | "
              f"{j:4d} ({s2},{t2})  {str(c2_int):>13} {c2_int[2]:>5}"
              + ("   ← n_3 parity: " + ("MATCH" if (c_int[2] - c2_int[2]) % 2 == 0 else "MISMATCH")))

    SIGMA = sigma_matrix(sigma_perm, n)

    # ---- P2: [σ, B(N1)] = 0 ----
    print("\n[P2] Commutation [σ, B(N1)] = 0")
    B_N1 = bloch_hashimoto(N1, directed)
    comm_norm = la.norm(SIGMA.conj().T @ B_N1 @ SIGMA - B_N1)
    print(f"     ||σ^† B(N1) σ - B(N1)|| = {comm_norm:.2e}    (must be 0)")
    assert comm_norm < 1e-12

    # ---- P3: NOT a symmetry at general k ----
    print("\n[P3] σ does NOT commute with B(k) at generic k (N1-specific symmetry)")
    rng = np.random.default_rng(0)
    max_norm_gen = 0.0
    for _ in range(5):
        k_gen = rng.uniform(-0.5, 0.5, size=3)
        Bk = bloch_hashimoto(k_gen, directed)
        nn = la.norm(SIGMA.conj().T @ Bk @ SIGMA - Bk)
        max_norm_gen = max(max_norm_gen, nn)
    print(f"     max ||σ^† B(k) σ - B(k)|| over 5 random k = {max_norm_gen:.4f}")
    assert max_norm_gen > 1.0  # generic k breaks the symmetry

    # Also at N2, N3 (other Bloch points) — should also be non-symmetry
    for label, kp in [("N2", np.array([0.5, 0.0, 0.0])),
                      ("N3", np.array([0.0, 0.5, 0.0]))]:
        Bp = bloch_hashimoto(kp, directed)
        nnp = la.norm(SIGMA.conj().T @ Bp @ SIGMA - Bp)
        print(f"     ||σ^† B({label}) σ - B({label})|| = {nnp:.4f}    "
              f"(σ is N1-only; analogous Z_2's exist at N2, N3 with different vertex swaps)")

    # ---- P4: projector swap σ^† P_T σ ----
    print("\n[P4] Projector action: σ^† P_T_i σ = P_T_{π(i)} where π = (0 1) on triangle index")
    P0 = projector_T(T_PAIRS[0], directed)
    P1 = projector_T(T_PAIRS[1], directed)
    P2 = projector_T(T_PAIRS[2], directed)
    n01 = la.norm(SIGMA.conj().T @ P0 @ SIGMA - P1)
    n10 = la.norm(SIGMA.conj().T @ P1 @ SIGMA - P0)
    n22 = la.norm(SIGMA.conj().T @ P2 @ SIGMA - P2)
    print(f"     ||σ^† P_T_0 σ - P_T_1|| = {n01:.2e}")
    print(f"     ||σ^† P_T_1 σ - P_T_0|| = {n10:.2e}")
    print(f"     ||σ^† P_T_2 σ - P_T_2|| = {n22:.2e}    (T_2 fixed)")
    assert n01 < 1e-12 and n10 < 1e-12 and n22 < 1e-12

    # ---- P5: σ^† U_T_0 σ = U_T_1 ----
    print("\n[P5] Operator conjugation: σ^† U_T_0 σ = U_T_1 (and σ^† U_T_2 σ = U_T_2)")
    B3 = B_N1 @ B_N1 @ B_N1
    U0 = P0 @ B3 @ P0
    U1 = P1 @ B3 @ P1
    U2 = P2 @ B3 @ P2
    op_norm = la.norm(SIGMA.conj().T @ U0 @ SIGMA - U1)
    op_norm2 = la.norm(SIGMA.conj().T @ U2 @ SIGMA - U2)
    print(f"     ||σ^† U_T_0 σ - U_T_1|| = {op_norm:.2e}")
    print(f"     ||σ^† U_T_2 σ - U_T_2|| = {op_norm2:.2e}")
    assert op_norm < 1e-12 and op_norm2 < 1e-12

    # ---- P6: V_Ram modes are σ-eigenvectors with c ∈ {±1} ----
    print("\n[P6] V_Ram(N1) modes are σ-eigenvectors with c_λ ∈ {±1}")
    eigs, V = la.eig(B_N1)
    ram_idx = [i for i in range(n) if abs(abs(eigs[i]) ** 2 - H_MOD_SQ_RAM) < 1e-6]
    assert len(ram_idx) == 8, f"V_Ram(N1) dim = {len(ram_idx)}, expected 8"

    # Distinct eigenvalues check
    ram_eigs = [eigs[i] for i in ram_idx]
    pairwise_min = min(abs(ram_eigs[i] - ram_eigs[j])
                       for i in range(len(ram_idx)) for j in range(len(ram_idx)) if i != j)
    print(f"     V_Ram has {len(ram_idx)} eigenvalues, pairwise min separation = "
          f"{pairwise_min:.4f}    (non-degenerate if > 0)")
    assert pairwise_min > 1e-6

    print(f"     σ-eigenvalues (c_λ = ⟨ψ|σ|ψ⟩):")
    sigma_eigs = []
    for i in ram_idx:
        psi = V[:, i]
        lam = eigs[i]
        norm_sq = np.real(psi.conj() @ psi)
        c_lam = (psi.conj() @ SIGMA @ psi) / norm_sq
        residual = la.norm(SIGMA @ psi - c_lam * psi)
        sigma_eigs.append(c_lam)
        print(f"       λ = {lam:+.4f}    c_λ = {c_lam:+.4f}    ||σψ - c·ψ|| = {residual:.2e}")
        assert abs(abs(c_lam) - 1.0) < 1e-9, "|c_λ| ≠ 1"
        assert abs(c_lam.imag) < 1e-9, "c_λ has non-trivial imaginary part — σ² ≠ I?"
        assert residual < 1e-9, "ψ_λ is not a σ-eigenvector"

    n_plus = sum(1 for c in sigma_eigs if c.real > 0)
    n_minus = sum(1 for c in sigma_eigs if c.real < 0)
    print(f"     σ-spectrum on V_Ram: {n_plus} (+1) + {n_minus} (-1) = {n_plus + n_minus} (= 8) ✓")

    # ---- P7: Diagonal expectation equality ⟨ψ|U_T_0|ψ⟩ = ⟨ψ|U_T_1|ψ⟩ ----
    print("\n[P7] Diagonal expectation equality on V_Ram modes")
    print(f"       (theorem conclusion: T_0 ≡ T_1 readings on every V_Ram mode)")
    max_diff = 0.0
    for i in ram_idx:
        psi = V[:, i]
        lam = eigs[i]
        norm_sq = np.real(psi.conj() @ psi)
        r0 = (psi.conj() @ U0 @ psi) / norm_sq
        r1 = (psi.conj() @ U1 @ psi) / norm_sq
        max_diff = max(max_diff, abs(r0 - r1))
        print(f"       λ = {lam:+.4f}    ⟨U_T_0⟩ = {r0:+.4f}    "
              f"⟨U_T_1⟩ = {r1:+.4f}    diff = {abs(r0-r1):.2e}")
    print(f"     max |⟨U_T_0⟩ - ⟨U_T_1⟩| over V_Ram = {max_diff:.2e}    (must be 0) ✓")
    assert max_diff < 1e-12

    # ---- P8: diagonal-only — operators differ off-diagonal on V_Ram ----
    print("\n[P8] Diagonal-only equality (operators differ off-diagonal on V_Ram)")
    V_ram_basis = V[:, ram_idx]
    diff = U0 - U1
    diff_in_ram = V_ram_basis.conj().T @ diff @ V_ram_basis  # 8x8
    diag_norm = la.norm(np.diag(np.diag(diff_in_ram)))
    offd_norm = la.norm(diff_in_ram - np.diag(np.diag(diff_in_ram)))
    print(f"     ||V_ram^† (U_T_0 - U_T_1) V_ram|| total       = {la.norm(diff_in_ram):.4f}")
    print(f"     ||diag part||     = {diag_norm:.2e}    (≈ 0 — Theorem)")
    print(f"     ||off-diag part|| = {offd_norm:.4f}    (nonzero — operators differ)")
    print(f"     ⇒ T_0 ≡ T_1 is *eigenbasis-diagonal*, not full operator equality.")
    print(f"     This is exactly what σ-symmetry produces: σ swaps U_T_0 ↔ U_T_1")
    print(f"     globally but fixes each V_Ram eigenvector individually (up to ±1).")
    assert diag_norm < 1e-12
    assert offd_norm > 1.0

    # ---- Summary ----
    print()
    print("=" * 76)
    print("THEOREM (proven)")
    print("=" * 76)
    print()
    print("  At N1 = (0, 0, 1/2), let σ be the unique lift of the K_4 vertex")
    print("  transposition (2 3) to directed edges. Then:")
    print()
    print("    1. σ² = I (involution)")
    print("    2. [σ, B(N1)] = 0")
    print("    3. σ^† P_T_0 σ = P_T_1")
    print("    4. σ^† U_T_0 σ = U_T_1")
    print("    5. V_Ram modes are σ-eigenvectors with c_λ ∈ {±1}")
    print()
    print("  Therefore ⟨ψ_λ| U_T_0 |ψ_λ⟩ = ⟨ψ_λ| U_T_1 |ψ_λ⟩ for every V_Ram")
    print("  mode ψ_λ, with |c_λ|² = 1 absorbing the σ-action.   ∎")
    print()
    print("STRUCTURAL SOURCE")
    print("-----------------")
    print("  σ is a Bloch-point-stabilizer (\"little group\") symmetry that exists")
    print("  only at N1: every directed edge i and its image σ(i) have cells with")
    print("  the same n_3 (mod 2), and the N1 phase factor exp(2πi · N1 · cell)")
    print("  = (-1)^{n_3} is determined by n_3 mod 2 alone. At general k, the")
    print("  phase factor sees n_1, n_2 as well, breaking the parity match. The")
    print("  N2 = (0.5, 0, 0) and N3 = (0, 0.5, 0) Bloch points carry analogous")
    print("  Z_2 symmetries with vertex swaps (1 2) and (1 3) respectively, by")
    print("  the structural Z_3 cycling N1 → N2 → N3 (1→3→2→1 on K_4 vertices).")
    print()
    print("PATH B IMPLICATIONS")
    print("-------------------")
    print("  The cycle-space basis {T_0, T_1, T_2} effectively reduces to two")
    print("  inequivalent classes at N1 — the (T_0 = T_1) class and the T_2 class.")
    print("  Combined with the eigenvalue (sign, conj, type) bits (cf.")
    print("  path_b_v_ram_cycle_space_bijection.py Part 3), this gives the")
    print("  observed 8 distinct V_Ram modes — but NOT as a 2³ tensor product;")
    print("  rather as a (Z_2 × Z_2 × Z_2) classification interleaved with the")
    print("  Z_3 action on the triangle basis (cf. §4 of the cycle-transfer-")
    print("  operator doc 2026-05-03).")
    print()
    print("  This closes the §6.1 pending item of path_b_cycle_transfer_operator_")
    print("  2026-05-03.md. Doc 1 closure tractability advances by one step.")
    print("=" * 76)


if __name__ == "__main__":
    main()
