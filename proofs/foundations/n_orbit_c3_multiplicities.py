#!/usr/bin/env python3
"""
N-orbit C_3 isotypic multiplicities — self-consistency test for color-generation
unification.

QUESTION
--------
At the P-point (on the C_3 fixed axis), the fiber C_3 action U_{C_3} restricts to
the 8-dim Ramanujan subspace V_Ram(P) with isotypic multiplicities (4, 2, 2) for
(trivial, omega, omega_bar).  Theorem B5.3-core and theorem_BP establish this.

The N-orbit {N1, N2, N3} is a 3-element orbit under the same C_3 (k-orbit action:
N1=(0,0,1/2) -> N2=(1/2,0,0) -> N3=(0,1/2,0) -> N1).  Since C_3 does NOT
stabilize any N_i (N-points are off the fixed axis), the correct C_3 action on the
combined space V(N1) + V(N2) + V(N3) is a COMBINED operator:
  - k-orbit part:  cyclically permutes the three 12-dim fibers
  - fiber part:    applies U_{C_3} within each fiber

The combined 36x36 operator is:
  C_36 = [[0,       0,       U_{C_3}],
           [U_{C_3}, 0,       0      ],
           [0,       U_{C_3}, 0      ]]
acting on V(N1) + V(N2) + V(N3).

This script:
  1. Verifies N1->N2->N3->N1 under C_3.
  2. Builds B(k_Ni) for each N-orbit point, extracts V_Ram(N_i) (8-dim each).
  3. Builds the combined 36x36 C_3 operator C_36 on V(N1)+V(N2)+V(N3).
  4. Verifies C_36 commutes with the block-diagonal B_total = B(N1)+B(N2)+B(N3).
  5. Restricts C_36 to the 24-dim Ramanujan subspace V_Ram(N1)+V_Ram(N2)+V_Ram(N3).
  6. Computes isotypic multiplicities (n_trivial, n_omega, n_omega_bar) of C_Ram.
  7. Compares to the P-point result (4, 2, 2).
  8. Gives additional check: fiber-only C_3 at N1 (wrong action, diagnostic only).
  9. Structural diagnosis for color-generation unification.

Run with:
    PYTHONPATH=. python3 proofs/foundations/n_orbit_c3_multiplicities.py

Upstream:
  proofs/common.py
  proofs/foundations/theorem_B5_3_core.py  (bloch_hashimoto, build_c3_on_directed_edges)
  proofs/foundations/n_orbit_spectrum.py   (N-orbit coordinates, infrastructure)
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, omega3
from proofs.foundations.theorem_B5_3_core import (
    build_directed_edges,
    bloch_hashimoto,
    build_c3_on_directed_edges,
    commutator_norm,
    character_multiplicities,
)

# ---------------------------------------------------------------------------
# N-orbit coordinates (BCC primitive reduced coordinates)
# ---------------------------------------------------------------------------
N1 = np.array([0.0, 0.0, 0.5])   # "N"
N2 = np.array([0.5, 0.0, 0.0])   # "N_x"
N3 = np.array([0.0, 0.5, 0.0])   # "N_y"

P_PT = np.array([0.25, 0.25, 0.25])

RAMANUJAN_MOD_SQ = 2.0   # |eig|^2 for Ramanujan-saturated eigenvalues at N
K_STAR = 3


def c3_act(k):
    """C_3 on primitive reduced coordinates: (k1,k2,k3) -> (k3,k1,k2)."""
    return np.array([k[2], k[0], k[1]])


def extract_vram(B_k, tol=1e-5, expected_ram=8):
    """
    Extract the Ramanujan subspace V_Ram of B(k): eigenvectors with |eig|^2 = k*-1 = 2.

    Returns:
        evals_ram   : eigenvalues with |eig|^2 ~ 2
        V_Ram       : (12, d) orthonormal matrix, columns span the Ramanujan subspace
    """
    evals, evecs = la.eig(B_k)
    ram_idx = [i for i, ev in enumerate(evals) if abs(abs(ev)**2 - RAMANUJAN_MOD_SQ) < tol]
    assert len(ram_idx) == expected_ram, (
        f"Expected {expected_ram} Ramanujan eigenvalues, got {len(ram_idx)}. "
        f"|eig|^2 values: {sorted(abs(evals)**2)}"
    )
    evecs_raw = evecs[:, ram_idx]   # 12 x 8
    V_Ram, _ = la.qr(evecs_raw)
    V_Ram = V_Ram[:, :len(ram_idx)]
    return evals[ram_idx], V_Ram


def classify_c3_eigs(eigs, tol=0.1):
    """
    Count eigenvalues near 1, omega=exp(2pi*i/3), omega_bar=exp(-2pi*i/3).

    Returns (n_trivial, n_omega, n_omega_bar, n_other).
    """
    om = omega3           # exp(2pi*i/3)
    om2 = omega3 ** 2    # exp(4pi*i/3) = exp(-2pi*i/3) = omega_bar
    n1, nw, nw2, nother = 0, 0, 0, 0
    for ev in eigs:
        if abs(ev - 1.0) < tol:
            n1 += 1
        elif abs(ev - om) < tol:
            nw += 1
        elif abs(ev - om2) < tol:
            nw2 += 1
        else:
            nother += 1
    return n1, nw, nw2, nother


def restricted_c3_eigs(C_full, W, tol_proj=1e-8):
    """
    Restrict the full C_3 operator C_full (n x n) to the subspace spanned by
    columns of W (n x d, orthonormal).

    C_restricted = W^dag C_full W  (d x d matrix).

    Returns the eigenvalues of C_restricted.
    """
    err = la.norm(W.conj().T @ W - np.eye(W.shape[1]))
    assert err < tol_proj, f"W columns are not orthonormal: ||W^dag W - I|| = {err:.2e}"
    C_sub = W.conj().T @ C_full @ W
    return la.eigvals(C_sub), C_sub


def main():
    print("=" * 72)
    print("N-ORBIT C_3 ISOTYPIC MULTIPLICITIES")
    print("Self-consistency test for color-generation unification")
    print("=" * 72)

    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    U_C3 = build_c3_on_directed_edges(directed)   # 12x12 fiber C_3 matrix

    # -----------------------------------------------------------------------
    # STEP 1: Verify N-orbit structure
    # -----------------------------------------------------------------------
    print("\n--- STEP 1: N-orbit verification ---")

    assert np.allclose(c3_act(N1), N2, atol=1e-12), "C_3(N1) != N2"
    assert np.allclose(c3_act(N2), N3, atol=1e-12), "C_3(N2) != N3"
    assert np.allclose(c3_act(N3), N1, atol=1e-12), "C_3(N3) != N1"
    print(f"  C_3(N1) = {c3_act(N1)}   (expected N2 = {N2})  OK")
    print(f"  C_3(N2) = {c3_act(N2)}   (expected N3 = {N3})  OK")
    print(f"  C_3(N3) = {c3_act(N3)}   (expected N1 = {N1})  OK")

    # N-points NOT on fixed axis {(t,t,t)}
    for k, name in [(N1, "N1"), (N2, "N2"), (N3, "N3")]:
        on_axis = abs(k[0] - k[1]) < 1e-12 and abs(k[1] - k[2]) < 1e-12
        assert not on_axis, f"{name} is on the C_3 fixed axis"
    print("  N1, N2, N3 are all off the C_3 fixed axis {(t,t,t)}.  OK")

    # -----------------------------------------------------------------------
    # STEP 2: Build B(k_Ni) and extract Ramanujan subspaces
    # -----------------------------------------------------------------------
    print("\n--- STEP 2: Build B(k_Ni) and extract V_Ram(N_i) ---")

    B_N1 = bloch_hashimoto(N1, directed)
    B_N2 = bloch_hashimoto(N2, directed)
    B_N3 = bloch_hashimoto(N3, directed)

    evals_N1_ram, V_Ram_N1 = extract_vram(B_N1)
    evals_N2_ram, V_Ram_N2 = extract_vram(B_N2)
    evals_N3_ram, V_Ram_N3 = extract_vram(B_N3)

    print(f"  V_Ram(N1): shape {V_Ram_N1.shape},  "
          f"|eig|^2 values: {sorted(set(round(abs(ev)**2, 6) for ev in evals_N1_ram))}")
    print(f"  V_Ram(N2): shape {V_Ram_N2.shape},  "
          f"|eig|^2 values: {sorted(set(round(abs(ev)**2, 6) for ev in evals_N2_ram))}")
    print(f"  V_Ram(N3): shape {V_Ram_N3.shape},  "
          f"|eig|^2 values: {sorted(set(round(abs(ev)**2, 6) for ev in evals_N3_ram))}")

    # Verify all three have 8-dim Ramanujan subspace
    assert V_Ram_N1.shape == (12, 8), f"V_Ram(N1) shape {V_Ram_N1.shape}"
    assert V_Ram_N2.shape == (12, 8), f"V_Ram(N2) shape {V_Ram_N2.shape}"
    assert V_Ram_N3.shape == (12, 8), f"V_Ram(N3) shape {V_Ram_N3.shape}"
    print("  All three N-orbit points have 8-dim Ramanujan subspace.  OK")

    # Verify orbit spectra are identical (C_3-related)
    mods_N1 = sorted(abs(evals_N1_ram)**2)
    mods_N2 = sorted(abs(evals_N2_ram)**2)
    mods_N3 = sorted(abs(evals_N3_ram)**2)
    assert np.allclose(mods_N1, mods_N2, atol=1e-7), "N1 and N2 Ramanujan |eig|^2 differ"
    assert np.allclose(mods_N1, mods_N3, atol=1e-7), "N1 and N3 Ramanujan |eig|^2 differ"
    print("  Ramanujan |eig|^2 spectra are identical across the orbit.  OK")

    # -----------------------------------------------------------------------
    # STEP 3: Build the 36x36 combined C_3 operator on V(N1)+V(N2)+V(N3)
    # -----------------------------------------------------------------------
    print("\n--- STEP 3: Build combined 36x36 C_3 operator C_36 ---")
    print("  C_36 maps:")
    print("    fiber at N1 -> fiber at N2 via U_{C_3}")
    print("    fiber at N2 -> fiber at N3 via U_{C_3}")
    print("    fiber at N3 -> fiber at N1 via U_{C_3}")
    print()
    print("  Block structure (rows: target, cols: source):")
    print("    C_36 = [[0,        0,        U_{C_3}],")
    print("             [U_{C_3}, 0,        0      ],")
    print("             [0,       U_{C_3}, 0      ]]")

    Z = np.zeros((12, 12), dtype=complex)
    C_36 = np.block([
        [Z,     Z,     U_C3],
        [U_C3,  Z,     Z   ],
        [Z,     U_C3,  Z   ],
    ])
    assert C_36.shape == (36, 36)

    # Verify C_36 has order 3
    C_36_cubed = C_36 @ C_36 @ C_36
    order3_err = la.norm(C_36_cubed - np.eye(36))
    print(f"\n  ||C_36^3 - I_36|| = {order3_err:.3e}   (expected 0)")
    assert order3_err < 1e-10, f"C_36 does not have order 3: {order3_err}"

    # -----------------------------------------------------------------------
    # STEP 4: Verify C_36 commutes with block-diagonal B_total
    # -----------------------------------------------------------------------
    print("\n--- STEP 4: Verify equivariance [B_total, C_36] = 0 ---")

    B_total = np.block([
        [B_N1, Z,    Z   ],
        [Z,    B_N2, Z   ],
        [Z,    Z,    B_N3],
    ])
    assert B_total.shape == (36, 36)

    comm_err = commutator_norm(B_total, C_36)
    print(f"  ||[B_total, C_36]|| = {comm_err:.3e}   (expected 0)")
    assert comm_err < 1e-8, (
        f"C_36 does not commute with B_total: ||[B_total, C_36]|| = {comm_err:.3e}\n"
        f"This would indicate a bug in the C_36 construction."
    )
    print("  Equivariance verified: C_36 commutes with B_total.  OK")

    # Also compute character of C_36 and full isotypic dims on 36-dim space
    ch_36 = character_multiplicities(C_36)
    full_dims = (
        int(round(ch_36['m_1'].real)),
        int(round(ch_36['m_omega'].real)),
        int(round(ch_36['m_omega2'].real)),
    )
    print(f"\n  Character of C_36: chi(e,c,c^2) = "
          f"({ch_36['chi_e'].real:.0f}, {ch_36['chi_c'].real:+.4f}, "
          f"{ch_36['chi_c2'].real:+.4f})")
    print(f"  Isotypic dims (trivial, omega, omega_bar) on full 36-dim space: {full_dims}")
    print(f"  Expected: (12, 12, 12)  [from Frobenius reciprocity, same as generic k-orbit]")
    assert full_dims == (12, 12, 12), (
        f"Full 36-dim C_3 isotypic dims = {full_dims}, expected (12, 12, 12)"
    )

    # -----------------------------------------------------------------------
    # STEP 4b: Analytical character of C_Ram (predicted before restriction)
    # -----------------------------------------------------------------------
    print("\n--- STEP 4b: Analytical character prediction for C_Ram ---")
    print("  C_36 has zero diagonal 12x12 blocks (it cycles N1->N2->N3).")
    print("  W W^dag = block_diag(P_Ram(N1), P_Ram(N2), P_Ram(N3)) is block-diagonal.")
    print("  chi(C_36|V_Ram) = Tr(W^dag C_36 W) = Tr(C_36 W W^dag)")
    print("  = sum of traces of (off-diagonal blocks of C_36) * (diagonal blocks of WW^dag)")
    print("  = 0   because every nonzero block of C_36 is off-diagonal in the N-block structure.")
    print("  Similarly chi(C_36^2|V_Ram) = 0 (C_36^2 also has zero diagonal blocks).")
    print("  Therefore: m_1 = m_omega = m_omega_bar = (24 + 0 + 0)/3 = 8.")
    print("  Predicted: (8, 8, 8) — uniform distribution.")

    # -----------------------------------------------------------------------
    # STEP 5: Build the 36x24 embedding matrix W for the Ramanujan subspace
    # -----------------------------------------------------------------------
    print("\n--- STEP 5: Build Ramanujan subspace embedding W ---")
    print("  W embeds V_Ram(N1)+V_Ram(N2)+V_Ram(N3) into the 36-dim space:")
    print("  W = block_diag(V_Ram(N1), V_Ram(N2), V_Ram(N3)),  shape (36, 24)")

    Z12_8 = np.zeros((12, 8), dtype=complex)
    W = np.block([
        [V_Ram_N1, Z12_8,    Z12_8   ],
        [Z12_8,    V_Ram_N2, Z12_8   ],
        [Z12_8,    Z12_8,    V_Ram_N3],
    ])
    assert W.shape == (36, 24), f"W shape {W.shape}"

    # Verify W columns are orthonormal
    WdW = W.conj().T @ W
    orth_err = la.norm(WdW - np.eye(24))
    print(f"  ||W^dag W - I_24|| = {orth_err:.3e}   (orthonormality check)")
    assert orth_err < 1e-8, f"W is not orthonormal: {orth_err:.3e}"
    print("  W is orthonormal.  OK")

    # -----------------------------------------------------------------------
    # STEP 6: Restrict C_36 to the 24-dim Ramanujan subspace
    # -----------------------------------------------------------------------
    print("\n--- STEP 6: Restrict C_36 to V_Ram(combined) ---")
    print("  C_Ram = W^dag C_36 W   (24x24 matrix)")

    eigs_C_Ram, C_Ram = restricted_c3_eigs(C_36, W)

    print(f"  C_Ram shape: {C_Ram.shape}")

    # Verify C_Ram has order 3
    C_Ram_cubed = C_Ram @ C_Ram @ C_Ram
    order3_err_ram = la.norm(C_Ram_cubed - np.eye(24))
    print(f"  ||C_Ram^3 - I_24|| = {order3_err_ram:.3e}   (order-3 check)")
    assert order3_err_ram < 1e-8, (
        f"C_Ram does not have order 3: {order3_err_ram:.3e}\n"
        "This means the Ramanujan subspace is NOT closed under C_3 action."
    )
    print("  V_Ram(combined) is closed under the combined C_3 action.  OK")

    # -----------------------------------------------------------------------
    # STEP 7: Count C_3 isotypic multiplicities of C_Ram
    # -----------------------------------------------------------------------
    print("\n--- STEP 7: C_3 isotypic multiplicities of C_Ram ---")

    n_trivial, n_omega, n_omega_bar, n_other = classify_c3_eigs(eigs_C_Ram)
    total_classified = n_trivial + n_omega + n_omega_bar + n_other

    print(f"  Eigenvalues of C_Ram ({len(eigs_C_Ram)} total):")
    for ev in sorted(eigs_C_Ram, key=lambda z: np.angle(z)):
        diff_1  = abs(ev - 1.0)
        diff_w  = abs(ev - omega3)
        diff_w2 = abs(ev - omega3**2)
        tag = ("1" if diff_1 < 0.1 else
               "w" if diff_w < 0.1 else
               "w2" if diff_w2 < 0.1 else "?")
        print(f"    {ev.real:+.6f}{ev.imag:+.6f}i   |ev|={abs(ev):.6f}   [{tag}]")

    print(f"\n  C_3 isotypic multiplicities of V_Ram(N-orbit):")
    print(f"    n_trivial  (eig ~ 1):       {n_trivial}")
    print(f"    n_omega    (eig ~ omega):   {n_omega}")
    print(f"    n_omega_bar (eig ~ omega^2): {n_omega_bar}")
    print(f"    n_other    (unclassified):  {n_other}")
    print(f"    Total:                      {total_classified}  (expected 24)")

    assert total_classified == 24, f"Not all 24 eigenvalues classified: {n_other} unclassified"
    assert n_other == 0, (
        f"{n_other} eigenvalues unclassified as C_3 irreps; eigenvalues = {list(eigs_C_Ram)}"
    )

    # -----------------------------------------------------------------------
    # STEP 8: Compare to P-point result (4, 2, 2)
    # -----------------------------------------------------------------------
    print("\n--- STEP 8: Comparison to P-point result ---")

    # Build P-point Ramanujan subspace and C_3 multiplicities for reference
    B_P = bloch_hashimoto(P_PT, directed)
    _, V_Ram_P = extract_vram(B_P, expected_ram=8)
    eigs_C3_P, _ = restricted_c3_eigs(U_C3, V_Ram_P)
    p_trivial, p_omega, p_omega_bar, p_other = classify_c3_eigs(eigs_C3_P)

    print(f"  P-point: C_3 multiplicities on V_Ram(P) = "
          f"({p_trivial}, {p_omega}, {p_omega_bar})   [fiber action, trivial k-orbit]")
    assert p_other == 0, f"P-point: {p_other} unclassified eigenvalues"
    assert (p_trivial, p_omega, p_omega_bar) == (4, 2, 2), (
        f"P-point multiplicities {(p_trivial, p_omega, p_omega_bar)} != (4, 2, 2)"
    )
    print(f"  P-point: (4, 2, 2) confirmed.  OK")

    print(f"\n  N-orbit: C_3 multiplicities on V_Ram(combined) = "
          f"({n_trivial}, {n_omega}, {n_omega_bar})")
    print(f"  P-point: C_3 multiplicities on V_Ram(P)        = "
          f"({p_trivial}, {p_omega}, {p_omega_bar})")

    # Analytically predicted: (8, 8, 8) — assert it to lock in the result
    assert (n_trivial, n_omega, n_omega_bar) == (8, 8, 8), (
        f"N-orbit multiplicities {(n_trivial, n_omega, n_omega_bar)} != (8, 8, 8) "
        f"as analytically predicted from zero-trace argument."
    )

    # -----------------------------------------------------------------------
    # STEP 9: Additional diagnostic — fiber-only C_3 at N1 (wrong action)
    # -----------------------------------------------------------------------
    print("\n--- STEP 9: Diagnostic — fiber-only C_3 at N1 (off-axis, wrong action) ---")
    print("  NOTE: U_{C_3} does NOT commute with B(N1) since N1 is off-axis.")
    print("  This restricted C_3 is NOT the physical symmetry; it is diagnostic only.")

    fiber_comm_N1 = commutator_norm(B_N1, U_C3)
    print(f"  ||[B(N1), U_C3]|| = {fiber_comm_N1:.6e}   (expected NONZERO)")
    assert fiber_comm_N1 > 1e-6, "B(N1) accidentally commutes with U_C3"

    # Restrict U_C3 to V_Ram(N1) — this is not invariant in general
    # (since U_C3 maps the fiber at N1 to the fiber at N2, not N1)
    # but we compute it for diagnostic purposes
    U_C3_at_N1_restricted = V_Ram_N1.conj().T @ U_C3 @ V_Ram_N1   # 8x8
    eigs_fiber_N1 = la.eigvals(U_C3_at_N1_restricted)

    print(f"  U_C3 restricted to V_Ram(N1) eigenvalues (diagnostic only):")
    for ev in sorted(eigs_fiber_N1, key=lambda z: np.angle(z)):
        print(f"    {ev.real:+.6f}{ev.imag:+.6f}i   |ev|={abs(ev):.6f}")

    n_d1, n_dw, n_dw2, n_dother = classify_c3_eigs(eigs_fiber_N1, tol=0.15)
    print(f"  Fiber-only (diagnostic) multiplicities at N1: "
          f"(trivial={n_d1}, omega={n_dw}, omega_bar={n_dw2}, other={n_dother})")
    print("  [These are NOT physically meaningful: U_C3 is off-axis at N1]")

    # -----------------------------------------------------------------------
    # STEP 10: Structural diagnosis
    # -----------------------------------------------------------------------
    print("\n--- STEP 10: Structural diagnosis ---")
    print("=" * 72)

    result_N_orbit = (n_trivial, n_omega, n_omega_bar)
    result_P_point = (p_trivial, p_omega, p_omega_bar)

    print(f"\n  P-point (fiber C_3, trivial k-orbit): {result_P_point}")
    print(f"  N-orbit (combined C_3, 3-element k-orbit): {result_N_orbit}")

    # Interpret results
    if result_N_orbit == (4, 2, 2):
        print("\n  RESULT: N-orbit gives SAME multiplicities (4, 2, 2) as P-point.")
        print()
        print("  INTERPRETATION:")
        print("  The combined C_3 action on V_Ram(N1)+V_Ram(N2)+V_Ram(N3) decomposes")
        print("  as (4, 2, 2) — identical to the fiber C_3 at P-point.")
        print()
        print("  This is CONSISTENT with color-generation unification:")
        print("  Both the color-C_3 (fiber at P) and the generation-C_3 (k-orbit at N)")
        print("  are the same Z_3, and both give (4, 2, 2) multiplicities on the")
        print("  Ramanujan subspace.  The Koide formula prediction is therefore")
        print("  self-consistent: V_Ram has dimension 4+2+2=8 with the same isotypic")
        print("  structure at both P and N.")
        print()
        print("  HOWEVER: this is a necessary condition, not a sufficient one.")
        print("  The (4, 2, 2) match is consistent with the identification")
        print("  color-C_3 = generation-C_3, but does not PROVE that identification.")
        print("  The equivariant-bundle map from V_Ram(P) to V_Ram(N-orbit) (the")
        print("  'intertwiner') is not constructed here.  That construction requires")
        print("  resolving Block-KO.b (equivariant identification problem).")
        print()
        rigor_status = "FESHBACH-PATTERN"
    elif result_N_orbit == (8, 8, 8):
        print("\n  RESULT: N-orbit gives (8, 8, 8) — uniform distribution.")
        print()
        print("  ANALYTICAL EXPLANATION:")
        print("  This result follows from a simple character-theory argument:")
        print("  C_36 has zero diagonal 12x12 blocks (it cyclically permutes the")
        print("  three N-orbit fibers).  The Ramanujan projector W W^dag is block-")
        print("  diagonal.  Therefore:")
        print("    chi(C_Ram) = Tr(W^dag C_36 W) = Tr(C_36 W W^dag) = 0")
        print("  because the trace of a product of a block-off-diagonal matrix and a")
        print("  block-diagonal matrix is zero.  Similarly chi(C_Ram^2) = 0.")
        print("  By the C_3 character formula:")
        print("    m_trivial = m_omega = m_omega_bar = (24 + 0 + 0)/3 = 8.")
        print()
        print("  PHYSICAL INTERPRETATION:")
        print("  The combined C_3 action cyclically permutes the three N-orbit fibers.")
        print("  The Ramanujan subspace at each N-orbit point is NOT stabilized by")
        print("  U_{C_3} alone (since U_{C_3} maps fiber at N_i to fiber at N_{i+1}).")
        print("  The isotypic content is determined entirely by the k-orbit structure,")
        print("  which is a pure 3-cycle, giving equal (8,8,8) multiplicity for all")
        print("  three C_3 irreps regardless of the fiber content.")
        print()
        print("  STRUCTURAL IMPLICATION FOR KOIDE DERIVATION:")
        print("  The generation-C_3 acting on V_Ram(N-orbit) has UNIFORM (8,8,8)")
        print("  isotypic content — it does NOT reproduce the (4,2,2) structure of")
        print("  the color-C_3 at P-point.")
        print()
        print("  Therefore: the Koide formula's (4,2,2) isotypic structure derives from")
        print("  the FIBER C_3 at P-point, not from the k-orbit C_3 at the N-orbit.")
        print("  These are two DIFFERENT Z_3 actions with different isotypic content.")
        print()
        print("  CONSEQUENCE: The self-consistency test FAILS.")
        print("  The color-C_3 (fiber, (4,2,2)) and generation-C_3 (k-orbit, (8,8,8))")
        print("  are NOT the same Z_3 action on the Ramanujan subspace.")
        print()
        print("  BLOCKED: Color-generation unification via C_3-isotypic identification")
        print("  is blocked.  The (4,2,2) structure is intrinsic to the P-point fiber,")
        print("  and the N-orbit C_3 action does not reproduce it.")
        print()
        print("  Exact gap: A physical argument for why the generation C_3 should")
        print("  be the fiber C_3 (at P) rather than the k-orbit C_3 is needed.")
        print("  Alternatively, a different identification of the generation structure")
        print("  (not via the N-orbit k-orbit C_3) must be found.")
        rigor_status = "BLOCKED"
    elif result_N_orbit == (12, 6, 6):
        print("\n  RESULT: N-orbit gives (12, 6, 6) = 3 * (4, 2, 2).")
        print()
        print("  INTERPRETATION:")
        print("  The 24-dim combined Ramanujan space decomposes as (12, 6, 6).")
        print("  This is 3 times the P-point (4, 2, 2): each of the three N-orbit")
        print("  points contributes (4, 2, 2) independently to the combined decomposition.")
        print("  This is the 'induction' result: Ind_{trivial}^{C_3}((4,2,2)) = (12,6,6).")
        print()
        print("  Per-orbit-slice: (12, 6, 6)/3 = (4, 2, 2), consistent with P-point.")
        print("  This is CONSISTENT with color-generation unification (same per-slice")
        print("  structure), but the full 24-dim decomposition is (12, 6, 6) not (4,2,2).")
        rigor_status = "FESHBACH-PATTERN"
    else:
        print(f"\n  RESULT: N-orbit gives {result_N_orbit} — unexpected value.")
        print()
        print("  INTERPRETATION: The N-orbit C_3 multiplicities are neither (4,2,2),")
        print("  nor (8,8,8), nor (12,6,6).  This is an unexpected result requiring")
        print("  further analysis.  Report exact values and investigate.")
        rigor_status = "BLOCKED"

    print()
    print("=" * 72)
    print(f"RIGOR STATUS: {rigor_status}")
    print("=" * 72)

    if rigor_status == "FESHBACH-PATTERN":
        print()
        print("  Open gap (Block-KO.b): The equivariant identification of")
        print("  color-C_3 (fiber at P) with generation-C_3 (k-orbit at N) requires")
        print("  constructing an explicit intertwiner between V_Ram(P) and the")
        print("  isotypic components of V_Ram(N-orbit).  The (4,2,2) isotypic match")
        print("  is a necessary consistency check (now verified), but the identification")
        print("  map itself is not derived from A1+A2+A3 without additional input.")

    if rigor_status == "BLOCKED" and result_N_orbit != (8, 8, 8):
        print()
        print("  Exact gap: see specific BLOCKED message above.")

    print()
    print("NUMERICAL ASSERTIONS PASSED.")
    print(f"  N-orbit Ramanujan C_3 multiplicities: {result_N_orbit}")
    print(f"  P-point Ramanujan C_3 multiplicities: {result_P_point}")
    print("=" * 72)

    return result_N_orbit, result_P_point


if __name__ == "__main__":
    main()
