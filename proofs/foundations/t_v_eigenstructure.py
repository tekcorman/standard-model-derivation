#!/usr/bin/env python3
"""
Sub-target F.1 — Visible-sector transition operator T_V at P.

FORMAL IDENTIFICATION (Step 1)
-------------------------------
Under the mass-as-flux framing (an internal working note §Definitions,
Transition operator), T_V := pi_MDL o T o iota_V is, restricted to the P-fibre
of the Bloch bundle, the Bloch Hashimoto operator B(P) acting on the 12-dim
directed-edge fibre.  The identification follows from:

  - ../../predictions/walker_dynamics_derivation.md §W3 + §W4: T projected onto the visible
    sector via pi_MDL IS the Hashimoto B (as the 1-step amplitude operator on
    directed edges of the srs lattice).
    "T projected onto V via pi_MDL is, in the framework's current incarnation,
    the Hashimoto operator B (per ../../predictions/walker_dynamics_derivation.md Step 6, with
    B the 1-step NB transition on directed edges)."

WHAT THIS SCRIPT VERIFIES (Steps 2-4)
---------------------------------------
1. Construct B(P) from the srs primitive-cell bond list (inheriting from
   proofs/foundations/theorem_B5_3_core.py infrastructure).
2. Construct U_{C_3}, the 12x12 permutation matrix for the body-diagonal C_3.
3. Verify [B(P), U_{C_3}] = 0 (commutation on the C_3-fixed P-point).
4. Project U_{C_3} onto the 8-dim Ramanujan subspace V_Ram of B(P)
   (eigenvalues {h, h*, -h, -h*}, each with multiplicity 2).
5. Compute the C_3-isotypic decomposition of V_Ram, confirming (4, 2, 2):
   - Channel 0 (trivial,   multiplicity 4): U_{C_3} acts as identity.
   - Channel 1 (omega,     multiplicity 2): U_{C_3} acts as omega = e^{2pi i/3}.
   - Channel 2 (omega^2,   multiplicity 2): U_{C_3} acts as omega^2.
6. Confirm C_3-isotypic decomposition of V_tree (eigenvalues {+1, -1},
   each mult 2) is (0, 2, 2).
7. Assert the full 12-dim decomposition is (4, 2, 2) + (0, 2, 2) = (4, 4, 4).

RIGOR STATUS
------------
  - The T_V = B(P) identification is adopted at the flux-operator framing level
    (an internal working note), following W3 of
    ../../predictions/walker_dynamics_derivation.md.  Status: framing-level adoption (not
    independently derived here; the derivation chain is W1-W3 + pi_MDL
    definition in the framing doc).
  - The spectral content (4, 2, 2) on V_Ram is STRICT-SOLID: it reuses the
    already-verified B(P) spectral data of ../../predictions/B_P_doubly_degenerate_h_derivation.md
    and docs/theorem_B5_3_core.md via the same numerical infrastructure.
  - The consistency check (Step 4 below) is a new verification in the flux
    language; it uses no new axioms beyond what theorem_B5_3_core.py already
    establishes.

UPSTREAM FILES (no new axioms needed)
--------------------------------------
  - docs/framework/framework_axioms.md (A1, A2)
  - ../../predictions/walker_dynamics_derivation.md W1-W3 (T_V = B at the visible-sector level)
  - ../../predictions/B_P_doubly_degenerate_h_derivation.md (B(P) spectrum, C_3-protection)
  - docs/theorem_B5_3_core.md (C_3-equivariant decomposition, (4,2,2) on V_Ram)
  - proofs/foundations/theorem_B5_3_core.py (infrastructure reused here)
  - proofs/common.py (find_bonds, C3_PERM, omega3)

Prints "OK: T_V eigenstructure verified — (4, 2, 2) on V_Ram confirmed." on success.
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, C3_PERM, omega3

# Reuse the directed-edge and Hashimoto infrastructure from theorem_B5_3_core.py.
# We import the functions by duplicating the minimal subset here to keep this
# script self-contained for auditability; they are identical in logic to
# theorem_B5_3_core.py.

H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2   # h = (sqrt(3)+i sqrt(5))/2
K_P = (0.25, 0.25, 0.25)                             # P = (1/4,1/4,1/4) in reduced coords


# -----------------------------------------------------------------------
# Infrastructure: directed edges, Bloch Hashimoto B(k), U_{C_3}
# -----------------------------------------------------------------------

def build_directed_edges(bonds):
    """Return list of (src, tgt, cell) tuples for all 12 directed edges."""
    directed = [tuple(b) for b in bonds]
    assert len(directed) == 12, f"expected 12 directed edges, got {len(directed)}"
    return directed


def bloch_hashimoto(k_frac, directed):
    """12x12 Bloch Hashimoto B(k) on directed edges.

    B(k)[e', e] = exp(2*pi*i * k . cell_{e'})  if e -> e' is a valid NB step,
                  0                             otherwise.

    Valid NB: target(e) = source(e') and e' != reverse(e).
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
    """Read sigma = (v_0)(v_1 v_3 v_2) from common.C3_PERM."""
    perm = {}
    for i in range(4):
        for j in range(4):
            if abs(C3_PERM[i, j] - 1.0) < 1e-12:
                perm[j] = i
    assert perm == {0: 0, 1: 3, 2: 1, 3: 2}, f"unexpected sigma: {perm}"
    return perm


def c3_cell_perm(cell):
    """C_3 on primitive-cell displacement: (n1, n2, n3) -> (n3, n1, n2)."""
    return (cell[2], cell[0], cell[1])


def build_c3_on_directed_edges(directed):
    """12x12 permutation matrix U_{C_3} for the C_3 action on directed edges."""
    vp = c3_vertex_perm()
    n = len(directed)
    edge_to_idx = {de: i for i, de in enumerate(directed)}
    U = np.zeros((n, n), dtype=complex)
    for i, (src, tgt, cell) in enumerate(directed):
        new_edge = (vp[src], vp[tgt], c3_cell_perm(cell))
        j = edge_to_idx.get(new_edge)
        if j is None:
            raise RuntimeError(
                f"C_3 mapped {(src, tgt, cell)} -> {new_edge}, not in directed set"
            )
        U[j, i] = 1.0
    return U


# -----------------------------------------------------------------------
# Isotypic decomposition helper
# -----------------------------------------------------------------------

def c3_isotypic_dims(U, tol=0.1):
    """Given a 12x12 (or smaller) unitary with eigenvalues in {1, omega, omega^2},
    count the multiplicities of each C_3 irrep.

    Returns (m_trivial, m_omega, m_omega2) as integers.
    """
    evals = la.eigvals(U)
    m1, mw, mw2 = 0, 0, 0
    for ev in evals:
        if abs(ev - 1.0) < tol:
            m1 += 1
        elif abs(ev - omega3) < tol:
            mw += 1
        elif abs(ev - omega3 ** 2) < tol:
            mw2 += 1
        else:
            raise ValueError(
                f"U eigenvalue {ev} is not within tol={tol} of 1, omega, or omega^2"
            )
    return (m1, mw, mw2)


def project_U_onto_subspace(U, evecs_subspace):
    """Project U onto a subspace spanned by evecs_subspace (n x m matrix),
    orthonormalize, and return the m x m restricted matrix.

    Caller is responsible for ensuring [B, U] = 0 on the full space so that
    the projection is well-defined.
    """
    Q, _ = la.qr(evecs_subspace)
    return Q.conj().T @ U @ Q


# -----------------------------------------------------------------------
# Main verification
# -----------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Sub-target F.1 — T_V eigenstructure at P-point")
    print("Verifying C_3-isotypic decomposition of T_V = B(P) on V_Ram: (4,2,2)")
    print("=" * 72)

    # --- Build directed edges and U_{C_3} -------------------------------
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    U = build_c3_on_directed_edges(directed)

    # --- Step 1 (formal identification, printed) -----------------------
    print()
    print("Step 1 — Formal identification: T_V = B(P)")
    print("  Under an internal working note §Definitions (Transition")
    print("  operator) and ../../predictions/walker_dynamics_derivation.md W3:")
    print("  T_V := pi_MDL o T o iota_V, restricted to the P-fibre of B, equals")
    print("  the Bloch Hashimoto operator B(P) on the 12-dim directed-edge fibre.")
    print("  This identification is the load-bearing step of Sub-target F.1;")
    print("  it is adopted at the flux-framing level (not independently derived")
    print("  here). The spectral content verified below is STRICT-SOLID,")
    print("  reusing the already-verified B(P) data of")
    print("  ../../predictions/B_P_doubly_degenerate_h_derivation.md and docs/theorem_B5_3_core.md.")

    # --- Step 2 — Build B(P) -------------------------------------------
    print()
    print("Step 2 — Construct B(P) (12x12 Bloch Hashimoto at k = P)")
    B_P = bloch_hashimoto(K_P, directed)
    assert B_P.shape == (12, 12), f"B(P) shape {B_P.shape}"

    # Verify U has order 3
    res_U3 = la.norm(U @ U @ U - np.eye(12))
    print(f"  ||U_{{C_3}}^3 - I|| = {res_U3:.2e}   (expected 0)")
    assert res_U3 < 1e-10, f"U is not order 3: {res_U3}"

    # --- Step 3 — Commutation [B(P), U_{C_3}] = 0 at P ----------------
    print()
    print("Step 3 — Verify [B(P), U_{{C_3}}] = 0 at k = P (C_3 fixes P)")
    comm_norm = la.norm(B_P @ U - U @ B_P)
    print(f"  ||[B(P), U_{{C_3}}]|| = {comm_norm:.2e}   (expected 0, C_3 fixed axis)")
    assert comm_norm < 1e-10, f"Commutator nonzero at P: {comm_norm}"

    # --- Step 4 — Eigendecompose B(P), classify into V_Ram and V_tree --
    print()
    print("Step 4 — Eigendecompose B(P); classify into V_Ram and V_tree")
    evals_B, evecs_B = la.eig(B_P)

    # Identify Ramanujan eigenvalues (|mu|^2 = k-1 = 2)
    ram_idx = [i for i, ev in enumerate(evals_B) if abs(abs(ev) ** 2 - 2.0) < 1e-6]
    tree_idx = [i for i, ev in enumerate(evals_B) if abs(abs(ev) ** 2 - 1.0) < 1e-6]
    other_idx = [i for i in range(12) if i not in ram_idx and i not in tree_idx]

    print(f"  Ramanujan eigenvalues (|mu|^2=2): {len(ram_idx)} eigenvectors")
    print(f"  Tree eigenvalues      (|mu|^2=1): {len(tree_idx)} eigenvectors")
    assert len(ram_idx) == 8, f"V_Ram dim should be 8, got {len(ram_idx)}"
    assert len(tree_idx) == 4, f"V_tree dim should be 4, got {len(tree_idx)}"
    assert len(other_idx) == 0, f"Unexpected eigenvalues: {[evals_B[i] for i in other_idx]}"

    # Verify the Ramanujan eigenvalues are {h, h*, -h, -h*} each with mult 2
    h_targets = [H_EXACT, H_EXACT.conjugate(), -H_EXACT, -H_EXACT.conjugate()]
    evals_ram = sorted(evals_B[ram_idx], key=lambda z: (round(z.real, 6), round(z.imag, 6)))
    for mu in evals_B[ram_idx]:
        assert any(abs(mu - t) < 1e-6 for t in h_targets), (
            f"Ramanujan eigenvalue {mu} not in {{h, h*, -h, -h*}}"
        )
    count_h = sum(1 for mu in evals_B[ram_idx] if abs(mu - H_EXACT) < 1e-6)
    count_hs = sum(1 for mu in evals_B[ram_idx] if abs(mu - H_EXACT.conjugate()) < 1e-6)
    print(f"  h = {H_EXACT.real:.4f}+{H_EXACT.imag:.4f}i  appears {count_h} times in V_Ram  (expected 2)")
    print(f"  h* = {H_EXACT.real:.4f}-{H_EXACT.imag:.4f}i appears {count_hs} times in V_Ram  (expected 2)")
    assert count_h == 2, f"h multiplicity {count_h}"
    assert count_hs == 2, f"h* multiplicity {count_hs}"

    # Verify tree eigenvalues are {+1, -1} each with mult 2
    for mu in evals_B[tree_idx]:
        assert abs(mu - 1.0) < 1e-6 or abs(mu + 1.0) < 1e-6, (
            f"Tree eigenvalue {mu} not +-1"
        )
    count_p1 = sum(1 for mu in evals_B[tree_idx] if abs(mu - 1.0) < 1e-6)
    count_m1 = sum(1 for mu in evals_B[tree_idx] if abs(mu + 1.0) < 1e-6)
    print(f"  +1 appears {count_p1} times in V_tree  (expected 2)")
    print(f"  -1 appears {count_m1} times in V_tree  (expected 2)")
    assert count_p1 == 2, f"+1 multiplicity {count_p1}"
    assert count_m1 == 2, f"-1 multiplicity {count_m1}"

    # --- Step 5 — Project U_{C_3} onto V_Ram; compute C_3-isotypic dims -
    print()
    print("Step 5 — C_3-isotypic decomposition of V_Ram (8-dim Ramanujan subspace)")
    evecs_ram = evecs_B[:, ram_idx]     # 12 x 8
    U_ram = project_U_onto_subspace(U, evecs_ram)   # 8 x 8 restricted
    dims_ram = c3_isotypic_dims(U_ram)
    print(f"  Isotypic dims (trivial, omega, omega^2) on V_Ram = {dims_ram}")
    print(f"  Expected: (4, 2, 2)")
    assert dims_ram == (4, 2, 2), (
        f"V_Ram C_3-isotypic decomposition is {dims_ram}, expected (4, 2, 2)"
    )

    # Interpret in flux language
    print()
    print("  Flux-channel interpretation (an internal working note §F.1):")
    print("  Channel 0 (trivial sector):  multiplicity 4")
    print("    -> 4 independent rewrite pathways on which U_{C_3} acts as identity.")
    print("  Channel 1 (omega sector):    multiplicity 2")
    print("    -> 2 independent rewrite pathways on which U_{C_3} acts as omega.")
    print("  Channel 2 (omega^2 sector):  multiplicity 2")
    print("    -> 2 independent rewrite pathways on which U_{C_3} acts as omega^2.")

    # --- Step 6 — C_3-isotypic dims on V_tree --------------------------
    print()
    print("Step 6 — C_3-isotypic decomposition of V_tree (4-dim tree subspace)")
    evecs_tree = evecs_B[:, tree_idx]   # 12 x 4
    U_tree = project_U_onto_subspace(U, evecs_tree)  # 4 x 4 restricted
    dims_tree = c3_isotypic_dims(U_tree)
    print(f"  Isotypic dims (trivial, omega, omega^2) on V_tree = {dims_tree}")
    print(f"  Expected: (0, 2, 2)")
    assert dims_tree == (0, 2, 2), (
        f"V_tree C_3-isotypic decomposition is {dims_tree}, expected (0, 2, 2)"
    )

    # --- Step 7 — Total consistency check ------------------------------
    print()
    print("Step 7 — Total consistency (V_Ram + V_tree = full 12-dim fibre)")
    total = tuple(dims_ram[i] + dims_tree[i] for i in range(3))
    print(f"  Total (V_Ram + V_tree): {total}   (expected (4, 4, 4))")
    assert total == (4, 4, 4), f"Total isotypic dims {total}"

    # Cross-check against full-space character (should agree with Step 2 of
    # theorem_B5_3_core.py: chi(e, c, c^2) = (12, 0, 0) => (4, 4, 4)).
    chi_e = np.trace(np.eye(12)).real
    chi_c = np.trace(U).real
    chi_c2 = np.trace(U @ U).real
    m1_full = round((chi_e + chi_c + chi_c2) / 3)
    mw_full = round(abs((chi_e + np.conj(omega3) * chi_c + np.conj(omega3) ** 2 * chi_c2) / 3))
    mw2_full = round(abs((chi_e + np.conj(omega3) ** 2 * chi_c + np.conj(omega3) * chi_c2) / 3))
    print(f"  Full-space character: chi(e, c, c^2) = ({chi_e:.0f}, {chi_c:.4f}, {chi_c2:.4f})")
    print(f"  Full-space multiplicities: ({m1_full}, {mw_full}, {mw2_full})")
    assert (m1_full, mw_full, mw2_full) == (4, 4, 4), (
        f"Full-space multiplicities {(m1_full, mw_full, mw2_full)}"
    )

    # --- Summary -------------------------------------------------------
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print("Formal identification (adopted at framing level, not derived here):")
    print("  T_V := pi_MDL o T o iota_V  at k = P  =  B(P)   [12x12 Bloch Hashimoto]")
    print("  Source: ../../predictions/walker_dynamics_derivation.md W3,")
    print("          an internal working note §Definitions.")
    print()
    print("Spectral content of T_V at P (STRICT-SOLID: reuses verified B(P) data):")
    print("  Eigenvalues: h = (sqrt(3)+i sqrt(5))/2 (mult 2),  h* (mult 2),")
    print("               -h (mult 2),  -h* (mult 2),  +1 (mult 2),  -1 (mult 2).")
    print()
    print("C_3-isotypic decomposition (STRICT-SOLID: sympy-verified via B5.3-core):")
    print("  V_Ram (8-dim Ramanujan subspace):  (trivial, omega, omega^2) = (4, 2, 2)")
    print("  V_tree (4-dim tree subspace):       (trivial, omega, omega^2) = (0, 2, 2)")
    print("  Full 12-dim fibre:                 (trivial, omega, omega^2) = (4, 4, 4)")
    print()
    print("Flux-channel interpretation (F.1 sub-target):")
    print("  T_V at P has three flux channels (trivial, omega, omega^2) with")
    print("  multiplicities (4, 2, 2) on V_Ram.  Under the mass-as-flux framing,")
    print("  these are the three sectors through which the substrate exchanges")
    print("  dark <-> visible content at the P-point.  The multiplicity of each")
    print("  channel is the number of independent rewrite pathways through that")
    print("  sector.")
    print()
    print("OK: T_V eigenstructure verified — (4, 2, 2) on V_Ram confirmed.")


if __name__ == "__main__":
    main()
