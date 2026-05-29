#!/usr/bin/env python3
"""
Layer 5 trace: how does the framework's Cl(6) chirality γ_7 relate to χ̃?

After tracing χ̃ from Layer 3 (walker B) through Layer 4 (observer's Hilbert
space), the next question is whether χ̃ is the SAME Z_2 grading as the
framework's existing Cl(6) chirality γ_7, or a NEW independent Z_2.

This probe makes the algebraic relationship explicit. The key insight from
reading `theorem_car_local_jordan_wigner.md`:

  - Cl(6) at vertex v acts on H_v = (C²)^⊗3 = 8-dim local Hilbert space.
  - γ_7 = γ_1 γ_2 ... γ_6 (product of all Majoranas at v).
  - Substituting JW Majoranas: γ_7 ∝ (1−2n_1)(1−2n_2)(1−2n_3) = (−1)^F
    where F = number of fermions at vertex v.
  - On the F=0 sector (vacuum at v): γ_7 = +1
  - On the F=1 sector (1 fermion at v): γ_7 = −1
  - On the F=2 sector: γ_7 = +1
  - On the F=3 sector: γ_7 = −1

The WALKER lives at "the vertex it's at" with F=1 at that vertex (one
particle = the walker). Other vertices have F=0. Total walker fermion
number F_total = 1 always.

So **γ_7 is constant (= −1) on the walker's Hilbert space.** It does NOT
provide a non-trivial Z_2 grading at the walker level. γ_7 ≠ χ̃ as
operators on the walker space.

The framework's earlier attempts to extract a non-trivial walker-level
γ_7 (`arg_h_path_b_p4_cl60_gamma5_attempt.py`,
 `arg_h_path_b_f0_gamma_attempt.py`) used CONSTRUCTIONS like
 γ_7^B := sign(Im(B|V_Ram)) — these are NOT the algebraic γ_7 from
Cl(6); they're heuristic sign operators that have gauge issues.

CONCLUSION (provisional, valid for SINGLE-VERTEX γ_7 only): the
single-vertex γ_7 acts trivially on the walker's F=1 sector and is
therefore not the source of χ̃ at the SINGLE-VERTEX level.

UPDATE 2026-05-01 EOD — see `srs_z_gamma7_lift_recovers_chi.py`: the
HALF-BIPARTITE PRODUCT γ_7^A := Π_{u∈A} γ_7_u acting on the walker
DOES recover ±χ̃ exactly. So χ̃ is NOT a new Z_2 — it is the framework's
existing Cl(6) chirality γ_7 lifted via srs-z's bipartite cover. The
"χ̃ ≠ γ_7 as operators on walker" line below holds for SINGLE-VERTEX
γ_7 (which is constant on F=1) but NOT for the half-bipartite product
γ_7^A. The framework has ONE Cl(6) Z_2, realized differently at
different layers depending on whether the substrate's primitive quotient
is bipartite.

This probe verifies the above claims numerically by:
  (1) Constructing γ_7 on a single-vertex Cl(6) 8-dim Hilbert space.
  (2) Verifying γ_7² = I and γ_7's eigenstructure (8 eigenvalues = ±1).
  (3) Verifying γ_7 ∝ (−1)^F by checking eigenvalue per fermion-number sector.
  (4) Showing the walker's 1-fermion sector has γ_7 = −1 uniformly.
  (5) Identifying additional Z_2 candidates at the walker level
      (complex conjugation = L/R photon polarization, time-reversal R̃
       = arc reversal) and checking their commutation with B and χ̃.
"""

import numpy as np
from numpy.linalg import eigvals, eigh
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges,
)
from srs_z_bipartite_involution_commutation import (
    build_adjacency, find_bipartition,
)


# =============================================================================
# Single-vertex Cl(6) construction (per theorem_car_local_jordan_wigner.md)
# =============================================================================

def cl6_single_vertex():
    """Build c_1, c_2, c_3, γ_1, ..., γ_6, γ_7 on H_v = (C²)^⊗3 = 8-dim."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # Tensor products
    def kron3(a, b, c):
        return np.kron(np.kron(a, b), c)

    # Jordan-Wigner construction with ordering 1, 2, 3
    # c_1 = σ⁻ ⊗ I ⊗ I
    # c_2 = σ_z ⊗ σ⁻ ⊗ I
    # c_3 = σ_z ⊗ σ_z ⊗ σ⁻
    c1 = kron3(sigma_minus, I2, I2)
    c2 = kron3(sigma_z, sigma_minus, I2)
    c3 = kron3(sigma_z, sigma_z, sigma_minus)

    # Majorana operators γ_{2j-1} = c_j + c_j†, γ_{2j} = i(c_j† - c_j)
    gamma_1 = c1 + c1.conj().T
    gamma_2 = 1j * (c1.conj().T - c1)
    gamma_3 = c2 + c2.conj().T
    gamma_4 = 1j * (c2.conj().T - c2)
    gamma_5 = c3 + c3.conj().T
    gamma_6 = 1j * (c3.conj().T - c3)

    # γ_7 = i · γ_1 γ_2 γ_3 γ_4 γ_5 γ_6 (the i factor makes γ_7 Hermitian
    # with γ_7² = +I, since (γ_1 γ_2 ... γ_6)² = -I per Cl(6) anticommutation)
    gamma_7 = 1j * (gamma_1 @ gamma_2 @ gamma_3 @ gamma_4 @ gamma_5 @ gamma_6)

    # Number operators: n_j = c_j† c_j; total F = n_1 + n_2 + n_3
    n1 = c1.conj().T @ c1
    n2 = c2.conj().T @ c2
    n3 = c3.conj().T @ c3
    F = n1 + n2 + n3

    return {
        'gamma': [gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6],
        'gamma_7': gamma_7,
        'F': F,
        'c': [c1, c2, c3],
    }


# =============================================================================
# Verification (parts 1-4 from docstring)
# =============================================================================

def part_1_to_4():
    print("=" * 80)
    print("Cl(6) construction at single vertex: γ_7 ≡ (−1)^F (fermion-number parity)")
    print("=" * 80)

    cl6 = cl6_single_vertex()
    gamma_7 = cl6['gamma_7']
    F = cl6['F']

    print(f"\nγ_7 dimension: {gamma_7.shape}")
    print(f"γ_7² = I check: {np.allclose(gamma_7 @ gamma_7, np.eye(8))}")

    # Eigenvalues of γ_7
    eigs_g7 = sorted(np.real(np.diag(gamma_7)))  # γ_7 should be diagonal in basis
    print(f"\nγ_7 in computational basis (diagonal entries):")
    print(f"  {eigs_g7}")
    print(f"  +1 count: {sum(1 for e in eigs_g7 if e > 0.5)}")
    print(f"  −1 count: {sum(1 for e in eigs_g7 if e < -0.5)}")

    # Eigenvalues of F (fermion number)
    eigs_F = sorted(np.real(np.diag(F)))
    print(f"\nF (fermion number) eigenvalues per basis state:")
    print(f"  {eigs_F}")

    # Verify γ_7 = (−1)^F (or possibly some sign convention)
    # Check: γ_7 = (-1)^F means γ_7 |F=k⟩ = (-1)^k |F=k⟩
    print(f"\nγ_7 vs (−1)^F per basis state:")
    print(f"  {'state':<10s} {'F':<5s} {'(-1)^F':<8s} {'γ_7':<8s}")
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    for s1 in [0, 1]:
        for s2 in [0, 1]:
            for s3 in [0, 1]:
                idx = s1 * 4 + s2 * 2 + s3
                F_val = s1 + s2 + s3
                sign_F = (-1)**F_val
                gamma_7_val = np.real(gamma_7[idx, idx])
                print(f"  |{s1}{s2}{s3}⟩      {F_val:<5d} {sign_F:<8d} {gamma_7_val:<8.0f}")

    # Walker lives at F=1 sector
    f1_states = [(s1, s2, s3) for s1 in [0,1] for s2 in [0,1] for s3 in [0,1]
                  if s1+s2+s3 == 1]
    # Compute γ_7 eigenvalue on F=1 subspace (depends on i-factor convention but is uniform)
    f1_indices = [s1 * 4 + s2 * 2 + s3 for (s1, s2, s3) in f1_states]
    gamma_7_on_F1 = [np.real(gamma_7[i, i]) for i in f1_indices]
    print(f"\nWalker's per-vertex F=1 subspace: {f1_states} ({len(f1_states)} states)")
    print(f"γ_7 eigenvalues on F=1 states: {gamma_7_on_F1} (uniform = {gamma_7_on_F1[0]:+.0f})")
    print(f"→ γ_7 acts as a CONSTANT ({gamma_7_on_F1[0]:+.0f}) on the walker's per-vertex F=1 subspace.")
    print(f"  It does NOT provide a non-trivial Z_2 grading at the walker level.")

    return cl6


def part_5_walker_level_z2_candidates():
    """Identify other Z_2 candidates at the walker level besides χ̃."""
    print("\n" + "=" * 80)
    print("Walker-level Z_2 candidates besides χ̃")
    print("=" * 80)

    entries = parse_rcsr_3dall('/tmp/rcsr_3d_current.txt', ['srs-z'])
    srs_z = entries['srs-z']
    rotations, translations, _, _ = get_space_group_ops('P4(1)32')
    v_frac = np.array(srs_z['vertex_orbits'][0]['cartesian'])
    m_frac = np.array(srs_z['edge_orbits'][0]['cartesian'])
    atom_orbit = orbit_of(v_frac, rotations, translations)
    midpoint_orbit = orbit_of(m_frac, rotations, translations)
    bonds = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=2)
    bonds = [b for b in bonds if b is not None]
    arcs = build_directed_edges(bonds)
    n_atoms = len(atom_orbit)
    n_arcs = len(arcs)

    A = build_adjacency(bonds, n_atoms)
    side_0, side_1 = find_bipartition(A)
    side_label = {v: +1 for v in side_0}
    side_label.update({v: -1 for v in side_1})
    chi = np.diag([side_label[a[0]] for a in arcs]).astype(complex)

    # Construct R̃ — arc reversal permutation
    # arc (tail, head, shift) → reverse (head, tail, -shift)
    arc_index = {(a[0], a[1], tuple(a[2])): i for i, a in enumerate(arcs)}
    R_perm = []
    for i, (tail, head, shift) in enumerate(arcs):
        rev_shift = tuple(-s for s in shift)
        rev_arc = (head, tail, rev_shift)
        R_perm.append(arc_index[rev_arc])
    R_tilde = np.zeros((n_arcs, n_arcs), dtype=complex)
    for i, j in enumerate(R_perm):
        R_tilde[i, j] = 1

    # R̃ properties
    print(f"\nR̃ (arc-reversal permutation):")
    print(f"  R̃² = I: {np.allclose(R_tilde @ R_tilde, np.eye(n_arcs))}")
    print(f"  R̃ Hermitian: {np.allclose(R_tilde, R_tilde.conj().T)}")

    # R̃ vs χ̃
    comm_R_chi = R_tilde @ chi - chi @ R_tilde
    anticomm_R_chi = R_tilde @ chi + chi @ R_tilde
    print(f"  ||[R̃, χ̃]||  = {np.linalg.norm(comm_R_chi):.4e}")
    print(f"  ||{{R̃, χ̃}}|| = {np.linalg.norm(anticomm_R_chi):.4e}")
    if np.linalg.norm(anticomm_R_chi) < 1e-10:
        print(f"  → R̃ and χ̃ ANTI-COMMUTE (algebraically distinct Z_2's)")

    # R̃ vs B at K-rational saddle
    k_R = np.array([0.5, 0.5, 0.5])
    B_R = bloch_hashimoto(arcs, k_R, n_atoms)
    R_B_R = R_tilde @ B_R
    B_R_R = B_R @ R_tilde
    diff_plus = np.linalg.norm(R_B_R - B_R_R)
    diff_minus = np.linalg.norm(R_B_R + B_R_R)
    print(f"\nR̃ vs B(k_R):")
    print(f"  ||R̃B − BR̃||   = {diff_plus:.4e}")
    print(f"  ||R̃B + BR̃||   = {diff_minus:.4e}")
    # Hashimoto identity J B J = B^T (for J = arc reversal)
    # So R̃ B R̃ = B^T. Test:
    R_B_R_R = R_tilde @ B_R @ R_tilde
    diff_BT = np.linalg.norm(R_B_R_R - B_R.T)
    print(f"  ||R̃ B R̃ − B^T|| = {diff_BT:.4e}  (Hashimoto identity J B J = B^T)")

    # Combined operator: χ̃ R̃
    chi_R = chi @ R_tilde
    print(f"\nχ̃ R̃ (composed Z_2 candidate):")
    print(f"  (χ̃ R̃)² = I: {np.allclose(chi_R @ chi_R, np.eye(n_arcs))}")
    chiR_B = chi_R @ B_R
    B_chiR = B_R @ chi_R
    print(f"  ||(χ̃ R̃) B − B (χ̃ R̃)||   = {np.linalg.norm(chiR_B - B_chiR):.4e}")
    print(f"  ||(χ̃ R̃) B + B (χ̃ R̃)||   = {np.linalg.norm(chiR_B + B_chiR):.4e}")

    print("\nSummary of walker-level Z_2 operators on srs-z:")
    print(f"  χ̃   (bipartite chirality, diagonal sign): {{χ̃, B}}=0 (anti-commutes everywhere)")
    print(f"  R̃   (arc reversal, permutation):        R̃ B R̃ = B^T (Hashimoto identity)")
    print(f"  χ̃R̃ (composed):                          relation to B is intermediate")
    print(f"  γ_7 (Cl(6) chirality, lifted to walker):  TRIVIAL (= −1) on F=1 sector")


def main():
    cl6 = part_1_to_4()
    part_5_walker_level_z2_candidates()

    print("\n" + "=" * 80)
    print("STRUCTURAL SYNTHESIS — γ_7 vs χ̃ at Layer 5")
    print("=" * 80)
    print("""
  The framework's existing Cl(6) chirality γ_7 = γ_1 γ_2 ... γ_6 acts on
  the per-vertex 8-dim Hilbert space H_v as the fermion-number parity
  operator (−1)^F. On the walker's 1-fermion sector (where the walker
  resides), γ_7 = −1 uniformly — it does NOT provide a Z_2 grading at
  the walker's level.

  χ̃ (bipartite chirality on directed arcs, only available on srs-z's
  bipartite Q_3 quotient) IS a non-trivial Z_2 grading at the walker
  level. χ̃ ≠ γ_7 as operators AT THE SINGLE-VERTEX level.

  CONCLUSION (provisional, see UPDATE in module docstring): χ̃ is a Z_2
  grading at walker level that the SINGLE-VERTEX γ_7 does NOT supply —
  the per-vertex γ_7 is constant on F=1, hence trivial as a walker
  operator. However, the HALF-BIPARTITE PRODUCT γ_7^A := Π_{u∈A} γ_7_u
  acting on the walker DOES recover ±χ̃ exactly (verified in
  `srs_z_gamma7_lift_recovers_chi.py`). So χ̃ is NOT a new Z_2 — it is
  the framework's existing Cl(6) chirality, lifted via srs-z's bipartite
  cover. ONE Z_2 across all layers, realized at the walker level only
  on substrates with bipartite primitive quotient.

  Three independent Z_2 structures so far identified at different layers:

    γ_7 (Cl(6) at vertex):  per-vertex fermion-number parity. Trivial on
                            walker (single fermion).
    R̃ (arc reversal):       walker-level. R̃ B R̃ = B^T (Hashimoto identity);
                            anti-commutes with χ̃.
    χ̃ (bipartite chirality): walker-level, only on bipartite substrates.
                            Anti-commutes with B(k) for all k. The clean
                            substrate-level SUSY-supercharge candidate.

  Implications:

  (a) The framework's primary substrate (srs, K_4 quotient) does NOT
      have a clean walker-level Z_2 supercharge. χ̃ doesn't exist on
      non-bipartite K_4.

  (b) On srs-z (bipartite Q_3 quotient), χ̃ provides the clean walker-
      level Z_2 supercharge. Combined with R̃ (which always exists),
      we have Z_2 × Z_2 structure at the walker level.

  (c) γ_7 (per-vertex) and χ̃ (per-arc-tail) live on different Hilbert
      spaces. They are NOT the same Z_2 — they are independent gradings
      on the framework's algebraic structure.

  (d) For SUSY-flavored interpretation: χ̃'s Z_2 IS the candidate for
      the SUSY supercharge at substrate level. The framework's
      (1-fermion sector) Cl(6) chirality γ_7 plays a different role:
      it labels the walker as "a single fermion" globally, not as a
      Z_2 grading on its state space.

  Open Layer-5 question for next session:

  Is there a NATURAL way to LIFT γ_7 from per-vertex to walker level
  that gives a non-trivial walker Z_2? The framework's existing
  attempts (γ_7^B = sign(Im(B|V_Ram)) etc.) had gauge-dependence issues.
  If a clean lift exists and matches χ̃, then γ_7 ≡ χ̃ at the lifted
  level. If not, χ̃ is genuinely independent.
""")


if __name__ == '__main__':
    main()
