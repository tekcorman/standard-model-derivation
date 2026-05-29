#!/usr/bin/env python3
"""
Layer 6+ trace: χ̃ × C_3 generation grading × Pati-Salam embedding.

The framework's 3 generations of SM fermions emerge from C_3 irrep
splitting (1, ω, ω²) of the threefold rotation at srs's K-rational saddle
(per `R3_observer_c3_generation.py`). At the saddle, the Hashimoto B(k)
eigenstates decompose into C_3-irreducible sectors corresponding to the
3 generations.

This probe asks: how does χ̃ interact with the C_3 generation grading on
srs-z?

  (i) Does χ̃ commute with the C_3 action on srs-z's walker?
      • If yes: independent gradings, structure is Z_3 × Z_2 = Z_6.
        3 generations × 2 supercharge-sectors = 6 sectors of fermions.
        Algebraically the right structure for "3 generations of SM
        particles + 3 generations of SUSY partners."
      • If no: gradings interact non-trivially, structure is more
        subtle.

  (ii) Pati-Salam ⊂ Spin(6) emerges from k=3 + local CAR. Same on srs
       and srs-z (k=3 both). χ̃ acts on walker space (where Pati-Salam
       reps live). Does χ̃ commute with the Pati-Salam action?

This is structural exploration — finding what algebraic constraints χ̃
imposes on the framework's higher-layer structures. NOT a derivation of
specific SUSY-partner masses or couplings.
"""

import sys
import os
import numpy as np
from numpy.linalg import eigvals
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges, identify_irrational
)
from srs_z_bipartite_involution_commutation import (
    build_adjacency, find_bipartition,
    space_group_perms_on_atoms, lift_perm_to_arcs, perm_to_matrix,
)


def get_data():
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
    return {
        'atom_orbit': atom_orbit,
        'bonds': bonds,
        'arcs': arcs,
        'rotations': rotations,
        'translations': translations,
    }


def find_C3_along_111(rotations, translations):
    """Find a P4_132 op corresponding to C_3 along the (1,1,1) body diagonal.
    Such op has rotation R = [[0,0,1],[1,0,0],[0,1,0]] (cyclic xyz → zxy).
    """
    R_target = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for i, (R, t) in enumerate(zip(rotations, translations)):
        if np.array_equal(R, R_target):
            return i, R, t
    # Try the inverse (z, x, y → y, z, x)
    R_target_inv = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    for i, (R, t) in enumerate(zip(rotations, translations)):
        if np.array_equal(R, R_target_inv):
            return i, R, t
    return None, None, None


def main():
    print("=" * 80)
    print("Layer 6+ trace: χ̃ × C_3 (generation grading) × Pati-Salam ⊂ Spin(6)")
    print("=" * 80)

    data = get_data()
    atom_orbit = data['atom_orbit']
    bonds = data['bonds']
    arcs = data['arcs']
    rotations = data['rotations']
    translations = data['translations']
    n_atoms = len(atom_orbit)
    n_arcs = len(arcs)

    # Build χ̃
    A = build_adjacency(bonds, n_atoms)
    side_0, side_1 = find_bipartition(A)
    side_label = {v: +1 for v in side_0}
    side_label.update({v: -1 for v in side_1})
    chi = np.diag([side_label[a[0]] for a in arcs]).astype(complex)

    # Find C_3 op
    print("\n" + "-" * 80)
    print("Step 1: Find C_3 along (1,1,1) in P4_132 space group ops")
    print("-" * 80)
    op_idx, R_C3, t_C3 = find_C3_along_111(rotations, translations)
    if op_idx is None:
        print("  C_3 along (1,1,1) not found in spglib's P4_132 ops.")
        return
    print(f"  Found C_3 op #{op_idx}:")
    print(f"    R = {R_C3.tolist()}")
    print(f"    t = {t_C3.tolist()}")
    print(f"    Acts on coords: (x,y,z) → ({R_C3[0,0]}x+{R_C3[0,1]}y+{R_C3[0,2]}z+{t_C3[0]:.2f}, ...)")

    # Compute C_3 vertex permutation on srs-z atoms
    sg_perms = space_group_perms_on_atoms(atom_orbit, rotations, translations)
    C3_perm = None
    for perm, idx in sg_perms:
        if idx == op_idx:
            C3_perm = perm
            break
    print(f"\n  C_3 permutation on 8 atoms: {C3_perm}")

    # Verify C_3³ = identity
    C3_perm_arr = list(C3_perm)
    C3_2 = [C3_perm_arr[C3_perm_arr[i]] for i in range(8)]
    C3_3 = [C3_perm_arr[C3_2[i]] for i in range(8)]
    is_C3 = (C3_3 == list(range(8)))
    print(f"  C_3³ = identity check: {is_C3}")

    # Lift C_3 to arcs
    print("\n" + "-" * 80)
    print("Step 2: Lift C_3 to directed-arc space (24-dim)")
    print("-" * 80)
    arc_perm_C3 = lift_perm_to_arcs(C3_perm, arcs, atom_orbit, R_C3, t_C3)
    n_unmapped = sum(1 for x in arc_perm_C3 if x is None)
    if n_unmapped > 0:
        print(f"  WARNING: {n_unmapped}/{n_arcs} arcs unmapped — try other op or augmented lift")
        return
    C3_tilde = perm_to_matrix(arc_perm_C3, n_arcs)
    print(f"  C_3 lifted: 24×24 permutation matrix")
    C3_3_tilde = C3_tilde @ C3_tilde @ C3_tilde
    print(f"  C̃_3³ = I check: {np.allclose(C3_3_tilde, np.eye(n_arcs))}")

    # Step 3: χ̃ commutation with C_3
    print("\n" + "-" * 80)
    print("Step 3: Does χ̃ commute with C_3 on the walker?")
    print("-" * 80)
    chi_C3 = chi @ C3_tilde
    C3_chi = C3_tilde @ chi
    diff = np.linalg.norm(chi_C3 - C3_chi)
    print(f"  ||χ̃·C_3 − C_3·χ̃|| = {diff:.4e}")
    if diff < 1e-10:
        print(f"  → χ̃ and C_3 COMMUTE.")
        print(f"    Structure is Z_2 × Z_3 = Z_6 — six sectors at the (χ̃, C_3-irrep) level.")
        print(f"    Algebraically: 3 generations × 2 supercharge-sectors = 6 sectors of fermions.")
    else:
        print(f"  → χ̃ and C_3 do NOT commute. Joint structure is non-trivial.")

    # Step 4: B(k_R) commutation with C_3
    print("\n" + "-" * 80)
    print("Step 4: Does C_3 commute with B(k_R)?")
    print("-" * 80)
    k_R = np.array([0.5, 0.5, 0.5])
    B_R = bloch_hashimoto(arcs, k_R, n_atoms)
    comm_BC3 = C3_tilde @ B_R - B_R @ C3_tilde
    print(f"  ||[C_3, B(k_R)]|| / ||B|| = {np.linalg.norm(comm_BC3) / np.linalg.norm(B_R):.4e}")
    if np.linalg.norm(comm_BC3) / np.linalg.norm(B_R) < 1e-10:
        print(f"  → C_3 commutes with B(k_R) at the K-rational saddle.")
        print(f"    Saddle eigenstates carry C_3-irrep labels: trivial (1) ⊕ ω ⊕ ω².")
    else:
        print(f"  → C_3 does NOT commute with B(k_R). C_3 might map k_R to a different k.")
        # Check: does R · k_R = k_R mod reciprocal lattice?
        R_k_R = R_C3 @ k_R
        diff_k = (R_k_R - k_R + 0.5) % 1.0 - 0.5
        print(f"    R_C3 · k_R = {R_k_R.tolist()},  difference from k_R mod 1: {diff_k.tolist()}")

    # Step 5: Joint χ̃ × C_3 sector decomposition
    print("\n" + "-" * 80)
    print("Step 5: Joint χ̃ × C_3 sector decomposition at k=R")
    print("-" * 80)

    if np.linalg.norm(chi_C3 - C3_chi) < 1e-10 and np.linalg.norm(comm_BC3) / np.linalg.norm(B_R) < 1e-10:
        # Diagonalize C_3 (eigenvalues 1, ω, ω²)
        # Since C_3³ = I, eigenvalues are cube roots of 1
        omega = np.exp(2j * np.pi / 3)
        omega_sq = np.exp(-2j * np.pi / 3)

        # Compute C_3 eigenvalues
        C3_eigs = eigvals(C3_tilde)
        C3_eig_counts = Counter([round(np.angle(e) / np.pi * 3, 0) for e in C3_eigs])
        print(f"  C_3 eigenvalue angles (in units of π/3): {dict(C3_eig_counts)}")
        n_trivial = sum(1 for e in C3_eigs if abs(e - 1) < 1e-6)
        n_omega = sum(1 for e in C3_eigs if abs(e - omega) < 1e-6)
        n_omega_sq = sum(1 for e in C3_eigs if abs(e - omega_sq) < 1e-6)
        print(f"  C_3 = +1 (trivial irrep):  {n_trivial}-dim sector")
        print(f"  C_3 = ω:                    {n_omega}-dim sector")
        print(f"  C_3 = ω²:                   {n_omega_sq}-dim sector")

        # Each C_3 sector splits by χ̃ (since they commute)
        # Compute simultaneous eigenspaces
        print(f"\n  Joint Z_2 × Z_3 = Z_6 sector dimensions + B(k_R) spectra:")
        # Build C_3 eigenvalue spectral projectors via P_λ = (1/3) Σ_j λ̄^j C_3^j
        I_n = np.eye(n_arcs, dtype=complex)
        C3_pow = [I_n, C3_tilde, C3_tilde @ C3_tilde]
        proj_1 = (C3_pow[0] + C3_pow[1] + C3_pow[2]) / 3.0
        proj_omega = (C3_pow[0] + omega.conjugate() * C3_pow[1] + omega_sq.conjugate() * C3_pow[2]) / 3.0
        proj_omega_sq = (C3_pow[0] + omega_sq.conjugate() * C3_pow[1] + omega.conjugate() * C3_pow[2]) / 3.0
        proj_chi_plus = (I_n + chi) / 2.0
        proj_chi_minus = (I_n - chi) / 2.0

        sector_data = []
        for c3_proj, c3_label in [(proj_1, "1"), (proj_omega, "ω"), (proj_omega_sq, "ω²")]:
            for chi_proj, chi_label in [(proj_chi_plus, "+"), (proj_chi_minus, "−")]:
                P_joint = c3_proj @ chi_proj
                # Trace gives dimension of the joint eigenspace
                dim = int(np.round(np.real(np.trace(P_joint))))
                # B(k_R) acts on this joint sector
                # eigenvalues of B(k_R) within it = eigenvalues of P_joint @ B_R @ P_joint, excluding zeros
                B_proj = P_joint @ B_R @ P_joint
                eigs_proj = eigvals(B_proj)
                # filter out zeros (outside this sector)
                eigs_in = sorted([e for e in eigs_proj if abs(e) > 1e-6],
                                 key=lambda x: (round(np.real(x), 3), round(np.imag(x), 3)))
                sector_data.append((c3_label, chi_label, dim, eigs_in))
                # Sample
                eig_samples = ", ".join(f"{e.real:+.3f}{e.imag:+.3f}i" for e in eigs_in[:3])
                if len(eigs_in) > 3: eig_samples += ", ..."
                print(f"    (C_3 = {c3_label}, χ̃ = {chi_label}):  dim = {dim}, "
                      f"B-eigs sample: {eig_samples}")

        # Verify dim sum
        total_dim = sum(d for _, _, d, _ in sector_data)
        print(f"\n  Total dim across 6 sectors: {total_dim} (should be 24)")

    # Step 6: comparison with srs (C_3 acts on K_4 too)
    print("\n" + "=" * 80)
    print("Step 6: Comparison with srs (no χ̃, but C_3 still gives 3-generation grading)")
    print("=" * 80)
    print("""
  srs's K_4 quotient also has C_3 symmetry (cycle 3 of the 4 vertices), giving
  the same trivial ⊕ ω ⊕ ω² decomposition at k_P. This is the framework's
  3-generation structure on srs.

  On srs-z, the SAME C_3-irrep decomposition works (since C_3 is a P4_132 op,
  and P4_132 contains C_3 along (1,1,1) just like I4_132 does). The 3
  generation structure transfers unchanged.

  What's NEW on srs-z is the χ̃ Z_2 grading layered ON TOP of the C_3
  generations. Each generation gets a +/- supercharge label.

  This algebraically realizes the structure of "supersymmetric SM" with
  3 generations × 2 supercharge sectors. NOT a derivation of MSSM masses;
  the structural identification of WHERE in the framework's algebra the
  Z_2 × Z_3 grading sits.

  Mass spectrum implication:
    On srs-z's K-rational saddle, the 24-dim B(k_R) splits as:
      • 3 C_3-irreps × 2 χ̃-sectors × (multiplicities from cell size)
    Each (irrep, χ̃) pair has the SAME |λ|² (from B² being χ̃-even).
    → SUSY-partner masses degenerate with their SM counterparts at the
      ALGEBRAIC level. Realistic mass splitting needs χ̃-symmetry-breaking.
""")

    # Step 7: Pati-Salam interaction
    print("=" * 80)
    print("Step 7: Pati-Salam ⊂ Spin(6) interaction with χ̃ — open question")
    print("=" * 80)
    print("""
  Pati-Salam SU(4) × SU(2)_L × SU(2)_R emerges from k=3 + Cl(2k=6) per
  `theorem_car_local_jordan_wigner.md`. It is INVARIANT under bipartite
  cover (Pati-Salam structure depends on k=3, |V|=4 K_4 vs |V|=8 Q_3
  doesn't change Cl(6)).

  But Pati-Salam acts on the per-vertex 8-dim Cl(6) Hilbert space. χ̃
  acts on the directed-edge walker space. They live on DIFFERENT spaces
  (per Layer 5 trace).

  Open structural question (multi-session research):
    Does the Pati-Salam action on per-vertex space preserve the χ̃ grading
    on the walker? If yes, Pati-Salam reps come in (χ̃ = +1, χ̃ = −1) pairs
    — i.e., each Pati-Salam multiplet gets a SUSY partner Pati-Salam
    multiplet. If no, the relationship is more complex.

  Concrete next probe (research-level):
    Build the Pati-Salam action on the walker space explicitly via the
    framework's chain (per-vertex Cl(6) tensor product → directed-edge
    projection per local CAR construction). Test whether each Pati-Salam
    generator commutes with χ̃.

  This is the boundary with full SUSY phenomenology — answers would identify
  what the SUSY partners of each SM particle (quarks, leptons, gauge bosons,
  Higgs) ARE in the framework's algebraic structure.
""")


if __name__ == '__main__':
    main()
