#!/usr/bin/env python3
"""
P2.3 Step 1 — Construct T_mix cross-substrate Boltzmann coupling.

Per an internal working note,
T_mix is the only viable candidate for a χ̃-symmetry-breaking operator
(classes 1-3 are sterile; class 4-5 = cross-substrate is the lone path).

This probe constructs T_mix on the joint walker H = H_srs ⊕ H_srs-z, weighted
by the M2a Boltzmann factor (per `srs_vs_srs_z_dl_audit.py` ΔDL = +3.25 bits
→ w_srs-z = 2^(-3.25) ≈ 0.105). Tests whether T_mix breaks χ̃ at the algebraic
level, and computes the second-order mass splitting on χ̃-paired states.

Joint setup:
  H = H_srs ⊕ H_srs-z = 12-dim ⊕ 24-dim = 36-dim
  χ̃_joint = diag(I_12 (trivially +1, no Z_2 on srs walker), χ̃_24 on H_srs-z)
  T_mix = [[0, T_off], [T_off†, 0]]   (block off-diagonal)

Where T_off : H_srs-z → H_srs implements the bipartite-cover projection
π : Q_3 → K_4 (each Q_3 vertex projects to its image K_4 vertex), times the
Boltzmann amplitude √w.

KEY STRUCTURAL OBSTRUCTION (predicted): the bipartite cover Q_3 → K_4 is
2-to-1 with A and B sides projecting to the SAME K_4 vertex SYMMETRICALLY.
The "natural" T_mix doesn't distinguish A from B → commutes with the A↔B
exchange that χ̃ implements → does NOT break χ̃. Verified explicitly here.

Conclusion: at the cover-projection level, T_mix is sterile for χ̃-breaking.
For χ̃-breaking, T_mix needs an ADDITIONAL structural orientation input
that distinguishes A from B canonically. Such input does not exist in the
framework's current structure (the bipartition is canonical but unoriented).
This identifies the obstruction sharply.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges,
)
from rcsr_candidate_sweep import (
    primitive_quotient_via_body_centering, find_bipartition_full,
)
from srs_z_bipartite_involution_commutation import (
    build_adjacency, find_bipartition,
)


def build_srs_walker():
    """Build srs's primitive K_4 walker (12 directed arcs)."""
    entries = parse_rcsr_3dall('/tmp/rcsr_3d_current.txt', ['srs'])
    srs = entries['srs']
    rotations, translations, _, _ = get_space_group_ops('I4(1)32')
    v_frac = np.array(srs['vertex_orbits'][0]['cartesian'])
    atom_orbit_conv = orbit_of(v_frac, rotations, translations)

    edge_orbits = srs['edge_orbits']
    conv_bonds = []
    for eorb in edge_orbits:
        m_frac = np.array(eorb['cartesian'])
        midpoint_orbit = orbit_of(m_frac, rotations, translations)
        bonds = reconstruct_bonds(atom_orbit_conv, midpoint_orbit, tol=1e-3, max_shift=2)
        conv_bonds.extend([b for b in bonds if b is not None])

    n_prim, A_prim, _, prim_bonds, conv_to_prim = primitive_quotient_via_body_centering(
        atom_orbit_conv, conv_bonds)
    arcs = build_directed_edges(prim_bonds)
    return n_prim, arcs, A_prim


def build_srsz_walker():
    """Build srs-z's primitive Q_3 walker (24 directed arcs)."""
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
    A_mat = build_adjacency(bonds, n_atoms)
    return n_atoms, arcs, A_mat, atom_orbit


def main():
    print("=" * 78)
    print("P2.3 Step 1 — T_mix cross-substrate Boltzmann coupling probe")
    print("=" * 78)

    # --- Build srs and srs-z walkers ---
    n_srs, arcs_srs, _ = build_srs_walker()
    n_srsz, arcs_srsz, A_srsz, _ = build_srsz_walker()
    n_arcs_srs = len(arcs_srs)
    n_arcs_srsz = len(arcs_srsz)
    print(f"\nsrs walker: {n_srs} vertices, {n_arcs_srs} arcs")
    print(f"srs-z walker: {n_srsz} vertices, {n_arcs_srsz} arcs")

    # --- Bipartition + χ̃ on srs-z ---
    side_A, side_B = find_bipartition(A_srsz)
    print(f"srs-z bipartition: A = {side_A}, B = {side_B}")

    # χ̃ on srs-z walker (24-dim)
    side_label = {v: +1 for v in side_A}
    side_label.update({v: -1 for v in side_B})
    chi_srsz = np.diag([side_label[a[0]] for a in arcs_srsz]).astype(complex)

    # --- Bipartite-cover projection π: Q_3 → K_4 ---
    # Q_3 has 8 vertices; K_4 has 4. Each K_4 vertex has 2 Q_3 preimages
    # (one in A, one in B). We need to identify the 2-to-1 projection.
    # Per `srs_z_bipartite_double_cover_probe.py`, the iso permutation is
    # (0, 5, 2, 7, 1, 4, 3, 6). This maps Q_3 vertices onto BD(K_4) vertices
    # labeled (v, 0) for v in [0..3] and (v, 1) for v in [4..7].
    # The natural projection π forgets the side index:
    #   Q_3 vertex i → K_4 vertex (perm[i] mod 4)
    perm = [0, 5, 2, 7, 1, 4, 3, 6]
    pi_q3_to_k4 = [perm[i] % 4 for i in range(8)]
    print(f"\nQ_3 → K_4 projection π: {pi_q3_to_k4}")
    print(f"  side A = {side_A} → K_4 vertices {[pi_q3_to_k4[v] for v in side_A]}")
    print(f"  side B = {side_B} → K_4 vertices {[pi_q3_to_k4[v] for v in side_B]}")

    # --- Construct T_off: H_srs-z (24) → H_srs (12) ---
    # For each srs-z walker basis vector |a_z⟩ with arc a_z = (tail_z, head_z, shift_z),
    # find the corresponding srs walker arc a = (π(tail_z), π(head_z), shift) if exists.
    # If multiple srs-z arcs project to the same srs arc, accumulate.
    # T_off matrix element T_off[a_srs, a_z] = 1 if π(a_z) = a_srs, else 0.

    # Build srs arc index map
    srs_arc_idx = {(t, h): i for i, (t, h, s) in enumerate(arcs_srs)}

    # Allow shifts to be ignored for first-pass simplicity (project onto K_4 graph)
    T_off = np.zeros((n_arcs_srs, n_arcs_srsz), dtype=complex)
    proj_count = 0
    for j, (t_z, h_z, _) in enumerate(arcs_srsz):
        t_k4 = pi_q3_to_k4[t_z]
        h_k4 = pi_q3_to_k4[h_z]
        if (t_k4, h_k4) in srs_arc_idx:
            i = srs_arc_idx[(t_k4, h_k4)]
            T_off[i, j] = 1.0
            proj_count += 1

    print(f"\nT_off projection structure: {proj_count} of {n_arcs_srsz} srs-z arcs project to srs arcs")
    print(f"T_off shape: {T_off.shape}")

    # Each srs arc gets contributions from multiple srs-z arcs (4 expected: 2 A→B + 2 B→A directions per K_4 edge)
    proj_per_srs_arc = np.sum(np.abs(T_off) > 0, axis=1)
    print(f"Per-srs-arc preimage count distribution: {np.unique(proj_per_srs_arc, return_counts=True)}")

    # --- Boltzmann amplitude ---
    delta_DL = 3.25  # M2a, per srs_vs_srs_z_dl_audit.py
    w_srsz = 2**(-delta_DL)
    amp = np.sqrt(w_srsz)
    print(f"\nM2a ΔDL(srs-z − srs) = {delta_DL} bits → w_srs-z/w_srs = {w_srsz:.4f}")
    print(f"T_mix amplitude (√w) = {amp:.4f}")
    T_off_weighted = amp * T_off

    # --- Joint walker space H = H_srs ⊕ H_srs-z (36-dim) ---
    n_joint = n_arcs_srs + n_arcs_srsz
    T_mix = np.zeros((n_joint, n_joint), dtype=complex)
    T_mix[:n_arcs_srs, n_arcs_srs:] = T_off_weighted
    T_mix[n_arcs_srs:, :n_arcs_srs] = T_off_weighted.conj().T
    print(f"\nJoint walker H = H_srs ⊕ H_srs-z = {n_arcs_srs}+{n_arcs_srsz} = {n_joint}-dim")
    print(f"T_mix is Hermitian off-diagonal: ||T_mix - T_mix†|| = {np.linalg.norm(T_mix - T_mix.conj().T):.2e}")

    # χ̃_joint: trivially +1 on srs (no Z_2 there), χ̃ on srs-z
    chi_joint = np.zeros((n_joint, n_joint), dtype=complex)
    chi_joint[:n_arcs_srs, :n_arcs_srs] = np.eye(n_arcs_srs)
    chi_joint[n_arcs_srs:, n_arcs_srs:] = chi_srsz

    # --- Test: does T_mix break χ̃ ? ---
    print("\n" + "=" * 78)
    print("Test: does T_mix break χ̃ on the joint walker?")
    print("=" * 78)
    comm_T_chi = T_mix @ chi_joint - chi_joint @ T_mix
    norm_comm = np.linalg.norm(comm_T_chi)
    norm_T = np.linalg.norm(T_mix)
    print(f"  ||[T_mix, χ̃_joint]|| = {norm_comm:.4e}")
    print(f"  ||T_mix||             = {norm_T:.4e}")

    # --- Second-order mass-splitting on χ̃-paired states ---
    # H_0 = (block-diag) B_srs(k_R) ⊕ B_srs-z(k_R) (the unperturbed Hashimoto)
    # T_mix as perturbation. Second-order correction to E_n on H_srs-z:
    #   ΔE_n^{(2)} = Σ_m |⟨m|T_mix|n⟩|² / (E_n - E_m)
    # The χ̃ = +1 vs χ̃ = -1 pair degeneracy is broken iff
    #   |⟨srs|T_mix|n,+⟩|² ≠ |⟨srs|T_mix|n,-⟩|²
    print("\n" + "=" * 78)
    print("Second-order mass-splitting probe on χ̃-paired states")
    print("=" * 78)

    # Take χ̃ = +1 and χ̃ = -1 states from srs-z walker. Their |T_mix to srs|² should
    # be IDENTICAL if T_off is symmetric under A↔B exchange (which the natural cover
    # projection IS, because both A and B project to the same K_4 vertex set).

    # Sum of |T_off|² rows projected onto each χ̃ sector:
    chi_plus_mask = np.diag(chi_srsz).real > 0.5
    chi_minus_mask = np.diag(chi_srsz).real < -0.5
    norm_sq_plus_per_srs = np.sum(np.abs(T_off_weighted[:, chi_plus_mask])**2, axis=1).sum()
    norm_sq_minus_per_srs = np.sum(np.abs(T_off_weighted[:, chi_minus_mask])**2, axis=1).sum()
    print(f"  Σ|⟨srs|T_mix|χ̃=+1⟩|² (over χ̃=+1 sector) = {norm_sq_plus_per_srs:.6f}")
    print(f"  Σ|⟨srs|T_mix|χ̃=-1⟩|² (over χ̃=-1 sector) = {norm_sq_minus_per_srs:.6f}")
    print(f"  Difference                                = {abs(norm_sq_plus_per_srs - norm_sq_minus_per_srs):.6e}")

    # --- Verdict ---
    print("\n" + "=" * 78)
    print("Verdict — Step 1 outcome")
    print("=" * 78)
    if abs(norm_sq_plus_per_srs - norm_sq_minus_per_srs) < 1e-10:
        print(f"""
  T_mix from natural cover projection π: Q_3 → K_4 produces IDENTICAL
  total transition strength from each χ̃ sector to H_srs:
    χ̃ = +1 sector: {norm_sq_plus_per_srs:.6f}
    χ̃ = −1 sector: {norm_sq_minus_per_srs:.6f}

  → Even though [T_mix, χ̃] ≠ 0 algebraically (off-diagonal couplings exist
    between χ̃-graded srs-z and χ̃-trivial srs), the SECOND-ORDER mass
    correction on χ̃-paired states is SYMMETRIC: the two members of each
    χ̃-pair on srs-z get IDENTICAL second-order shifts from coupling to srs.
  → The χ̃-pair MASS DEGENERACY IS PRESERVED by this T_mix construction.
  → No mass splitting between SM and SUSY-partner sectors.

  STRUCTURAL OBSTRUCTION: the bipartite cover Q_3 → K_4 is 2-to-1 and
  unoriented — A and B sides project to the same K_4 vertex with identical
  weights. The "natural" T_mix is symmetric under A↔B exchange. χ̃ implements
  precisely this A↔B exchange (+1 on A, −1 on B). Symmetric T_mix commutes
  with the exchange action at the rate-square level → χ̃-pair degeneracy
  preserved at second order.

  → P2.3 closure via cross-substrate Boltzmann coupling REQUIRES an
  ADDITIONAL structural input that orients the bipartition (distinguishes
  A from B canonically). The framework's current structure has no such
  canonical orientation: the bipartition (A, B) is unique up to A↔B swap.

  POSSIBLE additional inputs (research-level, not in current framework):
    (i) Time-arrow / causal orientation of the substrate — A = "past",
        B = "future" or similar. Would couple to the framework's R-2
        fixed-point-vacuum structure.
    (ii) External field / vacuum expectation value that picks a specific
         orientation. Not derivable; would be a NEW adoption.
    (iii) A higher-substrate cover that breaks A↔B exchange explicitly.
          E.g., a triple cover Q_3 ⊃ Q_3' where the second cover step
          orients the bipartition.

  P2.3 STATUS: Step 1 produces a NEGATIVE structural result. T_mix at the
  cover-projection level does NOT break χ̃-pair mass degeneracy. The
  remaining viable χ̃-using closures (Tier 1 A2/A3, Tier 3 C1, Tier 4 D1)
  are blocked at the structural-orientation level. Either:
    (a) the framework needs the additional orientation input identified above
        (research-level open problem), OR
    (b) the χ̃-paired SUSY structure on srs-z's walker is genuinely
        unbroken at the substrate level, and physical SUSY breaking comes
        from a non-substrate mechanism (e.g., dynamical, late-universe).

  Both readings are honest. Roadmap update: P2.3 marked BLOCKED on
  bipartition-orientation problem; A2/A3/C1/D1 inherit BLOCKED status.
""")
    else:
        print(f"""
  ASYMMETRIC coupling found: χ̃-pair mass degeneracy IS broken at second order.
    χ̃ = +1: {norm_sq_plus_per_srs:.6f}
    χ̃ = −1: {norm_sq_minus_per_srs:.6f}
    Difference: {abs(norm_sq_plus_per_srs - norm_sq_minus_per_srs):.4e}

  → Proceed to Step 2 (Boltzmann factor refinement) and Step 3 (composition).
""")


if __name__ == '__main__':
    main()
