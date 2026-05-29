#!/usr/bin/env python3
"""
B1 — V_ub structural labeling via Z_6 = χ̃ × C_3 on srs-z's V_Ram.

Per an internal working note route (d): "color = generation
unification" via the (4, 2, 2) at P-point fiber labeling BOTH color (per B6 +
sin²θ_W) AND generation. Closing this would resolve V_ub's Z_3 generation
identification gap.

This probe tests whether the χ̃ × C_3 = Z_6 grading on srs-z provides the
needed structural mechanism. Specifically:

  (1) On srs's K_4 quotient at P-point, fiber C_3 has multiplicities
      (4, 2, 2) on the 8-dim Cl(6,0) spinor (per `theorem_B6_bridge.py`).

  (2) On srs-z's Q_3 quotient (bipartite double cover of K_4), the body-
      diagonal C_3 acts on the 24-dim walker. We compute:
        (a) C_3 isotypic multiplicities on the FULL 24-dim walker
        (b) C_3 isotypic multiplicities on V_Ram(srs-z) = 16-dim subspace
            (|h|² = 2 eigenspace)
        (c) Joint χ̃ × C_3 multiplicities on V_Ram

  (3) If χ̃ × C_3 produces ASYMMETRIC isotypic dims across χ̃ sectors
      (e.g., (4, 2, 2) on χ̃=+1 sector but DIFFERENT on χ̃=−1 sector),
      then χ̃ provides the Z_2 distinction needed for the duality.
      → Route (d) closure mechanism identified.

  (4) If χ̃ × C_3 produces SYMMETRIC isotypic dims (same multiplicities
      in both χ̃ sectors), then χ̃ does NOT provide a new generation-
      distinguishing structure beyond what srs already has.
      → V_ub closure path (d) NOT closed by χ̃ × C_3 alone; alternative
      route (a/b/c) needed.

Either outcome is informative — the probe reports honestly.
"""

import numpy as np
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges,
)
from srs_z_bipartite_involution_commutation import (
    build_adjacency, find_bipartition,
    space_group_perms_on_atoms, lift_perm_to_arcs, perm_to_matrix,
)


def find_C3_along_111(rotations, translations):
    """Find the body-diagonal C_3 op (cyclic xyz → zxy) in P4_132."""
    R_target = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for i, (R, t) in enumerate(zip(rotations, translations)):
        if np.array_equal(R, R_target):
            return i, R, t
    R_target_inv = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    for i, (R, t) in enumerate(zip(rotations, translations)):
        if np.array_equal(R, R_target_inv):
            return i, R, t
    return None, None, None


def isotypic_multiplicities(C3_op, n_dim, omega):
    """Compute C_3 isotypic multiplicities (m_1, m_ω, m_{ω²}) on full n_dim space.

    Uses character formula: m_λ = (1/3)[χ(e) + λ̄·χ(c) + λ̄²·χ(c²)] where
    χ(g) = trace of representation matrix at g.
    """
    chi_e = n_dim
    chi_c = np.trace(C3_op).real
    chi_c2 = np.trace(C3_op @ C3_op).real
    m_1 = round((chi_e + chi_c + chi_c2) / 3)
    m_w = round((chi_e + omega.conjugate() * chi_c + (omega ** 2).conjugate() * chi_c2).real / 3)
    m_w2 = round((chi_e + (omega ** 2).conjugate() * chi_c + omega.conjugate() * chi_c2).real / 3)
    return (m_1, m_w, m_w2)


def restricted_character(C3_op, projector, omega):
    """Character of C_3 restricted to im(projector). Multiplicities of restriction."""
    P = projector
    chi_e = round(np.trace(P).real)
    chi_c = round(np.trace(P @ C3_op @ P).real)
    chi_c2 = round(np.trace(P @ C3_op @ C3_op @ P).real)
    # Use complex chi
    chi_e_c = np.trace(P).real
    chi_c_c = np.trace(P @ C3_op @ P).real
    chi_c2_c = np.trace(P @ C3_op @ C3_op @ P).real
    m_1 = round((chi_e_c + chi_c_c + chi_c2_c) / 3)
    m_w = round((chi_e_c + omega.conjugate() * chi_c_c + (omega ** 2).conjugate() * chi_c2_c).real / 3)
    m_w2 = round((chi_e_c + (omega ** 2).conjugate() * chi_c_c + omega.conjugate() * chi_c2_c).real / 3)
    return (m_1, m_w, m_w2)


def main():
    print("=" * 78)
    print("B1 — V_ub structural labeling: χ̃ × C_3 isotypic on srs-z's V_Ram")
    print("=" * 78)

    omega = np.exp(2j * np.pi / 3)

    # --- Section A: srs-z setup ---------------------------------------------
    print("\n--- Section A: srs-z primitive Q_3 walker ---")
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
    A_mat = build_adjacency(bonds, n_atoms)
    side_A, side_B = find_bipartition(A_mat)
    print(f"  Q_3: |V| = {n_atoms}, |arcs| = {n_arcs}")
    print(f"  Bipartition: A = {side_A}, B = {side_B}")

    # χ̃ on arc space
    side_label = {v: +1 for v in side_A}
    side_label.update({v: -1 for v in side_B})
    chi_tilde = np.diag([side_label[a[0]] for a in arcs]).astype(complex)
    print(f"  χ̃ built (24-dim diag, side-of-tail labeling)")

    # --- Section B: body-diagonal C_3 lifted to walker ----------------------
    print("\n--- Section B: body-diagonal C_3 lifted to 24-dim walker ---")
    op_idx, R_C3, t_C3 = find_C3_along_111(rotations, translations)
    sg_perms = space_group_perms_on_atoms(atom_orbit, rotations, translations)
    C3_perm = next((perm for perm, idx in sg_perms if idx == op_idx), None)
    print(f"  C_3 vertex permutation: {C3_perm}")
    arc_perm_C3 = lift_perm_to_arcs(C3_perm, arcs, atom_orbit, R_C3, t_C3)
    if any(x is None for x in arc_perm_C3):
        print("  WARNING: arc lift failed for some arcs; aborting.")
        return
    C3_walker = perm_to_matrix(arc_perm_C3, n_arcs)
    print(f"  C_3 walker matrix: {C3_walker.shape}, order check ||C_3³ - I|| = "
          f"{np.linalg.norm(C3_walker @ C3_walker @ C3_walker - np.eye(n_arcs)):.2e}")

    # --- Section C: C_3 isotypic on full walker -----------------------------
    print("\n--- Section C: C_3 isotypic multiplicities on full 24-dim walker ---")
    full_iso = isotypic_multiplicities(C3_walker, n_arcs, omega)
    print(f"  (m_1, m_ω, m_{{ω²}}) on full walker = {full_iso}  (sum = {sum(full_iso)})")
    print(f"  Per Layer 6 probe: expect uniform (8, 8, 8).")

    # --- Section D: V_Ram(srs-z) identification -----------------------------
    print("\n--- Section D: V_Ram(srs-z) at saddle k = R = (½, ½, ½) ---")
    k_R = np.array([0.5, 0.5, 0.5])
    B_R = bloch_hashimoto(arcs, k_R, n_atoms)
    eigvals_B, eigvecs_B = np.linalg.eig(B_R)
    # Ramanujan: |λ|² = k* - 1 = 2  → |λ| = √2 ≈ 1.4142
    ram_threshold = 1e-6
    ram_indices = [i for i in range(n_arcs) if abs(abs(eigvals_B[i])**2 - 2.0) < ram_threshold]
    eig_magsq_grouped = Counter([round(abs(e)**2, 3) for e in eigvals_B])
    print(f"  B(k_R) eigenvalue |λ|² distribution: {dict(eig_magsq_grouped)}")
    print(f"  V_Ram = ker(B† B − 2 · I) = |λ|² = 2 modes: {len(ram_indices)}-dim")
    print(f"    (expected 16 = 4 × {{±(√3+i√5)/2, ±(√3−i√5)/2}}; remaining 8 = real ±1 modes)")

    # Build V_Ram orthonormal basis from the |λ|² = 2 eigenvectors
    V_Ram_basis = eigvecs_B[:, ram_indices]
    Q_VRam, _ = np.linalg.qr(V_Ram_basis)
    P_VRam = Q_VRam @ Q_VRam.conj().T
    print(f"  V_Ram projector: rank {round(np.trace(P_VRam).real)}, "
          f"||P_VRam² − P_VRam|| = {np.linalg.norm(P_VRam @ P_VRam - P_VRam):.2e}")

    # --- Section E: C_3 isotypic on V_Ram ----------------------------------
    print("\n--- Section E: C_3 isotypic multiplicities on V_Ram ---")
    BdaggerB = B_R.conj().T @ B_R
    comm_C3_BdB = C3_walker @ BdaggerB - BdaggerB @ C3_walker
    print(f"  ||[C_3, B† B]|| / ||B† B|| = {np.linalg.norm(comm_C3_BdB) / np.linalg.norm(BdaggerB):.2e}")
    iso_VRam = restricted_character(C3_walker, P_VRam, omega)
    print(f"  (m_1, m_ω, m_{{ω²}}) on V_Ram = {iso_VRam}  (sum = {sum(iso_VRam)})")

    # --- Section F: χ̃ × C_3 joint isotypic on V_Ram ------------------------
    print("\n--- Section F: Joint χ̃ × C_3 isotypic multiplicities on V_Ram ---")
    # Verify χ̃ commutes with C_3 (Layer 6 found yes)
    comm_chi_C3 = chi_tilde @ C3_walker - C3_walker @ chi_tilde
    print(f"  ||[χ̃, C_3]|| = {np.linalg.norm(comm_chi_C3):.2e}")
    if np.linalg.norm(comm_chi_C3) > 1e-8:
        print(f"  WARNING: χ̃ and C_3 don't commute. Skip joint analysis.")
        return

    # χ̃ projectors
    P_chi_plus = (np.eye(n_arcs) + chi_tilde) / 2.0
    P_chi_minus = (np.eye(n_arcs) - chi_tilde) / 2.0

    # Joint projectors on V_Ram, χ̃ = ±1
    P_VRam_chi_plus = P_VRam @ P_chi_plus
    P_VRam_chi_minus = P_VRam @ P_chi_minus

    iso_plus = restricted_character(C3_walker, P_VRam_chi_plus, omega)
    iso_minus = restricted_character(C3_walker, P_VRam_chi_minus, omega)

    print(f"  V_Ram ∩ (χ̃ = +1): isotypic (m_1, m_ω, m_{{ω²}}) = {iso_plus}  (sum = {sum(iso_plus)})")
    print(f"  V_Ram ∩ (χ̃ = −1): isotypic (m_1, m_ω, m_{{ω²}}) = {iso_minus}  (sum = {sum(iso_minus)})")

    # --- Section G: Verdict on V_ub closure ---------------------------------
    print("\n" + "=" * 78)
    print("Section G — Verdict on V_ub structural labeling closure (B1)")
    print("=" * 78)

    if iso_plus == iso_minus:
        print(f"""
  χ̃ × C_3 multiplicities are SYMMETRIC across χ̃ sectors:
    χ̃ = +1: {iso_plus}
    χ̃ = −1: {iso_minus}

  → The Z_2 χ̃ does NOT distinguish the two copies of (m_1, m_ω, m_{{ω²}})
    structurally. Both χ̃ sectors carry IDENTICAL C_3 multiplicities.

  CONSEQUENCE FOR V_ub:
    Closure path (d) "color = generation duality via χ̃ × C_3" is NOT
    grounded by the χ̃ structure alone. The χ̃ Z_2 provides a Boltzmann-
    weighted shadow distinction (per the χ̃ ≡ γ_7^A unification's N=1
    SUSY-pair interpretation), but does NOT provide a Z_3 generation
    distinction beyond what srs's K_4 already has.

  ALTERNATIVE V_ub CLOSURE PATHS (per theorem_Vub_scoping.md):
    (a) Find a Z_3-asymmetric generation structure beyond χ̃ × fiber-C_3.
        Possibilities: (i) edge-label Z_3 trivalent connection (girth-cycle
        holonomy); (ii) some other non-spatial Z_3 in the substrate.
    (b) Reformulate V_ub via Level-3 walk-rep at L_ub ≠ 8 between non-Z_3
        labeled causal states. Requires identifying the labels.
    (c) Reframe V_ub as CKM unitarity-triangle consequence: with V_us, V_cb,
        δ_CP_CKM all closed at theorem grade, V_ub follows from unitarity +
        Wolfenstein ρ̄, η̄ apex coordinates if those can be derived
        structurally. Most tractable route given current closures.

  HONEST FRAMING: the χ̃ unification's structural payoff for V_ub is
  LIMITED. It provides the SUSY-pair algebra (closes Tier 1 A2/A3, Tier 3
  C1/C2 via P2.1 foundation) but does NOT close the V_ub labeling gap
  directly. B1 as originally proposed (closes 6 rows) is REVISED: V_ub
  + dependent rows (P32, P33, P34, P35, P36) remain in the labeling-
  data-anchored state via Row P14 inheritance.

  This is a NEGATIVE STRUCTURAL RESULT — informative because it sharpens
  what's needed for V_ub closure. Roadmap should add new entry: V_ub
  closure path (c) via unitarity triangle + δ_CP_CKM as a research follow-up.
""")
    else:
        print(f"""
  χ̃ × C_3 multiplicities DIFFER across χ̃ sectors:
    χ̃ = +1: {iso_plus}
    χ̃ = −1: {iso_minus}

  → The Z_2 χ̃ DOES distinguish the two C_3-multiplicity patterns.
  → Route (d) "color = generation via χ̃ × C_3" has structural ground.
  → V_ub closure path (d) IS grounded by the χ̃ machinery.

  Next step: identify which χ̃ sector labels color, which labels generation,
  and verify the resulting V_ub formula matches PDG.
""")


if __name__ == '__main__':
    main()
