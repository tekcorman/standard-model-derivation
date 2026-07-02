#!/usr/bin/env python3
"""
σ̃ commutation probe: does the bipartite involution commute with B(k) on srs-z?

CONCRETE STRUCTURAL QUESTION: the bipartite-double-cover identification
gives Q_3 (srs-z primitive quotient) a Z_2 grading via the bipartite
involution σ. For this Z_2 to be a SYMMETRY of the framework's walker
dynamics (i.e., for it to propagate from Layer 2 to Layer 3 mechanically),
σ̃ — the lift of σ to the 24-dim directed-edge space — must commute (or
anti-commute) with the Bloch Hashimoto B(k).

This probe:
  1. Constructs Q_3's vertex permutation σ that swaps the bipartite sides
     AND preserves adjacency (graph automorphism).
  2. Checks whether σ is realized as a P4_132 space group operation
     applied to srs-z's atom orbit (i.e., is σ a crystallographic symmetry
     or just an abstract graph symmetry?).
  3. Lifts σ to directed edges (σ̃ as a 24×24 permutation matrix) and
     checks σ̃·B(k) ?= ±B(k)·σ̃ at:
       - k = R = (1/2, 1/2, 1/2)  [srs-z's K-rational saddle]
       - k = Γ = (0, 0, 0)         [zone center]
       - several intermediate k along (t,t,t) body-diagonal axis
  4. Reports the structural verdict:
       - σ̃ commutes → Z_2 internal symmetry, eigenstates carry σ̃-labels
       - σ̃ anti-commutes → SUSY-supercharge-like structure
       - σ̃ neither → bipartite cover Z_2 NOT a walker-dynamics symmetry
"""

import sys
import os
import numpy as np
from numpy.linalg import eigvals, eigh
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges
)


# =============================================================================
# Step 1: Get srs-z's primitive cell + adjacency matrix
# =============================================================================

def get_srs_z_data():
    entries = parse_rcsr_3dall('/tmp/rcsr_3d_current.txt', ['srs-z'])
    srs_z = entries['srs-z']
    rotations, translations, _, _ = get_space_group_ops('P4(1)32')
    v_frac = np.array(srs_z['vertex_orbits'][0]['cartesian'])
    m_frac = np.array(srs_z['edge_orbits'][0]['cartesian'])
    atom_orbit = orbit_of(v_frac, rotations, translations)
    midpoint_orbit = orbit_of(m_frac, rotations, translations)
    bonds = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=2)
    return {
        'atom_orbit': atom_orbit,
        'midpoint_orbit': midpoint_orbit,
        'bonds': [b for b in bonds if b is not None],
        'rotations': rotations,
        'translations': translations,
    }


def build_adjacency(bonds, n_atoms):
    A = np.zeros((n_atoms, n_atoms), dtype=int)
    for i, j, _ in bonds:
        A[i, j] += 1
        if i != j:
            A[j, i] += 1
    return A


# =============================================================================
# Step 2: Find bipartite-swap automorphism of Q_3 (the abstract graph aut)
# =============================================================================

def find_bipartition(A):
    """Find bipartition of a bipartite graph via BFS 2-coloring."""
    n = len(A)
    color = [-1] * n
    color[0] = 0
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if A[u, v] > 0:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return None  # not bipartite
    side_0 = [i for i in range(n) if color[i] == 0]
    side_1 = [i for i in range(n) if color[i] == 1]
    return side_0, side_1


def is_graph_automorphism(perm, A, tol=1e-9):
    """Check if vertex permutation perm preserves adjacency."""
    n = len(A)
    A_arr = np.array(A)
    P = np.zeros((n, n), dtype=int)
    for i, j in enumerate(perm):
        P[i, j] = 1
    return np.array_equal(P @ A_arr @ P.T, A_arr)


def find_bipartite_swap_automorphisms(A):
    """Find all graph automorphisms that swap the bipartite sides.
    Returns list of permutations (as tuples)."""
    bp = find_bipartition(A)
    if bp is None:
        return []
    side_0, side_1 = bp
    n = len(A)
    if len(side_0) != len(side_1):
        return []
    # A permutation σ swaps the sides iff σ(side_0) = side_1 and σ(side_1) = side_0.
    # We need a bijection f: side_0 → side_1 (and inverse for side_1 → side_0)
    # that's a graph automorphism.
    swap_autos = []
    for f in permutations(side_1):
        # f maps side_0[i] → f[i] (an element of side_1)
        # define perm: perm[v] = f[idx_in_side_0] if v in side_0
        #              perm[v] = side_0[idx_of_f^(-1)(v)] if v in side_1
        f_inv = {f[i]: side_0[i] for i in range(len(side_0))}
        perm = [None] * n
        for i, v in enumerate(side_0):
            perm[v] = f[i]
        for v in side_1:
            perm[v] = f_inv[v]
        if is_graph_automorphism(perm, A):
            swap_autos.append(tuple(perm))
    return swap_autos


# =============================================================================
# Step 3: Check whether bipartite swap is in P4_132 space group action
# =============================================================================

def space_group_perms_on_atoms(atom_orbit, rotations, translations, tol=1e-6):
    """For each space group operation, compute the permutation it induces
    on atom_orbit. Returns list of (perm, op_index) for ops giving valid permutations."""
    n = len(atom_orbit)
    perms = []
    for op_idx, (R, t) in enumerate(zip(rotations, translations)):
        perm = [None] * n
        valid = True
        for i, p in enumerate(atom_orbit):
            new_p = (R @ p + t) % 1.0
            # Find which atom in orbit it maps to
            matched = None
            for j, q in enumerate(atom_orbit):
                diff = (new_p - q + 0.5) % 1.0 - 0.5
                if np.linalg.norm(diff) < tol:
                    matched = j
                    break
            if matched is None:
                valid = False
                break
            perm[i] = matched
        if valid and len(set(perm)) == n:
            perms.append((tuple(perm), op_idx))
    return perms


# =============================================================================
# Step 4: Lift σ to directed edges (σ̃) and check commutation with B(k)
# =============================================================================

def lift_perm_to_arcs(vertex_perm, arcs, atom_orbit, R_op, t_op):
    """Given σ on vertices induced by space group op (R, t), compute σ̃ on directed arcs.

    For arc a = (tail, head, shift_a): after applying op (R, t):
      - new tail position: R·p_tail + t = p_{σ(tail)} + δ_tail (where δ is integer shift)
      - new head position: R·(p_head + shift_a) + t = p_{σ(head)} + δ_head + R·shift_a
      - canonical (tail in cell 0): new arc = (σ(tail), σ(head), δ_head − δ_tail + R·shift_a)

    Returns arc-permutation array: arc_perm[i] = index of arc that arc i maps to under σ̃.
    """
    n_arcs = len(arcs)
    arc_perm = [None] * n_arcs
    arc_index = {(a[0], a[1], tuple(a[2])): i for i, a in enumerate(arcs)}

    # Compute integer shifts δ_i for each atom under the op
    deltas = []
    for i, p in enumerate(atom_orbit):
        new_p = R_op @ p + t_op
        target_p = atom_orbit[vertex_perm[i]]
        delta = new_p - target_p
        delta_int = np.round(delta).astype(int)
        deltas.append(delta_int)

    for i, (tail, head, shift) in enumerate(arcs):
        new_tail = vertex_perm[tail]
        new_head = vertex_perm[head]
        shift_arr = np.array(shift)
        new_shift = deltas[head] - deltas[tail] + R_op @ shift_arr
        new_shift_int = np.round(new_shift).astype(int)
        new_arc = (new_tail, new_head, tuple(new_shift_int.tolist()))
        if new_arc in arc_index:
            arc_perm[i] = arc_index[new_arc]
        else:
            arc_perm[i] = None
    return arc_perm


def perm_to_matrix(perm, n):
    """Convert permutation array to permutation matrix P where P[i, perm[i]] = 1."""
    P = np.zeros((n, n), dtype=complex)
    for i, j in enumerate(perm):
        if j is not None:
            P[i, j] = 1
    return P


def check_commutation(B, sigma_tilde, tol=1e-6):
    """Check whether σ̃·B = +B·σ̃ (commute), -B·σ̃ (anti-commute), or neither."""
    LHS = sigma_tilde @ B
    RHS_plus = B @ sigma_tilde
    RHS_minus = -B @ sigma_tilde
    diff_plus = np.linalg.norm(LHS - RHS_plus)
    diff_minus = np.linalg.norm(LHS - RHS_minus)
    norm_B = np.linalg.norm(B)
    if diff_plus < tol * norm_B:
        return 'commutes', diff_plus / norm_B
    elif diff_minus < tol * norm_B:
        return 'anti-commutes', diff_minus / norm_B
    else:
        return 'neither', min(diff_plus, diff_minus) / norm_B


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("σ̃ commutation probe — does bipartite involution propagate to walker?")
    print("=" * 80)

    data = get_srs_z_data()
    atom_orbit = data['atom_orbit']
    bonds = data['bonds']
    rotations = data['rotations']
    translations = data['translations']

    n_atoms = len(atom_orbit)
    A = build_adjacency(bonds, n_atoms)
    print(f"\nsrs-z primitive cell: {n_atoms} atoms, {len(bonds)} bonds")
    print(f"Adjacency matrix:\n{A}")

    # Step 1: Bipartition
    bp = find_bipartition(A)
    if bp is None:
        print("Graph is NOT bipartite. Probe doesn't apply.")
        return
    side_0, side_1 = bp
    print(f"\nBipartition:")
    print(f"  Side A: {side_0}")
    print(f"  Side B: {side_1}")

    # Step 2: Find bipartite-swap automorphisms
    print(f"\nSearching for bipartite-swap graph automorphisms of Q_3...")
    swap_autos = find_bipartite_swap_automorphisms(A)
    print(f"Found {len(swap_autos)} automorphism(s) that swap the bipartition.")
    if len(swap_autos) == 0:
        print("No bipartite-swap automorphism. Probe halts.")
        return
    for k, perm in enumerate(swap_autos[:5]):
        print(f"  σ_{k}: {perm}")
    if len(swap_autos) > 5:
        print(f"  ... and {len(swap_autos)-5} more")

    # Step 3: Are any of these σ in P4_132's space group action?
    sg_perms = space_group_perms_on_atoms(atom_orbit, rotations, translations)
    sg_perm_set = set(p for p, _ in sg_perms)
    print(f"\nP4_132 induces {len(sg_perm_set)} distinct atom permutations from {len(sg_perms)} space group ops.")

    sg_swap_autos = [s for s in swap_autos if s in sg_perm_set]
    print(f"\nBipartite-swap automorphisms that ARE space group ops: {len(sg_swap_autos)}")
    if not sg_swap_autos:
        print("\n  → The bipartite involution is NOT realized as a P4_132 space group operation.")
        print("    It is a purely abstract graph symmetry, not a crystallographic symmetry.")
        return

    print(f"\n  → The bipartite involution IS realized as a P4_132 space group operation.")
    arcs = build_directed_edges(bonds)
    print(f"\nDirected arcs: {len(arcs)}")

    # Try each bipartite-swap σ (each space group op that swaps bipartition);
    # find one(s) where σ̃ lifts cleanly to all 24 arcs.
    candidates = []
    for sigma_perm in sg_swap_autos:
        ops_idx_list = [op_idx for p, op_idx in sg_perms if p == sigma_perm]
        for op_idx in ops_idx_list:
            R_op = rotations[op_idx]
            t_op = translations[op_idx]
            arc_perm = lift_perm_to_arcs(sigma_perm, arcs, atom_orbit, R_op, t_op)
            n_lift = sum(1 for x in arc_perm if x is not None)
            if n_lift == len(arcs):
                candidates.append((sigma_perm, op_idx, R_op, t_op, arc_perm))

    print(f"  {len(candidates)} bipartite-swap σ candidate(s) lift cleanly to arc level.")
    if not candidates:
        print(f"  No σ lifts to all 24 arcs. Bipartite cover does not propagate to walker.")
        return

    # Use the first cleanly-lifting candidate
    sigma, sigma_op_idx, R_op, t_op, arc_perm = candidates[0]
    print(f"\n  Using σ = {sigma} via op #{sigma_op_idx}:")
    print(f"    R = {R_op.tolist()}")
    print(f"    t = {t_op.tolist()}")

    # Convert to matrix
    sigma_tilde = perm_to_matrix(arc_perm, len(arcs))
    # Verify σ̃² = I (involution)
    sigma_tilde_sq = sigma_tilde @ sigma_tilde
    is_involution = np.allclose(sigma_tilde_sq, np.eye(len(arcs)))
    print(f"  σ̃² = I (involution check): {is_involution}")

    # Step 5: Check commutation at various k-points
    print("\n" + "=" * 80)
    print("Commutation check σ̃·B(k) vs ±B(k)·σ̃ at various k-points")
    print("=" * 80)
    print(f"{'k-point':<25s} {'verdict':<20s} {'residual/||B||'}")
    print("-" * 80)

    k_points = [
        ('Γ = (0,0,0)',          np.array([0.0, 0.0, 0.0])),
        ('R = (1/2,1/2,1/2)',    np.array([0.5, 0.5, 0.5])),
        ('M = (1/2,1/2,0)',      np.array([0.5, 0.5, 0.0])),
        ('X = (1/2,0,0)',        np.array([0.5, 0.0, 0.0])),
        ('mid-body (1/4,1/4,1/4)', np.array([0.25, 0.25, 0.25])),
        ('(1/8,1/8,1/8)',        np.array([0.125, 0.125, 0.125])),
        ('(1/3,1/3,1/3)',        np.array([1.0/3, 1.0/3, 1.0/3])),
    ]

    results = {}
    for label, k_frac in k_points:
        B = bloch_hashimoto(arcs, k_frac, n_atoms)
        verdict, residual = check_commutation(B, sigma_tilde)
        results[label] = (verdict, residual)
        print(f"  {label:<25s} {verdict:<20s} {residual:.4e}")

    # Step 6: If σ̃ commutes (or anti-commutes) at some k, decompose B's spectrum
    # by σ̃-eigenvalues
    print("\n" + "=" * 80)
    print("Detailed spectral analysis at k = R (the K-rational saddle)")
    print("=" * 80)
    k_R = np.array([0.5, 0.5, 0.5])
    B_R = bloch_hashimoto(arcs, k_R, n_atoms)
    verdict_R, _ = results['R = (1/2,1/2,1/2)']

    # Eigenvalues of σ̃
    sigma_eigs = eigvals(sigma_tilde)
    print(f"\nσ̃ eigenvalues (should be ±1 since σ̃² = identity for bipartite swap):")
    for e in sigma_eigs[:6]:
        print(f"  {e}")

    if verdict_R == 'commutes':
        # σ̃ and B(k_R) share eigenvectors → can simultaneously diagonalize
        print(f"\nSince σ̃ commutes with B(k_R), eigenstates can carry σ̃-labels.")
        # Diagonalize σ̃ first to get + and − sectors
        sigma_eigvals, sigma_eigvecs = np.linalg.eig(sigma_tilde)
        # Group eigenvectors by sigma eigenvalue (±1)
        plus_mask = np.abs(sigma_eigvals - 1) < 1e-6
        minus_mask = np.abs(sigma_eigvals + 1) < 1e-6
        print(f"  σ̃ = +1 sector: {plus_mask.sum()}-dimensional")
        print(f"  σ̃ = −1 sector: {minus_mask.sum()}-dimensional")

        # Project B(k_R) onto each sector
        if plus_mask.sum() > 0 and minus_mask.sum() > 0:
            V_plus = sigma_eigvecs[:, plus_mask]
            V_minus = sigma_eigvecs[:, minus_mask]
            B_plus = V_plus.conj().T @ B_R @ V_plus
            B_minus = V_minus.conj().T @ B_R @ V_minus
            # Also check off-diagonal blocks (should be zero if commutes)
            B_plus_minus = V_plus.conj().T @ B_R @ V_minus
            print(f"  ||B_+,-|| (off-diagonal block) = {np.linalg.norm(B_plus_minus):.4e}")
            print(f"  Eigenvalues in σ̃=+1 sector ({plus_mask.sum()} of them):")
            for e in sorted(eigvals(B_plus), key=lambda x: (round(x.real, 4), round(x.imag, 4))):
                print(f"    {e.real:+.4f} + {e.imag:+.4f}i")
            print(f"  Eigenvalues in σ̃=−1 sector ({minus_mask.sum()} of them):")
            for e in sorted(eigvals(B_minus), key=lambda x: (round(x.real, 4), round(x.imag, 4))):
                print(f"    {e.real:+.4f} + {e.imag:+.4f}i")

    elif verdict_R == 'anti-commutes':
        print(f"\nσ̃ ANTI-commutes with B(k_R) → eigenvalues come in ± pairs.")
        print(f"This is the algebraic structure of a SUSY-like supercharge Q.")
        eigs_B = eigvals(B_R)
        # Check ± pair structure
        from collections import Counter
        rounded = [(round(e.real, 4), round(e.imag, 4)) for e in eigs_B]
        cnt = Counter(rounded)
        paired = sum(1 for (re, im), m in cnt.items() if (-re, -im) in cnt or (-re, im) in cnt)
        print(f"  {len(cnt)} distinct eigenvalues; pair structure analysis:")
        for (re, im), m in sorted(cnt.items(), key=lambda x: -x[1]):
            negation = (-re, -im)
            partner_mult = cnt.get(negation, 0)
            print(f"    {re:+.4f}+{im:+.4f}i  mult {m}  (paired with {negation}: mult {partner_mult})")

    elif verdict_R == 'neither':
        print(f"\nσ̃ neither commutes nor anti-commutes with B(k_R).")
        print(f"The bipartite Z_2 is NOT a structural symmetry of the walker dynamics")
        print(f"at the K-rational saddle. The Layer 2 → Layer 3 propagation is BROKEN.")

    # Final verdict summary
    print("\n" + "=" * 80)
    print("STRUCTURAL VERDICT")
    print("=" * 80)
    crystallographic = "YES (P4_132 space group op)" if sg_swap_autos else "NO (abstract graph aut only)"
    print(f"  σ realized as P4_132 space group op:  {crystallographic}")
    print(f"  σ̃ commutation with B(k_R):            {results['R = (1/2,1/2,1/2)'][0]}")
    print(f"  σ̃ commutation with B(Γ):              {results['Γ = (0,0,0)'][0]}")

    print()
    if not sg_swap_autos:
        print("  Bipartite Z_2 is NOT a crystallographic symmetry of srs-z.")
        print("  → The Layer-2 bipartite-cover structure does NOT lift cleanly to crystal-")
        print("    level Bloch dynamics. SUSY-flavored interpretation requires reformulation:")
        print("    the Z_2 must come from somewhere other than crystallographic symmetry.")
    elif verdict_R == 'commutes':
        print("  Bipartite Z_2 IS a crystallographic symmetry AND commutes with B(k_R).")
        print("  → Walker dynamics splits into Z_2-graded sectors.")
        print("  → SUSY-flavored interpretation (Z_2 grading on states) is mechanistically")
        print("    realized at the framework's walker level.")
    elif verdict_R == 'anti-commutes':
        print("  Bipartite Z_2 IS crystallographic AND ANTI-commutes with B(k_R).")
        print("  → σ̃ acts as a supercharge-Q-like operator on the walker.")
        print("  → SUSY algebra natively embeds in the framework's walker dynamics.")
    else:
        print("  Bipartite Z_2 is crystallographic but NEITHER commutes nor anti-commutes.")
        print("  → The Z_2 partly breaks under walker dynamics. Mechanism is partial.")


if __name__ == '__main__':
    main()
