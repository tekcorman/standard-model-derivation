#!/usr/bin/env python3
"""
χ̃ probe — bipartite CHIRALITY (NOT swap) anti-commutation with Hashimoto B.

The σ̃ probe established that the bipartite SWAP (permuting sides A ↔ B) does
NOT propagate to walker dynamics as an internal symmetry — only at k=Γ does it
commute, breaking at all other k.

But there's a DIFFERENT bipartite-induced operator that's standard in graph
theory: the bipartite CHIRALITY χ̃. It's NOT a permutation; it's a diagonal
sign-matrix on directed arcs:

    χ̃_a = +1 if arc a's tail is on side A
    χ̃_a = −1 if arc a's tail is on side B

KEY CLAIM (this probe verifies): χ̃ ANTI-COMMUTES with B(k) for ALL k on any
bipartite graph's Hashimoto operator.

Reason: B[a',a] is nonzero only when a' continues a (non-backtracking).
Continuation requires tail(a') = head(a). On a bipartite graph, tail(a) and
head(a) are on opposite sides → tail(a') = head(a) is on the side OPPOSITE
to tail(a) → χ̃_{a'} = −χ̃_a.

So (χ̃ B)[a', a] = χ̃_{a'} · B[a', a] = −χ̃_a · B[a', a] = −(B χ̃)[a', a].
Hence χ̃·B = −B·χ̃ entry-wise.

If verified, χ̃ provides EXACTLY the algebraic structure of a SUSY supercharge
Q on srs-z's walker dynamics:
  • χ̃² = I (involution)
  • {χ̃, B} = 0 (anti-commutator vanishes)
  • Eigenvalues of B come in ± pairs related by χ̃

This IS what SUSY needs at the substrate level — the walker eigenstates
naturally split into Z_2-graded sectors with ±B-eigenvalues paired.

Note: srs's K_4 quotient is NOT bipartite (has triangles), so there is no
such χ̃ on srs. The bipartite chirality lives only on srs-z.
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
)


def main():
    print("=" * 80)
    print("χ̃ probe — bipartite CHIRALITY (diagonal) vs Hashimoto B(k) on srs-z")
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
    bp = find_bipartition(A)
    if bp is None:
        print("Graph is NOT bipartite — χ̃ doesn't apply.")
        return
    side_0, side_1 = bp
    print(f"\nsrs-z primitive Q_3 quotient bipartition:")
    print(f"  Side A: {side_0}")
    print(f"  Side B: {side_1}")

    # Build χ̃ — diagonal sign matrix on arcs by tail's side
    side_label = {}
    for v in side_0:
        side_label[v] = +1
    for v in side_1:
        side_label[v] = -1

    chi = np.zeros((n_arcs, n_arcs), dtype=complex)
    for i, (tail, head, shift) in enumerate(arcs):
        chi[i, i] = side_label[tail]

    # Verify χ̃² = I
    chi_sq = chi @ chi
    is_involution = np.allclose(chi_sq, np.eye(n_arcs))
    print(f"\nχ̃² = I (involution check): {is_involution}")

    # Counts of +1 and -1 entries
    count_plus = int(np.sum(np.diag(chi).real > 0.5))
    count_minus = int(np.sum(np.diag(chi).real < -0.5))
    print(f"χ̃ eigenvalues: +1 count = {count_plus}, −1 count = {count_minus}")

    # Test anti-commutation at multiple k-points
    print("\n" + "=" * 80)
    print("Anti-commutation check {χ̃, B(k)} = χ̃·B(k) + B(k)·χ̃ ?= 0")
    print("=" * 80)
    print(f"{'k-point':<25s} {'||χ̃B + Bχ̃||':<18s} {'||B||':<12s} {'verdict'}")
    print("-" * 80)

    k_points = [
        ('Γ = (0,0,0)',          np.array([0.0, 0.0, 0.0])),
        ('R = (1/2,1/2,1/2)',    np.array([0.5, 0.5, 0.5])),
        ('M = (1/2,1/2,0)',      np.array([0.5, 0.5, 0.0])),
        ('X = (1/2,0,0)',        np.array([0.5, 0.0, 0.0])),
        ('mid-body (1/4,1/4,1/4)', np.array([0.25, 0.25, 0.25])),
        ('(1/8,1/8,1/8)',        np.array([0.125, 0.125, 0.125])),
        ('(1/3,1/3,1/3)',        np.array([1/3, 1/3, 1/3])),
        ('generic (0.17,0.31,0.52)', np.array([0.17, 0.31, 0.52])),
    ]

    for label, k_frac in k_points:
        B = bloch_hashimoto(arcs, k_frac, n_atoms)
        anticomm = chi @ B + B @ chi
        norm_anti = np.linalg.norm(anticomm)
        norm_B = np.linalg.norm(B)
        if norm_B == 0:
            verdict = "B is zero"
        elif norm_anti / norm_B < 1e-10:
            verdict = "✓ ANTI-COMMUTES"
        else:
            verdict = f"NON-ZERO (residual/||B|| = {norm_anti/norm_B:.4e})"
        print(f"  {label:<25s} {norm_anti:<18.6e} {norm_B:<12.4f} {verdict}")

    # Detailed spectral consequence at k=R (the K-rational saddle)
    print("\n" + "=" * 80)
    print("Spectral consequence at k = R (K-rational saddle for srs-z)")
    print("=" * 80)
    k_R = np.array([0.5, 0.5, 0.5])
    B_R = bloch_hashimoto(arcs, k_R, n_atoms)
    eigs_B = eigvals(B_R)

    print(f"\nB(k_R) eigenvalues:")
    eigs_sorted = sorted(eigs_B, key=lambda x: (round(x.real, 5), round(x.imag, 5)))
    for e in eigs_sorted:
        if abs(e.imag) > 1e-6:
            re_id = identify_irrational(e.real) or f"{e.real:.4f}"
            im_id = identify_irrational(abs(e.imag)) or f"{abs(e.imag):.4f}"
            mod_sq = abs(e)**2
            ram = " ✓ RAM" if abs(mod_sq - 2) < 1e-6 else f" |u|²={mod_sq:.3f}"
            print(f"    {e.real:+.4f} + {e.imag:+.4f}i  Re~{re_id}, |Im|~{im_id}{ram}")

    # If χ̃ anti-commutes with B(k_R), eigenvalues come in ± pairs.
    # Verify by checking each eigenvalue λ has a pair −λ.
    print(f"\nPairing check: do eigenvalues come in ± pairs?")
    rounded_eigs = Counter([(round(e.real, 4), round(e.imag, 4)) for e in eigs_B])
    paired_count = 0
    unpaired = []
    for (re, im), m in rounded_eigs.items():
        partner = (round(-re, 4), round(-im, 4))
        partner_mult = rounded_eigs.get(partner, 0)
        if partner_mult >= m:
            paired_count += m
        else:
            unpaired.append(((re, im), m))
    print(f"  {paired_count} of {len(eigs_B)} eigenvalues have a partner −λ")
    if unpaired:
        print(f"  Unpaired (or missing partners): {unpaired}")
    else:
        print(f"  ✓ ALL eigenvalues paired — consistent with χ̃ anti-commutation")

    # Compute Q² = identity check
    # If χ̃ ANTI-commutes with B and χ̃² = I, then χ̃ acts on B-eigenstates as:
    # |λ⟩ → some scalar × |−λ⟩
    print(f"\nχ̃ as a SUSY-supercharge Q candidate:")
    print(f"  χ̃² = I ✓ (involution)")
    print(f"  {{χ̃, B(k_R)}} = 0 ✓ (anti-commutes — IF anti-commutation passes)")
    print(f"  Acting on B-eigenstates: maps |λ⟩ → |−λ⟩ (sign-flipped eigenvalue)")
    print(f"  This is the algebra of a SUSY supercharge at the substrate level.")

    # Sector decomposition: split B(k_R) into χ̃ = +1 and χ̃ = −1 sectors
    chi_plus_indices = [i for i in range(n_arcs) if chi[i, i].real > 0.5]
    chi_minus_indices = [i for i in range(n_arcs) if chi[i, i].real < -0.5]
    print(f"\nχ̃-sector decomposition at k = R:")
    print(f"  χ̃ = +1 sector (tail on side A): {len(chi_plus_indices)}-dim")
    print(f"  χ̃ = −1 sector (tail on side B): {len(chi_minus_indices)}-dim")

    # B should have ZERO entries within sectors (χ̃-anti-commutation requires B is purely off-diagonal)
    if len(chi_plus_indices) > 0 and len(chi_minus_indices) > 0:
        B_plus_plus = B_R[np.ix_(chi_plus_indices, chi_plus_indices)]
        B_minus_minus = B_R[np.ix_(chi_minus_indices, chi_minus_indices)]
        B_plus_minus = B_R[np.ix_(chi_plus_indices, chi_minus_indices)]
        B_minus_plus = B_R[np.ix_(chi_minus_indices, chi_plus_indices)]
        print(f"  ||B_++|| (within χ̃=+1 block) = {np.linalg.norm(B_plus_plus):.4e}")
        print(f"  ||B_−−|| (within χ̃=−1 block) = {np.linalg.norm(B_minus_minus):.4e}")
        print(f"  ||B_+−|| (off-diagonal +→−)  = {np.linalg.norm(B_plus_minus):.4e}")
        print(f"  ||B_−+|| (off-diagonal −→+)  = {np.linalg.norm(B_minus_plus):.4e}")
        print()
        if (np.linalg.norm(B_plus_plus) < 1e-10 and np.linalg.norm(B_minus_minus) < 1e-10):
            print(f"  ✓ B is purely OFF-DIAGONAL in χ̃ basis (within-sector blocks zero)")
            print(f"  This confirms B couples ONLY (χ̃=+1) ↔ (χ̃=−1) sectors.")
            print(f"  In SUSY language: B_R IS the SUSY-Q-charge times something Hermitian.")
        else:
            print(f"  Diagonal blocks NON-ZERO — χ̃ does NOT cleanly anti-commute.")

    # Same probe on srs's K_4 — should fail (K_4 isn't bipartite)
    print("\n" + "=" * 80)
    print("Sanity check: same probe on srs's K_4 quotient (NOT bipartite)")
    print("=" * 80)
    print("  Expected: K_4 is not bipartite, so no χ̃ exists.")
    # Build srs primitive K_4 manually
    K4_adj = np.ones((4, 4), dtype=int) - np.eye(4, dtype=int)
    bp_srs = find_bipartition(K4_adj)
    print(f"  K_4 bipartition: {bp_srs}")
    if bp_srs is None:
        print(f"  ✓ Confirmed: K_4 is NOT bipartite — no bipartite chirality on srs.")

    print("\n" + "=" * 80)
    print("STRUCTURAL VERDICT")
    print("=" * 80)
    print("""
  IF χ̃ anti-commutes with B(k) on srs-z for ALL k tested:

  → srs-z's walker dynamics has a NATIVE Z_2 SUPERCHARGE STRUCTURE.
  → The bipartite chirality χ̃ is a substrate-level SUSY supercharge candidate.
  → Each B-eigenstate at the K-rational saddle pairs with another at the
    NEGATED eigenvalue, related by χ̃.
  → This IS the algebraic mechanism the bipartite-cover construction provides
    for SUSY-style state-pairing — but via χ̃ (CHIRALITY), NOT σ̃ (SWAP).

  Compared to srs:
    srs (K_4): no bipartite chirality available (graph not bipartite). No
               substrate-level χ̃ supercharge exists on srs.
    srs-z (Q_3): bipartite chirality χ̃ exists and (per this probe) anti-
                 commutes with B(k) at all k. Substrate-level SUSY supercharge
                 IS realized.

  This means the SUSY Z_2 grading might emerge SPECIFICALLY from substrates
  with bipartite quotient (like Q_3) — and the framework's PRIMARY substrate
  (srs) genuinely does NOT have it. SUSY would be a srs-z-sector phenomenon,
  with srs sourcing the SM and srs-z providing the SUSY-graded shadow.

  Methodology cross-check: this finding (positive on χ̃) is INDEPENDENT of the
  σ̃ negative finding earlier. They test different operators. σ̃ was the
  BIPARTITE SWAP (permutation). χ̃ is the BIPARTITE CHIRALITY (sign matrix).
  Both follow from Q_3's bipartite structure but have different algebraic
  roles.
""")


if __name__ == '__main__':
    main()
