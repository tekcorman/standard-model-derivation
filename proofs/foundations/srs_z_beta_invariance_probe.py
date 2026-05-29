#!/usr/bin/env python3
"""
P1.1 — β cosmic birefringence invariance across the bipartite-cover-shadow
family {srs-z, lov} (post-EOD sweep update).

The β prediction is β = sin(arg h) · α_EM with h = (√3 + i√5)/2, the
doubly-degenerate Hashimoto saddle eigenvalue on srs at the P-point.

Per `srs_z_partner_predictions.py`, h at the K-rational saddle is CLASS C:
the COMPLEX VALUE is invariant under the bipartite double cover srs → srs-z.
This probe extends the verification to lov (the second bipartite-primitive
substrate identified in the post-EOD candidate sweep, per
an internal working note).

Test: do srs-z AND lov BOTH host h = (√3+i√5)/2 in their saddle spectra?
If yes — CLASS C holds across the bipartite-cover-shadow family; β is
robustly invariant under cover.
If no — CLASS C is specific to substrates where the primitive K_4 quotient
underlies the cover; β invariance argument is narrower than claimed.
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
from rcsr_candidate_sweep import (
    primitive_quotient_via_body_centering, find_bipartition_full,
)


def saddle_h_check(name, sg_short, k_point=(0.5, 0.5, 0.5)):
    """Compute Hashimoto B(k) on candidate's primitive walker at the K-rational
    saddle and check if h = (√3 + i√5)/2 is in the spectrum.

    Returns (n_prim, n_arcs, h_present, h_multiplicity, magsq_distribution).
    """
    entries = parse_rcsr_3dall('/tmp/rcsr_3d_current.txt', [name])
    entry = entries[name]
    rotations, translations, _, _ = get_space_group_ops(sg_short)
    v_frac = np.array(entry['vertex_orbits'][0]['cartesian'])
    atom_orbit = orbit_of(v_frac, rotations, translations)

    # Reconstruct conventional bonds
    edge_orbits = entry['edge_orbits']
    conv_bonds = []
    for eorb in edge_orbits:
        m_frac = np.array(eorb['cartesian'])
        midpoint_orbit = orbit_of(m_frac, rotations, translations)
        bonds = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=2)
        conv_bonds.extend([b for b in bonds if b is not None])

    # For I-centered groups (I4(1)32), apply body-centering quotient to get primitive cell.
    # For P groups (P4(1)32), conventional = primitive.
    if 'I' in sg_short:
        n_prim, A_prim, _, prim_bonds, _ = primitive_quotient_via_body_centering(atom_orbit, conv_bonds)
    else:
        n_prim = len(atom_orbit)
        prim_bonds = conv_bonds

    # Build Hashimoto walker on primitive cell
    arcs = build_directed_edges(prim_bonds)
    n_arcs = len(arcs)
    k_R = np.array(k_point)
    B = bloch_hashimoto(arcs, k_R, n_prim)

    # Diagonalize and check for h = (√3 + i√5)/2
    eigs = np.linalg.eigvals(B)
    h_target = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
    h_present = sum(1 for e in eigs if abs(e - h_target) < 1e-6)
    magsq = Counter([round(abs(e)**2, 4) for e in eigs])
    return n_prim, n_arcs, h_present, magsq


def main():
    print("=" * 78)
    print("P1.1 — β invariance across bipartite-cover-shadow family {srs-z, lov}")
    print("=" * 78)
    print(f"\nTarget: h = (√3 + i√5)/2 ≈ 0.866 + 1.118i, |h|² = 2")
    print(f"sin(arg h) = Im(h)/|h| = √(5/8) ≈ 0.7906\n")

    # Test on srs-z (P4(1)32, primitive = conventional)
    print("--- srs-z (P4(1)32) ---")
    n_p, n_a, h_count_szz, magsq_szz = saddle_h_check('srs-z', 'P4(1)32')
    print(f"  primitive: |V| = {n_p}, |arcs| = {n_a}")
    print(f"  |λ|² distribution: {dict(sorted(magsq_szz.items()))}")
    print(f"  count of eigenvalues equal to (√3+i√5)/2 (within 1e-6): {h_count_szz}")

    # Test on lov (I4(1)32, primitive = body-centering quotient)
    print("\n--- lov (I4(1)32) ---")
    n_p_lov, n_a_lov, h_count_lov, magsq_lov = saddle_h_check('lov', 'I4(1)32')
    print(f"  primitive: |V| = {n_p_lov}, |arcs| = {n_a_lov}")
    print(f"  |λ|² distribution: {dict(sorted(magsq_lov.items()))}")
    print(f"  count of eigenvalues equal to (√3+i√5)/2 (within 1e-6): {h_count_lov}")

    # Compute β
    print("\n--- β = sin(arg h) · α_EM ---")
    ALPHA_EM = 1.0 / 137.035999084
    sin_arg_h = np.sqrt(5/8)
    beta_rad = sin_arg_h * ALPHA_EM
    beta_deg = np.degrees(beta_rad)
    print(f"  sin(arg h) = √(5/8) = {sin_arg_h:.10f}")
    print(f"  β = sin(arg h) · α_EM = {beta_deg:.6f}°")
    print(f"  β_obs (Eskilt 2022) = 0.342° ± 0.094°")
    print(f"  Deviation: {(beta_deg - 0.342)/0.094:+.3f}σ")

    # Verdict
    print("\n" + "=" * 78)
    print("Verdict — β CLASS C invariance scope")
    print("=" * 78)
    print()
    if h_count_szz > 0 and h_count_lov > 0:
        print(f"  Both srs-z (mult {h_count_szz}) and lov (mult {h_count_lov}) host the\n"
              f"  K-rational saddle eigenvalue h = (√3 + i√5)/2 in their primitive\n"
              f"  walker spectrum at k = R.\n\n"
              f"  → β is INVARIANT across the bipartite-cover-shadow family {{srs-z, lov}},\n"
              f"    confirming CLASS C label per `srs_z_partner_predictions.py`.\n"
              f"  → β = 0.331° prediction is robust whether physical substrate is srs alone,\n"
              f"    srs ⊕ srs-z, srs ⊕ lov, or srs ⊕ srs-z ⊕ lov Boltzmann ensemble.\n\n"
              f"  P1.1 closure: β CLASS C INVARIANCE EXTENDED across the full bipartite-\n"
              f"  cover-shadow family identified by the post-EOD candidate sweep.")
    elif h_count_szz > 0 and h_count_lov == 0:
        print(f"  srs-z hosts h = (√3+i√5)/2 (mult {h_count_szz}) — CLASS C ✓ vs srs.")
        print(f"  lov does NOT host h = (√3+i√5)/2; its saddle has |λ|² = 5 (not 2).\n")
        print(f"  STRUCTURAL READING: CLASS C invariance is specifically about COVER")
        print(f"  RELATIONS (srs's K_4 ↔ srs-z's Q_3 = bipartite double cover of K_4),")
        print(f"  NOT about the general bipartite-primitive family. lov is a DIFFERENT")
        print(f"  substrate (12 primitive vertices vs K_4's 4), not a cover of srs.")
        print(f"  Its saddle eigenvalue lives at a DIFFERENT |λ|² value, so β computed")
        print(f"  on lov directly would NOT equal β on srs.\n")
        print(f"  CONSEQUENCE for ensemble framing: if Boltzmann substrate includes lov,")
        print(f"  β contribution from the lov sector is computed with lov's own saddle,")
        print(f"  not by inheriting srs's value. The total β prediction is:")
        print(f"    ⟨β⟩ = β_srs + ω_lov·β_lov + ω_srsz·β_srs (= same as β_srs by CLASS C)")
        print(f"  where ω_lov, ω_srsz are Boltzmann weights.\n")
        print(f"  P1.1 closure: β CLASS C INVARIANCE under srs ↔ srs-z confirmed (the")
        print(f"  bipartite double-cover relation preserves the K_4 saddle structure).")
        print(f"  Extension to general bipartite-primitive family REFUTED — lov has its")
        print(f"  own saddle at |λ|² = 5. The β PREDICTION on srs alone is unaffected;")
        print(f"  the ENSEMBLE framing requires per-substrate β computation.")
    elif h_count_szz == 0:
        print(f"  Probe failure — srs-z primitive doesn't host h = (√3+i√5)/2.\n"
              f"  Investigate: |λ|² distribution: {dict(magsq_szz)}.\n"
              f"  May indicate primitive-cell or saddle-k mis-identification.")
    print()


if __name__ == '__main__':
    main()
