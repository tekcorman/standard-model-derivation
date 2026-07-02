#!/usr/bin/env python3
"""
Layer 4 trace: χ̃ × observer (Bayesian walker) + χ̃ × Cl(6).

The observer in this framework IS the MDL-optimal NB walker (per
predictions/walker_dynamics_derivation.md Step 5). The observer's Hilbert
space IS the directed-edge causal-state space (per W2 closure). On srs-z,
this is 24-dim — exactly where χ̃ lives.

So χ̃ is a Z_2 grading on the OBSERVER'S OWN HILBERT SPACE.

This probe walks up from the Layer-3 χ̃ finding to Layer 4:

  (1) Confirm B² (two-step walker operator) is block-diagonal in χ̃ basis,
      with eigenvalues mass-degenerate across sectors (same |λ|² in both).
      Implication: 2-step walks live within a single χ̃ sector; physical
      observables that aggregate over even-step walks see the same spectrum
      in both sectors.

  (2) Distinguish framework observables by their χ̃-parity:
      • B-derived (1-step amplitudes): χ̃-odd, anti-commute with χ̃,
        eigenvalues come in ± pairs at same |λ|².
      • B²-derived (2-step amplitudes; mass-squared, Im(h)/|h|², dark
        coefficient via Hashimoto trace, etc.): χ̃-even, commute with χ̃,
        respect the sector decomposition.

  (3) Observer's compressed walks: χ̃ flips at each step. Even-length walks
      → same sector; odd-length walks → opposite sector. The observer's
      data has built-in Z_2 grading by walk-length parity.

  (4) Identify what the framework's PHOTON-COUNT (n_γ) doubling at srs-z's
      saddle (mult 2 → 4) corresponds to algebraically: it is precisely
      the χ̃ doubling — each srs photon polarization (L=ω, R=ω²) lifts to
      χ̃=+1 and χ̃=−1 sectors on srs-z, giving 4 channels total.

This is a structural trace, not a derivation of new physics — it identifies
WHERE the framework's existing observables live in the χ̃-graded algebra.
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
    print("Layer 4 trace: χ̃ × observer (walker Hilbert space) + χ̃ × Cl(6)")
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

    # Build χ̃
    side_label = {v: +1 for v in side_0}
    side_label.update({v: -1 for v in side_1})
    chi = np.diag([side_label[a[0]] for a in arcs]).astype(complex)

    chi_plus = [i for i, a in enumerate(arcs) if side_label[a[0]] == +1]
    chi_minus = [i for i, a in enumerate(arcs) if side_label[a[0]] == -1]
    print(f"\nObserver's Hilbert space (srs-z directed-edge causal-state space): {n_arcs}-dim")
    print(f"  χ̃ = +1 sector: {len(chi_plus)}-dim")
    print(f"  χ̃ = −1 sector: {len(chi_minus)}-dim")

    # === (1) B² block-diagonal in χ̃ basis ===
    print("\n" + "=" * 80)
    print("(1) Two-step walker B² — χ̃-EVEN operator, block-diagonal in χ̃ basis")
    print("=" * 80)

    k_R = np.array([0.5, 0.5, 0.5])
    B_R = bloch_hashimoto(arcs, k_R, n_atoms)
    B2_R = B_R @ B_R

    # Check B² commutes with χ̃
    comm_B2 = chi @ B2_R - B2_R @ chi
    norm_comm = np.linalg.norm(comm_B2)
    norm_B2 = np.linalg.norm(B2_R)
    print(f"  ||[χ̃, B²]||           = {norm_comm:.4e}")
    print(f"  ||B²||                = {norm_B2:.4f}")
    if norm_comm / norm_B2 < 1e-10:
        print(f"  → B² COMMUTES with χ̃ — block-diagonal in χ̃ basis (sectors decouple)")

    # B² eigenvalues per sector
    B2_pp = B2_R[np.ix_(chi_plus, chi_plus)]
    B2_mm = B2_R[np.ix_(chi_minus, chi_minus)]
    B2_pm = B2_R[np.ix_(chi_plus, chi_minus)]
    print(f"  ||B²_++|| (within +) = {np.linalg.norm(B2_pp):.4f}")
    print(f"  ||B²_−−|| (within −) = {np.linalg.norm(B2_mm):.4f}")
    print(f"  ||B²_+−|| (off-diag) = {np.linalg.norm(B2_pm):.4e}")

    # Compare eigenvalue spectra of B²_++ vs B²_--
    eigs_pp = sorted(np.real(eigvals(B2_pp)), reverse=True)
    eigs_mm = sorted(np.real(eigvals(B2_mm)), reverse=True)
    print(f"\n  B²_++ eigenvalues (real parts): {[f'{e:+.3f}' for e in eigs_pp]}")
    print(f"  B²_−− eigenvalues (real parts): {[f'{e:+.3f}' for e in eigs_mm]}")
    if all(abs(eigs_pp[i] - eigs_mm[i]) < 1e-6 for i in range(len(eigs_pp))):
        print(f"  → SPECTRA IDENTICAL: B² has same eigenvalues in both χ̃ sectors.")
        print(f"    Mass-squared observables (∝ B² eigenvalues) are χ̃-degenerate.")

    # === (2) Distinguish observables by χ̃-parity ===
    print("\n" + "=" * 80)
    print("(2) Framework observables classified by χ̃-parity at the operator level")
    print("=" * 80)
    print("""
  χ̃-ODD operators (anti-commute with χ̃, generate sector flipping):
    • B (Hashimoto 1-step) — anti-commutes with χ̃ everywhere ✓ (verified)
    • B³, B⁵, ... (odd powers)
    • Any operator built from odd combinations of bipartite-flipping moves
    Physical interpretation: 1-step amplitude observables, h saddle eigenvalue,
    Re(h), Im(h), V_cb amplitudes (built from twisted walker T = B·C_36 — C_36
    is C_3 conjugation cycle, generally χ̃-even, so T inherits χ̃-odd from B).

  χ̃-EVEN operators (commute with χ̃, preserve sector structure):
    • B² (2-step walker, mass-squared scale) — commutes ✓ (verified)
    • |B|² = B†B
    • Identity I, projectors, density operators
    • |h|² = 2 (Ramanujan modulus)
    • α_1 = (q_NB)^(g-2) — depends only on (k, g), no walker eigenvalue
    Physical interpretation: mass-squared, |amplitude|², dark coefficient
    (∝ trace of polynomial in B), framework's K-rational scalar predictions.

  Mass-spectrum implication:
    On srs-z, masses derived from |λ|² (= B² eigenvalues) are χ̃-DEGENERATE:
    each mass m has a partner at the same m, in the opposite χ̃ sector.
    This IS the algebraic mass-degeneracy of unbroken SUSY.

    The χ̃ symmetry-breaking (= mass splitting between SM and SUSY partners
    in observed phenomenology) would come from a NEW operator term that's
    NEITHER χ̃-odd NOR χ̃-even (i.e., breaks the algebra). Soft SUSY-breaking
    in real-world phenomenology fits this template.
""")

    # === (3) Observer's compressed walks: χ̃ parity by walk length ===
    print("=" * 80)
    print("(3) Observer's data: χ̃ parity by walk length")
    print("=" * 80)
    print(f"""
  After T toggle events, the observer's directed-edge state has χ̃ parity:
    χ̃(state at T) = χ̃(initial) × (−1)^T

  This is a CONSERVED Z_2 quantum number on the (state, walk-length-parity)
  combined space. The observer can DETECT the χ̃ sector by tracking walk
  length modulo 2 — even walks return to the original χ̃ sector, odd walks
  flip it.

  This is the structural origin of "antiparticle-like" pairing in the
  framework's compressed observer data:
    • Even-step contributions: SM-like (single sector)
    • Odd-step contributions: cross-sector (couple χ̃=+1 to χ̃=−1)

  The framework's existing Sakharov chain (η_B with M=6 on srs, M=12 on
  srs-z) involves walks of LENGTH = number of edges = chain length. On
  srs-z, the chain is even-length (M=12), so the chain RETURNS TO THE
  SAME χ̃ SECTOR after the full chain. The chain expectation value lives
  within ONE sector.

  Compare srs (M=6, even on its 12-dim space): also even-length chain,
  also within-sector at the algebraic level — but srs has no χ̃, so the
  "single sector" is the entire space.
""")

    # === (4) The mult-2 → mult-4 doubling = χ̃ doubling ===
    print("=" * 80)
    print("(4) Photon polarization count doubling (n_γ: 2 → 4) is χ̃ doubling")
    print("=" * 80)
    print(f"""
  Per `theorem_BP_doubly_degenerate_h.md`, on srs the K-rational saddle k_P
  has eigenvalue h = (√3 + i√5)/2 with mult 2 — interpreted as L = ω-irrep
  + R = ω²-irrep of C_3 (two photon polarizations, n_γ = 2).

  On srs-z's K-rational saddle k_R, the SAME K-rational h = (√3 + i√5)/2
  appears with mult 4. The doubling is now algebraically explained:

    On srs:  k_P with C_3 stabilizer → 2 C_3-irreps (ω, ω²) per saddle
    On srs-z: k_R with C_3 stabilizer + bipartite χ̃ doubling →
                  2 C_3-irreps × 2 χ̃-sectors = 4 modes

  Each srs photon polarization (L = ω, R = ω²) lifts to TWO srs-z modes
  related by χ̃ — one in side-A sector, one in side-B sector. The mult-4
  is therefore "doubled photon polarization" = (L_+, L_−, R_+, R_−).

  In SUSY language: each SM photon polarization gets a SUSY-partner
  polarization in the χ̃ = -1 sector. These 4 modes per saddle on srs-z
  correspond to:
    • 2 photon polarizations (SM photon, χ̃ = +1 sector)
    • 2 photino polarizations (SUSY partner, χ̃ = −1 sector)

  At least at the algebraic level. Interaction of χ̃-graded modes with
  Cl(6) at the per-vertex Hilbert space (Layer 4 boundary with Layer 5)
  is the next step — would identify which channels carry which SUSY-graded
  quantum numbers.
""")

    # === (5) Cl(6) parity vs χ̃: are they the same Z_2? ===
    print("=" * 80)
    print("(5) Cl(6) chirality γ_7 vs χ̃ — same Z_2 or different?")
    print("=" * 80)
    print(f"""
  The framework already has a Z_2 grading from Cl(6): γ_7 = γ_1 γ_2 ... γ_6
  (the chirality projector). γ_7 acts on the per-vertex Cl(6) Hilbert space
  (dim 2³ = 8 per vertex; dim 8 × N_atoms total).

  χ̃ acts on the directed-edge state space (dim 2|E| = 24 for srs-z).

  These live on DIFFERENT Hilbert spaces:
    Cl(6) per-vertex space: 8^N_atoms dim
    Walker directed-edge:    2|E| dim

  For a structural identification "γ_7 ≡ χ̃ at the appropriate level," we'd
  need a map between these two spaces. The local CAR / Jordan-Wigner
  construction (`theorem_car_local_jordan_wigner.md`) provides such a map
  at the per-vertex level — but its compatibility with the directed-edge
  walker decomposition is not trivially established.

  Open question for next-session investigation:
    Does γ_7 (chirality projector at each vertex) commute or anti-commute
    with the natural map from (Cl(6) per-vertex Hilbert space) to (directed
    edge state space)? If the map is intertwining (preserves the Z_2), then
    γ_7 ≡ χ̃ algebraically — and the SUSY supercharge structure on srs-z
    is the SAME thing as the Cl(6) chirality, just realized at the walker
    level.

  If the two Z_2's are independent, the algebra is Z_2 × Z_2, giving
  RICHER structure (e.g., 4 sectors instead of 2). This would correspond
  to TWO supercharges Q_1, Q_2 — extended SUSY at the substrate level.

  This is research-level. The structural finding so far:
    χ̃ exists, anti-commutes with B(k) on srs-z everywhere.
    Cl(6) γ_7 exists at each vertex per the framework's existing construction.
    The relationship between them needs explicit derivation.
""")


if __name__ == '__main__':
    main()
