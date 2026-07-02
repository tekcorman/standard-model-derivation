#!/usr/bin/env python3
"""
χ̃-decomposed Feshbach probe — split a separate private derivation by the author Σ(h) by χ̃ sectors on srs-z.

Background. a separate private derivation by the author (a separate private derivation by the author
dark_correction_theorem_2026-04-14.md`) computes the Feshbach self-energy
Σ(h) from the Q-space (alternate-net) Ramanujan-circle spectrum. Per our
prior probe (`q_space_spectrum_probe.py`), srs-z's eigenvalues alone reproduce
a separate private derivation by the author Σ ≈ -0.022i with high accuracy. srs-z hosts the χ̃ algebra (γ_7^A → -χ̃
on walker per `srs_z_gamma7_lift_recovers_chi.py`).

This probe asks: when we decompose Σ(h) by χ̃ sectors of srs-z's bipartite
walker, do the χ̃ = +1 and χ̃ = -1 sectors give EQUAL contributions, or do
they differ?

  - EQUAL Σ_+ = Σ_-:  SUSY-pair structure preserved at the dark-correction
                       level. Consistent with P2.3 BLOCKED (no canonical
                       χ̃-symmetry-breaking mechanism in current framework).
                       Predicts no observable χ̃-asymmetry in dark-sector
                       observables — i.e., no SUSY-breaking signal.

  - UNEQUAL Σ_+ ≠ Σ_-: The Feshbach mechanism breaks the χ̃ symmetry at
                        the dark-correction level. This would be an observable
                        χ̃-asymmetry in V_us / m_ν dark corrections — a
                        substrate-level SUSY-breaking signal not currently
                        captured in the framework.

Methodology. For srs-z primitive walker, build χ̃ projector P_+ = (1+χ̃)/2
and P_- = (1-χ̃)/2. At each k-point, compute B(k) eigenstates and project
each onto P_+ and P_-. The Q-space contribution to Σ(h) split by χ̃:
  Σ_±(h) = α₁ · ⟨ψ_h|P_± · (h - QHQ)^{-1} · P_±|ψ_h⟩

Discrete approximation: for each Ramanujan-saturated B(k) eigenvalue λ
with eigenstate v, compute |P_± v|² and weight 1/(h-λ) by these projector
weights. Sum to get Σ_±(h).
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges,
)
from srs_z_bipartite_involution_commutation import find_bipartition, build_adjacency

H_SADDLE = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
ALPHA_1_BARE = (2 / 3) ** 8
RAMANUJAN_RADIUS_SQ = 2.0
K_GRID_RES = 10


def make_kgrid(n=K_GRID_RES):
    return [np.array([i / n, j / n, k / n])
            for i in range(n) for j in range(n) for k in range(n)]


def get_srs_z_walker():
    """Build srs-z's primitive walker arcs + χ̃ diagonal (on directed-arc space)."""
    entries = parse_rcsr_3dall('/tmp/rcsr_3d_current.txt', ['srs-z'])
    e = entries['srs-z']
    rotations, translations, _, _ = get_space_group_ops('P4(1)32')
    v_frac = np.array(e['vertex_orbits'][0]['cartesian'])
    m_frac = np.array(e['edge_orbits'][0]['cartesian'])
    atom_orbit = orbit_of(v_frac, rotations, translations)
    midpoint_orbit = orbit_of(m_frac, rotations, translations)
    bonds = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=2)
    bonds = [b for b in bonds if b is not None]
    arcs = build_directed_edges(bonds)
    n_atoms = len(atom_orbit)
    A = build_adjacency(bonds, n_atoms)
    side_a, side_b = find_bipartition(A)
    side_label = {v: +1 for v in side_a}
    side_label.update({v: -1 for v in side_b})
    chi_diag = np.array([side_label[a[0]] for a in arcs], dtype=complex)
    return arcs, n_atoms, chi_diag, side_a, side_b


def chi_decomposed_sigma(arcs, n_atoms, chi_diag, h, alpha_1, kpts, tol=1e-3):
    """Compute Σ_+(h) and Σ_-(h) from B(k) eigenstates, weighted by χ̃-projectors.

    Σ_±(h) = α₁ · (1/N_eigs) · Σ_λ |P_± v_λ|² / (h − λ)

    where v_λ is the normalized eigenstate of B(k) at eigenvalue λ.
    """
    P_plus = np.diag((1 + chi_diag) / 2)   # χ̃ = +1 projector
    P_minus = np.diag((1 - chi_diag) / 2)  # χ̃ = -1 projector
    sum_plus = 0.0 + 0.0j
    sum_minus = 0.0 + 0.0j
    n_total = 0
    n_plus_count = 0
    n_minus_count = 0
    for k_pt in kpts:
        B = bloch_hashimoto(arcs, k_pt, n_atoms)
        eigvals, eigvecs = np.linalg.eig(B)
        for i, lam in enumerate(eigvals):
            mod_sq = abs(lam) ** 2
            if abs(mod_sq - RAMANUJAN_RADIUS_SQ) > tol:
                continue
            v = eigvecs[:, i]
            v = v / np.linalg.norm(v)
            w_plus = float(np.real(v.conj() @ P_plus @ v))
            w_minus = float(np.real(v.conj() @ P_minus @ v))
            denom = h - lam
            if abs(denom) < 1e-9:
                continue
            sum_plus += w_plus / denom
            sum_minus += w_minus / denom
            n_total += 1
            if w_plus > w_minus:
                n_plus_count += 1
            else:
                n_minus_count += 1
    if n_total == 0:
        return None, None, n_total, n_plus_count, n_minus_count
    sigma_plus = alpha_1 * sum_plus / n_total
    sigma_minus = alpha_1 * sum_minus / n_total
    return sigma_plus, sigma_minus, n_total, n_plus_count, n_minus_count


def main():
    print("=" * 88)
    print("χ̃-DECOMPOSED FESHBACH PROBE — srs-z bipartite-cover sector decomposition")
    print("=" * 88)

    arcs, n_atoms, chi_diag, side_a, side_b = get_srs_z_walker()
    print(f"\n  srs-z walker: |V|={n_atoms}, |arcs|={len(arcs)}")
    print(f"  χ̃ = +1 sector: {sum(1 for c in chi_diag if c.real > 0)} dim")
    print(f"  χ̃ = -1 sector: {sum(1 for c in chi_diag if c.real < 0)} dim")
    print(f"  Side A vertices (χ̃ = +1): {side_a}")
    print(f"  Side B vertices (χ̃ = -1): {side_b}")

    print(f"\n  h saddle = ({H_SADDLE.real:.4f} + {H_SADDLE.imag:.4f}i),  α₁ᵇᵃʳᵉ = {ALPHA_1_BARE:.6f}")
    print(f"  a separate private derivation by the author uniform Σ_total = α₁·h̄/|h|² = ({ALPHA_1_BARE * H_SADDLE.conjugate() / abs(H_SADDLE)**2})")

    print(f"\n  Sampling {K_GRID_RES**3} k-points...")
    kpts = make_kgrid()
    sigma_plus, sigma_minus, n_total, n_plus_dom, n_minus_dom = chi_decomposed_sigma(
        arcs, n_atoms, chi_diag, H_SADDLE, ALPHA_1_BARE, kpts
    )
    sigma_total = sigma_plus + sigma_minus

    print("\n" + "-" * 88)
    print("RESULTS")
    print("-" * 88)
    print(f"  Total Ramanujan-saturated eigenvalues: {n_total}")
    print(f"  Eigenstates with χ̃-dominant +: {n_plus_dom}  ({100*n_plus_dom/n_total:.1f}%)")
    print(f"  Eigenstates with χ̃-dominant -: {n_minus_dom}  ({100*n_minus_dom/n_total:.1f}%)")
    print()
    print(f"  Σ_+(h)        = {sigma_plus.real:+.6f} + {sigma_plus.imag:+.6f}i,  |Σ_+| = {abs(sigma_plus):.6f}")
    print(f"  Σ_-(h)        = {sigma_minus.real:+.6f} + {sigma_minus.imag:+.6f}i,  |Σ_-| = {abs(sigma_minus):.6f}")
    print(f"  Σ_+(h) + Σ_-(h) = {sigma_total.real:+.6f} + {sigma_total.imag:+.6f}i,  |Σ_total| = {abs(sigma_total):.6f}")
    print()

    # Asymmetry metrics
    re_asym = abs(sigma_plus.real - sigma_minus.real) / max(abs(sigma_total.real), 1e-9)
    im_asym = abs(sigma_plus.imag - sigma_minus.imag) / max(abs(sigma_total.imag), 1e-9)
    mag_ratio = abs(sigma_plus) / max(abs(sigma_minus), 1e-9)
    print(f"  Re asymmetry: |Re Σ_+ − Re Σ_-| / |Re Σ_total| = {re_asym:.4f}")
    print(f"  Im asymmetry: |Im Σ_+ − Im Σ_-| / |Im Σ_total| = {im_asym:.4f}")
    print(f"  Magnitude ratio: |Σ_+| / |Σ_-| = {mag_ratio:.4f}")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if im_asym < 0.01 and re_asym < 0.01:
        print("""
  ‖Σ_+‖ ≈ ‖Σ_-‖ — χ̃ SECTORS GIVE EQUAL FESHBACH CONTRIBUTIONS

  The dark-correction Feshbach mechanism PRESERVES the χ̃-symmetry at the
  dark-sector level. SUSY-pair structure is intact at the dark-correction
  level. NO observable χ̃-asymmetry signal in V_us / m_ν dark corrections.

  This is consistent with P2.3 BLOCKED (no canonical χ̃-symmetry-breaking
  mechanism in current framework). The χ̃ algebra exists structurally on
  srs-z's walker, propagates into Q-space Feshbach corrections, but produces
  symmetric (= unbroken-SUSY) contributions to observables.
""")
    elif im_asym < 0.1 and re_asym < 0.1:
        print(f"""
  ‖Σ_+‖ ≈ ‖Σ_-‖ to ~10% — χ̃ sectors give APPROXIMATELY EQUAL contributions
  (Im asymmetry {im_asym:.3f}, Re asymmetry {re_asym:.3f}).

  Marginal χ̃-asymmetry at the dark-correction level. Worth further
  investigation but not a clean SUSY-breaking signal at this resolution.
""")
    else:
        print(f"""
  Σ_+ ≠ Σ_- AT NON-NEGLIGIBLE LEVEL — χ̃ symmetry IS broken at Feshbach level
  (Im asymmetry {im_asym:.3f}, Re asymmetry {re_asym:.3f}).

  The Feshbach mechanism produces UNEQUAL χ̃-sector contributions to dark
  corrections. This is a CHANNEL FOR OBSERVABLE χ̃-ASYMMETRY in dark-sector
  observables. Specifically, V_us / m_ν dark corrections would carry a
  χ̃-graded sub-structure that's currently NOT captured in the framework's
  predictions.

  This would be a substrate-level SUSY-breaking signal — independent of P2.3
  BLOCKED's bipartition-orientation problem. The Feshbach Q-space integrates
  over BOTH χ̃ orientations and may give a NET ASYMMETRY without needing a
  canonical orientation choice.
""")


if __name__ == '__main__':
    main()
