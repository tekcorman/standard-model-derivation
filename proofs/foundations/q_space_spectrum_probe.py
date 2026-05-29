#!/usr/bin/env python3
"""
Q-space cumulative spectrum probe — empirical density vs a separate private derivation by the author uniform ansatz.

Background. Per a separate private derivation by the author §4a, dark
corrections to amplitude observables (V_us, m_ν2, m_ν3) are derived as
Feshbach self-energies from the "ruliad complement" Q-space:

    Σ(h) = α₁ · ∮ dφ/(2π) · 1/(h − √(k−1)·e^(iφ))

assuming uniform angular density of Q-space eigenvalues on the Ramanujan
circle |λ| = √(k−1). This evaluates to Σ(h) = α₁/h, giving
|Im[Σ(h)]| = α₁·Im(h)/|h|² = α₁·√5/4 ≈ 0.0218 (matches V_us correction
to 0.0016%).

a separate private derivation by the author tested Kesten-McKay (NB-walker density on infinite tree) as alternative
and found it gives ~0.041 (1.9× wrong). Only uniform matches.

THIS PROBE asks: is the uniform-density ansatz physically equivalent to the
cumulative spectrum of the alternate substrates that fail the framework's
own MDL waterline? The waterline-failing substrates (srs-z, srs-c4, hcb-c4)
are the natural physical embodiment of the "ruliad complement" — they're
the part of substrate space that the observer doesn't compress.

Methodology:
  1. For each Q-space substrate, sample B(k) at a fine k-grid (10×10×10).
  2. Collect all 24×N_grid eigenvalues per substrate.
  3. Filter to Ramanujan-saturated ones (|λ|² ≈ 2 within tolerance).
  4. Aggregate phases φ = arg(λ).
  5. Compare empirical density ρ_emp(φ) to uniform ρ_uniform(φ) = 1/(2π).
  6. Compute discrete-sum Σ(h) = (1/N) · Σ_λ 1/(h − λ) over Ramanujan eigs.
  7. Compare discrete Σ(h) to a separate private derivation by the author continuous-uniform prediction α₁·h̄/|h|².
  8. Report whether the empirical density justifies a separate private derivation by the author ansatz.

If empirical ρ ≈ uniform: a separate private derivation by the author ansatz is justified by alternate-net structure.
If empirical ρ ≠ uniform: the actual dark correction may differ from a separate private derivation by the author,
                          and we should compute the predicted V_us correction
                          from the empirical density to see if it still matches PDG.
"""

import sys
import os
import math
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges,
)

# Q-space substrates: those that fail the waterline (per `rcsr_ensemble_closure_test.py`)
Q_SPACE_SUBSTRATES = ['srs-z', 'srs-c4', 'hcb-c4']

# Reference srs for comparison
P_SPACE_REFERENCE = 'srs'

# k = 3 for all candidates → Ramanujan circle |λ|² = k-1 = 2
RAMANUJAN_RADIUS_SQ = 2.0
RAMANUJAN_RADIUS = math.sqrt(2.0)

# h saddle eigenvalue (canonical srs)
H_SADDLE = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
ALPHA_1_BARE = (2 / 3) ** 8  # k=3, g=10 → α₁ = (2/3)^8

# k-grid resolution (10x10x10 = 1000 k-points)
K_GRID_RES = 10


def make_kgrid(n=K_GRID_RES):
    """Uniform k-point grid in fractional reciprocal coordinates [0, 1)^3."""
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pts.append(np.array([i / n, j / n, k / n]))
    return pts


def collect_ramanujan_phases(name, entry, tol=1e-3):
    """For substrate `name`, sample B(k) at K_GRID, collect Ramanujan-saturated
    eigenvalues, return their phases φ ∈ [0, 2π)."""
    sg = entry['sg_name']
    rotations, translations, _, _ = get_space_group_ops(sg)
    v_frac = np.array(entry['vertex_orbits'][0]['cartesian'])
    atom_orbit = orbit_of(v_frac, rotations, translations)
    midpoints = [orbit_of(np.array(eo['cartesian']), rotations, translations)
                 for eo in entry['edge_orbits']]
    midpoint_orbit = np.vstack(midpoints)
    bonds_conv = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=3)
    bonds_conv = [b for b in bonds_conv if b is not None]
    arcs = build_directed_edges(bonds_conv)
    n_atoms = len(atom_orbit)

    phases = []
    eigenvalues = []
    for k_pt in make_kgrid():
        B = bloch_hashimoto(arcs, k_pt, n_atoms)
        eigs = np.linalg.eigvals(B)
        for lam in eigs:
            mod_sq = abs(lam) ** 2
            if abs(mod_sq - RAMANUJAN_RADIUS_SQ) < tol:
                phi = math.atan2(lam.imag, lam.real)
                if phi < 0:
                    phi += 2 * math.pi
                phases.append(phi)
                eigenvalues.append(complex(lam))
    return phases, eigenvalues


def histogram_density(phases, n_bins=36):
    """Discrete density on [0, 2π) from phase samples."""
    counts = [0] * n_bins
    for phi in phases:
        idx = int(phi / (2 * math.pi) * n_bins) % n_bins
        counts[idx] += 1
    total = sum(counts) or 1
    density = [c / total / (2 * math.pi / n_bins) for c in counts]
    return density


def sigma_from_discrete(eigenvalues, h, alpha_1):
    """Σ(h) ≈ α₁ · (1/N) · Σ_λ 1/(h − λ)  over the discrete eigenvalue list.

    Note: this is the DISCRETE analog of a separate private derivation by the author continuous integral
    Σ(h) = α₁ · ∮ ρ(φ) dφ / (h − √(k−1)·e^(iφ)).
    """
    if not eigenvalues:
        return 0.0
    s = sum(1.0 / (h - lam) for lam in eigenvalues if abs(h - lam) > 1e-9)
    return alpha_1 * s / len(eigenvalues)


def main():
    print("=" * 88)
    print("Q-SPACE CUMULATIVE SPECTRUM PROBE — empirical vs a separate private derivation by the author uniform ansatz")
    print("=" * 88)

    rcsr_file = '/tmp/rcsr_3d_current.txt'
    all_substrates = list(Q_SPACE_SUBSTRATES) + [P_SPACE_REFERENCE]
    entries = parse_rcsr_3dall(rcsr_file, all_substrates)

    # Reference: a separate private derivation by the author uniform-density Σ(h) and observable correction
    print("\n" + "-" * 88)
    print("a separate private derivation by the author REFERENCE (uniform Q-space density on Ramanujan circle)")
    print("-" * 88)
    sigma_uniform = ALPHA_1_BARE * H_SADDLE.conjugate() / abs(H_SADDLE) ** 2
    print(f"  α₁ᵇᵃʳᵉ = (2/3)^8 = {ALPHA_1_BARE:.6f}")
    print(f"  h = (√3 + i√5)/2 = {H_SADDLE.real:.4f} + {H_SADDLE.imag:.4f}i,  |h|² = {abs(H_SADDLE)**2:.4f}")
    print(f"  Σ_uniform(h) = α₁ · h̄ / |h|² = {sigma_uniform.real:+.5f} + {sigma_uniform.imag:+.5f}i")
    print(f"  |Im[Σ_uniform(h)]| = α₁ · Im(h)/|h|² = α₁ · √5/4 = {ALPHA_1_BARE * math.sqrt(5)/4:.6f}")
    print(f"  Predicted dark correction to V_us: +{ALPHA_1_BARE * math.sqrt(5)/4 * 100:.4f}%")
    print(f"  (Matches PDG V_us = 0.22501 within 0.001% per a separate private derivation by the author.)")

    # --- Per-substrate Q-space sampling ---
    print("\n" + "-" * 88)
    print("PER-SUBSTRATE Q-SPACE EIGENVALUE COLLECTION (Ramanujan-saturated only)")
    print("-" * 88)
    print(f"  Sampling B(k) at {K_GRID_RES}×{K_GRID_RES}×{K_GRID_RES} = {K_GRID_RES**3} k-points per substrate")

    per_sub_phases = {}
    per_sub_eigs = {}
    for name in all_substrates:
        phases, eigs = collect_ramanujan_phases(name, entries[name])
        per_sub_phases[name] = phases
        per_sub_eigs[name] = eigs
        print(f"  {name:<10s}: {len(eigs)} Ramanujan-saturated eigenvalues collected "
              f"({len(eigs)/(K_GRID_RES**3):.2f} per k-point)")

    # --- Aggregate Q-space (failing substrates only) ---
    Q_phases = []
    Q_eigs = []
    for name in Q_SPACE_SUBSTRATES:
        Q_phases.extend(per_sub_phases[name])
        Q_eigs.extend(per_sub_eigs[name])
    print(f"\n  Aggregate Q-space (srs-z + srs-c4 + hcb-c4): {len(Q_eigs)} Ramanujan eigenvalues")

    # --- Histograms ---
    print("\n" + "-" * 88)
    print("ANGULAR DENSITY ρ(φ) — empirical histogram vs uniform 1/(2π) = 0.1592")
    print("-" * 88)
    n_bins = 12
    rho_uniform = 1.0 / (2 * math.pi)
    print(f"  φ-bin (deg)       Q-aggregate    srs (ref)    uniform")
    Q_density = histogram_density(Q_phases, n_bins=n_bins)
    srs_density = histogram_density(per_sub_phases['srs'], n_bins=n_bins)
    for i in range(n_bins):
        phi_lo = i * 360 / n_bins
        phi_hi = (i + 1) * 360 / n_bins
        print(f"  [{phi_lo:>6.1f}, {phi_hi:>6.1f})    {Q_density[i]:>10.4f}   {srs_density[i]:>10.4f}   {rho_uniform:>8.4f}")

    # Chi-square deviation from uniform
    def chi_sq_vs_uniform(density, rho_unif=rho_uniform):
        n = len(density)
        return sum((d - rho_unif) ** 2 / rho_unif for d in density) * (2 * math.pi / n)

    chi_Q = chi_sq_vs_uniform(Q_density)
    chi_srs = chi_sq_vs_uniform(srs_density)
    print(f"\n  Deviation from uniform (chi-square integral):")
    print(f"    Q-aggregate: {chi_Q:.4e}")
    print(f"    srs (ref):   {chi_srs:.4e}")
    if chi_Q < 0.01:
        print(f"  → Q-aggregate density is APPROXIMATELY UNIFORM. a separate private derivation by the author ansatz consistent.")
    elif chi_Q < chi_srs:
        print(f"  → Q-aggregate is closer to uniform than srs is. Partial support for ansatz.")
    else:
        print(f"  → Q-aggregate is MORE STRUCTURED than uniform. a separate private derivation by the author ansatz NOT obviously justified.")

    # --- Discrete-sum Σ(h) on Q-aggregate vs a separate private derivation by the author continuous-uniform ---
    print("\n" + "-" * 88)
    print("Σ(h) FROM Q-SPACE EIGENVALUES — discrete sum vs a separate private derivation by the author continuous uniform")
    print("-" * 88)
    sigma_emp = sigma_from_discrete(Q_eigs, H_SADDLE, ALPHA_1_BARE)
    print(f"  a separate private derivation by the author continuous uniform (Σ = α₁·h̄/|h|²):  Σ = {sigma_uniform.real:+.5f} + {sigma_uniform.imag:+.5f}i")
    print(f"  empirical discrete sum on Q-aggregate:    Σ = {sigma_emp.real:+.5f} + {sigma_emp.imag:+.5f}i")
    print(f"  ratio (empirical / uniform):              {abs(sigma_emp)/abs(sigma_uniform):.4f}")

    # Predicted V_us correction from empirical Σ
    pred_corr_emp = abs(sigma_emp.imag)
    pred_corr_uniform = abs(sigma_uniform.imag)
    print(f"\n  V_us dark correction prediction:")
    print(f"    a separate private derivation by the author uniform:       {pred_corr_uniform * 100:.4f}%  → V_us = (2/3)^(2+√3)·(1+{pred_corr_uniform:.5f}) = 0.22500")
    bare_v_us = (2/3)**(2 + math.sqrt(3))
    pred_v_us_emp = bare_v_us * (1 + pred_corr_emp)
    print(f"    empirical Q-sum:   {pred_corr_emp * 100:.4f}%  → V_us = {bare_v_us:.5f} · (1+{pred_corr_emp:.5f}) = {pred_v_us_emp:.5f}")
    print(f"    PDG V_us:          0.22501 ± 0.00067")
    pdg_v_us = 0.22501
    sigma_pdg = 0.00067
    shift_emp = (pred_v_us_emp - pdg_v_us) / sigma_pdg
    print(f"    empirical V_us shift from PDG: {shift_emp:+.2f}σ")

    # Per-substrate Σ contributions
    print("\n" + "-" * 88)
    print("PER-SUBSTRATE Σ(h) CONTRIBUTIONS (separate Q-space members)")
    print("-" * 88)
    print(f"  {'substrate':<10s} {'N_eigs':>8s} {'Re Σ':>10s} {'Im Σ':>10s} {'|Σ|':>10s}")
    for name in Q_SPACE_SUBSTRATES:
        eigs = per_sub_eigs[name]
        if not eigs:
            print(f"  {name:<10s} {len(eigs):>8d}  (no Ramanujan eigenvalues)")
            continue
        sig = sigma_from_discrete(eigs, H_SADDLE, ALPHA_1_BARE)
        print(f"  {name:<10s} {len(eigs):>8d} {sig.real:>+10.5f} {sig.imag:>+10.5f} {abs(sig):>10.5f}")

    # --- Conclusion ---
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"""
  a separate private derivation by the author assumes Q-space density is uniform on the Ramanujan circle |λ| = √2.
  This probe tests the assumption against the empirical eigenvalue
  distribution from the framework's waterline-failing substrates.

  Empirical findings:
    - Q-space density chi-square deviation from uniform: {chi_Q:.4e}
    - Empirical Σ(h) magnitude / a separate private derivation by the author uniform: {abs(sigma_emp)/abs(sigma_uniform):.3f}
    - Empirical V_us shift from PDG: {shift_emp:+.2f}σ vs a separate private derivation by the author matches to 0.001%

  Interpretation depends on whether |empirical Σ| / |uniform Σ| ≈ 1:
    - If ≈ 1 (within ~few %): a separate private derivation by the author uniform ansatz is consistent with the
      discrete spectrum of failing substrates. The framework's R-9 closure
      via Options 4 + 5 PLUS a separate private derivation by the author dark correction mechanism are
      compatible — they describe the same physics from two angles.
    - If significantly different: either the failing-substrate set in our
      candidate scope is too narrow (more substrates needed for the
      cumulative density to converge to uniform), OR a separate private derivation by the author uniform ansatz
      isn't actually equivalent to the alternate-net cumulative spectrum.
""")


if __name__ == '__main__':
    main()
