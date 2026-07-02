#!/usr/bin/env python3
"""
Investigation #1 — C_3 fiber decomposition of Q-space density via Fourier modes.

Per an internal working note, the
Q-space (= waterline-failing substrates {srs-z, srs-c4, hcb-c4}) has phase
density on the Ramanujan circle that's NOT uniform (χ² = 2443 vs ~12). This
probe quantifies the structure via Fourier mode decomposition.

For Q-space density ρ(φ), define moments:
    M_n := ⟨e^(inφ)⟩ = (1/N) Σ_λ exp(in·arg(λ))

Interpretation:
    M_0 = 1 (normalization)
    M_1 = ⟨cos φ⟩ + i⟨sin φ⟩ — first-moment dipole (zero by Hermiticity)
    M_2 = ⟨cos 2φ⟩ + i⟨sin 2φ⟩ — quadrupole, zero if φ↔−φ symmetric
    M_3 = ⟨cos 3φ⟩ + i⟨sin 3φ⟩ — **C_3 fundamental** — nonzero if C_3
          modulated
    M_6 = first C_3 harmonic
    M_9 = second C_3 harmonic
    Other M_n: non-C_3 modes

For uniform density: only M_0 = 1, all others zero.
For C_3-modulated: M_0, M_3, M_6, M_9, … nonzero; M_1, M_2, M_4, M_5, …
near zero.

Per-substrate decomposition: identifies which substrate contributes which
mode dominantly.

Compare to expected (4, 2, 2) from B6 theorem at the canonical h saddle:
    Multiplicities at φ = ±52° (h, h̄), ±128° (−h̄, −h)
    = 4·δ(φ−52°) + 4·δ(φ+52°) + 2·δ(φ−128°) + 2·δ(φ+128°) for one saddle k
    Scaled cumulative density across k_grid: gives a specific Fourier
    profile; deviations from this profile = Bloch dispersion broadening.
"""

import sys
import os
import math
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges,
)


# ===========================================================================
# Constants
# ===========================================================================

K_STAR = 3
RAM_RADIUS_SQ = K_STAR - 1  # = 2

Q_SPACE_SUBSTRATES = ['srs-z', 'srs-c4', 'hcb-c4']
SG_BY_NAME = {
    'srs-z':  'P4(1)32',
    'srs-c4': 'P4(2)32',
    'hcb-c4': 'P4(3)32',
}

# Canonical Ramanujan saddle phases (radians)
SADDLE_PHASES = {
    '+h':   math.atan2(math.sqrt(5)/2, math.sqrt(3)/2),    # ≈ +52.24°
    '+h̄':   -math.atan2(math.sqrt(5)/2, math.sqrt(3)/2),   # ≈ −52.24°
    '−h̄':   math.pi - math.atan2(math.sqrt(5)/2, math.sqrt(3)/2),    # ≈ +127.76°
    '−h':   -(math.pi - math.atan2(math.sqrt(5)/2, math.sqrt(3)/2)),  # ≈ −127.76°
}


def collect_phases(name, sg, k_grid=(7, 7, 7)):
    """Sample B(k) at a uniform k-grid; collect Ramanujan phases."""
    entries = parse_rcsr_3dall('/tmp/rcsr_3d_current.txt', [name])
    entry = entries[name]
    rotations, translations, _, _ = get_space_group_ops(sg)
    v_frac = np.array(entry['vertex_orbits'][0]['cartesian'])
    atom_orbit = orbit_of(v_frac, rotations, translations)
    edge_orbits = entry['edge_orbits']
    bonds = []
    for eo in edge_orbits:
        m_frac = np.array(eo['cartesian'])
        midpoint_orbit = orbit_of(m_frac, rotations, translations)
        bb = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=2)
        bonds.extend([b for b in bb if b is not None])
    arcs = build_directed_edges(bonds)
    n_atoms = len(atom_orbit)

    phases = []
    for i in range(k_grid[0]):
        for j in range(k_grid[1]):
            for l in range(k_grid[2]):
                k_frac = np.array([i / k_grid[0], j / k_grid[1], l / k_grid[2]])
                B = bloch_hashimoto(arcs, k_frac, n_atoms)
                eigs = np.linalg.eigvals(B)
                for lam in eigs:
                    if abs(abs(lam) ** 2 - RAM_RADIUS_SQ) < 1e-6:
                        phases.append(np.angle(lam))
    return phases


def fourier_moments(phases, n_max=12):
    """Compute M_n = ⟨e^(inφ)⟩ for n = 0 … n_max."""
    if not phases:
        return {n: complex(0, 0) for n in range(n_max + 1)}
    arr = np.array(phases)
    return {n: complex(np.mean(np.cos(n * arr)), np.mean(np.sin(n * arr)))
            for n in range(n_max + 1)}


def expected_saddle_signature():
    """If Q-space density were exactly the canonical (4, 2, 2) saddle structure
    at one k-point, what would the Fourier moments look like?

    At saddle: 4 eigenvalues at +h, 4 at +h̄, 2 at −h̄, 2 at −h (8-dim spinor;
    half goes to mult 4, half to mult 2 per direction). Counts adjusted for
    walker dimension scaling.

    For our 24-dim walker on srs-z: 4 at each of (+h, +h̄, −h, −h̄) at k=R.
    """
    # For (4, 4, 4, 4) at the 4 saddle phases (uniform across the 4 phases):
    canonical_phases = [SADDLE_PHASES[name] for name in ['+h', '+h̄', '−h̄', '−h']]
    weights = [4, 4, 4, 4]  # if all 4 saddle phases get equal mult
    n_total = sum(weights)
    moments = {}
    for n in range(13):
        z = sum(w * np.exp(1j * n * phi) for w, phi in zip(weights, canonical_phases))
        moments[n] = complex(z.real / n_total, z.imag / n_total)
    return moments


def main():
    print("=" * 88)
    print("Investigation #1 — C_3 Fourier decomposition of Q-space density")
    print("=" * 88)

    # --- Per-substrate phase collection ----------------------------------
    print("\nCollecting Ramanujan phases per substrate (k-grid 7×7×7) ...")
    per_substrate = {}
    for name in Q_SPACE_SUBSTRATES:
        phases = collect_phases(name, SG_BY_NAME[name])
        per_substrate[name] = phases
        print(f"  {name:<10s} ({SG_BY_NAME[name]}): {len(phases)} Ramanujan eigs")

    all_phases = sum(per_substrate.values(), [])
    print(f"\n  Total Q-space Ramanujan eigs: {len(all_phases)}")

    # --- Compute Fourier moments per substrate + aggregated --------------
    print("\n" + "=" * 88)
    print("Fourier moments M_n = ⟨e^(inφ)⟩ — per substrate + aggregated")
    print("=" * 88)
    per_sub_moments = {name: fourier_moments(phases, n_max=12)
                        for name, phases in per_substrate.items()}
    agg_moments = fourier_moments(all_phases, n_max=12)
    expected = expected_saddle_signature()

    print(f"\n{'n':>3s} | {'srs-z':>20s} | {'srs-c4':>20s} | {'hcb-c4':>20s} | {'AGGREGATE':>20s} | {'expected (4,4,4,4)':>20s}")
    print(f"{'':->3s}-+-{'':->20s}-+-{'':->20s}-+-{'':->20s}-+-{'':->20s}-+-{'':->20s}")
    for n in range(13):
        line = f"{n:>3d} |"
        for name in Q_SPACE_SUBSTRATES:
            m = per_sub_moments[name][n]
            line += f" {m.real:>+8.4f}{m.imag:>+8.4f}i |"
        m = agg_moments[n]
        line += f" {m.real:>+8.4f}{m.imag:>+8.4f}i |"
        m = expected[n]
        line += f" {m.real:>+8.4f}{m.imag:>+8.4f}i"
        print(line)

    # --- Identify dominant modes -----------------------------------------
    print("\n" + "=" * 88)
    print("Dominant Fourier modes in aggregate Q-space")
    print("=" * 88)
    print(f"\n{'n':>4s}  {'|M_n|':>10s}  {'arg(M_n) (deg)':>16s}  {'interpretation':<40s}")
    print("-" * 80)
    sorted_modes = sorted(range(13), key=lambda n: -abs(agg_moments[n]))
    for n in sorted_modes[:10]:
        m = agg_moments[n]
        interp = ""
        if n == 0:
            interp = "normalization (always 1)"
        elif n % 3 == 0 and n > 0:
            interp = f"C_3 mode (n=3k harmonic)"
        elif n == 1 or n == 2:
            interp = "low-order; should ≈ 0 if Hermitian"
        elif n in {4, 5, 7, 8, 10, 11}:
            interp = "non-C_3 mode"
        print(f"{n:>4d}  {abs(m):>10.6f}  {math.degrees(np.angle(m)) if abs(m)>1e-6 else 0:>16.2f}  {interp:<40s}")

    # --- Conclusion -------------------------------------------------------
    print("\n" + "=" * 88)
    print("Conclusion")
    print("=" * 88)
    n3_amp = abs(agg_moments[3])
    n6_amp = abs(agg_moments[6])
    n9_amp = abs(agg_moments[9])
    non_c3 = max(abs(agg_moments[n]) for n in [1, 2, 4, 5, 7, 8, 10, 11])
    print(f"""
  Aggregate Q-space density Fourier signature:

    |M_0|  = 1.0000    (normalization)
    |M_3|  = {n3_amp:.4f}    (C_3 fundamental)
    |M_6|  = {n6_amp:.4f}    (C_3 first harmonic)
    |M_9|  = {n9_amp:.4f}    (C_3 second harmonic)
    max non-C_3 mode amplitude = {non_c3:.4f}
""")
    if n3_amp > 5 * non_c3:
        print(f"  → C_3 modulation DOMINATES the structure (|M_3| ≫ non-C_3 modes).")
        print(f"    The Q-space density is genuinely C_3-graded, not just noise.")
    else:
        print(f"  → C_3 modulation is comparable to non-C_3 modes.")
        print(f"    Structure is more complex than pure C_3.")

    # Reproducing canonical signature?
    print(f"\n  Compared to canonical (4,4,4,4)-at-saddle expected signature:")
    print(f"    Aggregate |M_3| / Expected |M_3| = {n3_amp / abs(expected[3]):.4f}" if abs(expected[3]) > 1e-9 else "    Expected |M_3| ≈ 0")
    print(f"    Aggregate |M_6| / Expected |M_6| = {n6_amp / abs(expected[6]):.4f}" if abs(expected[6]) > 1e-9 else "    Expected |M_6| ≈ 0")
    print(f"\n  Per-substrate dominant modes:")
    for name in Q_SPACE_SUBSTRATES:
        moms = per_sub_moments[name]
        sorted_ms = sorted(range(1, 13), key=lambda n: -abs(moms[n]))
        top3 = sorted_ms[:3]
        print(f"    {name:<10s}: top non-zero modes = {[(n, f'|M|={abs(moms[n]):.3f}') for n in top3]}")


if __name__ == '__main__':
    main()
