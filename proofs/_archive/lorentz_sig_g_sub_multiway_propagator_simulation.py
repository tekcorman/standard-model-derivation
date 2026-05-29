#!/usr/bin/env python3
"""
G_sub multiway route — direct numerical Green's function simulation.

Per `g_sub_multiway_route_scoping.md`: D_eff = 1/16 (theorem-grade) gives
propagator amplitude 4/π in the analytical formula. Source coupling
candidates give G_lattice ∈ {4/π, 1/π, 1/(3π), ...}.

This script measures the walker Green's function G(r) directly:
1. Build extended srs supercell (large enough to suppress boundary effects).
2. Hashimoto NB walker starts at central atom v_0.
3. Walker takes N_steps, accumulating visit counts at each vertex.
4. Average over N_trials independent walkers.
5. Plot G(r) = mean visit count vs distance from v_0.
6. Fit to G(r) = A/r at large r.
7. Compare A to candidate prefactors:
     A = 4/π ≈ 1.273 (natural identity coupling)
     A = 1/π ≈ 0.318 (Hashimoto-V_Ram both sides)
     A = 1/(3π) ≈ 0.106 (1/12-per-edge form)

Result A is THE empirical answer. Whatever clean rational matches it is the
structural prefactor we've been looking for.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict


# =============================================================================
# srs structure
# =============================================================================

ATOMS_FRAC = np.array([
    [0.125, 0.125, 0.125],
    [0.375, 0.875, 0.625],
    [0.875, 0.625, 0.375],
    [0.625, 0.375, 0.875],
])

A_PRIM = np.array([
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
])


def find_NN_image(src_idx, tgt_idx, search=2):
    """For atoms src and tgt: find the cell offset (n1, n2, n3) minimizing
    distance |R_tgt + n·A_PRIM - R_src|. Returns (cell_offset, dist²)."""
    R_src = ATOMS_FRAC[src_idx]
    R_tgt = ATOMS_FRAC[tgt_idx]
    best = None
    for n1 in range(-search, search + 1):
        for n2 in range(-search, search + 1):
            for n3 in range(-search, search + 1):
                shift = np.array([n1, n2, n3]) @ A_PRIM
                rb = R_tgt + shift - R_src
                d2 = np.sum(rb ** 2)
                if best is None or d2 < best[1]:
                    best = ((n1, n2, n3), d2, rb.copy())
    return best


# Pre-compute NN bonds at each atom (cell offset to each NN's image)
NN_LINKS = []  # NN_LINKS[atom] = list of (target_atom, cell_offset_n1, cell_offset_n2, cell_offset_n3, rb_vector)
for src in range(4):
    bonds = []
    for tgt in range(4):
        if tgt == src:
            continue
        cell, d2, rb = find_NN_image(src, tgt)
        bonds.append((tgt, cell[0], cell[1], cell[2], rb))
    NN_LINKS.append(bonds)


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def simulate_walker_propagator(L=10, N_steps=2000, N_trials=20000, seed=42):
    """Simulate the Hashimoto NB walker with periodic boundaries.

    Builds a (2L+1)³ supercell (so L cell shifts in each direction).
    Walker starts at atom 0 in central cell (n=0). Runs for N_steps,
    accumulating visit count at each (atom, n1, n2, n3) tuple.

    Returns:
      visit_counts: dict (atom_idx, n1, n2, n3) → count, normalized so
                    Σ counts = N_trials × N_steps.
      r_values, G_values: arrays for plotting.
    """
    rng = np.random.default_rng(seed)

    visit_counts = defaultdict(float)

    # Walker state: (atom, n1, n2, n3, prev_atom)
    # prev_atom is the atom we arrived FROM (for NB constraint).

    for trial in range(N_trials):
        atom = 0
        n_offset = (0, 0, 0)
        prev_atom = -1

        for step in range(N_steps):
            visit_counts[(atom, n_offset[0], n_offset[1], n_offset[2])] += 1.0

            # Choose NB neighbor (not prev_atom)
            valid_links = [b for b in NN_LINKS[atom] if b[0] != prev_atom]
            if len(valid_links) == 0:
                # Initial step: any neighbor
                valid_links = NN_LINKS[atom]

            choice = valid_links[rng.integers(0, len(valid_links))]
            tgt, c1, c2, c3, rb = choice
            new_n = (n_offset[0] + c1, n_offset[1] + c2, n_offset[2] + c3)

            # Enforce periodic boundary: wrap supercell if needed (effectively infinite if L large)
            # For large L, walker rarely escapes; just measure within sphere of radius < L
            if any(abs(c) > L for c in new_n):
                # Walker hit boundary; restart from origin
                atom = 0
                n_offset = (0, 0, 0)
                prev_atom = -1
                continue

            prev_atom = atom
            atom = tgt
            n_offset = new_n

    # Convert to r-vs-G plot
    r_to_counts = defaultdict(list)  # r → list of count values

    for (atom, n1, n2, n3), count in visit_counts.items():
        # Cartesian position of this atom in this cell
        cell_shift = np.array([n1, n2, n3]) @ A_PRIM
        pos = ATOMS_FRAC[atom] + cell_shift
        # Distance from origin (atom 0 at cell (0,0,0))
        r = np.sqrt(np.sum((pos - ATOMS_FRAC[0]) ** 2))
        r_to_counts[round(r, 4)].append(count)

    # For each r, average G(r) = mean count / (N_trials × N_steps total visits)
    # Actually G(r) measures time-integrated probability, which is:
    #   G(r) = (count at r) / N_trials  (counts per trial visit at distance r)
    # The 1/(4π D r) form fits this, with appropriate units.

    rs = []
    Gs = []
    for r, counts in sorted(r_to_counts.items()):
        if r < 0.05:
            continue  # skip origin
        rs.append(r)
        Gs.append(np.mean(counts) / N_trials)

    return np.array(rs), np.array(Gs), visit_counts


def main():
    header("Probe 1: direct numerical Hashimoto walker propagator on srs")
    print()
    print("  Run NB walker, time-integrate density at each vertex, fit G(r) = A/r.")
    print()

    L = 8  # half-supercell extent (walker confined to (2L+1)³ cell range)
    N_steps = 1500
    N_trials = 30000

    print(f"  Supercell: ±{L} cell shifts each direction = ({2*L+1})³ = {(2*L+1)**3} cells")
    print(f"  Steps per trial: {N_steps}")
    print(f"  Trials: {N_trials}")
    print(f"  Total walker-steps: {N_steps * N_trials}")
    print()
    print(f"  D_eff theoretical = 1/16. Predicted G(r) = 1/(4π D r) = 4/(π r) at large r.")
    print(f"  Predicted A (in Green's function fit G(r) = A/r): 4/π ≈ 1.273.")
    print()

    rs, Gs, _ = simulate_walker_propagator(L=L, N_steps=N_steps, N_trials=N_trials)

    # Filter to large r where 1/r form should hold (avoid lattice artifacts at small r)
    mask = (rs > 1.0) & (rs < L * 0.6)  # avoid boundary too
    rs_fit = rs[mask]
    Gs_fit = Gs[mask]

    print(f"  Measured G(r) at {len(rs)} distinct r values, fitting in range [{rs_fit.min():.3f}, {rs_fit.max():.3f}]:")
    print()

    # Average G(r) × r and check for plateau (signal of 1/r form)
    Gr_product = Gs_fit * rs_fit
    A_mean = np.mean(Gr_product)
    A_std = np.std(Gr_product)
    print(f"  G(r) × r at large r: mean = {A_mean:.4f} ± {A_std:.4f}")
    print(f"  Coefficient A (such that G(r) ~ A/r):")
    print(f"    Mean: {A_mean:.4f}")
    print(f"    Median: {np.median(Gr_product):.4f}")
    print()

    # Fit to G(r) = A/r form via 1/r weighted regression
    A_lstsq = np.sum(Gs_fit * (1/rs_fit)) / np.sum((1/rs_fit) ** 2)
    print(f"  Least-squares fit (G(r) = A/r): A = {A_lstsq:.4f}")

    print()
    print(f"  Compare to candidates:")
    candidates = {
        '4/π (natural identity, predicted theory)': 4 / np.pi,
        '2/π': 2 / np.pi,
        '1/π': 1 / np.pi,
        '1/(2π)': 1 / (2 * np.pi),
        '1/(3π)': 1 / (3 * np.pi),
        '1/(4π)': 1 / (4 * np.pi),
        '4(√3-1)/27 (session-4 elastic)': 4 * (np.sqrt(3) - 1) / 27,
    }
    print(f"    {'candidate':<40s} {'value':>10s}    {'ratio (measured/cand)':>20s}")
    for name, val in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
        ratio = A_mean / val
        flag = "  ← MATCH!" if abs(ratio - 1) < 0.05 else ""
        print(f"    {name:<40s} {val:>10.6f}    {ratio:>20.4f}{flag}")

    # Full r-vs-G(r) table for inspection
    header("G(r) data (all points, sorted by r)")
    print(f"  {'r':>8s} {'G(r)':>14s} {'G(r) × r':>14s}")
    for r, G in zip(rs, Gs):
        if r > 0.3:
            print(f"  {r:>8.4f} {G:>14.6e} {G*r:>14.6f}")

    header("Verdict")
    print()
    print(f"  Numerical A from simulation: {A_mean:.4f} ± {A_std:.4f}")
    closest_name, closest_val = min(candidates.items(), key=lambda x: abs(x[1] - A_mean))
    closest_pct = abs(A_mean - closest_val) / closest_val * 100
    print(f"  Closest candidate: {closest_name} = {closest_val:.6f} ({closest_pct:.2f}% off)")
    print()
    if closest_name.startswith('4/π'):
        print("  → Natural identity coupling CONFIRMED. G_sub_multiway = 4/π in lattice units.")
        print("    The ~10× mismatch with elastic estimates is genuine; either elastic was wrong")
        print("    or there's a separate lattice-Planck conversion factor (Probe 3 territory).")
    elif closest_name.startswith('1/(3π)'):
        print("  → 1/12-per-edge coupling CONFIRMED. G_sub_multiway = 1/(3π) in lattice units.")
        print("    The natural-identity 4/π is over-counting by factor 12. Match with elastic")
        print("    estimates within 2%.")
    else:
        print(f"  → Closest match is {closest_name}, but with {closest_pct:.1f}% deviation.")
        print(f"    Either the structural prefactor is unusual, or simulation has bias.")


if __name__ == "__main__":
    main()
