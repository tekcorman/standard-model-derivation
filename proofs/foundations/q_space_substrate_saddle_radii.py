#!/usr/bin/env python3
"""
Q-space Investigation #3-followup — substrate saddle-radius structural probe.

QUESTION: Does each ledger substrate have a characteristic Bloch-Hashimoto
saddle |λ|², and does the saddle radius distinguish waterline-failing
(a separate private derivation by the author "Q-space") substrates from survivors?

MOTIVATION: Investigation #3 found that 2 of 4 survivors (srs-c8, lov)
produced 0 Ramanujan-saturated (|λ|²=2) eigenvalues at K_GRID=6, while
all 3 waterline-failing substrates plus iso-redundants saturated heavily.
Memory already records `lov has saddle |λ|² = 5 ≠ 2`. This probe tests
whether the pattern is structural across the full ledger.

If YES (waterline-failing all saturate at 2; survivors have varying
saddles), then **Ramanujan-saturation = spectral characterization of
the MDL-A2 waterline test** — a structural upgrade for the framework.

METHOD:
  1. For each of 9 ledger substrates, sample B(k) at K_GRID=5 (125 k-pts).
  2. Compute full eigenvalue distribution.
  3. Report: max |λ|, max |λ|², histogram of |λ|² in [0, 6].
  4. Identify dominant spectral peaks (modes where eig density concentrates).
  5. Tabulate: substrate | class | max|λ|² | n_eigs at |λ|²=2 (±0.05) | other peaks.
  6. Test the structural hypothesis.
"""

import sys, os, math
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges, SG_NAME_TO_HALL,
)

WATERLINE_EXCLUDED = ['srs-z', 'srs-c4', 'hcb-c4']
ISO_EXCLUDED       = ['srs-c27', 'okw']
SURVIVORS          = ['srs', 'srs-c8', 'lou', 'lov']
LEDGER             = WATERLINE_EXCLUDED + ISO_EXCLUDED + SURVIVORS

K_GRID_RES = 5  # 5^3 = 125 k-points; balance speed vs spectral resolution
RAMANUJAN_RADIUS_SQ = 2.0
TOLERANCE = 0.05    # |λ|² within ±0.05 of target counts as "at" that radius


def collect_full_spectrum(name, entry):
    """Sample B(k) at K_GRID_RES^3 points; return all eigenvalues (complex)."""
    sg = entry['sg_name']
    if sg not in SG_NAME_TO_HALL: return None, None
    rotations, translations, _, _ = get_space_group_ops(sg)
    v_frac = np.array(entry['vertex_orbits'][0]['cartesian'])
    atom_orbit = orbit_of(v_frac, rotations, translations)
    midpoints = []
    for eo in entry['edge_orbits']:
        midpoints.append(orbit_of(np.array(eo['cartesian']), rotations, translations))
    if not midpoints: return None, None
    midpoint_orbit = np.vstack(midpoints)
    bonds = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=3)
    bonds = [b for b in bonds if b is not None]
    if not bonds: return None, None
    arcs = build_directed_edges(bonds)
    n_atoms = len(atom_orbit)
    if not arcs: return None, None
    print(f"    [{name}: {n_atoms} atoms, {len(arcs)} arcs, sampling {K_GRID_RES**3} k-pts...]", flush=True)
    all_eigs = []
    for i in range(K_GRID_RES):
        for j in range(K_GRID_RES):
            for k in range(K_GRID_RES):
                k_pt = np.array([i / K_GRID_RES, j / K_GRID_RES, k / K_GRID_RES])
                B = bloch_hashimoto(arcs, k_pt, n_atoms)
                evs = np.linalg.eigvals(B)
                all_eigs.extend(complex(lam) for lam in evs)
    return all_eigs, len(arcs)


def eigval_summary(eigs):
    """Return (max|λ|², histogram of |λ|², peak locations)."""
    abs2 = np.array([abs(e)**2 for e in eigs])
    max_abs2 = abs2.max()
    # Histogram on |λ|² in [0, max+0.5]
    bin_edges = np.arange(0, max(max_abs2 + 0.5, 6.0), 0.25)
    counts, edges = np.histogram(abs2, bins=bin_edges)
    # Find peaks: bins with count > 5% of total and local max
    total = abs2.size
    peaks = []
    for i in range(1, len(counts) - 1):
        if counts[i] >= 0.03 * total and counts[i] >= counts[i-1] and counts[i] >= counts[i+1]:
            peaks.append((edges[i] + 0.125, counts[i]))  # bin center, count
    return max_abs2, list(zip(edges[:-1], counts)), peaks


def count_at_radius(eigs, target_sq, tol):
    return sum(1 for e in eigs if abs(abs(e)**2 - target_sq) < tol)


def main():
    print("=" * 92)
    print("INVESTIGATION #3-followup — substrate saddle radii structural probe")
    print("=" * 92)
    print(f"\n  Ledger ({len(LEDGER)}): {LEDGER}")
    print(f"  Sampling at K_GRID = {K_GRID_RES}^3 = {K_GRID_RES**3} k-points each")
    print(f"  Tolerance for radius matching: ±{TOLERANCE}\n")

    classification = {}
    for n in WATERLINE_EXCLUDED: classification[n] = 'waterline-out'
    for n in ISO_EXCLUDED:       classification[n] = 'iso-redundant'
    for n in SURVIVORS:          classification[n] = 'survivor'

    entries = parse_rcsr_3dall('/tmp/rcsr_3d_current.txt', LEDGER)
    results = {}
    for name in LEDGER:
        if name not in entries:
            print(f"    [{name}: missing from parser]")
            continue
        eigs, n_arcs = collect_full_spectrum(name, entries[name])
        if eigs is None: continue
        max_abs2, hist, peaks = eigval_summary(eigs)
        n_at_2 = count_at_radius(eigs, 2.0, TOLERANCE)
        n_at_5 = count_at_radius(eigs, 5.0, TOLERANCE)
        n_at_4 = count_at_radius(eigs, 4.0, TOLERANCE)
        results[name] = {
            'class': classification[name],
            'n_arcs': n_arcs,
            'n_eigs_total': len(eigs),
            'max_abs2': max_abs2,
            'n_at_2': n_at_2,
            'n_at_4': n_at_4,
            'n_at_5': n_at_5,
            'peaks': peaks,
        }

    # ------------------- Saddle-radius table -------------------
    print("\n" + "-" * 92)
    print("SADDLE-RADIUS STRUCTURE  (counts at |λ|² = 2.0, 4.0, 5.0 within ±%.2f)" % TOLERANCE)
    print("-" * 92)
    print(f"  {'name':<10s} {'class':<14s} {'arcs':>4s} {'n_eigs':>7s} {'max|λ|²':>9s} "
          f"{'n@|λ|²=2':>10s} {'n@|λ|²=4':>10s} {'n@|λ|²=5':>10s}  {'pct@2':>6s}")
    for name in LEDGER:
        if name not in results: continue
        r = results[name]
        pct = 100 * r['n_at_2'] / r['n_eigs_total']
        print(f"  {name:<10s} {r['class']:<14s} {r['n_arcs']:>4d} {r['n_eigs_total']:>7d} "
              f"{r['max_abs2']:>9.3f} {r['n_at_2']:>10d} {r['n_at_4']:>10d} {r['n_at_5']:>10d}  {pct:>5.1f}%")

    # ------------------- Spectral peaks per substrate -------------------
    print("\n" + "-" * 92)
    print("DOMINANT SPECTRAL PEAKS  (|λ|² locations with ≥3% of eigenvalues)")
    print("-" * 92)
    for name in LEDGER:
        if name not in results: continue
        r = results[name]
        peak_str = ", ".join(f"|λ|²≈{p[0]:.2f}({p[1]} eigs)" for p in r['peaks'])
        print(f"  {name:<10s} ({r['class']}): {peak_str if peak_str else '(no clear peak)'}")

    # ------------------- Structural hypothesis test -------------------
    print("\n" + "=" * 92)
    print("STRUCTURAL HYPOTHESIS TEST")
    print("=" * 92)
    print("\n  Claim: waterline-failing substrates SATURATE the Ramanujan circle (|λ|²=2);")
    print("         survivors DO NOT saturate (have different saddle radii).\n")

    waterline_subs = [n for n in WATERLINE_EXCLUDED if n in results]
    survivor_subs = [n for n in SURVIVORS if n in results]
    iso_subs      = [n for n in ISO_EXCLUDED if n in results]

    waterline_saturates = all(results[n]['n_at_2'] > 100 for n in waterline_subs)
    survivors_saturate = [n for n in survivor_subs if results[n]['n_at_2'] > 100]
    survivors_dont = [n for n in survivor_subs if results[n]['n_at_2'] <= 100]

    print(f"  Waterline-out: {waterline_subs}")
    print(f"    All saturate at |λ|²=2? {waterline_saturates}")
    for n in waterline_subs:
        print(f"      {n}: {results[n]['n_at_2']} eigs at |λ|²=2 (max |λ|²={results[n]['max_abs2']:.2f})")

    print(f"\n  Survivors: {survivor_subs}")
    print(f"    Saturating subset: {survivors_saturate}")
    print(f"    Non-saturating subset: {survivors_dont}")
    for n in survivor_subs:
        print(f"      {n}: {results[n]['n_at_2']} eigs at |λ|²=2 (max |λ|²={results[n]['max_abs2']:.2f})")

    print(f"\n  Iso-redundant: {iso_subs}")
    for n in iso_subs:
        print(f"      {n}: {results[n]['n_at_2']} eigs at |λ|²=2 (max |λ|²={results[n]['max_abs2']:.2f})")

    print("\n" + "-" * 92)
    print("VERDICT")
    print("-" * 92)
    if waterline_saturates and len(survivors_dont) >= len(survivors_saturate):
        print("  ✓ STRUCTURAL HYPOTHESIS SUPPORTED:")
        print("    - All waterline-failing substrates saturate the Ramanujan circle at |λ|²=2.")
        print(f"    - {len(survivors_dont)} of {len(survivor_subs)} survivors do NOT saturate.")
        print("    Ramanujan-saturation appears to be a SPECTRAL MARKER of the MDL-A2 waterline.")
    elif waterline_saturates and not survivors_dont:
        print("  ◐ PARTIAL: waterline-failing all saturate, but survivors ALSO saturate.")
        print("    Ramanujan-saturation alone does NOT distinguish — need finer marker.")
    else:
        print("  ✗ HYPOTHESIS NOT SUPPORTED in this scope.")
        print("    Some waterline-failing substrates do NOT saturate, OR no survivor non-saturation.")


if __name__ == '__main__':
    main()
