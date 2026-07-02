#!/usr/bin/env python3
"""
Investigation #3 — Extended Q-space probe over the framework's full 9-substrate
ledger.

Per a separate private derivation by the author (a separate private derivation by the author §4c.3):
at MDL optimum, Q-space spectral density on the Ramanujan circle is uniform
because any structured peak would be extracted into P-space.

Investigation #1 (`q_space_c3_fourier_decomposition.py`, 2026-05-02) showed
that the 3-substrate Q-space {srs-z, srs-c4, hcb-c4} has |M_2| ≈ 0.27
DOMINANT and |M_3| ≈ 10⁻⁴ near zero. The C_3 mode IS water-filled to
near-zero, but the cos(2φ) mode is NOT.

This probe asks the sharper question:

    Does the cumulative Q-space density approach uniform as we aggregate
    ALL 9 framework-ledger substrates (3 waterline-excluded + 2 iso-redundant
    + 4 survivors), and do per-substrate Fourier modes ADD coherently
    (→ no water-filling) or AVERAGE to zero (→ water-filling holds)?

If individual M_n values randomize across substrates, |⟨M_n⟩|/N drops as
1/√N and the cumulative density tends to uniform — water-filling theorem
is asymptotically supported.

If individual M_n values CONSTRUCTIVELY align in sign across substrates,
the cumulative |⟨M_n⟩|/N stays O(1) and density does NOT approach uniform —
water-filling theorem is contradicted in this framework.

Method:
  1. For each of 9 ledger substrates, compute Bloch-Hashimoto eigenvalues
     at K_GRID=6 (216 k-points).
  2. Filter to Ramanujan-saturated (|λ|² ≈ 2 within 1e-3).
  3. Compute per-substrate Fourier moments M_n = (1/N_eigs) Σ_λ exp(-i n φ_λ)
     for n = 0, 2, 3, 4, 6, 8, 10, 12 — note M_n is real for Hermitian B
     (λ↔λ̄ symmetry kills imaginary part).
  4. Compute cumulative ⟨M_n⟩ as substrates are added one-by-one.
  5. Test scaling: does |⟨M_n⟩_cumulative| × √N stay O(1) (random walk =
     water-filling) or does ⟨M_n⟩ stay nonzero asymptotically (coherent =
     no water-filling)?

Compares vs:
  - a separate private derivation by the author uniform-ansatz Σ = α₁·h̄/|h|² = 0.0218 ·  Im[Σ]
  - empirical cumulative discrete Σ
"""

import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges, SG_NAME_TO_HALL,
)

# Framework ledger (per `rcsr_survivors_full_ledger_walk.py`)
WATERLINE_EXCLUDED = ['srs-z', 'srs-c4', 'hcb-c4']  # a separate private derivation by the author Q-space
ISO_EXCLUDED       = ['srs-c27', 'okw']             # ≡ srs, ≡ lou
SURVIVORS          = ['srs', 'srs-c8', 'lou', 'lov']
LEDGER             = WATERLINE_EXCLUDED + ISO_EXCLUDED + SURVIVORS

K_GRID_RES = 6                # 6^3 = 216 k-points
RAMANUJAN_RADIUS_SQ = 2.0
H_SADDLE = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
ALPHA_1_BARE = (2 / 3) ** 8
N_BINS = 12
FOURIER_MODES = [0, 2, 3, 4, 6, 8, 10, 12]


def collect_ramanujan_eigs(name, entry, tol=1e-3):
    sg = entry['sg_name']
    if sg not in SG_NAME_TO_HALL: return None
    rotations, translations, _, _ = get_space_group_ops(sg)
    v_frac = np.array(entry['vertex_orbits'][0]['cartesian'])
    atom_orbit = orbit_of(v_frac, rotations, translations)
    midpoints = []
    for eo in entry['edge_orbits']:
        midpoints.append(orbit_of(np.array(eo['cartesian']), rotations, translations))
    if not midpoints: return None
    midpoint_orbit = np.vstack(midpoints)
    bonds = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=3)
    bonds = [b for b in bonds if b is not None]
    if not bonds: return None
    arcs = build_directed_edges(bonds)
    n_atoms = len(atom_orbit)
    if not arcs: return None
    print(f"    [{name}: {n_atoms} atoms, {len(arcs)} arcs, sampling {K_GRID_RES**3} k-pts...]", flush=True)
    eigs = []
    for i in range(K_GRID_RES):
        for j in range(K_GRID_RES):
            for k in range(K_GRID_RES):
                k_pt = np.array([i / K_GRID_RES, j / K_GRID_RES, k / K_GRID_RES])
                B = bloch_hashimoto(arcs, k_pt, n_atoms)
                evs = np.linalg.eigvals(B)
                for lam in evs:
                    if abs(abs(lam) ** 2 - RAMANUJAN_RADIUS_SQ) < tol:
                        eigs.append(complex(lam))
    print(f"      → {len(eigs)} Ramanujan-saturated eigs", flush=True)
    return eigs


def fourier_modes(eigs, modes):
    """M_n = (1/N) Σ_λ exp(-i n φ_λ).  For Hermitian B with λ↔λ̄, M_n is real."""
    if not eigs: return {n: 0.0 for n in modes}
    N = len(eigs)
    out = {}
    for n in modes:
        s = sum(np.exp(-1j * n * math.atan2(e.imag, e.real)) for e in eigs)
        out[n] = (s / N).real
    return out


def histogram_density(eigs, n_bins):
    counts = [0] * n_bins
    for e in eigs:
        phi = math.atan2(e.imag, e.real)  # in [-π, π]
        idx = int((phi + math.pi) / (2 * math.pi) * n_bins) % n_bins
        counts[idx] += 1
    total = sum(counts) or 1
    return [c / total / (2 * math.pi / n_bins) for c in counts]


def chi_sq_uniform(density):
    n = len(density)
    rho_unif = 1.0 / (2 * math.pi)
    bin_w = 2 * math.pi / n
    # Pearson-style chi²: Σ (O - E)² / E with O,E as counts not densities;
    # express via density × normalization.
    return sum((d - rho_unif) ** 2 / rho_unif for d in density) * bin_w


def sigma_from_eigs(eigs, h, alpha_1):
    if not eigs: return 0.0 + 0.0j
    s = sum(1.0 / (h - lam) for lam in eigs if abs(h - lam) > 1e-9)
    return alpha_1 * s / len(eigs)


def main():
    print("=" * 92)
    print("INVESTIGATION #3 — Extended Q-space probe over 9-substrate framework ledger")
    print("=" * 92)
    print(f"\n  Ledger ({len(LEDGER)}): {LEDGER}")
    print(f"  Sampling at K_GRID = {K_GRID_RES}^3 = {K_GRID_RES**3} k-points each")
    print(f"  Fourier modes computed: {FOURIER_MODES}")
    print()

    entries = parse_rcsr_3dall('/tmp/rcsr_3d_current.txt', LEDGER)
    per_sub_eigs = {}
    per_sub_modes = {}
    for name in LEDGER:
        if name not in entries:
            print(f"    [{name}: missing from parser]")
            continue
        eigs = collect_ramanujan_eigs(name, entries[name])
        if eigs is None or not eigs:
            print(f"    [{name}: no Ramanujan eigs at this k-grid; SKIPPED]")
            continue
        per_sub_eigs[name] = eigs
        per_sub_modes[name] = fourier_modes(eigs, FOURIER_MODES)

    if not per_sub_eigs:
        print("\nNo substrates produced eigenvalues. Exiting.")
        return

    # ------------------- Per-substrate Fourier mode table -------------------
    print("\n" + "-" * 92)
    print("PER-SUBSTRATE FOURIER MODES  (M_n real, signed)")
    print("-" * 92)
    header = f"  {'name':<10s} {'class':<14s} {'N_eigs':>7s} " + " ".join(f"{'M_'+str(n):>9s}" for n in FOURIER_MODES)
    print(header)
    classification = {}
    for n in WATERLINE_EXCLUDED: classification[n] = 'waterline-out'
    for n in ISO_EXCLUDED:       classification[n] = 'iso-redundant'
    for n in SURVIVORS:          classification[n] = 'survivor'
    for name in LEDGER:
        if name not in per_sub_eigs: continue
        m = per_sub_modes[name]
        modestr = " ".join(f"{m[n]:>+9.4f}" for n in FOURIER_MODES)
        print(f"  {name:<10s} {classification[name]:<14s} {len(per_sub_eigs[name]):>7d} {modestr}")

    # Sign-pattern analysis: for each mode, count signs across substrates.
    print("\n  --- Sign pattern analysis (does M_n alternate or align?) ---")
    for n in FOURIER_MODES:
        if n == 0: continue
        signs = [(name, per_sub_modes[name][n]) for name in per_sub_eigs]
        n_pos = sum(1 for _, v in signs if v > 0.01)
        n_neg = sum(1 for _, v in signs if v < -0.01)
        n_zero = sum(1 for _, v in signs if abs(v) <= 0.01)
        verdict = ("aligned (CONSTRUCTIVE)" if n_pos == 0 or n_neg == 0
                   else "mixed (toward cancellation)" if min(n_pos, n_neg) >= 2
                   else "biased (PARTIAL coherence)")
        print(f"    M_{n:<2d}: +{n_pos}  −{n_neg}  ~0:{n_zero}   → {verdict}")

    # ------------------- Cumulative aggregation -------------------
    print("\n" + "-" * 92)
    print("CUMULATIVE AGGREGATION  (substrates added in ledger order)")
    print("-" * 92)
    sigma_uniform = ALPHA_1_BARE * H_SADDLE.conjugate() / abs(H_SADDLE)**2
    print(f"  a separate private derivation by the author uniform Σ = α₁·h̄/|h|² = {sigma_uniform.real:+.5f} {sigma_uniform.imag:+.5f}i  (|Σ|={abs(sigma_uniform):.5f})\n")
    cum_eigs = []
    print(f"  {'after':<26s} {'N_eigs':>7s} {'chi²(unif)':>11s} {'M_2_cum':>9s} {'M_4_cum':>9s} {'|Σ|_cum':>10s} {'Im(Σ)_cum':>12s}")
    for name in LEDGER:
        if name not in per_sub_eigs: continue
        cum_eigs.extend(per_sub_eigs[name])
        modes_cum = fourier_modes(cum_eigs, FOURIER_MODES)
        dens = histogram_density(cum_eigs, N_BINS)
        chi = chi_sq_uniform(dens)
        sig = sigma_from_eigs(cum_eigs, H_SADDLE, ALPHA_1_BARE)
        print(f"  +{name:<25s} {len(cum_eigs):>7d} {chi:>11.5f} {modes_cum[2]:>+9.4f} {modes_cum[4]:>+9.4f} {abs(sig):>10.5f} {sig.imag:>+12.5f}")

    # ------------------- Random-phase scaling test -------------------
    print("\n" + "-" * 92)
    print("RANDOM-PHASE SCALING TEST")
    print("-" * 92)
    print("  If per-substrate M_n is a random ±|M_n| draw, cumulative ⟨M_n⟩")
    print("  should scale as 1/√N (water-filling holds asymptotically).")
    print("  If aligned in sign, cumulative ⟨M_n⟩ stays O(typical M_n).")
    print()
    for n in [2, 4, 6, 8, 10]:
        if n not in FOURIER_MODES: continue
        per_sub_vals = [per_sub_modes[name][n] for name in per_sub_eigs]
        mean = np.mean(per_sub_vals)
        absmean = np.mean([abs(v) for v in per_sub_vals])
        N_subs = len(per_sub_vals)
        # If random ±, expect |mean| ≈ absmean / √N
        random_walk_expect = absmean / math.sqrt(N_subs)
        ratio = abs(mean) / random_walk_expect if random_walk_expect > 0 else float('inf')
        diag = "ALIGNED → no water-fill" if ratio > 1.5 else "RANDOM → water-fill OK" if ratio < 0.7 else "MIXED"
        print(f"  M_{n:<2d}: ⟨M⟩={mean:+.4f}  ⟨|M|⟩={absmean:.4f}  random-walk expect ~{random_walk_expect:.4f}"
              f"  ratio={ratio:.2f}  → {diag}")

    # ------------------- Verdict -------------------
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    final_modes = fourier_modes(cum_eigs, FOURIER_MODES)
    final_dens = histogram_density(cum_eigs, N_BINS)
    final_chi = chi_sq_uniform(final_dens)
    final_sig = sigma_from_eigs(cum_eigs, H_SADDLE, ALPHA_1_BARE)
    print(f"\n  Final cumulative ({len(per_sub_eigs)} substrates, {len(cum_eigs)} eigs):")
    print(f"    chi²(uniform):                {final_chi:.5f}")
    print(f"    Final M_2:                    {final_modes[2]:+.4f}")
    print(f"    Final M_3 (C_3 fundamental):  {final_modes[3]:+.4f}")
    print(f"    Final M_4:                    {final_modes[4]:+.4f}")
    print(f"    |Σ_emp|/|Σ_unif|:             {abs(final_sig)/abs(sigma_uniform):.4f}")
    print(f"    Im(Σ_emp):                    {final_sig.imag:+.5f}  (a separate private derivation by the author: {sigma_uniform.imag:+.5f})")
    print()
    if abs(final_modes[2]) < 0.05 and abs(final_modes[4]) < 0.05 and final_chi < 0.005:
        print("  ✓ All Fourier modes water-filled to near-zero across full ledger.")
        print("    a separate private derivation by the author CONFIRMED for the framework's ledger.")
    elif abs(final_modes[2]) < 0.10:
        print("  ◐ M_2 partially filled (|M_2| < 0.1) but not fully.")
        print("    a separate private derivation by the author holds approximately; small residual coherent structure.")
    else:
        print("  ✗ Significant residual M_2 (or other even mode) survives full ledger aggregation.")
        print("    a separate private derivation by the author does NOT hold rigorously in this framework.")
        print("    The Q-space density retains coherent cos(nφ) structure across substrates.")


if __name__ == '__main__':
    main()
