#!/usr/bin/env python3
"""
Investigation #2 K-convergence — does the substrate ↔ dark coefficient
mapping sharpen to exact identity at finer k-grid?

Investigation #2 found at K_GRID=5:
    srs-c8  → 17/24 (multi-edge primitive)        within 2.1%
    lou     → √5/4 (m_ν family)                   within 9.1%
    srs     → 5/12 (V_us)                         within 12.6%
    srs-c4  → 1/3 (Ω_Λ = 1/k*)                    within 13.7%

If these matches are STRUCTURAL IDENTITIES, finer k-grid (more eigenvalues
sampled per substrate) should drive them toward 0% offset. If they
stabilize at finite percentage, the mapping is approximate (substrate-
level encoding plus per-substrate noise floor).

This probe runs the ledger at K_GRID = 5, 8, 10 in succession and tracks
the convergence of each substrate's Im(Σ_emp)/α_1 against the closest
known framework dark coefficient.
"""

import sys, os, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges, SG_NAME_TO_HALL,
)

LEDGER = ['srs-z', 'srs-c4', 'hcb-c4', 'srs-c27', 'srs', 'lou', 'srs-c8', 'okw', 'lov']
K_GRIDS = [10, 12]   # converged regime; K=12 = 1728 k-pts to settle loosening
RAMANUJAN_RADIUS_SQ = 2.0
TOLERANCE = 0.05
H_SADDLE = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
ALPHA_1_BARE = (2 / 3) ** 8

# Framework dark coefficients — extended list. Includes:
#   - canonical dark coeffs already in framework predictions
#   - geometric ratios from saddle h = (√3+i√5)/2
#   - simple rationals likely to emerge as substrate weights
#   - V_us, V_cb, etc. directly
DARK_COEFFS = {
    # Canonical dark coefficients used in predictions/
    '5/12 (V_us, m_H)':       5/12,
    '√5/4 (m_ν, Im(h)/|h|²)': math.sqrt(5)/4,
    '1/3 (Ω_Λ=1/k*)':         1/3,
    '7/40':                    7/40,
    '17/24 (multi-edge prim)': 17/24,
    # Geometric-from-saddle constants
    '√5/8':                    math.sqrt(5)/8,
    '√3/4 (Re(h)/|h|²)':      math.sqrt(3)/4,
    '√3/8':                    math.sqrt(3)/8,
    '√5/2 (Im(h))':            math.sqrt(5)/2,
    '√3/2 (Re(h))':            math.sqrt(3)/2,
    'arg(h)/π':                math.atan2(math.sqrt(5), math.sqrt(3)) / math.pi,
    # Simple rationals
    '1/2':  0.5,
    '1/4':  0.25,
    '1/6':  1/6,
    '1/8':  1/8,
    '1/9 (1/k*²)':  1/9,
    '1/12': 1/12,
    '2/3':  2/3,
    '3/4':  0.75,
    '5/6':  5/6,
    '7/8':  7/8,
    # Specific framework predictions / V_us etc.
    '9/40 (V_us=)':    9/40,
    '11/24':           11/24,
    '13/24':           13/24,
    '19/24':           19/24,
    # Combinations
    '1 - √5/4':        1 - math.sqrt(5)/4,
    '1 - √3/4':        1 - math.sqrt(3)/4,
    'sin(π/10)':       math.sin(math.pi/10),
    'sin(π/12)':       math.sin(math.pi/12),
    'cos(arg h)':      math.cos(math.atan2(math.sqrt(5), math.sqrt(3))),  # = √3/√8
    'sin(arg h)':      math.sin(math.atan2(math.sqrt(5), math.sqrt(3))),  # = √5/√8
    '1/k*' :           1/3,  # alias
    '2/k*' :           2/3,
    # Framework-internal Feshbach exponents
    '(2/3)^2': (2/3)**2,
    '(2/3)^3': (2/3)**3,
    '(2/3)^4': (2/3)**4,
}


def setup_substrate(name, entry):
    """Build arcs once; reused across k-grid scans."""
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
    return arcs, n_atoms


def collect_eigs_at_k(arcs, n_atoms, K):
    eigs = []
    for i in range(K):
        for j in range(K):
            for k in range(K):
                k_pt = np.array([i / K, j / K, k / K])
                B = bloch_hashimoto(arcs, k_pt, n_atoms)
                evs = np.linalg.eigvals(B)
                for lam in evs:
                    if abs(abs(lam)**2 - RAMANUJAN_RADIUS_SQ) < TOLERANCE:
                        eigs.append(complex(lam))
    return eigs


def sigma_emp(eigs, h, alpha_1):
    if not eigs: return 0.0 + 0.0j
    valid = [1.0/(h-lam) for lam in eigs if abs(h-lam) > 1e-9]
    if not valid: return 0.0 + 0.0j
    return alpha_1 * sum(valid) / len(valid)


def closest_match(value, coeffs):
    best = None
    best_off = float('inf')
    for cname, cval in coeffs.items():
        if cval == 0: continue
        off = abs(value - cval) / cval
        if off < best_off:
            best_off = off
            best = (cname, cval, off)
    return best


def main():
    print("=" * 100)
    print("INVESTIGATION #2 K-CONVERGENCE — substrate ↔ dark coefficient sharpening")
    print("=" * 100)
    print(f"\n  Saddle h = (√3+i√5)/2 = {H_SADDLE},  |h|² = {abs(H_SADDLE)**2:.4f}")
    print(f"  α_1 = (2/3)^8 = {ALPHA_1_BARE:.6f}")
    print(f"  Ledger: {LEDGER}")
    print(f"  K_GRID values to test: {K_GRIDS}")
    print(f"  Total k-pts per substrate: {[K**3 for K in K_GRIDS]}")

    # Setup substrates once (bond reconstruction is the slow part)
    print("\n  --- Building arcs (one-time setup per substrate) ---")
    entries = parse_rcsr_3dall('/tmp/rcsr_3d_current.txt', LEDGER)
    setups = {}
    for name in LEDGER:
        if name not in entries:
            print(f"    [{name}: missing from parser]")
            continue
        t0 = time.time()
        s = setup_substrate(name, entries[name])
        dt = time.time() - t0
        if s is None:
            print(f"    [{name}: setup failed ({dt:.1f}s)]")
            continue
        arcs, n_atoms = s
        setups[name] = (arcs, n_atoms)
        print(f"    [{name}: {n_atoms} atoms, {len(arcs)} arcs, setup {dt:.1f}s]", flush=True)

    # Run at each K_GRID
    print("\n  --- Eigenvalue collection + Σ_emp at each K_GRID ---")
    all_results = {}  # (name, K) → (eigs, sigma)
    for K in K_GRIDS:
        print(f"\n  K_GRID = {K} ({K**3} k-pts each):")
        for name, (arcs, n_atoms) in setups.items():
            t0 = time.time()
            eigs = collect_eigs_at_k(arcs, n_atoms, K)
            sigma = sigma_emp(eigs, H_SADDLE, ALPHA_1_BARE)
            dt = time.time() - t0
            all_results[(name, K)] = (eigs, sigma)
            print(f"    {name:<10s}: N_eigs={len(eigs):>6d}  "
                  f"Σ_emp = {sigma.real:+.5f} {sigma.imag:+.5f}i  "
                  f"|Σ|={abs(sigma):.5f}  Im/α_1={-sigma.imag/ALPHA_1_BARE:+.4f}  "
                  f"[{dt:.1f}s]", flush=True)

    # Convergence table per substrate
    print("\n" + "=" * 100)
    print("CONVERGENCE OF Im(Σ)/α_1 ACROSS K_GRID")
    print("=" * 100)
    header_K = "  ".join(f"K={K:<2d}({'Im/α_1':<10s} off%)" for K in K_GRIDS)
    print(f"\n  {'substrate':<10s}  {header_K}  best match @ K={K_GRIDS[-1]}")
    convergence = []
    for name in LEDGER:
        if name not in setups: continue
        row_pieces = []
        last_off = None
        last_match = None
        for K in K_GRIDS:
            _, sig = all_results[(name, K)]
            ratio = -sig.imag / ALPHA_1_BARE
            match = closest_match(ratio, DARK_COEFFS)
            if match is None:
                row_pieces.append(f"{ratio:>+10.4f} (---)")
            else:
                cname, cval, off = match
                row_pieces.append(f"{ratio:>+10.4f} {off*100:>5.1f}%")
                if K == K_GRIDS[-1]:
                    last_match = match
        row_str = "    ".join(row_pieces)
        last_label = (f"{last_match[0]} = {last_match[1]:.4f}, off {last_match[2]*100:.2f}%"
                      if last_match else "—")
        print(f"  {name:<10s}    {row_str}    {last_label}")
        if last_match:
            convergence.append((name, last_match[0], [closest_match(-all_results[(name, K)][1].imag / ALPHA_1_BARE, DARK_COEFFS) for K in K_GRIDS]))

    # Sharpness analysis: did matches tighten?
    print("\n" + "-" * 100)
    print("SHARPNESS ANALYSIS — did matches tighten at finer K?")
    print("-" * 100)
    print(f"\n  For each substrate, list off% at each K_GRID. ↓ = tightened, ↑ = loosened, ≈ = stable.\n")
    print(f"  {'substrate':<10s}  {'best match':<22s}  off%@K=" + "  ".join(f"{K:<3d}" for K in K_GRIDS) + "  trend")
    for name, match_name, per_k_matches in convergence:
        offs = [m[2]*100 if m else None for m in per_k_matches]
        # Compute trend: ratio of last to first
        if offs[0] is None or offs[-1] is None:
            trend = "?"
        elif offs[-1] < 0.5 * offs[0]:
            trend = "↓↓ (tightening; structural identity)"
        elif offs[-1] < 0.8 * offs[0]:
            trend = "↓ (mild tightening)"
        elif offs[-1] > 1.5 * offs[0]:
            trend = "↑ (loosening; not converged)"
        else:
            trend = "≈ (stable; floor reached)"
        off_str = "  ".join(f"{o:>5.1f}" if o is not None else "  ---" for o in offs)
        print(f"  {name:<10s}  {match_name:<22s}  {off_str}  {trend}")

    # Real-part analysis
    print("\n" + "-" * 100)
    print("REAL-PART ANALYSIS — a separate private derivation by the author have nonzero Re(Σ); test against real-valued coeffs")
    print("-" * 100)
    print(f"\n  {'substrate':<10s}  Re(Σ)/α_1@K=" + "  ".join(f"{K:<3d}" for K in K_GRIDS) + "  best @ K_max")
    for name in LEDGER:
        if name not in setups: continue
        re_pieces = []
        for K in K_GRIDS:
            _, sig = all_results[(name, K)]
            ratio = sig.real / ALPHA_1_BARE
            re_pieces.append(f"{ratio:>+8.4f}")
        re_str = "  ".join(re_pieces)
        last_re = all_results[(name, K_GRIDS[-1])][1].real / ALPHA_1_BARE
        match = closest_match(last_re, DARK_COEFFS)
        last_label = f"{match[0]} (off {match[2]*100:.1f}%)" if match else "—"
        print(f"  {name:<10s}  {re_str}  {last_label}")

    # Verdict
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    n_tightening = sum(1 for name, _, m in convergence
                       if m[0] is not None and m[-1] is not None and m[-1][2] < 0.5 * m[0][2])
    n_stable = sum(1 for name, _, m in convergence
                   if m[0] is not None and m[-1] is not None
                   and 0.5 * m[0][2] <= m[-1][2] <= 1.5 * m[0][2])
    n_loosening = sum(1 for name, _, m in convergence
                      if m[0] is not None and m[-1] is not None and m[-1][2] > 1.5 * m[0][2])
    print(f"\n  Of {len(convergence)} substrate ↔ coefficient matches:")
    print(f"    {n_tightening} tightening (matches improving with K — structural identity)")
    print(f"    {n_stable} stable (matches at noise floor)")
    print(f"    {n_loosening} loosening (matches degrading)")

    if n_tightening >= len(convergence) // 2:
        print("\n  ✓ CONVERGENCE CONFIRMED — matches tighten at finer K, supporting structural identity.")
    elif n_stable >= len(convergence) // 2:
        print("\n  ◐ MATCHES AT NOISE FLOOR — close-but-not-exact; either (a) finite-N convergence")
        print("    is slow, (b) matches are approximate (with structural origin to investigate),")
        print("    or (c) matches reflect partial substrate↔coefficient encoding.")
    else:
        print("\n  ✗ MATCHES NOT CONVERGING — substrate↔coefficient hypothesis weakens.")


if __name__ == '__main__':
    main()
