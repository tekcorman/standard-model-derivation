#!/usr/bin/env python3
"""
Z3-twisted spectral gap of the SRS net and V_us prediction.

Question: does the Z3 twist shift the spectral gap from 2-sqrt(3) = 0.2679
to a value that gives |V_us| = 0.2248?

We need the gap to INCREASE from 0.2679 to ~0.297 (L_us decreases from 3.732 to ~3.68).
"""

import numpy as np
from numpy import linalg as la
from itertools import product
from collections import defaultdict

# Import the SRS net construction from the existing module
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srs_bloch_hamiltonian import (
    build_unit_cell, find_connectivity, assign_edge_labels,
    bloch_hamiltonian, twisted_bloch_hamiltonian,
    bloch_laplacian, twisted_laplacian,
)

# ===========================================================================
# Constants
# ===========================================================================
V_US_OBS = 0.2248
UNTWISTED_GAP = 2 - np.sqrt(3)       # 0.26795...
UNTWISTED_LUS = 2 + np.sqrt(3)       # 3.73205...
UNTWISTED_VUS = (2/3)**UNTWISTED_LUS  # 0.2200


def vus_from_gap(gap):
    """Compute |V_us| = (2/3)^(1/gap)."""
    if gap < 1e-12:
        return 0.0
    L = 1.0 / gap
    return (2.0/3.0) ** L


def inverse_vus(target_vus):
    """What gap gives target |V_us|? L = log(V_us)/log(2/3), gap = 1/L."""
    L = np.log(target_vus) / np.log(2.0/3.0)
    return 1.0 / L


def main():
    print("=" * 70)
    print("Z3-TWISTED SPECTRAL GAP AND V_us PREDICTION")
    print("=" * 70)

    # Required gap for V_us = 0.2248
    required_gap = inverse_vus(V_US_OBS)
    required_L = 1.0 / required_gap
    print(f"\nTarget: |V_us| = {V_US_OBS}")
    print(f"Required gap:  {required_gap:.6f}")
    print(f"Required L_us: {required_L:.6f}")
    print(f"\nUntwisted gap:  {UNTWISTED_GAP:.6f}")
    print(f"Untwisted L_us: {UNTWISTED_LUS:.6f}")
    print(f"Untwisted V_us: {UNTWISTED_VUS:.6f}")
    print(f"\nGap must {'INCREASE' if required_gap > UNTWISTED_GAP else 'DECREASE'} "
          f"by {abs(required_gap - UNTWISTED_GAP):.6f}")

    # =======================================================================
    # Step 1: Build SRS net
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 1: BUILD SRS NET")
    print("=" * 70)

    verts = build_unit_cell()
    n_verts = len(verts)
    bonds = find_connectivity(verts)
    edge_labels = assign_edge_labels(bonds, n_verts)

    print(f"\n  {n_verts} vertices, {len(bonds)} directed bonds")
    print(f"  Edge labels: {edge_labels}")

    # Verify: count labels
    from collections import Counter
    label_counts = Counter(edge_labels)
    print(f"  Label distribution: {dict(label_counts)}")

    # =======================================================================
    # Step 2: Untwisted Laplacian at Gamma (verification)
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 2: UNTWISTED LAPLACIAN (VERIFICATION)")
    print("=" * 70)

    k0 = np.array([0, 0, 0], dtype=float)
    L_untw = bloch_laplacian(k0, bonds, n_verts)
    evals_untw = np.sort(np.real(la.eigvalsh(L_untw)))
    print(f"\n  Untwisted Laplacian eigenvalues at Gamma:")
    for i, ev in enumerate(evals_untw):
        print(f"    lambda_{i} = {ev:.10f}")
    print(f"  Expected: 0 (x1), {UNTWISTED_GAP:.6f} = 2-sqrt(3) (x?), 4 (x?)")

    # Minimum nonzero = spectral gap
    nonzero = [e for e in evals_untw if e > 1e-10]
    if nonzero:
        gap_untw_gamma = min(nonzero)
        print(f"\n  Gamma-point spectral gap: {gap_untw_gamma:.10f}")
        print(f"  2 - sqrt(3) = {UNTWISTED_GAP:.10f}")

    # =======================================================================
    # Step 3: Z3-Twisted Laplacian at Gamma
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Z3-TWISTED LAPLACIAN AT GAMMA")
    print("=" * 70)

    omega_z3 = np.exp(2j * np.pi / 3)
    print(f"\n  omega = exp(2pi i/3) = {omega_z3:.6f}")

    L_tw = twisted_laplacian(k0, bonds, edge_labels, n_verts, omega_z3)
    evals_tw = np.sort(np.real(la.eigvalsh(L_tw)))
    print(f"\n  Z3-Twisted Laplacian eigenvalues at Gamma:")
    for i, ev in enumerate(evals_tw):
        print(f"    lambda_{i} = {ev:.10f}")

    gap_tw_gamma = min(evals_tw)
    print(f"\n  Smallest eigenvalue (twisted gap at Gamma): {gap_tw_gamma:.10f}")
    if gap_tw_gamma > 1e-10:
        L_tw_gamma = 1.0 / gap_tw_gamma
        vus_tw_gamma = (2.0/3.0) ** L_tw_gamma
        print(f"  L_us_tw = 1/gap = {L_tw_gamma:.10f}")
        print(f"  |V_us|_tw = (2/3)^L = {vus_tw_gamma:.10f}")
        print(f"  Observed |V_us|   = {V_US_OBS}")
        print(f"  Deviation: {(vus_tw_gamma - V_US_OBS)/V_US_OBS * 100:.4f}%")
    else:
        print(f"  WARNING: twisted gap ~ 0 at Gamma (unexpected)")

    # =======================================================================
    # Step 4: BZ scan for global minimum eigenvalue of L_tw(k)
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 4: BRILLOUIN ZONE SCAN FOR GLOBAL MINIMUM OF L_tw(k)")
    print("=" * 70)

    n_grid = 50
    print(f"\n  Scanning {n_grid}^3 = {n_grid**3} k-points...")

    global_min_eval = 1e10
    global_min_k = None
    all_min_evals = []

    for n1, n2, n3 in product(range(n_grid), repeat=3):
        k = np.array([n1, n2, n3], dtype=float) / n_grid
        L_k = twisted_laplacian(k, bonds, edge_labels, n_verts, omega_z3)
        evals = np.sort(np.real(la.eigvalsh(L_k)))
        min_ev = evals[0]
        all_min_evals.append(min_ev)

        if min_ev < global_min_eval:
            global_min_eval = min_ev
            global_min_k = k.copy()

    print(f"\n  Global minimum eigenvalue of L_tw(k): {global_min_eval:.10f}")
    print(f"  At k = ({global_min_k[0]:.4f}, {global_min_k[1]:.4f}, {global_min_k[2]:.4f})")

    if global_min_eval > 1e-10:
        L_us_global = 1.0 / global_min_eval
        vus_global = (2.0/3.0) ** L_us_global
        print(f"\n  L_us = 1/gap = {L_us_global:.10f}")
        print(f"  |V_us|_tw = (2/3)^L = {vus_global:.10f}")
        print(f"  Observed |V_us|   = {V_US_OBS}")
        print(f"  Deviation: {(vus_global - V_US_OBS)/V_US_OBS * 100:.4f}%")
    else:
        print(f"  WARNING: global minimum ~ 0 (twist does not fully gap the spectrum)")

    # Min eigenvalue statistics
    all_min_evals = np.array(all_min_evals)
    print(f"\n  Min eigenvalue range over BZ: [{all_min_evals.min():.6f}, {all_min_evals.max():.6f}]")
    print(f"  Mean: {all_min_evals.mean():.6f}, Std: {all_min_evals.std():.6f}")

    # Also check: untwisted global min (should have zero eigenvalue at Gamma)
    print("\n  For comparison, untwisted Laplacian BZ scan...")
    untw_global_min = 1e10
    for n1, n2, n3 in product(range(n_grid), repeat=3):
        k = np.array([n1, n2, n3], dtype=float) / n_grid
        L_k = bloch_laplacian(k, bonds, n_verts)
        evals = np.sort(np.real(la.eigvalsh(L_k)))
        if evals[0] < untw_global_min:
            untw_global_min = evals[0]
    print(f"  Untwisted global min eigenvalue: {untw_global_min:.10f} (should be ~0)")

    # =======================================================================
    # Step 5: Twist angle scan
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 5: TWIST ANGLE SCAN (phi = 0 to pi)")
    print("=" * 70)

    n_phi = 100
    n_grid_scan = 30  # coarser for speed
    phis = np.linspace(0, np.pi, n_phi)
    scan_results = []

    print(f"\n  Scanning {n_phi} twist angles with {n_grid_scan}^3 k-grid...")
    for phi in phis:
        omega = np.exp(1j * phi)

        # Find global minimum eigenvalue of twisted Laplacian
        local_min = 1e10
        for n1, n2, n3 in product(range(n_grid_scan), repeat=3):
            k = np.array([n1, n2, n3], dtype=float) / n_grid_scan
            L_k = twisted_laplacian(k, bonds, edge_labels, n_verts, omega)
            evals = np.sort(np.real(la.eigvalsh(L_k)))
            if evals[0] < local_min:
                local_min = evals[0]

        L_val = 1.0 / local_min if local_min > 1e-10 else float('inf')
        v_us = (2.0/3.0) ** L_val if L_val < float('inf') else 0.0
        scan_results.append((phi, local_min, L_val, v_us))

    print(f"\n  {'phi/pi':>8}  {'gap':>12}  {'L_us':>12}  {'V_us':>12}  {'dev%':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
    for phi, gap, L, vus in scan_results[::5]:  # every 5th point
        dev = (vus - V_US_OBS) / V_US_OBS * 100 if vus > 0 else float('nan')
        print(f"  {phi/np.pi:>8.4f}  {gap:>12.6f}  {L:>12.6f}  {vus:>12.6f}  {dev:>+8.3f}")

    # Find the twist angle that gives V_us closest to observed
    best_idx = min(range(len(scan_results)),
                   key=lambda i: abs(scan_results[i][3] - V_US_OBS))
    bp, bg, bL, bv = scan_results[best_idx]
    print(f"\n  Best match to V_us = {V_US_OBS}:")
    print(f"    phi = {bp:.6f} ({bp/np.pi:.4f} pi)")
    print(f"    gap = {bg:.6f}")
    print(f"    L_us = {bL:.6f}")
    print(f"    V_us = {bv:.6f}")
    print(f"    Deviation: {(bv - V_US_OBS)/V_US_OBS * 100:.4f}%")

    # =======================================================================
    # Step 6: Analytic check at Gamma for Z3
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 6: ANALYTIC STRUCTURE OF TWISTED ADJACENCY AT GAMMA")
    print("=" * 70)

    # Build the twisted adjacency matrix at Gamma explicitly
    A_tw = twisted_bloch_hamiltonian(k0, bonds, edge_labels, n_verts, omega_z3)
    print(f"\n  Twisted adjacency matrix A_tw (Gamma point):")
    print(f"  Shape: {A_tw.shape}")

    # Check hermiticity
    print(f"  Hermitian? max|A - A^dag| = {la.norm(A_tw - A_tw.conj().T):.2e}")

    # Eigenvalues of A_tw
    evals_A = np.sort(np.real(la.eigvalsh(A_tw)))
    print(f"\n  Eigenvalues of A_tw:")
    for i, ev in enumerate(evals_A):
        print(f"    a_{i} = {ev:.10f}")

    print(f"\n  Laplacian eigenvalues = 3 - a_i:")
    for i, ev in enumerate(evals_A):
        lam = 3 - ev
        print(f"    lambda_{i} = {lam:.10f}")

    # The maximum eigenvalue of A_tw gives the minimum Laplacian eigenvalue
    max_a = max(evals_A)
    min_lap = 3 - max_a
    print(f"\n  Max adjacency eigenvalue: {max_a:.10f}")
    print(f"  Min Laplacian eigenvalue: {min_lap:.10f}")

    # =======================================================================
    # Step 7: Summary
    # =======================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
  Untwisted:
    Spectral gap (Gamma): {UNTWISTED_GAP:.10f} = 2 - sqrt(3)
    L_us = {UNTWISTED_LUS:.10f} = 2 + sqrt(3)
    |V_us| = (2/3)^L = {UNTWISTED_VUS:.10f}
    Deviation from 0.2248: {(UNTWISTED_VUS - V_US_OBS)/V_US_OBS * 100:.4f}%

  Z3-Twisted (Gamma point):
    Spectral gap: {gap_tw_gamma:.10f}
    L_us = {1.0/gap_tw_gamma if gap_tw_gamma > 1e-10 else float('inf'):.10f}
    |V_us| = {vus_tw_gamma if gap_tw_gamma > 1e-10 else 0:.10f}
    Deviation from 0.2248: {(vus_tw_gamma - V_US_OBS)/V_US_OBS * 100 if gap_tw_gamma > 1e-10 else float('nan'):.4f}%

  Z3-Twisted (global over BZ):
    Spectral gap: {global_min_eval:.10f}
    L_us = {1.0/global_min_eval if global_min_eval > 1e-10 else float('inf'):.10f}
    |V_us| = {vus_global if global_min_eval > 1e-10 else 0:.10f}
    Deviation from 0.2248: {(vus_global - V_US_OBS)/V_US_OBS * 100 if global_min_eval > 1e-10 else float('nan'):.4f}%

  Required for exact match:
    Gap = {required_gap:.10f}
    L_us = {required_L:.10f}

  Direction: gap must {'INCREASE' if required_gap > UNTWISTED_GAP else 'DECREASE'} from {UNTWISTED_GAP:.6f} to {required_gap:.6f}
  The Z3 twist {'DOES' if abs(gap_tw_gamma - required_gap) < 0.01 else 'DOES NOT'} achieve this.
""")

    # Key diagnostic: which direction did the gap move?
    if gap_tw_gamma > UNTWISTED_GAP:
        print("  >>> Z3 twist INCREASES the gap at Gamma (right direction)")
    elif gap_tw_gamma < UNTWISTED_GAP:
        print("  >>> Z3 twist DECREASES the gap at Gamma (wrong direction)")
    else:
        print("  >>> Z3 twist does not change the gap at Gamma")

    if global_min_eval > UNTWISTED_GAP + 1e-6:
        print("  >>> Global twisted gap is LARGER than untwisted (V_us moves wrong way)")
    elif global_min_eval < UNTWISTED_GAP - 1e-6:
        print("  >>> Global twisted gap is SMALLER than untwisted (V_us moves wrong way)")
    else:
        print("  >>> Global twisted gap is approximately EQUAL to untwisted")

    # =======================================================================
    # Step 8: Refine the BZ minimum with scipy
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 8: REFINE BZ MINIMUM (scipy.optimize)")
    print("=" * 70)

    from scipy.optimize import minimize

    def neg_min_eval_twisted(k_flat):
        k = k_flat
        L_k = twisted_laplacian(k, bonds, edge_labels, n_verts, omega_z3)
        evals = np.sort(np.real(la.eigvalsh(L_k)))
        return evals[0]  # minimize the smallest eigenvalue

    # Start from the grid minimum
    result = minimize(neg_min_eval_twisted, global_min_k,
                      method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 10000})
    k_opt = result.x
    gap_opt = result.fun

    print(f"\n  Optimized k = ({k_opt[0]:.8f}, {k_opt[1]:.8f}, {k_opt[2]:.8f})")
    print(f"  Refined twisted gap = {gap_opt:.14f}")
    print(f"  2 - sqrt(3)         = {UNTWISTED_GAP:.14f}")
    print(f"  Difference           = {gap_opt - UNTWISTED_GAP:.2e}")

    if gap_opt > 1e-10:
        L_opt = 1.0 / gap_opt
        vus_opt = (2.0/3.0) ** L_opt
        print(f"\n  L_us = {L_opt:.10f}")
        print(f"  |V_us| = {vus_opt:.10f}")
        print(f"  Observed = {V_US_OBS}")
        print(f"  Deviation: {(vus_opt - V_US_OBS)/V_US_OBS * 100:.4f}%")

    # Also refine the untwisted minimum
    def min_eval_untwisted(k_flat):
        L_k = bloch_laplacian(k_flat, bonds, n_verts)
        evals = np.sort(np.real(la.eigvalsh(L_k)))
        return evals[0]

    # We know the untwisted minimum is at a specific BZ point; search near several candidates
    best_untw = 1e10
    best_untw_k = None
    # Try many starting points
    for n1, n2, n3 in product(range(20), repeat=3):
        k_start = np.array([n1, n2, n3], dtype=float) / 20
        L_k = bloch_laplacian(k_start, bonds, n_verts)
        ev0 = np.sort(np.real(la.eigvalsh(L_k)))[0]
        if ev0 < 0.3 and ev0 > 1e-8:  # near the known gap
            res2 = minimize(min_eval_untwisted, k_start,
                           method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-14})
            if res2.fun < best_untw and res2.fun > 1e-10:
                best_untw = res2.fun
                best_untw_k = res2.x

    if best_untw_k is not None:
        print(f"\n  Refined UNTWISTED gap = {best_untw:.14f}")
        print(f"  At k = ({best_untw_k[0]:.8f}, {best_untw_k[1]:.8f}, {best_untw_k[2]:.8f})")
        print(f"  2 - sqrt(3)          = {UNTWISTED_GAP:.14f}")
        print(f"  Difference            = {best_untw - UNTWISTED_GAP:.2e}")

    # =======================================================================
    # Step 9: Check analytic values
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 9: ANALYTIC VALUE CHECK")
    print("=" * 70)

    # The Gamma-point twisted gap 0.3542... -- check if it's 3 - sqrt(7)
    candidates = [
        ("3 - sqrt(7)", 3 - np.sqrt(7)),
        ("3 - 2*sqrt(2)*cos(pi/9)", 3 - 2*np.sqrt(2)*np.cos(np.pi/9)),
        ("3 - 2*sqrt(2)*cos(2pi/9)", 3 - 2*np.sqrt(2)*np.cos(2*np.pi/9)),
        ("6 - 4*sqrt(2)", 6 - 4*np.sqrt(2)),
        ("3 - sqrt(2) - sqrt(5)", 3 - np.sqrt(2) - np.sqrt(5)),
        ("2/(3+sqrt(2))", 2/(3+np.sqrt(2))),
        ("sqrt(2) - 1", np.sqrt(2) - 1),
        ("3 - sqrt(7)", 3 - np.sqrt(7)),
        ("(7-sqrt(7))/6", (7-np.sqrt(7))/6),
    ]
    print(f"\n  Gamma-point twisted gap = {gap_tw_gamma:.10f}")
    for name, val in candidates:
        if abs(val - gap_tw_gamma) < 0.01:
            print(f"    Close to {name} = {val:.10f} (diff = {val - gap_tw_gamma:.2e})")

    # Check 3 - sqrt(7) more carefully
    val_check = 3 - np.sqrt(7)
    print(f"\n  3 - sqrt(7) = {val_check:.10f}")
    print(f"  Gap         = {gap_tw_gamma:.10f}")
    print(f"  Match: {abs(val_check - gap_tw_gamma) < 1e-8}")

    # Try to identify the exact eigenvalue
    # The twisted adjacency max eigenvalue was 2.6457513111
    # Check: sqrt(7) = 2.6457513111
    print(f"\n  Max adjacency eigenvalue: {max_a:.10f}")
    print(f"  sqrt(7) = {np.sqrt(7):.10f}")
    print(f"  Match: {abs(max_a - np.sqrt(7)) < 1e-8}")

    if abs(max_a - np.sqrt(7)) < 1e-8:
        print(f"\n  CONFIRMED: Gamma-point twisted gap = 3 - sqrt(7)")
        print(f"  L_us at Gamma = 1/(3 - sqrt(7)) = (3 + sqrt(7))/2 = {(3 + np.sqrt(7))/2:.10f}")

    # =======================================================================
    # Step 10: Does the BZ minimum approach 2-sqrt(3) exactly?
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 10: IS THE TWISTED BZ MINIMUM EXACTLY 2-sqrt(3)?")
    print("=" * 70)

    print(f"\n  Refined twisted gap:  {gap_opt:.14f}")
    print(f"  2 - sqrt(3):         {UNTWISTED_GAP:.14f}")
    print(f"  Difference:          {gap_opt - UNTWISTED_GAP:.2e}")
    print(f"  Relative difference: {(gap_opt - UNTWISTED_GAP)/UNTWISTED_GAP * 100:.6f}%")

    if abs(gap_opt - UNTWISTED_GAP) < 1e-6:
        print(f"\n  CONCLUSION: The Z3 twist does NOT change the global spectral gap.")
        print(f"  The BZ minimum of the twisted Laplacian coincides with 2-sqrt(3).")
        print(f"  Therefore V_us remains at (2/3)^(2+sqrt(3)) = {UNTWISTED_VUS:.6f}")
        print(f"  The 2.05% deviation from 0.2248 is NOT corrected by the Z3 twist.")
    else:
        print(f"\n  The Z3 twist shifts the gap by {gap_opt - UNTWISTED_GAP:.2e}")
        new_L = 1.0/gap_opt
        new_vus = (2.0/3.0)**new_L
        print(f"  New L_us = {new_L:.10f}")
        print(f"  New V_us = {new_vus:.10f}")
        print(f"  Shift in V_us: {(new_vus - UNTWISTED_VUS)/UNTWISTED_VUS * 100:.4f}%")


if __name__ == '__main__':
    main()
