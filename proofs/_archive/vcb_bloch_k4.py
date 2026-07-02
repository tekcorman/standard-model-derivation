#!/usr/bin/env python3
"""
proofs/_archive/vcb_bloch_k4.py

Bloch-twisted K4 Hashimoto computation for V_cb coefficient c.
Goal: determine c in V_cb = alpha_1 * (1 + c * alpha_1) from first principles.

Uses 8-atom BCC quotient of srs (24 directed edges). Since all girth-10
cycles are contractible (winding=0, confirmed vcb_holonomy_count.py §A2),
e^{ik.Delta} = 1 for all girth cycles, so k=0 is the only relevant momentum.

The computation:
  1. Extract crystal bond structure (type_i, type_j, R_ij) from supercell
  2. Build quotient Hashimoto H_Q (24x24, k=0)
  3. Compute H_Q^8 and H_Q^18
  4. Sum over directed-edge pairs with each Z3 holonomy class
  5. Extract V_cb amplitudes and coefficient c
"""

import numpy as np
from fractions import Fraction
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcb_holonomy_count import build_supercell, get_cell_indices

# -----------------------------------------------------------------
# 1.  CRYSTAL BOND TABLE  (from supercell center cell)
# -----------------------------------------------------------------

def extract_crystal_bonds(n_cells=3):
    """Return list of (type_i, type_j, dx, dy, dz) for each directed bond."""
    positions, edges, adjacency = build_supercell(n_cells)
    center = n_cells // 2
    center_start = (center * n_cells * n_cells + center * n_cells + center) * 8
    bonds = []
    for atype in range(8):
        vi = center_start + atype
        for vj in adjacency[vi]:
            cxj, cyj, czj, tj = get_cell_indices(vj, n_cells, 8)
            bonds.append((atype, tj, cxj - center, cyj - center, czj - center))
    return bonds

# -----------------------------------------------------------------
# 2.  QUOTIENT HASHIMOTO  (k=0)
# -----------------------------------------------------------------

def build_quotient_hashimoto(bonds):
    """
    Build the 24x24 quotient Hashimoto matrix at k=0.
    Directed edge e = (i, j, dx, dy, dz).
    H[e, f] = 1 if j==l (head(e)==tail(f)) and NOT backtrack.
    Backtrack: m==i and (dxf, dyf, dzf) == (-dxi, -dyi, -dzi).
    """
    E = len(bonds)
    ei = {b: idx for idx, b in enumerate(bonds)}
    H = np.zeros((E, E), dtype=np.float64)
    for (ti, tj, dxi, dyi, dzi), idx_e in ei.items():
        for (tl, tm, dxf, dyf, dzf), idx_f in ei.items():
            if tl != tj:
                continue  # head(e) != tail(f)
            if tm == ti and dxf == -dxi and dyf == -dyi and dzf == -dzi:
                continue  # backtrack
            H[idx_e, idx_f] = 1.0
    return H, ei

# -----------------------------------------------------------------
# 3.  Z3 EDGE LABELS
#     Two candidate definitions:
#     (a) direction-label: argmin |shift| component
#     (b) sublattice-pair: (type_i mod 4, type_j mod 4)
#         maps each bond to one of the 3 bond types of K4
# -----------------------------------------------------------------

def direction_labels(bonds):
    """Label each bond by index of coordinate with smallest |shift|."""
    labels = []
    for (ti, tj, dx, dy, dz) in bonds:
        shift = np.array([dx, dy, dz], dtype=float)
        if np.all(shift == 0):
            labels.append(0)  # intra-cell bond, fallback
        else:
            labels.append(int(np.argmin(np.abs(shift) + 1e-9 * np.arange(3))))
    return labels

def k4_pair_labels(bonds):
    """
    In the K4 quotient (4 vertex types), label by unordered pair of
    vertex types. srs BCC has types {0,1,2,3} and {4,5,6,7};
    use (min(ti,tj) mod 4, max(ti,tj) mod 4) -> one of 6 undirected bond types.
    Actually use (ti mod 4, tj mod 4) -> {01,02,03,12,13,23} -> 0..5.
    """
    pair_to_label = {}
    label_idx = 0
    labels = []
    for (ti, tj, dx, dy, dz) in bonds:
        a, b = ti % 4, tj % 4
        key = (min(a,b), max(a,b))
        if key not in pair_to_label:
            pair_to_label[key] = label_idx % 3  # 3-coloring mod 3
            label_idx += 1
        labels.append(pair_to_label[key])
    return labels, pair_to_label

# -----------------------------------------------------------------
# 4.  MAIN
# -----------------------------------------------------------------

def main():
    print("=" * 70)
    print("BLOCH K4 HASHIMOTO  —  V_cb coefficient c at k=0")
    print("=" * 70)

    bonds = extract_crystal_bonds(3)
    print(f"\nDirected bonds in BCC unit cell: {len(bonds)}")

    H, ei = build_quotient_hashimoto(bonds)
    E = len(bonds)
    k = 3

    # Sanity: each row of H has exactly k-1=2 ones (NB constraint)
    row_sums = H.sum(axis=1)
    assert np.all(row_sums == k-1), f"Row sums: {np.unique(row_sums)}"
    print("  Row sums = 2 (NB constraint): OK")

    # Compute H^8 and H^18
    H8  = np.linalg.matrix_power(H, 8)
    H18 = np.linalg.matrix_power(H, 18)

    alpha1 = (2/3)**8
    print(f"\n  alpha_1 = (2/3)^8 = {alpha1:.10f}")

    # ---  TOTAL NB WALK COUNTS  ---
    print("\n  Total NB walks of length 8 per starting edge:")
    print(f"    H^8 row sum: min={H8.sum(1).min():.0f}  max={H8.sum(1).max():.0f}")
    print(f"    Expected (k-1)^8 = 2^8 = {2**8}")

    # ---  DIAGONAL (CLOSED WALKS)  ---
    diag8  = np.diag(H8)
    diag10 = np.diag(np.linalg.matrix_power(H, 10))
    print(f"\n  Closed NB walks of length 8  (diagonal of H^8):")
    print(f"    Values: {np.unique(diag8.astype(int))}   (should be 0 if girth > 8)")
    print(f"  Closed NB walks of length 10 (diagonal of H^10):")
    print(f"    Sum = {diag10.sum():.0f}, per edge = {diag10.mean():.4f}")
    print(f"    Values: {np.unique(diag10.astype(int))}")

    # ---  Z3 HOLONOMY VIA DIRECTION LABELS  ---
    dir_lbl = direction_labels(bonds)
    edge_count_per_label = [sum(1 for l in dir_lbl if l == h) for h in range(3)]
    print(f"\n  Direction-label distribution: {edge_count_per_label}")

    print("\n  H^8 sums by (label_in, label_out):")
    matrix_8 = np.zeros((3,3), dtype=float)
    matrix_18 = np.zeros((3,3), dtype=float)
    for ei_idx, (li) in enumerate(dir_lbl):
        for ef_idx, (lf) in enumerate(dir_lbl):
            matrix_8[li, lf]  += H8[ei_idx, ef_idx]
            matrix_18[li, lf] += H18[ei_idx, ef_idx]

    print(f"    (rows=label_in, cols=label_out)")
    print(f"    H^8:\n{matrix_8.astype(int)}")
    print(f"    H^18:\n{matrix_18.astype(int)}")

    # Per-edge amplitude for each Δlabel
    n_per_lbl = 8  # 8 directed edges per label (8 atoms, 1 of each label)
    print("\n  Amplitude per (in_edge, Δlabel) via direction-label Z3:")
    for dlbl in range(3):
        # Sum H^8_{e, f} over all f with lf = (li + dlbl) % 3 for fixed li
        vcb1 = 0.0; vcb2 = 0.0
        for li in range(3):
            lf = (li + dlbl) % 3
            vcb1 += matrix_8[li, lf]
            vcb2 += matrix_18[li, lf]
        vcb1 *= (1/k)**8  / n_per_lbl   # per starting edge
        vcb2 *= (1/k)**18 / n_per_lbl
        ratio_to_alpha1 = vcb1 / alpha1
        c_val = vcb2 / vcb1**2 if vcb1 > 1e-15 else 0
        print(f"    Δlbl={dlbl}: V^(1)={vcb1:.8f}  V^(1)/alpha1={ratio_to_alpha1:.4f}"
              f"  V^(2)={vcb2:.8f}  c={c_val:.4f}")

    # ---  Z3 VIA K4 PAIR LABELS  ---
    k4_lbl, pair_map = k4_pair_labels(bonds)
    print(f"\n  K4 pair-label map: {pair_map}")
    edge_count_per_k4 = [sum(1 for l in k4_lbl if l == h) for h in range(3)]
    print(f"  K4 label distribution: {edge_count_per_k4}")

    matrix_8k = np.zeros((3,3), dtype=float)
    matrix_18k = np.zeros((3,3), dtype=float)
    for ei_idx, li in enumerate(k4_lbl):
        for ef_idx, lf in enumerate(k4_lbl):
            matrix_8k[li, lf]  += H8[ei_idx, ef_idx]
            matrix_18k[li, lf] += H18[ei_idx, ef_idx]

    n_per_k4 = edge_count_per_k4[0]
    print(f"\n  Amplitude per (in_edge, Δlabel) via K4-pair Z3:")
    for dlbl in range(3):
        vcb1 = 0.0; vcb2 = 0.0
        for li in range(3):
            lf = (li + dlbl) % 3
            vcb1 += matrix_8k[li, lf]
            vcb2 += matrix_18k[li, lf]
        vcb1 *= (1/k)**8  / n_per_k4
        vcb2 *= (1/k)**18 / n_per_k4
        ratio_to_alpha1 = vcb1 / alpha1
        c_val = vcb2 / vcb1**2 if vcb1 > 1e-15 else 0
        print(f"    Δlbl={dlbl}: V^(1)={vcb1:.8f}  V^(1)/alpha1={ratio_to_alpha1:.4f}"
              f"  V^(2)={vcb2:.8f}  c={c_val:.4f}")

    # ---  DIRECT CYCLE-COUNTING CHECK  ---
    # Sum of H^10 diagonal = total closed-NB-walks of length 10 in quotient
    # Divide by E to get per-directed-edge rate
    H10 = np.linalg.matrix_power(H, 10)
    closed10 = np.diag(H10).sum()
    print(f"\n  Closed NB walks of length 10 in quotient (total): {closed10:.0f}")
    print(f"  Per directed edge: {closed10/E:.4f}")
    # Compare to full supercell
    # Full supercell has 27 * 24 = 648 directed edges, each with ~15 closed-10-walks
    print(f"  Supercell equivalent: 15 girth cycles/vertex -> "
          f"{15 * 2 / (k)} cycles/directed-edge = {30/k:.4f}")

    # ---  EXACT RATIONAL CHECK for n_fixed=2  ---
    # alpha_1 = (2/3)^8 = (k-1)^8 / k^8
    # V_cb^(1) should = alpha_1 for one of the holonomy definitions
    print("\n  Checking: which Δlabel gives V^(1) = alpha_1?")
    for dlbl in range(3):
        vcb1_dir = 0.0
        for li in range(3):
            lf = (li + dlbl) % 3
            vcb1_dir += matrix_8[li, lf]
        vcb1_dir = vcb1_dir * (1/k)**8 / n_per_lbl
        if abs(vcb1_dir - alpha1) < 1e-8:
            print(f"    direction-label Δlbl={dlbl}: MATCHES alpha_1 ✓")
        else:
            print(f"    direction-label Δlbl={dlbl}: {vcb1_dir:.8f} vs {alpha1:.8f}")

    # ---  SUMMARY  ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  alpha_1 = (2/3)^8 = {alpha1:.10f}")
    print()

    # Holonomy-0 direction-label (= "trivial / same generation"):
    vcb1_0 = matrix_8[0,0] + matrix_8[1,1] + matrix_8[2,2]
    vcb1_0 *= (1/k)**8 / n_per_lbl
    vcb2_0 = matrix_18[0,0] + matrix_18[1,1] + matrix_18[2,2]
    vcb2_0 *= (1/k)**18 / n_per_lbl
    c0 = vcb2_0 / vcb1_0**2 if vcb1_0 > 0 else 0

    vcb1_1 = sum(matrix_8[li, (li+1)%3] for li in range(3))
    vcb1_1 *= (1/k)**8 / n_per_lbl
    vcb2_1 = sum(matrix_18[li, (li+1)%3] for li in range(3))
    vcb2_1 *= (1/k)**18 / n_per_lbl
    c1 = vcb2_1 / vcb1_1**2 if vcb1_1 > 0 else 0

    print(f"  Direction-label holonomy:")
    print(f"    Δlbl=0: V^(1)={vcb1_0:.8f}  V^(2)={vcb2_0:.8f}  c={c0:.4f}")
    print(f"    Δlbl=1: V^(1)={vcb1_1:.8f}  V^(2)={vcb2_1:.8f}  c={c1:.4f}")
    print()

    if abs(c1 - 1.0) < 0.01:
        print("  c = 1 CONFIRMED for Δlbl=1 (direction-label holonomy)")
    else:
        print(f"  c ≠ 1 for Δlbl=1 (c = {c1:.4f})")
        print("  Gate BLOCKED: coefficient c in V_cb = alpha_1*(1+c*alpha_1) is not 1.")
        print("  The formula V_cb = alpha_1*(1+alpha_1) lacks a gate-passing derivation.")


if __name__ == "__main__":
    main()
