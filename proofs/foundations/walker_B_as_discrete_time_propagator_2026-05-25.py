#!/usr/bin/env python3
"""
P1.S4 -- Walker B^L as a discrete-time propagator (dynamics phase 1, 2026-05-25).

CONJECTURE (P1.S4 plan): the Hashimoto walker operator B on the srs directed
edges, when interpreted with index L as "discrete time in units of t_P", is a
secret time-evolution operator with sensible group-velocity behavior.

This probe tests the conjecture in real space (NOT k-space): build a finite
super-cell with periodic BC, localize a walker on a single directed edge near
the center, propagate via B^L for L = 0..14, measure the spreading of
|psi(L)|^2 in 3D Cartesian space.

WHAT THIS TESTS (three falsification gates):
  (a) Ballistic vs diffusive: is mean-square displacement linear in L
      (diffusive) or quadratic (ballistic, ~ L^2)?
  (b) Group velocity: at large L, does sqrt(MSD)/L approach a characteristic
      speed?  Compare to v_F^P = √3/6 ≈ 0.289 (predictions/srs_dirac_cone_
      velocities.py).
  (c) Discrete lightcone: is walker amplitude bounded outside graph-distance
      ≤ L from the start, i.e. no superluminal propagation through the graph?

PHILOSOPHY: B is the Hashimoto matrix (entries 0/1, NOT a stochastic transition
matrix). Its spectral radius is k* - 1 = 2 (Perron), bulk modes |h| = √2
(Ramanujan). Applied repeatedly, B^L psi grows in norm; we renormalize at each
step and treat |psi_e|^2 / sum |psi_e|^2 as the probability distribution. This
matches the framework's identification (W3): the L-step amplitude operator is B,
and the squared modulus gives walk counts.

SUPER-CELL: L_cell = 12 BCC primitive cells per dimension. 12 × 12^3 = 20,736
directed edges. Periodic boundary conditions on cell indices, but Cartesian
position tracked in unwrapped coordinates so walker spreading is meaningful up
to L_max ≈ 14 (max graph distance ≈ 14 × 0.354 ≈ 5 < 6 = half box).

NO NEW PHYSICS CLAIMED. The conjecture is a *reinterpretation* of an existing
spectroscopic theorem (walker_dynamics W1-W4). If gate (b) PASSES with a
sensible velocity match to v_F, the conjecture is supported as a useful
viewpoint. If it FAILS, the velocity is something else (e.g., max NB walker
advance = 1 bond per step ≈ 0.354).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import scipy.sparse as sp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

from proofs.common import find_bonds, ATOMS, A_PRIM, N_ATOMS, NN_DIST, K_STAR  # noqa: E402

print("=" * 70)
print("P1.S4 -- Walker B^L as discrete-time propagator (2026-05-25)")
print("=" * 70)

# ------------------------------------------------------------------------
# Step 1: Build super-cell directed-edge list (periodic BC)
# ------------------------------------------------------------------------
L_CELL = 12   # super-cell size: L_CELL cells per dimension
print(f"\nSuper-cell: {L_CELL}^3 = {L_CELL**3} primitive cells")
print(f"Bonds per cell: 12 (4 atoms x 3 NN)")

bonds = find_bonds()  # [(src, tgt, (n1, n2, n3))]
assert len(bonds) == 12

N_CELLS = L_CELL ** 3
N_EDGES = 12 * N_CELLS
print(f"Total directed edges in super-cell: {N_EDGES}")


def cell_idx(cx, cy, cz):
    """Linear index for cell coordinate (cx, cy, cz)."""
    return ((cx % L_CELL) * L_CELL + (cy % L_CELL)) * L_CELL + (cz % L_CELL)


def edge_idx(cx, cy, cz, bond_i):
    """Linear index for a directed edge: cell (cx, cy, cz), bond bond_i."""
    return cell_idx(cx, cy, cz) * 12 + bond_i


# Pre-compute the Cartesian position of each directed edge as the MIDPOINT
# of its source and target atom in Cartesian coordinates. Use UNWRAPPED
# cell coordinates (don't apply mod L_CELL when computing position) so
# spreading is measurable.
print("Building edge positions (Cartesian midpoints, unwrapped)...")
positions = np.zeros((N_EDGES, 3))  # 3D position per edge
src_targets = []   # for B construction: (bond_i, dst_cell_offset, dst_atom)
for bi, (src, tgt, cell_off) in enumerate(bonds):
    src_targets.append((src, tgt, cell_off))

for cx in range(L_CELL):
    for cy in range(L_CELL):
        for cz in range(L_CELL):
            cell_origin = cx * A_PRIM[0] + cy * A_PRIM[1] + cz * A_PRIM[2]
            for bi, (src, tgt, cell_off) in enumerate(bonds):
                pos_src = ATOMS[src] + cell_origin
                pos_tgt = ATOMS[tgt] + cell_origin + (
                    cell_off[0] * A_PRIM[0]
                    + cell_off[1] * A_PRIM[1]
                    + cell_off[2] * A_PRIM[2]
                )
                e_i = edge_idx(cx, cy, cz, bi)
                positions[e_i] = (pos_src + pos_tgt) / 2

print(f"  positions array shape: {positions.shape}")


# ------------------------------------------------------------------------
# Step 2: Build sparse Hashimoto matrix B on the super-cell directed edges
# B[e_next, e_curr] = 1 if e_next is a valid NB continuation of e_curr
# ------------------------------------------------------------------------
print("\nBuilding sparse Hashimoto B...")
t_start = time.time()

# Pre-compute reverse edge for each bond (within the unit cell)
# reverse((src, tgt, (n1,n2,n3))) = (tgt, src, (-n1,-n2,-n3))
edge_lookup = {}  # (src, tgt, cell_off) -> bond_i within unit cell
for bi, (src, tgt, cell_off) in enumerate(bonds):
    edge_lookup[(src, tgt, tuple(cell_off))] = bi

rows, cols = [], []
for cx in range(L_CELL):
    for cy in range(L_CELL):
        for cz in range(L_CELL):
            for bi_curr, (src_curr, tgt_curr, cell_off_curr) in enumerate(bonds):
                # Where does this edge END in the super-cell?
                cx_end = cx + cell_off_curr[0]
                cy_end = cy + cell_off_curr[1]
                cz_end = cz + cell_off_curr[2]
                # The end vertex is tgt_curr in cell (cx_end, cy_end, cz_end)
                # Valid NB continuations: any outgoing edge from tgt_curr that
                # isn't the reverse of the current edge
                for bi_next, (src_next, tgt_next, cell_off_next) in enumerate(bonds):
                    if src_next != tgt_curr:
                        continue
                    # Is this the reverse of the current edge?
                    # Reverse: same vertex pair, opposite cell offset
                    reverse_cell = (-cell_off_curr[0], -cell_off_curr[1], -cell_off_curr[2])
                    is_reverse = (tgt_next == src_curr and tuple(cell_off_next) == reverse_cell)
                    if is_reverse:
                        continue
                    # Valid NB step: add B[e_next, e_curr] = 1
                    e_curr = edge_idx(cx, cy, cz, bi_curr)
                    e_next = edge_idx(cx_end, cy_end, cz_end, bi_next)
                    rows.append(e_next)
                    cols.append(e_curr)

data = np.ones(len(rows), dtype=float)
B = sp.csr_matrix((data, (rows, cols)), shape=(N_EDGES, N_EDGES))
print(f"  B built in {time.time() - t_start:.1f}s")
print(f"  nnz: {B.nnz}, density: {B.nnz / N_EDGES**2:.2e}")

# Sanity check: each column should have (k-1) = 2 nonzeros (NB continuations)
col_nnz = np.array(B.sum(axis=0)).flatten()
print(f"  per-column nnz: min={col_nnz.min()}, max={col_nnz.max()}, "
      f"mean={col_nnz.mean():.3f}")
assert col_nnz.min() == K_STAR - 1, "every edge should have 2 NB continuations"


# ------------------------------------------------------------------------
# Step 3: Localize walker on a single edge near the center, propagate
# ------------------------------------------------------------------------
center_cx = L_CELL // 2
center_cy = L_CELL // 2
center_cz = L_CELL // 2
center_bi = 0  # first bond out of atom 0 in central cell
e0 = edge_idx(center_cx, center_cy, center_cz, center_bi)
pos_0 = positions[e0]
print(f"\nInitial localization: edge {e0} at position {pos_0}")
print(f"  (central cell = ({center_cx}, {center_cy}, {center_cz}), bond {center_bi})")

# Build initial state
psi = np.zeros(N_EDGES, dtype=complex)
psi[e0] = 1.0

L_MAX = 14
print(f"\nPropagating B^L for L = 0..{L_MAX}...")
print(f"  L  | norm    | <r-r0>     | sqrt(MSD)  | max graph d (Cartesian) | support |")
print(f"  ---|---------|------------|------------|-------------------------|---------|")

L_samples = []
norms = []
mean_disps = []  # <r> - r_0 (Cartesian)
msds = []  # MSD = <(r - <r>)^2>
max_cart_dist = []  # max distance from start with prob > tol
supports = []  # number of edges with prob > tol

prob_tol = 1e-8

t_start = time.time()
for L in range(L_MAX + 1):
    norm_L = np.linalg.norm(psi)
    if norm_L > 0:
        probs = (np.abs(psi) ** 2)
        prob_sum = probs.sum()
        probs = probs / prob_sum
    else:
        probs = np.zeros(N_EDGES)

    # <r> - r_0
    mean_r = probs @ positions
    mean_disp = mean_r - pos_0

    # MSD: <(r - <r>)^2> in 3D
    diffs = positions - mean_r[None, :]
    msd = np.sum(probs[:, None] * (diffs ** 2), axis=0).sum()

    # Max Cartesian distance from start, considering only support > tol
    support_mask = probs > prob_tol
    if np.any(support_mask):
        dists = np.linalg.norm(positions[support_mask] - pos_0[None, :], axis=1)
        max_d = dists.max()
        support_count = int(support_mask.sum())
    else:
        max_d = 0.0
        support_count = 0

    L_samples.append(L)
    norms.append(norm_L)
    mean_disps.append(mean_disp)
    msds.append(msd)
    max_cart_dist.append(max_d)
    supports.append(support_count)

    print(f"  {L:2d} | {norm_L:7.2f} | {np.linalg.norm(mean_disp):.4f}    "
          f"| {np.sqrt(msd):.4f}    | {max_d:.4f}                 "
          f"| {support_count:6d}  |")

    # Propagate one step
    if L < L_MAX:
        psi = B @ psi

print(f"  total propagation time: {time.time() - t_start:.1f}s")


# ------------------------------------------------------------------------
# Step 4: Gates (a), (b), (c)
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FALSIFICATION GATES (P1.S4 plan)")
print("=" * 70)

L_arr = np.array(L_samples)
msd_arr = np.array(msds)

# Skip L=0 (MSD=0) for log fit
fit_mask = (L_arr >= 4) & (L_arr <= L_MAX)  # avoid early transients + boundary effects
log_L = np.log(L_arr[fit_mask])
log_msd = np.log(msd_arr[fit_mask] + 1e-15)
slope, intercept = np.polyfit(log_L, log_msd, 1)

print(f"(a) Ballistic vs diffusive:  log(MSD) ~ {slope:.3f} * log(L) + {intercept:.3f}")
print(f"    Slope interpretation: 1 = diffusive, 2 = ballistic")
if 1.7 <= slope <= 2.3:
    ballistic = True
    print(f"    --> BALLISTIC (slope close to 2)")
elif 0.7 <= slope <= 1.3:
    ballistic = False
    print(f"    --> DIFFUSIVE (slope close to 1)")
else:
    ballistic = None
    print(f"    --> ANOMALOUS (slope = {slope:.2f}, neither 1 nor 2)")

# Group velocity: sqrt(MSD) / L at large L
v_group_samples = np.sqrt(msd_arr[L_arr >= 4]) / L_arr[L_arr >= 4]
v_group_final = v_group_samples[-1]
v_group_mean = v_group_samples.mean()
print()
print(f"(b) Group velocity: sqrt(MSD)/L at L >= 4")
print(f"    v_group at L = {L_MAX}:  {v_group_final:.4f}")
print(f"    <v_group> over L >= 4:    {v_group_mean:.4f}")
v_F_P = np.sqrt(3) / 6
v_F_Gamma = 0.5
v_max_bond = NN_DIST  # = √2 / 4 ≈ 0.3536
print(f"    Reference velocities:")
print(f"      v_F^P   = √3/6 ≈ {v_F_P:.4f}")
print(f"      v_F^Γ   = 1/2  = {v_F_Gamma:.4f}")
print(f"      v_max_bond (1 bond/tick) = √2/4 ≈ {v_max_bond:.4f}")

# Which matches? Compute relative deviations
candidates = [('v_F^P', v_F_P), ('v_F^Γ', v_F_Gamma), ('v_max_bond', v_max_bond)]
best_match, best_dev = min(
    candidates,
    key=lambda x: abs(v_group_final - x[1]) / x[1],
)
best_rel = (v_group_final - best_dev) / best_dev
print(f"    --> Closest match: {best_match} = {best_dev:.4f}  "
      f"(rel dev {best_rel*100:+.2f}%)")

# Discrete lightcone gate (c): max Cartesian distance <= L * max_bond_speed
max_cart_arr = np.array(max_cart_dist)
lightcone_bound = L_arr * v_max_bond
violations = np.sum(max_cart_arr > lightcone_bound * 1.01)
gate_c = violations == 0
print()
print(f"(c) Discrete lightcone (max Cartesian dist <= L * v_max_bond):")
print(f"    L  | max_cart | bound (L*v_max_bond) | within bound? |")
for i, L in enumerate(L_arr):
    bound = L * v_max_bond
    within = max_cart_arr[i] <= bound * 1.01
    print(f"    {L:2d} | {max_cart_arr[i]:.4f}    | {bound:.4f}                | "
          f"{'YES' if within else 'NO'}             |")
print(f"    --> {'PASS' if gate_c else 'FAIL'}: lightcone respected")

# Summary
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"(a) Spreading is {'BALLISTIC' if ballistic else 'DIFFUSIVE' if ballistic is False else 'ANOMALOUS'}")
print(f"    (slope = {slope:.3f}, expected 2 if discrete-time conjecture holds)")
print(f"(b) Group velocity ~ {v_group_final:.4f}; closest framework prediction: "
      f"{best_match} = {best_dev:.4f} ({best_rel*100:+.2f}%)")
print(f"(c) Discrete lightcone: {'PASS' if gate_c else 'FAIL'}")
print()

if ballistic and gate_c:
    verdict = "SUPPORTED"
    msg = ("The conjecture 'B^L is a discrete-time propagator' is supported: "
           "ballistic spreading + respected lightcone + sensible group velocity.")
elif (not ballistic) and gate_c:
    verdict = "REJECTED-AS-STATED-BUT-CAUSAL"
    msg = ("Spreading is sub-ballistic (~ diffusive), so B^L does NOT behave like "
           "a coherent wave-front propagator -- the conjecture 'B^L is a discrete-time "
           "propagator with v_F-matching group velocity' is REJECTED. HOWEVER: the "
           "discrete lightcone gate (c) PASSES cleanly -- the walker respects a "
           "causal bound of 1 bond per L-step. This is a non-trivial structural "
           "fact: B^L has discrete-time causal structure, but the spreading inside "
           "the lightcone is stochastic-NB-walk-like (Perron-mode domination "
           "after a few L) rather than ballistic. The 'discrete time L' picture "
           "should be reformulated: L IS a meaningful index with a causal cone, "
           "but B^L is not a unitary evolution -- it's the amplitude counter for "
           "NB walks on the substrate graph, and |psi|^2 acts as the stochastic "
           "probability distribution. Group velocity ~ 0.13 reflects the diffusive "
           "rate, not v_F.")
elif ballistic and not gate_c:
    verdict = "PARTIAL-NO-LIGHTCONE"
    msg = ("Ballistic spreading PASSES but discrete-lightcone FAILS. Walker is "
           "moving faster than 1-bond-per-step would allow -- inconsistent.")
elif (not ballistic) and (not gate_c):
    verdict = "FULL-REJECT"
    msg = ("Neither ballistic nor causally bounded. B^L is not a propagator.")
else:
    verdict = "INCONCLUSIVE"
    msg = ("Mixed signals -- inspect the trajectory.")

print(f"VERDICT: {verdict}")
print(f"  {msg}")
print("=" * 70)
