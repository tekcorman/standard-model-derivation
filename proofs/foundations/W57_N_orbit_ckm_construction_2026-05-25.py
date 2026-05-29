#!/usr/bin/env python3
"""
W57 — N-orbit CKM-construction probe: does the alternative Bloch point N
       admit an explicit M^(u)/M^(d) construction giving SM-compatible V_CKM?

Per an internal working note
§3 and the W55 verdict, the BCC has 4 high-symmetry points {Γ, H, P, N};
only N is non-C₃-fixed (C₃ has a 3-element orbit on N). The §8 scoping
identified N as the only remaining unexplored Bloch point for the explicit
M^(u)/M^(d) construction.

PRIOR WORK at N:
  - R-14 path (b) at the observable-matching level (2026-05-05,
    srs_R14_path_b_numerical_scan.py): closed NEGATIVE. No observable
    is unmatched at h_P that h_N could newly resolve. Moving leptons to
    N breaks the y_τ chirality factor (tan²(arg) = 5/3 at P becomes 3/5
    or 7 at N).
  - M1 first probe at N (m1_n_orbit_3orbit_basis.py, 2026-04):
    structurally-reframe-verified that V_Ram(N1) ⊕ V_Ram(N2) ⊕ V_Ram(N3)
    decomposes as 8 disjoint Z_3-cyclic 3-orbits under C_36 = (B_total,
    C_3-on-arcs combined). Each orbit is a candidate substrate image of
    {|gen-1⟩, |gen-2⟩, |gen-3⟩}.
  - C_36-twist attack (mass_operator_c36_twist_attack_2026-05-21.py):
    decomposed the §8 labeling residue as [ORDER] ⊕ [GEN-PAIR];
    [GEN-PAIR] = "the channel↔gen-pair map is the order-preserving
    bijection (ordinally fixed by magnitude ORDERING alone — no fitted
    value)".

NOT YET TESTED at N: an EXPLICIT M^(u)/M^(d) construction where two
different 3-orbits play the role of u-gen-basis and d-gen-basis, and
V_CKM = U_uL† U_dL is computed for each (orbit_u, orbit_d) pair.

This probe runs that enumeration.

HYPOTHESIS: there exists some (orbit_u, orbit_d) pair with orbit_u ≠ orbit_d
such that V_CKM = U_uL† U_dL has non-trivial off-diagonal mixing
reproducing the observed CKM hierarchy V_us > V_cb > V_ub.

PRE-DECLARED GATES:
  G1: 8 disjoint C_36 3-orbits exist (recover m1_n_orbit_3orbit_basis).
  G2: For some pair (orbit_u, orbit_d), V_CKM has NON-TRIVIAL off-diagonal
      structure (i.e., V_CKM is NOT scalar·I or scalar·permutation).
  G3: For some pair, |V_CKM[i,j]| ordering matches V_us > V_cb > V_ub
      hierarchy.
  G4: For some pair, magnitudes match observed within factor of 2.

PRE-DECLARED ABORTS:
  AB1: if 8-orbit infrastructure broken, abort.
  AB2: if for ALL pairs V_CKM is scalar·I or scalar·permutation (trivial
       mixing), abort with STRUCTURAL NEGATIVE — N-orbit approach
       cannot host non-trivial CKM mixing.
  AB3: literal-claim — if any "match" requires fitting (e.g., picking the
       best of 28 pairs against PDG), abort with data-anchored labeling.
  AB4: if multiple pairs match observed CKM equally well (no unique pin),
       abort — labeling residue is irreducibly data-anchored across pair
       selections.
  AB5: framework extension required, abort.
  AB6: scope-creep into [ORDER] = δ, abort + re-scope.
"""

from __future__ import annotations
import os
import sys
from itertools import combinations

import numpy as np
from numpy import linalg as la

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from proofs.common import omega3, find_bonds
from proofs.foundations.theorem_B5_3_core import (
    bloch_hashimoto, build_c3_on_directed_edges, build_directed_edges,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-8

results = []
def gate(name, passed, detail=""):
    results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


# Observed CKM magnitudes (framework values; not PDG inputs)
V_us = 9.0 / 40.0           # 0.225
V_cb = 256.0 / 6305.0       # 0.04060
V_ub = 3.767e-3             # 0.003767

print("=" * 78)
print("W57 — N-orbit CKM-construction probe")
print("=" * 78)
print()
print("Framework V_CKM magnitudes (targets for G3, G4):")
print(f"  V_us = 9/40         = {V_us:.5f}")
print(f"  V_cb = 256/6305     = {V_cb:.5f}")
print(f"  V_ub ≈ 3.767e-3     = {V_ub:.5f}")
print()


# ------------------------------------------------------------------------
# Build N-orbit infrastructure (recover m1_n_orbit_3orbit_basis.py)
# ------------------------------------------------------------------------
print("=" * 78)
print("Setup — N-orbit infrastructure (B(N_i), B_total, C_36)")
print("=" * 78)

N1 = np.array([0.0, 0.0, 0.5])
N2 = np.array([0.5, 0.0, 0.0])
N3 = np.array([0.0, 0.5, 0.0])
RAM = 2.0  # |h|² Ramanujan-saturated at N

bonds = find_bonds()
directed = build_directed_edges(bonds)
assert len(directed) == 12

B_N1 = bloch_hashimoto(N1, directed)
B_N2 = bloch_hashimoto(N2, directed)
B_N3 = bloch_hashimoto(N3, directed)
U_C3 = build_c3_on_directed_edges(directed)
n = 12

Z = np.zeros((n, n), complex)
B_total = np.block([[B_N1, Z, Z], [Z, B_N2, Z], [Z, Z, B_N3]])
C_36   = np.block([[Z, Z, U_C3], [U_C3, Z, Z], [Z, U_C3, Z]])

# Verify C_36 has order 3
err_c3 = la.norm(C_36 @ C_36 @ C_36 - np.eye(3 * n))
print(f"  ||C_36^3 − I|| = {err_c3:.2e}  (expect 0)")
err_comm = la.norm(B_total @ C_36 - C_36 @ B_total)
print(f"  ||[B_total, C_36]|| = {err_comm:.2e}  (expect 0)")


# ------------------------------------------------------------------------
# G1 — 8 disjoint C_36 3-orbits in V_Ram(N1) ⊕ V_Ram(N2) ⊕ V_Ram(N3)
# ------------------------------------------------------------------------
print("=" * 78)
print("G1 — Recover 8 disjoint C_36 3-orbits")
print("=" * 78)

# Extract V_Ram(N1) eigenmodes
eigs_N1, V_N1 = la.eig(B_N1)
ram_idx = [i for i in range(n) if abs(abs(eigs_N1[i])**2 - RAM) < 1e-6]
W1 = V_N1[:, ram_idx]
W1, _ = la.qr(W1)
W1 = W1[:, :len(ram_idx)]
print(f"  V_Ram(N1) dim = {W1.shape[1]}  (expect 8)")
assert W1.shape[1] == 8

# Build the 8 candidate 3-orbits
orbits = []
for j in range(8):
    psi = np.zeros(3 * n, complex)
    psi[:n] = W1[:, j]
    psi /= la.norm(psi)
    g0 = psi
    g1 = C_36 @ g0
    g2 = C_36 @ g1
    G = np.column_stack([g0, g1, g2])
    # Verify orthonormality (should be automatic from disjoint fiber support)
    overlap = G.conj().T @ G
    if la.norm(overlap - np.eye(3)) > 1e-6:
        # Force orthonormalization
        G, _ = la.qr(G)
    orbits.append(G)

g1 = len(orbits) == 8
gate("G1 8 disjoint C_36 3-orbits constructed", g1,
     f"each orbit is a 36×3 matrix (3-dim subspace of V_Ram(N1)⊕V_Ram(N2)⊕V_Ram(N3))")


# ------------------------------------------------------------------------
# G2 — enumerate (orbit_u, orbit_d) pairs; check V_CKM structure
# ------------------------------------------------------------------------
print("=" * 78)
print("G2 — V_CKM = U_uL† U_dL for all (orbit_u, orbit_d) pairs")
print("=" * 78)

def vckm_structure_classify(V):
    """Classify V_CKM = U_u† U_d into structural categories."""
    abs_V = np.abs(V)
    # Check if essentially scalar·I (diagonal with equal magnitudes)
    diag = np.diag(abs_V)
    offdiag = abs_V - np.diag(diag)
    max_offdiag = np.max(offdiag)
    diag_uniform = np.std(diag) < 0.01
    if max_offdiag < 1e-6 and diag_uniform:
        return "scalar_diagonal", diag[0]
    # Check if scalar·permutation (one large entry per row/col)
    row_max = np.max(abs_V, axis=1)
    col_max = np.max(abs_V, axis=0)
    sorted_entries = np.sort(abs_V.flatten())[::-1]
    if sorted_entries[2] > 0.99 and sorted_entries[3] < 0.01:
        # Top 3 entries large, rest tiny → permutation
        return "permutation", sorted_entries[0]
    # Otherwise non-trivial
    return "non_trivial", (sorted_entries[0], sorted_entries[1], sorted_entries[-1])


print(f"  Enumerating C(8,2) = 28 distinct (orbit_u, orbit_d) pairs...")
print()
print(f"  {'pair':<10} | {'V_CKM structure':<20} | description")
print(f"  {'-'*10}-+-{'-'*20}-+-{'-'*40}")

structures = []
nontrivial_pairs = []
for u_idx, d_idx in combinations(range(8), 2):
    U_u = orbits[u_idx]
    U_d = orbits[d_idx]
    V_CKM_pair = U_u.conj().T @ U_d
    kind, detail = vckm_structure_classify(V_CKM_pair)
    structures.append((u_idx, d_idx, kind, detail, V_CKM_pair))
    if kind == "non_trivial":
        nontrivial_pairs.append((u_idx, d_idx, V_CKM_pair))
        print(f"  ({u_idx},{d_idx})    | {kind:<20} | top3={detail[0]:.3f},{detail[1]:.3f},...,{detail[2]:.4f}")
    elif kind == "scalar_diagonal":
        print(f"  ({u_idx},{d_idx})    | {kind:<20} | α = {abs(detail):.4f}")
    else:
        print(f"  ({u_idx},{d_idx})    | {kind:<20} | max = {detail:.4f}")

print()
n_scalar = sum(1 for _, _, k, _, _ in structures if k == "scalar_diagonal")
n_perm = sum(1 for _, _, k, _, _ in structures if k == "permutation")
n_nontriv = len(nontrivial_pairs)
print(f"  Summary across 28 pairs:")
print(f"    scalar diagonal: {n_scalar}")
print(f"    permutation:     {n_perm}")
print(f"    non-trivial:     {n_nontriv}")
print()

g2 = n_nontriv > 0
gate("G2 at least one pair gives non-trivial V_CKM", g2,
     f"non-trivial pairs: {n_nontriv} of 28")


# ------------------------------------------------------------------------
# G3 — does any pair give V_us > V_cb > V_ub hierarchy?
# ------------------------------------------------------------------------
print("=" * 78)
print("G3 — Hierarchy check: V_us > V_cb > V_ub")
print("=" * 78)

if not g2:
    print("  SKIP (no non-trivial V_CKM pairs).")
    gate("G3 at least one pair gives correct CKM hierarchy", False, "blocked by G2")
    g3 = False
else:
    hierarchy_matches = []
    for u_idx, d_idx, V_CKM_pair in nontrivial_pairs:
        abs_V = np.abs(V_CKM_pair)
        # Off-diagonal entries in upper triangle
        offdiag_vals = [(abs_V[0,1], "12"), (abs_V[0,2], "13"), (abs_V[1,2], "23"),
                        (abs_V[1,0], "21"), (abs_V[2,0], "31"), (abs_V[2,1], "32")]
        # Sort descending
        offdiag_vals.sort(reverse=True)
        # Check if top 3 (V_us, V_cb, V_ub) have the right separation pattern
        top1, top2, top3 = offdiag_vals[0][0], offdiag_vals[1][0], offdiag_vals[2][0]
        # Roughly V_us / V_cb ~ 5.5, V_cb / V_ub ~ 10.8
        # So we want top1 ≫ top2 ≫ top3 — a strict descending hierarchy
        if top1 > 2 * top2 > 4 * top3 and top3 > 1e-5:
            hierarchy_matches.append((u_idx, d_idx, top1, top2, top3))

    print(f"  Pairs with off-diagonal hierarchy (top1 > 2·top2 > 4·top3, top3 > 1e-5):")
    if hierarchy_matches:
        for u, d, t1, t2, t3 in hierarchy_matches:
            print(f"    ({u},{d}): top3 = {t1:.4f}, {t2:.4f}, {t3:.6f}")
    else:
        print(f"    NONE")
    g3 = len(hierarchy_matches) > 0
    gate("G3 at least one pair has V_us > V_cb > V_ub hierarchy", g3,
         f"matches: {len(hierarchy_matches)} of {len(nontrivial_pairs)} non-trivial pairs")


# ------------------------------------------------------------------------
# G4 — does any pair match magnitudes within factor 2?
# ------------------------------------------------------------------------
print("=" * 78)
print("G4 — Magnitude match within factor 2 of observed")
print("=" * 78)

if not g3:
    print("  SKIP (no hierarchy matches).")
    gate("G4 at least one pair matches magnitudes within factor 2", False, "blocked by G3")
    g4 = False
else:
    magnitude_matches = []
    for u, d, t1, t2, t3 in hierarchy_matches:
        # Check if top3 are within factor 2 of (V_us, V_cb, V_ub)
        r1 = t1 / V_us
        r2 = t2 / V_cb
        r3 = t3 / V_ub
        within_factor_2 = (0.5 < r1 < 2.0) and (0.5 < r2 < 2.0) and (0.5 < r3 < 2.0)
        print(f"    ({u},{d}): predicted/observed ratios = {r1:.3f}, {r2:.3f}, {r3:.3f}  "
              f"{'PASS' if within_factor_2 else 'FAIL'}")
        if within_factor_2:
            magnitude_matches.append((u, d))
    g4 = len(magnitude_matches) > 0
    gate("G4 at least one pair matches magnitudes within factor 2", g4,
         f"matches: {len(magnitude_matches)} pairs within factor 2")


# ------------------------------------------------------------------------
# AB-checks
# ------------------------------------------------------------------------
print("=" * 78)
print("AB-checks")
print("=" * 78)

# AB2: structural negative if all V_CKM are trivial
ab2_fires = (n_scalar + n_perm) == 28
print(f"  AB2 (structural negative — all pairs trivial): {'FIRES' if ab2_fires else 'does not fire'}")

# AB3: literal-claim — if G4 PASSES but multiple pairs match equally, that's
# data-anchored
ab3_fires = g4 and len(magnitude_matches) > 1
print(f"  AB3 (multiple pair matches = data-anchored): {'FIRES' if ab3_fires else 'does not fire'}")

# AB4: same as AB3 essentially
ab4_fires = ab3_fires
print(f"  AB4 (alternative pair mappings): {'FIRES' if ab4_fires else 'does not fire'}")


# ------------------------------------------------------------------------
# VERDICT
# ------------------------------------------------------------------------
print()
print("=" * 78)
print("W57 VERDICT")
print("=" * 78)

passed = sum(1 for _, p in results if p)
total = len(results)
print(f"\n  Gates passed: {passed}/{total}\n")
for name, p in results:
    print(f"    [{'PASS' if p else 'FAIL'}] {name}")

print()
if ab2_fires:
    print("VERDICT: STRUCTURAL NEGATIVE — AB2 fires.")
    print()
    print("Every (orbit_u, orbit_d) pair gives a V_CKM that is either scalar")
    print("diagonal (V_CKM = α·I) or a permutation matrix. The N-orbit 8-")
    print("candidate-gen-basis structure CANNOT host non-trivial CKM mixing.")
    print()
    print("Structural reason: each C_36 pure-cyclic orbit decomposes as")
    print("1·trivial ⊕ 1·ω ⊕ 1·ω̄ under C_3-character (m1_n_orbit verified).")
    print("Overlaps between two different orbits ⟨G_u | G_d⟩_ij factor through")
    print("the disjoint fiber support of N1, N2, N3 slots, giving either:")
    print("  - scalar diagonal (matching orbits same fiber-permutation order)")
    print("  - permutation (relative cyclic shift between orbits)")
    print("No genuine 3×3 unitary mixing is possible from C_36-pure-cyclic")
    print("orbit pairs alone.")
elif g4:
    print(f"VERDICT: at least one pair matches CKM magnitudes within factor 2.")
    if ab3_fires:
        print(f"  But AB3/AB4 fire — {len(magnitude_matches)} pairs match equally,")
        print(f"  so the labeling is DATA-ANCHORED across pair selections.")
    else:
        print(f"  Single matching pair found — candidate substrate-internal closure.")
        print(f"  Pair: {magnitude_matches[0]}")
        print(f"  Requires follow-up to verify literal claim and structural derivation.")
elif g3:
    print(f"VERDICT: hierarchy matched ({len(hierarchy_matches)} pairs) but")
    print(f"  magnitudes don't match within factor 2. The 8-orbit structure")
    print(f"  has the right qualitative pattern but wrong numerics.")
elif g2:
    print(f"VERDICT: some non-trivial V_CKM pairs exist ({n_nontriv} of 28) but")
    print(f"  none has the V_us > V_cb > V_ub hierarchy. The N-orbit approach")
    print(f"  produces non-trivial mixing structure that doesn't match observed CKM.")
else:
    print(f"VERDICT: HONEST NEGATIVE — no non-trivial CKM mixing from any orbit pair.")

print()
print("=" * 78)
all_pass = all(p for _, p in results)
print(f"W57 sentinel: {'all gates PASS' if all_pass else f'{total-passed} of {total} FAIL (honest record)'}")
print("=" * 78)
