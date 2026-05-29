#!/usr/bin/env python3
"""
A5(b) Level 3 sub-class scoping — Route 1 probe.

Reference: an internal working note §4 "Route 1".

EXPLORATORY. This script is a research probe, not a theorem-grade derivation.
Reports CAS-computed multiplicities; structural interpretation is for the
human researcher to evaluate.

QUESTION
--------
The framework's natural Z_3 (body-diagonal C_3 on srs) gives uniform (8, 8, 8)
multiplicities on V_Ram(N-orbit) (per `n_orbit_c3_multiplicities.py`). This
fails to distinguish generations.

Route 1 hypothesis: the orientation-augmented Z_2 × C_3 = Z_6 on directed
edges of Cayley(F_inv(E)) might provide the missing distinguisher. Specifically,
ΔGen=1 transitions (b→c, mu→tau, etc.) and ΔGen=2 transitions (b→u, etc.)
might accumulate distinct Z_6 phases.

PLAN
----
1. Build R: 12×12 orientation-reverse permutation on directed edges.
2. Verify R^2 = I, [R, U_C3] = 0 — confirms Z_2 × Z_3 = Z_6 acts on directed edges.
3. Build h = R · U_C3 (single generator of Z_6, order 6).
4. Compute Z_6-isotypic decomposition of V_Ram(P) (8-dim) and V_Ram(N-orbit) (24-dim).
5. Check whether R preserves V_Ram (necessary for the Z_6 action on V_Ram to be
   well-defined).
6. Report findings honestly. Do NOT claim closure.

Run with:
    PYTHONPATH=. python3 proofs/foundations/a5b_route1_z6_directed_edge_probe.py

Upstream:
  proofs/common.py
  proofs/foundations/theorem_B5_3_core.py  (build_c3_on_directed_edges)
  proofs/foundations/n_orbit_c3_multiplicities.py  (companion script for Z_3 case)
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, omega3
from proofs.foundations.theorem_B5_3_core import (
    build_directed_edges,
    bloch_hashimoto,
    build_c3_on_directed_edges,
    commutator_norm,
)


# ---------------------------------------------------------------------------
# Probe parameters
# ---------------------------------------------------------------------------

P_PT = np.array([0.25, 0.25, 0.25])
N1 = np.array([0.0, 0.0, 0.5])
N2 = np.array([0.5, 0.0, 0.0])
N3 = np.array([0.0, 0.5, 0.0])

RAMANUJAN_MOD_SQ = 2.0
TOL_COMM = 1e-10
TOL_ORDER = 1e-10
TOL_CHAR = 0.05

ZETA6 = np.exp(2j * np.pi / 6)   # primitive 6th root of unity


# ---------------------------------------------------------------------------
# Build R (orientation reverse) and verify Z_6 structure
# ---------------------------------------------------------------------------

def build_orientation_reverse(directed):
    """12×12 permutation matrix R: directed edge e ↦ reverse(e).

    For (src, tgt, cell), the reverse edge is (tgt, src, -cell).
    """
    edge_to_idx = {de: i for i, de in enumerate(directed)}
    n = len(directed)
    R = np.zeros((n, n), dtype=complex)
    for i, (src, tgt, cell) in enumerate(directed):
        rev = (tgt, src, tuple(-c for c in cell))
        j = edge_to_idx.get(rev)
        if j is None:
            raise RuntimeError(
                f"Reverse of {(src, tgt, cell)} = {rev} not in directed set"
            )
        R[j, i] = 1.0
    return R


def extract_vram(B_k, tol=1e-5, expected_ram=8):
    evals, evecs = la.eig(B_k)
    ram_idx = [i for i, ev in enumerate(evals)
               if abs(abs(ev)**2 - RAMANUJAN_MOD_SQ) < tol]
    assert len(ram_idx) == expected_ram, (
        f"Expected {expected_ram} Ramanujan eigenvalues, got {len(ram_idx)}."
    )
    evecs_raw = evecs[:, ram_idx]
    V_Ram, _ = la.qr(evecs_raw)
    V_Ram = V_Ram[:, :len(ram_idx)]
    return evals[ram_idx], V_Ram


def z6_isotypic(eigs, tol=TOL_CHAR):
    """Count eigenvalues near each Z_6 character ζ^k for k = 0..5."""
    counts = [0] * 6
    others = []
    for ev in eigs:
        best_k, best_dist = None, 1e9
        for k in range(6):
            d = abs(ev - ZETA6**k)
            if d < best_dist:
                best_dist = d
                best_k = k
        if best_dist < tol:
            counts[best_k] += 1
        else:
            others.append(ev)
    return counts, others


def restricted_eigs(M_full, W, tol_proj=1e-8):
    err = la.norm(W.conj().T @ W - np.eye(W.shape[1]))
    assert err < tol_proj, f"W not orthonormal: ||W^dag W - I|| = {err:.2e}"
    M_sub = W.conj().T @ M_full @ W
    return la.eigvals(M_sub), M_sub


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("A5(b) Level 3 sub-class — Route 1 probe (Z_2 × C_3 = Z_6 on directed edges)")
    print("=" * 76)
    print()
    print("Reference: an internal working note §4 R1")
    print("Mode: EXPLORATORY. Reports findings; does not assert closure.")
    print()

    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    U_C3 = build_c3_on_directed_edges(directed)
    R = build_orientation_reverse(directed)

    # ----------------------------------------------------------------------
    # STEP 1: verify R is a valid orientation-reverse permutation
    # ----------------------------------------------------------------------
    print("--- STEP 1: Verify R is order-2 permutation on directed edges ---")
    R_squared = R @ R
    err_R2 = la.norm(R_squared - np.eye(12))
    print(f"  ||R^2 - I|| = {err_R2:.3e}   (expected 0)")
    assert err_R2 < TOL_ORDER, f"R is not order-2: ||R^2 - I|| = {err_R2}"
    print("  R^2 = I confirmed.")

    # R should have no fixed points (each directed edge maps to a different edge)
    fixed_points = [i for i in range(12) if abs(R[i, i] - 1.0) < 1e-10]
    print(f"  Fixed points of R: {fixed_points}   (expected empty)")
    assert len(fixed_points) == 0, "R has fixed points (would mean self-reverse edges)"

    # ----------------------------------------------------------------------
    # STEP 2: verify [R, U_C3] = 0
    # ----------------------------------------------------------------------
    print("\n--- STEP 2: Verify R commutes with U_C3 (Z_2 × Z_3 = Z_6 structure) ---")
    comm_R_U = commutator_norm(R, U_C3)
    print(f"  ||[R, U_C3]|| = {comm_R_U:.3e}   (expected 0)")
    assert comm_R_U < TOL_COMM, f"R and U_C3 do not commute: {comm_R_U}"
    print("  [R, U_C3] = 0 confirmed.  Z_2 × Z_3 = Z_6 acts on directed edges.")

    # ----------------------------------------------------------------------
    # STEP 3: Build h = R · U_C3 as the Z_6 generator and verify order 6
    # ----------------------------------------------------------------------
    print("\n--- STEP 3: Build Z_6 generator h = R · U_C3 ---")
    h = R @ U_C3
    h_powers = [np.eye(12, dtype=complex)]
    for _ in range(6):
        h_powers.append(h_powers[-1] @ h)

    err_h6 = la.norm(h_powers[6] - np.eye(12))
    err_h3 = la.norm(h_powers[3] - np.eye(12))
    err_h2 = la.norm(h_powers[2] - np.eye(12))
    print(f"  ||h^6 - I|| = {err_h6:.3e}   (expected 0)")
    print(f"  ||h^3 - I|| = {err_h3:.3e}   (expected NONZERO if order is exactly 6)")
    print(f"  ||h^2 - I|| = {err_h2:.3e}   (expected NONZERO if order is exactly 6)")
    assert err_h6 < TOL_ORDER, f"h is not order 6: ||h^6 - I|| = {err_h6}"
    assert err_h3 > 1e-6, f"h has order ≤ 3 (degenerate)"
    assert err_h2 > 1e-6, f"h has order ≤ 2 (degenerate)"
    print("  h has order exactly 6.  OK.")

    # ----------------------------------------------------------------------
    # STEP 4: Z_6 multiplicities on the FULL 12-dim directed-edge space
    # ----------------------------------------------------------------------
    print("\n--- STEP 4: Z_6 isotypic on the full 12-dim directed-edge space ---")
    eigs_h_full = la.eigvals(h)
    counts_full, others_full = z6_isotypic(eigs_h_full)
    print(f"  Z_6 multiplicities on 12-dim full space:")
    for k in range(6):
        print(f"    ζ^{k} = {ZETA6**k:.4f}: {counts_full[k]}")
    if others_full:
        print(f"  Unclassified eigenvalues: {others_full}")
    assert sum(counts_full) == 12, f"missing classifications: {others_full}"
    expected_full = (2, 2, 2, 2, 2, 2)
    if tuple(counts_full) == expected_full:
        print(f"  Equal distribution {expected_full}: full directed-edge space "
              f"is the regular Z_6 representation.")
    else:
        print(f"  Distribution {tuple(counts_full)} (expected {expected_full} for regular).")

    # ----------------------------------------------------------------------
    # STEP 5: Check whether R preserves V_Ram(P) (8-dim Ramanujan at P-point)
    # ----------------------------------------------------------------------
    print("\n--- STEP 5: Z_6 isotypic on V_Ram(P) (8-dim Ramanujan at P-point) ---")
    B_P = bloch_hashimoto(P_PT, directed)

    # Check: does B(P) commute with R?
    comm_B_R = commutator_norm(B_P, R)
    print(f"  ||[B(P), R]|| = {comm_B_R:.3e}")

    if comm_B_R > TOL_COMM:
        print("  [!] B(P) does NOT commute with R.")
        print("      R does not preserve B(P)'s eigenspaces — V_Ram(P) is NOT R-invariant.")
        print("      Z_6 cannot act consistently on V_Ram(P) via this construction.")
        ram_p_invariant = False
    else:
        print("  B(P) commutes with R: V_Ram(P) is potentially R-invariant.")
        ram_p_invariant = True

    _, V_Ram_P = extract_vram(B_P)

    # Even if R doesn't fully commute with B, restrict h to V_Ram(P) and report
    # what we get — this tells us whether V_Ram(P) is at least a subspace of
    # eigenstates with consistent Z_6 labels.
    eigs_h_ramP, h_ramP = restricted_eigs(h, V_Ram_P)
    h_ramP_6 = la.matrix_power(h_ramP, 6)
    err_h_ramP_6 = la.norm(h_ramP_6 - np.eye(8))
    print(f"  ||h|_V_Ram(P)^6 - I_8|| = {err_h_ramP_6:.3e}")
    if err_h_ramP_6 > 1e-6:
        print(f"  [!] h restricted to V_Ram(P) is NOT order 6.")
        print(f"      V_Ram(P) is not stable under Z_6 — the restriction is a")
        print(f"      mixing of in-subspace and out-of-subspace components.")
        print(f"      Z_6 multiplicities on V_Ram(P) are NOT WELL-DEFINED via this restriction.")
        ram_p_z6_well_defined = False
    else:
        ram_p_z6_well_defined = True

    counts_ramP, others_ramP = z6_isotypic(eigs_h_ramP)
    print(f"  h|_V_Ram(P) eigenvalues:")
    for ev in sorted(eigs_h_ramP, key=lambda z: np.angle(z)):
        diffs = [abs(ev - ZETA6**k) for k in range(6)]
        k_best = diffs.index(min(diffs))
        tag = f"ζ^{k_best}" if min(diffs) < TOL_CHAR else "?"
        print(f"    {ev.real:+.6f}{ev.imag:+.6f}i   |ev|={abs(ev):.6f}   [{tag}]")
    print(f"  Provisional Z_6 multiplicities on V_Ram(P) (8-dim):")
    for k in range(6):
        print(f"    ζ^{k}: {counts_ramP[k]}")
    if others_ramP:
        print(f"  Unclassified: {others_ramP}  ({len(others_ramP)} eigenvalues NOT 6th roots of unity)")

    # ----------------------------------------------------------------------
    # STEP 6: Z_6 multiplicities on V_Ram(N-orbit) (24-dim)
    # ----------------------------------------------------------------------
    print("\n--- STEP 6: Z_6 isotypic on V_Ram(N-orbit) (24-dim) ---")
    print("  N-orbit C_3 already gave (8, 8, 8) uniform under U_C3 (per Z_3 probe).")
    print("  Adding Z_2 (orientation reverse) might or might not break this uniformity.")

    B_N1 = bloch_hashimoto(N1, directed)
    B_N2 = bloch_hashimoto(N2, directed)
    B_N3 = bloch_hashimoto(N3, directed)
    _, V_Ram_N1 = extract_vram(B_N1)
    _, V_Ram_N2 = extract_vram(B_N2)
    _, V_Ram_N3 = extract_vram(B_N3)

    # Build the 36-dim N-orbit space and the combined Z_6 generator
    Z = np.zeros((12, 12), dtype=complex)
    R_36 = np.block([
        [R, Z, Z],
        [Z, R, Z],
        [Z, Z, R],
    ])
    C_36 = np.block([
        [Z,    Z,    U_C3],
        [U_C3, Z,    Z   ],
        [Z,    U_C3, Z   ],
    ])
    h_36 = R_36 @ C_36
    err_h36_6 = la.norm(la.matrix_power(h_36, 6) - np.eye(36))
    print(f"  ||h_36^6 - I_36|| = {err_h36_6:.3e}")
    assert err_h36_6 < 1e-8, f"h_36 not order 6: {err_h36_6}"

    Z12_8 = np.zeros((12, 8), dtype=complex)
    W = np.block([
        [V_Ram_N1, Z12_8,    Z12_8   ],
        [Z12_8,    V_Ram_N2, Z12_8   ],
        [Z12_8,    Z12_8,    V_Ram_N3],
    ])

    # Check whether B_N_total commutes with h_36 (it should commute with C_36 already)
    B_total = np.block([
        [B_N1, Z,    Z   ],
        [Z,    B_N2, Z   ],
        [Z,    Z,    B_N3],
    ])
    comm_B_R36 = commutator_norm(B_total, R_36)
    print(f"  ||[B_total, R_36]|| = {comm_B_R36:.3e}")
    if comm_B_R36 > TOL_COMM:
        print(f"  [!] B_total does NOT commute with R_36; V_Ram(N-orbit) NOT R-invariant.")

    eigs_h_ramN, h_ramN = restricted_eigs(h_36, W)
    err_h_ramN_6 = la.norm(la.matrix_power(h_ramN, 6) - np.eye(24))
    print(f"  ||h|_V_Ram(N-orb)^6 - I_24|| = {err_h_ramN_6:.3e}")

    counts_ramN, others_ramN = z6_isotypic(eigs_h_ramN)
    print(f"  Provisional Z_6 multiplicities on V_Ram(N-orbit) (24-dim):")
    for k in range(6):
        print(f"    ζ^{k}: {counts_ramN[k]}")
    if others_ramN:
        print(f"  Unclassified: {len(others_ramN)} eigenvalues NOT 6th roots of unity")

    # ----------------------------------------------------------------------
    # STEP 7: Diagnosis
    # ----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("STRUCTURAL DIAGNOSIS")
    print("=" * 76)

    print(f"\n  P-point: ramP_z6_well_defined = {ram_p_z6_well_defined}")
    print(f"           Z_6 mult on V_Ram(P): {tuple(counts_ramP)}")
    print(f"  N-orbit: Z_6 mult on V_Ram(N-orbit): {tuple(counts_ramN)}")

    # Distinguishing-pattern test:
    # If Z_6 distinguishes ΔGen=1 from ΔGen=2 transitions, we'd expect:
    # - The "identity-coset" eigenspaces (ζ^0, ζ^3) host generation-diagonal modes
    # - The "ΔGen=1 coset" (ζ^1, ζ^4) hosts ΔGen=1 transitions (b→c, mu→tau)
    # - The "ΔGen=2 coset" (ζ^2, ζ^5) hosts ΔGen=2 transitions (b→u)
    # Specifically: a (4, 1, 1, 4, 1, 1)-style pattern would suggest the Z_2
    # part splits ΔGen=0 from ΔGen=±1, and the Z_3 part splits ΔGen=±1 cleanly.

    is_uniform_p = ram_p_z6_well_defined and len(set(counts_ramP)) == 1
    is_uniform_n = len(set(counts_ramN)) == 1

    print(f"\n  V_Ram(P) Z_6 uniform? {is_uniform_p} ({counts_ramP})")
    print(f"  V_Ram(N-orbit) Z_6 uniform? {is_uniform_n} ({counts_ramN})")

    print()
    print("  INTERPRETATION HINTS:")
    print("  - If V_Ram(N-orbit) Z_6 multiplicities are (4, 4, 4, 4, 4, 4): same uniform")
    print("    failure as the Z_3 case. Route 1 fails on N-orbit; consider routes 2/3/4.")
    print("  - If V_Ram(N-orbit) shows a (8, 0, 0, 8, 0, 0)-style pattern: the Z_2 part")
    print("    refines the uniform Z_3 (8,8,8) into orientation-graded blocks but with")
    print("    no Z_3 distinguisher inside each. Suggests the orientation reverse alone")
    print("    is not the missing distinguisher.")
    print("  - If V_Ram(N-orbit) shows a non-uniform DIFFERENT pattern: investigate.")
    print()
    print("  In all cases this script reports CAS findings. Closure of A5(b) sub-class")
    print("  identification requires structural-physical interpretation by the researcher.")

    return {
        'P_z6_mult': tuple(counts_ramP),
        'P_z6_well_defined': ram_p_z6_well_defined,
        'N_orbit_z6_mult': tuple(counts_ramN),
        'B_R_commutator_norm_P': float(comm_B_R),
        'B_R_commutator_norm_N': float(comm_B_R36),
    }


if __name__ == "__main__":
    result = main()
    print()
    print("=" * 76)
    print("STRUCTURED RESULT (for cross-referencing in scoping doc):")
    print(f"  {result}")
    print("=" * 76)
