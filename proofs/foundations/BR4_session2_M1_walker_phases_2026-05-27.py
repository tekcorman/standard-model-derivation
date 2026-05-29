#!/usr/bin/env python3
"""
BR4 session 2 — M1 twisted walker PHASES at L=8 across 8 V_Ram seeds.

Test of Candidate C from
an internal working note §7:

    δ^(s)  =  arg ⟨ g_{L mod 3} | T^L | g_0 ⟩

where T = B_total · C_36 is the M1 twisted walker, L = 8 (the ΔGen=1
lepton-cycle length), and g_0 is a species-relevant V_Ram(N₁) seed.

The M1 program closed the SQUARED amplitudes (V_cb, V_ub at theorem-grade).
This probe reads the *complex phases* of the same matrix elements for each
of the 8 V_Ram(N₁) seeds; if any seed's phase matches an empirical δ^(s)
within reasonable tolerance, Candidate C is structurally supported.

Five gates:
  G1: Reconstruct T = B_total · C_36; verify 8 V_Ram(N₁) seeds + cyclic basis.
  G2: For each seed mode ψ_i (i=0..7):
       - Build 3-orbit G_i = (ψ_i, C_36 ψ_i, C_36² ψ_i).
       - Compute amp_i = ⟨G_i[2] | T^8 | G_i[0]⟩.
       - Extract arg(amp_i), folded to Koide fundamental domain [0, 2π/3).
  G3: Are the 8 phases distinct? Or do they cluster (e.g. 4+2+2)?
  G4: Does any seed's phase match empirical δ^(s) (lepton 2/9 = 12.73°,
      down ≈ 6°, up ≈ 4°)? Honest report — closure if any matches.
  G5: L-dependence: compute the same 8 phases at L = 2, 4, 6, 8, 10, 14.
      Tests whether phase is independent of L (universal seed-phase) or
      L-dependent (per-walk phase).

Per W58 / no-numerology discipline: phases are COMPUTED from T directly.
No fits to empirical δ. The probe is "look at what's there" — pass/fail.
"""

import sys
import os
import math
import numpy as np
from numpy import linalg as la

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from proofs.common import find_bonds
from proofs.foundations.theorem_B5_3_core import (
    bloch_hashimoto, build_c3_on_directed_edges, build_directed_edges,
)


N1 = np.array([0.0, 0.0, 0.5])
N2 = np.array([0.5, 0.0, 0.0])
N3 = np.array([0.0, 0.5, 0.0])
RAMANUJAN_MOD_SQ = 2.0  # k* − 1 for k* = 3


def _build():
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    B = [bloch_hashimoto(N, directed) for N in (N1, N2, N3)]
    U_C3 = build_c3_on_directed_edges(directed)
    n = 12
    Z = np.zeros((n, n), complex)
    B_total = np.block([[B[0], Z, Z], [Z, B[1], Z], [Z, Z, B[2]]])
    C_36 = np.block([[Z, Z, U_C3], [U_C3, Z, Z], [Z, U_C3, Z]])
    return B_total, C_36, B[0], n


def _v_ram_n1(B_N1, n=12, tol=1e-6):
    eigs, V = la.eig(B_N1)
    ram_idx = [i for i in range(n) if abs(abs(eigs[i]) ** 2 - RAMANUJAN_MOD_SQ) < tol]
    # Use raw eigenvectors (not QR-orthogonalised) so each is a true B(N1) eigenvector
    return eigs[ram_idx], V[:, ram_idx]


def _build_orbit(seed, C_36, n=12):
    """Embed seed in N1-slot of the 36-dim space; act with C_36 to populate other slots."""
    g0 = np.zeros(3 * n, complex)
    g0[:n] = seed / la.norm(seed)
    g1 = C_36 @ g0
    g2 = C_36 @ g1
    return [g0, g1, g2]


def _fold_koide(phi_rad):
    """Fold phase into Koide fundamental domain [0, 2π/3); also report small branch."""
    period = 2 * math.pi / 3.0
    wrapped = phi_rad % period
    small = min(wrapped, period - wrapped)
    return wrapped, small


# -----------------------------------------------------------------------
# G1 — reconstruct + structural checks
# -----------------------------------------------------------------------

def G1_reconstruct():
    print("=" * 78)
    print("G1 — Reconstruct T = B_total · C_36; verify V_Ram(N1) dim 8")
    print("=" * 78)
    B_total, C_36, B_N1, n = _build()
    T = B_total @ C_36
    # C_36 order 3
    err_c3 = la.norm(C_36 @ C_36 @ C_36 - np.eye(3 * n))
    # commutes with B_total
    err_comm = la.norm(B_total @ C_36 - C_36 @ B_total)
    # V_Ram(N1)
    eigs, V = _v_ram_n1(B_N1, n=n)
    print(f"  ||C_36^3 − I||_F = {err_c3:.2e}")
    print(f"  ||[B_total, C_36]||_F = {err_comm:.2e}")
    print(f"  V_Ram(N1) dim = {V.shape[1]}  (expect 8)")
    print(f"  Ramanujan eigenvalues at N1: {[f'{e:.4f}' for e in eigs]}")
    passed = err_c3 < 1e-10 and err_comm < 1e-10 and V.shape[1] == 8
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed, T, V, n, C_36


# -----------------------------------------------------------------------
# G2 — phases of L=8 matrix element across 8 seeds
# -----------------------------------------------------------------------

def G2_phases_per_seed(T, V_ram, C_36, n):
    print()
    print("=" * 78)
    print("G2 — 8 V_Ram(N1) seed phases of ⟨g_2 | T^8 | g_0⟩")
    print("=" * 78)
    T_pow_8 = np.linalg.matrix_power(T, 8)
    results = []
    print(f"  {'i':>2} {'h_N1':>14} {'|amp|²':>10} {'arg(amp)':>12} {'arg(amp) °':>14} {'mod 2π/3 °':>14} {'small branch °':>16}")
    for i in range(V_ram.shape[1]):
        seed = V_ram[:, i]
        G = _build_orbit(seed, C_36, n=n)
        amp = G[2].conj() @ T_pow_8 @ G[0]
        amp_sq = abs(amp) ** 2
        phi = np.angle(amp)
        wrapped, small = _fold_koide(phi)
        results.append({
            "i": i,
            "amp": amp,
            "amp_sq": amp_sq,
            "phi_rad": phi,
            "phi_deg": math.degrees(phi),
            "wrapped_rad": wrapped,
            "wrapped_deg": math.degrees(wrapped),
            "small_branch_deg": math.degrees(small),
        })
        h_N1 = np.linalg.eigvals(T[:n, n:2*n])[0]  # not great, use eigs from G1 directly
        # Just report
        print(f"  {i:>2} {'(seed)':>14} {amp_sq:>10.4f} {phi:>+12.6f} {math.degrees(phi):>+14.4f} "
              f"{math.degrees(wrapped):>+14.4f} {math.degrees(small):>+16.4f}")
    return results


# -----------------------------------------------------------------------
# G3 — phase clustering analysis
# -----------------------------------------------------------------------

def G3_clustering(results):
    print()
    print("=" * 78)
    print("G3 — Phase clustering: 8 distinct? or 4+2+2? or 8 equal?")
    print("=" * 78)
    small_angles = sorted([r["small_branch_deg"] for r in results])
    print(f"  Sorted small-branch angles (°): {[f'{a:.3f}' for a in small_angles]}")
    # Cluster by approximate equality (within 0.1°)
    clusters = []
    for a in small_angles:
        placed = False
        for c in clusters:
            if abs(c[0] - a) < 0.1:
                c.append(a)
                placed = True
                break
        if not placed:
            clusters.append([a])
    print(f"  Clusters (Δ < 0.1°): {[len(c) for c in clusters]}")
    print(f"  Cluster centres (°): {[f'{sum(c)/len(c):.4f}' for c in clusters]}")
    return clusters


# -----------------------------------------------------------------------
# G4 — empirical δ matching
# -----------------------------------------------------------------------

def G4_match_empirical(results, clusters):
    print()
    print("=" * 78)
    print("G4 — Does any seed's phase match empirical δ^(s)?")
    print("=" * 78)
    targets = {
        "δ_lepton (theorem 2/9 rad)": math.degrees(2.0 / 9.0),
        "δ_down (PDG 2 GeV ~5.8°)": 5.80,
        "δ_down (PDG m_b scheme ~6.31°)": 6.31,
        "δ_up (PDG ~4.27°)": 4.27,
    }
    matched = []
    for label, t_deg in targets.items():
        # Find the closest cluster centre
        cluster_centres = [sum(c) / len(c) for c in clusters]
        best = min(cluster_centres, key=lambda c: abs(c - t_deg))
        diff = abs(best - t_deg)
        rel = diff / t_deg
        verdict = (
            "MATCH" if rel < 0.05 else
            "NEAR" if rel < 0.15 else
            "NO"
        )
        print(f"  {label:>34}: target {t_deg:>7.4f}° vs closest cluster {best:>7.4f}° "
              f"(Δ = {diff:>6.4f}°, {rel*100:>6.2f}%) → {verdict}")
        if verdict in ("MATCH", "NEAR"):
            matched.append((label, t_deg, best, rel))
    print()
    if matched:
        print("  RESULT: At least one empirical δ matched to within 15% of a seed phase.")
        print("  This is suggestive structural evidence for Candidate C.")
    else:
        print("  RESULT: No empirical δ matched within 15%. Candidate C downgraded.")
    return matched


# -----------------------------------------------------------------------
# G5 — L-dependence per seed
# -----------------------------------------------------------------------

def G5_L_dependence(T, V_ram, C_36, n):
    print()
    print("=" * 78)
    print("G5 — Phase at varying L: L ∈ {2, 4, 6, 8, 10, 14}")
    print("=" * 78)
    Ls = [2, 4, 6, 8, 10, 14]
    print(f"  {'i':>2} | " + " | ".join([f"arg(L={L}) °" for L in Ls]))
    for i in range(V_ram.shape[1]):
        seed = V_ram[:, i]
        G = _build_orbit(seed, C_36, n=n)
        row = [f"{i:>2}"]
        for L in Ls:
            M = np.linalg.matrix_power(T, L)
            end = L % 3
            amp = G[end].conj() @ M @ G[0]
            phi = math.degrees(np.angle(amp))
            _, small = _fold_koide(np.angle(amp))
            row.append(f"{math.degrees(small):>+8.4f}")
        print(f"  {' | '.join(row)}")
    print()
    print("  Reading: stable column = L-independent seed phase (universal mechanism)")
    print("           shifting column = L-dependent (per-walk accumulation)")


# -----------------------------------------------------------------------
# G6 — GAUGE-INVARIANT triple-product (Jarlskog-analog for M_gen)
# -----------------------------------------------------------------------

def G6_gauge_invariant_triple_product(T, V_ram, C_36, n):
    """
    The single matrix element ⟨g_(L mod 3) | T^L | g_0⟩ has GAUGE-DEPENDENT
    phase: g_i → e^{iα_i}·g_i rotates ⟨g_a | T^L | g_b⟩ → e^{-iα_a+iα_b}·amp.

    The Koide phase δ is gauge-invariant. The natural gauge-invariant
    construction is the triple product around the C₃ loop:

        Triple^(s)(L) = ⟨g_1|T^L|g_0⟩ · ⟨g_2|T^L|g_1⟩ · ⟨g_0|T^L|g_2⟩

    Under gauge g_i → e^{iα_i}g_i: each factor picks up phases that cancel
    around the closed loop. Hence Triple is gauge-invariant.

    For cyclic-Toeplitz Hermitian R = 1 + a_1·P + a_1*·P†:
        ⟨g_(i+1)|R|g_i⟩ = a_1 for all i (cyclic-invariant)
        Triple = a_1³ → arg(Triple) = 3·arg(a_1) = 3δ
    Hence:
        δ_extracted = (1/3) · arg(Triple) mod 2π/3
    """
    print()
    print("=" * 78)
    print("G6 — Gauge-invariant triple product (Jarlskog-analog)")
    print("=" * 78)
    print("  Triple^(s)(L) = ⟨g_1|T^L|g_0⟩ · ⟨g_2|T^L|g_1⟩ · ⟨g_0|T^L|g_2⟩")
    print("  δ = (1/3) · arg(Triple) mod 2π/3")
    print()
    print(f"  Direction-of-cycle convention: use ΔGen = (L mod 3) per edge")
    print(f"    L mod 3 = 1 → forward cycle 0→1→2→0 (gen +1 per step)")
    print(f"    L mod 3 = 2 → backward cycle 0→2→1→0 (gen -1 per step) = forward with -δ")
    print(f"    L mod 3 = 0 → diagonal (no off-diagonal, δ trivially 0)")
    print()
    print(f"  {'i':>2} | " + " | ".join([f"δ(L={L}) °" for L in [2, 4, 6, 8, 10, 14]]))
    triple_phases = {L: [] for L in [2, 4, 6, 8, 10, 14]}
    for i in range(V_ram.shape[1]):
        seed = V_ram[:, i]
        G = _build_orbit(seed, C_36, n=n)
        row = [f"{i:>2}"]
        for L in [2, 4, 6, 8, 10, 14]:
            M = np.linalg.matrix_power(T, L)
            step = L % 3
            if step == 0:
                # Diagonal — triple is real
                triple = (G[0].conj() @ M @ G[0]) ** 3
            else:
                # Each edge advances by `step` in mod 3
                a01 = G[(0 + step) % 3].conj() @ M @ G[0]
                a12 = G[(1 + step) % 3].conj() @ M @ G[1]
                a20 = G[(2 + step) % 3].conj() @ M @ G[2]
                triple = a01 * a12 * a20
            delta = np.angle(triple) / 3.0
            _, small = _fold_koide(delta)
            row.append(f"{math.degrees(small):>+8.4f}")
            triple_phases[L].append(math.degrees(small))
        print(f"  {' | '.join(row)}")

    print()
    print("  Match empirical δ:")
    targets = {
        "δ_lepton (2/9 rad = 12.73°)": 12.73,
        "δ_down (~5.8-6.3°)": 6.0,
        "δ_up (~4.27°)": 4.27,
    }
    matches = []
    for L in [2, 4, 6, 8, 10, 14]:
        phases = sorted(set(round(p, 3) for p in triple_phases[L]))
        for label, t in targets.items():
            best = min(phases, key=lambda p: abs(p - t))
            rel = abs(best - t) / t
            verdict = "MATCH" if rel < 0.05 else "NEAR" if rel < 0.15 else "NO"
            if verdict in ("MATCH", "NEAR"):
                print(f"  L={L:>2}, {label}: target {t:.3f}°, closest {best:.3f}° "
                      f"(rel {rel*100:.2f}%) → {verdict}")
                matches.append((L, label, t, best, rel))
    if not matches:
        print("  No NEAR/MATCH at any L tested.")
    return matches

if __name__ == "__main__":
    print()
    print("BR4 SESSION 2 PROBE — M1 walker phases at L=8 across 8 V_Ram(N1) seeds")
    print("=" * 78)
    print("Per an internal working note §7-§8")
    print()

    ok1, T, V_ram, n, C_36 = G1_reconstruct()
    results = G2_phases_per_seed(T, V_ram, C_36, n)
    clusters = G3_clustering(results)
    matched = G4_match_empirical(results, clusters)
    G5_L_dependence(T, V_ram, C_36, n)
    triple_matched = G6_gauge_invariant_triple_product(T, V_ram, C_36, n)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  G1: {'PASS' if ok1 else 'FAIL'}")
    print(f"  G2: 8 seed phases computed; see table above")
    print(f"  G3: clustering pattern {[len(c) for c in clusters]}")
    print(f"  G4: {len(matched)} empirical δ value(s) matched within 15%")
    print()
    if matched:
        print(f"  Candidate C (δ^(s) = arg(⟨g | T^L | g⟩) at species seed):")
        print(f"  STRUCTURAL EVIDENCE — {len(matched)} empirical match(es) within 15%.")
        for label, t, best, rel in matched:
            print(f"    {label}: closest cluster at {best:.3f}° (rel {rel*100:.2f}%)")
    else:
        print(f"  Candidate C: NO clean match. Probe HONEST NEGATIVE.")
    print()
    print("  Per W58 / no-numerology: this is what the substrate operator produces.")
    print("  No fits, no tuning — phases are the computed eigenvalue arguments.")
    print()
