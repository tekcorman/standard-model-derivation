#!/usr/bin/env python3
"""
pi_v_drude_weight_bloch_invariants_probe.py — analytical handle on Π_v's d.

Conceptual goal. The Π_JJ Kubo close-out (`gauge_beta_substrate_kubo_thread_closeout_2026-05-14.md`)
identified the structural gap: Π_TT (G_sub) has a clean analytical Drude
weight D_TT = -1/(⟨Tr H²⟩ × k*) = -1/36 derived from first principles
(`theorem_g_sub_drude_closure_2026-04-30.md`), but Π_JJ (gauge) lacks
the analog independent derivation for d_Π_v. Without it, the Kubo
extraction is method-dependent at the 10% level.

This probe computes candidate Bloch invariants of the velocity operator
v^μ(k) = ∂H/∂k_μ on srs's 4-atom adjacency Bloch H_bloch, and tests
which combination(s) give a structural denominator matching the
pairwise-extracted d ≈ -1/137.4 (from the point-eval probe at N=16).

Approach. Compute these BZ-averaged Bloch invariants (all are static,
no Kubo loop; just (1/V_BZ) ∫_BZ Tr[...] d³k):

  I1 = ⟨Tr v²⟩         = sum over μ of ⟨Tr (v^μ)²⟩  (analog of ⟨Tr H²⟩ = 12)
  I2 = ⟨Tr v⁴⟩         = ⟨Tr (v^μ v^μ v^ν v^ν)⟩  (4th moment, summed over μ,ν)
  I3 = ⟨Tr (v² H)⟩     = ⟨Tr (v^μ v^μ H)⟩
  I4 = ⟨Tr (v_μ H v^μ)⟩ = analog of velocity-mediated 2-vertex correlator
  I5 = ⟨Tr (v² H²)⟩    = velocity × matter-spectrum-squared
  I6 = ⟨|v_⊥|²⟩         = average squared velocity ortho to bond direction
  I7 = ⟨Tr v² × Tr H²⟩ = product of separate moments (= I1 × 12)

Structural-product candidates to compare against d_target ≈ -1/137.4:
  C1 = I1 × k*           (direct analog of ⟨Tr H²⟩ × k*)
  C2 = I3 × k*
  C3 = I1 × g            (girth instead of k*)
  C4 = I1 × N_atoms²
  C5 = I3 × N_atoms
  C6 = I5 / k*²
  ...

The probe enumerates these and reports which gives 137.4 ± small%.

Note: the BZ integration normalisation here is the same as
`lorentz_sig_g_sub_elastic_moduli.H_bloch` — half_extent = 2π,
MP-shifted grid. Cubic isotropy of srs ensures the tensor structure
of the invariants reduces to cleanly-defined scalars.
"""
from __future__ import annotations

import os
import sys
import time
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lorentz_sig_g_sub_elastic_moduli import (
    BOND_DISPLACEMENTS, H_bloch,
)
from gauge_beta_from_substrate_kubo_probe import velocity_matrix


def BZ_average(integrand_fn, N=12, half_extent=2 * np.pi):
    """Average a function over the BZ via MP-shifted grid."""
    ks = (np.arange(N) + 0.5) * (2 * half_extent / N) - half_extent
    total = 0.0
    n = 0
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                total += integrand_fn(np.array([k1, k2, k3]))
                n += 1
    return total / n


def Tr_H2_at_k(k):
    """Tr[H(k)²]."""
    H = H_bloch(k)
    return float(np.real(np.trace(H @ H)))


def Tr_v2_at_k(k):
    """Sum over μ of Tr[v^μ(k) v^μ(k)]."""
    total = 0.0
    for mu in range(3):
        v = velocity_matrix(k, mu)
        total += float(np.real(np.trace(v @ v)))
    return total


def Tr_v4_at_k(k):
    """Sum over μ, ν of Tr[v^μ v^μ v^ν v^ν]."""
    total = 0.0
    for mu in range(3):
        v_mu = velocity_matrix(k, mu)
        for nu in range(3):
            v_nu = velocity_matrix(k, nu)
            total += float(np.real(np.trace(v_mu @ v_mu @ v_nu @ v_nu)))
    return total


def Tr_v2_H_at_k(k):
    """Sum over μ of Tr[v^μ v^μ H]."""
    H = H_bloch(k)
    total = 0.0
    for mu in range(3):
        v = velocity_matrix(k, mu)
        total += float(np.real(np.trace(v @ v @ H)))
    return total


def Tr_vHv_at_k(k):
    """Sum over μ of Tr[v^μ H v^μ]."""
    H = H_bloch(k)
    total = 0.0
    for mu in range(3):
        v = velocity_matrix(k, mu)
        total += float(np.real(np.trace(v @ H @ v)))
    return total


def Tr_v2_H2_at_k(k):
    """Sum over μ of Tr[v^μ v^μ H²]."""
    H = H_bloch(k)
    H2 = H @ H
    total = 0.0
    for mu in range(3):
        v = velocity_matrix(k, mu)
        total += float(np.real(np.trace(v @ v @ H2)))
    return total


def Tr_v2H_v2_at_k(k):
    """Tr[(v² · H · v²)] — diamagnetic-style operator."""
    # Build v² operator first (sum over μ)
    v2 = np.zeros((4, 4), dtype=complex)
    for mu in range(3):
        v = velocity_matrix(k, mu)
        v2 += v @ v
    H = H_bloch(k)
    return float(np.real(np.trace(v2 @ H @ v2)))


def f_sum_rule_integrand(k):
    """Drude weight integrand at p=0, zero T: Σ_{n filled, m unfilled} (E_n - E_m) |⟨v^x⟩|².

    This is the static (ω → ∞ limit of Kubo) f-sum rule at p=0.
    """
    H = H_bloch(k)
    eigs, U = np.linalg.eigh(H)
    v_x = velocity_matrix(k, 0)
    v_in_basis = U.conj().T @ v_x @ U
    # Filled: E < 0; unfilled: E > 0 (μ = 0 half-filling)
    filled = eigs < -1e-9
    unfilled = eigs > 1e-9
    total = 0.0
    for n in np.where(filled)[0]:
        for m in np.where(unfilled)[0]:
            total += (eigs[n] - eigs[m]) * abs(v_in_basis[m, n]) ** 2
    return total


def f_sum_rule_integrand_at_p(k, p_z, mu=0.0):
    """Drude weight integrand at FINITE p_z, zero T (sharp Fermi step at μ=0).

    Σ_{n filled at k, m unfilled at k+p} (E_n(k) - E_m(k+p)) × |⟨m,k+p|v^x|n,k⟩|²

    This generalises f_sum_rule_integrand to nonzero external p_z; the leading
    p² coefficient (after BZ integration) gives d_phys for π_2_xx analytically.
    """
    k_plus_p = k + np.array([0.0, 0.0, p_z])
    k_mid = k + np.array([0.0, 0.0, p_z / 2])
    H_k = H_bloch(k)
    H_kp = H_bloch(k_plus_p)
    eigs_k, U_k = np.linalg.eigh(H_k)
    eigs_kp, U_kp = np.linalg.eigh(H_kp)
    # Sharp Fermi step
    filled_k = eigs_k < mu - 1e-9
    unfilled_kp = eigs_kp > mu + 1e-9
    # Velocity vertex at the symmetric midpoint k_mid (matches Π_v's symmetric vertex convention)
    v_x_mid = velocity_matrix(k_mid, 0)
    v_in_basis = U_kp.conj().T @ v_x_mid @ U_k
    total = 0.0
    for n in np.where(filled_k)[0]:
        for m in np.where(unfilled_kp)[0]:
            total += (eigs_k[n] - eigs_kp[m]) * abs(v_in_basis[m, n]) ** 2
    return total


def extract_d_from_fsum(N=16, p_z_values=(0.0, 0.05, 0.10, 0.15, 0.20)):
    """Compute leading p² coefficient of ⟨f-sum(p_z)⟩_BZ.

    Returns (a_phys, d_phys, raw_p_z_values, raw_fsum_values) where:
      a_phys × ω² + d_phys ≈ ω² × π_2_xx(ω) at large ω
    i.e., d_phys is the leading 1/ω² Drude weight in physical convention.
    """
    fsum_values = []
    for p_z in p_z_values:
        val = BZ_average(lambda k: f_sum_rule_integrand_at_p(k, p_z), N=N)
        fsum_values.append(val)
    p_arr = np.array(p_z_values)
    p2_arr = p_arr ** 2
    coeffs = np.polyfit(p2_arr, fsum_values, 2)
    a_4, a_2, a_0 = coeffs
    # In Kubo convention: Π^{xx}(p, ω → ∞) ≈ -(2/ω²) × f-sum(p_z)
    # ⟹ ω² × π_2_xx(ω → ∞) = -2 × (a_0 + a_2 p² + ...) at p²-coefficient: -2 × a_2
    # ⟹ d_raw = -2 × a_2 (raw fit convention)
    # ⟹ d_phys = -d_raw = +2 × a_2 (physical sign-flip)
    d_raw_predicted = -2 * a_2
    d_phys_predicted = -d_raw_predicted
    return {
        "a_0": a_0, "a_2": a_2, "a_4": a_4,
        "d_raw_predicted": d_raw_predicted,
        "d_phys_predicted": d_phys_predicted,
        "p_z_values": list(p_z_values),
        "fsum_values": fsum_values,
    }


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("Π_v Drude weight from substrate Bloch invariants — conceptual probe")
    print()
    print("  Goal: find the analog of ⟨Tr H²⟩ × k* = 12 × 3 = 36 (Π_TT's denominator)")
    print("        for the velocity vertex Π_v Drude weight.")
    print("  Target (pairwise extraction at N=16): d ≈ -1/137.4, |1/d| ≈ 137.4")
    print()

    N = 16  # match the pairwise extraction's grid
    d_target_inv = 137.4  # |1/d_pairwise|

    # ============================================================
    # Bloch invariants
    # ============================================================
    header("Step 1: Bloch invariants over BZ at N = " + str(N))
    print()
    invariants = {}
    for name, fn in [
        ("⟨Tr H²⟩", Tr_H2_at_k),
        ("⟨Tr v²⟩", Tr_v2_at_k),
        ("⟨Tr v⁴⟩", Tr_v4_at_k),
        ("⟨Tr v² H⟩", Tr_v2_H_at_k),
        ("⟨Tr v H v⟩", Tr_vHv_at_k),
        ("⟨Tr v² H²⟩", Tr_v2_H2_at_k),
        ("⟨Tr v² H v²⟩", Tr_v2H_v2_at_k),
    ]:
        t0 = time.time()
        val = BZ_average(fn, N=N)
        dt = time.time() - t0
        invariants[name] = val
        print(f"  {name:25s} = {val:+.6e}  ({dt:.1f}s)")

    # Π_TT match check
    print()
    Tr_H2 = invariants["⟨Tr H²⟩"]
    print(f"  Sanity: ⟨Tr H²⟩ = {Tr_H2:.6f}  (expected = 12 = k* × N_atoms)")
    assert abs(Tr_H2 - 12.0) < 1e-9, f"⟨Tr H²⟩ = {Tr_H2} ≠ 12"
    print(f"  [PASS] ⟨Tr H²⟩ = 12 confirms Bloch invariant calculation.")

    # ============================================================
    # f-sum rule / Drude weight at half-filling
    # ============================================================
    header("Step 2: f-sum rule = ⟨Σ_{nm fill/unfill} (E_n - E_m) |⟨v^x⟩|²⟩")
    print()
    print("  This is the leading 1/ω² coefficient of Π^{xx}_v(p=0, ω → ∞),")
    print("  i.e. the Drude weight at zero external momentum.")
    print()
    f_sum = BZ_average(f_sum_rule_integrand, N=N)
    print(f"  f-sum (over filled→unfilled, μ=0): {f_sum:.6f}")
    print(f"  (To compare to π_2_xx's d, need also the p² expansion of matrix elements.)")

    # ============================================================
    # Structural product candidates vs 137.4
    # ============================================================
    header("Step 3: structural products vs 1/|d| ≈ 137.4")
    print()
    Tr_v2 = invariants["⟨Tr v²⟩"]
    Tr_v4 = invariants["⟨Tr v⁴⟩"]
    Tr_v2H = invariants["⟨Tr v² H⟩"]
    Tr_vHv = invariants["⟨Tr v H v⟩"]
    Tr_v2H2 = invariants["⟨Tr v² H²⟩"]
    k_star, N_atoms, g_girth = 3, 4, 10

    candidates = [
        # Direct analogs of ⟨Tr H²⟩ × k* = 36
        ("⟨Tr v²⟩ × k*",            Tr_v2 * k_star),
        ("⟨Tr v²⟩ × N_atoms",       Tr_v2 * N_atoms),
        ("⟨Tr v²⟩ × g",             Tr_v2 * g_girth),
        ("⟨Tr v² H⟩ × k*",          Tr_v2H * k_star),
        ("⟨Tr v H v⟩ × k*",         Tr_vHv * k_star),
        ("⟨Tr v² H²⟩ × k*",         Tr_v2H2 * k_star),
        # Pure invariants
        ("⟨Tr v²⟩",                  Tr_v2),
        ("⟨Tr v² H⟩",                Tr_v2H),
        ("⟨Tr v H v⟩",               Tr_vHv),
        ("⟨Tr v² H²⟩",               Tr_v2H2),
        # Powers / ratios
        ("⟨Tr v²⟩²",                 Tr_v2 ** 2),
        ("⟨Tr v⁴⟩",                  Tr_v4),
        # With BZ-normalisation factors
        ("⟨Tr v²⟩ × π²",             Tr_v2 * np.pi ** 2),
        ("⟨Tr H²⟩ × k* × ⟨Tr v²⟩ / 12", Tr_H2 * k_star * Tr_v2 / 12),
    ]

    print(f"  {'candidate':>40s}  {'value':>14s}  {'ratio to 137.4':>15s}")
    print("  " + "-" * 75)
    best_match = None
    best_dev = float('inf')
    for name, value in candidates:
        if abs(value) < 1e-12:
            continue
        ratio = value / d_target_inv
        # Look for clean ratios: 1, 2, 1/2, π, 1/π, etc.
        dev = abs(value - d_target_inv) / d_target_inv * 100
        flag = ""
        if 0.95 < ratio < 1.05:
            flag = " ★"
            if abs(value - d_target_inv) < abs(best_dev):
                best_dev = abs(value - d_target_inv)
                best_match = name
        elif 0.45 < ratio < 0.55 or 1.95 < ratio < 2.05:
            flag = " (half/double)"
        print(f"  {name:>40s}  {value:>+14.6f}  {ratio:>+14.4f}{flag}")

    if best_match:
        print()
        print(f"  Best match: {best_match}")
        print(f"  Deviation from 137.4: {best_dev:.4f}")
    else:
        print()
        print(f"  No candidate matches 137.4 ± 5% directly.")

    # ============================================================
    # Conceptual readout
    # ============================================================
    header("Step 4: conceptual readout — what next?")
    print()
    print("  We've computed the natural Bloch invariants of (v, H) on srs.")
    print(f"  Key values:")
    print(f"    ⟨Tr H²⟩    = {Tr_H2:.4f}   (= k* × N_atoms = 12 ✓)")
    print(f"    ⟨Tr v²⟩    = {Tr_v2:.4f}")
    print(f"    ⟨Tr v² H⟩  = {Tr_v2H:.4f}")
    print(f"    ⟨Tr v H v⟩ = {Tr_vHv:.4f}")
    print(f"    ⟨Tr v² H²⟩ = {Tr_v2H2:.4f}")
    print(f"    ⟨Tr v⁴⟩    = {Tr_v4:.4f}")
    print()
    print(f"  Π_TT analog: D_TT = -1/(⟨Tr H²⟩ × k*) = -1/(12×3) = -1/36")
    print(f"  Π_v target:  d   ≈ -1/137.4 (pairwise extraction at N=16)")
    print()
    print(f"  If the analog Bloch denominator emerges as Tr[v²] × (combinatorial) = 137.4,")
    print(f"  we'd have the analytical handle. From the candidates above, the closest")
    print(f"  matches identify which combinations are structurally relevant.")
    print()
    print(f"  Next conceptual step: derive d analytically from the Kubo formula:")
    print(f"    d = lim_{{ω → ∞}} ω² × π_2_xx(ω)")
    print(f"      = -2/V_BZ × Σ_{{nm}} ∫_BZ (f_n - f_m)(E_n - E_m) × ∂²|⟨m,k+p|v^x|n,k⟩|²/∂p_z²|_0 d³k")
    print(f"    This is a specific BZ integral involving second derivative of matrix elements.")

    # ============================================================
    # Step 5: directly compute d via finite-difference p² expansion of f-sum
    # ============================================================
    header("Step 5: extract d analytically via f-sum(p_z) finite differences")
    print()
    print("  At large ω, Π^{xx}_v(p, ω) → -(2/ω²) × ⟨f-sum(p_z)⟩_BZ.")
    print("  The leading p² coef of f-sum(p_z) gives d_phys with no ω fit needed.")
    print(f"  p_z values: {(0.0, 0.05, 0.10, 0.15, 0.20)}")
    print()
    t0 = time.time()
    result = extract_d_from_fsum(N=N)
    dt = time.time() - t0
    print(f"  ⟨f-sum(p_z)⟩_BZ at each p_z (computed in {dt:.1f}s):")
    print(f"  {'p_z':>6s}  {'f-sum':>14s}")
    for p_z, val in zip(result["p_z_values"], result["fsum_values"]):
        print(f"  {p_z:>6.3f}  {val:>+14.6e}")
    print()
    print(f"  Polynomial fit:   f-sum(p) = a_0 + a_2 p² + a_4 p⁴")
    print(f"    a_0   = {result['a_0']:+.6e}  (= f-sum at p=0)")
    print(f"    a_2   = {result['a_2']:+.6e}  (= leading p² coefficient)")
    print(f"    a_4   = {result['a_4']:+.6e}")
    print()
    print(f"  Predicted Drude weight (sign-flip × 2 convention):")
    print(f"    d_raw_predicted  = -2 × a_2 = {result['d_raw_predicted']:+.6e}")
    print(f"    d_phys_predicted = +2 × a_2 = {result['d_phys_predicted']:+.6e}")
    print()
    # Compare to pairwise-extraction value
    d_phys_pairwise = -0.007279  # from point-eval probe N=16
    d_phys_phaseA = -0.005942    # from Phase A linear fit N=14
    print(f"  Cross-check vs full-Kubo extraction:")
    print(f"    Pairwise extraction (N=16, point-eval probe):  d_phys = {d_phys_pairwise:+.6e}")
    print(f"    Phase A linear fit  (N=14):                     d_phys = {d_phys_phaseA:+.6e}")
    print(f"    f-sum direct        (this probe, N={N}):           "
          f"d_phys = {result['d_phys_predicted']:+.6e}")
    dev_pair = abs(result['d_phys_predicted'] - d_phys_pairwise) / abs(d_phys_pairwise) * 100
    dev_pha = abs(result['d_phys_predicted'] - d_phys_phaseA) / abs(d_phys_phaseA) * 100
    print(f"    Deviation from pairwise: {dev_pair:+.3f}%")
    print(f"    Deviation from Phase A:  {dev_pha:+.3f}%")
    print()
    print(f"  If the f-sum extraction agrees with one of the Kubo extractions,")
    print(f"  THAT identifies which extraction method captures d_phys cleanly,")
    print(f"  AND the f-sum is the analytical handle on d_phys.")
    print()
    # Look for structural form of the f-sum's p² coefficient
    inv_d = -1.0 / result['d_phys_predicted'] if abs(result['d_phys_predicted']) > 1e-12 else float('inf')
    print(f"  |1/d_phys_predicted| = {abs(inv_d):.4f}")
    print(f"  Candidate structural forms for {abs(inv_d):.2f}:")
    candidates_2 = [
        ("⟨Tr v² H²⟩",                Tr_v2H2),
        ("⟨Tr v² H²⟩ × k* / 3",       Tr_v2H2 * k_star / 3),
        ("⟨Tr v² H²⟩ + ⟨Tr v²⟩",      Tr_v2H2 + Tr_v2),
        ("⟨Tr v² H²⟩ - ⟨Tr v²⟩",      Tr_v2H2 - Tr_v2),
        ("(g-k*) × ⟨Tr v² H²⟩",       (g_girth - k_star) * Tr_v2H2),
        ("g × ⟨Tr v²⟩",               g_girth * Tr_v2),
        ("k* × ⟨Tr v²⟩",              k_star * Tr_v2),
        ("(k*+1)² × ⟨Tr v²⟩",         (k_star + 1) ** 2 * Tr_v2),
        ("⟨Tr v² H²⟩ × (g-3)/(g-k*)", Tr_v2H2 * 7 / 7),  # both 7
        ("(N_atoms² - 1) × ⟨Tr v² H⟩²/12", 0.0),         # placeholder
    ]
    for name, value in candidates_2:
        if abs(value) < 1e-12:
            continue
        ratio = value / abs(inv_d)
        flag = " ★" if 0.99 < ratio < 1.01 else ""
        print(f"    {name:>40s} = {value:>+12.4f}   ratio = {ratio:+.5f}{flag}")


if __name__ == "__main__":
    main()
