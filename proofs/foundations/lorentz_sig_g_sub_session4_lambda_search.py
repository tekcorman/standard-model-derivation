#!/usr/bin/env python3
"""
G_sub session 4 Option A1: investigate natural Λ_cone + multi-valley sum.

Per an internal working note. After session 3
established the substrate is metallic at all natural μ (so standard
Sakharov fails for full Bloch matter), Option A1 stays within the
cone-effective Sakharov framework and asks:

  1. Is there a natural Λ_cone derivable from substrate physics?
  2. What does multi-valley sum give for G_sub^total?

This script:

  (S1) Compares the cone-effective spin-1 dispersion at Γ to the actual
       full Bloch eigenvalues near Γ along several BZ directions.
       Identifies the |q| at which the cone-effective deviates by 5%
       or more from the full Bloch.

  (S2) Same analysis at the H point (particle-hole conjugate of Γ).

  (S3) Same at the P point (2-band cone, v_F = √3/6).

  (S4) Multi-valley sum: with v_F values per cone and the cone-effective
       matter-loop coefficient, compute G_sub^total at the natural Λ.

Convention: rescaled-time substrate units (c = 1), per
`lorentz_sig_iorio_session4_einstein.py`.
"""
from __future__ import annotations

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_lichnerowicz_closure import (
    bloch_H_numeric, DIRECTED_BONDS,
)


# BCC primitive vectors (lattice constant = 1)
A_PRIM = np.array([
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
])
B_RECIP = 2 * np.pi * np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
])


def k_cart_to_frac(k_cart):
    """Cartesian k → fractional k via b·k_frac = k_cart."""
    return np.linalg.solve(B_RECIP.T, k_cart)


def H_at_k_cart(k_cart):
    k_frac = k_cart_to_frac(k_cart)
    return bloch_H_numeric(*k_frac)


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


# =============================================================================
# S1: cone-effective vs full Bloch near Γ
# =============================================================================

def step_S1_gamma_cone():
    header("S1: cone-effective vs full Bloch near Γ")
    print()
    print("  Cone-effective at Γ: E_h(q) = -1 + v_F h |q|, h ∈ {+1, 0, -1}, v_F = 1/2.")
    print("  T-irrep eigenvalues at Γ: {-1, -1, -1}; expanding around Γ gives 3 modes.")
    print()
    print("  Probe along several directions; compute deviation:")

    v_F = 0.5
    directions = [
        ("(1, 0, 0)", np.array([1.0, 0.0, 0.0])),
        ("(1, 1, 0)", np.array([1.0, 1.0, 0.0])),
        ("(1, 1, 1)", np.array([1.0, 1.0, 1.0])),
    ]
    q_magnitudes = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]

    print(f"\n  {'direction':>12s} {'|q|':>6s} {'cone E_+1':>12s} {'full E_max':>12s} {'cone E_0':>12s} {'full mid':>12s} {'cone E_-1':>12s} {'full E_min(T-irrep)':>20s}")
    print(f"  (T-irrep bands of full Bloch are 3 lowest; Perron is 4th)")
    print()
    for label, dhat in directions:
        dhat_norm = dhat / np.linalg.norm(dhat)
        for q in q_magnitudes:
            k_cart = q * dhat_norm
            H = H_at_k_cart(k_cart)
            eigs = sorted(np.linalg.eigvalsh(H).real)
            # T-irrep: lowest 3 bands (continuous from Γ at -1, -1, -1)
            # Perron: highest band (continuous from Γ at +3)
            T_eigs = eigs[:3]
            E_cone = [-1 + v_F * h * q for h in [-1, 0, 1]]  # ascending
            print(f"  {label:>12s} {q:>6.2f} {E_cone[2]:>+12.4f} {T_eigs[2]:>+12.4f} {E_cone[1]:>+12.4f} {T_eigs[1]:>+12.4f} {E_cone[0]:>+12.4f} {T_eigs[0]:>+15.4f}")

    print()
    print("  Cone-effective expectation: E_h = -1 + v_F h q; for q in any direction, the")
    print("  three eigenvalues are -1 - v_F q, -1, -1 + v_F q. Real bands deviate from this.")


def find_lambda_cone(direction='100', threshold=0.05, v_F=0.5):
    """
    Find Λ_cone where the cone-effective deviates from the full Bloch by `threshold`
    (fractional) in the dispersing modes.
    """
    if direction == '100':
        dhat = np.array([1.0, 0.0, 0.0])
    elif direction == '110':
        dhat = np.array([1.0, 1.0, 0.0])
        dhat /= np.linalg.norm(dhat)
    elif direction == '111':
        dhat = np.array([1.0, 1.0, 1.0])
        dhat /= np.linalg.norm(dhat)

    q_grid = np.linspace(0.02, 5.0, 200)
    devs = []
    for q in q_grid:
        k_cart = q * dhat
        H = H_at_k_cart(k_cart)
        eigs = sorted(np.linalg.eigvalsh(H).real)
        T_eigs = eigs[:3]  # T-irrep bands
        # Cone-effective predictions
        E_minus_full = T_eigs[0]
        E_plus_full = T_eigs[2]
        E_plus_cone = -1 + v_F * q
        E_minus_cone = -1 - v_F * q
        # Deviation
        dev_plus = abs(E_plus_full - E_plus_cone) / max(abs(E_plus_cone + 1), 0.01)
        dev_minus = abs(E_minus_full - E_minus_cone) / max(abs(E_minus_cone + 1), 0.01)
        devs.append(max(dev_plus, dev_minus))

    devs = np.array(devs)
    # Find first q where deviation exceeds threshold
    over = np.where(devs > threshold)[0]
    if len(over) > 0:
        return q_grid[over[0]], devs[over[0]]
    return None, None


def step_S2_lambda_cone():
    header("S2: identify natural Λ_cone (where cone-effective breaks down)")
    print()
    print("  For each direction, find smallest |q| where cone-effective deviates by 5% or more.")
    print()
    for d_label in ['100', '110', '111']:
        Lambda, dev = find_lambda_cone(direction=d_label, threshold=0.05)
        if Lambda is not None:
            print(f"  Direction {d_label}: Λ_cone (5% threshold) ≈ {Lambda:.3f} lattice units (deviation = {dev*100:.1f}%)")
        else:
            print(f"  Direction {d_label}: cone-effective never deviates by 5% (entire BZ-validity probed)")

    print()
    print("  For each direction, find smallest |q| where deviation = 1%, 5%, 10%, 20%:")
    for d_label in ['100', '110', '111']:
        print(f"  Direction {d_label}:")
        for thr in [0.01, 0.05, 0.1, 0.2, 0.5]:
            Lambda, dev = find_lambda_cone(direction=d_label, threshold=thr)
            if Lambda is not None:
                print(f"    {thr*100:.0f}% threshold: Λ ≈ {Lambda:.3f} (actual deviation {dev*100:.2f}%)")


# =============================================================================
# S3: P cone analysis
# =============================================================================

def step_S3_p_cone():
    header("S3: P cone analysis (2-band, v_F^P = √3/6)")
    print()
    P_frac = np.array([0.25, 0.25, 0.25])
    P_cart = P_frac @ B_RECIP
    print(f"  P point: fractional ({P_frac}), Cartesian = {P_cart}, |P_cart| = {np.linalg.norm(P_cart):.4f}")

    H_P = H_at_k_cart(P_cart)
    eigs_P = sorted(np.linalg.eigvalsh(H_P).real)
    print(f"  Spectrum at P: {eigs_P}")
    print(f"  Expected: ±√3 (each 2-fold) ≈ ±1.7321")
    print()
    print(f"  P cone has 2 + 2 = 4 bands. v_F^P = √3/6 ≈ {np.sqrt(3)/6:.4f}.")
    print(f"  Each cone (at +√3 and at -√3) has 2 bands disperse linearly off P.")


# =============================================================================
# S4: multi-valley sum estimate
# =============================================================================

def multi_valley_estimate(zeta=27/(512*np.pi**3), Lambda=np.pi):
    """
    Compute G_sub^total from multi-valley sum, assuming each cone contributes
    1/(16π G_cone) = ζ × Λ² / v_F.

    Cones in srs (per `predictions/srs_dirac_cone_velocities.py`):
    - Γ-cone (spin-1): v_F = 1/2, 1 cone
    - H-cone (spin-1, particle-hole conjugate of Γ): v_F = 1/2, 1 cone
    - P-cones (2-band Dirac): v_F = √3/6, 2 cones (at +√3 and -√3)

    Caveat: ζ = 27/(512π³) is sphere-Λ=π cone-effective specific. For different
    cone shapes (spin-1 vs 2-band), ζ may differ structurally — this estimate
    assumes a universal ζ, which is a working assumption for now.
    """
    cones = {
        'Γ (spin-1)': (0.5, 1),
        'H (spin-1)': (0.5, 1),
        'P (+√3, 2-band)': (np.sqrt(3)/6, 1),
        'P (-√3, 2-band)': (np.sqrt(3)/6, 1),
    }
    inv_16pi_G_total = 0
    print(f"\n  Multi-valley estimate (assuming universal ζ = {zeta:.4e}, Λ = {Lambda:.4f}):")
    print(f"  {'cone':>20s} {'v_F':>10s} {'multiplicity':>12s} {'1/(16π G_cone)':>18s}")
    for name, (vF, mult) in cones.items():
        contrib = mult * zeta * Lambda**2 / vF
        inv_16pi_G_total += contrib
        print(f"  {name:>20s} {vF:>.6f} {mult:>12d} {contrib:>.6e}")
    G_total = 1 / (16 * np.pi * inv_16pi_G_total)
    print(f"\n  Total: 1/(16π G_total) = {inv_16pi_G_total:.6e}")
    print(f"          G_sub^total = {G_total:.6e}")

    # Compare to single-cone Γ
    G_gamma = 1 / (16 * np.pi * zeta * Lambda**2 / 0.5)
    ratio = G_total / G_gamma
    print(f"\n  Single-cone Γ: G_sub^Γ = {G_gamma:.6e} (= 16/27 = {16/27:.6e} at Λ=π)")
    print(f"  Ratio G_total / G_Γ = {ratio:.4f}")


def step_S4_multi_valley():
    header("S4: multi-valley sum estimate (Option A1 candidate G_sub^total)")
    multi_valley_estimate(zeta=27/(512*np.pi**3), Lambda=np.pi)
    print()
    print("  STRUCTURAL FORM under universal-ζ assumption + ζ = 27/(512π³):")
    print()
    print("  1/(16π G_total) = ζ × Λ² × Σ(1/v_F_i)")
    print("                  = ζ × Λ² × [2/v_F_Γ + 2/v_F_P]")
    print("                  = ζ × Λ² × [4 + 4√3]")
    print("                  = 4(1+√3) ζ Λ²")
    print()
    print("  Solving for G_total:")
    print("  G_total = G_Γ × (1/v_F_Γ) / Σ(1/v_F_i)")
    print("          = G_Γ × 2/(4(1+√3))")
    print("          = G_Γ × 1/(2(1+√3))")
    print("          = G_Γ × (√3-1)/4   [rationalized]")
    print()
    print("  At Λ = π, v_F^Γ = 1/2 with G_Γ = 16/27:")
    print("  G_sub^total = (16/27) × (√3-1)/4 = 4(√3-1)/27")
    print()
    G_total_clean = 4 * (np.sqrt(3) - 1) / 27
    print(f"  Numerical: 4(√3-1)/27 = {G_total_clean:.10f}")
    print()
    print("  Structural decomposition: G_sub^total = (k*-1)² × (√k* - 1) / k*³")
    k_star = 3
    G_struct = (k_star - 1)**2 * (np.sqrt(k_star) - 1) / k_star**3
    print(f"  At k* = 3: (k*-1)² × (√k*-1)/k*³ = 4(√3-1)/27 = {G_struct:.10f}")
    print()
    print("  CAVEATS (multi-valley structural form is conditional on):")
    print("  1. Universal ζ assumption: spin-1 cones (Γ, H) and 2-band cones (P)")
    print("     all have ζ = 27/(512π³). Likely WRONG — they have different matter")
    print("     content; spin-1 ζ vs 2-band Weyl ζ differ structurally.")
    print("  2. Sharp spherical Λ = π convention: anisotropic cone validity (S2)")
    print("     means actual cutoff should be direction-dependent.")
    print("  3. The 27/(512π³) match was 0.06% from numerics, not exact.")
    print()
    print("  So: 4(√3-1)/27 ≈ 0.108 is a SUGGESTIVE structural form, not a closure.")
    print("  Closure would require: separate spin-1 vs 2-band ζ computation +")
    print("  resolution of cutoff-shape ambiguity.")


def main():
    header("G_sub session 4 Option A1: Λ_cone investigation + multi-valley")
    step_S1_gamma_cone()
    step_S2_lambda_cone()
    step_S3_p_cone()
    step_S4_multi_valley()

    header("STATUS")
    print("""
  S1+S2 findings: see Λ_cone deviation table above.
  S3: P cone confirmed at expected Cartesian location, spectrum ±√3.
  S4: multi-valley sum (universal-ζ assumption) gives G_sub^total estimate.

  Key questions for closure (deferred to session 5+):
    - Is the natural Λ for the substrate's emergent gravity ~ π or different?
    - Does ζ for the 2-band P cone differ from ζ for the 3-mode Γ cone?
    - The "universal ζ" assumption above is a placeholder — actual closure
      requires computing the matter loop separately at each cone with its
      own structural content.
""")


if __name__ == "__main__":
    main()
