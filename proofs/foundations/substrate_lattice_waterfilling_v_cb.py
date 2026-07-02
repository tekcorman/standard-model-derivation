#!/usr/bin/env python3
"""
Probe: substrate-lattice-axis waterfilling for V_cb, leveraging existing
methodology from `predictions/V_cb.py`.

CONTEXT
-------
V_cb = α₁_bare / (1 − α₁_bare) where α₁_bare = ((k-1)/k)^(g-2) per
`predictions/V_cb.py` (A5(b) Case B Hashimoto walk-rep with full A2
waterline geometric series). For srs: V_cb = (2/3)^8 / (1−(2/3)^8) =
256/6305 ≈ 0.04060 vs PDG 2024 0.0405 ± 0.0015 (+0.07σ).

This probe extends to the lattice-axis: per-candidate V_cb(C) =
α₁(C)/(1−α₁(C)) where α₁(C) = ((k_C-1)/k_C)^(g_C-2). Boltzmann-weighted
across A2-T-compressing chiral 3D 3-regular Bloch-decomposable candidates
(channel C1 spectral + C3 chirality required).

Methodology: SAME geometric-series formula as V_cb.py; just sum over
contributing candidates. No new mathematics.

CHANNEL FILTERING for V_cb (parallels V_us probe; same chirality + gauge gating)
- chiral 3D 3-regular Bloch-decomposable: contributes (srs + R-9 chirals)
- centrosymmetric (R-7 ths, R-8 dia, eta, utj, honeycomb 2D): R-12 chirality
  hard-gated (R-7 REFUTED at 167σ in `proofs/foundations/r7_ths_residue_check.py`)
- d>3 (R-4, R-5): GAUGE hard-gated (Cl(8)/Cl(10) Pati-Salam fail) → V_cb is
  visible-sector → contribute 0
- finite (Petersen, K_{3,3}): no infinite ensemble → 0
- R-13 hyperbolic: Bloch absent + weight ≤ 2^-41 negligible → 0
"""

import math


def alpha_1_bare(k_star, g):
    """α₁_bare = ((k-1)/k)^(g-2) per Hashimoto walk-rep (A5(b) Case B first winding)."""
    return ((k_star - 1) / k_star) ** (g - 2)


def v_cb(k_star, g):
    """
    V_cb = α₁_bare / (1 − α₁_bare) per `predictions/V_cb.py` (A2 waterline geometric series).
    """
    a1 = alpha_1_bare(k_star, g)
    return a1 / (1 - a1)


def boltzmann_weight(dl_struct):
    """Substrate Boltzmann weight w(C) = 2^(-DL_struct(C))."""
    return 2.0 ** (-dl_struct)


# Catalogued candidates with (DL, k, g, chirality, contributes, notes)
CANDIDATES = [
    ('srs (Laves, I4_132)',         12.17, 3, 10, True,  True,  'Unique chiral 3D 3-regular Bloch-decomposable'),
    ('R-7 ths (I4_1/amd)',          13.85, 3,  4, False, False, 'Centrosymmetric → R-12 hard-gate'),
    ('R-8 dia (Fd-3m)',             14.06, 3,  6, False, False, 'Centrosymmetric → R-12 hard-gate'),
    ('R-9 eta (P6_3/mmc)',          14.41, 3,  6, False, False, 'P6_3/mmc has inversion'),
    ('R-9 utj (P2_1/c)',            15.85, 3,  4, False, False, 'P2_1/c centrosymmetric'),
    ('R-4 d=4 crystallographic',    14.00, 4, None, None, False, 'Gauge channel hard-gated (Cl(8) Pati-Salam fail)'),
    ('R-5 d=5 crystallographic',    19.00, 5, None, None, False, 'Gauge channel hard-gated (Cl(10))'),
    ('Petersen (finite)',            5.32, 3,  5, None, False, 'No infinite ensemble'),
    ('K_{3,3} (finite)',             8.59, 3,  4, None, False, 'No infinite ensemble'),
    ('Honeycomb 2D (p6mm)',          9.67, 3,  6, False, False, 'Centrosymmetric + d=2 Gleason-gated'),
    ('R-13 hyperbolic Kleinian',    41.00, 3, None, None, False, 'No Bloch + weight ≤ 2^-41'),
]


def main():
    print("=" * 80)
    print("SUBSTRATE-LATTICE-AXIS WATERFILLING for V_cb (channels C1 spectral + C3 chirality)")
    print("Methodology: same geometric-series formula as V_cb.py, Boltzmann-weighted")
    print("over A2-T-compressing chiral 3D 3-regular alternatives.")
    print("=" * 80)

    print(f"\n{'Candidate':<35s} {'DL':>6s} {'k':>3s} {'g':>4s} {'chir':>5s} {'w(C)':>11s} {'V_cb(C)':>10s}")
    print("-" * 80)

    total_weight = 0.0
    weighted_sum = 0.0
    for name, dl, k, g, chiral, contributes, notes in CANDIDATES:
        w = boltzmann_weight(dl)
        if not contributes:
            chir_str = "✗" if chiral is False else ("?" if chiral is None else "✓")
            print(f"  {name:<33s} {dl:>6.2f} {k:>3d} {str(g):>4s} {chir_str:>5s} {'GATED':>11s} {'-':>10s}")
            continue
        v = v_cb(k, g)
        total_weight += w
        weighted_sum += w * v
        chir_str = "✓"
        print(f"  {name:<33s} {dl:>6.2f} {k:>3d} {g:>4d} {chir_str:>5s} {w:>11.3e} {v:>10.5f}")

    if total_weight == 0:
        print("\n*** ZERO CONTRIBUTORS — should be impossible (srs always contributes) ***")
        return

    v_cb_total = weighted_sum / total_weight
    v_cb_srs_only = v_cb(3, 10)
    v_cb_obs = 0.0405
    sigma_obs = 0.0015

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n  Total V_cb-channel weight Z       = {total_weight:.3e}")
    print(f"\n  V_cb (srs only, existing)         = {v_cb_srs_only:.6f}  = 256/6305")
    print(f"  V_cb (lattice-axis waterfilled)   = {v_cb_total:.6f}")
    print(f"  V_cb (observed PDG 2024 exclusive) = {v_cb_obs:.4f} ± {sigma_obs:.4f}")
    print(f"\n  srs-only - obs                    = {v_cb_srs_only - v_cb_obs:+.6f} = {(v_cb_srs_only - v_cb_obs)/sigma_obs:+.3f}σ")
    print(f"  waterfilled - obs                 = {v_cb_total - v_cb_obs:+.6f} = {(v_cb_total - v_cb_obs)/sigma_obs:+.3f}σ")
    print(f"  lattice-axis waterfilling shift   = {v_cb_total - v_cb_srs_only:+.6f}")

    n_contrib = sum(1 for c in CANDIDATES if c[5])
    print(f"\n  Number of contributing candidates: {n_contrib}")
    if n_contrib == 1:
        print(f"  → srs UNIQUE catalogued contributor under strict channel filtering.")
        print(f"  → V_cb(waterfilled) = V_cb(srs) exactly. No lattice-axis shift.")

    # Bound on hypothetical R-9 entries
    print("\n" + "=" * 80)
    print("BOUND ON R-9 (chiral non-srs 3D 3-regular nets, OPEN enumeration)")
    print("=" * 80)
    print(f"\n  Hypothetical R-9 entries shift V_cb because their α₁ = ((k-1)/k)^(g-2)")
    print(f"  depends on g. e.g. ths-style g=4 gives V_cb = 4/5 = 0.80; dia-style g=6")
    print(f"  gives V_cb = 16/65 ≈ 0.246. Both differ from srs's 256/6305 ≈ 0.04 by O(0.2-0.8).\n")

    for delta_dl, M, V_alt_label, V_alt in [
        (1, 1, 'g=6 (dia-style)',  v_cb(3, 6)),
        (2, 1, 'g=6',              v_cb(3, 6)),
        (3, 1, 'g=6',              v_cb(3, 6)),
        (5, 1, 'g=6',              v_cb(3, 6)),
        (1, 1, 'g=8',              v_cb(3, 8)),
        (3, 1, 'g=8',              v_cb(3, 8)),
        (5, 1, 'g=8',              v_cb(3, 8)),
    ]:
        w_r9 = boltzmann_weight(12.17 + delta_dl)
        f_r9 = M * w_r9 / (boltzmann_weight(12.17) + M * w_r9)
        shift = f_r9 * (V_alt - v_cb_srs_only)
        sigma_shift = abs(shift) / sigma_obs
        print(f"  ΔDL=+{delta_dl}, M={M}, alt={V_alt_label} V_cb(alt)={V_alt:.4f}: "
              f"f_R9={f_r9:.3f}, shift={shift:+.5f} ≈ {sigma_shift:.1f}σ overshoot")

    print(f"\n  Reading: SAME pattern as V_us — V_cb robustness depends on srs being THE unique")
    print(f"  chiral 3D 3-regular Bloch-decomposable. Even ΔDL=+5 bits, M=1 entry would")
    print(f"  shift V_cb by 1-15σ depending on g_alt. R-9 enumeration is load-bearing.")


if __name__ == '__main__':
    main()
