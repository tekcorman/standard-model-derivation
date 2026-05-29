#!/usr/bin/env python3
"""
Probe: substrate-lattice-axis waterfilling for V_us, leveraging existing
methodology from `predictions/V_us.py`.

CONTEXT
-------
V_us = k*² / (g · N_atoms) per `predictions/V_us.py` (Moore-bound counting,
Level 2 coupling density). For srs: V_us = 9/(10·4) = 9/40 = 0.22500
vs PDG 2024 0.22501 ± 0.00068 (−0.015σ).

This probe extends to the lattice-axis: under proper A2-T waterline
retention and channel-specific filtering (Phase 1 doc §2), each contributing
candidate produces its own V_us(C) = k_C²/(g_C · N_atoms_C), Boltzmann-weighted.

Methodology: SAME Moore-bound counting formula; just sum over candidates
that contribute to the V_us channel (C1+C2 spectral+combinatorial). No new
mathematics — extends established framework methodology to lattice axis.

CHANNEL FILTERING for V_us
--------------------------
V_us is a CKM matrix element → chiral observable (C3 chirality channel
required) AND combinatorial structural count (C2). Per Phase 1 doc §2b:

- chiral 3D 3-regular Bloch-decomposable: contributes (srs + R-9 chirals)
- centrosymmetric lattices (R-7 ths, R-8 dia, R-9 eta, R-9 utj, honeycomb 2D):
  CHIRALITY-CHANNEL HARD-GATED via R-12 (worked example R-7 refutation:
  167σ overshoot per `proofs/foundations/r7_ths_residue_check.py`)
- d>3 (R-4, R-5): GAUGE-CHANNEL HARD-GATED via Cl(8)/Cl(10) Pati-Salam fail;
  V_us is in the visible Pati-Salam SU(4) sector → R-4/R-5 contribute 0
- finite (Petersen, K_{3,3}): no infinite ensemble → 0
- R-13 hyperbolic Kleinian: Bloch absent + chirality unspecified → 0

Net: only CHIRAL 3D 3-regular Bloch-decomposable substrates contribute.
"""

import math


def v_us(k_star, g, n_atoms):
    """
    V_us = k*² / (g · N_atoms) per `predictions/V_us.py` (Moore-bound counting).
    """
    return (k_star ** 2) / (g * n_atoms)


def boltzmann_weight(dl_struct):
    """Substrate Boltzmann weight w(C) = 2^(-DL_struct(C))."""
    return 2.0 ** (-dl_struct)


# Catalogued candidates with (DL, k, g, N_atoms_primitive, contributes_to_V_us, chirality, notes)
# DL from `proofs/foundations/dl_comparison.py`; (g, N_atoms) from RCSR + International Tables.
CANDIDATES = [
    # (name, DL, k_C, g_C, N_atoms, chiral, contributes, notes)
    ('srs (Laves, I4_132)',         12.17, 3, 10, 4, True,  True,  'Unique chiral 3D 3-regular Moore-saturating; (3,10)-cage'),
    ('R-7 ths (I4_1/amd)',          13.85, 3,  4, 4, False, False, 'Centrosymmetric → R-12 chirality hard-gate; R-7 REFUTED at 167σ'),
    ('R-8 dia (Fd-3m)',             14.06, 3,  6, 2, False, False, 'Centrosymmetric → R-12 hard-gate'),
    ('R-9 eta (P6_3/mmc)',          14.41, 3,  6,12, False, False, 'Centrosymmetric (P6_3/mmc has inversion center)'),
    ('R-9 utj (P2_1/c)',            15.85, 3,  4, 8, False, False, 'P2_1/c centrosymmetric'),
    ('R-4 d=4 crystallographic',    14.00, 4, None, None, None, False, 'd=4 → Cl(8) gauge fails Pati-Salam; V_us channel hard-gated'),
    ('R-5 d=5 crystallographic',    19.00, 5, None, None, None, False, 'd=5 → Cl(10); gauge hard-gated'),
    ('Petersen (finite)',            5.32, 3,  5,10, None, False, 'Finite → no infinite ensemble; channel-zero'),
    ('K_{3,3} (finite)',             8.59, 3,  4, 6, None, False, 'Finite'),
    ('Honeycomb 2D (p6mm)',          9.67, 3,  6, 2, False, False, 'p6mm centrosymmetric + d=2 Gleason-gated'),
    ('R-13 hyperbolic Kleinian',    41.00, 3, None, None, None, False, 'No Bloch decomposition; weight ≤ 2^-41 anyway'),
]


def main():
    print("=" * 80)
    print("SUBSTRATE-LATTICE-AXIS WATERFILLING for V_us (channels C1 spectral + C2 combinatorial + C3 chirality)")
    print("Methodology: same Moore-bound formula as V_us.py, Boltzmann-weighted")
    print("over A2-T-compressing chiral 3D 3-regular alternatives.")
    print("=" * 80)

    print(f"\n{'Candidate':<35s} {'DL':>6s} {'k':>3s} {'g':>4s} {'N':>4s} {'chir':>5s} {'w(C)':>11s} {'V_us(C)':>9s}  Notes")
    print("-" * 80)

    total_weight = 0.0
    weighted_sum = 0.0
    for name, dl, k, g, n, chiral, contributes, notes in CANDIDATES:
        w = boltzmann_weight(dl)
        if not contributes:
            chir_str = "✗" if chiral is False else ("?" if chiral is None else "✓")
            print(f"  {name:<33s} {dl:>6.2f} {k:>3d} {str(g):>4s} {str(n):>4s} {chir_str:>5s} {'GATED':>11s} {'-':>9s}  {notes}")
            continue
        v = v_us(k, g, n)
        total_weight += w
        weighted_sum += w * v
        chir_str = "✓"
        print(f"  {name:<33s} {dl:>6.2f} {k:>3d} {g:>4d} {n:>4d} {chir_str:>5s} {w:>11.3e} {v:>9.4f}  {notes}")

    if total_weight == 0:
        print("\n*** ZERO CONTRIBUTORS to V_us channel — should be impossible (srs always contributes) ***")
        return

    v_us_total = weighted_sum / total_weight
    v_us_srs_only = v_us(3, 10, 4)
    v_us_obs = 0.22501
    sigma_obs = 0.00068

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n  Total V_us-channel weight Z       = {total_weight:.3e}")
    print(f"\n  V_us (srs only, existing)         = {v_us_srs_only:.6f}  = 9/40")
    print(f"  V_us (lattice-axis waterfilled)   = {v_us_total:.6f}")
    print(f"  V_us (observed PDG 2024)          = {v_us_obs:.5f} ± {sigma_obs:.5f}")
    print(f"\n  srs-only - obs                    = {v_us_srs_only - v_us_obs:+.6f} = {(v_us_srs_only - v_us_obs)/sigma_obs:+.3f}σ")
    print(f"  waterfilled - obs                 = {v_us_total - v_us_obs:+.6f} = {(v_us_total - v_us_obs)/sigma_obs:+.3f}σ")
    print(f"  lattice-axis waterfilling shift   = {v_us_total - v_us_srs_only:+.6f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION (per parameter_linter.md no-post-hoc-backfill)")
    print("=" * 80)

    n_contributors = sum(1 for c in CANDIDATES if c[6])
    print(f"\n  Number of contributing candidates: {n_contributors}")
    if n_contributors == 1:
        print(f"  → srs is the UNIQUE catalogued contributor under strict channel filtering.")
        print(f"  → V_us(waterfilled) = V_us(srs) exactly. No lattice-axis shift.")
        print(f"  → The +0.875σ residual (using older PDG; current is −0.015σ) is NOT")
        print(f"    explained by lattice-axis waterfilling; due to RG running, higher-")
        print(f"    order Feshbach corrections, or other mechanisms.")
        print(f"\n  CONDITIONAL: This conclusion presumes srs is the unique chiral 3D")
        print(f"  3-regular Bloch-decomposable substrate (R-9 OPEN per residue register).")
        print(f"  RCSR enumeration of any chiral non-srs 3D 3-regular nets would widen")
        print(f"  the contributor set and shift V_us by their Boltzmann-weighted contribution.")

    # Bound: hypothetical R-9 contribution
    print("\n" + "=" * 80)
    print("BOUND ON R-9 (chiral non-srs 3D 3-regular nets, OPEN enumeration)")
    print("=" * 80)
    print(f"\n  Hypothetical: if M chiral non-srs entries exist at average ΔDL = +2 bits")
    print(f"  over srs (DL ≈ 14.17), they would carry total weight w_R9 ≈ M·2^(-14.17)")
    print(f"  Each with V_us in range [9/40 to ~9/24] depending on (g, N_atoms).")

    for delta_dl in [1, 2, 3, 5]:
        for M in [1, 2, 5]:
            w_r9_total = M * 2.0**(-(12.17 + delta_dl))
            f_r9 = w_r9_total / (boltzmann_weight(12.17) + w_r9_total)
            print(f"  ΔDL=+{delta_dl}, M={M}: w_R9_total = {w_r9_total:.2e}, f_R9 = {f_r9:.3f}, "
                  f"max V_us shift ≈ {f_r9 * 0.15:.4f} (vs PDG σ={sigma_obs:.5f})")

    print(f"\n  Reading: even M=1 R-9 entry at ΔDL=+1 bit would shift V_us by {0.333*0.15:.3f},")
    print(f"  much larger than current 0.00068 PDG sensitivity. R-9 enumeration is therefore")
    print(f"  the load-bearing OPEN piece for V_us robustness.")


if __name__ == '__main__':
    main()
