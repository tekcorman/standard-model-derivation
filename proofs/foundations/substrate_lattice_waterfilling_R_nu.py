#!/usr/bin/env python3
"""
Probe: substrate-lattice-axis waterfilling for R_ν = Δm²₃₁/Δm²₂₁,
leveraging existing methodology from `predictions/R_nu_splitting.py`.

CONTEXT
-------
R_ν = 228/7 = 32.5714... per `predictions/R_nu_splitting.py`, derived from
the K_4 Green's function Chebyshev-U expansion at the Ihara phase
φ = arctan(√(4(k*−1)−1)) = arctan(√7), with distance n=5 selected by the
cubic uniqueness criterion q³ = 5q − 2 at q = k*−1 = 2.

PDG / NuFIT 6.0 (2024 NO):
    Δm²₂₁ = (7.49 ± 0.19) × 10⁻⁵ eV²
    Δm²₃₁ = (2.534 ± 0.024) × 10⁻³ eV²
    R_obs = 33.83 ± 0.92  →  framework deviation +1.4σ (soft tension; see
    an internal working note).

This probe extends to the lattice-axis: under proper A2-T waterline
retention and channel-specific filtering (an internal working note
§3.6), each contributing substrate alternative produces its own
R_ν(C) = 2/sin²(5 φ_C) − 4 with φ_C = arctan(√(4(k_C−1)−1)),
Boltzmann-weighted with w(C) = 2^(−DL_struct(C)).

Methodology: SAME K_4-Chebyshev formula as R_nu_splitting.py; just sum
over candidates that contribute under the M1b.ii §3.6 channel filtering.
No new mathematics — extends established framework methodology to the
lattice axis, parallel to substrate_lattice_waterfilling_v_{cb,us}.py.

CHANNEL FILTERING for R_ν (per M1b.ii §3.6 M1 line)
---------------------------------------------------
R_ν is a Pati-Salam-derived mass-ratio observable (inherits from m_ν2/m_ν3
which are PS-seesaw outputs). Channel = chirality (Pati-Salam SU(4) sector
requires chiral substrate per Row 17 + R-12 hard-gate):

- chiral 3D 3-regular Bloch-decomposable: contributes (srs + any R-9 chirals)
- centrosymmetric lattices (R-7 ths, R-8 dia, R-9 eta, R-9 utj, honeycomb 2D):
  R-12 chirality hard-gated (worked example R-7 refutation:
  167σ overshoot per `proofs/foundations/r7_ths_residue_check.py`)
- d>3 (R-4, R-5): GAUGE-CHANNEL HARD-GATED via Cl(8)/Cl(10) Pati-Salam fail;
  R_ν is a PS-sector observable → R-4/R-5 contribute 0
- finite (Petersen, K_{3,3}): no infinite ensemble → 0
- R-13 hyperbolic Kleinian: Bloch absent + weight ≤ 2⁻⁴¹ → 0

Net: only CHIRAL 3D 3-regular Bloch-decomposable substrates contribute.
Per Phase 1d RCSR enumeration, srs is the unique catalogued entry.
"""

import math
from fractions import Fraction


def r_nu(k_star):
    """
    R_ν on substrate with coordination k_star, mirroring
    `predictions/R_nu_splitting.predict_R_nu_splitting`.

    Formula: R_ν = 2/sin²(5 φ) − 4 with φ = arctan(√(4(k_star−1)−1)).

    Note: at k_star = 3 the cubic q³ = 5q − 2 selects n=5 uniquely
    (M1b.ii §3.6 M2 hard gate). For k_star ≠ 3 the cubic-uniqueness
    selector is NOT preserved; the value below is the formal extension
    only, kept for completeness of the lattice-axis sum.
    """
    ihara_sq = 4 * (k_star - 1) - 1
    if ihara_sq <= 0:
        return float('nan')
    phi = math.atan(math.sqrt(ihara_sq))
    sin2 = math.sin(5 * phi) ** 2
    if sin2 == 0:
        return float('nan')
    return 2 / sin2 - 4


def boltzmann_weight(dl_struct):
    """Substrate Boltzmann weight w(C) = 2^(−DL_struct(C))."""
    return 2.0 ** (-dl_struct)


# Catalogued candidates, parallel to substrate_lattice_waterfilling_v_us.py.
# DL from `proofs/foundations/dl_comparison.py`; channel-filter status from
# M1b.ii §3.6 M1 line.
CANDIDATES = [
    # (name, DL, k_C, chiral, contributes, notes)
    ('srs (Laves, I4_132)',         12.17, 3, True,  True,
     'Unique chiral 3D 3-regular Moore-saturating; (3,10)-cage'),
    ('R-7 ths (I4_1/amd)',          13.85, 3, False, False,
     'Centrosymmetric → R-12 chirality hard-gate; PS-sector blocked'),
    ('R-8 dia (Fd-3m)',             14.06, 3, False, False,
     'Centrosymmetric → R-12 hard-gate'),
    ('R-9 eta (P6_3/mmc)',          14.41, 3, False, False,
     'Centrosymmetric (P6_3/mmc has inversion centre)'),
    ('R-9 utj (P2_1/c)',            15.85, 3, False, False,
     'P2_1/c centrosymmetric'),
    ('R-4 d=4 crystallographic',    14.00, 4, None,  False,
     'd=4 → Cl(8) gauge fails Pati-Salam; R_ν is PS-sector → channel-gated'),
    ('R-5 d=5 crystallographic',    19.00, 5, None,  False,
     'd=5 → Cl(10) gauge fails Pati-Salam → channel-gated'),
    ('Petersen (finite)',            5.32, 3, None,  False,
     'Finite → no infinite ensemble; channel-zero'),
    ('K_{3,3} (finite)',             8.59, 3, None,  False,
     'Finite'),
    ('Honeycomb 2D (p6mm)',          9.67, 3, False, False,
     'p6mm centrosymmetric + d=2 Gleason-gated'),
    ('R-13 hyperbolic Kleinian',    41.00, 3, None,  False,
     'No Bloch decomposition; weight ≤ 2⁻⁴¹ negligible'),
]


def main():
    print("=" * 80)
    print("SUBSTRATE-LATTICE-AXIS WATERFILLING for R_ν (PS-sector, chirality channel)")
    print("Methodology: same K_4-Chebyshev formula as R_nu_splitting.py,")
    print("Boltzmann-weighted over A2-T-compressing chiral 3D 3-regular alternatives.")
    print("=" * 80)

    print(f"\n{'Candidate':<35s} {'DL':>6s} {'k':>3s} {'chir':>5s} {'w(C)':>11s} {'R_ν(C)':>9s}  Notes")
    print("-" * 80)

    total_weight = 0.0
    weighted_sum = 0.0
    for name, dl, k, chiral, contributes, notes in CANDIDATES:
        w = boltzmann_weight(dl)
        chir_str = "✗" if chiral is False else ("?" if chiral is None else "✓")
        if not contributes:
            print(f"  {name:<33s} {dl:>6.2f} {k:>3d} {chir_str:>5s} {'GATED':>11s} {'-':>9s}  {notes}")
            continue
        r = r_nu(k)
        total_weight += w
        weighted_sum += w * r
        print(f"  {name:<33s} {dl:>6.2f} {k:>3d} {chir_str:>5s} {w:>11.3e} {r:>9.4f}  {notes}")

    if total_weight == 0:
        print("\n*** ZERO CONTRIBUTORS — should be impossible (srs always contributes) ***")
        return

    r_nu_total = weighted_sum / total_weight
    r_nu_srs_only = float(Fraction(228, 7))
    r_nu_obs = 33.83
    sigma_obs = 0.92

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n  Total R_ν-channel weight Z        = {total_weight:.3e}")
    print(f"\n  R_ν (srs only, existing)          = {r_nu_srs_only:.6f}  = 228/7")
    print(f"  R_ν (lattice-axis waterfilled)    = {r_nu_total:.6f}")
    print(f"  R_ν (NuFIT 6.0 NO 2024)           = {r_nu_obs:.2f} ± {sigma_obs:.2f}")
    print(f"\n  srs-only - obs                    = {r_nu_srs_only - r_nu_obs:+.4f} = {(r_nu_srs_only - r_nu_obs)/sigma_obs:+.3f}σ")
    print(f"  waterfilled - obs                 = {r_nu_total - r_nu_obs:+.4f} = {(r_nu_total - r_nu_obs)/sigma_obs:+.3f}σ")
    print(f"  lattice-axis waterfilling shift   = {r_nu_total - r_nu_srs_only:+.6f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION (per parameter_linter.md no-post-hoc-backfill)")
    print("=" * 80)

    n_contributors = sum(1 for c in CANDIDATES if c[4])
    print(f"\n  Number of contributing candidates: {n_contributors}")
    if n_contributors == 1:
        print(f"  → srs is the UNIQUE catalogued contributor under M1b.ii §3.6 channel filtering.")
        print(f"  → R_ν(waterfilled) = R_ν(srs) = 228/7 exactly. No lattice-axis shift.")
        print(f"  → The +1.4σ residual vs NuFIT 6.0 is NOT explained by lattice-axis")
        print(f"    waterfilling; logged as Clause 8 soft tension, not a uniqueness vulnerability.")
        print(f"\n  CONDITIONAL: This conclusion presumes srs is the unique chiral 3D 3-regular")
        print(f"  Bloch-decomposable substrate (R-9 RCSR enumeration discharged via Phase 1d,")
        print(f"  per an internal working note). RCSR enumeration")
        print(f"  of any chiral non-srs 3D 3-regular nets would widen the contributor set;")
        print(f"  bound below.")

    # Bound: hypothetical R-9 contribution
    print("\n" + "=" * 80)
    print("BOUND ON R-9 (chiral non-srs 3D 3-regular nets, Phase 1d-discharged but")
    print("kept here for parallel structure with V_cb / V_us probes)")
    print("=" * 80)
    print(f"\n  Hypothetical: if M chiral non-srs entries exist at average ΔDL = +ΔDL bits")
    print(f"  over srs (DL ≈ 12.17), they would carry total weight w_R9 ≈ M·2^(−12.17−ΔDL).")
    print(f"  Each at k_C = 3 with the same K_4 quotient produces R_ν(C) = 228/7 (no shift).")
    print(f"  A different quotient structure (different K_C complete graph) would change R_ν;")
    print(f"  the bound is on the FRACTIONAL weight, not the magnitude of the shift.")

    for delta_dl in [1, 2, 3, 5]:
        for M in [1, 2, 5]:
            w_r9_total = M * 2.0**(-(12.17 + delta_dl))
            f_r9 = w_r9_total / (boltzmann_weight(12.17) + w_r9_total)
            # Maximal shift: hypothetical R_ν(R-9) anywhere in PDG-plausible range, say [20, 45]
            max_abs_shift = f_r9 * (45.0 - 20.0)
            print(f"  ΔDL=+{delta_dl}, M={M}: w_R9_total = {w_r9_total:.2e}, f_R9 = {f_r9:.3f}, "
                  f"max |R_ν shift| ≈ {max_abs_shift:.3f} (vs NuFIT σ={sigma_obs:.2f})")

    print(f"\n  Reading: even M=1 R-9 entry at ΔDL=+1 bit could shift R_ν by ~{0.333*25:.1f},")
    print(f"  much larger than current 0.92 NuFIT sensitivity. RCSR enumeration of chiral")
    print(f"  non-srs 3D 3-regular nets is therefore the load-bearing OPEN piece for")
    print(f"  R_ν robustness — same structural status as V_cb / V_us per Phase 1d closure.")


if __name__ == '__main__':
    main()
