#!/usr/bin/env python3
"""
Probe: substrate-lattice-axis waterfilling for Ω_DM/Ω_m, leveraging existing
dark-fraction methodology from `predictions/Omega_DM_over_Omega_m.py`.

CONTEXT
-------
The existing Omega_DM derivation does waterfilling on the MODE AXIS:
    Ω_DM/Ω_m = 1 - P(k ≤ k* | Poisson(2k*)) = 1 - P(k≤3 | Poisson(6)) = 0.8488
assuming the substrate is exactly srs (k* = 3, d = 3).

This probe extends to the LATTICE AXIS: under proper A2-T waterline retention
(`docs/framework/framework_axioms.md` §3), every above-waterline substrate alternative
is retained with Boltzmann weight w(C) = 2^(-DL_struct(C)). For dark/cosmo
channel C4 (per an internal working note §2),
each Bloch-decomposable infinite substrate alternative contributes its own
Ω_DM(C) = 1 - P(k ≤ k_C | Poisson(2k_C)), Boltzmann-weighted.

Methodology: SAME Poisson-tail formula as existing Omega_DM_over_Omega_m.py;
just sum over substrate alternatives instead of using srs alone. No new
mathematics — extends established framework methodology to the lattice axis.

OUTPUT
------
Channel-C4 Boltzmann-weighted Ω_DM prediction. Comparison to srs-only.
Result interpretation per `docs/parameters/parameter_linter.md` no-post-hoc-backfill.
"""

import math

# DL_struct values from `proofs/foundations/dl_comparison.py`
# (existing dl_comparison.py output for crystal-net candidates).
# d>3 candidates (R-4, R-5) estimated via log₂|SG(d)| + Wyckoff overhead.

CANDIDATES = [
    # (name, DL_struct (bits), k_C, d_C, channel_C4_contributes, notes)
    ('srs (chiral 3D 3-reg, I4_132)',           12.17, 3, 3, True,  'MDL minimum, all channels'),
    ('R-7 ths (centrosym 3D 3-reg)',            13.85, 3, 3, True,  'C3 chirality hard-gated; C4 alive'),
    ('R-8 dia (centrosym 3D 3-reg)',            14.06, 3, 3, True,  'C3 chirality hard-gated; C4 alive'),
    ('eta (non-vertex-trans 3D 3-reg)',         14.41, 3, 3, True,  'C1 partial; C4 alive'),
    ('utj (low-symm 3D 3-reg)',                 15.85, 3, 3, True,  'C1 partial; C4 alive'),
    ('R-4 d=4 crystallographic (k=4)',          14.00, 4, 4, True,  'C5 LIV + C6 gauge hard-gated; C4 alive'),
    ('R-5 d=5 crystallographic (k=5)',          19.00, 5, 5, True,  'C5 LIV + C6 gauge hard-gated; C4 alive'),
    ('Petersen (finite, k=3)',                   5.32, 3, None, False, 'No infinite ensemble; C4 zero'),
    ('K_{3,3} (finite, k=3)',                    8.59, 3, None, False, 'No infinite ensemble; C4 zero'),
    ('Honeycomb 2D (k=3)',                       9.67, 3, 2, True,  'd=2 Gleason-gated; C4 partial via 2D ensemble'),
    ('R-13 hyperbolic Kleinian (Plancherel)',   41.00, 3, 3, True,  'Bloch-equivalent absent; weight ≤ 2^-41 negligible'),
]


def omega_dm(k_star):
    """
    Ω_DM/Ω_m on substrate with coordination k_star.
    Mirrors `predictions/Omega_DM_over_Omega_m.py` predict_Omega_DM_over_Omega_m.

    Ω_DM/Ω_m = 1 - P(k ≤ k_star | Poisson(2 k_star))
    """
    lam = 2 * k_star
    cdf = sum(
        math.exp(-lam) * lam**j / math.factorial(j)
        for j in range(k_star + 1)
    )
    return 1.0 - cdf


def boltzmann_weight(dl_struct):
    """Substrate Boltzmann weight w(C) = 2^(-DL_struct(C))."""
    return 2.0 ** (-dl_struct)


def main():
    print("=" * 78)
    print("SUBSTRATE-LATTICE-AXIS WATERFILLING for Ω_DM/Ω_m (channel C4)")
    print("Methodology: same Poisson-tail formula as Omega_DM_over_Omega_m.py,")
    print("Boltzmann-weighted over A2-T-compressing lattice alternatives.")
    print("=" * 78)

    print(f"\n{'Candidate':<42s} {'DL':>6s} {'k_C':>4s} {'w(C)':>11s} {'Ω(C)':>8s}  Channel-C4")
    print("-" * 78)

    total_weight = 0.0
    weighted_sum = 0.0
    for name, dl, k_c, d_c, contributes, notes in CANDIDATES:
        w = boltzmann_weight(dl)
        if not contributes:
            print(f"  {name:<40s} {dl:>6.2f} {k_c:>4d} {'GATED':>11s} {'-':>8s}  {notes}")
            continue
        omega = omega_dm(k_c)
        total_weight += w
        weighted_sum += w * omega
        print(f"  {name:<40s} {dl:>6.2f} {k_c:>4d} {w:>11.3e} {omega:>8.4f}  {notes}")

    omega_total = weighted_sum / total_weight if total_weight > 0 else float('nan')
    omega_srs_only = omega_dm(3)
    omega_obs = 0.842
    sigma_obs = 0.016

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"\n  Total C4 weight Z_C4              = {total_weight:.3e}")
    print(f"  srs's share f_srs(C4) = w_srs/Z   = {boltzmann_weight(12.17)/total_weight:.4f}  (= {boltzmann_weight(12.17)/total_weight*100:.1f}%)")
    print(f"\n  Ω_DM (srs only, existing)         = {omega_srs_only:.6f}")
    print(f"  Ω_DM (lattice-axis waterfilled)   = {omega_total:.6f}")
    print(f"  Ω_DM (observed PDG 2024)          = {omega_obs:.4f} ± {sigma_obs:.3f}")
    print(f"\n  srs-only - obs                    = {omega_srs_only - omega_obs:+.5f} = {(omega_srs_only - omega_obs)/sigma_obs:+.2f}σ")
    print(f"  waterfilled - obs                 = {omega_total - omega_obs:+.5f} = {(omega_total - omega_obs)/sigma_obs:+.2f}σ")
    print(f"  lattice-axis waterfilling shift   = {omega_total - omega_srs_only:+.5f}")

    print("\n" + "=" * 78)
    print("INTERPRETATION (per parameter_linter.md no-post-hoc-backfill)")
    print("=" * 78)
    shift = omega_total - omega_srs_only
    if abs(shift) < sigma_obs / 3:
        verdict = "BELOW SENSITIVITY"
        comment = "Lattice-axis waterfilling shift is below 1/3 of observational uncertainty.\n  Existing srs-only prediction is robust against lattice-axis alternatives in C4."
    elif (omega_total - omega_obs) * (omega_srs_only - omega_obs) < 0:
        verdict = "OVERSHOOTS"
        comment = "Waterfilling shift crosses the observed value — overcorrection.\n  Some candidate(s) need additional channel-specific hard-gate."
    elif abs(omega_total - omega_obs) < abs(omega_srs_only - omega_obs):
        verdict = "IMPROVES MATCH"
        comment = "Waterfilling moves prediction closer to observation.\n  Non-srs lattice contributions partially explain the residual srs-only deviation."
    else:
        verdict = "WORSENS MATCH"
        comment = "Waterfilling moves prediction further from observation.\n  Non-srs lattice contributions worsen the fit; suggests some candidates\n  shouldn't contribute to C4 (additional hard-gate needed)."
    print(f"\n  Verdict: {verdict}")
    print(f"  {comment}")

    # R-13 specific bound (per substrate_a2t_waterfilling_program.md §3c)
    w_r13 = boltzmann_weight(41.0)
    r13_fraction = w_r13 / total_weight
    print("\n" + "=" * 78)
    print("R-13 BOUND (non-circular, per Phase 1 doc §3c)")
    print("=" * 78)
    print(f"\n  R-13 Boltzmann weight             = w_R13 = 2^-41 = {w_r13:.3e}")
    print(f"  R-13 fraction of C4 channel       = w_R13/Z_C4 = {r13_fraction:.3e}")
    print(f"  R-13 max contribution to Ω_DM     ≤ {r13_fraction * 1.0:.3e}")
    print(f"  vs PDG sensitivity                  ≈ {sigma_obs:.3f} = {sigma_obs/r13_fraction:.0e}× larger")
    print(f"\n  R-13 is BELOW SENSITIVITY by {sigma_obs/r13_fraction:.0e}× — proper non-circular bound,")
    print(f"  in contrast to Stage 1b' attempted categorical-exclusion (RETRACTED as circular).")


if __name__ == '__main__':
    main()
