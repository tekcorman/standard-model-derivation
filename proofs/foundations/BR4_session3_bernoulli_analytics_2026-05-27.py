#!/usr/bin/env python3
"""
BR4 session 3, Probe 1 — analytic test of Candidate B (Bernoulli δ = Q(1-Q))
across all 4 SM fermion species using the framework's existing ε² values.

The framework has theorem-grade ε² per species:
  ε²_lepton = 2     (W43, (4,2,2) at P)
  ε²_down   = 5/2   (W53, Type IV Perron)
  ε²_up     = 17/5  (Row P37)
  ε²_neutrino = (separate Type I spectral; not Koide-cosine)

The Koide-form algebraic identity gives Q = (1 + ε²/2) / 3.
The Bernoulli identity is δ = Q(1 - Q).

Test: does δ_Bernoulli^(s) match empirical δ_Koide^(s) for each species?
  - Lepton: known to match exactly (the framework's δ_Koide=2/9 IS Q(1-Q)).
  - Down: PREDICTS 3/16 ≈ 10.74° vs empirical ~5.8-6.3° at 2 GeV.
  - Up:   PREDICTS 9/100 ≈ 5.16° vs empirical ~4.27°.

Per `delta_Koide_derivation.md` 2026-05-08 note, the identification of
δ_Bernoulli (dimensionless variance) with δ_Koide (Koide cosine phase
in radians) is a NUMERICAL coincidence for the lepton — Need-B itself.
This probe quantifies for the first time whether the coincidence extends
to quarks.
"""

import math
from fractions import Fraction


def koide_Q_from_eps_sq(eps_sq):
    """Q = (1 + ε²/2) / 3 (algebraic identity from Koide cosine parametrization)."""
    return (1 + Fraction(eps_sq) / 2) / 3


def bernoulli_delta(Q):
    """δ_Bernoulli = Q (1 - Q)."""
    return Q * (1 - Q)


def main():
    print("=" * 76)
    print("BR4 session 3 — Candidate B Bernoulli identity analytics")
    print("=" * 76)
    print()
    print("Framework-derived ε² per species (theorem-grade):")
    print("  lepton: ε² = 2     (W43)")
    print("  down:   ε² = 5/2   (W53 Type IV Perron)")
    print("  up:     ε² = 17/5  (Row P37 ratio)")
    print()

    species = [
        ("lepton (charged)", Fraction(2),     0.222222, "2/9 (theorem-grade-conditional)"),
        ("down quark",       Fraction(5, 2),  None,     "PDG ~0.1011-0.1101 rad (5.80-6.31°)"),
        ("up quark",         Fraction(17, 5), None,     "PDG ~0.0745 rad (4.27°)"),
    ]

    # Empirical δ_down centred value (2 GeV scheme + m_b scheme average)
    delta_down_empirical_rad = math.radians((5.80 + 6.31) / 2)
    delta_up_empirical_rad = math.radians(4.27)

    print(f"{'Species':<20} {'ε²':>8} {'Q = (1+ε²/2)/3':>16} {'δ_Bernoulli = Q(1-Q)':>22} "
          f"{'δ in °':>10} {'Empirical':>26} {'Δ rel':>10}")
    print("-" * 130)

    results = []
    for name, eps_sq, delta_observed_rad, observed_label in species:
        Q = koide_Q_from_eps_sq(eps_sq)
        delta_B = bernoulli_delta(Q)
        delta_B_rad = float(delta_B)
        delta_B_deg = math.degrees(delta_B_rad)

        if delta_observed_rad is not None:
            rel = abs(delta_B_rad - delta_observed_rad) / delta_observed_rad
            rel_str = f"{rel*100:+.2f}%"
        else:
            # Use centred empirical
            if "down" in name:
                target = delta_down_empirical_rad
            elif "up" in name:
                target = delta_up_empirical_rad
            else:
                target = None
            if target is not None:
                rel = (delta_B_rad - target) / target
                rel_str = f"{rel*100:+.2f}%"
            else:
                rel_str = "N/A"

        results.append((name, eps_sq, Q, delta_B, delta_B_deg, rel_str))
        print(f"{name:<20} {str(eps_sq):>8} {str(Q):>16} {str(delta_B):>22} "
              f"{delta_B_deg:>+10.4f}° {observed_label:>26} {rel_str:>10}")

    print()
    print("Verdict per species:")
    for name, eps_sq, Q, delta_B, delta_B_deg, rel_str in results:
        # Parse rel_str for verdict
        rel_pct = float(rel_str.rstrip("%").lstrip("+"))
        abs_rel = abs(rel_pct)
        if abs_rel < 5:
            v = "MATCH (<5%)"
        elif abs_rel < 20:
            v = "BORDERLINE (5-20%)"
        else:
            v = "FAIL (>20%)"
        print(f"  {name:<20} δ_Bernoulli = {str(delta_B):<10} = {delta_B_deg:.4f}°   → {v}")

    print()
    print("=" * 76)
    print("STRUCTURAL READING")
    print("=" * 76)
    print()
    print("Candidate B predicts:")
    print(f"  Lepton: δ = 2/9 = 12.7324° (EXACT match to empirical 12.7324°)")
    print(f"  Down:   δ = 3/16 = 10.7430° (vs ~6° empirical → ~70-85% off → FAIL)")
    print(f"  Up:     δ = 9/100 = 5.1566° (vs 4.27° empirical → ~21% off → BORDERLINE)")
    print()
    print("Simple δ = Q(1-Q) Bernoulli identity does NOT unify across species.")
    print("The lepton match is the ORIGINAL Need-B coincidence; it does NOT generalise.")
    print()
    print("Possible interpretations:")
    print("  (a) The Bernoulli identity is genuinely a per-species coincidence,")
    print("      not a structural identity. Need-B requires a DIFFERENT mechanism.")
    print("  (b) The ε² values for quarks (5/2, 17/5) are correct at the ε level,")
    print("      but the δ-Bernoulli relation needs a per-species correction factor")
    print("      (e.g., walker-type-dependent prefactor).")
    print("  (c) The framework's ε²_down=5/2 and ε²_up=17/5 themselves need")
    print("      sub-class refinement (RG running, scheme-dependent correction).")
    print()
    print("This is HONEST NEGATIVE for the simple Bernoulli mechanism.")
    print()


if __name__ == "__main__":
    main()
