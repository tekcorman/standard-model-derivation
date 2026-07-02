"""
proofs/foundations/unused_ramanujan_saddles_investigation_2026-05-11.py

Investigation of the 3 previously-unused Ramanujan saddles found by
exhaustive enumeration: h_N, h_H, h_Γ. The framework uses only h_P;
this script asks what observables would emerge if each saddle were
substituted into existing dark-map / Feshbach predictions, and whether
any match known unclosed observable values.

Saddles (all with |h|² = k*−1 = 2):
  h_P = (√3 + i√5)/2   arg ≈ +52.24°   tan²(arg) = 5/3
  h_N = (√5 + i√3)/2   arg ≈ +37.76°   tan²(arg) = 3/5  ← REAL/IMAG SWAP of h_P
  h_H = (1 + i√7)/2    arg ≈ +69.30°   tan²(arg) = 7
  h_Γ = (-1 + i√7)/2   arg ≈ +110.70°  tan²(arg) = 7    ← negative real

Dark-map constants from h_P (framework's existing apparatus):
  Class-1 amplitude:    ν_amp(h) = |Im(h)|/|h|² = √5/4 ≈ 0.5590
  Class-2 mass²:        ν_mass²(h) = tan²(arg h) = 5/3 ≈ 1.6667
  Class-3 edge-local:   ν_edge(h) = 1 (handled by other structure)

For each unused saddle, compute these three constants and check against
unclosed observable values.
"""

import math
import sys
from pathlib import Path
from fractions import Fraction

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def saddle_dark_constants(h_re, h_im):
    """Return (|h|², arg in degrees, Class-1 amplitude, Class-2 mass²)."""
    h_abs2 = h_re ** 2 + h_im ** 2
    arg_deg = math.degrees(math.atan2(h_im, h_re))
    nu_amp = abs(h_im) / h_abs2  # Class-1 amplitude
    nu_mass2 = (h_im / h_re) ** 2  # Class-2 mass² (tan²(arg))
    return h_abs2, arg_deg, nu_amp, nu_mass2


# All four saddles
SADDLES = {
    'h_P': (math.sqrt(3) / 2, math.sqrt(5) / 2),       # Re=√3/2, Im=√5/2
    'h_N': (math.sqrt(5) / 2, math.sqrt(3) / 2),       # SWAP
    'h_H': (1 / 2, math.sqrt(7) / 2),                  # (1+i√7)/2
    'h_Gamma': (-1 / 2, math.sqrt(7) / 2),             # negation of h_H_bar
}


def main():
    print("=" * 100)
    print("Unused Ramanujan saddles investigation")
    print("=" * 100)
    print()

    print(f"{'saddle':<10} {'Re(h)':>10} {'Im(h)':>10} {'|h|²':>8} {'arg°':>8} {'tan²(arg)':>10} {'|Im|/|h|²':>10}")
    print("-" * 100)
    for name, (re, im) in SADDLES.items():
        h_abs2, arg, nu_amp, nu_mass2 = saddle_dark_constants(re, im)
        marker = ""
        if name == 'h_P':
            marker = "  ← USED in framework"
        else:
            marker = "  ← UNUSED in framework predictions"
        print(f"{name:<10} {re:>+10.6f} {im:>+10.6f} {h_abs2:>8.4f} {arg:>+8.2f} "
              f"{nu_mass2:>10.6f} {nu_amp:>10.6f}{marker}")

    print()
    print(f"  All saddles saturate the Ramanujan bound |h|² = k*-1 = 2.")
    print(f"  Each has a distinct (Re, Im) structure and thus distinct dark-map ratios.")
    print()

    # ============================================================
    # Hypothesis test: substitute unused saddles into known dark predictions
    # ============================================================
    print("=" * 100)
    print("Hypothesis test: each unused saddle as alternative dark sector")
    print("=" * 100)
    print()

    print(f"{'saddle':<10} {'Class-1 ν_amp':>14} {'Class-2 ν_mass²':>17} {'arg°':>10} {'known target match'}")
    print("-" * 100)

    KNOWN_TARGETS = {
        'V_us / V_cb / etc dark correction = 5/12': 5/12,
        '5/3 dark map ratio (h_P)': 5/3,
        '|Im(h_P)|/|h_P|² = √5/4': math.sqrt(5) / 4,
        '1 (Class-3 edge-local)': 1.0,
        '√5/4 ≈ 0.559': math.sqrt(5) / 4,
        '√3/4 ≈ 0.433': math.sqrt(3) / 4,
        '√7/4 ≈ 0.661': math.sqrt(7) / 4,
        'arctan(1/3) ≈ 18.43°': math.degrees(math.atan(1/3)),
        '70.53° ≈ arccos(1/3)': math.degrees(math.acos(1/3)),
    }

    KNOWN_ANGLES = {
        'arg(h_P)': math.degrees(math.atan2(math.sqrt(5)/2, math.sqrt(3)/2)),
        'arccos(1/3)': math.degrees(math.acos(1/3)),
        'π/4 = 45°': 45.0,
        'π/3 = 60°': 60.0,
        '37.76° (h_N)': math.degrees(math.atan2(math.sqrt(3)/2, math.sqrt(5)/2)),
        '69.30° (h_H, arctan√7)': math.degrees(math.atan(math.sqrt(7))),
        '110.70° (h_Γ)': 180 - math.degrees(math.atan(math.sqrt(7))),
    }

    for name, (re, im) in SADDLES.items():
        h_abs2, arg, nu_amp, nu_mass2 = saddle_dark_constants(re, im)
        # Match candidates
        matches = []
        for tname, tval in KNOWN_TARGETS.items():
            if abs(nu_amp - tval) < 0.005:
                matches.append(f"ν_amp~{tname}")
            if abs(nu_mass2 - tval) < 0.005:
                matches.append(f"ν_mass²~{tname}")
        for tname, tval in KNOWN_ANGLES.items():
            if abs(abs(arg) - tval) < 0.5:
                matches.append(f"arg~{tname}")
        match_str = "; ".join(matches[:2]) if matches else "—"
        print(f"{name:<10} {nu_amp:>14.6f} {nu_mass2:>17.6f} {arg:>+10.2f} {match_str}")

    print()

    # ============================================================
    # Specific quark/lepton differentiation hypothesis (R-14)
    # ============================================================
    print("=" * 100)
    print("R-14 hypothesis: h_P for one sector, h_N for the other (R/I swap)")
    print("=" * 100)
    print()
    print("  h_P → tan²(arg) = 5/3, used for y_τ family (lepton sector mass²)")
    print("  h_N → tan²(arg) = 3/5, COULD be used for quark sector mass²")
    print()
    print("  Test prediction: if quark mass-² Feshbach factor = 1 + (3/5)·α₁/(1−α₁)")
    print("  vs framework's lepton factor = 1 + (5/3)·α₁/(1−α₁)")
    print()
    alpha_1_full = (5/3) * (2/3)**8  # framework's α₁_full
    print(f"  α₁_full = (5/3)·(2/3)^8 = {alpha_1_full:.10f}")
    print()

    # Framework's lepton-sector mass² correction
    factor_lep = 1 + (5/3) * alpha_1_full / (1 - alpha_1_full)
    factor_quark_candidate = 1 + (3/5) * alpha_1_full / (1 - alpha_1_full)
    print(f"  Framework lepton (h_P):     1 + (5/3)·α₁/(1−α₁) = {factor_lep:.10f}")
    print(f"  Quark candidate (h_N):      1 + (3/5)·α₁/(1−α₁) = {factor_quark_candidate:.10f}")
    print()
    print(f"  Ratio lep/quark = {factor_lep / factor_quark_candidate:.10f}")
    print(f"  Difference (lep - quark) = {(factor_lep - factor_quark_candidate)*100:.4f}%")
    print()
    print(f"  Comparison to observed Yukawa ratios:")
    print(f"    y_b / y_τ ≈ 2.35 (observed)")
    print(f"    y_t / y_τ ≈ 97 (observed)")
    print(f"    If h_N were the quark-sector saddle and dark map gave a simple Class-2")
    print(f"    correction, the ratio between quark and lepton sectors would be")
    print(f"    (1 + (3/5)·α₁/(1−α₁)) / (1 + (5/3)·α₁/(1−α₁)) ≈ {factor_quark_candidate / factor_lep:.6f}")
    print(f"    — far from 2.35 or 97. Simple h_N-as-quark-sector substitution does NOT")
    print(f"    reproduce the observed quark Yukawa hierarchy.")
    print()
    print(f"  CONCLUSION on this specific hypothesis: NO MATCH for direct Yukawa ratios.")
    print(f"  But h_N is still a structurally distinct saddle and may map to different")
    print(f"  observables (sub-leading dark corrections, sector phases, etc.).")

    # ============================================================
    # PMNS angles candidates
    # ============================================================
    print()
    print("=" * 100)
    print("PMNS angle candidates from unused saddles")
    print("=" * 100)
    print()
    print(f"  Each saddle has its own arg(h). These are candidate angles for PMNS:")
    print()
    pmns = {
        'θ_12 ≈ 33.45° (NuFIT)': 33.45,
        'θ_13 ≈ 8.57° (NuFIT)': 8.57,
        'θ_23 ≈ 49.7° (NuFIT)': 49.7,
        'δ_CP_PMNS ≈ 177° (NuFIT)': 177,
        'TBM θ_12 = 35.26°': math.degrees(math.atan(math.sqrt(2)/2)),
    }
    for name, (re, im) in SADDLES.items():
        h_abs2, arg, nu_amp, nu_mass2 = saddle_dark_constants(re, im)
        print(f"  {name}: arg = {arg:+.2f}°")
        for pname, pval in pmns.items():
            diff = abs(abs(arg) - pval)
            if diff < 5:
                print(f"      candidate match: {pname} (Δ = {diff:.2f}°)")
            # Also try complementary
            diff_compl = abs(90 - abs(arg) - pval)
            if diff_compl < 5:
                print(f"      candidate match: 90°−arg = {pname} (Δ = {diff_compl:.2f}°)")

    print()
    print("=" * 100)
    print("Net assessment")
    print("=" * 100)
    print("""
  - 4 Ramanujan saddles confirmed at h.s. k-points, only 1 used.
  - h_N (real/imag swap of h_P) gives tan² = 3/5 = inverse of h_P's 5/3.
  - h_H, h_Γ give tan² = 7 (a new integer not in framework's current constants).
  - Direct quark/lepton-sector swap of h_P → h_N does NOT reproduce
    observed Yukawa hierarchy (y_b/y_τ, y_t/y_τ); the ratio would be O(1).
  - But h_N's structure (3/5, arg 37.76°) is a NEW substrate-derivable
    object that could enter as sub-leading correction, sector-phase, or
    PMNS-mixing component. Further investigation needed for specific
    matches to unmatched observables.
  - h_H, h_Γ: tan²=7 / arg 69.30° / 110.70° are also substrate-derivable
    new objects. Could match θ_23 ≈ 49.7° via complementary arg or other.
  - The triality of three Ramanujan saddles at (Γ, P, N, H) k-points is
    a structural feature the framework has NOT yet used. This is the
    actual exhaustive-enumeration finding.
""")


if __name__ == "__main__":
    main()
