#!/usr/bin/env python3
"""
Investigation #4 — Does Q-space M_2 cos(2φ) modulation predict the y_τ
+0.13% bridge-systematic correction?

CONTEXT: parallels q_space_m_nu2_subleading_verification.py (Investigation
#2-followup, Class C dark coefficient √5/4). That probe found that M_2
modulation DEGRADES the m_ν2 PDG match — leading-only is best. This is
empirical confirmation of a separate private derivation by the author for Class-C
observables.

Class 2 (mass²) observables use a different mechanism: the dark coefficient
is the chirality-enhancement prefactor c = 5/3 (Re² / Im² + 1 = 5/3 from
h = (√3+i√5)/2). The modulation of |h|² or Re²(h) by M_2 cos(2φ) gives
a structurally different shift than Im(h)/|h|² (Class C).

This probe applies M_2 modulation to the y_τ chain and computes the shift
in the predicted y_τ value. y_τ = α_1_full / k*² with α_1_full = (5/3)·α_1
in current framework. The cos(2φ) modulation enters via the Feshbach
contour over the Ramanujan circle — modifying the effective Re²(h) - Im²(h)
ratio that determines the c_2 prefactor.

If M_2 = -0.27 (Inv #1 empirical) shifts y_τ by +0.13% → closes
y_τ at theorem grade. If it shifts by some other amount or wrong sign →
y_τ +0.13% remains bridge-systematic, M_2 modulation does NOT propagate
to Class 2 either.

VERDICT FORMAT:
- Δy_τ predicted by M_2 = -0.27 modulation
- Compare with observed +0.13% target
- If matches: Investigation #4 closes y_τ
- If not: M_2 modulation also doesn't propagate to Class 2 (parallel to #2-followup)
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))


# ============================================================
# Constants
# ============================================================

ALPHA_1_BARE = (2/3)**8
K_STAR = 3
G_GIRTH = 10

# Saddle h = (√3 + i√5)/2 — Ramanujan circle eigenvalue
H_RE = math.sqrt(3) / 2
H_IM = math.sqrt(5) / 2
H_MAG_SQ = H_RE**2 + H_IM**2  # = 2 (Ramanujan saturation: |h|² = k*-1 = 2)
ARG_H = math.atan2(H_IM, H_RE)

# Framework's current y_τ: α_1_full / k*² with α_1_full = (5/3)·α_1
C_2_LEADING = 5/3                          # Class-2 chirality prefactor from |h|² + 1 over Re²
ALPHA_1_FULL_LEADING = C_2_LEADING * ALPHA_1_BARE
Y_TAU_LEADING = ALPHA_1_FULL_LEADING / K_STAR**2

# PDG observed y_τ
M_TAU_OBS_GEV = 1.77686       # ± 0.00012 GeV
V_HIGGS_OBS_GEV = 246.22      # ± 0.12 GeV
Y_TAU_OBS = M_TAU_OBS_GEV / V_HIGGS_OBS_GEV   # ≈ 7.218 × 10⁻³


def main():
    print("=" * 88)
    print("  Investigation #4 — y_τ via M_2 cos(2φ) Q-space modulation")
    print("=" * 88)

    print(f"\nFramework constants:")
    print(f"  α_1_bare      = (2/3)^8 = {ALPHA_1_BARE:.10f}")
    print(f"  k*            = {K_STAR}")
    print(f"  c_2 leading   = 5/3 = {C_2_LEADING:.6f} (chirality prefactor from |h|²+1 over Re²)")
    print(f"  α_1_full      = c_2 · α_1_bare = {ALPHA_1_FULL_LEADING:.10f}")

    print(f"\nLeading-only y_τ (current framework):")
    print(f"  y_τ = α_1_full / k*² = {Y_TAU_LEADING:.10f}")

    dev_leading = (Y_TAU_LEADING - Y_TAU_OBS) / Y_TAU_OBS * 100
    print(f"  y_τ (PDG)            = {Y_TAU_OBS:.10f}")
    print(f"  Δ y_τ / y_τ_obs      = {dev_leading:+.4f}%   (current bridge-systematic gap)")

    print(f"\nSaddle h = (√3+i√5)/2:")
    print(f"  Re(h) = {H_RE:.6f}, Im(h) = {H_IM:.6f}, |h|² = {H_MAG_SQ:.6f}, arg h = {math.degrees(ARG_H):.4f}°")

    # Mechanism: c_2 = 5/3 = (Re²(h) + Im²(h)) / Re²(h) = |h|² / Re²(h)
    # With M_2 cos(2φ) modulation of the Q-space angular density,
    # the effective ⟨|h|²⟩ and ⟨Re²(h)⟩ shift by integrating over ρ_Q(φ).
    #
    # On the Ramanujan circle |h|² = 2, all eigenvalues have |h|² = 2 (saturation).
    # So ⟨|h|²⟩ = 2 doesn't shift under angular modulation.
    # But ⟨Re²(h)⟩ = ⟨|h|² cos²(φ)⟩ = 2 · ⟨cos²(φ)⟩ DOES shift.
    #
    # ⟨cos²(φ)⟩ under ρ_Q(φ) = (1 + 2 M_2 cos(2φ) + ...) / (2π):
    #   = (1/2π) ∫ cos²(φ) [1 + 2 M_2 cos(2φ) + ...] dφ
    #   = 1/2 + M_2 · (1/2π) ∫ 2cos²(φ)cos(2φ) dφ
    #   = 1/2 + M_2 · (1/2π) ∫ [cos(0) + cos(2φ)] cos(2φ) dφ
    #   = 1/2 + M_2 · (1/2)
    #   = 1/2 + M_2/2
    #
    # So with M_2 modulation:
    #   ⟨Re²(h)⟩_modulated = 2 · (1/2 + M_2/2) = 1 + M_2
    #
    # And c_2_modulated = ⟨|h|²⟩ / ⟨Re²(h)⟩ = 2 / (1 + M_2)

    print(f"\nM_2 modulation effect on c_2 prefactor:")
    print(f"  Mechanism: c_2 = ⟨|h|²⟩ / ⟨Re²(h)⟩")
    print(f"  At Ramanujan saturation ⟨|h|²⟩ = 2 (no shift; all eigs saturated)")
    print(f"  ⟨Re²(h)⟩ = 2·⟨cos²(φ)⟩ = 2·(1/2 + M_2/2) = 1 + M_2")
    print(f"  → c_2_modulated = 2 / (1 + M_2)")

    # M_2 sweep
    print(f"\n  Sensitivity sweep (M_2 → c_2 → Δy_τ):")
    print(f"    {'M_2':>8} {'c_2':>10} {'α_1_full':>14} {'y_τ':>14} {'Δ y_τ %':>10} {'Verdict':>20}")
    print("  " + "-" * 80)

    for M_2 in [-0.40, -0.30, -0.27, -0.20, -0.10, 0.00, +0.10, +0.20]:
        c_2 = 2.0 / (1.0 + M_2)
        alpha_1_full_mod = c_2 * ALPHA_1_BARE
        y_tau_mod = alpha_1_full_mod / K_STAR**2
        dev = (y_tau_mod - Y_TAU_OBS) / Y_TAU_OBS * 100
        # Closer to zero is better
        verdict = "PDG-improved ✓" if abs(dev) < abs(dev_leading) else "PDG-worsened ✗"
        marker = " ← Inv #3 empirical" if abs(M_2 - (-0.27)) < 1e-3 else ""
        marker = " ← a separate private derivation by the author leading" if abs(M_2) < 1e-6 else marker
        print(f"    {M_2:+8.3f} {c_2:>10.4f} {alpha_1_full_mod:>14.6e} {y_tau_mod:>14.6e} {dev:>+10.4f} {verdict:>20}{marker}")

    print()
    M_2_inv3 = -0.27
    c_2_inv3 = 2.0 / (1.0 + M_2_inv3)
    alpha_1_full_inv3 = c_2_inv3 * ALPHA_1_BARE
    y_tau_inv3 = alpha_1_full_inv3 / K_STAR**2
    dev_inv3 = (y_tau_inv3 - Y_TAU_OBS) / Y_TAU_OBS * 100

    print("=" * 88)
    print("VERDICT — does M_2 = −0.27 modulation predict y_τ +0.13%?")
    print("=" * 88)
    print(f"\n  Leading-only:    y_τ = {Y_TAU_LEADING:.6e}, Δ = {dev_leading:+.4f}%")
    print(f"  M_2 = −0.27:     y_τ = {y_tau_inv3:.6e}, Δ = {dev_inv3:+.4f}%")
    print(f"  Δ shift from M_2 = {dev_inv3 - dev_leading:+.4f}%")
    print(f"  Target (closes +0.13% bridge gap): need Δ_modulated ≈ 0%, i.e., shift ≈ -0.13%")
    print()

    if abs(dev_inv3) < abs(dev_leading) and dev_inv3 > 0:
        print("  ✓ M_2 modulation IMPROVES match — Investigation #4 PARTIALLY POSITIVE")
        print("    Closer match suggests M_2 modulation DOES propagate to Class-2 observables.")
    elif abs(dev_inv3) < 0.05:
        print("  ✓✓ M_2 modulation CLOSES y_τ — Investigation #4 POSITIVE")
        print("    +0.13% bridge-systematic gap is structurally explained by M_2 cos(2φ).")
    elif dev_inv3 > 5 * dev_leading:
        print("  ✗✗ M_2 modulation DRAMATICALLY DEGRADES match — Inv #4 NEGATIVE")
        print("    Same pattern as Inv #2-followup (m_ν2): M_2 doesn't propagate to SM.")
    else:
        print(f"  ✗ M_2 modulation DEGRADES match (Δ_mod = {dev_inv3:+.3f}% vs leading {dev_leading:+.3f}%)")
        print("    Same pattern as Inv #2-followup (m_ν2): M_2 modulation is substrate noise")
        print("    below MDL waterline; does NOT propagate to Class-2 SM observables.")
        print("    a separate private derivation by the author holds for Class-2 as well.")

    print()
    print(f"  Inheritance from Inv #2-followup:")
    print(f"    Inv #2-followup tested M_2 on m_ν2 (Class C, Im(h)/|h|² family) — DEGRADES.")
    print(f"    Inv #4 tests M_2 on y_τ (Class 2, Re²/|h|² prefactor)        — see verdict.")
    print(f"  If both negative: a separate private derivation by the author water-filling holds across BOTH classes.")
    print(f"  y_τ +0.13% remains bridge-systematic; closure requires different mechanism.")


if __name__ == "__main__":
    main()
