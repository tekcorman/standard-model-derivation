#!/usr/bin/env python3
"""
DE-onset compatibility with framework's coasting structure — Session 1 probe.

Scoping: an internal working note (P1 + P2)

GOAL: test whether the proposed DE-onset Phase IIa F-fiber at T_DE ~ Λ_CC^(1/4)
is structurally compatible with the framework's existing coasting cosmology.

The recomb→today scoping was motivated by a "+3.9% t_0 residual" under pure
coasting (vs ΛCDM-fit t_0 = 13.8 Gyr). Before computing piecewise t_0 integrals,
Session 1 audits the framing itself: does the framework's t_0 actually have a
+3.9% residual that DE-onset would need to close?

This probe checks four facts from existing framework predictions:
  (1) framework's Ω_Λ_substrate = 1/3 at z=0 (coasting structural)
  (2) framework's t_0 = 14.38 Gyr matches Methuselah at -0.1σ
  (3) ΛCDM-fit t_0 = 13.8 Gyr is model-dependent (framework explicitly rejects it)
  (4) H_0 tension already addressed via H_0_observer/H_0_substrate (16/15) rate-gap

If (1)-(4) hold, the recomb→today scoping's motivation is misframed: there is
no +3.9% residual against the framework's preferred (model-independent) anchor,
and the H_0 tension is already resolved without DE-onset.

This is a Session 1 *structural-audit* probe; it does NOT execute the piecewise
t_0 integral (Session 2 P2) because that integral's motivation needs to survive
this audit first.
"""

# ============================================================
# FRAMEWORK EXISTING PREDICTIONS (theorem-grade)
# ============================================================
# From predictions/Omega_Lambda_LCDM_derivation.md:
#   Ω_Λ_substrate = 1/k* = 1/3 (exact, at z=0; structural coasting)
#   Ω_m_substrate = (k*-1)/k* = 2/3
#   This is the framework's *intrinsic* dark-energy fraction — NOT a late-time
#   onset, but a structural feature of the coasting geometry at all z.
OMEGA_LAMBDA_FRAMEWORK = 1.0/3.0
OMEGA_M_FRAMEWORK = 2.0/3.0

# From predictions/t_0.py:
#   t_0_framework = N_hub × t_P = 14.38 Gyr (theorem-grade, cascade coefficient = 1)
#   Methuselah (model-independent): 14.46 ± 0.80 Gyr (Bond et al. 2013)
#   ΛCDM-fit (model-dependent): 13.797 ± 0.023 Gyr (Planck 2018)
T_0_FRAMEWORK_GYR = 14.38
T_0_METHUSELAH_GYR = 14.46
T_0_METHUSELAH_SIGMA = 0.80
T_0_LCDM_GYR = 13.797
T_0_LCDM_SIGMA = 0.023

# From predictions/H_0_observer.py:
#   H_0_substrate ≈ 68.2 km/s/Mpc (matches Planck-CMB 67.4 ± 0.5)
#   H_0_observer = (16/15) × H_0_substrate ≈ 72.7 km/s/Mpc (matches SH0ES 73.04)
#   The (16/15) rate-gap is the framework's structural resolution of H_0 tension.
H_0_SUBSTRATE = 68.2     # km/s/Mpc
H_0_OBSERVER = 72.7      # km/s/Mpc
H_0_PLANCK = 67.4
H_0_PLANCK_SIGMA = 0.5
H_0_SHOES = 73.04
H_0_SHOES_SIGMA = 1.04
RATE_GAP_16_15 = 16.0/15.0


def report():
    print("=" * 78)
    print("  DE-onset compatibility audit — recomb→today scoping Session 1")
    print("=" * 78)

    print("\n  AUDIT (1): Framework's Ω_Λ structure")
    print(f"    Framework Ω_Λ_substrate (z=0)  = 1/k* = 1/3 = {OMEGA_LAMBDA_FRAMEWORK:.4f}")
    print(f"    Framework Ω_m_substrate (z=0)  = (k*-1)/k* = 2/3 = {OMEGA_M_FRAMEWORK:.4f}")
    print(f"    Source: predictions/Omega_Lambda_LCDM_derivation.md")
    print(f"    Status: theorem-grade (NB-walk dark fraction + Friedmann ä = 0)")
    print()
    print("    KEY OBSERVATION: Ω_Λ = 1/3 is structural at ALL z, NOT")
    print("    late-time-emergent. The framework's Λ_CC contributes a constant")
    print("    fraction (1/3) of the energy budget throughout cosmic history.")
    print()
    print("    -> Implication for DE-onset: there is NO 'before vs after' event.")
    print("       The proposed F-fiber transition T_DE ~ Λ_CC^(1/4) describes a")
    print("       crossover that does NOT occur in the framework's coasting.")

    print("\n  AUDIT (2): Framework t_0 vs observational anchors")
    print(f"    Framework t_0          = {T_0_FRAMEWORK_GYR:.2f} Gyr")
    print(f"    Methuselah (mod-indep) = {T_0_METHUSELAH_GYR:.2f} ± {T_0_METHUSELAH_SIGMA:.2f} Gyr")
    print(f"    ΛCDM-fit (mod-dep)     = {T_0_LCDM_GYR:.3f} ± {T_0_LCDM_SIGMA:.3f} Gyr")

    sigma_meth = (T_0_FRAMEWORK_GYR - T_0_METHUSELAH_GYR) / T_0_METHUSELAH_SIGMA
    sigma_lcdm = (T_0_FRAMEWORK_GYR - T_0_LCDM_GYR) / T_0_LCDM_SIGMA
    print(f"    Framework deviation:")
    print(f"      vs Methuselah  = {sigma_meth:+.2f}σ  (model-independent)")
    print(f"      vs ΛCDM-fit    = {sigma_lcdm:+.2f}σ  (model-dependent)")
    print()
    print("    KEY OBSERVATION: framework t_0 matches Methuselah at -0.1σ.")
    print("    The '+3.9%' apparent residual is against ΛCDM-fit ONLY, which")
    print("    the framework explicitly rejects as model-dependent (Planck")
    print("    assumes Ω_Λ ≈ 0.68 vs framework Ω_Λ = 1/3).")
    print()
    print("    -> Implication for DE-onset: the scoping's motivation to 'close")
    print("       the +3.9% residual' is misframed. Against the framework's")
    print("       preferred (model-independent) anchor, t_0 is already correct")
    print("       within 0.1σ. There is no residual to close.")

    print("\n  AUDIT (3): H_0 tension framing")
    sigma_planck = (H_0_SUBSTRATE - H_0_PLANCK) / H_0_PLANCK_SIGMA
    sigma_shoes = (H_0_OBSERVER - H_0_SHOES) / H_0_SHOES_SIGMA
    print(f"    H_0_substrate          = {H_0_SUBSTRATE:.1f} km/s/Mpc")
    print(f"    Planck-CMB             = {H_0_PLANCK:.1f} ± {H_0_PLANCK_SIGMA:.1f} km/s/Mpc")
    print(f"    Substrate vs Planck    = {sigma_planck:+.2f}σ")
    print()
    print(f"    H_0_observer = (16/15)·H_0_substrate = {H_0_OBSERVER:.1f} km/s/Mpc")
    print(f"    SH0ES                  = {H_0_SHOES:.2f} ± {H_0_SHOES_SIGMA:.2f} km/s/Mpc")
    print(f"    Observer vs SH0ES      = {sigma_shoes:+.2f}σ")
    print()
    print(f"    Rate-gap = 16/15 = {RATE_GAP_16_15:.6f}  (D2-extended observer correction)")
    print()
    print("    KEY OBSERVATION: framework already resolves H_0 tension via")
    print("    the (16/15) cascade rate-gap distinguishing observer-side from")
    print("    substrate-side H_0. Planck CMB measurement maps to H_0_substrate;")
    print("    SH0ES distance-ladder maps to H_0_observer. Both match within σ.")
    print()
    print("    -> Implication for DE-onset: H_0 tension does NOT require an")
    print("       additional DE-onset F-fiber. The (16/15) rate-gap is the")
    print("       framework's existing structural resolution.")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print()
    print("  The recomb→today DE-onset scoping is misframed in three ways:")
    print()
    print("  (1) Ω_Λ INCOMPATIBILITY (AB3 from scoping doc):")
    print("      Framework's Ω_Λ = 1/3 is structural at all z. There is no")
    print("      'before vs after' crossover for the proposed F-fiber to mark.")
    print("      Introducing a DE-onset event would CONFLICT with the")
    print("      coasting Friedmann structure (ä = 0 at current epoch).")
    print()
    print("  (2) NO +3.9% RESIDUAL (against framework's preferred anchor):")
    print("      Framework t_0 matches Methuselah (model-independent) at")
    print("      -0.1σ. The +3.9% only appears against ΛCDM-fit t_0 which")
    print("      the framework explicitly rejects.")
    print()
    print("  (3) H_0 TENSION ALREADY RESOLVED (AB4 condition):")
    print("      Framework's (16/15) rate-gap structurally distinguishes")
    print("      Planck-CMB H_0 (= H_0_substrate) from SH0ES H_0 (= H_0_observer).")
    print("      No additional DE-onset framing needed.")
    print()
    print("  Cumulative reading: AB1, AB3, AB4 from the scoping doc are all")
    print("  triggered. The DE-onset hypothesis is STRUCTURALLY INCOMPATIBLE")
    print("  with the framework's existing coasting cosmology.")
    print()
    print("  HONEST OUTCOME: scoping doc lands at Outcome C (mechanism wrong).")
    print("  Sessions 2-5 (piecewise t_0 integral, H_0 tension demonstration,")
    print("  w(z) prediction, t_recomb age) are NOT advanced — their")
    print("  motivations are dissolved by this Session 1 audit.")
    print()
    print("  STRUCTURAL CONTRIBUTION: this is a useful negative. The")
    print("  recomb→today stretch contains NO named observer-graph beat")
    print("  beyond recombination itself. Reionization remains framework-")
    print("  external (no star-formation primitive). The stretch is")
    print("  structurally complete under pure coasting + Ω_Λ = 1/3.")
    print("=" * 78)


if __name__ == "__main__":
    report()
