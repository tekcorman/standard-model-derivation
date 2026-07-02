#!/usr/bin/env python3
"""
Canonical prediction file for theta_23_PMNS (PMNS atmospheric mixing angle).
"""

# ============================================================
# PARAMETER: theta_23_PMNS (PMNS atmospheric mixing angle)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       49.2° ± 1.3°
# Source:      PDG 2024 Review of Particle Physics, Neutrino mixing §14
# PDG edition: 2024
# Note:        NuFIT 6.0 (Sep 2024) gives 49.0° ± 1.2° (normal ordering);
#              PDG 2024 used here for consistency with other project files.

# --- PREDICTED VALUE -----------------------------------------
# Value:       arctan((1 + α₁_full) / (1 − α₁_full))
#            = arctan(20963 / 18403)
#            ≈ 48.72°
# Deviation:   −0.48° absolute  (−0.37σ from PDG 2024)
# Status:      STRICT-SOLID THEOREM-GRADE (graduated 2026-04-28 via
#              `docs/theorems/theorem_dark_map_class2_closure.md` Theorem 5.1;
#              dark-map Class-2 taxonomy closed for θ_23). I-Feshbach
#              previously subsumed by A5(b) 2026-04-19. Audit anchor: Row P13
#              of `docs/parameters/parameter_uniqueness_ledger.md`.
#              Conditional on Row 18 (C³_obs), Rows 16/17 (PS), and A5(b)
#              Level 3 prescription. Clause 8 PASS at −0.37σ from PDG 2024.
#
# Bridge convention (docs/framework/framework_scheme_convention.md §7): θ_23 is a
# Level-2 srs-intrinsic α₁-dependent angle under A5(b) Case (A). The
# tan²(arg h) = 5/3 chirality content in α₁_full encodes the mass²-class
# Feshbach correction. The dark-map Class-2 taxonomy gap (cited in earlier
# 2026-04-19 banner) is now closed; residual −0.37σ is sub-Feshbach.

# --- DERIVED FORMULA -----------------------------------------
# θ₂₃ = arctan((1 + α₁_full) / (1 − α₁_full))
#
# where α₁_full = tan²(arg h) × α₁_bare = (5/3) × (2/3)^8 = 1280/19683.
#
# Derivation chain:
#
#   Step 1 — TBM baseline (strict-solid):
#     At P = (¼,¼,¼) the srs Bloch Hamiltonian has exact C₃ symmetry.
#     The 4 bands decompose as 2×trivial + ω + ω² under C₃.
#     The ω and ω² bands are degenerate: |E(ω)| = |E(ω²)| = λ₀ = √3.
#     TBM (tri-bimaximal mixing) gives θ₂₃ = arctan(1) = 45°.
#     Source: predictions/B_P_doubly_degenerate_h.py
#
#   Step 2 — σ_z = 0 theorem (strict-solid):
#     At P, the generation eigenstates satisfy |ω²⟩ = e^{iφ} conj(|ω⟩)
#     (a complex-conjugate pair). The dark sector perturbation δH is real
#     and symmetric (graph adjacency matrix). For any such δH:
#       ⟨ω|δH|ω⟩ = ⟨ω²|δH|ω²⟩  (proved below)
#     so the σ_z component vanishes in the generation subspace.
#     Proof: ⟨ω|δH|ω⟩ = Σ_{ab} ψ*_ω[a] δH[a,b] ψ_ω[b]
#                      = Σ_{ab} ψ_{ω²}[a] δH[a,b] ψ_ω[b]    (since conj(ψ_ω) = e^{-iφ} ψ_{ω²})
#            ⟨ω²|δH|ω²⟩ = Σ_{ab} ψ*_{ω²}[a] δH[a,b] ψ_{ω²}[b]
#                       = Σ_{ab} ψ_ω[b] δH[b,a] ψ_{ω²}[a]   (δH symmetric, swap a↔b)
#                       = conj(⟨ω|δH|ω⟩)*
#            Since diagonal elements of Hermitian operators are real:
#                       ⟨ω²|δH|ω²⟩ = ⟨ω|δH|ω⟩. QED.
#     Consequence: eigenvalue splitting is symmetric, ±ε, with no σ_z tilt.
#     Numerical confirmation: 10,000 Monte Carlo trials with random real
#     symmetric perturbations all give σ_z = 0 to machine precision.
#     Source: proofs/flavor/srs_theta23_sigma_x.py Parts D, F, H.
#
#   Step 3 — Dark coupling magnitude (adopted: I-Feshbach + dark-map Class 2):
#     The dark sector splits λ_μ and λ_τ symmetrically as:
#       λ_μ = λ₀(1 + α₁_full),   λ_τ = λ₀(1 − α₁_full)
#     where α₁_full = α₁_bare × tan²(arg h).
#
#     α₁_bare = (2/3)^8 is the NB walk survival probability over g−2 = 8
#     steps (Exponent Principle, feshbach_exponent_principle.py).
#
#     The factor tan²(arg h) = Im²(h)/Re²(h) = 5/3 is the Class 2 (mass²-
#     class) dark correction coefficient from dark_extraction_map.py.
#     It is exact algebra from h = (√3+i√5)/2:
#       tan²(arg h) = (√5/2)² / (√3/2)² = (5/4) / (3/4) = 5/3.
#
#     Adopted identification 1 (I-Feshbach): α₁_bare equals the physical
#     dark-sector coupling magnitude. Gap: explicit K_4-quotient Feshbach
#     matrix elements not yet computed at journal grade.
#     Reference: ../predictions/Feshbach_coupling_strength_derivation.md §9.
#
#     Adopted identification 2 (dark-map Class 2): θ₂₃ is a mixing angle
#     from mass-matrix diagonalization → diagonal under C₃ → Class 2
#     coefficient = tan²(arg h) = 5/3.
#     Reference: predictions/dark_extraction_map.py summary table.
#
#   Step 4 — Formula (strict-solid arithmetic given Step 3):
#     θ₂₃ = arctan(λ_μ / λ_τ) = arctan((1 + α₁_full) / (1 − α₁_full))
#     With α₁_full = 1280/19683:
#       numerator   = 19683 + 1280 = 20963
#       denominator = 19683 − 1280 = 18403
#       θ₂₃ = arctan(20963/18403) ≈ 48.72°

# --- INPUTS --------------------------------------------------
# symbol         | value              | status              | predictions/ file
# ---------------|--------------------|--------------------|-----------------------------
# k_star         | 3                  | [derived]           | predictions/k_star.py
# g_girth        | 10                 | [derived]           | predictions/g_girth.py
# h              | (√3+i√5)/2         | [derived]           | predictions/h_walker_eigenvalue.py
# alpha_1_bare   | (2/3)^8 = 256/6561 | [derived]           | predictions/alpha_1.py
# tan²(arg h)    | 5/3                | [derived, algebra]  | predictions/dark_extraction_map.py
# alpha_1_full   | (5/3)×(2/3)^8      | [derived arithmetic]| inline
# TBM baseline   | 45°                | [derived]           | predictions/B_P_doubly_degenerate_h.py
# dark Class 2   | coefficient 5/3    | [TAXONOMY gap]      | predictions/dark_extraction_map.py
# I-Feshbach     | α₁_bare = coupling | [axiom A5(b)]       | docs/framework/framework_axioms.md §5b
#
# UPDATE 2026-04-19 session 2: I-Feshbach now subsumed by A5(b) (the
# coupling clause of A5; docs/framework/framework_axioms.md §5b). The remaining
# adoption is the dark-map Class 2 classification — asserting that θ₂₃
# (a mass-matrix mixing angle) belongs to the diagonal-C₃ representation
# class. The 5/3 number itself is rigorous algebra from h; only the
# observable-to-class assignment is currently asserted by inspection,
# not derived from C₃ representation theory at journal grade.

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_1 import predict_alpha_1
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from srs_E_at_P import predict_srs_E_at_P
from dark_extraction_map import dark_coefficient_mass_squared
from p_toggle import predict_p_toggle
import functools

d = predict_d_spatial()
k = predict_k_star(d)
E = predict_srs_E_at_P(k)
p = predict_p_toggle()
h = predict_h_walker_eigenvalue(k, E, p)
g = predict_g_girth(k, d)

alpha_1_bare = predict_alpha_1(k, g)            # (2/3)^8
c_mass       = dark_coefficient_mass_squared(h)  # tan²(arg h) = 5/3  [strict-solid algebra]
alpha_1_full = c_mass * alpha_1_bare             # (5/3)(2/3)^8 = 1280/19683

# σ_z=0 theorem: eigenvalue splitting is ±α₁_full × λ₀ (symmetric)
# θ₂₃ = arctan(λ_μ / λ_τ) = arctan((1 + α₁_full) / (1 − α₁_full))
theta_23_rad = math.atan((1 + alpha_1_full) / (1 - alpha_1_full))
theta_23_deg = math.degrees(theta_23_rad)

print(f"k* = {k}, g = {g}")
print(f"h = ({h.real:.6f} + i{h.imag:.6f})")
print(f"α₁_bare  = ({k-1}/{k})^{g-2} = {alpha_1_bare:.10f}")
print(f"tan²(arg h) = Im²(h)/Re²(h) = {c_mass:.10f}  (= 5/3 exactly)")
print(f"α₁_full  = (5/3) × α₁_bare  = {alpha_1_full:.10f}  (= 1280/19683 exactly)")
print()
print(f"θ₂₃ = arctan((1 + α₁_full)/(1 − α₁_full))")
print(f"    = arctan({1+alpha_1_full:.10f} / {1-alpha_1_full:.10f})")
print(f"    = arctan({(1+alpha_1_full)/(1-alpha_1_full):.10f})")
print(f"    = {theta_23_deg:.6f}°")
print()
print(f"TBM baseline: 45.000°   (C₃ symmetry at P, B_P_doubly_degenerate_h.py)")
print(f"Dark shift:   {theta_23_deg - 45:.3f}°   (σ_z=0 theorem + Class 2 coupling)")
print()

obs = 49.2
err = 1.3
dev_abs = theta_23_deg - obs
dev_sigma = dev_abs / err

# Runner-facing canonical aliases (slug = "theta_23_PMNS"); aliases only.
theta_23_PMNS_pred  = theta_23_deg
theta_23_PMNS_obs   = obs
theta_23_PMNS_sigma = err

print(f"Observed (PDG 2024): {obs:.1f}° ± {err:.1f}°")
print(f"Predicted:           {theta_23_deg:.2f}°")
print(f"Deviation:           {dev_abs:+.2f}°  ({dev_sigma:+.2f}σ)")
print()
print("Status: STRICT-SOLID THEOREM-GRADE (Row P13, graduated 2026-04-28)")
print("  [A5(b)]        α₁_bare = physical dark coupling magnitude")
print("                 — closed under A5(b) (docs/framework/framework_axioms.md §5b)")
print("  [dark-map C2]  θ₂₃ is Class 2 (mass²-class, diagonal C₃)")
print("                 — CLOSED 2026-04-28 via theorem_dark_map_class2_closure.md Theorem 5.1")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_theta_23_PMNS(alpha_1_bare, h):
    """
    Computes the PMNS atmospheric mixing angle theta_23.

    Derivation: TBM baseline 45° from C₃ symmetry at P-point of srs.
    σ_z=0 theorem (conjugate generation eigenstates + real perturbation)
    forces symmetric eigenvalue splitting ±α₁_full × λ₀. The mixing
    angle is arctan of the eigenvalue ratio.

    Formula: θ₂₃ = arctan((1 + α₁_full)/(1 − α₁_full))
    where α₁_full = tan²(arg h) × α₁_bare = Im²(h)/Re²(h) × α₁_bare.

    Identifications (STRICT-SOLID THEOREM-GRADE post-2026-04-28):
      - I-Feshbach: α₁_bare equals the physical dark coupling strength
        (closed under A5(b), 2026-04-19).
      - dark-map Class 2: θ₂₃ belongs to the mass²-class dark correction
        regime, giving coefficient tan²(arg h) = 5/3 (closed via
        `docs/theorems/theorem_dark_map_class2_closure.md` Theorem 5.1,
        2026-04-28).

    Parameters
    ----------
    alpha_1_bare : float
        Bare NB walk survival (2/3)^8 from predict_alpha_1.
    h : complex
        Walker eigenvalue (√3+i√5)/2 from predict_h_walker_eigenvalue.

    Returns
    -------
    float
        Predicted θ₂₃ in degrees.
    """
    import math
    c_mass       = h.imag**2 / h.real**2          # tan²(arg h) = 5/3
    alpha_1_full = c_mass * alpha_1_bare           # (5/3)(2/3)^8
    return math.degrees(math.atan((1 + alpha_1_full) / (1 - alpha_1_full)))


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = theta_23_deg
    pure_result = predict_theta_23_PMNS(alpha_1_bare, h)

    print(f"\nImplementation: {impl_result:.10f}°")
    print(f"Pure function:  {pure_result:.10f}°")
    assert abs(impl_result - pure_result) < 1e-8, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")

    obs = 49.2
    err = 1.3
    dev_abs  = pure_result - obs
    dev_sigma = dev_abs / err
    print(f"    θ₂₃ predicted = {pure_result:.4f}°")
    print(f"    PDG 2024      = {obs:.1f}° ± {err:.1f}°")
    print(f"    Deviation     = {dev_abs:+.4f}°  ({dev_sigma:+.2f}σ)")
    print(f"    Rigor status: STRICT-SOLID THEOREM-GRADE (Row P13 graduated")
    print(f"                  2026-04-28 via theorem_dark_map_class2_closure.md)")
