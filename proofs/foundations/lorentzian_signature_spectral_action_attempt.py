#!/usr/bin/env python3
"""
Attempt: derive Lorentzian signature (-,+,+,+) via Connes spectral action on the
substrate.

Setup (already theorem-grade upstream):
    A     = L(F_inv(E))   — type II_1 factor (forward_construction_substrate_modular_structure.md)
    H     = L^2(F_inv(E)) ⊗ S    — substrate Hilbert space ⊗ Cl(6;ℂ) spinor
    D_sub = Σ_e γ^e ⊗ L_e        — substrate Dirac
    D²_sub = n·I + R_sub          — Lichnerowicz formula (forward_construction_substrate_lichnerowicz.md)
    n     = |E| = 6
    τ(R_sub)    = 0  (mean zero)
    τ(R_sub^2)  = n(n-1) = 30  (HS norm)

Standard Connes-Chamseddine (1996) for smooth d-dim manifolds:
    S(D, Λ) = Tr f(D²/Λ²) ~ f₄ Λ⁴ a₀ + f₂ Λ² a₂ + f₀ a₄ + O(Λ^{-2})
where the heat-kernel expansion τ(e^{-tD²}) ~ Σ_k a_k t^{(k-d)/2} as t → 0+ has UV
divergences as t → 0 because D² has unbounded spectrum on a smooth manifold.

This script computes the substrate analog and asks: does the Connes-Chamseddine
expansion give an Einstein-Hilbert action + signature for the substrate?

OUTCOME: NO. The substrate's D² is BOUNDED (spectrum in [0, n + ‖R_sub‖]).
Therefore τ(e^{-tD²}) is BOUNDED as t → 0+, with no UV divergences. The standard
Λ⁴ + Λ² + Λ⁰ asymptotic structure of Connes-Chamseddine does NOT apply. The
substrate's spectral action is a smooth Taylor expansion in 1/Λ², not a sum of
distinct positive powers of Λ. There is no Einstein-Hilbert term to read off,
and hence no Riemannian metric structure pulled out from the spectral action in
the standard way.

This is a qualitative obstruction, not a coefficient-fitting issue. The framework's
substrate, being finite/discrete (bounded D²), is structurally different from the
smooth-manifold setting where Connes-Chamseddine machinery was originally developed.

CONSEQUENCE: the Lorentzian signature derivation via Connes spectral action route
(handoff_substrate_spectral_action.md, Route C) does not directly close. The
substrate has rigorous non-commutative-geometric structure (Lichnerowicz, R_sub,
type II_1 factor) but does NOT inherit the smooth-manifold signature derivation
machinery without additional construction (Krein-space spectral triples for
Lorentzian NCG, or modified spectral action for finite triples).

The Lorentzian signature scoping note's BLOCKED finding stands; this script
identifies the specific structural obstacle to Route C closure.
"""

from sympy import Symbol, Rational, factorial, exp, oo, integrate, series
from sympy import Function, simplify, expand, collect, latex, sympify

# ============================================================================
# Substrate inputs (Type 4 from upstream)
# ============================================================================

n = 6  # |E| for srs

# Trace moments of R_sub (Lichnerowicz upstream)
tau_R_0 = 1                  # τ(I) = 1 (normalized type II_1 trace)
tau_R_1 = 0                  # τ(R_sub) = 0  (Lemma 3.2)
tau_R_2 = n * (n - 1)        # τ(R_sub²) = n(n-1) = 30  (Theorem 3.4)
# Higher moments τ(R_sub^k) for k ≥ 3 require explicit Cl(6) bivector + F_inv(E)
# word-length computations. For this analysis we only need k ≤ 2.

print("="*75)
print("Substrate spectral action — heat-kernel expansion attempt")
print("="*75)
print(f"\nUpstream inputs (theorem-grade):")
print(f"  n = |E| = {n}")
print(f"  τ(I)        = {tau_R_0}")
print(f"  τ(R_sub)    = {tau_R_1}        [Lichnerowicz Lemma 3.2: mean zero]")
print(f"  τ(R_sub²)   = {tau_R_2}        [Lichnerowicz Theorem 3.4: HS norm]")

# ============================================================================
# Heat kernel τ(e^{-tD²}) — Type 2 algebra, exact
# ============================================================================
#
# D²_sub = n·I + R_sub.  Since [n·I, R_sub] = 0:
#   exp(-t D²) = exp(-tn) · exp(-t R_sub)
# Taking τ:
#   τ(exp(-tD²)) = exp(-tn) · τ(exp(-t R_sub))
# Expand:
#   τ(exp(-t R_sub)) = Σ_k (-t)^k/k! · τ(R_sub^k)
#                    = 1 - t·0 + t²/2 · n(n-1) + O(t³)
# So:
#   τ(exp(-tD²)) = exp(-tn) · [1 + t² · n(n-1)/2 + O(t³)]

t, Lam = Symbol('t', positive=True), Symbol('Lambda', positive=True)

# τ(exp(-t R_sub)) up to order t^2:
tau_exp_minus_tR = 1 + 0*t + (t**2/2) * tau_R_2  # higher orders truncated

# τ(exp(-t D²)) = exp(-tn) * τ(exp(-tR))
tau_exp_minus_tDsq = exp(-t*n) * tau_exp_minus_tR

print(f"\nτ(exp(-t D²_sub)) = exp(-{n}t) · [1 + {tau_R_2}·t²/2 + O(t³)]")
print(f"                   = {tau_exp_minus_tDsq}")

# ============================================================================
# Asymptotic in t → 0+: ARE THERE UV DIVERGENCES?
# ============================================================================

print(f"\nAsymptotic of τ(exp(-tD²)) at t → 0+:")
limit_at_zero = tau_exp_minus_tDsq.subs(t, 0)
print(f"  τ(exp(-tD²))|_{{t=0}} = {limit_at_zero}")
print(f"  (FINITE — no UV divergence as t → 0+)")

# Power series at t = 0:
print(f"\n  Power series at t=0 (first 4 terms):")
ts = series(tau_exp_minus_tDsq, t, 0, 4).removeO()
print(f"    {ts}")
print(f"  (smooth Taylor series — no t^{{-d/2}} singularity)")

# By contrast, smooth-manifold expectation for d=4:
print(f"\nFor a smooth 4-manifold (Connes-Chamseddine standard):")
print(f"  τ(exp(-tD²)) ~ a₀ · t^{{-2}} + a₂ · t^{{-1}} + a₄ + O(t)  as t → 0+")
print(f"  (UV divergent — the t^{{-2}} and t^{{-1}} terms produce Λ⁴ and Λ² ")
print(f"   coefficients of the spectral action in the cutoff Λ → ∞ limit)")

# ============================================================================
# Spectral action S(D, Λ) = τ(f(D²/Λ²)) — smooth Taylor, not Λ⁴+Λ²+Λ⁰
# ============================================================================
#
# Since D² is bounded and τ(exp(-tD²)) is smooth at t=0, the spectral action
# as Λ → ∞ has a smooth Taylor expansion in 1/Λ², NOT a Laurent expansion
# with positive powers of Λ.
#
# Concretely, for ANY smooth cutoff function f(x) (with f(0) defined):
#   τ(f(D²/Λ²)) = f(0) + f'(0)·τ(D²)/Λ² + f''(0)/2 · τ(D⁴)/Λ⁴ + O(Λ^{-6})
#
# Compute the moments τ(D^{2k}) of D² = n·I + R_sub:
#   τ(D²)  = τ(n·I + R) = n·1 + 0 = n
#   τ(D⁴)  = τ((n·I + R)²) = n² + 2n·τ(R) + τ(R²) = n² + 0 + n(n-1) = 2n² - n

tau_Dsq_1 = n + tau_R_1                     # τ(D²)  = n
tau_Dsq_2 = n**2 + 2*n*tau_R_1 + tau_R_2    # τ(D⁴)  = 2n² - n

print(f"\nSubstrate spectral action moments:")
print(f"  τ(D²)  = n + τ(R)         = {n} + 0          = {tau_Dsq_1}")
print(f"  τ(D⁴)  = n² + 2n·τ(R) + τ(R²) = {n**2} + 0 + {tau_R_2} = {tau_Dsq_2}")

print(f"""
Substrate spectral action:
  τ(f(D²/Λ²)) = f(0) + f'(0) · {tau_Dsq_1}/Λ² + f''(0)/2 · {tau_Dsq_2}/Λ⁴ + O(Λ⁻⁶)

Compare to smooth-manifold Connes-Chamseddine (d=4):
  Tr(f(D²/Λ²)) = f₄ Λ⁴ a₀ + f₂ Λ² a₂ + f₀ a₄ + O(Λ⁻²)

The substrate's expansion is in 1/Λ²; the smooth-manifold's expansion has positive
powers of Λ. NO Λ² Einstein-Hilbert coefficient emerges in the substrate case.
""")

# ============================================================================
# CONCLUSION
# ============================================================================

print("="*75)
print("FINDING")
print("="*75)
print("""
The standard Connes-Chamseddine machinery (Λ⁴ cosmological constant + Λ²
Einstein-Hilbert + Λ⁰ Weyl² + Riemann²) does NOT apply directly to the substrate.

Reason: the substrate's D²_sub = n·I + R_sub is a BOUNDED operator. The heat
kernel τ(e^{-tD²}) is therefore SMOOTH at t = 0 (no UV divergence), and the
spectral action τ(f(D²/Λ²)) has a SMOOTH Taylor expansion in 1/Λ², not the
Laurent expansion in positive powers of Λ that is the hallmark of smooth-manifold
Connes-Chamseddine.

CONSEQUENCE FOR LORENTZIAN SIGNATURE DERIVATION:

The Connes spectral action route (Route C in
an internal working note, taken from
an internal note) does not directly produce a
Riemannian Einstein-Hilbert action from which the spatial signature could be
read off. The substrate has rigorous discrete-NCG curvature structure
(R_sub, Lichnerowicz) but does NOT inherit the smooth-manifold spectral
action machinery.

The Lorentzian signature derivation remains BLOCKED at parameter_linter
hard-quality gate. The Connes route's failure mode is now precisely identified:

    BLOCKING STEP. Substrate spectral action lacks the Λ²-Einstein-Hilbert
    coefficient because D² is bounded; standard Connes-Chamseddine
    machinery presupposes unbounded D² (smooth manifold) for its UV-asymptotic
    expansion in positive powers of Λ to be defined.

WHAT IS ESTABLISHED at theorem grade:
    1. Substrate spectral triple (A, H, D_sub) is Connes-regular.
    2. D²_sub = n·I + R_sub with τ(R) = 0, τ(R²) = n(n-1).
    3. Spectral action exists as smooth function of 1/Λ², with
       τ(f(D²/Λ²)) = f(0) + n·f'(0)/Λ² + (2n²-n)·f''(0)/(2Λ⁴) + O(Λ⁻⁶).

WHAT IS NOT CLOSED at theorem grade:
    - Riemannian (spatial) metric from spectral-action Einstein-Hilbert term.
    - Lorentzian (3+1) signature from Wick rotation of spatial Riemannian.
    - Both require either: (a) Krein-space spectral triple construction for
      Lorentzian NCG (Besnard-Bizi-Iochum 2017+, research-level), (b) a
      modified Connes-Chamseddine machinery for finite/bounded-D² spectral
      triples (no published version on file), or (c) the alternative routes
      from the scoping note (BLMS causal-set, Dirac-point linear dispersion).

NET RESULT:

Going through the route concretely sharpened the BLOCKED diagnosis: the
substrate's spectral action exists and is computable, but does NOT carry the
Einstein-Hilbert + signature machinery of smooth-manifold NCG. This is a
structural feature of finite/discrete spectral triples, NOT a missing
coefficient that can be filled in with another session of work. Closure of
Route C requires research-level NCG development beyond the framework's current
state.
""")
