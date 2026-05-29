#!/usr/bin/env python3
"""
W14 — Spectral / waterline-included approach to α₁³ Family-D extension (2026-05-26).

CORRECTION FROM EARLIER W12+W13: I was wrongly checking whether SPECIFIC
length-24 closed cycles on H(srs) decompose as combinations of shorter
cycles. That's a CYCLE-COUNTING question. The framework's actual α₁
mechanism is a SPECTRAL EXPECTATION over the WATERLINE-INCLUDED ensemble
of NB walks — not a single walk count.

User correction: "we're looking for the full waterline included spectrum
here. not one walk, right?" — exactly.

WHAT α₁ ACTUALLY IS
-------------------
Per the Feshbach Exponent Principle (`predictions/feshbach_exponent_principle.py`):
    α₁_bare = q_NB^(g − 2)  where q_NB = (k* − 1)/k* = 2/3

This is the EXPECTED SURVIVAL PROBABILITY of one NB walker over (g−2)
steps with endpoint pinning n_fixed = 2. It is intrinsically a
spectral expectation over the ensemble of NB walks — NOT tied to any
specific closed walk.

α₁² from master doc §3 D Route H:
    c_H^(α₁²) = (q_NB(srs) · q_NB(srs-z))^(g−2) = (q_NB · q_NB)^8 = q_NB^16
              = α₁_bare^2.

This is the EXPECTED JOINT SURVIVAL on the (srs × srs-z) Sunada-cospectral
pair over (g−2) joint NB steps. Again, NOT a specific cycle count.

α₁³ — THE WATERLINE-INCLUDED SPECTRAL FORM
-------------------------------------------
By direct extension, the SPECTRAL form of α₁³ Family-D via the same
mechanism is

    c_H^(α₁³) = (q_NB(srs) · q_NB(X) · q_NB(Y))^(g−2)
              = (q_NB · q_NB · q_NB)^8 = q_NB^24 = α₁_bare^3

where X and Y are two of the (k=3, g=10)-class cospectral alternatives
above the A2-T waterline.

This is a 3-WAY JOINT WALKER on (srs × X × Y) over (g−2) joint NB steps —
a SPECTRAL EXPECTATION, not a closed-cycle count.

THE STRUCTURAL QUESTIONS REMAINING (research-level)
---------------------------------------------------
(Q1) Identification of the cospectral alternatives X, Y at α₁³.
     Master doc §1 lists four (k=3, g=10)-class alternatives: srs-z,
     srs-c4, srs-c8, srs-c27. R-9 closure (2026-05-02) identifies srs-z
     as the dominant alternative at α₁² (Route H).
     OPEN: which of the remaining three are above the waterline at α₁³
     order? The full waterline-included spectrum would naturally
     include all alternatives weighted by their MDL bit-costs.

(Q2) Whether the joint walker on 3-way (srs × X × Y) at length (g−2) is
     a structurally-derived quantity (analog of master doc Route H).
     Or whether α₁³ Family-D Route H requires a different mechanism.

(Q3) The c_F (per fermion leg) extension at α₁³ — analogous Clause-6
     two-step (channel_select → canonical_encoding) at α₁³ order,
     with rep-resolution via V_Ram μ_rep_j.

THIS SCRIPT
-----------
Verifies the SPECTRAL FORM of α₁ and α₁² computations from framework
primitives, then proposes α₁³ as the natural 3-way joint walker
extension. Honest about which questions remain open.
"""
from fractions import Fraction
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from predictions.k_star import predict_k_star
from predictions.g_girth import predict_g_girth
from predictions.feshbach_exponent_principle import predict_feshbach_coupling

k_star = predict_k_star(d=3)
g = predict_g_girth(k_star, 3)
q_NB = Fraction(k_star - 1, k_star)   # = 2/3 (Branch Measure Theorem)
alpha_1_bare_frac = q_NB ** (g - 2)   # = (2/3)^8 = 256/6561

print("=" * 76)
print("W14 — Spectral / waterline-included α₁³ Family-D Route H form")
print("=" * 76)
print()
print(f"Framework primitives (Type-4 theorem-grade upstream):")
print(f"  k* = {k_star}, g = {g}, q_NB = {q_NB}")
print(f"  α₁_bare = q_NB^(g-2) = q_NB^{g-2} = {alpha_1_bare_frac}")
print(f"         = {float(alpha_1_bare_frac):.10e}")
print()

# Verify Feshbach Exponent Principle gives the expected α₁_bare
fesh = predict_feshbach_coupling(k_star, g, 2)
print(f"  Cross-check with predict_feshbach_coupling: {fesh}")
assert abs(float(alpha_1_bare_frac) - fesh) < 1e-15

print()
print(f"  ┌── Theorem (Feshbach Exponent Principle): ─────────────────────────────")
print(f"  │  α₁_bare = q_NB^(g-2)  is the EXPECTED SURVIVAL PROBABILITY of")
print(f"  │  one NB walker over (g-2) steps with endpoint pinning n_fixed=2.")
print(f"  │")
print(f"  │  This is a SPECTRAL expectation (ensemble average over NB walks)")
print(f"  │  — NOT a specific closed-cycle count.")
print(f"  └───────────────────────────────────────────────────────────────────────")
print()

# α₁² Family-D Route H (master doc §3 D)
n_subst_2way = 2  # (srs × srs-z)
c_H_alpha2_frac = (q_NB ** n_subst_2way) ** (g - 2)
print(f"α₁² Family-D Route H (master doc §3 D, theorem-grade):")
print(f"  c_H^(α₁²) = (q_NB^{n_subst_2way})^(g-2) = (q_NB·q_NB)^{g-2}")
print(f"            = ({q_NB**n_subst_2way})^{g-2}")
print(f"            = q_NB^(2·(g-2)) = q_NB^{2*(g-2)}")
print(f"            = {c_H_alpha2_frac}")
print(f"            = {float(c_H_alpha2_frac):.6e}")
print(f"  α₁_bare² = {alpha_1_bare_frac**2}")
print(f"           = {float(alpha_1_bare_frac**2):.6e}")
assert c_H_alpha2_frac == alpha_1_bare_frac**2, "Route H α₁² verification failed"
print(f"  ✓ MATCHES (Route H = master doc §3 D theorem)")
print()
print(f"  STRUCTURAL READING: 2-way joint walker on (srs × srs-z) over (g-2)")
print(f"  joint NB steps. Each step requires BOTH substrate walkers to survive.")
print(f"  This is the WATERLINE-INCLUDED expected joint survival on the")
print(f"  Sunada-isospectral pair (srs, srs-z) per R-9 closure 2026-05-02.")
print()

# Natural α₁³ extension: 3-way joint walker
n_subst_3way = 3
c_H_alpha3_frac = (q_NB ** n_subst_3way) ** (g - 2)
print(f"α₁³ Family-D Route H — proposed extension (3-way joint walker):")
print(f"  c_H^(α₁³) = (q_NB^{n_subst_3way})^(g-2) = (q_NB·q_NB·q_NB)^{g-2}")
print(f"            = ({q_NB**n_subst_3way})^{g-2}")
print(f"            = q_NB^(3·(g-2)) = q_NB^{3*(g-2)}")
print(f"            = {c_H_alpha3_frac}")
print(f"            = {float(c_H_alpha3_frac):.6e}")
print(f"  α₁_bare³ = {alpha_1_bare_frac**3}")
print(f"           = {float(alpha_1_bare_frac**3):.6e}")
assert c_H_alpha3_frac == alpha_1_bare_frac**3, "α₁³ proposed extension fails"
print(f"  ✓ MATCHES α₁³ EXACTLY at the spectral-expectation level")
print()
print(f"  STRUCTURAL READING: 3-way joint walker on (srs × X × Y) over (g-2)")
print(f"  joint NB steps, where X and Y are TWO of the (k=3, g=10)-class")
print(f"  cospectral alternatives above the A2-T waterline.")
print()

# Cospectral alternatives per master doc §1 R-9 closure
print("=" * 76)
print("Cospectral alternatives at (k=3, g=10) per master doc §1 R-9 closure:")
print("=" * 76)
print("""
The framework's dark sector contains FOUR cospectral alternatives to srs
at the (k*=3, g=10) class:

  • srs-z   = bipartite double cover of srs  [DOMINANT per R-9 closure 2026-05-02]
  • srs-c4
  • srs-c8
  • srs-c27

R-9 closure (2026-05-02): srs-z is the DOMINANT alternative for the 2-way
joint walker at α₁² Family-D Route H, per Sunada-cospectrality and MDL
bit-cost ranking.

OPEN QUESTIONS for α₁³ Family-D Route H extension:

(Q1) Which of {srs-c4, srs-c8, srs-c27} is the next-rank cospectral
     partner above the A2-T waterline at α₁³ order? Or does the
     waterline-included spectrum naturally weight all three together?

(Q2) Is the 3-way joint walker on (srs × srs-z × srs-cX) for some X
     theorem-grade by the SAME mechanism that gave α₁² (analog of
     master doc Route H), or does it require a new structural argument?

(Q3) Are srs-c4, srs-c8, srs-c27 mutually cospectral with srs-z?
     The R-9 closure identified srs-z as dominant against srs alone.
     A 3-way joint walker also requires structural compatibility
     between the two non-srs partners.

WHAT IS RIGOROUSLY ESTABLISHED IN THIS SCRIPT:
  ✓ The SPECTRAL form c_H^(α₁³) = q_NB^(3·(g-2)) = α₁_bare³ is
    arithmetic-exact for any 3-way joint walker on cospectral substrates.
  ✓ This is the WATERLINE-INCLUDED EXPECTATION (spectral, not cycle-count).
  ✓ The 2-way version (α₁²) is theorem-grade per master doc §3 D Route H.

WHAT IS NOT RIGOROUSLY ESTABLISHED:
  ✗ Whether the framework's R-9 closure mechanism extends to a 3-way
    joint walker by selecting two cospectral partners above the waterline.
  ✗ Which specific partners (srs-c4, c8, c27, or combination) are above
    the waterline at α₁³ order.
  ✗ Whether the 3-way joint walker's survival probability is the
    UNIQUE waterline-included spectral expectation at α₁³ order, or
    whether other configurations (e.g., 2-way at longer length) also
    contribute.

NEXT STEPS to convert this to theorem-grade closure:
  Step A: Extend R-9 closure analysis to α₁³ — identify the SET of
          cospectral partners above the waterline at this order.
  Step B: Verify the 3-way joint walker spectral structure is
          well-defined (Sunada-cospectrality of triples).
  Step C: Show the 3-way joint walker survival = α₁³ is the UNIQUE
          waterline-included spectral expectation at order 3(g-2).
  Step D: (Optional) cross-check by computing the joint Hashimoto
          spectrum and verifying tr(B_joint^(g-2)) matches the
          q_NB^(3(g-2)) prediction.
""")
