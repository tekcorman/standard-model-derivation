#!/usr/bin/env python3
"""(L) Ihara-map unification of Class A + Class B substrate constants.

The Ihara map u(λ) for a k-regular graph relates adjacency eigenvalues to
Hashimoto eigenvalues via the quadratic u² − λu + (k−1) = 0:

    u(λ) = (λ + √(λ² − 4(k−1))) / 2

For Perron eigenvalue λ = k:
  u(k) = (k + √(k² − 4(k−1))) / 2 = (k + (k−2)) / 2 = k − 1

  This is the Perron NB eigenvalue λ_B = k − 1.

Derivative:
  u'(λ) = (1 + λ/√(λ² − 4(k−1))) / 2

At λ = k:
  u'(k) = (1 + k/(k−2)) / 2 = (2k−2)/(2(k−2)) = (k−1)/(k−2)

Hmm wait, let me recompute. For k = 3, k-1 = 2, so:
  u'(3) = (1 + 3/√(9-8))/2 = (1 + 3/1)/2 = 2

Discrepancy: my closed-form (k-1)/(k-2) at k=3 gives 2/1 = 2 ✓. OK same.

For k=3: u(3) = 2, u'(3) = 2. Both equal k-1.

This is the key fact: at the Perron, u(λ) and u'(λ) are equal (both = k-1).
The Ihara map self-fixes at the Perron eigenvalue.
"""
from __future__ import annotations
import math
from fractions import Fraction
import sympy as sp

print("=" * 90)
print("(L) Ihara-map unification of Class A + Class B")
print("=" * 90)

# Symbolic Ihara map
lam = sp.Symbol('lambda', real=True)
k = sp.Symbol('k', positive=True, integer=True)

u_plus = (lam + sp.sqrt(lam**2 - 4*(k - 1))) / 2
u_minus = (lam - sp.sqrt(lam**2 - 4*(k - 1))) / 2

print("\n  Ihara map: u² − λu + (k−1) = 0")
print(f"    u₊(λ, k) = {u_plus}")
print(f"    u₋(λ, k) = {u_minus}")
print(f"    Vieta: u₊ + u₋ = λ,  u₊·u₋ = k−1")

# Derivative
u_prime = sp.diff(u_plus, lam)
u_prime_at_k = sp.simplify(u_prime.subs(lam, k))
u_prime_pp = sp.diff(u_prime, lam)
u_pp_at_k = sp.simplify(u_prime_pp.subs(lam, k))

print(f"\n  Derivatives at Perron (λ = k):")
print(f"    u(k)    = {sp.simplify(u_plus.subs(lam, k))}")
print(f"    u'(k)   = {u_prime_at_k}")
print(f"    u''(k)  = {u_pp_at_k}")

# At k = 3 (srs)
print(f"\n  At srs (k = 3):")
print(f"    u(3)   = {u_plus.subs([(lam, 3), (k, 3)])}     [= λ_B Perron]")
print(f"    u'(3)  = {u_prime.subs([(lam, 3), (k, 3)])}     [Ihara link factor]")
print(f"    u''(3) = {u_prime_pp.subs([(lam, 3), (k, 3)])}    [2nd-order link factor]")

# Class A connection: q_NB
print(f"\n{'='*90}")
print(f"Class A constants from Ihara at Perron:")
print(f"{'='*90}")
print(f"""
  q_NB = u(k) / k = (k−1) / k = 2/3 for k=3

  At srs:
    Numerator  = u(k) = k − 1 = λ_B Perron = 2
    Denominator = k = λ_A Perron = 3
    Ratio q_NB = 2/3 ✓

  This is the same identification as Row 23: q_NB is the Perron ratio,
  AND q_NB · k = u(k) = k − 1 (Ihara map at Perron).

  ε_CP = (k − u(k)) / (k + u(k)) = (k − (k−1)) / (2k − 1) = 1 / (2k−1)

  At srs k=3: ε_CP = 1/5.
""")

# Class B connection: D_NB / D_H from Ihara derivative
print(f"\n{'='*90}")
print(f"Class B constants from Ihara DERIVATIVE at Perron:")
print(f"{'='*90}")
print(f"""
  Ihara cross-walker theorem (lorentz_sig_ihara_lv_relation.py):

    D_NB           = u'(k) · D_H                      = 2 · D_H
    D4_aniso^NB    = u'(k) · D4_aniso^H               = 2 · D4_aniso^H
    D4_iso^NB      = u'(k) · D4_iso^H + (1/2) u''(k) · D_H²
                   = 2 · D4_iso^H − 2 · D_H²        [u''(3) = -4 from above]

  Substituting srs values:
    D_H = 1/16, D4_iso^H = -1/1024, D4_aniso^H = +1/1536

    D_NB        = 2 · (1/16)             = 1/8
    D4_aniso^NB = 2 · (1/1536)           = 1/768
    D4_iso^NB   = 2·(-1/1024) − 2·(1/256) = -1/512 − 1/128 = -5/512

  All match the framework's symbolically-verified values.

  η_NB := D4_aniso^NB / D_NB² = (1/768)/(1/8)² = 64/768 = 1/12
  η^H_NB := D4_aniso^H / D_H² = (1/1536)/(1/16)² = 256/1536 = 1/6
  Ratio:  η_NB / η^H_NB = (1/12)/(1/6) = 1/2

  This 1/2 is exactly 1/u'(k)² ... no wait, 1/u'(3) = 1/2.
  Specifically:  η_NB / η^H_NB = 1 / u'(k) = 1/2 for k=3.

  → The Ihara DERIVATIVE u'(k) is the Class A → Class B link factor.
""")

# Headline: the Ihara map unifies Classes A and B
print(f"\n{'='*90}")
print(f"UNIFIED PICTURE: Ihara map u(λ) connects Class A and Class B")
print(f"{'='*90}")
print(f"""
  Class A constants are FUNCTIONS of the value u(λ_Perron) = k−1:
    q_NB     = u(k)/k = (k−1)/k = 2/3
    α_1_bare = (u(k)/k)^(g-2) = (2/3)^8
    ε_CP     = (k−u(k))/(k+u(k)) = 1/(2k−1) = 1/5
    c        = (2(|E|−|V|)+1)/(2|E|) = 5/12  [structural, separate]

  Class B constants are FUNCTIONS of the GRADIENT u'(λ_Perron) = k−1:
    D_NB           = u'(k) · D_H = 2 D_H
    D4_aniso^NB    = u'(k) · D4_aniso^H = 2 D4_aniso^H
    η_NB / η^H_NB  = 1/u'(k) = 1/2

  Coincidence at k = 3: u(3) = u'(3) = 2 = k − 1.

  This means: at the Perron eigenvalue, the Ihara map's value AND its
  gradient are equal. The same number (k − 1 = 2 for srs) encodes both:
    - Class A "value" relations (q_NB, α_1, etc.)
    - Class B "gradient" relations (D_NB / D_H = u'(k) = 2)

  The framework's substrate has u(k) = u'(k) at its Perron eigenvalue —
  a HIGHLY non-generic spectral coincidence for k-regular graphs.

  General formula: u(k) = k - 1 (always)
                   u'(k) = (k-1)/(k-2)  [actually (k-1)/(k-2) for k > 2]

  Wait — let me recompute u'(k) symbolically for general k:
""")

# Recompute
u_prime_subs = u_prime.subs(lam, k)
u_prime_simplified = sp.simplify(u_prime_subs)
print(f"  u'(λ=k, k) = {u_prime_simplified}")

# Try specific k values
print(f"\n  u'(λ=k) for various k:")
for k_val in [2, 3, 4, 5, 6]:
    u_val = sp.simplify(u_plus.subs([(lam, k_val), (k, k_val)]))
    up_val = sp.simplify(u_prime.subs([(lam, k_val), (k, k_val)]))
    equal = '✓' if u_val == up_val else ''
    print(f"    k={k_val}: u(k) = {u_val}, u'(k) = {up_val}  {equal}")

# Now check which k satisfies u(k) = u'(k)
print(f"""
  Looking for k where u(k) = u'(k):
    u(k) = k − 1
    u'(k) = ?  Compute:
""")

# Symbolic
print(f"  Setting u(k) = u'(k), solve for k:")
eq = sp.Eq(sp.simplify(u_plus.subs(lam, k)), sp.simplify(u_prime.subs(lam, k)))
print(f"    {eq}")
sol = sp.solve(eq, k)
print(f"    Solutions for k: {sol}")

print(f"""
  Conclusion: u(k) = u'(k) holds for SPECIFIC k values. For srs k=3,
  this is satisfied (both = 2). This means srs's k* = 3 is structurally
  *special* in the Ihara map's geometry — the value/derivative coincidence
  at the Perron is exactly at k=3.

  This may be why the framework's Class A and Class B identifications
  are particularly clean for srs (k=3): the Ihara map's two informational
  channels (value u and gradient u') merge at the Perron.

  For other coordination numbers, u(k) and u'(k) would differ → Class A
  and Class B identifications would have different structural content,
  not the unified picture seen for k=3.
""")

# Test alternative cells
print(f"\n  For comparison, test k = 2 (path graph), k = 4 (4-regular):")
for k_val in [2, 4, 5]:
    u_val = sp.simplify(u_plus.subs([(lam, k_val), (k, k_val)]))
    up_val = sp.simplify(u_prime.subs([(lam, k_val), (k, k_val)]))
    print(f"    k={k_val}: u(k) = {u_val}, u'(k) = {up_val}  diff = {sp.simplify(u_val - up_val)}")
