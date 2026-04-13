#!/usr/bin/env python3
"""
srs_r_resolvent_theorem_v3.py — The Theorem.

KEY DISCOVERY from v2:
  R = 2/sin²(5φ) - 4 = 228/7   EXACTLY

where φ = arctan(√7), the spectral angle of the K₄ triplet.

The question: WHY n=5? And what is the MASS MATRIX?

ANSWER: The formula R = |h|²/sin²(nφ) - (k*+1) arises from the
CHEBYSHEV EXPANSION of the Green's function. Specifically:

  G_n(φ) = sin(nφ)/sin(φ)  (Chebyshev-U propagator of the second kind)

  sin²(nφ) = G_n² × sin²(φ)

  |h|²/(G_n² sin²φ) = |h|²/(G_n² × Im(h)²/|h|²) = |h|⁴/(G_n² Im(h)²)

For K₄: |h|² = 2, sin²φ = 7/8, and we need G_n = sin(nφ)/sin(φ).

The values G_n for the first several n:
  G₁ = 1
  G₂ = 2cos(φ) = 2/(2√2) = 1/√2
  G₃ = 4cos²φ - 1 = 4/8 - 1 = -1/2
  G₄ = 8cos³φ - 4cosφ = 8/(16√2) - 4/(2√2) = 1/(2√2) - 2/√2 = -3/(2√2) = -3√2/4
  G₅ = 16cos⁴φ - 12cos²φ + 1 = 16/64 - 12/8 + 1 = 1/4 - 3/2 + 1 = -1/4

So G₅ = -1/4 = -1/(k*+1).

R_n = |h|²/(G_n² sin²φ) - (k*+1) = 2/(G_n² × 7/8) - 4 = 16/(7G_n²) - 4

R₅ = 16/(7 × 1/16) - 4 = 256/7 - 4 = 228/7. ✓

But why n=5? THIS is the key.
"""

import math
import cmath
import numpy as np

sqrt7 = math.sqrt(7)
sqrt2 = math.sqrt(2)
sqrt21 = math.sqrt(21)
phi = math.atan(sqrt7)
h = (-1 + 1j * sqrt7) / 2
omega = cmath.exp(2j * cmath.pi / 3)

print("=" * 70)
print("THE R = 228/7 THEOREM")
print("=" * 70)
print()

# R_n = 16/(7 G_n²) - 4 where G_n = sin(nφ)/sin(φ), φ = arctan(√7)
print("Table of R_n = 16/(7G_n²) - 4:")
print(f"{'n':>4} {'G_n':>15} {'G_n²':>15} {'R_n':>15}")
print("-" * 55)

# Compute G_n via recurrence: G_1=1, G_2=2cosφ, G_{n+1}=2cosφ·G_n - G_{n-1}
cosφ = math.cos(phi)  # = 1/(2√2)
G = [0, 1, 2*cosφ]  # G[0] unused, G[1]=1, G[2]=2cosφ
for n in range(3, 20):
    G.append(2*cosφ*G[-1] - G[-2])

for n in range(1, 16):
    Gn = G[n]
    Gn2 = Gn**2
    if abs(Gn2) > 1e-30:
        Rn = 16/(7*Gn2) - 4
    else:
        Rn = float('inf')
    # Check if G_n is a nice fraction
    from fractions import Fraction
    gf = Fraction(Gn).limit_denominator(100)
    gf_str = f"{float(gf):.10f}" if abs(float(gf)-Gn) > 1e-8 else str(gf)
    mark = " <-- 228/7" if abs(Rn - 228/7) < 1e-6 else ""
    print(f"{n:4d} {Gn:>15.10f} {Gn2:>15.10f} {Rn:>15.8f}{mark}")

print()

# =====================================================================
# WHY n = 5?
# =====================================================================
print("=" * 70)
print("WHY n = 5?")
print("=" * 70)
print()

# Observation 1: G₅ = -1/(k*+1) = -1/4
# This is the UNIQUE n where |G_n| = 1/(k*+1).
# Is this true? Let me check.
print("Check: which n gives |G_n| = 1/(k*+1) = 1/4?")
while len(G) < 60:
    G.append(2*cosφ*G[-1] - G[-2])
for n in range(1, 50):
    if abs(abs(G[n]) - 0.25) < 1e-6:
        print(f"  n={n}: G_{n} = {G[n]:.15f}, |G_{n}| = 1/4 ✓")
print()

# Observation 2: 5 = number of edges of K₄ minus 1 = 6-1
# Or: 5 = |V|·(|V|-1)/2 - 1 where |V|=4
print(f"|E(K₄)| = 6, |E|-1 = 5")
print()

# Observation 3: The Ihara zeta function degree
# ζ_I(u)^{-1} = (1-u²)^{r-1} det(I - Au + qu²I)
# r = |E|-|V|+1 = 6-4+1 = 3 (cyclomatic number)
# det(I-Au+2u²I) is degree 2|V|=8 in u.
# Total degree of ζ_I^{-1}: 2(r-1) + 2|V| = 4+8 = 12 = 2|E|. ✓

# Observation 4: G₅ = -1/4 means sin(5φ) = -sin(φ)/4.
# Equivalently: 4sin(5φ) + sin(φ) = 0
# Using the expansion sin(5φ) = 16sin⁵φ - 20sin³φ + 5sinφ:
# 4(16s⁵ - 20s³ + 5s) + s = 0 where s = sinφ
# 64s⁵ - 80s³ + 20s + s = 0
# 64s⁵ - 80s³ + 21s = 0
# s(64s⁴ - 80s² + 21) = 0
# s≠0, so 64s⁴ - 80s² + 21 = 0
# With s² = 7/8: 64(7/8)² - 80(7/8) + 21 = 64·49/64 - 70 + 21 = 49 - 70 + 21 = 0 ✓

print("Verification: 64sin⁴φ - 80sin²φ + 21 = 0")
s2 = 7/8
print(f"  64×(7/8)² - 80×(7/8) + 21 = {64*s2**2 - 80*s2 + 21:.1f} ✓")
print()

# So G₅ = -1/4 is equivalent to the identity 64sin⁴φ - 80sin²φ + 21 = 0,
# which factors as (8sin²φ - 7)(8sin²φ - 3) = 0.
# sin²φ = 7/8 satisfies the first factor.
# The other solution sin²φ = 3/8 would give a DIFFERENT graph.

print("Factored: (8sin²φ - 7)(8sin²φ - 3) = 0")
print(f"  sin²φ = 7/8 ← K₄ (our case)")
print(f"  sin²φ = 3/8 ← different graph")
print()

# What graph has sin²φ = 3/8? cos²φ = 5/8.
# For a (q+1)-regular graph: cosφ = ±λ/(2√q)
# If cosφ = ±√(5/8) = ±√5/(2√2), then λ = ±√5 (not integer → not a simple graph)

# Observation 5: The key identity is:
# R₅ = 16/(7 × 1/16) - 4 = 256/7 - 4
# = (256 - 28)/7 = 228/7
# 256 = |h|⁴/(G₅² sin²φ · ... wait, simpler:
# 256/7 = 2/(7/128) = 2·128/7 = |h|²/sin²(5φ)
# 228/7 = 256/7 - 4 = (|h|² - (k*+1)sin²(5φ))/sin²(5φ)
# = (2 - 4·7/128)/sin²(5φ)
# = (2 - 7/32)/(7/128)
# = (64/32 - 7/32)/(7/128)
# = (57/32)/(7/128) = 57×128/(32×7) = 57×4/7 = 228/7 ✓

print("Alternative: R = (|h|² - (k*+1)sin²(5φ)) / sin²(5φ)")
print(f"  = (2 - 4×7/128) / (7/128)")
print(f"  = (2 - 7/32) / (7/128)")
print(f"  = (57/32) / (7/128) = 228/7 ✓")
print()

# =====================================================================
# THE PHYSICAL MASS MATRIX
# =====================================================================
print("=" * 70)
print("THE PHYSICAL MASS MATRIX")
print("=" * 70)
print()

# The Chebyshev propagator G_n at order n gives the AMPLITUDE for
# a quantum walk of n steps in the triplet sector.
# The SQUARED amplitude |G_n|² gives the transition probability.
#
# For the mass matrix, we need EIGENVALUES, not just G_n.
# G_n is the TRACE of the n-step propagator in the triplet sector.
# The individual eigenvalues are:
#
# σ_j^{(n)} = h^n ω^{jn}  (for the Z₃ circulant)
#
# The SQUARED eigenvalues:
# |σ_j^{(n)}|² = |h|^{2n} = q^n (DEGENERATE in j!)
#
# This is the SAME problem we found: the amplitude |h^n| doesn't depend on j.
# The splitting only comes from the INTERFERENCE between h^n and (h*)^n.
#
# The PHYSICAL mass involves the REAL PART of the propagator:
# M_jk^{(n)} = Re(σ_j^{(n)}) × (overlap with mass basis)
#
# For a HERMITIAN mass matrix from the seesaw:
# (M_ν)_jk = Σ_n c_n [h^n ω^{(j-k)n} + (h*)^n ω^{-(j-k)n}] / M_R
#           = 2Re[Σ_n c_n h^n ω^{(j-k)n}] / M_R
#
# This is a circulant with eigenvalues:
# λ_m = Σ_n c_n [h^n × 3δ_{n≡m mod 3} + (h*)^n × 3δ_{n≡-m mod 3}] / M_R
# (using Σ_k ω^{k(n-m)} = 3δ_{n≡m})
#
# If c_n = δ_{n,5} (only the 5-step propagator contributes):
# λ_m = [h^5 × 3δ_{5≡m mod 3} + (h*)^5 × 3δ_{5≡-m mod 3}]
# 5 mod 3 = 2, -5 mod 3 = 1
# λ_0 = 0, λ_1 = 3(h*)^5, λ_2 = 3h^5
# |λ_1| = |λ_2| → DEGENERATE

# So a SINGLE Chebyshev term can't split.
# But R = 228/7 comes from sin²(5φ) which is |G₅sin φ|² = |sin(5φ)|².
# The mass RATIO involves sin²(5φ) as a WHOLE, not generation-by-generation.

# REALIZATION: R = 228/7 is NOT a ratio of three generation masses.
# It's a SINGLE NUMBER derived from the K₄ spectral data.
# R = |h|²/sin²(5φ) - (k*+1) is a property of the GRAPH, not of generations.

print("KEY REALIZATION:")
print()
print("R = 228/7 is a SPECTRAL INVARIANT of the K₄ Ramanujan graph.")
print("It does NOT directly give the ratio (m₃²-m₂²)/(m₂²-m₁²)")
print("of three generation masses from a resolvent mass matrix.")
print()
print("Instead, it is:")
print("  R = q/sin²(nφ) - (q+2)")
print("where:")
print("  q = |h|² = 2          (Ramanujan eigenvalue norm)")
print("  φ = arctan(√7)        (spectral angle, from h = √q e^{i(π-φ)})")
print("  n = 5                 (Chebyshev order where G_n = -1/(q+2))")
print("  q+2 = k*+1 = 4       (vertex degree + 1)")
print()
print("The identity G₅ = -1/(k*+1) = -1/4 follows from:")
print("  U₄(cos φ) = -1/4  where  cos φ = 1/(2√q) = 1/(2√2)")
print()
print("This is a NECESSARY CONSEQUENCE of K₄ being the unique")
print("3-regular Ramanujan graph with discriminant Δ=7.")
print()

# =====================================================================
# WHAT R = 228/7 ACTUALLY IS
# =====================================================================
print("=" * 70)
print("WHAT R = 228/7 ACTUALLY IS")
print("=" * 70)
print()

# R = |h|²/sin²(5φ) - (k*+1)
# = q/(G_5² sin²φ) - (q+2)
# = 2/(1/16 × 7/8) - 4
# = 2×128/7 - 4
# = 256/7 - 28/7
# = 228/7

# Decompose: q/(G_5² sin²φ) = q × (q+2)² / sin²φ  [since G_5 = -1/(q+2)]
# = q(q+2)²/sin²φ
# = 2×16/(7/8) = 32/(7/8) = 256/7

# So R = q(q+2)²/sin²φ - (q+2)
# = (q+2)[q(q+2)/sin²φ - 1]
# = (q+2)[q(q+2)/(1-cos²φ) - 1]
# cos²φ = 1/(2q) = 1/4, sin²φ = (2q-1)/(2q) = 3/4... wait that's wrong
# cos φ = 1/(2√q) → cos²φ = 1/(4q) = 1/8 for q=2
# sin²φ = 1 - 1/8 = 7/8 ✓

# R = (q+2)[q(q+2)/((4q-1)/(4q)) - 1]
# = (q+2)[4q²(q+2)/(4q-1) - 1]
# = (q+2)[(4q³+8q² - 4q+1)/(4q-1)]

# For q=2:
# = 4[(32+32-8+1)/7] = 4[57/7] = 228/7 ✓

print("General formula for q-regular Ramanujan graph with G_n=-1/(q+2):")
print("  R(q) = q(q+2)²/(1-1/(4q)) - (q+2)")
print("       = q(q+2)² × 4q/(4q-1) - (q+2)")
print("       = (q+2) × [4q²(q+2)/(4q-1) - 1]")
print("       = (q+2) × (4q³+8q²-4q+1) / (4q-1)")
print()

for q in range(2, 8):
    sin2phi = 1 - 1/(4*q)
    R_q = q * (q+2)**2 / sin2phi - (q+2)
    R_q2 = (q+2) * (4*q**3 + 8*q**2 - 4*q + 1) / (4*q - 1)
    print(f"  q={q}: R = {R_q:.6f} = {R_q2:.6f}")
    # Check if R is a nice fraction
    from fractions import Fraction
    frac = Fraction(R_q).limit_denominator(1000)
    if abs(float(frac) - R_q) < 1e-8:
        print(f"        = {frac}")

print()

# =====================================================================
# THE DEEP IDENTITY: WHY G₅ = -1/(k*+1)
# =====================================================================
print("=" * 70)
print("WHY G₅ = -1/(k*+1) FOR K₄")
print("=" * 70)
print()

# U₄(x) = 16x⁴ - 12x² + 1
# At x = 1/(2√q) = 1/(2√2):
# U₄ = 16/(16q²) - 12/(4q) + 1 = 1/q² - 3/q + 1 = (1-3q+q²)/q²
# For q=2: (1-6+4)/4 = -1/4 = -1/(q+2) ✓

# In general: U₄(1/(2√q)) = (q²-3q+1)/q²
# This equals -1/(q+2) when:
# (q²-3q+1)/q² = -1/(q+2)
# (q+2)(q²-3q+1) = -q²
# q³-3q²+q+2q²-6q+2 = -q²
# q³-q²+q-6q+2 = -q²
# q³-5q+2 = 0... nope, that doesn't factor nicely.
# q³+0q²-5q+2 = 0
# Try q=2: 8-10+2 = 0 ✓!

print("U₄(1/(2√q)) = (q²-3q+1)/q²")
print("Setting this = -1/(q+2):")
print("  (q+2)(q²-3q+1) = -q²")
print("  q³ - 5q + 2 = 0")
print()
print("Roots:")
roots = np.roots([1, 0, -5, 2])
for r in roots:
    if abs(r.imag) < 1e-10:
        print(f"  q = {r.real:.10f}")
    else:
        print(f"  q = {r}")
print()
print("q = 2 is a root! This is the K₄ solution.")
print()

# Factor: q³ - 5q + 2 = (q-2)(q²+2q-1)
# q²+2q-1=0 → q = (-2±√8)/2 = -1±√2
# q = -1+√2 ≈ 0.414 (positive but < 1, not a valid graph degree)
# q = -1-√2 ≈ -2.414 (negative, unphysical)
print("Factored: q³-5q+2 = (q-2)(q²+2q-1)")
print(f"  q=2: K₄ (our case)")
print(f"  q=-1+√2 ≈ {-1+sqrt2:.6f} (sub-unit, unphysical)")
print(f"  q=-1-√2 ≈ {-1-sqrt2:.6f} (negative, unphysical)")
print()
print("CONCLUSION: q=2 (K₄) is the UNIQUE integer solution of")
print("  U₄(1/(2√q)) = -1/(q+2)")
print()
print("This means n=5 (i.e., U₄) giving G₅=-1/(k*+1) is")
print("SPECIFIC TO K₄. It's not a general property of Ramanujan graphs.")
print()

# =====================================================================
# BUT WHAT IS n=5 GEOMETRICALLY?
# =====================================================================
print("=" * 70)
print("GEOMETRIC MEANING OF n=5")
print("=" * 70)
print()

# K₄ has:
# - 4 vertices
# - 6 edges (12 directed)
# - NB girth = 3
# - Simple girth = 3
# - Diameter = 1 (complete graph)

# n=5 is NOT the girth, NOT half the girth, NOT the diameter.
# But: 5 = |V| + 1 = 4 + 1
# Or: 5 = 2|V| - 3
# Or: 5 = |E| - 1 = 6 - 1

# More meaningfully: n=5 is the CHEBYSHEV ORDER where the propagator
# hits the inverse degree: G₅ = ±1/(k*+1).
# This is determined by the spectral equation q³-5q+2=0.

# The NUMBER 5 in the Chebyshev index corresponds to U₄, which is
# a degree-4 polynomial. And 4 = |V|-1 = |V(K₄)|-1.

print("n = 5 is the smallest positive integer where |G_n| = 1/(k*+1).")
print()
print("Verification (scanning n):")
for n in range(1, 50):
    Gn = math.sin(n*phi)/math.sin(phi)
    if abs(abs(Gn) - 0.25) < 1e-6:
        print(f"  n={n}: G_{n} = {Gn:.15f}")
print()

# Check: is there a FORMULA for which n gives G_n = -1/(k*+1)?
# For K₄: n=5 is the first. Are there others?
# G_n satisfies the recurrence G_{n+1} = (1/√2)G_n - G_{n-1}
# (since 2cos φ = 1/√2 for K₄)
# The general solution: G_n = sin(nφ)/sin(φ)
# |G_n| = 1/4 when |sin(nφ)| = sin(φ)/4 = √(7/8)/4 = √7/(4√8) = √7/(8√2)
# sin²(nφ) = 7/128

print("sin²(nφ) = 7/128 for all matching n:")
for n in range(1, 200):
    s2 = math.sin(n*phi)**2
    if abs(s2 - 7/128) < 1e-8:
        print(f"  n={n}")
print()

# Is 5 the ONLY one? φ/π is irrational (φ = arctan√7), so sin(nφ) is
# quasi-periodic and may not return exactly to 7/128.
# But φ is a root of cos(φ) = 1/(2√2), and the values sin²(nφ) mod 1
# fill a dense set. However exact return to 7/128 requires algebraic structure.

# Check: is φ/π rational?
print(f"φ/π = {phi/math.pi:.15f}")
# This is irrational (arctan of algebraic irrational divided by π).
# So there are only FINITELY many n with sin²(nφ) = 7/128? Actually no.
# sin²(nφ) = 7/128 is equivalent to cos(2nφ) = 1 - 2×7/128 = 1 - 7/64 = 57/64
# So 2nφ = ±arccos(57/64) + 2kπ
# nφ = ±arccos(57/64)/2 + kπ

# If φ/π is irrational, the equation nφ = c + kπ has at most one solution
# for each value of c. And arccos(57/64)/(2π) is some specific irrational.
# So the solution n depends on whether φ and arccos(57/64) are commensurable.

# From our scan, n=5 is the UNIQUE solution in [1,200].

# BUT wait: let me check n=5 exactly.
# sin(5φ) where cos φ = 1/(2√2):
# Use T₅(cos φ) = cos(5φ) where T₅(x) = 16x⁵-20x³+5x
# cos(5φ) = T₅(1/(2√2)) = 16/(2√2)⁵ - 20/(2√2)³ + 5/(2√2)
# = 16/(32√2/√2·... let me just compute
# (2√2)⁵ = 2⁵·(√2)⁵ = 32·4√2 = 128√2
# (2√2)³ = 8·2√2 = 16√2
# cos(5φ) = 16/(128√2) - 20/(16√2) + 5/(2√2)
# = 1/(8√2) - 5/(4√2) + 5/(2√2)
# = [1 - 10 + 20]/(8√2) = 11/(8√2) = 11√2/16

cos5phi_exact = 11*sqrt2/16
print(f"cos(5φ) = 11√2/16 = {cos5phi_exact:.15f}")
print(f"cos(5φ) computed = {math.cos(5*phi):.15f}")
print(f"Match: {abs(cos5phi_exact - math.cos(5*phi)) < 1e-12}")
print()

# sin²(5φ) = 1 - cos²(5φ) = 1 - 121×2/256 = 1 - 242/256 = 14/256 = 7/128 ✓
print(f"sin²(5φ) = 1 - (11√2/16)² = 1 - 242/256 = 14/256 = 7/128 ✓")
print()

# Now for n=5 to be unique, we need 2nφ = ±arccos(57/64) mod 2π to have
# only n=5 as a positive integer solution. Since φ/π is irrational,
# the sequence {nφ mod π} is equidistributed, and the probability of
# hitting exactly sin²(nφ)=7/128 for another n is zero.
# So n=5 IS the unique solution.

print("n=5 is the UNIQUE positive integer with sin²(nφ) = 7/128")
print("(because φ/π is irrational, so nφ mod π is equidistributed)")
print()

# =====================================================================
# THE THEOREM
# =====================================================================
print("=" * 70)
print("T H E O R E M")
print("=" * 70)
print()
print("Let K₄ be the complete graph on 4 vertices.")
print("Let h = (-1+i√7)/2 be the Hashimoto eigenvalue (triplet sector).")
print("Let φ = arctan(√7), the spectral angle: h = √2·e^{i(π-φ)}.")
print("Let q = |h|² = 2 and k* = q+1 = 3.")
print()
print("Define the Chebyshev propagator G_n = sin(nφ)/sin(φ) = U_{n-1}(cos φ).")
print()
print("Then G₅ = -1/(k*+1) = -1/4, and this follows from the fact that")
print("q = 2 is the unique positive integer root of q³ - 5q + 2 = 0.")
print()
print("The spectral ratio is:")
print()
print("  R = q·(k*+1)²/sin²(φ) - (k*+1)")
print("    = q/sin²(5φ) - (k*+1)")
print("    = |h|²/(G₅·sinφ)² - (k*+1)")
print("    = 2/(7/128) - 4")
print("    = 256/7 - 4")
print("    = 228/7")
print()
print("Equivalently: R = (4q³+8q²-4q+1)(q+2)/(4q-1) evaluated at q=2.")
print(f"  = (32+32-8+1)×4/7 = 57×4/7 = 228/7 ✓")
print()
print("Each factor has a spectral origin:")
print(f"  |h|² = q = 2                    (Ramanujan bound, saturated)")
print(f"  sin²(φ) = 7/8 = (4q-1)/(4q)    (spectral angle from cos φ = 1/(2√q))")
print(f"  G₅² = 1/16 = 1/(k*+1)²         (5-step Chebyshev at spectral angle)")
print(f"  k*+1 = q+2 = 4                  (vertex degree + 1)")
print()
print(f"  R = q × (q+2)² × 4q/(4q-1) - (q+2) = 228/7")
print()
print("The number 228/7 is a spectral invariant of K₄, determined by")
print("the UNIQUE integer solution of q³ = 5q - 2.")
print()
print("QED")
print()

# =====================================================================
# CROSS-CHECKS
# =====================================================================
print("=" * 70)
print("CROSS-CHECKS")
print("=" * 70)
print()

# 1. Numerical verification
R_computed = 2 / math.sin(5*phi)**2 - 4
print(f"1. Numerical: R = 2/sin²(5φ) - 4 = {R_computed:.15f}")
print(f"   228/7 = {228/7:.15f}")
print(f"   Match: {abs(R_computed - 228/7) < 1e-12}")
print()

# 2. Via Chebyshev
G5 = math.sin(5*phi)/math.sin(phi)
R_cheb = 2/(G5**2 * math.sin(phi)**2) - 4
print(f"2. Chebyshev: G₅ = {G5:.15f} = -1/4")
print(f"   R = 2/(G₅²·sin²φ) - 4 = {R_cheb:.15f}")
print()

# 3. Via algebraic formula
q = 2
R_alg = (q+2) * (4*q**3 + 8*q**2 - 4*q + 1) / (4*q - 1)
print(f"3. Algebraic: R(q=2) = 4×57/7 = {R_alg:.15f}")
print()

# 4. Verify q³-5q+2=0 ↔ U₄(1/(2√q))=-1/(q+2)
lhs = q**3 - 5*q + 2
print(f"4. q³-5q+2 at q=2: {lhs} = 0 ✓")
U4_val = (q**2 - 3*q + 1)/q**2
target = -1/(q+2)
print(f"   U₄(1/(2√2)) = (4-6+1)/4 = {U4_val:.6f} = -1/4 = {target:.6f} ✓")
print()

# 5. Factorization
print("5. 228 = 4 × 57 = 4 × 3 × 19 = 12 × 19")
print("   228/7 is in lowest terms (gcd(228,7)=1)")
print()

print("=" * 70)
print("COMPLETE")
print("=" * 70)
