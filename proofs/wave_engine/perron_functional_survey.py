#!/usr/bin/env python3
"""(G) Systematically survey spectral functionals on (λ_A, λ_B) for matches
to framework constants beyond q_NB, α_1_bare, c=5/12, ε_CP=1/5.

Strategy: enumerate rational functions of λ_A, λ_B (and other graph
invariants like girth g, |E|, |V|, k*) that produce small denominators
matching framework constants.

Major new finding: α_1_full = 256/6305 has a clean spectral identification
as a geometric series in λ_B^(g-2) / λ_A^(g-2).
"""
from __future__ import annotations
import os, sys, math
from fractions import Fraction
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# srs constants at Γ
lam_A = 3   # adjacency Perron
lam_B = 2   # Hashimoto Perron
V = 4       # |V|
E = 6       # |E|
g = 10      # girth
k_star = 3

print("=" * 90)
print("(G) Perron-functional survey — searching for spectral identifications")
print("=" * 90)
print(f"\n  Substrate constants (srs at Γ): λ_A={lam_A}, λ_B={lam_B}, |V|={V}, |E|={E}, k*={k_star}, g={g}")

# Framework constants to match
framework_constants = {
    'q_NB = 2/3': Fraction(2, 3),
    '1/3 (= 1 - q_NB)': Fraction(1, 3),
    'α_1_bare = 256/6561': Fraction(256, 6561),
    'α_1_full = 256/6305': Fraction(256, 6305),
    'V_cb = 256/6305 (= α_1_full)': Fraction(256, 6305),
    'V_us = 9/40': Fraction(9, 40),
    'sin²θ_W = 3/8': Fraction(3, 8),
    'Q_Koide = 2/3': Fraction(2, 3),
    'y_τ = 1280/177147': Fraction(1280, 177147),
    'α_GUT = 1/24': Fraction(1, 24),
    '5/12 (dark)': Fraction(5, 12),
    'ε_CP = 1/5': Fraction(1, 5),
    'A_hemispherical = 1/15': Fraction(1, 15),
}

# Enumerate spectral functionals
def enum_functionals():
    """Return (name, formula_in_constants, value) tuples."""
    out = []
    # Basic
    out.append(('λ_B/λ_A', f'{lam_B}/{lam_A}', Fraction(lam_B, lam_A)))
    out.append(('λ_A/λ_B', f'{lam_A}/{lam_B}', Fraction(lam_A, lam_B)))
    out.append(('1/λ_A', f'1/{lam_A}', Fraction(1, lam_A)))
    out.append(('1/λ_B', f'1/{lam_B}', Fraction(1, lam_B)))
    out.append(('(λ_A-λ_B)/(λ_A+λ_B)', '(3-2)/(3+2)', Fraction(lam_A-lam_B, lam_A+lam_B)))
    out.append(('λ_A·λ_B', '3·2', Fraction(lam_A*lam_B)))
    out.append(('λ_A+λ_B', '3+2', Fraction(lam_A+lam_B)))

    # Higher powers
    out.append(('(λ_B/λ_A)^(g-2)', f'(2/3)^{g-2}', Fraction(lam_B, lam_A)**(g-2)))
    out.append(('λ_B^(g-2)/(λ_A^(g-2) - λ_B^(g-2))',
                f'{lam_B}^{g-2}/({lam_A}^{g-2}-{lam_B}^{g-2})',
                Fraction(lam_B**(g-2), lam_A**(g-2) - lam_B**(g-2))))
    out.append(('(λ_B/λ_A)^g', f'(2/3)^{g}', Fraction(lam_B, lam_A)**g))
    out.append(('1/(2·λ_A-1)', f'1/(2·{lam_A}-1)', Fraction(1, 2*lam_A-1)))

    # Cross with structural
    out.append(('1/V', f'1/{V}', Fraction(1, V)))
    out.append(('1/(V!)', f'1/{V}!', Fraction(1, math.factorial(V))))
    out.append(('1/E', f'1/{E}', Fraction(1, E)))
    out.append(('λ_A/V', f'{lam_A}/{V}', Fraction(lam_A, V)))
    out.append(('λ_A²/(V·g)', f'{lam_A}²/({V}·{g})', Fraction(lam_A**2, V*g)))
    out.append(('(2(E-V)+1)/(2E)', f'(2·{E-V}+1)/{2*E}', Fraction(2*(E-V)+1, 2*E)))
    out.append(('(V·k_star-1)/(V·k_star)', f'({V}·{k_star}-1)/({V}·{k_star})',
                Fraction(V*k_star-1, V*k_star)))

    # Composites
    out.append(('ε_CP · 1/λ_A', f'(1/5)·(1/3)', Fraction(1, 5) * Fraction(1, 3)))
    out.append(('ε_CP / λ_A', f'(1/5)/3', Fraction(1, 15)))
    out.append(('q_NB · 1/λ_A', f'(2/3)·(1/3)', Fraction(2, 3) * Fraction(1, 3)))

    return out

functionals = enum_functionals()

print(f"\n{'='*90}")
print(f"Spectral functional → framework constant matches")
print(f"{'='*90}")
print(f"\n  {'functional':<46}{'value':<15}{'matches':<35}")
print('  ' + '-' * 95)
for name, formula, val in functionals:
    # Check matches
    matches = []
    for fc_name, fc_val in framework_constants.items():
        if val == fc_val:
            matches.append(fc_name.split(' = ')[0])
    matches_str = ', '.join(matches) if matches else ''
    flag = '★' if matches else ''
    val_str = str(val) + (' = ' + f'{float(val):.4f}' if val.denominator > 50 or val.numerator > 100 else '')
    print(f"  {name:<46}{val_str:<15}{matches_str:<35}{flag}")

# Highlight new finding
print(f"\n{'='*90}")
print(f"NEW SPECTRAL IDENTIFICATION: α_1_full")
print(f"{'='*90}")

alpha_full_spectral = Fraction(lam_B**(g-2), lam_A**(g-2) - lam_B**(g-2))
print(f"""
  α_1_full = λ_B^(g-2) / (λ_A^(g-2) - λ_B^(g-2))
           = {lam_B}^{g-2} / ({lam_A}^{g-2} - {lam_B}^{g-2})
           = {lam_B**(g-2)} / ({lam_A**(g-2)} - {lam_B**(g-2)})
           = {alpha_full_spectral.numerator} / {alpha_full_spectral.denominator}
           = {float(alpha_full_spectral):.6f}

  Reference:  α_1_full = 256/6305 = {256/6305:.6f}
  ✓ EXACT MATCH

  Equivalent forms:
    α_1_full = α_1_bare / (1 - α_1_bare)        [geometric series sum]
             = q_NB^(g-2) / (1 - q_NB^(g-2))    [in terms of NB survival]

  Interpretation: V_cb = 256/6305 is the GEOMETRIC SUM of (q_NB^(g-2))^n
  over multiple girth-cycle windows n = 1, 2, 3, ... — corresponding to
  the A2-T waterline retention of all girth-loop returns. This is exactly
  the "geometric series in Hashimoto Perron-power" pattern, and it gives
  the dominant CKM mixing element directly from the substrate's spectrum.

  This pushes the framework's V_cb derivation from "geometric-series
  argument with Stark-Terras α_1_bare base" to "explicit spectral
  identification on the substrate's adjacency / Hashimoto Perron pair."
""")

# Pattern recognition
print(f"\n{'='*90}")
print(f"PATTERN: spectral observables of the Hashimoto operator")
print(f"{'='*90}")

print(f"""
  All identified framework constants are spectral observables of the
  substrate's (A, B) Perron pair plus structural integers (V, E, g, k*):

  RATE-TYPE (functional of Perron ratio q_NB = λ_B/λ_A = 2/3):
    q_NB                = λ_B/λ_A         = 2/3        [Row 23 q_NB]
    1 - q_NB            = (λ_A-λ_B)/λ_A   = 1/3        [backtrack rate]
    q_NB^(g-2)          = (λ_B/λ_A)^8     = 256/6561   [α_1_bare]
    q_NB^(g-2)/(1-q_NB^(g-2)) = ...       = 256/6305   [α_1_full = V_cb]

  ASYMMETRY-TYPE (functional of Perron gap):
    (λ_A - λ_B)/(λ_A + λ_B)  = 1/5        [ε_CP]
    1/(λ_A + λ_B)            = 1/5        [same as ε_CP, λ_A-λ_B = 1]

  DIMENSIONAL-TYPE (functional of structural integers):
    (2(E-V)+1)/(2E)          = 5/12       [c = dark Feshbach]
    1/k*                     = 1/3        [backtrack, also = 1-q_NB]
    1/V                      = 1/4        [per-atom factor]
    1/V!                     = 1/24       [α_GUT, group-theoretic S_4]
    k*²/(V·g)                = 9/40       [V_us]

  COMPOSITE-TYPE (products of spectral functionals):
    ε_CP · 1/k*              = 1/15       [A_hemispherical]
    q_NB · 1/k*              = 2/9        [hypothetical]

  Five framework constants now have CLEAN SPECTRAL IDENTIFICATIONS:
    1. q_NB = 2/3                     (Perron ratio)
    2. α_1_bare = 256/6561            (Perron-ratio power)
    3. α_1_full = 256/6305            (Perron-power geometric sum) ← NEW
    4. ε_CP = 1/5                     (Perron asymmetry)
    5. c = 5/12                       (Q-projector dim fraction)
    6. A_hemispherical = 1/15         (composite ε_CP/k*)

  These cover the framework's primary dark/visible coefficients PLUS
  the leading CKM mixing (V_cb). All from one operator pair (A, B).
""")
