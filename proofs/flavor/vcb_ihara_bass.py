#!/usr/bin/env python3
"""
V_cb from Ihara-Bass: symbolic resolvent on srs at P-point.

Strategy: work on the K4 quotient graph (4 vertices of the srs primitive
cell) with Bloch phases at the P-point. The adjacency matrix A(k_P) is
known (eigenvalues ±√3, multiplicity 2 each). The Ihara-Bass relation
connects A to the NB walk resolvent. We extract the vertex-to-vertex
NB Green's function and evaluate at the appropriate spectral parameter.

The key relation (Terras 2011, Theorem 4.1):

  For a k-regular graph with adjacency matrix A (n×n), the NB walk
  generating function from vertex a to vertex b is:

  G_ab(u) = Σ_d N_d(a,b) u^d = [(I - u²)^{-1} (I + uR(u)A - (k-1)u²R(u))]_{ab}

  where R(u) = (I - uA + (k-1)u²I)^{-1} is the adjacency resolvent.

We compute this symbolically at the P-point of srs.
"""

import sympy as sp
from sympy import sqrt, I, Matrix, eye, ones, simplify, Rational
from sympy import cos, sin, pi, exp, conjugate, Abs, re, im
from sympy import symbols, factor, collect, expand, cancel, together
from sympy import Poly, degree


def main():
    print("=" * 70)
    print("V_cb from Ihara-Bass: symbolic computation")
    print("=" * 70)

    u = symbols('u')
    k = 3  # k* = 3

    # ================================================================
    # STEP 1: Adjacency matrix of K4 at the P-point
    # ================================================================
    #
    # The srs primitive cell has 4 atoms. The quotient graph is K4
    # (complete graph on 4 vertices) with Bloch phases.
    #
    # At the P-point k_P = (π/2a)(1,1,1), each bond picks up a phase
    # exp(i k_P · d_ij) where d_ij is the bond displacement vector.
    #
    # For the srs Wyckoff 8a positions in the BCC primitive cell,
    # the A(k_P) eigenvalues are ±√3, each with multiplicity 2.
    # (Proven in predictions/srs_E_at_P_derivation.md)
    #
    # Rather than building A(k_P) from scratch, we use its known
    # spectral decomposition. A is 4×4 with eigenvalues {√3, √3, -√3, -√3}.
    #
    # We need the EIGENVECTORS to compute off-diagonal matrix elements.
    # These come from the C₃ decomposition at the P-point.

    print("\nStep 1: A(k_P) spectral decomposition")
    print(f"  Eigenvalues: +√3 (mult 2), -√3 (mult 2)")

    # The 4 atoms decompose under C₃ at the P-point as:
    # 2 trivial irreps (eigenvalue 1) → bands with E = ±√3
    # 1 ω-irrep, 1 ω²-irrep → also contribute to ±√3 bands
    #
    # For the resolvent computation, we only need the PROJECTORS
    # P+ and P- onto the ±√3 eigenspaces.
    #
    # By completeness: P+ + P- = I (4×4 identity)
    # By orthogonality: P+·P- = 0
    # By spectral theorem: A = √3·P+ - √3·P-
    #
    # So: P+ = (I + A/√3)/2, P- = (I - A/√3)/2
    #
    # We don't need the explicit A(k_P) — just the projectors!

    # ================================================================
    # STEP 2: Adjacency resolvent R(u)
    # ================================================================

    print("\nStep 2: Adjacency resolvent")

    # R(u) = (I - uA + (k-1)u²I)^{-1}
    # Since A = √3(P+ - P-), the resolvent in the spectral basis is:
    #
    # R(u) = P+ / (1 - u√3 + 2u²) + P- / (1 + u√3 + 2u²)

    f_plus = 1 - u*sqrt(3) + 2*u**2    # pole factor for +√3
    f_minus = 1 + u*sqrt(3) + 2*u**2   # pole factor for -√3

    print(f"  f+(u) = {f_plus}")
    print(f"  f-(u) = {f_minus}")

    # Roots of f+(u) = 0:
    roots_plus = sp.solve(f_plus, u)
    print(f"  Roots of f+(u): {roots_plus}")
    # Should be (√3 ± i√5)/4 = h/2, h*/2

    # Verify: h/2
    h = (sqrt(3) + I*sqrt(5)) / 2
    h_half = h / 2
    print(f"  h/2 = {h_half} = {simplify(h_half)}")
    print(f"  f+(h/2) = {simplify(f_plus.subs(u, h_half))}")

    # ================================================================
    # STEP 3: NB walk generating function
    # ================================================================

    print("\nStep 3: NB walk generating function")

    # The vertex-to-vertex NB Green's function (Terras 2011, eq. 4.3):
    #
    # G(u) = (1-u²)^{-1} [I + u·R(u)·A - (k-1)u²·R(u)]
    #
    # Using spectral decomposition:
    # R(u)·A = √3·P+/f+(u) - √3·P-/f-(u)
    # R(u) = P+/f+(u) + P-/f-(u)
    #
    # G(u) = (1-u²)^{-1} [I + u(√3·P+/f+ - √3·P-/f-) - 2u²(P+/f+ + P-/f-)]
    #
    # G_ab(u) = (1-u²)^{-1} [δ_ab + P+(a,b)(u√3 - 2u²)/f+ + P-(a,b)(-u√3 - 2u²)/f-]

    # For off-diagonal (a ≠ b) CKM elements: δ_ab = 0.
    # G_ab(u) = (1-u²)^{-1} [P+(a,b)(u√3 - 2u²)/f+ + P-(a,b)(-u√3 - 2u²)/f-]

    # Let's denote P+(a,b) = p and P-(a,b) = q.
    # Since P+ + P- = I and a ≠ b: p + q = 0, so q = -p.

    # Therefore:
    # G_ab(u) = p/(1-u²) × [(u√3 - 2u²)/f+ - (-u√3 - 2u²)/f-]
    #         = p/(1-u²) × [(u√3 - 2u²)/f+ + (u√3 + 2u²)/f-]

    p = symbols('p')  # P+(a,b) for the V_cb generation pair

    num1 = u*sqrt(3) - 2*u**2
    num2 = u*sqrt(3) + 2*u**2

    # Common denominator f+·f-
    combined = (num1 * f_minus + num2 * f_plus) / (f_plus * f_minus)
    combined_num = expand(num1 * f_minus + num2 * f_plus)

    print(f"\n  Numerator of combined fraction:")
    print(f"    num1·f- + num2·f+ = {combined_num}")
    combined_num = collect(expand(combined_num), u)
    print(f"    Collected: {combined_num}")

    # f+·f- = (1 + 2u²)² - 3u² = 1 + 4u² + 4u⁴ - 3u² = 1 + u² + 4u⁴
    fp_fm = expand(f_plus * f_minus)
    print(f"    f+·f- = {fp_fm}")

    # Full G_ab(u):
    # G_ab = p × combined_num / ((1-u²) × f+·f-)

    G_ab_num = p * combined_num
    G_ab_den = (1 - u**2) * fp_fm

    print(f"\n  G_ab(u) = p × {combined_num}")
    print(f"           / ({1-u**2}) × ({fp_fm})")

    # ================================================================
    # STEP 4: Power series expansion
    # ================================================================

    print("\nStep 4: Power series expansion of G_ab(u)")

    # Expand G_ab(u) as a power series in u to get walk amplitudes
    # G_ab(u) = Σ N_d(a,b) · u^d

    # Compute the ratio symbolically
    G_ratio = combined_num / ((1 - u**2) * fp_fm)

    # Series expansion around u=0
    series = sp.series(G_ratio, u, 0, n=20)
    print(f"\n  G_ab(u)/p = {series}")

    # Extract coefficients
    print(f"\n  Coefficients N_d/p:")
    for d in range(20):
        coeff = series.coeff(u, d)
        if coeff != 0:
            print(f"    d={d:2d}: {coeff}")

    # ================================================================
    # STEP 5: Evaluate at u = (k-1)/k = 2/3
    # ================================================================

    print("\nStep 5: Evaluate at u = 2/3")

    u_val = Rational(2, 3)

    # Compute G_ab(2/3) / p
    G_val = G_ratio.subs(u, u_val)
    G_val_simplified = simplify(G_val)
    print(f"\n  G_ab(2/3) / p = {G_val_simplified}")
    print(f"  Numerical: {float(G_val_simplified):.10f}")

    # Compare with alpha_1(1+alpha_1)
    alpha_1 = Rational(2, 3)**8
    V_cb_pred = alpha_1 * (1 + alpha_1)
    print(f"\n  α₁ = (2/3)^8 = {alpha_1} = {float(alpha_1):.10f}")
    print(f"  α₁(1+α₁) = {V_cb_pred} = {float(V_cb_pred):.10f}")
    print(f"  Ratio G/V_cb = {simplify(G_val_simplified / V_cb_pred)}")

    # ================================================================
    # STEP 6: What is p = P+(a,b)?
    # ================================================================

    print("\nStep 6: Projector P+(a,b) for V_cb generation pair")
    print("  P+ = (I + A/√3)/2")
    print("  For K4 quotient: A = J - I (all-ones minus identity)")
    print("  A/√3 has diagonal 0/√3 = 0, off-diagonal 1/√3")
    print("  P+ = (I + A/√3)/2:")
    print("    P+(a,a) = (1 + 0)/2 = 1/2")
    print("    P+(a,b) = (0 + 1/√3)/2 = 1/(2√3)  for a ≠ b")
    print()
    print("  BUT: this is for the UNTWISTED K4.")
    print("  At the P-point, the Bloch phases modify A.")
    print("  The correct P+(a,b) depends on the eigenvectors of A(k_P).")
    print()
    print("  For now, p = P+(a,b) is a proportionality constant that")
    print("  scales all off-diagonal elements equally (by K4 symmetry).")
    print("  The RELATIVE structure (which power of u contributes) is")
    print("  independent of p.")

    # ================================================================
    # STEP 7: Partial sums and comparison
    # ================================================================

    print("\nStep 7: Partial sums to match V_cb = α₁(1+α₁)")

    # From the series, what's the sum of terms at d=8 and d=16?
    c8 = series.coeff(u, 8)
    c16 = series.coeff(u, 16)
    partial_8_16 = c8 * u_val**8 + c16 * u_val**16

    print(f"  N_8/p = {c8}")
    print(f"  N_16/p = {c16}")
    print(f"  c8·(2/3)^8 + c16·(2/3)^16 = {simplify(partial_8_16)}")
    print(f"  Numerical: {float(simplify(partial_8_16)):.10f}")
    print(f"  V_cb/p = α₁(1+α₁)/p requires p·partial = α₁(1+α₁)")


if __name__ == '__main__':
    main()
