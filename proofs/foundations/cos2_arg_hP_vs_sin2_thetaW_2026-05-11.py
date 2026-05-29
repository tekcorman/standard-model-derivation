"""
proofs/foundations/cos2_arg_hP_vs_sin2_thetaW_2026-05-11.py

SETTLE: is cos²(arg h_P) = sin²θ_W = 3/8 a STRUCTURAL identity (giving a
second derivation of the Weinberg angle), or a k*=3-correlated coincidence?

Method:
  1. Derive cos²(arg h_P) as a closed form in substrate primitives.
  2. Derive sin²θ_W = 3/8 (GQW trace) as a closed form in substrate primitives.
  3. Check whether the two closed forms are the SAME function of k* (→ structural)
     or DIFFERENT functions that coincide at k*=3 (→ coincidence).
  4. Check whether there's a chain of reasoning linking the Hashimoto saddle
     to the gauge-coupling ratio that DEFINES sin²θ_W (→ structural) or not.
"""

import math
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.srs_engine.srs_substrate import SrsSubstrate
substrate = SrsSubstrate()


def main():
    print("=" * 100)
    print("SETTLE: cos²(arg h_P) = sin²θ_W = 3/8 — structural identity or coincidence?")
    print("=" * 100)
    print()

    k_star = 3
    V = 4
    E = 6
    g = 10

    # ============================================================
    # PART 1: cos²(arg h_P) closed form
    # ============================================================
    print("PART 1 — cos²(arg h_P) closed form in substrate primitives")
    print("-" * 100)
    print()

    # h_P comes from the Bass/Ihara formula μ² − λ·μ + (k*−1) = 0
    # at the P-point adjacency eigenvalue λ_P.
    # First: what IS λ_P (P-point adjacency eigenvalue)?
    A_P = substrate.adjacency_at_k('P')
    evals_A_P = sorted(la.eigvals(A_P).real, key=abs, reverse=True)
    print(f"  A(P) eigenvalues: {[f'{e:+.6f}' for e in evals_A_P]}")
    lambda_P = abs(evals_A_P[0])
    print(f"  λ_P = |A(P) eigenvalue| = {lambda_P:.6f}")
    print(f"  λ_P² = {lambda_P**2:.6f}")
    print()

    # Is λ_P² = k*?
    print(f"  Check λ_P² vs k*: {lambda_P**2:.6f} vs {k_star}")
    print(f"  λ_P² = k*  ⟺  ? (verify it's structural)")
    print()
    print(f"  Structural reason: A(P) is 4×4 Hermitian (K_4 quotient), traceless")
    print(f"  (no self-loops), with Tr(A(P)²) = 2|E| = {2*E} (sum of |off-diag|²).")
    print(f"  If A(P)² = c·I (all eigenvalues equal magnitude), then 4c = Tr(A(P)²) = 2|E|")
    print(f"  → c = 2|E|/|V| = {2*E}/{V} = {2*E//V} = k* (since k* = 2|E|/|V| for k*-regular)")
    A_P_sq = A_P @ A_P
    is_cI = la.norm(A_P_sq - k_star * np.eye(4)) < 1e-10
    print(f"  Verification: A(P)² = k*·I ?  {'YES ✓' if is_cI else 'NO ✗'}")
    print(f"    ||A(P)² − k*·I|| = {la.norm(A_P_sq - k_star*np.eye(4)):.2e}")
    print()
    if is_cI:
        print(f"  ★ STRUCTURAL FACT: A(P)² = k*·I, hence λ_P² = k* = 2|E|/|V|.")
        print(f"    (This is what makes P the 'canonical mass-content k-point' — A(P)")
        print(f"     acts like √k* times a reflection.)")
    print()

    # Bass formula: |h_P|² = k* − 1 (constant term), Re(h_P) = λ_P/2
    print(f"  Bass/Ihara formula μ² − λ_P·μ + (k*−1) = 0:")
    print(f"    h_P = (λ_P + i√(4(k*−1) − λ_P²))/2 = (√k* + i√(4(k*−1) − k*))/2")
    print(f"        = (√k* + i√(3k*−4))/2")
    # For k*=3: (√3 + i√5)/2 ✓
    h_P_re = math.sqrt(k_star) / 2
    h_P_im = math.sqrt(3*k_star - 4) / 2
    print(f"    For k*={k_star}: h_P = (√{k_star} + i√{3*k_star-4})/2 = ({h_P_re:.6f}, {h_P_im:.6f})")
    h_P_abs2 = h_P_re**2 + h_P_im**2
    print(f"    |h_P|² = {h_P_abs2:.6f}  (should equal k*−1 = {k_star-1})")
    print()

    # cos²(arg h_P) = Re²/|h_P|²
    cos2_arg_hP = h_P_re**2 / h_P_abs2
    print(f"  cos²(arg h_P) = Re(h_P)²/|h_P|² = (k*/4)/(k*−1) = k*/(4(k*−1))")
    print(f"    = {k_star}/(4·{k_star-1}) = {k_star}/{4*(k_star-1)} = {Fraction(k_star, 4*(k_star-1))}")
    print(f"    Numerically: {cos2_arg_hP:.6f}")
    print()
    print(f"  ★ CLOSED FORM: cos²(arg h_P) = k*/(4(k*−1))")
    print(f"    For k*=2: {Fraction(2, 4)} = 1/2")
    print(f"    For k*=3: {Fraction(3, 8)} = 3/8  ← framework's value")
    print(f"    For k*=4: {Fraction(4, 12)} = 1/3")
    print(f"    For k*=5: {Fraction(5, 16)} = 5/16")
    print()

    # ============================================================
    # PART 2: sin²θ_W = 3/8 closed form (GQW trace)
    # ============================================================
    print("PART 2 — sin²θ_W = 3/8 closed form (GQW trace identity)")
    print("-" * 100)
    print()
    print(f"  Framework's sin²θ_W = Tr(T_3²)/Tr(Q²) on the PS 16-state generation")
    print(f"  (= Cl(6) Fock dim 2^k* per chirality × 2 chiralities = 2^(k*+1) states)")
    print()
    print(f"  16-state PS generation = (4,2,1) ⊕ (4̄,1,2) under SU(4)×SU(2)_L×SU(2)_R")
    print()
    # Tr(T_3²): the (4,2,1) has 8 states in SU(2)_L doublets (T_3 = ±1/2);
    # the (4̄,1,2) has 8 SU(2)_L singlets (T_3 = 0).
    # So Tr(T_3²) = 8 × (1/2)² = 2.
    n_doublet_states = 8  # = 2^k* for k*=3
    Tr_T3_sq = n_doublet_states * (0.5)**2
    print(f"  Tr(T_3²) = (# states in SU(2)_L doublets) × (1/2)²")
    print(f"           = 2^k* × 1/4 = 2^(k*−2)")
    print(f"           = 2^{k_star} × 1/4 = {n_doublet_states}/4 = {Tr_T3_sq}")
    print()
    # Tr(Q²): standard PS charge assignment
    # u: Q=2/3, 3 colors × 2 (L/R) = 6 states
    # d: Q=-1/3, 6 states
    # e: Q=-1, 2 states (L/R)
    # ν: Q=0, 2 states
    charges = [(2/3, 6), (-1/3, 6), (-1, 2), (0, 2)]
    Tr_Q_sq = sum(q**2 * n for q, n in charges)
    print(f"  Tr(Q²) over 16 states (Q = T_3R + (B−L)/2, Slansky-normalized):")
    print(f"    u (Q=2/3, ×6): {(2/3)**2 * 6:.4f} = 24/9")
    print(f"    d (Q=-1/3, ×6): {(1/3)**2 * 6:.4f} = 6/9")
    print(f"    e (Q=-1, ×2): {1**2 * 2:.4f} = 2")
    print(f"    ν (Q=0, ×2): 0")
    print(f"    Tr(Q²) = 24/9 + 6/9 + 18/9 = 48/9 = 16/3 = {Fraction(16,3)}  (= {Tr_Q_sq:.6f})")
    print()
    sin2_thetaW = Tr_T3_sq / Tr_Q_sq
    print(f"  sin²θ_W = Tr(T_3²)/Tr(Q²) = 2/(16/3) = 6/16 = 3/8 = {Fraction(3,8)}")
    print(f"    Numerically: {sin2_thetaW:.6f}")
    print()
    print(f"  ★ Is this k*-dependent? Tr(T_3²) = 2^(k*−2). Tr(Q²) for the SM-PS rep")
    print(f"    is 16/3 for k*=3. The 16 = 2^(k*+1). For the formula to give 3/8,")
    print(f"    we'd need Tr(Q²) = 2^(k*−2)/(3/8) = 2^(k*+1)/3.")
    print(f"    Then sin²θ_W = 2^(k*−2)/(2^(k*+1)/3) = 3/2^3 = 3/8, k*-INDEPENDENT.")
    print(f"    HOWEVER: the PS embedding for k*≠3 is structurally different")
    print(f"    (Spin(8) triality at k*=4 etc.) — '2^(k*+1)/3' doesn't generalize")
    print(f"    cleanly. The framework only HAS k*=3 (MDL-dominant), so 'sin²θ_W")
    print(f"    at other k*' is not well-defined. What's clear: sin²θ_W's 3/8 is a")
    print(f"    rep-theory trace ratio on the PS multiplet, NOT a function of the")
    print(f"    form k*/(4(k*−1)).")
    print()

    # ============================================================
    # PART 3: Compare the two
    # ============================================================
    print("PART 3 — comparison")
    print("-" * 100)
    print()
    print(f"  cos²(arg h_P) = k*/(4(k*−1))   — substrate SPECTRAL fact (Bass formula")
    print(f"                                    at P + A(P)²=k*I + |h_P|²=k*−1)")
    print(f"  sin²θ_W       = Tr(T_3²)/Tr(Q²) — REPRESENTATION-THEORY fact (GQW trace")
    print(f"                                    on the PS multiplet)")
    print()
    print(f"  At k*=3: BOTH = 3/8.")
    print(f"  As functions of k*: cos²(arg h_P) = k*/(4(k*−1)) varies (1/2, 3/8, 1/3, 5/16, ...)")
    print(f"                      sin²θ_W = rep-theory trace, NOT k*/(4(k*−1)).")
    print()
    print(f"  Do they share a MECHANISM?")
    print(f"    - cos²(arg h_P) uses the ADJACENCY operator A(P) and its eigenvalue λ_P.")
    print(f"    - sin²θ_W uses the WEAK-ISOSPIN (T_3) and CHARGE (Q) operators on the")
    print(f"      PS multiplet.")
    print(f"    - These are DIFFERENT operators on DIFFERENT structures.")
    print(f"    - Both DO route through the (4,2,2) C_3 isotypic structure at P")
    print(f"      (V_Ram(P) = (4,2,2); each PS 8-state sector = (4,2,2) under C_3),")
    print(f"      but they extract 3/8 via different traces / spectral data ON that")
    print(f"      structure.")
    print()
    print(f"  Is there a chain h_P → ... → (gauge coupling ratio that DEFINES sin²θ_W)?")
    print(f"    NO. The Hashimoto saddle h_P enters MASS-content predictions (Koide,")
    print(f"    m_τ family) via V_Ram. It does NOT enter gauge-coupling predictions.")
    print(f"    sin²θ_W is defined by g'²/(g²+g'²) ≡ GQW trace at unification — a")
    print(f"    gauge-sector quantity. There's no derivation chain from the spectral")
    print(f"    saddle to the gauge coupling ratio.")
    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"""
  COINCIDENCE — specifically a 'k*=3-correlated coincidence', NOT a structural
  identity.

  Precise statement:
    - cos²(arg h_P) = k*/(4(k*−1)) is a clean closed form, derivable from
      substrate spectral facts (A(P)² = k*·I, Bass formula, |h_P|² = k*−1).
      For k*=3 it equals 3/8.
    - sin²θ_W = 3/8 is a representation-theory trace ratio (GQW) on the PS
      multiplet. It is NOT of the form k*/(4(k*−1)); it's a fixed number
      determined by the PS charge structure.
    - Both equal 3/8 because both are determined by the framework's
      commitment to k*=3 (which gives srs, hence the P-point spectrum AND
      the Cl(6) Fock + PS embedding). They are NOT linked by a shared
      mechanism.
    - There is no derivation chain from the Hashimoto saddle h_P to the
      gauge-coupling ratio that defines sin²θ_W.

  CONSEQUENCE: this does NOT give a second derivation of the Weinberg angle.
  The cos²(arg h_P) = k*/(4(k*−1)) = 3/8 fact is worth recording as a noted
  numerical coincidence (with the precise k*=3-correlation reason), so future
  work doesn't re-discover it and over-interpret. But it is not a structural
  identity and should not be added as an alternative derivation in
  theorem_sin2_theta_W_unification.md.

  NOTE the asymmetry: cos²(arg h_P) = k*/(4(k*−1)) IS a genuine substrate
  formula for the saddle's cos² — that part is structural. What's coincidental
  is that this particular substrate quantity happens to equal sin²θ_W.
""")


if __name__ == "__main__":
    main()
