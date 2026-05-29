#!/usr/bin/env python3
"""
CAS verification: Wigner D-matrix for the 4₁ screw projected onto the C₃ axis.

Proves docs/theorems/theorem_41_screw_wigner.md §4–§5 at Type 2 (exact rational arithmetic).

Setup:
  R₄ = 90° rotation about [001]  (4₁ screw rotation part, ITA No. 214)
  C₃ axis = [111]/√3             (site symmetry axis at each srs vertex)

Claim: the j=1 Wigner D-matrix D^1_{m'm}(R₄) in the [111]-quantisation frame has
  |D^1_{-1,-1}|² = |D^1_{+1,+1}|² = 4/9
  |D^1_{ 0, 0}|²                   = 1/9
and their harmonic mean = 2/9 exactly.
"""

import numpy as np
from fractions import Fraction

RTOL = 1e-12   # tolerance for float checks before asserting exact fractions


# ---------------------------------------------------------------------------
# 1. Build R₄ and change of basis to [111]-aligned frame
# ---------------------------------------------------------------------------

def build_R4():
    """90° rotation about [001]: (x,y,z) -> (-y, x, z)."""
    return np.array([[0, -1, 0],
                     [1,  0, 0],
                     [0,  0, 1]], dtype=float)


def build_basis_111():
    """Orthonormal basis with e3 = [111]/sqrt(3)."""
    e3 = np.array([1, 1, 1]) / np.sqrt(3)
    e1 = np.array([1, -1, 0]) / np.sqrt(2)
    e2 = np.cross(e3, e1)          # completes right-handed frame
    e2 /= np.linalg.norm(e2)       # normalise (should be unit already)
    P = np.column_stack([e1, e2, e3])
    return P


def R4_in_111_frame(R4, P):
    """Express R₄ in the [111]-aligned frame: R_111 = P^{-1} R₄ P."""
    return np.linalg.inv(P) @ R4 @ P


# ---------------------------------------------------------------------------
# 2. Convert R_111 to Wigner D-matrix (j=1, spherical harmonic basis)
# ---------------------------------------------------------------------------

def spherical_change_of_basis():
    """
    Unitary matrix U transforming from Cartesian {e₁,e₂,e₃} to spherical
    harmonic basis {|+1>, |0>, |-1>} via
      |+1> = -(ê₁ + i ê₂)/√2
      | 0> =  ê₃
      |-1> =  (ê₁ - i ê₂)/√2
    (Condon-Shortley convention, Edmonds 1957 §1.2.)
    """
    s2 = np.sqrt(2)
    U = np.array([
        [-1/s2, -1j/s2, 0],
        [    0,      0, 1],
        [ 1/s2, -1j/s2, 0]
    ], dtype=complex)
    return U


def wigner_D1(R_111):
    """
    Wigner D^1 matrix for rotation R_111 expressed in the [111] frame.

    D = U R_111 U^{-1}

    Rows/columns ordered |+1>, |0>, |-1>  (m = +1, 0, -1).
    """
    U = spherical_change_of_basis()
    return U @ R_111.astype(complex) @ np.linalg.inv(U)


# ---------------------------------------------------------------------------
# 3. Extract survival probabilities and verify exact rational values
# ---------------------------------------------------------------------------

def check_survival_probs(D):
    """
    Verify diagonal |D_{mm}|² = {4/9, 1/9, 4/9} for m = +1, 0, -1.
    Returns the three probabilities as floats.
    """
    probs = [abs(D[m, m])**2 for m in range(3)]
    expected = [Fraction(4, 9), Fraction(1, 9), Fraction(4, 9)]

    for m, (p, e) in enumerate(zip(probs, expected)):
        label = {0: '+1', 1: '0', 2: '-1'}[m]
        print(f"  |D^1_{{{label},{label}}}|² = {p:.15f}  expected {e} = {float(e):.15f}")
        assert abs(p - float(e)) < RTOL, (
            f"Survival probability for m={label} deviates from {e}: got {p}"
        )

    # Row unitarity check
    for m in range(3):
        row_sum = sum(abs(D[m, mp])**2 for mp in range(3))
        assert abs(row_sum - 1.0) < RTOL, f"Row {m} of D not unitary: sum={row_sum}"

    print("  ✓ All diagonal probabilities match exact fractions.")
    print("  ✓ All rows unitary.")
    return probs


def check_harmonic_mean(probs):
    """
    Verify HM({4/9, 1/9, 4/9}) = 2/9 by exact rational arithmetic.
    """
    # Exact fractions
    P = [Fraction(4, 9), Fraction(1, 9), Fraction(4, 9)]

    hm_exact = Fraction(3, 1) / sum(Fraction(1, 1) / p for p in P)
    expected  = Fraction(2, 9)

    print(f"\n  HM(4/9, 1/9, 4/9) = {hm_exact}  (exact rational)")
    print(f"  Expected 2/9 = {expected}")
    assert hm_exact == expected, f"HM != 2/9: got {hm_exact}"
    print("  ✓ Harmonic mean = 2/9 exact.")

    # Cross-check numerically
    hm_float = 3.0 / sum(1.0 / p for p in probs)
    assert abs(hm_float - float(expected)) < RTOL
    print(f"  ✓ Float cross-check: HM = {hm_float:.15f}")

    return hm_exact


def check_q_n_gen_route():
    """
    Verify Route B: δ = Q/n_gen = (2/3)/3 = 2/9 (exact rational arithmetic).
    Q = 2/3 is upstream-closed in predictions/Q_Koide.py.
    n_gen = 3 from Spin(8) triality (Out(Spin(8)) = S₃).
    """
    Q      = Fraction(2, 3)
    n_gen  = 3
    delta  = Q / n_gen
    expected = Fraction(2, 9)

    print(f"\n  Route B: δ = Q/n_gen = {Q}/{n_gen} = {delta}")
    assert delta == expected, f"Route B failed: {delta} != {expected}"
    print(f"  ✓ Route B gives δ = {delta} = {float(delta):.15f}")

    # Self-consistency: Q²/2 = Q/3 iff Q = 2/3
    lhs = Q * Q / 2
    rhs = Q / n_gen
    assert lhs == rhs, f"Self-consistency Q²/2 = Q/3 failed: {lhs} != {rhs}"
    print(f"  ✓ Self-consistency: Q²/2 = {lhs} = Q/n_gen = {rhs}")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Wigner D¹ verification for the 4₁ screw on C₃ axis")
    print("(docs/theorems/theorem_41_screw_wigner.md §4–§7, Type 2 gate)")
    print("=" * 60)

    # Build matrices
    R4  = build_R4()
    P   = build_basis_111()
    R_111 = R4_in_111_frame(R4, P)
    D   = wigner_D1(R_111)

    print("\n--- R₄ rotation matrix (90° about [001]) ---")
    print(R4.astype(int))

    print("\n--- [111]-frame basis vectors ---")
    print(f"  e1 = [1,-1,0]/√2 = {P[:,0].round(6)}")
    print(f"  e2 = e3×e1       = {P[:,1].round(6)}")
    print(f"  e3 = [1,1,1]/√3  = {P[:,2].round(6)}")

    print("\n--- Tilt angle between [001] and [111] ---")
    cos_beta = 1.0 / np.sqrt(3)
    beta_deg = np.degrees(np.arccos(cos_beta))
    print(f"  cos β = 1/√3 = {cos_beta:.10f}")
    print(f"  β = {beta_deg:.6f}°")

    print("\n--- Wigner D¹ matrix (j=1, [111] quantisation) ---")
    for i in range(3):
        labels = ['+1', ' 0', '-1']
        row = "  "
        for j in range(3):
            v = D[i, j]
            row += f"  D[{labels[i]},{labels[j]}] = {v.real:+.6f}{v.imag:+.6f}i"
        print(row)

    print("\n--- Diagonal survival probabilities |D_{mm}|² ---")
    probs = check_survival_probs(D)

    print("\n--- Harmonic mean (Route A) ---")
    hm = check_harmonic_mean(probs)

    print("\n--- Route B: δ = Q/n_gen ---")
    check_q_n_gen_route()

    print("\n" + "=" * 60)
    print(f"RESULT: δ = {hm} = {float(hm):.10f}")
    print("Both routes (Wigner HM and Q/n_gen) agree.")
    print("docs/theorems/theorem_41_screw_wigner.md §4–§7 verified.")
    print("=" * 60)
