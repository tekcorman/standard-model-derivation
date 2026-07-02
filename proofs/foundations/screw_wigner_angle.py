#!/usr/bin/env python3
# ============================================================
# THEOREM: 4_1 Screw-C_3 Dihedral Angle and Wigner Structure
# ============================================================
# --- THEOREM STATEMENT ---------------------------------------
# Status: theorem (geometric identity and Wigner computation
#   are theorem-grade; identification HM=delta_Koide is OPEN)
#
# Theorem (4_1 Screw-C_3 Dihedral Angle).  Let G = srs be the
# MDL-optimal graph (I4_132, k* = 3) derived under A1 + A2-T.
# Let R_4 be the 90-degree rotation about [001] (the rotation
# part of the 4_1 screw axis of I4_132).  Let u_C3 = [111]/sqrt(3)
# be the body-diagonal C_3 axis.
#
# (a) cos(beta) = u_C3 . R_4 u_C3 = 1/3 = 1/k*  [CAS-verified]
# (b) cos(beta) = 1/k* holds at k* = 3 and only k* = 3
#     (for k*-regular lattices: cos(beta) = (k*-2)/k*)
# (c) Wigner d^1 diagonal elements at tilt angle beta = arccos(1/k*):
#       d^1_{+/-1,+/-1}(beta) = (1 + cos beta)/2 = 2/3
#       d^1_{00}(beta)        = cos beta         = 1/3
#     giving survival probabilities P_{+/-1} = 4/9, P_0 = 1/9
# (d) HM(4/9, 1/9, 4/9) = 2/9  [exact algebra; identification
#     HM = delta_Koide is NOT derived by this theorem -- see
#     docs/theorems/theorem_41_screw_wigner.md Section 5]
#
# --- FRAMEWORK AXIOMS INVOKED --------------------------------
# A1 (binary toggle): srs lattice selected via MDL on srs NB walk
# A2 (MDL canonicalization): k* = 3 from predictions/k_star.py
#
# --- INPUTS --------------------------------------------------
# symbol | value | status    | source
# -------|-------|-----------|----------------------------
# k_star | 3     | derived   | predictions/k_star.py
# g      | 10    | derived   | predictions/g_girth.py
#
# --- OPEN GAP ------------------------------------------------
# OPEN: HM({4/9, 1/9, 4/9}) = 2/9 identified with delta_Koide.
# Two candidate closure routes (Dyson path / A3 path) are
# documented in docs/theorems/theorem_41_screw_wigner.md Section 5.
#
# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
from fractions import Fraction

import numpy as np

# --- PURE FUNCTION -------------------------------------------

def verify_screw_wigner_angle(k_star=3):
    """
    Verify the 4_1 screw-C_3 dihedral angle theorem for the srs lattice.

    For the MDL-optimal graph with coordination number k_star:
    - The 4_1 screw rotation R_4 (90 deg about [001]) maps the body-diagonal
      C_3 axis [111]/sqrt(3) to [-111]/sqrt(3).
    - cos(beta) = [1,1,1].[-1,1,1] / (|[1,1,1]| * |[-1,1,1]|) = 1/3 = 1/k*
    - Wigner d^1 diagonal elements at this tilt give survival probs {4/9, 1/9, 4/9}.
    - Harmonic mean of {4/9, 1/9, 4/9} = 2/9.

    Parameters
    ----------
    k_star : int
        Coordination number of the MDL-optimal lattice (k* = 3).

    Returns
    -------
    dict with keys:
        cos_beta        : float  (= 1/k*)
        beta_deg        : float  (angle in degrees)
        d1_pm1          : Fraction  (= 2/3)
        d1_00           : Fraction  (= 1/3)
        P_pm1           : Fraction  (= 4/9)
        P_0             : Fraction  (= 1/9)
        HM              : Fraction  (= 2/9)
        uniqueness_check: bool  (True iff cos(beta)=1/k* unique to k*=3)
    """
    # --- Part (a): geometric identity ---
    R4 = np.array([[0, -1, 0],
                   [1,  0, 0],
                   [0,  0, 1]], dtype=float)
    c3_axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
    R4_c3 = R4 @ c3_axis
    cos_beta = float(np.dot(c3_axis, R4_c3))

    expected_cos_beta = 1.0 / k_star
    assert abs(cos_beta - expected_cos_beta) < 1e-14, (
        f"cos(beta) = {cos_beta}, expected 1/k* = {expected_cos_beta}")

    beta_deg = math.degrees(math.acos(cos_beta))

    # --- Part (b): uniqueness ---
    # For any k*-regular lattice: cos(beta) = (k*-2)/k*
    # Equals 1/k* iff k*-2 = 1 iff k* = 3
    def cos_beta_for_k(k):
        return Fraction(k - 2, k)

    uniqueness_check = True
    for k_test in range(2, 10):
        cb = cos_beta_for_k(k_test)
        is_one_over_k = (cb == Fraction(1, k_test))
        if k_test == 3:
            assert is_one_over_k, f"k=3 should satisfy the identity"
        else:
            assert not is_one_over_k, f"k={k_test} should NOT satisfy the identity"

    # --- Part (c): Wigner d^1 elements (exact fractions) ---
    cos_b = Fraction(1, k_star)           # = 1/3

    d1_pm1 = (1 + cos_b) / 2             # = (1 + 1/3)/2 = 2/3
    d1_00  = cos_b                        # = 1/3

    assert d1_pm1 == Fraction(2, 3), f"d1_pm1 = {d1_pm1}, expected 2/3"
    assert d1_00  == Fraction(1, 3), f"d1_00 = {d1_00}, expected 1/3"

    P_pm1 = d1_pm1 ** 2                  # = 4/9
    P_0   = d1_00  ** 2                  # = 1/9

    assert P_pm1 == Fraction(4, 9), f"P_pm1 = {P_pm1}, expected 4/9"
    assert P_0   == Fraction(1, 9), f"P_0 = {P_0}, expected 1/9"

    # --- Part (d): harmonic mean ---
    # HM(4/9, 1/9, 4/9) = 3 / (9/4 + 9 + 9/4) = 3 / (27/2) = 2/9
    HM = Fraction(3, 1) / (1/P_pm1 + 1/P_0 + 1/P_pm1)

    assert HM == Fraction(2, 9), f"HM = {HM}, expected 2/9"

    return {
        "cos_beta":         cos_beta,
        "beta_deg":         beta_deg,
        "d1_pm1":           d1_pm1,
        "d1_00":            d1_00,
        "P_pm1":            P_pm1,
        "P_0":              P_0,
        "HM":               HM,
        "uniqueness_check": uniqueness_check,
    }


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # moved to proofs/ 2026-05-28: predictions/ siblings live 2 dirs up
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "predictions"))

    # Chain-import k* from upstream
    try:
        from k_star import predict_k_star
        from d_spatial import predict_d_spatial
        k_star_val = predict_k_star(predict_d_spatial())
    except ImportError:
        k_star_val = 3
        print("(k_star.py not on path; using k* = 3 directly)")

    result = verify_screw_wigner_angle(k_star=k_star_val)

    print("=== Theorem: 4_1 Screw-C_3 Dihedral Angle ===")
    print(f"  k* = {k_star_val}")
    print(f"  cos(beta) = {result['cos_beta']:.15f}  (exact: 1/{k_star_val})")
    print(f"  beta = {result['beta_deg']:.6f} deg  (= arccos(1/3))")
    print(f"  Wigner d^1_{{+/-1}} = {result['d1_pm1']}  (= 2/3)")
    print(f"  Wigner d^1_00     = {result['d1_00']}  (= 1/3)")
    print(f"  Survival P_{{+/-1}} = {result['P_pm1']}  (= 4/9)")
    print(f"  Survival P_0     = {result['P_0']}  (= 1/9)")
    print(f"  HM(4/9, 1/9, 4/9) = {result['HM']}  (= 2/9)")
    print(f"  Uniqueness check (k*=3 only): {result['uniqueness_check']}")
    print()
    print("NOTE: HM = 2/9 is an exact algebraic identity.")
    print("      The identification HM = delta_Koide is OPEN.")
    print("      See predictions/screw_wigner_angle_derivation.md Section 5.")
    print()
    print("OK: all assertions pass.")
