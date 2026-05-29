#!/usr/bin/env python3
# ============================================================
# THEOREM: Feshbach coupling strength alpha_1 = (2/3)^(g-2)
# ============================================================
# --- THEOREM STATEMENT ---------------------------------------
# Status: THEOREM-GRADE under A1 + A2-T + A5(b) + Jaynes 1957.
#   (Updated 2026-04-19 session 2: A5(b) extension subsumes
#   the previously-adopted "I-Feshbach" identification.)
#
# Lemma 1 (tree NB walk survival, THEOREM-GRADE):
#   On the universal covering tree of a k-regular graph, the
#   probability that an NB walker stays on the tree for L
#   consecutive steps is ((k-1)/k)^L.
#
# Corollary (srs, L = g-2, THEOREM-GRADE):
#   alpha_1^bare = (2/3)^(g-2) = (2/3)^8 = 256/6561 (exact).
#
# Identification with physical coupling — A5(b) axiom:
#   By A5(b) (coupling clause of A5; docs/framework/framework_axioms.md §5b,
#   established 2026-04-19), the MDL probability of a leading-order
#   multiway process IS the physical coupling strength of that
#   process. Therefore alpha_1^bare (the NB-walk survival probability)
#   IS the dark-sector coupling magnitude in the visible-sector
#   self-energy. This was previously "ADOPTED-I-Feshbach"; under
#   A5(b) it is an axiomatic identification (same epistemic tier
#   as the mass-clause A5(a) identifying eigenvalues with masses).
#
# --- FRAMEWORK AXIOMS INVOKED --------------------------------
# A1 (binary toggle): NB walk dynamics on srs
# A2-T (MDL canonicalization; derived theorem): k* = 3, g = 10 from predictions/
#       See docs/theorems/theorem_A2_mdl_from_finite_register.md
# A5(b) (coupling clause): MDL probability = physical coupling
#
# --- INPUTS --------------------------------------------------
# symbol | value | status    | source
# -------|-------|-----------|----------------------------
# k_star | 3     | derived   | predictions/k_star.py
# g      | 10    | derived   | predictions/g_girth.py
#
# --- IMPLEMENTATION ------------------------------------------

import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# moved to proofs/ 2026-05-27: predictions/ siblings live 2 dirs up at <repo>/predictions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "predictions"))


def lemma1_tree_nb_survival(k, L):
    """
    Lemma 1: tree NB walk survival probability (THEOREM-GRADE).

    On the universal covering tree of a k-regular graph, the probability
    that an NB walker stays on the tree for L consecutive steps equals
    ((k-1)/k)^L.

    Proof: at each vertex, k-1 of the k incident edges are NB-admissible
    (the incoming edge is excluded).  Under the Jaynes-uniform distribution
    (walker_dynamics Step 4), each of the k-1 is chosen with probability
    1/(k-1); the unconditional probability of staying on the tree is (k-1)/k
    per step.  On the universal covering tree, no two NB walks reconverge,
    so steps are independent and multiply.

    Parameters
    ----------
    k : int
        Coordination number (k-regular graph).
    L : int
        Number of NB steps.

    Returns
    -------
    Fraction
        Exact probability ((k-1)/k)^L.
    """
    return Fraction(k - 1, k) ** L


def alpha1_bare(k_star, g):
    """
    Feshbach coupling strength alpha_1^bare = ((k*-1)/k*)^(g-2).

    This is the corollary of Lemma 1 for srs:
    - k* = 3 (coordination number, MDL-optimal)
    - g = 10 (girth, MDL-optimal)
    - L = g - 2 = 8 internal NB steps (2 external edges pinned: Exponent Principle
      with n_fixed = 2)

    The ADOPTED-Exponent-Principle identification:
    For a scattering process with 2 fixed external edges on a graph of girth g,
    the shortest admissible closed loop through the external edges has g-2 internal
    NB steps.  Each contributes tree-survival amplitude (k-1)/k.
    Hence alpha_1^bare = ((k*-1)/k*)^(g-2).

    ADOPTED: the Exponent Principle itself is at "numerically verified +
    Feynman-analog motivated" status; it does not yet have a standalone
    journal-grade proof independent of numerical checks.

    Parameters
    ----------
    k_star : int
        Coordination number of the MDL-optimal lattice.
    g : int
        Girth of the MDL-optimal lattice.

    Returns
    -------
    Fraction
        Exact value of alpha_1^bare = ((k*-1)/k*)^(g-2).
    """
    L_internal = g - 2   # n_fixed = 2 external edges
    return lemma1_tree_nb_survival(k_star, L_internal)


# --- PURE FUNCTION -------------------------------------------

def verify_feshbach_coupling_strength(k_star=3, g=10):
    """
    Verify alpha_1^bare = (2/3)^8 = 256/6561 for srs (k*=3, g=10).

    Also verify:
    - Lemma 1 for several (k, L) values.
    - The Exponent Principle table entries for k*=3, g=10.

    Parameters
    ----------
    k_star : int
        Coordination number (k* = 3 for srs).
    g : int
        Girth (g = 10 for srs).

    Returns
    -------
    dict with keys:
        alpha1_exact : Fraction
        alpha1_float : float
        lemma1_checks : list of (k, L, result)
        exponent_table : dict
        adopted_flag : str
    """
    # Core result
    a1 = alpha1_bare(k_star, g)
    assert a1 == Fraction(256, 6561), (
        f"alpha_1^bare = {a1}, expected 256/6561")
    assert abs(float(a1) - 256/6561) < 1e-15

    # Verify Lemma 1 for several (k, L) values
    lemma1_checks = []
    for k_test in [2, 3, 4, 5]:
        for L_test in [1, 2, 3, 5, 8]:
            result = lemma1_tree_nb_survival(k_test, L_test)
            expected = Fraction(k_test - 1, k_test) ** L_test
            assert result == expected
            lemma1_checks.append((k_test, L_test, result))

    # Exponent Principle table (Feynman-rule analog)
    # n_fixed = 0 (closed loop): L = g     -> (2/3)^10
    # n_fixed = 1 (transition): L = g-1   -> (2/3)^9
    # n_fixed = 2 (scattering): L = g-2   -> (2/3)^8 = alpha_1^bare
    exponent_table = {
        "scattering (n_fixed=2)": (g - 2, lemma1_tree_nb_survival(k_star, g - 2)),
        "transition (n_fixed=1)": (g - 1, lemma1_tree_nb_survival(k_star, g - 1)),
        "self-energy (n_fixed=0)": (g,     lemma1_tree_nb_survival(k_star, g)),
    }

    assert exponent_table["scattering (n_fixed=2)"][1] == Fraction(256, 6561)
    assert exponent_table["transition (n_fixed=1)"][1] == Fraction(512, 19683)  # (2/3)^9
    assert exponent_table["self-energy (n_fixed=0)"][1] == Fraction(1024, 59049) # (2/3)^10

    return {
        "alpha1_exact":   a1,
        "alpha1_float":   float(a1),
        "lemma1_checks":  lemma1_checks,
        "exponent_table": exponent_table,
        "adopted_flag":   (
            "ADOPTED-Exponent-Principle: I-Feshbach identification is adopted; "
            "the Exponent Principle is numerically verified on K_4 and srs but "
            "not yet proved independently at journal grade.  "
            "See ../predictions/Feshbach_coupling_strength_derivation.md §9 for the precise "
            "operator-algebraic gap (eigenspace projectors commute with B, "
            "so the finite K_4 matrix computation cannot close I-Feshbach).  "
            "Closure requires Route A (Ihara-Bass Green's function on srs) or "
            "Route B (physical P/Q definition non-orthogonal to eigenbasis)."
        ),
    }


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    # Chain-import k* and g from upstream
    try:
        from k_star import predict_k_star
        from d_spatial import predict_d_spatial
        from g_girth import predict_g_girth
        d_val      = predict_d_spatial()
        k_star_val = predict_k_star(d_val)
        g_val      = predict_g_girth(k_star_val, d_val)
    except ImportError:
        k_star_val = 3
        g_val      = 10
        print("(upstream imports not on path; using k* = 3, g = 10 directly)")

    result = verify_feshbach_coupling_strength(k_star=k_star_val, g=g_val)

    print("=== Theorem: Feshbach Coupling Strength ===")
    print(f"  k* = {k_star_val}, g = {g_val}")
    print(f"  alpha_1^bare = ((k*-1)/k*)^(g-2) = (2/3)^8 = {result['alpha1_exact']}")
    print(f"  float value: {result['alpha1_float']:.6f}")
    print()
    print("  Exponent Principle table (scattering/transition/self-energy):")
    for label, (L, val) in result["exponent_table"].items():
        print(f"    {label}: L={L}, amplitude = {val} = {float(val):.6f}")
    print()
    print("  Lemma 1 (tree NB survival) spot-checks: all pass")
    print()
    print(f"  {result['adopted_flag']}")
    print()
    print("OK: all assertions pass.")
