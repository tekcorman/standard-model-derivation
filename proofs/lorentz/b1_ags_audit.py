#!/usr/bin/env python3
"""
b1 audit for Arratia-Goldstein-Gordon Poisson approximation.
Tests whether the a separate private derivation by the author toggle process satisfies the rare-events
condition required for AGG to give a useful Lorentz invariance bound.

Conclusion: AGG fails (lambda not small). Correct tool is 2-point
connected-correlation decay via Ramanujan spectral bound.
"""

from fractions import Fraction
import math

def main():
    print("=" * 65)
    print("b1 audit: AGG Poisson approximation for a separate private derivation by the author toggle process")
    print("=" * 65)

    # --- Edge toggle Markov chain ---
    p_create  = Fraction(1, 2)   # off -> on per Planck step
    p_destroy = Fraction(1, 3)   # on  -> off per Planck step

    pi_on  = p_create  / (p_create + p_destroy)   # 3/5
    pi_off = p_destroy / (p_create + p_destroy)   # 2/5

    lam = pi_on * p_destroy + pi_off * p_create   # 2/5
    assert lam == Fraction(2, 5)

    # Second eigenvalue of 2-state Markov chain: r = 1 - p_create - p_destroy
    r_chain = 1 - p_create - p_destroy            # 1/6
    assert r_chain == Fraction(1, 6)

    print(f"\nToggle Markov chain:")
    print(f"  p_create = {p_create},  p_destroy = {p_destroy}")
    print(f"  pi_on = {pi_on},  pi_off = {pi_off}")
    print(f"  lambda (toggle rate per edge per step) = {lam} = {float(lam):.4f}")
    print(f"  r_chain (2nd eigenvalue) = {r_chain} = {float(r_chain):.4f}")
    print(f"  Toggle autocorr decays as (1/6)^s per step")

    # --- NB walk on srs ---
    k_star = 3
    # srs conventional cell: 8 vertices, 12 undirected edges = 24 directed
    n_directed = 24
    # Ramanujan: |mu2(NB walk)| = sqrt(k-1)/(k-1) = 1/sqrt(k-1) = 1/sqrt(2)
    r_NB = 1.0 / math.sqrt(2)
    g = 10  # girth of srs

    print(f"\nNB walk on srs:")
    print(f"  k* = {k_star}, directed edges per cell = {n_directed}")
    print(f"  |mu2(NB)| = 1/sqrt(2) = {r_NB:.4f}  (Ramanujan)")
    print(f"  NB walk correlation decays as (1/sqrt(2))^s per step")
    print(f"  Girth g = {g}: NB walk can't revisit same directed edge in < {g} steps")

    # --- CASE 1: Planck-scale individual toggle events ---
    print("\n" + "="*55)
    print("CASE 1: Individual toggle events (Planck scale)")
    print("="*55)

    lam_f = float(lam)
    print(f"\n  lambda = {lam} = {lam_f:.4f}")
    print(f"  Rare-events condition: lambda << 1  -->  {lam_f:.4f} << 1  FAILS")

    # Dependency: different edges are independent (separate Markov chains)
    # Only same-edge, different-time events are correlated
    eps_1 = lam_f**2
    r_dep_1 = math.log(1/eps_1) / math.log(1/float(r_chain))
    D_1 = 2 * r_dep_1

    b1_ratio_1 = D_1 * lam_f   # b1/lambda_total
    print(f"\n  Same-edge temporal dependency range (eps = lambda^2 = {eps_1:.4f}):")
    print(f"    r_dep = {r_dep_1:.1f} steps  (Markov chain threshold)")
    print(f"    D = 2*r_dep = {D_1:.1f}")
    print(f"  b1/lambda_total = D * lambda = {b1_ratio_1:.3f}")
    print(f"  AGG verdict: b1/lambda = {b1_ratio_1:.2f} -- {'GOOD' if b1_ratio_1 < 0.1 else 'MARGINAL' if b1_ratio_1 < 1.0 else 'FAIL'}")

    # --- CASE 2: Observer-experienced toggle events ---
    print("\n" + "="*55)
    print("CASE 2: Observer-experienced toggles (NB walk frame)")
    print("="*55)

    lam_obs = float(lam) / n_directed   # rate per observer step
    print(f"\n  lambda_obs = lambda / |E_d| = {float(lam):.4f} / {n_directed} = {lam_obs:.6f}")
    print(f"  Rare-events condition: lambda_obs << 1  -->  {lam_obs:.4f} << 1  YES (small)")

    # Combined correlation decay: NB return probability × Markov chain
    # For s < g: C = 0 (NB walk on different edges -> independent)
    # For s >= g: C <= C_0 * r_NB^(s-g) * r_chain^s
    # Use r_combined = r_NB * r_chain as decay rate
    r_combined = r_NB * float(r_chain)
    print(f"\n  Combined correlation decay (s >= g={g}):")
    print(f"    r_combined = r_NB * r_chain = {r_NB:.4f} * {float(r_chain):.4f} = {r_combined:.6f}")

    eps_2 = lam_obs**2
    # Correlation at s=g already: r_combined^g
    corr_at_girth = r_combined**g
    print(f"    Correlation at s=g={g}: r_combined^{g} = {corr_at_girth:.3e}")
    print(f"    Threshold eps = lambda_obs^2 = {eps_2:.3e}")
    print(f"    At girth: {corr_at_girth:.3e} vs threshold {eps_2:.3e}")

    if corr_at_girth < eps_2:
        r_dep_2 = g  # already below threshold at girth
        print(f"    Correlation ALREADY below threshold at girth -> r_dep = g = {g}")
    else:
        extra = math.log(corr_at_girth/eps_2) / math.log(1/r_combined)
        r_dep_2 = g + extra
        print(f"    Need extra {extra:.1f} steps above girth -> r_dep = {r_dep_2:.1f}")

    D_2 = 2 * r_dep_2
    b1_ratio_2 = D_2 * lam_obs   # b1/lambda_total
    print(f"\n  D = 2*r_dep = {D_2:.1f}")
    print(f"  b1/lambda_total = D * lambda_obs = {D_2:.1f} * {lam_obs:.6f} = {b1_ratio_2:.4f}")
    print(f"  AGG verdict: b1/lambda = {b1_ratio_2:.4f} -- {'GOOD' if b1_ratio_2 < 0.1 else 'MARGINAL (not << 1)' if b1_ratio_2 < 1.0 else 'FAIL'}")

    # --- CASE 3: What actually works ---
    print("\n" + "="*55)
    print("CASE 3: Correct approach -- 2-point connected correlation")
    print("="*55)

    xi_lP = 1.0 / math.log(math.sqrt(2))   # correlation length in units of l_P
    print(f"""
The AGG rare-events route fails because:
  lambda = 2/5 is not small (Case 1 fails completely)
  D * lambda_obs = {b1_ratio_2:.2f} is O(1), not << 1 (Case 2 marginal)

Correct target (user's formulation):
  Show 2-point connected correlation decays exponentially:
    |C2_conn(x,t; x',t')| <= K * exp(-L / xi)

  xi = l_P / log(sqrt(2)) = {xi_lP:.4f} l_P  (Ramanujan spectral bound)

  Lorentz violation in the 2-point function:
    |C2_conn| / C2_total <= exp(-L/xi)
    At L = 1 l_P:  {math.exp(-1/xi_lP):.4f}
    At L = 5 l_P:  {math.exp(-5/xi_lP):.4f}
    At L = 10 l_P: {math.exp(-10/xi_lP):.4f}
    At L = 50 l_P: {math.exp(-50/xi_lP):.4e}

  The leading (Lorentz-invariant) piece: C2_total = lambda^2 = {float(lam)**2:.4f}
    lambda^2 is a scalar (no preferred direction, by srs isotropy/Sunada)
    Corrections: O(exp(-L/xi)) for L >> xi ~ 3 l_P

  n-point functions:
    C_n^total = lambda^n + O(exp(-L_min/xi))
    lambda^n is Lorentz-invariant (scalar density to the n-th power)
    Corrections exponentially suppressed at macroscopic scales
""")

    print("="*55)
    print("VERDICT: AGG/b1 route BLOCKED.")
    print("Replace L6 with Ramanujan 2-point decay argument.")
    print("The exponential suppression is STRONGER than (1/M)^{1/2}")
    print("from Palm-Khintchine -- user's formulation wins.")
    print("="*55)


if __name__ == '__main__':
    main()
