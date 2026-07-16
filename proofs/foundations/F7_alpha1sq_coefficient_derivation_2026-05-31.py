#!/usr/bin/env python3
# ============================================================
# F7 closure attempt: DERIVE the alpha1^2 coefficient F7 found
# (Delta eps^2_up ~ +1.066 alpha1^2) — or honestly conclude it isn't structural.
# ============================================================
#
# Scope: internal research notes §F7 closure target.
#
# F7 found m_u's +15.5% is consistent with a missing Delta(eps^2_up) = +1.066*alpha1^2.
# The closure target: derive that coefficient as the next-order analog of W4's
# leading  eps^2(n) = 2 + N_LQ*alpha1*n*f(n)  (N_LQ=6 leptoquark coset; W4 theorem).
#
# CRITICAL reading of the W4 theorem (theorem_quark_koide_eps_n):
#  - the cluster expansion is ENTIRELY O(alpha1): one-body (n) + two-body pair
#    correlation (n(n-1)/2 pairs, each carrying alpha1*(g-2)/g, a GEOMETRIC ratio,
#    NOT alpha1^2). f(n) is the n-combinatorics at leading coupling order.
#  - the next combinatorial term is 3-body, C(n,3), which VANISHES for n<=2.
#  - Clause 8: eps^2_U predicted 1.0925 vs PDG 1.094 = -0.1% (ALREADY correct);
#    residual explicitly attributed to RG-running/scheme (the ~1% quark systematic).
#
# So we test every natural structural alpha1^2 candidate against F7's +1.066*alpha1^2.
# If none matches -> the coefficient is NOT structural; F7's clue is the amplified
# RG/scheme residual the theorem already names.

import math

A1 = (5.0/3.0)*(2.0/3.0)**8     # alpha1_full
A1_BARE = (2.0/3.0)**8          # bare winding
G, N_LQ = 10, 6
def f_of_n(n): return 1.0 + (n-1)*(G-2)/(2.0*G)

# F7 target (from F7_up_sector_amplifier): Delta(eps^2) to fix m_u, in units of alpha1^2
F7_TARGET = {2: 1.066, 1: -0.687}   # up (n=2) clean; down (n=1) not-clean (m_d/m_s opposite)


def main():
    print("="*72)
    print("F7 closure: is the alpha1^2 coefficient (~+1.07 for up) STRUCTURAL?")
    print("="*72)
    print(f"alpha1_full={A1:.6f}  alpha1^2={A1**2:.6f}  alpha1_bare=(2/3)^8={A1_BARE:.6f}")
    print(f"F7 needs Delta(eps^2_up) = +1.066*alpha1^2 = {1.066*A1**2:+.6f}")
    print(f"  (= +0.146% of eps^2_up; W4 Clause-8 says eps^2_U is already -0.1% vs PDG)")

    print("\n[candidates] natural structural next-order terms, evaluated at n=2 (up),")
    print("  expressed as a coefficient x alpha1^2 (to compare with F7's +1.066):")
    n = 2; fn = f_of_n(n)
    leading = N_LQ*A1*n*fn          # the W4 leading term (for reference)
    cands = {
        "3-body cluster C(n,3)*...":      0.0,                       # C(2,3)=0
        "(leading)^2 / 2 (exponentiate)": 0.5*(leading**2)/A1**2,    # 1/2 (N_LQ a1 n f)^2
        "N_LQ^2 * a1^2 * n*f":            (N_LQ**2)*n*fn,            # 36 n f
        "N_LQ * a1^2 * n*f":              N_LQ*n*fn,                 # 6 n f
        "C(N_LQ,2) a1^2 * n*f (pairs)":   (N_LQ*(N_LQ-1)/2)*n*fn,    # 15 n f
        "a1^2 * n*f (coeff 1)":           n*fn,                      # 2.8
        "a1^2 * n (coeff 1, no f)":       float(n),                  # 2.0
        "a1^2 * f (coeff 1, no n)":       fn,                        # 1.4
        "winding-resum: leading*a1b/(1-a1b)/a1^2":
            leading*(1.0/(1-A1_BARE) - 1.0)/A1**2,                  # per-channel resummation
    }
    print(f"   {'candidate':<42} {'coeff x a1^2':>14}  {'vs F7 (+1.066)':>16}")
    best = None
    for name, c in cands.items():
        match = "MATCH" if abs(c - 1.066) < 0.15 else ("close" if abs(c-1.066) < 0.5 else "no")
        if best is None or abs(c-1.066) < abs(best[1]-1.066):
            best = (name, c)
        print(f"   {name:<42} {c:>14.3f}  {match:>16}")
    print(f"\n   closest structural candidate: {best[0]} = {best[1]:.3f} x alpha1^2")
    print(f"   (F7 fit = 1.066; none of the natural integer-coefficient structural")
    print(f"    terms lands on it — and the only ~O(1) ones are 'coeff-1' ansaetze")
    print(f"    with no structural origin for the '1'.)")

    print("\n[RG/scheme check] is F7's shift just the documented eps^2 residual?")
    eps2_U_pred = 2 + N_LQ*A1*2*f_of_n(2)
    eps2_U_pdg = 3.094                # W4 Clause-8 PDG extraction
    print(f"   eps^2_U predicted = {eps2_U_pred:.4f}; PDG (W4 Clause-8) = {eps2_U_pdg:.4f}")
    print(f"   PDG - pred = {eps2_U_pdg-eps2_U_pred:+.4f} = {100*(eps2_U_pdg-eps2_U_pred)/eps2_U_pred:+.3f}% "
          f"(the documented RG/scheme residual)")
    print(f"   F7 needs +{1.066*A1**2:.4f} = +0.146% (anchor-inflated; framework m_t is +0.82% high)")
    print(f"   -> F7's needed shift ~ the documented RG/scheme residual, SAME size.")

    print("\n" + "="*72)
    print("VERDICT — the derivation does NOT close (honest negative)")
    print("="*72)
    print("""  No natural structural alpha1^2 term reproduces F7's +1.066*alpha1^2:
   - the 3-body cluster term VANISHES for n=2 (and n=1);
   - the exponentiation (leading^2/2) and N_LQ-based second-order terms are
     1-2 ORDERS too big (~36-141 x alpha1^2);
   - the per-channel winding resummation is ~10x too big AND wrong form
     (alpha1*alpha1_bare, not alpha1^2) — and would over-correct m_u;
   - only structureless 'coeff = 1' ansaetze land near 1, with no derivation
     of the '1' (unlike the leading '6' = N_LQ).

  Meanwhile the W4 theorem's OWN Clause 8 says eps^2_U is already correct to
  ~0.1%, with the residual attributed to RG-running/scheme. F7's needed shift
  (+0.146% of eps^2) is the SAME size. So:

  m_u's +15.5% is the ~100x AMPLIFICATION (f_min -> 0) of the already-documented
  ~0.1% RG/scheme residual on eps^2 — NOT a missing structural coupling-order
  term. There is no alpha1^2 coefficient to derive; the 'alpha1^2' size match is
  coincidental (RG/scheme effects are alpha-sized).

  CONSEQUENCE / honest redirect:
   - F7's clue is REAL but points at the KNOWN universal quark RG/scheme
     systematic (the same ~1% that makes m_t +0.82%, m_b +2.15% — see the
     constellation RUN class), now seen through m_u's uniquely sensitive lever.
   - The 'closure target' is therefore NOT 'derive alpha1^2'. It is the
     substrate->MS-bar scale-matching for quark masses — a separate, harder,
     already-open problem (the RG-dynamical / d/dN axis), affecting all quarks.
   - The F7->F8 payoff (m_d-m_u ~ 2.45 ~ lattice 2.49) is CONDITIONAL on that
     scale-matching being right (the lattice value is at 2 GeV MS-bar); it is a
     scale-matched agreement, not a parameter-free structural one.

  So F7 honestly resolves: m_u is the best AMPLIFIER of the quark RG/scheme
  residual, not a handle on a new Koide term. The wide net found a real, sharp
  diagnostic — pointing back at the RG-dynamical axis, not a fresh structural
  coupling. That is the honest bottom of this thread.""")
    print("="*72)


if __name__ == "__main__":
    main()
