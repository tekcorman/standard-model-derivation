#!/usr/bin/env python3
# ============================================================
# WEDGE-1: can a tight prediction be re-derived from the observer
# posterior (Stack 1) instead of the srs spectrum (Stack 2)?
# ============================================================
#
# Scope: docs/scoping/observer_unification_scope_2026-05-29.md.
# The decisive proof-of-concept for whether the observer-learner story does
# real work on the flavor/coupling predictions, or is just the graph-selector.
#
# CANDIDATE: the number 2/3, which (a) underlies the ENTIRE srs flavor coupling
# sector — alpha_1 = (2/3)^8, V_cb, V_us, ... all built on the branch measure
# (k-1)/k = 2/3 (k*=3) — and (b) is ALSO the observer's Bayesian confirm
# probability (Beta(2,1) predictive mean = 2/3), with confirm-surprise
# -log2(2/3) = 0.585 bits = the OEF's S_confirm. If these are the SAME object
# structurally, the flavor couplings ARE the observer's accumulated Bayesian
# confidence (BRIDGE). If they coincide only numerically at k=3, the flavor
# sector is genuinely srs-geometric and does not reduce to the observer
# posterior (COINCIDENCE).
#
# THE TEST (bridge vs coincidence):
#   - observer side: predictive P(exists) after n confirming observations from a
#     uniform Beta(1,1) prior = (1+n)/(2+n).  [theorem_edge_surprise_thresholds.md
#     S4-S6]  -> 2/3 at n=1.  k-INDEPENDENT.
#   - srs side: branch-measure per-step survival = (k-1)/k.  -> 2/3 at k=3.
#     k-DEPENDENT.
#   - They are equal iff (1+n)/(2+n) = (k-1)/k  <=>  n = k-2.
#   A structural BRIDGE requires a framework principle forcing n = k-2 (the
#   observation count tied to the coordination) for ALL k. A COINCIDENCE is
#   when the two are independently fixed (n=1 "one observation"; k=3 by MDL) and
#   merely cross at one point.

import math

def observer_predictive(n_obs):
    """Beta(1+n_obs, 1) predictive mean = P(exists | n confirming obs, uniform prior)."""
    a, b = 1 + n_obs, 1
    return a / (a + b)

def srs_branch_survival(k):
    """Branch-measure per-step NB survival = (k-1)/k."""
    return (k - 1) / k

def bits(p):
    return -math.log2(p)

def main():
    print("=" * 70)
    print("WEDGE-1: observer Bayesian 2/3  vs  srs branch-measure 2/3")
    print("=" * 70)

    print("\n[1] The two 2/3's at the framework's fixed point (n=1 obs, k*=3):")
    po = observer_predictive(1)
    ps = srs_branch_survival(3)
    print(f"    observer  P(exists | 1 obs, Beta(1,1)->Beta(2,1)) = {po:.4f}")
    print(f"    srs       branch survival (k-1)/k, k=3            = {ps:.4f}")
    print(f"    equal? {abs(po-ps) < 1e-12}  -> both 2/3, confirm-surprise = {bits(po):.3f} bits")
    print(f"    (the flavor coupling alpha_1 = (2/3)^8 = 2^(-8*{bits(po):.3f}) = "
          f"{(2/3)**8:.6f}; the '8' = girth-2, also srs)")

    print("\n[2] BRIDGE TEST: do they agree AWAY from k=3 / n=1?")
    print("    The observer is fixed at 'one observation' (Beta(2,1)); MDL is fixed")
    print("    at k*=3. Vary each independently and see if the agreement survives.")
    print("    k :  srs (k-1)/k   observer@n=1   agree?   n needed for match (k-2)")
    for k in (2, 3, 4, 5, 6):
        ps_k = srs_branch_survival(k)
        po_1 = observer_predictive(1)
        n_match = k - 2
        print(f"    {k} :   {ps_k:.4f}        {po_1:.4f}       "
              f"{'YES' if abs(ps_k-po_1)<1e-12 else 'no ':>3}     n={n_match}"
              + (" (=1, the chosen value)" if k == 3 else " (NOT forced by anything)"))

    print("\n[3] Is n=k-2 forced by any framework principle?")
    print("    theorem_edge_surprise_thresholds.md derives Beta(2,1) as 'Beta(1,1)")
    print("    prior + ONE observation' (S5/S10) — chosen as the minimal non-trivial")
    print("    posterior, NOT tied to k. No principle forces the observation count to")
    print("    track the coordination. So n=1 and k=3 are INDEPENDENTLY fixed.")
    print("    => the agreement holds at exactly one point and is not structural.")

    print("\n[4] Surprise-side cross-check (the OEF reading):")
    print(f"    coupling alpha_1 = (2/3)^8 = 2^(-accumulated surprise),")
    print(f"    accumulated surprise = 8 * (-log2(2/3)) = 8 * {bits(2/3):.3f} = "
          f"{8*bits(2/3):.3f} bits.")
    print(f"    BUT the per-step 2/3 here is the srs BRANCH survival and the 8 is")
    print(f"    girth-2 — both srs. The OEF 'coupling = 2^(-surprise)' is a")
    print(f"    RELABELING that reads the srs surprise, not an independent observer")
    print(f"    derivation. The Bayesian S_confirm=0.585 only coincides with it at k=3.")

    print("\n" + "=" * 70)
    print("VERDICT: COINCIDENCE, not bridge.")
    print("=" * 70)
    print("""  The ubiquitous 2/3 (and log2(3)) shared between the observer's Bayesian
  posterior and the srs flavor sector is a NUMERICAL coincidence at k=3:
   - observer 2/3 = Beta(2,1) predictive = 'one observation', k-free;
   - srs 2/3 = (k-1)/k = 'two admissible of three directions', k=3.
  They are structurally DIFFERENT questions (pair-existence posterior vs
  NB-walk admissibility) that cross at one point because k=3. The flavor
  couplings (2/3)^8 are srs-geometric and CANNOT be re-derived from the
  observer's Bayesian posterior without importing k=3 and girth from srs.

  => For the flavor/coupling sector, the observer posterior does NOT bridge.
     This sector is genuinely the observer's SPATIAL model (srs), computed
     statically, and is a separate object from the Bayesian-temporal posterior.

  IMPORTANT non-negative: the observer DOES do real, non-coincidental work
  elsewhere — the cosmology sector (H_0=1/(N t_P), Lambda=1/N^2, w_DE, t_0) and
  the arrow of time run on N = observation count = the martingale/register time,
  which IS the observer's posterior dynamics. So the observer is NOT merely the
  graph-selector; it genuinely owns cosmology + time. The disconnection is
  WITHIN the observer: a SPATIAL model (srs -> flavor, static) and a TEMPORAL
  posterior (Beta/N -> cosmology/time, dynamic) that are not unified. The OEF
  (E=kappa*S) is the only named bridge between them, and it is explicitly
  unbuilt (theorem_observer_energy_functional.md S15: 'does NOT connect to
  cosmology/masses') — and its natural form (coupling=2^-surprise) reads the
  srs surprise, not the Bayesian one.""")
    print("=" * 70)

if __name__ == "__main__":
    main()
