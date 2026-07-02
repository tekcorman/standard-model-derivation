#!/usr/bin/env python3
# ============================================================
# Tier 1 (CORRECTED) — the arrow of time is a PURELY OBSERVER-GRAPH phenomenon.
# The substrate is TIMELESS. "Is there one arrow?" -> there is exactly one, and
# it lives on the observer graph; the substrate carries no arrow at all.
# ============================================================
#
# Scope: the time-flow / T-CP / arrow line-of-sight (strategy turn 2026-05-31),
# corrected per the user: the arrow of time is OBSERVER-graph, NOT substrate.
#
# THE CATEGORY ERROR THIS PROBE GUARDS AGAINST (a near-miss worth recording).
# It is tempting to read the substrate's NON-BACKTRACKING walk -- whose complex
# eigenvalue h = (sqrt3 + i*sqrt5)/2 looks "irreversible" -- as a microscopic
# arrow of time. That is WRONG. The non-backtracking constraint is a SPATIAL
# path constraint (don't immediately retrace an edge while hopping across the
# lattice); the walk's "steps" are SPATIAL hops, not clock ticks. The complex
# spectrum is the non-normality of a SPATIAL transfer operator. It yields static
# structure: masses (|h|, the persistence holonomy) and CP phases (arg h, static
# flavor invariants). It is NOT a clock and carries NO arrow.
#
# WHERE THE ARROW ACTUALLY LIVES: the OBSERVER GRAPH (the observer-learner stack
# -- Bayesian posterior / OEF / martingale-N / MDL). The arrow = the direction
# of register-N GROWTH = description-length (MDL ~ 1/2 log N) increase = Bayesian
# information accumulation. That direction is intrinsically irreversible: you
# cannot un-observe (posterior -> prior loses information; the likelihood
# martingale only accumulates). There is no second arrow to reconcile.
#
# DISCIPLINE (F7/g_A lesson, and the user's precision point): we (1) DEMONSTRATE
# the substrate is timeless/T-symmetric at the level it can be computed, (2) show
# explicitly that its complex non-backtracking spectrum is SPATIAL (re-labeling
# its meaning, not asserting time), and (3) state precisely that the single arrow
# is observer-side -- without smuggling time into the substrate.

import os
import sys
import math
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
import srs_graph_analysis as srs

K_STAR = 3
E_P = math.sqrt(K_STAR)


def build_A_and_B(cells=2):
    pos, edges, adj, _ = srs.build_supercell(cells)
    n = len(pos)
    A = np.zeros((n, n))
    for v in range(n):
        for w in adj[v]:
            A[v, w] = 1.0
    de, idx = [], {}
    for u in range(n):
        for v in adj[u]:
            idx[(u, v)] = len(de)
            de.append((u, v))
    m = len(de)
    B = np.zeros((m, m))
    for i, (u, v) in enumerate(de):
        for w in adj[v]:
            if w != u:
                B[idx[(v, w)], i] = 1.0
    return A, B, n, m


def main():
    print("=" * 76)
    print(" Tier 1 (corrected) — the arrow of time is OBSERVER-graph, not substrate")
    print("=" * 76)
    A, B, n, m = build_A_and_B(2)

    # -----------------------------------------------------------------------
    print("\n[1] The SUBSTRATE is TIMELESS (it carries no arrow):")
    evA = np.linalg.eigvals(A)
    print(f"    srs adjacency A ({n}x{n}): symmetric={np.allclose(A, A.T)}, "
          f"real spectrum={np.allclose(evA.imag, 0)} -> self-adjoint, T-symmetric.")
    print(f"    The static srs lattice is a SPATIAL object. It has geometry, not a")
    print(f"    clock. Nothing here distinguishes a 'forward' from a 'backward'.")

    # -----------------------------------------------------------------------
    print("\n[2] The substrate's complex spectrum is SPATIAL, not temporal:")
    evB = np.linalg.eigvals(B)
    print(f"    Non-backtracking (Hashimoto) B ({m}x{m}): non-normal, complex "
          f"spectrum (max|Im|={abs(evB.imag).max():.3f}), |eig|_max -> "
          f"sqrt(k*-1)=sqrt2={math.sqrt(2):.3f}.")
    print(f"    BUT 'non-backtracking' = a SPATIAL path rule (don't retrace the edge")
    print(f"    you just crossed while hopping over the lattice). The steps are")
    print(f"    SPATIAL hops, not time ticks. So h = (sqrt3+i*sqrt5)/2 is SPATIAL")
    print(f"    spectral data:")
    h = complex(E_P / 2, math.sqrt(4 * (K_STAR - 1) - E_P ** 2) / 2)
    print(f"      |h| = {abs(h):.5f}  -> mass / persistence holonomy   (STATIC)")
    print(f"      arg h = {math.degrees(math.atan2(h.imag, h.real)):.2f} deg"
          f"  -> CP phases / chirality select (STATIC)")
    print(f"    Reading this complex spectrum as 'microscopic time' is a CATEGORY")
    print(f"    ERROR: it imports a clock the substrate does not have. (This probe")
    print(f"    exists partly to record and foreclose that tempting misread.)")

    # -----------------------------------------------------------------------
    print("\n[3] The ARROW lives on the OBSERVER graph (and there is only ONE):")
    print(f"    The observer-learner stack (Bayesian posterior / OEF / martingale-N /")
    print(f"    MDL) supplies the arrow as the direction of REGISTER-N GROWTH:")
    print(f"      * N increases monotonically  -> the clock (H = 1/(N t_P))")
    print(f"      * description length MDL ~ (1/2) log N increases -> entropy arrow")
    print(f"      * Bayesian posterior only SHARPENS on accumulated data; you cannot")
    print(f"        un-observe (the likelihood martingale accumulates) -> irreversible")
    print(f"    These are ONE direction (information accumulation = N-growth = MDL")
    print(f"    increase). The 'thermodynamic' and the 'only' arrow coincide because")
    print(f"    there is no substrate arrow to bifurcate from.")
    # tiny illustration: MDL description length is monotone in N
    Ns = [10, 1e3, 1e6, 1e9]
    mdl = [0.5 * math.log(x) for x in Ns]
    print(f"    (illustration) MDL ~ 1/2 log N over N={Ns}: "
          f"{[round(x, 2) for x in mdl]}  -> strictly increasing.")

    print("\n" + "=" * 76)
    print(" VERDICT — Tier 1 (corrected)")
    print("=" * 76)
    print(f"""  ANSWER to 'is there one arrow?': YES, and it is exclusively OBSERVER-side.

   * The SUBSTRATE is timeless. Its adjacency is self-adjoint (real spectrum,
     T-symmetric); its non-backtracking complex spectrum h is SPATIAL transfer-
     operator data (mass via |h|, static CP/chirality via arg h), NOT a clock.
     There is no substrate arrow. Reading the complex h as 'microscopic time'
     is a category error -- explicitly foreclosed here.

   * The ARROW is the OBSERVER graph's single information-accumulation direction:
     register-N growth = MDL (1/2 log N) increase = Bayesian/martingale
     irreversibility ('you cannot un-observe'). There is no second arrow, so
     nothing to reconcile -- the thermodynamic arrow IS the observer's, period.

  WHAT THIS ADVANCES (precisely, no overclaim): it pins the altitude. The arrow
  of time is not a hidden substrate dynamical fact -- it is the observer graph's
  growth direction, full stop. The substrate supplies only STATIC structure that
  the observer READS (geometry -> masses; arg h -> CP phases; I4_132 -> a parity
  selection). Chirality and CP are therefore static substrate facts, NOT temporal.

  IMPLICATION for the line-of-sight program: the d/dN DYNAMICS layer (running,
  epochs, Gate-2) is OBSERVER-side too (the form of running = MDL 1/2 log N is
  already known to be observer-owned). And the Sakharov/eta route (Tier 2) must
  combine TWO DIFFERENT KINDS of object: a STATIC substrate CP invariant (arg h)
  read at the observer<->substrate bolt (Gleason d=3 -> |E|=3), with the OBSERVER's
  temporal arrow (N-growth). That kind-mismatch -- static-substrate CP vs
  observer-temporal arrow, joined only at the bolt -- is the precise structural
  statement of why baryogenesis is subtle here, and the correct Tier-2 entry.

  HONEST BOUNDS: this fixes the ALTITUDE/ontology of the arrow (observer, single,
  timeless substrate); it does not yet build the observer-graph dynamical
  entropy-production LAW (Tier 3), nor compute eta (Tier 2). It removes a wrong
  turn (arrow-in-substrate) rather than adding a new derived number.""")
    print("=" * 76)


if __name__ == "__main__":
    main()
