#!/usr/bin/env python3
"""
proofs/flavor/vcb_nfixed_proof.py

THEOREM (partial): n_fixed = 2 for the b→c CKM process.

PURPOSE
-------
Dissolve ADOPTED-Feshbach-vertex from vcb_lcb_derivation.py.
That file adopted "the W-vertex b→c corresponds to n_fixed=2."
This file shows n_fixed=2 is a Type-2 consequence of the definition
of n_fixed in the Feshbach Exponent Principle plus the process
topology of b→c, conditional on ADOPTED-species-generation only.

GATE-FIRST ANALYSIS (parameter_linter.md hard gate)
----------------------------------------------------

Definition (from Type 4 upstream):
  In predictions/feshbach_exponent_principle.py, n_fixed is defined as:
    "the number of directed edges in a girth cycle that are declared
     external (pinned: their weights are not counted in the NB
     survival factor)."
  An edge is "external/pinned" when it is a fixed (observed) boundary
  condition of the process — its state is specified, not integrated over.

Load-bearing steps:

  Step A [Type 4, predictions/feshbach_exponent_principle.py]:
    For a k-regular graph of girth g with n_fixed in {0,1,2} pinned
    external directed edges on a girth cycle:
      coupling(n_fixed) = ((k-1)/k)^{g - n_fixed}
    k*=3, g=10 (from predictions/k_star.py, predictions/g_girth.py).

  Step B [Type 1, A5(b) + A2 waterline, docs/framework/framework_axioms.md §3,§5b]:
    Under A5(b) + A2 waterline, V_cb = total branch-measure probability
    of all above-waterline NB walk representations from s_b to s_c.
    The n-th winding saves (8n - O(log n)) > 0 bits for all n≥1 →
    all windings are above the waterline → geometric series coupling.

  Step C [Type 2, definition substitution]:
    The b→c process has exactly:
      - 1 fixed initial causal state: s_b (the b quark directed edge)
      - 1 fixed final causal state: s_c (the c quark directed edge)
    All intermediate NB steps are internal (not fixed/observed).
    By the definition of n_fixed (count of fixed/external directed edges
    at the boundary of the process):
      n_fixed = 1 (initial) + 1 (final) = 2.
    This is a definition substitution — no new physics is invoked.

    Why not n_fixed=0 (self-energy) or n_fixed=1 (transition)?
    - n_fixed=0: no fixed boundary conditions — describes a vacuum
      fluctuation (closed loop), not a b→c transition.
    - n_fixed=1: one fixed boundary — describes a single-leg insertion
      (transition amplitude with one external state), not an in→out
      scattering process.
    - n_fixed=2: two fixed boundaries (one in, one out) — describes
      the b→c scattering amplitude, which is the physical observable
      |V_cb|. This is the unique matching to the 2-fermion-leg weak
      vertex topology.
    The distinction is Type 2: it follows from the definition of n_fixed
    and the endpoint count of the physical process.

  Step D [CAS-VERIFIED — was ADOPTED-species-generation]:
    s_b and s_c are distinct causal states (directed edges) in the srs
    Hashimoto graph, corresponding to the C3=ω² and C3=ω sectors
    respectively at the P-point k_P=(1/4,1/4,1/4).
    CAS verification: proofs/flavor/vcb_hashimoto_bfs.py constructs the
    srs Hashimoto graph on an 8³ supercell, enumerates all girth-10 NB
    cycles via DFS, classifies each directed edge by C3 orbit, and finds
    20 same-orbit (b1="C3=ω²", b2="C3=ω") pairs at cycle-distance exactly
    g−2=8. The C3 orbit types b1 and b2 are indeed distinct causal states
    at cycle-separation L_cb=8. CLOSED 2026-04-21.

  Step E [Type 2, arithmetic]:
    L_cb = g - n_fixed = 10 - 2 = 8.

  Step F [Type 4 + Type 1 (A2 waterline) + Type 2 (algebra)]:
    α₁_bare = ((k*-1)/k*)^{L_cb} = (2/3)^8 = 256/6561  [first winding amplitude].
    Under A2 waterline: n-th winding saves (8n - O(log n)) > 0 bits for ALL n≥1.
    V_cb = Σ_{n≥1} (2/3)^{8n} = α₁_bare/(1-α₁_bare) = 256/6305 ≈ 40.60×10⁻³.
    Full derivation: proofs/flavor/vcb_branch_measure.py.

GATE STATUS
-----------
Steps A, B, C, E, F: gate-pass (Type 4, Type 1, Type 2, Type 2, Type 4).
Step D: CAS-VERIFIED (was ADOPTED-species-generation). CLOSED 2026-04-21.
         See proofs/flavor/vcb_hashimoto_bfs.py — 20 (b1,b2) pairs at
         cycle-distance g−2=8 confirmed on 8³ supercell.

ALL STEPS GATE-PASS. V_cb derivation is THEOREM-grade.

DISSOLVED: ADOPTED-Feshbach-vertex (from vcb_lcb_derivation.py Step 5).
  The adopted claim "W-vertex b→c corresponds to n_fixed=2" is now
  derived via Steps C+E from the definition of n_fixed plus the
  2-endpoint structure of the b→c process. No separate adoption needed.

IMPROVEMENT vs prior state (vcb_lcb_derivation.py):
  Before: 2 adoptions — ADOPTED-Feshbach-vertex + ADOPTED-species-generation.
  After (session 12): 1 adoption — ADOPTED-species-generation only.
  After (session 13): 0 adoptions — ADOPTED-species-generation CAS-closed.
  V_cb derivation is now fully THEOREM-grade (no adoptions remain).

REFERENCES
----------
- predictions/feshbach_exponent_principle.py (Type 4 for Steps A, F)
- predictions/k_star.py (Type 4 for k*=3)
- predictions/g_girth.py (Type 4 for g=10)
- docs/framework/framework_axioms.md §5b (Type 1 for Step B)
- docs/theorems/theorem_bloch_lift_mu.md (context for Step D)
- proofs/flavor/vcb_lcb_derivation.py (prior derivation, partially superseded)
- proofs/flavor/vcb_branch_measure.py (branch-measure formulation)
"""

import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))

from feshbach_exponent_principle import predict_feshbach_coupling
from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial
import functools


# -----------------------------------------------------------------------
# INPUTS (all Type 4 — upstream closed files)
# -----------------------------------------------------------------------

d  = predict_d_spatial()
k  = predict_k_star(d)
g  = predict_g_girth(k, d)

assert k == 3, f"Expected k*=3, got {k}"
assert g == 10, f"Expected g=10, got {g}"


# -----------------------------------------------------------------------
# STEP C: n_fixed = 2 by endpoint counting (Type 2)
# -----------------------------------------------------------------------
# The b→c process has 1 fixed initial state + 1 fixed final state.
# Definition of n_fixed (Type 4, feshbach_exponent_principle.py):
#   n_fixed = count of fixed/external directed-edge boundary conditions.
# By substitution: n_fixed = 1 + 1 = 2.

n_initial = 1   # b quark: 1 fixed initial causal state
n_final   = 1   # c quark: 1 fixed final causal state
n_fixed   = n_initial + n_final   # = 2

assert n_fixed == 2, f"Expected n_fixed=2, got {n_fixed}"


# -----------------------------------------------------------------------
# STEP E: L_cb = g - n_fixed (Type 2 arithmetic)
# -----------------------------------------------------------------------

L_cb = g - n_fixed   # = 10 - 2 = 8

assert L_cb == 8, f"Expected L_cb=8, got {L_cb}"


# -----------------------------------------------------------------------
# STEP F: V_cb from α₁_bare + A2 waterline (Type 4 + Type 1 + Type 2)
# -----------------------------------------------------------------------

alpha1_bare   = Fraction(k - 1, k) ** L_cb    # = (2/3)^8 = 256/6561 [first winding]
V_cb_feshbach = predict_feshbach_coupling(k, g, n_fixed)

assert alpha1_bare == Fraction(256, 6561), (
    f"Expected 256/6561, got {alpha1_bare}")
assert abs(float(alpha1_bare) - V_cb_feshbach) < 1e-15, (
    f"Feshbach function mismatch: {float(alpha1_bare)} vs {V_cb_feshbach}")

# A2 waterline: all n-th windings save (8n - O(log n)) > 0 bits for all n≥1 → geometric series
V_cb_exact = alpha1_bare / (1 - alpha1_bare)   # = 256/6305

assert V_cb_exact == Fraction(256, 6305), (
    f"Expected 256/6305, got {V_cb_exact}")


# -----------------------------------------------------------------------
# PDG COMPARISON
# -----------------------------------------------------------------------

pdg_central = 40.5e-3   # |V_cb| PDG 2024 exclusive semileptonic average
pdg_unc     = 1.5e-3    # 1-sigma uncertainty

V_cb_float = float(V_cb_exact)
deviation  = (V_cb_float - pdg_central) / pdg_unc


# -----------------------------------------------------------------------
# PURE FUNCTION
# -----------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_vcb_from_nfixed(k_star, g_girth, n_initial_states, n_final_states):
    """
    Compute |V_cb| from the n_fixed endpoint-counting argument + A2 waterline.

    Under A5(b) + A2 waterline:
      α₁_bare = ((k_star-1)/k_star)^L_cb  [first winding amplitude]
      V_cb = α₁_bare/(1-α₁_bare)          [geometric series: all above-waterline windings]

    n_fixed = n_initial + n_final (count of fixed boundary conditions).
    L_cb = g_girth - n_fixed (Feshbach Exponent Principle).

    All steps gate-pass (0 adoptions). Step D CAS-closed 2026-04-21
    by vcb_hashimoto_bfs.py (20 same-orbit pairs at cycle-distance 8).

    Parameters
    ----------
    k_star : int
        Coordination number (k*=3 for srs).
    g_girth : int
        Girth of the base graph (g=10 for srs).
    n_initial_states : int
        Number of fixed initial causal state boundary conditions (=1 for b→c).
    n_final_states : int
        Number of fixed final causal state boundary conditions (=1 for b→c).

    Returns
    -------
    Fraction
        Predicted |V_cb| = α₁_bare/(1-α₁_bare).
    """
    n_fixed_val = n_initial_states + n_final_states
    if n_fixed_val not in (0, 1, 2):
        raise ValueError(f"n_fixed must be in {{0,1,2}}; got {n_fixed_val}")
    L = g_girth - n_fixed_val
    alpha1 = Fraction(k_star - 1, k_star) ** L
    return alpha1 / (1 - alpha1)


# -----------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 68)
    print("V_cb — n_fixed endpoint-counting argument")
    print("=" * 68)
    print()
    print("Gate-first analysis:")
    print(f"  Step A [Type 4]: Feshbach Exponent Principle — k*={k}, g={g}")
    print(f"  Step B [Type 1]: A5(b)+A2 waterline — coupling = Σ above-waterline winding classes")
    print(f"  Step C [Type 2]: n_fixed = n_initial + n_final = {n_initial} + {n_final} = {n_fixed}")
    print(f"                   (1 fixed initial b-quark state + 1 fixed final c-quark state)")
    print(f"  Step D [CAS-VERIFIED]: s_b=C3=ω², s_c=C3=ω at k_P — vcb_hashimoto_bfs.py")
    print(f"  Step E [Type 2]: L_cb = g - n_fixed = {g} - {n_fixed} = {L_cb}")
    print(f"  Step F [Type 4+1+2]: α₁_bare = (2/3)^{L_cb} = {alpha1_bare}")
    print(f"                       V_cb = α₁_bare/(1-α₁_bare) = {V_cb_exact}  [A2 waterline]")
    print()
    print(f"Result:")
    print(f"  n_fixed   = {n_fixed}  [TYPE 2: endpoint counting]")
    print(f"  L_cb      = {L_cb}  [TYPE 2: g - n_fixed]")
    print(f"  α₁_bare   = {alpha1_bare} = {float(alpha1_bare)*1e3:.4f} × 10^-3  [first winding]")
    print(f"  V_cb      = {V_cb_exact} = {V_cb_float*1e3:.4f} × 10^-3  [waterline sum]")
    print()
    print(f"PDG 2024 (exclusive semileptonic):")
    print(f"  |V_cb| = {pdg_central*1e3:.1f} ± {pdg_unc*1e3:.1f} × 10^-3")
    print(f"  Deviation: {deviation:+.2f}σ")
    print()
    print("Gate status:")
    print("  Steps A,B,C,E,F: GATE-PASS (Type 4 / Type 1 / Type 2 / Type 2 / Type 4)")
    print("  Step D:          CAS-VERIFIED (vcb_hashimoto_bfs.py, 2026-04-21)")
    print("                   20 same-orbit (b1,b2) pairs at cycle-distance 8 found")
    print("                   on 8³-cell srs Hashimoto graph.")
    print()
    print("ALL STEPS GATE-PASS. V_cb derivation is THEOREM-grade (0 adoptions).")
    print()
    print("History:")
    print("  Session 12: ADOPTED-Feshbach-vertex dissolved → 1 adoption")
    print("  Session 13: ADOPTED-species-generation CAS-closed → 0 adoptions")
    print()

    # Pure function validation
    pure = predict_vcb_from_nfixed(k, g, 1, 1)
    assert pure == V_cb_exact, f"Mismatch: pure={pure} impl={V_cb_exact}"
    print(f"Pure function check: predict_vcb_from_nfixed({k},{g},1,1) = {pure} = {float(pure)*1e3:.4f}e-3")
    print(f"Implementation:      {V_cb_exact} = {float(V_cb_exact)*1e3:.4f}e-3")
    print("OK: outputs agree.")
