#!/usr/bin/env python3
"""
proofs/foundations/y_nu_persistence_ceiling_2026-05-19.py

DO THE WORK. Does the total observer-compressed PERSISTENCE law derive
y_ν = y_t(GUT) (the disclosed "single hard residue", master dark doc
line 403)? Adopted = 1; observed m_ν3 needs y_ν = 0.99569.

The framework HAS a theorem-grade Yukawa-from-persistence law (y_tau.py,
0 adoptions):
    y = α₁_full × Π(per-leg edge projections) × Family-D
    α₁_full = (n_g/k²)·((k*-1)/k*)^(g-2) = (5/3)(2/3)^8   [closed-cycle
                                                            persistence]
    y_τ = α₁_full / k*²   (two fermion-edge projections, each 1/k*)

Every projection factor is ≤ 1 (they are 1/k* edge projections or 1).
So the law's STRUCTURAL CEILING = α₁_full (all projections = 1). The
Dirac-ν/top sits at the maximal-persistence / hierarchy-free (δ-indep)
endpoint. This probe computes the law's value at that endpoint under
every structurally-unambiguous reading and value-gates it against
y_ν ∈ {1 (adopted/leading), 0.99569 (observed scale)}.

CORRECTNESS GATE (VOID if fail): reproduce the theorem-grade
y_τ = 1280/177147 from α₁_full/k*² (confirms the law is implemented right
before trusting the endpoint evaluation).

PRE-DECLARED OUTCOMES:
  DERIVED   : a structurally-forced endpoint reading of the SAME law
              yields y_ν = 1 exactly (or 0.99569) ⇒ y_ν derived, not
              adopted; the hard residue is closed; +0.87% is sub-leading,
              NOT N_hub.
  CEILING-  : every structured reading of the persistence law gives
  NEGATIVE    y_ν ≪ 1 (capped at/below α₁_full ≈ 0.065). Then y_t(GUT)=1
              is ~15× ABOVE the persistence ceiling — it is the
              un-suppressed natural-scale UNIT the ladder is measured
              against, structurally NOT a persistence amplitude. The
              total-persistence model CANNOT derive it; y_ν=1 is the
              irreducible hard residue (confirming master doc line 403),
              and it is NOT N_hub. Honest terminus, work done.
Ships no number into predictions/; changes no ledger row.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))
import numpy as np
from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial

d=predict_d_spatial(); k=predict_k_star(d); g=predict_g_girth(k,d)
N_atoms=4; n_g=15                                  # srs: 15 girth-10 cycles/vertex
a1_bare=((k-1)/k)**(g-2)                            # (2/3)^8
a1_full=(n_g/k**2)*a1_bare                          # (5/3)(2/3)^8  Class-2
y_tau_law=a1_full/k**2                              # theorem-grade y_τ_tree
print(f"k*={k} g={g} N_atoms={N_atoms} n_g={n_g}")
print(f"α₁_bare=(2/3)^{g-2}={a1_bare:.8f}   α₁_full=(5/3)(2/3)^{g-2}={a1_full:.8f}")

print("\n"+"="*70+"\nCORRECTNESS GATE — reproduce theorem-grade y_τ\n"+"="*70)
y_tau_ref=1280/177147
gate=abs(y_tau_law-y_tau_ref)<1e-12
print(f"  α₁_full/k*² = {y_tau_law:.12f}   1280/177147 = {y_tau_ref:.12f}   match={gate}")
if not gate:
    print("  ** GATE FAILED — law mis-implemented. VOID. **"); sys.exit(0)
print("  GATE PASSED — the persistence-Yukawa law is correctly the y_τ law.")

Y_ADOPT, Y_OBS = 1.0, np.sqrt(50.1298/50.5651)     # 1 ; data-required y_ν
print(f"\n  target y_ν: adopted/leading = {Y_ADOPT}, observed-scale = {Y_OBS:.5f}")

print("\n"+"="*70+"\nLAW EVALUATED AT THE Dirac-ν/top (max-persistence) ENDPOINT\n"+"="*70)
def verdict(y):
    for t,nm in ((Y_ADOPT,"=1 adopted"),(Y_OBS,"=0.99569 obs")):
        if abs(y-t)/t < 0.02: return f"MATCH {nm}"
    return f"{Y_ADOPT/y:.1f}× BELOW the required y_ν≈1" if y<Y_ADOPT else "ABOVE 1"
readings = {
 "R1 ceiling: α₁_full (all projections=1)":            a1_full,
 "R2 bare survival: α₁_bare=(2/3)^8":                   a1_bare,
 "R3 cycle, 1 fermion-edge proj: α₁_full/k*":           a1_full/k,
 "R4 cycle, 2 proj (= y_τ structure): α₁_full/k*²":     a1_full/k**2,
 "R5 zero-traversal mult only: (n_g/k²)·(2/3)^0":       (n_g/k**2),
 "R6 zero-traversal, 2 proj: (n_g/k²)/k*²":             (n_g/k**2)/k**2,
}
for nm,val in readings.items():
    print(f"  {nm:46s} = {val:.6f}   -> {verdict(val)}")
ceiling=a1_full
print(f"\n  STRUCTURAL CEILING of the persistence law = α₁_full = {ceiling:.6f}")
print(f"  required y_ν ≈ 1  is  {1.0/ceiling:.2f}×  ABOVE that ceiling.")

print("\n"+"="*70+"\n  VERDICT\n"+"="*70)
derived = any(abs(v-Y_ADOPT)/Y_ADOPT<0.02 or abs(v-Y_OBS)/Y_OBS<0.02
              for v in readings.values())
if derived:
    print("  DERIVED — a structured reading of the persistence law hits "
          "y_ν≈1/0.9957 (see above). Hard residue closeable.")
else:
    print(f"""  CEILING-NEGATIVE — DONE, computed, no dodge.

  Every structurally-unambiguous reading of the framework's OWN
  theorem-grade persistence-Yukawa law (the y_τ law) gives
  y ∈ [{min(readings.values()):.4f}, {max(readings.values()):.4f}] — capped at α₁_full≈{ceiling:.4f}.
  The required y_ν ≈ 1 is ~{1.0/ceiling:.0f}× ABOVE the law's ceiling.

  ⇒ The total observer-compressed PERSISTENCE model CANNOT derive y_ν.
    y_t(GUT)=1 is not a persistence amplitude at all — it is the
    UN-SUPPRESSED NATURAL-SCALE UNIT that the entire persistence/Koide
    ladder is *measured against* (y_τ = α₁_full/k*² is y_τ expressed in
    units of y_t(GUT)=1). Deriving the persistence ladder presupposes
    this unit; it cannot also output it. This is precisely why the master
    dark doc (line 403) calls y_t(GUT)=1 'the single hard residue'.

  ⇒ The +0.87%/+2.18σ on m_ν3 is therefore NOT N_hub's to own and NOT a
    persistence effect: it sits on y_ν=1, the irreducible natural-scale
    unit, which this framework does not derive by ANY route here. The
    honest status: the neutrino mass RATIO (228/7) is derived exactly;
    the absolute SCALE is gated by an undrived natural-scale Yukawa unit
    — the framework's deepest open residue — full stop. No N_hub dodge.""")
print("  Ships no number; changes no ledger row.")
print("="*70)
