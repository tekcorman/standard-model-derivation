"""
simulator — the unified simulator pipeline (in build).

S0 axioms → S1 ENUMERATE (menus/) → S2 MDL-GATE (gating/) → S3 COMPUTE
(kernel + observables) → S4 MATCH (match/, separate package for now) → S5 verify
(proofs/**, ledger — they consume this) ; + `frontier` (the boundary). See
an internal working note for the absorb plan,
`README.md` for the rebuild status. Validation:
`proofs/foundations/simulator_validation.py`.

The pattern (not "hardcoded srs + a curated prediction list"): enumerate the
candidate space (Axis A Coxeter quotients × Axis B crystal-net realizations ×
vertex/edge algebras), MDL-gate it (Stage-1 waterline + Stage-2 channel
selection), apply observer-side conditioning (Gleason d=3 ⇒ k*=3; (A)'s
no-privilege ⇒ arc-transitive ⇒ Sunada ⇒ srs), and compute the observable
catalog for the conditioned slice.

Top-level API:

    from simulator import axioms, CountingKernel, Substrate, zoo, observables, frontier
    from simulator.gating import observer

    axioms.summary()                       # the {(A),(B),(I),A5-mass} slate + derived + adopted
    observer.conditioned_substrate()       # the observer-conditioned slice: |E|=3 region + srs
    fw = zoo.framework_slice()             # srs × Cl(6,0) × Cl(0,2) — what match/ consumes
    catalog = observables.all_substrate_outputs(fw)   # full physics catalog
    observables.substrate_selection()      # why srs (R-9 closure: (A)⟹arc-transitive⟹Sunada)
    frontier.list_gaps()                   # the ~11 genuine open gaps (R-9 is CLOSED, not here)

    # raw substrate-only MDL (the "skeptical bridge" — NOT srs): zoo.dominant_slice()
    # a subdominant zoo slice → Coxeter-GROUP-graph invariants only:
    #   observables.all_substrate_outputs(Substrate.from_names('H_4', '𝕆 (octonion)', 'Cl(0,2)'))
"""

from .substrate import Substrate
from .kernel import CountingKernel
from . import (axioms, menus, gating, zoo, observables, cayley, frontier,
               cosmology)

__all__ = ['axioms', 'Substrate', 'CountingKernel', 'menus', 'gating', 'zoo',
           'observables', 'cayley', 'frontier', 'cosmology']
