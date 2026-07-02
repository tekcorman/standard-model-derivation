"""
simulator.menus — candidate enumerators (no MDL gating yet).

Each submodule enumerates candidate structures at one layer of the
substrate stack. None of the enumerators apply MDL gating; that's the
responsibility of simulator.gating.

Layers:
- coxeter:         Axis A — Coxeter-quotient relation-structure menu (|E|=2..8 finite + affine + hyperbolic)
- vertex_algebras: local algebra at each vertex (Cl, Cayley-Dickson, magic-square)
- edge_algebras:   edge qubit algebra (Cl(0,p))
- fibers:          high-symmetry k-points (srs slice; other Coxeter TODO)
- readings:        reading-class + walk-class enums (R1-R7, W1-W10)
- crystal_nets:    Axis B — crystal-net spatial-realization menu (srs, srs-z, …) — the
                   framework's ACTUAL substrate candidate set; thin bridge to the mature
                   RCSR / dl_comparison / A2-T-waterfilling apparatus in proofs/foundations/
- gauge_tuples:    Tasks A-E gauge zoo — (substrate, vertex-alg, edge-alg) → gauge group
                   ((srs, Cl(6,0), Cl(0,2)≅ℍ) ⟹ SU(4)×SU(2)_L×SU(2)_R = Pati-Salam) + the
                   subdominant / Layer-1-escape tuples (audited barren — see frontier.layer1_escapes)
- matter:          the matter content — framework-derived Pati-Salam ((4,2,1)⊕(4̄,1,2) per gen
                   from Cl(6,0) Fock; 3 generations from C_3/Galois-ℤ_3 — theorem-grade) + the
                   adopted MSSM extension (≡ R-9's residue; see frontier.mssm_as_adoption)

A "substrate slice" on Axis A is a tuple (Coxeter, VertexAlgebra, EdgeAlgebra) ∈
coxeter × vertex × edge; the simulator.zoo module enumerates this
product and MDL-gates it. On Axis B the candidate set is the RCSR crystal nets
(`crystal_nets`); the framework's substrate is srs (forced STRUCTURALLY — R-9
CLOSED: (A) ⟹ arc-transitive ⟹ Sunada ⟹ srs). The two axes meet at the
observer-side conditioning d_spatial = 3 (Gleason) ⇒ k* = 3 ⇒ |E| = 3
(see simulator.gating.observer).
"""

from .coxeter import CoxeterSystem
from .vertex_algebras import VertexAlgebra
from .edge_algebras import EdgeAlgebra
from .fibers import Fiber
from .readings import ReadingClass, WalkClass
from .crystal_nets import CrystalNet
from .gauge_tuples import GaugeTuple
from .matter import MatterPiece
from .spectral_triple import SpectralTripleChoice
from .beta_contributions import BetaContributionCandidate, ParticleBundle
from .algebras import SubstrateAlgebra, AssociativityAxiom, LoopAxiom
from . import (crystal_nets, gauge_tuples, matter, spectral_triple,
               beta_contributions, algebras)

__all__ = [
    'CoxeterSystem', 'VertexAlgebra', 'EdgeAlgebra',
    'Fiber', 'ReadingClass', 'WalkClass', 'CrystalNet', 'GaugeTuple', 'MatterPiece',
    'SpectralTripleChoice', 'BetaContributionCandidate', 'ParticleBundle',
    'SubstrateAlgebra', 'AssociativityAxiom', 'LoopAxiom',
    'crystal_nets', 'gauge_tuples', 'matter', 'spectral_triple', 'beta_contributions',
    'algebras',
]
