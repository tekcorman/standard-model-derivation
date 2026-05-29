"""
simulator/ — physics-free substrate computer.

ARCHITECTURAL SEPARATION (2026-05-10):
  simulator/  computes observable features of the substrate exhaustively.
              Entirely unaware of physics; no SM observable names anywhere.
              Use this to enumerate what emerges from the substrate alone.

  match/      optional layer pairing substrate outputs with SM observables
              (V_us, m_τ, etc.) and PDG values. Loaded only when SM
              identification is needed.

This package exposes:
- CountingKernel: 7 foundational primitives (walk_count, orbit_count,
  equiv_class_count, mdl_above_waterline, branch_measure, toggle_markov,
  bloch_taylor_at_gamma).
- SrsSubstrate: srs-specific data (k*=3, |V|=4, |E|=6, g=10) + Bloch
  operators + spectral primitives.
- observables: physics-free output catalog that exhaustively dumps every
  substrate quantity the kernel + utilities can derive.

Usage:
    from simulator.srs_engine import CountingKernel
    from simulator.srs_engine.observables import all_substrate_outputs

    catalog = all_substrate_outputs()
    # catalog has every walk survival, every Bloch eigenvalue, every
    # Taylor coefficient, every isotypic multiplicity — without any
    # mention of SM observables.

    # If you want SM identifications + PDG matching, import the
    # match/ package separately:
    #     from match import V_us, get_particle, sm_match_table
"""
import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))


from .srs_substrate import SrsSubstrate
from .kernel import CountingKernel
from . import observables

__all__ = [
    'CountingKernel',
    'SrsSubstrate',
    'observables',
]
