"""
Bridge to the live simulator's srs substrate.

The rebuild parameterizes the kernel over a `Substrate` slice; for the
framework slice (srs ~ H_3, Cl(6,0), Cl(0,2)) the substrate-level Bloch
operations delegate to the existing, already-validated
`simulator.srs_substrate.SrsSubstrate` rather than re-implementing the
Cayley-graph machinery. Other slices need per-Coxeter Cayley-graph builders
(a TODO in the rebuild).
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_srs_singleton = None


def srs_substrate():
    """Return a (cached) live simulator.srs_substrate.SrsSubstrate instance."""
    global _srs_singleton
    if _srs_singleton is None:
        from .srs_engine.srs_substrate import SrsSubstrate
        _srs_singleton = SrsSubstrate()
    return _srs_singleton
