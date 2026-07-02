"""
The substrate zoo — enumerate × MDL-gate × emit retained slices.

This is the top-level entrypoint the simulator is built around. The pattern:

    1. ENUMERATE: Cartesian product of menus
       (Coxeter × VertexAlgebra × EdgeAlgebra)
    2. GATE: MDL waterline + frequency support per slice at N  (Stage 1)
    3. RETAIN: slices with combined weight ≥ threshold
    4. RANK: descending by combined weight

The top-ranked slice on the raw substrate-only metric (`dominant_slice`) is
NOT srs — substrate-only MDL prefers higher |E| / exceptional Coxeter once
the frequency penalty is inactive (this is the honest finding of
sector_coxeter_full_menu_ranking_audit.py). The framework's empirical
substrate (`framework_slice`) is the srs slice, which is singled out by the
*observer-side bridge* (Gleason d=3 ⇒ |E| = k* = 3; crystal-net edge-
transitivity Sunada-uniqueness; Cl(2k,0) at k=3; G_2 edge-qubit theorem) —
a separate, much stronger conditioning not folded into the raw ranking.

Subdominant slices are plurally co-retained per A2-T and produce their own
observable catalogs (most match no observed physics — that's the content).

Source pattern: Tasks A-E (commits 2c2a624 → d5bdc45, 2026-05-07) +
sector_coxeter_full_menu_ranking_audit.py + sector_cooling_cascade_audit.py.
"""

from typing import Optional

from .menus.coxeter import enumerate_full_menu as _enumerate_coxeter
from .menus.vertex_algebras import enumerate_full_menu as _enumerate_vertex
from .menus.edge_algebras import enumerate_full_menu as _enumerate_edge
from .gating import cooling as _cooling
from .gating.cooling import N_HUB_DEFAULT
from .substrate import Substrate


# ---------------------------------------------------------------------------
# Default menus (cached at module level)
# ---------------------------------------------------------------------------

def default_coxeter_menu() -> list:
    return _enumerate_coxeter()


def default_vertex_menu() -> list:
    return _enumerate_vertex()


def default_edge_menu() -> list:
    return _enumerate_edge()


# ---------------------------------------------------------------------------
# Enumeration / gating
# ---------------------------------------------------------------------------

def enumerate_all_slices(coxeter_E_max: int = 8, vertex_k_max: int = 8,
                         vertex_depth_max: int = 5, edge_p_max: int = 4,
                         require_compatible: bool = True) -> list[tuple]:
    """Cartesian product of candidate menus — no MDL gating applied yet.

    Returns list of (CoxeterSystem, VertexAlgebra, EdgeAlgebra) tuples. Use
    `saturated_zoo` / `cooling.retained_at` to filter to the retained zoo.
    """
    cox_menu = _enumerate_coxeter(coxeter_E_max)
    vert_menu = _enumerate_vertex(vertex_k_max, vertex_depth_max)
    edge_menu = _enumerate_edge(edge_p_max)
    out = []
    for cox in cox_menu:
        for vert in vert_menu:
            if require_compatible and not _cooling._is_compatible(cox, vert):
                continue
            for edge in edge_menu:
                out.append((cox, vert, edge))
    return out


def saturated_zoo(N: float = N_HUB_DEFAULT, threshold: float = 0.0,
                  coxeter_menu: Optional[list] = None,
                  vertex_menu: Optional[list] = None,
                  edge_menu: Optional[list] = None) -> list[Substrate]:
    """Full plurally-retained substrate zoo at observation length N.

    Returns Substrate instances sorted by combined MDL weight (descending).
    The first entry is the raw-MDL top slice (see `dominant_slice` caveat).
    """
    cox_menu = coxeter_menu if coxeter_menu is not None else default_coxeter_menu()
    vert_menu = vertex_menu if vertex_menu is not None else default_vertex_menu()
    edge_menu = edge_menu if edge_menu is not None else default_edge_menu()
    tuples = _cooling.saturated_zoo(cox_menu, vert_menu, edge_menu, N, threshold)
    return [Substrate.from_tuple(cox, vert, edge, weight=w)
            for (cox, vert, edge, w) in tuples]


def dominant_slice(N: float = N_HUB_DEFAULT) -> Substrate:
    """The top-ranked Substrate by RAW substrate-only combined weight at N.

    Caveat: this is NOT srs. On the freq-weighted audit's metric (Φ − ΣL +
    freq), once the frequency penalty is inactive at framework scale, higher
    |E| / exceptional Coxeter systems compress more of the substrate stream
    and rank above |E|=3 — sector_coxeter_full_menu_ranking_audit.py's
    finding, reproduced here. For the framework's empirical slice, use
    `framework_slice()`.
    """
    zoo = saturated_zoo(N)
    if not zoo:
        raise ValueError("dominant_slice: no slices above the waterline at N")
    return zoo[0]


def framework_slice() -> Substrate:
    """The framework's empirical slice: (srs ~ H_3, Cl(6,0), Cl(0,2) ≅ ℍ).

    Picked by the observer-side bridge, not the raw substrate-only argmax.
    The match package consumes this slice's observable catalog.
    """
    return Substrate.framework_default()


def subdominant_zoo(N: float = N_HUB_DEFAULT, k: int = 10) -> list[Substrate]:
    """Top-k retained slices after the raw-MDL top slice."""
    zoo = saturated_zoo(N)
    return zoo[1:1 + k]


def cooling_cascade_table(N_samples: Optional[list[float]] = None) -> dict:
    """Cooling cascade across N samples: combined weight per slice per N.

    Default N_samples = [1e3, 1e4, 1e5, 1e6, 1e9, 1e60] mirrors the existing
    sector_cooling_cascade_audit table. Returns
      { (coxeter_name, vertex_name, edge_name): { N: combined_weight } }.
    """
    return _cooling.cooling_cascade_table(
        default_coxeter_menu(), default_vertex_menu(), default_edge_menu(),
        N_samples)


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

def _demo():
    import math
    print("=" * 96)
    print(" simulator.zoo — enumerate × MDL-gate × emit retained slices")
    print("=" * 96)
    cox = default_coxeter_menu()
    vert = default_vertex_menu()
    edge = default_edge_menu()
    print(f" Coxeter menu  : {len(cox)} systems  "
          f"({sum(1 for c in cox if c.growth_class=='finite')} finite, "
          f"{sum(1 for c in cox if c.growth_class=='affine')} affine, "
          f"{sum(1 for c in cox if c.growth_class=='hyperbolic')} multi-gen, "
          f"{sum(1 for c in cox if c.growth_class=='free')} free)")
    print(f" Vertex menu   : {len(vert)} algebras")
    print(f" Edge menu     : {len(edge)} algebras")
    raw_product = len(cox) * len(vert) * len(edge)
    compat = enumerate_all_slices()
    print(f" Raw product   : {raw_product};  k-compatible slices: {len(compat)}")
    print()

    N = N_HUB_DEFAULT
    zoo = saturated_zoo(N)
    print(f" Slices above the waterline at N_hub ≈ {N:.2e}: {len(zoo)}")
    print()
    print(" Top 12 by raw substrate-only combined weight:")
    print(f"   {'rank':<5}{'weight':>16}  slice")
    for i, s in enumerate(zoo[:12], 1):
        w = s.weight
        ws = (f"+10^{math.log10(w):.2f}" if w > 1e9 else f"{w:+.2f}")
        print(f"   {i:<5}{ws:>16}  {s.name}")
    print()
    # Where does the framework (|E|=k*=3, H_3-region) slice land?
    fw_rank = next((i for i, s in enumerate(zoo, 1)
                    if 'H_3' in s.coxeter.name and s.vertex_algebra.name == 'Cl(6,0)'
                    and s.edge_algebra.signature == (0, 2)), None)
    if fw_rank is not None:
        print(f" The H_3 (|E|=3) × Cl(6,0) × Cl(0,2)≅ℍ slice is rank {fw_rank} of {len(zoo)}")
        print(" by raw substrate-only weight — i.e. substrate-only MDL does NOT")
        print(" single out the framework's |E|=k*=3 region. srs is the edge-")
        print(" transitive representative within it; it is the framework slice")
        print(" via the observer-side bridge (Gleason d=3 ⇒ |E|=k*=3; crystal-")
        print(" net edge-transitivity Sunada-uniqueness), not the raw MDL argmax.")
    fw = framework_slice()
    print()
    print(f" framework_slice() = {fw.name}")
    print(f"   structural counts: {fw.structural_counts}")
    print("=" * 96)


if __name__ == '__main__':
    _demo()
