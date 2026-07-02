"""
Cooling cascade — retention-vs-observation-length profile (Stage 1 gating).

For any menu of slice candidates, the cooling cascade traces which slices
clear the MDL waterline (Stage 1) as observation length N decreases below
per-slice N_attest. As N grows toward framework scale (N_hub ≈ 8.4×10⁶⁰),
progressively more slices enter the attested zoo.

This module is STAGE 1 only — it produces the set of physically-realized
slices at scale N. STAGE 2 (which retained slice / physical realization a
given observable reads) is `gating.mdl.channel_select`, applied at the
prediction layer (match package). The dominant slice (top-ranked here) is
the framework's empirical substrate; subdominant slices are plurally
co-retained and produce their own observable catalogs (most match no
observed physics — that's the honest content).

Source pattern: proofs/foundations/sector_cooling_cascade_audit.py (Task D,
commit 51edbc8) + sector_coxeter_full_menu_ranking_audit.py.

Outputs:
- cooling_profile(menu, N_samples)  — combined weights per slice per N
- retained_at(menu, N, threshold=0) — slices above waterline at scale N (Stage 1)
- saturated_zoo(...)                — full retained zoo at framework scale
- dominant_slice(...)               — top-ranked slice (see honest caveat below)
- subdominant_zoo(...)              — top-k slices after the dominant
"""

import math
from typing import Iterable, Optional

from ..menus.coxeter import CoxeterSystem
from ..menus.vertex_algebras import VertexAlgebra
from ..menus.edge_algebras import EdgeAlgebra
from . import mdl


# Framework-scale anchor — N_hub, the framework's adopted dimensional input (≈8.39e60).
N_HUB_DEFAULT = 8.394881e60


# ---------------------------------------------------------------------------
# Coxeter-only cooling profile
# ---------------------------------------------------------------------------

def cooling_profile(menu: Iterable[CoxeterSystem], N_samples: list[float]) -> dict:
    """For each Coxeter system in `menu`, W(M, N) at each N in N_samples.

    Returns { system_name: [(N_1, W_1), …, (N_k, W_k)] }.
    """
    return {cs.name: [(N, mdl.combined_weight(cs, N)) for N in N_samples]
            for cs in menu}


def retained_at(menu: Iterable, N: float = N_HUB_DEFAULT,
                threshold: float = 0.0) -> list:
    """Slices with combined_weight ≥ threshold at observation length N.

    `menu` may be CoxeterSystem instances or (coxeter, vertex, edge) tuples.
    Default N is framework scale; default threshold 0 retains all positive-
    Bayesian-weight slices. Returns list of (candidate, weight), descending.
    """
    return mdl.retained_above_waterline(menu, N, threshold)


# ---------------------------------------------------------------------------
# Slice-level (substrate × vertex × edge) zoo
# ---------------------------------------------------------------------------

def _is_compatible(coxeter: CoxeterSystem, vertex: VertexAlgebra) -> bool:
    """Vertex algebra compatible with the substrate's vertex coordination.

    Cl(2k,0) requires k = vertex coordination. We take vertex coordination =
    Coxeter rank for finite irreducible systems (= |E| for A_n etc.; = 3 for
    H_3 / srs). Cayley-Dickson / magic-square algebras carry empty k_compat
    (no coordination constraint — they're compatibility-permissive label
    candidates, as in the cooling-cascade audit's representative tuples).
    """
    if not vertex.k_compat:
        return True
    return coxeter.rank in vertex.k_compat


def saturated_zoo(coxeter_menu: list[CoxeterSystem],
                  vertex_menu: list[VertexAlgebra],
                  edge_menu: list[EdgeAlgebra],
                  N_hub: float = N_HUB_DEFAULT,
                  threshold: float = 0.0,
                  require_compatible: bool = True
                  ) -> list[tuple]:
    """Cartesian product of menus, MDL-gate at N_hub, return retained tuples.

    Returns list of (coxeter, vertex, edge, weight) sorted by weight descending.
    Per Task E (commit d5bdc45): the top-ranked tuple is the dominant slice;
    subsequent tuples are plurally co-retained.

    Honest note: the ranking here is the *substrate-only* combined weight
    (Φ − ΣL + freq) of the freq-weighted audit. At N_hub the frequency
    penalty is essentially never active for any m ≤ ~30 system, so Φ
    dominates and larger |W(M)| (higher |E|, exceptional Coxeter) ranks
    higher — i.e. substrate-only MDL does NOT, on this metric, single out
    srs / |E|=3. The framework's k* = 3 / srs slice comes from the
    *observer-side bridge* (Gleason d = 3 ⇒ |E| = k* = 3, the crystal-net
    edge-transitivity Sunada-uniqueness conditioning, etc.), which is a
    separate, much stronger conditioning not folded into this raw ranking.
    See `dominant_slice` and `srs_slice`.
    """
    out = []
    for cox in coxeter_menu:
        for vert in vertex_menu:
            if require_compatible and not _is_compatible(cox, vert):
                continue
            for edge in edge_menu:
                w = mdl.slice_combined_weight(cox, vert, edge, N_hub)
                if mdl.above_waterline(w, threshold):
                    out.append((cox, vert, edge, w))
    out.sort(key=lambda t: -t[3])
    return out


def dominant_slice(coxeter_menu: list[CoxeterSystem],
                   vertex_menu: list[VertexAlgebra],
                   edge_menu: list[EdgeAlgebra],
                   N_hub: float = N_HUB_DEFAULT) -> tuple:
    """The top-ranked (coxeter, vertex, edge) tuple in the saturated zoo.

    Returns (coxeter, vertex, edge) — the top of `saturated_zoo`. On the raw
    substrate-only combined-weight metric this is NOT srs / |E|=3 (see the
    `saturated_zoo` honest note); to get the framework's empirical slice,
    use `srs_slice` (which applies the observer-side d=3 / edge-transitivity
    conditioning) or pass a coxeter_menu restricted to the |E| = k* = 3
    sub-menu.
    """
    zoo = saturated_zoo(coxeter_menu, vertex_menu, edge_menu, N_hub)
    if not zoo:
        raise ValueError("dominant_slice: no slices above the waterline")
    cox, vert, edge, _w = zoo[0]
    return (cox, vert, edge)


def subdominant_zoo(coxeter_menu: list[CoxeterSystem],
                    vertex_menu: list[VertexAlgebra],
                    edge_menu: list[EdgeAlgebra],
                    N_hub: float = N_HUB_DEFAULT,
                    k: int = 10) -> list[tuple]:
    """Top-k subdominant tuples after the dominant slice (incl. weights)."""
    zoo = saturated_zoo(coxeter_menu, vertex_menu, edge_menu, N_hub)
    return zoo[1:1 + k]


def srs_slice() -> tuple:
    """The framework's empirical (observer-side-conditioned) dominant slice.

    (srs ~ H_3-like, Cl(6,0), Cl(0,2) ≅ ℍ). This is *not* the argmax of the
    raw substrate-only combined weight (see `saturated_zoo`); it is the slice
    picked once the observer-side bridge is conditioned in:
      - Gleason 1957 ⇒ d_spatial = 3 ⇒ vertex coordination k* = 3 ⇒ |E| = 3
      - among |E| = 3 3-regular crystal nets, srs is Sunada-unique as the
        edge-transitive 3-connected one ⇒ huge compression once conditioned
      - Cl(2k,0) at k=3 ⇒ Cl(6,0); G_2 edge-qubit theorem ⇒ Cl(0,2) ≅ ℍ.
    The match package consumes this slice's observable catalog.
    """
    from ..menus.coxeter import srs_equivalent
    from ..menus.vertex_algebras import framework_dominant as vertex_dominant
    from ..menus.edge_algebras import framework_dominant as edge_dominant
    return (srs_equivalent(), vertex_dominant(), edge_dominant())


def cooling_cascade_table(coxeter_menu: list[CoxeterSystem],
                          vertex_menu: list[VertexAlgebra],
                          edge_menu: list[EdgeAlgebra],
                          N_samples: Optional[list[float]] = None,
                          require_compatible: bool = True) -> dict:
    """Cooling cascade across N samples: combined weight per slice per N.

    Default N_samples = [1e3, 1e4, 1e5, 1e6, 1e9, 1e60] mirrors the existing
    sector_cooling_cascade_audit table. Returns
      { (coxeter_name, vertex_name, edge_name): { N: combined_weight } }.
    """
    if N_samples is None:
        N_samples = [1e3, 1e4, 1e5, 1e6, 1e9, 1e60]
    table = {}
    for cox in coxeter_menu:
        for vert in vertex_menu:
            if require_compatible and not _is_compatible(cox, vert):
                continue
            for edge in edge_menu:
                key = (cox.name, vert.name, edge.name)
                table[key] = {N: mdl.slice_combined_weight(cox, vert, edge, N)
                              for N in N_samples}
    return table
