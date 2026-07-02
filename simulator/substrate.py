"""
Substrate — tuple of (Coxeter system, vertex algebra, edge algebra) +
derived Cayley graph + Bloch decomposition.

A Substrate instance is one slice of the substrate menu. The simulator's
kernel operates on a Substrate (it doesn't hardcode srs). The framework's
empirical slice is obtained via `Substrate.framework_default()` (which
applies the observer-side d=3 / edge-transitivity conditioning — see
`gating.cooling.srs_slice`); the raw-MDL-top slice is `Substrate.dominant_at`.

Replaces the current SrsSubstrate (which hardcoded the dominant slice). For
the framework slice, the substrate-level Bloch operations delegate to the
existing simulator.srs_substrate.SrsSubstrate; for other slices they raise
NotImplementedError until the per-Coxeter Cayley-graph builders are wired.
"""

from dataclasses import dataclass, field
from typing import Optional

from .menus.coxeter import CoxeterSystem
from .menus.vertex_algebras import VertexAlgebra
from .menus.edge_algebras import EdgeAlgebra


@dataclass(frozen=True)
class Substrate:
    """One substrate slice from the saturated zoo.

    Attributes:
        coxeter        : Coxeter system M (substrate Cayley graph)
        vertex_algebra : local algebra at each vertex (Cl(2k,0), 𝕆, …)
        edge_algebra   : local algebra at each directed edge (Cl(0,p))
        name           : human-readable identifier
        weight         : MDL combined weight at framework scale (cached; 0 if unknown)
        is_framework_slice : True iff this is the srs / Cl(6,0) / Cl(0,2) slice
    """
    coxeter: CoxeterSystem
    vertex_algebra: VertexAlgebra
    edge_algebra: EdgeAlgebra
    name: str = ''
    weight: float = 0.0
    is_framework_slice: bool = field(default=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_tuple(cls, cox: CoxeterSystem, vert: VertexAlgebra,
                   edge: EdgeAlgebra, weight: float = 0.0) -> 'Substrate':
        """Wrap a (coxeter, vertex, edge) tuple (e.g. from the zoo) as a Substrate."""
        is_fw = ('srs' in cox.name and vert.name == 'Cl(6,0)'
                 and edge.signature == (0, 2))
        return cls(coxeter=cox, vertex_algebra=vert, edge_algebra=edge,
                   name=f'{cox.name} × {vert.name} × {edge.name}',
                   weight=weight, is_framework_slice=is_fw)

    @classmethod
    def framework_default(cls) -> 'Substrate':
        """The framework's empirical slice: (srs ~ H_3, Cl(6,0), Cl(0,2) ≅ ℍ).

        This is the observer-side-conditioned dominant slice (Gleason d=3 ⇒
        |E| = k* = 3; crystal-net edge-transitivity Sunada-uniqueness; Cl(2k,0)
        at k=3; G_2 edge-qubit theorem), NOT the argmax of the raw substrate-
        only combined weight. See gating.cooling.srs_slice / saturated_zoo's note.
        """
        from .gating.cooling import srs_slice
        cox, vert, edge = srs_slice()
        s = cls.from_tuple(cox, vert, edge)
        return cls(coxeter=cox, vertex_algebra=vert, edge_algebra=edge,
                   name='framework slice (srs × Cl(6,0) × Cl(0,2)≅ℍ)',
                   weight=s.weight, is_framework_slice=True)

    @classmethod
    def dominant_at(cls, N: float) -> 'Substrate':
        """The top-ranked slice in the saturated zoo at observation length N.

        NB — raw-MDL ranking: at framework scale this is *not* the srs slice
        (substrate-only MDL prefers higher |E| / exceptional Coxeter once the
        frequency penalty is inactive). For the framework's empirical slice
        use `framework_default()`. This stays available so the honest
        substrate-only ranking is inspectable.
        """
        from . import zoo
        return zoo.dominant_slice(N)

    @classmethod
    def from_names(cls, coxeter_name: str, vertex_name: str,
                   edge_name: str) -> 'Substrate':
        """Construct a Substrate by named lookup in the candidate menus.

        Used to instantiate specific subdominant slices for substrate-
        comparison studies. Names must match a menu entry's `.name` exactly
        (or be a substring of exactly one entry).
        """
        from .menus import coxeter as cox_menu
        from .menus import vertex_algebras as vert_menu
        from .menus import edge_algebras as edge_menu

        def _lookup(items, name, label):
            exact = [it for it in items if it.name == name]
            if exact:
                return exact[0]
            sub = [it for it in items if name in it.name]
            if len(sub) == 1:
                return sub[0]
            if not sub:
                raise ValueError(f"from_names: no {label} named {name!r}; "
                                 f"available: {[it.name for it in items]}")
            raise ValueError(f"from_names: {label} name {name!r} is ambiguous: "
                             f"{[it.name for it in sub]}")

        cox = _lookup(cox_menu.enumerate_full_menu(), coxeter_name, 'Coxeter system')
        vert = _lookup(vert_menu.enumerate_full_menu(), vertex_name, 'vertex algebra')
        edge = _lookup(edge_menu.enumerate_full_menu(), edge_name, 'edge algebra')
        return cls.from_tuple(cox, vert, edge)

    # ------------------------------------------------------------------
    # Derived structural data
    # ------------------------------------------------------------------

    @property
    def structural_counts(self) -> dict:
        """k* (= vertex coordination), |E| (generators), girth-class, etc.

        For the framework slice these come from simulator.srs_substrate; for
        other slices only the Coxeter-derivable quantities (k* ≈ rank, |E|)
        are available.
        """
        if self.is_framework_slice:
            srs = self._srs()
            return {
                'k_star': srs.K_STAR, 'n_atoms': srs.N_ATOMS,
                'n_edges': srs.N_EDGES, 'n_directed': srs.N_DIRECTED,
                'girth': srs.GIRTH, 'd_spatial': srs.D_SPATIAL,
            }
        return {
            'k_star': self.coxeter.rank,
            'n_generators': self.coxeter.generators,
            'coxeter_order': self.coxeter.order,
        }

    # ------------------------------------------------------------------
    # Substrate-level Bloch operations (framework slice → SrsSubstrate)
    # ------------------------------------------------------------------

    def _srs(self):
        if not self.is_framework_slice:
            raise NotImplementedError(
                f"Bloch operations for slice {self.name!r} not wired — only the "
                "framework (srs) slice delegates to simulator.srs_substrate. "
                "Per-Coxeter Cayley-graph builders are a TODO in the rebuild.")
        from .srs_bridge import srs_substrate
        return srs_substrate()

    def adjacency_at_k(self, k_frac):
        """Bloch adjacency at fractional k. Framework slice → SrsSubstrate."""
        return self._srs().adjacency_at_k(k_frac)

    def hashimoto_at_k(self, k_frac):
        """Bloch Hashimoto operator at fractional k. Framework slice → SrsSubstrate."""
        return self._srs().hashimoto_at_k(k_frac)

    def adjacency_spectrum_at_k(self, k_frac):
        """Sorted real eigenvalues of adjacency_at_k. Framework slice → SrsSubstrate."""
        return self._srs().adjacency_spectrum_at_k(k_frac)
