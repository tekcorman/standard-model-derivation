"""
Substrate-output catalog — Axis-A (Coxeter-quotient) slices + Axis-B (crystal-net) realizations.

There are two substrate axes (see `menus.crystal_nets`):

  AXIS A — Coxeter-quotient slices (the zoo: Coxeter × VertexAlgebra ×
  EdgeAlgebra). `all_substrate_outputs(substrate)`:
    • FRAMEWORK SLICE (srs ~ H_3 / Cl(6,0) / Cl(0,2) ≅ ℍ) — the full
      physics-feature catalog (structural counts, walk survivals, Feshbach at
      girth, NB geometric sums, Bloch eigenvalues at high-symmetry k-points,
      the Ramanujan saddle, C₃-isotypic decompositions, Bloch-Taylor coeffs,
      walker-phase windings, Bayesian outputs, Clifford grade dims, polytope
      dihedrals, cubic moments — delegated to the validated `simulator.observables`).
      The spatial substrate here IS the srs crystal net.
    • ANY OTHER ZOO SLICE — the abstract Coxeter-GROUP-graph structural
      invariants (Cay(W(M), S): |V|, degree, girth, diameter, adjacency
      spectrum, closed-walk counts — via `simulator.cayley`) plus the
      vertex/edge algebra facts. These are GROUP-THEORETIC INVARIANTS, NOT a
      spatial substrate: a general |E| Coxeter quotient has no crystal-net
      realization, and that's fine — the framework's substrate is fixed on
      Axis B (the crystal-net DL minimization, where srs wins), not by ranking
      Coxeter quotients. The catalog carries a `not_a_spatial_substrate` note.

  AXIS B — crystal-net realizations (the framework's ACTUAL substrate candidate
  set: srs, srs-z, …). `crystal_net_catalog(name)` returns the per-net
  structural+spectral+DL fingerprint via `menus.crystal_nets` (which bridges to
  the mature RCSR / dl_comparison / A2-T-waterfilling apparatus in
  proofs/foundations/). The framework substrate is srs; R-9 = the open
  srs-vs-srs-z gap.

NB: physics-free. Match-to-SM happens in match/.
"""

from typing import Optional

from .substrate import Substrate
from . import cayley as _cayley
from .menus import crystal_nets as _crystal_nets


def _algebra_facts(substrate: Substrate) -> dict:
    """Physics-free structural facts about the slice's vertex/edge algebras."""
    va, ea = substrate.vertex_algebra, substrate.edge_algebra
    return {
        'vertex_algebra': {
            'name': va.name, 'family': va.family, 'dim_real': va.dim_real,
            'dim_fock': va.dim_fock, 'associative': va.associative,
            'normed': va.normed, 'automorphism': va.automorphism,
        },
        'edge_algebra': {
            'name': ea.name, 'signature': ea.signature, 'dim_real': ea.dim_real,
            'dim_rep': ea.dim_rep, 'automorphism': ea.automorphism,
        },
    }


def _live_framework_catalog() -> dict:
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from .srs_engine.observables import all_substrate_outputs as _live
    return _live()


def all_substrate_outputs(substrate: Optional[Substrate] = None) -> dict:
    """Axis-A substrate-output catalog for `substrate` (default: framework slice).

    Framework slice → full physics-feature catalog (delegated to
    simulator.observables; spatial substrate = the srs crystal net). Any other
    zoo slice → abstract Coxeter-GROUP-graph structural invariants + algebra
    facts + a `not_a_spatial_substrate` note pointing to `menus.crystal_nets`
    (Axis B) for the real substrate-realization layer.
    """
    substrate = substrate if substrate is not None else Substrate.framework_default()
    if substrate.is_framework_slice:
        out = _live_framework_catalog()
        out.setdefault('_slice', {})
        out['_slice'].update({
            'name': substrate.name, 'is_framework_slice': True,
            'coxeter': substrate.coxeter.name,
            'spatial_substrate': 'srs crystal net ((10,3)-a / Laves graph, I4_132, girth 10)',
            **_algebra_facts(substrate),
        })
        return out
    return {
        '_slice': {
            'name': substrate.name, 'is_framework_slice': False,
            'coxeter': substrate.coxeter.name, 'mdl_weight': substrate.weight,
            **_algebra_facts(substrate),
        },
        'coxeter_group_graph_invariants': _cayley.structural_catalog(substrate.coxeter),
        'not_a_spatial_substrate': (
            'the above are invariants of the abstract Coxeter GROUP W(M)\'s '
            'Cayley graph — NOT a spatial substrate. A general |E| Coxeter '
            'quotient has no crystal-net realization; the framework\'s substrate '
            'is fixed on the crystal-net axis (DL minimization → srs), not by '
            'ranking Coxeter quotients. For the real substrate-realization '
            'candidate set (srs, srs-z, …) + fingerprints + DL comparison, see '
            'simulator.menus.crystal_nets and crystal_net_catalog().'),
    }


def crystal_net_catalog(name: str = 'srs') -> dict:
    """Axis-B per-net catalog: structural+spectral+DL fingerprint of a crystal net.

    Delegates to `menus.crystal_nets.rcsr_fingerprint(name)` — which returns
    static metadata (always) plus the live RCSR fingerprint when the RCSR data
    file is present. `name` ∈ {srs, srs-z, srs-c4, srs-c8, srs-c27, lou, lov,
    okw, hcb-c4, ths, dia}. Default 'srs' = the framework substrate.
    """
    return _crystal_nets.rcsr_fingerprint(name)


def crystal_net_dl_comparison() -> dict:
    """Axis-B DL comparison across crystal-net candidates — a CONSISTENCY CHECK.

    Delegates to `menus.crystal_nets.dl_comparison()`. srs is the DL minimum
    among 3-regular crystal nets; but the substrate is *selected* structurally
    (R-9 CLOSED: (A) ⟹ arc-transitive ⟹ Sunada ⟹ srs — see `substrate_selection()`),
    not by this DL comparison. (srs ties srs-z at 12.17 bits under M2a-only
    accounting — that's fine; the structural closure doesn't use the data term.)
    """
    return _crystal_nets.dl_comparison()


def substrate_selection() -> dict:
    """Why srs is the substrate — the R-9 structural closure chain (Axis B).

    Delegates to `menus.crystal_nets.framework_substrate_selection()`:
    (A) ⟹ no privileged direction ⟹ arc-transitive substrate model ⟹ (Sunada
    2012) srs unique. NOT a DL tiebreak; the DL comparison is a consistency
    check. srs-z is the bipartite double cover (the χ̃/SUSY layer), not a rival
    — R-9 ≡ the MSSM-adoption question.
    """
    return _crystal_nets.framework_substrate_selection()


def compare_slices(slice_a: Substrate, slice_b: Substrate) -> dict:
    """Compute Axis-A catalogs for two zoo slices and tabulate differences.

    Compares the abstract Coxeter-GROUP-graph structural invariants (the layer
    that IS computable for arbitrary zoo slices) plus the algebra facts —
    demonstrating that different zoo slices produce different structural numbers.
    For comparing crystal-net realizations (srs vs srs-z, …) use
    `crystal_net_catalog` / `crystal_net_dl_comparison` instead. List-valued
    entries (e.g. adjacency spectra) are compared by summary stats (len/min/max).
    """
    ca = _cayley.structural_catalog(slice_a.coxeter)
    cb = _cayley.structural_catalog(slice_b.coxeter)

    def _summ(v):
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return {'len': len(v), 'min': min(v), 'max': max(v)}
        return v

    keys = sorted(set(ca) | set(cb))
    diff = {}
    for k in keys:
        va, vb = _summ(ca.get(k)), _summ(cb.get(k))
        diff[k] = {'a': va, 'b': vb, 'same': va == vb}
    return {
        'slice_a': slice_a.name, 'slice_b': slice_b.name,
        'coxeter_group_graph_invariants': diff,
        'algebra_facts': {'a': _algebra_facts(slice_a), 'b': _algebra_facts(slice_b)},
        'note': ('compares abstract Coxeter-GROUP-graph invariants (computable for '
                 'any zoo slice) + algebra facts — NOT spatial substrates. For '
                 'crystal-net realization comparison see crystal_net_catalog / '
                 'crystal_net_dl_comparison.'),
    }
