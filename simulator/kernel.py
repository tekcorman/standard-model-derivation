"""
Counting kernel — primitives, generalized over Substrate slices.

This kernel takes a `Substrate` parameter (from simulator.zoo or
Substrate.framework_default / Substrate.dominant_at). The counting
primitives are unchanged in interface; what changes is that they operate on
the slice's Cayley graph / Bloch operators / local algebras rather than
hardcoded srs. For the framework slice the counting primitives delegate to
the already-validated `simulator.CountingKernel` (the rebuild does NOT
re-derive what the live kernel already computes); for other slices they
raise NotImplementedError until per-Coxeter Cayley-graph builders land.

PRIMITIVE INVENTORY (post-2026-05-12 MDL cleanup):

  Counting primitives (substrate enumeration):
    1. walk_count(walk_type, length)
    2. orbit_count(group_action)
    3. equiv_class_count(equivalence_relation)
    4. branch_measure(walk_class, length)
    5. toggle_markov(n_steps)
    6. bloch_taylor_at_gamma(order, prec)

  MDL primitives (waterline gating):
    7. mdl_above_waterline(model_bits, data|model, raw_bits) → bool
         The waterline THRESHOLD test (Stage 1). Says only "above or below".
    8. channel_select(candidates, channel) → the matching candidate
         The WATERFILLING-CORRECT selection (Stage 2). All candidates above
         the waterline are physically realized; for ONE observable the
         channel is fixed by a STRUCTURAL argument (the observable's
         substrate definition); channel_select picks the candidate whose
         `channel` field matches. K-equivalent matches → min-bit-cost rep.
    9. canonical_encoding(equivalence_class) → canonical representative
         Min-description-length representative of a class of K-EQUIVALENT
         encodings (all give the same numerical value). Distinct from
         channel_select (which spans physically-distinct K-rational candidates).

  RETIRED:
    mdl_select(candidates) — argmin over total bit cost. RETRACTED 2026-05
    per feedback_waterline_not_minimum_canonical_distinction. Kept for
    backwards-compat of audit checks only.

The MDL primitives delegate to simulator.gating.mdl (which has the
reference implementations, matching simulator/kernel.py on main).
"""

from typing import Optional

from .substrate import Substrate
from .gating import mdl as _mdl


class CountingKernel:
    """Counting kernel parameterized over substrate slices.

    Usage:
        # Framework slice (srs ~ H_3, Cl(6,0), Cl(0,2)):
        kernel = CountingKernel()                         # = framework slice
        kernel = CountingKernel(Substrate.framework_default())

        # Raw-MDL top slice (NOT srs — see zoo.dominant_slice):
        kernel = CountingKernel(Substrate.dominant_at(N_hub))

        # A specific subdominant slice:
        kernel = CountingKernel(Substrate.from_names(
            coxeter_name='H_4', vertex_name='𝕆 (octonion)',
            edge_name='Cl(0,2) ≅ ℍ edge'))
    """

    def __init__(self, substrate: Optional[Substrate] = None):
        self.substrate = substrate if substrate is not None else Substrate.framework_default()
        self._live = None  # lazily-bound live simulator.CountingKernel (framework slice only)

    # ------------------------------------------------------------------
    # Counting primitives — delegate to the live kernel for the framework slice
    # ------------------------------------------------------------------

    def _live_kernel(self):
        if not self.substrate.is_framework_slice:
            raise NotImplementedError(
                f"counting primitives for slice {self.substrate.name!r} not "
                "wired — only the framework (srs) slice delegates to the live "
                "simulator.CountingKernel. Per-Coxeter Cayley-graph builders "
                "are a TODO in the rebuild.")
        if self._live is None:
            import sys
            from pathlib import Path
            repo = Path(__file__).resolve().parents[1]
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from .srs_engine import CountingKernel as _LiveKernel
            self._live = _LiveKernel()
        return self._live

    def walk_count(self, walk_type: str, length: Optional[int] = None, exact: bool = True):
        return self._live_kernel().walk_count(walk_type, length=length, exact=exact)

    def orbit_count(self, group_action: str, orbit_class=None):
        return self._live_kernel().orbit_count(group_action, orbit_class=orbit_class)

    def equiv_class_count(self, equivalence_relation: str):
        return self._live_kernel().equiv_class_count(equivalence_relation)

    def branch_measure(self, walk_class: str, length: Optional[int] = None):
        return self._live_kernel().branch_measure(walk_class, length=length)

    def toggle_markov(self, n_steps: Optional[int] = None):
        return self._live_kernel().toggle_markov(n_steps=n_steps)

    def bloch_taylor_at_gamma(self, order: int = 4, prec: int = 300):
        return self._live_kernel().bloch_taylor_at_gamma(order=order, prec=prec)

    # ------------------------------------------------------------------
    # MDL primitives — waterline gating (delegate to gating.mdl)
    # ------------------------------------------------------------------

    def mdl_above_waterline(self, model_bits: float, data_bits_given_model: float,
                            raw_data_bits: float) -> bool:
        """Stage-1 waterline threshold test: L(M) + L(data|M) < L(raw)? Yes/no only."""
        return _mdl.mdl_above_waterline(model_bits, data_bits_given_model, raw_data_bits)

    def channel_select(self, candidates: list, channel: str):
        """Stage-2 waterfilling-correct selection — pick the channel-matching candidate.

        See simulator.gating.mdl.channel_select / simulator/kernel.py.
        """
        return _mdl.channel_select(candidates, channel)

    def canonical_encoding(self, equivalence_class: list):
        """Min-description-length representative of a K-EQUIVALENT encoding class."""
        return _mdl.canonical_encoding(equivalence_class)

    def mdl_select(self, candidates: list):
        """⚠️ RETIRED — argmin over total bit cost. DO NOT USE FOR NEW WORK.

        Conflates canonical_encoding with channel_select; silently discards
        above-waterline channels. Kept for backwards-compat of audit checks.
        Use `channel_select` (across physically distinct K-rational candidates)
        or `canonical_encoding` (within an encoding-equivalent class).
        """
        viable = [c for c in candidates if c.get('viable', True)]
        if not viable:
            raise ValueError("No viable candidates in mdl_select")
        return min(viable, key=lambda c: c['model_bits'] + c.get('data_bits_given_model', 0))

    # ------------------------------------------------------------------
    # Substrate weight (Stage-1 ranking helper)
    # ------------------------------------------------------------------

    def slice_weight(self, N: float) -> float:
        """Combined MDL weight of this kernel's substrate slice at observation N."""
        s = self.substrate
        return _mdl.slice_combined_weight(s.coxeter, s.vertex_algebra, s.edge_algebra, N)
