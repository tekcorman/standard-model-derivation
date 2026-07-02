"""
MDL waterline + frequency support — substrate-menu gating.

Reproduces the formulas in
  proofs/foundations/sector_coxeter_freq_weighted_audit.py
  proofs/foundations/sector_coxeter_full_menu_ranking_audit.py
  proofs/foundations/sector_path_B_multi_gen_audit.py
  proofs/foundations/sector_cooling_cascade_audit.py

  L_elias(m)              = 1 + 2·floor(log₂ m) for finite m; L_elias(∞) = 1.
  description_length(M)   = Σ_{i<j} L_elias(m_ij)               [Coxeter pairwise]
                          = L_elias(K) + L_elias(m)             [multi-gen single relator]
                          = Σ_{i<j} L_elias(∞)                  [free baseline]
  free_word_log_count(E,N)= log₂ of count of length-≤N reduced words on F_inv(E)
  compression_value(M,N)  = max(0, free_word_log_count(E,N) − log₂|W(M, N)|)
                            finite : log₂|W| static
                            affine : rank·log₂(N+1) + log₂|W_finite| (poly growth)
                            hyperbolic / free : 0 (no closed-form |W|)
  freq_factor(M,N)        = log₂(N) − max(L_r)·log₂(|E|)
  n_attest(M)             = |E|^max(L_r)
  combined_weight(M,N)    = compression_value(M,N) − description_length(M) + min(freq_factor(M,N), 0)

These are universal across the substrate Coxeter menu. Vertex / edge
algebras carry their own (smaller) description-length and N_attest scoring
per Tasks A, B (commits 2c2a624, 7748658, 2026-05-07); the slice-level
score combines all three layers.

TWO-STAGE GATING (post-2026-05-12 MDL cleanup):

  Stage 1 — WATERLINE THRESHOLD.  A candidate (Coxeter system, vertex
  algebra, edge algebra, or — at the prediction layer — a physical
  realization of an observable) is RETAINED iff combined_weight ≥ threshold
  at observation length N. All retained candidates are PHYSICALLY REALIZED —
  the framework does NOT pick a single minimum-cost winner. This is
  `mdl_above_waterline` / `above_waterline` / `retained_above_waterline`.

  Stage 2 — CHANNEL SELECT.  For ONE specific observable, which retained
  candidate it reads is fixed by a STRUCTURAL argument (the observable's
  substrate definition) — a `channel` string fixed BEFORE candidates are
  enumerated. `channel_select(candidates, channel)` picks the matching one.
  If multiple match (K-equivalent within the channel), the min-bit-cost
  canonical representative. The retracted `mdl_select` (argmin) wrongly
  collapsed Stages 1+2 — see `feedback_waterline_not_minimum_canonical_distinction`.
"""

import math
from typing import Iterable, Optional

from ..menus.coxeter import CoxeterSystem
from ..menus.vertex_algebras import VertexAlgebra
from ..menus.edge_algebras import EdgeAlgebra


# ============================================================================
# Encoding primitives (matching sector_coxeter_freq_weighted_audit.py)
# ============================================================================

def L_elias(m) -> float:
    """Elias-gamma code length for positive integer m. L_elias(∞) = 1."""
    if m == float('inf'):
        return 1.0
    if m < 1:
        return float('inf')
    return 1.0 + 2.0 * math.floor(math.log2(m))


def description_length(coxeter: CoxeterSystem) -> float:
    """L(M) for a Coxeter system.

    - free baseline   : Σ_{i<j} L_elias(∞) = C(|E|,2)·1
    - multi-generator : L_elias(K) + L_elias(m)  (single-relator system)
    - pairwise        : Σ_{i<j} L_elias(m_ij), with unspecified pairs m=2
    """
    E = coxeter.generators
    n_pairs = E * (E - 1) // 2
    if coxeter.growth_class == 'free':
        return n_pairs * L_elias(float('inf'))
    if coxeter.multi_gen_K is not None and coxeter.multi_gen_m is not None:
        return L_elias(coxeter.multi_gen_K) + L_elias(coxeter.multi_gen_m)
    total = 0.0
    for i in range(1, E + 1):
        for j in range(i + 1, E + 1):
            m = coxeter.m_pairs.get((i, j), 2)
            total += L_elias(m)
    return total


def vertex_algebra_description_length(va: VertexAlgebra) -> float:
    """L(V) for a vertex algebra — small constant, ~ log₂(dim_real). See Task A."""
    return va.description_bits


def edge_algebra_description_length(ea: EdgeAlgebra) -> float:
    """L(E) for an edge algebra — small constant, ~ log₂(dim_real). See Task B."""
    return ea.description_bits


def free_word_log_count(E: int, N: float) -> float:
    """log₂ of count of length-≤N reduced words on F_inv(E) = (Z/2)^{*E}.

    No two adjacent letters equal (involutive cancellation), so for E ≥ 2
    the count of length-L words is E·(E−1)^(L−1). Closed asymptotics:
    E=1 → 1 ; E=2 → 2N+1 ; E≥3 → N·log₂(E−1) + log₂(E/(E−2)).
    """
    if N == 0 or E == 0:
        return 0.0
    if E == 1:
        return 1.0 if N >= 1 else 0.0
    if E == 2:
        return math.log2(2 * N + 1) if N > 0 else 0.0
    return N * math.log2(E - 1) + math.log2(E / (E - 2))


def log2_W_at_N(coxeter: CoxeterSystem, N: float) -> Optional[float]:
    """log₂ of |W(M, N)| — distinct group elements reachable in length ≤ N.

    finite : log₂|W| (static).  affine : rank·log₂(N+1) + log₂|W_finite|.
    hyperbolic / free : None (no closed-form |W| — handled as Φ = 0).
    """
    if coxeter.growth_class == 'finite':
        return math.log2(coxeter.order)
    if coxeter.growth_class == 'affine':
        fin_ord = coxeter.finite_order if coxeter.finite_order else 1
        return coxeter.rank * math.log2(N + 1) + math.log2(fin_ord)
    return None  # hyperbolic / free


def compression_value(coxeter: CoxeterSystem, N: float) -> float:
    """Φ(M, N) = max(0, free_word_log_count(|E|, N) − log₂|W(M, N)|)."""
    w_log = log2_W_at_N(coxeter, N)
    if w_log is None:
        return 0.0  # free baseline & multi-gen (no closed-form |W|)
    f_log = free_word_log_count(coxeter.generators, N)
    return max(0.0, f_log - w_log)


def max_relation_length(coxeter: CoxeterSystem) -> int:
    """Length of the longest defining relator (2·max m_ij, or K·m for multi-gen)."""
    return coxeter.max_relation_length


def freq_factor(coxeter: CoxeterSystem, N: float) -> float:
    """log₂(N) − max(L_r)·log₂(|E|).

    Positive ⇒ rarest relation well-attested (compression-only ranking).
    Negative ⇒ rarest relation expected < 1× per stream (Bayesian weight suppressed).
    """
    if N <= 0:
        return float('-inf')
    return math.log2(N) - max_relation_length(coxeter) * math.log2(coxeter.generators)


def n_attest(coxeter: CoxeterSystem) -> float:
    """|E|^max(L_r) — threshold N at which freq_factor crosses 0."""
    m = max_relation_length(coxeter)
    return float(coxeter.generators ** m) if m > 0 else 1.0


def combined_weight(coxeter: CoxeterSystem, N: float) -> float:
    """Bayesian combined weight: Φ(M, N) − L(M) + min(freq_factor(M, N), 0)."""
    return (compression_value(coxeter, N)
            - description_length(coxeter)
            + min(freq_factor(coxeter, N), 0.0))


# ============================================================================
# Slice-level (combined substrate × vertex × edge)
# ============================================================================

def slice_n_attest(coxeter: CoxeterSystem, vertex: VertexAlgebra,
                   edge: EdgeAlgebra) -> float:
    """Combined N_attest for a (substrate, vertex, edge) tuple = max of layers.

    Per Task D (commit 51edbc8): tuple N_attest = max(substrate, vertex, edge).
    Tuple is attested in the zoo iff N ≥ tuple N_attest.
    """
    return max(n_attest(coxeter), float(vertex.n_attest), float(edge.n_attest))


def slice_combined_weight(coxeter: CoxeterSystem, vertex: VertexAlgebra,
                          edge: EdgeAlgebra, N: float) -> float:
    """Combined-tuple Bayesian weight at observation length N.

    Φ(coxeter, N) − L(coxeter) − L(vertex) − L(edge)  +  min(log₂(N/N_attest_slice), 0)

    The single tuple-level frequency penalty (over the max of all three
    layers' attestation thresholds) replaces per-layer freq penalties, so
    the coxeter's own freq term is folded into the slice term, not added twice.
    """
    nat = slice_n_attest(coxeter, vertex, edge)
    freq_slice = (math.log2(N) - math.log2(nat)) if (N > 0 and nat > 0) else float('-inf')
    return (compression_value(coxeter, N)
            - description_length(coxeter)
            - vertex_algebra_description_length(vertex)
            - edge_algebra_description_length(edge)
            + min(freq_slice, 0.0))


# ============================================================================
# Stage 1: waterline threshold (which candidates are retained at N)
# ============================================================================

def mdl_above_waterline(model_bits: float, data_bits_given_model: float,
                        raw_data_bits: float) -> bool:
    """Raw waterline test: L(model) + L(data|model) < L(raw). Yes/no only."""
    return (model_bits + data_bits_given_model) < raw_data_bits


def above_waterline(combined_weight_value: float, threshold: float = 0.0) -> bool:
    """Stage-1 retention test: combined_weight ≥ threshold.

    All candidates passing this are PHYSICALLY REALIZED — Stage 2
    (channel_select) picks which one each observable reads.
    """
    return combined_weight_value >= threshold


def _score(candidate, N: float) -> float:
    """Combined-weight score for a CoxeterSystem or a (cox, vert, edge) tuple."""
    if isinstance(candidate, CoxeterSystem):
        return combined_weight(candidate, N)
    if (isinstance(candidate, tuple) and len(candidate) >= 3
            and isinstance(candidate[0], CoxeterSystem)):
        return slice_combined_weight(candidate[0], candidate[1], candidate[2], N)
    raise TypeError(
        f"_score: expected CoxeterSystem or (CoxeterSystem, VertexAlgebra, "
        f"EdgeAlgebra) tuple, got {type(candidate).__name__}")


def retained_above_waterline(candidates: Iterable, N: float,
                             threshold: float = 0.0) -> list:
    """Filter a candidate list to those above the waterline at observation N.

    `candidates` may be CoxeterSystem instances or (cox, vertex, edge) tuples.
    Returns list of (candidate, weight) sorted by weight descending — the
    substrate-menu input to Stage 2 channel selection.
    """
    scored = [(c, _score(c, N)) for c in candidates]
    kept = [(c, w) for c, w in scored if above_waterline(w, threshold)]
    kept.sort(key=lambda cw: -cw[1])
    return kept


# ============================================================================
# Stage 2: channel selection (which retained candidate an observable reads)
# ============================================================================

def channel_select(candidates: list, channel: str):
    """Stage-2 selection — pick the candidate whose `channel` field matches.

    `candidates` is a list of dicts each with at minimum a 'channel' key
    (other keys — 'value', 'name', 'model_bits', … — preserved for the
    caller). ALL candidates are assumed already above the waterline (Stage 1
    filtered). If multiple candidates K-equivalently realize the same channel
    (encoding-equivalent: same value at different bit costs), the min-
    'model_bits' canonical representative is returned.

    The `channel` string is FIXED by the observable's substrate definition
    BEFORE candidates are enumerated — there is no goal-seeking ("declare
    alternatives, pick the one closest to PDG"). This is the load-bearing MDL
    primitive for the prediction layer; `mdl_select` (argmin) is RETIRED.

    Mirrors simulator/kernel.py CountingKernel.channel_select (the live
    reference implementation used across match/sm_predictions/).
    """
    matching = [c for c in candidates if c.get('channel') == channel]
    if not matching:
        available = sorted(set(c.get('channel') for c in candidates
                               if c.get('channel') is not None))
        raise ValueError(
            f"channel_select: no candidate matches channel {channel!r}. "
            f"Available channels: {available}")
    if len(matching) == 1:
        return matching[0]
    return min(matching, key=lambda c: c.get('model_bits', 0))


def canonical_encoding(equivalence_class: list):
    """Stage-2 helper — min-bit-cost representative of K-EQUIVALENT encodings.

    For a class of encodings that all yield the SAME numerical value at
    different bit costs (NOT physically-distinct K-rational candidates —
    that's channel_select's domain), return the canonical (minimum-cost) one.
    Each element is a dict with a 'model_bits' key (0 if absent).
    """
    if not equivalence_class:
        raise ValueError("canonical_encoding: empty equivalence class")
    return min(equivalence_class, key=lambda c: c.get('model_bits', 0))
