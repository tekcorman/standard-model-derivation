"""
Δb match gate — filter β-contribution candidates against MSSM target.

Companion to `simulator.menus.beta_contributions`.  Each candidate produces
a triple Δb_i for i ∈ {1, 2, 3}; we compare to the MSSM target
(+5/2, +25/6, +4) and apply the structural criteria P1-P6 from the scoping
doc `walk_based_delta_b_search_scoping_2026-05-14.md`:

  P1 — Three-from-one: bundles must share a single structural origin
       (not 3 independent constructions per gauge factor).
  P2 — Gauge-factor assignment is natural (not hand-tuned).
  P3 — Rational structure with substrate-bounded denominators.
  P4 — Substrate-derived turn-on scale.
  P5 — Consistent with framework-derived masses.
  P6 — MDL parsimony.

P1, P2, P5 are STRUCTURAL judgments — they require expert review beyond
what the gate can automate.  The gate flags candidates passing the
NUMERICAL/algebraic criteria (match, denominator-bounded, MDL-low) and
exposes the structural checks for human review.
"""

from fractions import Fraction
from dataclasses import dataclass, field
import math
from typing import Optional

from ..menus.beta_contributions import (
    BetaContributionCandidate, ParticleBundle, MSSM_DELTA_B,
    bundle_delta_b,
)


# ---------------------------------------------------------------------------
# Numerical match criteria
# ---------------------------------------------------------------------------

def matches_mssm_target(db: tuple[Fraction, Fraction, Fraction],
                        target: tuple[Fraction, Fraction, Fraction] = MSSM_DELTA_B
                        ) -> bool:
    """C1 — exact rational match to MSSM Δb target."""
    return db == target


def denominator_bound(db: tuple[Fraction, Fraction, Fraction],
                      max_den: int = 24) -> bool:
    """C2 — all denominators ≤ max_den.

    Substrate primitives bound natural denominators by:
      - k* = 3
      - g = 10
      - N_atoms = 4, N_edges = 6
      - n_channels = 2
      - α_GUT⁻¹ = 24
    So denominators ≤ 24 = α_GUT⁻¹ are the natural framework bound.
    """
    return all(f.denominator <= max_den for f in db)


# ---------------------------------------------------------------------------
# MDL parsimony
# ---------------------------------------------------------------------------

def L_elias(n: int) -> float:
    """Elias-gamma length for positive integer n."""
    if n < 1:
        return float('inf')
    return 1.0 + 2.0 * math.floor(math.log2(n))


def bundle_bits(b: ParticleBundle) -> float:
    """Description-length bits for one ParticleBundle.

    Encodes (rep_3, rep_2, |Y_num|, Y_den, statistics, n_real, mult)
    via Elias-gamma on the positive integers.
    """
    Y = b.Y
    bits = (L_elias(b.rep_3) + L_elias(b.rep_2)
            + L_elias(abs(Y.numerator) + 1) + L_elias(Y.denominator)
            + (1 if b.statistics == 'fermion' else 0)
            + L_elias(b.n_real) + L_elias(b.mult))
    return bits


def candidate_bits(c: BetaContributionCandidate) -> float:
    """Total description-length bits for a candidate.

    Sums per-bundle bits, deduping by (rep_3, rep_2, Y, statistics, n_real)
    and using mult as the count.
    """
    # Build dedup key → mult
    grouped: dict[tuple, int] = {}
    for b in c.bundles:
        key = (b.rep_3, b.rep_2, b.Y, b.statistics, b.n_real)
        grouped[key] = grouped.get(key, 0) + b.mult
    total = 0.0
    for (r3, r2, Y, stat, n_real), mult in grouped.items():
        proto = ParticleBundle(label='dedup', rep_3=r3, rep_2=r2, Y=Y,
                               statistics=stat, n_real=n_real, mult=mult)
        total += bundle_bits(proto)
    return total


# ---------------------------------------------------------------------------
# Structural criteria flags (P1, P2, P5 require human review)
# ---------------------------------------------------------------------------

@dataclass
class StructuralFlags:
    """Flags exposing the P1/P2/P5 structural checks to human review."""
    p1_three_from_one: Optional[bool] = None    # bundles share single origin
    p2_natural_gauge_split: Optional[bool] = None
    p5_mass_consistent: Optional[bool] = None


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------

@dataclass
class DeltaBCheckResult:
    """Outcome of running the gate on one BetaContributionCandidate."""
    candidate: BetaContributionCandidate
    delta_b: tuple[Fraction, Fraction, Fraction]
    matches_target: bool = False
    denominator_ok: bool = False
    mdl_bits: float = 0.0
    structural_flags: StructuralFlags = field(default_factory=StructuralFlags)

    @property
    def passes_numerical(self) -> bool:
        return self.matches_target and self.denominator_ok

    def summary(self) -> str:
        s = f'    Δb = ({self.delta_b[0]}, {self.delta_b[1]}, {self.delta_b[2]})'
        s += f'\n    matches target {MSSM_DELTA_B}: {self.matches_target}'
        s += f'\n    denominator ≤ 24: {self.denominator_ok}'
        s += f'\n    MDL bits: {self.mdl_bits:.1f}'
        if self.structural_flags.p1_three_from_one is not None:
            s += f'\n    P1 (three-from-one): {self.structural_flags.p1_three_from_one}'
        if self.structural_flags.p2_natural_gauge_split is not None:
            s += f'\n    P2 (natural gauge split): {self.structural_flags.p2_natural_gauge_split}'
        return s


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate(c: BetaContributionCandidate) -> DeltaBCheckResult:
    """Run the gate on one candidate.  Returns DeltaBCheckResult."""
    db = c.delta_b()
    return DeltaBCheckResult(
        candidate=c,
        delta_b=db,
        matches_target=matches_mssm_target(db),
        denominator_ok=denominator_bound(db),
        mdl_bits=candidate_bits(c),
    )


def filter_matching(candidates: list[BetaContributionCandidate]
                    ) -> tuple[list[DeltaBCheckResult], list[DeltaBCheckResult]]:
    """Filter candidates by numerical match (P-criteria deferred to human review).

    Returns (matching, all_results).  Matching candidates are those passing
    the exact-target check.  Sorted within matching by MDL bits ascending.
    """
    results = [evaluate(c) for c in candidates]
    matching = [r for r in results if r.matches_target]
    matching.sort(key=lambda r: r.mdl_bits)
    return matching, results
