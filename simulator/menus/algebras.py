"""
Substrate-level non-associative algebra menu — NA-4 Phase 3 scoping.

PHASE 3 SCOPING ONLY (2026-05-14).  This module enumerates the candidate
non-associative substrate composition laws that replace F_inv(E) — the
free involutive (associative) monoid currently underpinning every
`simulator.gating.mdl` primitive.  Per the Phase 3 scoping doc
an internal working note
§6, this is the first-session deliverable.  No MDL gating is applied
here; that lives in `simulator.gating.mdl_nonassoc` (DEFERRED — Phase 3
session 2 deliverable).

Distinction from `simulator.menus.vertex_algebras`:
- vertex_algebras.py = LOCAL ALGEBRA AT EACH VERTEX (Fock-space side;
  Cl(2k, 0), Cayley-Dickson ℝ/ℂ/ℍ/𝕆/sedenion, Tits-Freudenthal magic
  square ℝ⊗𝕆 = F_4 / ℂ⊗𝕆 = E_6 / ℍ⊗𝕆 = E_7 / 𝕆⊗𝕆 = E_8).  Non-associative
  vertex algebras already enumerated.
- algebras.py (this module) = SUBSTRATE COMPOSITION LAW (stream of toggles
  across vertices; what F_inv(E) plays the role of).  Phase 3's actual
  target: enumerate non-associative replacements for F_inv(E).

Per the handoff failure mode 4 ("Computational MDL primitives need
adaptation. channel_select and waterfilling were defined for associative
regime; extending them to non-associative algebras may require theoretical
work (1–2 sessions of foundational reformulation before applying)"), this
module is THE FIRST sub-deliverable of that reformulation.

Candidate space (NA-4 §6 tests (i), (ii)):
- Free magma M(E) — binary trees, no axioms (test (i))
- Free Moufang loop FM(E) — weakened associativity, octonion-natural (test (ii))
- Free Bol loop FB(E) — right-Bol identity only
- Octonionic Cayley-Dickson substrate (Cl(0,3) ≅ 𝕆-as-substrate)
- Sedenion substrate
- Tits-Freudenthal magic square at substrate level

Per `feedback_simulator_enumerate_dont_cherrypick.md` the enumeration is
intentionally complete; per `feedback_audit_for_smuggled_parameters_2026-
05-14.md` no candidate is presumed to "win" — selection is MDL bit-count
gated (Phase 3 session 3+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Algebraic-axiom classification
# ---------------------------------------------------------------------------

class AssociativityAxiom(Enum):
    """Which subset of associativity holds on the substrate composition law."""
    ASSOCIATIVE       = 'associative'           # (xy)z = x(yz) ∀ x,y,z
    DIASSOCIATIVE     = 'diassociative'         # assoc on any 2-generated subloop (Moufang)
    POWER_ASSOCIATIVE = 'power_associative'     # x^m · x^n = x^(m+n); single-generator
    ALTERNATIVE       = 'alternative'           # (xx)y = x(xy); (xy)y = x(yy)
    NONASSOCIATIVE    = 'nonassociative'        # no associativity beyond magma axioms


class LoopAxiom(Enum):
    """Which loop axioms hold (identity, inverses, …)."""
    MAGMA      = 'magma'        # binary product only; no identity, no inverses
    QUASIGROUP = 'quasigroup'   # left/right divisions; no identity in general
    LOOP       = 'loop'         # identity + two-sided inverses
    MOUFANG    = 'moufang'      # loop satisfying Moufang identities
    BOL        = 'bol'          # loop satisfying right-Bol identity
    BRUCK      = 'bruck'        # loop satisfying Bruck identity (= Moufang of exp 2)
    GROUP      = 'group'        # loop + full associativity (= group)


# ---------------------------------------------------------------------------
# SubstrateAlgebra dataclass — one candidate substrate composition law
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubstrateAlgebra:
    """One candidate substrate composition law.

    Attributes:
        name           : human-readable identifier
        family         : 'free_magma' | 'free_loop' | 'cayley_dickson' |
                          'magic_square' | 'group_baseline'
        associativity  : AssociativityAxiom value
        loop_axiom     : LoopAxiom value
        identity       : True iff the algebra has a two-sided identity element
        inverses       : True iff every element has a two-sided inverse
        norm_composition: True iff a multiplicative norm exists (Hurwitz)
        dim_real       : real dimension (∞ for free structures; finite for
                          Cayley-Dickson and magic-square entries)
        baseline_word_count: a string descriptor of length-L word-count growth
                          ('F_inv(E,N)', 'free_magma_M(E,N)', …) — interpreted
                          by `simulator.gating.mdl_nonassoc` (DEFERRED).
        notes          : provenance / scoping notes (NA-4 doc references, etc.)
    """
    name: str
    family: str
    associativity: AssociativityAxiom
    loop_axiom: LoopAxiom
    identity: bool
    inverses: bool
    norm_composition: bool
    dim_real: int | float  # int for finite; float('inf') for free structures
    baseline_word_count: str
    notes: str = ''


# ---------------------------------------------------------------------------
# Curated candidate algebras
# ---------------------------------------------------------------------------

# F_inv(E) baseline — the CURRENT framework substrate composition law.
# Included so MDL bit-count comparisons have an explicit associative baseline.
F_INV_E = SubstrateAlgebra(
    name='F_inv(E) = (Z/2)^*E free involutive monoid',
    family='group_baseline',
    associativity=AssociativityAxiom.ASSOCIATIVE,
    loop_axiom=LoopAxiom.GROUP,
    identity=True, inverses=True, norm_composition=False,
    dim_real=float('inf'),
    baseline_word_count='F_inv(E, N) — length-L count E·(E−1)^(L−1) for E ≥ 2',
    notes='The CURRENT framework substrate composition law.  '
          'simulator.gating.mdl.free_word_log_count implements this.  '
          'Phase 3 candidates must clear MDL bit-count over this baseline.'
)

FREE_MAGMA = SubstrateAlgebra(
    name='Free magma M(E)',
    family='free_magma',
    associativity=AssociativityAxiom.NONASSOCIATIVE,
    loop_axiom=LoopAxiom.MAGMA,
    identity=False, inverses=False, norm_composition=False,
    dim_real=float('inf'),
    baseline_word_count='M(E, N) — binary trees Catalan-bounded: '
                        '|E|^N · C_{N−1} ≈ (4|E|)^N / N^(3/2)',
    notes='**PRIOR A2-MDL CLOSURE 2026-05-08 (CLOSED NEGATIVE):** '
          '`proofs/foundations/sector_free_magma_walker_probe.py` + '
          'an internal working note '
          'show that imposing associativity on M(E) gives Φ ≈ 2N bits '
          'cumulative Catalan compression at small relator cost L ≈ '
          '6·log₂|E| + 13.  Combined weight = Φ − L grows as ~2N − const '
          '— F(E) wins by enormous margin.  NA-4 escape via free-magma '
          'walker is structurally BLOCKED.  Listed for completeness; '
          'flagged CLOSED so Phase 3 does not re-attempt.'
)

FREE_MOUFANG = SubstrateAlgebra(
    name='Free Moufang loop FM(E)',
    family='free_loop',
    associativity=AssociativityAxiom.DIASSOCIATIVE,
    loop_axiom=LoopAxiom.MOUFANG,
    identity=True, inverses=True, norm_composition=False,
    dim_real=float('inf'),
    baseline_word_count='FM(E, N) — diassoc on 2-letter sub-words; '
                        '3-letter associator weighted (Bruck 1958 / '
                        'Kuznetsov 1988 ref needed)',
    notes='**PRIOR A2-MDL CLOSURE 2026-05-08 (CLOSED NEGATIVE by '
          'subsumption):** any intermediate non-associative refinement '
          'of M(E) — including Moufang loops — gives STRICTLY LESS '
          'Catalan compression than imposing full associativity, so it '
          'is A2-DISFAVORED relative to F(E).  Per `free_magma_walker_'
          'probe_2026-05-08.md` §3.6.  Listed for completeness; flagged '
          'CLOSED so Phase 3 does not re-attempt the free-Moufang route.'
)

FREE_BOL = SubstrateAlgebra(
    name='Free Bol loop FB(E) (right-Bol)',
    family='free_loop',
    associativity=AssociativityAxiom.NONASSOCIATIVE,  # strictly weaker than Moufang
    loop_axiom=LoopAxiom.BOL,
    identity=True, inverses=True, norm_composition=False,
    dim_real=float('inf'),
    baseline_word_count='FB(E, N) — right-Bol identity weakens FM(E); '
                        'word count ≥ FM(E, N)',
    notes='**PRIOR A2-MDL CLOSURE 2026-05-08 (CLOSED NEGATIVE by '
          'subsumption):** weaker than FM(E), strictly less compression, '
          'A2-disfavored.  Same closure as FREE_MAGMA / FREE_MOUFANG.'
)

OCTONION_SUBSTRATE = SubstrateAlgebra(
    name='Octonionic substrate (Cl(0,3) ≅ 𝕆-as-composition)',
    family='cayley_dickson',
    associativity=AssociativityAxiom.ALTERNATIVE,
    loop_axiom=LoopAxiom.MOUFANG,  # unit octonions form Moufang loop
    identity=True, inverses=True, norm_composition=True,
    dim_real=8,
    baseline_word_count='Octonion-group quotient — finite, |𝕆_unit| = 16 '
                        '(Moufang loop M(16, 0, 2) / Cayley group)',
    notes='Cayley-Dickson d=3.  Hurwitz normed.  Restricted to |E| ≤ 7 '
          '(7 imaginary units).  Framework |E| = 6 fits.  E_8 candidate '
          'via Tits magic square 𝕆⊗𝕆.'
)

SEDENION_SUBSTRATE = SubstrateAlgebra(
    name='Sedenion substrate (Cayley-Dickson d=4)',
    family='cayley_dickson',
    associativity=AssociativityAxiom.POWER_ASSOCIATIVE,
    loop_axiom=LoopAxiom.LOOP,  # loop but not Moufang
    identity=True, inverses=False,  # zero divisors
    norm_composition=False,         # loses norm beyond octonions
    dim_real=16,
    baseline_word_count='Sedenion-class — finite, |S_unit| = 32; zero '
                        'divisors break inverse closure',
    notes='Cayley-Dickson d=4.  Loses norm composition.  Likely too weak '
          'for MDL retention; included for enumeration completeness.'
)

TITS_E8_SUBSTRATE = SubstrateAlgebra(
    name='Tits-Freudenthal 𝕆⊗𝕆 = E_8 (substrate-level Lie alg)',
    family='magic_square',
    associativity=AssociativityAxiom.NONASSOCIATIVE,  # Lie algebra, not algebra
    loop_axiom=LoopAxiom.MAGMA,  # Lie bracket = magma, not loop
    identity=False, inverses=False, norm_composition=False,
    dim_real=248,
    baseline_word_count='Lie-algebra root system E_8 — 240 roots + Cartan; '
                        'finite dim 248',
    notes='𝕆⊗𝕆 Tits magic-square highest entry.  NA-4 §4(b) candidate for '
          'saturation-state automorphism.  Framework Cl-bivector apparatus '
          'does NOT give E_8 without octonionic substrate '
          '(`saturation_state_scoping_2026-05-06.md` §3).'
)


# ---------------------------------------------------------------------------
# Enumerators
# ---------------------------------------------------------------------------

def enumerate_baseline() -> list[SubstrateAlgebra]:
    """The associative baseline.  Phase 3 MDL comparison reference."""
    return [F_INV_E]


def enumerate_free_structures() -> list[SubstrateAlgebra]:
    """Free non-associative structures (NA-4 §6 tests (i), (ii)).

    NB: all 3 entries here are CLOSED NEGATIVE at A2-MDL per
    `proofs/foundations/sector_free_magma_walker_probe.py` (2026-05-08)
    + an internal working note.
    F(E) is A2-dominant on every free-magma-quotient refinement.
    Enumerated here for catalog completeness; not viable Phase 3
    closure candidates.
    """
    return [FREE_MAGMA, FREE_MOUFANG, FREE_BOL]


def enumerate_open_phase3_candidates() -> list[SubstrateAlgebra]:
    """The Phase 3 candidates that are NOT closed by 2026-05-08.

    These are FINITE non-associative substrates — they are NOT
    quotients of M(E), so the Catalan-compression argument that
    closes the free-magma route does not apply.  They remain viable
    Phase 3 candidates pending MDL bit-count comparison via
    `simulator.gating.mdl_nonassoc` (DEFERRED).
    """
    return enumerate_cayley_dickson_substrates() + enumerate_magic_square_substrates()


def enumerate_cayley_dickson_substrates() -> list[SubstrateAlgebra]:
    """Cayley-Dickson tower at the substrate level (d=3, 4)."""
    return [OCTONION_SUBSTRATE, SEDENION_SUBSTRATE]


def enumerate_magic_square_substrates() -> list[SubstrateAlgebra]:
    """Tits-Freudenthal magic square at the substrate level (E_8 candidate)."""
    return [TITS_E8_SUBSTRATE]


def enumerate_full_menu() -> list[SubstrateAlgebra]:
    """Full Phase 3 candidate menu (associative baseline + 6 non-associative)."""
    return (enumerate_baseline()
            + enumerate_free_structures()
            + enumerate_cayley_dickson_substrates()
            + enumerate_magic_square_substrates())


# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

def _sentinel() -> None:
    """Catalog-consistency check.  No execution, no MDL comparison.

    Verifies:
      * enumerate_full_menu() returns the expected 7-algebra menu
      * baseline F_inv(E) is the only associative entry
      * Moufang / octonion entries have norm composition where expected
      * every entry's `associativity` and `loop_axiom` are recognised enums
    """
    menu = enumerate_full_menu()
    if len(menu) != 7:
        raise AssertionError(
            f'enumerate_full_menu() returned {len(menu)} algebras, expected 7'
        )
    assoc = [a for a in menu
             if a.associativity is AssociativityAxiom.ASSOCIATIVE]
    if len(assoc) != 1 or assoc[0].name != F_INV_E.name:
        raise AssertionError(
            f'expected single associative baseline F_inv(E); got {len(assoc)}'
        )
    if not OCTONION_SUBSTRATE.norm_composition:
        raise AssertionError(
            'octonion substrate must have norm_composition=True (Hurwitz)'
        )
    if SEDENION_SUBSTRATE.norm_composition:
        raise AssertionError(
            'sedenion substrate must have norm_composition=False '
            '(zero divisors)'
        )
    for a in menu:
        if not isinstance(a.associativity, AssociativityAxiom):
            raise AssertionError(
                f'{a.name}: associativity must be AssociativityAxiom enum'
            )
        if not isinstance(a.loop_axiom, LoopAxiom):
            raise AssertionError(
                f'{a.name}: loop_axiom must be LoopAxiom enum'
            )


def _summary() -> str:
    """Human-readable summary of the menu."""
    lines = [
        'Substrate-level non-associative algebra menu — NA-4 Phase 3',
        '=' * 70,
        '',
    ]
    for a in enumerate_full_menu():
        lines.append(f'  {a.name}')
        lines.append(f'    family:         {a.family}')
        lines.append(f'    associativity:  {a.associativity.value}')
        lines.append(f'    loop_axiom:     {a.loop_axiom.value}')
        lines.append(f'    dim_real:       {a.dim_real}')
        lines.append(f'    norm:           {a.norm_composition}')
        lines.append(f'    word count:     {a.baseline_word_count}')
        lines.append('')
    lines.append(f'  Total candidates: {len(enumerate_full_menu())}')
    lines.append('')
    lines.append('  Phase 3 status: SCOPING.  MDL bit-count comparison vs')
    lines.append('  F_inv(E) baseline lives in simulator.gating.mdl_nonassoc')
    lines.append('  (DEFERRED — Phase 3 session 2 deliverable).')
    return '\n'.join(lines)


if __name__ == '__main__':
    _sentinel()
    print(_summary())
