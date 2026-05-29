"""
S1 — the matter content: framework-derived (Pati-Salam, theorem-grade) +
the adopted MSSM extension (the honest gap ≡ R-9's residue).

Two distinct pieces, kept distinct here:

  DERIVED (theorem-grade — from the substrate + the gauge tuple):
    • per-generation matter = (4, 2, 1) ⊕ (4̄, 1, 2) under SU(4) × SU(2)_L ×
      SU(2)_R — the Cl(6,0) Fock rep at the trivalent srs vertex, with the
      Wedderburn classification picking the SM vertices uniquely
      (`p3_wedderburn_vertex_classification.py`; theorem P3 §4).
    • generation COUNT = 3 — the C_3 / Galois-ℤ_3 structure (Jones index 3 of
      M^α ⊂ M ⊂ M ⋊_α ℤ_3 ≅ M_3(ℂ) ⊗ M^α; the M1.B chain;
      `srs_generation_c3.py` / `theorem_generation_C3_bridge.py`).
    • chirality — the Witten γ_7 grading on the edge qubit (left-handed-only
      fermion identifications); hypercharges Y = T_{3R} + (B−L)/2.
    These are theorem-grade (under the standing slate); the per-GENERATION
    Yukawa hierarchy is NOT (that's `frontier.need_d3_species`).

  ADOPTED (the gap — taken empirically alongside N_hub):
    • the MSSM superpartner content + the 2-loop RG. ≡ R-9's RESIDUE: R-9
      closed by forcing srs; its residue is that srs-z (the bipartite DOUBLE
      COVER of srs) carries the Witten-SUSY-QM χ̃ ℤ_2 grading — so the
      substrate-level home of this adopted structure is the COVER of the
      substrate, not the substrate. "Is the MSSM derivable?" = "is the cover
      forced?". Derivation routes: Path E blocked (Witten γ_7 grades chirality,
      not statistics); the per-sector-β route is CLOSED-negative (M1 audit:
      F7's α_1 winding flow ≠ MSSM RG on 5 of 5 criteria); Path E' (does χ̃
      grade statistics, not just chirality?) and M6 (does ℍ⊗𝕆 = E_7 live on
      srs-z?) remain open. So: a *declared adoption*, NOT a substrate-
      uniqueness hole. — see `frontier.mssm_as_adoption`, `axioms.adoptions()`,
      `docs/audits/registers/adoption_register.md` (ADOPTED-MSSM-Sb),
      `proofs/foundations/{mssm_matter_content_required, m1_lambda_mu_map_audit}.py`,
      `proofs/masses/srs_susy_predictions.py`.

The framework's α_GUT = 1/24 + sin²θ_W = 3/8 STRUCTURALLY REQUIRE MSSM matter
for the PDG gauge-unification match (SM / 2HDM running from α_GUT = 1/24 gives a
negative α_s) — `mssm_matter_content_required.py`. So the adoption is load-
bearing for the gauge-unification cluster (rows P63-P71, UNIQUE-THEOREM-GRADE-
CONDITIONAL on (MSSM, N_hub)).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MatterPiece:
    """One piece of the matter content.

    Attributes:
        name           : 'PS generation (4,2,1)+(4̄,1,2)' | 'MSSM superpartners' | …
        reps           : the gauge-group rep content (string)
        per_generation : True if this is per-generation content
        n_generations  : 3 for the per-generation framework pieces; None otherwise
        origin         : where it comes from ('Cl(6,0) Fock @ trivalent srs vertex' | 'C_3 / Galois-ℤ_3' | 'adopted (MSSM extension) ≡ R-9 residue' | …)
        status         : 'theorem-grade' | 'adopted'
        adopted        : True iff `status == 'adopted'`
        notes          : provenance / cross-reference
    """
    name: str
    reps: str
    per_generation: bool
    n_generations: Optional[int]
    origin: str
    status: str
    adopted: bool
    notes: str = ''


MATTER = [
    MatterPiece(
        'Pati-Salam generation', '(4, 2, 1) ⊕ (4̄, 1, 2) under SU(4) × SU(2)_L × SU(2)_R',
        True, 3, 'Cl(6,0) Fock rep @ the trivalent srs vertex (Wedderburn classification)',
        'theorem-grade', False,
        notes='p3_wedderburn_vertex_classification.py (theorem P3 §4) — the SM vertices are picked '
              'uniquely; the K_4 quotient breaks SU(4)×SU(2)_L×SU(2)_R → SM. Hypercharges '
              'Y = T_{3R} + (B−L)/2; chirality from the Witten γ_7 grading on the edge qubit '
              '(left-handed-only identifications).'),
    MatterPiece(
        'generation count = 3', 'three copies of the PS generation',
        False, 3, 'C_3 / Galois-ℤ_3 structure (Jones index 3 of M^α ⊂ M ⊂ M ⋊_α ℤ_3 ≅ M_3(ℂ)⊗M^α; M1.B chain)',
        'theorem-grade', False,
        notes='srs_generation_c3.py / theorem_generation_C3_bridge.py. The COUNT is theorem-grade; the '
              'per-generation Yukawa HIERARCHY requires species labels at the M_3(ℂ) factor — '
              'frontier.need_d3_species (5 sessions / 8 attacks ruled out, foundational extension needed).'),
    MatterPiece(
        'MSSM superpartner content + 2-loop RG', 'the supersymmetric extension of the SM matter (chiral superfields, gauginos, higgsinos)',
        False, None, 'adopted (MSSM extension) ≡ R-9 residue — the substrate-level home is srs-z, the bipartite double cover of srs (the χ̃/Witten-SUSY-QM layer); whether the cover is forced is OPEN',
        'adopted', True,
        notes='Taken empirically alongside N_hub. Load-bearing for gauge unification (α_GUT = 1/24 + '
              'sin²θ_W = 3/8 ⟹ MSSM matter for the PDG match — mssm_matter_content_required.py; '
              'rows P63-P71 UNIQUE-THEOREM-GRADE-CONDITIONAL on (MSSM, N_hub)). Derivation routes: Path E '
              'blocked (γ_7 grades chirality not statistics); per-sector-β CLOSED-negative (M1 audit, '
              'm1_lambda_mu_map_audit.py — F7 α_1 flow ≠ MSSM RG on 5/5 criteria); Path E\' / M6 open. '
              'See frontier.mssm_as_adoption, axioms.adoptions(), adoption_register.md (ADOPTED-MSSM-Sb), '
              'srs_susy_predictions.py.'),
]


def enumerate_matter() -> list[MatterPiece]:
    """The full matter content (derived PS pieces + the adopted MSSM extension)."""
    return list(MATTER)


def pati_salam_generation() -> MatterPiece:
    """The framework-derived per-generation matter: (4,2,1) ⊕ (4̄,1,2) (theorem-grade)."""
    return next(m for m in MATTER if m.name == 'Pati-Salam generation')


def n_generations() -> int:
    """3 — from the C_3 / Galois-ℤ_3 structure (theorem-grade; the count, not the hierarchy)."""
    return next(m for m in MATTER if m.name == 'generation count = 3').n_generations


def derived_matter() -> list[MatterPiece]:
    """The theorem-grade pieces (PS generation; generation count = 3)."""
    return [m for m in MATTER if m.status == 'theorem-grade']


def adopted_matter() -> list[MatterPiece]:
    """The adopted pieces (the MSSM extension ≡ R-9's residue)."""
    return [m for m in MATTER if m.adopted]


def is_adopted_matter() -> bool:
    """True — the framework currently adopts the MSSM matter content (≡ R-9's residue)."""
    return any(m.adopted for m in MATTER)


def mssm_adoption() -> dict:
    """The MSSM-matter adoption record — content, why it's adopted, the derivation-route status.

    Cross-references `frontier.mssm_as_adoption` (the gap), `axioms.adoptions()`
    (the commitment ledger), and `gauge_tuples.framework_gauge_tuple()` (the
    gauge structure it lives in).
    """
    p = adopted_matter()[0]
    return {
        'content': p.reps,
        'status': 'declared adoption (alongside N_hub) — NOT a substrate-uniqueness hole',
        'equivalent_to': 'R-9\'s residue — the substrate-level home is srs-z (the bipartite double cover of srs, the χ̃/Witten-SUSY-QM layer); whether the cover is forced is the open question',
        'load_bearing_for': 'gauge unification (α_GUT = 1/24 + sin²θ_W = 3/8 ⟹ MSSM matter for the PDG match; rows P63-P71)',
        'derivation_routes': {
            'Path E (γ_7 grades statistics?)': 'BLOCKED — γ_7 grades chirality, not statistics',
            'per-sector β (F7 α_1 winding flow ⟹ MSSM RG?)': 'CLOSED-NEGATIVE (M1 audit — ≠ on 5/5 criteria: range, functional form, boundary, discreteness, direction)',
            "Path E' (χ̃ grades statistics, not just chirality?)": 'OPEN',
            'M6 (ℍ⊗𝕆 = E_7 on srs-z?)': 'OPEN/UNCONNECTED (M_mechanisms_synthesis_2026-05-07.md)',
        },
        'see': ['frontier.mssm_as_adoption', 'axioms.adoptions()', 'docs/audits/registers/adoption_register.md (ADOPTED-MSSM-Sb)',
                'proofs/foundations/{mssm_matter_content_required, m1_lambda_mu_map_audit, susy_path_a_anomaly_cancellation, susy_path_e_witten_substrate}.py',
                'proofs/masses/srs_susy_predictions.py', 'proofs/foundations/r9_srsz_simulator_run.py', 'simulator/srsz_substrate.py'],
    }


def summary() -> dict:
    return {
        'derived': [{'name': m.name, 'reps': m.reps, 'origin': m.origin} for m in derived_matter()],
        'n_generations': n_generations(),
        'adopted': [m.name for m in adopted_matter()],
        'adoption_note': 'MSSM matter ≡ R-9\'s residue (quotient vs cover); load-bearing for gauge unification; see frontier.mssm_as_adoption',
        'source': 'p3_wedderburn_vertex_classification.py + theorem_generation_C3_bridge.py + mssm_matter_content_required.py + adoption_register.md',
    }
