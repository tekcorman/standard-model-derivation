"""
S0 — the framework's commitment set (the top of the unified pipeline).

A thin, faithful index of `docs/framework/framework_axioms.md` §10-11 (the
post-2026-05-08 slate; A1 demoted to a derived theorem). Nothing here is
re-derived — this module just exposes the structure so the rest of the
simulator (and the verify layer) can name what is *committed* (4 items, one
each metaphysical / scoping / interpretive / empirical), what is *derived
theorem* (A1, A2, A3, A4, P1', substrate-agnosticism, field-selection, Gleason
d=3, …), and what is *declared adoption* (the things the framework currently
takes empirically — N_hub (the one adopted dimensional input — "which
  universe"); MSSM matter (the genuinely-unwanted structural gap)). It also records the
load-bearing CONSEQUENCES of (A)'s no-privilege principle, which the gating
layer uses (uniform substrate measure; absent inter-generator commutation;
arc-transitive substrate ⟹ Sunada ⟹ srs — the R-9 closure).

The post-2026-05-08 honest summary (from §10):
    (A) self-containment + (B) finite observer + (I) active reading of binary
    distinctions + A5-mass empirical labeling + standard published mathematics
    = the Standard Model.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Commitment:
    """One item in the framework's commitment ledger.

    kind ∈ {
      'metaphysical'   — irreducible; not derivable (only (A))
      'scoping'        — definitional; describes the framework's subject ((B))
      'interpretive'   — a named reading; alternatives exist but forfeit relational physics ((I))
      'empirical'      — irreducible empirical content; which-math-is-which-physics (A5-mass)
      'derived'        — a derived theorem (was a structural axiom in an earlier slate)
      'adopted'        — currently taken empirically: N_hub (the clean dimensional input — "which universe"; G_F is NOT an anchor, it's predicted); MSSM matter (the genuinely-unwanted structural gap ≡ R-9's residue)
    }
    """
    name: str
    statement: str
    kind: str
    source: str = ''      # framework_axioms.md section / theorem doc / register
    notes: str = ''


# ---------------------------------------------------------------------------
# The top-level slate (4 items) — framework_axioms.md §10
# ---------------------------------------------------------------------------

THE_SLATE = [
    Commitment(
        'A', 'Self-containment of the universe — nothing comes from outside, '
             'because there is no outside.', 'metaphysical',
        source='framework_axioms.md §10/§11',
        notes='Irreducible. Its no-privilege COROLLARY (no information may be '
              'supplied from outside ⟹ no privileged value of any otherwise-free '
              'choice) is load-bearing: ⟹ uniform substrate measure '
              '(theorem_toggle_from_self_containment.md Step 1); ⟹ absent '
              'inter-generator commutation (Step 7); ⟹ no privileged spatial '
              'direction / edge-orientation ⟹ arc-transitive substrate model ⟹ '
              '(Sunada 2012) srs (R-9 closure — see no_privilege_consequences()).'),
    Commitment(
        'B', 'Finite observer — the framework describes observers with finite '
             'memory capacity.', 'scoping',
        source='framework_axioms.md §10/§11',
        notes='Definitional (says what kind of subject the predictions describe). '
              'Subsumes the prior MR2 / "framework describes external reality" '
              'clause. With (A) it gives the toggle theorem, P1\' (the observer-'
              'as-finite-register), and ℂ-over-ℝ field selection (register-is-real).'),
    Commitment(
        'I', 'Active reading of binary distinctions — a binary distinction is '
             'read as an operation that moves between two values, not as a static '
             'attribute.', 'interpretive',
        source='framework_axioms.md §10/§11 + theorem_toggle_from_self_containment.md',
        notes='Named explicitly to avoid smuggling. Motivated by the relational '
              'stance (A) suggests; the static-attribute alternative forfeits '
              'relational physics. This is what makes the toggle T_e (an operation) '
              'rather than a label.'),
    Commitment(
        'A5-mass', 'The Ramanujan eigenvalues of the substrate\'s Bloch-Hashimoto '
                   'operator are identified with the SM mass spectrum (and, under '
                   'A5b, MDL probabilities with the couplings).', 'empirical',
        source='framework_axioms.md §5b/§10',
        notes='The empirical anchor — which math object = which physical observable. '
              'NOT load-bearing for field selection (ℂ from (B) alone, post-R-6).'),
]


# ---------------------------------------------------------------------------
# Derived theorems (were structural axioms in an earlier slate; content preserved)
# ---------------------------------------------------------------------------

DERIVED = [
    Commitment('A1', 'Binary self-inverse toggle T_e² = id; algebra is F_inv(E) = '
               '(ℤ/2)^{*E}; substrate = Cayley graph of F_inv(E).', 'derived',
               source='theorems/theorem_toggle_from_self_containment.md',
               notes='From (A)+(B)+Shannon+Jaynes+Cover-Thomas+Serre+(I) (2026-05-07).'),
    Commitment('A2', 'MDL waterline: every encoding with L(M)+L(data|M) < L(raw) is '
               'retained, plurally weighted by compression savings.', 'derived',
               source='theorems/theorem_A2_mdl_from_finite_register.md',
               notes='Finite-register source coding (Shannon+Rissanen+Grünwald). '
                     'Implemented in gating/mdl.py.'),
    Commitment('A3', 'Substrate state space is complex L²(F_inv(E)); mixed states '
               'are partial traces of pure substrate states.', 'derived',
               source='theorems/theorem_A3_complex_hilbert_from_multiway.md',
               notes='Multiway-level Stone + rapid-decay continuum + register-is-real '
                     'field selection (ℂ; ℝ and ℍ excluded — uniqueness ledger Row 5 / R-6).'),
    Commitment('A4', 'Local CAR at each k*-valent node, generating Cl(2k*; ℂ).', 'derived',
               source='theorems/theorem_car_local_jordan_wigner.md',
               notes='Local via Jordan-Wigner; global CAR (B1 ordering) open but not '
                     'load-bearing for any current prediction.'),
    Commitment("P1'", 'The observer exists within the framework as a finite register '
               'built from the same primitive (toggles) as the substrate, '
               'persisting across observations.', 'derived',
               source='theorems/theorem_p1_prime_derived_from_a1.md',
               notes='MR1→(A), MR2→(B), MR3→consequence of (B).'),
    Commitment('substrate-agnosticism', 'Observer-substrate response patterns '
               'partition substrate space into equivalence classes; the Cayley graph '
               'of F_inv(E) is the description-length-minimal canonical representative; '
               'predictions are invariant within the class.', 'derived',
               source='theorems/theorem_substrate_agnosticism.md',
               notes='Kolmogorov 1965 / Solomonoff 1964 / Li-Vitányi 2008. Used by the '
                     'R-9 closure: "the substrate IS the observer\'s DL-minimal canonical '
                     'model" ⟹ strong isotropy passes from the model to the substrate.'),
    Commitment('field-selection-ℂ', 'ℂ over ℝ (and over ℍ) for the substrate Hilbert '
               'space — register-is-real argument under (B).', 'derived',
               source='theorems/theorem_A3_complex_hilbert_from_multiway.md Step 7',
               notes='Closable from (B) via P1\' alone post-R-6 (2026-04-27); A5-mass '
                     'NOT load-bearing here.'),
    Commitment('Gleason-d=3', 'd_spatial = 3 ⟹ vertex coordination k* = 3 ⟹ |E| = 3 — '
               'via Gleason 1957 (Born-rule uniqueness needs Hilbert dim ≥ 3) + the '
               'MDL minimum-cost viable dimension.', 'derived',
               source='predictions/observer_dim_three_derivation.md + predictions/k_star_derivation.md + d_spatial_derivation.md',
               notes='Implemented in kernel.mdl_select_hilbert_dimension; the Axis-A↔Axis-B '
                     'bridge (collapses the Coxeter-quotient menu\'s high-|E| argmax to |E|=3).'),
]


# ---------------------------------------------------------------------------
# Declared adoptions (currently taken empirically — honest gaps)
# ---------------------------------------------------------------------------

ADOPTIONS = [
    Commitment('N_hub', 'The framework adopts ONE dimensional physical input: N_hub ≈ '
               '8.394881e60 — the universe\'s worldline length / hub count ("which '
               'universe / how big"). Everything dimensional is DERIVED from it — the '
               'cosmological cascade (Λ_CC ∝ N_hub⁻², t_0 = N_hub·t_Pl, H_0, A_s, …), '
               'the cosmic-epoch index N_obs ∈ [1, N_hub], the physical energy scales — '
               'AND the Fermi constant G_F (G_F = 1/(√2 v²), v from the BZJ cascade ← '
               'N_hub: G_F is a DOWNSTREAM PREDICTION, NOT an anchor). A unit-setting '
               'constant (M_Pl ≡ G_N ≡ t_Pl) is the conventional unit choice (and '
               'M_substrate = 1 makes it nearly derived via M_substrate/M_Pl = √π/8), '
               'not a physics anchor. The dimensionless STRUCTURE (gauge group, α_GUT '
               '= 1/24, sin²θ_W = 3/8, mass ratios, mixing angles) is N_hub-independent '
               '— a disconnected axis.', 'adopted',
               source='predictions/N_hub_derivation.md',
               notes='The framework\'s single adopted physical input. Its VALUE is '
                     'empirical (a contingent universe-scale fact, like G_N\'s value); '
                     'it is currently pinned to ppm precision by consistency with the '
                     'measured Fermi constant (predictions/N_hub.py:n_hub_from_g_f_consistency), '
                     'but that is a precision footnote, NOT a structural dependency — '
                     'nothing in the framework "is tied to G_F" (G_F is predicted). '
                     'See axioms.n_hub_pivot().'),
    Commitment('MSSM matter (≡ R-9 residue)', 'The MSSM matter content (and its 2-loop '
               'RG) is taken as an empirical input alongside N_hub (the OTHER adopted input — the scale anchor) — no framework-internal '
               'SUSY derivation. R-9 closed by forcing srs; its RESIDUE is that srs-z '
               '(the bipartite double cover of srs) carries the Witten-SUSY-QM χ̃ '
               'grading, i.e. the substrate-level home of this adopted structure is the '
               'COVER of the substrate, not the substrate. So "is the MSSM derivable?" '
               '= "is the cover forced?" — Path E blocked (Witten γ_7 grades chirality '
               'not statistics), per-sector-β route CLOSED-negative (M1 audit); Path E\' '
               '(does χ̃ grade statistics?) + M6 (ℍ⊗𝕆 = E_7 on srs-z?) remain open.', 'adopted',
               source='docs/audits/registers/adoption_register.md (ADOPTED-MSSM-Sb) + '
                      'structural_residue_register.md (R-9 closure) + mssm_matter_content_required.py',
               notes='To live in menus/matter.py with the adoption flag; frontier.py records '
                     'the closed/open derivation routes.'),
]


# ---------------------------------------------------------------------------
# (A)'s no-privilege principle — its load-bearing consequences
# ---------------------------------------------------------------------------

def n_hub_pivot() -> dict:
    """The N_hub-pivot decision (2026-05-12): N_hub is THE adopted dimensional input.

    The framework adopts ONE dimensional physical number — N_hub ≈ 8.394881e60,
    the universe's worldline length / hub count. Everything dimensional is derived
    from it; G_F (the Fermi constant) is a DOWNSTREAM PREDICTION, not an anchor —
    nothing in the framework "is tied to G_F". The repo's earlier "N_hub anchored
    from G_F" framing is RETRACTED. (No prediction changes — the adopted N_hub
    value equals the value the old G_F-inversion produced; only the labelling /
    logical direction flips.)
    """
    return {
        'adopted_dimensional_input': 'N_hub',
        'value': 8.394881e60,
        'meaning': "the universe's worldline length / hub count — 'which universe / how big'",
        'derives_from_it': ['the cosmological cascade (Λ_CC ∝ N_hub⁻², t_0 = N_hub·t_Pl, H_0, A_s normalization, Ω_DM/Ω_m normalization, η_B)',
                            'the cosmic-epoch index N_obs ∈ [1, N_hub] (small N_obs = early universe; N_obs = N_hub = today)',
                            'the physical energy scales for the running couplings (via the Planck scale)',
                            'the Fermi constant G_F = 1/(√2 v²), v from the BZJ cascade ← N_hub — G_F is a PREDICTION'],
        'not_affected': 'the dimensionless structure — gauge group, α_GUT = 1/24, sin²θ_W = 3/8, mass ratios, mixing angles — is N_hub-independent (a disconnected axis)',
        'unit_choice': 'a unit-setting constant (M_Pl ≡ G_N ≡ t_Pl) is the conventional unit, not a physics anchor; M_substrate = 1 makes M_Pl nearly derived (M_substrate/M_Pl = √π/8)',
        'value_provenance': "empirical (a contingent universe-scale fact); currently pinned to ppm precision by consistency with the measured Fermi constant (predictions/N_hub.py:n_hub_from_g_f_consistency) — a precision footnote, NOT a structural dependency",
        'retracted_framing': "RETRACTED 2026-05-12: 'N_hub anchored from G_F' / 'G_F = external anchor' (the repo-wide framing pre-2026-05-12) — G_F is now a downstream prediction; N_hub is the adopted input",
        'see': ['predictions/N_hub.py / N_hub_derivation.md', 'predictions/G_F.py / G_F_derivation.md (now: DERIVED)',
                'docs/parameters/parameter_uniqueness_ledger.md Row P17', 'match/anchors.py', 'frontier (the OTHER adoption — MSSM matter — is separate)'],
    }


def no_privilege_consequences() -> list[dict]:
    """The chain (A) ⟹ no-privilege ⟹ {…} that the gating / substrate layers use.

    Each entry: {consequence, mechanism, where_it_lands}. This is the spine that
    connects the metaphysical commitment (A) to the concrete substrate (srs).
    """
    return [
        {'consequence': 'uniform substrate measure (each F_inv(E) word equally weighted)',
         'mechanism': 'no privileged value of the per-word weight ⟹ uniform',
         'where': 'theorem_toggle_from_self_containment.md Step 1; kernel.branch_measure / gating/mdl freq factor'},
        {'consequence': 'no a-priori inter-generator commutation',
         'mechanism': 'a privileged commutation relation would be supplied which-structure ⟹ none assumed',
         'where': 'theorem_toggle_from_self_containment.md Step 7; menus/coxeter (the m-matrix is enumerated, not fixed)'},
        {'consequence': 'no privileged spatial direction or edge-orientation ⟹ arc-transitive substrate model ⟹ (Sunada 2012) srs',
         'mechanism': 'the walker\'s causal state is a directed edge (Shalizi-Crutchfield); a directionless observer must treat all directed edges as equivalent ⟹ strongly isotropic; substrate-agnosticism passes this to the substrate; Sunada ⟹ unique 3-reg 3-conn ℝ³ such net = srs',
         'where': 'R-9 closure; walker_dynamics_derivation.md Step 4b + g_girth_derivation.md Step 2; menus/crystal_nets.framework_substrate_selection()'},
        {'consequence': 'd_spatial = 3 (Born-rule uniqueness threshold)',
         'mechanism': 'Gleason 1957 (frame functions unique for Hilbert dim ≥ 3) + MDL minimum-cost viable dimension',
         'where': 'predictions/observer_dim_three_derivation.md; kernel.mdl_select_hilbert_dimension; gating/observer.py'},
    ]


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def slate() -> list[Commitment]:
    """The 4 top-level commitments {(A), (B), (I), A5-mass}."""
    return list(THE_SLATE)


def derived_theorems() -> list[Commitment]:
    """The derived theorems (A1, A2, A3, A4, P1', substrate-agnosticism, …)."""
    return list(DERIVED)


def adoptions() -> list[Commitment]:
    """The declared adoptions (currently-empirical inputs / honest gaps)."""
    return list(ADOPTIONS)


def all_commitments() -> list[Commitment]:
    return list(THE_SLATE) + list(DERIVED) + list(ADOPTIONS)


def get(name: str) -> Commitment:
    for c in all_commitments():
        if c.name == name:
            return c
    raise ValueError(f"axioms.get: no commitment named {name!r}; have "
                     f"{[c.name for c in all_commitments()]}")


def is_adopted(name: str) -> bool:
    try:
        return get(name).kind == 'adopted'
    except ValueError:
        return False


def summary() -> dict:
    return {
        'slate': [{'name': c.name, 'kind': c.kind, 'statement': c.statement} for c in THE_SLATE],
        'derived_count': len(DERIVED),
        'adoptions': [c.name for c in ADOPTIONS],
        'honest_summary': ('(A) self-containment + (B) finite observer + (I) active '
                           'reading of binary distinctions + A5-mass empirical labeling '
                           '+ standard published mathematics = the Standard Model (structure); '
                           'add the one adopted dimensional input N_hub ≈ 8.4e60 [+ a unit '
                           'convention] = the Standard Model at every cosmic epoch + cosmology. '
                           'G_F is a prediction, not an anchor.'),
        'source': 'docs/framework/framework_axioms.md §10-11 (post-2026-05-08 slate)',
    }
