"""
F(E)-associativity gate — NA-4 Phase 1 audit.

Classifies a closed prediction's derivation chain as

  F(E)-associative          derivation operates entirely after Bloch averaging
                            / F(E) flattening; insensitive to substrate-level
                            associator [a,b,c] := (ab)c − a(bc).

  substrate-Layer-1         derivation makes load-bearing use of associativity-
                            dependent primitives (F_inv(E) reduced-word walks,
                            cycle amplitudes, A2-T winding sums, …); a non-
                            associative substrate would alter the numerical
                            output through one of those primitives.

  mixed                     load-bearing chain mixes both kinds.  Substrate
                            non-associator would alter the Layer-1 factors but
                            leave the F(E)-invariant factors fixed.

  unknown                   no catalog match — manual review.

The classifier is the Phase 1 deliverable of the NA-4 simulator integration
program (handoff an internal working note).
Phase 1 by itself is "1-3 sessions, BOUNDED, NO RISK" per the handoff: even if
Need-D-3 / R-15 stay parked, the catalog is permanently useful as scoping
infrastructure for every future closure attempt — any new derivation can be
checked against the same gate.

Methodology
-----------
Per `feedback_simulator_enumerate_dont_cherrypick.md`, the classifier
ENUMERATES the full set of associativity-using primitives recognised by the
framework (`PRIMITIVE_CATALOG`) — it does NOT curate a dominant slice — and
classifies each chain by which catalog entries the load-bearing steps depend
on.  The catalog is intentionally complete:  primitives not used by the
current Phase-1 representative slice are still listed so the gate generalises
to future audits.

Per `feedback_audit_for_smuggled_parameters_2026-05-14.md`, classification is
read off STRUCTURAL features of the derivation chain (which primitives are
load-bearing).  No PDG-deviation / σ_theory inputs enter; results are reported
as catalog content only.

Per `feedback_reject_sigma_theory_2026-05-14.md`, the module never compares
predictions to PDG observations; the gate is internal to the structural
audit.

Companion modules in `simulator.gating`:
  delta_b_match          numerical β-coefficient matching gate
  spectral_consistency   C1-C4 spectral-triple consistency gate
  associativity          F(E)-associativity gate (this module)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Associativity-class enum
# ---------------------------------------------------------------------------

class AssociativityClass(Enum):
    """Outcome of running the gate on one derivation chain."""
    F_E_ASSOCIATIVE   = 'F(E)-associative'
    SUBSTRATE_LAYER_1 = 'substrate-Layer-1'
    MIXED             = 'mixed'
    UNKNOWN           = 'unknown'


# ---------------------------------------------------------------------------
# Per-primitive associativity-dependence catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Primitive:
    """One derivation primitive and its associativity-dependence class.

    `dependence` is either
      'depends_on_associator'  — primitive's output would change under non-
                                 associative substrate composition (because it
                                 uses free-monoid word reduction, walk-length
                                 grading, matrix products on associative
                                 algebras, etc.)
      'f_e_invariant'          — primitive is purely combinatorial /
                                 probabilistic / rep-theoretic at the F(E)-
                                 flattened level (local counts, Bloch-averaged
                                 moments, Bayesian probability axioms, etc.)
    """
    name: str
    dependence: str
    description: str


PRIMITIVE_CATALOG: dict[str, Primitive] = {

    # ----- ASSOCIATIVITY-DEPENDENT (substrate-Layer-1 potential) -----

    'f_inv_E_word_reduction': Primitive(
        name='f_inv_E_word_reduction',
        dependence='depends_on_associator',
        description='F_inv(E) reduced-word counting (Hashimoto NB walks).  '
                    'The "no immediate reversal" condition is built into '
                    'free-monoid (associative) concatenation; under a non-'
                    'associative substrate, (e_1 e_2) e_3 vs e_1 (e_2 e_3) '
                    'are distinct compositions and the word reduction is '
                    'ambiguous.'
    ),
    'hashimoto_walk_amplitude': Primitive(
        name='hashimoto_walk_amplitude',
        dependence='depends_on_associator',
        description='Geometric-series walk amplitudes ((k-1)/k)^L on the '
                    'Hashimoto NB graph.  Length-L is well-defined only '
                    'under associative concatenation; a substrate non-'
                    'associator would re-shuffle the L grading.'
    ),
    'girth_cycle_amplitude': Primitive(
        name='girth_cycle_amplitude',
        dependence='depends_on_associator',
        description='Cycle amplitudes (α₁_bare = (2/3)^8, α₁_full = (5/3)(2/3)^8).  '
                    'A girth cycle of length g is a closed walk — same '
                    'F_inv(E) dependence as Hashimoto walks.'
    ),
    'a2_waterline_winding_sum': Primitive(
        name='a2_waterline_winding_sum',
        dependence='depends_on_associator',
        description='A2-T waterline retained-winding sum Σ_{n≥1} α₁^n.  '
                    'Each n-th winding is the n-fold composition of girth '
                    'cycles — requires associative composition.'
    ),
    'bloch_eigenvalue_spectrum': Primitive(
        name='bloch_eigenvalue_spectrum',
        dependence='depends_on_associator',
        description='Bloch eigenvalues of B (or A) at high-symmetry k-points.  '
                    'Constructed via matrix products — depends on associative '
                    'matrix multiplication.  Eigenvalues themselves are '
                    'Bloch-invariants, but the BLOCH CONSTRUCTION uses '
                    'associativity at the operator level.'
    ),
    'cl_p_q_algebra_action': Primitive(
        name='cl_p_q_algebra_action',
        dependence='depends_on_associator',
        description='Cl(2k,0) / Cl(0,p) action on edge or vertex qubits.  '
                    'Clifford algebra is associative; substrate non-'
                    'associator would require Cayley-Dickson generalisation '
                    '(𝕆, sedenions).'
    ),
    'spectral_action_principle': Primitive(
        name='spectral_action_principle',
        dependence='depends_on_associator',
        description='Spectral-action coefficients on (A, H, D) — tr f(D/Λ) '
                    'asymptotic expansion uses heat-kernel composition, '
                    'i.e., operator products on associative B(H).  Not '
                    'currently load-bearing in the framework (M-arc closed '
                    'on KO-dim 0 GNS triple); listed for completeness.'
    ),
    'wedderburn_decomposition': Primitive(
        name='wedderburn_decomposition',
        dependence='depends_on_associator',
        description='Wedderburn decomposition of finite-dim associative '
                    'algebra (used in P3 PS-multiplet construction).  Theorem '
                    'is for associative algebras; non-associative analogue '
                    'is the magic-square / Tits construction (not currently '
                    'load-bearing).'
    ),

    # ----- F(E)-INVARIANT (Bloch / axiom-level) -----

    'local_neighborhood_count': Primitive(
        name='local_neighborhood_count',
        dependence='f_e_invariant',
        description='Local count primitives: k* (coordination), k*² (ordered '
                    'bond pairs per vertex), g (girth as a topological '
                    'invariant), N_ATOMS (BCC unit-cell vertex count).  '
                    'Net invariants — well-defined without any composition '
                    'law on E.'
    ),
    'moore_bound_identity': Primitive(
        name='moore_bound_identity',
        dependence='f_e_invariant',
        description='Algebraic identity g = k*² + 1 (Moore bound, srs).  '
                    'Pure combinatorics on local neighborhoods; no '
                    'composition law required.'
    ),
    'mdl_counting_fraction': Primitive(
        name='mdl_counting_fraction',
        dependence='f_e_invariant',
        description='MDL counting fraction over Moore-equivalent slots (e.g. '
                    'V_us = k*²/(g·N_ATOMS)).  The MDL is over local counts, '
                    'not walk lengths; F(E)-flattened by construction.'
    ),
    'bayesian_posterior': Primitive(
        name='bayesian_posterior',
        dependence='f_e_invariant',
        description='Bayesian Beta-conjugate update (ε_toggle = 1/5 from '
                    'Beta(1,1) → Beta(2,1)).  Axiom-level probability; no '
                    'composition law on E.'
    ),
    'bloch_geometric_moment': Primitive(
        name='bloch_geometric_moment',
        dependence='f_e_invariant',
        description='Bloch-averaged spatial geometric moment, e.g. '
                    '<(e·ẑ)²> = 1/k* for srs.  The Bloch averaging IS the '
                    'F(E)-flattening step; output is by construction an '
                    'F(E)-frame quantity.'
    ),
    'rep_branching': Primitive(
        name='rep_branching',
        dependence='f_e_invariant',
        description='Rep-theoretic branching SU(4)×SU(2)_L×SU(2)_R → SM (or '
                    'PS, SU(5), …).  Operates at the rep-theoretic level on '
                    'the FIXED algebra; non-associator at substrate would '
                    'change which algebra to use (input), not branching '
                    'rules (output).'
    ),
    'gleason_axiom': Primitive(
        name='gleason_axiom',
        dependence='f_e_invariant',
        description='Gleason 1957 theorem fixing d_spatial = 3 from minimum-'
                    'cost MDL.  Probability-axiom theorem; no composition '
                    'law on E.'
    ),
    'algebraic_identity': Primitive(
        name='algebraic_identity',
        dependence='f_e_invariant',
        description='Catch-all for pure algebraic post-processing '
                    '(e.g. V_cb = α₁/(1−α₁) closed form, m_H² = 2λv²).  These '
                    'do not themselves introduce or remove associativity '
                    'dependence; they propagate it from inputs.  Listed so '
                    'pure-algebra steps can be tagged without leaving '
                    'primitives empty.'
    ),
}


# ---------------------------------------------------------------------------
# Derivation-chain audit primitives
# ---------------------------------------------------------------------------

@dataclass
class ChainStep:
    """One load-bearing step in a closed prediction's derivation chain.

    `primitives` are keys into PRIMITIVE_CATALOG.  Steps that are not load-
    bearing (e.g. presentation algebra rearrangement) can be flagged
    `load_bearing=False`; the classifier ignores them.
    """
    label: str
    primitives: list[str]
    load_bearing: bool = True


@dataclass
class PredictionAudit:
    """Static audit catalog entry for ONE closed prediction.

    Hand-curated from reading the prediction's derivation file (under
    `predictions/`).  The classifier consumes this and outputs an
    `AuditResult` with an `AssociativityClass`.
    """
    name: str
    derivation_file: str
    value: str
    chain: list[ChainStep]
    notes: str = ''


# ---------------------------------------------------------------------------
# Audit catalog — 5 representative closed predictions (Phase 1)
# ---------------------------------------------------------------------------

PREDICTION_AUDITS: dict[str, PredictionAudit] = {

    'V_us': PredictionAudit(
        name='V_us',
        derivation_file='predictions/V_us.py',
        value='9/40',
        chain=[
            ChainStep(
                label='Moore bound g = k*² + 1 ⇒ k*² = 9 (algebraic identity)',
                primitives=['moore_bound_identity'],
            ),
            ChainStep(
                label='A2 edge process gives k*² ordered bond-pair couplings '
                      'per vertex (local neighborhood count)',
                primitives=['local_neighborhood_count'],
            ),
            ChainStep(
                label='Coupling-density MDL fraction k*²/(g·N_ATOMS) over '
                      'Moore-equivalent slots',
                primitives=['mdl_counting_fraction',
                            'local_neighborhood_count'],
            ),
        ],
        notes='Pure local-count derivation on the srs net (no walks, no '
              'cycle amplitudes).  Output is determined by net invariants '
              '(k*, g, N_ATOMS) only.'
    ),

    'V_cb': PredictionAudit(
        name='V_cb',
        derivation_file='predictions/V_cb.py',
        value='256/6305',
        chain=[
            ChainStep(
                label='α₁_bare = ((k*−1)/k*)^(g−n_fixed) = (2/3)^8 — first-'
                      'winding μ-moment on Hashimoto NB graph',
                primitives=['girth_cycle_amplitude',
                            'f_inv_E_word_reduction'],
            ),
            ChainStep(
                label='A2-T waterline retains all windings — geometric series',
                primitives=['a2_waterline_winding_sum'],
            ),
            ChainStep(
                label='V_cb = α₁/(1−α₁) closed form',
                primitives=['algebraic_identity'],
            ),
        ],
        notes='Level-3 Hashimoto NB-walk amplitude.  α₁ value depends on '
              'walk-length grading L=8, which uses F_inv(E) word reduction.'
    ),

    'y_tau': PredictionAudit(
        name='y_tau',
        derivation_file='predictions/y_tau.py',
        value='(5/3)(2/3)^8 / 9 = 1280/177147',
        chain=[
            ChainStep(
                label='α₁_full = (5/3)(2/3)^8 — Class-2 cycle amplitude with '
                      'admissible-cycle multiplicity 5/3 = n_g_edge/k*',
                primitives=['girth_cycle_amplitude',
                            'f_inv_E_word_reduction'],
            ),
            ChainStep(
                label='Fermion edge projection (ψ,ψ̄) = (1/k*)·(1/k*) — '
                      'local neighborhood count',
                primitives=['local_neighborhood_count'],
            ),
            ChainStep(
                label='Higgs edge × Cl(0,2) channel = 1 — Clifford-algebra '
                      'channel selection',
                primitives=['cl_p_q_algebra_action'],
            ),
            ChainStep(
                label='y_τ = α₁_full / k*²',
                primitives=['algebraic_identity'],
            ),
        ],
        notes='Inherits Level-3 walk-amplitude dependence through α₁_full.  '
              'Multiplying by 1/k*² does not remove the substrate-Layer-1 '
              'load-bearing factor.'
    ),

    'm_H': PredictionAudit(
        name='m_H',
        derivation_file='predictions/m_H.py',
        value='~125.578 GeV',
        chain=[
            ChainStep(
                label='λ_Higgs = 2·(5/3)·(2/3)^8 = 2560/19683 — same '
                      'cycle-amplitude origin as α₁_full',
                primitives=['girth_cycle_amplitude',
                            'f_inv_E_word_reduction'],
            ),
            ChainStep(
                label='v_Higgs derived from G_F + (5/12)·α₁/(1−α₁) dark vertex '
                      '(A2 winding sum)',
                primitives=['a2_waterline_winding_sum',
                            'girth_cycle_amplitude'],
            ),
            ChainStep(
                label='m_H = √(2λ)·v tree-level relation',
                primitives=['algebraic_identity'],
            ),
        ],
        notes='Both λ and v are downstream of cycle amplitudes / winding '
              'sums.  Non-associative substrate would change both factors.'
    ),

    'A_hemispherical': PredictionAudit(
        name='A_hemispherical',
        derivation_file='predictions/A_hemispherical.py',
        value='1/15',
        chain=[
            ChainStep(
                label='ε_toggle = 1/5 — Bayesian Beta(1,1)→Beta(2,1) '
                      'posterior asymmetry',
                primitives=['bayesian_posterior'],
            ),
            ChainStep(
                label='<(e·ẑ)²> = 1/k* — srs cubic moment (Bloch-averaged '
                      'spatial geometry)',
                primitives=['bloch_geometric_moment',
                            'local_neighborhood_count'],
            ),
            ChainStep(
                label='A = ε_toggle · <(e·ẑ)²>',
                primitives=['algebraic_identity'],
            ),
        ],
        notes='Bayesian probability + Bloch-averaged spatial moment.  No '
              'walk amplitudes, no F_inv(E) word reduction.'
    ),

    # ----- Extended slice (Phase 1+2 catalog extension, 2026-05-14) -----

    'J_CKM': PredictionAudit(
        name='J_CKM',
        derivation_file='predictions/J_CKM.py',
        value='c_12·c_13²·c_23·s_12·s_13·s_23·sin(δ_CP_CKM) ≈ 3.16×10⁻⁵',
        chain=[
            ChainStep(
                label='V_us = 9/40 input (Level-2 counting density)',
                primitives=['mdl_counting_fraction',
                            'local_neighborhood_count'],
            ),
            ChainStep(
                label='V_cb = 256/6305 input (Level-3 Hashimoto walk)',
                primitives=['girth_cycle_amplitude',
                            'f_inv_E_word_reduction',
                            'a2_waterline_winding_sum'],
            ),
            ChainStep(
                label='V_ub multi-cycle Σ_{m≥2} α_m/(1−α_m) input',
                primitives=['girth_cycle_amplitude',
                            'f_inv_E_word_reduction'],
            ),
            ChainStep(
                label='δ_CP_CKM = arccos(1/3) regular-tetrahedron dihedral',
                primitives=['local_neighborhood_count',
                            'algebraic_identity'],
            ),
            ChainStep(
                label='J_CKM = c_12·c_13²·c_23·s_12·s_13·s_23·sin(δ) '
                      '(trig closed form)',
                primitives=['algebraic_identity'],
            ),
        ],
        notes='Jarlskog inherits both Level-2 (V_us) F(E)-invariant input '
              'and Level-3 (V_cb, V_ub) Layer-1 inputs.  Mixed.',
    ),

    'alpha_GUT': PredictionAudit(
        name='alpha_GUT',
        derivation_file='predictions/alpha_GUT.py',
        value='1/(2^k*·k*) = 1/24',
        chain=[
            ChainStep(
                label='Local state space at trivalent node: 2^k* = 8 '
                      '(Fock-dim count) × k* = 3 (edge directions) = 24',
                primitives=['local_neighborhood_count'],
            ),
            ChainStep(
                label='Uniform MDL prior (Jaynes 1957): P = 1/24',
                primitives=['mdl_counting_fraction'],
            ),
        ],
        notes='Pure local-count derivation; the Cl(2k*) Fock DIMENSION is '
              'used (a counting fact), not the algebra action — no walks.'
    ),

    'sin2_theta_W': PredictionAudit(
        name='sin2_theta_W',
        derivation_file='predictions/sin2_theta_W.py',
        value='Tr(T_3,L²)/Tr(Q²) = 3/8',
        chain=[
            ChainStep(
                label='B3+B6: 16-state color-extended PS generation',
                primitives=['rep_branching'],
            ),
            ChainStep(
                label='Hypercharge formula Y_SM = T_3^R + (B−L)/2',
                primitives=['rep_branching'],
            ),
            ChainStep(
                label='Trace identity Σ T_3² = 2, Σ Q² = 16/3 (exact)',
                primitives=['algebraic_identity'],
            ),
            ChainStep(
                label='Georgi-Quinn-Weinberg: sin²θ_W = Σ T_3² / Σ Q² at '
                      'Killing-form unification',
                primitives=['rep_branching',
                            'algebraic_identity'],
            ),
        ],
        notes='Pure rep-theoretic trace on PS multiplet; no walks, no '
              'cycle amplitudes.'
    ),

    'Q_Koide': PredictionAudit(
        name='Q_Koide',
        derivation_file='predictions/Q_Koide.py',
        value='2/3',
        chain=[
            ChainStep(
                label='V_Ram = 8-dim Hashimoto eigenspace of B(P) at '
                      'Ramanujan saturation (|h|² = k*−1 = 2)',
                primitives=['bloch_eigenvalue_spectrum'],
            ),
            ChainStep(
                label='C_3 / Galois-Z_3 multiplicities (4, 2, 2) on V_Ram',
                primitives=['rep_branching'],
            ),
            ChainStep(
                label='Born rule (CDP 2011 / Gleason) on sqrt-multiplicity '
                      'amplitudes: Q = Σm / (Σ√m)² = 2/3',
                primitives=['gleason_axiom',
                            'algebraic_identity'],
            ),
        ],
        notes='Multiplicities (4,2,2) are rep-theoretic and F(E)-invariant, '
              'but the EXISTENCE of V_Ram as the 8-dim Ramanujan-saturated '
              'subspace requires the Bloch eigenvalue spectrum — Layer-1.  '
              'Q itself = sum-of-counts (F(E)-invariant) but built on '
              'Layer-1 substrate.  Mixed by conservative reading.'
    ),

    'm_e': PredictionAudit(
        name='m_e',
        derivation_file='predictions/m_e.py',
        value='m_τ·(f_min/f_max)² ≈ 511.6 keV',
        chain=[
            ChainStep(
                label='m_τ absolute scale via y_τ = α₁_full/k*² '
                      '(inherited Layer-1 from cycle amplitude)',
                primitives=['girth_cycle_amplitude',
                            'f_inv_E_word_reduction'],
            ),
            ChainStep(
                label='ε_Koide = √2, δ_Koide = 2/9 — Wigner-D¹ algebraic '
                      'identities on k*=3 (theorem-grade rationals)',
                primitives=['local_neighborhood_count',
                            'algebraic_identity'],
            ),
            ChainStep(
                label='f_j = 1 + ε·cos(2πj/k*+δ) on k*=3 Koide triplet',
                primitives=['local_neighborhood_count',
                            'algebraic_identity'],
            ),
            ChainStep(
                label='m_e/m_τ = (f_min/f_max)²',
                primitives=['algebraic_identity'],
            ),
        ],
        notes='Ratio (m_e/m_τ) is F(E)-invariant pure-algebra on k*=3; '
              'absolute m_e inherits Layer-1 dependence through m_τ.'
    ),

    'm_mu': PredictionAudit(
        name='m_mu',
        derivation_file='predictions/m_mu.py',
        value='m_τ·(f_mid/f_max)² ≈ 105.78 MeV',
        chain=[
            ChainStep(
                label='m_τ absolute scale (inherits Layer-1 from cycle '
                      'amplitude y_τ = α₁_full/k*²)',
                primitives=['girth_cycle_amplitude',
                            'f_inv_E_word_reduction'],
            ),
            ChainStep(
                label='ε_Koide = √2, δ_Koide = 2/9, k*=3 — same Koide '
                      'structure as m_e',
                primitives=['local_neighborhood_count',
                            'algebraic_identity'],
            ),
            ChainStep(
                label='m_μ/m_τ = (f_mid/f_max)²',
                primitives=['algebraic_identity'],
            ),
        ],
        notes='Same as m_e: ratio F(E)-invariant on local count k*=3; '
              'absolute m_μ inherits Layer-1 through m_τ.'
    ),

    'm_nu3': PredictionAudit(
        name='m_nu3',
        derivation_file='predictions/m_nu3.py',
        value='(k*·N_atoms)·M_Pl·N_hub^(-1/2) ≈ 50 meV',
        chain=[
            ChainStep(
                label='Local lattice primitives (k*=3, N_atoms=4) — '
                      'theorem-grade srs counts',
                primitives=['local_neighborhood_count'],
            ),
            ChainStep(
                label='M_Pl substrate-anchored via G_sub Drude closure '
                      '(D = -1/(⟨Tr H²⟩·k*) Bloch invariant)',
                primitives=['bloch_eigenvalue_spectrum',
                            'local_neighborhood_count'],
            ),
            ChainStep(
                label='m_ν₃ = (k*·N_atoms)·M_Pl·N_hub^(-1/2) — substrate-'
                      'spectral mass-as-flux template',
                primitives=['algebraic_identity'],
            ),
        ],
        notes='Local-count factor (k*·N_atoms) is F(E)-invariant; M_Pl '
              'anchor uses Bloch-invariant spectral sums (Layer-1).  '
              'No walk amplitudes / cycle amplitudes (the prior ADOPTED-PS '
              'M_R formulation has been retired).'
    ),

    'eta_B': PredictionAudit(
        name='eta_B',
        derivation_file='predictions/eta_B.py',
        value='(1/5)·(√3/2)·(2/3)^48 = (√3/10)·(2/3)^48 ≈ 6.11×10⁻¹⁰',
        chain=[
            ChainStep(
                label='ε_CP = 1/5 — Bayesian Beta(2,1) per-process '
                      'CP-violation asymmetry',
                primitives=['bayesian_posterior'],
            ),
            ChainStep(
                label='Re(h_P) = √3/2 — parity-even Hashimoto eigenvalue '
                      'at unique BZ saddle k_P',
                primitives=['bloch_eigenvalue_spectrum'],
            ),
            ChainStep(
                label='α₁^M = (2/3)^48 — Sakharov chain of M = N_atoms·k*/2 '
                      '= 6 winding amplitudes on Hashimoto NB walk',
                primitives=['girth_cycle_amplitude',
                            'f_inv_E_word_reduction',
                            'a2_waterline_winding_sum'],
            ),
        ],
        notes='Bayesian ε_CP × Hashimoto eigenvalue × cycle-amplitude chain.  '
              'Strong Layer-1 dominance (cycle amplitude (2/3)^48 sets the '
              'magnitude), but ε_CP factor keeps it formally mixed.'
    ),

    'Omega_DM_over_Omega_m': PredictionAudit(
        name='Omega_DM_over_Omega_m',
        derivation_file='predictions/Omega_DM_over_Omega_m.py',
        value='1 − P(k ≤ k* | Poisson(2k*)) ≈ 0.849',
        chain=[
            ChainStep(
                label='Local Fock primitives 2k*=6 (Cl(6) mode count), '
                      'k*=3 (MDL acceptance threshold)',
                primitives=['local_neighborhood_count'],
            ),
            ChainStep(
                label='Jaynes max-entropy on {0,1,2,…} with fixed mean '
                      '= Poisson(2k*) (axiom-level probability)',
                primitives=['bayesian_posterior'],
            ),
            ChainStep(
                label='Ω_DM/Ω_m = 1 − P(k ≤ k* | Poisson(2k*)) — Gleason '
                      'compressibility threshold',
                primitives=['gleason_axiom',
                            'mdl_counting_fraction'],
            ),
        ],
        notes='Pure local-count + axiom-level probability + Gleason '
              'compressibility.  No walks, no cycle amplitudes.'
    ),

    'm_W': PredictionAudit(
        name='m_W',
        derivation_file='predictions/m_W.py',
        value='M_Z·cos(θ_W) ≈ 80.69 GeV',
        chain=[
            ChainStep(
                label='M_Z = (1/2)·√(g_2²+g_Y²)·v — inherits v from BZJ '
                      'cycle amplitude (Layer-1)',
                primitives=['girth_cycle_amplitude',
                            'f_inv_E_word_reduction',
                            'a2_waterline_winding_sum'],
            ),
            ChainStep(
                label='sin²θ_W(M_Z) via single-regime MSSM-style RG from '
                      'tree value 3/8 — β-coefficients are rep counts',
                primitives=['rep_branching'],
            ),
            ChainStep(
                label='m_W = M_Z·cos(θ_W) tree relation',
                primitives=['algebraic_identity'],
            ),
        ],
        notes='M_Z piece is Layer-1 (BZJ cycle amplitude); sin²θ_W piece '
              'is F(E)-invariant rep branching.  Mixed.'
    ),

    'H_0': PredictionAudit(
        name='H_0',
        derivation_file='predictions/H_0.py',
        value='1/(N_hub·t_P) ≈ 68.2 km/s/Mpc (substrate)',
        chain=[
            ChainStep(
                label='Cascade theorem H = 1/(N·t_P) with coefficient = 1 '
                      'forced by k*=3 (D1+D2+D3 derivation)',
                primitives=['local_neighborhood_count',
                            'algebraic_identity'],
            ),
            ChainStep(
                label='N_hub is adopted dimensional input (the framework\'s '
                      'one declared adoption per axioms.adoptions); '
                      'numerical value pinned via measured G_F',
                primitives=['algebraic_identity'],
                load_bearing=False,
            ),
        ],
        notes='FORM of H_0 is theorem-grade from local count k*=3 (no '
              'walks).  Numerical value depends on adopted N_hub but the '
              'PREDICTIVE CONTENT is F(E)-invariant.'
    ),

}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    """Outcome of running the gate on one PredictionAudit."""
    audit: PredictionAudit
    primitives_used:        set[str] = field(default_factory=set)
    layer1_primitives:      set[str] = field(default_factory=set)
    invariant_primitives:   set[str] = field(default_factory=set)
    associativity_class: AssociativityClass = AssociativityClass.UNKNOWN
    structural_evidence: str = ''

    def summary(self) -> str:
        L1 = sorted(self.layer1_primitives)
        FE = sorted(self.invariant_primitives)
        s  = f'  {self.audit.name}  ({self.audit.value})\n'
        s += f'    class:            {self.associativity_class.value}\n'
        s += f'    Layer-1 used:     {L1 if L1 else "—"}\n'
        s += f'    F(E)-invariant:   {FE if FE else "—"}\n'
        s += f'    evidence:         {self.structural_evidence}'
        return s


def classify(audit: PredictionAudit) -> AuditResult:
    """Run the associativity gate on one PredictionAudit."""
    used: set[str] = set()
    L1:   set[str] = set()
    FE:   set[str] = set()
    for step in audit.chain:
        if not step.load_bearing:
            continue
        for p in step.primitives:
            if p not in PRIMITIVE_CATALOG:
                raise KeyError(
                    f'audit {audit.name!r} references primitive {p!r} '
                    f'not in PRIMITIVE_CATALOG — extend the catalog or fix '
                    f'the audit entry.'
                )
            used.add(p)
            entry = PRIMITIVE_CATALOG[p]
            if entry.dependence == 'depends_on_associator':
                L1.add(p)
            elif entry.dependence == 'f_e_invariant':
                FE.add(p)
            else:
                raise ValueError(
                    f'primitive {p!r} has unknown dependence '
                    f'{entry.dependence!r}'
                )
    # The `algebraic_identity` catch-all is F(E)-invariant but does NOT count
    # as evidence on its own — it only propagates dependence from its inputs.
    # If a chain has ONLY algebraic_identity tagged, manual review is needed.
    nontrivial_FE = FE - {'algebraic_identity'}
    if L1 and not nontrivial_FE:
        cls = AssociativityClass.SUBSTRATE_LAYER_1
        ev  = (f'Load-bearing primitives include {sorted(L1)} — all '
               f'associativity-dependent.  No F(E)-invariant primitive '
               f'serves as a structural anchor; substrate non-associator '
               f'would alter the prediction through one of the Layer-1 '
               f'inputs.')
    elif nontrivial_FE and not L1:
        cls = AssociativityClass.F_E_ASSOCIATIVE
        ev  = (f'Load-bearing primitives ({sorted(nontrivial_FE)}) are all '
               f'F(E)-invariant.  Output is determined entirely by net '
               f'invariants / Bloch-averaged moments / axiom-level prob; '
               f'substrate non-associator would not alter the prediction.')
    elif L1 and nontrivial_FE:
        cls = AssociativityClass.MIXED
        ev  = (f'Load-bearing chain mixes Layer-1 primitives {sorted(L1)} '
               f'with F(E)-invariant primitives {sorted(nontrivial_FE)}; '
               f'substrate non-associator would alter the Layer-1 factors '
               f'but leave the F(E)-invariant factors fixed.')
    else:
        cls = AssociativityClass.UNKNOWN
        ev  = ('No catalog primitives matched the load-bearing chain — '
               'manual review required.')
    return AuditResult(
        audit=audit,
        primitives_used=used,
        layer1_primitives=L1,
        invariant_primitives=FE,
        associativity_class=cls,
        structural_evidence=ev,
    )


# ---------------------------------------------------------------------------
# Catalog API
# ---------------------------------------------------------------------------

def run_audit(prediction_names: Optional[list[str]] = None) -> list[AuditResult]:
    """Run the gate on a list of prediction names; default = full catalog."""
    if prediction_names is None:
        prediction_names = list(PREDICTION_AUDITS.keys())
    results = []
    for name in prediction_names:
        if name not in PREDICTION_AUDITS:
            raise KeyError(
                f'prediction {name!r} not in PREDICTION_AUDITS — extend the '
                f'audit catalog (one entry per closed prediction to audit).'
            )
        results.append(classify(PREDICTION_AUDITS[name]))
    return results


def catalog_summary(results: list[AuditResult]) -> dict:
    """Aggregate class counts + Phase-2 recommendation."""
    counts = {c.value: 0 for c in AssociativityClass}
    for r in results:
        counts[r.associativity_class.value] += 1
    n = max(len(results), 1)
    layer1_frac = (counts[AssociativityClass.SUBSTRATE_LAYER_1.value]
                   + counts[AssociativityClass.MIXED.value]) / n
    fe_frac     = counts[AssociativityClass.F_E_ASSOCIATIVE.value] / n
    return {
        'n_audited': len(results),
        'class_counts': counts,
        'layer1_fraction': layer1_frac,
        'f_e_fraction':    fe_frac,
        # Phase-2 (Spec-A residue inspection) is justified if substantial
        # Layer-1 content exists in the audited slice.  Per handoff §"Decision
        # after Phase 1", "substantial" is operationalised here as ≥ 50%
        # Layer-1 (or mixed) representation.  This is a heuristic threshold,
        # not a load-bearing structural claim.
        'phase2_recommended': layer1_frac >= 0.5,
    }


def report(results: list[AuditResult]) -> str:
    """Format an audit catalog as a human-readable text report."""
    lines = [
        'Associativity audit — NA-4 Phase 1 (F(E)-associativity gate)',
        '=' * 70,
        '',
    ]
    for r in results:
        lines.append(r.summary())
        lines.append('')
    s = catalog_summary(results)
    lines.append('Summary')
    lines.append('-' * 70)
    lines.append(f'  Total audited:        {s["n_audited"]}')
    for cls, n in s['class_counts'].items():
        if n:
            lines.append(f'    {cls:<24s}{n}')
    lines.append(f'  Layer-1 fraction:     {s["layer1_fraction"]:.2f}')
    lines.append(f'  F(E)-invariant frac:  {s["f_e_fraction"]:.2f}')
    lines.append(f'  Phase 2 recommended:  {s["phase2_recommended"]}')
    return '\n'.join(lines)


# ===========================================================================
# Phase 2 — Spec-A residue inspection
# ===========================================================================
#
# Phase 2 (per handoff `session_handoff_2026-05-14_NA4_simulator_integration.md`
# §"Phase 2 — Spec-A residue inspection") tests NA-4 §6 test (iii): is Spec(B)
# on F_inv(E)/conj EXHAUSTIVE for framework observables?  If observables exist
# whose remaining residual aligns cleanly with a Layer-1 (non-F(E)-flattened)
# escape pattern, NA-4 hypothesis gains structural support.
#
# Method: for each named open residue (with extant scoping hypothesis in
# an internal working note), tag which catalog primitives the hypothesis invokes.
# If hypothesis invokes Layer-1 primitives, the residue ALIGNS with a Layer-1
# escape pattern; if only F(E)-invariant primitives (or data-side modeling
# choices), it does NOT.
#
# This is "first signal" — not a closure attempt.  Per the handoff, Phase 2 by
# itself is BOUNDED (2-4 sessions); Phase 3 (walker generalisation) is only
# justified if at least one residue shows Layer-1 alignment.
# ---------------------------------------------------------------------------

class ResidueClassification(Enum):
    """Outcome of running the residue classifier on one ResidueCandidate."""
    LAYER1_HYPOTHESIS = 'Layer-1 hypothesis (alignment confirmed)'
    F_E_DATA_SIDE     = 'F(E)-frame / data-side residue'
    NO_HYPOTHESIS     = 'no extant hypothesis (manual review)'


@dataclass(frozen=True)
class ResidueCandidate:
    """A named residual in the framework's closed predictions, with its
    hypothesised mechanism documented in an internal working note.

    `hypothesis_primitives` are keys into PRIMITIVE_CATALOG; the residue
    classifier reads off whether any of them are `depends_on_associator`.
    """
    name: str
    observable: str
    residual_size: str
    hypothesis: str
    hypothesis_primitives: list[str]
    scoping_doc: str
    blocked_on: str = ''


RESIDUE_CANDIDATES: dict[str, ResidueCandidate] = {

    'Lambda_CC_factor_two_path_B': ResidueCandidate(
        name='Lambda_CC_factor_two_path_B',
        observable='Λ_CC factor-of-two (Λ_LCDM / Λ_substrate ≈ 2 residual, '
                   'Path B alternative to Path A bias-function closure)',
        residual_size='~1× (factor of 2)',
        hypothesis='V_Ram 8-dim NB-survival sector splits 4+4 between '
                   'w_eff=-1 (cosmological-constant-like, "frozen" h↔h̄ '
                   'modes anti-aligned with cosmic arrow) and w_eff=0 '
                   '(matter-like, time-forward-propagating modes).  '
                   'Empirical reorganisation Ω_Λ_LCDM = Ω_Λ_framework + '
                   '(1/2)·Ω_m_framework matches percent-level.',
        hypothesis_primitives=['bloch_eigenvalue_spectrum',
                                'hashimoto_walk_amplitude'],
        scoping_doc='an internal working note'
                    'scoping_2026-05-05.md',
        blocked_on='Need A (multiway formalisation) — V_Ram h↔h̄ '
                   'cosmic-arrow-of-time classification not yet derived; '
                   'requires substrate-level mechanism distinguishing '
                   'co-rotating vs counter-rotating Hashimoto modes.',
    ),

    'Lambda_CC_factor_two_path_A_data_side': ResidueCandidate(
        # NB: this is the CONTRAST CASE — Path A closure is data-side, NOT
        # Layer-1.  Included so the classifier demonstrates discriminating
        # behaviour on a residue that has been closed via parametric-class-
        # translation (bias function family) at theorem-grade-conditional on
        # z_eff, with the remaining conditional being σ_θ_* data-side.
        name='Lambda_CC_factor_two_path_A_data_side',
        observable='Λ_CC factor-of-two via Path A (bias function family) — '
                   'σ_θ_* dependence in multi-dataset fit',
        residual_size='~0.1σ at z_eff=1.916; widens to factor of ~3 at '
                      'σ_θ_*=10⁻⁴ vs Planck-realistic 10⁻⁷',
        hypothesis='Parametric-class-translation: Λ_LCDM is Friedmann-class '
                   'extraction bias of framework coasting at multi-dataset-'
                   'specified z_eff.  Residue comes from σ_θ_* tuning, a '
                   'data-side modelling choice of the LCDM fit pipeline.',
        hypothesis_primitives=['rep_branching',         # parametric-class translation
                                'algebraic_identity'],  # bias-function algebra
        scoping_doc='an internal working note'
                    '05-09.md',
        blocked_on='(closure-conditional already on z_eff; residue is '
                   'data-side dataset weighting, not Layer-1)',
    ),

    'n_s_spectral_tilt': ResidueCandidate(
        name='n_s_spectral_tilt',
        observable='Scalar spectral index n_s = 0.965 ± 0.004 '
                   '(Δn_s ≈ −0.035 from scale-invariant n_s = 1)',
        residual_size='~3.5% from scale-invariance',
        hypothesis='Multiway causal-graph (Layer-1) fluctuation spectrum '
                   'has low-k slope giving n_s ≠ 1.  Bloch low-k dispersion '
                   '(Layer-2) gives only n_s ∈ {1, 3}; observed 0.965 '
                   'requires Layer-1 multiway primordial spectrum + '
                   'comoving-k ↔ Bloch-k unit map.',
        hypothesis_primitives=['f_inv_E_word_reduction',
                                'bloch_eigenvalue_spectrum'],
        scoping_doc='an internal working note',
        blocked_on='Need A (multiway formalisation) — no Layer-1 → Layer-2 '
                   'projection operator, no comoving-k ↔ Bloch-k physical '
                   'unit map, no canonical quantisation rule for Bloch '
                   'amplitudes.',
    ),

    'cosmology_item_5_pre_recombination': ResidueCandidate(
        name='cosmology_item_5_pre_recombination',
        observable='100·θ_* CMB acoustic peak (cascade-coasting prediction '
                   '~1098 vs Planck 1.04109; framework falsified at ~10⁵σ '
                   'at z = z_*)',
        residual_size='~10⁵σ at z = z_*',
        hypothesis='Multiway-branching pre-recombination phase: state count '
                   'N(t) ∝ t^p at z > z_eq for some p ≠ 1 (Candidate 5.1).  '
                   'Layer-1 multiway dynamics give non-coasting H(z) at '
                   'high z.  Diagnostic ruled out Candidate 5.2 (thermal-'
                   'soup transition at z > z_Planck) and 5.3 (Step C de '
                   'Sitter) as structural mechanisms.',
        hypothesis_primitives=['f_inv_E_word_reduction',
                                'a2_waterline_winding_sum'],
        scoping_doc='an internal working note'
                    'scoping_2026-05-05.md',
        blocked_on='Need A (multiway formalisation) — Candidate 5.1 has 2 '
                   'free parameters (p, z_trans), no framework-internal '
                   'derivation; requires multi-sprint Layer-1 development.',
    ),

    # ----- Extended residue catalog (2026-05-14) -----

    'm_H_higgs_quartic_feshbach': ResidueCandidate(
        name='m_H_higgs_quartic_feshbach',
        observable='m_H +3.43σ_PDG residual (Δλ_obs ≈ −7.81×10⁻⁴, '
                   'λ_obs < λ_tree = 2560/19683)',
        residual_size='+3.43σ_PDG (-0.60% on λ, +0.30% on m_H)',
        hypothesis='Un-derived Feshbach analog on the Higgs quartic λ, '
                   'analogous to the closed (5/12)·α₁/(1−α₁) Feshbach '
                   'correction on v.  Path 4 (multi-cycle 2-girth-cycle '
                   'pair amplitudes) + Path 5 (Hashimoto BZ Tr B^n '
                   'integration) tested in `theorem_mH_1loop_scoping.md` '
                   'session 25 — both falsified for simple forms.  '
                   'Residue persists at Layer-1 (cycle / winding sum) '
                   'level.',
        hypothesis_primitives=['girth_cycle_amplitude',
                                'a2_waterline_winding_sum',
                                'bloch_eigenvalue_spectrum'],
        scoping_doc='an internal working note',
        blocked_on='Bridge convention step 2.1 — un-derived Feshbach '
                   'analog on λ; cycle-amplitude paths falsified, deeper '
                   'multi-cycle / non-associative-substrate route '
                   'unexplored.',
    ),

    'm_tau_feshbach_analog': ResidueCandidate(
        name='m_tau_feshbach_analog',
        observable='m_τ +0.13% residual (Yukawa y_τ_pred − y_τ_obs)',
        residual_size='+0.13% relative on y_τ; ~0.4σ_PDG (m_τ '
                      'uncertainty-dominated by 0.012%)',
        hypothesis='Un-derived Feshbach analog on the fermion-Higgs vertex, '
                   'analogous to the closed (5/12)·α₁/(1−α₁) on v and the '
                   'open Feshbach analog on λ (m_H residue).  Per the '
                   'framework bridge convention, residual = magnitude of '
                   'the substrate Feshbach correction that has not been '
                   'investigated for the Yukawa-vertex case.',
        hypothesis_primitives=['girth_cycle_amplitude',
                                'a2_waterline_winding_sum'],
        scoping_doc='predictions/y_tau.py + bridge convention Priority 4.4 '
                    'step 2.2',
        blocked_on='Bridge convention step 2.2 — Feshbach analog on '
                   'fermion-Higgs vertex not yet attempted; downstream of '
                   'm_H Feshbach analog development.',
    ),

    'alpha_GUT_feshbach_analog': ResidueCandidate(
        name='alpha_GUT_feshbach_analog',
        observable='α_GUT cluster-drift: ⟨1/α_i(M_unif)⟩ ≈ 24.30 vs '
                   'framework bare 1/α_GUT = 1/24 (uniform across i=1,2,3)',
        residual_size='+0.0137 fractional (uniform across SU(3)/SU(2)_L/U(1)_Y); '
                      'M_Z proxy: −0.013% / +0.009% / +1.08% on '
                      '1/α_{1,2,3}(M_Z)',
        hypothesis='Substrate Feshbach analog of the (5/12) on v template, '
                   'with coefficient c_{α_GUT} ≈ 1/k_* = 1/3 (CLEAN '
                   'STRUCTURAL RATIONAL, unlike c_λ ≈ 0.148 and c_y ≈ 1/32).  '
                   'Two candidate structural routes: '
                   '(H) Hashimoto-spectral — c = (Perron sector dim)/'
                   '(NB total dim) = 4/12 if Perron multiplicity = N_atoms; '
                   '(C) cycle-counting — c = (directed-edge count)/'
                   '(N_atoms·k_*²) = 12/36.  Neither route closed.  '
                   'Calibrating constraint: any structural derivation '
                   'must reproduce c_v = 5/12 via the same mechanism.',
        hypothesis_primitives=['bloch_eigenvalue_spectrum',
                                'a2_waterline_winding_sum',
                                'girth_cycle_amplitude',
                                'local_neighborhood_count'],
        scoping_doc='an internal working note'
                    '2026-05-14.md + substrate_feshbach_analog_cluster_'
                    '2026-05-14.md §2.3',
        blocked_on='No structural derivation of c_{α_GUT} = 1/k_* yet; '
                   'Routes H and C are candidates pending Perron-sector '
                   'multiplicity verification (H) or directed-edge-per-cell '
                   'argument (C).  Probe '
                   '`proofs/foundations/alpha_GUT_dark_correction_'
                   'derivation.py` lands the numerical signature.',
    ),

    'w_DE_residual_zero_point_zero_two_eight': ResidueCandidate(
        name='w_DE_residual_zero_point_zero_two_eight',
        observable='w_DE = −1.028 ± 0.032 vs framework w = −1 (0.88σ)',
        residual_size='+0.028 absolute, +0.88σ_PDG',
        hypothesis='No active framework hypothesis for the +0.028 deviation.  '
                   'Bias-function closure (Path A) predicts w = −1 EXACTLY '
                   'at the z_eff self-consistency point; the 0.88σ residual '
                   'is "consistent with statistical noise at 1 σ; there is '
                   'no framework prediction of a non-trivial w deviation in '
                   'this approach."  Contrast case to the m_H / m_τ / Λ_CC '
                   'Path B Feshbach-analog residues.',
        hypothesis_primitives=['algebraic_identity'],
        scoping_doc='an internal working note',
        blocked_on='(no Layer-1 hypothesis; residue treated as data-side '
                   'statistical noise within current Phase A scope)',
    ),

}


def classify_residue(c: ResidueCandidate
                     ) -> tuple[ResidueClassification, str]:
    """Classify a residue by whether its hypothesis invokes Layer-1
    primitives."""
    if not c.hypothesis_primitives:
        return ResidueClassification.NO_HYPOTHESIS, 'No primitives tagged.'
    layer1 = sorted(
        p for p in c.hypothesis_primitives
        if PRIMITIVE_CATALOG[p].dependence == 'depends_on_associator'
    )
    invariant = sorted(
        p for p in c.hypothesis_primitives
        if PRIMITIVE_CATALOG[p].dependence == 'f_e_invariant'
        and p != 'algebraic_identity'
    )
    if layer1:
        ev = (f'Hypothesis invokes Layer-1 primitives {layer1} — '
              f'residue aligns with the NA-4 escape pattern.  '
              f'A non-associative substrate could in principle resolve '
              f'the residue through one of the named Layer-1 inputs.')
        return ResidueClassification.LAYER1_HYPOTHESIS, ev
    if invariant:
        ev = (f'Hypothesis uses only F(E)-invariant primitives {invariant} '
              f'— residue lives at F(E)-frame / data-side level; '
              f'non-associative substrate would not naturally resolve it.')
        return ResidueClassification.F_E_DATA_SIDE, ev
    return (ResidueClassification.F_E_DATA_SIDE,
            'Hypothesis tags only algebraic_identity — pure-algebra '
            'propagation, not a structural Layer-1 claim.')


@dataclass
class ResidueAuditResult:
    """Outcome of running the residue classifier on one ResidueCandidate."""
    candidate: ResidueCandidate
    classification: ResidueClassification
    evidence: str

    def summary(self) -> str:
        c = self.candidate
        s  = f'  {c.name}\n'
        s += f'    observable:   {c.observable}\n'
        s += f'    residual:     {c.residual_size}\n'
        s += f'    class:        {self.classification.value}\n'
        s += f'    primitives:   {sorted(c.hypothesis_primitives)}\n'
        s += f'    evidence:     {self.evidence}\n'
        if c.blocked_on:
            s += f'    blocked_on:   {c.blocked_on}\n'
        s += f'    scoping_doc:  {c.scoping_doc}'
        return s


def run_residue_audit(names: Optional[list[str]] = None
                      ) -> list[ResidueAuditResult]:
    """Run the Phase-2 residue classifier on a list of candidates;
    default = full RESIDUE_CANDIDATES."""
    if names is None:
        names = list(RESIDUE_CANDIDATES.keys())
    results = []
    for n in names:
        if n not in RESIDUE_CANDIDATES:
            raise KeyError(
                f'residue {n!r} not in RESIDUE_CANDIDATES — extend the '
                f'registry (one entry per named open residue to audit).'
            )
        c = RESIDUE_CANDIDATES[n]
        cls, ev = classify_residue(c)
        results.append(ResidueAuditResult(c, cls, ev))
    return results


def residue_summary(results: list[ResidueAuditResult]) -> dict:
    """Aggregate residue classifications + Phase-3 recommendation."""
    counts = {c.value: 0 for c in ResidueClassification}
    for r in results:
        counts[r.classification.value] += 1
    n = max(len(results), 1)
    layer1_n = counts[ResidueClassification.LAYER1_HYPOTHESIS.value]
    # Phase 3 (walker generalisation, 5–10 sessions) is justified per the
    # handoff if AT LEAST ONE residue shows clear Layer-1 alignment.
    return {
        'n_residues':            len(results),
        'class_counts':          counts,
        'layer1_aligned':        layer1_n,
        'phase3_recommended':    layer1_n >= 1,
    }


def residue_report(results: list[ResidueAuditResult]) -> str:
    """Format a residue catalog as a human-readable text report."""
    lines = [
        'Spec-A residue inspection — NA-4 Phase 2',
        '=' * 70,
        '',
    ]
    for r in results:
        lines.append(r.summary())
        lines.append('')
    s = residue_summary(results)
    lines.append('Phase 2 summary')
    lines.append('-' * 70)
    lines.append(f'  Residues audited:         {s["n_residues"]}')
    for cls, n in s['class_counts'].items():
        if n:
            lines.append(f'    {cls:<48s}{n}')
    lines.append(f'  Layer-1 aligned (≥1):     {s["layer1_aligned"]}')
    lines.append(f'  Phase 3 recommended:      {s["phase3_recommended"]}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Sentinel — round-trips for the 5 representative predictions
# ---------------------------------------------------------------------------

def _sentinel() -> None:
    """Built-in correctness check on the catalog (no PDG / observational input).

    Verifies:
      * every PrimitiveCatalog entry has a recognised `dependence` field
      * every PredictionAudit references only primitives in the catalog
      * the classifier output for the 5 representative predictions matches
        the expected class assignments worked out in the Phase 1 design
        notes:
          V_us           — F_E_ASSOCIATIVE   (pure local counts on srs)
          V_cb           — SUBSTRATE_LAYER_1 (Hashimoto walk amplitude only)
          y_τ            — MIXED             (α₁_full Layer-1 × 1/k*² F(E)-invariant)
          m_H            — SUBSTRATE_LAYER_1 (λ, v both via cycle amplitude)
          A_hemispherical — F_E_ASSOCIATIVE  (Bayesian + Bloch moment)
    """
    for k, p in PRIMITIVE_CATALOG.items():
        if p.dependence not in ('depends_on_associator', 'f_e_invariant'):
            raise AssertionError(
                f'PRIMITIVE_CATALOG[{k!r}].dependence = {p.dependence!r} '
                f'is not a recognised dependence class.'
            )
    for k, a in PREDICTION_AUDITS.items():
        for step in a.chain:
            for primitive_key in step.primitives:
                if primitive_key not in PRIMITIVE_CATALOG:
                    raise AssertionError(
                        f'PREDICTION_AUDITS[{k!r}] chain step '
                        f'{step.label!r} references unknown primitive '
                        f'{primitive_key!r}.'
                    )
    expected = {
        # Original Phase 1 representative slice
        'V_us':                  AssociativityClass.F_E_ASSOCIATIVE,
        'V_cb':                  AssociativityClass.SUBSTRATE_LAYER_1,
        'y_tau':                 AssociativityClass.MIXED,
        'm_H':                   AssociativityClass.SUBSTRATE_LAYER_1,
        'A_hemispherical':       AssociativityClass.F_E_ASSOCIATIVE,
        # Extended slice (2026-05-14)
        'J_CKM':                 AssociativityClass.MIXED,
        'alpha_GUT':             AssociativityClass.F_E_ASSOCIATIVE,
        'sin2_theta_W':          AssociativityClass.F_E_ASSOCIATIVE,
        'Q_Koide':               AssociativityClass.MIXED,
        'm_e':                   AssociativityClass.MIXED,
        'm_mu':                  AssociativityClass.MIXED,
        'm_nu3':                 AssociativityClass.MIXED,
        'eta_B':                 AssociativityClass.MIXED,
        'Omega_DM_over_Omega_m': AssociativityClass.F_E_ASSOCIATIVE,
        'm_W':                   AssociativityClass.MIXED,
        'H_0':                   AssociativityClass.F_E_ASSOCIATIVE,
    }
    for name, expected_cls in expected.items():
        got = classify(PREDICTION_AUDITS[name]).associativity_class
        if got is not expected_cls:
            raise AssertionError(
                f'classify({name!r}) = {got.value!r} but expected '
                f'{expected_cls.value!r}.'
            )
    # ----- Phase 2 sentinel: residue classifier round-trip -----
    for name, c in RESIDUE_CANDIDATES.items():
        for p in c.hypothesis_primitives:
            if p not in PRIMITIVE_CATALOG:
                raise AssertionError(
                    f'RESIDUE_CANDIDATES[{name!r}] references unknown '
                    f'primitive {p!r}.'
                )
    expected_residue = {
        'Lambda_CC_factor_two_path_B':         ResidueClassification.LAYER1_HYPOTHESIS,
        'Lambda_CC_factor_two_path_A_data_side': ResidueClassification.F_E_DATA_SIDE,
        'n_s_spectral_tilt':                   ResidueClassification.LAYER1_HYPOTHESIS,
        'cosmology_item_5_pre_recombination':  ResidueClassification.LAYER1_HYPOTHESIS,
        # Extended residue catalog (2026-05-14)
        'm_H_higgs_quartic_feshbach':          ResidueClassification.LAYER1_HYPOTHESIS,
        'm_tau_feshbach_analog':               ResidueClassification.LAYER1_HYPOTHESIS,
        'alpha_GUT_feshbach_analog':           ResidueClassification.LAYER1_HYPOTHESIS,
        'w_DE_residual_zero_point_zero_two_eight':
                                               ResidueClassification.F_E_DATA_SIDE,
    }
    for name, expected_cls in expected_residue.items():
        got, _ = classify_residue(RESIDUE_CANDIDATES[name])
        if got is not expected_cls:
            raise AssertionError(
                f'classify_residue({name!r}) = {got.value!r} but expected '
                f'{expected_cls.value!r}.'
            )


if __name__ == '__main__':
    _sentinel()
    print(report(run_audit()))
    print()
    print(residue_report(run_residue_audit()))
