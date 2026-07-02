"""
The frontier — the unified simulator's *boundary*.

Once everything else is absorbed, the simulator is "complete except for these."
Each gap below is a genuine hole in the abstraction (not a probe that re-derives
something the simulator already computes, and not a thing closed by a structural
argument — those become `simulator.query() + assert` shims or are recorded in
`axioms` / `crystal_nets.framework_substrate_selection`). Calling a gap's stub
raises `NotImplementedError` with the precise blocker; `list_gaps()` /
`get_gap(name)` enumerate them.

Source: an internal working note §4 (the
frontier register), `docs/audits/registers/structural_residue_register.md` (the
R-N catalog — note R-9 is now CLOSED, so it is NOT here), and the per-gap
`proofs/**` clusters.

NB: R-9 (srs vs srs-z) is **CLOSED — STRUCTURAL** (2026-05-12: (A) ⟹ arc-transitive
⟹ Sunada ⟹ srs); see `menus.crystal_nets.framework_substrate_selection()`. Its
*residue* (srs-z = the bipartite double cover ⟹ the χ̃/SUSY layer) is folded
into gap #5 ("MSSM matter as adoption ≡ R-9's residue").

NB: **Need-D-3 is DISSOLVED as a mechanism question** (2026-05-16
unified-oblique + quark-unification): the CKM amplitudes are zero-parameter
off-diagonal readings of the one B_NB(srs) resolvent. The `need_d3_species`
gap below is therefore RE-SCOPED to its live residue **Need-B** — the
*diagonal* per-generation quark-Koide phase δ reading of that *same*
resolvent — and is motivated against the simulator's existing machinery
(the G_NB resolvent + the screw-axis/Wigner-D Koide observables that already
give the solved-sibling lepton δ=2/9). Key kept stable (`need_d3_species`)
for the validation-probe contract; content reconciled to the 2026-05-16
capstone. Authority: `docs/state_of_the_derivation_2026-05-16.md` +
an internal working note.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Gap:
    """One genuine open gap at the simulator's boundary.

    Attributes:
        key            : short identifier for `get_gap` / the stub function name
        title          : one-line description
        blocker        : the precise reason it's not derivable yet
        status         : 'open-research' | 'open-bounded' | 'declared-adoption' | 'closed-negative-route(s)-but-thing-open' | 'extraction-layer'
        residue        : the residue-register R-N entry it maps to, or '' if none
        clusters       : the proofs/** file clusters that probe it
        affects        : what downstream content depends on closing it (or '' = nothing load-bearing)
    """
    key: str
    title: str
    blocker: str
    status: str
    residue: str = ''
    clusters: tuple = ()
    affects: str = ''


GAPS = [
    Gap('d_gt_3_substrates',
        'R-4 / R-5 — substrate alternatives of dimension > 3 (dark/cosmo channel)',
        'Not RCSR 3D crystal nets — no 3dall.txt entry; they live in the '
        'dark-sector buildup (the dim-count partition of F_inv(E)\'s ensemble), '
        'not the crystal-net realization menu. A proper enumeration of d>3 '
        'substrate models + their A2-T Boltzmann weight in the C4 channel is open.',
        'open-research', residue='R-4 / R-5',
        clusters=('predictions/H_multiway_dim_count.py', 'proofs/foundations/substrate_lattice_waterfilling_omega_dm.py'),
        affects='small (sub-σ) Boltzmann shifts in the dark/cosmo channel only; Ω_DM/Ω_m +0.002 below PDG sensitivity'),
    Gap('vram_cl6_fock_map',
        'V_Ram(P) ≅ Cl(6,0) Fock — the missing identification',
        'P4§6 #3: the (4,2,2) C_3 generation structure lives on V_Ram(P) (an '
        '8-dim space at the Ramanujan saddle) while the P3/P4 vertex form uses '
        'the Cl(6,0) Fock rep (a different 8-dim space). No canonical isomorphism '
        'between them is established — research-level.',
        'open-research', residue='',
        clusters=('proofs/foundations/sector_V_ab_*', 'proofs/foundations/b4_color_vram_*',
                  'proofs/foundations/cocycle_check_vram.py', 'proofs/foundations/path_b_v_ram_cycle_space_bijection.py'),
        affects='blocks the from-scratch τ_L → τ_R vertex matrix element (P4§6 #3); the y_τ corollary chain still stands via A5'),
    Gap('beta_dark',
        'β_dark — dark-sector renormalization (the running of α_dark)',
        'The substrate-internal F7 flow closed α_1 (winding cutoff Λ) but the '
        'dark-sector β-function itself is open; F7 §4.2(c) was FALSIFIED for the '
        '5/12 identification (gauge β_1 leading coefficient = α_1* ≈ 0.0406 ≠ '
        'c = 5/12 — different counting families). A genuine substrate derivation '
        'of β_dark is research-level.',
        'open-research', residue='',
        clusters=('proofs/foundations/dark_feshbach_*', 'proofs/wave_engine/dark_*_spectral*',
                  'proofs/foundations/wigner_weisskopf_dark.py'),
        affects='the dark-correction coefficient family (c, ν_amp, ν_mass²) is computed combinatorially; β_dark would put it on RG footing'),
    Gap('need_d3_species',
        'Need-B (quark Koide phase δ) — the diagonal G_NB per-generation reading '
        '[ex-"Need-D-3"; the mechanism question is DISSOLVED]',
        'Need-D-3 (CKM-from-species-labels-at-M_3(ℂ)) is DISSOLVED as a mechanism '
        'question (2026-05-16 unified-oblique + quark-unification; '
        'theorem_unified_oblique.md + quark_unification_over_determination_test): '
        'V_us/V_cb/V_ub/J are zero-parameter OFF-diagonal readings of the one '
        'B_NB(srs) resolvent the simulator already computes (the same resolvent '
        'that carries δ_r/δρ). The live residue is Need-B: the DIAGONAL '
        'per-generation-phase reading of that SAME resolvent — the quark Koide '
        'phase δ that turns the theorem-grade GJ=3 sector texture into the '
        'physical m_d:m_s:m_b (and m_u:m_c:m_t) splitting. Driven to its '
        'structural floor 2026-05-16 (needB R1/R4, self-validated): the lepton '
        'δ=2/9 derivation is robust only by a Q=2/3-SPECIFIC algebraic '
        'coincidence (screw-axis/Wigner-D Route A=Route B ⟺ Q=2/3); at triplet '
        'Q≈0.75 the routes diverge ~0.2 rad, so "specialise the lepton template '
        'by Hamming-valence increment" is ILL-POSED. The down ε² is the '
        "framework's OWN Row-P37 (3g−2)/g=14/5 (≈2.5, NOT lepton √2) and must "
        'itself be derived; the arg(h)/4 near-match is REFUTED, target re-pinned '
        'δ_down≈0.10 rad. ANTI-NUMEROLOGY GUARDRAIL (load-bearing): any δ must be '
        'derived as a substrate count/projection that ALSO reproduces lepton 2/9 '
        'by the same route — a data-fitting number lacking that mechanism does '
        'NOT count. It inherits the §6(i) HM↔δ physical identification = the deep '
        'T_mass layer. Up-type carries an ADDITIONAL gate: down-type rides '
        'v·y_τ·GJ=3 (all theorem-grade) but up-type has no anchor and the y_t '
        'operator is σ₊-nilpotent — "no route". MOTIVATION: this is not "find a '
        'new mechanism" — the simulator already has the machinery (the G_NB '
        'resolvent it reads off-diagonal for CKM; the screw-axis/Wigner-D Koide '
        'observables that give the solved-sibling lepton δ=2/9); Need-B is the '
        'named diagonal projection those do not yet perform.',
        'open-research', residue='R-14 / Need-B (≡ ex-Need-D-3, dissolved-as-mechanism)',
        clusters=('proofs/foundations/needB_R1_triplet_screw_wigner_2026-05-16.py',
                  'proofs/foundations/needB_R4_pin_delta_target_2026-05-16.py',
                  'proofs/foundations/quark_unification_over_determination_test_2026-05-16.py',
                  'proofs/foundations/fock_q3_laplacian.py',
                  'docs/theorems/theorem_41_screw_wigner.md',
                  'an internal working note',
                  'docs/state_of_the_derivation_2026-05-16.md'),
        affects='the per-generation quark Yukawa hierarchy: closing Need-B '
                'elevates down-type m_d/m_s/m_b BLOCKED→THEOREM-GRADE-CONDITIONAL '
                'in one step; up-type m_u/m_c/m_t additionally gated on the '
                'σ₊-nilpotent y_t anchor (no route). The generation COUNT (=3), '
                'C_3 structure, GJ=3 texture, and the CKM amplitudes (off-diagonal '
                'G_NB readings) are theorem-grade.'),
    Gap('mssm_as_adoption',
        'MSSM matter as a declared adoption — ≡ R-9\'s residue (quotient vs cover)',
        'R-9 closed by forcing srs; its residue: srs-z (the bipartite double cover '
        'of srs) carries the Witten-SUSY-QM χ̃ Z_2 grading. Multi-probe arc 2026-05-13 '
        'sharpened the obstruction: (a) the de-Rham SUSY Q = d + d* on the srs '
        'cochain complex IS a real fermion↔boson supercharge (vertex Cl(6) Fock = '
        'matter, edge Cl(0,2) = gauge+Higgs; Q swaps them) and lifts gauge-equivariantly '
        'at the OPERATOR-ALGEBRA level (de_rham_susy_fibered_v2_probe — Q̂_alg on '
        'C⁰_alg(256) ⊕ C¹_alg(24), gauge-equivariant at machine precision); but at '
        'the STATE level no gauge-equivariant linear projection ℂ⁸_v → ℂ²_e exists '
        '(rep-theory obstruction). (b) Standard CC bridge (Phase 1-2): framework CC '
        'matter content under SU(2)_L = Spin(3) ⊂ Spin(6) is the STANDARD PS '
        'generation (16 states/cell), Tr T_3L² = 2, Tr Q² = 16/3, sin²θ_W = 3/8 ✓. '
        'Phase 1\'s "4 quartets" finding was a wrong-embedding artefact (RETRACTED). '
        'So framework + standard CC = SM-like matter; SM β-functions don\'t reach '
        'observed α_s(M_Z); MSSM β-functions do; framework doesn\'t structurally '
        'produce the superpartner spectrum. Per-sector-β route CLOSED-negative (M1 '
        'audit). Path E\' / M6 still listed but octonion thread broadly barren. '
        'Operator-algebra spectral-action route (spectral_action_beta_probe) gives '
        'finite-cell heat-kernel flow not directly comparable to 4D continuum '
        'β-functions; would need a non-standard CC framework to extract gauge '
        'β-functions from Q̂_alg. So: declared adoption, structurally clarified — '
        'framework derives the GUT-scale boundary conditions cleanly (α_GUT⁻¹ = 24, '
        'sin²θ_W = 3/8) but not the running mechanism that bridges to M_Z.',
        'declared-adoption (de-Rham fermion-boson SUSY exists at operator-algebra level — gauge-equivariant — but doesn\'t single-handedly produce MSSM β-functions; state-level SUSY rep-theory-obstructed; per-sector-β closed-negative)',
        residue='R-9 (residue) ≡ ADOPTED-MSSM-Sb',
        clusters=('proofs/foundations/mssm_matter_content_required.py', 'proofs/foundations/susy_path_*',
                  'proofs/foundations/m1_lambda_mu_map_audit.py', 'proofs/foundations/srs_z_chi_*',
                  'proofs/foundations/r9_srsz_simulator_run.py', 'simulator/srsz_substrate.py',
                  'proofs/foundations/de_rham_susy_on_srs_probe.py',
                  'proofs/foundations/de_rham_susy_fibered_probe.py',
                  'proofs/foundations/de_rham_susy_fibered_v2_probe.py',
                  'proofs/foundations/spectral_action_beta_probe.py',
                  'proofs/foundations/connes_chamseddine_step1_su2_content_probe.py',
                  'proofs/foundations/connes_chamseddine_step2_su2R_hypercharge_probe.py',
                  'docs/audits/registers/adoption_register.md'),
        affects='the 2-loop-RG cluster (gauge unification / α_GUT=1/24 + sin²θ_W=3/8 ⟹ MSSM matter for PDG match); rows P63-P71 sit UNIQUE-THEOREM-GRADE-CONDITIONAL on (MSSM, N_hub). Research direction for eventual closure: non-standard CC framework that extracts gauge β-functions from the operator-algebra-level gauge-equivariant Q̂_alg (research-level, multi-session)'),
    Gap('delta_cp_arg_h_path_b',
        'δ_CP from arg(h) — Path B',
        'The CKM CP phase δ_CP candidate arg(h_H)/4 ≈ … is ~0.97σ from PDG but '
        'the from-substrate derivation (the Q\'-band Berry phase / Chern route) is '
        '~2.5% off and research-level; many sub-probes (Berry phase, Chern, SU(2) '
        'convergence) are CLOSED-negative.',
        'open-research', residue='',
        clusters=('proofs/foundations/arg_h_path_b_*', 'proofs/foundations/path_b_*',
                  'proofs/foundations/srs_R14_path_b_numerical_scan.py'),
        affects='δ_CP (CKM) is currently the combinatorial K_4-minus-eigenspace-dihedral reading; a clean arg(h) derivation would supersede it'),
    Gap('fibers_for_non_srs_realizations',
        'Per-realization fiber tables — non-srs / non-cubic crystal nets',
        '`rcsr_per_substrate_fingerprint.assess_net` builds the high-symmetry '
        'k-points only for the cubic 3-regular candidate set; for hexagonal / '
        'non-cubic RCSR entries (qtz, eta, etc, …) it can\'t build the net, so '
        'their Bloch fibers are not catalogued. (Only matters for Axis-B slices '
        'OTHER than srs — and the framework substrate IS srs.)',
        'open-bounded', residue='',
        clusters=('proofs/foundations/rcsr_per_substrate_fingerprint.py', 'simulator/menus/fibers.py'),
        affects='nothing load-bearing — the framework substrate (srs) has its full fiber table; this is for subdominant-slice studies'),
    Gap('acoustic_scale',
        'r_s / θ_* / native CMB C_l — the acoustic scale',
        'Not on `target_parameters.md`. A from-substrate derivation would need '
        'the photon-baryon fluid mechanics + FRW scalings, which are SIDE-LOADED '
        'physics — REJECTED as a framework claim. These are extraction-layer '
        'translations at best; the framework is ground-up from graph primitives.',
        'extraction-layer (out of scope as a framework claim)', residue='',
        clusters=('proofs/cosmology/sound_horizon.py', 'proofs/cosmology/native_CMB_subpiece_derivation.py',
                  'proofs/cosmology/recombination_running.py'),
        affects='nothing — explicitly out of the target-parameter set'),
    Gap('gleason_genericity',
        'C1 — Gleason genericity (the d=3 conditioning\'s soft point)',
        'The d_spatial = 3 conditioning (the Axis-A↔Axis-B bridge) rests on '
        'Gleason 1957 holding generically (frame functions unique for Hilbert '
        'dim ≥ 3). The genericity clause C1 of the seven-Gleason-sub-assumptions '
        'audit is flagged as the residual soft point — Theorem 8 sits '
        'THEOREM-GRADE-CONDITIONAL on it.',
        'open-bounded', residue='C1',
        clusters=('proofs/foundations/sector_C1_gleason_genericity_audit.py',
                  'proofs/foundations/theorem8_*', 'predictions/observer_dim_three_derivation.md'),
        affects='Theorem 8 (observer-MDL-selected d-periodic dominance) and its corollaries (d=3, k*=3, srs) are conditional on C1'),
    Gap('lambda_cc_factor_two',
        'Λ_CC factor-of-2 + w_DE LCDM — Phase-A cosmology',
        'Λ_CC = 3/N_hub² is right to a factor of ~2 vs the observed value; the '
        'extra factor isn\'t pinned without an LCDM-emulator translation '
        '(Phase A, no new substrate). Similarly w_DE is the LCDM-emulator value, '
        'not a from-substrate equation of state.',
        'open-bounded', residue='',
        clusters=('proofs/cosmology/cosmology_item*', 'proofs/cosmology/w_DE_*',
                  'an internal working note'),
        affects='the cosmological-constant prediction\'s precision; the N_hub-dependence (Λ ∝ N_hub⁻²) is structural'),
    Gap('layer1_escapes',
        'F3 / Layer-1 escapes — barren via audited channels',
        'The octonionic f_3 saturates at ~0.80 with a ~10⁻⁶⁰ suppression by the '
        'GUT epoch; the 7 access mechanisms M1-M7 (E_6→PS for one generation; '
        '22/24 I4_132 elements violate the octonion Φ; cooling instantaneous; '
        'f_3 ≈10⁻⁶⁰; M5/M6 need framework extensions; M7 saturates at 0.80) are '
        'audited NEGATIVE or UNCONNECTED. The saturated symmetry zoo is formally '
        'rich but observably barren via every audited channel. Recorded so the '
        'verdict is queryable, not so it\'s "open" in the do-this sense.',
        'closed-negative-via-audited-channels (verdict recorded)', residue='Layer-1 escape',
        clusters=('proofs/foundations/M{1,3,4,7}_*', 'proofs/foundations/theorem9_f3_quantification_on_srs.py',
                  'proofs/foundations/sector_f3_srs_explicit_computation.py'),
        affects='nothing — the framework\'s PS predictions are robust against all audited Layer-1 escapes'),
]

_BY_KEY = {g.key: g for g in GAPS}


# ---------------------------------------------------------------------------
# Registry API
# ---------------------------------------------------------------------------

def list_gaps() -> list[Gap]:
    """All frontier gaps (the simulator's boundary)."""
    return list(GAPS)


def get_gap(key: str) -> Gap:
    if key not in _BY_KEY:
        raise ValueError(f"frontier.get_gap: no gap {key!r}; have {sorted(_BY_KEY)}")
    return _BY_KEY[key]


def gaps_affecting_load_bearing_content() -> list[Gap]:
    """Gaps that, if closed, would upgrade currently-conditional load-bearing content."""
    return [g for g in GAPS if g.affects and 'nothing' not in g.affects.split(';')[0].lower()
            and 'out of' not in g.affects.split(';')[0].lower()]


def _raise(g: Gap):
    raise NotImplementedError(
        f"FRONTIER GAP [{g.key}] — {g.title}\n"
        f"  Status:   {g.status}\n"
        f"  Blocker:  {g.blocker}\n"
        f"  Residue:  {g.residue or '(none)'}\n"
        f"  Probes:   {', '.join(g.clusters)}\n"
        f"  Affects:  {g.affects or '(nothing load-bearing)'}\n"
        f"  See an internal working note §4.")


# ---------------------------------------------------------------------------
# Per-gap stubs (the explicit boundary; each raises with the precise blocker)
# ---------------------------------------------------------------------------

def d_gt_3_substrates():        _raise(get_gap('d_gt_3_substrates'))
def vram_cl6_fock_map():        _raise(get_gap('vram_cl6_fock_map'))
def beta_dark():                _raise(get_gap('beta_dark'))
def need_d3_species():          _raise(get_gap('need_d3_species'))
def mssm_as_adoption():         _raise(get_gap('mssm_as_adoption'))
def delta_cp_arg_h_path_b():    _raise(get_gap('delta_cp_arg_h_path_b'))
def fibers_for_non_srs_realizations(): _raise(get_gap('fibers_for_non_srs_realizations'))
def acoustic_scale():           _raise(get_gap('acoustic_scale'))
def gleason_genericity():       _raise(get_gap('gleason_genericity'))
def lambda_cc_factor_two():     _raise(get_gap('lambda_cc_factor_two'))
def layer1_escapes():           _raise(get_gap('layer1_escapes'))


def summary() -> dict:
    return {
        'n_gaps': len(GAPS),
        'by_status': {st: [g.key for g in GAPS if g.status.split()[0].strip('(') == st.split()[0].strip('(')]
                      for st in sorted({g.status for g in GAPS})},
        'load_bearing': [g.key for g in gaps_affecting_load_bearing_content()],
        'note': 'R-9 is NOT here — it is CLOSED (structural); see menus.crystal_nets.framework_substrate_selection().',
    }
