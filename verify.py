#!/usr/bin/env python3
"""
Verification runner for the standard-model-derivation framework.

Runs each backbone proof script as a subprocess and reports pass/fail.
All scripts are self-contained computational proofs that exit 0 on success.

Usage:
    python3 verify.py           # run all backbone proofs
    python3 verify.py --quick   # run only the 5 fastest proofs
"""

import subprocess
import sys
import time

# NOTE (panel C6, 2026-06-12): suite greenness is LOAD-CONDITIONAL -- the
# heavy spectral probes are compute-bound and can exceed their budget on a
# loaded machine; a TIMEOUT is not a gate failure. Heavy entries carry a
# per-entry timeout (4th tuple element, seconds).
# Backbone proofs: (category, script_path, short_description[, timeout_sec])
BACKBONE = [
    # Foundations
    ("foundations", "proofs/foundations/toggle_arity.py",
     "k*=3 optimal coordination number"),
    ("foundations", "proofs/foundations/dl_comparison.py",
     "srs unique by description length"),
    ("foundations", "proofs/foundations/srs_generation_c3.py",
     "Generation = C3 at P point"),
    ("foundations", "proofs/foundations/srs_p_point_algebra.py",
     "H^2 = k*I at P, Hashimoto eigenvalues"),
    ("foundations", "proofs/foundations/srs_ramanujan_theorem.py",
     "Ramanujan bound saturation"),
    ("foundations", "proofs/foundations/srs_foundation_closure.py",
     "All foundation theorems verified"),
    ("foundations", "proofs/foundations/zeta_factorization_srs_srsz_2026-06-10.py",
     "Ihara/Bass zeta + Z2 mirror-cover factorization (Phase 1.1)"),
    ("foundations", "proofs/foundations/zeta_channel_dictionary_probe_2026-06-10.py",
     "Counting+winding channels as zeta functionals; N_10=120, V_us=k*^3/N_10 (Phase 1.2)"),
    ("foundations", "proofs/foundations/phase1_3_s1_mirror_is_bodycentering_2026-06-11.py",
     "Mirror = body-centering translation; srs-z = folded srs = bipartite double (Phase 1.3 S1)"),
    ("foundations", "proofs/foundations/phase1_3_zeta_sectors_parity_mirrorgirth_2026-06-11.py",
     "Zeta sectors: parity theorem, mirror girth 3=k* on <111>, screw 4+8 (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_s3_winding_towers_2026-06-11.py",
     "Winding towers: 4_1 screw (4n:4) + C3 mirror (3n:3); u^8m = even screw windings (Phase 1.3 S3)"),
    ("foundations", "proofs/foundations/phase1_3_neutrino_mirror_saddles_2026-06-11.py",
     "Saddle orbit map: H = Gamma+Delta, P/N self-conjugate; holonomy inventory (Phase 1.3, panel-corrected)"),
    ("foundations", "proofs/foundations/phase1_3_trim_dichotomy_majorana_fork_2026-06-11.py",
     "TRIM dichotomy: P unique non-TRIM saddle (-P = P+Delta); Majorana fork 162.39 vs 27.05 (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_c3_characters_majorana_fork_2026-06-11.py",
     "C3 characters: H/Gamma Ramanujan = regular rep (H-reading alpha=0); P splits classes (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_c3_invariant_majorana_structure_2026-06-11.py",
     "C3-invariant Majorana bilinear: phase pi + degenerate pair => breaking required (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_od_nspillover_takagi_2026-06-11.py",
     "N-spillover OD refutation, Takagi-correct: 15+ placements, 0 passes (Phase 1.3, panel-ordered)"),
    ("foundations", "proofs/foundations/phase1_3_w49_koide_wiring_negative_2026-06-11.py",
     "NEGATIVE: W49/Koide zero-parameter nu-Dirac wiring fails R_nu (529/142724 vs 32.57) (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_L8_sector_census_2026-06-11.py",
     "L=8 sector census: 12x8 + 6x4 + 6x8 = 168 complete; address plural/unforced (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase1_3_delta_eps_budget_closure_2026-06-11.py",
     "delta/eps^2 budget closure: K2-PARTIAL; delta = (2/9)/{1,2,3}, eps^2_e = k*-1 (Phase 1.3)"),
    ("foundations", "proofs/foundations/phase2_1_grover_walk_konno_sato_2026-06-11.py",
     "Bloch-Grover unitary walker + Konno-Sato law; magic/tetrahedral anchors; mirror (Phase 2.1)"),
    ("foundations", "proofs/foundations/phase2_2_born_koide_weights_2026-06-11.py",
     "Born-Koide stage 1 (panel-corrected): weights derived; P2 PARTIAL, alignment lemma open (Phase 2.2)"),
    ("foundations", "proofs/foundations/phase2_2_saddle_uniqueness_koide_2026-06-11.py",
     "Q=2/3 P-unique among C3-fixed saddles; Gamma/H non-Koide (panel-promoted corroboration)"),
    ("foundations", "proofs/foundations/phase2_2_alignment_lemma_2026-06-11.py",
     "ALIGNMENT LEMMA: conjugate phases = Hermiticity of sqrt(M); Q=(1+eps^2/2)/3 (Phase 2.2 stage 2)"),
    ("foundations", "proofs/foundations/phase2_3_page_wootters_two_clocks_2026-06-11.py",
     "Two-clock conversion consistency note (PW framing, classical impl; Phase 2.3, grade-conflicted)"),
    ("foundations", "proofs/foundations/phase3_1_davies_gkls_compression_2026-06-11.py",
     "Lindblad DERIVED: Davies-scaled compression generators GKLS; unitality no-go (Phase 3.1)"),
    ("foundations", "proofs/foundations/phase3_2_s1_step_isometry_2026-06-11.py",
     "Step isometry: 3.1 channel = visible marginal of an explicit kept record (Phase 3.2 S1)"),
    ("foundations", "proofs/foundations/phase3_2_s2_dark_face_2026-06-11.py",
     "Dark face: edge labels canonically superselected; timing coherence FINDING (Phase 3.2 S2)"),
    ("foundations", "proofs/foundations/phase3_2_s3_s4_forced_leakage_2026-06-11.py",
     "Forced dark-sink leakage; LM1 p = 1-61e^-6 = Omega_DM zero-parameter (Phase 3.2 S3+S4)"),
    ("foundations", "proofs/foundations/phase3_3_ns_gate_2026-06-11.py",
     "n_s gate: rate density kappa-flat (null persists, L6 blocked); M1 m=1 lever (Phase 3.3)"),
    ("foundations", "proofs/foundations/phase5_1_little_groups_saddle_irreps_2026-06-11.py",
     "I4_132 little groups: P doublets FORCED (nontrivial cocycle), Gamma/H triplets FORCED, N free (Phase 5.1 S1-S3)"),
    ("foundations", "proofs/foundations/phase5_1_s4_spgrep_crosscheck_2026-06-11.py",
     "spgrep cross-check: orders 24/24/12/4 + menus confirmed, P all-even independent (Phase 5.1 S4; needs spgrep)"),
    ("foundations", "proofs/foundations/phase5_1_s5_ebr_decomposition_2026-06-11.py",
     "EBR layer: band rep = two midpoint EBRs (stages); adds NO forcing (gated negative) (Phase 5.1 S5)"),
    ("foundations", "proofs/foundations/phase5_1_s6_forced_vs_free_map_2026-06-11.py",
     "48-mode map: 24 LG-forced + 2 IB clusters; 8 family signatures distinct -> 0 free bits (Phase 5.1 S6)"),
    ("foundations", "proofs/foundations/phase5_2_repricing_enumeration_2026-06-11.py",
     "A5-mass tree verified: 8!->12->4->2->Z2=1.3-orientation-bit; row awaits panel (Phase 5.2)"),
    ("foundations", "proofs/foundations/phase5_2_ss22_grain_enumeration_2026-06-11.py",
     "A5-mass at the ledger-cited Sec-2.2 grain: 24->12->6 = 2.585 bits in-row; row 3.0 (Phase 5.2, panel-ordered)"),
    ("foundations", "proofs/foundations/phase4_1_spectral_triple_srsz_2026-06-11.py",
     "Even triple on srs-z: Phi gauge-covariant; UNDRESSED J forced crossing at P; KO-2 deck class (Phase 4.1)"),
    ("foundations", "proofs/foundations/phase4_2_heat_expansion_sectors_2026-06-11.py",
     "Heat expansion (panel-corrected): 3 sectors induced + sigma propagated; ladder m2/m4|m6|m8 (Phase 4.2)", 900),
    ("foundations", "proofs/foundations/phase4_3_beta_content_jeopardy_2026-06-11.py",
     "R-19 jeopardy (panel-scoped): b3 = -7 | anchor (2HDM); no gaugino seat in the triple (Phase 4.3)", 900),
    ("foundations", "proofs/foundations/phase4_e2_sigma_census_aliasfree_2026-06-12.py",
     "ERRATUM E2: alias-free sigma census BLOCK-DISCRIMINATING (octet sign-flip); GRID2 artifact shown", 900),
    ("foundations", "proofs/foundations/phase4_4_cS_mutual_information_2026-06-11.py",
     "c_S = 2 GIVEN named identification (ratification refused); cut-correlation = 2x record (Phase 4.4)"),
    ("foundations", "proofs/foundations/phase5_2_psign_omega_identity_2026-06-11.py",
     "ORDERED CHECK: P-sign bit = omega/omega2 convention (sign=class=conj content; mirror-paired); row 2.0 (Phase 5.2)"),
    ("foundations", "proofs/foundations/phase5_2_r1_perron_vev_uniform_mode_2026-06-11.py",
     "R1 Leg A: mean functional = Gamma-Perron coordinate (torus-exact, PF); row move awaits panel (Phase 5.2/R1)"),
    ("foundations", "proofs/foundations/phase5_3_b1_kitaev_srs_anchor_2026-06-11.py",
     "Kitaev-on-srs anchor: 6 colorings; Majorana FERMI SURFACE (codim-1); Z2 gauge at quadratic level (Phase 5.3 B1)"),
    ("foundations", "proofs/foundations/phase5_3_b2_gauge_sector_placement_2026-06-11.py",
     "Gauge sector = cycle space = the 18 trivial modes; 6 rings; Gamma anomaly + stable deficit 2 (Phase 5.3 B2)"),
    ("foundations", "proofs/foundations/phase5_3_b3_bridge_global_car_2026-06-11.py",
     "THE BRIDGE: global CAR from per-node Cl(6), link dressing only; D4 = end-root case (Phase 5.3 B3, panel-reworded)"),
    ("foundations", "proofs/foundations/phase5_3_b3b_lattice_extension_2026-06-11.py",
     "Lattice extension: middle-root automorphism; star CAR; cycle two-trees = gauge within flux sector (Phase 5.3 B3b)"),
    ("foundations", "proofs/foundations/dark_feshbach_a2_closure.py",
     "Dark correction c = 5/12 (theorem-grade, A2 winding series)"),
    ("foundations", "proofs/foundations/exponent_ladder.py",
     "Feshbach exponent ladder ((k-1)/k)^(g-n_fixed)"),
    ("foundations", "proofs/foundations/hashimoto_exponents.py",
     "Hashimoto exponent enumeration (K_4 + srs verification)"),

    # Gauge
    ("gauge", "proofs/gauge/cl8_verification.py",
     "Cl(6) = Cl(4) x Cl(2), gauge group"),

    # Flavor
    ("flavor", "proofs/flavor/srs_unified_mixing.py",
     "All PMNS angles from h"),
    ("flavor", "proofs/flavor/srs_final_pmns_theorem.py",
     "Self-consistent PMNS, chi2/dof = 0.22 (4 obs)"),
    ("flavor", "proofs/flavor/srs_hashimoto_seesaw_proof.py",
     "CP phases from Hashimoto eigenvalue"),
    ("flavor", "proofs/flavor/srs_ckm_tree_derivation.py",
     "CKM from tree approximation at z*=17/6"),
    ("flavor", "proofs/flavor/vus_l2_density.py",
     "V_us = 9/40 (theorem-grade, Level 2 counting density)"),
    ("flavor", "proofs/flavor/vcb_hashimoto_bfs.py",
     "V_cb = 256/6305 (theorem-grade, A2 geometric series)"),

    # Masses
    ("masses", "proofs/masses/koide_scale_proof.py",
     "Lepton masses from Koide formula"),
    ("masses", "proofs/masses/srs_mdl_meanfield_theorem.py",
     "MDL mean-field uniquely optimal"),
    ("masses", "proofs/masses/ytau_corollary.py",
     "y_τ = α₁_full/k*² (theorem, session 25)"),

    # Cosmology
    ("cosmology", "proofs/cosmology/srs_eta_b_exact.py",
     "Baryon asymmetry eta_B"),
    ("cosmology", "proofs/cosmology/dm_hierarchy_derivation.py",
     "Dark matter fraction and n_s"),

    # P2 parity sector
    ("parity", "proofs/cosmology/A_dilution_derivation.py",
     "A = 1/15 hemispherical + cubic moment (Theorems 1, 2)"),
    ("parity", "proofs/cosmology/path_c_beta_verify.py",
     "beta = sin(arg h) * alpha_EM (A-)"),
    ("parity", "proofs/cosmology/srs_photon_bloch_primitive.py",
     "B(P) doubly-degenerate h (Theorem 3)"),

    # Lorentz / LIV
    ("lorentz", "proofs/lorentz/hashimoto_dispersion_symbolic.py",
     "η_lattice = 1/12 dim-6 LIV (CAS-verified to 24+ digits)"),

    # Adapters — the grafts (verification contracts: foreign-theory axioms on the one object)
    ("adapters", "derivation_topdown/adapters/aqft_net.py",
     "G4 Haag–Kastler contract suite on the net {A(O)} (HK-0..7)", 600),
    ("adapters", "derivation_topdown/adapters/furey_stoica_labels.py",
     "G2 Furey–Stoica Witt-ladder labeling contract on the Cl(6) Fock (FS-0..6)", 600),
    ("adapters", "derivation_topdown/adapters/thermal_time.py",
     "G5a tick-KMS (Connes–Rovelli) contract: modular flow == the tick (KMS-0..6)", 600),
    ("adapters", "derivation_topdown/adapters/sunada_geometry.py",
     "G1 Kotani–Sunada standard-realization contract: b₁=3, BZ==Jacobian, isotropization (SR-0..5)", 600),
    ("adapters", "derivation_topdown/adapters/zeta_gauge.py",
     "G6a,b graph-zeta contracts: Bass on B, girth selection, det(I−uW_INT) == loop expansion (ZG-0..5)", 600),
    ("adapters", "derivation_topdown/adapters/ncg_spectral.py",
     "G3 NCG suite: KO anatomy 4+6≡2 ✓ (R2b) + the S2 Lagrangian bridge (zeta→spectrum→heat→a₄, AMPLITUDE-CONVERGENT)", 600),
    ("adapters", "derivation_topdown/adapters/reads_manifest.py",
     "S1 reads manifest (fast): ledger parse + Tier-A engine↔lock certification (zero mismatches)", 600),
    ("adapters", "derivation_topdown/adapters/quantum_foundations.py",
     "G7/S3+QF-2b quantum foundations: Born measured; smeared-Bell double-null (rank-1 lemma); flux closed; GKLS", 600),
    ("foundations", "proofs/foundations/S1d_epoch_api_2026-07-09.py",
     "S1d epoch API: N=time variable; calibration fence (19 excluded); era-explicit; locks/manifest untouched", 600),
    ("foundations", "proofs/foundations/I0b_RATIO_stage_BC_2026-07-10.py",
     "I-0b-RATIO: kappa-free binding ratios; ladder supercell(4)-STABLE; RATIO-MISS 13/3 booked OPEN; E_odd=0.3819 MeV measured; T0(mu_eff) named", 600),
    ("foundations", "proofs/foundations/I2_matsumoto_confront_2026-07-10.py",
     "I-2 Matsumoto/CK-KMS confront: SAME-OBJECT (QUALIFIED); beta*kappa = h_top/b_edge = ln2 all k; lit VERIFIED (EFW/OP/aHLRS-LN); full-algebra row pending", 600),
    ("foundations", "proofs/foundations/BRIDGE_LOCK_2026-07-10.py",
     "BRIDGE-LOCK (milestone II.2, route A): LENS-NULL theorem-grade -- the R1-forced O(2) family is band-edge-blind (3 lemmas); orbit ambiguity stands; hand-off to the modular-arrow route", 600),
    ("foundations", "proofs/foundations/IV4_T0_class_2026-07-10.py",
     "IV.4 T0-CLASS: NON-RIGID (T0=0.3311, 11.04% of U) + T1-FENCE (deep-contact exponent ~0.12, not linear-mu -> H/Ps re-homed to the connection sector); nuclear station stays live", 900),
    ("foundations", "proofs/foundations/BRIDGE_T_2026-07-10.py",
     "BRIDGE-T (II.2 route B): ARROW-BLIND theorem-grade -- symmetric-compression lemma: the R-odd run compression is forward/reversed-invariant; null extends to ALL state-level two-point data; hand to GEOM", 600),
    ("foundations", "proofs/foundations/MS1a_fusion_grading_2026-07-10.py",
     "MS-1a (III.2): NO-ADDITIVE-CHARGE THEOREM computed -- Z-charge nullity 0 on R(A4)/R(2T); fermion parity = the UNIQUE nontrivial Z2; no exact proton protection (consistent w/ eta_B Sakharov)", 600),
    ("foundations", "proofs/foundations/T0_NUCLEAR_2026-07-10.py",
     "T0-NUCLEAR gate (IV.4): CONSTRUCTION-MISMATCH stop-clause fired -- rung = overlap-info MINUS branch cost (3=5-2, 13=15-2 uniform); II3=0 on ALL ground triples; no 3-body number produced", 600),
    ("foundations", "proofs/foundations/T0_NUCLEAR2_2026-07-10.py",
     "T0-NUCLEAR-2 (IV.4): gate PASS (dS rungs 3/13 by construction) but KIN-WRONG-WAY booked raw -- solver collapsed onto coincidence (domain UNDEFINED by the certified ladder); T0-N-3 gated on the domain theorem", 900),
    ("foundations", "proofs/foundations/LE1_low_entropy_composition_2026-07-10.py",
     "LE-1 (III.3): COMPOSED -- S_register(N) <= N*b_edge; S(1)=0 exact; cold-start ~0.03 bits at u=alpha_1; three premises explicit; LE-2 bridge stays open", 600),
    ("foundations", "proofs/foundations/I2b_matsumoto_completion_2026-07-10.py",
     "I2b (II.5): COMPLETION -- S_d on word-Fock H_hist; Toeplitz defect 1-P_seed EXACT; run diagonal sharply KMS at beta_natural = 2*beta_gas (= beta' + h_top, per-path vs per-shell resolved); imports attach", 600),
    ("foundations", "proofs/foundations/CS0b_wint_redecoration_2026-07-10.py",
     "CS-0b (IV.7): W_INT re-decoration SURVIVES both branches (chiral asymmetry decoration-robust; magnitude was convention's; orbit ambiguity = overall sign); M-4 defect discharged", 600),
    ("foundations", "proofs/foundations/BRIDGE_GEOM_2026-07-10.py",
     "BRIDGE-GEOM (II.2 route C): K-DEPENDENT (branches live at Gamma/H, die at P/N) + ENANTIOMER-BLINDNESS THEOREM (I4_332 = exact conjugate problem at every k) -> MIRROR-REQUIRED excluded BY PROOF; T-like reframe; bridge program EXHAUSTED", 600),
    ("foundations", "proofs/foundations/X2b_flatband_angular_structure_2026-07-11.py",
     "X.2-b (IV.8/clock map): MEASURE-ZERO -- flat-band exactly-flat directions = 3 coordinate great circles (zero solid angle); X.2-a solid-angle-averaged-curvature premise VALIDATED", 600),
    ("foundations", "proofs/foundations/CS1_finite_k_propagator_2026-07-11.py",
     "CS-1 (IV.7): FORCED-MASSLESS-TRANSVERSE -- transverse p^2 coefficient nonzero/isotropic/grid-stable, u^8 girth-selected (the Maxwell rung exists); longitudinal exactly p-independent (S-3c, lemma pending); NO alpha_EM claim, scheme identification = named incomplete", 600),
    ("foundations", "proofs/foundations/FOCK0_dr_construction_2026-07-11.py",
     "FOCK-0 (II.2/II.4/IV.7): V4 INCONCLUSIVE-BLOCKED (architect adjudication w/ verifier, overturning the in-file proposed V3) -- I2b algebra ACCRETED (the_net Sec 7b) + sector grading + per-sector J_sigma constructed; generator-route Hom=0 = projectivity-mismatch LEMMA (M-1b-class reinforcement, NOT the frozen modular-conjugation class); GATE = the dart-side modular conjugation (FOCK-0b); conditional on ML-2b TD-limit twisted Haag duality", 600),
    ("foundations", "proofs/foundations/FOCK0c_full_pin_2026-07-11.py",
     "FOCK-0c (II.2): full-J per-sector-pair Tomita + R/F_bit pin -- nullity 16/384 shell 1 (GROUP-03=16, GROUP-12=0 algebraic), 64/768 shell 2; W3 above-allowance, excess = named incomplete equation; conditional on ML-2b TD-limit twisted Haag duality", 1800),
    ("foundations", "proofs/foundations/X2a_native_bz_2026-07-11.py",
     "X.2-a native-BZ (IV.8/clock map): N1 NO-CROSSING -- band bounded [-0.7321, 2.0000], rho_cone > rho_flat for all beta (peak ratio 0.885); 3rd no-global-KMS-crossing confirmation; X.2-c gated", 600),
    ("foundations", "proofs/foundations/CS1b_longitudinal_lemma_check_2026-07-11.py",
     "CS-1b (IV.7): PROVEN theorem-grade -- longitudinal exactly p-independent via Woodbury collapse; transverse provably escapes the collapse (12/12 checks)", 600),
    ("foundations", "proofs/foundations/T0N3_domain_theorem_2026-07-11.py",
     "T0-N-3 (IV track): D2 NO-EXCLUSION -- CAR cannot be STATED on first-quantized history space (missing premise = the weld); BANKED: unconditional orientation-compatibility lemma 648/648 pairs + 216/216 triples unique resolutions, coincidence collides 10/10", 600),
    ("foundations", "proofs/foundations/FOCK0d_temporal_pin_2026-07-11.py",
     "FOCK-0d (II.2/II.3): W2 CLOCK-INCOMMENSURABILITY OBSTRUCTION THEOREM (verifier confirmed) -- flow pin empty for ALL lambda>0, all 4 region orbits; K_hist scalar-per-shell c2=2c1 vs K_F single magnitude {0x4,+-eps x2}, C_A eigenvalue 1/2 forced; II.3 M_Z = PROVEN FENCE; ember: single-shell maps at lambda*=c1/eps=2.463; ML-2b/HK-7-conditional", 600),
    ("foundations", "proofs/foundations/CS2_coulomb_gate_2026-07-11.py",
     "CS-2 (IV.7): Part A CLASS-MISS raw -- Coulomb-shape kernel at deep operating point does NOT flip the mu-scaling class (exp +0.1118 vs +1); exponent climbs monotonically to +0.71 at g=0.3 = the class flip is a WEAK-COUPLING property, shape alone insufficient; single blocker for class-flip/atomic-block/quantitative-E_odd/Delta-alpha = the connection scale bridge (= CS-1 scheme identification); E_odd SIGN-CONSISTENT qualitative", 900),
    ("foundations", "proofs/foundations/A2d_minimal_weld_2026-07-12.py",
     "A2d (Push 2 FINAL): D2 NOT-FORCED -- THE WELD EXISTS as a mapped CP^2 family (minimal class grading+equivariance; nullity 6 real vs allowance 2; full 3-copy multiplicity space, two solver routes agree) but the DIRECTION is unselected = THE MULTIPLICITY SELECTOR (final named incomplete equation); coverage levels 1+2+3 of F, conjugate weld confined to level 2, image couples to all region-clock eigenspaces incl +-eps; P1/P2 non-vacuous and hold; tower = consistency not forcing; ML-2b/HK-7-conditional", 600),
    ("foundations", "proofs/foundations/A2c_equivariant_weld_2026-07-12.py",
     "A2c (Push 2): C3 EMPTY-AS-FROZEN (verifier confirmed; Schur/character-orthogonality proof) -- the pair-block pin confines the survivor to two 1-dim A4 characters, never equivariantly into the 3; 4th distinct obstruction mechanism; checker classification: pair-block pin = inheritance-only in a J-free class + D4 tension (species pairing as input); pair-block-FREE minimal class NONZERO (3 complex dims, all in the 3-isotypic, tower-consistent); ML-2b/HK-7-conditional", 600),
    ("foundations", "proofs/foundations/A2b_conjugate_pair_2026-07-12.py",
     "A2b (Push 2): B3 OBSTRUCTED-AS-FROZEN (verifier 9/9, trajectory 144->24->0) -- the inherited R/F_bit dictionary is the killer (pair-block survivor reversal-EVEN vs level-1 fermion-ODD); bare conjugate-pair class has 24 dims; graded-functor fermion parity = LENGTH parity automatically; L0a all level-additive self-J welds die beyond shell 1; L0b region clock can never pin the tower (N-hat only, rate c1); L0c reversed flow; ML-2b/HK-7-conditional", 600),
    ("foundations", "proofs/foundations/A2_weld_functor_2026-07-12.py",
     "A2 (Push 2, the weld): AF-3 OBSTRUCTED theorem-grade (verifier 9/9) -- level-preserving J-pinned functor class EMPTY (phi_1=0 forced, 0/384): field J = antiparticle pairing (levels 0<->3, 1<->2) vs shell-preserving history J = GRADING-PARITY MISMATCH; P1 Pauli truncation + P2 exclusion-in-image verified; J-orbit-pair diagnostic 144/384 at shell 1 ONLY (3rd shell-1-only ember); ML-2b/HK-7-conditional", 600),
    ("foundations", "proofs/foundations/FOCK0e_clock_linearity_2026-07-12.py",
     "FOCK-0e (A1, Push-1 W1): CLOCK LINEARITY PROVEN -- K_hist per-shell = c_n*I with c_n = n*c1 an algebraic identity from omega_diag's literal u^{2|w|}/Z form (truncation-independent); c1 = -2 ln alpha_1 = 16 ln(3/2) = beta_natural; machine-exact to shell 8 (worst 1.8e-15), Tomita route matches to 0; lambda_n = n*lambda_1 booked as STRUCTURE per region orbit (lambda_1 triangle = 2.463 = 0d ember); per-shell aggregate clock exactly affine at beta' (derives per-path/per-shell h_top identity); ML-2b/HK-7-conditional", 600),
    ("foundations", "scripts/export_bridge_vectors.py",
     "CB-1 bridge kit: exported cross-repo vectors == live recomputation (check mode; zero physics)", 300),
    ("foundations", "proofs/foundations/B2a_density_response_2026-07-09.py",
     "B2-a density response: chi0 Lindhard (net 4d regression); M2b-object finding; Mermin=diffusion", 600),
    ("foundations", "proofs/foundations/W2_BGK_two_moment_2026-07-10.py",
     "W2-BGK (fast): velocity vertex + two-moment conserving closure regression (NO-SOUND station)", 600),
    ("foundations", "proofs/foundations/W2_GAUGE_abelian_a2_2026-07-10.py",
     "W2-GAUGE-A (fast): triviality gate + magnetic-supercell construction checks (a2 WINDOW-LIMITED)", 600),
    ("foundations", "proofs/foundations/R1_HARVEST_2026-07-10.py",
     "Ring-1 harvest: coasting ladder + composites + m_bb band + structural wiring (locks 134)", 600),

    # Push 3 hygiene (W1 integration batch, 2026-07-13)
    ("foundations", "proofs/foundations/CA_half_lemma_check_2026-07-12.py",
     "W3a: C_A = 1/2 lemma standalone machine check (conj(C_A)=I-C_A mechanism; 31/31)", 120),
    ("foundations", "proofs/foundations/ML1d_selftest_regression_2026-07-13.py",
     "ML-1d/ML-1d-b Section 9/9b FAST regression (underlying functions directly, small sizes; "
     "NOT the station -- its R2 gate exits 1 by design, INSTRUMENT-CLOSED, intentionally NOT "
     "wired; NOT the full net self-test chain -- station-scale, architect-scoped out)", 120),

    # Push 3 hygiene (L9 verify.py wiring batch, 2026-07-14): the three station selftests that
    # were written self-contained and fast but explicitly left un-wired at write time.
    ("foundations", "proofs/foundations/W2_selftest_wire_2026-07-13.py",
     "W2 (Sec 10) selftest wired: welded-state construction, EXACT level-1 Schur mechanism + "
     "level-2 numeric confirmation, direction independence, T1/T2 well-posedness, honesty clause, "
     "PAIR J_F-conjugation identity", 120),
    ("foundations", "proofs/foundations/V1_selftest_wire_2026-07-13.py",
     "V1 (Sec 11) selftest wired: occupation transform, channel-state exactness, purity-reduction "
     "lemma, copy-overlap/Holevo identity, pair-functional phase drop + conj-invariance, pinned "
     "regression value 0.9361048420", 120),
    ("foundations", "proofs/foundations/FOCK2_selftest_wire_2026-07-14.py",
     "FOCK-2 (Sec 12) selftest wired: dim-1 sectors exactly blind, per-shell F2/F3 decomposition "
     "reconstructs the aggregate, per-sector read gauge-invariant across the triad", 120),

    # GEN-IDENT arc (2026-07-15): both routes to selecting the generation labeling from within the
    # substrate exhausted theorem-grade -- kinematic (A/B/C/D0/D1/D2) + dynamical (beta). Wired in
    # the 2026-07-16 hygiene pass (queued verify wiring cleared; all self-contained, goal-seek clean).
    ("foundations", "proofs/foundations/genident_A_offset_check_2026-07-15.py",
     "GEN-IDENT-A: the vertex-triad vs winding-isotype relative offset is FORCED (no full-construction "
     "symmetry carries v2->v0 while preserving grading + selector); S4 walk-symmetry scope (27/27)", 120),
    ("foundations", "proofs/foundations/genident_B_observer_residual_check_2026-07-15.py",
     "GEN-IDENT-B: <sigma,W>=A4 irreducible => joint commutant scalar => an observer respecting both "
     "collapses S3 -> Out(A4)=Z2 (one bit); but the triggering coupling is UN-BUILT (counterfactual) (22/22)", 120),
    ("foundations", "proofs/foundations/genident_C_coupling_check_2026-07-15.py",
     "GEN-IDENT-C: no forced numeric home for the observer C^3 on the finite H_hist(x)F carrier; "
     "exterior-power lift leaves a 24-dim moduli no pure-structure criterion selects (24/24)", 120),
    ("foundations", "proofs/foundations/genident_C_verification_2026-07-15.py",
     "GEN-IDENT-C independent verification: five force-the-anchor attempts all fail; Skolem-Noether "
     "obstruction upgraded to load-bearing (finite irreducible Fock carrier => every automorphism inner) (38/38)", 120),
    ("foundations", "proofs/foundations/genident_D_outerness_check_2026-07-15.py",
     "GEN-IDENT-D0: the winding-C3 alpha is PROPERLY OUTER on the type-II1 factor M=L(F_inv(6))~L(F_4), "
     "non-vacuously (Fourier/twisted-conjugation, every twisted class infinite by word-length growth) (54/54)", 120),
    ("foundations", "proofs/foundations/genident_D1_canonical_home_check_2026-07-15.py",
     "GEN-IDENT-D1: relative commutant M'&(M x|_alpha Z3)=C => the M3(C) observer leg is RIGID, no moduli "
     "(outerness forces each graded piece rank-1, the Sum d_i^2 over-count vanishes) (33/33)", 120),
    ("foundations", "proofs/foundations/genident_D2_half_descent_check_2026-07-15.py",
     "GEN-IDENT-D2 leg1: <sigma> self-normalizing Sylow-3 of A4 & <sigma,W>=A4 => W fails crossed-product "
     "descent => the canonical M3(C) home carries sigma but has NO forced W-action (32/32)", 120),
    ("foundations", "proofs/foundations/genident_D2_leg2_no_forced_coupling_check_2026-07-15.py",
     "GEN-IDENT-D2 leg2: adversarial pass confirms NO forced coupling exists (no forced *-embedding "
     "F -> M; tau-GNS shadow maximally-mixed U(3)-blind) => D2 orthogonal-forced, the II1 route closed (12/12)", 120),
    ("foundations", "proofs/foundations/genident_beta_endpoint_vertex_check_2026-07-15.py",
     "GEN-IDENT-beta: the substrate vertex -kappa*I(A;B)(s) is EXACTLY CONSTANT in the run-endpoint s -- "
     "s enters as a per-mode phase => local-unitary orbit => bipartite MI invariant (blind-by-theorem) (19/19)", 120),
]

# Quick subset: fastest-running proofs for rapid checks
QUICK = [
    "proofs/foundations/toggle_arity.py",
    "proofs/gauge/cl8_verification.py",
    "proofs/cosmology/dm_hierarchy_derivation.py",
    "proofs/foundations/srs_foundation_closure.py",
    "proofs/masses/koide_scale_proof.py",
]


def run_proof(path, description, timeout_sec=360):
    """Run a proof script, return (pass, elapsed, output)."""
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True,
            timeout=timeout_sec
        )
        elapsed = time.time() - t0
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        return passed, elapsed, output
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return False, elapsed, f"TIMEOUT after {timeout_sec}s"
    except Exception as e:
        elapsed = time.time() - t0
        return False, elapsed, str(e)


def main():
    quick = '--quick' in sys.argv
    proofs = [(e[0], e[1], e[2], e[3] if len(e) > 3 else 360) for e in BACKBONE
              if not quick or e[1] in QUICK]

    print("=" * 72)
    print("  Standard Model Derivation -- Verification Suite")
    print("=" * 72)
    print(f"  Running {'quick' if quick else 'full'} suite: "
          f"{len(proofs)} proofs\n")

    results = []
    total_time = 0

    for category, path, description, t_budget in proofs:
        sys.stdout.write(f"  [{category:12s}] {description:45s} ... ")
        sys.stdout.flush()

        passed, elapsed, output = run_proof(path, description, t_budget)
        total_time += elapsed
        results.append((category, path, description, passed, elapsed))

        status = "PASS" if passed else "FAIL"
        print(f"{status}  ({elapsed:.1f}s)")

        if not passed:
            # Show last 5 lines of output on failure
            lines = output.strip().split('\n')
            for line in lines[-5:]:
                print(f"         {line}")

    # Summary
    n_pass = sum(1 for _, _, _, p, _ in results if p)
    n_fail = len(results) - n_pass

    print()
    print("=" * 72)
    print(f"  RESULTS: {n_pass}/{len(results)} passed, "
          f"{n_fail} failed, {total_time:.1f}s total")
    print("=" * 72)

    if n_fail > 0:
        print("\n  FAILURES:")
        for cat, path, desc, passed, _ in results:
            if not passed:
                print(f"    - {path}: {desc}")
        return 1

    print("\n  All proofs verified successfully.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
