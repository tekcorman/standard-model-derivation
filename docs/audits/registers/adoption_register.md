# Adoption Register

> **⚑ CURRENT STATUS — read this first (2026-07-01).** This register is append-stateful: the
> "Active adoptions (N, post-…)" lines below are HISTORICAL snapshots, not the live count. The live
> status is the **★ ACTIVE-ADOPTIONS AUDIT (2026-07-01)** near the end of this file: of the named
> adoptions, only **two are substantively open and load-bearing** — **ADOPTED-DARK-MAP** (narrowed to
> β cosmic-birefringence + θ₁₃ PMNS) and **ADOPTED-NU-MAJ-PHASE** (the h^g Majorana phase, gating the
> unmeasured α₂₁/α₃₁). The rest are residual or non-load-bearing: ADOPTED-B3 (down to the
> lepton-vs-quark sector-label residue), ADOPTED-A5b-Sub3 (classifier un-graduated; the amplitudes it
> gated ship via independent derivations), ADOPTED-MSSM-Sb (β values now derived top-down; only the
> literal-sparticle interpretation residue remains), ADOPTED-K_P-TIEBREAK (vacuous for srs).

**Date:** 2026-04-28 (ADOPTED-A5b-Sub3 added).
**Purpose:** Canonical list of structural inputs accepted as adoptions — things the framework uses but has not derived from A1+A2+A3+A4 alone. An adoption is not an axiom and not a theorem. Every prediction pair that depends on an adoption must flag it explicitly (Feshbach pattern).

**A-IT3 (Landauer κ) GRADUATION (2026-07-07, M0-2R).** The Landauer conversion constant κ = k_B·T·ln2 (A-IT3, the OEF's information→energy constant) is no longer an external import: M0-2R derives it framework-internally as **κ = h/t_P**, dimensionless content forced. The three pieces — **ln2** (currency-consistency `β·κ=ln2`, = the multiway path-gas critical point `u_c=2^{−b_edge}`; T3), **T = the substrate tick temperature** (the run state is an exact KMS/Gibbs state of the tick number N̂; thermal time = the tick; T1), and **the 2π** (N̂ integer ⟹ modular flow is a compact U(1) of period 2π ⟹ one tick = one full-loop action quantum = h; T4) — are all derived. Only `t_P` (the standing time anchor) and the currency ontology (`E ∝ L`) remain as inputs, neither an adopted external number. A-IT3 retained as a cross-check. See `../../theorems/theorem_observer_energy_functional.md` §9 + internal research notes, internal research notes. No scoreboard value moved (κ's magnitude was already `h/t_P` in A1; this closes its *derivation*).

**Post-A5 update (2026-04-19).** A5 (`../../framework/framework_axioms.md` §5b) was adopted: "The framework is a theory of Standard Model particle physics." ADOPTED-P1 and ADOPTED-Y are both downstream restatements of A5 and are now CLOSED. ADOPTED-Z3 and ADOPTED-B3 remain active. Two new entries added: dark-map classification and Pati-Salam neutrino bare scale.

**Post-R3 update (2026-04-20, Sprint β).** ADOPTED-Z3 graduates from `adopted` to `mathematically complete` via `predictions/R3_observer_c3_generation.py` (L2 closed as pure rep theory; L3 uses observed charged-lepton mass non-degeneracy as external input under A5(a)). One adoption item moves off the active list.

**Post-α/β/γ grand update (Sprint α session 4 + Sprint β session 5 + Sprint γ session 6):**
- 2026-04-20 α: ADOPTED-DARK-MAP 5/3 coefficient graduated (classification residual remains active).
- 2026-04-20 β: ADOPTED-Z3 graduated (R3 via Observer-C^3).
- 2026-04-20 γ: Sprint γ (3b) CKM structural blocker CLOSED via `predictions/B3_chirality_bridge.py` (matching-partition Cartan from S_4 invariance). ADOPTED-B3 remains active but the specific sector-universality obstruction behind V_us = V_cb = V_ub = 0 is now resolved at the structural level.

**Post-A5b-Sub3 update (2026-04-28).** New adoption ADOPTED-A5b-Sub3 added: A5(b) Level 3 sub-class identification (within-Level-3 split between walk-rep, Moore-slot, and generation-distinguishing factor). Justified by convergent negative results from Routes 1 + 1' of an internal working note (CAS-verified at `proofs/foundations/a5b_route1_z6_directed_edge_probe.py` + `a5b_route1prime_z2_centralizer_probe.py`). Resolves 6 parameter ledger rows from BLOCKED to CONDITIONAL-on-adoption (P14, P15, P32, P33-partial, P34-partial, P45).

**Post-bridge-functoriality update (2026-04-28, same day).** **ADOPTED-A5b-Sub3 GRADUATED to theorem grade** via `../../theorems/theorem_bridge_functoriality_lemma.md`. The Level 3 sub-class classifier — specifically the generation-distinguishing factor (sub-class iii) — is now derivable from the bridge functoriality lemma (which composes V_cb's k=1 base case + multiway functoriality + 16-cycle decomposition CAS uniqueness). The 6 parameter rows that were CONDITIONAL-on-adoption now graduate to STRICT-SOLID under the bridge theorem (P33 and P34 still have separate non-sub-class gaps for PS embedding and arg(h) Path B'' respectively). V_ub = 3.767e-3 (−0.26σ from PDG combined excl+incl) ships theorem-grade.

**RETRACTION (2026-04-29).** The 2026-04-28 graduation of ADOPTED-A5b-Sub3 to theorem grade via `../../theorems/theorem_bridge_functoriality_lemma.md` is **RETRACTED**. The lemma's load-bearing structural argument (Z₃^m holonomy accumulation on m-cycle hosts) is refuted by three independent CAS findings:
- (R1) `proofs/flavor/z3_holonomy_cycles.py`: Z₃ connection on srs is FLAT (also load-bearing for Row P16 θ_QCD = 0). No Z₃ phase accumulates on any cycle.
- (R2) `proofs/flavor/vub_bridge_higher_m_pinning_probe.py`: every m-host class admits same-orbit pinned pairs at every lower-m diagonal cycle-distance — pinning topology is shared, not segregated.
- (R3) `proofs/flavor/vub_bridge_z3_shift_classifier.py`: same-orbit pairs split 50/50 between Z₃-shift (b₂=C₃b₁) and Z₃²-shift (b₂=C₃²b₁) at every (m, d) tested. Z₃ vs Z₃² distinction does not segregate ΔGen=1 from ΔGen=2.

The 6 parameter rows revert to **STRICT-SOLID conditional on ADOPTED-A5b-Sub3** (their post-2026-04-28 AM, pre-bridge-graduation grade). V_ub still ships at value 3.767×10⁻³ from the working multi-cycle sum formula (`proofs/flavor/vub_multicycle_sum.py`); the formula is empirically sound, but the substrate-side structural identification is genuinely open. See an internal working note for the deeper structural gap and the M1 (Bloch eigenmode) + M2 (multiway formalism) research routes.

**Active adoptions (4, post-retraction):** ADOPTED-B3 (Pati-Salam labeling), ADOPTED-DARK-MAP (β + θ_13 PMNS scope), ADOPTED-PS-SCALE, ADOPTED-A5b-Sub3 (Level 3 sub-class classifier — un-graduated).

**Post-2026-05-04 update — ADOPTED-PS-SCALE CLOSED.** m_ν₃ graduated to UNIQUE-THEOREM-GRADE-CONDITIONAL via global spectral-gap formula m_ν₃ = (k*·N_atoms) × M_Pl × N_hub^(-1/2); see `predictions/m_nu3.py` and the entry below. Active adoption count: 4 → 3.

**Active adoptions (3, post-2026-05-04):** ADOPTED-B3 (Pati-Salam labeling), ADOPTED-DARK-MAP (β + θ_13 PMNS scope), ADOPTED-A5b-Sub3 (Level 3 sub-class classifier — un-graduated).

**Post-cascade-Step-5 audit update (2026-05-06).** New adoption **ADOPTED-COSMOLOGICAL-IC-AMPLITUDE** added: the cosmological preferred axis ẑ has frozen anisotropy amplitude ε_toggle (= 1/5) at all cosmological epochs. Justification: Bridge 1 derives α_IC = ε_toggle at the N=1 boundary (theorem-grade-conditional, `proofs/cosmology/cascade_step5_claim_A_n_eq_1_BC.py`); persistence to N_hub is structurally undetermined under direction-uniform renewal Markov dynamics (5 routes closed in `cascade_step5_compression_integral_session1_scoping_2026-05-06.md`); 4-observable empirical match (H_0 SH0ES, A_dilution, t_0 substrate, A_s) at +0.18σ joint excludes alternatives ε/2 at 2.93σ and 2ε at 5.32σ. Resolves the inconsistency between Row P27 (A_hemis, currently graded UNIQUE-THEOREM-GRADE) and Rows P19/P20/P24 (cascade D2-extended, graded UNIQUE-THEOREM-GRADE-CONDITIONAL on Step 5). Active adoption count: 3 → 4. See ADOPTED-COSMOLOGICAL-IC-AMPLITUDE entry below for the canonical statement.

**Active adoptions (4, post-2026-05-06):** ADOPTED-B3 (Pati-Salam labeling), ADOPTED-DARK-MAP (β + θ_13 PMNS scope), ADOPTED-A5b-Sub3 (Level 3 sub-class classifier — un-graduated), **ADOPTED-COSMOLOGICAL-IC-AMPLITUDE** (cosmological preferred axis amplitude = ε_toggle, persistence open).

**Post-2026-05-07 update — ADOPTED-COSMOLOGICAL-IC-AMPLITUDE GRADUATED.** Closed at theorem grade via `../../theorems/theorem_observer_persistence_closure_IC_amplitude.md`. The persistence of α = ε_toggle from N=1 IC to N_hub is derived as composition of A1 → P1' (theorem) + A2-T waterline (theorem) + Bridge 1 (theorem-grade-conditional, Claim A) + DL accounting probe (`proofs/cosmology/observer_persistence_DL_accounting.py`, M_IC clears the waterline by ~10⁵⁹·⁴ bits margin). The closure operates under the framework's observer-MDL primary posture (post-2026-05-02 axiom slate {A1} alone): cosmological observables are functionals of the observer's compressed cosmological model, not direct readouts of substrate-Markov-stationary distributions. The prior 5-route audit's substrate-primary NEGATIVE remains valid as a substrate-side fact but does not block the observer-side closure. The "proves too much" concern is resolved by the IC-set vs operator-level structural partition: cosmological observables couple to global IC-set facts; particle observables couple to local operator-level facts. Pattern verified across the framework's existing predictions ledger. Rows P19, P20, P24 (rate-gap component), P27 graduate from UNIQUE-THEOREM-GRADE-CONDITIONAL to UNIQUE-THEOREM-GRADE. Active adoption count: 4 → 3.

**Active adoptions (3, post-2026-05-07):** ADOPTED-B3 (Pati-Salam labeling), ADOPTED-DARK-MAP (β + θ_13 PMNS scope), ADOPTED-A5b-Sub3 (Level 3 sub-class classifier — un-graduated).

**Post-2026-05-11 update — ADOPTED-MSSM-Sb added.** SU(2)_L Wilson-loop probe (an internal working note, probe `proofs/foundations/substrate_rg_beta_function_su2.py`) closes the last bounded route to deriving MSSM β-coefficients structurally from substrate. Combined with prior closures (Path A INOPERATIVE per `susy_path_a_anomaly_cancellation.py`; Path E BLOCKED per `susy_path_e_witten_substrate.py`; Path D PARTIAL — numerical necessity only per `mssm_matter_content_required.py`), no identified theorem-grade route to deriving MSSM matter content from substrate. Framing (a) of `per_sector_substrate_beta_function_gap_inventory_2026-05-11.md` becomes the linter-consistent endpoint: MSSM matter content is empirical input alongside the adopted dimensional input N_hub (value pinned via the measured G_F). New adoption ADOPTED-MSSM-Sb codifies this; Rows P63-P70 reframe from DOMINANT-CONDITIONAL on "Layer 5 closure with no identified route" to UNIQUE-THEOREM-GRADE-CONDITIONAL on (ADOPTED-MSSM-Sb, G_F) jointly. Active adoption count: 3 → 4.

**Active adoptions (4, post-2026-05-11):** ADOPTED-B3 (Pati-Salam labeling), ADOPTED-DARK-MAP (β + θ_13 PMNS scope), ADOPTED-A5b-Sub3 (Level 3 sub-class classifier — un-graduated), **ADOPTED-MSSM-Sb** (MSSM matter content as RG-running scheme).

---

## ADOPTED-P1: Ramanujan subspace support — CLOSED via A5 (2026-04-19)

**Formerly stated.** Mass amplitudes supported on V_Ram rather than V_tree.

**Closure.** A5 declares V_Ram eigenvalues ARE the SM mass spectrum. Mass content is supported on V_Ram by definition; V_tree eigenvalues ±1 are outside the SM spectrum. The MDL discriminability argument (`docs/theorem_P1_ramanujan_support.md` §6) proves this is the only A2-consistent form. Label retired; no independent adoption remains.

---

---

## ADOPTED-Z3: C_3 Fourier index = generation index — GRADUATED via R3 (2026-04-20)

**Graduation event.** Session 5 (Sprint β) — `predictions/R3_observer_c3_generation.py` closes ADOPTED-Z3 at `mathematically complete` grade via the Observer-C^3 route. L2 (U(3)-conjugacy uniqueness of the regular Z_3 representation) closed as pure rep theory via the spectral theorem (Halmos 1958 §83), CAS-verified on 50/50 Haar-random trials at residual ~1e-15. L3 uses observed charged-lepton mass non-degeneracy (PDG 2024) as listed external input under A5(a); upgrade to `theorem` grade pending a separate derivation of M_gen non-degeneracy from A1-A5 alone (Sprint 11 B7.3 territory).

**Refined statement (post-graduation).** The SM generation-Z_3 symmetry is the canonical cyclic-shift Z_3 ⊂ U(3) acting on the observer's n=3 Hilbert space C^3_obs from `predictions/observer_dim_three.py`. The three basis vectors of C^3_obs (in the mass basis) are the three physical fermion generations. The srs body-diagonal C_3 on V_Ram (via B6 bridge lift to SU(4)) is a DIFFERENT C_3, whose physical interpretation remains at fallback (β) pure algebraic SU(4) Cartan label per `../../framework/B3_B6_reconciliation.md`. The two C_3's might coincide under further identification, but R3 does not require them to.

**Historical record — prior routes 1-4 (now superseded or sharpened):**
- Route 3 (T = A_4 unique 3-dim irrep plus T-equivariance): DEFINITIVELY BLOCKED (an internal working note, non-split Z_4 central extension).
- Route 4 (B3 Pati-Salam descent): DEFINITIVELY FAILED (C_3 and Cartan bases mutually incompatible, per `../../framework/B3_B6_reconciliation.md`).
- Route 1 (B7.2 layer architecture): SHARPENED into R3 Observer-C^3 closure (4 load-bearing steps L1-L4; L2 closed).
- Route 2 ((4,2,2) MDL asymmetry): superseded (R3 does not rely on V_Ram's (4,2,2) structure).

**Original motivation (preserved for record).** Color identification is definitively ruled out: the body-diagonal C_3 of srs lifts to diag(1,1,omega,omega^2) on SU(4), which is not in Z(SU(3)_color) under any standard Pati-Salam embedding. Generation was the only remaining candidate among the (A), (β), (γ) alternatives of `../../framework/B3_B6_reconciliation.md`.

**Downstream effect.** Files chain-importing the generation label (Koide family: `Q_Koide.py`, `epsilon_Koide.py`, `delta_Koide.py`; retracted PMNS files) should now chain-import `predictions/R3_observer_c3_generation.py` for the generation label instead of citing ADOPTED-Z3. Their residual lists shrink by one adoption item.

---

## ADOPTED-Y: Mass scale from Higgs VEV × Yukawa couplings — CLOSED via A5 (2026-04-19)

**Formerly stated.** The mass scale M in the Koide formula is set by the Higgs VEV times the Yukawa coupling.

**Closure.** Under A5, the framework's Bloch-fiber amplitudes at each vertex are identified with Yukawa couplings (A5 applied to the Yukawa sector). The overall scale follows from the Higgs mechanism, which is part of the SM identification declared by A5. Label retired; no independent adoption remains.

---

## ADOPTED-B3: Pati-Salam gauge labeling (ACTIVE — HYPERCHARGE COMPONENT GRADUATED 2026-05-05 EOD+3)

**Statement.** The specific assignment of which gauge factors correspond to which physical forces — SU(2)_L, SU(3)_color, U(1)_{B-L} — follows the Pati-Salam model (Pati-Salam 1974). This includes: which Spin(4) factor is SU(2)_L (not SU(2)_R), what the hypercharge assignment Y = +1/2 is for the Higgs doublet, and which states are leptons vs quarks.

**Motivation.** The framework's spinor decomposition Cl(6,0) → Spin(4) × Spin(2) is dimensionally compatible with the Pati-Salam factorization. The dimensions are forced; the physical labeling is the remaining step.

**HYPERCHARGE COMPONENT GRADUATION 2026-05-05 EOD+3** (`docs/theorems/theorem_g2d_chirality_doubled.md`, `proofs/foundations/sector_G2D_chirality_doubled_formalization.py`). The U(1)_Y hypercharge ADOPTION sub-component of ADOPTED-B3 has graduated to **theorem-grade** via the chirality-doubled edge qubit mechanism. Specifically:
- **SU(2)_R** is now derived from the RH-srs edge qubit via the same G2 theorem-grade argument applied to the mirror-image lattice (Cl(1,1) algebra preserved under f_1 → -f_1, f_2 unchanged; post-A3 → Cl(0,2) ≅ ℍ → SU(2); machine-precision verified).
- **Combined with theorem-grade SU(4) (Cl(6) Fock per `theorem_charge_before_color §9`) + SU(2)_L (G2)**, the framework now derives the full Pati-Salam gauge group SU(4) × SU(2)_L × SU(2)_R from {A1 + A2-T + A3-T + Cl(6) Fock + chirality-doubled edge qubit}. No adoptions.
- **Hypercharge formula** Y = T_3R + (1/2)(B-L) follows from standard PS breaking SU(2)_R × U(1)_{B-L} → U(1)_Y. Verified for all 9 SM fermion types (ν_L, e_L, ν_R, e_R, u_L, d_L, u_R, d_R, H).
- **Anchor for chirality-doubled reading**: 5 framework sources explicitly state "both chiralities above the waterline simultaneously" (`framework_axioms.md` lines 62, 75; `narrative_spine.md`; `orientation.md`; `theorem_A2_mdl §11`). Physical doubling (not just MDL-equivalence) is the framework's standing reading.

**Remaining ADOPTED-B3 content (still ACTIVE adoption):**
- **PS fermion sector assignment**: which Cl(6) Fock states map to which SM fermion species (lepton vs quark, generation labels). Partially derived via Furey 2018 §3 + theorem_charge_before_color, with residual labeling ambiguity per (Z/2)³ Angle D verdict 2026-04-30. NOT addressed by G2-D closure.
- **Generation labeling**: ~~requires Need-A2 closure~~ **STATUS UPDATE 2026-05-08:** Need-A2 generation-Z₃ existence is **CLOSED** (rediscovered today) via R3 (`predictions/R3_observer_c3_generation_derivation.md`, 2026-04-20) + M1.B (an internal working note §7.5, 2026-04-28, theorem-grade closed) + substrate generation-charge conservation (`docs/theorems/theorem_substrate_generation_charge_conservation.md`, 2026-04-29, theorem-grade unconditional). M1.B identifies R3's cyclic-shift Z₃ on C³_obs with the Galois Z₃ of the sub-factor inclusion M^α ⊂ M ⊂ M ⋊_α Z₃ ≅ M_3(ℂ) ⊗ M^α. The structural separation: substrate body-diagonal C₃ induces TWO distinct actions — inner on Cl(6) Fock (color-Z₃) and outer on operator algebra L(F_inv(E)) (generation-Z₃ via Galois tower). M1.B closure is theorem-grade in the operator-algebra sense; the basis-match identification (which mass label = which Z₃ character) uses Koide mass-spectrum data, theorem-grade under {A1+A2-T+A3-T+A5} (A5 closed P1+Y on 2026-04-19, see this register §"ADOPTED-P1 — CLOSED via A5" and §"ADOPTED-Y — CLOSED via A5"). M_gen non-degeneracy (R3's single external input "observed lepton-mass non-degeneracy") closed via generic A2-T measure-theoretic argument 2026-05-08 (an internal working note, probe `proofs/foundations/sector_M_gen_nondegeneracy_generic.py` PASS 5/5 incl. 10000-sample numerical sanity check at min-gap median 0.68). A failed structural-forcing attempt for non-degeneracy 2026-05-08 confirmed only the generic argument applies; both probes have correction headers.

**Derivation status post-G2-D closure + 2026-05-08 Need-A2 rediscovery + non-degeneracy closure:** ADOPTED-B3 PARTIAL GRADUATION. Hypercharge sub-component is theorem-grade (G2-D). Generation-Z₃ EXISTENCE is theorem-grade (M1.B chain). Generation-LABELING (mass-basis identification) is theorem-grade under {A1+A2-T+A3-T+A5} via M1.B + Koide chain. M_gen non-degeneracy theorem-grade-conditional on A2-T-prior absolute continuity. Sector labeling (lepton vs quark) remains adopted via (Z/2)³ Angle D residue, research-level multi-session per 2026-05-05 Angle-D audit.

**Original derivation status (preserved historical record):** "BLOCKED (irreducible without observed physics input)." This statement was correct pre-EOD+3 G2-D closure; supplanted by the chirality-doubled mechanism identified via the Route 4 attack chain (Route 4 → G2-D scoping → G2-D formalization).

**Closure path (remaining content).** Hypercharge: CLOSED (G2-D, 2026-05-05 EOD+3). Generation-Z₃ existence: CLOSED (M1.B + R3 + gen-charge-conservation, April 2026, rediscovered 2026-05-08). M_gen non-degeneracy: CLOSED via generic argument 2026-05-08. Generation-LABELING (mass-basis): theorem-grade under {A1+A2-T+A3-T+A5} (A5 closed P1+Y in April; the "modulo P1+Y" framing in older docs predates A5). Sector labeling (lepton vs quark): (Z/2)³ Angle D residue, research-level multi-session.

**Gauge-status of PS factors (2026-05-06).** Per an internal working note, the gauge-status of SU(2)_R, SU(4)_PS, U(1)_{B-L} is NOT derived from framework's dynamical gauge field machinery. The framework's actual gauge-field-as-dynamical-excitation construction (`proofs/gauge/srs_gauge_field_definition.py`) is for ONE SU(2) test group only; the promotion to full SU(4)_PS × SU(2)_L × SU(2)_R is explicitly noted "per ADOPTED-B3" — citation, not derivation. Two consistent readings exist:
- **(a) PS is gauge** — requires ADOPTED-PS-BREAKING (a separate adoption tracked in `structural_residue_register.md`) for descent to SM gauge group SU(3)_c × SU(2)_L × U(1)_Y.
- **(b) PS is organizing/accidental symmetry** (Candidate E) — only SM gauge group is fundamental; SU(2)_R, SU(4) "leptoquarks", U(1)_{B-L} are transformation symmetries on local Cl-modules, not dynamical gauge fields. No breaking gap.

Numerical predictions (sin²θ_W = 3/8, α_GUT, M_unif, λ_Higgs, Y for all 9 SM fermions, all inherited rows) are INDEPENDENT of this choice. Reading (b) is structurally preferred (no breaking mechanism needed; Candidate E audit walked 8 load-bearing items, all compatible). Audit doc: an internal working note. Walk probe: `proofs/foundations/sector_PS_organizing_symmetry_audit.py`.

---

## ADOPTED-DARK-MAP: Dark correction map tan²(arg h) = 5/3 (ACTIVE; SCOPE NARROWED 2026-04-28)

**Statement.** The dark sector's contribution to gauge-invariant observables follows the ratio tan²(arg h) = 5/3, where h = (√3 + i√5)/2 is the srs Ramanujan eigenvalue.

**Scope narrowing event (2026-04-28).** The Class-2 / Pathway-3 (mass²-class) dark-map identification has graduated to theorem-grade for the {y_τ, m_τ, m_e, m_μ, λ_Higgs, m_H, θ_23 PMNS} family via `../../theorems/theorem_dark_map_class2_closure.md`. y_τ and m_τ family closed in session 25 (`theorem_ytau_corollary.md`); λ_Higgs, m_H, θ_23 PMNS now closed as corollaries of y_τ §10.3 ratio + standard SM relations. **ADOPTED-DARK-MAP for these observables is RETIRED.**

**Active scope (post-2026-04-28):** ADOPTED-DARK-MAP remains active ONLY for:
- **β cosmic birefringence** (Pathway-4 observable; separate scoping `../../theorems/theorem_cosmic_birefringence.md`)
- **θ_13 PMNS** (Pathway-4-like with possible Tr σ_x = 0 selection; separate scoping needed)

These are independent gaps not addressed by the Class-2 closure.

**Original motivation (preserved historical record):** The srs Bloch amplitude at P is h = (√3 + i√5)/2, with |Im(h)|² / |Re(h)|² = 5/3. The dark sector is identified with the imaginary part of h (the "dark" amplitude component at P), and visible-sector observables pick up a 5/3 ratio when corrected for dark-sector interference. Numerically consistent across multiple independent predictions.

**Derivation status:** Class-2 / Pathway-3 GRADUATED 2026-04-28 for {Higgs quartic + sector mass-mixing}. β and θ_13 PMNS remain BLOCKED with separate scoping docs.

**Remaining closure paths:**
- For β: see `../../theorems/theorem_cosmic_birefringence.md` β.A1 (Pathway-4 unit-phasor) or β.A2 (Pathway-2 photon-Hodge-bundle).
- For θ_13 PMNS: needs explicit derivation that Tr σ_x = 0 selects coefficient c=1; possibly Pathway-1 (combinatorial) rather than Pathway-4.

---

## ADOPTED-A5b-Sub3: A5(b) Level 3 sub-class identification — **GRADUATION RETRACTED 2026-04-29**

**Status.** ACTIVE adoption (un-graduated). The 2026-04-28 graduation to theorem grade via the bridge functoriality lemma is RETRACTED, see banner notice at top of this file. The adoption itself remains in place — it is the working assumption that ships the 6 parameter ledger rows at STRICT-SOLID-conditional grade.

**Retraction trigger.** Three independent CAS probes refute the lemma's load-bearing structural argument (Z₃^m holonomy accumulation distinguishing ΔGen). See an internal working note §2 for the R1/R2/R3 refutations and §4 for the M1/M2 research routes that could close the structural gap properly.

**Historical record — failed graduation attempt 2026-04-28 (preserved):** Closed-then-retracted at theorem grade via `../../theorems/theorem_bridge_functoriality_lemma.md`. The bridge functoriality lemma claimed to derive the Level 3 sub-class classifier (specifically the generation-distinguishing factor for ΔGen=k transitions) by composing V_cb's k=1 base case + multiway functoriality (`../../theorems/theorem_multiway_branch_measure.md` §3+§4) + 16-cycle decomposition CAS uniqueness (`proofs/flavor/hashimoto_16cycle_decomposition.py`: 100% of L=16 NB cycles are 2-girth-glued by 2-edge seam). The CAS-verified parts (multiway functoriality, 16-cycle decomposition) remain valid; the Z₃-holonomy step that linked ΔGen to m mod 3 is what was refuted.

**Effect on parameter ledger (post-retraction).** 6 rows return to STRICT-SOLID-conditional-on-adoption (their pre-2026-04-28-PM grade): P14 (V_ub ships value 3.767e-3 from `vub_multicycle_sum.py` formula at −0.26σ, conditional on adoption), P15 (δ_CP_CKM identification), P32 (θ_12_PMNS), P33 (θ_13_PMNS sub-class part), P34 (δ_CP_PMNS sub-class part), P45 (J_CKM via V_ub).

**Original statement (preserved historical record):**



**Statement.** Within Level 3 / Case B of `../../theorems/theorem_A5b_level_prescription.md` (the Hashimoto walk-sum sub-classification), the framework's couplings split into three sub-classes determined by the structural data {endpoint causal-state pinning, transition Δ-quantum-numbers, Moore-bound saturation}:

(i) **Walk-rep sum** $u^L / (1 - u^L) = \alpha_1 / (1 - \alpha_1)$ — applies when endpoints are distinct pinned causal states with no transition-Δ structure beyond the walker's intrinsic phases. Example: V_cb (b ↔ c at L = g − n_fixed = 8).

(ii) **Moore-slot counting** $k^{*2} / (g \cdot N_{\text{atoms}})$ — applies when the relevant cycle structure saturates Moore's bound ($\lfloor g/k^{*2}\rfloor = 1$ on srs). Example: V_us.

(iii) **Generation-distinguishing factor** walk-rep × additional Z_d phase factor — applies when the transition crosses a generation boundary (ΔGen ≥ 1), with the Z_d label associated to the transition's generation index. Example: V_ub candidate $V_{us} \cdot (2/3)^g = 128/32805$.

The classifier (which sub-class applies) is the ADOPTED structural input. The structural data {endpoint, Δ, Moore-saturation} is given by the framework; the rule that maps these to the sub-class is the adoption.

**Motivation.** The framework's sub-class behaviour is empirically successful: V_cb at +0.07σ (walk-rep), V_us at −0.015σ (Moore-slot), V_ub candidate at +0.40σ (generation-distinguishing). The sub-class structure is therefore correct in its observable outputs — what's lacking is a *derivation* of the classifier itself from {A1} + structural data alone.

**Derivation status: BLOCKED at the V_Ram(P) layer.** Two CAS probes (Routes 1 and 1' of an internal working note) test the most natural candidate Z_d × C_3 = Z_6 generation labelings on V_Ram(P) and converge to the same negative result:

- Route 1 (orientation-reverse Z_2 × C_3): R does not commute with B(P) — the Hashimoto NB structure breaks A1's involutive Z_2. `proofs/foundations/a5b_route1_z6_directed_edge_probe.py`.
- Route 1' (B(P)-commuting Z_2 candidates: spectral-conjugation σ, directed-edge perm Z_2, anti-linear T): all three classes give D_3 = S_3 (color labels), not Z_6 (generation labels). `proofs/foundations/a5b_route1prime_z2_centralizer_probe.py`.

The natural Z_2 × C_3 algebra on V_Ram(P) is D_3, with irreps {trivial, sign, standard-2-dim} = {color singlet, color sign, color (ω, ω̄)}. **No generation Z_d eigenvalue identification exists at this layer in the routes tested.**

**Closure path.** Route 2 (sub-leading Hashimoto Bloch eigenvectors at non-symmetric k-points) and Route 3 (μ-branch hierarchy) of the scoping doc remain available as theorem-grade closure routes — both research-level multi-session, neither attempted yet. Until either closes, the sub-class classifier is taken as ADOPTED. Status will graduate to mathematically-complete or theorem-grade if Route 2 or Route 3 succeeds; otherwise this adoption may become permanent (analogous to ADOPTED-B3's Pati-Salam labeling).

**NEW STRUCTURAL CANDIDATE — Route 4 (SU(2)_L Higgs doublet partner mechanism), identified 2026-05-05 EOD+3** (an internal working note, `proofs/foundations/sector_need_D_species_differentiation_audit.py`). The standard SM Yukawa Lagrangian L_Y = Y_d Q̄_L H d_R + Y_u Q̄_L H̃ u_R uses CONJUGATE Higgs representations for u-type vs d-type — H ↔ H̃ = iσ_2 H^*. In framework's edge-qubit Higgs (theorem_g2_edge_qubit_su2), H is Cl(0,2) ≅ ℍ; under SU(2)_L action, H transforms as σ representation while H̃ transforms as σ̄ (conjugate). If C³_gen carries an SU(2)_L action lifted from the edge qubit, Y_u and Y_d would differ by conjugate-SU(2)_L rep on C³_gen, giving a structurally-derivable non-trivial CKM rotation. **Route 4 was claimed BOUNDED conditional on Need-A2 closure in the EOD+3 audit; CORRECTED 2026-05-05 EOD+3 (later)** via direct attack probe `proofs/foundations/sector_route4_SU2L_pseudoreal_attack.py`. **Route 4's naive framing is STRUCTURALLY INCORRECT**: SU(2) is pseudoreal, so H and H̃ are the SAME SU(2)_L representation (machine-precision verified via 5 random SU(2) matrices that U·H̃ = iσ_2·(U·H)^*). The actual H vs H̃ distinction is U(1)_Y HYPERCHARGE (Y_H = +1/2 vs Y_H̃ = -1/2), which is BLOCKED in the framework per `theorem_g2_edge_qubit_su2 §7` ("G2-D: hypercharge U(1)_Y — requires ADOPTED-B3 or independent derivation"). **Route 4 is therefore BLOCKED on G2-D, NOT just Need-A2** — the EOD+3 audit's conditional was incorrect. Tested naive "Y_u = Y_d^*" mechanism: gives a permutation matrix CKM (entries ∈ {0, 1}), not the small-mixing pattern observed (|V_us| ≈ 0.225, |V_cb| ≈ 0.041). REVISED Route 4 closure target: needs BOTH G2-D (hypercharge) AND Need-A2 (generation-Z_3) closures, multi-session research-level (~3-5 sessions G2-D + ~1-2 sessions bridge if Need-A2 also closes). More SPECIFIC than M2 multiway formalism but multi-session anyway. This audit also confirmed that Hamming-weight species filter (Λ^1 ≅ Λ^2 isotypically at k*=3 via Hodge duality) and V_{-1}-T_{B-L} angle (T_{B-L}_color = +1/3 for both u and d, doesn't distinguish within color sector) are individually NECESSARY but NOT SUFFICIENT for Need-D-3 closure.

**Downstream effect.** Resolves 6 parameter-ledger rows from BLOCKED to CONDITIONAL-on-ADOPTED-A5b-Sub3 (a status improvement under `../../parameters/parameter_linter.md`, since adoptions are gate-passing while BLOCKED is not):

- **P14** (V_ub) — candidate formula $V_{us} \cdot (2/3)^g = 128/32805 = 3.90 \times 10^{-3}$ at +0.40σ from PDG ships as STRICT-SOLID conditional on the adoption.
- **P15** (δ_CP_CKM identification) — geometric value $\arccos(1/3) = 70.53°$ STRICT-SOLID; identification with physical CKM phase CONDITIONAL on the adoption.
- **P32** (θ_12_PMNS) — algebraic content $\cos\theta_{12} = \cos\theta_{TBM}/\cos\theta_C$ theorem-grade; bridge CONDITIONAL on the adoption.
- **P33** (θ_13_PMNS) — sub-class part CONDITIONAL on the adoption; remains additionally blocked on the PS embedding step (Priority 4.2, separate gap).
- **P34** (δ_CP_PMNS) — CONDITIONAL on the adoption. **REVIVED 2026-05-05** via the V₋₁-T_{B-L} identity (δ_CP_PMNS = 180°, THEOREM-GRADE-STRUCTURAL); the old arg(h) Path B'' route is no longer the live derivation.
- **P45** (J_CKM) — inherits P14; CONDITIONAL on the adoption once V_ub is shipped.

**Cross-references.**
- `../../theorems/theorem_A5b_level_prescription.md` — the closed Case A vs Case B level prescription (Level 2 vs Level 3 split).
- `proofs/foundations/a5b_route1_z6_directed_edge_probe.py` and `a5b_route1prime_z2_centralizer_probe.py` — CAS evidence.
- `../../parameters/parameter_uniqueness_ledger.md` Rows P14, P15, P32, P33, P34, P45 — affected rows.

---

## ADOPTED-PS-SCALE: Pati-Salam neutrino bare scale — CLOSED (2026-05-04)

**Closure event.** The PS seesaw formulation that required ADOPTED-PS-SCALE is SUPERSEDED by a global spectral-gap derivation of m_ν₃: m_ν₃ = (k* × N_atoms) × M_Pl × N_hub^(-1/2). The new chain uses only framework primitives (k*, N_atoms, M_Pl, N_hub) — no M_GUT, no m_t(GUT), no MSSM RG, no Pati-Salam bare scale.

**Closure docs:** `predictions/m_nu3.py`, `predictions/m_nu3_derivation.md`, an internal working note, `proofs/flavor/srs_M_R_step{1_structural,2_derivation,3_closure}.py`.

**Equivalence.** The seesaw form m_ν₃ = v²/M_R is preserved with M_R = δ⁴ × M_Pl / (2 × k* × N_atoms). The δ⁴ in v² and M_R cancel exactly, leaving the global form independent of the Koide phase δ.

**Status.** ADOPTED-PS-SCALE retired. Active adoption count: 4 → 3 (now ADOPTED-B3, ADOPTED-DARK-MAP, ADOPTED-A5b-Sub3).

**Numerical match (post-closure).** m_ν₃ = 50.57 meV vs NuFIT 6.0 50.13 ± 0.20 meV → +0.87% deviation = +2.18σ_PDG (FAIL Clause 8 against σ_PDG alone). The residual is consistent with the N_hub anchor variation between {G_F, m_τ, R_∞} calibrations.

**Historical statement (pre-closure, preserved for record).** ~~The bare neutrino mass scale in the seesaw formula is set by the Pati-Salam GUT scale (~10¹⁵ GeV)... Closure path: A derivation of the Planck-to-GUT-to-EW hierarchy from the srs lattice geometry would close this. Currently no approach identified.~~ Closure was achieved by REFRAMING away from the GUT-anchored bare scale, not by deriving M_GUT.

---

## ADOPTED-K_P-TIEBREAK: cross-substrate k_P selection tiebreaker (ACTIVE; surfaced 2026-04-30 EOD final)

**Statement.** When the framework's k_P selection rule "C_3-stable AND Hashimoto eigenvalue with multiplicity exactly 2 at the Ramanujan saddle" yields multiple candidate k-points for a given substrate, an additional tiebreaker is needed to pick a unique k_P. Currently un-specified.

**Motivation.** The framework's selection rule (per `predictions/B_P_doubly_degenerate_h.py` docstring) was inherited from srs's bcc geometry, where it happens to yield P uniquely (Γ excluded by mult 3, H/N not C_3-stable). For audit v2 cross-substrate work (e.g., qtz at k=4 with hexagonal P6_222 geometry), the rule produces MULTIPLE candidates in 13 of 13 tested bond list families — Γ always has a mult-2 saddle, K/A/H also have mult-2 saddles for many bond lists.

**Discovery.** Surfaced via audit v2 follow-up #3 (selection-rule audit, an internal working note). Cross-substrate audit revealed ambiguity not visible in the srs-only formulation.

**Derivation status: BLOCKED (no principled tiebreaker derived from A1+A2+...).** Possible candidate tiebreakers (none derived):
- Highest-symmetry k-point preference: would pick Γ.
- BZ-boundary preference: would pick K/H (analog of srs's body-diagonal P).
- Maximum-Im(h) preference: would pick whichever has largest Im(h) at the saddle.
- Maximum-distance-from-Γ preference: would pick BZ corner.

**Impact on audit v2 closures.** Vacuous for srs (rule yields P uniquely). Active for any substrate with multiple mult-2 candidates (all 13 tested qtz bond lists). For Row 4 v2 closure, this conditional is **NOT load-bearing** — the data-conditional MDL crush (~2×10⁸ bits) is independent of which k_P-analog gets selected, and crushes qtz observationally regardless. ADOPTED-K_P-TIEBREAK is named for transparency in cross-substrate audit work; resolving it would tighten M6-based closures but is not required for the predicted observable values.

**Closure path.** Either (a) a deeper principle from A1+A2+A3+A4 that uniquely picks one k-point for any substrate, or (b) explicit framework adoption of a specific tiebreaker rule. Approach (a) preferred; (b) is the fallback. Estimated scope: 1-3 sessions of foundational research.

**Cross-references.**
- `proofs/foundations/qtz_selection_rule_audit.py` (computational analysis).
- `predictions/B_P_doubly_degenerate_h.py` (framework's rule statement).

---

## ADOPTED-COSMOLOGICAL-IC-AMPLITUDE: cosmological preferred axis carries frozen anisotropy amplitude ε_toggle — GRADUATED 2026-05-07 via observer-MDL persistence theorem

**Status (post-2026-05-07):** GRADUATED to derived theorem via `../../theorems/theorem_observer_persistence_closure_IC_amplitude.md`. The named adoption is dissolved. The entry below is preserved as the historical record of the adoption's introduction and the structural commitment it codified before closure.

**Statement.** The substrate's cosmological preferred axis ẑ — the same axis that sources both the CMB hemispherical asymmetry (A_hemis) and the cascade D2-extended observer-substrate rate-gap (H_0/t_0/Λ_CC observer-side) — carries a frozen anisotropy amplitude α = ε_toggle = 1/5 at all cosmological epochs (N = 1 IC through N_hub ≈ 10⁶¹ observer epoch).

The amplitude appears in:
- Substrate per-direction event rate Π_ab = (1/k*)[δ_ab + ε_toggle ẑ_a ẑ_b] (cascade D2-extended, `theorem_cascade_D2_extended_observer_rate.md`).
- Observer-substrate rate-gap (H_obs/H_sub = 1 + ε_toggle/k = 16/15) at z = 0.
- CMB hemispherical asymmetry A_hemis = ε_toggle × ⟨(ê·ẑ)²⟩ = 1/15 at z ≈ 1100.

**Discovery.** Surfaced 2026-05-06 via cascade Step 5 amplitude compression-integral audits (Session 1 scoping + Session 2 audits, an internal working note).

**Decomposed structurally into two claims:**
- **Claim A (IC amplitude):** α_IC = ε_toggle at the N=1 boundary. Derived cleanly via 5-step Bridge 1 chain (cascade theorem N(0)=1 + Bayesian conjugate update + S_fresh/S_disconfirm predictives + linear-normalization composition); all steps Type 1–4 per parameter_linter.md. Probe: `proofs/cosmology/cascade_step5_claim_A_n_eq_1_BC.py` (Fraction-exact + sympy + cross-import checks).
- **Claim B (persistence):** α persists from N = 1 to N = N_hub. Five rescue routes audited; ALL CLOSED structurally:
  - B-route 1 (light-cone causality): preserves axis, not amplitude. Insufficient.
  - B-route 2 (Bloch zero-mode): NEGATIVE — k=0 stationary commutes with S_n direction-permutation symmetry; unique stationary is direction-uniform by Perron-Frobenius. `proofs/cosmology/cascade_step5_claim_B_bloch_zero_mode_audit.py` (verified machine-precision on 4-direction toy).
  - B-route 3 (per-event regeneration): circular without B-route 1 or 2.
  - B-route 4 (A2-T MDL waterline / observer-centric): reintroduces the "by analogy" move that the linter already rejected.
  - B-route 5 (full renewal Markov compression integral): dynamics-equivalent to Model 1 (`cascade_step5_claim_B_persistence_audit.py`); 400 replicas to N=5000 events give ⟨α(t→late)⟩ = −0.002 ± 0.17 vs target 0.20. IC anisotropy decays exponentially to noise level by N ≈ 100.

**Empirical anchor.** Joint 4-observable constraint (`cascade_step5_amplitude_via_A_dilution.py`):
| Observable | α (1σ range) | central | σ from ε_toggle |
|---|---|---|---|
| A_dilution (Planck + WMAP CMB hemis) | [0.135, 0.255] | 0.195 | +0.08σ |
| Cascade rate-gap (SH0ES H_0) | [0.168, 0.259] | 0.213 | +0.29σ |
| Joint (Gaussian inverse-variance) | 0.207 ± 0.036 | 0.207 | +0.18σ |

Alternative amplitudes excluded by joint: α = 2 ε_toggle = 0.400 at 5.32σ; α = ε_toggle/2 = 0.100 at 2.93σ. Empirical match pins α to ε_toggle ± few percent across 4 independent observables sharing the same structural form.

**Derivation status: BLOCKED at substrate stationary level under direction-uniform renewal Markov dynamics.** The framework's current axioms (A1, A2-T, A3-T, A4 partial via Jordan-Wigner, A5(a)+(b)) plus Stage 2c (energy functional / arrow of time) plus Stage 3 (Lorentz/causal sector) do not derive Claim B persistence. Five routes closed; all require structural input that BREAKS S_n direction-permutation symmetry of the local renewal Markov, and no current axiom supplies this.

**Closure paths (none derived; all multi-session research-level).**
- (a) **Direction-anisotropic local sampling mechanism.** Identify a structural reason events at substrate stationary sample directions with rate ∝ (1 + γ(ê·ẑ)²). The simplest candidate: cosmological IC's ẑ persistence by some yet-unidentified conservation law. Estimated 4–8 sessions; no candidate mechanism currently in framework.
- (b) **NESS with degenerate stationary manifold.** Show that Stage 2c arrow-of-time produces a non-equilibrium steady state with persistent fresh-creation > disconfirm flux maintaining anisotropy along ẑ. Requires connecting temporal asymmetry (S_disconfirm > S_fresh) to spatial preferred axis via a mechanism not currently axiomatized. Estimated 3–5 sessions; no candidate construction.
- (c) **Substrate as non-Markovian.** If substrate dynamics has memory beyond the per-vertex Beta state, persistence may follow. Requires departure from the framework's current Markov reading of A1+A2-T. No proposed reformulation.

**Downstream effect on parameter ledger.** Resolves an internal inconsistency: Row P27 (A_hemis) was previously graded UNIQUE-THEOREM-GRADE, while Rows P19/P20/P24 (cascade D2-extended observer-side) were graded UNIQUE-THEOREM-GRADE-CONDITIONAL on Step 5 amplitude — but both rest on the SAME structural identification α = ε_toggle. After Path 1:

- **P19** (H_0 observer-side = 72.72 km/s/Mpc): UNIQUE-THEOREM-GRADE-CONDITIONAL on ADOPTED-COSMOLOGICAL-IC-AMPLITUDE (cascade Step 5 conditional now NAMED).
- **P20** (t_0 observer-side = 13.45 Gyr): same.
- **P24** (Λ_CC, rate-gap component (16/15)²): same; matter/dark factor-of-2 residue remains independent.
- **P27** (A_hemis = 1/15): graduates *to* UNIQUE-THEOREM-GRADE-CONDITIONAL on ADOPTED-COSMOLOGICAL-IC-AMPLITUDE (was UNIQUE-THEOREM-GRADE; the previously-implicit gap is now named).

All four rows share ONE named conditional. Predictions and [historical σ_combined] match values are unchanged (4-observable joint at 1.06σ post-correction).

**Empirical evidence accumulates non-trivially.** Each new observable matching α = ε_toggle at <1σ is independent supporting evidence that this adoption is structurally correct (i.e., that whatever future framework axiom is added to derive persistence, it will give this amplitude). Currently 4 observables; future work that probes additional ẑ-anisotropy observables (CMB low-ℓ, large-scale structure dipole, gravitational wave anisotropy) would tighten or refute the joint constraint.

**Cross-references.**
- `proofs/cosmology/cascade_step5_claim_A_n_eq_1_BC.py` — Claim A closure.
- `proofs/cosmology/cascade_step5_claim_B_persistence_audit.py` — Claim B Model 1 negative.
- `proofs/cosmology/cascade_step5_claim_B_bloch_zero_mode_audit.py` — Claim B B-route 2 negative.
- `proofs/cosmology/cascade_step5_amplitude_via_A_dilution.py` — joint 4-observable empirical constraint.
- `docs/theorems/theorem_cascade_D2_extended_observer_rate.md` — theorem doc consuming this adoption.
- `docs/theorems/theorem_class_D_statistical.md` Derivation 3 — A_hemis composition rule (now footnoted with persistence gap).
- `proofs/cosmology/A_dilution_derivation.py` — A_hemis derivation (now footnoted).
- `docs/parameters/parameter_uniqueness_ledger.md` Rows P19, P20, P24, P27.

---

## ADOPTED-MSSM-Sb: MSSM matter content as RG-running scheme (ACTIVE; surfaced 2026-05-11)

**Statement (revised 2026-05-14 PM, eliminating M_SUSY).** The framework operates in **single-regime running**: one-loop β-coefficients [b_1, b_2, b_3] = [33/5, 1, −3] across the entire range from M_unif down to M_Z, with **no SUSY-breaking threshold scale M_SUSY**. The "MSSM" label is retained as a named convention for these specific β-coefficient values (derived as math-complete in `docs/theorems/theorem_beta_coefficients_derived.md`); it does NOT entail importing M_SUSY as a framework parameter. The two-regime picture (MSSM above M_SUSY ∈ [M_Z, 10 TeV] + SM below) is conventional MSSM phenomenology, not a framework necessity. The framework's native answer is the single-regime 1-loop algebra with deviations from PDG reported as absolute percentages (~6% on b_2; see §2.5 of theorem doc) without invoking M_SUSY threshold matching.

**What remains adopted under ADOPTED-MSSM-Sb:** the literal-particle-content question of whether the derived β-coefficients are realized by physical SUSY partners (sfermions, gauginos, Higgsinos, gravitino). No substrate mechanism for these particles is identified; this remains the honest residue.

**Motivation (structural necessity, not preference).** The framework's substrate-native inputs at M_unif are theorem-grade:
- α_GUT = 1/24 (Cl(6) Fock label count; Class C taxonomic)
- sin²θ_W(M_unif) = 3/8 (GQW trace identity on PS multiplet)
- M_unif ≈ 1.985×10¹⁶ GeV (cascade theorem)

Per `proofs/foundations/mssm_matter_content_required.py`: running α_i from these inputs to M_Z under the framework's *derived* matter content (3 generations + 2 Higgs doublets = 2HDM, no superpartners) gives α_s **negative** at M_Z — catastrophic, not just off. SM running (1 Higgs) gives the same catastrophic result. Only MSSM running (with superpartners) reaches PDG values within ~1–3%. The framework's structural inputs therefore *structurally require* MSSM β-coefficients for PDG match; this is not a preference among comparable alternatives but a near-uniqueness constraint at one loop.

**Discovery.** Surfaced 2026-05-10 via an audit of the per-prediction theoretical-uncertainty band (since retracted 2026-05-13). Initial framing: the uniform 1% theoretical-uncertainty band was a magic literal absorbing each observable's specific deviation (M2b anti-pattern). Phase A removed the literal. Deeper finding: the literal was a *symptom* of an undeclared adoption — MSSM matter content adopted in the RG running without explicit register. Sharpened: Layer 5 SUSY closure has no identified theorem-grade route through any scoped path (A/B/C/D/E). Closure attempt 2026-05-11 (SU(2)_L Wilson-loop probe) closed the last bounded route. Clause 8 is evaluated against σ_PDG only.

**Failed closure routes (preserved record).**
- **Path A — anomaly cancellation.** `proofs/foundations/susy_path_a_anomaly_cancellation.py` verifies all four standard chiral anomalies (U(1)_Y³, grav·Y, SU(2)_L²·Y, SU(3)_c²·Y) vanish per generation in framework's PS-extended matter content WITHOUT superpartners. C³_gen is a permutation symmetry, not gauged. **Anomaly cancellation does not force SUSY in this framework.** INOPERATIVE.
- **Path B — multiway branch-measure / dark-sector dynamics.** Speculative; no concrete handle identified. Status: SPECULATIVE.
- **Path C — deep multiway formalism.** Speculative; not formalized at substrate level. Status: SPECULATIVE.
- **Path D — numerical necessity (MSSM matter required for PDG match).** Shown via `mssm_matter_content_required.py`: SM and 2HDM running give catastrophic mismatch; MSSM matches. **Necessity demonstrated at one loop; structural uniqueness (that MSSM is the ONLY matter content giving correct β-coefficients) not proven.** PARTIAL.
- **Path E — Witten SUSY substrate uplift.** `proofs/foundations/susy_path_e_witten_substrate.py` (commit 29988ec) shows Witten SUSY-QM is automatic on the substrate (D_sub² Lichnerowicz; γ_7 chirality grading). BUT γ_7 grades the substrate's Cl(6) Fock by CHIRALITY (both ±-eigenspaces are FERMIONS), not by spin-statistics. MSSM SUSY requires bosonic partners; the framework's 8-dim Cl(6) Fock per vertex does not naturally double matter content into bosonic+fermionic supermultiplets. χ̃ bipartite SUSY-pair on srs-z exists structurally but observables are χ̃-invariant (B1, D5 NEGATIVE). **MSSM-style SUSY uplift BLOCKED.** BLOCKED. **— 2026-05-12 post-R-9 recheck: still BLOCKED, sharpened.** With R-9 closed (substrate = srs; its bipartite *double cover* = srs-z; χ̃ = that double cover's natural "edge from +-vertex vs −-vertex" Z₂, anticommuting with the Hashimoto operator for pure bipartite reasons), the full simulator+match run on srs-z (`proofs/foundations/r9_srsz_simulator_run.py`) shows χ̃ is observably inert in the *strongest* sense — every intensive output and every non-cell-size-dependent prediction is *bit-identical* between srs and srs-z; the only 14 differing predictions are all cell-extensive (they differ only because the double cover's cell is doubled, not because χ̃ touches the matter content). Structural reason the blocker is robust: the matter sector is a *Clifford module* (Cl(6) Fock via the local Jordan-Wigner CAR theorem) — all its states are spinors / fermionic; doubling it (8-dim → 16-dim ≈ Cl(7) module on srs-z) yields *more fermions* (fermion + mirror/Dirac partner), never fermion + boson; MSSM superpartners need boson/fermion splitting per multiplet. The only conceivable bridge to the bosonic (gauge) sector is the V_Ram ≅ Cl(6) Fock identification (P4§6 #3) — independently research-level open, and even if closed it is a gauge-sub-space-↔-matter-space iso, not the boson↔fermion partner map SUSY needs (the walker on the directed-edge space is not a separate particle from the gauge boson). Do not re-open Path E via χ̃ (confirmed inert) or via doubling the Cl(6) Fock (confirmed all-fermionic); ADOPTED-MSSM-Sb is the settled endpoint.
- **Path F — per-sector substrate β-functions.** SU(2)_L Wilson-loop probe (`proofs/foundations/substrate_rg_beta_function_su2.py`, 2026-05-11) tests whether F7's α_1 substrate-internal flow extends to SU(2)_L via natural closed-walk structures on the Cl(0,2) edge qubit. **All natural candidates (A, B, C) give clean negative**: Candidate A (fixed bivector) is period-4 oscillation; Candidate B (edge-orientation θ rotations) gives no geometric series for any natural θ; Candidate C (Haar-averaged SU(2) survival) is constant plateau 1/2 by Haar invariance. Structural reason: F7's geometric-series mechanism is a U(1) (scalar amplitude) phenomenon; SU(2) holonomies are unitary with |U|=1 per step. Candidate D (character expansion / heat-kernel decay) is structurally richer but requires substrate-derived heat-kernel time t — research-level ≥3 sessions, deferred. CLOSED for bounded routes; OPEN-RESEARCH for Candidate D only.
- **Path G — accept 2HDM running, recompute cluster.** Mentioned in the original 2026-05-10 audit as a "reconsidered substrate-PDG bridge." Not pursued — the catastrophic mismatch under SM/2HDM (α_s negative) makes this an unphysical option for cluster predictions.

**No identified theorem-grade route remains.** Closure of MSSM matter content from substrate primitives requires foundational structural insight beyond the framework's current apparatus. The framework already adopts N_hub as its one dimensional physical input (its value pinned via the measured G_F — an analogous calibration role; G_F itself is downstream). Adopting MSSM matter content as the RG-running scheme is the linter-consistent endpoint per `parameter_linter.md` hard quality gate (Clauses 1–8): the cluster's numerical match remains correct given MSSM, and the conditional is now *named* honestly rather than hidden behind an undeclared adoption.

**Derivation status: BLOCKED (no identified theorem-grade closure path).** All scoped paths A/B/C/D/E/F closed or speculative. Closure would require either:
- (i) A new structural argument (Path B/C formalization, or new path not yet conceived) deriving MSSM as the unique consistent matter content from substrate. Estimated effort: research-level, multi-sprint.
- (ii) Candidate D heat-kernel mechanism: substrate-derived "heat-kernel time t" producing character-expansion decay of SU(2) and SU(3) holonomies. Estimated effort: ≥3 sessions to scope; not single-probe closable.

**Downstream effect on parameter ledger.** Resolves the cluster's previous DOMINANT-CONDITIONAL on "Layer 5 SUSY closure (no identified route)" to UNIQUE-THEOREM-GRADE-CONDITIONAL on (ADOPTED-MSSM-Sb, G_F) jointly. The numerical match is unchanged; the conditional is now named honestly.

- **Row P63 (α_EM(M_Z)):** UNIQUE-THEOREM-GRADE-CONDITIONAL on (ADOPTED-MSSM-Sb, G_F) jointly. Was DOMINANT-CONDITIONAL on Layer 5 SUSY closure (no route).
- **Row P64 (M_Z):** same.
- **Rows P65-P70** (sin²θ_W(M_Z), g_1, g_2, g_3, α_s(M_Z), R∞): same.
- **Row P71 (m_W):** same (inherits cluster).

**Tolerance-band framing (revised 2026-05-14 PM; further retracted 2026-05-13).** The previous "MSSM threshold-correction envelope (M_SUSY ∈ [M_Z, 10 TeV])" framing is RETRACTED. M_SUSY is not a framework parameter; varying it to construct an envelope is fitting a free parameter to data. The framework's actual precision is reported as absolute % deviation from PDG at the framework's single-regime 1-loop prediction; Clause 8 is evaluated against σ_PDG only.

**2026-05-11 (later same day) — M5 candidate enumeration confirms adoption.** Per an internal working note and `proofs/foundations/m5_candidate_enumeration_2026-05-11.py`: applied the validated W(M, N) waterline formula to 12 standalone substrate candidates outside the visible alphabet (subdominant Lie algebras F_4/E_6/E_7/E_8, alternative vertex algebras 𝕆/Cl(8)/Cl(10), alternative edge algebras Cl(0,4)/Cl(0,6), and composites of dominant content). All 12 retained at N_hub; only 1 candidate (𝕆 octonion triality) has even a maybe-MSSM-partner-shaped structure, and its substrate access is blocked at M3 per `M_mechanisms_synthesis_2026-05-07.md`.

**2026-05-11 (PM, later same day) — REFRAMING of the M5 closure leg following exhaustive graph enumeration.** Per an internal working note: the M5 enumeration was correctly criticized as a TEXTBOOK Lie-algebra list, not the framework's actual substrate content. Exhaustive enumeration of operator outputs (adjacency, Hashimoto, Laplacian spectra at all 4 high-symmetry k-points; cycle counts; automorphism group; Clifford algebras) finds **3 previously-unused Ramanujan saddles** at the substrate's other k-points: h_Γ = (−1+i√7)/2, h_N = (√5+i√3)/2, h_H = (1+i√7)/2. All saturate |h|²=2=k*−1; each has distinct tan²(arg) (5/3, 3/5, 7, 7). C_3 isotypic V_Ram=(4,2,2) supported at 3 of 4 k-points (Γ, P, H), distinct from N's (2,0,0). These are concrete substrate-derivable objects the framework has not yet mapped to observables. **Revised verdict on ADOPTED-MSSM-Sb's third leg:** the M5 candidate enumeration over textbook Lie-algebra-list is REPLACED by acknowledgment that the substrate has at least 3 unused Ramanujan saddles plus their V_Ram structures, none of which has been ruled out as potential MSSM-partner content (the testing was not done). ADOPTED-MSSM-Sb remains the working position by default but the structural argument is now: "no current substrate-derived closure path identified" rather than "structurally forced by exhaustive search". The triple-closure claim is downgraded to **two-pronged-closure** (per-sector probe + Layer-5 path enumeration) with one substantive open candidate cluster (the 3 unused saddles + their associated V_Ram structures).

**2026-05-27 — "3 unused saddles" residue FULLY RESOLVED.** All three saddles flagged 2026-05-11 PM as "untested MSSM-partner candidates" now have closure status. None contributes substrate-derivable closure of the literal-particle residue.

| Saddle | k-point | Status | Reference |
|---|---|---|---|
| h_Γ = (−1+i√7)/2 | Γ | Assigned to NEUTRINO sector (chir-7) | `docs/theorems/theorem_neutrino_chir7_concentration_2026-05-21.md` (2026-05-21) |
| h_P = (√3+i√5)/2 | P | Framework canonical (used) | `predictions/h_walker_eigenvalue.py` |
| h_N = (√5+i√3)/2 | N | **NEGATIVE-inert** (frozen residue) | an internal working note (2026-05-27) |
| h_H = (1+i√7)/2 | H | Assigned to NEUTRINO sector (chir-7) | `docs/theorems/theorem_neutrino_chir7_concentration_2026-05-21.md` (2026-05-21) |

h_N closure mechanism (A4 Session 1 verdict, NEGATIVE-inert): the Path-E all-fermionic Cl(6) Fock blocker fires at h_N (same per-vertex Hilbert space as h_P, the Bloch operator does not act on per-vertex Fock), the Redundancy gate passes (V_Ram(N)=(2,0,0) vs V_Ram(P)=(4,2,2) are different multisets so P and N are not in the same BZ orbit, the `arg(h_P)+arg(h_N)=π/2` identity is a complex-number coincidence rather than a substrate-induced map), the auxiliary search finds h_N has not silently underwritten any framework constant (every h_N derived quantity is an R/I-swap shadow of an h_P-derived quantity), no single-rule gauge-sector β-derivation produces (33/5,1,−3) without fitting, and h_N lives INSIDE the framework's Cl(6) Fock so cannot match the multi-axial dark-sector theorem's OUTSIDE-Cl(6)-Fock dark substrate. h_N is a genuine structurally-independent substrate object with no framework projection rule that reads it — recorded as R-18 in `structural_residue_register.md`.

**Net consequence for ADOPTED-MSSM-Sb (2026-05-27 EOD, post-A1 + post-A4):** the 2026-05-11 PM "two-pronged-closure plus 3 unused saddles open" framing now reverts to **two-pronged-closure with all unused-saddle candidates closed AND the heat-kernel/Candidate-D route closed**. Substrate-side derivation of the literal-particle residue is now exhausted across:
- Paths A/B/C/D/E/F (closed 2026-05-11 through 2026-05-12)
- The saddle route A4 (closed 2026-05-27 NEGATIVE-inert via an internal working note)
- The heat-kernel route A1 / Candidate D (closed 2026-05-27 POSITIVE-substrate-derives-2HDM-no-modification via an internal working note)

**A1 closure mechanism (POSITIVE-substrate-derives-2HDM-no-modification, 2026-05-27).** The 2026-05-11 Candidate D structural blocker ("the substrate has no obvious analog of heat-kernel time t") is filled by the 2026-05-27 cosmic-history arc's thermal mechanism `T(N) = T_P · N^(−1/2)` (validated 0-8% across 14 beats post Phase-III universality). With substrate-derived t now available, A1 Session 1 computed one-loop b_2 via standard character expansion + framework matter content (3 PS generations × 4 SU(2)_L Weyl doublets + 2 Higgs doublets + NO superpartners). Result: b_2 = −3 (literature 2HDM value), Δb_2(substrate, 1-loop) = 0. The substrate's thermal apparatus IS standard-QFT-compatible at one loop, but β-coefficient at 1-loop is propagator-independent (set by matter content's representation theory alone) — so the substrate's thermal apparatus modifies the heat-kernel-time PARAMETRIZATION, not the β-coefficient itself. Anchor gate FAILS: α₂⁻¹(M_unif) = 45.4 vs target 24 (+89%). MSSM b_2 = +1 cross-check gives α₂⁻¹(M_unif) = 24.35 (+1.44%) — confirming the framework's existing algebraic-inversion β-derivation chain works precisely because the MSSM particle content is what unifies. The literal-particle residue is now precisely characterized: **Δb_2 = +4 at SU(2)_L** (analogous gaps expected at U(1)_Y and SU(3)_c, not yet computed). Recorded as R-19 in `structural_residue_register.md`. This INDEPENDENTLY re-derives `proofs/foundations/mssm_matter_content_required.py`'s structural fact (2HDM running catastrophic) via the substrate's thermal apparatus itself.

**A3 closure (CONFIRMATORY-NEGATIVE, 2026-05-27 same-day triage).** V_Ram ≅ Cl(6) Fock iso re-examination per an internal working note. Four predictions from 2026-05-12 Path-E recheck (P1: iso pairs gauge-sub-space ↔ matter-space NOT boson↔fermion; P2: walker on directed-edge space ≠ separate particle from gauge boson; P3: Cl(6) Fock stays all-fermionic; P4: ADOPTED-MSSM-Sb stays settled) all CONFIRMED. The iso theorem itself disclaims MSSM β derivation in its own §"What this theorem does NOT do" section: *"Does NOT deliver MSSM β coefficients (Layer 5 SUSY remains external — ADOPTED-MSSM-Sb). The iso pairs across matter/gauge boundary, not within multiplets like MSSM."* Four iso-specific features audited (D_i mixing within spinor module; Q_i Q_j = −Q_k as Cl(4)-volume algebra not SUSY algebra; Q_i ↔ generation intra-fermion-sector; T3 CLOSED-AS-NEGATIVE rules out continuous SU(4)_PS extension) — all clean negative for SUSY-route opening.

**Branch A status post-2026-05-27 EOD (three same-day closures):**
- **A1** (heat-kernel / Candidate D): CLOSED POSITIVE-substrate-derives-2HDM-no-modification — substrate's thermal apparatus produces 2HDM β-values; literal-particle gap precisely characterized as Δb_2 = +4 at SU(2)_L (R-19 in residue register)
- **A3** (V_Ram-iso re-examination): CLOSED CONFIRMATORY-NEGATIVE — iso closure confirms 2026-05-12 Path-E prediction; no new SUSY route
- **A4** (unused Ramanujan saddles): CLOSED NEGATIVE-inert — h_N structurally independent but observationally inert; h_Γ/h_H went to neutrino sector via chir-7 theorem (R-18 in residue register)
- **A2** (M_unif threshold matching): UNDEVELOPED — no scoping work performed; no concrete starting point identified

**Branch A is comprehensively exhausted at the level of bounded research routes.** Three independent same-day closures (A1, A3, A4) form a converging body of evidence that the framework's substrate-derivation surface cannot graduate ADOPTED-MSSM-Sb's literal-particle residue. A2 remains undeveloped but with no concrete attack path identified.

**Net programmatic position:** Branch C (retire "Framework commits to SUSY" language in `honest_assessment.md`; reclassify "No SUSY below 10 TeV" from falsifier to consistency-observation) is now FULLY VALIDATED. The framework's honest position is:
- α_GUT⁻¹ = 24, sin²θ_W = 3/8, M_unif (theorem-grade upstream)
- β-coefficients (33/5, 1, −3) DERIVED via algebraic inversion from PDG α(M_Z) (mathematically complete, theorem-grade)
- Substrate matter content: 3 PS generations + 2 Higgs doublets (theorem-grade), no superpartners
- Substrate-derived β-coefficients at one loop: 2HDM values (NOT MSSM); Δb_2 = +4 gap is the literal-particle residue, precisely characterized but NOT closable via substrate-side bounded routes
- "MSSM" is a NAMED CONVENTION for the β-values that match observation, with the literal-particle adoption now precisely bounded rather than open-and-uncharacterized
- "No SUSY below 10 TeV" is CONSISTENT WITH FRAMEWORK (substrate-derived matter content is structurally non-MSSM), NOT a falsifier

**2026-05-27 EOD+2 — A2 (Mechanism C threshold matching) CLOSED NEGATIVE-multi-regime-doesn't-help.** Per an internal working note: multi-regime composition (PS-running M_unif → M_R + tree-level Slansky matching at M_R + SM 2HDM running M_R → M_Z) produces effective single-regime β-coefficients (+3.21, −3.00, −7.31), with b_2_effective IDENTICAL to substrate-2HDM single-regime b_2 = −3. **0% of Δb_2 = +4 gap closed by multi-regime structure.** Structurally inevitable: PS-regime b_PS_2L = SM-regime b_SM_2L_2HDM = −3 (the (4,2,1) → (3,2)+(1,2) breaking at M_R preserves all SU(2)_L charges identically; SU(4)_PS factors cleanly out of SU(2)_L's β-formula). Anchor gate: 0/3 channels pass against PDG; multi-regime is catastrophically off — α_3⁻¹(M_Z) = −14.4 (negative) vs PDG +8.5. This makes Branch A's bounded research surface COMPREHENSIVELY EXHAUSTED across all four routes (A1 + A2 + A3 + A4); literal-particle interpretation remains a structural residue (R-19) with no substrate-derivation closure path identified by any framework-internal mechanism currently known. Branch C reframing (executed 2026-05-27 commit `af1cf79`) is empirically robust across all 4 Branch-A routes.

**2026-05-27 EOD+1 — SUSY-LOAD-BEARING AUDIT COMPLETE + BRANCH C EXECUTED.** Per an internal working note: 560 SUSY-pattern occurrences across 375 files audited; no framework prediction or theorem-grade derivation depends on literal SUSY particles. LB items localized to (a) public-face self-description language pre-dating the 2026-05-14 reframing, (b) orphaned MSSM-spectrum predictions in `predictions.md` (no DAG nodes), and (c) descriptive particle-classification entries. Verified against the live numerical comparison table `predicted_parameters.md` (~125 prediction rows, 5 SUSY mentions, all NAMED-CONVENTION). Branch C executed (~7 file edits) reframing public-face language from "Framework commits to SUSY" to "MSSM is a named convention for the β-coefficient values that match observation; substrate-derived matter content is 3 PS gens + 2HDM, no superpartners; literal-particle interpretation is one realization among others, not required by any framework prediction." Specific edits: `honest_assessment.md` L62 + falsifier #6; `master_plan.md` §"Literal-particle β-coefficient gap (R-19)" replacing §SUSY spectrum; `README.md` L99 + L144; `framework_architecture.md` Layer 5 section; `particle_type_classification.md` SUSY-partners row + Open Question #3; `predictions.md` §SUSY Spectrum reframed as honest-conditional; `parameter_uniqueness_ledger.md` Row P58 → RETIRED-conditional. ADOPTED-MSSM-Sb literal-particle adoption STAYS as an open theoretical residue (R-19), but is now precisely bounded rather than open-and-uncharacterized; the language commitment is retired.

**Cross-references.**
- Verdict doc (per-sector): an internal working note.
- Verdict doc (M5): an internal working note.
- Sanity check: `proofs/foundations/sanity_check_visible_alphabet_waterline_2026-05-11.py`.
- M5 enumeration: `proofs/foundations/m5_candidate_enumeration_2026-05-11.py`.
- Probe (per-sector): `proofs/foundations/substrate_rg_beta_function_su2.py`.
- Design: an internal working note.
- Audit history (the dedicated audit doc was deleted on 2026-05-13 along with the retracted theoretical-uncertainty-band framing).
- Gap inventory: an internal working note.
- Path D necessity: `proofs/foundations/mssm_matter_content_required.py`.
- Path E blocker: `proofs/foundations/susy_path_e_witten_substrate.py`.
- Path A inoperative: `proofs/foundations/susy_path_a_anomaly_cancellation.py`.

### 2026-05-14 — REFRAMING: β-coefficient piece is now DERIVED (mathematically complete)

Per the four-thread investigation (Probes A, B, C, D + P-D1) which closed-negative on every attempt to derive literal MSSM particles from substrate, the adoption was re-examined under linter discipline and found to conflate TWO logically separable pieces:

**(A) β-coefficient values (b_1, b_2, b_3) = (33/5, 1, −3).**  DERIVED TOP-DOWN (updated 2026-07-01) by the run's 4D time-completion — `derivation_topdown/bridge/the_run.py` `read_gauge_running`: the 2HDM β (Dynkin sums over the forced Cl(6)-Fock content) plus the COMPUTED completion `(1/3)ΣT_f + (2/3)ΣT_H + (2/3)C₂(G)` reproduces b₁,b₂,b₃ = {33/5, 1, −3} **EXACTLY, with NO PDG input** (verified; R-19 DE-ESCALATION in `structural_residue_register.md`). The +4 to MSSM is this computed time-completion, not an injected adoption.

- The earlier "mathematically complete via `theorem_beta_coefficients_derived.md` (2026-05-14)" route INVERTS b_i from the MEASURED α_i(M_Z) — `b_i = (2π/ln(M_unif/M_Z)) × (1/α_i(M_Z) − 24)`, producing (33/5, 1, −3) to ~1-6% (b_2 +6.22%; observable-level 0.6-2.8%). That is now the **DATA-SIDE CROSS-CHECK, NOT the derivation** (it uses PDG input; per the top-down law it cannot be the *source* of b_i). The exact values come from the completion; the inversion confirms consistency with the observed running.

The one genuine open piece: whether the top-down completion is FORCED (vs merely reproduces the values) — the β-formula's native origin — is ζ_{D₄}(0) (research-level; `docs/incomplete_equations_todo.md` §5). GATES NO VALUE.

**(B) Literal MSSM particle interpretation** (sfermions, gauginos, Higgsinos, gravitino as physical particles).  STILL ADOPTED.  No substrate-derived mechanism identified per the four-thread investigation closures (Probes A-D and P-D1 sessions 1-2, May 2026).  The framework's derived β-coefficients are compatible with — but do not require — literal SUSY particles.  Alternative realizations (threshold matching at M_unif, compositeness, non-perturbative substrate effects) remain candidate mechanisms; none yet structurally derived.

**Net effect on parameter ledger.**  Cluster predictions (P63-P71: α_EM, sin²θ_W, g_1/2/3, α_s, M_Z, m_W at M_Z) currently labeled UNIQUE-THEOREM-GRADE-CONDITIONAL on (ADOPTED-MSSM-Sb, ADOPTED-N_HUB) jointly.  After the reframing, the conditional becomes:

> UNIQUE-THEOREM-GRADE-CONDITIONAL on (β-coefficients DERIVED TOP-DOWN [exact, via the_run.py 4D-completion; forced-ness of the completion = ζ_{D₄}(0), research-level], ADOPTED-N_HUB) jointly.

The numerical content of the predictions does not change; the dependency chain becomes more explicit, and the "adoption" residue is restricted to its honest scope (particle-content interpretation only).

**The "MSSM" label.**  After reframing, "MSSM" in ADOPTED-MSSM-Sb is best read as a *named convention* identifying these specific β-coefficient values — they happen to be the b_i of the canonical MSSM, derived here from substrate boundary conditions.  Whether literal MSSM particles realize these b_i is an experimental/model-building question separate from the framework's derivation.

**Why the four-thread investigation closures didn't preclude this reframing.**  The closures (Path E blocked, Z order-1 fails, walk-based enum tautological, Hashimoto-seesaw GUT-only, Probe B direct closed, P-D1 substrate-direct β ≈ SM) all addressed the QUESTION of literal particle derivation.  They did NOT address the (simpler) question of whether the *β-coefficient values themselves* are derived.  Once the question is asked correctly, the answer is yes — by one-line algebra plus theorem-grade upstream values.

**Status of ADOPTED-MSSM-Sb after reframing:** PARTIALLY GRADUATED.

| component | status |
|---|---|
| β-coefficient values (33/5, 1, −3) | **DERIVED, mathematically complete** |
| Single-regime MSSM-style running (no M_SUSY threshold) | **NATIVE TO FRAMEWORK** (revised 2026-05-14 PM) |
| Literal MSSM particle content (sfermions, gauginos, Higgsinos) | STILL ADOPTED |
| MSSM β-running scheme as nomenclature | retained as named convention |
| M_SUSY (SUSY-breaking scale) | **ELIMINATED** — not a framework parameter (revised 2026-05-14 PM) |

The adoption-register entry remains ACTIVE but with narrowed scope.

**Cross-references (added 2026-05-14):**
- New theorem: `docs/theorems/theorem_beta_coefficients_derived.md`
- Scoping doc: an internal working note
- Four-thread investigation closures: memories `project_probes_CD_layer1_walk_sector_2026-05-14.md`, `project_probe_A_hashimoto_seesaw_2026-05-14.md`, `project_probe_B_C3_B_sign_lock_2026-05-14.md`, `project_Z_CC_reduction_order1_fails_2026-05-14.md`.
- Framework architecture Layer 5: `docs/framework/framework_architecture.md` lines 12, 89, 144.
- Ledger rows affected: `docs/parameters/parameter_uniqueness_ledger.md` Rows P63, P64, P65, P66, P67, P68, P69, P70, P71.
- Bridge convention: `docs/framework/framework_scheme_convention.md` (cluster predictions live at M_Z scale; SM/MSSM RG running, not the bridge convention).

---

## ADOPTED-NU-MAJ-PHASE: ν_R Majorana coupling phase = girth-ring walker holonomy (2026-05-12)

**Statement.** The right-handed-neutrino Majorana mass matrix is diagonal in the C_3 generation modes with

  M_R^(m,m) = |M_R| · h_m^g     (m ∈ {trivial, ω, ω²}),

i.e. the *phase* of the Majorana coupling on generation channel m is `g · arg(h_m)` — one girth-ring's worth of non-backtracking-walker holonomy, where g = 10 (srs girth) and h_m is the Hashimoto walker eigenvalue on that C_3 channel at the P-point (h_trivial = ±1 → phase 0; h_ω = (√3+i√5)/2; h_ω² = (-√3+i√5)/2). The *magnitude* |M_R| = δ⁴·M_Pl/(2·k*·N_atoms) is theorem-grade-conditional via the m_ν₃ closure and is NOT part of this adoption (it is real, phase-free).

**Why it is an adoption, not a theorem.** This is an A5(a)-adjacent identification (A5(a): "Ramanujan eigenvalues = SM mass spectrum"; this extends it to "and the Majorana coupling's phase = the channel eigenvalue's argument, raised to the girth power"). It was attempted to be *derived* — twice — and both attempts fail (`proofs/foundations/majorana_M_R_waterfilling.py`, 2026-05-12):
- **Route 1 (A2-T-waterfilled loop sum)** M_R^(m) = Σ_{L≥g} 2^{-DL(L)}·h_m^L does NOT converge: the Ramanujan saturation |h_m|² = k*−1 = 2 makes every retained closed-walk length contribute with equal magnitude (|2^{-L/2}·h_m^L| = 1 under the natural "which closed NB walk of length L" encoding), so no finite cutoff emerges from the A2-T surprise threshold, and the partial-sum phase drifts as ≈ (g+L_max)/2·arg(h_m). The predicted g·arg(h_m) is only the L_max = g (single-girth-structure) special case — which the rate-distortion machinery does not single out.
- **Route 2 (Path-B "cardinality-k orbit ↔ k girth rings")** is broken at the root: the K_4 cycle-space generators (triangles) have nonzero Z³ voltages {(1,0,0),(0,-1,0),(0,0,1),(1,1,1)}, so they do NOT lift to closed cycles in srs at all — the factor `g` in the chain's phase (k-1)·g·arg(h) is not sourced. (Correction notes added to an internal working note §2 and `path_b_cardinality_reconciliation_2026-05-02.md`.)

So the h^g phase factor is taken as a structural input, not derived. (Also non-derived: `srs_hashimoto_seesaw_verify.py`'s use of the single complex eigenvalue h_m of a C_3 block whose proper trace is real — the complex selection is the +Im-chirality convention.)

**Scope.** Affects ONLY the PMNS Majorana-phase predictions: Rows **P35 (α_21_PMNS ≈ 162.39°)** and **P36 (α_31_PMNS ≈ 324.78°)** — both currently *unmeasured* observables (only 0νββ gives weak combined bounds). The m_ν₂/m_ν₃ magnitude rows are NOT affected (they ride on the phase-free |M_R|). δ_CP_PMNS (Row P34) is independent of this Majorana-phase adoption — it is derived via the V₋₁-T_{B-L} identity (revived 2026-05-05, THEOREM-GRADE-STRUCTURAL = 180°), not via h^g.

**Effect on parameter ledger.** Rows P35, P36 re-graded from **UNIQUE-THEOREM-GRADE-CONDITIONAL** (2026-05-04 EOD+1 — inflated) to **STRUCTURAL-DERIVATION-CONDITIONAL** on (ADOPTED-NU-MAJ-PHASE, C³_gen-L3 mass-ordering, ADOPTED-B3). Same tier as R-9's γ.2 algebraic-K-complexity encoding choice. The predicted values 162.39° / 324.78° hold *under the identification*; they are not falsified, but they are identification-conditional — a correct derivation of the M_R phase might land on a different value.

**Closure paths.** (i) A correct A2-T derivation of the ν_R-Majorana-mass channel's effective object (single girth ring? a finite waterfilled set? something else) — open; the loop-sum route diverges and the K_4-cardinality route's `g` is unsourced. (ii) A phase mechanism not based on a walk power (e.g., directly from the complex Bloch wavefunction ψ_m(P)) — would give a *different*, non-`g`-power phase ⇒ a different prediction. (iii) Promotion of A5(a)/(b) to explicitly cover coupling *phases* (would make this a consequence of an extended axiom rather than a separate adoption).

**2026-05-19 update — CORRECTED (supersedes a same-day overclaim); NO
grade change; NOT narrowed.** A 9-probe arc
(`proofs/foundations/majorana_phase_*_2026-05-19.py`) terminated. An
interim commit (6ac4c69) claimed the phase **factorizes** as `[discrete
ΔL=2 holonomy — derived] × [spectral arg(h) — residual]` and that the
adoption was *narrowed*. **That factorization was an unverified
interpretive bridge and is RETRACTED.** Probe 9 (`majorana_phase_deltaL2_perron`,
gate-verified) proves at the operator level that the ΔL=2 / hypercharge-Y
constraint does **NOT** lift the Ramanujan degeneracy: `|μ_max|=√2` is
*exactly* degenerate (gap ≈ 2e-16) across the entire ΔL=2-relevant Y-tilt
regime **including the structurally-forced P-point**; the only isolated
Perron modes are the trivial closed-sector `|μ|=2` (real, ≠ Majorana, ≠
arg(h)). So there is **no** derived spectral factor; `arg(h)` /
162.39°/324.78° remains **entirely adopted**, irreducibly the
Ramanujan-degenerate spectral eigenphase. **What is real (kept):** probes
8/8a found a clean, cutoff-free, enantiomer-signed *discrete finite-group*
ΔL=2 holonomy (`i^Y·ω^{wY}`; `Y=Vx+Vy+Vz` the C₃-invariant U(1), ν_R its
singlet, `Y=+2/−2` conjugate) — a genuine sub-structure of the
lepton-number-violating sector, but it contains **no `arg(h)`** and does
**not** compose to the physical value. It is **decoupled** from the
Majorana-phase prediction and does **not** narrow this adoption.
`ADOPTED-NU-MAJ-PHASE` stands, undischarged, **unnarrowed**; Rows P35/P36
remain `STRUCTURAL-DERIVATION-CONDITIONAL` (unchanged, not narrowed).
Methodological failure recorded: two same-direction overclaims in one
session (the `J·A_H·J=A_H*` "proven algebra" — refuted; this "decomposition"
— committed then refuted by its own pre-declared probe 9). Full corrected
record: an internal working note.

**Cross-references.**
- Discharge-attempt probe + analysis: `proofs/foundations/majorana_M_R_waterfilling.py`.
- 8-probe arc + decomposition (2026-05-19): an internal working note; `proofs/foundations/majorana_phase_*_2026-05-19.py`.
- Existing constructions it formalizes: `proofs/flavor/srs_hashimoto_seesaw_verify.py` STEP 3, `proofs/foundations/path_b_M_R_upgrade.py`.
- Scoping: an internal working note (2026-05-12 update), an internal working note, `path_b_cardinality_reconciliation_2026-05-02.md`, `path_b_sterile_mode_resolution_2026-05-03.md`.
- Prediction files: `predictions/alpha_21_PMNS.py` + `_derivation.md`, `predictions/alpha_31_PMNS.py` + `_derivation.md`.
- Ledger rows: `docs/parameters/parameter_uniqueness_ledger.md` Rows P35, P36.
- A5(a): `docs/framework/framework_axioms.md` §5b.

**Active adoptions (5, post-2026-05-12):** ADOPTED-B3 (Pati-Salam labeling), ADOPTED-DARK-MAP (β + θ_13 PMNS scope), ADOPTED-A5b-Sub3 (Level 3 sub-class classifier — un-graduated), ADOPTED-MSSM-Sb (MSSM matter content as RG-running scheme), **ADOPTED-NU-MAJ-PHASE** (ν_R Majorana coupling phase = girth-ring walker holonomy h^g — 2026-05-19: a same-day "narrowed" claim was RETRACTED after probe 9; the adoption is NOT narrowed — arg(h) is irreducibly the Ramanujan-degenerate spectral eigenphase; a separate clean discrete ΔL=2 holonomy was found but is decoupled from the physical value).

**★ ACTIVE-ADOPTIONS AUDIT (2026-07-01) — the count-of-5 is STALE; only ~2 are substantively load-bearing.** Per-adoption current status (verified this date):
- **ADOPTED-B3** — down to the lepton-vs-quark **sector-labeling** discrete residue ((Z/2)³ Angle D); hypercharge (G2-D), generation-Z₃ existence + labeling (M1.B/R3), and M_gen non-degeneracy are ALL CLOSED. PS-as-organizing-symmetry (no breaking) is structurally preferred. The residual is A5-semantic-anchor-adjacent, NOT a value-derivation gap.
- **ADOPTED-DARK-MAP** — **genuinely open but NARROW**: only β (cosmic birefringence) + θ_13 PMNS remain (the mass²-class 7-observable family graduated 2026-04-28). LOAD-BEARING on those 2.
- **ADOPTED-A5b-Sub3** — un-graduated classifier, but the amplitudes it gated (V_ub/V_cb/V_us) were RE-DERIVED independently (M1 twisted walker / counting / geometric density) and ship theorem-grade ⇒ NOT load-bearing for any shipped value.
- **ADOPTED-MSSM-Sb** — the β-coefficient VALUES are now DERIVED TOP-DOWN (2026-07-01, `derivation_topdown/bridge/the_run.py` `read_gauge_running` 4D-completion; R-19 de-escalation); only the R-19 literal-particle-interpretation residue remains ⇒ NOT value-gating.
- **ADOPTED-NU-MAJ-PHASE** — **GENUINELY OPEN + identification-conditional**: the h^g Majorana phase (derivation failed twice) gates α_21/α_31 PMNS (both UNMEASURED).
- **ADOPTED-K_P-TIEBREAK** — VACUOUS for srs (the rule yields P uniquely); matters only for cross-substrate audit (qtz); "not required for the predicted observable values" ⇒ NOT a real open adoption for the framework's predictions.

**Net:** substantively open + load-bearing = **ADOPTED-DARK-MAP** (β, θ_13) + **ADOPTED-NU-MAJ-PHASE** (α_21, α_31, both unmeasured). The other four are residual / non-value-gating / vacuous (B3 sector-label, A5b-Sub3 re-derived, MSSM-Sb R-19-only, K_P-TIEBREAK vacuous).

**★ 2026-07-06 ADDENDUM — SCOPE CLARIFICATION: naming the IDENTIFICATION-LAYER adoptions (append-only).** NOTE: the 2026-07-01 "only 2 load-bearing" count was scoped to adoptions gating a *shipped ≤1σ value*, which is a DEFENSIBLE scope (an initial overclaim-audit framing of this as "undercounting" was overstated and retracted — `docs/audits/overclaim_audit_2026-07-06.md` header). But the framework's figure of merit is also "SM structure forced MODULO N maps," and the session `session_consolidation_identification_layer_and_overclaims_2026-07-06.md` established the recurring wall IS the identification layer. Three identification-layer adoptions gate currently-OPEN structure (so they fell outside the shipped-value scope) and are worth NAMING explicitly:
- **ADOPTED-SPECIES-LIFT** — the single-site Cl(6)-Fock occupation → extended-cycle (constituent-walk) SPECIES lift. Confirmed a GENUINE IRREDUCIBLE ADOPTION from THREE angles (EP-2/N1 geometry; E1 per-step; N1b closed-walk holonomy `W_A` conserves only Z₂ parity, mixes N̂). GATES: the entire B1 bound-state/nucleon continent's physical-hadron anchoring ("which ΔS class IS the proton"). Same class as A5(a) matter=Fock. Probes: `BOUND_EP2_dictionary_2026-07-06.py`, `BOUND_EP2_N1b_walk_fock_species_2026-07-06.py`. **PRICED + REFINED 2026-07-07 (`WS1_species_deck_correlation_2026-07-07.py`, pre-reg 5847ae8, ALL PASS, verify 65/65): the adoption is NOT the whole map — a FORCED single-site species×winding correlation core exists (the exact closed-form 4×3 table T(w,t)=Tr(P_w Π^F_t); U_π² dims (4,2,2); bit-even separates singlets/triplets, bit-odd = ONE universal chiral seed (0,±√3/6) shared by both particle-hole pairs). Forced core I(w;t)=0.1813 bits/site; the adoption's residual price = H(w|t)=1.6300 bits/site (of H(w)=1.8113). Book THIS number in any MDL accounting of the gate.** **WS2 (2026-07-07, `WS2_extended_cycle_carry_2026-07-07.py`, pre-reg 0d5942d, ALL PASS, verify 65/65) — the forced single-site correlation does NOT carry to cycles, by an EXACT structural annihilation: the quantity conserved on closed walks is the C₃-averaged coupled deck S², and since the C₃ dart permutation fixes NO darts (Tr(P3)=Tr(P3²)=0), the correlation-carrying (m=1,2) terms vanish ⟹ the conserved cycle-level winding is species-BLIND (I_static=0 exact; I_walk≈0.002). WS1's forced correlation is strictly single-site/Fock-local. ⟹ the natural forced route to a cycle species-assignment is STRUCTURALLY CLOSED, not just un-found; B1's anchoring adoption confirmed irreducible AT THE CYCLE LEVEL (6th angle, mechanism-grade). Residual price H(w|t)=1.630 bits/site stands.**
- **ADOPTED-WINDING-WELD** — the read↔ensemble winding weld (vector-C₃ mass label ↔ spinor-ℤ₆ coupled label, P₃⊗U_π). W1 falsified the forced-bijection PASS ⟹ un-forced. **CONFIRMED un-forced from a 4th angle 2026-07-07 (`ODD_O4_interacting_run_cone_2026-07-07.py`, the odd-channel arc's terminus): coupling E2a's interacting run G_int to the A5(b) CONE, the chiral asymmetry A is LIFT-DEPENDENT — the cone's forced Weyl frame overlaps the E2a Fock vacuum only 0.197 and A changes 60% under the admissible frame swap — so the cone does NOT force the generation resolution either.** GATES: the −70 ppm charged-lepton closure (`incomplete_equations_todo.md` §1). **STRATEGIC:** this is the SAME class as ADOPTED-SPECIES-LIFT (single-site→cycle species map) — the odd-channel arc (O0–O4) and the bound-state continent B1 terminate at ONE shared identification gate. The deck-superposition lead WAS fired 2026-07-07 (WS1, pre-reg 5847ae8) → STRUCTURE: a forced correlation core exists (see ADOPTED-SPECIES-LIFT priced entry); ALSO DERIVED: the cone frame is Z₃-BLIND (I(cone;t)=0 exactly) — the MECHANISM of O4's KILL-WELD (the frame swap collapsed the asymmetry because the cone frame carries zero deck information). The gate itself stays an adoption (labels/assignment un-forced; extended-cycle resolution untouched); its price is now QUANTIFIED at 1.63 bits/site.
- **ADOPTED-BBN-ADIABATICITY** — the adiabatic radiation-bath (α=1, ρ_rad∝a⁻⁴) imported by the √g_* Y_p mechanism, CONTRADICTING the framework's derived rate-balance (`A1_extra_dof_counting`). GATES: the (overclaimed) Y_p +0.8σ; under the honest rate-balance regime Y_p = −65σ. Adjudicated `B2_alpha_convention_Yp_crux_2026-07-06.py`.
**Revised honest figure of merit:** value-gating adoptions on SHIPPED ≤1σ rows = 2 (DARK-MAP, NU-MAJ-PHASE); PLUS ≥3 identification-layer adoptions gating OPEN structure (SPECIES-LIFT, WINDING-WELD, BBN-ADIABATICITY). Total named-and-open ≈ 5. N (the "modulo N maps" figure) is NOT 2. Do not cite "only 2 adoptions" as the theory's completeness without this qualifier.

**2026-06-11 update (Majorana-sector panel, append-only).** Three findings:
(i) the companion α₃₁ as implemented (`predictions/alpha_31_PMNS.py`:
p_toggle·g·arg(h) = 2g·arg(h_ω) = 324.775°) is NOT this adoption's
per-channel h_m^g form, which gives α₃₁ = arg(h_ω²^g) = 197.612° on the ω²
channel — two in-repo conventions conflict (preregistration register
annotation, row-8 defect). (ii) The computed P-saddle C3 character table
contradicts a clean class-to-eigenvalue bijection (every Ramanujan doublet
carries a trivial partner: {1,ω} or {1,ω²}; the ±1 doublets carry {ω,ω²}) —
probe `phase1_3_c3_characters_majorana_fork_2026-06-11.py` gate K3b.
(iii) Invariance status: the class-diagonal form is C3-ALLOWED at the
non-TRIM P saddle under the mirror-crossing law (−P = P+Δ, conjugated
characters) and C3-FORBIDDEN at TRIM saddles (Γ/H/N) under the same-fiber
law. Adoption stands, undischarged; grade unchanged.

**2026-06-12 — Phase-4 panel note on ADOPTED-MSSM-Sb.** Adoption UNCHANGED
at 3.0 bits (running convention). Literal-particle residue sharpened per the
R-19 Phase-4 annotation: the gaugino piece cannot be substrate-realized in
the frozen triple; the sfermion piece is exactly the σ-coupler freedom; the
higgsino piece is dictionary-conditional. Phase-hook row 4 did NOT fire on
either branch.
