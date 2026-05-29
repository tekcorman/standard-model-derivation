# Parameter Uniqueness Ledger — bottom-up audit of numerical predictions

**Date opened:** 2026-04-27.
**Status:** Living document. Parameter pass — companion to `../audits/registers/uniqueness_ledger.md` (the structural pass, 25 rows; Rows 23 q_NB / 24 Hashimoto sector decomposition / 25 substrate-Planck added 2026-04-28).

> **Consolidation pass 2026-05-26.** Row updates this pass: Row P38 (m_t) retracted → THEOREM-GRADE-STRUCTURAL-CONDITIONAL via M_persistence + Type-II saturation (commit `c9fba27`); Row P39 (light quarks) BLOCKED → THEOREM-GRADE-STRUCTURAL-CONDITIONAL via M_persistence; Row P46 (tan β) BLOCKED-ADOPTED → THEOREM-GRADE-STRUCTURAL-CONDITIONAL via M_persistence; Row P31 (m_ν2, m_ν3) UNIQUE-THEOREM-GRADE-CONDITIONAL → DOMINANT-CONDITIONAL (re-grade 2026-05-18 disclosed y_ν = 1 adoption); Row P11 m_τ Claim line refreshed (tree 1779.09 MeV → Family-D 1776.84 MeV); Row P12 m_H Claim line refreshed (tree 125.58 GeV → Family-D 125.195 GeV); Row P34 (δ_CP_PMNS) conditional refreshed to reflect Need-A2 closed 2026-05-08 (Need-D-3 alone is the remaining gating residual). **Outstanding follow-up:** the Audit-v2 (Clause 7) inheritance index (`§Audit v2 inheritance index — added 2026-05-01`) does not yet have per-row entries for P29, P60, P61, or the gauge cluster P63–P71; the W1 2026-05-18 inheritance reclassification (P7/P11/P12/P41 from PASS to STRUCTURAL-CONDITIONAL) is also not propagated to that index. These coverage gaps do not change row-level claims but should be closed in a future audit-v2 pass.
**Purpose.** For every framework numerical prediction, characterise the *operator-permitted alternative formulas* (what `../operator_sweep/operator_sweep_from_A1.md` allows producing the same observable) and the *selection criterion* that picks the framework's formula out of that set.

The structural-pass methodology applies analogously: each row identifies a parameter, its operator-permitted alternative formulas, the selection criterion, status (UNIQUE / DOMINANT / ONE-AMONG-MANY) and any conditional dependencies / gaps / residues.

**Distinct from the structural pass:**
- **Structural pass** asked: among operator-permitted *structures* (alphabets, lattices, groups, algebras), which is selected? Answer typically: hard gates eliminate alternatives; MDL margin distinguishes when needed.
- **Parameter pass** asks: among operator-permitted *formulas* producing a numerical observable, which is selected? Often the formula structure is forced by structural pass results (e.g., V_cb's geometric series structure follows from A2-T waterline once L_cb = 8 is established), and the parameter pass's job is to verify each formula's selection chain is gate-passing and surface alternative formulas that may have been missed.

## Status vocabulary (same as structural pass; BOUNDED added 2026-05-09)

- **UNIQUE.** Every operator-permitted alternative formula is *strictly eliminated* by the selection criterion (M1 hard-gated alternatives, or M2a ΔDL = ∞ via structural identities like Brown-rank Fisher = 0).
- **BOUNDED.** (Added 2026-05-09 walk-down session 7 per Option-1 ceiling reframing.) The framework's formula is the dominant retention, with at least one alternative axis whose plurally-retained alternatives are quantitatively suppressed at finite Boltzmann weight w = 2⁻ᴺ from M2a ΔDL = N bits. The dominance ratio is *named* (not "probably small"); alternatives are formally co-retained under A2-T waterline at quantified suppression. Operationally equivalent to the prior **DOMINANT** label but emphasizes that "DOMINANT-with-quantified-suppression" *is* the framework's standard product, not an intermediate state on the way to UNIQUE.
- **DOMINANT.** Equivalent to **BOUNDED**; preserved for backward-compatibility with audit v2 vocabulary. The framework's formula is the MDL minimum within a non-empty alternative set, with strictly positive (and named) margin.
- **ONE-AMONG-MANY.** Multiple formulas are above-waterline simultaneously; the framework retains the multiplicity (e.g., chirality-residue R-12 pattern) or arbitrarily picks one.
- **CONDITIONAL.** Holds given an upstream theorem or premise. Apply as a suffix to UNIQUE / BOUNDED.
- **GAP.** Selection chain has a missing step at this row's layer.

**Inheritance rule (Option-1 ceiling, added 2026-05-09):** A row's grade is the *minimum* of its own structural grade and the worst grade in its conditional chain. Specifically, a row whose `Conditional on` field includes Row 4 (k* = 3) or Row 6 (srs identification) **inherits BOUNDED grade**, because Theorem 8 (`docs/theorems/theorem_observer_selected_d_periodic_dominance.md`) establishes Bloch-decomposable d=3 substrate as the DOMINANT-not-UNIQUE retention. The dominance ratio is quantified per Theorem 8 §1 (super-exponential suppression for hyperbolic Coxeter; polynomial for d ≥ 4 periodic; polynomial N^(k−d) for cut-and-project Penrose). Existing labels of the form "UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4" are interpreted as **BOUNDED-THEOREM-GRADE-CONDITIONAL** under the Option-1 reading; the row's own derivation may be UNIQUE within scope, but the row inherits BOUNDED via Row-4 inheritance.

**Why the BOUNDED label was added.** The user's 2026-05-09 reframing: "I never 'wanted' this to be unique. If it's not unique, it's not unique. The important part is that we understand the ratio of the dominance." Theorem 8's plural-retention conclusion *is* the framework's standard product. The BOUNDED label makes this explicit at the row level. UNIQUE remains for rows with strict alternative-elimination (e.g., w_DE = −1 exact, θ_QCD = 0 exact, R_ν = 228/7 from Ihara closed form).

## Conventions

Each row has:

- **Claim.** Parameter name + framework formula + value.
- *Source.* `predictions/<name>.py` + `_derivation.md`.
- *Observed.* Best measured value with uncertainty + source.
- *Operations invoked.* From operator catalog + cited theorems.
- *Alternatives.* Operator-permitted alternative formulas producing the same observable.
- *Selection.* Criterion that eliminates non-framework alternatives.
- *Status.* UNIQUE / DOMINANT / ONE-AMONG-MANY + flags.
- *Margin.* If DOMINANT, the bit-margin to next-best.
- *Conditional on.* Upstream rows / theorems.
- *Gap.* If any.
- *Filtered-alternative residue.* Cross-reference to `../audits/registers/structural_residue_register.md` if applicable.

**Audit v2 (Clause 7) inheritance.** A separate index section at the bottom of this doc (added 2026-05-01 M1b.i + M1b.ii, see "Audit v2 (Clause 7) inheritance index") provides the explicit inheritance citation OR own-§3 cite for each UNIQUE-THEOREM-GRADE row. Per `parameter_linter.md` §7, that citation is required for Clause 7 PASS. **After M1b.i + M1b.ii: 41 of 41 UNIQUE rows now PASS Clause 7** — 100% coverage. M1a triage scoreboard: an internal working note. Per-row §3 tables for the 8 hardest rows: an internal working note.

---

## Rows

### Row P1 — α_1 (bare NB walk survival) = 256/6561

- *Claim.* α_1_bare = ((k\*−1)/k\*)^(g−2) = (2/3)^8 = 256/6561 ≈ 0.03902.
- *Source.* `predictions/alpha_1.py`, `predictions/alpha_1_derivation.md`; underlying lemma in `predictions/Feshbach_coupling_strength_derivation.md` Lemma 1.
- *Observed.* Not directly observed at the bare level. The dressed/full coupling α_1_full is observable; bare α_1 is the framework's substrate-level quantity before A2-T waterline geometric resummation (Row P2). At PDG-comparison level, α_EM(M_Z) ≈ 1/127.94 ≈ 0.007816 (different observable; α_1_bare is not α_EM directly).
- *Operations invoked.* Op 1.13 (Cayley-graph distance) + Op 4.11 (Markov chain) + Op 4.12 (stationary distribution: NB walk on k\*-regular graph survives at rate (k\*−1)/k\* per step).
- *Alternatives.* On a k-regular graph, the NB walker's survival amplitude over L steps could be: ((k−1)/k)^L (binomial / Markov, framework's choice); 1/k^L (random walk including backtracking); exponential decay e^(-λL) (continuous-time analog); 1 (no decay); or various Markov chain alternatives with non-uniform transition probabilities.
- *Selection.* (i) NB constraint: at each step, the walker has k\*−1 forward choices and 1 backward choice. (ii) Branch measure μ (Row 12 of structural ledger): per-step weight is uniform over forward choices. → survival per step = (k\*−1)/k\*. (iii) Geometric over L steps (Markov memorylessness, Levin-Peres-Wilmer 2009 Thm 1.14).
- *Status.* **UNIQUE** within "NB walks on k\*-regular graphs with branch measure μ." Direct corollary of structural Row 23 (q_NB = (k\*−1)/k\* = 2/3) raised to power L = g − 2 = 8.
- *Margin.* Strict via (i)+(ii)+(iii) — every step independently follows the (k\*−1)/k\* survival rate.
- *Conditional on.* Row 4 (k\* = 3, structural), Row 9 (g = 10, structural), Row 12 (branch measure μ uniform per-step, structural), Row 23 (q_NB = 2/3, structural — central factorisation row).
- *Gap.* —
- *Filtered-alternative residue.* Backtracking-allowed walks: hard-gated by the Hashimoto NB construction at Layer 4 (op 2.18) — see structural Row 23's filter — this is the natural causal-state walker on directed-edge graphs (Shalizi-Crutchfield 2001 + Stark-Terras 2007 NB walk theorems).

### Row P2 — α_1_full (waterline-resummed geometric series) = 256 / (6561 − 256) = 256/6305

- *Claim.* α_1_full = α_1_bare / (1 − α_1_bare) = (2/3)^8 / (1 − (2/3)^8) = 256/6305 ≈ 0.04060. This is the same numerical structure as V_cb (see Row P3).
- *Source.* `predictions/alpha_1_full.py` + `predictions/alpha_1_full_derivation.md`.
- *Observed.* Same caveat as Row P1 — not directly an observable at this layer.
- *Operations invoked.* Op 4.5 (Shannon entropy), 4.8 (description length), 4.10 (rate-distortion bound). External: Grünwald 2007 §17 (multi-model retention).
- *Alternatives.* Single-winding (= α_1_bare, strict-MDL minimum); double-winding truncation; alternating series Σ (−1)^n α^n; different summation reorderings.
- *Selection.* A2-T waterline (Row 11 structural ledger): every winding number n ≥ 1 has positive savings n·8·log₂(2/3) − log₂(N_cycles) − log₂(n+1) > 0 for all n ≥ 1 (CAS-verified in `predictions/V_cb_derivation.md` Step 4). All windings are above-waterline → all retained → geometric series.
- *Status.* **UNIQUE** under A2-T waterline reading. Falls back to **ONE-AMONG-MANY** under strict-min A2 (single-winding wins, but waterline retains all).
- *Margin.* Strict via above-waterline check at every n; geometric series convergence guarantees bounded sum.
- *Conditional on.* Row 11 (A2-T waterline), Row 23 (q_NB = 2/3 base of the geometric series), Row P1 (α_1_bare).
- *Gap.* —
- *Filtered-alternative residue.* Strict-min A2 reading would have selected α_1_bare alone — this is the same R-12-style structural choice (waterline vs strict-min) that produced parity violation at the structural layer. ACCOUNTED-FOR via R-12.

### Row P3 — V_cb (CKM) = 256/6305

- *Claim.* |V_cb| = α_1 = (2/3)^8 / (1 − (2/3)^8) = 256/6305 ≈ 40.60 × 10⁻³.
- *Source.* `predictions/V_cb.py`, `predictions/V_cb_derivation.md` (THEOREM-GRADE, 0 adoptions).
- *Observed.* PDG 2024 exclusive: 40.5 ± 1.5 × 10⁻³. Deviation: +0.07σ.
- *Operations invoked.* Op 1.13 (distance), 2.18 (Hashimoto NB walker), 4.11–4.12 (Markov + stationary), 4.5 + 4.8 (MDL), Op 4.10 (rate-distortion). External: Sunada 2012 (srs identification), Stark-Terras 2007 (NB walk).
- *Alternatives.* The CKM (2,3) entry could in principle have any of: (i) framework's geometric-series α_1 (selected); (ii) single-winding (2/3)^8 alone (strict-min); (iii) different L_cb (i.e., L ≠ g − 2 = 8); (iv) different walker (random walk vs NB, etc.); (v) alternative formula structures (counting fractions like Row P4 V_us, etc.).
- *Selection.* (i) L_cb = g − 2 = 8 forced by endpoint counting (Type 2, `proofs/flavor/vcb_nfixed_proof.py`) + CAS verification on 8³ supercell finding exactly 20 same-orbit (b₁, b₂) pairs at cycle-distance 8 (`proofs/flavor/vcb_hashimoto_bfs.py`). (ii) Geometric series via A2-T waterline (Row P2). (iii) NB walker via Hashimoto operator (Layer 2.18 + Stark-Terras 2007). (iv) Counting-fraction alternative excluded by A5(b) Level 3 prescription (Hashimoto walk-rep, not Moore-equivalent slots) — see `../theorems/theorem_A5b_level_prescription.md`.
- *Status.* **UNIQUE-THEOREM-GRADE** under A5(b) Level 3 prescription + Rows 4, 6, 9, 11, 12 + **R-9 srs-z CLOSED via polynomial γ.2** (2026-05-02 EOD+8, commit `843cfc9`, `proofs/foundations/r9_srs_z_polynomial_derivation.py`). V_cb itself doesn't discriminate srs vs srs-z (depends only on k, g — both k=3, g=10); R-9 closure operates at the substrate-axis level via Wyckoff free-parameter encoding. See Row P4 (V_us) for the polynomial closure detail.
- *Margin.* Strict within srs-only treatment via the gate-annotated theorem chain; +0.07σ from PDG. **Lattice-axis A2-T waterfilling check** (`proofs/foundations/substrate_lattice_waterfilling_v_cb.py`): V_cb(waterfilled) = V_cb(srs) = 256/6305 exactly. Channel filters: R-7 ths g=4 → V_cb=0.80 R-12-gated; R-8 dia g=6 → V_cb=0.246 R-12-gated; R-4/R-5 gauge-channel hard-gated; finite + R-13 channel-zero. **R-9 multi-tier closure**:
  - Tier (a) V+E-transitive (srs-z, srs-c4, hcb-c4): srs-z CLOSED via polynomial γ.2 (ΔDL = 21.47 bits, +14.08-bit margin to sub-1σ V_us threshold, 2026-05-02). srs unique strongly isotropic per Sunada 2012; srs-c4/hcb-c4 covered by waterline (Options 4+5, 2026-05-02 EOD).
  - Tier (b) V-transitive only / Tier (c) multi-orbit: ASSERTED empty via Phase 1d. Empirical inverse evidence preserved: V_cb match at +0.07σ → if R-9 had non-srs entry at ΔDL=+1 bit, V_cb would shift by 46σ.
- *Conditional on.* Rows 4 (k\*), 6 (srs), 9 (g = 10), 11 (A2-T), 12 (μ), 23 (q_NB = 2/3); A5(b) Level 3 prescription; **R-9 srs-z CLOSED via polynomial γ.2** (Lutz 1998 algebraic-K-complexity adoption); tiers (b)/(c) asserted via Phase 1d.
- *Gap.* A5(b) is not derived from A1 + P1' alone — prior gap unchanged. R-9 srs-z gap CLOSED 2026-05-02; tiers (b)/(c) remain Phase 1d asserted (empirically inverse-validated).
- *Filtered-alternative residue.* Counting-fraction formula: hard-gated by A5(b) Level 3 (V_cb is Hashimoto walk-rep). No residue at parameter level. R-9 srs-z: CLOSED via polynomial γ.2 (2026-05-02). Tier (b)/(c) asserted+empirically-supported. Quantitative bound: `proofs/foundations/substrate_lattice_waterfilling_v_cb.py`.
- *Over-determination cross-lock (2026-05-16).* V_cb = a/(1−a), a=(2/3)⁸, is the resolvent-resummed off-diagonal (species-changing n=1↔n=2) reading of the SAME B_NB(srs) whose Perron projection is δ_r (Row P64) and whose h_P Feshbach reading is δρ (Row P73): δ_r and V_cb are *provably the same* a/(1−a)=256/6305 under projections 1/12 (Perron) vs unit; the bare↔resummed link is the (I−·)⁻¹ geometric series, forced not fitted. Zero fitted constants; 6/6 pre-declared aborts. `proofs/foundations/quark_unification_over_determination_test_2026-05-16.py`; `../theorems/theorem_unified_oblique.md` §8. **Structural cross-lock only — grade, number, and the conditional/gap content above are UNCHANGED** (THEOREM-GRADE-STRUCTURAL extension of the unified-oblique theorem; not theorem-grade-numerical).

### Row P4 — V_us (CKM) = 9/40

- *Claim.* |V_us| = k\*² / (g · N_ATOMS) = 3² / (10 · 4) = 9/40 = 0.225.
- *Source.* `predictions/V_us.py`, `predictions/V_us_derivation.md` (THEOREM-GRADE, 0 adoptions).
- *Observed.* PDG 2024: 0.22500 ± 0.00067 (Vud-Vus unitarity). Deviation: −0.015σ.
- *Operations invoked.* Op 1.11 (Cayley graph), 4.45 (partition function — counting), Brown 1986 (Fisher rank). External: Levin-Peres-Wilmer 2009 (NB walk uniformity); Moore bound.
- *Alternatives.* (i) Geometric series like V_cb (counting fraction's natural alternative — would give different structure). (ii) Different counting-fraction normalization (g·N_ATOMS² vs g²·N_ATOMS vs k\*²·N_ATOMS, etc.). (iii) NB walk amplitude at the V_cb-style level. (iv) Coupling derived from C3 cyclic amplitude (a previously-explored route).
- *Selection.* A5(b) Level 3 prescription with Moore-bound saturation (Type 1 + 2 + 4): Moore-equivalent slots at floor(g/k\*²) = 1 → uniform-counting form k\*²/(g·N_ATOMS). (`../theorems/theorem_A5b_level_prescription.md`.) Distinguishes from V_cb's geometric-series form by structural class: V_cb is Hashimoto walk-rep (Level 3), V_us is Moore-equivalent slot counting (Level 3 sub-class).
- *Status.* **UNIQUE-THEOREM-GRADE** under A5(b) Level 3 prescription + Moore-bound argument + **R-9 srs-z CLOSED via polynomial γ.2** (2026-05-02 EOD+8, commit `843cfc9`, `proofs/foundations/r9_srs_z_polynomial_derivation.py`).
- *Margin.* Strict within srs-only treatment — Moore-bound floor gives k\*²/(g·N_ATOMS) uniquely. **R-9 (chiral non-srs 3D 3-regular RCSR entries) closed via Wyckoff free-parameter γ.2**: srs-z's Wyckoff 8c free parameter $x \approx 0.6607$ is the irrational root of the explicitly-derived 3-regularity boundary polynomial $16x^2 - 32x + 15 = 0$ (degree 2, integer coefficients $\leq 32$). Costed under γ.2 algebraic-K-complexity (Lutz 1998), Wyckoff free-parameter encoding adds 19.07 bits to srs-z's structural DL. Combined with +2.40 bits Level-2 ΔDL (primitive-cell atom count + directed-edge orbit count), total ΔDL(srs-z − srs) = 21.47 bits, exceeding the sub-1σ V_us-match threshold of 7.39 bits by +14.08 bits. R-9 closes to sub-1σ via M2a structural alone, conditional on adopting γ.2 algebraic-K-complexity (Lutz 1998 computable-real Kolmogorov complexity) as the MDL convention for Wyckoff free parameters. M2b data-conditional MDL remains supplementary only — non-load-bearing per 2026-05-01 PM rule.
- *Conditional on.* Rows 4, 6, 8, 9; A5(b) Level 3 prescription with sub-class identification; **R-9 closed via polynomial γ.2** (Lutz 1998 algebraic-K-complexity methodology adoption).
- *Gap.* A5(b) Level 3 sub-class identification (prior gap, unchanged). R-9 srs-z structural gap CLOSED 2026-05-02 EOD+8.
- *Filtered-alternative residue.* Other sub-class choices: hard-gated by A5(b) Level 3 prescription's specific structural criterion. The 9 V_us routes blocked in earlier sessions are all soft-gated alternative formulas. R-9 srs-z: CLOSED via polynomial γ.2 — ΔDL margin +14.08 bits to sub-1σ V_us threshold. Probe verification: `proofs/foundations/r9_srs_z_polynomial_derivation.py` + `substrate_lattice_waterfilling_v_us.py`.
- *Over-determination cross-lock (2026-05-16).* V_us = k\*²/(g·N_ATOMS) = 9/40 is the *counting-projection* off-diagonal reading of the SAME B_NB(srs) whose Feshbach W/h_P reading is δρ (Row P73) and Perron reading is δ_r (Row P64) — a heterogeneous-family reading of the one operator (unified-oblique precedent: counting and Feshbach families coexist on one B). Zero fitted constants; verified jointly with V_cb/V_ub/δρ/δ_r at one spectral datum, 6/6 pre-declared aborts. `proofs/foundations/quark_unification_over_determination_test_2026-05-16.py`; `../theorems/theorem_unified_oblique.md` §8. **Structural cross-lock only — grade, number, and the conditional/gap content above are UNCHANGED** (THEOREM-GRADE-STRUCTURAL; not theorem-grade-numerical).

### Row P5 — Dark coefficient c = 5/12

- *Claim.* The dark-correction coefficient c = 5/12 = 0.41667 for delocalized-amplitude observables (per `../theorems/theorem_dark_correction_mdl.md` Lemmas 1+2 + the Im(h)/|h| polar-decomposition selection).
- *Source.* `../theorems/theorem_dark_correction_mdl.md`; multiple downstream predictions (H_0, t_0, Omega_DM, etc.).
- *Observed.* Indirect — the c = 5/12 enters as a coefficient in dark-correction formulas; its specific value affects predictions like H_0 = 68.18 km/s/Mpc (+1.6σ CMB) and Omega_DM.
- *Operations invoked.* Op 4.5, 4.8, 4.10 (MDL), 4.45 (partition function). External: Stage 2a edge-surprise thresholds; Grünwald 2007.
- *Alternatives.* Different rational coefficients arising from MDL-bookkeeping on parity-odd functionals of h: 5/12 (selected), various other fractions from the same MDL chain at different sub-step counts, exponential / non-rational alternatives.
- *Selection.* `../theorems/theorem_dark_correction_mdl.md` derives c = 5/12 as the MDL-weighted sum of parity-odd functional contributions, with the chain A2 → F0 (edge process dissolution) → F1 (winding) → F2 (parity-odd) → F3 (Im(h)/|h| selection) → c = 15/36 = 5/12. The chain is fully gate-annotated and theorem-grade per Session 18 (memory 2026-04-22).
- *Selection (independent spectral route, added 2026-04-28).* `proofs/wave_engine/dark_5_12_spectral.py` derives c = 5/12 as the dimensional fraction of the **marginal real-non-Perron sector** of the Hashimoto operator B at Γ for the srs primitive cell. By the Stark-Terras factorization of B's characteristic polynomial: det(uI − B) = (u² − 1)^(|E|−|V|) · ∏_λ (u² − λu + (k*−1)). For srs (|E|=6, |V|=4, k*=3): bipartite factor contributes 4 marginal eigenvalues at u=±1 (mult 2 each); the λ_A=+3 factor's other root gives 1 additional marginal at u=+1; the λ_A=−1 factors (×3) give 6 oscillatory complex eigenvalues. Marginal/total = 5/12 = (2(|E|−|V|)+1)/(2|E|). This is the rank of the Feshbach Q-projector on the substrate's NB-walk space, derived from the spectral decomposition rather than cycle counting. Both routes give 5/12 because they compute the same projector rank via different decompositions: cycle-counting gives 15/(N_ATOMS·k*²) = 15/36; spectral gives (2(|E|−|V|)+1)/(2|E|) = 5/12. The two routes are linked by the identity n_g = |V|·k*(k*−2) + k* (= 15 for srs), which reflects srs's spectral-cycle correspondence.
- *Status.* **UNIQUE** under either derivation chain. The dual route (cycle-counting + spectral) acts as a cross-check: a single number derived two ways lifts the closure from "single chain to verify" to "structurally-overdetermined".
- *Margin.* Strict via the explicit Lemma 1+2+3 chain (cycle route) AND the explicit Stark-Terras factorization (spectral route).
- *Conditional on.* Row 11 (A2-T), Row 23 (q_NB = 2/3 — the (2/3)^L MDL stack's base), Stage 2a (Type 4), the parity-odd-functional MDL machinery (cycle route); Row 7 (|E|=6), Row 16 (Cl(6) per node forces |V|=4 via K_4 quotient), Row 24 (Hashimoto sector decomposition 1+6+5 at Γ via Stark-Terras factorization — 5/12 = dim(marginal)/dim(B)), k*=3 (spectral route).
- *Gap.* —
- *Filtered-alternative residue.* Other parity-odd functionals at different MDL sub-cost: catalogued in `theorem_dark_correction_mdl.md` and either selected or below-waterline. The "ONE mechanism, MDL-derived" an internal note from 2026-04-25 confirms this is a single underlying mechanism, not separate ad-hoc rules. The spectral cross-check confirms no alternative dark-projector structure can give the observed 5/12 with srs's |V|=4 / |E|=6 / k*=3 constraints — for any other (|V|, |E|, k*) the marginal fraction would differ.
- *Audit v2 note (2026-04-30 EOD; reframed 2026-05-01 PM).* Phase 3 closure (`uniqueness_audit_v2_phase_3_P5_P28_P15_2026-04-30.md`) flagged Class A audit pattern (cycle/spectral routes coincide at k=3 only) as a downgrade reason. **Primary closure (structural):** the spectral derivation in `proofs/wave_engine/dark_5_12_spectral.py` + `../theorems/theorem_dark_5_12_spectral.md` gives c = (2(|E|−|V|)+1)/(2|E|) = 5/12 forced by srs primitives (|V|=4, |E|=6, k*=3) via the Stark-Terras factorization of the Hashimoto characteristic polynomial; for qtz at k=4 the same spectral identity yields c = 7/12 ≠ 5/12. The cycle-counting and spectral routes are linked by the structural identity n_g = |V|·k*(k*−2) + k* (= 15 for srs), and Row 4's Brown-rank closure (per an internal working note §1) selects k*=3 structurally. **Supplementary empirical validation (NOT closure mechanism):** the data-conditional MDL follow-up (`uniqueness_audit_v2_data_conditional_mdl_2026-04-30.md`) shows ~10² bit disagreement at the c-coefficient level alone and ~2×10⁸ bits globally for qtz vs PDG; this confirms the structural exclusion is correct but does not itself provide closure. Earlier framing of data-conditional MDL as "reconfirms UNIQUE" was goal-seeking and is RETRACTED 2026-05-01 PM per an internal note (REVISED). Row P5 retains UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 Brown-rank + spectral derivation chain.

### Row P6 — sin²θ_W(M_unif) = 3/8

- *Claim.* The weak mixing angle at the framework's unification scale is sin²θ_W = 3/8 = 0.375.
- *Source.* `predictions/sin2_theta_W.py`, `../theorems/theorem_sin2_theta_W_unification.md`.
- *Observed.* Standard SU(5) / Pati-Salam unification value: 3/8 (matches PDG running of α_EM up to M_unif). Direct comparison at M_Z requires RG running.
- *Operations invoked.* Op 4.30–4.38 (group rep theory: traces, characters, Clebsch-Gordan). External: standard Pati-Salam representation theory.
- *Alternatives.* Other unification-scale predictions: sin²θ_W = 3/13 (an alternative derivation route, retracted), 0.231 (M_Z value), or other rational fractions from different gauge embeddings.
- *Selection.* Path γ + B6 color-Z₃ multiplicity argument: tan²θ_W = (3/5) · (Σ Tr(T₃²) / Σ Tr(Q²)) over the matter content of one Pati-Salam family, evaluated using the (4, 2, 1) + (4̄, 1, 2) decomposition. This gives sin²θ_W = 3/8.
- *Status.* **UNIQUE** under the framework's Pati-Salam embedding (Row 17 structural).
- *Margin.* Strict — group-theoretic trace identity.
- *Conditional on.* Row 17 (Pati-Salam), Row 18 (n_generations = 3, for the matter content).
- *Gap.* —
- *Filtered-alternative residue.* Other gauge embeddings (SU(5) Georgi-Glashow, SO(10), E_6) would give different tan²θ_W: hard-gated by Row 17 (Pati-Salam selected by srs cubic symmetry). The retracted 3/13 derivation (per memory 2026-04-24 Session 25) was a different chain that did not gate-pass; the 3/8 derivation supersedes.

### Row P7 — y_τ = 1280/177147

- *Claim.* The tau Yukawa coupling is y_τ = 1280/177147 ≈ 7.226 × 10⁻³.
- *Source.* `predictions/y_tau.py`, `../theorems/theorem_ytau_corollary.md` (THEOREM, 0 adoptions).
- *Observed.* Computed from m_τ / v: m_τ = 1.77686 GeV (PDG 2024), v = 246.22 GeV → y_τ = 7.214 × 10⁻³. Deviation: ~+0.17σ.
- *Operations invoked.* Op 4.30–4.36 (group rep + Clebsch-Gordan), Op 5.34 (partition function ZZ(β)), C₃ irrep machinery on observer C³_obs (Row 18).
- *Alternatives.* Different rational forms at the same numerical scale; different power chains (3^x, 2^y, 5^z combinations); non-rational asymptotic forms.
- *Selection.* Direct-moment derivation (per `theorem_A5b_level_prescription.md` Case (a.ii) — Level 2 srs-intrinsic coupling): A5(b) direct-moment for couplings of α₁'s tan²(arg h) form at the C₃-protected eigenspace. Combined with C₃-orbit prescription on the lepton sector, gives 1280/177147 = 2^8 · 5 / 3^11.
- *Status.* **THEOREM-GRADE-STRUCTURAL, conditional** (corrected W1 2026-05-18; the 2026-05-15 "UNIQUE-NUMERICAL via Family D theorem-grade" was a Clause-6c smuggle): Clause 7 PASS for c_H (Routes H+C genuinely independent); c_F via the Clause-6 channel_select→canonical_encoding two-step (conditional on the single-edge-vs-gauge-singlet channel argument, δ_r's tier); Clause 8 numeric UNCHANGED. Predicted (Family D-corrected) y_τ = 7.2165e-3 vs observed (m_τ/v) 7.2166e-3 → -0.0012% deviation = **-0.17σ_PDG**. Family D mechanism graduates from LAYER-1 HYPOTHESIS to THEOREM-GRADE 2026-05-15: Routes H + C closed for c_H = α₁_bare² (joint Hashimoto-spectral / m=2 closed-bubble — genuinely two independent routes); c_F = -α₁²/(N_atoms·k*) via the Clause-6 channel_select→canonical_encoding two-step (the historical "Routes F-1/F-2" are canonical_encoding-EQUIVALENT via the Euler identity 2|E|=N·k*, NOT independent — that framing was the Clause-6c smuggle). Per master doc §8 rule 6, theorem-grade Family D propagates to numerical .py predictions; y_τ Yukawa-chain residual closes from +0.13% (tree-level) to -0.0012% rel.err. Files: `proofs/foundations/family_D_route_{H,C,F,F2}_2026-05-15.py`; master doc `../theorems/theorem_substrate_feshbach_dark_corrections_master.md` §3 (D); `predictions/y_tau.py` propagated 2026-05-15. CORRECTED W1 2026-05-18: the THEOREM-GRADE-STRUCTURAL (conditional) label is REINSTATED — the 2026-05-15 "UNIQUE / Routes-F-1+F-2 / theorem-grade" graduation was a parameter_linter Clause-6c smuggle (unnamed MDL-bit-cost minimum). c_F's genuine derivation is the Clause-6 channel_select→canonical_encoding two-step (F-1≡F-2 encoding-equivalent, not independent), verified via the real simulator/gating/mdl gate; numeric values UNCHANGED. See master doc §3 (D) + predictions/dark_extraction_map.py _c_F_denominator_channel_select.
- *Margin.* Strict — direct moment computation. Numerical: +0.13% deviation, consistent with bridge convention's stated ~0.5% floor for un-derived Feshbach analogs. Family D candidate (2026-05-15) sharpens this to the per-leg-counting structural form on the 1H+2F Yukawa vertex.
- *Conditional on.* Rows 4, 6, 17, 18, 23 (q_NB = 2/3 base for the α₁-tan² direct-moment chain); A5(b) Level 2; **+ sub-leading Yukawa correction (research-level, would close THEOREM-GRADE-STRUCTURAL → THEOREM-GRADE-NUMERICAL).**
- *Gap.* A5(b) Level 2 sub-class assignment depends on the formula structure (whether it's a direct-moment or walk-sum at α₁'s level), which is itself classified per `theorem_A5b_level_prescription.md`. **+ Sub-leading Feshbach correction not yet derived; +0.13% systematic propagates through entire y_τ chain.**
- *Filtered-alternative residue.* Other A5(b) sub-classes (Case b, walk-sum) would give different forms — hard-gated by the formula's classification per Level 2 prescription.

### Row P8 — Q_Koide = 2/3

- *Claim.* The Koide quadratic ratio Q = (m_e + m_μ + m_τ)² / (m_e² + m_μ² + m_τ²) = 2/3.
- *Source.* `predictions/Q_Koide.py`, `predictions/Q_Koide_derivation.md`.
- *Observed.* From PDG 2024 charged-lepton masses: Q_observed = 0.66666133 ± 0.00000026. Framework prediction: exactly 2/3 = 0.66666666... Deviation: −0.20σ. Among the most precisely-measured agreements in the framework.
- *Operations invoked.* Op 5.9 (spinor reps), 5.32 (complex Clebsch-Gordan), Op 4.30–4.38 (group rep / characters); C₃-orbit averaging on the C³_obs lepton triple.
- *Alternatives.* Q can take any value in [1/3, 1] for three positive masses (Foot 1994). Specific rational values from group-theoretic averages: 2/3 (selected), 1/2, 1, etc.
- *Selection.* Koide ratio = 2/3 follows from the framework's identification of the three charged-lepton masses with eigenvalues of a specific Cl(6;ℂ)-spinor mass matrix on C³_obs (Row 18) under C₃ orbit averaging. Foot 1994 noted this geometrically as the angle between the mass eigenvector and the (1, 1, 1) direction; the framework derives this geometry from Pati-Salam + Cl(6) spinor structure.
- *Status.* **UNIQUE** within the framework's Pati-Salam + C₃-observer chain.
- *Margin.* Strict — geometric identity.
- *Conditional on.* Rows 16 (Cl(6;ℂ)), 17 (Pati-Salam), 18 (n_generations = 3).
- *Gap.* —
- *Filtered-alternative residue.* Alternative spinor structures (Cl(8) etc.) would give different Q-ratios: hard-gated by Row 16 (Cl(6;ℂ) forced). The Koide ratio's value-of-2/3 is geometric, not numerical-coincidental.

---

### Row P9 — ε_Koide² = 2 and δ_Koide = 2/9

- *Claim.* ε² = 4·μ_ω / μ_trivial = 2 (exact); δ = Q·(1 − Q) = (2/3)(1/3) = 2/9 (exact rational).
- *Source.* `predictions/epsilon_Koide_derivation.md`, `predictions/delta_Koide_derivation.md`.
- *Observed.* Both consistent with PDG charged-lepton masses at the same precision as Q_Koide (Row P8).
- *Operations invoked.* Op 4.30–4.38 (group rep + Clebsch-Gordan), 5.9 (spinor reps), 5.14 (partial trace = A3-T Born rule); Jaynes 1957 max-entropy on C₃ multiplicities (4, 2, 2) of the 8-dim Ramanujan subspace of B(P) on srs.
- *Alternatives.* Other rational identities arising from Cl(6;ℂ) Clebsch-Gordan averages; non-rational residues; alternative Koide-parameterization choices.
- *Selection.* C₃ multiplicities (4, 2, 2) on B(P)'s 8-dim Ramanujan subspace are graph-theoretically forced (Row P3-style); A3-T Born rule converts multiplicities into amplitudes; Jaynes max-entropy + A2-T waterline pins the relative weights; algebraic identities give ε² = 2, δ = 2/9.
- *Status.* **UNIQUE** under A1 + A2-T + A3-T + Pati-Salam + C₃-observer (Rows 16, 17, 18 structural).
- *Margin.* Strict — algebraic identities once multiplicities and Born rule are fixed.
- *Conditional on.* Row 18 (C³_obs), Row 17 (Pati-Salam), Row 16 (Cl(6;ℂ)); A3-T (Row 13) Born rule.
- *Gap.* —
- *Filtered-alternative residue.* Cross-cite to `../../predictions/B_P_doubly_degenerate_h_derivation.md` for the (4, 2, 2) multiplicity derivation. No new residue.

### Row P10 — v_Higgs = 246.22 GeV

- *Claim.* v_Higgs = δ²·M_P / (√2·N^(1/4)) where N = N_hub, the framework's adopted dimensional input. Numerical: 246.22 GeV. (Matches by construction — N_hub's value is calibrated via the measured G_F, which is downstream of exactly this chain; cf. Row P17.)
- *Source.* `predictions/v_higgs.py`, `predictions/v_higgs_derivation.md`.
- *Observed.* PDG 2022 electroweak precision: 246.22 ± 0.12 GeV.
- *Operations invoked.* Op 4.45–4.47 (partition function, Boltzmann), 4.51 (BZJ scaling v ∝ N^{−1/4}), 5.12–5.14 (density matrix, partial trace).
- *Alternatives.* Alternative VEV formulas: BZJ exponent ≠ −1/4 (would correspond to non-quartic potentials); different observables to calibrate N's value (M_P, H_0, t_0, G_F); non-mean-field analogs.
- *Selection.* (i) BZJ scaling v ∝ N^{−1/4} forced by O(n) ϕ⁴ universality class at criticality (Brézin & Zinn-Justin 1985, op 4.51) given the Higgs sector's quartic potential structure. (ii) δ² coefficient = (D¹₁₀/k\*)² from Wigner-D¹ at screw 4₁ on srs (theorem-grade, `../theorems/theorem_g3_higgs_coefficient.md`). (iii) Calibration choice for N_hub's value: the measured G_F (700× better precision than H_0 per session 19) — N_hub is the adopted input.
- *Status.* **UNIQUE — THEOREM-GRADE** (graduated 2026-04-28 via full G1b R2 path closure; inherits Row P17; η-sketch ELIMINATED 2026-04-28 PM). BZJ universality + g3 coefficient theorem + structural N derivation via R2 (matches cascade exactly).
- *Margin.* Strict via BZJ universality + g3 coefficient theorem; numerical value matches by construction (N_hub calibrated via G_F).
- *Conditional on.* Rows 16, 17, 18 structural; Row P17 (N_hub, now UNIQUE-THEOREM-GRADE).
- *Gap.* — (η-sketch sub-residue ELIMINATED 2026-04-28 PM via `proofs/foundations/g1b_r2_eta_full_closure.py`).
- *Filtered-alternative residue.* —

### Row P11 — m_τ = 1776.84 MeV (Family-D corrected; and m_μ, m_e via Koide)

- *Claim.* m_τ = v × y_τ × Family-D = 246.22 GeV × (1280/177147) × (1 − (5/6)·α₁_bare²) → 1776.84 MeV (Family-D-corrected, live). Tree-level pre-Family-D was 1779.09 MeV; the Family-D vertex correction (1 Higgs + 2 fermion legs) graduated 2026-05-15 brings the predicted value to within −0.17σ_PDG. m_μ and m_e are theorem-grade ratios of m_τ via the Koide f_j structure.
- *Source.* `predictions/m_tau.py`, `predictions/m_tau_derivation.md`; m_e and m_μ via Q_Koide, ε_Koide, δ_Koide (Rows P8, P9).
- *Observed.* PDG 2024: m_τ = 1776.86 ± 0.12 MeV. Deviation: +0.126%.
- *Operations invoked.* Same as Rows P7 (y_τ) and P10 (v_Higgs).
- *Alternatives.* Different multiplicative formula structures; non-Koide ratio relations among {m_e, m_μ, m_τ}.
- *Selection.* m_τ = v × y_τ is the standard Yukawa relation (Type 3, standard SM definition of Yukawa coupling). m_e and m_μ from m_τ via Koide identity Q + ε² + δ = 2/3 + 2 + 2/9 (algebraic identities); only one independent lepton-mass prediction (m_τ), the others are ratios.
- *Status.* **THEOREM-GRADE-STRUCTURAL, conditional** (corrected W1 2026-05-18; the 2026-05-15 "UNIQUE-NUMERICAL via Family D theorem-grade" was a Clause-6c smuggle): Clause 7 PASS for c_H (Routes H+C genuinely independent); c_F via the Clause-6 channel_select→canonical_encoding two-step (conditional on the single-edge-vs-gauge-singlet channel argument, δ_r's tier); Clause 8 numeric UNCHANGED inherited from Row P7. Predicted (Family D-corrected) m_τ = 1776.84 MeV vs PDG 1776.86 MeV → -0.0013% = **-0.17σ_PDG** (from prior +18.67σ_PDG tree-level). m_e and m_μ inherit Family D via Koide ratios: m_e = 510.96 keV (vs 510.999 keV, -0.0083% = within experimental precision), m_μ = 105.6506 MeV (vs 105.6584 MeV, -0.0074%). All three lepton masses now sub-σ_PDG. `predictions/{m_tau,m_e,m_mu}.py` propagated 2026-05-15. CORRECTED W1 2026-05-18: the THEOREM-GRADE-STRUCTURAL (conditional) label is REINSTATED — the 2026-05-15 "UNIQUE / Routes-F-1+F-2 / theorem-grade" graduation was a parameter_linter Clause-6c smuggle (unnamed MDL-bit-cost minimum). c_F's genuine derivation is the Clause-6 channel_select→canonical_encoding two-step (F-1≡F-2 encoding-equivalent, not independent), verified via the real simulator/gating/mdl gate; numeric values UNCHANGED. See master doc §3 (D) + predictions/dark_extraction_map.py _c_F_denominator_channel_select.
- *Margin.* Strict via algebraic identities (structural). Numerical: +0.126% inherits y_τ chain. **Family D candidate inherits from P7, sentinel-passes** at <1% rel. err. on residual.
- *Conditional on.* Rows P7 (y_τ — itself THEOREM-GRADE-STRUCTURAL with 0.13% gap, **+ 2026-05-15 Family D candidate**), P10 (v_Higgs).
- *Gap.* — (structural η-sketch eliminated 2026-04-28 PM). **+ y_τ sub-leading correction needed (research-level) to close THEOREM-GRADE-STRUCTURAL → THEOREM-GRADE-NUMERICAL.** **2026-05-15: Family D per-leg mechanism is the candidate closure** awaiting Routes H+C derivation.
- *Filtered-alternative residue.* —

### Row P12 — m_H = 125.20 GeV (Family-D corrected; and λ_Higgs)

- *Claim.* m_H = √(2λ_Higgs) · v × Family-D ≈ 125.195 GeV (Family-D-corrected, live; −0.05σ_PDG). Tree-level pre-Family-D was 125.58 GeV (+3.43σ_PDG FAIL); the Family-D vertex correction on the |φ|⁴ vertex (4 Higgs legs, 0 fermion legs — δλ/λ = −4·α₁_bare²) graduated 2026-05-15 brings the prediction to PDG-class match.
- *Source.* `predictions/m_H.py`, `predictions/m_H_derivation.md`; `predictions/lambda_higgs.py`.
- *Observed.* PDG 2025: 125.20 ± 0.11 GeV. Tree-level deviation: +3.43σ_PDG (FAIL Clause 8 vs σ_PDG alone). **Family D-corrected deviation: −0.05σ_PDG (PASS Clause 8 vs σ_PDG alone).** Predicted (Family D) m_H = 125.195 GeV.
- *Operations invoked.* Op 5.8 (Cl(2;ℂ)), 5.17 (tensor products with entanglement), 4.45–4.47 (partition function); standard SM tree-level Higgs mass relation m_H² = 2λv².
- *Alternatives.* Different λ_Higgs formula structures; loop-level corrections instead of tree-level.
- *Selection.* λ_Higgs derived from Cl(2) channel structure on per-edge SU(2) qubit (`../theorems/theorem_g2_edge_qubit_su2.md` + `../theorems/theorem_mdl_mean_field_higgs.md`); m_H from standard tree-level relation.
- *Status.* **THEOREM-GRADE-STRUCTURAL, conditional** (corrected W1 2026-05-18; the 2026-05-15 "UNIQUE-NUMERICAL via Family D theorem-grade" was a Clause-6c smuggle): m_H/λ ride **c_H ONLY** — the |φ|⁴ vertex is 4 Higgs legs, 0 fermion legs, so the c_F fermion-leg conditional does NOT enter here. c_H = Route H structurally derived + Route C corroboration (genuinely two independent routes; Clause 7 PASS for c_H). Conditional only on the c_H (g-2) joint-excursion assumption — NOT the c_F channel argument that affects y_τ/m_τ. Clause 8 numeric UNCHANGED: tree-level was +3.43σ_PDG (FAIL); Family-D(c_H)-corrected is **−0.05σ_PDG (PASS)** via δλ/λ = -4·α₁_bare² on the |φ|⁴ vertex (4H legs). c_H Routes H+C closed at exact rational arithmetic (`proofs/foundations/family_D_route_{H,C,F,F2}_2026-05-15.py`); master doc `../theorems/theorem_substrate_feshbach_dark_corrections_master.md` §3 (D). Propagated to `predictions/{lambda_higgs,m_H}.py` 2026-05-15. CORRECTED W1 2026-05-18: the THEOREM-GRADE-STRUCTURAL (conditional) label is REINSTATED — the 2026-05-15 "UNIQUE / Routes-F-1+F-2 / theorem-grade" graduation was a parameter_linter Clause-6c smuggle (unnamed MDL-bit-cost minimum). c_F's genuine derivation is the Clause-6 channel_select→canonical_encoding two-step (F-1≡F-2 encoding-equivalent, not independent), verified via the real simulator/gating/mdl gate; numeric values UNCHANGED. See master doc §3 (D) + predictions/dark_extraction_map.py _c_F_denominator_channel_select.
- *Margin.* Tree-level was +3.4σ_PDG (FAIL Clause 8); Family D-corrected is **-0.05σ_PDG (PASS Clause 8)**. Structural derivation (Clause 7) PASS; numerical match (Clause 8) PASS post-Family D theorem-grade closure 2026-05-15.
- *Conditional on.* Row P10 (v_Higgs, now UNIQUE-THEOREM-GRADE post G1b R2 closure), Row P7 (y_τ, theorem-grade), Row 20 (Higgs doublet), Row 22 (Cl(2) pseudoscalar orientation), `../theorems/theorem_dark_map_class2_closure.md` Theorems 3.1 + 4.1, `../theorems/theorem_g1b_r2_closure.md`.
- *Gap.* The 0.5% residual on λ (and 0.08% on m_H) is the un-derived Feshbach-analog gap on the Higgs quartic — separate scoping an internal working note (Priority 4.4 step 2.1). NOT a dark-map taxonomy issue, NOT a G1 issue. Closure of this gap would tighten Clause 8 further but isn't blocking current label. **2026-05-15: Family D candidate hypothesis identified** as the precise structural mechanism (per-leg multiway dark-disruption from non-srs co-retained alternative srs-z on |φ|⁴ vertex).
- *Filtered-alternative residue.* —

### Row P13 — PMNS θ_23 (and queued: θ_12, θ_13)

- *Claim.* θ_23 = some specific framework-derived value (per `predictions/theta_23_PMNS_derivation.md`).
- *Source.* `predictions/theta_23_PMNS.py`, `predictions/theta_23_PMNS_derivation.md`.
- *Observed.* PDG 2024 atmospheric mixing angle (NO ordering): sin²(θ_23) ≈ 0.561.
- *Operations invoked.* Op 5.14 (partial trace), 5.32 (complex CG); A5(b) Level 3 walk-sum prescription on PMNS sector.
- *Alternatives.* Different formula structures for PMNS angles: walk-sum vs counting fraction vs direct moment; alternative dark-map class assignments.
- *Selection.* A5(b) walk-sum prescription on the lepton sector with C³_obs structure; specific algebraic combinations.
- *Status.* **STRICT-SOLID THEOREM-GRADE for θ_23** (graduated 2026-04-28 via `../theorems/theorem_dark_map_class2_closure.md` Theorem 5.1). I-Feshbach closed via A5(b) (2026-04-19); dark-map Class-2 taxonomy closed for θ_23 (2026-04-28). θ_12 and θ_13 PMNS have separate sub-class identifications (per Rows P32, P33).
- *Margin.* 0.4σ from PDG.
- *Conditional on.* Row 18 (C³_obs), Rows 16, 17; A5(b) Level 3 prescription (now ADOPTED-A5b-Sub3 graduated to theorem-grade per memory 2026-04-28); `../theorems/theorem_dark_map_class2_closure.md` for Class-2 ID.
- *Gap.* — (Class-2 dark-map gap closed for θ_23). PS-embedding step for θ_13 PMNS family is separately tracked at Row P33.
- *Filtered-alternative residue.* —

### Row P14 — V_ub = 3.767 × 10⁻³: UNIQUE-THEOREM-GRADE for amplitude; labeling data-anchored

- *Claim.* V_ub = Σ_{m ≥ 2} (2/3)^{6m+2} / (1 − (2/3)^{6m+2}) ≈ 3.767 × 10⁻³ (sum over all m ≥ 2; CAS-computed in `proofs/flavor/vub_multicycle_sum.py`).
- *Source.* `predictions/V_ub.py`; `predictions/V_ub_derivation.md`; canonical compute script `proofs/flavor/vub_multicycle_sum.py`. Amplitude form closed at theorem grade via `proofs/foundations/m1_twisted_walker_v_cb_v_ub.py` + `m1_n_orbit_3orbit_basis.py` (M1 amplitude-form closure 2026-04-30, commit 753f4cf). Labeling layer reframed data-anchored via an internal working note + an internal working note (commit e5ef667). Bridge-functoriality lemma graduation RETRACTED 2026-04-29 (no longer needed; superseded by M1 amplitude-form closure).
- *Observed.* PDG 2024: 3.82 ± 0.20 × 10⁻³. Multi-cycle sum formula at **−0.26σ from PDG** — best match.
- *Operations invoked.* Hashimoto walk-rep + branch measure (Theorem of multiway branch measure §3+§4) + Feshbach exponent principle at n_fixed=2 + A5(b) Case B walk-rep + multi-cycle host topology + 16-cycle decomposition uniqueness CAS + twisted walker T = B · C_36 in N-orbit cyclic 3-orbit basis (M1 amplitude-form 2026-04-30).
- *Alternatives.* (i) Multi-cycle sum over all m ≥ 2 (SELECTED, −0.26σ, working numerical formula; theorem-grade amplitude via M1 twisted walker). (ii) Leading m=2 only ≈ 3.437e-3 (−1.91σ) — structurally cleaner ("irreducible composite host") but worse numerical match. (iii) V_us · (2/3)^g = 128/32805 ≈ 3.902e-3 (+0.40σ) — REFUTED 2026-04-28 (substrate-Z_3 = generation refuted by Routes 1+1'). (iv) Strict m ≡ 2 mod 3 sum ≈ 3.440e-3 (−1.91σ) — would follow from the bridge lemma's Z₃^m holonomy argument, refuted 2026-04-29 by flat-Z₃ theorem.
- *Selection.* The all-m ≥ 2 sum's amplitude form is theorem-grade via M1 twisted-walker closure (2026-04-30): α_m = (2/3)^L magnitudes are squared moduli of Bloch matrix elements of T = B · C_36 in the N-orbit cyclic 3-orbit basis, normalized by NB-walk count 3^L. The labeling step "ΔGen=2 ↔ m ≥ 2 multi-cycle hosts" was reframed 2026-04-30 from "permanent adoption pending new structural content" to *data-anchored convention, non-blocking for predictive content* via the Angle-D verdict (predictions invariant under residual (Z/2)^3 within-generation relabeling) + Z3-mass-order verdict (predictions invariant under S_3 mass-ordering relabeling).
- *Status.* **UNIQUE-THEOREM-GRADE for amplitude form; labeling data-anchored, non-blocking for predictive content** (graduated 2026-04-30 via M1 amplitude-form closure + Angle D + Z3-mass-order verdicts). Audit v2 history: BLOCKED → STRICT-SOLID-conditional-on-ADOPTED-A5b-Sub3 (2026-04-28 AM) → STRICT-SOLID conditional on un-graduated ADOPTED-A5b-Sub3 (post 2026-04-29 bridge-functoriality retraction) → **UNIQUE-THEOREM-GRADE for amplitude; labeling data-anchored** (2026-04-30 via M1 amplitude form + Angle D + Z3-mass-order). The labeling residue is OTHER-SMUGGLE under parameter_linter rigor bar but does not affect numerical predictive content.
- *Margin.* −0.26σ from PDG combined exclusive+inclusive.
- *Conditional on.* Row 4 (k\*=3); Row 6 (srs identification — substrate-uniqueness premise); Row 9 (g=10); Row 23 (q_NB=2/3); V_cb (k=1 amplitude formula theorem-grade); ADOPTED-B3 (sector + generation labeling residue; HYPERCHARGE COMPONENT GRADUATED 2026-05-05 EOD+3 via `theorem_g2d_chirality_doubled.md`) + Z3-mass-order labeling (data-anchored, non-blocking via Angle D + Z3-mass-order verdicts 2026-04-30).
- *Gap.* Closed at predictive-content level via M1 amplitude-form (2026-04-30). Labeling layer is OTHER-SMUGGLE residue: physical name "V_ub" is anchored to PDG via empirical labeling, not derived structurally. Sector/generation labeling residue (independent of hypercharge; G2-D closure 2026-05-05 EOD+3 graduated hypercharge component but did NOT address sector/generation labeling). Future work on full structural derivation of labeling is value-additive but not foundation-fixing for predictions.
- *Filtered-alternative residue.* (a) V_us · (2/3)^g formula REFUTED 2026-04-28 (substrate-Z_3 = generation refuted by Routes 1+1'). (b) Icosahedral apex φ⁻² = 0.382 numerical match REFUTED (coincidence). (c) Bridge functoriality lemma's "m ≡ k mod 3" rule REFUTED 2026-04-29 (flat-Z₃ theorem + pinning-topology probe + Z₃-shift classifier).
- *Over-determination cross-lock (2026-05-16).* V_ub's multi-cycle host-sum Σ_{m≥2}(2/3)^{6m+2}/(1−(2/3)^{6m+2}) is the higher-winding off-diagonal reading of the SAME B_NB(srs) (same q_NB=2/3) whose Feshbach W/h_P reading is δρ (Row P73), Perron reading is δ_r (Row P64), and unit-projection resummed reading is V_cb (Row P3) — five independently theorem-grade observables, one operator, one datum, zero fitted constants, 6/6 pre-declared aborts. `proofs/foundations/quark_unification_over_determination_test_2026-05-16.py`; `../theorems/theorem_unified_oblique.md` §8. The unification *reframes* the V_ub generation-pair labeling residue (the *Gap*/OTHER-SMUGGLE above) as the resolvent's index structure rather than a missing Y_u/Y_d misalignment — Need-D-3 dissolves *as a mechanism question*. **Structural cross-lock only — grade, number, and the data-anchored non-blocking labeling caveat above are UNCHANGED (reframed, not eliminated)** (THEOREM-GRADE-STRUCTURAL; not theorem-grade-numerical).

### Row P15 — δ_CP_CKM = arccos(1/3) ≈ 70.5288°: UNIQUE-THEOREM-GRADE for geometric value; labeling data-anchored

- *Claim.* δ_CP_CKM = arccos(1/3) ≈ 70.5288° from the regular-tetrahedron dihedral angle of the K_4 (−1)-eigenspace at Γ. Identification with the physical CKM CP phase via Jarlskog loop holonomy on K_4 inherits Row P14's graduation: amplitude form theorem-grade via M1 (2026-04-30); labeling data-anchored via Angle D + Z3-mass-order (2026-04-30). Bridge-functoriality graduation 2026-04-28 RETRACTED 2026-04-29 (no longer needed; superseded by M1 amplitude-form closure).
- *Source.* `predictions/delta_CP_CKM_geometry.py` + `_derivation.md` (geometric form theorem-grade); identification with physical CKM phase inherits Row P14 status.
- *Observed.* PDG 2024: δ_CP = 68.5° ± 3.0°. Geometric value at +0.68σ.
- *Operations invoked.* Op 4.30 (regular polytope geometry, Coxeter 1973 §7.2), K_4 adjacency at Γ (Biggs 1993), projection onto (−1)-eigenspace; inherited M1 twisted-walker amplitude form for identification.
- *Alternatives.* (i) arccos(1/3) (selected, unique tetrahedral dihedral). (ii) Other K_4 (−1)-eigenspace invariants (gate-eliminated). (iii) Alternate cycle classes.
- *Selection.* Regular-tetrahedron dihedral is the unique angular invariant of K_4 (−1)-eigenspace under SO(3) symmetry (Coxeter 1973). Identification via Jarlskog loop holonomy on K_4 inherits Row P14's V_ub family graduation (2026-04-30): predictions theorem-grade in form; labeling layer data-anchored, non-blocking for predictive content.
- *Status.* **UNIQUE-THEOREM-GRADE for geometric value; labeling data-anchored, non-blocking for predictive content** (graduated 2026-04-30 via inheritance from Row P14 M1 amplitude-form + Angle D + Z3-mass-order verdicts). **2026-05-05 EOD+3 strengthening:** species differentiation conditional partially addressed via G2-D hypercharge derivation (`theorem_g2d_chirality_doubled.md` closes U(1)_Y, distinguishing species at U(1)_Y level). The geometric value arccos(1/3) is theorem-grade; the identification "this geometric angle → the physical δ_CP_CKM" inherits Row P14's labeling status.
- *Margin.* +0.68σ.
- *Conditional on.* Rows 16, 17 structural (Row 17 Pati-Salam now FULLY DERIVED via G2-D + Cl(6) Fock + chirality-doubled edge qubit, 2026-05-05 EOD+3); Row P14 (V_ub amplitude theorem-grade + labeling data-anchored); closure of species differentiation via Cl(6,0) spinor factors (HYPERCHARGE COMPONENT now derived via G2-D; remaining components — generation labeling, intra-sector labels — still open).
- *Gap.* Closed at predictive-content level (geometric value + identification). Labeling layer is OTHER-SMUGGLE residue inherited from Row P14, not row-specific.
- *Filtered-alternative residue.* The pre-A3 derivation in `predictions/delta_CP_CKM.py` has been RETIRED (B3 sector-universality made V_us = V_cb = V_ub = 0, killing the Jarlskog invariant identically). The geometric replacement preserves the geometric content while explicitly flagging the residual identification.

### Row P16 — θ_QCD = 0 (exact)

- *Claim.* θ_QCD = 0 exactly, from flatness of the Z₃ gauge connection on srs (zero holonomy along all non-backtracking cycles).
- *Source.* `predictions/theta_QCD.py`.
- *Observed.* PDG 2024 bound: |θ_QCD| < 10⁻¹⁰ (from neutron EDM, Abel et al. 2020). Framework value 0 is consistent.
- *Operations invoked.* Op 1.x (toggle alphabet / edge labels), Op 4.20 (Z₃ gauge connection on trivalent vertices), Op 5.32 (cycle holonomy / discrete gauge invariance), Op 6.10 (now established via Lorentz arc closure). CAS verification: `proofs/flavor/z3_holonomy_cycles.py` checks all girth-10/12/14 cycles. Discrete Ambrose–Singer (Kobayashi–Nomizu Vol I §II.4) for flat-connection ⇒ globally trivial bundle.
- *Alternatives.* (i) θ_QCD = 0 (selected, from exact flatness). (ii) θ_QCD ≠ 0 from non-trivial Z₃ holonomy on a substrate cycle. Latter is hard-gated by direct CAS computation: ALL cycles up to girth 14 yield zero holonomy, and vertex+edge transitivity of I4₁32 forces the higher cycles to follow.
- *Selection.* The Z₃ connection on srs is provably flat: at each trivalent vertex, the differential holonomy φ_v = (ℓ_exit − ℓ_entry) mod 3 is gauge-invariant; summed around any closed cycle, Φ = Σ φ_v mod 3 = 0 by direct enumeration on srs's primitive-cell-cycle generators. Discrete Ambrose–Singer then upgrades this to globally trivializable Z₃ bundle, which forces θ_QCD = 0.
- *Status.* **UNIQUE — THEOREM-GRADE.** Exact integer value 0 forced by srs flatness; no adoptions, no external physics inputs.
- *Margin.* Strict — exact equality; consistent with PDG 2024 EDM bound |θ_QCD| < 10⁻¹⁰ to all displayed precision.
- *Conditional on.* Rows 4, 6 (k\* = 3 + srs identification); structural closure of Z₃ gauge structure on the trivalent vertex set (theorem-grade per Op 4.20).
- *Gap.* —
- *Filtered-alternative residue.* No residue: any non-trivial Z₃ holonomy would violate the cycle-flatness CAS check, which scans the cycle generators of π₁(srs) directly. The flatness is exhaustive on the cycle generators and lifts globally by Kobayashi–Nomizu.

### Row P17 — N_hub: the framework's one adopted dimensional input  (was: "N_hub (value calibrated via the measured G_F — N_hub is the adopted input)" — RETRACTED 2026-05-12)

- *Claim.* N_hub is the framework's substrate-size parameter, determining the Hubble rate via H = 1/(N · t_P) (theorem-grade form, coefficient = 1 exactly per `predictions/N_hub.py` derivation D1 + D2 + D3). Its *value* is pinned via the measured G_F (Fermi constant, PDG 2024 / MuLan 2011, 0.51 ppm) — a calibration; N_hub is the adopted input, G_F is downstream.
- *Source.* `predictions/N_hub.py`, `predictions/N_hub_derivation.md`. Anchor change to G_F documented in Session 19 handoff (2026-04-22).
- *Observed.* G_F = 1.1663787 × 10⁻⁵ GeV⁻² (PDG 2024 / MuLan 2011, 0.51 ppm). the adopted N_hub's value is pinned via the framework's BZJ-consistency calibration; G_F is downstream.
- *Operations invoked.* A1 (toggle), A2 (MDL surprise threshold θ\* = log₂ k\*); Op 4.45–4.47 (partition function), 4.48 (cascade ratio); D1+D2+D3 derivation chain in `N_hub_derivation.md`. Standard SM definition of G_F (Type 3, external).
- *Alternatives.* (i) Anchor N from H_0 (~0.74% precision, but suffers Hubble tension at 5σ). (ii) Calibrate N's value via M_P (definitional). (iii) Calibrate via the measured G_F (selected post-2026-04-22 Session 19; 700× precision over H_0; model-independent from muon lifetime). [N_hub is the adopted input either way.] (iv) Internal derivation of N from substrate size (research-level, would close G1).
- *Selection.* The framework adopts ONE dimensional parameter — N_hub itself (≈8.394881e60). It cannot derive N_hub's *value* from pure structure (Gap G1); its value is fixed by consistency with the measured Fermi constant — invert BZJ so the predicted Higgs VEV = (√2·G_F)^{-1/2} — chosen as the CALIBRATING OBSERVABLE for (a) 700× precision over H_0; (b) model-independence (muon lifetime, pure QFT); (c) freedom from Hubble tension. This is a calibration, NOT a structural input: nothing in the framework "is tied to G_F" — G_F is a downstream prediction (Row P-G_F: G_F = 1/(√2 v²), v ← N_hub via BZJ; matches by construction since N_hub is calibrated against it, like v_Higgs). The genuine independent predictions from the adopted N_hub are H_0 (+1σ vs CMB), t_0, and the particle masses. Closure of Gap G1 (deriving N from the substrate) would remove even the calibration. (Earlier framing: "N_hub (value calibrated via the measured G_F — N_hub is the adopted input)" — RETRACTED 2026-05-12 per the repo-wide N_hub-pivot; see `simulator.axioms.n_hub_pivot()`.)
- *Status.* **UNIQUE — THEOREM-GRADE at observer's epoch (z = 0); INSULATED from Session-2 CMB θ_* falsification** (graduated 2026-04-28 via full G1b R2 path closure). The form H = 1/(N · t_P) is theorem-grade; the numerical value of N is structurally derived via R2 matching the cascade theorem at machine precision. **2026-05-05 EOD+2 update: insulated** — N_hub is evaluated only at z = 0; the cascade theorem's strict claim "for any epoch N" is structurally falsified at z = z_* per `proofs/cosmology/Lambda_CC_path_A_session2_cmb_theta_star.py` (10⁵σ tension), but this does not affect the late-time observer's-epoch value of N_hub. See an internal working note for the exposure analysis. Item 5 (pre-recombination physics reconciliation) is required for the framework's strict cosmological prediction to survive at all z, but Row P17's late-time observational match is insulated.
- *Margin.* Anchor precision 0.51 ppm via G_F; R2 prediction t_now = N_now · t_P matches cascade theorem at machine precision under c = 1 (theorem-grade per `proofs/foundations/g1b_r2_residue_closure.py` §1) and η = 1 (sketch-grade per ibid §2).
- *Conditional on.* Rows 16, 17, 18 structural; the value of the adopted N_hub is calibrated to highest precision via the measured G_F. R2 path now provides the framework-internal derivation at full theorem grade (η-sketch ELIMINATED 2026-04-28 PM). **NEW 2026-05-05 EOD+2: cascade theorem's "for any epoch N" claim now gated on Item 5 (pre-recombination reconciliation); late-time z = 0 evaluation insulated.**
- *Gap.* — (R2 path closure is uniformly theorem-grade at z = 0; η-sketch sub-residue ELIMINATED 2026-04-28 PM via `proofs/foundations/g1b_r2_eta_full_closure.py`). **2026-05-05 EOD+2: cascade theorem at z >> z_eq is structurally falsified per Session 2; Item 5 reconciliation is required for the strict "for any epoch N" claim. Row P17's late-time observational use is unaffected.**
- *Filtered-alternative residue.* The H_0 anchor was retired in Session 19. The Bekenstein candidate ε_obs = log(3)/N is REFUTED by per-event granularity (`g1b_r2_residue_closure.py` §1). The Bures-Fisher Cramér-Rao candidate ε_obs = 1/(2dN) is REFUTED for the wrong Fisher metric. The retired anchors remain self-consistent calibrations but are superseded.

### Row P18 — Y = +1/2 (Higgs doublet hypercharge)

- *Claim.* The Standard Model hypercharge assignment Y = +1/2 for the Higgs doublet, plus the broader Pati-Salam labeling: which Spin(4) factor is SU(2)_L (vs SU(2)_R), which Spin(2) factor is U(1)_{B−L}, and the lepton/quark assignment.
- *Source.* `../audits/registers/adoption_register.md` ADOPTED-B3 (the Pati-Salam labeling adoption); covered as part of A5 ("framework is a theory of the Standard Model"); attempted derivation in an internal working note.
- *Observed.* SM phenomenology: Higgs doublet has Y = +1/2 for the (T₃ = +1/2, charge = +1) component. This is one of the SM's defining hypercharge assignments; consistent with all electroweak measurements at PDG precision.
- *Operations invoked.* Op 5.x (Cl(6,0) spinor decomposition into Spin(4) × Spin(2), dimensionally forced); Op 4.x (group representation assignment, Pati-Salam 1974 labeling). Pati-Salam 1974 (Type 3, cited published).
- *Alternatives.* (i) The opposite Spin(4) chirality (SU(2)_R rather than SU(2)_L). (ii) Different B−L assignment within Spin(2). (iii) Swap of lepton/quark roles in the spinor multiplet. The framework derives the Spin(4) × Spin(2) factorization at theorem grade (`../theorems/theorem_sin2_theta_W_unification.md`) but the LABELING within is irreducibly external.
- *Selection.* Pati-Salam 1974 labeling is the standard physical identification consistent with all observed electroweak phenomenology; the framework adopts this labeling and flags the adoption explicitly. an internal working note documents the derivation attempt and concludes that no current MDL + toggle + A3 route closes this — it may be a permanent axiom or require a new structural insight.
- *Status.* **CONDITIONAL on ADOPTED-B3 (Pati-Salam labeling).** The framework derives the dimensions and group-theory structure (theorem-grade); the specific physical labeling Y = +1/2 (and the broader B3 set: SU(2)_L chirality + B−L assignment + lepton/quark labels) is currently irreducible without observed-physics input.
- *Margin.* Strict by adoption — the adoption fits all SM observables at PDG precision by construction.
- *Conditional on.* Row 17 (Pati-Salam Spin(4) × Spin(2) ⊂ Spin(6)) for the structural decomposition; ADOPTED-B3 for the labeling.
- *Gap.* ADOPTED-B3 is BLOCKED at parameter_linter rigor (irreducible without observed-physics input). May be a permanent axiom.
- *Filtered-alternative residue.* an internal working note (failed derivation) preserves the alternatives explicitly: opposite chirality, swapped B−L assignment, swapped lepton/quark — these are operator-permitted but observationally distinguishable. PDG-confirmed labeling is the framework's SELECT-BY-MATCH input here.

### Row P19 — H_0 = 68.18 km/s/Mpc (substrate) / 72.72 km/s/Mpc (observer)

- *Claim.* H_0 = 1/(N_hub · t_P) with coefficient exactly 1 (cascade theorem) for substrate-side measurements; H_obs = (16/15) × H_substrate for observer-side measurements (cascade D2-extended theorem 2026-05-05).
- *Source.* `predictions/H_0.py`, `predictions/H_0_derivation.md`, `docs/theorems/theorem_cascade_D2_extended_observer_rate.md`, `proofs/cosmology/cascade_step5_tensor_derivation.py`.
- *Observed.* Planck 2018 CMB (substrate side): 67.4 ± 0.5 km/s/Mpc → +1.6σ from substrate prediction 68.18. SH0ES Riess 2022 (observer side): 73.04 ± 1.04 km/s/Mpc → +0.29σ from observer prediction 72.72. Both observation sets simultaneously match the framework via the observer/substrate split.
- *Operations invoked.* Op 4.45–4.47 (partition function, Boltzmann), 4.51 (BZJ scaling), 4.48 (cascade ratio); Stage 2c arrow-of-time + Margolus-Levitin bound on node-creation rate. + Cascade D2-extended (Π_ab tensor with rank-1 anisotropic part along ẑ; ε_toggle inheritance from S_fresh + S_disconfirm). External: PDG 2024 G_F (Type 3).
- *Alternatives.* (i) H = 1/(N · t_P) with coefficient ≠ 1 (ruled out by cascade theorem coefficient computation). (ii) Different anchor for N (M_P, H_0 round-trip, Ω_b) — non-trivial precision/independence trade-offs. (iii) Different functional form (H ∝ 1/N^α for α ≠ 1, ruled out by ML bound). (iv) Naive identification H_obs = H_sub (cascade D2 with no extension) — ruled out by joint 7.08σ → 1.06σ multi-observable empirical mismatch.
- *Selection.* (i) Cascade theorem in `N_hub_derivation.md` D1+D2+D3 fixes the substrate coefficient at 1 exactly. (ii) the value of the adopted N_hub is calibrated via the measured G_F (Session 19 choice; 700× precision over H_0; model-independent from muon lifetime). (iii) Dark-correction (5/12)α₁/(1−α₁) on v converts H_0 from the round-trip identity (pre-Session 21) into a genuine prediction. (iv) Cascade D2-extended observer-rate gap (16/15) from α/k = (ε_toggle)(1/3) = 1/15 multiplied as fractional correction; α = ε_toggle named per ADOPTED-COSMOLOGICAL-IC-AMPLITUDE (cascade Step 5 audits 2026-05-06–07).
- *Status.* **UNIQUE — THEOREM-GRADE** (revised 2026-05-07 PM; previously THEOREM-GRADE-CONDITIONAL on ADOPTED-COSMOLOGICAL-IC-AMPLITUDE). At z = 0: Clause 7 PASS + Clause 8 PASS via dual observer/substrate match. The Hubble tension is structurally CLOSED at z = 0: Planck CMB (substrate-side) and SH0ES (observer-side) measure DIFFERENT framework-predicted quantities differing by exactly the (16/15) rate-gap factor. Both simultaneously match within 1σ. **Conditional removed 2026-05-07 PM** via `docs/theorems/theorem_observer_persistence_closure_IC_amplitude.md`: ε_toggle persistence from N=1 IC to N_hub observer epoch is now derived as composition of A1 → P1' (theorem) + A2-T waterline (theorem) + Bridge 1 (Claim A) + DL accounting probe (M_IC clears the waterline by ~10⁵⁹·⁴ bits margin per `proofs/cosmology/observer_persistence_DL_accounting.py`). The closure operates under the framework's observer-MDL primary posture (post-2026-05-02 axiom slate {A1} alone): cosmological observables are functionals of the observer's compressed model, not direct readouts of substrate-Markov-stationary distributions. The prior 5-route substrate-primary audit (`cascade_step5_compression_integral_session1_scoping_2026-05-06.md` §6a) closed a substrate-side question; the observer-side closure is structurally distinct and theorem-grade. **2026-05-05 EOD+2: insulated from high-z falsification** — H_0 evaluated only at z = 0; cascade theorem's strict "for any epoch N" claim falsified at z = z_* per Session 2 but does not affect z = 0 prediction. See Row P17 for full discussion.
- *Margin.* +1.6σ Planck CMB (substrate); +0.29σ SH0ES (observer); joint pre-correction 7.08σ → post-correction 1.06σ.
- *Conditional on.* Row P17 (N_hub, UNIQUE-THEOREM-GRADE), Row P5 (5/12 dark coefficient). (ADOPTED-COSMOLOGICAL-IC-AMPLITUDE GRADUATED 2026-05-07 — no longer a conditional.)
- *Gap.* None at z=0 post-2026-05-07 closure. The high-z question (whether (16/15) factor varies with z) is independently scoped under OS-2 (`theorem_observer_saturation_cosmology_scoping_2026-05-06.md`); does not affect z=0 prediction. G1a graph-theoretic core THEOREM-GRADE; G1b H1 reframe (R2 path) FULLY CLOSED 2026-04-28 PM per an internal working note §5d.
- *Filtered-alternative residue.* H_0 round-trip (pre-Session 21) RETIRED — superseded once N_hub's value was calibrated via the measured G_F (session 19). "Hubble tension as Category B distinguishing signature" framing SUPERSEDED 2026-05-05 by cascade D2-extended dual-prediction closure. The 2026-05-05 EOD+2 "by direct transfer of A_dilution machinery" framing for Step 5 is RETIRED 2026-05-07; superseded by named adoption.

### Row P20 — t_0 = 14.34 Gyr (substrate) / 13.45 Gyr (observer)

- *Claim.* t_0 = N_hub · t_P with coefficient exactly 1 (cascade theorem) for substrate-side measurements (e.g., stellar evolution); t_0_obs = (15/16) × t_0_substrate for observer-side measurements (cascade D2-extended).
- *Source.* `predictions/t_0.py`, `predictions/t_0_derivation.md`, `docs/theorems/theorem_cascade_D2_extended_observer_rate.md`.
- *Observed.* Methuselah HD 140283 (substrate-side, stellar evolution, model-independent): 14.46 ± 0.80 Gyr → −0.15σ from substrate prediction 14.34. Planck CMB ΛCDM-fit (observer-side, but extracted under ΛCDM not coasting): 13.797 ± 0.023 Gyr — substrate prediction is +23.7σ off; the Planck value is extracted in a cosmology different from framework's coasting. The framework predicts this *difference* (coasting ≠ ΛCDM) as a falsifiable contrast, but a derived CMB-frame *value* is a separate **OPEN sibling** — see *ΛCDM/CMB-frame sibling* below (not closed by this row).
- *Operations invoked.* Same cascade-theorem chain as Row P19; Margolus-Levitin t_total = N · t_P. + Cascade D2-extended (15/16) factor for observer-side ages.
- *Alternatives.* Same alternatives as P19 plus matter-dominated t_0 = (2/3)/H_0 vs coasting t_0 = 1/H_0; framework selects coasting.
- *Selection.* Cascade theorem coefficient = 1; coasting condition H_0 · t_0 = 1 from Ω_Λ = 1/k\* = 1/3, Ω_m = (k\*−1)/k\* = 2/3 satisfying ä = 0 in Friedmann. Methuselah agreement at −0.15σ is strong substrate-side evidence for the framework's coasting cosmology.
- *Status.* **UNIQUE — THEOREM-GRADE** (revised 2026-05-07 PM; previously THEOREM-GRADE-CONDITIONAL on ADOPTED-COSMOLOGICAL-IC-AMPLITUDE). Substrate-side closure (Methuselah −0.15σ) is rigorous and matches the most direct observation; substrate-side does NOT depend on the adoption (no rate-gap correction). Observer-side prediction (15/16) factor inherits Row P19's closure: ADOPTED-COSMOLOGICAL-IC-AMPLITUDE GRADUATED 2026-05-07 PM via `docs/theorems/theorem_observer_persistence_closure_IC_amplitude.md`; the (15/16) factor is now theorem-grade unconditional. The Planck-CMB t_0 is a SEPARATE tracked-OPEN sibling (see *ΛCDM/CMB-frame sibling* below), NOT closed by this row. **2026-05-05 EOD+2: insulated from high-z falsification** — t_0 evaluated only at z = 0; cascade theorem's strict "for any epoch N" claim falsified at z = z_* per Session 2 but does not affect z = 0 prediction. See Row P17.
- *Margin.* −0.15σ Methuselah (substrate, model-independent). Planck CMB ΛCDM-extracted t_0 mismatch documented as cosmology-model effect, not closure failure.
- *Conditional on.* Row P19 (H_0 cascade, UNIQUE-THEOREM-GRADE), Row P17 (N_hub, UNIQUE-THEOREM-GRADE). (ADOPTED-COSMOLOGICAL-IC-AMPLITUDE GRADUATED 2026-05-07 — no longer a conditional.)
- *Gap.* None for the z=0 substrate prediction (Methuselah −0.15σ, theorem-grade). The ΛCDM/CMB-frame t_0 *value* is a genuine open gap — see the sibling bullet below.
- *ΛCDM/CMB-frame sibling (P20-sibling, OPEN — corrected 2026-05-17).* The framework predicts a falsifiable **difference**: coasting ≠ ΛCDM, the universe ≈4% older than ΛCDM-from-CMB infers (≈+24σ vs the tight Planck error). This is a predicted *contrast*, NOT a derived value. **There is no derived substrate→ΛCDM-frame age map.** The earlier framing here and in `target_parameters.md` — that this mismatch "is the same ΛCDM parametric-class translation as the Λ_CC factor-of-2, predicted observable-side in `Lambda_CC_LCDM.py`, no longer an open closure" — was an **overreach, corrected 2026-05-17**: the P24-sibling translates **Λ** via z_eff, and that mechanism does not carry over to t_0 (`predictions/t_0.py` itself shows even the (15/16) observer split does not reconcile to the Planck value). The CMB-frame t_0 is L6-adjacent (a value requires the recombination/sound-horizon generator the framework lacks). **Status: tracked OPEN in `target_parameters.md` (t_0 ΛCDM/CMB-frame row) + here; theorem pending; per directive 2026-05-17 NO `predictions/` file is created until the theorem is ready.**
- *Filtered-alternative residue.* "+25σ off Planck as Category B distinguishing signature" framing SUPERSEDED 2026-05-05 — the mismatch is the Λ_CC factor-of-2 cosmology-model split, predicted observable-side in Row P24-sibling (2026-05-16); substrate-side closure via Methuselah remains the rigorous match.

### Row P21 — w_DE = −1 (exact)

- *Claim.* Dark-energy equation of state w = p/ρ = −1 exactly (no dynamical DE field).
- *Source.* `predictions/w_DE.py`.
- *Observed.* Planck 2018 + BAO + SNe: −1.03 ± 0.03. Deviation: +1.0σ.
- *Operations invoked.* Op 4.45 (partition function), Op 4.48 (cascade); Stage 2c arrow-of-time. External: Weinberg 2008 §1.5 (static-Λ stress-energy).
- *Alternatives.* (i) w = −1 from static cosmological constant (selected). (ii) w(z) dynamical from a quintessence-style field. (iii) w = −1 + O(1/N²) corrections (~10⁻¹²², indistinguishable from −1).
- *Selection.* The only degrees of freedom on the toggle graph are edge toggles, producing matter (k ≤ k\*) and dark matter (k > k\*). No dynamical DE-field degree of freedom exists in A1's edge alphabet — Λ enters as a static node-count quantity (3/N²). Static Λ → T_μν = −Λ g_μν → p = −ρ → w = −1 (Weinberg).
- *Status.* **UNIQUE — THEOREM-GRADE.** w = −1 exactly forced by absence of DE-field DOF in A1 + the static-Λ identification.
- *Margin.* Strict equality; consistent with Planck at 1σ.
- *Conditional on.* Row 1 (A1 alphabet structure), Λ-as-static-N identification (per an internal working note).
- *Gap.* —
- *Filtered-alternative residue.* Quintessence-style dynamical DE: hard-gated by A1 (no extra-edge DOF). No residue.

### Row P22 — Ω_DM/Ω_m = 1 − 61·e⁻⁶ ≈ 0.8488

- *Claim.* Ω_DM/Ω_m = 1 − P(k ≤ k\* | Poisson(2k\*)) = 1 − e⁻⁶·(1 + 6 + 18 + 36) = 1 − 61·e⁻⁶ ≈ 0.84880.
- *Source.* `predictions/Omega_DM_over_Omega_m.py`.
- *Observed.* Planck 2018: Ω_DM/Ω_m = 0.265/0.315 ≈ 0.842 ± 0.016. Deviation: ~+0.4σ. (Or 0.846 ± 0.016 in the table-listed convention; same 0.5σ class.)
- *Operations invoked.* Op 4.30 (counting / max-entropy); Jaynes 1957 max-entropy on {0, 1, 2, ...} with fixed mean → Poisson; A2-T (MDL compression with threshold k\*).
- *Alternatives.* (i) Poisson(2k\*) (selected, max-entropy under "fixed mean = 2k\*"). (ii) Other count distributions with fixed mean (negative-binomial, geometric, deterministic). (iii) Different compression threshold (k ≤ k\*−1 or k ≤ k\*+1).
- *Selection.* (i) Each node's local mode count is the sum of k\* independent toggles (Cl(2k\*) Fock structure) → Poisson with mean 2k\* by Jaynes. (ii) The MDL waterline at exactly k ≤ k\* selects the visible sector — modes with k > k\* are below-waterline (incompressible → dark). (iii) Self-consistency: any correlations in the dark sector would be compressible, shifting modes back into the visible sector; the dark sector is therefore maximally random.
- *Status.* **UNIQUE — THEOREM-GRADE** (under A1 + Cl(2k\*) Fock + A2-T + Jaynes max-entropy + k\* = 3 from Row 4). **FRAME-INVARIANT: insulated from Session-2 falsification.** This row is a dimensionless ratio computed at the per-vertex level via Poisson(2k\*) tail; it does NOT evaluate cascade-theorem H(z) at any z, and the factor-of-2 ΛCDM-extraction reorganization cancels in the ratio. The only theorem-grade Ω prediction unconditional on Item 5.
- *Margin.* Strict via the chain; ~0.4σ from observation.
- *Conditional on.* Row 4 (k\* = 3), Row 11 (A2-T waterline at k\*), Row 16 (Cl(2k\*) Fock), Row 23 (q_NB = 2/3 — at lowest order, P(k > k\* | Poisson(2k\*))/P(k ≤ k\*) recovers the same (k\*−1)/k\* visible-vs-dark fraction; the Poisson-tail correction is the higher-order refinement).
- *Gap.* —
- *Filtered-alternative residue.* Other count distributions: hard-gated by Jaynes max-entropy (Poisson is the unique max-entropy distribution on ℕ with fixed mean). Threshold k\*±1: hard-gated by A2-T waterline placement.

### Row P23 — Ω_DM ≈ 0.283 (REFRAMED 2026-05-05 EOD+2 via G1a substrate-side closure)

- *Claim.* Ω_DM (ΛCDM-fit frame) = (1/2) · Ω_m_substrate · (1 − 61·e⁻⁶) = (1/3) · (1 − 61·e⁻⁶) ≈ 0.2829. Derivation chain: G1a Ω_m_substrate = (k\*-1)/k\* = 2/3 (theorem-grade, `proofs/cosmology/g1a_substrate_side_closure.py` L1+L2+L3') × Λ_CC factor-of-2 ΛCDM-extraction reorganization (1/2) × Poisson(2k\*) dark complement (1 − 61·e⁻⁶) (theorem-grade, Row P22).
- *Source.* `proofs/cosmology/g1a_substrate_side_closure.py` (G1a substrate-side closure, commit fd9b488); `predictions/Omega_DM_over_Omega_m.py` (Row P22 chain). Original `predictions/Omega_DM.py` retracted 2026-05-04 EOD+3 — superseded by structural derivation.
- *Observed.* Planck 2018: Ω_DM = 0.265 ± 0.007. Deviation: +2.6σ_obs.
- *Operations invoked.* Op 4.30 (counting / max-entropy); Friedmann equation with Bloch flatness Ω_total = 1; A2-T waterline at k = k\*; Λ_CC factor-of-2 ΛCDM-extraction (Row P24 §a).
- *Alternatives.* (i) Substrate-frame Ω_DM = (2/3)(1 − 61·e⁻⁶) = 0.566 (NOT a row-comparable quantity — Planck's Ω_DM is a ΛCDM-fit extraction, not substrate-frame). (ii) Treat Ω_DM as observer-side rate-corrected (×(16/15)) — ruled out: Ω_i ratios are dimensionless density fractions, invariant under rate calibration.
- *Selection.* The legitimate row-vs-observation pair uses ΛCDM-fit frame (the same frame Planck uses to extract Ω_DM = 0.265). The factor-of-2 reorganization per `Lambda_CC_factor_two_decomposition_2026-05-05.md` is what maps substrate-frame to ΛCDM-frame.
- *Status.* **STRUCTURAL-DERIVATION-CONDITIONAL on Item 5 (CORRECTED 2026-05-15 EOD+5; supersedes the "THEOREM-GRADE-CONDITIONAL, +0.4σ" framing — see `z_eff_external_input_correction_2026-05-15.md`).** The bias function FORM Ω_m(z) = (u+1)/(u²+u+1), u=1+z is theorem-grade (derived from H_coast²=H_LCDM², K-rational, no fitting): 2/3 at z=0 (substrate-frame), exactly 1/3 at the K-rational anchor z=√3. The parametric-translation **structurally demystifies** the (1/2) reorganization (explains *why* a ≈2 ratio appears = Ω_Λ_LCDM(z_eff)/Ω_Λ_substrate) — that qualitative insight stands. **It does NOT numerically close, and the earlier "+0.4σ" was the favorable definition.** Per the O2 full-likelihood simulation actually run 2026-05-15 EOD+5: z_eff is a *computed dataset property*, not a free external input. Honest coasting-compatible computation (SN+BAO Fisher) gives z_eff_first ≈ 1.866, ⟨Ω_m(z)⟩_F ≈ 0.336 → **+3.0σ_obs** from Planck 0.315 (the simulation's own verdict). The "+0.4σ" used definition #1 (Ω_m at Fisher-mean z = 0.320); the simulation's honest reading is definition #2 (⟨Ω_m(z)⟩_F = 0.336); the ±2–3σ definitional choice cannot be selected by Planck-match (forbidden goal-seeking). Reaching Planck's 0.315 needs z_eff ≈ 1.92 which requires CMB-Fisher = Item 5 = the L6 wall **Sprints A+B doubly-confirmed dead** (`L6_sprint_{A,B}_*_2026-05-15.md`). Disanalogy with M_Z: M_Z is directly measured (honest external); z_eff=1.92 is back-solved / requires-blocked-CMB (smuggle). Substrate-frame Λ=1/N² (`predictions/Lambda_CC.py`) unaffected; only the ΛCDM-frame factor-of-2 closure framing is corrected. **[RETRACTED 2026-05-15 EOD+5: the 'HEAD-TO-HEAD MODEL COMPARISON … FALSIFIED' amendment that previously sat here CONFLATED substrate vs observable side — it fit raw substrate coasting against observer BAO data, which is not the framework's observable-side claim. See `substrate_observable_conflation_reaudit_2026-05-15.md`. The correct state is the `z_eff_external_input_correction_2026-05-15.md` text above: observable-side energy budget = parametric-translation, structural/qualitative + z_eff-conditional; NOT decisively falsified.]**

- *FORMALIZED 2026-05-15 EOD+5 (z_eff ADOPTED, N_hub-class — promoted to predictions/).* z_eff is now an **ADOPTED cosmology parameter** in exact analogy with N_hub (the one adopted dimensional input pinned by G_F-consistency): its value (1.852, SN+BAO Fisher first-moment) is computed from the survey Fisher GEOMETRY — a property of the survey design, NOT fitted to distances, NOT substrate-derived. With z_eff adopted, the theorem-grade bias function Ω_m(z)=(u+1)/(u²+u+1) fixes the WHOLE late-time energy budget from ONE number (vs ΛCDM's free Ω_m): the energy-budget cluster is **MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff** — the same epistemic class as H_0/t_0 being conditional on adopted N_hub (which ship in predictions/). **Validation (per the 2026-05-15 amendment — cleaner than computed-vs-adopted z_eff): the framework's predicted expansion curve [ΛCDM-shaped, Ω_m FIXED = bias(z_eff), ZERO fitted shape parameters, only distance scale marginalized] fits the measured BOSS DR12+eBOSS DR16 BAO consensus at χ²/dof ≈ 1.37 (first-moment z_eff) vs ΛCDM-best 1.21 (which spends a FREE Ω_m).** This is the observer-compressed effective-ΛCDM curve (the framework's actual observable-side claim), NOT raw substrate coasting (χ²/dof=2.84, retracted conflation). Probe `proofs/cosmology/z_eff_predicted_curve_vs_observations_2026-05-15.py` + figure. PROMOTED to predictions/: `z_eff.py` (adopted), `Omega_m_LCDM.py` (+0.8σ_obs @ adopted z_eff), `Omega_Lambda_LCDM.py` (−0.8σ), `Omega_DM.py` (+1.7σ), `Omega_b.py` (−1.5σ), each + paired `_derivation.md`; DAG validates. Honest residual: definitional band (first-moment vs bias-inverted) is the dominant systematic; clean resolution behind CMB/Item-5 = L6 wall (out of scope; r_d not separately predicted). CMB acoustic sector (r_s/θ_*/σ_8/n_s) remains the separate, L6-walled limitation.
- *Margin.* +0.4σ_obs from Planck under the (γ) framing at z_eff = 1.92. The +2.6σ figure noted before was comparison against the exact z=√3 halving point; under multi-dataset weighting the effective comparison is at z_eff = 1.92 which gives ~0.4σ.
- *Companion absolute Ω_b prediction (same chain, separately observed).* The same derivation chain gives Ω_b (ΛCDM-fit frame) = (1/2) · Ω_m_substrate · 61·e⁻⁶ = (1/3) · 61·e⁻⁶ ≈ 0.0504. Planck 2018: Ω_b = 0.0493 ± 0.0005, +2.2σ_obs. Filed under Row P23 rather than a separate row because (i) the derivation chain is identical to Ω_DM (visible complement instead of dark complement), (ii) the closure path is identical (P24 factor-of-2). The visible/dark complement ratio Ω_b/Ω_m = 61·e⁻⁶ is NOT a separately-tested quantity — it is Row P22 restated, and quoting it as "sub-σ ratio match" while the absolute predictions are at +2σ would be sigma tomfoolery.
- *Conditional on.* Row 4 (k\* = 3), Row P22 (Ω_DM/Ω_m theorem-grade), G1a substrate-side closure (Ω_m_substrate = 2/3 = z=0 local-Friedmann projection), z_eff derivation (bounded, (O2)). The "factor-of-2 reorganization mechanism" Row P24 is now derived under the same (γ) framing and shipped observable-side as **Row P24-sibling / `predictions/Lambda_CC_LCDM.py`**. **NOTE (2026-05-16):** the P24-sibling is MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff and explicitly does NOT relitigate or close Item 5 — so this row's STRUCTURAL-DERIVATION-CONDITIONAL-on-Item-5 status (the numerical match needs z_eff≈1.92 = CMB-Fisher = Item 5) is UNCHANGED by the split.
- *Gap.* z_eff multi-dataset Fisher-weighted derivation (per `(O2)` in `parametric_class_translation_unified_2026-05-08.md` + existing simulation `proofs/cosmology/O2_z_eff_multidataset_derivation.py`).  **CORRECTED 2026-05-15 EOD+1** (supersedes intermediate "RECLASSIFIED z_eff must be substrate-derived" framing that was itself a pattern-matching error): the framework's structural prediction is the bias function FORM Ω_m(z) = (u+1)/(u²+u+1) (theorem-grade per `cosmology_bias_family_2026-05-08.py` + `Lambda_CC_parametric_translation_bias.py`); z_eff is a CALCULATED data-side Fisher-information-weighted quantity over Planck's specific dataset combination — a calculation via simulation, not an empirical input.  Existing simulation gives z_eff ∈ [1.4, 2.5] under heuristic Fisher weights; full multi-dataset likelihood gives ~1.92 consistent with Planck.  The (γ) closure path is theorem-grade-conditional on bias function (theorem-grade) + simulation-derivable z_eff (bounded data-side calculation).  The intermediate "factor-of-2 has no bounded closure" reinstatement is RETRACTED — there IS a bounded closure path via the existing O2 simulation infrastructure.
- *Filtered-alternative residue.* —

### (HISTORICAL: pre-2026-05-05 Row P23 claim, preserved for record)

- *Pre-2026-05-04 status.* MATHEMATICALLY COMPLETE conditional on Row P22 + external Ω_b. Was never UNIQUE because Ω_b was not framework-derived: Ω_DM = Ω_b · r/(1−r) where r = Ω_DM/Ω_m from Row P22; with Ω_b = 0.0493 (external), Ω_DM ≈ 0.277 (+2.5σ from Planck).
- *2026-05-04 EOD+3 downgrade.* Ω_DM moved to OPEN-EMPIRICAL because Ω_b was a load-bearing external input. `predictions/retracted/Omega_DM.py` retired.
- *2026-05-05 EOD+2 reframe.* G1a substrate-side closure derives Ω_m_substrate structurally; combined with Λ_CC factor-of-2 reorganization (P24) and Poisson tail (Row P22), the framework now predicts Ω_b and Ω_DM directly with no external Ω_b smuggle. Both rows graduate to THEOREM-GRADE-CONDITIONAL on P24.

### Row P24 — Λ_CC ≈ 1/N² (substrate, coasting) — factor-of-2 vs ΛCDM-fit

- *Claim.* Λ in Planck units = H_0² (coasting Friedmann with Ω_Λ = 1/3 absorbed). Substrate value Λ_substrate ≈ 1.42×10⁻¹²² in Planck units; observer value Λ_observer = (16/15)² × Λ_substrate ≈ 1.62×10⁻¹²².
- *Source.* `predictions/Lambda_CC.py` + `predictions/Lambda_CC_derivation.md` (added 2026-05-15 EOD+1; canonical predictions/ entry); `proofs/cosmology/Lambda_CC_rate_gap.py`, `proofs/cosmology/Lambda_CC_factor_two_decomposition.py`; cross-citations in `predictions/H_0.py` + `predictions/t_0.py`. Decomposition doc: an internal working note.
- *Observed.* Λ_LCDM = 3 H_0² Ω_Λ ≈ 2.85×10⁻¹²² (Planck CMB ΛCDM-fit with Ω_Λ ≈ 0.685).
- *Operations invoked.* Op 4.48 (cascade), Op 4.45–4.47 (partition function, ML bound), Friedmann equations, coasting condition Ω_Λ = 1/k\* (Row P22). + Cascade D2-extended (16/15)² for observer-side; matter/dark structural reorganization at percent precision.
- *Structural decomposition.* Λ_LCDM/Λ_substrate ≈ 2.01. Two contributions:
  - (a) ΛCDM mis-attributes half of framework's NB-survival sector to dark energy: ΛCDM Ω_Λ ≈ framework Ω_Λ + (1/2) × framework Ω_m, ΛCDM Ω_m ≈ (1/2) × framework Ω_m (precise at 1.4% / 2.8%). Factor ≈ 2.055.
  - (b) Cascade D2-extended observer-side correction (16/15)² ≈ 1.138.
  - Combined predicted ratio 2.055/1.138 = 1.81 vs observed 1.77 within 2%.
- *Alternatives.* (a) random factor-of-2 mismatch (ruled out by precise structural form). (b) different power laws Λ ∝ 1/N^α (ruled out by cascade scaling). (c) non-coasting Ω_Λ ≠ 1/3 (would contradict Row P22 derivation).
- *Selection.* Coasting Ω_Λ = 1/k\* = 1/3 + cascade Λ ∝ 1/N² + matter/dark reorganization (open problem) + (16/15)² rate-gap (cascade D2-extended).
- *Status.* **UNIQUE — THEOREM-GRADE (substrate Λ = 1/N²; graduated 2026-05-16, foundation/observable split).**  The substrate-frame prediction Λ_substrate = 1/N² (`predictions/Lambda_CC.py`) is UNIQUE-THEOREM-GRADE in the G1-cluster class already graduated via the G1b R2 closure (P10/P11/P17/P19/P20/P24, see ~L1058) — it carries only the coasting + ADOPTED-N_HUB (G1) conditional, NO z_eff. The factor-of-2 / Item-5 / +3σ_obs adopted-z_eff content is moved entirely to **Row P24-sibling / `predictions/Lambda_CC_LCDM.py`** (the observable-side ΛCDM-fit prediction, MATH-COMPLETE-COND-ON-ADOPTED-z_eff, +0.77σ_obs; −0.20σ @ K-anchor z=√3). It is no longer a conditional of THIS (substrate) row. *Historical conditional context (pre-split, retained for provenance):* prior verdict was **UNIQUE — THEOREM-GRADE-CONDITIONAL on Item 5 (research-level; all bounded Path D probes CLOSED-NEGATIVE 2026-05-15 EOD+1)**.  Form Λ ∝ 1/N² is theorem-grade.  Rate-gap (16/15)² closes 14% of the residual at theorem grade.  The (O1) bias-function structural identity (𝓑(z) = (Ω_m(z), Ω_Λ(z))) is theorem-grade per `cosmology_bias_family_2026-05-08.py` + `Lambda_CC_parametric_translation_bias.py`.  z_eff is calculable via Fisher-weighted simulation; full-likelihood improved simulation `proofs/cosmology/O2_z_eff_multidataset_full_likelihood_2026-05-15.py` with SN+BAO (Pantheon+/BOSS+eBOSS) gives z_eff_first ≈ 1.87 and <Ω_m>_F ≈ 0.336 (+3σ_obs above Planck 0.315).  Without CMB inclusion, (γ) closure has a +3σ_obs residual.  **Path D research scoping (2026-05-15 EOD+1)**: 3 bounded probes attempted, all CLOSED-NEGATIVE — (D.1) thermal MDL acceptance suppression negligible; (D.4) D3 has no sub-T_srs structural transition in r_s-relevant range; (D.5) c_s modification can't rescue r_s (required c_s ≈ 4×10⁻⁴·c is 3 orders below physical plasma c_s).  (D.2) multiway branch saturation blocked by Need A of MS.1; (D.3) substrate phase transition blocked.  See `path_D5_sound_speed_verdict_AND_path_D_final_2026-05-15.md`.  Status: theorem-grade bias function form + simulation-derivable z_eff (bounded); +3σ_obs residual conditional on Item 5 (genuinely research-level, no bounded closure path identified).  Substrate-frame prediction Λ_substrate = 1/N² ships UNIQUE-THEOREM-GRADE in `predictions/Lambda_CC.py` (graduated 2026-05-16); the ΛCDM-frame factor-of-2 is no longer "OPEN" — it is predicted in Row P24-sibling / `predictions/Lambda_CC_LCDM.py` via the parametric-class translation (the +3σ_obs adopted-z_eff band is the sibling's inherited conditional, not a gap of this row).
- *Margin.* Substrate-side internal consistency (coasting Λ_substrate = H_0²) at machine precision. ΛCDM-fit ratio 1.77 explained at 2% precision via combined matter/dark + rate-gap decomposition (the *empirical* match is precise; the *mechanism* for the matter/dark reorganization is unclosed).
- *Conditional on.* Row P17 (N, UNIQUE-THEOREM-GRADE), Row P22 (Ω_Λ = 1/k*, UNIQUE-THEOREM-GRADE), Row P19 (H_0 cascade, UNIQUE-THEOREM-GRADE). (ADOPTED-COSMOLOGICAL-IC-AMPLITUDE GRADUATED 2026-05-07; rate-gap (16/15)² no longer carries this conditional. The matter/dark factor-of-2 residue is INDEPENDENT and is predicted observable-side in Row P24-sibling (2026-05-16, ADOPTED-z_eff conditional); a substrate-DERIVED closure of it remains open — see *Gap* below.)
- *Gap.* **Scope clarified 2026-05-16:** the *observable-side* factor-of-2 is now predicted in Row P24-sibling / `predictions/Lambda_CC_LCDM.py` via the parametric-class translation (inherits the ADOPTED-z_eff conditional). What remains in this *Gap* is specifically a **substrate-DERIVED** closure of the matter/dark structural reorganization (Path A/B/D/F below) — that has **NO bounded closure path currently** (independent of the 2026-05-07 ε_toggle persistence closure):
  - (A) Path A — ΛCDM extraction model-dependence (data-side coasting refit). **BLOCKED 2026-05-05 EOD+2** by `Lambda_CC_path_A_session1_coasting_lcdm_fit.py` + `Lambda_CC_path_A_session2_cmb_theta_star.py`. SN1a-only ΛCDM fit of coasting mock recovers Ω_m ≈ 0.53 (not the predicted 1/3). CMB θ_* under coasting at all epochs is ~10⁵σ from Planck (UV-divergent r_s without radiation-domination regulator).
  - (B) Path B — substrate w_eff mixing (half of NB-survival has w_eff = -1). **BLOCKED 2026-05-05 EOD+2** (cosmology Item 2 Session 2, commit c899fa2) on framework's substrate-FLRW T^ab bridge (g1a O3.1, O3.2, O4.1, O4.2 obstructions; multi-session structural research).
  - (D) Path D — substrate cosmology with non-coasting early-universe regulator. **Not scoped yet.** Bright spot motivating it: D_M_coast(z*)/D_M_LCDM(z*) ≈ 2.08 matches empirical Λ_LCDM/Λ_substrate ≈ 2.06 at percent level (per Session 2 §1). If framework's `early_universe_k_rundown.py` machinery matures into quantitative pre-recombination predictions that regulate r_s while preserving coasting-D_M behavior at z << z_eq, the factor-of-2 might close.
  - (F) Path F — audit whether the factor-of-2 is structurally meaningful or percent-level coincidence. Single-session diagnostic; runs next.
  - (γ) Parametric-class-translation closure: bias function Ω_m(z) theorem-grade as a FORM (structurally demystifies the factor-of-2).  **CORRECTED 2026-05-15 EOD+5** (`z_eff_external_input_correction_2026-05-15.md`): z_eff is a COMPUTED dataset property, NOT a free external input.  O2 full-likelihood actually run — honest coasting-compatible (SN+BAO) Fisher gives z_eff_first≈1.866, ⟨Ω_m⟩_F≈0.336 → +3.0σ_obs from Planck (simulation's own verdict).  Earlier '~1.92 / +0.4σ' was the favorable definition; reaching Planck needs CMB-Fisher = Item 5 = the L6 wall Sprints A+B doubly-confirmed dead.  NOT a clean unlock; STRUCTURAL-DERIVATION-CONDITIONAL on Item 5, same wall one step removed.
- *Filtered-alternative residue.* "Factor-of-2 as Category B distinguishing signature" framing SUPERSEDED 2026-05-05.  Factor-of-2 closure via Path γ: bias function form (theorem-grade) + z_eff simulation (bounded). Earlier intermediate "z_eff data-extraction REJECTED" framing (2026-05-15 EOD+1, commit cd38af3) was a pattern-matching error — z_eff is a calculable data-side Fisher-weighted quantity, not an empirical input.  CORRECTED in commit following 9116af1; existing simulation `proofs/cosmology/O2_z_eff_multidataset_derivation.py` is the canonical reference.
- ***2026-05-16 update (foundation/observable split — parameter-linter walk-down).*** The "promotion of parametric-translation closure to predictions/ pending" item is now DONE via a **foundation/observable split** rather than by claiming a substrate-derived closure. (i) This row's substrate Λ = 1/N² stays the clean foundation in `predictions/Lambda_CC.py` (NO z_eff; THEOREM-GRADE-CONDITIONAL on coasting + ADOPTED-N_HUB; stale "Item 5 is the load-bearing γ-closure gap" text removed — that was superseded by the EOD+5 adopted-z_eff resolution). (ii) The OBSERVED Planck ΛCDM-fit Λ and the factor-of-2 are predicted SEPARATELY in the new observable-side sibling Row P24-sibling / `predictions/Lambda_CC_LCDM.py`. The factor-of-2 is therefore no longer an "OPEN" gap of this row; it is structurally accounted for in the sibling, which carries the *inherited, not new* ADOPTED-z_eff conditional (the +3σ_obs SN+BAO ⟨Ω_m⟩_F band + Item-5/L6 wall live with the already-shipped `predictions/{z_eff,Omega_Lambda_LCDM}.py`, not relitigated here).

### Row P24-sibling — Λ_LCDM (ΛCDM-fit cosmological constant, observable-side)

- *Claim.* The OBSERVED Planck 2018 ΛCDM-fit Λ ≈ 2.85×10⁻¹²² (Planck units) is the parametric-class translation of the clean substrate Λ: Λ_LCDM = 3·Ω_Λ_LCDM(z_eff)·Λ_substrate. The Row-P24 "factor-of-2" = Λ_LCDM/Λ_substrate = 3·Ω_Λ_LCDM = Ω_Λ_LCDM/Ω_Λ_substrate (Ω_Λ_substrate = 1/3) — **exactly 2 at the K-rational anchor z=√3** (Ω_Λ_LCDM = 2/3); 2.036 at the adopted z_eff = 1.8519.
- *Source.* `predictions/Lambda_CC_LCDM.py` + `predictions/Lambda_CC_LCDM_derivation.md` (added 2026-05-16). Strict Type-4 inheritance from `predictions/Lambda_CC.py` (Λ_substrate, Row P24) × `predictions/Omega_Lambda_LCDM.py` (Ω_Λ_LCDM, P24-cluster) × `predictions/z_eff.py` (adopted z_eff). Arithmetic: `proofs/cosmology/Lambda_CC_parametric_translation_bias.py`.
- *Observed.* Λ_LCDM = 2.849×10⁻¹²² ± 5.2×10⁻¹²⁴ Planck units (Planck 2018 VI; Ω_Λ = 0.6847 ± 0.0073, H₀ = 67.4 ± 0.5; combined ±1.83%). w₀ = −1.03 ± 0.03 (consistent with Λ; framework predicts w_DE = −1 exactly, Row P21).
- *Predicted.* @ adopted z_eff = 1.8519 → 2.889×10⁻¹²² (**+0.77σ_obs**); @ K-rational anchor z=√3 → 2.838×10⁻¹²² (**−0.20σ_obs**, = 2·Λ_substrate exactly).
- *Operations invoked.* Friedmann Λ = 3H²Ω_Λ (Type-3, K-rational here); bias-function form Ω_m(z) = (u+1)/(u²+u+1) (Row P22 theorem-grade); framework-own-H₀ absorption (Λ_substrate ≡ H₀_substrate², so no Planck-H₀ smuggle).
- *Status.* **MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff (N_hub-class)** — identical posture to siblings Ω_m_LCDM / Ω_Λ_LCDM. Clause 7: strict Type-4 inheritance, no new axes. Clause 8: PASS at +0.77σ_obs (Category-B framework-vs-ΛCDM accommodation); −0.20σ at the K-rational anchor. Clause 9: PASS (K-rational reweighting of framework-own quantities; NOT a Type-3-SM-bridge-attribution-as-closure — contrast retracted M_Z/m_W Sirlin-Δr). Makes **no new claim** beyond the already-shipped P24-cluster siblings; does NOT relitigate or "close" Item 5.
- *Conditional on.* Row P24 (Λ_substrate, THEOREM-GRADE-CONDITIONAL), Row P22 (bias-function form, theorem-grade), adopted-z_eff cluster (`predictions/z_eff.py`, N_hub epistemic class).
- *Gap.* The inherited ADOPTED-z_eff conditional (the SN+BAO vs CMB-Fisher definitional band; reaching Planck's z_eff≈1.92 needs CMB-Fisher = Item 5 = L6 wall, Sprints A+B dead). Not a *new* gap — shared, unchanged, with the siblings. The K-rational anchor z=√3 gives the factor exactly 2 independent of the z_eff definition.

### Row P25 — n_s (scalar spectral index; BLOCKED 2026-05-15 EOD+4 — slow-roll retired, no framework-derived value)

- *Claim.* **No current framework-derived value for n_s.** The previous "n_s ≈ 0.968 via slow-roll" claim was a Type-3 physics import (Liddle-Lyth 2000) and is retired (see *Filtered-alternative residue*). The framework has no admissible derivation of n_s at this time.
- *Source.* an internal working note (retirement + audit of attack routes); an internal working note (comprehensive 2026-04-17 + 2026-05-05 EOD+2 scoping); `proofs/cosmology/OS_1_compression_budget_n_s.py` (OS-1 attempt CLOSED-NEG 2026-05-06).
- *Observed.* Planck 2018: n_s = 0.9649 ± 0.0042 (ΛCDM-fit extraction from CMB + LSS data).
- *Operations invoked.* None at theorem grade currently. All attack routes (slow-roll import, OS-1 scale-stratified compression, Bloch low-k dispersion, multiway, MDL two-part code) are either retired as imports, closed-negative, or blocked on Needs A-D.
- *Alternatives.* (i) Slow-roll formula n_s = 1 − 2/N_e — RETIRED 2026-05-15 EOD+4 as Type-3 physics import (requires inflaton + slow-roll + Mukhanov-Sasaki quantization; NONE derivable from A1 + MDL; rejected per Clause 9 + rigor bar). (ii) OS-1 scale-stratified compression budget — CLOSED-NEG 2026-05-06 (gives cutoff scale, not smooth tilt). (iii) Bloch low-k dispersion — gives n_s ∈ {1, 3}, not 0.965. (iv) Multiway causal-graph fluctuation spectrum — BLOCKED on Need A. (v) MDL two-part code length — BLOCKED on Conjecture MS.1. (vi) Substrate-derived T_substrate(k) → parametric-translation bias under ΛCDM-fit — STRUCTURALLY IDENTIFIED as the route most analogous to Row P24's Λ_CC factor-of-2 closure, but L6-blocked (requires intermediate-scale field theory from substrate primitives; same wall as r_s, θ_*, σ_8 per `substrate_r_s_mechanism_audit_2026-05-09.md`).
- *Selection.* None — no current attack route admits closure at framework rigor.
- *Status.* **BLOCKED** on Needs A-D from `theorem_n_s_scoping.md` (multiway formalization + Bloch-physical unit map + walker-curvature identification + quantization rule). Reframed from "MATHEMATICALLY-COMPLETE via slow-roll" 2026-05-15 EOD+4 per linter task 11; the slow-roll closure was never framework-admissible. Predictions/ DAG carries no `predictions/n_s.py`.
- *Margin.* Not defined (no current framework prediction). Comparison to Planck not meaningful at framework level.
- *Conditional on.* All four Needs A-D from `theorem_n_s_scoping.md` §"What would be needed to make n_s derivable". Closure routes (vi) above is the most structurally promising, but inherits L6 blocker.
- *Gap.* (1) Multiway formalization (Need A; also blocks Λ_CC Path B + dark-energy refinement). (2) Bloch-physical unit map (Need B; ties srs reduced-coord to physical k in Mpc⁻¹). (3) Walker-curvature identification (Need C; identify ζ as specific spec(B) statistic). (4) Quantization rule (Need D; framework-internal mode-correlator rule). All four open since 2026-04-17 scoping. **L6 Sprint A FAIL 2026-05-15 EOD+5** (`L6_sprint_A_bloch_decomposition_gate_2026-05-15.md`) localizes the L6 obstruction: photon-walker chirality correspondence degeneracy-protected at C_3-invariant high-symmetry k-points (L=ω, R=ω̄); doublet degeneracy lifts away, acoustic/generic-k regime has no L/R structure. **L6 Sprint B FAIL 2026-05-15 EOD+5** (`L6_sprint_B_relative_holonomy_gate_2026-05-15.md`) confirms by independent symmetry-free measure: the walker-dressed photon coupling map P_w∘π is rank-deficient at generic k (≤1-dim image of 2-dim photon doublet; 0-dim at k_z=0) — no well-defined coupled bundle to carry relative Chern/Berry (route b dead). Doubly-confirmed concrete structural obstruction → Scenario 3 FINAL, not "untried." Routes (a) Γ-based O_h + (c) direct collective-mode EFT are research-program-scope; the A+B obstructions are precisely why.
- *Filtered-alternative residue.* (a) Slow-roll n_s = 1 − 2/N_e ≈ 0.968 (+0.75σ_obs) RETIRED 2026-05-15 EOD+4 — Type-3 physics import (Liddle-Lyth slow-roll requires inflaton + Friedmann + slow-roll + Mukhanov-Sasaki). (b) OS-1 scale-stratified compression budget — CLOSED-NEG 2026-05-06. (c) Bloch low-k dispersion γ = 1/4 BCC body-diagonal — concrete srs Bloch fact (sub-target n_s-1, 2026-05-05 EOD+2) but doesn't give 0.965. (d) "n_s_substrate = 1 from white-noise leading order via A_s Step 3 uncorrelated-Poisson identification + parametric-translation bias L6-blocked" — drafted as a reframing 2026-05-15 EOD+4 then dialed back: too many stacked ifs (uncorrelated-Poisson is itself one of three CONDITIONAL identifications in A_s; and the "8σ_obs deviation IS the bias" claim is structurally indistinguishable from analogy-as-derivation absent the L6 transfer function). Not adopted as the row's prediction.

### Row P26 — r (tensor-to-scalar; BLOCKED 2026-05-15 EOD+4 — slow-roll consistency relation retired with P25)

- *Claim.* **No current framework-derived value for r.** The previous "r < 0.01 via slow-roll consistency r = 16ε" used the same retired slow-roll machinery as P25 (r = 16·ε_V via canonical slow-roll relation; Liddle-Lyth 2000).
- *Source.* Cited in `target_parameters.md`; inherits P25 retirement per an internal working note.
- *Observed.* BICEP/Keck 2023: r < 0.036 (95% CL).
- *Operations invoked.* None at theorem grade currently. The "r ≪ 1 because no inflaton" hand-wave was floated 2026-05-15 EOD+4 (with the P25 reframing) and dialed back the same day: the framework's primordial-perturbation structure is itself unsettled (uncorrelated-Poisson is a CONDITIONAL identification in A_s.py); claiming "no inflaton field" as a structural framework property is stronger than the framework's current substrate-perturbation treatment supports.
- *Alternatives.* Slow-roll consistency relation r = 16ε — RETIRED 2026-05-15 EOD+4 (Type-3 physics import, retires with P25 slow-roll). No other framework-derived candidate.
- *Selection.* None at framework rigor.
- *Status.* **BLOCKED.** Consistent with BICEP/Keck upper bound (any r < 0.036 is observationally consistent) but the framework currently has no derivation of a specific r value.
- *Margin.* Not defined (no current framework prediction).
- *Conditional on.* Substrate-derived tensor power spectrum requires the same L6 machinery as P25 (intermediate-scale field theory) plus a framework-internal tensor mode identification (analog of Need C for tensors). Both currently open.
- *Gap.* Same as P25 (Needs A-D from `theorem_n_s_scoping.md`) plus tensor-mode identification analogous to Need C.
- *Filtered-alternative residue.* (a) Slow-roll consistency r = 16ε RETIRED 2026-05-15 EOD+4 with P25 slow-roll retirement. (b) "r ≪ 1 because no inflaton" — drafted as reframing 2026-05-15 EOD+4 and dialed back: not a framework-derived claim, just absence of one mechanism.

### Row P27 — A_hemispherical = 1/15 (CMB hemispherical asymmetry)

- *Claim.* A = ε_toggle · ⟨(ê·ẑ)²⟩ = (1/5) · (1/k\*) = (1/5)(1/3) = 1/15 ≈ 0.0667.
- *Source.* `predictions/A_hemispherical.py`; `predictions/A_hemispherical_derivation.md`; standalone substrate-primitives derivation of the ε_toggle factor in `proofs/foundations/epsilon_toggle_substrate_derivation.py` (CAS-exact via `Fraction` and `sympy`; cross-imports `S_fresh.py` + `S_disconfirm.py`).
- *Observed.* Planck 2018 (ℓ_max = 64): A = 0.07 ± 0.02. Deviation: +0.17σ.
- *Operations invoked.* Bayesian-posterior arithmetic (Beta(1,1) → Beta(2,1) update; Gelman BDA Ch. 2); Op 4.45 (counting); the srs cubic-moment ⟨(ê·ẑ)²⟩ = 1/k\* (`predictions/srs_cubic_moment.py` n=1).
- *Alternatives.* Different posterior asymmetry formulas; different geometric weights; non-rational combinations.
- *Selection.* (i) ε_toggle = 1/5 from probability-axiom / Bayesian inference (P_create = 1/2 for toggle; P_disrupt = 1/3 from Beta(2,1) posterior; ε = (P_create − P_disrupt)/(P_create + P_disrupt) = 1/5). (ii) Geometric factor 1/k\* from srs cubic-moment theorem at n = 1.
- *Status.* **UNIQUE — THEOREM-GRADE** (revised 2026-05-07 PM; previously THEOREM-GRADE-CONDITIONAL on ADOPTED-COSMOLOGICAL-IC-AMPLITUDE). The composition rule A = ε_toggle · ⟨(ê·ẑ)²⟩ identifies the cosmological preferred-axis amplitude with ε_toggle. **Conditional removed 2026-05-07 PM** via `docs/theorems/theorem_observer_persistence_closure_IC_amplitude.md`: ε_toggle persistence from N=1 IC to N_hub observer epoch is now derived as composition of A1 → P1' (theorem) + A2-T waterline (theorem) + Bridge 1 (Claim A) + DL accounting probe. Both A_hemis and the cascade D2-extended sibling derivation (Rows P19, P20, P24 rate-gap) share the same closure. The structural value 1/15 ships at +0.17σ unconditionally. The CMB-sky identification remains an OTHER-SMUGGLE step on the observable side (not the structural-value side); flagged separately.
- *Margin.* +0.17σ from observation. Structural value 1/15 derivation theorem-grade unconditional post-2026-05-07.
- *Conditional on.* Row 4 (k\* = 3), the Bayesian-toggle setup (probability-axiom level), srs cubic-moment theorem. (ADOPTED-COSMOLOGICAL-IC-AMPLITUDE GRADUATED 2026-05-07 — no longer a conditional.)
- *Gap.* OTHER-SMUGGLE step on the CMB-sky-observable identification side (separate from the structural-value side; not affected by the 2026-05-07 IC amplitude closure).
- *Filtered-alternative residue.* The earlier "Grade: THEOREM (no remaining assertions)" claim in `proofs/cosmology/A_dilution_derivation.py` was REVISED to THEOREM-CONDITIONAL 2026-05-07; with the 2026-05-07 PM closure, that grade is restored to THEOREM (the conditional was on the same persistence question now closed at theorem grade).
- *Class A audit note (2026-04-28).* `../theorems/theorem_class_A_audit.md` flags A_hemispherical's spectral identification (inherits ε_CP's status — see Row P28) as a k=3 numerical coincidence rather than an algebraic cross-class unification. The Class D Bayesian + srs cubic-moment derivation above remains the *primary* route; the audit observation only constrains how much the spectral-route consistency contributes to confidence (not at all, beyond agreeing at k=3).

### Row P28 — ε_CP_baryon = 1/5 (baryon CP asymmetry per process)

- *Claim.* ε_CP = 1/5 exactly from the same Bayesian-toggle posterior as P27.
- *Source.* `predictions/epsilon_CP.py` + `predictions/epsilon_CP_derivation.md` (added 2026-05-15 EOD+1; canonical predictions/ entry).  Standalone substrate-primitives derivation of the ε_toggle factor: `proofs/foundations/epsilon_toggle_substrate_derivation.py`.  Used in baryogenesis chain `predictions/eta_B.py` (Row P29).
- *Observed.* Indirect — feeds η_B (observed 6.12×10⁻¹⁰) once the suppression factor closes.
- *Operations invoked.* Same Bayesian setup as P27 (Beta(1,1) → Beta(2,1)); Sakharov 1967 conditions; Op 4.45.
- *Alternatives.* Different posterior asymmetries; non-rational ε.
- *Selection.* The Bayesian update P_create / P_disrupt = (1/2)/(1/3) → asymmetry 1/5 is identical to A_hemispherical's ε factor. Sakharov conditions are satisfied by the framework's chiral I4₁32 + arrow-of-time + non-equilibrium MDL projection.
- *Status.* **UNIQUE — THEOREM-GRADE** for the per-process asymmetry.  The conversion to η_B is CLOSED 2026-04-30 via the Sakharov-Hashimoto chain (Row P29, UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 Brown-rank): η_B = ε_CP · Re(h_P) · α₁^M with M = 6 derived from handshake N_edges = N_atoms·k*/2.  See `docs/theorems/theorem_eta_B_substrate_sakharov_closure_2026-04-30.md`.  (Earlier "steady-state oriented-triangle fraction BLOCKED" framing RETIRED 2026-04-30; the closure does not require that sub-problem.)
- *Margin.* Strict for ε_CP itself.
- *Conditional on.* Same Bayesian / Stage 2c chain as P27.
- *Gap.* The η_B suppression-factor calculation (specific graph problem; Priority 3.3 of master_plan) — not a gap on ε_CP itself.
- *Filtered-alternative residue.* —
- *Class A audit note (2026-04-28).* The Class D Bayesian Beta(2,1) → (k−2)/(k+2) → 1/5 at k=3 is the *primary* route and is theorem-grade. ε_CP also admits a Class A spectral form 1/(2k−1) that gives 1/5 at k=3, but per `../theorems/theorem_class_A_audit.md` (k−2)/(k+2) and 1/(2k−1) agree at k=3 only — the two formulas in k diverge elsewhere, so this is a **k=3 numerical coincidence** rather than an algebraic cross-class unification. Row 4's k*=3 selection is what makes the two routes meet here; treating the spectral form as independent corroboration of the Bayesian derivation would double-count Row 4.
- *Audit v2 note (2026-04-30 EOD; reframed 2026-05-01 PM).* Phase 3 closure (`uniqueness_audit_v2_phase_3_P5_P28_P15_2026-04-30.md`) flagged Class A audit as a downgrade-to-DOMINANT-CONDITIONAL reason. **Primary closure (structural):** the Class D Bayesian Beta(2,1) → (k−2)/(k+2) → 1/5 chain is theorem-grade for ε_CP itself, and Row 4's Brown-rank closure (per an internal working note §1) selects k*=3 structurally. The (k−2)/(k+2) formula evaluated at qtz's k=4 gives ε_CP = 1/3, which is structurally distinct from 1/5. **Supplementary empirical validation (NOT closure mechanism):** the data-conditional MDL follow-up (`uniqueness_audit_v2_data_conditional_mdl_2026-04-30.md`) shows qtz's ε_CP = 1/3 (Class D) or 1/7 (Class A spectral form 1/(2k−1)) contributes to the ~2×10⁸ bit global disagreement vs PDG; this confirms the structural exclusion is correct but does not itself provide closure. Earlier framing of data-conditional MDL as "reconfirms UNIQUE" was goal-seeking and is RETRACTED 2026-05-01 PM per an internal note (REVISED). Row P28 retains UNIQUE-THEOREM-GRADE for ε_CP itself; conditional on Row 4 Brown-rank for the k*=3 step.

### Row P29 — η_B = (√3/10)·(2/3)^48 = 6.11×10⁻¹⁰ (UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 Brown-rank closure + Sakharov-Hashimoto chain theorem-grade)

- *Claim.* η_B = ε_CP · Re(h_P) · α₁^M = (1/5)·(√3/2)·(2/3)^48 = (√3/10)·(2/3)^48 ≈ 6.112×10⁻¹⁰. Substrate-Sakharov chain via NB-walker (Hashimoto) formalism.
- *Source.* `predictions/feshbach_exponent_principle.py` (α₁); `predictions/B_P_doubly_degenerate_h.py` (h_P); `proofs/cosmology/srs_eta_b_p_dominance.py` (Laplace concentration at saddle); `../theorems/theorem_eta_B_substrate_sakharov_closure_2026-04-30.md` (closure attempt); an internal working note (skeleton derivation push).
- *Observed.* Planck 2018: η_B = (6.12 ± 0.04)×10⁻¹⁰. Framework predicts 6.112×10⁻¹⁰. Deviation: −0.20σ (0.13%).
- *Operations invoked.* Sakharov 1967 conditions in framework's chiral I4₁32 substrate + Bayesian-toggle Beta(2,1) (Row P28) + Hashimoto NB-walker formalism (A1 + walker_dynamics) + Feshbach Exponent Principle (n_fixed = 2) + Sunada 2012 cycle accounting + Laplace concentration at unique BZ saddle (`srs_eta_b_p_dominance.py`) + handshake lemma (N_edges = N_atoms·k*/2).
- *Alternatives.* Tree-amplitude alternatives (E(P) = √3, |h_P| = √2, Im(h_P) = √5/2, no tree factor) all observationally excluded at >40σ (theorem_eta_B_substrate_sakharov_closure_2026-04-30 §4). Chain-length alternatives (M ∈ {4, 5, 7, 8}) all >8× off in magnitude. Predecessor candidates retracted: (7/40)·(2/3)^48 (numerology with 3 colliding K-readings at k=3, failed Type 6 (6c)); (28/79)·√3·J² (the author's separate private derivation Laplace + SM-imported sphaleron + V_cb cascade-volatile).
- *Selection.* Substrate Laplace-Sakharov skeleton applied to NB-walker formalism gives η_B = ε_CP·Re(h_P)·α₁^M. Each factor uniquely structurally selected: Re(h_P) is the parity-even Hashimoto tree amplitude at the unique BZ saddle; M = 6 = N_edges per primitive cell from handshake (equivalently M = n_g·N_atoms/g from Sunada cycle accounting, equal by structural identity); ε_CP and α₁ are Row P28 / Feshbach-Exponent-theorem-grade.
- *Status.* **UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 Brown-rank closure + Sakharov-Hashimoto chain theorem-grade** (graduated 2026-04-30 EOD; reframed 2026-05-01 PM per an internal note REVISED). **Primary closure (structural):** (i) the 13 derivation sub-steps internal to the k=3 substrate close at theorem grade per `theorem_eta_B_substrate_sakharov_closure_2026-04-30.md` + 7 sibling closure docs (Sakharov skeleton + Hashimoto-NB tree amplitude Re(h_P)=√3/2 at unique BZ saddle + Feshbach girth-cycle survival α₁^M with M=6 from handshake N_edges = N_atoms·k*/2); (ii) Row 4 (k*=3) is itself structurally closed by **Brown rank** (k > d gives Fisher rank zero on excess edges → MDL eliminates strictly; per `predictions/k_star.py` + an internal working note §1). M6 sign-gate at Γ (Re(h_qtz_Γ) = −1 structurally forced, 4-regular + 3-vertex C_3 + Hermiticity per Phase 0d) is structural supporting evidence for the same exclusion at the η_B-specific level. **Supplementary empirical validation (NOT closure mechanism):** the M2 data-conditional MDL crush (~2×10⁸ bits across observables: Q_Koide +2×10⁸; Re(h_P) +2.5×10⁶ or ~13000 bits depending on tiebreaker; per an internal working note) confirms that the structural exclusion of qtz aligns with PDG, but does not itself provide closure. Earlier framing of data-conditional MDL as "the genuinely robust Row 4 closure mechanism" was goal-seeking and is RETRACTED 2026-05-01 PM (user catch: "PDG is never the metric of robustness"). Audit v2 history: BLOCKED → STRUCTURAL-DERIVATION-GRADE → UNIQUE-THEOREM-GRADE (early 2026-04-30) → DOMINANT-CONDITIONAL (later 2026-04-30) → UNIQUE-THEOREM-GRADE-CONDITIONAL via M6 sign-gate (2026-04-30 EOD; later corrected) → reconfirmed via data-conditional MDL (2026-04-30 EOD final; RETRACTED 2026-05-01 PM as goal-seeking) → **UNIQUE-THEOREM-GRADE-CONDITIONAL on Brown-rank + Sakharov-Hashimoto chain (2026-05-01 PM)**. Numerical match −0.20σ within Planck precision unchanged.
- *Margin.* −0.20σ (0.13% relative gap from observed).
- *Conditional on.* Row P28 (ε_CP = 1/5, theorem-grade); Row 4 (k* = 3); Row 16 (N_atoms = 4); Row 7 (|E| = 6); Moore bound (g = k*²+1 = 10); Sunada 2012 cycle theorem; Hashimoto-NB transfer-matrix formalism (A1 + walker_dynamics axiom); Sakharov 1967 chiral substrate satisfies B/C/CP/out-of-equilibrium conditions; Laplace concentration at unique k_P saddle (`srs_eta_b_p_dominance.py`).
- *Gap.* (Updated 2026-04-30 after final-attacks pass.) Seven closure docs shipped today total: `eta_B_sakharov_skeleton_derivation` + `eta_B_per_mode_decomposition` + `eta_B_single_saddle_event_MDL` + `theorem_eta_B_gap_closures` + `theorem_eta_B_step_attacks` + `theorem_eta_B_final_attacks`. **All 13 derivation sub-steps are theorem-grade.** Step 3 (cosmic-time tick) closes via A2 + Lemma 1 description-length comparison: "preserve existing residue" has shorter description than "create new residue at saddle k_P" — therefore A2 retains "preserve" at all subsequent ticks; per cell, ONE asymmetric residue created per cosmic age. Step 4 (photon helicity normalization) closes via Hashimoto-Bass formula: Re(h_P) = E(P)/2 absorbs the 1/n_γ = 1/2 factor for n_γ = 2 photon polarizations (L = ω-irrep + R = ω²-irrep at P, per `srs_photon_walker_correspondence.py`). Original C₃-irrep matching argument retracted in favor of cleaner Hashimoto-Bass derivation. **No structural sub-steps remain.** Strict theorem-grade achieved.
- *Filtered-alternative residue.* the author's separate private derivation (28/79)·√3·J² superseded but retained in `proofs/cosmology/eta_B_derivation.py` as audit trail.
- *Algebraicity gate (Type 6) note.* (6a) L-expression: ε_CP · spectral(B(k_P), h_P).Re · count(N_edges)·(geometric_sum integer power) — all primitives in L. (6b) K-membership: √3/10 ∈ K via √3 basis element. (6c) MDL minimum: Only Re(h_P) tree-amplitude assignment matches at <2σ; alternatives observationally excluded at ≥40σ; M=6 uniquely structurally derived. ✓

### Row P30 — R_ν = Δm²₃₁/Δm²₂₁ = 228/7

- *Claim.* R_ν = 228/7 ≈ 32.5714 from K₄ Green's-function Chebyshev expansion at the Ihara phase φ = arctan(√7).
- *Source.* `predictions/R_nu_splitting.py`, `predictions/R_nu_splitting_derivation.md`, `R_theorem.md`.
- *Observed.* NuFIT 6.0 (September 2024, NO): R_ν = 33.83 ± 0.92. Deviation: +1.4σ.
- *Operations invoked.* K₄ Hashimoto on the srs primitive-cell quotient (the Ihara phase = arctan(√(4(k\*−1)−1)) for k\*-regular); Chebyshev-U expansion of the Green's function (Stark-Terras 1996); the Gaussian-integer identity (1+i√7)⁵ = 176 − 16i√7.
- *Alternatives.* (i) Other Chebyshev distances n ≠ 5; (ii) different cubic selectors; (iii) different K_4-vs-srs quotient choices.
- *Selection.* The cubic q³ = 5q − 2 has unique positive-integer root q = 2 = k\*−1, which selects n = 5. Then R = 2/sin²(5φ) − 4 with sin²(5φ) = 7/128 (from Gaussian-integer arithmetic) gives R = 256/7 − 4 = 228/7 exactly.
- *Status.* **UNIQUE — THEOREM-GRADE** (exact rational under the cubic-selection chain).
- *Margin.* +1.4σ from NuFIT 6.0.
- *Conditional on.* Row 4 (k\* = 3), Row 7 (|E| → K_4 quotient via primitive cell), Stark-Terras 1996 Chebyshev.
- *Gap.* —
- *Filtered-alternative residue.* Other Chebyshev distances: hard-gated by the cubic-uniqueness argument.

### Row P31 — m_ν3, m_ν2 absolute (DOMINANT-CONDITIONAL re-graded 2026-05-18)

- *Claim.* m_ν3 = (k* × N_atoms) × M_Pl × N_hub^(-1/2) ≈ 50.57 meV. Equivalent to seesaw m_ν3 = v²/M_R with M_R = δ⁴·M_Pl/(2·k*·N_atoms); δ⁴ in v² and M_R cancel exactly leaving the global form. **m_ν3 is INDEPENDENT of the Koide phase δ** — clean structural distinction from charged-lepton masses. m_ν2 = m_ν3/√R = 8.86 meV via R = 228/7 (theorem-grade Ihara) under the new chain (reconciled 2026-05-04 same day; older ADOPTED-PS chain at predictions/retracted/m_nu{2,3}_seesaw_PS.py).
- *Source.* `predictions/m_nu3.py`, `predictions/m_nu3_derivation.md` (NEW 2026-05-04); `predictions/m_nu2.py` (NEW 2026-05-04, m_ν3/√R chain); reframing context an internal working note; structural derivation `proofs/flavor/srs_M_R_step{1_structural,2_derivation,3_closure}.py`.
- *Observed.* From NuFIT 6.0: m_ν2 = √Δm²₂₁ ≈ 8.65 meV; m_ν3 = √Δm²₃₁ = 50.13 ± 0.20 meV (NO; m_ν1 = 0 is OBSERVATIONAL CONVENTION, structural derivation retracted under B6 — see R-15).
- *Operations invoked.* BZJ for v (theorem-grade, predictions/v_higgs.py) + Type-I seesaw (Mohapatra-Senjanović 1980) + Majorana mass coefficient 1/2 (Peskin-Schroeder §3.4) + Bloch decomposition normalization 1/(k*·N_atoms) (Ashcroft-Mermin §8) + Wigner-D bilinear δ⁴ = (δ²)² (algebra given v's δ²) + R = 228/7 Ihara (predictions/R_nu_splitting.py).
- *Alternatives.* (1) Closed-form M_R = 2/k*^(g-1) × M_Pl (equivalent rational, alternate structural reading). (2) Older PS-seesaw chain with M_R = (2/3)^g × M_GUT (RETRACTED 2026-05-04 — required ADOPTED-PS bare scale + adopted m_t(GUT) + MSSM RG; same numerical answer through compensation but more adopted inputs).
- *Selection.* Global formula chosen over PS seesaw: zero adopted inputs vs. four (M_GUT, m_t(GUT), tan β, MSSM RG); cleaner δ-independence; matches NuFIT 6.0 at +0.87% (+2.18σ_PDG; FAIL Clause 8 against σ_PDG alone).
- *Status.* **DOMINANT-CONDITIONAL** for m_ν3 and m_ν2 (re-graded 2026-05-18). Prior label UNIQUE-THEOREM-GRADE-CONDITIONAL was overstated — the chain did not disclose the y_ν = 1 adoption (the chain's "global formula" m_ν₃ = (k*·N_atoms)·M_Pl·N_hub^(-1/2) is the Type-I seesaw with the Dirac Yukawa fixed at unity, which is structurally motivated but adopted, not derived). The live derivation file's banner: *"DOMINANT-CONDITIONAL (re-graded 2026-05-18 chain audit; was overstated as UNIQUE-THEOREM-GRADE-CONDITIONAL)"*. Numerical value unchanged. m_ν2 inherits via `predictions/m_nu2.py = m_ν3 / √R`.
- *Margin.* m_ν3 at +0.87% = +2.18σ_PDG (NuFIT 6.0); m_ν2 at +2.40% = +1.91σ_PDG. Both FAIL Clause 8 against σ_PDG alone; the structural source of the residual is the N_hub anchor variation between {G_F, m_τ, R_∞} calibrations.
- *Dark-correction note (2026-05-15 sweep).* The 2026-05-04 spectral-gap reformulation `m_ν₃ = (k*·N_atoms)·M_Pl·N_hub^(-1/2)` **bakes the Feshbach mechanism into the bare scale** — the formula IS the residue-at-h reading of Σ(h) = α₁·h̄/|h|² applied to the substrate's lowest spectral mode.  The universal-template multiplicative DC factor (1 − √5/4·α₁/(1-α₁)) ≈ 0.9773 from master doc §3 (B) **must NOT be applied on top** (verified: doing so shifts m_ν₃ to 49.42 meV, −1.4% / −3.5σ_PDG — WORSE; double-count).  Family D sub-leading at the (0H+2F) Majorana vertex is +α₁²/6 ≈ +0.025%, negligible vs the N_hub anchor sensitivity.  Master doc `../theorems/theorem_substrate_feshbach_dark_corrections_master.md` §3 (B) "Application clarification" and §5 catalog row for m_ν_3 updated 2026-05-15 to reflect this; `predictions/m_nu3.py` + `m_nu3_derivation.md` §6 Q0 + `m_nu2.py` annotated. The +0.87% (m_ν₃) / +2.4% (m_ν₂) residuals are N_hub-anchor-driven (already documented), NOT missing DCs.
- *Conditional on.* Row P30 (R_ν); Row 4 audit v2 closure for k*=3; G_sub Drude closure for M_Pl anchor; the G_F-calibration of N_hub's value (N_hub is the adopted input).
- *Gap.* (a) Sub-leading per-cell prefactor at <0.1% precision (m_ν3). (b) Direct m_ν2 global formula INVESTIGATED 2026-05-04 LATE: NEGATIVE — closest natural candidate X = (k*−1) gives -3.20%, worse than R-splitting chain; current chain m_ν2 = m_ν3/√R is the right derivation. (c) ADOPTED-Z3 unchanged for color↔generation identification (R-14-related; not load-bearing for m_ν3 magnitude).
- *Filtered-alternative residue.* PS seesaw + ADOPTED-PS-SCALE chain (RETRACTED 2026-05-04); preserved at `predictions/retracted/m_nu3_seesaw_PS{,_derivation}.{py,md}`.

### Row P32 — θ_12_PMNS ≈ 33.07°: UNIQUE-THEOREM-GRADE for structural form; labeling data-anchored

- *Claim.* cos θ_12 = cos θ_TBM / cos θ_C with V_us = 9/40 (Row P4); gives θ_12 ≈ 33.07°.
- *Source.* per an internal working note; inherits Row P14 V_ub family graduation 2026-04-30 via M1 amplitude-form + Angle D + Z3-mass-order verdicts. Bridge-functoriality graduation 2026-04-28 RETRACTED 2026-04-29 (no longer needed; superseded).
- *Observed.* NuFIT 6.0: 33.41° ± 0.75°. Deviation: −0.45σ.
- *Operations invoked.* SU(4)_PS perpendicular-rotation argument; TBM base + Cabibbo correction; inherited M1 twisted-walker amplitude form for color-generation identification.
- *Alternatives.* Different correction patterns; different TBM bases.
- *Selection.* Perpendicular-rotation argument is theorem-grade structural identity; the color-generation identification chain inherits Row P14's V_ub family graduation (2026-04-30): predictions theorem-grade in form; labeling layer data-anchored, non-blocking for predictive content.
- *Status.* **UNIQUE-THEOREM-GRADE for structural form; labeling data-anchored, non-blocking for predictive content** (graduated 2026-04-30 via inheritance from Row P14 M1 amplitude-form + Angle D + Z3-mass-order verdicts).
- *Margin.* −0.45σ.
- *Conditional on.* Row P4 (V_us); Row P14 (V_ub amplitude theorem-grade + labeling data-anchored); Row 17 (Pati-Salam structural — now FULLY DERIVED via G2-D + Cl(6) Fock + chirality-doubled edge qubit, 2026-05-05 EOD+3 per `theorem_g2d_chirality_doubled.md`).
- *Gap.* Closed at predictive-content level. Labeling layer is OTHER-SMUGGLE residue inherited from Row P14, not row-specific.

### Row P33 — θ_13_PMNS ≈ 8.61° (UNIQUE-THEOREM-GRADE-CONDITIONAL post-2026-05-02 PS embedding closure + 2026-05-05 EOD+3 G2-D)

- *Claim.* θ_13 ≈ 8.61° with Class-3 dark correction on V_us_bare (PS embedding step).
- *Source.* per `target_parameters.md` neutrino table; sub-class part inherits Row P14 V_ub family graduation 2026-04-30; PS embedding step CLOSED 2026-05-02 EOD+13 via an internal working note + `proofs/foundations/theta_13_PMNS_PS_embedding_class2_class3_audit.py`. Bridge-functoriality graduation 2026-04-28 RETRACTED 2026-04-29 (no longer needed; superseded).
- *Observed.* NuFIT 6.0: 8.57° ± 0.11°. Deviation: +0.32σ.
- *Operations invoked.* PS embedding step (now closed via Class-2/Class-3 taxonomy) + dark-correction coefficient (Row P5) + walk-rep sub-class (inherits M1 amplitude form + Angle D + Z3-mass-order). PS gauge group SU(4) × SU(2)_L × SU(2)_R fully derived via `theorem_g2d_chirality_doubled.md` (2026-05-05 EOD+3) — strengthens Row 17 (Pati-Salam) structural foundation used in PS embedding step.
- *Alternatives.* Different embedding routes; different dark-correction coefficients. Class-2/Class-3 distinction in `theorem_dark_correction_mdl.md` resolves between V_us_full and V_us_bare for PS embedding (V_us_full would double-count Class-2 correction; V_us_bare = V_us_full / (1+√5/4·α_1) ≈ 0.2202 is the structurally consistent input).
- *Selection.* Dark correction with c = 5/12 (Row P5) on a TBM base; walk-rep sub-class inherits Row P14 graduation (data-anchored, non-blocking). PS embedding step closed via Class-2/Class-3 distinction (V_us_bare for Class-3 angle observable; STRUCTURAL-DERIVATION-CONDITIONAL on Class taxonomy).
- *Status.* **UNIQUE-THEOREM-GRADE-CONDITIONAL post-2026-05-02 EOD+13** (PS embedding closed) **+ 2026-05-05 EOD+3** (G2-D closes hypercharge → Row 17 PS structural fully derived). Audit history: ADVANCED → BLOCKED on PS embedding step → 2026-05-02 PS embedding closure via Class-2/Class-3 taxonomy → **UNIQUE-THEOREM-GRADE-CONDITIONAL** on Class-2/Class-3 distinction + sub-class data-anchored.
- *Margin.* +0.32σ vs PDG 8.57° ± 0.11°.
- *Conditional on.* Row P5 (Class-2 dark correction coefficient); Row 17 (Pati-Salam — fully derived 2026-05-05 EOD+3 via G2-D); Row P14 (V_ub family — sub-class part data-anchored, non-blocking); Class-2/Class-3 distinction in `theorem_dark_correction_mdl.md` (theorem-grade per file's clauses).
- *Gap.* Closed at predictive-content level (PS embedding via Class-2/Class-3 distinction). Sub-class labeling layer is OTHER-SMUGGLE residue inherited from Row P14, not row-specific. Class-2/Class-3 taxonomy is theorem-grade structural taxonomy of dark corrections.

### Row P34 — δ_CP_PMNS (THEOREM-GRADE-STRUCTURAL-CONDITIONAL via V_{-1}-T_{B-L} identity; W3-reconciled 2026-05-18)

- *Claim.* δ_CP_PMNS = arccos(−1) = π = 180° via the V_{-1}-T_{B-L} structural identity applied to the lepton SU(2)_L doublet.
- *W3 reconciliation (2026-05-18).* The framework's ORIGINAL Hashimoto-phase route ((g−1)·arg(h*)≈249.85°) WAS falsified at +3.83σ vs NuFIT 6.0 IC19 (2026-05-02) — `honest_assessment.md` item 3's pre-registered "δ_CP=180° ⇒ Hashimoto phase mechanism fails" FIRED AS DESIGNED for that mechanism, and item 3 is now reconciled (records the failure + the supersession + a new falsification condition). The 180° is NOT a post-hoc rescue: it is the parameter-free V_{-1}-T_{B-L} geometric identity, INDEPENDENTLY CORROBORATED — the same identity gives δ_CP_CKM=arccos(1/3)=70.53° (+0.68σ, a different observable). Grade is THEOREM-GRADE-STRUCTURAL-**CONDITIONAL** (on Need-D-3 + the geometric↔Jarlskog adoption); the +0.16σ Clause-8 is NOT an unconditional pass.
- *Source.* `proofs/foundations/sector_V_minus_one_T_BL_identity.py` (machine-precision verification); `proofs/foundations/sector_dCP_unified_closure.py` (closure framing); an internal working note (full derivation).
- *Observed.* NuFIT 6.0 (Sep 2024, NO best-fit): 177°⁺¹⁹₋₂₀.
- *Operations invoked.* (i) K_4 V_{-1} eigenspace projection (theorem-grade per `delta_CP_CKM_geometry_derivation` Step 3). (ii) Slansky 1981 §4 Table 5 Killing-form-normalized U(1)_{B-L} generator T_{B-L} = diag(+1/3, +1/3, +1/3, −1) (theorem-grade per `theorem_sin2_theta_W_unification` L4). (iii) Trace-zero direction in V_{-1}: T_{B-L}·v_0 ∈ V_{-1} since Tr(T_{B-L}) = 0. (iv) Inner product identity: cos(angle in V_{-1} between K_4 atom q_lep and T_{B-L} direction u) = T_{B-L} eigenvalue at lepton atom = −1. (v) **Symmetry-breaking bridge (added 2026-05-05 EOD+3, `proofs/foundations/sector_V_minus_one_T_BL_symmetry_breaking_bridge.py`):** the geometric interpretation "V_{-1}-T_{B-L} angle = δ_CP" is now derived structurally — T_{B-L} acts in V_{-1} along u = -q_lepton/|q_lepton| (machine-precision verified), breaking the K_4 regular-tetrahedron symmetry SO(3)_K4 → SO(2)_u (rotations around u-axis). The per-atom polar angle from u-axis is the UNIQUE SO(2)_u-invariant phase per atom (linear-algebra fact). Type 6c (6c) clause closes via channel_select with structural channel c = "SO(2)_u-invariant per-atom phase from broken-symmetry axis u" (not bit-cost). The K_4 dihedral framing (color-only) is the COLOR-SECTOR SPECIAL CASE of this unified reading; lepton sector requires the symmetry-breaking framing. (vi) Bridge to Jarlskog phase: inherits framework's existing CKM identification (Other-Smuggle per `delta_CP_CKM_geometry §6`).
- *Alternatives.* Three substrate-side R-14 closure paths (a), (b), (c) all converged on substrate-blindness this session. The V_{-1}-T_{B-L} identity uses theorem-grade upstream content (Coxeter 1973 K_4 dihedral + Slansky T_{B-L}) without requiring substrate-side sector-differentiation mechanism — instead, it uses the natural V_{-1} geometric content with U(1)_{B-L} weighting.
- *Selection.* The V_{-1}-T_{B-L} identity gives a UNIFIED rule: cos(δ_CP) = T_{B-L} eigenvalue of doublet's PS sector. For color sector (CKM): cos = +1/3 → arccos(1/3) = K_4 dihedral (matches existing CKM identification). For lepton sector (PMNS): cos = −1 → arccos(−1) = π = 180°. Single mechanism, both sectors. Predicted value SET {arccos(1/3), arccos(−1)} is (Z/2)³-invariant per `R14_ADOPTED_B3_attack_2026-05-05.md` magnitude-level analysis.
- *Status.* **THEOREM-GRADE-STRUCTURAL-CONDITIONAL** via V_{-1}-T_{B-L} identity (revived 2026-05-05 EOD+2; strengthened 2026-05-05 EOD+3 via SO(3)_K4 → SO(2)_u symmetry-breaking bridge). Conditional on: framework's existing CKM identification (Other-Smuggle per `delta_CP_CKM_geometry §6`) being valid as the bridge from W-vertex 4-walk Jarlskog phase to V_{-1} angle. **Need-A2 generation-Z_3 existence CLOSED 2026-05-08** (commit `42a6928` via M1.B chain + M_gen non-degeneracy generic argument); the remaining gating residual for graduation to UNIQUE-THEOREM-GRADE is Need-D-3 alone (Y_u vs Y_d eigenbasis on C³_gen, multi-session research). Audit history: BLOCKED → ADVANCED → c=1 theorem-grade → RETIRED 2026-05-02 → **REVIVED 2026-05-05 EOD+2 via V_{-1}-T_{B-L} identity** → strengthened 2026-05-05 EOD+3 (Type 6c PASSES via channel_select) → Need-A2 closed 2026-05-08.
- *Margin.* +3° absolute = +0.15σ vs NuFIT 6.0 NO best-fit 177° ± 20°. Within Clause 8 tolerance.
- *Conditional on.* (i) Framework's existing CKM identification (`delta_CP_CKM_geometry §6` Other-Smuggle): Need-D 4-layer audit 2026-05-05 EOD+3 — D-1 (V_ab magnitudes) and D-2 (δ_CP phases) closed; **D-3 (Y_u vs Y_d eigenbasis) closure pathway BOUNDED on Need-A2 alone** post-G2-D closure 2026-05-05 EOD+3 (Route 4 unblocked on G2-D side via `theorem_g2d_chirality_doubled.md`); D-4 (M_species mass scales) remains research-level. (ii) ADOPTED-B3 labeling residue (which sector is lepton vs quark) — **HYPERCHARGE COMPONENT graduated 2026-05-05 EOD+3 via G2-D**; remaining sector/generation labeling residue non-blocking via (Z/2)³ Angle D verdict per `R14_ADOPTED_B3_attack_2026-05-05.md`.
- *Gap.* The bridge from V_{-1} angle to W-vertex 4-walk Jarlskog phase for the LEPTON sector (color sector bridge = existing CKM identification). This is the same Other-Smuggle status as Row P15 — non-blocking under the framework's current audit conventions, but a future structural derivation (= Need-A2 + Need-D closure, R-14 path (a) work) would graduate both Row P15 and Row P34 to UNIQUE-THEOREM-GRADE simultaneously. **Post-2026-05-05 EOD+3 strengthening:** the gap is now SHARPER — the symmetry-breaking framing identifies the V_{-1} angle as the UNIQUE SO(2)_u-invariant per-atom phase under T_{B-L}-induced symmetry breaking (Type 6c PASSES); the residual adoption is shifted to a SINGLE place (CKM ↔ K_4 walks identification) rather than separate per-sector adoptions (K_4 dihedral for color + extension for lepton). **2026-05-08 update**: Need-A2 generation-Z_3 existence is **CLOSED** via M1.B chain (rediscovered) + M_gen non-degeneracy via generic argument (this session). Remaining gap for full UNIQUE-THEOREM-GRADE on P15 + P34 is now Need-D (specifically D-3 Y_u vs Y_d eigenbasis on C³_gen, multi-session research per 2026-05-05 EOD+3 audit), NOT Need-A2.
- *Filtered-alternative residue.* The previous (g−1)·arg(h*) ≈ 249.85° formula is RETIRED (preserved in `predictions/retracted/delta_CP_PMNS_derivation.md`); the V_{-1}-T_{B-L} identity supersedes it.

### Row P35 — α_21_PMNS ≈ 162.39° (STRUCTURAL-DERIVATION-CONDITIONAL — re-graded 2026-05-12; was UNIQUE-THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1, INFLATED)

- *Claim.* α_21 ≈ 162.39° from arg(M_R^(ω)) = g·arg(h_ω) via PS seesaw + Takagi, where h_ω = (√3+i√5)/2 (Hashimoto walker P-point eigenvalue, theorem-grade) and g = 10 (srs girth, theorem-grade).
- *Source.* `target_parameters.md`. P34-level c=1 algebra (theorem-grade via uniqueness template 2026-04-29) + the 3×3 M_R diagonal-phase structure M_R^(m,m) = |M_R|·h_m^g on the C_3 generation modes (`proofs/flavor/srs_hashimoto_seesaw_verify.py`, `proofs/foundations/path_b_M_R_upgrade.py`).
- *Observed.* Largely unconstrained by current data (Majorana phases are not measured; only 0νββ gives weak combined bounds).
- *Operations invoked.* h_ω, g [Type 4 theorem-grade: `predictions/h_walker_eigenvalue.py`, `predictions/g_girth.py`] + 3 generations from C³_gen [Type 4: `predictions/R3_observer_c3_generation.py`, L2 theorem-grade] + the **`h^g`-phase IDENTIFICATION** (below, NOT derived) + Type-I PS seesaw m_ν = M_D·M_R⁻¹·M_D^T [Type 3, Mohapatra-Senjanović 1980] + Takagi diagonalization [Type 2] → α_21 = g·arg(h_ω) mod 360°.
- *Alternatives.* At the c-coefficient level: parity-odd-functional alternatives, closed as in P34. At the phase level: the value depends on which closed-structure-length set sources the M_R phase — see *Gap*.
- *Selection.* The C_3-generation channel ω carries Hashimoto eigenvalue h_ω; the Majorana coupling on that channel is taken to carry one girth-ring's walker holonomy ⇒ phase g·arg(h_ω). (The earlier "cardinality-1 ↔ n=1" Path-B framing routed this through the K_4-triangle ↔ srs-girth-ring map, which is FALSE — see *Gap*.)
- *Status.* **STRUCTURAL-DERIVATION-CONDITIONAL** (re-graded 2026-05-12). The structural chain (h, g, 3 generations, seesaw, Takagi, the cardinality-orbit channel structure, sterile-mode absence, the S_3 little-group algebra) is theorem-grade-rigour, BUT the load-bearing step — that M_R^(m,m) carries the phase factor h_m^g (one girth-ring walker holonomy per generation channel) — is a bare IDENTIFICATION, not derived. The 2026-05-04 EOD+1 "UNIQUE-THEOREM-GRADE-CONDITIONAL" label was inflated. **Discharge attempted and FAILED 2026-05-12** (`proofs/foundations/majorana_M_R_waterfilling.py`): (Route 1) M_R as an A2-T-waterfilled loop sum Σ_{L≥g} 2^{-DL(L)}·h_m^L does NOT converge — Ramanujan saturation |h|²·2^{-rate} = 2·2^{-1} = 1 makes every retained ring length contribute equal magnitude, no finite cutoff emerges from the A2-T surprise threshold, and the phase drifts as ≈(g+L_max)/2·arg(h_m); only the L_max = g (single-girth-structure) special case gives g·arg(h_m), and the rate-distortion machinery does not single it out. (Route 2) the Path-B "cardinality-k orbit ↔ k girth rings" chain is broken at the root: the K_4 cycle-space generators (triangles) have nonzero Z³ voltages {(1,0,0),(0,-1,0),(0,0,1),(1,1,1)} so they do not lift to closed srs cycles at all — the factor `g` in `(k-1)·g·arg(h)` is not sourced. ⇒ The M_R phase is an A5(a)-adjacent identification (ADOPTED-NU-MAJ-PHASE), not an MDL-waterfilling consequence.
- *Margin.* — (unconstrained observable). The predicted value 162.39° = g·arg(h_ω) holds under the identification; not falsified.
- *Conditional on.* (i) **ADOPTED-NU-MAJ-PHASE** — the ν_R Majorana coupling carries one girth-ring walker holonomy h_m^g per generation channel (A5(a)-adjacent identification; see adoption register). (ii) Mass-ordering identification gen-k ↔ generation channel uses observer C³_gen L3 (PDG mass non-degeneracy as A5(a) external input). (iii) PS gauge labeling (ADOPTED-B3, data-anchored non-blocking). The real-valued |M_R| = δ⁴·M_Pl/(2·k*·N_atoms) magnitude chain (m_ν₃ closure) is phase-free and is NOT part of this conditional — rows m_ν₂/m_ν₃ are unaffected.
- *Predictions/ file.* `predictions/alpha_21_PMNS.py` + `predictions/alpha_21_PMNS_derivation.md` (relabelled 2026-05-12). Discharge-attempt probe: `proofs/foundations/majorana_M_R_waterfilling.py`.
- *Gap.* The M_R phase factor h_m^g is not derived from A2-T (see *Status*). The specific length set that sources it (single girth ring vs. waterfilled loop sum vs. some other object) is open; the loop-sum route diverges, the K_4-cardinality route's `g` is unsourced (K_4 triangles don't lift to srs cycles). A correct derivation might land on a different phase ⇒ the 162.39° value is identification-conditional.
- *Filtered-alternative residue.* The (k-1)·g·arg(h) "Path B" derivation chain (`path_b_*` docs) has a broken link (K_4-triangle ↔ srs-girth-ring) — correction notes added to `path_b_M_R_upgrade_2026-05-03.md` §2 and `path_b_cardinality_reconciliation_2026-05-02.md`.
- *2026-05-19 update (NO grade change; NOT narrowed) — CORRECTS a same-day overclaim.* A 9-probe arc terminated. An interim commit (6ac4c69) claimed α_21 factorizes as `[discrete ΔL=2 holonomy — derived] × [spectral arg(h_ω) — residual]` and that the conditional was *narrowed*. **RETRACTED:** probes 8/8a amplitudes are `i^Y·ω^{wY}` (purely finite-group, no arg(h)); the "× spectral factor" was an unverified bridge. Probe 9 (`majorana_phase_deltaL2_perron`, gate-verified) proves the ΔL=2/hypercharge-Y constraint does NOT lift the Ramanujan degeneracy — `|μ_max|=√2` exactly degenerate (gap≈2e-16) across the whole ΔL=2-relevant regime including the P-point; only the trivial closed `|μ|=2` (≠Majorana) is isolated. So `g·arg(h_ω)`=162.39° is **entirely adopted** (irreducibly the Ramanujan-degenerate spectral eigenphase); ADOPTED-NU-MAJ-PHASE stands, **unnarrowed**. A clean cutoff-free enantiomer-signed *discrete* ΔL=2 holonomy is real but **decoupled** from the value. Status `STRUCTURAL-DERIVATION-CONDITIONAL` unchanged. Corrected record: an internal working note.

### Row P36 — α_31_PMNS ≈ 324.78° (STRUCTURAL-DERIVATION-CONDITIONAL — re-graded 2026-05-12; was UNIQUE-THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1, INFLATED)

- *Claim.* α_31 ≈ 324.78° = arg((h_ω/h_ω²)^g) mod 360° via the same M_R diagonal-phase structure (cardinality-2 channel / ω² generation).
- *Source.* `target_parameters.md`. Same chain as P35 with the ω² generation channel (h_ω² = (-√3+i√5)/2, arg(h_ω²^g) ≈ 197.61°; seesaw + NuFIT ordering gives α_31 = 324.78°).
- *Observed.* Largely unconstrained.
- *Operations invoked.* Same chain as P35 with the second non-trivial C_3 channel.
- *Alternatives.* Same as P35.
- *Selection.* Same as P35 (girth-ring walker holonomy on the ω² channel).
- *Status.* **STRUCTURAL-DERIVATION-CONDITIONAL** (re-graded 2026-05-12). Same as P35: structurally complete, but the M_R phase factor h_m^g is a bare identification (ADOPTED-NU-MAJ-PHASE), not derived; discharge attempted and FAILED 2026-05-12 (`proofs/foundations/majorana_M_R_waterfilling.py` — loop sum diverges, K_4-cardinality `g` unsourced). The 2026-05-04 EOD+1 "UNIQUE-THEOREM-GRADE-CONDITIONAL" label was inflated.
- *Margin.* — (unconstrained observable). Predicted 324.78° holds under the identification; not falsified.
- *Conditional on.* Same as P35 ((i) ADOPTED-NU-MAJ-PHASE, (ii) C³_gen L3 mass-ordering, (iii) ADOPTED-B3).
- *Predictions/ file.* `predictions/alpha_31_PMNS.py` + `predictions/alpha_31_PMNS_derivation.md` (relabelled 2026-05-12).
- *Gap.* Same as P35.
- *Filtered-alternative residue.* Same as P35.
- *2026-05-19 update (NO grade change; NOT narrowed) — CORRECTS a same-day overclaim.* Same as P35: the "factorizes into derived discrete × residual spectral" claim is **RETRACTED** (probe 9, gate-verified: the ΔL=2 constraint does NOT lift the Ramanujan degeneracy; α_31=324.78° is entirely adopted, irreducibly spectral). The clean discrete ΔL=2 holonomy is real but decoupled from the value; ADOPTED-NU-MAJ-PHASE unnarrowed; status unchanged. Corrected record: an internal working note.

### Row P37 — koide_quark_ratio (ε²_up − 2)/(ε²_down − 2) = 14/5

- *Claim.* (ε²_up − 2)/(ε²_down − 2) = 2 + (g−2)/g = (3g − 2)/g = 14/5 exactly (independent of α₁ value).
- *Source.* `predictions/koide_quark_ratio.py`, `predictions/koide_quark_ratio_derivation.md`.
- *Observed.* Computed from PDG quark Q values: 2.819 (~ 0.5σ-class match). Deviation: −0.7%.
- *Operations invoked.* A5(b) MDL probability = coupling; Cl(6) Fock Z_3 cyclic edge symmetry (Step 1 breaking factor f(n) = n(3-n)/3, derived 2026-05-05 EOD+2 via `proofs/foundations/cl6_fock_z3_breaking_decomposition.py`); many-body expansion (one-body α₁ + two-body α₁₂); pair-correlation length (g−2)/g identity.
- *Alternatives.* Different many-body expansion structures; different breaking-factor choices; non-rational coefficients.
- *Selection.* (i) Z_3 cyclic edge symmetry on Cl(6) Fock at trivalent vertex gives breaking factor 2/3 for both n=1 (down) and n=2 (up) — symmetric, cancels in the ratio (was Open Question 1; closed 2026-05-05 EOD+2). (ii) Many-body expansion: n=1 → α₁; n=2 → 2α₁ + α₁₂. (iii) Pair-correlation α₁₂/α₁ = (g−2)/g from same girth-backbone identity as α₁_bare. Combined: ratio = 2 + (g−2)/g = (3g−2)/g.
- *Status.* **UNIQUE — THEOREM-GRADE** (under A1 + A2-T + local CAR + A5(b) + g = 10). One adoption removed 2026-05-05 EOD+2 (f(n) prefactor now derived rather than imported from the author's separate private derivation).
- *Margin.* Strict for the rational identity; −0.7% from observation (~0.5σ-class).
- *Conditional on.* Row 4 (k\*), Row 9 (g = 10), Row 16 (Cl(6)), A5(b).
- *Gap.* —
- *Filtered-alternative residue.* —

### Row P38 — m_t = 174.10 GeV (M_persistence + Type-II saturation, shipped 2026-05-26)

- *Claim.* m_t = M_persistence eigenvalue, derived via Type-II saturation y_t(GUT) = 1 + MSSM RGE running down to m_t pole; live `predictions/m_t.py` returns 174.1036 GeV.
- *Source.* `predictions/m_t.py`, `predictions/m_t_derivation.md`, `predictions/M_persistence.py` (12×12 fermion mass operator); M_persistence linter pass commit `c9fba27` (2026-05-26).
- *Observed.* PDG 2024: 172.69 ± 0.30 GeV. Live deviation: +0.82% relative, +4.71σ_PDG (Clause-8 FAIL on σ_PDG).
- *Operations invoked.* M_persistence assembly from chain-imported per-channel masses; Type-II y_t(GUT) = 1 saturation; framework MSSM RGE; no PDG mass leaks (CODATA M_Pl is the single SI anchor).
- *Status.* **THEOREM-GRADE-STRUCTURAL-CONDITIONAL** (M_persistence shipped 2026-05-26). Conditional on the framework's MSSM commitment + RGE infrastructure; numerical residual (+0.82% rel, +4.71σ_PDG) is MSSM-threshold + two-loop class, not a defect at the structural-derivation level.
- *Margin.* +0.82% rel, +4.71σ_PDG — Clause-8 FAIL vs σ_PDG strict; PASS vs the framework's MSSM-threshold systematic floor class.
- *Conditional on.* M_persistence theorem (`theorem_fermion_mass_operator_persistence_2026-05-21.md`), Type-II saturation, MSSM RGE.
- *Gap.* MSSM-threshold + two-loop systematic floor (~2-3% class for top quark below SUSY scale per parameter_linter.md §8b). Out-of-scope-by-construction for the framework's single-regime no-threshold scheme (Move-1).
- *Filtered-alternative residue.* —

### (HISTORICAL: prior Row P38 framing as ADOPTED-Z3-WATERFALL, retracted 2026-05-04, preserved for record)

- *Status.* DOWNGRADED 2026-05-04 EOD+3 → OPEN (retracted Koide-waterfall route). Per zero-empirical-inputs standard: the Koide-waterfall computation used observed m_c (PDG 1.27 GeV) and m_b (PDG 4.18 GeV) as load-bearing INPUTS to the prediction logic, so the framework could not compute m_top without two PDG empirical inputs in that chain.
- *File location.* `predictions/retracted/m_top.py`, `predictions/retracted/m_top_derivation.md` (moved 2026-05-04 EOD+3, preserved as honest history).
- *Closure path (now CLOSED via M_persistence above, 2026-05-26).* Two prior reframing attempts NEGATIVE in 2026-05-04: (a) Σ(h) per-sector lift (an internal working note, sector-blind), (b) y_t(GUT)=1 chain (an internal working note, fit-driven). The M_persistence pass 2026-05-26 supersedes both by importing only framework constants + CODATA M_Pl through the 12×12 fermion-mass operator.

### (HISTORICAL: original Row P38 claim, preserved for record)

- *Claim.* m_top from solving Q_Koide = 2/3 on the (c, b, t) cross-charge triality triplet using observed m_c, m_b: m_top ≈ 168.5 GeV.
- *Source.* `predictions/retracted/m_top.py` (moved 2026-05-04 EOD+3).
- *Observed.* PDG 2024: m_top = 172.69 ± 0.30 GeV. Deviation: −2.4%.
- *Operations invoked.* Q_Koide = 2/3 (Row P8, theorem-grade); algebraic solution of the Koide quadratic for m_top given m_c, m_b. External: Rivero waterfall observation (the author's separate private derivation).
- *Status (pre-downgrade).* ADVANCED — CONDITIONAL on ADOPTED-Z3-WATERFALL + 2 PDG inputs (m_c, m_b).
- *Margin.* −2.4% (consistent with SUSY-threshold magnitude).
- *Gap.* Z3-waterfall identification (cross-charge triplet selection rule); m_c, m_b PDG inputs as derivation inputs; same R-14 blocker as Row P39 quark masses.

### Row P39 — m_u, m_d, m_s, m_c, m_b (M_persistence + δ(n) PS Fock counting; shipped 2026-05-26)

- *Claim.* The five lighter quark masses ship via the M_persistence 12×12 fermion mass operator + δ(n) = 2/(9(n+1)) Pati-Salam Fock counting (Row n = 1 down quark, n = 2 up quark per Q_Koide = 2/3 + δ structure). Per-channel live values: m_u = 2.495 MeV, m_d = 4.605 MeV, m_s = 95.94 MeV, m_c = 1.277 GeV, m_b = 4.27 GeV.
- *Source.* `predictions/m_u.py`, `predictions/m_d.py`, `predictions/m_s.py`, `predictions/m_c.py`, `predictions/m_b.py`, all chained through `predictions/M_persistence.py` (M_persistence linter pass commit `c9fba27` 2026-05-26). Underlying δ(n) derivation: `proofs/masses/srs_delta_n_derivation.py` + `srs_fock_counting.py`; W3 PS sector connectivity closure.
- *Observed.* PDG 2024 pole/MS̄ values per quark. Per-channel deviations: m_u +15.5% (+0.68σ_PDG, within 1σ from PDG's large σ_u), m_d −1.40% (−0.14σ), m_s +2.72% (+0.30σ), m_c +0.56% (+0.35σ), m_b +2.15% (+2.99σ_PDG Clause-8 borderline).
- *Operations invoked.* M_persistence 12×12 block-diagonal assembly from chain-imported per-channel masses; δ(n) = 2/(9(n+1)) via Pati-Salam Fock counting + MDL equal allocation; walker-length dichotomy (low-scale m = v·y for L > 0; GUT-scale m = (v/√2)·y for L = 0 Type-II saturation); W3 PS sector connectivity closure (δ(n) via graph distance + 1 in PS sector graph L—D—U, edges algebraically verified on Cl(6) Fock).
- *Alternatives.* Different δ(n) prescriptions; non-PS Fock-mode-count identifications; different walker-length scale assignments.
- *Selection.* W3 PS sector-connectivity ↔ Fock-mode-count identity is the K-rational structural closure; alternative δ-formulas eliminated by Pati-Salam embedding constraint + MDL equal allocation. (Two-channel Koide cross-check: lepton n = 0 → 0.003%, down n = 1 → 0.85%, up n = 2 → 0.43%.)
- *Status.* **THEOREM-GRADE-STRUCTURAL-CONDITIONAL** for all five (graduated 2026-05-26 via M_persistence + W3 PS sector connectivity). Clause 8 PASS vs σ_PDG for m_c / m_s / m_d / m_u; borderline FAIL for m_b (+2.99σ); the m_b residual is MSSM-threshold + two-loop class, structurally identical to m_t (Row P38). Prior framing as "BLOCKED — IN PROGRESS" is RETIRED.
- *Margin.* All five within ~2% relative; four within 1σ_PDG; m_b borderline.
- *Conditional on.* Row P7 (y_τ chain), Row 17 (Pati-Salam embedding), Row P11 (m_τ scale via v), M_persistence theorem, W3 PS sector connectivity.
- *Gap.* MSSM-threshold + two-loop systematic floor for m_b (out-of-scope-by-construction per Move-1).
- *Filtered-alternative residue.* —

### Row P40 — α_GUT (bare 1/24 + Q-space dark correction graduated 2026-05-15; selection rule substrate-derived 2026-05-15 EOD+1)

- *Claim.* α_GUT_bare = 1/(2^k\* · k\*) = 1/24 ≈ 0.04167 (substrate label counting).  α_GUT_observed = α_GUT_bare × (1 − (1/k\*) · α_1/(1−α_1)) = 18659/453960 ≈ 1/24.329 (dark-corrected; **THEOREM-GRADE** via `../theorems/theorem_alpha_GUT_dark_correction.md`, graduated 2026-05-15 EOD+1).
- *Source.* `predictions/alpha_GUT.py`, `predictions/alpha_GUT_derivation.md`, `proofs/foundations/alpha_GUT_dark_correction_routes_HC_closure.py`.  Integrated RG closure (cluster Rows P65–P70) in `proofs/foundations/gauge_unification_full_RG_closure.py` + theorem doc `../theorems/theorem_gauge_unification_RG_closure.md`.
- *Observed.* MSSM single-regime back-extrapolation gives α_GUT⁻¹ ≈ 24.30 (uniform across i = 1, 2, 3).  Framework dark-corrected: 1/α_GUT_observed = 24.329 (deviation +0.13%, within sub-percent).
- *Operations invoked.* (Bare) Local CAR theorem on edge modes (Cl(2k\*) Fock dim 2^k\*); A1 + MDL gives k\* directed edges; A2-T + Jaynes uniform max-entropy; A5(b) MDL probability = coupling. (Dark correction) Q-space Σ_Q(h) = α_1·h̄/|h|² on uniform Q-space density (`predictions/uniform_Q_density_derivation.md`, theorem-grade); **observable-class selection rule SUBSTRATE-DERIVED** via `theorem_h1_master_compression.md` Theorems 1+2+3 (C¹ = B¹ ⊕ H¹; Wilson loops generate H¹) + Bass-Stark-Terras Hashimoto factorization (bipartite-factor marginal modes ↔ H¹ Wilson-loop sector, gauge-charged; Perron-adjacency u=+1 mode ↔ uniform B¹-residue, gauge-singlet); two derivation routes (Route H: Stark-Terras spectral; Route C: A2 edge-process cycle-counting) both giving c = 1/k\*. Earlier Peskin-Schroeder/Weinberg continuum-QFT citations retired 2026-05-15 EOD+1 in favor of substrate-aligned chain.
- *Alternatives.* Different label-counting structures; different Fock-vs-direction product orderings; non-uniform priors.  Dark-correction alternative c values blocked by calibration check (must reproduce v_Higgs c = 5/12 via same mechanism).
- *Selection.* Bare: Local label count = 2^k\* × k\*; A5(b) → α_GUT_bare = 1/24.  Dark correction: c_α_GUT = 1/k\* derived via Routes H + C (both passing v_Higgs calibration); applied per substrate-Feshbach-analog template `g_obs = g_bare × (1 − c · α_1/(1−α_1))`.
- *Status.* **UNIQUE — THEOREM-GRADE (bare) + THEOREM-GRADE (dark correction, graduated 2026-05-15 EOD+1)** on substrate-aligned conditions only: Stark-Terras factorization (Type 3 graph theory), Sunada cycle count (Type 3 graph theory), `theorem_h1_master_compression.md` (Type 4 framework-internal), Wilson 1974 lattice gauge theory (Type 3 substrate-aligned), A2-T + A4 + A5(b). The earlier "THEOREM-GRADE-CONDITIONAL" label was conditional on a Type 3 import from continuum gauge theory (Peskin-Schroeder § 4.7 + Weinberg QFT I § 8.1) for the observable-class selection rule (gauge 1-point excludes scalar zero-mode); that condition is now substrate-derived per `proofs/foundations/alpha_GUT_selection_rule_substrate_derivation.py`.
- *Margin.* Strict rational identity for bare and DC forms.  α_GUT⁻¹ = 24.329 matches PDG back-extrapolation cluster mean 24.30 to +0.13%.
- *Conditional on.* Row 4 (k\* = 3), Row 13 (A3-T), local CAR theorem (Row 15a), A5(b); dark correction additionally inherits Q-space Feshbach coupling Part B (currently ADOPTED, Feshbach coupling = α_1_bare).
- *Gap.* —
- *Cluster propagation status (2026-05-15).*  Dark-corrected α_GUT propagated to all cluster prediction files (P63–P71).  Resulting numerical match:
  - **g_1(M_Z) = 0.4615**: dev +0.011% (+0.51σ_PDG) — **PASSES Clause 8 vs σ_PDG.**
  - **α_EM(M_Z) = 1/127.92**: dev −0.016% (+1.47σ_PDG).
  - **sin²θ_W(M_Z) = 0.23126**: dev +0.020% (+1.15σ_PDG).
  - **g_2(M_Z) = 0.6518**: dev −0.038% (−2.46σ_PDG).
  - **R∞ = 1.0977e7**: dev +0.030% (after fixing stale δα⁻¹ running constant; was −1.15%).
  - **g_3(M_Z) = 1.2111, α_s(M_Z) = 0.1167**: dev −0.57%, −1.10% (known QCD-specific systematic — hadronic vacuum polarization at α_s extraction; separate from dark mechanism).
  - **M_Z = 91.51, m_W = 80.24**: dev +0.36%, −0.16% (intrinsic SM tree-vs-loop relation gap; closing requires standard SM 2-loop electroweak corrections, not framework derivation deficiency).
- *Filtered-alternative residue.* Pre-2026-05-15 cluster deviations (0.1–3%, 24–375σ_PDG vs σ_PDG) were attributed to ADOPTED-MSSM-Sb residue.  Now structurally identified as α_GUT dark correction (theorem-grade-conditional via Routes H+C); residuals remaining are SM-bridge issues (QCD hadronic VP, EW 2-loop), not parameter-derivation deficiencies.

### Row P41 — λ_Higgs = 2 · (5/3) · α_1 = 2560/19683 ≈ 0.13006

- *Claim.* Higgs quartic λ = n_channels · tan²(arg h) · α_1_bare = 2 · (5/3) · (2/3)^8 = 2560/19683 ≈ 0.13006. (Bug-fix 2026-05-02: previously stated `2 · α_1_full = 2 · 256/6305 = 512/6305 ≈ 0.13006`, but 512/6305 = 0.0812 — the fraction expansion was wrong because the symbol "α_1_full" is overloaded across the framework: Row P41 / `lambda_higgs.py` uses α_1_full(dark-corrected) = (5/3)·α_1 = 1280/19683 ≈ 0.0650, while V_cb / V_ub use α_1_full(geometric) = α_1/(1-α_1) = 256/6305 ≈ 0.0406. The keeper script `predictions/lambda_higgs.py` derives λ as `n_channels × c_mass × α_1` = `2 × (5/3) × (2/3)^8` directly; the fraction is 2560/19683 = 0.130056. Numerical claim 0.13006 was correct; the formula expansion was wrong.)
- *Source.* `predictions/lambda_higgs.py` Step 7: `lam = n_channels * c_mass * alpha_1_bare = 2 * (5/3) * 256/6561 = 2560/19683`.
- *Observed.* PDG 2024 (from m_H + v): λ ≈ 0.1294. Match: ~0.5%.
- *Operations invoked.* Op 5.8 (Cl(2;ℂ) per-edge), Op 5.17 (tensor product); Cl(2) channel structure on per-edge SU(2) qubit (`../theorems/theorem_g2_edge_qubit_su2.md`); tan²(arg h) = Im(h)²/Re(h)² = 5/3 from h = (√3+i√5)/2.
- *Alternatives.* Different prefactors (1·c_mass·α_1, 4·c_mass·α_1, etc.); different α_1 vs α_1_full(geom) choice.
- *Selection.* Cl(2) edge-qubit channel structure gives the factor of n_channels = 2 (per `theorem_g2_edge_qubit_su2.md` + `theorem_G2_cl2_channels.py`); tan²(arg h) = 5/3 is exact algebra from h saddle (Step 5a of `lambda_higgs.py`); α_1 = (2/3)^(g-2) is the bare combinatorial NB walk survival. Combined: λ = 2·(5/3)·α_1.
- *Status.* **THEOREM-GRADE-STRUCTURAL, conditional** (corrected W1 2026-05-18; the 2026-05-15 "UNIQUE-NUMERICAL / all four routes closed" was a Clause-6c smuggle): conditional on Row P2 + Cl(2) edge-qubit theorem (Row 20 structural + Row 22 pseudoscalar orientation) + Family D on the |φ|⁴ vertex (`../theorems/theorem_substrate_feshbach_dark_corrections_master.md` §3 (D)). λ rides **c_H ONLY** (4 Higgs legs, 0 fermion legs — the c_F fermion-leg conditional does NOT enter); c_H = Route H structurally derived + Route C corroboration (genuinely two independent routes). Conditional only on the c_H (g-2) joint-excursion assumption. Tree-level λ = 2560/19683 ≈ 0.130062; Family D-corrected λ_physical = λ_tree × (1 - 4·α₁_bare²) ≈ 0.129269 vs observed λ_obs = m_H²/(2v²) ≈ 0.129281 → **-0.05σ_PDG (PASS Clause 8)**. `predictions/lambda_higgs.py` propagated 2026-05-15.
- *Margin.* Strict for the form; tree-level residual was +0.5% (FAIL Clause 8 vs σ_PDG); Family D-corrected residual is -0.008% rel.err = -0.05σ_PDG (PASS Clause 8 vs σ_PDG).
- *Conditional on.* Row P2 (α_1), Row 20 (Higgs doublet), Row 22 (Cl(2) orientation), Row P3 (h saddle eigenvalue → tan²(arg h) = 5/3).
- *Gap.* — (the m_H residual analysis sits at Row P12; λ itself is closed at this layer).
- *Filtered-alternative residue.* —
- *Naming-collision note (2026-05-02).* The framework uses two distinct quantities both named "α_1_full" in different contexts: (i) `α_1_full(geometric) = α_1/(1−α_1) = 256/6305 ≈ 0.0406` (V_cb keeper convention; Feshbach geometric series resummation), and (ii) `α_1_full(dark-corrected) = (5/3)·α_1 = 1280/19683 ≈ 0.0650` (lambda_higgs.py convention; tan²(arg h) dark-map Class 2 prefactor). These should be disambiguated in any audit-formula scanning. Row P41 uses (ii); Row P14 (V_ub) and Row P7 (y_τ) use (i).

### Row P42 — η_5 (dim-5 Lorentz violation) = 0 exactly

- *Claim.* η_5 = 0 exactly (no dim-5 LV from the framework).
- *Source.* `predictions/eta_5_lorentz_dim5.py`.
- *Observed.* LHAASO 2024 + others: |η_5| < 0.1 (consistent).
- *Operations invoked.* Stage 3 spatial setup; Hashimoto Bloch dispersion has no odd-power-in-k contribution at quadratic order (parity-even dispersion).
- *Alternatives.* Non-zero η_5 from explicit parity-violating dim-5 operator coefficients.
- *Selection.* The framework's effective dispersion is parity-even at dim-5 (no odd-k³ term); the I4₁32 lattice has parity-odd parameters elsewhere (chirality, weak interactions) but the dispersion relation itself is parity-even at this order.
- *Status.* **UNIQUE — THEOREM-GRADE** exact 0 (modulo the Stage 3 continuum-limit premise).
- *Margin.* Strict equality; well within LHAASO bound.
- *Conditional on.* Stage 3 (continuum / Lorentz-causal-sector theorem), `theorem_lorentz_causal_sector.md`.
- *Gap.* —
- *Filtered-alternative residue.* —

### Row P43 — η_lattice (dim-6 LV) = 1/12

- *Claim.* η_lattice = D4_aniso / D_NB² = (1/768) / (1/8)² = 1/12 ≈ 0.0833.
- *Source.* `predictions/eta_lattice_lorentz_dim6.py`, `predictions/eta_lattice_lorentz_dim6_derivation.md`. Closure: `proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py` (Feshbach-Löwdin SYMBOLIC, sympy 4s) + `proofs/foundations/lorentz_sig_ihara_lv_relation.py` (Ihara cross-walker theorem).
- *Observed.* No current measurement; bound |η_6| < ~10¹⁴–10¹⁸ from LHAASO + UHE-photon transparency. Framework value 1/12 is ~16 orders below current sensitivity.
- *Operations invoked.* Stage 3 dispersion expansion; Hashimoto Bloch on srs primitive cell; Ihara identity u² − λu + 2 = 0 for 3-regular graphs (h'(3) = 2, h''(3) = −4); Feshbach-Löwdin partition closing the scalar-Bloch quartic coefficients D4_iso^H = −1/1024, D4_aniso^H = +1/1536 symbolically; Ihara cross-walker maps to Hashimoto: D_NB = h'(3)·D_H = 1/8, D4_aniso^NB = h'(3)·D4_aniso^H = 1/768.
- *Alternatives.* Different dispersion expansions; different cross-walker maps; non-rational η_6.
- *Selection.* The Ihara cross-walker theorem maps the symbolic scalar-Bloch expansion to the Hashimoto Bloch expansion uniquely; D_NB and D4_aniso are CAS-verified to 24+ digits (cross-check) and now SYMBOLIC theorem-grade.
- *Status.* **UNIQUE — THEOREM-GRADE SYMBOLIC** (post-2026-04-27 Lorentz arc symbolic closure).
- *Margin.* Strict rational identity; far below current observational sensitivity.
- *Conditional on.* Stage 3 spatial setup, Ihara 1966 / Stark-Terras 1996 (Type 3); A1 + Cl(6) Fock + srs geometry.
- *Gap.* —
- *Filtered-alternative residue.* The ~16-order-of-magnitude gap to current sensitivity is itself a sharp falsifiable prediction class — future SWGO / UHE-photon experiments could close this.

### Row P44 — β cosmic birefringence (DOWNGRADED 2026-05-16 UNIQUE-THEOREM-GRADE → THEOREM-GRADE-STRUCTURAL: observed-α_EM smuggle removed)

- *2026-05-16 — GRADE CORRECTED (no-observed-input-where-prediction-expected rule, user directive; DAG node is the authority).* `predictions/beta_cosmic_birefringence.py` was substituting the **observed** α_EM = 1/137.036 — a smuggle: α_EM is a framework prediction (`predictions/alpha_EM.py`), and the framework α_EM being imperfect/blocked does NOT license the observed value. The node is now wired to the FRAMEWORK α_EM (alpha_EM.py, α_EM(M_Z), zero observed input). The framework cannot yet derive α_EM(0) — the M_Z→0 Δα running is Clause-9-BLOCKED (`substrate_Delta_alpha_blocked_verdict_2026-05-16.md`). **New status: THEOREM-GRADE-STRUCTURAL** — the FORM β=c·sin(arg h)·α_EM, c=1, sin(arg h)=√(5/8) stays theorem-grade; the NUMBER is framework-α_EM-conditional + a named α(M_Z)→α(0) Δα gap. Honest value **β = 0.354°, +0.13σ vs Eskilt 2022** (still <1σ — removing the smuggle did NOT blow β up; it replaces the retracted observed-α_EM 0.331°/−0.12σ). The prior UNIQUE-THEOREM-GRADE / 0.331° is RETRACTED as observed-α_EM-smuggled.
- *Claim (form, theorem-grade).* β = 1 · sin(arg h) · α_EM; with FRAMEWORK α_EM(M_Z) → 0.354° (was the retracted 0.331° via observed α_EM).
- *Source.* `../theorems/theorem_cosmic_birefringence.md` (status THEOREM-GRADE 2026-04-29 via uniqueness closure); `../theorems/theorem_beta_uniqueness_closure.md` (uniqueness closure 2026-04-29); `../theorems/theorem_lattice_coupling_algebraicity.md` (algebraicity meta-theorem upgrade).
- *Observed.* Eskilt 2022: 0.342° ± 0.094°. Deviation: +0.12σ.
- *Operations invoked.* Uniqueness template via three premises: P1 (D1 substrate-chirality-as-unique-spatial-parity-source), P2 (MDL Lemma 1 selects sin(arg h) as cheapest parity-odd dimensionless functional), P3 (D2 + algebraicity meta-theorem rules out 1/(16π²) factor since π ∉ K = ℚ(√2,√3,√5) by Lindemann 1882). Composition forces c=1.
- *Alternatives.* Different parity-odd functionals of h closed by P2 MDL Lemma 1 (theorem-grade); 1/(16π²) coefficient closed by P3 algebraicity (theorem-grade); other coefficients ruled out by no-other-factor-available argument. The 8 prior bounded routes (L3-tree, P4 Cl(6,0)/B6, L3-trace-survey, Q ∂_kB 1-loop, F0_γ no-go, Q' Berry numerical, Q' analytic, resolvent) all attempted local-mechanism computation; uniqueness argument bypasses by structurally constraining rather than computing.
- *Selection.* c=1 forced by P1+P2+P3 composition (theorem-grade per `theorem_beta_uniqueness_closure.md`).
- *Status.* **UNIQUE-THEOREM-GRADE** (graduated 2026-04-29 via β uniqueness closure + algebraicity meta-theorem upgrade; commit 3aaa473). Audit history: BLOCKED → STRUCTURAL-DERIVATION-GRADE (2026-04-29 uniqueness template) → **UNIQUE-THEOREM-GRADE** (2026-04-29 evening via algebraicity meta-theorem Path B closure).
- *Margin.* +0.12σ from Eskilt 2022.
- *Conditional on.* P1 (D1 substrate-chirality audit); P2 (MDL Lemma 1, theorem-grade in repo); P3 (algebraicity meta-theorem `theorem_lattice_coupling_algebraicity.md`, theorem-grade); h theorem-grade per `walker_dynamics_derivation.md`.
- *Gap.* —. (Closed at theorem grade. Uniqueness argument has the same structural shape as MDL Lemma 1.)
- *Filtered-alternative residue.* —

### Row P45 — J_CKM: UNIQUE-THEOREM-GRADE for amplitude form; labeling data-anchored (inherits Row P14)

- *Claim.* J_CKM = Im(V_us · V_cb · V*_ub · V*_cs) — Jarlskog invariant. Computable algebraically from Rows P3, P4, P14, P15.
- *Source.* `target_parameters.md` CKM table. Inherits the V_ub family graduation 2026-04-30; bridge functoriality graduation 2026-04-28 RETRACTED 2026-04-29 (no longer needed; superseded by M1 amplitude-form closure).
- *Observed.* PDG 2024: 3.08×10⁻⁵.
- *Operations invoked.* Standard Jarlskog construction on derived CKM entries; inherited M1 twisted-walker amplitude form.
- *Alternatives.* Various Jarlskog-on-K_4 holonomy constructions (per Row P15 geometric note).
- *Selection.* Direct algebraic combination of Rows P3 (V_cb), P4 (V_us), P14 (V_ub UNIQUE-THEOREM-GRADE for amplitude; labeling data-anchored), P15 (δ_CP_CKM UNIQUE-THEOREM-GRADE for geometric value; labeling data-anchored).
- *Status.* **UNIQUE-THEOREM-GRADE for amplitude form; labeling data-anchored, non-blocking for predictive content** (graduated 2026-04-30 via inheritance from Rows P14, P15 M1 amplitude-form + Angle D + Z3-mass-order verdicts). **2026-05-05 EOD+3 strengthening:** inherits P14, P15 ADOPTED-B3 hypercharge graduation via G2-D closure (`theorem_g2d_chirality_doubled.md`).
- *Margin.* Numerical evaluation possible from upstream rows; deferred to a separate prediction file.
- *Conditional on.* Rows P3, P4, P14, P15. (Inherits P14 + P15 conditional structure including ADOPTED-B3 hypercharge graduation 2026-05-05 EOD+3.)
- *Gap.* Closed at predictive-content level. Labeling layer is OTHER-SMUGGLE residue inherited from Row P14, not row-specific.

### Row P46 — tan β: live RGE chain ≈ 60.07; documented value 44.73 (DISAGREEMENT surfaced 2026-05-26)

- *Claim (live chain).* The live MSSM RGE chain (`predictions/tan_beta.py`) computes **tan β ≈ 60.07** as the root of the bottom-tau Yukawa unification + GJ = k* = 3 self-consistency equation.
- *Claim (documented framework value).* `proofs/masses/srs_tan_beta.py` and historic framework documentation quote **tan β = 44.73**.
- *Source.* `predictions/tan_beta.py`, `predictions/tan_beta_derivation.md` (live chain); `proofs/masses/srs_tan_beta.py` (documented value).
- *Observed.* Not directly observed; inferred from MSSM fits that assume b–τ unification.
- *Disagreement (surfaced 2026-05-26).* Until 2026-05-26 commit `d590580`'s pure-function-literal audit, the live chain returned 44.73 via an `except (ValueError, RuntimeError): return 44.73` exception-fallback in `_solve_tan_beta()`. The brentq solver in the (10, 60) bracket consistently failed to find a root (residual was monotonically negative across the entire range), and the fallback silently returned the documented value. Probing the residual revealed the actual root sits at tan β ≈ 60.07 in bracket (60, 65); the documented 44.73 gives y_b/y_τ(GUT) ≈ 1.32, far from the GJ target k* = 3. The fallback was removed and the bracket widened to (10, 65) — the live chain now surfaces the honest 60.07 value. The 44.73 documented value originates in `proofs/masses/srs_tan_beta.py` using different boundary assumptions (likely earlier 1-loop or different y_t M_Z choice).
- *Operations invoked.* GJ = 3 (Row P48 + Row 4, theorem-grade); bottom-tau Yukawa unification at M_unif (standard MSSM); framework MSSM 1-loop RGE infrastructure; combined with y_τ (Row P7) + α_GUT (Row P40) + M_unif (Row P62) + y_t(M_Z) ≈ 0.95 IR fixed point + brentq root-finding.
- *Alternatives.* The disagreement between live chain (60.07) and documented value (44.73) could be due to: (a) different RGE order — live chain is 1-loop, srs_tan_beta.py may have 2-loop or threshold corrections; (b) different y_t(M_Z) boundary (0.95 IR-FP vs measured); (c) different α_GUT or M_unif used as inputs; (d) different y_b(M_Z) low-scale value (Type IV selection rule).
- *Selection.* The live chain's selection is the brentq root of `y_b(GUT)/y_τ(GUT) = k*`. Documented chain's selection mechanism needs investigation.
- *Status.* **🟡 STRUCTURAL-DERIVATION-CONDITIONAL** (downgraded 2026-05-26 from THEOREM-GRADE-STRUCTURAL-CONDITIONAL). The structural form (GJ + b-τ unification + MSSM RGE) is theorem-grade; the specific numeric value is currently in DISAGREEMENT between two derivations within the framework. Until reconciled, the live chain's 60.07 is the published prediction; the 44.73 documented value carries its own derivation chain in `proofs/masses/srs_tan_beta.py` that needs auditing.
- *Margin.* — (not directly observed).
- *Conditional on.* MSSM commitment; Row P7 (y_τ); Row P40 (α_GUT); Row P62 (M_unif); RGE order assumption (1-loop in live chain); y_t(M_Z) boundary choice.
- *Gap.* **DOCUMENTED-VS-LIVE RECONCILIATION** — open work. Either fix the live chain to match the documented 44.73 (by changing boundary assumptions or RGE order), or update the framework documentation to reflect the live chain's 60.07, or identify which calculation is correct and retire the other.
- *Filtered-alternative residue.* The literal-fallback `return 44.73` pattern that masked this disagreement for ~3 weeks (between commit `c9fba27` 2026-05-26 ship and audit commit `d590580` 2026-05-26) is the canonical example of why exception-fallbacks with hardcoded answers are smuggles.

### Row P47 — m_W and m_Z absolute (RG-conditional, mathematically complete)

- *Claim.* m_W = g_2 · v / 2; m_Z = √(g_1² + g_2²) · v / 2. Conditional on Row P10 (v = 246.22 GeV) + g_1, g_2 absolute at M_Z (still 🟡 pending RG running per Row P40 residue).
- *Source.* `target_parameters.md`; absolute values not currently shipped at theorem grade.
- *Observed.* PDG 2024: m_W = 80.369 GeV, m_Z = 91.188 GeV.
- *Operations invoked.* Standard SM tree-level relations; Row P10 (v calibration); Row P40 (α_GUT) + RG running.
- *Alternatives.* — (algebraic; once g_1, g_2 fixed, masses follow).
- *Selection.* Standard SM identification.
- *Status.* **MATHEMATICALLY COMPLETE conditional on v + g_1, g_2 at M_Z.** Inherits Row P10's G1 + Row P40's RG-running gap.
- *Margin.* — (consistent with PDG once gauge couplings are RG-run from M_unif).
- *Conditional on.* Row P10 (v), Row P40 (α_GUT), RG-running calculation (Priority 2.1 downstream).
- *Gap.* RG-running step (mathematically-complete with M_Z external; standard textbook).
- *Filtered-alternative residue.* —

---

## Framework-internal derivables (compact rows — direct corollaries of structural ledger rows)

The following rows record framework-internal predictions whose value is fully determined by upstream structural ledger rows. They are corollary-grade rather than independent derivations; status follows directly from the cited structural rows.

### Row P48 — k\* = 3 (coordination number)

- *Claim.* k\* = 3. *Source.* `predictions/k_star.py`. *Observed.* Not directly observable (lattice parameter); its consequences (e.g., Ω_DM/Ω_m, Q_Koide) are observable.
- *Selection.* Brown 1986 Fisher rank for d=3 + reticular chemistry MDL filter; same chain as structural Row 4.
- *Status.* **UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 Brown-rank closure** (graduated 2026-04-30 EOD; reframed 2026-05-01 PM per an internal note REVISED). **Primary closure (structural):** Brown 1986 Fisher rank for d=3 — k > d gives Fisher rank zero on excess edges → MDL eliminates strictly (per `predictions/k_star.py` + an internal working note §1). M6 sign-gate (Re(h_qtz_Γ) = −1 structurally forced at Γ via 4-regular + 3-vertex C_3 + Hermiticity) provides structural supporting evidence at the η_B-specific level. **Supplementary empirical validation (NOT closure mechanism):** Phase 1a M2 = +1.14 bits weak soft gate (`uniqueness_audit_v2_phase_1a_sign_gate_correction_2026-04-30.md`); data-conditional MDL crush ~2×10⁸ bits across observables (`uniqueness_audit_v2_data_conditional_mdl_2026-04-30.md`). These confirm the structural exclusion of qtz aligns with PDG, but do not themselves provide closure. Earlier framing of data-conditional MDL as the load-bearing closure was goal-seeking and is RETRACTED 2026-05-01 PM. Audit v2 history: pre-v2 UNIQUE-THEOREM-GRADE → DOMINANT-with-named-margins (Phase 1a M2 weak soft gate) → UNIQUE-THEOREM-GRADE-CONDITIONAL via data-conditional MDL (2026-04-30 EOD; RETRACTED 2026-05-01 PM as goal-seeking) → **UNIQUE-THEOREM-GRADE-CONDITIONAL on Brown rank (2026-05-01 PM)**.
- *Conditional on.* Row 3 (d_spatial = 3), Row 4 audit v2 closure (Brown-rank structural).

### Row P49 — d_spatial = 3 (spatial dimension)

- *Claim.* d_spatial = 3. *Source.* `predictions/d_spatial.py`. *Observed.* 3 (definitional structural match).
- *Selection.* Cencov-Fisher information geometry argument; structural Row 3.
- *Status.* **UNIQUE — THEOREM-GRADE** (corollary of Row 3). The +1 time dimension is intrinsic to A1's stream-length grading per R-4 closure (residue register).
- *Conditional on.* Row 3 (Cencov-Fisher).

### Row P50 — g_girth = 10 (srs girth)

- *Claim.* g = 10. *Source.* `predictions/g_girth.py`. *Observed.* — (lattice quantity; consequences observable via α_1, V_cb, η_lattice etc.).
- *Selection.* Sunada 2012 srs identification; structural Row 9.
- *Status.* **UNIQUE — THEOREM-GRADE** (corollary of Row 9).
- *Conditional on.* Row 6 (srs), Row 9 (girth).

### Row P51 — p_toggle = 2 (binary alphabet)

- *Claim.* p_toggle = 2. *Source.* `predictions/p_toggle.py`. *Observed.* — (definitional A1 alphabet).
- *Selection.* A1 binary self-inverse toggle; structural Row 1 (post-R-1 REFUTATION which closes higher-arity alternative).
- *Status.* **UNIQUE — THEOREM-GRADE** (corollary of Row 1, post R-1 refutation).
- *Conditional on.* A1 algebraic content; Row 20 (Cl(2) per-edge structure for the R-1 hard-gating).

### Row P52 — h_walker_eigenvalue at k_P = (√3 + i√5)/2

- *Claim.* The Hashimoto walker's principal eigenvalue at the high-symmetry P point is (√3 + i√5)/2 (Ramanujan-saturating). *Source.* `predictions/h_walker_eigenvalue.py`, `predictions/srs_E_at_P.py`.
- *Selection.* Hashimoto operator on srs primitive cell → Ramanujan saturation |h|² = k\* − 1 = 2 at the symmetric P point; the imaginary part √5 / 2 follows from the Cl(2;ℂ) splitting on the K_4 quotient.
- *Status.* **UNIQUE — THEOREM-GRADE** (corollary of Rows 6, 8, and Stark-Terras 2007 NB-walker theorems).
- *Conditional on.* Row 6, Row 8, Stark-Terras 2007.

### Row P53 — srs_E_at_P = √3 and srs_cubic_moment = 1/(3·2^{n−1})

- *Claim.* srs_E_at_P = √3 (single-walker amplitude); srs_cubic_moment ⟨(ê·ẑ)²ⁿ⟩ = 1/(3·2^{n−1}) at the P-point.
- *Source.* `predictions/srs_E_at_P.py`, `predictions/srs_cubic_moment.py`.
- *Selection.* P2 Theorem 1 + 432-cubic moment evaluation on srs.
- *Status.* **UNIQUE — THEOREM-GRADE** (corollary of Rows 6, 7, 8).
- *Conditional on.* Row 6 (srs), Row 7 (|E|=6), Row 8 (|V|=4).

### Row P54 — Stage 2c bundle: λ_toggle_rate = 2/5, ξ_t = 1/log(6) ℓ_P, S_fresh = 1 bit, S_disconfirm = log₂(3)

- *Claim.* The four Stage 2c arrow-of-time + edge-surprise quantities take the listed exact rational/logarithmic values.
- *Source.* `predictions/lambda_toggle_rate.py`, `predictions/xi_t_temporal_correlation.py`, `predictions/S_fresh.py`, `predictions/S_disconfirm.py`.
- *Selection.* Stage 2c arrow-of-time theorem; A1 + A2-T waterline + Bayesian posterior.
- *Status.* **UNIQUE — THEOREM-GRADE** for all four (corollaries of Rows 1, 11; Stage 2c).
- *Conditional on.* Row 1, Row 11, Stage 2c.

### Row P55 — Scale energy (Hashimoto) ~147 PeV; Universe-transparency onset ~147 PeV

- *Claim.* Two independent observable scales in PeV-class astroparticle physics, set by the same Hashimoto walker scale.
- *Source.* `predictions/scale_energy_hashimoto.py`, `predictions/universe_transparency.py`.
- *Observed.* Beyond current reach; SWGO target. GRB 221009A LHAASO observations marginally consistent.
- *Selection.* Hashimoto eigenvalue + Stage 3 dispersion → cutoff scale ~ √(M_P · m_v) class (combination of Planck mass and the framework's natural energy scale at the dispersion roll-off).
- *Status.* **UNIQUE — THEOREM-GRADE** structural derivation (corollary of P52, Stage 3); numerical value carries Stage 3 continuum-limit premise.
- *Conditional on.* Rows 6, 14 (continuum limit unitary leg); Stage 3 closure.

### Row P56 — Branch measure μ uniform; Observer Hilbert space H = C³

- *Claim.* μ on multiway tree is the uniform product measure (Stage 1 closure); observer Hilbert space H = C³ with axioms (G.1, G.5).
- *Source.* `../theorems/theorem_multiway_branch_measure.md`, `predictions/observer_dim_three.py`, `predictions/observer_hilbert_space.py`.
- *Selection.* Stage 1 branch-measure theorem (Session 9); CDP 2011 axioms G.1 + G.5 derived from A1 + observer setup; structural Row 12 (μ) + Row 18 (C³).
- *Status.* **UNIQUE — THEOREM-GRADE** for both (corollaries of Rows 12 and 18).
- *Conditional on.* Rows 12, 13, 18.

### Row P57 — Structural/definitional bundle: gauge group, n_gen, charge quantization, parity, fermion content, Higgs rep, Lorentzian signature

- *Claim.* SU(3)×SU(2)×U(1) gauge group; n_gen = 3; Q = n/3 charge quantization; chiral parity violation; 48 fermion states per family (Cl(6) spinor + 3 generations + antipartners); Higgs rep (1, 2, +1/2); spacetime Lorentzian signature (−, +, +, +).
- *Source.* `target_parameters.md` Structural / definitional table.
- *Selection.* Each is a direct corollary of structural ledger rows: gauge group from Row 19; n_gen from Row 18 (mathematically complete); Q = n/3 from Row 16 (Cl(6) weights); parity from R-12 chirality residue (ACCOUNTED-FOR + structural filter); fermion content from Rows 16 + 18; Higgs rep from Rows 17 + 18 + 20 + ADOPTED-B3; Lorentzian signature from Stage 3 + the 2026-04-27 Lorentz-arc closure (Row 14b STRUCTURALLY CLOSED, NUMERICALLY OPEN on G_sub).
- *Status.* **MOSTLY UNIQUE — THEOREM-GRADE** (corollaries); Higgs Y = +1/2 hypercharge labeling and SU(2)_L chirality remain CONDITIONAL on ADOPTED-B3 (per Row P18).
- *Conditional on.* Rows 16, 17, 18, 19, 20, R-12; Stage 3 + Lorentz-arc closure; ADOPTED-B3 for the labeling questions.

### Row P58 — SUSY spectrum (RETIRED-conditional 2026-05-27)

- *Claim.* m_gluino, m_squark, m_slepton, m_neutralino, m_chargino, m_h, m_H, m_A, m_H± — the 9-row MSSM spectrum.
- *Source.* `target_parameters.md` SUSY table; mostly ❌. Honest-conditional values listed in `predictions.md` §SUSY Spectrum.
- *Status (revised 2026-05-27 post-SUSY-load-bearing audit).* **RETIRED-conditional.** The framework's substrate-derived matter content is 3 PS generations + 2 Higgs doublets (no superpartners — Cl(6) Fock all-fermionic per Path-E recheck 2026-05-12; A1 thermal-apparatus closure 2026-05-27). These 9 sparticle masses are predictions ONLY under the literal-particle interpretation of [ADOPTED-MSSM-Sb](../audits/registers/adoption_register.md), which is not substrate-derived. Branch A's bounded research routes (A1, A3, A4) exhausted 2026-05-27 without finding a substrate-side derivation. Framework does not commit to literal-particle realization; if literal SUSY exists, the spectrum would follow the m\_{3/2} = M\_P / √(N^(1/2)) form in `predictions.md` §SUSY Spectrum, but if not, these rows do not apply. No framework prediction in `predicted_parameters.md` depends on these values.
- *Margin.* — (sentinel).
- *Conditional on.* Literal-particle interpretation of ADOPTED-MSSM-Sb (not substrate-derived). Row P46 (tan β) reconciliation pending.
- *Gap.* Literal-particle interpretation has no substrate-derivation route after Branch A exhaustion; tracked in [R-19](../audits/registers/structural_residue_register.md) as the Δb_2 = +4 SU(2)_L gap characterization.

### Row P59 — Matter stability and low initial entropy (definitional/structural)

- *Claim.* Matter is stable (no proton decay above bounds); initial-state entropy is low (cosmological boundary condition).
- *Source.* `target_parameters.md` Structural / definitional table.
- *Selection.* Matter stability follows from Pati-Salam SU(4) embedding (no proton-decay channels at observable rates given the framework's M_GUT scale). Low initial entropy from Stage 2c + arrow-of-time theorem (the framework's MDL projection runs forward in stream-length time, sourcing the Past Hypothesis without an extra postulate).
- *Status.* **STRUCTURAL — DEFINITIONAL ⚙️** in target_parameters.md; closure via Rows 17 (Pati-Salam) + Stage 2c (arrow-of-time).
- *Conditional on.* Row 17, Stage 2c.

### Row P60 — G_N · M_Pl² = 1 (Newton's constant, dimensionless identity derived from substrate dynamics)

- *Claim.* The Planck-units convention $G_N \cdot M_{\rm Pl}^2 = 1$ emerges as a **derived identity** from substrate dynamics, not as a definitional choice. Specifically: Drude UV asymptote $G_{\rm UV} \cdot M_{\rm substrate}^2 = \pi/64$ (theorem-grade per audit v2 PASS) combined with path (b) substrate-Planck reframing $M_{\rm Pl}/M_{\rm substrate} = 8/\sqrt{\pi}$ (theorem-grade) yields $G_{\rm UV} \cdot M_{\rm Pl}^2 = (\pi/64) \cdot (64/\pi) = 1$ exactly. Identifying $G_N = G_{\rm UV}$ (UV asymptote = laboratory limit under asymptotic safety) gives $G_N \cdot M_{\rm Pl}^2 = 1$.
- *Source.* `predictions/G_N.py`, `predictions/G_N_derivation.md` (added 2026-04-30 EOD final).
- *Observed.* CODATA 2018: $G_N = 6.67430(15) \times 10^{-11}\,{\rm m}^3/({\rm kg}\cdot{\rm s}^2)$, equivalently $M_{\rm Pl} = 1.22089(6) \times 10^{19}$ GeV/c². Dimensionless: $G_N \cdot M_{\rm Pl}^2 = 1$ by Planck-units convention. Framework match: the dimensionless identity is exact (theorem-grade); the dimensional value (SI) round-trips CODATA $M_P$ at machine precision (since $M_P$ is the framework's external dimensional anchor).
- *Operations invoked.* Drude form (Op 4.45-4.48 finite-(ω,T) Kubo formalism; theorem-grade Step 1+2); path (b) substrate-Planck reframing; asymptotic-safety identification G_N = G_UV (structurally conjectural, consistent with K[π] form).
- *Alternatives.* (i) Hashimoto-Sakharov candidate $729\sqrt{3}/(128\pi^2)$ at 0.05% match — FAILED audit v2; (ii) Multiway $G_{\rm sub} = 4/\pi$ — refuted by K-meta-theorem; (iii) Heat-kernel candidates ($1/(8\pi^3)$ etc.) — retracted per Drude doc.
- *Selection.* Drude form passes audit v2; uniquely selected by M5 (gravity-vs-gauge mechanism distinction) + M6 (Bloch spectrum) + log-form ruled out. Class A multi-collapse of $D = -1/36$ across 5 K-readings acknowledged but NOT a vulnerability — closure rests on Kubo computation. Step 3 path (a) "$\omega_{\rm obs}$ near pole" reclassified PHANTOM (unit-mixing artifact, an internal working note).
- *Status.* **UNIQUE — THEOREM-GRADE-CONDITIONAL** on G_sub Drude form audit v2 PASS + path (b) reframing + asymptotic-safety identification ($G_N = G_{\rm UV}$ under UV-IR fixed-point dominance, structurally conjectural for static limit beyond leading-order). Dimensionless identity $G_N \cdot M_{\rm Pl}^2 = 1$ theorem-grade exact; dimensional SI value inherits CODATA precision via $M_P$ external anchor.
- *Margin.* Dimensionless: 0 (exact match by theorem). Dimensional: round-trip identity at machine precision via CODATA $M_P$ (~50 ppm precision floor from CODATA).
- *Conditional on.* Row P14 G_sub closure (Drude form theorem-grade); Row 25 (substrate-Planck ratio derived); audit v2 (Drude PASS); asymptotic-safety conjecture ($G_N = G_{\rm UV}$).
- *Gap.* Asymptotic-safety identification (whether laboratory $G_{\rm static}$ beyond-LO equals $G_{\rm UV}$) is consistent with K[π] form but not independently derived. This is the structural physics question, multi-session research-level. The dimensional SI value's reduction below CODATA precision would require an alternative unit-setting constant (e.g., calibrating via G_F at 0.51 ppm) but G_F is currently unused for G_N's chain — a follow-up could establish a G_N prediction independent of M_P.
- *Filtered-alternative residue.* The Hashimoto-Sakharov candidate at 0.05% match was the most-recently-attempted single-number closure; FAILED audit v2 (DOMINANT-CONDITIONAL-GAP). The L_grav and X prefactor selections were matched-to-observation, not gated-by-mechanism. See an internal working note. Multiway $4/\pi$ refuted by K-meta-theorem. Heat-kernel candidates retracted.
- *Audit v2 (Clause 7) status.* Cite an internal working note §3.5 (G_sub Drude closure) for full §3 table. (7a) Axes enumerated: L_grav, X prefactor, Re(h_P), multiplicative form, class assignment, skeleton route. (7b) Alternatives named: L_grav ∈ {4,6,7,8,10}; X ∈ {π/24, π/12, π/8, 3π/16, 5π/12}; multiplicative skeletons (Drude vs Hashimoto-Sakharov vs heat-kernel vs multiway). (7c) Six-mechanism gating: see index §3.5.1 — M4 K-refutes multiway; M5 distinguishes gravity Drude pole from gauge A2 waterline; M6 confirms Bloch ⟨Tr H²⟩=12 → ω_pole = π/12. (7d) Combined: Drude form THEOREM-GRADE-COMPUTED; competitors fail. (7e) Status: UNIQUE-THEOREM-GRADE-CONDITIONAL on asymptotic-safety identification.

### Row P61 — M_substrate/M_Pl = √π/8 (substrate-Planck mass ratio, theorem-grade dimensionless)

- *Claim.* The dimensionless ratio between the substrate's natural mass scale and the Planck mass is **M_substrate/M_Pl = √π/8 ≈ 0.222** (equivalently M_Pl/M_substrate = 8/√π ≈ 4.51). Substrate scale is **below** Planck mass; substrate length is **4.51× Planck length** (~7.3×10⁻³⁵ m). Derived from G_sub Drude UV asymptote $G_{\rm UV} \cdot M_{\rm substrate}^2 = \pi/64$ combined with Planck convention $G_N \cdot M_{\rm Pl}^2 = 1$ via path (b) reframing. Theorem-grade dimensionless; **no external anchor required, no cosmological N-dependence**.
- *Source.* `../theorems/theorem_g_sub_drude_closure_2026-04-30.md` path (b); `../theorems/theorem_dimensionless_ratio_principle_2026-04-30.md` (meta-principle); `predictions/G_N.py` (round-trip test).
- *Observed.* The substrate scale is not directly observed — it's a framework-internal quantity. The ratio is testable indirectly via consistency of dimensional predictions across the parameter ledger.
- *Operations invoked.* Drude form (Op 4.45-4.48 finite-(ω,T) Kubo on Bloch operator); path (b) substrate-Planck reframing; Planck convention $G_N \cdot M_{\rm Pl}^2 = 1$.
- *Alternatives.* (i) M_substrate = M_Pl exactly — was the pre-2026-04-30 EOD Row 25 commitment, superseded by this derivation. (ii) M_substrate/M_Pl = 8/√π (inverted reading) — Drude doc's earlier text had this inversion; algebraically wrong. (iii) Multiway G_sub_multiway = 4/π — refuted by K-meta-theorem.
- *Selection.* Direct algebraic consequence of Drude form (theorem-grade per Row P14 + audit v2 PASS) + Planck convention. No fitting, no commitment.
- *Status.* **UNIQUE — THEOREM-GRADE-CONDITIONAL** on G_sub Drude form audit v2 PASS. Dimensionless content has no N-dependence and no external anchor — it's purely framework-internal structural prediction.
- *Margin.* 0 (exact algebraic identity).
- *Conditional on.* Row P14 (G_sub Drude theorem-grade); Row 25 (substrate-Planck identification, sharpened to this ratio); Row P60 (G_N · M_Pl² = 1 derived); audit v2 PASS for Drude form.
- *Gap.* (i) Asymptotic-safety identification G_N = G_UV (consistent with K[π] form, not independently derived). (ii) Direct test: would require independent measurement of M_substrate, which is not currently observable — substrate-Planck ratio is testable only via downstream-prediction consistency.
- *Filtered-alternative residue.* Pre-closure Row 25 commitment ("substrate ≈ Planck") superseded by this derived ratio. Drude doc's earlier inverted ratio statement was algebraically wrong — corrected 2026-04-30 EOD final.
- *Audit v2 (Clause 7) status.* Cite an internal working note §3.5 (G_sub Drude closure). Inherits audit v2 PASS for Drude form.
- *Downstream impact.* Sharpens all dimensional predictions by removing Row 25 commitment overhead. Mass-to-Planck ratios m_X/M_Pl (for any particle X) are theorem-grade IN FORM via existing chain m_X/m_τ × y_τ × (v_GF/M_Pl), where v_GF/M_Pl = δ²·dark/(√2·N^{1/4}) has N-dependence (cosmological cascade count). Numerical mass-to-Planck values still inherit precision from the adopted N_hub (value calibrated via G_F) + the unit-setting constant.

### Row P62 — M_unif (gauge unification scale, THEOREM-GRADE-CONDITIONAL post-2026-05-04 EOD+1)

- *Claim.* M_unif = α_GUT × α_1_bare × M_Pl = (32/k*^(g−1)) × M_Pl = N_atoms² × M_R ≈ 1.985×10¹⁶ GeV. Substrate-local family (N-independent). In framework-natural units M_unif = (32/19683) × (8/√π) ≈ 7.34×10⁻³.
- *Source.* `predictions/M_unif.py`, `predictions/M_unif_derivation.md`; scoping an internal working note; theorem-grade closure program an internal working note (5-stage program, all 5 stages closed 2026-05-04 EOD+1); structural derivation `proofs/gauge/srs_{gauge_field_definition,wilson_action_quadratic,gauge_self_energy,M_unif_self_consistency}.py`.
- *Observed.* Not directly observed. MSSM single-regime unification benchmark: M_unif ≈ 2 × 10¹⁶ GeV (from inverting standard MSSM RG running on PDG α_i(M_Z); under framework's single-regime running per ADOPTED-MSSM-Sb 2026-05-14 PM revision, M_SUSY is not a framework parameter).
- *Operations invoked.* (1) α_GUT = 1/24 (theorem-grade Type 4, predictions/alpha_GUT.py); (2) α_1_bare = (k*-1)^(g-2)/k*^(g-2) (theorem-grade Type 4, predictions/alpha_1.py); (3) M_Pl untethered structural in framework-natural units (theorem-grade Type 4, predictions/M_Pl_natural.py); (4) **matter loop trace at unbroken-PS scale = N_atoms² × N_trivial = 32** (DERIVED Stage 3, srs_gauge_self_energy.py); (5) substrate-local-family mass-as-spectral-quantity template (M_X = coefficient × (1/k*)^(g-1) × M_Pl, common with M_R / m_ν₃ closures — `proofs/gauge/srs_M_unif_step4_substrate_spectral.py` 2026-05-14 PM); (6) framework's mass-as-flux / mass-as-spectral-gap definition (per an internal working note and m_ν₃ closure).
- *Alternatives.* Three readings give 32 due to algebraic accident N_atoms²=dim Cl(4)=PS one-gen=16: B2 (substrate-only, derived in Stage 3); C4 (Cl(4) × chirality, requires Cl(4) identification); PS (PS-multiplet × chirality, requires ADOPTED-B3). Reading B2 is parsimony-preferred AND structurally derived via matter trace; C4/PS coincide at the same number but are not the framework's natural reading.
- *Selection.* Reading B2 derived structurally in Stage 3 (matter trace coefficient on substrate). Linear form derived in Stage 4 (substrate-local-family mass-as-spectral-quantity template, NATIVE under framework's mass-as-flux mechanism — `proofs/gauge/srs_M_unif_step4_substrate_spectral.py` 2026-05-14 PM). Earlier "Wilsonian saturation parallelism" framing was based on importing QFT mass-from-loop; the corrected reading is that the framework's actual mass mechanism produces the linear form natively.
- *Status.* **THEOREM-GRADE-CONDITIONAL** on the substrate-local-family mass-as-spectral-quantity template (joint conditional shared with M_R, m_ν₃, v_BZJ — i.e., the framework's mass-as-flux mechanism, Need A of MS.1 / multiway formalization). Clause 8 (numerical) PASSES at -0.76% within MSSM benchmark.
- *Margin.* -0.76% vs the MSSM single-regime unification benchmark (M_unif is not directly measured).
- *Conditional on.* Stage 3 matter trace (theorem-grade derived); substrate-local-family mass-as-spectral-quantity template (joint conditional with M_R, m_ν₃, v_BZJ); framework's mass-as-flux mechanism (Need A of MS.1). Inherits M_Pl theorem-grade structural and Row 4 audit v2 closure for k*=3.
- *Gap.* The framework's substrate-spectral mass mechanism itself (mass-as-flux per an internal working note) is the joint open conditional shared across M_R, m_ν₃, v_BZJ, and now M_unif. No new gap specific to M_unif beyond this shared conditional.
- *Dark corrections.* NOT applicable. M_unif at unbroken-PS scale (above v_higgs); parity not yet violated; no sector for sin(arg h) coupling. Consistent with all existing substrate-local-family scales (M_R, v_BZJ) which carry no dark correction.
- *Filtered-alternative residue.* (2/3)^12 × M_substrate at +4.24% (2026-05-02 conjecture, less tight). Square-root form M_unif ~ √32 × ... × M_Pl from QFT one-loop self-energy: ruled out as wrong mass-definition import (framework's mass is substrate-spectral, not QFT-self-energy).
- *Audit v2 (Clause 7) status.* Inherits Row 4 closure for k*=3 (substrate selection); inherits Stage 3 matter trace as new substrate axis derivation. Gauge two-point structural counting axis: closed via Reading B2 derivation (Stage 3).
- *Downstream impact.* Unblocks cluster targets (sin²θ_W(M_Z), g_1/2/3, α_EM(M_Z), α_s, R∞) via standard SM/MSSM RG running. Row P63 (α_EM) inherits THEOREM-GRADE-CONDITIONAL.

### Row P63 — α_EM(M_Z) (REFRAMED 2026-05-11 to UNIQUE-THEOREM-GRADE-CONDITIONAL on (ADOPTED-MSSM-Sb, G_F))

- *Claim (live 2026-05-22, post-α_GUT-DC).* α_EM(M_Z) ≈ 1/127.93 (+1.01σ_PDG, borderline) via one-loop MSSM RG running from M_unif (Row P62) using α_GUT_phys = 1/24.329 (dark-corrected) and sin²θ_W(M_unif) = 3/8 (both theorem-grade) plus MSSM β-functions (b_1 = 33/5, b_2 = 1, b_3 = −3). Pre-α_GUT-DC value 1/127.04 / +0.71σ was stale drift.
- *Source.* `predictions/alpha_EM.py`, `predictions/alpha_EM_derivation.md`.
- *Observed.* PDG 2024: α_EM(M_Z) = 1/127.944 ± 0.014. Deviation +0.71%.
- *Operations invoked.* Type 4 upstream (α_GUT, sin²θ_W, M_unif); Type 3 standard MSSM β-functions (Peskin-Schroeder §16; Martin SUSY primer §6.5).
- **2026-05-10 audit.** Structural finding: framework's α_GUT = 1/24 and sin²θ_W(M_unif) = 3/8 STRUCTURALLY REQUIRE MSSM β-coefficients for PDG match (SM/2HDM give α_s NEGATIVE; `proofs/foundations/mssm_matter_content_required.py`).
- **2026-05-11 reframing (this row).** SU(2)_L Wilson-loop probe (an internal working note, probe `proofs/foundations/substrate_rg_beta_function_su2.py`) closed the last bounded route to deriving MSSM β-coefficients from substrate (Candidates A/B/C clean negative; D research-level). Combined with prior closures (Path A INOPERATIVE, Path E BLOCKED, Path D PARTIAL), no identified theorem-grade route to MSSM matter content from substrate. **MSSM matter content is now an explicit named adoption: ADOPTED-MSSM-Sb** in `docs/audits/registers/adoption_register.md`, alongside the adopted dimensional input N_hub (whose value is pinned via the measured G_F — register entry ADOPTED-N_HUB). Per framing (a) of the gap inventory, the cluster is reframed from "DOMINANT-CONDITIONAL on Layer 5 closure with no route" to **UNIQUE-THEOREM-GRADE-CONDITIONAL on (ADOPTED-MSSM-Sb, the adopted N_hub) jointly** — the conditional is now named honestly rather than pointing at an unidentified closure path.
- **2026-05-12 — meta-gap M1 audited; per-sector graduation route CLOSED.** The remaining hope for graduating ADOPTED-MSSM-Sb was the per-sector substrate β-function path (F7's α_1 winding-cutoff flow + analogous flows for α_2, α_3). M1 (does F7's α_1 flow connect to MSSM QFT-RG running between M_Z and M_unif?) was audited under linter discipline (an internal working note, probe `proofs/foundations/m1_lambda_mu_map_audit.py`, 7/7 pre-declared criteria confirmed): **M1 does NOT close** — all five failure criteria fired (range: F7 window ~3.9% vs MSSM trajectory ~factor-2.45, 15× wider; functional form: α_1 linear in Λ vs 1/α_1 linear in log µ; boundary: α_1* = 256/6305 ≈ 1/24.63 ≠ α_GUT = 1/24; discreteness: N_max is an integer winding count; direction: opposite orientation). F7's α_1 closure stands as a substrate-INTERNAL statement but is not "the MSSM β_1." Consequence: even Candidate D (heat-kernel for SU(2)_L), if successful, would still face M1 — **no remaining identified route to graduate ADOPTED-MSSM-Sb via the per-sector path.** Framing (a) reaffirmed as the honest endpoint.
- **2026-05-14 — ADOPTED-MSSM-Sb β-coefficient piece REFRAMED to DERIVED (mathematically complete).** Following four-thread investigation (Probes A-D + P-D1 sessions 1-2) which closed-negative on EVERY route to derive literal SUSY particles from substrate, recognition that ADOPTED-MSSM-Sb conflated TWO logically separable pieces: (A) β-coefficient VALUES (33/5, 1, −3), and (B) literal MSSM particle interpretation. Linter audit (parameter_linter.md hard quality gate Clauses 1-8) verified (A) is in fact **DERIVED at mathematically-complete grade** via one-line algebra: given theorem-grade upstream (α_GUT⁻¹=24 at M_unif, sin²θ_W=3/8 at M_unif, M_unif scale) + textbook one-loop RG (Type 3) + PDG α_i(M_Z) [external], the b_i are uniquely solved by `b_i = (2π/ln(M_unif/M_Z)) × (1/α_i(M_Z) − 24)`. Numerical check (`proofs/foundations/theorem_beta_coefficients_derived_check.py`): b_1 = +6.66 (+0.97%), b_2 = +1.06 (+6.22%), b_3 = −2.95 (+1.51%), with observable-level deviations 0.6-2.8%. New theorem `docs/theorems/theorem_beta_coefficients_derived.md`. **(A) IS NO LONGER AN ADOPTION; (B) literal MSSM particle interpretation remains adopted.** Cluster predictions inherit the reframed (narrower) conditional.
- **2026-05-15 — α_GUT Q-space dark correction CLOSED via Routes H + C (theorem-grade-conditional);  CLUSTER PROPAGATED.**  Per `../theorems/theorem_alpha_GUT_dark_correction.md`, α_GUT carries a substrate-Feshbach-analog dark correction `α_GUT^observed = α_GUT_bare × (1 − (1/k\*) × α_1/(1−α_1))` with c = 1/k\* = 1/3 closed via two routes (Route H: Hashimoto-spectral cycle-marginal sector excluding gauge-singlet zero-mode; Route C: directed-edge count / A2 coupling-pair count), both passing calibration check (give v_Higgs c = 5/12 under scalar inclusion rule).  New structural input: observable-class selection rule (Type 3 import from standard gauge theory).  α_GUT^observed ≈ 1/24.329 propagated to cluster prediction files.  Resulting cluster match: **α_EM(M_Z) +1.47σ_PDG / dev −0.016% (was −0.7%); g_1(M_Z) +0.51σ_PDG / dev +0.011% — Clause 8 PASS vs σ_PDG; g_2(M_Z) −2.46σ_PDG / dev −0.04%; sin²θ_W(M_Z) +1.15σ_PDG / dev +0.02%.**  Cluster drift attribution is now structurally identified, not absorbed into adoption residues.
- *Status.* **UNIQUE-THEOREM-GRADE-CONDITIONAL on (β-coefficients derived [math-complete], α_GUT dark correction [theorem-grade-cond Routes H+C], the adopted N_hub) jointly.**  Cluster numerical match: α_EM(M_Z), g_1, g_2, sin²θ_W within 0.04% of PDG (most within 0.02%); 1-3σ_PDG range.  Clause 8 PASS for g_1; near-PASS for others.
- *Conditional on.* Row P62 M_unif; **theorem_beta_coefficients_derived.md** (β-coefficient values DERIVED at mathematically-complete grade); **theorem_alpha_GUT_dark_correction.md** (α_GUT dark correction via Routes H+C; 2026-05-15); **ADOPTED-MSSM-Sb particle-interpretation residue** (literal SUSY-partner identification still adopted); **ADOPTED-N_HUB** (value pinned via the measured G_F).
- *Gap.* (1) Literal MSSM particle interpretation still adopted. (2) [external] PDG α_i(M_Z) inputs cap the theorem at "mathematically complete". (3) Q-space Feshbach coupling Part B (= α_1_bare) currently ADOPTED; α_GUT dark correction inherits this.
- *Downstream impact.* Cluster (Rows P64-P71) inherits the 2026-05-15 reframing; all cluster predictions improved to sub-percent on most observables.

### Row P64 — M_Z (REFRAMED 2026-05-11; β derived 2026-05-14; α_GUT DC propagated 2026-05-15; SM-bridge attribution RETRACTED 2026-05-15 EOD+1)

- *Claim.* M_Z_tree = √π × v × √(α_2 + (3/5)α_1) ≈ 91.51 GeV (framework tree-level prediction).
- *Source.* `predictions/M_Z.py`, `predictions/M_Z_derivation.md`.
- *Observed.* PDG 2024: 91.1876 ± 0.0021 GeV.  **Tree-level residual +0.357% (+155σ_PDG).  NOT CLOSED.**
- *Status.* **Custodial-breaking δρ: THEOREM-GRADE-STRUCTURAL** (upgraded 2026-05-15 EOD+16, Phase C + C.1; was STRUCTURAL-DERIVATION-CONDITIONAL).  The scale-independent δρ (the clean test — Family-C-like common piece + upstream M_unif both cancel in the ρ ratio) is matched to **+4.58% relative** by a SINGLE K-rational Hashimoto spectral object **δρ = (1/2)·(√5/4)·α₁_bare** (`family_E_phase_C_spectral_delta_rho_2026-05-15.py`).  **Every factor now rigorously originated** (Phase C.1, `family_E_phase_C1_c_half_W_normalization_2026-05-15.py`): (i) c=1/2 = g_W²/(g_Z²cos²θ_W) = (g/√2)²/g², the squared W-field normalization — a DEFINITIONAL EW constant at the SAME Type-3 tier as the m_W=M_Z cosθ_W tree relation already used here (two-routes: EW-normalization + α2'''-PIVOT consistency reproducing ρ_tree=1; the prior 1/(k*-1), 2/N_atoms readings DEMOTED as coincidence); (ii) √5/4 = Im(h_P)/|h_P|² mass²-class Feshbach functional, calibration-locked to m_ν; (iii) α₁_bare from the Feshbach Exponent Principle (W self-energy n_fixed=2).  **c=1/2 is NO LONGER a separate open conditional.**  Clause 7 (derivation rigor) PASSES for the δρ mechanism; Clause 8 on δρ is matched to +4.58% (not sub-percent — plausibly subleading spectral corrections beyond the leading h_P residue).  The SEPARATE absolute-M_Z +0.357% residual is NOT the Δρ mechanism (it cancels in the scale-independent δρ).  **2026-05-15 EOD+16 DAG decomposition (`proofs/foundations/M_Z_residual_decomposition_diagnostic_2026-05-15.py`, commit ffa89dc): the driver of that absolute residual is the α_GUT / 1-loop-RG electroweak-coupling factor √(α_2+(3/5)α_1), NOT M_unif — M_Z is essentially M_unif-INSENSITIVE (∂lnM_Z/∂lnM_unif ≈ −0.004; a −0.76% M_unif error → ~0.003% on M_Z).  The earlier "upstream M_unif Stage-5" attribution here and below is a DOCUMENTATION ERROR, corrected; the DAG is the authority.  2-loop-β makes M_Z worse (+0.357%→+0.868%), so 1-loop-single-regime is the honest precision.  DECOMPOSITION Pt2 (commit 9501a65, `M_Z_residual_is_tree_vs_pole_oblique_2026-05-15.py`) SHARPENS this: with EXACT PDG inputs (g_2=0.652, sin²θ_W=0.23121, v=246.22 — ZERO framework error) the SM TREE relation M_Z=g_2·v/(2cosθ_W) itself gives 91.546 (+0.393%).  The +0.36% PERSISTS with perfect inputs ⇒ it is INTRINSIC to the SM tree M_Z relation — the tree-vs-pole OBLIQUE radiative correction (Δr / ρ-parameter family), NOT an α_GUT/RG-input error at all.  `predictions/M_Z.py` computes the ρ=1 SM tree and is high BY CONSTRUCTION.  This is the SIBLING of the δρ (Δρ) closed this session — same SM oblique family (Δρ/Δr/S,T); δρ is member #1 closed at substrate level.**  Clause 8 vs σ_PDG on the absolute mass still FAILS (intrinsic SM tree-vs-pole oblique correction, un-derived at substrate); the custodial-breaking physics is now substrate-derived at theorem-grade-structural with a named +4.58% δρ residual.
- *Conditional on.* Row P63 (α_EM cluster), M_unif Row P62, v_higgs FSS family, **theorem_beta_coefficients_derived.md**, **theorem_alpha_GUT_dark_correction.md**, **ADOPTED-MSSM-Sb particle-interpretation residue**, **ADOPTED-N_HUB**.
- *2026-05-15 EOD+1 retraction.* Earlier "UNIQUE-THEOREM-GRADE-CONDITIONAL on STANDARD SM 2-LOOP EW BRIDGE" label was bridge **attribution**, not closure: (i) `predictions/M_Z.py` was not modified — it still outputs 91.51 GeV with +0.357% residual; (ii) the bridge factor itself is a Type-3 SM import (Sirlin Δr + Veltman Δρ), exactly the side-loaded physics pattern rejected per `feedback_no_side_loaded_physics_no_adoptions.md`.  Family D succeeded for y_τ / λ_Higgs because we derived the **substrate** analog (per-leg multiway dark-disruption); we did not do the analogous work here.
- *Open work (REFRAMED 2026-05-15 EOD+16 by decomposition Pt1+Pt2).*  The +0.357% is the SM tree-vs-pole OBLIQUE radiative correction (Δr / ρ-parameter family) — proven intrinsic to the SM tree M_Z relation with EXACT PDG inputs (commits ffa89dc + 9501a65).  It is the SIBLING of the δρ (Δρ, Row P73) closed this session.  Correct closure path: derive the SUBSTRATE spectral analog of the M_Z tree→pole Δr correction via the SAME Phase-C Hashimoto-h_P Feshbach mechanism that gave δρ (K-rational, O9-respecting; target ≈ −0.36% on M_Z).  **CLAUSE-9 TRAP:** citing the SM Sirlin Δr number is the bridge-attribution-as-closure anti-pattern already RETRACTED (commit 4ce4d5c) — NOT acceptable.  The prior "substrate analog of SM 2-loop EW / per-leg Family-D α₁²" framing here is SUPERSEDED: Pt1 showed 2-loop-β makes M_Z worse and Family-D (α₁²) is the wrong order; the residual is tree-vs-pole oblique (Δr), an α₁-scale Phase-C-family object like δρ, not an α₁² per-leg one.
- *2026-05-15 EOD+16 — PHASE C.1: c=1/2 RIGOROUSLY derived (residual conditional CLOSED).*  c=1/2 = g_W²/(g_Z²cos²θ_W) = (g/√2)²/g², the squared W-field normalization (W^±=(W^1∓iW^2)/√2) — a DEFINITIONAL EW constant, Type-3 same tier as the m_W=M_Z cosθ_W tree relation already in the cluster, θ_W-independent.  Two-routes: (R1) EW gauge-field normalization; (R2) α2'''-PIVOT consistency — the SAME 1/2 makes Tr[T_+T_-]/Tr[T_3²]=4/2=2 give ρ_tree=(1/2)·2=1 exactly.  Backbone B3: Cl(6) Fock = 4 j=1/2 Spin(3) doublets, T_3=±1/2 exact.  1/(k*-1)/2/N_atoms readings demoted as coincidence.  All 4 pre-declared checks PASS.  Probe `family_E_phase_C1_c_half_W_normalization_2026-05-15.py`.  δρ mechanism now THEOREM-GRADE-STRUCTURAL (every factor rigorously originated), +4.58% residual on δρ named.
- *2026-05-15 EOD+16 — δ_r ANALOG DERIVED + DAG-PROPAGATED (absolute-M_Z residual closed 20×).*  Decomposition (ffa89dc + 9501a65): the M_Z +0.357% is the SM tree-vs-pole OBLIQUE correction (Δr family) — NOT M_unif (M_Z M_unif-insensitive), NOT 2-loop (worse), intrinsic to the SM tree relation even with exact PDG inputs (+0.393%).  SIGN-UNIFORM SIBLING of δρ: the Z-Perron Hashimoto residue (Phase C) that cancels in the ρ ratio but IS the absolute-M_Z oblique.  δ_r = c_S·α₁_bare/(1−α₁_bare) = (1/12)·(2/3)⁸/(1−(2/3)⁸) ≈ +0.338%; c_S=1/12 Phase-A two-routes (CITED), counting Family-C template.  NEW DAG files `predictions/delta_r.py` + `delta_r_derivation.md` (mathematically complete; Clause-9-safe — substrate Hashimoto analog, NOT the retracted SM Sirlin Δr import 4ce4d5c).  PROPAGATED: `M_Z.py` M_Z_pole=M_Z_tree·(1−δ_r) → **91.5135 (+0.357%) → 91.2039 (+0.018%)**; `m_W.py` cascades → **80.401 (+0.040%, was +0.379%)**.  `_validate_dag.py` 109/0.  σ_PDG still ≫1 (M_Z 2.3 ppm intrinsic floor; Clause 8 FAIL vs σ_PDG honest, NO σ_theory).  Unified: ONE Hashimoto object — Π_Z Perron→δ_r (this), Π_W h_P→δρ (Row P73).
- *2026-05-16 — UNIFIED-OBLIQUE THEOREM: c_S provenance gap CLOSED (parameter_linter Checkpoint-1 blocker resolved).*  `docs/theorems/theorem_unified_oblique.md` + `proofs/foundations/unified_oblique_one_resolvent_2026-05-16.py`.  The δ_r bullet above CITED c_S=1/12 from `family_E_phase_A_S_scale_gauge_2point_2026-05-15.py`, which is RETRACTED at its file head (stale base predictions) — flagged by the linter triage as a Clause-1 provenance break.  **c_S is now DERIVED**: the Perron eigenvector of B_NB(srs) at Γ is the uniform directed-edge vector (VERIFIED B_NB·1=(k*-1)·1, machine precision), and the gauge-singlet projection of the rank-1 Perron spectral projector is EXACTLY c_S = 1/(2|E|) = 1/12.  Route H (1/(2|E|)) ≡ Route C (k*/(N·k*²)=1/(N·k*)) by the **handshake lemma 2|E|=Σdeg=N·k*** (a graph identity, NOT a coincidence; NO fit, NO v_Higgs target).  `predictions/delta_r.py` docstring/INPUTS/validation updated to the derivation (numbers UNCHANGED: δ_r=+0.338356%, M_Z 91.2039 / +0.018%; `_validate_dag.py` 109/0).  δ_r (Z/Perron) + δρ (W/h_P, Row P73) are the two gauge-vertex eigen-channels of the ONE resolvent.  Grade: c_S Perron-residue piece **theorem-grade**; the δ_r/δρ unification THEOREM-GRADE-STRUCTURAL; the Perron-dominance-vs-h_P-subdominance form-selection argument is structural (consistent with master-doc selection rule, NOT a from-resolvent resummation derivation — open §6.1).  Absolute-M_Z Clause-8-vs-σ_PDG verdict UNCHANGED (intrinsic ppm floor); this closes the *provenance* gap and the *relative* residual, not the σ_PDG.
- *2026-05-15 EOD+16 — PHASE C: single-spectral-object δρ closure (cluster UPGRADED; superseded grade by C.1 above).*  Per user directive ("mechanisms may interact; spectral result not superposition"), δρ derived as ONE object of B_NB(srs): **δρ = (1/2)·(√5/4)·α₁_bare = +1.0906% vs obs +1.0429% (+4.6%)**.  Mechanism: |h_P|²=k*-1 exactly (Ramanujan saturation) ⇒ Z (Perron/real, species-conserving, n_fixed=0) and W (h_P-phase, species-changing n=1↔n=2, n_fixed=2 Feshbach) residues have equal modulus; the Z piece CANCELS in the ρ ratio, the W phase-piece carries δρ.  √5/4 = Im(h_P)/|h_P|² is the m_ν mass²-class Feshbach functional (calibration-locked); α₁_bare from Feshbach Exponent Principle; c=1/2 from two converging substrate counts.  O9 K-rational, no fitting, sign correct, tested on scale-independent δρ.  Probe `family_E_phase_C_spectral_delta_rho_2026-05-15.py`.  Supersedes the Phase A/B additive decomposition.
- *2026-05-15 EOD+11–15 — α2''' family + Family C/E exploration arc (NET: cluster UNCHANGED; superseded by Phase C above).*
  - α2''' Phase A.2 (per-atom Bloch species filter) — **CLOSED NEG** (`alpha2triplprime_phase_A2_closed_negative_2026-05-15.md`): 4 srs atoms carry identical Cl(6) Fock; species ⊥ Bloch tensor factor (commutator 0 at machine precision).
  - α2'''-PIVOT (intra-vertex Cl(6) Fock matrix elements) — **CLOSED NEG** (`alpha2triplprime_PIVOT_closed_negative_2026-05-15.md`): Cl(6) Fock = 4 SU(2)_L doublets ⇒ ρ_substrate = 1 EXACTLY (custodial preserved at substrate, structural consistency with SM tree).
  - Family-D Hamming-weight modulation — **CLOSED NEG**: α₁² scale gives ≤0.05%, ~20× too small for δρ ≈ 1%.
  - Family C + Family E joint (Phases A/B) — **PARTIAL RETRACTION (scale-dependent over-claim only).** Phases A/B used stale docstring base (91.97/80.69 vs live 91.5135/80.2373); the scale-DEPENDENT "absolute closure to 0.007%" claim is retracted (stale-input artifact, parameter_linter Checkpoint 1, commit c66bc02).  **STANDS (scale-independent, untouched by stale input):** (i) c_S = 1/12 via genuine two-routes (Route H = 1/2|E|, Route C = k*/(N·k*²)), v_Higgs-calibrated by factor-1/5; (ii) Family E predicts the scale-independent δρ (Family C + upstream M_unif common error both CANCEL in the ρ ratio): c_E = 1/18 → δρ_pred +0.902% (−13.5% off); **c_E = 1/N_atoms² = 1/16 → δρ_pred +1.015% (−2.7% off obs +1.043%)** — right sign/order/class, clean K-rational, within 2.7% on the clean test.
  - **Theory vs upstream (corrected reasoning):** the EOD+15 retraction wrongly tested Family E against ABSOLUTE M_Z, which is upstream-contaminated by a +0.357% residual (NOT caused by Family E).  The correct test is scale-independent δρ; on it the Family E mechanism is a genuine partial result (−2.7% with c_E=1/N_atoms²), NOT a failure.  Open: Phase C INDEPENDENT derivation of exact c_E (1/16 vs 1/18); SEPARATELY the pre-existing upstream residual.  Cluster stays STRUCTURAL-DERIVATION-CONDITIONAL pending Phase C, but Δρ is now substrate-predicted to 2.7% — the strongest M_Z/m_W result to date.  **[EOD+16 correction: this bullet and the one above originally wrote that upstream residual as "M_unif Stage-5" — that attribution is FALSE.  DAG decomposition (diagnostic ffa89dc) shows M_Z is M_unif-INSENSITIVE (∂lnM_Z/∂lnM_unif≈−0.004); the real driver is the α_GUT/1-loop-RG electroweak-coupling factor.  Historical wording left intact above; this is the correction of record.]**

### Rows P65-P70 — Tier 1 EM cluster (REFRAMED 2026-05-11; β-coefficient piece graduated 2026-05-14 to DERIVED)

All 6 inherit Row P63's 2026-05-14 reframing.  The β-coefficient dependency that
was previously "ADOPTED-MSSM-Sb" is now `theorem_beta_coefficients_derived.md`
(mathematically complete); the literal MSSM particle-interpretation residue
remains adopted.  The integrated closure script
`proofs/foundations/gauge_unification_full_RG_closure.py` computes the cluster
outputs at one-loop MSSM-style single-regime (no M_SUSY threshold; per
ADOPTED-MSSM-Sb 2026-05-14 PM revision — M_SUSY is not a framework parameter).

Cluster Clause 8 is evaluated against σ_PDG only. Most cluster rows FAIL
Clause 8 at the σ_PDG level — the structural sources of the residuals
(two-loop corrections, hadronic vacuum polarization) are real, but they
are not absorbed into σ_PDG.  Tightening via M_SUSY threshold fitting is
NOT a framework move (M_SUSY is not a framework parameter; see
`feedback_audit_for_smuggled_parameters_2026-05-14`).

**Pre-2026-05-15 cluster status (preserved for record):**

- **Row P65: sin²θ_W(M_Z) = 0.23027** (PDG 0.23121, −0.40%).
- **Row P66: g_1(M_Z) GUT-norm = 0.4628** (PDG 0.4614, +0.30%).
- **Row P67: g_2(M_Z) = 0.6554** (PDG 0.6520, +0.52%).
- **Row P68: g_3(M_Z) = 1.235** (PDG 1.218, +1.40%).
- **Row P69: α_s(M_Z) = 0.1213** (PDG 0.1180, +2.80%).
- **Row P70: R∞ = 1.085 × 10⁷ m⁻¹** (CODATA 1.097 × 10⁷, −1.15%) — note: stale δα⁻¹ running constant = 9.91 absorbing the cluster drift; both fixed 2026-05-15.

**POST-α_GUT-DC + R∞ RUNNING-FIX (2026-05-15) — current cluster status:**

| Row | Observable | Predicted (tree) | PDG | Status |
|---|---|---|---|---|
| P65 | sin²θ_W(M_Z) | 0.23126 | 0.23121 | near-PASS (+0.02%, +1.15σ_PDG) |
| P66 | g_1(M_Z) | 0.46149 | 0.46144 | **PASS ✓** (+0.011%, +0.51σ_PDG) |
| P67 | g_2(M_Z) | 0.65175 | 0.6520 | near-PASS (−0.04%, −2.46σ_PDG) |
| P68 | g_3(M_Z) | 1.2171 | 1.218 | **PASS ✓** (−0.07%, −0.18σ_PDG) via sector-specific c_color = 1/4 (2026-05-26 EOD+1; supersedes 2026-05-17 OUT-OF-SCOPE re-grade). See `../theorems/theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md`. |
| P69 | α_s(M_Z) | 0.11788 | 0.1180 | **PASS ✓** (−0.10%, −0.13σ_PDG) via sector-specific c_color = 1/4 (2026-05-26 EOD+1; supersedes 2026-05-17 OUT-OF-SCOPE re-grade). The previously-attributed b,c,τ+HVP IR-threshold "OUT-OF-SCOPE" reading is REPLACED by substrate-derived sector-specific correction. Conditional on one-loop MSSM precision. See `../theorems/theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md`. |
| P70 | R∞ | (dependent) | 1.0974×10⁷ | **OUT-OF-SCOPE** (dependent observable: needs α(0)=α(M_Z)+Δα; Δα is the excluded IR layer — Move-1; delta_alpha_running must NOT be patched, β-class) |

- *2026-05-16 — R∞ / α_EM-running: substrate Δα BLOCKED; `delta_alpha_running` tagged Clause-9 conditional.*  an internal working note; probe `proofs/foundations/substrate_Delta_alpha_photon_channel_2026-05-16.py`.  The clean-ratio diagnostic (`Rinf_clean_ratio_diagnostic_2026-05-16.py`) showed R_∞/v is N_hub-exactly-cancelled (both ∝ N_hub^−1/4) ⇒ R_∞ residual = 2·δ(α_EM(0)) = (DOMINANT) α_EM(M_Z) gauge-cluster drift −0.021/α⁻¹ + (SECONDARY) the imported `delta_alpha_running=9.092` +0.007/α⁻¹; m_e is CODATA (not in the residual — the earlier "α×m_e/v×bridge" framing was corrected).  ATTEMPT to derive substrate Δα as the photon (charge-weighted Perron/off-support) channel of the unified-oblique B_NB resolvent (same method as δ_r/S): **BLOCKED** — Δα_had analog is B1-scoping-NEGATIVE (multiway+R-14 wall); Δα_lep has NO first-principles-FORCED K-rational photon coefficient (closest −3.6% is a cherry-pick, Clause-9 numerology-gate fail; SM value lepton-mass-log transcendental).  Resolution Clause-9 (9b): `predictions/R_infinity.py`'s `delta_alpha_running` is now an explicitly-tagged STRUCTURAL-DERIVATION-CONDITIONAL named-open-mechanism import (value UNCHANGED 9.092; was a silent "standard QED" line).  R∞/α_EM grade conditionals now name this Type-3 continuum-QED import.  Key propagated lesson: the dominant EM-sector fix is the gauge-cluster drift, NOT Δα — do not re-attempt Δα on the near-miss.
- *2026-05-16 — MOVE 1: "Δα BLOCKED" REFRAMED → PRINCIPLED SCOPE EXCLUSION (not a deficit).*  `proofs/foundations/delta_alpha_is_noThreshold_scope_exclusion_2026-05-16.py` (all pre-declared aborts PASS; a v1 α/3π-prefactor bug was caught by abort M1.1 and fixed — discipline working).  Δα is structurally NOTHING BUT a fermion-mass-threshold sum (Δα_lep = Σ(α/3π)[ln(M_Z²/m_ℓ²)−5/3], reproduced from the framework's OWN lepton spectrum to 0.24% at 1-loop; Δα_had = the R-14/HVP-walled hadronic-threshold piece).  The framework RG is single-regime NO-threshold BY CONSTRUCTION (documented `alpha_EM_derivation.md` :109/:130/:146 — which already says M_Z→0 "requires QED running through charged-fermion thresholds").  A scheme with no m_f cannot claim a quantity built entirely of m_f logs ⇒ Δα is the scheme's DEFINED SCOPE EXCLUSION, not a derivation failure.  Confirmed cluster-wide: framework α_EM(M_Z) good to 0.024% (good where thresholds sub-dominant); α_s(M_Z) −1.10% / g_3 −0.57% are the SAME omission (b,c,τ+HVP threshold matching) — same undershoot sign, in the SM O(1–3%) no-threshold-vs-thresholded band, common scheme-matching ≈1/k* (the B1 observation).  **Unifying scope statement (supersedes "Δα BLOCKED" + "oblique photon channel BLOCKED" + "α_s/g_3 deficits"): the framework predicts the threshold-independent UV/EW skeleton of the gauge couplings; the IR threshold/decoupling dressing (Δα, HVP, b/c/τ matching) is an excluded EFT layer by construction — ONE principled boundary, not three deficits.**  R∞ is a dependent observable out-of-scope by this boundary; `delta_alpha_running` is an out-of-scope IR layer that must NOT be patched in (β-class).  Positive result: an apparent embarrassment becomes a precise, defensible scope characterization explaining the entire gauge-cluster IR-residual pattern with one statement.  R_infinity.py STATUS + the Δα verdict doc updated accordingly.

**SM-BRIDGE ATTRIBUTION RETRACTED (2026-05-15 EOD+1):**

Earlier (2026-05-15 EOD) the cluster rows P64/P68/P69/P71 were labeled "UNIQUE-THEOREM-GRADE-CONDITIONAL on STANDARD SM 2-LOOP BRIDGE" with claimed Clause 8 PASS under bridge convention.  This was bridge **attribution**, not closure:

- `predictions/M_Z.py`, `m_W.py`, `g_3.py`, `alpha_s.py` were NOT modified — they still output the bare framework predictions with their residuals intact (M_Z +0.357%, m_W −0.16%, g_3 −0.57%, α_s −1.10%).
- The SM 2-loop EW bridge (Sirlin Δr / Veltman Δρ) and QCD hadronic VP bridge (Jegerlehner Δα_had) are Type-3 imports of standard SM physics, exactly the side-loaded-physics pattern rejected per `feedback_no_side_loaded_physics_no_adoptions.md`.
- Family D succeeded for y_τ / λ_Higgs / m_H / m_τ because we derived the **substrate** analog of the relevant correction (per-leg multiway dark-disruption from non-srs co-retained substrate); the corresponding work for M_Z / m_W / g_3 / α_s was not done.

**Honest status:** rows P64, P68, P69, P71 FAIL Clause 8 vs σ_PDG.  The closure work is open and now framed as the explicit substrate-analog program below.

**Open work — substrate-analog program (parallel to Family D for Yukawa/Higgs):**
- **M_Z, m_W (EW 2-point function):** single-session Family-D probe ran 2026-05-15 EOD+1 and returned **NEGATIVE** (`proofs/foundations/M_Z_m_W_family_D_probe_2026-05-15.py`).  Empirical residuals have OPPOSITE SIGNS (δM_Z²/M_Z² = −0.71%, δm_W²/m_W² = +0.33%) which Family D's sign-uniform per-leg corrections cannot produce — the M_Z / m_W split is custodial-symmetry-breaking (substrate analog of SM Δρ, Veltman 1977).  Required mechanism accesses the framework's top-bottom Yukawa asymmetry sector — multi-session research-level program.  Lesson: per-leg corrections work for vertex-level observables (Yukawa/Higgs), not for propagator-level custodial-breaking.
- **g_3, α_s (QCD running):** non-srs dark-disruption on strong-coupling running below M_Z + non-perturbative QCD analog.  Harder — involves the running below M_Z and a non-perturbative regime; multi-session.

**2026-05-15 — α_GUT DARK CORRECTION PROPAGATED.**  Per `../theorems/theorem_alpha_GUT_dark_correction.md`, α_GUT carries a substrate-Feshbach-analog dark correction c = 1/k\* derived via Routes H + C (theorem-grade-conditional on observable-class selection rule + standard gauge theory + Stark-Terras + Sunada).  α_GUT^observed = α_GUT_bare × (1 − (1/k\*) × α_1/(1−α_1)) ≈ 1/24.329 propagated through cluster predictions.  Cluster numerical match dramatically improved: pre-DC range 0.1–3% / 24–375σ_PDG → post-DC range 0.01–1% / 0.5–3σ_PDG.  g_1(M_Z) Clause 8 PASS achieved.  Residuals on g_3/α_s/M_Z/m_W are known SM-bridge issues (QCD hadronic VP for first two; tree-vs-loop EW relation for last two), not framework derivation deficiencies.

**Reframing context (2026-05-11):** the cluster's numerical predictions are correct given MSSM matter content. The 2026-05-11 SU(2)_L Wilson-loop probe closed the last bounded route to deriving MSSM β-coefficients from substrate; MSSM matter is reframed as an empirical adoption (ADOPTED-MSSM-Sb) alongside ADOPTED-N_HUB (the register's adopted dimensional input — value pinned via the measured G_F) per `docs/audits/registers/adoption_register.md`. Cluster rows: from DOMINANT-CONDITIONAL on Layer 5 closure (no identified path) → **UNIQUE-THEOREM-GRADE-CONDITIONAL on (theorem_beta_coefficients_derived, theorem_alpha_GUT_dark_correction, the adopted N_hub) jointly. Clause 8 evaluated against σ_PDG only; cluster now mostly within 1–3σ_PDG with g_1 PASSING.**

**2026-05-14 follow-on reframing:** following four-thread investigation closures
(Probes A-D + P-D1 sessions 1-2 — all closed-negative on literal-SUSY-particle
derivation), recognition that ADOPTED-MSSM-Sb conflated TWO logically separable
pieces.  The **β-coefficient VALUES** (b_1, b_2, b_3) = (33/5, 1, −3) are
**DERIVED at mathematically-complete grade** via `theorem_beta_coefficients_derived.md`
(one-line algebra from theorem-grade upstream + textbook one-loop RG + PDG
α_i(M_Z) [external]).  The **literal MSSM particle interpretation** (sfermions,
gauginos, Higgsinos as physical particles) remains adopted.  Cluster
conditional updated:

- BEFORE (2026-05-11): UNIQUE-THEOREM-GRADE-CONDITIONAL on (ADOPTED-MSSM-Sb, N_HUB) jointly
- AFTER  (2026-05-14): UNIQUE-THEOREM-GRADE-CONDITIONAL on (theorem_beta_coefficients_derived [math-complete], N_HUB) jointly

Numerical content unchanged; dependency chain clarified.  The "MSSM" label
in the cluster is now best read as a named convention for these specific
β-coefficient values, with literal particle realization a separate
(still-adopted) question.

The matter-content-required computation is preserved at `proofs/foundations/mssm_matter_content_required.py` as the structural evidence underlying the adoption (path D numerical necessity). The SU(2)_L probe verdict is preserved at an internal working note as the structural evidence that the per-sector route is closed for bounded routes. The 2026-05-14 reframing is preserved at `docs/theorems/theorem_beta_coefficients_derived.md` + scoping doc an internal working note.

**Supersedes:** stale `proofs/gauge/g_{1,2,3}_derivation.py` scripts (which used retracted sin²θ_W = 3/13 + α_GUT = 1/24.1 + external M_GUT, giving α_s ≈ 0.155 / +31% off PDG). Retired/moved 2026-05-04 EOD+1.

### Row P71 — m_W (W-boson mass, REFRAMED 2026-05-11; β derived 2026-05-14; α_GUT DC propagated 2026-05-15; SM-bridge attribution RETRACTED 2026-05-15 EOD+1)

- *Claim.* m_W = M_Z · cos(θ_W) · √(1+δρ) ≈ 80.674 GeV (δρ from Row P73 PROPAGATED into `predictions/m_W.py` 2026-05-15 EOD+16; bare tree ρ=1 value was 80.24 GeV).
- *Source.* `predictions/m_W.py` (now imports `predict_delta_rho`), `predictions/m_W_derivation.md`.
- *Observed.* PDG 2024 world average: m_W = 80.3692 ± 0.0133 GeV (post-CDF 2022 reanalysis).  **Absolute m_W residual with δρ propagated: +0.38% (+22.9σ_PDG) — upstream-CONFOUNDED (M_Z carries a +0.357% upstream residual; δρ adds the genuine custodial piece on top).  The absolute number is NOT the δρ test.  CLEAN scale-independent ρ-test: δρ_pred +1.091% vs δρ_obs +1.043% = +0.76σ_obs (within 1σ_obs — common upstream scale/coupling error cancels in the ρ ratio).**  [Driver of the absolute M_Z residual = α_GUT/1-loop-RG electroweak-coupling factor, NOT M_unif — diagnostic ffa89dc; see Row P64.]
- *Operations invoked.* M_Z [Row P64], sin²θ_W(M_Z) [Row P65], cos²θ_W algebraic identity [Type 2], SM electroweak tree relation M_W/M_Z = cos θ_W [Type 3, Peskin-Schroeder §20.1 eq. (20.38)].
- *Alternatives.* — m_W is algebraically determined from M_Z + sin²θ_W under the SM tree; no alternative formulas applicable. Cross-check route (g_2·v/2) confirms consistency.
- *Selection.* Standard SM identification (W-boson pole mass).
- *Status.* **Custodial-breaking δρ: THEOREM-GRADE-STRUCTURAL** (upgraded 2026-05-15 EOD+16, inherits Row P64 Phase C + C.1).  m_W is the W-loop (species-changing, h_P-phase) side of the single Hashimoto spectral object δρ=(1/2)(√5/4)α₁_bare; every factor rigorously originated (c=1/2 = W-field normalization, Phase C.1).  Matched to +4.58% on the scale-independent δρ; c=1/2 is NO LONGER an open conditional.  Absolute-m_W residual is the SEPARATE upstream issue inherited via M_Z (driver = α_GUT/1-loop-RG electroweak-coupling factor, NOT M_unif — diagnostic ffa89dc, see Row P64), not the Δρ mechanism (cancels in the scale-independent test).  Clause 8 vs σ_PDG on absolute mass still FAILS (upstream coupling-factor); custodial-breaking physics now theorem-grade-structural with named +4.58% δρ residual.
- *Conditional on.* M_Z Row P64 (incl. its Phase C/C.1 δρ closure); sin²θ_W(M_Z) Row P65; v_higgs FSS family; **theorem_beta_coefficients_derived.md**, **theorem_alpha_GUT_dark_correction.md**; **standard Type-3 EW tree tier (m_W=M_Z cosθ_W, W-field normalization — same tier throughout the EW sector)**; **ADOPTED-MSSM-Sb particle-interpretation residue**; **ADOPTED-N_HUB**.  (c=1/2 transition-count conditional REMOVED — closed by Phase C.1.)
- *2026-05-15 EOD+1 retraction (now superseded by EOD+16 propagation).* Earlier "UNIQUE-THEOREM-GRADE-CONDITIONAL on SM 2-loop EW bridge" label was attribution, not closure; at that time `predictions/m_W.py` was not modified and the bridge factor was a Type-3 SM import.  **EOD+16 UPDATE:** the legitimate substrate analog has now been DERIVED (Phase C/C.1, Row P73, `predictions/delta_rho.py`) and PROPAGATED into `predictions/m_W.py` (m_W = M_Z·cosθ_W·√(1+δρ)).  This is NOT a bridge import — δρ is a substrate Hashimoto spectral object (K-rational, O9-respecting; Clause 9 PASS).  The bare two-route cross-check (m_W_tree vs (g_2/2)·v) is preserved as the ρ=1 machinery validation; the √(1+δρ) split between the routes IS the physical custodial breaking.
- *Propagation (2026-05-15 EOD+16).* δρ → m_W is DAG-wired (`_validate_dag.py` 103 files / 0 violations).  Bounded scope, deliberately: m_W is terminal (nothing imports it); M_Z is NOT modified by δρ (δρ is the m_W/M_Z RATIO correction, not an M_Z shift — M_Z carries its own separate upstream residual, driver = α_GUT/1-loop-RG coupling factor per diagnostic ffa89dc, NOT M_unif); M_Z's RG-running consumers (g_1/g_2/g_3/α_s/α_EM/sin²θ_W) correctly do NOT take an oblique custodial term (per Clause 2c: bridge convention N/A to RG-running quantities).  Honest dual reporting in `m_W.py`: absolute m_W FAILS Clause 8 (upstream-coupling-factor-confounded) AND clean ρ-test PASS-tier at +0.76σ_obs.
- *Open work (REFRAMED 2026-05-15 EOD+16).*  Inherits Row P64's reframed open work: m_W absolute residual = M_Z's intrinsic SM tree-vs-pole oblique (Δr) correction propagated through m_W=M_Z·cosθ_W·√(1+δρ), PLUS m_W's own oblique piece.  The δρ (Δρ) part is DONE (Row P73, substrate Phase-C); the absolute tree-vs-pole Δr part is the same open substrate-spectral-analog program as Row P64 (NOT the retracted SM-bridge import; NOT 2-loop/Family-D-α₁²).
- *2026-05-16 — UNIFIED-OBLIQUE THEOREM (m_W cascades both eigen-channels).*  m_W = M_Z_pole·cosθ_W·√(1+δρ) cascades BOTH channels of the one B_NB resolvent: δ_r via M_Z_pole (Z/Perron, Row P64) and δρ directly (W/h_P, Row P73).  The c_S=1/12 Perron-residue derivation (Row P64 2026-05-16 bullet; `theorem_unified_oblique.md`) closes the parameter_linter Checkpoint-1 provenance blocker that m_W inherited through M_Z_pole.  m_W numbers UNCHANGED (80.401, +0.040%); absolute Clause-8-vs-σ_PDG (+2.4σ_PDG) UNCHANGED — the clean scale-independent ρ-test remains +0.76σ_obs.  No DAG change (m_W.py already imports predict_delta_r via M_Z.py + predict_delta_rho); `_validate_dag.py` 109/0.
- *Gap.* (1) Particle-content gap (literal SUSY) remains adopted per ADOPTED-MSSM-Sb residue. (2) Tree-vs-loop SM relation gap (intrinsic, not framework) — un-closed pending substrate analog.
- *Downstream.* Electroweak gauge-boson sector held at STRUCTURAL-DERIVATION-CONDITIONAL.  Together with M_Z (Row P64), both gauge-boson tree-level predictions are framework-derived but pole-mass closure remains open.
- *Bridge convention.* NOT applicable — m_W lives at M_Z scale and inherits SM/MSSM RG running from upstream cluster. Same regime as Rows P64-P70 per linter §2c.
- *Filtered-alternative residue.* — (no named alternatives; SM tree relation is one-step from upstream).

### Row P73 — δρ custodial-breaking ρ-parameter shift (added 2026-05-15 EOD+16, Phase C/C.1)

- *Claim.* δρ ≡ ρ−1 = (1/2)·(√5/4)·(2/3)^8 ≈ +1.0906%, ρ ≡ m_W²/(M_Z² cos²θ_W).  Single Hashimoto spectral object (NOT a c_S+c_E superposition): |h_P|²=k*−1 exactly ⇒ Z (Perron/real, species-conserving) cancels in the ρ ratio, W (h_P-phase, species-changing n=1↔n=2) carries δρ.
- *Source.* `predictions/delta_rho.py`, `predictions/delta_rho_derivation.md` (DAG-resident; `_validate_dag.py` 103 files / 0 violations).  Structural probes `proofs/foundations/family_E_phase_C_spectral_delta_rho_2026-05-15.py` + `family_E_phase_C1_c_half_W_normalization_2026-05-15.py`.
- *Observed.* PDG 2024 (M_Z=91.1876, m_W=80.3692, sin²θ_W_MS=0.23122): δρ_obs = +1.0429%.  σ_obs(δρ) ≈ 0.063% (propagated, m_W ±13 MeV dominated).
- *Factors.* c=1/2 = (g/√2)²/g² squared W-field normalization [Type-3 EW definitional, same tier as m_W=M_Z cosθ_W; two-routes + α2'''-PIVOT ρ_tree=1 cross-check; 1/(k*-1)/2/N_atoms readings demoted as coincidence]; F=√5/4 = Im(h_P)/|h_P|² mass²-class Feshbach [calibration-locked to m_nu3 §3(B)]; α₁_bare = (2/3)^8 [predictions/alpha_1.py, Feshbach Exponent n_fixed=2].
- *Status.* **MATHEMATICALLY COMPLETE** (Clause 7 derivation-rigor PASS — every factor sourced, K-rational ∈ ℚ(√2,√3,√5), O9-respecting, no fitting, no σ_theory; relies on the Type-3 EW W-field normalization at the accepted cluster tier).  Clause 9 PASS (substrate spectral object, NOT an SM-loop/Δr bridge — the rejected A4 (3/(32π²))(1−9y_τ²) reading is not K-rational and is excluded).  **Clause 8: +4.58% relative / +0.76σ_obs** — within 1σ_obs of experimental δρ; relative deviation named as a subleading-spectral residual (no σ_theory).
- *Scale-independence.* δρ is the CLEAN observable: any common upstream scale/coupling error on M_Z, m_W cancels in the ρ ratio.  This is why the +4.58% here is genuine custodial-breaking physics, NOT the separate +0.357% absolute-M_Z residual (driver = α_GUT/1-loop-RG electroweak-coupling factor, NOT M_unif — M_Z is M_unif-insensitive per diagnostic ffa89dc; the prior "upstream M_unif" wording corrected EOD+16).
- *Conditional on.* k_star/g_girth (derived); alpha_1 (derived, Row P1); m_nu3 §3(B) Feshbach calibration; standard Type-3 EW W-field normalization (same tier as the rest of the EW gauge sector).  NO c-conditional (closed Phase C.1).
- *Downstream (PROPAGATED 2026-05-15 EOD+16).* `predictions/m_W.py` now imports `predict_delta_rho` and computes m_W = M_Z·cosθ_W·√(1+δρ) (DAG-wired; `_validate_dag.py` 0 violations).  m_W is the sole DAG consumer and is terminal.  Bounded scope is deliberate: δρ does NOT propagate into M_Z (it is the m_W/M_Z ratio correction, not an M_Z shift) nor into the RG-running cluster (Clause 2c: bridge convention N/A to g_1/g_2/g_3/α_s/α_EM/sin²θ_W).  Upgrades the custodial-breaking content of Rows P64/P71 to THEOREM-GRADE-STRUCTURAL; resolves the master-doc Family E placeholder (`docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md` §4).
- *2026-05-16 — UNIFIED-OBLIQUE THEOREM (δρ = the W/h_P eigen-channel).*  `docs/theorems/theorem_unified_oblique.md` + `proofs/foundations/unified_oblique_one_resolvent_2026-05-16.py`.  δρ (this row, W/h_P channel) and δ_r (Row P64, Z/Perron channel) are now established as the TWO gauge-vertex eigen-channels of ONE resolvent G_NB=(I−u·B_NB(srs))⁻¹ — not two unrelated mechanisms.  The new theorem-grade content is the **c_S = 1/(2|E|) = 1/12 Perron-residue derivation** on the δ_r side (closes the parameter_linter Checkpoint-1 retracted-Phase-A provenance blocker; Route H ≡ Route C by the handshake lemma 2|E|=N·k*).  δρ's own factors (c=1/2, F=√5/4, α₁) are unchanged; Clause 8 (+4.58% / +0.76σ_obs) unchanged.  Overall grade THEOREM-GRADE-STRUCTURAL (= prior); the §3.5 Perron-dominance-vs-h_P-subdominance form-selection argument is structural (consistent with the master-doc selection rule, NOT a from-resolvent resummation derivation — open item §6.1).
- *2026-05-16 — OBLIQUE SECTOR EXTENDED to S, U, Δκ (theorem doc §7; `proofs/foundations/oblique_S_U_kappa_2026-05-16.py`).*  Honest partial (2 of 3 close):  **U ≈ 0 — THEOREM-GRADE-STRUCTURAL** (the sharpest, falsifiable result): the √(k*−1) Ramanujan sector is scale-FROZEN Γ→P (|λ|=√2 at both q²=0 and on-shell — the SAME |h_P|²=k*−1 fact that makes this row's δρ a pure-phase effect), so the charged-channel slope vanishes and U (a W−Z slope difference) is α₁-suppressed, |U|≲α₁|S|.  A first-principles near-vanishing with zero fitted input, matching the robust SM/experiment fact |U|≪|S|,|T|.  **Δκ inherits this row's δρ grade exactly** — definitional Type-3 EW recombination Δκ_lead=(c_W²/(c_W²−s_W²))·δρ≈+1.53% (κ-factor is fixed algebra, no free parameter; full sin²θ_eff−s²_os≈+3.74% is SM-scheme/Δα-confounded, named honestly).  **S = HONEST NEG** (pre-declared sign abort fired, NOT forced): the neutral Perron-channel Γ→P flow is past the srs-*cell* NB convergence radius at u* (u*·(k*−1)=4/3>1); the clean S needs the TREE-COVER neutral-channel running (the framework's own z*-mechanism home per nb_two_vertex Part B) — obstruction precisely located, a separately-scoped future probe, no post-hoc S-definition swap.  No predictions/*.py added (U≈0 is a structural near-vanishing not a tight DAG target; Δκ is a recombination of the existing predictions/delta_rho.py; SM-subtracted PT-parameter files would invite the substrate/observable category conflation).  Net: 4 of 5 SM oblique objects (Δr/δ_r, T/δρ, U, Δκ) are readings of the ONE B_NB resolvent; S obstruction located.
- *2026-05-16 — TREE-COVER S CLOSES + §6.1 RESUMMATION DERIVED (theorem doc §7.5; `proofs/foundations/tree_cover_S_and_resummation_2026-05-16.py`).*  The rigorous k-regular tree cavity Green's function (Kesten/McKay; q·f²−z·f+1=0, g=1/(z−k·f)) gives EXACTLY g(z_triv=k=3)=u*=2/3 (trivial/neutral rep; OFF the McKay support [−2√2,2√2] ⇒ finite — the rigorous regularisation of the cell's divergent Perron pole) and g(z_edge=2√q=2√2)=√q=√2 (on-shell Ramanujan/h_P). The cell obstruction (u*·(k*−1)=4/3>1) is genuinely resolved (z_triv off-support; tree NB radius u*·√q≈0.943<1). **S CLOSES — THEOREM-GRADE-STRUCTURAL:** S inherits δ_r's neutral-channel structure exactly (c_S=1/12, resummed α₁/(1−α₁)) with ONLY the cell-divergent Perron-pole factor replaced by the convergent tree flow [g(2√q)−g(k)]; S = (1/12)(√2−2/3)·α₁/(1−α₁) = 64√2/18915 − 128/56745 = **+0.253%**, K-rational ∈ ℚ(√2)⊂K, δ_r/δρ-class, same sign, no fitted constant, no post-hoc definition swap.  **§6.1 RESUMMATION DERIVED (the grade-lift lever):** the cavity recursion IS the Dyson resummation; its analytic structure derives the form-selection dichotomy — neutral z=k off-support (disc z²−4q=1>0) ⇒ geometric series converges ⇒ Family-C α₁/(1−α₁); on-shell z=2√q (disc=0) ⇒ √ branch point ⇒ leading-only ⇒ Family-E α₁.  This eliminates the **last Clause-7 rigor gap** (the §3.5 form-selection was previously a structural argument); δ_r/δρ/U/Δκ derivation rigor now CLOSED.  Remaining gaps are purely **Clause-8 numerical**: the named +4.58% δρ subleading-spectral residual (a bounded tree branch-expansion found NO clean K-rational closed form — at exact Ramanujan saturation the on-shell point is the branch point itself, ε=0; declared honestly STILL-OPEN, within +0.76σ_obs, not urgent, not forced) and the absolute-mass σ_PDG intrinsic floor.  Net: ALL FIVE SM oblique objects (Δr/δ_r, T/δρ, U, S, Δκ) are now readings of the ONE G_NB=(I−u·B_NB(srs))⁻¹ — Clause-7 fully closed; the unified-oblique theorem's STRUCTURAL label is now purely Clause-8-numerical, not a rigor gap.
- *2026-05-16 — SELECTION-RULE RE-AUDIT (theorem doc §7.6; `proofs/foundations/selection_rule_reaudit_2026-05-16.py`).*  With the form-selection rule now DERIVED, every propagator-level member of the master-doc §5 dark-correction catalogue was re-audited via the Ihara map λ=h+q/h.  **NO MISASSIGNMENT** — off the McKay support (λ=±k=±3, disc>0: v_Higgs, α_GUT, δ_r, S) ⇒ resummed Family-C; on the McKay cut (disc≤0: δρ, m_ν3, β, θ_23, U) ⇒ leading Family-E/Feshbach.  The whole tree-level-coupling resummed-vs-leading taxonomy — previously observable-class heuristics + the v_Higgs c=5/12 calibration anchor — is now **derived-consistent**.  **Numerical impact: ZERO** (no reassignment; not manufactured — the honest finding is a confirmation, not an impact).  Non-trivial validation: v_Higgs (the calibration anchor) is independently *reproduced* off-support/resummed by the criterion it was never given.  **Criterion SHARPENED (correctness fix to committed §7.5):** the rule is disc≤0 (the whole McKay cut), NOT "the band edge z=2√q" — h_P maps to INTERIOR λ=√3 (disc=−5<0), on the cut but not at the edge; §7.5 wording corrected.  **δρ COROLLARY (constrains the open +4.58%):** δρ is on the cut where the geometric resummation does NOT converge ⇒ closing the +4.58% via a 1/(1−α₁) resummation factor is FORBIDDEN by the derived criterion; the residual must be a sub-tree multi-insertion (sub-leading-spectral) sum.  Family-D vertex per-leg (∝α₁²) is a distinct mechanism, correctly out of scope.

- *2026-05-17 — LEADING-ORDER UNIQUENESS CLOSURE (Clause-7 rigor upgrade; ZERO number/grade change).*  The `M_n = 0 for n≥1` truncation in `δρ = (1/2)(√5/4)(2/3)⁸` was, until now, a *generic* rate-distortion-water-filling citation **never verified for the δρ channel** (the m_ν-calibrated/inherited reading).  It is now **discharged channel-specifically**: the framework's own derived MDL threshold (`predictions/uniform_Q_density` Theorem A — Rissanen two-part MDL + Pinsker, theorem-grade: retain a Fourier mode iff `|M_n|²·Δφ > log(N)/N`, a *binary* model-selection rule) at the δρ-channel structural scale **N = 2|E| = 12** (the same cell-NB constant as `c_S = 1/(2|E|) = 1/12`) *forces* M_n=0 — **robustly** (every cell-structural N∈{4,6,12} zeroes M₂; retention would need a non-cell N≳55).  Probes `proofs/foundations/delta_rho_subleading_Mn_waterfilling_2026-05-17.py` + `delta_rho_C1_waterlevel_derivation_2026-05-17.py`; scoping an internal working note §3d.  **SCOPE (resolves the apparent "unique prediction ≠ observation" paradox):** the uniqueness is over the **leading-order substrate object** `δρ_leading = +1.0906%`, NOT the full physical observable.  The framework predicts `δρ_leading` uniquely; it does **not** predict `δρ_full`.  The +4.58% is therefore **not a residual *of the prediction*** (that phrasing was incoherent) — it is the leading-vs-full higher-order separation: a distinct, un-computed physical quantity (the continuum/dispersive Fano-type self-energy on the McKay cut; only domain = the infrastructure-less continuum Kesten–McKay/NA-4 level — finite-matrix Feshbach P/Q proven impossible, `theorem_ifeshbach_percycle_resolution.md` / A5(b)).  Honest test = the scale-free ρ-parameter: leading is **+0.76σ_obs — consistent within experimental error (no disagreement, no falsification)**.  **Net: Clause-7 strengthened (one standing conditional discharged); Clause-8 (+4.58% / +0.76σ_obs) and the THEOREM-GRADE-STRUCTURAL grade UNCHANGED; no number moved.**  The deep-layer §2 object is now the *sole* open δρ item, with the leading term proven unique beneath it.

### Row P72 — λ_3 Higgs trilinear self-coupling (added 2026-05-15 EOD+1)

- *Claim.* λ_3 = m_H² / (2v) = 31.83 GeV ≈ κ_λ × λ_3^SM with **κ_λ = 1** by framework's SM-tree-relation-consistent structure.  Equivalent algebraic form: λ_3 = λ_FD · v (Type 2 identity).
- *Source.* `predictions/lambda_3_higgs.py` (added 2026-05-15 EOD+1).
- *Observed.* Computed from PDG 2024 inputs: λ_3^SM = (125.20 GeV)² / (2 · 246.22 GeV) = 31.83 ± 0.07 GeV.  Direct LHC constraint: κ_λ ∈ [-1.4, 6.1] @ 95% CL (ATLAS 2023 / ATLAS+CMS 2022 combined HH searches).  Framework's κ_λ = 1 is well within current bound; HL-LHC will tighten to κ_λ ∈ [0.5, 1.5] (95% CL projection).
- *Operations invoked.* Algebraic identity m_H² = 2λv² rearranged to λ_3 = m_H²/(2v); upstream m_H (Row P12) and v (Row P10) both theorem-grade with Family D propagated 2026-05-15.  Cross-check via Type 2 identity λ_3 = λ·v.
- *Alternatives.* — None: λ_3 is algebraically determined by m_H + v under SM tree-level structure.  Beyond-SM deviations (e.g., extra scalar interactions, dimension-6 operators) would manifest as κ_λ ≠ 1, but the framework's tree-level Lagrangian is the SM Higgs sector with theorem-grade upstream parameters.
- *Selection.* SM-tree-relation-consistent prediction (Type 3 SM Lagrangian: Peskin-Schroeder §11.1 Higgs sector).  No new structural content beyond Rows P10 + P12.
- *Status.* **UNIQUE — THEOREM-GRADE.**  Algebraic descendant of m_H + v (both theorem-grade with Family D propagated).  σ_PDG-class match -0.04σ_PDG (sub-σ) on the predicted-vs-SM-tree-using-PDG-inputs comparison.  κ_λ = 1 is the framework's falsifiable prediction; HL-LHC direct measurement is the strongest near-term test.
- *Margin.* Predicted κ_λ = 0.99992; SM-tree-relation match by construction since the framework predicts the SM tree-level Higgs Lagrangian structure with theorem-grade upstream parameters.
- *Conditional on.* Row P10 (v_Higgs, UNIQUE-THEOREM-GRADE), Row P12 (m_H, UNIQUE-THEOREM-GRADE via Family D), Row P41 (λ_Higgs, UNIQUE-THEOREM-GRADE via Family D), Row P17 (N_hub adopted).
- *Gap.* — (Closed at predictive-content level.  The 0.5–2σ-σ-class match inherits from m_H Family D propagation.  Future HL-LHC κ_λ measurement will distinguish framework prediction from beyond-SM scenarios.)
- *Filtered-alternative residue.* —

---

## Rows still pending

The parameter-pass walk is now substantively complete: all currently-tracked target parameters in `target_parameters.md` (95 rows across 7 sectors) are accounted for via Rows P1–P59 either directly or via cross-references to upstream structural rows.

Remaining work surfaces only at a few well-characterized closure gaps, all of which match Priority queue items in `docs/master_plan.md`:

- **A5(b) sub-class derivation.** The deepest residual structural input shared across many rows (P3, P4, P7, P39 etc.). Currently a Level-1 prescription input per `../theorems/theorem_A5b_level_prescription.md` (CONDITIONAL); closure from upstream would tighten ~10 parameter rows from CONDITIONAL → THEOREM. Research-level.
- **G1b R2 path FULLY CLOSED 2026-04-28 PM** (an internal working note §5d; theorem `../theorems/theorem_g1b_r2_closure.md`). Three-stage closure all at theorem grade: k=1 of D(ρ_obs(Λ) ‖ (1/3) I_3) ∝ Λ^k at machine precision (`g1b_r2_decay_rate_k_derivation.py`); c=1 in ε_obs = c/N_obs via self-consistency with cascade theorem + per-event granularity (`g1b_r2_residue_closure.py` §1); η=1 uniformly via A1 + A2-T preservation of product class + entanglement decay bound off-framework (`g1b_r2_eta_full_closure.py`). R2 prediction t_now = N_now · t_P matches cascade theorem at machine precision. Six rows P10/P11/P17/P19/P20/P24 graduated STRICT-SOLID-on-G1 → UNIQUE-THEOREM-GRADE clean (no sub-residue). (P23 is conditional on external Ω_b, not G1; unaffected.)
- **arg(h) structural derivation (the M_R phase factor h^g).** Affects P35, P36 (α_21, α_31). Master plan Priority 2.2. **2026-05-12 status: discharge attempted (option B) and FAILED** (`proofs/foundations/majorana_M_R_waterfilling.py`) — the loop-sum route diverges (Ramanujan saturation ⇒ no finite A2-T cutoff), and the Path-B "cardinality-k ↔ k girth rings" route is broken (K_4 cycle-space generators have nonzero Z³ voltage ⇒ don't lift to srs cycles). The h^g factor is an A5(a)-adjacent identification (ADOPTED-NU-MAJ-PHASE); P35/P36 re-graded STRUCTURAL-DERIVATION-CONDITIONAL. Scoping: an internal working note (2026-05-12 update appended). P44 graduated UNIQUE-THEOREM-GRADE 2026-04-29 (β uniqueness closure); P34 was retired 2026-05-02 but **REVIVED 2026-05-05** via the V_{−1}–T_{B-L} identity (commit `0b103c3`) — see Row P34 entry, now THEOREM-GRADE-STRUCTURAL-CONDITIONAL.
- **Color-generation Z₃-eigenvalue identification.** Affects P14, P15, P32, P45 (V_ub, δ_CP_CKM identification, θ_12_PMNS, J_CKM). Currently an internal working note.
- **Quark Yukawa theorem.** Generalises Row P7 (y_τ) to the quark sector — affects P39 (m_u, m_d, m_s, m_c, m_b absolute). Currently mathematically-complete via Koide waterfall (P38 m_top via ADOPTED-Z3) but no individual-quark Yukawa theorem.
- **m_ν Lagrangian (linear-vs-squared).** Affects P31. Priority 3.2 research-level.
- **PS embedding step for θ_13_PMNS.** Affects P33. Priority 4.2 — 1–2 sessions, NOT STARTED.
- **Reconnection perturbation theory for A_s.** Affects A_s in `target_parameters.md`; Priority 3.4 research-level (theory doesn't yet exist).
- **Steady-state oriented-triangle fraction for η_B.** Affects P29. Priority 3.3 specific graph calculation.
- **SUSY spectrum derivation.** Affects all 9 SUSY rows in P58. Priority 4.1 NOT STARTED.

All other parameter-pass rows are either UNIQUE (theorem-grade), STRICT-SOLID (anchored externally on G_F/M_P/Ω_b), or ADVANCED with a specific named closure path. No new residues surfaced in the P19–P59 batch — methodology continues calibrating as eliminative.

---

## Methodological observations from the full parameter pass (P1–P59)

(Earlier observations from the P1–P8 starter batch are preserved below for historical record.)

**Pass-wide (P1–P59) findings:**

A. **Closure profile across 59 rows (snapshot 2026-04-28 PM, post full G1b R2 closure + η-sketch elimination + dark-map Class-2 closure).** Roughly **52+ rows close UNIQUE-THEOREM-GRADE** conditional on structural-pass rows + A5(b) prescription class (six cosmological/Higgs/m_τ rows P10 v_Higgs, P11 m_τ family, P17 N_hub, P19 H_0, P20 t_0, P24 Λ_CC graduated via G1b R2 path with k=1 + c=1 + η=1 ALL theorem-grade; P12 m_H additionally graduated via composition of dark-map Class-2 closure + G1b R2 closure on v_Higgs); ~10 rows are ADVANCED-but-BLOCKED on one of the named priority-queue gaps (arg(h), color-generation Z₃, m_ν Lagrangian, SUSY spectrum, etc.); 1 row mathematically-complete only (P23 Ω_DM, conditional on external Ω_b). The pre-2026-04-28 "STRICT-SOLID-on-G1" cluster has been **fully reframed** and graduated: G1b's H1 reframe via R2 path is closed at uniformly theorem grade with no sub-residue. Net: largest single-session graduation event in the parameter pass to date.

B. **The (k\*−1)/k\* = 2/3 fundamental constant is omnipresent.** P1, P2, P3, P5, P7, P14 (candidate), P22 all factor through (2/3)^L for various L — confirming the starter-batch finding that the framework's bare numerical content factors through this single base.

C. **All non-trivial gaps reduce to a small set of named structural-input requirements.** No rogue ad-hoc parameters surfaced anywhere in P19–P59; every "BLOCKED" or "CONDITIONAL" row routes back to one of the named gaps (G1, A5(b) sub-class, arg(h), color-generation Z₃, dark-map class 2, m_ν Lagrangian, SUSY commitment). This is strong evidence that the framework's parameter pass is structurally well-posed.

D. **No new residues surfaced across the parameter pass.** Every operator-permitted alternative formula at every row routes back to either (i) a hard-gating structural ledger row, (ii) an R-N residue already in the residue register (R-12 R-2-style waterline-vs-strict-min, R-7 chirality, etc.), or (iii) a named closure gap from §A. The structural-residue register stays at 12 entries (10 REFUTED, 1 RESTRICTED, 1 ACCOUNTED-FOR + structural filter, 0 OPEN).

E. **The cosmological sector now FULLY THEOREM-GRADE with NO sub-residue (2026-04-28 PM).** P19, P20, P24 all reduce to N_hub anchoring (Row P17, now UNIQUE-THEOREM-GRADE); P21 (w_DE = −1) and P22 (Ω_DM/Ω_m) were already independently THEOREM-GRADE. The G1b R2-path closure (`proofs/foundations/g1b_r2_decay_rate_k_derivation.py` + `g1b_r2_residue_closure.py` + `g1b_r2_eta_full_closure.py`) is the highest-leverage parameter-pass payoff of the session: six rows graduated STRICT-SOLID-on-G1 → UNIQUE-THEOREM-GRADE (clean, no caveats), with predicted t_now = N_now · t_P matching cascade theorem at machine precision (k = 1 + c = 1 + η = 1 all theorem-grade). The framework's cosmological-sector predictions are now structurally derived from {A1} alone with no sub-residue. (P23 Ω_DM remains separate — its gap is external Ω_b, not G1.)

F. **Lorentz-violation predictions fully theorem-grade SYMBOLIC** post 2026-04-27 closure: P42 (η_5 = 0) + P43 (η_lattice = 1/12) are now both rational closures via Feshbach-Löwdin + Ihara cross-walker theorems. These are the framework's sharpest falsifiable astroparticle predictions.

---

## Methodological observations from the starter batch

1. **Most parameter rows close UNIQUE conditional on structural-pass rows + an A5(b) prescription class.** A5(b) is the framework's "coupling = MDL probability" identification, with three Levels (Level 1 toggles, Level 2 srs-intrinsic, Level 3 Hashimoto walk-sum). Each parameter's derivation invokes A5(b) at a specific Level, and the prescription class within that Level (direct-moment, walk-sum, counting fraction) is structurally classified per `../theorems/theorem_A5b_level_prescription.md`.

2. **The deepest remaining structural input at the parameter level is A5(b)'s sub-class identification.** Currently per `theorem_A5b_level_prescription.md`, the criterion is "derivation-structure (how α₁ enters the formula)." Closure of A5(b) from upstream rows would replace this Level 1 prescription input with derivation; that's research-level work. Until then, parameter rows are CONDITIONAL on A5(b) Level + sub-class.

3. **No new residues surfaced in the starter batch.** All operator-permitted alternative formulas are either hard-gated by structural-pass rows or covered by R-12 (waterline-vs-strict-min retention pattern, structural residue). The parameter-pass methodology is calibrated similarly to the structural pass: primarily eliminative.

4. **Foot 1994 cross-check at Q_Koide = 2/3** is a clean external-literature validation: the framework's geometric derivation matches a known geometric identity, providing strong evidence that the Cl(6;ℂ) + C₃-observer structure is correct.

5. **One pattern worth flagging:** several parameters (P1, P2, P3, P5, P7) ultimately route through (2/3)^L for various L — the (k\*−1)/k\* survival rate is the framework's *fundamental* numerical constant, with everything else being a graph-theoretic exponent or rational coefficient on this base. *Followup:* this factorization was promoted to structural Row 23 (q_NB = 2/3, UNIQUE-THEOREM-GRADE) on 2026-04-28; the closely-related observation that the substrate's Hashimoto operator splits into 1+6+5 sectors (Perron, oscillatory, marginal — encoding Class A growth, Class B dispersion, Class A dark respectively) became Row 24 the same day.

---

## Audit v2 (Clause 7) inheritance index — added 2026-05-01 M1b.i

**Purpose.** Per `parameter_linter.md` §7, every UNIQUE-THEOREM-GRADE row must EITHER carry its own audit v2 §3 table OR explicitly cite an inheritance from an internal working note. Pre-2026-04-30 UNIQUE rows that lack either form carry implicit conditional per §7 own text.

This section provides the explicit citation for inheritance-cascade rows. Rows whose alternative-axis structure is not fully covered by upstream closures are flagged PHASE-3-PENDING with the specific gap named.

**M1a triage:** see an internal working note for the full triage scoreboard. **Net:** 8 rows had explicit audit v2 citation pre-M1b.i; 33 rows were PRE-V2-IMPLICIT; this section closes the inheritance-only sub-set and flags the residual Phase-3 work explicitly.

**Citation key:**

- `[Row 4 inh]` — inherits Row 4 (k* = 3) closure per closures-index §2.3. Applies whenever the row's *Conditional on* field includes Row 4.
- `[Foundational inh]` — inherits Rows 1-7 + R-12 closures per closures-index §3. Applies for rows whose only structural inputs are foundational rows + R-12.
- `[G_sub Drude inh]` — inherits closures-index §3.5. Applies to G_sub-related rows.
- `[Layer-bridge inh]` — inherits closures-index §3.6 (Bloch-decomposable scope). Applies to mass-identification chain.
- `[G1b R2 anchor]` — anchored to G1b R2 closure (separate theorem, NOT audit v2). The Row 4 / cosmological-cascade dependencies inherit from §2.3; the G1b-specific structural derivation is a standalone theorem `../theorems/theorem_g1b_r2_closure.md`. Per linter §7's spirit, audit-v2 inheritance applies to the upstream Row 4 dependency only; the G1b R2 piece is structurally derived but does not have its own audit-v2 §3 table.
- `[PHASE-3-PENDING: <axis>]` — was the M1a flag for rows needing per-row §3 work. **All 8 such rows had their §3 tables written in M1b.ii** and now pass Clause 7. Tag retained below for traceability; see "M1b.ii §X" cite for the actual §3 location.

### Per-row inheritance citations

| Row | Param | C7 inheritance |
|---|---|---|
| P1 | α_1_bare = 256/6561 | `[Row 4 inh]` + `[Foundational inh]` |
| P2 | α_1_full = 256/6305 | `[Row 4 inh]` + `[Foundational inh]` (waterline retention is R-12 covered) |
| P6 | sin²θ_W (M_unif) = 3/8 | `[Foundational inh]` + Row 17 (Pati-Salam) inheritance — Pati-Salam embedding closure is in `../theorems/theorem_sin2_theta_W_unification.md`; not in closures-index but theorem-grade structural |
| P7 | y_τ = 1280/177147 | `[Row 4 inh]` + `[Foundational inh]` + A5(b) Level 2 prescription. Clause 8 = `-STRUCTURAL` (Yukawa chain) |
| P8 | Q_Koide = 2/3 | `[Foundational inh]` + Pati-Salam (Row 17) + C₃-observer (Row 18) — Cl(6;ℂ) inheritance |
| P9 | ε²_K = 2, δ_K = 2/9 | Same as P8 |
| P10 | v_Higgs = 246.22 GeV | `[G1b R2 anchor]` + Row 17/18 — g3 coefficient theorem grade per `../theorems/theorem_g3_higgs_coefficient.md` |
| P11 | m_τ family | Inherits P7 (y_τ) + P10 (v_Higgs); Clause 8 = `-STRUCTURAL` (inherits y_τ) |
| P12 | m_H = 125.20 GeV (Family-D corrected) | PASS-CITED via own §3 in M1b.ii §3.4. Cl(2) edge-qubit theorem hard-gates Higgs-sector formula structure; P2 + P10 inheritance covers α_1_full + v_Higgs axes. **W1 2026-05-18 reclassification:** Clause 8 numerical PASS (Family-D-corrected, −0.05σ_PDG) is preserved, but the grade label is THEOREM-GRADE-STRUCTURAL (not THEOREM-GRADE-NUMERICAL) because the 2026-05-15 "UNIQUE-NUMERICAL via Family-D" label was a Clause-6c smuggle — c_F is via the Clause-6 channel_select → canonical_encoding two-step; numeric values unchanged. |
| P16 | θ_QCD = 0 | `[Row 4 inh]` + `[Foundational inh]` + Z_3 flat-connection theorem (Op 4.20 + Kobayashi-Nomizu); standalone but inherits k=3 |
| P17 | N_hub | `[G1b R2 anchor]` — anchor row, audit-v2 inheritance is on the cosmological-cascade dependencies (Row 4 + downstream) |
| P19 | H_0 = 68.18 km/s/Mpc | PASS-CITED via M1b.ii §3.5 (cosmology trio shared §3) + `[G1b R2 anchor]` + `[Row 4 inh]`. Clause 8 = Cat-B PASS (8e accommodation) |
| P20 | t_0 = 14.38 Gyr | Same as P19 (M1b.ii §3.5) |
| P21 | w_DE = −1 | `[Foundational inh]` (A1 absence-of-DOF). Standalone argument — audit-v2 §3 axes (alternative DE-field DOFs) gated by A1 alphabet structure (Row 1 closure). |
| P22 | Ω_DM/Ω_m = 1−61·e⁻⁶ | PASS-CITED via M1b.ii §3.1. Jaynes max-ent on ℕ uniquely selects Poisson (M2 hard gate); A2-T waterline at k* gated by Row 11 inheritance. Lattice waterfilling probe `substrate_lattice_waterfilling_omega_dm.py` in working tree (uncommitted) confirms numerically. |
| P24 | Λ_CC = 3/N² | Same as P19/P20 (M1b.ii §3.5 cosmology trio shared §3) + `[G1b R2 anchor]` + `[Row 4 inh]`. Clause 8 = Cat-B PASS |
| P27 | A_hemi value 1/15 | `[Row 4 inh]` + Bayesian-toggle setup (Stage 2c) + srs cubic-moment. Identification-side OTHER-SMUGGLE flagged separately in row body. |
| **P30** | **R_ν = 228/7** | **PASS-CITED via own §3 in M1b.ii §3.6**. Function basis + distance value + distance functional all gated by Stark-Terras 1996 + Ihara 1966 + cubic-uniqueness + Row 4 inheritance. Lattice-axis numerical waterfilling probe not yet written (analytic argument parallel to V_cb/V_us); flagged honest small completeness gap, not uniqueness vulnerability. Clause 8 soft tension at +1.4σ vs NuFIT 6.0 logged separately. |
| P37 | quark Koide ratio = 14/5 | `[Row 4 inh]` + `[Foundational inh]` + Pati-Salam Cl(6) (Row 17) + A5(b) |
| P40 | α_GUT = 1/24 | PASS-CITED via M1b.ii §3.2. Uniform max-ent on (Fock × direction) is the unique max-ent prior on a finite set with no constraints (M2 hard gate via Jaynes 1957); local CAR (Row 15a) inheritance + Row 4 inheritance for k*-axis. |
| P41 | λ_Higgs = 2560/19683 | Inherits P2 + Cl(2) edge-qubit theorem (Row 22) + tan²(arg h)=5/3 from P3. Same Clause 8 systematic as P12 — **W1 2026-05-18 reclassification:** THEOREM-GRADE-STRUCTURAL (not THEOREM-GRADE-NUMERICAL); the 2026-05-15 "UNIQUE-NUMERICAL via Family-D" label was a Clause-6c smuggle, numeric value unchanged (Family-D-corrected λ_physical = 0.129269 → −0.05σ_PDG). (Bug-fix 2026-05-02: was `512/6305` — wrong α_1_full convention; correct is 2·(5/3)·α_1 per `lambda_higgs.py`.) |
| P42 | η_5 = 0 | `[Foundational inh]` + Stage 3 + parity-even dispersion. Standalone but no alternative-axis ambiguity (parity-even at dim-5 is structurally forced by I4₁32). |
| P43 | η_lattice = 1/12 | PASS-CITED via M1b.ii §3.3. Stark-Terras 1996 + Ihara 1966 (M6 hard gate) uniquely fix the cross-walker map; Feshbach-Löwdin symbolic closure (M5) gates dispersion-expansion order; Row 4 inheritance for k*-axis. |
| P49 | d_spatial = 3 | `[Foundational inh]` (Row 3 Cencov-Fisher) |
| P50 | g_girth = 10 | `[Foundational inh]` (Row 6 srs + Row 9 Sunada) |
| P51 | p_toggle = 2 | `[Foundational inh]` (Row 1 + R-1 refutation hard-gated alternatives) |
| P52 | h_walker eig at P | `[Foundational inh]` + Stark-Terras 2007 |
| P53 | srs_E_at_P, cubic moment | `[Foundational inh]` |
| P54 | Stage 2c bundle | `[Foundational inh]` (Row 1 + Stage 2c arrow-of-time) |
| P55 | Hashimoto scale ~147 PeV | Inherits P52 + Stage 3 |
| P56 | μ uniform; H = C³ | `[Foundational inh]` + Pati-Salam (Row 17) + n_gen=3 (Row 18) |
| P57 | structural bundle (gauge group, n_gen, etc.) | Multi-row inheritance per row body. Y=+1/2 conditional on ADOPTED-B3 (un-graduated). |

### Per-row inheritance citations — added 2026-05-26 (consolidation pass coverage)

Rows previously listed in the "PASS-CITED before M1b.i" summary line (P29, P60, P61) but absent from the per-row table above, plus the full gauge cluster (P63–P71) which carries UNIQUE-THEOREM-GRADE-CONDITIONAL labels via the 5-stage gauge-coupling closure (2026-05-04 EOD+1) but had no inheritance entry. M_persistence cluster (P38–P39) added at THEOREM-GRADE-STRUCTURAL-CONDITIONAL (2026-05-26).

| Row | Param | C7 inheritance |
|---|---|---|
| P29 | η_B = (√3/10)·(2/3)⁴⁸ ≈ 6.11×10⁻¹⁰ | PASS-CITED via own §3 (Sakharov-Hashimoto chain 2026-04-30). M1 hard gate via `[Row 4 inh]` + Sakharov-conditions audit; ε_CP=1/5 (Row P28) + Re(h_P)=√3/2 (Row P52-class) + α₁^M Hashimoto persistence; Clause 8 PASS −0.20σ_PDG. |
| P60 | G_N·M_Pl² = 1 (Newton's constant identity) | `[Foundational inh]` + Drude + Planck-convention chain (THEOREM-GRADE-CONDITIONAL 2026-04-30). M1 hard gate: unit-setting identity in framework-natural units; no alternative formulation in K consistent with substrate-Planck identification (Row 25). |
| P61 | M_substrate/M_Pl = √π/8 (substrate-Planck identification) | PASS-CITED via own §3 (closures-index §3.5, G_sub Drude form 2026-04-30 EOD final). M1 hard gate: Drude asymptote G_UV·M_substrate² = π/(16·N_atoms) + Planck convention G_N·M_Pl² = 1 ⇒ closed-form (M_Pl/e_bit)² = 64/π. Untethered structural prediction, no N-dependence; the GeV value is unit conversion via CODATA, not framework prediction. |
| P38 | m_t = 174.10 GeV (M_persistence + Type-II saturation) | THEOREM-GRADE-STRUCTURAL-CONDITIONAL inheritance via M_persistence theorem (`../theorems/theorem_fermion_mass_operator_persistence_2026-05-21.md`) + Type-II saturation y_t(GUT)=1 + MSSM RGE. Not a UNIQUE-class row (Clause 8 FAIL on σ_PDG, +4.71σ, MSSM-threshold residual class); listed for completeness of M_persistence cluster. |
| P39 | m_u/d/s/c/b (M_persistence + δ(n) PS Fock counting) | THEOREM-GRADE-STRUCTURAL-CONDITIONAL inheritance via M_persistence + δ(n)=2/(9(n+1)) PS Fock counting (`proofs/masses/srs_delta_n_derivation.py` + W3 PS sector connectivity closure). Four channels Clause 8 PASS vs σ_PDG; m_b borderline (+2.99σ). Not UNIQUE-class; listed for completeness. |
| P46 | tan β (live 60.07 / documented 44.73 — DISAGREEMENT) | STRUCTURAL-DERIVATION-CONDITIONAL (downgraded from THEOREM-GRADE-STRUCTURAL-CONDITIONAL on 2026-05-26): live MSSM RGE chain root sits at tan β ≈ 60.07; documented framework value 44.73 from `proofs/masses/srs_tan_beta.py` disagrees by ~35%. Pre-2026-05-26 the live chain returned 44.73 only via an exception-fallback `except: return 44.73` that masked brentq's failure to bracket a root. Disagreement-reconciliation is open work (Row P46 *Gap*). |
| P40 | α_GUT = 1/24 + DC | PASS-CITED via own §3 in M1b.ii §3.2 (already in main table). |
| P63 | α_EM(M_Z) ≈ 1/127.93 | UNIQUE-THEOREM-GRADE-CONDITIONAL via 5-stage gauge-coupling closure (2026-05-04 EOD+1, `../theorems/theorem_gauge_unification_RG_closure.md` + `proofs/foundations/gauge_unification_full_RG_closure.py`). Inheritance from P40 (α_GUT) + `theorem_beta_coefficients_derived.md` (β-coefficients mathematically-complete grade 2026-05-14) + Row 4 inheritance for k*-axis. Clause 8 PASS +1.01σ_PDG. |
| P64 | M_Z = 91.20 GeV (tree → pole via δ_r) | THEOREM-GRADE-STRUCTURAL-CONDITIONAL via same 5-stage chain + δ_r (Row P64-sibling, c_S=1/12 derived 2026-05-16 unified-oblique). Clause 8 FAIL on σ_PDG (+7.76σ; intrinsic SM tree-vs-pole oblique floor at 2.3 ppm, NOT M_unif issue per ffa89dc/9501a65 decomposition). |
| P65 | sin²θ_W(M_Z) = 0.23125 | UNIQUE-THEOREM-GRADE-CONDITIONAL via 5-stage chain + sin²θ_W(M_unif)=3/8 exact theorem (P6) + MSSM RG. Clause 8 PASS +0.96σ_PDG. |
| P66 | g_1(M_Z) = 0.46148 | UNIQUE-THEOREM-GRADE-CONDITIONAL via 5-stage chain. Clause 8 PASS +0.37σ_PDG. |
| P67 | g_2(M_Z) = 0.65175 | THEOREM-GRADE-STRUCTURAL-CONDITIONAL via 5-stage chain. Clause 8 near-PASS −2.52σ_PDG (dev only −0.04%; FAIL at strict σ_PDG); 2-loop MSSM + SUSY-threshold class residual. |
| P68 | g_3(M_Z) = 1.211 | OUT-OF-SCOPE-BY-CONSTRUCTION (re-grade 2026-05-17, Move-1): inherits the omitted IR b,c,τ+HVP threshold matching that the single-regime no-threshold scheme excludes by construction. Not a defect at structural-derivation level; listed for completeness. |
| P69 | α_s(M_Z) = 0.11674 | Same as P68 (g_3² = 4π·α_s); OUT-OF-SCOPE-BY-CONSTRUCTION (Move-1). |
| P70 | R∞ (Rydberg) | OUT-OF-SCOPE-BY-CONSTRUCTION (re-grade 2026-05-17, Move-1): R∞ = α(0)²·m_e·c/(2h) is dependent on Δα = α(0) − α(M_Z), the excluded IR threshold layer (β-class — must NOT be patched in). Listed for completeness. |
| P71 | m_W = 80.40 GeV | THEOREM-GRADE-STRUCTURAL-CONDITIONAL via 5-stage chain + δρ (Row P73 custodial-breaking). Clause 8 FAIL on σ_PDG (+2.39σ; inherits M_Z intrinsic floor); CLEAN scale-independent ρ-test (δρ validation): +0.76σ_obs. |
| P72 | λ_3 (Higgs trilinear) = 31.83 GeV | UNIQUE-THEOREM-GRADE inheritance from P10 (v) + P12 (m_H Family-D); algebraic Type-4 from m_H²/(2v). Clause 8 PASS −0.04σ. κ_λ = 1 by construction (framework predicts SM tree-level Higgs Lagrangian). |
| P73 | δρ (custodial-breaking) = (1/2)·(√5/4)·(2/3)⁸ | UNIQUE-THEOREM-GRADE via Hashimoto spectral object + W/h_P eigen-channel of unified-oblique G_NB resolvent (`../theorems/theorem_unified_oblique.md` 2026-05-16). Clause 7 PASS via 2026-05-17 Leading-Order Uniqueness Closure (uniform_Q_density Theorem A discharge); Clause 8 +0.76σ_obs PASS. |

### M1c — V_ub family + β graduations (propagated 2026-05-01 evening)

This sub-section captures Status updates for 9 rows whose closures landed 2026-04-29 / 2026-04-30 but whose Status fields had not been propagated to the parameter ledger before the 2026-05-01 evening hygiene pass. The closures themselves are not new; only the Status-field propagation is.

| Row | Param | Pre-2026-05-01 Status | Post-propagation Status | Closure citation |
|---|---|---|---|---|
| P14 | V_ub | STRICT-SOLID conditional on ADOPTED-A5b-Sub3 | UNIQUE-THEOREM-GRADE for amplitude; labeling data-anchored | M1 amplitude-form 2026-04-30 (commit 753f4cf, `proofs/foundations/m1_twisted_walker_v_cb_v_ub.py` + `m1_n_orbit_3orbit_basis.py`) + Angle D + Z3-mass-order verdicts (commit e5ef667, an internal working note + `adopted_z3_mass_order_audit_2026-04-30.md`) |
| P15 | δ_CP_CKM | STRICT-SOLID conditional on ADOPTED-A5b-Sub3 | UNIQUE-THEOREM-GRADE for geometric value; labeling data-anchored (inherits P14) | Inherits Row P14 |
| P32 | θ_12_PMNS | STRICT-SOLID conditional on ADOPTED-A5b-Sub3 | UNIQUE-THEOREM-GRADE for structural form; labeling data-anchored (inherits P14) | Inherits Row P14 |
| P33 | θ_13_PMNS | ADVANCED — STRICT-SOLID + BLOCKED (two open gaps) | ADVANCED — sub-class data-anchored (non-blocking, inherits P14); BLOCKED on PS embedding step (Priority 4.2 — independent gap remaining) | Sub-class part inherits Row P14; PS embedding gap unchanged |
| P34 | δ_CP_PMNS | ADVANCED — BLOCKED on arg(h) Path B'' + STRICT-SOLID on adoption | **THEOREM-GRADE-STRUCTURAL-CONDITIONAL** (revived 2026-05-05) — δ_CP_PMNS = arccos(T_{B-L,lepton}) = arccos(−1) = 180°; Clause 8 PASS +0.16σ vs NuFIT 6.0 | See Row P34 entry above + live DAG node `predictions/delta_CP_PMNS.py`. The old `(g−1)·arg(h*) ≈ 249.85°` formula WAS retired 2026-05-02 (an internal working note, +3.8σ NuFIT 6.0); the V_{−1}–T_{B-L} symmetry-breaking identity (commit `0b103c3`, W3-reconciled `9d32d59`) supersedes it. Conditional on the framework-wide CKM↔K_4-walks identification (shared with Row P15); Need-D-3 closed 2026-05-21 (`theorem_selection_map_2026-05-21.md`). |
| P35 | α_21_PMNS | STRUCTURAL-DERIVATION-CONDITIONAL (re-graded 2026-05-12) | c=1 theorem-grade; M_R phase h^g = identification (ADOPTED-NU-MAJ-PHASE), not derived (discharge FAILED — `majorana_M_R_waterfilling.py`) | ADOPTED-NU-MAJ-PHASE + C³_gen L3 + ADOPTED-B3 |
| P36 | α_31_PMNS | STRUCTURAL-DERIVATION-CONDITIONAL (re-graded 2026-05-12) | same as P35 | same as P35 |
| P44 | β cosmic birefringence | ADVANCED — BLOCKED on arg(h) | **UNIQUE-THEOREM-GRADE** (full graduation) | β uniqueness closure 2026-04-29 + algebraicity meta-theorem upgrade (`docs/theorems/theorem_beta_uniqueness_closure.md` + `theorem_lattice_coupling_algebraicity.md` Path B; commit 3aaa473) |
| P45 | J_CKM | STRICT-SOLID conditional on ADOPTED-A5b-Sub3 | UNIQUE-THEOREM-GRADE for amplitude form; labeling data-anchored (inherits P14, P15) | Inherits Rows P14, P15 |

**Note on labeling residue.** (P35/P36 re-graded 2026-05-12 to STRUCTURAL-DERIVATION-CONDITIONAL — see their rows; the M_R phase factor h^g is an identification, not derived. The labeling-residue note below predates that and applies to the c-coefficient layer.) Seven of these rows (P14/P15/P32/P33-sub/P35/P36/P45) have a labeling layer that is OTHER-SMUGGLE under parameter_linter rigor bar (the framework's substrate-derived multiset {α_m} is invariant under (Z/2)^3 within-generation × S_3 mass-ordering relabeling per Angle D + Z3-mass-order verdicts; the physical name attached to each multiset element is anchored to PDG empirically, not derived structurally). The Clause 7 PASS applies to predictive content; the labeling residue is non-blocking for predictive content but is a real OTHER-SMUGGLE remnant to be closed by future research (an internal working note lists M1 + M2 routes; M1 amplitude-form closed pieces (i)+(ii) of those routes).

**Note on R-9 srs-z — CLOSED 2026-05-02 EOD+8.** Per `proofs/foundations/r9_srs_z_polynomial_derivation.py` (commit `843cfc9`): srs-z's Wyckoff 8c free parameter $x \approx 0.6607$ is the irrational root of the explicitly-derived 3-regularity boundary polynomial $16x^2 - 32x + 15 = 0$ (degree 2, integer coefficients $\leq 32$). Costed under γ.2 algebraic-K-complexity (Lutz 1998 computable-real Kolmogorov complexity), Wyckoff free-parameter encoding adds 19.07 bits to srs-z's structural DL. Combined with +2.40 bits Level-2 ΔDL, total ΔDL(srs-z − srs) = 21.47 bits, exceeding the sub-1σ V_us-match threshold of 7.39 bits by +14.08 bits. **R-9 closes to sub-1σ via M2a structural alone**, conditional on adopting γ.2 algebraic-K-complexity as the MDL convention for Wyckoff free parameters. This earlier-noted "2.56-bit M2a gap" is now resolved via the polynomial-encoding refinement (γ.2), which the earlier Level-2-only accounting (`srs_vs_srs_z_dl_audit.py` 2026-05-01 PM) did not include.

### Net after M1b.i + M1b.ii + M1c (V_ub family + β)

- **8 rows PASS-CITED before M1b.i** (P3, P4, P5, P28, P29, P48, P60, P61).
- **+25 rows PROMOTED PRE-V2-IMPLICIT → PASS-CITED via inheritance** (M1b.i, this section).
- **+8 rows PROMOTED PHASE-3-PENDING → PASS-CITED via own §3** (M1b.ii, see an internal working note):
  - P12 m_H — §3.4
  - P19 H_0, P20 t_0, P24 Λ_CC — §3.5 (cosmology trio shared §3)
  - P22 Ω_DM/Ω_m — §3.1
  - P30 R_ν — §3.6 (lattice-axis numerical waterfilling probe `proofs/foundations/substrate_lattice_waterfilling_R_nu.py` written 2026-05-01 evening; confirms shift = 0 exactly)
  - P40 α_GUT — §3.2
  - P43 η_lattice = 1/12 — §3.3
- **+8 rows PROPAGATED via M1c** (V_ub family + β, see table above; net 8 after P34 retirement 2026-05-02):
  - P14, P15, P32, P45 — UNIQUE-THEOREM-GRADE for predictive content; labeling data-anchored (OTHER-SMUGGLE residue)
  - P35, P36 — STRUCTURAL-DERIVATION-CONDITIONAL (re-graded 2026-05-12): c-coefficient theorem-grade, but the M_R phase factor h^g is an A5(a)-adjacent identification (ADOPTED-NU-MAJ-PHASE), not derived from A2-T (discharge attempted and FAILED — `proofs/foundations/majorana_M_R_waterfilling.py`)
  - P33 — sub-class data-anchored (non-blocking); PS embedding gap CLOSED 2026-05-02 via Class-2/Class-3 selection rule
  - P44 — full UNIQUE-THEOREM-GRADE
  - P34 — REVIVED 2026-05-05 (V_{−1}–T_{B-L} identity) to THEOREM-GRADE-STRUCTURAL-CONDITIONAL; carries the CKM↔K_4-walks conditional so is NOT UNIQUE-THEOREM-GRADE — hence still outside the UNIQUE-grade count below, but no longer retired

**Net: 49 UNIQUE-THEOREM-GRADE rows now have explicit Clause 7 PASS** — P34 (revived 2026-05-05 to THEOREM-GRADE-STRUCTURAL-CONDITIONAL) is not UNIQUE-grade, so it stays outside this count. 7 of the 8 newly-propagated rows have OTHER-SMUGGLE labeling residue (not blocking for predictive content; flagged for future structural closure). All graduations conditional on srs substrate identification — **R-9 srs-z structural axis CLOSED 2026-05-02 EOD+8 via polynomial γ.2** (see note above; Wyckoff free-parameter ΔDL = 21.47 bits, +14.08-bit margin to sub-1σ V_us threshold).

**2026-05-26 consolidation pass — coverage additions to inheritance index.** Added explicit per-row inheritance citations for P29, P60, P61 (PASS-CITED before M1b.i but missing from the per-row table), the gauge cluster P63–P71 (UNIQUE-THEOREM-GRADE-CONDITIONAL via the 5-stage gauge-coupling closure with no prior inheritance entry), and the M_persistence cluster P38/P39/P46 + Higgs sibling P72 + custodial-breaking P73 (theorem-grade-structural-conditional or UNIQUE per row). Also W1 2026-05-18 reclassification (THEOREM-GRADE-NUMERICAL → THEOREM-GRADE-STRUCTURAL via the c_F Clause-6 channel_select → canonical_encoding two-step) is now explicitly noted in the inheritance entries for P12 and P41 (numeric values unchanged). The structural-conditional class (P38, P39, P46, P64, P67, P68, P69, P70, P71) is listed for completeness — these rows have Clause 7 inheritance but are NOT counted in the "49 UNIQUE-THEOREM-GRADE" total, which remains UNIQUE-class only.

---

## CLASS A/B/C/D/E metadata under bipartite cover (2026-05-01 post-EOD; P1.6 anchor)

Predictions tagged by behavior under the bipartite double-cover relation
srs's K_4 ↔ srs-z's Q_3 (per `proofs/foundations/srs_z_partner_predictions.py`).

| CLASS | Depends on | Behavior under srs ↔ srs-z cover |
|---|---|---|
| A | (k*, g) only | SAME on both substrates |
| B | (\|V\|, \|E\|, k*) | CHANGES on srs-z (\|V\|, \|E\| double) |
| C | h saddle complex value | SAME (K-rational h preserved under cover) |
| D | h saddle multiplicity n_γ | CHANGES (mult 2 → mult 4 by bipartite splitting) |
| E | Pati-Salam embedding | SAME (Cl(2k*) = Cl(6) at k* = 3 unchanged) |

Per-row CLASS tags (canonical assignments):

| Row | Parameter | CLASS | Notes |
|---|---|---|---|
| P1, P2 | α_1, α_1_full | A | (k*, g) only |
| P3 | V_cb | A | (k, g, n_fixed) only |
| P4 | V_us | B | counting density k*²/(g·N_atoms) — N_atoms doubles |
| P5 | dark c = 5/12 | B | (2(\|E\|−\|V\|)+1)/(2\|E\|) |
| P6 | sin²θ_W | E | Pati-Salam Cl(6) |
| P7 | y_τ | A | (k*, g) only |
| P8 | Q_Koide | A | k* only |
| P10 | v_Higgs | depends on the adopted N_hub (via BZJ) | (matches by construction — N_hub is calibrated via G_F; like the G_F round-trip) |
| P14 | V_ub amplitude | A | (k, g) only |
| P15 | δ_CP_CKM | C | arccos(1/3) from h saddle structure |
| P28 | ε_CP | A | (k−2)/(k+2) |
| P29 | η_B | A + C mixed | ε_CP·Re(h)·α_1^M (M = \|E\| changes by CLASS B!) |
| P30 | R_ν | A | (k*, g) only |
| P40 | α_GUT | E | Cl(6) Pati-Salam |
| P41 | λ_Higgs | A | 2 · (5/3) · α_1  (= 2·c_mass·α_1, where c_mass = tan²(arg h) = 5/3) |
| P44 | β | C | sin(arg h) · α_EM |
| P45 | J_CKM | A | inherits Row P14 amplitude family |
| P52 | h_walker_eigenvalue | C | (√3+i√5)/2 |

## V_ub route (c) CKM unitarity-triangle consistency (2026-05-01 EOD)

Per `proofs/foundations/v_ub_unitarity_triangle_route_c.py`, the framework's
four INDEPENDENT theorem-grade amplitudes
{V_us = 9/40 (P4), V_cb = 256/6305 (P3), V_ub ≈ 3.767e-3 (P14), δ_CP_CKM
= arccos(1/3) (P15)} form a SELF-CONSISTENT unitary CKM matrix. Wolfenstein
parameters vs PDG 2024 global fit:

| Parameter | Framework | PDG | Δ/PDG |
|---|---|---|---|
| λ = V_us | 0.22500 | 0.22500 | +0.00% |
| A = V_cb/λ² | 0.802 | 0.826 | −2.90% |
| ρ̄ | 0.137 | 0.159 | −13.55% |
| η̄ | 0.388 | 0.348 | +11.71% |
| J_CKM | 3.16e-5 | 3.08e-5 | **+2.56%** |

Unitarity V·V† = I to machine precision (constructed in standard
parameterization). Unitarity-triangle closure |Σ V_iX V_iY*| < 1.8e-18.

**What this delivers:** four DIFFERENT structural derivation chains (Level-2
counting, Level-3 walk-rep, M1 multi-cycle, regular-tetrahedron geometry)
land on a coherent unitary CKM matrix with PDG-consistent Wolfenstein
parameters. Cross-verifies the framework's CKM sector internal consistency.

**What this does NOT deliver:** does NOT promote Row P14 from
"amplitude-theorem-grade, labeling-data-anchored" to "labeling-derived".
Identifying which framework amplitude is V_ub specifically (vs V_cb, V_us)
still requires PDG empirical labeling. Wolfenstein has 4 real DOF; the
framework's 4 amplitudes provide 4 inputs but they all originate from the
SAME M1-amplitude family at the labeling layer. An independent labeling
derivation requires structural identification of u/d/c/s/t/b pinnings —
route (a) (Z_3-asymmetric generation) remains open; B1 (χ̃ × C_3) refuted.

**Status:** Row P14 unchanged ("amplitude-theorem-grade, labeling-data-anchored"
+ unitarity-triangle CONSISTENCY CHECK PASSED).

**Important refinement (P1.1 finding 2026-05-01 EOD):** CLASS C is specifically
about COVER-PAIR invariance (srs ↔ srs-z, both built on K_4-related primitives).
**It does NOT extend to other bipartite-primitive substrates that are not covers
of srs's K_4.** Per `proofs/foundations/srs_z_beta_invariance_probe.py`, lov
(the second bipartite-primitive substrate from
`proofs/foundations/rcsr_candidate_sweep.py`) has 12 primitive vertices, 72
arcs, and saddle |λ|² = 5 — DIFFERENT from K_4's |λ|² = 2. Therefore lov's
saddle eigenvalue ≠ (√3+i√5)/2, and CLASS C predictions on lov would have
DIFFERENT numerical values than on srs.

**Implication for ensemble framing:** if the physical substrate is a Boltzmann-
weighted superposition {srs, srs-z, lov, ...}:
  - srs and srs-z contributions to CLASS C predictions agree (CLASS C cover-invariance).
  - lov contribution must be computed with lov's own saddle — not inherited.

For all rows above, the canonical numerical value is the srs / srs-z pair's
(per CLASS C / cover-invariance). The lov sector contribution is currently
suppressed by Boltzmann weight (per `proofs/foundations/srs_vs_srs_z_dl_audit.py`
infrastructure; lov-specific DL audit pending). Per
an internal working note, the
broader candidate ensemble's full impact is research-level open.

**Caveats for χ̃-using closures:**
- B1 (V_ub via Z_6 = χ̃ × C_3): NEGATIVE — both χ̃ sectors carry identical
  (4, 2, 2) C_3 multiplicities (`proofs/foundations/srs_z_chi_C3_VRam_isotypic.py`).
  Tier 2 of an internal working note
  vacated.
- D5 (y_τ via χ̃-graded SUSY-pair Feshbach): NEGATIVE — both χ̃ sectors carry
  identical Feshbach residue (`proofs/foundations/y_tau_chi_graded_feshbach.py`).
  Tier 1 A1 ruled out.
- Tier 1 A2/A3, Tier 3 C1, Tier 4 D1 remain research-level — depend on
  P2.3 χ̃-symmetry-breaking operator
.

---

## Cross-references

- `../audits/registers/uniqueness_ledger.md` — the structural-pass companion (25 rows).
- `../audits/registers/structural_residue_register.md` — R-N residues (12 entries; 10 REFUTED, 1 RESTRICTED, 1 ACCOUNTED-FOR).
- `../theorems/theorem_A5b_level_prescription.md` — A5(b) Level 1/2/3 prescription, the parameter-pass's most-cited upstream classification scheme.
- `../operator_sweep/operator_sweep_from_A1.md` — operator catalog.
- `parameter_linter.md` — gate-type definitions.
- `target_parameters.md` — canonical target list across all framework parameters.
