# Operator Sweep Audit — Layer 4

**Date:** 2026-04-26.
**Status:** Per-operation audit. Layer-by-layer execution of the operation-constructor workstream.
**Source catalog:** `operator_sweep_from_A1.md` §Layer 4.
**Predecessors:** `operator_sweep_audit_layer_0_1.md`, `operator_sweep_audit_layer_2.md`, `operator_sweep_audit_layer_3.md`.

## Layer 4 — Probability, information theory, harmonic analysis, statistical mechanics

49 operations grouped into 8 sub-sections. Largest single layer in the catalog.

### 4.A — Probability (4 ops)

| # | Operation | Verdict | Citation |
|---|---|---|---|
| 4.1 | Probability measure P on F_inv(E) | invoked-direct | `../theorems/theorem_multiway_branch_measure.md` (μ on F_inv(E) words). |
| 4.2 | Expectation E_P[f] | invoked-direct | `predictions/v_higgs.py` mean-field expectation; `predictions/m_tau.py` waterline weighting. |
| 4.3 | Joint and marginal distributions | invoked-direct | `../theorems/theorem_lorentz_toggle_correlations.md` joint distributions of toggle events; `../theorems/theorem_A3_complex_hilbert_from_multiway.md` joint state-and-auxiliary structure. |
| 4.4 | Conditional probability P(A \| B) / Bayes update | invoked-direct | `predictions/S_disconfirm.py` Bayesian Beta posterior; `predictions/A_hemispherical.py` conditional inference. |

### 4.B — Information theory (6 ops)

| # | Operation | Verdict | Citation |
|---|---|---|---|
| 4.5 | Shannon entropy | invoked-direct | `predictions/Q_Koide.py`, `predictions/alpha_GUT.py`, `predictions/feshbach_exponent_principle.py`. |
| 4.6 | KL divergence D(P ∥ Q) | invoked-direct | `proofs/foundations/srs_foundation_closure.py` (sector-overlap error δ_U = exp(−n · D_KL)). |
| 4.7 | Mutual information I(X; Y) | invoked-direct | `../framework/information_theoretic_stability_axioms.md` §A-IT4 data-processing inequality; rate-distortion R(D) = min I(X; X̂). |
| 4.8 | Description length L(M) | invoked-direct | `predictions/N_hub.py`, `predictions/ADOPTED_P1_ramanujan_support.py`, `predictions/S_fresh.py` (MDL across the framework). |
| 4.9 | Source coding (optimal length = entropy) | invoked-direct | `../theorems/theorem_A2_mdl_from_finite_register.md` (A2-T derives MDL waterline from Shannon source coding on finite register). |
| 4.10 | Rate-distortion bound | invoked-direct | `predictions/N_fit.py`, `predictions/N_hub.py`, `predictions/alpha_1_full.py` (rate-distortion arguments for waterline-truncation thresholds). |

### 4.C — Stochastic dynamics (5 ops)

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 4.11 | Discrete-time Markov chain on F_inv(E) | invoked-direct | `predictions/lambda_toggle_rate.py` toggle-rate Markov; `predictions/xi_t_temporal_correlation.py`. |
| 4.12 | Stationary distribution | invoked-direct | `../theorems/theorem_observer_energy_functional.md` stationary measure; `../theorems/theorem_lorentz_causal_sector.md` ergodic average. |
| 4.13 | Continuous-time Markov process (via Layer 3 limit) | unused-applied-negative | See application sketch §4.13 below. |
| 4.14 | Correlation function C_n(s) | invoked-direct | `../theorems/theorem_lorentz_toggle_correlations.md`, `predictions/xi_t_temporal_correlation.py`. |
| 4.15 | Decay rate / correlation length | invoked-direct | ξ_t = 1/log 6 in `../theorems/theorem_lorentz_causal_sector.md` (Stage 3); `predictions/xi_t_temporal_correlation.py`. |

### 4.D — Harmonic analysis under group symmetries (5 ops)

| # | Operation | Verdict | Citation |
|---|---|---|---|
| 4.16 | Isotypic decomposition | invoked-direct | C₃ isotypic decomposition of V_tree, V_Ram in `predictions/Q_Koide.py`, `predictions/epsilon_Koide.py`, `predictions/ADOPTED_P1_ramanujan_support.py`. |
| 4.17 | Bloch decomposition for translation-invariant operators | invoked-direct | `../theorems/theorem_bloch_lift_mu.md` (foundational theorem); `predictions/eta_5_lorentz_dim5.py`, `predictions/B_P_doubly_degenerate_h.py`. |
| 4.18 | Per-Brillouin-point fibers T(k) | invoked-direct | h_walker eigenvalue at k = P-point in `predictions/h_walker_eigenvalue.py`; `predictions/Q_Koide.py`. |
| 4.19 | Symmetry-protected degeneracies | invoked-direct | P+ projector / 2-3 sector degeneracy in `predictions/B_P_doubly_degenerate_h.py`; `predictions/R3_observer_c3_generation.py`. |
| 4.20 | Alon-Boppana bound | invoked-direct | Ramanujan saturation \|h\|² = k − 1 = 2 in `../theorems/theorem_bloch_lift_mu.md`; `predictions/h_walker_eigenvalue.py`. |

### 4.E — Quotients and coarse-graining (5 ops)

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 4.21 | Group quotient F_inv(E)/N | invoked-direct | srs as K_4-quotient: `predictions/G2_cl2_channels_derivation.md`, `proofs/foundations/B3_chirality_bridge_derivation.md`, `proofs/flavor/srs_bloch_ckm.py`. (Layer 1.10 was previously marked invoked-negatively for the abelianization use; the K_4 quotient is the substantive constructive use, so total verdict for 4.21 is invoked-direct.) |
| 4.22 | Quotient under any equivalence relation | invoked-direct | K_4-quotient and BZ-quotient are equivalence quotients; `proofs/foundations/theorem_B5_3_core.py` (BZ/C_3 quotient). |
| 4.23 | Coarse-graining map (lossy projection) | invoked-direct | `predictions/observer_hilbert_space_derivation.md` (CDP coarse-graining); `predictions/d_spatial_derivation.md`; A2 canonicalization is coarse-graining. |
| 4.24 | Partial trace over tensor sub-factor | invoked-direct | A3-T's purification ↔ partial trace correspondence: `predictions/m_tau_derivation.md`, `predictions/y_tau_derivation.md`, `predictions/R3_observer_c3_generation_derivation.md`. |
| 4.25 | Conditional expectation E[· \| sub-σ-algebra] | unused-applied-derivable | See application sketch §4.25 below. |

### 4.F — Group representation machinery (9 ops)

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 4.30 | Group representation ρ: G → 𝒰(V) | invoked-direct | C₃, S₄, Cl(6;ℂ) reps throughout (`predictions/Q_Koide.py`, `proofs/gauge/k4_pati_salam_cl8.py`). |
| 4.31 | Character χ_ρ(g) = Tr(ρ(g)) | invoked-direct | C₃ characters in `predictions/B_P_doubly_degenerate_h.py`, `predictions/ADOPTED_P1_ramanujan_support.py`, `predictions/Q_Koide.py`. |
| 4.32 | Representation matrix elements ρ_{mn}(g) | invoked-direct | CKM matrix elements ⟨c\|T\|b⟩ in `../theorems/theorem_A5b_level_prescription.md`; D-matrix elements in `proofs/masses/srs_delta_n_derivation.py`. |
| 4.33 | Schur orthogonality of irreducible matrix elements | invoked-direct | `predictions/ADOPTED_P1_ramanujan_support_derivation.md` Step 4 (Serre 1977 Schur lemma). Cross-cited from `predictions/B_P_doubly_degenerate_h_derivation.md`. |
| 4.34 | Peter-Weyl decomposition for compact G | invoked-indirect | F_inv(E) is non-compact, so direct application doesn't apply; but framework uses Peter-Weyl on finite subgroups (C₃, S₄), which is the regular-rep ↔ direct-sum-of-irreps statement subsumed by 4.16 + 4.33. Cited in operator_sweep §4.D. |
| 4.35 | Wigner d-matrices d^j_{mm'}(θ) for SO(3) | invoked-direct | `predictions/screw_wigner_angle.py`, `../theorems/theorem_41_screw_wigner.md`, `../theorems/theorem_g3_higgs_coefficient.md`. |
| 4.36 | Clebsch-Gordan decomposition | invoked-direct | `proofs/masses/srs_nu_mass_ps.py` Pati-Salam CG factor; `proofs/masses/greens_mass_predictions.py` SU(5) Clebsches. |
| 4.37 | Clebsch-Gordan coefficients | invoked-direct | Same — explicit CG coefficient values at SU(5) GUT-scale Yukawa relations. |
| 4.38 | Trace identities under group representations | invoked-direct | Σ T_3², Σ Q² traces in `predictions/sin2_theta_W.py`; GQW identity. |

### 4.G — Lie group / Lie algebra (real) (6 ops)

| # | Operation | Verdict | Citation |
|---|---|---|---|
| 4.39 | Matrix Lie group | invoked-direct | Spin(6), SU(2)_L × SU(2)_R, SU(3)_c throughout (`predictions/sin2_theta_W.py`; `proofs/gauge/k4_pati_salam_cl8.py`). |
| 4.40 | Lie algebra | invoked-direct | Cl(6,0) bivectors as so(6) Lie algebra (`predictions/sin2_theta_W_derivation.md`); commutator brackets in `proofs/foundations/theorem_B6_bridge.py`. |
| 4.41 | Exponential map exp(X) | invoked-direct | `proofs/foundations/theorem_B6_bridge.py` (Spin lift); `proofs/foundations/matching_brauer_weyl_sigma.py`. |
| 4.42 | Structure constants f^c_{ab}: [T_a, T_b] = i f^c_{ab} T_c | invoked-direct | [T_a, U_{C_3}^S] commutators in `proofs/foundations/theorem_B3_B6_reconciliation.py`, `proofs/foundations/K4_matchings_C3_check.py`. |
| 4.43 | Killing form K(X, Y) = Tr(ad_X · ad_Y) | invoked-direct | `predictions/sin2_theta_W.py` Killing-form unification at M_unif; `predictions/sin2_theta_W_derivation.md`. |
| 4.44 | One-parameter subgroup t ↦ exp(tX) | invoked-direct | Continuous-time evolution at Layer 3 (Stone) is the one-parameter subgroup of the unitary group. |

### 4.H — Statistical mechanics (9 ops)

| # | Operation | Verdict | Citation |
|---|---|---|---|
| 4.45 | Partition function Z(β) = Σ_s exp(−β E(s)) | invoked-direct | `../theorems/theorem_observer_energy_functional.md`; `../theorems/theorem_mdl_mean_field_higgs.md`. |
| 4.46 | Free energy F(β) | invoked-direct | Same — mean-field Higgs free-energy minimization. |
| 4.47 | Boltzmann distribution | invoked-direct | `proofs/cosmology/proton_stability_thermodynamic.py`; `proofs/foundations/fluctuation_spectrum.py` (Boltzmann at T = 1, half-amplitude convention). |
| 4.48 | Order parameter and phase diagram | invoked-direct | v_higgs = order parameter for electroweak transition (`predictions/v_higgs.py`). |
| 4.49 | Critical exponents | invoked-direct | `predictions/v_higgs.py`, `predictions/m_H.py`, `../theorems/theorem_mdl_mean_field_higgs.md`. |
| 4.50 | Mean-field approximation | invoked-direct | `predictions/v_higgs.py`, `predictions/G_F.py`, `../theorems/theorem_mdl_mean_field_higgs.md`. |
| 4.51 | BZJ scaling: v ∝ N^{−1/4} for quartic O(n) potential at criticality | invoked-direct | `predictions/N_fit.py`, `predictions/H_0.py`, `predictions/G_F.py` (load-bearing for v_Higgs prediction). |
| 4.52 | Renormalization group flow | invoked-direct | `proofs/masses/mssm_rg_running.py`, `proofs/gauge/_mssm_rge.py` (MSSM 2-loop RGE for gauge couplings); `predictions/alpha_GUT_derivation.md`. |
| 4.53 | Curie-Weiss mean-field model | invoked-direct | `predictions/G_F.py`, `predictions/v_higgs.py` (mean-field for fermionic chains feeding Higgs sector). |

---

## §4.13 — Continuous-time Markov process (application sketch)

**Operation.** Continuous-time analog of 4.11 (discrete-time Markov chain): a stochastic process X_t indexed by t ∈ ℝ_{≥0} with the Markov property and a generator (rate matrix) determining transition probabilities P_t = exp(t L) where L is the rate matrix.

**Application to substrate.** F_inv(E) substrate has two well-defined dynamics:
- **Discrete-time Markov chain** (4.11) at the toggle-rate level: at each toggle event (rate λ_toggle), an edge is created or annihilated according to p_create = 1/2, p_destroy = 1/3.
- **Continuous-time UNITARY evolution** at the continuum limit (Layer 3.13): U(t) = exp(−iHt) on L²(F_inv(E)).

The continuous-time *classical* Markov process — where X_t is a stochastic process on F_inv(E) with rate matrix derived from edge dynamics — is NOT what the continuum limit produces. The continuum limit produces a unitary group, not a classical Markov process. The Wick-rotated version (imaginary time τ = it) of unitary dynamics IS a classical Markov-like process (Euclidean partition function ↔ heat-kernel evolution), but the framework operates in real time (Lorentzian signature) where the dynamics is unitary, not stochastic.

**Output.** First-pass: 4.13 doesn't fit the framework's substrate dynamics structurally. The continuum limit is unitary (real time) or heat-kernel-like (imaginary time / Wick-rotated), not a classical CTMC.

**Compressibility check.** A continuous-time *classical* Markov process on F_inv(E) would have rate matrix L with Σ_g L_{g,h} = 0 (probability conservation) and L_{g,h} ≥ 0 for g ≠ h. The framework's discrete-time stochastic dynamics (4.11) at finite λ_toggle gives a per-step transition matrix P; in the continuum limit P^{1/Δt} → exp(t L) for some L. But the framework uses the discrete-time chain directly (toggle events) and the continuum-unitary evolution separately — never the bridge classical CTMC.

**SM observable check.** No new SM-matching invariant emerges from invoking continuous-time *classical* Markov machinery. The framework's existing waterline survival probabilities (e.g., (2/3)^L per girth-cycle winding) are computed via discrete-time chain combinatorics, not via a continuous-time L-matrix exponential.

**Verdict.** unused-applied-negative. The operation is permitted but the framework's continuum dynamics is unitary, not stochastic; the discrete-time stochastic dynamics is at the toggle level (4.11). 4.13 is a "ghost" entry — a permitted operation whose ecological niche is occupied by a different (unitary) operation. Soft negative; not a structural obstruction.

---

## §4.25 — Conditional expectation (application sketch)

**Operation.** E[X \| 𝒢] is the L²-orthogonal projection of random variable X onto L²(Ω, 𝒢, P) where 𝒢 ⊂ ℱ is a sub-σ-algebra. Generalizes conditional probability (4.4) to random variables. Tower property E[E[X \| 𝒢] \| ℋ] = E[X \| ℋ] for ℋ ⊂ 𝒢 organizes nested filtrations.

**Application to substrate.** The framework's MDL canonicalization (A2-T) is a *lossy projection* (Layer 4.23 coarse-graining) from full substrate state to observer-readable state. The candidate question: is A2-T's MDL canonicalization equal to a conditional expectation?

A conditional expectation E[X \| 𝒢] is the unique σ(𝒢)-measurable random variable Y minimizing E[(X − Y)²] (L² best approximation). MDL canonicalization minimizes description length, not L²-distance. These are *different* optimization criteria — L² minimization gives Gaussian-like residuals; MDL minimization gives Shannon-optimal coding residuals.

But: the *operation* of "best lossy projection onto a sub-σ-algebra" is the right shape. The framework already uses partial trace (4.24) which is the QUANTUM analog (best operator-norm projection, given sub-tensor-factor). Conditional expectation is the CLASSICAL analog (best L² projection, given sub-σ-algebra).

**Output.** If A2-T's MDL canonicalization is reformulated as a conditional expectation in the appropriate σ-algebra (the σ-algebra of "observer-readable events"), this would:
1. Provide the L²-orthogonality structure for free (not currently exploited).
2. Enable the tower property — compositions of MDL canonicalizations factor through nested σ-algebras.
3. Cross-validate A2-T from a different mathematical apparatus.

**Compressibility check.** Conditional expectation is structurally compressible — it has the universal property of orthogonal projection onto a closed subspace, which is high information density.

**SM observable check.** No direct SM-matching invariant from this op alone. But cross-validation of A2-T (the framework's most foundational theorem after A1) would be a *category-2 yield* per the rubric (cross-validation).

**Verdict.** unused-applied-derivable. Worth a focused investigation: write A2-T's canonicalization as a conditional expectation and check whether the L²-orthogonality structure produces new derivable consequences. Estimated 1-2 sessions for a self-contained reformulation; could spawn `docs/forward_construction_a2t_conditional_expectation.md` as a candidate cross-validation theorem. **Highest-priority unused op found in the sweep so far.**

---

## Aggregate (Layer 4)

| Status | 4.A | 4.B | 4.C | 4.D | 4.E | 4.F | 4.G | 4.H | Total |
|---|---|---|---|---|---|---|---|---|---|
| invoked-direct | 4 | 6 | 4 | 5 | 4 | 8 | 6 | 9 | 46 |
| invoked-indirect | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| unused-applied-negative | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| unused-applied-derivable | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| **Layer total** | **4** | **6** | **5** | **5** | **5** | **9** | **6** | **9** | **49** |

**Coverage.** 49/49 catalog entries audited.

**Forward-construction docs spawned this pass.** None yet, but 4.25 conditional expectation is queued as the highest-priority unused op found in the sweep so far. Expected `docs/forward_construction_a2t_conditional_expectation.md` after a focused investigation session.

---

## Honest verdict on Layer 4 sweep

**Yield categories from the rubric:**
1. New low-MDL invariant matching SM observable: **none**.
2. Cross-validation of existing prediction via distinct route: **one candidate** (4.25 conditional expectation as alternative formulation of A2-T's MDL canonicalization). Not yet produced; queued for focused investigation.
3. Pinned obstruction: **none**. (4.13 is a soft negative — different ecological niche occupied by unitary dynamics.)

Layer 4 is the most-invoked layer so far: 47/49 ops cited to existing predictions/theorems, mostly with multiple file references. The framework's mathematical content is dense at Layer 4 — probability + information theory (the MDL apparatus), harmonic analysis (Bloch + isotypic), rep theory (Schur, characters, CG), Lie algebra (gauge structure), stat mech (Higgs sector). Almost all of it is load-bearing.

The two unused entries (4.13 ghost, 4.25 conditional expectation) are revealing:
- **4.13** highlights that the framework's continuum dynamics is *unitary*, not classical-stochastic. The continuous-time Markov process op is a permitted-but-displaced apparatus.
- **4.25** is the first unused op with positive forward-construction potential. Reformulating A2-T as a conditional expectation could yield cross-validation. This is the kind of finding the search-instrument framing was designed to surface.

Layer 4 also revealed that **K_4 quotient** (the srs construction) operates as both group quotient (4.21) and equivalence quotient (4.22) simultaneously, providing a clean example of how a single physical construction draws on multiple catalog entries.

---

## Cumulative through Layer 4

| Layer | Ops | invoked-direct | invoked-indirect | invoked-negatively | unused-applied-negative | unused-applied-derivable | unused-deferred |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| 1 | 13 | 10 | 0 | 1 | 0 | 0 | 2 |
| 2 | 33 | 30 | 1 | 0 | 1 | 0 | 1 |
| 3 | 13 | 12 | 0 | 0 | 1 | 0 | 0 |
| 4 | 49 | 46 | 1 | 0 | 1 | 1 | 0 |
| **Cumulative** | **112** | **102** | **2** | **1** | **3** | **1** | **3** |

**Headline:** 112 ops audited; 105 invoked (any flavor); 7 unused (3 applied-negative, 1 applied-derivable, 3 deferred); 0 SM-matching positive yields; 1 cross-validation candidate queued (§4.25).

---

## Cross-references

- `operator_sweep_from_A1.md` §Layer 4 — source catalog.
- `../theorems/theorem_A2_mdl_from_finite_register.md` — central derivation citation hub for 4.B (information theory).
- `../theorems/theorem_bloch_lift_mu.md` — central citation for 4.D (harmonic analysis).
- `../theorems/theorem_observer_energy_functional.md` — central citation for 4.H (stat mech start).
- `predictions/sin2_theta_W.py`, `predictions/sin2_theta_W_derivation.md` — citation hub for 4.G (Lie / Killing form).
- Predecessor audits: `operator_sweep_audit_layer_0_1.md`, `_2.md`, `_3.md`.

---

## Ontology backfill (added 2026-04-26)

This audit was written before the three-lens format was adopted at Layer 5. The ontological-grounding lens is appended below.

### What Layer 4 grounds in QFT/physics ontology

Layer 4 is the **biggest single layer** (49 ops) and the **richest ontological territory of the substrate-side machinery**. Spans 8 sub-sections covering probability, information theory, stochastic dynamics, harmonic analysis, quotients/coarse-graining, group representations, Lie algebra, and statistical mechanics.

| Substrate object | Standard QFT/physics analog | Grounding |
|---|---|---|
| **Probability measure on F_inv(E)** (4.1) | Classical probability | Branch measure μ from `../theorems/theorem_multiway_branch_measure.md`. |
| **Shannon entropy H(P)** (4.5) | Information-theoretic entropy | Foundational for MDL apparatus; appears in `predictions/Q_Koide.py`, `predictions/alpha_GUT.py`. |
| **KL divergence D(P‖Q)** (4.6) | Relative entropy | Sector-overlap error δ_U = exp(−n·D_KL) in `proofs/foundations/srs_foundation_closure.py`. |
| **Mutual information I(X;Y)** (4.7) | QFT correlation measure | A-IT4 data-processing inequality; rate-distortion R(D) = min I(X;X̂). |
| **Description length L(M) / MDL** (4.8) | (Not standard in QFT) | The framework's *primary* compression metric. A2-T derives MDL canonicalization from finite-register Shannon source coding. **One of the framework's substantive contributions to physics foundations.** |
| **Discrete-time Markov chain** (4.11) | Lattice transfer matrix | Toggle-rate Markov dynamics with p_create=1/2, p_destroy=1/3. |
| **Correlation function C_n(s)** (4.14) | QFT correlator | Stage 3 rapid-decay (`../theorems/theorem_lorentz_causal_sector.md`). |
| **Correlation length / decay rate** (4.15) | QFT correlation length | ξ_t = 1/log 6 ≈ 0.558 ℓ_P sub-Planckian. |
| **Isotypic decomposition** (4.16) | Symmetry-protected sectors of QFT | C₃ isotypic decomposition of V_tree, V_Ram in Koide-family predictions. |
| **Bloch decomposition** (4.17) | Translation-invariant quantization | Foundational; `../theorems/theorem_bloch_lift_mu.md`. Analog of momentum-space decomposition in QFT. |
| **Per-Brillouin fibers T(k)** (4.18) | Momentum-space operator at fixed k | h_walker at k = P-point in `predictions/h_walker_eigenvalue.py`. |
| **Symmetry-protected degeneracies** (4.19) | Crystal-symmetry-protected band structures | P+ Bloch projector, 2-3 sector degeneracy. |
| **Alon-Boppana bound** (4.20) | (Not standard in QFT; specific to graph spectra) | Ramanujan saturation \|h\|² = k−1 = 2; central spectral feature of substrate. |
| **K_4 / equivalence quotient** (4.21, 4.22) | Lattice quotient / unit cell | srs as K_4-quotient; `predictions/G2_cl2_channels_derivation.md`, `proofs/foundations/B3_chirality_bridge_derivation.md`. |
| **Coarse-graining / lossy projection** (4.23) | RG transformation / decimation | A2-T's MDL canonicalization is the substrate's coarse-graining op. |
| **Partial trace** (4.24) | Reduced density operator in open QFT | A3-T realization of MDL canonicalization as partial trace. |
| **Group representation ρ: G → 𝒰(V)** (4.30) | Symmetry representation in QFT | C₃, S₄, Cl(6;ℂ) reps throughout. |
| **Character χ_ρ(g) = Tr(ρ(g))** (4.31) | Character of QFT symmetry rep | C₃ characters {1, ω, ω²} pervasive in Koide work. |
| **Schur orthogonality / lemma** (4.33) | Orthogonality of irreps | Load-bearing in `predictions/ADOPTED_P1_ramanujan_support_derivation.md`. |
| **Wigner d^j matrices** (4.35) | Angular-momentum reps in QM | `predictions/screw_wigner_angle.py`; Wigner-d-derivable predictions. |
| **Clebsch-Gordan decomposition + coefficients** (4.36, 4.37) | CG in particle physics | SU(5) Clebsches in `proofs/masses/greens_mass_predictions.py`; PS CG in `proofs/masses/srs_nu_mass_ps.py`. |
| **Trace identities under group reps** (4.38) | sin²θ_W = ΣT₃² / ΣQ² | GQW identity; load-bearing for sin²θ_W = 3/8 prediction. |
| **Lie algebra + structure constants** (4.40, 4.42) | Gauge-algebra in QFT | Cl(6,0) bivectors as so(6); [T_a, U_C₃] commutators. |
| **Killing form K(X,Y)** (4.43) | Common gauge-coupling normalization at GUT scale | Killing-form unification at M_unif → sin²θ_W = 3/8. |
| **Exponential map exp(X)** (4.41) | Lie-group exponentiation | Spin-lift in `proofs/foundations/theorem_B6_bridge.py`. |
| **Partition function Z(β)** (4.45) | Classical partition function | Stat-mech foundation for v_Higgs, m_H predictions. |
| **Free energy F(β)** (4.46) | Free energy | Mean-field Higgs minimization. |
| **Boltzmann distribution** (4.47) | Boltzmann weight in stat mech | `proofs/cosmology/proton_stability_thermodynamic.py`. |
| **Order parameter / critical exponents** (4.48, 4.49) | Phase transitions | v_higgs as order parameter; m_H from critical structure. |
| **Mean-field approximation / Curie-Weiss** (4.50, 4.53) | Mean-field theory of QFT condensates | Higgs sector. |
| **BZJ scaling v ∝ N^{−1/4}** (4.51) | Brézin-Zinn-Justin critical scaling | Load-bearing for v_Higgs prediction; substrate's N → v bridge. |
| **Renormalization group flow** (4.52) | RG flow in QFT | MSSM 2-loop RGE used (invoked via literature, not derived from substrate). |

### QFT-postulated objects this layer informs

Per `../framework/framework_qft_ontology.md`:
- **Shannon entropy / KL / mutual information** (§6) — Layer 4.5–4.7.
- **MDL apparatus** (§6) — Layer 4.8 + 4.9 + A2-T.
- **Markov dynamics** (§6) — Layer 4.11.
- **BZJ scaling / Higgs vacuum value** (§6) — Layer 4.51 + N → v_Higgs bridge.
- **Mean-field theory / Curie-Weiss** (§6) — Layer 4.50, 4.53.
- **Renormalization-group flow** (§6) — Layer 4.52 invoked but not derived (open gap).
- **Killing-form gauge unification** (§5) — Layer 4.43.
- **Lie algebra + structure constants** (§5) — Layer 4.40, 4.42.
- **Complex characters / charge labels** (§5) — Layer 4.31, 5.31.

### Per-op ontology — unused entries

**§4.13 continuous-time Markov process (unused-applied-negative).** **Substrate:** the framework's continuum dynamics is *unitary* (real-time) or heat-kernel-like (Wick-rotated), not a classical CTMC. **Why displaced:** Layer 3 unitary apparatus occupies the ecological niche. **QFT ground:** classical Markov in QFT (Schwinger-Keldysh) is also a downstream construct, not fundamental.

**§4.25 conditional expectation (unused-applied-derivable — Tier 1 candidate).** **Substrate:** would reformulate A2-T's MDL canonicalization as L²-orthogonal projection onto observer-readable σ-algebra. **QFT ground:** classical analog of partial trace; pairs with A.15 martingales for substrate conservation-law structure. **Highest-priority unused op identified in the sweep at this stage; investigation queued as part of Tier 1 quantum-information cluster.**

**§4.34 Peter-Weyl decomposition (invoked-indirect).** **Substrate:** F_inv(E) is non-compact; Peter-Weyl direct application doesn't apply. But framework uses Peter-Weyl on *finite subgroups* (C₃, S₄) via 4.16 + 4.33, which is the regular-rep ↔ direct-sum-of-irreps statement subsumed. **QFT ground:** Peter-Weyl underlies harmonic analysis on compact groups; framework's finite-subgroup version is what's used.

---

## Status

Layer 4 audit complete with ontology backfill. **First positive-leaning finding of the sweep:** 4.25 conditional expectation as candidate cross-validation of A2-T's MDL canonicalization. Now part of Tier 1 quantum-information cluster.

Next: §F (field-selection derivation, 0 ops — a structural derivation, no operation count); Layer 5 (~38 ops on quantum / complex-Hilbert structures, post-field-selection); §C (continuum-limit closure, 0 ops); Layer 6 (~24 ops on continuum / differential geometry / GR). Plus the Appendix (21 explicitly-unused ops).

Total remaining: ~83 catalog ops + 21 appendix = ~104 entries.
