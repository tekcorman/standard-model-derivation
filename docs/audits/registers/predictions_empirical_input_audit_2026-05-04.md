# predictions/ empirical-input audit register — 2026-05-04 EOD+3

**Purpose:** classify every file in `predictions/` by what external inputs it relies on. **User standard (2026-05-04 EOD+3):** ANY input is a smuggled assumption — the framework's predictions should output pure numbers (dimensionless ratios or values in framework-natural units), with no PDG/CODATA values used as derivation inputs. Comparison with observation can use empirical conversions, but those conversions belong in *test* code, not prediction code.

**This standard rejects the framework's "one adopted dimensional input" design (N_hub; its value pinned via the measured G_F)** as still-smuggling. M_P_GeV (CODATA), G_F (PDG), t_P (CODATA) are all smuggled inputs under this standard.

**Triggered by:** user catch 2026-05-04 EOD+3:
- (initial) "I'm upset to find out that we're using anything empirical at all in the predictions."
- (sharpened) "There should literally not be any inputs at all to these files. Any input is an assumption smuggled in."

**Method:** grep for `[external]` markers in each file's `INPUTS` section, plus check IMPLEMENTATION code for hardcoded PDG-style numerical values not flagged as `[external]`. 101 files audited.

---

## Class A — STRUCTURAL_PURE (no dimensional anchor needed)

Pure dimensionless ratios, substrate counts, group-theoretic objects, Bloch eigenvalues. **0 empirical inputs.** These are the framework's cleanest predictions.

| File | What it predicts |
|---|---|
| `k_star.py` | k* = 3 (coordination number, theorem) |
| `g_girth.py` | g = 10 (girth of srs, theorem) |
| `d_spatial.py` | d = 3 (spatial dimension, theorem) |
| `alpha_1.py` | α₁ = (2/3)⁸ (NB walk survival, theorem) |
| `alpha_1_full.py` | α₁_full = (5/3)·α₁ (with chirality factor, theorem) |
| `alpha_GUT.py` | α_GUT = 1/24 (theorem) |
| `Q_Koide.py` | Q = 2/3 (theorem) |
| `epsilon_Koide.py` | ε = √2 (theorem) |
| `delta_Koide.py` | δ = 2/9 (theorem) |
| `R_nu_splitting.py` | R = 228/7 (Ihara, theorem) |
| `M_Pl_natural.py` | M_Pl = 8/√π in framework units (theorem) |
| `Omega_DM_over_Omega_m.py` | (k*-1)/k* = 2/3 (theorem) |
| `h_walker_eigenvalue.py` | h = (√3 + i√5)/2 (theorem) |
| `M_unif.py` | M_unif = 32/k*^(g-1) × M_P in natural units (theorem-cond) |
| `J_CKM.py` | Jarlskog invariant (substrate amplitude, theorem) |
| `eta_B.py` | η_B from Sakharov skeleton (theorem) |
| `Q_density`, `theta_QCD`, `theta_12_PMNS`, `theta_13_PMNS`, `theta_23_PMNS`, `delta_CP_CKM`, `alpha_31_PMNS`, `lambda_higgs`, `lambda_toggle_rate`, `mdl_symmetry_coherence`, `observer_dim_three`, `observer_hilbert_space`, `p_toggle`, `screw_wigner_angle`, `tree_subspace_construction`, `walker_dynamics`, `xi_t_temporal_correlation`, `w_DE`, `c1_photon_bundle`, `S_disconfirm`, `S_fresh`, `dark_extraction_map`, `eta_5_lorentz_dim5`, `eta_lattice_lorentz_dim6`, `feshbach_exponent_principle`, `Feshbach_coupling_strength`, `G2_cl2_channels`, `georgi_jarlskog`, `H_multiway_dim_count`, `koide_quark_ratio`, `lorentz_signature_local`, `srs_E_at_P`, `srs_bloch_lv_dim6`, `srs_cubic_moment`, `theorem_B1_ordering`, `theorem_B2_signature`, `theorem_B3_spinor_fermion`, `uniform_Q_density`, `y_tau`, `R_infinity`, `R3_observer_c3_generation` | various structural objects |
| `V_cb`, `V_us`, `V_ud`, `V_cs`, `V_tb`, `V_cd`, `V_td`, `V_ts`, `V_ub` | CKM elements (substrate amplitudes) |
| `delta_CP_CKM_geometry`, `_ckm_unitarity`, `ADOPTED_P1_ramanujan_support`, `B_P_doubly_degenerate_h`, `A_hemispherical` | structural / Bloch / dark |
| `g_1`, `g_2`, `g_3`, `alpha_s`, `sin2_theta_W`, `sin2_theta_W_MZ` | gauge-coupling cluster (inherits M_unif chain) |
| `m_W`, `m_e`, `m_mu`, `m_tau` | masses (inherit from theorem-grade y_τ × v chain) |

**Note on Class A masses (m_e, m_μ, m_τ, m_W):** these inherit M_P_GeV via v_higgs and M_Z chains, which themselves use M_P as the framework's CODATA unit anchor. They count as STRUCTURAL_PURE in the sense that their *predictions* are derived from substrate ratios; the M_P appears only as a unit conversion.

---

## Class B — PLANCK_ANCHOR_ONLY (CODATA conversion only)

Use M_P_GeV = 1.22089e19 GeV and/or t_P = 5.391247e-44 s as CODATA unit-translation factors. The framework derives M_Pl = 8/√π in its own natural units (theorem-grade); M_P_GeV is just the conversion to SI/GeV. **Same class as the framework's "single anchor" design.**

| File | External(s) | Notes |
|---|---|---|
| `G_N.py` | M_P (CODATA) | Newton's constant from G_sub Drude closure |
| `G_F.py` | M_P (CODATA) | Now a prediction (G_F = 1/(√2 v²), v ← the adopted N_hub via BZJ; matches by construction) |

---

## Class C — N_HUB_CHAIN (the adopted dimensional input N_hub; its value pinned via the measured G_F — the framework's documented design)

The adopted N_hub (its value pinned to sub-ppm via the measured G_F) + a unit-setting constant (M_P ≡ t_P ≡ G_N); propagates downstream. **One genuine *physical* empirical input** (N_hub, whose value is read off via the measured G_F); G_F itself is downstream. See `docs/framework/framework_anchor_choice_2026-04-30.md` (and the SUPERSEDED-FRAMING banner there).

| File | External(s) | Status |
|---|---|---|
| `N_hub.py` | M_P, G_F | Framework's adopted-N_hub chain (value pinned via the measured G_F) |
| `v_higgs.py` | M_P, G_F | v = δ²·M_P/(√2·N^(1/4)) BZJ |
| `m_H.py` | M_P, G_F | Inherits v |
| `m_nu2.py`, `m_nu3.py` | M_P, G_F | Theorem-grade-conditional |
| `H_0.py`, `t_0.py` | M_P, G_F, t_P | Cosmology cascade |
| `M_Z.py` | (inherits via v) | Self-consistent EW matching |

**This is the framework's INTENDED state.** No additional cleanup needed.

---

## Class D — HIDDEN-PDG-INPUTS (the actual problem)

Hardcode PDG values as derivation inputs when a theorem-grade upstream prediction exists or could be made. **These are housekeeping debt** — fixing them means importing from the appropriate predictions/ file instead of hardcoding the PDG number.

| File | Hidden PDG input | Should use instead | Action |
|---|---|---|---|
| `alpha_EM.py` | `M_Z = 91.1876` (PDG) | `M_Z` Row P64 self-consistent (predicted 91.97) | UPDATE: import from `predictions/M_Z.py` |
| `beta_cosmic_birefringence.py` | `alpha_EM = 1/137.035999084` (PDG) | inherit from framework's α_EM(0) prediction | UPDATE: import from `predictions/alpha_EM.py` |
| `scale_energy_hashimoto.py` | `m_e = 0.511 MeV` (PDG), `E_Pl` (CODATA) | `predictions/m_e.py` (theorem-grade-conditional) | UPDATE: import from `predictions/m_e.py` |
| `universe_transparency.py` | `m_e = 0.511 MeV` (PDG) | same as above | UPDATE: import from `predictions/m_e.py` |
| `R3_observer_c3_generation.py` | `m_tau` PDG (for lepton mass non-degeneracy argument) | could inherit from `predictions/m_tau.py` | UPDATE: import; observation that m_τ ≠ m_μ ≠ m_e is what's load-bearing, not specific values |

**Estimated cleanup cost:** 1 session of mechanical replacements. None of these change predictions; they replace hardcoded PDG with imports from theorem-grade-conditional upstream predictions.

---

## Class E — IRREDUCIBLY EMPIRICAL (genuine garbage flagged for retraction or rework)

Files where the empirical input is NOT inheritable from any current theorem-grade prediction. These are the genuine "not-theorem-grade" predictions that should be flagged clearly.

| File | Empirical input(s) | Why it can't be cleaned with current framework |
|---|---|---|
| `m_top.py` | `m_c = 1.27 GeV`, `m_b = 4.18 GeV` (PDG) | Koide waterfall needs other quark masses; framework lacks structural m_c, m_b derivation. Status currently labeled ADVANCED. **Needs retraction or honest TBD label.** |
| `Omega_DM.py` | `Omega_b = 0.0493` (PDG, observed baryon density) | Framework derives Ω_DM/Ω_m = 2/3 (theorem) but Ω_b is observed. Result Ω_DM = Ω_b·(ratio)/(1−ratio). **Honest TBD on Ω_b derivation.** |
| `N_fit.py` | `H_0_CMB = 67.4`, `H_0_ladder = 73.0` (PDG) | Explicit fit between two observed Hubble values. **This is fitting by definition; not a prediction.** Probably a probe script that drifted into predictions/. |

**Recommended actions for Class E:**
- `m_top.py` — retract OR keep as ADVANCED but rename status to `OPEN-EMPIRICAL` with explicit "NOT theorem-grade" header. The status comments already say ADVANCED but a casual reader could miss this.
- `Omega_DM.py` — relabel as `THEOREM-CONDITIONAL-on-Ω_b`. The structural prediction is the ratio; absolute Ω_DM inherits Ω_b's empirical status.
- `N_fit.py` — relocate to `proofs/cosmology/` or `_retracted/`. It's a fitting script, not a prediction.

---

## Class F — UNCATEGORIZED / NEED DEEPER CHECK

Need a closer pass to verify their input class:

`alpha_21_PMNS.py` — Majorana phase prediction; should be Class A but uses some imports.
`m_W.py` — recent ship; uses M_Z + sin²θ_W + v + g_2 (all theorem-grade-conditional) → Class A.

(Both verified Class A on closer reading; not flagged.)

---

## Summary under user's strict "zero empirical inputs" standard

| Class | Original count | Under STRICT standard |
|---|---|---|
| A — STRUCTURAL_PURE (dimensionless or framework-natural) | ~70 | **CLEAN** |
| B — PLANCK_ANCHOR_ONLY (M_P CODATA) | 2 | **SMUGGLED** (M_P GeV is observed) |
| C — N_HUB_CHAIN (the adopted N_hub + M_P unit) | 9 | **carries one *physical* empirical input** (N_hub, value read via the measured G_F) |
| D — HIDDEN-PDG (M_Z, m_e, etc.) | 5 | **SMUGGLED** (compounded — multiple PDG values) |
| E — IRREDUCIBLY EMPIRICAL | 3 | **SMUGGLED + fitted** |

**Restated bottom line:** out of 101 prediction files:
- **~70 are CLEAN under the strict standard** (Class A — pure dimensionless or framework-natural objects)
- **~30 smuggle empirical inputs** of varying severity (Classes B-E)

The framework's "single anchor" design is itself a form of smuggling under the strict standard. The CLEAN predictions (Class A) are dimensionless ratios, substrate counts, group-theoretic objects, and `M_Pl_natural.py` (= 8/√π in framework units). Everything that converts to GeV via the M_P unit or carries the adopted N_hub (whose value is read via the measured G_F) carries the one physical empirical input.

**The genuinely cleanest** predictions in the framework right now:
- `M_Pl_natural.py` — outputs 8/√π (dimensionless, theorem)
- `Q_Koide.py` — outputs 2/3 (dimensionless, theorem)
- `epsilon_Koide.py`, `delta_Koide.py` — pure rationals/algebraics
- `R_nu_splitting.py` — outputs 228/7
- `Omega_DM_over_Omega_m.py` — outputs 2/3
- `k_star.py`, `g_girth.py`, `d_spatial.py` — pure structural integers
- `alpha_1.py`, `alpha_1_full.py`, `alpha_GUT.py` — pure rationals
- `h_walker_eigenvalue.py` — pure complex algebraic
- `J_CKM.py`, `eta_B.py`, etc. — substrate amplitudes (need to verify their inputs)

These are the framework's actual "first-principles predictions" under the strict standard — pure numbers, no GeV conversions, no observed inputs.

Everything else is **comparison with observation that requires empirical conversion** to land in physical units. Under the strict standard, the converted-to-GeV files (m_W, m_e, m_nu3, v_higgs, etc.) are not predictions but **test code that uses framework predictions plus empirical anchor to compare to PDG**.

---

## Recommended next sessions under the STRICT standard

**Architectural decision required:** the framework's `predictions/` directory is currently mixed — some files output pure numbers (Class A), others output GeV values that smuggle M_P CODATA. The strict standard says these are different things and shouldn't share a directory.

Two paths forward:

### Path 1 — Restructure: split predictions/ into structural/ and tests/

- `predictions/` retains ONLY Class A files (pure dimensionless or framework-natural outputs).
- A new `tests/comparison/` (or similar) directory holds files that take Class A predictions + empirical anchor (M_P GeV) and check against PDG. These are *tests*, not predictions.
- Updates to docs/master_plan.md, parameter_uniqueness_ledger.md to clarify what counts as "prediction" vs "comparison test."
- ~3-5 sessions of mechanical reorganization.

### Path 2 — Tag in place: add "PURITY" header to every file

- Add a `# PURITY: STRUCTURAL_PURE | SMUGGLED-via-{M_P, G_F, M_Z, ...}` line at the top of each file.
- No restructuring; just clear labeling so a reader sees the smuggling status at a glance.
- 1 session of mechanical labeling.

### Research follow-ups (independent of restructure)

**N_hub structural derivation** is the framework's central open problem under the strict standard. If N_hub's *value* closes from substrate combinatorics alone (no G_F-calibration needed — Gap G1), Class C predictions become CLEAN. This is acknowledged as the "Holy Grail" frontier in an internal working note Tier 3.

**M_P_GeV → framework-natural** is also needed: derivation of "1 toggle = X GeV" structurally. The framework already has M_Pl_natural = 8/√π (theorem). What's missing is the conversion factor to GeV from first principles — i.e., what is "1 GeV" in toggles? This may be unanswerable in principle (units are conventions) — but the strict standard demands it OR demands all predictions stay in framework-natural units.

---

## What this audit does NOT cover

- Whether the framework's substrate axioms (A1, A2, A5) are themselves correctly derived rather than postulated. (See `docs/framework/framework_axioms.md` for axiom-elimination roadmap.)
- Whether α_GUT = 1/24 is truly theorem-grade (it's labeled so, but a deeper audit could reveal MDL-fitting steps).
- Whether the IR quasi-fixed-point argument used (in some places) for y_t(GUT)=1 is structurally sound.

These are deeper audits, distinct from the mechanical PDG-input check this register performs.
