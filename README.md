# Standard Model from First Principles

[![DOI](https://zenodo.org/badge/1208084761.svg)](https://doi.org/10.5281/zenodo.19520591)

### 📖 Read the interactive explainer → **[tekcorman.github.io/standard-model-derivation](https://tekcorman.github.io/standard-model-derivation/)**

*A visual narrative — the story, the two graphs, the walks, the predictions — with animations and an interactive 3D crystal viewer. This README is the technical summary; the explainer is the guided tour.*

---

> **The same substrate object, read 12 different ways from one resolvent — a single argument `a = (2/3)⁸`, zero fitted constants. Eleven of the twelve match PDG within 1σ_PDG; the twelfth (δ_r) has no direct PDG observable — the M_Z it feeds carries the framework's one large, honestly-open oblique residual.**

The 12 readings are 7 quark-sector (y_t, y_b, V_us, V_cb, V_ub, δ_r, δρ), 4 lepton/PMNS (y_τ, θ_12, θ_13, θ_23), and the A_s cosmological prefactor — all from the **non-backtracking resolvent** G_NB = (I − u·B_NB(srs))⁻¹ with one argument `a = (2/3)⁸` and zero fitted constants. **Eleven match PDG within 1σ_PDG** (V_us, V_cb, V_ub, δρ, y_τ→m_τ, θ_12, θ_13, θ_23, A_s — and, since the resolvent's own forced first-girth-return dark correction Σ = α₁/h shipped 2026-06-25 with zero adoption, y_t→m_t at **−0.95σ** and y_b→m_b at **+0.22σ**). δ_r has no direct PDG observable; the M_Z it feeds carries a +7.76σ residual — a forced substrate-vs-SM oblique difference, traced to its floor by the BZ-integrated vacuum polarization (2026-06-30) and **logged open, not claimed closed** ([`docs/incomplete_equations_todo.md`](docs/incomplete_equations_todo.md)). **Then 79 more parameters fall out of the same substrate.** Cross-validation audit: an internal working note.

G_NB is itself a *read* of the framework's one master object **D = B(srs⊗srs-z) ⊗ ∂_N** (the non-backtracking operator on the joint mirror cover, dressed by the run — [`derivation_topdown/bridge/the_run.py`](derivation_topdown/bridge/the_run.py)): masses are its diagonal, mixings its off-diagonal, the gauge running its ∂_N zero-mode, the species labels its Cl(6) Fock grading.

**The framework in one sentence.** Three meta-commitments — self-containment of the universe, finite observer, active reading of binary distinctions — plus standard published mathematics, single out the **srs** crystal net (the dominant member of a small MDL-waterline survivor set; the data-free uniqueness discriminator is a logged open equation, [`docs/incomplete_equations_todo.md`](docs/incomplete_equations_todo.md) §6) whose spectral content is the Standard Model. One empirical labeling rule (A5-mass: which substrate eigenvalues are which observed masses) attaches contact with experiment, and one measured scale (G_F) calibrates the single dimensional unit (N_hub). There are no further inputs.

## Status (2026-07-03)

Across **125 tracked targets** (the 123-target 2026-06-22 baseline + the two EW width ratios registered 2026-07-02), counted in **honest σ_PDG** (parameter-linter Clause-8, σ_PDG only — no σ_theory widening; the 2026-06-22 baseline split is in [`docs/parameters/honest_sigma_count_2026-06-22.md`](docs/parameters/honest_sigma_count_2026-06-22.md); counts below carry the 2026-06-25 dark-correction shipment and the 2026-07-02 EW-width-layer registration forward):

- **44 match within 1σ_PDG** — genuine numerical closures (the 2026-06-22 count of 39, plus **m_t (−0.95σ), m_b (+0.22σ), g_2 (−0.18σ)**, closed 2026-06-25 when the forced dark self-energy Σ = α₁/h and the scheme-consistent g_2 target shipped; plus **Γ_Z/M_Z (−0.55σ) and Γ_W/Γ_Z (+0.14σ)**, closed 2026-07-02 when the derived EW radiative width layer registered — a pre-registered blind computation: layer −1.81 loop units vs the pre-registered demand −1.62 ± 0.34, closing a +4.76σ open header BY DERIVATION to the SM's own −0.53σ residual; grade honestly bridge-conditional, Clause 9b): all CKM except the two unitarity-tension entries, the PMNS angles + δ_CP, all six quark masses, m_τ, η_B, β-birefringence, the gauge couplings, the Higgs sector, the Z-width ratios.
- **~19 forced/exact structural** (k\*, |V|, N_gen, θ_QCD = 0, Q_Koide = 2/3, the unmeasured ν phases) — no σ to test.
- **9 OPEN-GAP** (>1σ_PDG, no established closure): α_EM (+1.01σ), M_Z (+7.76σ), m_W (+2.39σ), V_ts/V_tb (riding the ~3.3σ V_cb exclusive/inclusive data self-tension), m_ν2 (+1.87σ), m_ν3 (+2.18σ), Ω_DM/Ω_b (conditional on the adopted z_eff). These are **understood, not random** — M_Z was traced to its honest floor by the BZ-integrated vacuum polarization (2026-06-30: a forced ~4%-relative substrate-vs-SM oblique difference; open, not closed); m_W inherits it (the custodial ρ-test itself passes at +0.76σ). We do **not** claim closures we don't have.
- **2 open ppm-scale misses** (m_e −70.3 ppm, m_μ −60.5 ppm relative): the un-derived subleading correction to the charged-lepton mass read — a logged open equation ([`docs/incomplete_equations_todo.md`](docs/incomplete_equations_todo.md) §1). The 2026-07-02 loop program sharpened it to its tightest form yet: the hard (m_τ-free) core is ONE number, the chiral completion ε = δ_eff − 2/9 of the generation phase (demand −1.7515×10⁻⁷ ± 3.9×10⁻¹⁰ rad); the interacting ensemble that must supply it is now a derived theorem, and a pre-registered kill chain (E2b, E2c) excluded both the bare-channel functional and every state-block projection of that ensemble — the open equation is the read↔ensemble winding weld. The measurements are sub-ppb, so σ_PDG is unreachable either way — but the miss is a miss, and it stays open.
- **8 framework-vs-ΛCDM coasting** (H_0, t_0, Λ, Ω_m/Ω_Λ …) — tested against the framework-side observation set; the Hubble-tension split is a *prediction*, not a miss.
- The remainder ❌ genuinely open (the L6 cosmology cluster n_s, σ_8, r_s, θ_*) or out-of-scope (Δα hadronic, Clause-9).

**The open gaps are channel-structured** — a *derived* organization, not a list of failures. The **saturation** (m_t) and **Perron** (m_b) channels closed when their forced first-girth-return dark corrections shipped (2026-06-25). The **EW rate side** closed 2026-07-02: the derived radiative width layer took Γ_Z/M_Z from +4.76σ to −0.55σ in a pre-registered blind computation. What remains: the **pole-side oblique** (M_Z, and m_W by inheritance) bottoms out at a forced substrate-vs-SM difference of ~4% relative on the oblique itself — the framework's honestly-logged open equation, not a fittable residual (the width layer is rates-only and does not touch it); the **neutrino absolute scale** (m_ν2, m_ν3) is formula-incomplete because the one free scale is **spent on v** (a global unit, not a per-parameter dial); Ω_DM/Ω_b await the z_eff derivation. Full diagnosis in the honest-σ document above.

The `predictions/` directory is the source of truth: each parameter has a `.py` (the prediction) and a `_derivation.md` (the journal-grade write-up); `_validate_dag.py` enforces self-containment (114 files, 0 forbidden imports); every live value is pinned in the value-lock harness (104 values). Run `python3 verify.py` (65 backbone proofs, ~60 seconds); if any fail, the framework is wrong, full stop.

Highlights: the Lorentz arc closed (2026-04-27); the five-stage gauge-coupling chain closed including sector-specific c_color = 1/4 (2026-05-04 → 2026-05-26); the 12-observable §8 over-determination landed (2026-05-16/23); the multi-axial dark-sector waterfilling theorem promoted to theorem-grade-structural (2026-05-24); the M_persistence 12-mass fermion operator shipped (2026-05-26). Since then: the whole framework was consolidated as reads of **one object D = B(srs⊗srs-z) ⊗ ∂_N** (2026-06-23, [`derivation_topdown/bridge/the_run.py`](derivation_topdown/bridge/the_run.py)); the generation phase δ = 2/9 was **derived** as the forced directed phase of the ∂_N run (2026-06-21); the unified dark self-energy Σ = α₁/h closed m_b (+0.22σ) and m_t (−0.95σ) with zero adoption (2026-06-25/28); the M_Z oblique was traced to its honest floor by a BZ-integrated vacuum polarization and stays logged open (2026-06-30); the substrate-selection claim was sharpened to its honest form — **srs is the dominant member of an MDL-waterline survivor set**, with the Sunada strong-isotropy chain retained as provenance and the data-free discriminator a logged open equation (R-9 supersession 2026-06-15; ruling 2026-07-01); and the 2026-07-02 loop program — eleven git-witnessed pre-registered sittings in one day — **closed the EW width residual by derivation** (Γ_Z/M_Z +4.76σ → −0.55σ, blind, registered) and drove the walk↔Fock identification layer to theorem grade (the step lift forced outright with zero covariant freedom; the seam quarantined; the interacting ensemble's form and its chiral channel proven), with every negative outcome pre-registered, banked as-run, and converted to named structure (the −70 ppm's open equation now sits at the read↔ensemble winding weld). These are partial readings of one substrate object — consolidated (not newly derived) in [`docs/theorems/theorem_walker_matter_unification_2026-05-27.md`](docs/theorems/theorem_walker_matter_unification_2026-05-27.md). Per-closure detail in [`docs/honest_assessment.md`](docs/honest_assessment.md) and the open ledger in [`docs/incomplete_equations_todo.md`](docs/incomplete_equations_todo.md).

## Foundations

The framework rests on three irreducible commitments plus one empirical labeling rule plus standard published mathematics — nothing else. Each commitment is named explicitly rather than absorbed silently into a single "axiom".

- **(A) Self-containment** (metaphysical). The universe is closed to itself; nothing comes from outside, because nothing is outside. This is a refusal to import external structure — no boundary conditions, no anthropic priors, no multiverse selection. Cannot be derived; it stipulates that nothing more fundamental is supplied.
- **(B) Finite observer** (scoping). The framework describes observers with finite memory. A scoping definition — about the *subject* of the predictions — not a physical postulate.
- **(I) Active reading** (interpretive). A binary distinction is read as an *operator* T_e moving between two values, not a static label. Adopted explicitly because alternative readings (passive, asymmetric) yield strictly weaker frameworks.
- **A5-mass** (empirical labeling). Identifies which Bloch-Hashimoto eigenvalues correspond to which Standard Model masses. The framework's only empirical anchor; downstream of (A)+(B)+(I).

Under (A)+(B)+(I) + Shannon-Jaynes-Serre mathematics, the observer's primitive update is **uniquely forced** to be a binary self-inverse toggle generating the free involutive monoid F_inv(E) (proof: [`docs/theorems/theorem_toggle_from_self_containment.md`](docs/theorems/theorem_toggle_from_self_containment.md), 2026-05-07; 8-step derivation). This is the content previously postulated as the axiom **A1**; A1 retains its name in the 117 downstream files that cite it, but is now a derived theorem rather than a postulate. A2 (MDL canonicalization), A3 (complex Hilbert from multiway), A4 (Jordan-Wigner local CAR) are likewise derived theorems. Full demotion chain: [`docs/framework/framework_axioms.md`](docs/framework/framework_axioms.md).

One unit identification: G (Newton's constant) sets the Planck scale (one toggle = one Planck time).

> *Toggle activity, the recurrence patterns within it, and the fundamental-operator catalog of physics are three names for the same thing.*
>
> — [`docs/framework/narrative_spine.md`](docs/framework/narrative_spine.md) §1.4

## Where this sits

| Programme | Predicts SM parameter values? | From a single object? |
|---|---|---|
| String landscape | Conjecturally, via vacuum selection across ~10⁵⁰⁰ vacua | No |
| Connes noncommutative geometry | Predicts SM *gauge structure* + a specific m_H value from spectral triples; full numerical mass spectrum not pinned | One spectral triple per choice of algebra |
| Wolfram–Gorard | Multiway substrate hypothesized; SM derivation as open program | One substrate, but specific rewriting rule unfixed |
| **This framework** | **44 matched within 1σ_PDG + ~19 forced-exact; 12 cross-validated via a single resolvent (9 open gaps, channel-structured & understood)** | **One graph (srs, the dominant MDL-waterline survivor); the non-backtracking resolvent G_NB on it — a read of D = B(srs⊗srs-z) ⊗ ∂_N** |

*None of these are strawmen — each programme has substantive open work. The table compares specifically on whether numerical Standard Model parameter values are pinned (not adjusted, not selected from a landscape, not left as free parameters in the action).*

## What follows

From these commitments, the MDL-dominant graph is the **srs crystal net** (space group I4₁32, coordination number k\* = 3, girth g = 10). Two algebraic structures on this graph encode all physics:

| Pillar | Object | What it encodes |
|--------|--------|-----------------|
| **Flavor** | h = (√3+i√5)/2 | Mixing angles, CP phases, mass hierarchies |
| **Gauge** | Cl(6) = Cl(4)⊗Cl(2) | Gauge group, couplings, fermion content |

h is the Hashimoto (non-backtracking) walker eigenvalue at the P point of the BCC Brillouin zone. Both pillars derive from k\* = 3.

## Selected results

Representative quantitative matches. Predicted/observed/σ values are pulled from the auto-generated `predicted_parameters.md` (run `python3 run_predictions.py` to regenerate); grades are the framework's vocabulary as set by [`docs/parameters/parameter_linter.md`](docs/parameters/parameter_linter.md) Clauses 7–9.

| Quantity | Predicted | Observed | σ vs PDG / grade |
|----------|-----------|----------|---|
| **V_us** | **9/40 = 0.22500** | 0.22501 ± 0.00068 | **−0.01σ** ✅ UNIQUE-THEOREM-GRADE-CONDITIONAL (Level-2 srs counting density) |
| **V_cb** | **256/6305 = 0.04060** | 0.0406 ± 0.0009 (PDG-2024 exclusive, Belle) | **+0.00σ** ✅ UNIQUE-THEOREM-GRADE-CONDITIONAL (Level-3 Hashimoto BFS at girth L=8); ~3.3σ excl/incl tension in PDG, framework reads exclusive |
| **V_ub** | **Σ_{m≥2} (2/3)^(6m+2)/(1−(2/3)^(6m+2)) = 3.77×10⁻³** | 3.82×10⁻³ ± 0.20×10⁻³ | −0.26σ ✅ UNIQUE-THEOREM-GRADE-CONDITIONAL (M1 twisted-walker amplitude) |
| δ_CP^CKM | arccos(1/3) = 70.53° | 68.5° ± 3.0° | +0.68σ ✅ UNIQUE-THEOREM-GRADE-CONDITIONAL (K_4 tetrahedral dihedral; same V_{−1}–T_{B-L} identity that fixes δ_CP^PMNS) |
| **δ_CP^PMNS** | **π = 180°** (V_{−1}–T_{B-L} = arccos(−1)) | 177°⁺¹⁹₋₂₀ (NuFIT 6.0 IC19) | **+0.16σ** ✅ THEOREM-GRADE-STRUCTURAL-CONDITIONAL (revived 2026-05-05 via the parameter-free identity; supersedes the Hashimoto-phase 249.85° route, which was falsified at +3.83σ in 2026-05-02 and is preserved in `predictions/retracted/`) |
| θ_23^PMNS | 48.72° | 49.2° ± 1.3° | −0.37σ ✅ |
| **R_ν = Δm²₃₁/Δm²₂₁** | **228/7 = 32.571** (Ihara closed form on K_4) | 32.576 | <0.01σ ✅ UNIQUE-THEOREM-GRADE (zero dark correction; topological invariant) |
| θ_QCD | 0 exactly | < 10⁻¹⁰ | ✅ UNIQUE-THEOREM-GRADE (strong-CP solved by srs Z₃ holonomy flatness) |
| Q_Koide / ε_Koide / δ_Koide | 2/3, √2, 2/9 | matches | ✅ STRICT-SOLID (k\* = 3) exact algebraic identities; δ = 2/9 derived as the forced directed phase of the ∂_N run (2026-06-21); the −70 ppm subleading per-rep correction to the charged-lepton masses is a logged OPEN equation ([`docs/incomplete_equations_todo.md`](docs/incomplete_equations_todo.md) §1) |
| Higgs VEV v | 246.22 GeV | 246.22 ± 0.12 GeV | −0.00σ — UNIQUE-THEOREM-GRADE **form** (G1b R2 path 2026-04-28); the **value** is the framework's one G_F-calibration round-trip (N_hub pinned from measured G_F), so the 0σ is the calibration closing, not a prediction |
| Higgs mass m_H | 125.20 GeV | 125.20 ± 0.11 GeV | −0.05σ ✅ THEOREM-GRADE-STRUCTURAL (Family-D propagated 2026-05-15; conditional on c_H Route H/C closure) |
| Higgs quartic λ | 0.12927 = 2·α₁_full | 0.1294 | −0.05σ ✅ THEOREM-GRADE-STRUCTURAL (W1 2026-05-18) |
| m_τ | 1.7768 GeV = v·y_τ·Family-D | 1.77686 ± 0.00012 | −0.19σ ✅ THEOREM-GRADE-STRUCTURAL (W1 2026-05-18 reinstatement; prior "UNIQUE-THEOREM-GRADE-NUMERICAL Family-D" was a Clause-6c smuggle) |
| m_t | 172.41 GeV (Type-II saturation y_t(GUT) = 1 + forced dark ×(1−α₁/h_P²)) | 172.69 ± 0.30 GeV | **−0.95σ** ✅ THEOREM-GRADE-STRUCTURAL-CONDITIONAL via M_persistence + the unified dark self-energy (Row P38; the bare pre-dark value was +4.71σ — closed 2026-06-25, `predictions/heavy_quark_anchor_dark.py`) |
| m_b / m_c / m_s / m_d / m_u | M_persistence 12×12 fermion mass operator (+ forced dark ×(1−α₁/h_P) on m_b) | PDG 2024 | all five within 1σ_PDG ✅ (m_b = 4.187 GeV at +0.22σ; bare pre-dark +2.99σ closed 2026-06-25); THEOREM-GRADE-STRUCTURAL-CONDITIONAL via Row P39 |
| **Γ_Z/M_Z** | **0.027350** (α-form tree × derived EW width layer −0.4864%) | 0.0273634 ± 0.0000252 | **−0.55σ** ✅ SM-REPRODUCTION/bridge-conditional (Clause 9b) — closed BY DERIVATION 2026-07-02 from a +4.76σ open header, in a pre-registered blind computation (layer −1.81 loop units vs pre-registered demand −1.62 ± 0.34); equals the SM's own −0.53σ residual; the M_Z pole oblique is untouched |
| H₀ (substrate / CMB-side) | 68.18 km/s/Mpc | 67.4 ± 0.5 (Planck CMB) | +1.6σ ✅ UNIQUE-THEOREM-GRADE (coasting H_0·t_0 = 1) |
| H₀ (observer / SH0ES-side) | 72.72 km/s/Mpc = (16/15)·H_0^substrate | 73.04 ± 1.04 (SH0ES) | −0.30σ ✅ — the (16/15) rate-gap *is* the Hubble tension as a structural prediction |
| t₀ | 14.34 Gyr (substrate) | 14.46 ± 0.80 (Methuselah HD 140283) | −0.15σ ✅ UNIQUE-THEOREM-GRADE (coasting) |
| Ω_DM / Ω_m | 0.8488 = 1 − P(k≤k\* \| Poisson(2k\*)) | 0.846 ± 0.016 | +0.17σ ✅ UNIQUE-THEOREM-GRADE |
| Λ_substrate | 1/N² = 1.42×10⁻¹²² (Planck units) | factor-of-2 vs ΛCDM-fit | ✅ UNIQUE-THEOREM-GRADE clean substrate (graduated 2026-05-16); the observable-side Λ_LCDM = 3·Ω_Λ_LCDM·Λ_substrate sibling at +0.77σ_obs |
| w_DE | −1 exactly | −1.03 ± 0.03 | +1.0σ ✅ UNIQUE-THEOREM-GRADE |
| A (CMB hemispherical) | 1/15 = 0.0667 | 0.07 ± 0.02 | −0.17σ ✅ UNIQUE-THEOREM-GRADE (Theorem 2 of parity-theorems) |
| **η_B** | **(√3/10)·(2/3)⁴⁸ = 6.11×10⁻¹⁰** | 6.12×10⁻¹⁰ ± 0.04×10⁻¹⁰ | −0.20σ ✅ UNIQUE-THEOREM-GRADE-CONDITIONAL (Row 4 + Sakharov–Hashimoto chain, 2026-04-30) |
| β cosmic birefringence | 0.354° = sin(arg h)·α_EM(M_Z) | 0.342° ± 0.094° | +0.13σ ✅ THEOREM-GRADE-STRUCTURAL (downgraded from UNIQUE 2026-05-16; framework α_EM, with Δα Clause-9-blocked as named gap) |

Full per-parameter scorecard with grade flags, file pointers, and dependency chain: [`docs/parameters/target_parameters.md`](docs/parameters/target_parameters.md).

## Falsification criteria — any one would refute the framework

The framework makes specific numerical predictions, several at the precision frontier. Any of these going against measurement is fatal.

> **Externally pre-registered.** The forward-looking predictions below (plus Σm_ν = 59.4 meV with
> normal ordering, V_cb on the exclusive side of the current data tension, and the framework's two
> declared open misses) are frozen with kill windows in an externally timestamped, immutable deposit:
> **[DOI 10.5281/zenodo.21124065](https://doi.org/10.5281/zenodo.21124065)** (Freeze v1, 2026-07-02;
> values-match-audited against this repository's DAG). Predictions are scored against the freeze
> that pre-dates the measurement — a post-measurement revision cannot rescue a killed row.

> **Quantified against the numerology objection.** The [MDL ledger](docs/audits/mdl_ledger.md)
> (methods frozen *before* counting; engine: `scripts/mdl_ledger.py`, re-runnable) prices every
> discrete choice, adoption, calibration, and documented dead-end against the measured data
> explained under deliberately conservative conventions: **specification + search = 136.4 bits vs
> 304.3 bits of SM-parameter data explained — a +168-bit margin (2.23×)** that survives all
> stress-tests stacked (+23.6 bits worst-case). The open misses (M_Z, m_e/m_μ) *pay bits* in this
> accounting rather than being explained away; the hostile prose-length comparison, which the
> framework loses, is reported in the same document.

| Prediction | What kills it | Experiment | Timeline |
|---|---|---|---|
| **δ_CP^PMNS = 180°** | Clearly maximal-CP-violating ~270° (outside ~180° ± 30°) | DUNE, Hyper-K | 2028+ |
| θ_23 = 48.72° (non-maximal) | Exactly maximal: 45.00 ± 0.3° | DUNE | 2028 |
| **m_ν₁ = 0 exactly** | 0νββ never observed AND m_ν₁ > 0 established | KATRIN, Project 8, nEXO | 2027+ |
| No WIMP dark matter (DM = gauge-decoupled uncompressed multiway) | WIMP found at LZ / XENONnT | LZ, XENONnT | ongoing |
| MSSM-valued β {33/5, 1, −3} derived top-down from 2HDM content by the run's 4D time-completion (the Δb = +4 time-shadow, `the_run.py` read_gauge_running; forcedness grade = ζ_{D₄}(0), research-level) | α_s(M_Z) precisely contradicts b_3 = −3 under α_GUT⁻¹ = 24 inversion | LHC, FCC-hh (gauge-coupling precision) | ongoing |
| \|β cosmic birefringence\| ≤ α_EM ≈ 0.418° (hard cap from c_1 = 0) | β measured > 0.418° | LiteBIRD (~0.05° precision) | ~2032 |
| η_5 = 0 exactly (dim-5 LIV) | dim-5 LIV detected | LHAASO, HESS | ongoing |
| α₂₁ = 162.39° (Majorana phase) | α₂₁ outside ~162° ± 30° | nEXO, LEGEND-1000 | 2030+ |
| α₃₁ = 324.78° (Majorana phase) | α₃₁ outside ~325° ± 30° | future 0νββ | — |
| m_ββ ≈ 2.55 meV (0νββ amplitude from m_ν₂ + α_21 chain) | m_ββ measured outside ~1–5 meV | nEXO, LEGEND-1000 | 2030+ |

**One historical falsification has fired as designed.** The original Hashimoto-phase route for δ_CP^PMNS predicted 249.85° (±30°). It failed at +3.83σ vs NuFIT 6.0 IC19 on 2026-05-02 and was retired. The current value 180° came from an **independent parameter-free identity** — V_{−1}–T_{B-L} = arccos(−1) — three days later (2026-05-05). The retracted derivation is preserved at `predictions/retracted/delta_CP_PMNS.py` as honest record. The same identity simultaneously fixes δ_CP^CKM = arccos(1/3) = 70.53° at +0.68σ on a *different* observable — independent corroboration. This is the falsification-and-revival pattern the framework is built around.

## Audit infrastructure

The claim that no step curve-fits is held by **seven audit instruments**, each watching a distinct way derivations can cheat:

1. **Parameter linter** ([`docs/parameters/parameter_linter.md`](docs/parameters/parameter_linter.md)) — Type-1/2/3/4/5/6 gate per derivation step. Clauses 7 / 8 / 9 are blocking gates (multi-axis multi-mechanism uniqueness defense; σ_PDG-strict numerical match; K-rationality bright-line audit), added 2026-04-30 / 2026-05-01 / 2026-05-15.
2. **Structural uniqueness ledger** ([`docs/audits/registers/uniqueness_ledger.md`](docs/audits/registers/uniqueness_ledger.md)) — 25 rows. For every structural choice (k\* = 3, srs lattice, Cl(6,0) algebra, q_NB = 2/3, …), audits why this one and not an alternative.
3. **Parameter uniqueness ledger** ([`docs/parameters/parameter_uniqueness_ledger.md`](docs/parameters/parameter_uniqueness_ledger.md)) — 68 P-rows. Same audit at the formula layer (V_us = 9/40, α₁_bare = (2/3)⁸, V_cb = 256/6305, η_B = (√3/10)·(2/3)⁴⁸, …).
4. **Structural residue register** ([`docs/audits/registers/structural_residue_register.md`](docs/audits/registers/structural_residue_register.md)) — soft-gated alternatives with status (OPEN / TRACED / ACCOUNTED-FOR / REFUTED).
5. **Wave engine** ([`proofs/wave_engine/simulator.py`](proofs/wave_engine/simulator.py)) — 219 catalog operations; all firing post-2026-04-27 LORENTZ_SIG / CCLOSE → NC_GEOM joint closure; open-frontier tags = ∅.
6. **Citation validator** ([`scripts/validate_citations.py`](scripts/validate_citations.py)) — operational pre-commit-hook candidate.
7. **Value-lock harness** ([`scripts/value_lock.py`](scripts/value_lock.py), added 2026-07-01) — every live predicted value is pinned in `predictions/_value_locks.json` and re-checked in CI; a value can only move via a deliberate, reviewable re-freeze. No silent value drift.

Full descriptions: [`docs/orientation.md`](docs/orientation.md) §4.

## Quick start

```bash
python3 verify.py                                      # Run 65 backbone proofs (~60 s)
python3 run_predictions.py                             # (Re)generate predicted_parameters.md at repo root
python3 proofs/flavor/vcb_hashimoto_bfs.py             # V_cb = 256/6305 from A2 geometric series
python3 proofs/flavor/vus_l2_density.py                # V_us = 9/40 from Level 2 counting density
python3 proofs/foundations/srs_p_point_algebra.py      # H²=k*I, Ramanujan saturation
python3 proofs/foundations/srs_generation_c3.py        # Generation definition at P point
```

> **Note on `predicted_parameters.md`:** the headline PDG-comparison table at the
> repo root is auto-generated by `run_predictions.py` and is **gitignored** — a fresh
> clone will not contain it. Run `python3 run_predictions.py` after cloning to
> generate the table locally. If the table appears stale relative to the live
> `predictions/` DAG, regenerate.

## What to do next

- **Run the verifier.** `python3 verify.py` runs 65 backbone proofs in ~60 seconds. If any fail, the framework is wrong, full stop.
- **Read [`docs/honest_assessment.md`](docs/honest_assessment.md)** for explicitly what's proven, what's adopted, what's open, what would falsify.
- **Browse [`docs/parameters/target_parameters.md`](docs/parameters/target_parameters.md)** for every tracked parameter with its current grade and derivation file pointer.
- **Read the conceptual story** in [`docs/framework/narrative_spine.md`](docs/framework/narrative_spine.md) — *Mechanisms of recurrence: from toggle substrate to Standard Model* (the readable counterpart to the ~180-operation operator catalog).
- **See [`docs/incomplete_equations_todo.md`](docs/incomplete_equations_todo.md)** for the current open equations (the −70 ppm charged-lepton subleading, the M_Z oblique floor, the dark-sign formal lemma, the ζ_{D₄}(0) gauge-β formula origin, the substrate-selection discriminator), and [`docs/master_plan.md`](docs/master_plan.md) for the longer-horizon worklist (L6 cosmology cluster, Need-B δ_quark, N_hub first-principles derivation, the composite/bound-state sector).
- **Found a mistake?** File an issue. A substantive refutation against any closed prediction would be a real contribution and will be credited.

## Honest assessment

Every `*_derivation.md` under `predictions/` is a computational verification, not a fit — no parameters are adjusted, no coefficients are selected to match observation. The framework has honestly walked back items where re-audit found problems (m_t Koide waterfall 2026-05-04 then revived via M_persistence + Type-II saturation 2026-05-26; δ_CP^PMNS Hashimoto-phase 249.85° falsified 2026-05-02 then revived via V_{−1}–T_{B-L} identity 2026-05-05; n_s/r/σ_8/r_s/θ_* L6-blocked post 2026-05-15 Sprints A+B). It has also been explicit about boundaries (Δα low-energy hadronic threshold is out-of-scope-by-construction per the Clause-9 K-rationality bright-line; the absolute value of N_hub is pinned via the measured G_F as the framework's one calibration). What this framework does NOT derive — and what would falsify it — is at [`docs/honest_assessment.md`](docs/honest_assessment.md).

## Requirements

Python 3.8+ with `numpy`, `scipy`, `sympy`, and `matplotlib`.

```bash
pip install numpy scipy sympy matplotlib
```

## Repository structure

The [`docs/`](docs/) directory has its own index — [`docs/README.md`](docs/README.md) — mapping every subdirectory and its entry point. For a full cold-start tour of the layout, file-type conventions, and rigor machinery, see [`docs/orientation.md`](docs/orientation.md).

```
verify.py                     # Run 65 backbone proofs (~60 s)
run_predictions.py            # Regenerate predicted_parameters.md
predictions/                  # The per-parameter DAG (script + derivation pairs)
  retracted/                  # Honest archive of derivations that failed re-audit
proofs/                       # Proof scripts and exploratory machinery, by sector
  foundations/  flavor/  masses/  cosmology/  gauge/  lorentz/  wave_engine/
explorations/                 # Hypothesis-tested-and-archived scripts
scripts/validate_citations.py # Citation discipline tool
docs/
  README.md                   # Index to subdirs
  orientation.md              # Cold-start machinery doc — read this first
  quickstart.md               # 5-minute "show me one result" intro
  honest_assessment.md        # What's proven, what's not, what would falsify
  master_plan.md              # Canonical priority queue + framework state
  north_star.md               # The finish-line goal — read when scoping work
  framework/                  # Axioms + architecture + narrative + ontology
  theorems/                   # Closed theorem statements (~92 files)
  forward_constructions/      # Constructive bridges substrate → QFT objects (15)
  operator_sweep/             # Operator catalog + per-layer audits
  parameters/                 # Linter, target list, DAG chains, parameter ledger
  audits/
    registers/                # Live: uniqueness ledger, residue register, adoption register
  wave_engine/                # 219-op simulator catalog + cost methodology
```

## License

MIT
