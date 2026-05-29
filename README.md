# Standard Model from First Principles

### 📖 Read the interactive explainer → **[tekcorman.github.io/standard-model-derivation](https://tekcorman.github.io/standard-model-derivation/)**

*A visual narrative — the story, the two graphs, the walks, the predictions — with animations and an interactive 3D crystal viewer. This README is the technical summary; the explainer is the guided tour.*

---

> **The same substrate object, read 12 different ways, matches PDG observations on all 12 — with zero fitted constants.**

The 12 observables are 7 quark-sector (y_t, y_b, V_us, V_cb, V_ub, δ_r, δρ), 4 lepton/PMNS (y_τ, θ_12, θ_13, θ_23), and the A_s cosmological prefactor — all read from the **non-backtracking resolvent** G_NB = (I − u·B_NB(srs))⁻¹ with one argument `a = (2/3)⁸` and zero fitted constants. **Then 79 more parameters fall out of the same substrate.** Cross-validation audit: [`docs/state_of_the_lepton_pmns_over_determination_2026-05-23.md`](docs/state_of_the_lepton_pmns_over_determination_2026-05-23.md).

**The framework in one sentence.** Three meta-commitments — self-containment of the universe, finite observer, active reading of binary distinctions — plus standard published mathematics, force a substrate (the srs crystal net) whose spectral content is the Standard Model. One empirical labeling rule (A5-mass: which substrate eigenvalues are which observed masses) attaches contact with experiment. There are no further inputs.

## Status (2026-05-26)

Across **123 tracked targets**: **91 ✅ closed** (matches observation at the parameter-linter grade), **9 🟡 in progress** (file exists, structural gap named), **13 ❌ open or out-of-scope** (genuinely open: n_s, σ_8, r_s, θ_*, recombination quantities — the L6 cluster; out-of-scope-by-construction: Δα low-energy hadronic threshold per the Clause-9 K-rationality bright-line), **10 ⚙️ structural** (definitional identifications).

The `predictions/` directory is the source of truth: each parameter has a `.py` (the prediction) and a `_derivation.md` (the journal-grade write-up); `_validate_dag.py` enforces self-containment (120 files, 0 forbidden imports). Run `python3 verify.py` (25 backbone proofs, ~10 seconds); if any fail, the framework is wrong, full stop.

Highlights of the last six weeks: the Lorentz arc closed (2026-04-27); the five-stage gauge-coupling chain closed including sector-specific c_color = 1/4 (2026-05-04 → 2026-05-26); the substrate-net was forced uniquely to be **srs** via Sunada 2012 + the no-privilege principle (R-9 closure, 2026-05-12); the 12-observable §8 over-determination landed (2026-05-16/23); the multi-axial dark-sector waterfilling theorem promoted to theorem-grade-structural (2026-05-24); the M_persistence 12-mass fermion operator shipped (2026-05-26). These are partial readings of one substrate object — consolidated (not newly derived) in [`docs/theorems/theorem_walker_matter_unification_2026-05-27.md`](docs/theorems/theorem_walker_matter_unification_2026-05-27.md). Per-closure detail in [`docs/honest_assessment.md`](docs/honest_assessment.md) and [`docs/master_plan.md`](docs/master_plan.md).

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
| **This framework** | **91 numerical parameters closed; 12 cross-validated via a single resolvent** | **One graph (srs); the non-backtracking resolvent G_NB on it** |

*None of these are strawmen — each programme has substantive open work. The table compares specifically on whether numerical Standard Model parameter values are pinned (not adjusted, not selected from a landscape, not left as free parameters in the action).*

## What follows

From these commitments, the MDL-optimal graph is the **srs crystal net** (space group I4₁32, coordination number k\* = 3, girth g = 10). Two algebraic structures on this graph encode all physics:

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
| Q_Koide / ε_Koide / δ_Koide | 2/3, √2, 2/9 | matches | ✅ STRICT-SOLID (k\* = 3) exact algebraic identities; the lift to the *observed charged-lepton* Koide ratio is empirical phenomenology — flagged in `Q_Koide_derivation.md` |
| Higgs VEV v | 246.22 GeV | 246.22 ± 0.12 GeV | −0.00σ ✅ UNIQUE-THEOREM-GRADE (G1b R2 path 2026-04-28; N_hub-class anchor — value pinned via the measured G_F) |
| Higgs mass m_H | 125.20 GeV | 125.20 ± 0.11 GeV | −0.05σ ✅ THEOREM-GRADE-STRUCTURAL (Family-D propagated 2026-05-15; conditional on c_H Route H/C closure) |
| Higgs quartic λ | 0.12927 = 2·α₁_full | 0.1294 | −0.05σ ✅ THEOREM-GRADE-STRUCTURAL (W1 2026-05-18) |
| m_τ | 1.7768 GeV = v·y_τ·Family-D | 1.77686 ± 0.00012 | −0.19σ ✅ THEOREM-GRADE-STRUCTURAL (W1 2026-05-18 reinstatement; prior "UNIQUE-THEOREM-GRADE-NUMERICAL Family-D" was a Clause-6c smuggle) |
| m_t | 174.10 GeV (Type-II saturation + MSSM RGE) | 172.69 ± 0.30 GeV | +0.82% rel (+4.71σ_PDG; residual is MSSM-threshold + two-loop class) — THEOREM-GRADE-STRUCTURAL-CONDITIONAL via M_persistence (Row P38) |
| m_b / m_c / m_s / m_d / m_u | M_persistence 12×12 fermion mass operator | PDG 2024 | m_c, m_s, m_d, m_u within 1σ_PDG ✅; m_b +2.99σ borderline (Clause-8 within ~2% relative); THEOREM-GRADE-STRUCTURAL-CONDITIONAL via Row P39 |
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

The framework makes specific numerical predictions, several at the precision frontier. Any of these going against measurement is fatal:

| Prediction | What kills it | Experiment | Timeline |
|---|---|---|---|
| **δ_CP^PMNS = 180°** | Clearly maximal-CP-violating ~270° (outside ~180° ± 30°) | DUNE, Hyper-K | 2028+ |
| θ_23 = 48.72° (non-maximal) | Exactly maximal: 45.00 ± 0.3° | DUNE | 2028 |
| **m_ν₁ = 0 exactly** | 0νββ never observed AND m_ν₁ > 0 established | KATRIN, Project 8, nEXO | 2027+ |
| No WIMP dark matter (DM = gauge-decoupled uncompressed multiway) | WIMP found at LZ / XENONnT | LZ, XENONnT | ongoing |
| Δb_2 = +4 gap between substrate-derived 2HDM β (−3) and observation-imposed MSSM β (+1) at SU(2)_L — predicted as a precise characterization, not closed | α_s(M_Z) precisely contradicts MSSM b_3 = −3 under α_GUT⁻¹ = 24 inversion | LHC, FCC-hh (gauge-coupling precision) | ongoing |
| \|β cosmic birefringence\| ≤ α_EM ≈ 0.418° (hard cap from c_1 = 0) | β measured > 0.418° | LiteBIRD (~0.05° precision) | ~2032 |
| η_5 = 0 exactly (dim-5 LIV) | dim-5 LIV detected | LHAASO, HESS | ongoing |
| α₂₁ = 162.39° (Majorana phase) | α₂₁ outside ~162° ± 30° | nEXO, LEGEND-1000 | 2030+ |
| α₃₁ = 324.78° (Majorana phase) | α₃₁ outside ~325° ± 30° | future 0νββ | — |
| m_ββ ≈ 2.55 meV (0νββ amplitude from m_ν₂ + α_21 chain) | m_ββ measured outside ~1–5 meV | nEXO, LEGEND-1000 | 2030+ |

**One historical falsification has fired as designed.** The original Hashimoto-phase route for δ_CP^PMNS predicted 249.85° (±30°). It failed at +3.83σ vs NuFIT 6.0 IC19 on 2026-05-02 and was retired. The current value 180° came from an **independent parameter-free identity** — V_{−1}–T_{B-L} = arccos(−1) — three days later (2026-05-05). The retracted derivation is preserved at `predictions/retracted/delta_CP_PMNS.py` as honest record. The same identity simultaneously fixes δ_CP^CKM = arccos(1/3) = 70.53° at +0.68σ on a *different* observable — independent corroboration. This is the falsification-and-revival pattern the framework is built around.

## Audit infrastructure

The claim that no step curve-fits is held by **six audit instruments**, each watching a distinct way derivations can cheat:

1. **Parameter linter** ([`docs/parameters/parameter_linter.md`](docs/parameters/parameter_linter.md)) — Type-1/2/3/4/5/6 gate per derivation step. Clauses 7 / 8 / 9 are blocking gates (multi-axis multi-mechanism uniqueness defense; σ_PDG-strict numerical match; K-rationality bright-line audit), added 2026-04-30 / 2026-05-01 / 2026-05-15.
2. **Structural uniqueness ledger** ([`docs/audits/registers/uniqueness_ledger.md`](docs/audits/registers/uniqueness_ledger.md)) — 25 rows. For every structural choice (k\* = 3, srs lattice, Cl(6,0) algebra, q_NB = 2/3, …), audits why this one and not an alternative.
3. **Parameter uniqueness ledger** ([`docs/parameters/parameter_uniqueness_ledger.md`](docs/parameters/parameter_uniqueness_ledger.md)) — 68 P-rows. Same audit at the formula layer (V_us = 9/40, α₁_bare = (2/3)⁸, V_cb = 256/6305, η_B = (√3/10)·(2/3)⁴⁸, …).
4. **Structural residue register** ([`docs/audits/registers/structural_residue_register.md`](docs/audits/registers/structural_residue_register.md)) — soft-gated alternatives with status (OPEN / TRACED / ACCOUNTED-FOR / REFUTED).
5. **Wave engine** ([`proofs/wave_engine/simulator.py`](proofs/wave_engine/simulator.py)) — 219 catalog operations; all firing post-2026-04-27 LORENTZ_SIG / CCLOSE → NC_GEOM joint closure; open-frontier tags = ∅.
6. **Citation validator** ([`scripts/validate_citations.py`](scripts/validate_citations.py)) — operational pre-commit-hook candidate.

Full descriptions: [`docs/orientation.md`](docs/orientation.md) §4.

## Quick start

```bash
python3 verify.py                                      # Run backbone proofs (~10 s)
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

- **Run the verifier.** `python3 verify.py` runs 25 backbone proofs in ~10 seconds. If any fail, the framework is wrong, full stop.
- **Read [`docs/honest_assessment.md`](docs/honest_assessment.md)** for explicitly what's proven, what's adopted, what's open, what would falsify.
- **Browse [`docs/parameters/target_parameters.md`](docs/parameters/target_parameters.md)** for every tracked parameter with its current grade and derivation file pointer.
- **Read the conceptual story** in [`docs/framework/narrative_spine.md`](docs/framework/narrative_spine.md) — *Mechanisms of recurrence: from toggle substrate to Standard Model* (the readable counterpart to the ~180-operation operator catalog).
- **See [`docs/master_plan.md`](docs/master_plan.md)** for the current frontier (L6 cosmology cluster, Need-B δ_quark, N_hub first-principles derivation, two-loop substrate-thermal corrections, literal-particle β-coefficient gap [R-19]).
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
verify.py                     # Run backbone proofs (~10 s)
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
