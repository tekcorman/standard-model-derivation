# PAPER ASSIGNMENT — PRESORT (2026-07-10, for architect review)

**Status: DRAFT PRESORT — not registered anywhere.** This document assigns every row of the full
parameter universe to a paper (I / II / III / IV-1 / IV-2 / IV-3 / IV-4 / EXTERNAL-appendix) per the
seam test in internal research notes. It edits nothing else and commits
nothing.

## Sources read (state at read time)

- internal research notes — the seam test + per-paper explicit content
  lists (quoted directly wherever a row is named there; those citations are load-bearing and override
  generic heuristics).
- `docs/parameters/reads_manifest.md` — the machine classification of the **161** existing ledger rows
  (Tier/bin/status/blocker/calibration/N-tag), commit lock `2f7066a`, frozen 2026-07-09. Used as the
  master row list for the "existing" universe (more reliable than `target_parameters.md`'s own
  self-admittedly-drifted hand counts — see Disagreements below).
- `docs/parameters/target_parameters.md` — read in full (571 lines). **No expansion/Class-A..G section
  had been appended by the concurrent registration agent at read time** (file ends at the
  "Linter-ceiling audit notes" section). Per the task's fallback instruction, the 47 expansion rows
  were therefore taken directly from the standalone draft.
- `docs/parameters/target_expansion_draft_2026-07-10.md` — the 47-row expansion draft (Classes A–G),
  read in full.
- internal research notes — read in full. Its new row, **m_ββ (H-3), is the same
  row as the expansion draft's EXP-F1** — the harvest is computing the value for a row the draft already
  proposed, not a 209th row. No double count.

## The seam-test method actually applied (mechanical rule, in priority order)

1. **Explicit architecture citation wins.** Where `publication_architecture_2026-07-10.md` or the
   task's own method text names a row by symbol (e.g. "Ω_DM/Ω_m", "m_ν1=0", "V_ts/V_tb→I",
   "Ω_bh²/Ω_ch²→II", "S_BH/Hawking coefficient... fences-or-results→III"), that citation is used
   as-is over any general heuristic.
2. **Blocker/bin hints** (per the task's own authority #2): `species-lift/ML-2` → IV-2;
   `response/B2` → IV-4; `local-metric/ML-3/4/D1b` → IV-4; `external` → EXTERNAL-appendix of the
   architecture-declared host paper (Δα family, SUSY family → II; τ/z_reion → IV).
3. **Vertex-consuming beats dimension.** A row whose computation requires an actual S-matrix/decay-rate
   vertex insertion is Paper IV *regardless of being a dimensionless ratio* (the seam test's own
   example: branching ratios). This is why Γ_Z/M_Z and Γ_W/Γ_Z are pulled out of the "gauge-family"
   cluster into IV-1 (flagged — see Adjudication).
4. **Unit-bearing values default to Paper II** (need "the unit/tether") unless caught by rule 3.
   Dimensionless numbers/ratios/angles/exact counts with no adopted external input default to Paper I.
5. **Needing the ADOPTED z_eff** (external SN+BAO survey-design input) pulls an otherwise-dimensionless
   cosmological fraction (Ω_m_LCDM, Ω_Λ_LCDM, Ω_DM/Ω_b ΛCDM-frame) into Paper II — z_eff is "the state,"
   not spectrum-of-D-alone.
6. Genuinely two-sided calls are **not forced** — they are assigned a lean and cross-listed in the
   ADJUDICATION-NEEDED table at the end.

---

## PAPER I — THE SPECTRUM

### Gauge/Higgs structure (dimensionless, no scale threshold)

| Row | Status | Paper | Reason |
|---|---|---|---|
| α_GUT | ✅ | I | 1/24 exact at unification; no energy-scale threshold needed |
| sin²θ_W (at M_unif) | ✅ | I | 3/8 exact GQW-trace count; explicit Paper I item |
| λ (Higgs quartic) | ✅ | I | = 2·α₁_full, pure α₁ composition, no tether at all |
| δρ (ρ-param shift) | ✅ | I | pure α₁_bare/Feshbach resolvent identity — no M_Z scale value enters the formula |
| δ_r (M_Z tree→pole oblique) | ✅ | I | same resolvent family, pure function of α₁_bare |

### CKM (4 + phases)

| Row | Status | Paper | Reason |
|---|---|---|---|
| V_us, V_cb, V_ub, V_ud, V_cd, V_cs, V_td | ✅ | I | dimensionless mixing magnitudes |
| V_ts, V_tb | 🟡 | I | explicit architecture line 52: "V_ts/V_tb data-tension note→I" |
| δ_CP^CKM | ✅ | I | dimensionless geometric phase (arccos 1/3) |
| J_CKM (Jarlskog) | ✅ | I | dimensionless invariant |
| Georgi-Jarlskog ratio | ✅ | I | exact dimensionless count (k*=3) |

### QCD / Neutrino sector (dimensionless)

| Row | Status | Paper | Reason |
|---|---|---|---|
| θ_QCD | ✅ | I | 0 exactly, dimensionless |
| m_ν1 | ✅ | I | explicit Paper I item "m_ν1=0" |
| R_ν = Δm²_31/Δm²_21 | ✅ | I | explicit Paper I item "R_ν"; dimensionless splitting ratio |
| θ_12/13/23_PMNS, δ_CP_PMNS, α_21/31_PMNS | ✅ | I | dimensionless mixing angles/phases; explicit "PMNS magnitudes+phases" |

### Cosmology — spectral-only dimensionless reads

| Row | Status | Paper | Reason |
|---|---|---|---|
| A_s (primordial amplitude) | 🟡 | I | pure ratio chain (Feshbach exponent × 16/15 × α_GUT × (M_unif/M_Pl)²) — every factor is a dimensionless internal ratio; bin=S in the manifest, and target_parameters.md's own "12-observable §8-family" groups it with the other Paper-I quark/lepton reads |
| Ω_DM/Ω_m | ✅ | I | explicit Paper I item; dimensionless fraction (1−61e⁻⁶) |
| A_hemispherical | ✅ | I | dimensionless, ε_toggle·⟨(ê·ẑ)²⟩ |
| ε_CP_baryon | ✅ | I | explicit "ε_CP"; dimensionless 1/5 |
| η_B (baryon-to-photon) | ✅ | I | explicit "η_B"; dimensionless ratio |
| N_eff | ✅ | I | explicit "N_eff=3"; exact dimensionless count |
| w_DE (dark-energy component EoS) | ✅ | I | dimensionless ratio; the (16/15)² rate-gap cancels exactly, so no tether numerically enters |

### Structural / definitional ("gear")

| Row | Status | Paper | Reason |
|---|---|---|---|
| Spacetime dimension | ⚙️ | I | spectral count (d_spatial=3 Cencov-Fisher + time) |
| Gauge group | ⚙️ | I | explicit Paper I item |
| Number of generations | ⚙️ | I | explicit "N_gen=3" |
| Charge quantization | ⚙️ | I | explicit "Q=N̂/3" |
| Parity violation | ⚙️ | I | chiral asymmetry, pure spectral/structural fact |
| Fermion content | ⚙️ | I | Cl(6) spinor count |
| Higgs rep | ⚙️ | I | adopted labeling, structural |
| Lorentzian signature | ✅ | I | algebraic/spectral derivation (Cl(3,1)); bundled with lattice-structure results, distinct from Paper III's "exact light cone" locality theorem |

### Framework-internal (lattice / couplings / Koide / information)

| Row | Status | Paper | Reason |
|---|---|---|---|
| k*, d_spatial, g_girth, p_toggle | ✅ | I | lattice-structure integers |
| h_walker_eigenvalue | ✅ | I | dimensionless complex eigenvalue |
| srs_E_at_P, srs_cubic_moment | ✅ | I | dimensionless spectral numbers |
| srs_dirac_cone_velocities | ✅ | I | dimensionless velocity ratios (Lorentz-sig algebra) |
| srs_bloch_lv_dim6, η_5, η_lattice | ✅ | I | explicit Paper I "LIV pair" family |
| M_Pl_natural | ✅ 🔬 | I | pure multiple of e_bit, no GeV unit reported, N-independent |
| α_1_bare, α_1_full | ✅ | I | pure dimensionless couplings |
| y_τ (tau Yukawa) | ✅ | I | dimensionless Yukawa ratio |
| Feshbach exponent principle | ✅ | I | dimensionless formula |
| Q_Koide, ε_Koide, δ_Koide, koide_quark_ratio | ✅ | I | explicit "Koide + mass ratios"; dimensionless |
| λ toggle rate | ✅ 🔬 | I | dimensionless exact fraction (2/5), N-independent |
| S_fresh, S_disconfirm | ✅ 🔬 | I | pure information-theoretic bit counts, no physical unit |
| Observer H = C³ | ✅ | I | dimension count = 3, same family as N_gen |
| e_bit (energy of one substrate edge toggle) | ✅ 🔬 | **I†** | dimensionless (=1), N-independent — **flagged, see Adjudication #3** |
| ξ_t temporal correlation | ✅ 🔬 | **I†** | N-independent framework-internal constant — **flagged, see Adjudication #4** |

---

## PAPER II — THE CLOCK

### Gauge/Higgs at the physical scale (needs the tether)

| Row | Status | Paper | Reason |
|---|---|---|---|
| g_1, g_2, g_3 (at M_Z) | ✅ | II | RG-run down to the physical M_Z scale; manifest tags calibration-curve |
| sin²θ_W (at M_Z) | ✅ | II | same |
| α_s (M_Z) | ✅ | II | same |
| α_EM (M_Z) | 🟡 | II | same |
| M_unif | ✅ | **II‡** | dimensionful GUT scale (GeV) — **flagged, see Adjudication #8** |
| M_Z, m_W | 🟡 | II | explicit "Paper II... carries M_Z"; "M_Z/m_W (bridge-resolved)" |
| v (Higgs VEV) | ✅ | II | THE tether itself — "exactly ONE unit enters" |
| m_H (Higgs mass) | ✅ | II | explicit; dimensionful |
| λ_3 (Higgs trilinear) | ✅ | II | GeV, uses v + m_H |
| G_F (Fermi constant) | 🔬 | II | explicit; calibration round-trip |

### Charged fermion masses (all absolute masses)

| Row | Status | Paper | Reason |
|---|---|---|---|
| m_e | ✅ | **II§** | absolute mass; base value needs only tether — **flagged, see Adjudication #2** |
| m_μ | ✅ | **II§** | same — **flagged #2** |
| m_τ, m_u, m_d, m_s, m_c, m_b, m_t | ✅ | II | explicit "all absolute masses" |

### Neutrino masses (absolute)

| Row | Status | Paper | Reason |
|---|---|---|---|
| m_ν2, m_ν3 | 🟡 | **II§** | absolute mass (meV); "m_ν absolute" explicit — **flagged, see Adjudication #2** |

### Cosmology — the clock/tether/z_eff family

| Row | Status | Paper | Reason |
|---|---|---|---|
| Ω_b h², Ω_c h² | ⏳ CANDIDATE | II | explicit "Ω_bh²/Ω_ch²"; needs h = H_0/100 |
| Ω_DM (ΛCDM-frame), Ω_b (ΛCDM-frame) | 🟡 | II | need the ADOPTED z_eff (external survey-design input) |
| Ω_m_LCDM, Ω_Λ_LCDM | ✅ | II | same, need z_eff |
| Ω_k | ⏳ CANDIDATE | II | explicit Paper II item |
| Σm_ν | ⏳ CANDIDATE | II | explicit Paper II item |
| z_eff | ✅ | II | the adopted cosmology parameter itself (the state) |
| Λ_CC (substrate), Λ_LCDM | ✅ | II | explicit "H₀/t₀/Λ" family |
| w_eff (total fluid EoS) | ⏳ CANDIDATE | II | "the coasting suite" explicit |
| H_0 (substrate), H_0 (observer) | ✅ | II | explicit "H₀"; coasting/Hubble-tension family |
| t_0 (substrate) | ✅ | II | explicit "t₀" |
| t_0 (ΛCDM/CMB frame) | ❌ | II | coasting-suite sibling; H-1 harvest computes it via the MC-4 clock-map factor, not via r_*/recombination — NOT local-metric despite the row's older "L6-adjacent" note (stale pre-harvest characterization) |
| H(z), q_0 | ⏳ CANDIDATE | II | "the coasting suite" explicit |
| D_C(z), D_A(z), D_L(z), D_V(z) | ⏳ CANDIDATE | II | "the... distance ladder (Category-B genre)" explicit |
| β (cosmic birefringence) | ✅ | II | needs α_EM(M_Z), a tether-fenced input |
| T_ν_dec | 🟡 | II | explicit Paper II item |
| T_e_ann | ✅ | II | unit-bearing (MeV) |
| T(N) propagation function | ✅ | II | explicit Paper II item |

### Framework-adjacent / derived scales

| Row | Status | Paper | Reason |
|---|---|---|---|
| G (Newton's constant) | ⚙️ | II | task's own explicit instruction: "G's VALUE → II" (the separate 2π-fence *statement* is Paper III narrative content, not a distinct ledger row) |
| Scale energy (Hashimoto) | ✅ | **II¶** | dimensionful (PeV) derived hierarchy scale — **flagged, see Adjudication #9** |
| Universe transparency onset | ✅ | **II¶** | same — **flagged #9** |

### Expansion Class F (new row)

| Row | Status | Paper | Reason |
|---|---|---|---|
| EXP-F1 m_ββ (0νββ effective Majorana mass) | ⏳ CANDIDATE | II | explicit "m_ν absolute + m_ββ + Σm_ν (the DESI-contested zone)"; computed from certified PMNS angles + Majorana phases + masses only — no vertex. This is the harvest's H-3 row (same row, not an additional one). |

### EXTERNAL-APPENDIX of Paper II

**Δα (atomic-frame) family** — blocked on the α(M_Z)→α(0) bridge, declared external per architecture line 52 ("Δα→II"):

| Row | Status | Paper | Reason |
|---|---|---|---|
| R∞ (Rydberg) | ❌ | II-ext | needs derived Δα, un-derivable in-framework |
| T_HeII, T_HeI, T_recomb | ❌ | II-ext | same Δα blocker |

**SUSY family** — declared external per architecture line 52 ("SUSY retirement→II appendix"):

| Row | Status | Paper | Reason |
|---|---|---|---|
| tan β | 🟡 | II-ext | SUSY-cluster row |
| SUSY scale, m_gluino, m_squark, m_slepton, m_neutralino, m_chargino | ❌ | II-ext | not derived; SUSY appendix |
| m_h (light Higgs, MSSM) | 🟡 | II-ext | matches SM Higgs; SUSY appendix |
| m_H, m_A, m_H± (heavy Higgs) | ❌ | II-ext | SUSY appendix |

**Annotation-only rows** (per the architecture's own prior ruling, `publication_architecture_2026-07-10.md` lines 55–56: "the 2 composite-or-skip rows (σ_T, Λ_MS̄) register as annotations on their parents, not new rows"):

| Row | Status | Paper | Reason |
|---|---|---|---|
| EXP-A8 σ_T (Thomson cross section) | ❌ (composite-or-skip) | II-annotation | pure Type-4 composite of α_EM + m_e (both II); architecture already ruled this is not a standalone row |
| EXP-B8 Λ_QCD (MS̄) | ❌ (composite-or-skip) | II-annotation | derivable by running from the existing α_s(M_Z) row (II); same ruling |

---

## PAPER III — THE NET

| Row | Status | Paper | Reason |
|---|---|---|---|
| Matter stability | ⚙️ | III | task's own method text explicitly places this in the III list |
| Low initial entropy | ⚙️ | III | same |
| Branch measure μ | ✅ | **III†** | state-structure per the task's own "[state-structure → judgment: flag]" instruction — **flagged, see Adjudication #5** |
| Observer Hilbert space | ✅ | **III†** | axiom pairing (G.1,G.5)=(True,C), foundational algebra-of-observables content — **flagged, see Adjudication #6** |
| EXP-E1 S_BH coefficient | ❌ (STRUCT-TARGET) | III | explicit architecture text: "S_BH/Hawking coefficient statements live here as fences-or-results" |
| EXP-E2 T_H coefficient | ❌ (STRUCT-TARGET) | III | same; this is also literally the incomplete-equation object named by MG-1d (the emergent local Unruh/BW temperature) |

*(Narrative-only, not a distinct ledger row: the "G's 2π open-miss" fence statement itself — as opposed to G's numeric value, which is Row above in Paper II — is Paper III boundary-statement content per the task's own instruction.)*

---

## PAPER IV — THE TRANSACTION

### IV-1 — THE VERTEX (perturbative/leptonic)

| Row | Status | Paper | Reason |
|---|---|---|---|
| Γ_Z/M_Z | ✅ | **IV-1‡** | genuine decay-width/vertex-consuming quantity (radiative EW width layer) despite being a dimensionless ratio — the seam test's own branching-ratio example — **flagged, see Adjudication #1** |
| Γ_W/Γ_Z | ✅ | **IV-1‡** | same — **flagged #1** |
| EXP-A1 a_e | ❌ | IV-1 | the Schwinger-benchmark vertex row |
| EXP-A2 a_μ | ❌ | IV-1 | vertex |
| EXP-A6 τ_π0 (π⁰→γγ) | ❌ | IV-1 | vertex + anomaly channel; draft's own gate note cites no composite dependency |
| EXP-A7 R-ratio | ❌ | IV-1 | draft's own gate: "vertex (LSZ)" |
| EXP-A9 σ(νN) benchmark | ❌ | IV-1 | draft's own gate: "vertex (weak channel)" |
| EXP-A10 BR(H→bb̄)/BR(H→γγ) | ❌ | IV-1 | tree-vs-loop discriminator, reachable post-I-0 |
| EXP-A11 B(μ→eγ) | ❌ | IV-1 | draft's own gate: "vertex + PMNS machinery" |
| EXP-A12 B(b→sγ) | ❌ | IV-1 | one-loop + CKM chain; draft cites no composite/bound-state dependency |
| EXP-C1 m_DM | ❌ (prediction-only) | IV-1 | explicit architecture: "m_DM as the flat-band self-energy gap" |

### IV-2 — COMPOSITES (bound-state scale, incl. BBN)

| Row | Status | Paper | Reason |
|---|---|---|---|
| z_drag | ❌ NOT STARTED | IV-2 | blocker species-lift/ML-2 |
| T_BBN-1, T_BBN_D | ❌ | IV-2 | same blocker |
| Y_p | ❌ | IV-2 | same blocker |
| D/H, ³He/H, ⁷Li/H | ❌ NOT STARTED | IV-2 | same blocker |
| EXP-A3 τ_n (neutron lifetime) | ❌ | IV-2 | draft's own gate cites g_A (EXP-B16, a composite/bound-state coupling) — pulled out of nominal "Class A" into IV-2 by the "latest-consumed-object" rule |
| EXP-A5 τ_π± | ❌ | IV-2 | draft's own gate cites f_π (EXP-B6, composite) |
| EXP-B1 m_p, EXP-B2 m_n | ❌ | IV-2 | Bethe–Salpeter bound-state masses |
| EXP-B3 Q_np (=m_n−m_p) | ❌ | IV-2 | named missing BBN input, now tracked |
| EXP-B4 m_π±, EXP-B5 m_π0 | ❌ | IV-2 | composite pion masses |
| EXP-B6 f_π | ❌ | IV-2 | composite decay constant |
| EXP-B7 m_K | ❌ | IV-2 | composite kaon masses |
| EXP-B9 r_p | ❌ | IV-2 | atomic bound-state structure (+ Δα co-gate, noted) |
| EXP-B10 H 1S–2S | ❌ (Δα-gated) | IV-2 | atomic bound state; double-gated (vertex + Δα) |
| EXP-B11 Lamb shift | ❌ (Δα-gated) | IV-2 | same |
| EXP-B12 21cm hyperfine | ❌ (Δα-gated) | IV-2 | same, plus nucleon μ_p |
| EXP-B13 positronium 1S–2S | ❌ (Δα-gated) | IV-2 | pure-lepton atom, still Δα-gated |
| EXP-B14 positronium lifetimes | ❌ | IV-2 | rates, NOT Δα-gated per the draft's own note |
| EXP-B15 B_d (deuteron binding) | ❌ | IV-2 | named missing nuclear input, now tracked |
| EXP-B16 g_A (nucleon axial coupling) | ❌ | IV-2 | named missing input (feeds τ_n, Y_p), now tracked |
| EXP-F2 N_eff^BBN | ❌ | IV-2 | draft's own gate: "rides the BBN block" |
| EXP-G5 Y_p^CMB | ❌ | IV-2 | draft's own gate: "the BBN block for EXP-G5" — pulled out of nominal "Class G" (thematically CMB) into IV-2 because its stated gate is BBN, not acoustics |

### IV-3 — STRONG DYNAMICS (confined scale)

| Row | Status | Paper | Reason |
|---|---|---|---|
| EXP-D1 √σ (string tension) | ❌ | IV-3 | confinement/twisted-zeta order parameter |
| EXP-D2 T_c (deconfinement) | ❌ | IV-3 | same |
| EXP-D3 m(0⁺⁺) glueball | ❌ | IV-3 | same |
| EXP-D4 m_η′ | ❌ | IV-3 | anomaly/topology channel |
| EXP-D5 χ_top | ❌ | IV-3 | same |

### IV-4 — COSMOLOGICAL DYNAMICS (macro scale)

| Row | Status | Paper | Reason |
|---|---|---|---|
| 100θ_MC | ❌ | IV-4 | blocker local-metric/ML-3/4/D1b |
| n_s | ❌ | IV-4 | blocker response/B2 |
| r (tensor-to-scalar) | ❌ | IV-4 | growth/primordial-tensor family (dedupe of EXP's "r") |
| σ_8, S_8, D(z), f(z), fσ_8(z) | ❌ | IV-4 | blocker response/B2; explicit "growth (σ_8, S_8, n_s, D/f/fσ_8)" |
| r_*, r_drag, θ_* | ❌ | IV-4 | blocker local-metric/ML-3/4/D1b; explicit "acoustics (θ_*, r_*, r_s)" |
| z_* | ⏳ CANDIDATE | IV-4 | same blocker |
| z_eq | ❌ | IV-4 | explicit "z_eq (the KMS-transition reframe)" |
| EXP-C2 σ_SI | ❌ (bound row) | IV-4 | explicit "dark-sector astro confrontations... σ_SI" |
| EXP-C3 σ/m (Bullet Cluster) | ❌ (bound row) | IV-4 | astro confrontation of the self-interaction claim |
| EXP-E3 γ−1 (PPN) | ❌ | IV-4 | explicit "PPN γ/c_gw if a path materializes" |
| EXP-E4 \|c_gw/c−1\| | ❌ | IV-4 | same |
| EXP-G1 ℓ1 (peak position/height) | ❌ | IV-4 | acoustic block |
| EXP-G2 peak height ratios | ❌ | IV-4 | acoustic block |
| EXP-G3 k_D (damping scale) | ❌ | IV-4 | acoustic block |
| EXP-G4 A_L (lensing) | ❌ | IV-4 | growth/response block |

### EXTERNAL-APPENDIX of Paper IV

| Row | Status | Paper | Reason |
|---|---|---|---|
| τ (reionization optical depth) | ❌ NOT STARTED (framework-external) | IV-ext | explicit architecture: "τ/z_reion→IV"; stellar UV, no framework primitive |
| z_reion | ❌ NOT STARTED (framework-external) | IV-ext | same |

---

## ADJUDICATION-NEEDED (not forced — competing assignments + lean)

| # | Row(s) | Competing assignments | Lean | Why it's genuinely open |
|---|---|---|---|---|
| 1 | Γ_Z/M_Z, Γ_W/Γ_Z | IV-1 (vertex/decay-rate) vs II (currently shipped inside the RG/oblique gauge-family cluster, calibration-curve-tagged like sin²θ_W(M_Z)) | IV-1 | Physically these ARE decay-rate/branching-ratio-class S-matrix objects (the "ew_width_layer" radiative-width computation), unlike δρ/δ_r which are pure self-energy/resolvent identities — but the ledger currently treats all four as one "gauge-family" cluster |
| 2 | m_e, m_μ, m_ν2, m_ν3 (the −70 ppm / "m_ν scale" family) | II (current base absolute-mass value, tether-only, no vertex yet in the derivation) vs IV-1 (architecture explicitly names "the ppm resolution (m_e, m_μ, m_ν scale)" as IV-1 content) | II now; IV-1 owns the eventual resolution of the same symbol | The row as it stands today is II-computable; the still-open residual specifically is IV-1's target once a vertex loop correction exists — same symbol, two papers across time |
| 3 | e_bit | I (dimensionless =1, N-independent, self-normalizing natural unit) vs II (architecture's own text: "h derived, κ=h/t_P, **Landauer**, thermal time"; e_bit's docstring itself cites "Landauer quantum" as one of its three equivalent statements) | I | Direct textual echo of "Landauer" pulls toward II even though the reported value carries no physical unit |
| 4 | ξ_t_temporal_correlation | I (N-independent) vs II (value explicitly carries "ℓ_P," a Planck-scale reference, and sits in the same "thermal time" cluster) | I | Same tension as #3, one level milder |
| 5 | Branch measure μ | I (pure Stage-1 multiway-measure theorem, structural) vs III (task's own instruction explicitly flags this row for judgment) | III | Following the task's explicit flag instruction rather than overriding it |
| 6 | Observer Hilbert space | I (paired with "Observer H=C³=3," which is confidently I) vs III (the (G.1,G.5) axiom pairing is foundational algebra-of-observables content, adjacent to Haag–Kastler) | III | Genuinely closer to Paper III's axiom-setup territory than to a bare spectral count |
| 7 | EXP-A4 τ_μ | IV-1 (nominal S-matrix lifetime observable) vs EXCLUDED/II-round-trip (τ_μ is literally the observable G_F is extracted from, and G_F tethers N_hub) | Register, but flag CALIBRATION-ROUND-TRIP; do not score until the τ_μ→G_F→N_hub loop is audited (the draft's own recommendation) | A framework "prediction" of τ_μ risks pure circularity with Paper II's own tether |
| 8 | M_unif | II (dimensionful GeV scale, needs the physical-unit conversion) vs I (N-tag "independent" — a fixed structural multiple of M_Pl, no N_hub dependence, same flavor as "α_GUT⁻¹=24") | II | The dimension/unit rule is applied consistently as the tie-breaker per the task's explicit method wording |
| 9 | Scale energy (Hashimoto), Universe transparency onset | II (dimensionful PeV derived-hierarchy scale, like M_unif) vs IV-1 (arguably a γγ pair-production-threshold S-matrix confrontation against SWGO/GRB data) | II | No cross-section is actually computed by the framework here — it is a derived energy scale phenomenologically corroborated, not a computed rate |

---

## Manifest / ledger / draft disagreements noted

- **KO-6** appears in the architecture's explicit Paper I list ("gauge group, Q=N̂/3, KO-6") but there is
  **no distinct ledger row named KO-6** in either `target_parameters.md` or `reads_manifest.md`. It
  likely refers to the KO-dimension/exotic-presentation theorem (R2b, MEMORY) rather than a tracked
  target-parameter row. No assignment made; flagged for the architect in case it should become its own
  row.
- **`target_parameters.md`'s own row counts are stale and self-admittedly drifted** (the file's own
  banners cite 123, then 114, then a "recount... 114," with explicit notes that later hand-sync passes
  were never fully reconciled). This presort used `reads_manifest.md`'s machine-generated 161-row table
  as the authoritative existing-row list instead, per the task's own steer that the manifest is "the
  machine classification."
- **No expansion section was present in `target_parameters.md`** at read time — the concurrent
  registration agent had not yet appended one. All 47 expansion rows were sourced from the standalone
  draft per the task's fallback instruction. If the registration agent has since appended a section with
  different row content or paper tags, this presort should be reconciled against it before use.
- **m_ββ is not a 209th row.** `R1_harvest_prereg_2026-07-10.md`'s H-3 ("NEW ROW... m_ββ") and the
  expansion draft's EXP-F1 are the same proposed row — the harvest station is actively computing the
  value for the row the draft already proposed. Counted once.
- **Two of the draft's "47 new rows" are already downgraded by the architecture document itself**
  (EXP-A8 σ_T, EXP-B8 Λ_QCD/Λ_MS̄ — "composite-or-skip... register as annotations on their parents, not
  new rows," `publication_architecture_2026-07-10.md` lines 55–56). This presort honored that prior
  ruling rather than re-opening it; both are listed as annotations under Paper II, not as standalone
  IV-1/IV-2 rows despite sitting in the draft's Class A/B (S-matrix/composite) sections thematically.

---

## Sanity totals

**Manifest existing rows: 161** (verified by direct count of `reads_manifest.md`'s "full ledger-row
table"; matches the manifest's own stated total).

**Expansion draft new rows: 47** (Class A 12 + B 16 + C 3 + D 5 + E 4 + F 2 + G 5 = 47; matches the
draft's own checklist total). The draft's separately-declared **6 dedupes** (R∞, Ω_DM h², Σm_ν, r,
τ_reion, S_8) are annotation-refresh suggestions on rows **already inside the 161** — they add zero new
rows and are not double-counted.

**Grand total: 161 + 47 = 208 rows** in the full parameter universe presorted here. (The task's framing
"~41 new + 6 dedupes" appears to derive from 47 − 6 = 41 by mis-treating the dedupes as if subtracted
from the 47; the draft's own text keeps the two counts — 47 new, 6 dedupe-annotations on existing rows
— strictly separate and non-overlapping. 208 is the reconciled figure.)

### Per-paper counts (208 total; flagged/adjudicated rows counted under their stated lean, not forced)

| Paper | Count | Of which from manifest-161 | Of which from expansion-47 |
|---|---|---|---|
| **I** | 66 | 66 (64 confident + 2 flagged-lean: e_bit, ξ_t) | 0 |
| **II** (core + z_eff/coasting/tether family) | 68 | 54 (incl. m_e/m_μ/m_ν2/m_ν3 flagged-lean, and M_unif/Hashimoto-scale flagged-lean) + 13 external-appendix | 1 (EXP-F1 m_ββ) |
| **III** | 6 | 4 (2 confident: Matter stability, Low initial entropy; 2 flagged-lean: Branch measure μ, Observer Hilbert space) | 2 (EXP-E1, EXP-E2) |
| **IV** (all sub-volumes + appendix) | 65 | 24 (2 flagged-lean IV-1: Γ_Z/M_Z, Γ_W/Γ_Z; 7 IV-2 species-lift; 13 IV-4 local-metric+response/B2+others; 2 IV-ext: τ, z_reion) | 41 (9 IV-1 + 19 IV-2 + 5 IV-3 + 8 IV-4) |
| **Annotation-only** (not standalone rows; land on Paper II parents) | 2 | 0 | 2 (EXP-A8 σ_T, EXP-B8 Λ_QCD) |
| **Unassigned pending adjudication** (EXP-A4 τ_μ — calibration-round-trip risk, no paper forced) | 1 | 0 | 1 |
| **Total** | **208** | **161** | **47** |

Reconciliation: 66 + 68 + 6 + 65 + 2 + 1 = 208. ✓

**ADJUDICATION-NEEDED entries: 9** (covering 13 distinct symbols: Γ_Z/M_Z, Γ_W/Γ_Z, m_e, m_μ, m_ν2,
m_ν3, e_bit, ξ_t_temporal_correlation, Branch measure μ, Observer Hilbert space, τ_μ, M_unif, Scale
energy/Universe transparency-onset pair) — all counted under their stated lean above, none forced past
the lean.

## ARCHITECT ADJUDICATIONS (2026-07-10, architect) — the 9 flagged rows, decided
1. **Γ_Z/M_Z, Γ_W/Γ_Z → PAPER II.** The seam test rules on what the CURRENT computation consumes:
   both ship today from spectrum+state (no vertex). Re-confronted in IV's loop-exposure like all of
   Ring 0. (IV-1 reports corrections, not custody.)
2. **m_e, m_μ, m_ν2, m_ν3 → PAPER II**, each carrying a forward-reference: "subleading
   correction/scale resolution = IV-1." The row lives where its value is computed; the correction
   paper reports the delta and the re-confrontation.
3. **e_bit → PAPER II** (dimensionful — energy per bit; the Landauer anchor is clock-layer).
4. **ξ_t → PAPER II** (carries the ℓ_P reference; unit-bearing).
5. **Branch measure μ → PAPER III** (it IS the Born-measure object; QF-1's home).
6. **Observer Hilbert space → PAPER III** (foundational/measurement structure).
7. **τ_μ → PAPER II's CALIBRATION SECTION** (not a confrontation row: it is the tether's own source
   observable, alongside G_F/v; the future tuning-fork station is its audit gate).
8. **M_unif → PAPER II** (GeV scale; its N-independence tag ≠ unit-freeness).
9. **Scale energy (Hashimoto) + universe transparency onset → PAPER II** (epoch/unit-bearing).
KO-6: confirmed a THEOREM (R2b), Paper I content, not a ledger row — no row created.
**FINAL COUNTS (208+1 recount): PAPER I = 65 (count corrected 2026-07-10 per the claim-sheet agent's two independent re-derivations; the earlier 64 was an arithmetic slip in this doc) · PAPER II = 73 (incl. 13 external-appendix + calibration section)
· PAPER III = 6 rows (+ ~25 structural theorems/fences — III is results-heavy, row-light by design) ·
PAPER IV = 63 (9 IV-1 / 19 IV-2 / 5 IV-3 / 8 IV-4 + 22 manifest-side incl. the ppm/BBN/acoustic/growth
blocks) · annotations = 2. Reconciled: 65+73+6+63+2 = 209 (one row was double-counted-out in the earlier arithmetic; the ledger row set is unchanged).**
