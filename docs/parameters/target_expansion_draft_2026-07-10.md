# TARGET-LIST EXPANSION — DRAFT FOR ARCHITECT REVIEW (2026-07-10)

> **STATUS: DRAFT — NOT REGISTERED.** Nothing in this file is on the ledger. This is the drafting
> agent's deliverable per the master closure program's expansion directive
> (internal research notes): the new observable rows proposed for
> registration on `docs/parameters/target_parameters.md`. That file has **not** been edited.
> Registration is an architect decision, row by row.
>
> **Verification discipline:** every measured value below was web-verified 2026-07-10 against the
> stated source (PDG 2024/2025, CODATA 2022, Planck 2018/PLA, ACT DR6, or the named collaboration
> paper). Anything that could not be verified to citation grade carries an explicit **TO-VERIFY**
> flag — no guessed numbers. Peak-height *ratios* in Class G are computed from primary-verified
> heights and are marked DERIVED.
>
> **Dedupe discipline:** `target_parameters.md` was grepped per row before drafting. Existing rows
> get an ANNOTATION entry (no duplicate row). 6 dedupe hits found; each is listed in its class
> section and in the checklist.
>
> **Conventions:** table format, status-flag vocabulary (❌ NOT STARTED / ⏳ CANDIDATE, with
> qualifiers in parentheses), and the "Observed | Predicted | Status | File | Notes" column order
> match the ledger exactly. All Predicted cells are "—" (nothing is claimed here); all File cells
> are "(no file)". Rows are keyed EXP-⟨class⟩⟨n⟩ for review reference only — keys are not proposed
> ledger content. Grade language from the private derivation is not used, per the ledger's rules.

---

## CLASS A — S-matrix / decay-rate observables (12 rows)

**Gate object (master program):** RING 3 — the I-0 reconciled vertex (the −κ·I(A;B) MDL
information transaction), fanning out through the existing LSZ scaffold (`forward_constructions`;
its own logged gap: "requires vertex insertions"). **Integrity check:** the Schwinger-class
benchmark — the reconciled vertex's leading loop correction to a_e must reproduce the α/2π-class
term or the vertex is wrong (Ring 2, I-0 acceptance gate). Until I-0 lands, every row here is
❌ NOT STARTED by construction.

| Symbol | Observed | Predicted | Status | File | Notes |
|---|---|---|---|---|---|
| a_e (electron anomalous moment) | 1.159 652 180 59(13) × 10⁻³ (Fan, Myers, Sturm, Gabrielse, PRL 130, 071801, 2023; g/2 = 1.001 159 652 180 59(13), 0.13 ppt) | — | ❌ NOT STARTED | (no file) | **EXP-A1.** The single most precisely confronted number in physics; the master program's Schwinger benchmark row — the I-0 vertex's leading correction must reproduce the α/2π class or the vertex is wrong. CODATA 2022 concurs (last-two-digit transcription **TO-VERIFY** at registration). HONEST CAVEAT: the SM-theory comparison depends on which α feeds it — the Berkeley Rb (α⁻¹ = 137.035 999 206(11), 2018) vs LKB Cs (137.035 999 046(27), 2020) atom-recoil determinations disagree at ~5.5σ, so "a_e agrees/disagrees with SM" is currently α-input-dependent. The framework confronts a_e directly (its own α), sidestepping that fork — a genuinely clean 12–13-digit target. |
| a_μ (muon anomalous moment) | 116 592 070.5 ± 11.4(stat) ± 9.1(syst) ± 2.1(ext) × 10⁻¹¹ (Fermilab Muon g−2 FINAL, full Run 1–6, announced 2025-06-03; 127 ppb) | — | ❌ NOT STARTED | (no file) | **EXP-A2.** TENSION STATUS (state honestly, both sides): vs the 2020 Theory Initiative White Paper (data-driven e⁺e⁻ HVP, a_μ^SM = 116 591 810(43) × 10⁻¹¹) the experiment sits at ~5σ; vs WP25 (2025, lattice-QCD HVP incl. BMW, a_μ^SM = 116 592 033(62) × 10⁻¹¹) the tension collapses to ~0.6σ. The WP20→WP25 theory shift is itself ~3σ (lattice vs dispersive HVP un-reconciled) — the "anomaly" now lives INSIDE the SM theory error budget, not between experiment and theory. A framework value would adjudicate a live dispute. |
| τ_n (neutron lifetime) | 878.4 ± 0.5 s (PDG world avg, scale-factor-inflated). Bottle: 877.75 ± 0.28 +0.22/−0.16 s (UCNτ, PRL 2021). Beam: 888.0 ± 2.1 s (NIST) | — | ❌ NOT STARTED | (no file) | **EXP-A3.** HONEST DISCREPANCY: beam and bottle methods differ by ~9.8 ± 2.0 s (~4σ) — a famous unresolved experimental problem. Any framework confrontation must declare WHICH τ_n it targets (bottle = total decay rate; beam = decay-to-proton channel), and the discrepancy itself is a candidate discriminator. Gate: vertex + weak harness + g_A (EXP-B16) — the same nucleon-sector chain as Y_p (§9). |
| τ_μ (muon lifetime) | 2.196 981 1(22) μs (PDG; dominated by MuLan, 1 ppm) | — | ❌ NOT STARTED (CALIBRATION-ROUND-TRIP RISK) | (no file) | **EXP-A4.** ⚠ FLAG AS DIRECTED: τ_μ is THE observable G_F is extracted from (1/τ_μ = G_F² m_μ⁵/(192π³)·(1+Δq) → G_F = 1.166 3788(6) × 10⁻⁵ GeV⁻²), and the ledger's G_F row (🔬) is downstream of the N_hub calibration, which is G_F-tethered (CALIBRATION SECTION, per protocol). A framework "prediction" of τ_μ therefore risks a calibration round-trip — likely CALIBRATION-EXCLUDED. Recommend registering WITH an explicit round-trip audit gate: no confrontation scored until the τ_μ→G_F→N_hub loop is audited per the epoch/calibration guardrail. |
| τ_π± (charged pion lifetime) | 2.6033(5) × 10⁻⁸ s (PDG) | — | ❌ NOT STARTED | (no file) | **EXP-A5.** Weak decay π⁺→μ⁺ν_μ; the rate carries f_π (EXP-B6) × V_ud (✅ existing row) × G_F — a composite confrontation once the vertex + f_π land. |
| τ_π0 (neutral pion lifetime) | 8.43(13) × 10⁻¹⁷ s (PDG world avg). Most precise input: 8.34 ± 0.19 × 10⁻¹⁷ s (PrimEx-II, Larin et al., Science 368, 506, 2020) | — | ❌ NOT STARTED | (no file) | **EXP-A6.** π⁰→γγ is the direct test of the chiral (Adler–Bell–Jackiw) anomaly — the anomaly-normalization observable. HONEST CAVEAT: the two most precise inputs (CERN direct-timing vs PrimEx-II Primakoff) differ ~6% inside the average. Framework-side, this row confronts the same anomaly machinery as the W2 chiral seed / graded-blindness results — but no numeric claim exists; gate = vertex + anomaly channel. |
| R-ratio (σ(e⁺e⁻→hadrons)/σ(e⁺e⁻→μ⁺μ⁻)) at declared energies | R̄_uds = 2.224 ± 0.019 ± 0.089 at √s = 3.650 GeV (BES); R = 3.56 ± 0.01 ± 0.07 at √s = 10.52 GeV (CLEO, PRD 76, 072008, 2007) | — | ❌ NOT STARTED | (no file) | **EXP-A7.** Two DECLARED benchmark energies (below charm-threshold-region resonances; below open-bottom), registered in advance so any future confrontation is non-cherry-picked. Tests color counting × charge quantization (both ⚙️ STRUCT-CLOSED) at the RATE level — the first S-matrix-normalization test of the Cl(6) charge assignment. Gate: vertex (LSZ). |
| σ_T (Thomson cross section) | 6.652 458 7051(62) × 10⁻²⁹ m² (CODATA 2022) | — | ❌ NOT STARTED (COMPOSITE — assess before registering) | (no file) | **EXP-A8.** ASSESSMENT AS DIRECTED: **α-derived/trivial — CONFIRMED.** CODATA computes σ_T exactly as (8π/3)(αℏ/m_e c)²; there is no independent Thomson-scattering experiment in the adjustment — its uncertainty is entirely inherited from α and m_e. RECOMMENDATION: register only as a Type-4 composite of the existing α_EM + m_e rows (adds no independent physics), OR skip. Note it rides α(0), i.e. the same Δα (α(M_Z)→α(0)) bridge that blocks R∞ Row P70 — so even the composite is Δα-gated. Counted separately in the checklist as a conditional row. |
| σ(νN) benchmark (TeV-scale neutrino-nucleon cross section) | 1.30 +0.21/−0.19 (stat) +0.39/−0.43 (syst) × SM DIS prediction, 6.3–980 TeV (IceCube, Nature 551, 596, 2017; Earth-absorption, 10 784 events) | — | ❌ NOT STARTED | (no file) | **EXP-A9.** Chosen benchmark: highest-energy measured ν-N cross section (justification: cleanest published cross-section MEASUREMENT spanning decades of energy; tests the weak vertex at momenta far above M_Z). Low-energy alternative if preferred at registration: Daya Bay IBD yield (5.91 ± 0.09) × 10⁻⁴³ cm²/fission (PRD 100, 052004, 2019). Gate: vertex (weak channel). |
| BR(H→bb̄)/BR(H→γγ) | **TO-VERIFY** — no clean citable ATLAS+CMS combined ABSOLUTE BR table exists: the Run-1 combination (JHEP 08 (2016) 045) reports BR ratios (e.g. B_bb/B_ZZ) + μ = 1.09 ± 0.11; Run-2 (ATLAS 139 fb⁻¹; CMS Nature 607, 2022) report per-channel signal strengths μ ≈ 1 at few-%–10% precision. SM reference (NOT measured): BR(bb̄) ≈ 57.7%, BR(γγ) ≈ 0.227%, ratio ≈ 254 | — | ❌ NOT STARTED | (no file) | **EXP-A10.** Registration recommendation: carry the observable as measured SIGNAL-STRENGTH ratios (μ_bb, μ_γγ) or a BR double-ratio, not absolute BRs — that is what experiment actually reports. The framework side would confront y_b (existing ✅ chain) vs the loop-induced γγ vertex — a tree-vs-loop discriminator, only reachable post-I-0. Value cell to be filled at registration from the specific chosen parameterization. |
| B(μ→eγ) | < 1.5 × 10⁻¹³ (90% CL) (MEG II, 2021–2022 data, arXiv:2504.15711, 2025) | — | ❌ NOT STARTED | (no file) | **EXP-A11.** Strongest charged-lepton-flavor-violation bound (2.4× better than MEG). BOUND row: the framework's lepton-sector structure should either force ~0 (a STRUCT-consistency pass) or predict a rate (falsifiable). Gate: vertex + PMNS machinery (all angles ✅). |
| B(b→sγ) | (3.43 ± 0.21 ± 0.07) × 10⁻⁴ (HFLAV world avg, E_γ > 1.6 GeV) | — | ❌ NOT STARTED | (no file) | **EXP-A12.** Inclusive radiative penguin — the canonical loop/FCNC probe; value mildly photon-energy-threshold-dependent (declared: 1.6 GeV). Confronts the vertex at one loop with the CKM chain (V_ts row currently 🟡 within the excl/incl data self-tension — this row would enter that same corner honestly). |

---

## CLASS B — Composite states: hadrons, nuclei, atoms (16 rows + 1 dedupe)

**Gate object (master program):** RING 3 — the I-0 vertex → Bethe–Salpeter (ran once in June) +
the ΔS binding ladder {1,2,3,4,6,13} (B1 bound-state/nucleon sector; E_bind = −κ·ΔS), then the BBN
block. The ATOMIC sub-block (EXP-B9–B14) additionally rides the Δα (α(M_Z)→α(0)) bridge — the
same named blocker as the existing R∞ Row P70 — so those rows are double-gated (vertex + Δα).
Hadron masses are the FIRST composite-spectrum targets; three of them (Q_np, g_A, B_d) are already
named as missing INPUTS by existing ledger rows (T_BBN-1, Y_p, T_BBN_D) — registering them
converts undeclared imports into tracked targets.

| Symbol | Observed | Predicted | Status | File | Notes |
|---|---|---|---|---|---|
| m_p (proton mass) | 938.272 089 43(29) MeV (CODATA 2022) | — | ❌ NOT STARTED | (no file) | **EXP-B1.** The first bound-state mass target; ~99% of the mass is confinement dynamics, not Higgs Yukawas — a qualitatively NEW class vs every closed fermion-mass row. Gate: vertex + Bethe–Salpeter + ΔS ladder. |
| m_n (neutron mass) | 939.565 421 94(48) MeV (CODATA 2022) | — | ❌ NOT STARTED | (no file) | **EXP-B2.** Companion to EXP-B1; udd vs uud under the same binding object. |
| m_n − m_p (= Q_np, nucleon splitting) | 1.293 332 51(38) MeV (CODATA 2022, derived) | — | ❌ NOT STARTED | (no file) | **EXP-B3.** THE named missing input of the BBN block — the existing T_BBN-1 and Y_p rows both cite absent Q_np as their blocker; this row makes it a tracked target. Physics content: EM self-energy (≈ −0.76 ± 0.30 MeV) vs strong isospin (m_d − m_u) of opposite sign — the measured total is the target, the decomposition is theory. Feeds τ_n (EXP-A3) and Y_p. |
| m_π± (charged pion mass) | 139.570 61 ± 0.000 24 MeV (PDG 2024) | — | ❌ NOT STARTED | (no file) | **EXP-B4.** Lightest bound state = the pseudo-Goldstone of chiral breaking; the ΔS ladder's first rung candidate. (Newer ~1 ppm single measurement 139.570 21 ± 0.000 14 MeV is consistent; PDG avg is the registered target.) |
| m_π0 (neutral pion mass) | 134.9768 ± 0.0005 MeV (PDG 2024) | — | ❌ NOT STARTED | (no file) | **EXP-B5.** With EXP-B4: the π±−π⁰ splitting (4.5936 MeV) is pure-EM — a second decomposition test riding the same vertex. |
| f_π (pion decay constant) | f_π± = 130.2 ± 0.8 MeV (FLAG 2024, N_f = 2+1+1, arXiv:2411.04268; PDG normalization). ChPT convention: F_π = f_π/√2 = 92.1 ± 0.6 MeV | — | ❌ NOT STARTED | (no file) | **EXP-B6.** ⚠ CONVENTION FLAG AS DIRECTED: two conventions differing by exactly √2 circulate (130-MeV normalization enters the π→μν rate formula and is PDG's; 92-MeV is ChPT/Gasser–Leutwyler). Any framework confrontation MUST declare its convention at pre-registration or the √2 becomes a free factor. Feeds τ_π± (EXP-A5). |
| m_K (kaon masses) | m_K± = 493.677 ± 0.013 MeV; m_K0 = 497.611 ± 0.013 MeV (PDG 2024, flavor-eigenstate) | — | ❌ NOT STARTED | (no file) | **EXP-B7.** First strange-quark-carrying bound state (ΔS-ladder second rung candidate). Note: K_L mass eigenstate differs slightly (≈ 497.978 MeV, CP mixing); the flavor-eigenstate value is the registered target. |
| Λ_QCD (MS̄ scheme) | Λ_MS̄^(5) ≈ 210 +34/−30 MeV (n_f = 5; from PDG 2024 α_s(M_Z) = 0.1180 ± 0.0009 via 4-loop running). n_f = 4: ≈ 292 ± 31 MeV; n_f = 3: ≈ 332–342 MeV (FLAG-class) | — | ❌ NOT STARTED (COMPOSITE — assess before registering) | (no file) | **EXP-B8.** ⚠ SCHEME-DEPENDENCE FLAG AS DIRECTED: value nearly doubles from n_f = 5 to n_f = 3; never quote without scheme + n_f. ASSESSMENT: Λ_MS̄ is DERIVED from α_s(M_Z) by running — and the ledger already closes α_s(M_Z) ✅ (−0.13σ). So this row is plausibly a Type-4 composite of an EXISTING ✅ row, not new physics. RECOMMENDATION: register as composite-of-α_s (cheap, Ring 1-style wiring) OR skip as redundant; architect's call. Counted as conditional in the checklist. |
| r_p (proton charge radius) | 0.840 75(64) fm (CODATA 2022) | — | ❌ NOT STARTED | (no file) | **EXP-B9.** PUZZLE STATUS (honest): the 2010 muonic-H measurement (0.841 84(67) fm, Pohl et al., Nature) sat ~7σ below the old electronic/CODATA 0.8768 fm; subsequent electronic-H measurements (Beyer 2017 2S–4P; Bezginov 2019 FOSOF, Science 365, 1007 → 0.833(10) fm) converged to the SMALL value, which CODATA 2022 now adopts. Largely resolved in favor of the small radius — but NOT universally closed: CODATA's own muonic-excluded subset fit gives 0.8529(43) fm (~2.8σ residual internal tension). First structure-observable (form factor, not mass) — gate: vertex + bound-state wavefunction. |
| H 1S–2S transition | 2 466 061 413 187 035(10) Hz (4.2 × 10⁻¹⁵ fractional; Parthey et al., PRL 107, 203001, 2011, MPQ) | — | ❌ NOT STARTED (Δα-gated) | (no file) | **EXP-B10.** The most precisely measured optical transition; QED bound-state anchor. Double-gated: vertex + the Δα bridge (needs atomic-frame α(0) — same blocker as R∞ Row P70; a framework value here without derived Δα would be a Clause-9 smuggle, per the R∞ precedent). |
| Lamb shift (H, n=2) | Classic full 2S₁/₂–2P₁/₂: 1057.845(9) MHz (Lundeen & Pipkin, 1981). Modern hyperfine-resolved 2S₁/₂(F=0)→2P₁/₂(F=1): 909.8717 ± 0.0032 MHz (Bezginov et al., Science 365, 1007, 2019) | — | ❌ NOT STARTED (Δα-gated) | (no file) | **EXP-B11.** THE historic pure-QED (vacuum-fluctuation) observable — zero in tree-level Dirac theory; the cleanest "the vertex's loop is real" target. ⚠ Two citable numbers exist for different sub-intervals — do not conflate; declare which at pre-registration. Same Δα gate as EXP-B10. |
| 21 cm hyperfine (H ground state) | 1 420 405 751.768 Hz (~10⁻¹² relative; Hellwig et al., IEEE Trans. Instrum. Meas. 19, 200, 1970 — H-maser metrology) | — | ❌ NOT STARTED (Δα-gated) | (no file) | **EXP-B12.** Spin-spin (proton magnetic moment × electron) bound-state splitting; brings the NUCLEAR magnetic moment into scope — strictly harder than EXP-B10/B11 (needs μ_p, i.e. nucleon structure, on top of Δα). Also the observable underlying any future 21-cm cosmology confrontation. |
| Positronium 1S–2S (1³S₁–2³S₁) | 1 233 607 216.4 ± 3.2 MHz (Fee, Chu, Mills et al., PRA 48, 192, 1993; 2024 Rydberg-method cross-check 1 233 607 210.5 ± 49.6 MHz consistent, arXiv:2407.02443) | — | ❌ NOT STARTED (Δα-gated) | (no file) | **EXP-B13.** Pure-lepton atom — NO nucleon-structure contamination, so it isolates the vertex + Δα chain with none of EXP-B12's hadronic load. The cleanest bound-state QED target the framework could confront first. |
| Positronium lifetimes (p-Ps, o-Ps) | p-Ps: ≈ 125 ps (Al-Ramadhan & Gidley, PRL 72, 1632, 1994). o-Ps: Γ = 7.0404(10)(8) μs⁻¹ ⟺ τ ≈ 142.05 ns (Vallery, Zitzewitz, Gidley, PRL 90, 203402, 2003) | — | ❌ NOT STARTED | (no file) | **EXP-B14.** Annihilation RATES (2γ vs 3γ) — S-matrix, not spectroscopy, so NOT Δα-gated in the same way (rates carry α powers explicitly). HONEST HISTORY: the 1980s–90s "o-Ps lifetime puzzle" (5–9σ vs QED) was RESOLVED in 2003 as a thermalization systematic — cautionary precedent for treating multi-σ anomalies as physics. |
| B_d (deuteron binding energy) | 2.224 566 MeV (from S(d) = 2.388 170 07(42) × 10⁻³ u, ~1.8 × 10⁻⁷ relative; Kessler et al. Bragg-spectrometer n+p→d+γ, ~1999) | — | ❌ NOT STARTED | (no file) | **EXP-B15.** The first NUCLEAR binding target and the named external input of the existing T_BBN_D row (§8) — registering converts that import into a target. The natural first confrontation for E_bind = −κ·ΔS beyond single hadrons (deuterium bottleneck → D/H). ⚠ Rounding variants (2.224 575 / 2.224 644) circulate from different measurement generations + u→MeV conversions; the registered value follows the cited crystal-spectrometer result. |
| g_A (nucleon axial coupling) | λ = g_A/g_V = 1.2754 ± 0.0013 (PDG 2024; scale factor 2.7). Most precise single: 1.276 41 ± 0.000 56 (PERKEO III); modern combined: 1.276 25 ± 0.000 50 | — | ❌ NOT STARTED | (no file) | **EXP-B16.** DEDUPE CHECKED: g_A appears in the ledger ONLY as a cited missing input inside the Y_p row (§9) — no row of its own; this registers it. ⚠ PDG's error is inflated ×2.7 by historic scatter; the modern beta-asymmetry generation (PERKEO III/UCNA) is mutually consistent and tighter — declare which target at pre-registration. Feeds τ_n (EXP-A3) and the BBN block. |

**DEDUPE (no new row):**
- **R_∞ (Rydberg)** — already present as **Row P70** (SM gauge table, ❌; moved out of `predictions/`
  2026-05-28 over the Δα blocker). ANNOTATION SUGGESTION only: its observed cell carries the CODATA
  2018-era value 1.097 373 156 8160(21) × 10⁷ m⁻¹; update to **CODATA 2022:
  1.097 373 156 8157(12) × 10⁷ m⁻¹** (arXiv:2409.03787). No status change.

---

## CLASS C — Dark sector (3 rows + 1 dedupe)

**Gate object (master program):** RING 3 — the vertex applied to the FLAT BAND (the DM mass = the
flat band's gap under the vertex); the band is already proven immobile + B-inert, and the ledger's
Ω_DM/Ω_m row carries the structural claims "zero direct-detection (STRUCTURAL), zero
self-interaction (CONSISTENCY-OBSERVATION)". So for this class the experimental NULLS are the
framework's FRIENDS: the plausible framework output is a falsifiable null (direct detection) plus
a mass (falsifiable by any future positive detection at a different mass).

| Symbol | Observed | Predicted | Status | File | Notes |
|---|---|---|---|---|---|
| m_DM (dark-matter particle mass) | **NO measured value exists** (observationally unconstrained). Strongest exclusion reach: LZ 2024 excludes σ_SI down to 2.2 × 10⁻⁴⁸ cm² at 40 GeV/c² (90% CL; arXiv:2410.17036); XENONnT 2025: 1.7 × 10⁻⁴⁷ cm² at 30 GeV/c² (arXiv:2502.18005) | — | ❌ NOT STARTED (PREDICTION-ONLY) | (no file) | **EXP-C1.** STATUS AS DIRECTED: PREDICTION-ONLY — there is no measured target; the row exists so that a future framework value (the flat band's gap under the vertex) is REGISTERED BEFORE any comparison. Falsification logic: (a) if the framework predicts a mass + nonzero coupling inside the excluded region, it dies; (b) if it predicts a gauge-decoupled band (per the existing structural claims), the LZ/XENONnT nulls are CONFIRMATIONS; (c) any future positive detection at an incompatible mass falsifies. Note the "peak sensitivity ~40 GeV" is detector kinematics, not physics. |
| σ_SI (spin-independent DM-nucleon cross section) | Upper limits (90% CL, no signal): 2.2 × 10⁻⁴⁸ cm² at 40 GeV/c² (LZ, 4.2 t·yr, arXiv:2410.17036, 2024 — world-leading; median sensitivity 5.1 × 10⁻⁴⁸); 1.7 × 10⁻⁴⁷ cm² at 30 GeV/c² (XENONnT, 3.1 t·yr, arXiv:2502.18005, 2025) | — | ❌ NOT STARTED (BOUND row) | (no file) | **EXP-C2.** The ledger's dark-sector theorem already claims zero direct-detection STRUCTURALLY (Ω_DM/Ω_m row notes, §§9-10 multi-axial doc) — this row registers the measured bound that claim must live against, and sharpens under every future exposure. A structural-null prediction is CONFIRMED by these limits and falsified by any robust positive signal. |
| σ/m (DM self-interaction, Bullet Cluster) | < 1.25 cm²/g (Randall, Markevitch, Clowe et al., ApJ 679, 1173, 2008; common shorthand "≲ 1 cm²/g". Earlier offset-only bound: < 5 cm²/g, Markevitch 2004) | — | ❌ NOT STARTED (BOUND row) | (no file) | **EXP-C3.** Confronts the ledger's "zero self-interaction (CONSISTENCY-OBSERVATION — honest downgrade)" claim. ⚠ Kahlhoefer et al. 2016 (arXiv:1605.04307) argue the 2008 constraint is somewhat optimistic/model-dependent (merger geometry) — treat the bound as O(1) cm²/g, not a precision number. |

**DEDUPE (no new row):**
- **Ω_DM h² (relic density)** — already present as **Ω_c h²** (Cosmology §1, ⏳ CANDIDATE,
  composite of Ω_DM × h², observed cell already carries Planck 0.1200 ± 0.0012). Verified against
  Planck 2018 VI (arXiv:1807.06209, TT,TE,EE+lowE+lensing) — the existing cell is correct and
  current. No new row; the pending Type-4 lint is Ring 1 work already scheduled.

---

## CLASS D — Strong dynamics / confinement (5 rows)

**Gate object (master program):** RING 4 — confinement dynamics: the twisted-zeta / one-form
order parameter (Artin–Ihara import) + the vertex. May terminate FENCED; the holonomy-triviality
theorem (192/192 cover-closed cycles = +I) already localizes confinement to the finite-k/
non-vacuum/tick sector, so these rows have a NAMED home even before a number exists. Note: these
targets are lattice-QCD determinations, not collider measurements — the "observed" column is the
lattice literature's number, flagged per row.

| Symbol | Observed | Predicted | Status | File | Notes |
|---|---|---|---|---|---|
| √σ (string tension) | ≈ 440 MeV (quenched lattice CONVENTION via Sommer scale r₀ = 0.5 fm, r₀√σ ≈ 1.65; Necco & Sommer, Nucl. Phys. B 622, 328, 2002) | — | ❌ NOT STARTED | (no file) | **EXP-D1.** ⚠ AS DIRECTED: this is a lattice scale-setting convention, NOT a collider measurement — the value shifts ±10–20 MeV with the scale-setting scheme. Register with the convention declared (Sommer r₀). Framework target: the linear coefficient of the static potential from the twisted-zeta order parameter. |
| T_c (deconfinement / chiral crossover) | 156.5 ± 1.5 MeV (HotQCD, Bazavov et al., PLB 795, 15, 2019, arXiv:1812.08235; chiral-susceptibility peak, continuum, physical masses). Wuppertal-Budapest concurs: ≈ 155(3)(3) MeV | — | ❌ NOT STARTED | (no file) | **EXP-D2.** ⚠ HONEST DEFINITION CAVEAT: this is a smooth CROSSOVER, not a phase transition — "T_c" varies ~150–160 MeV with the defining observable. Any framework confrontation must declare which definition it computes. Heavy-ion phenomenology corroborates the scale (chemical freeze-out). |
| m(0⁺⁺) (lightest glueball) | LATTICE PREDICTION: 1730 ± 50 ± 80 MeV (quenched; Morningstar & Peardon, PRD 60, 034509, 1999). EXPERIMENT: **no glueball has ever been unambiguously identified** — f₀(1500)/f₀(1710) are candidates in mixing scenarios, no consensus | — | ❌ NOT STARTED | (no file) | **EXP-D3.** ⚠ HONEST STATUS AS DIRECTED: the "observed" value is itself a THEORY number (lattice, quenched; unquenching shifts it via qq̄ mixing) — this row is a lattice-vs-framework cross-check, not an experimental confrontation, and must be labeled as such if registered. A pure-gauge bound state = the cleanest test of the confinement object with zero fermion-sector load. |
| m_η′ (eta-prime mass) | 957.78 ± 0.06 MeV (PDG, η′(958); CLEO precision 957.793 ± 0.054 ± 0.036 MeV) | — | ❌ NOT STARTED | (no file) | **EXP-D4.** THE U(1)_A anomaly row (as directed): the η′ is anomalously heavy vs the Goldstone octet; the Witten–Veneziano mechanism prices that mass in topological susceptibility (EXP-D5). Unlike EXP-D3 this IS a precisely measured experimental number. Framework-side: the anomaly channel + topology of the confinement object. |
| χ_top (topological susceptibility) | χ^(1/4) = 191 ± 5 MeV (quenched SU(3), continuum; Del Debbio, Giusti & Pica, PRL 94, 032003, 2005). Unquenched physical-point determinations spread ~175–200 MeV | — | ❌ NOT STARTED | (no file) | **EXP-D5.** Vacuum topological-charge fluctuation ⟨Q²⟩/V — the Witten–Veneziano input for EXP-D4; consistency pair {D4, D5} tests the anomaly channel twice. ⚠ Quenched-vs-unquenched and T-dependence caveats; the T=0 quenched value is the declared benchmark. Adjacent: the existing θ_QCD = 0 ✅ row lives in the same topological sector. |

---

## CLASS E — Horizon / gravity (4 rows)

**Gate object (master program):** RING 4 — G's 2π / the horizon coordinate: the crossed-product
(observer-clock algebra) or an honest limit-of-lattice-math fence; Class E rides the same object
(the MG-1d OPEN-MISS-AT-2π and its incomplete equation = the emergent local Unruh/BW temperature
→ ML-1). EXP-E1/E2 are STRUCT-TARGETS (coefficient checks, not measurements); EXP-E3/E4 are
measured confrontations of the emergent-metric layer.

| Symbol | Observed | Predicted | Status | File | Notes |
|---|---|---|---|---|---|
| S_BH coefficient (horizon entropy) | Target FORM: S = A/(4 G ℏ) — coefficient exactly 1/4. Not a measurement: no observed black-hole entropy exists; the "observed" content is the theoretical consistency requirement (Hawking area law / first law) | — | ❌ NOT STARTED (STRUCT-TARGET) | (no file) | **EXP-E1.** STATUS AS DIRECTED: STRUCT-TARGET — a structural check, not a data confrontation. The framework must reproduce the 1/4 from its own horizon coordinate (the crossed-product object); getting 1/(4·2π) or similar would be a sharp parameter-free miss of exactly the MG-1d class. PASS/FAIL is exact, no σ. |
| T_H coefficient (Hawking/Unruh temperature) | Target FORM: T = ℏκ_surface/(2πc k_B) — the 2π is the Rindler/KMS periodicity. Not measured (no observed Hawking flux); analogue-gravity observations (e.g. Steinhauer BEC horizons) corroborate the form only | — | ❌ NOT STARTED (STRUCT-TARGET) | (no file) | **EXP-E2.** STRUCT-TARGET, same object as EXP-E1 — and THE named incomplete equation of MG-1d (the emergent LOCAL Unruh/BW temperature is exactly what G's open 2π miss is waiting on). This row is where that incomplete equation gets confronted when the crossed-product build lands. PASS/FAIL exact, no σ. |
| γ − 1 (PPN light-bending/Shapiro parameter) | (2.1 ± 2.3) × 10⁻⁵ (Cassini Doppler/Shapiro delay; Bertotti, Iess & Tortora, Nature 425, 374, 2003). Still the best direct bound as of 2026 | — | ❌ NOT STARTED | (no file) | **EXP-E3.** GR predicts γ = 1 exactly; the measured bound confronts the framework's emergent metric (ML-1 layer): if the emergent Lorentz/metric object deviates from GR's space-curvature coefficient at > 10⁻⁵, this row falsifies it. A cheap STRUCT-consistency pass if the emergent metric is exactly GR-form at this order — but that must be SHOWN, not assumed. |
| \|c_gw/c − 1\| (GW propagation speed) | −3 × 10⁻¹⁵ ≤ (v_GW − c)/c ≤ +7 × 10⁻¹⁶ (GW170817 + GRB 170817A, 1.74 s delay over ~40 Mpc; Abbott et al., ApJL 848, L13, 2017) | — | ❌ NOT STARTED | (no file) | **EXP-E4.** The tensor sector must propagate on the SAME emergent cone as the photon sector to ~10⁻¹⁵. Framework-relevant: the substrate's dim-5/dim-6 LIV rows (η₅ = 0 ✅, η_lattice = 1/12 ✅ below sensitivity) cover the photon side; this row demands the GRAVITON side match — a genuinely new structural constraint on the Layer-3 metric object. |

---

## CLASS F — Neutrino nature (2 rows + 1 dedupe)

**Gate object (master program):** RING 1 (m_ββ — THE HARVEST: ingredients fully certified — PMNS
angles ✅, Majorana phases α₂₁ = 162.39°/α₃₁ = 324.78° ✅, masses m_ν1 = 0/m_ν2/m_ν3; the harvest
station is computing the framework value in parallel — THIS ROW CARRIES THE MEASURED WINDOW ONLY).
N_eff^BBN rides the BBN block (Ring 3, species-lift gate).

| Symbol | Observed | Predicted | Status | File | Notes |
|---|---|---|---|---|---|
| m_ββ (effective Majorana mass, 0νββ) | < 28–122 meV (90% CL; KamLAND-Zen 800 full dataset, T₁/₂ > 3.8 × 10²⁶ yr in ¹³⁶Xe, arXiv:2406.11438, PRL 2024/25 — range is nuclear-matrix-element spread, not experimental error). Corroborating: LEGEND-200+GERDA+MJD T₁/₂ > 1.9 × 10²⁶ yr ⟹ < 75–200 meV (arXiv:2505.10440, 2025). PROJECTED REACH: nEXO ≈ 6–18 meV; LEGEND-1000 T₁/₂ ~ 10²⁸ yr ⟹ ≈ 10–20 meV | — | ⏳ CANDIDATE | (no file) | **EXP-F1.** The master program's own named NEW ROW (Ring 1): m_ββ = \|Σ U²_ei m_i\| from certified PMNS + Majorana phases + masses — the first native number in the neutrino-nature sector, being computed by the harvest station IN PARALLEL with this draft; per directive this row registers the measured WINDOW only, no predicted value here. DECISION STRUCTURE: with m_ν1 = 0 (normal ordering, framework-forced) the generic NO expectation is m_ββ ≈ 1–4 meV — BELOW even nEXO/LEGEND-1000 reach — so the likely confrontation is "framework predicts non-observation by nEXO"; a framework value INSIDE the KamLAND-Zen window would instead be immediately hunted. Either way falsifiable; no 0νββ signal exists anywhere to date. |
| N_eff^BBN (relativistic species count, BBN epoch) | 2.976 ± 0.093 (BBN-only: primordial D/H + Y_p global fit, no CMB input; Fields/Olive-class analyses, e.g. 2024 BBN update arXiv:2401.15054 vicinity) | — | ❌ NOT STARTED | (no file) | **EXP-F2.** THE BBN-vs-CMB consistency row (as directed): the SAME quantity at ~1 s must match its value at recombination. The ledger already carries N_eff (CMB) ✅ (§8: Planck 2.99 ± 0.17 vs framework 3 exactly); this sibling registers the independent EARLY-epoch determination (consistent: 2.976 vs 2.99). Framework-side it is NOT free: the coasting-BBN network (species-lift gate) must reproduce it at its own epoch — a cross-epoch structural test the CMB row alone cannot provide. |

**DEDUPE (no new row):**
- **Σm_ν (cosmological neutrino-mass bound)** — already present (Cosmology §2, ⏳ CANDIDATE,
  composite m_ν1+m_ν2+m_ν3). ANNOTATION SUGGESTION: its observed cell ("< 0.12 eV Planck+BAO") is
  now stale — update to the current landscape, stated honestly: DESI DR1 2024 + CMB gives
  < 0.072 eV (95%, arXiv:2404.03002); DESI DR2 2025 + CMB gives < 0.0642 eV (95%, ΛCDM,
  arXiv:2503.14744) — but the bound is UNSTABLE: it relaxes to ~0.10–0.12 eV under alternative
  Planck likelihoods (HiLLiPoP) or +SNe (arXiv:2406.14554), sits in 2.7–4σ tension with the
  oscillation floor (Σ ≳ 0.059 eV, NO), mildly prefers unphysical negative effective mass in
  ΛCDM, and eases to < 0.163 eV in w₀wₐCDM. The framework's composite (m_ν2+m_ν3 ≈ 0.059 eV with
  m_ν1 = 0) sits exactly at the oscillation floor — i.e. INSIDE the contested zone: the DESI
  tightening is a live, honest falsification pressure on this existing row and should be noted
  there.

---

## CLASS G — CMB fine structure (5 rows + 3 dedupes)

**Gate object (master program):** RING 3 — the acoustic block (vertex → RPA sound → the dormant
c_s² = 1/3 two-routes confront → θ_*, r_*) for EXP-G1–G3; the growth/response block for EXP-G4;
the BBN block for EXP-G5. These rows EXTEND the existing §6 acoustic cluster (θ_*, r_*, r_drag,
100θ_MC all already registered ❌ L6-blocked → now re-gated on the acoustic build): they register
the FINER structure (peak morphology, damping, lensing smoothing, helium) so the eventual acoustic
build is confronted across the full peak structure, not just the first-peak scale.

| Symbol | Observed | Predicted | Status | File | Notes |
|---|---|---|---|---|---|
| ℓ₁ (first acoustic peak position, TT) | ℓ₁ = 220.0 ± 0.5, height D_ℓ = 5717 ± 35 μK² (Planck 2015 XI, arXiv:1507.02704, Table E.2 — the peak table was not re-fit in the 2018 release; primary-verified) | — | ❌ NOT STARTED | (no file) | **EXP-G1.** The cleanest single acoustic-morphology number. Chosen parameterization (as directed, "cleanest Planck reports"): the Planck extrema table (position + height), NOT a re-derived statistic. Depends on the same acoustic build as the existing θ_* row (§6) but adds the HEIGHT (potential-envelope/driving physics) which θ_* alone does not constrain. |
| Peak height ratios H₂/H₁, H₃/H₁ (TT) | H₂/H₁ = 0.4516 ± 0.0034; H₃/H₁ = 0.4413 ± 0.0032 (DERIVED from Planck 2015 XI Table E.2 primary heights: pk2 = 2582 ± 11 μK² at ℓ = 537.5 ± 0.7; pk3 = 2523 ± 10 μK² at ℓ = 810.8 ± 0.7; uncorrelated-error propagation) | — | ❌ NOT STARTED | (no file) | **EXP-G2.** The classic baryon-loading (H₂/H₁) and matter-density (H₃/H₁) diagnostics — these are the rows that would confront the framework's Ω_b/Ω_c chain THROUGH the acoustic physics rather than through the ΛCDM fit, once the acoustic build exists. ⚠ Ratio values are computed from the primary table (marked DERIVED), not quoted by Planck as ratios; re-derive at registration if correlated errors matter. |
| k_D (photon diffusion / damping scale) | k_D = 0.140 90 ± 0.000 32 Mpc⁻¹ (Planck 2018 VI, arXiv:1807.06209, Table 2, TT,TE,EE+lowE+lensing; primary-verified). Angular form: 100θ_D = 0.160 76 ± 0.000 17 (PLA baseline chains table, base_plikHM_TTTEEE_lowl_lowE_lensing) | — | ❌ NOT STARTED | (no file) | **EXP-G3.** The damping-tail row (as directed): Silk diffusion cuts the peaks exponentially; it tests the RECOMBINATION-WIDTH + photon-mean-free-path physics — a different microphysical lever than θ_* (geometry). ⚠ Note: an often-quoted "100θ_D ≈ 0.1614" is the PLANCK 2013 value; the 2018 chains give 0.16076 — the draft registers 2018. Y_p^CMB (EXP-G5) is measured largely THROUGH this damping tail — the two rows are correlated and should be confronted jointly. |
| A_L (lensing consistency amplitude) | Planck 2018: A_L = 1.180 ± 0.065 (~2.8σ above the ΛCDM/GR value A_L ≡ 1; arXiv:1807.06209). ACT DR6: A_lens = 1.013 ± 0.023 — fully consistent with 1 | — | ❌ NOT STARTED | (no file) | **EXP-G4.** ⚠ ANOMALY STATUS (honest, as directed): the Planck-only TT preference for excess peak-smoothing is NOT confirmed by ACT DR6's independent measurement, and a 2024 re-analysis (arXiv:2310.03127, ApJ) localizes it near the ecliptic/217 GHz — current best reading: likely Planck systematic, not physics. Registered anyway because the framework's growth/response block predicts the lensing smoothing INDEPENDENTLY of the Planck-vs-ACT dispute; the target for a correct theory is A_L = 1 exactly (any framework deviation is a falsifiable claim, not an anomaly-chasing fit). |
| Y_p^CMB (helium fraction from CMB damping tail) | 0.239 +0.024/−0.025 (95% CL; Planck 2018 VI Table 4, free-Y_P one-parameter extension, TT,TE,EE+lowE+lensing; primary-verified. +BAO: 0.242 +0.023/−0.024; with N_eff also free: 0.247 +0.034/−0.036) | — | ❌ NOT STARTED | (no file) | **EXP-G5.** TWO-ROWS-OR-ONE (directive question) — RECOMMENDATION: **two rows.** The existing Y_p row (§9, ❌, Aver 2020 spectroscopic 0.245 ± 0.003) is an ASTROPHYSICAL measurement of primordial helium; this row is the CMB-inferred value via the damping tail — independent channel, independent systematics, both confronting the same framework object (the coasting-BBN network / species-lift gate). Keeping them separate preserves the cross-channel consistency test (0.239 CMB vs 0.245 spectroscopic — currently consistent) and prevents a single-row value swap from hiding a channel discrepancy. NOTE: Planck's BASELINE analyses do not fit Y_P (they impose BBN consistency ≈ 0.2454); only the free-Y_P extension is a measurement — the row must cite Table 4, never the baseline. |

**DEDUPES (no new rows):**
- **r (tensor-to-scalar ratio)** — already present as **Row P26** (Cosmology §1, ❌). ANNOTATION
  SUGGESTION: the bound is unchanged — r₀.₀₅ < 0.036 (95% CL, σ(r) = 0.009) remains the latest
  published (BK18, data through 2018; reprised in arXiv:2405.19469, 2024; no tighter bound
  published; σ(r) ≲ 0.003 projected via 2027 data). The row's "(BICEP/Keck 2023)" label should read
  "BK18 (2021), current as of 2025". Ring 4 gate (primordial-tensor mechanism, unbuilt) is named
  at the master program — consistent with the row's existing blocker note.
- **τ_reion** — already present (Cosmology §1, ❌ NOT STARTED, framework-external, 0.054 ± 0.007
  Planck). No new row; the declared-external disposition is Ring EXTERNAL in the master program —
  no change proposed.
- **S_8** — already present (Cosmology §5, ❌, 0.832 ± 0.013 Planck, inherits σ_8's wall — now
  re-gated on the growth block per the master program). No new row.

---

## REGISTRATION CHECKLIST

**Total new rows proposed: 47**

| Class | New rows | Of which conditional / special-status |
|---|---|---|
| A — S-matrix | 12 (EXP-A1…A12) | A4 τ_μ CALIBRATION-ROUND-TRIP RISK (recommend audit gate before scoring); A8 σ_T COMPOSITE-assessment (register as Type-4 of α_EM+m_e or skip — confirmed α-derived); A10 Higgs BR ratio carries TO-VERIFY on the measured value |
| B — Composites | 16 (EXP-B1…B16) | B8 Λ_QCD COMPOSITE-assessment (derivable from the EXISTING ✅ α_s(M_Z) row — register as composite or skip); B10–B13 double-gated (vertex + Δα bridge, per R∞ Row P70 precedent) |
| C — Dark | 3 (EXP-C1…C3) | C1 m_DM PREDICTION-ONLY (no measured target by construction); C2/C3 BOUND rows confronting existing structural null claims |
| D — Strong dynamics | 5 (EXP-D1…D5) | D1 √σ + D3 glueball + D5 χ_top are lattice-theory targets, not experimental measurements — label as lattice-cross-checks if registered |
| E — Horizon/gravity | 4 (EXP-E1…E4) | E1 S_BH + E2 T_H STRUCT-TARGETS (exact coefficient checks, no σ) |
| F — Neutrino nature | 2 (EXP-F1, F2) | F1 m_ββ status ⏳ CANDIDATE (Ring 1 harvest computing framework value in parallel; this row = measured window only) |
| G — CMB fine structure | 5 (EXP-G1…G5) | G2 ratios DERIVED from primary heights (re-derive with correlations at registration); G5 two-row recommendation vs existing §9 Y_p |

**Dedupes found: 6 (annotation suggestions only — no duplicate rows drafted)**
1. R_∞ → existing Row P70 (suggest CODATA 2018→2022 value refresh).
2. Ω_DM h² → existing Ω_c h² §1 ⏳ (observed cell verified correct; no change).
3. Σm_ν → existing §2 ⏳ (suggest observed-bound refresh to DESI DR1/DR2 with the full instability caveat — live falsification pressure on the framework's Σ ≈ 0.059 eV composite).
4. r → existing Row P26 (bound unchanged; label fix "2023" → "BK18 2021/current 2025").
5. τ_reion → existing §1 row (declared-external; no change).
6. S_8 → existing §5 row (no change; now gated on the growth block).

Additionally, three EXISTING-row cross-connections converted from undeclared imports to tracked
targets: Q_np (EXP-B3; cited as missing by T_BBN-1 + Y_p rows), g_A (EXP-B16; cited as missing by
the Y_p row), B_d (EXP-B15; cited as external input by the T_BBN_D row).

**TO-VERIFY items outstanding: 3**
1. **EXP-A10** — a citable MEASURED value for the Higgs BR(bb̄)/BR(γγ) parameterization: the Run-1
   ATLAS+CMS combination reports ratio observables (B_bb/B_ZZ etc.) and Run-2 papers report signal
   strengths; the specific measured number must be pulled from the chosen parameterization at
   registration (SM reference ratio ≈ 254 is theory, NOT a measured target).
2. **EXP-A1** — CODATA 2022 a_e last-two-digit transcription (the Fan et al. 2023 primary value in
   the row is verified; the CODATA cross-quote had a digit-count inconsistency in research notes —
   one-minute NIST lookup at registration).
3. **EXP-G2** — peak-height ratios are DERIVED here with uncorrelated-error propagation from the
   primary-verified Planck 2015 XI Table E.2 heights; if the ledger wants quoted-only numbers,
   register the raw heights instead (those are primary-verified).

**Verification summary:** 55+ measured values web-verified across PDG 2024/2025, CODATA 2022,
FLAG 2024, HFLAV, Planck 2018 VI (+PLA chains, +2015 XI Table E.2), ACT DR6, DESI DR1/DR2,
LZ 2024, XENONnT 2025, KamLAND-Zen 800, LEGEND-200, MEG II 2025, Fermilab g−2 final 2025,
UCNτ/beam τ_n, Cassini 2003, GW170817 (ApJL 848 L13), and named lattice-QCD papers (HotQCD 2019,
Morningstar–Peardon 1999, Del Debbio–Giusti–Pica 2005, Necco–Sommer 2002). Three items above
remain flagged rather than guessed. Nine honest-caveat structures are carried in-row (α-input
fork on a_e; WP20→WP25 theory shift on a_μ; beam-vs-bottle on τ_n; π⁰ lifetime input tension;
proton-radius residual subset tension; f_π √2 convention; Λ_QCD scheme-dependence; DESI Σm_ν
instability; A_L likely-systematic status) — none suppressed.
