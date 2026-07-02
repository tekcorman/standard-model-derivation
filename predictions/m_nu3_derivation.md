# m_ν₃ — Heaviest light neutrino mass (normal ordering)

**Status:** **DOMINANT-CONDITIONAL** (re-graded 2026-05-18 chain audit;
was overstated as UNIQUE-THEOREM-GRADE-CONDITIONAL). The absolute m_ν3
scale rests on FOUR non-derived choices, ranked by severity:
(1) **y_ν = 1** — the Dirac-neutrino/top Yukawa set to unity by
assertion (Step 3); the framework's own admitted "single hard residue"
that retracted m_top. Load-bearing for the entire scale.
(2) **N_hub circular** — pinned to the measured G_F (v ← N_hub ← G_F).
(3) **δ⁴ engineered** — M_R's δ-power is the *charged-lepton* Koide
phase, asserted equal so it cancels v²'s δ⁴; the advertised
"δ-independence" is built-in, not derived (master doc §8
discovery-order discipline).
(4) **off-channel** — the 1/(k*·N_atoms)=1/12 normalization is the
Γ-Perron OFF-support object, but the framework's own §7.6 Ihara-map
rule places m_ν3 ON the cut (the §7.6-mandated channel gives ≈196 meV,
~4× off). See `docs/parameters/parameter_uniqueness_ledger.md` /
this session's chain audit. Clause 8 vs σ_PDG only.
The genuinely anchor-free neutrino prediction is the splitting ratio
R = 228/7 (Step "Companion"), NOT the absolute m_ν3.
**Date:** 2026-05-04 (supersedes 2026-05-02 EOD+9 ADOPTED-PS+ADOPTED-Z3 derivation).
**Companion:** `predictions/m_nu3.py`
**Reframing context:** an internal working note

## 1. Abstract

We predict the heaviest light neutrino mass as the **substrate's mean-field critical fluctuation gap** scaled by per-cell channel multiplicity:

$$m_{\nu_3} \;=\; (k^* \times N_{\text{atoms}}) \,\times\, M_{\text{Pl}} \,\times\, N_{\text{hub}}^{-1/2}$$

For srs (k* = 3, N_atoms = 4) this gives **m_ν₃ = 12 × M_Pl / √N_hub ≈ 50.57 meV**, vs observed 50.13 ± 0.20 meV (NuFIT 6.0). Deviation = +0.87% (+2.18σ_PDG; Clause 8 FAIL against σ_PDG alone).

The formula has two equivalent readings:

  **Global:** m_ν₃ = (k*·N_atoms) × M_Pl × N_hub^(-1/2)
  **Seesaw:** m_ν₃ = v² / M_R, with v = δ²·M_Pl/(√2·N^(1/4)) and M_R = δ⁴·M_Pl/(2·k*·N_atoms)

The δ⁴ in v² and M_R cancel exactly along with the 1/2. The result is **independent of the Koide phase δ** — a clean structural distinction from charged-lepton masses, which all depend on δ via the Koide formula.

This supersedes the 2026-05-02 ADOPTED-PS + ADOPTED-Z3 derivation (which used m_ν₃_bare = 0.048277 eV from Pati-Salam seesaw at M_R = (2/3)^g·M_GUT). The new derivation **uses zero adopted inputs**: all factors derive from substrate primitives (k*, N_atoms, M_Pl, N_hub) themselves theorem-grade or framework-internal.

## 2. Framework axioms invoked

- **A1** (`docs/framework/framework_axioms.md`): binary self-inverse toggle on the substrate edge alphabet.
- **A2** (MDL canonicalization): the observer retains the MDL-optimal effective theory for any observable; underpins the BZJ mean-field selection for v.
- **A5(a)** (mass clause): the substrate's spectral gap at criticality identifies with physical fermion mass.

## 3. Derivation

### Step 1 — Higgs VEV (BZJ, theorem-grade)

By `predictions/v_higgs.py` (Brézin-Zinn-Justin 1985 finite-size scaling at the MDL-selected mean-field critical point):

$$v \;=\; \frac{\delta^2 \, M_{\text{Pl}}}{\sqrt{2} \, N_{\text{hub}}^{1/4}} \,\cdot\, \left(1 - \tfrac{5}{12}\,\alpha_1/(1-\alpha_1)\right)$$

with δ = 2/9 (Wigner D¹₁₀ matrix element from `predictions/h_walker_eigenvalue.py`) and the (5/12)·α₁/(1-α₁) dark correction theorem-grade per `proofs/foundations/dark_feshbach_a2_closure.py`. **Type 4** (upstream `predictions/`).

For the leading-order derivation here, we use the bare BZJ form $v \approx \delta^2 M_{\text{Pl}}/(\sqrt{2}\,N^{1/4})$; the dark correction is a sub-leading multiplicative factor.

### Step 2 — Right-handed Majorana mass scale M_R (structural)

The substrate-level Majorana mass operator for the right-handed neutrino has the form

$$M_R \;=\; \frac{\delta^4 \, M_{\text{Pl}}}{2 \,\cdot\, k^* \cdot N_{\text{atoms}}}$$

with the four factors deriving as follows:

**δ⁴ — Wigner D bilinear (Type 2 algebra given Step 1).**

The Higgs VEV's δ² is the Wigner D¹₁₀ matrix element |⟨v₀(Γ)|ψ_H(P)⟩| (`predictions/v_higgs.py` Step 3). For the ν_R Majorana mass — a fermion BILINEAR (ν_R)^T C ν_R with two ν_R fields — both fields contribute the same Wigner D matrix element, giving |⟨v₀|ψ_νR⟩|² = δ⁴.

**1/2 — Majorana mass term coefficient (Type 3 standard QFT).**

The standard form of a Majorana mass term in the Lagrangian is

$$\mathcal{L} \;\supset\; -\tfrac{1}{2}\, M_R \, (\nu_R)^T C \nu_R + \text{h.c.}$$

with the 1/2 coefficient required to avoid double-counting of the two ν_R contractions in the same field (Peskin-Schroeder §3.4; Mohapatra-Senjanović 1980, Phys. Rev. Lett. 44, 912).

**1/(k* × N_atoms) — per-cell directed-edge Bloch normalization (Type 3 standard solid-state).**

For a coupling defined per directed edge of the substrate primitive cell, the Bloch decomposition normalizes by the total directed-edge count per cell:

$$N_{E,\text{directed}} \;=\; k^* \,\cdot\, N_{\text{atoms}} \;=\; 12 \;\;\text{for srs}$$

This is the standard Bloch-decomposition normalization (Ashcroft-Mermin §8, or any solid-state text). Each Bloch mode at any k receives 1/N_E equal weight from each directed edge.

**M_Pl — substrate-anchored Planck scale (Type 4).**

Per `predictions/G_N.py` and the G_sub Drude closure: $M_{\text{Pl}}/M_{\text{substrate}} = 8/\sqrt{\pi}$ (theorem-grade), with M_Pl externally anchored via CODATA. The dimensionless content $G_N \cdot M_{\text{Pl}}^2 = 1$ is a derived identity, not a definitional convention.

### Step 3 — Type-I seesaw (Type 3) — **LOAD-BEARING ADOPTION EXPOSED (2026-05-18 audit)**

The standard Type-I seesaw (Mohapatra-Senjanović 1980), M_R ≫ m_D:

$$m_{\nu_3} \;=\; \frac{m_D^2}{M_R} \;=\; \frac{y_\nu^2\, v^2}{M_R}$$

m_D = y_ν·v is the **Dirac** neutrino mass; y_ν is the Dirac neutrino
Yukawa. Under the Pati-Salam embedding M_D^{(ν)} = M_u^T, m_D is the
**top mass at the unification scale**, i.e. y_ν = y_t(GUT).

**This step previously read "m_D = v at leading order" — that is the
adoption y_ν = 1, dressed as a derivation via the hand-wave 'the
bilinear δ⁴ already captures the field content'. It is NOT derived.**
It is the *same* undischarged up-sector Yukawa natural-scale anchor that
the master dark-correction doc calls "the single hard residue" and that
forced the **m_top retraction (Row P38)**. The linter hard gate forbids
"at leading order" identifications without proof; the honest form keeps
y_ν explicit:

$$\boxed{\,m_{\nu_3} \;=\; y_\nu^2 \,\cdot\, \frac{v^2}{M_R}\,}$$

The framework **adopts y_ν = 1** (giving the boxed Step-4 result). With
y_ν = 1, m_ν3 = 50.57 meV (+0.87%). The observed m_ν3 corresponds to
y_ν = 0.9957 — NOT predicted. Realistic GUT top-Yukawa values
y_t(GUT) ≈ 0.5–0.7 give m_ν3 ≈ 13–25 meV (off by 2–4×). **The entire
absolute scale of m_ν3 rests on the adopted y_ν = 1; without it the
scale is free.** Only the *combination* below (and the splitting ratio
R = 228/7) is anchor-free.

### Step 4 — Algebraic simplification (Type 2, machine-verified)

Substituting Steps 1 and 2:

$$\frac{v^2}{M_R} \;=\; \frac{\delta^4 \, M_{\text{Pl}}^2 / (2\,N^{1/2})}{\delta^4 \, M_{\text{Pl}} / (2 \,k^* N_{\text{atoms}})} \;=\; (k^* \times N_{\text{atoms}}) \,\times\, \frac{M_{\text{Pl}}}{N_{\text{hub}}^{1/2}}$$

The δ⁴ factors cancel. The 1/2 factors cancel. The result is

$$\boxed{m_{\nu_3} \;=\; (k^* \times N_{\text{atoms}}) \,\times\, M_{\text{Pl}} \,\times\, N_{\text{hub}}^{-1/2}}$$

**The result is independent of δ** — a clean structural distinction between neutrinos and charged leptons (the latter all carry δ-dependent Koide hierarchies).

Verification (machine precision): $(\delta^4/4) \times k^{*\,(g-1)} = k^* \times N_{\text{atoms}}$ — i.e., the closed-walk form $M_R = 2 \cdot M_{\text{Pl}}/k^{*\,(g-1)}$ and the Wigner-bilinear form $M_R = \delta^4 M_{\text{Pl}}/(2 k^* N_{\text{atoms}})$ are exactly equivalent. See `proofs/flavor/srs_M_R_step1_structural.py` and `srs_M_R_step3_closure.py`.

## 4. Result

$$m_{\nu_3} \;=\; (k^* \times N_{\text{atoms}}) \,\times\, M_{\text{Pl}} \,\times\, N_{\text{hub}}^{-1/2}$$

Numerical evaluation (with k* = 3, N_atoms = 4, M_Pl = 1.22089 × 10¹⁹ GeV, N_hub = 8.395 × 10⁶⁰ from the adopted N_hub (value pinned via the measured G_F)):

$$m_{\nu_3} \;\approx\; 12 \,\times\, 1.22089 \times 10^{28}\,\text{eV} \,\times\, (8.395 \times 10^{60})^{-1/2} \;\approx\; 50.57 \,\text{meV}$$

Companion predictions:

- **m_ν₂** via R = 228/7 splitting (theorem-grade Ihara, `predictions/R_nu_splitting.py`): m_ν₂ = m_ν₃/√R ≈ 8.86 meV from the global formula. Note that `predictions/m_nu2.py` currently uses the older ADOPTED-PS + Feshbach chain at -0.10σ match. The two chains give different absolute m_ν₂ predictions; reconciliation requires a follow-up update of `m_nu2.py` to use the global m_ν₃ as input. The R = 228/7 splitting itself is theorem-grade and independent of which m_ν₃ chain is used.
- **m_ν₁** = 0 — ✅ **DERIVED 2026-05-21 (W44 reframe + W45 computation; THEOREM-GRADE-CONDITIONAL on A5(a) + Probe-B Re-sign-lock, NOT on Need-D-3).** The framework's Majorana mass M_R = |M_R|·h^g is a girth-ring walker holonomy; on the Hashimoto operator B(P) the trivial-C_3 |h|=1 modes carry trivial holonomy h^g=+1 (no dynamical Majorana ν_R) while the ω, ω² Ramanujan modes carry the live α_21/δ_CP phases. The substrate produces exactly 2 dynamical Majorana ν_R ⇒ rank-2 Type-I seesaw ⇒ m_ν1 ≡ 0. (Supersedes the B6-retracted "M_D(trivial)=0 at P-point" route — see `structural_residue_register.md` §R-15, an internal working note.)

## 5. Comparison with experiment

| quantity | predicted | observed (NuFIT 6.0) | deviation | σ_PDG |
|---|---|---|---|---|
| m_ν₃ | 50.57 meV | 50.13 ± 0.20 meV | +0.43 meV (+0.87%) | +2.18σ_PDG |
| m_ν₂ via global+R | 8.86 meV | 8.65 ± 0.05 meV | +0.21 meV (+2.4%) | +1.91σ_PDG |
| m_ν₂ via `predictions/m_nu2.py` (older chain) | 8.64 meV | 8.65 ± 0.05 meV | -0.01 meV (-0.1%) | -0.1σ_PDG |
| m_ν₁ | 0 | 0 (assumed; lightest) | — | — | — |
| Δm²₃₁ via global | 2.557 × 10⁻³ eV² | 2.513 × 10⁻³ eV² | +1.7% | — | — |

**Clause 8 (σ_PDG only):** σ_PDG ≈ 0.40% (NuFIT 6.0); deviation +0.87% =
+2.18σ_PDG ⇒ **FAIL** against σ_PDG alone. Structural sources of the
residual (sub-leading per-cell mixing, BZJ next-order, N_hub anchor variation
between {G_F, m_τ, R_∞} calibrations) are not absorbed into σ_PDG.

**Note on m_ν₂:** Applying R = 228/7 to the new global m_ν₃ gives m_ν₂ = 8.86 meV (+2.4% vs NuFIT 6.0). The existing `predictions/m_nu2.py` file uses the older ADOPTED-PS + Feshbach chain and matches at -0.10σ. The two chains differ at the m_ν₃ level (50.57 vs 49.33 meV); reconciliation is a follow-up. **This file (`m_nu3_derivation.md`) makes claims only about m_ν₃**; m_ν₂ predictions are deferred to `predictions/m_nu2.py` until a unified derivation is established. The R = 228/7 ratio is theorem-grade and independent of either chain.

## 6. Open questions

0. **Dark-correction sweep clarification (2026-05-15).** Master doc `../theorems/theorem_substrate_feshbach_dark_corrections_master.md` §3 (B) catalogs neutrino masses as taking the Feshbach Im(h)/|h|² = √5/4 multiplicative DC, theorem-grade per `theorem_m_nu_dark_correction_uniqueness_closure.md`. The spectral-gap reformulation in this derivation **bakes the Feshbach mechanism into the bare scale** (the spectral-gap formula IS the residue-at-h evaluation of Σ(h) = α₁·h̄/|h|²) — applying the universal-template multiplicative factor (1 − √5/4·α₁/(1−α₁)) ≈ 0.9773 on top would double-count, shifting m_ν₃ to 49.42 meV (−1.4%, −3.5σ_PDG; WORSE than current +0.87%). Family D sub-leading at the (0H+2F) Majorana vertex is +α₁²/6 ≈ +0.025%, negligible vs the N_hub anchor sensitivity. Master doc §3 (B) "Application clarification" and §5 catalog row for m_ν_3 updated 2026-05-15 to reflect this. The +0.87% Clause-8-FAIL residual is N_hub-anchor-driven (Q1 below), not a missing DC.

1. **N_hub anchor refinement.** The adopted N_hub's value is currently calibrated via the measured G_F (predictions/N_hub.py). The chi² fit across all v-anchored masses prefers N_hub ≈ 8.435 × 10⁶⁰ (m_τ-anchored), which gives m_ν₃ = 50.45 meV and deviation +0.65%. This is the dominant structural source of the residual.

2. **Sub-leading prefactor structure.** The per-cell normalization 1/(k* × N_atoms) is the leading-order Bloch decomposition. Sub-leading mode-mixing corrections at O(δ²) ≈ 5% level are bounded but not explicitly computed; they could refine the prefactor from 12 to (12 + small correction).

3. **PMNS Majorana phases (h^g).** Rows P35 (α_21 ≈ 162.39°), P36 (α_31 ≈ 324.78°). The phase structure rides on the M_R diagonal phase factor h_m^g on the C_3 ω, ω² Bloch modes (orthogonal to the C_3-trivial scale-setting direction; the two pieces don't interfere — the *scale* derivation here is phase-free and unaffected). **NOTE (2026-05-12):** the h^g phase factor is the **ADOPTED-NU-MAJ-PHASE** identification, NOT theorem-grade — a discharge was attempted and FAILED (`proofs/foundations/majorana_M_R_waterfilling.py`: A2-T loop-sum diverges; the Path-B "cardinality ↔ girth rings" route is broken — K_4 cycle-space generators have nonzero Z³ voltage). Rows P35/P36 = STRUCTURAL-DERIVATION-CONDITIONAL, not "theorem-grade per srs_hashimoto_seesaw_verify.py" (that earlier claim is corrected here). (δ_CP_PMNS / Row P34 is RETIRED on independent grounds.) The m_ν₃ *magnitude* derivation in this file is unaffected.

4. **m_ν₂ residual.** At +2.4% (+1.91σ_PDG), m_ν₂ FAILS Clause 8 against σ_PDG. A more careful treatment of the R = 228/7 splitting at next-to-leading order could refine.

## 7. References

### Framework upstream
- `predictions/v_higgs.py` + `predictions/v_higgs_derivation.md` — BZJ scaling, theorem-grade.
- `predictions/N_hub.py` + `predictions/N_hub_derivation.md` — the adopted N_hub (value calibrated via the measured G_F).
- `predictions/G_N.py` + `predictions/G_N_derivation.md` — G_sub Drude closure; M_Pl/M_substrate = 8/√π.
- `predictions/k_star.py`, `predictions/d_spatial.py`, `predictions/g_girth.py` — substrate primitives.
- `predictions/h_walker_eigenvalue.py` — δ = 2/9 from Wigner D¹.
- `predictions/R_nu_splitting.py` + `docs/parameters/R_theorem.md` — R = 228/7 (Ihara).
- `proofs/flavor/srs_hashimoto_seesaw_verify.py` — PMNS phases (h^g).
- `proofs/flavor/srs_M_R_step1_structural.py`, `step2_derivation.py`, `step3_closure.py` — structural derivation chain (this work).

### External
- Brézin, E. & Zinn-Justin, J. (1985). Finite size effects in phase transitions. *Nucl. Phys. B* **257**, 867–893.
- Peskin, M.E. & Schroeder, D.V. (1995). *An Introduction to Quantum Field Theory*. §3.4 (Majorana mass term coefficient).
- Mohapatra, R.N. & Senjanović, G. (1980). Neutrino mass and spontaneous parity nonconservation. *Phys. Rev. Lett.* **44**, 912.
- Ashcroft, N.W. & Mermin, N.D. (1976). *Solid State Physics*. §8 (Bloch decomposition normalization).
- Sunada, T. (2012). *Topological Crystallography*. Springer. Theorem 3.1 (srs uniqueness).
- NuFIT collaboration (2024). Three-flavor neutrino oscillation analysis, NuFIT 6.0. http://www.nu-fit.org.

## Audit v2 (Clause 7) status

Inherits Row 4 audit v2 closure (k* = 3 selection) per an internal working note §2.1.

- **Status:** **DOMINANT-CONDITIONAL** (2026-05-18 re-grade; prior
  UNIQUE-THEOREM-GRADE-CONDITIONAL did not disclose the y_ν=1 adoption).
- **Conditionals (ranked):** (1) **y_ν = 1** Dirac/top Yukawa adopted,
  not derived — the m_top-retracting hard residue, load-bearing for the
  entire scale; (2) N_hub G_F-circular; (3) δ⁴ cancellation engineered
  (δ-independence built-in, not derived); (4) 1/12 = Γ-Perron
  off-support channel, contradicts the framework's own §7.6 (on-cut →
  ≈196 meV); plus Row 4 (k*=3) and G_sub Drude for M_Pl.
- **Persistence-derivation of y_ν tested & STRUCTURALLY CLOSED-NEGATIVE
  (2026-05-19; computed, gate-verified):** the persistence route for
  deriving y_ν was probed
  (`proofs/foundations/nu_mass_half_sided_persistence_2026-05-19.py`,
  `y_nu_persistence_ceiling_2026-05-19.py`). Per-chirality-occupancy
  readings are a SMUGGLE (inject a quantity the observer-MDL model does
  not compress) — VOID. The framework's OWN theorem-grade
  persistence-Yukawa law (the y_τ law `α₁_full/k*²`, reproduced exactly
  as a correctness gate) has STRUCTURAL CEILING α₁_full ≈ 0.065 (all
  per-leg projections ≤ 1); every structurally-unambiguous reading at
  the Dirac-ν/top max-persistence endpoint gives y ∈ [0.0072, 1.667],
  i.e. the required y_ν ≈ 1 is **~15× ABOVE the law's ceiling**. ⇒ The
  persistence model CANNOT derive y_ν: `y_t(GUT)=1` is not a persistence
  amplitude, it is the un-suppressed natural-scale UNIT the entire
  persistence/Koide ladder is measured against (y_τ = α₁_full/k*² is
  y_τ *in units of* y_t(GUT)=1); the law presupposes the unit and cannot
  output it. This *is* the master dark doc's "single hard residue"
  (line 403). **CORRECTION of a prior dodge:** the +0.87%/+2.18σ does
  NOT "stay N_hub-gated" — it sits on the un-derived natural-scale
  Yukawa unit y_ν=1, which this derivation's OWN ranked conditionals
  list as load-bearing **#1** (N_hub is only #2). No status change;
  y_ν=1 stays the named adoption; what changed is the honest attribution
  of the residual to the irreducible Yukawa unit, not to N_hub.
- **Anchor-free content (genuine):** ONLY the splitting ratio
  R = 228/7 (Ihara, theorem-grade, independent of y_ν/N_hub/δ/channel).
- **Known structural sources of residual:** the y_ν=1-vs-0.9957 gap +
  N_hub anchor + the off-channel normalization (NOT just N_hub).
- **Six-mechanism gating** (M1-M6, Clause 7c):
  - M1 (chirality residue): inherited from Row 4 (k*=3 chirality fixed).
  - M2a (structural MDL waterline): MDL selects Curie-Weiss mean-field critical, theorem-grade per BZJ.
  - M3 (dark-sector amplitude): C_3-trivial Bloch projection at P, structurally specified via PS singlet.
  - M4 (multiway branch measure): not directly relevant (formula doesn't carry per-walk amplitudes summed over multiway).
  - M5 (Feshbach resummation): captured in v's (5/12)·α₁/(1-α₁) dark correction.
  - M6 (operator-wave spectrum): H(P)² = k*I (Ramanujan saturation) specifies the trivial-sector dimension as 2.
  - Combined: PASS for inheritance + extension to neutrino sector.

## Audit v2 (Clause 8) status

- σ_PDG = 0.40% (NuFIT 6.0 m_ν₃ uncertainty).
- Deviation = +0.87% = +2.18σ_PDG ⇒ **FAIL** against σ_PDG alone.

Clause 8 FAIL. The structural derivation is **DOMINANT-CONDITIONAL**
(NOT UNIQUE-THEOREM-GRADE): the absolute scale is set by the adopted
y_ν = 1 (Step 3) — undisclosed in the prior grade — on top of the
G_F-circular N_hub, the engineered δ⁴ cancellation, and the §7.6-
contradicted off-support normalization. The +0.87% is the y_ν=1-vs-
0.9957 gap plus these, NOT merely N_hub anchor variation. The
anchor-free, genuinely-predicted neutrino quantity is R = 228/7.
