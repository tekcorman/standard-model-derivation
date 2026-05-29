# One Hashimoto B, many readings — graceful unification of SM observables

**Date:** 2026-05-10 (consolidation pass).
**Status:** STRUCTURAL CONSOLIDATION. Pre-existing theorem-grade pieces
(`theorem_multiway_branch_measure.md` §7+§11, `theorem_dark_map_class2_closure.md`,
`theorem_analytical_feshbach_ramanujan_boundary.md`,
`predictions/dark_extraction_map_derivation.md`) are consolidated into a
single thesis: every SM observable is a specific reading of one operator
— the Hashimoto walker B on srs — with observer-MDL selecting the reading
rule per observable.

**Predecessors (the load-bearing theorem-grade pieces):**
- `theorem_multiway_branch_measure.md` (μ on multiway DAG; §7 Corollary 2:
  mass spectrum = closed-cycle spectral content of B; §11: CKM = inter-class
  branch moments).
- `theorem_analytical_feshbach_ramanujan_boundary.md` (Σ(h) = α₁/h at the
  Ramanujan saddle; closed-form contour integral).
- `theorem_dark_map_class2_closure.md` (4-class taxonomy of dark
  corrections; observable's quantum numbers select the class).
- `predictions/dark_extraction_map_derivation.md` (Class 1 amplitude,
  Class 2 mass², Class 3 edge-local, Class 4 direct h-functional).
- `theorem_ytau_corollary.md` (y_τ as Class-2 reading of the Ramanujan
  saddle).
- `theorem_substrate_pati_salam_conservation.md` (PS embedding gives
  character / representation readings).

This document does NOT introduce new derivation. It NAMES the unifying
form and catalogs every SM observable's (fiber, walk-class, reading-class,
mode-pair) tuple.

---

## 1. The unifying form

Every Standard Model observable O takes the form

> **O = Reading[reading-class] (B (fiber) , walk-class , mode-pair , μ)**

where:
- **B** = Hashimoto walker on srs, 2|E| = 12-dimensional per primitive cell.
- **fiber** = Bloch momentum k. Distinguished fibers: Γ (zero), P (Ramanujan
  saddle, k_P = (1/4, 1/4, 1/4)), Γ-cone, P-cone (linear-dispersion limits).
- **walk-class** = which class of NB walks on the multiway DAG contributes
  (closed cycle / NB-geometric / multi-cycle host / single-step / pair).
- **reading-class** = which observable-quantum-number selects the dark-
  extraction-map class (Class 1/2/3/4 per `dark_extraction_map_derivation.md`)
  or one of the auxiliary readings (Born / character / Bloch-Taylor / cascade).
- **mode-pair** = (A, A) for self-energy diagonal Σ_AA = α₁/h, or (A, B)
  for off-diagonal Σ_AB (Yukawa / inter-class transitions).
- **μ** = canonical multiway branch measure (`theorem_multiway_branch_measure.md`).

**Observer-MDL** selects the reading rule per observable: which class
applies, which fiber to read at, which walk-class contributes — none of
these are free, they are forced by the observable's substrate-side
quantum numbers (C_3 charge, parity, vertex symmetry).

---

## 2. Reading classes

Per `dark_extraction_map_derivation.md` §3 and Class-2 closure theorem:

| Class | Coupling | Coefficient | Triggers when |
|---|---|---|---|
| **R1 (amplitude)** | Im[Σ(h)] | √5/4 · α₁ | observable is generation-changing (off-diagonal C₃ charge ω or ω̄) |
| **R2 (mass²)** | Im²(h)/Re²(h) | (5/3) · α₁ | observable is mass-mixing diagonalization (C₃-trivial × M² perturbation) |
| **R3 (edge-local)** | bare | 1 · α₁ | observable lives at C₃-symmetric vertex (Tr(σ_x) = 0 kills enhancement) |
| **R4 (direct h-functional)** | sin(arg h), cos(arg h), arg(h)·n | structural | observable couples to walker phase (cosmic birefringence, PMNS Majorana) |
| **R5 (Born / combinatorial)** | |amp_j|² or count ratios | rational | observable is pure substrate count (CKM elements, Q_Koide, α_GUT, ε_CP) |
| **R6 (character / rep theory)** | character traces, polytope dihedrals | algebraic | observable is gauge / chirality / parity selection (sin²θ_W, δ_CP, θ_QCD) |
| **R7 (Bloch-Taylor)** | k² and k⁴ coefficients | rational | observable is dispersion / Lorentz-violation (η_lattice, D_H, v_F) |

R1, R2, R3 are the original 2026-04-26 dark-extraction map; R4, R5, R6, R7
are the auxiliary readings the framework has accumulated since.

---

## 3. Walk classes

Per `theorem_multiway_branch_measure.md` §11 + `feshbach_exponent_principle.py`:

| Walk class | Definition | Result on srs |
|---|---|---|
| **W1 (n_fixed=2)** | NB closed cycle, in/out pinned | (2/3)⁸ = 256/6561 = α₁_bare |
| **W2 (n_fixed=0)** | NB closed cycle, fully closed loop | (2/3)¹⁰ |
| **W3 (n_fixed=1)** | NB transition, one pinned edge | (2/3)⁹ |
| **W4 (geometric series)** | Σ_{n≥1} (α₁_bare)ⁿ over windings | 256/6305 = α₁* |
| **W5 (multi-cycle host)** | Σ_{m≥2} α_m / (1-α_m), L_eff = m·g − 2(m-1)·s_seam − n_fixed | V_ub = Σ (2/3)^{6m+2}/(1-(2/3)^{6m+2}) |
| **W6 (site-stabilizer)** | k* indistinguishable edge slots | 1/k* |
| **W7 (coupling-pair)** | k*² pair types per girth cycle | 9 |
| **W8 (Cl(6) Fock slots)** | 2^k* × k* | 24 |
| **W9 (Hashimoto modes)** | 2|E| directed-edge modes | 12 |
| **W10 (cycle space)** | 2(|E|-|V|)+1 marginal modes | 5 |

W1-W5 are walk-survival readings (Feshbach Exponent Principle and
extensions). W6-W10 are mode-counting readings (substrate-cell
combinatorics). All are moments of μ per the branch-measure theorem.

---

## 4. Master table — every SM observable, one line each

Each row gives the observable, its (fiber, walk-class, reading-class)
triple, and the resulting framework value. "Aggregate" fiber means the
observable is structural / count-based and not localized at a specific
Bloch fiber.

### Standard-Model gauge & masses

| Observable | Fiber | Walk-class | Reading-class | Result |
|---|---|---|---|---|
| α₁_bare | aggregate | W1 | R5 | (2/3)⁸ |
| α₁_full | aggregate | W1 | R2 | (5/3)·(2/3)⁸ |
| α_GUT | aggregate | W8 | R5 | 1/24 |
| sin²θ_W (M_unif) | aggregate | n/a | R6 | 3/8 |
| sin²θ_W (M_Z) | aggregate | RG run | R2 cascade | 0.230 |
| V_us | aggregate | W6 ⊗ W7 | R5 | 9/40 |
| V_cb | aggregate | W4 | R5 | 256/6305 |
| V_ub | aggregate | W5 | R5 | 128/32805 |
| V_cd, V_cs, V_td, V_ts, V_tb, V_ud | aggregate | unitarity | R5 inherited | (PDG) |
| J_CKM | aggregate | inherits | R5 | 3.16×10⁻⁵ |
| y_τ | P-saddle | W1 | R2 × W6² (edge-slot 1/9) | 1280/177147 |
| λ_H | P-saddle | W1 | R2 × 2 (quartic factor) | 2560/19683 |
| Q_Koide, ε_Koide, δ_Koide | P-saddle | n/a | R5 (Born on (4,2,2) C₃) | 2/3, √2, 2/9 |
| m_τ | P-saddle | y_τ × Higgs VEV | R2 cascade | 1.78 GeV |
| m_μ, m_e | P-saddle | Koide ratios | R5 | inherits |
| m_H | P-saddle | √(2λ)·v | R2 cascade | 125.6 GeV |
| M_Z, m_W | P-saddle | EW matching | R2 cascade | 91.97, 80.69 GeV |
| g_1, g_2, g_3, α_s, α_EM | aggregate | RG run | R2 cascade | (PDG ±1σ) |
| M_unif | aggregate | (32/k*^(g-1))·M_Pl | R5 | 1.985×10¹⁶ GeV |
| R_∞ | aggregate | α_EM(0)² · m_e · c / 2h | R2 cascade | 1.099×10⁷ m⁻¹ |
| θ_QCD | aggregate | Z_3 holonomy flat | R6 | 0 |
| m_t | — | — | DOWNGRADED | None |

### Neutrino sector

| Observable | Fiber | Walk-class | Reading-class | Result |
|---|---|---|---|---|
| m_ν₂, m_ν₃ | P-saddle | global formula via N_hub | R5 + cosmology | 8.86, 50.6 meV |
| R_ν = Δm²₃₁/Δm²₂₁ | aggregate | 228/7 (theorem) | R5 | 32.57 |
| θ_12 PMNS | aggregate | TBM × V_us perp rotation | R3 (edge-local) | 33.07° |
| θ_13 PMNS | aggregate | V_us_bare via Class-2 strip | R3 | 8.61° |
| θ_23 PMNS | P-saddle | mass-matrix 2×2 diag | R2 | 48.72° |
| α_21 PMNS | P-saddle | g·arg(h) winding | R4 | 162.39° |
| α_31 PMNS | P-saddle | 2g·arg(h) winding | R4 | 324.78° |
| δ_CP_PMNS | n/a | T_{B-L} identification | R6 | 180° |
| δ_CP_CKM | K_4 polytope | (-1)-eigenvector dihedral | R6 | 70.53° |

### Cosmology

| Observable | Fiber | Walk-class | Reading-class | Result |
|---|---|---|---|---|
| H_0 | aggregate | 1/(N_hub · t_Pl) | R5 + N_hub anchor | 68.18 km/s/Mpc |
| t_0 | aggregate | 1/H_0 | (cascade) | 14.34 Gyr |
| Λ_CC | aggregate | 3/N_hub² | R5 + N_hub anchor | 4.26×10⁻¹²² |
| w_DE | n/a | -1 (CC structure) | (structural) | -1 |
| Ω_DM/Ω_m | aggregate | Poisson + k* | R5 | 0.849 |
| η_B | P-saddle | ε_CP·Re(h_P)·α₁^M | R1 cascade (Sakharov) | 6.11×10⁻¹⁰ |
| A_s | aggregate | α_GUT · (2/3)¹⁰ · (M_GUT/M_Pl)² | R5 + cascade | 2.04×10⁻⁹ |
| ε_CP | aggregate | (k-2)/(k+2) Bayesian | R5 | 1/5 |
| A_hemis | aggregate | ε_CP · 1/k* | R5 | 1/15 |
| β cosmic birefringence | P-saddle | sin(arg h) · α_EM | R4 | 0.331° |

### Lorentz / dim-6 LV

| Observable | Fiber | Walk-class | Reading-class | Result |
|---|---|---|---|---|
| η_lattice (Hashimoto NB) | Γ-Bloch | Bloch-Taylor 4th-order | R7 | 1/12 |
| η_NB^H (scalar Bloch) | Γ-Bloch (scalar) | Bloch-Taylor 4th-order | R7 | 1/6 |
| D_H | Γ-Bloch (scalar) | Bloch-Taylor 2nd-order | R7 | 1/16 |
| D4_iso^H, D4_aniso^H | Γ-Bloch (scalar) | Bloch-Taylor 4th-order | R7 | -1/1024, +1/1536 |
| η_5 | n/a | parity + isotropy | R6 (selection rule) | 0 |
| v_F (Γ-cone) | Γ-cone | Bloch gradient 1st-order | R7 | 1/2 |
| v_F (P-cone) | P-cone | Bloch gradient 1st-order | R7 | √3/6 |
| screw Wigner cos β | aggregate | (k*-2)/k* | R6 | 1/3 |
| srs_cubic_moment(n) | aggregate | edge projection 2n-th moment | R5 | 1/(k* · 2^(n-1)) |

### Framework-internal

| Observable | Fiber | Walk-class | Reading-class | Result |
|---|---|---|---|---|
| Feshbach n_fixed=0/1/2 | aggregate | W2 / W3 / W1 | R5 | (2/3)¹⁰, (2/3)⁹, (2/3)⁸ |
| ξ_t | aggregate | renewal Markov eigenvalue | R5 | 1/log(6) |
| λ_toggle | aggregate | renewal Markov rate | R5 | 2/5 |
| p_toggle | aggregate | A1 axiom | (axiom) | 2 |
| e_bit | n/a | natural unit | (definitional) | 1 |
| M_Pl natural | aggregate | 8/√π | R5 | ≈ 4.51 lattice units |
| srs_E_at_P | P | adjacency Perron | (Bloch eigenvalue) | √3 |
| h_walker | P | Hashimoto Perron | (Bloch eigenvalue) | (√3+i√5)/2 |
| S_fresh, S_disconfirm | n/a | Beta(1,1), Beta(2,1) | R5 | 1, log₂(3) |

---

## 5. Net unification claim

Roughly **65 SM-relevant predictions** map cleanly to the (fiber, walk-class,
reading-class) tuple structure. Of these:

- **~30** read directly off the P-saddle Ramanujan eigenvalue h (Σ(h) = α₁/h
  + dark-class projector): all charged-fermion masses, PMNS angles, CP phases,
  η_B, β cosmic birefringence.
- **~15** read off Γ-Bloch dispersion (Bloch-Taylor coefficients): all
  Lorentz-violation coefficients, dim-5 / dim-6 LV bounds, Fermi velocities.
- **~15** read off the multiway DAG combinatorial structure (W1-W10 walk
  classes + R5 / R6 reading): all CKM elements, α_GUT, structural counts,
  Koide ratios, Feshbach exponents at all n_fixed.
- **~5** are cosmology cascade chains anchored on N_hub (G_F-derived).

The unification is **graceful** in the sense that the same operator B and
the same measure μ produce all 65, with observer-MDL selecting the reading
rule per observable's quantum numbers.

---

## 6. Honest scope — what is NOT a B-reading

### 6.1 External anchors (load-bearing inputs)
- **N_hub ≈ 8.395×10⁶⁰** — G_F-anchored cosmology cascade input. G_F is
  the framework's empirical anchor; the entire cosmology cascade (H_0, t_0,
  Λ_CC, m_ν₃ via global formula) inherits this anchor.
- **M_Pl_GeV (CODATA SI)** — anthropocentric SI conversion factor, not
  itself a substrate output.
- **MSSM RG β-coefficients** (b_1=33/5, b_2=1, b_3=-3) — Type-3 standard-QFT
  inheritance for gauge-coupling running.
- **α_EM(0) running below M_Z** — Type-3 standard QED via charged-fermion
  thresholds.

### 6.2 Standing axioms / adoptions
- **A5-mass labeling** — identification of Ramanujan eigenvalues with SM
  mass spectrum is axiomatic (substrate-structural would close it via the
  V_Ram ≅ Cl(6) Fock identification — see §7).
- **ADOPTED-B3** (Slansky T_{B-L} sign convention / chirality / lepton-quark
  labels) — predictions are (Z/2)³-invariant (Angle D verdict 2026-04-30),
  but the labeling itself is empirical anchor.

### 6.3 Identification gaps (research-level)
- **V_Ram ↔ Cl(6) Fock** (newly surfaced 2026-05-10 via F4-followup).
  P3/P4 vertex-form derivation tacitly identifies two different 8-dim
  spaces; explicit construction pending.
- **C³_gen mass operator construction** (B7.3a.v) — would unblock m_ν₁,
  Y_u/Y_d eigenbasis (Need-D-3), m_top sector.

### 6.4 Out of scope (multi-audit BLOCKED)
- **n_s, r, σ_8, native CMB C_l, r_s, θ_*** — primordial cosmology
  multi-audit-converged BLOCKED per substrate NA-4 + observer OS-1.

---

## 7. Open structural items the unification surfaces

The "one B, many readings" frame makes the framework's remaining open
research items easier to name:

1. **V_Ram ≅ Cl(6) Fock — RE-BOUNDED 2026-05-12**
. The
   C_3-equivariant iso already exists (B6 Spin(6)≅SU(4) lift; B5.3-core
   directed-edge C_3; `χ_{C₃}=(8,2,2)` both sides). The residue is the
   *interpretation* of the (4,2,2): shown intrinsically NOT alignable with the
   standard PS species labels (`U_{C₃}^S ∉ T_species`) ⇒ reading (β) (bare
   SU(4)-Cartan label), not "generation". So `⟨τ_L|γ^a h⁰_a|τ_R⟩` is ill-posed
   as a state matrix element; the from-scratch `y_τ` is the geometric derivation
   (`predictions/y_tau_derivation.md`) and P4 §6 #3 closes in that reframed
   sense — definitively: the block-trace alternative is ruled out (y_τ's
   magnitude is a girth-cycle *graph* quantity, not a quantity on the 8-dim
   fiber; the spinor carries quantum numbers only).

2. **β_dark for marginal cycle sector** — does the dark Feshbach c = 5/12
   appear as an RG fixed point on W10 (cycle space), parallel to α₁* on
   W4 (geometric series)? F7 §4.2(c) reframed.

3. **CKM permutation rigidity** (Need-D-3) — whether the joint-Feshbach
   formalism's inter-pattern spectral measure ρ_{ud}(φ) has the missing
   degree of freedom that single-pattern eigenstructure forced to permutation.

4. **Quark-sector Q_Koide extension (F3)** — does the (4, 2, 2) C_3
   isotypic structure on V_Ram have an analog at the quark pattern that
   produces m_t/m_u ~ 10⁵?

5. **Continuum-limit lift to standard Wilsonian RG** (F7 §4.3) — substrate
   β-function reducing in long-wavelength limit to standard SM β-functions.

Each is a specific research item with bounded scope, identified by its
position in the unification table.

---

## 8. Probe verification

Self-contained probe `proofs/foundations/one_hashimoto_many_readings.py`
walks the master table: for ~30 representative observables across all
reading classes, it verifies the simulator's existing prediction matches
the unified-form computation expressed via kernel + utility primitives.
Honest scope flags surface what's still anchored vs. what's a clean
B-reading.

---

## 9. Why this matters

Before this consolidation, the framework's predictions read as ~60
independent computations, each with its own derivation chain. The
consolidation makes explicit that they're not independent — they're 60
different readings of the same operator B on srs, selected by observer-MDL
rules.

**Compression:** the framework's ~60 SM-relevant predictions reduce to one
operator (B) plus one measure (μ) plus seven reading classes (R1-R7) plus
ten walk classes (W1-W10). Roughly 19 structural primitives generate
60+ predictions. Compression ratio ≈ 3:1 at the prediction level, much
higher when accounting for the universal substrate (k*, |V|, |E|, g) +
single saddle (h_walker) inputs.

**Predictability:** any observable's value can be derived in one step
from the table — pick the (fiber, walk-class, reading-class) triple, plug
in. The framework is now a substrate-derivation machine, not a pile of
sector-specific calculations.

**Falsifiability:** any observable that does NOT fit the table is a
prediction failure (or surfaces a structural item, as F4-followup did
for V_Ram ↔ Cl(6) Fock). The unification table itself becomes a tool for
discovering structural research items.

---

## Cross-references

**Theorems consolidated:**
- `theorem_multiway_branch_measure.md` (the canonical measure μ).
- `theorem_analytical_feshbach_ramanujan_boundary.md` (Σ(h) = α₁/h closed form).
- `theorem_dark_map_class2_closure.md` (4-class dark-extraction taxonomy).
- `predictions/dark_extraction_map_derivation.md` (Class 1/2/3 derivations).
- `theorem_ytau_corollary.md` (y_τ as Class-2 reading).
- `theorem_substrate_pati_salam_conservation.md` (PS rep theory readings).
- `theorem_F_inv_E_to_srs_compression.md` (substrate selection).

**Honest-negative session findings:**
- F4-followup (2026-05-10): V_Ram ≠ Cl(6) Fock identification gap.
- F7 §4.2(c) (2026-05-10): gauge β_1 leading coefficient ≠ 5/12.

**Probe deliverable:**
- `proofs/foundations/one_hashimoto_many_readings.py` (this consolidation).
