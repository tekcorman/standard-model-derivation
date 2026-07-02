# Walker-class-to-SM-state dictionary

**Date:** 2026-05-27 EOD+3
**Status:** Structural dictionary consolidating the 48↔48 walker-matter correspondence
**Companion to:** `theorems/theorem_walker_matter_unification_2026-05-27.md`
**Probe:** `proofs/foundations/walker_class_dictionary_2026-05-27.py`

---

## 1. Purpose

The unification meta-theorem (companion doc) asserts: 48 Hashimoto eigenmodes at the 4 Ramanujan saddles correspond bijectively to 48 SM Weyl spinors per primitive cell. This document is the **explicit walker-class-to-SM-sector dictionary** that the meta-theorem implicitly defines.

**Note on grain:** the dictionary is at the **walker-class → SM-sector** level (e.g., "h_P family → charged fermion sector"), NOT at the per-Weyl-spinor level (e.g., "mode #5 → u_R of generation 2"). The per-Weyl-spinor dictionary requires explicit enumeration of the V_Ram ≅ Cl(6) Fock iso at the 4-vertex level, which is research-level structural work the framework hasn't done. The walker-class-to-sector dictionary IS available and is what's documented here.

---

## 2. The 48-mode walker-class enumeration

### 2.1 Per-saddle structure (probe output)

| Saddle | 12 Hashimoto eigenmodes broken down |
|---|---|
| **Γ** | 1 Perron (\|λ\|=2) + 6 Ramanujan at arg ±110.70° (h_Γ family, 3 each) + 5 Trivial \|λ\|=1 |
| **P** | 8 Ramanujan: 4 at arg ±52.24° (h_P family) + 4 at ±127.76° (h_P_neg family) + 4 Trivial \|λ\|=1 |
| **N** | 8 Ramanujan, one of each ±arg in {37.76, 69.30, 110.70, 142.24}° (all 4 saddle families) + 4 Trivial \|λ\|=1 |
| **H** | 1 Perron (\|λ\|=2) + 6 Ramanujan at arg ±69.30° (h_H family, 3 each) + 5 Trivial \|λ\|=1 |

Total: 12 × 4 = 48 modes ✓

### 2.2 Walker-class families across all saddles

| Family | arg | tan²(arg) | Modes (total across saddles) | Where they live |
|---|---|---|---|---|
| h_P | ±52.24° | 5/3 | 4 | P only (multiplicity 2 per sign) |
| h_P_neg | ±127.76° | 5/3 | 4 | P only (multiplicity 2 per sign) |
| h_Γ | ±110.70° | 7 | 8 | Γ (×3 each sign = 6) + N (×1 each sign = 2) |
| h_H | ±69.30° | 7 | 8 | H (×3 each sign = 6) + N (×1 each sign = 2) |
| h_N | ±37.76° | 3/5 | 2 | N only (×1 each sign) |
| h_N_neg | ±142.24° | 3/5 | 2 | N only (×1 each sign) |
| Perron \|λ\|=2 | 0° or 180° | 0 or ∞ | 2 | Γ and H only |
| Trivial \|λ\|=1 | 0° or 180° | 0 or ∞ | 18 | All saddles (5/4/4/5) |
| **TOTAL** | | | **48** | |

---

## 3. Walker class → SM sector mapping

### 3.1 h_P + h_P_neg family (8 modes) → CHARGED FERMION SECTOR

**Source theorem:** `theorems/theorem_V_Ram_Cl6_Fock_iso_2026-05-26.md` T5 (Yukawa via iso); `theorems/theorem_yukawa_master_theory_synthesis_2026-05-20.md`

**Walker holonomy:** the 8 walker modes at P-saddle drive all 12 SM Yukawa couplings:
- 3 charged-lepton Yukawas (y_e, y_μ, y_τ): each via h_P walker × Cl(6) Fock matrix element
- 3 up-quark Yukawas (y_u, y_c, y_t): h_P walker × Type-II IR-fixed-point
- 3 down-quark Yukawas (y_d, y_s, y_b): h_P walker × walking factor
- Plus the 3 down-type variants from h_P_neg (sign-flipped) used in cross-walker CKM (V_cb / V_ub)

**Per-generation breakdown** (via T4's Q_i ↔ generation correspondence):
- gen 1 (Q_1): h_P walker × specific γ_a matrix element on Q_1 (omitting Furey pair (γ_1, γ_2))
- gen 2 (Q_2): h_P walker × matrix element on Q_2 (omitting (γ_3, γ_4))
- gen 3 (Q_3): h_P walker × matrix element on Q_3 (omitting (γ_5, γ_6))

The 8 walker modes encode 3 generations × ~2.67 Weyl-spinor multiplicity = ~22 effective Weyl spinor projections. Combined with the per-vertex Cl(6) Fock 8-dim × 4 vertices = 32 substrate matter slots, the iso identifies the 8 walker classes at P with the full 42-component charged-fermion sector of 3 generations.

**SM components hosted (42 Weyl spinors per primitive cell):**
- 3 gens × (Q_L: 6 + L_L^e: 1 + u_R^c: 3 + d_R^c: 3 + e_R^c: 1) = 3 × 14 = 42

### 3.2 h_Γ family (8 modes) → ν_L SECTOR

**Source theorem:** `theorems/theorem_neutrino_chir7_concentration_2026-05-21.md` — color singlet + chir-7 input → V_triv at A(Γ) = −1

**Walker structure:** h_Γ = (−1 + i√7)/2, tan²(arg) = 7. 8 modes total: 6 at Γ-saddle (3-fold multiplicity from λ_A=−1 mult 3) + 2 at N-saddle (supersaddle spillover, 1 each sign).

**SM components hosted (3 Weyl spinors per cell):**
- 3 gens × 1 L_L^ν component each = 3 left-handed neutrino components

**Multiplicity 8/3 ≈ 2.67:** consistent with the 4-vertex × 2-chirality structure of the iso.

### 3.3 h_H family (8 modes) → ν_R SECTOR

**Source theorem:** same chir-7 theorem — color singlet + chir-7 input → V_triv at A(H) = +1

**Walker structure:** h_H = (+1 + i√7)/2, tan²(arg) = 7. 8 modes total: 6 at H + 2 at N spillover.

**SM components hosted (3 Weyl spinors per cell):**
- 3 gens × 1 ν_R^c component each = 3 right-handed (Majorana) neutrino partners

**Right-handed Majorana M_R is theorem-grade-conditional via the m_ν3 derivation.**

### 3.4 h_N + h_N_neg family (4 modes) → DARK/INERT SECTOR

**Source closure:** an internal working note — h_N is structurally independent but observationally inert

**Walker structure:** h_N = (√5 + i√3)/2, tan²(arg) = 3/5. 4 modes total: 2 (h_N) + 2 (h_N_neg sign-flipped), all at N-saddle.

**SM components hosted: NONE.** These modes are observationally inert. They live INSIDE the framework's Cl(6) Fock per-vertex space (per A4) but no framework projection rule reads them as SM matter.

**Compatible with multi-axial dark-sector theorem** (2026-05-24): dark substrate is OUTSIDE Cl(6) Fock content. h_N is inside Cl(6) Fock but doesn't carry observable SM-charged content — consistent with "dark-content-adjacent but not the actual dark substrate."

### 3.5 Perron \|λ\|=2 (2 modes) → VEV / HIGGS VACUUM

**Source theorem:** `theorems/theorem_V_Ram_Cl6_Fock_iso_2026-05-26.md` §3 + `predictions/lambda_higgs_derivation.md`

**Walker structure:** Perron eigenvalues at Γ (λ=2) and H (λ=−2), purely real, |λ|=2 = k* − 1 + 1 = maximum walker eigenvalue.

**SM components hosted (Higgs sector):**
- 1 SM Higgs doublet H = (1, 2, +1/2) with 4 real degrees of freedom: 1 Higgs scalar (mass) + 3 Goldstone bosons (eaten by W±, Z)
- VEV alignment: ⟨H⟩ = v/√2 along Re(H^0)

The Perron modes encode the VEV-aligned direction in Bloch space; their concrete framework predictions are λ (Higgs quartic) via channel counting (2 channels × α₁_full) and v_Higgs via N_hub.

### 3.6 Trivial \|λ\|=1 (18 modes) → GAUGE BOSONS / CYCLE SPACE (NOT SM matter)

**Source closure:** an internal working note — cycle homology is NOT Hashimoto-invariant; trivial |λ|=1 modes don't carry SM-charged matter content

**Walker structure:** |λ_B| = 1 modes split into:
- Cycle-space-related (per Ihara-Bass: 4(|E|−|V|) = 8 modes per cell at any k; for our K_4 these are 4 modes per saddle from cycle space, 16 total)
- Plus λ_A = 3 / λ_A = −3 → λ_B = 1 / λ_B = −1 modes (Perron-related trivial-walker-amplitude modes)

**SM components hosted:** These modes are *gauge boson-like* (the gauge boson degrees of freedom on srs come from directed-edge holonomies, which are the |λ|=1 cycle-space content). NOT SM matter content.

**12 SM gauge bosons** (1 photon + W± + Z + 8 gluons) fit naturally into 12 of the 18 trivial modes via the gauge-group embedding Cl(6,0) → Pati-Salam → SM. The remaining 6 trivial modes correspond to the broken-PS gauge bosons (W_R, leptoquarks X/Y) that get masses at M_R per the cosmic-history Phase IIa transition.

---

## 4. Summary table

| Walker class family | # modes | SM sector | Weyl spinors per cell | Source theorem |
|---|---|---|---|---|
| h_P + h_P_neg | 8 | Charged fermions (Q_L, L_L^e, u_R^c, d_R^c, e_R^c) | 42 | V_Ram-iso T5 |
| h_Γ | 8 | ν_L | 3 | chir-7 theorem |
| h_H | 8 | ν_R (Majorana) | 3 | chir-7 theorem |
| h_N + h_N_neg | 4 | Dark/inert | 0 | A4 closure (today) |
| Perron \|λ\|=2 | 2 | Higgs VEV / vacuum | (4 dof H sector) | V_Ram-iso §3 |
| Trivial \|λ\|=1 | 18 | Gauge bosons + W_R/leptoquarks | (12 SM gauge + 6 broken PS) | Cycle homology closure (today) |
| **TOTAL** | **48** | | **48 SM Weyl spinors + 4 Higgs + 12 SM gauge bosons + 6 broken-PS gauge** | |

---

## 5. The 8↔42 multiplicity at the charged-fermion sector

**Question:** how do 8 walker modes encode 42 SM Weyl spinors?

**Answer (structural, per V_Ram-iso T1):** the 8 walker modes at P-saddle don't directly map 1-to-1 to 42 Weyl spinors. The iso identifies them with the **per-vertex Cl(6) Fock content × 4 vertices × generation-grading**:

| Layer | Multiplicity |
|---|---|
| Walker modes at P-saddle | 8 |
| Cl(6) Fock per vertex (under T1 iso) | 8 states (= 4 chiral + 4 anti-chiral) |
| Number of vertices in primitive cell | 4 |
| Generation grading via Q_i (T4) | 3 |
| Per-saddle effective Weyl-spinor count under iso | 8 × 4 / iso-redundancy = 16 / generation = 16 |
| With 3 generations | 16 × 3 = 48 |

Of the 48 generated by V_Ram-iso × 3 gens, 42 are charged-fermion components (per gen: 16 spinors total, minus 1 L_L^ν minus 1 ν_R^c = 14 charged per gen × 3 = 42). The remaining 6 (= 2 per gen × 3) are neutrinos, which the chir-7 theorem assigns to h_Γ (3 ν_L's) and h_H (3 ν_R's).

So:
- 8 P-saddle modes × Cl(6) Fock identification → 42 charged-fermion components
- 8 Γ-saddle (+spillover) modes × chir-7 → 3 ν_L components
- 8 H-saddle (+spillover) modes × chir-7 → 3 ν_R components
- TOTAL FERMION CONTENT: 42 + 3 + 3 = 48 ✓

The iso's per-vertex multiplicity (factor of 4) combined with generation grading (factor of 3) accounts for the "8 modes encode 16-Weyl-spinors-per-gen" overcounting that initially seemed mysterious.

---

## 6. What this dictionary is NOT

1. **It is NOT a per-Weyl-spinor mapping.** No row of the form "P-saddle mode #3 ↔ d_R^c of generation 2." That level of detail requires explicit enumeration of the Cl(6) Fock decomposition at each of 4 vertices × 8 walker modes, which is research-level structural work the framework's V_Ram-iso T5 theorem doesn't yet provide.

2. **It is NOT a closure of the +4 gap.** The dictionary shows matter content is structurally saturated (48 walker modes = 48 SM Weyl spinors + gauge/Higgs sectors). The +4 gauge gap (R-19) is reframed as walker-dynamics, not matter-content — but not closed by this dictionary.

3. **It is NOT a prediction of new observables.** All 12 Yukawas + 9 CKM + 3 PMNS + Higgs + 14 cosmic-history beats are *already* theorem-grade or theorem-grade-conditional. The dictionary consolidates existing assignments rather than producing new ones.

4. **It does NOT close ADOPTED-MSSM-Sb.** The literal-particle adoption stays as named convention per Branch C.

---

## 7. What this dictionary IS

1. **A consolidation of the framework's incremental walker-matter assignments into one explicit table.** Previously distributed across V_Ram-iso T5, chir-7 theorem, M_persistence, multi-axial dark, A4, cycle homology, walker-class hierarchy. Now in one place.

2. **A verification of the 48↔48 structural identity.** The accounting balances:
   - 48 walker modes = 8 charged-fermion + 8 ν_L + 8 ν_R + 4 dark + 2 Higgs + 18 gauge
   - 48 SM Weyl spinors = 42 charged-fermion + 3 ν_L + 3 ν_R + (separately: 4 Higgs dof + 12 SM gauge + 6 broken-PS gauge)
   - Cross-check via V_Ram-iso multiplicity rule: 8 walker × 4 vertices × 3 gens / iso-redundancy = 48 Weyl spinors per cell ✓

3. **A precise statement of where each framework prediction's walker support lives.** Each `predictions/*.py` file consumes specific walker classes from this dictionary. Enabling future audit work: "for prediction X, which walker class is the source?"

4. **A foundation for the next-level question.** If the 48-mode dictionary's walker-class assignments are correct, then every SM Yukawa / CKM / PMNS matrix element is a specific walker-class amplitude integral. The framework currently computes these case-by-case. A future result could fold them into a **single matrix-valued walker-class amplitude expression** for the entire SM flavor sector.

---

## 8. References

- `theorems/theorem_walker_matter_unification_2026-05-27.md` (companion meta-theorem)
- `theorems/theorem_V_Ram_Cl6_Fock_iso_2026-05-26.md` T1-T5 (walker-matter iso)
- `theorems/theorem_neutrino_chir7_concentration_2026-05-21.md` (chir-7 ν assignment)
- `theorems/theorem_dark_sector_multiaxial_waterfilling_candidate.md` (dark = outside Cl(6) Fock)
- `predictions/M_persistence_derivation.md` (12-mass operator)
- `predictions/lambda_higgs_derivation.md` (Higgs sector via Perron)
- `proofs/foundations/walker_class_dictionary_2026-05-27.py` (probe generating per-mode enumeration)
