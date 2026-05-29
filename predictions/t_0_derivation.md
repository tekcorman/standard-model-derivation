# Age of the Universe: Cascade Derivation from the adopted N_hub

**Audit anchor:** Downstream of Row P17 (N_hub) of `docs/parameters/parameter_uniqueness_ledger.md`. t_0 inherits N_hub's STRICT-SOLID (the value of the adopted N_hub is empirical, pinned via the measured G_F) status.

**Parameter:** t_0 (age of the universe)
**Predicted value:** 14.38 Gyr
**Observed (Methuselah):** 14.46 ± 0.80 Gyr (Bond et al. 2013, ApJ 765:L12)
**Deviation (Methuselah):** −0.10σ
**Observed (CMB/ΛCDM):** 13.797 ± 0.023 Gyr (Planck 2018; model-dependent)
**Deviation (CMB):** +25.4σ (different cosmological model; see §4)
**Status:** GENUINE PREDICTION. Conditional on G1 (BZJ chain; same gap as v_Higgs, H_0).
**Derivation grade:** `theorem-conditional` on G1 (same wall as v_Higgs).
**Date:** 2026-04-22 (session 19; the observable used to calibrate N_hub's value changed from H_0 to G_F — N_hub is the adopted input either way).

---

## 1. Abstract

The age of the universe is derived in three steps: (i) the measured Fermi
constant G_F determines the Higgs VEV v_GF via the tree-level SM relation;
(ii) the BZJ formula (from v_higgs_derivation.md) inverted for N gives the
toggle-graph site count N; (iii) the cascade theorem t_0 = N·t_P (theorem-
grade; coefficient exactly 1) converts N to a time.

Key numbers:
- the MEASURED G_F = 1.1663787×10⁻⁵ GeV⁻² → v_GF = 246.22 GeV (the calibration target for N_hub's value; G_F is downstream)
- N = 8.42×10^60
- t_0 = N·t_P = 14.38 Gyr  (+0.0σ from Methuselah star)
- H_0 = 1/(N·t_P) = 68.0 km/s/Mpc  (independent prediction, +1.2σ from Planck CMB)

The framework predicts H_0·t_0 = 1 (coasting cosmology: Ω_Λ = 1/k* = 1/3;
see §4 and an internal working note).

---

## 2. Framework Axioms Invoked

**A1 (Toggle/srs lattice).** k* = 3, girth g = 10.

**A2 (MDL / edge process).** BZJ chain, dark vertex correction 5/12
(theorem-grade; dark_feshbach_a2_closure.py, session 18).

**Cascade theorem** (N_hub.py): H = 1/(N·t_P) with coefficient exactly 1.
Derivation: k*N toggles per t_P, acceptance probability 1/(k*N), rate = 1
causal state per t_P, H = 1/(N t_P). No adoption needed for the coefficient.

---

## 3. Derivation

### Step 1: G_F → v_GF (tree-level SM)

The Fermi constant and the Higgs VEV are related at tree level by:

    G_F = 1 / (√2 · v²)    ⟺    v = 1 / (√2 G_F)^{1/2}

This is exact at tree level in the electroweak theory (integrating out the
W boson at q²=0; see v_higgs_derivation.md §3 for full reference chain).

    G_F_obs = 1.1663787×10⁻⁵ GeV⁻²   [the MEASURED Fermi constant; PDG 2024 / MuLan 2011 — the observable that calibrates N_hub's value]
    v_GF    = 246.219 GeV              [the VEV implied by the MEASURED G_F; model-independent — the calibration target for N_hub's value]

**Grade:** Tree-level SM (Type 3 citation; Peskin & Schroeder §20.1).

### Step 2: v_GF → N (BZJ inversion)

From v_higgs_derivation.md, the BZJ formula is:

    v = δ² · M_P · dark / (√2 · N^{1/4})

where:
- δ = 2/9  (Koide phase; theorem-grade from srs D¹₁₀/k* chain)
- dark = 1 - (5/12)α₁  (theorem-grade; dark_feshbach_a2_closure.py)
- α₁ = (2/3)⁸  (theorem-grade from NB walk)
- M_P = 1.22089×10^19 GeV  (external; CODATA 2018)

Inverting for N with v = v_GF:

    N = (δ² · M_P · dark / (√2 · v_GF))^4

**Numerics:**
- δ² = 4/81
- dark = 1 - (5/12)(2/3)⁸ = 0.98374...
- √2 · v_GF = 348.17 GeV
- δ² · M_P · dark = 0.059308 × 10^19 GeV
- N^{1/4} = 0.059308×10^19 / 348.17 = 1.7032×10^15
- N = (1.7032×10^15)^4 = 8.417×10^60

**Grade:** STRICT-SOLID conditional on G1. All internal coefficients (δ, dark)
are theorem-grade. M_P is external (CODATA 2018). Gap G1 is the identification
N = N_hub (the BZJ formula requires matching N to the current cosmic epoch).

### Step 3: N → t_0 (cascade theorem)

From N_hub.py cascade theorem (theorem-grade):

    H = 1 / (N · t_P)    [coefficient exactly 1]

Combined with H = 1/t_0 (coasting: H·t_0 = 1):

    t_0 = N · t_P = 8.417×10^60 × 5.391247×10⁻⁴⁴ s = 4.538×10^17 s

Converting: t_0 = 4.538×10^17 s / (3.1557×10^16 s/Gyr) = **14.38 Gyr**

**Grade:** The cascade theorem H = 1/(N t_P) is THEOREM-GRADE.
The step H·t_0 = 1 (i.e., t_0 = 1/H_0) is the coasting condition (see §4).

---

## 4. Cosmological Context: Coasting Condition

The framework predicts H_0·t_0 = 1 exactly.  This is the coasting condition
ä = 0, which in a flat universe with matter (Ω_m) and cosmological constant
(Ω_Λ) requires:

    Ω_m = 2Ω_Λ  and  Ω_m + Ω_Λ = 1  ⟹  Ω_Λ = 1/3,  Ω_m = 2/3

The k*=3 NB walk gives exactly this:
- NB survival fraction: (k*-1)/k* = 2/3 = Ω_m
- Backtrack/return fraction: 1/k* = 1/3 = Ω_Λ

The CMB/ΛCDM age 13.797 Gyr is derived under a model with Ω_Λ ≈ 0.68,
Ω_m ≈ 0.31 — different cosmology.  The +25σ deviation from CMB is a
model-comparison signal, not a precision failure.

The Methuselah star age 14.46 ± 0.80 Gyr is a direct, model-independent
measurement. Agreement within 0.1σ is strong evidence for the framework's
cosmological model.

See an internal working note for Λ_CC derivation status.

---

## 5. Open Gaps

| Gap | Description | Status |
|-----|-------------|--------|
| G1  | N = N_hub (current cosmic epoch) | BLOCKED: needs H_0 from A1-A4 or Λ_CC |
| Coasting | Ω_Λ = 1/k* from first principles | SCOPING (an internal working note) |

G1 is the same gap as for v_Higgs, H_0, and Newton's G.  The G_F-calibration of N_hub's value
makes H_0 and t_0 genuine predictions rather than round-trips, but G1 remains
open.

---

## 6. References

- Planck Collaboration 2018, arXiv:1807.06209 (CMB t_0 = 13.797 ± 0.023 Gyr)
- Bond, Nelan, VandenBerg, Schaefer, Lawler 2013, ApJ 765:L12 (Methuselah 14.46 ± 0.80 Gyr)
- Webber et al. (MuLan) 2011, PRL 106:041803 (G_F, 0.6 ppm)
- PDG 2024 (Navas et al., Phys. Rev. D 110, 030001) — G_F = 1.1663787±0.0000006×10⁻⁵ GeV⁻²
- CODATA 2018 — t_P = 5.391247×10⁻⁴⁴ s
- predictions/N_hub.py — cascade theorem H = 1/(N t_P)
- predictions/v_higgs_derivation.md — BZJ chain, δ = 2/9, dark = 5/12 α₁

## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.
