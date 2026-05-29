# Framework natural units — toggle = bit = natural energy

**Date:** 2026-05-04 EOD.
**Status:** Canonical convention. Builds on closed theorems (Stage 2a edge-surprise thresholds, Stage 2c observer energy functional, G_sub Drude closure). Resolves the long-standing implicit ambiguity in what "framework-natural units" means.

**Purpose.** Make explicit and canonical the framework's natural unit system, so that mass-scale predictions (M_Pl, M_substrate, M_R, M_unif candidate, …) are expressible as derived structural numbers without unit-conversion ambiguity. The framework's natural unit IS the substrate's primitive dynamic — one edge toggle, equivalently one bit of A1 substrate-state information, equivalently one Landauer-quantum of energy at the substrate's intrinsic temperature.

---

## 1. The identification

Three equivalent statements of the same primitive:

1. **Substrate dynamics:** one edge toggle (A1's primitive event).
2. **Information content:** one bit of substrate-state description (A1's binary alphabet on each edge).
3. **Energy quantum:** one Landauer-quantum κ = k_B·T_substrate·ln(2) of energy.

The substrate is *defined* by A1 to have edges that toggle binary states. Each toggle is a single dynamical event. The Shannon-information content of one toggle is one bit (Stage 2a §3 maintains the framework's binary-alphabet convention). The energy associated with one bit's worth of irreversible information processing is, by Landauer 1961 + Bennett 1973 (A-IT3, framework-load-bearing, theorem-grade per `theorem_observer_energy_functional.md`), exactly k_B·T·ln(2) at temperature T.

When T is the substrate's intrinsic temperature T_substrate (set by the substrate's own dynamical scales, *not* by an observer or environment), the conversion factor is fixed: one toggle ↔ one bit ↔ κ_substrate of energy. There is no calibration freedom — the substrate's primitive is its own unit.

**Framework-natural unit convention (canonical):**

```
1 toggle event = 1 bit of substrate-state description = 1 natural energy unit
```

In these units, k_B·T_substrate·ln(2) = 1.

This is the framework's analog of "ℏ = c = 1" — a unit choice that makes the substrate's primitive dynamic dimensionless. Combined with the existing conventions (lattice spacing = 1, tick = 1, ℏ = c = 1), it eliminates *all* dimensional ambiguity.

---

## 2. Why this is the right convention

Three reasons:

**(a) The substrate has no other natural unit to choose.** A1's only primitive is the edge toggle. The substrate has no internal mass, length, or time scale that comes from outside the toggle dynamics. Stage 2c's energy functional E_obs = κ·S_total identifies κ as the *unique* dimensional bridge between substrate-information and physical energy, with κ already at theorem-grade.

**(b) It's automatic in any A1-only framework.** Any theory whose only primitive is "binary toggles on a discrete graph" must reduce to information-theoretic units at the deepest level. Choosing some other natural unit (e.g., M_Pl = 1 in Planck units, or m_e = 1 in atomic units) would import structure not present in A1 — the resulting framework would have an "external dimensional input" baked into its unit definition.

**(c) It fixes the previously-implicit T.** Stage 2c notes "this theorem does not calibrate T to a specific value — κ serves as an information-to-energy conversion constant." That non-calibration is *exactly* the freedom this convention removes: T_substrate is fixed by the substrate's own dynamics (one toggle per tick, with energy ℏ × ω_tick where ω_tick = 2π/t_tick), giving κ_substrate as a derived structural number once the lattice tick is set as the time unit.

---

## 3. Concrete consequences for mass-scale predictions

In framework-natural units (1 toggle = 1 natural energy unit, lattice tick = 1, lattice spacing = 1, ℏ = c = 1):

### 3.1 The substrate-local family

All substrate-local masses are predicted as specific dimensionless numbers:

| Quantity | Formula | Numerical value | Source |
|---|---|---|---|
| M_substrate | unit choice (Drude reference) | 1.0 | `predictions/G_N.py` Drude form |
| M_Pl | 8/√π × M_substrate | 4.5135... | `predictions/M_Pl_natural.py`, Row P61 |
| M_R | 2/k*^(g−1) × M_Pl | 4.586 × 10⁻⁴ | `proofs/flavor/srs_M_R_step{1,2,3}*.py`, Row P31 |
| M_unif (candidate) | 32/k*^(g−1) × M_Pl | 7.338 × 10⁻³ | an internal working note |

These numbers are *predictions*. There is no external observation that fixes them; they are computed from substrate combinatorics (k*, g, N_atoms) plus the Drude relation (which gives 8/√π and π/64 from finite-(ω,T) Kubo on the Bloch operator).

### 3.2 The FSS family (N-dependent)

The FSS-family scales involve the substrate's overall size N_hub:

| Quantity | Formula | Status |
|---|---|---|
| v | δ²·M_Pl/(√2·N^(1/4)) | predicted given N_hub |
| m_τ | y_τ × v | predicted given N_hub |
| m_ν₃ | 12 × M_Pl·N^(-1/2) | predicted given N_hub |

In framework-natural units, these are predicted numbers *as functions of N_hub*. Currently, N_hub is anchored externally to ppm precision via consistency with the measured G_F (Fermi coupling round-trip; a calibration, not a structural tie). To convert each to a single dimensionless prediction independent of external observation, N_hub must itself be derivable from substrate combinatorics — see §3.3 below and the "N_hub first-principles derivation" frontier entry in [`../master_plan.md`](../master_plan.md).

### 3.3 The remaining genuine open frontier

**N_hub** — the substrate's site count, currently anchored externally via G_F. The master plan flags this: "Reducing to zero anchors needs a 5th independent dimensionless theorem-grade relation (no candidate identified)."

In the natural-unit framing of this doc, the question is sharper: is N_hub *intrinsically derivable* from substrate combinatorics, or is it a "size of the universe" parameter that requires either anthropic argument or external observation? This is the **only** dimensional unknown that survives the toggle-bit-energy identification.

---

## 4. Relation to "GeV value" of M_Pl

The CODATA value M_Pl ≈ 1.22 × 10¹⁹ GeV is *not* a framework prediction in the natural-unit framing. It is a unit translation: it tells us how many of *our* SI-derived energy units (GeV) equal one of the framework's natural energy units (toggle).

Specifically:
- Framework predicts M_Pl = 8/√π in framework-natural units.
- We measure (via CODATA G_N or M_P): 1 toggle event ≈ √π/8 × 1.22e19 GeV ≈ 2.7e18 GeV.
- Therefore M_Pl ≈ 1.22e19 GeV by composition.

The conversion factor "1 toggle ≈ 2.7e18 GeV" is a measured quantity, like "1 meter = 3.28 feet." It encodes no physics — it's a unit translation between the framework's own natural system and our anthropic SI system.

This framing replaces the older "external dimensional anchor" framing. The framework predicts physics in its natural units; SI-conversion is one-time experimental.

---

## 5. What this canonicalizes vs. leaves open

**Canonicalized (after this doc):**
- The framework's natural unit is the toggle/bit/Landauer-quantum.
- M_Pl, M_substrate, M_R, M_unif (candidate) are derived numbers in this unit system.
- "GeV value of M_Pl" is a unit translation, not a free parameter.
- Prediction file `predictions/M_Pl_natural.py` reads M_Pl = 8/√π as a natural-unit prediction.

**Still open (separate questions):**
- N_hub from substrate combinatorics (master plan §3.1; only genuine remaining dimensional unknown in the framework).
- M_unif candidate's structural proof (currently scoped as Reading B2: gauge two-point bilinear × trivial walker; numerical match at machine precision but Reading uniqueness not yet established).
- Whether the substrate's tick equals the Planck time exactly (Row 25 commitment) or admits a small ratio adjustment.

**Out of scope (deliberately):**
- Calibrating individual SI conversion factors (k_B in J/K, ℏ in J·s, c in m/s). These are external CODATA inputs, not framework predictions, by construction.
- Connecting to laboratory thermodynamic measurements of the substrate's "actual" temperature. Stage 2c's remark that "T is observer-dependent" applies only when T is set by an environment/observer, not when T is the substrate's intrinsic dynamical scale (the convention here).

---

## 6. Cross-references

- `docs/theorems/theorem_observer_energy_functional.md` (Stage 2c, κ = k_B·T·ln(2) closed)
- `docs/theorems/theorem_edge_surprise_thresholds.md` (Stage 2a, surprise in bits)
- `docs/theorems/theorem_g_sub_drude_closure_2026-04-30.md` (M_Pl/M_substrate = 8/√π)
- `docs/theorems/theorem_dimensionless_ratio_principle_2026-04-30.md` (meta-principle: predictions are ratios)
- `predictions/M_Pl_natural.py` (M_Pl = 8/√π in natural units, untethered structural prediction)
- `predictions/G_N.py` (G_N · M_Pl² = 1 derived identity)
- `docs/parameters/parameter_uniqueness_ledger.md` Rows P60, P61
- `docs/master_plan.md` §3.1 (N_hub from substrate, the remaining frontier)
- `docs/framework/framework_axioms.md` §9 (A-IT3 Landauer)

---

## 7. Recommended downstream uses

When a future prediction or analysis invokes "the Planck mass" or "framework-natural units":

- Cite this doc to fix the convention (no need to re-explain).
- M_Pl is a derived structural number = 8/√π in these units.
- "M_Pl in GeV" is a unit translation, handled at the prediction-file level via one CODATA conversion factor.
- N_hub remains the only genuine "external" dimensional input until structurally derived.

This doc supersedes implicit unit conventions in earlier framework files. Update existing files that say "external dimensional anchor for M_Pl" to instead say "unit translation to SI via one CODATA conversion."
