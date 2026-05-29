# e_bit — energy of one substrate edge toggle event

**Status:** THEOREM-GRADE (zero empirical inputs)
**Date:** 2026-05-04 EOD+3
**Companion:** `predictions/e_bit.py`, `predictions/M_Pl_natural.py`, `docs/framework/framework_natural_units.md`

## 1. Abstract

`e_bit` is the framework's primitive energy unit: the energy of one substrate edge toggle event. We derive `e_bit ≡ M_substrate ≡ 1` in framework-natural units via the three-way identification (toggle = bit = Landauer-quantum) established in `docs/framework/framework_natural_units.md` (commit dc36e04). The derivation is theorem-grade with zero empirical inputs — the substrate's primitive dynamic IS its own natural unit, fixed by A1 (substrate primitive) plus Stage 2a (1 toggle = 1 bit) plus Stage 2c (Landauer κ = k_B·T·ln(2)) plus the canonical T_substrate identification from substrate's own ω_tick.

In framework-natural units: **e_bit = 1** (exact). The "GeV value" (≈ 2.71×10¹⁸ GeV via M_Pl × √π/8 with CODATA M_Pl) is an anthropocentric SI translation, not a framework prediction.

This file is the canonical source for the framework's natural energy unit. Every downstream prediction that needs an energy/mass scale should import `e_bit` from here and express its value as `(structural coefficient) × e_bit`, replacing previous hardcoded CODATA M_P_GeV anchors.

## 2. Framework axioms invoked

- **A1** (substrate primitive): edges of the substrate graph carry a binary self-inverse toggle. Each toggle event is the substrate's irreducible dynamical event.
- **A-IT3** (Landauer): the minimum free-energy cost of erasing one bit of information at temperature T is k_B·T·ln(2). Cited as a framework-load-bearing information-theoretic axiom (`docs/framework/framework_axioms.md` §9).
- **Stage 2a** (theorem, derived from A1 + counting): one toggle event corresponds to exactly one bit of substrate-state information change. (`docs/theorems/theorem_edge_surprise_thresholds.md` §3.)
- **Stage 2c** (theorem, derived from A-IT3 + observer model): the energy associated with one bit at temperature T equals κ = k_B·T·ln(2). (`docs/theorems/theorem_observer_energy_functional.md`.)
- **Canonical convention** (`docs/framework/framework_natural_units.md` §1c, dc36e04): T_substrate is fixed by the substrate's own dynamics — ℏ·ω_tick = κ_substrate with ω_tick = 2π/t_tick.

No other axioms invoked. No PDG / CODATA / observed inputs anywhere in the chain.

## 3. Derivation

### Step 1 — A1 substrate primitive [Type 1]

The substrate is defined by A1 to consist of edges that toggle binary states. Each toggle event is the substrate's primitive dynamical occurrence. There is no smaller event in the substrate's ontology.

### Step 2 — Stage 2a: 1 toggle = 1 bit [Type 4]

Per `docs/theorems/theorem_edge_surprise_thresholds.md` §3 (binary-alphabet convention on A1 maintained throughout the framework), the Shannon-information content of one toggle event is exactly one bit. Each edge has two states (toggled / untoggled); a toggle event flips exactly one edge's state, conveying log₂(2) = 1 bit of substrate-state information.

### Step 3 — Stage 2c Landauer [Type 4]

Per `docs/theorems/theorem_observer_energy_functional.md` (citing Landauer 1961 §2 and Bennett 1973), the minimum free-energy cost associated with one bit of irreversible information processing at temperature T is

$$\kappa := k_B T \ln 2$$

Equivalently, by the time-reverse of Landauer's argument: the minimum energy that must be associated with one bit's worth of state-change at temperature T is $\kappa$.

### Step 4 — Substrate sets its own temperature [Type 4]

Per `docs/framework/framework_natural_units.md` §1c (canonical, commit dc36e04), the substrate's intrinsic temperature T_substrate is not a free parameter: it is fixed by the substrate's own dynamical scales. Specifically, the energy quantum of one toggle (Step 3) must equal the action quantum per substrate tick:

$$\kappa_{\text{substrate}} = \hbar \, \omega_{\text{tick}}, \qquad \omega_{\text{tick}} = \frac{2\pi}{t_{\text{tick}}}$$

Combining with Step 3:

$$k_B T_{\text{substrate}} \ln 2 = \hbar \omega_{\text{tick}} \quad\Longrightarrow\quad T_{\text{substrate}} = \frac{\hbar \omega_{\text{tick}}}{k_B \ln 2}$$

This fixes T_substrate as a derived quantity — there is no calibration freedom once t_tick is set as the time unit.

### Step 5 — Unit identification [Type 1]

In framework-natural units, choose ℏ = c = 1 and let the substrate's primitive event define the natural unit:

- **Time unit:** one substrate tick (with ω_tick = 1 in these units, so t_tick = 2π)
- **Energy unit:** one toggle's energy = e_bit
- **Mass unit:** by mass-energy equivalence (c = 1), M_substrate ≡ e_bit
- **Length unit:** lattice spacing (set to 1 by convention)
- **Temperature:** T_substrate = 1/(k_B ln 2), or with k_B = 1: T_substrate = 1/ln(2)

In these units:

$$\boxed{\,e_{\text{bit}} \;=\; M_{\text{substrate}} \;=\; 1 \quad \text{[exact, definitional]}\,}$$

This unit choice is forced by parsimony: the substrate's only primitive is the toggle, so the only natural energy unit available is one toggle's energy. Any other choice (e.g., setting M_Pl = 1 in Planck units) imports structure not present in A1.

## 4. Result

In framework-natural units (lattice spacing = 1, ω_tick = 1, ℏ = c = k_B = 1, M_substrate = 1):

| Quantity | Value | Status |
|---|---|---|
| e_bit | 1 | Theorem-grade, exact |
| M_substrate | 1 | Same as e_bit (by mass-energy equivalence) |
| ω_tick | 1 | Substrate angular frequency |
| t_tick | 2π | Substrate tick time |
| T_substrate | 1/ln(2) ≈ 1.4427 | Substrate intrinsic temperature |
| M_Pl / e_bit | 8/√π ≈ 4.5135 | From `predictions/M_Pl_natural.py` (Drude + Planck convention) |

## 5. Comparison with experiment

There is **no observational comparison applicable** to the framework-natural value e_bit = 1. This is a unit identification, not a measured observable. The CODATA value e_bit ≈ 2.71×10¹⁸ GeV (computed as M_Pl × √π/8 with CODATA M_Pl_GeV ≈ 1.22089×10¹⁹ GeV) is an **anthropocentric SI translation**, telling us how many of our SI-derived energy units (GeV) equal one of the framework's natural energy units (toggle).

The conversion factor "1 toggle ≈ 2.71×10¹⁸ GeV" is a measured quantity, like "1 meter = 3.28 feet." It encodes no physics — it's a unit translation between the framework's own natural system and our anthropic SI system. It belongs in comparison/test code, not in this prediction file.

**Clause 8 status:** N/A — there is no PDG-comparable quantity. The exact framework-internal value e_bit = 1 is a definitional identity by Step 5.

## 6. Open questions

1. **t_tick = t_Planck commitment.** The framework's Row 25 commits t_tick = t_Planck, but that commitment is currently theorem-grade-conditional rather than theorem-grade. If the commitment opens, the Step 5 identification still holds, but the relation to Planck units (and hence to CODATA) shifts. This affects the SI translation factor, not the framework-natural value.

2. **Generalization to non-binary alphabets.** Stage 2a assumes binary toggle alphabet on A1. If the framework ever generalizes A1 to non-binary alphabets, "1 toggle = 1 bit" generalizes to "1 toggle = log₂(alphabet size) bits" and the Landauer-energy quantum scales accordingly. Not currently relevant under canonical A1.

## 7. References

### Framework upstream
- `docs/framework/framework_axioms.md` — A1 (substrate primitive), A-IT3 (Landauer)
- `docs/theorems/theorem_edge_surprise_thresholds.md` §3 — Stage 2a (1 toggle = 1 bit)
- `docs/theorems/theorem_observer_energy_functional.md` — Stage 2c (κ = k_B·T·ln 2)
- `docs/framework/framework_natural_units.md` (commit dc36e04) — canonical convention identifying toggle = bit = Landauer-quantum

### Downstream consumers (this file is canonical for them)
- `predictions/M_Pl_natural.py` — M_Pl = (8/√π) × e_bit (theorem-derived ratio via Drude + Planck convention)
- All future prediction files needing an energy/mass scale should import `e_bit` from here.

### External (cited theorems, not used as inputs)
- Landauer, R. (1961). *Irreversibility and heat generation in the computing process.* IBM J. Res. Dev. 5, 183–191.
- Bennett, C. H. (1973). *Logical reversibility of computation.* IBM J. Res. Dev. 17, 525–532.

## 8. Audit v2 status

**Clause 7:** Inheritance citation. The unit identification is forced by parsimony (A1 has only the toggle as primitive, so the natural energy unit is fixed by Steps 1-5). No alternative axes to defend against — there is no other "natural energy unit" available in any framework with A1 alone. Clause 7 PASS via parsimony argument.

**Clause 8:** N/A as noted in §5 — e_bit is a unit, not a measurable observable. Framework-internal value = 1 exactly.

**Combined status:** **THEOREM-GRADE.** Zero empirical inputs, derivation chain Steps 1-5 entirely Type 1/4 admissible under the linter's hard quality gate.
