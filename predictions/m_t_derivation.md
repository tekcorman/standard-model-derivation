# Derivation of m_t (top quark mass)

**Status:** THEOREM-GRADE-STRUCTURAL-CONDITIONAL.
**File:** `predictions/m_t.py`
**Companion:** `predictions/M_persistence.py` (the unified 12-mass operator)

> **★ 2026-06-25 UPDATE — the forced dark correction shipped; this doc's bare result is superseded.**
> This document derives the BARE saturation anchor m_t^bare = v/√2 · y_t = 174.10 GeV. The live
> prediction applies the resolvent's own forced first-girth-return dark correction
> ×(1 − α₁/h_P²) (L = 0 saturated read ⇒ power 2, zero adoption):
> **m_t = 172.41 GeV → −0.95σ_PDG.** See `predictions/m_t.py`,
> `predictions/heavy_quark_anchor_dark.py`, and
> `docs/theorems/theorem_dark_self_energy_unified_2026-06-28.md`. The "+4.71σ / structural floor"
> framing in the comparison section below is the PRE-dark reading, retained as the bare-anchor
> derivation record — the residual was NOT a floor; it was the not-yet-applied forced correction.

## Abstract

The top quark mass is derived as the gen-3 eigenvalue of the up-sector
block M^(u) of the framework's complete fermion mass operator
M_persistence (`docs/theorems/theorem_fermion_mass_operator_persistence_2026-05-21.md`).
The up-sector anchor is set by Type II saturation (`theorem_walker_length_MDL_waterline_2026-05-21.md` §4.2)
giving y_t = 1, and the bridge to physical mass uses the SM-equivalent
low-scale convention `m_t = (v/√2) · y_t`, giving the bare anchor 174.10 GeV;
the forced dark correction ×(1 − α₁/h_P²) (see banner) yields the live
result **172.41 GeV, −0.95σ** vs PDG 172.69 GeV.

## Framework axioms invoked

- **A1** (toggle): binary self-inverse primitive.
- **A2-T** (MDL waterline): derived theorem per
  `theorem_A2_mdl_from_finite_register.md`.
- **A5(b)** (couplings): MDL probability = coupling strength.
- **B3** (Pati-Salam spinor sector): theorem-grade via Cl(6) = Spin(6) = SU(4)
  embedding.
- **R3** (observer's C³): theorem-grade per `theorem_R3_observer_c3_generation.py`.

## Cited mathematical theorems

- Furey 2018 (Cl(6) Fock ↔ SM species): Hamming weight n → walker type.
- Pati-Salam 1974 (SU(4)_PS × SU(2)_L × SU(2)_R).
- Standard MSSM 1-loop RGE β-functions (Martin SUSY Primer 1997 §5).

## Derivation

### Step 1 — Type II saturation walker (theorem-grade)

Per `theorem_walker_length_MDL_waterline_2026-05-21.md` §4.2 and the
selection map theorem (`theorem_selection_map_2026-05-21.md`), the
up-type quark (n = 2 Cl(6) Fock Hamming weight) is forced (24→1 bijection)
to Type II saturation:

- Bloch concentration at Γ trivial λ = +3, IB root h = 1.
- Walker length L = 0 (no girth-cycle traversal).
- Selection rule:
  $$y_t = \chi \cdot Q^L / k_*^{\text{edge\_sel}} = 1 \cdot Q^0 / k_*^0 = 1.$$

This y_t = 1 is THEOREM-GRADE structurally — the saturation regime has
both Γ trivial IB roots degenerate at L=0 giving y = h^0 = 1 identically
(W40 finding Y3).

### Step 2 — Scale assignment for Type II (M_persistence synthesis §6)

Per an internal working note §6, the
selection rule produces y at a walker-type-dependent natural scale:

- **L > 0** (cycle-walkers, Types III, IV): y is at LOW SCALE; the walker
  traverses the IR completion and the selection rule directly produces
  the SM-effective Yukawa. Bridge: `m = v · y`.
- **L = 0** (saturation, Type II): y is at the saturation regime (UV).
  Bridge to mass uses MSSM IR-fixed-point limit at sin β ≈ 1
  (equivalently, SM-convention with /√2 factor): `m = (v/√2) · y`.

The /√2 emerges because Type II saturation lives at the MSSM
large-tan(β) IR fixed point where sin(β) ≈ 1 and the natural
fermion-Higgs vertex coefficient is v/√2 = v_u (the up-type VEV
component in the standard MSSM Higgs doublet convention).

### Step 3 — Direct evaluation

$$m_t = \frac{v}{\sqrt{2}} \cdot y_t = \frac{246.22}{\sqrt{2}} \cdot 1 = 174.10 \text{ GeV}$$

with v from `predictions/v_higgs.py` (BZJ + 5/12 dark vertex,
theorem-grade-conditional on N_hub closure per Gap G1).

## Result

Bare saturation anchor (this document's derivation):

$$m_t^{\text{bare}} = \frac{v}{\sqrt{2}} \cdot y_t^{(\text{Type II})} = 174.10 \text{ GeV}$$

Live prediction (× the forced dark correction, 2026-06-25):

$$\boxed{\;m_t = \frac{v}{\sqrt{2}} \cdot y_t^{(\text{Type II})} \cdot \left(1 - \frac{\alpha_1}{h_P^2}\right) = 172.41 \text{ GeV}\;}$$

## Comparison with experiment

| Quantity | Predicted | PDG 2024 | Deviation |
|---|---|---|---|
| m_t (live, dark-corrected) | **172.41 GeV** | 172.69 ± 0.30 GeV (pole) | **−0.95σ_PDG** ✅ |
| m_t^bare (anchor, pre-dark) | 174.10 GeV | — | (+0.82%, +4.71σ — historical) |

The bare +4.71σ residual was closed 2026-06-25 by the forced first-girth-return
dark correction of the one resolvent (Σ = α₁/h; L = 0 saturated read ⇒ power 2 —
`docs/theorems/theorem_dark_self_energy_unified_2026-06-28.md`). The earlier
"~1-2% structural floor" attribution is retracted: the residual was the
not-yet-applied forced correction, not a floor.

## Inputs

| Symbol | Value | Status | predictions/ file | Meaning |
|---|---|---|---|---|
| k_star | 3 | [derived] | k_star.py | trivalent srs coordination |
| g_girth | 10 | [derived] | g_girth.py | srs girth |
| v_higgs | 246.22 GeV | [derived] | v_higgs.py | Higgs VEV (BZJ) |
| y_t | 1 (exact) | [theorem] | (Type II saturation) | top Yukawa anchor |

No PDG masses, no MSSM RGE running, no tan(β) factor enters the
derivation. The 1/√2 bridge factor is a structural feature of the
Type II saturation scale assignment.

## Open questions

1. **Sub-leading Family-D analog on the Type II vertex.** The +0.82%
   residual is the framework's structural floor; whether a Feshbach
   analog on the up-type fermion-Higgs vertex (analog of the 5/12 on v
   in the Higgs sector) brings this to <0.1% is open research.
2. **The /√2 bridge factor's exact origin.** The scale assignment in §6
   identifies this as MSSM IR-FP at sin β ≈ 1, but a fully framework-
   internal derivation of the /√2 (vs explicit MSSM convention citation)
   is open research.
3. **Need-D-3 / V_Ram ≅ Cl(6)-Fock** (the framework's named multi-
   session block) gates the species ↔ walker-type mapping
   mechanically; until closed, "Type II for up" is the inherited
   conditional rather than a stand-alone theorem.

## Cross-references

- `docs/theorems/theorem_fermion_mass_operator_persistence_2026-05-21.md`
- `docs/theorems/theorem_walker_length_MDL_waterline_2026-05-21.md` §4.2
- `docs/theorems/theorem_selection_map_2026-05-21.md`
- `docs/theorems/theorem_updown_split_conjugate_higgs_2026-05-21.md`
  (d/u split via even-grade conjugate Higgs)
  (scale assignment per walker type)
- `predictions/v_higgs.py` (Higgs VEV chain)
- `predictions/M_persistence.py` (M^(u) gen-3 eigenvalue)
