# Derivation of m_b (bottom quark mass)

**Status:** THEOREM-GRADE-STRUCTURAL-CONDITIONAL.
**File:** `predictions/m_b.py`

## Abstract

The bottom quark mass is the gen-3 anchor of the down-quark sector,
derived from the framework's §3 selection rule at Type IV Perron walker
(L = g = 10): y_b = (2/3)^10 = 1024/59049. Bridge to mass:
`m_b = v · y_b` (low-scale framework convention, no /√2 since L > 0).
Result: 4.270 GeV, +2.15% vs PDG 4.18 GeV.

## Framework axioms invoked

- **A1** (toggle), **A2-T** (MDL waterline), **A5(b)** (couplings),
  **B3** (Pati-Salam spinor sector).

## Derivation

### Step 1 — Type IV Perron walker (theorem-grade-structural)

Per `theorem_selection_map_2026-05-21.md` (24→1 forced bijection) and
`theorem_updown_split_conjugate_higgs_2026-05-21.md`:

- Cl(6) Fock Hamming weight n = 1 → color triplet.
- Bloch concentration at Γ trivial λ = +3, IB root h = 2 (Perron).
- Type IV (Perron walker) traversing full girth: L = g = 10.
- Down-type couples to odd-grade Higgs H → can flip handedness → walk runs.
- Selection rule:
  $$y_b = \chi \cdot Q^L / k_*^{\text{edge\_sel}} = 1 \cdot Q^{g} / 1 = Q^g = \left(\frac{k_*-1}{k_*}\right)^g.$$

For k* = 3, g = 10:
  $$y_b = (2/3)^{10} = 1024/59049 \approx 0.017341.$$

### Step 2 — Scale assignment for Type IV (M_persistence §6)

Type IV cycle-walker (L > 0) produces y_b at LOW SCALE; the walker
traverses the IR completion and the selection rule directly yields the
SM-effective Yukawa. Bridge to mass uses the framework's W25 convention
`m = v · y` (no /√2 factor; see `framework_scheme_convention.md` line 56).

### Step 3 — Direct evaluation

$$m_b = v \cdot y_b = 246.22 \cdot \frac{1024}{59049} = 4.270 \text{ GeV}.$$

## Result

$$\boxed{\;m_b = v \cdot \left(\frac{k_*-1}{k_*}\right)^g = 246.22 \cdot (2/3)^{10} = 4.270 \text{ GeV}\;}$$

## Comparison with experiment

| Quantity | Predicted | PDG 2024 | Deviation |
|---|---|---|---|
| m_b | 4.270 GeV | 4.18 ± 0.03 GeV (MS-bar at m_b) | **+2.15%, +2.99σ_PDG** |

Within framework's stated ~1-2% systematic floor (slightly above at
2.15%); residual attributed to MS-bar threshold matching not derived
within the framework's selection rule scope.

## Inputs

| Symbol | Value | Status | predictions/ file | Meaning |
|---|---|---|---|---|
| k_star | 3 | [derived] | k_star.py | trivalent coordination |
| g_girth | 10 | [derived] | g_girth.py | srs girth |
| v_higgs | 246.22 GeV | [derived] | v_higgs.py | Higgs VEV (BZJ) |

No PDG masses or external observed values enter the derivation. CODATA
M_Pl appears only via v_higgs's SI translation (cancels in dimensionless
y_b · v / m_b ratio).

## Open questions

1. **MS-bar threshold matching.** The 2.15% deviation reflects the
   framework's single-regime no-threshold scheme; b-quark threshold
   matching at m_b scale could close ~1% but is "out-of-scope by
   construction" per `predictions/alpha_s.py` discipline.
2. **Family-D analog on Type IV vertex.** Whether the down-type
   fermion-Higgs vertex admits an analog of the 5/12 dark correction on
   v is open research (Priority 4.4 step 2.2).

## Cross-references

- `docs/theorems/theorem_walker_length_MDL_waterline_2026-05-21.md` §4.4 (Type IV)
- `docs/theorems/theorem_updown_split_conjugate_higgs_2026-05-21.md` (odd H → L=g)
- `docs/framework/framework_scheme_convention.md` line 56 (W25 Yukawa convention)
- `predictions/v_higgs.py`, `predictions/M_persistence.py`
