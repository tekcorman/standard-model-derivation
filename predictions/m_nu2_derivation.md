# m_ν₂ — Second light neutrino mass (normal ordering)

**Status:** THEOREM-GRADE-CONDITIONAL (inherits m_ν₃ chain)
**Date:** 2026-05-04 (supersedes 2026-05-02 EOD+9 ADOPTED-PS+ADOPTED-Z3 derivation)
**Companion:** `predictions/m_nu2.py`, `predictions/m_nu3.py`, `predictions/m_nu3_derivation.md`

## 1. Abstract

m_ν₂ inherits from m_ν₃ via the theorem-grade Ihara R-splitting:

$$m_{\nu_2} \;=\; m_{\nu_3} \,/\, \sqrt{R}, \quad R = \frac{228}{7}$$

with $m_{\nu_3} = (k^* \times N_{\text{atoms}}) \times M_{\text{Pl}} \times N_{\text{hub}}^{-1/2}$ from `predictions/m_nu3_derivation.md` (UNIQUE-THEOREM-GRADE-CONDITIONAL).

This supersedes the older chain that anchored to a Pati-Salam GUT-scale bare neutrino mass (ADOPTED-PS-SCALE) and applied a Class-1 Feshbach correction. The new chain uses ZERO adopted inputs — all factors derive from substrate primitives.

**Result:** m_ν₂ ≈ 8.86 meV.
**Observed:** m_ν₂ = √Δm²₂₁ = 8.654 ± 0.110 meV (NuFIT 6.0, normal ordering).
**Deviation:** +0.21 meV (+2.40%, +1.91σ_PDG) — Clause 8 FAIL vs σ_PDG alone.

## 2. Framework axioms invoked

Inherits from m_ν₃ derivation: A1 (toggle alphabet), A2 (MDL canonicalization), A5(a) (mass clause).

R = 228/7 derives independently from the Ihara 5-step Chebyshev recurrence on the trivalent NB graph at q = k* − 1 = 2 (cubic identity q³ = 5q − 2; see `predictions/R_nu_splitting.py` and `docs/parameters/R_theorem.md`).

## 3. Derivation

### Step 1 — m_ν₃ from global spectral-gap formula

By `predictions/m_nu3.py` (UNIQUE-THEOREM-GRADE-CONDITIONAL per `predictions/m_nu3_derivation.md`):

$$m_{\nu_3} \;=\; (k^* \times N_{\text{atoms}}) \times M_{\text{Pl}} \times N_{\text{hub}}^{-1/2}$$

For srs (k* = 3, N_atoms = 4, M_Pl = 1.22089 × 10¹⁹ GeV, N_hub = 8.395 × 10⁶⁰): m_ν₃ ≈ 50.57 meV.

**Type 4** (upstream `predictions/m_nu3.py`).

### Step 2 — R = 228/7 splitting (theorem-grade Ihara)

By `predictions/R_nu_splitting.py` (closed form via Ihara 5-step Chebyshev):

$$R = \frac{\Delta m^2_{31}}{\Delta m^2_{21}} = \frac{2}{\sin^2(5\varphi)} - 4 = \frac{228}{7}$$

where $\varphi = \arctan(\sqrt{7})$ encodes the trivalent NB cubic identity at q = 2.

**Type 4** (upstream `predictions/R_nu_splitting.py`).

### Step 3 — Definition of R with m_ν₁ = 0

With m_ν₁ = 0 (NuFIT normal-ordering convention, lightest-massless):

$$R = \frac{\Delta m^2_{31}}{\Delta m^2_{21}} = \frac{m_{\nu_3}^2 - m_{\nu_1}^2}{m_{\nu_2}^2 - m_{\nu_1}^2} = \frac{m_{\nu_3}^2}{m_{\nu_2}^2}$$

**Type 2** algebra (definition of R given m_ν₁ = 0).

### Step 4 — Solve for m_ν₂

$$m_{\nu_2} = \frac{m_{\nu_3}}{\sqrt{R}}$$

**Type 2** algebra.

## 4. Result

$$m_{\nu_2} \;=\; \frac{(k^* \times N_{\text{atoms}}) \times M_{\text{Pl}} \times N_{\text{hub}}^{-1/2}}{\sqrt{228/7}}$$

Numerical evaluation:

$$m_{\nu_2} \;\approx\; \frac{50.57 \,\text{meV}}{\sqrt{32.5714}} \;\approx\; 8.86 \,\text{meV}$$

## 5. Comparison with experiment

| quantity | predicted | observed (NuFIT 6.0) | deviation | σ_PDG |
|---|---|---|---|---|
| m_ν₂ | 8.86 meV | 8.654 ± 0.110 meV | +0.21 meV (+2.40%) | +1.91σ_PDG |

**Clause 8 (σ_PDG only):** Deviation +2.40% = +1.91σ_PDG ⇒ **FAIL** against σ_PDG alone.

## 6. Open questions

1. **R-splitting next-to-leading.** The 228/7 closed form captures the leading Chebyshev recurrence. Sub-leading corrections at O(α₁) ~ 4% level are not currently derived; their inclusion could refine the m_ν₂ prediction.

2. **m_ν₂ via direct global formula.** The current chain goes m_ν₂ = m_ν₃ / √R. An alternative would be a direct global formula for m_ν₂ paralleling m_ν₃'s. Whether such a formula exists (with what prefactor) is open.

3. **Comparison to old chain.** The retracted ADOPTED-PS chain matched at -0.10σ_PDG (better than +1.91σ_PDG here). The new chain trades empirical precision for fewer adopted inputs. Whether this is the right tradeoff is a methodology question — per the framework's "PDG is never the metric of robustness" rule (`feedback_pdg_leverage_stratification.md`), structural cleanness (zero adoptions) takes precedence.

4. **m_ν₁ = 0 status.** Currently observational convention (NuFIT normal-ordering); structural derivation retracted under B6 (R-15). m_ν₂ central value depends on this convention; if m_ν₁ ≠ 0 is preferred by future data, R = m_ν₃²/m_ν₂² needs to be replaced by R = (m_ν₃² − m_ν₁²)/(m_ν₂² − m_ν₁²) with m_ν₁ as additional input.

## 7. References

### Framework upstream
- `predictions/m_nu3.py`, `predictions/m_nu3_derivation.md` — m_ν₃ via global spectral-gap formula (UNIQUE-THEOREM-GRADE-CONDITIONAL).
- `predictions/R_nu_splitting.py`, `docs/parameters/R_theorem.md` — R = 228/7 (theorem-grade Ihara).
- `predictions/k_star.py`, `predictions/g_girth.py`, `predictions/alpha_1.py`, `predictions/N_hub.py` — substrate primitives.

### External
- NuFIT collaboration (2024). Three-flavor neutrino oscillation analysis, NuFIT 6.0. http://www.nu-fit.org.

## Audit v2 status

Inherits m_ν₃'s Clause 7 status (UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 + N_hub anchor + G_sub Drude). The R-splitting itself is theorem-grade independent.

**Clause 8 (σ_PDG only):** Deviation +2.40% = +1.91σ_PDG. **FAIL.**

**Combined status:** THEOREM-GRADE-CONDITIONAL (inherits m_ν₃'s UNIQUE-THEOREM-GRADE-CONDITIONAL + R-splitting theorem-grade); Clause 8 FAIL against σ_PDG. Further work (R-splitting next-to-leading or direct m_ν₂ formula) could close the +2.4% gap.
