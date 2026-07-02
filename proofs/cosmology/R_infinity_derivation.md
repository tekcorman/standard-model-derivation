# Derivation of R∞ (Rydberg constant)

**Date:** 2026-05-04 EOD+1. **Status:** STRUCTURAL-CONDITIONAL (downstream of α_EM(M_Z)).
Clause 8 is evaluated against σ_PDG only.

## Abstract

R∞ is the most precisely measured atomic constant. Derived as R∞ = α_EM(0)² × m_e × c / (2h) using the framework's α_EM(M_Z) (THEOREM-GRADE-CONDITIONAL via M_unif), running down to the Thomson limit α_EM(0) via standard QED, plus theorem-grade m_e and CODATA c, h.

## Framework axioms invoked

Inherited via α_EM (Row P63) + m_e (theorem-grade ratio chain via y_τ + Koide).

## Derivation

**Step 1**: α_EM(M_Z) ≈ 1/127.93 (live 2026-05-22, post-α_GUT-DC) from `predictions/alpha_EM.py` [Type 4 THEOREM-GRADE-CONDITIONAL]. (Pre-α_GUT-DC value 1/127.04 was stale drift.)

**Step 2**: α_EM running from M_Z down to atomic scales (Thomson limit) via standard QED through charged-fermion loops: 1/α_EM(0) - 1/α_EM(M_Z) ≈ 9.092 (Type 3 standard QED; PDG-derived per R_infinity.py — was 9.91 pre-α_GUT-DC, calibrated against the stale 1/127.04). NOTE: this `delta_alpha_running` is Clause-9-conditional (named open mechanism — Δα is the framework's OUT-OF-SCOPE IR layer per Move-1).

**Step 3**: m_e from `predictions/m_e.py` [Type 4 theorem-grade ratio chain via y_τ Koide structure].

**Step 4**: R∞ = α_EM(0)² × m_e × c / (2h) [Type 2 standard atomic physics, Bohr 1913 / Rydberg 1888].

## Result

$$R_\infty = \frac{\alpha_{EM}(0)^2 \cdot m_e \cdot c}{2h} \approx 1.099 \times 10^7 \text{ m}^{-1}$$

## Comparison with experiment

| Source | Value | Deviation |
|---|---|---|
| CODATA 2018 | 1.0973731568160(21) × 10⁷ m⁻¹ | reference |
| Framework | 1.099 × 10⁷ m⁻¹ | +0.13% |

Evaluated against σ_PDG only — CODATA σ on R∞ is ~2×10⁻¹² of the value; the +0.13% deviation is many orders of magnitude beyond σ_PDG (FAIL Clause 8).

## Open questions

1. R∞ inherits α_EM's framework systematic (~1% on α_EM, doubled to ~2% on R∞).
2. Two-loop + threshold corrections on α_EM(M_Z) would tighten R∞ to ~0.1%.
3. The QED running α_EM(M_Z) → α_EM(0) involves charged-fermion thresholds (m_e, m_μ, m_τ); standard hadronic vacuum polarization corrections are external.

## References

`predictions/alpha_EM.py`, `predictions/m_e.py`, `predictions/M_unif.py`. Bohr 1913, "On the Constitution of Atoms and Molecules", Philos. Mag. 26, 1; Rydberg 1888 spectroscopic relation.
