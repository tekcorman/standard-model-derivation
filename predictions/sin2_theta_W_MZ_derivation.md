# Derivation of sin²θ_W(M_Z)

**Date:** 2026-05-04 EOD+1. **Status:** THEOREM-GRADE-CONDITIONAL.

## Abstract

sin²θ_W(M_Z) by RG running from theorem-grade sin²θ_W(M_unif) = 3/8 down to the framework-derived M_Z, using one-loop MSSM β-functions.

## Framework axioms invoked

Inherited via sin²θ_W(M_unif) theorem (`docs/theorems/theorem_sin2_theta_W_unification.md`) + M_unif (Row P62) + M_Z (Row TBD).

## Derivation

**Step 1**: At unification, sin²θ_W(M_unif) = 3/8 [Type 4 theorem-grade per `predictions/sin2_theta_W.py`].

**Step 2**: At any scale, sin²θ_W = α_Y/(α_2 + α_Y) where α_Y = (3/5)α_1 [SU(5) embedding, Type 1].

**Step 3**: One-loop MSSM RG from α_GUT at M_unif to M_Z [Type 3, Peskin-Schroeder §16].

**Step 4**: Algebraic combination [Type 2].

## Result

$$\sin^2\theta_W(M_Z) = \frac{\alpha_Y(M_Z)}{\alpha_2(M_Z) + \alpha_Y(M_Z)} \approx 0.23125$$

## Comparison with experiment

| Source | Value | Deviation |
|---|---|---|
| PDG 2024 (on-shell) | 0.23121 ± 0.00004 | reference |
| Framework (live 2026-05-22, post-α_GUT-DC) | 0.23125 | +0.02% (+0.96σ_PDG, **PASS**) |
| Framework (pre-α_GUT-DC, stale) | 0.23027 | −0.41% |

Evaluated against σ_PDG only — Clause 8 **PASS** at the live (post-α_GUT-DC) value. The pre-α_GUT-DC value 0.23027 was stale drift; updated 2026-05-22 to match the live `gauge_unification_full_RG_closure.py` output.

## Open questions

Two-loop + threshold corrections would tighten to ~0.1%. Per parameter linter §2c, this parameter requires SM/MSSM RG running by definition; bridge convention does NOT apply.

## References

`predictions/sin2_theta_W.py` (M_unif theorem), `predictions/M_unif.py`, `predictions/M_Z.py`, `predictions/alpha_EM.py`.
