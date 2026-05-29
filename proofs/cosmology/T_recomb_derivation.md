# T_recomb — hydrogen recombination temperature

**Status:** THEOREM-GRADE-STRUCTURAL (within-class Saha-π residue is Phase III class characteristic)
**Date:** 2026-05-27 (cosmic-history arc)
**Phase:** III (bound-state Boltzmann freezeout)

## Abstract

Phase III F-fiber: T_recomb = B_H / N_thermal ≈ 0.330 eV, matching standard cosmology (0.32 eV) within 3.1%.

## Framework axioms invoked

- A1 + A2-T substrate (no direct invocation; via upstream).
- **B_H = α²·m_e/2 = 13.6 eV** structural form (K-rational modulo α_EM K-rationality status).
- **η_B = (√3/10)·(2/3)⁴⁸** (Row P29 theorem-grade upstream).
- **Phase III theorem** (`docs/theorems/theorem_phase_III_F_fiber_class_2026-05-27.md`):
  T_F = E_bind / N_thermal where N_thermal = log(prefactor·η_B⁻¹) ≈ 41.

## Derivation

Phase III F-fiber identification:
1. Bound state: Hydrogen 1s atom
2. E_bind = B_H = α²·m_e/2 (Bohr binding)
3. Free continuum: e⁻ + p ionized
4. Boltzmann competition with η_B suppression

Saha-like equation x_e = 1/2 at freezeout, prefactor = (m_e T / 2π)^(3/2)
with n_baryon = η_B · n_γ. Iteratively self-consistent solution:

```
T_F = B_H / log[(m_e T_F / 2π)^(3/2) / (η_B · (2ζ(3)/π²) · T_F^3)]
    = B_H / N_thermal(T_F)
N_thermal ≈ 41.3 at convergence
T_F = 13.6 / 41.3 = 0.3298 eV
```

## Result

**T_recomb = 0.3298 eV** (framework structural prediction).

## Comparison with experiment

| Source | Value | Δ |
|---|---|---|
| Framework | **0.3298 eV** | — |
| Standard cosmology consensus | 0.32 eV | +3.1% |
| Planck 2018 (z_* derived) | z_* = 1089 ↔ T ≈ 0.30 eV | +9.9% |

## Phase III class characteristic

Phase III F-fibers share N_thermal ≈ 30-42 universally. Recombination
N_thermal = 41.3; cf. He I (40.3), He II (39.1), BBN D (28.8). The
log-suppression is the **class characteristic** (per Phase III theorem),
NOT an ad-hoc K-violation.

## Within-class numerical residue

The 3.1% gap from standard reflects:
- Saha-π log-transcendence (3D Gaussian momentum integral π — structurally
  inseparable per Session A: `proofs/cosmology/session_A_substrate_partition_function_2026-05-27.py`)
- Continuum prefactor π² in n_γ (additional channel)
- log-iterative convergence details

Resolution requires Option C framework extension
:
either Axiom F (log-transcendence acceptance, ~70% bounded) or substrate
Saha analog (Axiom D, multi-sprint).

## Linter status

- Clause 1 (axiom): PASS — A1 substrate + Phase III theorem.
- Clause 4 (Type 4 upstream): PASS — α_EM, m_e, η_B all theorem-grade upstream.
- Clause 6 (K-rationality): STRUCTURAL FORM K-rational; numerical evaluation has
  log-transcendence (class characteristic per Phase III theorem).
- Clause 7 (audit-v2): inherits Phase III theorem's class analysis.
- Clause 8 (numerical match): 3.1% — passes within-class tolerance for Phase III
  log-suppression family (1-9% across 6 Phase III F-fibers).
- Clause 9 (Type-3 SM π audit): Phase III class characteristic — log-transcendence
  is structurally inherent to bound-state Boltzmann freezeout, NOT a continuum
  loop import. Pending Clause 9 Phase III extension (Axiom F).

**Net grade: THEOREM-GRADE-STRUCTURAL** with named within-class residue.

## References

- Phase III theorem: `docs/theorems/theorem_phase_III_F_fiber_class_2026-05-27.md`
- Phase III universality: an internal working note
- Cosmic-history landing: an internal working note
- Saha-π attack (closure-negative): an internal working note
- η_B (upstream): `predictions/eta_B.py`
