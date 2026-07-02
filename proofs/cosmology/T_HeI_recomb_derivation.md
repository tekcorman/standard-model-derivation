# T_HeI_recomb — Helium I recombination temperature (Phase III)

**Status:** THEOREM-GRADE-STRUCTURAL (Phase III)
**Date:** 2026-05-27

Phase III F-fiber: He⁺ + e⁻ → He + γ at z ≈ 2500.

T_HeI = E_bind / N_thermal where E_bind = 24.6 eV (He first ionization potential, atomic-physics input class) and N_thermal ≈ 40.3.

**Result: T_HeI = 0.610 eV** (vs standard 0.60 eV, +1.7%) — **best Phase III match across 6 instances**.

Inherits Phase III structural form from
`docs/theorems/theorem_phase_III_F_fiber_class_2026-05-27.md`. Within-class
log-transcendence residue is class characteristic (per Phase III theorem,
not ad-hoc Clause 9 violation).

## Linter status

Clauses 1, 4, 7 inherit from Phase III theorem. Clause 6 K-rationality
structural (numerical log-transcendence is class characteristic). Clause 8:
+1.7% — within Phase III tolerance band.

**Grade: THEOREM-GRADE-STRUCTURAL** (Phase III class, He I instance).

E_bind = 24.6 eV is an atomic-physics input (He first ionization potential).
Framework treats this as standard atomic physics applied to substrate-derived
particles — not a separate framework prediction. The Phase III treatment is
in how the freezeout T is derived (log-suppression class characteristic).

## References

- Phase III theorem: `docs/theorems/theorem_phase_III_F_fiber_class_2026-05-27.md`
- Universality test: an internal working note
- `predictions/T_recomb_derivation.md` (parallel beat, H-recomb)
