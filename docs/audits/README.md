# Audits

Two kinds of artifact live here:

- **`registers/`** — *live* audit registers. Updated on every closure. Load-bearing: every new prediction or theorem must consult and (where applicable) update them.
- **`sessions/`** — *archived* session-arc audit artifacts (audit-v2, clause sweeps, follow-up roadmaps). Preserved for record. The conclusions are digested into the live registers; cite the register row, not the session file.

For the full picture of how the audit machinery interlocks with the parameter linter, the wave engine, and the citation validator, see [`../orientation.md`](../orientation.md) §4.

---

## Live registers

### [`registers/uniqueness_ledger.md`](registers/uniqueness_ledger.md) — structural pass

For every load-bearing structural claim (alphabet = 2, lattice = srs, dimension = 3, gauge algebra = Cl(6), …), the ledger names the operator-permitted alternative set, the selection criterion, and the resulting status: **UNIQUE** / **DOMINANT** / **ONE-AMONG-MANY**, with **CONDITIONAL** and **GAP** flags. ~25 rows.

### [`registers/structural_residue_register.md`](registers/structural_residue_register.md)

The framework's MDL reading retains *every* above-waterline alternative simultaneously, not just the dominant one. Soft-gated structural alternatives (eliminated by a finite MDL margin rather than an exact algebraic zero) carry non-zero Boltzmann-style weight and may produce downstream artifacts. Each is registered as **R-N** with status **OPEN** / **TRACED** / **ACCOUNTED-FOR** / **REFUTED**.

**Discipline rule:** every new prediction or theorem must consult this register and either include the residue's contribution or argue why it is hard-gated for that observable.

### [`registers/adoption_register.md`](registers/adoption_register.md)

Tracks every "ADOPTED-X" label — places where the framework adds an empirical or definitional input that is not derived from A1. Each adoption has scope, justification, downstream consumers, and a graduation path.

### Companion: parameter pass

The parameter-level companion to `uniqueness_ledger.md` lives in [`../parameters/parameter_uniqueness_ledger.md`](../parameters/parameter_uniqueness_ledger.md) — same audit applied to numerical claims (V_us = 9/40, α₁ = 256/6305, …).

---

## Session archive

See [`sessions/README.md`](sessions/README.md). 28 dated artifacts preserving working detail from the audit-v2 program (2026-04-30 → 2026-05-01) and adjacent clause sweeps. The conclusions are digested into the live registers above; cite those, not the session files.
