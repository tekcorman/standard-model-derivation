# docs/

Documentation for the framework. Start here:

- **[orientation.md](orientation.md)** — cold-start doc covering the directory layout, file-type conventions, rigor machinery (gates, ledgers, residue register, citation validator, wave engine), and how a derivation moves through the lifecycle. Read this first if you are new to the repo or returning after a break.
- **[../README.md](../README.md)** — top-level claim + headline results.
- **[quickstart.md](quickstart.md)** — five-minute "show me one result" intro (R = 228/7 from the Ihara zeta of K₄).
- **[honest_assessment.md](honest_assessment.md)** — what is proven, what is adopted, what is open, what would falsify.
- **[master_plan.md](master_plan.md)** — canonical priority queue, axiom slate, framework state.
- **[north_star.md](north_star.md)** — the finish-line goal: what "done" means (a complete, derived CSCO). Judge any "what to do next" call against it.

## Subdirectory map

| Directory | Contents | Entry point |
|---|---|---|
| `framework/` | Axioms, layered architecture, narrative spine, ontology, scheme conventions, observable/particle catalogues. The framework's definitional content. | `framework/framework_axioms.md` |
| `theorems/` | Closed theorem statements + proofs (~92 files as of 2026-05-26). One file per theorem; each is the canonical reference for downstream `predictions/` to cite under Type-3 / Type-4 gates. | `theorems/` (browse by name) |
| `forward_constructions/` | Constructive bridges from substrate to QFT objects (substrate Wick, propagator, LSZ, Wightman, RG, Lichnerowicz, Atiyah-Singer, modular structure, etc.). Tier-1+ machinery that the theorems cite. | `forward_constructions/` |
| `operator_sweep/` | The foundational operator catalog (`operator_sweep_from_A1.md`) plus per-layer audits. Defines what operations are permitted at each layer between A1 and observables. | `operator_sweep/operator_sweep_from_A1.md` |
| `parameters/` | Per-parameter status (`target_parameters.md`), the parameter linter, the parameter uniqueness ledger, DAG chains, cross-cutting parameter docs (R_theorem, parity_theorems, derivations, predictions). | `parameters/target_parameters.md` |
| `audits/registers/` | Live, load-bearing audit registers: structural uniqueness ledger, structural residue register, adoption register. Updated on every closure. | `audits/registers/uniqueness_ledger.md` |
| `wave_engine/` | The 219-operation derivation-engine catalog and simulator docs. | `wave_engine/README.md` |

## Conventions in one paragraph

A `docs/theorems/theorem_*.md` file is a closed theorem statement, citeable under the parameter linter's Type-3/Type-4 gates. A `docs/forward_constructions/forward_construction_*.md` file is a constructive bridge from the substrate to a standard QFT object. A `predictions/<name>.py` + `predictions/<name>_derivation.md` pair is a per-parameter prediction; the `.md` file is the journal-grade write-up. A `docs/audits/registers/*.md` file is a live audit register that every new closure must consult and update.

For the full picture, see [orientation.md](orientation.md).
