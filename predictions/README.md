# predictions/ — per-parameter derivation outputs

This folder is the authoritative current output of the framework's rigor pass. Each tracked parameter has up to two files:

| File | Role |
|------|------|
| `<parameter>.py` | Computational prediction script. Produces a numerical value via the framework's machinery, no fitted constants. Importable; cached via `lru_cache`. |
| `<parameter>_derivation.md` | Human-readable derivation: axioms invoked, step-by-step argument, grade annotations, dependency chain, deviation from PDG. |

A parameter with both files has a derivation that has been written up at journal-grade rigor. A parameter with only the `.py` file is computed but the prose write-up is still pending (or has been retracted — see `retracted/`). A parameter with neither has no `predictions/` entry yet; see `docs/parameters/target_parameters.md` for the canonical status table covering all ~93 tracked targets.

## How to read a `*_derivation.md`

Each derivation has the following structure:

1. **Header line** — parameter name + current grade (THEOREM-GRADE, STRICT-SOLID, A-grade, etc.). The grade reflects the strictest reading the file currently passes.
2. **Supersedes / Date** — pointer to any historical derivation this replaces, plus the date of the current write-up. Grades change as the rigor pass progresses; the date matters.
3. **Abstract** — one paragraph: what is derived, from what inputs, to what numerical value, with what σ vs. PDG.
4. **Framework axioms invoked** — explicit table mapping each load-bearing step to one of the five framework axioms (A1, A2, A3, A4, A5/A5(b)). See `docs/framework/framework_axioms.md` for the canonical axiom statement.
5. **Derivation** — sectioned proof. Every step is annotated with a parameter-linter gate type:
   - **Type 1: Axiom** — direct invocation of A1–A5.
   - **Type 2: Algebra (CAS-verified)** — symbolic computation, with sympy/CAS verification reference.
   - **Type 3: Cited theorem** — published mathematical result with author + year.
   - **Type 4: Upstream closed file** — reduces to another `predictions/*` file or a closed `docs/theorem_*.md`.
6. **Deviation from observation** — predicted vs. PDG with σ pull (where applicable).
7. **Open questions / future closure** — what would tighten the grade further, if anything.

The four gate types are the heart of the rigor pass — see `docs/parameters/parameter_linter.md` for the full discipline.

## Grades

Roughly, in decreasing rigor:

| Grade | Meaning |
|-------|---------|
| **THEOREM-GRADE** | Every load-bearing step passes the parameter-linter gate. Zero adopted residuals. The number is forced by the axioms and stated mathematics. |
| **STRICT-SOLID** | All steps pass the gate but one or more depend on a clearly-named open structural lemma. Closes to theorem-grade once the lemma is proven. |
| **A / A−** | Complete derivation chain with one identifiable assertion. The assertion is documented and tracked for promotion. Typical A− cases: an extraction-map convention chosen by post-hoc match to observation. |
| **🟡 IN PROGRESS** | Prediction file exists but a known structural gap blocks promotion (e.g., V_ub awaiting V_cb concatenation closure). |
| **BLOCKED** | Prediction script exists, value is computed, but a foundational gap prevents claiming any current grade. |

The canonical per-parameter status table — including ✅/🟡/❌/⚙️/🔬 status flags across all 93 tracked targets — is `docs/parameters/target_parameters.md`. Grades in individual derivations may lag or lead it during a rigor-pass session.

## Verifying

```bash
python3 verify.py                              # backbone proof suite (26 proofs, ~10s)
python3 run_predictions.py                     # regenerate predicted_parameters.md table
python3 predictions/V_us.py                    # run a single prediction script
```

`verify.py` exercises a curated set of theorem-grade derivations across foundations / gauge / flavor / masses / cosmology / parity / Lorentz. `run_predictions.py` pulls every prediction script's output into `predicted_parameters.md` for the canonical PDG-comparison table.

## Notes on specific subdirectories

- `retracted/` — derivations that produced a number but failed the rigor pass on re-audit. Kept for historical visibility, not for current claims.

## Cross-references

- `docs/parameters/target_parameters.md` — canonical status list of all 93 tracked targets.
- `docs/framework/framework_axioms.md` — A1–A5 canonical statements.
- `docs/parameters/parameter_linter.md` — four-gate rigor discipline.
- `docs/honest_assessment.md` — what is and isn't proven, what would falsify the framework.
- `proofs/` — supporting computations and theorem files; some predictions/`*.py` files are thin wrappers over `proofs/` machinery.
