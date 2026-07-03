# Repo orientation

A cold-start guide to the layout, conventions, and rigor machinery of this repository. Read this once when you arrive or return after a long break; everything else can be navigated from here.

For the framework's *content* — what is claimed, what is derived, what is open — see [`../README.md`](../README.md), [`honest_assessment.md`](honest_assessment.md), and [`parameters/target_parameters.md`](parameters/target_parameters.md). For the honest σ_PDG count and the **combined math+physics frontier state + framing discipline** (read before working on mass/flavor — it records the recurring missteps), see [`parameters/honest_sigma_count_2026-06-22.md`](parameters/honest_sigma_count_2026-06-22.md) and an internal working note. This document is about the *organization*.

---

## 1. What this repo is

A derivation of Standard Model parameters from three irreducible commitments — (A) self-containment of the universe (metaphysical), (B) finite observer (scoping), (I) active reading of binary distinctions (interpretive) — plus one empirical labeling rule (A5-mass: which substrate eigenvalues correspond to which SM masses), plus standard published mathematics. Under (A)+(B)+(I), the observer's primitive update is forced to be a binary self-inverse toggle generating F_inv(E) (`docs/theorems/theorem_toggle_from_self_containment.md`, 2026-05-07) — the content previously postulated as the axiom A1. In honest σ_PDG (no σ_theory widening), **44** of 125 tracked targets match observation within 1σ_PDG (the 2026-06-22 count of 39, plus m_t/m_b/g_2, closed 2026-06-25 by the forced dark self-energy and the scheme-consistent g_2 target; plus Γ_Z/M_Z/Γ_W/Γ_Z, closed 2026-07-02 by the derived EW radiative width layer — pre-registered, blind, bridge-conditional grade) and **~19** are forced/exact structural identities; **9** are open gaps (>1σ_PDG, channel-structured and located — see `docs/incomplete_equations_todo.md`), 2 are open ppm-scale misses (m_e −70.3 ppm, m_μ −60.5 ppm relative — the un-derived subleading, logged open), 8 framework-vs-ΛCDM coasting. (The earlier "~90 closed at theorem grade" conflated numerical matches with structural identities and soft-failers; corrected 2026-06-22.) Audit infrastructure is designed to make curve-fitting impossible. The canonical comparison table is the auto-generated `predicted_parameters.md` at the repo root.

---

## 2. Directory layout

```
README.md                 ← claim, headline results, what to run
predicted_parameters.md   ← auto-regenerated PDG-comparison table (gitignored anyway)
verify.py                 ← runs the backbone proof suite (65 proofs, ~60 s)
run_predictions.py        ← regenerates predicted_parameters.md

predictions/              ← THE DAG. Per-parameter (script + derivation) pairs.
  <name>.py               ← computational prediction. Imports only from other predictions/, stdlib, and approved third-party.
  <name>_derivation.md    ← journal-grade write-up. Linter-gated.
  retracted/              ← derivations that failed re-audit. Kept for honest history.
  README.md               ← contract + grade scale
  _validate_dag.py        ← run before commit; checks import discipline

proofs/                   ← proof scripts and exploratory machinery, organized by sector.
  common.py               ← shared lattice infrastructure
  foundations/            ← k*=3, srs, generations, Hashimoto, Ramanujan, branch measure μ
  flavor/                 ← CKM, PMNS, V_us / V_cb, CP phases
  masses/                 ← hierarchies, Koide, neutrino masses
  cosmology/              ← Λ, Ω_DM, η_B, n_s, H_0, w_DE
  gauge/                  ← Cl(6), Pati-Salam, R-parity
  lorentz/                ← Lorentz invariance, dispersion symbolic verification
  wave_engine/            ← 219-op simulator + audit-pilot scripts

explorations/             ← hypothesis-tested-and-archived scripts (successes and failures)

docs/                     ← documentation. See §3.

scripts/validate_citations.py  ← citation validator; pre-commit hook candidate
```

### Inside `docs/`

```
docs/
  README.md               ← thin index to subdirs
  orientation.md          ← this file
  quickstart.md           ← 5-minute intro (R = 228/7 demo)
  honest_assessment.md    ← what is and isn't proven
  master_plan.md          ← canonical priority queue + framework state

  framework/              ← axioms + architecture + narrative + ontology
  theorems/               ← ~92 closed theorem statements
  forward_constructions/  ← 15 constructive bridges (substrate → QFT objects)
  operator_sweep/         ← operator catalog + 7 per-layer audits
  parameters/             ← parameter linter, target list, DAG chains, cross-cutting
  audits/
    registers/            ← live: uniqueness_ledger, residue_register, adoption_register
  wave_engine/            ← simulator + cost methodology
```

## 2.1 Relevance tiers — what is load-bearing

Not every file carries equal weight. This is the map of current vs supporting
vs historical:

| Tier | What | Size |
|------|------|------|
| **Executable core** | `verify.py` + the `predictions/` DAG + the `proofs/` modules they import | ~115 `.py` — this is the *product* |
| **Probe archive** | `proofs/` — standalone verification scripts, cited by docs as evidence | 800+ scripts, tiered current / archive-only / orphan in [`proofs/README.md`](../proofs/README.md) |
| **Live docs** | `theorems/`, `predictions/*_derivation.md`, `framework/`, `forward_constructions/` | cross-linked from the entry docs |
| **Archive by design** | `predictions/retracted/` | honest record of failed re-audits; not load-bearing |

Relevance here is *computed*, not asserted: a `proofs/` script counts as
"current" only if a non-archived doc cites it (method in `proofs/README.md`); a
doc is live only if it is reachable from the entry docs. When in doubt about a
file, trace it — don't assume.

---

## 3. File-type conventions

Every doc and script in the repo falls into one of these types. Identifying the type tells you how citeable it is and what discipline applies.

| Type | Pattern | Citeable as | Lifecycle |
|---|---|---|---|
| **Framework definition** | `docs/framework/framework_*.md` | Type-1 (axiom) input | Stable; rarely edited |
| **Closed theorem** | `docs/theorems/theorem_*.md` | Type-3 (cited theorem) or Type-4 (upstream) | Once closed, stable. May be retracted if a re-audit falsifies it. |
| **Forward construction** | `docs/forward_constructions/forward_construction_*.md` | Type-3 / Type-4 (depending on grade) | Constructive bridge; cited by theorems and predictions |
| **Operator-sweep entry** | `docs/operator_sweep/operator_sweep_*.md` | Type-3 reference for "what operations are permitted at layer N" | Stable catalog |
| **Audit register** | `docs/audits/registers/*.md` | Updated on every closure; row entries cited by predictions | Living document |
| **Prediction script** | `predictions/<name>.py` | Type-4 upstream input for downstream predictions | Edited or retracted |
| **Prediction derivation** | `predictions/<name>_derivation.md` | The journal-grade artifact | Linted; grade reflects current rigor |
| **Proof script** | `proofs/<sector>/<name>.py` | Verification machinery; not directly cited as upstream | Edited, refactored, sometimes promoted to a `predictions/*.py` |
| **Retracted prediction** | `predictions/retracted/<name>.py` + `_derivation.md` | **Not citeable** | Honest record of failed re-audits |

---

## 4. The rigor machinery

The framework is held together by seven interlocking instruments (plus the DAG-chain map), each watching a different way a derivation can cheat. **Every new prediction or theorem must pass through all of them.**

### 4.1 Parameter linter — four-gate discipline

Canonical: [`parameters/parameter_linter.md`](parameters/parameter_linter.md).

Every load-bearing step in a `predictions/<name>_derivation.md` must be one of:

- **Type 1 — Foundation.** Direct invocation of A5-mass (empirical labeling), or of A1 / A2 / A3 / A4 / P1' as foundational theorems (post-2026-05-08 these are derived from (A)+(B)+(I) per [`framework/framework_axioms.md`](framework/framework_axioms.md); downstream files cite them by their stable A-names).
- **Type 2 — Algebra (CAS-verified).** Symbolic computation with sympy/CAS verification reference inline.
- **Type 3 — Cited theorem.** Published mathematical result with author + year + page.
- **Type 4 — Upstream closed file.** Reduces to another `predictions/*` file or a closed `docs/theorems/theorem_*.md`.

The linter's `predictions/` DAG contract: if every directory except `predictions/` were deleted, predictions would still run. Run `predictions/_validate_dag.py` before commits.

A grade higher than **THEOREM-GRADE** (every step gates) requires every load-bearing step to pass these four gates with zero adopted residuals. Lower grades — STRICT-SOLID, A, A−, IN-PROGRESS, BLOCKED — mark named gaps that are tracked for promotion.

### 4.2 Structural uniqueness ledger

Canonical: [`audits/registers/uniqueness_ledger.md`](audits/registers/uniqueness_ledger.md).

For every load-bearing *structural* claim (alphabet size = 2, lattice = srs, dimension = 3, gauge algebra = Cl(6), …), the ledger names the operator-permitted alternative set, the selection criterion, and the resulting status:

- **UNIQUE** — every alternative strictly eliminated.
- **DOMINANT** — alternative set non-empty but framework's choice is the MDL minimum with strictly positive margin.
- **ONE-AMONG-MANY** — multiple alternatives clear simultaneously; framework retains the multiplicity (e.g., chirality both-hands above the A2 waterline) or arbitrarily picks one (a flagged gap).

Flags: **CONDITIONAL** on an upstream row, **GAP** at this row's own layer.

### 4.3 Parameter uniqueness ledger

Canonical: [`parameters/parameter_uniqueness_ledger.md`](parameters/parameter_uniqueness_ledger.md).

Same audit applied to *numerical* claims (V_us = 9/40, α₁ = 256/6305, …). Identifies operator-permitted alternative formulas producing the same observable; selection criterion; status.

### 4.4 Structural residue register

Canonical: [`audits/registers/structural_residue_register.md`](audits/registers/structural_residue_register.md).

The framework's MDL reading retains *every* above-waterline alternative simultaneously, not just the dominant one. Soft-gated structural alternatives (eliminated by a finite MDL margin rather than an exact algebraic zero) carry non-zero Boltzmann-style weight and may produce downstream artifacts. Each such residue is registered as **R-N** with status:

- **OPEN** — suspicion only.
- **TRACED** — downstream signature estimated; consistent with current data.
- **ACCOUNTED-FOR** — the residue *is* an existing framework derivation (catalogued as a worked example).
- **REFUTED** — downstream check ruled out the alternative.

**Discipline rule:** every new prediction or theorem must consult this register and either include the residue's contribution or argue why it is hard-gated for that observable.

### 4.5 Adoption register

Canonical: [`audits/registers/adoption_register.md`](audits/registers/adoption_register.md).

Tracks every "ADOPTED-X" label — places where the framework adds an empirical or definitional input that is not derived from the (A)+(B)+(I)+A5-mass foundation. Each adoption has scope, justification, downstream consumers, and a graduation path. Adoptions get *graduated* to theorem grade when the structural derivation closes.

### 4.6 Citation validator

Canonical: [`../scripts/validate_citations.py`](../scripts/validate_citations.py).

Operational discipline tool. Scans `predictions/*` + `docs/theorems/theorem_*.md` + `docs/forward_constructions/*.md` for citations to upstream theorems / ledger rows / residue register entries / operator catalog ops. Run with `--strict` as a pre-commit hook to fail commits that introduce uncited claims.

### 4.7 Wave engine

Canonical: [`wave_engine/README.md`](wave_engine/README.md), [`../proofs/wave_engine/simulator.py`](../proofs/wave_engine/simulator.py).

A 219-operation derivation-engine catalog. Every operation is gated by the audit machinery above. Open-frontier tags = ∅ post-2026-04-27 LORENTZ_SIG/CCLOSE→NC_GEOM joint closure. The wave engine is the framework's structural completeness check: if any operation cannot fire from upstream, that's a structural gap.

### 4.8 Value-lock harness (added 2026-07-01)

Canonical: [`../scripts/value_lock.py`](../scripts/value_lock.py) + [`../predictions/_value_locks.json`](../predictions/_value_locks.json).

Every live predicted value in the `predictions/` DAG is pinned (101 values at first freeze; 104
after the 2026-07-02 width-layer registration's deliberate re-freeze) and
re-checked in CI after the backbone proofs. A predicted value can only change through a deliberate
`--freeze`, which shows up in review as a lock-file diff alongside the derivation change. This makes
silent value drift — prose lagging code, or code moving without anyone noticing — mechanically
impossible. (Motivating incident: the m_H docstring/live divergence found and fixed 2026-07-01.)

### 4.9 DAG chains

Canonical: [`parameters/parameter_DAG_chains.md`](parameters/parameter_DAG_chains.md).

Per-parameter dependency chains. Names every upstream prediction file and every cited theorem. Used to compute "what closes when X closes" and to surface the small set of fundamental commitments at the leaves (~29: 25 structural ledger rows + 4 master-theorem class theorems).

---

## 5. Lifecycle of a derivation

```
   idea
    ↓
  exploration (proofs/<sector>/*.py or explorations/*.py)
    ↓
  proof script + theorem doc (proofs/<sector>/*.py + docs/theorems/theorem_*.md)
    ↓
  prediction file pair (predictions/<name>.py + <name>_derivation.md)
    ↓
  parameter linter pass — Type 1/2/3/4 gates fire on every step
    ↓
  ledger row added to parameter_uniqueness_ledger.md (and structural ledger if relevant)
    ↓
  residue register consulted; downstream residues registered if applicable
    ↓
  citation validator passes
    ↓
  wave engine op fires (if applicable)
    ↓
  THEOREM-GRADE
    ↓
  [if a re-audit later falsifies it]
    ↓
  retract: move to predictions/retracted/, mark register row, update target_parameters.md
```

The path is not linear; predictions can sit at lower grades (STRICT-SOLID, A, A−, IN-PROGRESS, BLOCKED) for a long time with named gaps.

---

## 6. Where to look for X

| Question | Answer |
|---|---|
| What does this framework claim? | [`../README.md`](../README.md) |
| Show me one result in 5 minutes. | [`quickstart.md`](quickstart.md) |
| What's actually proven? | [`honest_assessment.md`](honest_assessment.md) |
| Status of every parameter. | [`parameters/target_parameters.md`](parameters/target_parameters.md) |
| What are the axioms? | [`framework/framework_axioms.md`](framework/framework_axioms.md) |
| What's the layered architecture? | [`framework/framework_architecture.md`](framework/framework_architecture.md) |
| What's the conceptual story? | [`framework/narrative_spine.md`](framework/narrative_spine.md) |
| What operations are permitted from A1 alone? | [`operator_sweep/operator_sweep_from_A1.md`](operator_sweep/operator_sweep_from_A1.md) |
| What's the priority queue? | [`master_plan.md`](master_plan.md) |
| Has X been audited? | Search [`audits/registers/uniqueness_ledger.md`](audits/registers/uniqueness_ledger.md), [`parameters/parameter_uniqueness_ledger.md`](parameters/parameter_uniqueness_ledger.md), [`audits/registers/structural_residue_register.md`](audits/registers/structural_residue_register.md) |
| Where's the prediction for X? | `predictions/<X>.py` + `predictions/<X>_derivation.md`; canonical pointer in [`parameters/target_parameters.md`](parameters/target_parameters.md) |
| Where's the theorem for X? | `docs/theorems/theorem_<X>.md` |
| Why was X retracted? | `predictions/retracted/<X>_derivation.md`; relevant ledger row will note the retraction |
| What's the rigor bar for adding X? | [`parameters/parameter_linter.md`](parameters/parameter_linter.md) |
| Run all backbone proofs. | `python3 verify.py` |
| Regenerate the PDG comparison table. | `python3 run_predictions.py` |
| Validate citations. | `python3 scripts/validate_citations.py --strict` |
| Validate the predictions/ DAG. | `python3 predictions/_validate_dag.py` |

---

## 7. Common tasks

### 7.1 Add a new prediction

1. Verify a target row exists in [`parameters/target_parameters.md`](parameters/target_parameters.md). If not, add it (do not delete; mark RETIRED if abandoned).
2. Write the proof in `proofs/<sector>/<name>.py` and exercise the framework machinery (Hashimoto walker, Bloch lift, Cl(6), etc.).
3. If a new theorem is required, add `docs/theorems/theorem_<name>.md` with full proof and citations.
4. Create the `predictions/<name>.py` with **only** stdlib + approved third-party + other `predictions/` imports. The DAG contract (§4.1) is hard.
5. Write `predictions/<name>_derivation.md` with header line (parameter + grade), supersedes/date, abstract, framework axioms invoked table, derivation sectioned with Type-1/2/3/4 gates per step, deviation-from-observation, open questions.
6. Run `python3 predictions/_validate_dag.py`.
7. Run `python3 scripts/validate_citations.py --strict <new-files>`.
8. Add a row to [`parameters/parameter_uniqueness_ledger.md`](parameters/parameter_uniqueness_ledger.md): claim, source, observed value, operations invoked, alternatives, selection, status, conditional dependencies, gaps, residue cross-refs.
9. Consult [`audits/registers/structural_residue_register.md`](audits/registers/structural_residue_register.md) for soft-gated alternatives affecting the target observable. Either include their contribution or argue why hard-gated.
10. Update [`parameters/target_parameters.md`](parameters/target_parameters.md): flip 🟡 → ✅, link the new file.
11. If applicable, update [`parameters/parameter_DAG_chains.md`](parameters/parameter_DAG_chains.md).

### 7.2 Retract a prediction

1. `git mv predictions/<name>.py predictions/retracted/<name>.py` and same for the derivation.
2. Add a "RETRACTED — date — reason" header at the top of the retracted derivation.
3. Update the relevant row in [`parameters/parameter_uniqueness_ledger.md`](parameters/parameter_uniqueness_ledger.md) — change status to GAP / OPEN with retraction note.
4. Update [`parameters/target_parameters.md`](parameters/target_parameters.md): flip ✅ → 🟡 / ❌ as appropriate.
5. If a residue register entry changes status, update [`audits/registers/structural_residue_register.md`](audits/registers/structural_residue_register.md).

### 7.3 Close a structural ledger row

1. The structural derivation ships as a `docs/theorems/theorem_*.md` file.
2. Update the row in [`audits/registers/uniqueness_ledger.md`](audits/registers/uniqueness_ledger.md): change status, drop GAP / CONDITIONAL flags, link the new theorem.
3. Any downstream rows whose CONDITIONAL flag pointed at this row may auto-graduate; check and update them.
4. Update [`audits/registers/structural_residue_register.md`](audits/registers/structural_residue_register.md) if any residues now close or change status.

---

## 8. Naming conventions

- **Theorems:** `theorem_<topic>.md` in `docs/theorems/`.
- **Forward constructions:** `forward_construction_<topic>.md` in `docs/forward_constructions/`.
- **Predictions:** `<observable>.py` and `<observable>_derivation.md`. Lowercase; underscores; observable name as written in physics (V_us, m_H, lambda_higgs, theta_23_PMNS).
- **Theorem-mode prediction wrappers:** `theorem_B<n>_<short>.py` (these wrap a closed theorem so it can be cited as Type-4 upstream from other predictions).
- **Audits:** live registers carry no date suffix.
- **No v2 suffixes.** Use the base filename; superseded predictions go to `retracted/`.

---

## 9. Framework discipline rules

These are durable rules that keep the derivation honest:

- **No post-hoc structural backfill** — structure must precede numerics; never construct a structural reason after a numerical match.
- **Walk the uniqueness auditor at every conclusion** — at every closure, walk the ledgers + residue register + operator sweep before claiming theorem grade.
- **Three-level hierarchy** — L1 toggles (μ), L2 srs graph (α₁), L3 Hashimoto/causal-observer (CKM). Don't search L2 quantities on the L3 graph or vice versa.
- **A2 is a waterline, not a strict optimum** — every above-threshold compression is retained simultaneously. Soft-gated alternatives populate the residue register.
- **Predictions/ folder policy** — theorem-grade or Feshbach pairs only. New entries require a parameter-linter pass before shipping.
