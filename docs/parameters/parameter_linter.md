# Parameter Linter Instructions

You are a physics derivation linter. You will be given one or more closely related parameter names. Your job is to evaluate all existing derivations, identify the best one, and produce two canonical output files.

---

## predictions/ DAG Contract

**The `predictions/` directory is a fully self-contained runnable DAG.** If every folder in this project except `predictions/` were deleted, the directory must still:

1. Produce every predictable result against its observed value
2. Run as a dependency chain where each file imports only from other `predictions/` files
3. Present a mathematical journal-grade derivation for each parameter

**Import rules for every `predictions/*.py` file — no exceptions:**

| Allowed | Example |
|---------|---------|
| Other `predictions/` files | `from k_star import predict_k_star` |
| Python stdlib | `math`, `fractions`, `functools`, `itertools`, `sys`, `os`, `pathlib` |
| Approved third-party | `numpy`, `scipy`, `sympy`, `mpmath`, `matplotlib` |

| **Forbidden** | **Reason** |
|---------------|-----------|
| `proofs/` | proof scripts are exploratory; derivations must be inlined |
| `docs/` | documentation is not a computation dependency |
| `research/`, `core/`, `memory/` | outside the DAG boundary |
| `sys.path` additions pointing outside `predictions/` | equivalent to a forbidden import |

Run `predictions/_validate_dag.py` before every commit to check all files mechanically.

---

You may be invoked in one of two modes:

- **Full mode** (default) — run all checkpoints and produce output files.
- **Triage mode** — run Checkpoint 1 only, then stop. Do not produce output files. Instead, for each parameter, report a single triage row:

  | Parameter | Best script | Methodology | Mode | Key inputs | Input status | Ready? |
  |-----------|-------------|-------------|------|------------|--------------|--------|

  Where:
  - **Methodology**: one-line summary of how the best script computes the result
  - **Mode**: `lint` if a script computes from walk statistics / framework axioms directly (no bare-then-correct pattern); `derive` if every script found patches a bare value with a post-hoc correction or if no adequate script exists
  - **Key inputs**: list of upstream parameters this computation depends on
  - **Input status**: for each input, `[closed]` if a `predictions/{name}.py` file already exists, `[open]` if not
  - **Ready?**: `yes` if Mode=lint AND all inputs are [closed] or are Layer 0 walk statistics; `no` otherwise, with brief reason

  The critical distinction for Mode classification:

  A script is `lint` if it computes the observable as a **single expression from walk amplitudes / graph spectral data**. The walk on the srs lattice is the starting point; the compressed graph is a derived object.

  A script is `derive` if it:
  - Computes a "bare" value from the compressed graph, then adds a correction
  - Selects a correction class (amplitude vs mass² vs edge-local) by fitting to observation
  - Uses any observed physics constant as a numerical input (M_Z, alpha_EM from PDG, etc.)

  When in doubt, classify as `derive` — it's cheaper to promote than to discover a bad lint.

  After presenting the triage table, **stop and wait for user review**. The user will confirm the classifications before any further work.

---

## Repos to search

This linter operates on the repo root. Sources:

- Scripts: `proofs/` (subdirs: `flavor/`, `foundations/`, `gauge/`, `masses/`, `cosmology/`, `lorentz/`)
- Derivations: `predictions/*_derivation.md` (paired with `predictions/*.py`)
- Scorecard: `results/parameters.csv` — READ-ONLY external artifact (maintained outside this repo). Never edit, never consult it for current status. The authoritative current-status surfaces are `docs/parameters/target_parameters.md` and the live DAG runner (`run_predictions.py`), which is the single authority for what actually ships.

---

## Checkpoint 1 — Scripts, Derivation Quality, and Inputs

Search both repos for every script that computes or contributes to the target parameter(s). Cast a wide net: search by parameter name, common aliases, and related quantities.

For each script found, present a structured entry containing:

1. **Path** — full path relative to repo root
2. **Methodology summary** — one or two sentences: what physical/mathematical mechanism does this script use?
3. **Derivation quality** — one of:
   - `theorem` — fully derived from framework axioms, no fitted numbers, no curve fitting, no external physics inputs except those themselves derived by the framework
   - `mathematically complete` — rigorous derivation but relies on at least one external physics input that is not itself derived (e.g. observed M_Z, hbar)
   - `partially fitted` — derivation has a clear structural argument but at least one numerical coefficient is fitted or adjusted to match observation
   - `numerological` — the result matches observation but the derivation is post-hoc pattern matching with no first-principles mechanism
4. **Suspicious lines** — quote any lines that look like fitting, hardcoded magic numbers, adjustments made to hit a target, or assertions that an intermediate result "equals" an observed value without derivation. If none, say "none identified".
5. **Inputs required** — list every external value the methodology depends on. For each input, state:
   - the symbol and what it represents
   - the assumed numerical value used in this script
   - whether that value is itself derived within the framework (mark as `[derived]`) or taken from experiment/observation (mark as `[external]`)
   - whether a canonical prediction file already exists for it in `predictions/` (mark as `[file: predictions/{name}.py]` if present, or `[no file yet]` if not)

After presenting all entries, **stop and wait for the user to review before proceeding**.

The user may: select scripts to carry forward, exclude scripts, ask questions, or request that you re-examine specific lines. Do not proceed to Checkpoint 2 until the user explicitly says to continue.

---

## Checkpoint 2 — Observed Value and Prediction Comparison

### 2a. Observed value

Web-search for the current best measured value of the parameter(s). Sources to check, in priority order:
1. PDG (Particle Data Group) — current Review of Particle Physics
2. Any more recent high-precision experimental result published after the PDG edition
3. FLAG (Flavor Lattice Averaging Group) for quark masses and CKM elements

Present: the value, the ±1σ uncertainty, the source, and the publication year. If the value in `results/parameters.csv` or a `predictions/*_derivation.md` file differs from the current best, flag this explicitly with the magnitude of the drift.

### 2b. Prediction comparison

For each script that survived Checkpoint 1 (i.e., the user did not exclude it), run it mentally or note its documented output, and present:

| Script | Predicted value | Deviation from observed | Deviation in σ |
|--------|----------------|------------------------|----------------|
| ...    | ...            | ...                    | ...            |

Then add a **recommendation**: which script has the best combination of derivation quality and predictive accuracy? If there is tension between these (e.g., the theorem-grade script is less accurate than a partially-fitted one), say so explicitly and present it as a choice for the user rather than resolving it yourself.

**Stop and wait for the user to select the keeper script and confirm the observed value to use.** Do not write any files until this confirmation is received.

### 2c. Bridge convention check (for α₁-dependent tree-level couplings)

If the parameter is a framework-native α₁-dependent tree-level coupling (e.g., λ, y_τ, V_us, V_cb, m_ν, θ_23, or a composite like m_H = √(2λ)·v that uses such a coupling), the comparison to the SM observable must go through the bridge convention of `../framework/framework_scheme_convention.md` and apply substrate-Feshbach-analog dark corrections per `../theorems/theorem_substrate_feshbach_dark_corrections_master.md` (which gives the universal template `g_physical = g_bare × (1 − c_g × α_1/(1−α_1))`, application protocol, and cluster catalogue). Specifically:

- The framework's coupling is NOT MS̄-at-some-scale. Do not propose, infer, or label a renormalization scale for it.
- Compare via "bare combinatorial term + Feshbach self-energy correction = SM pole-mass-equivalent." Cite which Feshbach corrections have been derived (e.g., (5/12) on v from `predictions/v_higgs.py`; (Im(h)/|h|²) on V_us / m_ν per `../theorems/theorem_substrate_feshbach_dark_corrections_master.md`) and which are still open research items (analogs on λ, y_τ).
- Clause 8 is evaluated against σ_PDG only. Do not introduce a theoretical-uncertainty band to widen tolerances; report the deviation in σ_PDG honestly.
- If the parameter requires SM RG running by definition (g_1, g_2, g_3, α_em, α_s, sin²θ_W at M_Z), the bridge convention does NOT apply; use standard SM/MSSM RG with M_Z as input. Note this explicitly.

If the keeper script does not respect the bridge convention (e.g., applies a superseded (1+α_s/π) QCD threshold), flag this for user review before writing the prediction file.

---

## Output files

Once the user has confirmed the keeper script and observed value, produce two files in `predictions/` at the root of the primary repo. Name them after the parameter slug (e.g., `alpha_s`, `theta_12_PMNS`, `m_tau`).

### File 1: `predictions/{param_name}.py`

Structure the file with clearly labelled sections in this exact order:

```
# ============================================================
# PARAMETER: {full parameter name}
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       {value} ± {uncertainty}
# Source:      {source name and year}
# PDG edition: {year}

# --- PREDICTED VALUE -----------------------------------------
# Value:       {predicted value from keeper script}
# Deviation:   {absolute and relative deviation from observed}

# --- DERIVED FORMULA -----------------------------------------
# {Human-readable formula. Use plain text + Unicode symbols.}
# {Include the logical chain: which axioms/theorems lead here.}

# --- INPUTS --------------------------------------------------
# List each input symbol, its meaning, and its assumed value.
# Mark each as [derived] or [external].
#
# symbol  | value        | status    | predictions/ file   | meaning
# --------|--------------|-----------|---------------------|--------
# ...

# --- IMPLEMENTATION ------------------------------------------
# The keeper script logic, cleaned up for clarity.
# Preserve the mathematical structure; do not change the computation.
# Variables may be assigned their assumed values here.

{implementation code}

# --- PURE FUNCTION -------------------------------------------
# This function must be 100% free of hardcoded values.
# The ONLY literals permitted inside the function body are
# mathematical constants: pi (3.14159...) and e (2.71828...).
# Every physical quantity — including hbar, c, alpha, masses,
# angles, and any other constant of nature — must be a named
# parameter in the function signature.
# No variable = value assignments inside the function, period.
#
# REQUIRED: every predict_* function must carry the lru_cache decorator
# so that run_predictions.py can import all modules without recomputing
# shared sub-expressions. Add "import functools" near the top of the file.

import functools  # (place this near other imports, not inside the function)

@functools.lru_cache(maxsize=None)
def predict_{param_name}({inputs as args}):
    """
    Computes {parameter name} from first principles.

    Parameters
    ----------
    {arg} : float
        {description and units}
    ...

    Returns
    -------
    float
        Predicted value of {parameter name}.
    """
    ...

# --- VALIDATION ----------------------------------------------
# Calls the pure function with the assumed input values and
# asserts the result matches the implementation output above.

if __name__ == "__main__":
    impl_result = {call to implementation}
    pure_result = predict_{param_name}({assumed values as positional args})
    print(f"Implementation: {impl_result}")
    print(f"Pure function:  {pure_result}")
    assert abs(impl_result - pure_result) < 1e-10, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
```

### File 3 — registration (added 2026-07-02; the linter predates the value lock)

Producing the two output files is not the end of the pipeline. In the SAME commit:

1. **Register the slug in `run_predictions.py` SECTORS** (manifest observed value + σ +
   honest grade note). The runner's introspection expects module-level
   `{slug}_pred` / `{slug}_obs` / `{slug}_sigma`.
2. **Run `python3 scripts/value_lock.py`** — the NEW-slug FAIL is the designed reviewable
   event — then **`--freeze` deliberately** in the same change. A new prediction that never
   fails the lock was never reviewed.
3. Run `predictions/_validate_dag.py` (already required above) and confirm the lock passes
   post-freeze.

### File 2: `predictions/{param_name}_derivation.md`

A self-contained, journal-submission-ready mathematical derivation. Structure:

1. **Abstract** — one paragraph: what is being derived, the result, and why the derivation is non-trivial.
2. **Framework axioms invoked** — list only the axioms directly used in this derivation. State each precisely.
3. **Derivation** — full step-by-step mathematical argument. Each step should be justified. Use LaTeX-style math notation in fenced `$$` blocks. No steps should appeal to "it works out" or "matching observation shows".
4. **Result** — the closed-form expression and its numerical evaluation with the assumed input values.
5. **Comparison with experiment** — the predicted value, the observed value, the deviation in absolute and σ units.
6. **Open questions** — any gaps, pending closures, or assumptions that still depend on external physics inputs. Be honest.

---

## Hard quality gate (BLOCKING)

**Every step in the derivation must be one of the following. No exceptions.**

1. **An axiom of the framework**, explicitly stated as such (e.g., "the toggle is binary and self-inverse").
2. **A step of algebra or arithmetic** that is shown explicitly — every line follows from the previous by standard mathematical operations that could be checked by a CAS.
3. **A known mathematical theorem** with a precise citation (author, year, theorem number, or textbook section). Examples: Sunada 2012 Theorem 3.1, Ihara 1966, Wigner d-matrix formula (Sakurai §3.8).
4. **A result proven in another `predictions/` file**, referenced by filename — and that file must already exist and itself pass this same standard.
5. **A direct member or chain from a class master theorem** (added 2026-04-28): the prediction equals a Level 1 master-theorem member documented in `../theorems/theorem_unified_spectral_dark.md` (Class A) / `theorem_class_B_dispersion.md` (B) / `theorem_class_D_statistical.md` (D) / `theorem_class_E_combinatorial.md` (E), OR a Level 2 chain from such members with explicit dependency citations to (i) the relevant class theorem doc, (ii) structural ledger rows it depends on (per `../audits/registers/uniqueness_ledger.md`), and (iii) any other parameter ledger rows in the chain. The chain must be traceable via `parameter_DAG_chains.md`. Class C parameters (taxonomic — no master theorem) continue to use Type 1-4 only.

6. **Algebraicity gate via the K-meta-theorem** (added 2026-04-29): for a prediction in Class A/B/C/E (excludes Class D), the coefficient closes as theorem-grade if all three conditions hold:
   - **(6a) L-expression.** The derivation is expressible in the framework's structural derivation language L (`../theorems/theorem_lattice_coupling_general.md` §2). L admits: arithmetic on K-elements, root extraction within K, spectral data of K(i)-valued matrices for the framework's specific operators, Bloch gradients at high-symmetry k-points in REDUCED Bloch coordinates, integer counts of paths/cycles/orbits, group orders, traces in framework reps, geometric series of K-elements, **`canonical_encoding(S)`** (lowest-bit-cost representative within an encoding-equivalence class — all elements of S evaluate to the same numerical value), and **`channel_select(S, c)`** (selects the K-element of S whose structural reading matches a stated channel constraint c — physical-channel selection, NOT bit-cost ordering across channels). NOT admissible: continuum unbounded loop integrals, γ_5 traces of continuum Dirac fields, transcendental functions of energy scales, exp/log of arbitrary arguments, π factors from continuum measures, **single-operator "MDL bit-cost minimum across all K-candidates"** (this conflates `canonical_encoding` with `channel_select` and contradicts A2-T waterline; reformulate as the explicit two-step). State the L-expression explicitly, naming each selection step as `canonical_encoding` or `channel_select`.
   - **(6b) K-membership.** The dimensionless coefficient lies in K = ℚ(√2, √3, √5). Cite `../theorems/theorem_lattice_coupling_general.md` Theorem 3 (every Class A/B/C/E prediction in K). Verify the coefficient is in K by exhibiting the algebraic combination (e.g., 1, √3/6, 9/40, etc.).
   - **(6c) Selection step is waterline-consistent.** Every selection step in the derivation must be one of the two A2-T-waterline-consistent forms (per `theorem_dark_correction_mdl.md` Lemma 1, REFORMULATED 2026-05-05; per `feedback_waterline_not_minimum_canonical_distinction.md`):
     - **`canonical_encoding(S)`** — applied within an encoding-equivalence class S (every element of S evaluates to the SAME numerical value). Bit-cost ranking selects the canonical (cheapest) representative; encoding-equivalents at higher bit-cost are NOT discarded from physical realization, they are alternative expressions of the same physical content. Cite the bit-cost ranking AND verify by inspection that S is encoding-equivalent.
     - **`channel_select(S, c)`** — applied across physically distinct K-rational candidates that lie in DIFFERENT structural channels (different operator couplings, different functionals with different numerical values). The relevant channel c is fixed by a structural argument (dimensional matching, operator algebra, gauge invariance); within the chosen channel, only one K-candidate is realized. Alternatives in OTHER channels remain above-waterline and physically realized — but they couple to other observables, not this one. Cite (i) the structural argument fixing channel c, AND (ii) the observational exclusion bounds confirming the chosen K-candidate matches.

     The strict-minimum framing "MDL bit-cost minimum across all K-candidates" is NOT acceptable — it conflates the two and silently discards above-waterline channels. A derivation citing only "MDL bit-cost minimum" without naming whether the selection is `canonical_encoding` or `channel_select` is a smuggle and BLOCKS Type 6 closure.

   When all three (6a)+(6b)+(6c) hold, the coefficient closes at theorem-grade (same rigor level as MDL Lemma 1 — language-restricted theorem). Class D (statistical, Poisson e^x) is structurally excluded from Type 6 — its transcendental factors come from random-graph statistics, a different mechanism.

   **Limitation note for Type 6:** the meta-theorem holds for the framework's CURRENT class theorem derivations. If a future derivation requires an operation not in L, the meta-theorem does NOT apply automatically; either the L grammar must be extended (with proof that the new operation preserves K) or the prediction falls back to Types 1-5.

7. **Audit v2 multi-axis multi-mechanism uniqueness defense** (added 2026-04-30): for any prediction labelled UNIQUE-THEOREM-GRADE, the prediction's `_derivation.md` must include or cite a defense satisfying an internal working note. The defense must explicitly:
   - **(7a) Enumerate alternative axes** — at minimum: topology, k, d, group, formula-in-primitives, class-mechanism, functional, convention. Declare each axis "sharp because ..." with explicit reason OR enumerate alternatives in (7b).
   - **(7b) Name alternatives explicitly** — RCSR codes for lattices, group names for symmetry, formulas for Class A audit-pattern checks. No "hypothetical" alternatives.
   - **(7c) Six-mechanism gating per alternative** — for EACH alternative, populate the table covering all six discovered substrate mechanisms: M1 hard residue (R-N register); **M2a structural MDL waterline ΔDL** (substrate complexity bit-count: Rissanen DL on encodings, Brown 1986 Fisher rank, Stark-Terras spectral identities, etc. — LOAD-BEARING); M3 dark-sector amplitude on alternative's compressed graph; M4 multiway branch measure on alternative's Cayley structure; M5 non-local Feshbach resummation cumulative chain; M6 operator-wave spectrum at alternative's k-point. **Empty cells block UNIQUE-THEOREM-GRADE.** "N/A" requires a stated reason.
   - **(7c-bis) M2b data-conditional Gaussian-likelihood penalty is SUPPLEMENTARY ONLY, NEVER load-bearing** (added 2026-05-01 PM per an internal note REVISED). The "alternative's predictions disagree with PDG by Nσ → ~Nσ²·log₂(e)/2 bits Gaussian-likelihood penalty" computation MAY be cited in the §3 table as supplementary empirical validation that the structural gating (M1 + M2a + M3 + M4 + M5 + M6) aligns with observation, but it does NOT itself contribute to the combined gating verdict in (7d). Citing M2b as the load-bearing closure for any axis is goal-seeking — using PDG match as the test of structural validity — and is the credibility-reducing pattern explicitly retracted 2026-05-01 PM (user catch: "PDG is never the metric of robustness; never the metric"). A row whose ONLY non-trivial mechanism citation is M2b — with no M2a / M1 / M3 / M4 / M5 / M6 structural backbone — automatically downgrades to DOMINANT-CONDITIONAL on "structural backbone pending."
   - **(7d) Combined contribution computed** — product of M1, M2a, M3, M4, M5, M6 gating factors only. M2b is excluded from the product. No arbitrary cutoffs ("probably small" not acceptable). Compute explicitly.
   - **(7e) Status assigned per audit-v2 vocabulary** — UNIQUE / DOMINANT / ONE-AMONG-MANY with margin and conditionals named.

   When all five (7a)+(7b)+(7c)+(7d)+(7e) hold, the prediction passes Clause 7. Predictions failing Clause 7 are downgraded to DOMINANT-CONDITIONAL (with named margin and conditionals) or labeled GAP (audit work explicitly named).

   **Citation shortcut for inheritance predictions** (added 2026-04-30 EOD): a prediction whose audit v2 §3 table is fully inherited from an upstream row's closure (e.g., predictions inheriting Row 4 closure for k*=3 selection) MAY satisfy Clause 7 by citing an internal working note and naming the inheritance section. The cited index consolidates Phase 0/1/2 closures with explicit (7a)-(7e) content. This avoids per-prediction §3 table duplication while maintaining auditability. Predictions with NEW alternative axes beyond the upstream closure must still produce their own §3 table (Phase 3 territory).

   **Why Clause 7 was added:** repeated rediscovery of the same systemic gap — UNIQUE-graded rows defended along ONE alternative axis with ONE gating mechanism, with orthogonal axes and downstream-amplifying mechanisms silently assumed away. Most recent surfacing 2026-04-30 (k-axis variation in Row 4 / k* = 3, missed in δ_CP_CKM and η_B audits despite explicit single-axis defenses passing). The framework has discovered six non-trivial substrate mechanisms (chirality residue, MDL waterline, dark-sector corrections, multiway branch measure, non-local Feshbach resummation, operator-wave enumeration); each is load-bearing for some closure. Defenses that don't check against ALL of them carry silent conditionals.

   **Why the M2a/M2b split was added (2026-05-01 PM):** between 2026-04-30 and 2026-05-01 multiple Phase 1d/2/3 closures (Row 4 audit-v2 reconfirmation, Rows P3/P4/P5/P28/P29/P48 reframings) silently used M2 = "data-conditional Gaussian-likelihood penalty against PDG" as the load-bearing closure mechanism. User catch 2026-05-01 PM: "I 100% disagree with this mindset. goal seeking reduces the credibility of this project. PDG is a warm fuzzy. never the metric of robustness." The corrective rule (M2a structural load-bearing, M2b data-conditional supplementary) is now codified at the protocol level so future audits cannot silently re-introduce the goal-seeking framing.

   **Inventory pass:** older predictions graded UNIQUE-THEOREM-GRADE before 2026-04-30 may not satisfy Clause 7. Per audit v2 protocol §6, those predictions carry an implicit conditional "UNIQUE conditional on every silently-assumed axis being sharp and every silently-assumed mechanism being inert" until Clause 7 audit is performed. Status downgrade or explicit conditional naming is required.

   **Audit v2 closure status (2026-04-30 EOD):** Phase 0+1+2 substantive structural pass complete. Foundational rows + Row 4 + parameter ledger cascade closed in single session. Index doc: an internal working note. Phase 3 row-specific audits + Phase 4 per-prediction sweep deferred (~2-5.5 sessions). UNIQUE-THEOREM-GRADE predictions inheriting Row 4 closure satisfy Clause 7 via index citation.

   **G_sub Drude form audit v2 (added 2026-04-30 EOD final):** The Drude running form $1/(16\pi G(\omega)) = 4/\pi^2 - 1/(36\omega^2)$ PASSES audit v2 (theorem-grade-computed via Kubo on Bloch operator, both coefficients verified <0.7%). The parallel Hashimoto-Sakharov candidate $729\sqrt{3}/(128\pi^2)$ at 0.05% match FAILS audit v2 (matched-to-observation, not gated-by-mechanism). Step 3 path (a) "$\omega_{\rm obs}$ near pole" reclassified PHANTOM (unit-mixing artifact). Closure rests on path (b) substrate-Planck reframing: $M_{\rm substrate}/M_{\rm Pl} = \sqrt{\pi}/8 \approx 0.222$ (equivalently $M_{\rm Pl}/M_{\rm substrate} = 8/\sqrt{\pi}$ — substrate is BELOW Planck mass / LONGER than Planck length), giving $G_{\rm UV}$ in Planck units $= (\pi/64)(M_{\rm Pl}/M_{\rm substrate})^2 = 1$ EXACTLY (matches observed $G_N$). Index §3.5: an internal working note. Methodology lesson: an internal note.

8. **Numerical-match audit (added 2026-05-01)**: for any prediction labelled UNIQUE-THEOREM-GRADE, the prediction's numerical value must MATCH the observed value within stated systematic precision OR be explicitly downgraded to STRUCTURAL grade with the numerical gap declared. Clause 7 verifies derivation rigor (uniqueness against M1-M6); Clause 8 verifies empirical accuracy.

   **(8a) Compute prediction-vs-observation deviation** in absolute, relative, and σ units. The σ unit must use COMBINED uncertainty: $\sigma_{\rm combined} = \sqrt{\sigma_{\rm obs}^2 + \sigma_{\rm theory}^2}$ where $\sigma_{\rm theory}$ is the framework's stated systematic floor (e.g., the bridge convention's "un-derived sub-leading Feshbach analog" magnitude for Yukawa-derived quantities, ~0.5%; or zero for genuinely-no-systematic predictions).

   **(8b) State the systematic floor** explicitly in the prediction's `_derivation.md`. Examples:
   - Yukawa-derived quantities (y_τ chain): ~0.5% un-derived sub-leading Feshbach analog.
   - 1-loop Higgs sector quantities (m_H, λ): ~0.5% un-derived 1-loop Feshbach analog for λ.
   - Quark masses below SUSY scale (m_top): ~2-3% MSSM threshold corrections.
   - Cosmological-coasting predictions (H_0, t_0, Λ_CC, Ω_Λ): no framework systematic — deviations from ΛCDM-anchored observations are GENUINE PREDICTIONS distinguishing coasting from ΛCDM, with cross-validation against alternative observation sets (e.g., Methuselah for t_0; CMB side of Hubble tension for H_0).
   - "Pure" structural predictions (V_us, V_cb, η_B, sin²θ_W, etc.): zero framework systematic; deviation = observation precision floor only.

   **(8c) PASS criterion (THEOREM-GRADE-NUMERICAL):** the deviation is ≤ 1σ_PDG. The prediction matches observation within experimental precision.

   **(8d) DOWNGRADE criterion:** if deviation > 1σ_PDG, the prediction's label is DOWNGRADED to **THEOREM-GRADE-STRUCTURAL** with explicit numerical gap declared in `_derivation.md`. The structural derivation remains rigorous; the numerical match is pending.

   **(8e) Label vocabulary distinction:**
   - **THEOREM-GRADE-STRUCTURAL**: derivation passes Clauses 1-7; numerical match pending closure of stated systematic correction.
   - **THEOREM-GRADE-NUMERICAL**: derivation passes Clauses 1-7 AND numerical value matches observation within stated systematic.
   - **UNIQUE-THEOREM-GRADE**: requires BOTH structural (Clause 7) and numerical (Clause 8) PASS.

   **Special accommodation (Category B, framework vs ΛCDM):** for predictions that intentionally distinguish framework cosmology from ΛCDM (Λ_CC, Ω_Λ, H_0, t_0 in coasting cosmology), Clause 8 PASSES if the prediction matches the framework-side observation set (e.g., Methuselah cosmic age, Planck-CMB H_0 within ~1σ), and the deviation from ΛCDM-fit is documented as a PREDICTED DIFFERENCE not a failure mode. The systematic floor for these predictions is zero (framework predicts a specific cosmology), and the test becomes "matches alternative-observation set" rather than "matches every observation set."

   **When all five (8a)+(8b)+(8c)+(8d)+(8e) hold, the prediction passes Clause 8.** Predictions failing Clause 8 retain Clause 7 derivation rigor but are labeled THEOREM-GRADE-STRUCTURAL only.

   **Why Clause 8 was added:** repeated discovery that THEOREM-GRADE-labeled predictions carried known systematic deviations from observation (y_τ +0.13%, m_τ +0.13%, m_e +0.12%, m_μ +0.12%, m_H +0.30% — all inheriting from y_τ chain or 1-loop Feshbach analogs) without the label reflecting this. User feedback 2026-05-01: "do any of the problem parameters have a unique theorem label on them? If they do, then our audit mechanism is missing something too." Audit v2 (Clause 7) verified structural derivation uniqueness; it didn't verify empirical accuracy. Clause 8 closes this gap.

   **Inventory pass:** older predictions graded UNIQUE-THEOREM-GRADE before 2026-05-01 may not satisfy Clause 8. The retroactive re-labeling sweep (per an internal working note) updates affected rows in `parameter_uniqueness_ledger.md`.

**The following are NOT acceptable and BLOCK file production:**

- "This identification follows from..." without a proof or citation
- "The structural argument is..." without explicit mathematical steps
- "It can be shown that..." — show it, or cite where it is shown
- "By analogy with..." or "In the continuum limit..."
- Any step whose justification is "it gives the right answer" or "it matches observation"
- Any step that selects between alternatives by comparing to experimental data
- Bare-then-correct patterns: computing a "bare" value from the compressed graph and then adding a "dark correction" — unless the full expression is derived as a single computation from walk statistics
- **Bridge-attribution-as-closure** (added 2026-05-15 EOD+2): citing a SM 2-loop / hadronic-VP / Δr / Δρ / Δα_had mechanism as the closure of a residual without (a) deriving the substrate analog OR (b) tagging the row as bridge-convention-only with explicit acknowledgement that K-rationality of the closure form is BROKEN. The SM 2-loop EW bridge attribution for M_Z/m_W (commit f878f82, retracted 4ce4d5c) is the canonical example. See Clause 9 below for the audit step.
- **Silent fallback returns** (added 2026-05-27): a `try/except` (or `if`-guard) whose except/else branch `return`s a hard-coded literal. This is strictly worse than an assignment literal — it masks the real disagreement when the primary computation fails. Canonical example: `predictions/tan_beta.py` once carried `except (ValueError, RuntimeError): return 44.73`, which hid a ~35% gap between the live RGE solver (≈60.07) and the documented value (44.73). If the primary path can fail, the failure must surface (raise), not be papered over with a magic return.

9. **Type-3 SM import π-audit** (added 2026-05-15 EOD+2):

   When a derivation cites a Type-3 SM mechanism (continuum QFT result imported as a structural building block), check whether the cited mechanism's value is K-rational or contains continuum loop factors.

   **Bright-line rule:** If the cited mechanism's value is set by continuum loop integration (1/(16π²), 1/(32π²), Δr ≈ 0.038, Δα_had ≈ 0.0277, etc.) — these are transcendental over K by Lindemann 1882 (π) — then citing the mechanism's NUMERICAL VALUE as the closure of a framework residual is K-INVALID per `../theorems/theorem_lattice_coupling_algebraicity.md` Theorem 3.

   Permitted resolutions:

   (9a) **Substrate analog derivation:** derive a K-rational substrate quantity that plays the structural role of the SM mechanism. The substrate quantity will look DIFFERENT from the SM form — it cannot contain 1/(16π²) — but should produce comparable magnitude. Example: Family D per-leg multiway dark-disruption is a K-rational substrate analog of vertex-level QFT loops (theorem-grade, master doc §3 (D)).

   (9b) **Bridge-convention tag:** explicitly tag the row as STRUCTURAL-DERIVATION-CONDITIONAL with named open mechanism. The bridge is cited as the SM-side closure that gives the right numerical PDG value, but the ROW grade remains structural-conditional. NOT theorem-grade.

   (9c) **Tensor-character mismatch:** if the observable's tensor character does not fit Families A/B/C/D per master doc §6, recognize this as the family-assignment gap (e.g., propagator-level custodial-breaking observables need Family E, not yet identified — see master doc §4.5). Tag as STRUCTURAL-DERIVATION-CONDITIONAL with named family-assignment gap.

   **Implicit-π-import detection:** if the cited SM mechanism's textbook value involves π or e factors at higher orders (typical of multi-loop QFT), the citation cannot serve as theorem-grade closure. The bridge-attribution-as-closure pattern is a Clause 9 violation.

   **Canonical exemplar (2026-05-15 EOD+1):** SM 2-loop EW bridge attribution for M_Z/m_W residuals (commit f878f82 retracted 4ce4d5c). Sirlin Δr ≈ 0.038 closed the M_Z residual to sub-σ_PDG numerically, but Δr = (continuum 2-loop QFT) and is not K-rational. The substrate analog (Family E custodial-breaking, blocked on R-14) is the legitimate closure path. Rows P64/P71/P68/P69 reverted to STRUCTURAL-DERIVATION-CONDITIONAL per Clause 9 (9b/9c).

10. **Rate-observable clauses (widths / lifetimes / branching fractions; added 2026-07-02, F4 S3):**

   **(10a) No dark on width fractions or width ratios — forbidden by theorem, not by convention.**
   CAS lemma `proofs/foundations/F4_S2b_width_ratio_dark_lemma_2026-07-02.py`: a REAL
   multiplicative dressing (the gauge sector's matching-point dark reads the exactly-real Perron
   channel) leaves Γ/M invariant identically and cancels exactly in common width ratios; a
   complex-pole shell dressing is stability-excluded (predicts Γ/m ≈ 4.4% for every shell
   fermion; over-applies ×1.6e16 vs the muon; contradicts Γ_e = 0). A width row that applies a
   dark correction is a Clause 10a violation. Any genuine width-side term must come from the
   ω-resolved Σ_X(ω) (`incomplete_equations_todo.md` §7) — new physics, not a dressing knob.

   **(10b) Frozen pre-registered assembly.** A multi-term SM-structure assembly (tree × QCD ×
   corrections) must be FROZEN — with omitted terms stated-not-applied and sizes estimated —
   BEFORE any comparison to observation. Adding, removing, or re-weighting a term after seeing
   the deviation is the assembly-tuning form of fitting and blocks the row. The in-file
   validation must assert the pre-registered band, and where an OPEN residual is claimed as
   located, also assert the residual's PRESENCE (so a silent vanish fails loudly and forces a
   re-audit instead of a quiet "improvement").

   **(10c) Phase space is Type-3 until derived natively.** The golden-rule per-channel factor
   (1/(48π), 1/(12π), 192π³, …) is a continuum loop import — Clause 9 applies; width rows cap at
   bridge-conditional (9b). The band-geometric derivation route is CLOSED by computation
   (`proofs/foundations/F4_cone_spectral_function_2026-07-02.py`: substrate cones are chirally
   warped spin-1 multifolds, pair channel q²-dark — do not re-walk); the open native route is
   the Clifford γ-trace layer.

   **(10d) Lifetime circularity check.** G_F is calibrated FROM τ_μ (MuLan;
   `predictions/G_F.py`). Any lifetime row whose assembly consumes G_F (directly or via v) is
   circular against its own observable and blocked. Width rows must state the G_F-unused audit
   explicitly.

**If ANY step in the derivation fails this gate:** STOP. Do not produce output files. Instead, report:
1. Which step fails
2. What specifically is asserted without proof
3. What would be needed to close the gap (a proof, a citation, or a reformulation)

Classify the parameter as `blocked` with the specific gap noted. This is not a failure — it's an honest assessment that protects everything downstream.

---

## Standards and rules

**On derivation quality:**
- A derivation is `theorem`-grade only if every input is itself derived from framework axioms AND every step passes the hard quality gate above. If any input is taken from experiment — even a well-known constant like hbar — the grade is at most `mathematically complete`.
- Do not upgrade a derivation grade without explicit justification.
- Curve fitting is any procedure where a free parameter is chosen to minimize distance to an observed value. This includes "choosing n=5 because it gives the right answer" unless there is an independent structural reason to select n=5 that is itself proven.

**On the pure function:**
- This is a strict contract. A reviewer should be able to call `predict_{param_name}(...)` with any values and get a consistent computation.
- Do not add default argument values to the signature. All inputs must be explicitly passed by the caller.
- The function must be deterministic and have no side effects.

**On literals beyond the pure function (added 2026-05-27):**
- The no-literal contract is not confined to the `predict_*` body. NO module-level assignment in a predictions file may hard-code a physical or framework constant on the RHS of `=`. If a number appears on the right of `=` anywhere in the file, it must resolve to either (a) an import from a canonical leaf, (b) an expression built from imported leaves, or (c) π / e. The only RHS literals exempt from this rule are the comparison-only values described below.
- **Single-source principle.** Every duplicated constant has exactly ONE authoritative leaf; consumers import it, never re-hard-code it.
    - CODATA / SI / IAU values → `M_Pl_natural.py` (M_Pl, ℏ, c, GeV→J, Mpc, SI prefixes).
    - Framework substrate integers → `k_star`, `d_spatial`, `p_toggle`, `g_girth`, `V_count`, `E_count`.
    - Derived constants → their named leaf (`delta_Koide`=2/9, `Q_Koide`=2/3, `mssm_beta_coefficients`, `c_vertex_dark`=5/12, etc.).
  Re-deriving `2.0/9.0` or `33.0/5.0` inline when a leaf exists is a violation even though the value is "correct."
- **Operator literals need a reason.** An inline `/2`, `*4`, `**3` etc. is only acceptable if the number is a structural quantity that traces to a leaf (e.g., `/p_toggle`, `*V_count`, `k_star**d_spatial`). A bare `/2` with no derivational reason is a magic number — replace it with the named quantity it represents.
- **Comparison-only values are exempt.** Module-level `*_obs`, `*_sigma`, `*_pdg` (and the observed/uncertainty literals inside the `# --- OBSERVED VALUE` block) are NOT parameters — they never feed the `predict_*` computation; they exist solely for the σ-deviation report. Hard-coding the PDG/observed value and its uncertainty here is correct and required. Do not "source" them from a leaf and do not flag them.
- **Leaf-proof exception.** A leaf file whose entire purpose is to PROVE a structural value (k*=3, |V|=4, |E|=6, …) may return that integer — the file IS the derivation, and the returned literal is its QED, not a smuggled input. This applies only to a file that establishes the value from axioms/theorems; any file that CONSUMES the value must import it, never restate it.

**On git usage:**
- You may commit locally if needed, but **never push to any remote**. No `git push`, no `gh` commands that write to remote, no pull request creation.

**On inputs marked `[external]`:**
- These are honest admissions that the derivation is not yet fully closed. Do not hide them. List them in the derivation's "Open questions" section.
- An `[external]` input does not disqualify a script from being the keeper; it just sets the derivation grade.

**On inconsistencies:**
- If two scripts predict the same value via different mechanisms, note both in the derivation file's "Open questions" section.
- If no script achieves better than `partially fitted` quality, say so clearly and do not dress it up as complete.
- If the predicted value lies outside 3σ of the observed value, flag this prominently and do not silently select that script as keeper.
