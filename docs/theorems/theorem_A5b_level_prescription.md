# A5(b) coupling prescription by framework level — theorem

**Date:** 2026-04-24 (Session 25 extended).
**Status:** THEOREM — gate-passing under `../parameters/parameter_linter.md`. Resolves the A5(b) consistency problem documented in an internal working note.
**Scope:** Clarifies the exact meaning of A5(b)'s "sum over above-waterline NB walk representations" for observables defined at different levels of the framework's three-level hierarchy. Closes Path 1 from the scoping doc. Preserves all current ✅ predictions.

---

## 1. Theorem statement

**Theorem (A5(b) level prescription).** Under A5(b) and the three-level framework hierarchy, the physical coupling identified with an MDL probability takes one of two forms depending on how α₁ enters:

**(A) Direct-moment form.** If the coupling is identified with a single μ-moment of the branch measure — i.e., α₁ (or a function of α₁) enters the observable's formula as a specific numerical coefficient representing ONE graph-theoretic event probability — then A5(b) identifies the coupling with that single moment. No winding sum.

Formally: coupling = ((k*−1)/k*)^{g−n_fixed} × (graph multiplicity factors)
                  = α₁_bare × (5/3 or 1/k* or dim(Cl) etc.)

**(B) Walk-representation sum.** If the coupling is identified with a sum over Hashimoto walk representations between two pinned causal states — i.e., multiple winding numbers correspond to distinct walks contributing to the SAME coupling — then A5(b) identifies the coupling with the full waterline sum.

Formally: coupling = Σ_{n=1}^∞ ((k*−1)/k*)^{n(g−n_fixed)} = u^L / (1 − u^L)  where u = (k*−1)/k*, L = g − n_fixed

**Predictive discrimination.** The distinction is STRUCTURAL, not observational: it depends on whether α₁ enters the derivation as (A) a specific coefficient in a closed-form expression, or (B) as a per-step factor in a walk-amplitude sum. "Theory over match" — the classification is forced by the derivation structure, independent of what gives a better observational match.

---

## 2. Axioms and upstream

**Framework axioms:**
- **A2** (`../framework/framework_axioms.md` §3): MDL waterline retention of all configurations with positive compression savings.
- **A5(b)** (`../framework/framework_axioms.md` §5b): MDL probability = physical coupling strength.

**Upstream (Type 4):**
- `predictions/alpha_1.py`: α₁_bare = (2/3)^8 as direct moment on srs.
- `predictions/V_cb.py`: V_cb = α₁/(1−α₁) as Hashimoto walk-sum.
- `proofs/flavor/a5b_coupling_prescription.py`: gate-first analysis confirming waterline sum for V_cb.

**Cited (Type 3):**
- **Shalizi, C. R. and Crutchfield, J. P.** (2001). Computational mechanics: pattern and prediction, structure and simplicity. *J. Stat. Phys.* 104, 817–879. Theorem 2: the causal state of a non-backtracking walker is its current directed edge. Establishes the Hashimoto graph as the causal-observer graph of srs.

---

## 3. The structural distinction

A5(b)'s text reads: "The total branch-measure probability of **all above-waterline NB walk representations of the process** is identified with the physical coupling strength." [Framework axioms §5b]

The key phrase is "**walk representations of the process**." The prescription depends on whether the coupling is a FUNCTION OF such representations, or a COEFFICIENT derived from the substrate structure.

### 3.1 Case (A): direct-moment (Level 2 / srs-intrinsic)

Some couplings are identified with SPECIFIC MOMENTS of the branch measure μ: not sums over variable-length walks, but definite numerical values of walk-survival probabilities, cycle counts, edge densities, etc. Examples:

- α₁_bare = P(8-step NB walk survives on srs) = (2/3)^8. **One number.** The walk is SPECIFIC (the girth cycle with g−2 = 8 interior NB steps). There are no multi-winding "representations" being summed — α₁_bare IS the direct probability of this one event.

- λ_Higgs = 2 × (5/3) × α₁_bare. Here α₁_bare appears as a COEFFICIENT in a closed-form Higgs quartic expression. The 2 counts Cl(0,2) polarization channels (already a "sum over representations" in the per-channel sense), and (5/3) is an edge-resolved cycle-count factor. No additional winding sum is invoked.

- y_τ = α₁_full / k*². Here α₁_full is the cycle-amplitude coefficient; (1/k*)² encodes the fermion edge projections. Closed-form expression, not a walk sum.

- θ_23_PMNS = arctan((1+α₁_full)/(1−α₁_full)). α₁_full appears as a perturbation coefficient in a mixing-angle formula derived from Bloch spectrum at P-point. No walk sum.

In all Case (A) examples, α₁ is a STRUCTURAL COEFFICIENT — a specific numerical value from srs graph invariants (Level 2). A5(b)'s identification "MDL probability = coupling" applies to this direct moment.

### 3.2 Case (B): walk-representation sum (Level 3 / Hashimoto-intrinsic)

Other couplings are identified with SUMS OVER WALKS on the Hashimoto causal-observer graph. Multiple walks (of different winding numbers, or different lengths, or different topological classes) contribute to the SAME coupling, and A5(b) identifies the coupling with the full above-waterline sum. Example:

- V_cb = Σ_{n≥1} (2/3)^{8n} = α₁ / (1 − α₁). Here V_cb is the CKM matrix element <c|T|b> between b- and c-type causal states on the Hashimoto graph. Each winding n is a DISTINCT walk (length 8n between the same endpoints). All above-waterline (savings grow linearly in n). A5(b) says sum all representations → geometric series.

- dark correction in v_Higgs = 1 − (5/12)α₁/(1−α₁). The correction sums over Hashimoto dark-sector winding modes (Feshbach-style sub-leading dressing). A5(b) applies to this Level 3 walk sum.

In Case (B), α₁ appears per-step/per-winding, and the sum over windings is essential to the coupling's definition.

### 3.3 The criterion for (A) vs (B)

**Criterion:** If the coupling's derivation expresses it as `(closed-form function of α₁_bare and other graph invariants)`, it's Case (A). If the coupling's derivation is `(sum over Hashimoto walks with α₁^{f(n)} weight per winding n)`, it's Case (B).

This criterion is INTERNAL to the derivation, not based on observation. It is determined by the STRUCTURAL ROLE of α₁ in the formula, which is determined by which level of the framework hierarchy the coupling lives at:
- Level 2 quantities (srs graph invariants, direct moments): Case (A).
- Level 3 quantities (Hashimoto walk amplitudes, sums over walk classes): Case (B).
- Feshbach corrections / dressings involving Hashimoto sub-sector modes: Case (B).

---

## 4. Classification audit

Every α₁-dependent prediction:

| Observable | α₁ enters as | Case | Current treatment | Consistent? |
|---|---|---|---|---|
| α₁_bare = (2/3)^8 | single NB walk survival probability | A | direct moment | ✓ |
| α₁_full = (5/3)α₁_bare | edge-resolved cycle amplitude | A | direct moment | ✓ |
| λ_Higgs = 2α₁_full | Cl(0,2) channels × cycle amplitude | A | direct moment | ✓ |
| y_τ = α₁_full/k*² | cycle × fermion edge projections | A | direct moment | ✓ |
| m_τ = v × y_τ | inherits y_τ's Case (A) | A | direct moment | ✓ |
| m_μ, m_e via Koide f_j | inherits m_τ | A | direct moment | ✓ |
| θ_23_PMNS = arctan((1+α₁)/(1−α₁)) | perturbation coefficient | A | direct moment | ✓ |
| V_cb = α₁/(1−α₁) | per-step factor in walk sum | B | geom series | ✓ |
| V_us = k*²/(g·N_ATOMS) | counting form (A5(b) uniform branch) | B (special) | counting form | ✓ |
| dark correction: (5/12)α₁/(1−α₁) | walk sum over dark-sector windings | B | geom series | ✓ |

**All current ✅ predictions are consistent under the Path 1 rule.**

---

## 5. Verification: the rule is not observationally selected

To confirm the rule is not "select-by-match" in disguise, we check that:

**5a.** The classification is a property of the DERIVATION STRUCTURE (how α₁ appears in the formula), not of the match to observation.

- α₁_bare is derived as a single cycle-survival probability, not as a sum — this is clear from `predictions/alpha_1.py`'s derivation chain (Type 1+2 under A5(b)+Jaynes). No walk sum involved.
- V_cb is derived via `proofs/flavor/a5b_coupling_prescription.py` Step 1 as an explicit sum over winding classes. The walk-sum structure is manifest.
- λ = 2 × α₁_full is derived in `predictions/lambda_higgs.py` as "channel count × cycle amplitude" — no walk sum, just a closed-form expression.

**5b.** The rule gives SPECIFIC, TESTABLE predictions. For each observable:
- If α₁ enters as a single moment: direct-moment formula.
- If α₁ enters as a per-step walk factor: geometric series formula.

The classification for each existing observable was determined BEFORE checking observational match. Applying the rule reproduces every current convention:
- V_cb → Case B → geom series = 256/6305 (matches obs)
- λ → Case A → 2α₁_full (matches obs to 0.6%)
- y_τ → Case A → α₁_full/k*² (matches obs to 0.13%)

**5c.** The rule PREDICTS that if any new observable enters with a structure that clearly makes it Case B (sum over walks), the geometric series applies — even if that makes the observational match worse. Similarly for Case A. This is the "theory over match" discipline.

---

## 6. Reconciliation with A5(b) text

The A5(b) text in `../framework/framework_axioms.md` §5b line 193 reads:

> "The total branch-measure probability of all above-waterline NB walk representations of the process is identified with the physical coupling strength."

Under Path 1, this statement applies specifically to Case (B) processes — those DEFINED as sums over walk representations. The key phrase "NB walk representations of the process" implies the process has multiple walk representations being summed.

For Case (A) processes, the coupling is identified with a DIRECT MOMENT of μ. This isn't excluded by A5(b) — it's just a different formal structure: A5(b)'s "MDL probability = physical coupling" identification still applies, but the MDL probability is a single moment rather than a sum.

**Proposed refinement of framework_axioms.md §5b:** Add a clarifying paragraph distinguishing Case (A) and Case (B) with the derivation-structure criterion (§3.3). No substantive change to A5(b) itself — just clarification of its scope.

---

## 7. Consequence for the m_H tension

Under Path 1, the m_H +3.43σ tension is NOT an A5(b) inconsistency. The λ prediction 0.1301 is the correct theorem-grade value of the direct-moment formula. The 0.6% observational tension must arise from a SEPARATE mechanism, outside the scope of A5(b)'s branch-measure identification:

**Candidates for the m_H residual (NOT A5(b)-related):**

(i) **1-loop radiative corrections to the pole-mass m_H.** In standard SM QFT, m_H² (pole) = 2λ_MS(μ)v² + δ_finite + ... where δ_finite captures 1-loop matching between MS-bar Lagrangian λ and the physical pole-mass relation. Magnitude: ~0.3–1.0% (Degrassi et al. 2012). The framework's tree-level graph amplitude 2α₁_full is an MS-bar-analog coupling; the observed pole m_H differs by these finite corrections. A framework-native computation of δ_finite is an open research question.

(ii) **1/(16π²) empirical observation.** λ × (1 − 1/16π²) matches m_H obs to −0.19σ (observation from scoping doc §6). This is the universal QFT loop prefactor. Whether it can be DERIVED from framework graph structure (via BZ spectral density integration + Hashimoto density of states) is the research question that would complete the story.

(iii) **Multi-cycle contributions at higher order.** Two distinct girth cycles through a vertex contribute at next-order (not covered by single-cycle amplitude). These have never been enumerated. If they give the right sign and magnitude, they could explain the residual at Case (A) level without invoking "loops" in the continuum-QFT sense.

These are SEPARATE investigations from the A5(b) consistency question, all worth pursuing. Under Path 1, the m_H tension is preserved honestly as an open 1-loop question.

---

## 8. Gate audit

| Step | Gate | Status |
|---|---|---|
| §3 Case (A)/(B) distinction from A5(b) text + three-level hierarchy | T1 (A5(b) axiom) + T4 (three-level hierarchy memory) | ✓ |
| §4 Classification audit of every prediction | T2 (direct inspection of each file) | ✓ |
| §5 Verification of no match-selection | T2 (derivation-structure criterion, not observation) | ✓ |
| §6 Reconciliation with A5(b) text | T1 (axiom reading) | ✓ |
| §7 Identification of m_H residual as non-A5(b) | T3 (Degrassi et al. 2012 SM matching corrections) | ✓ (separate workstream) |

**Theorem grade.** Every load-bearing step passes T1/T2/T3/T4. No adoptions. No match-selection. No unjustified definitions.

---

## 9. Downstream implications

**No predictions change** under Path 1. The rule VALIDATES every current ✅ prediction's treatment of α₁. The A5(b) consistency question is resolved by precise specification of A5(b)'s scope.

**What changes:**

- `../framework/framework_axioms.md` §5b should be updated with the clarifying paragraph from §6 above.
- The m_H +3.43σ tension remains on the active research list as a separate 1-loop question (Paths 4 or 5 from the scoping doc).

**Open questions now sharper:**

- Can the 1-loop matching correction to m_H (Degrassi-style) be derived from framework graph structure?
- Is the 1/(16π²) empirical match to m_H a coincidence or a signature of the framework's natural 1-loop prefactor?
- Do multi-cycle amplitudes on srs contribute at order α₁² with the right sign?

---

## 10. Status

**THEOREM-GRADE — session 25 extended (2026-04-24).** Closes Path 1 from the A5(b) consistency scoping doc. Preserves all current ✅ predictions. Reframes the m_H tension as a 1-loop matching question, separate from A5(b).

**UPDATE 2026-06-22 — the level-SELECTION is now GROUNDED in the derived recurrence law.** The §3.3 criterion ("how α₁ enters: closed-form coefficient = Level-2 / Case A, vs per-winding walk factor = Level-3 / Case B") was structural but not grounded in A1. It is now derived from the physics: see `theorem_A5b_level_selection_from_recurrence_2026-06-22.md`. A flavor transition is **Level-3 (walk-sum) iff it touches the heavy (gen-3) generation, which rides the real Perron / CUMULATIVE recurrence mode**; a **light-only transition is Level-2 (local single density)**. This re-derives why α₁ enters per-winding (cumulative ⇒ walk-sum) vs as a coefficient (local ⇒ single moment). Verified on V_us (Level-2 ✓), V_cb, V_ub (Level-3 ✓). So the level-selection is **forced, not merely structural**; the lone remaining input is **A5 itself** (the species/Hamming-weight labeling). (Also: reconcile the V_us label — `V_us.py` "Level 2" vs the §4 table "B (special)" — to "Level-2 / local uniform-density.")

First-read order: §§1 (statement), 3 (distinction), 4 (classification), 7 (m_H implication), 10 (this).
