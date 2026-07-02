# Framework Scheme Convention — how framework-native couplings compare to Standard Model observables

> **For the unified treatment of the framework's substrate-Feshbach-analog dark-correction mechanism (universal template, observable classes, derivation routes, cluster catalogue, application protocol), see `docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md` (2026-05-15).** This scheme convention is the foundational scoping doc; the master theorem doc is what to consult when applying corrections to a new parameter.

**Date:** 2026-04-25 (session 26 scoping; supersedes the implicit "tree-level Lagrangian-analog plus generic ~1% theoretical-uncertainty band" treatment used in `predictions/m_H.py` and elsewhere prior to this date).
**Status:** SCOPING-DECLARATION. This document establishes the framework's working convention for how its numerical couplings are to be compared to Standard Model observables. The convention is grounded in derived structural mechanisms (the Feshbach substrate self-energy developed in an external research note on the dark-correction theorem §4a–§4c.5b, ported into this repo as the (5/12) dark correction on v in `predictions/v_higgs.py` and the dark amplitude correction on V_us / m_ν).
**Scope:** This is not an axiom. It is the declaration of what the framework's tree-level numerical outputs ARE — namely, structurally-corrected MDL-effective couplings, not conventional MS̄-at-scale objects. Every prediction file that compares a framework-derived coupling to a Standard Model observable inherits this convention.

---

## 1. Why this document exists

In conventional quantum field theory, a coupling constant is not a single number. It is a family of numbers indexed by two choices: a renormalization **scheme** (pole, on-shell, MS̄, …) and a renormalization **scale** μ. The same physical observable corresponds to different numerical values of "the coupling" depending on which (scheme, scale) you adopt; conversion rules between conventions are part of the standard machinery.

The framework derives numerical values for couplings (α₁, λ, y_τ, V_us, V_cb, dark corrections, ...) as rational/algebraic expressions in graph-combinatorial constants. These derivations have no scheme label and no scale label. Until now, predictions that compared framework numbers to Standard Model observables (notably m_H = √(2λ)·v, m_τ = v·y_τ) implicitly assumed pole/on-shell identification at the electroweak scale, with a generic ~1% theoretical-uncertainty band absorbing the ambiguity.

This implicit treatment created two problems:

1. **Epistemological:** the ~1% band was a borrowed estimate from Standard Model 1-loop matching (Degrassi et al. 2012), used as a placeholder for "loop physics the framework hasn't derived." It was not derived from the framework's own structure. It conflated framework theoretical uncertainty with experimental PDG precision in ways that produced apparent multi-σ tensions that did not reflect actual disagreement, and was retracted: Clause 8 is now evaluated against σ_PDG only and structural residuals are reported in σ_PDG honestly.

2. **Structural:** the framework was treating its tree-level couplings as if they were MS̄-at-some-scale Standard Model couplings missing only loop matching. They are not. The framework's derivations operate in a different regime — substrate-versus-rendering — that has no natural counterpart to MS̄ scale running, but does have its own structural corrections (Feshbach self-energies from substrate the observer's MDL projection threw away). The right comparison to Standard Model observables uses these structural corrections, not borrowed loop machinery.

This document fixes the structural problem by stating the framework's actual scheme convention. The accompanying retraction of the ~1% theoretical-uncertainty band fixes the epistemological problem: residuals are reported against σ_PDG only.

---

## 2. The framework's couplings are NOT MS̄-at-some-scale objects

The framework's worldview has two layers, made canonical in `framework_architecture.md`:

- **Substrate** (Layer 1): the full uncompressed multiway toggle history. Combinatorially, this is the free involutive monoid F_inv(E) of A1, with all branches present.
- **Visible** (Layer 2): the observer's MDL-optimal compressed rendering of the substrate, which is the chiral 3-regular srs (Laves) lattice.

A coupling computed on the visible srs is a **bare** framework number — it counts MDL-licensed structures (girth cycles, Hashimoto walks, vertex configurations) without reference to any energy scale. The graph has a fixed combinatorial resolution; there is no "running with scale" the way Wilsonian RG has, because there is no continuum to integrate out.

But a bare srs computation is incomplete. The MDL projection from substrate to visible threw away parts of the substrate that are still influencing observables. These parts re-enter through a **Feshbach self-energy**: the walker can excursion out into the discarded substrate, traverse a girth cycle (the minimum-length non-backtracking loop that re-enters the visible rendering), and return. That excursion has amplitude α₁_bare = (k−1/k)^(g−2) = (2/3)^8 (for srs at k=3, g=10), and dresses every observable with a structural correction.

Crucially, **this correction is tree-level in the framework's own sense**. It is not a loop in the Feynman-diagram sense and has no analog in conventional perturbative QFT (which lacks the substrate-vs-rendering structure). What looks to a conventional physicist like "1-loop matching" is, in the framework, a tree-level fact of the substrate-vs-rendering geometry.

Once this is recognized, the question "what scale is the framework's λ MS̄-defined at?" becomes ill-posed. The framework's λ is not MS̄-at-any-scale. It is a different kind of object: a bare srs combinatorial number, intended to be augmented by its Feshbach self-energy and then compared directly to Standard Model observables defined at the corresponding physical scale (pole mass for the Higgs, etc.).

---

## 3. Convention statement

**The framework's bridge to Standard Model observables is:**

> A framework-native coupling C is the sum of a bare combinatorial term C_bare (computed on the visible srs from MDL-licensed structures per A5(b)) and a Feshbach self-energy correction Σ_C (computed from the substrate complement of the MDL projection, per the contour integral of an external research note on the dark-correction theorem §4a). The total C = C_bare + Σ_C is intended to equal the Standard Model pole-mass-equivalent coupling at the observable's natural physical scale, without further scheme/scale machinery.

**Concretely:**

1. There is no MS̄ scale μ chosen anywhere in the framework. The framework's couplings do not run.
2. There is no on-shell vs pole vs MS̄ choice. The framework's couplings are framework-native objects with their own structural corrections.
3. The mapping to Standard Model observables is: **bare + Feshbach = pole-mass-equivalent**, applied to the framework's tree-level Lagrangian relations (m_H² = 2λv², m_τ = v·y_τ, etc.).

   **Convention note (added 2026-05-20 per an internal working note):** the framework's Yukawa coupling `y` is defined as the dimensionless coupling to the *full* Higgs field, giving `y = m/v` (no /√2). This is what `predictions/y_tau.py`, `predictions/m_tau.py`, and `theorem_ytau_corollary.md` §10 use operationally. The relation to the SM Peskin convention (where `y_SM = √2 · m/v` and `m = y_SM · v/√2`) is `y_framework = y_SM / √2`. The version of this point that previously read `m_τ = v · y_τ / √2` was a mixed-convention typo that contradicted the operational convention; W25 audit (6/6 checks, machine precision) verified that the framework's quoted "y_τ matches at +0.13%" is in the framework convention, not in Peskin's. The two differ by exactly √2.
4. Residuals after applying all derived Feshbach corrections are interpreted as **un-derived Feshbach analogs**, not as un-applied loop matching. They are reported as σ_PDG-only deviations in prediction files; widening tolerances by attributing residuals to a theoretical-uncertainty band is no longer the convention.

---

## 4. Worked examples

### 4.1 Higgs VEV — (5/12) Feshbach correction is theorem-grade

The Higgs VEV is a 2-point quadratic field expectation `⟨φ†φ⟩`, so its Feshbach correction enters with **squared chirality content** Im²(h) (rather than linear Im(h) for amplitude observables). The contour integral on the Ramanujan circle (`dark_correction_theorem_2026-04-14.md` §4c.5b) gives:

    c_Higgs = k* · (Im²(h) / k*²) = Im²(h) / k* = (5/4) / 3 = 5/12

so

    v = v_bare × (1 − (5/12) · α₁ / (1 − α₁))

with the geometric series structure (winding sum) coming from the A2 waterline. This is theorem-grade per session 18+21 closure (`predictions/v_higgs.py`).

### 4.2 V_us, m_ν2, m_ν3 — linear Feshbach correction

These are 1-point amplitude observables, so their Feshbach correction enters with **linear chirality content** Im(h). The same contour integral gives `Im(h)/|h|² · α₁_bare`, applied to all three sectors with one derived coefficient. See `dark_correction_theorem_2026-04-14.md` §4a, §1 table.

### 4.3 Higgs quartic λ — ~~NOT YET DERIVED~~ **CLOSED by Family D (2026-05-15)**

> **⚠ STALE — superseded (corrected 2026-06-22).** This doc is dated 2026-04-25; the λ Feshbach analog was
> derived three weeks later as **Family D (per-leg multiway dark-disruption)**: δλ/λ = −4·α₁_bare² (4 Higgs legs
> on |φ|⁴) ⟹ λ = 0.129269 vs obs 0.129281 = **−0.05σ_PDG PASS**; m_H +3.43σ → −0.05σ. LIVE in
> `predictions/lambda_higgs.py`; ledger Row P41. The "1/(16π²)" hint below is a REJECTED Clause-9 K-violation;
> Family D's 4α₁² is the K-rational derived correction. **Verify against the live ledger / `predictions/*.py`,
> not this doc.** Only residual: the Family-D *conditional* (a 2nd independent route for c_F). Text below is historical:

The framework's tree-level λ = 2α₁_full = 2560/19683 is theorem-grade under A5(b) Case (A) Level-2 direct moment (`../theorems/theorem_A5b_level_prescription.md`). Compared to λ_obs = m_H²/(2·v_obs²) ≈ 0.129280, there is a +0.6% residual on λ.

Under this convention, that residual is interpreted as **an un-derived Feshbach analog on λ**, not as un-applied 1-loop matching. Three naive analog forms (an internal working note §3-5) were tested in session 25 and falsified. Empirical hints exist (the universal QFT 1/(16π²) prefactor matches λ_obs/λ_tree to 0.033%, well within PDG σ_λ), but no derivation has been produced. This is the concrete deliverable of "step 2.1" in the Priority 4.4 scope.

The framework's full m_H prediction `√(2·λ_tree)·v_corrected` lands at 125.578 GeV, which is +3.43σ_PDG against the observed 125.20 ± 0.11 GeV. The (5/12) Feshbach correction on v IS being applied (`predictions/v_higgs.py` runs and outputs v = 246.2197 GeV essentially matching v_obs = 246.22 GeV via G_F round-trip), but in the m_H = √(2λ)·v chain v matches v_obs by construction (N_hub is anchored via G_F), so the (5/12) correction does not numerically compensate the λ residual in this composite prediction. The full +3.43σ_PDG corresponds to the missing Feshbach analog on λ.

### 4.4 y_τ — ~~status open~~ **CLOSED by Family D (2026-05-15)**

> **⚠ STALE — superseded (corrected 2026-06-22).** The y_τ Feshbach analog was derived as **Family D**:
> δy_τ/y_τ = −(5/6)·α₁_bare² (Yukawa vertex = 1 Higgs leg + 2 fermion legs; c_H=α₁², c_F=−α₁²/12 each) ⟹
> **−0.17σ_PDG PASS**; m_τ +18.67σ → −0.17σ. LIVE in `predictions/y_tau.py`; ledger Row P7. The fermion-leg
> coefficient c_F currently rests on a single channel_select argument (its two historical routes were
> encoding-equivalent) — discharging it with a 2nd independent route is the open *conditional*. A separate
> sub-leading D1 −10.8 ppm residual is a known tar-pit (don't pattern-match α-powers). Text below is historical:

The framework's tree-level y_τ = α₁_full / k*² = 1280/177147 is theorem-grade under A5(b) Case (A) Level-2 direct moment with zero adoptions (`../theorems/theorem_ytau_corollary.md`). The +0.13% residual on y_τ (= ~+0.4σ relative to a 1% scale) has not been investigated under this convention. Whether y_τ admits a Feshbach analog of its own (analogous to the (5/12) on v but for a fermion-Higgs vertex rather than a Higgs self-coupling) is open research.

---

## 5. Clause 8 is evaluated against σ_PDG only

Prior to this convention, prediction files set a theoretical-uncertainty band ≈ 1% × observable (sourced from Degrassi et al. 2012 SM 1-loop matching estimates) and reported Clause 8 PASS/FAIL against the quadrature combination of σ_PDG and that band. The convention is now:

- Clause 8 is evaluated against σ_PDG only. Prediction files report Deviation, σ_PDG, and (Deviation / σ_PDG) honestly, without widening tolerances.
- Residuals after derived Feshbach corrections are reported in σ_PDG and named as the un-derived Feshbach analog they correspond to (e.g., the +3.43σ_PDG on m_H names the un-derived analog on λ).
- The convention's positive content — bare + Feshbach = pole-mass-equivalent — is preserved. The retracted piece is the theoretical-uncertainty band that absorbed residuals.

This is the harsher and more honest accounting. Cluster predictions that previously reported "Clause 8 PASS within the theoretical-uncertainty band" now report multi-σ_PDG failures, which directly diagnose where structural work remains.

---

## 6. What this convention does NOT do

This convention is a **declaration**, not a derivation. It states what the framework's tree-level couplings are, grounded in the structural Feshbach mechanism that has been derived. It does not:

1. **Close the m_H residual.** The +3.43σ_PDG residual on m_H remains open as a missing Feshbach analog on λ — v matches v_obs essentially exactly via the G_F round-trip, so the entire residual lives on λ. Deriving the analog is research-level work captured under Priority 4.4 step 2.1.
2. **Close the y_τ residual.** Same status, separately open.
3. **Prove that Feshbach exhausts all corrections.** It is conceivable that sub-Feshbach corrections (higher-order substrate effects, non-trivial vertex topologies, etc.) exist at sub-percent level. The convention asserts "Feshbach is the framework-native correction class for tree-level α₁-dependent couplings"; it does not assert "no other corrections exist." Convergence and exhaustiveness of the Feshbach expansion are open questions.
4. **Apply to all couplings uniformly.** See §7.

The honest claim is: the convention is the right framing for the framework's α₁-dependent tree-level couplings (λ, y_τ, V_us, m_ν). It is the framing under which the (5/12) and (Im(h)/|h|²) corrections, both of which have rigorous structural derivations, are the leading-order corrections. Further structural work is needed to derive analogs on couplings that don't yet have one.

### 6.1 Speculative structural question — does the framework reproduce 1/(16π²) on λ?

A speculative pattern worth flagging for step 2.1 research. Two facts:

**Fact 1.** The (5/12) Higgs-VEV correction is derived as Im²(h)/k* from a vertex-class Feshbach mechanism — vertex-counting × quadratic chirality. This is a *3D crystal* construction: 3 edges per vertex (k* = 3), squared chirality content (5/4) of the walker eigenvalue, no continuum geometry invoked anywhere.

**Fact 2.** The empirical residual on λ matches the universal QFT 1-loop prefactor 1/(16π²) ≈ 0.633% to 0.033% (within PDG σ_λ; an internal working note §2 match (β)). 1/(16π²) is conventionally a 4D-loop volume factor, arising from d⁴k integration (2π)⁻⁴ after Wick rotation.

These two facts live in apparently different geometries — one combinatorial-3D-crystal, the other continuum-4D-momentum. A naive expectation under the convention would be that the framework's analog on λ is *another* combinatorial-crystal expression like (5/12) — perhaps Im²(h)/k*² or Im(h)·n_g/k*² or similar — not a continuum-loop factor. The fact that the residual matches 1/(16π²) is therefore either:

(i) **Genuine structural connection.** The framework's analog on λ involves a Brillouin-zone integral on the srs Hashimoto operator (`../theorems/theorem_bloch_lift_mu.md`), and the BZ integral with appropriate spectral measure produces 1/(16π²) from graph-structural data. If true, this would be a striking framework-level derivation of QFT's universal loop volume factor from finite-dimensional combinatorics + chirality. Session 25 tested several naive BZ-integrated forms (an internal working note §4 Path 5) and falsified them, but a sufficiently sophisticated form (graph-QFT loop machinery with proper vertex factors) was not tested.

(ii) **Numerical coincidence.** 1/(16π²) is a specific transcendental number; the framework's analog on λ produces a different rational/algebraic combination of graph constants that happens to be very close numerically. Plausible at sub-percent precision, but not a deep statement.

The distinction matters for step 2.1's research direction: under (i), the right tactic is to set up the framework's BZ-loop machinery carefully and check whether 1/(16π²) emerges; under (ii), the right tactic is to enumerate combinatorial-crystal candidates (Im^a(h)/k*^b · n_g^c · g^d · ...) systematically and find the one that lands at the empirical residual without invoking the QFT loop factor.

This is a step 2.1 research question. The convention itself does not commit either way — it states only that whatever the analog turns out to be, it is the framework-native correction (Feshbach or Feshbach-equivalent), not borrowed SM 1-loop matching.

---

## 7. Scope — which couplings does this apply to?

**This convention applies to:** framework-native tree-level α₁-dependent couplings derived under A5(b), including but not limited to:

- α₁, α₁_full (Layer-2 srs gauge coupling; intrinsic to the rendering)
- λ_Higgs (Higgs quartic; Level-2 srs-intrinsic)
- y_τ (tau Yukawa; Level-2 srs-intrinsic)
- v_Higgs (Higgs VEV; Level-2 with derived (5/12) Feshbach correction)
- V_us (CKM; Level-2 with derived (Im(h)/|h|²) Feshbach correction)
- m_ν (neutrino masses; Level-2 with same derived correction as V_us)
- θ_23 (PMNS; mass²-class with tan²(arg h) chirality content)

**This convention does NOT apply to:**

- Couplings that explicitly require Standard Model RG running (g_1, g_2, g_3, α_s, α_em, sin²θ_W at M_Z). These are renormalized SM couplings by definition and live in MS̄-at-scale by definition; they take observed M_Z as input and run via standard SM/MSSM machinery (`../parameters/target_parameters.md` row sin²θ_W). The framework's prediction at the **unification scale** (sin²θ_W(M_unif) = 3/8, theorem-grade per session 25) is in the framework's native scheme; the running to M_Z is outside this scope.
- SUSY threshold corrections to quark Yukawas / PMNS angles. These are framework commitments per `../parameters/target_parameters.md` SUSY rows, and their derivation under MSSM threshold mechanics is a separate research program (Priority 4.1). The convention here addresses Higgs-sector and lepton-Yukawa tree-level couplings, where SUSY contributions are explicitly ruled out per Results 22.4 and 26.4 of an external research note on the trivalent standard model.
- Cosmological observables (Λ, H_0, t_0). These have their own scope (internal working notes).

---

## 8. Implications for prior framings (what this supersedes)

This convention supersedes:

1. **The "scheme/scale convention" framing of `docs/master_plan.md` Priority 4.4 as originally written (session 25).** That framing posited that the framework needed to "commit to MS̄-at-some-scale and derive the natural μ from A5(b) + graph structure." Under the present convention, that framing was a category error: the framework is not in MS̄ and does not have a scale to derive. The deliverable of Priority 4.4 step 2.0 is the present convention; step 2.1+ is the still-open derivation of un-derived Feshbach analogs.
2. **The ~1% theoretical-uncertainty boilerplate in `predictions/m_H.py` (session 25 wrap).** That estimate cited Degrassi et al. 2012 NNLO SM matching. Under this convention, that citation is the wrong reference class. Clause 8 is now evaluated against σ_PDG only (see §5).
3. **The "+3.43σ tension" framing in an internal working note §1 — preserved as accurate, reframed in interpretation.** The figure is correct as the framework's actual full m_H prediction tension under σ_PDG alone (framework outputs m_H = 125.578 GeV against PDG 125.20 ± 0.11 GeV; the (5/12) correction on v matches v_obs essentially exactly via the G_F round-trip, so the entire residual lives on λ). It is not a λ-vs-1-loop-matching gap to be closed by picking a scheme, but the magnitude of the un-derived Feshbach analog on λ. The rest of that scoping doc's analysis (what falsified naive analog forms, what empirical hints exist) is preserved and load-bearing for step 2.1.

This convention does NOT supersede:

- The (5/12) derivation in `predictions/v_higgs.py` (theorem-grade; load-bearing here).
- The Feshbach amplitude correction derivation in an external research note on the dark-correction theorem §4a (load-bearing here).
- The session-25 m_H 1-loop scoping doc's enumerated falsifications (they remain valid; they tell us which naive Feshbach analog forms on λ DON'T work).
- A5(b) and the three-level hierarchy rule (`../theorems/theorem_A5b_level_prescription.md`). Tree-level λ = 2α₁_full and y_τ = α₁_full/k*² remain theorem-grade as **framework-native couplings**; the present convention governs how these framework-native numbers map to Standard Model observables, not how they're derived.

---

## 9. Minimal action list to bring the framework into compliance

1. ✅ **This document exists.** (Created 2026-04-25.)
2. **Update `framework_axioms.md`** to add a brief subsection / scope note pointing to this document as the bridge convention. (Done 2026-04-25 — see framework_axioms.md §6.5.)
3. **Update `../parameters/parameter_linter.md`** to include scheme-convention check in the gate sequence — specifically, that any prediction comparing a framework number to an SM observable must cite either this convention or an alternative bridge derivation. (Done 2026-04-25.)
4. **Update `docs/master_plan.md` Priority 4.4** to reflect the new framing (step 2.0 = this declaration, completed; step 2.1+ = un-derived Feshbach analog research, open). (Done 2026-04-25.)
5. **Update an internal working note** with a header note pointing to this convention; preserve the falsification analysis as load-bearing for step 2.1. (Done 2026-04-25.)
6. **Update `predictions/m_H.py`** to drop the ~1% theoretical-uncertainty boilerplate, point to this scheme-convention reference, and report Clause 8 against σ_PDG only. (Done 2026-04-25; theoretical-uncertainty band fully removed 2026-05-13.)
7. **Update `predictions/y_tau.py`, `predictions/m_tau.py`** with brief scheme-convention pointers. (Done 2026-04-25.)
8. **Future:** as new Feshbach analogs are derived (step 2.1), update this document's §4 with each new worked example. Residuals are reported as σ_PDG-only deviations; no theoretical-uncertainty band is introduced.

---

## 10. Cross-references

### Load-bearing structural derivations

- External research note on the dark-correction theorem §4a — Feshbach contour integral on the Ramanujan circle (linear chirality, amplitude class).
- External research note on the dark-correction theorem §4c.5b — (5/12) Higgs VEV correction (quadratic chirality, vertex class).
- `predictions/v_higgs.py` — (5/12) dark correction applied; theorem-grade per session 18+21.
- `../theorems/theorem_A5b_level_prescription.md` — A5(b) Case (A) for Level-2 srs-intrinsic couplings (load-bearing for tree-level λ, y_τ, α₁, θ_23).

### Scope & framing

- `framework_axioms.md` — canonical axioms (A1–A5); this convention sits below the axiom set as a bridge declaration.
- `framework_architecture.md` — multi-layer architecture (substrate / visible / dark) underpinning the substrate-projection picture.
- `docs/master_plan.md` Priority 4.4 — the work item that opened this scoping conversation; reframed under this convention (step 2.0 done, step 2.1+ open).

### Open problems referenced

- (open) Feshbach analog on y_τ — not yet investigated; opens with this document.

### External-research-note context (heritage)

- External research note on the trivalent standard model, Results 22.1, 22.4, 26.4 — SUSY committed for quark Yukawa / PMNS deviations only; explicitly ruled out for Higgs-sector and lepton-Yukawa tree-level. Justifies excluding SUSY contributions from the present convention's scope.
- That note's earlier (1+α_s/π) QCD threshold (Result 31.4) is superseded by the structurally-derived (5/12) correction and is not adopted in this repo. The two were addressing the same numerical slot; only the (5/12) is structurally derived.
