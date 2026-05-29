# W22 — Path to agreement, NO Yukawa-budget fudge

**Date:** 2026-05-26
**Supersedes W21:** strips the "~0.5% Yukawa systematic budget" framing per user pushback (correct: that's an admission of failure, not a legitimate precision limitation).

## What the "0.5% Yukawa budget" actually is

Per parameter_linter.md Clause 8b and master doc §8b, the framework names a "~0.5% un-derived sub-leading Feshbach analog" for Yukawa-derived quantities. This was added to allow THEOREM-GRADE-STRUCTURAL grade for predictions whose numerical residuals exceed σ_PDG.

**The user is correct: this is a fudge.** A framework claiming to derive the SM from substrate first-principles should not need a 0.5% "we haven't derived this yet" budget. Calling residuals "within budget" is intellectual cover for missing derivations.

**Honest reframing:** every sub-percent residual is an OPEN STRUCTURAL DEFECT, not a "named systematic." The framework owes derivations for ALL of them, not budget-covering.

## The actual open defects

After this session's analysis (W4–W21), the framework has the following ACTUAL OPEN STRUCTURAL DEFECTS in the m_e/m_μ/m_τ/y_τ sector:

### Defect 1: y_τ residual (−11 ppm)

`predictions/y_tau.py`:
- y_τ_pred (Family-D α₁²) = 0.00721647
- y_τ_obs = m_τ_obs/v_obs = 0.00721655
- Residual = −10.8 ppm

**Root cause:** master doc §3 D Family-D only derives the α₁² leading correction. The α₁³ and higher-order Family-D corrections are NOT DERIVED.

**Magnitude expectation:** if Family-D extends to α₁³ via Route H joint walker or Route C cycle counting at length 3(g−2), the α₁³ piece is at ~60 ppm scale (= α₁³ = (2/3)²⁴). The y_τ residual of −11 ppm is consistent with a sub-leading c_H or c_F coefficient at α₁³ order with K-rational denominator ~5.5.

**What's needed:** derive the α₁³ Family-D y_τ correction structurally. This was blocked in this session by:
- 24-cycle Route C decomposition falsified (W12, 12.2% rate)
- 3-way joint walker Route H blocked by R-9 closure (W14-W15)
- 2-cycle 24-cycle decompositions falsified (W13, 41% combined)

The α₁³ Family-D mechanism requires either:
(a) An alternative substrate mechanism at α₁³ not yet identified
(b) Multi-cycle decompositions (4+ cycles) on H(srs)
(c) Spectral approach via tr(B^L) with specific channel projections

All research-level.

### Defect 2: m_τ residual (−13 ppm)

`predictions/m_tau.py`: m_τ_pred = v · y_τ. With v matching exactly via G_F (round-trip absorbs into N_hub anchor), m_τ residual EQUALS y_τ residual propagation: ~−13 ppm.

**Closure:** automatic once Defect 1 closes.

### Defect 3: m_e/m_μ Koide-ratio m_τ-independent asymmetry (+9.83 ppm)

`predictions/m_e.py` and `predictions/m_mu.py` use Koide formula m_j = m_τ · (f_j/f_max)². The bare ratio r_e_bare/r_μ_bare = 4.836284e−3 differs from m_e_obs/m_μ_obs = 4.836332e−3 by **+9.83 ppm**.

This is m_τ-INDEPENDENT (cancels in the m_e/m_μ ratio).

**Root cause:** the bare Koide formula with δ = 2/9 = Q(1−Q) doesn't exactly match observation at the 10 ppm precision level. Either δ is slightly off, ε is slightly off, or the cos-form is missing sub-leading harmonics.

**What's needed:** derive the sub-leading correction to the Koide formula structurally. Candidate mechanisms:
- Berry-phase Family-A at α₁³ rep-resolved (W7 candidate): δ_Berry · sgn_rep(j) with γ ≈ 1/(2k*²)
- Sub-leading modification to ε² = 6Q−2 or δ = Q(1−Q) at higher order
- Per-rep dark correction on V_Ram amplitudes (W18/W20 candidate)

None has clean K-rational derivation yet.

### Defect 4 (composite): m_e and m_μ residuals depend on m_τ_pred

When using framework's m_τ_pred = 1.776837 (which carries Defect 2):
- m_e residual = −84 ppm = −13 ppm (m_τ defect) + −70 ppm (Koide-ratio gap at m_τ_obs central)
- m_μ residual = −74 ppm = −13 ppm + −61 ppm

The Koide-ratio gap at m_τ_obs central could in principle be anywhere from +4 ppm to +138 ppm depending on where TRUE m_τ sits within PDG ±67 ppm — this is m_τ-DEPENDENT and cannot be cleanly separated from Defect 2 closure without external m_τ improvement.

But the m_τ-INDEPENDENT signal (Defect 3 = +9.83 ppm) is what the framework MUST close at theorem-grade regardless of m_τ PDG status.

## The honest path: derive every defect

No budgeting. Each defect requires a structural theorem.

### Theorem to derive — α₁³ Family-D for y_τ (closes Defect 1 + 2)

**Target:** y_τ_corrected = y_τ_tree · (1 − (5/6)α₁² + δ_α₁³_y_τ) where δ_α₁³_y_τ = +11 ppm

**Specific K-rational candidates** (need structural derivation):
- α₁³/5.5 = 10.8 ppm — coefficient 1/5.5 not clean
- α₁³ · (1/3 − some integer) — various
- α₁³ · √5/something — Berry-like sub-leading

**Difficulty:** α₁³ Family-D Route C extension was FALSIFIED in W12 (24-cycle decomposition doesn't work). Route H 3-way joint walker BLOCKED by R-9 (W15). A new mechanism at α₁³ order is needed.

**Possible approaches:**
1. Spectral tr(B^24) decomposition — characterize what's actually in the joint walker survival at length 24 on srs alone (not joint with srs-z)
2. Multi-cycle decomposition (4+ cycles) — extend beyond 2/3-cycle compositions
3. Family-A sub-leading at α₁² order — Berry-phase contribution to y_τ via Im(h)/|h|² = √5/4

Research-level, multi-session.

### Theorem to derive — α₁²/α₁³ Koide-asymmetry correction (closes Defect 3)

**Target:** modify Koide ratio prediction at the 10 ppm m_τ-independent level.

**W7 candidate** (best so far): Berry-phase sub-leading at α₁³ with rep-dependent sign
   δ_Berry_j = c_A · α₁³ · sin(arg h) · sgn_rep(j)
   c_A = 1/(2k*²) gives 5.22 ppm asymmetry at f-level; observed 4.92 ppm; match at 94%.

**What's needed:**
1. Derive c_A = 1/(2k*²) from substrate Family-A mechanism at α₁³ rep-resolved
2. Account for the 6% magnitude mismatch (possibly α₁⁴ refinement)
3. Master doc §3 A extension theorem

### Combined: derive ALL FOUR α-orders down

The framework's existing derivations cover α₁¹ (v_Higgs, Family-C) and α₁² (Family-D leading). Defects 1-3 all require α₁³-order extensions that haven't been derived.

**The proper framework completion** requires extending the Family system to α₁³:
- α₁³ Family-C (universal scale correction)
- α₁³ Family-D (per-leg vertex correction) — addresses Defect 1
- α₁³ Family-A (Berry-phase Berry-rep-resolved) — addresses Defect 3
- α₁³ Family-E (custodial-breaking) — affects δρ but parallel structure

This is the framework's MISSING α₁³ THEOREMS. The "0.5% Yukawa budget" is hiding the fact that NONE of these α₁³ extensions have been derived.

## Honest grade

The framework's m_e, m_μ, m_τ, y_τ predictions are at **THEOREM-GRADE-STRUCTURAL with three named open defects (D1, D2, D3 above)**, NOT at "theorem-grade within ~0.5% budget."

Each open defect represents a specific missing α₁³ extension theorem. The framework has α₁¹ and α₁² derivations; it lacks α₁³ throughout.

## What needs to happen for PDG-precision agreement

Without budget hiding:

1. **Derive α₁³ Family-D for y_τ** — closes Defects 1+2 (m_τ residual goes to 0)
2. **Derive α₁³ Family-A rep-resolved Berry-phase** — closes Defect 3 (Koide-asymmetry goes to 0)
3. **(Probably also) derive α₁³ Family-C** — affects v_Higgs sub-leading, currently absorbed in N_hub anchor
4. **(For full sector) derive α₁³ Family-E** — affects δρ sub-leading

After all four α₁³ extensions, the framework's predictions for m_e, m_μ, m_τ, y_τ should match PDG observations to within PDG precision (limited only by external measurement uncertainty).

## What's actually been done this session

The W4–W21 work attempted α₁³ extensions and discovered SEVERAL of the natural candidates DON'T WORK:
- W12: 24-cycle Route C extension (12.2% rate vs 100% at 16-cycle) — FALSIFIED
- W13: 2-cycle 24-cycle decompositions (41% combined) — FALSIFIED
- W14-W15: 3-way joint walker Route H — STRUCTURALLY BLOCKED by R-9
- W17: c_F + c_F^(rep) addition — BREAKS m_τ closure
- W18/W20: c_F^(rep) vanishing at trivial — L-expression valid but mechanism underivable from existing primitives
- W7: Berry-phase at α₁³ with γ = 1/(2k*²) — 94% match, coefficient ad hoc

The session HONEST output: the natural α₁³ extensions don't close cleanly within existing framework structure. The framework needs NEW structural ingredients at α₁³ order — currently missing.

## What I owe (corrected from W21)

I shouldn't be saying "we're at agreement within budget." I should be saying:

**The framework has three named open structural defects (D1, D2, D3). Closing them requires deriving the α₁³ extensions of Family-D and Family-A. Several natural candidates have been falsified or block-tested in this session; a successful derivation requires either NEW structural ingredients OR identification of mechanism families I haven't enumerated.**

The path to agreement is HARD. It's research-level, multi-session, and requires not budgeting around the difficulties.

## Specific next actions (if pursuing)

1. **Survey what's structurally novel at α₁³ that we haven't tested.** Beyond Route H/C extensions and 2/3-cycle decompositions. Possible candidates:
   - Spectral tr(B^L) characterization at L=24 (cycle counts)
   - Multi-cycle compositions (4+ cycles)
   - Mixed-mechanism (e.g., girth+girth+girth with crossings or interference)
   - Non-cycle objects (paths with specific endpoint conditions)

2. **Look at related framework α₁³ closures** for analogous mechanisms. Does the framework derive ANY α₁³ correction anywhere? If yes, that's the template.

3. **Take the W18/W20 candidate seriously as a NEW mechanism family** — possibly Family-D needs a "rep-resolved" extension that the master doc doesn't currently have. Write the full theorem.

4. **Take the W7 Berry-phase candidate seriously** — derive c_A = 1/(2k*²) structurally, not just by data-fitting.

## Predictions DAG status

**UNCHANGED.** And per the honest reframing, the framework's m_e, m_μ, m_τ, y_τ predictions should be reported NOT as "within Yukawa budget" but as "STRUCTURAL-WITH-NAMED-DEFECTS (D1, D2, D3)" pending α₁³ extensions.

The HONEST status is: the framework has unfinished α₁³-order theorems for the Yukawa sector. The remaining residuals are structural defects, not "budget."
