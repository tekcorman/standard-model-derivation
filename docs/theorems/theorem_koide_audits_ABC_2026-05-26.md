# Theorem: Audits A/B/C of the Koide derivation chain (foundational structural inventory)

**Date:** 2026-05-26
**Status:** Theorem-grade decomposition of where the m_e/m_μ Koide-ratio defects D1/D3 actually live within the framework's existing derivation chain
**Predictions DAG:** UNCHANGED
**Probe script:** `proofs/foundations/W23_audits_ABC_koide_2026-05-26.py`
**Supersedes / continues:** `docs/theorems/theorem_m_e_m_mu_koide_observability_2026-05-26.md`
(observability theorem); an internal working note
(handoff that listed Audits A/B/C as next direction).

## Abstract

The framework's m_e and m_μ predictions carry two m_τ-independent residuals
(D1 = y_τ −10.8 ppm; D3 = m_e/m_μ ratio +9.83 ppm). The prior session
(W1-W22, 2026-05-26) attempted α₁_bare-power Family-D extensions to close
these and found all natural candidates falsified. This document presents
three structural AUDITS of the framework's *existing* Koide derivation
chain — not new mechanism families, but checks of where the residual
could possibly hide given the framework's actual derivations.

**Result:** the structural defect is sharpened from "unknown sub-leading
correction" to "the cos-form δ ≠ 0 in `predictions/m_e.py` and
`predictions/m_mu.py` is not derived from substrate." The defect is
ALREADY ACKNOWLEDGED in `predictions/delta_Koide_derivation.md` as
Need-B (the framework's own scoping doc names it explicitly).

The three audits decompose the surface as follows:

| Audit | Question | Verdict |
|---|---|---|
| A | Are (μ_t, μ_ω, μ_ω̄) = (4, 2, 2) topologically exact? | **YES — settled negative as residual source.** Algebraic identity from Ihara-Bass + Schur. |
| B | Is the cos-form `f_j = 1 + ε·cos(2πj/k* + δ)` substrate-derived? | **NO — D3 lives here.** The substrate Q_Koide construction gives δ ≡ 0 (m_1 = m_2 degenerate). δ = 2/9 in m_e.py is a numerical-coincidence parametric input. |
| C | Does walker holonomy h^g enter mass eigenvalues to break the degeneracy? | **Candidate mechanism but value mismatch.** Natural integer L gives walker phases that don't match δ_emp ≈ 0.222 rad. |

The synthesis: the structural defect is located in the substrate→cos-form
bridge, which is parametric. No new mechanism families are needed in
future sessions — the work is to derive a SUBSTRATE EXPRESSION for the
m_1 ≠ m_2 split (or equivalently, the cos-phase δ) that doesn't rely on
the numerical-coincidence identification δ_Bernoulli = Q(1−Q).

## Audit A — (μ_t, μ_ω, μ_ω̄) = (4, 2, 2) topologically exact

### Claim

The C_3-isotypic multiplicities (μ_t, μ_ω, μ_ω̄) = (4, 2, 2) of the
8-dim Ramanujan subspace V_Ram of B(P) are integer DIMENSIONS of finite-
dimensional subspaces, fixed by algebraic identities. They admit no
ppm-scale corrections at any structural layer.

### Derivation (already established in framework)

Per `predictions/B_P_doubly_degenerate_h_derivation.md` Steps 5 + 9:

1. **A(P) characteristic polynomial.** The 4×4 Hermitian scalar Bloch
   adjacency at P = (1/4, 1/4, 1/4) has char poly (λ² − 3)². So A(P)'s
   eigenvalues are ±√3, each with multiplicity 2.

2. **C_3 invariance of A(P).** [A(P), P_σ] = 0 where σ = (v_0)(v_1 v_3 v_2)
   is the body-diagonal rotation. A(P) and P_σ share an eigenbasis.

3. **C_3 isotypic decomposition of A-eigenspaces** (Step 5, corrected
   2026-04-15):
   - (+√3) eigenspace = (trivial) ⊕ (ω)
   - (−√3) eigenspace = (trivial) ⊕ (ω²)

4. **Ihara-Bass transport to B(P).** Each A-eigenvalue λ lifts to two
   B-eigenvalues μ via μ² − λ·μ + (k−1) = 0:
   - λ = +√3 → {h, h*} where h = (√3 + i√5)/2
   - λ = −√3 → {−h, −h*}
   The squared inner factor in Ihara-Bass `(4u⁴ + u² + 1)²` gives each
   B-eigenvalue multiplicity 2.

5. **C_3 isotypic transport.** The Ihara-Bass map preserves C_3 action.
   Each A-mode lifts to TWO B-modes (one in each ±h sector), so:
   - h-eigenspace (2-dim) = (trivial) ⊕ (ω)
   - h*-eigenspace (2-dim) = (trivial) ⊕ (ω)
   - (−h)-eigenspace (2-dim) = (trivial) ⊕ (ω²)
   - (−h*)-eigenspace (2-dim) = (trivial) ⊕ (ω²)

6. **Counting on V_Ram.** Summing per isotypic across the four B-
   eigenspaces:
   - trivial: 1+1+1+1 = 4 ✓
   - ω: 1+1 = 2 ✓
   - ω²: 1+1 = 2 ✓
   Total = 8 = dim V_Ram ✓

### Why this rules out (4,2,2) as residual source

The multiplicities are integer DIMENSIONS, not coupling constants or
spectral expectation values. They cannot acquire fractional corrections.
A possible "sub-leading correction" to (4,2,2) would need to be either
(a) a different decomposition altogether (which violates Schur's lemma
applied to A(P)), or (b) a non-integer dimension (which contradicts the
definition of subspace dimension on a finite-dim Hilbert space).

**Verdict A**: settled negative. The (4,2,2) multiplicities are not the
source of D1 or D3 residuals.

## Audit B — the cos-form is parametric, not substrate-derived

### Claim

The cos-form `f_j = 1 + ε·cos(2πj/k* + δ)` used in
`predictions/m_e.py` and `predictions/m_mu.py` with δ = 2/9 is a
parametric phenomenological extrapolation, NOT derived from the
substrate. The δ ≠ 0 introduces the m_1 ≠ m_2 split that the substrate's
Born-rule construction does not produce.

### The two cos-forms in the framework

**Form 1 — substrate-derived** (`predictions/Q_Koide.py`):

amp_j = √μ_t + √μ_ω · ω^j + √μ_ω̄ · ω^{−j}

With (4, 2, 2): amp_j = 2 + 2√2 · cos(2πj/3).
- j=0: amp = 2 + 2√2 ≈ 4.828
- j=1: amp = 2 − √2 ≈ 0.586
- j=2: amp = 2 − √2 ≈ 0.586 ← identical to j=1

Born rule m_j = |amp_j|² gives:
- m_0 = 12 + 8√2 ≈ 23.31 → m_τ slot
- m_1 = m_2 = 6 − 4√2 ≈ 0.343 → m_μ = m_e degenerate

**This form has NO δ in the cos argument.** It produces a degenerate
spectrum {m_τ, m_μ = m_e}.

**Form 2 — parametric phenomenological** (`predictions/m_e.py`,
`predictions/m_mu.py`):

f_j = 1 + ε · cos(2πj/k* + δ)

With ε = √2, δ = 2/9 rad:
- j=0: f ≈ 0.0403 (electron — near-cancellation)
- j=1: f ≈ 0.5802 (muon)
- j=2: f ≈ 2.3794 (tau)

m_j = m_τ · (f_j/f_max)² produces three distinct masses matching PDG.

### δ is a FREE parameter in the cos-form

Algebraically: for f_j = 1 + ε·cos(2πj/3 + δ),

Q = Σ_j f_j² / (Σ_j f_j)² = (1 + ε²/2)/3

which is INDEPENDENT of δ (verified symbolically in W23). The cos-phase
δ is a free parameter that does NOT enter Q at all. It only modulates
the SPREAD of the three f_j values around their common mean.

### The framework's own admission

`predictions/delta_Koide_derivation.md` line 3 (2026-05-08 status update):

> NOTE: the IDENTIFICATION of δ_Bernoulli (variance, dimensionless) with
> the Koide cosine PHASE δ in radians (the parameter that gives 3-distinct
> lepton mass values via sqrt(m_j) = sqrt(M)·(1+ε·cos(2πj/3+δ))) is a
> NUMERICAL coincidence (2/9 ≈ 12.73° matches observed Koide phase).
> Whether this coincidence has a structural derivation is **Need-B** of
> an internal working note — a SEPARATE multi-
> session research question.

The framework has δ_Bernoulli := Q(1−Q) = 2/9 (a dimensionless variance
moment, in `predictions/delta_Koide.py`), and observed Koide cos-phase
δ_emp ≈ 0.2222 rad ≈ 12.73°. Their numerical equality (when interpreting
δ_Bernoulli as a value in radians) is a dimensional category error
that is acknowledged but not derived.

### Why this is the D3 source

The substrate construction gives m_e = m_μ. The framework's reported
m_e ≠ m_μ predictions in `predictions/m_e.py` / `predictions/m_mu.py`
use a DIFFERENT formula (cos-form with δ ≠ 0) that produces non-
degenerate masses by introducing a free phase parameter.

D3 (= +9.83 ppm m_e/m_μ ratio residual) measures the gap between the
δ = 2/9 numerical-coincidence value and the actual best-fit Koide phase
to PDG. It is the residual error of using "δ = Q(1−Q) interpreted as
radians" rather than the precise empirical phase. This residual is
NOT a precision floor — it is the imprint of a parametric assumption
that has no derivation.

**Verdict B**: D3 lives here. The structural defect is the cos-phase δ
in the f_j parametrization. Need-B is its name in the framework.

## Audit C — walker holonomy h^g as candidate cos-phase mechanism

### Question

Could walker holonomy at length g produce a cos-phase δ via a CC-conjugate
asymmetry between V_Ram's ω and ω² isotypics?

### Structural possibility

If the substrate amp construction included walker-holonomy weighting:

amp_j_walker = √μ_t + √μ_ω · h^g · ω^j + √μ_ω̄ · (h*)^g · ω^{−j}

then the ω and ω̄ isotypic amplitudes acquire CC-conjugate phases. With
μ_ω = μ_ω̄ = 2, this collapses to:

amp_j_walker = √μ_t + 2√μ_ω · cos(2πj/3 + arg(h^g))

precisely the cos-form with δ = arg(h^g).

This IS a structural mechanism that produces a δ ≠ 0 in the cos-form
from substrate first-principles.

### Numerical check (W23)

For arg(h) = arctan(√5/√3) ≈ 52.239° = 0.91152 rad:

| L | arg(h^L) mod 360° | rad | m_μ/m_e predicted | Note |
|---|---|---|---|---|
| 1 | 52.24° | 0.912 | 14.6 | — |
| 7 | 5.67° | 0.099 | 5.67 | closest small phase |
| 10 (girth) | 162.39° | 2.834 | 14.0 | α_21 PMNS Majorana phase |
| 14 | 11.34° | 0.198 | 69.2 | closest to δ_emp = 2/9 |
| — | — | 0.222 (emp) | 206.8 | |

The framework's natural L = 10 (girth) gives 162.39° = 2.834 rad — off
from δ_emp = 0.222 rad by a factor of 13.

The closest natural integer L to reproducing δ_emp is L = 14 with
φ = 11.34° = 0.198 rad — still 11% off and giving wrong m_μ/m_e (69
vs 207).

Solving L · arg(h) mod 2π = 2/9 exactly: L ≈ 0.244 + 6.89·n for integer
n. No natural integer L produces this.

### Verdict on Audit C

Walker holonomy is a **viable structural mechanism** for the cos-phase
(it would produce δ = arg(h^L) for some characteristic L). But the
**natural framework values don't match observation**:

- L = 10 (girth, the framework's PMNS phase choice): 162.4°, ≫ δ_emp
- All small integer L: don't reproduce 12.73° at any clean L
- Best integer match L = 14: 11% off in phase, 3× off in m_μ/m_e

**Verdict C**: walker holonomy could in principle produce the cos-phase,
but the framework's natural walker-length scales (girth g = 10, length L = 1)
don't reproduce δ_emp at any precision useful for closing D3.

A "non-natural" L (e.g., L ≈ 14, or fractional L) would need a structural
motivation that the framework doesn't currently have. So Audit C does
NOT close the defect, but it sharpens what would be required: a structural
identification of the characteristic walker length L_Koide ≈ 0.244 (mod
2π/arg(h)) ≈ 7.14 ≈ 14.03 that determines the mass-Koide phase.

## Synthesis

The three audits decompose the open structural question for m_e/m_μ
Koide-ratio residuals as follows:

```
Substrate (B(P) on srs)
  ↓ Born rule (CDP 2011)
  ↓ Jaynes max-entropy on V_Ram
  ↓ C_3 Fourier transform with (4,2,2) multiplicities ← Audit A: exact
  ↓
amp_j = 2 + 2√2·cos(2πj/3)  ← substrate-derived form
  ↓
m_j (substrate) = (m_τ, m_μ = m_e, m_e = m_μ)  ← DEGENERATE
  ↓
  ⊥ [STRUCTURAL DEFECT D3 — Audit B identifies]
  ↓
f_j = 1 + ε·cos(2πj/3 + δ)   ← parametric phenomenological form
                              with δ = 2/9 (numerical coincidence)
  ↓
m_j (phenomenological) = (m_τ, m_μ, m_e) ← matches PDG
```

The defect is the parametric injection of δ ≠ 0. The substrate
provides ε = √2 (theorem-grade) but NOT δ.

Audit C identifies walker holonomy as a structural mechanism that
*could* provide δ from substrate, but the natural framework length
scales don't match.

## What this implies for future structural work

The Koide cos-phase δ is the **last unclosed substrate-to-mass primitive**
in the lepton sector. It is named Need-B and acknowledged as open.

For future sessions:

1. **Do NOT chase α-power K-rational fits to D3.** That surface was
   exhausted by W4-W22. The defect is upstream of α-power expansions —
   it's in the substrate→cos-form bridge.

2. **The right direction is deriving δ structurally.** Three candidate
   directions:
   - **C-extended:** walker holonomy at non-natural L (would require
     justifying L ≈ 14 or fractional L via framework primitive)
   - **C³_gen sub-structure:** the C³_gen / Need-D-3 problem (the
     unsolved generation-Z₃ closure) may carry the cos-phase
     information naturally
   - **Different functional form:** the substrate's amp_j may admit
     non-cos sub-leading harmonics (next-order correction in the
     Jaynes max-entropy that breaks the j=1 ↔ j=2 degeneracy)

3. **The Yukawa-budget framing is rejected** (per user 2026-05-26).
   The residuals are defects with named structural origins (D1, D3),
   not "within ~0.5% systematic floor." The framework owes a
   substrate derivation of δ.

4. **Predictions DAG remains UNCHANGED.** No predictions/*.py file
   modification is justified by these audits. The current grades stand
   with D1/D3 named explicitly as Need-B-blocked.

## Honest grade

The three audits produce one structural-conditional theorem, two
structural-elimination results, and one candidate-mechanism identification:

- **Theorem (Audit A):** (4,2,2) multiplicities are topologically exact;
  not the residual source. Theorem-grade-structural under Ihara-Bass +
  Schur (existing framework derivations).
- **Theorem (Audit B):** the cos-form δ = 2/9 is parametric numerical-
  coincidence, NOT substrate-derived. The framework already names this
  as Need-B. Theorem-grade-structural identification of the defect
  location.
- **Candidate (Audit C):** walker holonomy h^g is a viable structural
  mechanism for producing a substrate-derived δ, but the natural framework
  L = 10 (girth) gives 162.4°, far from the empirical 12.73°.
  Candidate-grade structural identification with magnitude mismatch.

This is honest structural progress: the work is no longer "find a magic
K-rational form for the residual" but "derive δ structurally from
substrate." That is a deep, multi-session research question (Need-B)
NOT a precision floor.

## Cross-references

- `docs/theorems/theorem_m_e_m_mu_koide_observability_2026-05-26.md` —
  prior observability decomposition (m_τ-uncertainty propagation;
  m_τ-INDEPENDENT direct test for D3)
  fresh-context handoff that listed A/B/C as candidate next directions
- `predictions/delta_Koide_derivation.md` — framework's own admission
  of Need-B
- `predictions/Q_Koide_derivation.md` — substrate (4,2,2) derivation
- `predictions/B_P_doubly_degenerate_h_derivation.md` — Audit A inputs
- `proofs/foundations/W23_audits_ABC_koide_2026-05-26.py` — this session's
  numerical probe (output: A confirmed; B substrate vs phenomenological
  forms compared; C walker phase scan)
- `proofs/foundations/W45_nu_R_modecount_holonomy_2026-05-21.py` — walker
  holonomy mechanism applied to PMNS Majorana phases (the existing
  framework precedent for h^g entering an observable)

## What this session did NOT do

These audits IDENTIFY the structural defect and rule out (4,2,2)
multiplicities as its source. They do NOT close Need-B (the substrate
derivation of δ). That remains the open structural research question
for future multi-session work.
