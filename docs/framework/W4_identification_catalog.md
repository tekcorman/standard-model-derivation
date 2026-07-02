# Per-file Spectral-Statistic Bookkeeping: Layer-1+ Predictions

**Date:** 2026-04-17
**Purpose (reframed):** Catalog which specific spectral statistic of `B` each Layer-1+ prediction file computes. Useful as a per-file audit index — which file uses eigenvalue magnitude, which uses character decomposition, which uses walk-survival, which uses TBM + perturbation.
**Upstream:** `../../predictions/walker_dynamics_derivation.md` (walker-observable gap closed, observables = spec(B) statistics by ruliad interpretation).
**Method:** bottom-up examination of each Layer-1+ file's specific spectral statistic.

> **Reframed from earlier draft (2026-04-17).** An earlier version of this document listed "postulates P1–P5 needed to close the gap." That framing treated W4 as a foundational gap requiring additional axioms. Under the ruliad interpretation (observer = MDL compression, observables = walker statistics by definition), no additional postulates are needed at the foundational level. What remains is per-file rigor audit: each prediction file must show that its specific computation on spec(B) follows from walker_dynamics + cited mathematics, under the standard rigor bar. The four Types (A/B/C/D) below are useful as organizational groupings for that audit, not as axiom candidates.

## 1. The four types of W4 identifications

Reading across the 19 Layer-1+ files, the identifications partition into four types:

| Type | Identification pattern | Example files | Spectral data used |
|------|------------------------|---------------|---------------------|
| A | mass amplitude = coherent sum over C3-irrep sectors of Ramanujan subspace | Q_Koide, ε_Koide, δ_Koide | C3-irrep multiplicities within 8-dim {h, h*, −h, −h*} subspace |
| B | CKM/PMNS mixing matrix element = NB walk survival amplitude at girth-related distance | V_cb, V_us, V_ub | (k−1)/k raised to a distance-dependent power |
| C | phase observable = eigenvalue-argument accumulated over a closed walk | α_21, α_31, δ_CP | arg(h), arg(h*) |
| D | mixing angle = TBM baseline (from Ramanujan degeneracy) + dark-extraction correction | θ_12, θ_13, θ_23, PMNS + CKM angles | C3 × parity decomposition of the h-eigenspace + Wigner-d rotation algebra |

Within each Type, the internal algebra is closed from the walker-dynamics theorem plus cited mathematics. The *identifications that link the spectral quantity to the specific observable* are what remain postulate-level.

## 2. Type-by-type audit

### Type A — mass amplitude

**Mapping:** generation-j mass amplitude `√m_j` is the `√(multiplicity)`-weighted coherent sum over C3 irrep sectors of the 8-dim Ramanujan-saturated subspace of `B(P)`.

Under this mapping:

- `√m_j ∝ √mult(trivial) + √mult(ω) · ω^j + √mult(ω²) · ω^{−j}` = `2 + √2 ω^j + √2 ω^{−j}` = `2(1 + √2 cos(2πj/3))`.
- Koide `ε = √2`, `Q = (k−1)/k = 2/3` recovered.

**Closed:** the algebra after the identification (§`predictions/Q_Koide_derivation.md` et al.). Verified numerically: `explorations/bp_h_eigenspace_c3.py`.

**Postulate needed (P-mass-amplitude):**
> The generation-j mass amplitude is the √multiplicity-weighted coherent sum over C3 irrep sectors of the 8-dim Ramanujan-saturated subspace {h, h*, −h, −h*} of `B(P)`.

**Sub-postulates inside P-mass-amplitude:**
1. The "mass amplitude" lives on the Ramanujan subspace (not the tree ±1 subspace).
2. The aggregation rule is √(multiplicity) within each irrep sector.
3. The aggregation is coherent (in-phase) across multiplicity copies.

These three sub-elements are arguably motivated (Ramanujan = non-trivial dynamics via Ihara zeta, √mult = maximum-amplitude coherent sum) but each is a postulate, not a theorem.

**Also needed for full closure of Type A:**
- Generation labelling (which `j` is electron, which is tau): symmetry-breaking input from δ, see Type C.
- Mass sector choice (why charged leptons specifically, not quarks or neutrinos): currently handled by dark_extraction_map's P2 parity work; needs verification.

### Type B — NB walk survival amplitude

**Mapping:** CKM/PMNS element = amplitude of a closed NB walk on srs at a girth-related distance.

Examples:
- `V_cb = α_1 · (1 + α_1)` with `α_1 = ((k−1)/k)^(g−2) = (2/3)^8`. Identification: "V_cb is the NB walk survival at distance g−2 plus one girth-cycle correction, summing the first two terms of the Ihara-Bass Green's function series."
- `V_us` similar, different distance.
- `V_ub` similar.

**Closed:** NB walk survival `((k−1)/k)^L` is elementary combinatorics (used inside theorem_walker_dynamics too). The geometric series `α_1 · (1 + α_1 + …)` sums as a specific Green's function on srs.

**Postulate needed (P-mixing-from-Green's-function):**
> CKM/PMNS matrix elements are specific matrix elements of the NB walk Green's function on srs, at a distance determined by the girth structure.

**Sub-postulates:**
1. Mixing elements are walk Green's function matrix elements (not eigenvalue ratios).
2. The "distance" for each specific element is fixed by structural features of the graph (e.g., girth g − 2, girth g − 1).
3. The generation pairing (which walk endpoint ↔ which quark) is fixed by additional rules.

Type B's rigor is weaker than Type A: the `c=1` coefficient in V_cb's girth-cycle correction is flagged in the file itself as "structural but not a formal proof" (`predictions/V_cb.py` lines 37–40). These sub-structural arguments are the tightest point in Type B.

### Type C — spectral phase / holonomy

**Mapping:** phase observable = `n · arg(h)` mod 360° for some integer `n` determined by the walker topology.

Examples:
- `α_21 = g · arg(h) mod 360°` (full girth cycle).
- `α_31 = 2g · arg(h) mod 360°` (two girth cycles for inter-generation transition).
- `δ_CP = (g−1) · arg(h*) mod 360°` (Jarlskog loop, one edge fixed by C3 transition).

**Closed:** de Moivre's theorem gives `arg(h^n) = n · arg(h)` — pure complex analysis. The specific `n` for each observable comes from walk-topology counting (e.g., Jarlskog invariant structure).

**Postulate needed (P-phase-from-holonomy):**
> Physical phases (Majorana, Dirac CP) are accumulated arguments of `h^n` (or `h*^n`) over specific closed walks on srs, with `n` determined by the walk's topological invariants.

**Sub-postulates:**
1. Phases are walk holonomies, not other spectral quantities.
2. The specific walk class (girth cycle, Jarlskog loop, etc.) for each phase observable is fixed.
3. CP conjugation maps h ↔ h* (follows from the Peskin-Schroeder definition, but needs the "h is the CP-covariant walker eigenvalue" identification).

### Type D — mixing angle from TBM + dark correction

**Mapping:** mixing angle θ = arctan-based formula combining TBM baseline (from Ramanujan degeneracy proved in theorem_BP) and a dark perturbation coefficient (from C3 × parity extraction map in `predictions/dark_extraction_map.py`).

Examples:
- `θ_23 = arctan((1 + α_1^full)/(1 − α_1^full))` with α_1^full = (5/3)α_1.
- `θ_12`, `θ_13` similar.

**TBM baseline is closed.** It follows from theorem_walker_dynamics + theorem_BP's C3-protected h-degeneracy. Degenerate perturbation theory (Sakurai §5.2) supplies the splitting algebra.

**Dark-extraction coefficients are NOT closed.** Audited this session against the rigor bar:

- Class 3 (edge-local, coefficient 1): CLOSED. Uses only Tr(σ_x) = 0 at C3-symmetric vertices — a consequence of character orthogonality (Serre 1977 §2.4 Theorem 3). Applies to θ_13, V_cb.
- Class 2 (mass², coefficient 5/3): BLOCKED. Chain: θ_23 derivation → dark_extraction_map.py Class 2 → Σ(h) = α_1/h from dark_correction_theorem_2026-04-14.md §4a. §4a's derivation uses:
  - A "Q-space" (ruliad complement) whose spectral density is POSTULATED uniform on the Ramanujan circle |λ| = √(k−1).
  - The uniform choice is justified against Kesten-McKay only by comparing numerical matches to observation (§4a' lines 444–454: "KM gives 1.9× too large, uniform matches"). This is fitting to observation, explicitly flagged as such in §4a'.
  - A contour-integral boundary pole (|h/√(k−1)| = 1, exactly on the contour) is handled by principal-value prescription and the author keeps only the interior residue. Non-trivial choice.
  - The "P·H·Q coupling strength = α_1" claim is asserted without derivation — "because the minimum-length ruliad excursion is a girth-cycle NB walk."
- Class 1 (amplitude, coefficient √5/4): BLOCKED for the same reasons as Class 2. Applies to V_us, m_ν2, m_ν3.

Also: `b_0 = 1/2` in Class 2 uses the formula `1/k* · k*/2 = 1/2` (dark_extraction_map_derivation.md line 83). The "k*/2" factor is unclear — the derivation does not state where it comes from.

**Net status for Type D:**
- θ_13 (Class 3): CLOSED.
- θ_23 (Class 2): BLOCKED at §4a's Q-space density ansatz.
- θ_12 (class TBD): depends on which class is invoked; likely blocked similarly.

**Postulate needed for Class 1/2 closure (P-dark-density):**
> The ruliad Q-space complement of the MDL-optimal srs projection carries a uniform spectral density on the Ramanujan circle |λ| = √(k−1), and the P·H·Q Feshbach coupling has strength α_1 = ((k−1)/k)^(g−2).

This is one postulate (or two sub-postulates), currently fit-motivated rather than derived. Under the framework's rigor bar, Type D's Class 1 and Class 2 observables are not theorem-grade.

## 3. Minimum postulate set

Consolidating the four Types, the minimum additional content needed beyond MDL + toggle + theorem_walker_dynamics to close all 19 Layer-1+ files:

**(P1) Ramanujan selection.** Physical observables depend only on the 8-dim Ramanujan-saturated subspace {h, h*, −h, −h*} of `B(P)`, not on the trivial ±1 tree subspace. Motivation: growth rate, Ihara zeta pole structure.

**(P2) Sector-aggregation rule.** When extracting from the Ramanujan subspace, aggregation within each C3 irrep sector uses `√(multiplicity)` coherent weighting. Motivation: symmetry-covariant observer, maximum amplitude principle.

> **2026-06-11 update (Phase 2.2 stage-1 panel, verdict PARTIAL).** P2 is
> REDUCED, not discharged. The MAGNITUDE half (√multiplicity weights) is now
> DERIVED: the joint {U(P), C3} CSCO of the unitary Bloch-Grover walker has 8
> one-dim walk eigenspaces with characters (4,2,2); uniform measure over the
> CSCO branches forces character weights (1/2, 1/4, 1/4) (conditional on the
> ADOPTED-P1/A5(a)-class walk-sector support + a NEW priced identification:
> uniform-over-CSCO-eigenmodes, a Jaynes domain-extension of
> theorem_multiway_branch_measure, ~1 bit). The RESIDUE (named, K2-class):
> (i) the conjugate-aligned per-channel phase read (z_α = |P_α ψ|, phases
> (0, +δ, −δ)) — load-bearing (generic phases give Q ∈ [1/3, 2/3]; the
> aligned read on the decohered ρ = I/8 also gives 2/3, so the √-coherence
> lives in the READ); derivation target: Hermiticity+positivity of a
> C3-circulant √M under R3 Fourier duality. (ii) "mass = Born weight at P" —
> a NEW A5-class identification (canonical A5(a) identifies eigenvalues,
> not Born weights, with masses; ~1 bit). Probe:
> proofs/foundations/phase2_2_born_koide_weights_2026-06-11.py.

**(P3) Observable-class mapping.** Each physical observable is assigned to one of the four Types A/B/C/D by its symmetry content:
- Scalar under C3 (mass, mass ratio) → Type A.
- Matrix element between different C3 sectors → Type B (Green's function) or D (TBM + perturbation), depending on diagonal vs off-diagonal.
- Phase-accumulating invariant → Type C.

**(P4) Class-specific encoding:**
- Type A: √m_j = `√mult(ρ) · character(ρ, j)` summed over irreps ρ.
- Type B: matrix element = specific Green's function kernel at a girth-related distance.
- Type C: phase = arg(h^n) for n from walk-topology invariants.
- Type D: angle = arctan formula with TBM + dark-extraction perturbation.

**(P5) Dark-extraction density.** The ruliad Q-space (complement of the MDL-optimal srs projection) has uniform spectral density on the Ramanujan circle, and the P·H·Q Feshbach coupling has strength α_1. This postulate is what the dark_correction_theorem §4a currently fits rather than derives. Without this postulate, Type D Classes 1 and 2 (θ_23, V_us, m_ν2, m_ν3) are not closed.

(P1) and (P2) are foundational. (P3) and (P4) are effectively encoding conventions that match each observable to its mechanism.

## 4. Rigor assessment

Under the framework's rigor bar (`../parameters/parameter_linter.md`), these postulates are ADDITIONS to the two axioms. They cannot be derived from MDL + toggle + walker_dynamics alone.

Each postulate is:
- **Motivated** (each has a physical / mathematical rationale).
- **Specific** (each states a concrete rule, falsifiable by checking against observations).
- **Non-derivable** from the existing axioms (no cited theorem produces it).

This is the Option-2 path from the W1–W4 walker-identification scoping doc: the framework must admit additional structural postulates beyond the two axioms. The present catalog replaces "19 case-by-case smuggled identifications" with "four postulates + algebraic closure."

**Honest reframing:**

The framework's current state is:
- Two foundational axioms (MDL + toggle).
- One structural theorem (walker_dynamics, closes W1–W3).
- Four additional structural postulates (P1–P4) covering the observable-identification gap (W4).

This is a stronger honest position than "two axioms + 19 smuggled identifications."

## 5. Can the four postulates reduce further?

Two plausible reductions to explore:

**Reduction A (unify P2 and P4):** If the `√multiplicity` aggregation rule is derivable from a single "symmetric-observer" principle (the observer's instruments commute with the symmetry group, so they see the symmetric sum of multiplicity copies), then P2 becomes a consequence of the observer-task specification. Requires care about what "symmetric observer" means mathematically.

**Reduction B (justify P1 from Ihara zeta):** If the Ihara zeta function's physical interpretation (poles = dynamical scales) is derivable from MDL on the walker's generating-function content, then P1 becomes a theorem. Requires development of the Ihara-zeta-from-MDL bridge, which is non-trivial.

Both reductions are plausible multi-week research projects, comparable in difficulty to closing W1–W3.

**Reduction C (reduce P3 and P4 to a single "observable class" theorem):** If there's a classification theorem showing that any C3-compatible observable on srs falls into exactly one of the four Types, then P3 and P4 merge into a single structural postulate. This is closer to the research plan's A4 Route (ii) (representation theory).

## 6. Next-session scoping

The cleanest next deliverables, in priority order:

1. **Verify the Type A closure on ε_Koide and δ_Koide.** Currently only Q_Koide has the numerical check. If ε and δ also emerge from (P1, P2) without additional inputs, Type A is fully nailed down.
2. **Audit dark_extraction_map.py rigorously.** If the C3 × parity derivation there passes the rigor bar, Type D is fully closed (mod TBM, which is already a theorem). This would reduce P3+P4 for Type D to a pure citation.
3. **Attempt Reduction A (√multiplicity from symmetric-observer).** If it works, one postulate collapses into the walker-dynamics theorem.
4. **Attempt Reduction B (P1 from Ihara zeta MDL).** Higher risk, higher payoff.

Any of 1–2 is a one-session deliverable. Item 3 is a multi-session project; item 4 is multi-week.

## 7. Files referenced

- the W1–W4 walker-identification scoping doc (W1–W4 gap statement)
- `../../predictions/walker_dynamics_derivation.md` (W1–W3 closed)
- `../../predictions/B_P_doubly_degenerate_h_derivation.md` (Ramanujan eigenstructure at P)
- `predictions/Q_Koide_derivation.md`, `predictions/epsilon_Koide_derivation.md`, `predictions/delta_Koide_derivation.md` (Type A)
- `predictions/V_cb.py`, `predictions/V_us_derivation.md` (Type B)
- `predictions/alpha_21_PMNS_derivation.md`, `predictions/alpha_31_PMNS_derivation.md`, `predictions/delta_CP_PMNS_derivation.md` (Type C)
- `predictions/theta_12_PMNS_derivation.md`, `predictions/theta_13_PMNS_derivation.md`, `predictions/theta_23_PMNS_derivation.md` (Type D)
- `predictions/dark_extraction_map.py` (invoked by Type D)
- `explorations/bp_h_eigenspace_c3.py` (numerical computation supporting Type A closure)
