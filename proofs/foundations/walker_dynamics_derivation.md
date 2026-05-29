# Walker dynamics on srs — Hashimoto B as the 1-step amplitude operator

**Date:** 2026-04-17 (ported 2026-04-19)
**Status:** theorem (rigor: closed). Closes W1-W4.
**Script:** `predictions/walker_dynamics.py`
**Detailed proof script:** `proofs/foundations/theorem_walker_dynamics.py`
**Supersedes:** the W1–W4 walker-identification scoping doc (W1-W4 gap), `../predictions/walker_dynamics_derivation.md`.

**Post-2026-05-08 axiom slate note.** A1 and A2 (cited as framework axioms below) are now derived theorems of (A) self-containment + (B) finite observer + standard math + (I) active reading, per `docs/theorems/theorem_toggle_from_self_containment.md` and `docs/theorems/theorem_A2_mdl_from_finite_register.md`. References to "A1" and "A2" remain semantically valid; the W1–W4 closures are unchanged. See `docs/framework/framework_axioms.md` §10 for the updated top-level summary.

## Abstract

Under axioms A1 (binary self-inverse toggle) and A2 (MDL canonicalization), the MDL-optimal observer on the srs lattice is equivalent to a non-backtracking (NB) walker, with the Hashimoto operator B as the 1-step amplitude operator. This theorem closes four previously-open claims:

- **W1**: Observer's data representation = reduced word in F_inv(E) = NB walk of srs.
- **W2**: Causal state = current directed edge (2|E|=12 states per primitive cell).
- **W3**: 1-step amplitude operator = Hashimoto B; L-step amplitude = B^L.
- **W4**: Physical observables are spectral statistics of B (corollary of the ruliad interpretation).

## Framework axioms invoked

- **(A1)** Binary self-inverse toggle (`docs/framework/framework_axioms.md` §2): each edge e carries T_e with T_e*T_e=I. Toggle streams form the free involutive monoid F_inv(E).
- **(A2)** MDL (`docs/framework/framework_axioms.md` §3): observer selects the reduced-word (minimum-length) representative of each equivalence class in F_inv(E).

## Reading conventions (four readings of B)

The matrix identity B[e',e]=1 iff e->e' is a valid NB transition is reading-neutral. What differs is the interpretation of matrix elements:

1. **(Markov)**: B is a transition-probability matrix. Used in cosmological/dark-sector predictions.
2. **(Unitary)**: B generates complex amplitude evolution; eigenvalue h=(sqrt(3)+i*sqrt(5))/2 carries genuine phase information. Used in flavor predictions (Koide, CKM, PMNS).
3. **(Open System — framework's most accurate reading)**: B is the visible-sector unitary part; W2 cancellation events are Lindblad jump operators coupling the Hilbert visible sector to the non-Hilbert dark measure space. This is the only reading consistent with the dark-sector-non-Hilbert constraint (dark content fails compressibility; Gleason structure unavailable on dark).
4. **(Agnostic)**: Spectral data of B (eigenvalues, multiplicities, Bloch dependence) are reading-invariant.

## Proof chain

```
Axiom A1 (self-inverse toggle)
      |
      v
Step 1: Free involutive monoid F_inv(E)         [Serre 1980 §I.1 Prop. 4]
      |
      v
Step 2: MDL canonicalization = reduced words     [A2 + Grunwald 2007 §5.1-5.3]
      |
      v
Step 3: Reduced words = NB walks                 [Serre 1980; Terras 2011 §2.1]
      |                                          (closes W1)
      v
Step 4: Jaynes-uniform over k-1=2 NB choices     [Jaynes 1957 + d_spatial]
      |
      v
Step 5: Causal state = directed edge             [Shalizi-Crutchfield 2001 Thm 2]
      |                                          (closes W2)
      v
Step 6: 1-step operator = Hashimoto B            [Hashimoto 1989; Terras 2011 §2.2]
      |
      v
Step 7: L-step amplitude = B^L                   [matrix composition]
      |                                          (closes W3)
      v
Step 8: P-point: h with mult 2 (C3-protected)    [theorem_BP_doubly_degenerate_h.md]
      |
      v
Ruliad interpretation: observer = compression    [operational reading]
      |                                          (closes W4 as corollary)
      v
Spec(B) = set of physical observables
```

## Step-by-step derivation

### Step 1 — Free involutive monoid from A1

A1 forces T_e*T_e=I for each edge e. Toggle streams form the free monoid E*; under the congruence {e*e ~ epsilon}, we get F_inv(E) = E* / (e*e ~ epsilon), the free involutive monoid (free product of Z/2 groups; Serre 1980 §I.1 Proposition 4).

### Step 2 — MDL canonicalization selects reduced words

Each equivalence class [w] in F_inv(E) has a unique minimum-length representative r(w) with no two adjacent equal letters (reduced word; uniqueness: Serre 1980, Terras 2011 §2.1). Under A2, among representations with identical predictive content, MDL picks the one with minimum symbol count. Since description length = symbol count (up to constant; Shannon 1948, Cover-Thomas 2006 Theorem 5.4.3), the MDL encoder emits reduced words.

### Step 3 — Reduced words = NB walks

For a graph-admissible reduced word (each letter incident to the current vertex), the walk induced by orienting edges by traversal direction has no consecutive reverse-edge step. This is the NB condition. Bijection: Serre 1980 §I.1 Proposition 4; Terras 2011 §2.1. Closes W1.

### Step 4 — Jaynes-uniform over k-1 NB choices; isotropy ⟹ arc-transitive substrate ⟹ (Sunada) srs

**4a — the (k−1)/k value.** Jaynes 1957 maximum-entropy applied to toggle events at a k=3 vertex gives the uniform distribution over the 3 edges. Why uniform: by (A) (self-containment — `theorem_toggle_from_self_containment.md` Step 1), nothing is supplied from outside that could privilege one direction over another, so the max-entropy distribution carries zero "which-direction" information ⟹ uniform — the same no-privilege principle that forces the uniform substrate measure (toggle theorem Step 1) and the absent inter-generator commutation (Step 7). Under Step 2 canonicalization, conditional on the walk being extended, the next edge is then uniform over the k−1=2 NB continuations, so the per-step NB survival is (k−1)/k. This value holds at every directed edge of any k-regular graph (k-regularity + Jaynes); arc-transitivity is not needed for the *value*. (The backtrack case is erased from the observer's compressed data by A2; see Open System reading for the Lindblad interpretation.)

**4b — isotropy is observer-side, and it forces an arc-transitive substrate.** The walker's causal state is a *directed edge* (Step 5 below — Shalizi-Crutchfield 2001). By (A), nothing supplied from outside privileges any direction at a vertex *or* either orientation of an edge: a privileged direction/orientation would be a piece of "which-way" information, and (A) forbids supplying it (this is the toggle-theorem-Step-1 no-privilege principle applied to *spatial* labels — the same principle the d-spatial derivation already invokes when it works "under isotropic toggle dynamics", `d_spatial_derivation.md` Lemma "Shannon chain rule"). Therefore the observer's model must treat **all directed edges as equivalent** — its crystallographic automorphism group must act transitively on (vertex, directed-edge) pairs. The observer's model is **strongly isotropic (arc-transitive)**. By substrate-agnosticism (`theorem_substrate_agnosticism.md` — the substrate *is* the observer's description-length-minimal canonical model), the substrate is strongly isotropic. So strong isotropy is **not an adopted lattice property** — it is (A)'s no-privilege applied to the walker's directed-edge state, exactly on par with the no-privilege that gives the uniform measure and the missing commutation.

**4c — uniqueness.** Sunada 2012 (*Notices AMS* **59**(2), 208–215): the **unique** 3-regular 3-connected ℝ³ crystal net that is strongly isotropic (Aut transitive on (vertex, directed-edge) pairs) is **srs** (the Laves / K₄ / (10,3)-a net), up to handedness. Combined with k* = 3, d = 3 (`k_star_derivation.md`, `d_spatial_derivation.md`): the substrate is srs. This is the load-bearing front-end of the "srs is the MDL minimum" argument in `g_girth_derivation.md` Step 2 — that doc's case analysis assumes the "strongly isotropic" *category*; §4b here is *why* that category (and why the 8 V+E-transitive-but-not-arc-transitive RCSR candidates — srs-z, srs-c4, srs-c8, … — are excluded: their models would carry ≥2 arc-orbits, i.e. "which-arc-type" structure the directionless observer cannot justify, costing extra description, and unlike srs they cannot be specified by symmetry alone). The Row-4 audit-v2 closure (`row4_audit_v2_revision_session2_2026-05-05.md`) that invokes arc-transitivity as load-bearing is therefore *correct* — it inherits §4b's chain, now made explicit. (History: an interim 2026-05-12 edit mislabeled §4b "motivational, not load-bearing" — that downgrade is retracted; §4b is load-bearing and the chain it sits in is the structural closure of R-9.)

### Step 5 — Causal state = directed edge

Shalizi-Crutchfield 2001 Theorem 2: the causal state (minimal sufficient statistic for prediction) is unique. For the srs NB walk:
- Vertex alone is insufficient: two different incoming edges at the same vertex give different forbidden-next-edge sets.
- Directed edge is sufficient (order-1 Markov) and minimal.
Directed-edge entropy: H(next | directed edge) = log2(k-1) = 1.0 bit.
Vertex entropy: H(next | vertex) = log2(k) = 1.585 bits > 1.0.
Closes W2. The 2|E|=12 directed edges per primitive cell form the state space.

### Step 6 — 1-step operator = Hashimoto B

Hashimoto 1989 (Terras 2011 §2.2): B[e',e] = 1 if e->e' is a valid 1-step NB transition, else 0. This is the transition/amplitude operator on the directed-edge state space. Under the framework's Open System reading, B is the visible-sector 1-step unitary amplitude operator.

### Step 7 — L-step amplitude = B^L

By matrix multiplication, (B^L)[e_L,e_0] counts NB walks from e_0 to e_L of length L. Under the Markov reading, L-step probability = (1/(k-1))^L * (B^L)[e_L,e_0]. Under the Unitary/Open System reading, B^L is the L-step amplitude operator. Closes W3.

### Step 8 — P-point spectrum

Sunada 2012 Bloch decomposition: B = integral(B(k) dk) over BZ. At the P-point k_P=(1/4,1/4,1/4), the doubly-degenerate h theorem (`../predictions/B_P_doubly_degenerate_h_derivation.md`) establishes: B(k_P) has eigenvalue h=(sqrt(3)+i*sqrt(5))/2 with multiplicity 2, C3-protected. This is the Ramanujan-saturating eigenvalue of srs (|h|^2 = k-1 = 2).

### W4: ruliad interpretation

Under "observer = MDL compression process," there is no physics outside the walker's compressed dynamics. Physical observables are by construction statistics of the NB walk process = spectral quantities of B. This closes W4 as a corollary of what the word "observer" means, not as a new axiom.

## Consequences for downstream predictions

The 19+ Layer-1+ prediction files that previously required a W1-W4 identification can now cite this theorem. Their numerical predictions (Koide Q=2/3, V_cb~(2/3)^8, PMNS mixing angles, Majorana phases, etc.) are empirical content derived from spec(B) under the Open System reading.

## Per-step rigor status

| Step | Claim | Rigor |
|------|-------|-------|
| 1 | F_inv(E) from A1 | STRICT-SOLID |
| 2 | MDL -> reduced words | STRICT-SOLID (A2 + Grunwald 2007) |
| 3 | Reduced = NB | STRICT-SOLID (Serre, Terras) |
| 4 | Jaynes-uniform (k−1)/k; isotropy → arc-transitive substrate → (Sunada) srs | STRICT-SOLID (4a: Jaynes 1957 + (A); 4b: (A) no-privilege + Step-5 directed-edge state + substrate-agnosticism; 4c: Sunada 2012). Front-end of g_girth Step 2; closes R-9. |
| 5 | Causal state = edge | STRICT-SOLID (Shalizi-Crutchfield 2001 Thm 2) |
| 6 | B = 1-step op | STRICT-SOLID (Hashimoto 1989 definition) |
| 7 | B^L = L-step | STRICT-SOLID (matrix multiplication) |
| 8 | P-point h, mult 2 | STRICT-SOLID (cites theorem_BP_doubly_degenerate_h) |
| W4 | obs in spec(B) | COROLLARY (operational reading) |

## Open items resolved by this theorem

From the W1–W4 walker-identification scoping doc:
- **B1.1** (closed observable enumeration): resolved via Shalizi-Crutchfield minimality.
- **B1.2** (edge-state MDL competitive): dissolved by causal-state reframing.
- **B1.3** (labeling under edge-transitivity): resolved — causal equivalence distinguishes vertices by Markov kernel position, not automorphism orbit.
- **A2-strategy flaw** ("trajectory is deterministic function of reduced word"): bypassed by working at data-representation level, not trajectory level.

## Time evolution (P1.S3, 2026-05-25)

W1–W4 establishes that the observer's data is a non-backtracking walk on srs with one-step amplitude operator B (Hashimoto). W4 gives observables as spectral data of B. This section records the first *live time-evolution* probe on the substrate-spinor 96-dim Hilbert space, integrating the master equation forward from arbitrary initial pure states.

**Probe:** `proofs/foundations/lindblad_spinor_time_evolution_2026-05-25.py` (matrix-free Lindblad superoperator via `scipy.integrate.solve_ivp` with DOP853, no full 9216×9216 storage).

**Construction it dynamizes:** `proofs/foundations/lindblad_spinor_coupled_construction.py` — 96-dim H_visible⊗S Hilbert space, 27 jump operators (3 family-I C₃-isotypic + 24 family-II directed-edge × B-L species), Hamiltonian H_full from Bloch P-point Hashimoto B_P. The construction file builds the algebra and verifies unitality; this probe runs the actual dynamics.

**Falsification gates (all PASS):**
- (i) trace preservation throughout: max |Tr(ρ) − 1| ≈ 10⁻¹⁴ across 3 initial conditions over t ∈ [0, 200].
- (ii) positivity preservation: min eigenvalue stays > 10⁻⁷; one IC even gains positivity over time (physical for pure → mixed evolution).
- (iii) each IC reaches a fixed point of L_super: ‖L_super(ρ_final)‖_F ≈ 10⁻¹⁰ at t=200 (dρ/dt at t_final ≈ 0 to numerical precision).
- (iv) slowest non-zero relaxation rate ≈ 0.108 (exponential fit to convergence trajectory), within an order of magnitude of the framework's W4 substrate cancellation rate γ = 1/k* = 1/3.

**Physical finding — non-ergodicity (kernel dim > 1).** The three initial pure states converge to *three distinct* fixed points (pairwise Frobenius distance 0.19–0.29). The maximally-mixed I/96 is *a* fixed point but not the unique one. Mechanism: family I preserves the C₃-isotypic block structure on the visible side; family II (P_e ⊗ Π_s) preserves the B-L species block on the spinor side. Combined, the constructed dissipator preserves (isotypic × species) and admits up to 3×2 = 6 independent steady states.

**Note on construction design intent.** The construction file's step D predicted "kernel dim 1 (unique steady state desired)" from adding family II to a pure family-I Lindblad (which had kernel dim 12 per `predictions/lindblad_isotypic_at_P.py`). This probe shows that family II *partially* lifts the family-I degeneracy but does not produce ergodicity. Closing the kernel to dim 1 (if desired) would require additional jump operators that break the conserved (isotypic × species) tensor structure — e.g., jumps that mix species across the visible C₃ basis, or that mix isotypics independently of edge.

**What this enables.** First live open-system dynamics in the repo, available as a sandbox for testing initial conditions, jump-operator additions, and the multi-axial axis-specific dynamics targeted in dynamics-phase-2 audits.

### A scoping note: B^L is NOT a discrete-time propagator (P1.S4, 2026-05-25)

W3 names B as the "1-step amplitude operator," and a natural conjecture is that B^L is then the substrate's *discrete-time propagator* with L = t/t_P. This conjecture was tested in `proofs/foundations/walker_B_as_discrete_time_propagator_2026-05-25.py` and **rejected as stated**, with a substantive partial finding.

Probe setup: 12³ super-cell with periodic BC, walker localized on a single directed edge, propagated via sparse B^L for L = 0..14, measured |ψ|² spreading in 3D Cartesian space.

- **Discrete causal cone**: PASS. Maximum Cartesian distance with non-trivial amplitude is bounded by L × (1 bond) = L × √2/4 for every L tested. B^L respects a discrete lightcone with v_max = 1 bond per L step.
- **Ballistic spreading**: FAIL. Log-MSD vs log-L slope = 1.20 (sub-ballistic). The Perron eigenvalue 2 dominates the bulk Ramanujan modes |h| = √2 after a few L, producing classical-NB-random-walk-like spreading rather than coherent wavefront propagation.
- **Group velocity match to v_F**: FAIL. sqrt(MSD)/L ≈ 0.13 at L = 14, whereas v_F^P = √3/6 ≈ 0.289. The framework's Bloch group velocities are properties of the vertex-level scalar Bloch H, a *different* operator than B (which lives at the edge level).

**Correct reading:** L *is* a meaningful discrete-time index with a causal cone, but B^L is the amplitude counter for non-backtracking walks (which is what W1–W4 actually proved) — not a unitary discrete-time evolution. The "B^L is a propagator" reading is narrower than W3 admits. See an internal working note for the full finding and possible refinements (Perron-projected walker, stochastic NB walk T = B/(k−1), Szegedy/Grover unitarization).

### Observer-side time as a martingale index (P1.O3, 2026-05-25)

Companion to the substrate-side L = discrete-time-with-causal-cone finding above: on the *observer* side, the natural discrete time is the **observation count N** (which generates the observer's filtration F_N). The Bayesian posterior over any F_∞-measurable event is a martingale w.r.t. {F_N} (Doob/Lévy), with Doob convergence to consistency under the framework's A2-T true measure.

Sketch theorem: `docs/theorems/theorem_observer_martingale_time_2026-05-25.md`. Probe: `proofs/foundations/observer_martingale_walk_2026-05-25.py` — all three gates PASS (martingale identity to machine precision, Doob convergence to 0.998 mean by N=200, log-odds slope matches theoretical KL at 0.5%).

**Two-clocks reading.** The framework has TWO discrete-time indices, with different operational content:

| | Substrate side | Observer side |
|---|---|---|
| Index | L (step count of B^L) | N (observation count) |
| Counts | causal-cone advance (bonds per step) | σ-algebra refinement events |
| Built-in object | B^L = amplitude counter for NB walks | F_N = observer filtration |
| Has lightcone? | Yes (P1.S4, v_max = 1 bond/L) | N/A — N is information-theoretic |
| Has martingale? | N/A (B^L is not unitary) | Yes (P1.O3, posterior is martingale on F_N) |
| Identification with t_P | t_substrate = L · t_P (one tick per substrate primitive event) | t_observer = N · t_P (one tick per observation) |

L and N are both unit-multiplied by t_P but **are not equal** — not every substrate primitive event produces an observation (A2-T's MDL canonicalization erases backtracks, `walker_dynamics_derivation.md` §4a parenthetical). The relationship between the two clocks is itself a research-level question (an instance of the L6 BRIDGE wall, but for the discrete-time index rather than for recombination kinematics).

## References

- Bass, H. (1992). The Ihara-Selberg zeta function. *Int. J. Math.* 3, 717-797.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Theorem 5.4.3.
- Crutchfield, J. P. & Young, K. (1989). Inferring statistical complexity. *Phys. Rev. Lett.* 63, 105-108.
- Grunwald, P. (2007). *The Minimum Description Length Principle*. MIT Press. §5.1-5.3.
- Hashimoto, K. (1989). Zeta functions of finite graphs. *Adv. Stud. Pure Math.* 15, 211-280.
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Phys. Rev.* 106, 620-630.
- Levin, D. A., Peres, Y. & Wilmer, E. L. (2009). *Markov Chains and Mixing Times*. AMS. Theorem 1.14.
- Serre, J.-P. (1980). *Trees*. Springer. §I.1 (free involutive monoids; reduced words).
- Shalizi, C. R. & Crutchfield, J. P. (2001). Computational mechanics. *J. Stat. Phys.* 104, 817-879. Theorem 2.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell Syst. Tech. J.* 27, 379-423.
- Sunada, T. (2012). *Topological Crystallography*. Springer. §§5-6 (Bloch decomposition).
- Terras, A. (2011). *Zeta Functions of Graphs*. Cambridge. §2.1 (NB walks), §2.2 (Hashimoto).

## Files referenced

- `predictions/d_spatial.py` — srs as the MDL-optimal graph; C^3 per vertex.
- `predictions/k_star.py` — k*=3 (chain-imported).
- `predictions/h_walker_eigenvalue.py` — h=(sqrt(3)+i*sqrt(5))/2 (chain-imported).
- `../predictions/B_P_doubly_degenerate_h_derivation.md` — P-point spectrum (h, mult 2; cites as closed).
- `proofs/foundations/theorem_walker_dynamics.py` — detailed numerical verification (6 checks).
- the W1–W4 walker-identification scoping doc — the gap this theorem closes.
