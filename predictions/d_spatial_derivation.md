# Derivation of d (spatial dimension)

**NOTE (post-2026-04-26 demotion):** A2 and A3 are derived theorems; structural slate is {A1} + P1' + A5-mass per docs/framework/framework_axioms.md §10. The closure chain referenced here is preserved; only the axiomatic-status labels change. This derivation invokes Gleason 1957, which presupposes G.1 (Hilbert-space structure on the observer's model class). Under A1 + A2-T + A3-T, G.1 and G.5 are DERIVED via the Chiribella-D'Ariano-Perinotti 2011 Theorem 25 chain; see predictions/observer_hilbert_space.py.

## Abstract

We derive that the number of spatial dimensions is $d = 3$ from MDL compression and Gleason's theorem. The argument: an MDL-optimal observer assigns probabilities non-contextually (Lemma 1). The effective dimension of the observer's statistical model equals the spatial dimension $d$ of the crystal net (proven via exponential family Fisher information — Jaynes 1957, Brown 1986, Čencov 1982). Non-contextual probability assignments on a $d$-dimensional measurement space are unique only for $d \geq 3$ (Gleason 1957). At $d = 2$, the observer faces unbounded model-selection ambiguity. Among $d \geq 3$, MDL selects $d = 3$ uniquely: the graph description length is bounded below by $\log_2 |SG(d)|$ (Shannon 1948 + crystallographic enumeration OEIS A006227), the data description length equals $n \log_2 d$ on any $d$-regular graph under isotropic toggle dynamics (Shannon chain rule), and both are strictly monotonically increasing in $d$. The derivation uses two framework axioms and six established mathematical theorems.

## Framework axioms invoked

1. **MDL compression.** The observer selects models by minimizing total description length $F = \text{DL}(\text{model}) + \text{DL}(\text{data}|\text{model})$. This is the framework's second axiom.

2. **Toggle events are measurements.** At each node of the graph, the observer measures "which edge was toggled." The $k$ edges at a node define $k$ mutually exclusive measurement outcomes. Each event "edge $i$ toggled" carries the edge displacement vector $v_i$ as its sufficient statistic — this is what the event IS (a transition in direction $v_i$).

## Derivation

### Step 1: MDL forces non-contextuality

**Lemma 1** (dimension_three_theorem.md, proven): An MDL-optimal observer uses non-contextual probability assignments.

*Proof summary.* A **contextual** model assigns probabilities $P_B(e \mid B)$ that depend on which basis $B$ is measured. For $k$ measurement outcomes, this requires $k^2 + k - 1$ parameters (a probability distribution per basis, after normalization). A **non-contextual** model assigns $P(e)$ independent of basis, requiring $k^2 - 1$ parameters (one density matrix). The difference is $k$ parameters. Since $k \geq 2$, MDL always selects the shorter (non-contextual) model. ∎

This is parameter counting — explicit arithmetic, no assumptions beyond MDL.

### Step 2: Effective Hilbert space dimension = spatial dimension

We prove $n_{\text{eff}} = d$ in three sub-steps, each citing an established theorem.

#### Step 2a: MDL selects exponential family models

At a node with $k$ edges, the observer models toggle events. Edge $i$ has displacement vector $v_i \in \mathbb{R}^d$. The event "edge $i$ toggled" carries sufficient statistic $T(\text{edge } i) = v_i$.

**Theorem** (Jaynes, *Phys. Rev.* **106**, 620–630, 1957): The maximum-entropy distribution consistent with given expected sufficient statistics minimizes description length among all distributions with the same expectations.

The maximum-entropy distribution over $k$ discrete outcomes with vector-valued sufficient statistics is the exponential family:

$$P(\text{edge } i \mid \eta) = \frac{\exp(\eta \cdot v_i)}{Z(\eta)}, \qquad Z(\eta) = \sum_{j=1}^{k} \exp(\eta \cdot v_j) \tag{2.1}$$

where $\eta \in \mathbb{R}^d$ is the natural parameter. MDL selects this model.

#### Step 2b: Fisher information matrix has rank $d$

The Fisher information matrix of the exponential family (2.1) is the Hessian of the log-partition function $A(\eta) = \log Z(\eta)$:

$$F_{ab}(\eta) = \frac{\partial^2 A}{\partial \eta_a \, \partial \eta_b} = \text{Cov}_\eta[v_a, \, v_b] \tag{2.2}$$

This is the covariance matrix of the edge vectors under the softmax weights $P(i|\eta)$.

**Theorem** (Brown, *Fundamentals of Statistical Exponential Families*, IMS Lecture Notes Vol. 9, 1986, Theorem 1.13): For a regular exponential family with sufficient statistic $T(x)$ taking values in a set $\mathcal{T} \subset \mathbb{R}^p$, if the affine hull of $\mathcal{T}$ has dimension $p$, then the Fisher information matrix $F(\eta)$ is positive definite (rank $p$) at every interior point of the natural parameter space.

**Application:** The sufficient statistics are $\{v_1, \ldots, v_k\} \subset \mathbb{R}^d$. For a $d$-dimensional crystal net, the edge vectors must span $\mathbb{R}^d$ — otherwise the net would have periodicity in fewer than $d$ directions. This is a theorem of reticular chemistry (Delgado-Friedrichs & O'Keeffe, *Acta Cryst.* A **59**, 351–360, 2003, §2.1: the edge vectors of a $d$-periodic net generate the translation lattice $\mathbb{Z}^d$).

Since $\{v_1, \ldots, v_k\}$ spans $\mathbb{R}^d$, their affine hull has dimension $d$. By Brown's theorem:

$$\boxed{\text{rank}(F) = d} \tag{2.3}$$

#### Step 2c: Fisher rank determines effective Hilbert space dimension

**Theorem** (Čencov, *Statistical Decision Rules and Optimal Inference*, AMS Translations, 1982, Theorem 11.1): The Fisher metric is the unique (up to scale) Riemannian metric on a statistical manifold that is invariant under sufficient statistics.

The manifold of exponential family distributions (2.1) with rank-$d$ Fisher metric is a $d$-dimensional statistical manifold. By Čencov's uniqueness, this is the natural geometry for the observer's model. The standard $L^2$ embedding of probability distributions $p \mapsto \sqrt{p}$ (Bhattacharyya 1943, *Bull. Calcutta Math. Soc.* **35**, 99–109) gives a rank-$d$ REAL Hilbert space $\mathbb{R}^d$; promotion to the complex Hilbert space $\mathbb{C}^d$ used by Gleason's theorem (§Step 3 below) follows from A3-T (the framework's purification theorem; see `docs/theorems/theorem_A3_complex_hilbert_from_multiway.md` and an internal audit of the seven Gleason sub-assumptions for the audit that identified this step as G.5-dependent under the prior two-axiom formulation).

Therefore:

$$n_{\text{eff}} = \text{rank}(F) = d \tag{2.4}$$

#### Summary of Step 2

$$\text{Toggle events} \xrightarrow{\text{MDL + Jaynes 1957}} \text{Exponential family (2.1)} \xrightarrow{\text{Brown 1986, Thm 1.13}} \text{rank}(F) = d \xrightarrow{\text{Čencov 1982, Thm 11.1}} n_{\text{eff}} = d$$

Every link is a cited theorem applied to the toggle dynamics. No new mathematics.

**Consequence for $k > d$:** The $k$ edge vectors span only $\mathbb{R}^d$ (rank $d < k$). The Fisher information matrix has rank $d$, not $k$. The extra $k - d$ edges contribute zero Fisher information — they are informationally redundant. MDL eliminates redundant parameters.

**Consequence for $k < d$:** Impossible for a $d$-dimensional crystal net: $k$ vectors cannot span $\mathbb{R}^d$ if $k < d$.

Therefore: $k \geq d$ and $k_{\text{eff}} = d$.

### Step 3: Gleason's theorem requires $d \geq 3$

**Theorem** (Gleason, *J. Math. Mech.* **6**, 885–893, 1957): Let $H$ be a Hilbert space of dimension $n \geq 3$. Every frame function $f: S(H) \to [0,1]$ satisfying $\sum_{i=1}^n f(e_i) = 1$ for every orthonormal basis $\{e_1, \ldots, e_n\}$ has the form:

$$f(e) = \text{Tr}(\rho \, |e\rangle\langle e|)$$

for a unique density operator $\rho$.

**Failure at $n = 2$** (Lemma 2, dimension_three_theorem.md, proven): In dimension 2, the frame function constraint $f(e) + f(e^\perp) = 1$ admits infinitely many solutions beyond the Born rule. The space of valid frame functions on $\mathbb{CP}^1$ is infinite-dimensional. To specify which frame function the observer uses requires unbounded description length (the metric entropy of the function space diverges).

**MDL consequence:** At $d = 2$ ($n_{\text{eff}} = 2$), the observer's total description length includes an unbounded model-selection penalty for choosing among non-unique frame functions. At $d \geq 3$ ($n_{\text{eff}} \geq 3$), Gleason pins down the probability rule uniquely — zero selection cost.

Therefore: $d \geq 3$.

### Step 4: MDL selects $d = 3$ uniquely among $d \geq 3$

We prove that the total description length $F(d) = \text{DL}_{\text{graph}}(d) + \text{DL}_{\text{model}}(d) + \text{DL}_{\text{data}}(d)$ is strictly monotonically increasing in $d$ for $d \geq 3$, so MDL selects $d = 3$.

#### Step 4a: Graph description length is strictly monotone

Any uniquely decodable code for the set of $d$-dimensional crystallographic space groups has expected length at least $\log_2 |SG(d)|$ bits (Shannon, *Bell Syst. Tech. J.* **27**, 379–423, 1948; source coding theorem applied to the uniform prior over $|SG(d)|$ symbols). Therefore:

$$\text{DL}_{\text{graph}}(d) \geq \log_2 |SG(d)| \tag{4.1}$$

The crystallographic space group enumeration (OEIS A006227):

| $d$ | $|SG(d)|$ | $\log_2|SG(d)|$ | Reference |
|-----|-----------|-----------------|-----------|
| 3 | 230 | 7.845 | Fedorov 1891; Schönflies 1891 |
| 4 | 4,894 | 12.257 | Brown, Bülow, Neubüser, Wondratschek, Zassenhaus 1978 (corrected by Neubüser, Souvignier, Wondratschek 2002) |
| 5 | 222,097 | 17.761 | Plesken & Schulz 2000 |

This is strictly monotonically increasing in $d$.

#### Step 4b: Data description length is $n \log_2 d$ on any $d$-regular graph

**Lemma (Shannon chain rule).** On any $d$-regular graph under isotropic toggle dynamics, the minimum encoding length of an $n$-event toggle stream is $n \log_2 d$ bits, independent of the graph's automorphism structure.

*Proof.* The framework's first axiom (binary self-inverse toggle, see `predictions/p_toggle.py`) specifies no preferred edge direction at a vertex. Therefore the per-step distribution over the $d$ edges at any vertex is uniform: $p_i = 1/d$ for $i = 1, \ldots, d$. Per-event entropy is $\log_2 d$.

Consider an observer who partitions the $d$ edges at each vertex into $m$ orbits of sizes $k_1, \ldots, k_m$ (with $\sum_i k_i = d$) and models the stream using orbit identities as intermediate labels. The encoding cost per event decomposes as:

$$\underbrace{H(k_1/d, \ldots, k_m/d)}_{\text{orbit identity}} + \underbrace{\sum_{i=1}^{m} \frac{k_i}{d} \log_2 k_i}_{\text{edge within orbit}}$$

By the Shannon chain rule $H(X, Y) = H(Y) + H(X|Y)$ (Shannon 1948; Cover & Thomas, *Elements of Information Theory*, 2nd ed., 2006, Thm 2.5.1):

$$H\!\left(\tfrac{k_1}{d}, \ldots, \tfrac{k_m}{d}\right) + \sum_{i=1}^{m} \tfrac{k_i}{d} \log_2 k_i \; = \; -\sum_{i=1}^{m} \tfrac{k_i}{d} \log_2 \tfrac{k_i}{d} + \sum_{i=1}^{m} \tfrac{k_i}{d} \log_2 k_i \; = \; \sum_{i=1}^{m} \tfrac{k_i}{d} \log_2 d \; = \; \log_2 d$$

The total entropy per event is always $\log_2 d$, for every partition of edges. Any lossless encoding of the stream satisfies $\text{DL}_{\text{data}}(d) \geq n \log_2 d$ (Shannon source coding, 1948). ∎

This closes the gap that the external `dimension_three_theorem.md` left open: there is no "data-fit benefit" to higher $d$. The "savings" from coarse-graining edge types are exactly canceled by the cost of resolving specific edges within each type, giving the universal bound $n \log_2 d$.

#### Step 4c: Model description length is non-negative

$$\text{DL}_{\text{model}}(d) \geq 0 \tag{4.3}$$

by definition of description length.

#### Step 4d: Strict monotonicity of $F(d)$

Combining (4.1), (4.2), (4.3):

$$F(d) \geq \log_2 |SG(d)| + n \log_2 d \tag{4.4}$$

At $d = 3$, the srs graph achieves $\text{DL}_{\text{graph}}(\text{srs}) = 12.17$ bits (see `predictions/g_girth_derivation.md` and `proofs/foundations/dl_comparison.py`). Combined with $\text{DL}_{\text{model}} = 0$ at the max-entropy density matrix $\rho = I/3$ (forced by graph symmetry: edge-transitivity of srs means the symmetry-respecting $\rho$ is uniquely determined up to a global trace, leaving zero free parameters), and $\text{DL}_{\text{data}} = n \log_2 3 = 1.585\, n$:

$$F_{\text{srs}}(3) = 12.17 + 0 + 1.585\, n$$

For any $d \geq 4$:

$$F(d) - F_{\text{srs}}(3) \geq \bigl(\log_2 |SG(d)| - 12.17\bigr) + n \bigl(\log_2 d - \log_2 3\bigr)$$

Evaluating at $d = 4$:

$$F(4) - F_{\text{srs}}(3) \geq (12.257 - 12.17) + n(2.000 - 1.585) = 0.087 + 0.415\, n > 0 \quad \text{for all } n \geq 0$$

Evaluating at $d = 5$:

$$F(5) - F_{\text{srs}}(3) \geq (17.761 - 12.17) + n(2.322 - 1.585) = 5.591 + 0.737\, n > 0 \quad \text{for all } n \geq 0$$

For $d \geq 6$: $|SG(d)|$ grows monotonically in $d$ (elementary: the map $G \mapsto G \times \mathbb{Z}$ injects $d$-dim space groups into $(d{+}1)$-dim, so $|SG(d{+}1)| \geq |SG(d)|$), and $n \log_2 d$ is strictly monotone. Therefore $F(d) > F_{\text{srs}}(3)$ for all $d \geq 4$.

$$\boxed{d^* = 3}$$

### Consistency check: surprise balance

At $d = k = 3$ and $p = 2$ (binary toggle, from `predictions/p_toggle.py`):

**Surprise per toggle event:**
$$S(k, p) = 1 + \log_2(k) + \log_2(p-1) = 1 + \log_2(3) + 0 \approx 2.585 \text{ bits}$$

**Per-edge maintenance cost** (Bayesian thresholds from Dirichlet(1,1) prior):
$$\theta_{\text{create}} = \log_2(p) = 1 \text{ bit}$$
$$\theta_{\text{persist}} = \log_2\!\left(\frac{p+1}{p-1}\right) = \log_2(3) \approx 1.585 \text{ bits}$$
$$\theta_{\text{create}} + \theta_{\text{persist}} = 1 + \log_2(3) \approx 2.585 \text{ bits}$$

These are equal: $S(3, 2) = \theta_{\text{create}} + \theta_{\text{persist}}$. This is not assumed — it is a consequence of $k = 3$ and $p = 2$. The surprise balance equation and $k = 3$ are equivalent statements for binary toggles.

## Result

$$\boxed{d = 3}$$

Three spatial dimensions. Exact, from MDL + Gleason's theorem + Fisher information. Zero free parameters.

## Comparison with experiment

| Quantity | Predicted | Observed | Deviation |
|----------|-----------|----------|-----------|
| Spatial dimensions | 3 | 3 | 0 (exact) |

## Open questions

1. **The +1: time dimension and Lorentzian signature.** This derivation gives $d = 3$ spatial dimensions. Two related but distinct sub-questions about the (3+1) Lorentzian structure used in downstream framework predictions:

   (a) **The time *axis* itself.** *Closed (2026-04-27, via R-4 residue closure):* the time axis is intrinsic to A1's stream-length grading. A1's toggle stream is ordered by composition length L = 0, 1, 2, …; the multiway DAG is L-graded with a strict partial order; "arrow of time = corollary" per `docs/theorems/theorem_observer_energy_functional.md` (Stage 2c). The +1 time dimension is therefore not a derived structural element from the d_spatial chain (which is about *spatial* dimensions only) but a built-in feature of A1 itself. See `docs/audits/registers/structural_residue_register.md` R-4 + an internal working note for the closure outcome that reframed this question.

   (b) **The metric *signature* (−, +, +, +).** *Structurally CLOSED at theorem grade (2026-04-27, parallel session)* via Route B (substrate Dirac cones + Iorio-elastic vielbein + linearised Einstein). The closure: Hashimoto Bloch spectrum on srs has Dirac cones at Γ (v_F = 1/2 spin-1), H (PH-conjugate), P (v_F = √3/6 2-band); Wigner-Eckart-forced SO(3) on the T-irrep at each cone produces local (1+3) Minkowski; β = 1 Iorio-elastic vielbein + spin connection $(1/4)\,\Omega \cdot (k \times S)$ + linearised Einstein $-\Box u^{ab} = 8\pi G_{\rm sub} T^{ab}$. Numerical $G_{\rm sub}$ (the dimensionless prefactor) remains research-level pending (Sakharov-BZ or Lichnerowicz, 2–3 sessions). Verified: `proofs/foundations/lorentz_sig_spin1_dirac_decomposition.py`, `proofs/foundations/lorentz_sig_iorio_session{2,3,4}_*.py`. See `memory/session_handoff_2026-04-27_lorentz_arc_plus_qft_cascade.md` and an internal working note (closure update section).

   Routes A and C remain catalogued in `lorentzian_signature_scoping.md` as historical alternatives. Route C (Connes spectral action) was attempted 2026-04-26 and BLOCKED at the bounded-D² obstruction (`proofs/foundations/lorentzian_signature_spectral_action_attempt.py`); Route B's success does not depend on resolving Route C.


## Audit v2 (Clause 7) status

This prediction inherits Row 3 (d = 3) audit v2 closure. See
an internal working note §3 (foundational rows)
and an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 3 audit v2 (DOMINANT via Gleason d≥3 + R-4/R-5 empirical anchors + M5 no new amplification).
- **Named conditional:** M1 R-N hard-gates at d-axis (Gleason d=2, R-4 d=4, R-5 d≥5). M5 non-local amplification check returns no new amplification for d=4 alternatives (η_5 = 0 empirical anchor stands).

## References

- Brown, H., Bülow, R., Neubüser, J., Wondratschek, H. & Zassenhaus, H. (1978). *Crystallographic Groups of Four-Dimensional Space*. Wiley Monographs in Crystallography, Vol. 4.
- Brown, L.D. (1986). *Fundamentals of Statistical Exponential Families*. IMS Lecture Notes — Monograph Series, Vol. 9. Theorem 1.13.
- Čencov, N.N. (1982). *Statistical Decision Rules and Optimal Inference*. AMS Translations of Mathematical Monographs, Vol. 53. Theorem 11.1.
- Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*, 2nd ed. Wiley-Interscience. Theorem 2.5.1 (Chain rule for entropy).
- Delgado-Friedrichs, O. & O'Keeffe, M. (2003). Identification of and symmetry computation for crystal nets. *Acta Cryst.* A **59**, 351–360. §2.1.
- Fedorov, E.S. (1891). Симметрія правильныхъ системъ фигуръ. *Zap. Mineral. Obshch.* **28**, 1–146. [Enumeration of 230 three-dimensional space groups.]
- Gleason, A.M. (1957). Measures on the closed subspaces of a Hilbert space. *J. Math. Mech.* **6**, 885–893.
- Jaynes, E.T. (1957). Information theory and statistical mechanics. *Phys. Rev.* **106**, 620–630.
- Neubüser, J., Souvignier, B. & Wondratschek, H. (2002). Corrections to Crystallographic Groups of Four-Dimensional Space. *Acta Cryst.* A **58**, 301.
- OEIS Foundation Inc. (2024). Sequence A006227: Number of $n$-dimensional space groups (including enantiomorphs). The On-Line Encyclopedia of Integer Sequences, <https://oeis.org/A006227>.
- Plesken, W. & Schulz, T. (2000). Counting crystallographic groups in low dimensions. *Experimental Math.* **9**, 407–411. [Enumeration of 222,097 five-dimensional space groups.]
- Schönflies, A. (1891). *Krystallsysteme und Krystallstructur*. Leipzig: Teubner.
- Shannon, C.E. (1948). A mathematical theory of communication. *Bell Syst. Tech. J.* **27**, 379–423, 623–656.
