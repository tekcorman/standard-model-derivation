# Theorem — Layer 1 → Layer 2 substrate bridge (Path Dominant)

**Date:** 2026-04-30 (initial closure); 2026-05-01 (Stage 1b' R-13 closure RETRACTED as circular; scope restriction applied per Option A).
**Status:** **STRUCTURAL-DERIVATION-GRADE-CONDITIONAL within the Bloch-decomposable-substrate class.** Closure of S1'+S2'+S3'+S4'+S5'(crystal+finite+quasiperiodic-rejected) via inheritance + tightening. S5' hyperbolic sub-axis (R-13) is **OUT OF SCOPE** of this bridge (per Option A scope restriction 2026-05-01). The Stage 1b' attempt to close R-13 via Bloch-decomposition hard-gate was found CIRCULAR (the framework's Bloch P-point eigenvalue is derived from srs identification, not an independent constraint); RETRACTED. See §0a below for the scope statement.
**Companion to:** `../audits/registers/uniqueness_ledger.md` (Rows 3-10), `../audits/registers/structural_residue_register.md` (R-4, R-5, R-7, R-9, R-10, R-12, R-13 — the latter now reframed as scope-disclosure residue), an internal working note (Row 3 §3, Row 4 §2, Row 6 §3).
**Supersedes:** the strict-UNIQUE Path Full statement of an internal working note §1. The scoping doc's S4-before-S5 ordering was wrong; corrected here.
**Companion to:** `../audits/registers/uniqueness_ledger.md` (Rows 3-10), `../audits/registers/structural_residue_register.md` (R-4, R-5, R-7, R-9, R-10, R-11, R-12, R-13), an internal working note (Row 3 §3, Row 4 §2, Row 6 §3).

---

## 0a. Scope restriction (Option A, 2026-05-01)

**The Path Dominant theorem applies WITHIN the Bloch-decomposable substrate class.**

Specifically: candidate Layer-2 realizations C of F_inv(E) are restricted to those admitting **Bloch decomposition** in the sense of Sunada 2013 *Topological Crystallography* §6 Theorem 6.4 — i.e., C carries a rank-d abelian translation subgroup ℤ^d acting freely with finite quotient (primitive cell), so the Hashimoto operator on C decomposes as $K = \int_{\rm BZ} B(\mathbf{k}) d\mathbf{k}$ over the Brillouin zone $\mathbb{T}^d$.

Substrates outside this class — most notably hyperbolic Cayley graphs of cocompact non-amenable Kleinian groups (R-13 candidates, which lack rank-3 abelian translation subgroups per Selberg + Margulis + Preissmann) — are **out of scope** of this bridge theorem. The bridge says nothing about whether such substrates could produce SM-matching predictions via different (non-Bloch) machinery (e.g., Plancherel-decomposed spectra). That is a separate open question (R-13 OPEN as scope-disclosure residue).

**Why this scope restriction is principled, not arbitrary.** The framework's prediction machinery (V_us, V_cb, Q_Koide, h_walker, m_H, β, η_B, all CKM/PMNS angles, dark corrections) is constructed using Bloch P-point eigenvalue + Bloch fibre decomposition + Sunada 2013 §6 Thm 6.4 explicitly. Within the Bloch-decomposable substrate class, the framework's prediction formulas have well-defined meaning. Outside this class, the same formulas don't apply — extending the framework to non-Bloch substrates would require constructing analogous prediction machinery (e.g., via Plancherel measures or non-amenable representation theory), which the framework currently does not develop. The bridge cannot legitimately claim to exclude substrates whose physics the framework doesn't even describe.

**Why this is NOT the original "crystal-net premise" gap.** The pre-bridge state had Row 4 *Conditional on* "the substrate is a d-periodic crystal net" with no justification for why crystal nets are the right candidate class. Under Option A, the analogous condition is "the substrate is Bloch-decomposable" with the JUSTIFICATION that this is precisely the class on which the framework's prediction machinery is well-defined — and the bridge claims DOMINANT minimum within this class (S2'+S3'+S5' crystal-subclass closure). Less ambitious than strict-UNIQUE, more honest than circular categorical exclusion of out-of-class alternatives.

**Stage 1b' RETRACTION.** The 2026-05-01 attempt (an internal working note §0) to close R-13 via Bloch-decomposition hard-gate as a categorical exclusion was found CIRCULAR by user inspection: using framework-internal machinery (Bloch P-point eigenvalue h = (√3+i√5)/2) as exclusion criterion for non-srs alternatives reduces to "alternatives that don't fit the machinery built for srs are excluded." The Selberg+Margulis+Preissmann+Sunada chain showing R-13 candidates lack Bloch decomposition REMAINS true; what doesn't follow is that this constitutes a substrate-independent exclusion. R-13 reverts to OPEN status under scope-disclosure framing (§10 item 1).

---

## 1. Theorem statement (Path Dominant — within Bloch-decomposable substrate class)

**Path Dominant (scope-restricted).** Under {A1, A5-mass} (per `../framework/framework_axioms.md` §10; P1' formerly cited here is now a derived theorem under A1 per `theorem_p1_prime_derived_from_a1.md` and no longer enters as an independent axiom), with the Layer-1 substrate the Cayley graph of F_inv(E) for unspecified |E| ≥ 1 AND the candidate Layer-2 realizations restricted to Bloch-decomposable substrates (per §0a), there exists an MDL functional **F** on (Layer-2 realization of F_inv(E)) such that:

(i) **F = DL_struct + DL_obs** is well-defined uniformly across all *Bloch-decomposable* candidate classes (crystallographic 3D, crystallographic d≠3, finite as Bloch-trivial limit, quasiperiodic-as-Cayley-quotient case-by-case) via a Kolmogorov-complexity floor for DL_struct + a tree-native Shannon-source-coding floor for DL_obs. **Outside the Bloch-decomposable class**, F is not constructed here; this is the scope restriction of §0a.

(ii) **F has a DOMINANT minimum** at the (srs, space group I4₁32, |E|=6, |V|=4 (Wyckoff 8a), k*=3, g=10, both chiralities, vertex+edge-transitive) configuration. The margin to non-srs candidates within each enumerated class is strictly positive and computable.

(iii) **The per-stream-length margin** to any candidate with effective coordination k_eff > 3 is at least n · log₂(k_eff/3) bits, where n is the observation stream length. This drives steep stream-length-amplified dropoff: any d > 3 or k_eff > 3 alternative is **MDL-permitted at finite n** but exponentially Boltzmann-suppressed as n grows.

**What this does NOT claim.** (1) This is *not* a strict global UNIQUE theorem within the Bloch-decomposable class. d > 3 Bloch-decomposable substrates (e.g., d=4 crystallographic Cayley quotients) are MDL-permitted at small observation windows; they are catalogued as soft-gated residues (R-4, R-5, R-9). The framework's predictions are robust against this multi-realization retention because (a) the per-stream-length term 0.415·n bits/event for d=4 vs d=3 is large at any phenomenologically relevant n, and (b) the framework's η_5 = 0 prediction (R-4 closure) provides an empirical anchor. (2) **This theorem makes no claim about substrates outside the Bloch-decomposable class** — see §0a scope restriction. R-13 (hyperbolic Kleinian Cayley graphs) is OPEN as a scope-disclosure residue, not refuted.

**Consequence for the structural ledger.** Rows 3, 4, 6, 7, 8 of `../audits/registers/uniqueness_ledger.md` retain their current per-row classifications, with the **"crystal-net premise" gap referenced in Row 4** now closed by inheritance from this theorem. The chain Cayley(F_inv(E)) → d-periodic crystal net is the F-functional's MDL-minimum-Layer-2-realization output (sharp-peak case within the Bloch-decomposable scope: F-functional uniqueness gives a single dominant Layer-2 realization at every n; per `feedback_a2_waterline.md`, waterline = strict-min agree in the unique-peak regime, so the "MDL minimum" framing here is genuine and not subject to the `canonical_encoding`/`channel_select` operator split of `theorem_lattice_coupling_general.md` §2), not a separately stipulated premise.

---

## 2. Decomposition into sub-theorems and their closure status

The scoping doc's Path Full S1-S5 decomposition is preserved structurally but reframed under the softer DOMINANT statement. **S4 is reordered as logically downstream of S5'** — the framework's existing R-11 closure (`../audits/registers/structural_residue_register.md` R-11) already derives |E|=6 from k*·|V|/2 GIVEN srs identification; no upstream MDL-over-n minimization machinery exists or is needed for the DOMINANT statement.

| Sub | Strict Path Full statement | Path Dominant restatement | Closure status |
|---|---|---|---|
| S1' | F well-defined and unique form | F well-defined uniformly via Kolmogorov-floor + class encodings | THEOREM-GRADE (this doc §3) |
| S2' | d_eff = 3 strict UNIQUE | d_eff = 3 DOMINANT with computable margin per class | STRUCTURAL-DERIVATION-GRADE-CONDITIONAL (this doc §4) |
| S3' | k_eff = 3 strict UNIQUE | k_eff = 3 DOMINANT with structural M2 margin (+1.14 bits) | STRUCTURAL-DERIVATION-GRADE-CONDITIONAL (this doc §5; inherits Row 4 audit-v2 structural piece) |
| S4' | \|E\| = 6 from MDL-over-n | \|E\| = 6 inherited from R-11 closure (downstream of S5') | UNIQUE-CONDITIONAL on Row 6 (R-11 already closed) |
| S5' | srs strict UNIQUE across all realizations | srs DOMINANT in crystal subclass (Sunada + dl_comparison); finite hard-gated (R-10); quasiperiodic excluded by Cayley-quotient definition; hyperbolic R-13 OUT-OF-SCOPE (Option A, 2026-05-01 — within Bloch-decomposable class only) | PARTIAL within Bloch-decomposable class; out-of-class candidates not addressed |

---

## 3. S1' — Uniform F-functional

### 3.1 Statement

**S1'.** There exists a functional **F** : {(Layer-2 realization C of F_inv(E)) : |E| ≥ 1} → ℝ≥0 such that:

(a) **F(C) = DL_struct(C) + DL_obs(C; n)** for any candidate class.

(b) **DL_struct(C) ≥ K(C)** where K(C) is the prefix-free Kolmogorov complexity of C (Li-Vitanyi 2008, *An Introduction to Kolmogorov Complexity and Its Applications*, 3rd ed., §3.1). For each class, a concrete tighter encoding is available:
  - **Crystallographic d-dim**: log₂|SG(d)| + Wyckoff overhead (`d_spatial_derivation.md` §4a; OEIS A006227; Plesken-Schulz 2000 for d=5).
  - **Finite N-vertex graphs**: O(N² log N) bits for adjacency list, or tighter via labelled-graph enumeration (`dl_comparison.py:248-271` for Petersen, K_{3,3}, random-N).
  - **Cayley quotients of F_inv_n at finite index**: log₂|F_inv_n / N| + relation-encoding cost (Rissanen 1983 universal prior + Magnus-Karrass-Solitar 1976 residual-finiteness enumeration).
  - **Quasiperiodic**: NOT a Cayley quotient of finite-index normal subgroup (Senechal 1995 §3; Baake-Grimm 2013 — quasiperiodic tilings are not group orbits). Excluded by Path Dominant's candidate-set definition.

(c) **DL_obs(C; n) ≥ n · log₂(k_eff(C))** where k_eff(C) is the effective branching of the causal-state DAG of C (Shalizi-Crutchfield 2001 Thm 2). This is **tree-native**: derives from A1 (binary self-inverse → uniform per-step on toggle alphabet) + Shalizi-Crutchfield causal-state quotient + Shannon source-coding 1948. Already proved as `d_spatial_derivation.md` Step 4b without crystal-net premise.

### 3.2 Hard quality gate

- (b)'s Kolmogorov floor: Type 3 (Li-Vitanyi 2008 Theorem 2.4.1).
- (b)'s class encodings: each Type 3 with cited theorem.
- (c)'s data-DL bound: Type 5 chain via `predictions/d_spatial_derivation.md` Step 4b → Shannon 1948 → Cover-Thomas 2006 Thm 2.5.1.

**Verdict:** S1' passes the gate as a Type 3 + 5 chain. THEOREM-GRADE.

### 3.3 What S1' does NOT do

S1' establishes the *form* of F and the *floors* on its components. It does NOT compute exact F-values for every realization. It does NOT promise that the floor is tight (a class may admit a tighter encoding than the floor). What S1' does is make the inter-class comparison well-defined: every candidate class has a computable lower bound on F, so finite candidates are hard-gated (R-10), quasiperiodic are excluded (definition), and crystallographic candidates can be ranked among themselves (`dl_comparison.py`).

---

## 4. S2' — d_eff = 3 dominant

### 4.1 Statement

**S2'.** Among Layer-2 realizations of F_inv(E) at varying effective spatial dimension d_eff ∈ ℤ≥1, the F-minimum is at d_eff = 3, with strict-positive margin to every alternative:

- **d_eff ≤ 2**: hard-gated. Gleason 1957 frame-function calculus has unbounded model-selection penalty for d_eff ≤ 2 (`d_spatial_derivation.md` §3, citing dimension_three_theorem.md Lemma 2). This is dimension-universal — not crystal-net-specific.
- **d_eff = 4**: F(d=4) − F(d=3) ≥ ε_struct(d=4-class) + 0.415·n bits. For crystallographic candidates, ε_struct ≥ +0.087 bits (`d_spatial_derivation.md` §4d). For non-crystallographic d=4 Cayley-quotient alternatives, ε_struct could in principle be negative; the per-stream-length term 0.415·n bits/event drives strict positivity for n > N*₄ := |ε_struct|/0.415. Empirical anchor R-4 (η_5 = 0 exact, `predictions/eta_5_lorentz_dim5.py`) further hard-gates by foreclosing the downstream observable channel.
- **d_eff ≥ 5**: F(d=5) − F(d=3) ≥ ε_struct(d=5-class) + 0.737·n bits. Empirical anchor R-5 (inherits R-4's mechanism via Cl(10) GUT-analog argument).

### 4.2 Hard quality gate

- Gleason d≥3 gate: Type 3 (Gleason 1957 *J. Math. Mech.* 6, 885-893).
- Crystallographic margin: Type 5 chain via `d_spatial_derivation.md` §4d.
- Tree-native data-DL margin n·log₂(d/3): Type 5 chain via `d_spatial_derivation.md` §4b (no crystal-net premise; already on file).
- Non-crystallographic alternatives' Kolmogorov floor: Type 3 via Li-Vitanyi 2008.
- R-4 / R-5 anchors: Type 4 via `../audits/registers/structural_residue_register.md` R-4 + `predictions/eta_5_lorentz_dim5.py`.

**Verdict:** S2' passes the gate as a Type 3 + 4 + 5 chain. STRUCTURAL-DERIVATION-GRADE-CONDITIONAL on (Gleason + R-4 + tree-native Step 4b).

### 4.3 Audit v2 (Clause 7) status

S2' inherits Row 3 audit v2 closure per an internal working note §3. Specifically:

- **(7a) Axes enumerated:** topology (crystal/finite/quasiperiodic/hyperbolic/non-vertex-transitive), d (1, 2, 3, 4, ≥5).
- **(7b) Alternatives named:** d=2 (Gleason hard-gate), d=4 (crystal: Brown rank → Cl(8) gauge fails Pati-Salam per R-4; non-crystal: per-stream-length margin), d≥5 (Cl(10) GUT-analog per R-5; per-stream-length margin), finite graphs (R-10 hard-gated by A2-T + A3-T), quasiperiodic (excluded by Cayley-quotient definition), hyperbolic Cayley quotients (R-13 OPEN, deferred).
- **(7c) M1-M6 gating:** M1 hard-gates d≤2 (Gleason) and finite (R-10) and quasiperiodic (definition). M2 = +0.087 + 0.415·n bits structural for d=4 crystal (closures_index §3 inheritance). M5 = no new amplification per R-4 closure (η_5=0 anchor). M3, M4, M6 = generic, parametric (closures_index §1).
- **(7d) Combined contribution:** d=4 crystal alternative Boltzmann-suppressed at any n > 0; d=4 non-crystal awaiting R-13 closure but bounded above by per-stream-length term; d≤2 hard-gated; finite hard-gated; quasiperiodic excluded.
- **(7e) Status:** DOMINANT-with-named-margins. R-13 (hyperbolic d=4 Cayley quotients) is the only OPEN sub-axis.

---

## 5. S3' — k_eff = 3 dominant

### 5.1 Statement

**S3'.** Conditional on S2' (d_eff = 3), the F-minimum over k_eff ∈ ℤ≥3 is at k_eff = 3, with strict-positive structural margin:

- **k_eff = 4**: F(qtz) − F(srs) ≥ +1.14 bits structural (an internal working note §2.1, M2 structural row). For long-chain observables (M ≥ 6), the M6 sign-flip mechanism at Γ + K (Re(h_qtz_Γ) = −1 forced) hard-gates qtz observationally. For data-conditional MDL, qtz is annihilated by ~2×10⁸ bits across V_cb + V_us + Q_Koide + Re(h_P) (closures_index §2.1 row).
- **k_eff ≥ 5**: hard-gated by tree-native Brown rank reformulation. At a vertex of the causal-state DAG with k incoming directed edges, Shalizi-Crutchfield 2001 Thm 2 gives k causal states; the rank of the Fisher information of the per-edge toggle distribution is bounded above by d_eff = 3. Excess edges (k − 3) contribute zero Fisher information and have positive model bits → MDL strictly eliminates them.

### 5.2 Hard quality gate

- Tree-native k-elimination: Type 3 (Brown 1986 Thm 1.13 applied to spectral-rank reformulation; Shalizi-Crutchfield 2001 Thm 2).
- qtz structural margin: Type 5 chain via closures_index §2.1.
- qtz data-conditional / M6 sign-gate: Type 5+7 via closures_index §2.1 — but **NOT inherited by S3'** because they are downstream of S5' (qtz spectrum requires lattice realization to compute). S3' uses **only the structural M2 piece** (+1.14 bits) to avoid circularity.

**Verdict:** S3' passes the gate as a Type 3 + 5 chain via the structural M2 margin alone. STRUCTURAL-DERIVATION-GRADE-CONDITIONAL on S2' + Row 4 audit-v2 structural row.

### 5.3 Why NOT inherit Row 4's full audit-v2 closure

Row 4's UNIQUE-on-η_B status (closures_index §2.2) uses M6 sign-gate (Re(h_qtz_Γ) = −1) and M2 data-conditional (Q_Koide ~2×10⁸ bits). These are *downstream* of srs identification — they require Hashimoto-spectrum computations on srs vs qtz. Citing them upstream of S5' would be circular for the bridge.

S3' uses only the structural M2 piece (+1.14 bits) which IS upstream-safe: it is computable from `dl_comparison.py` (which compares space-group + Wyckoff overheads, not lattice-realization spectra). The +1.14-bit margin is positive and grows with stream length via S2''s data-DL term.

The downstream-safe Row 4 mechanisms (M6, M2 data-conditional, M5 Feshbach) remain available to subsequent predictions that legitimately depend on srs spectrum.

### 5.4 Audit v2 (Clause 7) status

S3' inherits the structural piece of Row 4 audit v2 closure per closures_index §2.1.

- **(7a) Axes enumerated:** k_eff ∈ {3, 4, ≥5}.
- **(7b) Alternatives named:** qtz at k=4 (the closures_index named alternative), generic k≥5 (hard-gated by tree-native Brown rank).
- **(7c) M1-M6 gating:** M2 structural +1.14 bits (used). M6, M2 data-conditional, M5 NOT used here (downstream of S5'; available to inheritors).
- **(7d) Combined contribution:** +1.14 bits + 0.415·n per-stream-length amplification.
- **(7e) Status:** DOMINANT with computable structural margin.

---

## 6. S4' — |E| = 6 (REORDERED downstream of S5')

### 6.1 Statement

**S4'.** Conditional on S5' (srs identification), |E| = 6 follows from elementary arithmetic |E| = k*·|V|/2 = 3·4/2 = 6 (where |V| = 4 from srs Wyckoff position 8a per Row 8, k* = 3 from S3', and the /2 from undirected-edge sharing forced by A1 involutivity).

**This is NOT an upstream-of-S5' MDL minimization.** The scoping doc's S4-before-S5 ordering presupposed an MDL-over-n search machinery that does not exist in the framework and is not needed for the Path Dominant statement. The R-11 closure (`../audits/registers/structural_residue_register.md` R-11) already provides this derivation at UNIQUE-CONDITIONAL grade.

### 6.2 Hard quality gate

- |E| = k*·|V|/2 arithmetic: Type 2 (algebra).
- k* = 3: Type 5 via S3'.
- |V| = 4: Type 3 via International Tables for Crystallography Vol. A (I4₁32 Wyckoff 8a multiplicity) + S5' (srs identification).
- /2 from undirected edges: Type 1 via A1 involutivity.

**Verdict:** S4' passes as a Type 1+2+3+5 chain. UNIQUE-CONDITIONAL on Row 6 (S5'). This is the existing R-11 closure (`../audits/registers/structural_residue_register.md` R-11), preserved.

### 6.3 What this concedes

The original Path Full ambition was to derive |E| from MDL-over-alphabet-size FIRST and THEN identify srs as a consequence. Path Dominant abandons this ambition. The framework's actual derivation chain has |E| as a *downstream consequence* of srs identification, not an upstream MDL minimization. This is consistent with the framework's other "primitive cell vertex count" derivations (Row 8) and does not introduce new circularity.

If a future research item ever produces an upstream-of-srs MDL-over-n argument, S4' would be promoted from UNIQUE-CONDITIONAL to UNIQUE-THEOREM-GRADE. Until then, |E|=6 is conditional on srs.

---

## 7. S5' — srs uniqueness across realizations (PARTIAL closure)

### 7.1 Statement

**S5'.** Among Layer-2 realizations satisfying d_eff = 3 (S2') and k_eff = 3 (S3'), the F-minimum is at the srs lattice in space group I4₁32 with both chiralities retained, with closure status varying by candidate subclass:

- **Crystallographic 3D 3-regular subclass:** srs is DOMINANT by +1.68 bits over nearest competitor (ths) via `dl_comparison.py`; ths and dia hard-gated by R-7/R-8 (centrosymmetric, fail R-12 chirality requirement); restriction to chiral subclass gives srs UNIQUE among entries currently catalogued in dl_comparison.py (R-9 OPEN-restricted-to-chiral, RCSR enumeration of any other chiral 3D 3-regular nets remains pending).
- **Finite 3-regular graph subclass:** hard-gated by R-10 (Rows 11 + 13 require infinite substrate for A2-T waterline + Stone's theorem + Bloch decomposition + von-Neumann type II_1 factor structure).
- **Quasiperiodic subclass:** excluded by Path Dominant's candidate-set definition (quasiperiodic tilings are not Cayley quotients of finite-index normal subgroups; Senechal 1995 §3).
- **Hyperbolic Cayley quotient subclass:** **R-13 OPEN, deferred to research-level enumeration.** F_inv(6) admits hyperbolic Cayley graph realizations; their MDL bounds vs srs require explicit case-by-case analysis. Bounded above by S2''s tree-native per-stream-length term (0.415·n at d_eff=4, 0.737·n at d_eff=5); a hyperbolic d_eff=3 quotient would have the same data-DL as srs and require structural-DL comparison case-by-case.
- **Non-vertex-transitive Cayley quotient subclass:** Brown rank applies per-orbit; multi-orbit candidates pay log₂|orbit-count| structural-DL overhead. Subsumed by `g_girth_derivation.md` Cases 2-3 within the crystal subclass; non-crystallographic non-vertex-transitive case shares the R-13 enumeration treatment.

### 7.2 Hard quality gate

- Crystallographic closure: Type 3 (Sunada 2012) + Type 5 (`dl_comparison.py`).
- Finite hard-gate: Type 5+7 via R-10 + closures_index inheritance.
- Quasiperiodic exclusion: Type 3 (Senechal 1995 §3).
- Hyperbolic R-13 deferred: BLOCKED-CONDITIONAL pending future research.

**Verdict:** S5' partial closure passes the gate for crystal + finite + quasiperiodic subclasses. Hyperbolic subclass is acknowledged-OPEN.

### 7.3 The chirality requirement

R-12 (chirality both-hands retention) is **load-bearing for the framework's parity violation** (`../audits/registers/structural_residue_register.md` R-12). It hard-gates centrosymmetric alternatives (R-7, R-8 REFUTED via R-12) and restricts the candidate set to chiral nets. R-12 itself is derived from A2-T waterline (`../framework/framework_axioms.md` §3 — "regime where waterline admits multiple representations simultaneously") + mirror-image isomorphism of srs's two hands. It is not an external structural input.

The "infinite-extent filter" referenced in `uniqueness_ledger.md` Row 6 Gap is now closed by R-10 (the conjunction of A2-T finite-register + Stone's theorem + Bloch decomposition + type II_1 factor requirements forces infinite substrate).

---

## 8. New residue: R-13 (hyperbolic Cayley quotient class)

A new R-N entry is added to `../audits/registers/structural_residue_register.md`:

**R-13 — Hyperbolic Cayley quotient class.**
- *Source row.* This theorem, S5' partial closure.
- *Discarded alternative.* Hyperbolic Cayley graphs of F_inv_n (n=6 minimum) at d_eff = 3 (or d_eff = 4 with per-stream-length amplification).
- *Soft-gating margin.* For d_eff = 3 hyperbolic quotients, structural-DL difference vs srs to be enumerated; per-stream-length data-DL is 0 (same d_eff). For d_eff = 4 hyperbolic quotients, ≥ 0.415·n per-stream-length amplification.
- *Candidate downstream observable.* Currently unknown. Hyperbolic Cayley graphs have negative curvature and characteristic Gromov-hyperbolic spectral signatures; downstream observable would be a curvature-induced correction to spectral observables. NO known phenomenology.
- *Status.* **OPEN, research-deferred.** Enumeration requires explicit construction of finite-index hyperbolic Cayley quotients of F_inv(6) and computation of their structural-DL via either Kolmogorov floor (Li-Vitanyi) or explicit hyperbolic-tessellation encoding (Coxeter group-action approach). Multi-session research item.

---

## 9. Summary of closure status

| Sub-theorem | Closure status | Hard quality gate types | Audit v2 Clause 7 |
|---|---|---|---|
| S1' | THEOREM-GRADE | 3 + 5 | N/A (foundational machinery) |
| S2' | STRUCTURAL-DERIVATION-GRADE-CONDITIONAL on Gleason + R-4 + Step 4b | 3 + 4 + 5 | inherits Row 3 (closures_index §3) |
| S3' | STRUCTURAL-DERIVATION-GRADE-CONDITIONAL on S2' + Row 4 structural M2 | 3 + 5 | inherits Row 4 structural piece (closures_index §2.1) |
| S4' | UNIQUE-CONDITIONAL on Row 6 (R-11 closure) | 1 + 2 + 3 + 5 | inherits Row 7 (closures_index §3) |
| S5' (crystal subclass) | DOMINANT via Sunada + dl_comparison | 3 + 5 | inherits Row 6 (closures_index §3) |
| S5' (finite subclass) | hard-gated via R-10 | 5 + 7 | R-10 closure |
| S5' (quasiperiodic subclass) | excluded by definition | 3 | Senechal 1995 §3 |
| S5' (hyperbolic subclass) | **OUT-OF-SCOPE per Option A (2026-05-01)** — Bloch-decomposable scope restriction; not addressed by this bridge | — | scope statement §0a; R-13 OPEN as scope-disclosure residue per an internal working note §0 |
| Path Dominant overall | **STRUCTURAL-DERIVATION-GRADE-CONDITIONAL within Bloch-decomposable substrate class** (Option A scope restriction 2026-05-01) | composite | composite |

**Net for the structural ledger.** The "crystal-net premise" gap referenced in `uniqueness_ledger.md` Row 4 *Conditional on* field is closed by inheritance from this theorem: the chain Cayley(F_inv(E)) → d-periodic crystal net is the F-functional's MDL-minimum-Layer-2-realization output (sharp-peak case within the Bloch-decomposable scope: F-functional uniqueness gives a single dominant Layer-2 realization at every n; per `feedback_a2_waterline.md`, waterline = strict-min agree in the unique-peak regime, so the "MDL minimum" framing here is genuine and not subject to the `canonical_encoding`/`channel_select` operator split of `theorem_lattice_coupling_general.md` §2) (S2' + S5' partial), not a separately stipulated premise. R-13 is the only OPEN sub-axis.

---

## 10. What the bridge does NOT close

Honest list of acknowledged-open items:

1. **R-13 (hyperbolic Cayley quotient enumeration) — OUT-OF-SCOPE per Option A (2026-05-01).** Build plan: an internal working note. Stage 1a (2026-04-30): 5/6 sub-classes REFUTED via existing hard-gates. Stage 1b interim (2026-05-01): Sub-case B/E exists in principle (Bruno-Mecchia theta-curve orbifolds) but natural constructions collapse to triangle groups. **Stage 1b' attempted closure RETRACTED 2026-05-01** (an internal working note §0): the Bloch-decomposition-hard-gate argument was found CIRCULAR by user inspection (using framework-internal Bloch P-point eigenvalue h = (√3+i√5)/2 as exclusion criterion for non-srs alternatives is circular when the bridge is supposed to JUSTIFY that machinery's domain). **Option A fix applied (this revert)**: bridge scope explicitly restricted to Bloch-decomposable substrates (§0a). R-13 candidates are out-of-scope of THIS bridge — neither refuted nor included. Closing R-13 substantively would require: (a) showing no Plancherel-based mass identification on R-13 candidates can match observed discrete particle masses, OR (b) independently deriving Bloch-decomposability from {A1, P1', A5-mass} alone. Neither done; both are research items.
2. **Strict global UNIQUE.** Path Dominant explicitly does NOT claim strict UNIQUE — d > 3 substrates are MDL-permitted at finite n. The user's softer DOMINANT-with-steep-dropoff statement is the actual achievable target.
3. **Quantitative Kolmogorov floor for non-crystallographic candidates.** S1' establishes the floor *exists* (Li-Vitanyi 2008); concrete encodings beyond the named subclasses are not exhibited.
4. **Asymptotic-safety / RG-running questions.** Whether the Path Dominant statement survives at observation lengths beyond current empirical reach is an open structural question. Empirical anchors (η_5 = 0, etc.) provide finite-n protection.

---

## 11. Loose-end maintenance

This theorem closes:
- The "crystal-net premise" Gap in `uniqueness_ledger.md` Row 4 *Conditional on* field (now: bridge-DOMINANT-conditional).
- The "infinite-extent filter" Gap in `uniqueness_ledger.md` Row 6 (now closed by R-10's conjunction).
- The Path Full S4-before-S5 ordering in `theorem_substrate_layer1_layer2_bridge_scoping.md` §5 (now: S4 logically downstream).
- Question 1 of the scoping doc §10 (infinite-extent filter — closed via R-10 + S5').
- Question 2 of the scoping doc §10 (chirality — derived from A2-T waterline per R-12, not external input).
- Question 3 of the scoping doc §10 (quasiperiodic / hyperbolic) — quasiperiodic excluded by definition; **hyperbolic R-13 OUT-OF-SCOPE per Option A scope restriction 2026-05-01** (Stage 1b' closure attempt RETRACTED as circular).

This theorem does NOT close:
- Question 4 of the scoping doc §10 (what if S4 fails) — S4 is reordered as downstream of S5', so the question is reframed.
- Question 5 of the scoping doc §10 (interaction with W4 catalog) — orthogonal; W4 remains open separately.

---

## 12. References

- Brown, L.D. (1986). *Fundamentals of Statistical Exponential Families*. IMS Lecture Notes Vol. 9. Theorem 1.13.
- Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*, 2nd ed. Theorem 2.5.1.
- Delgado-Friedrichs, O. & O'Keeffe, M. (2003). *Acta Cryst.* A **59**, 351-360. §2.1.
- Gleason, A.M. (1957). Measures on the closed subspaces of a Hilbert space. *J. Math. Mech.* **6**, 885-893.
- Li, M. & Vitanyi, P. (2008). *An Introduction to Kolmogorov Complexity and Its Applications*, 3rd ed. Springer. §3.1, Theorem 2.4.1.
- Magnus, W., Karrass, A. & Solitar, D. (1976). *Combinatorial Group Theory*, 2nd ed. Dover. (Residual finiteness of free products.)
- OEIS Foundation Inc. (2024). Sequence A006227: Number of n-dimensional space groups.
- Plesken, W. & Schulz, T. (2000). Counting crystallographic groups in low dimensions. *Experimental Math.* **9**, 407-411.
- Rissanen, J. (1983). A universal prior for integers. *Ann. Statist.* **11**, 416-431.
- Senechal, M. (1995). *Quasicrystals and Geometry*. Cambridge UP. §3.
- Serre, J.-P. (1980). *Trees*. Springer. §I.1 Prop. 4.
- Shalizi, C.R. & Crutchfield, J.P. (2001). Computational mechanics. *J. Stat. Phys.* **104**, 817-879. Theorem 2.
- Shannon, C.E. (1948). A mathematical theory of communication. *Bell Syst. Tech. J.* **27**, 379-423.
- Sunada, T. (2012). Crystals that nature might miss creating. *Notices AMS* **59**(2), 208-215.

Internal:
- `../audits/registers/uniqueness_ledger.md` Rows 3, 4, 6, 7, 8, 10.
- `../audits/registers/structural_residue_register.md` R-4, R-5, R-7, R-8, R-9, R-10, R-11, R-12, R-13 (new).
- `../framework/framework_axioms.md` §§2, 3 (A1, A2-T).
- `predictions/d_spatial_derivation.md` §§2-4.
- `predictions/walker_dynamics_derivation.md` Steps 1-7.
- `predictions/observer_hilbert_space.py` (G.1 + G.5 closure under A3-T via CDP 2011).
- `predictions/k_star.py` and `predictions/k_star_derivation.md`.
- `predictions/g_girth.py` and `predictions/g_girth_derivation.md`.
- `proofs/foundations/dl_comparison.py`.
- `proofs/foundations/r11_alphabet_localization_check.py`.
