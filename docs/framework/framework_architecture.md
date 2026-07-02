# Framework architecture — multi-layer view

**Date:** 2026-04-17
**Status:** Consolidation / orientation document. Not a theorem. Anchors scattered prior work into a single stack.

## Purpose

Prior to this doc, the framework's multi-layer structure was implicit in derivation chains and scattered across external research notes. This document makes the layering explicit so (a) new contributors can orient quickly, (b) Sprint planning can target specific layers, and (c) it is clear which phenomena live in which layer. No new derivations are performed here; only existing rigorous or partially-rigorous results from the framework are referenced.

## One-line summary

The framework stacks seven layers between the two axioms and observed Standard Model phenomenology. The layers are (from substrate upward): **axiom → multiway substrate → compressed visible sector (srs) → gauge rep factor → observer Hilbert space → SUSY closure layer → dark sector residue**. Physical particles live at specific layers with different compression behavior (local / global / Dirac); three generations arise from the *observer Hilbert space* layer (C³_gen), not from a symmetry of the visible sector.

## The layers

### Layer 0 — Axioms (updated 2026-05-08)

> **Status (post-2026-05-08 axiom-slate revision):** Both A1 and A2 are now *derived theorems* of the framework's three irreducible commitments — (A) self-containment + (B) finite observer + (I) active reading of binary distinctions — plus standard published mathematics (Shannon, Jaynes, Rissanen, Cover-Thomas, Serre). See [`framework_axioms.md`](framework_axioms.md) §10 for the canonical current statement and the demotion chain. The layering below remains structurally valid; only the framing of which items are "axioms" vs "theorems" has shifted. A1 and A2 retain their names in this and downstream docs because the content is unchanged.

**(A1)** Binary self-inverse toggle. A two-state flip, applicable at locations in a multiway state space, self-inverse so that two applications return the original state. *Now derived* via [`../theorems/theorem_toggle_from_self_containment.md`](../theorems/theorem_toggle_from_self_containment.md) (2026-05-07).

**(A2)** Minimum Description Length (MDL). Not "minimum information" — the framework's reading follows Rissanen 1978/1983, Grünwald 2007 §§5.1–5.3: *selective retention of information that pays for itself*. A model M is retained if its description cost L(M) plus its residual data cost L(data|M) is less than the data cost alone. Noise is discarded; detail that reduces description length is retained. *Now derived* via [`../theorems/theorem_A2_mdl_from_finite_register.md`](../theorems/theorem_A2_mdl_from_finite_register.md).

The framework's foundational commitments at Layer 0 are therefore (A) + (B) + (I); A1, A2 (and A3, A4 at Layers 4–5) are derived theorems above this commitment layer.

### Layer 1 — Multiway substrate

Toggle dynamics are not random. Following Wolfram / Gorard 2020, applications of the toggle at distinct locations in a state space produce a *multiway* graph: branches at each rewrite step, with causal structure between branches. The substrate is this multiway graph.

**Currently in the repo:**
- `../../predictions/walker_dynamics_derivation.md` §Step 8 cites Sunada 2012 §§5–6 for the Bloch decomposition that compresses multiway content into per-k fibers.
- An external research note on dark-matter compression treats dark matter as multiway branches that fail MDL compression.
- A separate external research note on dark-energy branching proposes branching rate ε ~ 10⁻⁶¹ as tied to Hubble parameter / cosmological constant.

**Gaps:** no formal derivation of branching rate from axioms. No theorem connecting multiway causal invariance to the specific srs structure.

### Layer 2 — Compressed visible sector (srs + Hashimoto walker)

The MDL observer's optimal projection of the multiway substrate. `../../predictions/walker_dynamics_derivation.md` (W1–W3) closes: observables are spectral statistics of a non-backtracking (NB) walk on the srs lattice, where the Hashimoto matrix B is the 1-step transition operator on directed edges.

**Lattice characteristics** (all derived, Layer 0/0.5):
- Spatial dim d = 3 via MDL + Gleason + Fisher (`predictions/d_spatial_derivation.md`).
- Coordination number k* = 3 via MDL-optimal edge redundancy (`predictions/k_star.py`).
- Girth g = 10 (srs is the {10,3}-a Laves graph, Sunada 2012).
- Space group I4_132 with point group 432.
- Hashimoto eigenvalue h = (√3 + i√5)/2 at the P-point, C_3-protected mult 2 (`../../predictions/B_P_doubly_degenerate_h_derivation.md`).
- Cubic moment and symmetric invariants (`predictions/srs_cubic_moment_derivation.md`).

This layer is **label-agnostic**: it is pure graph/walk theory. Physics labels enter only in subsequent layers.

### Layer 3 — Gauge rep factor

The srs Clifford algebra content: under B1.b (invariant tensor-algebra construction) + B2 (signature (6,0)) + B3 (spinor decomposition), the 8-dim Cl(6,0) spinor decomposes under Spin(4)×Spin(2) = SU(2)_L × SU(2)_R × U(1)_{B−L} as one Pati-Salam family (ν, e, u, d) × (L, R), with color factored out.

Under B6 (2026-04-17, `docs/theorem_B6_bridge.md`): the body-diagonal C_3 of point group 432 lifts via Lawson-Michelsohn Spin(6)≅SU(4) to the Z_3 center of SU(3)_color. **The srs point group's C_3 is color-Z_3**, not a generation index.

**Rigorously closed at this layer:**
- One Pati-Salam family per gauge-rep-factor instance (B3).
- Color-Z_3 via srs C_3 lifted through Spin(6)≅SU(4)→PS (B6).
- SU(2)_L × SU(2)_R × U(1)_{B−L} electroweak + B−L structure (B3).

**Adopted-postulate at this layer:**
- Pati-Salam labeling of Spin(4)×Spin(2) factors. This labeling is *dimensionally forced* by the 8-state Cl(6,0) spinor fitting one PS family but not a horizontal-flavor structure.

**Blocked:**
- SU(3)_color as a continuous gauge group beyond its Z_3 center: B4 route-(iv) fallback. Requires external structural input.

### Layer 4 — Observer Hilbert space (C³_gen)

Separate tensor factor orthogonal to the gauge rep factor. **Three generations live here.**

**Derivation chain** (developed in an external research note on the dimension-three theorem, partially imported into `predictions/d_spatial_derivation.md`):
1. The MDL observer must assign probabilities non-contextually (Lemma 1: contextual models carry more parameters, ~n²+n−1 vs n²−1, MDL strictly prefers non-contextual).
2. Non-contextuality requires Hilbert space dim n ≥ 3 (Lemma 2 + Gleason 1957: in n=2 the frame function is underdetermined; in n≥3 Gleason forces Born rule).
3. MDL selects n = 3 as minimum viable dimension (Lemma 4: model cost grows as n², data-fit benefit as log n).
4. Three basis states of C³_gen = three SM fermion generations (stated as Result #7 in an external research note, to be formalized as Sprint 11 B7.2).

**Structural consequence:** Gauge reps tensor with C³_gen, so all three generations inherit identical gauge charges — matches observed SM (e, μ, τ all charge −1; u, c, t all charge +2/3; etc.).

**Gaps / Sprint 11 scope:**
- B7.1: formalize as `../../predictions/observer_dim_three_derivation.md` with srs-specific proof; close the "Proposition" (n ↔ d_s) gap.
- B7.2: prove "three basis vectors = three SM generations" as a bridge theorem.
- B7.3: derive mass operator M on C³_gen from MDL-optimal observer + srs walker dynamics.
- B7.4: re-derive Q_Koide, ε_Koide, δ_Koide under M on C³_gen.
- B7.5: re-derive PMNS / CKM from (U_charged − U_neutral) / (U_up − U_down) mismatch on C³_gen.

### Layer 5 — gauge β-coefficient values (named-MSSM convention)

The substrate-derived matter content is 3 PS generations + 2 Higgs doublets, with no superpartners (Cl(6) Fock all-fermionic per Path-E recheck 2026-05-12; A1 thermal-apparatus closure 2026-05-27). The MSSM β-coefficient values (33/5, 1, −3) that the framework predicts are derived TOP-DOWN by the run's 4D time-completion (`derivation_topdown/bridge/the_run.py` `read_gauge_running` — exact `{33/5,1,−3}`, no PDG input; [R-19 de-escalation](../audits/registers/structural_residue_register.md)); the earlier PDG-inversion ([`theorem_beta_coefficients_derived.md`](../theorems/theorem_beta_coefficients_derived.md)) is now the data-side cross-check confirming consistency. Forced-ness of the completion = ζ_{D₄}(0) (research-level).

The gap between substrate-derived 2HDM β (b_2 = −3) and observation-imposed MSSM β (b_2 = +1) is precisely characterized as **Δb_2 = +4 at SU(2)_L** ([R-19](../audits/registers/structural_residue_register.md) in the structural residue register, A1 Session 1 closure 2026-05-27). Literal sparticle realization is one possible mechanism for the observed β-values; the framework does not commit to literal-particle interpretation. R-parity is explicitly VIOLATED (I4_132 has no inversion symmetry; proton stability maintained by Z_3 triality regardless of whether literal SUSY partners exist).

Substrate-side derivation of the literal-particle interpretation comprehensively exhausted 2026-05-27 across Branch A's bounded routes:
- A1 (heat-kernel / Candidate D): substrate's thermal apparatus is standard-QFT-compatible at one loop but does not modify b_2 from the 2HDM value.
- A3 (V_Ram ≅ Cl(6) Fock iso re-examination): the iso is matter↔gauge pairing, not boson↔fermion partner map; confirmed 2026-05-12 Path-E prediction.
- A4 (unused Ramanujan saddles): h_N structurally independent but observationally inert; h_Γ/h_H went to neutrino sector via [chir-7 theorem](../theorems/theorem_neutrino_chir7_concentration_2026-05-21.md).

See the SUSY-load-bearing audit for the full classification: no framework prediction or theorem-grade derivation depends on literal SUSY particles.

**Unified-theory framing.** Layers 1-6 are partial readings of one substrate object — see [`theorem_walker_matter_unification_2026-05-27.md`](../theorems/theorem_walker_matter_unification_2026-05-27.md), which consolidates the layer-by-layer architecture into a single statement (the substrate's Hashimoto walker structure organizes matter + gauge + Yukawa + cosmology). This is a consolidation of existing theorem-grade results, not a new derivation.

### Layer 6 — Dark sector (uncompressed residue)

Multiway branches that have internal structure but fail MDL's "pays for itself" criterion at the geometric level. They retain quantum labels (spin, mass) but don't simplify into srs-lattice-like form.

**Gravitational coupling universal (structure-independent); gauge coupling branch-specific** (per an external research note on dark-matter compression). This predicts:
- Dark matter gravitates but has no SM gauge interaction.
- Dark matter fraction ≈ 0.85 (derived 2/3 + corrections — see `predictions/Omega_DM_over_Omega_m.py` 0.842 theorem grade).
- Cosmological constant from branching-rate → Hubble (speculative, per an external research note on dark-energy branching).

**Gap:** Dark-sector *structure* beyond "uncompressed" is not formalized. The dark-matter-fraction derivation stays at theorem grade; dark-sector particle spectrum is not derived.

### Layer 7 — Particle classification (local / global / Dirac)

Particles in the framework come in three classes with different compression behavior:

- **Local graph features**: vertex or edge observables (e.g., toggle state at a specific vertex). Compress well; have spatially localized interpretation.
- **Global Bloch modes**: delocalized over the entire lattice (e.g., photons — Bloch modes of the walker Bloch operator). Compress well globally but have no local particle-like identity.
- **Dirac spinors**: 8-dim Cl(6,0) spinor content (B3) — neither local nor global in the graph sense; they are representation-theoretic objects with both local (at each primitive cell) and global (via Bloch factor) aspects.

This classification is **implicit** in the repo. Explicit statement and cross-layer consistency audit is a Sprint 11 B7.6 sub-task.

## Layer interactions

| Layer | Depends on | Produces |
|---|---|---|
| 0 Axioms | — | Toggle operator, MDL principle |
| 1 Multiway substrate | 0 | Branching multiway graph |
| 2 Visible srs | 0, 1 | srs lattice + Hashimoto walker + Bloch spectral data |
| 3 Gauge rep | 2 | Cl(6,0), Spin(6)≅SU(4), PS family (no colour), color-Z_3 |
| 4 C³_gen | 0, 1 (via Gleason) | Three generations as basis of observer Hilbert space |
| 5 SUSY | 3, 4 | MSSM spectrum, gravity-mediated soft masses |
| 6 Dark | 1, 2 | Uncompressed multiway residue; Ω_DM ≈ 0.842 |
| 7 Particle types | 2, 3, 4 | Local/global/Dirac classification |

The full fermion Hilbert space per particle type factorizes as:

$$\mathcal{H}_{\text{fermion}} = C^3_{\text{gen}} \otimes \text{(gauge rep factor)} \otimes \text{(Cl(6,0) Dirac spinor)}$$

with the C³_gen (Layer 4) carrying generation index, the gauge rep factor (Layer 3) carrying color / electroweak charges, and the Cl(6,0) factor carrying chirality and species information.

## What this architecture clarifies

1. **Three generations are orthogonal to color**, not the same symmetry. Both are Z_3-like at the group-theoretic level, but they act on different tensor factors. B6 retires the previous conflation.

2. **Koide Q = 2/3** is a color-sector arithmetic identity at Layer 3; its match to the charged-lepton Koide relation (which is a C³_gen-layer relation) is either a coincidence or signals a deep cross-layer structure not yet derived. Sprint 11 B7.4 tests the latter.

3. **Dark matter is a framework consequence**, not an add-on. Layer 6 is populated automatically by MDL's selective retention: structure that doesn't compress is still there, just invisible to gauge interactions.

4. **SUSY is framework-required**, not assumed. If Sprint 11 B7.6 closes, this becomes a theorem rather than an adopted result.

5. **The multiway substrate (Layer 1) is the information-theoretic deep structure**. Everything else is a projection. Questions like "why 3 generations not 4" may ultimately require a multiway-level combinatorial argument (how many strata pay for themselves under MDL).

## Status of each layer

| Layer | Status | Notes |
|---|---|---|
| 0 Axioms | ✅ Fixed | Memory memos: `feedback_rigor_bar.md` locks these as the only axioms. |
| 1 Multiway substrate | 🟡 Scoping doc (2026-04-17) | an internal working note unifies dark + multiway; Conjecture MS.1 (stratification-cost) proposes B7.3 framing. Formal multiway rewrite-system treatment still open. |
| 2 Visible srs | ✅ Closed | W1–W3 (`theorem_walker_dynamics.md`) + BP + Layer 0/0.5 derivations |
| 3 Gauge rep | ✅ Electroweak + color-Z_3 closed (B1.b, B2, B3, B6); 🟡 full SU(3)_c is route-(iv) fallback (B4) |
| 4 C³_gen | ✅ Structural closure (Sprint 11 B7.1 + B7.2, 2026-04-17): observer dim=3 (`../../predictions/observer_dim_three_derivation.md`) + three-basis-vectors=three-generations bridge. Open: mass operator M_gen (Sprint 11 B7.3). |
| 5 SUSY | 🟡 Scoping doc (2026-04-17) | an internal working note; necessity paths A–D identified, Path B (MSSM RG) most tractable pending Sprint 7 v_Higgs. |
| 6 Dark sector | 🟡 Ω_DM closed + scoping doc | Ω_DM = 0.842 at theorem grade (`predictions/Omega_DM_over_Omega_m.py`); structural framing via an internal working note; particle spectrum not derived. |
| 7 Particle types | ✅ Classification doc (2026-04-17) | `particle_type_classification.md` — 5 classes, every SM particle mapped, BSM candidates tabulated. Makes B7.2 Step 5 (gauge bosons lack C³_gen) rigorous. |

## References

- `../../predictions/walker_dynamics_derivation.md` — Layer 2 (W1–W3).
- `../../predictions/B_P_doubly_degenerate_h_derivation.md` — Layer 2 (P-point spectrum).
- `../../predictions/theorem_B1_ordering_derivation.md`, `theorem_B2_signature.md`, `theorem_B3_spinor_fermion.md` — Layer 3 (Cl(6,0), PS family).
- `docs/theorem_B5_3_core.md` — Layer 3 (C_3-isotypic sub-bundles).
- `docs/theorem_B6_bridge.md` — Layer 3 (color-Z_3 identification).
- `predictions/d_spatial_derivation.md` — Layer 4 (MDL + Gleason).
- External research note on the dimension-three theorem — Layer 4 upstream (to be ported Sprint 11 B7.1).
- External research note on analytical-results session 2, §7 — Layer 4 (Result #7 on three generations).
- External research note on dark-matter compression — Layer 6.
- External research note on dark-energy branching — Layer 6 / Sprint 6.
- `../parameters/predictions.md` lines 47–66 — Layer 5 (MSSM spectrum).
- `docs/master_plan.md` §Sprint 11 — path to formalize Layers 4–7.

## Scope honesty

This document does not derive anything. It organizes existing content. The honest state of the framework post-B6 (2026-04-17) is:

- Layer 2 is rigorously closed (modulo a few structural sub-items).
- Layer 3 electroweak + color-Z_3 closed; full SU(3)_c requires external input.
- Layer 4 (three generations) has a partial derivation chain via dim-3-theorem; full formalization is Sprint 11's main deliverable.
- Layers 5–7 are invoked but not formalized at theorem grade.

The bulk of flavor observables (Koide, PMNS, CKM, fermion masses) are BLOCKED pending Sprint 11 closure of Layer 4.
