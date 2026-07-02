# Self-MDL ledger — the framework's own description length

**Instrument:** `scripts/self_mdl_audit.py` (on-demand; not wired to CI/hooks).
**Question:** is the framework *shorter than the data it predicts*?
`net = earned − spent`, in bits. Positive net is the MDL standard of evidence —
the claim that survives forking-paths objections in a way a match-count never can.

**Status: PROVISIONAL v0 (2026-06-10).** The number moves as the program moves —
that is the design: this is a **living tracker, not a one-shot verdict**. Each
program phase updates specific rows (see "Phase hooks" below). Trust gates before
citing the number externally:
- [ ] Adversarial methodology review (ultracode gate) — its job is to find
  **uncounted freedom** on the spent side (post-hoc form choices, reading-rule
  degrees of freedom not in the registers).
- [ ] Full P-row coverage (all 68 parameter-ledger rows scored, not the ~35 headline rows).

## Frozen scoring rules (change only via dated amendment)

1. **earned(Q)** = log₂(prior width / achieved width), achieved = max(σ_exp, |residual|).
   The framework is credited only down to the precision it actually delivers
   (e.g. α_EM is NOT credited at its ppm experimental precision).
2. **Prior classes (frozen):** angles uniform [0°, 90°]; phases uniform [0°, 360°);
   magnitudes in (0,1) uniform; couplings log-uniform [10⁻⁶, 4π]; masses/dimensionful
   log-uniform [10⁻³ eV, M_Pl]; O(1) ratios log-uniform [10⁻³, 10³]; small ratios
   log-uniform [10⁻¹⁵, 1]. Exact-zero predictions earn log₂(prior/bound).
3. **Independence rule:** earned side counts an independent generating set only
   (no R_ν *and* both Δm²; no Yukawas *and* masses; t₀ excluded as derived from H₀;
   **v excluded** — it is the calibration, paid on the spent side).
4. **spent:** UNIQUE-unconditional rows pay 0; discrete selections pay log₂(k) over
   the row's enumerated alternatives (DEFAULT k=8 when unenumerated); continuous
   calibrations pay the full information of the pinning measurement (N_hub↔G_F:
   27.1 bits); A5-mass labeling pays its residual assignment freedom; plus two
   CONVENTION charges (axiom slate 10, unenumerated-freedom buffer 10).
5. **Conservative direction:** earned rounds down, spent rounds up. Open targets
   (L6 cluster) earn 0 and are listed explicitly.

## Headline (updated 2026-06-12: Phase-4 pricing +4.6 ⊕ Phase-5.2 re-price −1.0, merge-composed)

| | bits |
|---|---|
| earned (35 quantities, 4 open rows at 0) | **377.1** |
| spent (18 choice-point rows) | **105.7** |
| **net** | **+271.4** (ratio 3.57) |

[Merge composition 2026-06-12: Phase-4 panel pricing (+4.6, four new
rows) and the Phase-5.2 ordered-check re-price (A5-mass 3.0 → 2.0,
−1.0) landed on parallel branches; composed here. Discharge floor if
Phase-4 items 2–3 discharge (D2-forcing ratified + E3 executed): spent
103.1, net +274.0, ratio 3.66.]

Largest single earners: θ_QCD exact-zero (35.9), m_τ (20.0), m_H (16.3).
Largest spends: N_hub↔G_F (27.1), A5(b) channel levels (19.0),
form-selection residual (15.0). [A5-mass labeling re-priced 15.3 → 3.0,
2026-06-11 5.2 panel; → 2.0 same day on the panel-ordered P-sign↔ω/ω²
identity check; v0 headline was 114.4 spent / +262.7 net / 3.30.]

## Phase hooks — where the program moves this number

| Program phase | Rows affected | Expected direction |
|---|---|---|
| 1 (zeta factorization) | "A5(b) channel levels" (19.0) → toward 0 if channels are forced functionals of one zeta; "form-selection residual" shrinks | spent ↓ |
| 2 (Born-rule Koide) | P2 aggregation moves from postulate to theorem; mass-sector form bits → 0 | spent ↓ |
| 3 (derived Lindblad) | dissipation-form adoption never enters the ledger at all | spent flat, robustness ↑ |
| 4 (spectral action) | ADOPTED-MSSM-Sb (3.0) → 0 if β-coefficients are induced; α_s/sin²θ_W earned rows gain precision — **Phase-4 outcome (panel 2026-06-12): hook did NOT fire on either branch.** ADOPTED-MSSM-Sb (3.0) → 0 branch: induced content is 2HDM-shaped, MSSM gaugino content structurally excluded within the frozen triple — the running convention stays adopted at 3.0 bits. Earned branch (α_s/sin²θ_W precision): no earned row moved (JAJ⁻¹ dressing unexecuted; b₂ column dictionary-conditional; b₁ unclaimed) | did not fire |
| 5 (EBR labeling) | A5-mass labeling (15.3) → log₂(residual assignments) after symmetry-forcing — **CLOSED 2026-06-11 (5.2 panel): 15.3 → 3.0**; hook fired as bookkeeping correction (−11.71, a v0 over-price) + residual enumeration (2.585 → 3.0), with **(b) = 0 credited to EBR/5.1 labels** — 5.1's product banked zero-cost as conditionality-hardening | spent ↓ (fired) |
| L6 closure (if 3 unlocks it) | n_s, σ_8, r_s, θ_* move off 0 | earned ↑ |

## Honest reading of v0

The +262.7 is *not yet citable*: the spent side is only as complete as the
enumerated choice points, and the entire history of this project says the danger
is freedom that doesn't look like a choice (channel selection, level prescriptions,
which saddle, which walk length). The CONVENTION buffer (10 bits) is a placeholder
for exactly that, and the adversarial review exists to replace it with a real count.
What v0 *does* establish: the accounting framework, the conservative direction,
and the per-phase tracking hooks — so improvement claims in later phases are
measured, not asserted.

## Amendment log

- 2026-06-12 (PHASE-4 PANEL, wf_46045a02-c63; verdict PARTIALLY WON, no
  kill fires, K4 recording clause triggered; errata E2 + corrections
  C1–C6 landed in 55b2d04 BEFORE this entry per the ordering constraint)
  — **ADD 4.6 bits, four rows:** (1) **+1.0 — 4.3 SECTOR-ANCHOR BRIDGE:**
  the dictionary-licensed identification triple-H → 3 generations (native
  per-fiber trace 8 vs anchor 6, ratio 4/3). It alone carries the absolute
  b₃ = −7 (native 8 → −17/3 = NO target; the discriminator is anchor-free).
  Carved OUT of the unenumerated-freedom buffer as now-enumerated;
  cross-ref the A5-mass dependency chain, NOT additive with it. This
  pricing is the acceptance-(iii) remedy executed. (2) **+1.6 —
  DIRAC-CLASS CONDITIONALITY of the Φ-condensate findings (K4 recording
  clause):** the mirror-even condensate Φ (and the 768 Higgs-quadratic
  share = 4TrΦ²) is D2-SPECIFIC (D3 fully mirror-odd, D1 mixed); charged
  to the Higgs/mirror rows ONLY (controls agree on volume/curvature/
  gauge); log₂(3); SENSITIVITY: → 0 if a future panel ratifies the
  scheme→D2 forcing argument (the frozen gauge scheme requires the Cl(6)
  factor only D2 has). (3) **+1.0 — JAJ⁻¹ SCHEME-TRUNCATION:** the frozen
  fluctuation scheme ran A-only; the logged deferral IOU lapsed
  unredeemed. Structural results panel-established robust (JAJ⁻¹
  off-block exactly 0); the gauge-kinetic NORMALIZATION rows are NOT
  certified. DISCHARGEABLE to 0 by erratum probe E3; no earned-side
  gauge-normalization claim until redeemed. (4) **+1.0 — CUT-LOCALIZED
  GRAVITATING-ENTROPY IDENTIFICATION (adoption-class, like I2):** the
  record-vs-cut-correlation fork resolved by role assignment;
  single-homed at the gravity final-state doc; does NOT ride free under
  "I2 applied at the cut". RATIFICATION REFUSED by the panel (the
  Clausius asset's native S_total is a single-stream record count);
  ratification path named in the doc. ARITHMETIC: spent 102.1 → 106.7;
  net +275.0 → +270.4 (ratio 3.53); discharge floor 104.1 / +273.0 /
  3.62. **BANKED ZERO-COST** (several UPGRADED to analytic by panel
  re-derivation): the 2Φ mirror decomposition in its gauge-COVARIANT
  form; deck-J globality + KO signs (−1,+1,−1) → KO-dim 2 for the
  canonical class; TRIM-as-J-crossing as the atom-trivial-class statement
  (now SEARCH-COMPLETE: intertwiner space exactly empty; dressed KO-6
  intra class exists at P, no global extension); heat anchors 384/3456
  ANALYTIC; the integer σ-potential m₄(t) = 3456 + 1536t² + 64t⁴ with the
  768+768 split ANALYTIC from {D,M} = 2MΦ; m₄ q-flatness + σ-m₆ zero
  (POINTWISE identities); chirality excess +128 = 2·dim; g₄ = 10496×15 AS
  THE DISTINGUISHED-BIVECTOR-BASIS STATEMENT ONLY; gaugino-absence
  C₂(su4) = 15/4·I (scoped: the content is that the frozen H is a spinor
  module); the σ-census BLOCK DECOMPOSITION (structure, not blindness);
  no-mirror-crossing-one-forms = 0 exactly, JAJ⁻¹ included; the I = 2S
  purity facts; trilemma-as-three-quantities (the identity only — the
  gravitating identification is NOT banked). **EXCLUDED FROM THE BANK:**
  the census blindness claim (REFUTED — GRID2 aliasing artifact, erratum
  E2: alias-free census block-discriminating, octet sign-flip); "FULL
  su(4)_PS UNIFORMITY" as an invariant statement (basis-relative); the m₈
  σ value +89362.23 (erratum: +28140.53); unconditional "MSSM structurally
  excluded" wording (scope: within the frozen triple); the W7 quartic-sign
  control as evidence (vacuous); "157440" as a unified-kinetic reading.
  **PROMOTION OBLIGATIONS carried:** (a) execute E3 before any earned
  gauge-normalization claim; (b) σ↔ths only via a new frozen spec (first
  gate = the alias-free census; profile class frozen; target = ths
  sfermion content (+2,+2,+2); kill = the b₁ ths-doubling 8/5 overshoot);
  (c) the b₂ su(2)_L seat stays a declared CANDIDATE (0 bits while
  unused); (d) the c_S identification's 1 bit stays until derived or
  discriminated.
- 2026-06-11 (R1 RATIFICATION PANEL — verdict PARTIAL; A5-mass row
  UNCHANGED at 2.0): the frozen sensitivity "R1 ratified → in-row 0" was
  adjudicated at ultracode (4 refuters + judge, wf_ecc5e682) and NOT
  executed. K-R1-1 circularity CLEAR (Leg B provenance git-verified
  pre-dictionary), K-R1-2 math CLEAR (Leg A probe
  `phase5_2_r1_perron_vev_uniform_mode_2026-06-11.py` airtight after
  ordered rewordings), K-R1-3 FIRES on the composition: mode-selection
  not closed (v_higgs Step 2 excludes pairwise couplings only, not
  staggered single-mode CW vacua), vertex↔edge mean transfer unstated,
  G3 pricing-home misattributed (axiom slate prices (A)+(B)+(I) only).
  Refile path recorded in the 5.2 spec (the closing argument exists —
  critical-mode uniqueness from the banked action + MDL mode-naming —
  but was not filed; panel cannot ratify unfiled arguments). POSITIVE
  YIELD: homogeneity-premise attack surface closed (framework-derived
  via the MDL uniform-zero-mode theorem, git-verified provenance).
  Standing notes: future R1 ratification inherits the Higgs/VEV(2)
  budget's dictionary-conditionality; "downstream consumes constant v"
  is consistency, not forcing — inadmissible as a future R1 case.

- 2026-06-11 (5.2 ORDERED IDENTITY CHECK executed — A5-mass 3.0 → 2.0):
  the panel-ordered P-sign↔ω/ω² check is DISCHARGED POSITIVE (gated probe
  `phase5_2_psign_omega_identity_2026-06-11.py`, gates I1–I5, verify-
  registered): the Sec-2.2 P-sign partition of the 8 Ramanujan P modes
  coincides exactly with the little-group irrep-class partition AND the
  conjugate C₃-content partition ({1,ω} ↔ +Re, {1,ω²} ↔ −Re); the mirror
  composite λ → −λ̄ maps the families onto each other at machine zero with
  ALL convention-free invariants identical (|λ| = √2, |arg λ¹⁰| =
  162.3876°, two forced 2-dim projective irreps each). The only
  distinguishing datum is the character NAME ⟹ the in-row "which P-sign
  family is up-type" bit IS the ω/ω² channel-labeling convention, priced
  ~1 bit at the Majorana-panel line (its single home; annotation there
  closed). Row prints 2.0 (in-row = Higgs placement log₂3 = 1.585,
  rounds up) per the panel-frozen sensitivity — no new panel required.
  Arithmetic: spent 102.1 → 101.1; net **+276.0** (ratio 3.73).
  Remaining sensitivity: R1 (Higgs-homogeneity) ratified → in-row 0.

- 2026-06-11 (5.2 RE-PRICING PANEL, judge continuation wf_b7e7b0db-569;
  executed in the same commit-set as the phase5-ebr merge, gated on the
  Sec-2.2-grain probe `phase5_2_ss22_grain_enumeration_2026-06-11.py`
  passing in verify.py — gates 24 → 12 → 6) — **A5-mass labeling row
  RE-PRICED 15.3 → 3.0 bits.** GRAIN FROZEN: walker_class_dictionary
  2026-05-27 §2.2 family partition (4,4,8,8,2,2,2,18), the partition this
  row cites. THREE-WAY ATTRIBUTION: **(a) −11.71 bits BOOKKEEPING /
  LEDGER CORRECTION** — log₂(8!) counted bijections violating banked
  sector mode-budgets (2!·2!·3! = 24 size-admissible) and double-counted
  the two content-identical dark/inert roles (/2! → 12 distinct
  admissible assignments = 3.585 bits); a v0 over-price, NOT an EBR/5.1
  result, never to be presented as one. **(b) 0.0 bits credited to Phase
  5.1** (REFUTED as a credit category — space-group labels are
  SM-content-blind; the 12-assignment enumeration consumes no 5.1
  output). **(c) 3.585 bits residual scope**, decomposed: 1.0 bit
  ν-orientation (which of h_Γ/h_H is ν_L = the Γ↔H mirror Z₂) lives
  OUTSIDE this row at its SINGLE HOME, the Phase-1.3 ν_L/ν_R orientation
  line (cross-referenced both ways, NOT additive); IN-ROW residual =
  2 (which P-sign family is up-type/CKM-source — newly named) ×
  3 (which size-2 family is the Higgs condensate; R1 Higgs-homogeneity ⟹
  Perron NOT granted zero cost) = 6 assignments = 2.585 bits → **row
  prints 3.0 (spent rounds up)**. R2 (branch-tie) MOOT at the frozen
  grain: spill sets = roots of λ² ∓ λ + 2 = parent Ramanujan sets — a
  spectral family definition, not a reading rule. **DEPENDENCY-CHAIN
  CONDITIONALITY (the row text):** "This row prices residual assignment
  freedom WITHIN the 2026-05-27 sector-level walker-class dictionary and
  is conditional on: the A5(a) axiom row; the dictionary's sector-level
  grade (the per-Weyl/Phase-1b expansion and the 8↔42 iso-redundancy are
  OUTSIDE this row, their freedom held by the unenumerated-freedom
  buffer row); the banked sector mode-budgets (V_Ram-iso T5, chir-7, A4
  closure, cycle-homology closure — priced at their own rows); the
  frozen §2.2 family partition; and the single-homed Phase-1.3
  ν-orientation and ω/ω² convention bits (cross-referenced, not
  additive)." MIRROR STATUS: not a space-group element; never a forcing
  or counting mechanism; earns no credit (orientation-blind; ν-pairing
  evidential weight C3-KILLED); its one bit is OUTSIDE the row at Phase
  1.3. **SENSITIVITIES FROZEN WITH THE ROW:** R1 ratified as
  framework-derived → row 1.0; P-sign proven identical to the ω/ω²
  channel-labeling convention (identity-check ORDERED, open) → row 2.0;
  both → row 0 in-row; any future panel striking the banked-budget rows
  re-opens (a), not (c). BANKED zero-cost (5.1's product,
  conditionality-hardening of the mass-sector rows, explicitly NOT
  spent-side reduction): P-saddle doublets are KINEMATICS
  (projective-forcing theorem — no 1-dim ω-irreps exist, any commutant
  operator has all-even multiplicities); Γ/H triplets forced 3-dim (two
  copies of ONE class, split = dynamics); the LG-vs-Ihara-Bass
  two-mechanism separation; the EBR layer adds no forcing.
  ARITHMETIC: spent 114.4 → 102.1; net +262.7 → **+275.0** (ratio 3.69).
  PROPAGATION: the Phase-1.3 amendment's "~18 bits inherited" reads
  "~6 bits inherited (A5-mass 3.0 + NU-MAJ-PHASE 3.0)" from this date
  (annotation at that entry).

- 2026-06-11 (COMBINED PANEL: Phase-2 wrap + 3.1 + 3.2 + 3.3; both phases
  close PARTIAL; 3.2 bet PARTIALLY WON, no kill fired) — Phase-3 rows:
  ADD ~1 bit: 3.2/S3 uniform-i.i.d.-Bernoulli(p) leak substituted for the
  frozen LM1 local degree-content event rule (load-bearing for the T3 I/12
  finding; structure-correlated leak untested). ADD ~1 bit: 3.2/S4 LM1
  referent bridge (per-VERTEX toggle census read as per-EDGE per-step leak;
  value-relevant — alternative size-biased bridge gives 0.938031 vs
  0.848796). ADD ~1 bit: 3.2 absorption/no-return premise (spec-frozen but
  underived; in tension with theorem_substrate_feshbach_dark_corrections —
  reconciliation obligation (a)). ADD 0-bit notes: 1-dim dark absorber
  (WLOG); LM-class narrowing to LM1-only (within the ≥1-model clause;
  LM2-unconstructed + LM3-rate-imported counted as tried patterns).
  ADD ≤1.6 bits: 3.1 model class M1/M2/M3 declared in-probe only (no frozen
  3.1 spec; 3 tried patterns). ADD ~1 bit: 3.3 rate-density readout
  identification (post-hoc split of the frozen kill wording; panel-vindicated
  under BOTH readouts — M1 spectrally empty, S(κ)² = I). ADD corrected 3.2
  acceptance scorecard row: (i) PARTIAL / (ii) PASS / (iii) PASS; S3
  construction-order gate FAILED (symmetry-forced, over-spec'd) — supersedes
  the probe's "(iii) FAILED" self-report. BANKED zero-cost: the exact step
  isometry; one-step diagonality theorem; same-slot superselection;
  full-history-commutativity impossibility (‖K₀†K_e‖ = √(q(1−q))); the
  panel-computed continuum substantiation of 3.1 ((Φ_τ)ⁿ → e^L at O(1/n));
  the n_s null under both readouts + n_s ≥ 4 structural in-class.
  PROMOTION OBLIGATIONS (named): (a) Feshbach bidirectional-flux
  reconciliation; (b) S1↔S3 visible-marginal tension (diff 0.086);
  (c) genuine branch-counting derivation of p.


- 2026-06-11 (Phase 2.2 stage-2 ratification panel: lemma RATIFIED; P2 =
  PARTIAL, K2 STRUCK) — the named K2-class residue (conjugate-aligned
  per-channel phase rule) is DISCHARGED: derived as Hermiticity + positivity
  of the canonical positive root √M (probe phase2_2_alignment_lemma; bits
  moved: ZERO — the lemma adds no selections). P2 residue is now EXACTLY
  three priced ~1-bit ids: fused √-placement (K3; Parseval rewording adopted
  — "Born weights decompose tr(M) over C₃ channels" forces p = ½ uniquely),
  uniform-over-CSCO, W2-on-U wiring (NOT erased by "canonical CSCO"). ADD
  0-bit convention line: canonical-positive-root (unique Q-maximizing
  branch; data-validated; no in-repo competitor). BANKED zero-cost: the
  lemma; window π/12 exact; Q = tr(M)/(tr √M)² = 1/(3w₀) basis-free;
  Hermiticity-requires-w₁=w₂ falsifiability lock; mass ratios match PDG
  μ/e, τ/e < 0.2%. LEADS (unpromoted): alignment ⟺ Q-maximality;
  Parseval p=½ forcing. DOWNGRADE: stage-1 "blind-side corroboration" →
  "sector-assignment consistency" (conflict-if-promoted flag attached:
  promoted, the saddle predicts two massless ν vs m_ν₁ = 0 single).
  Falsifiability lever: δ_u at 91.15% of its window (~2–6% ε² headroom).


- 2026-06-11 (Phase 2.2 stage-1 panel, verdict PARTIAL) — ADD ~3 bits:
  W2-on-U wiring amendment (~1; executed branch set is the U(P) walk sector,
  provably not the frozen V_Ram modes, ‖P_walk−P_Ram‖_F = 2.31);
  uniform-over-CSCO-eigenmodes identification (~1; NEW Jaynes application,
  not covered by theorem_multiway_branch_measure); "mass = Born weight at P"
  identification (~1; K3, new A5-class clause). NAMED K2-class residue
  (0 numeric bits, tracked as open lemma): the conjugate-aligned per-channel
  phase rule — load-bearing for Q = 2/3 (aligned read on ρ = I/8 also gives
  2/3); collapses iff the alignment lemma (Hermitian-positive C₃-circulant
  √M under R3) is derived. Tried-pattern: W3 dead by symmetry (referee-run,
  now counted; W1 already recorded). BANKED zero-cost: B(P) frame-free
  Born-incapacity (Gram defect 2/√3); canonical {U(P),C3} CSCO; weights
  (½,¼,¼) Pythagoras-forced; exact Brannen algebra incl. ε = √2 = tan(magic);
  positivity window |δ| ≤ π/12; Γ/H blind-side P-uniqueness corroboration.
  P2 ledger entry REDUCED (not deleted): magnitude half DERIVED conditional
  on priced adoptions; residue = alignment lemma + 2 identifications.

- 2026-06-11 (Phase 1 closure) — honest phase-hook outcome: the Phase-1 hook
  "A5(b) channel levels 19.0 → toward 0 if channels forced" did NOT fire:
  channels were re-expressed (9/12 exactly zeta-functional) but the address/
  level freedom was LOCALIZED, not removed (L=8 plural census; sector
  addresses ~2 bits; Koide n_s ~2.6 bits; quark ε² dressings unpriced-form).
  Zero-bit establishments: ε²_e = 2 = k\*−1 (Ramanujan), δ family one-number
  reduction. Earned-side note: row m_ν₂'s achieved width must widen to the
  JUNO residual (+2.92σ ⟹ achieved ≈ 2.3% on the mass, was 3% — no change
  at current rounding; recheck at JUNO full dataset). Net v0 number
  effectively unchanged; the value of Phase 1 to this ledger is the
  NAMING of the freedom, which is what a future forcing rule must delete.

- 2026-06-11 (Majorana panel) — fork-day accounting completed per panel
  correction 7: N-spillover anchor priced as a hypothesis (~2 bits: phase
  source selection among saddle-holonomy candidates) and REFUTED over a
  3→19-placement tried-pattern extension (all counted; 0 promoted). The
  P-reading consumes two previously unpriced conventions, now priced:
  +Im-chirality selection (~1 bit) and ω/ω² channel labeling (~1 bit).
  [2026-06-11 5.2-panel annotation on the ω/ω² bit: possibly the same
  freedom as the A5-mass row's P-sign up/down bit — identity-or-
  distinctness check ORDERED, open; if identical, single-home with
  cross-ref and re-print the A5-mass row at 2.0; both priced until
  proven (conservative).]
  [2026-06-11 ORDERED CHECK EXECUTED — IDENTITY ESTABLISHED (gated probe
  `phase5_2_psign_omega_identity_2026-06-11.py`, I1–I5): the P-sign
  partition of the 8 Ramanujan P modes IS the ω/ω² character partition
  (sign ⟺ irrep class ⟺ conjugate C₃ content, exact), the two families
  are mirror-paired with identical convention-free invariants, so the
  only datum distinguishing them is the character name. THIS LINE is the
  single home of that ~1 bit; the A5-mass row's P-sign component is
  cross-referenced here, NOT additive; A5-mass re-printed 3.0 → 2.0 per
  the panel-frozen sensitivity.]
  Banked at zero cost: the TRIM dichotomy, the saddle character tables,
  the C3-invariant Majorana structure theorem (J1–J4, panel-verified),
  the zero-diagonal π-invariance lemma.

- 2026-06-10 — v0 frozen (rules 1–5, prior classes, conservative direction).
- 2026-06-11 — Phase 1.3 panel amendment (ultracode adjudication, verdict
  PARTIAL): the saddle-mirror sector-assignment work INHERITS ~18 bits of
  already-priced conditionality (A5-mass labeling 15.3 + ADOPTED-NU-MAJ-PHASE
  3.0) [2026-06-11 5.2-panel propagation: A5-mass re-priced 15.3 → 3.0;
  inherited conditionality reads ~6 bits (A5-mass 3.0 + NU-MAJ-PHASE 3.0)
  from that date] and ADDS ~3–6 bits of address/orientation selections not
  previously priced: L=8 sector-class choice among 3 (~1.6),
  even-windings-only (~1), L=10 class among ~8 (≤3), ν_L/ν_R orientation
  convention (~1) [2026-06-11 5.2-panel annotation: this line is the
  SINGLE HOME of the A5-mass residual mirror Z₂ (= the joint ν_L/ν_R +
  spill swap = which of h_Γ/h_H is ν_L); cross-referenced from the
  A5-mass row; NOT additive — counted once, here]. These are
  spent bits of the Phase 1.3 CANDIDATE readings; they collapse to 0 only if
  a forcing rule is derived. Banked at zero cost (forced/exact): the matrix
  antiperiod, the zeta/sign-voltage tie, the saddle orbit map, the parity
  theorem, the tower counts (forced symmetry content).
