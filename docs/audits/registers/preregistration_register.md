# Preregistration register — frozen forward-looking predictions

**Purpose.** Timestamped, immutable record of the framework's falsifiable numerical
predictions for measurements that have not yet been made (or will be sharpened by
upcoming experiments). The evidential value of a match is highest when the predicted
value verifiably predates the measurement; this register is that record.

**Freeze protocol.**
1. Rows are append-only. A frozen value is NEVER edited in place; if a derivation
   changes, add a new row with its own freeze date and mark the old row SUPERSEDED
   (the old row remains — supersession history is part of the record, per the
   δ_CP^PMNS 249.85° → 180° precedent).
2. Each row carries: predicted value, derivation source, rigor grade at freeze time,
   the experiment that adjudicates it, and the kill condition.
3. External timestamp: each revision of this file is snapshotted to Zenodo
   (DOI per release). Git history is the internal timestamp.

**Status: DRAFT v0 (2026-06-10) + v1-candidate rows appended 2026-07-01.** Pending before v1:
- [x] **High-effort values-match audit — PASSED 2026-07-02 (formal per-row, 20/20 checks, 0
  problems).** Every frozen numeric row re-derived from the live `predictions/` DAG (values
  lock-identical to freeze commit 77a6392 per `predictions/_value_locks.json`, verified through
  HEAD): row 1 δ_CP = 180 exact; row 2 θ₂₃ = 48.720748; row 3 structural-exact (W45 kernel, no
  module); row 4 m_ν₂ = 8.859967 meV; row 5 m_ν₃ = 50.565058 meV; row 6 R_ν = 32.571429 = 228/7;
  row 7 α₂₁ = 162.38756; row 8 α₃₁ = 324.77512 (frozen convention; defect annotation stands); row 9
  m_ββ recomputed register-consistently = **1.54 meV (rows-as-frozen) / 3.56 meV
  (adoption-consistent) — BOTH inside the 1–5 meV window** (external freeze quotes the window, per
  Annotations §3); row 10 β = 0.35407°; row 11 η₅ = 0 exact; row 12 structural; row 13 w = −1 exact;
  row 14 H₀ = 68.178394 / 72.72362, ratio = 16/15 to 1e-9; row 15 b-values reproduce exactly from
  `the_run.py` read_gauge_running (+12/5→33/5, +4→1, +4→−3); row 16 θ_QCD = 0 exact; candidates:
  row 17 Σm_ν = 59.425 meV; row 18 V_cb = 0.040602696 = 256/6305 to 1e-9; row 19 η_lattice = 1/12
  to 1e-9. Audit script: session scratchpad `prereg_values_match_audit.py` (method: import each
  live module via `run_predictions` introspection; tolerances 1e-9 for exact rationals, ≤5e-4 for
  computed decimals). **v1 freeze is values-clean; external deposit unblocked.**
- [x] **Zenodo snapshot — DEPOSITED 2026-07-02: DOI 10.5281/zenodo.21124065** ("Pre-registered
  predictions of the srs-substrate framework — Freeze v1"). The external timestamp for rows 1–16 +
  candidates 17–19 (frozen honestly: m_ββ as the 1–5 meV window per Annotations §3; α₃₁ with its
  recorded convention defect; conditional rows flagged). Scoring per the freeze's §6 refinement rule:
  rows are scored against the most recent freeze PRE-dating the measurement.
- External-facing freeze document drafted 2026-07-01: internal research notes
  (private staging; publishes only by user action).
- Note: the 2026-07-01 R-9 ruling (srs = dominant waterline survivor, not uniquely forced) changes NO
  frozen value — all rows are srs reads regardless of the selection-claim level.

Grades are quoted as-is from the live DAG at freeze time. Preregistration timestamps
a value; it does NOT upgrade its rigor grade. CONDITIONAL rows are frozen as
conditional predictions.

---

## Frozen rows (v0, 2026-06-10, values read from live DAG at commit a63abbd)

| # | Quantity | Frozen prediction | Kill condition | Experiment (ETA) | Source | Grade at freeze |
|---|---|---|---|---|---|---|
| 1 | δ_CP^PMNS | **180.000° exactly** (arccos(−1), V_{−1}–T_{B-L} identity) | outside ~180° ± 30°, or clearly near-maximal ~270° | DUNE, Hyper-K (2028+) | `predictions/delta_CP_PMNS.py` | THEOREM-GRADE-STRUCTURAL (+ CKM↔K₄-walks adoption) |
| 2 | θ₂₃^PMNS | **48.7207°** (non-maximal, upper octant) | exactly maximal: 45.00° ± 0.3° | DUNE (2028) | `predictions/theta_23_PMNS.py` | STRICT-SOLID THEOREM-GRADE (Row P13) |
| 3 | m_ν₁ | **0 exactly** (W45 trivial-mode walker holonomy kernel) | 0νββ never observed AND m_ν₁ > 0 established | KATRIN, Project 8, nEXO (2027+) | `docs/theorems/` W45 (2026-05-20) | THEOREM-GRADE |
| 4 | m_ν₂ | **8.860 meV** (normal ordering) | outside oscillation-constrained band under NO | oscillation + cosmology (ongoing) | `predictions/m_nu2.py` | per live DAG |
| 5 | m_ν₃ | **50.565 meV** (normal ordering) | outside oscillation-constrained band under NO | oscillation + cosmology (ongoing) | `predictions/m_nu3.py` | per live DAG |
| 6 | R_ν = Δm²₃₁/Δm²₂₁ | **228/7 = 32.5714** (Ihara closed form, K₄) | precision value excludes 228/7 | JUNO-era precision (2026+) | `predictions/m_nu2.py` cross-check | UNIQUE-THEOREM-GRADE |
| 7 | α₂₁ (Majorana) | **162.388°** | outside ~162° ± 30° | nEXO, LEGEND-1000 (2030+) | `predictions/alpha_21_PMNS.py` | STRUCTURAL-DERIVATION-CONDITIONAL (ADOPTED-NU-MAJ-PHASE) |
| 8 | α₃₁ (Majorana) | **324.775°** | outside ~325° ± 30° | future 0νββ | `predictions/alpha_31_PMNS.py` | STRUCTURAL-DERIVATION-CONDITIONAL (ADOPTED-NU-MAJ-PHASE) |
| 9 | m_ββ (0νββ amplitude) | **≈ 2.55 meV** (m_ν₂ + α₂₁ chain) | measured outside ~1–5 meV | nEXO, LEGEND-1000 (2030+) | α₂₁/m_ν₂ chain (README falsification table) | inherits #4/#7 conditionals |
| 10 | β (cosmic birefringence) | **0.3541°**, with hard cap \|β\| ≤ α_EM ≈ 0.418° (c₁ = 0) | β measured > 0.418° (cap); β excludes 0.354° at precision | LiteBIRD (~2032, ~0.05°) | `predictions/beta_cosmic_birefringence.py` | THEOREM-GRADE-STRUCTURAL (Δα Clause-9 named gap) |
| 11 | η₅ (dim-5 LIV) | **0 exactly** (undirected-graph symmetry) | any dim-5 LIV detection | LHAASO, HESS (ongoing) | `predictions/eta_5_lorentz_dim5.py` | THEOREM-GRADE |
| 12 | WIMP dark matter | **none** (DM = gauge-decoupled uncompressed multiway; outside Cl(6) Fock) | WIMP direct detection | LZ, XENONnT (ongoing) | multi-axial dark-sector theorem (2026-05-24) | THEOREM-GRADE-STRUCTURAL |
| 13 | w_DE | **−1 exactly** | \|w + 1\| established ≳ 0.05 | DESI, Euclid (ongoing) | `predictions/w_DE.py` | UNIQUE-THEOREM-GRADE |
| 14 | H₀ two-clock structure | substrate **68.18**, observer **72.72** km/s/Mpc; ratio **16/15 exactly** | tension resolves to a single value incompatible with both; or ratio ≠ 16/15 at precision | CMB vs local-ladder programs (ongoing) | `predictions/H_0.py` | UNIQUE-THEOREM-GRADE (substrate side) |
| 15 | β-coefficient leg | α_i(M_Z) consistent with MSSM b = (33/5, 1, −3) under α_GUT⁻¹ = 24 inversion | α_s(M_Z) precision contradicts the inversion | LHC, FCC-hh (ongoing) | `derivation_topdown/bridge/the_run.py` `read_gauge_running` (top-down derivation of the exact b); `docs/theorems/theorem_beta_coefficients_derived.md` (PDG-inversion cross-check) | CONDITIONAL (R-19 DE-ESCALATED 2026-06-23 — β values derived top-down/exact; forced-ness = ζ_{D₄}(0), research-level) |
| 16 | θ_QCD | **0 exactly** (srs Z₃ holonomy flatness) | nonzero θ_QCD established | nEDM programs (ongoing) | `predictions/theta_QCD.py` | UNIQUE-THEOREM-GRADE |

## v1-CANDIDATE rows (appended 2026-07-01; pending the values-match audit — NOT yet frozen)

| # | Quantity | Candidate prediction | Kill condition | Experiment (ETA) | Source | Grade at append |
|---|---|---|---|---|---|---|
| 17 | Σm_ν + mass ordering | **Σm_ν = 59.4 meV** (= 0 + 8.860 + 50.565, rows 3/4/5), **normal ordering** (m₃ > m₂ > m₁ = 0) | robust cosmological bound Σm_ν < ~55 meV; or inverted ordering established | DESI/CMB joint (ongoing — actively tightening), JUNO ordering (2026+) | arithmetic on frozen rows 3/4/5 | inherits rows 3/4/5 (the absolute scale carries the framework's own +~2σ open residual vs NuFIT-derived values — flagged honestly) |
| 18 | V_cb (excl/incl adjudication) | **256/6305 = 0.040603** — the framework sides with the EXCLUSIVE camp in the live ~3.3σ exclusive/inclusive data self-tension | the dispute resolves decisively at the inclusive value (≈ 0.0422), excluding 256/6305 at >3σ in the resolved world average | Belle II + LHCb + lattice (ongoing) | `predictions/V_cb.py` (A2 waterline geometric series) | UNIQUE-THEOREM-GRADE-CONDITIONAL |
| 19 | η_lattice (dim-6 LIV) | **1/12** (Hashimoto dispersion; CAS-verified) | dim-6 LIV measured incompatible with the 1/12 coefficient pattern | LHAASO, CTA (ongoing) | `predictions/eta_lattice_lorentz_dim6.py` | THEOREM-GRADE (CAS) |

## Historical record (falsifications that fired)

| Quantity | Retired prediction | Fired | Replacement |
|---|---|---|---|
| δ_CP^PMNS | 249.85° (Hashimoto-phase route) | +3.83σ vs NuFIT 6.0 IC19, 2026-05-02 | row 1 above (independent V_{−1}–T_{B-L} identity, 2026-05-05); retired route preserved at `predictions/retracted/delta_CP_PMNS.py` |

## Maintenance

- Owner: Phase 0.2 of the unification program.
- Re-audit cadence: on every Zenodo release; on any change to a source derivation
  (which triggers a SUPERSEDED row, never an edit).
- Related: `README.md` §"Falsification criteria", `docs/honest_assessment.md`
  §"What would falsify the framework" (those describe; this file freezes).

## Annotations (append-only)

### 2026-06-11 — Rows 7/8/9 (Majorana sector): panel adjudication annotation

Adversarial panel (7 refuters + judge) on the Phase 1.3 Majorana fork
(internal research notes). Recorded BEFORE any 0νββ
measurement; kill conditions stand unchanged; no supersession (no derivation landed).

1. **FORK STATUS.** Rows 7/8 are the P-reading branch of a documented fork.
Adjudicated: (i) P-reading stands ONLY as ADOPTED-NU-MAJ-PHASE (grade
unchanged); the class-diagonal M_R at P is C3-ALLOWED under the
mirror-crossing invariance law (−P = P+Δ, conjugated characters; the
2026-06-11 C3-invariance strike against it is withdrawn as mis-aimed — the
same-fiber law holds at TRIM saddles only). (ii) The H-reading endpoint
α₂₁ = α₃₁ = 0 is REFUTED: M_R ∝ 1 at the TRIM H saddle is C3-forbidden; the
C3-invariant H completion gives relative phase π (collapses into the
C3-invariant branch). (iii) The C3-invariant branch stands as a TRIM-branch
structure theorem: exact C3 forces phase π with |m₂| = |m₃|, contradicted by
R_ν = 228/7 → physical phase = π − δ_breaking, δ underived; the minimal
N-spillover anchor (17.612°) is REFUTED over 19 breaking placements
(0 passes; Takagi-correct invariants).

2. **ROW 8 CONSISTENCY DEFECT** (values-match audit, anticipated in the v0
pending list). Under the cited adoption's own form
M_R = |M_R|·diag(1, h_ω^g, h_ω²^g) with row 3's ordering (m₁ = trivial
channel), α₃₁ = arg(h_ω²^10) = 197.612°. The frozen 324.775° = 2g·arg(h_ω) =
(φ_ω − φ_ω²) mod 360 — a φ₂−φ₃ quantity recorded as φ₃−φ₁
(predictions/alpha_31_PMNS.py computes p_toggle·g·arg(h), not the adoption's
per-channel h_m^g; the in-repo m_ββ chain uses 197.612°). With m₁ = 0 only
one Majorana phase combination is physical: |α₃₁ − α₂₁| = 35.225°
adoption-consistent — rows 7/8 jointly over-specify. Row 8 stands as frozen
per protocol; any future match must be scored against this recorded defect.

3. **ROW 9 VALUES-MATCH AUDIT FAILURE.** The frozen ≈2.55 meV reproduces only
via proofs/flavor/srs_unified_mixing.py §8, which uses δ_CP = 249.851° (the
RETIRED route — this register's own historical-falsification row),
α₃₁ = 197.612° (contradicting row 8 as frozen), and observed NuFIT masses
(not rows 4/5). Register-consistent recomputation from frozen rows 1/4/5/7/8
gives ≈1.5 meV.

4. **FORK-INSENSITIVITY OF THE ROW-9 WINDOW.** All adjudicated branch
endpoints lie INSIDE the frozen 1–5 meV window: rows-as-frozen P ≈ 1.5 meV;
C3-branch π ≈ 1.4 meV; adoption-consistent P ≈ 3.7 meV; H-constructive
≈ 3.9 meV (refuted branch). A 2030 m_ββ inside 1–5 meV is NOT an
unconditional row-7-chain hit and must be scored against this fork; the
surviving fork (P-adoption vs π−δ) is experimentally degenerate at
nEXO/LEGEND precision under the rows-as-frozen convention (Δ ≲ 0.1 meV).
The v1 gate (values-match audit) remains BLOCKED until the row-8/row-9
source inconsistency is resolved by derivation or new frozen rows.

### 2026-06-11 — First blind scoring event: JUNO first oscillation data (Phase 1.4)

JUNO's first reactor-oscillation measurement (59.1 days, arXiv:2511.14593 /
Nature s41586-026-10538-z; world-leading 1.55% on the solar splitting):
sin²θ₁₂ = 0.3092 ± 0.0087; Δm²₂₁ = (7.50 ± 0.12)×10⁻⁵ eV² (NO). Scored
against the rows AS FROZEN 2026-06-10 (genuinely blind: rows predate the
scorer's knowledge of the JUNO numbers):

- **Row 4 (m_ν₂ = 8.860 meV ⟹ Δm²₂₁ = 7.850×10⁻⁵ eV²): +2.92σ TENSION**
  (JUNO-implied m_ν₂ = 8.660 ± 0.069 meV). Kill condition not fired, but
  honesty note: this row was already +1.9σ vs the KamLAND-era value at
  freeze time; JUNO sharpened the same deviation. JUNO's full dataset
  (sub-percent) will adjudicate decisively. STATUS: WATCH.
- **Row 5 (m_ν₃ = 50.565 meV) and Row 6 (R_ν = 228/7 = 32.5714):** await a
  same-precision Δm²₃₁ (JUNO's first paper profiles it; not yet
  world-leading). Provisional with global-fit Δm²₃₁: row 6 within ~1–2σ,
  precision-limited. STATUS: PENDING JUNO Δm²₃₁.
- Informational (not a frozen row): framework θ₁₂ = 33.07° ⟹ sin²θ₁₂ =
  0.2977 → −1.32σ vs JUNO. Consistent.

This is the register operating as designed: a pre-frozen value met an
unknown measurement and the score is recorded regardless of direction.

- **Row 15 (2026-06-11 — Phase-4 spectral-action jeopardy annotation; spec
  frozen at b4bb97b pre-computation; probe
  `phase4_3_beta_content_jeopardy_2026-06-11.py`; panel-adjudicated
  2026-06-12).** Among the three pre-registered targets, the
  action-determinate content lands on 2HDM-SM at the b₃ discriminator:
  b₃ = −11 + (2/3)·6 + 0 = −7, zero new continuous parameters (K3 silent).
  CONDITIONALITY: the exact −7 is anchor-conditional on the declared
  dictionary-licensed 3-generation sector anchor (native per-fiber trace 8
  would give −17/3, matching NO target; the target discriminator (+4/+2/0)
  is anchor-free; anchor priced 1 bit, ledger 2026-06-12) and on the
  textbook one-loop formula; the −11 gauge weight is textbook (native
  support candidate-grade, banked 05-28). MSSM EXCLUSION, scoped WITHIN
  THE FROZEN TRIPLE, three-way split of Δb₂ = +4: wino +4/3 STRUCTURALLY
  EXCLUDED (C₂(su4) = 15/4·I exact; H is a pure spinor module 4+4̄ with no
  adjoint fermion seat); higgsino +2/3 DICTIONARY-CONDITIONAL (two
  color-singlet fermion slots per fiber exist, anchor-assigned to leptons;
  independent ths-side exclusion stands); sfermion +2 = the σ-coupler
  freedom (the frozen algebra generates NO mirror-crossing one-forms —
  σ-couplers are external posits the action propagates but does not
  induce; the census-blindness claim is WITHDRAWN pending alias-free
  recomputation, erratum E2 — executed 2026-06-12, census
  block-discriminating). Adjudicated on the b₃ column alone (b₂
  dictionary-conditional, b₁ unclaimed). Grade remains CONDITIONAL;
  ADOPTED-MSSM-Sb stays; kill condition unchanged.
