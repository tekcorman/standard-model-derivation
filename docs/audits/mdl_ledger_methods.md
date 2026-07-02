# The MDL Ledger — accounting methods (fixed before counting)

**Status:** METHODS v1, committed 2026-07-02 — **frozen BEFORE any count was computed.** The counting
script and results land in separate, later commits; this document's git hash proves the conventions
were not tuned to flatter the outcome. Any later convention change is a versioned supersession of
this file, never an edit (the preregistration discipline, applied to the ledger itself).

**Purpose.** The framework's standing claim is *zero fitted constants*: a short discrete
specification, plus one calibrated scale, reproduces the Standard Model's measured parameter table.
This ledger makes that claim quantitative in the framework's own native currency — description
length. It answers the numerology objection with a number: **how many bits does the specification
cost, versus how many bits of measured data it explains — after paying honestly for every trial,
every adoption, and every open miss.**

---

## §1 The formal claim being tested

Two-part-code comparison (Rissanen). Encode the SM's measured parameter table two ways:

- **Framework code:** L_spec (the specification: choice-points, adoptions, calibrations, trials)
  + L_resid (the cost of encoding each measured value *given* the framework's prediction).
- **Baseline code (SM-as-fit):** the SM treats each parameter as a measured input; its cost is the
  full information content of every parameter at experimental precision, ~26 entries (19 classic +
  the neutrino-sector extension).

Shared machinery (the QFT formalism, spacetime, the gauge-theory *language*) appears in both codes
and cancels; the comparison is strictly about the **parameters**. The framework's claim survives iff

  L_spec + L_trials + L_resid  <<  L_SM-baseline,

with a margin large enough that no plausible re-pricing of the fuzzy items reverses it.

## §2 Data-side conventions (bits explained per observable)

**Per-row formula.** For observable i with prior width W_prior, measurement uncertainty σ_i, and
framework residual Δ_i = |predicted − measured|:

  b_i = log₂( W_prior / max(2σ_i, 2Δ_i) )

- A hit within 1σ earns the full measurement information (you cannot claim more resolution than the
  experiment has). A miss still earns its genuine compression (M_Z at 180 ppm compresses enormously
  relative to an O(1) prior) but pays the residual penalty automatically — **the open misses cost
  bits by construction; they are never relabeled away.** (Top-down law compliant: the miss stays
  open and stays priced.)
- Exact structural hits with bounded measurements (θ_QCD = 0 vs < 10⁻¹⁰) earn
  log₂(W_prior / bound-width).

**Priors, fixed per class — always the CONSERVATIVE (narrow) choice; every judgment call goes
against the framework:**

| class | prior | note |
|---|---|---|
| mixing-matrix moduli, sin², fractions | uniform on [0, 1] | natural compact domain |
| CP phases, Majorana phases | uniform on [0°, 360°) | |
| dimensionless couplings (α_GUT, y_f, λ) | log-uniform on [10⁻⁶, 4π] | perturbativity cap |
| mass ratios / masses in units of v | log-uniform on [10⁻¹², 1] ∪ measured-family span — use the NARROWER defensible span per row | narrow = fewer claimed bits |
| mass-squared ratios (R_ν) | log-uniform on [1, 10⁴] | |
| cosmological fractions (Ω) | uniform on [0, 1] | |
| dimensionful cosmology (H₀, t₀, Λ) | log-uniform over ±3 decades around measured | conservative |

**Inclusion rules (the conservative headline ledger):**
1. **Round-trips excluded**: v (G_F-calibration artifact), G_F itself, and anything whose 0σ is
   by-construction.
2. **Conditional rows**: included ONLY if their adoption is priced into L_spec (§3); otherwise
   excluded. (DARK-MAP-conditional θ₁₃/β: adoption priced, rows included. z_eff-conditional
   Ω_DM/Ω_b: excluded from the headline.)
3. **Category-B coasting rows excluded from the headline** (their comparison set is framework-side
   by construction); reported in a clearly-labeled expanded variant only.
4. **Data-tension rows** (V_ts/V_tb riding the V_cb dispute): excluded from the headline.
5. **Degrees-of-freedom guard**: unitarity-derived rows earn no independent bits — the CKM sector
   earns at most its 4 dof (V_us, V_cb, V_ub, δ_CP), PMNS its 4 (+2 Majorana only if E4 closes),
   the ν masses their 2 (+ordering as 1 bit).
6. **Structural integers** (k*, d, N_gen = 3, |V|, |E|): counted only once, as N_gen = 3 and d = 3
   vs their observed values (2 rows, small bits each); the rest are spec-internal.

## §3 Spec-side conventions (bits paid)

Reported in **two columns**; the claim must survive BOTH:

- **Column A — choice-point ledger (the framework's own accounting).** Information = selections
  among operator-permitted alternatives. Derived items cost 0; every discrete CHOICE costs
  log₂|alternative set|, with alternative sets sourced from the uniqueness ledgers (25 structural
  rows; 68 P-rows), the adoption register, and the R-9 record. Priced items include, at minimum:
  the substrate selection post-R-9 (**log₂ 4 = 2 bits** — the waterline survivor set {srs, srs-c8,
  lou, lov}; the honest cost the 2026-07-01 ruling created), each ACTIVE adoption (NU-MAJ-PHASE:
  log₂ of its documented fork-branch count; DARK-MAP: its classifier scope; A5b-Sub3: the
  sub-class assignment content; B3: the residual sector-label bit), and each reading convention not
  yet theorem-closed.
- **Column B — hostile ceiling.** The compressed length (gzip) of the minimal formal statement of
  the axioms + object definition + dictionary, as an upper bound on "the whole framework text is
  the spec." Reported without spin; the margin must survive it.

**Continuous inputs.** N_hub is calibrated from measured G_F: it costs the consumed measurement
bits at the precision actually used (≈ log₂(1/relative-precision) ≈ 20 bits for ppm-class). The
Planck unit identification carries zero dimensionless content (type-III₁ scale-freeness) in Column
A; in Column B it is priced as one consumed measurement.

**L_trials — the look-elsewhere payment.** The framework can pay its trials cost *from receipts*:
the append-only registers, `predictions/retracted/` (15 versions), `explorations/negative_results.md`
(16 killed hypotheses), the retraction log, and each P-row's documented alternative-formula audit.
Rule: per observable family, charge log₂(1 + N_documented_candidates). Where documentation is
absent, charge a default 3 bits/observable (8 assumed candidates) — again against ourselves. State
plainly: the registers BOUND the true trials count only to the extent the record is complete; the
default term is the hedge.

## §4 The baseline

L_SM-baseline = Σ over the ~26 measured SM(+ν) parameters of log₂(W_prior / 2σ_i), using the SAME
priors table (§2) — identical conventions on both sides. No trials charge to the SM (it is a fit by
construction; that is the point of the comparison).

## §5 Protocol

1. This methods file commits FIRST (this commit).
2. `scripts/mdl_ledger.py` implements §2–§4 mechanically off the live DAG (the same introspection
   as the value-lock harness), so the data side is machine-checkable and re-runs with every lock.
3. Results land in `docs/audits/mdl_ledger.md` in a later commit, quoting this file's hash.
4. Judgment calls discovered during counting that §2–§3 do not already fix are resolved
   CONSERVATIVELY (fewer explained bits / more spec bits), logged in the results file, and folded
   into a methods v2 only by supersession.
5. The headline number is the **margin under Column B with all exclusions applied** — the worst
   honest case. Friendlier variants (Column A, expanded row set) are reported alongside, labeled.

## §6 What this ledger is NOT

- Not a rigor claim: a bit-cheap formula can still be wrong; grades live in the registers.
- Not a substitute for the open-equations ledger: the M_Z and charged-lepton misses appear here as
  residual costs, and remain OPEN there.
- Not a Bayes factor: no likelihoods over theories are claimed — only description lengths under
  stated conventions, all of which are printed.
