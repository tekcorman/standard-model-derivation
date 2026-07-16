# The MDL Ledger — results (v1, 2026-07-02)

**Methods:** frozen BEFORE counting in [`mdl_ledger_methods.md`](mdl_ledger_methods.md) (commit
`c474e27`). **Engine:** [`scripts/mdl_ledger.py`](../../scripts/mdl_ledger.py) — computes the data
side mechanically off the live `predictions/` DAG (the same introspection as the value-lock
harness), with explicit, auditable manifests for rows, exclusions, spec choice-points, and the SM
baseline. Re-run anytime: `python3 scripts/mdl_ledger.py`.

**The question this answers, quantitatively:** *is the framework a compression of the Standard
Model's measured parameter table, or a fit dressed up?* In MDL terms: does the specification (every
discrete choice, every adoption, the one calibration, plus an honest look-elsewhere charge for every
documented dead-end) cost fewer bits than the data it explains?

---

## Headline (Column A — the two-part-code comparison)

| quantity | bits |
|---|---:|
| **Specification** (all choice-points: srs-among-survivors 2.0, adoptions 14.5 + species-lift residual 1.6 [booked 2026-07-13, user-ruled conservative self-over-charge; `working notes/adoptions_bucket_audit_2026-07-13.md`], dark-sign reading 1.6, n_fixed=2 reading 1.6 [booked 2026-07-14, user-ruled PRICE IT — same class as the dark-sign reading; no forcing theorem found for the girth-10 two-struts-held-fixed convention], N_hub↤G_F calibration 20.0) | **41.3** |
| **Trials** (look-elsewhere: log₂(1+N) per family at the 8-candidate floor; 167 receipts documented ≈ 6/family) | **98.3** |
| **Total paid** | **139.6** |
| **Data explained** — 27 SM-parameter rows (conservative priors; misses Δ-priced) | **304.3** |
| **Surplus** — 5 rows the SM's parameters cannot encode at all (η_B, A_hemis, Ω_DM/Ω_m, β birefringence, N_gen) | **+27.9** |
| **SM-as-fit baseline** — the same table as 25 measured inputs, identical priors | 327.8 |

**Margin: +164.8 bits (164.7503 unrounded). Compression ratio 2.18×.** The framework encodes the table the SM buys for
327.8 bits of measurement for 139.6 bits of specification-plus-search, and additionally explains
27.9 bits of data outside the SM's parameter set. To attribute this to formula-shopping, the true
search would need to have been ~2¹⁶⁵ times larger than the documented record.

## The rows (predicted from the live DAG; measured per PDG 2024 / NuFIT 6.0 / Planck 2018)

Highlights of the full table (printed by the engine):

- **θ_QCD = 0 is the single most informative row: 35.9 bits** — the strong-CP problem, quantified;
  one flat-holonomy theorem outweighs any mass.
- **The open misses pay, by construction** (top-down law): m_e earns 17.3 bits instead of the ~35 a
  perfect hit would earn — the −70.3 ppm open miss visibly costs ~18 bits; likewise m_μ, M_Z
  (+7.76σ), m_W, α_EM, m_ν₃. Six rows carry MISS-priced penalties; nothing is relabeled.
- Exact structural hits (R_ν = 228/7, V_us = 9/40, V_cb = 256/6305, δ_CP identities) earn their full
  measurement information — they cannot be "close," only right or dead.

**Exclusions (all against the framework):** v/G_F (calibration round-trip); g₁/g₂/g₃, λ_H, δρ, δ_r,
α_GUT, M_unif (dof re-parametrizations); unitarity-derived CKM entries and the V_cb-tension rows;
Koide re-parametrizations; m_ν₂ (derived from R_ν + m_ν₃); ALL Category-B coasting cosmology;
z_eff-conditional Ω rows; all unmeasured freeze rows.

## Sensitivity (does the margin survive hostile re-pricing?)

| stress | margin |
|---|---:|
| methods default (8-candidate trials floor) | **+164.8** |
| trials floor 16 candidates/family | +136.3 |
| trials floor 64 candidates/family | +76.3 |
| N_hub priced double (40 bits) | +144.8 |
| drop θ_QCD entirely (the largest row) | +128.9 |
| **all four stresses simultaneously** | **+20.4** |

**Break-even requires ~357 secretly-tried candidate formulas per observable** — against append-only
registers documenting ~6 per family. The margin is not an artifact of any single row, any single
convention, or any plausible under-count of the search.

## Column B (the hostile prose ceiling) — reported without spin

Gzip of the framework's minimal formal statement: **8,848 bits**; symmetric gzip of a minimal SM
statement (which must embed its ~26 irreducible decimal parameters): **5,344 bits**. The framework
**loses the prose comparison by ~3,500 bits.** Interpretation: gzip-of-prose measures
*mechanism-description length*, and any generative theory loses that metric to a parameter list —
Newton's Principia gzips worse than a table of planetary positions. The framework trades 26
irreducible decimals for a mechanism; mechanisms cost prose. The MDL-relevant measure is the
two-part code (Column A). Both numbers are printed so the reader makes the call.
*(Logged judgment call, methods §5.4: v1 methods defined Column B without a symmetric SM-side gzip;
the symmetric baseline was added during counting and is a supersession candidate for methods v2.
Gzip figures are style-sensitive at the ±30% level; the sign of the prose gap is robust.)*

## Caveats (methods §6, restated)

1. **Not a rigor claim.** A bit-cheap formula can still be wrong; grades live in the registers, and
   the open equations stay open (`incomplete_equations_todo.md`).
2. **Not a Bayes factor.** Description lengths under stated conventions, all printed, all frozen
   before counting.
3. **The trials charge is bounded by the record.** The registers are append-only and comprehensive
   by design, but the charge is only as complete as the receipts; the sensitivity table prices that
   doubt, out to break-even.
4. **Conditional rows** (θ₁₃, β via DARK-MAP; the Majorana sector is excluded entirely as
   unmeasured) are included only with their adoptions priced in the specification.

## Relation to the pre-registration freeze

The freeze (DOI 10.5281/zenodo.21124065) proves the predictions *pre-date* the future data; this
ledger proves the specification is *cheaper than* the existing data, after paying for the search.
Together: the numbers were not fitted to what is measured, and cannot be retro-fitted to what will
be. Future dynamical results (widths, binding energies) add data-side rows at near-zero added
specification cost — the ledger's margin is designed to grow.
