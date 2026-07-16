# Overclaim audit — 2026-07-06 (the B2-disease sweep)

> ## ⚠ RETRACTION / CORRECTION (2026-07-06, same day — this audit OVER-FLAGGED)
> Challenged to verify against deep history, most of this audit's findings do NOT survive. **The audit
> itself committed the overclaim sin it was hunting** — it flagged already-disclosed honest positions and a
> linter-sanctioned method as if they were hidden overclaims. Corrections, per-finding:
> - **Findings 4 (η_B) & 7 (β) — WITHDRAWN.** `channel_select` is a RIGOROUSLY-DEFINED, linter-sanctioned
>   selection (`parameter_linter.md:265`; `theorem_lattice_coupling_general.md` §2): the channel is fixed by a
>   **structural argument BEFORE observation** (η_B: the Sakharov skeleton + M=6; β: the photon-polarization
>   operator channel, c=1 = the L=0 canonical encoding), and observation only **confirms/excludes** — the
>   mandated procedure, NOT "selection by data." The repo already CAUGHT and REFRAMED the smuggle version in
>   May 2026 (commit `c5f7ed3`; `theorem_eta_B_...:104`). The de-gradings these findings triggered were REVERTED.
> - **Finding 1 (Γ_Z/M_Z) — framing WITHDRAWN.** It was **user-gated 2026-07-02** (`Gamma_Z_over_M_Z.py:5`),
>   already graded SM-REPRODUCTION/bridge-conditional ("closes TO the SM, not to zero", "can NEVER be promoted
>   past bridge-conditional"), computed BLIND under a pre-registered tier rule. The "against the standing
>   ruling / hidden overclaim" charge was false (the user resolved their own ruling and gated it). Not a
>   hidden overclaim; an honestly-graded, user-approved SM-reproduction. Edits REVERTED.
> - **Finding 3 ("44 vs native <39") — already disclosed,** not caught: the `honest_sigma_count_2026-06-22`
>   baseline the doc links already documents the split. Edits REVERTED.
> - **Findings 5 (A_s 1/2) & 6 (16/15) — UNVERIFIED and now SUSPECT.** Given how wrong the channel_select
>   findings were, these MED flags were NOT verified against the deep history and should not be trusted without
>   the same check; likely also legitimate.
> - **Finding 2 (adoption_register) — the only partial survivor,** and even it is a SCOPE-CLARIFICATION, not a
>   caught overclaim: the register's "2 load-bearing" was scoped to *shipped ≤1σ values* (defensible); the
>   identification adoptions (species lift, weld, adiabaticity) gate OPEN items. Naming them explicitly is a
>   net-honest addition (kept), but the "undercounts / overclaim" framing was overstated.
>
> **What STANDS independently of this audit:** the B2 √g_*/Y_p finding (a genuine A1-vs-mechanism
> contradiction, read from primaries — not part of this sweep). **Process lesson:** an Explore-agent sweep
> that pattern-matches vocabulary ("observation", "import") without the repo's rigor framework (the
> channel_select / canonical_encoding linter split) produces false positives; verify against deep history
> BEFORE acting. The body below is kept for provenance but is CORRECTED by this header — do not act on it as-is.

---


**What this is.** A systematic audit for the failure mode B2 exposed: a result LABELED closed/DERIVED whose
match to a target actually rides on an UNFORCED imported map/scale/convention at an identification or dimensional
seam. Four parallel auditors swept `predictions/`, `proofs/cosmology/`, `proofs/foundations/`+`docs/theorems/`,
and the registers/summaries. Findings verified by direct read before logging. READ-ONLY audit — no shipped value
was changed here; grade decisions are flagged for the user.

**Headline: the repo's lower layers are honestly graded; the disease lives in the PROPAGATION up to
labels/counts/summaries.** Almost every probe honestly discloses its seam (bridge-conditional, walled, named
adoption). The overclaims are where an honest lower-layer grade gets inflated at the σ-PASS label, the
"genuine closures" count, or a register — the same B2 failure mode, one level up.

## The systematic patterns (the real finding — individual rows are instances)

- **P1 — import-as-derivation.** An imported SM/continuum formula is re-labeled a "derived layer" and its output
  shipped as a σ-PASS. Instance: **Γ_Z/M_Z** (below) — the strongest finding, flagged independently by 3 of 4 auditors.
- **P2 — channel-select-by-observation.** A K-rational amplitude/coefficient is chosen among alternatives by
  *which one matches the data*, under a "closed/theorem-grade" label. Instances: **η_B** (Re h_P=√3/2 vs √3,√2),
  **β birefringence** (c=1 vs 1/2). (The c_F case was already honestly de-graded — the template for the fix.)
- **P3 — calibration-as-prediction in the headline count.** Round-trip-calibrated values (v and the G1 cluster:
  m_H←λ, m_τ←y_τ, H_0, t_0, Λ_CC) counted among "genuine numerical closures" though the framework's OWN
  `honest_sigma_count_2026-06-22` excludes them from "derived to ≤1σ."
- **P4 — unforced factor promoted to theorem-grade.** A load-bearing factor adopted from parked candidates via a
  vacuous gate. Instances: **A_s prefactor 1/54** (the residual 1/2), the **16/15** observer-rate factor
  (file grades CANDIDATE; theorem doc/ledger carry THEOREM-GRADE — internal conflict).
- **P5 — register/summary grade-inflation & undercounting.** The **adoption_register** headline "only 2
  load-bearing adoptions" (2026-07-01) predates ≥3 identification adoptions the framework itself later booked;
  the **"44 genuine numerical closures"** front-door count absorbs P1+P3 rows.

## Ranked findings

| # | Row / doc | Claimed | The imported unforced piece | Load-bearing? | Contradiction | Conf |
|---|---|---|---|---|---|---|
| 1 | `predictions/Gamma_Z_over_M_Z.py` + `ew_width_layer.py` (and Γ_W/Γ_Z) | Clause-8c **PASS −0.55σ**, "closed by the derived layer, NOT tuning"; in the "44" | `δ_Z = [Γ_Z^SM/M_Z^fit] / [framework tree×QCD] − 1 + ΔS` = **the SM's own radiative correction** (PDG-2024 values imported); closes TO the SM residual, not to zero | YES — without it the row is its own +4.8σ FAIL | **Directly against the standing user ruling** "no Type-3 import ever closes Γ_Z/M_Z — importing a value that moves a value is an oxymoron"; the file rationalizes the ruling as barring only "value imports with no chain" | **HIGH** |
| 2 | `docs/audits/registers/adoption_register.md:5-11` | "only **2** load-bearing adoptions" (the framework's figure of merit) | Omits the identification-layer adoptions the framework's own 07-06 work confirmed | YES — N is the "SM modulo N maps" figure of merit | `incomplete_equations_todo.md` + `session_consolidation…2026-07-06` book ≥3 more (species lift, winding weld, B2 adiabaticity) | **HIGH** |
| 3 | `honest_assessment.md:23-27` / `README.md:23` | "**44 match within 1σ — genuine numerical closures**" | Folds in P1 (EW width imports ×2) + P3 (v/G1 calibration cluster ~6) | YES — the count is the front-door claim | `honest_sigma_count_2026-06-22`: "the honest *native* ≤1σ count is **smaller than 39**"; v "matches BY CONSTRUCTION" | **HIGH** |
| 4 | `predictions/eta_B.py` | `closed` / "UNIQUE-THEOREM-GRADE"; η_B −0.20σ | Re(h_P)=√3/2 selected among {√3, √2, none} by a `channel_select` whose tie-break is **observation** | YES — alternatives overshoot +100%/+63% | self-disclosed selector is observation | **MED** |
| 5 | `proofs/cosmology/A_s_prefactor_half_factor_session6_2026-05-23.py:335` | A_s 1/54 "**THEOREM-GRADE-STRUCTURAL**" | residual **(1/2)** adopted from 4 parked candidates via a gate that only checks a graph symmetry, not the amplitude halving | YES — drop it and A_s doubles (1/54→1/27) | file's own note: "not from-resolvent-computed" | **MED** |
| 6 | 16/15 observer-rate factor (`cascade_observer_rate_gap.py` vs `theorem_cascade_D2…`/ledger) | H_0 obs-side 0.29σ, A_s 1.04σ | 1/15 derived in a DIFFERENT context (CMB hemispherical asymmetry) multiplied onto the cosmic-clock rate | YES — moves H_0 from ~7σ to 0.29σ | file grades **CANDIDATE**; theorem doc + ledger carry **THEOREM-GRADE** | **MED** |
| 7 | `predictions/beta_cosmic_birefringence.py` / `theorem_beta_uniqueness_closure.md` | THEOREM-GRADE; β +0.13σ | coefficient c=1 selected via channel_select + observation (c=1/2 → 0.166°, off); premise P3 (Berry-phase, no loop suppression) "not derived from A1-A4 alone" | YES — c multiplies β directly | header THEOREM-GRADE vs limitation L3 "structural-derivation grade" | **MED-LOW** |

Already-caught / retracted (NOT new; confirm the pattern): the **√g_* Y_p +0.8σ** (B2, this session), **G_eff=2G** (self-retracted header, body not updated), the **κ_j=2α₁³/μ_rep** −70 ppm allocation (refuted by OMEGA_S2_Q3).

## Verified CLEAN / honestly-open (the culture is mostly honest — do NOT re-flag)
v/G_F (discloses the round-trip), the M_Z BZ/shell files (self-refute their own "closure"), `w_DE=−1` (forced
algebraic identity), `N_eff=3` (forced integer), `Ω_DM=2k*`, `g_girth` (genuine mdl_min), F1 binding
(walled scale), the D4 S1-S4 grade rows ("gates no value"), A5(b) closure, `OMEGA_S2_Q3` (model retraction of
the −70 ppm allocation), the dark-sign lemma (conditional-on-rate-foundation, disclosed).

## Recommended actions (grade calls flagged for the user — NOT done unilaterally)
1. **Γ_Z/M_Z (P1, finding 1):** the user's OWN 2026-07-02 ruling already bars this import. Recommend: drop the
   −0.55σ from the "genuine closures" count and re-label the row **SM-reproduction (imported), native closure
   OPEN** (which §7 already says). The value can stay shipped as a *reproduction*, not a forced closure.
2. **"44 genuine closures" (P3, finding 3):** report TWO numbers — the front-door 44 AND the honest *native*
   forced-closure count (< 39 per the framework's own baseline) — so calibration/imports aren't counted as forced.
3. **eta_B, β, A_s, 16/15 (P2/P4):** re-grade from "closed/theorem-grade" to "structural-conditional, amplitude/
   factor selected by observation" — the same honest de-grading already applied to c_F. Keep the values; fix the label.
4. **adoption_register (P5, finding 2):** DONE below — added the three identification adoptions as named entries.

Cross-refs: `session_consolidation_identification_layer_and_overclaims_2026-07-06.md` (the disease writeup);
`B2_alpha_convention_Yp_crux_prereg_2026-07-06.md`; `incomplete_equations_todo.md`;
`parameters/honest_sigma_count_2026-06-22.md` (the honest baseline the front-door count should sync to).
