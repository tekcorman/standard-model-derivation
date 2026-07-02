# W26 Verdict — Candidate B (Higgs VEV direction) HONEST NEGATIVE

**Date:** 2026-05-26 EOD+1 + 1
**Status:** Candidate B closed at session 1. The substrate Higgs VEV direction does NOT pick a canonical 1-dim sub-direction of V_scalar.

---

## Empirical findings (W26)

For each of 4 orientation conventions on K_4:

| Convention | Higgs VEV direction at edge {u,v} (u<v) | rank(P_V_scalar · HVM) | C_3 isotypic |
|---|---|---|---|
| ORIENT A | (v→u) | 2 (= all of V_scalar) | (0, 2-pair) |
| ORIENT B | (u→v) | 2 (= all of V_scalar) | (0, 2-pair) |
| ORIENT C | symmetric ((u→v)+(v→u))/√2 = J=+1 | 2 (full V_scalar) | (0, 2-pair) |
| ORIENT D | antisymmetric = J=-1 | 0 (orthogonal) | — |

**ORIENT A vs ORIENT B alignment** in V_scalar: |⟨v_A | v_B⟩| = 0.0000 (orthogonal).

The two specific-orientation choices A, B give DIFFERENT 2-dim sub-spaces in V_scalar, together spanning all of V_scalar. Neither picks a canonical 1-dim direction.

The fully-symmetric (J=+1) Higgs VEV mode contains V_scalar entirely (since both are J=+1). The fully-antisymmetric (J=-1) mode is orthogonal to V_scalar (J=±1 are orthogonal sub-sectors).

## Structural conclusion

The "+1 mode" canonical pick within V_scalar for c_EW = 4/12 = 1/3 is **NOT** given by:
1. Graph-level operators (W21: V_scalar is C_3-faithful irreducible 2-pair, no graph-canonical split)
2. SU(3)_c color rotation (W22: doesn't commute with B)
3. One-loop vs two-loop running (W23: two-loop makes the gap worse, not better)
4. Substrate Higgs VEV direction (W26: projects onto all of V_scalar, not 1-dim)

## Reframing — the c_EW = 1/3 derivation is ALGEBRAIC, not geometric

The right framing for c_EW = 1/3:

$$c_{\rm EW} = \frac{2(|E|-|V|)}{2|E|} = \frac{4}{12} = \frac{1}{3}$$

is the **BS-T bipartite-factor algebraic multiplicity** divided by total directed-edge dim. This is the EXISTING uniform-c derivation in `theorem_alpha_GUT_dark_correction.md` (Route H §3.1) and inherits its theorem-grade-conditional status.

The c_EW = 1/3 reading does NOT require identifying a canonical 1-dim sub-direction within V_scalar. The algebraic count 2(|E|-|V|) = 4 includes:
- 3 J=-1 Wilson-loop carriers (= V_cycle, the H¹ lift)
- 1 J=+1 BS-T-bipartite-extra mode (algebraically distinct from Perron-adjacency by BS-T factorization, but geometrically degenerate per W21 since Perron eigenvector ψ_3 is uniform on K_4)

The geometric ambiguity (W21) does NOT propagate to the algebraic count — the BS-T multiplicity is exact, and the count of 4 follows from |E|, |V|, k* alone.

## What this means for the framework

The c_color = 1/4 vs c_EW = 1/3 split (today's W24 + theorem doc) stands as:
- **c_color = β_1/(2|E|) = 1/4** — SU(3)_c-specific refinement: Wilson-loop H¹ content only, derived via standard SU(N) lattice gauge restriction (Wilson 1974 + H¹ master theorem). THEOREM-GRADE-NUMERICAL.
- **c_EW = 2(|E|-|V|)/(2|E|) = 1/3** — joint U(1)_Y/SU(2)_L coupling at the BS-T-bipartite-factor algebraic count. Existing `theorem_alpha_GUT_dark_correction.md` Route H derivation; THEOREM-GRADE-CONDITIONAL.

These two values are structurally DIFFERENT (β_1 vs 2(|E|-|V|) differ by 1 on K_4 because β_1 = |E|-|V|+1 = 2(|E|-|V|)/2 + 1 for the specific (|V|=4, |E|=6) case). The "+1 in c_EW vs c_color" is a graph-theoretic identity (β_1 = (|E|-|V|) + 1), NOT a missing structural mechanism.

## Open structural questions remaining

1. **Why does U(1)_Y / SU(2)_L not also restrict to β_1 like SU(3)_c does?** I.e., why doesn't c_EW = c_color = 1/4?

   The answer per the existing theorem_alpha_GUT_dark_correction.md: U(1)_Y / SU(2)_L gauge bosons couple to the full BS-T bipartite-factor algebraic sector (multiplicity 4 = 2(|E|-|V|)), not just to the β_1 cycle-mode sub-sector.

   The DEEPER structural reason — what makes SU(3)_c special — is the Z_3 = center(SU(3)) center cohomology matching β_1 on K_4 (since k_*=3 → Z_3-bits per cycle saturate H¹(K_4; Z_3) = Z_3^3 = Z_3^{β_1}). For SU(2)_L (Z_2 center), H¹(K_4; Z_2) = Z_2^{β_1} has 2^3 = 8 distinct sectors but coupling-mode count stays at 2(|E|-|V|) = 4. For U(1)_Y (U(1) continuous center), H¹(K_4; U(1)) = U(1)^{β_1} has continuous structure but the marginal-mode count stays at 4.

   This reading is internally consistent: SU(3)_c gets the FURTHER restriction to β_1 only because its center-cohomology structure (Z_3 on a 3-regular graph) is "saturated" by β_1 modes. Other gauge groups don't saturate similarly and retain the full bipartite-factor count.

   A formal proof of this "saturation argument" would tighten the c_EW = 1/3 derivation from THEOREM-GRADE-CONDITIONAL to THEOREM-GRADE-STRUCTURAL. Multi-session research; this is the residual structural frontier post-W24.

2. **The +0.008 sub-leading offset** in c_EM = 1/3 + 0.008 (from R_∞ ppt-precision constraint) — separate from c_EW = 1/3 leading. Likely a Family-D-analog vertex correction on EW gauge bosons (per the Higgs-sector Family-D mechanism in `theorem_substrate_feshbach_dark_corrections_master.md`). Separate session.

## Recommendation

The W24 closure for c_color = 1/4 is solid and shipped (commit 6224a76). The "+1 mode" question is structurally REFRAMED — it's an algebraic count, not a geometric direction. The c_EW = 1/3 reading stands at THEOREM-GRADE-CONDITIONAL under the existing uniform-c theorem.

**Honest scope of the gauge-cluster sector-specific c story:**
- c_color = 1/4: THEOREM-GRADE-NUMERICAL (W24, today)
- c_EW = 1/3: THEOREM-GRADE-CONDITIONAL (existing, unchanged; algebraic BS-T count)
- c_v_Higgs = 5/12: THEOREM-GRADE (existing anchor)

The R_∞ ppt-precision frontier (+0.008 sub-leading) and the formal saturation argument for c_EW = 1/3 are separate multi-session research directions.

## Files

- `proofs/foundations/W26_higgs_vev_direction_V_scalar_2026-05-26.py` — this probe
- `proofs/foundations/W26_verdict_higgs_vev_candidate_B_2026-05-26.md` — this verdict
- Companion: W21-W25 probes + verdicts (commit 6224a76)
- Predecessor: an internal working note §6 "Candidate B"
