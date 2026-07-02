# ℓ²-Betti Number ↔ Generation-Count Cross-Validation Candidate — NEGATIVE

**Audit anchor:** Cross-validation NEGATIVE finding for Row 18 (n_generations = 3) of `../audits/registers/uniqueness_ledger.md`. The β_1^{(2)}-via-L²-Betti route does NOT reproduce the framework's generation count under standard parameterizations.

**Date:** 2026-04-26.
**Status:** Forward-construction *negative finding*. Candidate cross-validation flagged in `forward_construction_noncommutative_iprojection.md` §6.3 does not land under standard parameterizations.
**Source ops:** Appendix A.6 (group von Neumann algebra L(F_inv(E))).
**Predecessor:** `forward_construction_noncommutative_iprojection.md`.

---

## Question (recap)

The third Tier 1 deliverable flagged a suggestive cross-validation candidate: the substrate's first ℓ²-Betti number β_1^{(2)}(F_inv(E)) potentially equals the SM generation count (3). If confirmed under canonical parameterization, this would be a category-2 cross-validation of the framework's R3 generation-count derivation from operator-algebra / ℓ²-cohomology, completely independent of the C₃ cyclic-shift on C³_obs derivation.

This document verifies the candidate under standard parameterizations.

---

## Result (preview)

**Negative.** Under the framework's canonical convention (`../framework/framework_axioms.md` §3, `dim H_visible^(L) = n(n-1)^{L-1}` with n = |E| = 6), the substrate's first ℓ²-Betti number computes to **β_1^{(2)} = 2**, not 3.

Alternative parameterizations (|E|=3 abstract involutive generators; F_3 free group on 3 generators) give **β_1^{(2)} ∈ {1/2, 2}** — none of which equals the SM generation count.

The closest match β_1^{(2)} = 2 is off by one from the SM generation count. Either the parameterization isn't quite right, or the cross-validation candidate is structurally invalid.

**Verdict: cross-validation candidate FALSIFIED at first-pass.** The substrate's ℓ²-cohomology rank is *not* the framework's generation count under any standard convention. Marks the candidate as a category-3 yield (pinned negative finding) and closes this specific cross-validation route.

---

## 1. Setup

The substrate group F_inv(E) is the free product of |E| copies of ℤ/2 (per `../framework/framework_axioms.md` §2).

Per Cheeger-Gromov 1986 / Lück 2002 standard formula for ℓ²-Betti numbers of free products:

$$\beta_1^{(2)}\big(*_{i=1}^n G_i\big) = (n - 1) - \sum_{i=1}^n \frac{1}{|G_i|} + \sum_{i=1}^n \beta_1^{(2)}(G_i)$$

For finite groups G_i, β_1^{(2)}(G_i) = 0 (all higher ℓ²-Betti numbers vanish for finite groups; only β_0^{(2)}(G_i) = 1/|G_i| is non-zero).

Specializing to G_i = ℤ/2 (|G_i| = 2):

$$\beta_1^{(2)}\big(*_{i=1}^n \mathbb{Z}/2\big) = (n - 1) - \frac{n}{2} = \frac{n - 2}{2}$$

---

## 2. Computation under candidate |E| values

### 2.1 |E| = 6 (framework's canonical convention per `../framework/framework_axioms.md` line 300)

n = 6:

$$\beta_1^{(2)} = \frac{6 - 2}{2} = 2$$

### 2.2 |E| = 3 (abstract involutive generators on srs)

If we count only the 3 distinct bond-directions as involutive generators (treating directed edges as having inverse pairs), n = 3:

$$\beta_1^{(2)} = \frac{3 - 2}{2} = \frac{1}{2}$$

This is non-integer, indicating |E| = 3 is the wrong abstract-group convention for srs (would give a 3-regular Cayley graph where each generator produces 1 undirected edge per vertex — consistent with srs valence k = 3, but ℓ²-Betti is fractional).

### 2.3 F_3 free group on 3 generators (alternative interpretation)

If F_inv(E) is interpreted as a *free group* F_3 (3 generators, no involution constraint), then by standard result β_1^{(2)}(F_n) = n − 1:

$$\beta_1^{(2)}(F_3) = 2$$

### 2.4 Summary table

| Parameterization | n | β_1^{(2)} | = generation count (3)? |
|---|---|---|---|
| Free product *_n ℤ/2, |E|=6 (framework canonical) | 6 | 2 | ✗ |
| Free product *_n ℤ/2, |E|=3 (abstract involutive) | 3 | 1/2 | ✗ |
| Free group F_n, n=3 | 3 | 2 | ✗ |

**No standard parameterization gives β_1^{(2)} = 3.**

---

## 3. Why the conjecture failed

The conjecture in the predecessor doc was based on a quick estimate using Dykema's free-dimension parameter t for L(*_n ℤ/2). I claimed t = 1 + n/2 = 4 for n = 6, then "free-dim minus 1 = 3 = generation count."

The issues:
1. **Dykema's parameterization isn't quite "t = 1 + n/2".** The exact Dykema formula for free products of finite groups depends on the convention; one standard form gives t = 1 + Σ_i (1 - 1/|G_i|) (Dykema 1994 §4), which for *_6 ℤ/2 gives t = 1 + 6 · (1/2) = 4. But this t is the *free dimension* in a specific Voiculescu-Dykema sense, NOT the ℓ²-Betti number.
2. **"Free dim minus 1 = β_1^{(2)}"** holds for *free groups* F_n (where t = n), but not for free products of finite groups. The relationship is more subtle (Connes-Shlyakhtenko 2005 has a formula relating ℓ²-Betti to non-commutative-de-Rham cohomology that doesn't reduce to "free-dim minus 1" in general).
3. **The Cheeger-Gromov formula is the rigorous answer** (Section 1) and it gives 2, not 3.

So the predecessor doc's flag was based on an unverified analogy with free groups; rigorous computation rules it out.

---

## 4. Searches for alternative integer cross-validations

Given the candidate fails, I checked whether *any* combination of standard parameterizations yields the SM generation count.

| Candidate quantity | Value for srs |E|=6 | Match generation = 3? |
|---|---|---|---|
| β_1^{(2)}(F_inv(E)) | 2 | No |
| β_0^{(2)}(F_inv(E)) | 0 (since F_inv(E) is infinite) | No |
| Dykema free-dim t | 4 (under Dykema convention) | No (4 ≠ 3) |
| t − 1 | 3 | **Coincides numerically — but this isn't a structural identification** |
| Number of Cl(2) factors per node (k=3 trivalent) | 3 | **Coincides — but is the substrate locality, not an ℓ²-cohomological invariant** |

**The numerical coincidence "t − 1 = 3 = generation count" is not a structural identification.** Dykema's t is an algebra-classification parameter, not the cohomological rank β_1^{(2)}. Without a structural reason linking generation count to t − 1 (which would require a specific theorem we don't have), the coincidence is at best heuristic.

The framework's *existing* derivation of generation count = 3 (R3 theorem, `predictions/R3_observer_c3_generation.py`) uses the C₃ cyclic-shift on C³_obs — which is on the *observer* side, not the substrate vN algebra side. The two are conceptually distinct, and there's no clean bridge between observer-C³ and substrate-ℓ²-cohomology that would yield the cross-validation.

---

## 5. Honest scope

1. **No exhaustive search.** I checked the standard parameterizations (Cheeger-Gromov for free products of finite groups; F_n free group; Dykema free dimension). There may be more exotic ℓ²-cohomological invariants (e.g., higher-dim ℓ²-Betti numbers, or torsion-corrected versions) that yield 3, but I have no first-principles reason to expect them.

2. **The closest match β_1^{(2)} = 2 is suggestive but not a match.** "2" is not the generation count; closest physical quantity it might match is "Pati-Salam SU(2)_L × SU(2)_R isospin doublet count" or "graph-T-symmetry vs toggle-process T-asymmetry pair count". Neither of these is a clean cross-validation either.

3. **Dykema's t = 4 is suggestive of a different identification.** t = 4 for srs |E|=6. The number 4 appears in the framework as the K_4 cell-quotient (4 atoms per cell) and in the Pati-Salam SU(4) gauge group. Whether t = 4 has a structural identification with these is a separate question, not flagged here as a primary candidate.

4. **The candidate was speculative from the start.** The predecessor doc explicitly flagged this as "suggestive but not yet rigorous"; the present doc confirms it doesn't survive rigorous verification.

---

## 6. Status

**Cross-validation candidate FALSIFIED.** β_1^{(2)}(F_inv(E)) = 2, not 3, under the framework's canonical |E| = 6. No standard parameterization yields 3.

**Category:** category-3 yield (pinned negative finding). Closes the candidate route "operator-algebra ℓ²-cohomology cross-validates SM generation count from a different direction".

**Effect on framework:** None negative. The framework's generation-count derivation (R3) stands on its own via C₃ cyclic-shift on C³_obs. The candidate cross-validation route is closed; the framework loses no existing structure.

**Effect on Tier 1 cluster:** Removes one speculative candidate from the queue. Cumulative Tier 1 progress unchanged: 3 main ops complete (§4.25, A.15, A.5–A.6), 1 candidate falsified, 2 main ops remaining (substrate thermal + A.4 Atiyah-Singer index).

**Honest verdict on the predecessor doc's flag:** the previous doc's §6.3 was correctly cautious ("suggestive but not yet rigorous"; "may be a coincidence of conventions"). This document confirms it was a coincidence of conventions, not a structural identification.

---

## 7. Cross-references

- `forward_construction_noncommutative_iprojection.md` §6.3 — predecessor flag (now falsified).
- `predictions/R3_observer_c3_generation.py` — the framework's existing generation-count derivation, unaffected by this finding.
- `../operator_sweep/operator_sweep_audit_appendix.md` §A.6 — context.

**Type 3 (cited published) references:**

- **Cheeger, J. & Gromov, M.** (1986). L²-cohomology and group cohomology. *Topology* 25(2), 189–215. (Foundational; ℓ²-Betti numbers of groups.)
- **Lück, W.** (2002). *L²-Invariants: Theory and Applications to Geometry and K-Theory.* Springer. §6.5 (ℓ²-Betti numbers of free products).
- **Dykema, K. J.** (1994). Interpolated free group factors. *Pacific J. Math.* 163(1), 123–135. (Free-dimension parameter for free products of finite groups; cited but not load-bearing for the negative finding.)
- **Connes, A. & Shlyakhtenko, D.** (2005). L²-homology for von Neumann algebras. *Journal für die reine und angewandte Mathematik* 586, 125–168. (Non-commutative ℓ²-cohomology; doesn't yield generation count = 3 for substrate either.)

---

## 8. Lesson for forward-construction methodology

This is the first negative candidate to land in the Tier 1 program. It illustrates the value of the operator-sweep search-instrument's *honest negative finding* category:

- The candidate was *suggestive* (numerical coincidence under one parameterization).
- Quick verification *falsified* it (rigorous Cheeger-Gromov gives a different value).
- Negative finding *closes the route* and prevents future re-discovery + over-investment.

Per an internal note and the search-instrument rubric, negative findings are valuable: they are category-3 yields that close research directions cleanly. **This document is a clean category-3 yield.** It cost ~1 session of work and saved future sessions from chasing a structurally invalid lead.

---

## 9. Status

Falsification complete. Cross-validation candidate closed. Tier 1 program continues with substrate thermal apparatus (§5.34–§5.38 + A.7 KMS) as the next op.
