# Theorem — D2 leg 1: the vertex axis W does NOT descend to the canonical M₃(ℂ) home

**Date:** 2026-07-15 · **Station:** GEN-IDENT-D2, leg 1 (the "quick decisive half"; user-gated
"seal leg 1 only, then reassess"). **Grade:** theorem (finite-group + one self-contained II₁
outerness lemma; exact, sealed-concurred). **Builds on (sealed, NOT re-litigated):** GEN-IDENT-A
(⟨σ,W⟩=A4 irreducible on ρ₃), D0 (α=α_σ properly outer on M=L(F_inv(6)), M a II₁ factor), D1
(the canonical M₃(ℂ) leg of M⋊_α ℤ₃ is rigid and carries σ). **NOTHING here reads any
mass/ppm/Koide/mixing value** (goal-seek guard; AST self-scan in the driver).

**Receipts:** driver `proofs/foundations/genident_D2_half_descent_check_2026-07-15.py` (32/32 PASS,
< 1s, pure integer combinatorics + exact free-product word arithmetic, `OMP_NUM_THREADS=4`);
sealed adversarial check CONCUR-WITH-CORRECTION (independent fresh code), corrections applied and
recorded in §6 + internal research notes.

**Verdict leaf:** **W does not descend to an automorphism of M⋊_α ℤ₃; the canonical M₃(ℂ)
observer home carries σ but NO forced/canonical W-action.** This is the clean, decisive HALF of
D2 = ORTHOGONAL (the automorphism route). The remaining leg — whether a *vertex-mediated,
non-automorphism* coupling −κ·I(A;B) could still reach W across the level gap — is leg 2, held for
user reassessment; the D2-a sweep found it un-posable without the (D)-trap or an unforced gluing.

---

## 0. Setup

`G = F_inv(6) = *_{i=1}^{6} ⟨t_i | t_i²=1⟩`, `M = L(G)` (type II₁ factor, D0-i). The winding/
generation ℤ₃ is `α = α_σ`, the lift of the generator-relabelling `σ̂(t_i)=t_{σ(i)}` where
`σ = (1 2 3)(4 5 6)` is the K4 **edge-action** of a tetra-vertex 3-cycle (D0). The canonical
observer home is the `M₃(ℂ)` leg of `M ⋊_α ℤ₃ ≅ M₃(ℂ) ⊗ M^α` (D0/D1). The vertex axis `W` is a
second order-3 element of A4 with `⟨σ,W⟩ = A4` (GEN-IDENT-A); its edge-action `W̃ ∈ S₆` realizes
it on M's generators, giving `α_W ∈ Aut(M)`.

**The question (leg 1):** does the canonical home *carry* W — i.e. does `α_W` descend to an
automorphism of `M ⋊_α ℤ₃` restricting to it on `M`? (Why this is the right test: §2.)

---

## 1. The finite-group core: ⟨σ⟩ is a self-normalizing Sylow-3, and W is outside it

`⟨σ⟩ ≅ ℤ₃` is a Sylow-3 subgroup of A4. A4 has `n₃ = 4` conjugate Sylow-3 subgroups
(driver 2c), so `|N_{A4}(⟨σ⟩)| = |A4|/n₃ = 12/4 = 3`, i.e. **`N_{A4}(⟨σ⟩) = ⟨σ⟩`
(self-normalizing)** (driver 2a/2b). Since `⟨σ,W⟩ = A4 ≠ ⟨σ⟩`, `W ∉ ⟨σ⟩`; and because the
normalizer is `⟨σ⟩` itself, **`W ∉ N_{A4}(⟨σ⟩)`, i.e. `W σ W⁻¹ ∉ ⟨σ⟩`** (driver 3a; `WσW⁻¹` is a
generator of a *different* Sylow-3). The verification adds the sharp structural note: `W` **is**
A4-conjugate to `σ` (same class of 4), but generates a different Sylow-3 — which is exactly why
`WσW⁻¹ ∉ ⟨σ⟩` despite `W, σ` being "the same type" of element.

**Non-vacuity (driver 4):** of A4's 12 elements, **exactly the 3 elements of `⟨σ⟩` normalize
`⟨σ⟩`** — the double-transpositions fail too, so the criterion is `⟨σ⟩`-membership-specific, not a
parity/coset artifact. The positive controls `σ, σ²` pass (they *do* descend). W is one of the 9
that fail. The test discriminates.

---

## 2. The descent criterion and why it is the right test (functoriality)

**Descent criterion (standard crossed-product covariance).** An automorphism `β ∈ Aut(M)` extends
to `Φ ∈ Aut(M⋊_α ℤ₃)` with `Φ|_M = β` iff `Φ(u) = w·u^k` for a unitary `w∈M` and some `k` with
`gcd`-compatibility, and covariance then forces `β α β⁻¹ = Ad(w) ∘ α^k`, i.e.

> **`β` descends (fixing M)  ⟺  `[β α β⁻¹] ∈ ⟨[α]⟩` in `Out(M)`**

(`k=0` is excluded — it makes `Φ` non-surjective). Only the *outer class* of the implementing
unitary enters, so the criterion is well-posed in `Out(M)`. (verification re-derived this from the
covariance relation independently and confirmed the `Out(M)`-not-`Aut(M)` reading is correct and
not a loophole; refs: Takesaki, *Theory of Operator Algebras*; Brown–Ozawa §4.)

**Functoriality — why "does `α_W` descend from M" answers "does the M₃ leg carry W".** The
decomposition `M⋊_α ℤ₃ ≅ M₃(ℂ) ⊗ M^α` is constructed **functorially from the pair `(M, α)` alone**
(D0/D1). Hence any *forced* action of a substrate symmetry on the `M₃(ℂ)` leg must factor through
an extension of that symmetry across `(M, α)` — that is, through an automorphism of the crossed
product restricting to it on `M`. So "no such extension" = "no forced W-action on the leg." (The
verification flagged this should be stated explicitly rather than left implicit; done here.)

**Honest scoping (what leg 1 does NOT claim).** `M₃(ℂ) = B(ℂ³)` is a full matrix algebra, so an
*inner* unitary `ρ₃(W)` realizing W on `ℂ³` of course **exists** (Skolem–Noether). Leg 1 denies
only the *forced/canonical/functorial* W-action, **not** the existence of such a unitary. Choosing
one is an *unforced* frame choice — precisely the U(3)-freedom GEN-HOMES named — and nothing in
`(M,α)` fixes it. (verification confirmed the driver/writeup do not overreach into the false
"no unitary exists" claim.)

---

## 3. The Generalized-Outerness Lemma (the transfer's engine — proved self-contained)

The transfer needs `S₆ ↪ Out(M)` injectivity, which needs proper outerness for *arbitrary*
nontrivial generator-permutations — the sealed D0-iii text proved it only for the
**fixed-point-free** `σ, σ²`, and the transfer requires a permutation (`τ₁`, §4) with fixed
points. This lemma closes that gap.

> **Lemma.** For `G = F_inv(6)` and **any nontrivial** `τ ∈ S₆`, the lift `α_τ ∈ Aut(M)`
> (`α_τ(λ_g)=λ_{τ̂(g)}`, `τ̂(t_i)=t_{τ(i)}`) is **properly outer**.

**Proof.** `M` is a factor (D0-i), so on `M` "not inner" ≡ "properly outer" (D0-iii's factor
dichotomy). Suppose `α_τ = Ad(u)`, `u = Σ_h c_h λ_h ∈ M`, `Σ_h|c_h|² = ‖u‖₂² = 1`. From
`α_τ(λ_g) = u λ_g u^*`, i.e. `u λ_g = λ_{τ̂(g)} u`, Fourier-coefficient matching in `ℓ²(G)` gives
(exactly as in D0-iii)

  `c_h = c_{τ̂(g)\,h\,g^{-1}}`  for all `g, h ∈ G`,

so `c` is **constant on every τ̂-twisted conjugacy orbit** `g ⋆ h := τ̂(g)\,h\,g^{-1}`. If any
orbit is infinite and `c_{h}≠0` on it, `Σ|c_h|² = ∞` — contradiction. It thus suffices to show
**every τ̂-twisted class is infinite.**

Fix `h ∈ G`; if `h≠e` let `j_1, j_k` be its first/last syllable factors. Because `τ ≠ id`, choose:
- **`h = e`:** any `b` with `τ(b) ≠ b` — *exists since `τ ≠ id`* — and any `a ≠ b`;
- **`h ≠ e`:** any `b ∉ {τ^{-1}(j_1),\, j_k}` (≤2 forbidden of 6 ⟹ ≥4 valid), and any `a ≠ b`.

Set `g_n = (t_a t_b)^n` (reduced, length `2n`). Then `τ̂(g_n) = (t_{τ(a)} t_{τ(b)})^n`, reduced
(since `τ(a)≠τ(b)` as `τ` injective and `a≠b`), length `2n`, and consider
`w_n = τ̂(g_n)\, h\, g_n^{-1}`:
- **left junction:** last letter `t_{τ(b)}` of `τ̂(g_n)` meets the first syllable of `h` (`t_{j_1}`
  if `h≠e`, else the first letter `t_b` of `g_n^{-1}`). The choice gives `τ(b)≠j_1` (`h≠e`) or
  `τ(b)≠b` (`h=e`) — **no cancellation**;
- **right junction:** last syllable of `h` (`t_{j_k}`; absent if `h=e`) meets first letter `t_b` of
  `g_n^{-1}`; `b≠j_k` — **no cancellation**;
- `τ̂(g_n)` and `g_n^{-1}` are internally alternating in two distinct letters, hence reduced.

So `w_n` is reduced exactly as written: `|w_n| = 2n + |h| + 2n = 4n + |h|`, strictly increasing,
so the `w_n` are pairwise distinct — the twisted class of `h` is **infinite**. Hence `c ≡ 0`,
`u = 0`, contradicting `‖u‖₂ = 1`. Therefore `α_τ` is not inner, hence properly outer. `∎`

**Where fixed-point-freeness was (not) needed.** It entered D0-iii *only* at the `h=e` junction
(needing `τ(b)≠b` for the chosen `b`). For general nontrivial `τ`, `∃b: τ(b)≠b` is automatic; the
`h≠e` cases never used it. So fixed-point-freeness was a convenience for `σ`, never a requirement.

**Corollary (S₆ ↪ Out(M) injective on generator-permutations).** `α_π = α_ρ` in `Out(M)`
`⟺ α_{πρ^{-1}}` inner `⟺` (Lemma) `πρ^{-1} = id ⟺ π = ρ` (using `π ↦ α_π` a homomorphism). `∎`

**Driver anchor (STEP 5B):** for the exact `τ_k := (WσW^{-1})·σ^{-k}` (`k=0,1,2`) the transfer
needs, the driver verifies with exact free-product word arithmetic that each `τ_k` is nontrivial
and every `τ_k`-twisted orbit grows as `4n+|h|` (witnesses `h ∈ {e, t_0, t_0t_1, t_3t_4t_5}`,
`n=0..8`) — **including `τ_1 = (5,4,2,3,1,0)`, which has two fixed points `{2,3}`**, the case the
sealed D0-iii text did not literally cover.

---

## 4. The transfer: W does not descend

Take `β = α_W`. The edge-action `A4 → S₆` is a homomorphism (and faithful — driver 5a/5b), so
`β α β^{-1} = α_W α_σ α_W^{-1} = α_{W̃ σ̃ W̃^{-1}} = α_{\,\text{edge}(WσW^{-1})}`. Now

  `[β α β^{-1}] ∈ ⟨[α]⟩ = \{[id],[α_σ],[α_{σ²}]\}`
  `⟺ α_{(WσW^{-1})·σ^{-k}} = α_{τ_k}` inner for some `k∈\{0,1,2\}`
  `⟺` (Lemma) `τ_k = id` for some `k`
  `⟺ WσW^{-1} ∈ ⟨σ⟩`.

By §1 this is **false** (`WσW^{-1} ∉ ⟨σ⟩`): each `τ_k` is nontrivial, hence `α_{τ_k}` properly
outer (Lemma), hence `[β α β^{-1}] ∉ ⟨[α]⟩`. By the descent criterion (§2),

> **`α_W` does not descend to an automorphism of `M ⋊_α ℤ₃`.** `∎`

By functoriality (§2), the canonical `M₃(ℂ)` home therefore carries **no forced W-action**.

---

## 5. Verdict, and the honest bound

# **D2 LEG 1: W DOES NOT DESCEND — the canonical M₃(ℂ) home has no forced W-action.**

- Rigorous and sealed-concurred: the finite-group core (§1), the descent criterion + functoriality
  (§2), the self-contained Generalized-Outerness Lemma (§3), the transfer (§4).
- **Discriminates** (§1 non-vacuity): only `⟨σ⟩`'s 3 elements pass; W is one of the 9 failers;
  `σ, σ²` pass as positive controls.
- **Scoped honestly:** this kills the *automorphism/functorial* route only. It does **not** claim
  no unitary `ρ₃(W)` exists on `ℂ³` (false by Skolem–Noether) — only that none is *forced*.

**What leg 1 does NOT settle (leg 2, held for reassessment).** Whether a *vertex-mediated,
non-automorphism* coupling `−κ·I(A;B)` across the level gap (finite `H_hist⊗F` ↔ the II₁ home)
could still transmit W. The D2-a sweep found this un-posable without the (D)-trap (F's level-1 ≡
substrate ρ₃) or an unforced gluing (a smuggled datum) — i.e. **no forced coupling exists**, which
is the top-down verdict, but softer than leg 1's proof. Leg 1 + leg 2 together = D2 = ORTHOGONAL;
leg 1 alone = "the canonical route puts no W on the home." **Books nothing about −70 ppm or the
e/μ/τ labeling** — the labeling stays external via this route (the deepest positive this route
could reach, the one-bit Schur collapse of GEN-IDENT-B, is *not* triggered). Route β (dynamical pin
of the run-endpoint `s`) is the orthogonal fork and remains available.

---

## 6. verification concurrence + corrections applied

Independent verification (fresh code, no reuse of the architect's helpers): **CONCUR-WITH-
CORRECTION.** It re-derived all finite-group facts, audited the descent criterion (correct, no
verdict-flipping loophole), audited the scoping (honest, no overreach), and confirmed goal-seek/
(D)-trap clean. Two corrections, both applied above and **non-verdict-flipping**:

1. **Transfer citation gap (real).** The original driver anchored `S₆↪Out(M)` on D0-**ii**
   (group-outer, `t_i≁t_j`), weaker than the needed `Out(M)`-outerness (D0-**iii**), whose sealed
   text is scoped to fixed-point-free permutations — and `τ₁` has two fixed points. **Fix:** the
   Generalized-Outerness Lemma (§3), proved self-contained, plus the STEP-5B word-arithmetic anchor
   for the exact `τ_0,τ_1,τ_2`. (The verification verified the generalization computationally to
   `n=10`; §3 now supplies the from-scratch written proof it asked for.)
2. **Functoriality implicit.** Made explicit (§2).

Residual doubt named by the verification (the generalized lemma was verified computationally, not
written as a proof) is now discharged by §3's proof.

---

## Regression anchor

`proofs/foundations/genident_D2_half_descent_check_2026-07-15.py` — 32/32 PASS, < 1s, pure integer
combinatorics + exact free-product word arithmetic (no floating point; AST self-scan confirms).
Anchors: `⟨σ⟩` self-normalizing Sylow-3 (`n₃=4`); `WσW⁻¹∉⟨σ⟩`; the discrimination census; the
faithful edge-action `A4→S₆`; the `τ_k` (`k=0,1,2`) proper-outerness via `4n+|h|` twisted-orbit
growth (incl. `τ₁`'s fixed points). `the_run.py`/Layer-1 untouched; no `the_net.py` accretion
(leg 1 is a structural/rigidity result about `(M,α)`, not a numeric read); not wired into
`verify.py` (matches D0/D1/C precedent).
