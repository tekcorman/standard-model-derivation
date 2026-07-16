# Theorem — D1: the canonical M₃(ℂ) leg of M ⋊_α ℤ₃ carries NO residual moduli

**Date:** 2026-07-15 · **Station:** GEN-IDENT-D, D1 (freeze
internal research notes §2 "D1") · **Builds on (sealed,
NOT re-litigated):** D0 = OUTER-CONFIRMED (`docs/theorems/genident_D_outerness_2026-07-15.md`,
sealed-concurred internal research notes). **Grade:**
theorem (finite-dimensional operator-algebra lemma, self-contained, using D0-iii/iv as an already-
proven external input; NOT re-deriving outerness). **NOTHING BELOW IS BOOKED AS FORCED** — this is
implementation pass output for the sealed adversarial check, per L8.

**Receipts:** driver `proofs/foundations/genident_D1_canonical_home_check_2026-07-15.py` (33/33
PASS, runtime < 2s, exact sympy algebraic arithmetic, `OMP_NUM_THREADS=4`); working note
internal research notes.

**Verdict leaf (freeze §3): D1 PINS THE CANONICAL M₃(ℂ) — relative commutant `M' ∩ (M ⋊ ℤ₃) = ℂ`,
no moduli.** Proceed-ready for D2. See §5 for the one place rigor is calibrated-not-fully-closed
(the "linear independence of Fourier coefficients" step, standard/definitional but not re-derived
from first principles here), stated honestly rather than glossed over.

---

## 0. What D1 is asked to show, and what it is not

D0 (sealed) established: `α` is properly outer on `M = L(F_inv(6))`, `α²` is also properly outer, so
the ℤ₃-action `{id, α, α²}` is **free**, and the standard Galois theorem for free finite-group
actions on a II₁ factor gives

> `M ⋊_α ℤ₃ ≅ M₃(ℂ) ⊗ M^α`.

D1's job is to show the `M₃(ℂ)` leg of this decomposition is **canonical** — pinned, with no leftover
unitary freedom — in exact contrast to GEN-IDENT-C, where the analogous finite-carrier recipe left a
**24-complex-dimensional** moduli space (`U(4)×U(2)×U(2)`, the orbit of `U(M^σ)` acting on a choice
of "anchor" vector) that no structural criterion could collapse to a point (five independent attempts
failed, sealed-confirmed).

**D1 does NOT:** put `W` (the vertex axis) on the home, label e/μ/τ, or derive −70 ppm. That is D2,
explicitly out of scope here (freeze §2, "D2 — ONLY if D1").

---

## 1. The rigidity theorem: `M' ∩ (M ⋊_α ℤ₃) = ℂ`

### 1.0 Setup

Write elements of `M ⋊_α ℤ₃` as `x = x_0 + x_1 u + x_2 u²`, `x_i ∈ M`, where `u` is the canonical
unitary implementing `α` (`u x u^{-1} = α(x)` for `x ∈ M`, `u³ = 1`). This is the standard algebraic
presentation of the crossed product of a von Neumann algebra by a single automorphism of finite order
(e.g. realized concretely via the covariant representation on `L²(M,τ) ⊗ ℓ²(ℤ₃)`,
`π(x)(ξ ⊗ δ_n) = α^{-n}(x)ξ ⊗ δ_n`, `π(u)(ξ ⊗ δ_n) = ξ ⊗ δ_{n+1}` — a direct check, done explicitly
in the driver's Section 2 covariance test, confirms `π(u)π(x)π(u)^{-1} = π(α(x))` as required).
**Fourier uniqueness** (every element of `M ⋊_α ℤ₃` has a UNIQUE expansion `Σ x_n u^n`, `x_n ∈ M`) is
standard/definitional for this construction — the same kind of "matching coefficients of an
orthogonal grading" fact D0-iii used for `ℓ²(G)`, here for the `ℤ₃`-grading of the crossed product;
see §5 for the honest calibration note on this step.

### 1.1 The key lemma (the one genuinely new piece of mathematics this station supplies)

> **Lemma.** Let `M` be a **finite** (i.e. type II₁ or finite type I) **factor**, `α ∈ Aut(M)`. If
> there is a **nonzero** `w ∈ M` with
> `a·w = w·α(a)` for **all** `a ∈ M`,
> then `α` is **inner**.

**Proof.** Take adjoints of `aw = wα(a)`: `w*a* = α(a)*w* = α(a*)w*`; relabel `a* → a` (a is
universally quantified, so this relabelling is legitimate): `w*a = α(a)w*` — call this (★).

Multiply the original relation `aw = wα(a)` on the left by `w*`: `w*aw = w*wα(a)`. Substitute
`w*a = α(a)w*` (★) into the left side: `w*aw = (α(a)w*)w = α(a)(w*w)`. So:

> `α(a)(w*w) = (w*w)α(a)` for all `a ∈ M`.

Since `α` is surjective (an automorphism), this says `w*w` commutes with **every** element of `M`:
`w*w ∈ M' ∩ M`. Since `M` is a **factor**, `M' ∩ M = ℂ·1`, so `w*w = λ·1` for some scalar `λ ≥ 0`
(positivity since `w*w ≥ 0`). Since `w ≠ 0`, `λ > 0`.

Write `w = √λ · v` where `v := w/√λ` satisfies `v*v = 1` — i.e. `v` is an **isometry** in `M`. Since
`M` is **finite** (faithful normal trace `τ`), every isometry is a unitary: `τ(vv*) = τ(v*v) = τ(1)`
(trace cyclicity), and `vv* ≤ 1` with equal trace to `1` forces `vv* = 1` by faithfulness of `τ`
(`τ(1 - vv*) = 0`, `1-vv* ≥ 0` ⟹ `1-vv*=0`). So `v` is a genuine **unitary**.

Substitute `w = √λ v` into `aw = wα(a)`: `√λ·av = √λ·vα(a)` ⟹ `av = vα(a)` ⟹
`v^{-1}av = α(a)` for all `a ∈ M`, i.e. **`Ad(v^{-1}) = α`** — `α` is inner, implemented by `v^{-1}`.
`∎`

**Where each hypothesis is used, made explicit (so a checker can attack the weakest link):**
- `M` a **factor**: used once, to collapse `w*w ∈ M'∩M` to a scalar.
- `M` **finite**: used once, to upgrade the isometry `v` to a unitary. (This is exactly where D0-i's
  ICC/factor result is load-bearing a second time: `M = L(F_inv(6))` is type II₁, hence finite, by
  D0-i.) **This hypothesis is not decorative** — for a type I∞ or III factor the lemma is false in
  general (isometries need not be unitaries there); the argument genuinely needs `M` finite, and D0
  supplied exactly that.
- Surjectivity of `α`: trivial (automorphisms are surjective by definition).

### 1.2 Applying the lemma with D0's outerness

Let `y = y_0 + y_1u + y_2u² ∈ M' ∩ (M ⋊_α ℤ₃)`, i.e. `y` commutes with every `x ∈ M` (note: `M ⊂ M⋊ℤ₃`
via `x ↦ x·u^0`, and the relative commutant is by definition commutation with this copy of `M`, not
with the whole crossed product). For `x ∈ M`:

`xy = x y_0 + (x y_1)u + (x y_2)u²`,
`yx = y_0 x + y_1(ux) + y_2(u²x) = y_0x + y_1 α(x) u + y_2 α²(x) u²`

(using `u^n x = α^n(x) u^n`, a direct consequence of `uxu^{-1}=α(x)` iterated). Equate `xy = yx` and
match the (unique, §1.0) Fourier coefficients of `u^0, u^1, u^2`:

- `u^0`: `x y_0 = y_0 x` for all `x ∈ M` ⟹ `y_0 ∈ M' ∩ M = ℂ·1` (M a factor, D0-i).
- `u^1`: `x y_1 = y_1 α(x)` for all `x ∈ M`. By the Lemma (§1.1) applied to `α` (properly outer,
  D0-iii): if `y_1 ≠ 0` then `α` would be inner — **contradicts D0-iii**. So `y_1 = 0`.
- `u^2`: `x y_2 = y_2 α²(x)` for all `x ∈ M`. By the Lemma applied to `α²` (also properly outer,
  D0-iv): if `y_2 ≠ 0` then `α²` would be inner — **contradicts D0-iv**. So `y_2 = 0`.

Hence `y = y_0 · 1 = c·1` for a scalar `c`. **`M' ∩ (M ⋊_α ℤ₃) = ℂ·1`.** `∎`

This is precisely the standard characterization "free/outer action ⟺ `M` irreducible in `M⋊G`"
(cf. Jones, *Actions of finite groups on the hyperfinite type II₁ factor*, Mem. AMS 1980; Connes,
*Outer conjugacy classes of automorphisms of factors*, Ann. Sci. ENS 1975 — same citation family D0
already used), derived here **self-contained** rather than merely cited, exactly as the freeze
demands ("derive it self-contained... from the Fourier/twisted-conjugation machinery already used in
D0").

**Driver anchor (calibration, not a proof of the infinite-dimensional statement — see the driver's
own header for the epistemic distinction):** `proofs/foundations/genident_D1_canonical_home_check_2026-07-15.py`
Section 2 realizes the **bare `u^1` relation** `x y_1 = y_1 α(x)` concretely in `M_2(ℂ)` (where `α`
is FORCED inner by Skolem–Noether) and verifies **exactly** the lemma's central equivalence in the
one regime where it is directly computable: `y_1 := u0^{-1}` solves the relation (exhaustively, on
the full 4-element basis of `M_2(ℂ)`), and the corresponding crossed-product element
`Y = π(u0^{-1})π(u)` **does** commute with all of `M` (the positive witness) — while a generic
non-solving `y_1` (`E_12`) **does not** give a commuting element (the negative control, confirming
the mechanism discriminates rather than being vacuously true). This calibrates the "if `w≠0` exists,
`α` inner" direction of the Lemma; the REAL station's use of the Lemma is the **contrapositive**
("`α` properly outer [D0] ⟹ no such `w`"), which cannot be finite-dimensionally instantiated (no
finite factor hosts a properly outer ℤ₃ action, Skolem–Noether — this is exactly the driver's Section
3 dimension-mismatch witness: `dim(M⋊ℤ₃)=12` definitionally but the tensor-decomposition formula
would need `18`, an inconsistency that only resolves for a genuinely free action).

---

## 2. Canonicity of the M₃(ℂ) leg: pinpointing exactly where GEN-IDENT-C's 24-dim count fails

### 2.1 What GEN-IDENT-C's moduli space actually was, restated precisely

GEN-IDENT-C (sealed) built `U_F ∈ B(F) = M_8(ℂ)`, the order-3 unitary lift of `σ`, with eigenspaces
`H_0, H_ω, H_ω²` of dimensions `(4,2,2)`. Its **commutant** — `M^σ := \{X ∈ B(F) : XU_F = U_F X\}` —
has complex dimension `Σ dᵢ² = 16+4+4 = 24` (sealed-confirmed two independent ways: eigenspace-block
formula and SVD nullity). The verifier's control B4 showed explicitly: **any** unitary
`W ∈ U(M^σ) ≅ U(4)×U(2)×U(2)` moves a valid "anchor" `v = (v_0+v_ω+v_ω²)/√3` (one unit vector chosen
independently in each eigenspace) to another equally valid, but generically **different**, anchor
`Wv` — because `W` commutes with `U_F`, so `\{Wv, U_F Wv, U_F²Wv\} = W·\{v,U_Fv,U_F²v\}` is still an
orthonormal triple. **The moduli space IS the orbit of `U(M^σ)` acting on a seed anchor, dimension
24, with no fixed point** (a nontrivial unitary group of dimension `>1` acting on a vector space has
no vector fixed by the whole group).

### 2.2 The direct analogue in `M ⋊_α ℤ₃`, computed

The analogous object to "the commutant of the implementing unitary" is the commutant of `u` **inside
the crossed product itself** (the natural ambient algebra for this construction, playing the role
`B(F)` played for GEN-IDENT-C). Compute it directly: for `x = Σ x_n u^n ∈ M⋊ℤ₃` to commute with `u`,
match `u·x = x·u` in Fourier coefficients (`u·(x_n u^n) = α(x_n) u^{n+1}`; matching against
`x·u = Σ x_n u^{n+1}` gives, per grade, `α(x_n) = x_n`):

> **The commutant of `u` in `M ⋊_α ℤ₃` is `\{Σ_n x_n u^n : x_n ∈ M^α ∀n\}`, i.e. exactly
> `M^α ⊗ ℂ[ℤ₃]` (as a vector space; `ℂ[ℤ₃]` here just tracks the three grades).**

This is a **completely different structural shape** from GEN-IDENT-C's `M^σ`: it is not "a product of
full matrix algebras on eigenspaces" (which is what gave `U(4)×U(2)×U(2)`, dimension 24, a *matrix
group* acting *within* each eigenspace) — it is **`M^α` itself, appearing once per grade**, with **no
extra internal unitary freedom beyond `U(M^α)`** at each grade. Concretely: the commutant of `u`
contains no analogue of "an independent `U(4)` rotating within a 4-dimensional eigenspace," because
there is no such eigenspace here at all — the grade-1 piece `Mu` is not "a bare vector space of some
dimension `d`" the way GEN-IDENT-C's `H_0` was; as an `M^α`-`M^α` bimodule, it is a **free module of
rank exactly 1** (see §2.3), and a rank-1 free module's only compatible unitary self-maps are
`M^α`-scalar multiplication — which is already accounted for as the `M^α` leg of the target
decomposition, not an extra modulus on top of it.

### 2.3 Why the rank is exactly 1 (the precise place GEN-IDENT-C's mechanism fails)

**GEN-IDENT-C's mechanism, restated as a rank statement:** on the finite carrier, `U_F` is **inner**
(`U_F ∈ B(F)` itself — Skolem–Noether forces this). Because `U_F` is an honest element of the ambient
algebra, its eigenspaces `H_0, H_ω, H_ω²` carry **no forced module structure at all** — they are bare
finite-dimensional Hilbert spaces, and the group of unitaries "compatible with the eigenspace
decomposition" is the FULL unitary group of each eigenspace, `U(d_i)`, because nothing constrains a
vector's choice within an eigenspace beyond orthonormality. This is precisely why `M^σ ≅ ⊕ M_{d_i}(ℂ)`
is a **product of full matrix algebras**, not a scalar-multiple-only algebra — the "multiplicity" of
the trivial building block in each eigenspace is `d_i > 1`, and Skolem–Noether guarantees every
automorphism of that full block is realized inside it, so there is no obstruction to rotating freely.

**In `M ⋊_α ℤ₃`, by contrast:** the grade-`n` piece `Mu^n` (`n=1,2`), regarded as a left-`M^α`,
right-`M^α` bimodule (via `x·(mu^n)·y = xmα^n(y)u^n` for `x,y ∈ M^α`), is a bimodule over `M^α` — and
**properness of the outer action is exactly the statement that this bimodule has rank 1**, i.e. is
isomorphic to `M^α` itself as an `M^α`-`M^α` bimodule (this is the standard content of "the index
`[M:M^α] = |G|` exactly," the Jones-index computation for a free finite-group action — `M`, as a
right `M^α`-module, is free of rank `|G|=3`, one rank-1 copy per grade, not a higher-multiplicity
module). **A rank-1 free bimodule's group of bimodule-automorphisms compatible with the ℤ₃-grading is
`U(M^α)` acting by right multiplication — nothing more.** This `U(M^α)` freedom is not "extra
moduli left over on top of the `M₃(ℂ)⊗M^α` decomposition" — it **is** the `M^α` tensor leg, already
present and accounted for on the right-hand side of the isomorphism. Once you quotient by it (i.e.
factor it into the `⊗M^α` leg, exactly as the theorem states), the `M₃(ℂ)` leg has **zero** remaining
freedom — which is exactly the content of `M' ∩ (M⋊ℤ₃) = ℂ` proven in §1: any element that could
"rotate the M₃(ℂ) matrix-unit structure while fixing `M`" would have to live in this relative
commutant, and there is nothing there but scalars.

**The single-sentence pinpoint the freeze asks for:** GEN-IDENT-C's `U(4)×U(2)×U(2)` count is `Σd_i²`
— the dimension of the **full endomorphism algebra of each eigenspace treated as a bare vector
space**, which is available *only because* the implementing unitary is inner (an honest operator
already sitting inside the same finite ambient algebra, so its eigenspaces are unconstrained vector
spaces). The moment the action is properly outer (D0), the "eigenspace" `Mu^n` is no longer a bare
vector space at all — it is a **rank-1 `M^α`-bimodule** (a consequence of outerness/freeness, via the
index formula), whose only compatible unitary self-maps are `U(M^α)` acting by bimodule scalars, which
is precisely the `M^α` factor the theorem already isolates. **The `Σd_i²`-type over-count is
structurally impossible here because there is no eigenspace of dimension `d_i>1` to over-count in the
first place — outerness collapses "eigenspace" to "rank-1 bimodule."**

### 2.4 A second, independent confirmation: the dimension mismatch (finite-toy arithmetic)

The driver's Section 3 makes this concrete with plain integer arithmetic, using the `M_2(ℂ)` toy
(where `α` is *necessarily* inner, Skolem–Noether): `dim_ℂ(M⋊ℤ₃) = |ℤ₃|·dim(M) = 3·4 = 12`
(definitional, holds regardless of inner/outer). The clean tensor formula would require
`dim(M⋊ℤ₃) = |ℤ₃|²·dim(M^α) = 9·2 = 18` (since `M^α` = diagonal matrices, dim 2, computed directly
from which basis elements commute with `u0`). **`12 ≠ 18`** — the tensor decomposition is
**dimensionally impossible** for this inner action. This is not merely "the recipe gives a messy
answer" (GEN-IDENT-C's finding) but a flat arithmetic **inconsistency** — confirming from a completely
different angle (counting, not operator algebra) that properness of outerness is not a cosmetic
upgrade but the load-bearing hypothesis that makes the `M₃(ℂ)⊗M^α` formula even well-posed. (For the
real, infinite-dimensional `M = L(F_inv(6))`, this dimension count is not literally checkable — but
the standard index formula `[M:M^α] = 3` exactly, which the driver's finite mismatch illustrates the
necessity of, is precisely what D0's properness of outerness delivers via the Jones/GHJ theorem D0-iv
already invoked.)

---

## 3. Stripping the labeling: what survives, Koide/DFT-clean

By construction, **nothing above used any mass, ppm, Koide-Q, mass-ordering, mixing, CKM, or PMNS
value**, and no Z₃-Fourier/DFT conjugation matrix (`m1b_c_basis_match.py:280–285,289`, contaminated
per the freeze's §1 hard rail) was built or referenced anywhere — verified by the driver's AST
self-scan (Section 3, zero physics-codebase imports, zero floating-point literals) exactly as D0's
own scan did.

**What survives, stated precisely:**
- A **canonical `M₃(ℂ)` observer home** exists inside `M ⋊_α ℤ₃`, pinned by outerness alone
  (`M'∩(M⋊ℤ₃)=ℂ`, §1) — the exact object GEN-IDENT-C proved could not be built on the finite carrier.
- This home carries `σ` (the winding/generation ℤ₃ action) by construction — it *is* the crossed
  product of `M` by `σ`'s own lift `α`.
- The `M^α` leg is a genuine (large, type-II₁) subalgebra, **not** small or degenerate — per the
  freeze's own rail, this is expected and fine; canonicity of the `M₃(ℂ)` leg comes from outerness,
  not from `M^α` being trivial.
- The residual discrete `ℤ₂ = Out(A4)` bit from GEN-IDENT-B's Schur collapse is **untouched** by this
  station — it remains external, exactly as the goal-seek guard requires (naming the home is not
  resolving the label).

**What is explicitly NOT established here (the honest bound, §4 below):**
- Whether `W` (the vertex axis, GEN-IDENT-A) sits on this home at all — that is D2, a genuinely new
  cross-level question (the home lives on `M`, type II₁; `W` lives on the finite `H_hist⊗F` carrier)
  and is **not attempted** in this station.
- No labeling of e/μ/τ, no −70 ppm, no mass-ordering.

---

## 4. Verdict

# **D1 PINS THE CANONICAL M₃(ℂ) — relative commutant = ℂ, no moduli. Proceed-ready for D2.**

- `M' ∩ (M ⋊_α ℤ₃) = ℂ·1`, proven self-contained (§1) from D0-i (factor), D0-iii (`α` properly
  outer), D0-iv (`α²` properly outer), via a new short lemma (§1.1: finite factor + a nonzero
  relation-solving element ⟹ inner) applied twice.
- The `M₃(ℂ)` leg carries **no residual moduli**: the commutant of `u` inside `M⋊ℤ₃` is exactly
  `M^α` (§2.2), not an inflated matrix-block algebra, because outerness forces each graded piece
  `Mu^n` to be a **rank-1** `M^α`-bimodule (§2.3) — this is the precise, named place GEN-IDENT-C's
  `Σd_iᵢ²` over-count (available only because the finite-carrier lift was inner, hence eigenspaces
  were bare, unconstrained vector spaces) becomes structurally unavailable. A second, independent,
  purely-arithmetic confirmation (§2.4) shows the tensor decomposition is dimensionally impossible
  for an inner action at all.
- Koide/DFT-clean (§3): the surviving structure is a canonical `M₃(ℂ)` home carrying `σ`, with the
  `ℤ₂` residual (GEN-IDENT-B) untouched and nothing data-fixed.

---

## 5. Honest bound and where rigor is calibrated, not fully re-derived from first principles

**D1 does NOT label e/μ/τ, does NOT derive −70 ppm, and does NOT put `W` on the canonical home.**
D2 (the cross-level mediation test between this `M₃(ℂ)` home and the vertex's `W`-carrier on the
finite `H_hist⊗F`) is explicitly out of scope, per the freeze's own gate structure.

**Where this write-up is honest about being less than a from-scratch re-derivation of every standard
fact:**

1. **§1.0 "Fourier uniqueness"** (every crossed-product element has a unique `Σx_nu^n` expansion) is
   used as a standard/definitional structural fact about the crossed-product construction (verified
   concretely for the covariant representation in the driver's Section 2, via the explicit
   `π(u)π(x)π(u)^{-1}=π(α(x))` covariance check), but is **not independently re-derived from first
   principles for the infinite-dimensional `M=L(F_inv(6))`** the way D0-iii re-derived the
   `ℓ²(G)`-Fourier identity from scratch. This is the one place a verifier should look hardest:
   is there any subtlety in the infinite-dimensional crossed product (e.g. involving the *type* of
   completion — algebraic vs. von Neumann crossed product) that could weaken "unique expansion" for
   `M=L(F_inv(6))` specifically? I believe not (this is completely standard for crossed products by a
   finite group, appearing in every treatment, e.g. via the conditional expectation `E:M⋊G→M`,
   `E(Σx_nu^n)=x_0`, being well-defined and faithful) — but I have not personally re-derived the
   *general* finite-group-crossed-product uniqueness theorem from the von Neumann algebra axioms here,
   only verified it concretely in the driver's finite covariant representation.
2. **The rank-1-bimodule / index-`|G|` claim (§2.3)** is stated as the standard content of the
   free-action Galois theorem (the same GHJ/Connes–Takesaki citation family D0-iv already invoked for
   the isomorphism itself) rather than independently proven here from bimodule first principles. The
   §1 relative-commutant proof (fully self-contained) is the load-bearing rigor for the **verdict**
   (no moduli); §2.3's bimodule-rank language is the **explanation/pinpoint** the freeze specifically
   asked for, and is standard but cited rather than re-derived line-by-line.
3. **The dimension-mismatch witness (§2.4, driver Section 3)** is a finite-toy illustration, not a
   proof about the infinite-dimensional case — flagged as such in both the driver and here.
4. **The §2 "no-moduli" pinpoint silently assumes `M^α` is a factor** (surfaced by the verification
   2026-07-15). The construction of a grade-mixing element in the commutant of `u` is blocked precisely
   because `M^α` has trivial center — i.e. `M^α` is itself a II₁ factor, which is standard for a *free*
   finite-group action on a factor (the fixed-point algebra of an outer action is a subfactor). This is
   an unstated dependency of the §2.3 *explanation*, NOT of the §1 verdict-bearing proof (which needs
   only `M' ∩ (M⋊ℤ₃) = ℂ`, established self-contained). Named here so the dependency is explicit; it
   holds for our free ℤ₃ action by D0-iv.

None of these are "gaps that could flip the verdict" as far as I can tell — they are standard
structural facts about crossed products, used the same way D0 used (and the verifier approved)
citations for the Galois theorem itself. But per the goal-seek/honesty rail, they are named explicitly
rather than presented as fully re-derived, so a verifier knows exactly where to press.

---

## Regression anchor

`proofs/foundations/genident_D1_canonical_home_check_2026-07-15.py` — 33/33 PASS, < 2s, exact sympy
algebraic arithmetic (no floating point). Anchors: Section 1 (matrix-unit closure with a genuinely
nontrivial `M^α` leg, dim 2, plus an honest computational flag that the freeze's literal
`e=(1/3)(1+u+u²)` shorthand does not itself work as the matrix-unit seed — a precision correction,
not a verdict-affecting defect); Section 2 (the decisive contrast: the finite/inner `M_2(ℂ)` toy
concretely exhibits a non-scalar relative-commutant element `Y=π(u0^{-1})π(u)`, with a genuine
negative control confirming the mechanism discriminates, plus the `y_0`-alone check); Section 3
(the `12≠18` dimension-mismatch witness for why outerness is structurally necessary, plus the AST
goal-seek self-scan). `the_run.py`/Layer-1 untouched. Not wired into `verify.py` (matches D0's and
GEN-IDENT-C's own precedent). No accretion into `the_net.py` §11 — per the freeze's own artifact map,
accretion happens for a numeric/coupling construction; D1 is a structural/rigidity result about `M`,
not a numeric read, and D2 (the genuinely new cross-level construction) is where accretion would
first become relevant, if it runs.
