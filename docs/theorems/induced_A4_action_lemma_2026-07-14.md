# The induced-A4-action lemma — dart_rep's right-regular symmetry on the multiplicity space

**Date:** 2026-07-14. **Scope:** upgrades the group-theoretic backbone of
internal research notes (T1/T2/T3) from
**MACHINE-VERIFIED** to **PROVEN**, for the parts of that station's claim that admit a clean
analytic proof, and states precisely — without overclaiming — the one part (the triad-orbit claim)
that does **not** admit one and must remain machine-verified only. **Read-only**: no existing
document section is modified, nothing is wired into `verify.py`, no selector claim is made, no new
construction is added to `derivation_topdown/state/the_net.py`. `OMP_NUM_THREADS=4`; the
verification script below runs in ~10s.

**Verification:** `induced_A4_lemma_check.py` (inlined in full in the Appendix; also preserved at
`/tmp/an assistant-1000/-home-adam-projects-standard-model-derivation/2654b83d-2deb-4bb8-bc64-565ad56b5385/scratchpad/induced_A4_lemma_check.py`)
— every claim below marked PROVEN is checked to machine floor (`~1e-15`–`1e-16`, i.e. floating-point
round-off, not an approximation of a nonzero discrepancy).

---

## 0. Setup — the objects, verbatim from the construction

All objects below are the module's own, unmodified:

- `NV = srs.NV = 4` (`the_net.py:48`, `derivation_topdown/dirac_srs_mdl/srs.py:11`): the graph is
  **K₄**, the complete graph on 4 vertices — `EDGES` (`srs.py:14`) lists **all** `C(4,2) = 6` pairs,
  so every permutation of the 4 vertex labels automatically sends edges to edges.
- `ND = 12` (`the_net.py:51`): darts = ordered pairs `(i,j)`, `i≠j`, of the 4 vertices — the 2
  orientations of each of the 6 edges.
- `A4v = _a4_vertex_group()` (`the_net.py:1216`): the 12 **even** permutations of `{0,1,2,3}`, as
  dicts, i.e. `A4` realized as vertex permutations of `K₄`. Group law used throughout the module:
  `comp(g,h) := {i : g[h[i]]}`, i.e. `comp(g,h) = g∘h` (apply `h` first, then `g`) — ordinary
  function composition.
- `dart_rep(sig)` (`the_net.py:1223`): the induced action of a vertex permutation `sig` on the 12
  darts, `(i,j) ↦ (sig[i], sig[j])` (well-defined for every `sig`, precisely because `K₄` has every
  pair as an edge).
- `rho3 = _a4_standard_3irrep()[1]` (`the_net.py:5016`): `A4`'s honest (non-projective) 3-dim irrep,
  built as the sum-zero subspace of the 4-vertex permutation representation, with an orthonormal
  basis (via `np.linalg.qr`) — so `rho3(g)` is **real orthogonal** for every `g` by construction
  (a permutation matrix compressed onto an orthonormal-basis invariant subspace).
- `phi_basis = _a2d_abstract_hom_basis()[1]` (`the_net.py:5952`): 3 complex `(3,ND)` matrices
  `phi_1,phi_2,phi_3` spanning `Hom_A4(dart_rep, rho3) := {φ : C^ND → C^3 | φ·dart_rep(g) = rho3(g)·φ
  ∀g}`, obtained as the (unit-norm, mutually orthogonal — see §3) null-space basis of an SVD. This
  **is** the multiplicity space: `u ∈ C^3 ↦ Φ_u := Σᵢ uᵢ φᵢ` (`w2_family_phi_d`, `the_net.py:6985`,
  via the `(ND,3)`-transposed convention `_a2d_phi1_list`, `the_net.py:6095`) is exactly the
  coordinatization `w2_family_direction` (`the_net.py:6996`) uses everywhere downstream in the V1
  station.

**The question this lemma answers analytically:** does `A4` act on the 3-dim `u`-space itself (not
just on the 12-dim dart domain), compatibly with the equivariance defining that space — and if so,
which representation is it, exactly?

---

## 1. L1 — `dart_rep` is the left regular representation of `A4` (simply transitive)

**Claim.** `dart_rep: A4 → GL(12,ℝ)` is an honest group homomorphism
(`dart_rep(comp(g,h)) = dart_rep(g)·dart_rep(h)`), and `A4` acts **simply transitively** on the 12
darts (a single orbit, trivial stabilizers) — i.e. `dart_rep` is (isomorphic to) the left regular
representation of `A4` on `C[A4] ≅ C^12`.

**Proof.**

*Homomorphism.* `dart_rep(sig)` relabels each dart `(i,j) ↦ (sig[i],sig[j])`. Relabeling by `h` then
by `g` sends `(i,j) ↦ (h[i],h[j]) ↦ (g[h[i]],g[h[j]]) = (comp(g,h)[i], comp(g,h)[j])` — the
composite relabeling `comp(g,h)`. This is immediate from the definition, not a numerical
coincidence; the driver re-confirms it over all `12×12 = 144` pairs at residual **exactly `0.0`**
(these are `0/1` permutation matrices — the identity is exact, not floor-limited).

*Simple transitivity — a self-contained combinatorial argument.* Darts are ordered pairs `(i,j)`,
`i≠j`, from a 4-element set — there are `4·3 = 12` of them, matching `|A4| = 12`. It suffices to
show the evaluation map `A4 → darts`, `sig ↦ (sig[0], sig[1])`, is **injective** (bijective by equal
cardinality; then orbit–stabilizer forces the stabilizer of every dart to be trivial, giving free +
transitive = simply transitive on *every* dart, not just the basepoint).

Suppose `sig, sig' ∈ A4` agree on the pair: `sig[0]=sig'[0]`, `sig[1]=sig'[1]`. Let
`t := sig⁻¹∘sig'`; then `t` fixes `0` and `1`, so `t` restricted to `{2,3}` is either the identity or
the transposition `(2 3)`. But `(2 3)` alone is an **odd** permutation of `{0,1,2,3}`, while `A4`
(closed under products and inverses) forces `t = sig⁻¹∘sig' ∈ A4` to be **even**. The only even
option fixing `{0,1}` is `t = \mathrm{id}`, so `sig = sig'`. Injectivity, hence bijectivity, hence
simple transitivity.

(Numerically reconfirmed as a sanity check, not as the proof: the driver verifies `sig↦(sig[0],sig[1])`
hits 12 distinct pairs, and separately that the orbit map `g ↦ dart_rep(g)·e_{d0}` for the basepoint
dart `d0=0` is a bijection onto all 12 darts.)

Also reconfirmed: the order histogram of `A4v` is `{identity: 1, order-2: 3, order-3: 8}` — the
correct conjugacy structure of `A4` (3 double-transpositions, 8 three-cycles).

**L1 status: PROVEN.** (Machine check: homomorphism residual `0.0` exactly; bijectivity of the
orbit map; order histogram exact match.)

---

## 2. L2 — the right translation `R_h` commutes with `dart_rep(g)` (pure associativity)

Because `dart_rep` is simply transitive (L1), fix the basepoint dart `d0=0` and identify
`A4 ↔ darts` via `g ↔ \mathrm{dart}(g) := \mathrm{dart\_rep}(g)·e_{d0}` — a genuine bijection.
Define, for each `h ∈ A4`, the **right translation**
$$R_h(\mathrm{dart}(g)) := \mathrm{dart}(\mathrm{comp}(g,h)) = \mathrm{dart}(g h)$$
(shorthand `gh := \mathrm{comp}(g,h)`), realized as a `12×12` permutation matrix (`build_Rh` in the
driver — this is exactly the map named `R_h` in the prior station's report).

**Claim.** `dart_rep(g')·R_h = R_h·dart_rep(g')` for **all** `g', h ∈ A4` — an exact identity, not
an approximate one.

**Proof.** Both sides are permutations of the 12 darts; evaluate on `dart(g)`:
$$
\mathrm{dart\_rep}(g')\big(R_h(\mathrm{dart}(g))\big) = \mathrm{dart\_rep}(g')(\mathrm{dart}(gh)) = \mathrm{dart}(g'(gh))
$$
$$
R_h\big(\mathrm{dart\_rep}(g')(\mathrm{dart}(g))\big) = R_h(\mathrm{dart}(g'g)) = \mathrm{dart}((g'g)h).
$$
These agree because `comp` is **associative** — it is literal function composition of vertex
permutations, `comp(comp(g',g),h) = comp(g', comp(g,h))` (both sides send `i ↦ g'[g[h[i]]]`). This
is the *only* fact used: the commutation of the left action with right translation is exactly the
statement that function composition is associative. Machine-checked: worst residual over all
`144` `(g,h)` pairs is **`0.0` exactly** (again `0/1` permutation-matrix identities).

**Bonus fact used later (L3).** `R_h` is itself the **right-regular anti-representation**:
`R_h∘R_{h'} = R_{\mathrm{comp}(h',h)}` (order reversed — a right action, verified over all `144`
pairs, residual `0.0`). This is the source of the "double contravariance" that turns the induced
action on the Hom space (L3) back into a genuine (non-anti) representation.

**L2 status: PROVEN.**

---

## 3. L3 — precomposition with `R_h` gives an exact representation `M ≅ ρ₃` on the multiplicity space

**Claim.** For `φ ∈ \mathrm{Hom}_{A4}(\mathrm{dart\_rep}, ρ₃)`, the precomposition `φ∘R_h` (matrix
product `φ·R_h`) is again in `\mathrm{Hom}_{A4}(\mathrm{dart\_rep}, ρ₃)`; expanding in the basis
`{φ₁,φ₂,φ₃}` defines a `3×3` matrix `M(h)` by `φᵢ∘R_h = Σⱼ M(h)ⱼᵢ φⱼ`; `h ↦ M(h)` is a genuine
representation of `A4` (`M(\mathrm{comp}(a,b)) = M(a)M(b)`); `M(h)` is **real orthogonal** for every
`h`; and `M` is related to `ρ₃` by an **exact, explicit** conjugation `M(h) = S^{-1}ρ₃(h)S` — not
merely "isomorphic to" it, and not its dual/conjugate.

### 3a. Closure of the Hom space under precomposition

For `φ ∈ \mathrm{Hom}_{A4}(\mathrm{dart\_rep},ρ₃)` and any `g' ∈ A4`:
$$
(φ∘R_h)(\mathrm{dart\_rep}(g')x) = φ\big(R_h(\mathrm{dart\_rep}(g')x)\big) = φ\big(\mathrm{dart\_rep}(g')(R_h x)\big)\quad\text{[L2]}
$$
$$
= ρ₃(g')·φ(R_h x) = ρ₃(g')·(φ∘R_h)(x)\quad\text{[φ's own equivariance]}.
$$
So `φ∘R_h` satisfies the same equivariance law — it lies in the same 3-dim Hom space. This is why
solving `φᵢ∘R_h = Σⱼ M(h)ⱼᵢ φⱼ` by least squares (as the source station did) is not really a fit: in
exact arithmetic the residual is *zero*, and the observed `≈4.9e-16` (reconfirmed here) is pure
floating-point round-off, not evidence of an approximate closure.

### 3b. `M` is a genuine homomorphism (not an anti-homomorphism)

Write `T(h): φ ↦ φ∘R_h` for the linear operator on the Hom space, so `T(h)Φ_c = Φ_{M(h)c}` for
`Φ_c := Σᵢcᵢφᵢ` (immediate from the definition of `M(h)`). Using `R_h∘R_{h'} = R_{\mathrm{comp}(h',h)}`
(L2's bonus fact):
$$
T(h')\big(T(h)φ\big) = (φ∘R_h)∘R_{h'} = φ∘(R_h∘R_{h'}) = φ∘R_{\mathrm{comp}(h',h)} = T(\mathrm{comp}(h',h))\,φ.
$$
In coordinates this reads `Φ_{M(h')M(h)c} = Φ_{M(\mathrm{comp}(h',h))c}` for all `c`, i.e.
$$
M(h')M(h) = M(\mathrm{comp}(h',h)) \quad\Longleftrightarrow\quad M(\mathrm{comp}(a,b)) = M(a)M(b)
$$
(set `a:=h'`, `b:=h`) — **exactly the same composition convention `dart_rep` and `ρ₃` use**, i.e. a
genuine left representation, not its opposite. The two contravariances (precomposition reverses
order once; `R_h` being a right/anti-action reverses it again) cancel exactly. Machine-checked over
all `144` pairs `(a,b)`: `M(\mathrm{comp}(a,b)) - M(a)M(b)` has worst entry `~1e-15`.

### 3c. `M(h)` is orthogonal (Frobenius-inner-product argument, not numerology)

Equip the Hom space with the Frobenius/Hilbert–Schmidt inner product on `(3,ND)` matrices,
`⟨A,B⟩ := \mathrm{tr}(AB^\dagger)`. Since `phi_basis` is the null-space basis of an SVD, its
elements are **orthonormal** in the standard complex inner product on the vectorized `(3·ND)`-space
— reshaping preserves the inner product, so `⟨φᵢ,φⱼ⟩ = δᵢⱼ` **exactly** (reconfirmed: Gram matrix
residual `~7e-16`). For any orthogonal `R_h` (a real permutation matrix, `R_hR_h^T=I`):
$$
⟨φ∘R_h,\ φ'∘R_h⟩ = \mathrm{tr}\big(φR_h(φ'R_h)^\dagger\big) = \mathrm{tr}\big(φR_hR_h^Tφ'^\dagger\big) = \mathrm{tr}(φφ'^\dagger) = ⟨φ,φ'⟩.
$$
So precomposition by `R_h` is an **isometry** of the Hom space in this inner product. Expressed in
the *orthonormal* basis `{φᵢ}`, an isometry is represented by a unitary matrix — here real, so
`M(h)` is **real orthogonal**: `M(h)^TM(h)=I`. Machine-checked over all 12 `h`: worst residual
`~1.3e-15`; `M(h)` found real (imaginary part exactly `0.0`) for every `h`.

This is the clean resolution of the caution raised against this lemma: `M`'s orthogonality is not an
accident visible only because `ρ₃` happens to be self-dual — it follows **directly** from
`phi_basis` orthonormality (an SVD guarantee) plus `R_h` orthogonality (a permutation-matrix
guarantee), independent of any property of `ρ₃` at all.

### 3d. Exact identification `M(h) = S⁻¹ρ₃(h)S` — not the dual, not merely isomorphic

The general fact for a regular representation: for `Φ_c := Σᵢcᵢφᵢ ∈ \mathrm{Hom}_{A4}(\mathrm{dart\_rep},ρ₃)`
and any `g`, evaluating the equivariance law at the basepoint dart, `Φ_c(\mathrm{dart}(g)) =
Φ_c(\mathrm{dart\_rep}(g)e_{d0}) = ρ₃(g)·Φ_c(e_{d0})`. So every element of the Hom space is
determined **entirely** by its value at the basepoint, `v(c) := Φ_c(e_{d0}) = S c`, where
`S_{:,i} := φᵢ(e_{d0}) = φᵢ[:,d0]` (a `3×3` matrix built directly from `phi_basis`, no group
averaging). Evaluating `T(h)Φ_c = Φ_{M(h)c}` at the basepoint two ways:
$$
\big(T(h)Φ_c\big)(e_{d0}) = Φ_c(R_h e_{d0}) = Φ_c(\mathrm{dart}(h)) = ρ₃(h)\,v(c) = ρ₃(h)\,Sc,
$$
$$
\big(T(h)Φ_c\big)(e_{d0}) = Φ_{M(h)c}(e_{d0}) = S\,(M(h)c).
$$
Equating for all `c`: `S\,M(h) = ρ₃(h)\,S`, i.e.
$$
\boxed{M(h) = S^{-1}ρ₃(h)\,S \quad\text{EXACTLY, for every } h \in A4.}
$$
This is the **explicit** intertwiner promised in the task: it is manifestly `ρ₃` itself (not `ρ₃*`
or `\overline{ρ₃}`) that appears, because the evaluation-at-basepoint argument never invokes duality
or complex conjugation — the "double contravariance" of §3b already absorbed the only place a dual
could have entered. (Since `ρ₃` happens to be real orthogonal, i.e. self-dual, this distinction would
have been numerically invisible either way — which is exactly why the derivation, not the numerics,
is what settles it.) Machine-checked: `\max_h |M(h) - S^{-1}ρ₃(h)S|` over all 12 `h` is **`~1e-15`**.

**Bonus exact fact (not required, but explains a clean number the driver produced without
comment):** `S^\dagger S = \tfrac{1}{12}I_3` **exactly** (`cond(S)=1.0000000000000007`, i.e. `S` is
`1/\sqrt{12}` times an honest orthogonal matrix). This follows from the same orthogonality argument
as §3c: `⟨φᵢ,φⱼ⟩ = Σ_{g∈A4} ⟨φᵢ(\mathrm{dart}(g)),φⱼ(\mathrm{dart}(g))⟩ = Σ_g ⟨ρ₃(g)v_i,ρ₃(g)v_j⟩ =
12⟨v_i,v_j⟩` (`ρ₃(g)` orthogonal cancels the sum to `12×` the summand), and `⟨φᵢ,φⱼ⟩=δᵢⱼ`, so
`⟨v_i,v_j⟩ = δᵢⱼ/12` exactly — i.e. `S^\dagger S = I/12`. `|A4|=12` appearing here is exactly the
expected regular-representation normalization (Peter–Weyl bookkeeping), confirming the whole
identification is self-consistent, not merely numerically close.

**L3 status: PROVEN** — closure, homomorphism law, orthogonality, and the exact conjugation
`M(h)=S^{-1}ρ₃(h)S` are all established analytically and confirmed to machine floor.

---

## 4. L4 — the covariance identity `Φ_{M(h)u} = Φ_u∘R_h`

**Claim.** For every `u ∈ C^3` and every `h ∈ A4`, `Φ_{M(h)u} = Φ_u∘R_h` (matrix form:
`Φ_{M(h)u} = Φ_u·R_h`).

**Proof.** This is definitionally §3b's `T(h)Φ_u = Φ_{M(h)u}` together with `T(h)Φ_u := Φ_u∘R_h`
(the very definition of `T(h)`) — no new content beyond L3, restated in the `u`-coordinate language
`w2_family_direction`/`w2_family_phi_d` use. Machine-checked directly (not merely inherited) on 6
random complex `u` against all 12 `h`: worst residual `~1.7e-15`.

**Consequently:** the 3-dim multiplicity space carries an **exact construction symmetry**
`M: A4 → O(3)`, conjugate to `ρ₃` via the explicit `S` of §3d — this is a genuine, provable fact
about the construction, independent of any numerical triad.

**What L4 does NOT say:** `R_h` is **not** a symmetry of the walk structure the physical
functionals `c₂,c₃,F1` are built from — reconfirmed here (as in the source station):
`\max_h|B_0 R_h - R_h B_0| = 1.0` and `\max_h|R_{\mathrm{rev}}R_h - R_hR_{\mathrm{rev}}| = 1.0`
(hashimoto walk `B_0` and reversal `R_{\mathrm{rev}}`), both **far** from zero, a genuine
non-commutation. So `M` is a real, exact symmetry of the *multiplicity space's own linear-algebra
structure* (§1–§3), but it is **not** accompanied by any compensating symmetry of the *walk-sum*
structure — this is why `c₂,c₃,F1` are not `M`-invariant (§5), and it is not a gap in this lemma's
proof; it is a true structural fact about which part of the construction `M` lives in.

**L4 status: PROVEN.**

---

## 5. L5 — the triad: what is provably true, and what is honestly still open

The source station (`V1_gapB2_W_A4_check_2026-07-14.md`, T2) found `M(A4v[5]) = W` and
`M(A4v[9]) = W^2 = W^T` to `7.686301e-08` (the Newton-polish/orthogonality floor), where `W` is the
empirically-discovered order-3 rotation carrying `u_A ↦ u_B ↦ u_C ↦ u_A` (the triad at
`c₂=1/6, c₃=1/72`). L1–L4 above **prove** that `M(A4v[5])` is a genuine, exact `A4` construction
element (order exactly 3, real orthogonal, conjugate to `ρ₃` via `S`) — so the *existence* of a
matching construction symmetry is no longer merely observed; it is derived. What remains to assess
is whether the **specific triad** is provably an orbit of this symmetry, or only observed to be one.

**PROVEN (cheap, general, from L3's orthogonality alone):**
- `M(h)` maps the real unit sphere in `ℝ³` to itself (trivial: real + orthogonal). Reconfirmed
  numerically (imaginary part exactly `0.0`, norm preserved to `~1e-16`).
- `M(A4v[5])^3 = I` exactly (order 3, inherited from `A4v[5]` having order 3 and `M` being an honest
  homomorphism, L3b) — reconfirmed, residual `4.4e-16`.
- **A genuine, if limited, invariance is available**: for *any* scalar function `f(u)` and the
  order-3 element `m := A4v[5]`, the symmetrized product
  `G(u) := \big(f(u)-f_0\big)\big(f(Mu)-f_0\big)\big(f(M^2u)-f_0\big)` (for any target value `f_0`)
  is **exactly `M`-invariant as a function of `u`**: `G(Mu)=G(u)`, because `M^3=I` merely cyclically
  permutes the three factors. This holds for `f=c₂` (or `c₃`) and reconfirmed numerically to
  `~7e-21` on a random direction. **This is the correct, honest form of "the system is
  `M`-invariant"** — not `c₂` itself (which genuinely is not `M`-invariant, reconfirmed below), but
  the *cyclically symmetrized combination* built from it.

**NOT PROVEN — confirmed to genuinely fail, not merely unchecked (reconfirmed here, matching the
source station's own T3 finding):**
- `c₂(u)` and `c₃(u)` are **not** individually `M`-invariant: `c₂(v) ≠ c₂(M(5)v)` and
  `c₃(v)≠c₃(M(5)v)` for a generic random `v` (differences `O(10^{-3})`–`O(1)`, far above floor,
  reconfirmed here independently). So the naive argument *"the defining system `{c₂=1/6,c₃=1/72}`
  is `M`-invariant, hence its zero set is permuted by `M`"* is **not available** — the system as
  literally written (individual `c₂`, `c₃` conditions) is not `M`-invariant, only the symmetrized
  triple-product `G` built from it is, and `G(u)=0` is a *weaker*, three-times-degenerate condition
  than the actual system solved (which pins each of the three triad points independently via Newton
  polish from three different starting guesses).

**Consequently, the following remains open — MACHINE-VERIFIED ONLY, not analytically derived here
or anywhere on record:**

> That the **specific** triad `{u_A,u_B,u_C}` — three points independently Newton-polished from
> three unrelated starting guesses to satisfy `c₂=1/6, c₃=1/72` — coincides exactly with the orbit
> `{v₀, M(5)v₀, M(9)v₀}` for some `v₀` (equivalently: that `W = M(A4v[5])` exactly, not merely to
> `7.7e-8`, and that this identity forces the triad to be a single `M`-orbit rather than three
> independent points that merely happen to be permuted by a symmetry of the ambient space). No
> mechanism is known that would derive this from the `{c₂=1/6,c₃=1/72}` system itself, since that
> system is not `M`-invariant (previous paragraph). It is recorded here, exactly as the source
> station recorded it, as an exact-to-floor **empirical coincidence** (`7.686301e-08`, the same
> order as the construction's own orthogonality floor) — real, reproducible, non-vacuous (it rejects
> 6 of 8 order-3 candidates), but not a corollary of L1–L4.

This honesty boundary is not a weakness introduced by this lemma — if anything, L1–L4 **sharpen**
what is missing: before this lemma, one could imagine the triad-orbit fact might follow from some
unidentified joint symmetry of the whole construction (dart space *and* multiplicity space
together). L1–L4 rule that out precisely: the **only** `A4`-action on the multiplicity space that
exists in this construction is `M` (derived uniquely from `dart_rep`'s own regular-representation
structure — there is no other candidate to invoke), and `M` is now proven not to preserve `c₂,c₃`
individually. So the triad-orbit fact, if true beyond floor, is a fact about the specific rational
point `c₂=1/6,c₃=1/72` and the specific triad — not a generic consequence of any symmetry available
in this construction. Also unresolved (identified, not newly solved, by the source station's T3):
why `F2/F3` specifically **are** insensitive to `M` while `c₂,c₃,F1` are not.

**L5 status: PARTIAL.** PROVEN: sphere-preservation, exact order-3, and the correct (symmetrized)
form of the invariance available from `M`'s existence. OPEN (machine-verified only, `~1e-8`, not
analytically derivable from what is proven): the specific triad being an exact `M`-orbit.

---

## 6. Honesty status table

| # | claim | status | evidence |
|---|---|---|---|
| L1 | `dart_rep` is a homomorphism | **PROVEN** | direct from definition (relabeling is functorial); machine residual `0.0` exactly, 144 pairs |
| L1 | `A4` acts simply transitively on the 12 darts | **PROVEN** | elementary injectivity argument (§1) using only "`A4` = even permutations of 4 points"; machine-confirmed bijectivity |
| L2 | `[\mathrm{dart\_rep}(g), R_h] = 0` for all `g,h` | **PROVEN** | pure associativity of function composition; machine residual `0.0` exactly, 144 pairs |
| L2 | `R_h` is the right-regular anti-representation | **PROVEN** | direct computation; machine residual `0.0` exactly |
| L3 | precomposition with `R_h` preserves `\mathrm{Hom}_{A4}(\mathrm{dart\_rep},ρ₃)` | **PROVEN** | direct algebra (§3a); machine residual `~5e-16` |
| L3 | `h ↦ M(h)` is a genuine representation (`M(\mathrm{comp}(a,b))=M(a)M(b)`) | **PROVEN** | double-contravariance cancellation (§3b); machine residual `~1e-15`, 144 pairs |
| L3 | `M(h)` real orthogonal for every `h` | **PROVEN** | Frobenius-inner-product isometry argument (§3c), independent of `ρ₃`'s self-duality | machine residual `~1.3e-15` |
| L3 | `M(h) = S^{-1}ρ₃(h)S` exactly (not the dual) | **PROVEN** | basepoint-evaluation argument (§3d), explicit `S` | machine residual `~1e-15` |
| L3 | `S^\dagger S = I/12` (bonus, explains the driver's un-flagged `cond(S)=1`) | **PROVEN** | Peter–Weyl-style averaging identity (§3d bonus) | machine residual exact to display precision |
| L4 | `Φ_{M(h)u} = Φ_u \circ R_h` | **PROVEN** | restatement of §3b in `u`-coordinates | machine residual `~1.7e-15` |
| L4 | `R_h` does **not** commute with the walk (`B_0`, `reversal`) | **PROVEN** (a genuine non-symmetry, not a gap) | direct computation, residual `1.0` (not floor-limited) |
| L5 | `M(h)` preserves the real unit sphere | **PROVEN** | trivial corollary of orthogonality + realness |
| L5 | `M(A4v[5])^3 = I` | **PROVEN** | corollary of L3's homomorphism property |
| L5 | the symmetrized triple product `G(u)` built from `c₂` (or `c₃`) is `M`-invariant | **PROVEN** | `M^3=I` cyclically permutes 3 factors (§5) |
| L5 | `c₂`, `c₃` individually `M`-invariant | **FALSE** (reconfirmed) | direct numeric counterexample, `O(10^{-3})`–`O(1)` |
| L5 | the **specific** triad `{u_A,u_B,u_C}` is exactly the orbit `{v₀,Mv₀,M²v₀}` | **OPEN / MACHINE-VERIFIED ONLY** | Gap-B2's own `7.686301e-08` match; no analytic mechanism identified; explicitly NOT a corollary of L1–L4 |
| — | why `F2,F3` specifically are `M`-insensitive while `c₂,c₃,F1` are not | **OPEN**, unchanged from the source station | not addressed by this lemma either |

---

## Appendix — verification script (`induced_A4_lemma_check.py`)

Reproduces every residual quoted above. `OMP_NUM_THREADS=4`; runtime ≈10s. Imports `the_net.py`
read-only.

```python
"""
Induced A4 action lemma (L1-L5), verification driver.
Upgrades internal research notes from
MACHINE-VERIFIED to PROVEN for L1-L4, and nails down precisely what remains
open in L5.  Read-only; imports the_net.py, no wiring, no verify.py edit.
OMP_NUM_THREADS=4.  Runtime ~10-15s.
"""
import sys, os, math, itertools
sys.path.insert(0, ".")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

from derivation_topdown.state.the_net import (
    w2_family_direction, w2_gamma_table, _v1_gamma_mode_table, v1_F2_F3,
    _a2d_abstract_hom_basis, _a4_vertex_group, _a4_standard_3irrep, dart_rep,
    reversal, hashimoto_gamma, NV, ND,
)

np.set_printoptions(precision=6, suppress=False, linewidth=140)


def comp(g, h):
    """A4 group law used throughout the_net.py: comp(g,h) = g o h (apply h then g)."""
    return {i: g[h[i]] for i in range(NV)}


# ===========================================================================
# L1 -- dart_rep is the left regular representation of A4 (simply transitive)
# ===========================================================================
print("=" * 78)
print("L1 -- dart_rep = left regular rep of A4 (simply transitive on 12 darts)")
print("=" * 78)

A4v = _a4_vertex_group()
assert len(A4v) == 12, f"|A4| = {len(A4v)} != 12"
e_id = {i: i for i in range(NV)}
print(f"NV={NV}, ND={ND}, |A4|={len(A4v)}")

# --- (a) ANALYTIC: A4 acts simply transitively on ordered pairs (i,j), i!=j,
# hence on darts (each dart IS such a pair). Proof: the map A4 -> ordered pairs,
# sig -> (sig[0], sig[1]), is injective. If sig, sig' in A4 agree at 0 and 1,
# then t := sig^{-1} sig' fixes 0 and 1, so t restricted to {2,3} is either the
# identity or the transposition (2 3). But (2 3) alone is an ODD permutation of
# {0,1,2,3}, and A4 is a group of EVEN permutations closed under inverses/
# products, so t in A4 forces t even; the only even option fixing {0,1} is
# t = identity. Hence sig = sig'. Injective + |A4| = |ordered pairs| = 12 =>
# bijective => the stabilizer of (0,1) is trivial and A4 is (sharply)
# transitive on the 12 ordered pairs -- i.e. simply transitive on darts.
def perm_sign_even(sig):
    inv = 0
    for i in range(NV):
        for j in range(i + 1, NV):
            if sig[i] > sig[j]:
                inv += 1
    return inv % 2 == 0


injective_ok = True
seen_pairs = set()
for sig in A4v:
    pair = (sig[0], sig[1])
    if pair in seen_pairs:
        injective_ok = False
    seen_pairs.add(pair)
print("hand-argument check: sig -> (sig[0],sig[1]) injective over A4:", injective_ok,
      "  (distinct pairs seen:", len(seen_pairs), "/ 12 )")

# --- (b) numeric confirmation on dart_rep itself: orbit map of the basepoint
# dart d0=0 is a bijection A4 -> {0,...,11}.
d0 = 0
D_of = {}
for k, g in enumerate(A4v):
    col = dart_rep(g)[:, d0]
    assert np.sum(np.abs(col) > 0.5) == 1, "dart_rep(g) column not a single 1 (not a permutation)"
    D_of[k] = int(np.argmax(col))
simply_transitive = len(set(D_of.values())) == 12
print("orbit map g -> dart_rep(g).e_{d0} bijective onto 12 darts:", simply_transitive)

# --- (c) dart_rep is a genuine homomorphism (not merely a set of permutations):
# this follows directly from the DEFINITION (relabeling vertices under sig is
# functorial: applying h then g relabels exactly as applying comp(g,h)), and is
# reconfirmed numerically over all 144 ordered pairs.
worst_hom = 0.0
for g in A4v:
    for h in A4v:
        worst_hom = max(worst_hom, float(np.max(np.abs(dart_rep(comp(g, h)) - dart_rep(g) @ dart_rep(h)))))
print("dart_rep(comp(g,h)) == dart_rep(g) @ dart_rep(h), worst over 144 pairs:", worst_hom)

# order histogram sanity (A4 conjugacy structure: 1 + 3 + 8)
def perm_order(g):
    cur = dict(g)
    n = 1
    while cur != e_id:
        cur = comp(g, cur)
        n += 1
    return n


orders = [perm_order(g) for g in A4v]
order_hist = {o: orders.count(o) for o in sorted(set(orders))}
print("order histogram:", order_hist)
order3_idx = [k for k, o in enumerate(orders) if o == 3]

L1_pass = injective_ok and simply_transitive and worst_hom < 1e-12 and order_hist == {1: 1, 2: 3, 3: 8}
print("L1 VERDICT:", "PROVEN" if L1_pass else "FAIL")

# ===========================================================================
# L2 -- the right translation R_h commutes with dart_rep(g): pure associativity
# ===========================================================================
print()
print("=" * 78)
print("L2 -- R_h commutes with dart_rep(g) for all g,h (associativity)")
print("=" * 78)

key_to_idx = {tuple(sorted(g.items())): k for k, g in enumerate(A4v)}


def idx_of(g):
    return key_to_idx[tuple(sorted(g.items()))]


def build_Rh(hk):
    """R_h: dart(g) -> dart(comp(g,h)) = dart(g.h) in the basepoint identification
    dart(g) := dart_rep(g).e_{d0}.  Well-defined because dart_rep is simply
    transitive (L1): the map g -> dart(g) is a bijection A4 -> darts."""
    Rh = np.zeros((ND, ND))
    h = A4v[hk]
    for k, g in enumerate(A4v):
        gh_idx = idx_of(comp(g, h))
        Rh[D_of[gh_idx], D_of[k]] = 1.0
    return Rh


Rhs = [build_Rh(hk) for hk in range(12)]

# ANALYTIC: dart_rep(g') R_h (dart(g)) = dart_rep(g') dart(g.h) = dart(g'.(g.h))
#         = dart((g'.g).h) [associativity of function composition `comp`]
#         = R_h (dart(g'.g)) = R_h dart_rep(g') (dart(g))
# so dart_rep(g') R_h = R_h dart_rep(g') for ALL g', h -- an EXACT permutation
# identity (both sides are 0/1 matrices), not merely floor-small.
assoc_ok = True
for g, h1 in itertools.product(A4v, A4v):
    for h2 in A4v:
        if comp(comp(g, h1), h2) != comp(g, comp(h1, h2)):
            assoc_ok = False
print("comp() itself associative over a spot-check of 12^3 triples:", assoc_ok)

worst_comm = max(float(np.max(np.abs(dart_rep(g) @ Rhs[hk] - Rhs[hk] @ dart_rep(g))))
                  for hk in range(12) for g in A4v)
print("[dart_rep(g), R_h] = 0 numerically, worst over 144 pairs:", worst_comm)

# R_h itself is a genuine ANTI-representation of A4 (R_h R_h' = R_{h'.h}), the
# right-regular action -- verify directly (needed for L3's homomorphism proof).
worst_anti = 0.0
for hk1 in range(12):
    for hk2 in range(12):
        h1h2 = idx_of(comp(A4v[hk2], A4v[hk1]))  # comp(h2,h1) = h2 o h1 -> "h1 then h2" NOT what we want
        # we want R_{h1} R_{h2} == R_{comp(h2,h1)} per the derivation R_h R_h' = R_{h'h}
        # (setting h := hk1, h' := hk2): R_{hk1} @ R_{hk2} should equal R_{comp(hk2,hk1)}
        target = Rhs[idx_of(comp(A4v[hk2], A4v[hk1]))]
        worst_anti = max(worst_anti, float(np.max(np.abs(Rhs[hk1] @ Rhs[hk2] - target))))
print("R_h R_h' == R_{comp(h',h)} (right-regular anti-homomorphism law), worst:", worst_anti)

L2_pass = assoc_ok and worst_comm < 1e-12 and worst_anti < 1e-9
print("L2 VERDICT:", "PROVEN" if L2_pass else "FAIL")

# ===========================================================================
# L3 -- precomposition with R_h preserves Hom_A4(dart_rep,rho3); M is a
#        homomorphism; identify M as conjugate to rho3 (not its dual) via an
#        explicit basepoint-evaluation intertwiner S; M is orthogonal.
# ===========================================================================
print()
print("=" * 78)
print("L3 -- M(h) is a genuine A4-representation, M(h) = S^{-1} rho3(h) S, M orthogonal")
print("=" * 78)

A4v_chk, rho3, worst_honest, char_resid = _a4_standard_3irrep()
assert A4v_chk == A4v
print("rho3 honest-homomorphism residual (from the_net.py's own construction):", worst_honest)
print("rho3 character-match residual vs fusion_ring 3-irrep:", char_resid)

worst_rho3_orth = max(float(np.max(np.abs(rho3[k].T @ rho3[k] - np.eye(3)))) for k in range(12))
print("rho3(g) orthogonal for all g, worst residual:", worst_rho3_orth)

A4v_chk2, phi_basis3xND, n_phi, worst_law = _a2d_abstract_hom_basis()
assert n_phi == 3
print("phi_basis equivariance residual (phi_i @ dart_rep(g) == rho3(g) @ phi_i):", worst_law)

# orthonormality of phi_basis w.r.t. the Frobenius/HS inner product on (3,ND)
# matrices (guaranteed by construction: phi_basis = right singular vectors of
# an SVD, which are orthonormal in the standard complex inner product on
# vec(3,ND) = C^{3ND}; reshaping does not change the inner product).
Gram = np.zeros((3, 3), dtype=complex)
for i in range(3):
    for j in range(3):
        Gram[i, j] = np.sum(phi_basis3xND[i] * np.conj(phi_basis3xND[j]))
print("phi_basis Gram matrix (should be I_3):\n", Gram)
orthonorm_resid = float(np.max(np.abs(Gram - np.eye(3))))
print("phi_basis orthonormality residual:", orthonorm_resid)

Phi_mat = np.stack([phi_basis3xND[j].reshape(-1, order="F") for j in range(3)], axis=1)


def induced_M(hk):
    Rh = Rhs[hk]
    M = np.zeros((3, 3), dtype=complex)
    resid = 0.0
    for i in range(3):
        Xi = phi_basis3xND[i] @ Rh
        vec_Xi = Xi.reshape(-1, order="F")
        coeffs, *_ = np.linalg.lstsq(Phi_mat, vec_Xi, rcond=None)
        M[:, i] = coeffs
        resid = max(resid, float(np.max(np.abs(Phi_mat @ coeffs - vec_Xi))))
    return M, resid


M_list = {}
worst_M_resid = 0.0
for hk in range(12):
    M, resid = induced_M(hk)
    M_list[hk] = M
    worst_M_resid = max(worst_M_resid, resid)
print("worst residual of M(h) closing phi-basis exactly (all 12 h):", worst_M_resid)

# (i) M is a genuine homomorphism: M(comp(a,b)) == M(a) @ M(b) for all a,b.
# This is the analytic content proven by hand in the writeup (double
# contravariance: precomposition + R_h's own anti-homomorphism law compose to
# a plain homomorphism). Verify over all 144 pairs.
worst_M_hom = 0.0
for ak in range(12):
    for bk in range(12):
        ab_idx = idx_of(comp(A4v[ak], A4v[bk]))
        worst_M_hom = max(worst_M_hom, float(np.max(np.abs(M_list[ak] @ M_list[bk] - M_list[ab_idx]))))
print("M(comp(a,b)) == M(a) @ M(b), worst over 144 pairs:", worst_M_hom)

# (ii) M(h) is orthogonal for every h (Frobenius-inner-product preservation by
# the orthogonal matrix R_h, expressed in the orthonormal phi_basis).
worst_M_orth = max(float(np.max(np.abs(M_list[hk].conj().T @ M_list[hk] - np.eye(3)))) for hk in range(12))
print("M(h)^dagger M(h) == I for all h (unitary/orthogonal), worst:", worst_M_orth)
worst_M_real = max(float(np.max(np.abs(M_list[hk].imag))) for hk in range(12))
print("M(h) real (imag part), worst:", worst_M_real)

# (iii) identify M EXACTLY (not merely "isomorphic to rho3") via the explicit
# basepoint-evaluation intertwiner S: S_{:,i} := phi_i(e_{d0}) = phi_i[:, d0].
# Derivation (see writeup): Phi_c(x_g) = rho3(g) Phi_c(e_{d0}) for Phi_c = sum
# c_i phi_i (general Hom-space fact), applied twice gives S M(h) = rho3(h) S,
# i.e. M(h) = S^{-1} rho3(h) S EXACTLY.
S = np.stack([phi_basis3xND[i][:, d0] for i in range(3)], axis=1)  # (3,3), col i = phi_i(e_{d0})
print("S (columns = phi_i evaluated at basepoint dart) =\n", S)
S_cond = np.linalg.cond(S)
print("cond(S):", S_cond, " (finite => invertible => the identification is valid)")
Sinv = np.linalg.inv(S)

worst_conj = 0.0
for hk in range(12):
    lhs = M_list[hk]
    rhs = Sinv @ rho3[hk] @ S
    worst_conj = max(worst_conj, float(np.max(np.abs(lhs - rhs))))
print("M(h) == S^{-1} rho3(h) S EXACTLY, worst over 12 h:", worst_conj)

# bonus exact fact: S^dagger S = I/12 (Peter-Weyl style averaging identity)
SHS = S.conj().T @ S
print("S^H S =\n", SHS, " (should be I/12 exactly)")

L3_pass = (worst_honest < 1e-9 and worst_rho3_orth < 1e-9 and worst_law < 1e-8
           and orthonorm_resid < 1e-8 and worst_M_resid < 1e-9 and worst_M_hom < 1e-6
           and worst_M_orth < 1e-6 and worst_M_real < 1e-6 and worst_conj < 1e-6)
print("L3 VERDICT:", "PROVEN" if L3_pass else "FAIL")

# ===========================================================================
# L4 -- the covariance identity Phi_{M(h)u} = Phi_u o R_h for every u, h
# ===========================================================================
print()
print("=" * 78)
print("L4 -- Phi_{M(h)u} = Phi_u @ R_h for every u, h (direct consequence of L3's defn)")
print("=" * 78)


def Phi_of(u_vec):
    u = np.asarray(u_vec, dtype=complex).reshape(-1)
    return sum(u[i] * phi_basis3xND[i] for i in range(3))


rng = np.random.default_rng(20260714)
worst_L4 = 0.0
for _ in range(6):
    u = rng.normal(size=3) + 1j * rng.normal(size=3)
    for hk in range(12):
        Mu = M_list[hk] @ u
        lhs = Phi_of(Mu)
        rhs = Phi_of(u) @ Rhs[hk]
        worst_L4 = max(worst_L4, float(np.max(np.abs(lhs - rhs))))
print("Phi_{M(h)u} == Phi_u @ R_h, worst over 6 random u x 12 h:", worst_L4)
L4_pass = worst_L4 < 1e-8
print("L4 VERDICT:", "PROVEN" if L4_pass else "FAIL")

# non-commutation with the walk structure (load-bearing for the T3 finding
# this lemma inherits, reconfirmed here since M(h) determines R_h's relevance)
B0 = hashimoto_gamma()
Rrev = reversal()
worst_B0 = max(float(np.max(np.abs(B0 @ Rhs[hk] - Rhs[hk] @ B0))) for hk in range(12))
worst_Rrev = max(float(np.max(np.abs(Rrev @ Rhs[hk] - Rhs[hk] @ Rrev))) for hk in range(12))
print("R_h commutes with hashimoto B0? worst (should be large, NOT a symmetry):", worst_B0)
print("R_h commutes with reversal R?  worst (should be large, NOT a symmetry):", worst_Rrev)

# ===========================================================================
# L5 -- what is and is not provable about the triad being an orbit of M(5)
# ===========================================================================
print()
print("=" * 78)
print("L5 -- the triad-orbit claim: what is PROVEN vs what remains numeric/open")
print("=" * 78)

hk5, hk9 = 5, 9
print(f"comp(A4v[{hk5}],A4v[{hk9}]) == identity:", comp(A4v[hk5], A4v[hk9]) == e_id)
M5, M9 = M_list[hk5].real, M_list[hk9].real
print("M5 @ M5 @ M5 - I, max abs (order exactly 3):", np.max(np.abs(M5 @ M5 @ M5 - np.eye(3))))

# PROVEN (trivial from orthogonality+realness of M): M(h) maps the real unit
# sphere in R^3 to itself. This is a genuine, cheap, exact invariance.
v = rng.normal(size=3)
v = v / np.linalg.norm(v)
Mv = M5 @ v
print("M(5) maps a real unit vector to a real unit vector: im part max =",
      np.max(np.abs(Mv.imag)) if np.iscomplexobj(Mv) else 0.0,
      " norm =", np.linalg.norm(Mv))

# NOT PROVEN / MACHINE-VERIFIED ONLY: c2, c3 (the level-set defining the
# triad) are NOT M-invariant as functions of u -- reconfirm minimally.
def c1_c2_c3(u_vec, r=1.0):
    d_vec, _ = w2_family_direction(u_vec, r=r)
    gt = w2_gamma_table(d_vec, N_max=4, max_length=3)
    c2 = float(np.sum(np.abs(gt["by_length"][2]["vectors"]) ** 2))
    c3 = float(np.sum(np.abs(gt["by_length"][3]["vectors"]) ** 2))
    return c2, c3


c2_v, c3_v = c1_c2_c3(v)
c2_Mv, c3_Mv = c1_c2_c3(Mv)
print(f"c2(v)={c2_v:.6f}  c2(M5.v)={c2_Mv:.6f}  (differ => c2 is NOT M-invariant, confirms Gap-B2's T3)")
print(f"c3(v)={c3_v:.6f}  c3(M5.v)={c3_Mv:.6f}  (differ => c3 is NOT M-invariant)")

# The ONE positive structural fact available without redoing the triad polish:
# the symmetric combination G(u) := (c2(u)-1/6)(c2(Mu)-1/6)(c2(M^2 u)-1/6) IS
# M-invariant as a FUNCTION of u (since M^3=I permutes the 3 factors cyclically),
# even though c2 itself is not. This does NOT establish that the specific
# triad {u_A,u_B,u_C} found by independent Newton polishes is exactly an
# M(5)-orbit -- it only shows the *symmetrized* zero-locus condition would be
# M-invariant if imposed as the actual defining system (it is not the system
# that was actually solved).
def G_of(u):
    c2_0, _ = c1_c2_c3(u)
    c2_1, _ = c1_c2_c3(M5 @ u)
    c2_2, _ = c1_c2_c3(M5 @ M5 @ u)
    return (c2_0 - 1.0 / 6) * (c2_1 - 1.0 / 6) * (c2_2 - 1.0 / 6)


Gv = G_of(v)
GMv = G_of(Mv)
print(f"G(v)={Gv:.8f}  G(M5.v)={GMv:.8f}  |diff|={abs(Gv-GMv):.3e}  (symmetrized combo IS M-invariant)")

print()
print("L5 SUMMARY (see writeup for full honesty table):")
print(" PROVEN: M(5) has exact order 3; M(h) preserves realness + unit norm (trivial from orthogonality).")
print(" PROVEN: the symmetrized combination G(u) built by cycling under M(5) is M-invariant as a function.")
print(" NOT PROVEN (remains MACHINE-VERIFIED ONLY, per Gap-B2 T2/T4, cited not rederived here):")
print("   that the SPECIFIC triad {u_A,u_B,u_C} (independently Newton-polished to c2=1/6,c3=1/72 from")
print("   3 different starting points) is exactly the orbit {v0, M(5)v0, M(9)v0} for some v0 -- this")
print("   is an empirical coincidence confirmed to ~1e-8, with no analytic derivation available given")
print("   that c2/c3 themselves are not M-invariant functions.")

print()
print("DONE.")
```

---

## Conclusion

**L1, L2, L3, L4: PROVEN**, upgrading the source station's MACHINE-VERIFIED induced-action
construction to a fully analytic result, including the exact identification `M(h) = S^{-1}ρ₃(h)S`
(settling, with an explicit computation rather than an appeal to self-duality, that the induced
representation is `ρ₃` itself and not its dual) and the exact orthogonality of `M(h)` (from an
inner-product-preservation argument, not from numerology). **L5 is honestly PARTIAL**: the general
structural facts available from `M`'s existence (sphere-preservation, exact order, the correct
*symmetrized* invariance) are proven, but the specific claim used informally in selector-freeze
deliberation — that the empirically-found triad `{u_A,u_B,u_C}` is exactly an `M`-orbit — is not a
corollary of L1–L4 and remains exactly where the source station left it: an exact-to-floor
(`7.686301e-08`) empirical fact with no known analytic derivation, because the defining system
(`c₂=1/6,c₃=1/72`) is provably not `M`-invariant. Nothing here is booked as forced; this is a
group-theory upgrade of the mechanism, not a new selector claim.
