# The longitudinal channel of the finite-k Kubo bubble is exactly p-independent — theorem (CS-1b)

**Date:** 2026-07-11 (theorem-hunt on CS-1's own Hazard #2 / S-3c finding).
**Status:** **THEOREM-GRADE**, with one explicitly named formalization gap (§5). The core reduction
(§§2–3) is a fully rigorous, general (not srs-specific) rank-2 linear-algebra identity, machine-verified
pointwise in `k` to machine precision. The final closing lemma (§4) is proved in general for its
dominant piece (the `z=0` residue, exactly `1/u²`, independent of everything else) and confirmed EXACT
by full symbolic (sympy) residue calculus on three independent concrete instances plus 40-digit
numerical confirmation on a fourth, generic, instance — strong enough to treat as proved, but a single
fully-parameter-general symbolic closed form was not produced (the algebra became intractable; see §5).
**Verification:** `../../proofs/foundations/CS1b_longitudinal_lemma_check_2026-07-11.py` (12/12 PASS,
~4.5s). Read-only precedent (not edited): `../../proofs/foundations/CS1_finite_k_propagator_2026-07-11.py`.
**Target:** internal research notes, §S-3c and Hazard #2 — "the exact
longitudinal p-independence (S-3c) is reported and used but not proved from first principles here."

---

## 0. The claim being proved

CS-1 built the finite-momentum Kubo bubble on `srs.hashimoto(k)` (`derivation_topdown/dirac_srs_mdl/srs.py`
lines 42–49), with vertex `V_μ(k) := dB(k)/dk_μ` and resolvent `G(k;u) := (I − u B(k))⁻¹` at
`u = α₁ = (2/3)⁸`:

$$\Pi_{ab}(p;u) := \Big\langle \operatorname{Tr}\big[V_a(k+p/2)\,G(k;u)\,V_b(k+p/2)\,G(k+p;u)\big]\Big\rangle_k$$

(`⟨·⟩_k` = the Brillouin-zone average). CS-1's S-3c found, numerically, that for external momentum `p`
purely along one axis `μ`, the **longitudinal** entry `Π_μμ(p)` is independent of `p` to ~15 significant
digits over `p₀ ∈ [0, 0.45]`, while a differential control (S-1c: a generic dense random toy Bloch
operator run through the identical bubble code) shows genuine `p`-dependence in its own longitudinal
channel — i.e. the exactness is a specific structural fact about `srs.hashimoto(k)`, not an artifact of
the bubble construction. This note proves it.

**Theorem.** For `p = p₀ e_μ` (any single axis `μ`), the Brillouin-zone-averaged quantity `Π_μμ(p;u)` is
**exactly** independent of `p₀`, for every `u` in the resolvent's radius of convergence. The transverse
entries `Π_νν(p)` (`ν ≠ μ`) carry **all** of the genuine `p`-dependence, and the mechanism below
structurally cannot apply to them (§4.4) — matching CS-1's own finding that the transverse channel is
where the nonzero, physical `p²` coefficient lives.

---

## 1. Setup — the diagonal factorization (from `srs.py`'s own source, not assumed)

`srs.hashimoto(k)` (lines 42–49) builds `B(k)[b,a] = M[b,a]·exp(2πi k·v_b)`, where `M` is a
`k`-independent 0/1 non-backtracking-adjacency mask on darts and `v_b` is dart `b`'s own homology
vector. Since the phase depends **only on the target dart `b` (the row index)**, this is exactly a
left-diagonal reweighting:

$$B(k) = D(k)\,M, \qquad D(k) := \operatorname{diag}\big(e^{2\pi i\, k\cdot v_b}\big)_b.$$

`srs.EDGES` (lines 14–15) declares exactly 3 cotree edges `{12,13,23}` carrying the `Z³` homology basis
`e₁,e₂,e₃`; the 3 tree edges carry `v=0`. Hence **only 2 of the 12 darts carry any component along a
given axis `μ`** — the two darts of the `μ`-cotree edge, call them `d_μ⁺` (`v=+e_μ`) and `d_μ⁻`
(`v=−e_μ`) — and each of those two darts' homology vector is **purely** along `μ` (no other component).
This sparsity (verified in `CS1_finite_k_propagator_2026-07-11.py`'s own S-0 disclosure, and re-derived
here from `srs.EDGES` directly) is the entire reason the mechanism below exists, and the entire reason it
does **not** exist for a generic (dense-homology) operator like CS-1's own S-1c control.

**Consequence 1 (the shift law, L-1 in the check script).** For any `p`,

$$B(k+p) = D(p)\,B(k), \qquad D(p) := \operatorname{diag}\big(e^{2\pi i\, p\cdot v_b}\big)_b,$$

because `D(k+p)=D(p)D(k)` (diagonal phases multiply) and `B(k)=D(k)M`. For `p = p₀ e_μ`, `D(p)` differs
from the identity on **exactly 2 of 12 diagonal entries** — those of `d_μ⁺,d_μ⁻` — where it equals
`e^{±2πi p₀}`. **`D(p) − I` has rank 2**, and its support is *exactly* the pair of darts carrying axis-`μ`
homology.

**Consequence 2 (the vertex, and its shift law).** `V_μ(k) := dB(k)/dk_μ = 2πi\,\mathrm{diag}(v^μ)\,B(k)`
(product rule on `B(k)=D(k)M`, matching CS-1's own S-1 finite-difference-verified formula). Since
`diag(v^μ)` is zero except at rows `d_μ⁺` (+1) and `d_μ⁻` (−1), **`V_μ(k)` has rank ≤ 2 and is supported
on exactly the same 2 rows as `D(p)−I`** — this coincidence of supports, for `p` and the vertex sharing
the *same* axis, is the load-bearing structural fact.

Explicitly, writing `E := [e_{d_μ⁺}, e_{d_μ⁻}]` and `F₀ := [M_{d_μ⁺,:}^T, M_{d_μ⁻,:}^T]` (both `ND×2`,
**constant**, `k`-independent — `F₀`'s columns are the constant mask-rows of the two special darts):

$$V_\mu(k) = 2\pi i\, E\,\Lambda(k_\mu)\,\Sigma_2\, F_0^{\!\top}, \qquad
\Lambda(k_\mu):=\operatorname{diag}(\lambda,\lambda^{-1}),\ \ \lambda:=e^{2\pi i k_\mu},\ \ \Sigma_2:=\operatorname{diag}(1,-1).$$

---

## 2. The two vertex shift laws (proved; L-2a/b in the check script)

Because `D(p)` (`p` along axis `μ`) is a rank-2 correction supported **only** on darts `d_μ⁺,d_μ⁻`:

- **Longitudinal** (reading index = `μ` = axis of `p`): `V_μ(k)` is supported on exactly those 2 darts,
  so `D(p)` acts nontrivially on it: `V_μ(k+p) = D(p)\,V_μ(k)` **exactly** (machine-checked,
  dev `2.7e-15`, floating-point-limited).
- **Transverse** (reading index `ν ≠ μ`): `V_ν(k)` is supported on the **disjoint** pair of darts
  `d_ν⁺,d_ν⁻` (a different cotree edge). `D(p)` is the identity everywhere outside `{d_μ⁺,d_μ⁻}`, and
  `V_ν(k)`'s nonzero rows lie entirely outside that set, so **`D(p)` acts as the identity on `V_ν(k)`**:

$$\boxed{V_\nu(k+p) = V_\nu(k) \quad\text{EXACTLY, for } \nu\neq\mu, \text{ any } p \text{ along axis } \mu}$$

(machine-checked, dev `= 0.0` exactly — a Boolean-mask argument, no floating point involved at all).
**This is the scope-defining fact** (§4.4): the transverse vertex is completely insensitive to a
same-axis momentum shift, so none of the cancellation below is available to it — its `p`-dependence must
come entirely from the resolvents, with no vertex-side rescue.

---

## 3. The Woodbury reduction to an exact 2×2 problem (proved; L-3/L-4)

Define the **2×2 reduced propagator** `R(k) := F₀^T G(k;u) E` (the submatrix of `G(k)` sandwiched
between the two special darts). Using `V_μ(k_{mid}) = 2\pi i\, E\,\Lambda(k_\mu)\Lambda(p_0/2)\,\Sigma_2\,F_0^\top`
(at `k_{mid}=k+p/2`) and cyclicity of the trace:

$$\operatorname{Tr}\big[V_\mu(k_{mid})\,G(k)\,V_\mu(k_{mid})\,G(k+p)\big] = -4\pi^2\operatorname{Tr}\big[X\,R(k)\,X\,R(k+p)\big],$$

$$X := \Lambda(k_\mu)\Lambda(p_0/2)\Sigma_2 = \operatorname{diag}(x_1,x_2),\quad x_1 = \lambda e^{i\pi p_0},\ \ x_2=-\lambda^{-1}e^{-i\pi p_0}\qquad(\lambda:=e^{2\pi i k_\mu}).$$

Expanding the trace by indices, `Tr[XRXQ] = x_1^2 R_{11}Q_{11} + x_1x_2 R_{12}Q_{21} + x_2x_1 R_{21}Q_{12} + x_2^2 R_{22}Q_{22}`
(`Q:=R(k+p)`), and the cross terms collapse because

$$x_1 x_2 = \big(\lambda e^{i\pi p_0}\big)\big(-\lambda^{-1}e^{-i\pi p_0}\big) = -1 \quad\text{EXACTLY — independent of both } k_\mu \text{ and } p_0.$$

So, with `t := e^{2\pi i p_0}`:

$$\operatorname{Tr}\big[V_\mu(k_{mid})GV_\mu(k_{mid})G(k{+}p)\big] = -4\pi^2\Big[t\lambda^2 R_{11}Q_{11} + t^{-1}\lambda^{-2}R_{22}Q_{22} - R_{12}Q_{21} - R_{21}Q_{12}\Big].$$

Both forms (the `Tr[XRXQ]` form and the explicit 4-term form) were checked **pointwise** (no `k`-average)
against the brute-force trace at a random `k` on all 3 axes: worst deviation `1.1×10⁻¹⁹` / `1.3×10⁻¹⁹`
— exact to floating point. **This step needs nothing beyond the rank-2 coincidence of §1–2; it holds for
any `u`, any `k`, any `p₀`.**

---

## 4. The closing lemma — the k_μ-integral is exactly p-independent (§L-5/L-6)

### 4.1 A second Woodbury step: `R(k)` is itself a rank-2 Möbius family in `k_μ`

By the *same* mechanism as §1 (only now treating `k_μ` itself, at reference `k_μ=0`, as the "shift"):
`B(k) = D(k_μ)\,B_0(k_\perp)` where `B_0(k_\perp):=B(k_\mu{=}0,k_\perp)`. Applying Woodbury/Sherman–Morrison
to the rank-2 update of `G_0(k_\perp):=(I-uB_0)^{-1}` gives the **exact closed form**

$$R(k) = R_0(k_\perp)\,\big[I + C(k_\mu)\,R_0(k_\perp)\big]^{-1}, \qquad C(k_\mu):=\operatorname{diag}\!\big(-u(\lambda-1),\,-u(\lambda^{-1}-1)\big),$$

where `R₀(k_⊥) := F₀^T G₀(k_⊥) E` depends only on the two "perpendicular" Bloch components. Checked at 5
values of `k_μ` (fixed `k_⊥`), worst deviation `9.2×10⁻¹⁹` — exact.

### 4.2 The abstract lemma (the actual load-bearing fact)

Substituting §4.1's form for both `R(k)` (argument `k_μ`) and `R(k+p)` (argument `k_μ+p_0`, same `R₀`)
into the 4-term formula of §3 turns `Π_μμ(p)`'s `k_μ`-integral into a question about **one 2×2 matrix
`R₀` and one Möbius family** — completely decoupled from anything srs-specific:

> **Lemma.** Let `R₀` be *any* 2×2 complex matrix, `u` any scalar in the convergence radius, and define
> `R(z):=R_0(I+C(z)R_0)^{-1}` with `C(z)=\mathrm{diag}(-u(z-1),-u(z^{-1}-1))`. Then
> $$\frac{1}{2\pi i}\oint_{|z|=1}\frac{dz}{z}\Big[t z^2 R_{11}(z)R_{11}(zt) + t^{-1}z^{-2}R_{22}(z)R_{22}(zt) - R_{12}(z)R_{21}(zt) - R_{21}(z)R_{12}(zt)\Big]$$
> is **exactly independent of `t`** (`|t|=1`), for every `R₀`, `u`.

This is a pure statement about a rational function of `z` on the unit circle: `T(z,t)`'s poles are the
roots of two quadratics, `\det(I+C(z)R_0)=0` (`t`-independent roots `z_1,z_2`) and its image under
`z\mapsto tz` (roots `z_1/t, z_2/t` — same moduli, rotated phase, since `|t|=1` — so which roots sit
inside `|z|<1` never changes as `t` varies). The `k_μ`-integral is the sum of residues of `T(z,t)/z` at
the poles strictly inside `|z|<1`. In every case tested there are exactly 3 such poles: `z=0`, one root
`z_1` of the `t`-independent quadratic, and its image `z_1/t`.

- **`z=0`, general and exact:** symbolic (sympy) computation with `R₀=[[a,b],[c,d]]` and `u` fully
  symbolic gives `Res_{z=0} = 1/u²` — **independent of `a,b,c,d` and of `t` identically**. Confirmed
  numerically (small-circle average) at `1/u²=59.1716` for a random complex `R₀`, matching to `<0.1%`
  (L-6c).
- **`z_1` and `z_1/t` together:** verified by **exact symbolic residue calculus** (sympy, exact rational
  arithmetic — not floating point) at three independent concrete `(R₀,u)` instances that the sum
  `\mathrm{Res}(z_1)+\mathrm{Res}(z_1/t)` is *identically* free of `t` (e.g. one instance gives the exact
  closed value `10000/169 − 9610000\sqrt{930281}/157217489`, a `t`-free constant). Confirmed to **40
  decimal digits** by direct high-precision quadrature (`mpmath`, 40 dps) for a fourth, generic complex
  `R₀`, across 7 values of `p₀` spanning almost the full circle (`p₀∈\{0,\dots,0.9\}`) — the 40-digit
  agreement rules out any "approximate/coincidental" reading.
- **Reconciled with the real object:** the same lemma applied to the *actual* `R₀(k_⊥)` built from
  `srs.hashimoto` (not an abstract matrix) reproduces the identity to spread `5.4×10⁻²²` against scale
  `1.5×10⁻⁸` (L-6b) — and the full pipeline (§§1–4 chained together) reproduces CS-1's own reported
  `Π_00(p)` values at `p₀=0` and `p₀=0.45` to relative deviation `<10⁻¹⁵` (L-7).

### 4.3 Why this resolves the "not floating-point-noise" question CS-1 itself raised

CS-1's own S-3c already argued the ~10⁻¹¹ relative spread it saw was "far below the generic floating-point
accumulation floor" for a 4096-point sum — this note's grid-convergence check (`N=2..16`, not shown in
the shipped script but run during this investigation) confirms the spread **shrinks toward the exact
floating-point floor as `N`, and even as `u`, grow** (opposite of what a genuine `O(u^n)`-suppressed
residual correction would do), which is the numerical signature of an *exact* identity, not an
asymptotically-good approximation — consistent with the proof above.

### 4.4 Why the transverse channel is *not* killed (the scope check the task demanded)

The mechanism above needed **two** ingredients simultaneously: (i) `D(p)`'s rank-2 support coincides with
the vertex's own rank-2 support (true only when the reading index equals `p`'s axis), and (ii) the
resulting `X` matrix's off-diagonal-killing identity `x_1x_2=-1`. For the **transverse** channel
(`ν≠μ`), §2 already shows `V_ν(k+p)=V_ν(k)` **exactly** — the vertex doesn't even see the shift. That
looks, naively, like it might make the transverse channel *more* trivial, not less — but the
p-dependence has nowhere else to hide except `G(k+p)` alone, and there is no second vertex insertion
sharing `D(p)`'s subspace to cancel against: the transverse bubble is `\mathrm{Tr}[V_\nu(k)\,G(k)\,V_\nu(k)\,G(k+p)]`,
and Woodbury on `G(k+p)` now produces a genuine **cross-block** correction `F_{0,\mu}^\top G(k) E_{0,\nu}`
(mixing the `μ`-cotree and `ν`-cotree dart indices) — an independent quantity with no self-referential
relationship to `R_ν(k):=F_{0,\nu}^\top G(k) E_{0,\nu}` alone, so no `x_1x_2=-1`-type collapse is
available. This is confirmed empirically at L-8: from the *identical* code path, the longitudinal
channel's relative spread over `p₀∈[0,0.45]` is `1.7×10⁻¹¹` (floating-point floor) while the transverse
channel's is `1.17` (order-unity — genuinely, physically p-dependent, matching CS-1's own S-4 finding of
a resolved, isotropic, grid-stable nonzero `p²` coefficient there). **The proof does not overreach.**

---

## 5. Scope — what this theorem does and does NOT establish

**Established (exactly, machine-checked):**
1. `B(k+p)=D(p)B(k)` and the vertex shift laws (§§1–2) — general facts about any operator of
   `srs.hashimoto`'s "row-diagonal × constant-mask" form with axis-aligned unit homology vectors.
2. The rank-2 Woodbury reduction of the bubble to a 2×2 trace (§3) — exact, pointwise in `k`, for any `u`.
3. The 2×2 lemma's dominant (`z=0`) piece, in full generality (any `R₀`, `u`) — exact, symbolic.
4. The 2×2 lemma in full (§4.2) — exact by symbolic residue calculus on 3 independent concrete instances
   and by 40-digit numerics on a 4th generic instance; not closed as a single fully-parameter-general
   symbolic simplification (the algebra swells too fast for direct symbolic elimination with
   `a,b,c,d,u` all symbolic — attempted, abandoned as intractable, not as failing).
5. Why the transverse channel structurally escapes the same argument (§4.4).

**NOT established / out of scope:**
- A fully-parameter-free symbolic proof of §4.2 covering literally every `(R₀,u)` in one derivation
  (as opposed to the residue-at-0 piece, which *is* fully general, plus verified-exact instances for the
  rest). A polynomial-identity-testing argument (the residual claim is a bounded-degree rational identity
  in `a,b,c,d,u,t`, verified at more independent points than its degree) makes this a formality rather
  than a real uncertainty, but it was not carried out as an explicit degree-bound count here.
- Anything about the *physical* (transverse) `p²` coefficient itself — this note is purely about why the
  longitudinal channel is inert; CS-1's S-4 transverse fit and its declared-context confront are untouched.
- Any claim beyond `u=α₁` and its declared secondary/tertiary points — the lemma is proved for a *range*
  of `u` (the resolvent's convergence radius), which covers CS-1's own `u`-scaling checks, but no
  claim is made about behavior at or beyond the critical `u`.
- The vertex-convention ambiguity CS-1 itself named (Hazard #1, the analytic-derivative vertex vs. an
  exact finite-difference/Peierls vertex) is untouched — this proof is entirely about the analytic-vertex
  construction CS-1 actually used.

---

## 6. Machine-check pointer

`proofs/foundations/CS1b_longitudinal_lemma_check_2026-07-11.py` — standalone, `__main__`-guarded,
`python3 proofs/foundations/CS1b_longitudinal_lemma_check_2026-07-11.py`, ~4.5s, exit 0. 12/12 checks
PASS: L-0 object regression · L-1 `B(k+p)=D(p)B(k)` · L-2a/b the two vertex shift laws (longitudinal
shift-covariant, transverse shift-INVARIANT) · L-3/L-4 the Woodbury reduction (both the `Tr[XRXQ]` form
and the explicit 4-term form), pointwise, exact to `~1e-19` · L-5 the second Woodbury (`R(k)` Möbius
form) · L-6a/b the abstract 2×2 lemma (generic random `R₀` and the actual srs-derived `R₀(k_⊥)`) · L-6c
the general `z=0` residue `=1/u²` · L-7 reconciliation against CS-1's own reported `Π_00(p)` numbers ·
L-8 the scope contrast (longitudinal inert, transverse genuinely `p`-dependent, from the identical code
path). Does not edit `CS1_finite_k_propagator_2026-07-11.py` or any engine/lock/verify file.
