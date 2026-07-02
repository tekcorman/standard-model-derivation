# Substrate Lichnerowicz formula — rigorous derivation

**Date:** 2026-04-26 (PM — follow-on to operator-sweep completion + substrate-modular M1).
**Status:** Theorem-grade closure of G2 from an internal note. Closes honest-scope item 2 of `forward_construction_substrate_atiyah_singer.md` §6.
**Source op:** Layer 6 audit's "discrete scalar curvature" candidate; `forward_construction_substrate_atiyah_singer.md` §1.4 + §4 sketch.
**Predecessors:** `forward_construction_substrate_atiyah_singer.md` (Tier 1 setup of substrate Dirac D_sub); `../theorems/theorem_bloch_lift_mu.md` (Bloch decomposition of D_sub).

---

## Question

The Atiyah-Singer Tier 1 doc sketched a substrate Lichnerowicz formula:

$$D_{\text{sub}}^2 = |E| \cdot I + (\text{substrate spin-curvature})$$

with the spin-curvature term given by a sum involving [γ^e, γ^{e'}] ⊗ [L_e, L_{e'}]. The honest-scope flag (§6 item 2): "rigorous derivation requires explicit computation of [L_e, L_{e'}] commutators on F_inv(E) and identification of the resulting bivector-valued curvature with discrete scalar curvature in a precise sense."

This document supplies that derivation at theorem grade.

---

## Result (preview)

**Theorem (substrate Lichnerowicz).** With n := |E|, the substrate Dirac operator D_sub satisfies:

$$D_{\text{sub}}^2 = n \cdot I_{\text{spinor}} \otimes I_{\text{position}} + R_{\text{sub}}$$

where the **substrate curvature operator**

$$R_{\text{sub}} = \tfrac{1}{2} \sum_{e \neq e'} \gamma^e \gamma^{e'} \otimes [L_e, L_{e'}] = \sum_{e < e'} \gamma^e \gamma^{e'} \otimes [L_e, L_{e'}]$$

is bounded, self-adjoint, and characterized by:

1. **Mean zero.** $\tau(R_{\text{sub}}) = 0$ under the canonical normalized trace $\tau = \tfrac{1}{\dim S}\operatorname{Tr}_{\text{spinor}} \otimes \tau_{\text{vN}}$.
2. **Hilbert-Schmidt norm.** $\|R_{\text{sub}}\|_\tau^2 := \tau(R_{\text{sub}}^* R_{\text{sub}}) = n(n-1)$. For srs (n = 6): $\|R_{\text{sub}}\|^2 = 30$.
3. **Abelianization vanishing.** $R_{\text{sub}} = 0$ iff F_inv(E) is replaced by its abelianization (Z/2)^n.

The classical Riemannian Lichnerowicz formula $D^2 = \nabla^*\nabla + R/4$ has substrate analogs:

| Riemannian Lichnerowicz | Substrate Lichnerowicz |
|---|---|
| $\nabla^*\nabla$ (connection Laplacian) | $n \cdot I$ (constant from $L_e^2 = I$) |
| $R/4$ (scalar curvature) | $R_{\text{sub}}$ (operator-valued; not a scalar) |
| $R = R(\text{point})$ varies on manifold | $R_{\text{sub}}$ has mean 0 + variance $n(n-1)$ |

The substrate has *operator-valued* curvature, not scalar curvature — the substrate is a *non-commutative geometry* (Connes 1994) rather than a Riemannian manifold. Scalar curvature recovered as a moment of $R_{\text{sub}}$.

---

## 1. Setup

### 1.1 The group F_inv(E)

Per Layer 2.13: F_inv(E) = ⟨e_1, …, e_n | e_i^2 = 1⟩ is the free product of $n = |E|$ copies of $\mathbb{Z}/2$. Elements are reduced words in {e_1, …, e_n} where no consecutive letters coincide.

For srs (Wyckoff 8a, 4 atoms, 12 directed bonds, paired into 6 undirected): n = 6.

### 1.2 Hilbert space and operators

$\mathcal{H}_{\text{position}} = \ell^2(F_{\text{inv}}(E))$ with orthonormal basis $\{|g\rangle : g \in F_{\text{inv}}(E)\}$.

$L_e: \mathcal{H}_{\text{position}} \to \mathcal{H}_{\text{position}}$ is the left-regular representation: $L_e|g\rangle = |eg\rangle$ (with reduction $e \cdot e = 1$ if $g$ starts with $e$). Each $L_e$ is unitary. Since $L_e^2 = I$ (involutivity, A1), each $L_e$ is also self-adjoint.

The von Neumann group trace: $\tau_{\text{vN}}(L_g) = \delta_{g, 1_{F_{\text{inv}}(E)}}$ for $g \in F_{\text{inv}}(E)$, extended by linearity and continuity. This is the canonical tracial state on the group von Neumann algebra $L(F_{\text{inv}}(E))$ (which is a type II_1 factor for non-abelian free products with $\geq 2$ generators).

### 1.3 Spinor structure and substrate Dirac

$\mathcal{H}_{\text{spinor}} = S$, the irreducible Cl(6,0;ℂ)-spinor representation (dim S = 8). Per `predictions/theorem_B3_spinor_fermion.py`, S decomposes as one Pati-Salam family.

Substrate Dirac operator on $\mathcal{H} = S \otimes \mathcal{H}_{\text{position}}$:

$$D_{\text{sub}} = \sum_{e \in E} \gamma^e \otimes L_e.$$

Properties (per Atiyah-Singer doc §1.2):
- $D_{\text{sub}}$ is self-adjoint.
- $D_{\text{sub}}$ anticommutes with $\gamma_5 = \gamma^1 \gamma^2 \cdots \gamma^6$ (chirality grading).

### 1.4 Canonical trace

On $\mathcal{H} = S \otimes \mathcal{H}_{\text{position}}$, the canonical normalized trace is

$$\tau(T) := \frac{1}{\dim S} \operatorname{Tr}_S(T_{\text{spinor block}}) \cdot \tau_{\text{vN}}(T_{\text{position block}})$$

extended to general operators by linearity. For $T = A \otimes B$: $\tau(A \otimes B) = \frac{1}{\dim S}\operatorname{Tr}(A) \cdot \tau_{\text{vN}}(B)$.

Norms used below:
- HS-norm under $\tau$: $\|T\|_\tau := \tau(T^* T)^{1/2}$. (Finite for HS operators in the type II_1 factor sense.)

---

## 2. The Lichnerowicz decomposition

**Theorem 2.1 (Lichnerowicz formula).** $D_{\text{sub}}^2 = n \cdot I + R_{\text{sub}}$, where

$$R_{\text{sub}} = \tfrac{1}{2} \sum_{e \neq e'} \gamma^e \gamma^{e'} \otimes [L_e, L_{e'}] = \sum_{\{e, e'\} \text{ unordered}, e \neq e'} \gamma^e \gamma^{e'} \otimes [L_e, L_{e'}].$$

(The two sum forms agree: $\gamma^{e'} \gamma^e \otimes [L_{e'}, L_e] = \gamma^e \gamma^{e'} \otimes [L_e, L_{e'}]$ for each unordered pair.)

*Proof.* Direct computation:

$$D_{\text{sub}}^2 = \sum_{e, e' \in E} \gamma^e \gamma^{e'} \otimes L_e L_{e'}.$$

Split into diagonal $e = e'$ and off-diagonal $e \neq e'$. For $e = e'$: $(\gamma^e)^2 = +1$ (Cl(6,0)) and $L_e^2 = I$ (A1 involutivity). So $\sum_e (\gamma^e)^2 \otimes L_e^2 = n \cdot I_S \otimes I_{\text{position}} = n \cdot I_{\mathcal{H}}$.

For $e \neq e'$, symmetrize over the pair $(e, e') \leftrightarrow (e', e)$:

$$\sum_{e \neq e'} \gamma^e \gamma^{e'} \otimes L_e L_{e'} = \tfrac{1}{2}\sum_{e \neq e'} \big(\gamma^e \gamma^{e'} \otimes L_e L_{e'} + \gamma^{e'} \gamma^e \otimes L_{e'} L_e\big).$$

Use Cl(6,0) anticommutation $\{γ^e, γ^{e'}\} = 0$ for $e \neq e'$ to get $\gamma^{e'} \gamma^e = -\gamma^e \gamma^{e'}$. Substitute:

$$= \tfrac{1}{2}\sum_{e \neq e'} \gamma^e \gamma^{e'} \otimes (L_e L_{e'} - L_{e'} L_e) = \tfrac{1}{2}\sum_{e \neq e'} \gamma^e \gamma^{e'} \otimes [L_e, L_{e'}].$$

QED. $\square$

---

## 3. Properties of R_sub

**Lemma 3.1 (self-adjointness).** $R_{\text{sub}}^* = R_{\text{sub}}$.

*Proof.* Each $\gamma^e$ is Hermitian (A4 Hermitian-Cl(6,0) generators); each $L_e$ is Hermitian (per §1.2 + $L_e^2 = I$). So:

$(\gamma^e \gamma^{e'})^* = (\gamma^{e'})^* (\gamma^e)^* = \gamma^{e'} \gamma^e = -\gamma^e \gamma^{e'}$ (anti-Hermitian for $e \neq e'$).

$([L_e, L_{e'}])^* = (L_e L_{e'})^* - (L_{e'} L_e)^* = L_{e'} L_e - L_e L_{e'} = -[L_e, L_{e'}]$ (anti-Hermitian).

Tensor product of two anti-Hermitian operators is Hermitian: $(γ^e γ^{e'} \otimes [L_e, L_{e'}])^* = (-γ^e γ^{e'}) \otimes (-[L_e, L_{e'}]) = γ^e γ^{e'} \otimes [L_e, L_{e'}]$.

Hence each summand of $R_{\text{sub}}$ is Hermitian, and so is $R_{\text{sub}}$. $\square$

**Lemma 3.2 (mean zero).** $\tau(R_{\text{sub}}) = 0$.

*Proof.* $\tau(\gamma^e \gamma^{e'} \otimes [L_e, L_{e'}]) = \frac{1}{\dim S} \operatorname{Tr}(\gamma^e \gamma^{e'}) \cdot \tau_{\text{vN}}([L_e, L_{e'}])$.

Either factor vanishes:
- The Cl(6,0) bivector $\gamma^e \gamma^{e'}$ for $e \neq e'$ is traceless in the irreducible spinor representation (general property: non-trivial Clifford monomials of degree $\geq 1$ are traceless). $\operatorname{Tr}(\gamma^e \gamma^{e'}) = 0$.
- Equivalently: $\tau_{\text{vN}}([L_e, L_{e'}]) = \tau_{\text{vN}}(L_{ee'}) - \tau_{\text{vN}}(L_{e'e}) = 0 - 0 = 0$ (both $ee'$ and $e'e$ are non-identity length-2 reduced words).

Either way, every summand has $\tau$-trace zero. $\square$

**Lemma 3.3 (vanishing iff abelian).** $R_{\text{sub}} = 0$ iff F_inv(E) is replaced by its abelianization $(\mathbb{Z}/2)^n$.

*Proof.* $(\Leftarrow)$ In $(\mathbb{Z}/2)^n$, $L_e L_{e'} = L_{e + e'} = L_{e' + e} = L_{e'} L_e$, so all commutators vanish.

$(\Rightarrow)$ The bivectors $\{\gamma^e \gamma^{e'} : e < e'\}$ are linearly independent in Cl(6,0) (they span $\Lambda^2 \mathbb{R}^6 \subset$ Cl(6,0), the bivector subspace, of dimension $\binom{6}{2} = 15$). Hence $R_{\text{sub}} = 0 \Leftrightarrow [L_e, L_{e'}] = 0$ for all pairs $e \neq e'$. In F_inv(E) (free product), $ee' \neq e'e$ as reduced words for $e \neq e'$, so $[L_e, L_{e'}] = L_{ee'} - L_{e'e} \neq 0$. Hence $R_{\text{sub}} \neq 0$ on F_inv(E). $\square$

**Theorem 3.4 (Hilbert-Schmidt norm).** $\|R_{\text{sub}}\|_\tau^2 = \tau(R_{\text{sub}}^2) = n(n-1)$.

For srs ($n = 6$): $\|R_{\text{sub}}\|_\tau^2 = 30$.

*Proof.*

$$R_{\text{sub}}^2 = \sum_{\substack{e<e' \\ f<f'}} (\gamma^e \gamma^{e'})(\gamma^f \gamma^{f'}) \otimes [L_e, L_{e'}][L_f, L_{f'}].$$

Apply $\tau$. We compute the spinor and position factors separately.

**Spinor factor.** $\frac{1}{\dim S}\operatorname{Tr}(\gamma^e \gamma^{e'} \gamma^f \gamma^{f'})$ is non-zero iff the monomial $\gamma^e \gamma^{e'} \gamma^f \gamma^{f'}$ reduces to ±1, which requires $\{e, e'\} = \{f, f'\}$ as unordered pairs. Given $e < e'$ and $f < f'$, this forces $(f, f') = (e, e')$. Then:

$$\gamma^e \gamma^{e'} \gamma^e \gamma^{e'} = \gamma^e (-\gamma^e \gamma^{e'}) \gamma^{e'} = -(\gamma^e)^2 (\gamma^{e'})^2 = -1.$$

So $\frac{1}{\dim S}\operatorname{Tr}(\gamma^e \gamma^{e'} \gamma^e \gamma^{e'}) = -1$.

**Position factor.** Compute $[L_e, L_{e'}]^2$:

$$[L_e, L_{e'}]^2 = L_{ee'}^2 - L_{ee'}L_{e'e} - L_{e'e}L_{ee'} + L_{e'e}^2.$$

Each term reduces in F_inv(E):
- $L_{ee'}^2 = L_{ee'ee'}$. The word $ee'ee'$ is reduced (length 4, alternating). Non-identity.
- $L_{e'e}^2 = L_{e'ee'e}$. Length 4, alternating. Non-identity.
- $L_{ee'}L_{e'e} = L_{ee'\cdot e'e} = L_{e (e' e') e} = L_{e \cdot e} = L_{1} = I$.
- $L_{e'e}L_{ee'} = L_{e'e\cdot ee'} = L_{e' (e e) e'} = L_{e' \cdot e'} = I$.

So:

$$[L_e, L_{e'}]^2 = L_{ee'ee'} + L_{e'ee'e} - 2I.$$

Apply $\tau_{\text{vN}}$: $\tau(L_{ee'ee'}) = 0$, $\tau(L_{e'ee'e}) = 0$, $\tau(2I) = 2$. So $\tau_{\text{vN}}([L_e, L_{e'}]^2) = -2$.

**Combine.** Per unordered pair $(e, e')$ with $e < e'$, the contribution to $\tau(R_{\text{sub}}^2)$ is $(-1) \cdot (-2) = 2$. Number of unordered pairs: $\binom{n}{2} = n(n-1)/2$. Total:

$$\tau(R_{\text{sub}}^2) = \binom{n}{2} \cdot 2 = n(n-1).$$

For $n = 6$: 30. $\square$

---

## 4. Comparison to Riemannian Lichnerowicz

Classical Lichnerowicz (1963; Lawson-Michelsohn 1989 §II.8): on a Riemannian spin manifold $(M, g)$ with Levi-Civita connection $\nabla$ on the spinor bundle, the Dirac operator $D = \gamma^a \nabla_a$ satisfies

$$D^2 = \nabla^* \nabla + \tfrac{1}{4} R(x)$$

where $\nabla^* \nabla$ is the connection Laplacian (Bochner Laplacian) and $R(x)$ is the scalar curvature, a smooth real-valued function on $M$.

**Substrate analog.** Under the dictionary

$$\text{(connection Laplacian)} \ \nabla^*\nabla \;\longleftrightarrow\; n \cdot I \quad \text{(constant from } L_e^2 = I)$$

$$\text{(scalar curvature)} \ \tfrac{1}{4} R(x) \;\longleftrightarrow\; R_{\text{sub}} \ \text{(operator-valued)}$$

the substrate's Lichnerowicz formula has the form $D_{\text{sub}}^2 = (\text{kinetic}) + (\text{curvature})$ but the curvature is **not** a scalar function — it is a *non-commutative-geometric* operator on $S \otimes L^2(F_{\text{inv}}(E))$.

Two structural differences from the Riemannian case:

1. **Constant kinetic.** $\nabla^*\nabla$ varies on $M$; on F_inv(E), $\sum_e L_e^2 = n \cdot I$ is uniform. The substrate kinetic is "flat" in this sense — the non-trivial structure resides entirely in $R_{\text{sub}}$.

2. **Operator curvature.** $R(x)$ is a scalar function in Riemannian Lichnerowicz; $R_{\text{sub}}$ is an operator. The substrate is a *non-commutative spectral triple* (Connes 1994) rather than a manifold; the "scalar curvature" is the trace-moment.

The natural moments of $R_{\text{sub}}$ play the role of the Riemannian scalar curvature $R$:
- **Mean** $\tau(R_{\text{sub}}) = 0$ — the substrate is "Ricci-flat in the mean".
- **Variance** $\tau(R_{\text{sub}}^2) = n(n-1)$ — non-zero curvature variance; substrate is "rms-curved".
- **Higher moments** computable from group-theory structure of F_inv(E).

For srs: rms scalar curvature (in this normalization) $= \sqrt{30} \approx 5.48$.

---

## 5. Substrate Riemann tensor analog

Riemannian Riemann tensor: $R^a_{\ bcd}$ encodes $[\nabla_c, \nabla_d] V^a = R^a_{\ bcd} V^b$ on a vector field. For substrate: action of $[L_e, L_{e'}]$ on right-translation by a generator $L_f$ (or on $|g\rangle$ for any $g$) yields the substrate analog.

**Definition 5.1 (substrate Riemann tensor).** For $e, e', f \in E$ and $g \in F_{\text{inv}}(E)$:

$$R^{ee'f}(g) := \langle g | [L_e, L_{e'}] L_f | 1 \rangle = \delta_{g, ee'f} - \delta_{g, e'ef}.$$

This is a 3-index tensor (indexed by $e, e', f$) valued in functions $F_{\text{inv}}(E) \to \{-1, 0, +1\}$.

**Properties.**

1. **Antisymmetric in first two indices:** $R^{e'ef}(g) = -R^{ee'f}(g)$ (immediate from $[L_{e'}, L_e] = -[L_e, L_{e'}]$).
2. **Support.** $R^{ee'f}(g) \neq 0$ only for $g \in \{ee'f, e'ef\}$. Hence the tensor has finite support per index combination: the substrate's Riemann tensor is *sparse* on F_inv(E).
3. **No general algebraic Bianchi identity.** $R^{ee'f}(g) + R^{e'fe}(g) + R^{fee'}(g)$ is not identically zero for non-abelian F_inv(E). The substrate's Riemann analog is more general than Riemannian (which obeys Bianchi as a consequence of Levi-Civita torsion-freeness).

**Ricci analog.** Contract one upper index with the position direction:

$$\text{Ric}^{ef}(g) := \sum_{e' \in E} R^{ee'f}(g) = \sum_{e' \in E} (\delta_{g, ee'f} - \delta_{g, e'ef}).$$

For each $g \in F_{\text{inv}}(E)$, only finitely many $(e, f, e')$ contribute, so $\text{Ric}^{ef}(g)$ is a finite-valued function.

**Scalar Ricci moment.** The first non-trivial scalar invariant from $\text{Ric}$:

$$\bar{R}_{\text{Ric}}^2 := \sum_{e, f} \big\| \text{Ric}^{ef}(\cdot) \big\|_{\ell^2(F_{\text{inv}}(E))}^2 = \sum_{e, f, g} |\text{Ric}^{ef}(g)|^2.$$

This is a (finite) integer count of length-3 reduced words of certain forms in F_inv(E), computable in closed form. For srs ($n = 6$), the count is bounded by $n^2 \cdot (n-1) \cdot 2 = 360$ (each $(e, f)$ contributes from $n - 1$ choices of $e'$, each producing 2 length-3 words).

This sketch suffices to establish the structural picture; full Riemann-analog computation is of bounded effort (counts on F_inv(E)) and tractable in 1–2 follow-up sessions if needed.

---

## 6. Implications for QFT ontology

### 6.1 Connection to GR ontology gap

`../framework/framework_qft_ontology.md` §7 / §8 flagged **Riemann curvature R^a_{bcd}** and **scalar curvature R** as open ontology gaps pending §C smooth-manifold closure. This document grounds *discrete* analogs of both:

- **Substrate scalar curvature** — operator $R_{\text{sub}}$ with explicit Lichnerowicz role and computable trace-moments.
- **Substrate Riemann tensor** — 3-index tensor $R^{ee'f}(g)$ with explicit closed-form expression.

These do **not** require §C closure. The substrate has intrinsic (discrete, non-commutative) curvature even before the smooth-manifold limit is taken.

### 6.2 Status of GR objects under substrate grounding

| QFT-postulated object | Substrate grounding | §C-required? |
|---|---|---|
| Riemann curvature $R^a_{\ bcd}$ | Substrate $R^{ee'f}(g)$ (this doc §5) | No — discrete grounded |
| Ricci tensor $R_{\mu\nu}$ | Substrate $\text{Ric}^{ef}(g)$ (this doc §5) | No — discrete grounded |
| Scalar curvature $R$ | Substrate $R_{\text{sub}}$ operator + moments | No — discrete grounded |
| Lichnerowicz formula | $D_{\text{sub}}^2 = n I + R_{\text{sub}}$ (this doc §2) | No — discrete grounded |
| Levi-Civita connection $\Gamma^a_{bc}$ | (substrate's $L_e$ are the analog "covariant derivatives") | Partial — uses substrate generators |
| Smooth-manifold continuum | (open) | **Yes** — §C still required |
| Einstein equations | (open, §C-pending) | **Yes** — Gorard-style emergent-Einstein |

The discrete-curvature stack (Riemann, Ricci, scalar, Lichnerowicz) is now grounded. The continuum (smooth manifold) and dynamics (Einstein equations) remain Tier 3 research-level open per workstream.

### 6.3 Bridges to other workstreams

- **Quantum-Information workstream Q5 / Atiyah-Singer.** Same operator $R_{\text{sub}}$; this doc rigorizes Q5 via the Lichnerowicz channel.
- **Substrate modular workstream (M1 partial).** $D_{\text{sub}}^2$ via Lichnerowicz pairs with Selberg-zeta / Ihara-Bass spectral side; complementary to Hecke a_2 = √3 finding.
- **GR-G3 emergent-Einstein.** This doc provides the substrate-side curvature inputs needed for any Gorard-style derivation.

---

## 7. Honest scope

1. **Theorem-grade closure.** The Lichnerowicz formula $D_{\text{sub}}^2 = n I + R_{\text{sub}}$, self-adjointness, mean-zero, and HS-norm are rigorous. Proofs above.

2. **Riemann-tensor analog is structural.** Section 5 gives the explicit closed-form $R^{ee'f}(g)$ but does NOT compute all moments / contractions. Full Riemann-Ricci-scalar accounting is a 1–2 session combinatorial follow-up; not blocking.

3. **No new SM-prediction emerges.** Like all Tier 1 / Tier 2 ontology grounding, this is structural. No numerical SM constant is derived; the deliverable is rigorous discrete-curvature grounding.

4. **Substrate is a non-commutative geometry, not a manifold.** The substrate's curvature is operator-valued, not scalar-function-valued. This is the framework's substrate matching Connes (1994) non-commutative-geometric structure, NOT the Riemannian-manifold structure presupposed by classical GR.

5. **Smooth-manifold closure (§C) still open.** The discrete Lichnerowicz / Riemann tensor *do not* substitute for §C smooth-manifold closure. Continuum-limit derivation of smooth scalar curvature $R(x)$ from substrate $R_{\text{sub}}$ remains a Tier 3 open problem.

---

## 8. Status

**Substrate Lichnerowicz formula: theorem-grade.** Formal closure of `forward_construction_substrate_atiyah_singer.md` §6 honest-scope item 2. Closes G2 of an internal note.

**Substrate Riemann tensor analog: structural definition + properties.** Full combinatorial evaluation deferrable as 1–2 session follow-up.

**Category:** category-2 yield (4 additional QFT objects discrete-grounded: Lichnerowicz formula precise, scalar-curvature moments, Riemann tensor analog, Ricci analog).

**Effect on framework:**
- Discrete-curvature stack rigorously grounded; §C dependence removed for the discrete portion.
- Substrate's non-commutative-geometric character explicit: curvature is operator, not scalar function.
- Bridges to A.4 Atiyah-Singer + GR workstream G3 (emergent Einstein, Tier 3).

**Effect on QFT ontology meta-doc:** `../framework/framework_qft_ontology.md` should add Lichnerowicz-formula and Riemann-tensor entries with grounding pointer to this document.

---

## 9. Cross-references

- `forward_construction_substrate_atiyah_singer.md` §1.4, §4, §6 — predecessor sketch; §6 honest-scope item 2 closed by this doc.
- `../theorems/theorem_bloch_lift_mu.md` — Bloch decomposition of $D_{\text{sub}}$.
- `predictions/theorem_B3_spinor_fermion.py` — Cl(6,0) Dirac spinor structure.
- `../framework/framework_qft_ontology.md` §7 (cosmology / GR), §8 (open gaps) — pending update.
- `../operator_sweep/operator_sweep_audit_layer_6.md` — Layer 6 unused-deferred cluster (Riemann curvature, scalar curvature flagged as ungrounded; partially closed by this doc).

**Type 3 (cited published) references:**

- **Lichnerowicz, A.** (1963). Spineurs harmoniques. *C. R. Acad. Sci. Paris* 257, 7–9.
- **Lawson, H. B. & Michelsohn, M.-L.** (1989). *Spin Geometry.* Princeton University Press, §II.8.
- **Connes, A.** (1994). *Noncommutative Geometry.* Academic Press. (Operator-valued curvature; spectral triples; Connes spectral action.)
- **Voiculescu, D., Dykema, K., Nica, A.** (1992). *Free Random Variables.* CRM Monograph 1. (Free-product group von Neumann algebras / type II_1 factor for L(F_inv(E)).)

All citations to peer-reviewed published work / monographs.

---

## 10. Next forward-construction steps

1. **Riemann-analog combinatorial evaluation** (1–2 sessions, bounded effort): full closed-form computation of $\text{Ric}^{ef}$ contractions, scalar-Ricci moments on F_inv(E). Optional follow-up.
2. **Bridge to substrate Atiyah-Singer index** (continuation of A.4): use Lichnerowicz to relate ind(D_sub) to topological / spectral data of substrate, with explicit Bloch-fiber index computation at the P-point.
3. **GR-G3 emergent-Einstein direction** (research-level): with discrete-curvature stack grounded, attempt Gorard-style continuum limit deriving Einstein equations. Multi-month.
4. **Connes-spectral-action evaluation**: Connes' spectral action $\operatorname{Tr}(f(D_{\text{sub}}^2/\Lambda^2))$ for cutoff function $f$ gives an effective action including curvature. Substrate-side computation could connect to gravity + SM unification (Chamseddine-Connes 1996). Tier 3.
