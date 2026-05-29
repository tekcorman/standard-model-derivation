# 3b — Canonical Cartan subalgebra of Cl(V, Q) via S_4 invariance (B3-chirality-bridge)

**Date:** 2026-04-20 (Sprint γ, Session 6 spike).
**Grade:** mathematically complete.
**Script:** `predictions/B3_chirality_bridge.py`.
**Scoping + spike findings:** an internal working note §7.
**CAS verifications:** `proofs/foundations/K4_matchings_C3_check.py`, `K4_C3_equivariant_pairings.py`, `K4_S4_A4_invariant_pairings.py`.

## Abstract

Under Theorem B1.b (invariant Clifford construction on the 6-edge K_4 quotient space of srs, `predictions/theorem_B1_ordering.py`), the Clifford algebra Cl(V, Q) is manifestly $S_6$-equivariant: no canonical ordering of the 6 generators is forced, and any physical structure extracted from Cl(V, Q) must be $S_6$-representation-theoretically natural. We show that when a Cartan subalgebra of Cl(V, Q) is chosen consistent with the srs primitive-cell vertex-symmetry group $S_4 \subset S_6$ (from the space-group point group $432 = O \cong S_4$ acting on the 4 Wyckoff 8a positions as the 4 body-diagonals of the conventional cubic cell), the choice is uniquely the matching-partition Cartan: the bivector triple

$$
(T_1, T_2, Y) = \left( \tfrac{1}{2i}\,\Gamma_{M_1},\ \tfrac{1}{2i}\,\Gamma_{M_2},\ \tfrac{1}{2i}\,\Gamma_{M_3} \right),
\qquad \Gamma_{M_a} = \Gamma_e \Gamma_{e'} \text{ for } M_a = \{e, e'\},
$$

where $M_1, M_2, M_3$ are the three perfect matchings of $K_4$. The body-diagonal $C_3$ generator $\sigma = (v_0)(v_1\, v_3\, v_2)$ cyclically permutes the three Cartan generators.

**Narrowed scope (2026-04-20 session 7 correction):** This is an algebraic theorem about the canonical Cartan subalgebra. It does NOT by itself unblock CKM numerically or structurally. Session 7 re-verification (`proofs/foundations/LS3_optionB_reverify.py`) showed that any $\sigma_{\text{combined}} = \sigma_S \otimes \sigma_{\text{obs}}$-invariant tensor-product mass operator on $S \otimes C^3_{\text{obs}}$ yields circulant Yukawa matrices $Y_X$ for every species $X$, all diagonalized by the same $\mathrm{DFT}_3$, hence $V_{\mathrm{CKM}} = I$ identically. Original claims about "CKM structural unblock" below have been retracted; see Open Questions §4 and an internal working note §11 for the revised gap analysis.

## Framework axioms invoked

- **A1** (binary self-inverse toggle): enters via upstream B1.b and the srs walker → K_4 quotient derivation.
- **A2** (MDL, refined selective retention): enters as B1.b's forcing of the invariant Clifford construction (S_6-equivariant). Also enters at the "canonical Cartan" step (§Step 6), where among the $C_3$-equivariant pair-partitions only the $S_4$-invariant one — the matching-partition — is retained as canonical; the two vertex-sharing partitions fail $S_4$-equivariance and are rejected under A2.
- **Upstream theorems:** B1.b (S_6-equivariance), BP (C_3 action σ on the primitive cell, from `../../predictions/B_P_doubly_degenerate_h_derivation.md` Step 2).

## Cited mathematical theorems and crystallographic data

- **Lawson, H. B. & Michelsohn, M.-L. (1989).** *Spin Geometry.* Princeton Univ. Press. Ch. I §1 (tensor-algebra Clifford construction), Ch. I §6 (Cartan subalgebras of spin Lie algebras).
- **Dummit, D. S. & Foote, R. M. (2004).** *Abstract Algebra,* 3rd ed. §2.2 Exercise 4 (Aut(K_n) = S_n).
- **Coxeter, H. S. M. (1973).** *Regular Polytopes,* 3rd ed. Dover. §4.4 (octahedral rotation group $O \cong S_4$ via body-diagonal action on cube).
- **International Tables for Crystallography, Volume A.** Space group I4_1 32 (#214); point group 432; Wyckoff 8a site.
- **Euler, L. (1736).** Eulerian trails (upstream to B1.b's walker obstruction).

## Setup

### The srs K_4 quotient

The srs lattice (space group I4_1 32, Wyckoff 8a, `predictions/d_spatial.py`) has primitive-cell quotient graph $K_4$ with 4 vertices $\{v_0, v_1, v_2, v_3\}$ and 6 edges $e_{ij} = \{v_i, v_j\}$ for $0 \leq i < j \leq 3$. The 6-dim quadratic space $V = \mathbb{R}^{E(K_4)} = \mathbb{R}^6$ carries the signature-$(6,0)$ form $Q$ (`../../predictions/theorem_B2_signature_derivation.md`).

### The C_3 action

Per `../../predictions/B_P_doubly_degenerate_h_derivation.md` Step 2, the body-diagonal $C_3$ axis through $v_0$ induces the vertex permutation

$$
\sigma = (v_0)(v_1\, v_3\, v_2)
$$

on the primitive cell, and via the permutation of the 6 K_4 edges gives a 6-dim orthogonal representation of $C_3$ on $V$.

### The S_4 action (new for 3b)

The space group I4_1 32 has point group 432 = O (order 24). Under the well-known isomorphism $O \cong S_4$ (Coxeter 1973 §4.4), realized by the action of the rotation group on the 4 body-diagonals of the underlying cubic cell, the 4 primitive-cell vertices $\{v_0, v_1, v_2, v_3\}$ (at Wyckoff 8a = body-diagonal centers modulo body-centering) are permuted by the full symmetric group $S_4$. The body-diagonal $C_3$'s are a cyclic subgroup $\langle \sigma \rangle \cong \mathbb{Z}_3 \subset S_4$; together with the three face-center $C_2$'s and the six edge-midpoint $C_2$'s, the 24 rotations of 432 realize all 24 permutations of $\{v_0, v_1, v_2, v_3\}$.

## Derivation

### Step 1 — Cl(V, Q) is S_6-equivariant (B1.b chain-import)

From `predictions/theorem_B1_ordering.py` (Theorem B1.b): the Clifford algebra on the 6-edge space $V$ must be defined via the tensor-algebra quotient

$$
\mathrm{Cl}(V, Q) := T(V) \,\big/\, \langle v \otimes v - Q(v) \cdot 1 : v \in V \rangle
$$

(Lawson-Michelsohn 1989 Ch. I §1, Eq. (1.1)). This construction is manifestly $S_6$-equivariant: no ordering of the 6 generators is selected. No MDL-canonical ordering exists (B1.b Step 3: all 30 $S_4$-orbits of orderings have identical model-cost; B1.b Step 4: the Eulerian trail obstruction blocks walker-induced orderings).

### Step 2 — srs point-group action as S_4 on K_4 vertices

Space group I4_1 32 has rotation point group 432 = O, order 24 (International Tables Vol. A). By Coxeter 1973 §4.4, $O$ is isomorphic to $S_4$ via the action on the 4 body-diagonals of the underlying cubic cell. Under the body-centering quotient to primitive cell, the 4 Wyckoff 8a sites are identified with the 4 body-diagonals, giving an $S_4$ action on the 4 primitive-cell K_4 vertices. This $S_4$ action is faithful on the edges (Dummit-Foote §2.2 Ex 4: $\mathrm{Aut}(K_n) = S_n$).

### Step 3 — Pair-partitions of the 6 K_4 edges

A 3-way pair-partition of the 6 K_4 edges is a partition of $E(K_4)$ into 3 disjoint 2-element subsets. By elementary combinatorics,

$$
\text{Number of 3-way pair-partitions of 6 objects} = \frac{6!}{2^3 \cdot 3!} = 15.
$$

Each such partition $P = \{P_1, P_2, P_3\}$ defines three bivectors $B_a = \Gamma_{e_a^{(1)}} \Gamma_{e_a^{(2)}}$ for $P_a = \{e_a^{(1)}, e_a^{(2)}\}$. These are mutually commuting (all three bivectors involve disjoint pairs of generators in the abstract Cl(V, Q)), so they span a 3-dimensional abelian subspace of $\mathfrak{spin}(V)$. By Lawson-Michelsohn 1989 I §6, this is a Cartan subalgebra (maximal abelian subalgebra of $\mathfrak{spin}(V)$; rank $= 3 = \lfloor 6/2 \rfloor$).

### Step 4 — C_3-equivariant pair-partitions (§7.7 of scoping)

Direct enumeration (CAS-verified, `proofs/foundations/K4_C3_equivariant_pairings.py`) shows that among the 15 pair-partitions, exactly **3 are $C_3$-equivariant** under $\sigma$:

- **Matching partition** $P_M = \{\{e_{03}, e_{12}\}, \{e_{01}, e_{23}\}, \{e_{02}, e_{13}\}\}$. Each pair is a perfect matching (disjoint edges).
- **Vertex-sharing partition** $P_A = \{\{e_{01}, e_{12}\}, \{e_{02}, e_{23}\}, \{e_{03}, e_{13}\}\}$. Each pair shares the rim vertex $v_1, v_2, v_3$ respectively.
- **Vertex-sharing partition** $P_B = \{\{e_{01}, e_{13}\}, \{e_{02}, e_{12}\}, \{e_{03}, e_{23}\}\}$. Same pattern, different assignment.

All three are regular representations of $C_3$ (order-3 action on the 3-element pair-space). No trivial-rep equivariant partition exists.

### Step 5 — S_4-uniqueness of the matching partition (§7.9 of scoping)

Direct enumeration (CAS-verified, `proofs/foundations/K4_S4_A4_invariant_pairings.py`) shows that among the 3 $C_3$-equivariant partitions:

- $P_M$ is **$S_4$-invariant** (and $A_4$-invariant).
- $P_A$ and $P_B$ are **not $S_4$-invariant** (counterexample: transposition $(v_2\, v_3)$ maps $P_A$ out of itself).

Heuristic: the vertex-sharing partitions single out $v_0$ as the "center" vertex — an object only the $C_3$ stabilizer of $v_0$ cares about. Under the full $S_4$, no vertex is distinguished; only the matching partition (defined by the graph-theoretic property of disjointness, independent of labels) is invariant.

By Step 4 ($P_M, P_A, P_B$ are all $C_3$-equivariants) and Step 5 ($P_M$ is the unique $S_4$-invariant), the matching partition is the unique pair-partition that is both $C_3$- and $S_4$-invariant. Since $C_3 \subset S_4$, the stronger condition is just $S_4$-invariance: $P_M$ is the unique $S_4$-invariant pair-partition of the 6 K_4 edges. (Direct recomputation in the prediction script confirms: among all 15 partitions, exactly 1 is $S_4$-invariant.)

### Step 6 — Canonical Cartan choice via A2-MDL selective retention

Under A2 refined (selective retention, `docs/framework/framework_axioms.md` §3), the framework retains invariant structures uniformly under the ambient symmetry. The ambient symmetry of the Cl(V, Q) construction combines:

- $S_6$-equivariance (B1.b; Step 1).
- $S_4$-equivariance on the 6 edges (from the srs primitive-cell vertex-permutation group; Step 2).

The 14 non-$S_4$-invariant pair-partitions are each singled out by an $S_4$-breaking choice — an arbitrary preference for one vertex or one non-invariant structure. Under A2 refined, such singled-out choices carry no lower description length than the MDL-minimum S_4-invariant choice (the matching partition). The matching partition is the unique choice that requires no $S_4$-breaking input.

Therefore the canonical Cartan subalgebra $\mathfrak{h} \subset \mathfrak{spin}(V)$ consistent with A1+A2+B1.b+srs-point-group is the matching-partition Cartan:

$$
\mathfrak{h} = \mathrm{span}_{\mathbb{R}}\left\{ B_1, B_2, B_3 \right\},
\qquad B_a = \Gamma_{e_a^{(1)}} \Gamma_{e_a^{(2)}} \text{ for } M_a = \{e_a^{(1)}, e_a^{(2)}\}.
$$

Setting $(T_1, T_2, Y) := (B_1/(2i), B_2/(2i), B_3/(2i))$ gives three mutually commuting Hermitian generators with the standard $\mathrm{spin}(6)$ Cartan structure.

### Step 7 — C_3 cycles the Cartan triple

$\sigma$ acts on the 3 matchings as a 3-cycle (`proofs/foundations/K4_matchings_C3_check.py`). Hence $\sigma$ maps $B_a \mapsto B_{\sigma(a)}$ for some permutation $\sigma$ on the indices $\{1, 2, 3\}$ of order 3. Correspondingly, the Cartan generators $(T_1, T_2, Y)$ are cyclically permuted by $\sigma$. **None of $T_1, T_2, Y$ is individually $C_3$-invariant**; the only $C_3$-invariants in $\mathfrak{h}$ are the power-sum combinations $T_1 + T_2 + Y$ and $T_1 T_2 T_3$ (plus higher symmetric functions).

~~This is the non-trivial coupling that `predictions/V_us_derivation.md` §3(iii) identified as the missing structural input to unblock CKM.~~

**RETRACTED (session 7 correction):** the original draft claimed this completed the CKM structural unblock of `V_us_derivation.md` §3(iii). That claim was wrong. While the Cartan triple is genuinely $C_3$-cycled on the spinor factor, the restriction of any tensor-product $\sigma_{\text{combined}}$-invariant mass operator to a species gives a circulant Yukawa matrix on $C^3_{\text{obs}}$, and all circulants diagonalize via the same $\mathrm{DFT}_3$. Hence $V_{\mathrm{CKM}} = I$ identically for this class of mass operators. See Open Questions §4.

## Result

**Theorem 3b (canonical Cartan via S_4 invariance).** Under A1 + A2 + B1.b + srs primitive-cell $S_4$ vertex-symmetry, the canonical Cartan subalgebra $\mathfrak{h}$ of $\mathfrak{spin}(V) \subset \mathrm{Cl}(V, Q)$ is the matching-partition Cartan

$$
\mathfrak{h}_M = \mathrm{span}\{\,\Gamma_{M_1}, \Gamma_{M_2}, \Gamma_{M_3}\,\}
$$

where $M_1, M_2, M_3$ are the three perfect matchings of $K_4$. The body-diagonal $C_3$ generator $\sigma$ cyclically permutes the three generators.

**Numerical checks (from the prediction script):**
- Total 3-way pair-partitions of 6 edges: 15.
- $S_4$-invariant pair-partitions: exactly 1 (the matching partition).
- Perfect matchings of $K_4$: 3.
- $\sigma$-order on matching-space: 3 (regular rep).

**Scope disclaimer (post-session-7):** this is a purely algebraic theorem about Cl(V, Q)'s canonical Cartan subalgebra. It says nothing directly about CKM, masses, or sector-universality of Yukawa matrices. Attempts to chain this into numerical V_us/V_cb/V_ub are documented in an internal working note; current gap is §11 (tensor-product mass operators give $V_{\mathrm{CKM}} = I$ identically, regardless of species labeling).

## Comparison with experiment

| Quantity | Predicted | Observed (PDG 2024) | Status |
|----------|-----------|---------------------|--------|
| Canonical Cartan partition | matching $P_M$ | not directly observable | structural |
| $\sigma$-action order on Cartan | 3 | not directly observable | structural |

This theorem is purely algebraic. It has no direct experimental comparison. Whether it contributes to downstream SM-observable predictions is an open question not resolved here.

## Open questions

1. **Derivation that A2 forces the canonical Cartan to be $S_4$-invariant.** Step 6 invokes "selective retention of $S_4$-invariant structures" as an A2 consequence. Under A2 refined, the framework retains $S_4$-equivariant structures; whether this single-handedly FORCES the canonical Cartan choice to be the unique $S_4$-invariant one (vs, e.g., retaining a $C_3$-invariant superposition of $P_M, P_A, P_B$) is the one sub-step not fully nailed. Plausible under the symmetric structure (the matching-partition is MDL-minimal by its characterization via pure symmetry); not yet derived with full rigor under parameter_linter's hard gate. This is the one reason the current grade is `mathematically complete` rather than `theorem`.

2. **B3's Pati-Salam labeling as a C_3-orbit.** B3's conventional labeling (`predictions/theorem_B3_spinor_fermion_derivation.md` Step 2) singles out $Y$ as the $U(1)_{B-L}$ generator. Under Step 7 of this theorem, $(T_1, T_2, Y)$ is a $C_3$-3-cycle — no individual generator is distinguished. The resolution candidate: the three $C_3$-cycled Pati-Salam frames coexist under refined A2, and B3's labeling is a choice of frame. Structurally analogous to R3's regular-rep observer $Z_3$ where the generation label is a $C_3$-cycled choice. Full formalization not attempted here.

3. ~~**Numerical $V_{us}, V_{cb}, V_{ub}$.** [Previously claimed as "mechanical in principle."] This was wrong — see Open Question 4.~~

4. **CKM remains BLOCKED; 3b by itself does NOT unblock it.** Session 7 verification showed: for any $\sigma_{\text{combined}}$-invariant tensor-product mass operator $M = \sum \alpha_{a,b}\, \sigma_S^a \otimes \sigma_{\text{obs}}^b$ on $S \otimes C^3_{\text{obs}}$, the restricted Yukawa matrices $Y_X$ for every species $X$ are circulant on $C^3_{\text{obs}}$ — all diagonalized by the same $\mathrm{DFT}_3$ unitary. Therefore $V_{\mathrm{CKM}} = U_u^\dagger U_d = I$ identically, regardless of which species is assigned which $\sigma_S$ eigenvalue. The canonical-Cartan content of this theorem does not propagate to a CKM unblock. For CKM $\neq I$ one needs a mass operator that breaks $\sigma_{\text{obs}}$-invariance; this is a separate structural question not in 3b's scope. See an internal working note §11.

4. **A_4 vs S_4 invariance.** Step 5 established that the matching partition is both $A_4$- and $S_4$-invariant. Under I4_1 32 (chiral space group), the point group is $432 = O$, which is isomorphic to $S_4$. Under the mirror space group I4_3 32 (the opposite chirality, present under refined A2), the point group is also isomorphic to $S_4$ (the rotation groups coincide). The two chiralities' $S_4$ actions differ by a mirror twist; details of how this interacts with $C_3$-sign swaps (an internal working note §4a) warrant further analysis.

5. **Relation to gamma_7 chirality attempt.** an internal working note transferred $\Gamma_7$ to $V_{\text{Ram}}$ via the B6 bridge but was BLOCKED by gauge dependence (U(4) trivial sector). The present theorem sidesteps that issue by working directly with the matching-partition Cartan on Cl(V, Q) rather than lifting through V_Ram. Whether the two approaches coincide after symmetry reduction is open.

## References

- Coxeter, H. S. M. (1973). *Regular Polytopes*, 3rd ed. Dover. §4.4.
- Dummit, D. S. & Foote, R. M. (2004). *Abstract Algebra*, 3rd ed. Wiley. §2.2 Ex 4.
- International Tables for Crystallography, Vol. A (2016). Space group I4_1 32 (No. 214).
- Lawson, H. B. & Michelsohn, M.-L. (1989). *Spin Geometry.* Princeton Univ. Press. Ch. I §1, §6.
- Particle Data Group (2024). *Review of Particle Physics*. CKM review for $V_{us}, V_{cb}, V_{ub}$.

## Files referenced

- `predictions/B3_chirality_bridge.py` — this file's script.
- `predictions/theorem_B1_ordering.py` — upstream B1.b (chain-imported).
- `proofs/foundations/K4_matchings_C3_check.py` — K_4 perfect matchings + sigma cyclicity (§Step 7).
- `proofs/foundations/K4_C3_equivariant_pairings.py` — enumeration of 3 C_3-equivariant partitions (§Step 4).
- `proofs/foundations/K4_S4_A4_invariant_pairings.py` — uniqueness of matching under S_4 (§Step 5).
- `predictions/theorem_B3_spinor_fermion_derivation.md` — B3 Brauer-Weyl realization (context for Pati-Salam labeling).
- `docs/framework/B3_B6_reconciliation.md` — prior finding that $[T_a, U_{C_3}^S] = 2.0$ (consistency with Step 7).
- `predictions/V_us_derivation.md` — downstream consumer; structural input (iii) now provided.
- `docs/parameters/parameter_linter.md` — rigor gate applied.

## Verification

```
python predictions/B3_chirality_bridge.py
```

Expected final line: `OK: predictions/B3_chirality_bridge.py verification complete.`
