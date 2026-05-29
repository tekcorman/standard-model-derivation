# Theorem: the three fermion generations are an A₄ irreducible triplet

> ## ⚠ RETRACTED — 2026-05-22 (Stage 14)
>
> **This theorem does not stand.** Its load-bearing cited input — "the
> geometric A₄ acts at the P-point" — was verified in Stage 14
> (`gauge_hub_stage14_p_point_little_group_2026-05-22.py`) and **fails as
> stated**. The scalar Bloch adjacency `A(P)` has eigenvalue-multiplicity
> partition **(2,2)**; a *linear* A₄ permutation representation on the 4
> atoms is `1 ⊕ 3`, whose commuting Hermitian operators have partition
> `(1,3)` (Schur). `(2,2) ≠ (1,3)` — so A(P) is **not** linear-A₄-permutation
> -equivariant. The P-point little group does act (all 12 even permutations
> admit phase-dressed *monomial* symmetries) but **projectively** — it is the
> binary tetrahedral group `2T = SL(2,3)`, and the `(2,2)` degeneracy is
> carried by a **2-dimensional** `2T`-irrep. Consequently the 4-atom rep `H₄`
> is built from 2-dim irreps, `V_Ram = 2·H₄` has **no 3-dimensional
> subrepresentation**, and §3's Corollary is vacuous. The three generations
> are **not** established as an A₄ irreducible triplet.
>
> **What survives:** B7.1 (`dim C³_gen = 3`) — never used A₄; the C₃
> body-diagonal action at P and `V_Ram`'s C₃-decomposition `(4,2,2)`;
> Stage 5's independent gauge-hub wall. The text below is kept as a recorded
> attempt. Any future generation-symmetry derivation must use `2T` and its
> 2-dim irreps — a different representation theory. See
> `gauge_hub_stage14_*` and the scoping doc Stage 14.

**Date:** 2026-05-22
**Status:** ~~THEOREM-GRADE-CONDITIONAL~~ **RETRACTED 2026-05-22** (Stage 14 —
the cited input "A₄ acts at P" is false; the little group acts projectively
as `2T`). The text below is a recorded attempt; Need-A2's logical core is
**not** closed.
**Probes:** `proofs/foundations/gauge_hub_stage{9,10,11,12,13}_*.py` (built the
attempt); `gauge_hub_stage14_*` (retracts it).
**Lineage:** the 2026-05-21/22 gauge-hub arc — see
an internal working note Stages 9–12.

---

## 1. Statement

The observer's generation space `C³_gen` (3-dimensional by B7.1) carries the
**unique 3-dimensional irreducible representation of A₄**, the geometric
tetrahedral group. The three fermion generations are therefore a derived
**A₄ triplet** — not a posited "there are three." The fermion mass operator,
a generic Hermitian operator on `C³_gen`, *breaks* A₄ and so produces three
distinct masses — exactly the architecture of A₄ discrete-flavour models,
here derived rather than assumed.

## 2. The objects

- **A₄** — the geometric tetrahedral group, the chiral point-symmetry of the
  srs lattice. It is a subgroup of srs's point group `432 = O` and acts on
  the 4 atoms of the primitive cell (the vertices of the K₄ quotient). It is
  the stabiliser of the Bloch point `P = (¼,¼,¼)`.
- **`V_Ram`** — the 8-dimensional Ramanujan subspace of the non-backtracking
  operator `B(P)`: the part of the 12-dim arc space whose spectrum lies on
  the Ramanujan circle `|μ| = √(k*−1) = √2`.
- **`C³_gen`** — the observer's generation Hilbert space; `dim = 3` by B7.1
  (`predictions/observer_dim_three_derivation.md`, MDL + Gleason).

## 3. The proof, assembled

**Stage 9 — A₄ acts on the data.** The geometric tetrahedral A₄ = ⟨C₃, V_B⟩
lies in srs's point group `432`; srs is vertex-transitive; hence A₄'s
elements are genuine substrate automorphisms — they permute the directed
edges and so act unitarily on the walker, on `B(P)`, and on `V_Ram`.
(`gauge_hub_stage9`.)

**Stage 11 — the A₄-decomposition of `V_Ram`.** By the Ihara–Bass
correspondence the Ramanujan subspace is two copies of the 4-dim scalar
adjacency space `A(P)` — one per non-backtracking branch `μ₊/μ₋`; the branch
space is a *trivial* 2-dim A₄-rep because A₄ preserves eigenvalues. So
`V_Ram ≅ H₄ ⊕ H₄` as A₄-representations, where `H₄` is A₄ acting on the 4
atoms. `H₄` is the permutation representation of A₄ on 4 points, `= 1 ⊕ 3`.
Therefore

```
        V_Ram  =  2·(1)  ⊕  2·(3).
```

The C₃-shadow `(4,2,2)` reproduces the framework's prior theorem-grade
decomposition. (`gauge_hub_stage11`.)

**Stage 12, Lemma (MDL exploits symmetry).** *For A₄-invariant substrate
data, the MDL-optimal model is A₄-equivariant.* Proof: (i) A₄ acts unitarily
on the data space and fixes the data; (ii) the description-length functional
is then A₄-invariant; (iii) the data-fit term `L(data|ρ) = −Σ nᵈ log Tr(ρΠᵈ)`
is convex in `ρ` (−log of a quantity linear in `ρ`); (iv) for any optimal
model `M*`, the A₄-average `M̄ = |A₄|⁻¹ Σ U_g M* U_g†` is A₄-equivariant and,
by convexity, attains the same data-fit optimum; (v) an equivariant model is
specified within the commutant — `(d_comm − 1) ≤ n²−1` parameters — so costs
no more, and is the canonical optimum. ∎ (`gauge_hub_stage12` G1–G4, verified
exactly.)

**Stage 12, Corollary (the triplet).** The MDL-optimal `C³_gen` is therefore
a 3-dimensional A₄-equivariant compression of `V_Ram` — i.e. a 3-dim
A₄-subrepresentation (A₄ finite ⇒ semisimple). `V_Ram = 2·(1) ⊕ 2·(3)` has
**no `1′`, no `1″`, and only two trivials** — so its *only* 3-dimensional
subrepresentation is a copy of the irreducible triplet `3`. Hence

```
        C³_gen  ≅  the A₄ irreducible triplet 3.       ∎
```

**Why the sharper, Stage-10 reading still matters.** Stage 10 showed that,
*among the abstract 3-dim representations of A₄*, equivariant-MDL singles out
the irrep by minimal commutant dimension (`1`, versus `3/5/9` for the
reducibles), and that the Klein-four `V_B` — not the node-local `C₃` — is the
discriminator that makes that distinction (`C₃` alone cannot separate `3`
from `1+1′+1″`). The Corollary above is sharper still: given `V_Ram`'s actual
content, the irrep is the *only* option, not merely the shortest. Stage 10
defeated the **Block-C2** objection ("MDL is blind to representation
content") that had made this route look dead — that objection counted a
generic density operator; the symmetry-adapted count is the commutant
dimension, which sees the representation.

## 4. What it resolves

This closes the **T-equivariance sub-target of Need-A2 Route 3**. The
candidate-route doc named three required steps; all are now supplied:
formal A₄-invariance of the frame-function space (Stage 9 + Lemma step i–ii);
the proof that A₄-invariance of `B(P)` ⇒ the MDL extraction is A₄-equivariant
(the Lemma); and the argument that the resulting 3-dim representation is the
irrep, not a sum of 1-dimensionals (the Corollary, via `V_Ram`'s content).
**Block-C2 is defeated.**

## 5. Honest grade — cited inputs and what remains

**THEOREM-GRADE-CONDITIONAL.** The chain is rigorous given:

1. **B7.1** — `dim C³_gen = 3` (MDL + Gleason; `observer_dim_three_derivation.md`,
   theorem-grade).
2. **A₄ = the P-point stabiliser**, acting on the 4 atoms by the K₄-quotient
   permutation (Bradley–Cracknell Table 3.7; Stage 9 — A₄ ⊆ point group `432`).
3. The B7.1 **data-fit functional is a Born-rule code length** (so convex in
   `ρ`) — B7.1's own Step 4.

**Block-1′ — RESOLVED for the symmetries (2026-05-22, Stage 13).**
`gauge_hub_stage13_block1prime_generation_vs_colour_2026-05-22.py` (4/4).
Block-1′ feared a substrate-derived generation-Z₃ that is secretly the
colour-Z₃. The decisive point: Stages 9–12 derive a generation-**A₄** — the
non-abelian tetrahedral group — *not* a generation-Z₃. A non-abelian order-12
group is not the abelian colour-Z₃ (G1). Even restricted to the C₃: the
colour centre acts on its triplet as a *scalar*, the generation C₃ as the
*regular* representation — three distinct eigenvalues (G2). As an SU(4)
element the body-diagonal C₃ has eigenvalue partition `(2,1,1)`, the
colour-centre element `(3,1)` — not conjugate, hence distinct (G3,
corroborating `B3_B6_reconciliation.md` Finding 2, `‖[U_C₃,PS-Cartan]‖ = 2`).
Independent origin: the generation A₄ is the geometric point group `432`
(Stage 9), colour SU(3) is the internal Cl(6) structure; the Klein-four `V_B`
is what colour lacks and what makes the triplet irreducible (Stage 10).
**One finer question remains, separate and lesser:** whether `C³_gen` and
`C³_colour` are distinct *spaces* (separate tensor factors). The framework
asserts this — R3 Lemma L1, `H_fermion = C³_gen ⊗ H_gauge ⊗ H_spinor` — but
the identification `V_Ram ≅ Cl(6) Fock` is flagged research-open. It does not
threaten the symmetry-distinctness above (A₄ ≠ SU(3); the triplet's
irreducibility rests on `V_B`, which colour has no analogue of). With
Block-1′ resolved for the symmetries, **Need-A2's logical core is closed**;
the residual is the space-level identification only.

## 6. Cross-references

- `proofs/foundations/gauge_hub_stage9_scope_gap_vc_vb_2026-05-21.py` — A₄ acts on the walker.
- `proofs/foundations/gauge_hub_stage10_generation_irrep_equivariant_mdl_2026-05-22.py` — equivariant-MDL vs Block-C2.
- `proofs/foundations/gauge_hub_stage11_v_ram_a4_decomposition_2026-05-22.py` — `V_Ram = 2·(1) ⊕ 2·(3)`.
- `proofs/foundations/gauge_hub_stage12_mdl_symmetry_lemma_2026-05-22.py` — the Lemma + Corollary.
- `proofs/foundations/gauge_hub_stage13_block1prime_generation_vs_colour_2026-05-22.py` — Block-1′ (generation A₄ ≠ colour-Z₃).
- `docs/framework/B3_B6_reconciliation.md` — Finding 2 (`‖[U_C₃, PS-Cartan]‖ = 2`).
- `predictions/observer_dim_three_derivation.md` — B7.1 (`dim = 3`).
