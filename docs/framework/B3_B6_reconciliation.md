# B3-B6 reconciliation — what does U_{C_3}^S actually do on the Cl(6,0) spinor?

**Date:** 2026-04-17
**Status:** Structural reconciliation report. Identifies a precise mathematical
error in B6 Step 7 (the "C_3 = Z(SU(3)_color) in PS" identification) and
proposes a corrected interpretation. Does NOT modify B3, B5.3-core, or B6.
**Script:** `proofs/foundations/theorem_B3_B6_reconciliation.py`
**Sprint:** 9, follow-up to an internal working note.

## The conflict, stated precisely

Two upstream theorems make claims about the same 8-dim Cl(6, 0) Dirac spinor S
that, taken at face value, are mutually inconsistent.

### B3's claim (verbatim from `../../predictions/theorem_B3_spinor_fermion_derivation.md`)

> "the 8-dim Cl(6, 0) Dirac spinor decomposes under the natural Spin(4) ×
> Spin(2) = SU(2)_L × SU(2)_R × U(1)_{B−L} subgroup as the electroweak
> content of exactly one Standard-Model generation **with colour factored
> out**, i.e. the colour-trivialised Pati–Salam multiplet (ν, e, u, d) ×
> (L, R)." (Abstract.)

> "Collapsing the Pati–Salam SU(4) fundamental 4 → {lepton, quark} by
> **factoring out colour** … is the one-generation electroweak fermion
> content of the Standard Model, {ν_L, e_L}, {u_L, d_L}, {ν_R, e_R},
> {u_R, d_R}, one SU(2) doublet per (chirality, lepton-or-quark) pair,
> eight states in total." (Statement.)

So: each of B3's 8 states is a single colorless lepton or quark species, and
the named "u_L, d_L" are SU(2)_L-doublet quark axes without color structure.

### B6's claim (verbatim from `docs/theorem_B6_bridge.md`)

> "Let U_{C_3}^S : S → S denote the lift of the C_3 action on H_{K_4} via
> Spin(6), realized concretely on S ≅ ℂ^4 ⊕ ℂ^4 by the SU(4) element with
> eigenvalues (1, 1, ω, ω²) on the fundamental 4 …" (Statement.)

> "Under the PS embedding SU(4) → SU(3)_color × U(1)_{B−L}, the SU(4)
> eigenvalues (1, 1, ω, ω²) split as: the '1' pair contains the lepton
> (singlet under SU(3)_color) and one quark color (one entry of the color
> triplet); the (ω, ω²) pair contains the other two quark colors. The C_3
> acts on the color triplet as the cyclic Z_3 ⊂ SU(3)_color (the center of
> SU(3)_color acting on the fundamental). On the lepton singlet, C_3 acts
> trivially." (Statement (iii).)

> "the C_3 of B5.3-core is identified, through the bridge, with the cyclic
> Z_3 ⊂ SU(3)_color in the Pati-Salam embedding. The three C_3 irreps label
> color components (within one generation), not generations." (Result.)

So: the same S carries SU(3)_color structure, with each chirality split as
1 lepton + 3 quark colors (= 4 states per chirality), and the C_3 acts as
the Z_3 center of SU(3)_color, permuting the 3 quark colors and fixing the
lepton.

### Why these can't both be literal

Per chirality, B3 reads S^± as `(lepton SU(2)_L doublet) ⊕ (quark SU(2)_L
doublet)`: 2 lepton states + 2 quark states. B6 reads S^± as `1 lepton + 3
quark colors` (with no SU(2)_L doublet structure within S^±). The state
count per chirality is 4 either way, but the assignment of which states
are leptons and which are quarks differs sharply, and the symmetry group
structure (SU(2)_L doublet structure vs SU(3)_color triplet structure) is
incompatible.

The conflict is sharper than a labeling ambiguity, because the C_3 action
of B6 should preserve the lepton/quark distinction (since Z(SU(3)_color)
fixes the lepton singlet and only permutes the color triplet) — but B3's
lepton/quark distinction is the eigenvalue of Y = (1/2i) Γ_{56}, the
U(1)_{B−L} generator. So the question is sharp: does U_{C_3}^S commute
with Y or not?

## What the math says (sympy/numpy verification)

`proofs/foundations/theorem_B3_B6_reconciliation.py` constructs the
explicit Spin(6) lift U_{C_3}^S using B6's exact recipe, applied to B3's
exact Brauer-Weyl Cl(6, 0) realization, and computes:

### Finding 1 — U_{C_3}^S has the SU(4) eigenvalue spectrum B6 claims

Per Weyl sector: eigenvalues `{1, 1, ω, ω²}` exactly on S^+, and `{1, 1,
ω², ω}` exactly on S^−. Total isotypic decomposition on S is `(4, 2, 2)`,
matching both the prediction of B6 and the (4, 2, 2) Ramanujan content of
B(P).

This part of B6 is correct.

### Finding 2 — U_{C_3}^S does NOT commute with B3's Cartan generators

Computed commutator norms (machine precision):

| Operator | `||[A, U_{C_3}^S]||` |
|----------|----------------------|
| T_1      | 2.000                |
| T_2      | 2.000                |
| Y        | 2.000                |
| T_L = T_1 + T_2 | 2.828         |
| T_R = T_1 − T_2 | 2.828         |

In particular **`||[Y, U_{C_3}^S]|| = 2.000`, not zero.** This is the
load-bearing check: Y is the U(1)_{B−L} generator that B3 uses to
distinguish "lepton" (Y < 0 in the chosen convention) from "quark"
(Y > 0). U_{C_3}^S mixes Y-eigenvalues, so it does NOT preserve B3's
lepton/quark distinction.

### Finding 3 — Lepton ↔ quark mixing is maximal in B3's species basis

The 8 × 8 matrix of U_{C_3}^S in B3's species basis has uniform
absolute-value entries `|M_{ij}| ∈ {0, 0.5}`, with the non-zero entries
covering 4 columns × 4 rows in two interlocking "blocks" connected by
chirality. Per chirality, in B3's species basis (e_L, ν_L, u_L, d_L) for
example:

```
            e_L    nu_L   u_L    d_L
   e_L:   0.500  0.500  0.500  0.500
   nu_L:  0.500  0.500  0.500  0.500
   u_L:   0.500  0.500  0.500  0.500
   d_L:   0.500  0.500  0.500  0.500
```

This is essentially a 4 × 4 Hadamard-type rotation on each chirality —
U_{C_3}^S maps each B3 species state to a uniform superposition of all
four species states of the same chirality. **There is no preserved
"lepton subspace" inside S^+ under U_{C_3}^S.**

### Finding 4 — B6 Step 7's claim "(1, 1, ω, ω²) = Z(SU(3)_color)" is wrong

The Z_3 center of SU(3)_color, embedded in SU(4) under the standard PS
embedding SU(4) ⊃ SU(3)_color × U(1)_{B−L}, acts on the SU(4)
fundamental 4 = 3_{+1/3} ⊕ 1_{−1} as

$$
g_{Z_3} = \mathrm{diag}(z, z, z, z^{-3}), \qquad z^3 = 1
$$

(or, equivalently, `diag(z, z, z, 1)` modulo the U(1)_{B−L} compensator
that keeps the determinant = 1). Its SU(4) eigenvalues are
`(z, z, z, z^{-3})` — three repetitions of the same root, plus one for
the lepton.

By contrast, the body-diagonal C_3 of srs lifts to SU(4) with eigenvalues
`(1, 1, ω, ω²)` — a generic order-3 Cartan element of SU(4) with
eigenvalue multiplicities `(2, 1, 1)`, not `(3, 1)`. **This is not in
the Z_3 center of SU(3)_color** under any embedding compatible with the
standard PS branching.

To verify: a Z_3 center element with z = ω acts on Λ^2(C^4) (the V_6 =
Spin(6) vector representation) with eigenvalues `{z·z, z·z, z·z, z·1,
z·1, z·1} = {ω², ω², ω², ω, ω, ω}` — 3 of ω² and 3 of ω. The actual srs
body-diagonal C_3 on V_6 = ℝ^6 has eigenvalues `{1, 1, ω, ω, ω², ω²}`
(per B6 Step 2, verified). These two spectra are different. So B6's
identification is mathematically incorrect on its own setup.

### Finding 5 — sin²θ_W on S is not 3/8 under either reading

(Already computed in an internal working note.
Recapped here for completeness.)

| Reading | dim | Tr(T_3²) | Tr(Q²) | sin²θ_W |
|---------|-----|----------|--------|---------|
| (I) B3 colorless: 1 lepton + 1 quark axis per chirality | 8 | 1 | 28/9 ≈ 3.111 | 9/28 ≈ 0.321 |
| (II) B6 colored: L Weyl = (ν_L, u_L^{r,g,b}), R Weyl = (e_R, d_R^{r,g,b}) | 8 | 2 | 8/3 ≈ 2.667 | 3/4 = 0.750 |
| Standard SM target | 16 | 2 | 16/3 ≈ 5.333 | 3/8 = 0.375 |

Neither reading gives 3/8 on the 8-dim S. The standard SM target requires
16 dimensions (3 quark colors × SU(2)_L doublet + 1 lepton SU(2)_L
doublet = 8 per chirality, 16 total).

## Mapping to options (I)–(IV)

The structural question proposed four options. The math supports a
modified version of (IV).

**(I)** B6 right, B3 loose. **REJECTED.** B6's "C_3 acts on S as
Z(SU(3)_color)" claim does not hold algebraically: the SU(4) eigenvalues
(1, 1, ω, ω²) of the actual lift are NOT the eigenvalues of any
Z(SU(3)_color) center element under standard PS embeddings.

**(II)** B3 right, B6 loose. **REJECTED.** B6 is correct that U_{C_3}^S
acts non-trivially on S — it has the (4, 2, 2) isotypic decomposition
verifiably. The C_3 does live ON S, so B3's species labels (which require
the C_3 to act trivially or to permute leptons among themselves) are
not C_3-stable.

**(III)** Both right on different layers (tensor factorization H_full = S
⊗ H_color). **REJECTED.** No such factorization exists in the framework.
B6's lift is constructed via Spin(6) ≅ SU(4) ON THE SAME 6-edge
quadratic space that defines Cl(6, 0) → S; the resulting U_{C_3}^S acts
on S itself, not on a separate factor. There is no H_color layer to which
the C_3 could be relegated without external augmentation (B4 Route iv).

**(IV)** Both need revision. **THE CORRECT VERDICT, in the following
precise form:**

- B3's "colorless reading" is **structurally consistent** as an SU(2)_L ×
  SU(2)_R × U(1)_{B−L} = Spin(4) × Spin(2) decomposition of S. The
  species labels (ν, e, u, d) × (L, R) are valid Spin(4) × Spin(2)
  weight labels. **What B3 must NOT claim** is that S contains the
  electroweak content of one PS generation with color "factored out" in
  a way that survives the addition of an external color action — there
  is no consistent way to add 3 colors to B3's S to reach 16 SM states
  unless an external tensor factor C^3_color is explicitly postulated
  (B4 Route iv). The current language "colour factored out" is misleading
  because it suggests color has been removed, but in fact color was never
  there to begin with on S; B3 = electroweak only.

- B6's algebraic content (the SU(4) lift, the eigenvalues (1,1,ω,ω²),
  the (4,2,2) isotypic split, the explicit Spin(6) construction on S)
  is **correct**. **What B6 must NOT claim** is that the resulting
  C_3 action is the Z_3 center of SU(3)_color in the PS embedding
  (Step 7). The element with eigenvalues (1,1,ω,ω²) on the SU(4)
  fundamental is not in Z(SU(3)_color) under any embedding consistent
  with the standard PS branching 4 → 3_{+1/3} + 1_{−1}. It is a generic
  order-3 Cartan element of SU(4).

The math therefore supports a corrected joint interpretation:

## Corrected interpretation (proposed theorem)

**Theorem (B3 ⊕ B6 reconciled; proposal — verification deferred until
B3 / B6 themselves are revised).** Let S = ℂ^8 be the Cl(6, 0) Dirac
spinor of B3, and let U_{C_3}^S : S → S be the Spin(6) lift of the body-
diagonal C_3 action on the 6-edge K_4 quadratic space (per B6 Steps 1–5).
Then:

(a) **(Spin(4) × Spin(2) structure on S, from B3.)** Under Spin(4) ×
Spin(2) = SU(2)_L × SU(2)_R × U(1)_{B−L}, S decomposes as

$$
S = \big[(\mathbf{2}, \mathbf{1})_{+1} \oplus (\mathbf{1}, \mathbf{2})_{-1}\big]_L
   \oplus \big[(\mathbf{2}, \mathbf{1})_{-1} \oplus (\mathbf{1}, \mathbf{2})_{+1}\big]_R
$$

with weight labels (ε_1, ε_2, ε_Y) ∈ {±1}^3. **S carries no color
structure.**

(b) **(SU(4) Cartan structure on S, from B6.)** U_{C_3}^S is the image
in SU(4) ≅ Spin(6) of the SU(4) Cartan element

$$
g_{C_3} = \mathrm{diag}(1, 1, \omega, \omega^2) \in \mathrm{SU}(4),
$$

acting as `(1, 1, ω, ω²)` on S^+ = 4 and `(1, 1, ω², ω)` on S^− = 4̄,
with C_3-isotypic multiplicities (4, 2, 2) on S.

(c) **(The two structures are not aligned.)** U_{C_3}^S does not commute
with the Spin(4) × Spin(2) Cartan generators T_1, T_2, Y, T_L, T_R. In
particular [Y, U_{C_3}^S] ≠ 0 (commutator norm 2.0 in the standard
normalization). Hence a state cannot simultaneously have a definite B3
species label (T_1, T_2, Y) and a definite C_3 irrep label.

(d) **(The C_3 is NOT Z(SU(3)_color).)** The element g_{C_3} =
diag(1, 1, ω, ω²) is not in Z(SU(3)_color) ⊂ SU(4) under any embedding
consistent with the PS branching 4 → 3_{+1/3} + 1_{−1}. The Z_3 center
of SU(3)_color in this embedding has eigenvalues of the form (z, z, z,
z^{-3}) on the SU(4) fundamental — three equal roots plus one — which
is not the (1, 1, ω, ω²) spectrum.

(e) **(Status of the (4, 2, 2) decomposition.)** The (4, 2, 2) isotypic
decomposition of S under U_{C_3}^S is genuine and matches the (4, 2, 2)
Ramanujan multiplicities at P (B5.3-core / BP). The bridge of B6 closes
in the dimension-counting / character-matching sense: dim S = 8 =
dim Ram(P), and the C_3-character of the two sides agrees. **What does
NOT close** is the physical identification of the C_3 irrep label with
"color" — that identification, as stated in B6 Step 7, is a
misidentification.

(f) **(What can the (4, 2, 2) be, then?)** Three live options remain:

  - **(α) Generation index** (the original pre-B6 reading from B5.3-core
    Step 4 / W4 catalog Type A). The (4, 2, 2) is a generation label
    on S — i.e., the trivial sector contains 1 generation worth of
    species and the ω, ω² sectors share the other 2 generations. This
    requires that S not be "one generation" but somehow encode 3
    generations × ⅔ each, which is not B3's reading. This option needs
    a separate justification.

  - **(β) An SU(4) Cartan label** (purely algebraic). The (4, 2, 2) is
    just the irrep multiplicity of a generic Cartan element of SU(4)
    on the 4 ⊕ 4̄. It carries no physical interpretation absent
    additional structure. Most honest fallback.

  - **(γ) A non-PS embedding.** The (1, 1, ω, ω²) might be the Z_3
    center of some non-standard SU(3) ⊂ SU(4) with a different lepton
    assignment than the PS one (e.g., one in which the "lepton" slot
    is one of the (1, 1) pair and the "color triplet" is the other
    three slots arranged as (1, ω, ω²) — i.e., a non-trivial cyclic
    color permutation rather than the trivial Z_3 center action). This
    would be a different group than SU(3)_color in the literal sense,
    and would need a separate identification argument. **Note**:
    diag(1, ω, ω²) ∈ SU(3) is a Weyl element of SU(3) but is NOT in
    Z(SU(3)) — it's the cyclic generator of the maximal torus modulo
    Z_3. So even under (γ) the identification "Z_3 center of color
    SU(3)" is wrong; what we'd have is "the cyclic permutation of color
    triplet via a maximal torus generator", which is a different
    physical statement.

The default reading until further structure is found is (β): the (4, 2,
2) is an SU(4) Cartan multiplicity, not a color label.

## Consequences for downstream

### `docs/theorem_B6_bridge.md` — Step 7 needs revision

Step 7's claim "the C_3 induced on S is precisely the Z_3 center of
SU(3)_color in the PS embedding" is mathematically incorrect. The
specific arguments to revise:

- **Step 7 paragraph "Pati & Salam 1974 §II … the four eigenvalues (1, 1,
  ω, ω²) of the C_3 element on 4 split as: ONE eigenvalue 1 assigned to
  the lepton singlet, and the remaining (1, ω, ω²) assigned to the three
  quark color components."** — This conflates two different SU(3)
  subgroups of SU(4): the *Z_3 center* of SU(3)_color (which is
  diag(z, z, z, z^{-3})) vs *a maximal torus element* of SU(3) (which
  is diag(1, ω, ω²) modulo a U(1) compensator). Only the latter has
  the (1, ω, ω²) spectrum on the color slot, but it is not in
  Z(SU(3)_color).

- **Conclusion of Step 7: "C_3 = color-Z_3" — INCORRECT as stated.** The
  honest replacement is: "C_3 is a generic order-3 Cartan element of
  SU(4); it lies in some SU(3) maximal torus but not in the Z_3 center."

- **Result paragraph 2 ("C_3 = color-Z_3").** Remove the color
  identification. State only that the (4, 2, 2) isotypic decomposition
  matches.

- **Result paragraph 3 ("CKM = I at tree level").** The dimensional
  argument (universal C_3-isotypic structure across up- and down-type
  Yukawas) still holds independently of whether the C_3 is "color" or
  "generation" — it depends only on the C_3-character of the
  representation. So this conclusion is unaffected; only its
  *interpretation* changes.

### `docs/theorem_B5_3_core.md` — interpretation re-opens

B5.3-core itself is unaffected as a structural theorem: the (4, 4, 4)
on the 12-dim Bloch Hashimoto bundle and the (4, 2, 2) on the 8-dim
Ramanujan subspace at P are both rigorous representation-theoretic
facts about the C_3 action. **What is no longer settled** is the
*physical interpretation* of the (4, 2, 2) — under B6 (corrected) this
is just the SU(4) Cartan multiplicity (4, 2, 2) of the body-diagonal
C_3 on the SU(4) ≅ Spin(6) lift, and is not yet identified with either
"color" or "generation."

The downstream "Type A" mass-amplitude derivations (`Q_Koide.py`,
`epsilon_Koide.py`, `delta_Koide.py`) currently rest on the
identification "C_3 irrep index j = generation index" (B5.3-core §
Consequence 2). Under the corrected B6, this identification reverts to
"interpretation TBD" — neither "color" (B6's revision) nor "generation"
(B5.3-core's pre-B6 reading) is currently derived. The Type A
predictions remain well-defined as numerical formulas, but their
*physical meaning* is now open.

### an internal working note — strengthened

Route γ for sin²θ_W stalled because the 8-dim S has too few states
(8 < 16) and because the C_3 was claimed to be Z(SU(3)_color) but
acted on the existing 8 states rather than generating 8 more.

Under the corrected B6, the second part of this stall is *sharper*:
the C_3 is not Z(SU(3)_color) at all — so even the "discrete color
multiplicity" intuition that motivated Route γ is misplaced.
Route γ is therefore fully closed as failed, with the additional
finding that B6's Step 7 needs revision.

The doc already states (in §"Reading B3's species labels under B6") that
the conflict between B3 colorless and B6 colored is "an open structural
question of the framework not addressed by either B3 or B6 alone." The
present reconciliation answers that question: **the colored reading of
B6 is mathematically wrong; the C_3 on S is genuinely a non-color SU(4)
Cartan element**. The colorless reading of B3 is consistent (modulo
removing the misleading "factored out" phrasing) but cannot give 3/8.

### CKM diagnosis

The Step 8 conclusion of B6 ("CKM = I at tree level") is **unchanged**:
it depends only on (Y_u, Y_d) sharing the same C_3-eigenbasis structure
on H_graph, which is a character-level statement. Whether the C_3 is
"color," "generation," or just an "SU(4) Cartan label" doesn't affect
this — universally the up- and down-type Yukawas inherit the same
character, and so they diagonalize together to give CKM = I at tree
level. CKM remains BLOCKED for the same reasons.

### The B4 Route (iv) external-color posture

B4 Route (iv) (color is external; the framework's Cl(6, 0) does NOT
contain SU(3)_color) is *strengthened* by the present finding. The
attempted internalization of color via the B6 C_3 = Z(SU(3)_color)
identification is rejected; the framework remains in the position B4
documents, with color as an external structural input.

## Whether B3 / B6 / B5.3-core need revision

### B3 (`../../predictions/theorem_B3_spinor_fermion_derivation.md`): minor language revision
recommended

The mathematical content of B3 is unaffected. The phrasing "with colour
factored out" and "colour-trivialised Pati–Salam multiplet" in the
abstract and the "Collapsing the Pati–Salam SU(4) fundamental 4 →
{lepton, quark} by factoring out colour" in Step 4 should be replaced
by "colour-free" or "without colour": there is no mathematical operation
of "factoring out color" on S; rather, S is the electroweak content
absent any color structure. The current phrasing invites the
B6-style misreading that color is somehow latent in S.

**Recommended replacement language**: "the electroweak fermion content of
one Standard-Model generation, restricted to colour-singlet states (no
SU(3)_color action is realized on S; color SU(3) is structurally external
per B4 Route iv)."

### B6 (`docs/theorem_B6_bridge.md`): Step 7 + Result + Open Questions need
substantive revision

- **Step 7** — the "C_3 = Z(SU(3)_color)" identification is incorrect
  algebraically. The element g_{C_3} = diag(1, 1, ω, ω²) is in SU(4) but
  not in Z(SU(3)_color) under any standard PS embedding. Replace Step 7
  with: "U_{C_3}^S is a generic order-3 Cartan element of SU(4) on the
  4 ⊕ 4̄. The physical interpretation of the (4, 2, 2) isotypic split
  remains open: candidates include (α) generation index, (β) pure
  algebraic SU(4) Cartan label with no physical content yet,
  (γ) cyclic SU(3) maximal-torus action (NOT the Z_3 center). None is
  currently derived from MDL + toggle."

- **Statement (iii)** of the theorem — same correction.

- **Corollary (bridge identification)** — keep the dimension and
  character match (it's correct), but remove the parenthetical "with
  C_3 acting as the cyclic Z_3 ⊂ SU(3)_color in the PS embedding."

- **Result §"What the bridge closes," item 2 ("C_3 = color-Z_3")** —
  remove. Replace with "The (4, 2, 2) isotypic decomposition matches
  on both sides; the physical interpretation of this label is open."

- **Open questions** — add: "(B6.6) Physical interpretation of the
  (4, 2, 2). The original B6 identification with Z(SU(3)_color) is
  algebraically incorrect; alternative readings (generation index, pure
  algebraic Cartan label, non-center SU(3) action) need to be
  evaluated."

### B5.3-core (`docs/theorem_B5_3_core.md`): no revision needed

B5.3-core is a representation-theoretic theorem about the C_3 action on
the 12-dim Bloch Hashimoto bundle, and is unaffected by the B6 revision.
What changes is the *downstream interpretation* B5.3-core's Consequences
section assigns to the (4, 4, 4) and (4, 2, 2) — but the theorem's
mathematical content is fine.

The "promotion" claim in the Abstract — "The 'generation = C_3 irrep
index j ∈ {1, ω, ω²}' identification … is **promoted from postulate to
theorem** for the on-axis (Γ-P) content of the Bloch bundle." — is
restored by the present finding (B6's "C_3 = color" challenge is
withdrawn), so the original "C_3 = generation" reading remains an open
candidate, on equal footing with B6's now-rejected "C_3 = color"
reading. **Neither identification is currently a derived theorem.**

The Consequences section of B5.3-core (items 2 and 3, downstream Koide /
PMNS scripts) should be flagged: "the upstream justification 'C_3 =
generation irrep at P' is currently not a derived theorem. The B6 bridge
attempted to identify C_3 = color and rejected this on `docs/theorem_B3_
B6_reconciliation.md`. The C_3 label on H_graph remains uninterpreted
physically."

## Open question

**(R1) What is the C_3 isotypic label, physically?** Of the three live
options:

  (α) "C_3 = generation" — the pre-B6 default; consistent with the
  Type A mass-amplitude derivations but not derived from MDL + toggle
  (no proof that this is the right identification rather than just a
  numerical fit).

  (β) "C_3 = pure algebraic SU(4) Cartan label" — consistent with all
  the math but assigns no physics to the (4, 2, 2). Most honest
  fallback.

  (γ) "C_3 = some other SU(3) action (not Z(SU(3)_color), not
  generation)" — would need a new structural identification.

Without a derivation that picks one, the framework should default to
(β) and treat all downstream derivations resting on the C_3 = generation
or C_3 = color reading as having an unverified interpretive premise.

## References

- `../../predictions/theorem_B3_spinor_fermion_derivation.md` — B3, the colorless reading.
- `docs/theorem_B6_bridge.md` — B6, the color reading (Step 7 incorrect).
- `docs/theorem_B5_3_core.md` — the upstream (4, 4, 4) and (4, 2, 2)
  decomposition.
  posture; strengthened by the present finding.
  attempt that surfaced this conflict; closed as failed and now sharpened
  by the rejection of B6's color identification.
- `../../predictions/B_P_doubly_degenerate_h_derivation.md` — the upstream (4, 2, 2)
  Ramanujan content at k = P.
- Lawson, H.B. & Michelsohn, M.-L. (1989). *Spin Geometry.* Princeton
  Univ. Press. Ch. I §6 (Spin(6) ≅ SU(4)).
- Pati, J.C. & Salam, A. (1974). Lepton number as the fourth color.
  *Phys. Rev. D* 10, 275–289. §II (PS embedding SU(4) → SU(3)_color
  × U(1)_{B−L}).
- Slansky, R. (1981). Group theory for unified model building. *Phys.
  Rep.* 79, 1–128. §6 (SU(4) ⊃ SU(3) × U(1) branching tables).

## Files referenced

- `proofs/foundations/theorem_B3_B6_reconciliation.py` — explicit
  numerical/algebraic verification of all findings above.
- `proofs/foundations/theorem_B3_spinor_fermion.py` — B3 Brauer-Weyl
  realization (used unmodified).
- `proofs/foundations/theorem_B6_bridge.py` — B6 Spin(6) lift recipe
  (used unmodified).

## Verification

```
python proofs/foundations/theorem_B3_B6_reconciliation.py
```

Final line: `OK: theorem_B3_B6_reconciliation computation complete.`
