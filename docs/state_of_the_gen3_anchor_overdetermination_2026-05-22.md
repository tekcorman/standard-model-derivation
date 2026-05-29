# State — the gen-3 anchor over-determination meets north-star condition 3 (narrowly)

> **⚠ UPDATE 2026-05-23 (next day) — the "narrow" framing is BROADER than
> stated here.** A systematic over-determination audit of the LEPTON /
> PMNS sector (`docs/state_of_the_lepton_pmns_over_determination_2026-05-23.md`)
> found four additional §8-family readings — y_τ, θ_12_PMNS, θ_13_PMNS,
> θ_23_PMNS — all using the same `a = (2/3)^8` (or V_us = 9/40 which is
> itself a §8 reading of `a`) with zero fitted constants. Combined with
> this doc's 7 quark-sector observables, the §8 over-determination spans
> **11 observables** read from the same `B_NB`, zero fitted constants.
>
> This doc's "narrow" framing remains accurate at the LITERAL gen-3
> anchor slot of the QUARK sector (which is what it claims). The
> condition-3 over-determination is **broader** than that — the 11-
> observable joint reading covers gen-3 anchors AND mixing angles
> across both sectors. Within-species generations (the within-species
> δ) remain δ-bound (5-way bounded-route elimination 2026-05-23).
>
> Read alongside the 2026-05-23 lepton-PMNS doc + `state_of_2026-05-23.md`
> §2.5 (updated to reflect the broader framing).

**Date:** 2026-05-22
**Status:** ASSESSMENT. Records a real (narrow) over-determination result —
the one that survives the negative side of the 2026-05-22 Yukawa-walker
session (see `session_handoff_yukawa_walker_route_2026-05-22.md`). Not a new
theorem; the components are theorem-grade upstream. The new content is the
*joint reading* — that they all anchor to the same substrate object,
parameter-free.

---

## 1. The claim

> **The gen-3 anchor slot of the mass sector meets north-star condition 3.**
> The same `B_NB` whose §8 reading gives the CKM amplitudes (V_us, V_cb,
> V_ub — zero fitted constants) is the one whose IB-root reading at trivial
> λ=+3, combined with the forced γ_7 species split (conjugate-Higgs
> theorem), gives y_t (h=1, Type II saturation) and y_b (h=2, Type IV
> Perron) — parameter-free, observed agreement at framework precision.

This is **narrow**: only the gen-3 anchor slot over-determines. The
within-species 3-generation structure (Koide rotations, mass ratios, CKM
mixings) remains δ-bound (Need-B δ-physical, deep frontier). But the
gen-3 anchor slot is a real landing — the first substrate-derived
species-distinct mass-sector reading anchored to the same object as the
oblique/CKM sector.

## 2. The two readings and the shared object

The north star's finish-line definition of condition 3: *"the same substrate
object that yields the oblique/CKM observables, read for masses, agrees
without new input."* The shared object is `B_NB` (the non-backtracking
Hashimoto of srs-z); two readings of it:

### Reading A — the CKM amplitudes (`theorem_unified_oblique.md` §8)

The resolvent `G_NB = (I − u·B_NB)⁻¹`, Bloch-integrated. Produces:

| observable | value | precision |
|---|---|---|
| `V_us` | 9/40 = 0.225 | matches PDG 0.2243 to ~0.4σ |
| `V_cb` | derived multi-cycle sum | within 1σ_PDG |
| `V_ub` | derived multi-cycle sum | within 1σ_PDG |
| `δ_r`, `δρ` | derived | matched |

Five independently-known observables, **zero fitted constants**. This is
the framework's existing condition-3 landing on the oblique/CKM side.

### Reading B — the gen-3 Yukawa anchors (§4(C) + §4(D) + conjugate-Higgs)

The Hashimoto at k=Γ on its trivial λ=+3 sub-block. Ihara–Bass:
`h² − 3h + 2 = 0` gives `h ∈ {1, 2}`. The species split between IB-roots
is forced (theorem-grade-structural):

- **Up-type** (n=2) couples to the conjugate Higgs `H̃ = iσ₂H*` (even-grade,
  cannot flip handedness) ⇒ L=0 ⇒ saturation root **h=1**.
- **Down-type** (n=1) couples to `H` (odd-grade, flips) ⇒ L=g=10 ⇒ Perron
  root **h=2**.

Walker-length formulas (`theorem_walker_length_MDL_waterline_2026-05-21.md`):

| anchor | type | formula | predicted | observed | dev |
|---|---|---|---|---|---|
| `y_t` | II (sat, L=0) | `h^0 = 1` | 1.000 | `m_t·√2/v = 0.992` | +0.82% |
| `y_b` | IV (Perron, L=g) | `(2/3)^10` | 0.01734 | `m_b/v = 0.01699` | +2.06% |

Zero fitted constants. Residuals are conditional on Family-D corrections +
M_unif threshold per the master synthesis §5.

### The shared object

Both readings are functions of the same `B_NB`. Reading A uses
`G_NB = (I − u·B_NB)⁻¹` (Bloch-integrated). Reading B uses `B(Γ)` (a
fiber). They are different functions of the same substrate operator. The
gen-3 anchor positioning that Reading B forces (via the IB-root split) is
*consistent with* Reading A's CKM amplitudes — Reading A includes V_cb,
which is the gen-2/gen-3 mixing and depends on the gen-3 anchor's
positioning. The §4(C) IB-root split forces that positioning; Reading A
doesn't contradict it.

This is the joint over-determination: **one `B_NB`, two parameter-free
readings, both anchored to observation, no new input between them.**

## 3. Verification

Direct numerical confirmation:
`proofs/foundations/yukawa_walker_stage_0_1_ibroot_eigenspaces_2026-05-22.py`
(7/7) — built `B(K_4)` at k=Γ explicitly, diagonalised, confirmed the
Ihara–Bass spectrum, confirmed h=1 and h=2 are present at trivial λ=+3,
and confirmed the IB-root eigenvectors live in the trivial Bloch sub-
sector (the h=2 eigenvector is exactly C₃-trivial isotypic; the h=1 IB
component is the unique head-aligned mode of its 3-dim eigenspace).

## 4. What this is NOT — the narrowness

The within-species 3-generation structure does **not** over-determine on
`B(Γ)`. From the Stage-1 finding:

- The IB-root eigenspaces at trivial λ=+3 are each 1-dimensional. There is
  no 3-generation basis at `B(Γ)|_{triv λ=+3}` — only the gen-3 anchor.
- The within-species mass ratios (gen-1, gen-2 vs gen-3) come from the
  Koide rotation `R^(s)`, parameterised by the Koide phase δ.
- The CKM mixings (V_us, V_cb, V_ub as functions of the within-species
  rotations) require both `δ_up` and `δ_down`, which differ — and δ is
  the deep-frontier Need-B δ-physical object.

So the over-determination claim **does not extend** to:
- 3-generation Koide ratios.
- CKM mixings as basis misalignment over generations.
- Lepton-sector PMNS over-determination (parallel structure, same δ
  problem).
- Absolute mass scales (still partly adopted per the master dark-correction
  doc).

The full mass-sector condition-3 landing requires δ-physical. This doc
records only the gen-3 anchor slot.

## 5. Honest grade

**ASSESSMENT (NARROW POSITIVE).** Not a new theorem; the components are all
theorem-grade-structural upstream:

- `theorem_unified_oblique.md` §8 — Reading A, the CKM-amplitude over-
  determination (theorem-grade).
- `theorem_walker_length_MDL_waterline_2026-05-21.md` — §4(D), the four
  walker types (theorem-grade-structural; Need-D-3 discharged via the
  selection map + conjugate-Higgs theorem).
- `theorem_color_triplet_Gamma_concentration_2026-05-21.md` — §4(C), the
  γ_7 IB-root split (theorem-grade-structural).
- `theorem_updown_split_conjugate_higgs_2026-05-21.md` — the species-split
  forcing (theorem-grade).
- `proofs/foundations/yukawa_walker_stage_0_1_ibroot_eigenspaces_2026-05-22.py`
  — numerical verification that the IB-roots exist as predicted (7/7).

The new content here is the **joint reading**: that all of these anchor to
the *same* `B_NB` and the gen-3 anchor positioning is over-determined
between Reading A's CKM amplitudes and Reading B's IB-root walker formulas.

## 6. What this means for the program

- **The framework has its first substrate-derived mass-sector condition-3
  landing**, restricted to the gen-3 anchor slot. Up vs down at gen-3 are
  forced apart by an IB-root split of the same Hashimoto that gives the
  CKM amplitudes, not posited via a parallel construction.
- **The CKM and within-species generation structure remain δ-bound.** The
  full mass-sector condition-3 landing — `B_NB` over-determines all 12
  fermion masses *and* the CKM + PMNS jointly, with no new input — is not
  available without δ-physical. That's the named multi-session research
  program (Need-B δ-physical) and has no bounded entry.
- **The right next bounded direction is elsewhere.** Per the morning's
  quark-sector handoff and this session's verdict: the gauge hub (with
  Route β closing the generation-symmetry question one way or the other),
  or eventually N_hub / Gap G1.

This is a real landing — narrow, defensible, recorded — and it does not
override the prior session handoff's verdict that route-by-route attacks
on the CKM / within-species generation structure terminate at δ-physical.

## 7. Cross-references

- `north_star.md` §"The finish line — four conditions" — condition 3
  definition and the over-determination criterion.
  — the session that established this assessment.
- `proofs/foundations/yukawa_walker_stage_0_1_ibroot_eigenspaces_2026-05-22.py`
  — numerical verification.
- `docs/theorems/theorem_unified_oblique.md` §8 — Reading A (CKM
  amplitudes).
- `docs/theorems/theorem_walker_length_MDL_waterline_2026-05-21.md` —
  Reading B (walker-length anchors).
- `docs/theorems/theorem_color_triplet_Gamma_concentration_2026-05-21.md`
  — the IB-root split mechanism.
- `docs/theorems/theorem_updown_split_conjugate_higgs_2026-05-21.md` —
  the species-split forcing.
- `docs/theorems/theorem_fermion_mass_operator_persistence_2026-05-21.md`
  — the broader M_persistence operator framing; this assessment is the
  condition-3 reading of its gen-3 anchor slot.
