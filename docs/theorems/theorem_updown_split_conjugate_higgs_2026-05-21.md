# Theorem — the up/down walker-length split: why the top is heavy and the bottom is light

**Date:** 2026-05-21
**Status:** THEOREM-GRADE-STRUCTURAL. Closes the selection map's last entry —
the colour-triplet d/u assignment — which was `state_of_the_derivation_2026-05-16.md`
§3 **mask #1** ("y_t up-anchor — σ₊-nilpotent, no route"). The derivation is
standard Clifford algebra + a machine-verified computation (2000 random cases)
+ one framework-structural input (the oscillatory srs↔srs-z Yukawa walk). No
pin, no transport law, no convention.

**Probe:** `proofs/foundations/needD3_mask1_proof_2026-05-21.py` (6/6 gates).

**Purpose.** The selection-map theorem (`theorem_selection_map_2026-05-21.md`)
forced the species→walker-type bijection down to one residual: which colour-
triplet species (d or u) walks the full girth (L=g, Type IV) and which has no
walk (L=0, Type II). This was mask #1, the framework's deepest named residue.
This theorem derives it.

---

## 1. Statement

**Theorem (up/down split).** Of the two colour-triplet fermions, the down-type
quark has walker length `L = g` (Type IV, Perron) and the up-type quark has
`L = 0` (Type II, saturation). Equivalently `y_b = q_NB^g` (suppressed → light)
and `y_t = q_NB^0 = 1` (un-suppressed → heavy).

**Mechanism.** The Yukawa walk oscillates between the substrate's two sheets
(srs / srs-z); every step is a *handedness flip*; the Higgs mediates each step.
A species can run the walk only if its Higgs can flip handedness. The down-type
fermion couples to the Higgs `H`; the up-type to the **conjugate Higgs**
`H̃ = iσ₂H*` (forced by hypercharge). `H` flips handedness; `H̃` cannot —
therefore the down-type walk runs the full girth and the up-type walk cannot
start.

---

## 2. The handedness operator is the edge-qubit volume element

Each substrate edge carries the edge qubit — the Clifford algebra `Cl(0,2) ≅ ℍ`
with two generators (`theorem_g2_edge_qubit_su2.md` §4).

**The handedness (chirality) operator of any Clifford algebra is its volume
element** — the product of all generators. This is the standard definition
(`γ⁵ = γ⁰γ¹γ²γ³` is the same statement for the Dirac algebra). For the edge
qubit, with two generators, the volume element `ω` is their product.

*Defining property, verified (probe P1):* `ω` **anticommutes** with every
generator (the grade-1 / "odd" elements) and **commutes** with every grade-0/2
("even") element. Hence: an odd-grade element flips handedness (maps the `+ω`
eigenspace to the `−ω` eigenspace); an even-grade element preserves it.

**Consistency check, verified (probe P2).** The framework's mirror — the
physical LH-srs ↔ RH-srs sheet swap — must, if it is genuinely a handedness
swap, flip the handedness operator. Computed: `mirror(ω) = −ω`. So `ω` is
confirmed as the handedness operator against the framework's own structure —
not merely assumed from the Clifford definition. (This resolves an earlier
confusion: the mirror is *implemented by conjugation by a generator*, but the
handedness *operator* is the volume element — distinct roles.)

---

## 3. The down-type Higgs is odd; the up-type conjugate Higgs is even

The framework's Higgs is the edge qubit; a Higgs-doublet component is a grade-1
("odd") element (`theorem_g2_edge_qubit_su2.md`; `h⁰↔f₁`, `h⁺↔f₂`). So:

- **down-type Higgs `H`** — couples to down-type fermions — is **odd-grade**.
- **up-type Higgs `H̃ = iσ₂H*`** — the conjugate Higgs, which up-type fermions
  couple to (forced by hypercharge — `H̃` carries the opposite U(1)_Y).

**Theorem (probe P3).** For *every* grade-1 Higgs `H`, the conjugate
`H̃ = iσ₂H*` is **purely even-grade**.

*Proof.* `iσ₂` is itself a generator of the edge qubit. `H*` (complex
conjugation) preserves grade, so `H*` is grade-1. Then `H̃ = iσ₂·H*` is a
product of two grade-1 generators, which is always even-grade (grade 0 ⊕ grade
2). *Verified on 2000 random grade-1 Higgs fields: the grade-1 component of
`H̃` is zero to machine precision in every case.* ∎

**This eliminates the only soft input.** The result holds for *any* `H` — so it
does **not** depend on the `h⁰↔f₁` VEV-direction identification (which W21
flagged as empirically pinned by y_τ). The down-type Higgs is odd and the
up-type Higgs is even, full stop.

---

## 4. Odd flips handedness, even cannot ⇒ the walk lengths

By §2: an operator flips handedness iff it anticommutes with `ω`.

- **down-type `H` (odd):** anticommutes with `ω` ⇒ **flips handedness** (P4).
- **up-type `H̃` (even):** commutes with `ω` ⇒ **cannot flip handedness** (P4).

**Framework-structural input.** The Yukawa walk is the oscillatory walk between
the substrate sheets srs and srs-z. srs-z is the bipartite double cover —
*every edge crosses between the two sheets* — so every step of the walk is a
sheet crossing = a handedness flip. The Higgs mediates each step. (This is the
framework's Yukawa walk: srs-z the bipartite double cover; the Yukawa the L↔R
flip dynamics — the volcano/mirror reading; W20–W22.)

Therefore (probe P5):

- **down-type:** the odd Higgs flips handedness ⇒ it mediates every step ⇒ the
  walk runs the full girth ⇒ **`L = g`** ⇒ `y_b = q_NB^g` — suppressed
  step-by-step ⇒ the down/bottom quark is **light**.
- **up-type:** the even Higgs cannot flip handedness ⇒ it cannot mediate even
  one step ⇒ the walk cannot start ⇒ **`L = 0`** ⇒ `y_t = q_NB^0 = 1` —
  un-suppressed ⇒ the up/top quark is **heavy**.

**The top quark is heavy because the conjugate Higgs it couples to is an
even-grade element that cannot flip handedness — so the suppressing
handedness-flip oscillation never gets going.**

The four species fall out: down-type {d, e} couple to `H` (odd) → genuine walk
(d: L=g; e: L=g−2, the lepton cycle); up-type {ν, u} couple to `H̃` (even) → no
walk (u: L=0 saturation; ν: L=∞ spectral band edge).

---

## 5. What this closes

- **mask #1 — closed.** `state_of_the_derivation_2026-05-16.md` §3 listed the
  "y_t up-anchor" as mask #1 of the deep frontier, characterised "σ₊-nilpotent,
  no route." There is now a route: the up-type's `L=0` is derived from the
  even-grade character of the conjugate Higgs. `theorem_unified_oblique.md` §8's
  "single genuine hard residue (σ₊ nilpotent ⇒ eigenvalue 0)" is this same
  result — the even-grade conjugate Higgs cannot flip handedness, so the walk
  has eigenvalue/length 0.
- **The selection map — fully closed.** `theorem_selection_map_2026-05-21.md`
  forced 3 of 4 entries theorem-grade and reduced the 4th (the d/u split) to
  mask #1. With mask #1 closed, the selection map is a forced, *fully derived*
  bijection.
- **`north_star.md` condition 2 — met.** The selection map is now a derived
  theorem with no per-particle input, at THEOREM-GRADE-STRUCTURAL — the working
  grade of conditions 1 and 3.

---

## 6. Honest scope

- **THEOREM-GRADE-STRUCTURAL.** The derivation is: standard Clifford algebra
  (§2), a machine-verified computation on 2000 random cases (§3), the
  hypercharge-forced up-type↔H̃ coupling (Standard Model), and **one
  framework-structural input** (§4): the Yukawa walk is the oscillatory
  srs↔srs-z walk, every step a handedness flip. That input is a structural
  fact about *what the walk is* — srs-z is the bipartite double cover, the
  Yukawa is the L↔R flip dynamics — the same walk the rest of the framework
  uses. It is not a pin or a convention.
- **Not theorem-grade-numerical.** This theorem derives the *walk lengths*
  (`L=g`, `L=0`), hence the structural forms `y_b = q_NB^g`, `y_t = 1`. The
  residual numerical precision (`y_b` +2.06%, `m_t` +0.82%) is the existing
  Family-D / M_unif-threshold conditional, unchanged.
- **No `predictions/*.py` changed; no ledger row moved.** This is a structural
  closure: it derives the selection-map entry that was open.

---

## 7. Cross-references

**Proof:** `proofs/foundations/needD3_mask1_proof_2026-05-21.py` (6/6).

**Closes / updates:**
- `state_of_the_derivation_2026-05-16.md` §3 — mask #1 (y_t up-anchor).
- `theorem_selection_map_2026-05-21.md` — the d/u entry (§6 residue).
- `theorem_walker_length_MDL_waterline_2026-05-21.md` (§4(D)) — the species
  mapping conditional.
- `theorem_color_triplet_Gamma_concentration_2026-05-21.md` (§4(C)) — the γ₇
  IB-root split, was "conditional on §4(D)".
- `theorem_unified_oblique.md` §8 — the "single genuine hard residue".
- `docs/north_star.md` — condition 2.

**Builds on:**
- `theorem_g2_edge_qubit_su2.md` — the edge qubit `Cl(0,2)`, the Higgs on it.
- `theorem_g2d_chirality_doubled.md`, W20–W22 — the mirror, the srs-z bipartite
  double cover, the broken Higgs vacuum.
- `theorem_selection_map_2026-05-21.md` — the bijection this completes.
