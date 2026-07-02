# Theorem (candidate) — Grade-blind mass classification: mass is non-trivial mirror holonomy

**Date:** 2026-06-03
**Status:** SYNTHESIS-GRADE / CLASSIFICATION-CANDIDATE. This document states *one*
mass-existence condition spanning both Clifford grades, and shows the massive
particle content is the **bounded solution set** of that condition over a
structurally-finite carrier — fermions *and* bosons under one criterion. It
**consolidates** existing theorem-grade pieces (the persistence operator, the
selection map, the gauge pillar) into a single grade-blind statement, and names
**four open lemmas** (two per quadrant, of matching shape) that gate full
theorem-grade closure. It changes no prediction value.
**Companion probe:** `proofs/foundations/grade_blind_mass_classification_2026-06-03.py` (14/14 gates PASS).

> **★ UPDATE 2026-06-04 (a) — bridge round, later found to be a mis-identification.**
> (`mirror_bridge_gamma7_hodge_2026-06-04.py`, 5/5.) Tested whether `γ₇` is the
> boson-mass mirror; found `γ₇` adjoint-central (gaps no gauge boson) and
> concluded the "single mirror" was refuted. **This conclusion was wrong** — see
> (b). The probe's positive content stands: `γ₇` is one operator with two faces,
> the **Hodge star** on forms and **chirality** on spinors. Its negative content
> (`γ₇` is central on gauge) is also true — but `γ₇` is the chirality *grading*,
> **not the Higgs**, so it was the wrong object to test.
>
> **★★ CORRECTION 2026-06-04 (b) — the single mirror IS the odd-grade Higgs; verdict
> REVERSED** (`mirror_single_object_higgs_2026-06-04.py`, 6/6). The framework's
> Higgs is a **grade-1 (odd)** element (`theorem_updown_split_conjugate_higgs`:
> "a Higgs-doublet component is a grade-1 odd element ... an odd-grade element
> flips handedness"). Odd grade does **both** jobs with **one object**: a grade-1
> `f₁` (i) **anticommutes with `γ₇`** ⇒ flips fermion L↔R ⇒ Yukawa mass, and (ii)
> is **non-central on the gauge bivectors** (anticommutes with the 5 sharing its
> index) ⇒ gaps gauge bosons ⇒ W/Z mass. `γ₇` (grade-6, even) is central on gauge
> precisely *because* it is even — that is the grading, not the mirror. So **the
> single mass-mirror is the odd-grade Higgs**, which the framework already builds;
> the "two different operators" reading of (a) was an artifact of testing the
> even-grade pseudoscalar instead of the odd-grade Higgs. §6's single-mirror
> unification is **restored**, with the mirror identified as the odd Higgs `f₁`
> and `γ₇` as the (distinct) chirality grading.
**Consolidates:**
`docs/theorems/theorem_fermion_mass_operator_persistence_2026-05-21.md`,
`docs/theorems/theorem_selection_map_2026-05-21.md`,
`docs/theorems/theorem_walker_length_MDL_waterline_2026-05-21.md`.

---

## 1. The move — from enumeration to classification

The mass sector is usually presented by *enumerating* species and walker types
and then forcing a bijection. This theorem instead states the **conditions under
which a mass-mode can exist** and lets the particle content fall out as the
**complete bounded solution set** of those conditions. The four fermion species,
the massless neutrino, the massive and massless gauge bosons, and the Higgs are
then *outputs* of solving one constrained problem, not entries of a table.

## 2. The mass-existence conditions

A substrate excitation is a vector in the `Cl(6)`-graded mode space over the
`srs ↔ srs-z` double cover. It carries **rest mass** iff:

- **(M1) Self-sustaining.** It reproduces itself as a closed non-backtracking
  walk (odd grade) or a propagating cochain (even grade) — it persists.
- **(M2) Standing.** It is a *stationary* (band-critical, zero group-velocity)
  mode, so it can be localized / have a rest frame. Over the compact Brillouin
  zone a non-flat band is Morse, so **its critical points are finite** — this is
  the boundedness generator.
- **(M3) Label-definite.** It carries definite internal quantum numbers
  (generation for fermions; gauge charge for bosons), which requires the little
  group to contain the substrate symmetry there.
- **(M4) Massive ⟺ non-trivial mirror holonomy.** The decisive, **grade-blind**
  condition: a mode has rest mass iff the `srs ↔ srs-z` mirror acts non-trivially
  on it. Trivial mirror action ⇒ it never has to respond, streams at `c`, and
  lies in the **kernel** (massless).

> **Mass = the cost of responding to the mirror.** A mode invisible to the mirror
> streams (massless); a mode the mirror moves must stand and respond (massive).
> The mirror is the single mass-giver — the substrate's symmetry-breaking order
> parameter (the Higgs vacuum) — for every grade.

## 3. The bounded carrier (no input)

Every factor is finite for a structural reason, before any reference to data:

- local algebra at a `k* = 3` vertex is `Cl(2·3) = Cl(6)` — spinor dim `2³ = 8`
  (odd quadrant), bivector+LR adjoint dim `21` (even quadrant). Bounded by `k*=3`.
- the odd quadrant's Bloch domain is the **finite** C3-fixed-and-critical point
  set `{Γ, H, P}` (§4); the even quadrant's breaking data is a **finite mirror
  chain** (§5).
- generation multiplicity is `C³` — exactly 3 by Gleason (minimal
  non-contextual dimension).

## 4. The odd quadrant — fermions (solve a quadratic over a finite domain)

A self-sustaining walk is a root `h` of the Ihara–Bass characteristic equation
`h² − E_k·h + (k*−1) = 0`. Solving the **same** quadratic over the finite
C3-fixed point set, the discriminant `E²−8` partitions into exactly the walker
classes (probe G1):

| Bloch point | `E_k` | discriminant | roots | sector |
|---|---|---|---|---|
| Γ (λ=+3) | 3 | +1 → real pair | `{2, 1}` | up/down split: `h=2` Perron (down), `h=1` saturation (up) |
| Γ/H (λ=−1) | −1 | −7 → complex, \|h\|²=2 | `(−1±i√7)/2` | neutrino sector (band edge) |
| P (λ=√3) | √3 | −5 → complex, \|h\|²=2 | `(√3±i√5)/2` | charged/chiral sector (Ramanujan saddle) |

The four walker types are the **root-classes**, not posited entries. The walk
length compresses to one integer whose sign selects the regime (probe G3):

```
   L = g·(is_down) − 2·(is_lepton)      g = girth = 10
   L > 0 → localized cycle  (d: L=10 Perron ; e: L=8 lepton cycle)
   L = 0 → saturation h=1   (u: y_t = Q^0 = 1)
   L < 0 → underflow → no localized cycle → spectral band edge (ν: L=−2 → L=∞)
```

**M4 gives the kernel as a non-existence result:** the trivial-holonomy modes
(`h^g = +1`) carry no oscillation ⇒ massless ⇒ exactly one massless light
neutrino `m_ν1 = 0` (the rank-2 seesaw count is theorem-grade, W45). The
fermion masslessness is "M4 has no solution in that cell," not a 12th input.

## 5. The even quadrant — gauge bosons + Higgs (the mirror response)

For even-grade modes the mirror action is computed directly: the gauge
mass-squared matrix is `M²_ab = ⟨T_a φ | T_b φ⟩` — *how much the mirror (the
symmetry-breaking direction `φ`) displaces each generator* (probe G4):

- eigenvalue `0` ⟺ generator **commutes with the mirror** (`T·φ = 0`) ⟺
  **harmonic / unbroken** ⟺ **massless** (photon = `T₃+Y`; gluons, being
  color and EW-singlet, untouched).
- eigenvalue `> 0` ⟺ generator **moved by the mirror** ⟺ **gapped** ⟺
  **massive** (W±, Z), with W± degenerate and below Z.

This is the exact even-grade analog of M4: *non-trivial mirror action ⇒ mass*.
The bounded count closes via a **mirror chain** (two stages, two scales; probe G5):

```
   21 gauge modes (PS adjoint 15+3+3)  +  1 Higgs scalar
   → 9 MASSLESS  = 8 gluon + 1 photon          (mirror-commuting / harmonic)
   → 12 MASSIVE  = 3 (W,Z) + 9 leptoquark      (mirror-gapped: EW + PS→SM stages)
   →  1 MASSIVE  Higgs (order-parameter amplitude mode)
```

## 6. The unified statement

> **The Standard-Model rest-mass spectrum is the non-trivial-mirror-holonomy
> sector of the `Cl(6)`-graded operator on `srs ↔ srs-z`.** The three **mirror
> kernels** are the massless particles — `ν₁` (odd), `γ` and `g` (even). The
> massive set is the **odd quadrant** (the fermions, classified by Ihara–Bass
> roots at the finite C3-fixed points `{Γ, H, P}`) ⊕ the **even quadrant**
> (W, Z, Higgs and the heavy leptoquarks, gapped by the mirror chain).

*(Per the 2026-06-04(b) correction: the single mirror is the **odd-grade Higgs
`f₁`**, which does both jobs — it anticommutes with `γ₇` (flips fermion chirality)
and is non-central on the gauge bivectors (gaps W/Z). The distinct operator `γ₇`
(grade-6, even) is the chirality **grading** — and, via the Clifford–form
correspondence, equals the Hodge star on forms — but is central on gauge, so it is
not itself the mass-giver. So: one mirror (odd `f₁`) gives both masses; one grading
(`γ₇` = Hodge-`*`) defines the L/R and form-duality the mirror acts against.)*

The two quadrants are the same theorem at different Clifford grade:

| | fermion (odd) | boson (even) |
|---|---|---|
| carrier (bounded) | `Cl(6)` spinor, dim 8 → 4 species × `C³` | `Cl(6)` bivectors+LR, dim 21 + 1 Higgs |
| domain | finite C3-fixed-and-critical points `{Γ,H,P}` | finite mirror chain (PS→SM, EW→em) |
| massive ⟺ | non-trivial girth holonomy `h^g ≠ 1` | mirror-gapped (`T·φ ≠ 0`) |
| massless (kernel) | `ν₁` | `γ`, `g` |

## 7. The four open lemmas (named, not hidden)

Closure to theorem-grade requires two lemmas per quadrant, of matching shape:

- **(L1, odd) Pin the domain.** Prove that a discrete, generation-definite mass
  mode concentrates only on the C3-fixed-and-critical set, and that this set is
  exactly `{Γ, H, P}`. *Status:* the C3-fixed **locus is a line** (probe G2), so
  finiteness is carried by the **criticality** condition (M2), not by symmetry
  alone; the reduction to three points additionally uses the mod-G* fixing of H.
  A **protected-degeneracy-at-one-fiber** premise is needed to exclude C3
  orbit-star superpositions (motivated by the three generations sharing gauge
  charges, but stated as a premise).
- **(L2, odd) One band computation.** Confirm the srs non-backtracking band has
  no accidental interior critical point on the Γ–P segment.
- **(L3, even) Pin the mirror.** Derive that the `srs-z` involution (the edge
  qubit, identified with the Higgs vacuum) is exactly the EW-breaking direction
  with the correct hypercharge embedding — the even-grade analog of L1.
- **(L4, even) One internal-cohomology computation.** Confirm the massless
  (mirror-commuting) gauge directions are the **internal** (Lie-algebra /
  Cl(6)-adjoint) harmonic part — the centralizer of the mirror chain in the PS
  adjoint — with count equal to the unbroken-group dimension (9).
  *(Update 2026-06-03, `proofs/foundations/mirror_internal_vs_spatial_2026-06-03.py`:
  the count is **internal, not spatial** — the spatial graph cohomology is
  `b₁(srs quotient = K4) = 3`, NOT 9, so "massless = spatial harmonic" is false.
  The 9 is the centralizer of the natural PS involution `J = diag(1,1,1,−1)`
  ("lepton = 4th color") in `su(4)`, `= su(3)⊕u(1)`. Separately, `γ₇` commutes
  with every gauge bivector, so the fermion-mass mirror and the gauge-mass mirror
  are **different operators** — that they are one duality is a motif, not yet a
  theorem.)*

## 8. Honest scope

- **Structure, not values.** This theorem classifies *which* modes are massive
  and *why* (grade-blind mirror holonomy), and bounds the count. It does **not**
  supply the mass *values*. The fermion anchors carry their existing residuals
  (selection rule + Family-D); the Weinberg angle and the W/Z mass ratio, and the
  Higgs mass, come from the separate `α`/RG and dark-correction tracks
  (`predictions/`), not from this classification. The toy EW computation in §5
  reproduces the *structure* (1 massless + 3 massive, Z above W) but uses
  `g = g′`, so it does not fix `θ_W`.
- **Candidate, not closed.** With L1–L4 open this is a classification *framing*
  with a verified mechanism and a bounded count, not an unconditional theorem.
  Its value is that it states the species/particle content as a solved existence
  problem rather than an enumeration, and exposes a falsifiable closure claim
  (§9).
- **No prediction value changed; no ledger row moved.** Consolidation only.

## 9. What it would predict if closed

Exhaustiveness becomes falsifiable: once M4 is grade-blind and the carrier is
bounded, every gapped substrate mode must be accounted for. The classification
then asserts the massive content is exactly `{SM fermions} ⊕ {W, Z, h, heavy
leptoquarks} ⊕ {dark sector} ⊕ {composites}` with **no exotic extra massive
particle** — a structural no-surprise claim. It also forbids a fourth generation
(Gleason fixes `C³`) and a fifth fermion species (the `Cl(6)` Fock dimension 8 is
exhausted).

## 10. Cross-references

- `docs/theorems/theorem_fermion_mass_operator_persistence_2026-05-21.md` — the
  odd-quadrant operator and the `m_ν1 = 0` kernel.
- `docs/theorems/theorem_selection_map_2026-05-21.md` — the forced species map
  (here re-read as the odd-quadrant solution set).
- `docs/theorems/theorem_walker_length_MDL_waterline_2026-05-21.md` — the four
  walker types (here re-read as Ihara–Bass root-classes).
- `predictions/M_persistence_derivation.md` — the assembled fermion operator.
- `proofs/foundations/grade_blind_mass_classification_2026-06-03.py` — the probe
  (14/14).
