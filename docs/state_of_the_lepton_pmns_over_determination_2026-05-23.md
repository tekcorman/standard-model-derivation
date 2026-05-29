# State — the lepton/PMNS sector meets north-star condition 3 via §8-family over-determination (2026-05-23)

> **EOD 2026-05-23 EXTENSION:** A_s (cosmological scalar perturbation amplitude)
> joins the §8 family as the **cosmology-sector counterpart** to this lepton
> landing, via `theorem_unified_oblique.md` §9 amendment (commit `7fa9c1c`).
> Combined: **12 observables across 4 sectors** — 11 here (quark 7 + lepton 4)
> plus A_s as the 12th (cosmology). See unified-oblique §9 + memory catalogue
> `reference_11_observable_section8_overdetermination_2026-05-23`.

> **DARK-SIDE COMPLEMENT (2026-05-24):** the 12-observable §8 family
> articulated here is the **gauge-side / observer-compressed-sector**
> complement of the dark-side `Ω_DM/Ω_m` landing. The same compression
> boundary that defines the §8-readable sector ALSO defines what dark
> matter IS — substrate-graph structure outside that compressed sector.
> See `theorem_dark_sector_multiaxial_waterfilling_candidate.md` §9-10
> (added 2026-05-24): the gauge-vs-gravity asymmetry framing unifies
> both landings under the compression boundary at k\* = 3, while the
> (β) audit (`dark_sector_kstar_nhub_unification_audit_2026-05-24.md`)
> documents the framework's deliberate two-primitive position (k\*
> independent of N_hub).

**Date:** 2026-05-23
**Status:** ASSESSMENT. Records a *real* over-determination result for the
LEPTON sector — parallel to and broader than the 2026-05-22 quark gen-3
anchor landing (`state_of_the_gen3_anchor_overdetermination_2026-05-22.md`).
Not a new theorem; every component is theorem-grade upstream. The new
content is the **joint reading**: that four lepton-side observables —
y_τ, θ_12_PMNS, θ_13_PMNS, θ_23_PMNS — are all §8-family readings of the
same `B_NB`, with zero fitted constants, at framework precision.

Companion audit: `proofs/foundations/lepton_pmns_over_determination_audit_2026-05-23.py`.

---

## 1. The claim

> **The lepton sector meets north-star condition 3.**
> Four lepton/PMNS observables (y_τ, θ_12_PMNS, θ_13_PMNS, θ_23_PMNS)
> are §8-family readings of the same `a = (2/3)^8 = α_1_bare` from the
> one Bloch-decorated Hashimoto `B_NB` — the same operator whose §8
> readings give the quark gen-3 anchor over-determination (y_t, y_b,
> V_us, V_cb, V_ub, δ_r, δρ). Combined across sectors: **11 observables,
> zero fitted constants.**

This is **broader** than the 2026-05-22 narrow framing: the gen-3 anchor
over-determination doc tagged only the QUARK sector (gen-3 Yukawas + CKM
amplitudes). The same one-B_NB machinery extends DIRECTLY to the lepton
sector via the four observables above, when audited systematically.

## 2. The four lepton-side §8-family readings

All four use the same `a = (2/3)^8 = α_1_bare = 256/6561` (the Feshbach
W1 amplitude on B_NB at P; theorem_unified_oblique.md §8) or its
resummed/dressed forms — with substrate-structural integers (k*=3, g=10,
chir 5/3, Klein h_P=(√3+i√5)/2) and zero fitted constants.

### 2.1 y_τ (gen-3 charged-lepton Yukawa)

```
   y_τ  =  α_1_full / k*²  =  (5/3)·(2/3)^8 / 9  =  1280/177147 ≈ 7.226 × 10⁻³
```

- α_1_full = (5/3)·a is the singlet-chirality-dressed Feshbach amplitude
  (Row P1 grade; chir 5/3 from §4(B')).
- k*² = 9 is the substrate counting projection (the same projection
  family as V_us = k*²/(g·N_atoms) = 9/40 in §8).
- Observed: m_τ/v = 7.217 × 10⁻³.
- **Match: +0.13%.** Theorem-grade (Row P74, `theorem_ytau_corollary.md`).

### 2.2 θ_12_PMNS (solar mixing)

```
   cos θ_12_PMNS  =  cos θ_TBM / cos θ_C
                  =  √(2/3) / √(1 − V_us²),    V_us = 9/40
```

- V_us = 9/40 is the §8 counting-projection reading of `a` (Row P4).
- θ_TBM = arctan(1/√2) is the tribimaximal Galois-symmetric angle.
- SU(4)_PS perpendicular-rotation theorem (Row P32, theorem-grade form).
- Predicted: 33.07°. Observed: 33.41° (PDG 2024).
- **Match: −0.45σ.** UNIQUE-THEOREM-GRADE.

### 2.3 θ_13_PMNS (reactor mixing)

```
   θ_13_PMNS = arcsin( V_us_bare · (1 − a) / √2 ),
   V_us_bare = V_us / (1 + (√5/4)·a)
```

- V_us = 9/40 (§8 reading of `a`).
- a = (2/3)^8 = α_1_bare (§8 fundamental amplitude).
- √5/4 = Im(h_P)/|h_P|² (Class-2 mass²-stripping coefficient; Klein h_P).
- (1 − a) is the resummation factor — the same `(I − u·B_NB)⁻¹`
  geometric structure as §8's `a/(1−a)` (V_cb).
- Pred ~8.7°. Observed 8.57° (PDG 2024).
- **Match: ~+0.3σ.** UNIQUE-THEOREM-GRADE-CONDITIONAL (Row P31).

### 2.4 θ_23_PMNS (atmospheric mixing)

```
   θ_23_PMNS = arctan( (1 + α_1_full) / (1 − α_1_full) )
```

- α_1_full = (5/3)·(2/3)^8 = (5/3)·a (§8-family Feshbach amplitude).
- σ_z = 0 theorem + dark-map Class 2 (Row P13, strict-solid theorem-grade).
- The arctan structure mirrors §8's `a/(1−a)` resummation in form (it is
  the dressed analog).
- Predicted: 48.72°. Observed: 49.2° (PDG 2024).
- **Match: −0.37σ.** STRICT-SOLID THEOREM-GRADE.

## 3. Combined over-determination tally

The complete §8-family over-determination across sectors (zero fitted
constants throughout):

| sector | observables | count |
|---|---|---|
| Quark gen-3 anchor | y_t, y_b | 2 |
| Quark CKM | V_us, V_cb, V_ub | 3 |
| Quark oblique | δ_r, δρ | 2 |
| **Lepton gen-3 anchor** | **y_τ** | **1** |
| **PMNS mixing angles** | **θ_12, θ_13, θ_23** | **3** |
| **TOTAL** | | **11** |

All 11 observables are readings of the same `a = (2/3)^8` (and its
resummations/dressings) under structural projections involving the
substrate's counting integers (k*=3, g=10, N_atoms=4), chir values
(5/3, 7), and the Klein h_P = (√3+i√5)/2. **Zero fitted constants
between them.**

## 4. What's NEW vs prior framing

The 2026-05-22 gen-3 anchor doc tagged the quark sector explicitly as
condition-3-met under §8 over-determination. The LEPTON sector was
implicitly in the same family — each prediction's individual derivation
chain (Row P74 for y_τ; Row P32 for θ_12; Row P31 for θ_13; Row P13 for
θ_23) traces to the same `a`, but the SHARED-OBJECT over-determination
framing was not made explicit across all four.

This doc makes it explicit:

- **Not a new theorem** — every component is theorem-grade upstream.
- **Not a new prediction** — the four observables already had predictions
  with theorem-grade derivations matching observation.
- **New: the joint reading.** The four lepton observables are not four
  independent structural coincidences with the same `a` — they are four
  readings of the *same one B_NB*, the same operator that yields the
  quark sector's 7 observables.

## 5. Honest caveats — what this does NOT close

| | status |
|---|---|
| **Within-species δ** (Koide phase for non-anchor generations) | Still δ-bound; 5-way bounded-route eliminated 2026-05-23 (R1, R2, R3, route 4, route 1). Need-B δ-physical genuine deep frontier. |
| **y_ν3** (Type I Bloch spectral) | Separate family (Laplacian band-edge), theorem-grade-conditional. Not §8. |
| **m_ν3 absolute** (global spectral gap) | Separate family (k*·N_atoms·M_Pl/√N_hub), theorem-grade-conditional. Not §8. |
| **δ_CP_PMNS = 180°** | Other-Smuggle geometric theorem (V₋₁-T_{B-L}). Different family. Matches observation (~7% off, within 1.2σ). |
| **δ_Koide = 2/9** | Algebraic identity at Q=2/3 (algebraic-identity-only — same gap as quark δ). |
| **α_31_PMNS** (Majorana phase) | TBD, separate. |
| **Light-lepton mass ratios** (m_e/m_μ, etc.) | Bound by Koide phase δ (the within-species 5-way-eliminated frontier). |
| **N_hub absolute** | Gap G1, named ~6-12mo new math. |

## 6. What this means for the program

**The framework's condition-3 over-determination is substantially
broader than previously documented.** With the 2026-05-22 quark gen-3
anchor + this lepton/PMNS audit, **11 distinct observables** are
read from the same `a = (2/3)^8` of the same one `B_NB`. The
condition-3 diagnostic — "one object, read many ways, forced to agree"
— has 11 forced agreements across the unified mass/flavor/oblique
sector.

Relative to the north-star (`north_star.md`) finish line:

| condition | status |
|---|---|
| 1. Universal mass operator exists | DERIVED (B_NB; theorem-grade) |
| 2. Selection map is a derived theorem | THEOREM-GRADE-STRUCTURAL (2026-05-21) |
| **3. Mass sector over-determined** | **MET FOR GEN-3 ANCHOR + MIXING** (11-observable family — quark gen-3, CKM, lepton gen-3, PMNS mixings, oblique). Within-species generations still δ-bound. |
| 4. MDL closure | Open (Gap G1 / N_hub + within-species δ + others) |

The narrow "gen-3 anchor only" framing of 2026-05-22 is now superseded
by the broader "gen-3 + mixings, both sectors" framing of today's audit
— still narrow in the sense of NOT closing within-species generations,
but substantially less narrow than tagged.

## 7. Probe + cross-references

- `proofs/foundations/lepton_pmns_over_determination_audit_2026-05-23.py`
  — this audit (catalogues and classifies, no number changes).
- `docs/state_of_the_gen3_anchor_overdetermination_2026-05-22.md` — the
  quark sector landing this generalizes.
- `docs/theorems/theorem_unified_oblique.md` §8 — the over-determination
  family (extended in 2026-05-21 to mass sector via y_t up-anchor).
- `docs/north_star.md` — condition 3 definition.
- Individual upstream theorems (all theorem-grade or theorem-grade-cond):
  - `theorem_ytau_corollary.md` (Row P74)
  - `theorem_theta12_PMNS_scoping.md` (Row P32)
  - `theorem_dark_correction_mdl.md` Class 2 + Class 3 (Row P31)
  - `theorem_dark_map_class2_closure.md` (Row P13)
- `docs/state_of_2026-05-23.md` — week-in-review (this finding extends
  §2.5's narrow framing).
