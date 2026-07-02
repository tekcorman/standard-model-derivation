# A5(b) level-selection forced by local-vs-cumulative recurrence — theorem

**Date:** 2026-06-22.
**Status:** THEOREM (a connection of two already-derived pieces; no new derivation, no adoption). Grounds the
A5(b) level-*selection* — previously "structural but not A1-grounded" (parameter_uniqueness_ledger P3/P4) — in
the derived ∂_N recurrence law. Verified, no tuning.

---

## 1. Statement

The A5(b) **level-selection** (Level-2 single-density vs Level-3 walk-sum/tower) is **forced** by the derived
local-vs-cumulative recurrence law. Forced criterion:

> **A flavor transition's A5(b) level = the recurrence-type of the heaviest generation it touches.**
> - Touches **gen-3** (the heavy quark, which rides the **real Perron / cumulative** mode) ⇒ **Level-3
>   walk-sum** (geometric / multi-cycle series in (2/3)^L).
> - **Light-only** (both generations on the **complex shell / local** modes) ⇒ **Level-2 single density**
>   (k*²/(g·N_atoms), uniform, no geometric series).

This re-derives the §3.3 "how α₁ enters the formula" criterion of `theorem_A5b_level_prescription.md` from the
physics: *touching the cumulative/Perron generation is precisely what makes α₁ enter as a per-winding walk
factor (Level-3) rather than a closed-form coefficient (Level-2).*

## 2. The two derived pieces it connects (both pre-existing, verified)

1. **Local vs cumulative recurrence** — `derivation_topdown/time_bridge/explore_t10_integrated_history_heavy.py`
   (+ FINDINGS §t10; memory heavy-bosons-integrated-history): complex-winding shell modes (|h|²=k−1) give
   **local / single-slice** recurrence; real non-winding Perron / inter-copy modes give **cumulative /
   history-integrated (tower)** recurrence.
2. **The heaviest generation rides the real Perron (cumulative) mode** —
   `derivation_topdown/bridge/derive_generation_spectrum.py` (lines 59–79 mode level; 141–194 generation
   level): of the three C₃ windings, **ω⁰ carries the real Perron return |h|²=(k−1)²=4**; ω¹/ω² are the
   complex-conjugate pure-shell pair. At the generation level the heaviest generation is the **C₃-trivial
   channel** = the symmetric combination the Perron all-ones eigenvector occupies (a representation-theoretic
   identification: heaviest gen ↔ C₃-trivial ↔ Perron real mode — *not* "the heavy generation is literally
   winding ω⁰"; the spectrum is the C₃-Fourier of the windings).

## 3. Verification against the CKM assignments (no tuning)

Mass ordering: up (u<c<t = gen 1,2,3), down (d<s<b = gen 1,2,3); b = the heavy (gen-3) down quark.

| transition | gens touched | touches gen-3 (heavy/Perron)? | predicted | actual form (file) | match |
|---|---|---|---|---|---|
| **V_us** (u–s) | up-1, down-2 | no (light-only) | Level-2 density | k*²/(g·N_atoms)=9/40, uniform (`V_us.py`) | ✓ |
| **V_cb** (c–b) | up-2, down-3 | yes (b) | Level-3 walk-sum | (2/3)⁸/(1−(2/3)⁸)=256/6305, geom series (`V_cb.py`) | ✓ |
| **V_ub** (u–b) | up-1, down-3 | yes (b) | Level-3 walk-sum | Σ_{m≥2}(2/3)^(6m+2)/(1−·), multi-cycle (`V_ub.py`) | ✓ |

The other five CKM elements (V_ud, V_cs, V_td, V_ts, V_tb) are **unitarity-derived** from the four primitives
— consistent by inheritance, not independent tests. The lepton/PMNS angles are Case-A/Level-2 (consistent;
assigned by the orthogonal "how-α₁-enters" criterion, the touches-heavy rule not independently exercised there).

## 4. Consequence (the closure, and the residual it leaves)

**Closes:** the A5(b) level-*selection* is now forced by the derived recurrence law, not chosen. Combined with
the already-derived per-element winding-lengths (V_cb L=8 via n_fixed=2, CAS-verified; V_ub L=6m+2; V_us
Moore-bound) and the level-*forms* (Case A/B), **the entire flavor level-structure is forced GIVEN A5** (the
species/Hamming-weight labeling).

**Residual (now singular):** whether **A5 itself** — the species labeling (which Hamming weight is which
quark) — is forced by A1+MDL, or is the framework's one irreducible empirical anchor. This is the single
remaining input; everything in the flavor sector is downstream of it and derived.

## 5. Caveats (honest)

- This is a **connection / reframing** that holds on the three independently-assigned CKM elements; it is not
  the literal §3.3 text of `theorem_A5b_level_prescription.md` but explains *why* that criterion lands where
  it does. No element refutes it.
- **V_us labeling inconsistency to fix:** `predictions/V_us.py` calls it "Level 2 coupling density"; the
  A5(b) theorem table calls it "Case B (special) — counting form." Both agree on the *form* (uniform density
  9/40, no geometric series), which is what the synthesis predicts for a light-only transition. Naming nuance,
  not a numerical conflict — reconcile to "Level-2 / local uniform-density."

*Captured per directive 2026-06-22: results must land in the framework's docs, not session memory.*
