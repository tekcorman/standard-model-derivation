# Theorem — the interaction-selected axis is FORCED-rotated off the generation/winding axis

**Date:** 2026-07-15 · **Station:** GEN-IDENT-A (freeze
internal research notes) · **Grade:** theorem
(finite-group, exact, verified) for the group fact; adjudicated-open for the physical
interpretation.
**Receipts:** opening REFUTE internal research notes;
implementation pass internal research notes; independent verification
(concur, extended to full S₄ — see below); runnable
`proofs/foundations/genident_A_offset_check_2026-07-15.py` (27/27 PASS).

---

## Statement

Let `G_walk` be the construction's full walk-symmetry group: all permutations of the 12 darts
commuting with **both** the non-backtracking operator `hashimoto_gamma()` (B0) and `reversal()` (R),
the operators the selector functionals c₂/c₃/F1 are built from. Two independently-written
backtracking automorphism searches give **`G_walk = dart_rep(S₄)`, order 24** (strictly larger than
the A4-left image; the odd elements are genuine walk symmetries too).

Within `G_walk`, consider the three-fold structures on the one forced ρ₃:
- the **winding/generation axis** `v0` — the C₃ fixed by the deck screw `σ=(123)`, whose
  eigenspaces are the generation isotypes (GEN-HOMES);
- the **interaction/selector axis** `v2` — the C₃ fixed by `A4v[5]=W`, whose orbit is the July
  vertex-selected triad.

**Theorem.** There is **no** `g ∈ G_walk` that simultaneously
(a) preserves the winding grading (fixes the isotype subspaces / normalizes `⟨σ⟩`),
(b) preserves the selector constraint values (c₂=1/6, c₃=1/72, F1; F2/F3 invariant), and
(c) carries `v2 → v0`.

Equivalently `Sel_full ∩ {v2→v0 movers}_full = ∅`, where (computed at full S₄ scope by the sealed
check, using the honest extension `M(h)=S⁻¹ρ₃(h)S` with ρ₃ the S₄ *standard* representation):
- `Sel_full` = the selector-preserving subgroup = the Klein-four `{id, (23), (01), (01)(23)}` (order
  4; the A4-restricted `Sel={id,(01)(23)}` is properly contained in it);
- `{v2→v0 movers}_full` = 6 elements (the even `{4,8,10}` plus 3 odd 4-cycles);
- the intersection is **empty** — the 3 even movers and all 3 odd movers each shift c₂/c₃ by
  O(1e-2) (nowhere near the ~1e-16 floor), so none preserves the selector.

**Therefore the relative orientation between the interaction-selected three-fold axis and the
winding/generation three-fold axis is FORCED** — it cannot be removed by any symmetry of the
construction. It is gauge-invariant (the u-basis conjugation `S` drops out via `M(g)=S⁻¹ρ₃(g)S`).

## Why it is a genuine verdict, not an always-FORCED artifact (the mandatory control)

The test was shown to reach **both** verdicts (freeze T6, reproduced independently by the sealed
check): dropping the selector filter returns **GAUGE** (recovers plain A4/S₄ transitivity on the 4
axes); an order-3-vs-order-2 axis comparison returns **FORCED** (order is a conjugation invariant).
The machinery separates the two controls, so FORCED for the actual pair is real content. All
FORCED-content lives in the selector filter: with the filter dropped, an axis-mover aligns the
selector frame onto `σ` *exactly* (T5: zero residual, no σ/σ² split) — the axes are only unrelatable
*once the selector constraints are imposed*.

## Independent verification corrections (kept, per sealed protocol)

1. The implementation pass's "Sel cannot be evaluated beyond A4" is an artifact of its right-regular-
   translation construction (odd permutations fail to close, residual ~0.37), **not** a real
   obstruction: ρ₃ extends to the honest S₄ standard representation (group law ~1e-16;
   `det ρ₃(h)=sign(h)`), reproducing the A4 `M(h)` exactly on even elements. The verification ran the
   verdict at this full scope — the one place a flip to GAUGE could have hidden — and it did not flip.
2. The freeze's literal "`Stab ∩ Sel`" reading is a tautology (any g normalizing `⟨σ⟩` fixes `v0`,
   so it can never be a `v2→v0` mover). The well-posed non-vacuous reading is `Sel ∩ {movers}`, which
   is *logically stronger* (implies the literal one), so the operational substitution is
   conservative, not a smuggle. Adjudicated legitimate.

## What this DOES and DOES NOT establish (architect adjudication — honest bounds)

**Establishes (positive, forced):** the interaction sector (the vertex −κ·I(A;B) selector) genuinely
carries **non-gauge structural information** about the generation frame — a specific, forced relative
orientation between the interaction-selected axis and the generation/winding axis. This answers the
boot packet's framing question: the vertex coupling and the winding/generation grading are **NOT the
same object** (REFUTE, opening check) but they are **forced-related** (this theorem) — two distinct
three-fold structures of the one A4⊂S₄ locked at a rigid relative angle. That rigid link is the raw
material any identification would consume.

**Does NOT establish (stays open, by rule):** this does **not** supply the e/μ/τ labeling, does
**not** break the S₃ freedom, and does **not** derive the −70/−60.5 ppm subleading correction. No
mass/ppm/Koide/mixing value was read anywhere (goal-seek guard honored). The external identification
datum GEN-HOMES named remains external here.

**The next question (the real test of significance, NOT answered here):** does this forced offset
*reduce* GEN-HOMES's "consistent up to U(3)" residual on the observer factor (iii)? I.e., when the
observer couples to the substrate through the vertex, does the interaction's forced frame pin (part
of) the U(1)²/S₃ freedom that is the ppm-wall datum-slot? Framed as the GEN-IDENT-A FORCED routing
(freeze §5); not booked, not begun.

## Corollary (GEN-IDENT-B, 2026-07-15 — COUNTERFACTUAL, verified)

Because `⟨σ, W⟩ = A4` acts irreducibly on ρ₃, the joint commutant of `{ρ₃(σ), ρ₃(W)}` in U(3) is
**scalars only** (dim 1), versus the U(1)³ torus (dim 3) for `ρ₃(σ)` alone (Schur). Consequence: an
observer factor ℂ³_obs forced to respect BOTH forced structures loses its **entire continuous U(1)²
label-freedom** (GEN-HOMES 2-C's residual), and the surviving discrete freedom collapses from S₃
(order 6, the σ-alone baseline) to **`Out(A4)≅ℤ₂` (order 2, a single bit)** — the joint
normalizer-up-to-power keeps only the identity plus one outer order-2 element. This lands **exactly**
at the no-go theorem's single-datum floor.

**This is COUNTERFACTUAL.** GEN-IDENT-B established (source-level, verified) that no built
coupling ties the observer factor to the vertex −κ·I(A;B): the collapse's raw material exists but
nothing triggers it. The observer↔substrate coupling is therefore a NAMED INCOMPLETE EQUATION
(`docs/incomplete_equations_todo.md`, GEN-IDENT-B entry) and the next construction target — build it
and the label-freedom collapses to one bit automatically. Receipts:
internal research notes,
`proofs/foundations/genident_B_observer_residual_check_2026-07-15.py` (22/22).

## Regression anchor

`proofs/foundations/genident_A_offset_check_2026-07-15.py` (27/27 PASS) anchors: `G_walk=S₄`, the
Sel subgroups, the mover sets, the empty intersection, and both T6 controls. verify.py wiring queued
for the next L9 hygiene batch. `the_run.py`/Layer-1 untouched.
