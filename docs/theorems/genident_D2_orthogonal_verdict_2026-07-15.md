# Verdict — GEN-IDENT-D2: the vertex does NOT put W on the canonical home (ORTHOGONAL-FORCED)

**Date:** 2026-07-15 · **Station:** GEN-IDENT-D2 (the mediation test — the last step of the
type-II₁ route). **Grade:** verdict, two legs — leg 1 a sealed theorem, leg 2 an adversarially
sealed negative. **Closes:** the GEN-IDENT type-II₁ route (D0→D1→D2). **NOTHING reads any
mass/ppm/Koide/mixing value** (goal-seek guard; AST self-scans in both drivers).

**The question (D2):** the canonical M₃(ℂ) observer home is rigid and forced (D0/D1) and carries σ.
Does the vertex `−κ·I(A;B)` reach across from the substrate to put **W** on it (⟹ by GEN-IDENT-B,
Schur collapse to one ℤ₂ bit — the deepest positive the route could reach)? Or is it blind to the
observer's σ-frame (⟹ labeling external, hand to Route β)?

# **VERDICT: D2 = ORTHOGONAL-FORCED. The vertex does not put W on the home; the generation labeling stays external via the type-II₁ route.**

---

## The two legs

**Leg 1 — the automorphism route (SEALED THEOREM).** W does not descend to an automorphism of
`M⋊_α ℤ₃`: `⟨σ⟩` is a self-normalizing Sylow-3 of A4 and `⟨σ,W⟩=A4`, so `WσW⁻¹∉⟨σ⟩` — W fails the
crossed-product descent criterion (`β` descends ⟺ `[βαβ⁻¹]∈⟨[α]⟩` in `Out(M)`). Transferred to
`M=L(F_inv(6))` via the self-contained Generalized-Outerness Lemma (any nontrivial generator-
permutation induces a properly-outer automorphism). By functoriality of the `M₃⊗M^α` split in
`(M,α)`, the canonical home therefore carries **no forced W-action**. Discriminates (only ⟨σ⟩'s 3
elements descend). Scoped honestly: denies the *forced* action, not the Skolem–Noether existence of
an unforced unitary `ρ₃(W)` on `B(ℂ³)`. **Receipt:**
`docs/theorems/genident_D2_leg1_W_no_descent_2026-07-15.md`; driver 32/32; sealed CONCUR-WITH-
CORRECTION (both applied).

**Leg 2 — the mediation route (ADVERSARIALLY SEALED NEGATIVE).** The only remaining route is a
vertex-*mediated*, non-automorphism coupling. It cannot be forced: an independent adversarial pass
(mandate = build a forced coupling) found every route collapses —
- honest-crossed-product: embedding the substrate into the atomless II₁ M is a free choice (∞ many
  inequivalent embeddings) → **arbitrary**, unless it collapses the M₃ leg onto F's level-1 = the
  **(D)-trap**;
- F_inv(6)↔walk bridge: the (3,3) cycle-type match is *forced* group theory, but supplies no
  ∗-embedding/state-map between `L²(M,τ)` and `H_hist⊗F` → **no bridge**;
- τ-induced GNS shadow: the only canonical state on the leg is maximally-mixed `I/3`, `U(3)`-blind
  → **vacuous** (bit-EVEN = democratic = blind);
- product state → `I(A;B)≡0` identically (verified; entangled control `2·log₂3` shows the functional
  discriminates, so the zero is genuine vacuity).

None is FORCED — the top-down verdict. **Receipt:**
internal research notes; anchor driver
`proofs/foundations/genident_D2_leg2_no_forced_coupling_check_2026-07-15.py` (12/12); sealed CONCUR.

---

## What this closes, and the honest bound

- **The GEN-IDENT type-II₁ route is complete and closed.** D0 (α properly outer) + D1 (canonical
  home rigid) built the deepest home the route could reach; D2 shows the vertex does not reach it.
  The canonical home EXISTS and is rigid, but the labeling is external because the vertex is blind to
  the observer's σ-frame — a sharp honest negative one level deeper than GEN-IDENT-C.
- **Parameter impact: NONE.** D2 does NOT label e/μ/τ and does NOT derive −70 ppm. The one-bit Schur
  collapse (GEN-IDENT-B) is NOT triggered via this route. The labeling stays a separate external
  datum; the magnitude (−70 ppm) stays a separate incomplete equation (top-down law).
- **The one reopener (booked honest caveat):** if a future station constructs a *forced* (not
  chosen) ∗-embedding `F ↪ M` — via a canonical projection fixed by the free product's own
  combinatorics, not an arbitrary corner — routes 1/2 of leg 2 reopen. No candidate found (II₁
  factors have no minimal projections, no canonical finite corner); judged exhausted, named for
  honesty.

## Hand-off: Route β

D2 = ORTHOGONAL is the licensed predicate for **Route β** — the dynamical pin of the run-endpoint
`s` via a run fixed-point (spontaneous breaking), ORTHOGONAL to this entire kinematic obstruction,
able to force the bit regardless. Nothing learned in D0/D1/D2 is wasted for β: the canonical home
(D0/D1) is a real, rigid object; D2 only shows the *kinematic vertex* does not select on it, which
is exactly why a *dynamical* selector (β) is the next thing to try.

---

## Regression anchors

- Leg 1: `proofs/foundations/genident_D2_half_descent_check_2026-07-15.py` (32/32).
- Leg 2: `proofs/foundations/genident_D2_leg2_no_forced_coupling_check_2026-07-15.py` (12/12).
Both pure/small, `OMP_NUM_THREADS=4`, no float physical constants (AST self-scan). `the_run.py`/
Layer-1 untouched; no `the_net.py` accretion (D2 is a structural negative, not a numeric read — no
forced observer–substrate read exists to accrete); not wired into `verify.py` (matches D0/D1/C/leg-1
precedent).
