# Verdict — GEN-IDENT-β: the run-endpoint `s` is a ONE-BODY PHASE, invisible to the substrate vertex (BLIND-BY-THEOREM)

**Date:** 2026-07-15 (adjudicated 2026-07-16) · **Station:** GEN-IDENT-β (Route β, reshaped — the
dynamical/functional selector on the generation run). **Grade:** theorem-grade negative (BLIND-BY-THEOREM),
sealed-concurred. **Closes:** the DYNAMICAL route to pinning the generation run-endpoint `s` from within the
substrate. **NOTHING reads any mass/ppm/Koide/mixing value** (goal-seek guard; AST self-scan + grep, both
drivers/agents). Freeze: internal research notes.

**The question (β):** the generation mass-shape is forced to `m_j(u) = (1 + 2cos(u − 2πj/3))²`, `u = φ·s`,
with the run-endpoint `s` the single residual. Route β asked whether a FORCED dynamical/functional condition
on the run pins `s` spontaneously. Its literal forms (fixed-point, criticality, MDL/free-energy variational)
were already refuted in committed code (`explore_t12_observer_position.py`, `gap_fixes_s_scratch.py`,
`endpoint_search`). The one untested object: the substrate-internal **vertex functional `V(s) = −κ·I(A;B)(s)`**
— mutual information between the forced C₃-winding sectors of the endpoint-`s` run state on the forced
`Λ•(ℂ³)=(4,2,2)` carrier — swept over `s`. Does it have a distinguished, S₃-breaking, non-degenerate interior
stationary `s*`?

# **VERDICT: β = BLIND-BY-THEOREM. The vertex functional is EXACTLY CONSTANT in `s`, for a structural reason; no `s*` exists. The run-endpoint `s` is irreducibly the framework's one free Cauchy axis via the dynamical route too.**

---

## The mechanism (the theorem)

The forced amplitudes carry ALL of the endpoint dependence as a **pure per-mode phase**:
`c(s) = (1, e^{+iφs}, e^{−iφs})` — moduli are always constant (Ramanujan-shell rigid), only the winding
phases run. Therefore the endpoint-`s` state is a **local (single-mode) unitary orbit** of the `s=0` state:

> `|Ψ(s)⟩ = U(s)|Ψ(0)⟩`,  `U(s) = exp(iφs·N_{ω¹})·exp(−iφs·N_{ω²})`,

with generator `Σ_t ±φs·N_{ω^t}` a **sum of single-mode number operators**. A single-mode operator is local for
**every** mode bipartition of `Λ•(ℂ³)`. Bipartite entanglement entropy — hence mutual information
`I(A;B) = S_A + S_B − S_{AB} = 2S_A` on the pure state — is invariant under local unitaries. So `I(A;B)(s)`
is **forced constant**, for every promotion and every mode bipartition. Verified numerically flat to ≤1.6e-15
across 18 cells (3 promotions × 3 bipartitions × 2 modulus forks) and analytically (local-unitary residual
2.7e-16).

**This is NOT vacuity.** The entanglement is genuinely present (0.6–2.0 bits across cells) — the blindness is a
DISTINCT mechanism (**phase-locality**) from GEN-IDENT-D2-leg-2's product-state `I(A;B)≡0`. The vertex sees a
richly entangled state; it simply cannot see a one-body phase.

## What the verification killed (the load-bearing adversarial result)

The dangerous worry: is the blindness rigged by the forced Perron bipartition `A={ω⁰}` (s-independent) `|`
`B={ω¹,ω²}` (carries the phase)? The verifier flipped the **phase-CARRYING** mode onto side A
(`A={ω¹}`), by hand and numerically — entropy was **still exactly constant**. The reason is general:
entanglement entropy depends only on **Schmidt-value magnitudes**, which are insensitive to a local phase
**regardless of which party holds it**. The (4,2,2)-isotype grading and the fermion-parity grading are
**direct-sum** gradings (4+2+2=8, 4+4=8), not tensor factorizations — so neither is an eligible entanglement
cut, and no fourth FORCED bipartition exists. The (4,2,2) asymmetry (the freeze's hoped-for crack) plays no
role: blindness holds identically in the S₃-symmetric control. verification: independent Jordan–Wigner CAR
build from scratch, 18-cell table reproduced, driver 19/19 independently, **CONCUR**.

---

## What this closes, and the honest bound

- **The DYNAMICAL route to pinning `s` is closed.** Route β's literal forms (fixed-point/criticality/
  variational) were refuted in committed code; this station closes the one untested object (the vertex
  functional) with a MECHANISM: `s` is a one-body/gauge-like phase coordinate, and bipartite information is
  provably blind to it. Joined with the kinematic closure (GEN-IDENT-D0→D2), **both** routes to selecting the
  generation labeling from within the substrate are now exhausted — the kinematic vertex does not descend
  (D2), and the dynamical vertex functional cannot resolve `s` (β).
- **THE BOUNDARY THEOREM, earned.** This joins the four committed negatives — `explore_t12` (self-consistency
  fixed point + MDL waterline), `gap_fixes_s` (NJL gap), `endpoint_search` (kinematic scan), the
  `generation_splitting_no_go` (4 proofs) — into one sealed verdict: **the run-endpoint `s` is irreducibly the
  framework's one free Cauchy axis; the generation labeling is external.** And it upgrades the scattered
  "featureless" nulls into a REASON: `s` lives in the one-body sector that democratic bipartite functionals
  cannot reach — the sharp form of `δ = φ·s =` "the observer's run-position." Consistent with the Master
  Chirality Lens (bit-even/democratic = blind) and the L1 blindness wall.
- **Parameter impact: NONE on −70 ppm / e-μ-τ.** β does NOT label e/μ/τ and does NOT derive −70 ppm. The
  labeling stays external; the ppm MAGNITUDE stays a separate incomplete equation (top-down law).
- **The reopener (booked honest caveat, not a loophole):** a functional that is NOT bipartite mutual
  information — a genuine ONE-BODY/single-mode observable, or a functional of the absolute winding phase —
  COULD resolve `s`. But any such object is exactly an **observable of the absolute phase**, i.e. the single
  EXTERNAL observer datum the no-go theorem says the breaking requires. It would SUPPLY the datum, not derive
  it. So the reopener is the restatement of "external datum," not an escape from the boundary theorem. Pursuing
  it is a different object and a NEW freeze.

## Receipts / regression anchors

- Freeze: internal research notes.
- Implement return: internal research notes.
- verification: internal research notes (CONCUR).
- Accretion: `beta_endpoint_vertex_read` in `derivation_topdown/state/the_net.py` (reuses the V1 primitives).
- Driver: `proofs/foundations/genident_beta_endpoint_vertex_check_2026-07-15.py` (19/19; `OMP_NUM_THREADS=4`,
  AST self-scan clean; `the_run.py`/Layer-1 untouched; verify.py wiring queued, matches D-arc precedent).
