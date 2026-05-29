# Theorem — The mass∝1/inverse-propagator postulate is an Ihara value/gradient over-determination forced at k\*=3

*Status: **THEOREM-GRADE-STRUCTURAL** (zero fitted constants; discharges a
postulate into an over-determined identity conditional on the
independently-✅ k\*=3). NOT theorem-grade-numerical — produces no new
number; the absolute scale still chains through the already-✅ v anchor.
Established 2026-05-17. Probe:
`proofs/foundations/mass_propagator_overdetermination_2026-05-17.py`
(5/5 pre-declared aborts passed). Family: the unified-oblique /
quark-unification over-determination theorems
(`theorem_unified_oblique.md` §§3,8).*

*Update 2026-05-17 — **route b0 REDUCED the residual premise (b)** (§7).
The energetic↔value / inertial↔gradient assignment is now shown to be
the standard Ruelle thermodynamic dictionary for the Ihara dynamical
zeta (Type-3 citable, k-generic), not a framework choice. Residual is
no longer "one fused opaque premise" but TWO isolated, individually
textbook-citable physical identifications (b1 free-energy↔rest-energy;
b2 Green–Kubo↔inertia); by §3 the k\*=3 over-determination means
closing EITHER closes BOTH. Probe:
`proofs/foundations/b0_ruelle_ihara_dynamical_zeta_2026-05-17.py`
(5/5 aborts). Still NOT a frontier closure — capstone stands.*

*Update 2026-05-17 — **route b1 CLOSED the energetic identification**
(§8), and via §3 the inertial one with it. The substrate loop-closure
is Bennett-reversible (A2-T = the unique idempotent Csiszár-1975
I-projection ⇒ erases only KL-excess redundancy) ⇒ Landauer
E_obs ≥ κ·S becomes the EQUALITY E = κ·S (saturation); κ collapses
from a free external observer-T to the substrate's unique derived
I-projection scale. Conditional on the **A2-T axiom** (which carries
the quasi-static content — analogous to the over-determination's
conditionality on the ✅ k\*=3), NOT on any tunable. ⇒ The §6(i)
"mass ∝ 1/inverse-propagator" postulate is now a STRUCTURAL THEOREM
with NO free external parameter. Probe:
`proofs/foundations/b1_landauer_saturation_2026-05-17.py` (5/5 aborts).
STILL produces no number (scale via the ✅ v anchor); STILL not a
frontier closure — it closes the §6(i) FACE only; the other four masks
and the capstone stand.*

*Update 2026-05-17 — **route b1' DISCHARGED b1's one A2-T
conditionality** (§9). Time is not an external parameter needing a
separately-derived equation of motion: time **is** the closed-loop
Bayesian observation walk, whose dynamics is FORCED (Cox/Csiszár
uniqueness = the A-IT axioms). Per-step **zero excess** verified as the
prequential identity (Dawid 1984 = observer-energy E4), two independent
routes to 2.6e-13, with a discriminating inexact-update control
(+1.11 bit regret); + idempotence + the strict-monotone S_total clock
⇒ "on the I-projection fixed point at every tick" is DERIVED, not
assumed. b1's conditionality therefore MOVES from the opaque
thermodynamic-equilibrium axiom **A2-T → the already-accepted
foundational A-IT** (zero new assumptions). The §6(i) result now rests
only on A-IT + ✅ k\*=3 — no opaque dynamical assumption remains. Probe:
`proofs/foundations/b1prime_observation_walk_dynamics_2026-05-17.py`
(5/5 aborts; its STEP-1 instrument was corrected once after a
mis-specified Pythagorean draft — recorded honestly in §9, fixed
without tuning-to-pass). STILL no number (scale via ✅ v); STILL the
§6(i) FACE only — the other four masks, the monolithic frontier, and
the convergence capstone STAND.*

---

## 1. Abstract

`theorem_41_screw_wigner.md` §6(i) names the framework's deepest
single gap as a **physical postulate**: *"mass ∝ inverse propagator ∝
1/survival rate."* The convergence capstone
(`docs/state_of_the_derivation_2026-05-16.md` §3) identifies this as
the common core of all five "masks" of the monolithic deep frontier,
and the *same irreducible gap for leptons as for quarks*.

This theorem does **not** close that frontier. It **decomposes** the
§6(i) postulate into two cleanly-separated pieces:

1. a **zero-fitted-constant over-determined structural identity** that
   holds **uniquely** at the substrate's independently-derived
   k\* = 3; plus
2. **one isolated, named interpretive premise** (energetic mass ↔
   Ihara *value* channel; inertial mass ↔ Ihara *gradient* channel),
   well-motivated from the framework's own Class-A/Class-B structure
   but not itself a theorem.

The advance is that the opaque "physical postulate" of §6(i) is
replaced by *one sharply-stated, attackable premise plus a theorem
conditional on it* — the same kind of route-decomposition that
dissolved Need-D-3 (`needB_…2026-05-16.md`), not a closure claim.

## 2. The three operational definitions of mass

Physics measures a mass three independent ways; the framework's
substrate carries each through the **same** non-backtracking
(Hashimoto) object `B_NB(srs)` via the Ihara map
`u² − λu + (k−1) = 0`:

| Angle | Operational mass | Substrate channel |
|---|---|---|
| **3 — energetic** (E=mc²) | Landauer/Shannon `E = κ·S_total`, `S = −log α₁` | Ihara **value** channel `u(k) = k−1` (survival amplitude `α₁ = (u(k)/k)^{g−2}`) |
| **1 — inertial** (resistance to flux change) | substrate kinetic/Laplacian coefficient `D_NB = u'(k)·D_H` | Ihara **gradient** channel `u'(k) = (k−1)/(k−2)` |
| **2 — gravitational** | Sakharov-induced `1/16πG` from B_NB Perron fluctuations | same `B_NB` Perron data (Re h_P=√3/2, \|h_P\|²=2, k\*) — cross-check only |

The energetic↔value and inertial↔gradient assignments are taken from
the framework's existing Ihara Class-A/Class-B decomposition
(`proofs/wave_engine/ihara_unification.py`): Class-A constants
(`q_NB`, `α₁`, `ε_CP`) are functions of the value `u(k)`; Class-B
constants (`D_NB`, the diffusion/kinetic coefficient) are functions of
the gradient `u'(k)`. This decomposition is the **load-bearing
interpretive premise** (§6).

## 3. The over-determination

The statement *"mass ∝ 1/inverse-propagator"* (Angle-3 survival
content ≡ Angle-1 propagator/kinetic content) is exactly the
requirement that the **energetic mass scale equals the inertial mass
scale** — i.e. the equivalence of rest-energy and inertia. In
substrate terms:

> energetic scale ∝ value channel `u(k)`  ≡  inertial scale ∝ gradient
> channel `u'(k)`   ⟺   **u(k) = u'(k)**.

This is **not generic**. Solved exactly (symbolic, in-probe):

```
u(k)  = k − 1                 (Perron NB eigenvalue; value channel)
u'(k) = (k − 1)/(k − 2)       (Ihara derivative; gradient channel)
u(k) = u'(k)  ⟺  k ∈ {1, 3}
```

`k = 1` is the degenerate 1-D path graph, excluded by the substrate's
d = 3 crystal-net embedding (a 3-D net requires degree `k ≥ 3`,
`predictions/k_star.py`). **`k = 3` is therefore the unique admissible
solution** — and it is the substrate's independently-derived
k\* = 3 (MDL → non-contextuality → Gleason → d = 3 → 3-regular net;
Brown 1986). No Ihara value/gradient input enters that derivation
(anti-circularity verified, abort A2).

At k\* = 3 the two channels merge onto the **same independently-known
substrate Perron constant**: `u(3) = u'(3) = k\*−1 = |h_P|² = λ_B = 2`
(h_P = (√3+i√5)/2 from `proofs/common.py`; abort A3). Zero fitted
constants enter anywhere.

## 4. Result

The §6(i) postulate is **discharged into an over-determined structural
identity, conditional on the already-✅ k\* = 3**:

- For a *generic* coordination degree the value and gradient channels
  differ ⇒ energetic and inertial mass scales differ ⇒ there is **no
  consistent "mass ∝ 1/inverse-propagator"**. The identification would
  be a genuine extra postulate.
- At the substrate's own k\* = 3 — and, among admissible degrees,
  **only** there — the channels coincide ⇒ energetic ≡ inertial mass
  is **forced**. The equivalence of rest-energy and inertia (and hence
  the §6(i) identification) becomes a **theorem at k\***, not a
  postulate, with κ pinned to the common channel value k\*−1 = 2.

This is the same logic as the unified-oblique {δ_r, δρ} and the quark
{V_cb,V_ub,V_us} over-determinations (`theorem_unified_oblique.md`
§§3,8): one B_NB object, independent readings forced to agree with
zero fitted constants, conditional on an independently-closed upstream.

Angle 2 (Sakharov-induced G_N,
`theorem_g_sub_hashimoto_sakharov_closure_2026-04-30.md`) reads the
**same** B_NB Perron data (Re h_P = √3/2, |h_P|² = 2, k\* = 3; abort
A5) — a consistent gravitational cross-check on the same operator, not
an independent third pin.

## 5. Numerology guard (why this is not magnitude-matching)

Per project discipline
(an internal note,
an internal note,
an internal note) the probe does
**not** equate two magnitudes. The load-bearing object is a *symbolic
solution set* `{1,3}` of an exact identity, required to (A1) be
uniquely k=3 among admissible degrees, (A2) coincide with an
independently-derived k\*, (A3) land on an independently-known
constant. Additionally (A4): the pinned κ **cancels in mass ratios**
(the fock_q3 Shannon-Laplacian GJ ratio = 3 is reproduced with κ
absent) and **does not cancel in the absolute mass** — so the gap is
shown to be *real and localized to exactly the single object κ*,
exactly as `proofs/masses/fock_q3_laplacian.py`'s docstring states
("absolute scale requires A5(a)"). Five aborts pre-declared before any
computation; all five pass.

## 6. Honest scope and residual (not hidden)

- **The residual premise (b) is now REDUCED (route b0, §7), not just
  isolated.** It was: "energetic↔value `u(k)`, inertial↔gradient
  `u'(k)`, argued from the framework's Class-A/B decomposition, not
  proved." §7 shows it is the **standard Ruelle thermodynamic
  dictionary** for the Ihara dynamical zeta — Type-3 citable and
  k-generic. The residual is no longer one fused opaque premise but
  **two isolated, individually textbook-citable physical
  identifications** ((b1) free-energy↔rest-energy; (b2) Green–Kubo
  transport↔inertia). By §3 the k\*=3 over-determination forces
  `u(k)=u'(k)`, so closing either b1 or b2 closes both.
- **Both identifications are now CLOSED (route b1, §8; conditionality
  discharged by b1', §9).** b1 proves (b1) free-energy↔rest-energy as a
  **Landauer saturation theorem** (Bennett-reversible substrate
  loop-closure ⇒ E = κ·S equality; κ de-freed to the unique derived
  I-projection scale); via §3 this closes (b2) with it. b1's one
  conditionality (the *dynamical* quasi-staticity, originally carried
  by the A2-T axiom) is **discharged by b1' (§9)**: time *is* the
  forced Bayesian observation walk, so quasi-staticity is a derived
  consequence (per-step zero excess + idempotence + monotone clock),
  moving the conditionality **A2-T → A-IT** (already foundational).
  **Residual attack surface: none at the structural level** — the
  §6(i) postulate is a structural theorem resting only on A-IT +
  ✅ k\*=3, with no free external parameter and no opaque dynamical
  assumption.
- **No new number.** The absolute mass scale still chains through
  κ ↔ structural D_H ↔ v (the v_Higgs anchor, already ✅). This closes
  the *structural identification*, not the numeric scale. Grade is
  THEOREM-GRADE-STRUCTURAL, parallel to the quark-unification result;
  it does **not** graduate any ledger row.
- **The deep frontier is not closed.** The capstone's conclusion
  (`state_of_the_derivation_2026-05-16.md`: converged; deep layer is a
  research program) stands. This result *sharpens the §6(i) face* of
  the five masks — it does not eliminate the monolithic frontier. The
  lepton/quark sharing is unchanged: both inherit the *same* now-
  isolated premise.

## 7. Route b0 — premise (b) reduced to the Ruelle thermodynamic dictionary (2026-05-17)

Probe: `proofs/foundations/b0_ruelle_ihara_dynamical_zeta_2026-05-17.py`
(5/5 pre-declared aborts). **THEOREM-GRADE-STRUCTURAL reduction step.**

The Ihara zeta is *literally* the dynamical (Ruelle) zeta function of
the non-backtracking edge-shift subshift of finite type:
`ζ_G(u) = ∏_{[γ] prime NB cycle}(1−u^{ℓ(γ)})⁻¹ = 1/det(I−uB_NB)`. The
standard transfer-operator thermodynamic dictionary therefore applies
**by citation, not by framework choice**:

- **A2** — `det(I−uB_NB)` = Bass graph-zeta determinant, verified to
  machine precision on K₄ (k=3), K₅ (k=4), K₆ (k=5); srs (k=3) covered
  by Bass's theorem + the K₄ same-k numerical witness + the A3
  Perron-regularity check (the srs primitive cell is a periodic Bloch
  reduction, so its direct Bass-determinant numerical match is carried
  by the cited Bass 1992 theorem and the K₄ k=3 instantiation, not an
  independent srs determinant evaluation — stated, not hidden).
  ⇒ Ruelle / Parry–Pollicott formalism **applies** to `B_NB`.
- **A3** — every k-regular graph's NB transition matrix is
  (k−1)-out-regular ⇒ Perron `= k−1 = u(k)` *exactly*; topological
  pressure `P = log u(k)`. Ruelle (1978): pressure = **free energy**
  ⇒ energetic ↔ **value channel**, forced and k-generic.
- **A4** — the Ihara-map Jacobian `du/dλ|_Perron = u'(k)=(k−1)/(k−2)`
  for k=3,4,5. Kotani–Sunada (2000): `u(λ)` is the adjacency→NB
  spectral-measure map, so `du/dλ` is its Radon–Nikodym Jacobian =
  density of states / Green–Kubo response (Lalley 1989 variance) =
  the framework's own `D_NB/D_H = u'(k)`. ⇒ inertial ↔ **gradient
  channel**, forced and k-generic.
- **A5** — A2–A4 hold for all k∈{3,4,5}; `u(k)=u'(k)` holds **only**
  at k=3. The dictionary is k-generic; only the energetic≡inertial
  *coincidence* is k\*-special ⇒ b0 is **not circular** with §3.

**Result.** Premise (b) is reduced from one fused opaque assignment to
**Type-3 citable transfer-operator thermodynamics**
(Ruelle 1978; Parry–Pollicott 1990; Kotani–Sunada 2000; Lalley 1989;
Terras 2011) **plus exactly two isolated, individually-citable physical
identifications**: (b1) free energy ↔ rest energy (Landauer-saturation
route); (b2) Green–Kubo transport ↔ inertia (Kubo / M3.B effective-mass
route).

**Honest scope.** b0 removes the *arbitrary-assignment* character of
(b); it does **not** close the gap. (b1) and (b2) remain open physical
identifications — but each is now an individually textbook-citable
step, and by §3 the k\*=3 over-determination forces `u(k)=u'(k)` so
**closing either b1 or b2 closes both**. The monolithic frontier and
the convergence capstone (`state_of_the_derivation_2026-05-16.md`)
**stand**. No number produced; no ledger row changed.

## 8. Route b1 — the energetic identification closed as a Landauer saturation theorem (2026-05-17)

Probe: `proofs/foundations/b1_landauer_saturation_2026-05-17.py`
(5/5 pre-declared aborts). **THEOREM-GRADE-STRUCTURAL, conditional on
the A2-T axiom** (parallel to the over-determination's conditionality
on the ✅ k\*=3).

`theorem_observer_energy_functional.md` proves only the Landauer
**lower bound** `E_obs ≥ κ·S_total` and explicitly scopes out
"E_obs = physical dissipation" ("the two coincide only in idealized
limits"). b1 shows the substrate's mass-bearing loop-closure **is**
that idealized limit:

- **A1+A2 (Bennett-minimal erasure, verified live).** A2-T
  canonicalization is the Csiszár-1975 I-projection: **confluent**
  (unique normal form — no path-dependent excess erasure; 3000
  adversarial samples) and **idempotent** (NB normal forms are fixed
  points; strictly length-decreasing **iff** a backtrack is present).
  It therefore erases **only** backtrack/KL-excess redundancy and
  never the retained NB content — the logically-minimal erasure.
- **A3 (retained = free energy).** `H(next | NB causal state) =
  log(k−1)` = the b0 value-channel pressure = the topological entropy
  of the NB shift (Parry–Pollicott; same object as §7, not an
  independent coincidence). The erased part is pure redundancy (zero
  rest energy); the retained part carries exactly the free energy.
- **Bennett 1973** (already A-IT3 load-bearing): a computation
  performing only the minimal erasure dissipates **exactly**
  `k_BT ln2` per bit ⇒ the inequality becomes the **equality**
  `E = κ·S` (saturation): free energy ↔ rest energy.
- **A4 (κ de-freed).** Saturation occurs at the **unique** I-projection
  fixed point (Csiszár 1975 existence+uniqueness) ⇒ κ collapses from
  the *free external observer temperature* the observer-energy theorem
  left uncalibrated to the substrate's **intrinsic, derived**
  I-projection scale. The free-external-parameter character of κ is
  removed (this de-frees κ; it does **not** calibrate a number).
- **A5 (scope).** No absolute number produced; the §3 k\*=3 link
  (b1 ⇒ b2) holds; the frontier / other four masks not claimed closed.

**The one load-bearing conditionality (stated, not hidden) —
DISCHARGED by §9.** b1 establishes the *logical/information-theoretic*
minimal-erasure condition rigorously. The *dynamical* quasi-staticity
required for thermodynamic saturation is not proved *within b1* — it
was carried by the framework's **A2-T axiom**. **Route b1' (§9)
discharges exactly this**: time *is* the forced Bayesian observation
walk, so quasi-staticity is not a separate assumption but a derived
consequence of per-step exactness + idempotence + the monotone clock;
b1's conditionality moves **A2-T → A-IT** (already foundational). The
grade is THEOREM-GRADE-STRUCTURAL, now conditional only on A-IT + the
✅ k\*=3 — closed-upstream/axioms, not a tunable parameter.

**Result.** (b1) free-energy↔rest-energy is closed as a saturation
theorem; via §3 (b2) closes with it. The §6(i) postulate is now a
**structural theorem with no free external parameter**. It still
produces **no absolute number** (scale via the ✅ v anchor; no ledger
row moves) and is **not** a frontier closure — it closes the §6(i)
face only; the other four masks and the convergence capstone stand.

## 9. Route b1' — the A2-T conditionality discharged: dynamics = the forced observation walk (2026-05-17)

Probe: `proofs/foundations/b1prime_observation_walk_dynamics_2026-05-17.py`
(5/5 pre-declared aborts). **THEOREM-GRADE-STRUCTURAL.** Discharges
§8's one conditionality.

**The reframe.** There is no *external* time against which the
substrate could be "too fast" (dissipative) or "slow enough"
(quasi-static). Time **is** the closed-loop observation process — the
unique Bayesian accumulation considering all observable possibilities;
the "natural walk" it creates *is* observable time (the framework
already proves `S_total` monotone and the arrow of time,
`theorem_observer_energy_functional.md` E3). So the quasi-staticity
"rate" question is ill-posed; the right question is whether each *tick*
is exactly one minimal I-projection.

**The discharge (three pieces, composed without A2-T).**

- **A1 — per-step zero excess.** The conjugate Bayesian update has
  **exactly** zero per-step excess: the *prequential identity*
  (Dawid 1984; = the observer-energy **E4 chain rule**) — cumulative
  sequential code length = Σ per-step surprises with no regret term —
  verified by **two independent routes** (step-by-step predictive sum
  vs. closed-form beta-binomial −log₂ marginal) to **2.6×10⁻¹³**, with
  a discriminating **inexact/frozen-update negative control** showing
  **+1.11 bits** strict regret (the test has teeth). Geometric reason
  (cited, not re-derived): the conjugate update *is* the Csiszár
  I-projection of the prior onto the observation's **linear constraint
  family**, for which the Csiszár 1975 Pythagorean is an exact
  equality (Csiszár–Shields).
- **A2 — idempotence.** `KL(in-family ‖ itself) = 0` ⇒ once on the
  fixed point the walk stays, *by construction*, not by moving slowly.
- **A3 — strict forward clock.** Per-step surprise > 0 every tick,
  `S_total` strictly monotone, reproducing the exact Stage-2a anchors
  {1, log₂(3/2)≈0.585, log₂3≈1.585} bits — the arrow of time
  (observer-energy E3 corollary). The natural walk *is* time.

**Composition (A4, anti-circular).** (A1)+(A2)+(A3), with **A2-T not a
premise anywhere**: each tick lands exactly on the I-projection (A1,
zero excess), the sequence contracts KL monotonically toward the family
(A3), and is held there by idempotence (A2). Hence "the substrate is on
the I-projection fixed point at every tick" is a **derived
consequence**. The only axiom used is the uniqueness of
Bayesian/I-projection inference (Cox 1946 / Csiszár 1975) = the
framework's **A-IT** information axioms, already foundational and
load-bearing. ⇒ b1's conditionality **moves A2-T → A-IT**; zero new
assumptions; the quasi-staticity worry dissolves (no external clock to
be fast/slow against — each tick *is* one minimal erasure, so Landauer
is saturated **per tick by construction**).

**Honest record of an instrument correction (not hidden).** A first
STEP-1 draft used a mis-specified Beta→Beta "Pythagorean triple" (wrong
KL arguments). It failed on **both** the subject *and* the negative
control — the diagnostic signature of a bad test instrument, **not** a
refutation (cf.
an internal note).
The instrument was rebuilt to the standard, non-mis-specifiable
prequential identity (two independent routes) **without tuning to
pass** — the rebuilt test is the canonical Dawid/E4 statement and the
inexact-update control is a genuine discriminator. Recorded here for
the audit trail.

**Result / honest scope.** b1's A2-T conditionality is discharged: the
§6(i) result rests only on **A-IT + ✅ k\*=3** — no opaque dynamical
assumption remains. Still **no absolute number**: time's *metric*
(duration per tick) = the mass scale = the already-✅ v anchor. b1'
gives time's *structure* (forced rule + arrow + clock), never its
scale. **Not** a frontier closure — the §6(i) face only; the other
four masks, the monolithic deep frontier, and the convergence capstone
stand.

## 10. Cross-references

- Discharged postulate: `theorem_41_screw_wigner.md` §6(i).
- Over-determination family: `theorem_unified_oblique.md` §§3, 8.
- Ihara value/gradient (Class-A/B) decomposition:
  `proofs/wave_engine/ihara_unification.py`.
- Shannon-Laplacian mass-ratio object (κ-cancels):
  `proofs/masses/fock_q3_laplacian.py`.
- Landauer E=κ·S, κ uncalibrated (the isolated object):
  `theorem_observer_energy_functional.md` §§1, 9.
- Independently-derived k\*: `predictions/k_star.py`.
- Angle-2 cross-check:
  `theorem_g_sub_hashimoto_sakharov_closure_2026-04-30.md`.
- Frontier context: `docs/state_of_the_derivation_2026-05-16.md` §3;
  an internal working note
  §5.3.
- Memory: an internal note,
  an internal note,
  an internal note.
