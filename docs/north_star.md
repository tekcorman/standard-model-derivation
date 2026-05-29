# North Star — the finish-line goal

**Date:** 2026-05-21
**Status:** DURABLE. Not a dated snapshot, not a plan. This is the standing
definition of what "done" means for the program. Read it when scoping any
deep-layer work or deciding what to do next.

---

## The objective in one sentence

Replace the SM's particle-by-particle parameter list with **one derived,
complete set of commuting observables (CSCO)** — a single underlying object
from which every particle's full label set (mass, spin, charge, color, weak
reps, generation) follows, with no per-particle input.

## Why "particle-by-particle" is not the finish line

The program currently derives observables one at a time: y_τ by its chain,
V_cb by another, m_ν3 by its own. Even where each derivation is theorem-grade,
a *list of independent derivations is an enumeration, not a unified theory*. A
unified theory is **generative**: a short fixed structure produces the whole
spectrum, and adding particle N+1 costs zero new input.

## The three gaps between "a mass operator" and "a unified theory"

A universal mass operator (the T_mass hunt) fills *one slot* of the CSCO.
Three things separate that from the finish line.

**Gap 1 — the domain.** An operator weighs whatever you hand it; a theory must
derive *what there is to hand it* — the particle catalogue (3 generations, 48
states each, these reps and not others).
→ Status: **largely closed at ⚙️-structural grade** — fermion content,
generation count, charge quantization descend from Cl(6) / observer C³.

**Gap 2 — the joint algebra.** Mass is one CSCO operator; spin, charge, color
are siblings. Unification means the *whole commuting set, and the algebra that
makes it commute*, descend from one object.
→ Status: **largely closed at ⚙️-structural grade** — gauge group, spin,
charges all descend from one substrate. **Mass is the holdout slot.** This is
what T_mass is for.

**Gap 3 — generativity (the live frontier).** Even a perfect universal T_mass
is not unification if each species' mass still needs a *hand-matched*
selection — which Bloch point, which walker length. The **selection map**
(quantum-number content → what the operator evaluates) must itself be a
derived theorem.
→ Status: **met at THEOREM-GRADE-STRUCTURAL** (2026-05-21,
`theorems/theorem_selection_map_2026-05-21.md`). The selection map is a
*forced unique bijection* — 24 a-priori species↔walker-type assignments
collapse to 1 under the theorem-grade sub-theorems §4(B′)/§4(B)/§4(C); the
W55 over-determination is its acceptance test and passes. **All four entries
are derived** — the fourth (the d/u split, formerly mask #1 of the deep
frontier) by `theorems/theorem_updown_split_conjugate_higgs_2026-05-21.md`:
the up-type couples to the conjugate Higgs, which is even-grade and cannot
flip handedness ⇒ no walk ⇒ L=0; the down-type Higgs is odd-grade ⇒ flips ⇒
L=g. The selection map joins §8's over-determined family.

## The diagnostic: over-determination

Genuine unification = **one object, read many ways, forced to agree**, where
the agreement is a nontrivial constraint not put in by hand.

- **Passed, one sector.** The non-backtracking resolvent
  G_NB = (I − u·B_NB(srs))⁻¹ produces δ_r, δρ, V_cb, V_ub, V_us — five
  independently-known observables, zero fitted constants, forced consistent
  (`theorems/theorem_unified_oblique.md` §8). That sector is
  genuinely unified. **Expanded 2026-05-23/26:** the family now reads out 12
  observables (7 quark-sector + 4 lepton/PMNS + 1 cosmological A_s prefactor)
  from the same B_NB resolvent with `a = (2/3)⁸` and zero fitted constants —
  see `state_of_the_lepton_pmns_over_determination_2026-05-23.md`.
- **Not yet.** The mass sector has no such over-determination. Bringing it in
  is the entire point of T_mass.

## The finish line — four conditions

The program is done — a complete, derived CSCO — when:

1. **A universal mass operator exists** — T_mass, the substrate
   propagator-pole map. (In progress; Wall A of the
   `state_of_the_derivation` capstone.)
2. **The selection map is a derived theorem** — quantum-number content → what
   T_mass evaluates, one generator for every species. (Met at THEOREM-GRADE-
   STRUCTURAL, 2026-05-21 — `theorems/theorem_selection_map_2026-05-21.md`;
   forced bijection, all four entries derived — the d/u split by
   `theorem_updown_split_conjugate_higgs_2026-05-21.md`, closing mask #1.)
3. **The mass sector is over-determined** — the same substrate object that
   yields the oblique/CKM observables, read for masses, agrees without new
   input. (The real finish line.)
4. **MDL closure** — adding any particle costs zero new input; the whole CSCO
   follows from fixed structure.

**One-sentence test:** the day the mass sector is forced into the same
resolvent that already governs the oblique/CKM sector, and it agrees — the
program is done.

## How to use this document

When scoping deep-layer work (T_mass dynamics, the NA-4 non-associative
extension, the M2–M3 substrate-evolution formalism), ask of each candidate
step: *does it move conditions 1–4?* Specifically — does it turn the Bloch
selection rule from sketched into derived (Gap 3, condition 2), and does it
open a path to mass-sector over-determination (condition 3)?

Work that produces another isolated theorem-grade number, but does not move
the selection map toward generativity or toward over-determination, is
enumeration — useful, but not progress toward this north star.

## Active frontier (2026-05-21)

With the mass sector unified into `B_NB` (the 2026-05-21 mass-operator session
+ W55), the named next target is the **gauge-hub merge** — bringing the gauge
couplings (g_1, g_2, g_3, α_GUT, sin²θ_W) into the *same* `B_NB` resolvent.
Attack plan: an internal working note.

**Progress (2026-05-21, Stages 0/2/3/4/5 run — 5 probes, all on `main`):**
- The merge **mechanism is established** — `B_NB^U` (B_NB decorated with the
  gauge link variables) is *one* operator; its trivial-rep sector is the
  scalar `B_NB` (mass/oblique/flavor), its zeta factors over the irreps of
  the gauge group (non-abelian Artin–Ihara).
- The **physical α_GUT = 1/24.329** is read into it: the dark-correction
  factor `(1/k*)·V_cb` is a *verified* `B_NB` resolvent reading (the W55/§8
  family) — over-determination, condition 3, reaching the gauge sector.
- **The open core is RESOLVED — as a WALL (Stage 5,
  `gauge_hub_stage5_structure_group_forcing_2026-05-21.py`, 5/5).** The gauge
  structure group is *not* forced by the bare-count reading: `1/24 =
  dim(triv)/|G|` is group-blind (true for all 15 order-24 groups), so it
  forces only `|G| = 24 = N_local` — the existing input. The substrate's
  natural label group is `Z₂×A₄`, *not* `S₄ = Aut(K₄)` — `24 = 2^k*·k* =
  |S₄|` is a coincidence of two counts of non-isomorphic groups. The
  irrep-forcing route is the forbidden numerology (and positively refuted).
- **Honest standing:** conceptual unification + dark-factor over-determination
  reached; **input reduction (condition 4) NOT — and now known not to be
  reachable through the α_GUT bare factor.** The gauge group *itself* is
  forced, separately, by Cl(6) (`theorem_g2_edge_qubit_su2`). The
  gauge-hub merge is complete as far as it goes; the next input-reducing
  target is elsewhere (N_hub / Gap G1).
