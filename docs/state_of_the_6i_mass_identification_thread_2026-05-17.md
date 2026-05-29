# State of the §6(i) mass-identification thread — session close & fresh-start handoff (2026-05-17)

**Purpose.** One honest top-level statement of where the
`theorem_41_screw_wigner.md` §6(i) "mass ∝ 1/inverse-propagator"
postulate stands after this session, and the explicit fresh-start
entry point. Read this *after*
`docs/state_of_the_derivation_2026-05-16.md` (the convergence capstone,
still authoritative and **not reopened** by this thread). This doc is a
handoff, not a plan.

---

## 1. What this session did

Starting from the user's observation that the deep-frontier "five
masks" all share one core — the §6(i) physical postulate that *mass ∝
inverse propagator ∝ 1/survival* — and their proposal of three
operational views of mass (inertial / gravitational / energetic), the
§6(i) postulate was resolved **to the structural floor** in four
disciplined steps, each zero-fitted, each with pre-declared aborts and
an available honest-negative, none producing a number:

```
postulate
  →[decompose]  theorem_mass_propagator_overdetermination.md §§3,4
                = an over-determined identity (energetic≡inertial scale
                  ⟺ Ihara value u(k)=gradient u'(k)) forced UNIQUELY at
                  the independently-✅ k*=3.  Residual: one premise (b).
  →[b0]         §7  premise (b) = the standard RUELLE thermodynamic
                  dictionary for the Ihara=dynamical-zeta (k-generic,
                  Type-3 citable). Residual: 2 textbook physical IDs.
  →[b1]         §8  both IDs CLOSED — substrate loop-closure is
                  Bennett-reversible (A2-T = unique idempotent Csiszár
                  I-projection) ⇒ Landauer E=κ·S EQUALITY (saturation);
                  κ de-freed external→derived. b1 ⇒ b2 via k*=3.
                  Residual: one A2-T conditionality (quasi-staticity).
  →[b1']        §9  that A2-T conditionality DISCHARGED — time *is* the
                  forced Bayesian observation walk; quasi-staticity is a
                  derived consequence (per-step zero excess via the
                  prequential identity, two independent routes to
                  2.6e-13, + idempotence + monotone S_total clock), not
                  an assumption. Conditionality moves A2-T → A-IT.
```

**Result.** The §6(i) "mass ∝ 1/inverse-propagator" identification is
now a **structural theorem with no free external parameter and no
opaque dynamical assumption**, at grade THEOREM-GRADE-STRUCTURAL,
conditional only on the framework's **already-foundational axioms**
(✅ k\*=3 and the A-IT information axioms), not on any tunable and not
on the bespoke A2-T quasi-staticity assumption (that was discharged by
b1', §2).

Canonical record: `docs/theorems/theorem_mass_propagator_overdetermination.md`
(§§1–10). Probes (all exit 0, 5/5 pre-declared aborts):
- `proofs/foundations/mass_propagator_overdetermination_2026-05-17.py`
- `proofs/foundations/b0_ruelle_ihara_dynamical_zeta_2026-05-17.py`
- `proofs/foundations/b1_landauer_saturation_2026-05-17.py`
- `proofs/foundations/b1prime_observation_walk_dynamics_2026-05-17.py`

Commits (main, not pushed): `87f4aed` (decompose), `e11a456` (b0),
`a4e6fd7` (b1), `06e5172` (handoff), `30de5cf` (b1').

## 2. The one conditionality — discharged (b1')

b1 proved the *logical/information-theoretic* minimal-erasure condition
rigorously but left the *dynamical quasi-staticity* carried by the
**A2-T axiom**. **Route b1' (canonical doc §9) discharged it.** The
reframe (user-supplied): there is no *external* time against which the
substrate is fast or slow — **time *is* the closed-loop Bayesian
observation walk**, whose dynamics is *forced* (Cox/Csiszár uniqueness
= the A-IT axioms). Quasi-staticity is then a *derived consequence*:
per-step **zero excess** (the prequential identity = observer-energy
E4, two independent routes to 2.6×10⁻¹³, with a discriminating
inexact-update control showing +1.11 bit regret) + idempotence + the
strict-monotone `S_total` clock ⇒ "on the I-projection fixed point at
every tick" is derived, not assumed. The conditionality therefore
**moves A2-T → A-IT** (an already-foundational, load-bearing axiom set;
zero new assumptions). One instrument correction was made and recorded
honestly (a first mis-specified Pythagorean draft failed on subject
*and* control — the bad-instrument signature — and was rebuilt to the
standard prequential identity *without tuning to pass*; canonical doc
§9). No reader should mistake the result for unconditional: it rests on
A-IT + ✅ k\*=3 — but those are the framework's own foundations, not a
bespoke assumption.

## 3. What this explicitly does NOT do

- **No absolute number.** The mass scale is, and always was, the
  already-✅ v anchor. No ledger row moved; `parameters.csv` untouched.
  This closed a *structural identification*, not a numeric prediction.
- **Not a frontier closure.** Only the §6(i) **face** was touched. The
  other four masks — y_t up-anchor, Need-A2-unconditional, L6
  acoustic, δρ-subleading — are **untouched**. The monolithic deep
  frontier and the 2026-05-16 convergence capstone **stand,
  unreopened**.
- **Need-B unchanged.** Need-B's δ still inherits this §6(i) terminus;
  its *absolute* per-generation value still needs the deep
  per-generation dynamics. Need-B remains parked; only its terminus is
  now characterized to the structural floor (see that doc's addendum).

## 4. Fresh-start entry point (read this first next session)

1. **Do NOT re-open §6(i) as an opaque postulate, as "premise (b)", or
   as "the A2-T conditionality".** All are resolved/discharged. Cite
   `theorem_mass_propagator_overdetermination.md` §§3,7,8,9.
2. **Do NOT claim the deep frontier is closed.** It is not. The
   capstone is authoritative. This thread is one face only, and it
   produced **no number**.
3. **The absolute-scale residue (corrected — an earlier overclaim here
   is RETRACTED; see `docs/state_of_the_absolute_scale_2026-05-17.md`
   retraction banner).** Verified: the scale = (a) one unit
   *definition* (M_Pl=8/√π exact theorem; GeV = one declared
   conventional choice; zero fitted) + (b) the toggle dynamics tying
   time↔N are **DERIVED, theorem-grade** (`p_toggle=1/(k*N)`,
   `H=1/(N·t_P)` coeff 1, growth law — no adoption) + everything in
   natural units parameter-free **except the current value of one pure
   integer, N_hub**. **Retracted:** the claim that N_hub is "the age /
   'now' / provably not closeable / the epoch floor / not a research
   target / producing a substrate value = numerology". Corrected
   status (framework's own `N_hub_spectral_gap_attempt.py`): FORM
   derived/CLOSED, current epoch VALUE = **Gap G1, OPEN and BOUNDED**
   (needs a substrate epoch-selection / walk-origin self-consistency;
   the rate law is scale-invariant in N, Λ=1/N² is N-derived so no
   independent handle). A *principled* substrate self-consistency
   deriving N would be the legitimate closure (NOT numerology — only a
   *fitted* match is). The N-waterline shot
   (`proofs/foundations/n_waterline_epoch_selection_2026-05-17.py`)
   tests exactly this. The A2-T-removal task is **done** (b1'); the
   scale question is **reduced to one open bounded integer**, not
   answered-and-closed.
4. State, ledger, DAG, `parameters.csv` are all consistent and clean;
   nothing is mid-edit. Memory index

   carries the full four-step arc and the "how to apply" guard.

## 5. Disposition

The §6(i) mass-identification thread is **resolved to the structural
floor and honestly recorded**: postulate → decomposed → b0-reduced →
b1-closed → b1'-discharged, resting only on the framework's own
foundations (A-IT + ✅ k\*=3), no bespoke dynamical assumption, no
number. The press is **stopped here by decision**. The
absolute-scale residue is **reduced, not closed**: the scale = one
unit definition + *derived* toggle dynamics + one open bounded integer
N_hub (Gap G1). An earlier "answered / epoch floor / not a research
target" claim here is **RETRACTED**
(`docs/state_of_the_absolute_scale_2026-05-17.md` banner); Gap G1 is a
genuine sharply-posed open target (the N-waterline epoch-selection
self-consistency), neither closed nor proven impossible. The
repository is clean and ready for a fresh session.
