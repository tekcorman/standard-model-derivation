# Framework note — the bridge: learning is the engine (2026-07-10)

**Status: NON-LOAD-BEARING framework note.** Nothing here moves a number, a lock, or a claim grade.
This is the durable harvest of a cross-repo synthesis (architect session, 2026-07-09/10) connecting this
repo to its documented grandparent, the the upstream engine compression engine (`~/projects/the upstream engine`). It is the
seed document for the origin-story reveal that `the upstream engine/RELEASE_STRATEGY.md` holds in reserve ("the
'these physics predictions came from a cognitive architecture research program' framing is
high-leverage and one-shot"). Papers remain paused; this is scoping, not a draft.
Operational counterpart: the CB/PB side-quest roadmaps
(internal research notes · `the upstream engine/core/PHYSICS_BRIDGE_SIDE_QUEST_2026_07_09.md`).

---

## 1. The two repos share an axiom set, not an analogy

the upstream engine's founding axioms are Toggle (binary state change on a graph node) and MDL (the system minimizes
description length as free energy). This repo's foundations rest on the same pair — explicitly:
`docs/theorems/theorem_A2_mdl_from_finite_register.md` grounds the finite-register observer in the upstream engine's
`research/consciousness/no_free_bits.md` §1.1, and the upstream engine's `research/consciousness/CROSS_REPO_CITATIONS.md`
maps the reverse flow. The lineage is circular and on the books in both directions. The "MDL waterline"
metaphor used below is the upstream engine's own (`no_free_bits.md:294`: "MDL is the waterline that separates *is a
thing* from *is ambient noise*").

## 2. The precise fact under the resonance: walk ≡ bitstring

A non-backtracking walk on a trivalent graph makes exactly one binary choice per step (k−1 = 2).
The walk IS a bitstring. Both programs discovered this independently, in opposite registers:

- **the upstream engine (engineering):** the walk-memory address φ(v,τ) = v·2^(g−1) + π(τ) with τ ∈ {±1}^(g−1) — the
  turn-history literally serialized as bits ("the bijection IS the address function",
  `the upstream engine/core/PHASE3_0_C_perfect_hash.md`).
- **This repo (ontology):** the arrow theorem — arrow of time ⟺ sub-criticality of the walk gas,
  u < u_c = 1/(k−1) = **2^(−b_edge)**
  (`proofs/foundations/M0_2R_T2_T3_arrow_criticality_currency_2026-07-07.py`). On the trivalent
  substrate the critical point is exactly **one bit per edge**.

Trivalence is the minimal valence at which a walk computes at all (k=2 has no choice), and the unique
minimal valence at which the walk's currency is exactly the bit. The 2-input NAND gate — the upstream engine's single
grounding primitive — is a 3-wire object; with fanout made explicit as 1-in-2-out COPY nodes, a
{NAND, COPY} netlist is a trivalent directed graph whose two node types are Landauer duals (NAND merges
information, COPY duplicates it). *That last construction is session synthesis, not a claim either repo
has yet made in code — flagged as such.*

Caution, stated plainly: trivalence alone is a weak invariant. The physics substrate's force comes from
trivalent + vertex-transitive + chiral + Sunada strong isotropy + MDL-selected-uniquely
(`proofs/foundations/dl_comparison.py`: srs at 13.02 bits, nearest 3D rival ths +0.83). A generic NAND
DAG is irregular, acyclic, and asymmetric. The gap between "NAND is 3-valent" and "the physics applies"
is exactly what the PB-5 experiment (below) measures.

## 3. The Landauer lock is a theorem about the upstream engine's axiom

β·κ = ln 2 is derived in this repo from precisely the statement "p = 2^(−L) and E = κ·L are one
quantity" — which IS the upstream engine's MDL-currency axiom. So this repo has already proven, without saying so:
*any system running MDL as its currency, if it admits a thermal description, carries a forced
bits↔energy exchange rate.* Same axiom, machine-checked consequence, sitting in the descendant repo.
The G5a corollary: a register with zero input sits in the Gibbs state ρ ∝ e^(−β·N̂) — with depth
measured in bits, that is p(structure) ∝ 2^(−DL): the currency axiom applied to memory retention.
(the upstream engine port: PB-3.)

## 4. The waterline, sharpened

The intuition "below the MDL waterline runs a perpetual-motion machine; above it, information costs
energy" is almost right, with one correction that makes it exact:

- **Below the waterline** nothing is learned, so nothing is paid. The idle engine is not perpetual
  motion — it is **break-even**, and β·κ = ln 2 is the break-even statement. The structure below the
  waterline is forced (zero free bits); its "dynamics" is thermal/modular flow of the tick state, and
  the constants are derivable precisely because they are the fixed point of compression with no data.
- **Above the waterline** every contingent bit costs kT ln 2 (Landauer proper). A measurement outcome
  reaches an observer only as a causal chain of physical records, each copy priced — that part is
  Szilard/Bennett orthodoxy, not mysticism. The genuinely novel claim of this program is that the
  price of that chain and the constants of the thing being measured are denominated in **one derived
  currency**.
- **The arrow:** sub-criticality means the run stays below one bit per edge — the register never
  saturates. Spare capacity is what an arrow *is*; a full register has no arrow. (Equilibrium = the
  full register = no more learning = no more time.)
- **Speculative, flagged:** the entanglement first law δS = δ⟨K⟩ (the unexploited graft named in
  internal research notes) reads as *the equation of learning above the
  waterline* — bit gained equals modular energy paid. the upstream engine's edge-admission gate
  (revenue > L(e) + ½log₂n + 1) is a discrete inequality of the same shape in the same currency.
  Nothing is built on this; it is a direction, booked here so it is not lost.

## 5. What transfers, and what never does

Structural theorems transfer: the criticality law, the Landauer lock, the Gibbs idle state, the zeta
machinery, the Sunada selection apparatus. **Physics magnitudes never transfer** — α₁, masses, mixings
are reads of THIS object's spectrum; a learner's crystal has its own spectrum and will not contain the
fine-structure constant of thought. Every cross-repo flow is now machine-gated: the bridge kit
(`bridge_kit/bridge_vectors.json`, verify entry #75) replaces prose-copying with a regression-checked
export, and the external-corroboration register
(`docs/audits/registers/external_corroboration_register.md`) freezes the interpretation of every
possible experimental outcome before any data exists.

## 6. The two instruments now pointed at this repo's foundations

The side quest converts the resonance into two falsifiable readings, both run on the the upstream engine side, both
with verdicts pre-booked here:

1. **PB-2 (criticality):** does a learner's pattern-admission process show the proliferation
   transition at u_c = 1/(k̄−1), with k̄ *measured* rather than forced? A hit is the first
   out-of-repo instance of the arrow law — corroboration of T2's genericity, the closest thing to an
   experiment this repo's foundational layer can get. A miss books as TENSION against the genericity
   clause (the srs-instance theorem is untouched either way).
2. **PB-5 (substrate selection):** does srs remain MDL-minimal when the crystallographic vocabulary
   (space groups, Wyckoff) is banned and the objective is re-derived from learning requirements only —
   with k priced, not fixed, so trivalence itself is allowed to lose? This is the only available answer
   to the frame-dependence objection against the derivation's first move, because every internal
   re-derivation shares the frame. The crystal-frame margin was +0.83 bits; the learning-frame margin
   is new information in every branch.

## 7. The metaphysical fork, left open on purpose

The mathematics is compatible with two readings and silent between them: the **realist** reading (the
structure is out there; any sufficiently good compressor rediscovers it — which is why physics is
universal and mathematical) and the **participatory** reading (Wheeler's it-from-bit: "matter" is what
stable registration looks like from inside, and the universe's substrate is the space of possible
registrations). Both predict the same spectrum; no experiment in either repo distinguishes them. The
correct posture is to hold the fork open explicitly — the derivation's ontology is
information-processing all the way down (register, tick, walk, bit; matter never appears below Layer-1
labels), and that fact is remarkable enough without resolving the fork by enthusiasm. What would move
the needle is PB-5: SRS-EMERGES would show the learner's own optimization finding the universe's
lattice, collapsing "the substrate of physics" and "the optimal memory" into one discovered object.

*Session provenance: claims verified against running code 2026-07-09/10 (both repos); the vetting
record and station detail live in the side-quest roadmaps and the CB-1 pre-registration.*
