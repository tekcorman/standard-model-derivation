# The gauge-hub merge — synthesis

**Date:** 2026-05-21
**Status:** SYNTHESIS-GRADE / COMPLETE-WITH-HONEST-WALL. Consolidates the
2026-05-21 gauge-hub session — the voltage→L-function dictionary (§2) and
Stages 0/2/3/4/5 (§§3–7), each a probe at 5–7/7. The merge *mechanism* is
established (§5); the merge is **conceptual unification + dark-factor
over-determination, and is *not* an input-reducing derivation** — Stage 5 (§7)
proved that the open core is a **WALL**: the gauge structure group is not
forced by the bare-count reading, and the route that could force it is
blocked. The wall is precisely characterized — that is the result.
**Scoping / attack plan:** an internal working note.
**Frontier:** `docs/north_star.md` — the gauge-hub merge.
**Probes:** `proofs/foundations/gauge_hub_stage{0,2,3,4,5}_*_2026-05-21.py`.

---

## 1. What the merge claims

The framework's predictive content is "one spectral object read many ways" —
the non-backtracking resolvent `G_NB = (I − u·B_NB(srs))⁻¹`. W55 and
`theorem_unified_oblique.md` §8 established that `B_NB` carries the **mass**,
**oblique**, and **flavor** sectors. The gauge couplings (g₁, g₂, g₃, α_GUT,
sin²θ_W) sat in a *structurally separate* hub — `α_GUT` from cycle-counting,
the 5-stage RG closure.

**The gauge-hub merge claims those are not two hubs.** The gauge bundle and
`B_NB` are the trivial-rep and non-trivial-rep sectors of one operator
`B_NB^U` — `B_NB` decorated with the gauge connection. This document records
how far that claim is established.

## 2. The dictionary — voltage graphs, covers, Artin–Ihara L-functions

The mathematical spine (Stark–Terras, *Zeta functions of finite graphs and
coverings*, Adv. Math. 1996/2000), and its framework translation.

**Voltage graph.** A base graph plus a group `G` plus a voltage assignment
`α: arcs → G` with `α(reverse) = α(arc)⁻¹`. The voltage modulo coboundary
(switching `α(e) → f(t)·α(e)·f(s)⁻¹`) is a class in `H¹(graph, G)`; distinct
classes give distinct covering graphs.

**Artin–Ihara factorization.** The Ihara zeta of the `G`-covering graph
factors over the irreducible representations of `G`:

```
   ζ_cover(u)⁻¹  =  ∏_ρ  L(u, ρ)^{dim ρ},
       L(u, ρ)⁻¹  =  det( I − u·B_NB^ρ + (k*−1)u²·I ),
```

where `B_NB^ρ` is the non-backtracking operator with each arc weighted by
`ρ(α(arc))`. The trivial rep gives `L(triv) = ζ_base`. This is "one `B_NB`
read once per irrep of `G`."

**Framework dictionary.**

| voltage-graph object | framework object |
|---|---|
| base graph | srs |
| `U(1)³` voltage (lattice translations) | the Bloch momentum `k`; the BZ `T³` is the abelian voltage moduli space |
| the `Z₂` (parity) voltage | srs-z, the chirality double cover — a Bloch half-period |
| `B_NB^ρ`, trivial rep | the scalar `B_NB` carrying mass/oblique/flavor (W55, §8) |
| `B_NB^U` (arcs decorated with link variables `U_e ∈ G`) | the gauge-covariant non-backtracking operator |
| the connection / link variables `U_e` | the gauge bundle of `srs_gauge_field_definition.py` |
| holonomy of `B_NB^U` round a cycle | the gauge-invariant Wilson loop |
| `L(ρ ≠ triv)` | the gauge sectors |

The reframe this dictionary makes precise: **the framework's Bloch machinery
already *is* abelian voltage-graph theory** — `k` is a `U(1)³` voltage. The
gauge-hub merge extends it to non-abelian `G`.

## 3. Stage 0 — the Z₂ Artin–Ihara factorization (verified machinery)

`gauge_hub_stage0_*` (7/7). srs-z's non-backtracking operator block-decomposes
under the `Z₂` deck involution into `+B_NB(srs)` (trivial rep) ⊕ `−B_NB(srs)`
(sign rep); `spec(B_cover) = spec(B_NB) ⊔ spec(−B_NB)` at Γ/P/H + 12 random
Bloch points; `ζ_cover⁻¹ = L(triv)⁻¹·L(sign)⁻¹`; the h-multiplicity 2→4 srs-z
fact is reproduced. The covering machinery is sound.

*Archive note.* The pre-existing C₃-twisted probe (`proofs/_archive/`) was
archived because its voltage was a **coboundary** (`Z_ω = Z_0` — a trivial
class), not because the machinery is broken. Voltages must be non-trivial
`H¹` classes.

## 4. Stage 2 — H¹(srs): the voltage space (covers vs. bundle)

`gauge_hub_stage2_*` (7/7). The abelian voltage room of srs's primitive cell
is **exactly the Bloch torus**: `H¹ = ℤ³`, all translational — girth 10
forbids internal loops, so the 3 fundamental cycles are the 3 lattice
generators. srs-z's `Z₂` is a Bloch half-period (a BZ corner), not separate
cohomology.

**The route split.** Every abelian voltage is a point of `T³`; a continuous
non-abelian group cannot be a covering deck group at all. So: covers handle
the **abelian/discrete** gauge content (U(1), the Z₂ chirality, plausibly a
Z₃ colour-centre); the **continuous non-abelian** factors SU(2), SU(3) are
**bundle structure** — the structure group of a bundle over srs, not a cover.
The framework already carries the ingredients as edge/vertex algebra: the edge
qubit `Cl(0,2) ≅ ℍ` is an SU(2) per edge (`theorem_g2_edge_qubit_su2`); Cl(6)
on the vertex Fock space is the full gauge content.

## 5. Stage 3 — B_NB^U: the merge mechanism

`gauge_hub_stage3_*` (6/6). The gauge bundle already exists
(`proofs/gauge/srs_gauge_field_definition.py`). The probe establishes that it
and `B_NB` are **one operator**: the gauge-covariant non-backtracking operator
`B_NB^U` (every Hashimoto arc decorated with `U_e ∈ G`) has

- trivial-rep sector = *exactly* the scalar `B_NB` (mass/oblique/flavor);
- zeta factoring over the irreps of `G` — the **non-abelian** Artin–Ihara
  L-function, verified for the non-abelian test group S₃:
  `det(B_reg) = det(triv)·det(sign)·det(std)²`;
- holonomy = a gauge-covariant Wilson loop.

So `B_NB` and the srs gauge bundle are the trivial- and non-trivial-rep
sectors of the one operator `B_NB^U`. This is the merge **mechanism**.

## 6. Stage 4 — the physical α_GUT as a B_NB^U reading

`gauge_hub_stage4_*` (6/6). The *physical* gauge coupling is α_GUT = 1/24.329
(the bare 1/24 is the pre-dark-correction counting value). It reads as a
two-factor `B_NB^U` quantity:

```
   α_GUT_phys  =  (1/24)  ·  (1 − (1/k*)·V_cb)   =  1/24.329
                  bare         dark
```

- **Dark factor — verified merge content.** The dark correction is
  `DC = (1/k*)·a/(1−a)` with `a = (2/3)⁸`; and `a/(1−a) = 256/6305 = V_cb`
  *exactly* (W55, §8). `δ_r = (1/12)·V_cb`; `δρ` and `V_cb` are the same
  resolvent object at other `c`. So the α_GUT dark correction is a c-scaled
  reading of the *one* `B_NB` resolvent — in the W55/§8 over-determined
  family, exact, zero free parameters. (`alpha_GUT.py`'s "Route H" derives
  this same DC Hashimoto-spectrally — independently a `B_NB` route.)
- **Bare factor — conceptual.** `1/24` is the trivial-rep fraction
  `dim(triv)/|G|` of an order-24 group, `24 = 2^k*·k* = |Aut(K₄)| = |S₄|` —
  the MDL weight of the gauge-singlet channel of `B_NB^U`.

## 7. Honest grade and the open core

**Reached.** The merge *mechanism* — `B_NB^U` is one operator, the gauge
bundle and `B_NB` are its rep sectors. And **over-determination**
(`north_star.md` condition 3) for the gauge sector: the physical α_GUT's dark
correction is a verified reading of the same `B_NB` resolvent as the masses
and mixings.

**Not reached.** **Input reduction** (`north_star.md` condition 4). α_GUT was
already a zero-free-parameter theorem; the merge unifies it (one object), it
removes no input — there was none to remove. The gauge sector's genuine
external inputs (M_Z, the MSSM particle interpretation in the RG running) are
untouched.

**The open core — RESOLVED 2026-05-21 (Stage 5): it is a WALL.**
`gauge_hub_stage5_structure_group_forcing_2026-05-21.py` (5/5) settled
whether the substrate **forces** the gauge structure group. The honest
verdict: the open-core question dissolves into three, and the input-reducing
one fails.

- **(A) Is the gauge group SU(3)×SU(2)×U(1) forced?** *Yes, already* — via
  Cl(6) and the edge qubit Cl(0,2) ≅ ℍ (`theorem_g2_edge_qubit_su2`,
  "forced, not an ansatz"). Never open.
- **(B) Is bare α_GUT = 1/24 forced?** *Yes, already* — `1/N_local`,
  `N_local = 2^k*·k* = 24` (the MDL uniform prior). Theorem-grade. Never open.
- **(C) Does the "1/24 = dim(triv)/|G| of an order-24 group" reframe force a
  new group / reduce an input?** ***No*** — this is the real result:
  - **Group-blind.** `dim(triv)/|G| = 1/24` for *all 15* order-24 groups
    (the trivial rep is always 1-dim). The reframe uses only `|G| = 24`,
    which *is* `N_local` — the existing input. It forces no group, reduces
    no input.
  - **The coincidence is real.** The substrate's natural group on the 24
    local labels is `(Z₂)³ ⋊ Z₃ = Z₂ × A₄` (3 edge qubits = (Z₂)³; the
    body-diagonal C₃ cycles them) — **not** `S₄ = Aut(K₄)`. They are
    non-isomorphic order-24 groups (center Z₂ vs. trivial; Sylow-2 (Z₂)³
    vs. D₄). So `24 = 2^k*·k* = |S₄|` is a genuine coincidence of two counts
    of *non-isomorphic* groups. `alpha_GUT_derivation.md`'s Reading B
    (24 = |Aut(K₄)|) is **retired** as a structural claim, by that doc's own
    stated criterion.
  - **The forcing route is blocked.** The only way to force a *specific*
    group is the non-trivial irreps "reading" g₂/g₃/sin²θ_W. But the irrep
    multiset is not substrate-fixed (S₄ {1,1,2,3,3} vs. Z₂×A₄ {1⁶,3,3} vs.
    abelian {1²⁴}); the substrate's own label group is Z₂×A₄; and a finite
    group's irreps are not a *continuous* gauge group's representations.
    Matching `{1,1,2,3,3}` to `{U(1),SU(2),SU(3)}` dims is the forbidden
    numerology — and it is here positively *refuted*, not merely disallowed.

**Net.** The merge is genuine *conceptual* unification (§5: `B_NB^U` is one
operator) plus genuine *over-determination* of the dark-correction factor
(§6: `DC = (1/k*)·V_cb`, a verified `B_NB` resolvent reading). It **cannot**
be made input-reducing through the bare `1/24` factor — there is no input
there to reduce, and no group is forced by it. Naming the wall precisely is
the result.

## 8. Cross-references

- `proofs/foundations/gauge_hub_stage0_z2_artin_ihara_2026-05-21.py` — Stage 0 (7/7).
- `proofs/foundations/gauge_hub_stage2_h1_voltage_space_2026-05-21.py` — Stage 2 (7/7).
- `proofs/foundations/gauge_hub_stage3_bnb_connection_2026-05-21.py` — Stage 3 (6/6).
- `proofs/foundations/gauge_hub_stage4_alpha_gut_reading_2026-05-21.py` — Stage 4 (6/6).
- `proofs/foundations/gauge_hub_stage5_structure_group_forcing_2026-05-21.py` — Stage 5 (5/5): the open-core verdict — a precisely characterized wall.
- `docs/theorems/theorem_unified_oblique.md` §3/§8 — `G_NB`; the oblique+flavor over-determination.
- `proofs/gauge/srs_gauge_field_definition.py` — the pre-existing gauge bundle (the connection of `B_NB^U`).
- `predictions/alpha_GUT.py` — the α_GUT derivation; flags the |S₄| "algebraic-equivalence open".
- `docs/theorems/theorem_g2_edge_qubit_su2.md` — the edge qubit Cl(0,2) ≅ ℍ = SU(2) per edge.
