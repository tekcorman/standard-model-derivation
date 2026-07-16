# Binding energy functional — theorem

**Date:** 2026-07-04.
**Status:** **THEOREM-GRADE-STRUCTURAL-CONDITIONAL** — gate-passing under `../parameters/parameter_linter.md`. Every load-bearing step is Type 2 (explicit algebra, machine-verified on the srs object) or Type 4 (upstream closed theorem: `theorem_observer_energy_functional.md`, hereafter OEF; `../../proofs/foundations/H_multiway_construction.py`, the B_VD=0 no-go). The theorem forces the FORM of the composite binding energy and the fact that it introduces **no new constant and no new adoption**. It is conditional on the framework's standing identification *physical energy = κ · (description length)* (the OEF's own conditional; the OEF theorem §15 flags it as "an energy functional, not THE physical energy"), and it does **not** fix the absolute scale (see §8).
**Scope:** narrow. Proves that the binding energy of a composite (a compound closed walk on srs) is forced to be `E_bind = −κ·ΔS`, with κ the OEF's own Landauer constant and ΔS the MDL net description saving; that binding requires shared description (extensivity); and that this evades the B_VD=0 dynamical-coupling no-go.
**Out of scope (named, unchanged):** the ABSOLUTE value of κ (= the substrate energy quantum `e_bit` in physical units). That is the WALLED constituent-coupling / gauge-running keystone — the same wall that bounds the M_Z pole and every absolute binding MAGNITUDE. No magnitude is computed or claimed here.
**Verification:** `../../proofs/foundations/BOUND_F1_oef_two_subsystem_2026-07-04.py` (ALL PASS; pre-registered internal research notes §12, commit 5dd1654, before the probe). Program context: bound-state Stages 0–3b (existence, threshold, discrete ΔS spectrum) in the same scoping doc.

---

## 1. Theorem statement

**Theorem (Binding Energy Functional).** Let a *composite* be a compound closed walk on srs — a finite set of constituent girth cycles `{C_1, …, C_m}` with edge sets `{E_1, …, E_m}`. For an edge-set configuration `X`, let its description length under the framework's MDL/NB convention be

$$L(X) := \Big(\,|\!\bigcup_i E_i| \;+\; \sum_{v} \max(\deg_v - 2,\ 0)\,\Big)\cdot b_{\text{edge}}, \qquad b_{\text{edge}} = \log_2(k^\*-1) = 1\ \text{bit},$$

where the degrees are taken in the union edge set (a lone girth cycle is degree-2 everywhere, so `L(C_i) = ` girth). Let `κ := k_B T \ln 2` be the OEF's Landauer constant (`theorem_observer_energy_functional.md` §9), and let the OEF energy of any configuration be `E(X) = κ·L(X)` (OEF §10 evaluated on the observation stream that specifies `X`).

Define the **net description saving** (the MDL mutual description of the constituents)

$$\Delta S \;:=\; \sum_i L(C_i) \;-\; L\Big(\bigcup_i C_i\Big) \;=\; \sum_e (\text{mult}_e - 1) \;-\; \sum_v \max(\deg_v-2,0) \;\ge\; 0.$$

Then the composite's **binding energy** — the OEF energy of the joint description minus that of the independent descriptions — satisfies:

$$\boxed{\,E_{\text{bind}} \;:=\; E\Big(\bigcup_i C_i\Big) - \sum_i E(C_i) \;=\; -\,\kappa\,\Delta S\,.}$$

Moreover:

- **(Forcing / no new adoption)** The only dimensional constant in `E_bind` is the OEF's own `κ`; `ΔS` is a dimensionless integer. The two-subsystem functional is the single-stream OEF `E = κ·L` evaluated on two descriptions of the same edges — it introduces **no new constant and no new functional**.
- **(Binding ⇔ shared description)** `E_bind < 0` (bound) iff `ΔS > 0` iff the constituents share description length. **Disjoint** constituents give `E_bind = 0` (extensivity).
- **(No-go evasion)** `ΔS` is a function of edge-set combinatorics alone; it involves no Hamiltonian or matrix element, so it evades the B_VD=0 no-go that zeroes the canonical dynamical coupling.

---

## 2. Axioms and cited upstream

- **OEF** (`theorem_observer_energy_functional.md`, THEOREM) — Type 4. Provides: `E_obs = κ·S_total` with `E = κ·(description length)`; non-negativity, monotonicity, **extensivity (E8)**; `κ = k_B T \ln 2` (Landauer 1961 / Bennett 1973 via A-IT3). The OEF's standing conditional (E_obs is *an* energy functional, physical identification out of its scope, §15) is inherited here and named.
- **MDL / NB description convention** — Type 4 / Type 2. `b_edge = log₂(k*−1) = 1` bit is the same per-NB-step cost behind `α₁ = ((k−1)/k)^{g−2}`; the ΔS decomposition is the Stage-0 result (`bound_state_mdl_compression_probe_2026-05-28.py`), re-derived here as inclusion-exclusion.
- **B_VD = 0 no-go** (`../../proofs/foundations/H_multiway_construction.py`) — Type 4. The canonical visible↔dark dynamical coupling is identically zero; this theorem's binding lives in the orthogonal description-length channel.
- **Description-length currency** (internal research notes) — the framework ontology under which `E = κ·ΔL` (energy rep) and `p = 2^{−ΔL}` (amplitude rep) are one quantity; this theorem is that principle's composite-sector application, now at structural theorem grade.

No fabricated citations; no post-hoc fitting; no magnitude matched.

---

## 3. Setup

A single girth cycle `C_i` is described by specifying its edges; being degree-2 throughout, `L(C_i) = ` girth · b_edge (no junction cost). Two descriptions of a composite are available and both are legitimate OEF observation streams over the same edges:

1. **Independent** — describe each `C_i` separately. Shared edges are specified once *per constituent* (multiply). Cost `L_indep = Σ_i L(C_i)`.
2. **Joint** — describe the union as one object. Each distinct edge is specified once; each union vertex of degree `d > 2` costs `(d−2)` extra NB choices (the junction is no longer a forced degree-2 step). Cost `L_joint = L(∪_i C_i)`.

By source coding, the accumulated OEF surprise of the stream that specifies a configuration equals that configuration's description length; hence `E = κ·L` for each description, with the **same** `κ` (OEF §9–10). This is the only place the OEF enters, and it enters identically for both descriptions.

---

## 4. Step B1 — the two OEF energies (Type 4 + Type 2)

`E_indep = κ·L_indep` and `E_joint = κ·L_joint`, both by the OEF `E = κ·L` (§3). The constant is the OEF's `κ` in both. ∎

## 5. Step B2 — inclusion-exclusion (Type 2, verified)

$$L_{\text{indep}} - L_{\text{joint}} = \sum_i |E_i| - \Big(|\!\textstyle\bigcup_i E_i| + \sum_v\max(\deg_v-2,0)\Big) = \sum_e(\text{mult}_e-1) - \sum_v\max(\deg_v-2,0) = \Delta S.$$

The first equality is the definition of `L`; the second is `Σ_i|E_i| = Σ_e \text{mult}_e` and `|∪E_i| = Σ_e 1`. **Verified exactly** (error 0.0) on all 8,100 overlapping cycle pairs and all 277,020 connected cycle triples in the 3³ srs supercell (probe §S-1). ∎

## 6. Step B3 — the binding energy (Type 2)

$$E_{\text{bind}} := E_{\text{joint}} - E_{\text{indep}} = \kappa\,(L_{\text{joint}} - L_{\text{indep}}) = -\kappa\,\Delta S. \qquad \blacksquare$$

## 7. Step B4 — extensivity ⇒ binding requires sharing (Type 4 + Type 2, verified)

By OEF extensivity (E8), energy adds over independent descriptions. If the constituents are **disjoint** (`E_i ∩ E_j = ∅`, no vertex degree raised above 2 in the union), then `L_joint = Σ_i L(C_i) = L_indep`, so `ΔS = 0` and `E_bind = 0`. **Verified exactly** on 3,000 disjoint pairs and a disjoint triple (probe §S-2). Binding is therefore not put in by hand: it arises **iff** the constituents share description length. ∎

## 8. Step B5 — no new constant, no new adoption (Type 2)

The binding law `E_bind = −κ·ΔS` contains exactly one dimensional constant, the OEF's own `κ`; `ΔS ∈ ℤ_{≥0}` is combinatorial (§5). The two-subsystem case is not a new functional — it is `E = κ·L` (single-stream OEF) evaluated on two descriptions of one edge set. Therefore the composite/binding layer introduces **no new adoption** beyond the framework's standing `energy = κ·(description length)` identification. The scoping-doc §8 risk-2 worry ("a new two-subsystem mutual-information functional may require a named adoption") is resolved negatively. ∎

## 9. Step B6 — B_VD=0 evasion (Type 4 + Type 2)

Every quantity above (`|∪E_i|`, `mult_e`, `deg_v`, `ΔS`) is a function of the edge-set combinatorics alone; no Hamiltonian, propagator, or matrix element appears. Binding is a description-length quantity, orthogonal to the canonical dynamical coupling that the B_VD=0 no-go (`H_multiway_construction.py`) sets to zero. The description-length channel is exactly the one that no-go leaves open. ∎

---

## 10. Parameter_linter gate summary

| Step | Claim | Gate type | Source |
|---|---|---|---|
| B1 | `E = κ·L` for both descriptions, same κ | Type 4 + 2 | OEF §9–10 evaluated on the specifying stream (§4) |
| B2 | `L_indep − L_joint = ΔS` | Type 2 | inclusion-exclusion; **verified 8,100 pairs + 277,020 triples, err 0.0** (§5) |
| B3 | `E_bind = −κ·ΔS` | Type 2 | B1 + B2 (§6) |
| B4 | disjoint ⇒ `E_bind = 0` | Type 4 + 2 | OEF extensivity E8; **verified 3,000 pairs + triple** (§7) |
| B5 | no new constant / adoption | Type 2 | one constant κ; ΔS ∈ ℤ (§8) |
| B6 | B_VD=0 evaded | Type 4 + 2 | ΔS combinatorial; H_multiway no-go (§9) |

All steps gate-passing. One Type-4 upstream (OEF, itself gate-passing); one Type-4 no-go (H_multiway); the rest Type-2 algebra machine-verified on the object.

---

## 11. What this theorem closes

- **The load-bearing bolt of the composite sector.** Binding = compression is now a forced *energy* statement, not an assumption: `E_bind = −κ·ΔS`, with the OEF's own κ and no new input.
- **Zero new adoptions for the bound-state layer.** The worry that a two-subsystem functional would cost a named adoption (bound-state scoping §8 risk 2) is retired. The layer stands on the *same single* standing identification the OEF, the arrow of time, and the mass/coupling currency already carry.
- **Binding ⇔ shared description, on the object.** Extensivity makes independent subsystems additive; only shared description binds — verified, not posited.
- **The no-go is cleanly evaded.** Binding is description length, not the (vanishing) dynamical coupling.
- **Downstream unblocking (grade, not value):** F2-ii (the geometry→composite dictionary) and F3 (nucleon → Q_np, g_A → the BBN/Y_p gate) now provably rest on this same identification plus the walled scale — nothing more.

---

## 12. What this theorem does NOT close

- **The absolute scale.** The value of κ (= `e_bit` in physical units, eV/MeV) is **not** fixed here. It is the constituent-coupling / gauge-running keystone (walled; `scale_bridge_binding_energy_closure_2026-06-01.py`). **No binding-energy magnitude is derived** — the deuteron 2.2 MeV, hydrogen 13.6 eV, and every absolute value stay OPEN.
- **The physical-energy identification.** This theorem inherits, and does not itself prove, the OEF's standing conditional that `E_obs = κ·(description length)` *is* physical energy (OEF §15 declines to claim this). Full promotion to "physical binding energy" — and to `docs/framework/` per the currency doc's promotion path — is gated on that identification and on the `e_bit = t` absolute closure.
- **Composite mass and selection rules.** The mass of a composite (a compound-walk holonomy) and its species assignment are separate, unproven (the F2-ii dictionary; bound-state scoping §8 risk 5).

---

## 13. Honesty

`ΔS` is the *net* saving (shared edges specified once fewer, **minus** the junction NB-overhead), not the raw shared-edge count; the junction cost is a legitimate part of the joint description length, so `E_bind = −κ·ΔS` with this ΔS is exactly the OEF energy difference. The forcing is genuine — a priori the two-subsystem case could have required a new interaction constant (indeed the *dynamical* route is dead, B_VD=0); what this theorem shows is that the *description-length* route needs nothing new. But the result is **structural**: it fixes the functional form and the adoption count, not a number. The absolute scale — the only thing standing between this and a physical binding-energy prediction — is the walled keystone, and it stays open. An open miss stays open.

---

## 14. Downstream consequences

- **F2-ii (geometry→composite dictionary):** de-risked (Stage-3b skeleton) and now known to sit on the F1 identification; the anchoring (which config is which physical composite, via constituent species) is the remaining derivation.
- **F3 (nucleon → BBN gate):** its structural cost is now zero new adoptions; the magnitude route remains keystone-walled (Q_np's QED part explicitly, bound-state scoping §8 risk 4).
- **Currency doc promotion:** `description_length_currency_principle_2026-05-31.md` §3's "CANDIDATE" application reaches structural theorem grade with this file; the `e_bit = t` absolute piece remains its open gate.

---

## 15. References

### Framework (upstream)
- `theorem_observer_energy_functional.md` — the OEF (Type 4; `E = κ·L`, extensivity, κ).
- internal research notes — the currency ontology.
- internal research notes — Stages 0–3b + the F1 pre-registration (§12) and outcome.
- `../../proofs/foundations/H_multiway_construction.py` — the B_VD=0 no-go.

### Cited published theorems (via the OEF)
- **Landauer, R.** (1961). *Irreversibility and heat generation in the computing process.* IBM J. Res. Dev. 5, 183–191.
- **Bennett, C. H.** (1973). *Logical reversibility of computation.* IBM J. Res. Dev. 17, 525–532.
- **Shannon, C. E.** (1948). *A Mathematical Theory of Communication.* (Via the OEF's Stage 2a.)

### Verification
- `../../proofs/foundations/BOUND_F1_oef_two_subsystem_2026-07-04.py` — ALL PASS; identities exact on 8,100 pairs + 277,020 triples + disjoint controls.

---

## 16. Status

**THEOREM-GRADE-STRUCTURAL-CONDITIONAL** (closed under the parameter-linter hard gate for the FORM and adoption-count; conditional on the OEF's standing `energy = κ·description-length` identification; absolute scale out of scope, walled). Every load-bearing step annotated and machine-verified. No fabricated citations; no post-hoc fitting; no magnitude claimed. The binding-energy VALUE stays OPEN.
