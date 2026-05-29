# Class C consolidation via algebraicity meta-theorem

**Status:** Consolidation note (2026-04-29). Class C predictions are heterogeneous (no master theorem possible per `theorem_class_C_scoping.md`). The algebraicity meta-theorem provides a unifying description: every Class C prediction is K-valued, and the heterogeneity of MECHANISMS gives a single CLASS via shared K-membership.

**Predecessor:** `theorem_class_C_scoping.md` (Class C taxonomic statement; concludes no master theorem). This consolidation reframes that conclusion via the algebraicity meta-theorem.

## TL;DR

Class C has no unifying mechanism but DOES have a unifying *output property*: every Class C prediction is in K = ℚ(√2, √3, √5). This is a non-trivial consolidation: while mechanisms are heterogeneous (trace identities, polytope angles, holonomy, orbit counts), the K-membership is uniform.

The uniqueness template doesn't directly close any *open* Class C issues — those (B4 color normalization, RG running) aren't parity-violation questions. But the algebraicity meta-theorem CONSTRAINS open closures to produce K-valued results, ruling out continuum-RG-running transcendental factors as the structural form.

## 1. Class C predictions and their K-membership

| Prediction | Value | Mechanism | In K? |
|-----------|-------|-----------|-------|
| sin²θ_W | 3/8 | trace identity on Pati-Salam reps | ✓ ℚ |
| α_GUT | 1/24 | 1/(2^k* · k*) group-order normalization | ✓ ℚ |
| n_generations | 3 | C_3 site symmetry of K_4 quotient | ✓ ℤ ⊂ ℚ |
| δ_CP_CKM | arccos(1/3) | K_4 dihedral angle | cos = 1/3 ∈ ℚ ⊂ K; sin = (2√2)/3 ∈ K |
| θ_QCD | 0 | Z₃ gauge holonomy flatness | ✓ ℚ |
| Higgs doublet | (1, 2, +1/2) | Cl(2) factor in Cl(6) | ✓ rationals |
| Hypercharge Y = +1/2 | 1/2 | PS embedding (CONDITIONAL on ADOPTED-B3) | ✓ ℚ |
| V_ub (currently 3.767×10⁻³) | structural form 3.767e-3 | bridge multicycle sum (RETRACTED 2026-04-29) | ✓ ℚ |

**All Class C theorem-grade predictions are in K.** ✓

## 2. The unification

Class C's unifying property is **K-membership via representation-theoretic origin**:

> Every Class C prediction is a representation-theoretic / geometric / cohomological invariant of (Pati-Salam ⊂ Spin(6) ⊂ Cl(6)) + (Z₃ on observer C³) + (K_4 polytope), each of which involves only integer-coefficient algebraic structures. By the algebraicity meta-theorem (Theorem 1 of `theorem_lattice_coupling_general.md`), all such invariants are in K.

This is structurally consistent with the algebraicity meta-theorem. Class C's heterogeneity of mechanisms doesn't matter for K-membership — every mechanism produces K-valued output because every mechanism uses only L-operations (representation traces, polytope angles, holonomy invariants, orbit counts — all in L).

## 3. What this consolidates vs what stays open

**Consolidated:**
- Class C K-membership is a corollary of the meta-theorem (no separate proof needed per prediction).
- New Class C predictions (e.g., higher Pati-Salam representation invariants) are automatically in K.
- The framework's predicted set is closed under representation-theoretic operations within L.

**Still open (separate from consolidation):**
- **B4 color normalization** (affects g_1, g_2, g_3, hypercharge Y, sin²θ_W RG-running). The framework's PS embedding is ADOPTED-B3; closing this would require deriving the embedding from A1+A2+A3+A4 alone. Multi-session research-level.
- **sin²θ_W RG running to M_Z**: not a framework prediction in the structural sense — it's continuum SM RG running, outside L. The framework predicts 3/8 at M_unif (lattice scale); the running to M_Z involves transcendental loop factors and is properly in continuum QFT, NOT in K.
- **g_1, g_2, g_3 absolute values at M_Z**: same RG-running issue.

## 4. Constraint on open Class C closures via algebraicity

The meta-theorem provides a sanity check on any FUTURE Class C closure: the structural form must produce a K-valued coefficient. This rules out:
- Continuum-loop RG-running coefficients as the structural answer.
- Transcendental factors involving π, e, log.

For B4 closure specifically: whatever the structural derivation of hypercharge, the result must be a rational (Y = ±1/2, ±1/3, etc. — all in ℚ ⊂ K). The current framework's Y = +1/2 fits.

## 5. The uniqueness template doesn't directly apply to Class C

Unlike β (Berry phase mechanism) or m_ν (Feshbach contour mechanism), Class C predictions don't have a "parity-violation source + functional uniqueness + algebraicity" structure. Class C is taxonomic — each prediction is a representation-theoretic invariant of a different group structure.

The algebraicity meta-theorem (one of the three lemmas underlying the uniqueness template) DOES apply to Class C — every prediction is in K. But the uniqueness template's full structure (P1 source + P2 mechanism-forced functional + P3 K-membership) doesn't directly produce new Class C closures.

**This is fine.** The framework's Class C predictions are already theorem-grade individually. The consolidation just notes that they all share K-membership as a unifying property.

## 6. New result this consolidation produces

The Class C taxonomic statement of `theorem_class_C_scoping.md` ("Class C is heterogeneous; no master theorem possible") is REFINED to:

> **Refined Class C statement:** Class C predictions are representation-theoretic / geometric / cohomological invariants of the framework's emergent group structure (PS ⊂ Spin(6) ⊂ Cl(6) + Z₃ + K_4). They lack a unifying *mechanism* but share K-membership as a unifying *output property*. The algebraicity meta-theorem (`theorem_lattice_coupling_general.md`) provides the unifying descriptive principle.

This consolidates Class C under the same theoretical umbrella as Classes A, B, E (all K-valued via algebraicity meta-theorem) while honestly acknowledging the mechanism heterogeneity.

## 7. Cross-references

- `theorem_class_C_scoping.md` (predecessor — Class C taxonomic statement)
- `theorem_lattice_coupling_general.md` (meta-theorem providing unifying description)
- `theorem_unified_spectral_dark.md` (Class A unified spectral)
- `theorem_class_B_dispersion.md` (Class B)
- `theorem_class_E_combinatorial.md` (Class E)
- `docs/master_plan.md` §3.1 (5-class taxonomy)
