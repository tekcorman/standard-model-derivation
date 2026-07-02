# Class C scoping report: group-theoretic / geometric parameters

**Status:** Scoping conclusion — Class C does NOT have a tight unifying master theorem in the same form as Class A (spectral) or Class D (statistical). The members are heterogeneous derivations sharing upstream structural dependencies (Rows 16-20) but not a single mathematical mechanism.

**Written:** 2026-04-28.

## Class C candidates

Surveying framework parameters whose primary derivation is group-theoretic or geometric (not spectral A, dispersion B, statistical D, or combinatorial E):

| coefficient | derivation | mathematical object |
|---|---|---|
| sin²θ_W = 3/8 | Pati-Salam trace identity | Σ Tr(T₃²) / Σ Tr(Q²) on (4,2,1)+(4̄,1,2) reps |
| δ_CP_CKM = arccos(1/3) | Regular-tetrahedron dihedral | Polytope geometry on K_4 (−1)-eigenspace |
| θ_QCD = 0 | Z₃ gauge holonomy flatness | Cycle-holonomy on srs |
| n_generations = 3 | Cyclic Z₃ orbit on observer | Group action dim |
| V_ub = 128/32805 | Z₃-eigenvalue + walk-rep | THEOREM-GRADE under bridge functoriality lemma (graduated 2026-04-28) |
| Higgs doublet | Cl(2) + Pati-Salam descent | Sub-rep selection |
| Hypercharge Y = +1/2 | (CONDITIONAL on ADOPTED-B3) | Pati-Salam → SM matching |

These all use group/representation theory or polytope geometry, but the **specific mathematical objects differ substantially**:

1. **Trace identities** on Pati-Salam reps → sin²θ_W
2. **Polytope angles** on K_4 (−1)-eigenspace → δ_CP_CKM
3. **Holonomy invariants** on Z₃ gauge bundle → θ_QCD
4. **Orbit dimensions** of cyclic group actions → n_generations
5. **Sub-rep selection** under embedding chains → Higgs doublet, Y

No single mathematical operation unifies these — they're a heterogeneous collection.

## Why no master theorem

The Class A (spectral) and Class D (statistical) master theorems both succeed because their members share a SPECIFIC mechanism:
- Class A: all members are functionals of (λ_max(A), λ_max(B)) + structural integers on the same Hashimoto operator.
- Class D: all members are moments of max-entropy / Bayesian-posterior distributions on toggle structure.

Class C members don't share such a specific mechanism. They share only:
- **Upstream structural rows** (Row 16: Cl(6) per node; Row 17: Pati-Salam; Row 18: 3 generations; Row 19: SM gauge group; Row 20: Higgs doublet).
- **Reliance on group/geometric structures** that emerge from Rows 16-20.

This is too broad to constitute a master theorem. "Use representation theory" doesn't predict any specific value — you have to do the case-by-case calculation for each parameter.

## What CAN be said about Class C

While no tight master theorem exists, several structural observations are useful:

### 1. Class C is NOT free of structural input

All Class C derivations *use* structural inputs from Rows 16-20:
- sin²θ_W requires Pati-Salam embedding (Row 17) + matter content (Row 18).
- δ_CP_CKM requires K_4 (−1)-eigenspace structure (Row 16).
- θ_QCD requires Z₃ gauge bundle on srs (Row 16 + flatness theorem).
- n_generations requires cyclic-Z₃ observer (Row 18).

The structural rows are the framework's "physics input"; Class C parameters are *consequences* of those rows under standard mathematical operations (trace, holonomy, orbit dim, etc.).

### 2. Class C has no free parameters

Despite the heterogeneity, every Class C value is determined by structural rows — there's no fitted constant. This is the same property as Classes A, D, E: the framework's parameters are forced by structural inputs alone.

### 3. Class C is primarily ONE-OFF

Each Class C parameter has its own theorem doc:
- sin²θ_W: `theorem_sin2_theta_W_unification.md`.
- δ_CP_CKM: `predictions/delta_CP_CKM_geometry_derivation.md` + `theorem_bridge_functoriality_lemma.md`.
- θ_QCD: `predictions/theta_QCD_derivation.md` (Z₃ holonomy + Ambrose-Singer).
- n_generations: structural ledger Row 18 (mathematically complete).
- Higgs doublet: structural ledger Row 20.
- V_ub: `predictions/V_ub_derivation.md` (THEOREM-GRADE under `theorem_bridge_functoriality_lemma.md` since 2026-04-28).

Each is rigorous on its own terms. The lack of a unifying theorem doesn't weaken individual closures.

### 4. The closest thing to a master theorem

If we squint, Class C members ARE all "representation-theoretic invariants of the framework's emergent symmetry structure." But this is more a *taxonomy* than a *theorem*:

> **Class C taxonomic statement:** Framework parameters that arise from representation-theoretic / geometric invariants of (Pati-Salam ⊂ Spin(6) ⊂ Cl(6)) embedding (Row 17) + Z₃-action on observer (Row 18) + K_4 polytope geometry (Row 16). Each parameter requires its own case-by-case derivation; no single closed-form formula covers them.

This is the *master statement* I can write, but it's not a *master theorem* — it doesn't predict any specific value, only categorizes derivations.

## Comparison to Classes A and D

| feature | Class A | Class D | Class C |
|---|---|---|---|
| Underlying object | Hashimoto B(Γ) + adjacency A(Γ) | Toggle distributions | Pati-Salam embedding + Z₃ action + K_4 |
| Master mechanism | Spectral functionals | Max-entropy / Bayesian | (none — heterogeneous) |
| Number of members | 6 | 3 | 5+ |
| Master theorem? | ✓ | ✓ | ✗ |
| Cross-class redundancy | shared with E (5/12, 9/40) | shared with A (ε_CP) | minimal |
| Structural rigor | high (over-determined) | high (over-determined) | per-derivation; rigorous |

## Implications

1. **Class C is structurally well-defined but mechanically heterogeneous.** All members share upstream structural dependencies (Rows 16-20) but use different mathematical operations. This is honest: not every framework class admits a single master theorem.

2. **Class C parameters are no less rigorous** than Class A/D parameters; they just don't share a unifying mechanism. Each has its own theorem doc and is independently verified.

3. **Including Class C as a bin in the 5-class taxonomy is justified** even without a master theorem — the bin captures "everything that's group-theoretic / geometric and emerges from the framework's symmetry structure". Master_plan §3.1's classification is correct.

4. **The framework's master-theorem coverage is roughly:**
   - Class A: 6 parameters (spectral master theorem).
   - Class B: many LV coefficients (k·p master theorem; numerical G_sub blocked).
   - Class C: 5+ parameters (no master; case-by-case).
   - Class D: 3 parameters (Bayesian/max-entropy master theorem).
   - Class E: 3 parameters (combinatorial master theorem).

   So roughly **18 framework parameters** have either a master theorem or a class-level closure structure. The remaining parameters in the 59-row ledger are mostly chained derivations (e.g., m_τ via Yukawa × Koide) or scoping gaps.

## Closure status of this report

- Scoping complete.
- Class C is documented as having no master theorem; each member's individual derivation continues to be the load-bearing argument for that parameter.
- Master_plan §3.1's taxonomy stands; the "Class C" bin is taxonomic, not a master-theorem unifier.

## Recommendation for next steps

Since Class C doesn't admit a master theorem, the natural next step is **Class A audit** — confirming the unified spectral dark theorem's six members are theorem-grade rigorous and either firming up or retracting the Ihara cross-validation claim. Class A is the framework's largest master-theorem class; tightening it has highest leverage.

After that:
- **Class B** numerical closure (G_sub).
- Class C parameter-by-parameter, as before (V_ub closures, hypercharge, etc.).
- Tier 2-5 carry-over work from an internal working note.

## References

- `docs/master_plan.md` §3.1 — 5-class taxonomy.
- `theorem_class_D_statistical.md` — Class D master theorem (for comparison).
- `theorem_class_E_combinatorial.md` — Class E master theorem (for comparison).
- `theorem_unified_spectral_dark.md` — Class A master theorem (largest class).
- `theorem_class_B_dispersion.md` — Class B framing.
- Individual Class C parameter derivations cited in the Class C candidates table above.
