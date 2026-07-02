# Parameter ledger DAG — chain structure of framework predictions

**Purpose:** Make the dependency-DAG of the framework's parameter predictions explicit. Each parameter ledger row chains from a class master theorem and/or structural-ledger rows to its predicted value. This document maps the chains; the parameter ledger remains the per-row source of truth.

**Written:** 2026-04-28.

> **Authority discipline (refreshed 2026-05-26 consolidation pass).** This is a *structure* map. Chain topology + dependency identification are authoritative here; specific numeric values should be verified against the auto-generated `predicted_parameters.md` at the repo root (the DAG-runner output), which is the canonical comparison table. The Level-3 / Coverage / DAG-leaves sections were refreshed 2026-05-26 to reflect M_persistence (m_t, light quarks, tan β shipped 2026-05-26), the L6 wall closure (n_s, r, σ_8, r_s, θ_* shipped as BLOCKED 2026-05-15), and the foundation/observable split for Λ_CC.

## DAG depth structure

Three levels:

**Leaves (Level 0):** Structural ledger rows + class master theorems.
- Structural rows 1–25 (uniqueness ledger). Row 25 is the substrate-scale = Planck-scale anchor identification (added 2026-04-28 PM).
- Class master theorems: A (spectral), B (dispersion), D (statistical), E (combinatorial). Class C is taxonomic.

**Level 1: Direct master-theorem members.** Parameter rows whose value is *directly* given by a class theorem formula (no further chaining).

**Level 2+: Chained parameters.** Parameter rows whose values depend on Level 1 + structural rows + external inputs.

**Open:** Rows still needing closure (gaps, scoping, NOT STARTED).

## Level 1 — Direct master-theorem members

### Class A (spectral, algebraically unified)
| row | claim | leaf chain |
|---|---|---|
| P1 | α_1_bare = (2/3)^8 | Class A: q_NB^(g−2); Row 4 (k*=3), Row 9 (g=10), Row 23 (q_NB) |
| P2 | α_1_full = 256/6305 | Class A: q_NB^(g−2)/(1−q_NB^(g−2)); Row 11 (A2-T), Row 23 |
| P3 | V_cb = 256/6305 | = α_1_full (Class A); Rows 4, 7, 9, 23 |
| P5 | c = 5/12 (dark) | Class A spectral OR Class E cycle (over-determined); Rows 7, 16, 23, 24 |

### Class B (dispersion at Dirac points)
| row | claim | leaf chain |
|---|---|---|
| (no parameter ledger row directly) v_F = 1/2 (Γ) | Class B: k·p at Γ Dirac; Row 16, Row 24 sector split |
| (no parameter ledger row directly) v_F = √3/6 (P) | Class B: k·p at P Dirac; Row 16, Row 24 |
| (D_H, D4_iso^H, etc.) | Class B Taylor coefficients; theorem-grade SYMBOLIC |
| G_sub | Class B; **STRUCTURALLY OPEN** — earlier 1/(8π³) candidate retracted 2026-04-28 PM; correct closure via dynamic matter 1-loop polarization (multi-session) |

### Class C (group-theoretic, taxonomic — no master theorem)
| row | claim | leaf chain |
|---|---|---|
| P6 | sin²θ_W = 3/8 | Pati-Salam trace identity; Row 17 (PS), Row 18 (3 gen) |
| P14 | V_ub = 128/32805 | Theorem-grade via bridge functoriality lemma; Rows 16, 17, 23 |
| P15 | δ_CP_CKM = arccos(1/3) | Tetrahedral dihedral on K_4; Row 16 + bridge |
| P16 | θ_QCD = 0 | Z₃ holonomy flatness on srs; Rows 4, 6 |
| (Row 18 structural) | n_generations = 3 | Cyclic Z₃ orbit on observer; mathematically complete |
| P21 | α_GUT = 1/24 | (Class E preferred); 1/(2^k*·k*); Rows 4, 16 + A2 + Jaynes |

### Class D (statistical)
| row | claim | leaf chain |
|---|---|---|
| P22 | Ω_DM/Ω_m ≈ 0.8488 | Class D: 1−P(k≤k*\|Poisson(2k*)); Rows 4, 11, 16, 23 |
| P27 | A_hemispherical = 1/15 | Class D × Class E composite: ε_CP·(1/k*); Row P28 + cubic moment |
| P28 | ε_CP = 1/5 | Class D Bayesian (primary); Rows 4, 16 + Beta(2,1) update |

### Class E (combinatorial)
| row | claim | leaf chain |
|---|---|---|
| P4 | V_us = 9/40 | Class E: k*²/(g·\|V\|) = Moore-bound saturation; Rows 4, 7, 9, 16 + ADOPTED-A5b-Sub3 (now theorem-grade) |
| P5 | c = 5/12 | Class E cycle OR Class A spectral (over-determined) |
| P21 | α_GUT = 1/24 | Class E label count: 1/(2^k*·k*); Rows 4, 16 + A2 + Jaynes |

## Level 2 — Chained parameters

These rows depend on Level 1 outputs + additional structural / external inputs.

### Lepton mass tree (chain through Class A + Class C)
| row | claim | chain |
|---|---|---|
| P7 | y_τ = 1280/177147 | α_1_full / k*²; Class A + Row 4 |
| P8 | Q_Koide = 2/3 | Class C: Pati-Salam + C₃ observer + Foot 1994; Rows 17, 18 |
| P9 | ε_Koide² = 2 + δ_Koide = 2/9 | Class C: Pati-Salam + Born rule (A3-T) |
| P10 | v_Higgs = 246.22 GeV | UNIQUE-THEOREM-GRADE post G1b R2 closure (2026-04-28 PM); chain through N_hub + α_1_full |
| P11 | m_τ + m_e + m_μ | UNIQUE-THEOREM-GRADE post G1b R2 closure; chain through P10 + P7 (Yukawa × Koide); Rows 17, 18 |

### Higgs sector (chain through Class A)
| row | claim | chain |
|---|---|---|
| P12 | m_H = 125.20 GeV (Family-D corrected) | THEOREM-GRADE-STRUCTURAL conditional on c_H Route H/C closure (W1 2026-05-18 reinstatement; prior "THEOREM-GRADE-NUMERICAL via Family D" was a Clause-6c smuggle, numeric unchanged); chain through λ × v_Higgs; −0.05σ_PDG. |
| (λ_Higgs = 2·α_1_full) | Class A: 2·256/6305 ≈ 0.130; Rows 11, 17, 23 |

### CKM chain
| row | claim | chain |
|---|---|---|
| P13 | (CKM details) | chain through P3 (V_cb), P4 (V_us), P14 (V_ub), P15 (δ_CP) |
| P32-P34 | PMNS angles | chain through Class C bridge + dark-map; theorem-grade per Row P14 closure |
| P45 | J_CKM | inherits P14 |

### Cosmology chain
| row | claim | chain |
|---|---|---|
| P19 | H_0 = 68.18 km/s/Mpc | UNIQUE-THEOREM-GRADE post G1b R2 closure; chain through N_hub (G_F anchor) + cascade theorem; Row 4 + Row 16 |
| P20 | t_0 = 14.34 Gyr (substrate); 13.45 Gyr (observer = 15/16·substrate) | UNIQUE-THEOREM-GRADE post G1b R2 closure; 1/H_0 (coasting condition); inherits P19. (Was single "14.38" — superseded by the substrate/observer dual.) |
| P23 | Ω_DM ≈ 0.277 | algebraic from Ω_DM/Ω_m + external Ω_b (P22 + Ω_b) |
| P24 | Λ_substrate = 1/N² (clean foundation, THEOREM-GRADE, `predictions/Lambda_CC.py`) + sibling Λ_CC_LCDM (observable-side Type-4, `predictions/Lambda_CC_LCDM.py`) | Foundation/observable split 2026-05-16 (Row P24 + P24-sibling); the factor vs ΛCDM-fit is a cosmology-model split, OPEN. (Was "3/N²" — superseded by the split.) |

### Other
| row | claim | chain |
|---|---|---|
| P17 | N_hub | UNIQUE-THEOREM-GRADE post G1b R2 closure; G_F retained as numerical anchor; chain through Class A α_1_full + dark correction |
| P18 | Y = +1/2 | CONDITIONAL on ADOPTED-B3; structural |

## Level 3 (Open)

Rows still needing closure beyond what master-theorem framing covers (refreshed 2026-05-26):
| row | status | gap |
|---|---|---|
| P25 | n_s | L6 wall — slow-roll claim retired 2026-05-15; photon-walker chirality primitive is C₃-high-symmetry-k-point-specific, absent in the acoustic regime. Sprints A + B closed-negative. |
| P26 | r | same L6 wall as P25 |
| P25-sibling | σ_8 | same L6 wall |
| (recombination cluster) | r_s, θ_*, t₀^ΛCDM-frame | same L6 wall; no `predictions/` file per 2026-05-17 directive (no file without closure theorem) |
| A_s | DOMINANT-THEOREM-GRADE-CONDITIONAL (Item 3 Session 2 closure 2026-05-05) | not in SECTORS manifest per 2026-05-15 EOD+5 theorem-grade-only policy; predictions/A_s.py not standalone |
| 8 SUSY rows | NOT STARTED | external/separable (tan β shipped 2026-05-26 is the exception) |

The earlier "η_B / Λ_CC / m_t are open" framings in this doc are SUPERSEDED — η_B is UNIQUE-THEOREM-GRADE-CONDITIONAL (Row P29, 2026-04-30), Λ_substrate is UNIQUE-THEOREM-GRADE (Row P24, graduated 2026-05-16), m_t ships at THEOREM-GRADE-STRUCTURAL-CONDITIONAL via M_persistence (Row P38, 2026-05-26).

## Coverage summary

The 5-class master-theorem DAG covers the majority of the framework's parameter predictions; the remaining closures use per-row derivations that live in the structural and parameter ledgers. As of 2026-05-26, of 123 tracked targets:

| Status | Count | Mechanism |
|---|---|---|
| ✅ CLOSED | 98 | master-theorem DAG (Level 1+2) + per-row theorem-grade derivations |
| 🟡 IN PROGRESS | 3 | live `predictions/` files with named structural gaps (M_Z / m_W intrinsic σ_PDG-precision-floor; g_3 / α_s / R∞ out-of-scope-by-construction per Move-1) |
| ❌ NOT STARTED / RETIRED | 12 | mostly L6-blocked recombination cluster + SUSY-spectrum rows + t₀^ΛCDM-frame |
| ⚙️ STRUCTURAL | 10 | definitional identifications (gauge group, generations, spatial dim, Lorentz signature, parity violation, charge quantization, etc.) |

Authoritative per-parameter status: [`target_parameters.md`](target_parameters.md). Numerical authority: auto-generated `predicted_parameters.md` at repo root.

## DAG leaves (ultimate dependencies)

Tracing all chains back to their ultimate roots, the **leaves of the DAG** are:

### Structural ledger rows (25 rows, 2026-04-28 PM)
The framework's irreducible structural facts: k* = 3 (Row 4), srs lattice (Row 6), |E| = 6 (Row 7), |V| = 4 (Row 16), Pati-Salam embedding (Row 17), 3 generations (Row 18), q_NB = 2/3 (Row 23), Hashimoto sector decomposition (Row 24), etc.

### Class master theorems (4 documented)
- Class A spectral: `theorem_unified_spectral_dark.md` (4 algebraically unified members + 2 k=3 coincidences)
- Class B dispersion: `theorem_class_B_dispersion.md` (k·p framework, ~9 theorem-grade + G_sub pending)
- Class D statistical: `theorem_class_D_statistical.md` (3 members)
- Class E combinatorial: `theorem_class_E_combinatorial.md` (3 members)

### External anchors (acknowledged inputs)
- G_F (Fermi constant) for cosmology chain (P19 H_0 etc.)
- Ω_b (baryon density) for P23 Ω_DM
- M_P, t_P (Planck units) for cosmological time scaling

These are external because the framework hasn't derived them from first principles. They're flagged in the parameter ledger explicitly.

### Adoptions (deliberate structural inputs)
- ADOPTED-B3 (Pati-Salam labeling): affects Y hypercharge and SU(2)_L chirality.
- ~~ADOPTED-PS-SCALE~~ — CLOSED 2026-05-04 via global spectral-gap formula m_ν₃ = (k*·N_atoms)·M_Pl·N_hub^(-1/2); see `audits/registers/adoption_register.md`.
- ADOPTED-DARK-MAP (Class 2): affects m_H, θ_23.
- ADOPTED-A5b-Sub3 → graduated to theorem-grade 2026-04-28 via bridge functoriality lemma.

## Implications for the framework's predictive structure

1. **The DAG is real and explicit.** All ~46 master-theorem-DAG-covered parameters trace through documented chains to a small leaf set: 24 structural rows + 4 class theorems + a handful of external anchors and adoptions.

2. **Coverage is partial but systematic.** ~50% of target parameters are under master-theorem DAG; another ~36% are CLOSED via per-row derivations; ~14% open.

3. **The framework's prediction structure is auditable end-to-end.** Every closed prediction has a documented chain (in `*Conditional on*` lines + theorem references) that bottoms out at structural ledger rows or master theorems. No prediction is "unmoored" from the framework's structural commitments.

4. **The leaf set is small.** 24 structural rows + 4 class theorems = 28 fundamental commitments. From these (plus external anchors and adoptions), all predicted parameters follow.

## Recommended companion for the parameter linter

The parameter linter (`parameter_linter.md`) currently gates each prediction by Type 1/2/3/4. With the master-theorem DAG, an additional gate is natural:

> **Type 5 — chained from class theorem:** prediction equals a documented Level 1 master-theorem member, OR a Level 2 chain from such a member with explicit dependency citations to structural rows + class theorems.

This makes "I'm chained from Class A theorem + Row 23" a first-class gate type, complementing Type 1 (axiom), Type 2 (CAS), Type 3 (cited theorem), Type 4 (upstream closed file). Most ledger rows that are "chained from a Class theorem" can be re-classified under Type 5 once the chain is explicit.

The next step (after this DAG mapping doc): update master_plan + parameter_linter to formalize Type 5 and require chain documentation in every new ledger row.

## Reference

This document is a *navigation aid* derived from:
- `../audits/registers/uniqueness_ledger.md` — structural rows.
- `parameter_uniqueness_ledger.md` — parameter rows + their `*Conditional on*` lines.
- `../theorems/theorem_unified_spectral_dark.md` + `theorem_class_A_audit.md` — Class A.
- `../theorems/theorem_class_B_dispersion.md` — Class B.
- `../theorems/theorem_class_C_scoping.md` — Class C taxonomic.
- `../theorems/theorem_class_D_statistical.md` — Class D.
- `../theorems/theorem_class_E_combinatorial.md` — Class E.
- `docs/master_plan.md` §3.1 — 5-class taxonomy.
