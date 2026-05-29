# W24 Verdict — BS-T × J=±1 sector counts match empirical (1/4, 1/3, 5/12)

**Status:** STRUCTURAL POSITIVE FINDING with one remaining structural gap (a milder version of W21's non-canonical-pick obstruction).

---

## What W24 establishes (positive structural finding)

On K_4 (= srs primitive cell at Γ), the joint (B-eigenvalue u, J-eigenvalue) decomposition of the directed-edge space ℂ^{12} gives:

| sector | dim | Wilson-loop rank | content |
|---|---|---|---|
| (u=+1, J=+1) | 2 | 0 | V_scalar's u=+1 part |
| (u=+1, J=-1) | 1 | 1 | Part of V_cycle |
| (u=-1, J=+1) | 0 | 0 | — |
| (u=-1, J=-1) | 2 | 2 | Rest of V_cycle |
| **V_pm total** | **5** | — | — |
| **V_cycle (J=-1)** | **3** | **3 = β_1** | H¹ lifts, Wilson-loop carriers |
| **V_scalar (J=+1)** | **2** | **0** | gauge-singlet content |

**Key clean facts:**
1. V_cycle (J=-1, dim 3) has Wilson-loop rank EXACTLY β_1 = 3 → it IS the full H¹(K_4; ℝ) lift to Hashimoto u=±1.
2. V_scalar (J=+1, dim 2) has Wilson-loop rank 0 → it is OUTSIDE C¹(K_4; ℝ) = B¹ ⊕ H¹ entirely.
3. The (u, J) joint eigenvalue labeling is CANONICAL (B and J are both global K_4 invariants).

**Proposed sector-specific c values:**

| sector | formula | value | empirical | Δ |
|---|---|---|---|---|
| c_color (SU(3)_c) | V_cycle / (2|E|) | 3/12 = 1/4 | 0.2414 | +0.009 |
| c_EW (U(1)_Y, SU(2)_L) | (V_cycle + 1)/(2|E|) | 4/12 = 1/3 | 0.343, 0.332 | ±0.01 |
| c_v_Higgs (scalar 2-point) | V_pm / (2|E|) | 5/12 | (anchor) | — |

All three match empirical / anchor values within 0.01.

---

## The "+1 in V_scalar" is the remaining structural gap (milder W21)

The c_EW = 4/12 = 1/3 formula adds "1 mode" from V_scalar (out of its 2 J=+1 modes). This "+1" is the structural element that needs canonical justification.

**Three candidate justifications, ordered by structural cleanliness:**

### Candidate A — Pati-Salam SU(2)_L doublet ↔ V_scalar projection

V_scalar is C_3-irreducible (faithful 2-pair, per yesterday's probe). SU(2)_L acts on Pati-Salam doublets {nu_L, e_L} and {u_L, d_L} as the standard SU(2) fundamental. If the framework's SU(2)_L embedding into the substrate selects a CANONICAL 1-dim sub-direction in V_scalar (e.g., the T_3^L eigenvector), then c_EW = 4/12 is structurally derived.

This requires showing: under the framework's substrate-to-Pati-Salam embedding, V_scalar's 2 modes correspond to the SU(2)_L T_3 = ±1/2 components of a doublet, and exactly ONE of them is the "EW-active" direction relevant for the dark Q-projector.

**Effort estimate:** 2-3 sessions. **Closure probability:** ~30%.

### Candidate B — Higgs vacuum direction picks out the +1

The v_Higgs vacuum ⟨φ⟩ ≠ 0 breaks SU(2)_L × U(1)_Y → U(1)_EM. This vacuum direction sits in a specific 1-dim subspace of the Higgs doublet (post-EWSB, the gauge-fixed direction). If this "Higgs vacuum direction" inside the substrate corresponds to a 1-dim sub-direction of V_scalar, then c_EW = 4/12 follows.

This is mechanistically similar to Candidate A but tied to v_Higgs's existing theorem-grade derivation rather than SU(2)_L kinematic structure.

**Effort estimate:** 1-2 sessions. **Closure probability:** ~25%.

### Candidate C — J=+1 Hamming-weight (n=0 or n=3) sub-direction

Per `theorem_charge_before_color.md` §9, the Cl(6) Fock per vertex decomposes as 1 ⊕ 3 ⊕ 3̄ ⊕ 1 under SU(3) (Hamming weights n=0,1,2,3). The n=0 and n=3 are SU(3) singlets (lepton-like). The "1 J=+1 BS-T-extra mode" of V_scalar might correspond to one of these singlet weight strata.

This requires the directed-edge basis ↔ Fock-state mapping I attempted in W22 (and which failed to commute with B without lattice gauge link variables). May require resolving the W22 obstruction too.

**Effort estimate:** 4-5 sessions. **Closure probability:** ~15%.

---

## Why the W21 obstruction is MILDER here than in the original framing

The original Route H scoping (yesterday's doc) wanted to identify ONE OF V_scalar's 2 modes as the "Perron-adjacency mode" (uniform-Frobenius-like, gauge-singlet) and the OTHER as the "BS-T-bipartite extra" (zero-Wilson-loop but coupled to U(1)/SU(2)). W21 showed this Perron-adj vs bipartite-extra split is non-canonical at the graph level on K_4.

W24's framing requires LESS — only that some canonical 1-dim sub-direction in V_scalar can be picked out by framework structure. The 1-dim sub-direction doesn't have to correspond to "Perron-adjacency mode" specifically. Any framework-supplied direction works (Higgs vacuum, SU(2)_L T_3 eigenvector, Cl(6) singlet projection, etc.).

This is a weaker structural requirement and admits more candidate derivations.

---

## What needs to happen for theorem-grade closure

1. **Pick a candidate** (A, B, or C) and run a focused probe.
2. If positive: write up the structural derivation, update `theorem_alpha_GUT_dark_correction.md` to sector-specific form (c_color = 1/4, c_EW = 1/3) with the new "+1 mode" mechanism, and re-derive the predictions cluster.
3. If negative: try another candidate. If all three fail, fall through to Path 3 (document residual).

The +0.008 sub-leading offset on c_EM (R_∞ ppt-precision constraint) is NOT addressed by W24 either — that's a separate higher-order question.

---

## Files

- `proofs/foundations/W24_BST_J_algebraic_sector_count_2026-05-26.py` — this probe
- `proofs/foundations/W24_verdict_BST_J_sector_count_2026-05-26.md` — this verdict
- Companion: `W21_W22_sector_specific_c_obstruction_verdict_2026-05-26.md` (W21/W22/W23 obstructions)
- Existing framework: `theorem_h1_master_compression.md` (B¹ ⊕ H¹ ⊂ J=-1 sector), `theorem_alpha_GUT_dark_correction.md` (uniform c = 1/3, current)
